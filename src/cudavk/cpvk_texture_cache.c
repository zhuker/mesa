#include "cpvk_private.h"
#include "cp_debug.h"
#include "util/u_math.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CPVK_TEXTURE_CACHE_BUDGET (384ull * 1024ull * 1024ull)

struct cpvk_cache_format {
   CUarray_format format;
   unsigned channels;
   unsigned bytes;
   unsigned conversion;
   unsigned targets; /* bit0 2D, bit1 cube, bit2 3D */
};

static bool
cpvk_cache_format(VkFormat format, struct cpvk_cache_format *out)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNSIGNED_INT8, 1, 1, 0, 1 };
      return true;
   case VK_FORMAT_R8G8_UNORM:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNSIGNED_INT8, 2, 2, 0, 5 };
      return true;
   case VK_FORMAT_R8G8B8A8_UNORM:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNSIGNED_INT8, 4, 4, 0, 7 };
      return true;
   case VK_FORMAT_R16G16_UNORM:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNSIGNED_INT16, 2, 4, 0, 1 };
      return true;
   case VK_FORMAT_R16G16_SFLOAT:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_HALF, 2, 4, 0, 1 };
      return true;
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_HALF, 4, 8, 0, 1 };
      return true;
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNORM_INT_101010_2, 4, 4, 0, 1 };
      return true;
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_HALF, 4, 8, 1, 3 };
      return true;
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNSIGNED_INT8, 4, 4, 2, 1 };
      return true;
   case VK_FORMAT_BC3_UNORM_BLOCK:
      *out = (struct cpvk_cache_format){ CU_AD_FORMAT_UNSIGNED_INT8, 4, 4, 3, 1 };
      return true;
   default:
      return false;
   }
}

static void
cpvk_cache_device_loss(struct cpvk_device *dev)
{
   atomic_store_explicit(&dev->device_lost, true, memory_order_release);
}

void
cpvk_texture_cache_fatal(void *private_data)
{
   cpvk_cache_device_loss((struct cpvk_device *)private_data);
}

static bool
cpvk_cache_soft_optional_create_error(CUresult err)
{
   return err == CUDA_ERROR_OUT_OF_MEMORY || err == CUDA_ERROR_INVALID_VALUE ||
          err == CUDA_ERROR_NOT_SUPPORTED;
}

static bool
cpvk_cache_image_eligible(struct cpvk_image *image,
                          struct cpvk_cache_format *format)
{
   return image && image->cache_create_eligible && !image->cache_alias &&
          image->mem &&
          image->mem->kind == CPVK_MEM_DEVICE &&
          !image->mem->binding_ledger_failed &&
          image->vk.tiling == VK_IMAGE_TILING_OPTIMAL &&
          image->vk.samples == VK_SAMPLE_COUNT_1_BIT &&
          cpvk_cache_format(image->vk_format, format);
}

static unsigned
cpvk_cache_depth(const struct cpvk_image *image, unsigned level)
{
   if (image->vk.image_type == VK_IMAGE_TYPE_3D)
      return MAX2(image->vk.extent.depth >> level, 1u);
   return MAX2(image->vk.array_layers, 1u);
}

static size_t
cpvk_cache_image_bytes(const struct cpvk_image *image,
                       const struct cpvk_cache_format *format)
{
   size_t bytes = 0;
   for (unsigned l = 0; l < image->vk.mip_levels; l++) {
      size_t w = MAX2(image->vk.extent.width >> l, 1u);
      size_t h = MAX2(image->vk.extent.height >> l, 1u);
      size_t d = cpvk_cache_depth(image, l);
      if (w > SIZE_MAX / h || w * h > SIZE_MAX / d ||
          w * h * d > SIZE_MAX / format->bytes)
         return SIZE_MAX;
      size_t level_bytes = w * h * d * format->bytes;
      if (bytes > SIZE_MAX - level_bytes)
         return SIZE_MAX;
      bytes += level_bytes;
   }
   return bytes;
}

static bool
cpvk_cache_cleanup_partial(struct cpvk_texture_cache *cache)
{
   bool ok = true;
   for (unsigned l = 0; l < CPVK_MAX_MIP_LEVELS; l++) {
      if (cache->surfaces[l]) {
         if (cuSurfObjectDestroy(cache->surfaces[l]) != CUDA_SUCCESS)
            ok = false;
         cache->surfaces[l] = 0;
      }
   }
   if (cache->ready) {
      if (cuEventDestroy(cache->ready) != CUDA_SUCCESS)
         ok = false;
      cache->ready = NULL;
   }
   if (cache->array) {
      if (cuMipmappedArrayDestroy(cache->array) != CUDA_SUCCESS)
         ok = false;
      cache->array = NULL;
   }
   return ok;
}

static bool
cpvk_cache_allocate(struct cpvk_image *image,
                    const struct cpvk_cache_format *format)
{
   struct cpvk_device *dev = image->dev;
   size_t bytes = cpvk_cache_image_bytes(image, format);
   if (bytes == SIZE_MAX || bytes > CPVK_TEXTURE_CACHE_BUDGET ||
       dev->texture_cache_bytes > CPVK_TEXTURE_CACHE_BUDGET - bytes)
      return false;

   CUDA_ARRAY3D_DESCRIPTOR desc = {
      .Width = image->vk.extent.width,
      .Height = image->vk.extent.height,
      .Format = format->format,
      .NumChannels = format->channels,
   };
   if (format->conversion)
      desc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
   if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
      desc.Depth = image->vk.extent.depth;
   } else if (image->vk.array_layers > 1) {
      desc.Depth = image->vk.array_layers;
      if (image->vk.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
         desc.Flags |= CUDA_ARRAY3D_CUBEMAP |
            (image->vk.array_layers > 6 ? CUDA_ARRAY3D_LAYERED : 0);
      else
         desc.Flags |= CUDA_ARRAY3D_LAYERED;
   }

   struct cpvk_texture_cache *cache = calloc(1, sizeof(*cache));
   if (!cache)
      return false;
   dev->texture_cache_stats.array_alloc_attempts++;
   CUresult err = cp_debug->texture_cache_fail_array_alloc_at ==
                     dev->texture_cache_stats.array_alloc_attempts
      ? CUDA_ERROR_OUT_OF_MEMORY
      : cuMipmappedArrayCreate(&cache->array, &desc, image->vk.mip_levels);
   if (err != CUDA_SUCCESS) {
      free(cache);
      dev->texture_cache_stats.alloc_failures++;
      if (!cpvk_cache_soft_optional_create_error(err))
         cpvk_cache_device_loss(dev);
      return false;
   }
   bool inject_event = cp_debug->texture_cache_fail_create_stage == 1;
   err = inject_event ? CUDA_ERROR_INVALID_VALUE :
      cuEventCreate(&cache->ready, CU_EVENT_DISABLE_TIMING);
   if (err != CUDA_SUCCESS) {
      bool cleanup_ok = cpvk_cache_cleanup_partial(cache);
      free(cache);
      dev->texture_cache_stats.alloc_failures++;
      /* OOM is disposable refusal. INVALID_VALUE has no admissible descriptor
       * cause here and EventCreate may report an earlier async failure. */
      if (inject_event && cp_debug->texture_cache_stats)
         fprintf(stderr, "cudavk: injected fatal cache-create stage=1 "
                 "before FS attempts=%" PRIu64 "\n",
                 dev->renderer.hardware_texture.fs_attempts);
      if (err != CUDA_ERROR_OUT_OF_MEMORY || !cleanup_ok)
         cpvk_cache_device_loss(dev);
      return false;
   }
   if (format->conversion) {
      for (unsigned l = 0; l < image->vk.mip_levels; l++) {
         CUarray level = NULL;
         CUDA_RESOURCE_DESC resource = { .resType = CU_RESOURCE_TYPE_ARRAY };
         err = cuMipmappedArrayGetLevel(&level, cache->array, l);
         if (err == CUDA_SUCCESS) {
            resource.res.array.hArray = level;
            bool inject_surface =
               cp_debug->texture_cache_fail_create_stage == 2;
            err = inject_surface ? CUDA_ERROR_INVALID_VALUE :
               cuSurfObjectCreate(&cache->surfaces[l], &resource);
            if (inject_surface && cp_debug->texture_cache_stats)
               fprintf(stderr, "cudavk: injected fatal cache-create stage=2 "
                       "before FS attempts=%" PRIu64 "\n",
                       dev->renderer.hardware_texture.fs_attempts);
         }
         if (err != CUDA_SUCCESS) {
            bool cleanup_ok = cpvk_cache_cleanup_partial(cache);
            free(cache);
            dev->texture_cache_stats.object_failures++;
            /* The level is loop-bounded and the array was created with
             * SURFACE_LDST. GetLevel/SurfObjectCreate refusal is therefore
             * either a driver/programming error or a deferred async error. */
            (void)cleanup_ok;
            cpvk_cache_device_loss(dev);
            return false;
         }
      }
   }
   cache->bytes = bytes;
   image->texture_cache = cache;
   image->texture_cache_next = dev->texture_cache_images;
   image->texture_cache_prev = &dev->texture_cache_images;
   if (image->texture_cache_next)
      image->texture_cache_next->texture_cache_prev =
         &image->texture_cache_next;
   dev->texture_cache_images = image;
   dev->texture_cache_bytes += bytes;
   dev->texture_cache_stats.peak_bytes =
      MAX2(dev->texture_cache_stats.peak_bytes, dev->texture_cache_bytes);
   dev->texture_cache_stats.arrays++;
   return true;
}

bool
cpvk_texture_cache_image_written(struct cpvk_image *image, CUstream stream)
{
   if (!image || !cp_debug->texture_cache)
      return true;
   struct cpvk_device *dev = image->dev;
   simple_mtx_lock(&dev->texture_cache_lock);
   struct cpvk_cache_format ignored;
   if (!cpvk_cache_image_eligible(image, &ignored)) {
      simple_mtx_unlock(&dev->texture_cache_lock);
      return true;
   }
   uint64_t old_epoch = atomic_load_explicit(&image->content_epoch,
                                              memory_order_relaxed);
   if (old_epoch == UINT64_MAX) {
      cpvk_cache_device_loss(dev);
      simple_mtx_unlock(&dev->texture_cache_lock);
      return false;
   }
   atomic_store_explicit(&image->content_epoch, old_epoch + 1,
                         memory_order_release);
   CUresult err = CUDA_SUCCESS;
   if (!image->writer_event) {
      err = cuEventCreate(&image->writer_event, CU_EVENT_DISABLE_TIMING);
      if (err != CUDA_SUCCESS)
         goto fail;
   }
   err = cuEventRecord(image->writer_event, stream);
   if (err != CUDA_SUCCESS)
      goto fail;
   image->writer_event_valid = true;
   simple_mtx_unlock(&dev->texture_cache_lock);
   return true;
fail:
   cpvk_cache_device_loss(dev);
   simple_mtx_unlock(&dev->texture_cache_lock);
   return false;
}

void
cpvk_texture_cache_written(void *private_data, uint64_t image_cookie,
                           CUstream stream)
{
   struct cpvk_device *dev = private_data;
   /* This callback consumes the trusted internal image token stored by a
    * framebuffer command, never an application descriptor cookie. Its
    * lifetime follows Vulkan command-resource lifetime rules. */
   struct cpvk_image *image = (struct cpvk_image *)(uintptr_t)image_cookie;
   if (!image) {
      cpvk_cache_device_loss(dev);
      return;
   }
   cpvk_texture_cache_image_written(image, stream);
}

static bool
cpvk_cache_rebuild(struct cpvk_image *image, CUstream stream,
                   const struct cpvk_cache_format *format, uint64_t epoch)
{
   struct cpvk_device *dev = image->dev;
   struct cpvk_texture_cache *cache = image->texture_cache;
   CUresult err;
   if (image->writer_event_valid) {
      err = cuStreamWaitEvent(stream, image->writer_event, 0);
      if (err != CUDA_SUCCESS)
         goto device_fail;
   }

   for (unsigned l = 0; l < image->vk.mip_levels; l++) {
      unsigned w = MAX2(image->vk.extent.width >> l, 1u);
      unsigned h = MAX2(image->vk.extent.height >> l, 1u);
      unsigned d = cpvk_cache_depth(image, l);
      if (format->conversion) {
         unsigned src_rows = format->conversion >= 2 ? DIV_ROUND_UP(h, 4) : h;
         struct cp_cache_convert_args args = {
            .src = image->mem->dev_ptr + image->offset + image->level_offset[l],
            .surface = cache->surfaces[l],
            .src_pitch = image->row_stride[l],
            .src_slice = image->vk.image_type == VK_IMAGE_TYPE_3D
               ? (uint64_t)image->row_stride[l] * src_rows
               : image->level_size[l],
            .width = w,
            .height = h,
            .depth = d,
            .target = image->vk.image_type == VK_IMAGE_TYPE_3D ? 2
               : ((image->vk.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
                  ? 3 : (image->vk.array_layers > 1 ? 1 : 0)),
            .format = format->conversion,
         };
         void *params[] = { &args };
         cp_ctx_check("cpvk_cache_rebuild", dev->cu_ctx);
         dev->texture_cache_stats.conversion_enqueue_attempts++;
         bool inject = cp_debug->texture_cache_fail_conversion_enqueue_at ==
            dev->texture_cache_stats.conversion_enqueue_attempts;
         err = inject ? CUDA_ERROR_INVALID_VALUE :
            cuLaunchKernel(dev->cp_dev.kernels.cache_convert,
                           DIV_ROUND_UP(w, 16), DIV_ROUND_UP(h, 16), d,
                           16, 16, 1, 0, stream, params, NULL);
         if (inject && cp_debug->texture_cache_stats)
            fprintf(stderr, "cudavk: injected fatal converted-cache enqueue "
                    "attempt=%" PRIu64 " before FS attempts=%" PRIu64
                    " prior_hw=%" PRIu64 "/%" PRIu64
                    " prior_direct=%" PRIu64 "/%" PRIu64 "\n",
                    dev->texture_cache_stats.conversion_enqueue_attempts,
                    dev->renderer.hardware_texture.fs_attempts,
                    dev->renderer.hardware_texture.hits,
                    dev->renderer.hardware_texture.fs_attempts,
                    dev->renderer.hardware_texture.path_hits[0],
                    dev->renderer.hardware_texture.fs_attempts);
      } else {
         CUarray level = NULL;
         err = cuMipmappedArrayGetLevel(&level, cache->array, l);
         if (err != CUDA_SUCCESS)
            goto device_fail;
         CUDA_MEMCPY3D copy = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .srcDevice = image->mem->dev_ptr + image->offset +
                         image->level_offset[l],
            .srcPitch = image->row_stride[l],
            .srcHeight = h,
            .dstMemoryType = CU_MEMORYTYPE_ARRAY,
            .dstArray = level,
            .WidthInBytes = (size_t)w * format->bytes,
            .Height = h,
            .Depth = d,
         };
         err = cuMemcpy3DAsync(&copy, stream);
      }
      if (err != CUDA_SUCCESS)
         goto device_fail;
   }

   err = cuEventRecord(cache->ready, stream);
   if (err != CUDA_SUCCESS)
      goto device_fail;
   cache->ready_valid = true;
   cache->scheduled_epoch = epoch;
   cache->ready_generation++;
   if (!cache->ready_generation)
      goto device_fail;
   dev->texture_cache_stats.rebuilds++;
   return true;

device_fail:
   cpvk_cache_device_loss(dev);
   return false;
}

static bool
cpvk_cache_address_mode(unsigned wrap, CUaddress_mode *out)
{
   switch (wrap) {
   case 0: *out = CU_TR_ADDRESS_MODE_WRAP; return true;
   case 2: *out = CU_TR_ADDRESS_MODE_CLAMP; return true;
   case 3: *out = CU_TR_ADDRESS_MODE_BORDER; return true;
   case 4: *out = CU_TR_ADDRESS_MODE_MIRROR; return true;
   default: return false;
   }
}

static bool
cpvk_cache_sampler_desc(struct cpvk_device *dev, unsigned sampler_index,
                        bool cube, CUDA_TEXTURE_DESC *desc)
{
   if (sampler_index >= dev->renderer.num_samplers)
      return false;
   const struct cp_sampler_info *s =
      &dev->renderer.sampler_table_host[sampler_index];
   if (s->unnormalized_coords || s->compare_enable ||
       s->reduction_mode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE ||
       s->non_seamless_cube || s->min_img_filter != s->mag_img_filter)
      return false;
   if (!cpvk_cache_address_mode(s->wrap_s, &desc->addressMode[0]) ||
       !cpvk_cache_address_mode(s->wrap_t, &desc->addressMode[1]) ||
       !cpvk_cache_address_mode(s->wrap_r, &desc->addressMode[2]))
      return false;
   desc->filterMode = s->min_img_filter == 1
      ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
   desc->mipmapFilterMode = s->min_mip_filter == 1
      ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
   desc->flags = CU_TRSF_NORMALIZED_COORDINATES |
                 (cube ? CU_TRSF_SEAMLESS_CUBEMAP : 0);
   float aniso = s->max_anisotropy > 1.0f ? s->max_anisotropy : 1.0f;
   if (aniso > 16.0f || aniso != floorf(aniso))
      return false;
   desc->maxAnisotropy = (unsigned)aniso;
   desc->mipmapLevelBias = s->lod_bias;
   desc->minMipmapLevelClamp = s->min_lod;
   desc->maxMipmapLevelClamp = s->max_lod;
   if (s->min_mip_filter == 2)
      desc->minMipmapLevelClamp = desc->maxMipmapLevelClamp = 0.0f;
   memcpy(desc->borderColor, s->border_color, sizeof(desc->borderColor));
   return true;
}

static bool
cpvk_cache_view_eligible(const struct cpvk_image_view *view, bool *cube)
{
   const struct cpvk_image *image = view->image;
   if (!view->cache_swizzle_identity || view->vk.view_format != image->vk_format ||
       view->vk.aspects != VK_IMAGE_ASPECT_COLOR_BIT ||
       !view->vk.level_count || !view->vk.layer_count ||
       view->vk.base_mip_level != 0 ||
       view->vk.level_count != image->vk.mip_levels ||
       view->vk.base_array_layer != 0 ||
       (image->vk.image_type != VK_IMAGE_TYPE_3D &&
        view->vk.layer_count != image->vk.array_layers))
      return false;
   *cube = view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE;
   if (*cube)
      return image->vk.image_type == VK_IMAGE_TYPE_2D &&
             (image->vk.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
             image->vk.array_layers == 6 && view->vk.layer_count == 6;
   if (view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D)
      return image->vk.image_type == VK_IMAGE_TYPE_3D &&
             view->vk.base_array_layer == 0 && view->vk.layer_count == 1;
   if (view->vk.view_type == VK_IMAGE_VIEW_TYPE_2D)
      return image->vk.image_type == VK_IMAGE_TYPE_2D &&
             image->vk.array_layers == 1 && view->vk.base_array_layer == 0 &&
             view->vk.layer_count == 1;
   return false;
}

static struct cpvk_texture_object *
cpvk_cache_object(struct cpvk_image_view *view, unsigned sampler_index,
                  const CUDA_TEXTURE_DESC *texture_desc)
{
   struct cpvk_image *image = view->image;
   struct cpvk_device *dev = image->dev;
   struct cpvk_texture_cache *cache = image->texture_cache;
   for (struct cpvk_texture_object *o = cache->objects; o; o = o->next)
      if (o->view == view && o->sampler_index == sampler_index)
         return o;

   CUDA_RESOURCE_DESC resource = {
      .resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
      .res.mipmap.hMipmappedArray = cache->array,
   };
   struct cpvk_texture_object *object = calloc(1, sizeof(*object));
   if (!object)
      return NULL;
   dev->texture_cache_stats.object_create_attempts++;
   CUresult err = cp_debug->texture_cache_fail_object_create_at ==
                     dev->texture_cache_stats.object_create_attempts
      ? CUDA_ERROR_OUT_OF_MEMORY
      : cuTexObjectCreate(&object->object, &resource, texture_desc, NULL);
   if (err != CUDA_SUCCESS) {
      free(object);
      dev->texture_cache_stats.object_failures++;
      if (!cpvk_cache_soft_optional_create_error(err))
         cpvk_cache_device_loss(dev);
      return NULL;
   }
   object->view = view;
   object->sampler_index = sampler_index;
   object->next = cache->objects;
   cache->objects = object;
   dev->texture_cache_stats.objects++;
   return object;
}

struct cpvk_batch_wait {
   struct cpvk_texture_cache *cache;
   uint64_t generation;
};

static enum cp_texture_cache_result
cpvk_texture_cache_resolve_locked(struct cpvk_device *dev,
                                  uint64_t image_cookie,
                                  uint64_t sampler_cookie, CUstream stream,
                                  uint64_t stream_serial, CUtexObject *out,
                                  struct cpvk_batch_wait *waits,
                                  size_t *num_waits, size_t max_waits)
{
   struct cpvk_image_view *view = image_cookie &&
      image_cookie <= dev->texture_cache_view_count
      ? dev->texture_cache_views[image_cookie - 1] : NULL;
   *out = 0;
   if (!view || !view->image || !sampler_cookie) {
      dev->texture_cache_stats.fallbacks++;
      return CP_TEXTURE_CACHE_SOFT_FALLBACK;
   }
   unsigned sampler_index = (unsigned)(sampler_cookie - 1);
   struct cpvk_image *image = view->image;
   bool cube = false;
   CUDA_TEXTURE_DESC texture_desc = {0};
   if (!cpvk_cache_view_eligible(view, &cube) ||
       !cpvk_cache_sampler_desc(dev, sampler_index, cube, &texture_desc))
      goto miss;
   struct cpvk_cache_format format;
   if (!cpvk_cache_image_eligible(image, &format)) {
      if (cp_debug->debug_tex)
         fprintf(stderr, "cudavk: texture cache image miss create=%u alias=%u "
                 "mem=%p kind=%d ledger=%u tiling=%u samples=%u format=%u\n",
                 image->cache_create_eligible, image->cache_alias,
                 (void *)image->mem, image->mem ? image->mem->kind : -1,
                 image->mem ? image->mem->binding_ledger_failed : 0,
                 image->vk.tiling, image->vk.samples, image->vk_format);
      goto miss;
   }
   unsigned target = view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ? 2
      : (view->vk.view_type == VK_IMAGE_VIEW_TYPE_3D ? 4 : 1);
   if (!(format.targets & target))
      goto miss;
   if (!image->texture_cache && !cpvk_cache_allocate(image, &format)) {
      if (cp_debug->debug_tex)
         fprintf(stderr, "cudavk: texture cache array allocation miss\n");
      goto miss;
   }
   struct cpvk_texture_object *object =
      cpvk_cache_object(view, sampler_index, &texture_desc);
   if (!object) {
      if (cp_debug->debug_tex)
         fprintf(stderr, "cudavk: texture cache object miss view=%u levels=%u "
                 "layers=%u sampler=%u\n", view->vk.view_type,
                 view->vk.level_count, view->vk.layer_count, sampler_index);
      goto miss;
   }
   uint64_t epoch = atomic_load_explicit(&image->content_epoch,
                                         memory_order_acquire);
   bool rebuilt_here = false;
   if (!image->texture_cache->ready_valid ||
       image->texture_cache->scheduled_epoch != epoch) {
      if (!cpvk_cache_rebuild(image, stream, &format, epoch))
         goto miss;
      rebuilt_here = true;
   }
   if (atomic_load_explicit(&image->content_epoch, memory_order_acquire) != epoch)
      goto miss;
   struct cpvk_texture_cache *cache = image->texture_cache;
   bool memo_hit = rebuilt_here;
   if (stream_serial) {
      for (unsigned i = 0; i < ARRAY_SIZE(cache->persistent_waits); i++)
         if (cache->persistent_waits[i].serial == stream_serial &&
             cache->persistent_waits[i].generation == cache->ready_generation) {
            memo_hit = true;
            break;
         }
   }
   if (rebuilt_here && *num_waits < max_waits) {
      waits[*num_waits] = (struct cpvk_batch_wait) { cache, cache->ready_generation };
      (*num_waits)++;
   }
   for (size_t i = 0; !memo_hit && i < *num_waits; i++)
      if (waits[i].cache == cache && waits[i].generation == cache->ready_generation) {
         memo_hit = true;
         break;
      }
   if (!memo_hit) {
      CUresult err = cuStreamWaitEvent(stream, cache->ready, 0);
      if (err != CUDA_SUCCESS) {
         cpvk_cache_device_loss(dev);
         goto miss;
      }
      dev->texture_cache_stats.event_waits++;
      if (*num_waits < max_waits) {
         waits[*num_waits] = (struct cpvk_batch_wait) { cache, cache->ready_generation };
         (*num_waits)++;
      } else {
         dev->texture_cache_stats.event_wait_overflow++;
      }
   } else {
      dev->texture_cache_stats.event_wait_skips++;
   }
   if (stream_serial) {
      for (unsigned i = 0; i < ARRAY_SIZE(cache->persistent_waits); i++) {
         if (cache->persistent_waits[i].serial == stream_serial ||
             !cache->persistent_waits[i].serial) {
            cache->persistent_waits[i].serial = stream_serial;
            cache->persistent_waits[i].generation = cache->ready_generation;
            break;
         }
      }
   }
   *out = object->object;
   dev->texture_cache_stats.hits++;
   return CP_TEXTURE_CACHE_READY;
miss:
   dev->texture_cache_stats.fallbacks++;
   bool fatal = atomic_load_explicit(&dev->device_lost,
                                     memory_order_acquire);
   return fatal ? CP_TEXTURE_CACHE_FATAL : CP_TEXTURE_CACHE_SOFT_FALLBACK;
}

enum cp_texture_cache_result
cpvk_texture_cache_resolve(void *private_data, uint64_t image_cookie,
                           uint64_t sampler_cookie, CUstream stream,
                           uint64_t stream_serial, CUtexObject *out)
{
   struct cpvk_device *dev = private_data;
   simple_mtx_lock(&dev->texture_cache_lock);
   struct cpvk_batch_wait wait;
   size_t num_waits = 0;
   enum cp_texture_cache_result result = cpvk_texture_cache_resolve_locked(
      dev, image_cookie, sampler_cookie, stream, stream_serial, out,
      &wait, &num_waits, 1);
   simple_mtx_unlock(&dev->texture_cache_lock);
   return result;
}

static uint64_t
cpvk_cookie_hash(uint64_t image, uint64_t sampler)
{
   uint64_t x = image ^ (sampler + 0x9e3779b97f4a7c15ull +
                         (image << 6) + (image >> 2));
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ull;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebull;
   return x ^ (x >> 31);
}

enum cp_texture_cache_result
cpvk_texture_cache_resolve_batch(void *private_data,
                                 const uint64_t *image_cookies,
                                 const uint64_t *sampler_cookies,
                                 size_t count, CUstream stream,
                                 uint64_t stream_serial,
                                 CUtexObject *objects)
{
   struct cpvk_device *dev = private_data;
   if (!count || count > SIZE_MAX / sizeof(*objects) ||
       count > SIZE_MAX / sizeof(*image_cookies) ||
       count > SIZE_MAX / sizeof(*sampler_cookies))
      return CP_TEXTURE_CACHE_SOFT_FALLBACK;

   size_t cap = 1;
   bool can_dedup = count <= SIZE_MAX / 2 && count < UINT32_MAX &&
                    count <= SIZE_MAX / sizeof(uint64_t);
   while (can_dedup && cap < count * 2) {
      if (cap > SIZE_MAX / 2) {
         can_dedup = false;
         break;
      }
      cap *= 2;
   }
   uint32_t *slots = NULL, *scatter = NULL;
   uint64_t *unique_images = NULL, *unique_samplers = NULL;
   CUtexObject *unique_objects = NULL;
   struct cpvk_batch_wait *waits = NULL;
   size_t off = 0, add = 0;
   bool workspace_ok = can_dedup;
#define CPVK_BATCH_ADD(n, type) do {                                      \
      if (__builtin_mul_overflow((n), sizeof(type), &add) ||              \
          __builtin_add_overflow(off, add, &off))                         \
         workspace_ok = false;                                            \
   } while (0)
   CPVK_BATCH_ADD(cap, uint32_t);
   CPVK_BATCH_ADD(count, uint32_t);
   if (off > SIZE_MAX - 7)
      workspace_ok = false;
   off = workspace_ok ? ALIGN_POT(off, 8) : 0;
   CPVK_BATCH_ADD(count, uint64_t);
   CPVK_BATCH_ADD(count, uint64_t);
   CPVK_BATCH_ADD(count, CUtexObject);
   CPVK_BATCH_ADD(count, struct cpvk_batch_wait);
#undef CPVK_BATCH_ADD
   if (workspace_ok && off > dev->texture_batch_workspace_size) {
      void *grown = malloc(off);
      if (grown) {
         free(dev->texture_batch_workspace);
         dev->texture_batch_workspace = grown;
         dev->texture_batch_workspace_size = off;
      } else {
         workspace_ok = false;
      }
   }
   if (workspace_ok) {
      char *workspace = dev->texture_batch_workspace;
      size_t cursor = 0;
      slots = (uint32_t *)(workspace + cursor);
      cursor += cap * sizeof(*slots);
      scatter = (uint32_t *)(workspace + cursor);
      cursor += count * sizeof(*scatter);
      cursor = ALIGN_POT(cursor, 8);
      unique_images = (uint64_t *)(workspace + cursor);
      cursor += count * sizeof(*unique_images);
      unique_samplers = (uint64_t *)(workspace + cursor);
      cursor += count * sizeof(*unique_samplers);
      unique_objects = (CUtexObject *)(workspace + cursor);
      cursor += count * sizeof(*unique_objects);
      waits = (struct cpvk_batch_wait *)(workspace + cursor);
      memset(slots, 0, cap * sizeof(*slots));
   }
   bool dedup = workspace_ok;
   size_t unique = 0;
   if (dedup) {
      for (size_t i = 0; i < count; i++) {
         size_t slot = cpvk_cookie_hash(image_cookies[i], sampler_cookies[i]) &
                       (cap - 1);
         while (slots[slot]) {
            size_t u = slots[slot] - 1;
            if (unique_images[u] == image_cookies[i] &&
                unique_samplers[u] == sampler_cookies[i])
               break;
            slot = (slot + 1) & (cap - 1);
         }
         if (!slots[slot]) {
            unique_images[unique] = image_cookies[i];
            unique_samplers[unique] = sampler_cookies[i];
            slots[slot] = (uint32_t)(unique + 1);
            unique++;
         }
         scatter[i] = slots[slot] - 1;
      }
   } else {
      unique = count;
   }

   size_t num_waits = 0;
   size_t max_waits = waits ? unique : 0;
   simple_mtx_lock(&dev->texture_cache_lock);
   dev->texture_cache_stats.resolve_batches++;
   dev->texture_cache_stats.resolve_cells += count;
   dev->texture_cache_stats.resolve_unique += unique;
   enum cp_texture_cache_result result = CP_TEXTURE_CACHE_READY;
   for (size_t i = 0; i < unique; i++) {
      uint64_t image = dedup ? unique_images[i] : image_cookies[i];
      uint64_t sampler = dedup ? unique_samplers[i] : sampler_cookies[i];
      CUtexObject *object = dedup ? &unique_objects[i] : &objects[i];
      result = cpvk_texture_cache_resolve_locked(
         dev, image, sampler, stream, stream_serial, object,
         waits, &num_waits, max_waits);
      if (result != CP_TEXTURE_CACHE_READY)
         break;
   }
   if (result == CP_TEXTURE_CACHE_READY && dedup) {
      for (size_t i = 0; i < count; i++)
         objects[i] = unique_objects[scatter[i]];
      /* Keep resource statistics in logical table-cell units. */
      dev->texture_cache_stats.hits += count - unique;
   } else if (result != CP_TEXTURE_CACHE_READY) {
      memset(objects, 0, count * sizeof(*objects));
   }
   simple_mtx_unlock(&dev->texture_cache_lock);

   return result;
}

void
cpvk_texture_cache_view_destroy(struct cpvk_image_view *view)
{
   if (!view)
      return;
   struct cpvk_device *dev = view->dev;
   /* The entry point's CPVK_CTX_SCOPE already made this device's context
    * current; this used to set it and fold the result into `ok`. */
   bool ok = true;
   simple_mtx_lock(&dev->texture_cache_use_lock);
   if (ok)
      ok = cuCtxSynchronize() == CUDA_SUCCESS;
   simple_mtx_lock(&dev->texture_cache_lock);
   if (view->cache_cookie &&
       view->cache_cookie <= dev->texture_cache_view_count &&
       dev->texture_cache_views[view->cache_cookie - 1] == view)
      dev->texture_cache_views[view->cache_cookie - 1] = NULL;
   view->cache_cookie = 0;
   if (view->image && view->image->texture_cache) {
      struct cpvk_texture_object **link = &view->image->texture_cache->objects;
      while (*link) {
         if ((*link)->view == view) {
            struct cpvk_texture_object *dead = *link;
            *link = dead->next;
            if (cuTexObjectDestroy(dead->object) != CUDA_SUCCESS)
               ok = false;
            free(dead);
            continue;
         }
         link = &(*link)->next;
      }
   }
   if (!ok)
      cpvk_cache_device_loss(dev);
   simple_mtx_unlock(&dev->texture_cache_lock);
   simple_mtx_unlock(&dev->texture_cache_use_lock);
}

void
cpvk_texture_cache_sampler_destroy(struct cpvk_sampler *sampler)
{
   (void)sampler; /* immutable table index remains valid by design */
}

static bool
cpvk_cache_drop_locked(struct cpvk_image *image,
                       uint64_t *arrays, uint64_t *objects)
{
   struct cpvk_device *dev = image->dev;
   struct cpvk_texture_cache *cache = image->texture_cache;
   if (!cache)
      return true;
   bool ok = true;
   while (cache->objects) {
      struct cpvk_texture_object *next = cache->objects->next;
      if (cuTexObjectDestroy(cache->objects->object) != CUDA_SUCCESS)
         ok = false;
      if (objects)
         (*objects)++;
      free(cache->objects);
      cache->objects = next;
   }
   for (unsigned l = 0; l < CPVK_MAX_MIP_LEVELS; l++)
      if (cache->surfaces[l] &&
          cuSurfObjectDestroy(cache->surfaces[l]) != CUDA_SUCCESS)
         ok = false;
   if (cache->ready && cuEventDestroy(cache->ready) != CUDA_SUCCESS)
      ok = false;
   if (cache->array) {
      if (cuMipmappedArrayDestroy(cache->array) != CUDA_SUCCESS)
         ok = false;
      if (arrays)
         (*arrays)++;
   }
   if (image->texture_cache_prev) {
      *image->texture_cache_prev = image->texture_cache_next;
      if (image->texture_cache_next)
         image->texture_cache_next->texture_cache_prev =
            image->texture_cache_prev;
   }
   image->texture_cache_next = NULL;
   image->texture_cache_prev = NULL;
   dev->texture_cache_bytes -= cache->bytes;
   free(cache);
   image->texture_cache = NULL;
   return ok;
}

void
cpvk_texture_cache_use_begin(void *private_data)
{
   struct cpvk_device *dev = private_data;
   simple_mtx_lock(&dev->texture_cache_use_lock);
}

void
cpvk_texture_cache_use_end(void *private_data)
{
   struct cpvk_device *dev = private_data;
   simple_mtx_unlock(&dev->texture_cache_use_lock);
}

enum cp_texture_cache_purge_result
cpvk_texture_cache_purge(void *private_data)
{
   struct cpvk_device *dev = private_data;
   if (!dev || !cp_debug->texture_cache)
      return CP_TEXTURE_CACHE_PURGE_NONE;
   simple_mtx_lock(&dev->texture_cache_use_lock);
   simple_mtx_lock(&dev->texture_cache_lock);
   if (cuCtxSynchronize() != CUDA_SUCCESS) {
      cpvk_cache_device_loss(dev);
      simple_mtx_unlock(&dev->texture_cache_lock);
      simple_mtx_unlock(&dev->texture_cache_use_lock);
      return CP_TEXTURE_CACHE_PURGE_FATAL;
   }
   bool purged = dev->texture_cache_images != NULL;
   bool ok = true;
   uint64_t reclaimed = dev->texture_cache_bytes;
   uint64_t arrays = 0, objects = 0;
   while (dev->texture_cache_images)
      ok &= cpvk_cache_drop_locked(dev->texture_cache_images,
                                   &arrays, &objects);
   if (purged && ok) {
      dev->texture_cache_stats.purges++;
      dev->texture_cache_stats.purge_reclaimed_bytes += reclaimed;
      dev->texture_cache_stats.purge_reclaimed_arrays += arrays;
      dev->texture_cache_stats.purge_reclaimed_objects += objects;
   }
   if (!ok)
      cpvk_cache_device_loss(dev);
   simple_mtx_unlock(&dev->texture_cache_lock);
   simple_mtx_unlock(&dev->texture_cache_use_lock);
   return !ok ? CP_TEXTURE_CACHE_PURGE_FATAL :
      (purged ? CP_TEXTURE_CACHE_PURGE_RECLAIMED :
                CP_TEXTURE_CACHE_PURGE_NONE);
}

void
cpvk_texture_cache_image_destroy(struct cpvk_image *image)
{
   if (!image)
      return;
   struct cpvk_device *dev = image->dev;
   bool ok = true;
   simple_mtx_lock(&dev->texture_cache_use_lock);
   if (ok)
      ok = cuCtxSynchronize() == CUDA_SUCCESS;
   simple_mtx_lock(&dev->texture_cache_lock);
   ok &= cpvk_cache_drop_locked(image, NULL, NULL);
   if (image->writer_event) {
      if (cuEventDestroy(image->writer_event) != CUDA_SUCCESS)
         ok = false;
      image->writer_event = NULL;
      image->writer_event_valid = false;
   }
   if (!ok)
      cpvk_cache_device_loss(dev);
   simple_mtx_unlock(&dev->texture_cache_lock);
   simple_mtx_unlock(&dev->texture_cache_use_lock);
}

void
cpvk_texture_cache_report(struct cpvk_device *dev)
{
   if (!cp_debug->texture_cache_stats)
      return;
   fprintf(stderr, "cudavk: texture cache resources: hits=%" PRIu64
           " fallbacks=%" PRIu64 " rebuilds=%" PRIu64 " bytes=%" PRIu64
           " peak_bytes=%" PRIu64 " arrays=%" PRIu64 " objects=%" PRIu64
           " alloc_failures=%" PRIu64 " object_failures=%" PRIu64
           " event_waits=%" PRIu64 " event_wait_skips=%" PRIu64
           " event_wait_overflow=%" PRIu64
           " resolve_batches=%" PRIu64 " resolve_cells=%" PRIu64
           " resolve_unique=%" PRIu64 " purges=%" PRIu64
           " purge_reclaimed_bytes=%" PRIu64
           " purge_reclaimed_arrays=%" PRIu64
           " purge_reclaimed_objects=%" PRIu64 "\n",
           dev->texture_cache_stats.hits, dev->texture_cache_stats.fallbacks,
           dev->texture_cache_stats.rebuilds, dev->texture_cache_bytes,
           dev->texture_cache_stats.peak_bytes, dev->texture_cache_stats.arrays, dev->texture_cache_stats.objects,
           dev->texture_cache_stats.alloc_failures,
           dev->texture_cache_stats.object_failures,
           dev->texture_cache_stats.event_waits,
           dev->texture_cache_stats.event_wait_skips,
           dev->texture_cache_stats.event_wait_overflow,
           dev->texture_cache_stats.resolve_batches,
           dev->texture_cache_stats.resolve_cells,
           dev->texture_cache_stats.resolve_unique,
           dev->texture_cache_stats.purges,
           dev->texture_cache_stats.purge_reclaimed_bytes,
           dev->texture_cache_stats.purge_reclaimed_arrays,
           dev->texture_cache_stats.purge_reclaimed_objects);
}
