/*
 * Descriptor sets, command buffers and dispatch.
 *
 * The compute argument ABI is the Gallium-hosted driver's, unchanged, because
 * the generated kernel is the same kernel: args[0] points at the grid size,
 * and args[18 + i] is the base address of buffer i, which is what
 * emit_const_buf_base() dereferences for a 32-bit-index load_ubo or
 * load_ssbo. So binding a descriptor set is filling in those slots, and a
 * descriptor set is an array of device addresses -- no descriptor memory, no
 * VkDeviceMemory per set, and none of the fault storm that arrangement caused.
 */

#include "vk_format.h"
#include "util/format/u_format.h"
#include <time.h>
#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"

/* Last: intercepts the CUDA entry points for the iteration 26 census. */
#include "cp_smallop_tele.h"

/* CPVK_ARG_UBO_BASE and CPVK_MAX_ARG_BUFS come from cp_shader_abi.h, which is
 * where the kernels' argument-block layout is stated and asserted. */
#define CPVK_ARG_SLOTS      34

/* ---------------------------------------------------------------- pools */

static void
cpvk_descriptor_set_free(struct cpvk_device *dev,
                         struct cpvk_descriptor_set *set)
{
   if (!set)
      return;

   if (set->pool) {
      struct cpvk_descriptor_pool *pool = set->pool;
      if (pool->allocated_sets)
         pool->allocated_sets--;
      for (unsigned type = 0; type < CPVK_DESCRIPTOR_TYPE_COUNT; type++) {
         assert(pool->used[type] >= set->pool_counts[type]);
         pool->used[type] -= set->pool_counts[type];
      }
      struct cpvk_descriptor_set **p = &pool->sets;
      while (*p && *p != set)
         p = &(*p)->pool_next;
      if (*p)
         *p = set->pool_next;
   }
   vk_object_free(&dev->vk, NULL, set);
}

static void
cpvk_descriptor_pool_clear(struct cpvk_device *dev,
                           struct cpvk_descriptor_pool *pool)
{
   while (pool && pool->sets) {
      struct cpvk_descriptor_set *set = pool->sets;
      pool->sets = set->pool_next;
      set->pool = NULL;
      vk_object_free(&dev->vk, NULL, set);
   }
   if (pool) {
      pool->allocated_sets = 0;
      memset(pool->used, 0, sizeof(pool->used));
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorPool(VkDevice _device,
                          const VkDescriptorPoolCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_descriptor_pool *pool =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                       VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->max_sets = pCreateInfo->maxSets;
   for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; i++) {
      unsigned type = pCreateInfo->pPoolSizes[i].type;
      if (type < CPVK_DESCRIPTOR_TYPE_COUNT)
         pool->capacity[type] += pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   *pDescriptorPool = cpvk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool _pool,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_descriptor_pool, pool, _pool);
   CPVK_CTX_SCOPE(dev);

   if (pool) {
      cpvk_descriptor_pool_clear(dev, pool);
      vk_object_free(&dev->vk, pAllocator, pool);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_AllocateDescriptorSets(VkDevice _device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_descriptor_pool, pool,
                  pAllocateInfo->descriptorPool);
   CPVK_CTX_SCOPE(dev);

   uint64_t required[CPVK_DESCRIPTOR_TYPE_COUNT] = {0};
   if (!pool || pAllocateInfo->descriptorSetCount >
                pool->max_sets - pool->allocated_sets)
      return vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);
   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      if (!layout)
         continue;
      for (unsigned b = 0; b < layout->num_bindings; b++) {
         unsigned type = layout->bindings[b].type;
         if (type >= CPVK_DESCRIPTOR_TYPE_COUNT)
            return vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);
         required[type] += layout->bindings[b].count;
      }
   }
   for (unsigned type = 0; type < CPVK_DESCRIPTOR_TYPE_COUNT; type++)
      if (required[type] > pool->capacity[type] - pool->used[type])
         return vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);

   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      unsigned num_bindings = layout ? layout->num_bindings : 0;
      unsigned num_descriptors = layout ? layout->num_descriptors : 0;
      size_t desc_size = (size_t)num_descriptors * sizeof(struct cpvk_descriptor);
      size_t bindings_size =
         (size_t)num_bindings * sizeof(struct cpvk_descriptor_binding);
      size_t addrs_size = (size_t)num_descriptors * sizeof(CUdeviceptr);
      size_t views_size =
         (size_t)num_descriptors * sizeof(struct cpvk_image_view *);
      size_t flags_size = num_descriptors * sizeof(bool);

      struct cpvk_descriptor_set *set =
         vk_object_zalloc(&dev->vk, NULL, sizeof(*set) + desc_size +
                          bindings_size + addrs_size + views_size + flags_size,
                          VK_OBJECT_TYPE_DESCRIPTOR_SET);
      if (!set) {
         for (uint32_t j = 0; j < i; j++) {
            VK_FROM_HANDLE(cpvk_descriptor_set, s, pDescriptorSets[j]);
            cpvk_descriptor_set_free(dev, s);
            pDescriptorSets[j] = VK_NULL_HANDLE;
         }
         return vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);
      }
      char *tail = (char *)(set + 1);
      set->host = desc_size ? (struct cpvk_descriptor *)tail : NULL;
      tail += desc_size;
      set->bindings = (struct cpvk_descriptor_binding *)tail;
      tail += bindings_size;
      set->addrs = (CUdeviceptr *)tail;
      tail += addrs_size;
      set->views = (struct cpvk_image_view **)tail;
      tail += views_size;
      set->immutable = (bool *)tail;
      if (layout) {
         set->num_bindings = num_bindings;
         set->num_descriptors = num_descriptors;
         memcpy(set->bindings, layout->bindings, bindings_size);
         memcpy(set->immutable, layout->immutable, flags_size);
      }
      set->pool = pool;
      if (pool) {
         set->pool_next = pool->sets;
         pool->sets = set;
         pool->allocated_sets++;
         if (layout) {
            for (unsigned b = 0; b < layout->num_bindings; b++) {
               unsigned type = layout->bindings[b].type;
               set->pool_counts[type] += layout->bindings[b].count;
               pool->used[type] += layout->bindings[b].count;
            }
         }
      }

      if (desc_size) {
         /* A descriptor nothing ever writes still gets read by some shaders.
          * Point it at the null page instead of address zero. */
         for (unsigned d = 0; d < layout->num_descriptors; d++) {
            set->host[d].base = dev->null_data;
            if (layout->immutable[d])
               set->host[d].sampler_index_or_img_stride =
                  layout->immutable_sampler[d];
            if (layout->immutable[d])
               set->host[d].sampler_cookie =
                  (uint64_t)layout->immutable_sampler[d] + 1;
         }
      }

      pDescriptorSets[i] = cpvk_descriptor_set_to_handle(set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_FreeDescriptorSets(VkDevice _device, VkDescriptorPool pool,
                        uint32_t count, const VkDescriptorSet *pSets)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set, set, pSets[i]);
      cpvk_descriptor_set_free(dev, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_ResetDescriptorPool(VkDevice _device, VkDescriptorPool pool,
                         VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_descriptor_pool, desc_pool, pool);
   CPVK_CTX_SCOPE(dev);
   cpvk_descriptor_pool_clear(dev, desc_pool);
   return VK_SUCCESS;
}

/*
 * One descriptor written, shared by vkUpdateDescriptorSets and the template
 * path. They differ only in where the VkDescriptorImageInfo and
 * VkDescriptorBufferInfo come from -- an array, or a stride into a blob of
 * application memory -- and a second copy of this switch would be a second
 * thing to be right about a descriptor the first one is wrong about.
 */
/*
 * The storage-image half of a descriptor, resolved from the view.
 *
 * Recomputed rather than only written once: vkBindImageMemory2 may legally
 * follow the descriptor write, and a row cached before the bind names the
 * null page for the life of the set.
 */
static void
cpvk_storage_image_row(struct cpvk_descriptor *desc,
                       const struct cpvk_image_view *view)
{
   const struct cpvk_image *img = view ? view->image : NULL;
   if (!img || !img->mem)
      return;

   unsigned level = view->vk.base_mip_level;
   uint64_t layer_offset =
      (uint64_t)view->vk.base_array_layer * img->level_size[level];
   desc->base = img->mem->dev_ptr + img->offset +
                img->level_offset[level] + layer_offset;
   /* The view's subresource, not the whole image, is addressable storage. */
   desc->width_or_range = MAX2(img->vk.extent.width >> level, 1u);
   desc->height = MAX2(img->vk.extent.height >> level, 1u);
   desc->depth = img->vk.image_type == VK_IMAGE_TYPE_3D
      ? MAX2(img->vk.extent.depth >> level, 1u)
      : MAX2(view->vk.layer_count, 1u);
   desc->row_stride = img->row_stride[level];
   /* One z step is one slice of this level, not the whole level. */
   enum pipe_format pfmt = vk_format_to_pipe_format(view->vk.format);
   unsigned rows = util_format_get_nblocksy(
      pfmt, MAX2(img->vk.extent.height >> level, 1u));
   desc->sampler_index_or_img_stride = img->row_stride[level] * rows;
   desc->base_offset = 0;
}

static bool
cpvk_descriptor_location(const struct cpvk_descriptor_set *set,
                         uint32_t binding, uint32_t element,
                         VkDescriptorType type, unsigned *flat)
{
   unsigned i = 0;
   while (i < set->num_bindings && set->bindings[i].binding != binding)
      i++;
   while (i < set->num_bindings) {
      const struct cpvk_descriptor_binding *map = &set->bindings[i];
      if (map->type != type)
         return false;
      if (element < map->count) {
         *flat = map->flat + element;
         return *flat < set->num_descriptors;
      }
      element -= map->count;
      i++;
   }
   return false;
}

static void
cpvk_write_descriptor(struct cpvk_descriptor_set *set, unsigned flat,
                      VkDescriptorType type,
                      const VkDescriptorImageInfo *ii,
                      const VkDescriptorBufferInfo *bi)
{
   if (cp_debug->debug_rt)
      fprintf(stderr, "descw flat=%u type=%u ii=%p bi=%p\n", flat, type,
              (const void *)ii, (const void *)bi);

   if (flat >= set->num_descriptors)
      return;

   /* A rewritten descriptor is a new kind; drop any storage-image view the
    * previous write left behind. */
   set->views[flat] = NULL;
   set->host[flat].image_cookie = 0;
   if (!set->immutable[flat])
      set->host[flat].sampler_cookie = 0;

   switch (type) {
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: {
      if (!bi)
         break;
      VK_FROM_HANDLE(cpvk_buffer, buffer, bi->buffer);
      set->addrs[flat] = buffer && buffer->mem
         ? buffer->mem->dev_ptr + buffer->offset + bi->offset : 0;
      if (set->host) {
         set->host[flat].base = set->addrs[flat];
         /*
          * The bound range, which `buffer.length()` divides by its array
          * stride. VK_WHOLE_SIZE is what is left of the buffer after the
          * descriptor's own offset; a dynamic offset moves the base and does
          * not shorten the range, which is what the spec says the shader
          * sees.
          */
         uint64_t size = buffer ? buffer->vk.size : 0;
         uint64_t start = MIN2(bi->offset, size);
         uint64_t range = bi->range == VK_WHOLE_SIZE ? size - start
                                                     : bi->range;
         set->host[flat].width_or_range = (uint32_t)MIN2(range, size - start);
      }
      break;
   }

   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_SAMPLER: {
      if (!set->host || !ii)
         break;
      if (ii->imageView) {
         VK_FROM_HANDLE(cpvk_image_view, view, ii->imageView);
         set->host[flat].texture_info = view ? view->tex_info : 0;
         set->host[flat].image_cookie = view ? view->cache_cookie : 0;
      }
      if (ii->sampler && !set->immutable[flat]) {
         VK_FROM_HANDLE(cpvk_sampler, samp, ii->sampler);
         set->host[flat].sampler_index_or_img_stride = samp ? samp->index : 0;
         set->host[flat].sampler_cookie = samp ? (uint64_t)samp->index + 1 : 0;
      }

      /*
       * The six faces of a cube, read back from where the sampler looks:
       * base + layer * level_size. Done here rather than at view creation
       * because the view exists before anything has been copied into the
       * image, so the only honest place to ask is when the descriptor that
       * will be sampled is written.
       */
      if (cp_debug->debug_faces && ii->imageView) {
         VK_FROM_HANDLE(cpvk_image_view, fv, ii->imageView);
         struct cpvk_image *fi = fv ? fv->image : NULL;
         if (fi && fi->mem && fi->vk.array_layers == 6) {
            for (unsigned f = 0; f < 6; f++) {
               uint16_t h[4] = { 0 };
               CUdeviceptr a = fi->mem->dev_ptr + fi->offset +
                               fi->level_offset[0] +
                               (uint64_t)f * fi->level_size[0];
               cuMemcpyDtoH(h, a, sizeof(h));
               fprintf(stderr, "face %u @%p: %04x %04x %04x %04x  (%ux%u)\n",
                       f, (void *)(uintptr_t)a, h[0], h[1], h[2], h[3],
                       fi->vk.extent.width, fi->vk.extent.height);
            }
         }
      }

      if (cp_debug->debug_rt) {
         VK_FROM_HANDLE(cpvk_image_view, dv, ii->imageView);
         fprintf(stderr, "desc flat=%u type=%u tex=%p "
                 "samp=%u img=%ux%u\n", flat, type,
                 (void *)(uintptr_t)set->host[flat].texture_info,
                 set->host[flat].sampler_index_or_img_stride,
                 (dv && dv->image) ? dv->image->vk.extent.width : 0,
                 (dv && dv->image) ? dv->image->vk.extent.height : 0);
      }

      /*
       * A storage image is addressed directly by the shader rather than
       * sampled, so it needs the four fields the backend's
       * bindless_image_load and _store read out of the descriptor.
       */
      if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && ii->imageView) {
         VK_FROM_HANDLE(cpvk_image_view, view, ii->imageView);
         set->views[flat] = view;
         set->has_storage_views = true;
         cpvk_storage_image_row(&set->host[flat], view);
         set->host[flat].image_cookie = view ? view->cache_cookie : 0;
      }

      break;
   }

   default:
      break;
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_UpdateDescriptorSets(VkDevice _device, uint32_t writeCount,
                          const VkWriteDescriptorSet *pWrites,
                          uint32_t copyCount,
                          const VkCopyDescriptorSet *pCopies)
{
   if (cp_debug->debug_rt)
      fprintf(stderr, "updsets n=%u copies=%u\n", writeCount, copyCount);

   for (uint32_t w = 0; w < writeCount; w++) {
      const VkWriteDescriptorSet *write = &pWrites[w];
      VK_FROM_HANDLE(cpvk_descriptor_set, set, write->dstSet);
      if (!set)
         continue;
      for (uint32_t e = 0; e < write->descriptorCount; e++) {
         unsigned flat;
         if (!cpvk_descriptor_location(set, write->dstBinding,
                                       write->dstArrayElement + e,
                                       write->descriptorType, &flat))
            continue;
         cpvk_write_descriptor(set, flat, write->descriptorType,
                               write->pImageInfo ? &write->pImageInfo[e] : NULL,
                               write->pBufferInfo ? &write->pBufferInfo[e] : NULL);
      }
   }

   for (uint32_t c = 0; c < copyCount; c++) {
      const VkCopyDescriptorSet *copy = &pCopies[c];
      VK_FROM_HANDLE(cpvk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(cpvk_descriptor_set, dst, copy->dstSet);
      if (!src || !dst)
         continue;
      const struct cpvk_descriptor_binding *src_map =
         cpvk_find_binding(src->bindings, src->num_bindings, copy->srcBinding);
      const struct cpvk_descriptor_binding *dst_map =
         cpvk_find_binding(dst->bindings, dst->num_bindings, copy->dstBinding);
      if (!src_map || !dst_map || src_map->type != dst_map->type)
         continue;
      unsigned count = MIN2(copy->descriptorCount, src->num_descriptors);
      struct cpvk_descriptor *descriptors =
         calloc(count, sizeof(*descriptors));
      CUdeviceptr *addrs = calloc(count, sizeof(*addrs));
      struct cpvk_image_view **views = calloc(count, sizeof(*views));
      bool *valid = calloc(count, sizeof(*valid));
      if (!descriptors || !addrs || !views || !valid) {
         free(descriptors); free(addrs); free(views); free(valid);
         continue;
      }
      /* Copy through a temporary row: source and destination may overlap in
       * one set, and Vulkan defines the update as if all sources were read
       * before any destination is written. */
      for (unsigned e = 0; e < count; e++) {
         unsigned sf;
         if (cpvk_descriptor_location(src, copy->srcBinding,
                                      copy->srcArrayElement + e,
                                      src_map->type, &sf)) {
            descriptors[e] = src->host[sf];
            addrs[e] = src->addrs[sf];
            views[e] = src->views[sf];
            valid[e] = true;
         }
      }
      for (unsigned e = 0; e < count; e++) {
         unsigned df;
         if (!valid[e] ||
             !cpvk_descriptor_location(dst, copy->dstBinding,
                                       copy->dstArrayElement + e,
                                       dst_map->type, &df))
            continue;
         uint32_t immutable_sampler =
            dst->host[df].sampler_index_or_img_stride;
         uint64_t immutable_sampler_cookie = dst->host[df].sampler_cookie;
         dst->host[df] = descriptors[e];
         dst->addrs[df] = addrs[e];
         dst->views[df] = views[e];
         dst->has_storage_views |= views[e] != NULL;
         if (dst->immutable[df]) {
            dst->host[df].sampler_index_or_img_stride = immutable_sampler;
            dst->host[df].sampler_cookie = immutable_sampler_cookie;
         }
      }
      free(descriptors);
      free(addrs);
      free(views);
      free(valid);
   }
}

/* ------------------------------------------------- update templates */

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorUpdateTemplate(
   VkDevice _device, const VkDescriptorUpdateTemplateCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkDescriptorUpdateTemplate *pTemplate)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_descriptor_update_template *tmpl =
      vk_object_zalloc(&dev->vk, pAllocator,
                       sizeof(*tmpl) + pCreateInfo->descriptorUpdateEntryCount *
                                       sizeof(*tmpl->entries),
                       VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE);
   if (!tmpl)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   tmpl->entry_count = pCreateInfo->descriptorUpdateEntryCount;
   memcpy(tmpl->entries, pCreateInfo->pDescriptorUpdateEntries,
          tmpl->entry_count * sizeof(*tmpl->entries));

   *pTemplate = cpvk_descriptor_update_template_to_handle(tmpl);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDescriptorUpdateTemplate(VkDevice _device,
                                     VkDescriptorUpdateTemplate _tmpl,
                                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_descriptor_update_template, tmpl, _tmpl);
   CPVK_CTX_SCOPE(dev);

   if (tmpl)
      vk_object_free(&dev->vk, pAllocator, tmpl);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_UpdateDescriptorSetWithTemplate(VkDevice _device, VkDescriptorSet _set,
                                     VkDescriptorUpdateTemplate _tmpl,
                                     const void *pData)
{
   VK_FROM_HANDLE(cpvk_descriptor_set, set, _set);
   VK_FROM_HANDLE(cpvk_descriptor_update_template, tmpl, _tmpl);

   if (!set || !tmpl)
      return;

   for (uint32_t i = 0; i < tmpl->entry_count; i++) {
      const VkDescriptorUpdateTemplateEntry *e = &tmpl->entries[i];
      for (uint32_t j = 0; j < e->descriptorCount; j++) {
         const char *src = (const char *)pData + e->offset + j * e->stride;
         unsigned flat;
         if (!cpvk_descriptor_location(set, e->dstBinding,
                                       e->dstArrayElement + j,
                                       e->descriptorType, &flat))
            continue;
         cpvk_write_descriptor(set, flat, e->descriptorType,
                               (const VkDescriptorImageInfo *)src,
                               (const VkDescriptorBufferInfo *)src);
      }
   }
}

/* Vulkan 1.0 captures may enable the extension and use its aliases. */
VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorUpdateTemplateKHR(
   VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *info,
   const VkAllocationCallbacks *alloc, VkDescriptorUpdateTemplate *tmpl)
{
   return cpvk_CreateDescriptorUpdateTemplate(device, info, alloc, tmpl);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDescriptorUpdateTemplateKHR(
   VkDevice device, VkDescriptorUpdateTemplate tmpl,
   const VkAllocationCallbacks *alloc)
{
   cpvk_DestroyDescriptorUpdateTemplate(device, tmpl, alloc);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_UpdateDescriptorSetWithTemplateKHR(
   VkDevice device, VkDescriptorSet set, VkDescriptorUpdateTemplate tmpl,
   const void *data)
{
   cpvk_UpdateDescriptorSetWithTemplate(device, set, tmpl, data);
}

/* -------------------------------------------------------- command buffers */

static void
cpvk_cmd_release_objects(struct cpvk_cmd_buffer *cmd)
{
   for (unsigned i = 0; i < cmd->num_retained_pipelines; i++)
      cpvk_pipeline_unref(cmd->retained_pipelines[i]);
   cmd->num_retained_pipelines = 0;
   for (unsigned i = 0; i < cmd->num_retained_queries; i++)
      cpvk_query_pool_unref(cmd->retained_queries[i]);
   cmd->num_retained_queries = 0;
   for (unsigned i = 0; i < cmd->num_retained_events; i++)
      cpvk_event_unref(cmd->retained_events[i]);
   cmd->num_retained_events = 0;
}

static bool
cpvk_cmd_retain_pipeline(struct cpvk_cmd_buffer *cmd,
                         struct cpvk_pipeline *pipeline)
{
   if (!pipeline)
      return true;
   for (unsigned i = 0; i < cmd->num_retained_pipelines; i++)
      if (cmd->retained_pipelines[i] == pipeline)
         return true;
   if (cmd->num_retained_pipelines == cmd->max_retained_pipelines) {
      unsigned cap = cmd->max_retained_pipelines ?
                     cmd->max_retained_pipelines * 2 : 8;
      void *p = realloc(cmd->retained_pipelines,
                        cap * sizeof(*cmd->retained_pipelines));
      if (!p)
         return false;
      cmd->retained_pipelines = p;
      cmd->max_retained_pipelines = cap;
   }
   cpvk_pipeline_ref(pipeline);
   cmd->retained_pipelines[cmd->num_retained_pipelines++] = pipeline;
   return true;
}

static bool
cpvk_cmd_retain_query(struct cpvk_cmd_buffer *cmd,
                      struct cpvk_query_pool *pool)
{
   if (!pool)
      return true;
   for (unsigned i = 0; i < cmd->num_retained_queries; i++)
      if (cmd->retained_queries[i] == pool)
         return true;
   if (cmd->num_retained_queries == cmd->max_retained_queries) {
      unsigned cap = cmd->max_retained_queries ?
                     cmd->max_retained_queries * 2 : 4;
      void *p = realloc(cmd->retained_queries,
                        cap * sizeof(*cmd->retained_queries));
      if (!p)
         return false;
      cmd->retained_queries = p;
      cmd->max_retained_queries = cap;
   }
   cpvk_query_pool_ref(pool);
   cmd->retained_queries[cmd->num_retained_queries++] = pool;
   return true;
}

static bool
cpvk_cmd_retain_event(struct cpvk_cmd_buffer *cmd, struct cpvk_event *event)
{
   if (!event)
      return true;
   for (unsigned i = 0; i < cmd->num_retained_events; i++)
      if (cmd->retained_events[i] == event)
         return true;
   if (cmd->num_retained_events == cmd->max_retained_events) {
      unsigned cap = cmd->max_retained_events ?
                     cmd->max_retained_events * 2 : 4;
      void *p = realloc(cmd->retained_events,
                        cap * sizeof(*cmd->retained_events));
      if (!p)
         return false;
      cmd->retained_events = p;
      cmd->max_retained_events = cap;
   }
   cpvk_event_ref(event);
   cmd->retained_events[cmd->num_retained_events++] = event;
   return true;
}

/* The staging blocks vkCmdUpdateBuffer allocated. Released when the recording
 * they belong to goes away, which Vulkan forbids while a submission of it is
 * still running. */
static void
cpvk_cmd_free_inline_blocks(struct cpvk_cmd_buffer *cmd)
{
   for (unsigned i = 0; i < cmd->num_inline_blocks; i++)
      cuMemFree(cmd->inline_blocks[i]);
   cmd->num_inline_blocks = 0;
}

static void
cpvk_cmd_buffer_reset(struct vk_command_buffer *vk_cmd,
                      VkCommandBufferResetFlags flags)
{
   struct cpvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct cpvk_cmd_buffer, vk);

   cpvk_cmd_release_objects(cmd);
   vk_command_buffer_reset(&cmd->vk);
   cmd->num_dispatches = 0;
   cmd->num_ops = 0;
   cmd->num_scopes = 0;
   cmd->active_scope = CP_RENDER_SCOPE_NONE;
   cmd->next_scope_serial = 0;
   cmd->graphics_pipeline = NULL;
   cmd->compute_pipeline = NULL;
   memset(cmd->graphics_addrs, 0, sizeof(cmd->graphics_addrs));
   memset(cmd->compute_addrs, 0, sizeof(cmd->compute_addrs));
   memset(cmd->vb_base, 0, sizeof(cmd->vb_base));
   cmd->num_vb = 0;
   cmd->has_fb = false;
   cmd->index_ptr = NULL;
   cmd->index_size = 0;
   cmd->vs_push_size = 0;
   cmd->fs_push_size = 0;
   cmd->compute_push_size = 0;
   for (unsigned i = 0; i < cmd->num_desc_retired; i++) {
      cuMemFree(cmd->desc_retired[i].dev);
      free(cmd->desc_retired[i].host);
   }
   cmd->num_desc_retired = 0;
   cmd->desc_arena_used = 0;
   cmd->desc_arena_dirty = false;
   cpvk_cmd_free_inline_blocks(cmd);
}

static void
cpvk_cmd_buffer_destroy(struct vk_command_buffer *vk_cmd)
{
   struct cpvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct cpvk_cmd_buffer, vk);

   cpvk_cmd_release_objects(cmd);
   vk_command_buffer_finish(&cmd->vk);
   cpvk_cmd_free_inline_blocks(cmd);
   free(cmd->inline_blocks);
   for (unsigned i = 0; i < cmd->num_desc_retired; i++) {
      cuMemFree(cmd->desc_retired[i].dev);
      free(cmd->desc_retired[i].host);
   }
   free(cmd->desc_retired);
   free(cmd->retained_pipelines);
   free(cmd->retained_queries);
   free(cmd->retained_events);
   if (cmd->desc_arena)
      cuMemFree(cmd->desc_arena);
   if (cmd->desc_arena_host)
      free(cmd->desc_arena_host);
   free(cmd->ops);
   free(cmd->scopes);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

static VkResult
cpvk_cmd_buffer_create(struct vk_command_pool *pool,
                       VkCommandBufferLevel level,
                       struct vk_command_buffer **out)
{
   struct cpvk_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &cpvk_cmd_buffer_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }

   *out = &cmd->vk;
   return VK_SUCCESS;
}

const struct vk_command_buffer_ops cpvk_cmd_buffer_ops = {
   .create = cpvk_cmd_buffer_create,
   .reset = cpvk_cmd_buffer_reset,
   .destroy = cpvk_cmd_buffer_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                        const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   cpvk_cmd_free_inline_blocks(cmd);
   cmd->num_dispatches = 0;
   cmd->num_ops = 0;
   cmd->num_scopes = 0;
   cmd->active_scope = CP_RENDER_SCOPE_NONE;
   cmd->next_scope_serial = 0;
   cmd->graphics_pipeline = NULL;
   cmd->compute_pipeline = NULL;
   memset(cmd->graphics_addrs, 0, sizeof(cmd->graphics_addrs));
   memset(cmd->compute_addrs, 0, sizeof(cmd->compute_addrs));
   memset(cmd->vb_base, 0, sizeof(cmd->vb_base));
   cmd->num_vb = 0;
   cmd->has_fb = false;
   cmd->index_ptr = NULL;
   cmd->index_size = 0;
   cmd->vs_push_size = 0;
   cmd->fs_push_size = 0;
   cmd->compute_push_size = 0;
   for (unsigned i = 0; i < cmd->num_desc_retired; i++) {
      cuMemFree(cmd->desc_retired[i].dev);
      free(cmd->desc_retired[i].host);
   }
   cmd->num_desc_retired = 0;
   cmd->desc_arena_used = 0;
   cmd->desc_arena_dirty = false;
   if (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)
      cmd->active_scope = CP_RENDER_SCOPE_INHERITED;
   return VK_SUCCESS;
}

static bool cpvk_draws_mergeable(const struct cpvk_draw_cmd *a,
                                 const struct cpvk_draw_cmd *b);

/*
 * The recorded pass is a program, so decide the batch partition when the
 * program is complete instead of rediscovering it every submission. This is
 * where the Gallium-shaped "watch the draws go past" model starts giving way:
 * the pairwise mergeability of consecutive draws is a property of the
 * recording, computed here once, and a command buffer submitted N times pays
 * for its partition once instead of N times.
 *
 * Only consecutive draw ops in one scope are planned. A draw after a clear, a
 * copy or a barrier keeps plan_prev NULL and takes the dynamic path, which
 * also keeps imported secondary ops correct: this walk runs over the primary's
 * final array, overwriting whatever plan the secondary computed for its own.
 */
static void
cpvk_plan_batches(struct cpvk_cmd_buffer *cmd)
{
   const struct cpvk_op *prev = NULL;
   for (unsigned s = 0; s < cmd->num_scopes; s++) {
      cmd->scopes[s].planned_draws = 0;
      cmd->scopes[s].planned_blended_draws = 0;
      cmd->scopes[s].planned_tris = 0;
   }
   for (unsigned i = 0; i < cmd->num_ops; i++) {
      struct cpvk_op *op = &cmd->ops[i];
      if (op->kind == CPVK_OP_DRAW) {
         struct cpvk_draw_cmd *d = &op->draw_cmd;
         if (op->scope_index < cmd->num_scopes) {
            struct cp_render_scope *sc = &cmd->scopes[op->scope_index];
            sc->planned_draws++;
            sc->planned_tris +=
               (uint64_t)cp_triangles_for_draw(d->call.mode, d->range.count) *
               MAX2(d->call.instance_count, 1u);
            if (d->pipeline && d->pipeline->blend.enable)
               sc->planned_blended_draws++;
         }
         if (prev && prev->kind == CPVK_OP_DRAW &&
             prev->scope_index == op->scope_index) {
            d->plan_prev = &prev->draw_cmd;
            d->plan_mergeable = cpvk_draws_mergeable(d, &prev->draw_cmd);
         } else {
            d->plan_prev = NULL;
            d->plan_mergeable = false;
         }
      }
      prev = op;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_plan_batches(cmd);
   return vk_command_buffer_end(&cmd->vk);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindPipeline(VkCommandBuffer commandBuffer,
                     VkPipelineBindPoint pipelineBindPoint,
                     VkPipeline _pipeline)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_pipeline, pipeline, _pipeline);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!cpvk_cmd_retain_pipeline(cmd, pipeline)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
      cmd->graphics_pipeline = pipeline;
   else if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
      cmd->compute_pipeline = pipeline;
}

/* Keep an outgrown arena alive until the command buffer is reset. */
static bool
cpvk_arena_retire(struct cpvk_cmd_buffer *cmd, CUdeviceptr arena,
                  void *host, size_t used)
{
   if (cmd->num_desc_retired >= cmd->max_desc_retired) {
      unsigned want = cmd->max_desc_retired ? cmd->max_desc_retired * 2 : 8;
      void *p = realloc(cmd->desc_retired,
                        want * sizeof(*cmd->desc_retired));
      if (!p)
         return false;
      cmd->desc_retired = p;
      cmd->max_desc_retired = want;
   }
   unsigned i = cmd->num_desc_retired++;
   cmd->desc_retired[i].dev = arena;
   cmd->desc_retired[i].host = host;
   cmd->desc_retired[i].used = used;
   return true;
}

/*
 * Copy a descriptor set into memory this command buffer owns and return its
 * device address. The host mirror remains available for sampler specialization;
 * queue submission uploads it once before either CUDA stream can consume it.
 */
static bool
cpvk_arena_append(struct cpvk_cmd_buffer *cmd, const void *src, size_t bytes,
                  CUdeviceptr *out_addr, void **out_host)
{
   if (cmd->desc_arena_used + bytes > cmd->desc_arena_size) {
      size_t want = MAX2(cmd->desc_arena_size * 2,
                         cmd->desc_arena_used + bytes);
      want = MAX2(want, (size_t)64 * 1024);
      cp_ctx_check("cpvk_arena_append",
                   cpvk_cmd_buffer_device(cmd)->cu_ctx);
      CUdeviceptr fresh = 0;
      void *fresh_host = malloc(want);
      /* Recording only writes this mirror and submit performs a synchronous
       * HtoD copy, so page-locked memory buys no transfer overlap. */
      if (!fresh_host || cuMemAlloc(&fresh, want) != CUDA_SUCCESS) {
         free(fresh_host);
         if (fresh)
            cuMemFree(fresh);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }

      /* Counted for CUDAVK_PLAN_STATS: a pass whose descriptor volume was
       * known in advance would size this once. */
      cmd->arena_grows++;

      /* The old arena is still referenced by draws already recorded, so keep
       * both halves until reset and upload them together at submit. */
      if (cmd->desc_arena &&
          !cpvk_arena_retire(cmd, cmd->desc_arena, cmd->desc_arena_host,
                             cmd->desc_arena_used)) {
         cuMemFree(fresh);
         free(fresh_host);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }
      cmd->desc_arena = fresh;
      cmd->desc_arena_host = fresh_host;
      cmd->desc_arena_size = want;
      cmd->desc_arena_used = 0;
   }

   char *host = (char *)cmd->desc_arena_host + cmd->desc_arena_used;
   CUdeviceptr addr = cmd->desc_arena + cmd->desc_arena_used;
   if (src)
      memcpy(host, src, bytes);
   cmd->desc_arena_used += bytes;
   cmd->desc_arena_dirty = true;
   *out_addr = addr;
   if (out_host)
      *out_host = host;
   return true;
}

static CUdeviceptr
cpvk_snapshot_set(struct cpvk_cmd_buffer *cmd, struct cpvk_descriptor_set *set)
{
   if (!set->host || !set->num_descriptors) {
      fprintf(stderr, "cudavk: descriptor set has no buffer (%u descriptors, "
              "host=%p); shaders reading it will fault\n",
              set->num_descriptors,
              (void *)set->host);
      return 0;
   }

   /* Re-resolve storage rows: their image memory may have been bound after
    * the descriptor was written, which Vulkan permits. Sets without storage
    * images -- every set in both captures -- skip the walk entirely. */
   if (set->has_storage_views) {
      for (unsigned d = 0; d < set->num_descriptors; d++)
         if (set->views[d])
            cpvk_storage_image_row(&set->host[d], set->views[d]);
   }

   size_t bytes = (size_t)set->num_descriptors *
                  sizeof(struct cpvk_descriptor);
   CUdeviceptr addr = 0;
   if (!cpvk_arena_append(cmd, set->host, bytes, &addr, NULL))
      return 0;
   return addr;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindDescriptorSets2(VkCommandBuffer commandBuffer,
                            const VkBindDescriptorSetsInfo *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_pipeline_layout, layout, pInfo->layout);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   unsigned dyn = 0;
   VkShaderStageFlags graphics_stages =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
      VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
   bool bind_graphics = (pInfo->stageFlags & graphics_stages) != 0;
   bool bind_compute = (pInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
   if (!bind_graphics && !bind_compute)
      bind_graphics = bind_compute = true;

   for (uint32_t i = 0; i < pInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set, set, pInfo->pDescriptorSets[i]);
      if (!set)
         continue;

      /*
       * The set's buffer goes in the set's slot -- a copy of it, so that a
       * later rebind cannot reach back into the draws already recorded.
       * Every binding in it, buffer or texture or storage image alike, is an
       * offset from here.
       */
      unsigned slot = layout->set_slot[pInfo->firstSet + i];
      CUdeviceptr snap_addr = cpvk_snapshot_set(cmd, set);
      if (!snap_addr)
         return;
      struct cpvk_descriptor *snap;
      if (snap_addr >= cmd->desc_arena &&
          snap_addr < cmd->desc_arena + cmd->desc_arena_size)
         snap = (struct cpvk_descriptor *)
            ((char *)cmd->desc_arena_host + (snap_addr - cmd->desc_arena));
      else
         snap = (struct cpvk_descriptor *)(uintptr_t)snap_addr;
      if (slot < CPVK_MAX_ARG_BUFS) {
         if (bind_graphics)
            cmd->graphics_addrs[slot] = snap_addr;
         if (bind_compute)
            cmd->compute_addrs[slot] = snap_addr;

         /* The snapshot itself is the per-draw descriptor row. Its contents
          * need not be hashed now that batches carry distinct rows. */
      }

      /*
       * Dynamic offsets: a dynamic uniform buffer binding names one buffer
       * and the draw picks the element out of it with an offset given at
       * bind time. They are consumed in binding order over the dynamic
       * descriptors of each set, which is what the spec says and what the
       * sample relies on -- and they are written into the snapshot rather
       * than the set, because the next bind of the same set carries a
       * different offset and must not rewrite this draw's.
       */
      for (unsigned b = 0; b < set->num_bindings &&
                           dyn < pInfo->dynamicOffsetCount; b++) {
         VkDescriptorType ty = set->bindings[b].type;
         if (ty != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
             ty != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
            continue;
         for (unsigned e = 0; e < set->bindings[b].count &&
                              dyn < pInfo->dynamicOffsetCount; e++) {
            unsigned flat = set->bindings[b].flat + e;
            uint32_t off = pInfo->pDynamicOffsets[dyn++];
            if (snap && flat < set->num_descriptors && set->addrs[flat])
               snap[flat].base = set->addrs[flat] + off;
         }
      }
   }
}

static struct cpvk_op *
cpvk_op_alloc(struct cpvk_cmd_buffer *cmd, enum cpvk_op_kind kind);

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                     uint32_t baseGroupY, uint32_t baseGroupZ,
                     uint32_t groupCountX, uint32_t groupCountY,
                     uint32_t groupCountZ)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!cmd->compute_pipeline)
      return;

   /* In the op list, so that it runs where it was recorded. */
   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_DISPATCH);
   if (!op)
      return;
   struct cpvk_dispatch *d = &op->dispatch;
   d->pipeline = cmd->compute_pipeline;
   d->grid[0] = groupCountX;
   d->grid[1] = groupCountY;
   d->grid[2] = groupCountZ;
   memcpy(d->addrs, cmd->compute_addrs, sizeof(d->addrs));
   memcpy(d->push, cmd->compute_push, sizeof(d->push));
   d->push_size = cmd->compute_push_size;
}

/*
 * vkCmdDispatchIndirect. vk_common_CmdDispatchIndirect forwards to
 * CmdDispatchIndirect2KHR, which this driver does not implement, so the common
 * path is a valid pointer onto a null one. The grid is three words in device
 * memory and is read at submit; see cpvk_execute_dispatch().
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                         VkDeviceSize offset)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, buffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!cmd->compute_pipeline || !buf || !buf->mem)
      return;

   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_DISPATCH);
   if (!op)
      return;
   struct cpvk_dispatch *d = &op->dispatch;
   d->pipeline = cmd->compute_pipeline;
   d->grid[0] = d->grid[1] = d->grid[2] = 0;
   d->indirect = buf->mem->dev_ptr + buf->offset + offset;
   memcpy(d->addrs, cmd->compute_addrs, sizeof(d->addrs));
   memcpy(d->push, cmd->compute_push, sizeof(d->push));
   d->push_size = cmd->compute_push_size;
}

/* ------------------------------------------------------------- execution */

VkResult
cpvk_execute_dispatch(struct cpvk_device *dev,
                       const struct cpvk_dispatch *d)
{
   /*
    * An indirect dispatch reads its grid from device memory, which whatever
    * ran before it in this submission may have written, so the stream has to
    * have reached this point before the three words mean anything. A host
    * wait per vkCmdDispatchIndirect, and the same trade the indirect draws
    * make.
    */
   struct cpvk_dispatch resolved;
   if (d->indirect) {
      cp_batch_flush_why(&dev->renderer, "Vulkan order point");
      cp_pass_finish(&dev->renderer);
      if (cuStreamSynchronize(dev->renderer.stream) != CUDA_SUCCESS)
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      resolved = *d;
      resolved.indirect = 0;
      if (cuMemcpyDtoH(resolved.grid, d->indirect, sizeof(resolved.grid)) !=
          CUDA_SUCCESS)
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      if (!resolved.grid[0] || !resolved.grid[1] || !resolved.grid[2])
         return VK_SUCCESS;
      d = &resolved;
   }

   struct cp_shader_binary *bin = d->pipeline->bin;
   struct cp_shader_exec *exec = bin
      ? &bin->exec[CP_SHADER_EXEC_CLASSIC] : NULL;
   if (!exec || !exec->kernel)
      return VK_SUCCESS;

   /* A linked sampler reads its state table through module globals. Graphics
    * publishes both globals before shading; compute has no quads, but must
    * publish the same table and explicitly leave derivatives disabled. Do not
    * query an untextured module: cuModuleGetGlobal triggers its lazy JIT and
    * moved a one-second cost into the first measured compute frame. */
   if (exec->sampler_ptx) {
      if (!exec->globals_resolved) {
         CUdeviceptr sym;
         size_t sym_size;
         if (cuModuleGetGlobal(&sym, &sym_size, exec->module,
                               "cp_sampler_table") == CUDA_SUCCESS)
            exec->sym_sampler_table = sym;
         if (cuModuleGetGlobal(&sym, &sym_size, exec->module,
                               "cp_quad_derivs") == CUDA_SUCCESS)
            exec->sym_quad_derivs = sym;
         exec->last_sampler_table = ~(uint64_t)0;
         exec->last_quad_derivs = -1;
         exec->globals_resolved = true;
      }
      if (dev->renderer.sampler_table && exec->sym_sampler_table &&
          exec->last_sampler_table != (uint64_t)dev->renderer.sampler_table) {
         uint64_t addr = (uint64_t)dev->renderer.sampler_table;
         if (cuMemcpyHtoD(exec->sym_sampler_table, &addr, sizeof(addr)) !=
             CUDA_SUCCESS)
            return vk_error(dev, VK_ERROR_DEVICE_LOST);
         exec->last_sampler_table = addr;
      }
      if (exec->sym_quad_derivs && exec->last_quad_derivs != 0) {
         int off = 0;
         if (cuMemcpyHtoD(exec->sym_quad_derivs, &off, sizeof(off)) !=
             CUDA_SUCCESS)
            return vk_error(dev, VK_ERROR_DEVICE_LOST);
         exec->last_quad_derivs = 0;
      }
   }

   /* A dispatch is an operation in the same Vulkan queue as graphics.  Flush
    * anything recorded before it, then launch on the renderer stream so CUDA
    * stream order implements that queue order in both directions. */
   struct cp_context *cp = &dev->renderer;
   cp_batch_flush(cp);

   /* One device-only upload block: the argument pointer array, its grid, and
    * the optional push constants.  Managed allocations made the first warp
    * fault these pages back from the host on every dispatch, then forced a
    * stream drain merely so they could be freed. */
   const size_t args_bytes = CPVK_ARG_SLOTS * sizeof(void *);
   const size_t grid_bytes = 3 * sizeof(uint32_t);
   const size_t push_off = ALIGN_POT(args_bytes + grid_bytes, 16);
   const size_t total = push_off + d->push_size;
   void *host = NULL;
   CUdeviceptr block = cp_upload_begin(cp, total, &host);
   if (!block || !host)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   memset(host, 0, total);
   void **slots = host;
   slots[0] = (void *)(uintptr_t)(block + args_bytes);
   for (unsigned s = 0; s < CPVK_MAX_ARG_BUFS; s++)
      slots[CPVK_ARG_UBO_BASE + s] = (void *)(uintptr_t)
         (d->addrs[s] ? d->addrs[s] : dev->null_desc);

   if (d->push_size) {
      memcpy((char *)host + push_off, d->push, d->push_size);
      slots[CPVK_ARG_UBO_BASE + CPVK_UBO_PUSH_SLOT] =
         (void *)(uintptr_t)(block + push_off);
   }

   uint32_t *grid = (uint32_t *)((char *)host + args_bytes);
   grid[0] = d->grid[0];
   grid[1] = d->grid[1];
   grid[2] = d->grid[2];
   cp_upload_end(cp, block, host, total);

   void *kernel_args[] = { &block };
   unsigned bx = MAX2(d->pipeline->local_size[0], (uint16_t)1);
   unsigned by = MAX2(d->pipeline->local_size[1], (uint16_t)1);
   unsigned bz = MAX2(d->pipeline->local_size[2], (uint16_t)1);
   CUresult err = cp_launch(cp, exec->kernel, d->grid[0], d->grid[1],
                            d->grid[2], bx, by, bz, 0, cp->stream,
                            kernel_args, NULL);
   if (err != CUDA_SUCCESS) {
      /* Name it. A submit that returns DEVICE_LOST and says nothing else
       * is indistinguishable from every other way a replay can stop. */
      const char *name = NULL;
      cuGetErrorName(err, &name);
      fprintf(stderr, "cudavk: compute dispatch %ux%ux%u failed: %s (%d)\n",
              d->grid[0], d->grid[1], d->grid[2], name ? name : "?", err);
      fprintf(stderr, "cudavk:   pipeline %p, push=%u bytes, buffer slots:",
              (void *)d->pipeline, d->push_size);
      for (unsigned s = 0; s < CPVK_MAX_ARG_BUFS; s++)
         if (d->addrs[s])
            fprintf(stderr, " [%u]=%p", s, (void *)(uintptr_t)d->addrs[s]);
      fprintf(stderr, "\n");
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }

   return VK_SUCCESS;
}

/*
 * Secondary command buffers, replayed into the primary.
 *
 * A secondary records the same ops a primary does, so executing one is
 * appending its list. The alternative -- keeping a reference and walking into
 * it at submit -- would have to answer what happens when the secondary is
 * reset before the primary is submitted, and the answer would be a
 * use-after-free.
 */
struct cpvk_arena_map {
   CUdeviceptr src, dst;
   size_t bytes;
};

static CUdeviceptr
cpvk_remap_secondary_addr(CUdeviceptr addr, const struct cpvk_arena_map *maps,
                          unsigned count)
{
   for (unsigned i = 0; i < count; i++)
      if (addr >= maps[i].src && addr < maps[i].src + maps[i].bytes)
         return maps[i].dst + (addr - maps[i].src);
   return addr;
}

static bool
cpvk_import_secondary_arenas(struct cpvk_cmd_buffer *dst,
                             const struct cpvk_cmd_buffer *src,
                             struct cpvk_arena_map **out_maps,
                             unsigned *out_count)
{
   unsigned capacity = src->num_desc_retired + (src->desc_arena_used ? 1 : 0);
   struct cpvk_arena_map *maps = capacity ? calloc(capacity, sizeof(*maps)) : NULL;
   if (capacity && !maps) {
      vk_command_buffer_set_error(&dst->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   unsigned count = 0;
   for (unsigned i = 0; i < src->num_desc_retired; i++) {
      if (!src->desc_retired[i].used)
         continue;
      CUdeviceptr imported = 0;
      if (!cpvk_arena_append(dst, src->desc_retired[i].host,
                             src->desc_retired[i].used, &imported, NULL)) {
         free(maps);
         return false;
      }
      maps[count++] = (struct cpvk_arena_map) {
         .src = src->desc_retired[i].dev,
         .dst = imported,
         .bytes = src->desc_retired[i].used,
      };
   }
   if (src->desc_arena_used) {
      CUdeviceptr imported = 0;
      if (!cpvk_arena_append(dst, src->desc_arena_host,
                             src->desc_arena_used, &imported, NULL)) {
         free(maps);
         return false;
      }
      maps[count++] = (struct cpvk_arena_map) {
         .src = src->desc_arena,
         .dst = imported,
         .bytes = src->desc_arena_used,
      };
   }

   *out_maps = maps;
   *out_count = count;
   return true;
}

static void
cpvk_remap_secondary_op(struct cpvk_cmd_buffer *cmd, struct cpvk_op *op,
                        const struct cpvk_arena_map *maps, unsigned count,
                        uint32_t scope_base, uint32_t inherited_scope)
{
   CUdeviceptr *addrs = NULL;
   if (op->kind == CPVK_OP_DRAW)
      addrs = op->draw_cmd.addrs;
   else if (op->kind == CPVK_OP_DISPATCH)
      addrs = op->dispatch.addrs;
   if (addrs) {
      for (unsigned i = 0; i < CPVK_MAX_ARG_BUFS; i++)
         addrs[i] = cpvk_remap_secondary_addr(addrs[i], maps, count);
   }

   if (op->kind != CPVK_OP_BEGIN_RENDER &&
       op->kind != CPVK_OP_END_RENDER && op->kind != CPVK_OP_DRAW)
      return;

   uint32_t scope = op->scope_index;
   if (scope == CP_RENDER_SCOPE_INHERITED)
      scope = inherited_scope;
   else if (scope != CP_RENDER_SCOPE_NONE)
      scope += scope_base;
   op->scope_index = scope;
   if (op->kind == CPVK_OP_DRAW)
      op->draw_cmd.scope_index = scope;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t count,
                        const VkCommandBuffer *pCommandBuffers)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(cpvk_cmd_buffer, sec, pCommandBuffers[i]);
      if (!sec || !sec->num_ops)
         continue;

      for (unsigned p = 0; p < sec->num_retained_pipelines; p++) {
         if (!cpvk_cmd_retain_pipeline(cmd, sec->retained_pipelines[p])) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return;
         }
      }
      for (unsigned q = 0; q < sec->num_retained_queries; q++) {
         if (!cpvk_cmd_retain_query(cmd, sec->retained_queries[q])) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return;
         }
      }
      for (unsigned e = 0; e < sec->num_retained_events; e++) {
         if (!cpvk_cmd_retain_event(cmd, sec->retained_events[e])) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return;
         }
      }

      uint32_t scope_base = cmd->num_scopes;
      if (cmd->num_scopes + sec->num_scopes > cmd->max_scopes) {
         unsigned want = MAX2(cmd->max_scopes ? cmd->max_scopes * 2 : 8,
                              cmd->num_scopes + sec->num_scopes);
         void *p = realloc(cmd->scopes, want * sizeof(*cmd->scopes));
         if (!p) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return;
         }
         cmd->scopes = p;
         cmd->max_scopes = want;
      }
      for (unsigned s = 0; s < sec->num_scopes; s++) {
         cmd->scopes[cmd->num_scopes] = sec->scopes[s];
         cmd->scopes[cmd->num_scopes].serial = cmd->next_scope_serial++;
         cmd->num_scopes++;
      }

      struct cpvk_arena_map *maps = NULL;
      unsigned num_maps = 0;
      if (!cpvk_import_secondary_arenas(cmd, sec, &maps, &num_maps))
         return;

      if (cmd->num_ops + sec->num_ops > cmd->max_ops) {
         unsigned want = MAX2(cmd->max_ops ? cmd->max_ops * 2 : 64,
                              cmd->num_ops + sec->num_ops);
         struct cpvk_op *ops = realloc(cmd->ops, want * sizeof(*ops));
         if (!ops) {
            free(maps);
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return;
         }
         cmd->ops = ops;
         cmd->max_ops = want;
      }

      unsigned first = cmd->num_ops;
      memcpy(cmd->ops + first, sec->ops,
             sec->num_ops * sizeof(*sec->ops));
      cmd->num_ops += sec->num_ops;
      for (unsigned o = first; o < cmd->num_ops; o++)
         cpvk_remap_secondary_op(cmd, &cmd->ops[o], maps, num_maps,
                                 scope_base, cmd->active_scope);
      free(maps);
   }
}

/* Room for one more operation, or NULL if it cannot be had. */
static struct cpvk_op *
cpvk_op_alloc(struct cpvk_cmd_buffer *cmd, enum cpvk_op_kind kind)
{
   if (cmd->num_ops >= cmd->max_ops) {
      unsigned want = cmd->max_ops ? cmd->max_ops * 2 : 64;
      struct cpvk_op *ops = realloc(cmd->ops, want * sizeof(*ops));
      if (!ops) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
      cmd->ops = ops;
      cmd->max_ops = want;
   }

   struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
   memset(op, 0, sizeof(*op));
   op->kind = kind;
   return op;
}

/* ------------------------------------------------------- rendering + draw */

static uint32_t
cpvk_scope_append(struct cpvk_cmd_buffer *cmd, const struct cp_fb_desc *fb,
                  const struct cp_depth_attachment *depth, unsigned samples)
{
   if (cmd->num_scopes == cmd->max_scopes) {
      unsigned want = cmd->max_scopes ? cmd->max_scopes * 2 : 8;
      void *p = realloc(cmd->scopes, want * sizeof(*cmd->scopes));
      if (!p) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return CP_RENDER_SCOPE_NONE;
      }
      cmd->scopes = p;
      cmd->max_scopes = want;
   }
   uint32_t index = cmd->num_scopes++;
   cmd->scopes[index] = (struct cp_render_scope) {
      .fb = *fb,
      .depth = *depth,
      .attachment_samples = MAX2(samples, 1u),
      .serial = cmd->next_scope_serial++,
   };
   return index;
}

static uint64_t cpvk_image_end(const struct cpvk_image *img);

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                       const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   const VkRenderingAttachmentInfo *stencil =
      pRenderingInfo->pStencilAttachment;
   bool unsupported_stencil = stencil && stencil->imageView &&
      (!pRenderingInfo->pDepthAttachment ||
       pRenderingInfo->pDepthAttachment->imageView != stencil->imageView ||
       stencil->resolveMode != (pRenderingInfo->pDepthAttachment
                                ? pRenderingInfo->pDepthAttachment->resolveMode
                                : VK_RESOLVE_MODE_NONE) ||
       false);
   if (pRenderingInfo->colorAttachmentCount > 1 ||
       pRenderingInfo->layerCount > 1 || pRenderingInfo->viewMask != 0 ||
       unsupported_stencil) {
      fprintf(stderr, "cudavk: unsupported rendering colors=%u layers=%u "
              "viewMask=%u stencil=%d\n", pRenderingInfo->colorAttachmentCount,
              pRenderingInfo->layerCount, pRenderingInfo->viewMask,
              unsupported_stencil);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   struct cp_fb_desc fb = {
      .width = pRenderingInfo->renderArea.offset.x +
               pRenderingInfo->renderArea.extent.width,
      .height = pRenderingInfo->renderArea.offset.y +
                pRenderingInfo->renderArea.extent.height,
      .nr_cbufs = pRenderingInfo->colorAttachmentCount,
      .color_encoding = -1,
   };

   const VkRenderingAttachmentInfo *cat =
      pRenderingInfo->colorAttachmentCount ?
      &pRenderingInfo->pColorAttachments[0] : NULL;
   const VkRenderingAttachmentInfo *dat = pRenderingInfo->pDepthAttachment;
   struct cpvk_image *cimg = NULL, *dimg = NULL;
   struct cpvk_image_view *cview = NULL;
   unsigned color_level = 0;
   enum pipe_format color_format = PIPE_FORMAT_NONE;
   struct cp_depth_attachment depth = {0};

   if (cat && cat->imageView) {
      VK_FROM_HANDLE(cpvk_image_view, view, cat->imageView);
      if (view && view->image && view->image->mem) {
         cview = view;
         cimg = view->image;
         /*
          * The view's subresource, not the image's base. A cube map is
          * rendered one face at a time through a view whose baseArrayLayer
          * selects the face, and a mip chain likewise through baseMipLevel.
          * Ignoring both meant every face of pbribl's environment cube was
          * rendered over face zero, so its spheres reflected nothing.
          */
         color_level = MIN2(view->vk.base_mip_level,
                            CPVK_MAX_MIP_LEVELS - 1);
         fb.width = u_minify(cimg->vk.extent.width, color_level);
         fb.height = u_minify(cimg->vk.extent.height, color_level);
         fb.color = (void *)(uintptr_t)(cimg->mem->dev_ptr + cimg->offset +
                                        cimg->level_offset[color_level] +
                                        (uint64_t)view->vk.base_array_layer *
                                        cimg->level_size[color_level]);
         /*
          * The view's format decides the encoding, not the image's. They are
          * usually the same and were assumed to be; when they are not, the
          * difference is a channel order, and the sample suite's triangle came
          * out with red and blue exchanged and every pixel otherwise exact.
          */
         const struct cpvk_format_info *vf = cpvk_format_info(view->vk.format);
         color_format = vk_format_to_pipe_format(view->vk.format);
         fb.color_encoding = vf ? vf->color : cimg->color;
         /*
          * A format the writeback kernel cannot encode is refused here, in
          * the same shape as the unsupported depth format below, because
          * the alternative is not "no picture" but a wrong one. Every draw
          * path resolves the encoding as MAX2(fb.color_encoding, 0)
          * (cp_renderer.c:4129, :4560, :8465, :8935) and 0 is
          * CP_COLOR_R8G8B8A8_UNORM, so a -1 attachment was rendered as if
          * it were RGBA8: measured on VK_FORMAT_R16G16_UNORM, a
          * LOAD_OP_CLEAR packed the real format correctly and the draw then
          * wrote the bytes ff 00 00 ff over it, which reads back as
          * R16=255 G16=65280. No error was returned at any point.
          *
          * vkCreateImage stays permissive on purpose (see cpvk_image.c), and
          * vkGetPhysicalDeviceImageFormatProperties2 already refuses this
          * usage, so this is the first place a client that never asked can
          * be told -- and it is the last place before the pixels are wrong.
          */
         if (fb.color_encoding < 0) {
            fprintf(stderr, "cudavk: unsupported colour attachment format %u "
                    "(no writeback encoding)\n", view->vk.format);
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         /* How far apart the samples are, which the renderer needs in order
          * to write them and the resolve needs in order to find them. */
         fb.color_sample_stride = (unsigned)cimg->sample_stride;
         /* Trusted internal mutation token, not a descriptor cookie. Its lifetime
          * follows Vulkan command-resource lifetime rules. */
         fb.texture_cookie = (uint64_t)(uintptr_t)view->image;

         if (cp_debug->debug_rt)
            fprintf(stderr, "rt: %ux%u layer=%u level=%u layers=%u img=%ux%u "
                    "fmt=%u base=%p\n", fb.width, fb.height,
                    view->vk.base_array_layer, view->vk.base_mip_level,
                    pRenderingInfo->layerCount, cimg->vk.extent.width,
                    cimg->vk.extent.height, view->vk.format, fb.color);
      }
   }

   if (dat && dat->imageView) {
      /* Sample zero is the one mode advertised, and it is a copy of the first
       * sample plane: it needs no averaging kernel and gives the packed
       * depth-stencil aspects the same answer. */
      if (dat->resolveMode != VK_RESOLVE_MODE_NONE &&
          (dat->resolveMode != VK_RESOLVE_MODE_SAMPLE_ZERO_BIT ||
           dat->resolveImageView == VK_NULL_HANDLE)) {
         fprintf(stderr, "cudavk: unsupported depth resolve %u\n",
                 dat->resolveMode);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      VK_FROM_HANDLE(cpvk_image_view, view, dat->imageView);
      if (!view || !view->image || !view->image->mem ||
          (view->vk.format != VK_FORMAT_D32_SFLOAT &&
           view->vk.format != VK_FORMAT_D32_SFLOAT_S8_UINT &&
           view->vk.format != VK_FORMAT_D24_UNORM_S8_UINT &&
           view->vk.format != VK_FORMAT_D16_UNORM)) {
         fprintf(stderr, "cudavk: unsupported depth attachment format %u\n",
                 view ? view->vk.format : 0);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      dimg = view->image;
      unsigned level = MIN2(view->vk.base_mip_level,
                            CPVK_MAX_MIP_LEVELS - 1);
      if (!cimg) {
         fb.width = u_minify(dimg->vk.extent.width, level);
         fb.height = u_minify(dimg->vk.extent.height, level);
      }
      bool store = dat->storeOp == VK_ATTACHMENT_STORE_OP_STORE;
      bool full_area = pRenderingInfo->renderArea.offset.x == 0 &&
                       pRenderingInfo->renderArea.offset.y == 0 &&
                       pRenderingInfo->renderArea.extent.width ==
                          u_minify(dimg->vk.extent.width, level) &&
                       pRenderingInfo->renderArea.extent.height ==
                          u_minify(dimg->vk.extent.height, level);
      depth = (struct cp_depth_attachment) {
         .data = dimg->mem->dev_ptr + dimg->offset +
                 dimg->level_offset[level] +
                 (uint64_t)view->vk.base_array_layer * dimg->level_size[level],
         .row_stride = dimg->row_stride[level],
         .sample_stride = dimg->sample_stride,
         .pixel_stride = util_format_get_blocksize(
            vk_format_to_pipe_format(view->vk.format)),
         /* The packing the load and store kernels switch on, not the
          * VkFormat: 0 = D32_SFLOAT, 1 = D32_SFLOAT_S8_UINT,
          * 2 = D24_UNORM_S8_UINT, 3 = D16_UNORM. */
         .format = view->vk.format == VK_FORMAT_D24_UNORM_S8_UINT ? 2 :
                   view->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT ? 1 :
                   view->vk.format == VK_FORMAT_D16_UNORM ? 3 : 0,
         .load = dat->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ||
                 (store && !full_area),
         /* A resolve reads the attachment image, so its contents must be
          * written back even when the application discards them. */
         .store = store || dat->resolveMode != VK_RESOLVE_MODE_NONE,
         /* A stencil clear on the shared aspect is honoured by the store,
          * which is the only place this driver writes stencil bits. Without
          * a store the aspect's contents are undefined anyway. */
         .stencil_clear = stencil && stencil->imageView &&
                          stencil->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR,
         .stencil_value = stencil ?
            stencil->clearValue.depthStencil.stencil : 0,
      };
      fb.has_zs = true;

      if (dat->resolveMode == VK_RESOLVE_MODE_SAMPLE_ZERO_BIT) {
         VK_FROM_HANDLE(cpvk_image_view, rview, dat->resolveImageView);
         struct cpvk_image *rimg = rview ? rview->image : NULL;
         unsigned rlevel = rview ? MIN2(rview->vk.base_mip_level,
                                        CPVK_MAX_MIP_LEVELS - 1) : 0;
         if (!rimg || !rimg->mem || rview->vk.format != view->vk.format ||
             rimg->vk.samples != VK_SAMPLE_COUNT_1_BIT) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         unsigned bpp = depth.pixel_stride;
         uint64_t offset_y = (uint64_t)pRenderingInfo->renderArea.offset.y;
         uint64_t offset_x = (uint64_t)pRenderingInfo->renderArea.offset.x;
         cmd->depth_resolve = (struct cpvk_copy) {
            .dst_image = rimg,
            .src = depth.data + offset_y * depth.row_stride + offset_x * bpp,
            .dst = rimg->mem->dev_ptr + rimg->offset +
                   rimg->level_offset[rlevel] +
                   (uint64_t)rview->vk.base_array_layer *
                      rimg->level_size[rlevel] +
                   offset_y * rimg->row_stride[rlevel] + offset_x * bpp,
            .src_pitch = depth.row_stride,
            .dst_pitch = rimg->row_stride[rlevel],
            .width_bytes =
               (size_t)pRenderingInfo->renderArea.extent.width * bpp,
            .rows = pRenderingInfo->renderArea.extent.height,
            .src_end = cpvk_image_end(dimg),
            .dst_end = cpvk_image_end(rimg),
            .samples = 1,
            .encoding = -1,
            .dst_encoding = -1,
         };
         cmd->depth_resolve_valid = true;
      }
   }



   if (pRenderingInfo->renderArea.offset.x < 0 ||
       pRenderingInfo->renderArea.offset.y < 0 ||
       (uint64_t)pRenderingInfo->renderArea.offset.x +
          pRenderingInfo->renderArea.extent.width > fb.width ||
       (uint64_t)pRenderingInfo->renderArea.offset.y +
          pRenderingInfo->renderArea.extent.height > fb.height) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }


   if (cimg && dimg && cimg->vk.samples != dimg->vk.samples) {
      fprintf(stderr, "cudavk: attachment sample mismatch %u/%u\n",
              cimg->vk.samples, dimg->vk.samples);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   cmd->fb = fb;
   cmd->has_fb = true;

   /* What a vkCmdClearAttachments in this pass will clear, resolved once
    * here: the same subresource LOAD_OP_CLEAR below writes. */
   cmd->clear_color_image = cimg;
   cmd->clear_color_stride = cimg ? cimg->row_stride[color_level] : 0;
   cmd->clear_color_format = color_format;
   cmd->clear_area = pRenderingInfo->renderArea;

   cmd->resolve_valid = false;
   if (cat && cview && cat->resolveImageView != VK_NULL_HANDLE &&
       cat->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(cpvk_image_view, rview, cat->resolveImageView);
      struct cpvk_image *rimg = rview ? rview->image : NULL;
      unsigned rlevel = rview ? MIN2(rview->vk.base_mip_level,
                                     CPVK_MAX_MIP_LEVELS - 1) : 0;
      enum pipe_format rformat = rview
         ? vk_format_to_pipe_format(rview->vk.format) : PIPE_FORMAT_NONE;
      unsigned bpp = util_format_get_blocksize(color_format);
      unsigned rbpp = util_format_get_blocksize(rformat);
      const struct cpvk_format_info *rf = rview
         ? cpvk_format_info(rview->vk.format) : NULL;
      if (cat->resolveMode != VK_RESOLVE_MODE_AVERAGE_BIT ||
          !rimg || !rimg->mem || cimg->vk.samples <= 1 ||
          rimg->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
          bpp != 4 || rbpp != bpp || !rf ||
          rf->color != fb.color_encoding ||
          (uint64_t)pRenderingInfo->renderArea.offset.x +
             pRenderingInfo->renderArea.extent.width >
             u_minify(rimg->vk.extent.width, rlevel) ||
          (uint64_t)pRenderingInfo->renderArea.offset.y +
             pRenderingInfo->renderArea.extent.height >
             u_minify(rimg->vk.extent.height, rlevel)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      uint64_t src_offset =
         (uint64_t)pRenderingInfo->renderArea.offset.y *
            cimg->row_stride[color_level] +
         (uint64_t)pRenderingInfo->renderArea.offset.x * bpp;
      uint64_t dst_offset = rimg->level_offset[rlevel] +
         (uint64_t)rview->vk.base_array_layer * rimg->level_size[rlevel] +
         (uint64_t)pRenderingInfo->renderArea.offset.y *
            rimg->row_stride[rlevel] +
         (uint64_t)pRenderingInfo->renderArea.offset.x * bpp;
      cmd->resolve = (struct cpvk_copy) {
         .dst_image = rimg,
         .src = (CUdeviceptr)(uintptr_t)fb.color + src_offset,
         .dst = rimg->mem->dev_ptr + rimg->offset + dst_offset,
         .src_pitch = cimg->row_stride[color_level],
         .dst_pitch = rimg->row_stride[rlevel],
         .width_bytes =
            (size_t)pRenderingInfo->renderArea.extent.width * bpp,
         .rows = pRenderingInfo->renderArea.extent.height,
         .src_end = cpvk_image_end(cimg),
         .dst_end = cpvk_image_end(rimg),
         .samples = MAX2(cimg->vk.samples, 1u),
         .sample_stride = cimg->sample_stride,
         .encoding = fb.color_encoding,
         .dst_encoding = rf->color,
      };
      cmd->resolve_valid = true;
   }

   /*
    * Bind the framebuffer here, as its own op, and not at the first draw.
    *
    * cp_context_set_framebuffer is what allocates the renderer's depth
    * buffer, and cp_clear_depthbuf clears whatever is allocated now. Doing
    * the bind at draw time meant the depth clear ran against the previous
    * framebuffer's buffer and the new one arrived uncleared, so every
    * fragment failed the depth test: renderheadless rendered its clear colour
    * and all three triangles vanished, with the rasterizer reporting them
    * shaded.
    */
   {
      /* The attachment's sample count comes from the image rather than the
       * pipeline because framebuffer-sized buffers follow the attachment. */
      cmd->fb_samples = cimg ? MAX2(cimg->vk.samples, 1u) :
                        dimg ? MAX2(dimg->vk.samples, 1u) : 1;
      uint32_t scope = cpvk_scope_append(cmd, &fb, &depth, cmd->fb_samples);
      if (scope == CP_RENDER_SCOPE_NONE)
         return;
      cmd->active_scope = scope;

      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_BEGIN_RENDER);
      if (!op)
         return;
      op->scope_index = scope;
   }

   /* LOAD_OP_CLEAR, recorded in order with the draws that follow it. */
   if (cat && cimg && cat->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      unsigned bpp = util_format_get_blocksize(color_format);
      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
      if (!op)
         return;
      op->clear = (struct cpvk_clear) {
         .image = cimg,
         .data = fb.color,
         .offset =
            (uint64_t)pRenderingInfo->renderArea.offset.y *
               cimg->row_stride[color_level] +
            (uint64_t)pRenderingInfo->renderArea.offset.x * bpp,
         .width = pRenderingInfo->renderArea.extent.width,
         .height = pRenderingInfo->renderArea.extent.height,
         .stride = cimg->row_stride[color_level],
         .pixel_size = bpp,
         .samples = cmd->fb_samples,
         .sample_stride = fb.color_sample_stride,
      };
      util_format_pack_rgba(color_format, op->clear.value,
                            cat->clearValue.color.float32, 1);
   }

   if (dat && dat->imageView && dat->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
      if (!op)
         return;
      op->clear = (struct cpvk_clear) {
         .depth = true,
         .depth_value = dat->clearValue.depthStencil.depth,
         .offset = ((uint64_t)pRenderingInfo->renderArea.offset.y * fb.width +
                    pRenderingInfo->renderArea.offset.x) * sizeof(uint32_t),
         .width = pRenderingInfo->renderArea.extent.width,
         .height = pRenderingInfo->renderArea.extent.height,
         .stride = fb.width * sizeof(uint32_t),
         .pixel_size = sizeof(uint32_t),
         .samples = cmd->fb_samples,
         .sample_stride = (uint64_t)fb.width * fb.height * sizeof(uint32_t),
      };
   }
}


VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                           uint32_t bindingCount, const VkBuffer *pBuffers,
                           const VkDeviceSize *pOffsets,
                           const VkDeviceSize *pSizes,
                           const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   for (uint32_t i = 0; i < bindingCount; i++) {
      unsigned b = firstBinding + i;
      if (b >= 16)
         continue;
      VK_FROM_HANDLE(cpvk_buffer, buf, pBuffers[i]);
      cmd->vb_base[b] = (buf && buf->mem)
         ? buf->mem->dev_ptr + buf->offset + pOffsets[i] : 0;
      cmd->num_vb = MAX2(cmd->num_vb, b + 1);
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPushConstants2(VkCommandBuffer commandBuffer,
                       const VkPushConstantsInfo *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (pInfo->offset + pInfo->size > CPVK_MAX_PUSH_BYTES)
      return;

   unsigned end = pInfo->offset + pInfo->size;
   if (pInfo->stageFlags & VK_SHADER_STAGE_VERTEX_BIT) {
      memcpy(cmd->vs_push + pInfo->offset, pInfo->pValues, pInfo->size);
      cmd->vs_push_size = MAX2(cmd->vs_push_size, end);
   }
   if (pInfo->stageFlags & VK_SHADER_STAGE_FRAGMENT_BIT) {
      memcpy(cmd->fs_push + pInfo->offset, pInfo->pValues, pInfo->size);
      cmd->fs_push_size = MAX2(cmd->fs_push_size, end);
   }
   if (pInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      memcpy(cmd->compute_push + pInfo->offset, pInfo->pValues, pInfo->size);
      cmd->compute_push_size = MAX2(cmd->compute_push_size, end);
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                      VkShaderStageFlags stageFlags, uint32_t offset,
                      uint32_t size, const void *pValues)
{
   VkPushConstantsInfo info = {
      .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
      .layout = layout, .stageFlags = stageFlags,
      .offset = offset, .size = size, .pValues = pValues,
   };
   cpvk_CmdPushConstants2(commandBuffer, &info);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindIndexBuffer2(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                         VkDeviceSize offset, VkDeviceSize size,
                         VkIndexType indexType)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, _buffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   cmd->index_ptr = (buf && buf->mem)
      ? (const void *)(uintptr_t)(buf->mem->dev_ptr + buf->offset + offset)
      : NULL;
   cmd->index_size = indexType == VK_INDEX_TYPE_UINT16 ? 2 :
                     indexType == VK_INDEX_TYPE_UINT8 ? 1 : 4;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
                        VkDeviceSize offset, VkIndexType indexType)
{
   cpvk_CmdBindIndexBuffer2(commandBuffer, buffer, offset, VK_WHOLE_SIZE,
                            indexType);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetViewportWithCount(VkCommandBuffer commandBuffer, uint32_t count,
                             const VkViewport *pViewports)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!count)
      return;
   /* Vulkan's clip space already has y running downward, which is why the
    * scale is not flipped again here; see cp_cull_mode's note on winding. */
   cmd->viewport.scale[0] = pViewports[0].width * 0.5f;
   cmd->viewport.scale[1] = pViewports[0].height * 0.5f;
   cmd->viewport.scale[2] = pViewports[0].maxDepth - pViewports[0].minDepth;
   cmd->viewport.translate[0] = pViewports[0].x + pViewports[0].width * 0.5f;
   cmd->viewport.translate[1] = pViewports[0].y + pViewports[0].height * 0.5f;
   cmd->viewport.translate[2] = pViewports[0].minDepth;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport,
                    uint32_t count, const VkViewport *pViewports)
{
   if (firstViewport == 0)
      cpvk_CmdSetViewportWithCount(commandBuffer, count, pViewports);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetScissorWithCount(VkCommandBuffer commandBuffer, uint32_t count,
                            const VkRect2D *pScissors)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!count)
      return;
   cmd->scissor = (struct cp_rect) {
      .minx = pScissors[0].offset.x,
      .miny = pScissors[0].offset.y,
      .maxx = pScissors[0].offset.x + pScissors[0].extent.width,
      .maxy = pScissors[0].offset.y + pScissors[0].extent.height,
   };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor,
                   uint32_t count, const VkRect2D *pScissors)
{
   if (firstScissor == 0)
      cpvk_CmdSetScissorWithCount(commandBuffer, count, pScissors);
}

/* One recorded draw, shared by vkCmdDraw and vkCmdDrawIndexed. */
static void
cpvk_record_draw_cmd(struct cpvk_cmd_buffer *cmd, unsigned count, unsigned first,
                 unsigned instance_count, unsigned first_instance,
                 int vertex_offset, bool indexed)
{
   if (!cmd->graphics_pipeline)
      return;

   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_DRAW);
   if (!op)
      return;
   struct cpvk_draw_cmd *d = &op->draw_cmd;

   op->scope_index = cmd->active_scope;
   d->pipeline = cmd->graphics_pipeline;
   d->scope_index = cmd->active_scope;
   d->viewport = cmd->graphics_pipeline->static_viewport
      ? cmd->graphics_pipeline->viewport : cmd->viewport;
   d->scissor = cmd->graphics_pipeline->static_scissor
      ? cmd->graphics_pipeline->scissor : cmd->scissor;
   d->range = (struct cp_draw_range) {
      .start = first,
      .count = count,
      .index_bias = vertex_offset,
   };
   d->call = (struct cp_draw_call) {
      .mode = cmd->graphics_pipeline->topology,
      .instance_count = MAX2(instance_count, 1u),
      .start_instance = first_instance,
      .index_size = indexed ? cmd->index_size : 0,
      .index_ptr = indexed ? cmd->index_ptr : NULL,
   };
   memcpy(d->vb_base, cmd->vb_base, sizeof(d->vb_base));
   d->num_vb = cmd->num_vb;
   memcpy(d->addrs, cmd->graphics_addrs, sizeof(d->addrs));
   memcpy(d->vs_push, cmd->vs_push, sizeof(d->vs_push));
   memcpy(d->fs_push, cmd->fs_push, sizeof(d->fs_push));
   d->vs_push_size = cmd->vs_push_size;
   d->fs_push_size = cmd->fs_push_size;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
             uint32_t instanceCount, uint32_t firstVertex,
             uint32_t firstInstance)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_draw_cmd(cmd, vertexCount, firstVertex, instanceCount,
                    firstInstance, 0, false);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                    uint32_t instanceCount, uint32_t firstIndex,
                    int32_t vertexOffset, uint32_t firstInstance)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_draw_cmd(cmd, indexCount, firstIndex, instanceCount, firstInstance,
                    vertexOffset, true);
}

/*
 * vkCmdDrawIndirect and vkCmdDrawIndexedIndirect.
 *
 * Implemented here for the reason vkCmdFillBuffer is: vk_common_CmdDrawIndirect
 * forwards to CmdDrawIndirect2KHR, which this driver does not implement, so the
 * common path is a valid pointer that jumps through a null one -- worse than a
 * NULL entry point, because vkGetDeviceProcAddr answers as if it worked.
 * maxDrawIndirectCount was already advertised as UINT32_MAX.
 *
 * The parameters are read at submit rather than here, because device memory is
 * where they are and a dispatch recorded earlier in the same command buffer is
 * a normal way to produce them. What is recorded is an ordinary draw with the
 * state this one would have had, plus where to find its counts.
 */
static void
cpvk_record_draw_indirect(struct cpvk_cmd_buffer *cmd, struct cpvk_buffer *buf,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride, bool indexed)
{
   if (!buf || !buf->mem || !drawCount)
      return;

   cpvk_record_draw_cmd(cmd, 0, 0, 1, 0, 0, indexed);
   if (!cmd->num_ops || cmd->ops[cmd->num_ops - 1].kind != CPVK_OP_DRAW)
      return;   /* no pipeline bound, or the op array could not grow */

   struct cpvk_draw_cmd *d = &cmd->ops[cmd->num_ops - 1].draw_cmd;
   d->indirect = buf->mem->dev_ptr + buf->offset + offset;
   d->indirect_draws = drawCount;
   /* A stride of zero is legal when there is one draw, and the array is
    * tightly packed by definition when the application leaves it out. */
   d->indirect_stride = stride ? stride :
      (indexed ? sizeof(VkDrawIndexedIndirectCommand)
               : sizeof(VkDrawIndirectCommand));
   d->indirect_indexed = indexed;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                     VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, buffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_draw_indirect(cmd, buf, offset, drawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                            VkDeviceSize offset, uint32_t drawCount,
                            uint32_t stride)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, buffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_draw_indirect(cmd, buf, offset, drawCount, stride, true);
}

/* A recorded clear, at submit. */
bool
cpvk_execute_clear(struct cpvk_device *dev, const struct cpvk_clear *c)
{
   struct cp_context *cp = &dev->renderer;

   /* So does a clear. */
   cp_batch_flush(cp);

   if (c->depth) {
      /*
       * A mid-pass clear covers a rectangle inside a buffer whose rest may
       * never have been written: this pass may have loaded nothing and cleared
       * nothing, in which case the first draw would run the lazy full clear
       * and wipe what was cleared here. Run that clear first instead. The
       * contents it invents are the ones the lazy path would have invented,
       * because a depth buffer nothing loaded or cleared is undefined.
       */
      if (c->mid_pass && !cp->depthbuf_cleared)
         cp_clear_depthbuf(cp, 1.0f);

      uint32_t value[4] = { cp_depth_to_sortable(c->depth_value) };
      bool ok = true;
      for (unsigned s = 0; s < MAX2(c->samples, 1u); s++)
         ok &= cp_clear_rect(cp,
                       (void *)(uintptr_t)(cp->depthbuf +
                                           s * c->sample_stride),
                       c->offset, c->width, c->height, c->stride,
                       c->pixel_size, value, true);
      /* A mid-pass clear says nothing about the pixels outside its rectangle,
       * so it may only confirm the flag, never set it. */
      if (!c->mid_pass)
         cp->depthbuf_cleared = ok;
      return ok;
   }

   /* Once per sample plane. */
   bool ok = true;
   for (unsigned s = 0; s < MAX2(c->samples, 1u); s++) {
      char *plane = (char *)c->data + s * c->sample_stride;
      ok &= c->masked
         ? cp_clear_rect_masked(cp, plane, c->offset, c->width, c->height,
                                c->stride, c->pixel_size, c->value, c->mask)
         : cp_clear_rect(cp, plane, c->offset, c->width, c->height, c->stride,
                         c->pixel_size, c->value, false);
   }
   return ok;
}

/* ------------------------------------------------------------- batching */

/*
 * The pipeline state two draws must share to merge, in the form this front
 * end has it.
 *
 * A VkPipeline is immutable and holds the rasterizer, depth, blend, vertex
 * layout and topology, so its address covers all of them at once -- which is
 * the shape cp_draw_types.h anticipated for this driver. Only what a command
 * buffer can change without a new pipeline goes beside it.
 */
struct cpvk_batch_state {
   const void *pipeline;
   struct cp_viewport_state viewport;
   unsigned fb_samples;
   /*
    * The fragment stage's bindings. The renderer carries them per draw and
    * says so, so in principle they need not be a merge condition -- but
    * taking them out moved pushconstants from 5.098 to 7.336 and bought
    * nothing measurable (23.32 ms against 24.01 on Crossroads), so they stay
    * until something explains that.
    */
   uint64_t fs_ubos[CP_MAX_CONST_BUFFERS];
};

static const struct cp_batch_state_field cpvk_batch_state_field_table[] = {
   { "pipeline",   offsetof(struct cpvk_batch_state, pipeline),
                   sizeof(((struct cpvk_batch_state *)0)->pipeline) },
   { "viewport",   offsetof(struct cpvk_batch_state, viewport),
                   sizeof(((struct cpvk_batch_state *)0)->viewport) },
   { "fb_samples", offsetof(struct cpvk_batch_state, fb_samples),
                   sizeof(((struct cpvk_batch_state *)0)->fb_samples) },
   { "fs_ubos",    offsetof(struct cpvk_batch_state, fs_ubos),
                   sizeof(((struct cpvk_batch_state *)0)->fs_ubos) },
};

const struct cp_batch_state_field *
cp_batch_state_fields(unsigned *count)
{
   *count = ARRAY_SIZE(cpvk_batch_state_field_table);
   return cpvk_batch_state_field_table;
}

static_assert(sizeof(struct cpvk_batch_state) <= CP_BATCH_STATE_BYTES,
              "the batch key's state blob is too small for this front end");

/*
 * Everything that has to match for two draws to merge, filled from the
 * context the draw was just staged into. Compared with memcmp and nothing
 * else, so a field left out is simply not a merge condition -- which is why
 * this fills a zeroed struct and copies whole values rather than deriving
 * any of them.
 */
static void
cpvk_build_batch_key(struct cpvk_device *dev, const struct cpvk_draw_cmd *d,
                     const struct cp_draw_packet *packet,
                     struct cp_batch_key *key, bool blended)
{
   struct cp_context *cp = &dev->renderer;

   memset(key, 0, sizeof(*key));

   key->vs = packet->state.vs;
   key->fs = packet->state.fs;

   /* The colour target's identity. Natively an image view names it, and its
    * memory is what the kernels write, so both go in. */
   key->cbuf_texture = packet->scope->fb.color;
   key->color_data = packet->scope->fb.color;
   key->zs_texture = packet->scope->fb.has_zs ? (const void *)(uintptr_t)1 : NULL;
   key->visbuf = cp->visbuf;
   key->depthbuf = cp->depthbuf;
   key->fb_w = packet->scope->fb.width;
   key->fb_h = packet->scope->fb.height;
   key->fb_nr_cbufs = packet->scope->fb.nr_cbufs;
   key->fb_samples = packet->scope->attachment_samples;
   key->cbuf_format = (uint32_t)packet->scope->fb.color_encoding;

   key->mode = d->call.mode;
   key->index_size = d->call.index_size;
   key->start_instance = d->call.start_instance;
   key->index_resource = d->call.index_ptr;

   key->scissor = d->scissor;
   key->blend_enabled = blended;

   struct cpvk_batch_state state;
   memset(&state, 0, sizeof(state));
   state.pipeline = d->pipeline;
   state.viewport = d->viewport;
   state.fb_samples = packet->scope->attachment_samples;
   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++)
      state.fs_ubos[i] = packet->state.fs_ubos[i];
   memcpy(key->state, &state, sizeof(state));

   key->num_vertex_elements = packet->state.num_vertex_elements;
   key->vertex_stride = packet->state.vertex_stride;
   key->num_vertex_buffers = packet->state.num_vertex_buffers;
   key->num_fs_ubos = packet->state.num_fs_ubos;
   key->num_vs_ubos = packet->state.num_vs_ubos;
   key->sampler_table = cp->sampler_table;
   key->num_samplers = cp->num_samplers;
}

/*
 * Whether this draw may be held back at all: a property of the draw and the
 * state bound for it, never of what came before. Mirrors cp_batch_structural
 * in the Gallium adapter, which is the worked example.
 */
static bool
cpvk_batch_structural(struct cpvk_device *dev, const struct cp_render_scope *scope,
                      const struct cpvk_draw_cmd *d)
{
   struct cp_context *cp = &dev->renderer;
   const struct cpvk_pipeline *p = d->pipeline;

   if (!p || !p->vs ||
       !p->vs->exec[CP_SHADER_EXEC_CLASSIC].kernel ||
       !cp_shader_has_standalone_exec(p->fs))
      return false;
   if (d->call.mode != MESA_PRIM_TRIANGLES)
      return false;
   if (d->call.index_size && !d->call.index_ptr)
      return false;

   /* A batch replays vertex buffers; a shader building positions from
    * gl_VertexIndex has nothing to gain and stays on the single-draw path. */
   if (!d->num_vb || !d->vb_base[0])
      return false;

   if (!scope->fb.nr_cbufs || !scope->fb.color || !cp->visbuf || !cp->depthbuf)
      return false;

   uint64_t tris = (uint64_t)cp_triangles_for_draw(d->call.mode,
                                                   d->range.count) *
                   MAX2(d->call.instance_count, 1u);
   if (tris == 0 || tris > CP_MAX_BATCH_TRIS)
      return false;

   return true;
}

static bool cpvk_batch_eligible(struct cpvk_device *dev,
                                const struct cp_render_scope *scope,
                                const struct cpvk_draw_cmd *d, bool *blended);

/*
 * Whether this draw can join whatever is pending -- answered without touching
 * the context, because the answer decides whether the context may be written.
 * The key is built from the draw and its pipeline; the few context fields it
 * needs (the framebuffer's buffers, the sampler table) belong to the pass and
 * are already current.
 */
/*
 * Whether two draws may be merged, decided by comparing the draws themselves.
 *
 * Everything a draw carries that must match is here, and everything a batch is
 * allowed to differ in -- the index range, the vertex-stage bindings, the
 * per-draw scissor -- is deliberately absent, exactly as cp_batch_key
 * documents for the Gallium side. The comparison touches no driver state at
 * all, which is what makes it safe to run before the draw is staged.
 */
/*
 * Which key component broke a batch, tallied by label for
 * CUDAVK_PLAN_STATS. The labels are string literals, so pointer identity
 * is the key. This is the attribution the merge-rate number lacks: the old
 * capture merges 71% of its draws where Crossroads merges 84%, and the
 * difference is some specific field of this comparison.
 */
static struct { const char *what; uint64_t n; } cpvk_break_tally[32];

static void
cpvk_batch_break_note(const char *what)
{
   if (!cp_debug->plan_stats)
      return;
   for (unsigned i = 0; i < 32; i++) {
      if (cpvk_break_tally[i].what == what || !cpvk_break_tally[i].what) {
         cpvk_break_tally[i].what = what;
         cpvk_break_tally[i].n++;
         return;
      }
   }
}

void
cpvk_batch_break_report(void)
{
   if (!cp_debug->plan_stats || !cpvk_break_tally[0].what)
      return;
   fprintf(stderr, "cudavk: batch breaks by cause:\n");
   for (unsigned i = 0; i < 32 && cpvk_break_tally[i].what; i++)
      fprintf(stderr, "  %8" PRIu64 "  %s\n",
              cpvk_break_tally[i].n, cpvk_break_tally[i].what);
}

static bool
cpvk_draws_mergeable(const struct cpvk_draw_cmd *a, const struct cpvk_draw_cmd *b)
{
#define CPVK_DIFF(cond, what)                                   \
   do {                                                         \
      if (cond) {                                               \
         if (cp_debug->debug_batchdiff)                          \
            fprintf(stderr, "batchdiff: %s\n", what);           \
         cpvk_batch_break_note(what);                            \
         return false;                                          \
      }                                                         \
   } while (0)

   /*
    * The pipeline's state, not its identity. gltfscenerendering builds one
    * pipeline per material and draws them back to back; comparing addresses
    * made every batch a single draw, where the Gallium adapter compares the
    * state itself and merges them.
    */
   if (a->pipeline != b->pipeline) {
      const struct cpvk_pipeline *pa = a->pipeline, *pb = b->pipeline;
      CPVK_DIFF(!pa || !pb, "pipeline");
      CPVK_DIFF(pa->vs != pb->vs, "vertex shader");
      CPVK_DIFF(pa->fs != pb->fs, "fragment shader");
      CPVK_DIFF(memcmp(&pa->raster, &pb->raster, sizeof(pa->raster)), "rasterizer");
      CPVK_DIFF(memcmp(&pa->depth, &pb->depth, sizeof(pa->depth)), "depth state");
      CPVK_DIFF(memcmp(&pa->blend, &pb->blend, sizeof(pa->blend)), "blend state");
      CPVK_DIFF(pa->topology != pb->topology, "topology");
      CPVK_DIFF(pa->num_velem != pb->num_velem ||
                pa->vertex_stride != pb->vertex_stride ||
                memcmp(pa->velem, pb->velem,
                       pa->num_velem * sizeof(pa->velem[0])), "vertex layout");
      CPVK_DIFF(pa->samples != pb->samples, "sample count");
   }
   CPVK_DIFF(memcmp(&a->viewport, &b->viewport, sizeof(a->viewport)), "viewport");
   /*
    * The scissor need not match when the renderer will carry one rectangle
    * per draw. cp_draw_execute does that for a batch of more than one whose
    * primitives resolve to their draw -- its `rows_stable` condition, which
    * blending satisfies -- and says so: "draws that disagree on it merge".
    * The front end was stricter than the renderer, and 3,339 of the
    * Crossroads capture's blended refusals were this.
    */
   if (cp_debug->no_merge_scissor || !a->pipeline || !b->pipeline ||
       !a->pipeline->blend.enable || !b->pipeline->blend.enable)
      CPVK_DIFF(memcmp(&a->scissor, &b->scissor, sizeof(a->scissor)),
                "scissor");
   /* An indirect draw's counts are not known here, so it can neither be
    * merged into a batch nor have one merged into it: the batch key would be
    * built from placeholders. It is resolved into ordinary draws at submit,
    * and those merge normally with each other. */
   CPVK_DIFF(a->indirect || b->indirect, "indirect draw");
   CPVK_DIFF(a->call.mode != b->call.mode, "topology");
   CPVK_DIFF(a->call.index_size != b->call.index_size, "index size");
   /*
    * The index buffer is *not* a merge condition. The slice table carries a
    * per-draw index-buffer base (cp_draw_slice.ib_base_*), and the fetch
    * rebases before applying index_bytes, so draws bound to different index
    * buffers merge. Refs never read a batch's index buffer on the host:
    * batching requires MESA_PRIM_TRIANGLES with a classic-exec VS, which is
    * the skip_refs condition. 23,276 separations on the occlusion capture.
    * CUDAVK_KEEP_IBKEY restores it.
    */
   if (cp_debug->keep_ibkey)
      CPVK_DIFF(a->call.index_ptr != b->call.index_ptr, "index buffer");
   CPVK_DIFF(a->call.start_instance != b->call.start_instance, "start instance");
   /*
    * The vertex offset, until the batched path stops taking it from the first
    * draw. cp_draw_execute builds its vertex fetch with
    * `first_vertex = draws[0].index_bias` for the whole batch, so draws whose
    * vertexOffset differs must not merge -- which is what gltfscenerendering's
    * per-primitive offsets are.
    */
   /*
    * The vertex offset. `cp_draw_execute` builds a single draw's fetch with
    * `first_vertex = draws[0].index_bias`, but a *batch* builds a slice table
    * and sets `slices[d].first_vertex = draws[d].index_bias` per draw, so the
    * batched path already honours it and this condition costs merges for
    * nothing.
    *
    * It is the single largest merge blocker in the Crossroads capture: 13,281
    * separations of 38,155, ahead of the fragment shader's 12,068.
    * CUDAVK_KEEP_VOFF restores it.
    */
   if (cp_debug->keep_voff)
      CPVK_DIFF(a->range.index_bias != b->range.index_bias, "vertex offset");
   /*
    * The instance count is *not* a merge condition. The renderer's own
    * cp_batch_key omits it and cp_draw_execute is handed an instance_counts[]
    * row per draw, which the vertex fetch reads -- "slices[d].verts_per_instance
    * = inst > 1 ? dverts : 0" -- so a batch of draws with different instance
    * counts is what that table exists for.
    *
    * It was the second largest refusal in the Crossroads capture once the
    * vertex-offset condition stopped masking it: 9,168 of 32,668.
    */
   if (cp_debug->keep_instkey)
      CPVK_DIFF(a->call.instance_count != b->call.instance_count, "instance count");
   /* Descriptor bindings deliberately differ inside a batch.  The renderer's
    * per-draw UBO rows point at the immutable snapshots owned by this command
    * buffer; requiring equal sets only split otherwise identical work. */
   CPVK_DIFF(a->num_vb != b->num_vb, "vertex buffer count");
   /*
    * The vertex-buffer bindings are *not* a merge condition. The batch
    * snapshots one row of per-element base addresses per merged draw
    * (cp_batch_record_packet), the upload at cp_renderer.c's elem_bases_dev
    * hands them to the fetch, and cp_vf_lane.h reads its element base
    * per row -- "so draws bound to different vertex buffers merge", in that
    * file's own words. Every batched draw has a classic-exec vertex shader
    * (cpvk_batch_structural), so the no-VS passthrough path that still reads
    * the batch-wide state->vb_base cannot see a merged batch.
    *
    * It was the largest merge blocker on the occlusion capture: 144,008
    * separations of 332,027, ahead of the vertex shader's 95,448.
    * CUDAVK_KEEP_VBKEY restores it.
    */
   if (cp_debug->keep_vbkey)
      CPVK_DIFF(memcmp(a->vb_base, b->vb_base, sizeof(a->vb_base)), "vertex buffers");
   CPVK_DIFF(a->vs_push_size != b->vs_push_size ||
             a->fs_push_size != b->fs_push_size, "push constant size");
   /*
    * The push block is *not* in the key. While it was, it was what stopped
    * multithreading and pushconstants batching: 446.8 draws a frame against
    * the Gallium driver's 21.9, and 56.7 against 9.2.
    *
    * It does not have to be. The block is bound as UBO slot
    * CPVK_UBO_PUSH_SLOT, each staged draw uploads its own, and the batch
    * snapshots the whole uniform row per draw. cpvk_batchpush merges eleven
    * draws with twelve distinct push blocks and renders byte-identical to
    * that driver, with and without a descriptor UBO beside them.
    *
    * CUDAVK_KEEP_PUSHKEY puts it back. Those two samples still render wrong
    * when they merge, for a reason the test does not yet reproduce -- see
    * CUDAVK_VK_NATIVE.md. The switch is here so the next attempt can bisect
    * from a passing three-draw case toward the failing sample rather than the
    * other way round.
    */
   if (cp_debug->keep_pushkey)
      CPVK_DIFF(memcmp(a->vs_push, b->vs_push, a->vs_push_size) ||
                memcmp(a->fs_push, b->fs_push, a->fs_push_size),
                "push constants");
   return true;
#undef CPVK_DIFF
}

/* State the renderer reads once for a whole pass episode.  Shader,
 * descriptor, vertex-layout, vertex-buffer, push-constant, scissor and draw
 * changes are captured per segment; these are not. */
static bool
cpvk_draws_episode_compatible(const struct cp_render_scope *sa,
                              const struct cpvk_draw_cmd *a,
                              const struct cp_render_scope *sb,
                              const struct cpvk_draw_cmd *b)
{
   const struct cpvk_pipeline *pa = a->pipeline, *pb = b->pipeline;
   if (!pa || !pb)
      return false;

   return sa && sb && sa->serial == sb->serial &&
          !memcmp(&a->viewport, &b->viewport, sizeof(a->viewport)) &&
          !memcmp(&pa->raster, &pb->raster, sizeof(pa->raster)) &&
          !memcmp(&pa->depth, &pb->depth, sizeof(pa->depth)) &&
          !memcmp(&pa->blend, &pb->blend, sizeof(pa->blend)) &&
          pa->samples == pb->samples;
}

static bool
cpvk_batch_can_join(struct cpvk_device *dev,
                    const struct cp_render_scope *scope,
                    const struct cpvk_draw_cmd *d, bool *out_blended)
{
   struct cp_context *cp = &dev->renderer;
   bool blended = false;

   if (!cpvk_batch_eligible(dev, scope, d, &blended)) {
      if (cp->batch.pending)
         cpvk_batch_break_note("draw not batchable");
      return false;
   }

   /* Out to the caller, which stages it on the batch: it decides at flush
    * whether the batch appends to a blended pass episode or an opaque one. */
   *out_blended = blended;

   unsigned tris = cp_triangles_for_draw(d->call.mode, d->range.count) *
                   MAX2(d->call.instance_count, 1u);

   if (!cp->batch.pending)
      return true;

   if (!dev->prev_draw_valid || !dev->prev_draw || !dev->prev_scope) {
      cpvk_batch_break_note("no previous draw");
      return false;
   }
   if (scope->serial != dev->prev_scope->serial) {
      cpvk_batch_break_note("render scope changed");
      return false;
   }
   /* The plan bit is this exact comparison, precomputed at
    * vkEndCommandBuffer against this exact draw. */
   bool mergeable;
   if (d->plan_prev == dev->prev_draw) {
      cp->plan.plan_hits++;
      mergeable = d->plan_mergeable;
   } else {
      cp->plan.plan_misses++;
      mergeable = cpvk_draws_mergeable(d, dev->prev_draw);
   }
   if (!mergeable) {
      if (cp_debug->debug_batchdiff)
         fprintf(stderr, "batchdiff: the draws differ\n");
      return false;
   }
   if (cp->batch.ndraws >= (unsigned)cp_debug->batch_max) {
      cpvk_batch_break_note("batch full");
      return false;
   }
   if (cp->batch.tris + tris > CP_MAX_BATCH_TRIS) {
      cpvk_batch_break_note("triangle cap");
      return false;
   }

   return true;
}

static bool
cpvk_pipeline_order_free(const struct cpvk_pipeline *p)
{
   if (!p || p->blend.enable || p->fs->uses_discard)
      return false;
   /*
    * CUDAVK_UNSAFE_FORCE_OPAQUE has to relax the depth conditions too, or it
    * measures the wrong thing. A transparent draw is drawn with depth writes
    * OFF, so clearing `blend.enable` alone leaves it failing this test, still
    * classified blended by cpvk_batch_eligible, still routed into
    * cp_pass_append -- where the segment backs out (cp_renderer.c, the
    * `!abuf` arm) because nothing is blended any more and the A-buffer is
    * never set up. It then re-executes classically and pays the vertex stage
    * twice. That is a back-out penalty, not the price of an opaque world.
    * Under this flag the draw becomes genuinely order-free so it reaches an
    * opaque episode. It is a diagnostic; the pixels are wrong either way.
    */
   if (cp_debug->unsafe_force_opaque)
      return true;
   if (!p->depth.depth_enabled || !p->depth.depth_writemask)
      return false;
   switch (p->depth.depth_func) {
   case CP_FUNC_LESS:
   case CP_FUNC_LEQUAL:
   case CP_FUNC_GREATER:
   case CP_FUNC_GEQUAL:
      return true;
   default:
      return false;
   }
}

static bool
cpvk_batch_eligible(struct cpvk_device *dev,
                    const struct cp_render_scope *scope,
                    const struct cpvk_draw_cmd *d, bool *blended)
{
   /*
    * Off, and measured rather than abandoned.
    *
    * Batching is worth a great deal here: with it the Crossroads capture
    * replays at 5.63 ms against 24.28 without, and the old capture at 12.71
    * against 78.90 -- both faster than the Gallium-hosted driver's 7.13 and
    * 25.20. The machinery is the renderer's and already linked in.
    *
    * But this front end's key is not yet a complete statement of what has to
    * match. With it on, computeshader goes from 4.427 to 83.474 against the
    * Gallium driver, bloom from 17.386 to 26.071 and instancing from 11.851
    * to 14.467, while particlesystem improves from 41.036 to 6.785. Adding
    * the fragment stage's bindings to the state blob fixed multithreading
    * (0.641 to 0.012) and moved none of the others, and the blended path is
    * not implicated: disabling it changes nothing, so the fault is in the
    * opaque half.
    *
    * Something a draw carries and this key does not is still varying inside a
    * batch. Finding it is the remaining work, and CUDAVK_DEBUG_BATCHDIFF
    * against the Gallium driver's key on the same sample is the way in.
    */
   /*
    * Off. It is correct now -- every sample matches the unbatched result --
    * and it is worth nothing measurable, because almost nothing merges: the
    * key holds the pipeline's *address*, so two pipelines with identical
    * state never merge, where the Gallium adapter compares the state itself
    * and merges them. Keying on the resolved state instead is the work that
    * would make this pay, and until then an always-false batcher is honest
    * about what it does.
    */
   /*
    * On by default.
    *
    * It was off because it was measured worth nothing: almost nothing merged,
    * and the note here said so. What it was actually blocked on was the
    * instance count, which this front end made a merge condition and the
    * renderer's own cp_batch_key does not -- a batch is handed an
    * instance_counts[] row per draw. With that removed the same capture
    * replays at 8.81 ms against 9.45 and the heavier one at 31.2 against 34.5,
    * two runs each way, and 17/18 samples stay pixel-correct.
    */
   if (cp_debug->no_batch)
      return false;
   if (!cpvk_batch_structural(dev, scope, d))
      return false;

   if (cpvk_pipeline_order_free(d->pipeline)) {
      *blended = false;
      return true;
   }
   /*
    * The blended half, on by default.
    *
    * A blended draw merges through the A-buffer, where the order fragments
    * composite in is decided per pixel rather than by submission order, so
    * merging is sound; `cpvk_batchblend` shows nine blended draws over nine
    * textures at nine depths byte-identical to their unbatched output.
    *
    * It is also where the time is. Measured on the Crossroads capture with a
    * clean environment: batching alone 23.49 ms, batching with this 8.79 ms,
    * neither 24.50. A blended draw that does not join a batch never reaches an
    * A-buffer pass episode, and without episodes every one of them builds,
    * sorts and peels its own A-buffer.
    *
    * CUDAVK_NO_BATCH_BLEND turns it off.
    */
   if (!cp_debug->no_batch_blend) {
      *blended = true;
      return true;
   }
   return false;
}

static void
cpvk_prepare_draw(struct cpvk_device *dev, const struct cp_render_scope *scope,
                  const struct cpvk_draw_cmd *d, struct cp_draw_packet *packet)
{
   struct cp_context *cp = &dev->renderer;
   struct cpvk_pipeline *p = d->pipeline;
   memset(packet, 0, sizeof(*packet));
   packet->scope = scope;
   packet->state.vs = p->vs;
   packet->state.fs = p->fs;
   packet->state.viewport = d->viewport;
   packet->state.raster = p->raster;
   packet->state.depth = p->depth;
   packet->state.blend = p->blend;
   packet->state.pipeline_samples = p->samples;
   memcpy(packet->state.velem, p->velem, sizeof(packet->state.velem));
   memcpy(packet->state.vb_base, d->vb_base, sizeof(packet->state.vb_base));
   packet->state.num_vertex_buffers = d->num_vb;
   packet->state.num_vertex_elements = p->num_velem;
   packet->state.vertex_stride = p->vertex_stride;
   packet->state.num_vs_ubos = CP_MAX_CONST_BUFFERS;
   packet->state.num_fs_ubos = CP_MAX_CONST_BUFFERS;
   packet->call = d->call;
   packet->range = d->range;
   packet->scissor = d->scissor;

   CUdeviceptr vs_push_dev = 0, fs_push_dev = 0;
   if (d->vs_push_size) {
      void *host = NULL;
      vs_push_dev = cp_upload_begin(cp, d->vs_push_size, &host);
      if (vs_push_dev) {
         memcpy(host, d->vs_push, d->vs_push_size);
         cp_upload_end(cp, vs_push_dev, host, d->vs_push_size);
      }
   }
   /* One upload when both stages see the same bytes, which is what an
    * application pushing to ALL_GRAPHICS produces and what every measured
    * workload here does. */
   if (d->fs_push_size == d->vs_push_size &&
       !memcmp(d->fs_push, d->vs_push, d->fs_push_size)) {
      fs_push_dev = vs_push_dev;
   } else if (d->fs_push_size) {
      void *host = NULL;
      fs_push_dev = cp_upload_begin(cp, d->fs_push_size, &host);
      if (fs_push_dev) {
         memcpy(host, d->fs_push, d->fs_push_size);
         cp_upload_end(cp, fs_push_dev, host, d->fs_push_size);
      }
   }

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++) {
      uint64_t addr = d->addrs[i] ? d->addrs[i] : dev->null_desc;
      packet->state.vs_ubos[i] = addr;
      packet->state.fs_ubos[i] = addr;
   }
   packet->state.vs_ubos[CPVK_UBO_PUSH_SLOT] =
      vs_push_dev ? vs_push_dev : dev->null_desc;
   packet->state.fs_ubos[CPVK_UBO_PUSH_SLOT] =
      fs_push_dev ? fs_push_dev : dev->null_desc;
}

static void cpvk_execute_order_point(struct cpvk_device *dev);

/*
 * An indirect draw, resolved.
 *
 * The counts are in device memory, and what wrote them may be a dispatch or a
 * copy recorded earlier in this same command buffer, so everything issued so
 * far has to have run before they can be read. That is a host wait per
 * vkCmdDrawIndirect, on a driver that already waits about seventeen times a
 * frame; it is the price of resolving the parameters on the host, and the
 * alternative -- a device-side draw the whole pipeline could be launched from
 * -- is a different driver.
 *
 * Each resolved draw is an ordinary recorded draw with its counts filled in,
 * so batching, the vertex fetch and the index path treat it as one.
 */
static void
cpvk_execute_draw_indirect(struct cpvk_device *dev,
                           const struct cp_render_scope *scope,
                           const struct cpvk_draw_cmd *d)
{
   cpvk_execute_order_point(dev);
   if (cuStreamSynchronize(dev->renderer.stream) != CUDA_SUCCESS) {
      fprintf(stderr, "cudavk: indirect draw could not drain the stream\n");
      return;
   }

   if (!dev->indirect_draws) {
      dev->indirect_draws = calloc(CPVK_INDIRECT_DRAW_SLOTS,
                                   sizeof(*dev->indirect_draws));
      if (!dev->indirect_draws)
         return;
   }

   for (uint32_t i = 0; i < d->indirect_draws; i++) {
      /* VkDrawIndirectCommand is four words and VkDrawIndexedIndirectCommand
       * is five, with vertexOffset signed. */
      uint32_t p[5] = { 0 };
      size_t bytes = d->indirect_indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                         : sizeof(VkDrawIndirectCommand);
      if (cuMemcpyDtoH(p, d->indirect + (uint64_t)i * d->indirect_stride,
                       bytes) != CUDA_SUCCESS) {
         fprintf(stderr, "cudavk: indirect draw parameters unreadable\n");
         return;
      }

      struct cpvk_draw_cmd *r =
         &dev->indirect_draws[dev->indirect_draw_next++ %
                              CPVK_INDIRECT_DRAW_SLOTS];
      *r = *d;
      r->indirect = 0;
      r->indirect_draws = 0;
      /* The recorded plan was computed for the placeholder draw, and this
       * copy is not the op the plan named. */
      r->plan_prev = NULL;
      r->plan_mergeable = false;
      r->range.count = p[0];
      r->call.instance_count = MAX2(p[1], 1u);
      r->range.start = p[2];
      r->range.index_bias = d->indirect_indexed ? (int32_t)p[3] : 0;
      r->call.start_instance = d->indirect_indexed ? p[4] : p[3];
      if (!r->range.count)
         continue;
      cpvk_execute_draw_cmd(dev, scope, r);
   }
}

/* Run one recorded draw through the renderer. */
void
cpvk_execute_draw_cmd(struct cpvk_device *dev, const struct cp_render_scope *scope,
                      const struct cpvk_draw_cmd *d)
{
   struct cp_context *cp = &dev->renderer;

   if (d->indirect) {
      cpvk_execute_draw_indirect(dev, scope, d);
      return;
   }

   /*
    * Decide about the batch *before* staging this draw's state.
    *
    * A flush renders what is already held back, and it reads the context to
    * do it -- shaders, descriptors, vertex bindings. Staging first and
    * deciding afterwards meant every flush rendered the previous batch with
    * this draw's state. It showed up as computeshader being wrong even at
    * CUDAVK_BATCH_MAX=1, where each draw is its own batch and the result is
    * supposed to be bit-identical to not batching at all -- which is exactly
    * what that flag is for.
    */
   bool batch_blended = false;
   cp->plan.merge_tests++;
   bool batch_ok = cpvk_batch_can_join(dev, scope, d, &batch_blended);
   if (batch_ok)
      cp->plan.merges++;
   else
      cp->plan.key_breaks++;
   if (!batch_ok) {
      /*
       * A batch-key break is a per-segment change, but only while the state
       * read once for the whole episode is unchanged.  The old unconditional
       * full flush made every native episode one segment (and Crossroads
       * 4.7x slower); unconditionally deferring crossed viewport/depth/blend
       * changes and broke six samples.  Compare the episode-wide state first,
       * then let an eligible new draw begin the next segment.
       */
      bool can_defer = cp->batch.pending && dev->prev_draw_valid &&
                       dev->prev_draw &&
                       cpvk_draws_episode_compatible(scope, d, dev->prev_scope,
                                                      dev->prev_draw);
      bool eligible = false;
      if (can_defer)
         eligible = cpvk_batch_eligible(dev, scope, d, &batch_blended);

      if (can_defer && eligible)
         cp_batch_flush_defer_why(cp, "the next draw starts a segment");
      else
         cp_batch_flush_why(cp, "the next draw cannot join");

      /* With the previous batch submitted, an eligible draw starts a fresh
       * batch which can become the next segment. */
      batch_ok = cpvk_batch_can_join(dev, scope, d, &batch_blended);
   }

   dev->prev_draw = d;
   dev->prev_scope = scope;
   dev->prev_draw_valid = true;

   struct cp_draw_packet packet;
   cpvk_prepare_draw(dev, scope, d, &packet);

   /* The renderer takes draw state through explicit launch arguments and
    * immutable batch snapshots; no device-global mutable state is published. */

   /*
    * Hold the draw back if it can join the one before it. The machinery is
    * the renderer's -- cp_batch_record, the key, the flush discipline -- and
    * what this front end supplies is the key and the decision, exactly as
    * cp_draw_vbo does for Gallium.
    *
    * Measured worth: with batching off the Gallium driver replays Crossroads
    * at 27.88 ms and with it at 7.13, and this driver had no batching at all.
    */
   if (batch_ok) {
      unsigned tris = cp_triangles_for_draw(d->call.mode, d->range.count) *
                      MAX2(d->call.instance_count, 1u);
      if (!cp->batch.pending) {
         struct cp_batch_key key;
         cp->plan.key_builds++;
         cpvk_build_batch_key(dev, d, &packet, &key, batch_blended);
         cp_batch_begin_packet(cp, &packet, &key, batch_blended);
      }
      cp_batch_record_packet(cp, &packet, tris);
      return;
   }
   cp->plan.direct_draws++;

   struct cp_draw_batch direct = {
      .state = packet.state,
      .scope = *packet.scope,
      .info = packet.call,
      .ndraws = 1,
      .draws = { packet.range },
      .instance_counts = { packet.call.instance_count },
      .scissors = { packet.scissor },
   };
   cp_draw_execute_batch(cp, &direct);
}

/* ----------------------------------------------------------- transfers */

static struct cpvk_copy *
cpvk_record_copy(struct cpvk_cmd_buffer *cmd)
{
   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_COPY);
   return op ? &op->copy : NULL;
}

/*
 * vkCmdFillBuffer.
 *
 * Implemented here rather than left to the runtime: vk_common_CmdFillBuffer
 * forwards to CmdFillMemoryKHR, which this driver does not implement, so the
 * common path jumps through a null pointer. A driver that means to support
 * this has to answer the entry point itself.
 *
 * VK_WHOLE_SIZE and the round-down to a multiple of four are the spec's, and
 * both matter to a caller that poisons a buffer before reusing it: the whole
 * point is that every byte it expects to be overwritten actually is.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                   VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, dst, dstBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!dst || !dst->mem)
      return;

   if (size == VK_WHOLE_SIZE)
      size = dst->vk.size - dstOffset;
   size &= ~(VkDeviceSize)3;
   if (!size)
      return;

   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_FILL);
   if (!op)
      return;
   op->fill = (struct cpvk_fill) {
      .dst = dst->mem->dev_ptr + dst->offset + dstOffset,
      .words = size / 4,
      .value = data,
   };
}

/*
 * vkCmdUpdateBuffer: up to 65536 inline bytes, copied into the buffer where
 * the command was recorded.
 *
 * vk_common_CmdUpdateBuffer forwards to CmdUpdateMemoryKHR, unimplemented
 * here, so this is the third member of the vkCmdFillBuffer family of holes: a
 * pointer that resolves and then jumps through zero.
 *
 * The bytes belong to the caller only for the duration of the call, so they
 * are staged into device memory now and the recording keeps an ordinary copy
 * op. One allocation per call is affordable for a command with a 64 KiB cap
 * that no per-draw path uses, and it removes every question about which
 * storage a re-submitted command buffer reads.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                     VkDeviceSize dstOffset, VkDeviceSize dataSize,
                     const void *pData)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, dst, dstBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!dst || !dst->mem || !dataSize)
      return;

   if (cmd->num_inline_blocks == cmd->max_inline_blocks) {
      unsigned want = cmd->max_inline_blocks ? cmd->max_inline_blocks * 2 : 8;
      void *p = realloc(cmd->inline_blocks, want * sizeof(*cmd->inline_blocks));
      if (!p) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      cmd->inline_blocks = p;
      cmd->max_inline_blocks = want;
   }

   CUdeviceptr staging = 0;
   if (cuMemAlloc(&staging, dataSize) != CUDA_SUCCESS ||
       cuMemcpyHtoD(staging, pData, dataSize) != CUDA_SUCCESS) {
      if (staging)
         cuMemFree(staging);
      fprintf(stderr, "cudavk: vkCmdUpdateBuffer could not stage %llu bytes\n",
              (unsigned long long)dataSize);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   cmd->inline_blocks[cmd->num_inline_blocks++] = staging;

   struct cpvk_copy *c = cpvk_record_copy(cmd);
   if (!c)
      return;
   *c = (struct cpvk_copy) {
      .src = staging,
      .dst = dst->mem->dev_ptr + dst->offset + dstOffset,
      .width_bytes = dataSize,
      .rows = 1,
      .src_end = staging + dataSize,
      .dst_end = dst->mem->dev_ptr + dst->offset + dst->vk.size,
   };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyBuffer2(VkCommandBuffer commandBuffer,
                    const VkCopyBufferInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, src, pInfo->srcBuffer);
   VK_FROM_HANDLE(cpvk_buffer, dst, pInfo->dstBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!src || !dst || !src->mem || !dst->mem)
      return;

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferCopy2 *r = &pInfo->pRegions[i];
      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      *c = (struct cpvk_copy) {
         .src = src->mem->dev_ptr + src->offset + r->srcOffset,
         .dst = dst->mem->dev_ptr + dst->offset + r->dstOffset,
         .width_bytes = r->size,
         .rows = 1,
         .src_end = src->mem->dev_ptr + src->offset + src->vk.size,
         .dst_end = dst->mem->dev_ptr + dst->offset + dst->vk.size,
      };
   }
}

/* The byte offset of one level of an image, and its row pitch. */
static bool
cpvk_image_plane(const struct cpvk_image *img, unsigned level,
                 CUdeviceptr *base, size_t *pitch, unsigned *bpp)
{
   if (!img || !img->mem || level >= CPVK_MAX_MIP_LEVELS)
      return false;
   *base = img->mem->dev_ptr + img->offset + img->level_offset[level];
   *pitch = img->row_stride[level];
   *bpp = util_format_get_blocksize(vk_format_to_pipe_format(img->vk.format));
   return true;
}

/* One past the last byte an image's allocation covers. */
static uint64_t
cpvk_image_end(const struct cpvk_image *img)
{
   return (img && img->mem) ? img->mem->dev_ptr + img->offset + img->size : 0;
}

/* ---------------------------------------------------------------- clears */

/*
 * vkCmdClearColorImage, vkCmdClearDepthStencilImage and vkCmdClearAttachments.
 *
 * Implemented here for the same reason vkCmdFillBuffer is: nothing else
 * implements them. There is no vk_common_CmdClearAttachments,
 * vk_common_CmdClearColorImage or vk_common_CmdClearDepthStencilImage anywhere
 * in src/vulkan, so the three dispatch slots were NULL, vkGetDeviceProcAddr
 * returned NULL for all three, and an application that called one jumped to
 * address zero -- three segfaults with no driver output at all, which is how
 * an audit of this driver found them. They are core Vulkan 1.0 with no feature
 * or extension gate, so "not implemented" was never a legal answer.
 *
 * All three record a CPVK_OP_CLEAR and let the submit loop replay it in
 * sequence, which is the whole point: a clear recorded after a draw must not
 * run before it. An immediate blit here would be that bug. It also gets the
 * texture-cache invalidation right, because the submit loop already does that
 * for any CPVK_OP_CLEAR that names an image.
 */

static uint32_t
cpvk_f32_bits(float f)
{
   uint32_t u;
   memcpy(&u, &f, sizeof(u));
   return u;
}

/*
 * One clear per mip level of a subresource range.
 *
 * A level's rows are contiguous and so are the array layers behind them
 * (cpvk_image_layout: level_size = row_stride * h * d, layers stride by
 * level_size), so one rectangle of `h * d * layers` rows covers every layer of
 * a level at once. Samples are planes and the executor walks them.
 */
static bool
cpvk_record_image_clear(struct cpvk_cmd_buffer *cmd, struct cpvk_image *img,
                        const VkImageSubresourceRange *range,
                        unsigned pixel_size, const uint32_t value[4],
                        const uint32_t mask[4])
{
   const uint32_t levels = vk_image_subresource_level_count(&img->vk, range);
   const uint32_t layers = vk_image_subresource_layer_count(&img->vk, range);
   enum pipe_format pfmt = vk_format_to_pipe_format(img->vk.format);

   for (uint32_t l = 0; l < levels; l++) {
      unsigned level = range->baseMipLevel + l;
      if (level >= CPVK_MAX_MIP_LEVELS || level >= img->vk.mip_levels)
         return false;

      unsigned w = util_format_get_nblocksx(
         pfmt, u_minify(img->vk.extent.width, level));
      unsigned h = util_format_get_nblocksy(
         pfmt, u_minify(img->vk.extent.height, level));
      unsigned d = u_minify(img->vk.extent.depth, level);
      uint64_t rows = (uint64_t)h * d * layers;
      if (!w || !rows)
         continue;

      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
      if (!op)
         return false;
      op->clear = (struct cpvk_clear) {
         .image = img,
         .data = (void *)(uintptr_t)(img->mem->dev_ptr + img->offset +
                                     img->level_offset[level] +
                                     (uint64_t)range->baseArrayLayer *
                                        img->level_size[level]),
         .width = w,
         .height = (unsigned)rows,
         .stride = img->row_stride[level],
         .pixel_size = pixel_size,
         .samples = MAX2(img->vk.samples, 1u),
         .sample_stride = img->sample_stride,
         .masked = mask != NULL,
      };
      memcpy(op->clear.value, value, sizeof(op->clear.value));
      if (mask)
         memcpy(op->clear.mask, mask, sizeof(op->clear.mask));
   }
   return true;
}

static void
cpvk_clear_refuse(struct cpvk_cmd_buffer *cmd, const char *what)
{
   fprintf(stderr, "cudavk: %s\n", what);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image,
                        VkImageLayout imageLayout,
                        const VkClearColorValue *pColor,
                        uint32_t rangeCount,
                        const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, img, image);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!img || !img->mem) {
      cpvk_clear_refuse(cmd, "vkCmdClearColorImage on an image with no memory");
      return;
   }

   enum pipe_format pfmt = vk_format_to_pipe_format(img->vk.format);
   unsigned bpp = util_format_get_blocksize(pfmt);
   /* The clear kernel indexes on the element size and has no default arm, so a
    * size it does not name would write nothing at all -- refuse instead. That
    * excludes the three-component formats R8G8B8, R16G16B16 and R32G32B32. */
   if (util_format_is_compressed(pfmt) ||
       util_format_is_depth_or_stencil(pfmt) ||
       (bpp != 1 && bpp != 2 && bpp != 4 && bpp != 8 && bpp != 16)) {
      cpvk_clear_refuse(cmd, "vkCmdClearColorImage: unsupported format");
      return;
   }

   /*
    * The float/int/uint union is decided by the format, not by the caller:
    * util_format_pack_rgba reinterprets the same bytes as uint32, int32 or
    * float according to whether the format is pure-integer. This is the same
    * call LOAD_OP_CLEAR makes, so the two agree by construction.
    */
   uint32_t value[4] = { 0 };
   util_format_pack_rgba(pfmt, value, pColor->float32, 1);

   for (uint32_t i = 0; i < rangeCount; i++) {
      if (pRanges[i].aspectMask != VK_IMAGE_ASPECT_COLOR_BIT) {
         cpvk_clear_refuse(cmd, "vkCmdClearColorImage: non-colour aspect");
         return;
      }
      if (!cpvk_record_image_clear(cmd, img, &pRanges[i], bpp, value, NULL)) {
         cpvk_clear_refuse(cmd, "vkCmdClearColorImage: subresource out of range");
         return;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image,
                               VkImageLayout imageLayout,
                               const VkClearDepthStencilValue *pDepthStencil,
                               uint32_t rangeCount,
                               const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, img, image);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!img || !img->mem) {
      cpvk_clear_refuse(cmd,
         "vkCmdClearDepthStencilImage on an image with no memory");
      return;
   }

   /*
    * The three formats vkCmdBeginRendering accepts as a depth attachment, and
    * for the same reason: these are the packings the depth load and store
    * kernels know. Anything else would be cleared into a layout nothing else
    * in this driver reads.
    */
   VkFormat fmt = img->vk.format;
   if (fmt != VK_FORMAT_D32_SFLOAT && fmt != VK_FORMAT_D24_UNORM_S8_UINT &&
       fmt != VK_FORMAT_D32_SFLOAT_S8_UINT && fmt != VK_FORMAT_D16_UNORM) {
      cpvk_clear_refuse(cmd,
         "vkCmdClearDepthStencilImage: unsupported depth/stencil format");
      return;
   }
   unsigned bpp = util_format_get_blocksize(vk_format_to_pipe_format(fmt));

   float depth = CLAMP(pDepthStencil->depth, 0.0f, 1.0f);
   uint32_t stencil = pDepthStencil->stencil & 0xffu;

   for (uint32_t i = 0; i < rangeCount; i++) {
      VkImageAspectFlags aspects = pRanges[i].aspectMask;
      bool want_depth = aspects & VK_IMAGE_ASPECT_DEPTH_BIT;
      bool want_stencil = aspects & VK_IMAGE_ASPECT_STENCIL_BIT;

      if (aspects & ~(VkImageAspectFlags)(VK_IMAGE_ASPECT_DEPTH_BIT |
                                          VK_IMAGE_ASPECT_STENCIL_BIT) ||
          !aspects ||
          (want_stencil && (fmt == VK_FORMAT_D32_SFLOAT ||
                            fmt == VK_FORMAT_D16_UNORM))) {
         cpvk_clear_refuse(cmd,
            "vkCmdClearDepthStencilImage: aspect the format does not have");
         return;
      }

      /*
       * The stencil byte is written here, in the image, which is where the
       * depth store writes it too (cp_clear.cu). Nothing else in this driver
       * reads or writes stencil -- there is no stencil test -- so a cleared
       * stencil aspect survives exactly as far as a stored one does.
       *
       * The mask is what keeps the aspect that was not named: in D24S8 both
       * live in one word, and in D32S8 the stencil byte shares the element
       * with the depth float.
       */
      uint32_t value[4] = { 0 };
      uint32_t mask[4] = { 0 };
      if (fmt == VK_FORMAT_D24_UNORM_S8_UINT) {
         uint32_t d24 = (uint32_t)lrintf(depth * 16777215.0f);
         value[0] = (stencil << 24) | (d24 & 0x00ffffffu);
         mask[0] = (want_depth ? 0x00ffffffu : 0u) |
                   (want_stencil ? 0xff000000u : 0u);
      } else if (fmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
         value[0] = cpvk_f32_bits(depth);
         value[1] = stencil;
         mask[0] = want_depth ? 0xffffffffu : 0u;
         mask[1] = want_stencil ? 0x000000ffu : 0u;
      } else if (fmt == VK_FORMAT_D16_UNORM) {
         /* Two bytes per texel and no stencil to preserve. The masked kernel's
          * 16-bit arm takes the low half of value[0] and mask[0], and the
          * quantisation is the one cp_depth_attachment_store uses, so a
          * cleared texel and a stored texel of the same depth agree. */
         value[0] = (uint32_t)lrintf(depth * 65535.0f);
         mask[0] = 0x0000ffffu;
      } else {
         value[0] = cpvk_f32_bits(depth);
         mask[0] = 0xffffffffu;
      }

      if (!cpvk_record_image_clear(cmd, img, &pRanges[i], bpp, value, mask)) {
         cpvk_clear_refuse(cmd,
            "vkCmdClearDepthStencilImage: subresource out of range");
         return;
      }
   }
}

/*
 * vkCmdClearAttachments: inside the pass, on whatever is bound, in sequence.
 *
 * Three things make this different from the two image clears above.
 *
 * It is ordered against the draws already recorded in this pass, so it is an
 * op like any other rather than something done to the image now.
 *
 * It clears the attachment's *view* -- the mip level and array layer
 * vkCmdBeginRendering resolved -- and it is clipped to the render area, which
 * is why the geometry comes from what BeginRendering worked out and not from
 * the image.
 *
 * And the depth aspect is not the depth image: this driver's depth test reads
 * a renderer-side buffer that the pass loads at the start and stores at the
 * end, exactly as LOAD_OP_CLEAR does, so an in-pass depth clear clears that
 * buffer.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdClearAttachments(VkCommandBuffer commandBuffer,
                         uint32_t attachmentCount,
                         const VkClearAttachment *pAttachments,
                         uint32_t rectCount, const VkClearRect *pRects)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   /*
    * A secondary command buffer recorded with RENDER_PASS_CONTINUE inherits
    * its render pass and never sees vkCmdBeginRendering, so nothing here knows
    * which subresource is bound or how wide its rows are. Refusing is the only
    * honest answer; guessing would clear the wrong memory.
    */
   if (!cmd->has_fb || cmd->active_scope >= cmd->num_scopes) {
      cpvk_clear_refuse(cmd, "vkCmdClearAttachments outside a render pass this "
                        "command buffer began (an inherited pass in a "
                        "secondary cannot resolve its attachments)");
      return;
   }
   struct cp_render_scope *scope = &cmd->scopes[cmd->active_scope];

   for (uint32_t a = 0; a < attachmentCount; a++) {
      const VkClearAttachment *at = &pAttachments[a];

      for (uint32_t r = 0; r < rectCount; r++) {
         const VkClearRect *cr = &pRects[r];

         /* One layer, because one layer is what a scope binds: the view's
          * baseArrayLayer is already folded into the base address. */
         if (cr->baseArrayLayer != 0 || cr->layerCount > 1) {
            cpvk_clear_refuse(cmd, "vkCmdClearAttachments: layered clear "
                              "(this driver binds one layer per pass)");
            return;
         }

         /* Clipped to the render area, which the spec requires the rectangle
          * to be inside anyway. */
         int64_t x0 = MAX2((int64_t)cr->rect.offset.x,
                           (int64_t)cmd->clear_area.offset.x);
         int64_t y0 = MAX2((int64_t)cr->rect.offset.y,
                           (int64_t)cmd->clear_area.offset.y);
         int64_t x1 = MIN2((int64_t)cr->rect.offset.x + cr->rect.extent.width,
                           (int64_t)cmd->clear_area.offset.x +
                              cmd->clear_area.extent.width);
         int64_t y1 = MIN2((int64_t)cr->rect.offset.y + cr->rect.extent.height,
                           (int64_t)cmd->clear_area.offset.y +
                              cmd->clear_area.extent.height);
         if (x1 <= x0 || y1 <= y0)
            continue;
         unsigned w = (unsigned)(x1 - x0), h = (unsigned)(y1 - y0);

         if (at->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
            if (at->colorAttachment != 0 || !cmd->fb.color ||
                !cmd->clear_color_image) {
               cpvk_clear_refuse(cmd, "vkCmdClearAttachments: no such colour "
                                 "attachment (this driver binds one)");
               return;
            }
            unsigned bpp =
               util_format_get_blocksize(cmd->clear_color_format);
            if (bpp != 1 && bpp != 2 && bpp != 4 && bpp != 8 && bpp != 16) {
               cpvk_clear_refuse(cmd,
                  "vkCmdClearAttachments: unsupported colour element size");
               return;
            }

            struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
            if (!op)
               return;
            op->clear = (struct cpvk_clear) {
               .image = cmd->clear_color_image,
               .data = cmd->fb.color,
               .offset = (uint64_t)y0 * cmd->clear_color_stride +
                         (uint64_t)x0 * bpp,
               .width = w,
               .height = h,
               .stride = cmd->clear_color_stride,
               .pixel_size = bpp,
               .samples = cmd->fb_samples,
               .sample_stride = cmd->fb.color_sample_stride,
               .mid_pass = true,
            };
            util_format_pack_rgba(cmd->clear_color_format, op->clear.value,
                                  at->clearValue.color.float32, 1);
         }

         if (at->aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT |
                               VK_IMAGE_ASPECT_STENCIL_BIT)) {
            if (!cmd->fb.has_zs) {
               cpvk_clear_refuse(cmd, "vkCmdClearAttachments: no depth/stencil "
                                 "attachment is bound");
               return;
            }

            /*
             * Stencil is written by the pass's depth store and nowhere else,
             * which is the behaviour LOAD_OP_CLEAR already has: the store
             * writes one value over the attachment. It can therefore honour a
             * clear of the whole render area and nothing narrower, so a
             * sub-rectangle is refused rather than quietly widened.
             */
            if (at->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) {
               if (x0 != cmd->clear_area.offset.x ||
                   y0 != cmd->clear_area.offset.y ||
                   w != cmd->clear_area.extent.width ||
                   h != cmd->clear_area.extent.height) {
                  cpvk_clear_refuse(cmd,
                     "vkCmdClearAttachments: a stencil clear of part of the "
                     "render area (this driver writes stencil only at the "
                     "pass's depth store, which covers the attachment)");
                  return;
               }
               scope->depth.stencil_clear = 1;
               scope->depth.stencil_value =
                  at->clearValue.depthStencil.stencil & 0xffu;
            }

            if (at->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
               struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
               if (!op)
                  return;
               op->clear = (struct cpvk_clear) {
                  .depth = true,
                  .depth_value = at->clearValue.depthStencil.depth,
                  .offset = ((uint64_t)y0 * cmd->fb.width + x0) *
                            sizeof(uint32_t),
                  .width = w,
                  .height = h,
                  .stride = cmd->fb.width * sizeof(uint32_t),
                  .pixel_size = sizeof(uint32_t),
                  .samples = cmd->fb_samples,
                  .sample_stride = (uint64_t)cmd->fb.width * cmd->fb.height *
                                   sizeof(uint32_t),
                  .mid_pass = true,
               };
            }
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyImage2(VkCommandBuffer commandBuffer,
                   const VkCopyImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, src, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_image, dst, pInfo->dstImage);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageCopy2 *r = &pInfo->pRegions[i];
      CUdeviceptr sb, db;
      size_t sp, dp;
      unsigned sbpp, dbpp;
      bool sok = cpvk_image_plane(src, r->srcSubresource.mipLevel,
                                  &sb, &sp, &sbpp);
      bool dok = cpvk_image_plane(dst, r->dstSubresource.mipLevel,
                                  &db, &dp, &dbpp);
      if (!sok || !dok)
         return;

      enum pipe_format sfmt = vk_format_to_pipe_format(src->vk.format);
      enum pipe_format dfmt = vk_format_to_pipe_format(dst->vk.format);
      /* VkImageCopy expresses the extent in source texels. Convert it to
       * source-format elements once and copy that many equally sized raw
       * elements on both sides. This is what makes a valid BC1 -> 8-byte
       * uncompressed copy 1 block -> 1 texel rather than 1 block -> 4 texels.
       * Vulkan format compatibility guarantees equal element sizes. */
      size_t copy_row = (size_t)util_format_get_nblocksx(
         sfmt, r->extent.width) * sbpp;
      unsigned copy_rows = util_format_get_nblocksy(sfmt, r->extent.height);
      if (sbpp != dbpp) {
         fprintf(stderr, "cudavk: refusing incompatible image-copy "
                 "element sizes %u -> %u bytes\n", sbpp, dbpp);
         continue;
      }

      unsigned sl = r->srcSubresource.mipLevel;
      unsigned dl = r->dstSubresource.mipLevel;
      unsigned sh = MAX2(src->vk.extent.height >> sl, 1u);
      unsigned dh = MAX2(dst->vk.extent.height >> dl, 1u);
      size_t src_slice = sp * util_format_get_nblocksy(sfmt, sh);
      size_t dst_slice = dp * util_format_get_nblocksy(dfmt, dh);
      bool src_3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
      bool dst_3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;
      unsigned slices = src_3d ? MAX2(r->extent.depth, 1u) :
                                 MAX2(r->srcSubresource.layerCount, 1u);
      size_t src_xy = (size_t)util_format_get_nblocksy(
                         sfmt, r->srcOffset.y) * sp +
                      (size_t)util_format_get_nblocksx(
                         sfmt, r->srcOffset.x) * sbpp;
      size_t dst_xy = (size_t)util_format_get_nblocksy(
                         dfmt, r->dstOffset.y) * dp +
                      (size_t)util_format_get_nblocksx(
                         dfmt, r->dstOffset.x) * dbpp;

      /* A copy has one sequence of slices. A slice is a z plane for a 3D
       * image and an array layer otherwise; multiplying layerCount by depth
       * would turn a valid 2D-array <-> 3D copy into N squared copies. */
      for (unsigned s = 0; s < slices; s++) {
         uint64_t src_plane = src_3d
            ? (uint64_t)(r->srcOffset.z + (int32_t)s) * src_slice
            : (uint64_t)(r->srcSubresource.baseArrayLayer + s) *
                 src->level_size[sl];
         uint64_t dst_plane = dst_3d
            ? (uint64_t)(r->dstOffset.z + (int32_t)s) * dst_slice
            : (uint64_t)(r->dstSubresource.baseArrayLayer + s) *
                 dst->level_size[dl];
         struct cpvk_copy *c = cpvk_record_copy(cmd);
         if (!c)
            return;
         *c = (struct cpvk_copy) {
            .dst_image = dst,
            .src = sb + src_plane + src_xy,
            .dst = db + dst_plane + dst_xy,
            .src_pitch = sp,
            .dst_pitch = dp,
            .width_bytes = copy_row,
            .rows = copy_rows,
            .src_end = cpvk_image_end(src),
            .dst_end = cpvk_image_end(dst),
         };
      }
   }
}

/*
 * Buffer <-> image, which is how every texture in the sample suite is
 * uploaded: staged into a buffer, then copied in. bufferRowLength of zero
 * means tightly packed, which is not the same as the image's row pitch and is
 * the whole reason this cannot be one flat memcpy.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer,
                           const VkCopyBufferToImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, pInfo->srcBuffer);
   VK_FROM_HANDLE(cpvk_image, img, pInfo->dstImage);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!buf || !buf->mem)
      return;

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pInfo->pRegions[i];
      CUdeviceptr ib;
      size_t ip;
      unsigned bpp;
      if (!cpvk_image_plane(img, r->imageSubresource.mipLevel, &ib, &ip, &bpp))
         return;

      /*
       * In blocks, not texels. A block-compressed format has a blocksize per
       * 4x4 block, so measuring a row as width * blocksize overstates it
       * sixteenfold: the capture's 1024-wide BC image wanted a two-megabyte
       * read out of a 128 KB staging buffer, and cuMemcpy2DAsync answers that
       * by faulting inside libcuda. For an uncompressed format the block is
       * one texel and this is the same arithmetic as before.
       */
      enum pipe_format pfmt = vk_format_to_pipe_format(img->vk.format);
      unsigned row_texels = r->bufferRowLength ? r->bufferRowLength
                                               : r->imageExtent.width;
      unsigned img_rows = r->bufferImageHeight ? r->bufferImageHeight
                                               : r->imageExtent.height;
      size_t src_row = (size_t)util_format_get_nblocksx(pfmt, row_texels) * bpp;
      size_t copy_row = (size_t)util_format_get_nblocksx(pfmt,
                                                        r->imageExtent.width) * bpp;
      unsigned copy_rows = util_format_get_nblocksy(pfmt, r->imageExtent.height);
      size_t src_slice = src_row *
                         util_format_get_nblocksy(pfmt, img_rows);
      unsigned level = r->imageSubresource.mipLevel;
      unsigned mip_h = MAX2(img->vk.extent.height >> level, 1u);
      size_t dst_slice = ip * util_format_get_nblocksy(pfmt, mip_h);
      unsigned layers = MAX2(r->imageSubresource.layerCount, 1u);
      unsigned depth = MAX2(r->imageExtent.depth, 1u);

      /* Array layers and 3D depth slices are consecutive buffer images.  The
       * old 2D-only loop copied slice zero of a 3D upload and the sampler then
       * read uninitialised slices as its z coordinate changed. */
      for (unsigned l = 0; l < layers; l++) {
         for (unsigned z = 0; z < depth; z++) {
            struct cpvk_copy *c = cpvk_record_copy(cmd);
            if (!c)
               return;
            *c = (struct cpvk_copy) {
               .dst_image = img,
               .src = buf->mem->dev_ptr + buf->offset + r->bufferOffset +
                      ((size_t)l * depth + z) * src_slice,
               .dst = ib +
                      (size_t)(r->imageSubresource.baseArrayLayer + l) *
                         img->level_size[level] +
                      (size_t)(r->imageOffset.z + (int32_t)z) * dst_slice +
                      (size_t)util_format_get_nblocksy(
                         pfmt, r->imageOffset.y) * ip +
                      (size_t)util_format_get_nblocksx(
                         pfmt, r->imageOffset.x) * bpp,
               .src_pitch = src_row,
               .dst_pitch = ip,
               .width_bytes = copy_row,
               .rows = copy_rows,
               /* Both ends bounded: this is the path that uploads every
                * texture and every mip level in a capture. */
               .src_end = buf->mem->dev_ptr + buf->offset + buf->vk.size,
               .dst_end = cpvk_image_end(img),
            };
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer,
                           const VkCopyImageToBufferInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, img, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_buffer, buf, pInfo->dstBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!buf || !buf->mem)
      return;

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pInfo->pRegions[i];
      CUdeviceptr ib;
      size_t ip;
      unsigned bpp;
      if (!cpvk_image_plane(img, r->imageSubresource.mipLevel, &ib, &ip, &bpp))
         return;

      enum pipe_format pfmt = vk_format_to_pipe_format(img->vk.format);
      unsigned row_texels = r->bufferRowLength ? r->bufferRowLength
                                               : r->imageExtent.width;
      unsigned img_rows = r->bufferImageHeight ? r->bufferImageHeight
                                               : r->imageExtent.height;
      size_t dst_row = (size_t)util_format_get_nblocksx(pfmt, row_texels) * bpp;
      size_t dst_slice = dst_row * util_format_get_nblocksy(pfmt, img_rows);
      unsigned level = r->imageSubresource.mipLevel;
      unsigned mip_h = MAX2(img->vk.extent.height >> level, 1u);
      size_t src_slice = ip * util_format_get_nblocksy(pfmt, mip_h);
      unsigned layers = MAX2(r->imageSubresource.layerCount, 1u);
      unsigned depth = MAX2(r->imageExtent.depth, 1u);

      for (unsigned l = 0; l < layers; l++) {
         for (unsigned z = 0; z < depth; z++) {
            struct cpvk_copy *c = cpvk_record_copy(cmd);
            if (!c)
               return;
            *c = (struct cpvk_copy) {
               .src = ib +
                      (size_t)(r->imageSubresource.baseArrayLayer + l) *
                         img->level_size[level] +
                      (size_t)(r->imageOffset.z + (int32_t)z) * src_slice +
                      (size_t)util_format_get_nblocksy(
                         pfmt, r->imageOffset.y) * ip +
                      (size_t)util_format_get_nblocksx(
                         pfmt, r->imageOffset.x) * bpp,
               .dst = buf->mem->dev_ptr + buf->offset + r->bufferOffset +
                      ((size_t)l * depth + z) * dst_slice,
               .src_pitch = ip,
               .dst_pitch = dst_row,
               .width_bytes = (size_t)util_format_get_nblocksx(
                  pfmt, r->imageExtent.width) * bpp,
               .rows = util_format_get_nblocksy(pfmt, r->imageExtent.height),
               .src_end = cpvk_image_end(img),
               .dst_end = buf->mem->dev_ptr + buf->offset + buf->vk.size,
            };
         }
      }
   }
}

static bool
cpvk_is_bgra(VkFormat f)
{
   return f == VK_FORMAT_B8G8R8A8_UNORM;
}

static bool
cpvk_is_blit_rgba8(VkFormat f)
{
   return f == VK_FORMAT_R8G8B8A8_UNORM ||
          f == VK_FORMAT_B8G8R8A8_UNORM;
}

/*
 * A blit, for the case this driver actually sees: same extent, same texel
 * size, no scaling and no format conversion. That is what an offscreen app
 * does to get a rendered image into a linear buffer it can save.
 *
 * A scaling or converting blit is refused rather than approximated, because a
 * silently wrong image is the failure this driver's history is made of. It
 * will be a kernel when something needs it.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBlitImage2(VkCommandBuffer commandBuffer,
                   const VkBlitImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, src, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_image, dst, pInfo->dstImage);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageBlit2 *r = &pInfo->pRegions[i];
      CUdeviceptr sb, db;
      size_t sp, dp;
      unsigned sbpp, dbpp;
      if (!cpvk_image_plane(src, r->srcSubresource.mipLevel, &sb, &sp, &sbpp) ||
          !cpvk_image_plane(dst, r->dstSubresource.mipLevel, &db, &dp, &dbpp)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      int sw = r->srcOffsets[1].x - r->srcOffsets[0].x;
      int sh = r->srcOffsets[1].y - r->srcOffsets[0].y;
      int dw = r->dstOffsets[1].x - r->dstOffsets[0].x;
      int dh = r->dstOffsets[1].y - r->dstOffsets[0].y;

      if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || sbpp != dbpp) {
         fprintf(stderr, "cudavk: vkCmdBlitImage %dx%d -> %dx%d (%u/%u bpp) "
                 "changes texel size or inverts, which is not implemented\n",
                 sw, sh, dw, dh, sbpp, dbpp);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      bool scaling = (sw != dw || sh != dh);
      if (scaling && (src->color < 0 || dst->color < 0)) {
         fprintf(stderr, "cudavk: vkCmdBlitImage scales unsupported formats "
                 "%u -> %u\n", src->vk.format, dst->vk.format);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      /* Same size and same texel width, so the only conversion a blit can be
       * asked for here is a channel order. Anything else is refused rather
       * than approximated. */
      bool swap_rb = false;
      if (src->vk.format != dst->vk.format) {
         if (cpvk_is_blit_rgba8(src->vk.format) &&
             cpvk_is_blit_rgba8(dst->vk.format) &&
             cpvk_is_bgra(src->vk.format) != cpvk_is_bgra(dst->vk.format)) {
            swap_rb = true;
         } else {
            fprintf(stderr, "cudavk: vkCmdBlitImage %u -> %u is a format "
                    "conversion that is not implemented\n",
                    src->vk.format, dst->vk.format);
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
      }

      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      /*
       * The array layer, as in vkCmdCopyImage. A blit into a cube face or an
       * array slice names it in the subresource, and without this every one
       * of them lands on layer zero.
       */
      uint64_t bs_layer = (uint64_t)r->srcSubresource.baseArrayLayer *
                          src->level_size[r->srcSubresource.mipLevel];
      uint64_t bd_layer = (uint64_t)r->dstSubresource.baseArrayLayer *
                          dst->level_size[r->dstSubresource.mipLevel];

      *c = (struct cpvk_copy) {
         .dst_image = dst,
         .src = sb + bs_layer + (size_t)r->srcOffsets[0].y * sp +
                (size_t)r->srcOffsets[0].x * sbpp,
         .dst = db + bd_layer + (size_t)r->dstOffsets[0].y * dp +
                (size_t)r->dstOffsets[0].x * dbpp,
         .src_pitch = sp,
         .dst_pitch = dp,
         .width_bytes = (size_t)sw * sbpp,
         .rows = sh,
         .src_end = cpvk_image_end(src),
         .dst_end = cpvk_image_end(dst),
         .swap_rb = swap_rb,
         .src_w = (scaling || swap_rb) ? (unsigned)sw : 0,
         .src_h = (scaling || swap_rb) ? (unsigned)sh : 0,
         .dst_w = (scaling || swap_rb) ? (unsigned)dw : 0,
         .dst_h = (scaling || swap_rb) ? (unsigned)dh : 0,
         .bpp = sbpp,
         .filter_linear = pInfo->filter == VK_FILTER_LINEAR,
         .encoding = src->color,
         .dst_encoding = dst->color,
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   /* The colour resolve is ordered before the explicit end marker. */
   if (cmd->resolve_valid) {
      struct cpvk_copy *copy = cpvk_record_copy(cmd);
      if (copy)
         *copy = cmd->resolve;
   }

   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_END_RENDER);
   if (op)
      op->scope_index = cmd->active_scope;

   /* After the end marker: the depth attachment image is written by the
    * scope's store, and the resolve reads what that store produced. */
   if (cmd->depth_resolve_valid) {
      struct cpvk_copy *copy = cpvk_record_copy(cmd);
      if (copy)
         *copy = cmd->depth_resolve;
      cmd->depth_resolve_valid = false;
   }

   cmd->active_scope = CP_RENDER_SCOPE_NONE;
   cmd->has_fb = false;
   cmd->resolve_valid = false;
}

/*
 * A multisample resolve: the average of the samples, which are planes
 * cpvk_image::sample_stride apart.
 *
 * It took sample zero and said so out loud until images were sized for their
 * samples, because until then there was nothing else in memory to average.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResolveImage2(VkCommandBuffer commandBuffer,
                      const VkResolveImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, src, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_image, dst, pInfo->dstImage);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageResolve2 *r = &pInfo->pRegions[i];
      CUdeviceptr sb, db;
      size_t sp, dp;
      unsigned sbpp, dbpp;
      if (!cpvk_image_plane(src, r->srcSubresource.mipLevel, &sb, &sp, &sbpp) ||
          !cpvk_image_plane(dst, r->dstSubresource.mipLevel, &db, &dp, &dbpp))
         return;
      if (sbpp != dbpp || sbpp != 4) {
         fprintf(stderr, "cudavk: vkCmdResolveImage of a %u-byte texel is "
                 "not implemented\n", sbpp);
         return;
      }

      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      *c = (struct cpvk_copy) {
         .dst_image = dst,
         .src = sb + (size_t)r->srcOffset.y * sp + (size_t)r->srcOffset.x * sbpp,
         .dst = db + (size_t)r->dstOffset.y * dp + (size_t)r->dstOffset.x * dbpp,
         .src_pitch = sp,
         .dst_pitch = dp,
         .width_bytes = (size_t)r->extent.width * sbpp,
         .rows = r->extent.height,
         .src_end = cpvk_image_end(src),
         .dst_end = cpvk_image_end(dst),
         .samples = MAX2(src->vk.samples, 1u),
         .sample_stride = src->sample_stride,
         .encoding = src->color,
      };
   }
}


/* ------------------------------------------------------------------ events
 *
 * The device-side half is empty on purpose. One stream, program order: a
 * vkCmdWaitEvents2 recorded after the vkCmdSetEvent2 that satisfies it is
 * already ordered behind it, and a wait on an event set by the host cannot be
 * reached until that host call has happened. The host-side half is real,
 * because vkSetEvent and vkGetEventStatus are questions with answers.
 */
VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateEvent(VkDevice _device, const VkEventCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_event *event =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event),
                       VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   event->dev = dev;
   event->alloc = pAllocator ? *pAllocator : dev->vk.alloc;
   atomic_init(&event->refcnt, 1);
   if (mtx_init(&event->lock, mtx_plain) != thrd_success) {
      vk_object_free(&dev->vk, &event->alloc, event);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   if (cnd_init(&event->changed) != thrd_success) {
      mtx_destroy(&event->lock);
      vk_object_free(&dev->vk, &event->alloc, event);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *pEvent = cpvk_event_to_handle(event);
   return VK_SUCCESS;
}

void
cpvk_event_ref(struct cpvk_event *event)
{
   if (event)
      atomic_fetch_add_explicit(&event->refcnt, 1, memory_order_relaxed);
}

void
cpvk_event_unref(struct cpvk_event *event)
{
   if (!event ||
       atomic_fetch_sub_explicit(&event->refcnt, 1,
                                 memory_order_acq_rel) != 1)
      return;
   cnd_destroy(&event->changed);
   mtx_destroy(&event->lock);
   vk_object_free(&event->dev->vk, &event->alloc, event);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyEvent(VkDevice _device, VkEvent _event,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   cpvk_event_unref(event);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_GetEventStatus(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   if (!event)
      return VK_EVENT_RESET;
   mtx_lock(&event->lock);
   bool signaled = event->signaled;
   mtx_unlock(&event->lock);
   return signaled ? VK_EVENT_SET : VK_EVENT_RESET;
}

static void
cpvk_event_set_state(struct cpvk_event *event, bool signaled)
{
   if (!event)
      return;
   mtx_lock(&event->lock);
   event->signaled = signaled;
   cnd_broadcast(&event->changed);
   mtx_unlock(&event->lock);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_SetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   cpvk_event_set_state(event, true);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_ResetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   cpvk_event_set_state(event, false);
   return VK_SUCCESS;
}

static void
cpvk_record_event(struct cpvk_cmd_buffer *cmd, struct cpvk_event *event,
                  enum cpvk_op_kind kind)
{
   if (!event)
      return;
   if (!cpvk_cmd_retain_event(cmd, event)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   struct cpvk_op *op = cpvk_op_alloc(cmd, kind);
   if (op)
      op->event.event = event;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_event, event, _event);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_event(cmd, event, CPVK_OP_EVENT_SET);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                    VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_event, event, _event);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_event(cmd, event, CPVK_OP_EVENT_RESET);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                    const VkEvent *pEvents,
                    const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   for (uint32_t i = 0; i < eventCount; i++) {
      VK_FROM_HANDLE(cpvk_event, event, pEvents[i]);
      cpvk_record_event(cmd, event, CPVK_OP_EVENT_WAIT);
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetEvent(VkCommandBuffer commandBuffer, VkEvent _event,
                 VkPipelineStageFlags stageMask)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_event, event, _event);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_event(cmd, event, CPVK_OP_EVENT_SET);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResetEvent(VkCommandBuffer commandBuffer, VkEvent _event,
                   VkPipelineStageFlags stageMask)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_event, event, _event);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_event(cmd, event, CPVK_OP_EVENT_RESET);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount,
                   const VkEvent *pEvents,
                   VkPipelineStageFlags srcStageMask,
                   VkPipelineStageFlags dstStageMask,
                   uint32_t memoryBarrierCount,
                   const VkMemoryBarrier *pMemoryBarriers,
                   uint32_t bufferMemoryBarrierCount,
                   const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                   uint32_t imageMemoryBarrierCount,
                   const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   for (uint32_t i = 0; i < eventCount; i++) {
      VK_FROM_HANDLE(cpvk_event, event, pEvents[i]);
      cpvk_record_event(cmd, event, CPVK_OP_EVENT_WAIT);
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPipelineBarrier(VkCommandBuffer commandBuffer,
                        VkPipelineStageFlags srcStageMask,
                        VkPipelineStageFlags dstStageMask,
                        VkDependencyFlags dependencyFlags,
                        uint32_t memoryBarrierCount,
                        const VkMemoryBarrier *pMemoryBarriers,
                        uint32_t bufferMemoryBarrierCount,
                        const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                        uint32_t imageMemoryBarrierCount,
                        const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_op_alloc(cmd, CPVK_OP_BARRIER);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                         const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_op_alloc(cmd, CPVK_OP_BARRIER);
}

static void
cpvk_execute_order_point(struct cpvk_device *dev)
{
   cp_batch_flush_why(&dev->renderer, "Vulkan order point");
   cp_pass_finish(&dev->renderer);
}

struct cpvk_event_callback {
   struct cpvk_event *event;
   enum cpvk_op_kind kind;
};

static void CUDA_CB
cpvk_event_callback_run(void *data)
{
   struct cpvk_event_callback *callback = data;
   if (callback->kind == CPVK_OP_EVENT_WAIT) {
      mtx_lock(&callback->event->lock);
      while (!callback->event->signaled)
         cnd_wait(&callback->event->changed, &callback->event->lock);
      mtx_unlock(&callback->event->lock);
   } else {
      cpvk_event_set_state(callback->event,
         callback->kind == CPVK_OP_EVENT_SET);
   }
   cpvk_event_unref(callback->event);
   free(callback);
}

VkResult
cpvk_execute_order_op(struct cpvk_device *dev, const struct cpvk_op *op)
{
   cpvk_execute_order_point(dev);
   if (op->kind == CPVK_OP_BARRIER)
      return VK_SUCCESS;

   struct cpvk_event *event = op->event.event;
   struct cpvk_event_callback *callback = malloc(sizeof(*callback));
   if (!callback)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   callback->event = event;
   callback->kind = op->kind;
   cpvk_event_ref(event);
   CUresult err = cuLaunchHostFunc(dev->renderer.stream,
                                  cpvk_event_callback_run, callback);
   if (err != CUDA_SUCCESS) {
      cpvk_event_unref(event);
      free(callback);
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }
   return VK_SUCCESS;
}


VkResult
cpvk_execute_copy(struct cpvk_device *dev, const struct cpvk_copy *c)
{
   struct cp_context *cp = &dev->renderer;

   /* A copy observes rendering, so whatever is held back has to run first. */
   cp_batch_flush(cp);


   /*
    * Refuse a copy that cannot be one. A zero endpoint is an image or buffer
    * whose memory was never bound, and a pitch narrower than the row is a
    * region computed from the wrong subresource; CUDA takes both as an
    * invitation to walk off the end.
    */
   /*
    * A scaling blit describes its geometry in src_w/src_h and dst_w/dst_h and
    * carries the source's row width in width_bytes, so the pitch and reach
    * tests below do not apply to it -- they rejected three perfectly good mip
    * downscales before this line existed.
    */
   uint64_t src_reach = c->src_w ? 0 :
                        c->src + (uint64_t)(c->rows ? c->rows - 1 : 0) *
                        (c->src_pitch ? c->src_pitch : c->width_bytes) +
                        c->width_bytes;
   uint64_t dst_reach = c->src_w ? 0 :
                        c->dst + (uint64_t)(c->rows ? c->rows - 1 : 0) *
                        (c->dst_pitch ? c->dst_pitch : c->width_bytes) +
                        c->width_bytes;

   if (!c->src || !c->dst || !c->width_bytes || !c->rows ||
       (!c->src_w && c->src_pitch && c->src_pitch < c->width_bytes) ||
       (!c->src_w && c->dst_pitch && c->dst_pitch < c->width_bytes) ||
       (c->src_end && src_reach > c->src_end) ||
       (c->dst_end && dst_reach > c->dst_end)) {
      fprintf(stderr, "cudavk: refusing copy src=%p dst=%p %zux%zu "
              "pitch %zu->%zu reach %p/%p ends %p/%p\n",
              (void *)(uintptr_t)c->src, (void *)(uintptr_t)c->dst,
              c->width_bytes, c->rows, c->src_pitch, c->dst_pitch,
              (void *)(uintptr_t)src_reach, (void *)(uintptr_t)dst_reach,
              (void *)(uintptr_t)c->src_end, (void *)(uintptr_t)c->dst_end);
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }

   if (c->rows <= 1 && !c->src_pitch && !c->dst_pitch) {
      if (cuMemcpyDtoDAsync(c->dst, c->src, c->width_bytes,
                            cp->stream) != CUDA_SUCCESS)
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      return VK_SUCCESS;
   }

   if (c->samples > 1 && cp->dev->kernels.resolve_samples &&
       c->encoding >= 0 && c->width_bytes && c->rows) {
      /*
       * Resolve on the device, with the kernel that already exists.
       *
       * The host path below reads every sample plane back, averages it on the
       * CPU and writes the result out again, once per render pass -- and its
       * comment said that was not on a frame's critical path. It is: the
       * sweep harness measures multisampling at 13.29 ms a frame against the
       * Gallium driver's 1.49, and that driver launches this same kernel from
       * cp_resource.c rather than draining the stream three times a frame.
       */
      struct cp_resolve_msaa_args ra = {
         .src = c->src,
         .dst = c->dst,
         .width = (uint32_t)(c->width_bytes / 4),
         .height = (uint32_t)c->rows,
         .src_stride = (uint32_t)(c->src_pitch ? c->src_pitch : c->width_bytes),
         .dst_stride = (uint32_t)(c->dst_pitch ? c->dst_pitch : c->width_bytes),
         .sample_stride = (uint32_t)c->sample_stride,
         .num_samples = c->samples,
         .encoding = c->encoding,
      };
      void *params[] = { &ra };
      if (cp_launch(cp, cp->dev->kernels.resolve_samples,
                    (ra.width + 15) / 16, (ra.height + 15) / 16, 1,
                    16, 16, 1, 0, cp->stream, params, NULL) == CUDA_SUCCESS)
         return VK_SUCCESS;
   }

   if (c->samples > 1) {
      fprintf(stderr, "cudavk: device MSAA resolve is unavailable\n");
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }


   if (cp_debug->debug_rt) {
      /* The first texels of the source, as halves: what the pass just
       * rendered, before this copy places it. */
      uint16_t h[8] = { 0 };
      cuStreamSynchronize(cp->stream);
      if (cuMemcpyDtoH(h, c->src, sizeof(h)) == CUDA_SUCCESS) {
         fprintf(stderr, "copysrc halfs:");
         for (int k = 0; k < 8; k++)
            fprintf(stderr, " %04x", h[k]);
         fprintf(stderr, "\n");
      }
   }

   if (c->src_w && c->encoding >= 0 && c->dst_encoding >= 0 &&
       cp->dev->kernels.blit_linear) {
      /*
       * A scaling blit, on the device.
       *
       * The host path below reads the source back, filters it on the CPU and
       * writes the result out, and its comment says this builds a mip chain
       * at load time and is not on a frame's critical path. The sweep harness
       * disagrees: texturemipmapgen is 1.77x the reference and pbribl spends
       * 33 seconds building its irradiance cube, which is 11.7x that driver's
       * whole run.
       *
       * cp_blit_linear implements LINEAR and NEAREST at Vulkan pixel centres
       * and decodes/encodes the source and destination independently, including
       * RGBA/BGRA conversion. No host fallback remains.
       */
      struct cp_blit_linear_args ba = {
         .src = c->src,
         .dst = c->dst,
         .src_width = c->src_w,
         .src_height = c->src_h,
         .dst_width = c->dst_w,
         .dst_height = c->dst_h,
         .src_stride = (uint32_t)(c->src_pitch ? c->src_pitch
                                               : (size_t)c->src_w * c->bpp),
         .dst_stride = (uint32_t)(c->dst_pitch ? c->dst_pitch
                                               : (size_t)c->dst_w * c->bpp),
         .src_layer_stride = 0,
         .dst_layer_stride = 0,
         .layers = 1,
         .src_encoding = c->encoding,
         .dst_encoding = c->dst_encoding,
         .filter_linear = c->filter_linear,
      };
      void *params[] = { &ba };
      CUresult err = cp_launch(cp, cp->dev->kernels.blit_linear,
                               (c->dst_w + 15) / 16,
                               (c->dst_h + 15) / 16, 1,
                               16, 16, 1, 0, cp->stream, params, NULL);
      if (err != CUDA_SUCCESS) {
         fprintf(stderr, "cudavk: device blit failed: %d\n", err);
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      }
      return VK_SUCCESS;
   }

   if (c->src_w) {
      fprintf(stderr, "cudavk: unsupported device blit encoding %d -> %d\n",
              c->encoding, c->dst_encoding);
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }


   CUDA_MEMCPY2D m = {
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = c->src,
      .srcPitch = c->src_pitch ? c->src_pitch : c->width_bytes,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = c->dst,
      .dstPitch = c->dst_pitch ? c->dst_pitch : c->width_bytes,
      .WidthInBytes = c->width_bytes,
      .Height = c->rows,
   };
   if (cuMemcpy2DAsync(&m, cp->stream) != CUDA_SUCCESS)
      return vk_error(dev, VK_ERROR_DEVICE_LOST);

   if (cp_debug->debug_rt) {
      /* And what landed, read back from the destination this copy just
       * wrote: the face of the cube level, not the offscreen it came from. */
      uint16_t hd[8] = { 0 };
      cuStreamSynchronize(cp->stream);
      if (cuMemcpyDtoH(hd, c->dst, sizeof(hd)) == CUDA_SUCCESS) {
         fprintf(stderr, "copydst rows=%zu wb=%zu halfs:", (size_t)c->rows,
                 (size_t)c->width_bytes);
         for (int k = 0; k < 8; k++)
            fprintf(stderr, " %04x", hd[k]);
         fprintf(stderr, "\n");
      }
   }
   return VK_SUCCESS;
}



/* ------------------------------------------------------------- queries */

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateQueryPool(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_query_pool *pool =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                       VK_OBJECT_TYPE_QUERY_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->dev = dev;
   pool->alloc = pAllocator ? *pAllocator : dev->vk.alloc;
   atomic_init(&pool->refcnt, 1);
   pool->type = pCreateInfo->queryType;
   pool->count = pCreateInfo->queryCount;
   pool->results = calloc(pool->count, sizeof(*pool->results));
   pool->available = calloc(pool->count, sizeof(*pool->available));
   if (!pool->results || !pool->available ||
       mtx_init(&pool->lock, mtx_plain) != thrd_success) {
      free(pool->results);
      free(pool->available);
      pool->results = NULL;
      pool->available = NULL;
      vk_object_free(&dev->vk, pAllocator, pool);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   if (cnd_init(&pool->changed) != thrd_success) {
      mtx_destroy(&pool->lock);
      free(pool->results);
      free(pool->available);
      vk_object_free(&dev->vk, pAllocator, pool);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *pQueryPool = cpvk_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

void
cpvk_query_pool_ref(struct cpvk_query_pool *pool)
{
   if (pool)
      atomic_fetch_add_explicit(&pool->refcnt, 1, memory_order_relaxed);
}

void
cpvk_query_pool_unref(struct cpvk_query_pool *pool)
{
   if (!pool ||
       atomic_fetch_sub_explicit(&pool->refcnt, 1,
                                 memory_order_acq_rel) != 1)
      return;
   cnd_destroy(&pool->changed);
   mtx_destroy(&pool->lock);
   free(pool->results);
   free(pool->available);
   vk_object_free(&pool->dev->vk, &pool->alloc, pool);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyQueryPool(VkDevice _device, VkQueryPool _pool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   cpvk_query_pool_unref(pool);
}

static void
cpvk_record_query(struct cpvk_cmd_buffer *cmd, struct cpvk_query_pool *pool,
                  uint32_t first, uint32_t count, bool reset)
{
   if (!pool)
      return;
   if (!cpvk_cmd_retain_query(cmd, pool)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_QUERY);
   if (!op)
      return;
   op->query = (struct cpvk_query_op) {
      .pool = pool, .first = first, .count = count, .reset = reset,
   };
}

/*
 * vkCmdCopyQueryPoolResults.
 *
 * vk_common_CmdCopyQueryPoolResults forwards to
 * CmdCopyQueryPoolResultsToMemoryKHR, unimplemented here, so this was another
 * valid pointer onto a null one.
 *
 * It writes exactly what vkGetQueryPoolResults writes -- including that this
 * driver's queries are stubs and its timestamps carry no valid bits -- but in
 * the queue's order rather than the host's, which is the whole difference
 * between the two commands and the only thing this can honestly promise.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                             uint32_t firstQuery, uint32_t queryCount,
                             VkBuffer dstBuffer, VkDeviceSize dstOffset,
                             VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   VK_FROM_HANDLE(cpvk_buffer, dst, dstBuffer);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));

   if (!pool || !dst || !dst->mem || !queryCount)
      return;
   if (!cpvk_cmd_retain_query(cmd, pool)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_QUERY_COPY);
   if (!op)
      return;
   op->query_copy = (struct cpvk_query_copy) {
      .pool = pool, .first = firstQuery, .count = queryCount,
      .dst = dst->mem->dev_ptr + dst->offset + dstOffset,
      .stride = stride, .flags = flags,
   };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                       uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_query(cmd, pool, firstQuery, queryCount, true);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWriteTimestamp2(VkCommandBuffer commandBuffer,
                        VkPipelineStageFlags2 stage, VkQueryPool _pool,
                        uint32_t query)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_query(cmd, pool, query, 1, false);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWriteTimestamp(VkCommandBuffer commandBuffer,
                       VkPipelineStageFlagBits stage, VkQueryPool pool,
                       uint32_t query)
{
   cpvk_CmdWriteTimestamp2(commandBuffer, stage, pool, query);
}

/* Occlusion and statistics queries: recorded so the pool becomes available,
 * counted as zero because nothing counts them. */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool pool,
                   uint32_t query, VkQueryControlFlags flags)
{
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                 uint32_t query)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   CPVK_CTX_SCOPE(cpvk_cmd_buffer_device(cmd));
   cpvk_record_query(cmd, pool, query, 1, false);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_GetQueryPoolResults(VkDevice _device, VkQueryPool _pool,
                         uint32_t firstQuery, uint32_t queryCount,
                         size_t dataSize, void *pData, VkDeviceSize stride,
                         VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   CPVK_CTX_SCOPE(dev);

   if (!pool)
      return VK_ERROR_UNKNOWN;

   if (flags & VK_QUERY_RESULT_WAIT_BIT) {
      mtx_lock(&pool->lock);
      for (uint32_t i = 0; i < queryCount; i++) {
         uint32_t q = firstQuery + i;
         if (q >= pool->count)
            continue;
         while (!atomic_load_explicit(&pool->available[q],
                                      memory_order_acquire)) {
            if (atomic_load_explicit(&dev->device_lost,
                                     memory_order_acquire)) {
               mtx_unlock(&pool->lock);
               return vk_error(dev, VK_ERROR_DEVICE_LOST);
            }
            if (cnd_wait(&pool->changed, &pool->lock) == thrd_error) {
               mtx_unlock(&pool->lock);
               return vk_error(dev, VK_ERROR_DEVICE_LOST);
            }
         }
      }
      mtx_unlock(&pool->lock);
   }

   char *out = pData;
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < queryCount; i++) {
      uint32_t q = firstQuery + i;
      bool avail = q < pool->count &&
         atomic_load_explicit(&pool->available[q], memory_order_acquire);
      uint64_t value = avail ? pool->results[q] : 0;

      /* NOT_READY is for a query neither waited for nor reported partially. */
      if (!avail && !(flags & (VK_QUERY_RESULT_WAIT_BIT |
                               VK_QUERY_RESULT_PARTIAL_BIT)))
         result = VK_NOT_READY;

      char *slot = out + (size_t)i * stride;
      bool write_value = avail || (flags & VK_QUERY_RESULT_PARTIAL_BIT);
      if (flags & VK_QUERY_RESULT_64_BIT) {
         uint64_t *p = (uint64_t *)slot;
         if (write_value)
            p[0] = value;
         if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
            p[1] = avail;
      } else {
         uint32_t *p = (uint32_t *)slot;
         if (write_value)
            p[0] = (uint32_t)value;
         if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
            p[1] = avail;
      }
   }

   return result;
}

struct cpvk_query_callback {
   struct cpvk_query_pool *pool;
   struct cpvk_query_op op;
};

static void CUDA_CB
cpvk_query_complete(void *data)
{
   struct cpvk_query_callback *cb = data;
   struct cpvk_query_pool *pool = cb->pool;
   const struct cpvk_query_op *q = &cb->op;

   mtx_lock(&pool->lock);
   if (q->reset) {
      for (uint32_t i = 0; i < q->count; i++)
         if (q->first + i < pool->count) {
            pool->results[q->first + i] = 0;
            atomic_store_explicit(&pool->available[q->first + i], false,
                                  memory_order_release);
         }
   } else if (q->first < pool->count) {
      uint64_t value = 0;
      if (pool->type == VK_QUERY_TYPE_TIMESTAMP) {
         struct timespec ts;
         clock_gettime(CLOCK_MONOTONIC, &ts);
         value = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
      }
      pool->results[q->first] = value;
      atomic_store_explicit(&pool->available[q->first], true,
                            memory_order_release);
   }
   cnd_broadcast(&pool->changed);
   mtx_unlock(&pool->lock);

   cpvk_query_pool_unref(pool);
   free(cb);
}

/*
 * The recorded result copy, at submit.
 *
 * The results are filled by a host callback launched on the renderer stream,
 * so they are known once the stream has reached this point -- which is also
 * exactly the set of queries "recorded before this command" names. Draining
 * once here is what makes reading pool->results on the host legitimate, and
 * VK_QUERY_RESULT_WAIT_BIT is then already satisfied for everything this
 * submission could have completed.
 */
VkResult
cpvk_execute_query_copy(struct cpvk_device *dev,
                        const struct cpvk_query_copy *qc)
{
   struct cpvk_query_pool *pool = qc->pool;

   cpvk_execute_order_point(dev);
   if (cuStreamSynchronize(dev->renderer.stream) != CUDA_SUCCESS)
      return vk_error(dev, VK_ERROR_DEVICE_LOST);

   bool wide = qc->flags & VK_QUERY_RESULT_64_BIT;
   bool with_avail = qc->flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
   unsigned words = 1 + (with_avail ? 1 : 0);
   size_t slot_size = words * (wide ? sizeof(uint64_t) : sizeof(uint32_t));
   uint64_t stride = qc->stride ? qc->stride : slot_size;

   for (uint32_t i = 0; i < qc->count; i++) {
      uint32_t q = qc->first + i;
      bool avail = q < pool->count &&
         atomic_load_explicit(&pool->available[q], memory_order_acquire);
      uint64_t value = avail ? pool->results[q] : 0;
      /* A query neither available nor asked for partially is left alone, as
       * vkGetQueryPoolResults leaves its slot alone. */
      bool write_value = avail || (qc->flags & VK_QUERY_RESULT_PARTIAL_BIT);
      if (!write_value && !with_avail)
         continue;

      uint64_t slot64[2] = { value, avail };
      uint32_t slot32[2] = { (uint32_t)value, avail };
      const void *src = wide ? (const void *)slot64 : (const void *)slot32;
      size_t unit = wide ? sizeof(uint64_t) : sizeof(uint32_t);
      CUdeviceptr at = qc->dst + (uint64_t)i * stride;

      if (write_value &&
          cuMemcpyHtoD(at, src, unit) != CUDA_SUCCESS)
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      if (with_avail &&
          cuMemcpyHtoD(at + unit, (const char *)src + unit, unit) !=
             CUDA_SUCCESS)
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }
   return VK_SUCCESS;
}

VkResult
cpvk_execute_query(struct cpvk_device *dev, const struct cpvk_query_op *q)
{
   /* A query observes everything recorded before it, which includes pass
    * segments a batch flush alone would leave open. */
   cpvk_execute_order_point(dev);

   struct cpvk_query_callback *cb = malloc(sizeof(*cb));
   if (!cb)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   cb->pool = q->pool;
   cb->op = *q;
   cpvk_query_pool_ref(cb->pool);
   if (cuLaunchHostFunc(dev->renderer.stream, cpvk_query_complete, cb) !=
       CUDA_SUCCESS) {
      cpvk_query_pool_unref(cb->pool);
      free(cb);
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }
   return VK_SUCCESS;
}


