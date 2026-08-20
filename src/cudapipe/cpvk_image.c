/*
 * Images, image views and the format table.
 *
 * The layout is the driver's own and deliberately plain: linear, with each
 * level at its own offset. "Every level sits at its own mip_offsets[level]"
 * is one of the more expensive lessons in this tree -- leaving it out of an
 * address did not merely lose the small levels, it wrote level 0 over and
 * over while the rest stayed untouched, and stayed invisible until textureLod
 * started being honoured.
 *
 * Format support comes from the same encodings the kernels decode, so what
 * the driver advertises and what its kernels can actually read cannot drift
 * apart. Anything not in the table is refused rather than clamped.
 */

#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_image.h"
#include "vk_util.h"

#include "kernels/cp_rast_types.h"

/*
 * What the capture and the sample set actually need, and nothing else. The
 * list grows as kernels grow: advertising a format the writeback cannot encode
 * is the failure mode lavapipe's sample counts already demonstrated here.
 */
static const struct cpvk_format_info cpvk_formats[] = {
   { VK_FORMAT_R8G8B8A8_UNORM,      CP_TEXEL_R8G8B8A8_UNORM,     CP_COLOR_R8G8B8A8_UNORM,      false },
   { VK_FORMAT_R8G8B8A8_SRGB,       CP_TEXEL_R8G8B8A8_UNORM,     CP_COLOR_R8G8B8A8_SRGB,       false },
   { VK_FORMAT_B8G8R8A8_UNORM,      CP_TEXEL_B8G8R8A8_UNORM,     CP_COLOR_B8G8R8A8_UNORM,      false },
   { VK_FORMAT_B8G8R8A8_SRGB,       CP_TEXEL_B8G8R8A8_UNORM,     CP_COLOR_B8G8R8A8_SRGB,       false },
   { VK_FORMAT_R8G8_UNORM,          CP_TEXEL_R8G8_UNORM,         -1,                           false },
   { VK_FORMAT_R8_UNORM,            CP_TEXEL_R8_UNORM,           CP_COLOR_R8_UNORM,            false },
   { VK_FORMAT_R16G16B16A16_SFLOAT, CP_TEXEL_R16G16B16A16_FLOAT, CP_COLOR_R16G16B16A16_FLOAT,  false },
   { VK_FORMAT_R32G32B32A32_SFLOAT, CP_TEXEL_R32G32B32A32_FLOAT, CP_COLOR_R32G32B32A32_FLOAT,  false },
   { VK_FORMAT_R32G32_SFLOAT,       CP_TEXEL_R32G32_FLOAT,       -1,                           false },
   { VK_FORMAT_R32_SFLOAT,          CP_TEXEL_R32_FLOAT,          -1,                           false },
   { VK_FORMAT_R16G16_SFLOAT,       0,                           CP_COLOR_R16G16_SFLOAT,       false },
   { VK_FORMAT_R16_SFLOAT,          0,                           CP_COLOR_R16_SFLOAT,          false },
   { VK_FORMAT_B10G11R11_UFLOAT_PACK32, CP_TEXEL_R11G11B10_FLOAT, CP_COLOR_R11G11B10_FLOAT,    false },
   { VK_FORMAT_A2B10G10R10_UNORM_PACK32, 0,                      CP_COLOR_A2B10G10R10_UNORM,   false },
   { VK_FORMAT_R5G6B5_UNORM_PACK16, CP_TEXEL_R5G6B5_UNORM,       -1,                           false },
   { VK_FORMAT_D32_SFLOAT,          0,                           -1,                           true  },
   { VK_FORMAT_D32_SFLOAT_S8_UINT,  0,                           -1,                           true  },
};

const struct cpvk_format_info *
cpvk_format_info(VkFormat format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(cpvk_formats); i++)
      if (cpvk_formats[i].vk == format)
         return &cpvk_formats[i];
   return NULL;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice,
                                        VkFormat format,
                                        VkFormatProperties2 *pFormatProperties)
{
   const struct cpvk_format_info *info = cpvk_format_info(format);
   VkFormatFeatureFlags linear = 0, buffer = 0;

   if (info) {
      if (info->texel)
         linear |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                   VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
      if (info->color >= 0)
         linear |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                   VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
      if (info->depth)
         linear |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   }

   pFormatProperties->formatProperties = (VkFormatProperties) {
      .linearTilingFeatures = linear,
      /* Optimal tiling is the same thing: the rasterizer walks pitch-linear
       * memory and `ncu` put every kernel's store far from the scattered
       * signature, so there is nothing a swizzle would buy that is measured. */
      .optimalTilingFeatures = linear,
      .bufferFeatures = buffer,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_GetPhysicalDeviceImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
   VkImageFormatProperties2 *pImageFormatProperties)
{
   const struct cpvk_format_info *info =
      cpvk_format_info(pImageFormatInfo->format);

   if (!info) {
      pImageFormatProperties->imageFormatProperties =
         (VkImageFormatProperties) { 0 };
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties) {
      .maxExtent = { 16384, 16384, 2048 },
      .maxMipLevels = 15,
      .maxArrayLayers = 2048,
      .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxResourceSize = 1ull << 31,
   };
   return VK_SUCCESS;
}

/* ------------------------------------------------------------------ images */

static void
cpvk_image_layout(struct cpvk_image *image)
{
   const struct vk_image *vk = &image->vk;
   unsigned bpp = vk_format_get_blocksize(vk->format);
   uint64_t offset = 0;

   /*
    * Samples are planes: sample n of the whole image follows sample n-1, and
    * sample_stride is how far apart they are. That is the layout the renderer
    * already assumes -- cp_fb_desc carries exactly one number for it -- and
    * it is the layout a resolve needs in order to find the samples it is
    * averaging.
    *
    * The sample count was ignored entirely until now, so a four-sample image
    * asked for a quarter of the memory it needs and everything that wrote to
    * it wrote past the end of what the application had allocated.
    */
   /*
    * In blocks, like the copies that fill these levels. `bpp` is the size of
    * a block, and for a block-compressed format a row is width/4 blocks, not
    * width of them -- computing it in texels made every stride and every
    * level sixteen times too large. The allocation was merely wasteful; the
    * stride was not, because it reached cuMemcpy2DAsync as a pitch far larger
    * than the memory behind it and came back as CUDA_ERROR_INVALID_VALUE.
    *
    * For an uncompressed format a block is one texel and this is unchanged.
    */
   enum pipe_format pfmt = vk_format_to_pipe_format(vk->format);

   for (unsigned l = 0; l < vk->mip_levels; l++) {
      unsigned w = util_format_get_nblocksx(pfmt, u_minify(vk->extent.width, l));
      unsigned h = util_format_get_nblocksy(pfmt, u_minify(vk->extent.height, l));
      unsigned d = u_minify(vk->extent.depth, l);

      image->row_stride[l] = align(w * bpp, 64);
      image->level_offset[l] = offset;
      image->level_size[l] = (uint64_t)image->row_stride[l] * h * d;
      offset += image->level_size[l] * vk->array_layers;
   }

   image->sample_stride = offset;
   image->size = MAX2(offset * MAX2(vk->samples, 1u), 1);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   if (pCreateInfo->mipLevels > CPVK_MAX_MIP_LEVELS)
      return vk_error(dev, VK_ERROR_FORMAT_NOT_SUPPORTED);

   struct cpvk_image *image =
      vk_image_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   const struct cpvk_format_info *info = cpvk_format_info(pCreateInfo->format);
   image->texel = info ? info->texel : 0;
   image->color = info ? info->color : -1;
   cpvk_image_layout(image);

   *pImage = cpvk_image_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyImage(VkDevice _device, VkImage _image,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_image, image, _image);

   if (image)
      vk_image_destroy(&dev->vk, pAllocator, &image->vk);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetImageMemoryRequirements2(VkDevice _device,
                                 const VkImageMemoryRequirementsInfo2 *pInfo,
                                 VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(cpvk_image, image, pInfo->image);

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements) {
      .size = image->size,
      .alignment = 256,
      .memoryTypeBits = 0x7,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_BindImageMemory2(VkDevice _device, uint32_t bindInfoCount,
                      const VkBindImageMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(cpvk_image, image, pBindInfos[i].image);
      VK_FROM_HANDLE(cpvk_device_memory, mem, pBindInfos[i].memory);

      image->mem = mem;
      image->offset = pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetImageSubresourceLayout(VkDevice _device, VkImage _image,
                               const VkImageSubresource *pSubresource,
                               VkSubresourceLayout *pLayout)
{
   VK_FROM_HANDLE(cpvk_image, image, _image);
   unsigned l = pSubresource->mipLevel;

   *pLayout = (VkSubresourceLayout) {
      .offset = image->level_offset[l] +
                (uint64_t)pSubresource->arrayLayer * image->level_size[l],
      .size = image->level_size[l],
      .rowPitch = image->row_stride[l],
      .arrayPitch = image->level_size[l],
      .depthPitch = image->level_size[l],
   };
}

/* The kernel's texture targets; cp_rast_types.h names them. */
static uint32_t
cpvk_tex_target(VkImageViewType t)
{
   switch (t) {
   case VK_IMAGE_VIEW_TYPE_1D:         return CP_TEX_1D;
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   return CP_TEX_1D_ARRAY;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   return CP_TEX_2D_ARRAY;
   case VK_IMAGE_VIEW_TYPE_3D:         return CP_TEX_3D;
   case VK_IMAGE_VIEW_TYPE_CUBE:       return CP_TEX_CUBE;
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return CP_TEX_CUBE_ARRAY;
   default:                            return CP_TEX_2D;
   }
}

/* ------------------------------------------------------------------- views */

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateImageView(VkDevice _device,
                     const VkImageViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_image_view *view =
      vk_image_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->image = cpvk_image_from_handle(pCreateInfo->image);

   /*
    * The sampler reaches a texture through one of these and nothing else: a
    * descriptor holds a pointer to it at CP_DESC_IMAGE_FUNCTIONS_OFFSET, and
    * that is the whole interface. Managed, because the host reads descriptors
    * when it specialises a shader on its sampler state.
    */
   struct cpvk_image *img = view->image;
   if (img) {
      cuCtxSetCurrent(dev->cu_ctx);
      if (cuMemAllocManaged(&view->tex_info, sizeof(struct cp_texture_info),
                            CU_MEM_ATTACH_GLOBAL) == CUDA_SUCCESS) {
         view->tex_info_host = (struct cp_texture_info *)(uintptr_t)view->tex_info;
         memset(view->tex_info_host, 0, sizeof(*view->tex_info_host));

         const VkImageSubresourceRange *r = &pCreateInfo->subresourceRange;
         enum pipe_format pfmt = vk_format_to_pipe_format(pCreateInfo->format);
         unsigned levels = MIN2(img->vk.mip_levels, CP_MAX_TEXTURE_LEVELS);

         *view->tex_info_host = (struct cp_texture_info) {
            .base = img->mem ? img->mem->dev_ptr + img->offset : 0,
            .width = img->vk.extent.width,
            .height = img->vk.extent.height,
            .depth = MAX2(img->vk.extent.depth, img->vk.array_layers),
            .format = pfmt,
            .target = cpvk_tex_target(pCreateInfo->viewType),
            .first_level = r->baseMipLevel,
            .last_level = r->baseMipLevel +
                          (r->levelCount == VK_REMAINING_MIP_LEVELS ?
                           img->vk.mip_levels - r->baseMipLevel :
                           r->levelCount) - 1,
            .first_layer = r->baseArrayLayer,
            .encoding = img->texel,
            .blocksize = util_format_get_blocksize(pfmt),
            .is_srgb = util_format_is_srgb(pfmt),
         };
         for (unsigned l = 0; l < levels; l++) {
            view->tex_info_host->row_stride[l] = img->row_stride[l];
            view->tex_info_host->img_stride[l] = img->level_size[l];
            view->tex_info_host->mip_offset[l] = img->level_offset[l];
         }

         if (cp_debug->debug_tex) {
            const struct cp_texture_info *ti = view->tex_info_host;
            fprintf(stderr, "cudapipe: texture handle %ux%u fmt=%u enc=%u "
                    "target=%u levels=%u..%u stride=%u base=%p\n",
                    ti->width, ti->height, ti->format, ti->encoding,
                    ti->target, ti->first_level, ti->last_level,
                    ti->row_stride[0], (void *)(uintptr_t)ti->base);
         }
      }
   }

   *pView = cpvk_image_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyImageView(VkDevice _device, VkImageView _view,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_image_view, view, _view);

   if (view)
      vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
}


/* ------------------------------------------------------------- samplers */

/*
 * The sampler kernel's own constants, which live in cp_sampler.cu because
 * NVRTC compiles that file from a stringified copy and it cannot include
 * anything. The values are pipe_tex_wrap's and pipe_tex_filter's, which is
 * why the Gallium adapter never needed a conversion and this does.
 * cp_sampler.cu is the source of truth; these mirror it.
 */
enum {
   CP_WRAP_REPEAT = 0,
   CP_WRAP_CLAMP,
   CP_WRAP_CLAMP_TO_EDGE,
   CP_WRAP_CLAMP_TO_BORDER,
   CP_WRAP_MIRROR_REPEAT,
   CP_WRAP_MIRROR_CLAMP,
   CP_WRAP_MIRROR_CLAMP_TO_EDGE,
   CP_WRAP_MIRROR_CLAMP_TO_BORDER,
};
enum { CP_FILTER_NEAREST = 0, CP_FILTER_LINEAR };
enum { CP_MIPFILTER_NEAREST = 0, CP_MIPFILTER_LINEAR, CP_MIPFILTER_NONE };

static uint32_t
cpvk_wrap(VkSamplerAddressMode m)
{
   switch (m) {
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:        return CP_WRAP_CLAMP_TO_EDGE;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:      return CP_WRAP_CLAMP_TO_BORDER;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:      return CP_WRAP_MIRROR_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: return CP_WRAP_MIRROR_CLAMP_TO_EDGE;
   default:                                           return CP_WRAP_REPEAT;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkSampler *pSampler)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   struct cp_context *cp = &dev->renderer;

   struct cpvk_sampler *sampler =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*sampler),
                       VK_OBJECT_TYPE_SAMPLER);
   if (!sampler)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct cp_sampler_info info = {
      .wrap_s = cpvk_wrap(pCreateInfo->addressModeU),
      .wrap_t = cpvk_wrap(pCreateInfo->addressModeV),
      .wrap_r = cpvk_wrap(pCreateInfo->addressModeW),
      .min_img_filter = pCreateInfo->minFilter == VK_FILTER_LINEAR ?
                        CP_FILTER_LINEAR : CP_FILTER_NEAREST,
      .mag_img_filter = pCreateInfo->magFilter == VK_FILTER_LINEAR ?
                        CP_FILTER_LINEAR : CP_FILTER_NEAREST,
      .min_mip_filter = pCreateInfo->mipmapMode ==
                        VK_SAMPLER_MIPMAP_MODE_LINEAR ?
                        CP_MIPFILTER_LINEAR : CP_MIPFILTER_NEAREST,
      .unnormalized_coords = pCreateInfo->unnormalizedCoordinates,
      .min_lod = pCreateInfo->minLod,
      .max_lod = pCreateInfo->maxLod,
      .lod_bias = pCreateInfo->mipLodBias,
      .max_anisotropy = pCreateInfo->anisotropyEnable ?
                        pCreateInfo->maxAnisotropy : 0.0f,
   };
   if (pCreateInfo->borderColor == VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE ||
       pCreateInfo->borderColor == VK_BORDER_COLOR_INT_OPAQUE_WHITE) {
      for (int i = 0; i < 4; i++)
         info.border_color[i] = 1.0f;
   } else if (pCreateInfo->borderColor == VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK ||
              pCreateInfo->borderColor == VK_BORDER_COLOR_INT_OPAQUE_BLACK) {
      info.border_color[3] = 1.0f;
   }

   /* Deduplicated, like the Gallium adapter's table: descriptors refer to
    * entries by index and identical states must land on one entry, because
    * the shader specialisation compares the state and not the index. */
   unsigned index = cp->num_samplers;
   for (unsigned i = 0; i < cp->num_samplers; i++) {
      if (!memcmp(&cp->sampler_table_host[i], &info, sizeof(info))) {
         index = i;
         break;
      }
   }
   if (index == cp->num_samplers) {
      if (cp->num_samplers >= CP_MAX_SAMPLERS) {
         vk_object_free(&dev->vk, pAllocator, sampler);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      cuCtxSetCurrent(dev->cu_ctx);
      if (!cp->sampler_table &&
          cuMemAllocManaged(&cp->sampler_table,
                            CP_MAX_SAMPLERS * sizeof(struct cp_sampler_info),
                            CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
         vk_object_free(&dev->vk, pAllocator, sampler);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      cp->sampler_table_host[index] = info;
      ((struct cp_sampler_info *)(uintptr_t)cp->sampler_table)[index] = info;
      cp->num_samplers++;
   }

   sampler->index = index;
   *pSampler = cpvk_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroySampler(VkDevice _device, VkSampler _sampler,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_sampler, sampler, _sampler);

   /* The table entry stays: descriptors already written refer to it by index
    * and nothing renumbers them. The table is bounded and per device. */
   if (sampler)
      vk_object_free(&dev->vk, pAllocator, sampler);
}
