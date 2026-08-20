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

struct cpvk_format_info {
   VkFormat vk;
   uint32_t texel;              /* enum cp_texel_format, 0 = not sampleable */
   int color;                   /* enum cp_color_encoding, -1 = not renderable */
   bool depth;
};

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

static const struct cpvk_format_info *
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

   for (unsigned l = 0; l < vk->mip_levels; l++) {
      unsigned w = u_minify(vk->extent.width, l);
      unsigned h = u_minify(vk->extent.height, l);
      unsigned d = u_minify(vk->extent.depth, l);

      image->row_stride[l] = align(w * bpp, 64);
      image->level_offset[l] = offset;
      image->level_size[l] = (uint64_t)image->row_stride[l] * h * d;
      offset += image->level_size[l] * vk->array_layers;
   }
   image->size = MAX2(offset, 1);
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
