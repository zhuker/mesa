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
   { VK_FORMAT_R8G8_UNORM,          CP_TEXEL_R8G8_UNORM,         CP_COLOR_R8G8_UNORM,          false },
   { VK_FORMAT_R8_UNORM,            CP_TEXEL_R8_UNORM,           CP_COLOR_R8_UNORM,            false },
   { VK_FORMAT_R16G16B16A16_SFLOAT, CP_TEXEL_R16G16B16A16_FLOAT, CP_COLOR_R16G16B16A16_FLOAT,  false },
   { VK_FORMAT_R32G32B32A32_SFLOAT, CP_TEXEL_R32G32B32A32_FLOAT, CP_COLOR_R32G32B32A32_FLOAT,  false },
   { VK_FORMAT_R32G32_SFLOAT,       CP_TEXEL_R32G32_FLOAT,       -1,                           false },
   { VK_FORMAT_R32_SFLOAT,          CP_TEXEL_R32_FLOAT,          -1,                           false },
   /*
    * These two decoded as texel encoding 0, which is R8G8B8A8_UNORM -- the
    * first enumerator, not a deliberate choice. The sampler has had the right
    * decode for both all along. pbribl's BRDF lookup table is R16G16_SFLOAT,
    * so every half pair came back as four bytes of near-zero, its whole
    * image-based lighting term evaluated to zero, and its spheres rendered
    * with direct light only: exactly neutral, against a reference that is
    * warm.
    */
   { VK_FORMAT_R16G16_SFLOAT,       CP_TEXEL_R16G16_SFLOAT,      CP_COLOR_R16G16_SFLOAT,       false },
   { VK_FORMAT_R16_SFLOAT,          CP_TEXEL_R16_SFLOAT,         CP_COLOR_R16_SFLOAT,          false },
   { VK_FORMAT_R16G16_UNORM,        CP_TEXEL_R16G16_UNORM,       CP_COLOR_R16G16_UNORM,        false },
   { VK_FORMAT_R32_SINT,            CP_TEXEL_R32_SINT,           -1,                           false },
   { VK_FORMAT_R16_SINT,            CP_TEXEL_R16_SINT,           -1,                           false },
   { VK_FORMAT_B10G11R11_UFLOAT_PACK32, CP_TEXEL_R11G11B10_FLOAT, CP_COLOR_R11G11B10_FLOAT,    false },
   { VK_FORMAT_A2B10G10R10_UNORM_PACK32, CP_TEXEL_A2B10G10R10_UNORM, CP_COLOR_A2B10G10R10_UNORM, false },
   { VK_FORMAT_R5G6B5_UNORM_PACK16, CP_TEXEL_R5G6B5_UNORM,       -1,                           false },
   /*
    * The one renderable integer format, and it is renderable only: there is
    * no CP_TEXEL_* decode for it, so it is a colour attachment and a transfer
    * end, not a texture. That is what the sampler can honestly do today --
    * an integer texel would come back through a float filter path -- and the
    * feature bits below say exactly that and no more.
    */
   { VK_FORMAT_R8G8B8A8_UINT,       CP_TEXEL_UNSUPPORTED,        CP_COLOR_R8G8B8A8_UINT,       false },
   { VK_FORMAT_BC1_RGB_UNORM_BLOCK,  CP_TEXEL_DXT1_RGB,           -1,                           false },
   { VK_FORMAT_BC1_RGB_SRGB_BLOCK,   CP_TEXEL_DXT1_RGB,           -1,                           false },
   { VK_FORMAT_BC1_RGBA_UNORM_BLOCK, CP_TEXEL_DXT1_RGBA,          -1,                           false },
   { VK_FORMAT_BC1_RGBA_SRGB_BLOCK,  CP_TEXEL_DXT1_RGBA,          -1,                           false },
   { VK_FORMAT_BC2_UNORM_BLOCK,      CP_TEXEL_DXT3_RGBA,          -1,                           false },
   { VK_FORMAT_BC2_SRGB_BLOCK,       CP_TEXEL_DXT3_RGBA,          -1,                           false },
   { VK_FORMAT_BC3_UNORM_BLOCK,      CP_TEXEL_DXT5_RGBA,          -1,                           false },
   { VK_FORMAT_BC3_SRGB_BLOCK,       CP_TEXEL_DXT5_RGBA,          -1,                           false },
   { VK_FORMAT_D32_SFLOAT,          CP_TEXEL_R32_FLOAT,           -1,                           true  },
   /*
    * The two combined depth/stencil formats have no texel decode: their depth
    * aspect is not a whole texel -- D24S8 packs 24 bits of depth beside a
    * stencil byte, D32S8 pads to eight -- and the sampler addresses texels,
    * not aspects. Spelled CP_TEXEL_UNSUPPORTED rather than 0 so that the set
    * of undecodable rows can be found by grep; a sampled view over one of
    * them is reported by cpvk_view_report_undecodable().
    */
   { VK_FORMAT_D32_SFLOAT_S8_UINT,  CP_TEXEL_UNSUPPORTED,         -1,                           true  },
   { VK_FORMAT_D24_UNORM_S8_UINT,   CP_TEXEL_UNSUPPORTED,         -1,                           true  },
   /*
    * D16 is a depth attachment *and* a texture. The row carried `texel = 0`
    * when the attachment half landed, on the grounds that the sampler did not
    * know the packing -- and a captured application samples one anyway. It
    * got opaque black for every lookup, 753 descriptor writes a replay, and
    * whole faces of the shadow-receiving geometry rendered near-black with
    * hard polygon edges (.audit/d16_gate3.md). The decode is two bytes over
    * 65535, exactly what cp_depth_attachment_store wrote, so
    * CP_TEXEL_R16_UNORM is the same number read back rather than a
    * reinterpretation.
    */
   { VK_FORMAT_D16_UNORM,           CP_TEXEL_R16_UNORM,           -1,                           true  },
};

const struct cpvk_format_info *
cpvk_format_info(VkFormat format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(cpvk_formats); i++)
      if (cpvk_formats[i].vk == format)
         return &cpvk_formats[i];
   return NULL;
}

/*
 * Storage images: what the backend's bindless load/store can address. It
 * computes the texel stride from the format and either packs 8-bit-per-channel
 * UNORM or moves the raw texel, so a plain uncompressed format of at most four
 * bytes works and a block-compressed or wider one does not.
 */
static bool
cpvk_format_storage(VkFormat format)
{
   const struct cpvk_format_info *info = cpvk_format_info(format);
   if (!info || !info->texel || info->depth)
      return false;
   enum pipe_format pfmt = vk_format_to_pipe_format(format);
   const struct util_format_description *desc = util_format_description(pfmt);
   return desc && desc->layout == UTIL_FORMAT_LAYOUT_PLAIN &&
          util_format_get_blocksize(pfmt) <= 4;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice,
                                        VkFormat format,
                                        VkFormatProperties2 *pFormatProperties)
{
   const struct cpvk_format_info *info = cpvk_format_info(format);
   VkFormatFeatureFlags linear = 0, buffer = 0;

   /*
    * Vertex buffers do not go through the image format table: the fetch
    * kernel converts any plain format whose channels are equal, whole bytes
    * and at most 32 bits wide, and copies anything else verbatim -- which is
    * not a conversion and must not be advertised.
    */
   const struct util_format_description *desc =
      util_format_description(vk_format_to_pipe_format(format));
   if (desc && desc->layout == UTIL_FORMAT_LAYOUT_PLAIN &&
       desc->nr_channels >= 1 && desc->channel[0].size % 8 == 0 &&
       desc->channel[0].size <= 32) {
      bool uniform = true;
      for (unsigned c = 1; c < desc->nr_channels; c++)
         uniform &= desc->channel[c].size == desc->channel[0].size &&
                    desc->channel[c].type == desc->channel[0].type &&
                    desc->channel[c].normalized == desc->channel[0].normalized;
      if (uniform)
         buffer |= VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
   }

   if (info) {
      if (info->texel) {
         enum pipe_format pfmt = vk_format_to_pipe_format(format);
         linear |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
         /* Integer payloads are bitcast by the sampler and cannot be
          * interpolated as floats. */
         if (!util_format_is_pure_integer(pfmt))
            linear |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
         /* Format feature bits make every advertised source/destination pair
          * legal. The implementation handles scaled same-format RGBA8 UNORM
          * and the RGBA/BGRA channel-order conversion between that pair. Keep
          * the advertised set to exactly that closed family; SRGB and packed
          * formats would require cross-format decode/encode the copy kernel
          * does not yet have. */
         if (format == VK_FORMAT_R8G8B8A8_UNORM ||
             format == VK_FORMAT_B8G8R8A8_UNORM)
            linear |= VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                      VK_FORMAT_FEATURE_BLIT_DST_BIT;
      }
      if (cpvk_format_storage(format))
         linear |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
      if (info->color >= 0) {
         linear |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
         /*
          * Blending is a float equation over colours and an integer
          * attachment has none, which is why Vulkan forbids blendEnable on
          * one. Not advertising the bit is what makes that refusal the
          * application's to obey rather than this driver's to discover:
          * cpvk_pipeline.c clears blend.enable for such an attachment as
          * well, so an application that ignores it still gets its bits
          * written rather than blended.
          */
         if (!util_format_is_pure_integer(vk_format_to_pipe_format(format)))
            linear |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
      }
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

   bool storage = cpvk_format_storage(pImageFormatInfo->format);
   /* Multisampling exists only for attachments this driver can allocate,
    * clear and resolve; a sampled or storage view of a multisample image has
    * no plane semantics here, so the answer for those usages is one. */
   bool attachment_usage = (pImageFormatInfo->usage &
      (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0;
   bool other_usage = (pImageFormatInfo->usage &
      ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)) != 0;
   /*
    * An integer attachment is single sample here. cp_resolve_samples averages
    * its sample planes, which is the one thing a resolve of an integer
    * attachment may not do -- Vulkan allows only SAMPLE_ZERO for those -- and
    * averaging bit patterns would produce a number no shader wrote. Offering
    * the sample counts and then resolving them wrongly is the failure this
    * table exists to avoid, so the honest answer is one.
    */
   bool integer_color =
      util_format_is_pure_integer(
         vk_format_to_pipe_format(pImageFormatInfo->format));
   bool msaa = info && attachment_usage && !other_usage &&
      (info->depth ||
       (info->color >= 0 && !integer_color &&
        vk_format_get_blocksize(pImageFormatInfo->format) == 4));
   if (!info ||
       ((pImageFormatInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        info->color < 0) ||
       ((pImageFormatInfo->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
        !info->depth) ||
       ((pImageFormatInfo->usage & VK_IMAGE_USAGE_STORAGE_BIT) && !storage) ||
       ((pImageFormatInfo->usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        !info->texel)) {
      pImageFormatProperties->imageFormatProperties =
         (VkImageFormatProperties) { 0 };
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties) {
      .maxExtent = { 16384, 16384, 2048 },
      .maxMipLevels = 15,
      .maxArrayLayers = 2048,
      .sampleCounts = msaa ? VK_SAMPLE_COUNT_1_BIT |
                              VK_SAMPLE_COUNT_4_BIT |
                              VK_SAMPLE_COUNT_8_BIT
                           : VK_SAMPLE_COUNT_1_BIT,
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

      /* The shared renderer addresses colour attachments as packed
       * pixel_index * blocksize.  Padding a native image row here made its
       * writes walk a different layout from copies and sampling: a 916-wide
       * atlas was rendered at 3664 bytes per row and read back at 3712,
       * shearing every row diagonally.  CUDA copies accept an arbitrary
       * tightly packed pitch, so keep the image and renderer layouts equal. */
      image->row_stride[l] = w * bpp;
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
   CPVK_CTX_SCOPE(dev);

   if (wsi_common_is_swapchain_image(pCreateInfo))
      return wsi_common_create_swapchain_image(&dev->pdev->wsi_device,
                                               pCreateInfo, pImage);

   const struct cpvk_format_info *info = cpvk_format_info(pCreateInfo->format);
   /*
    * Creation is deliberately permissive; the promise is in
    * vkGetPhysicalDeviceImageFormatProperties2, which refuses every format
    * and usage combination this driver cannot serve. Refusing here as well
    * breaks the two stored GFXReconstruct captures, which create a sampled
    * D16 image and a 4x multisample sampled depth image without ever asking:
    * vkCreateImage then fails, and the replayer dereferences the null image
    * it recorded rather than reporting the error. A sampled D16 image is now
    * a usage this driver serves; a 4x multisample sampled depth image is
    * still not. What the driver cannot do with such an image is still
    * refused where the work happens -- attachment binding, blit and resolve,
    * and the sampled/storage descriptor paths -- and a sampled view over a
    * format the sampler cannot decode is reported by
    * cpvk_view_report_undecodable() below rather than returned as black.
    */
   if (pCreateInfo->mipLevels > CPVK_MAX_MIP_LEVELS)
      return vk_error(dev, VK_ERROR_FORMAT_NOT_SUPPORTED);

   struct cpvk_image *image =
      vk_image_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->texel = info ? info->texel : 0;
   image->color = info ? info->color : -1;
   image->dev = dev;
   image->vk_format = pCreateInfo->format;
   image->writer_event = NULL;
   image->writer_event_valid = false;
   image->cache_alias = false;
   image->texture_cache = NULL;
   image->texture_cache_next = NULL;
   image->texture_cache_prev = NULL;
   image->views = NULL;
   atomic_init(&image->content_epoch, 1);
   const VkExternalMemoryImageCreateInfo *external =
      vk_find_struct_const(pCreateInfo->pNext,
                           EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   image->cache_create_eligible =
      !(pCreateInfo->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                              VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                              VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                              VK_IMAGE_CREATE_ALIAS_BIT |
                              VK_IMAGE_CREATE_PROTECTED_BIT)) &&
      (!external || !external->handleTypes) &&
      !(pCreateInfo->usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
      pCreateInfo->samples == VK_SAMPLE_COUNT_1_BIT;
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
   CPVK_CTX_SCOPE(dev);

   if (image) {
      cpvk_memory_note_unbind(dev, image->mem, image);
      if (cp_debug->texture_cache) {
         cpvk_DeviceWaitIdle(_device);
         cpvk_texture_cache_image_destroy(image);
      }
      vk_image_destroy(&dev->vk, pAllocator, &image->vk);
   }
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

static void cpvk_image_view_refresh(struct cpvk_image_view *view);

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_BindImageMemory2(VkDevice _device, uint32_t bindInfoCount,
                      const VkBindImageMemoryInfo *pBindInfos)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(cpvk_image, image, pBindInfos[i].image);
      VK_FROM_HANDLE(cpvk_device_memory, mem, pBindInfos[i].memory);

      image->mem = mem;
      image->offset = pBindInfos[i].memoryOffset;
      if (image->offset > mem->vk.size ||
          image->size > mem->vk.size - image->offset)
         image->cache_alias = true;
      cpvk_memory_note_bind(dev, mem, image->offset, image->size,
                            image, image);
      simple_mtx_lock(&dev->view_lock);
      for (struct cpvk_image_view *view = image->views; view;
           view = view->image_next)
         cpvk_image_view_refresh(view);
      simple_mtx_unlock(&dev->view_lock);
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
   enum pipe_format pfmt = vk_format_to_pipe_format(image->vk.format);
   unsigned h = util_format_get_nblocksy(
      pfmt, MAX2(image->vk.extent.height >> l, 1u));

   *pLayout = (VkSubresourceLayout) {
      .offset = image->level_offset[l] +
                (uint64_t)pSubresource->arrayLayer * image->level_size[l],
      .size = image->level_size[l],
      .rowPitch = image->row_stride[l],
      .arrayPitch = image->level_size[l],
      .depthPitch = (uint64_t)image->row_stride[l] * h,
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

static void
cpvk_image_view_refresh(struct cpvk_image_view *view)
{
   struct cpvk_image *img = view->image;
   if (!img || !view->tex_info_host)
      return;

   enum pipe_format pfmt = vk_format_to_pipe_format(view->vk.format);
   unsigned levels = MIN2(img->vk.mip_levels, CP_MAX_TEXTURE_LEVELS);
   *view->tex_info_host = (struct cp_texture_info) {
      .base = img->mem ? img->mem->dev_ptr + img->offset : 0,
      .width = img->vk.extent.width,
      .height = img->vk.extent.height,
      .depth = MAX2(img->vk.extent.depth, img->vk.array_layers),
      .format = pfmt,
      .target = cpvk_tex_target(view->vk.view_type),
      .first_level = view->vk.base_mip_level,
      .last_level = view->vk.base_mip_level + view->vk.level_count - 1,
      .first_layer = view->vk.base_array_layer,
      .encoding = img->texel,
      .blocksize = util_format_get_blocksize(pfmt),
      .is_srgb = util_format_is_srgb(pfmt),
   };
   for (unsigned l = 0; l < levels; l++) {
      view->tex_info_host->row_stride[l] = img->row_stride[l];
      unsigned h = util_format_get_nblocksy(
         pfmt, MAX2(img->vk.extent.height >> l, 1u));
      view->tex_info_host->img_stride[l] = img->row_stride[l] * h;
      view->tex_info_host->mip_offset[l] = img->level_offset[l];
   }
}

/*
 * The one thing worse than refusing a format is serving it wrongly, and this
 * is where that used to happen. An image whose format has no CP_TEXEL_*
 * decode reaches cp_fetch_texel with encoding CP_TEXEL_UNSUPPORTED, and that
 * function's `default:` arm returns opaque black -- deterministically, on a
 * device path where no diagnostic is possible. VK_FORMAT_D16_UNORM spent one
 * release like that: the driver allowed the image, allowed the view, allowed
 * 753 combined-image-sampler writes a replay, and returned 0.0 for every
 * shadow lookup, which rendered whole faces of the receiving geometry black
 * (.audit/d16_gate3.md). Nothing anywhere said so.
 *
 * So say so, once per format, on the host, where stderr exists. This is a
 * report rather than a refusal on purpose: an image created with SAMPLED
 * usage may still only ever be bound as an attachment, and both stored
 * GFXReconstruct captures create images this driver cannot fully serve
 * without ever asking whether it can -- refusing at vkCreateImage for
 * exactly that reason is what broke them once already, and the replayer
 * dereferences the null handle rather than reporting the error. The line is
 * unconditional: a flag would make it invisible to the next person, who by
 * construction does not know to set it.
 */
static void
cpvk_view_report_undecodable(const struct cpvk_image_view *view)
{
   const struct cpvk_image *img = view->image;
   if (!img || img->texel != CP_TEXEL_UNSUPPORTED ||
       !(img->vk.usage & VK_IMAGE_USAGE_SAMPLED_BIT))
      return;

   /* Once per format per process. The set is small by construction -- it is
    * the rows of cpvk_formats[] with no decode, plus whatever is not in the
    * table at all -- and a bound of eight has never been approached. */
   static simple_mtx_t reported_lock = SIMPLE_MTX_INITIALIZER;
   static VkFormat reported[8];
   static unsigned num_reported;

   simple_mtx_lock(&reported_lock);
   bool seen = false;
   for (unsigned i = 0; i < num_reported; i++)
      seen |= reported[i] == img->vk_format;
   if (!seen && num_reported < ARRAY_SIZE(reported))
      reported[num_reported++] = img->vk_format;
   simple_mtx_unlock(&reported_lock);
   if (seen)
      return;

   fprintf(stderr,
           "cudavk: sampled image view over format %u, which this driver "
           "cannot decode as a texture: every texture read of it returns "
           "opaque black (0,0,0,1). Add a CP_TEXEL_* decode in "
           "kernels/cp_sampler.cu and the row in cpvk_image.c to fix it.\n",
           (unsigned)img->vk_format);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateImageView(VkDevice _device,
                     const VkImageViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_image_view *view =
      vk_image_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->dev = dev;
   view->image = cpvk_image_from_handle(pCreateInfo->image);
   view->cache_swizzle_identity =
      pCreateInfo->components.r == VK_COMPONENT_SWIZZLE_IDENTITY &&
      pCreateInfo->components.g == VK_COMPONENT_SWIZZLE_IDENTITY &&
      pCreateInfo->components.b == VK_COMPONENT_SWIZZLE_IDENTITY &&
      pCreateInfo->components.a == VK_COMPONENT_SWIZZLE_IDENTITY;
   if (cuMemAllocManaged(&view->tex_info, sizeof(struct cp_texture_info),
                         CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
      vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   view->tex_info_host = (struct cp_texture_info *)(uintptr_t)view->tex_info;
   cpvk_image_view_refresh(view);
   cpvk_view_report_undecodable(view);

   if (view->image) {
      simple_mtx_lock(&dev->view_lock);
      view->image_next = view->image->views;
      view->image->views = view;
      simple_mtx_unlock(&dev->view_lock);
   }

   view->cache_cookie = 0;
   if (cp_debug->texture_cache) {
      simple_mtx_lock(&dev->texture_cache_lock);
      if (dev->texture_cache_view_count == dev->texture_cache_view_cap) {
         size_t old_cap = dev->texture_cache_view_cap;
         size_t new_cap = old_cap ? old_cap * 2 : 256;
         if (new_cap > old_cap &&
             new_cap <= SIZE_MAX / sizeof(*dev->texture_cache_views)) {
            void *grown = realloc(dev->texture_cache_views,
                                  new_cap * sizeof(*dev->texture_cache_views));
            if (grown) {
               dev->texture_cache_views = grown;
               memset(dev->texture_cache_views + old_cap, 0,
                      (new_cap - old_cap) * sizeof(*dev->texture_cache_views));
               dev->texture_cache_view_cap = new_cap;
            }
         }
      }
      if (dev->texture_cache_view_count < dev->texture_cache_view_cap) {
         size_t slot = dev->texture_cache_view_count++;
         dev->texture_cache_views[slot] = view;
         view->cache_cookie = slot + 1;
      }
      simple_mtx_unlock(&dev->texture_cache_lock);
   }

   if (cp_debug->debug_tex) {
      const struct cp_texture_info *ti = view->tex_info_host;
      fprintf(stderr, "cudavk: texture handle %p %ux%u fmt=%u enc=%u "
              "target=%u levels=%u..%u stride=%u base=%p\n",
              (void *)(uintptr_t)view->tex_info,
              ti->width, ti->height, ti->format, ti->encoding,
              ti->target, ti->first_level, ti->last_level,
              ti->row_stride[0], (void *)(uintptr_t)ti->base);
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
   CPVK_CTX_SCOPE(dev);

   if (view) {
      if (cp_debug->texture_cache) {
         cpvk_DeviceWaitIdle(_device);
         cpvk_texture_cache_view_destroy(view);
      }
      if (view->image) {
         simple_mtx_lock(&dev->view_lock);
         struct cpvk_image_view **link = &view->image->views;
         while (*link && *link != view)
            link = &(*link)->image_next;
         if (*link)
            *link = view->image_next;
         simple_mtx_unlock(&dev->view_lock);
      }
      if (view->tex_info) {
         cuMemFree(view->tex_info);
         view->tex_info = 0;
         view->tex_info_host = NULL;
      }
      vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
   }
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
   CPVK_CTX_SCOPE(dev);
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
      .compare_enable = pCreateInfo->compareEnable,
#ifdef VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT
      .non_seamless_cube =
         !!(pCreateInfo->flags & VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT),
#endif
   };
   const VkSamplerReductionModeCreateInfo *reduction =
      vk_find_struct_const(pCreateInfo->pNext, SAMPLER_REDUCTION_MODE_CREATE_INFO);
   info.reduction_mode = reduction ? reduction->reductionMode
                                   : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE;
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

   if (cp_debug->debug_tex)
      fprintf(stderr, "cudavk: sampler wrap=%u,%u filt=%u/%u mip=%u "
              "lod=%.1f..%.1f bias=%.1f aniso=%.1f -> index %u\n",
              info.wrap_s, info.wrap_t, info.min_img_filter,
              info.mag_img_filter, info.min_mip_filter, info.min_lod,
              info.max_lod, info.lod_bias, info.max_anisotropy, index);

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
   CPVK_CTX_SCOPE(dev);

   /* The table entry stays: descriptors already written refer to it by index
    * and nothing renumbers them. The table is bounded and per device. */
   if (sampler)
      vk_object_free(&dev->vk, pAllocator, sampler);
}
