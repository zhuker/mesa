#include "cp_screen.h"
#include "cp_context.h"
#include "cp_resource.h"
#include "kernels/cp_rast_types.h"

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_transfer_helper.h"
#include "util/u_surface.h"
#include "util/format/u_format.h"
#include "util/format/u_format_pack.h"

#include <cuda.h>
#include <math.h>
#include <string.h>
#include <stddef.h>


/*
 * Lay out every mip level and return the total size.
 *
 * Each level needs its own stride and offset: uploads locate their
 * destination through row_stride[level] and mip_offsets[level], so a level
 * left at zero would be written over the top of level 0.
 */
static uint64_t
cp_resource_layout(struct cp_resource *res, const struct pipe_resource *tmpl)
{
   if (tmpl->target == PIPE_BUFFER)
      return tmpl->width0;

   unsigned block_size = util_format_get_blocksize(tmpl->format);
   unsigned num_layers = tmpl->target == PIPE_TEXTURE_3D
      ? 1 : MAX2(tmpl->array_size, 1);
   uint64_t offset = 0;

   for (unsigned level = 0; level <= tmpl->last_level; level++) {
      unsigned width = u_minify(tmpl->width0, level);
      unsigned height = u_minify(tmpl->height0, level);
      unsigned depth = u_minify(MAX2(tmpl->depth0, 1), level);
      unsigned nblocksx = util_format_get_nblocksx(tmpl->format, width);
      unsigned nblocksy = util_format_get_nblocksy(tmpl->format, height);

      res->lpr.row_stride[level] = nblocksx * block_size;
      res->lpr.img_stride[level] = (uint64_t)res->lpr.row_stride[level] * nblocksy;
      res->lpr.mip_offsets[level] = offset;

      offset += res->lpr.img_stride[level] * depth * num_layers;
   }

   return offset;
}

static struct pipe_resource *
cp_resource_create(struct pipe_screen *screen,
                   const struct pipe_resource *tmpl)
{
   struct cp_resource *res = CALLOC_STRUCT(cp_resource);
   if (!res)
      return NULL;

   res->lpr.base = *tmpl;
   res->lpr.base.screen = screen;
   pipe_reference_init(&res->lpr.base.reference, 1);

   uint64_t size = cp_resource_layout(res, tmpl);

   /* Multisample surfaces store nr_samples copies of every pixel, one whole
    * plane after another, so a sample's plane starts at s * sample_stride. */
   unsigned nr_samples = MAX2(tmpl->nr_samples, 1);
   res->lpr.sample_stride = size;
   size *= nr_samples;

   if (size > 0) {
      CUresult err = cuMemAllocManaged(&res->device_ptr, size,
                                       CU_MEM_ATTACH_GLOBAL);
      if (err != CUDA_SUCCESS) {
         FREE(res);
         return NULL;
      }
      void *ptr = (void *)(uintptr_t)res->device_ptr;
      res->lpr.data = ptr;
      res->lpr.tex_data = ptr;
      res->cuda_managed = true;
      res->owns_data = true;
      cuMemsetD8(res->device_ptr, 0, size);
   }

   return &res->lpr.base;
}

static struct pipe_resource *
cp_resource_create_unbacked(struct pipe_screen *screen,
                            const struct pipe_resource *tmpl,
                            uint64_t *size_required)
{
   struct cp_screen *cp = (struct cp_screen *)screen;
   struct cp_resource *res = CALLOC_STRUCT(cp_resource);
   if (!res)
      return NULL;

   res->lpr.base = *tmpl;
   res->lpr.base.screen = screen;
   pipe_reference_init(&res->lpr.base.reference, 1);

   uint64_t size = cp_resource_layout(res, tmpl);

   /* Same sample layout as the backed path, and the same multiplication — a
    * multisample attachment created this way was reporting one sample's worth
    * of memory, so its later planes had nothing behind them. */
   res->lpr.sample_stride = size;
   size *= MAX2(tmpl->nr_samples, 1);

   if (size_required)
      *size_required = size;


   return &res->lpr.base;
}

static void
cp_resource_destroy(struct pipe_screen *screen, struct pipe_resource *pt)
{
   struct cp_resource *res = cp_resource(pt);
   if (res->owns_data) {
      if (res->cuda_managed && res->device_ptr)
         cuMemFree(res->device_ptr);
      else if (res->lpr.data)
         FREE(res->lpr.data);
   }
   FREE(res);
}

static void *
cp_buffer_map(struct pipe_context *ctx, struct pipe_resource *resource,
              unsigned level, unsigned usage, const struct pipe_box *box,
              struct pipe_transfer **out_transfer)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_resource *res = cp_resource(resource);
   struct pipe_transfer *transfer = CALLOC_STRUCT(pipe_transfer);
   if (!transfer)
      return NULL;

   /* If reading GPU-written data, ensure all kernels have finished. */
   if (!(usage & PIPE_MAP_DISCARD_WHOLE_RESOURCE) &&
       !(usage & PIPE_MAP_DISCARD_RANGE)) {
      cuCtxSetCurrent(cp->screen->cuda_ctx);
      cuCtxSynchronize();
   }

   transfer->resource = resource;
   transfer->level = level;
   transfer->usage = usage;
   transfer->box = *box;
   transfer->stride = res->lpr.row_stride[level];
   transfer->layer_stride = res->lpr.img_stride[level];

   *out_transfer = transfer;

   void *data = cp_resource_data(res);
   if (!data)
      return NULL;
   if (getenv("CUDAPIPE_DEBUG_DRAW") && (usage & PIPE_MAP_READ) &&
       resource->width0 * resource->height0 >= 921600)
      fprintf(stderr, "cudapipe: map READ %ux%u data=%p managed=%d tex=%d\n",
              resource->width0, resource->height0, data, res->cuda_managed,
              llvmpipe_resource_is_texture(&res->lpr.base));

   if (resource->target == PIPE_BUFFER)
      return (char *)data + box->x;

   uint64_t offset = res->lpr.mip_offsets[level] +
                     (uint64_t)box->z * res->lpr.img_stride[level] +
                     (uint64_t)box->y * res->lpr.row_stride[level] +
                     (uint64_t)box->x * util_format_get_blocksize(resource->format);
   return (char *)data + offset;
}

static void
cp_buffer_unmap(struct pipe_context *ctx, struct pipe_transfer *transfer)
{
   FREE(transfer);
}

static void
cp_resource_copy_region(struct pipe_context *ctx, struct pipe_resource *dst,
                        unsigned dst_level, unsigned dstx, unsigned dsty,
                        unsigned dstz, struct pipe_resource *src,
                        unsigned src_level, const struct pipe_box *src_box)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_resource *src_res = cp_resource(src);
   struct cp_resource *dst_res = cp_resource(dst);
   void *src_data = cp_resource_data(src_res);
   void *dst_data = cp_resource_data(dst_res);

   if (!src_data || !dst_data)
      return;

   unsigned src_stride = src_res->lpr.row_stride[src_level];
   unsigned dst_stride = dst_res->lpr.row_stride[dst_level];
   unsigned pixel_size = util_format_get_blocksize(src->format);

   if (!src_stride) src_stride = src->width0 * pixel_size;
   if (!dst_stride) dst_stride = dst->width0 * pixel_size;

   unsigned src_img_stride = src_res->lpr.img_stride[src_level];
   unsigned dst_img_stride = dst_res->lpr.img_stride[dst_level];

   cuCtxSetCurrent(cp->screen->cuda_ctx);
   cuCtxSynchronize();

   /*
    * Each mip level sits at its own offset inside the resource. Leaving that
    * out does not merely lose the small levels: every copy lands on level 0
    * instead, so a mip chain built by copying into successive levels ends up
    * with one level written over and over and the rest untouched. It stays
    * invisible until something samples above level 0.
    */
   unsigned src_off = src_res->lpr.mip_offsets[src_level];
   unsigned dst_off = dst_res->lpr.mip_offsets[dst_level];

   for (int z = 0; z < src_box->depth; z++) {
      char *s = (char *)src_data + src_off +
                (src_box->z + z) * src_img_stride +
                (unsigned)src_box->y * src_stride +
                (unsigned)src_box->x * pixel_size;
      char *d = (char *)dst_data + dst_off +
                (dstz + z) * dst_img_stride +
                dsty * dst_stride +
                dstx * pixel_size;
      unsigned row_bytes = (unsigned)src_box->width * pixel_size;
      for (int y = 0; y < src_box->height; y++) {
         memcpy(d + y * dst_stride, s + y * src_stride, row_bytes);
      }
   }
}

static void
cp_blit(struct pipe_context *ctx, const struct pipe_blit_info *info)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_resource *src_res = cp_resource(info->src.resource);
   struct cp_resource *dst_res = cp_resource(info->dst.resource);
   void *src_data = cp_resource_data(src_res);
   void *dst_data = cp_resource_data(dst_res);
   if (getenv("CUDAPIPE_DEBUG_DRAW"))
      fprintf(stderr, "cudapipe: blit %ux%u -> %ux%u src=%p dst=%p "
              "srcsamples=%u dstsamples=%u fmt=%u->%u\n",
              info->src.box.width, info->src.box.height,
              info->dst.box.width, info->dst.box.height, src_data, dst_data,
              info->src.resource->nr_samples, info->dst.resource->nr_samples,
              info->src.format, info->dst.format);
   if (!src_data || !dst_data)
      return;

   unsigned src_stride = src_res->lpr.row_stride[info->src.level];
   unsigned dst_stride = dst_res->lpr.row_stride[info->dst.level];
   unsigned src_pixel_size = util_format_get_blocksize(info->src.resource->format);
   unsigned dst_pixel_size = util_format_get_blocksize(info->dst.resource->format);

   unsigned src_img_stride = src_res->lpr.img_stride[info->src.level];
   unsigned dst_img_stride = dst_res->lpr.img_stride[info->dst.level];

   if (!src_stride) src_stride = info->src.resource->width0 * src_pixel_size;
   if (!dst_stride) dst_stride = info->dst.resource->width0 * dst_pixel_size;

   int src_w = info->src.box.width;
   int src_h = info->src.box.height;
   int dst_w = info->dst.box.width;
   int dst_h = info->dst.box.height;

   /*
    * Resolve: a multisample source read into a single-sample destination is
    * the average of the sample planes, which is the whole point of rendering
    * multisampled in the first place. Same format and size, so it is the
    * copy below with an averaging step in front of it.
    */
   unsigned src_samples = MAX2(info->src.resource->nr_samples, 1u);
   unsigned dst_samples = MAX2(info->dst.resource->nr_samples, 1u);
   if (src_samples > 1 && dst_samples == 1 &&
       info->src.format == info->dst.format &&
       src_w == dst_w && src_h == dst_h &&
       src_res->cuda_managed && dst_res->cuda_managed &&
       cp->screen->kernels.resolve_samples) {
      cuCtxSetCurrent(cp->screen->cuda_ctx);
      struct cp_resolve_msaa_args ra = {
         .src = (uint64_t)(uintptr_t)src_data,
         .dst = (uint64_t)(uintptr_t)dst_data,
         .width = (uint32_t)src_w,
         .height = (uint32_t)src_h,
         .src_stride = src_stride,
         .dst_stride = dst_stride,
         .sample_stride = (uint32_t)src_res->lpr.sample_stride,
         .num_samples = src_samples,
         .encoding = cp_color_encoding_from_format(info->src.format),
      };
      if (getenv("CUDAPIPE_DEBUG_DRAW"))
         fprintf(stderr, "  resolve %ux%u samples=%u sstride=%u srcstride=%u "
                 "dststride=%u enc=%d\n", ra.width, ra.height, ra.num_samples,
                 ra.sample_stride, ra.src_stride, ra.dst_stride, ra.encoding);
      void *params[] = { &ra };
      cuLaunchKernel(cp->screen->kernels.resolve_samples,
                     (src_w + 15) / 16, (src_h + 15) / 16, 1, 16, 16, 1,
                     0, NULL, params, NULL);
      return;
   }

   /* Same format and same size */
   if (info->src.format == info->dst.format && src_w == dst_w && src_h == dst_h) {
      cuCtxSetCurrent(cp->screen->cuda_ctx);
      /* If both are CUDA-managed, use GPU copy (stays in stream order).
       * Otherwise sync and memcpy (one side is host-only memory). */
      if (src_res->cuda_managed && dst_res->cuda_managed) {
         for (int z = 0; z < info->src.box.depth; z++) {
            CUDA_MEMCPY2D copy = {0};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = (CUdeviceptr)(uintptr_t)src_data +
                             src_res->lpr.mip_offsets[info->src.level] +
                             (info->src.box.z + z) * src_img_stride +
                             (unsigned)info->src.box.y * src_stride +
                             (unsigned)info->src.box.x * src_pixel_size;
            copy.srcPitch = src_stride;
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = (CUdeviceptr)(uintptr_t)dst_data +
                             dst_res->lpr.mip_offsets[info->dst.level] +
                             (info->dst.box.z + z) * dst_img_stride +
                             (unsigned)info->dst.box.y * dst_stride +
                             (unsigned)info->dst.box.x * dst_pixel_size;
            copy.dstPitch = dst_stride;
            copy.WidthInBytes = (unsigned)src_w * src_pixel_size;
            copy.Height = (unsigned)src_h;
            cuMemcpy2D(&copy);
         }
      } else {
         cuCtxSynchronize();
         for (int z = 0; z < info->src.box.depth; z++) {
            char *s = (char *)src_data +
                      src_res->lpr.mip_offsets[info->src.level] +
                      (info->src.box.z + z) * src_img_stride +
                      (unsigned)info->src.box.y * src_stride +
                      (unsigned)info->src.box.x * src_pixel_size;
            char *d = (char *)dst_data +
                      dst_res->lpr.mip_offsets[info->dst.level] +
                      (info->dst.box.z + z) * dst_img_stride +
                      (unsigned)info->dst.box.y * dst_stride +
                      (unsigned)info->dst.box.x * dst_pixel_size;
            unsigned row_bytes = (unsigned)src_w * src_pixel_size;
            for (int y = 0; y < src_h; y++)
               memcpy(d + y * dst_stride, s + y * src_stride, row_bytes);
         }
      }
      return;
   }

   /* Anything else is done on the CPU. Sync first — preceding GPU kernels may
    * have written src_data. */
   cuCtxSetCurrent(cp->screen->cuda_ctx);
   cuCtxSynchronize();

   /* Same size but a different format: a straight format conversion. This is
    * the path a readback takes, where an application blits its B8G8R8A8
    * render target into an R8G8B8A8 staging image and expects the blit to
    * reorder the channels for it. */
   if (src_w == dst_w && src_h == dst_h) {
      for (int z = 0; z < info->src.box.depth; z++) {
         const char *s = (const char *)src_data +
                         src_res->lpr.mip_offsets[info->src.level] +
                         (info->src.box.z + z) * src_img_stride;
         char *d = (char *)dst_data +
                   dst_res->lpr.mip_offsets[info->dst.level] +
                   (info->dst.box.z + z) * dst_img_stride;
         util_format_translate(info->dst.format, d, dst_stride,
                               info->dst.box.x, info->dst.box.y,
                               info->src.format, s, src_stride,
                               info->src.box.x, info->src.box.y,
                               src_w, src_h);
      }
      return;
   }

   /*
    * Different size. A linear blit is not a nicety here: an application builds
    * its mip chain by blitting each level into the next, so dropping to
    * nearest decimates instead of averaging and the error compounds down the
    * chain — some levels alias close to the right answer and others badly,
    * which is what banded agreement across a mipmapped surface looks like.
    *
    * Work in float RGBA so the filtering is done once, in one place, whatever
    * the two formats are.
    */
   bool linear = info->filter == PIPE_TEX_FILTER_LINEAR &&
                 !util_format_is_pure_integer(info->src.format);

   void *row = malloc((size_t)dst_w * src_pixel_size);
   float (*taps)[4] = linear ? malloc(sizeof(*taps) * (size_t)src_w * 2) : NULL;
   float (*out_rgba)[4] = linear ? malloc(sizeof(*out_rgba) * (size_t)dst_w) : NULL;
   if (!row || (linear && (!taps || !out_rgba))) {
      free(row);
      free(taps);
      free(out_rgba);
      return;
   }

   for (int z = 0; z < info->dst.box.depth; z++) {
      int sz = info->src.box.depth > 1
         ? info->src.box.z + z * info->src.box.depth / info->dst.box.depth
         : info->src.box.z;
      for (int y = 0; y < dst_h; y++) {
         int sy = src_h == dst_h ? y : y * src_h / dst_h;
         const char *s = (const char *)src_data +
                         src_res->lpr.mip_offsets[info->src.level] +
                         sz * src_img_stride +
                         (info->src.box.y + sy) * src_stride +
                         (unsigned)info->src.box.x * src_pixel_size;

         if (linear) {
            /* Sample at pixel centres, matching the usual blit convention. */
            float fy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
            int y0 = (int)floorf(fy);
            float wy = fy - (float)y0;
            int y1 = CLAMP(y0 + 1, 0, src_h - 1);
            y0 = CLAMP(y0, 0, src_h - 1);

            const char *r0 = (const char *)src_data +
                             src_res->lpr.mip_offsets[info->src.level] +
                             sz * src_img_stride +
                             (info->src.box.y + y0) * src_stride +
                             (unsigned)info->src.box.x * src_pixel_size;
            const char *r1 = (const char *)src_data +
                             src_res->lpr.mip_offsets[info->src.level] +
                             sz * src_img_stride +
                             (info->src.box.y + y1) * src_stride +
                             (unsigned)info->src.box.x * src_pixel_size;
            util_format_unpack_rgba(info->src.format, taps, r0, src_w);
            util_format_unpack_rgba(info->src.format, taps + src_w, r1, src_w);

            for (int x = 0; x < dst_w; x++) {
               float fx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
               int x0 = (int)floorf(fx);
               float wx = fx - (float)x0;
               int x1 = CLAMP(x0 + 1, 0, src_w - 1);
               x0 = CLAMP(x0, 0, src_w - 1);

               for (int c = 0; c < 4; c++) {
                  float a = taps[x0][c] + (taps[x1][c] - taps[x0][c]) * wx;
                  float b = taps[src_w + x0][c] +
                            (taps[src_w + x1][c] - taps[src_w + x0][c]) * wx;
                  out_rgba[x][c] = a + (b - a) * wy;
               }
            }

            char *d = (char *)dst_data +
                      dst_res->lpr.mip_offsets[info->dst.level] +
                      (info->dst.box.z + z) * dst_img_stride;
            util_format_pack_rgba(info->dst.format,
                                  d + (size_t)(info->dst.box.y + y) * dst_stride +
                                  (size_t)info->dst.box.x *
                                     util_format_get_blocksize(info->dst.format),
                                  out_rgba, dst_w);
            continue;
         }

         for (int x = 0; x < dst_w; x++) {
            int sx = src_w == dst_w ? x : x * src_w / dst_w;
            memcpy((char *)row + (unsigned)x * src_pixel_size,
                   s + (unsigned)sx * src_pixel_size, src_pixel_size);
         }

         char *d = (char *)dst_data +
                   dst_res->lpr.mip_offsets[info->dst.level] +
                   (info->dst.box.z + z) * dst_img_stride;
         util_format_translate(info->dst.format, d, dst_stride,
                               info->dst.box.x, info->dst.box.y + y,
                               info->src.format, row, 0, 0, 0,
                               dst_w, 1);
      }
   }

   free(row);
   free(taps);
   free(out_rgba);
}

static void
cp_clear_buffer(struct pipe_context *ctx, struct pipe_resource *res,
                unsigned offset, unsigned size,
                const void *clear_value, int clear_value_size)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_resource *cp_res = cp_resource(res);
   void *data = cp_resource_data(cp_res);
   if (!data)
      return;

   CUdeviceptr dev = (CUdeviceptr)(uintptr_t)data + offset;
   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* Use cuMemsetD32 for common 4-byte patterns, cuMemsetD8 for 1-byte */
   if (clear_value_size == 4) {
      uint32_t val;
      memcpy(&val, clear_value, 4);
      cuMemsetD32(dev, val, size / 4);
   } else if (clear_value_size == 1) {
      cuMemsetD8(dev, *(const unsigned char *)clear_value, size);
   } else {
      char *dst = (char *)data + offset;
      for (unsigned i = 0; i < size; i += clear_value_size)
         memcpy(dst + i, clear_value, clear_value_size);
   }
}

static void
cp_clear_render_target(struct pipe_context *ctx, struct pipe_surface *dst,
                       const union pipe_color_union *color,
                       unsigned dstx, unsigned dsty,
                       unsigned width, unsigned height,
                       bool render_condition_enabled)
{
   if (!dst || !dst->texture)
      return;
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_resource *res = cp_resource(dst->texture);
   void *data = cp_resource_data(res);
   if (!data)
      return;

   unsigned pixel_size = util_format_get_blocksize(dst->format);
   unsigned stride = res->lpr.row_stride[dst->level];

   struct cp_clear_args args = {
      .target = (uint64_t)(uintptr_t)data,
      .width = width, .height = height,
      .stride = stride,
      .pixel_size = pixel_size,
   };
   util_format_pack_rgba(dst->format, args.clear_value, color, 1);

   cuCtxSetCurrent(cp->screen->cuda_ctx);
   void *params[] = { &args };
   cuLaunchKernel(cp->screen->kernels.clear_kernel,
      (width + 15) / 16, (height + 15) / 16, 1,
      16, 16, 1,
      0, NULL, params, NULL);
}

static void
cp_clear_depth_stencil(struct pipe_context *ctx, struct pipe_surface *dst,
                       unsigned clear_flags, double depth, unsigned stencil,
                       unsigned dstx, unsigned dsty,
                       unsigned width, unsigned height,
                       bool render_condition_enabled)
{
   if (!dst || !dst->texture)
      return;
   struct cp_resource *res = cp_resource(dst->texture);
   void *data = cp_resource_data(res);
   if (!data)
      return;

   unsigned pixel_size = util_format_get_blocksize(dst->format);
   unsigned stride = res->lpr.row_stride[dst->level];

   uint32_t clear_val = 0;
   if (clear_flags & PIPE_CLEAR_DEPTH) {
      if (pixel_size == 4) {
         float f = (float)depth;
         memcpy(&clear_val, &f, 4);
      } else {
         clear_val = (uint32_t)(depth * 65535.0);
      }
   }

   for (unsigned y = dsty; y < dsty + height; y++) {
      char *row = (char *)data + y * stride + dstx * pixel_size;
      for (unsigned x = 0; x < width; x++)
         memcpy(row + x * pixel_size, &clear_val, pixel_size);
   }
}

static void
cp_clear_texture(struct pipe_context *ctx, struct pipe_resource *res,
                 unsigned level, const struct pipe_box *box, const void *data)
{
   struct cp_resource *cp_res = cp_resource(res);
   void *tex_data = cp_resource_data(cp_res);
   if (!tex_data)
      return;

   unsigned pixel_size = util_format_get_blocksize(res->format);
   unsigned stride = cp_res->lpr.row_stride[level];
   unsigned img_stride = cp_res->lpr.img_stride[level];

   for (int z = box->z; z < box->z + box->depth; z++) {
      for (int y = box->y; y < box->y + box->height; y++) {
         char *row = (char *)tex_data + z * img_stride + y * stride + box->x * pixel_size;
         for (int x = 0; x < box->width; x++) {
            memcpy(row + x * pixel_size, data, pixel_size);
         }
      }
   }
}

static void
cp_clear(struct pipe_context *ctx, unsigned buffers,
         uint32_t color_clear_mask, uint8_t stencil_clear_mask,
         const struct pipe_scissor_state *scissor,
         const union pipe_color_union *color, double depth,
         unsigned stencil)
{
   struct cp_context *cp_ctx = (struct cp_context *)ctx;
   struct cp_screen *screen = cp_ctx->screen;
   struct pipe_framebuffer_state *fb = &cp_ctx->framebuffer;

   cuCtxSetCurrent(screen->cuda_ctx);

   if (getenv("CUDAPIPE_DEBUG_DRAW"))
      fprintf(stderr, "cudapipe: clear buffers=0x%x color=[%.2f,%.2f,%.2f,%.2f]\n",
              buffers, color ? color->f[0] : 0, color ? color->f[1] : 0,
              color ? color->f[2] : 0, color ? color->f[3] : 0);

   /* Clear color attachments */
   if ((buffers & PIPE_CLEAR_COLOR) && color && screen->kernels.clear_kernel) {
      for (unsigned i = 0; i < fb->nr_cbufs; i++) {
         if (!(color_clear_mask & (1 << i)))
            continue;

         struct pipe_surface *surf = &fb->cbufs[i];
         if (!surf->texture)
            continue;

         struct cp_resource *res = cp_resource(surf->texture);
         void *data = cp_resource_data(res);
         if (!data)
            continue;

         unsigned w = fb->width;
         unsigned h = fb->height;
         unsigned pixel_size = util_format_get_blocksize(surf->format);
         unsigned stride = res->lpr.row_stride[surf->level];

         struct cp_clear_args args = {
            .target = (uint64_t)(uintptr_t)data,
            .width = w, .height = h,
            .stride = stride,
            .pixel_size = pixel_size,
         };

         /* Pack clear color */
         union pipe_color_union clamped = *color;
         util_format_pack_rgba(surf->format, args.clear_value, &clamped, 1);

         /* Every sample plane, or a multisample attachment keeps whatever was
          * left in samples 1..n-1 and the resolve averages it in. */
         unsigned samples = MAX2(surf->texture->nr_samples, 1u);
         for (unsigned smp = 0; smp < samples; smp++) {
            args.target = (uint64_t)(uintptr_t)data +
                          (uint64_t)smp * res->lpr.sample_stride;
            void *params[] = { &args };
            cuLaunchKernel(screen->kernels.clear_kernel,
               (w + 15) / 16, (h + 15) / 16, 1,
               16, 16, 1,
               0, NULL, params, NULL);
         }
      }
   }

   /* The rasterizer tests against its own depth buffer, so clear that too —
    * not just the application's depth attachment. */
   if (buffers & PIPE_CLEAR_DEPTH)
      cp_clear_depthbuf(cp_ctx, (float)depth);

   /* Clear depth */
   if ((buffers & PIPE_CLEAR_DEPTH) && fb->zsbuf.texture && screen->kernels.clear_depth_kernel) {
      struct pipe_surface *surf = &fb->zsbuf;
      struct cp_resource *res = cp_resource(surf->texture);
      void *data = cp_resource_data(res);
      if (data) {
         unsigned w = fb->width;
         unsigned h = fb->height;
         unsigned pixel_size = util_format_get_blocksize(surf->format);
         unsigned stride = res->lpr.row_stride[surf->level];

         struct cp_clear_args args = {
            .target = (uint64_t)(uintptr_t)data,
            .width = w, .height = h,
            .stride = stride,
            .pixel_size = pixel_size,
         };

         /* Pack depth as appropriate format */
         if (pixel_size == 4) {
            float f = (float)depth;
            memcpy(&args.clear_value[0], &f, 4);
         } else if (pixel_size == 2) {
            args.clear_value[0] = (uint32_t)(depth * 65535.0);
         }

         void *params[] = { &args };
         cuLaunchKernel(screen->kernels.clear_depth_kernel,
            (w + 15) / 16, (h + 15) / 16, 1,
            16, 16, 1,
            0, NULL, params, NULL);
      }
   }

   cuCtxSynchronize();
}

/*
 * Backing for VkDeviceMemory. This has to be memory the GPU can read: the
 * vertex fetch and shader kernels dereference application buffers directly,
 * so host-only memory here faults the kernel with CUDA_ERROR_ILLEGAL_ADDRESS.
 *
 * lavapipe allocates from its submit thread, which has no current context of
 * its own, hence the cuCtxSetCurrent. A context may be current on several
 * threads at once, so binding it here doesn't disturb the main thread.
 */
static struct pipe_memory_allocation *
cp_allocate_memory(struct pipe_screen *screen, uint64_t size)
{
   struct cp_screen *cp = (struct cp_screen *)screen;
   CUdeviceptr dev = 0;

   cuCtxSetCurrent(cp->cuda_ctx);
   if (cuMemAllocManaged(&dev, size, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS)
      return NULL;

   cuMemsetD8(dev, 0, size);
   return (struct pipe_memory_allocation *)(uintptr_t)dev;
}

static void
cp_free_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
   struct cp_screen *cp = (struct cp_screen *)screen;

   if (!mem)
      return;

   cuCtxSetCurrent(cp->cuda_ctx);
   cuMemFree((CUdeviceptr)(uintptr_t)mem);
}

static void *
cp_map_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
   return (void *)mem;
}

static void
cp_unmap_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
}

static bool
cp_resource_bind_backing(struct pipe_screen *screen, struct pipe_resource *pt,
                         struct pipe_memory_allocation *mem, uint64_t fd_offset,
                         uint64_t size, uint64_t mem_offset)
{
   struct cp_resource *res = cp_resource(pt);
   void *ptr = (char *)mem + mem_offset;
   res->lpr.data = ptr;
   res->lpr.tex_data = ptr;
   /* cp_allocate_memory hands out managed memory, so the GPU paths (blit,
    * vertex fetch) can use this resource directly. The allocation itself is
    * owned by the VkDeviceMemory, not by the resource. */
   res->cuda_managed = true;
   res->device_ptr = (CUdeviceptr)(uintptr_t)ptr;
   return true;
}

static bool
cp_resource_get_param(struct pipe_screen *screen, struct pipe_context *context,
                      struct pipe_resource *resource, unsigned plane,
                      unsigned layer, unsigned level,
                      enum pipe_resource_param param, unsigned handle_usage,
                      uint64_t *value)
{
   struct cp_resource *res = cp_resource(resource);

   switch (param) {
   case PIPE_RESOURCE_PARAM_STRIDE:
      *value = res->lpr.row_stride[level];
      return true;
   case PIPE_RESOURCE_PARAM_OFFSET:
      *value = res->lpr.mip_offsets[level] + layer * res->lpr.img_stride[level];
      return true;
   case PIPE_RESOURCE_PARAM_LAYER_STRIDE:
      *value = res->lpr.img_stride[level];
      return true;
   default:
      *value = 0;
      return false;
   }
}

void
cudapipe_init_screen_resource_funcs(struct pipe_screen *screen)
{
   screen->resource_create = cp_resource_create;
   screen->resource_create_unbacked = cp_resource_create_unbacked;
   screen->resource_destroy = cp_resource_destroy;
   screen->allocate_memory = cp_allocate_memory;
   screen->free_memory = cp_free_memory;
   screen->resource_bind_backing = cp_resource_bind_backing;
   screen->resource_get_param = cp_resource_get_param;
   screen->map_memory = cp_map_memory;
   screen->unmap_memory = cp_unmap_memory;
}

void
cudapipe_init_context_resource_funcs(struct pipe_context *ctx)
{
   ctx->buffer_map = cp_buffer_map;
   ctx->buffer_unmap = cp_buffer_unmap;
   ctx->texture_map = cp_buffer_map;
   ctx->texture_unmap = cp_buffer_unmap;
   ctx->resource_copy_region = cp_resource_copy_region;
   ctx->blit = cp_blit;
   ctx->clear = cp_clear;
   ctx->clear_buffer = cp_clear_buffer;
   ctx->clear_texture = cp_clear_texture;
   ctx->clear_render_target = cp_clear_render_target;
   ctx->clear_depth_stencil = cp_clear_depth_stencil;
}
