#include "cp_screen.h"
#include "cp_context.h"
#include "cp_resource.h"
#include "cp_debug.h"
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
#include "util/simple_mtx.h"
#include "util/u_atomic.h"

#include <cuda.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <stdlib.h>


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

/*
 * Zero a freshly made managed allocation.
 *
 * `cuMemsetD8` is the blocking variant, and on a small buffer essentially all
 * of what it costs is the host round trip rather than the write: 64 bytes
 * measured at 31 us average, and 37 us at the call. `dynamicuniformbuffer`
 * makes 764 transient resources a frame — a dynamic UBO slice per draw — and
 * paid that for every one, which put `cuMemsetD8_v2` at **28.7% of all
 * host-side CUDA API time**, ahead of `cuLaunchKernel`. It was the largest
 * single memory-operation cost in the driver and none of it was bandwidth.
 *
 * The memory is managed and was allocated a moment ago, so its pages are
 * host-resident and the host can simply write it. That is nanoseconds at these
 * sizes, and it leaves the pages where the caller is about to write them
 * anyway — a small resource is one the host fills.
 *
 * Past the threshold the trade reverses: a megabyte-scale clear is real
 * bandwidth, the device does it an order of magnitude faster, and pulling the
 * whole allocation to the host to zero it is exactly the migration the rest of
 * this driver has spent two passes removing. So large allocations keep the
 * device memset.
 */
#define CP_HOST_ZERO_MAX ((uint64_t)64 * 1024)

static void
cp_zero_managed(CUdeviceptr ptr, uint64_t size)
{
   if (size <= CP_HOST_ZERO_MAX)
      memset((void *)(uintptr_t)ptr, 0, size);
   else
      cuMemsetD8(ptr, 0, size);
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
      cp_zero_managed(res->device_ptr, size);
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

   /* A map is where the host looks at what the GPU drew, so any draws still
    * being held back for merging have to be submitted before the sync below
    * — otherwise the readback waits for a queue they were never put on. */
   cp_batch_flush(cp);

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
   cp_batch_flush(cp);
   void *src_data = cp_resource_data(src_res);
   void *dst_data = cp_resource_data(dst_res);

   if (!src_data || !dst_data)
      return;

   unsigned src_stride = src_res->lpr.row_stride[src_level];
   unsigned dst_stride = dst_res->lpr.row_stride[dst_level];

   /*
    * The unit of a copy is a block, not a pixel. They are the same thing for
    * an uncompressed format, but a BC1 block is 8 bytes covering 4x4 pixels,
    * so measuring the box in pixels reads and writes sixteen times the real
    * extent — far enough past the end of the allocation to fault. The box
    * itself is in pixels, hence the conversion here rather than at the call.
    */
   unsigned block_size = util_format_get_blocksize(src->format);
   unsigned blocks_w = util_format_get_nblocksx(src->format, src_box->width);
   unsigned blocks_h = util_format_get_nblocksy(src->format, src_box->height);
   unsigned src_bx = util_format_get_nblocksx(src->format, src_box->x);
   unsigned src_by = util_format_get_nblocksy(src->format, src_box->y);
   unsigned dst_bx = util_format_get_nblocksx(dst->format, dstx);
   unsigned dst_by = util_format_get_nblocksy(dst->format, dsty);

   if (!src_stride)
      src_stride = util_format_get_nblocksx(src->format, src->width0) * block_size;
   if (!dst_stride)
      dst_stride = util_format_get_nblocksx(dst->format, dst->width0) * block_size;

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
                src_by * src_stride +
                src_bx * block_size;
      char *d = (char *)dst_data + dst_off +
                (dstz + z) * dst_img_stride +
                dst_by * dst_stride +
                dst_bx * block_size;
      unsigned row_bytes = blocks_w * block_size;
      for (unsigned y = 0; y < blocks_h; y++) {
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
   cp_batch_flush(cp);
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
                     0, cp->stream, params, NULL);
      return;
   }

   /* Same format and same size */
   if (info->src.format == info->dst.format && src_w == dst_w && src_h == dst_h) {
      cuCtxSetCurrent(cp->screen->cuda_ctx);

      /*
       * In blocks, not pixels — the same confusion cp_resource_copy_region had.
       * util_format_get_blocksize is bytes per block while the box is in
       * pixels, so a BC1 blit would walk sixteen times its real extent. Both
       * formats are equal in this arm, so one set of block counts serves both
       * sides; for an uncompressed format a block is one pixel and every value
       * below is what it was.
       */
      unsigned bw = util_format_get_nblocksx(info->src.format, src_w);
      unsigned bh = util_format_get_nblocksy(info->src.format, src_h);
      unsigned sbx = util_format_get_nblocksx(info->src.format, info->src.box.x);
      unsigned sby = util_format_get_nblocksy(info->src.format, info->src.box.y);
      unsigned dbx = util_format_get_nblocksx(info->dst.format, info->dst.box.x);
      unsigned dby = util_format_get_nblocksy(info->dst.format, info->dst.box.y);
      /* If both are CUDA-managed, use GPU copy (stays in stream order).
       * Otherwise sync and memcpy (one side is host-only memory). */
      if (src_res->cuda_managed && dst_res->cuda_managed) {
         for (int z = 0; z < info->src.box.depth; z++) {
            CUDA_MEMCPY2D copy = {0};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = (CUdeviceptr)(uintptr_t)src_data +
                             src_res->lpr.mip_offsets[info->src.level] +
                             (info->src.box.z + z) * src_img_stride +
                             sby * src_stride + sbx * src_pixel_size;
            copy.srcPitch = src_stride;
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = (CUdeviceptr)(uintptr_t)dst_data +
                             dst_res->lpr.mip_offsets[info->dst.level] +
                             (info->dst.box.z + z) * dst_img_stride +
                             dby * dst_stride + dbx * dst_pixel_size;
            copy.dstPitch = dst_stride;
            copy.WidthInBytes = bw * src_pixel_size;
            copy.Height = bh;
            cuMemcpy2D(&copy);
         }
      } else {
         cuCtxSynchronize();
         for (int z = 0; z < info->src.box.depth; z++) {
            char *s = (char *)src_data +
                      src_res->lpr.mip_offsets[info->src.level] +
                      (info->src.box.z + z) * src_img_stride +
                      sby * src_stride + sbx * src_pixel_size;
            char *d = (char *)dst_data +
                      dst_res->lpr.mip_offsets[info->dst.level] +
                      (info->dst.box.z + z) * dst_img_stride +
                      dby * dst_stride + dbx * dst_pixel_size;
            unsigned row_bytes = bw * src_pixel_size;
            for (unsigned y = 0; y < bh; y++)
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
   cp_batch_flush(cp);
   void *data = cp_resource_data(cp_res);
   if (!data)
      return;

   CUdeviceptr dev = (CUdeviceptr)(uintptr_t)data + offset;
   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* Use cuMemsetD32 for common 4-byte patterns, cuMemsetD8 for 1-byte */
   if (clear_value_size == 4) {
      uint32_t val;
      memcpy(&val, clear_value, 4);
      cuMemsetD32Async(dev, val, size / 4, cp->stream);
   } else if (clear_value_size == 1) {
      cuMemsetD8Async(dev, *(const unsigned char *)clear_value, size,
                      cp->stream);
   } else {
      char *dst = (char *)data + offset;
      for (unsigned i = 0; i < size; i += clear_value_size)
         memcpy(dst + i, clear_value, clear_value_size);
   }
}

/*
 * Fill one rectangle of a managed surface with an already-packed value, as a
 * kernel on cp->stream.
 *
 * The point of the kernel is not that it is faster than the host loop it
 * replaces — it is that it is *ordered*. Every caller here flushes the batch
 * first, but cp_batch_flush() only submits the held-back draws: it launches on
 * cp->stream and returns without synchronising. A host store into the same
 * managed pages is therefore a write-after-write race against kernels that may
 * still be running, with no edge in either direction. Launching on cp->stream
 * supplies the edge, because stream order serialises this against everything
 * already enqueued there, including the batch just flushed.
 *
 * `value` is taken already packed: cp_clear_texture is handed a value that
 * util_pack_color_union has packed for it, so packing here as well would
 * silently produce a different colour.
 *
 * Returns false when the caller has to keep the host loop: the kernel has no
 * arm for the pixel size and would write nothing at all, or the resource is
 * not managed memory.
 */
static bool
cp_clear_rect_kernel(struct cp_context *cp, struct cp_resource *res,
                     uint64_t offset, unsigned width, unsigned height,
                     unsigned stride, unsigned pixel_size,
                     const uint32_t value[4], bool depth)
{
   struct cp_screen *screen = cp->screen;
   CUfunction fn = depth ? screen->kernels.clear_depth_kernel
                         : screen->kernels.clear_kernel;
   void *data = cp_resource_data(res);

   if (!fn || !data || !res->cuda_managed)
      return false;

   /* Both kernels index on the pixel size with no else arm, so a size they do
    * not name writes nothing rather than something wrong — which is the worse
    * failure of the two, because nothing looks like "the clear did not run".
    * Depth is Z16/Z32F/Z24X8 only (cp_screen.c), colour excludes the
    * three-component formats R8G8B8, R16G16B16 and R32G32B32. */
   if (depth) {
      if (pixel_size != 2 && pixel_size != 4)
         return false;
   } else if (pixel_size != 1 && pixel_size != 2 && pixel_size != 4 &&
              pixel_size != 8 && pixel_size != 16) {
      return false;
   }

   if (!width || !height)
      return true;

   struct cp_clear_args args = {
      .target = (uint64_t)(uintptr_t)data + offset,
      .width = width, .height = height,
      .stride = stride,
      .pixel_size = pixel_size,
   };
   memcpy(args.clear_value, value, sizeof(args.clear_value));

   cuCtxSetCurrent(screen->cuda_ctx);
   void *params[] = { &args };
   cuLaunchKernel(fn,
      (width + 15) / 16, (height + 15) / 16, 1,
      16, 16, 1,
      0, cp->stream, params, NULL);
   return true;
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
   cp_batch_flush(cp);
   struct cp_resource *res = cp_resource(dst->texture);
   void *data = cp_resource_data(res);
   if (!data)
      return;

   unsigned pixel_size = util_format_get_blocksize(dst->format);
   unsigned stride = res->lpr.row_stride[dst->level];

   uint32_t value[4] = { 0 };
   util_format_pack_rgba(dst->format, value, color, 1);

   /* Level and layer are deliberately not biased in: the draw path renders to
    * level 0 of layer 0 with no bias of its own, so a clear that honoured them
    * would land somewhere the draws never look. dstx/dsty are a different
    * matter — they are within the one surface both agree on, and were being
    * dropped, which put a partial vkCmdClearAttachments in the corner. */
   uint64_t offset = (uint64_t)dsty * stride + (uint64_t)dstx * pixel_size;

   if (cp_clear_rect_kernel(cp, res, offset, width, height, stride,
                            pixel_size, value, false))
      return;

   for (unsigned y = 0; y < height; y++) {
      char *row = (char *)data + offset + (uint64_t)y * stride;
      for (unsigned x = 0; x < width; x++)
         memcpy(row + (size_t)x * pixel_size, value, pixel_size);
   }
}

static void
cp_clear_depth_stencil(struct pipe_context *ctx, struct pipe_surface *dst,
                       unsigned clear_flags, double depth, unsigned stencil,
                       unsigned dstx, unsigned dsty,
                       unsigned width, unsigned height,
                       bool render_condition_enabled)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   cp_batch_flush(cp);
   if (!dst || !dst->texture)
      return;
   struct cp_resource *res = cp_resource(dst->texture);
   void *data = cp_resource_data(res);
   if (!data)
      return;

   /* There is no stencil buffer and no stencil test in this driver
    * (cp_screen.c takes only Z16, Z32F and Z24X8), so a stencil-only clear has
    * nothing to write. It used to run the loop anyway with clear_val left at
    * zero, which zeroed the whole depth buffer — a clear of an aspect that
    * does not exist destroying the one that does. */
   if (!(clear_flags & PIPE_CLEAR_DEPTH))
      return;

   unsigned pixel_size = util_format_get_blocksize(dst->format);
   unsigned stride = res->lpr.row_stride[dst->level];

   uint32_t value[4] = { 0 };
   if (pixel_size == 4) {
      float f = (float)depth;
      memcpy(&value[0], &f, 4);
   } else {
      value[0] = (uint32_t)(depth * 65535.0);
   }

   /* Level and layer left unbiased for the same reason as the colour
    * attachment clear above. */
   uint64_t offset = (uint64_t)dsty * stride + (uint64_t)dstx * pixel_size;

   if (cp_clear_rect_kernel(cp, res, offset, width, height, stride,
                            pixel_size, value, true))
      return;

   for (unsigned y = 0; y < height; y++) {
      char *row = (char *)data + offset + (uint64_t)y * stride;
      for (unsigned x = 0; x < width; x++)
         memcpy(row + (size_t)x * pixel_size, value, pixel_size);
   }
}

static void
cp_clear_texture(struct pipe_context *ctx, struct pipe_resource *res,
                 unsigned level, const struct pipe_box *box, const void *data)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_resource *cp_res = cp_resource(res);
   cp_batch_flush(cp);
   void *tex_data = cp_resource_data(cp_res);
   if (!tex_data)
      return;

   unsigned pixel_size = util_format_get_blocksize(res->format);
   unsigned stride = cp_res->lpr.row_stride[level];
   unsigned img_stride = cp_res->lpr.img_stride[level];

   /* Already packed for us by util_pack_color_union in the frontend — packing
    * it again here would produce a plausible wrong colour. Copied out into a
    * full uint32_t[4] because the kernel argument is that wide and a narrow
    * format only supplies the first few bytes. */
   uint32_t value[4] = { 0 };
   memcpy(value, data, MIN2(pixel_size, sizeof(value)));

   /* A mip level lives at its own offset — clearing level 1 without this wrote
    * over the top of level 0, which is where cp_buffer_map and
    * cp_resource_copy_region have always looked (mip_offsets[level] + z *
    * img_stride). This is an image operation with no draw path to disagree
    * with, unlike the attachment clears above, so it can be made to agree with
    * the rest of the addressing. A buffer has one level and no layout. */
   uint64_t base = res->target == PIPE_BUFFER ? 0
                                              : cp_res->lpr.mip_offsets[level];

   for (int z = box->z; z < box->z + box->depth; z++) {
      uint64_t offset = base +
                        (uint64_t)z * img_stride +
                        (uint64_t)box->y * stride +
                        (uint64_t)box->x * pixel_size;

      if (cp_clear_rect_kernel(cp, cp_res, offset, box->width, box->height,
                               stride, pixel_size, value, false))
         continue;

      for (int y = 0; y < box->height; y++) {
         char *row = (char *)tex_data + offset + (uint64_t)y * stride;
         for (int x = 0; x < box->width; x++)
            memcpy(row + (size_t)x * pixel_size, data, pixel_size);
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

   /* A clear overwrites what the held-back draws were going to draw into. */
   cp_batch_flush(cp_ctx);

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
               0, cp_ctx->stream, params, NULL);
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
            0, cp_ctx->stream, params, NULL);
      }
   }

   /*
    * No sync. Both clears above are kernels on the same stream as everything
    * that reads what they wrote, so the ordering they need is already there;
    * draining the device for it only stalled the host at the top of every
    * frame. A host reader of the cleared surface goes through the map path,
    * which synchronises on its own account.
    */
}

/*
 * A slab allocator for small VkDeviceMemory, and why the driver needs one.
 *
 * lavapipe backs every VkDescriptorSet with its own VkDeviceMemory —
 * `lvp_descriptor_set_create()` calls `allocate_memory()` for a block whose
 * floor is 64 bytes — and frees it when the set dies. A gfxrecon replay of a
 * real frame does that about a thousand times per frame: 638,608 of the
 * 701,468 managed allocations in a 199-frame window were 64, 96, 192 or 768
 * bytes, and every one of them was freed from `lvp_descriptor_set_destroy()`.
 *
 * `cuMemAllocManaged` hands each of those a fresh mapping, and because they
 * are freed as fast as they are made, CUDA recycles the same ~1 MB of virtual
 * address space over and over. Managed memory migrates at page granularity,
 * so the 64 bytes are irrelevant: the host writes a descriptor, the whole
 * 64 KB page around it moves to the host, and then every vertex and fragment
 * launch that reads a descriptor faults it back. Measured on that replay:
 * 31,081 GPU page faults per frame with 99% of them landing in fourteen
 * distinct pages, 17.5 MB/frame of migration to serve 0.9 MB of hot data, and
 * 22.4 ms/frame — 34% of all kernel time — spent in fault service.
 *
 * The fix is to stop the ping-pong rather than to speed it up, and there are
 * two independent things wrong. Both were measured separately, because the
 * cheap one might have been the whole story and it was not:
 *
 *  - the *churn*: a thousand cuMemAllocManaged/cuMemFree pairs a frame, each
 *    a driver call, none of which buys anything. A slab with per-size-class
 *    free lists serves them from memory that is mapped once at startup. Worth
 *    12% of the replay on its own, and — measured on `particlesystem`, which
 *    is kernel-bound and gains nothing from it — it costs exactly nothing.
 *  - the *residency*: memory the host writes and the device reads has to live
 *    somewhere, and under UVM "somewhere" oscillates. Settling it on the host
 *    lets the host write at full speed and the device read over PCIe with no
 *    fault and no migration, and is worth another 17%. This is the half that
 *    can go wrong: a remote read is only cheap for memory a grid reads once,
 *    and a uniform buffer that every thread reads is exactly the case where
 *    it is not. Hence the ceiling below, which is what keeps the two apart.
 *
 * `pinned` (cuMemAllocHost) and `advise` (managed, told to prefer the host and
 * stay mapped to the device) reach the same place by different roads and
 * measured identical — 131.7 s against 131.4 s on the replay, 484 against 508
 * faults a frame. `advise` is the default because it is the one that degrades
 * safely: SET_PREFERRED_LOCATION is a hint, so UVM may still migrate a page
 * when it has to — for a device-side atomic, say — where pinned host memory
 * cannot move and the atomic has nowhere correct to go.
 *
 * The ceiling is 128 bytes, and it is not arbitrary. lavapipe's descriptor
 * sets in this replay are 64 and 96 bytes; `particlesystem`'s hot uniform
 * buffers are 140 and 208. Measured across that boundary: a 128-byte ceiling
 * is worth 29.3% of the replay and costs `particlesystem` 0.4%, while a 1 KB
 * ceiling is worth the same 28.8% and costs it 2.5%. Everything the fault
 * data pointed at is 96 bytes or under; everything above that was giving away
 * device locality for no measured return. CUDAPIPE_SMALL_ALLOC_MAX raises it
 * again up to CP_ARENA_MAX_SIZE for anyone re-testing that boundary.
 *
 * A ceiling alone is not enough, though, and `gltfscenerendering` is why: see
 * CP_ARENA_DEFAULT_WARMUP below.
 *
 * The 64-byte minimum class matches the alignment lavapipe reports for buffers
 * and images (lvp_device.c:2398, lvp_image.c:55); the only larger alignment it
 * ever asks for is 64 KB for sparse images, far above this ceiling.
 *
 * CUDAPIPE_SMALL_ALLOC selects the backing: `advise` (default), `pinned`,
 * `managed` (slab over ordinary managed memory — the control that isolates the
 * churn win from the residency win), `blocksonly` (open and advise the blocks
 * but serve nothing out of them — the control that isolates the cost of the
 * mapping from the cost of what is put in it), and `off` (per-allocation
 * cuMemAllocManaged, what the driver did before).
 *
 * CUDAPIPE_SMALL_ALLOC_STATS prints the allocation mix and the arena's hit
 * rates at exit, whether or not the arena ever opened.
 */
#define CP_ARENA_BLOCK_SIZE  ((size_t)2 * 1024 * 1024)
#define CP_ARENA_MAX_BLOCKS  64


struct cp_arena_block {
   char *base;
   char *end;
   int cls;
};

static struct {
   simple_mtx_t lock;
   /* Every screen calls cuCtxCreate for itself (cp_screen.c:409), and pinned
    * host memory is only mapped into the context that allocated it, so the
    * arena belongs to whichever screen opened it and a second screen in the
    * same process falls back to the per-allocation path. Freeing stays
    * screen-agnostic: it is decided by which block the pointer lands in. */
   struct cp_screen *owner;
   struct cp_arena_block blocks[CP_ARENA_MAX_BLOCKS];
   unsigned nblocks;
   void *free_list[CP_ARENA_CLASSES];
   char *bump[CP_ARENA_CLASSES];
   char *bump_end[CP_ARENA_CLASSES];
} cp_arena = { .lock = SIMPLE_MTX_INITIALIZER };

static enum cp_arena_mode
cp_arena_mode(void)
{
   static int mode = -1;

   if (mode < 0) {
      const char *v = getenv("CUDAPIPE_SMALL_ALLOC");
      if (!v || !strcmp(v, "advise"))
         mode = CP_ARENA_ADVISE;
      else if (!strcmp(v, "pinned"))
         mode = CP_ARENA_PINNED;
      else if (!strcmp(v, "managed"))
         mode = CP_ARENA_MANAGED;
      else if (!strcmp(v, "blocksonly"))
         mode = CP_ARENA_BLOCKSONLY;
      else
         mode = CP_ARENA_OFF;
   }
   return (enum cp_arena_mode)mode;
}

/*
 * The ceiling is tunable because it is the whole trade: everything under it
 * moves to host residency, which is free for memory the device reads once per
 * launch and a remote PCIe read for memory it reads hard.
 */
static uint64_t
cp_arena_max_size(void)
{
   static uint64_t max = 0;

   if (max == 0) {
      const char *v = getenv("CUDAPIPE_SMALL_ALLOC_MAX");
      max = v ? strtoull(v, NULL, 0) : CP_ARENA_DEFAULT_MAX;
      if (max > CP_ARENA_MAX_SIZE)
         max = CP_ARENA_MAX_SIZE;
   }
   return max;
}

/* Instrumentation. Plain counters, no getenv on the path; dumped at exit only
 * when CUDAPIPE_SMALL_ALLOC_STATS is set, which is read once, there. */
static struct {
   int armed;
   uint64_t n_alloc, n_free_hit, n_free_miss, n_reuse, n_grow;
   uint64_t by_size[CP_ARENA_MAX_SIZE + 1];
} cp_arena_stats;

static void
cp_arena_stats_dump(void)
{
   if (!getenv("CUDAPIPE_SMALL_ALLOC_STATS"))
      return;
   fprintf(stderr, "cudapipe arena: alloc %" PRIu64 " reuse %" PRIu64
           " grow %" PRIu64 " free_hit %" PRIu64 " free_miss %" PRIu64 "\n",
           cp_arena_stats.n_alloc, cp_arena_stats.n_reuse,
           cp_arena_stats.n_grow, cp_arena_stats.n_free_hit,
           cp_arena_stats.n_free_miss);
   for (uint64_t s = 0; s <= CP_ARENA_MAX_SIZE; s++)
      if (cp_arena_stats.by_size[s])
         fprintf(stderr, "cudapipe arena: size %4" PRIu64 " x %" PRIu64 "\n",
                 s, cp_arena_stats.by_size[s]);
}

/*
 * How many small allocations the driver has to see before the arena opens.
 *
 * The arena is not free for a workload that does not churn, and the size
 * ceiling above cannot tell the two apart. `gltfscenerendering` makes exactly
 * four small allocations — 64, 64, 84, 128 bytes — in a 600-frame pass, and
 * paid 9.2% for them: 15.27 ms a frame against 13.97 with the arena off,
 * monotone over three alternating reps. The capture makes 1,461,069 of the
 * same sizes and saves 29%. Nothing about a 64-byte request separates them.
 *
 * What went wrong there is what the ceiling exists to prevent, one level down.
 * lavapipe backs a descriptor set with VkDeviceMemory and binds it as a
 * constant buffer, so the kernel dereferences it (cp_context.c:5971) — every
 * thread of every launch reads it. A set that is allocated, read once and
 * freed a thousand times a frame belongs on the host. A set that is allocated
 * at startup and then read by every thread for the rest of the run belongs on
 * the device, and moving it into a host-preferred block is a remote read per
 * thread forever after. Measured: `CUDAPIPE_SMALL_ALLOC=blocksonly`, which
 * opens and advises the same blocks but serves nothing out of them, costs
 * `gltfscenerendering` 0.1% — so it is not the mapping, the advice or the
 * allocator, it is the four allocations that went into it. `managed`, which
 * packs them with no advice at all, still costs 8.7%: pack a device-hot
 * allocation into a shared 2 MB managed block and it stops being device-local
 * whether or not it is told to.
 *
 * Counting the allocations separates them, because churn is the thing the
 * arena exists to absorb. Measured over the whole suite: gltfscenerendering 4,
 * texturemipmapgen 3, pushconstants 5, particlesystem 6, texture 6,
 * instancing 10, multisampling 10, computeshader 11, texturecubemap 17,
 * pbribl 21, vulkanscene 21, multithreading 46 — for whole runs — against
 * dynamicuniformbuffer at 226,379 and the capture at 1,461,069. So a static
 * workload never opens a block, and a churning one crosses inside its first
 * frame and pays for it once: the capture keeps 29.0% of the 29.4% it had
 * ungated (131.7 s against 131.0, from 185.5 s off) and dynamicuniformbuffer
 * keeps all of its 35%.
 *
 * The gap is not as clean as it first looked, and the count is not really
 * churn. `bloom` makes 168 small allocations and reuses two, so it crosses
 * this gate on volume without churning, opens two blocks, and gets the
 * residency it would have got ungated. That is the failure mode this threshold
 * is supposed to prevent, and it costs bloom -0.9% — four of five alternating
 * pairs negative, i.e. marginally faster. So the shape is reachable and, where
 * it has been reached, harmless: bloom's allocations are evidently not the
 * long-lived device-hot kind gltfscenerendering's four are.
 *
 * A gate that counted frees instead would separate the two exactly, at the
 * cost of tracking which pointers are small during warm-up. Worth doing if
 * something ever crosses on volume AND pays for it; nothing measured yet does.
 *
 * This also subsumes the ceiling's remaining casualty. `computeshader`'s small
 * allocations are 112 and 128 bytes and it paid 11.6% for them; lowering
 * CUDAPIPE_SMALL_ALLOC_MAX to 64 was the only way to give that back, at the
 * cost of the 65-128 byte band everywhere. It makes eleven allocations in a
 * run, so the gate returns it to parity with the ceiling left at 128.
 *
 * CUDAPIPE_SMALL_ALLOC_WARMUP overrides it; 0 is the ungated behaviour.
 */

static uint64_t cp_arena_seen;

static uint64_t
cp_arena_warmup(void)
{
   static uint64_t warmup = UINT64_MAX;

   if (warmup == UINT64_MAX) {
      const char *v = getenv("CUDAPIPE_SMALL_ALLOC_WARMUP");
      warmup = v ? strtoull(v, NULL, 0) : CP_ARENA_DEFAULT_WARMUP;
   }
   return warmup;
}

/* -1 for anything the arena does not serve. */
static int
cp_arena_class(uint64_t size)
{
   if (size == 0 || size > cp_arena_max_size())
      return -1;

   /* Armed on the first small allocation rather than on the first block, so
    * that a run whose arena never opens still says how few there were — which
    * is the whole answer for a sample the gate below keeps out. */
   if (!cp_arena_stats.armed &&
       p_atomic_cmpxchg(&cp_arena_stats.armed, 0, 1) == 0)
      atexit(cp_arena_stats_dump);

   cp_arena_stats.n_alloc++;
   cp_arena_stats.by_size[size]++;

   int cls = 0;
   uint64_t chunk = (uint64_t)1 << CP_ARENA_MIN_SHIFT;
   while (chunk < size) {
      chunk <<= 1;
      cls++;
   }
   return cls;
}

/* Caller holds cp_arena.lock. */
static bool
cp_arena_grow(struct cp_screen *cp, int cls, enum cp_arena_mode mode)
{
   void *host = NULL;

   if (cp_arena.nblocks >= CP_ARENA_MAX_BLOCKS)
      return false;

   if (mode == CP_ARENA_BLOCKSONLY)
      mode = CP_ARENA_ADVISE;

   if (mode == CP_ARENA_PINNED) {
      CUresult err = cuMemAllocHost(&host, CP_ARENA_BLOCK_SIZE);
      if (err != CUDA_SUCCESS) {
         CP_CU_WARN(err, "cuMemAllocHost for the small-allocation arena");
         return false;
      }
   } else {
      CUdeviceptr dev = 0;
      CUresult err = cuMemAllocManaged(&dev, CP_ARENA_BLOCK_SIZE,
                                       CU_MEM_ATTACH_GLOBAL);
      if (err != CUDA_SUCCESS) {
         CP_CU_WARN(err, "cuMemAllocManaged for the small-allocation arena");
         return false;
      }
      host = (void *)(uintptr_t)dev;

      if (mode == CP_ARENA_ADVISE) {
         /* Keep the pages on the host and keep them mapped to the device, so
          * a device read is a remote read rather than a fault and a move. */
         cuMemAdvise(dev, CP_ARENA_BLOCK_SIZE,
                     CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_DEVICE_CPU);
         cuMemAdvise(dev, CP_ARENA_BLOCK_SIZE,
                     CU_MEM_ADVISE_SET_ACCESSED_BY, cp->cuda_device);
      }
   }

   cp_arena.blocks[cp_arena.nblocks].base = (char *)host;
   cp_arena.blocks[cp_arena.nblocks].end = (char *)host + CP_ARENA_BLOCK_SIZE;
   cp_arena.blocks[cp_arena.nblocks].cls = cls;
   cp_arena.nblocks++;
   cp_arena_stats.n_grow++;

   cp_arena.bump[cls] = (char *)host;
   cp_arena.bump_end[cls] = (char *)host + CP_ARENA_BLOCK_SIZE;
   return true;
}

static void *
cp_arena_alloc(struct cp_screen *cp, int cls, enum cp_arena_mode mode)
{
   size_t chunk = (size_t)1 << (CP_ARENA_MIN_SHIFT + cls);
   void *p;

   simple_mtx_lock(&cp_arena.lock);

   if (!cp_arena.owner)
      cp_arena.owner = cp;
   if (cp_arena.owner != cp) {
      simple_mtx_unlock(&cp_arena.lock);
      return NULL;
   }

   if (mode == CP_ARENA_BLOCKSONLY) {
      /* Open the block and advise it, then hand back nothing, so that the cost
       * of the mapping can be measured apart from the cost of moving the
       * allocations into it. */
      if (!cp_arena.bump_end[cls])
         cp_arena_grow(cp, cls, mode);
      simple_mtx_unlock(&cp_arena.lock);
      return NULL;
   }

   if (cp_arena.free_list[cls]) {
      cp_arena_stats.n_reuse++;
      p = cp_arena.free_list[cls];
      /* The free list is threaded through the free chunks themselves; they
       * are at least 64 bytes, so the link always fits. */
      cp_arena.free_list[cls] = *(void **)p;
   } else {
      if ((size_t)(cp_arena.bump_end[cls] - cp_arena.bump[cls]) < chunk &&
          !cp_arena_grow(cp, cls, mode)) {
         simple_mtx_unlock(&cp_arena.lock);
         return NULL;
      }
      p = cp_arena.bump[cls];
      cp_arena.bump[cls] += chunk;
   }

   simple_mtx_unlock(&cp_arena.lock);
   return p;
}

/* True if the pointer came out of the arena, in which case it is now free. */
static bool
cp_arena_free(void *p)
{
   simple_mtx_lock(&cp_arena.lock);

   for (unsigned i = 0; i < cp_arena.nblocks; i++) {
      if ((char *)p < cp_arena.blocks[i].base ||
          (char *)p >= cp_arena.blocks[i].end)
         continue;

      /* Every block serves exactly one size class for its whole life, so the
       * block the pointer falls in names the free list it belongs on — which
       * stays true after the class has moved its cursor to a newer block. */
      int cls = cp_arena.blocks[i].cls;
      *(void **)p = cp_arena.free_list[cls];
      cp_arena.free_list[cls] = p;
      cp_arena_stats.n_free_hit++;

      simple_mtx_unlock(&cp_arena.lock);
      return true;
   }

   cp_arena_stats.n_free_miss++;
   simple_mtx_unlock(&cp_arena.lock);
   return false;
}

/*
 * Backing for VkDeviceMemory. This has to be memory the GPU can read: the
 * vertex fetch and shader kernels dereference application buffers directly,
 * so host-only memory here faults the kernel with CUDA_ERROR_ILLEGAL_ADDRESS.
 * Pinned host memory qualifies — under unified addressing the GPU can follow a
 * cuMemAllocHost pointer directly — which is what lets the arena above use it.
 *
 * lavapipe allocates from its submit thread, which has no current context of
 * its own, hence the cuCtxSetCurrent. A context may be current on several
 * threads at once, so binding it here doesn't disturb the main thread.
 */
static struct pipe_memory_allocation *
cp_allocate_memory(struct pipe_screen *screen, uint64_t size)
{
   struct cp_screen *cp = (struct cp_screen *)screen;
   enum cp_arena_mode mode = cp_arena_mode();
   CUdeviceptr dev = 0;
   int cls;

   cuCtxSetCurrent(cp->cuda_ctx);

   if (mode != CP_ARENA_OFF && (cls = cp_arena_class(size)) >= 0 &&
       p_atomic_inc_return(&cp_arena_seen) > cp_arena_warmup()) {
      void *p = cp_arena_alloc(cp, cls, mode);
      if (p) {
         /* A plain store rather than cp_zero_managed: in the two host-resident
          * modes the pages are where the host already is, and in the managed
          * control mode this is the same host memset that path took anyway. */
         memset(p, 0, size);
         return (struct pipe_memory_allocation *)p;
      }
      /* Out of arena; fall through to the allocator that was always here. */
   }

   /*
    * Nobody up the stack checks this. lavapipe takes the result straight to
    * memset() in lvp_descriptor_set_create(), so a failure here arrives as a
    * segfault in libc with the driver nowhere in the backtrace — which is
    * exactly how a sticky CUDA_ERROR_ILLEGAL_ADDRESS from some earlier kernel
    * presented, three frames removed from the kernel that caused it.
    */
   CUresult err = cuMemAllocManaged(&dev, size, CU_MEM_ATTACH_GLOBAL);
   if (err != CUDA_SUCCESS) {
      CP_CU_WARN(err, "cuMemAllocManaged for VkDeviceMemory");
      return NULL;
   }

   cp_zero_managed(dev, size);
   return (struct pipe_memory_allocation *)(uintptr_t)dev;
}

static void
cp_free_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
   struct cp_screen *cp = (struct cp_screen *)screen;

   if (!mem)
      return;

   if (cp_arena_free((void *)mem))
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
