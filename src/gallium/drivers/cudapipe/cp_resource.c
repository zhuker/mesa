#include "cp_screen.h"
#include "cp_context.h"
#include "cp_resource.h"
#include "kernels/cp_rast_types.h"

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_transfer_helper.h"
#include "util/u_surface.h"
#include "util/format/u_format.h"
#include "util/format/u_format_pack.h"

#include <cuda.h>
#include <string.h>
#include <stddef.h>


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

   uint64_t size;
   if (tmpl->target == PIPE_BUFFER) {
      size = tmpl->width0;
   } else {
      unsigned nblocksx = util_format_get_nblocksx(tmpl->format, tmpl->width0);
      unsigned nblocksy = util_format_get_nblocksy(tmpl->format, tmpl->height0);
      unsigned block_size = util_format_get_blocksize(tmpl->format);
      res->lpr.row_stride[0] = nblocksx * block_size;
      res->lpr.img_stride[0] = (uint64_t)res->lpr.row_stride[0] * nblocksy;
      size = res->lpr.img_stride[0] * MAX2(tmpl->depth0, 1) * MAX2(tmpl->array_size, 1);
   }

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
   struct cp_resource *res = CALLOC_STRUCT(cp_resource);
   if (!res)
      return NULL;

   res->lpr.base = *tmpl;
   res->lpr.base.screen = screen;
   pipe_reference_init(&res->lpr.base.reference, 1);

   uint64_t size;
   if (tmpl->target == PIPE_BUFFER) {
      size = tmpl->width0;
   } else {
      unsigned nblocksx = util_format_get_nblocksx(tmpl->format, tmpl->width0);
      unsigned nblocksy = util_format_get_nblocksy(tmpl->format, tmpl->height0);
      unsigned block_size = util_format_get_blocksize(tmpl->format);
      res->lpr.row_stride[0] = nblocksx * block_size;
      res->lpr.img_stride[0] = (uint64_t)res->lpr.row_stride[0] * nblocksy;
      size = res->lpr.img_stride[0] * MAX2(tmpl->depth0, 1) * MAX2(tmpl->array_size, 1);
   }

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
   struct cp_resource *res = cp_resource(resource);
   struct pipe_transfer *transfer = CALLOC_STRUCT(pipe_transfer);
   if (!transfer)
      return NULL;

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

   if (resource->target == PIPE_BUFFER)
      return (char *)data + box->x;

   unsigned offset = box->z * res->lpr.img_stride[level] +
                     box->y * res->lpr.row_stride[level] +
                     box->x * util_format_get_blocksize(resource->format);
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

   for (int z = 0; z < src_box->depth; z++) {
      for (int y = 0; y < src_box->height; y++) {
         char *s = (char *)src_data + (src_box->z + z) * src_img_stride +
                   (src_box->y + y) * src_stride + src_box->x * pixel_size;
         char *d = (char *)dst_data + (dstz + z) * dst_img_stride +
                   (dsty + y) * dst_stride + dstx * pixel_size;
         memcpy(d, s, src_box->width * pixel_size);
      }
   }
}

static void
cp_blit(struct pipe_context *ctx, const struct pipe_blit_info *info)
{
   /* TODO: Phase 6 - blit kernel */
}

static void
cp_clear_buffer(struct pipe_context *ctx, struct pipe_resource *res,
                unsigned offset, unsigned size,
                const void *clear_value, int clear_value_size)
{
   struct cp_resource *cp_res = cp_resource(res);
   void *data = cp_resource_data(cp_res);
   if (!data)
      return;

   char *dst = (char *)data + offset;
   for (unsigned i = 0; i < size; i += clear_value_size)
      memcpy(dst + i, clear_value, clear_value_size);
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
   struct cp_resource *res = cp_resource(dst->texture);
   void *data = cp_resource_data(res);
   if (!data)
      return;

   unsigned pixel_size = util_format_get_blocksize(dst->format);
   unsigned stride = res->lpr.row_stride[dst->level];

   uint32_t clear_val[4] = {0};
   util_format_pack_rgba(dst->format, clear_val, color, 1);

   for (unsigned y = dsty; y < dsty + height; y++) {
      char *row = (char *)data + y * stride + dstx * pixel_size;
      for (unsigned x = 0; x < width; x++)
         memcpy(row + x * pixel_size, clear_val, pixel_size);
   }
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

         void *params[] = { &args };
         cuLaunchKernel(screen->kernels.clear_kernel,
            (w + 15) / 16, (h + 15) / 16, 1,
            16, 16, 1,
            0, NULL, params, NULL);
      }
   }

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

static struct pipe_memory_allocation *
cp_allocate_memory(struct pipe_screen *screen, uint64_t size)
{
   CUdeviceptr ptr;
   if (cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS)
      return NULL;
   cuMemsetD8(ptr, 0, size);
   return (struct pipe_memory_allocation *)(uintptr_t)ptr;
}

static void
cp_free_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
   cuMemFree((CUdeviceptr)(uintptr_t)mem);
}

static void *
cp_map_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
   /* For managed memory, the device pointer IS host-accessible */
   return (void *)(uintptr_t)mem;
}

static void
cp_unmap_memory(struct pipe_screen *screen, struct pipe_memory_allocation *mem)
{
}

static bool
cp_resource_bind_backing(struct pipe_screen *screen, struct pipe_resource *pt,
                         struct pipe_memory_allocation *mem, uint64_t offset,
                         uint64_t size, uint64_t alignment)
{
   struct cp_resource *res = cp_resource(pt);
   void *ptr = (char *)mem + offset;
   res->lpr.data = ptr;
   res->lpr.tex_data = ptr;
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
