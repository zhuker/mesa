#include "cp_screen.h"
#include "cp_resource.h"

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_transfer_helper.h"
#include "util/format/u_format.h"

#include <cuda.h>
#include <string.h>

struct cp_resource {
   struct pipe_resource base;
   CUdeviceptr device_ptr;
   void *data;
   uint64_t size;
   unsigned row_stride;
   unsigned layer_stride;
   bool cuda_managed;
};

static struct cp_resource *
cp_resource(struct pipe_resource *pt)
{
   return (struct cp_resource *)pt;
}

static struct pipe_resource *
cp_resource_create(struct pipe_screen *screen,
                   const struct pipe_resource *tmpl)
{
   struct cp_resource *res = CALLOC_STRUCT(cp_resource);
   if (!res)
      return NULL;

   res->base = *tmpl;
   res->base.screen = screen;
   pipe_reference_init(&res->base.reference, 1);

   if (tmpl->target == PIPE_BUFFER) {
      res->size = tmpl->width0;
   } else {
      unsigned nblocksx = util_format_get_nblocksx(tmpl->format, tmpl->width0);
      unsigned nblocksy = util_format_get_nblocksy(tmpl->format, tmpl->height0);
      unsigned block_size = util_format_get_blocksize(tmpl->format);
      res->row_stride = nblocksx * block_size;
      res->layer_stride = res->row_stride * nblocksy;
      res->size = (uint64_t)res->layer_stride * tmpl->depth0 * tmpl->array_size;
   }

   if (res->size > 0) {
      CUresult err = cuMemAllocManaged(&res->device_ptr, res->size,
                                       CU_MEM_ATTACH_GLOBAL);
      if (err != CUDA_SUCCESS) {
         FREE(res);
         return NULL;
      }
      res->data = (void *)(uintptr_t)res->device_ptr;
      res->cuda_managed = true;
      cuMemsetD8(res->device_ptr, 0, res->size);
   }

   return &res->base;
}

static struct pipe_resource *
cp_resource_create_unbacked(struct pipe_screen *screen,
                            const struct pipe_resource *tmpl,
                            uint64_t *size_required)
{
   struct cp_resource *res = CALLOC_STRUCT(cp_resource);
   if (!res)
      return NULL;

   res->base = *tmpl;
   res->base.screen = screen;
   pipe_reference_init(&res->base.reference, 1);

   if (tmpl->target == PIPE_BUFFER) {
      res->size = tmpl->width0;
   } else {
      unsigned nblocksx = util_format_get_nblocksx(tmpl->format, tmpl->width0);
      unsigned nblocksy = util_format_get_nblocksy(tmpl->format, tmpl->height0);
      unsigned block_size = util_format_get_blocksize(tmpl->format);
      res->row_stride = nblocksx * block_size;
      res->layer_stride = res->row_stride * nblocksy;
      res->size = (uint64_t)res->layer_stride * tmpl->depth0 * tmpl->array_size;
   }

   if (size_required)
      *size_required = res->size;

   return &res->base;
}

static void
cp_resource_destroy(struct pipe_screen *screen, struct pipe_resource *pt)
{
   struct cp_resource *res = cp_resource(pt);
   if (res->cuda_managed && res->device_ptr)
      cuMemFree(res->device_ptr);
   else if (res->data && !res->cuda_managed)
      FREE(res->data);
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
   transfer->stride = res->row_stride;
   transfer->layer_stride = res->layer_stride;

   *out_transfer = transfer;

   if (resource->target == PIPE_BUFFER)
      return (char *)res->data + box->x;

   unsigned offset = box->z * res->layer_stride + box->y * res->row_stride +
                     box->x * util_format_get_blocksize(resource->format);
   return (char *)res->data + offset;
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
   /* TODO: implement copy */
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
   if (!cp_res->data)
      return;

   char *dst = (char *)cp_res->data + offset;
   for (unsigned i = 0; i < size; i += clear_value_size)
      memcpy(dst + i, clear_value, clear_value_size);
}

static void
cp_clear(struct pipe_context *ctx, unsigned buffers,
         uint32_t color_clear_mask, uint8_t stencil_clear_mask,
         const struct pipe_scissor_state *scissor,
         const union pipe_color_union *color, double depth,
         unsigned stencil)
{
   /* TODO: Phase 3 - clear kernel */
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
   res->data = (char *)mem + offset;
   return true;
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
}
