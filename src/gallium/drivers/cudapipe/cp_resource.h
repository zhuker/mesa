#ifndef CP_RESOURCE_H
#define CP_RESOURCE_H

#include "pipe/p_state.h"
#include <cuda.h>
#include <stdbool.h>

/*
 * IMPORTANT: tex_data and data must match llvmpipe_resource offsets
 * because lavapipe's descriptor code calls llvmpipe_resource functions
 * that read from these fixed offsets.
 *   tex_data: offset 440 (used for textures/images)
 *   data:     offset 456 (used for buffers)
 */
#define CP_RESOURCE_TEX_DATA_OFFSET 440
#define CP_RESOURCE_DATA_OFFSET 456

struct cp_resource {
   struct pipe_resource base;
   char _pad[CP_RESOURCE_TEX_DATA_OFFSET - sizeof(struct pipe_resource)];
   void *tex_data;                  /* MUST be at offset 440 */
   char _pad2[456 - 440 - sizeof(void *)];
   void *data;                      /* MUST be at offset 456 */
   CUdeviceptr device_ptr;
   uint64_t size;
   unsigned row_stride;
   unsigned layer_stride;
   bool cuda_managed;
   bool owns_data;
};

static inline struct cp_resource *
cp_resource(struct pipe_resource *pt)
{
   return (struct cp_resource *)pt;
}

void cudapipe_init_screen_resource_funcs(struct pipe_screen *screen);
void cudapipe_init_context_resource_funcs(struct pipe_context *ctx);

#endif
