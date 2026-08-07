#ifndef CP_RESOURCE_H
#define CP_RESOURCE_H

#include "pipe/p_state.h"
#include <cuda.h>
#include <stdbool.h>

/*
 * IMPORTANT: The 'data' field must be at the same offset as
 * llvmpipe_resource.data (offset 512 on 64-bit) because lavapipe's
 * descriptor set code calls llvmpipe_resource_data() which casts to
 * llvmpipe_resource and reads .data. We pad our struct to match.
 */
#define CP_RESOURCE_DATA_OFFSET 456

struct cp_resource {
   struct pipe_resource base;
   char _pad[CP_RESOURCE_DATA_OFFSET - sizeof(struct pipe_resource)];
   void *data;                      /* MUST be at offset CP_RESOURCE_DATA_OFFSET */
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
