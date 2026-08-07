#ifndef CP_RESOURCE_H
#define CP_RESOURCE_H

#include "pipe/p_state.h"
#include <cuda.h>
#include <stdbool.h>

struct cp_resource {
   struct pipe_resource base;
   CUdeviceptr device_ptr;
   void *data;
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
