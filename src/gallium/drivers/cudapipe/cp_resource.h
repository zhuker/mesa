#ifndef CP_RESOURCE_H
#define CP_RESOURCE_H

#include "pipe/p_state.h"
#include <cuda.h>
#include <stdbool.h>

/*
 * We reuse llvmpipe_resource directly because lavapipe's descriptor code
 * calls llvmpipe functions that cast pipe_resource to llvmpipe_resource
 * and read fields at specific offsets (row_stride, tex_data, data, etc).
 *
 * We add our CUDA-specific fields after the llvmpipe_resource struct.
 */
#include "../llvmpipe/lp_texture.h"

struct cp_resource {
   struct llvmpipe_resource lpr;    /* compatible with lavapipe's expectations */
   CUdeviceptr device_ptr;
   bool cuda_managed;
   bool owns_data;
};

static inline struct cp_resource *
cp_resource(struct pipe_resource *pt)
{
   return (struct cp_resource *)pt;
}

static inline void *
cp_resource_data(struct cp_resource *res)
{
   if (llvmpipe_resource_is_texture(&res->lpr.base))
      return res->lpr.tex_data;
   return res->lpr.data;
}

void cudapipe_init_screen_resource_funcs(struct pipe_screen *screen);
void cudapipe_init_context_resource_funcs(struct pipe_context *ctx);

#endif
