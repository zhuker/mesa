#ifndef CP_SCREEN_H
#define CP_SCREEN_H

#include "pipe/p_screen.h"
#include "util/u_thread.h"

#include <cuda.h>
#include "cp_kernels.h"

struct cp_screen {
   struct pipe_screen base;

   struct sw_winsys *winsys;

   CUdevice cuda_device;
   CUcontext cuda_ctx;
   int sm_major;
   int sm_minor;

   struct cp_kernels kernels;

   char renderer_string[128];
};

static inline struct cp_screen *
cp_screen(struct pipe_screen *pipe)
{
   return (struct cp_screen *)pipe;
}

#endif
