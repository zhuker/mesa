#ifndef CP_SCREEN_H
#define CP_SCREEN_H

#include "pipe/p_screen.h"
#include "util/u_thread.h"

#include <cuda.h>
#include "cp_kernels.h"

/*
 * What a pipe_fence_handle points at. One queue submit's fence is stored by
 * every vk_sync it signals, so the handle is refcounted; the last release
 * destroys the event. Created by cp_flush, waited by cp_fence_finish.
 */
struct cp_fence {
   CUevent event;
   int32_t refcount;
};

/*
 * The device the renderer runs on: a CUDA context, the compiled kernels and
 * the compute capability, and nothing else. This is everything the draw
 * pipeline ever wanted from the screen -- it reads exactly these four things
 * across 118 sites -- so it is what a Vulkan front end supplies instead of a
 * pipe_screen.
 */
struct cp_device {
   CUdevice cuda_device;
   CUcontext cuda_ctx;
   int sm_major;
   int sm_minor;

   struct cp_kernels kernels;
};

struct cp_screen {
   struct pipe_screen base;

   struct sw_winsys *winsys;

   struct cp_device dev;

   char renderer_string[128];
};

static inline struct cp_screen *
cp_screen(struct pipe_screen *pipe)
{
   return (struct cp_screen *)pipe;
}

#endif
