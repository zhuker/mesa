#ifndef CP_SCREEN_H
#define CP_SCREEN_H

#include "pipe/p_screen.h"
#include "util/u_thread.h"

#include <cuda.h>
#include "cp_kernels.h"
#include "cp_device.h"

/*
 * What a pipe_fence_handle points at. One queue submit's fence is stored by
 * every vk_sync it signals, so the handle is refcounted; the last release
 * destroys the event. Created by cp_flush, waited by cp_fence_finish.
 */
struct cp_fence {
   CUevent event;
   int32_t refcount;
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
