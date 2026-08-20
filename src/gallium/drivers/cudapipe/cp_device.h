/*
 * The device a renderer runs on.
 *
 * Deliberately free of Gallium: a CUDA context, the compiled kernels and the
 * compute capability are everything the draw pipeline reads of a screen -- it
 * does so across 118 sites -- so this is what a Vulkan front end supplies
 * instead of a pipe_screen. cp_screen embeds one; the native driver holds one
 * of its own.
 */

#ifndef CP_DEVICE_H
#define CP_DEVICE_H

#include <cuda.h>

#include "cp_kernels.h"

struct cp_device {
   CUdevice cuda_device;
   CUcontext cuda_ctx;
   int sm_major;
   int sm_minor;

   struct cp_kernels kernels;
};

#endif /* CP_DEVICE_H */
