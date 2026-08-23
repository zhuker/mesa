/*
 * The device a renderer runs on.
 *
 * A CUDA context, the compiled kernels and the compute capability:
 * everything the draw pipeline needs of the device it runs on.
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
