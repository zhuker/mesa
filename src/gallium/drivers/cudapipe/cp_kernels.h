#ifndef CP_KERNELS_H
#define CP_KERNELS_H

#include <cuda.h>
#include <stdbool.h>

struct cp_screen;

struct cp_kernels {
   CUmodule module;

   /* Clear kernels */
   CUfunction clear_kernel;
   CUfunction clear_depth_kernel;

   /* Rasterization kernels */
   CUfunction rasterize_triangles;
   CUfunction clear_visbuf;
   CUfunction resolve_visbuf;

   bool initialized;
};

bool cp_kernels_init(struct cp_kernels *k, struct cp_screen *screen);
void cp_kernels_destroy(struct cp_kernels *k);

#endif
