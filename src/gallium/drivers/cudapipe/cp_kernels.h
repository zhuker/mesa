#ifndef CP_KERNELS_H
#define CP_KERNELS_H

#include <cuda.h>
#include <stdbool.h>

struct cp_screen;

struct cp_kernels {
   CUmodule module;
   CUmodule clear_module;
   CUmodule fs_module;

   /* Clear kernels */
   CUfunction clear_kernel;
   CUfunction clear_depth_kernel;

   /* Rasterization kernels (3-stage adaptive) */
   CUfunction rasterize_triangles; /* alias for stage1, used by legacy code */
   CUfunction rasterize_stage1;
   CUfunction rasterize_stage2;
   CUfunction rasterize_stage3;
   CUfunction clear_visbuf;
   CUfunction resolve_visbuf;

   /* Fragment stage kernels bracketing the compiled fragment shader */
   CUfunction fs_interpolate;
   CUfunction fs_writeback;

   /* Vertex fetch kernel — gathers attributes on GPU */
   CUmodule vfetch_module;
   CUfunction vertex_fetch;

   /* Relocatable PTX for the texture sampler, linked into each shader that
    * samples textures. Owned here; see cp_compile_nir_to_ptx(). */
   char *sampler_ptx;

   bool initialized;
};

bool cp_kernels_init(struct cp_kernels *k, struct cp_screen *screen);
void cp_kernels_destroy(struct cp_kernels *k);

#endif
