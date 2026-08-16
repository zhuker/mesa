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
   /* The same three with the A-buffer's count/fill branch compiled in. Only
    * the count and fill passes launch these; see emit_fragment(). */
   CUfunction rasterize_stage1_abuf;
   CUfunction rasterize_stage2_abuf;
   CUfunction rasterize_stage3_abuf;
   CUfunction clip_triangles;
   CUfunction clear_visbuf;
   CUfunction peel_advance;
   CUfunction resolve_samples;
   CUfunction resolve_visbuf;

   /* A-buffer build, and the peel-loop log its verification compares against. */
   CUfunction abuf_scan_block;
   CUfunction abuf_scan_add;
   CUfunction abuf_worklist;
   CUfunction abuf_sort;
   CUfunction abuf_peel_log;
   CUfunction abuf_peel_log_list;
   CUfunction abuf_block_worklist;
   CUfunction abuf_quad_count;
   CUfunction abuf_clamp_runs;
   CUfunction abuf_fill_recs;
   CUfunction abuf_clear_slots;
   CUfunction abuf_quad_fill;
   CUfunction abuf_seg_count;
   CUfunction abuf_seg_scatter;
   CUfunction abuf_interpolate;
   CUfunction abuf_scatter_colors;
   CUfunction abuf_composite;

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

/* Whether the kernels were compiled with the census and A-buffer verification
 * instrumentation. See cp_kernels.c. */
bool cp_kernels_instrumented(void);

bool cp_kernels_init(struct cp_kernels *k, struct cp_screen *screen);
void cp_kernels_destroy(struct cp_kernels *k);

#endif
