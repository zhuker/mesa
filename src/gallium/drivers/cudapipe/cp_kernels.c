#include "cp_kernels.h"
#include "cp_screen.h"
#include "util/u_memory.h"
#include "util/macros.h"

#include <nvrtc.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Embedded kernel sources, stringified from the .cu files at build time. */
static const char cp_clear_src[] =
#include "cp_clear.cu.inc"
;

static const char cp_rasterize_src[] =
#include "cp_rasterize.cu.inc"
;

static const char cp_fs_src[] =
#include "cp_fs.cu.inc"
;

static const char cp_sampler_src[] =
#include "cp_sampler.cu.inc"
;

static const char cp_vertex_fetch_src[] =
#include "cp_vertex_fetch.cu.inc"
;

/* The kernels share this header; NVRTC has no filesystem, so hand it over
 * in memory rather than pointing it at an include directory. */
static const char cp_rast_types_src[] =
#include "cp_rast_types.h.inc"
;

/*
 * Whether the debug instrumentation is compiled into the kernels at all.
 *
 * This is the fragment census and the machinery that compares the A-buffer
 * against the peel loop — not the A-buffer itself, which is now always built
 * and is selected by launching a different kernel rather than by a branch.
 * What is left here is what genuinely has to sit inside emit_fragment and
 * cp_fs_interpolate, both of which run hundreds of millions of times a frame,
 * so it must not exist unless something is going to read it. NVRTC compiles at
 * run time, so it simply does not, the same way CP_SMALL_THRESHOLD and friends
 * are handed over below.
 */
bool
cp_kernels_instrumented(void)
{
   static int on = -1;
   if (on < 0) {
      /* CUDAPIPE_ABUF_COMPILE=1 compiles the branches in without turning any
       * feature on, which is the only way to measure what they cost; =0
       * refuses them outright. */
      const char *force = getenv("CUDAPIPE_ABUF_COMPILE");
      if (force && *force)
         on = atoi(force) != 0;
      else
         on = (getenv("CUDAPIPE_ABUFFER_VERIFY") ||
               getenv("CUDAPIPE_FRAG_CENSUS")) ? 1 : 0;
   }
   return on == 1;
}

/*
 * Compile a kernel source with NVRTC.
 *
 * When `relocatable` is set the result is device-relocatable PTX suitable for
 * cuLinkAddData(), which is how the sampler gets linked into shader PTX.
 */
static char *
compile_cuda_source(const char *source, const char *name, int sm_major,
                    int sm_minor, bool relocatable)
{
   const char *header_srcs[] = { cp_rast_types_src };
   const char *header_names[] = { "cp_rast_types.h" };

   nvrtcProgram prog;
   nvrtcResult res = nvrtcCreateProgram(&prog, source, name, 1,
                                        header_srcs, header_names);
   if (res != NVRTC_SUCCESS) {
      fprintf(stderr, "cudapipe: nvrtcCreateProgram failed: %d\n", res);
      return NULL;
   }

   char arch_opt[32];
   /* A virtual architecture keeps the output as PTX for the linker to JIT,
    * rather than baking in SASS for one chip. */
   snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d",
            sm_major, sm_minor);

   /*
    * The rasterizer thresholds decide how much of the machine a triangle
    * gets, and the right values are a property of the workload rather than
    * something to be reasoned out once. NVRTC compiles at run time, so they
    * can be swept from the environment without a rebuild, which is what makes
    * a sweep of them repeatable rather than a series of builds nobody can
    * reproduce. Unset means the header's default.
    */
   char small_opt[64], medium_opt[64], point_opt[64], tilebound_opt[64];
   const char *opts[8];
   unsigned num_opts = 0;
   opts[num_opts++] = arch_opt;
   opts[num_opts++] = "--std=c++14";
   if (relocatable)
      opts[num_opts++] = "--relocatable-device-code=true";
   if (cp_kernels_instrumented())
      opts[num_opts++] = "-DCP_ABUF_INSTRUMENT=1";

   const char *small_env = getenv("CUDAPIPE_SMALL_THRESHOLD");
   if (small_env && *small_env) {
      snprintf(small_opt, sizeof(small_opt), "-DCP_SMALL_THRESHOLD=%d",
               atoi(small_env));
      opts[num_opts++] = small_opt;
   }
   const char *medium_env = getenv("CUDAPIPE_MEDIUM_THRESHOLD");
   if (medium_env && *medium_env) {
      snprintf(medium_opt, sizeof(medium_opt), "-DCP_MEDIUM_THRESHOLD=%d",
               atoi(medium_env));
      opts[num_opts++] = medium_opt;
   }
   const char *point_env = getenv("CUDAPIPE_POINT_THRESHOLD");
   if (point_env && *point_env) {
      snprintf(point_opt, sizeof(point_opt), "-DCP_POINT_THRESHOLD=%d",
               atoi(point_env));
      opts[num_opts++] = point_opt;
   }
   /* Not a threshold but the same kind of knob: whether stage 3 walks the
    * whole tile or only the part of it the primitive's bounding box reaches.
    * CUDAPIPE_TILE_BOUND=0 compiles the full-tile walk back, so the two can be
    * compared without a rebuild. */
   const char *tilebound_env = getenv("CUDAPIPE_TILE_BOUND");
   if (tilebound_env && *tilebound_env) {
      snprintf(tilebound_opt, sizeof(tilebound_opt), "-DCP_TILE_BOUND=%d",
               atoi(tilebound_env));
      opts[num_opts++] = tilebound_opt;
   }

   assert(num_opts <= ARRAY_SIZE(opts));
   res = nvrtcCompileProgram(prog, num_opts, opts);
   if (res != NVRTC_SUCCESS) {
      size_t log_size;
      nvrtcGetProgramLogSize(prog, &log_size);
      char *log = malloc(log_size);
      nvrtcGetProgramLog(prog, log);
      fprintf(stderr, "cudapipe: NVRTC compile failed for %s:\n%s\n", name, log);
      free(log);
      nvrtcDestroyProgram(&prog);
      return NULL;
   }

   size_t ptx_size;
   nvrtcGetPTXSize(prog, &ptx_size);
   char *ptx = malloc(ptx_size);
   nvrtcGetPTX(prog, ptx);
   nvrtcDestroyProgram(&prog);

   return ptx;
}

/* Compile one kernel source and load it as a module. */
static bool
build_module(CUmodule *out, const char *src, const char *name,
             struct cp_screen *screen)
{
   char *ptx = compile_cuda_source(src, name, screen->sm_major,
                                   screen->sm_minor, false);
   if (!ptx) {
      fprintf(stderr, "cudapipe: failed to compile %s\n", name);
      return false;
   }

   CUresult err = cuModuleLoadData(out, ptx);
   free(ptx);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: cuModuleLoadData(%s) failed: %d\n", name, err);
      return false;
   }
   return true;
}

bool
cp_kernels_init(struct cp_kernels *k, struct cp_screen *screen)
{
   memset(k, 0, sizeof(*k));

   if (!build_module(&k->clear_module, cp_clear_src, "cp_clear.cu", screen))
      return false;
   cuModuleGetFunction(&k->clear_kernel, k->clear_module, "cp_clear_kernel");
   cuModuleGetFunction(&k->clear_depth_kernel, k->clear_module, "cp_clear_depth_kernel");

   if (!build_module(&k->module, cp_rasterize_src, "cp_rasterize.cu", screen))
      goto fail;
   cuModuleGetFunction(&k->rasterize_stage1, k->module, "cp_rasterize_stage1");
   cuModuleGetFunction(&k->rasterize_stage2, k->module, "cp_rasterize_stage2");
   cuModuleGetFunction(&k->rasterize_stage3, k->module, "cp_rasterize_stage3");
   cuModuleGetFunction(&k->clip_triangles, k->module, "cp_clip_triangles");
   k->rasterize_triangles = k->rasterize_stage1;
   cuModuleGetFunction(&k->clear_visbuf, k->module, "cp_clear_visbuf");
   cuModuleGetFunction(&k->peel_advance, k->module, "cp_peel_advance");
   cuModuleGetFunction(&k->resolve_visbuf, k->module, "cp_resolve_visbuf");

   /*
    * The A-buffer's own specialisation of the three rasterizer stages. Same
    * source, compiled a second time with the count-and-fill branch live, so
    * that the branch is chosen by which kernel the host launches rather than
    * tested on the device once per coverage event. See emit_fragment().
    */
   cuModuleGetFunction(&k->rasterize_stage1_abuf, k->module,
                       "cp_rasterize_stage1_abuf");
   cuModuleGetFunction(&k->rasterize_stage2_abuf, k->module,
                       "cp_rasterize_stage2_abuf");
   cuModuleGetFunction(&k->rasterize_stage3_abuf, k->module,
                       "cp_rasterize_stage3_abuf");

   cuModuleGetFunction(&k->abuf_scan_block, k->module, "cp_abuf_scan_block");
   cuModuleGetFunction(&k->abuf_scan_add, k->module, "cp_abuf_scan_add");
   cuModuleGetFunction(&k->abuf_worklist, k->module, "cp_abuf_worklist");
   cuModuleGetFunction(&k->abuf_sort, k->module, "cp_abuf_sort");
   cuModuleGetFunction(&k->abuf_block_worklist, k->module,
                       "cp_abuf_block_worklist");
   cuModuleGetFunction(&k->abuf_quad_count, k->module, "cp_abuf_quad_count");
   cuModuleGetFunction(&k->abuf_clear_slots, k->module, "cp_abuf_clear_slots");
   cuModuleGetFunction(&k->abuf_clamp_runs, k->module, "cp_abuf_clamp_runs");
   cuModuleGetFunction(&k->abuf_quad_fill, k->module, "cp_abuf_quad_fill");
   /* Only present when the instrumentation was compiled in, so only looked
    * up then; the draw path checks the pointers before launching. */
   if (cp_kernels_instrumented()) {
      cuModuleGetFunction(&k->abuf_peel_log, k->module, "cp_abuf_peel_log");
      cuModuleGetFunction(&k->abuf_peel_log_list, k->module,
                          "cp_abuf_peel_log_list");
   }

   if (!build_module(&k->fs_module, cp_fs_src, "cp_fs.cu", screen))
      goto fail;
   cuModuleGetFunction(&k->fs_interpolate, k->fs_module, "cp_fs_interpolate");
   cuModuleGetFunction(&k->fs_writeback, k->fs_module, "cp_fs_writeback");
   cuModuleGetFunction(&k->resolve_samples, k->fs_module, "cp_resolve_samples");
   /* The quad-stream interpolator lives beside the peel one so both call the
    * same cp_interp_pixel; see cp_fs.cu. */
   cuModuleGetFunction(&k->abuf_interpolate, k->fs_module,
                       "cp_abuf_interpolate");
   cuModuleGetFunction(&k->abuf_composite, k->fs_module, "cp_abuf_composite");
   if (cp_kernels_instrumented())
      cuModuleGetFunction(&k->abuf_scatter_colors, k->fs_module,
                          "cp_abuf_scatter_colors");

   if (!build_module(&k->vfetch_module, cp_vertex_fetch_src, "cp_vertex_fetch.cu", screen))
      goto fail;
   cuModuleGetFunction(&k->vertex_fetch, k->vfetch_module, "cp_vertex_fetch");

   /* Kept as relocatable PTX rather than a module: it is linked into each
    * shader that samples textures, not launched on its own. */
   k->sampler_ptx = compile_cuda_source(cp_sampler_src, "cp_sampler.cu",
                                        screen->sm_major, screen->sm_minor, true);
   if (!k->sampler_ptx) {
      fprintf(stderr, "cudapipe: failed to compile texture sampler\n");
      goto fail;
   }

   k->initialized = true;
   return true;

fail:
   cp_kernels_destroy(k);
   return false;
}

void
cp_kernels_destroy(struct cp_kernels *k)
{
   if (k->module)
      cuModuleUnload(k->module);
   if (k->clear_module)
      cuModuleUnload(k->clear_module);
   if (k->fs_module)
      cuModuleUnload(k->fs_module);
   free(k->sampler_ptx);
   memset(k, 0, sizeof(*k));
}
