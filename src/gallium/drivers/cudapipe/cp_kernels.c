#include "cp_kernels.h"
#include "cp_screen.h"
#include "util/u_memory.h"
#include "util/macros.h"

#include <nvrtc.h>
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

   const char *opts[3];
   unsigned num_opts = 0;
   opts[num_opts++] = arch_opt;
   opts[num_opts++] = "--std=c++14";
   if (relocatable)
      opts[num_opts++] = "--relocatable-device-code=true";

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
   cuModuleGetFunction(&k->resolve_visbuf, k->module, "cp_resolve_visbuf");

   if (!build_module(&k->fs_module, cp_fs_src, "cp_fs.cu", screen))
      goto fail;
   cuModuleGetFunction(&k->fs_interpolate, k->fs_module, "cp_fs_interpolate");
   cuModuleGetFunction(&k->fs_writeback, k->fs_module, "cp_fs_writeback");

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
