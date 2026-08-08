#include "cp_kernels.h"
#include "cp_screen.h"
#include "util/u_memory.h"

#include <nvrtc.h>
#include <stdio.h>
#include <string.h>

/* Embedded kernel sources */
static const char cp_clear_src[] =
#include "kernels/cp_clear.cu.inc"
;

static const char cp_rasterize_src[] =
#include "kernels/cp_rasterize.cu.inc"
;

static char *
compile_cuda_source(const char *source, const char *name, int sm_major, int sm_minor)
{
   nvrtcProgram prog;
   nvrtcResult res = nvrtcCreateProgram(&prog, source, name, 0, NULL, NULL);
   if (res != NVRTC_SUCCESS) {
      fprintf(stderr, "cudapipe: nvrtcCreateProgram failed: %d\n", res);
      return NULL;
   }

   char arch_opt[32];
   snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=sm_%d%d", sm_major, sm_minor);

   const char *opts[] = {
      arch_opt,
      "--std=c++14",
      "-I/home/coder/git/mesa/src/gallium/drivers/cudapipe/kernels",
   };

   res = nvrtcCompileProgram(prog, 3, opts);
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

bool
cp_kernels_init(struct cp_kernels *k, struct cp_screen *screen)
{
   memset(k, 0, sizeof(*k));

   /* Compile clear kernels */
   char *clear_ptx = compile_cuda_source(cp_clear_src, "cp_clear.cu",
                                         screen->sm_major, screen->sm_minor);
   if (!clear_ptx) {
      fprintf(stderr, "cudapipe: failed to compile clear kernels\n");
      return false;
   }

   /* Compile rasterization kernels */
   char *rast_ptx = compile_cuda_source(cp_rasterize_src, "cp_rasterize.cu",
                                        screen->sm_major, screen->sm_minor);
   if (!rast_ptx) {
      free(clear_ptx);
      fprintf(stderr, "cudapipe: failed to compile rasterization kernels\n");
      return false;
   }

   /* Load clear module */
   CUresult err;
   CUmodule clear_mod;
   err = cuModuleLoadData(&clear_mod, clear_ptx);
   free(clear_ptx);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: cuModuleLoadData(clear) failed: %d\n", err);
      free(rast_ptx);
      return false;
   }

   cuModuleGetFunction(&k->clear_kernel, clear_mod, "cp_clear_kernel");
   cuModuleGetFunction(&k->clear_depth_kernel, clear_mod, "cp_clear_depth_kernel");

   /* Load rasterization module */
   CUmodule rast_mod;
   err = cuModuleLoadData(&rast_mod, rast_ptx);
   free(rast_ptx);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: cuModuleLoadData(rast) failed: %d\n", err);
      cuModuleUnload(clear_mod);
      return false;
   }

   cuModuleGetFunction(&k->rasterize_triangles, rast_mod, "cp_rasterize_triangles");
   cuModuleGetFunction(&k->clear_visbuf, rast_mod, "cp_clear_visbuf");
   cuModuleGetFunction(&k->resolve_visbuf, rast_mod, "cp_resolve_visbuf");

   k->module = rast_mod;
   k->initialized = true;
   return true;
}

void
cp_kernels_destroy(struct cp_kernels *k)
{
   if (k->module)
      cuModuleUnload(k->module);
   memset(k, 0, sizeof(*k));
}
