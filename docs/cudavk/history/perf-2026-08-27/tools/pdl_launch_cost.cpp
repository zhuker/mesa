/*
 * cuLaunchKernelEx host-cost microbenchmark.
 *
 * Programmatic dependent launch replaces cuLaunchKernel with cuLaunchKernelEx
 * plus a config struct and an attribute array. At PDL level 3 the driver makes
 * about 550 extended launches a frame, and at level 4 it would make about 760,
 * so a per-launch premium of a few hundred nanoseconds is the same order as
 * the win. This measures that premium directly, on an idle GPU, with no
 * capture and no driver involved.
 *
 * Three arms, so the premium is attributed rather than just observed:
 *   plain  cuLaunchKernel
 *   ex0    cuLaunchKernelEx with numAttrs = 0        -- the entry point alone
 *   ex1    cuLaunchKernelEx with the PDL attribute   -- entry point + attribute
 *
 * Only the HOST side is timed: the kernel is empty and the loop never
 * synchronises, so what is measured is the cost of getting a launch into the
 * queue. Arms are interleaved and repeated; the minimum over repeats is the
 * headline, because this is a latency measurement and the minimum is the least
 * contaminated statistic. The mean is printed next to it so that a noisy run
 * is visible rather than hidden.
 *
 *   g++ -O2 -o pdl_launch_cost pdl_launch_cost.cpp \
 *       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lnvrtc
 *   ./pdl_launch_cost              # 200000 launches, 7 repeats
 *   ./pdl_launch_cost 200000 7
 *
 * Check the GPU is idle first:
 *   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
 */
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
   const char *n_ = "?"; cuGetErrorName(r_, &n_); \
   fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, n_); \
   return 1; } } while (0)

static double now_ns(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

int main(int argc, char **argv)
{
   unsigned n = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 0) : 200000u;
   unsigned reps = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 7u;

   CK(cuInit(0));
   CUdevice dev; CK(cuDeviceGet(&dev, 0));
   CUcontext ctx; CK(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, dev));
   int major = 0, minor = 0;
   CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
   CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

   /* An empty kernel, compiled at run time so the program needs no nvcc and
    * no separate cubin to find. */
   nvrtcProgram prog;
   char arch[32];
   snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);
   const char *opts[] = { arch };
   if (nvrtcCreateProgram(&prog, "extern \"C\" __global__ void k(void) { }\n",
                          "k.cu", 0, NULL, NULL) != NVRTC_SUCCESS ||
       nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS) {
      fprintf(stderr, "nvrtc failed\n");
      return 1;
   }
   size_t ptx_size = 0;
   nvrtcGetPTXSize(prog, &ptx_size);
   char *ptx = (char *)malloc(ptx_size);
   nvrtcGetPTX(prog, ptx);
   nvrtcDestroyProgram(&prog);

   CUmodule mod; CK(cuModuleLoadData(&mod, ptx));
   CUfunction f; CK(cuModuleGetFunction(&f, mod, "k"));
   CUstream s; CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));

   CUlaunchAttribute attr;
   memset(&attr, 0, sizeof(attr));
   attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
   attr.value.programmaticStreamSerializationAllowed = 1;
   CUlaunchConfig cfg;
   memset(&cfg, 0, sizeof(cfg));
   cfg.gridDimX = cfg.gridDimY = cfg.gridDimZ = 1;
   cfg.blockDimX = cfg.blockDimY = cfg.blockDimZ = 1;
   cfg.hStream = s;

   double best[3] = { 1e30, 1e30, 1e30 }, sum[3] = { 0, 0, 0 };
   const char *name[3] = { "cuLaunchKernel      ",
                           "cuLaunchKernelEx a=0",
                           "cuLaunchKernelEx PDL" };

   for (unsigned r = 0; r < reps; r++) {
      for (int arm = 0; arm < 3; arm++) {
         cfg.attrs = arm == 2 ? &attr : NULL;
         cfg.numAttrs = arm == 2 ? 1u : 0u;

         /* One warm launch, so no lazy first-use work lands inside the timer. */
         if (arm == 0) CK(cuLaunchKernel(f, 1,1,1, 1,1,1, 0, s, NULL, NULL));
         else          CK(cuLaunchKernelEx(&cfg, f, NULL, NULL));
         CK(cuStreamSynchronize(s));

         double t0 = now_ns();
         if (arm == 0)
            for (unsigned i = 0; i < n; i++)
               cuLaunchKernel(f, 1,1,1, 1,1,1, 0, s, NULL, NULL);
         else
            for (unsigned i = 0; i < n; i++)
               cuLaunchKernelEx(&cfg, f, NULL, NULL);
         double per = (now_ns() - t0) / (double)n;
         CK(cuStreamSynchronize(s));

         sum[arm] += per;
         if (per < best[arm]) best[arm] = per;
      }
   }

   printf("sm_%d%d, %u launches x %u repeats, host time per launch (ns)\n",
          major, minor, n, reps);
   for (int a = 0; a < 3; a++)
      printf("  %s  min %7.1f   mean %7.1f\n", name[a], best[a], sum[a] / reps);
   printf("  premium, Ex entry point      %7.1f ns\n", best[1] - best[0]);
   printf("  premium, Ex + PDL attribute  %7.1f ns\n", best[2] - best[0]);
   printf("\nAt 550 extended launches a frame that is %.4f ms/frame;\n"
          "at 760 it is %.4f ms/frame.\n",
          (best[2] - best[0]) * 550.0 / 1e6, (best[2] - best[0]) * 760.0 / 1e6);

   cuStreamDestroy(s);
   cuModuleUnload(mod);
   cuCtxDestroy(ctx);
   free(ptx);
   return 0;
}
