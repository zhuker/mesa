/*
 * Best-case throughput gate for a persistent shader-bytecode interpreter.
 *
 * Build and run on the target GPU:
 *   nvcc -std=c++17 -O3 -arch=sm_120 --ptxas-options=-v \
 *     cp_shader_interp_bench.cu -o /tmp/cp_shader_interp_bench
 *   /tmp/cp_shader_interp_bench
 *
 * All threads execute the same cached bytecode and branch uniformly.  There
 * are no textures, divergent control flow, barriers, or fixed-function jobs.
 * This therefore favors the interpreter more than a Vulkan shader VM would.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                                   \
   cudaError_t err_ = (call);                                                    \
   if (err_ != cudaSuccess) {                                                    \
      fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,    \
              cudaGetErrorString(err_));                                         \
      exit(1);                                                                   \
   }                                                                             \
} while (0)

struct inst {
   unsigned char op, dst, a, b;
   float imm;
};

__global__ static void
interpret_mixed(const struct inst *code, int nops, const float *in, float *out,
                int n)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= n)
      return;

   /* Dynamic indices force the general VM register file into local memory. */
   float r[8];
#pragma unroll
   for (int i = 0; i < 8; i++)
      r[i] = in[(tid + i) & 1023] + i * .01f;

   for (int pc = 0; pc < nops; pc++) {
      struct inst x = code[pc];
      float a = r[x.a], b = r[x.b];
      switch (x.op) {
      case 0: r[x.dst] = fmaf(a, b, x.imm); break;
      case 1: r[x.dst] = a + b; break;
      case 2: r[x.dst] = a * b; break;
      default: r[x.dst] = fminf(a, b); break;
      }
   }

   out[tid] = r[0] + r[1] + r[2] + r[3] +
              r[4] + r[5] + r[6] + r[7];
}

/* An optimistic one-opcode VM: no opcode switch at all. */
__global__ static void
interpret_fma(const struct inst *code, int nops, const float *in, float *out,
              int n)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= n)
      return;

   float r[8];
#pragma unroll
   for (int i = 0; i < 8; i++)
      r[i] = in[(tid + i) & 1023] + i * .01f;
   for (int pc = 0; pc < nops; pc++) {
      struct inst x = code[pc];
      r[x.dst] = fmaf(r[x.a], .999f, r[x.b] * .001f);
   }
   out[tid] = r[0] + r[1] + r[2] + r[3] +
              r[4] + r[5] + r[6] + r[7];
}


__global__ static void
interpret_fma_shared(const struct inst *code, int nops, const float *in,
                     float *out, int n)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= n)
      return;
   extern __shared__ float regs[];
#define R(i) regs[(i) * blockDim.x + threadIdx.x]
#pragma unroll
   for (int i = 0; i < 8; i++)
      R(i) = in[(tid + i) & 1023] + i * .01f;
   for (int pc = 0; pc < nops; pc++) {
      struct inst x = code[pc];
      R(x.dst) = fmaf(R(x.a), .999f, R(x.b) * .001f);
   }
   out[tid] = R(0) + R(1) + R(2) + R(3) + R(4) + R(5) + R(6) + R(7);
#undef R
}

struct scalar_regs {
   float r0, r1, r2, r3, r4, r5, r6, r7;
};

__device__ __forceinline__ static float
reg_get(const struct scalar_regs &r, unsigned i)
{
   switch (i) {
   case 0: return r.r0; case 1: return r.r1;
   case 2: return r.r2; case 3: return r.r3;
   case 4: return r.r4; case 5: return r.r5;
   case 6: return r.r6; default: return r.r7;
   }
}

__device__ __forceinline__ static void
reg_set(struct scalar_regs &r, unsigned i, float v)
{
   switch (i) {
   case 0: r.r0 = v; break; case 1: r.r1 = v; break;
   case 2: r.r2 = v; break; case 3: r.r3 = v; break;
   case 4: r.r4 = v; break; case 5: r.r5 = v; break;
   case 6: r.r6 = v; break; default: r.r7 = v; break;
   }
}

__global__ static void
interpret_fma_scalar(const struct inst *code, int nops, const float *in,
                     float *out, int n)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= n)
      return;
   struct scalar_regs r = {
      in[(tid + 0) & 1023] + 0 * .01f, in[(tid + 1) & 1023] + 1 * .01f,
      in[(tid + 2) & 1023] + 2 * .01f, in[(tid + 3) & 1023] + 3 * .01f,
      in[(tid + 4) & 1023] + 4 * .01f, in[(tid + 5) & 1023] + 5 * .01f,
      in[(tid + 6) & 1023] + 6 * .01f, in[(tid + 7) & 1023] + 7 * .01f,
   };
   for (int pc = 0; pc < nops; pc++) {
      struct inst x = code[pc];
      reg_set(r, x.dst, fmaf(reg_get(r, x.a), .999f,
                             reg_get(r, x.b) * .001f));
   }
   out[tid] = r.r0 + r.r1 + r.r2 + r.r3 + r.r4 + r.r5 + r.r6 + r.r7;
}


__global__ static void
interpret_fma_quad(const struct inst *code, int nops, const float *in,
                   float *out, int n)
{
   int q = blockIdx.x * blockDim.x + threadIdx.x;
   int base = q * 4;
   if (base >= n)
      return;
   extern __shared__ float4 regs4[];
#define R4(i) regs4[(i) * blockDim.x + threadIdx.x]
#pragma unroll
   for (int i = 0; i < 8; i++) {
      R4(i) = make_float4(in[(base + 0 + i) & 1023] + i * .01f,
                          in[(base + 1 + i) & 1023] + i * .01f,
                          in[(base + 2 + i) & 1023] + i * .01f,
                          in[(base + 3 + i) & 1023] + i * .01f);
   }
   for (int pc = 0; pc < nops; pc++) {
      struct inst x = code[pc];
      float4 a = R4(x.a), b = R4(x.b), v;
      v.x = fmaf(a.x, .999f, b.x * .001f);
      v.y = fmaf(a.y, .999f, b.y * .001f);
      v.z = fmaf(a.z, .999f, b.z * .001f);
      v.w = fmaf(a.w, .999f, b.w * .001f);
      R4(x.dst) = v;
   }
   float4 sum = make_float4(0, 0, 0, 0);
#pragma unroll
   for (int i = 0; i < 8; i++) {
      float4 v = R4(i);
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
   }
   out[base + 0] = sum.x; out[base + 1] = sum.y;
   out[base + 2] = sum.z; out[base + 3] = sum.w;
#undef R4
}


template<int lanes>
__global__ static void
interpret_fma_vector(const struct inst *code, int nops, const float *in,
                     float *out, int n)
{
   int group = blockIdx.x * blockDim.x + threadIdx.x;
   int base = group * lanes;
   if (base >= n)
      return;
   extern __shared__ float regs_v[];
#define RV(reg, lane) regs_v[((reg) * lanes + (lane)) * blockDim.x + threadIdx.x]
#pragma unroll
   for (int i = 0; i < 8; i++)
#pragma unroll
      for (int lane = 0; lane < lanes; lane++)
         RV(i, lane) = in[(base + lane + i) & 1023] + i * .01f;
   for (int pc = 0; pc < nops; pc++) {
      struct inst x = code[pc];
#pragma unroll
      for (int lane = 0; lane < lanes; lane++)
         RV(x.dst, lane) = fmaf(RV(x.a, lane), .999f,
                                RV(x.b, lane) * .001f);
   }
#pragma unroll
   for (int lane = 0; lane < lanes; lane++) {
      float sum = 0;
#pragma unroll
      for (int i = 0; i < 8; i++)
         sum += RV(i, lane);
      out[base + lane] = sum;
   }
#undef RV
}

template<int pc, int nops, bool fma_only>
__device__ __forceinline__ void
compiled_program(float (&r)[8])
{
   if constexpr (pc < nops) {
      constexpr int d = pc % 8;
      constexpr int a = (pc + 1) % 8;
      constexpr int b = (pc + 3) % 8;
      if constexpr (fma_only)
         r[d] = fmaf(r[a], .999f, r[b] * .001f);
      else if constexpr (pc % 4 == 0)
         r[d] = fmaf(r[a], r[b], .001f * (pc + 1));
      else if constexpr (pc % 4 == 1)
         r[d] = r[a] + r[b];
      else if constexpr (pc % 4 == 2)
         r[d] = r[a] * r[b];
      else
         r[d] = fminf(r[a], r[b]);
      compiled_program<pc + 1, nops, fma_only>(r);
   }
}

template<int nops, bool fma_only>
__global__ static void
compiled(const float *in, float *out, int n)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= n)
      return;

   float r[8];
#pragma unroll
   for (int i = 0; i < 8; i++)
      r[i] = in[(tid + i) & 1023] + i * .01f;
   compiled_program<0, nops, fma_only>(r);
   out[tid] = r[0] + r[1] + r[2] + r[3] +
              r[4] + r[5] + r[6] + r[7];
}

template<typename launch>
static float
median_launch_ms(launch f)
{
   cudaEvent_t begin, end;
   CUDA_CHECK(cudaEventCreate(&begin));
   CUDA_CHECK(cudaEventCreate(&end));
   for (int i = 0; i < 10; i++)
      f();
   CUDA_CHECK(cudaDeviceSynchronize());

   std::vector<float> samples;
   for (int sample = 0; sample < 9; sample++) {
      CUDA_CHECK(cudaEventRecord(begin));
      for (int i = 0; i < 100; i++)
         f();
      CUDA_CHECK(cudaEventRecord(end));
      CUDA_CHECK(cudaEventSynchronize(end));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      samples.push_back(ms / 100.0f);
   }
   std::sort(samples.begin(), samples.end());
   CUDA_CHECK(cudaEventDestroy(begin));
   CUDA_CHECK(cudaEventDestroy(end));
   return samples[samples.size() / 2];
}

#define RUN_MIXED(nops) do {                                                     \
   float vm = median_launch_ms([&] {                                             \
      interpret_mixed<<<grid, block>>>(dev_code, nops, in, out, n);             \
   });                                                                           \
   float native = median_launch_ms([&] {                                         \
      compiled<nops, false><<<grid, block>>>(in, out, n);                       \
   });                                                                           \
   printf("mixed ops=%d compiled=%.6f ms interpreter=%.6f ms ratio=%.3fx\n",  \
          nops, native, vm, vm / native);                                        \
} while (0)

#define RUN_FMA(nops) do {                                                       \
   float vm = median_launch_ms([&] {                                             \
      interpret_fma<<<grid, block>>>(dev_code, nops, in, out, n);               \
   });                                                                           \
   float native = median_launch_ms([&] {                                         \
      compiled<nops, true><<<grid, block>>>(in, out, n);                        \
   });                                                                           \
   printf("one-opcode ops=%d compiled=%.6f ms interpreter=%.6f ms "            \
          "ratio=%.3fx\n", nops, native, vm, vm / native);                    \
} while (0)


#define RUN_SHARED(nops) do {                                                    \
   float vm = median_launch_ms([&] {                                             \
      interpret_fma_shared<<<grid, block, 8 * block.x * sizeof(float)>>>(       \
         dev_code, nops, in, out, n);                                            \
   });                                                                           \
   float native = median_launch_ms([&] {                                         \
      compiled<nops, true><<<grid, block>>>(in, out, n);                        \
   });                                                                           \
   printf("shared-register ops=%d compiled=%.6f ms interpreter=%.6f ms "       \
          "ratio=%.3fx\n", nops, native, vm, vm / native);                   \
} while (0)
#define RUN_SCALAR(nops) do {                                                    \
   float vm = median_launch_ms([&] {                                             \
      interpret_fma_scalar<<<grid, block>>>(dev_code, nops, in, out, n);         \
   });                                                                           \
   float native = median_launch_ms([&] {                                         \
      compiled<nops, true><<<grid, block>>>(in, out, n);                        \
   });                                                                           \
   printf("scalar-switch ops=%d compiled=%.6f ms interpreter=%.6f ms "         \
          "ratio=%.3fx\n", nops, native, vm, vm / native);                   \
} while (0)


#define RUN_QUAD(nops) do {                                                      \
   dim3 qgrid(((n / 4) + block.x - 1) / block.x);                               \
   float vm = median_launch_ms([&] {                                             \
      interpret_fma_quad<<<qgrid, block, 8 * block.x * sizeof(float4)>>>(       \
         dev_code, nops, in, out, n);                                            \
   });                                                                           \
   float native = median_launch_ms([&] {                                         \
      compiled<nops, true><<<grid, block>>>(in, out, n);                        \
   });                                                                           \
   printf("quad-shared ops=%d compiled=%.6f ms interpreter=%.6f ms "           \
          "ratio=%.3fx\n", nops, native, vm, vm / native);                   \
} while (0)


#define RUN_VECTOR(nops, lanes, threads) do {                                    \
   dim3 vblock(threads);                                                         \
   dim3 vgrid(((n / lanes) + vblock.x - 1) / vblock.x);                         \
   float vm = median_launch_ms([&] {                                             \
      interpret_fma_vector<lanes><<<vgrid, vblock,                              \
         8 * lanes * vblock.x * sizeof(float)>>>(dev_code, nops, in, out, n);   \
   });                                                                           \
   float native = median_launch_ms([&] {                                         \
      compiled<nops, true><<<grid, block>>>(in, out, n);                        \
   });                                                                           \
   printf("vector%d ops=%d compiled=%.6f ms interpreter=%.6f ms "              \
          "ratio=%.3fx\n", lanes, nops, native, vm, vm / native);            \
} while (0)

int
main(void)
{
   constexpr int n = 1 << 20;
   float *in, *out;
   CUDA_CHECK(cudaMallocManaged(&in, 1024 * sizeof(*in)));
   CUDA_CHECK(cudaMalloc(&out, n * sizeof(*out)));
   for (int i = 0; i < 1024; i++)
      in[i] = 1.0001f + i * .00001f;

   struct inst host_code[128];
   for (int i = 0; i < 128; i++) {
      host_code[i] = {(unsigned char)(i % 4), (unsigned char)(i % 8),
                      (unsigned char)((i + 1) % 8),
                      (unsigned char)((i + 3) % 8), .001f * (i + 1)};
   }
   struct inst *dev_code;
   CUDA_CHECK(cudaMalloc(&dev_code, sizeof(host_code)));
   CUDA_CHECK(cudaMemcpy(dev_code, host_code, sizeof(host_code),
                         cudaMemcpyHostToDevice));

   dim3 block(256), grid((n + block.x - 1) / block.x);
   RUN_MIXED(8);
   RUN_MIXED(16);
   RUN_MIXED(32);
   RUN_MIXED(64);
   RUN_MIXED(128);
   RUN_FMA(8);
   RUN_FMA(16);
   RUN_FMA(32);
   RUN_FMA(64);
   RUN_FMA(128);
   RUN_SHARED(8);
   RUN_SHARED(16);
   RUN_SHARED(32);
   RUN_SHARED(64);
   RUN_QUAD(8);
   RUN_QUAD(16);
   RUN_QUAD(32);
   RUN_QUAD(64);
   RUN_VECTOR(8, 8, 128);
   RUN_VECTOR(16, 8, 128);
   RUN_VECTOR(32, 8, 128);
   RUN_VECTOR(64, 8, 128);
   RUN_VECTOR(8, 16, 64);
   RUN_VECTOR(16, 16, 64);
   RUN_VECTOR(32, 16, 64);
   RUN_VECTOR(64, 16, 64);
   RUN_SCALAR(8);
   RUN_SCALAR(16);
   RUN_SCALAR(32);
   RUN_SCALAR(64);
   CUDA_CHECK(cudaGetLastError());
   CUDA_CHECK(cudaDeviceSynchronize());
   CUDA_CHECK(cudaFree(dev_code));
   CUDA_CHECK(cudaFree(out));
   CUDA_CHECK(cudaFree(in));
   return 0;
}
