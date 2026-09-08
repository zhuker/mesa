/*
 * Pricing the last idea for the 4x goal (FOURX_DIAGNOSIS_2026-09-05.md).
 *
 * cudavk's fragment shading is 97% stall: 2.77 ms/frame heavy against an
 * instruction-issue floor of 0.08 ms, 64-77% long_scoreboard, DRAM 0.14%.
 * The stalls are the per-fragment gather of the owning triangle's vertex
 * attributes out of global memory.
 *
 * This measures what tile-local staging would buy, at the driver's real
 * shapes: for each fragment, read 3 vertices x VARYINGS floats and
 * interpolate, either
 *   A) gathered from global memory by triangle id (what the driver does), or
 *   B) read from shared memory after the tile's triangles are staged once
 *      (what a tile-resident shading pass would do).
 *
 * Everything else -- the arithmetic, the fragment count, the varying count --
 * is identical between the arms, so the difference is the memory model alone.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define VARY 8            /* floats per vertex: the driver's median varying count */
#define TILE_TRIS 64      /* triangles staged per tile */
#define NTHREADS 256

/* A: gather from global, triangle id per fragment (scattered) */
__global__ __launch_bounds__(NTHREADS) void
shade_gather(const float* __restrict__ attrs, const uint32_t* __restrict__ tri,
             const float3* __restrict__ bary, uint32_t n, float* __restrict__ out)
{
   uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
   if (i >= n) return;
   uint32_t t = tri[i];
   float3 b = bary[i];
   float acc = 0.f;
   #pragma unroll
   for (int v = 0; v < VARY; v++) {
      float a0 = attrs[(size_t)(t*3+0)*VARY + v];
      float a1 = attrs[(size_t)(t*3+1)*VARY + v];
      float a2 = attrs[(size_t)(t*3+2)*VARY + v];
      acc += b.x*a0 + b.y*a1 + b.z*a2;
   }
   out[i] = acc;
}

/* B: the block's triangles staged in shared memory once, then read from there */
__global__ __launch_bounds__(NTHREADS) void
shade_tiled(const float* __restrict__ attrs, const uint32_t* __restrict__ tri,
            const float3* __restrict__ bary, uint32_t n, float* __restrict__ out,
            uint32_t tris_per_block)
{
   __shared__ float s[TILE_TRIS*3*VARY];
   uint32_t base_tri = blockIdx.x * tris_per_block;
   uint32_t m = tris_per_block < TILE_TRIS ? tris_per_block : TILE_TRIS;
   for (uint32_t k = threadIdx.x; k < m*3*VARY; k += blockDim.x)
      s[k] = attrs[(size_t)base_tri*3*VARY + k];
   __syncthreads();

   uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
   if (i >= n) return;
   uint32_t t = tri[i] % m;            /* the tile's own triangles */
   float3 b = bary[i];
   float acc = 0.f;
   #pragma unroll
   for (int v = 0; v < VARY; v++) {
      float a0 = s[(t*3+0)*VARY + v];
      float a1 = s[(t*3+1)*VARY + v];
      float a2 = s[(t*3+2)*VARY + v];
      acc += b.x*a0 + b.y*a1 + b.z*a2;
   }
   out[i] = acc;
}

int main(int argc, char** argv)
{
   uint32_t n = (argc>1)? atoi(argv[1]) : 2000000;   /* fragments per frame */
   uint32_t ntris = 354000;
   cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
   printf("device %s, %d SMs\n", p.name, p.multiProcessorCount);
   printf("%u fragments, %u triangles, %d varyings/vertex\n", n, ntris, VARY);

   std::mt19937 rng(7); std::uniform_int_distribution<uint32_t> ti(0,ntris-1);
   std::vector<uint32_t> h_tri(n); std::vector<float3> h_b(n);
   for (uint32_t i=0;i<n;i++){ h_tri[i]=ti(rng); h_b[i]=make_float3(0.3f,0.3f,0.4f); }
   std::vector<float> h_a((size_t)ntris*3*VARY);
   for (auto &x : h_a) x = 1.0f;

   float *d_a,*d_out; uint32_t *d_tri; float3 *d_b;
   CK(cudaMalloc(&d_a,h_a.size()*4)); CK(cudaMalloc(&d_out,(size_t)n*4));
   CK(cudaMalloc(&d_tri,(size_t)n*4)); CK(cudaMalloc(&d_b,(size_t)n*12));
   CK(cudaMemcpy(d_a,h_a.data(),h_a.size()*4,cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_tri,h_tri.data(),(size_t)n*4,cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_b,h_b.data(),(size_t)n*12,cudaMemcpyHostToDevice));

   uint32_t blocks=(n+NTHREADS-1)/NTHREADS;
   uint32_t tpb = ntris/blocks; if(!tpb) tpb=1; if(tpb>TILE_TRIS) tpb=TILE_TRIS;
   cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
   const int reps=200;
   for(int i=0;i<5;i++) shade_gather<<<blocks,NTHREADS>>>(d_a,d_tri,d_b,n,d_out);
   CK(cudaDeviceSynchronize());
   CK(cudaEventRecord(a));
   for(int i=0;i<reps;i++) shade_gather<<<blocks,NTHREADS>>>(d_a,d_tri,d_b,n,d_out);
   CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
   float g; CK(cudaEventElapsedTime(&g,a,b)); g/=reps;

   for(int i=0;i<5;i++) shade_tiled<<<blocks,NTHREADS>>>(d_a,d_tri,d_b,n,d_out,tpb);
   CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
   CK(cudaEventRecord(a));
   for(int i=0;i<reps;i++) shade_tiled<<<blocks,NTHREADS>>>(d_a,d_tri,d_b,n,d_out,tpb);
   CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
   float t; CK(cudaEventElapsedTime(&t,a,b)); t/=reps;

   printf("\n  A gather from global (driver today) : %.4f ms\n", g);
   printf("  B tile-staged in shared             : %.4f ms   (%.2fx)\n", t, g/t);
   printf("\n  cudavk main, heavy band: 2.77 ms for ~%u fragments\n", n);
   return 0;
}
