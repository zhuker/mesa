/* Focused correctness gate for the default-off run clip/bin CUDA ABI. */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "../kernels/cp_rasterize.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
   fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, \
           cudaGetErrorString(e)); exit(1); } } while (0)
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "check failed: %s (%s:%d)\n", \
   #x, __FILE__, __LINE__); exit(1); } } while (0)

/* Trusted comparison: the existing per-segment clip entry point runs first;
 * this wrapper only applies the existing setup_triangle() helper to its fixed
 * stable slots and records the same 32-pixel references. */
__global__ static void
reference_setup(struct cp_run_clip_bin_device_segment seg, uint32_t seg_idx,
                struct cp_run_clip_bin_args args)
{
   struct cp_rasterize_args rast = seg.rast;
   rast.positions = seg.clip.out;
   rast.prim_refs = seg.clip.prim_refs;
   rast.num_triangles = seg.clip.max_triangles;
   rast.num_varyings = seg.clip.num_slots - 1;
   rast.tri_count = 0; rast.active_ids = 0;
   for (uint32_t id = threadIdx.x; id < seg.clip.max_triangles;
        id += blockDim.x) {
      struct cp_tri_setup setup = {};
      if (!setup_triangle(&rast, id, &setup)) continue;
      int tx0 = max(setup.ix_min, 0) / (int)CP_RUN_CLIP_BIN_TILE_SIZE;
      int ty0 = max(setup.iy_min, 0) / (int)CP_RUN_CLIP_BIN_TILE_SIZE;
      int tx1 = min(setup.ix_max,
                    (int)(args.tiles_x * CP_RUN_CLIP_BIN_TILE_SIZE - 1u)) /
                (int)CP_RUN_CLIP_BIN_TILE_SIZE;
      int ty1 = min(setup.iy_max,
                    (int)(args.tiles_y * CP_RUN_CLIP_BIN_TILE_SIZE - 1u)) /
                (int)CP_RUN_CLIP_BIN_TILE_SIZE;
      if (tx0 > tx1 || ty0 > ty1) continue;
      uint32_t at = atomicAdd((uint32_t *)(uintptr_t)args.live_count, 1u);
      if (at >= args.live_capacity) {
         atomicExch((uint32_t *)(uintptr_t)args.overflow, 1u); continue;
      }
      ((cp_run_clip_bin_live *)(uintptr_t)args.live)[at] =
         {seg.prim_base + id, seg_idx};
      uint32_t *counts = (uint32_t *)(uintptr_t)args.tile_counts;
      for (int ty = ty0; ty <= ty1; ty++)
         for (int tx = tx0; tx <= tx1; tx++)
            atomicAdd(counts + ty * args.tiles_x + tx, 1u);
   }
}

static cp_rasterize_args rast(unsigned cull, int x0, int y0, int x1, int y1,
                              float sx, float sy, float tx, float ty)
{
   cp_rasterize_args a = {};
   a.width = 128; a.height = 96; a.num_samples = 1;
   a.vp_scale_x = sx; a.vp_scale_y = sy;
   a.vp_trans_x = tx; a.vp_trans_y = ty;
   a.clip_x0 = x0; a.clip_y0 = y0; a.clip_x1 = x1; a.clip_y1 = y1;
   a.cull_mode = cull;
   return a;
}

int main(void)
{
   CHECK(CP_RUN_CLIP_BIN_TILE_SIZE == 32);
   constexpr uint32_t N = 6, SLOTS = N * CP_CLIP_MAX_OUT;
   const float4 tris[N][3] = {
      /* fully inside */
      {{-0.5f,-0.5f,0.5f,1},{0.5f,-0.5f,0.5f,1},{0,0.5f,0.5f,1}},
      /* crosses the right homogeneous clip plane */
      {{0.2f,-0.6f,0.5f,1},{1.8f,-0.2f,0.5f,1},{0.2f,0.7f,0.5f,1}},
      /* entirely behind z=0 */
      {{-0.3f,-0.3f,-0.5f,1},{0.3f,-0.3f,-0.5f,1},{0,0.3f,-0.5f,1}},
      /* fully inside, but culled by winding */
      {{-0.4f,-0.4f,0.5f,1},{0.4f,-0.4f,0.5f,1},{0,0.4f,0.5f,1}},
      /* same geometry under a small, offset viewport/scissor */
      {{-0.9f,-0.9f,0.5f,1},{0.9f,-0.9f,0.5f,1},{0,0.9f,0.5f,1}},
      /* second live segment, disjoint scissor */
      {{-0.9f,-0.9f,0.5f,1},{0.9f,-0.9f,0.5f,1},{0,0.9f,0.5f,1}},
   };

   cp_run_clip_bin_device_segment hs[N] = {};
   float4 *din[N], *dout[N]; uint64_t *drefs[N]; uint32_t *dcount[N];
   cp_rasterize_args rs[N] = {
      rast(0,0,0,127,95,64,48,64,48),
      rast(0,0,0,127,95,64,48,64,48),
      rast(0,0,0,127,95,64,48,64,48),
      rast(2,0,0,127,95,64,48,64,48),
      rast(0,16,8,47,39,24,20,32,24),
      rast(0,80,48,111,79,24,20,96,64),
   };
   for (uint32_t i = 0; i < N; i++) {
      CK(cudaMalloc(&din[i], sizeof(tris[i])));
      CK(cudaMalloc(&dout[i], CP_CLIP_MAX_OUT * 3 * sizeof(float4)));
      CK(cudaMalloc(&drefs[i], CP_CLIP_MAX_OUT * sizeof(uint64_t)));
      CK(cudaMalloc(&dcount[i], sizeof(uint32_t)));
      CK(cudaMemcpy(din[i], tris[i], sizeof(tris[i]), cudaMemcpyHostToDevice));
      CK(cudaMemset(dout[i], 0, CP_CLIP_MAX_OUT * 3 * sizeof(float4)));
      CK(cudaMemset(drefs[i], 0, CP_CLIP_MAX_OUT * sizeof(uint64_t)));
      CK(cudaMemset(dcount[i], 0, sizeof(uint32_t)));
      hs[i].clip.vs_out = (uint64_t)din[i]; hs[i].clip.out = (uint64_t)dout[i];
      hs[i].clip.out_count = (uint64_t)dcount[i];
      hs[i].clip.prim_refs = (uint64_t)drefs[i];
      hs[i].clip.num_triangles = 1; hs[i].clip.num_slots = 1;
      hs[i].clip.max_triangles = CP_CLIP_MAX_OUT; hs[i].clip.stable = 1;
      hs[i].rast = rs[i]; hs[i].prim_base = i * CP_CLIP_MAX_OUT;
      hs[i].draw_row = 100 + i;
   }

   cp_run_clip_bin_device_segment *ds;
   cp_run_clip_bin_live *run_live, *ref_live;
   uint32_t *run_counts, *ref_counts, *run_n, *ref_n, *run_over, *ref_over;
   const uint32_t TX=4, TY=3;
   CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
   CK(cudaMalloc(&run_live, SLOTS*sizeof(*run_live)));
   CK(cudaMalloc(&ref_live, SLOTS*sizeof(*ref_live)));
   CK(cudaMalloc(&run_counts, TX*TY*sizeof(uint32_t))); CK(cudaMalloc(&ref_counts, TX*TY*sizeof(uint32_t)));
   CK(cudaMalloc(&run_n,4)); CK(cudaMalloc(&ref_n,4)); CK(cudaMalloc(&run_over,4)); CK(cudaMalloc(&ref_over,4));
   CK(cudaMemset(run_counts,0,TX*TY*4)); CK(cudaMemset(ref_counts,0,TX*TY*4));
   CK(cudaMemset(run_n,0,4)); CK(cudaMemset(ref_n,0,4)); CK(cudaMemset(run_over,0,4)); CK(cudaMemset(ref_over,0,4));

   cp_run_clip_bin_args run = {(uint64_t)ds,(uint64_t)run_live,(uint64_t)run_n,
      (uint64_t)run_over,(uint64_t)run_counts,N,SLOTS,SLOTS,TX,TY};
   cp_run_clip_setup_tile_count<<<dim3(1,N),64>>>(run);
   CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

   cp_run_clip_bin_args ref = run; ref.live=(uint64_t)ref_live;
   ref.live_count=(uint64_t)ref_n; ref.overflow=(uint64_t)ref_over;
   ref.tile_counts=(uint64_t)ref_counts;
   for (uint32_t i=0;i<N;i++) {
      cp_clip_triangles<<<1,32>>>(hs[i].clip);
      reference_setup<<<1,32>>>(hs[i],i,ref);
   }
   CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

   uint32_t an=0,bn=0,ao=0,bo=0;
   CK(cudaMemcpy(&an,run_n,4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&bn,ref_n,4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(&ao,run_over,4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&bo,ref_over,4,cudaMemcpyDeviceToHost));
   CHECK(an==bn && !ao && !bo && an<=SLOTS);
   std::vector<cp_run_clip_bin_live> a(an), b(bn);
   std::vector<uint32_t> ac(TX*TY), bc(TX*TY);
   CK(cudaMemcpy(a.data(),run_live,an*sizeof(a[0]),cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(b.data(),ref_live,bn*sizeof(b[0]),cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(ac.data(),run_counts,ac.size()*4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(bc.data(),ref_counts,bc.size()*4,cudaMemcpyDeviceToHost));
   auto less=[](const cp_run_clip_bin_live &x,const cp_run_clip_bin_live &y) {
      return x.stable_prim<y.stable_prim || (x.stable_prim==y.stable_prim && x.segment<y.segment);
   };
   std::sort(a.begin(),a.end(),less); std::sort(b.begin(),b.end(),less);
   CHECK(memcmp(a.data(),b.data(),an*sizeof(a[0])) == 0); CHECK(ac == bc);
   auto has=[&](uint32_t id){ for(auto &v:a) if(v.stable_prim==id) return true; return false; };
   CHECK(has(0)); CHECK(!has(2*CP_CLIP_MAX_OUT)); CHECK(!has(3*CP_CLIP_MAX_OUT));
   CHECK(has(4*CP_CLIP_MAX_OUT) && has(5*CP_CLIP_MAX_OUT));
   unsigned clipped=0; for(auto &v:a) if(v.segment==1) clipped++; CHECK(clipped>=2);
   uint32_t refs=0; for(uint32_t v:ac) refs+=v; CHECK(refs>0);
   /* Counts define the exact number of future dense (stable ID, segment)
    * references. Different clip rectangles must occupy disjoint tile columns. */
   CHECK(ac[0]+ac[4] > 0); CHECK(ac[2]+ac[6]+ac[10] > 0);
   printf("run clip/bin gate: %u segments, %u live, %u tile references -- PASS\n",N,an,refs);
   return 0;
}
