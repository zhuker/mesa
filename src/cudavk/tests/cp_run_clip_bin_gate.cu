/* Focused correctness gate for the default-off run clip/bin/direct CUDA ABI. */
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


__global__ static void
reference_direct_setup(struct cp_run_clip_bin_device_segment seg,
                       uint32_t seg_idx, cp_run_clip_direct_live *live,
                       uint32_t *count)
{
   struct cp_rasterize_args rast = seg.rast;
   rast.positions=seg.clip.out; rast.prim_refs=seg.clip.prim_refs;
   rast.num_triangles=seg.clip.max_triangles;
   rast.num_varyings=seg.clip.num_slots-1; rast.tri_count=0; rast.active_ids=0;
   for (uint32_t id=threadIdx.x; id<seg.clip.max_triangles; id+=blockDim.x) {
      cp_tri_setup setup={}; if (!setup_triangle(&rast,id,&setup)) continue;
      uint32_t at=atomicAdd(count,1u);
      live[at].setup=setup; live[at].stable_prim=seg.prim_base+id;
      live[at].segment=seg_idx;
   }
}


__global__ static void
reference_fill(struct cp_run_clip_bin_device_segment seg, uint32_t seg_idx,
               const uint32_t *counts, const uint32_t *offsets,
               uint32_t *cursors, cp_opaque_tile_ref *refs,
               uint32_t tiles_x, uint32_t tiles_y)
{
   struct cp_rasterize_args rast = seg.rast;
   rast.positions=seg.clip.out; rast.prim_refs=seg.clip.prim_refs;
   rast.num_triangles=seg.clip.max_triangles;
   rast.num_varyings=seg.clip.num_slots-1; rast.tri_count=0; rast.active_ids=0;
   for (uint32_t id=threadIdx.x; id<seg.clip.max_triangles; id+=blockDim.x) {
      cp_tri_setup setup={}; if (!setup_triangle(&rast,id,&setup)) continue;
      uint32_t tx0=(uint32_t)setup.ix_min/CP_RUN_CLIP_BIN_TILE_SIZE;
      uint32_t ty0=(uint32_t)setup.iy_min/CP_RUN_CLIP_BIN_TILE_SIZE;
      uint32_t tx1=(uint32_t)setup.ix_max/CP_RUN_CLIP_BIN_TILE_SIZE;
      uint32_t ty1=(uint32_t)setup.iy_max/CP_RUN_CLIP_BIN_TILE_SIZE;
      for(uint32_t ty=ty0;ty<=ty1&&ty<tiles_y;ty++) for(uint32_t tx=tx0;tx<=tx1&&tx<tiles_x;tx++) {
         uint32_t t=ty*tiles_x+tx, pos=atomicAdd(cursors+t,1u);
         if(pos<counts[t]) refs[offsets[t]+pos]={seg.prim_base+id,(uint16_t)seg_idx,0};
      }
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
   }

   cp_run_clip_bin_device_segment *ds;
   cp_run_clip_bin_live *run_live, *ref_live;
   uint32_t *run_counts, *ref_counts, *run_n, *ref_n, *run_over, *ref_over;
   uint32_t *offsets, *run_cursors, *ref_cursors, *errors;
   uint64_t *checked_total;
   cp_opaque_tile_ref *run_refs, *ref_refs;
   const uint32_t TX=4, TY=3;
   CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
   CK(cudaMalloc(&run_live, SLOTS*sizeof(*run_live)));
   CK(cudaMalloc(&ref_live, SLOTS*sizeof(*ref_live)));
   CK(cudaMalloc(&run_counts, TX*TY*sizeof(uint32_t))); CK(cudaMalloc(&ref_counts, TX*TY*sizeof(uint32_t)));
   CK(cudaMalloc(&run_n,4)); CK(cudaMalloc(&ref_n,4)); CK(cudaMalloc(&run_over,4)); CK(cudaMalloc(&ref_over,4));
   CK(cudaMalloc(&offsets,TX*TY*4)); CK(cudaMalloc(&run_cursors,TX*TY*4));
   CK(cudaMalloc(&ref_cursors,TX*TY*4)); CK(cudaMalloc(&errors,4));
   CK(cudaMalloc(&checked_total, sizeof(*checked_total)));
   CK(cudaMalloc(&run_refs,SLOTS*TX*TY*sizeof(*run_refs)));
   CK(cudaMalloc(&ref_refs,SLOTS*TX*TY*sizeof(*ref_refs)));
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

   /* Direct arm: fused clipping/setup must compact byte-identical setup to
    * the legacy clip followed by the shared setup helper. */
   cp_run_clip_direct_live *direct_live, *direct_ref;
   uint32_t *direct_n, *direct_ref_n;
   CK(cudaMalloc(&direct_live,SLOTS*sizeof(*direct_live)));
   CK(cudaMalloc(&direct_ref,SLOTS*sizeof(*direct_ref)));
   CK(cudaMalloc(&direct_n,4)); CK(cudaMalloc(&direct_ref_n,4));
   CK(cudaMemset(direct_n,0,4)); CK(cudaMemset(direct_ref_n,0,4));
   CK(cudaMemset(run_over,0,4));
   cp_run_clip_bin_args direct=run; direct.live=(uint64_t)direct_live;
   direct.live_count=(uint64_t)direct_n;
   cp_run_clip_setup_direct<<<dim3(1,N),64>>>(direct);
   for(uint32_t i=0;i<N;i++)
      reference_direct_setup<<<1,32>>>(hs[i],i,direct_ref,direct_ref_n);
   CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
   uint32_t dn=0,drn=0; CK(cudaMemcpy(&dn,direct_n,4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(&drn,direct_ref_n,4,cudaMemcpyDeviceToHost));
   CHECK(dn==drn && dn==an);
   std::vector<cp_run_clip_direct_live> dl(dn), dr(drn);
   CK(cudaMemcpy(dl.data(),direct_live,dn*sizeof(dl[0]),cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(dr.data(),direct_ref,drn*sizeof(dr[0]),cudaMemcpyDeviceToHost));
   auto dless=[](const cp_run_clip_direct_live&x,const cp_run_clip_direct_live&y){
      return x.stable_prim<y.stable_prim;
   };
   std::sort(dl.begin(),dl.end(),dless); std::sort(dr.begin(),dr.end(),dless);
   CHECK(memcmp(dl.data(),dr.data(),dn*sizeof(dl[0]))==0);

   /* The trusted path above supplies counts. Scan them on the CPU, as a
    * standalone gate may; the eventual renderer scan remains device-side. */
   std::vector<uint32_t> off(ac.size()); uint32_t total_refs=0;
   for(size_t i=0;i<ac.size();i++){ off[i]=total_refs; total_refs+=ac[i]; }
   CK(cudaMemcpy(offsets,off.data(),off.size()*4,cudaMemcpyHostToDevice));
   CK(cudaMemset(errors, 0, 4));
   cp_run_clip_bin_check_counts_args check_counts = {
      (uint64_t)run_counts, (uint64_t)checked_total, (uint64_t)errors,
      TX * TY, total_refs};
   cp_run_clip_bin_check_counts<<<1, 256>>>(check_counts);
   CK(cudaDeviceSynchronize());
   uint64_t checked = 0;
   CK(cudaMemcpy(&checked, checked_total, sizeof(checked), cudaMemcpyDeviceToHost));
   uint32_t check_error = 0;
   CK(cudaMemcpy(&check_error, errors, 4, cudaMemcpyDeviceToHost));
   CHECK(checked == total_refs && check_error == 0);
   CK(cudaMemset(run_cursors,0,TX*TY*4)); CK(cudaMemset(ref_cursors,0,TX*TY*4));
   CK(cudaMemset(errors,0,4));
   cp_run_clip_bin_fill_args fill={(uint64_t)ds,(uint64_t)run_live,(uint64_t)run_n,
      (uint64_t)run_counts,(uint64_t)offsets,(uint64_t)run_cursors,
      (uint64_t)run_refs,(uint64_t)errors,N,SLOTS,SLOTS,TX,TY,total_refs};
   cp_run_clip_bin_fill<<<1,128>>>(fill);
   cp_run_clip_bin_validate_cursors<<<1,64>>>(fill);
   for(uint32_t i=0;i<N;i++) reference_fill<<<1,32>>>(hs[i],i,ref_counts,offsets,
      ref_cursors,ref_refs,TX,TY);
   CK(cudaDeviceSynchronize()); uint32_t err=0;
   CK(cudaMemcpy(&err,errors,4,cudaMemcpyDeviceToHost)); CHECK(err==0);
   std::vector<uint32_t> rc(TX*TY),qc(TX*TY);
   CK(cudaMemcpy(rc.data(),run_cursors,rc.size()*4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(qc.data(),ref_cursors,qc.size()*4,cudaMemcpyDeviceToHost));
   CHECK(rc==ac && qc==ac);
   std::vector<cp_opaque_tile_ref> rr(total_refs), qr(total_refs);
   CK(cudaMemcpy(rr.data(),run_refs,total_refs*sizeof(rr[0]),cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(qr.data(),ref_refs,total_refs*sizeof(qr[0]),cudaMemcpyDeviceToHost));
   auto rless=[](const cp_opaque_tile_ref&x,const cp_opaque_tile_ref&y){
      return x.global_prim<y.global_prim || (x.global_prim==y.global_prim && x.segment<y.segment); };
   for(size_t t=0;t<ac.size();t++) {
      std::sort(rr.begin()+off[t],rr.begin()+off[t]+ac[t],rless);
      std::sort(qr.begin()+off[t],qr.begin()+off[t]+ac[t],rless);
      CHECK(memcmp(rr.data()+off[t],qr.data()+off[t],ac[t]*sizeof(rr[0]))==0);
   }
   /* Compare the direct visibility result with the old exact tile arm. */
   uint64_t *direct_vis, *tile_vis;
   cp_rasterize_args *tile_rasts;
   CK(cudaMalloc(&direct_vis,128*96*sizeof(uint64_t)));
   CK(cudaMalloc(&tile_vis,128*96*sizeof(uint64_t)));
   CK(cudaMalloc(&tile_rasts,N*sizeof(*tile_rasts)));
   CK(cudaMemset(direct_vis,0xff,128*96*sizeof(uint64_t)));
   CK(cudaMemset(tile_vis,0xff,128*96*sizeof(uint64_t)));
   for(uint32_t i=0;i<N;i++) {
      hs[i].rast.framebuffer=(uint64_t)direct_vis;
      rs[i]=hs[i].rast;
      rs[i].positions=hs[i].clip.out;
      rs[i].prim_refs=hs[i].clip.prim_refs;
      rs[i].num_triangles=hs[i].clip.max_triangles;
      rs[i].num_varyings=hs[i].clip.num_slots-1;
      rs[i].abuf_prim_base=hs[i].prim_base;
      rs[i].framebuffer=(uint64_t)tile_vis;
   }
   CK(cudaMemcpy(ds,hs,sizeof(hs),cudaMemcpyHostToDevice));
   CK(cudaMemcpy(tile_rasts,rs,sizeof(rs),cudaMemcpyHostToDevice));
   CK(cudaMemset(errors,0,4));
   cp_run_clip_direct_raster_args dra={(uint64_t)ds,(uint64_t)direct_live,
      (uint64_t)direct_n,(uint64_t)errors,N,SLOTS,SLOTS};
   cp_run_clip_direct_raster<<<8,256>>>(dra);
   cp_run_clip_direct_raster_large<<<8,256>>>(dra);
   cp_opaque_tile_raster_args ota={(uint64_t)tile_rasts,(uint64_t)run_counts,
      (uint64_t)offsets,(uint64_t)run_refs,(uint64_t)errors,N,TX,TY,128,96};
   cp_opaque_tile_raster<<<TX*TY,256>>>(ota);
   CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
   std::vector<uint64_t> dv(128*96),tv(128*96);
   CK(cudaMemcpy(dv.data(),direct_vis,dv.size()*8,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(tv.data(),tile_vis,tv.size()*8,cudaMemcpyDeviceToHost));
   CHECK(dv==tv);

   auto has=[&](uint32_t id){ for(auto &v:a) if(v.stable_prim==id) return true; return false; };
   CHECK(has(0)); CHECK(!has(2*CP_CLIP_MAX_OUT)); CHECK(!has(3*CP_CLIP_MAX_OUT));
   CHECK(has(4*CP_CLIP_MAX_OUT) && has(5*CP_CLIP_MAX_OUT));
   unsigned clipped=0; for(auto &v:a) if(v.segment==1) clipped++; CHECK(clipped>=2);
   uint32_t refs=total_refs; CHECK(refs>0);
   /* Counts define the exact number of future dense (stable ID, segment)
    * references. Different clip rectangles must occupy disjoint tile columns. */
   CHECK(ac[0]+ac[4] > 0); CHECK(ac[2]+ac[6]+ac[10] > 0);

   /* Capacity, record bounds, and cursor mismatch each have distinct bits. */
   CK(cudaMemset(errors,0,4)); fill.capacity=total_refs-1;
   CK(cudaMemset(run_cursors,0,TX*TY*4)); cp_run_clip_bin_fill<<<1,128>>>(fill);
   cp_run_clip_bin_validate_cursors<<<1,64>>>(fill); CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&err,errors,4,cudaMemcpyDeviceToHost));
   CHECK(err & CP_RUN_CLIP_BIN_ERR_REF_OVERFLOW);
   CK(cudaMemset(errors,0,4)); CK(cudaMemset(run_cursors,0,TX*TY*4));
   cp_run_clip_bin_validate_cursors<<<1,64>>>(fill); CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&err,errors,4,cudaMemcpyDeviceToHost));
   CHECK(err & CP_RUN_CLIP_BIN_ERR_CURSOR_MISMATCH);
   cp_run_clip_bin_live bad={0,N}; CK(cudaMemcpy(run_live,&bad,sizeof(bad),cudaMemcpyHostToDevice));
   uint32_t one=1; CK(cudaMemcpy(run_n,&one,4,cudaMemcpyHostToDevice));
   fill.capacity=total_refs; CK(cudaMemset(errors,0,4));
   cp_run_clip_bin_fill<<<1,1>>>(fill); CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&err,errors,4,cudaMemcpyDeviceToHost));
   CHECK(err & CP_RUN_CLIP_BIN_ERR_SEGMENT_BOUNDS);
   bad = {SLOTS, 0};
   CK(cudaMemcpy(run_live, &bad, sizeof(bad), cudaMemcpyHostToDevice));
   CK(cudaMemset(errors, 0, 4));
   cp_run_clip_bin_fill<<<1, 1>>>(fill); CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&err, errors, 4, cudaMemcpyDeviceToHost));
   CHECK(err & CP_RUN_CLIP_BIN_ERR_PRIM_BOUNDS);

   /* A prior checked-scan/count failure suppresses all fill writes. */
   err = CP_RUN_CLIP_BIN_ERR_REF_OVERFLOW;
   CK(cudaMemcpy(errors, &err, 4, cudaMemcpyHostToDevice));
   CK(cudaMemset(run_cursors, 0, TX * TY * 4));
   cp_run_clip_bin_fill<<<1, 128>>>(fill); CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(rc.data(), run_cursors, rc.size() * 4,
                 cudaMemcpyDeviceToHost));
   CHECK(std::all_of(rc.begin(), rc.end(), [](uint32_t v) { return v == 0; }));

   /* The uint64 pre-scan check catches totals that would wrap the existing
    * uint32 prefix scan, before offsets or visibility can be consumed. */
   std::vector<uint32_t> huge(TX * TY, 0);
   huge[0] = 0xffffffffu; huge[1] = 2;
   CK(cudaMemcpy(run_counts, huge.data(), huge.size() * 4,
                 cudaMemcpyHostToDevice));
   CK(cudaMemset(errors, 0, 4));
   check_counts.capacity = 0xffffffffu;
   cp_run_clip_bin_check_counts<<<1, 256>>>(check_counts);
   CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&checked, checked_total, sizeof(checked), cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(&err, errors, 4, cudaMemcpyDeviceToHost));
   CHECK(checked == 0x100000001ull);
   CHECK(err & CP_RUN_CLIP_BIN_ERR_REF_OVERFLOW);

   cp_run_clip_bin_device_segment malformed = hs[0];
   malformed.clip.num_slots = 0;
   CK(cudaMemcpy(ds, &malformed, sizeof(malformed), cudaMemcpyHostToDevice));
   CK(cudaMemset(run_over, 0, 4));
   cp_run_clip_bin_args malformed_args = run;
   malformed_args.num_segments = 1;
   cp_run_clip_setup_tile_count<<<dim3(1, 1), 64>>>(malformed_args);
   CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&err, run_over, 4, cudaMemcpyDeviceToHost));
   CHECK(err & CP_RUN_CLIP_BIN_ERR_PRIM_BOUNDS);
   CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));

   /* Capacity failure is explicit and happens before any renderer visibility
    * write. The future transaction must select exact fallback on this word. */
   CK(cudaMemset(run_counts,0,TX*TY*4)); CK(cudaMemset(run_n,0,4));
   CK(cudaMemset(run_over,0,4));
   run.live_capacity = 1;
   cp_run_clip_setup_tile_count<<<dim3(1,N),64>>>(run);
   CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(&an,run_n,4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(&ao,run_over,4,cudaMemcpyDeviceToHost));
   CHECK(ao == 1 && an > run.live_capacity);

   printf("run clip/bin/direct gate: %u segments, %u live, %u tile references; "
          "overflow fallback signalled -- PASS\n",N,bn,refs);
   return 0;
}
