/*
 * Standalone REDESIGN P1 benchmark for the real run clip/setup/bin kernels.
 *
 * Build from the Mesa root:
 *   nvcc -O3 -arch=native -std=c++17 -Isrc/cudavk \
 *     src/cudavk/tests/cp_run_clip_bin_bench.cu -o /tmp/cp_run_clip_bin_bench
 *
 * This intentionally includes the CUDA source.  It measures exactly the
 * entry points loaded by cp_kernels.c without changing or linking the ICD.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <numeric>

#pragma nv_diag_suppress 177
#include "../kernels/cp_rasterize.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
   fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, \
           cudaGetErrorString(e_)); exit(1); } } while (0)
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "check failed: %s (%s:%d)\n", \
   #x, __FILE__, __LINE__); exit(1); } } while (0)

static constexpr uint32_t W = 1280, H = 720, TILE = 32;
static constexpr uint32_t TX = 40, TY = 23, NT = TX * TY;

/* NT is 920 at P1.  This is the same uint32 exclusive-prefix result that the
 * renderer needs between checked count and fill. */
__global__ static void
scan_920(const uint32_t *counts, uint32_t *offsets)
{
   __shared__ uint32_t v[1024];
   uint32_t x = threadIdx.x < NT ? counts[threadIdx.x] : 0;
   v[threadIdx.x] = x;
   __syncthreads();
   for (uint32_t d = 1; d < 1024; d <<= 1) {
      uint32_t add = threadIdx.x >= d ? v[threadIdx.x - d] : 0;
      __syncthreads();
      v[threadIdx.x] += add;
      __syncthreads();
   }
   if (threadIdx.x < NT)
      offsets[threadIdx.x] = v[threadIdx.x] - x;
}

struct allocation {
   float4 *in = nullptr, *out = nullptr;
   uint32_t *out_count = nullptr;
};

struct bench {
   uint32_t n, nseg, total_slots;
   std::vector<allocation> alloc;
   std::vector<cp_run_clip_bin_device_segment> hseg;
   cp_run_clip_bin_device_segment *segments;
   cp_run_clip_bin_live *live;
   uint32_t *live_count, *errors, *counts, *offsets, *cursors;
   uint64_t *total_refs;
   uint64_t *prim_refs_all;
   cp_opaque_tile_ref *refs = nullptr;
   uint32_t ref_capacity = 0;
   cp_run_clip_bin_args count_args;
   cp_run_clip_bin_check_counts_args check_args;
   cp_run_clip_bin_fill_args fill_args;
   uint32_t grid_x;

   bench(uint32_t count, uint32_t segs, bool clipping) : n(count), nseg(segs) {
      CHECK(nseg >= 1 && nseg <= CP_RUN_CLIP_BIN_MAX_SEGS);
      CHECK(TILE == CP_RUN_CLIP_BIN_TILE_SIZE && NT <= 1024);
      total_slots = n * CP_CLIP_MAX_OUT;
      alloc.resize(nseg); hseg.resize(nseg);
      uint32_t first = 0, max_n = 0;
      for (uint32_t s = 0; s < nseg; s++) {
         uint32_t sn = n / nseg + (s < n % nseg);
         max_n = std::max(max_n, sn);
         std::vector<float4> pos((size_t)sn * 3);
         for (uint32_t j = 0; j < sn; j++) {
            uint32_t i = first + j;
            if (clipping) {
               /* Every primitive crosses x <= w and becomes two stable
                * triangles.  Keep the surviving sliver in the edge tile. */
               float py = 8.0f + (float)((i * 37u) % 704u);
               auto ndcx = [](float x) { return x / 640.0f - 1.0f; };
               auto ndcy = [](float y) { return y / 360.0f - 1.0f; };
               pos[3*j+0] = make_float4(ndcx(1260), ndcy(py-2), .5f, 1);
               pos[3*j+1] = make_float4(1.08f,      ndcy(py),   .5f, 1);
               pos[3*j+2] = make_float4(ndcx(1260), ndcy(py+2), .5f, 1);
            } else {
               /* Exactly 8% straddle one x tile boundary.  The rest have a
                * small bounding box wholly inside one tile: target 1.08 refs
                * per input triangle with overwhelmingly no clip work. */
               bool two = i < (uint64_t)n * 8u / 100u;
               uint32_t ty = (i * 17u) % TY;
               float cy = std::min(ty * 32.0f + 16.0f, 712.0f);
               float cx;
               if (two)
                  cx = 32.0f * (1u + (i * 13u) % 39u);
               else
                  cx = 16.0f + 32.0f * ((i * 13u) % TX);
               auto ndcx = [](float x) { return x / 640.0f - 1.0f; };
               auto ndcy = [](float y) { return y / 360.0f - 1.0f; };
               pos[3*j+0] = make_float4(ndcx(cx-1.25f), ndcy(cy-1), .5f, 1);
               pos[3*j+1] = make_float4(ndcx(cx+1.25f), ndcy(cy),   .5f, 1);
               pos[3*j+2] = make_float4(ndcx(cx-1.25f), ndcy(cy+1), .5f, 1);
            }
         }
         allocation &a = alloc[s];
         CK(cudaMalloc(&a.in, pos.size() * sizeof(float4)));
         CK(cudaMalloc(&a.out, (size_t)sn * CP_CLIP_MAX_OUT * 3 * sizeof(float4)));
         CK(cudaMalloc(&a.out_count, sizeof(uint32_t)));
         CK(cudaMemcpy(a.in, pos.data(), pos.size() * sizeof(float4), cudaMemcpyHostToDevice));

         cp_run_clip_bin_device_segment &d = hseg[s];
         d = {};
         d.clip.vs_out = (uint64_t)a.in;
         d.clip.out = (uint64_t)a.out;
         d.clip.out_count = (uint64_t)a.out_count;
         d.clip.num_triangles = sn;
         d.clip.num_slots = 1;
         d.clip.max_triangles = sn * CP_CLIP_MAX_OUT;
         d.clip.stable = 1;
         d.rast.width = W; d.rast.height = H; d.rast.num_samples = 1;
         d.rast.vp_scale_x = W * .5f; d.rast.vp_scale_y = H * .5f;
         d.rast.vp_trans_x = W * .5f; d.rast.vp_trans_y = H * .5f;
         d.rast.clip_x0 = 0; d.rast.clip_y0 = 0;
         d.rast.clip_x1 = W - 1; d.rast.clip_y1 = H - 1;
         d.prim_base = first * CP_CLIP_MAX_OUT;
         first += sn;
      }
      CHECK(first == n);
      grid_x = (max_n + 127) / 128;
      CK(cudaMalloc(&prim_refs_all, (size_t)total_slots * sizeof(uint64_t)));
      for (uint32_t s = 0; s < nseg; s++)
         hseg[s].clip.prim_refs =
            (uint64_t)(prim_refs_all + hseg[s].prim_base);
      CK(cudaMalloc(&segments, nseg * sizeof(*segments)));
      CK(cudaMemcpy(segments, hseg.data(), nseg * sizeof(*segments), cudaMemcpyHostToDevice));
      CK(cudaMalloc(&live, (size_t)total_slots * sizeof(*live)));
      CK(cudaMalloc(&live_count, 4)); CK(cudaMalloc(&errors, 4));
      CK(cudaMalloc(&counts, NT * 4)); CK(cudaMalloc(&offsets, NT * 4));
      CK(cudaMalloc(&cursors, NT * 4)); CK(cudaMalloc(&total_refs, 8));
      for (auto &a : alloc) CK(cudaMemset(a.out_count, 0, 4));
      count_args = {(uint64_t)segments, (uint64_t)live, (uint64_t)live_count,
         (uint64_t)errors, (uint64_t)counts, nseg, total_slots, total_slots, TX, TY};
   }

   ~bench() {
      if (refs) cudaFree(refs);
      cudaFree(total_refs); cudaFree(cursors); cudaFree(offsets);
      cudaFree(counts); cudaFree(errors); cudaFree(live_count);
      cudaFree(live); cudaFree(segments); cudaFree(prim_refs_all);
      for (auto &a : alloc) {
         cudaFree(a.out_count);
         cudaFree(a.out); cudaFree(a.in);
      }
   }

   void clear() {
      /* Renderer fallback scans all stable slots, so holes must start null. */
      CK(cudaMemsetAsync(prim_refs_all, 0,
                         (size_t)total_slots * sizeof(uint64_t)));
      CK(cudaMemsetAsync(live_count, 0, 4));
      CK(cudaMemsetAsync(errors, 0, 4));
      CK(cudaMemsetAsync(counts, 0, NT * 4));
      CK(cudaMemsetAsync(cursors, 0, NT * 4));
   }
   void count() { cp_run_clip_setup_tile_count<<<dim3(grid_x,nseg),128>>>(count_args); }
   void check() { cp_run_clip_bin_check_counts<<<1,256>>>(check_args); }
   void scan() { scan_920<<<1,1024>>>(counts, offsets); }
   void fill() { cp_run_clip_bin_fill<<<std::max(1u,(total_slots+127)/128),128>>>(fill_args); }
   void validate() { cp_run_clip_bin_validate_cursors<<<(NT+127)/128,128>>>(fill_args); }

   uint32_t prepare_exact() {
      clear(); count(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
      std::vector<uint32_t> hc(NT);
      uint32_t nl, err;
      CK(cudaMemcpy(hc.data(), counts, NT*4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&nl, live_count, 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&err, errors, 4, cudaMemcpyDeviceToHost));
      CHECK(err == 0 && nl <= total_slots);
      uint64_t sum = std::accumulate(hc.begin(), hc.end(), uint64_t(0));
      CHECK(sum && sum <= 0xffffffffu);
      ref_capacity = (uint32_t)sum;
      /* No slack: both stable clipping storage and dense reference storage
       * have their exact ABI capacities. */
      CK(cudaMalloc(&refs, (size_t)ref_capacity * sizeof(*refs)));
      check_args = {(uint64_t)counts, (uint64_t)total_refs, (uint64_t)errors,
                    NT, ref_capacity};
      fill_args = {(uint64_t)segments, (uint64_t)live, (uint64_t)live_count,
         (uint64_t)counts, (uint64_t)offsets, (uint64_t)cursors,
         (uint64_t)refs, (uint64_t)errors, nseg, total_slots, total_slots,
         TX, TY, ref_capacity};
      return nl;
   }

   void transaction() { clear(); count(); check(); scan(); fill(); validate(); }

   void verify(uint32_t expected_live) {
      uint32_t nl, err; uint64_t total;
      std::vector<uint32_t> hc(NT), ho(NT), hcur(NT);
      CK(cudaMemcpy(&nl, live_count, 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&err, errors, 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&total, total_refs, 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hc.data(), counts, NT*4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ho.data(), offsets, NT*4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hcur.data(), cursors, NT*4, cudaMemcpyDeviceToHost));
      CHECK(err == 0 && nl == expected_live && total == ref_capacity);
      uint64_t running = 0;
      for (uint32_t i=0; i<NT; i++) {
         CHECK(ho[i] == running && hcur[i] == hc[i]);
         running += hc[i];
      }
      CHECK(running == ref_capacity);
      std::vector<cp_opaque_tile_ref> hr(ref_capacity);
      CK(cudaMemcpy(hr.data(), refs, hr.size()*sizeof(hr[0]), cudaMemcpyDeviceToHost));
      for (auto &r : hr)
         CHECK(r.segment < nseg && r.global_prim < total_slots && r.flags == 0);
   }
};

static float elapsed(cudaEvent_t a, cudaEvent_t b) {
   float v; CK(cudaEventElapsedTime(&v,a,b)); return v;
}

static void run_case(const char *name, uint32_t n, uint32_t segs, int reps,
                     bool clipping)
{
   bench b(n,segs,clipping);
   uint32_t expected_live = b.prepare_exact();
   printf("\n%s: %u input, %u descriptors, %u live, %u refs (%.3f/input)\n",
          name,n,segs,expected_live,b.ref_capacity,(double)b.ref_capacity/n);
   double input_mib=(double)n*3*sizeof(float4)/(1<<20);
   double output_mib=(double)n*CP_CLIP_MAX_OUT*3*sizeof(float4)/(1<<20);
   double primref_mib=(double)b.total_slots*sizeof(uint64_t)/(1<<20);
   double live_mib=(double)b.total_slots*sizeof(cp_run_clip_bin_live)/(1<<20);
   double denseref_mib=(double)b.ref_capacity*sizeof(cp_opaque_tile_ref)/(1<<20);
   printf("  stable slots %u; input %.1f MiB, output %.1f MiB, prim-ref %.1f MiB, live %.1f MiB, dense-ref %.1f MiB\n",
          b.total_slots,input_mib,output_mib,primref_mib,live_mib,denseref_mib);
   printf("  principal device storage %.1f MiB (plus descriptors and 920-entry scratch)\n",
          input_mib+output_mib+primref_mib+live_mib+denseref_mib);
   for (int i=0;i<8;i++) b.transaction();
   CK(cudaDeviceSynchronize()); CK(cudaGetLastError()); b.verify(expected_live);

   enum {CLEAR,COUNT,CHECK,SCAN,FILL,VALIDATE,PHASES};
   const char *names[PHASES] = {"clear","clip/setup/count","check-counts","prefix-scan","fill","cursor-validate"};
   std::vector<cudaEvent_t> ev((size_t)reps*(PHASES+1));
   for(auto &e:ev) CK(cudaEventCreate(&e));
   for(int r=0;r<reps;r++) {
      size_t q=(size_t)r*(PHASES+1); CK(cudaEventRecord(ev[q]));
      b.clear(); CK(cudaEventRecord(ev[q+1]));
      b.count(); CK(cudaEventRecord(ev[q+2]));
      b.check(); CK(cudaEventRecord(ev[q+3]));
      b.scan(); CK(cudaEventRecord(ev[q+4]));
      b.fill(); CK(cudaEventRecord(ev[q+5]));
      b.validate(); CK(cudaEventRecord(ev[q+6]));
   }
   CK(cudaEventSynchronize(ev.back()));
   double phase[PHASES]={};
   for(int r=0;r<reps;r++) for(int p=0;p<PHASES;p++)
      phase[p] += elapsed(ev[(size_t)r*(PHASES+1)+p],ev[(size_t)r*(PHASES+1)+p+1])/reps;
   for(auto e:ev) CK(cudaEventDestroy(e));

   cudaEvent_t a,z; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&z));
   CK(cudaEventRecord(a)); for(int r=0;r<reps;r++) b.transaction();
   CK(cudaEventRecord(z)); CK(cudaEventSynchronize(z));
   double total=elapsed(a,z)/reps;
   for(int p=0;p<PHASES;p++) printf("  %-18s %8.4f ms\n",names[p],phase[p]);
   printf("  %-18s %8.4f ms  (%d reps after warmup)\n","TOTAL",total,reps);
   b.verify(expected_live);
}

int main(int argc,char **argv)
{
   uint32_t n=argc>1?(uint32_t)strtoul(argv[1],0,0):354000;
   uint32_t segs=argc>2?(uint32_t)strtoul(argv[2],0,0):64;
   int reps=argc>3?atoi(argv[3]):50;
   int dev=0; if(const char *x=getenv("CP_BENCH_DEV")) dev=atoi(x);
   CK(cudaSetDevice(dev)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
   printf("device: %s, %d SMs, sm_%d%d, %.0f MiB global\n",p.name,
          p.multiProcessorCount,p.major,p.minor,(double)p.totalGlobalMem/(1<<20));
   printf("P1: %ux%u, %u %upx tiles; exact stable allocations; CUDA events\n",
          W,H,NT,TILE);
   run_case("representative mostly-unclipped",n,segs,reps,false);
   run_case("clipping-heavy (all cross x<=w)",n,segs,reps,true);
   puts("\nvalidation: checked totals, exclusive offsets, count==cursor for every tile, and every dense ref -- PASS");
   return 0;
}
