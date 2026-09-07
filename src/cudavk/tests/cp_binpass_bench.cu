/*
 * Probe 5 of REDESIGN_PLAN_2026-09-05.md: the bin pass.
 *
 * The run-level renderer replaces cp_clip_rast_fused, stage 2 and stage 3 for
 * a covered run with one clip-and-bin launch: every post-clip triangle is
 * setup, then appended to the 16 px tile lists it covers, with per-tile
 * counters and a device prefix sum so every list is dense and its length is
 * known before the walk launches.
 *
 * This measures the binning half alone, at the sizes the M0 census recorded:
 * 354,000 post-clip triangles per heavy frame, 1.16 tile references each,
 * over an 80 x 45 grid of 16 px tiles (1280 x 720).
 *
 * GATE: <= 0.15 ms per heavy frame. Above that the plan stops.
 *
 * Two passes, not one, and deliberately: counting then scattering costs the
 * triangle read twice and leaves every list exactly sized. Sizing from a
 * worst-case bound is dead end 11, and a single-pass append needs either a
 * bound or a resize, which is the same entry.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if (e_) { \
   fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
           cudaGetErrorString(e_)); exit(1); } } while (0)

#define TILE 16
#define SCR_W 1280
#define SCR_H 720
#define TX ((SCR_W + TILE - 1) / TILE)      /* 80 */
#define TY ((SCR_H + TILE - 1) / TILE)      /* 45 */
#define NTILES (TX * TY)                    /* 3600 */

/* Only what binning reads: the screen bounding box and the row it came from. */
struct tri_box {
   int16_t x0, y0, x1, y1;
   uint32_t row;
};

__global__ void
bin_count(const struct tri_box * __restrict__ tris, uint32_t n,
          uint32_t * __restrict__ counts)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= n) return;
   struct tri_box t = tris[i];
   int tx0 = t.x0 / TILE, tx1 = t.x1 / TILE;
   int ty0 = t.y0 / TILE, ty1 = t.y1 / TILE;
   for (int ty = ty0; ty <= ty1; ty++)
      for (int tx = tx0; tx <= tx1; tx++)
         atomicAdd(&counts[ty * TX + tx], 1u);
}

/* One block, 1024 threads: 3,600 counters is a single scan. */
__global__ void
bin_scan(const uint32_t * __restrict__ counts, uint32_t * __restrict__ offs,
         uint32_t * __restrict__ total)
{
   __shared__ uint32_t s[1024];
   uint32_t acc = 0;
   for (uint32_t base = 0; base < NTILES; base += 1024) {
      uint32_t i = base + threadIdx.x;
      uint32_t v = (i < NTILES) ? counts[i] : 0u;
      s[threadIdx.x] = v;
      __syncthreads();
      /* Hillis-Steele over the block. */
      for (uint32_t d = 1; d < 1024; d <<= 1) {
         uint32_t x = (threadIdx.x >= d) ? s[threadIdx.x - d] : 0u;
         __syncthreads();
         s[threadIdx.x] += x;
         __syncthreads();
      }
      if (i < NTILES) offs[i] = acc + s[threadIdx.x] - v;
      uint32_t blocksum = s[1023];
      __syncthreads();
      acc += blocksum;
   }
   if (threadIdx.x == 0) *total = acc;
}

__global__ void
bin_scatter(const struct tri_box * __restrict__ tris, uint32_t n,
            const uint32_t * __restrict__ offs, uint32_t * __restrict__ cursor,
            uint64_t * __restrict__ refs)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= n) return;
   struct tri_box t = tris[i];
   int tx0 = t.x0 / TILE, tx1 = t.x1 / TILE;
   int ty0 = t.y0 / TILE, ty1 = t.y1 / TILE;
   for (int ty = ty0; ty <= ty1; ty++)
      for (int tx = tx0; tx <= tx1; tx++) {
         uint32_t tile = ty * TX + tx;
         uint32_t slot = atomicAdd(&cursor[tile], 1u);
         /* (setup index, draw row) -- the walk's reference word. */
         refs[offs[tile] + slot] = (uint64_t)i | ((uint64_t)t.row << 32);
      }
}

int main(int argc, char **argv)
{
   uint32_t n = (argc > 1) ? (uint32_t)atoi(argv[1]) : 354000u;
   int dev = 0;
   if (const char *e = getenv("CP_BENCH_DEV")) dev = atoi(e);
   CK(cudaSetDevice(dev));
   cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
   printf("device: %s  SMs %d  sm_%d%d\n", prop.name,
          prop.multiProcessorCount, prop.major, prop.minor);
   printf("binning %u triangles into %d tiles of %d px (%dx%d screen)\n",
          n, NTILES, TILE, SCR_W, SCR_H);

   /*
    * Triangle sizes are drawn so the mean tile coverage lands on M0's
    * measured 1.16 references per triangle: most triangles are far smaller
    * than a tile, a tail spans several.
    */
   std::mt19937 rng(1234u);
   std::lognormal_distribution<double> ln(1.1, 1.05);   /* px extent */
   std::uniform_int_distribution<int> px(0, SCR_W - 1), py(0, SCR_H - 1);
   std::vector<struct tri_box> h(n);
   size_t refs_expected = 0;
   for (uint32_t i = 0; i < n; i++) {
      int w = (int)ln(rng), hgt = (int)ln(rng);
      if (w < 1) w = 1; if (hgt < 1) hgt = 1;
      if (w > 400) w = 400; if (hgt > 400) hgt = 400;
      int x = px(rng), y = py(rng);
      h[i].x0 = (int16_t)x;
      h[i].y0 = (int16_t)y;
      h[i].x1 = (int16_t)std::min(x + w, SCR_W - 1);
      h[i].y1 = (int16_t)std::min(y + hgt, SCR_H - 1);
      h[i].row = i & 0xFFFFu;
      refs_expected += (size_t)(h[i].x1 / TILE - h[i].x0 / TILE + 1) *
                       (h[i].y1 / TILE - h[i].y0 / TILE + 1);
   }
   printf("generated: %.3f references per triangle (M0 measured 1.16), "
          "%zu references\n", (double)refs_expected / n, refs_expected);

   struct tri_box *d_tris; CK(cudaMalloc(&d_tris, n * sizeof(*d_tris)));
   CK(cudaMemcpy(d_tris, h.data(), n * sizeof(*d_tris),
                 cudaMemcpyHostToDevice));
   uint32_t *d_counts, *d_offs, *d_cursor, *d_total;
   CK(cudaMalloc(&d_counts, NTILES * sizeof(uint32_t)));
   CK(cudaMalloc(&d_offs, NTILES * sizeof(uint32_t)));
   CK(cudaMalloc(&d_cursor, NTILES * sizeof(uint32_t)));
   CK(cudaMalloc(&d_total, sizeof(uint32_t)));
   uint64_t *d_refs;
   CK(cudaMalloc(&d_refs, (refs_expected + 1024) * sizeof(uint64_t)));

   const int T = 256, B = (n + T - 1) / T;
   auto one = [&](void) {
      CK(cudaMemset(d_counts, 0, NTILES * sizeof(uint32_t)));
      CK(cudaMemset(d_cursor, 0, NTILES * sizeof(uint32_t)));
      bin_count<<<B, T>>>(d_tris, n, d_counts);
      bin_scan<<<1, 1024>>>(d_counts, d_offs, d_total);
      bin_scatter<<<B, T>>>(d_tris, n, d_offs, d_cursor, d_refs);
   };

   for (int i = 0; i < 5; i++) one();
   CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

   uint32_t total = 0;
   CK(cudaMemcpy(&total, d_total, sizeof total, cudaMemcpyDeviceToHost));
   printf("device counted %u references (host expected %zu) %s\n",
          total, refs_expected,
          total == refs_expected ? "-- MATCH" : "-- MISMATCH");

   const int reps = 50;
   cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
   CK(cudaEventRecord(a));
   for (int i = 0; i < reps; i++) one();
   CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
   float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
   double per = ms / reps;

   /* the three phases separately */
   double ph[3] = {0, 0, 0};
   const char *nm[3] = { "count", "scan", "scatter" };
   for (int p = 0; p < 3; p++) {
      CK(cudaEventRecord(a));
      for (int i = 0; i < reps; i++) {
         if (p == 0) { CK(cudaMemset(d_counts, 0, NTILES * sizeof(uint32_t)));
                       bin_count<<<B, T>>>(d_tris, n, d_counts); }
         else if (p == 1) bin_scan<<<1, 1024>>>(d_counts, d_offs, d_total);
         else { CK(cudaMemset(d_cursor, 0, NTILES * sizeof(uint32_t)));
                bin_scatter<<<B, T>>>(d_tris, n, d_offs, d_cursor, d_refs); }
      }
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float m2 = 0; CK(cudaEventElapsedTime(&m2, a, b));
      ph[p] = m2 / reps;
   }

   printf("\nBIN PASS, one heavy frame's triangles\n");
   for (int p = 0; p < 3; p++)
      printf("  %-8s %.4f ms\n", nm[p], ph[p]);
   printf("  %-8s %.4f ms  <-- plan gate: 0.15 ms/heavy frame  [%s]\n",
          "TOTAL", per, per <= 0.15 ? "PASS" : "FAIL");
   printf("\nfor scale, what it replaces on the RTX (union-exclusive, heavy):\n"
          "  clip_all 0.520 + stage2 0.187 + stage3 0.392 = 1.099 ms/frame\n");
   return 0;
}
