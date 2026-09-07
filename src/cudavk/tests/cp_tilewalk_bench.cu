/*
 * cp_tilewalk_bench.cu -- the decisive Renderer 2 M1 gate, standalone.
 *
 * THE QUESTION: how many cycles of block time does the M1 tile walk cost per
 * tile reference on this GPU?  RENDERER2_M1_DESIGN.md section 4.5 budgets
 * <= 28 cycles at the p99 tile (12,823 references); DEAD_ENDS.md entry 24's
 * prototype measured 3,409 and died of it.
 *
 * This is NOT driver code.  It links nothing from src/cudavk, changes no
 * kernel, and is not in meson.build.  It copies -- verbatim, by hand, from
 * src/cudavk/kernels/ at 229a7a3e0b8 -- exactly the four things the answer
 * depends on:
 *
 *   PACK_VISBUF / VISBUF_EMPTY      cp_rasterize.cu:68-72
 *   edge_function                   cp_rasterize.cu:99-106  (round-to-nearest
 *                                   intrinsics; the contraction matters)
 *   edge_is_top_left / edge_inside  cp_rasterize.cu:115-129
 *   cp_float_to_sortable_uint       cp_rast_types.h
 *   CP_WINDOW_DEPTH                 cp_rast_types.h
 *   struct cp_tri_setup (80 B)      cp_rast_types.h:1339-1348
 *
 * and the inner loop of cp_rast_small_or_defer() (cp_rasterize.cu:880-905),
 * which is the loop M1 section 4.1 says the walk is.
 *
 * BUILD
 *   nvcc -O3 -arch=sm_120 -lineinfo -o /tmp/cp_tilewalk_bench \
 *        src/cudavk/tests/cp_tilewalk_bench.cu
 *
 * RUN (take the GPU lock first: exec 9>/tmp/cudavk-gpu.lock; flock 9)
 *   /tmp/cp_tilewalk_bench
 *
 * Results and method: docs/cudavk/notes/RENDERER2_M1_WALKBENCH.md
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if (e_) { \
   printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
   exit(1); } } while (0)

/* ------------------------------------------------------------------ *
 * Verbatim from the driver.  Do not "improve" any of this.
 * ------------------------------------------------------------------ */

#define PACK_VISBUF(depth_uint, tri_id) \
   (((uint64_t)(depth_uint) << 32) | (uint64_t)(~(uint32_t)(tri_id)))
#define VISBUF_DEPTH(packed) ((uint32_t)((packed) >> 32))
#define VISBUF_TRIID(packed) (~(uint32_t)((packed) & 0xFFFFFFFF))
#define VISBUF_EMPTY 0xFFFFFFFFFFFFFFFFULL

#define CP_WINDOW_DEPTH(ndc_z, scale, translate) \
   ((translate) + (ndc_z) * (scale))

struct cp_tri_setup {
   float sx0, sy0, sx1, sy1, sx2, sy2;
   float ndc_z0, ndc_z1, ndc_z2;
   float inv_area;
   uint8_t e0_top_left, e1_top_left, e2_top_left;
   int32_t ix_min, iy_min, ix_max, iy_max;
   uint8_t is_point;
   float pt_x0, pt_y0, pt_x1, pt_y1;
};
static_assert(sizeof(struct cp_tri_setup) == 80, "cp_tri_setup ABI");

__host__ __device__ __forceinline__ uint32_t
cp_float_to_sortable_uint(float f)
{
   union { float f; uint32_t u; } bits;
   bits.f = f;
   uint32_t mask = -((int32_t)bits.u >> 31) | 0x80000000;
   return bits.u ^ mask;
}

__host__ __device__ __forceinline__ float
edge_function(float ax, float ay, float bx, float by, float px, float py)
{
#ifdef __CUDA_ARCH__
   return __fsub_rn(__fmul_rn(bx - px, ay - py),
                    __fmul_rn(by - py, ax - px));
#else
   volatile float p0 = (bx - px) * (ay - py);
   volatile float p1 = (by - py) * (ax - px);
   return p0 - p1;
#endif
}

__host__ __device__ __forceinline__ bool
edge_is_top_left(float ax, float ay, float bx, float by)
{
   float dx = bx - ax;
   float dy = by - ay;
   return dy > 0.0f || (dy == 0.0f && dx < 0.0f);
}

__host__ __device__ __forceinline__ bool
edge_inside(float e, bool top_left)
{
   return top_left ? (e >= 0.0f) : (e > 0.0f);
}

/* ------------------------------------------------------------------ *
 * The walk's own types.  M1 section 4.4: an 8 B reference record.
 *
 *   {uint32 setup_index; uint16 draw_row; uint16 near_key_hi}
 *
 * near_key_hi is the top 16 bits of the reference's conservative nearest
 * sortable depth key (min over ndc_z0/1/2 put through
 * float_to_sortable_uint(CP_WINDOW_DEPTH(...))), TRUNCATED, so
 * (near_key_hi << 16) <= true key.  Rejecting when that is greater than the
 * tile ceiling is therefore conservative: it can never drop a reference that
 * could have won a pixel.  Section 4.3 spends the same field on a 3-bit slab
 * id; see the results document for why 16 truncated bits are used instead
 * (the raw top 3 bits of a sortable float key put over half of a uniform
 * [0,1] depth range into one slab and the slab walk does nothing).
 * ------------------------------------------------------------------ */

struct tile_desc {
   uint32_t off;   /* first reference index */
   uint32_t n;     /* reference count */
};

#define REF_SETUP(r)   ((uint32_t)(r))
#define REF_ROW(r)     ((uint16_t)((r) >> 32))
#define REF_NEARHI(r)  ((uint32_t)((r) >> 48))

#define TILE_W 16
#define TILE_H 16
#define NTHREADS 256

/* ------------------------------------------------------------------ *
 * The walk.  One block per tile, 256 threads, ONE THREAD PER REFERENCE
 * (M1 section 4.1: this is the inversion of entry 24's one-block-per-tile,
 * one-reference-at-a-time, thread-0-does-setup-behind-a-barrier shape).
 *
 * Per-block state (M1 section 4.2): uint64_t vis[256] = 2,048 B of shared,
 * packed PACK_VISBUF exactly as the driver packs it, resolved by shared
 * atomicMin; plus tile_ceiling.
 * ------------------------------------------------------------------ */
template <bool EARLY_OUT>
__global__ __launch_bounds__(NTHREADS) void
cp_tile_walk(const struct cp_tri_setup * __restrict__ setups,
             const uint64_t * __restrict__ refs,
             const struct tile_desc * __restrict__ tiles,
             float depth_scale, float depth_translate,
             uint64_t * __restrict__ out_vis,
             long long * __restrict__ out_cycles,
             unsigned * __restrict__ out_processed,
             unsigned * __restrict__ out_covered,
             const uint32_t * __restrict__ chunk_tile = nullptr)
{
   /* unsigned long long, not uint64_t: that is what atomicMin takes and
    * what NVRTC's uint64_t already is when the driver compiles the real
    * kernel. Same 8 bytes, same packing. */
   __shared__ unsigned long long vis[NTHREADS];
   __shared__ uint32_t s_ceiling;
   __shared__ uint32_t s_warpmax[NTHREADS / 32];
   __shared__ uint32_t s_stop;
   __shared__ unsigned s_processed, s_covered;

   const struct tile_desc t = tiles[blockIdx.x];
   const uint32_t tid = threadIdx.x;

   /* Per-pixel state is one uint64 and there is no separate coverage mask
    * (M1 section 4.2).  Saturation is the ceiling: while any pixel is
    * VISBUF_EMPTY its depth word is 0xFFFFFFFF and no early-out can fire. */
   vis[tid] = VISBUF_EMPTY;
   if (tid == 0) { s_ceiling = 0xFFFFFFFFu; s_stop = 0;
                   s_processed = 0; s_covered = 0; }
   __syncthreads();

   long long t0 = clock64();

   unsigned my_processed = 0, my_covered = 0;

   /* The pixel this lane owns for the ceiling reduction only; coverage is
    * per reference, not per pixel. */
   for (uint32_t base = 0; base < t.n; base += NTHREADS) {
      if (EARLY_OUT) {
         /* tile_ceiling refresh: a 256-lane shared max reduction (8 steps),
          * once every 256 references (M1 section 4.3). */
         uint32_t d = (uint32_t)(vis[tid] >> 32);
         #pragma unroll
         for (int o = 16; o; o >>= 1)
            d = max(d, __shfl_xor_sync(0xFFFFFFFFu, d, o));
         if ((tid & 31u) == 0) s_warpmax[tid >> 5] = d;
         __syncthreads();
         if (tid == 0) {
            uint32_t m = 0;
            #pragma unroll
            for (int i = 0; i < NTHREADS / 32; i++) m = max(m, s_warpmax[i]);
            s_ceiling = m;
            /* Whole-walk termination.  References are ordered near to far by
             * the bin pass, so the head of this chunk bounds every reference
             * left in the list. */
            uint32_t head = REF_NEARHI(refs[t.off + base]) << 16;
            s_stop = (head > m);
         }
         __syncthreads();
         if (s_stop) break;
      }

      uint32_t i = base + tid;
      if (i < t.n) {
         uint64_t r = refs[t.off + i];                       /* 8 B */
         uint32_t sidx = REF_SETUP(r);
         bool go = true;
         if (EARLY_OUT) {
            /* Per-reference rejection, before any edge evaluation. */
            uint32_t near = REF_NEARHI(r) << 16;
            go = (near <= s_ceiling);
         }
         if (go) {
            my_processed++;
            /* 80 B of setup.  Read as five int4 so the whole record moves,
             * the way it would if the point path below were ever taken --
             * a field-by-field read lets the compiler drop is_point and the
             * four pt_ words and understate the traffic. */
            const int4 *sp = (const int4 *)(setups + sidx);
            int4 w0 = sp[0], w1 = sp[1], w2 = sp[2], w3 = sp[3], w4 = sp[4];
            struct cp_tri_setup s;
            int4 *dw = (int4 *)&s;
            dw[0] = w0; dw[1] = w1; dw[2] = w2; dw[3] = w3; dw[4] = w4;

            int px0 = max(s.ix_min, (int)0);
            int py0 = max(s.iy_min, (int)0);
            int px1 = min(s.ix_max, TILE_W - 1);
            int py1 = min(s.iy_max, TILE_H - 1);

            if (s.is_point) {
               /* Never taken by the generated data; present so the loads of
                * is_point and pt_* are real. */
               px0 = max(px0, (int)floorf(s.pt_x0));
               px1 = min(px1, (int)ceilf(s.pt_x1));
               py0 = max(py0, (int)floorf(s.pt_y0));
               py1 = min(py1, (int)ceilf(s.pt_y1));
            }

            /* cp_rasterize.cu:880-905, verbatim in structure: edges evaluated
             * from the vertices at every pixel, never stepped. num_samples 1. */
            for (int py = py0; py <= py1; py++) {
               for (int px = px0; px <= px1; px++) {
                  float cx = (float)px + 0.5f, cy = (float)py + 0.5f;

                  float e0 = edge_function(s.sx1, s.sy1, s.sx2, s.sy2, cx, cy);
                  float e1 = edge_function(s.sx2, s.sy2, s.sx0, s.sy0, cx, cy);
                  float e2 = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, cx, cy);

                  if (edge_inside(e0, s.e0_top_left) &&
                      edge_inside(e1, s.e1_top_left) &&
                      edge_inside(e2, s.e2_top_left)) {
                     float b0 = e0 * s.inv_area;
                     float b1 = e1 * s.inv_area;
                     float b2 = 1.0f - b0 - b1;
                     float ndc_z = b0 * s.ndc_z0 + b1 * s.ndc_z1 + b2 * s.ndc_z2;

                     uint32_t depth_uint = cp_float_to_sortable_uint(
                        CP_WINDOW_DEPTH(ndc_z, depth_scale, depth_translate));

                     /* The one write of the whole walk, and it is shared. */
                     atomicMin(&vis[py * TILE_W + px],
                               (unsigned long long)PACK_VISBUF(depth_uint, sidx));
                     my_covered++;
                  }
               }
            }
         }
      }
   }

   __syncthreads();
   long long t1 = clock64();

   /* Census, outside the timed region as far as the reduction goes. */
   #pragma unroll
   for (int o = 16; o; o >>= 1) {
      my_processed += __shfl_xor_sync(0xFFFFFFFFu, my_processed, o);
      my_covered   += __shfl_xor_sync(0xFFFFFFFFu, my_covered, o);
   }
   if ((tid & 31u) == 0) {
      atomicAdd(&s_processed, my_processed);
      atomicAdd(&s_covered, my_covered);
   }
   __syncthreads();
   if (tid == 0) {
      out_cycles[blockIdx.x] = t1 - t0;
      out_processed[blockIdx.x] = s_processed;
      out_covered[blockIdx.x] = s_covered;
   }
   /*
    * REDESIGN_PLAN_2026-09-05.md probe 4. Ordinary tiles own their output
    * slot and store it. A *split* tile has several blocks resolving disjoint
    * slices of one tile's list, so they merge with the same atomicMin the
    * walk uses internally -- which is order-independent, so splitting cannot
    * change the result. chunk_tile maps a block to the tile it belongs to.
    */
   if (chunk_tile) {
      atomicMin((unsigned long long *)
                &out_vis[(size_t)chunk_tile[blockIdx.x] * NTHREADS + tid],
                (unsigned long long)vis[tid]);
   } else {
      out_vis[(size_t)blockIdx.x * NTHREADS + tid] = vis[tid];
   }
}

/* ------------------------------------------------------------------ *
 * Calibration 1: an empty block.  Everything the walk kernel costs that is
 * not the walk -- launch, block scheduling, the descriptor load.
 * ------------------------------------------------------------------ */
__global__ __launch_bounds__(NTHREADS) void
cal_empty(long long *out_cycles)
{
   long long t0 = clock64();
   __syncthreads();
   long long t1 = clock64();
   if (threadIdx.x == 0) out_cycles[blockIdx.x] = t1 - t0;
}

/* ------------------------------------------------------------------ *
 * Calibration 2: the memory floor.  Same block shape, same reference list,
 * same 88 B read per reference (8 B record + 80 B setup), and nothing else --
 * no edges, no depth, no atomic.  If the walk is not comfortably above this,
 * the walk is memory bound and its instruction count is irrelevant; if the
 * walk is far above it, the arithmetic or the atomics own the time.
 * ------------------------------------------------------------------ */
__global__ __launch_bounds__(NTHREADS) void
cal_memfloor(const struct cp_tri_setup * __restrict__ setups,
             const uint64_t * __restrict__ refs,
             const struct tile_desc * __restrict__ tiles,
             long long * __restrict__ out_cycles,
             float * __restrict__ out_sink)
{
   const struct tile_desc t = tiles[blockIdx.x];
   const uint32_t tid = threadIdx.x;
   __syncthreads();
   long long t0 = clock64();
   /* Every one of the twenty words is consumed, by XOR rather than by a float
    * add: with only five of them used nvcc drops the vector loads and issues
    * ninety-two scalar LDGs instead, which reads a THIRD of the bytes over
    * four times as many transactions and makes the "floor" slower than the
    * walk it is supposed to bound.  Check the SASS after touching this:
    *   cuobjdump -sass ./cp_tilewalk_bench | grep -c LDG.E.128
    * must be 5 per reference in this kernel as well as in the walk. */
   uint32_t acc = 0;
   for (uint32_t base = 0; base < t.n; base += NTHREADS) {
      uint32_t i = base + tid;
      if (i < t.n) {
         uint64_t r = refs[t.off + i];
         const int4 *sp = (const int4 *)(setups + REF_SETUP(r));
         int4 w0 = sp[0], w1 = sp[1], w2 = sp[2], w3 = sp[3], w4 = sp[4];
         acc ^= (uint32_t)(w0.x ^ w0.y ^ w0.z ^ w0.w);
         acc ^= (uint32_t)(w1.x ^ w1.y ^ w1.z ^ w1.w);
         acc ^= (uint32_t)(w2.x ^ w2.y ^ w2.z ^ w2.w);
         acc ^= (uint32_t)(w3.x ^ w3.y ^ w3.z ^ w3.w);
         acc ^= (uint32_t)(w4.x ^ w4.y ^ w4.z ^ w4.w);
         acc ^= (uint32_t)(r >> 32);
      }
   }
   __syncthreads();
   long long t1 = clock64();
   if (tid == 0) out_cycles[blockIdx.x] = t1 - t0;
   out_sink[(size_t)blockIdx.x * NTHREADS + tid] = (float)acc;
}

/* Calibration 3: SM clock rate, so cycles and microseconds can be converted
 * without trusting nvidia-smi's idle reading. */
__global__ void cal_spin(long long cycles, long long *out)
{
   long long t0 = clock64();
   long long t1 = t0;
   while (t1 - t0 < cycles) t1 = clock64();
   if (threadIdx.x == 0) *out = t1 - t0;
}

/* ================================================================== *
 * Host: geometry and reference-list synthesis.
 * ================================================================== */

struct gen_stats {
   double mean_edge;
   double mean_area;
   double mean_bbox_px;
   size_t tris;
};

/*
 * The triangle pool.
 *
 * ASSUMPTIONS, stated so they can be attacked:
 *
 * 1. Coordinates are TILE LOCAL (0..16 in x and y), not absolute screen
 *    coordinates.  The walk never touches a pixel outside its own tile, so
 *    tile-local coordinates change no arithmetic, no memory traffic and no
 *    atomic contention -- but they let every tile draw from ONE shared pool,
 *    which is what makes the working set match a real scope's instead of
 *    being 3,600 private copies.  What it does not model is float magnitude
 *    in the edge functions, which costs the same cycles either way.
 *
 * 2. Triangle size: M0/entry 24 measure a mean edge of ~1.7 px
 *    (DEAD_ENDS.md:1271-1276).  Centres are drawn uniformly over
 *    [-1.5, 17.5]^2 -- i.e. slightly outside the tile -- so that a share of
 *    triangles straddle the tile edge and are clipped, which is where M0's
 *    1.16 references per triangle comes from.  Vertices are a rotated
 *    triangle of circumradius r with per-vertex jitter, r chosen so the
 *    measured mean edge lands at 1.7; the generator PRINTS the achieved mean
 *    edge, area and bbox so the assumption is checkable, not asserted.
 *
 * 3. Depth: ndc_z uniform in [0,1] per triangle, with +-0.002 of per-vertex
 *    variation.  depth_scale 1, depth_translate 0, so window depth is the
 *    ndc depth and the sortable key is monotone in it.  A UNIFORM depth
 *    distribution is the neutral assumption: it is neither the front-to-back
 *    submission order that would make the early-out look best nor the
 *    back-to-front order that would make it look worst.  The early-out arm's
 *    result is a function of this choice and is labelled as such; THE ARM
 *    THAT DECIDES THE GATE IS THE ONE WITH THE EARLY-OUT OFF, which does not
 *    depend on it at all.
 *
 * 4. The pool holds 176,000 triangles = 14.1 MB, which is one drawing
 *    scope's post-clip triangle count (M0: ~354k triangles/frame at ~2.02
 *    drawing scopes/frame, so ~175k per scope).  Every tile's list indexes
 *    into it uniformly at random.  This reproduces the real cache situation
 *    -- a scope's setup array is L2 resident and is re-read by many tiles --
 *    rather than a synthetic streaming one.
 */
static void
gen_pool(std::vector<struct cp_tri_setup> &pool, std::vector<uint32_t> &nearkey,
         size_t n_tris, unsigned seed, struct gen_stats *st)
{
   std::mt19937 rng(seed);
   std::uniform_real_distribution<float> uc(-1.5f, 17.5f);
   std::uniform_real_distribution<float> ur(0.0f, 1.0f);
   /* mean edge of an equilateral triangle of circumradius r is r*sqrt(3);
    * jitter widens it a little, so r is trimmed to land the measured mean
    * at 1.7 px. */
   const float R = 1.7f / 1.732050808f * 0.93f;

   pool.clear(); nearkey.clear();
   pool.reserve(n_tris); nearkey.reserve(n_tris);

   double sum_edge = 0, sum_area = 0, sum_bbox = 0;
   size_t nedge = 0;

   while (pool.size() < n_tris) {
      float cx = uc(rng), cy = uc(rng);
      float th = ur(rng) * 6.28318531f;
      float z = ur(rng);
      float vx[3], vy[3], vz[3];
      for (int k = 0; k < 3; k++) {
         float a = th + k * 2.09439510f;
         float rr = R * (0.6f + 0.8f * ur(rng));
         vx[k] = cx + rr * cosf(a);
         vy[k] = cy + rr * sinf(a);
         vz[k] = z + (ur(rng) - 0.5f) * 0.004f;
         if (vz[k] < 0.0f) vz[k] = 0.0f;
         if (vz[k] > 1.0f) vz[k] = 1.0f;
      }

      struct cp_tri_setup s;
      memset(&s, 0, sizeof s);
      s.sx0 = vx[0]; s.sy0 = vy[0];
      s.sx1 = vx[1]; s.sy1 = vy[1];
      s.sx2 = vx[2]; s.sy2 = vy[2];
      s.ndc_z0 = vz[0]; s.ndc_z1 = vz[1]; s.ndc_z2 = vz[2];

      float area = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, s.sx2, s.sy2);
      if (area == 0.0f) continue;
      if (area < 0.0f) {
         std::swap(s.sx1, s.sx2); std::swap(s.sy1, s.sy2);
         std::swap(s.ndc_z1, s.ndc_z2);
         area = -area;
      }
      s.inv_area = 1.0f / area;
      s.e0_top_left = edge_is_top_left(s.sx1, s.sy1, s.sx2, s.sy2);
      s.e1_top_left = edge_is_top_left(s.sx2, s.sy2, s.sx0, s.sy0);
      s.e2_top_left = edge_is_top_left(s.sx0, s.sy0, s.sx1, s.sy1);

      float min_x = fminf(fminf(s.sx0, s.sx1), s.sx2);
      float min_y = fminf(fminf(s.sy0, s.sy1), s.sy2);
      float max_x = fmaxf(fmaxf(s.sx0, s.sx1), s.sx2);
      float max_y = fmaxf(fmaxf(s.sy0, s.sy1), s.sy2);
      /* clip rect = the tile, exactly what the bin pass hands the walk */
      s.ix_min = std::max((int)floorf(min_x), 0);
      s.iy_min = std::max((int)floorf(min_y), 0);
      s.ix_max = std::min((int)ceilf(max_x), TILE_W - 1);
      s.iy_max = std::min((int)ceilf(max_y), TILE_H - 1);
      if (!(s.ix_min <= s.ix_max && s.iy_min <= s.iy_max))
         continue;   /* no reference in this tile */
      s.is_point = 0;

      float zmin = fminf(fminf(s.ndc_z0, s.ndc_z1), s.ndc_z2);
      nearkey.push_back(cp_float_to_sortable_uint(
         CP_WINDOW_DEPTH(zmin, 1.0f, 0.0f)));
      pool.push_back(s);

      for (int k = 0; k < 3; k++) {
         float ax = vx[k], ay = vy[k];
         float bx = vx[(k+1)%3], by = vy[(k+1)%3];
         sum_edge += sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by));
         nedge++;
      }
      sum_area += 0.5 * area;
      sum_bbox += (double)(s.ix_max - s.ix_min + 1) * (s.iy_max - s.iy_min + 1);
   }

   st->mean_edge = sum_edge / nedge;
   st->mean_area = sum_area / pool.size();
   st->mean_bbox_px = sum_bbox / pool.size();
   st->tris = pool.size();
}

/* Build a per-tile reference list.  order: 0 = near to far (what the bin
 * pass produces), 1 = arbitrary submission order. */
static void
gen_refs(std::vector<uint64_t> &refs, std::vector<struct tile_desc> &tiles,
         const std::vector<uint32_t> &nearkey, const std::vector<uint32_t> &lens,
         int order, unsigned seed)
{
   std::mt19937 rng(seed);
   std::uniform_int_distribution<uint32_t> pick(0, (uint32_t)nearkey.size() - 1);
   size_t total = 0;
   for (uint32_t l : lens) total += l;
   refs.clear(); refs.resize(total);
   tiles.clear(); tiles.resize(lens.size());

   size_t off = 0;
   std::vector<uint64_t> tmp;
   for (size_t b = 0; b < lens.size(); b++) {
      uint32_t n = lens[b];
      tmp.clear(); tmp.reserve(n);
      for (uint32_t i = 0; i < n; i++) {
         uint32_t idx = pick(rng);
         uint32_t key = nearkey[idx];
         uint64_t r = (uint64_t)idx
                    | ((uint64_t)(i & 0xFFFFu) << 32)      /* draw_row */
                    | ((uint64_t)(key >> 16) << 48);       /* near_key_hi */
         tmp.push_back(r);
      }
      if (order == 0)
         std::sort(tmp.begin(), tmp.end(),
                   [](uint64_t a, uint64_t b2){ return (a >> 48) < (b2 >> 48); });
      memcpy(&refs[off], tmp.data(), (size_t)n * 8);
      tiles[b].off = (uint32_t)off;
      tiles[b].n = n;
      off += n;
   }
}

/* ================================================================== */

struct dev_bufs {
   struct cp_tri_setup *setups = nullptr;
   uint64_t *refs = nullptr;
   struct tile_desc *tiles = nullptr;
   uint64_t *vis = nullptr;
   long long *cycles = nullptr;
   unsigned *processed = nullptr;
   unsigned *covered = nullptr;
   float *sink = nullptr;
};

static double g_clock_hz = 0.0;

struct arm_result {
   double cyc_per_ref_max;    /* longest block */
   double cyc_per_ref_mean;   /* reference-weighted over all blocks */
   long long max_block_cycles;
   uint32_t max_block_refs;
   double wall_ms;
   double processed_frac;
   double covered_per_ref;
   size_t total_refs;
};

template <bool EARLY>
static struct arm_result
run_walk(struct dev_bufs &d, int nblocks, const std::vector<uint32_t> &lens,
         int reps)
{
   struct arm_result r;
   size_t total = 0;
   for (int i = 0; i < nblocks; i++) total += lens[i];

   /* warm up */
   for (int i = 0; i < 3; i++)
      cp_tile_walk<EARLY><<<nblocks, NTHREADS>>>(d.setups, d.refs, d.tiles,
         1.0f, 0.0f, d.vis, d.cycles, d.processed, d.covered);
   CK(cudaDeviceSynchronize());
   CK(cudaGetLastError());

   cudaEvent_t a, b;
   CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
   CK(cudaEventRecord(a));
   for (int i = 0; i < reps; i++)
      cp_tile_walk<EARLY><<<nblocks, NTHREADS>>>(d.setups, d.refs, d.tiles,
         1.0f, 0.0f, d.vis, d.cycles, d.processed, d.covered);
   CK(cudaEventRecord(b));
   CK(cudaEventSynchronize(b));
   float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
   r.wall_ms = ms / reps;
   CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));

   std::vector<long long> cyc(nblocks);
   std::vector<unsigned> proc(nblocks), cov(nblocks);
   CK(cudaMemcpy(cyc.data(), d.cycles, nblocks * sizeof(long long),
                 cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(proc.data(), d.processed, nblocks * sizeof(unsigned),
                 cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(cov.data(), d.covered, nblocks * sizeof(unsigned),
                 cudaMemcpyDeviceToHost));

   double best = 0; int bi = 0;
   double sum_cyc = 0; size_t sum_proc = 0, sum_cov = 0;
   for (int i = 0; i < nblocks; i++) {
      if (lens[i] == 0) continue;
      double cpr = (double)cyc[i] / lens[i];
      if ((double)cyc[i] > best) { best = (double)cyc[i]; bi = i; }
      sum_cyc += (double)cyc[i];
      sum_proc += proc[i];
      sum_cov += cov[i];
   }
   r.max_block_cycles = cyc[bi];
   r.max_block_refs = lens[bi];
   r.cyc_per_ref_max = (double)cyc[bi] / lens[bi];
   r.cyc_per_ref_mean = sum_cyc / (double)total * nblocks / nblocks;
   /* reference-weighted mean cost: total block cycles / total references */
   r.cyc_per_ref_mean = sum_cyc / (double)total;
   r.processed_frac = (double)sum_proc / (double)total;
   r.covered_per_ref = (double)sum_cov / (double)total;
   r.total_refs = total;
   return r;
}

static void
print_arm(const char *name, const struct arm_result &r, int nblocks)
{
   printf("  %-34s refs %9zu  longest block %8lld cyc over %6u refs = "
          "%7.2f cyc/ref | weighted mean %6.2f cyc/ref | wall %8.4f ms | "
          "processed %5.1f%% | covered %.3f px/ref\n",
          name, r.total_refs, r.max_block_cycles, r.max_block_refs,
          r.cyc_per_ref_max, r.cyc_per_ref_mean, r.wall_ms,
          100.0 * r.processed_frac, r.covered_per_ref);
   (void)nblocks;
}

/* ------------------------------------------------------------------ *
 * Harness sanity: the numbers are worthless if the walk does not resolve
 * the tile correctly, and the early-out arm is only legal if it produces a
 * BIT-IDENTICAL visibility buffer to the arm that looks at every reference.
 * Both are checked here against a host reference walk.
 * ------------------------------------------------------------------ */
static bool
verify(struct dev_bufs &d, int nblocks,
       const std::vector<uint64_t> &h_refs,
       const std::vector<struct tile_desc> &h_tiles,
       const std::vector<struct cp_tri_setup> &pool, int check_tiles)
{
   std::vector<uint64_t> vis_off((size_t)nblocks * NTHREADS);
   std::vector<uint64_t> vis_on((size_t)nblocks * NTHREADS);

   cp_tile_walk<false><<<nblocks, NTHREADS>>>(d.setups, d.refs, d.tiles,
      1.0f, 0.0f, d.vis, d.cycles, d.processed, d.covered);
   CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(vis_off.data(), d.vis, vis_off.size() * 8,
                 cudaMemcpyDeviceToHost));
   cp_tile_walk<true><<<nblocks, NTHREADS>>>(d.setups, d.refs, d.tiles,
      1.0f, 0.0f, d.vis, d.cycles, d.processed, d.covered);
   CK(cudaDeviceSynchronize());
   CK(cudaMemcpy(vis_on.data(), d.vis, vis_on.size() * 8,
                 cudaMemcpyDeviceToHost));

   size_t diff = 0;
   for (size_t i = 0; i < vis_off.size(); i++)
      if (vis_off[i] != vis_on[i]) diff++;

   /* host reference over the first check_tiles tiles */
   size_t host_bad = 0, host_ulp = 0, host_covered = 0;
   long long host_maxulp = 0;
   for (int b = 0; b < check_tiles && b < nblocks; b++) {
      uint64_t ref[NTHREADS];
      for (int i = 0; i < NTHREADS; i++) ref[i] = VISBUF_EMPTY;
      const struct tile_desc t = h_tiles[b];
      for (uint32_t i = 0; i < t.n; i++) {
         uint64_t r = h_refs[t.off + i];
         const struct cp_tri_setup &s = pool[REF_SETUP(r)];
         for (int py = std::max(s.iy_min, 0);
              py <= std::min(s.iy_max, TILE_H - 1); py++)
         for (int px = std::max(s.ix_min, 0);
              px <= std::min(s.ix_max, TILE_W - 1); px++) {
            float cx = (float)px + 0.5f, cy = (float)py + 0.5f;
            float e0 = edge_function(s.sx1, s.sy1, s.sx2, s.sy2, cx, cy);
            float e1 = edge_function(s.sx2, s.sy2, s.sx0, s.sy0, cx, cy);
            float e2 = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, cx, cy);
            if (edge_inside(e0, s.e0_top_left) &&
                edge_inside(e1, s.e1_top_left) &&
                edge_inside(e2, s.e2_top_left)) {
               float b0 = e0 * s.inv_area, b1 = e1 * s.inv_area;
               float b2 = 1.0f - b0 - b1;
               float z = b0 * s.ndc_z0 + b1 * s.ndc_z1 + b2 * s.ndc_z2;
               uint64_t v = PACK_VISBUF(
                  cp_float_to_sortable_uint(CP_WINDOW_DEPTH(z, 1.0f, 0.0f)),
                  REF_SETUP(r));
               if (v < ref[py * TILE_W + px]) ref[py * TILE_W + px] = v;
               host_covered++;
            }
         }
      }
      /* The host reference is compiled by the host compiler, which contracts
       * the three-term barycentric depth interpolation differently from
       * nvcc -- the edge functions are pinned by the round-to-nearest
       * intrinsics but the interpolation is not.  A winner that agrees on
       * the primitive and whose key is within a few ULP is that, not a bug;
       * a DIFFERENT primitive is.  Both are counted, and the largest ULP
       * gap seen is printed so the tolerance cannot hide a drift.  Measured
       * on this data: 147 of ~1,000 resolved pixels sit 1-2 ULP apart and
       * every one of them names the same primitive. */
      for (int i = 0; i < NTHREADS; i++) {
         uint64_t g = vis_off[(size_t)b * NTHREADS + i];
         if (ref[i] == g) continue;
         int64_t dk = (int64_t)VISBUF_DEPTH(ref[i]) - (int64_t)VISBUF_DEPTH(g);
         if (VISBUF_TRIID(ref[i]) == VISBUF_TRIID(g)) {
            host_ulp++;
            if (llabs((long long)dk) > host_maxulp)
               host_maxulp = llabs((long long)dk);
         } else {
            host_bad++;
         }
      }
   }

   printf("  visbuf: early-out ON vs OFF -> %zu of %zu pixels differ%s\n",
          diff, vis_off.size(), diff ? "  *** EARLY-OUT IS NOT LEGAL ***" : "");
   printf("  visbuf: GPU vs host reference over %d tiles -> %zu wrong winners, "
          "%zu same-winner keys differing (max %lld ULP), %zu host coverage "
          "events%s\n", check_tiles, host_bad, host_ulp, host_maxulp,
          host_covered,
          (host_bad || host_maxulp > 4) ? "  *** WALK IS WRONG ***" : "  OK");
   return diff == 0 && host_bad == 0 && host_maxulp <= 4;
}


/* ------------------------------------------------------------------ *
 * REDESIGN_PLAN_2026-09-05.md, probe 4: hot-tile splitting.
 *
 * Dead end 24 and Renderer 2 both died on the same signature -- one block
 * holding one enormous tile set 94% of the grid's duration. The plan's answer
 * is to split any over-long tile list into fixed chunks AT BIN TIME, from
 * counts the bin pass already produced: no dynamic claiming, no persistent
 * kernel. This measures whether that works.
 *
 * The gate: p99 chunk <= 30 us, and the merge <= 10% of walk time.
 * ------------------------------------------------------------------ */
struct split_result {
   double p50_us, p99_us, max_us, wall_ms;
   int nchunks;
};

static struct split_result
run_split(struct dev_bufs &d, const std::vector<uint32_t> &tile_off,
          const std::vector<uint32_t> &tile_len, unsigned chunk, int reps,
          bool merge)
{
   std::vector<struct tile_desc> chunks;
   std::vector<uint32_t> owner;
   for (size_t t = 0; t < tile_len.size(); t++) {
      uint32_t n = tile_len[t], off = tile_off[t];
      if (!n) { chunks.push_back({off, 0}); owner.push_back((uint32_t)t); continue; }
      for (uint32_t s = 0; s < n; s += chunk) {
         struct tile_desc c;
         c.off = off + s;
         c.n = (n - s < chunk) ? (n - s) : chunk;
         chunks.push_back(c);
         owner.push_back((uint32_t)t);
      }
   }
   const int nb = (int)chunks.size();

   struct tile_desc *d_chunks = nullptr; uint32_t *d_owner = nullptr;
   CK(cudaMalloc(&d_chunks, nb * sizeof(*d_chunks)));
   CK(cudaMemcpy(d_chunks, chunks.data(), nb * sizeof(*d_chunks),
                 cudaMemcpyHostToDevice));
   CK(cudaMalloc(&d_owner, nb * sizeof(*d_owner)));
   CK(cudaMemcpy(d_owner, owner.data(), nb * sizeof(*d_owner),
                 cudaMemcpyHostToDevice));

   long long *d_cyc = nullptr;
   CK(cudaMalloc(&d_cyc, nb * sizeof(long long)));
   unsigned *d_proc = nullptr, *d_cov = nullptr;
   CK(cudaMalloc(&d_proc, nb * sizeof(unsigned)));
   CK(cudaMalloc(&d_cov, nb * sizeof(unsigned)));
   uint64_t *d_vis = nullptr;
   CK(cudaMalloc(&d_vis, (size_t)tile_len.size() * NTHREADS * sizeof(uint64_t)));

   for (int i = 0; i < 3; i++) {
      CK(cudaMemset(d_vis, 0xFF,
                    (size_t)tile_len.size() * NTHREADS * sizeof(uint64_t)));
      cp_tile_walk<true><<<nb, NTHREADS>>>(d.setups, d.refs, d_chunks,
         1.0f, 0.0f, d_vis, d_cyc, d_proc, d_cov, merge ? d_owner : nullptr);
   }
   CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

   cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
   CK(cudaEventRecord(a));
   for (int i = 0; i < reps; i++)
      cp_tile_walk<true><<<nb, NTHREADS>>>(d.setups, d.refs, d_chunks,
         1.0f, 0.0f, d_vis, d_cyc, d_proc, d_cov, merge ? d_owner : nullptr);
   CK(cudaEventRecord(b));
   CK(cudaEventSynchronize(b));
   float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));

   std::vector<long long> cyc(nb);
   CK(cudaMemcpy(cyc.data(), d_cyc, nb * sizeof(long long),
                 cudaMemcpyDeviceToHost));
   std::vector<double> us;
   us.reserve(nb);
   for (int i = 0; i < nb; i++)
      us.push_back((double)cyc[i] / g_clock_hz * 1e6);
   std::sort(us.begin(), us.end());

   struct split_result r;
   r.nchunks = nb;
   r.wall_ms = ms / reps;
   r.p50_us = us[us.size() / 2];
   r.p99_us = us[(size_t)(us.size() * 0.99)];
   r.max_us = us.back();

   CK(cudaFree(d_chunks)); CK(cudaFree(d_owner)); CK(cudaFree(d_cyc));
   CK(cudaFree(d_proc)); CK(cudaFree(d_cov)); CK(cudaFree(d_vis));
   CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
   return r;
}

int main(int argc, char **argv)
{
   /* argv[1] selects one arm, so that a profiler can be pointed at those
    * launches and nothing else: "mixed", "iso", or absent for all of them. */
   const char *mode = (argc > 1) ? argv[1] : "all";
   const bool want_homog = !strcmp(mode, "all");
   const bool want_iso   = !strcmp(mode, "all") || !strcmp(mode, "iso");
   const bool want_mixed = !strcmp(mode, "all") || !strcmp(mode, "mixed");
   int dev = 0;
   if (const char *e = getenv("CP_BENCH_DEV")) dev = atoi(e);   /* a test
      program, not the ICD: tests/ is on cp_no_getenv.py's allowlist */
   CK(cudaSetDevice(dev));
   cudaDeviceProp prop;
   CK(cudaGetDeviceProperties(&prop, dev));
   printf("device: %s  SMs %d  L2 %.1f MB  clockRate(nominal) %.3f GHz  "
          "sm_%d%d\n", prop.name, prop.multiProcessorCount,
          prop.l2CacheSize / 1048576.0, prop.clockRate / 1e6,
          prop.major, prop.minor);

   /* ---- SM clock, measured ---- */
   {
      long long *dout; CK(cudaMalloc(&dout, sizeof(long long)));
      cal_spin<<<1, 32>>>(50000000LL, dout);   /* warm the clocks up */
      CK(cudaDeviceSynchronize());
      cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      cal_spin<<<1, 32>>>(300000000LL, dout);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms, a, b));
      long long c; CK(cudaMemcpy(&c, dout, sizeof c, cudaMemcpyDeviceToHost));
      g_clock_hz = (double)c / (ms / 1000.0);
      printf("measured SM clock during a busy kernel: %.3f GHz "
             "(%lld cycles in %.4f ms)\n", g_clock_hz / 1e9, c, ms);
      CK(cudaFree(dout)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
   }

   /* ---- occupancy and footprint ---- */
   {
      cudaFuncAttributes fa;
      CK(cudaFuncGetAttributes(&fa, (const void *)cp_tile_walk<true>));
      int nb = 0;
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,
            (const void *)cp_tile_walk<true>, NTHREADS, 0));
      printf("cp_tile_walk<early=on> : %d regs, %zu B static shared, "
             "%d blocks/SM, %.1f%% theoretical occupancy\n",
             fa.numRegs, (size_t)fa.sharedSizeBytes, nb,
             100.0 * nb * NTHREADS / prop.maxThreadsPerMultiProcessor);
      CK(cudaFuncGetAttributes(&fa, (const void *)cp_tile_walk<false>));
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,
            (const void *)cp_tile_walk<false>, NTHREADS, 0));
      printf("cp_tile_walk<early=off>: %d regs, %zu B static shared, "
             "%d blocks/SM, %.1f%% theoretical occupancy\n",
             fa.numRegs, (size_t)fa.sharedSizeBytes, nb,
             100.0 * nb * NTHREADS / prop.maxThreadsPerMultiProcessor);
   }

   /* ---- the triangle pool ---- */
   const size_t POOL = 176000;   /* one drawing scope of post-clip triangles */
   std::vector<struct cp_tri_setup> pool;
   std::vector<uint32_t> nearkey;
   struct gen_stats gs;
   gen_pool(pool, nearkey, POOL, 12345u, &gs);
   printf("pool: %zu triangles, %.2f MB; measured mean edge %.3f px, "
          "mean area %.3f px^2, mean bbox %.2f px tested per reference\n",
          gs.tris, gs.tris * 80.0 / 1048576.0, gs.mean_edge, gs.mean_area,
          gs.mean_bbox_px);

   const int NBLOCKS = 3600;   /* 80 x 45 tiles of 16 px at 1280x720 */

   /* device buffers, sized for the largest arm */
   struct dev_bufs d;
   size_t max_refs = (size_t)NBLOCKS * 14814;
   CK(cudaMalloc(&d.setups, pool.size() * sizeof(struct cp_tri_setup)));
   CK(cudaMemcpy(d.setups, pool.data(), pool.size() * sizeof(struct cp_tri_setup),
                 cudaMemcpyHostToDevice));
   CK(cudaMalloc(&d.refs, max_refs * 8));
   CK(cudaMalloc(&d.tiles, NBLOCKS * sizeof(struct tile_desc)));
   CK(cudaMalloc(&d.vis, (size_t)NBLOCKS * NTHREADS * 8));
   CK(cudaMalloc(&d.cycles, NBLOCKS * sizeof(long long)));
   CK(cudaMalloc(&d.processed, NBLOCKS * sizeof(unsigned)));
   CK(cudaMalloc(&d.covered, NBLOCKS * sizeof(unsigned)));
   CK(cudaMalloc(&d.sink, (size_t)NBLOCKS * NTHREADS * 4));

   std::vector<uint64_t> h_refs;
   std::vector<struct tile_desc> h_tiles;

   auto upload = [&](const std::vector<uint32_t> &lens, int order) {
      gen_refs(h_refs, h_tiles, nearkey, lens, order, 999u);
      CK(cudaMemcpy(d.refs, h_refs.data(), h_refs.size() * 8,
                    cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d.tiles, h_tiles.data(),
                    h_tiles.size() * sizeof(struct tile_desc),
                    cudaMemcpyHostToDevice));
   };

   /* ---- calibration: the empty block ---- */
   {
      for (int i = 0; i < 3; i++) cal_empty<<<NBLOCKS, NTHREADS>>>(d.cycles);
      CK(cudaDeviceSynchronize());
      cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      for (int i = 0; i < 200; i++) cal_empty<<<NBLOCKS, NTHREADS>>>(d.cycles);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms, a, b));
      std::vector<long long> cyc(NBLOCKS);
      CK(cudaMemcpy(cyc.data(), d.cycles, NBLOCKS * sizeof(long long),
                    cudaMemcpyDeviceToHost));
      long long mx = 0; double sum = 0;
      for (long long c : cyc) { mx = std::max(mx, c); sum += c; }
      printf("\nCALIBRATION\n");
      printf("  empty 3600x256 kernel: wall %.4f ms/launch, block cycles "
             "mean %.1f max %lld\n", ms / 200, sum / NBLOCKS, mx);
   }

   /* the four M0 list lengths, homogeneous over the whole grid */
   struct { const char *tag; uint32_t n; } pts[] = {
      { "p50 hottest tile   237", 237 },
      { "p90              9,208", 9208 },
      { "p99             12,823", 12823 },
      { "max             14,814", 14814 },
   };

   /* ---- calibration: the memory floor at p99 ---- */
   {
      std::vector<uint32_t> lens(NBLOCKS, 12823);
      upload(lens, 0);
      for (int i = 0; i < 3; i++)
         cal_memfloor<<<NBLOCKS, NTHREADS>>>(d.setups, d.refs, d.tiles,
                                             d.cycles, d.sink);
      CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
      cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      for (int i = 0; i < 5; i++)
         cal_memfloor<<<NBLOCKS, NTHREADS>>>(d.setups, d.refs, d.tiles,
                                             d.cycles, d.sink);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms, a, b));
      std::vector<long long> cyc(NBLOCKS);
      CK(cudaMemcpy(cyc.data(), d.cycles, NBLOCKS * sizeof(long long),
                    cudaMemcpyDeviceToHost));
      long long mx = 0; double sum = 0;
      for (long long c : cyc) { mx = std::max(mx, c); sum += c; }
      printf("  88 B/ref load-only floor, 3600 x 12,823: wall %.4f ms/launch, "
             "longest block %lld cyc = %.2f cyc/ref, mean %.2f cyc/ref\n",
             ms / 5, mx, (double)mx / 12823, sum / NBLOCKS / 12823);
   }

   /* ---- homogeneous arms: every tile at the same M0 percentile ---- */
   printf("\nHOMOGENEOUS GRID (all 3,600 blocks at the same list length)\n");
   for (auto &p : pts) {
      if (!want_homog) break;
      std::vector<uint32_t> lens(NBLOCKS, p.n);
      int reps = p.n > 1000 ? 5 : 50;
      printf(" %s\n", p.tag);
      upload(lens, 0);
      print_arm("early-out OFF (near-to-far list)",
                run_walk<false>(d, NBLOCKS, lens, reps), NBLOCKS);
      print_arm("early-out ON  (near-to-far list)",
                run_walk<true>(d, NBLOCKS, lens, reps), NBLOCKS);
      upload(lens, 1);
      print_arm("early-out ON  (submission order)",
                run_walk<true>(d, NBLOCKS, lens, reps), NBLOCKS);
   }

   /* ---- the isolated tail block: one block, alone on the machine ----
    * Entry 24's whole lesson is that one block sets the kernel's duration,
    * so the tail block's cost with no contention is the number the gate is
    * really about. */
   printf("\nISOLATED TAIL BLOCK (1 block of 12,823 refs, machine otherwise idle)\n");
   if (want_iso) {
      std::vector<uint32_t> lens(NBLOCKS, 12823);
      upload(lens, 0);
      std::vector<uint32_t> one(1, 12823);
      print_arm("early-out OFF", run_walk<false>(d, 1, one, 50), 1);
      print_arm("early-out ON ", run_walk<true>(d, 1, one, 50), 1);
   }

   /* ---- the realistic mixed grid: one drawing scope ----
    * Lengths are lognormal(mu, sigma) with mu, sigma fitted so the median
    * non-empty tile lands in M0's peak log2 bucket (16-63) and the maximum
    * over the non-empty tiles lands on M0's refs_max p99 of 12,823.  1,744
    * of the 3,600 tiles are non-empty, which is M0's "mean refs per
    * non-empty scope-tile 117" solved against ~204k references per scope --
    * the rest exit immediately, which M1 section 4.2 prices at under 0.2 ns
    * per idle block (DEAD_ENDS.md:2090-2092 rule 6).
    */
   printf("\nMIXED GRID (one drawing scope: lognormal tile lists, "
          "1,744 of 3,600 non-empty)\n");
   if (want_mixed) {
      std::mt19937 rng(4242u);
      std::lognormal_distribution<double> ln(3.466, 1.879);
      std::vector<uint32_t> lens(NBLOCKS, 0u);
      std::vector<uint32_t> v;
      for (int i = 0; i < 1744; i++) {
         double x = ln(rng);
         if (x < 1) x = 1;
         if (x > 12823) x = 12823;
         v.push_back((uint32_t)x);
      }
      std::sort(v.begin(), v.end());
      v.back() = 12823;              /* pin the hot tile to M0's p99 */
      std::shuffle(v.begin(), v.end(), rng);
      for (size_t i = 0; i < v.size(); i++) lens[i] = v[i];
      std::shuffle(lens.begin(), lens.end(), rng);

      std::vector<uint32_t> srt = v;
      std::sort(srt.begin(), srt.end());
      size_t tot = 0; for (uint32_t x : v) tot += x;
      printf("  non-empty tiles %zu, total refs %zu, mean %.1f, "
             "p50 %u, p90 %u, p99 %u, max %u\n",
             v.size(), tot, (double)tot / v.size(),
             srt[srt.size()/2], srt[(size_t)(srt.size()*0.90)],
             srt[(size_t)(srt.size()*0.99)], srt.back());

      upload(lens, 0);
      printf("  HARNESS CHECK\n");
      verify(d, NBLOCKS, h_refs, h_tiles, pool, 64);
      struct arm_result off = run_walk<false>(d, NBLOCKS, lens, 20);
      struct arm_result on  = run_walk<true>(d, NBLOCKS, lens, 20);
      print_arm("early-out OFF", off, NBLOCKS);
      print_arm("early-out ON ", on, NBLOCKS);


      /* ---- REDESIGN_PLAN probe 4: split the hot tile ---- */
      printf("\nPROBE 4: HOT-TILE SPLITTING (plan gate: p99 chunk <= 30 us, "
             "merge <= 10%% of walk time)\n");
      {
         std::vector<uint32_t> offs(NBLOCKS);
         uint32_t acc = 0;
         for (int i = 0; i < NBLOCKS; i++) { offs[i] = acc; acc += lens[i]; }
         struct split_result base = run_split(d, offs, lens, 1u << 30, 20, false);
         printf("  unsplit (one block per tile)        chunks %6d  "
                "p50 %7.2f us  p99 %7.2f us  max %7.2f us  wall %.4f ms\n",
                base.nchunks, base.p50_us, base.p99_us, base.max_us,
                base.wall_ms);
         for (unsigned ch : { 512u, 1024u, 2048u }) {
            struct split_result nm = run_split(d, offs, lens, ch, 20, false);
            struct split_result mg = run_split(d, offs, lens, ch, 20, true);
            printf("  chunk %4u  no-merge chunks %6d  p50 %7.2f  p99 %7.2f  "
                   "max %7.2f  wall %.4f ms\n",
                   ch, nm.nchunks, nm.p50_us, nm.p99_us, nm.max_us, nm.wall_ms);
            printf("  chunk %4u  +atomicMin merge     "
                   "                                        wall %.4f ms  "
                   "merge cost %+.1f%%\n",
                   ch, mg.wall_ms,
                   100.0 * (mg.wall_ms - nm.wall_ms) / nm.wall_ms);
         }
      }

      printf("\nPER-SCOPE ARITHMETIC (M1 section 4.6: budget 0.15 ms/scope, "
             "21%% of the 0.724 ms/scope exclusive it replaces)\n");
      printf("  walk kernel, one scope, early-out OFF : %.4f ms/scope  "
             "(%.1f%% of 0.724)\n", off.wall_ms, 100.0 * off.wall_ms / 0.724);
      printf("  walk kernel, one scope, early-out ON  : %.4f ms/scope  "
             "(%.1f%% of 0.724)\n", on.wall_ms, 100.0 * on.wall_ms / 0.724);
      printf("  at 2.02 drawing scopes/frame          : %.4f / %.4f ms/frame\n",
             off.wall_ms * 2.02, on.wall_ms * 2.02);
   }

   printf("\nGATE (M1 section 4.5): <= 28 cycles of block time per reference "
          "at the p99 tile (12,823 refs).\n"
          "Entry 24's prototype: 3,409 cycles/reference.\n");
   return 0;
}
