/*
 * cudapipe 3-stage adaptive triangle rasterizer — visibility buffer approach.
 *
 * Stage 1: 1 thread per triangle. Small triangles (bbox <= CP_SMALL_THRESHOLD)
 *           are rasterized in place with incremental edge stepping. Larger
 *           triangles are pushed to the nontrivial queue.
 *
 * Stage 2: 1 warp per triangle from the nontrivial queue. Medium triangles
 *           (bbox <= CP_MEDIUM_THRESHOLD) are rasterized cooperatively. Huge
 *           triangles are decomposed into tiles and pushed to the huge queue.
 *
 * Stage 3: 1 block (64 threads) per tile-triangle pair from the huge queue.
 *           Uses trivial accept/reject at tile corners to skip edge tests for
 *           interior tiles.
 *
 * Resolve: 1 thread per pixel. Read winning triID from visibility buffer,
 *          recompute barycentrics, output color.
 */
#include "cp_rast_types.h"

#define PACK_VISBUF(depth_uint, tri_id) \
   (((uint64_t)(depth_uint) << 32) | (uint64_t)(~(uint32_t)(tri_id)))
#define VISBUF_DEPTH(packed) ((uint32_t)((packed) >> 32))
#define VISBUF_TRIID(packed) (~(uint32_t)((packed) & 0xFFFFFFFF))
#define VISBUF_EMPTY 0xFFFFFFFFFFFFFFFFULL

static __device__ __forceinline__ float
edge_function(float ax, float ay, float bx, float by, float cx, float cy)
{
   return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

static __device__ __forceinline__ uint32_t
float_to_sortable_uint(float f)
{
   uint32_t u = __float_as_uint(f);
   uint32_t mask = -((int32_t)u >> 31) | 0x80000000;
   return u ^ mask;
}

struct tri_setup {
   float sx0, sy0, sx1, sy1, sx2, sy2;
   float ndc_z0, ndc_z1, ndc_z2;
   float inv_area;
   float e0_dx, e0_dy, e1_dx, e1_dy, e2_dx, e2_dy;
   int ix_min, iy_min, ix_max, iy_max;
};

static __device__ __forceinline__ bool
setup_triangle(struct cp_rasterize_args *args, uint32_t tri_id,
               struct tri_setup *s)
{
   float4 *positions = (float4 *)(uintptr_t)args->positions;
   uint32_t pos_stride = args->num_varyings + 1;

   float4 v0 = positions[(tri_id * 3 + 0) * pos_stride];
   float4 v1 = positions[(tri_id * 3 + 1) * pos_stride];
   float4 v2 = positions[(tri_id * 3 + 2) * pos_stride];

   float inv_w0 = 1.0f / v0.w;
   float inv_w1 = 1.0f / v1.w;
   float inv_w2 = 1.0f / v2.w;

   s->ndc_z0 = v0.z * inv_w0;
   s->ndc_z1 = v1.z * inv_w1;
   s->ndc_z2 = v2.z * inv_w2;

   s->sx0 = v0.x * inv_w0 * args->vp_scale_x + args->vp_trans_x;
   s->sy0 = v0.y * inv_w0 * args->vp_scale_y + args->vp_trans_y;
   s->sx1 = v1.x * inv_w1 * args->vp_scale_x + args->vp_trans_x;
   s->sy1 = v1.y * inv_w1 * args->vp_scale_y + args->vp_trans_y;
   s->sx2 = v2.x * inv_w2 * args->vp_scale_x + args->vp_trans_x;
   s->sy2 = v2.y * inv_w2 * args->vp_scale_y + args->vp_trans_y;

   float area = edge_function(s->sx0, s->sy0, s->sx1, s->sy1, s->sx2, s->sy2);
   if (args->cull_mode == 1 && area > 0) return false;
   if (args->cull_mode == 2 && area < 0) return false;
   if (area == 0.0f) return false;

   if (area < 0.0f) {
      float tmp;
      tmp = s->sx1; s->sx1 = s->sx2; s->sx2 = tmp;
      tmp = s->sy1; s->sy1 = s->sy2; s->sy2 = tmp;
      tmp = s->ndc_z1; s->ndc_z1 = s->ndc_z2; s->ndc_z2 = tmp;
      area = -area;
   }
   s->inv_area = 1.0f / area;

   /* Edge function gradients for incremental stepping */
   s->e0_dx = s->sy2 - s->sy1;
   s->e0_dy = s->sx1 - s->sx2;
   s->e1_dx = s->sy0 - s->sy2;
   s->e1_dy = s->sx2 - s->sx0;
   s->e2_dx = s->sy1 - s->sy0;
   s->e2_dy = s->sx0 - s->sx1;

   float min_x = fminf(fminf(s->sx0, s->sx1), s->sx2);
   float min_y = fminf(fminf(s->sy0, s->sy1), s->sy2);
   float max_x = fmaxf(fmaxf(s->sx0, s->sx1), s->sx2);
   float max_y = fmaxf(fmaxf(s->sy0, s->sy1), s->sy2);

   s->ix_min = max((int)floorf(min_x), 0);
   s->iy_min = max((int)floorf(min_y), 0);
   s->ix_max = min((int)ceilf(max_x), (int)args->width - 1);
   s->iy_max = min((int)ceilf(max_y), (int)args->height - 1);

   return true;
}

static __device__ __forceinline__ void
rasterize_pixel(struct cp_rasterize_args *args, struct tri_setup *s,
                uint32_t tri_id, int px, int py)
{
   float cx = (float)px + 0.5f;
   float cy = (float)py + 0.5f;

   float w0 = edge_function(s->sx1, s->sy1, s->sx2, s->sy2, cx, cy) * s->inv_area;
   float w1 = edge_function(s->sx2, s->sy2, s->sx0, s->sy0, cx, cy) * s->inv_area;
   float w2 = 1.0f - w0 - w1;

   if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
      return;

   float depth = w0 * s->ndc_z0 + w1 * s->ndc_z1 + w2 * s->ndc_z2;
   depth = depth * 0.5f + 0.5f;

   uint32_t depth_uint = float_to_sortable_uint(depth);

   if (args->depth_test && args->depthbuf) {
      uint32_t prev =
         ((const uint32_t *)(uintptr_t)args->depthbuf)[py * args->width + px];
      bool pass;
      switch (args->depth_func) {
      case CP_FUNC_NEVER:     pass = false; break;
      case CP_FUNC_LESS:      pass = depth_uint <  prev; break;
      case CP_FUNC_EQUAL:     pass = depth_uint == prev; break;
      case CP_FUNC_LEQUAL:    pass = depth_uint <= prev; break;
      case CP_FUNC_GREATER:   pass = depth_uint >  prev; break;
      case CP_FUNC_NOTEQUAL:  pass = depth_uint != prev; break;
      case CP_FUNC_GEQUAL:    pass = depth_uint >= prev; break;
      default:                pass = true; break;
      }
      if (!pass)
         return;
   }

   uint32_t key = args->depth_key_invert ? ~depth_uint : depth_uint;
   uint64_t packed = PACK_VISBUF(key, tri_id);
   uint64_t *visbuf = (uint64_t *)(uintptr_t)args->framebuffer;
   atomicMin(&visbuf[py * args->width + px], packed);
}

/*
 * Stage 1: 1 thread per triangle.
 * Small triangles are rasterized with incremental edge stepping.
 * Large triangles are pushed to the nontrivial queue for stage 2.
 */
extern "C" __global__ void
cp_rasterize_stage1(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   uint32_t tri_id = blockIdx.x * blockDim.x + threadIdx.x;
   if (tri_id >= args.num_triangles)
      return;

   struct tri_setup s;
   if (!setup_triangle(&args, tri_id, &s))
      return;

   int bb_w = s.ix_max - s.ix_min + 1;
   int bb_h = s.iy_max - s.iy_min + 1;
   int bb_area = bb_w * bb_h;

   if (bb_area <= 0)
      return;

   if (bb_area > CP_SMALL_THRESHOLD) {
      uint32_t *counter = (uint32_t *)(uintptr_t)queues.nontrivial_count;
      uint32_t idx = atomicAdd(counter, 1u);
      if (idx < CP_MAX_NONTRIVIAL) {
         uint32_t *queue = (uint32_t *)(uintptr_t)queues.nontrivial;
         queue[idx] = tri_id;
      }
      return;
   }

   /* Incremental edge stepping for small triangles */
   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;

   float start_x = (float)s.ix_min + 0.5f;
   float start_y = (float)s.iy_min + 0.5f;

   float e0_init = edge_function(s.sx1, s.sy1, s.sx2, s.sy2, start_x, start_y);
   float e1_init = edge_function(s.sx2, s.sy2, s.sx0, s.sy0, start_x, start_y);
   float e2_init = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, start_x, start_y);

   float e0_row = e0_init;
   float e1_row = e1_init;
   float e2_row = e2_init;

   for (int py = s.iy_min; py <= s.iy_max; py++) {
      float e0 = e0_row;
      float e1 = e1_row;
      float e2 = e2_row;

      for (int px = s.ix_min; px <= s.ix_max; px++) {
         if (e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f) {
            float w0 = e0 * s.inv_area;
            float w1 = e1 * s.inv_area;
            float w2 = 1.0f - w0 - w1;

            float depth = w0 * s.ndc_z0 + w1 * s.ndc_z1 + w2 * s.ndc_z2;
            depth = depth * 0.5f + 0.5f;

            uint32_t depth_uint = float_to_sortable_uint(depth);

            if (args.depth_test && args.depthbuf) {
               uint32_t prev =
                  ((const uint32_t *)(uintptr_t)args.depthbuf)[py * args.width + px];
               bool pass;
               switch (args.depth_func) {
               case CP_FUNC_NEVER:     pass = false; break;
               case CP_FUNC_LESS:      pass = depth_uint <  prev; break;
               case CP_FUNC_EQUAL:     pass = depth_uint == prev; break;
               case CP_FUNC_LEQUAL:    pass = depth_uint <= prev; break;
               case CP_FUNC_GREATER:   pass = depth_uint >  prev; break;
               case CP_FUNC_NOTEQUAL:  pass = depth_uint != prev; break;
               case CP_FUNC_GEQUAL:    pass = depth_uint >= prev; break;
               default:                pass = true; break;
               }
               if (!pass) {
                  e0 += s.e0_dx;
                  e1 += s.e1_dx;
                  e2 += s.e2_dx;
                  continue;
               }
            }

            uint32_t key = args.depth_key_invert ? ~depth_uint : depth_uint;
            uint64_t packed = PACK_VISBUF(key, tri_id);
            atomicMin(&visbuf[py * args.width + px], packed);
         }
         e0 += s.e0_dx;
         e1 += s.e1_dx;
         e2 += s.e2_dx;
      }
      e0_row += s.e0_dy;
      e1_row += s.e1_dy;
      e2_row += s.e2_dy;
   }
}

/*
 * Stage 2: 1 warp (32 threads) per nontrivial triangle.
 * Medium triangles are rasterized cooperatively. Huge triangles are decomposed
 * into tiles and pushed to the huge queue.
 */
extern "C" __global__ void
cp_rasterize_stage2(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   /* Each warp claims one triangle from the nontrivial queue */
   uint32_t lane_id = threadIdx.x % 32;
   uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

   uint32_t *nt_counter = (uint32_t *)(uintptr_t)queues.nontrivial_count;
   uint32_t num_nontrivial = *nt_counter;
   if (warp_id >= num_nontrivial)
      return;

   uint32_t *nt_queue = (uint32_t *)(uintptr_t)queues.nontrivial;
   uint32_t tri_id = nt_queue[warp_id];

   /* Bounds check: tri_id must be < num_triangles */
   if (tri_id >= args.num_triangles)
      return;

   /* Lane 0 does triangle setup */
   float sx0, sy0, sx1, sy1, sx2, sy2;
   float ndc_z0, ndc_z1, ndc_z2;
   float inv_area;
   float e0_dx, e0_dy, e1_dx, e1_dy, e2_dx, e2_dy;
   int ix_min, iy_min, ix_max, iy_max;
   int valid = 0;

   if (lane_id == 0) {
      struct tri_setup s;
      if (setup_triangle(&args, tri_id, &s)) {
         sx0 = s.sx0; sy0 = s.sy0;
         sx1 = s.sx1; sy1 = s.sy1;
         sx2 = s.sx2; sy2 = s.sy2;
         ndc_z0 = s.ndc_z0; ndc_z1 = s.ndc_z1; ndc_z2 = s.ndc_z2;
         inv_area = s.inv_area;
         e0_dx = s.e0_dx; e0_dy = s.e0_dy;
         e1_dx = s.e1_dx; e1_dy = s.e1_dy;
         e2_dx = s.e2_dx; e2_dy = s.e2_dy;
         ix_min = s.ix_min; iy_min = s.iy_min;
         ix_max = s.ix_max; iy_max = s.iy_max;
         valid = 1;
      }
   }

   /* Broadcast setup from lane 0 to all lanes */
   valid   = __shfl_sync(0xFFFFFFFF, valid, 0);
   if (!valid)
      return;

   sx0     = __shfl_sync(0xFFFFFFFF, sx0, 0);
   sy0     = __shfl_sync(0xFFFFFFFF, sy0, 0);
   sx1     = __shfl_sync(0xFFFFFFFF, sx1, 0);
   sy1     = __shfl_sync(0xFFFFFFFF, sy1, 0);
   sx2     = __shfl_sync(0xFFFFFFFF, sx2, 0);
   sy2     = __shfl_sync(0xFFFFFFFF, sy2, 0);
   ndc_z0  = __shfl_sync(0xFFFFFFFF, ndc_z0, 0);
   ndc_z1  = __shfl_sync(0xFFFFFFFF, ndc_z1, 0);
   ndc_z2  = __shfl_sync(0xFFFFFFFF, ndc_z2, 0);
   inv_area = __shfl_sync(0xFFFFFFFF, inv_area, 0);
   e0_dx   = __shfl_sync(0xFFFFFFFF, e0_dx, 0);
   e0_dy   = __shfl_sync(0xFFFFFFFF, e0_dy, 0);
   e1_dx   = __shfl_sync(0xFFFFFFFF, e1_dx, 0);
   e1_dy   = __shfl_sync(0xFFFFFFFF, e1_dy, 0);
   e2_dx   = __shfl_sync(0xFFFFFFFF, e2_dx, 0);
   e2_dy   = __shfl_sync(0xFFFFFFFF, e2_dy, 0);
   ix_min  = __shfl_sync(0xFFFFFFFF, ix_min, 0);
   iy_min  = __shfl_sync(0xFFFFFFFF, iy_min, 0);
   ix_max  = __shfl_sync(0xFFFFFFFF, ix_max, 0);
   iy_max  = __shfl_sync(0xFFFFFFFF, iy_max, 0);

   int bb_w = ix_max - ix_min + 1;
   int bb_h = iy_max - iy_min + 1;
   int bb_area = bb_w * bb_h;

   /* Huge triangles: decompose into tiles and push to stage 3 */
   if (bb_area > CP_MEDIUM_THRESHOLD) {
      if (lane_id == 0) {
         int tile_x_min = ix_min / CP_TILE_SIZE;
         int tile_y_min = iy_min / CP_TILE_SIZE;
         int tile_x_max = ix_max / CP_TILE_SIZE;
         int tile_y_max = iy_max / CP_TILE_SIZE;
         int num_tiles = (tile_x_max - tile_x_min + 1) *
                         (tile_y_max - tile_y_min + 1);

         uint32_t *huge_counter = (uint32_t *)(uintptr_t)queues.huge_count;
         uint32_t base_idx = atomicAdd(huge_counter, (uint32_t)num_tiles);

         struct cp_tile_pair *huge_queue =
            (struct cp_tile_pair *)(uintptr_t)queues.huge_tiles;

         int idx = 0;
         for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
            for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
               if (base_idx + idx < CP_MAX_HUGE_TILES) {
                  huge_queue[base_idx + idx].tri_id = tri_id;
                  huge_queue[base_idx + idx].tile_x = (uint16_t)(tx * CP_TILE_SIZE);
                  huge_queue[base_idx + idx].tile_y = (uint16_t)(ty * CP_TILE_SIZE);
               }
               idx++;
            }
         }
      }
      return;
   }

   /* Medium triangle: all 32 lanes iterate the bounding box with stride 32 */
   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;

   for (int i = (int)lane_id; i < bb_area; i += 32) {
      int px = ix_min + (i % bb_w);
      int py = iy_min + (i / bb_w);

      if (px < 0 || px >= (int)args.width || py < 0 || py >= (int)args.height)
         continue;

      float cx = (float)px + 0.5f;
      float cy = (float)py + 0.5f;

      float e0 = (cx - sx1) * (sy2 - sy1) - (cy - sy1) * (sx2 - sx1);
      float e1 = (cx - sx2) * (sy0 - sy2) - (cy - sy2) * (sx0 - sx2);
      float e2 = (cx - sx0) * (sy1 - sy0) - (cy - sy0) * (sx1 - sx0);

      if (e0 < 0.0f || e1 < 0.0f || e2 < 0.0f)
         continue;

      float w0 = e0 * inv_area;
      float w1 = e1 * inv_area;

      float depth = w0 * ndc_z0 + w1 * ndc_z1 + (1.0f - w0 - w1) * ndc_z2;
      depth = depth * 0.5f + 0.5f;

      uint32_t depth_uint = float_to_sortable_uint(depth);

      if (args.depth_test && args.depthbuf) {
         uint32_t prev =
            ((const uint32_t *)(uintptr_t)args.depthbuf)[py * args.width + px];
         bool pass;
         switch (args.depth_func) {
         case CP_FUNC_NEVER:     pass = false; break;
         case CP_FUNC_LESS:      pass = depth_uint <  prev; break;
         case CP_FUNC_EQUAL:     pass = depth_uint == prev; break;
         case CP_FUNC_LEQUAL:    pass = depth_uint <= prev; break;
         case CP_FUNC_GREATER:   pass = depth_uint >  prev; break;
         case CP_FUNC_NOTEQUAL:  pass = depth_uint != prev; break;
         case CP_FUNC_GEQUAL:    pass = depth_uint >= prev; break;
         default:                pass = true; break;
         }
         if (!pass)
            continue;
      }

      uint32_t key = args.depth_key_invert ? ~depth_uint : depth_uint;
      uint64_t packed = PACK_VISBUF(key, tri_id);
      atomicMin(&visbuf[py * args.width + px], packed);
   }
}

/*
 * Stage 3: 1 block (64 threads) per tile-triangle pair.
 * Evaluates edge functions at tile corners for trivial accept/reject,
 * then rasterizes the tile with per-pixel tests only where needed.
 */
extern "C" __global__ void
cp_rasterize_stage3(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   uint32_t tile_idx = blockIdx.x;

   uint32_t *huge_counter = (uint32_t *)(uintptr_t)queues.huge_count;
   uint32_t num_tiles = *huge_counter;
   if (tile_idx >= num_tiles)
      return;

   struct cp_tile_pair *huge_queue =
      (struct cp_tile_pair *)(uintptr_t)queues.huge_tiles;

   uint32_t tri_id = huge_queue[tile_idx].tri_id;
   int tile_x = (int)huge_queue[tile_idx].tile_x;
   int tile_y = (int)huge_queue[tile_idx].tile_y;

   /* Setup triangle (shared across block via shared memory) */
   __shared__ float sh_sx0, sh_sy0, sh_sx1, sh_sy1, sh_sx2, sh_sy2;
   __shared__ float sh_ndc_z0, sh_ndc_z1, sh_ndc_z2;
   __shared__ float sh_inv_area;
   __shared__ float sh_e0_dx, sh_e0_dy, sh_e1_dx, sh_e1_dy, sh_e2_dx, sh_e2_dy;
   __shared__ uint32_t sh_plane_mask;
   __shared__ int sh_valid;

   if (threadIdx.x == 0) {
      struct tri_setup s;
      sh_valid = 0;
      if (setup_triangle(&args, tri_id, &s)) {
         sh_sx0 = s.sx0; sh_sy0 = s.sy0;
         sh_sx1 = s.sx1; sh_sy1 = s.sy1;
         sh_sx2 = s.sx2; sh_sy2 = s.sy2;
         sh_ndc_z0 = s.ndc_z0; sh_ndc_z1 = s.ndc_z1; sh_ndc_z2 = s.ndc_z2;
         sh_inv_area = s.inv_area;
         sh_e0_dx = s.e0_dx; sh_e0_dy = s.e0_dy;
         sh_e1_dx = s.e1_dx; sh_e1_dy = s.e1_dy;
         sh_e2_dx = s.e2_dx; sh_e2_dy = s.e2_dy;

         /* Trivial accept/reject: evaluate edges at tile corners */
         float tx0 = (float)tile_x + 0.5f;
         float ty0 = (float)tile_y + 0.5f;
         float tx1 = tx0 + (float)(CP_TILE_SIZE - 1);
         float ty1 = ty0 + (float)(CP_TILE_SIZE - 1);

         /* Edge 0 at 4 corners */
         float e0_tl = (tx0 - s.sx1) * (s.sy2 - s.sy1) - (ty0 - s.sy1) * (s.sx2 - s.sx1);
         float e0_tr = e0_tl + s.e0_dx * (float)(CP_TILE_SIZE - 1);
         float e0_bl = e0_tl + s.e0_dy * (float)(CP_TILE_SIZE - 1);
         float e0_br = e0_tl + s.e0_dx * (float)(CP_TILE_SIZE - 1) +
                       s.e0_dy * (float)(CP_TILE_SIZE - 1);

         /* Edge 1 at 4 corners */
         float e1_tl = (tx0 - s.sx2) * (s.sy0 - s.sy2) - (ty0 - s.sy2) * (s.sx0 - s.sx2);
         float e1_tr = e1_tl + s.e1_dx * (float)(CP_TILE_SIZE - 1);
         float e1_bl = e1_tl + s.e1_dy * (float)(CP_TILE_SIZE - 1);
         float e1_br = e1_tl + s.e1_dx * (float)(CP_TILE_SIZE - 1) +
                       s.e1_dy * (float)(CP_TILE_SIZE - 1);

         /* Edge 2 at 4 corners */
         float e2_tl = (tx0 - s.sx0) * (s.sy1 - s.sy0) - (ty0 - s.sy0) * (s.sx1 - s.sx0);
         float e2_tr = e2_tl + s.e2_dx * (float)(CP_TILE_SIZE - 1);
         float e2_bl = e2_tl + s.e2_dy * (float)(CP_TILE_SIZE - 1);
         float e2_br = e2_tl + s.e2_dx * (float)(CP_TILE_SIZE - 1) +
                       s.e2_dy * (float)(CP_TILE_SIZE - 1);

         /* Check for trivial reject (any edge fully outside) */
         bool e0_reject = (e0_tl < 0) && (e0_tr < 0) && (e0_bl < 0) && (e0_br < 0);
         bool e1_reject = (e1_tl < 0) && (e1_tr < 0) && (e1_bl < 0) && (e1_br < 0);
         bool e2_reject = (e2_tl < 0) && (e2_tr < 0) && (e2_bl < 0) && (e2_br < 0);

         if (e0_reject || e1_reject || e2_reject) {
            sh_valid = 0;
         } else {
            sh_valid = 1;
            /* Determine which edges need per-pixel testing */
            uint32_t mask = 0;
            bool e0_accept = (e0_tl >= 0) && (e0_tr >= 0) && (e0_bl >= 0) && (e0_br >= 0);
            bool e1_accept = (e1_tl >= 0) && (e1_tr >= 0) && (e1_bl >= 0) && (e1_br >= 0);
            bool e2_accept = (e2_tl >= 0) && (e2_tr >= 0) && (e2_bl >= 0) && (e2_br >= 0);
            if (!e0_accept) mask |= 1;
            if (!e1_accept) mask |= 2;
            if (!e2_accept) mask |= 4;
            sh_plane_mask = mask;
         }
      }
   }
   __syncthreads();

   if (!sh_valid)
      return;

   /* Each thread handles one column of the tile */
   int col = (int)threadIdx.x;
   if (col >= CP_TILE_SIZE)
      return;

   int px = tile_x + col;
   if (px >= (int)args.width)
      return;

   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;
   uint32_t plane_mask = sh_plane_mask;

   /* Compute edge values at the start of this column */
   float cx = (float)px + 0.5f;
   float cy = (float)tile_y + 0.5f;

   float e0 = (cx - sh_sx1) * (sh_sy2 - sh_sy1) - (cy - sh_sy1) * (sh_sx2 - sh_sx1);
   float e1 = (cx - sh_sx2) * (sh_sy0 - sh_sy2) - (cy - sh_sy2) * (sh_sx0 - sh_sx2);
   float e2 = (cx - sh_sx0) * (sh_sy1 - sh_sy0) - (cy - sh_sy0) * (sh_sx1 - sh_sx0);

   for (int row = 0; row < CP_TILE_SIZE; row++) {
      int py = tile_y + row;
      if (py >= (int)args.height)
         break;

      /* Test only partial edges */
      bool inside = true;
      if (plane_mask & 1) inside &= (e0 >= 0.0f);
      if (plane_mask & 2) inside &= (e1 >= 0.0f);
      if (plane_mask & 4) inside &= (e2 >= 0.0f);

      if (inside) {
         float w0 = e0 * sh_inv_area;
         float w1 = e1 * sh_inv_area;

         float depth = w0 * sh_ndc_z0 + w1 * sh_ndc_z1 +
                       (1.0f - w0 - w1) * sh_ndc_z2;
         depth = depth * 0.5f + 0.5f;

         uint32_t depth_uint = float_to_sortable_uint(depth);

         if (args.depth_test && args.depthbuf) {
            uint32_t prev =
               ((const uint32_t *)(uintptr_t)args.depthbuf)[py * args.width + px];
            bool pass;
            switch (args.depth_func) {
            case CP_FUNC_NEVER:     pass = false; break;
            case CP_FUNC_LESS:      pass = depth_uint <  prev; break;
            case CP_FUNC_EQUAL:     pass = depth_uint == prev; break;
            case CP_FUNC_LEQUAL:    pass = depth_uint <= prev; break;
            case CP_FUNC_GREATER:   pass = depth_uint >  prev; break;
            case CP_FUNC_NOTEQUAL:  pass = depth_uint != prev; break;
            case CP_FUNC_GEQUAL:    pass = depth_uint >= prev; break;
            default:                pass = true; break;
            }
            if (!pass) {
               e0 += sh_e0_dy;
               e1 += sh_e1_dy;
               e2 += sh_e2_dy;
               continue;
            }
         }

         uint32_t key = args.depth_key_invert ? ~depth_uint : depth_uint;
         uint64_t packed = PACK_VISBUF(key, tri_id);
         atomicMin(&visbuf[py * args.width + px], packed);
      }

      e0 += sh_e0_dy;
      e1 += sh_e1_dy;
      e2 += sh_e2_dy;
   }
}

/*
 * Clear visibility buffer to "empty" (max depth, invalid triID)
 */
extern "C" __global__ void
cp_clear_visbuf(uint64_t *visbuf, uint32_t width, uint32_t height)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x < width && y < height)
      visbuf[y * width + x] = VISBUF_EMPTY;
}

extern "C" __global__ void
cp_resolve_visbuf(struct cp_resolve_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= args.width || y >= args.height)
      return;

   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.visbuf;
   uint32_t *color_out = (uint32_t *)(uintptr_t)args.color_out;
   float4 *positions = (float4 *)(uintptr_t)args.positions;

   uint64_t entry = visbuf[y * args.width + x];
   if (entry == VISBUF_EMPTY) {
      return;
   }

   uint32_t tri_id = VISBUF_TRIID(entry);

   float4 v0 = positions[tri_id * 3 + 0];
   float4 v1 = positions[tri_id * 3 + 1];
   float4 v2 = positions[tri_id * 3 + 2];

   float sx0 = (v0.x / v0.w) * args.vp_scale_x + args.vp_trans_x;
   float sy0 = (v0.y / v0.w) * args.vp_scale_y + args.vp_trans_y;
   float sx1 = (v1.x / v1.w) * args.vp_scale_x + args.vp_trans_x;
   float sy1 = (v1.y / v1.w) * args.vp_scale_y + args.vp_trans_y;
   float sx2 = (v2.x / v2.w) * args.vp_scale_x + args.vp_trans_x;
   float sy2 = (v2.y / v2.w) * args.vp_scale_y + args.vp_trans_y;

   float cx = (float)x + 0.5f;
   float cy = (float)y + 0.5f;
   float area = edge_function(sx0, sy0, sx1, sy1, sx2, sy2);
   float inv_area = 1.0f / area;
   float w0 = edge_function(sx1, sy1, sx2, sy2, cx, cy) * inv_area;
   float w1 = edge_function(sx2, sy2, sx0, sy0, cx, cy) * inv_area;
   float w2 = 1.0f - w0 - w1;

   float r, g, b, a;
   if (args.colors) {
      uint8_t *color_base = (uint8_t *)(uintptr_t)args.colors;
      uint32_t cs = args.color_stride;
      float4 *c0 = (float4 *)(color_base + (tri_id * 3 + 0) * cs);
      float4 *c1 = (float4 *)(color_base + (tri_id * 3 + 1) * cs);
      float4 *c2 = (float4 *)(color_base + (tri_id * 3 + 2) * cs);
      r = w0 * c0->x + w1 * c1->x + w2 * c2->x;
      g = w0 * c0->y + w1 * c1->y + w2 * c2->y;
      b = w0 * c0->z + w1 * c1->z + w2 * c2->z;
      a = w0 * c0->w + w1 * c1->w + w2 * c2->w;
   } else {
      r = g = b = a = 1.0f;
   }

   uint32_t ri = (uint32_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f + 0.5f);
   uint32_t gi = (uint32_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f + 0.5f);
   uint32_t bi = (uint32_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f + 0.5f);
   uint32_t ai = (uint32_t)(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f + 0.5f);
   color_out[y * args.width + x] = ri | (gi << 8) | (bi << 16) | (ai << 24);
}
