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

/*
 * Signed area of the triangle (a, b, p), positive when p is on the inside of
 * the directed edge a -> b.
 *
 * Two properties make coverage watertight, and both are load bearing.
 *
 * It is a cross product of the two vectors from p rather than the more obvious
 * (p - a) x (b - a). Swapping a and b then reorders the same two products
 * around the subtraction, so the result negates bit for bit. Two triangles
 * that share an edge walk it in opposite directions, so they see exactly
 * opposite values and the fill rule below can hand the pixel to one of them.
 * The (p - a) form rounds differently for each ordering instead, which lets
 * both triangles compute a small negative and drop the pixel.
 *
 * The products and the subtraction are the explicit round-to-nearest
 * intrinsics, because nvcc contracts a * b - c * d into fma(a, b, -(c * d)) by
 * default. That keeps only one of the two products exact, and which one it is
 * depends on the order they were written in — so the contracted form is not
 * antisymmetric either, and swapping a and b gives a value that is not the
 * negation but, at a pixel centre almost exactly on the edge, the very same
 * small negative. Both triangles then reject.
 *
 * That is what put a one pixel diagonal line through the quads in
 * computeshader: the shared edge runs corner to corner, so it grazes pixel
 * centres for its whole length.
 */
static __device__ __forceinline__ float
edge_function(float ax, float ay, float bx, float by, float px, float py)
{
   return __fsub_rn(__fmul_rn(bx - px, ay - py),
                    __fmul_rn(by - py, ax - px));
}

/*
 * The top-left fill rule, which breaks the +-0 tie above so that exactly one
 * of the two triangles claims a pixel centred on their shared edge.
 *
 * The inside of the edge a -> b lies towards (dy, -dx), so with y pointing
 * down a left edge is one with dy > 0, and a top edge is horizontal with its
 * inside below it.
 */
static __device__ __forceinline__ bool
edge_is_top_left(float ax, float ay, float bx, float by)
{
   float dx = bx - ax;
   float dy = by - ay;
   return dy > 0.0f || (dy == 0.0f && dx < 0.0f);
}

static __device__ __forceinline__ bool
edge_inside(float e, bool top_left)
{
   /* -0.0f >= 0.0f is true, so an inclusive edge takes the pixel whichever
    * sign of zero it computed. */
   return top_left ? (e >= 0.0f) : (e > 0.0f);
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
   bool e0_top_left, e1_top_left, e2_top_left;
   int ix_min, iy_min, ix_max, iy_max;
   /* A point covers this screen-space square instead of the edges above. */
   bool is_point;
   float pt_x0, pt_y0, pt_x1, pt_y1;
};

/*
 * Clipping against the two planes that make the perspective divide meaningful,
 * one thread per input triangle.
 *
 * Vulkan's view volume is 0 <= z <= w. Two of its planes matter here:
 *
 *   z >= 0      the near plane. A ground plane running to the horizon crosses
 *               it, and the part behind the eye must be cut or it reappears
 *               mirrored in front.
 *   w >  0      not a clip plane of its own but implied by z <= w. A vertex
 *               with w <= 0 divides to a garbage position however small its z,
 *               so it has to go before setup_triangle() touches it.
 *
 * Clipping on either plane alone is not enough: samples exist that only the
 * first fixes and samples that only the second fixes.
 *
 * Interpolation happens in clip space, before the divide, which is what makes
 * the split exact for both position and varyings.
 */
#define CP_CLIP_MAX_VERTS 5   /* a triangle cut by two planes */

static __device__ __forceinline__ float
clip_dist(const float4 *v, int plane)
{
   return plane == 0 ? v[0].z : v[0].w - 1e-6f;
}

static __device__ __forceinline__ void
clip_lerp(float4 *dst, const float4 *a, const float4 *b, float t, uint32_t slots)
{
   for (uint32_t s = 0; s < slots; s++) {
      float4 va = a[s], vb = b[s];
      dst[s] = make_float4(va.x + (vb.x - va.x) * t,
                           va.y + (vb.y - va.y) * t,
                           va.z + (vb.z - va.z) * t,
                           va.w + (vb.w - va.w) * t);
   }
}

static __device__ __forceinline__ void
clip_copy(float4 *dst, const float4 *src, uint32_t slots)
{
   for (uint32_t s = 0; s < slots; s++)
      dst[s] = src[s];
}

/* Sutherland-Hodgman against one plane. Returns the new vertex count. */
static __device__ int
clip_poly(float4 *dst, const float4 *src, int n, uint32_t slots, int plane)
{
   int out_n = 0;

   for (int i = 0; i < n; i++) {
      const float4 *cur = src + (size_t)i * slots;
      const float4 *nxt = src + (size_t)((i + 1) % n) * slots;
      float dc = clip_dist(cur, plane);
      float dn = clip_dist(nxt, plane);
      bool in_c = dc >= 0.0f, in_n = dn >= 0.0f;

      if (in_c && out_n < CP_CLIP_MAX_VERTS)
         clip_copy(dst + (size_t)out_n++ * slots, cur, slots);

      if (in_c != in_n && out_n < CP_CLIP_MAX_VERTS) {
         float denom = dc - dn;
         if (denom != 0.0f)
            clip_lerp(dst + (size_t)out_n++ * slots, cur, nxt, dc / denom, slots);
      }
   }

   return out_n;
}

static __device__ __forceinline__ float4 *
clip_emit(struct cp_clip_args *args, float4 *out, uint32_t slots)
{
   uint32_t o = atomicAdd((unsigned int *)(uintptr_t)args->out_count, 1u);
   if (o >= args->max_triangles)
      return NULL;
   return out + (size_t)o * 3 * slots;
}

extern "C" __global__ void
cp_clip_triangles(struct cp_clip_args args)
{
   uint32_t tri = blockIdx.x * blockDim.x + threadIdx.x;
   if (tri >= args.num_triangles)
      return;

   uint32_t slots = args.num_slots;
   if (slots > CP_MAX_CLIP_SLOTS)
      slots = CP_MAX_CLIP_SLOTS;

   const float4 *in = (const float4 *)(uintptr_t)args.vs_out;
   float4 *out = (float4 *)(uintptr_t)args.out;
   const float4 *v = in + (size_t)tri * 3 * slots;

   /* The overwhelming majority of triangles are wholly inside, so check that
    * first and copy straight through — the polygon buffers below live in local
    * memory and are worth avoiding. */
   int inside = 0;
   for (int i = 0; i < 3; i++) {
      const float4 *vi = v + (size_t)i * slots;
      if (clip_dist(vi, 0) >= 0.0f && clip_dist(vi, 1) >= 0.0f)
         inside++;
   }

   if (inside == 3) {
      float4 *dst = clip_emit(&args, out, slots);
      if (dst)
         clip_copy(dst, v, 3 * slots);
      return;
   }

   float4 poly_a[CP_CLIP_MAX_VERTS * CP_MAX_CLIP_SLOTS];
   float4 poly_b[CP_CLIP_MAX_VERTS * CP_MAX_CLIP_SLOTS];

   int n = clip_poly(poly_a, v, 3, slots, 0);
   if (n < 3)
      return;
   n = clip_poly(poly_b, poly_a, n, slots, 1);
   if (n < 3)
      return;

   /* Fan-triangulate the clipped polygon, which keeps the original winding. */
   for (int i = 1; i + 1 < n; i++) {
      float4 *dst = clip_emit(&args, out, slots);
      if (!dst)
         return;
      clip_copy(dst, poly_b, slots);
      clip_copy(dst + slots, poly_b + (size_t)i * slots, slots);
      clip_copy(dst + 2 * slots, poly_b + (size_t)(i + 1) * slots, slots);
   }
}

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

   /*
    * A point is one vertex wearing a square. The host expands POINT_LIST into
    * degenerate triangles, so there are no edges to test and no winding to
    * cull — the square comes from gl_PointSize, which the vertex shader wrote
    * into one of its output slots, and every pixel inside it takes the
    * vertex's own depth and varyings.
    */
   if (args->point_mode) {
      s->is_point = true;
      s->ndc_z0 = s->ndc_z1 = s->ndc_z2 = v0.z * inv_w0;

      float cx = v0.x * inv_w0 * args->vp_scale_x + args->vp_trans_x;
      float cy = v0.y * inv_w0 * args->vp_scale_y + args->vp_trans_y;

      float size = 1.0f;
      if (args->psiz_slot >= 0)
         size = positions[(tri_id * 3 + 0) * pos_stride + args->psiz_slot].x;

      /* A zero or negative size draws nothing; NaN fails this too. */
      if (!(size > 0.0f))
         return false;
      if (size > CP_MAX_POINT_SIZE)
         size = CP_MAX_POINT_SIZE;

      float half = size * 0.5f;
      s->pt_x0 = cx - half; s->pt_x1 = cx + half;
      s->pt_y0 = cy - half; s->pt_y1 = cy + half;
      s->sx0 = cx; s->sy0 = cy;
      s->inv_area = 1.0f;

      s->ix_min = max((int)floorf(s->pt_x0), 0);
      s->iy_min = max((int)floorf(s->pt_y0), 0);
      s->ix_max = min((int)ceilf(s->pt_x1), (int)args->width - 1);
      s->iy_max = min((int)ceilf(s->pt_y1), (int)args->height - 1);
      return true;
   }

   s->is_point = false;

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

   /* After the flip the winding is fixed, so the fill rule can be decided
    * once per triangle rather than per pixel. */
   s->e0_top_left = edge_is_top_left(s->sx1, s->sy1, s->sx2, s->sy2);
   s->e1_top_left = edge_is_top_left(s->sx2, s->sy2, s->sx0, s->sy0);
   s->e2_top_left = edge_is_top_left(s->sx0, s->sy0, s->sx1, s->sy1);

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

/*
 * Whether this triangle already discarded at this pixel on an earlier pass of
 * an alpha-tested draw. Visibility is resolved before shading, so without this
 * the pixel would keep choosing the same transparent fragment and whatever is
 * behind it would never be drawn.
 */
static __device__ __forceinline__ bool
cp_tri_rejected(const struct cp_rasterize_args *args, uint32_t tri_id,
                int px, int py)
{
   if (!args->reject || !args->reject_passes)
      return false;

   const uint32_t *rej = (const uint32_t *)(uintptr_t)args->reject +
      (size_t)((uint32_t)py * args->width + (uint32_t)px) * args->reject_layers;
   for (uint32_t i = 0; i < args->reject_passes; i++)
      if (rej[i] == tri_id)
         return true;
   return false;
}

/*
 * Depth test one fragment and stake its claim on the pixel. Shared by the
 * point path below; the triangle stages inline the same sequence because they
 * carry the edge values along with them.
 */
static __device__ __forceinline__ void
emit_fragment(struct cp_rasterize_args *args, uint32_t tri_id,
              int px, int py, float ndc_z)
{
   if (cp_tri_rejected(args, tri_id, px, py))
      return;

   uint32_t depth_uint = float_to_sortable_uint(ndc_z * 0.5f + 0.5f);

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

   uint64_t *visbuf = (uint64_t *)(uintptr_t)args->framebuffer;

   /*
    * Ordered blending picks by submission order rather than by depth: the
    * lowest numbered primitive this pixel has not composited yet. Keying the
    * visibility buffer on the primitive index makes the same atomicMin do it.
    */
   if (args->blend_peel) {
      uint32_t idx = py * args->width + px;
      const uint32_t *next = (const uint32_t *)(uintptr_t)args->peel_next;
      if (tri_id < next[idx])
         return;
      atomicMin(&visbuf[idx], PACK_VISBUF(tri_id, tri_id));
      /* Tells the host another pass is worth running. Racing writers all store
       * the same value, so the read first is only there to spare the traffic. */
      uint32_t *any = (uint32_t *)(uintptr_t)args->peel_any;
      if (any && !*any)
         *any = 1u;
      return;
   }

   uint32_t key = args->depth_key_invert ? ~depth_uint : depth_uint;
   atomicMin(&visbuf[py * args->width + px], PACK_VISBUF(key, tri_id));
}

/*
 * Every pixel whose centre falls inside the point's square, at the vertex's
 * own depth. Half-open on both axes so two points that abut do not both claim
 * the shared row, the same reason the triangle path has a fill rule.
 */
static __device__ __forceinline__ void
rasterize_point(struct cp_rasterize_args *args, struct tri_setup *s,
                uint32_t tri_id)
{
   for (int py = s->iy_min; py <= s->iy_max; py++) {
      float cy = (float)py + 0.5f;
      if (cy < s->pt_y0 || cy >= s->pt_y1)
         continue;
      for (int px = s->ix_min; px <= s->ix_max; px++) {
         float cx = (float)px + 0.5f;
         if (cx < s->pt_x0 || cx >= s->pt_x1)
            continue;
         emit_fragment(args, tri_id, px, py, s->ndc_z0);
      }
   }
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
   /* After clipping the count lives on the device, so the grid is sized for
    * the worst case and each thread bounds itself. */
   uint32_t num_triangles = args.tri_count
      ? *(const volatile uint32_t *)(uintptr_t)args.tri_count
      : args.num_triangles;
   if (tri_id >= num_triangles)
      return;

   struct tri_setup s;
   if (!setup_triangle(&args, tri_id, &s))
      return;

   /* Points never go to the later stages: their size is clamped, so the loop
    * is bounded and one thread can always afford it. */
   if (s.is_point) {
      rasterize_point(&args, &s, tri_id);
      return;
   }

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

   /*
    * Small triangles, one pixel at a time. The edge functions are evaluated
    * from the vertices at every pixel rather than stepped by their gradients
    * across the bounding box: stepping accumulates rounding, and then the two
    * triangles either side of a shared edge no longer see exactly opposite
    * values, which is the property the fill rule needs to keep coverage
    * watertight. Stepping is worth restoring only with a form that stays
    * exact, such as snapping the vertices to a subpixel grid and iterating in
    * integers the way llvmpipe's lp_setup_tri.c does.
    */
   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;

   for (int py = s.iy_min; py <= s.iy_max; py++) {
      float cy = (float)py + 0.5f;

      for (int px = s.ix_min; px <= s.ix_max; px++) {
         float cx = (float)px + 0.5f;

         float e0 = edge_function(s.sx1, s.sy1, s.sx2, s.sy2, cx, cy);
         float e1 = edge_function(s.sx2, s.sy2, s.sx0, s.sy0, cx, cy);
         float e2 = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, cx, cy);

         if (edge_inside(e0, s.e0_top_left) &&
             edge_inside(e1, s.e1_top_left) &&
             edge_inside(e2, s.e2_top_left)) {
            float w0 = e0 * s.inv_area;
            float w1 = e1 * s.inv_area;
            float w2 = 1.0f - w0 - w1;

            emit_fragment(&args, tri_id, px, py,
                          w0 * s.ndc_z0 + w1 * s.ndc_z1 + w2 * s.ndc_z2);
         }
      }
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
   int e0_tl, e1_tl, e2_tl;
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
         e0_tl = s.e0_top_left;
         e1_tl = s.e1_top_left;
         e2_tl = s.e2_top_left;
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
   e0_tl   = __shfl_sync(0xFFFFFFFF, e0_tl, 0);
   e1_tl   = __shfl_sync(0xFFFFFFFF, e1_tl, 0);
   e2_tl   = __shfl_sync(0xFFFFFFFF, e2_tl, 0);
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

      float e0 = edge_function(sx1, sy1, sx2, sy2, cx, cy);
      float e1 = edge_function(sx2, sy2, sx0, sy0, cx, cy);
      float e2 = edge_function(sx0, sy0, sx1, sy1, cx, cy);

      if (!edge_inside(e0, e0_tl) ||
          !edge_inside(e1, e1_tl) ||
          !edge_inside(e2, e2_tl))
         continue;

      float w0 = e0 * inv_area;
      float w1 = e1 * inv_area;

      emit_fragment(&args, tri_id, px, py,
                    w0 * ndc_z0 + w1 * ndc_z1 + (1.0f - w0 - w1) * ndc_z2);
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
   __shared__ int sh_e0_tl, sh_e1_tl, sh_e2_tl;
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
         sh_e0_tl = s.e0_top_left;
         sh_e1_tl = s.e1_top_left;
         sh_e2_tl = s.e2_top_left;

         /*
          * Trivial reject: an edge that is outside at all four tile corners is
          * outside everywhere in the tile, because the edge function is
          * linear.
          *
          * There is no matching trivial accept any more. The mirror test —
          * every corner inside, so skip that edge per pixel — cannot be made
          * to respect the fill rule: a corner sitting exactly on an exclusive
          * edge reads as inside, and the tile would then claim a row of
          * pixels that belongs to its neighbour. Restoring it needs corner
          * values that bound the per-pixel ones, which the exact subpixel
          * form noted in stage 1 would give.
          */
         float tx0 = (float)tile_x + 0.5f;
         float ty0 = (float)tile_y + 0.5f;
         float tx1 = tx0 + (float)(CP_TILE_SIZE - 1);
         float ty1 = ty0 + (float)(CP_TILE_SIZE - 1);

         bool reject = false;
         for (int e = 0; e < 3 && !reject; e++) {
            float ax = e == 0 ? s.sx1 : (e == 1 ? s.sx2 : s.sx0);
            float ay = e == 0 ? s.sy1 : (e == 1 ? s.sy2 : s.sy0);
            float bx = e == 0 ? s.sx2 : (e == 1 ? s.sx0 : s.sx1);
            float by = e == 0 ? s.sy2 : (e == 1 ? s.sy0 : s.sy1);
            reject = edge_function(ax, ay, bx, by, tx0, ty0) < 0.0f &&
                     edge_function(ax, ay, bx, by, tx1, ty0) < 0.0f &&
                     edge_function(ax, ay, bx, by, tx0, ty1) < 0.0f &&
                     edge_function(ax, ay, bx, by, tx1, ty1) < 0.0f;
         }

         sh_valid = reject ? 0 : 1;
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

   float cx = (float)px + 0.5f;

   for (int row = 0; row < CP_TILE_SIZE; row++) {
      int py = tile_y + row;
      if (py >= (int)args.height)
         break;

      float cy = (float)py + 0.5f;

      float e0 = edge_function(sh_sx1, sh_sy1, sh_sx2, sh_sy2, cx, cy);
      float e1 = edge_function(sh_sx2, sh_sy2, sh_sx0, sh_sy0, cx, cy);
      float e2 = edge_function(sh_sx0, sh_sy0, sh_sx1, sh_sy1, cx, cy);

      if (edge_inside(e0, sh_e0_tl) &&
          edge_inside(e1, sh_e1_tl) &&
          edge_inside(e2, sh_e2_tl)) {
         float w0 = e0 * sh_inv_area;
         float w1 = e1 * sh_inv_area;

         emit_fragment(&args, tri_id, px, py,
                       w0 * sh_ndc_z0 + w1 * sh_ndc_z1 +
                       (1.0f - w0 - w1) * sh_ndc_z2);
      }
   }
}

/*
 * Step every pixel past the layer it just composited, so the next pass picks
 * up the one after it. A pixel the pass did not touch keeps its place.
 */
extern "C" __global__ void
cp_peel_advance(uint64_t *visbuf, uint32_t *peel_next,
                uint32_t width, uint32_t height)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height)
      return;
   uint64_t entry = visbuf[y * width + x];
   if (entry != VISBUF_EMPTY)
      peel_next[y * width + x] = VISBUF_TRIID(entry) + 1u;
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
