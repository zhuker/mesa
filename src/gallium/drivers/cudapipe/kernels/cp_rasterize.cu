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

/*
 * Where a sample sits inside its pixel. One sample is the centre, which keeps
 * the single-sample path bit-identical; the four are Vulkan's standard
 * locations for VK_SAMPLE_COUNT_4_BIT.
 */
static __device__ __forceinline__ void
cp_sample_pos(uint32_t num_samples, int s, float *ox, float *oy)
{
   if (num_samples <= 1) {
      *ox = 0.5f; *oy = 0.5f;
      return;
   }
   if (num_samples <= 4) {
      const float xs[4] = { 0.375f, 0.875f, 0.125f, 0.625f };
      const float ys[4] = { 0.125f, 0.375f, 0.625f, 0.875f };
      *ox = xs[s & 3]; *oy = ys[s & 3];
      return;
   }
   const float xs[8] = { 0.5625f, 0.4375f, 0.8125f, 0.3125f,
                         0.1875f, 0.0625f, 0.6875f, 0.9375f };
   const float ys[8] = { 0.3125f, 0.6875f, 0.5625f, 0.1875f,
                         0.8125f, 0.4375f, 0.9375f, 0.0625f };
   *ox = xs[s & 7]; *oy = ys[s & 7];
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
 * Clipping against the planes that make the perspective divide meaningful,
 * one thread per input triangle.
 *
 * Vulkan's view volume is 0 <= z <= w. Three planes matter here:
 *
 *   z >= 0      one of the depth planes. Under the conventional projection it
 *               is the near plane, and a ground plane running to the horizon
 *               crosses it.
 *   z <= w      the other one. Under a reversed-Z projection — depth cleared
 *               to 0 and tested GREATER_OR_EQUAL, which is what the real
 *               application uses — this is the near plane instead, and the two
 *               swap roles. Whichever it is, the part outside must be cut or
 *               it reappears mirrored in front: a vertex closer than the near
 *               plane still has w > 0 and z > 0, so neither of the other two
 *               tests rejects it, and it divides to a coordinate millions of
 *               pixels off screen.
 *   w >  0      not a clip plane of its own but implied by z <= w. A vertex
 *               with w <= 0 divides to a garbage position however small its z,
 *               so it has to go before setup_triangle() touches it.
 *
 * No one plane is enough: samples exist that only one of them fixes.
 *
 * Interpolation happens in clip space, before the divide, which is what makes
 * the split exact for both position and varyings.
 */
#define CP_CLIP_NUM_PLANES 3
#define CP_CLIP_MAX_VERTS 6   /* a triangle gains at most one vertex per plane */

static __device__ __forceinline__ float
clip_dist(const float4 *v, int plane)
{
   return plane == 0 ? v[0].z
        : plane == 1 ? v[0].w - 1e-6f
                     : v[0].w - v[0].z;
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
      bool in = true;
      for (int p = 0; p < CP_CLIP_NUM_PLANES; p++)
         in = in && clip_dist(vi, p) >= 0.0f;
      if (in)
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

   /* Ping-pong between the two buffers, one plane at a time. */
   const float4 *src = v;
   int n = 3;
   for (int p = 0; p < CP_CLIP_NUM_PLANES; p++) {
      float4 *dst = (p & 1) ? poly_b : poly_a;
      n = clip_poly(dst, src, n, slots, p);
      if (n < 3)
         return;
      src = dst;
   }

   /* Fan-triangulate the clipped polygon, which keeps the original winding. */
   for (int i = 1; i + 1 < n; i++) {
      float4 *dst = clip_emit(&args, out, slots);
      if (!dst)
         return;
      clip_copy(dst, src, slots);
      clip_copy(dst + slots, src + (size_t)i * slots, slots);
      clip_copy(dst + 2 * slots, src + (size_t)(i + 1) * slots, slots);
   }
}

/*
 * How many triangles this draw actually has.
 *
 * Clipping runs on the device and writes its output count there, so when it
 * has run the host only knows the worst case it sized the buffer for. Every
 * stage has to agree on this: reading args.num_triangles directly is reading
 * the count from before the clipper ran, and a clipped triangle's id can be up
 * to three times that.
 */
static __device__ __forceinline__ uint32_t
cp_num_triangles(const struct cp_rasterize_args *args)
{
   return args->tri_count
      ? *(const volatile uint32_t *)(uintptr_t)args->tri_count
      : args->num_triangles;
}

/*
 * How many entries of a queue are readable. The counter is an unclamped
 * atomicAdd, so on overflow it counts past the end of the allocation and the
 * entries beyond it were never written.
 */
static __device__ __forceinline__ uint32_t
cp_queue_used(uint32_t count, uint32_t capacity)
{
   return count < capacity ? count : capacity;
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

      s->ix_min = max((int)floorf(s->pt_x0), args->clip_x0);
      s->iy_min = max((int)floorf(s->pt_y0), args->clip_y0);
      s->ix_max = min((int)ceilf(s->pt_x1), args->clip_x1);
      s->iy_max = min((int)ceilf(s->pt_y1), args->clip_y1);
      return s->ix_min <= s->ix_max && s->iy_min <= s->iy_max;
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

   /*
    * The bounding box is clamped to the clip rectangle rather than to the
    * framebuffer, which is what keeps a primitive inside its viewport. An
    * empty box means the primitive is entirely outside it: say so, so that no
    * stage walks a box with a negative width — two negative sides multiply
    * into a plausible-looking area.
    */
   s->ix_min = max((int)floorf(min_x), args->clip_x0);
   s->iy_min = max((int)floorf(min_y), args->clip_y0);
   s->ix_max = min((int)ceilf(max_x), args->clip_x1);
   s->iy_max = min((int)ceilf(max_y), args->clip_y1);

   return s->ix_min <= s->ix_max && s->iy_min <= s->iy_max;
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
 *
 * ABUF — whether this is one of the A-buffer's count and fill passes — is a
 * compile-time parameter rather than a test on args->abuf_mode. This function
 * runs about 960 million times a frame on a peeled draw, and those passes are
 * their own launches, so the question can be settled by which kernel the host
 * launched instead of by a load and a compare per coverage event. Measured: a
 * runtime test costs multisampling 3.4% for branches it never takes.
 * PERFORMANCE_PLAN.md §0.1 quotes CuRast making the same choice for the same
 * reason. Both specialisations are instantiated below as separate entry
 * points, so <false> compiles to what a build without the A-buffer compiles
 * to.
 */
template <bool ABUF>
static __device__ __forceinline__ void
emit_fragment(struct cp_rasterize_args *args, uint32_t tri_id,
              int px, int py, int sample, float ndc_z)
{
#if CP_ABUF_INSTRUMENT
   /* TEMPORARY: raw coverage census. Counted first so it is one entry per
    * (primitive, pixel, sample) the rasterizer produced, before rejection,
    * before the depth test and before the visibility buffer. */
   if (args->census)
      atomicAdd((unsigned int *)(uintptr_t)args->census +
                (uint32_t)py * args->width + (uint32_t)px, 1u);
#endif

   if (cp_tri_rejected(args, tri_id, px, py))
      return;

   uint32_t plane = (uint32_t)sample * args->width * args->height;
   uint32_t at = plane + (uint32_t)py * args->width + (uint32_t)px;

   uint32_t depth_uint = float_to_sortable_uint(ndc_z * 0.5f + 0.5f);

   if (args->depth_test && args->depthbuf) {
      uint32_t prev = ((const uint32_t *)(uintptr_t)args->depthbuf)[at];
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

#if CP_ABUF_INSTRUMENT
   /* TEMPORARY: same census, restricted to what survives the depth test. */
   if (args->census_depth)
      atomicAdd((unsigned int *)(uintptr_t)args->census_depth +
                (uint32_t)py * args->width + (uint32_t)px, 1u);
#endif

   /*
    * A-buffer build. This is the depth-passing population — the same set the
    * peel loop composites, because peeling also selects among fragments that
    * got this far. Both modes return before the visibility buffer, so neither
    * launch can change what is rendered.
    */
   if (ABUF) {
      uint32_t p = (uint32_t)py * args->width + (uint32_t)px;
      if (args->abuf_mode == CP_ABUF_COUNT) {
         atomicAdd((unsigned int *)(uintptr_t)args->abuf_counts + p, 1u);
      } else {
         uint32_t slot = atomicAdd((unsigned int *)(uintptr_t)args->abuf_cursor
                                   + p, 1u);
         uint32_t n = ((const uint32_t *)(uintptr_t)args->abuf_counts)[p];
         uint32_t base = ((const uint32_t *)(uintptr_t)args->abuf_offsets)[p];
         /* Bounded by the pixel's own run as well as by the array, so that a
          * fill that disagrees with the count is reported rather than allowed
          * to write over the next pixel's fragments. */
         if (slot < n && base + slot < args->abuf_capacity)
            ((uint32_t *)(uintptr_t)args->abuf_frags)[base + slot] = tri_id;
         else
            atomicAdd((unsigned int *)(uintptr_t)args->abuf_overflow, 1u);
      }
      return;
   }

   uint64_t *visbuf = (uint64_t *)(uintptr_t)args->framebuffer;

   /*
    * Ordered blending picks by submission order rather than by depth: the
    * lowest numbered primitive this pixel has not composited yet. Keying the
    * visibility buffer on the primitive index makes the same atomicMin do it.
    */
   if (args->blend_peel) {
      const uint32_t *next = (const uint32_t *)(uintptr_t)args->peel_next;
      if (tri_id < next[(uint32_t)py * args->width + (uint32_t)px])
         return;
      atomicMin(&visbuf[at], PACK_VISBUF(tri_id, tri_id));
      /* Tells the host another pass is worth running. Racing writers all store
       * the same value, so the read first is only there to spare the traffic. */
      uint32_t *any = (uint32_t *)(uintptr_t)args->peel_any;
      if (any && !*any)
         *any = 1u;
      return;
   }

   uint32_t key = args->depth_key_invert ? ~depth_uint : depth_uint;
   atomicMin(&visbuf[at], PACK_VISBUF(key, tri_id));
}

/*
 * Every pixel whose centre falls inside the point's square, at the vertex's
 * own depth. Half-open on both axes so two points that abut do not both claim
 * the shared row, the same reason the triangle path has a fill rule.
 *
 * The bounding box is walked flattened rather than as two nested loops so that
 * a caller can hand it a lane and a stride: one thread covers a point with
 * (0, 1) and a warp covers it with (lane_id, 32). gl_PointSize is clamped at
 * CP_MAX_POINT_SIZE, which is 256, so a single point can be 65,536 pixels and
 * is worth more than one thread.
 */
template <bool ABUF>
static __device__ __forceinline__ void
rasterize_point(struct cp_rasterize_args *args, struct tri_setup *s,
                uint32_t tri_id, uint32_t lane, uint32_t stride)
{
   int bb_w = s->ix_max - s->ix_min + 1;
   int area = bb_w * (s->iy_max - s->iy_min + 1);

   for (int i = (int)lane; i < area; i += (int)stride) {
      int px = s->ix_min + (i % bb_w);
      int py = s->iy_min + (i / bb_w);

      for (int sm = 0; sm < (int)args->num_samples; sm++) {
         float ox, oy;
         cp_sample_pos(args->num_samples, sm, &ox, &oy);
         float cx = (float)px + ox, cy = (float)py + oy;
         if (cx < s->pt_x0 || cx >= s->pt_x1 ||
             cy < s->pt_y0 || cy >= s->pt_y1)
            continue;
         emit_fragment<ABUF>(args, tri_id, px, py, sm, s->ndc_z0);
      }
   }
}

/*
 * Hand lane 0's setup to the rest of the warp.
 *
 * Word by word over the struct rather than field by field: the fields that
 * matter depend on what was set up — a point carries a square where a triangle
 * carries edges — and a broadcast that names them individually has to be
 * extended every time one is added, which is how the point path came to be
 * stuck in stage 1.
 */
static __device__ __forceinline__ void
cp_broadcast_setup(struct tri_setup *s)
{
   uint32_t *w = (uint32_t *)s;
#pragma unroll
   for (int i = 0; i < (int)(sizeof(struct tri_setup) / sizeof(uint32_t)); i++)
      w[i] = __shfl_sync(0xFFFFFFFF, w[i], 0);
}

/*
 * Stage 1: 1 thread per triangle.
 * Small triangles are rasterized with incremental edge stepping.
 * Large triangles are pushed to the nontrivial queue for stage 2.
 */
template <bool ABUF>
static __device__ __forceinline__ void
cp_rasterize_stage1_body(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   uint32_t tri_id = blockIdx.x * blockDim.x + threadIdx.x;
   /* After clipping the count lives on the device, so the grid is sized for
    * the worst case and each thread bounds itself. */
   if (tri_id >= cp_num_triangles(&args))
      return;

   struct tri_setup s;
   if (!setup_triangle(&args, tri_id, &s))
      return;

   int bb_w = s.ix_max - s.ix_min + 1;
   int bb_h = s.iy_max - s.iy_min + 1;
   int bb_area = bb_w * bb_h;

   if (bb_area <= 0)
      return;

   /*
    * Too big for one thread: queue it for a warp. Points go the same way as
    * triangles here — a sprite is clamped to 256 pixels a side, which is
    * 65,536 pixels and no more affordable on one lane than a triangle of the
    * same size. Leaving them in this stage is what left particlesystem
    * rasterizing its fire one thread at a time.
    */
   if (bb_area > CP_SMALL_THRESHOLD) {
      /* A reusing pass already has this id in the queue from the first pass,
       * at the same index and behind the same counter. The classification is
       * still done — it is what decides this thread does not rasterize — but
       * the append is not. */
      if (queues.mode != CP_QUEUE_REUSE) {
         uint32_t *counter = (uint32_t *)(uintptr_t)queues.nontrivial_count;
         uint32_t idx = atomicAdd(counter, 1u);
         if (idx < CP_MAX_NONTRIVIAL) {
            uint32_t *queue = (uint32_t *)(uintptr_t)queues.nontrivial;
            queue[idx] = tri_id;
         }
      }
      return;
   }

   if (s.is_point) {
      rasterize_point<ABUF>(&args, &s, tri_id, 0, 1);
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
      for (int px = s.ix_min; px <= s.ix_max; px++) {
         for (int sm = 0; sm < (int)args.num_samples; sm++) {
            float ox, oy;
            cp_sample_pos(args.num_samples, sm, &ox, &oy);
            float cx = (float)px + ox, cy = (float)py + oy;

            float e0 = edge_function(s.sx1, s.sy1, s.sx2, s.sy2, cx, cy);
            float e1 = edge_function(s.sx2, s.sy2, s.sx0, s.sy0, cx, cy);
            float e2 = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, cx, cy);

            if (edge_inside(e0, s.e0_top_left) &&
                edge_inside(e1, s.e1_top_left) &&
                edge_inside(e2, s.e2_top_left)) {
               float w0 = e0 * s.inv_area;
               float w1 = e1 * s.inv_area;
               float w2 = 1.0f - w0 - w1;

               emit_fragment<ABUF>(&args, tri_id, px, py, sm,
                                   w0 * s.ndc_z0 + w1 * s.ndc_z1 +
                                   w2 * s.ndc_z2);
            }
         }
      }
   }
}

/*
 * Stage 2: 1 warp (32 threads) per nontrivial triangle.
 * Medium triangles are rasterized cooperatively. Huge triangles are decomposed
 * into tiles and pushed to the huge queue.
 */
template <bool ABUF>
static __device__ __forceinline__ void
cp_rasterize_stage2_body(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   uint32_t lane_id = threadIdx.x % 32;
   uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   uint32_t num_warps = (gridDim.x * blockDim.x) / 32;

   uint32_t *nt_counter = (uint32_t *)(uintptr_t)queues.nontrivial_count;
   uint32_t num_nontrivial = cp_queue_used(*nt_counter, CP_MAX_NONTRIVIAL);
   uint32_t *nt_queue = (uint32_t *)(uintptr_t)queues.nontrivial;
   uint32_t num_triangles = cp_num_triangles(&args);

   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;

   /*
    * A warp claims queue entries by striding the grid, rather than the grid
    * being one warp per entry. The grid cannot be sized to the queue: stage 1
    * fills it on the device and the host would have to synchronise to read the
    * count. Striding a fixed grid is what covers a queue of any length without
    * that round trip — and without it every entry past the grid was dropped,
    * which at 512 warps meant a mesh silently lost everything after its 512th
    * non-trivial triangle.
    */
   for (uint32_t q = warp_id; q < num_nontrivial; q += num_warps) {
      uint32_t entry = nt_queue[q];

      /*
       * A marked entry was decomposed into tiles by an earlier pass of this
       * same draw, and those tiles are still in the huge queue. Skipping it
       * here is what makes the reuse worth anything: the alternative is to
       * redo the setup on every pass purely to rediscover that the primitive
       * is too large for this stage. Nothing marks an entry unless the host
       * asked for CP_QUEUE_BUILD, so this test is inert everywhere else.
       */
      if (entry & CP_NT_HUGE)
         continue;

      uint32_t tri_id = entry;
      if (tri_id >= num_triangles)
         continue;

      /* Lane 0 does triangle setup for the whole warp. */
      struct tri_setup s;
      int valid = 0;

      if (lane_id == 0)
         valid = setup_triangle(&args, tri_id, &s) ? 1 : 0;

      valid = __shfl_sync(0xFFFFFFFF, valid, 0);
      if (!valid)
         continue;
      cp_broadcast_setup(&s);

      int bb_w = s.ix_max - s.ix_min + 1;
      int bb_h = s.iy_max - s.iy_min + 1;
      int bb_area = bb_w * bb_h;

      /*
       * Huge primitives: decompose into tiles and push to stage 3. Points come
       * this way too. A warp is 32 lanes however large the primitive, and a
       * particle sprite covering some seven thousand pixels leaves each lane
       * a couple of hundred atomicMins to issue back to back — which is
       * latency, not arithmetic, and the way to hide it is more threads.
       * Stage 3 gives the same sprite a block per tile.
       */
      if (bb_area > (s.is_point ? CP_POINT_THRESHOLD : CP_MEDIUM_THRESHOLD)) {
         /* Reaching here on a reusing pass would mean an entry that is huge
          * and unmarked, which the pass that built the queue cannot leave
          * behind; appending its tiles again to a counter that is no longer
          * reset would duplicate them, so the mode is checked rather than
          * assumed. */
         if (lane_id == 0 && queues.mode != CP_QUEUE_REUSE) {
            /* Tell the passes after this one that this entry's tiles are
             * already queued, so they can skip it before the setup above. */
            if (queues.mode == CP_QUEUE_BUILD)
               nt_queue[q] = tri_id | CP_NT_HUGE;

            int tile_x_min = s.ix_min / CP_TILE_SIZE;
            int tile_y_min = s.iy_min / CP_TILE_SIZE;
            int tile_x_max = s.ix_max / CP_TILE_SIZE;
            int tile_y_max = s.iy_max / CP_TILE_SIZE;
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
         continue;
      }

      /* A point of this size has a square rather than edges; the warp strides
       * it the same way it would a bounding box. */
      if (s.is_point) {
         rasterize_point<ABUF>(&args, &s, tri_id, lane_id, 32);
         continue;
      }

      /* Medium triangle: all 32 lanes iterate the bounding box with stride 32 */
      for (int i = (int)lane_id; i < bb_area; i += 32) {
         int px = s.ix_min + (i % bb_w);
         int py = s.iy_min + (i / bb_w);

         for (int sm = 0; sm < (int)args.num_samples; sm++) {
            float ox, oy;
            cp_sample_pos(args.num_samples, sm, &ox, &oy);
            float cx = (float)px + ox, cy = (float)py + oy;

            float e0 = edge_function(s.sx1, s.sy1, s.sx2, s.sy2, cx, cy);
            float e1 = edge_function(s.sx2, s.sy2, s.sx0, s.sy0, cx, cy);
            float e2 = edge_function(s.sx0, s.sy0, s.sx1, s.sy1, cx, cy);

            if (!edge_inside(e0, s.e0_top_left) ||
                !edge_inside(e1, s.e1_top_left) ||
                !edge_inside(e2, s.e2_top_left))
               continue;

            float w0 = e0 * s.inv_area;
            float w1 = e1 * s.inv_area;

            emit_fragment<ABUF>(&args, tri_id, px, py, sm,
                                w0 * s.ndc_z0 + w1 * s.ndc_z1 +
                                (1.0f - w0 - w1) * s.ndc_z2);
         }
      }
   }
}

/*
 * Stage 3: 1 block (64 threads) per tile-triangle pair.
 * Evaluates edge functions at tile corners for trivial accept/reject,
 * then rasterizes the tile with per-pixel tests only where needed.
 */
template <bool ABUF>
static __device__ __forceinline__ void
cp_rasterize_stage3_body(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   uint32_t *huge_counter = (uint32_t *)(uintptr_t)queues.huge_count;
   uint32_t num_tiles = cp_queue_used(*huge_counter, CP_MAX_HUGE_TILES);
   uint32_t num_triangles = cp_num_triangles(&args);

   struct cp_tile_pair *huge_queue =
      (struct cp_tile_pair *)(uintptr_t)queues.huge_tiles;

   /*
    * The setup, shared across the block. The whole struct rather than the
    * fields a triangle happens to need, for the reason cp_broadcast_setup()
    * gives: a point carries a square where a triangle carries edges, and a
    * copy that names fields has to be extended for each new kind.
    */
   __shared__ struct tri_setup sh_s;
   __shared__ int sh_valid;

   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;

   /*
    * A block strides the tile queue for the same reason stage 2 strides its
    * own: the length is only known on the device. Nothing below may return
    * early, because every thread of the block has to reach the barriers that
    * guard the shared setup — a lane leaving the loop while its neighbours
    * wait at __syncthreads() is what turns a dropped tile into a hang.
    */
   for (uint32_t tile_idx = blockIdx.x; tile_idx < num_tiles;
        tile_idx += gridDim.x) {
      uint32_t tri_id = huge_queue[tile_idx].tri_id;
      int tile_x = (int)huge_queue[tile_idx].tile_x;
      int tile_y = (int)huge_queue[tile_idx].tile_y;

      /* The previous iteration's readers must be done before it is rewritten. */
      __syncthreads();

      if (threadIdx.x == 0) {
         struct tri_setup s;
         sh_valid = 0;
         if (tri_id < num_triangles && setup_triangle(&args, tri_id, &s)) {
            sh_s = s;

            /* A point's square is tested per pixel below; it has no edges to
             * reject a tile with, and its bounding box already selected the
             * tiles it was filed under. */
            if (s.is_point) {
               sh_valid = 1;
            } else {
            /*
             * Trivial reject: an edge that is outside at all four tile corners
             * is outside everywhere in the tile, because the edge function is
             * linear.
             *
             * There is no matching trivial accept any more. The mirror test —
             * every corner inside, so skip that edge per pixel — cannot be
             * made to respect the fill rule: a corner sitting exactly on an
             * exclusive edge reads as inside, and the tile would then claim a
             * row of pixels that belongs to its neighbour. Restoring it needs
             * corner values that bound the per-pixel ones, which the exact
             * subpixel form noted in stage 1 would give.
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
      }
      __syncthreads();

      if (!sh_valid)
         continue;

#if CP_TILE_BOUND
      /*
       * Only the part of the tile the primitive's bounding box reaches.
       *
       * The tiles were enumerated from that box, so the last tile of each row
       * and column is partly covered — and a sprite two tiles across covers a
       * quarter of each of its four tiles on average. The box is a superset of
       * coverage by construction, so the pixels this skips are exactly ones
       * the per-pixel test below would have rejected; that test is unchanged
       * and remains the coverage decision.
       *
       * Flattened rather than a column each: bounding the columns alone saves
       * nothing, since the 64 threads run in parallel and an idle column costs
       * no time, and the block's duration is set by the rows each active
       * thread walks. Striding w*h positions instead is what stops threads
       * idling — the same shape stage 2's medium path uses.
       *
       * The order fragments are emitted in changes with it. That is safe: the
       * visibility buffer resolves by atomicMin, and a minimum does not depend
       * on the order its inputs arrive in.
       *
       * No clip test here, unlike the full-tile walk below, and for the reason
       * the other two stages do not have one either — setup_triangle() clamps
       * the box to the clip rectangle, so [x0, x1] x [y0, y1] is inside it by
       * construction. It was the tile that ran past the clip rectangle, and
       * the tile is no longer what is being walked.
       */
      int x0 = max(tile_x, sh_s.ix_min);
      int y0 = max(tile_y, sh_s.iy_min);
      int x1 = min(tile_x + CP_TILE_SIZE - 1, sh_s.ix_max);
      int y1 = min(tile_y + CP_TILE_SIZE - 1, sh_s.iy_max);
      int bw = x1 - x0 + 1;
      int bh = y1 - y0 + 1;
      if (bw <= 0 || bh <= 0)
         continue;

      for (int i = (int)threadIdx.x; i < bw * bh; i += (int)blockDim.x) {
         int px = x0 + (i % bw);
         int py = y0 + (i / bw);
         {
#else
      /*
       * A tile is enumerated from the bounding box but covers whole tiles, so
       * its edges run past it — the clip rectangle has to be applied here
       * rather than inherited from the clamped box the other two stages walk.
       */
      int col = (int)threadIdx.x;
      int px = tile_x + col;

      if (col < CP_TILE_SIZE && px >= args.clip_x0 && px <= args.clip_x1) {
         for (int row = 0; row < CP_TILE_SIZE; row++) {
            int py = tile_y + row;
            if (py > args.clip_y1)
               break;
            if (py < args.clip_y0)
               continue;
#endif

            for (int sm = 0; sm < (int)args.num_samples; sm++) {
               float ox, oy;
               cp_sample_pos(args.num_samples, sm, &ox, &oy);
               float sx = (float)px + ox, sy = (float)py + oy;

               if (sh_s.is_point) {
                  if (sx >= sh_s.pt_x0 && sx < sh_s.pt_x1 &&
                      sy >= sh_s.pt_y0 && sy < sh_s.pt_y1)
                     emit_fragment<ABUF>(&args, tri_id, px, py, sm, sh_s.ndc_z0);
                  continue;
               }

               float e0 = edge_function(sh_s.sx1, sh_s.sy1, sh_s.sx2, sh_s.sy2, sx, sy);
               float e1 = edge_function(sh_s.sx2, sh_s.sy2, sh_s.sx0, sh_s.sy0, sx, sy);
               float e2 = edge_function(sh_s.sx0, sh_s.sy0, sh_s.sx1, sh_s.sy1, sx, sy);

               if (edge_inside(e0, sh_s.e0_top_left) &&
                   edge_inside(e1, sh_s.e1_top_left) &&
                   edge_inside(e2, sh_s.e2_top_left)) {
                  float w0 = e0 * sh_s.inv_area;
                  float w1 = e1 * sh_s.inv_area;

                  emit_fragment<ABUF>(&args, tri_id, px, py, sm,
                                      w0 * sh_s.ndc_z0 + w1 * sh_s.ndc_z1 +
                                      (1.0f - w0 - w1) * sh_s.ndc_z2);
               }
            }
         }
      }
   }
}

/*
 * The two specialisations of each stage, as separate entry points.
 *
 * The rendering ones are what every draw launches and compile to code with no
 * mention of the A-buffer in it at all; the `_abuf` ones are launched only by
 * the count and fill passes, which never reach the visibility buffer. Naming
 * them apart is what lets the choice be made by the host once per pass rather
 * than by the device once per coverage event.
 */
extern "C" __global__ void
cp_rasterize_stage1(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   cp_rasterize_stage1_body<false>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage2(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   cp_rasterize_stage2_body<false>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage3(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   cp_rasterize_stage3_body<false>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage1_abuf(struct cp_rasterize_args args,
                         struct cp_rast_queues queues)
{
   cp_rasterize_stage1_body<true>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage2_abuf(struct cp_rasterize_args args,
                         struct cp_rast_queues queues)
{
   cp_rasterize_stage2_body<true>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage3_abuf(struct cp_rasterize_args args,
                         struct cp_rast_queues queues)
{
   cp_rasterize_stage3_body<true>(args, queues);
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
 * ---------------------------------------------------------------------------
 * A-buffer support kernels
 * ---------------------------------------------------------------------------
 *
 * None of these runs for a draw that is not taking the A-buffer path, so they
 * cost a compile rather than a branch and are built unconditionally. Between
 * them they turn the per-pixel fragment counts the rasterizer produced into a
 * per-pixel run of primitive ids, sorted ascending, and merge those into the
 * quad stream the fragment stage consumes.
 */

/*
 * One block's exclusive prefix sum, Hillis-Steele in shared memory, with the
 * block's total left in `sums` for the level above. Double-buffered so a step
 * reads one buffer and writes the other, which is what removes the second
 * barrier per step.
 */
extern "C" __global__ void
cp_abuf_scan_block(const uint32_t *in, uint32_t *out, uint32_t *sums,
                   uint32_t n)
{
   __shared__ uint32_t buf[2][CP_ABUF_SCAN_BLOCK];
   uint32_t tid = threadIdx.x;
   uint32_t i = blockIdx.x * CP_ABUF_SCAN_BLOCK + tid;
   uint32_t v = i < n ? in[i] : 0u;

   int pin = 0;
   buf[pin][tid] = v;
   __syncthreads();
   for (uint32_t off = 1; off < CP_ABUF_SCAN_BLOCK; off <<= 1) {
      uint32_t x = buf[pin][tid];
      if (tid >= off)
         x += buf[pin][tid - off];
      pin ^= 1;
      buf[pin][tid] = x;
      __syncthreads();
   }

   uint32_t incl = buf[pin][tid];
   if (i < n)
      out[i] = incl - v;
   if (tid == CP_ABUF_SCAN_BLOCK - 1 && sums)
      sums[blockIdx.x] = incl;
}

/* Add each block's scanned base back into its elements. Must be launched with
 * CP_ABUF_SCAN_BLOCK threads, which is what makes blockIdx the block index the
 * level above scanned. */
extern "C" __global__ void
cp_abuf_scan_add(uint32_t *data, const uint32_t *sums, uint32_t n)
{
   uint32_t i = blockIdx.x * CP_ABUF_SCAN_BLOCK + threadIdx.x;
   if (i < n)
      data[i] += sums[blockIdx.x];
}

/*
 * Pixels with at least `min_count` fragments. Only 4.6% of the framebuffer is
 * covered at all, so this turns a grid over every pixel into a grid over the
 * ~40,000 that have anything in them.
 *
 * The sort asks for 2, since a run of 0 or 1 is already sorted. The composite
 * asks for 1, since a run of 1 still has to be blended.
 */
extern "C" __global__ void
cp_abuf_worklist(const uint32_t *counts, uint32_t n, uint32_t min_count,
                 uint32_t *list, uint32_t *list_count)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n && counts[i] >= min_count) {
      uint32_t at = atomicAdd(list_count, 1u);
      list[at] = i;
   }
}

/*
 * Cut every pixel's run down to what the fragment array can hold.
 *
 * The host used to drain after the scan, see that the count was larger than
 * the array, and send the draw to the peel loop before anything indexed it.
 * It no longer looks, so the guard has to be here: the offsets are a prefix
 * sum of the counts, so a total past the end of the array leaves later pixels
 * with runs that start or end outside it — and the sort and the merge index
 * `frags + offsets[p]` with no bound of their own.
 *
 * What is clamped is reported as overflow, which is the same counter the fill
 * bumps and the same one the eligibility check reads, so a draw this touches
 * at all is a draw the peel loop renders. It is the truncation being made
 * harmless, not accepted.
 *
 * The common case is one cached load and a return: the arrays are sized with
 * headroom and no draw on the traced workloads reaches this at all.
 */
extern "C" __global__ void
cp_abuf_clamp_runs(uint32_t *counts, const uint32_t *offsets,
                   const uint32_t *total, uint32_t n, uint32_t capacity,
                   uint32_t *overflow)
{
   if (*total <= capacity)
      return;

   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += stride) {
      uint32_t c = counts[p];
      if (!c)
         continue;
      uint32_t off = offsets[p];
      uint32_t room = off < capacity ? capacity - off : 0;
      if (c > room) {
         counts[p] = room;
         atomicAdd(overflow, c - room);
      }
   }
}

/*
 * Sort each pixel's run ascending: one block per pixel, grid-strided over the
 * worklist. A run is padded up to a power of two with 0xFFFFFFFF and sorted
 * bitonically in shared memory, so the padding lands past the end and only the
 * first `n` entries are written back. Runs longer than CP_ABUF_SORT_MAX get an
 * insertion sort from thread 0 instead — correct, slow, and not expected to
 * happen at a measured maximum depth of 411.
 */
extern "C" __global__ void
cp_abuf_sort(uint32_t *frags, const uint32_t *offsets, const uint32_t *counts,
             const uint32_t *list, const uint32_t *list_count,
             uint32_t *long_runs)
{
   __shared__ uint32_t key[CP_ABUF_SORT_MAX];
   uint32_t work = *list_count;

   for (uint32_t wi = blockIdx.x; wi < work; wi += gridDim.x) {
      uint32_t p = list[wi];
      uint32_t n = counts[p];
      uint32_t *run = frags + offsets[p];

      if (n > CP_ABUF_SORT_MAX) {
         if (threadIdx.x == 0) {
            atomicAdd(long_runs, 1u);
            for (uint32_t i = 1; i < n; i++) {
               uint32_t v = run[i];
               int32_t j = (int32_t)i - 1;
               while (j >= 0 && run[j] > v) {
                  run[j + 1] = run[j];
                  j--;
               }
               run[j + 1] = v;
            }
         }
         __syncthreads();
         continue;
      }

      uint32_t m = 1;
      while (m < n)
         m <<= 1;

      for (uint32_t i = threadIdx.x; i < m; i += blockDim.x)
         key[i] = i < n ? run[i] : 0xFFFFFFFFu;
      __syncthreads();

      for (uint32_t k = 2; k <= m; k <<= 1) {
         for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            /* Each pair is touched once, by the thread holding the lower
             * index, so a strided loop needs no further exclusion. */
            for (uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
               uint32_t ixj = i ^ j;
               if (ixj > i) {
                  bool up = (i & k) == 0;
                  uint32_t a = key[i], b = key[ixj];
                  if ((a > b) == up) {
                     key[i] = b;
                     key[ixj] = a;
                  }
               }
            }
            __syncthreads();
         }
      }

      for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
         run[i] = key[i];
      __syncthreads();
   }
}

#if CP_ABUF_INSTRUMENT
/*
 * What the peel loop selected at every pixel on this pass. VISBUF_TRIID of an
 * empty entry is 0, which is a valid primitive, so the empty case is written
 * out explicitly.
 *
 * Verification only (CUDAPIPE_ABUFFER_VERIFY): the A-buffer path does not run
 * the peel loop, so there is nothing for this to log unless the two are being
 * compared.
 */
extern "C" __global__ void
cp_abuf_peel_log(const uint64_t *visbuf, uint32_t *log, uint32_t layers,
                 uint32_t pass, uint32_t n)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= n || pass >= layers)
      return;
   uint64_t entry = visbuf[i];
   log[(size_t)i * layers + pass] =
      entry == VISBUF_EMPTY ? 0xFFFFFFFFu : VISBUF_TRIID(entry);
}

/* The same, for a short list of pixels and to full depth, so that the deep
 * tail is compared rather than only its first CP_ABUF_LOG_LAYERS entries. */
extern "C" __global__ void
cp_abuf_peel_log_list(const uint64_t *visbuf, const uint32_t *list,
                      uint32_t *log, uint32_t layers, uint32_t pass,
                      uint32_t n)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= n || pass >= layers)
      return;
   uint64_t entry = visbuf[list[i]];
   log[(size_t)i * layers + pass] =
      entry == VISBUF_EMPTY ? 0xFFFFFFFFu : VISBUF_TRIID(entry);
}
#endif /* CP_ABUF_INSTRUMENT */

/*
 * ---------------------------------------------------------------------------
 * Per-pixel lists -> quad stream
 * ---------------------------------------------------------------------------
 *
 * The fragment shader is fed quads, not pixels: four lanes in a 2x2 block, so
 * the sampler can take a screen-space derivative by shuffling between them. A
 * derivative is only meaningful across one surface, so a quad has to belong to
 * a single primitive — which is why cp_fs_interpolate emits one quad per
 * distinct primitive it finds in a block and shades the pixels that primitive
 * missed as helper lanes.
 *
 * These build the same thing from the A-buffer, for every layer at once. Each
 * block's four pixels hold a sorted list of the primitives covering them, so
 * the distinct primitives of the block are their 4-way merge, and the pixels
 * one primitive covers are the lists it was found in — a 4-bit mask, which is
 * exactly what the interpolator writes into `coverage`.
 */

/*
 * Blocks worth visiting. 4.6% of the framebuffer is covered on
 * particlesystem, which is order 8,000-10,000 blocks against 230,400, and a
 * grid over all of them would spend its time finding nothing.
 */
extern "C" __global__ void
cp_abuf_block_worklist(const uint32_t *counts, uint32_t width, uint32_t height,
                       uint32_t quad_width, uint32_t nblocks, uint32_t *list,
                       uint32_t *list_count)
{
   uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
   if (b >= nblocks)
      return;

   uint32_t qx = (b % quad_width) * 2;
   uint32_t qy = (b / quad_width) * 2;
   uint32_t any = 0;
   for (int i = 0; i < 4; i++) {
      uint32_t x = qx + (i & 1), y = qy + (i >> 1);
      /* A block on the edge of an odd-sized framebuffer has corners outside
       * it. cp_fs_interpolate ignores those when it looks for triangles — it
       * only clamps their addresses so the quad stays whole — so they cannot
       * contribute coverage here either. */
      if (x < width && y < height)
         any |= counts[(size_t)y * width + x];
   }

   if (any) {
      uint32_t at = atomicAdd(list_count, 1u);
      list[at] = b;
   }
}

/*
 * One block's merge, shared by the counting and the filling pass so that the
 * two cannot disagree about what a block contains.
 *
 * Returns the number of distinct primitives. With `out_prim` non-null it also
 * writes them, ascending, from `out_base`, each with the mask of the block's
 * four pixels whose list held it.
 *
 * `out_shade_slot`, when given, is filled the other way round: for every
 * A-buffer slot this merge consumed, which of the quad stream's shading slots
 * will hold that fragment's colour. The composite walks a pixel's run and
 * needs its shaded colours in that order, and the merge is the one place that
 * knows both indices at once — so recording it here is what spares the
 * composite a binary search per layer.
 */
static __device__ __forceinline__ uint32_t
cp_abuf_merge_block(const uint32_t *frags, const uint32_t *offsets,
                    const uint32_t *counts, uint32_t width, uint32_t height,
                    uint32_t quad_width, uint32_t b,
                    uint32_t *out_prim, unsigned char *out_mask,
                    uint32_t *out_peel_mask, uint32_t *out_block,
                    uint32_t *out_shade_slot,
                    uint32_t out_base, uint32_t capacity, uint32_t *overflow)
{
   const uint32_t *run[4];
   uint32_t n[4], cur[4], head[4], off[4];

   uint32_t qx = (b % quad_width) * 2;
   uint32_t qy = (b / quad_width) * 2;
   for (int i = 0; i < 4; i++) {
      uint32_t x = qx + (i & 1), y = qy + (i >> 1);
      run[i] = NULL;
      n[i] = 0;
      cur[i] = 0;
      head[i] = 0;
      off[i] = 0;
      if (x < width && y < height) {
         uint32_t p = (uint32_t)y * width + x;
         off[i] = offsets[p];
         run[i] = frags + off[i];
         n[i] = counts[p];
         if (n[i])
            head[i] = run[i][0];
      }
   }

   uint32_t emitted = 0;
   for (;;) {
      /* The smallest primitive still unconsumed in any of the four lists.
       * Heads are kept in registers: reloading all four every time round is
       * four global loads per quad produced, where only the lists that
       * advanced can have changed. Tracked with a flag rather than a sentinel
       * value, because every 32-bit primitive id is a legal one. */
      bool have = false;
      uint32_t best = 0;
      for (int i = 0; i < 4; i++) {
         if (cur[i] < n[i] && (!have || head[i] < best)) {
            best = head[i];
            have = true;
         }
      }
      if (!have)
         break;

      /* Consume it from every list holding it. The step-2 check established
       * that a run is strictly increasing, so `while` can only ever run once —
       * it is here so that a run that ever stops being would produce one quad
       * rather than an endless loop. */
      uint32_t mask = 0;
      uint32_t took[4] = { 0, 0, 0, 0 };
      for (int i = 0; i < 4; i++) {
         while (cur[i] < n[i] && head[i] == best) {
            mask |= 1u << i;
            /* Which entry of this pixel's run was consumed, so the slot it
             * occupies can be pointed at the shading slot below. */
            took[i] = cur[i] + 1;
            if (++cur[i] < n[i])
               head[i] = run[i][cur[i]];
         }
      }

      if (out_prim) {
         uint32_t at = out_base + emitted;
         if (at < capacity) {
            out_prim[at] = best;
            out_mask[at] = (unsigned char)mask;
            /*
             * cp_abuf_interpolate gives quad q the four shading slots 4q..4q+3,
             * lane i being the block's pixel i. So the fragment this quad
             * carries for pixel i is shaded at 4*at + i, and it lives in
             * A-buffer slot off[i] + (the entry just consumed).
             */
            if (out_shade_slot) {
               for (int i = 0; i < 4; i++) {
                  if (mask & (1u << i))
                     out_shade_slot[off[i] + took[i] - 1] = at * 4u + (uint32_t)i;
               }
            }
            /* Which block the quad came from. The array is grouped by block
             * and the offsets say where each block's group starts, but a
             * consumer walking quads rather than blocks would have to invert
             * that; one word per quad is cheaper than a search per quad. */
            if (out_block)
               out_block[at] = b;
            /* Cleared here rather than by a memset over the whole capacity:
             * the peel loop ORs into it from the next launch onwards. */
            if (out_peel_mask)
               out_peel_mask[at] = 0;
         } else if (overflow) {
            atomicAdd(overflow, 1u);
         }
      }
      emitted++;
   }

   return emitted;
}

/*
 * Quads per block, into a dense array for the prefix sum. Grid-strided over
 * the worklist, whose length lives on the device — the host never learns it,
 * so nothing here costs a drain.
 */
extern "C" __global__ void
cp_abuf_quad_count(const uint32_t *frags, const uint32_t *offsets,
                   const uint32_t *counts, uint32_t width, uint32_t height,
                   uint32_t quad_width, const uint32_t *list,
                   const uint32_t *list_count, uint32_t *blk_counts)
{
   uint32_t work = *list_count;
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t wi = blockIdx.x * blockDim.x + threadIdx.x; wi < work;
        wi += stride) {
      uint32_t b = list[wi];
      blk_counts[b] = cp_abuf_merge_block(frags, offsets, counts, width, height,
                                          quad_width, b, NULL, NULL, NULL, NULL,
                                          NULL, 0, 0, NULL);
   }
}

/*
 * Clear the merge's slot map over the run this draw actually uses.
 *
 * The length of that run is the count pass's total, and since the drain
 * between the count and the fill was removed the host does not have it — so
 * the bound is read on the device instead. Clamped to the capacity because a
 * count larger than the array is exactly the case the fill reports and the
 * eligibility check refuses, and this must not write past the array while that
 * is being found out.
 */
extern "C" __global__ void
cp_abuf_clear_slots(uint32_t *slots, const uint32_t *total, uint32_t capacity)
{
   uint32_t n = *total;
   if (n > capacity)
      n = capacity;
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
      slots[i] = 0xFFFFFFFFu;
}

/* The same merge again, this time writing the quads at the scanned offsets. */
extern "C" __global__ void
cp_abuf_quad_fill(const uint32_t *frags, const uint32_t *offsets,
                  const uint32_t *counts, uint32_t width, uint32_t height,
                  uint32_t quad_width, const uint32_t *list,
                  const uint32_t *list_count, const uint32_t *blk_offsets,
                  uint32_t *quad_prim, unsigned char *quad_mask,
                  uint32_t *quad_peel_mask, uint32_t *quad_block,
                  uint32_t *shade_slot,
                  uint32_t capacity, uint32_t *overflow)
{
   uint32_t work = *list_count;
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t wi = blockIdx.x * blockDim.x + threadIdx.x; wi < work;
        wi += stride) {
      uint32_t b = list[wi];
      cp_abuf_merge_block(frags, offsets, counts, width, height, quad_width, b,
                          quad_prim, quad_mask, quad_peel_mask, quad_block,
                          shade_slot, blk_offsets[b], capacity, overflow);
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
