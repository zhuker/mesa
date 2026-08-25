/*
 * cudavk 3-stage adaptive triangle rasterizer — visibility buffer approach.
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

/*
 * Clipping against the planes that make the perspective divide meaningful,
 * one thread per input triangle.
 *
 * Gallium presents clip coordinates with the OpenGL depth convention,
 * -w <= x,y,z <= w. The six view-volume planes plus a small positive-W
 * guard matter here:
 *
 *   z >= -w     one of the depth planes. Under the conventional projection it
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
 *   -w <= x,y <= w
 *               must be clipped in homogeneous space. Merely clamping the
 *               screen-space bounding box leaves enormous post-divide edge
 *               coefficients, which lose enough precision to drop parts of
 *               large triangles near the side of the view.
 *
 * No one plane is enough: samples exist that only one of them fixes.
 *
 * Interpolation happens in clip space, before the divide, which is what makes
 * the split exact for both position and varyings.
 */
#define CP_CLIP_NUM_PLANES 7
#define CP_CLIP_MAX_VERTS 10

static __device__ __forceinline__ float
clip_dist(const float4 *v, int plane)
{
   return plane == 0 ? v[0].z + v[0].w
        : plane == 1 ? v[0].w - 1e-6f
        : plane == 2 ? v[0].w - v[0].z
        : plane == 3 ? v[0].x + v[0].w
        : plane == 4 ? v[0].w - v[0].x
        : plane == 5 ? v[0].y + v[0].w
                     : v[0].w - v[0].y;
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

static __device__ __forceinline__ bool
clip_alloc(struct cp_clip_args *args, uint32_t tri, uint32_t k,
           uint32_t *id_out)
{
   /* Stable mode owns its fixed per-triangle range outright, so there is no
    * counter to contend on and no order to lose; see cp_clip_args::stable. */
   uint32_t o = args->stable
      ? tri * CP_CLIP_MAX_OUT + k
      : atomicAdd((unsigned int *)(uintptr_t)args->out_count, 1u);
   *id_out = o;
   if (o >= args->max_triangles)
      return false;

   /* In stable mode o is the primitive's ordered fixed-slot ID. Append that
    * ID to a compact worklist instead of making the rasterizer visit holes. */
   if (args->stable && args->active_ids) {
      uint32_t at = atomicAdd((unsigned int *)(uintptr_t)args->out_count, 1u);
      if (at < args->max_triangles)
         ((uint32_t *)(uintptr_t)args->active_ids)[at] = o;
   }
   return true;
}

static __device__ __forceinline__ float4 *
clip_dst(float4 *out, uint32_t slots, uint32_t id)
{
   return out + (size_t)id * 3 * slots;
}

static __device__ __forceinline__ void
clip_publish_ref(struct cp_clip_args *args, uint32_t id, const float4 *base)
{
   if (args->prim_refs)
      ((uint64_t *)(uintptr_t)args->prim_refs)[id] = (uint64_t)(uintptr_t)base;
}

/*
 * Retire this input triangle's unused slots.
 *
 * setup_triangle() rejects a zero area, so three identical vertices are a
 * primitive the rasterizer never walks. w = 1 rather than 0, so that nothing
 * on the way there divides by zero and produces a NaN the area test would let
 * through.
 */
static __device__ __forceinline__ void
clip_retire(struct cp_clip_args *args, float4 *out, uint32_t slots,
            uint32_t tri, uint32_t k)
{
   for (uint32_t i = k; i < CP_CLIP_MAX_OUT; i++) {
      uint32_t o = tri * CP_CLIP_MAX_OUT + i;
      if (o >= args->max_triangles)
         return;
      if (args->prim_refs) {
         /* Stable raster without active_ids walks every fixed ID. Null makes
          * the unused slot retire without writing three degenerate vertices. */
         ((uint64_t *)(uintptr_t)args->prim_refs)[o] = 0;
      } else {
         float4 *dst = out + (size_t)o * 3 * slots;
         dst[0] = dst[slots] = dst[2 * (size_t)slots] =
            make_float4(0.0f, 0.0f, 0.0f, 1.0f);
      }
   }
}

/*
 * Clip one input triangle, with every side effect the standalone kernel has:
 * the emitted output triangles, the compact counter, the active-ID worklist
 * and the retired slots. The caller may also collect the emitted primitive
 * IDs — that is the only addition, and it is what lets the fused kernel
 * rasterize what it just clipped without re-deriving the ID assignment.
 * Returns how many triangles were emitted (and recorded in ids[]).
 */
static __device__ int
cp_clip_one(struct cp_clip_args *args, uint32_t tri, uint32_t *ids)
{
   uint32_t slots = args->num_slots;
   if (slots > CP_MAX_CLIP_SLOTS)
      slots = CP_MAX_CLIP_SLOTS;

   const float4 *in = (const float4 *)(uintptr_t)args->vs_out;
   float4 *out = (float4 *)(uintptr_t)args->out;
   const float4 *v = in + (size_t)tri * 3 * slots;
   uint32_t id;
   int emitted = 0;

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
      if (clip_alloc(args, tri, 0, &id)) {
         if (args->prim_refs) {
            /* Immutable VS output stays live through every consumer of the
             * clipped stream, so acceptance is pointer publication only. */
            clip_publish_ref(args, id, v);
         } else {
            float4 *dst = clip_dst(out, slots, id);
            clip_copy(dst, v, 3 * slots);
         }
         if (ids)
            ids[emitted] = id;
         emitted++;
      }
      if (args->stable && !args->active_ids)
         clip_retire(args, out, slots, tri, 1);
      return emitted;
   }

   float4 poly_a[CP_CLIP_MAX_VERTS * CP_MAX_CLIP_SLOTS];
   float4 poly_b[CP_CLIP_MAX_VERTS * CP_MAX_CLIP_SLOTS];

   /* Ping-pong between the two buffers, one plane at a time. */
   const float4 *src = v;
   int n = 3;
   for (int p = 0; p < CP_CLIP_NUM_PLANES; p++) {
      float4 *dst = (p & 1) ? poly_b : poly_a;
      n = clip_poly(dst, src, n, slots, p);
      if (n < 3) {
         if (args->stable && !args->active_ids)
            clip_retire(args, out, slots, tri, 0);
         return 0;
      }
      src = dst;
   }

   /* Fan-triangulate the clipped polygon, which keeps the original winding. */
   int k = 0;
   for (int i = 1; i + 1 < n; i++) {
      if (!clip_alloc(args, tri, (uint32_t)k, &id))
         break;
      k++;
      float4 *dst = clip_dst(out, slots, id);
      clip_copy(dst, src, slots);
      clip_copy(dst + slots, src + (size_t)i * slots, slots);
      clip_copy(dst + 2 * slots, src + (size_t)(i + 1) * slots, slots);
      /* Publish only after all three vertices are complete. A fused caller's
       * same thread consumes this in program order; later raster/interpolation
       * launches are ordered by the kernel boundary. */
      clip_publish_ref(args, id, dst);
      if (ids)
         ids[emitted] = id;
      emitted++;
   }
   if (args->stable && !args->active_ids)
      clip_retire(args, out, slots, tri, (uint32_t)k);
   return emitted;
}

extern "C" __global__ void
cp_clip_triangles(struct cp_clip_args args)
{
   uint32_t tri = blockIdx.x * blockDim.x + threadIdx.x;
   if (tri >= args.num_triangles)
      return;
   cp_clip_one(&args, tri, NULL);
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

/* Map a compact work item back to the fixed primitive ID stable clipping
 * assigned it.  Queues, visibility and fragment records all carry that fixed
 * ID; only kernels which enumerate the initial work use this mapping. */
static __device__ __forceinline__ uint32_t
cp_triangle_id(const struct cp_rasterize_args *args, uint32_t work)
{
   return args->active_ids
      ? ((const uint32_t *)(uintptr_t)args->active_ids)[work]
      : work;
}

static __device__ __forceinline__ bool
cp_triangle_id_valid(const struct cp_rasterize_args *args, uint32_t tri)
{
   return tri < (args->active_ids ? args->num_triangles
                                  : cp_num_triangles(args));
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
               struct cp_tri_setup *s)
{
   uint32_t pos_stride = args->num_varyings + 1;
   const float4 *positions = (const float4 *)cp_primitive_base(
      args->prim_refs, args->positions, tri_id, pos_stride * 16u);
   if (!positions)
      return false;

   /*
    * The clip rectangle this primitive is bounded by: the batch-wide one, or
    * its own draw's when the batch carries per-draw scissors. The rects are
    * precomputed as the full intersection on the host, so no second clamp.
    */
   int clip_x0 = args->clip_x0, clip_y0 = args->clip_y0;
   int clip_x1 = args->clip_x1, clip_y1 = args->clip_y1;
   if (args->clip_rects) {
      const struct cp_draw_slice *sl =
         (const struct cp_draw_slice *)(uintptr_t)args->rect_draw_slices;
      uint32_t vert = (tri_id >> args->rect_prim_shift) * 3u;
      uint32_t lo = 0, hi = args->num_rect_slices - 1;
      while (lo < hi) {
         uint32_t mid = (lo + hi + 1) >> 1;
         if (sl[mid].vert_begin <= vert)
            lo = mid;
         else
            hi = mid - 1;
      }
      const int4 r = ((const int4 *)(uintptr_t)args->clip_rects)[lo];
      clip_x0 = r.x; clip_y0 = r.y; clip_x1 = r.z; clip_y1 = r.w;
   }

   float4 v0 = positions[0 * pos_stride];
   float4 v1 = positions[1 * pos_stride];
   float4 v2 = positions[2 * pos_stride];

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
         size = positions[args->psiz_slot].x;

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

      s->ix_min = max((int)floorf(s->pt_x0), clip_x0);
      s->iy_min = max((int)floorf(s->pt_y0), clip_y0);
      s->ix_max = min((int)ceilf(s->pt_x1), clip_x1);
      s->iy_max = min((int)ceilf(s->pt_y1), clip_y1);
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
   s->ix_min = max((int)floorf(min_x), clip_x0);
   s->iy_min = max((int)floorf(min_y), clip_y0);
   s->ix_max = min((int)ceilf(max_x), clip_x1);
   s->iy_max = min((int)ceilf(max_y), clip_y1);

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
         if (args->abuf_recs) {
            /* Warp-aggregated append: the cursor is one word the whole device
             * hammers, so the warp's emitting lanes reserve their slots with
             * a single atomic. The blocks are 1-D multiples of 32, and the
             * code from the mask capture to the shuffle is straight-line, so
             * the opportunistic-warp pattern is sound here. Order across
             * warps is arbitrary — the per-pixel sort erases it, the same
             * way it erases today's fill-cursor races. */
            unsigned mask = __activemask();
            unsigned lane = threadIdx.x & 31u;
            unsigned leader = __ffs(mask) - 1u;
            uint32_t base = 0;
            if (lane == leader)
               base = atomicAdd((unsigned int *)(uintptr_t)args->abuf_rec_cursor,
                                (unsigned)__popc(mask));
            base = __shfl_sync(mask, base, leader);
            uint32_t slot = base + __popc(mask & ((1u << lane) - 1u));
            if (slot < args->abuf_capacity)
               ((uint64_t *)(uintptr_t)args->abuf_recs)[slot] =
                  ((uint64_t)p << 32) |
                  (uint64_t)(args->abuf_prim_base + tri_id);
            else
               atomicAdd((unsigned int *)(uintptr_t)args->abuf_overflow, 1u);
         }
      } else {
         uint32_t slot = atomicAdd((unsigned int *)(uintptr_t)args->abuf_cursor
                                   + p, 1u);
         uint32_t n = ((const uint32_t *)(uintptr_t)args->abuf_counts)[p];
         uint32_t base = ((const uint32_t *)(uintptr_t)args->abuf_offsets)[p];
         /* Bounded by the pixel's own run as well as by the array, so that a
          * fill that disagrees with the count is reported rather than allowed
          * to write over the next pixel's fragments. */
         if (slot < n && base + slot < args->abuf_capacity)
            ((uint32_t *)(uintptr_t)args->abuf_frags)[base + slot] =
               args->abuf_prim_base + tri_id;
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
   atomicMin(&visbuf[at],
             PACK_VISBUF(key, args->abuf_prim_base + tri_id));
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
rasterize_point(struct cp_rasterize_args *args, struct cp_tri_setup *s,
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
cp_broadcast_setup(struct cp_tri_setup *s)
{
   uint32_t *w = (uint32_t *)s;
#pragma unroll
   for (int i = 0; i < (int)(sizeof(struct cp_tri_setup) / sizeof(uint32_t)); i++)
      w[i] = __shfl_sync(0xFFFFFFFF, w[i], 0);
}

/*
 * Classify one primitive, and rasterize it on the spot when it is small.
 *
 * This is stage 1's whole job, factored per primitive so the fused kernel can
 * run it on the triangles it just clipped. Nontrivial primitives are appended
 * to the unchanged stage-2 queue when requested.
 */
template <bool ABUF>
static __device__ __forceinline__ void
cp_rast_small_or_defer(struct cp_rasterize_args *args,
                       struct cp_rast_queues *queues, uint32_t tri_id,
                       bool append)
{

   struct cp_tri_setup s;
   if (!setup_triangle(args, tri_id, &s))
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
      if (append) {
         uint32_t *counter = (uint32_t *)(uintptr_t)queues->nontrivial_count;
         uint32_t idx = atomicAdd(counter, 1u);
         if (idx >= CP_MAX_NONTRIVIAL)
            return;
         uint32_t *queue = (uint32_t *)(uintptr_t)queues->nontrivial;
         queue[idx] = tri_id;
      }
      return;
   }

   if (s.is_point) {
      rasterize_point<ABUF>(args, &s, tri_id, 0, 1);
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
   for (int py = s.iy_min; py <= s.iy_max; py++) {
      for (int px = s.ix_min; px <= s.ix_max; px++) {
         for (int sm = 0; sm < (int)args->num_samples; sm++) {
            float ox, oy;
            cp_sample_pos(args->num_samples, sm, &ox, &oy);
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

               emit_fragment<ABUF>(args, tri_id, px, py, sm,
                                   w0 * s.ndc_z0 + w1 * s.ndc_z1 +
                                   w2 * s.ndc_z2);
            }
         }
      }
   }
   return;
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
   uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
   /* After clipping the count lives on the device, so the grid is sized for
    * the worst case and each thread bounds itself. */
   if (work >= cp_num_triangles(&args))
      return;
   uint32_t tri_id = cp_triangle_id(&args, work);

   /* A reusing pass already has this id in the queue from the first pass,
    * at the same index and behind the same counter. The classification is
    * still done — it is what decides this thread does not rasterize — but
    * the append is not. */
   cp_rast_small_or_defer<ABUF>(&args, &queues, tri_id,
                                queues.mode != CP_QUEUE_REUSE);
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
      if (!cp_triangle_id_valid(&args, tri_id))
         continue;

      /* Lane 0 does triangle setup for the whole warp. */
      struct cp_tri_setup s;
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

            /* Publish one immutable setup before any tile refers to it.
             * The stage-2/stage-3 kernel boundary is the publication barrier. */
            uint32_t tile_tri_id = tri_id;
            if (num_tiles >= CP_SETUP_CACHE_MIN_TILES &&
                queues.setup_cache && queues.setup_capacity) {
               uint32_t setup_idx = atomicAdd(
                  (uint32_t *)(uintptr_t)queues.setup_count, 1u);
               if (setup_idx < queues.setup_capacity) {
                  struct cp_setup_cache_entry *cache =
                     (struct cp_setup_cache_entry *)(uintptr_t)queues.setup_cache;
                  cache[setup_idx].tri_id = tri_id;
                  cache[setup_idx].setup = s;
                  tile_tri_id = CP_TILE_SETUP_TAG | setup_idx;
               }
            }

            uint32_t *huge_counter = (uint32_t *)(uintptr_t)queues.huge_count;
            uint32_t base_idx = atomicAdd(huge_counter, (uint32_t)num_tiles);

            struct cp_tile_pair *huge_queue =
               (struct cp_tile_pair *)(uintptr_t)queues.huge_tiles;

            int idx = 0;
            for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
               for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
                  if (base_idx + idx < CP_MAX_HUGE_TILES) {
                     huge_queue[base_idx + idx].tri_id = tile_tri_id;
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
 * Clipping and stage 1 in one launch: one thread per *input* triangle clips
 * it (every side effect of cp_clip_triangles included — the emitted
 * triangles, the compact counter, the active-ID worklist, the retired
 * slots), then runs stage 1's classify-or-rasterize on the outputs it just
 * emitted, whose IDs it knows without re-deriving the assignment. Small
 * primitives are rasterized on the spot; nontrivial ones are appended to
 * the queue exactly as classic stage 1 appends them, and stages 2 and 3
 * follow as their own launches.
 *
 * Stage 2 is deliberately *not* folded in. A first version processed each
 * warp's nontrivial primitives warp-locally, and the old capture's median
 * went from 24.2 to 27.2 ms: classic stage 2 spreads the queue over up to
 * 4,096 warps, while the warp-local form serialized up to 32 entries behind
 * the one warp that classified them — consumer parallelism has to scale
 * with the work, not with the producers, and only a launch whose grid is
 * sized to the machine does that. The same argument keeps stage 3 separate,
 * with its grid of 2,048 blocks.
 *
 * Launched wherever the classic sequence was clip then stage 1 back to
 * back, which is any first rasterization of a draw (CP_QUEUE_FILL or
 * CP_QUEUE_BUILD); a reusing pass never clips. CUDAVK_NO_FUSED_RAST
 * restores the standalone pair.
 */
template <bool ABUF>
static __device__ __forceinline__ void
cp_clip_rast_fused_body(struct cp_clip_args cargs,
                        struct cp_rasterize_args args,
                        struct cp_rast_queues queues)
{
   if (args.path_flag &&
       (!!*(const volatile uint32_t *)(uintptr_t)args.path_flag) !=
          !!args.path_value)
      return;

   uint32_t tri = blockIdx.x * blockDim.x + threadIdx.x;
   if (tri >= cargs.num_triangles)
      return;

   uint32_t emitted[CP_CLIP_MAX_OUT];
   int n = cp_clip_one(&cargs, tri, emitted);
   for (int i = 0; i < n; i++) {
      cp_rast_small_or_defer<ABUF>(&args, &queues, emitted[i],
                                   queues.mode != CP_QUEUE_REUSE);
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

   struct cp_tile_pair *huge_queue =
      (struct cp_tile_pair *)(uintptr_t)queues.huge_tiles;

   /*
    * The setup, shared across the block. The whole struct rather than the
    * fields a triangle happens to need, for the reason cp_broadcast_setup()
    * gives: a point carries a square where a triangle carries edges, and a
    * copy that names fields has to be extended for each new kind.
    */
   __shared__ struct cp_tri_setup sh_s;
   __shared__ uint32_t sh_tri_id;
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
         struct cp_tri_setup s;
         uint32_t original_tri_id = tri_id;
         bool setup_valid = false;
         sh_valid = 0;

         if (tri_id & CP_TILE_SETUP_TAG) {
            uint32_t setup_idx = tri_id & CP_TILE_SETUP_INDEX_MASK;
            if (queues.setup_cache && setup_idx < queues.setup_capacity) {
               const struct cp_setup_cache_entry *cache =
                  (const struct cp_setup_cache_entry *)(uintptr_t)queues.setup_cache;
               struct cp_setup_cache_entry cached = cache[setup_idx];
               original_tri_id = cached.tri_id;
               s = cached.setup;
               setup_valid = cp_triangle_id_valid(&args, original_tri_id);
            }
         } else if (cp_triangle_id_valid(&args, tri_id)) {
            setup_valid = setup_triangle(&args, tri_id, &s);
         }

         if (setup_valid) {
            sh_s = s;
            sh_tri_id = original_tri_id;

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
      tri_id = sh_tri_id;

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

      if (col < CP_TILE_SIZE && px >= sh_s.ix_min && px <= sh_s.ix_max) {
         for (int row = 0; row < CP_TILE_SIZE; row++) {
            int py = tile_y + row;
            if (py > sh_s.iy_max)
               break;
            if (py < sh_s.iy_min)
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
   if (args.path_flag &&
       (!!*(const volatile uint32_t *)(uintptr_t)args.path_flag) !=
          !!args.path_value)
      return;
   cp_rasterize_stage1_body<false>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage2(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   if (args.path_flag &&
       (!!*(const volatile uint32_t *)(uintptr_t)args.path_flag) !=
          !!args.path_value)
      return;
   cp_rasterize_stage2_body<false>(args, queues);
}

extern "C" __global__ void
cp_rasterize_stage3(struct cp_rasterize_args args, struct cp_rast_queues queues)
{
   if (args.path_flag &&
       (!!*(const volatile uint32_t *)(uintptr_t)args.path_flag) !=
          !!args.path_value)
      return;
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
 * The fused clip+stage1 forms, in the same two specialisations. The host
 * launches one of these with one thread per input triangle wherever the
 * classic sequence was clip then stage 1 back to back; stages 2 and 3 follow
 * as their own launches either way. CUDAVK_NO_FUSED_RAST restores the
 * standalone pair.
 */
extern "C" __global__ void
cp_clip_rast_fused(struct cp_clip_args cargs, struct cp_rasterize_args args,
                   struct cp_rast_queues queues)
{
   cp_clip_rast_fused_body<false>(cargs, args, queues);
}

extern "C" __global__ void
cp_clip_rast_fused_abuf(struct cp_clip_args cargs,
                        struct cp_rasterize_args args,
                        struct cp_rast_queues queues)
{
   cp_clip_rast_fused_body<true>(cargs, args, queues);
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
cp_abuf_scan_add(uint32_t *data, const uint32_t *sums, uint32_t n,
                 uint32_t *counts, const uint32_t *total, uint32_t capacity,
                 uint32_t *overflow)
{
   uint32_t i = blockIdx.x * CP_ABUF_SCAN_BLOCK + threadIdx.x;
   if (i < n) {
      data[i] += sums[blockIdx.x];
      if (counts && total && *total > capacity) {
         uint32_t count = counts[i];
         uint32_t room = data[i] < capacity ? capacity - data[i] : 0;
         if (count > room) {
            counts[i] = room;
            atomicAdd(overflow, count - room);
         }
      }
   }
}

/*
 * ---------------------------------------------------------------------------
 * The same scan in two launches instead of three or five
 * ---------------------------------------------------------------------------
 *
 * `cp_abuf_scan_block` + `cp_abuf_scan_add` need one level per 512 elements
 * and an add-back per level, so a 921,600-pixel scan costs five launches. The
 * pair below costs two for any n, and produces bit-identical offsets.
 *
 * The host picks one tiling and every kernel here shares it:
 *
 *     ept  = max(1, ceil(n / (512*512)))     elements per thread
 *     grid = ceil(n / (512*ept))             <= 512, by that choice of ept
 *
 * `grid <= 512` is the whole trick: the array of per-block sums is small
 * enough that *every* block can scan all of it itself, in shared memory, and
 * read out both its own exclusive base and the grand total without waiting for
 * or talking to any other block. There is no queue, no claiming and no
 * residency here — each block still has a static index range and one exit.
 * The redundant top-level scan is ~512 loads out of a 2 KB array that every
 * block reads, which is L2-resident by the second block.
 */
extern "C" __global__ void
cp_abuf_scan_reduce(const uint32_t *in, uint32_t *sums, uint32_t n,
                    uint32_t ept)
{
   __shared__ uint32_t red[CP_ABUF_SCAN_BLOCK];
   uint32_t tid = threadIdx.x;
   uint32_t base = blockIdx.x * (CP_ABUF_SCAN_BLOCK * ept);

   uint32_t acc = 0;
   for (uint32_t j = 0; j < ept; j++) {
      uint32_t i = base + j * CP_ABUF_SCAN_BLOCK + tid;
      if (i < n)
         acc += in[i];
   }

   red[tid] = acc;
   __syncthreads();
   for (uint32_t off = CP_ABUF_SCAN_BLOCK >> 1; off; off >>= 1) {
      if (tid < off)
         red[tid] += red[tid + off];
      __syncthreads();
   }
   if (tid == 0)
      sums[blockIdx.x] = red[0];
}

/*
 * The other half: the block's base out of the redundant top scan, then the
 * block's own tile in `ept` rounds of the same Hillis-Steele scan
 * `cp_abuf_scan_block` runs, writing the final offsets straight out.
 *
 * `zero`, when given, is cleared over the same n. It is the fill cursor, whose
 * memset was a separate 3.7 MB device operation immediately after this one.
 *
 * `counts`/`capacity`/`overflow` are `cp_abuf_scan_add`'s clamp, carried over
 * unchanged except that the grand total it tests is already in a register.
 * `break_mode` is the negative control and is 0 in every shipping path.
 */
extern "C" __global__ void
cp_abuf_scan_finish(const uint32_t *in, uint32_t *out, const uint32_t *sums,
                    uint32_t nsums, uint32_t n, uint32_t ept,
                    uint32_t *total_out, uint32_t *zero, uint32_t *counts,
                    uint32_t capacity, uint32_t *overflow, uint32_t break_mode)
{
   __shared__ uint32_t buf[2][CP_ABUF_SCAN_BLOCK];
   __shared__ uint32_t sh_base, sh_total;
   uint32_t tid = threadIdx.x;

   /* The top level, redundantly, in every block. */
   uint32_t s = tid < nsums ? sums[tid] : 0u;
   int pin = 0;
   buf[pin][tid] = s;
   __syncthreads();
   for (uint32_t off = 1; off < CP_ABUF_SCAN_BLOCK; off <<= 1) {
      uint32_t x = buf[pin][tid];
      if (tid >= off)
         x += buf[pin][tid - off];
      pin ^= 1;
      buf[pin][tid] = x;
      __syncthreads();
   }
   if (tid == blockIdx.x)
      sh_base = buf[pin][tid] - s;
   if (tid == CP_ABUF_SCAN_BLOCK - 1)
      sh_total = buf[pin][tid];
   __syncthreads();

   uint32_t run = break_mode == 1u ? 0u : sh_base;
   uint32_t total = sh_total;
   if (total_out && blockIdx.x == 0 && tid == 0)
      *total_out = total;

   uint32_t base = blockIdx.x * (CP_ABUF_SCAN_BLOCK * ept);
   for (uint32_t j = 0; j < ept; j++) {
      uint32_t i = base + j * CP_ABUF_SCAN_BLOCK + tid;
      uint32_t v = i < n ? in[i] : 0u;

      __syncthreads();
      pin = 0;
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
      uint32_t off_i = run + incl - v;
      if (i < n) {
         out[i] = off_i;
         if (zero)
            zero[i] = 0u;
         /* Identical to cp_abuf_scan_add's clamp, including the counter it
          * reports into: a run that cannot fit is cut and the difference is
          * the same overflow the drain's verdict already reads. */
         if (counts && total > capacity) {
            uint32_t c = counts[i];
            uint32_t room = off_i < capacity ? capacity - off_i : 0u;
            if (c > room) {
               counts[i] = room;
               atomicAdd(overflow, c - room);
            }
         }
      }
      run += buf[pin][CP_ABUF_SCAN_BLOCK - 1];
   }
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
 * The fill, replayed from the records the count pass appended instead of
 * rasterized a second time. Each record already names its pixel and its
 * primitive, so placing it is the same cursor-and-bounds sequence
 * emit_fragment's fill mode runs — kept identical on purpose, including the
 * double bound and the overflow report, so the drain's verdict means the same
 * thing whichever fill produced it. The record count lives on the device (the
 * host never drained for it), clamped to the array because an overflowing
 * count pass ran the cursor past the end.
 */
extern "C" __global__ void
cp_abuf_fill_recs(const uint64_t *recs, const uint32_t *rec_cursor,
                  uint32_t rec_capacity, uint32_t *frags,
                  const uint32_t *offsets, const uint32_t *counts,
                  uint32_t *cursor, uint32_t *overflow, uint32_t capacity)
{
   uint32_t nrec = *rec_cursor;
   if (nrec > rec_capacity)
      nrec = rec_capacity;
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nrec;
        i += stride) {
      uint64_t rec = recs[i];
      uint32_t p = (uint32_t)(rec >> 32);
      uint32_t slot = atomicAdd(cursor + p, 1u);
      uint32_t n = counts[p];
      uint32_t base = offsets[p];
      if (slot < n && base + slot < capacity)
         frags[base + slot] = (uint32_t)rec;
      else
         atomicAdd(overflow, 1u);
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

/* Short runs dominate ordinary transparency. Giving each of them a 256-thread
 * block costs more scheduling than sorting; one thread scanning the framebuffer
 * handles the common 2..32 element case and leaves only long runs on the
 * block-wide worklist path. */
extern "C" __global__ void
cp_abuf_sort_short(uint32_t *frags, const uint32_t *offsets,
                   const uint32_t *counts, uint32_t npixels,
                   uint32_t max_short, uint32_t *covered_list,
                   uint32_t *covered_count, uint32_t min_long,
                   uint32_t *long_list, uint32_t *long_count)
{
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < npixels;
        p += stride) {
      uint32_t n = counts[p];
      if (!n)
         continue;
      uint32_t at = atomicAdd(covered_count, 1u);
      covered_list[at] = p;
      if (n >= min_long) {
         uint32_t long_at = atomicAdd(long_count, 1u);
         long_list[long_at] = p;
      }
      if (n < 2 || n > max_short)
         continue;
      uint32_t *run = frags + offsets[p];
      for (uint32_t i = 1; i < n; i++) {
         uint32_t value = run[i];
         int32_t j = (int32_t)i - 1;
         while (j >= 0 && run[j] > value) {
            run[j + 1] = run[j];
            j--;
         }
         run[j + 1] = value;
      }
   }
}

#if CP_ABUF_INSTRUMENT
/*
 * What the peel loop selected at every pixel on this pass. VISBUF_TRIID of an
 * empty entry is 0, which is a valid primitive, so the empty case is written
 * out explicitly.
 *
 * Verification only (CUDAVK_ABUFFER_VERIFY): the A-buffer path does not run
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
 * ---------------------------------------------------------------------------
 * The same quad build in three launches instead of six
 * ---------------------------------------------------------------------------
 *
 * `block_worklist` compacts the covered blocks, `quad_count` merges them into
 * `blk_counts`, a three-launch scan turns those into `blk_offsets`, and
 * `quad_fill` merges again at those offsets — six launches and a 0.9 MB memset
 * of `blk_counts` beforehand.
 *
 * The pair below is laid out on the scan's own tiling of `nblocks`, so the
 * counting pass can both replace the compaction (it tests coverage itself,
 * from the four counts `block_worklist` was reading anyway) and produce the
 * scan's per-block sums (its tile *is* the scan's tile). What is left is
 * count -> `cp_abuf_scan_finish` -> fill.
 *
 * `blk_counts` is written for **every** b < nblocks, zero included, which is
 * what retires the memset. That is the property `CUDAVK_ABUF_FUSE_CHECK`
 * proves by pre-filling the array with a sentinel and counting survivors,
 * rather than by reading the index arithmetic.
 */
extern "C" __global__ void
cp_abuf_quad_count_all(const uint32_t *frags, const uint32_t *offsets,
                       const uint32_t *counts, uint32_t width, uint32_t height,
                       uint32_t quad_width, uint32_t nblocks,
                       uint32_t *blk_counts, uint32_t *sums, uint32_t ept,
                       uint32_t break_mode)
{
   __shared__ uint32_t red[CP_ABUF_SCAN_BLOCK];
   uint32_t tid = threadIdx.x;
   uint32_t base = blockIdx.x * (CP_ABUF_SCAN_BLOCK * ept);

   uint32_t acc = 0;
   for (uint32_t j = 0; j < ept; j++) {
      uint32_t b = base + j * CP_ABUF_SCAN_BLOCK + tid;
      if (b >= nblocks)
         continue;

      uint32_t qx = (b % quad_width) * 2;
      uint32_t qy = (b / quad_width) * 2;
      uint32_t any = 0;
      for (int i = 0; i < 4; i++) {
         uint32_t x = qx + (i & 1), y = qy + (i >> 1);
         if (x < width && y < height)
            any |= counts[(size_t)y * width + x];
      }

      uint32_t c = 0;
      if (any)
         c = cp_abuf_merge_block(frags, offsets, counts, width, height,
                                 quad_width, b, NULL, NULL, NULL, NULL, NULL,
                                 0, 0, NULL);
      /* break_mode 2 is the negative control: leave the uncovered entry
       * alone, which is exactly what a missing memset would look like. */
      if (any || break_mode != 2u)
         blk_counts[b] = c;
      acc += c;
   }

   red[tid] = acc;
   __syncthreads();
   for (uint32_t off = CP_ABUF_SCAN_BLOCK >> 1; off; off >>= 1) {
      if (tid < off)
         red[tid] += red[tid + off];
      __syncthreads();
   }
   if (tid == 0 && sums)
      sums[blockIdx.x] = red[0];
}

/*
 * The fill, over every block rather than over the compacted list. A block with
 * no quads is one load and a return; a block with quads runs the identical
 * merge at the identical offset.
 */
extern "C" __global__ void
cp_abuf_quad_fill_all(const uint32_t *frags, const uint32_t *offsets,
                      const uint32_t *counts, uint32_t width, uint32_t height,
                      uint32_t quad_width, uint32_t nblocks,
                      const uint32_t *blk_counts, const uint32_t *blk_offsets,
                      uint32_t *quad_prim, unsigned char *quad_mask,
                      uint32_t *quad_peel_mask, uint32_t *quad_block,
                      uint32_t *shade_slot, uint32_t capacity,
                      uint32_t *overflow)
{
   uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
   if (b >= nblocks || !blk_counts[b])
      return;
   cp_abuf_merge_block(frags, offsets, counts, width, height, quad_width, b,
                       quad_prim, quad_mask, quad_peel_mask, quad_block,
                       shade_slot, blk_offsets[b], capacity, overflow);
}

/*
 * The equivalence gate. Never launched unless CUDAVK_ABUF_FUSE_CHECK is set,
 * so it costs nothing in any timed or shipping run.
 *
 * out[0] elements that differ, out[1] entries never written (still the
 * sentinel), out[2] blocks where "has quads" disagrees with "is covered".
 */
extern "C" __global__ void
cp_abuf_fuse_cmp(const uint32_t *a, const uint32_t *b, uint32_t n,
                 uint32_t sentinel, uint32_t check_sentinel, uint32_t *out)
{
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
      if (a[i] != b[i])
         atomicAdd(out, 1u);
      if (check_sentinel && a[i] == sentinel)
         atomicAdd(out + 1, 1u);
   }
}

extern "C" __global__ void
cp_abuf_fuse_cover(const uint32_t *counts, uint32_t width, uint32_t height,
                   uint32_t quad_width, uint32_t nblocks,
                   const uint32_t *blk_counts, uint32_t *out)
{
   uint32_t stride = gridDim.x * blockDim.x;
   for (uint32_t b = blockIdx.x * blockDim.x + threadIdx.x; b < nblocks;
        b += stride) {
      uint32_t qx = (b % quad_width) * 2;
      uint32_t qy = (b / quad_width) * 2;
      uint32_t any = 0;
      for (int i = 0; i < 4; i++) {
         uint32_t x = qx + (i & 1), y = qy + (i >> 1);
         if (x < width && y < height)
            any |= counts[(size_t)y * width + x];
      }
      if ((any != 0u) != (blk_counts[b] != 0u))
         atomicAdd(out + 2, 1u);
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

/*
 * Pass-episode quad bucketing — see struct cp_abuf_seg_args. Which segment a
 * quad belongs to follows from its global primitive id: the last segment
 * whose first primitive slot is not past it. Segments are few (<= 64) and
 * the table is in cache, so the search is a handful of steps.
 */
static __device__ __forceinline__ uint32_t
cp_seg_of_prim(const uint32_t *base, uint32_t nsegs, uint32_t prim)
{
   uint32_t lo = 0, hi = nsegs - 1;
   while (lo < hi) {
      uint32_t mid = (lo + hi + 1) >> 1;
      if (base[mid] <= prim)
         lo = mid;
      else
         hi = mid - 1;
   }
   return lo;
}

extern "C" __global__ void
cp_abuf_seg_count(struct cp_abuf_seg_args args)
{
   /* The quad total lives on the device when this launches, so the grid is a
    * fixed statement of how much machine to use and the threads stride. */
   uint32_t total = *(const uint32_t *)(uintptr_t)args.num_quads_dev;
   if (total > args.num_quads)
      total = args.num_quads;
   for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < total;
        q += gridDim.x * blockDim.x) {
      uint32_t prim = ((const uint32_t *)(uintptr_t)args.quad_prim)[q];
      unsigned active = __activemask();
      uint32_t seg = cp_seg_of_prim(
         (const uint32_t *)(uintptr_t)args.seg_prim_base, args.nsegs, prim);
      ((unsigned char *)(uintptr_t)args.quad_seg)[q] = (unsigned char)seg;
      if (!args.seg_counts)
         continue;
      if (!args.warp_aggregate) {
         atomicAdd((unsigned int *)(uintptr_t)args.seg_counts + seg, 1u);
         continue;
      }
      unsigned peers = __match_any_sync(active, seg);
      int leader = __ffs(peers) - 1;
      if ((int)(threadIdx.x & 31) == leader)
         atomicAdd((unsigned int *)(uintptr_t)args.seg_counts + seg,
                   (unsigned)__popc(peers));
   }
}

/*
 * Tile shader census, pass 1: mark which shaders reach which tile.
 *
 * One thread per quad. The quad's 2x2 block gives its tile, its primitive
 * gives its segment by the same search the bucketing uses, and the segment
 * gives the distinct fragment shader the host resolved. Nothing is written
 * that any rendering kernel reads.
 */
extern "C" __global__ void
cp_tile_census_mark(struct cp_tile_census_args args)
{
   uint32_t total = *(const uint32_t *)(uintptr_t)args.num_quads_dev;
   if (total > args.num_quads)
      total = args.num_quads;

   const uint32_t *quad_prim = (const uint32_t *)(uintptr_t)args.quad_prim;
   const uint32_t *quad_block = (const uint32_t *)(uintptr_t)args.quad_block;
   const uint8_t *seg_shader = (const uint8_t *)(uintptr_t)args.seg_shader;
   unsigned long long *tile_mask =
      (unsigned long long *)(uintptr_t)args.tile_mask;
   unsigned int *tile_quads = (unsigned int *)(uintptr_t)args.tile_quads;
   unsigned int *tile_smin = (unsigned int *)(uintptr_t)args.tile_smin;
   unsigned int *tile_smax = (unsigned int *)(uintptr_t)args.tile_smax;

   for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < total;
        q += gridDim.x * blockDim.x) {
      uint32_t prim = quad_prim[q];
      uint32_t blk = quad_block[q];
      uint32_t bx = blk % args.quad_width;
      uint32_t by = blk / args.quad_width;
      uint32_t tx = (bx * 2) / args.tile;
      uint32_t ty = (by * 2) / args.tile;
      if (tx >= args.tiles_x || ty >= args.tiles_y)
         continue;
      uint32_t t = ty * args.tiles_x + tx;

      uint32_t seg = cp_seg_of_prim(
         (const uint32_t *)(uintptr_t)args.seg_prim_base, args.nsegs, prim);
      uint32_t s = seg_shader[seg];
      if (s >= CP_TILE_CENSUS_MAX_SHADERS)
         s = CP_TILE_CENSUS_MAX_SHADERS - 1;
      /* Ordered by the pass-global draw sequence: primitive ids restart at
       * every episode, so they cannot be compared across one. */
      uint32_t seq = ((const uint32_t *)(uintptr_t)args.seg_seq)[seg];

      atomicOr(&tile_mask[t], 1ull << s);
      atomicAdd(&tile_quads[t], 1u);
      uint32_t at = t * CP_TILE_CENSUS_MAX_SHADERS + s;
      atomicMin(&tile_smin[at], seq);
      atomicMax(&tile_smax[at], seq);
   }
}

/*
 * The same marking from the opaque side. Opaque draws never enter the
 * A-buffer's quad stream: their coverage is the shared visibility buffer,
 * one winner per pixel, carrying the episode-global primitive that won it.
 * A tile only has to be able to *call* the shader of a fragment it shades,
 * and for opaque geometry that is the winner — so this is the right set,
 * not every primitive that touched the tile.
 */
extern "C" __global__ void
cp_tile_census_mark_vis(struct cp_tile_census_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= args.width || y >= args.height)
      return;

   uint64_t entry =
      ((const uint64_t *)(uintptr_t)args.visbuf)[(size_t)y * args.width + x];
   if (entry == VISBUF_EMPTY)
      return;

   uint32_t tx = x / args.tile;
   uint32_t ty = y / args.tile;
   if (tx >= args.tiles_x || ty >= args.tiles_y)
      return;
   uint32_t t = ty * args.tiles_x + tx;

   uint32_t prim = VISBUF_TRIID(entry);
   uint32_t seg = cp_seg_of_prim(
      (const uint32_t *)(uintptr_t)args.seg_prim_base, args.nsegs, prim);
   uint32_t s = ((const uint8_t *)(uintptr_t)args.seg_shader)[seg];
   if (s >= CP_TILE_CENSUS_MAX_SHADERS)
      s = CP_TILE_CENSUS_MAX_SHADERS - 1;
   uint32_t seq = ((const uint32_t *)(uintptr_t)args.seg_seq)[seg];

   atomicOr((unsigned long long *)(uintptr_t)args.tile_mask + t, 1ull << s);
   atomicAdd((unsigned int *)(uintptr_t)args.tile_quads + t, 1u);
   uint32_t at = t * CP_TILE_CENSUS_MAX_SHADERS + s;
   atomicMin((unsigned int *)(uintptr_t)args.tile_smin + at, seq);
   atomicMax((unsigned int *)(uintptr_t)args.tile_smax + at, seq);
}

/*
 * Pass 2: one thread per tile, reducing straight into a persistent histogram
 * so the host never reads a per-tile array back.
 *
 * "Disjoint" means every shader present owns a contiguous stretch of this
 * tile's primitive order, so the tile could be rendered as that many ordered
 * passes, each with one statically linked shader. Overlapping ranges mean the
 * shaders interleave and only a kernel that can dispatch among them would
 * keep the order.
 */
/*
 * The bin's own size, which is a different quantity from the shaded one: a
 * triangle spanning twenty tiles costs twenty references and may shade none
 * of them. This is what sets a tiler's memory ceiling and what the previous
 * prototype's fixed 2,000,000-reference capacity ran into, so it is measured
 * rather than assumed. Conservative bounding-box binning, the same rule the
 * prototype used, so the number is an upper bound on an exact-coverage bin.
 */
extern "C" __global__ void
cp_tile_census_refs(struct cp_rasterize_args rast,
                    struct cp_tile_census_args args)
{
   uint32_t n = cp_num_triangles(&rast);
   for (uint32_t work = blockIdx.x * blockDim.x + threadIdx.x; work < n;
        work += gridDim.x * blockDim.x) {
      uint32_t tri = cp_triangle_id(&rast, work);
      struct cp_tri_setup setup;
      if (!setup_triangle(&rast, tri, &setup))
         continue;
      uint32_t tx0 = (uint32_t)setup.ix_min / args.tile;
      uint32_t ty0 = (uint32_t)setup.iy_min / args.tile;
      uint32_t tx1 = (uint32_t)setup.ix_max / args.tile;
      uint32_t ty1 = (uint32_t)setup.iy_max / args.tile;
      for (uint32_t ty = ty0; ty <= ty1 && ty < args.tiles_y; ty++)
         for (uint32_t tx = tx0; tx <= tx1 && tx < args.tiles_x; tx++)
            atomicAdd((unsigned int *)(uintptr_t)args.tile_refs +
                         ty * args.tiles_x + tx, 1u);
   }
}

static __device__ __forceinline__ uint32_t
cp_census_log_bucket(uint32_t v)
{
   uint32_t b = 32u - (uint32_t)__clz((int)v);   /* 1 -> 1, 2..3 -> 2, ... */
   return b < CP_TILE_CENSUS_LOG ? b : CP_TILE_CENSUS_LOG - 1;
}

extern "C" __global__ void
cp_tile_census_reduce(struct cp_tile_census_args args)
{
   uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
   if (t >= args.tiles_x * args.tiles_y)
      return;

   unsigned long long *hist = (unsigned long long *)(uintptr_t)args.hist;

   /* The bin size is independent of whether anything shaded here: a tile can
    * hold references and lose every one of them to the depth test. */
   unsigned int refs = args.tile_refs
      ? ((const unsigned int *)(uintptr_t)args.tile_refs)[t] : 0u;
   if (refs) {
      atomicAdd(&hist[CP_TILE_CENSUS_REFS + cp_census_log_bucket(refs)], 1ull);
      atomicAdd(&hist[CP_TILE_CENSUS_GLOBALS + 2], (unsigned long long)refs);
      atomicAdd(&hist[CP_TILE_CENSUS_GLOBALS + 4], 1ull);
      atomicMax((unsigned long long *)&hist[CP_TILE_CENSUS_GLOBALS + 0],
                (unsigned long long)refs);
   }

   unsigned long long mask =
      ((const unsigned long long *)(uintptr_t)args.tile_mask)[t];
   if (!mask)
      return;

   {
      unsigned int sh = ((const unsigned int *)(uintptr_t)args.tile_quads)[t];
      if (sh) {
         atomicAdd(&hist[CP_TILE_CENSUS_SHADED + cp_census_log_bucket(sh)],
                   1ull);
         atomicAdd(&hist[CP_TILE_CENSUS_GLOBALS + 3], (unsigned long long)sh);
         atomicAdd(&hist[CP_TILE_CENSUS_GLOBALS + 5], 1ull);
         atomicMax((unsigned long long *)&hist[CP_TILE_CENSUS_GLOBALS + 1],
                   (unsigned long long)sh);
      }
   }

   const unsigned int *tile_smin = (const unsigned int *)(uintptr_t)args.tile_smin;
   const unsigned int *tile_smax = (const unsigned int *)(uintptr_t)args.tile_smax;
   uint32_t base = t * CP_TILE_CENSUS_MAX_SHADERS;

   uint32_t lo[CP_TILE_CENSUS_MAX_SHADERS];
   uint32_t hi[CP_TILE_CENSUS_MAX_SHADERS];
   uint32_t n = 0;
   for (uint32_t s = 0; s < CP_TILE_CENSUS_MAX_SHADERS; s++) {
      if (!(mask & (1ull << s)))
         continue;
      lo[n] = tile_smin[base + s];
      hi[n] = tile_smax[base + s];
      n++;
   }

   int disjoint = 1;
   for (uint32_t i = 0; i < n && disjoint; i++)
      for (uint32_t j = i + 1; j < n; j++)
         if (lo[i] <= hi[j] && lo[j] <= hi[i]) {
            disjoint = 0;
            break;
         }

   uint32_t bin = n < CP_TILE_CENSUS_BINS ? n : CP_TILE_CENSUS_BINS - 1;
   unsigned int quads = ((const unsigned int *)(uintptr_t)args.tile_quads)[t];
   atomicAdd(&hist[bin * 4 + (disjoint ? 0 : 1)], 1ull);
   atomicAdd(&hist[bin * 4 + (disjoint ? 2 : 3)], (unsigned long long)quads);
}

extern "C" __global__ void
cp_abuf_seg_scatter(struct cp_abuf_seg_args args)
{
   uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t exact = args.num_quads_dev
      ? *(const uint32_t *)(uintptr_t)args.num_quads_dev : args.num_quads;
   if (q >= args.num_quads || q >= exact)
      return;
   uint32_t seg = ((const unsigned char *)(uintptr_t)args.quad_seg)[q];
   if (!args.warp_aggregate) {
      uint32_t pos = atomicAdd(
         (unsigned int *)(uintptr_t)args.seg_cursor + seg, 1u);
      uint32_t at = ((const uint32_t *)(uintptr_t)args.seg_base)[seg] + pos;
      ((uint32_t *)(uintptr_t)args.grouped)[at] = q;
      uint32_t dense = pos;
      if (args.seg_group && args.group_base) {
         uint32_t group = ((const uint8_t *)(uintptr_t)args.seg_group)[seg];
         dense = at - ((const uint32_t *)(uintptr_t)args.group_base)[group];
      }
      ((uint32_t *)(uintptr_t)args.quad_dense)[q] = dense;
      return;
   }
   unsigned active = __activemask();
   unsigned peers = __match_any_sync(active, seg);
   int leader = __ffs(peers) - 1;
   uint32_t group_pos = 0;
   if ((int)(threadIdx.x & 31) == leader)
      group_pos = atomicAdd((unsigned int *)(uintptr_t)args.seg_cursor + seg,
                            (unsigned)__popc(peers));
   group_pos = __shfl_sync(peers, group_pos, leader);
   unsigned lane_mask = (1u << (threadIdx.x & 31)) - 1u;
   uint32_t pos = group_pos + (uint32_t)__popc(peers & lane_mask);
   uint32_t at = ((const uint32_t *)(uintptr_t)args.seg_base)[seg] + pos;
   ((uint32_t *)(uintptr_t)args.grouped)[at] = q;
   uint32_t dense = pos;
   if (args.seg_group && args.group_base) {
      uint32_t group = ((const uint8_t *)(uintptr_t)args.seg_group)[seg];
      dense = at - ((const uint32_t *)(uintptr_t)args.group_base)[group];
   }
   ((uint32_t *)(uintptr_t)args.quad_dense)[q] = dense;
}

extern "C" __global__ void
cp_abuf_seg_prefix(struct cp_abuf_seg_prefix_args args)
{
   if (blockIdx.x || threadIdx.x)
      return;
   const uint32_t *counts = (const uint32_t *)(uintptr_t)args.seg_counts;
   const uint8_t *groups = (const uint8_t *)(uintptr_t)args.seg_group;
   uint32_t *bases = (uint32_t *)(uintptr_t)args.seg_base;
   uint32_t *group_base = (uint32_t *)(uintptr_t)args.group_base;
   uint32_t *group_counts = (uint32_t *)(uintptr_t)args.group_counts;
   uint32_t running = 0;
   for (uint32_t g = 0; g < args.ngroups; g++) {
      group_base[g] = running;
      uint32_t first = running;
      for (uint32_t s = 0; s < args.nsegs; s++) {
         if (groups[s] != g)
            continue;
         bases[s] = running;
         running += counts[s];
      }
      group_counts[g] = running - first;
   }
}

extern "C" __global__ void
cp_abuf_prepare_shade_count(struct cp_abuf_shade_count_args args)
{
   if (blockIdx.x || threadIdx.x)
      return;
   *(uint32_t *)(uintptr_t)args.slots =
      4u * *(const uint32_t *)(uintptr_t)args.count;
}

extern "C" __global__ void
cp_opaque_tile_count(struct cp_opaque_tile_build_args args)
{
   uint32_t n = cp_num_triangles(&args.rast);
   for (uint32_t work = blockIdx.x * blockDim.x + threadIdx.x; work < n;
        work += gridDim.x * blockDim.x) {
      uint32_t tri = cp_triangle_id(&args.rast, work);
      struct cp_tri_setup setup;
      if (!setup_triangle(&args.rast, tri, &setup))
         continue;
      uint32_t tx0 = (uint32_t)setup.ix_min / CP_OPAQUE_TILE_SIZE;
      uint32_t ty0 = (uint32_t)setup.iy_min / CP_OPAQUE_TILE_SIZE;
      uint32_t tx1 = (uint32_t)setup.ix_max / CP_OPAQUE_TILE_SIZE;
      uint32_t ty1 = (uint32_t)setup.iy_max / CP_OPAQUE_TILE_SIZE;
      for (uint32_t ty = ty0; ty <= ty1 && ty < args.tiles_y; ty++)
         for (uint32_t tx = tx0; tx <= tx1 && tx < args.tiles_x; tx++)
            atomicAdd((uint32_t *)(uintptr_t)args.tile_counts +
                         ty * args.tiles_x + tx, 1u);
   }
}

extern "C" __global__ void
cp_opaque_tile_fill(struct cp_opaque_tile_build_args args)
{
   uint32_t n = cp_num_triangles(&args.rast);
   for (uint32_t work = blockIdx.x * blockDim.x + threadIdx.x; work < n;
        work += gridDim.x * blockDim.x) {
      uint32_t tri = cp_triangle_id(&args.rast, work);
      struct cp_tri_setup setup;
      if (!setup_triangle(&args.rast, tri, &setup))
         continue;
      uint32_t tx0 = (uint32_t)setup.ix_min / CP_OPAQUE_TILE_SIZE;
      uint32_t ty0 = (uint32_t)setup.iy_min / CP_OPAQUE_TILE_SIZE;
      uint32_t tx1 = (uint32_t)setup.ix_max / CP_OPAQUE_TILE_SIZE;
      uint32_t ty1 = (uint32_t)setup.iy_max / CP_OPAQUE_TILE_SIZE;
      for (uint32_t ty = ty0; ty <= ty1 && ty < args.tiles_y; ty++) {
         for (uint32_t tx = tx0; tx <= tx1 && tx < args.tiles_x; tx++) {
            uint32_t tile = ty * args.tiles_x + tx;
            uint32_t pos = atomicAdd(
               (uint32_t *)(uintptr_t)args.tile_cursors + tile, 1u);
            uint32_t count = ((const uint32_t *)(uintptr_t)args.tile_counts)[tile];
            uint32_t at = ((const uint32_t *)(uintptr_t)args.tile_offsets)[tile] + pos;
            if (pos < count && at < args.capacity) {
               struct cp_opaque_tile_ref *refs =
                  (struct cp_opaque_tile_ref *)(uintptr_t)args.tile_refs;
               refs[at].global_prim = args.rast.abuf_prim_base + tri;
               refs[at].segment = (uint16_t)args.segment;
               refs[at].flags = 0;
            } else {
               atomicAdd((uint32_t *)(uintptr_t)args.overflow, 1u);
            }
         }
      }
   }
}

static __device__ __forceinline__ bool
cp_opaque_depth_pass(const struct cp_rasterize_args *args, uint32_t at,
                     uint32_t depth)
{
   if (!args->depth_test || !args->depthbuf)
      return true;
   uint32_t prev = ((const uint32_t *)(uintptr_t)args->depthbuf)[at];
   switch (args->depth_func) {
   case CP_FUNC_NEVER: return false;
   case CP_FUNC_LESS: return depth < prev;
   case CP_FUNC_EQUAL: return depth == prev;
   case CP_FUNC_LEQUAL: return depth <= prev;
   case CP_FUNC_GREATER: return depth > prev;
   case CP_FUNC_NOTEQUAL: return depth != prev;
   case CP_FUNC_GEQUAL: return depth >= prev;
   default: return true;
   }
}

extern "C" __global__ void
cp_opaque_tile_raster(struct cp_opaque_tile_raster_args args)
{
   uint32_t tile = blockIdx.x;
   uint32_t ntiles = args.tiles_x * args.tiles_y;
   if (tile >= ntiles || *(const uint32_t *)(uintptr_t)args.overflow)
      return;
   uint32_t count = ((const uint32_t *)(uintptr_t)args.tile_counts)[tile];
   if (!count)
      return;

   uint32_t tx = tile % args.tiles_x;
   uint32_t ty = tile / args.tiles_x;
   uint32_t x0 = tx * CP_OPAQUE_TILE_SIZE;
   uint32_t y0 = ty * CP_OPAQUE_TILE_SIZE;
   const struct cp_rasterize_args *rasts =
      (const struct cp_rasterize_args *)(uintptr_t)args.rast_args;
   const struct cp_opaque_tile_ref *refs =
      (const struct cp_opaque_tile_ref *)(uintptr_t)args.tile_refs;
   uint32_t base = ((const uint32_t *)(uintptr_t)args.tile_offsets)[tile];
   __shared__ struct cp_tri_setup setup;
   __shared__ struct cp_rasterize_args rast;
   __shared__ uint32_t global_prim, local_prim;
   __shared__ int setup_valid;
   uint64_t winner[4] = { VISBUF_EMPTY, VISBUF_EMPTY,
                          VISBUF_EMPTY, VISBUF_EMPTY };

   for (uint32_t i = 0; i < count; i++) {
      if (threadIdx.x == 0) {
         struct cp_opaque_tile_ref ref = refs[base + i];
         setup_valid = 0;
         if (ref.segment < args.num_segments) {
            rast = rasts[ref.segment];
            global_prim = ref.global_prim;
            local_prim = global_prim - rast.abuf_prim_base;
            setup_valid = setup_triangle(&rast, local_prim, &setup) ? 1 : 0;
         }
      }
      __syncthreads();
      if (setup_valid) {
         for (uint32_t j = 0; j < 4; j++) {
            uint32_t owned = threadIdx.x + j * blockDim.x;
            uint32_t px = x0 + owned % CP_OPAQUE_TILE_SIZE;
            uint32_t py = y0 + owned / CP_OPAQUE_TILE_SIZE;
            if (px >= args.width || py >= args.height ||
                px < (uint32_t)setup.ix_min || px > (uint32_t)setup.ix_max ||
                py < (uint32_t)setup.iy_min || py > (uint32_t)setup.iy_max ||
                cp_tri_rejected(&rast, local_prim, px, py))
               continue;
         float sx = (float)px + 0.5f, sy = (float)py + 0.5f;
         float ndc_z;
         if (setup.is_point) {
            if (sx < setup.pt_x0 || sx >= setup.pt_x1 ||
                sy < setup.pt_y0 || sy >= setup.pt_y1)
               continue;
            ndc_z = setup.ndc_z0;
         } else {
            float e0 = edge_function(setup.sx1, setup.sy1, setup.sx2,
                                     setup.sy2, sx, sy);
            float e1 = edge_function(setup.sx2, setup.sy2, setup.sx0,
                                     setup.sy0, sx, sy);
            float e2 = edge_function(setup.sx0, setup.sy0, setup.sx1,
                                     setup.sy1, sx, sy);
            if (!edge_inside(e0, setup.e0_top_left) ||
                !edge_inside(e1, setup.e1_top_left) ||
                !edge_inside(e2, setup.e2_top_left))
               continue;
            float w0 = e0 * setup.inv_area;
            float w1 = e1 * setup.inv_area;
            ndc_z = w0 * setup.ndc_z0 + w1 * setup.ndc_z1 +
                    (1.0f - w0 - w1) * setup.ndc_z2;
         }
         uint32_t depth = float_to_sortable_uint(ndc_z * 0.5f + 0.5f);
         uint32_t pixel = py * args.width + px;
         if (!cp_opaque_depth_pass(&rast, pixel, depth))
            continue;
         uint32_t key = rast.depth_key_invert ? ~depth : depth;
            uint64_t packed = PACK_VISBUF(key, global_prim);
            if (packed < winner[j])
               winner[j] = packed;
         }
      }
      __syncthreads();
   }
   for (uint32_t j = 0; j < 4; j++) {
      uint32_t owned = threadIdx.x + j * blockDim.x;
      uint32_t px = x0 + owned % CP_OPAQUE_TILE_SIZE;
      uint32_t py = y0 + owned / CP_OPAQUE_TILE_SIZE;
      if (px < args.width && py < args.height)
         ((uint64_t *)(uintptr_t)rasts[0].framebuffer)
            [py * args.width + px] = winner[j];
   }
}
