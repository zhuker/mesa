/*
 * cudapipe fragment stage.
 *
 * The rasterizer leaves a visibility buffer holding the winning triangle per
 * pixel. These two kernels bracket the compiled fragment shader:
 *
 *   cp_fs_interpolate  compacts covered pixels and interpolates the vertex
 *                      shader's varyings at each one, producing the fragment
 *                      shader's input buffer
 *   <fragment shader>  runs as its own kernel, one thread per covered pixel
 *   cp_fs_writeback    blends the shader's colour output into the attachment
 */
#include "cp_rast_types.h"

#define VISBUF_EMPTY 0xFFFFFFFFFFFFFFFFULL

/* Same form as edge_function in cp_rasterize.cu, and it has to stay the same:
 * the sign of the area decides whether the triangle's vertices are swapped,
 * and if the two kernels disagreed on a sliver the varyings here would be
 * fetched in a different order than the one the depth was interpolated in. */
static __device__ __forceinline__ float
cp_edge(float ax, float ay, float bx, float by, float px, float py)
{
   return __fsub_rn(__fmul_rn(bx - px, ay - py),
                    __fmul_rn(by - py, ax - px));
}

/*
 * Interpolate one 2x2 quad of pixels.
 *
 * Pixels are compacted in quads rather than one by one so the fragment shader
 * can take screen-space derivatives by shuffling across the four lanes, the way
 * hardware does. That is what lets a texture coordinate computed inside the
 * shader — a reflection vector, say — pick a mip level at all; tracing the
 * coordinate back to a varying at compile time only ever worked for coordinates
 * that came straight from one.
 *
 * A corner of the quad that no triangle covers is still shaded, as a helper
 * lane, using a covered corner's triangle so its interpolated values lie on the
 * same surface. cp_fs_writeback drops it.
 */
struct cp_interp_tri {
   float sx0, sy0, sx1, sy1, sx2, sy2;
   float ndc_z0, ndc_z1, ndc_z2;
   float inv_w0, inv_w1, inv_w2, inv_area;
   int vidx1, vidx2;
   uint64_t base;
   bool front;
};

static __device__ __forceinline__ bool
cp_interp_setup(const struct cp_fs_interp_args *args, uint32_t tri_id,
                struct cp_interp_tri *tri)
{
   uint32_t pos_stride = args->vs_out_stride / 16;
   if (pos_stride == 0) pos_stride = 1;
   tri->base = (uint64_t)(uintptr_t)cp_primitive_base(
      args->prim_refs, args->positions, tri_id, args->vs_out_stride);
   if (!tri->base)
      return false;
   const float4 *positions = (const float4 *)(uintptr_t)tri->base;
   float4 v0 = positions[0 * pos_stride];
   float4 v1 = positions[1 * pos_stride];
   float4 v2 = positions[2 * pos_stride];

   tri->inv_w0 = 1.0f / v0.w;
   tri->inv_w1 = 1.0f / v1.w;
   tri->inv_w2 = 1.0f / v2.w;

   tri->sx0 = v0.x * tri->inv_w0 * args->vp_scale_x + args->vp_trans_x;
   tri->sy0 = v0.y * tri->inv_w0 * args->vp_scale_y + args->vp_trans_y;
   tri->sx1 = v1.x * tri->inv_w1 * args->vp_scale_x + args->vp_trans_x;
   tri->sy1 = v1.y * tri->inv_w1 * args->vp_scale_y + args->vp_trans_y;
   tri->sx2 = v2.x * tri->inv_w2 * args->vp_scale_x + args->vp_trans_x;
   tri->sy2 = v2.y * tri->inv_w2 * args->vp_scale_y + args->vp_trans_y;

   tri->ndc_z0 = v0.z * tri->inv_w0;
   tri->ndc_z1 = v1.z * tri->inv_w1;
   tri->ndc_z2 = v2.z * tri->inv_w2;

   /* The rasterizer flips clockwise triangles so barycentrics come out
    * positive; mirror that here, and carry the swap through to the vertex
    * indices so varyings are fetched in the matching order. */
   float area = args->point_mode ? 1.0f
              : cp_edge(tri->sx0, tri->sy0, tri->sx1, tri->sy1,
                        tri->sx2, tri->sy2);
   tri->vidx1 = 1;
   tri->vidx2 = 2;
   if (area < 0.0f) {
      float t;
      t = tri->sx1; tri->sx1 = tri->sx2; tri->sx2 = t;
      t = tri->sy1; tri->sy1 = tri->sy2; tri->sy2 = t;
      t = tri->ndc_z1; tri->ndc_z1 = tri->ndc_z2; tri->ndc_z2 = t;
      t = tri->inv_w1; tri->inv_w1 = tri->inv_w2; tri->inv_w2 = t;
      tri->vidx1 = 2;
      tri->vidx2 = 1;
      area = -area;
   }
   if (area == 0.0f)
      return false;

   tri->inv_area = 1.0f / area;
   tri->front = args->point_mode
      ? true : ((tri->vidx1 == 1) == (args->front_ccw != 0));
   return true;
}

static __device__ __forceinline__ bool
cp_interp_pixel_prepared(struct cp_fs_interp_args *args, uint32_t tri_id,
                         uint32_t pixel, uint32_t slot,
                         const struct cp_interp_tri *tri)
{
   float cx = (float)(pixel % args->width) + 0.5f;
   float cy = (float)(pixel / args->width) + 0.5f;

   /*
    * gl_FrontFacing. The sign of the area before the flip above is the
    * winding; which winding is the front is rasterizer state. A point has no
    * winding, and Vulkan calls it front-facing.
    */
   if (args->front_face) {
      ((unsigned char *)(uintptr_t)args->front_face)[slot] = tri->front ? 1 : 0;
   }

   /* A helper lane lies outside the triangle, so its barycentrics go negative.
    * That is exactly what makes the derivative across the quad correct. */
   float b0 = args->point_mode ? 1.0f
            : cp_edge(tri->sx1, tri->sy1, tri->sx2, tri->sy2, cx, cy) *
              tri->inv_area;
   float b1 = args->point_mode ? 0.0f
            : cp_edge(tri->sx2, tri->sy2, tri->sx0, tri->sy0, cx, cy) *
              tri->inv_area;
   float b2 = 1.0f - b0 - b1;

   ((uint32_t *)(uintptr_t)args->pixel_list)[slot] = pixel;

   float persp0 = b0 * tri->inv_w0;
   float persp1 = b1 * tri->inv_w1;
   float persp2 = b2 * tri->inv_w2;
   float inv_persp = 1.0f / (persp0 + persp1 + persp2);

   float4 fc;
   fc.x = cx;
   fc.y = cy;
   fc.z = (b0 * tri->ndc_z0 + b1 * tri->ndc_z1 + b2 * tri->ndc_z2) *
          0.5f + 0.5f;
   fc.w = persp0 + persp1 + persp2;
   if (args->frag_coord)
      ((float4 *)(uintptr_t)args->frag_coord)[slot] = fc;

   const char *vs_out = (const char *)(uintptr_t)tri->base;
   char *fs_in = (char *)(uintptr_t)args->fs_in + (size_t)slot * args->fs_in_stride;

   /*
    * A point has one vertex, so there is nothing to interpolate: every input
    * takes that vertex's value outright. What the fragment shader does vary
    * over is gl_PointCoord, which no vertex shader output drives — it is the
    * position within the point's square, and it is written here.
    *
    * Helper lanes land outside the square and so read outside [0, 1], which is
    * what a derivative of gl_PointCoord needs to come out as 1/size. The
    * particle shaders pick a mip level from it.
    */
   if (args->point_mode) {
      for (uint32_t i = 0; i < args->num_fs_inputs && i < CP_MAX_FS_INPUTS; i++) {
         float4 value = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
         int32_t src = args->input_vs_slot[i];
         if (src >= 0)
            value = *(const float4 *)(vs_out +
               (size_t)src * 16);
         *(float4 *)(fs_in + i * 16) = value;
      }

      if (args->pntc_input >= 0 &&
          (uint32_t)args->pntc_input < args->num_fs_inputs) {
         float size = 1.0f;
         if (args->psiz_slot >= 0)
            size = ((const float4 *)vs_out)[args->psiz_slot].x;
         if (!(size > 0.0f))
            size = 1.0f;
         if (size > CP_MAX_POINT_SIZE)
            size = CP_MAX_POINT_SIZE;

         float px0 = tri->sx0 - size * 0.5f;
         float py0 = tri->sy0 - size * 0.5f;
         *(float4 *)(fs_in + args->pntc_input * 16) =
            make_float4((cx - px0) / size, (cy - py0) / size, 0.0f, 1.0f);
      }

      /* gl_FragCoord — see pos_input. */
      if (args->pos_input >= 0 &&
          (uint32_t)args->pos_input < args->num_fs_inputs)
         *(float4 *)(fs_in + args->pos_input * 16) = fc;
      return true;
   }

   for (uint32_t i = 0; i < args->num_fs_inputs && i < CP_MAX_FS_INPUTS; i++) {
      int32_t src = args->input_vs_slot[i];
      float4 value = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

      if (src >= 0) {
         const float4 *a0 = (const float4 *)(vs_out +
            (size_t)src * 16);
         const float4 *a1 = (const float4 *)(vs_out +
            (size_t)tri->vidx1 * args->vs_out_stride + src * 16);
         const float4 *a2 = (const float4 *)(vs_out +
            (size_t)tri->vidx2 * args->vs_out_stride + src * 16);

         value.x = (a0->x * persp0 + a1->x * persp1 + a2->x * persp2) * inv_persp;
         value.y = (a0->y * persp0 + a1->y * persp1 + a2->y * persp2) * inv_persp;
         value.z = (a0->z * persp0 + a1->z * persp1 + a2->z * persp2) * inv_persp;
         value.w = (a0->w * persp0 + a1->w * persp1 + a2->w * persp2) * inv_persp;
      }

      *(float4 *)(fs_in + i * 16) = value;
   }

   /* gl_FragCoord — see pos_input. Written after the loop so that it overwrites
    * the clip-space position the location match would otherwise have left
    * there. */
   if (args->pos_input >= 0 &&
       (uint32_t)args->pos_input < args->num_fs_inputs)
      *(float4 *)(fs_in + args->pos_input * 16) = fc;
   return true;
}

static __device__ __forceinline__ bool
cp_interp_pixel(struct cp_fs_interp_args *args, uint32_t tri_id,
                uint32_t pixel, uint32_t slot)
{
   struct cp_interp_tri tri;
   return cp_interp_setup(args, tri_id, &tri) &&
          cp_interp_pixel_prepared(args, tri_id, pixel, slot, &tri);
}

#if CP_ABUF_INSTRUMENT
/*
 * Verification only (CUDAPIPE_ABUFFER_VERIFY). Which A-buffer slot holds
 * (pixel, primitive).
 *
 * The A-buffer is one sorted run of primitive ids per pixel, so a fragment's
 * slot is a binary search away — and every fragment either path shades has
 * exactly one, which is what lets the two paths' colours be compared without
 * either of them agreeing on an order. Returns ~0 if the primitive is not in
 * the pixel's run, which is a mismatch rather than a normal outcome.
 */
static __device__ __forceinline__ uint32_t
cp_abuf_slot_for(const struct cp_fs_interp_args *args, uint32_t pixel,
                 uint32_t prim)
{
   const uint32_t *frags = (const uint32_t *)(uintptr_t)args->abuf_frags;
   const uint32_t *offsets = (const uint32_t *)(uintptr_t)args->abuf_offsets;
   const uint32_t *counts = (const uint32_t *)(uintptr_t)args->abuf_counts;
   if (!frags || !offsets || !counts)
      return 0xFFFFFFFFu;

   uint32_t base = offsets[pixel];
   uint32_t lo = 0, hi = counts[pixel];
   while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      uint32_t v = frags[base + mid];
      if (v == prim)
         return base + mid;
      if (v < prim)
         lo = mid + 1;
      else
         hi = mid;
   }
   return 0xFFFFFFFFu;
}
#endif /* CP_ABUF_INSTRUMENT */

/*
 * Step 3b: the same interpolation, driven by the merged quad stream instead of
 * by the visibility buffer.
 *
 * One thread per quad, and a quad's four slots are 4q..4q+3 — the peel path
 * has to take them from an atomic because it does not know how many quads a
 * block will produce until it has looked, and here the merge has already said.
 * Slots stay four-aligned either way, which is what the shader's cross-lane
 * derivatives require.
 *
 * Everything a slot holds comes from cp_interp_pixel, the function the peel
 * interpolator calls, so the two cannot drift: a helper lane is a lane the
 * quad's mask does not name, and it is interpolated exactly like a covered one
 * — outside the triangle, with negative barycentrics.
 *
 * (The helper below is inserted between this comment and the kernel it
 * describes only because both belong here; cp_abuf_interpolate follows it.)
 */
/*
 * Which merged draw a primitive came from, and the four slots that carry the
 * answer to the fragment shader. See cp_fs_interp_args::out_batch_rows.
 *
 * The search is the one cp_vertex_fetch does, over the same table: the last
 * slice whose first assembled vertex is not past this primitive's.
 * `prim >> prim_shift` undoes the clipper's stable layout, which reserves four
 * output slots per input triangle; the shift is zero for a draw that did not
 * clip at all, where the primitive index is the input triangle already. The
 * host sets it from what it actually launched rather than from what it meant
 * to, so a clip that was skipped cannot silently shift every primitive into
 * the wrong draw's material.
 */
static __device__ __forceinline__ void
cp_write_batch_rows(const struct cp_fs_interp_args *args, uint32_t prim,
                    uint32_t base)
{
   uint32_t *rows = (uint32_t *)(uintptr_t)args->out_batch_rows;
   if (!rows)
      return;

   uint32_t row = 0;
   if (args->num_draw_slices > 1) {
      const struct cp_draw_slice *s =
         (const struct cp_draw_slice *)(uintptr_t)args->draw_slices;
      uint32_t vert = (prim >> args->prim_shift) * 3u;
      uint32_t lo = 0, hi = args->num_draw_slices - 1;
      while (lo < hi) {
         uint32_t mid = (lo + hi + 1u) >> 1;
         if (s[mid].vert_begin <= vert)
            lo = mid;
         else
            hi = mid - 1;
      }
      row = lo;
   }

   /* row_base places a merged group's row in its concatenated table; zero
    * everywhere else. */
   row += args->row_base;

   for (int i = 0; i < 4; i++)
      rows[base + i] = row;
}

static __device__ __forceinline__ bool
cp_resolve_seg_range(struct cp_fs_interp_args *args, uint32_t gprim)
{
   if (!args->seg_ranges || !args->num_seg_ranges)
      return true;

   const struct cp_seg_range *ranges =
      (const struct cp_seg_range *)(uintptr_t)args->seg_ranges;
   uint32_t lo = 0, hi = args->num_seg_ranges - 1;
   while (lo < hi) {
      uint32_t mid = (lo + hi + 1u) >> 1;
      if (ranges[mid].prim_base <= gprim)
         lo = mid;
      else
         hi = mid - 1;
   }
   const struct cp_seg_range *range = &ranges[lo];
   if (gprim < range->prim_base || gprim >= range->prim_end)
      return false;

   args->positions = range->positions;
   args->prim_refs = range->prim_refs;
   args->vs_out = range->positions;
   args->abuf_prim_base = range->prim_base;
   args->draw_slices = range->draw_slices;
   args->num_draw_slices = range->num_draw_slices;
   args->prim_shift = range->prim_shift;
   args->row_base = range->row_base;
   return true;
}

/*
 * Fused direct shading, step 2 of 2: interpolate one compacted slot from
 * inside the generated fragment shader.
 *
 * cp_fs_compact below has already allocated this slot, named its pixel and
 * primitive, and written its sample mask into coverage; what is left is
 * exactly the interpolation cp_fs_interpolate would have done for it --
 * setup from the primitive's positions, then cp_interp_pixel_prepared into
 * fs_in / frag_coord / front_face. A degenerate primitive zeroes the slot's
 * coverage, which is the value the peel interpolator writes for one, and
 * returns invalid so the shader body is skipped -- the writeback then drops
 * the slot the same way it always did.
 *
 * A slot at or past max_pixels exists only when the compaction's atomic ran
 * past its refusal, where the peel interpolator wrote nothing either; it is
 * declined rather than read out of bounds.
 */
static __device__ int
cp_fs_direct_lane(const struct cp_fs_interp_args *source, uint32_t slot)
{
   struct cp_fs_interp_args args = *source;
   if (slot >= args.max_pixels)
      return 0;

   uint32_t pixel = ((const uint32_t *)(uintptr_t)args.pixel_list)[slot];
   uint32_t gprim =
      ((const uint32_t *)(uintptr_t)args.out_prim_list)[slot >> 2];
   unsigned char *coverage = (unsigned char *)(uintptr_t)args.coverage;

   /* Cannot fail for a slot the compaction allocated -- it resolved the same
    * primitive before allocating -- but the refusal stays the same shape. */
   bool ok = cp_resolve_seg_range(&args, gprim);
   if (ok) {
      uint32_t prim = gprim - args.abuf_prim_base;
      struct cp_interp_tri tri;
      ok = cp_interp_setup(&args, prim, &tri) &&
           cp_interp_pixel_prepared(&args, prim, pixel, slot, &tri);
   }
   if (!ok && coverage)
      coverage[slot] = 0;
   return ok ? 1 : 0;
}

extern "C" __device__ int
cp_abuf_interpolate_lane(const struct cp_fs_interp_args *source, uint32_t slot)
{
   /* One entry point serves both fused forms, because the same compiled
    * shader binary is launched on the A-buffer path and the direct path and
    * only the argument block says which chain this launch belongs to. */
   if (source->fused_direct)
      return cp_fs_direct_lane(source, slot);

   struct cp_fs_interp_args args = *source;
   uint32_t iq = slot >> 2;
   uint32_t lane = slot & 3u;
   uint32_t nq = args.abuf_num_quads;
   if (args.num_quads_dev) {
      uint32_t exact = *(const uint32_t *)(uintptr_t)args.num_quads_dev;
      if (exact < nq)
         nq = exact;
   }
   if (iq >= nq)
      return 0;

   uint32_t list_base = args.quad_list_base_dev
      ? *(const uint32_t *)(uintptr_t)args.quad_list_base_dev
      : args.quad_list_base;
   uint32_t q = args.quad_list
      ? ((const uint32_t *)(uintptr_t)args.quad_list)[list_base + iq] : iq;
   uint32_t block = ((const uint32_t *)(uintptr_t)args.abuf_quad_block)[q];
   uint32_t gprim = ((const uint32_t *)(uintptr_t)args.abuf_quad_prim)[q];
   if (!cp_resolve_seg_range(&args, gprim))
      return 0;

   uint32_t prim = gprim - args.abuf_prim_base;
   uint32_t qx = (block % args.quad_width) * 2;
   uint32_t qy = (block / args.quad_width) * 2;
   uint32_t x = qx + (lane & 1u);
   uint32_t y = qy + (lane >> 1);
   bool in_fb = x < args.width && y < args.height;
   uint32_t pixel = (y < args.height ? y : args.height - 1) * args.width +
                    (x < args.width ? x : args.width - 1);
   uint32_t mask = ((const unsigned char *)(uintptr_t)args.abuf_quad_mask)[q];

   uint32_t *rows = (uint32_t *)(uintptr_t)args.out_batch_rows;
   if (rows) {
      uint32_t row = 0;
      if (args.num_draw_slices > 1) {
         const struct cp_draw_slice *s =
            (const struct cp_draw_slice *)(uintptr_t)args.draw_slices;
         uint32_t vert = (prim >> args.prim_shift) * 3u;
         uint32_t lo = 0, hi = args.num_draw_slices - 1;
         while (lo < hi) {
            uint32_t mid = (lo + hi + 1u) >> 1;
            if (s[mid].vert_begin <= vert)
               lo = mid;
            else
               hi = mid - 1;
         }
         row = lo;
      }
      rows[slot] = row + args.row_base;
   }

   struct cp_interp_tri tri;
   bool tri_ok = false;
   if (lane == 0)
      tri_ok = cp_interp_setup(&args, prim, &tri);
   unsigned src_lane = threadIdx.x & ~3u;
#define CP_QUAD_BCAST(field) \
   tri.field = __shfl_sync(0xFFFFFFFFu, tri.field, src_lane)
   CP_QUAD_BCAST(sx0); CP_QUAD_BCAST(sy0);
   CP_QUAD_BCAST(sx1); CP_QUAD_BCAST(sy1);
   CP_QUAD_BCAST(sx2); CP_QUAD_BCAST(sy2);
   CP_QUAD_BCAST(ndc_z0); CP_QUAD_BCAST(ndc_z1); CP_QUAD_BCAST(ndc_z2);
   CP_QUAD_BCAST(inv_w0); CP_QUAD_BCAST(inv_w1); CP_QUAD_BCAST(inv_w2);
   CP_QUAD_BCAST(inv_area);
   CP_QUAD_BCAST(vidx1); CP_QUAD_BCAST(vidx2);
   uint32_t base_lo = __shfl_sync(0xFFFFFFFFu, (uint32_t)tri.base, src_lane);
   uint32_t base_hi = __shfl_sync(0xFFFFFFFFu,
                                  (uint32_t)(tri.base >> 32), src_lane);
   tri.base = (uint64_t)base_lo | ((uint64_t)base_hi << 32);
   tri.front = __shfl_sync(0xFFFFFFFFu, (int)tri.front, src_lane) != 0;
   tri_ok = __shfl_sync(0xFFFFFFFFu, (int)tri_ok, src_lane) != 0;
#undef CP_QUAD_BCAST
   bool ok = tri_ok &&
             cp_interp_pixel_prepared(&args, prim, pixel, slot, &tri);
   if (args.coverage)
      ((unsigned char *)(uintptr_t)args.coverage)[slot] =
         (ok && in_fb && (mask & (1u << lane))) ? 1u : 0u;
   return ok ? 1 : 0;
}

template <bool RANGES>
static __device__ __forceinline__ void
cp_abuf_interpolate_body(struct cp_fs_interp_args args)
{
   uint32_t iq = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t nq = args.abuf_num_quads;
   if (args.num_quads_dev) {
      uint32_t d = *(const uint32_t *)(uintptr_t)args.num_quads_dev;
      if (d < nq)
         nq = d;
   }
   if (iq >= nq)
      return;

   /* Pass-episode mode shades one segment's quads densely: thread i takes
    * the i-th quad of this segment's slice of the grouped list and shading
    * slots 4i..4i+3; outside an episode the list is null and q == i. The
    * quad stream's primitive ids are episode-global, so the segment's base
    * is subtracted before its own vertex stream and slices are addressed. */
   uint32_t list_base = args.quad_list_base_dev
      ? *(const uint32_t *)(uintptr_t)args.quad_list_base_dev
      : args.quad_list_base;
   uint32_t q = args.quad_list
      ? ((const uint32_t *)(uintptr_t)args.quad_list)[list_base + iq] : iq;

   uint32_t b = ((const uint32_t *)(uintptr_t)args.abuf_quad_block)[q];
   uint32_t gprim = ((const uint32_t *)(uintptr_t)args.abuf_quad_prim)[q];

   /*
    * A merged group's launch spans segments, so the launch-wide vertex
    * stream, slice table and bases are placeholders; resolve this quad's own
    * through the range table — the same last-base-not-past search the quad
    * bucketing runs — into the by-value argument copy, and everything below
    * reads as it always did. vs_out is the same buffer as positions on this
    * path, exactly as the per-segment shade passes them.
    */
   if (RANGES && !cp_resolve_seg_range(&args, gprim))
      return;

   uint32_t prim = gprim - args.abuf_prim_base;
   uint32_t mask = ((const unsigned char *)(uintptr_t)args.abuf_quad_mask)[q];

   uint32_t qx = (b % args.quad_width) * 2;
   uint32_t qy = (b / args.quad_width) * 2;
   uint32_t base = iq * 4u;

   cp_write_batch_rows(&args, prim, base);

   unsigned char *coverage = (unsigned char *)(uintptr_t)args.coverage;
   struct cp_interp_tri interp_tri;
   bool tri_ok = cp_interp_setup(&args, prim, &interp_tri);
#if CP_ABUF_INSTRUMENT
   uint32_t *dbg_slot = (uint32_t *)(uintptr_t)args.dbg_slot;
#endif

   for (int i = 0; i < 4; i++) {
      uint32_t x = qx + (i & 1);
      uint32_t y = qy + (i >> 1);
      bool in_fb = x < args.width && y < args.height;
      /* Clamped the way cp_fs_interpolate clamps, so an odd-sized framebuffer
       * still shades a whole quad and the two agree on which pixel a corner
       * outside it borrows. */
      uint32_t pixel = (y < args.height ? y : args.height - 1) * args.width +
                       (x < args.width ? x : args.width - 1);
      bool covered = in_fb && (mask & (1u << i));

      bool ok = tri_ok && cp_interp_pixel_prepared(
         &args, prim, pixel, base + i, &interp_tri);
      /* Single-sampled only, so a covered pixel wins sample 0 and nothing
       * else; the host refuses this path for anything else. */
      if (coverage)
         coverage[base + i] = (covered && ok) ? 1u : 0u;
#if CP_ABUF_INSTRUMENT
      if (dbg_slot)
         dbg_slot[base + i] = (covered && ok)
            ? cp_abuf_slot_for(&args, pixel, gprim) : 0xFFFFFFFFu;
#endif
   }
}

extern "C" __global__ void
cp_abuf_interpolate(struct cp_fs_interp_args args)
{
   cp_abuf_interpolate_body<false>(args);
}

extern "C" __global__ void
cp_abuf_interpolate_ranges(struct cp_fs_interp_args args)
{
   cp_abuf_interpolate_body<true>(args);
}

#if CP_ABUF_INSTRUMENT
/*
 * Verification only (CUDAPIPE_ABUFFER_VERIFY). Deposit each shaded fragment's
 * colour in the A-buffer slot for its (pixel, primitive), so that the peel path
 * and the
 * quad-stream path — which shade in completely different orders — can be
 * compared element by element.
 *
 * Helper lanes carry ~0 and are skipped: both paths shade them and both drop
 * them. `writes` counts rather than flags, so a slot claimed twice is visible
 * instead of looking like agreement.
 */
extern "C" __global__ void
cp_abuf_scatter_colors(uint64_t fs_out, uint32_t fs_out_stride,
                       const uint32_t *dbg_slot, const uint32_t *counter,
                       uint32_t max_slots, uint32_t capacity,
                       float4 *colors, uint32_t *writes, uint32_t *dbg)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t limit = counter ? *counter : max_slots;
   if (limit > max_slots)
      limit = max_slots;
   if (i >= limit)
      return;

   uint32_t slot = dbg_slot[i];
   if (slot == 0xFFFFFFFFu)
      return;
   if (slot >= capacity) {
      atomicAdd(dbg + CP_ABUF_DBG_SLOT_BAD, 1u);
      return;
   }

   colors[slot] = *(const float4 *)((const char *)(uintptr_t)fs_out +
                                    (size_t)i * fs_out_stride);
   atomicAdd(writes + slot, 1u);
}
#endif /* CP_ABUF_INSTRUMENT */

extern "C" __global__ void
cp_fs_interpolate(struct cp_fs_interp_args args)
{
   uint32_t quad = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t quad_h = (args.height + 1) / 2;
   if (quad >= args.quad_width * quad_h)
      return;

   uint32_t qx = (quad % args.quad_width) * 2;
   uint32_t qy = (quad / args.quad_width) * 2;

   const uint64_t *visbuf = (const uint64_t *)(uintptr_t)args.visbuf;
   uint32_t samples = args.num_samples ? args.num_samples : 1u;
   uint32_t plane = args.width * args.height;
   uint32_t pix[4];
   bool in_fb[4];

   for (int i = 0; i < 4; i++) {
      uint32_t x = qx + (i & 1);
      uint32_t y = qy + (i >> 1);
      in_fb[i] = x < args.width && y < args.height;
      /* Clamp so an odd-sized framebuffer still shades a full quad. */
      pix[i] = (y < args.height ? y : args.height - 1) * args.width +
               (x < args.width ? x : args.width - 1);
   }

   /*
    * A quad must belong to a single triangle. Two triangles meeting inside one
    * 2x2 block would otherwise be differenced against each other, and the
    * derivative at every silhouette and every seam would be meaningless —
    * which costs far more than the seams are worth. Emit one quad per distinct
    * triangle instead; the pixels that triangle does not cover ride along as
    * helpers on the same surface.
    *
    * Multisampling widens the search: a block holds 4 pixels times as many
    * samples, so more triangles can meet inside it, and a triangle counts if
    * it won any one sample.
    */
   uint32_t tris[CP_MAX_BLOCK_TRIS];
   int ntris = 0;
   for (int i = 0; i < 4 && ntris < CP_MAX_BLOCK_TRIS; i++) {
      if (!in_fb[i])
         continue;
      for (uint32_t sm = 0; sm < samples && ntris < CP_MAX_BLOCK_TRIS; sm++) {
         uint64_t entry = visbuf[(size_t)sm * plane + pix[i]];
         if (entry == VISBUF_EMPTY)
            continue;
         /* Complemented so atomicMin favours the last primitive on ties. */
         uint32_t t = ~(uint32_t)(entry & 0xFFFFFFFFu);
         bool seen = false;
         for (int k = 0; k < ntris; k++)
            seen |= (tris[k] == t);
         if (!seen)
            tris[ntris++] = t;
      }
   }

   if (ntris == 0)
      return;

   unsigned char *coverage = (unsigned char *)(uintptr_t)args.coverage;

   for (int t = 0; t < ntris; t++) {
      struct cp_fs_interp_args tri_args = args;
      if (!cp_resolve_seg_range(&tri_args, tris[t]))
         continue;
      uint32_t prim = tris[t] - tri_args.abuf_prim_base;
      uint32_t base = atomicAdd((unsigned int *)(uintptr_t)args.counter, 4u);
      if (base + 4 > args.max_pixels) {
#if CP_ABUF_INSTRUMENT
         /* TEMPORARY: quads lost here are quads the comparison would report as
          * missing from the peel side, so say so rather than let it look like
          * a merge that invented them. */
         if (args.dbg_counters)
            atomicAdd((unsigned int *)(uintptr_t)args.dbg_counters +
                      CP_ABUF_DBG_FULL, (unsigned int)(ntris - t));
#endif
         return;
      }

      cp_write_batch_rows(&tri_args, prim, base);
      struct cp_interp_tri interp_tri;
      bool tri_ok = cp_interp_setup(&tri_args, prim, &interp_tri);

      /* TEMPORARY: the quad's coverage as one 4-bit mask, which is the form
       * the A-buffer merge produces. Costs nothing when compiled out. */
      uint32_t quad_mask = 0;
      bool degenerate = false;

      for (int i = 0; i < 4; i++) {
         /* Which of this pixel's samples this triangle actually won. Zero
          * makes the lane a helper: shaded for its derivatives, dropped by
          * the writeback. */
         uint32_t mask = 0;
         if (in_fb[i]) {
            for (uint32_t sm = 0; sm < samples; sm++) {
               uint64_t entry = visbuf[(size_t)sm * plane + pix[i]];
               if (entry != VISBUF_EMPTY &&
                   (~(uint32_t)(entry & 0xFFFFFFFFu)) == tris[t])
                  mask |= 1u << sm;
            }
         }
         bool ok = tri_ok && cp_interp_pixel_prepared(
            &tri_args, prim, pix[i], base + i, &interp_tri);
         if (coverage)
            coverage[base + i] = ok ? (unsigned char)mask : 0;
         if (mask && !ok)
            degenerate = true;
         if (mask && ok)
            quad_mask |= 1u << i;
#if CP_ABUF_INSTRUMENT
         /* TEMPORARY: where this fragment's shaded colour is to be deposited,
          * so the A-buffer path's colour for the same (pixel, primitive) can
          * be compared against it. A helper lane has no slot. */
         if (args.dbg_slot)
            ((uint32_t *)(uintptr_t)args.dbg_slot)[base + i] =
               (mask && ok) ? cp_abuf_slot_for(&tri_args, pix[i], tris[t])
                            : 0xFFFFFFFFu;
#endif
      }

#if CP_ABUF_INSTRUMENT
      /*
       * TEMPORARY: record the triple this quad is, against the merged quad
       * array. The merge holds one entry per distinct primitive per block,
       * ascending, so the entry for this one is a binary search away; ORing
       * into it accumulates the mask over the passes, because one primitive
       * can win different pixels of a block on different passes.
       */
      if (args.dbg_quad_prim) {
         atomicAdd((unsigned int *)(uintptr_t)args.dbg_counters +
                   CP_ABUF_DBG_PEEL_QUADS, 1u);
         if (degenerate)
            atomicAdd((unsigned int *)(uintptr_t)args.dbg_counters +
                      CP_ABUF_DBG_DEGENERATE, 1u);

         uint32_t lo = ((const uint32_t *)(uintptr_t)args.dbg_blk_offsets)[quad];
         uint32_t cnt = ((const uint32_t *)(uintptr_t)args.dbg_blk_counts)[quad];
         const uint32_t *prims = (const uint32_t *)(uintptr_t)args.dbg_quad_prim;
         uint32_t a = 0, bnd = cnt, found = 0xFFFFFFFFu;
         while (a < bnd) {
            uint32_t mid = a + (bnd - a) / 2;
            uint32_t v = prims[lo + mid];
            if (v == tris[t]) { found = mid; break; }
            if (v < tris[t]) a = mid + 1; else bnd = mid;
         }
         if (found != 0xFFFFFFFFu)
            atomicOr((unsigned int *)(uintptr_t)args.dbg_peel_mask + lo + found,
                     quad_mask);
         else
            atomicAdd((unsigned int *)(uintptr_t)args.dbg_counters +
                      CP_ABUF_DBG_NOT_FOUND, 1u);
      }
#else
      (void)quad_mask;
      (void)degenerate;
#endif
   }
}

/*
 * Fused direct shading, step 1 of 2: compact covered pixels into slots
 * without interpolating them.
 *
 * cp_fs_interpolate above does three jobs per screen quad: find the distinct
 * triangles in the visibility block, allocate four-aligned shading slots for
 * each, and interpolate every slot's varyings. When the interpolation is
 * fused into the generated fragment shader (cp_fs_direct_lane, reached
 * through CP_ARG_SLOT_FUSED_INTERP), only the first two remain a separate
 * launch, because they end in an atomic allocation whose result the shader
 * cannot reproduce. Everything written here is exactly what
 * cp_fs_interpolate writes for the same slot, minus what the lane
 * recomputes: the pixel, the sample mask (which the lane zeroes for a
 * degenerate primitive, the case where the interpolator would have written
 * zero), the batch row -- and the primitive per quad in out_prim_list, the
 * one thing the lane cannot recover from the visibility buffer, because a
 * helper lane's pixel does not name the triangle it borrows.
 *
 * The instrumented records (CP_ABUF_INSTRUMENT) are not produced here: the
 * host declines the fused form entirely when the instrumentation is
 * compiled in, so a verification run always sees the peel interpolator's.
 */
extern "C" __global__ void
cp_fs_compact(struct cp_fs_interp_args args)
{
   uint32_t quad = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t quad_h = (args.height + 1) / 2;
   if (quad >= args.quad_width * quad_h)
      return;

   uint32_t qx = (quad % args.quad_width) * 2;
   uint32_t qy = (quad / args.quad_width) * 2;

   const uint64_t *visbuf = (const uint64_t *)(uintptr_t)args.visbuf;
   uint32_t samples = args.num_samples ? args.num_samples : 1u;
   uint32_t plane = args.width * args.height;
   uint32_t pix[4];
   bool in_fb[4];

   for (int i = 0; i < 4; i++) {
      uint32_t x = qx + (i & 1);
      uint32_t y = qy + (i >> 1);
      in_fb[i] = x < args.width && y < args.height;
      /* Clamp so an odd-sized framebuffer still shades a full quad. */
      pix[i] = (y < args.height ? y : args.height - 1) * args.width +
               (x < args.width ? x : args.width - 1);
   }

   /* One quad per distinct triangle, exactly as cp_fs_interpolate emits
    * them; see the comment there for why a quad must not straddle two. */
   uint32_t tris[CP_MAX_BLOCK_TRIS];
   int ntris = 0;
   for (int i = 0; i < 4 && ntris < CP_MAX_BLOCK_TRIS; i++) {
      if (!in_fb[i])
         continue;
      for (uint32_t sm = 0; sm < samples && ntris < CP_MAX_BLOCK_TRIS; sm++) {
         uint64_t entry = visbuf[(size_t)sm * plane + pix[i]];
         if (entry == VISBUF_EMPTY)
            continue;
         /* Complemented so atomicMin favours the last primitive on ties. */
         uint32_t t = ~(uint32_t)(entry & 0xFFFFFFFFu);
         bool seen = false;
         for (int k = 0; k < ntris; k++)
            seen |= (tris[k] == t);
         if (!seen)
            tris[ntris++] = t;
      }
   }

   if (ntris == 0)
      return;

   unsigned char *coverage = (unsigned char *)(uintptr_t)args.coverage;
   uint32_t *pixel_list = (uint32_t *)(uintptr_t)args.pixel_list;
   uint32_t *prim_list = (uint32_t *)(uintptr_t)args.out_prim_list;

   for (int t = 0; t < ntris; t++) {
      struct cp_fs_interp_args tri_args = args;
      if (!cp_resolve_seg_range(&tri_args, tris[t]))
         continue;
      uint32_t prim = tris[t] - tri_args.abuf_prim_base;
      uint32_t base = atomicAdd((unsigned int *)(uintptr_t)args.counter, 4u);
      if (base + 4 > args.max_pixels)
         return;

      cp_write_batch_rows(&tri_args, prim, base);
      prim_list[base >> 2] = tris[t];

      for (int i = 0; i < 4; i++) {
         /* Which of this pixel's samples this triangle actually won. Zero
          * makes the lane a helper: shaded for its derivatives, dropped by
          * the writeback. */
         uint32_t mask = 0;
         if (in_fb[i]) {
            for (uint32_t sm = 0; sm < samples; sm++) {
               uint64_t entry = visbuf[(size_t)sm * plane + pix[i]];
               if (entry != VISBUF_EMPTY &&
                   (~(uint32_t)(entry & 0xFFFFFFFFu)) == tris[t])
                  mask |= 1u << sm;
            }
         }
         pixel_list[base + i] = pix[i];
         if (coverage)
            coverage[base + i] = (unsigned char)mask;
      }
   }
}

/* pipe_blendfactor */
enum {
   CP_BLENDFACTOR_ONE = 1,
   CP_BLENDFACTOR_SRC_COLOR = 2,
   CP_BLENDFACTOR_SRC_ALPHA = 3,
   CP_BLENDFACTOR_DST_ALPHA = 4,
   CP_BLENDFACTOR_DST_COLOR = 5,
   CP_BLENDFACTOR_SRC_ALPHA_SATURATE = 6,
   CP_BLENDFACTOR_CONST_COLOR = 7,
   CP_BLENDFACTOR_CONST_ALPHA = 8,
   CP_BLENDFACTOR_ZERO = 0x11,
   CP_BLENDFACTOR_INV_SRC_COLOR = 0x12,
   CP_BLENDFACTOR_INV_SRC_ALPHA = 0x13,
   CP_BLENDFACTOR_INV_DST_ALPHA = 0x14,
   CP_BLENDFACTOR_INV_DST_COLOR = 0x15,
   CP_BLENDFACTOR_INV_CONST_COLOR = 0x17,
   CP_BLENDFACTOR_INV_CONST_ALPHA = 0x18,
};

/* pipe_blend_func */
enum {
   CP_BLEND_ADD = 0,
   CP_BLEND_SUBTRACT,
   CP_BLEND_REVERSE_SUBTRACT,
   CP_BLEND_MIN,
   CP_BLEND_MAX,
};

static __device__ __forceinline__ float
cp_blend_factor(uint32_t factor, float src, float src_a, float dst, float dst_a)
{
   switch (factor) {
   case CP_BLENDFACTOR_ONE:           return 1.0f;
   case CP_BLENDFACTOR_SRC_COLOR:     return src;
   case CP_BLENDFACTOR_SRC_ALPHA:     return src_a;
   case CP_BLENDFACTOR_DST_ALPHA:     return dst_a;
   case CP_BLENDFACTOR_DST_COLOR:     return dst;
   case CP_BLENDFACTOR_SRC_ALPHA_SATURATE: return fminf(src_a, 1.0f - dst_a);
   case CP_BLENDFACTOR_ZERO:          return 0.0f;
   case CP_BLENDFACTOR_INV_SRC_COLOR: return 1.0f - src;
   case CP_BLENDFACTOR_INV_SRC_ALPHA: return 1.0f - src_a;
   case CP_BLENDFACTOR_INV_DST_ALPHA: return 1.0f - dst_a;
   case CP_BLENDFACTOR_INV_DST_COLOR: return 1.0f - dst;
   default:                           return 1.0f;
   }
}

static __device__ __forceinline__ float
cp_blend_combine(uint32_t func, float src, float dst)
{
   switch (func) {
   case CP_BLEND_SUBTRACT:         return src - dst;
   case CP_BLEND_REVERSE_SUBTRACT: return dst - src;
   case CP_BLEND_MIN:              return fminf(src, dst);
   case CP_BLEND_MAX:              return fmaxf(src, dst);
   default:                        return src + dst;
   }
}

static __device__ __forceinline__ float
cp_linear_to_srgb(float c)
{
   if (c <= 0.0031308f)
      return c * 12.92f;
   return 1.055f * __powf(c, 1.0f / 2.4f) - 0.055f;
}

static __device__ __forceinline__ float
cp_srgb_to_linear_wb(float c)
{
   if (c <= 0.04045f)
      return c * (1.0f / 12.92f);
   return __powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

static __device__ __forceinline__ float
cp_unorm8_to_float(uint32_t v)
{
   return (float)v * (1.0f / 255.0f);
}

static __device__ __forceinline__ uint32_t
cp_float_to_unorm8(float v)
{
   v = fminf(fmaxf(v, 0.0f), 1.0f);
   return (uint32_t)(v * 255.0f + 0.5f);
}

static __device__ __forceinline__ float
cp_half_to_float_wb(unsigned short h)
{
   unsigned sign = (unsigned)(h >> 15) << 31;
   unsigned exp = (h >> 10) & 0x1F;
   unsigned mant = h & 0x3FF;
   unsigned bits;
   if (exp == 0)
      bits = sign;
   else if (exp == 0x1F)
      bits = sign | 0x7F800000u | (mant << 13);
   else
      bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
   return __int_as_float((int)bits);
}

/*
 * The blend equation, once.
 *
 * Both cp_fs_writeback and cp_abuf_composite call this, on the same
 * struct cp_blend_desc the host filled from the draw's pipe_rt_blend_state.
 * Neither of them may hold its own copy of the arithmetic: this driver already
 * carries one pair of routines that drifted apart (CUDAPIPE_HANDOFF.md gap 12)
 * and the whole point of the A-buffer path is that it composites what the peel
 * path composited.
 *
 * `dst` is the attachment's decoded value; `out` may alias neither.
 */
static __device__ __forceinline__ void
cp_blend_resolve(const struct cp_blend_desc *b, const float *src,
                 const float *dst, float *out)
{
   if (b->enable) {
      for (int c = 0; c < 3; c++) {
         float sf = cp_blend_factor(b->rgb_src_factor, src[c], src[3], dst[c], dst[3]);
         float df = cp_blend_factor(b->rgb_dst_factor, src[c], src[3], dst[c], dst[3]);
         out[c] = cp_blend_combine(b->rgb_func, src[c] * sf, dst[c] * df);
      }
      float sfa = cp_blend_factor(b->alpha_src_factor, src[3], src[3], dst[3], dst[3]);
      float dfa = cp_blend_factor(b->alpha_dst_factor, src[3], src[3], dst[3], dst[3]);
      out[3] = cp_blend_combine(b->alpha_func, src[3] * sfa, dst[3] * dfa);

      /* Channels masked out keep the destination value. */
      for (int c = 0; c < 4; c++) {
         if (!(b->colormask & (1u << c)))
            out[c] = dst[c];
      }
   } else {
      for (int c = 0; c < 4; c++)
         out[c] = (b->colormask & (1u << c)) ? src[c] : dst[c];
   }
}

static __device__ void
cp_load_dst(const void *ptr, uint32_t encoding, float *out);

static __device__ __forceinline__ uint32_t
cp_bytes_per_pixel(uint32_t encoding)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT: return 16;
   case CP_COLOR_R16G16B16A16_FLOAT: return 8;
   case CP_COLOR_R16G16_SFLOAT:      return 4;
   case CP_COLOR_R16_SFLOAT:         return 2;
   case CP_COLOR_R8G8_UNORM:         return 2;
   case CP_COLOR_R8_UNORM:           return 1;
   default:                          return 4;
   }
}

static __device__ void
cp_load_dst(const void *ptr, uint32_t encoding, float *out)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT: {
      const float4 v = *(const float4 *)ptr;
      out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
      break;
   }
   case CP_COLOR_R16G16B16A16_FLOAT: {
      const unsigned short *h = (const unsigned short *)ptr;
      out[0] = cp_half_to_float_wb(h[0]); out[1] = cp_half_to_float_wb(h[1]);
      out[2] = cp_half_to_float_wb(h[2]); out[3] = cp_half_to_float_wb(h[3]);
      break;
   }
   case CP_COLOR_R11G11B10_FLOAT: {
      unsigned p = *(const uint32_t *)ptr;
      out[0] = cp_half_to_float_wb((unsigned short)((p & 0x7FF) << 4));
      out[1] = cp_half_to_float_wb((unsigned short)(((p >> 11) & 0x7FF) << 4));
      out[2] = cp_half_to_float_wb((unsigned short)(((p >> 22) & 0x3FF) << 5));
      out[3] = 1.0f;
      break;
   }
   case CP_COLOR_A2B10G10R10_UNORM: {
      uint32_t p = *(const uint32_t *)ptr;
      out[0] = (float)(p & 0x3FF) * (1.0f / 1023.0f);
      out[1] = (float)((p >> 10) & 0x3FF) * (1.0f / 1023.0f);
      out[2] = (float)((p >> 20) & 0x3FF) * (1.0f / 1023.0f);
      out[3] = (float)((p >> 30) & 0x3) * (1.0f / 3.0f);
      break;
   }
   case CP_COLOR_R16_SFLOAT: {
      out[0] = cp_half_to_float_wb(*(const unsigned short *)ptr);
      out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   case CP_COLOR_R16G16_SFLOAT: {
      const unsigned short *h = (const unsigned short *)ptr;
      out[0] = cp_half_to_float_wb(h[0]); out[1] = cp_half_to_float_wb(h[1]);
      out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   case CP_COLOR_R8_UNORM: {
      out[0] = cp_unorm8_to_float(*(const uint8_t *)ptr);
      out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   case CP_COLOR_R8G8_UNORM: {
      const uint8_t *v = (const uint8_t *)ptr;
      out[0] = cp_unorm8_to_float(v[0]);
      out[1] = cp_unorm8_to_float(v[1]);
      out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   default: {
      uint32_t p = *(const uint32_t *)ptr;
      float c0 = cp_unorm8_to_float(p & 0xFF);
      float c1 = cp_unorm8_to_float((p >> 8) & 0xFF);
      float c2 = cp_unorm8_to_float((p >> 16) & 0xFF);
      float c3 = cp_unorm8_to_float((p >> 24) & 0xFF);
      if (encoding == CP_COLOR_B8G8R8A8_UNORM || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         out[0] = c2; out[1] = c1; out[2] = c0; out[3] = c3;
      } else {
         out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
      }
      if (encoding == CP_COLOR_R8G8B8A8_SRGB || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         out[0] = cp_srgb_to_linear_wb(out[0]);
         out[1] = cp_srgb_to_linear_wb(out[1]);
         out[2] = cp_srgb_to_linear_wb(out[2]);
      }
      break;
   }
   }
}

static __device__ __forceinline__ unsigned short
cp_float_to_half(float f)
{
   unsigned bits = __float_as_int(f);
   unsigned sign = (bits >> 16) & 0x8000;
   int exp = ((bits >> 23) & 0xFF) - 127 + 15;
   unsigned mant = bits & 0x7FFFFF;

   if (exp <= 0) {
      return (unsigned short)sign;
   } else if (exp >= 0x1F) {
      return (unsigned short)(sign | 0x7C00);
   }
   return (unsigned short)(sign | (exp << 10) | (mant >> 13));
}

static __device__ void
cp_store_dst(void *ptr, uint32_t encoding, const float *c)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT:
      *(float4 *)ptr = make_float4(c[0], c[1], c[2], c[3]);
      break;
   case CP_COLOR_R16G16B16A16_FLOAT: {
      unsigned short *h = (unsigned short *)ptr;
      h[0] = cp_float_to_half(c[0]); h[1] = cp_float_to_half(c[1]);
      h[2] = cp_float_to_half(c[2]); h[3] = cp_float_to_half(c[3]);
      break;
   }
   case CP_COLOR_R11G11B10_FLOAT: {
      /* 11-bit float: 5-bit exp, 6-bit mantissa; 10-bit: 5-bit exp, 5-bit mantissa */
      unsigned short hr = cp_float_to_half(c[0]);
      unsigned short hg = cp_float_to_half(c[1]);
      unsigned short hb = cp_float_to_half(c[2]);
      unsigned r11 = (hr >> 4) & 0x7FF;
      unsigned g11 = (hg >> 4) & 0x7FF;
      unsigned b10 = (hb >> 5) & 0x3FF;
      *(uint32_t *)ptr = r11 | (g11 << 11) | (b10 << 22);
      break;
   }
   case CP_COLOR_A2B10G10R10_UNORM: {
      float r = fminf(fmaxf(c[0], 0.0f), 1.0f);
      float g = fminf(fmaxf(c[1], 0.0f), 1.0f);
      float b = fminf(fmaxf(c[2], 0.0f), 1.0f);
      float a = fminf(fmaxf(c[3], 0.0f), 1.0f);
      uint32_t p = ((uint32_t)(r * 1023.0f + 0.5f)) |
                   ((uint32_t)(g * 1023.0f + 0.5f) << 10) |
                   ((uint32_t)(b * 1023.0f + 0.5f) << 20) |
                   ((uint32_t)(a * 3.0f + 0.5f) << 30);
      *(uint32_t *)ptr = p;
      break;
   }
   case CP_COLOR_R16_SFLOAT: {
      *(unsigned short *)ptr = cp_float_to_half(c[0]);
      break;
   }
   case CP_COLOR_R16G16_SFLOAT: {
      unsigned short *h = (unsigned short *)ptr;
      h[0] = cp_float_to_half(c[0]); h[1] = cp_float_to_half(c[1]);
      break;
   }
   case CP_COLOR_R8_UNORM: {
      *(uint8_t *)ptr = (uint8_t)cp_float_to_unorm8(c[0]);
      break;
   }
   case CP_COLOR_R8G8_UNORM: {
      uint8_t *v = (uint8_t *)ptr;
      v[0] = (uint8_t)cp_float_to_unorm8(c[0]);
      v[1] = (uint8_t)cp_float_to_unorm8(c[1]);
      break;
   }
   default: {
      float r = c[0], g = c[1], b = c[2];
      if (encoding == CP_COLOR_R8G8B8A8_SRGB || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         r = cp_linear_to_srgb(r);
         g = cp_linear_to_srgb(g);
         b = cp_linear_to_srgb(b);
      }
      uint32_t p;
      if (encoding == CP_COLOR_B8G8R8A8_UNORM || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         p = cp_float_to_unorm8(b) | (cp_float_to_unorm8(g) << 8) |
             (cp_float_to_unorm8(r) << 16) | (cp_float_to_unorm8(c[3]) << 24);
      } else {
         p = cp_float_to_unorm8(r) | (cp_float_to_unorm8(g) << 8) |
             (cp_float_to_unorm8(b) << 16) | (cp_float_to_unorm8(c[3]) << 24);
      }
      *(uint32_t *)ptr = p;
      break;
   }
   }
}

static __device__ __forceinline__ void
cp_fs_writeback_one(const struct cp_fs_writeback_args &args, uint32_t i)
{

   /* A helper lane exists only to supply derivatives to its quad. Without
    * multisampling the mask is 0 or 1; with it, one bit per sample won. */
   uint32_t cov = args.coverage
      ? ((const unsigned char *)(uintptr_t)args.coverage)[i] : 1u;
   if (!cov)
      return;

   uint32_t samples = args.num_samples ? args.num_samples : 1u;
   uint32_t pixel = ((const uint32_t *)(uintptr_t)args.pixel_list)[i];
   uint32_t plane = args.width * args.height;
   /* The lowest sample this fragment won, for the per-pixel bookkeeping that
    * has no per-sample equivalent. */
   uint32_t first_sample = __ffs((int)cov) - 1;

   /* Once a pixel has taken a fragment, later passes of an alpha-tested draw
    * must leave it alone rather than blend into it again. */
   unsigned char *resolved = (unsigned char *)(uintptr_t)args.resolved;
   if (resolved && resolved[pixel])
      return;

   /* A discarded fragment contributes neither colour nor depth. Record which
    * triangle it was so the next pass can pick the one behind it. */
   if (args.discard_mask && ((const unsigned char *)(uintptr_t)args.discard_mask)[i]) {
      if (args.reject && args.reject_pass < args.reject_layers && args.visbuf) {
         uint64_t entry = ((const uint64_t *)(uintptr_t)args.visbuf)
            [(size_t)first_sample * plane + pixel];
         uint32_t tri = ~(uint32_t)(entry & 0xFFFFFFFFu);
         ((uint32_t *)(uintptr_t)args.reject)[(size_t)pixel * args.reject_layers +
                                              args.reject_pass] = tri;
      }
      return;
   }

   if (resolved)
      resolved[pixel] = 1;

   /* This fragment survived the depth test during rasterization, so commit its
    * depth before the next draw tests against it. */
   if (args.depth_write && args.depthbuf && args.visbuf) {
      /* Depth is per sample: only the samples this fragment won advance. */
      for (uint32_t sm = 0; sm < samples; sm++) {
         if (!(cov & (1u << sm)))
            continue;
         size_t at = (size_t)sm * plane + pixel;
         uint32_t key = (uint32_t)(((const uint64_t *)(uintptr_t)args.visbuf)[at] >> 32);
         ((uint32_t *)(uintptr_t)args.depthbuf)[at] =
            args.depth_key_invert ? ~key : key;
      }
   }

   const float4 *fs_out =
      (const float4 *)((const char *)(uintptr_t)args.fs_out +
                       (size_t)i * args.fs_out_stride);
   float src[4] = { fs_out->x, fs_out->y, fs_out->z, fs_out->w };

   uint32_t bpp;
   bpp = cp_bytes_per_pixel(args.color_encoding);
   /*
    * The shader ran once for the pixel, so every sample it covers takes the
    * same colour — that is what per-fragment shading means. Blending reads and
    * writes each sample's own plane, so a fragment covering two of four
    * samples blends into two of them and the resolve weights it accordingly.
    */
   for (uint32_t sm = 0; sm < samples; sm++) {
   if (!(cov & (1u << sm)))
      continue;
   void *dst_ptr = (char *)(uintptr_t)args.color_out +
      (size_t)sm * args.sample_stride + (size_t)pixel * bpp;

   float out[4];
   /* The destination is only read when the equation needs it — an unblended,
    * unmasked write does not, and that is the common case. */
   if (args.blend.enable || args.blend.colormask != 0xF) {
      float dst[4];
      cp_load_dst(dst_ptr, args.color_encoding, dst);
      cp_blend_resolve(&args.blend, src, dst, out);
   } else {
      for (int c = 0; c < 4; c++)
         out[c] = src[c];
   }

   cp_store_dst(dst_ptr, args.color_encoding, out);
   }
}

/*
 * One thread per shaded slot, striding: the slot count lives on the device,
 * so the grid the host launches is a statement of how much machine to use
 * rather than a bound anything depends on — the launch used to be sized to
 * the framebuffer's worst case and spent more time scheduling idle blocks
 * than writing pixels on small draws.
 */
extern "C" __global__ void
cp_fs_writeback(struct cp_fs_writeback_args args)
{
   uint32_t limit = args.pixel_counter
      ? *(const uint32_t *)(uintptr_t)args.pixel_counter
      : args.num_pixels;
   for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < limit;
        i += gridDim.x * blockDim.x)
      cp_fs_writeback_one(args, i);
}

/*
 * Composite an A-buffer into the colour attachment.
 *
 * The peel loop reads and writes the attachment once per layer, because a pass
 * only knows about its own layer. Here the pixel's whole run is in hand, so
 * the attachment is touched twice for the pixel rather than twice for each of
 * its ~400 fragments.
 *
 * The blend is nevertheless carried through the attachment's own encoding at
 * every layer, and that is not an oversight. The peel loop stores each layer
 * and reloads it, so on an 8-bit attachment each layer is quantised before the
 * next one blends against it; carrying full float across the run would give a
 * *different* — arguably better — answer, and the claim this path is verified
 * against is that it produces the peel loop's image bit for bit. The round
 * trip is through the same cp_store_dst/cp_load_dst pair, into 16 bytes of
 * stack, so it costs arithmetic rather than memory traffic.
 */
extern "C" __global__ void
cp_abuf_composite(struct cp_abuf_composite_args args)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= *(const uint32_t *)(uintptr_t)args.list_count)
      return;
   uint32_t pixel = ((const uint32_t *)(uintptr_t)args.list)[i];

   uint32_t n = ((const uint32_t *)(uintptr_t)args.counts)[pixel];
   uint32_t base = ((const uint32_t *)(uintptr_t)args.offsets)[pixel];
   if (!n)
      return;
   if (args.max_layers && n > args.max_layers)
      n = args.max_layers;

   const uint32_t *shade_slot = (const uint32_t *)(uintptr_t)args.shade_slot;
   const unsigned char *coverage =
      (const unsigned char *)(uintptr_t)args.coverage;
   const unsigned char *discard_mask =
      (const unsigned char *)(uintptr_t)args.discard_mask;

   uint32_t bpp = cp_bytes_per_pixel(args.color_encoding);
   void *dst_ptr = (char *)(uintptr_t)args.color_out + (size_t)pixel * bpp;

   /* Single-sampled only — the host refuses this path for anything else — so
    * there is one plane and no per-sample coverage to fold in. */
   float dst[4];
   cp_load_dst(dst_ptr, args.color_encoding, dst);

   bool wrote = false;
   for (uint32_t k = 0; k < n; k++) {
      uint32_t s = base + k;
      if (s >= args.capacity)
         break;
      uint32_t slot = shade_slot[s];
      /* A slot the merge could not place. It cannot be read, and dropping the
       * rest of the run with it would compose the layers out of order. */
      if (slot == 0xFFFFFFFFu)
         continue;

      /*
       * Pass-episode resolution: shading ran per segment into dense arrays,
       * so the global slot — quad * 4 + lane — resolves through the quad's
       * segment and dense position to that segment's own buffers. Outside an
       * episode the launch-wide arrays below stand.
       */
      const char *fs_out_base = (const char *)(uintptr_t)args.fs_out;
      uint32_t fs_out_stride = args.fs_out_stride;
      const unsigned char *cov = coverage;
      const unsigned char *dis = discard_mask;
      if (args.seg_desc) {
         uint32_t qd = slot >> 2u;
         uint32_t seg = ((const unsigned char *)(uintptr_t)args.quad_seg)[qd];
         const struct cp_seg_desc *sd =
            &((const struct cp_seg_desc *)(uintptr_t)args.seg_desc)[seg];
         uint32_t dense = sd->global_slots ? slot :
            ((const uint32_t *)(uintptr_t)args.quad_dense)[qd] * 4u +
            (slot & 3u);
         if (dense >= sd->num_slots)
            continue;
         slot = dense;
         fs_out_base = (const char *)(uintptr_t)sd->fs_out;
         fs_out_stride = sd->fs_out_stride;
         cov = (const unsigned char *)(uintptr_t)sd->coverage;
         dis = (const unsigned char *)(uintptr_t)sd->discard;
      } else if (slot >= args.num_slots) {
         continue;
      }

      /* What cp_fs_writeback drops before blending: a lane the interpolation
       * refused, and a fragment the shader discarded. Neither contributes
       * colour, and on this path neither contributes depth either, since a
       * draw that writes depth is not eligible. */
      if (cov && !cov[slot])
         continue;
      if (dis && dis[slot])
         continue;

      const float4 *fs_out =
         (const float4 *)(fs_out_base + (size_t)slot * fs_out_stride);
      float src[4] = { fs_out->x, fs_out->y, fs_out->z, fs_out->w };

      float out[4];
      cp_blend_resolve(&args.blend, src, dst, out);

      /* Quantise exactly where the peel loop quantises: through the
       * attachment's encoding, into 16 bytes of stack rather than into the
       * attachment itself. The union is float4-headed so that the widest
       * encoding's 16-byte store lands on something aligned for it. */
      union { float4 align; unsigned char b[16]; } tmp;
      cp_store_dst(tmp.b, args.color_encoding, out);
      cp_load_dst(tmp.b, args.color_encoding, dst);
      wrote = true;
   }

   if (wrote)
      cp_store_dst(dst_ptr, args.color_encoding, dst);
}

/*
 * Resolve a multisample attachment: the average of its sample planes.
 *
 * Averaging the decoded values rather than the packed bytes, because an sRGB
 * attachment has to average in linear light — meaning the encoded bytes would
 * darken exactly the edges the multisampling is there to smooth.
 */
extern "C" __global__ void
cp_resolve_samples(struct cp_resolve_msaa_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= args.width || y >= args.height)
      return;

   uint32_t bpp = cp_bytes_per_pixel((uint32_t)args.encoding);
   uint32_t n = args.num_samples ? args.num_samples : 1u;
   float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

   for (uint32_t s = 0; s < n; s++) {
      const void *px = (const char *)(uintptr_t)args.src +
         (size_t)s * args.sample_stride +
         (size_t)y * args.src_stride + (size_t)x * bpp;
      float c[4];
      cp_load_dst(px, (uint32_t)args.encoding, c);
      for (int k = 0; k < 4; k++)
         sum[k] += c[k];
   }

   float inv = 1.0f / (float)n;
   for (int k = 0; k < 4; k++)
      sum[k] *= inv;

   void *out = (char *)(uintptr_t)args.dst +
      (size_t)y * args.dst_stride + (size_t)x * bpp;
   cp_store_dst(out, (uint32_t)args.encoding, sum);
}

static __device__ __forceinline__ uint32_t
cp_float_to_ufloat(float value, unsigned mantissa_bits)
{
   uint32_t bits = __float_as_uint(value);
   bool negative = (bits >> 31) != 0;
   int exponent = (int)((bits >> 23) & 0xff) - 127;
   uint32_t mantissa = bits & 0x7fffff;
   uint32_t mantissa_mask = (1u << mantissa_bits) - 1u;

   if (exponent == 128) {
      if (negative && !mantissa)
         return 0;
      return (31u << mantissa_bits) | (mantissa ? 1u : 0u);
   }
   if (negative || value == 0.0f)
      return 0;

   float max_value = mantissa_bits == 6 ? 65024.0f : 64512.0f;
   if (value > max_value)
      return (30u << mantissa_bits) | mantissa_mask;

   if (exponent > -15) {
      int rounded = __float2int_rn(ldexpf(value,
                                          (int)mantissa_bits - exponent));
      if (rounded >= (2 << mantissa_bits)) {
         rounded >>= 1;
         exponent++;
      }
      return ((uint32_t)(exponent + 15) << mantissa_bits) |
             ((uint32_t)rounded & mantissa_mask);
   }

   int rounded = __float2int_rn(ldexpf(value, (int)mantissa_bits + 14));
   if ((unsigned)rounded >> mantissa_bits)
      return 1u << mantissa_bits;
   return (uint32_t)rounded;
}

static __device__ __forceinline__ void
cp_store_r11g11b10_exact(void *ptr, const float *color)
{
   uint32_t r = cp_float_to_ufloat(color[0], 6);
   uint32_t g = cp_float_to_ufloat(color[1], 6);
   uint32_t b = cp_float_to_ufloat(color[2], 5);
   *(uint32_t *)ptr = r | (g << 11) | (b << 22);
}

extern "C" __global__ void
cp_blit_linear(struct cp_blit_linear_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   uint32_t layer = blockIdx.z;
   if (x >= args.dst_width || y >= args.dst_height || layer >= args.layers)
      return;

   uint32_t src_bpp = cp_bytes_per_pixel(args.src_encoding);
   uint32_t dst_bpp = cp_bytes_per_pixel(args.dst_encoding);
   const char *src = (const char *)(uintptr_t)args.src +
                     (size_t)layer * args.src_layer_stride;
   char *dst = (char *)(uintptr_t)args.dst +
               (size_t)layer * args.dst_layer_stride;
   float out[4];

   if (args.filter_linear) {
      float fx = ((float)x + 0.5f) * (float)args.src_width /
                 (float)args.dst_width - 0.5f;
      float fy = ((float)y + 0.5f) * (float)args.src_height /
                 (float)args.dst_height - 0.5f;
      int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
      float wx = fx - (float)x0, wy = fy - (float)y0;
      int x1 = x0 + 1, y1 = y0 + 1;
      x0 = x0 < 0 ? 0 : (x0 >= (int)args.src_width ? (int)args.src_width - 1 : x0);
      x1 = x1 < 0 ? 0 : (x1 >= (int)args.src_width ? (int)args.src_width - 1 : x1);
      y0 = y0 < 0 ? 0 : (y0 >= (int)args.src_height ? (int)args.src_height - 1 : y0);
      y1 = y1 < 0 ? 0 : (y1 >= (int)args.src_height ? (int)args.src_height - 1 : y1);
      float c00[4], c10[4], c01[4], c11[4];
      cp_load_dst(src + (size_t)y0 * args.src_stride + (size_t)x0 * src_bpp,
                  args.src_encoding, c00);
      cp_load_dst(src + (size_t)y0 * args.src_stride + (size_t)x1 * src_bpp,
                  args.src_encoding, c10);
      cp_load_dst(src + (size_t)y1 * args.src_stride + (size_t)x0 * src_bpp,
                  args.src_encoding, c01);
      cp_load_dst(src + (size_t)y1 * args.src_stride + (size_t)x1 * src_bpp,
                  args.src_encoding, c11);
      for (unsigned c = 0; c < 4; c++) {
         float top = c00[c] + (c10[c] - c00[c]) * wx;
         float bottom = c01[c] + (c11[c] - c01[c]) * wx;
         out[c] = top + (bottom - top) * wy;
      }
   } else {
      uint32_t sx = min((uint32_t)(((uint64_t)(2 * x + 1) * args.src_width) /
                                   (2 * args.dst_width)), args.src_width - 1);
      uint32_t sy = min((uint32_t)(((uint64_t)(2 * y + 1) * args.src_height) /
                                   (2 * args.dst_height)), args.src_height - 1);
      cp_load_dst(src + (size_t)sy * args.src_stride + (size_t)sx * src_bpp,
                  args.src_encoding, out);
   }

   void *dst_pixel = dst + (size_t)y * args.dst_stride + (size_t)x * dst_bpp;
   if (args.dst_encoding == CP_COLOR_R11G11B10_FLOAT)
      cp_store_r11g11b10_exact(dst_pixel, out);
   else
      cp_store_dst(dst_pixel, args.dst_encoding, out);
}
