/*
 * Fragment interpolation shared by the NVRTC kernels and the LLVM-bitcode
 * inline helper.  Callers provide the small compiler-specific spelling layer
 * below; the math and dataflow remain one owned implementation.
 */
#ifndef CP_FS_INTERP_H
#define CP_FS_INTERP_H

#ifndef CP_INTERP_INLINE
#error "CP_INTERP_INLINE must name the device always-inline qualifier"
#endif
#ifndef CP_FMUL_RN
#error "CP_FMUL_RN must provide an uncontracted round-to-nearest multiply"
#endif
#ifndef CP_FSUB_RN
#error "CP_FSUB_RN must provide an uncontracted round-to-nearest subtract"
#endif
#ifndef CP_MAKE_FLOAT4
#error "CP_MAKE_FLOAT4 must construct float4"
#endif

/* Same form as edge_function in cp_rasterize.cu, and it has to stay the same:
 * the sign of the area decides whether the triangle's vertices are swapped,
 * and if the two kernels disagreed on a sliver the varyings here would be
 * fetched in a different order than the one the depth was interpolated in. */
CP_INTERP_INLINE float
cp_edge(float ax, float ay, float bx, float by, float px, float py)
{
   return CP_FSUB_RN(CP_FMUL_RN(bx - px, ay - py),
                    CP_FMUL_RN(by - py, ax - px));
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

CP_INTERP_INLINE bool
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

CP_INTERP_INLINE bool
cp_interp_pixel_prepared(struct cp_fs_interp_args *args, uint32_t tri_id,
                         uint32_t pixel, uint32_t slot,
                         const struct cp_interp_tri *tri,
                         float4 *local_fs_in, uint32_t live_slots,
                         uint32_t num_inputs, int32_t pntc_input,
                         int32_t pos_input)
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
   fc.z = CP_WINDOW_DEPTH(b0 * tri->ndc_z0 + b1 * tri->ndc_z1 +
                          b2 * tri->ndc_z2,
                          args->depth_scale, args->depth_translate);
   fc.w = persp0 + persp1 + persp2;
   if (args->frag_coord)
      ((float4 *)(uintptr_t)args->frag_coord)[slot] = fc;

   const char *vs_out = (const char *)(uintptr_t)tri->base;
   char *fs_in = local_fs_in ? (char *)local_fs_in :
      (char *)(uintptr_t)args->fs_in + (size_t)slot * args->fs_in_stride;

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
#ifdef CP_INTERP_FIXED_SLOTS
      #pragma clang loop unroll(full)
      for (uint32_t i = 0; i < CP_MAX_FS_INPUTS; i++) {
         if (i >= num_inputs || !(live_slots & (1u << i)))
            continue;
#else
      for (uint32_t i = 0; i < num_inputs && i < CP_MAX_FS_INPUTS; i++) {
#endif
         float4 value = CP_MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
         int32_t src = args->input_vs_slot[i];
         if (src >= 0)
            value = *(const float4 *)(vs_out +
               (size_t)src * 16);
         *(float4 *)(fs_in + i * 16) = value;
      }
#ifdef CP_INTERP_FIXED_SLOTS
      (void)live_slots;
#endif

      if (pntc_input >= 0 &&
          (uint32_t)pntc_input < num_inputs) {
         float size = 1.0f;
         if (args->psiz_slot >= 0)
            size = ((const float4 *)vs_out)[args->psiz_slot].x;
         if (!(size > 0.0f))
            size = 1.0f;
         if (size > CP_MAX_POINT_SIZE)
            size = CP_MAX_POINT_SIZE;

         float px0 = tri->sx0 - size * 0.5f;
         float py0 = tri->sy0 - size * 0.5f;
         *(float4 *)(fs_in + pntc_input * 16) =
            CP_MAKE_FLOAT4((cx - px0) / size, (cy - py0) / size, 0.0f, 1.0f);
      }

      /* gl_FragCoord — see pos_input. */
      if (pos_input >= 0 &&
          (uint32_t)pos_input < num_inputs)
         *(float4 *)(fs_in + pos_input * 16) = fc;
      return true;
   }

#ifdef CP_INTERP_FIXED_SLOTS
   #pragma clang loop unroll(full)
   for (uint32_t i = 0; i < CP_MAX_FS_INPUTS; i++) {
      if (i >= num_inputs || !(live_slots & (1u << i)))
         continue;
#else
   for (uint32_t i = 0; i < num_inputs && i < CP_MAX_FS_INPUTS; i++) {
#endif
      int32_t src = args->input_vs_slot[i];
      float4 value = CP_MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 1.0f);

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
   if (pos_input >= 0 &&
       (uint32_t)pos_input < num_inputs)
      *(float4 *)(fs_in + pos_input * 16) = fc;
   return true;
}

CP_INTERP_INLINE bool
cp_interp_pixel(struct cp_fs_interp_args *args, uint32_t tri_id,
                uint32_t pixel, uint32_t slot)
{
   struct cp_interp_tri tri;
   return cp_interp_setup(args, tri_id, &tri) &&
          cp_interp_pixel_prepared(args, tri_id, pixel, slot, &tri,
                                   (float4 *)0, 0xFFFFu,
                                   args->num_fs_inputs, args->pntc_input,
                                   args->pos_input);
}

CP_INTERP_INLINE bool
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

#endif /* CP_FS_INTERP_H */
