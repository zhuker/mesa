/*
 * The per-lane vertex fetch, shared by two compilers.
 *
 * `kernels/cp_vertex_fetch.cu` compiles it with NVRTC as the body of the
 * standalone kernel; `kernels/cp_vs_inline.c` compiles it with clang to
 * NVPTX bitcode that is linked into the generated vertex shader and inlined
 * away. Both must produce bit-identical results for the same input, which is
 * why the half decode below stays hand-written rather than becoming an LLVM
 * conversion: the software decode is the definition.
 *
 * This is a move, not a rewrite. Every expression is the one that was in
 * cp_vertex_fetch.cu; only the destination of the results changed, so that a
 * caller can keep them in registers instead of storing them.
 */
#ifndef CP_VF_LANE_H
#define CP_VF_LANE_H

#include "cp_rast_types.h"

#ifndef CP_VF_INLINE
#define CP_VF_INLINE __device__ static inline
#endif

/*
 * The loops that write the caller's slots are bounded by the format's own
 * maximum -- four components, four bytes each, sixteen bytes of slot -- and
 * skip on the runtime count inside rather than breaking out of the loop. A
 * fixed bound with no data-dependent exit is what lets the fused build unroll
 * them into stores at constant offsets, which is what lets SROA promote the
 * caller's array into registers. cp_fs_interp.h's CP_INTERP_FIXED_SLOTS is the
 * same idiom for the same reason.
 */
#ifndef CP_VF_UNROLL
#define CP_VF_UNROLL _Pragma("unroll")
#endif

CP_VF_INLINE uint32_t
cp_vf_expand(uint32_t raw, uint32_t bytes, uint32_t conv)
{
   uint32_t bits = bytes * 8;
   if (bits >= 32)
      return raw;

   int32_t sext = (int32_t)(raw << (32 - bits)) >> (32 - bits);
   float f;

   switch (conv) {
   case CP_VF_CONV_UINT:
      return raw;
   case CP_VF_CONV_SINT:
      return (uint32_t)sext;
   case CP_VF_CONV_UNORM:
      f = (float)raw / (float)((1u << bits) - 1u);
      break;
   case CP_VF_CONV_SNORM:
      /* The most negative value maps to -1.0 and so does the one below it,
       * which is why this clamps rather than dividing by 2^(bits-1). */
      f = (float)sext / (float)((1u << (bits - 1)) - 1u);
      if (f < -1.0f)
         f = -1.0f;
      break;
   case CP_VF_CONV_USCALED:
      f = (float)raw;
      break;
   case CP_VF_CONV_SSCALED:
      f = (float)sext;
      break;
   case CP_VF_CONV_FLOAT16: {
      uint32_t h = raw & 0xffff;
      uint32_t sign = (h & 0x8000u) << 16;
      uint32_t exp = (h >> 10) & 0x1f;
      uint32_t man = h & 0x3ffu;

      if (exp == 0) {
         if (man == 0)
            return sign;                      /* +-0 */
         /* Subnormal: normalise into the float exponent range. */
         uint32_t shift = 0;
         do {
            man <<= 1;
            shift++;
         } while (!(man & 0x400u));
         man &= 0x3ffu;
         return sign | ((127u - 15u - shift + 1u) << 23) | (man << 13);
      }
      if (exp == 31)
         return sign | 0x7f800000u | (man << 13);   /* inf / nan */
      return sign | ((exp + 127u - 15u) << 23) | (man << 13);
   }
   default:
      return raw;
   }

   return __float_as_uint(f);
}

/*
 * Resolve one assembled vertex's identity: which merged draw it belongs to,
 * its vertex id and its instance id. Returns 0 when this lane has no vertex,
 * which is the seam a bounds-checked or robust-access fetch would use; today
 * the only decline is the count check.
 *
 * Split from the gather below because the fused shader emits the gather one
 * element at a time with the element index a compile-time constant: a loop
 * over the elements indexes the caller's slot array dynamically, and a
 * dynamically indexed alloca is local memory, which is the whole thing this
 * iteration exists to avoid. The identity part has no such index and runs
 * once either way.
 */
CP_VF_INLINE int
cp_vf_lane_ids(const struct cp_vertex_fetch_args *ap, uint32_t v,
               uint32_t *out_vertex_id, uint32_t *out_instance_id,
               uint32_t *out_row)
{
   if (v >= ap->num_verts)
      return 0;

   /* Determine the actual vertex index for this assembled vertex */
   uint32_t vertex_id;
   uint32_t instance_id;
   /* Which merged draw this vertex belongs to; stays 0 on the refs path,
    * which never coexists with a batch. The gather below indexes the
    * per-draw base table with it. */
   uint32_t row = 0;

   if (ap->vertex_ids) {
      /* Pre-expanded topology (strip/fan/points): read from the refs array */
      vertex_id = ((const uint32_t *)(uintptr_t)ap->vertex_ids)[v];
      instance_id = ((const uint32_t *)(uintptr_t)ap->instance_ids)[v];
   } else {
      /*
       * TRIANGLE_LIST: the ids follow from the thread index alone. Every
       * instance replays the same index range, so the vertex within the
       * instance is v modulo the instance's vertex count and the instance is
       * the quotient. verts_per_instance is num_verts for a single-instance
       * draw, which leaves this the plain v it was before.
       */
      uint32_t local = v;
      uint32_t first_vertex = ap->first_vertex;
      const char *ib = (const char *)(uintptr_t)ap->index_buffer;

      /*
       * A batch concatenates the draws it merged, and they need not share an
       * index range: this thread's draw is the last one that starts at or
       * before it. Binary search, over a table bounded by the batch cap and so
       * a few hundred bytes at most. Nothing here runs for an unbatched draw —
       * num_draw_slices is zero and every value above stands.
       */
      uint32_t verts_per_instance = ap->verts_per_instance;
      if (ap->num_draw_slices) {
         const struct cp_draw_slice *sl =
            (const struct cp_draw_slice *)(uintptr_t)ap->draw_slices;
         uint32_t lo = 0, hi = ap->num_draw_slices - 1;
         while (lo < hi) {
            uint32_t mid = (lo + hi + 1) >> 1;
            if (sl[mid].vert_begin <= v)
               lo = mid;
            else
               hi = mid - 1;
         }
         row = lo;
         local = v - sl[row].vert_begin;
         first_vertex = sl[row].first_vertex;
         ib += sl[row].index_bytes;
         /* A batch carries the per-draw instance shape here; the launch-wide
          * scalar cannot describe more than one draw and is passed zero. */
         verts_per_instance = sl[row].verts_per_instance;
      }

      instance_id = 0;
      if (verts_per_instance) {
         instance_id = local / verts_per_instance;
         local = local % verts_per_instance;
      }

      if (ap->index_buffer && ap->index_size > 0) {
         if (ap->index_size == 2)
            vertex_id = (uint32_t)((const unsigned short *)ib)[local] + first_vertex;
         else
            vertex_id = ((const uint32_t *)ib)[local] + first_vertex;
      } else {
         vertex_id = local + first_vertex;
      }
   }

   /* The results the caller publishes: the classic kernel stores them into the
    * per-vertex arrays, the fused shader keeps them in registers. */
   *out_vertex_id = vertex_id;
   *out_instance_id = instance_id;
   *out_row = row;

   return 1;
}

/*
 * Gather one attribute of one vertex into its 16-byte slot.
 *
 * `e` is a compile-time constant in the fused build -- the shader emits one
 * call per input slot it actually reads -- so every table lookup and every
 * store lands at a constant offset and the caller's array is promotable to
 * registers. The standalone kernel calls this in a runtime loop over the
 * bound elements, which is exactly the loop it always ran.
 *
 * An element the pipeline does not bind, or one whose vertex buffer is null,
 * leaves the slot reading zero: the classic caller's buffer was memset, and
 * the fused caller's registers are zeroed here, because a shader that
 * declares more inputs than the pipeline binds must not read whatever was in
 * a register.
 */
CP_VF_INLINE void
cp_vf_lane_element(const struct cp_vertex_fetch_args *ap, uint32_t e,
                   uint32_t vertex_id, uint32_t instance_id, uint32_t row,
                   unsigned char *dst_slot)
{
   char *dst = (char *)dst_slot;
#ifdef CP_VF_FIXED_SLOTS
   /* Nothing zero-filled these; they are registers. */
   ((uint32_t *)dst)[0] = 0;
   ((uint32_t *)dst)[1] = 0;
   ((uint32_t *)dst)[2] = 0;
   ((uint32_t *)dst)[3] = 0;
   if (e >= ap->num_elements)
      return;
#endif

   /* A batch carries one row of per-element base addresses per merged
    * draw, so draws bound to different vertex buffers merge; without one
    * the launch-wide bases stand, which is the unbatched path unchanged. */
   uint64_t vb_base = ap->elem_bases
      ? ((const uint64_t *)(uintptr_t)ap->elem_bases)
           [row * CP_VB_TABLE_STRIDE + e]
      : ap->vb_bases[ap->elem_vb_idx[e]];
   if (!vb_base)
      return;

   uint32_t index = ap->elem_instance_divisor[e]
      ? ap->start_instance + instance_id / ap->elem_instance_divisor[e]
      : vertex_id;

   const char *src = (const char *)(uintptr_t)vb_base +
                     (uint64_t)index * ap->elem_src_stride[e] +
                     ap->elem_src_offset[e];

   /* Copy attribute bytes into the 16-byte slot. The output was zero-filled
    * by the caller (scratch arena is zeroed on grow). */
   uint32_t size = ap->elem_attr_size[e];
   if (size > 16)
      size = 16;

   /*
    * Vulkan only requires a vertex attribute to be aligned to its component
    * size, so a vec2 may sit on a 4 byte boundary and an 8 bit format on any
    * byte at all. A float4/float2 load on those faults the kernel with
    * CUDA_ERROR_MISALIGNED_ADDRESS, so use the widest unit the source
    * address actually allows. dst is always 16 byte aligned.
    */
   if (ap->elem_conv[e] != CP_VF_CONV_COPY32) {
      /*
       * Narrower than 32 bits per component. The shader addresses component
       * c of attribute e at e * 16 + c * 4 whatever the format, so each
       * component has to be widened into its own slot — copying the bytes
       * would leave them packed in the first one.
       */
      uint32_t nch = ap->elem_nr_chan[e];
      uint32_t cb = ap->elem_chan_bytes[e];
      uint32_t conv = ap->elem_conv[e];
      uint32_t swz = ap->elem_swizzle[e];
      uint32_t *d32 = (uint32_t *)dst;

      /*
       * The loops that write the slot are bounded by the format's own maximum
       * -- four components, four bytes each, sixteen bytes of slot -- and skip
       * on the runtime count inside rather than exiting early. A fixed bound
       * with no data-dependent exit is what lets these unroll into stores at
       * constant offsets; the bytes that land are the same either way.
       */
      CP_VF_UNROLL
      for (uint32_t c = 0; c < 4; c++) {
         if (c >= nch)
            continue;
         uint32_t sc = (swz >> (c * 4)) & 0xf;
         if (sc >= nch)
            continue;   /* leaves the zero fill, and fill_w below */

         const unsigned char *p = (const unsigned char *)src + sc * cb;
         uint32_t bytes_read = 0;
         /* Byte at a time: Vulkan aligns an attribute only to its component
          * size, so a 16 bit component can sit on an odd address. */
         CP_VF_UNROLL
         for (uint32_t b = 0; b < 4; b++) {
            if (b >= cb)
               continue;
            bytes_read |= (uint32_t)p[b] << (b * 8);
         }

         d32[c] = cp_vf_expand(bytes_read, cb, conv);
      }
#ifdef CP_VF_FIXED_SLOTS
   } else {
      /*
       * One shape for every 32-bit-per-component case, because the
       * destination is registers here and not memory: a slot written once as
       * bytes and once as words is an alloca SROA refuses to split, and
       * refusing it puts the whole gather back in local memory. The bytes
       * that land are the same bytes -- a COPY32 format's size is a whole
       * number of words by construction, four bytes per component, so the
       * word loop covers exactly what the byte loop covered.
       */
      const unsigned char *bytes = (const unsigned char *)src;
      uint32_t *d32 = (uint32_t *)dst;
      bool aligned = ((uintptr_t)src & 3) == 0;
      CP_VF_UNROLL
      for (uint32_t w = 0; w < 4; w++) {
         if (w >= size / 4)
            continue;
         d32[w] = aligned
            ? ((const uint32_t *)src)[w]
            : ((uint32_t)bytes[w * 4 + 0] |
               ((uint32_t)bytes[w * 4 + 1] << 8) |
               ((uint32_t)bytes[w * 4 + 2] << 16) |
               ((uint32_t)bytes[w * 4 + 3] << 24));
      }
   }
#else
   } else if (size == 16 && ((uintptr_t)src & 15) == 0) {
      *(float4 *)dst = *(const float4 *)src;
   } else if (((uintptr_t)src & 3) == 0 && (size & 3) == 0) {
      for (uint32_t w = 0; w < size / 4; w++)
         ((uint32_t *)dst)[w] = ((const uint32_t *)src)[w];
   } else {
      for (uint32_t b = 0; b < size; b++)
         dst[b] = src[b];
   }
#endif

   /* A format with fewer than four components reads as (0, 0, 0, 1). The
    * slot was zero-filled, so only the one has to be written — and it
    * matters: a shader taking a vec3 position as vec4 otherwise gets w = 0
    * and loses the translation column of its matrix. */
   uint32_t fill_w = ap->elem_fill_w[e];
   if (fill_w)
      ((uint32_t *)dst)[3] = fill_w;
}

#endif /* CP_VF_LANE_H */
