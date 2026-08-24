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
 * Gather one assembled vertex. Returns 0 when this lane has no vertex, which
 * is the seam a bounds-checked or robust-access fetch would use; today the
 * only decline is the count check.
 *
 * `slots` is `num_elements * 16` bytes owned by the caller: the packed input
 * buffer for the classic kernel, a function-entry alloca for the fused
 * shader. The ids and the batch row come back by pointer; publishing them to
 * the per-vertex arrays is the caller's business, because the fused shader
 * has no reason to write them to memory at all.
 */
CP_VF_INLINE int
cp_vf_lane(const struct cp_vertex_fetch_args *ap, uint32_t v,
           unsigned char *slots, uint32_t *out_vertex_id,
           uint32_t *out_instance_id, uint32_t *out_row)
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

   /* Gather all attributes for this vertex into the packed slots. */
   char *out = (char *)slots;

   for (uint32_t e = 0; e < ap->num_elements; e++) {
      /* A batch carries one row of per-element base addresses per merged
       * draw, so draws bound to different vertex buffers merge; without one
       * the launch-wide bases stand, which is the unbatched path unchanged. */
      uint64_t vb_base = ap->elem_bases
         ? ((const uint64_t *)(uintptr_t)ap->elem_bases)
              [row * CP_VB_TABLE_STRIDE + e]
         : ap->vb_bases[ap->elem_vb_idx[e]];
      if (!vb_base)
         continue;

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
      char *dst = out + e * 16;

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

         for (uint32_t c = 0; c < nch && c < 4; c++) {
            uint32_t sc = (swz >> (c * 4)) & 0xf;
            if (sc >= nch)
               continue;   /* leaves the zero fill, and fill_w below */

            const unsigned char *p = (const unsigned char *)src + sc * cb;
            uint32_t bytes_read = 0;
            /* Byte at a time: Vulkan aligns an attribute only to its component
             * size, so a 16 bit component can sit on an odd address. */
            for (uint32_t b = 0; b < cb; b++)
               bytes_read |= (uint32_t)p[b] << (b * 8);

            d32[c] = cp_vf_expand(bytes_read, cb, conv);
         }
      } else if (size == 16 && ((uintptr_t)src & 15) == 0) {
         *(float4 *)dst = *(const float4 *)src;
      } else if (((uintptr_t)src & 3) == 0 && (size & 3) == 0) {
         for (uint32_t w = 0; w < size / 4; w++)
            ((uint32_t *)dst)[w] = ((const uint32_t *)src)[w];
      } else {
         for (uint32_t b = 0; b < size; b++)
            dst[b] = src[b];
      }

      /* A format with fewer than four components reads as (0, 0, 0, 1). The
       * slot was zero-filled, so only the one has to be written — and it
       * matters: a shader taking a vec3 position as vec4 otherwise gets w = 0
       * and loses the translation column of its matrix. */
      uint32_t fill_w = ap->elem_fill_w[e];
      if (fill_w)
         ((uint32_t *)dst)[3] = fill_w;
   }

   return 1;
}

#endif /* CP_VF_LANE_H */
