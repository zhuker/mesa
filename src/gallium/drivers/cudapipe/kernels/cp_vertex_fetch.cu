/*
 * GPU vertex fetch kernel — replaces the CPU-side attribute gathering loop.
 *
 * One thread per assembled vertex. Each thread reads its vertex index from the
 * index buffer (or computes it for non-indexed draws), then gathers all
 * attributes from the bound vertex buffers into the packed VS input layout.
 */
#include "cp_rast_types.h"

extern "C" __global__ void
cp_vertex_fetch(struct cp_vertex_fetch_args args)
{
   uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
   if (v >= args.num_verts)
      return;

   /* Determine the actual vertex index for this assembled vertex */
   uint32_t vertex_id;
   uint32_t instance_id;

   if (args.vertex_ids) {
      /* Pre-expanded topology (strip/fan/points): read from the refs array */
      vertex_id = ((const uint32_t *)(uintptr_t)args.vertex_ids)[v];
      instance_id = ((const uint32_t *)(uintptr_t)args.instance_ids)[v];
   } else {
      /*
       * TRIANGLE_LIST: the ids follow from the thread index alone. Every
       * instance replays the same index range, so the vertex within the
       * instance is v modulo the instance's vertex count and the instance is
       * the quotient. verts_per_instance is num_verts for a single-instance
       * draw, which leaves this the plain v it was before.
       */
      uint32_t local = v;
      uint32_t first_vertex = args.first_vertex;
      const char *ib = (const char *)(uintptr_t)args.index_buffer;

      /*
       * A batch concatenates the draws it merged, and they need not share an
       * index range: this thread's draw is the last one that starts at or
       * before it. Binary search, over a table bounded by the batch cap and so
       * a few hundred bytes at most. Nothing here runs for an unbatched draw —
       * num_draw_slices is zero and every value above stands.
       */
      uint32_t row = 0;
      if (args.num_draw_slices) {
         const struct cp_draw_slice *sl =
            (const struct cp_draw_slice *)(uintptr_t)args.draw_slices;
         uint32_t lo = 0, hi = args.num_draw_slices - 1;
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
      }

      /* The vertex shader picks its own draw's uniform bindings out of this. */
      if (args.out_batch_rows)
         ((uint32_t *)(uintptr_t)args.out_batch_rows)[v] = row;

      instance_id = 0;
      if (args.verts_per_instance) {
         instance_id = local / args.verts_per_instance;
         local = local % args.verts_per_instance;
      }

      if (args.index_buffer && args.index_size > 0) {
         if (args.index_size == 2)
            vertex_id = (uint32_t)((const unsigned short *)ib)[local] + first_vertex;
         else
            vertex_id = ((const uint32_t *)ib)[local] + first_vertex;
      } else {
         vertex_id = local + first_vertex;
      }
   }

   /* Hand the ids to the vertex shader, which reads them per thread. Doing it
    * here rather than on the host is the point of the block above: the arrays
    * are megabytes for an instanced draw and every byte of them is derivable
    * from v. */
   if (args.out_vertex_ids)
      ((uint32_t *)(uintptr_t)args.out_vertex_ids)[v] = vertex_id;
   if (args.out_instance_ids)
      ((uint32_t *)(uintptr_t)args.out_instance_ids)[v] = instance_id;

   /* Gather all attributes for this vertex into the packed output */
   char *out = (char *)(uintptr_t)args.output + (uint64_t)v * args.vs_in_stride;

   for (uint32_t e = 0; e < args.num_elements; e++) {
      uint64_t vb_base = args.vb_bases[args.elem_vb_idx[e]];
      if (!vb_base)
         continue;

      uint32_t index = args.elem_instance_divisor[e]
         ? args.start_instance + instance_id / args.elem_instance_divisor[e]
         : vertex_id;

      const char *src = (const char *)(uintptr_t)vb_base +
                        (uint64_t)index * args.elem_src_stride[e] +
                        args.elem_src_offset[e];

      /* Copy attribute bytes into the 16-byte slot. The output was zero-filled
       * by the caller (scratch arena is zeroed on grow). */
      uint32_t size = args.elem_attr_size[e];
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
      if (size == 16 && ((uintptr_t)src & 15) == 0) {
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
      uint32_t fill_w = args.elem_fill_w[e];
      if (fill_w)
         ((uint32_t *)dst)[3] = fill_w;
   }
}
