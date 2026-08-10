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
      /* TRIANGLE_LIST with index buffer: direct indexing */
      if (args.index_buffer && args.index_size > 0) {
         const char *ib = (const char *)(uintptr_t)args.index_buffer;
         if (args.index_size == 2)
            vertex_id = (uint32_t)((const unsigned short *)ib)[v] + args.first_vertex;
         else
            vertex_id = ((const uint32_t *)ib)[v] + args.first_vertex;
      } else {
         vertex_id = v + args.first_vertex;
      }
      instance_id = 0;
   }

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
      char *dst = out + e * 16;

      /* Unrolled copy for common sizes */
      if (size >= 16) {
         *(float4 *)dst = *(const float4 *)src;
      } else if (size == 12) {
         *(float *)dst = *(const float *)src;
         *(float *)(dst + 4) = *(const float *)(src + 4);
         *(float *)(dst + 8) = *(const float *)(src + 8);
      } else if (size == 8) {
         *(float2 *)dst = *(const float2 *)src;
      } else if (size == 4) {
         *(float *)dst = *(const float *)src;
      } else {
         for (uint32_t b = 0; b < size && b < 16; b++)
            dst[b] = src[b];
      }
   }
}
