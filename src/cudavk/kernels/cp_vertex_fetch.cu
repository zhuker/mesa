/*
 * GPU vertex fetch kernel — replaces the CPU-side attribute gathering loop.
 *
 * One thread per assembled vertex. Each thread reads its vertex index from the
 * index buffer (or computes it for non-indexed draws), then gathers all
 * attributes from the bound vertex buffers into the packed VS input layout.
 */
#include "cp_rast_types.h"

#include "cp_vf_lane.h"

/*
 * One thread per assembled vertex. The gather itself lives in cp_vf_lane.h so
 * that the generated vertex shader can inline the same code; this kernel is
 * the standalone form, and it is what runs for any shader the fused build is
 * not admitted for.
 */
extern "C" __global__ void
cp_vertex_fetch(struct cp_vertex_fetch_args args)
{
   uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;

   /*
    * The counters the clipper and the rasterizer are about to fill. One
    * thread writes them; the kernel boundary is what publishes them, exactly
    * as it published the host's clears. Done before the bounds check below so
    * that the seeding does not depend on this thread having a vertex.
    */
   if (v == 0) {
      if (args.seed_counts) {
         uint32_t *counts = (uint32_t *)(uintptr_t)args.seed_counts;
         counts[0] = 0;
         counts[1] = 0;
         counts[2] = 0;
      }
      if (args.seed_clip_count)
         *(uint32_t *)(uintptr_t)args.seed_clip_count = args.clip_seed;
   }

   unsigned char *slots = (unsigned char *)(uintptr_t)args.output +
                          (uint64_t)v * args.vs_in_stride;
   uint32_t vertex_id, instance_id, row;
   if (!cp_vf_lane_ids(&args, v, &vertex_id, &instance_id, &row))
      return;

   /* Every bound element, in a runtime loop: this kernel does not know which
    * of them the shader reads, and gathering all of them is what it has always
    * done. The fused execution knows, and emits one call per live slot with
    * the index a constant, which is where that becomes worth exploiting. */
   for (uint32_t e = 0; e < args.num_elements &&
                        e < CP_MAX_VERTEX_ELEMENTS_VF; e++)
      cp_vf_lane_element(&args, e, vertex_id, instance_id, row,
                         slots + e * 16);

   /* The vertex shader picks its own draw's uniform bindings out of this.
    * Only the derived path resolves a row; the refs path never coexists with
    * a batch, which is why this is not written there. */
   if (!args.vertex_ids && args.out_batch_rows)
      ((uint32_t *)(uintptr_t)args.out_batch_rows)[v] = row;

   /* Hand the ids to the vertex shader, which reads them per thread. Doing it
    * here rather than on the host is the point: the arrays are megabytes for
    * an instanced draw and every byte of them is derivable from v. */
   if (args.out_vertex_ids)
      ((uint32_t *)(uintptr_t)args.out_vertex_ids)[v] = vertex_id;
   if (args.out_instance_ids)
      ((uint32_t *)(uintptr_t)args.out_instance_ids)[v] = instance_id;
}
