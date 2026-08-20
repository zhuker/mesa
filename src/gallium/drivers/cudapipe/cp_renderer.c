/*
 * The renderer, compiled into both front ends.
 *
 * This file exists so that code the draw pipeline needs can be shared by the
 * Gallium-hosted driver and the native Vulkan one without either including
 * the other's headers. It starts with bring-up; the rest of the pipeline
 * follows it out of cp_context.c as each piece stops depending on Gallium.
 */

#include "cp_renderer.h"

#include <string.h>

/*
 * Bring a renderer up on a device.
 *
 * Everything here is CUDA and the driver's own bookkeeping -- a stream, the
 * flush generation ring, the GPU-resident state block, the device arena and
 * its host staging, and the rasterizer's queues. None of it needs a
 * pipe_screen or a pipe_context, which is the point: a Vulkan front end calls
 * this with a cp_device and gets a renderer it can draw with.
 */
bool
cp_context_init(struct cp_context *cp, struct cp_device *dev)
{
   cp->screen = dev;

   /* Allocate persistent GPU state (managed) and device-only arena */
   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* Every frame-path launch, memset and copy goes here; see cp_context.h for
    * why it is a default-flagged stream rather than a non-blocking one. If it
    * cannot be created the field stays zero, which is the legacy NULL stream
    * and exactly the behaviour this replaces. */
   if (cuStreamCreate(&cp->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
      cp->stream = NULL;

   /* The flush's generation ring; see cp_flush. A failed event creation
    * falls back to the draining flush by leaving flush_retire[0] null. */
   cp->flush_gens = cp_debug->flush_drain ? 1 : CP_FLUSH_GENS;
   if (cp->flush_gens > 1) {
      for (unsigned i = 0; i < cp->flush_gens; i++) {
         if (cuEventCreate(&cp->flush_retire[i],
                           CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
            for (unsigned j = 0; j < i; j++) {
               cuEventDestroy(cp->flush_retire[j]);
               cp->flush_retire[j] = NULL;
            }
            cp->flush_retire[0] = NULL;
            break;
         }
      }
   }
   CUdeviceptr state_dev;
   if (cuMemAllocManaged(&state_dev, sizeof(struct cp_gpu_state),
                         CU_MEM_ATTACH_GLOBAL) == CUDA_SUCCESS) {
      cp->gpu_state = (struct cp_gpu_state *)(uintptr_t)state_dev;
      memset(cp->gpu_state, 0, sizeof(struct cp_gpu_state));
   }

   /* 256MB arena — device-only, never touched by CPU */
   cuMemAlloc(&cp->arena_base, 256 * 1024 * 1024);
   cp->arena_size = 256 * 1024 * 1024;
   cp->arena_offset = 0;

   /* Staging for it. A frame of the heaviest sample in the sweep uploads
    * under a megabyte, and running out only costs a synchronisation. */
   if (cuMemAllocHost(&cp->upload_host, 8 * 1024 * 1024) == CUDA_SUCCESS)
      cp->upload_size = 8 * 1024 * 1024;

   /* Adaptive rasterizer queues — allocated once, reused across draws.
    *
    * The two queue counters share one allocation and sit adjacent, so the
    * pass loop zeroes both with a single cuMemsetD32 of two words rather than
    * one call each. They are zeroed once per rasterizer pass and a blended
    * draw runs hundreds of passes, so this is two host calls per pass rather
    * than two bytes of memory. Only the base is freed. */
   cuMemAlloc(&cp->rast_nontrivial,
              (size_t)CP_MAX_NONTRIVIAL * sizeof(uint32_t));
   cuMemAlloc(&cp->rast_counts, 256);
   cp->rast_nontrivial_count = cp->rast_counts;
   cp->rast_huge_count = cp->rast_counts + sizeof(uint32_t);
   cuMemAlloc(&cp->rast_huge_tiles,
              (size_t)CP_MAX_HUGE_TILES * sizeof(struct cp_tile_pair));

   /* What cp_draw_execute builds its queue struct from; a pass-episode
    * segment append swaps in its stream's own set and restores this one. */
   cp->cur_qset.nontrivial = cp->rast_nontrivial;
   cp->cur_qset.huge_tiles = cp->rast_huge_tiles;
   cp->cur_qset.counts = cp->rast_counts;

   return true;
}
