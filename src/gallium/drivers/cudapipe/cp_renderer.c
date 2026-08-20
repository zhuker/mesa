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
#include <stdlib.h>

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

/*
 * Hand out a slice of the draw's scratch arena.
 *
 * Returns managed memory, so the pointer is valid on both host and device. The
 * arena is only resized between draws, so a request that doesn't fit is served
 * by a one-off allocation and the arena grows to cover it next time rather
 * than moving memory that this draw is already pointing at.
 */
void *
cp_scratch_alloc(struct cp_context *cp, size_t bytes)
{
   if (!bytes)
      return NULL;

   unsigned cur = cp->scratch.current;
   size_t offset = ALIGN_POT(cp->scratch.used, 256);
   size_t end = offset + bytes;

   if (end <= cp->scratch.size[cur]) {
      cp->scratch.used = end;
      cp->scratch.peak = MAX2(cp->scratch.peak, end);
      return (void *)(uintptr_t)(cp->scratch.base[cur] + offset);
   }

   cp->scratch.peak = MAX2(cp->scratch.peak, end);

   /* Arena full — grow it in place by allocating a new larger chunk.
    * This replaces the current arena (the old one stays alive until flush
    * since the GPU may still be reading it). */
   size_t want = MAX2(end, cp->scratch.size[cur] * 2);
   want = MAX2(want, 1 << 20); /* at least 1MB */

   /*
    * Refuse to grow past a size no legitimate draw needs. The arena is
    * managed memory, so it is backed by system RAM as much as by the GPU, and
    * an allocation loop that runs away does not fail — it takes the machine
    * down with the OOM killer. That is not hypothetical: a multi-pass draw
    * that allocated its shading buffers per pass instead of reusing them
    * asked for tens of gigabytes and did exactly that. Failing here turns the
    * same mistake into a black frame and a message.
    */
   if (end > CP_SCRATCH_MAX_BYTES) {
      fprintf(stderr, "cudapipe: scratch arena wants %zu bytes, over the %zu "
              "cap — refusing. A stage is almost certainly allocating per "
              "pass instead of reusing.\n",
              end, (size_t)CP_SCRATCH_MAX_BYTES);
      return NULL;
   }
   want = MIN2(want, (size_t)CP_SCRATCH_MAX_BYTES);
   CUdeviceptr new_base;
   CUresult err = cuMemAllocManaged(&new_base, want, CU_MEM_ATTACH_GLOBAL);
   if (err != CUDA_SUCCESS) {
      /* Callers treat NULL as "skip this stage", which renders nothing and
       * looks like a shader bug, so say what actually happened. */
      fprintf(stderr, "cudapipe: scratch arena grow to %zu bytes failed (%d)\n",
              want, err);
      return NULL;
   }

   /* Stash the old arena pointer for freeing at flush */
   if (cp->scratch.base[cur] &&
       cp->scratch.num_overflow < ARRAY_SIZE(cp->scratch.overflow))
      cp->scratch.overflow[cp->scratch.num_overflow++] = cp->scratch.base[cur];

   cp->scratch.base[cur] = new_base;
   cp->scratch.size[cur] = want;
   cp->scratch.used = end;
   return (void *)(uintptr_t)(new_base + offset);
}

/*
 * The same, out of memory the host cannot reach. See cp_context.h for why the
 * distinction is worth having; the growth rules are the arena's above.
 *
 * Returns a device address rather than a pointer, so that a caller who
 * dereferences it does not compile.
 */
CUdeviceptr
cp_scratch_alloc_device(struct cp_context *cp, size_t bytes)
{
   if (!bytes)
      return 0;

   size_t offset = ALIGN_POT(cp->dscratch.used, 256);
   size_t end = offset + bytes;

   if (end <= cp->dscratch.size) {
      cp->dscratch.used = end;
      cp->dscratch.peak = MAX2(cp->dscratch.peak, end);
      return cp->dscratch.base + offset;
   }

   cp->dscratch.peak = MAX2(cp->dscratch.peak, end);

   size_t want = MAX2(end, cp->dscratch.size * 2);
   want = MAX2(want, (size_t)1 << 20);
   if (end > CP_SCRATCH_MAX_BYTES) {
      fprintf(stderr, "cudapipe: device scratch wants %zu bytes, over the %zu "
              "cap — refusing.\n", end, (size_t)CP_SCRATCH_MAX_BYTES);
      return 0;
   }
   want = MIN2(want, (size_t)CP_SCRATCH_MAX_BYTES);

   CUdeviceptr new_base;
   CUresult err = cuMemAlloc(&new_base, want);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: device scratch grow to %zu bytes failed (%d)\n",
              want, err);
      return 0;
   }

   if (cp->dscratch.base &&
       cp->dscratch.num_overflow < ARRAY_SIZE(cp->dscratch.overflow))
      cp->dscratch.overflow[cp->dscratch.num_overflow++] = cp->dscratch.base;

   cp->dscratch.base = new_base;
   cp->dscratch.size = want;
   cp->dscratch.used = end;
   return new_base + offset;
}

/*
 * Hand the device a block of per-draw constants.
 *
 * Every kernel here takes its parameters through a block in memory — the
 * argument pointer array the generated shaders read, the strides, the counts.
 * Those were being written by the host straight into the managed scratch
 * arena, which is the worst place for them: the host's write pulls the page
 * over to the host, and then the first warp of the shader that reads it stalls
 * while the page comes back. That stall is charged to the kernel, so it reads
 * as a slow vertex shader rather than as what it is. A frame of multithreading
 * makes roughly 1,800 such round trips.
 *
 * Device-only memory cannot fault, so the block goes in the arena that has
 * been sitting unused since the context was created and reaches it by DMA.
 * The copy is stream ordered, so it lands after the previous draw's kernels
 * have finished reading whatever occupied that space, and both offsets reset
 * at flush, which is already a synchronisation point.
 *
 * cp_upload_begin() reserves room and hands back both ends of it — the device
 * address the block will land at, and the staging bytes to write it into — so
 * that a block can refer to itself. The vertex shader's argument array holds a
 * pointer to the per-draw uniform table sitting behind it in the same block,
 * and that pointer is a device address, which is only knowable once the
 * destination has been chosen. Writing the block and uploading it afterwards
 * cannot express that.
 */
CUdeviceptr
cp_upload_begin(struct cp_context *cp, size_t size, void **host_out)
{
   if (!cp->arena_base || !cp->upload_host || !size)
      return 0;

   /* 256 bytes keeps every block on its own cache line and matches the
    * alignment the constant-buffer path already promises. */
   size_t dev_off = ALIGN_POT(cp->arena_offset, 256);
   size_t host_off = ALIGN_POT(cp->upload_offset, 256);

   /* The ring is partitioned into generations; this epoch owns one slice of
    * it and the flush rewinds to the next. flush_gens is 1 when the flush
    * still drains, which makes the slice the whole ring, as before. */
   size_t dev_slice = cp->arena_size / cp->flush_gens;
   size_t host_slice = cp->upload_size / cp->flush_gens;
   unsigned gen = cp->scratch.current;

   if (dev_off + size > (gen + 1) * dev_slice ||
       host_off + size > (gen + 1) * host_slice) {
      /*
       * Out of room before a flush came round. Rewinding would let this draw
       * overwrite staging a previous draw's copy has not read yet, so fall
       * back to a full drain, after which the whole slice is reusable.
       */
      cuCtxSynchronize();
      cp->arena_offset = gen * dev_slice;
      cp->upload_offset = gen * host_slice;
      dev_off = cp->arena_offset;
      host_off = cp->upload_offset;
      if (size > dev_slice || size > host_slice)
         return 0;
   }

   cp->arena_offset = dev_off + size;
   cp->upload_offset = host_off + size;
   *host_out = (char *)cp->upload_host + host_off;
   return cp->arena_base + dev_off;
}

/* Send a block reserved above, once the caller has finished writing it. */
void
cp_upload_end(struct cp_context *cp, CUdeviceptr dst, const void *host,
              size_t size)
{
   cuMemcpyHtoDAsync(dst, host, size, cp->stream);
}

CUdeviceptr
cp_upload(struct cp_context *cp, const void *data, size_t size)
{
   void *host;
   CUdeviceptr dst = cp_upload_begin(cp, size, &host);
   if (!dst)
      return 0;
   memcpy(host, data, size);
   cp_upload_end(cp, dst, host, size);
   return dst;
}

void
cp_scratch_begin(struct cp_context *cp)
{
   /*
    * Reclaim when the arena has expanded a few times, or when it has simply
    * handed out too much. The bump pointer is not reset between draws, so that
    * one draw's kernels can still be reading their buffers while the host sets
    * up the next — but that means a frame's draws accumulate, and counting
    * expansions alone does not notice: once the arena is large enough that
    * nothing has to grow, the counter stops moving and the pointer climbs
    * forever. A frame of bloom reached eleven gigabytes that way. Reclaiming
    * costs a sync, so the limit is high enough that ordinary draws still
    * pipeline.
    */
   if (cp->scratch.num_overflow >= 5 ||
       cp->scratch.used > CP_SCRATCH_RECLAIM_BYTES ||
       cp->dscratch.num_overflow >= 5 ||
       cp->dscratch.used > CP_SCRATCH_RECLAIM_BYTES) {
      cuCtxSynchronize();
      cp_scratch_reset(cp);
   }
}

/* Reset scratch after all GPU work is done. Frees overflow arenas (old
 * arenas that were replaced during growth) and resets the bump pointer.
 * The current arena is kept at its grown size. */
void
cp_scratch_reset(struct cp_context *cp)
{
   for (unsigned i = 0; i < cp->scratch.num_overflow; i++)
      cuMemFree(cp->scratch.overflow[i]);
   cp->scratch.num_overflow = 0;
   cp->scratch.used = 0;

   for (unsigned i = 0; i < cp->dscratch.num_overflow; i++)
      cuMemFree(cp->dscratch.overflow[i]);
   cp->dscratch.num_overflow = 0;
   cp->dscratch.used = 0;

   /* Callers of this have already waited for the whole device, so the
    * staging the uploads were copied out of is free to be written over
    * again — rewound to the current generation's slice, since the epoch
    * arithmetic in cp_upload_begin keeps running either way. */
   cp->arena_offset = (size_t)cp->scratch.current *
                      (cp->arena_size / cp->flush_gens);
   cp->upload_offset = (size_t)cp->scratch.current *
                       (cp->upload_size / cp->flush_gens);
}

void
cp_scratch_destroy(struct cp_context *cp)
{
   cuCtxSynchronize();
   for (unsigned i = 0; i < cp->scratch.num_overflow; i++)
      cuMemFree(cp->scratch.overflow[i]);
   for (unsigned i = 0; i < CP_FLUSH_GENS; i++) {
      if (cp->scratch.base[i])
         cuMemFree(cp->scratch.base[i]);
   }
   memset(&cp->scratch, 0, sizeof(cp->scratch));

   for (unsigned i = 0; i < cp->dscratch.num_overflow; i++)
      cuMemFree(cp->dscratch.overflow[i]);
   if (cp->dscratch.base)
      cuMemFree(cp->dscratch.base);
   memset(&cp->dscratch, 0, sizeof(cp->dscratch));
}

/* Per-stage timing for a draw, printed under CUDAPIPE_DEBUG_TIME. See
 * cp_stage_end() below for why it is measured with events and not a clock. */
bool
cp_timing_enabled(void)
{
   return cp_debug->debug_time;
}


/* Record that `stage` has just finished. The first mark of a draw carries no
 * stage and only starts the clock. */
void
cp_stage_end(struct cp_context *cp, int stage)
{
   if (!cp_timing_enabled())
      return;

   struct cp_stage_timer *t = &cp->timer;
   if (t->num == t->cap) {
      unsigned cap = t->cap ? t->cap * 2 : 64;
      CUevent *ev = realloc(t->events, cap * sizeof(*ev));
      int *st = realloc(t->stages, cap * sizeof(*st));
      if (!ev || !st) {
         free(ev ? ev : t->events);
         free(st ? st : t->stages);
         t->events = NULL; t->stages = NULL; t->cap = t->num = 0;
         return;
      }
      t->events = ev;
      t->stages = st;
      /* Events are created once and re-recorded, since creating one costs
       * more than recording it and a deep draw records thousands. */
      for (unsigned i = t->cap; i < cap; i++)
         if (cuEventCreate(&t->events[i], CU_EVENT_DEFAULT) != CUDA_SUCCESS)
            return;
      t->cap = cap;
   }

   t->stages[t->num] = stage;
   cuEventRecord(t->events[t->num], cp->stream);
   t->num++;
}
