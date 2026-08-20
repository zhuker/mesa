/*
 * The renderer, compiled into both front ends.
 *
 * This file exists so that code the draw pipeline needs can be shared by the
 * Gallium-hosted driver and the native Vulkan one without either including
 * the other's headers. It starts with bring-up; the rest of the pipeline
 * follows it out of cp_context.c as each piece stops depending on Gallium.
 */

#include "cp_renderer.h"
#include "nir_to_ptx/cp_nir_to_llvm.h"

#include <inttypes.h>

#include "util/u_memory.h"
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

/*
 * ---------------------------------------------------------------------------
 * TEMPORARY INSTRUMENTATION: fragment census (CUDAPIPE_FRAG_CENSUS=1)
 * ---------------------------------------------------------------------------
 *
 * Costing out an A-buffer needs the size of the fragment population a blended
 * draw would produce in one rasterization pass. emit_fragment() is the single
 * site where coverage becomes a fragment, so a per-pixel atomic counter there
 * counts exactly that. This dumps the histogram after the *first* peel pass of
 * a peeled draw — the first pass rasterizes the whole draw, so one pass is the
 * whole population — and then clears the pointers so the remaining passes
 * count nothing. Delete all of this; it is not a feature.
 */
bool
cp_census_enabled(void)
{
   return cp_debug->frag_census;
}

static int
cp_census_cmp_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
   return x < y ? -1 : x > y ? 1 : 0;
}

void
cp_census_dump(const char *what, unsigned draw_seq, unsigned peel_seq,
               unsigned num_triangles, unsigned num_samples,
               const uint32_t *counts, unsigned w, unsigned h)
{
   size_t pixels = (size_t)w * h;
   uint64_t total = 0, capped = 0;
   size_t covered = 0, over_cap = 0;
   uint32_t max = 0;

   for (size_t i = 0; i < pixels; i++) {
      total += counts[i];
      capped += MIN2(counts[i], (uint32_t)CP_BLEND_LAYERS);
      if (counts[i] > (uint32_t)CP_BLEND_LAYERS)
         over_cap++;
      if (counts[i]) {
         covered++;
         if (counts[i] > max)
            max = counts[i];
      }
   }

   uint32_t med = 0, p90 = 0, p99 = 0;
   if (covered) {
      uint32_t *nz = malloc(covered * sizeof(uint32_t));
      if (nz) {
         size_t n = 0;
         for (size_t i = 0; i < pixels; i++)
            if (counts[i])
               nz[n++] = counts[i];
         qsort(nz, n, sizeof(uint32_t), cp_census_cmp_u32);
         med = nz[n / 2];
         p90 = nz[(size_t)(n * 90 / 100)];
         p99 = nz[(size_t)(n * 99 / 100)];
         free(nz);
      }
   }

   /* Per fragment: linked list is {prim id, next}, counted layout is a prim id
    * in a contiguous run. Both need a per-pixel word — a head pointer or a run
    * offset — so it is reported apart from the per-fragment part. */
   double mb = 1024.0 * 1024.0;
   fprintf(stderr,
           "census[%s] draw=%u peeled=%u tris=%u samples=%u\n"
           "  fragments=%" PRIu64 "  pixels_covered=%zu/%zu (%.2f%%)\n"
           "  per-covered-pixel: median=%u p90=%u p99=%u max=%u mean=%.2f\n"
           "  bytes: linked(8B/frag)=%.2f MB  counted(4B/frag)=%.2f MB"
           "  per-pixel word=%.2f MB\n",
           what, draw_seq, peel_seq, num_triangles, num_samples,
           total, covered, pixels, 100.0 * (double)covered / (double)pixels,
           med, p90, p99, max,
           covered ? (double)total / (double)covered : 0.0,
           (double)(total * 8) / mb, (double)(total * 4) / mb,
           (double)(pixels * 4) / mb);

   /* What peeling actually gets through: one fragment per pixel per pass, and
    * at most CP_BLEND_LAYERS passes, so anything deeper than that is dropped. */
   fprintf(stderr,
           "  peel reaches %" PRIu64 " of them (cap %d/pixel); %zu pixels are "
           "deeper than the cap and lose %" PRIu64 " fragments\n",
           capped, CP_BLEND_LAYERS, over_cap, total - capped);

   /* Coarse shape of the tail, so the distribution is not just five numbers. */
   static const uint32_t edges[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
   fprintf(stderr, "  histogram (fragments per covered pixel):");
   for (unsigned e = 0; e < ARRAY_SIZE(edges); e++) {
      uint32_t lo = edges[e];
      uint32_t hi = e + 1 < ARRAY_SIZE(edges) ? edges[e + 1] : 0xFFFFFFFFu;
      size_t n = 0;
      for (size_t i = 0; i < pixels; i++)
         if (counts[i] >= lo && counts[i] < hi)
            n++;
      if (n)
         fprintf(stderr, " [%u,%u)=%zu", lo, hi, n);
   }
   fprintf(stderr, "\n");
}

/*
 * ---------------------------------------------------------------------------
 * TEMPORARY INSTRUMENTATION: A-buffer build and check (CUDAPIPE_ABUFFER=1)
 * ---------------------------------------------------------------------------
 *
 * A blended draw is rendered by peeling: the whole draw is re-rasterized and
 * re-shaded once per layer, up to CP_BLEND_LAYERS times. What would replace
 * that is one rasterization that records every covered fragment, shaded and
 * composited once. This builds that record — and nothing more. The draw is
 * still rendered by the peel loop underneath; the list is built beside it and
 * checked against what the loop composites.
 *
 * Four steps, each timed separately with CUDA events:
 *
 *   1. count   a rasterization pass that only increments a per-pixel counter,
 *              for the fragments that pass the depth test — which is the set
 *              peeling composites, since emit_fragment tests depth before it
 *              touches the visibility buffer
 *   2. scan    exclusive prefix sum of those counts into per-pixel offsets
 *   3. fill    a second rasterization pass writing each fragment's primitive
 *              id at offset[pixel] + atomicAdd(&cursor[pixel], 1)
 *   4. sort    each pixel's run ascending by primitive index
 *
 * The claim being checked is that after the sort, pixel p's run is exactly the
 * sequence the peel loop composites at p, in order — peeling selects, on pass
 * k, the k-th smallest primitive index still covering p. So the loop is made
 * to log what it selected, and the two are compared element by element.
 *
 * Restrictions, all of them bail-outs rather than approximations: single
 * sample only, the peel path only, and depth writes off — a draw that writes
 * depth changes the depth-test outcome between peel passes, so a population
 * counted before the loop would not be the one the loop sees.
 */

/* 128 MB of primitive ids. The measured population is 2.7M fragments, so this
 * is about twelve times what particlesystem needs; a draw that wants more is
 * refused rather than allocated for. */
#define CP_ABUF_MAX_FRAGS (32u * 1024u * 1024u)
#define CP_ABUF_MIN_FRAGS (5u * 1024u * 1024u)

/* Both paths' shaded colours, one float4 and one write count per slot each.
 * The measured population needs about 135 MB of this; a draw wanting more is
 * shaded and timed without being checked, rather than allocated for. */
#define CP_ABUF_MAX_COLOR_BYTES (512.0 * 1024.0 * 1024.0)


/* Pixels whose whole run is compared against the peel loop, deepest first. */

/*
 * How the fragment array is sized, and when it is resized.
 *
 * The array used to be sized once from the first draw ever seen, at 25% over
 * its count, and never grown — which over 600 frames of particlesystem left
 * the population at 87.1% of capacity. That is a scene whose animation happens
 * not to grow much; one that did would quietly stop being eligible and cost
 * seven times as much, because an overrun falls back to the peel loop.
 *
 * So: allocate at twice the first count, and grow when a *later* count comes
 * within CP_ABUF_GROW_AT of capacity. The growth is deliberately not a
 * doubling of the array — it is twice the population that triggered it — so
 * that a scene which grew once does not keep paying for the growth rate it had
 * at the time.
 *
 * **The growth happens between draws, never inside the peel loop or the merge,
 * and at most once per draw.** `CUDAPIPE_HANDOFF.md` records this driver
 * invoking the OOM killer with an allocator that grew inside a loop. It is
 * additionally bounded three ways: by CP_ABUF_MAX_FRAGS, by
 * CP_ABUF_MAX_GROWTHS over the process, and by the fact that every growth is
 * announced on stderr. A refused growth is not an error — the draw falls back
 * to the peel loop, which is correct and slow.
 */
#define CP_ABUF_HEADROOM     2u      /* times the count that sized it */
#define CP_ABUF_SLACK        65536u  /* plus this, so a tiny first draw is not tiny */


struct cp_abuf cp_abuf = { .enabled = -1 };

/*
 * Record an A-buffer stage boundary, but only when someone is going to read
 * it. cuEventElapsedTime is called only under CUDAPIPE_ABUFFER_TIMING, which
 * is off by default, and sixteen records per eligible draw is nothing beside
 * the kernels they bracket when the numbers are wanted.
 *
 * When they are not, it is 11.8M calls and 3% of the frame on a capture that
 * takes the A-buffer path 482 times per frame — which only became visible
 * once the A-buffer stopped disabling itself and started running.
 */
void
cp_abuf_mark(CUevent ev, CUstream stream)
{
   if (cp_abuf.timing)
      cuEventRecord(ev, stream);
}

bool
cp_abuf_enabled(void)
{
   if (cp_abuf.enabled < 0) {
      /* On by default, off with CUDAPIPE_NO_ABUFFER=1 — the same shape as
       * CUDAPIPE_NO_BATCH and CUDAPIPE_NO_BINCACHE. CUDAPIPE_ABUFFER=1 still
       * means what it always did and is now a no-op, so a command line or a
       * script written against the opt-in version still does what it says. */
      cp_abuf.enabled = cp_debug->no_abuffer ? 0 : 1;

      /*
       * Compositing and verifying are exclusive, because the verification is
       * an element-wise comparison against what the peel loop produced and
       * compositing is exactly not running the peel loop. Asking for the
       * check is therefore a way of asking for the old behaviour, and
       * CUDAPIPE_ABUFFER_COMPOSITE=0 is the other.
       *
       * The default is to composite: the ~240 MB of comparison buffers and
       * the host-side walks over them are not something a frame being
       * measured should be carrying.
       */
      cp_abuf.verify = cp_debug->abuffer_verify;
      cp_abuf.composite = cp_debug->abuffer_composite;

      cp_abuf.verify_max = cp_debug->abuffer_verify_draws;
      /* The per-draw event breakdown costs a drain and a line of stderr per
       * draw, which nothing on the default path wants — it was on by default
       * while the path was opt-in and something being examined, and is off by
       * default now that it is how blended draws are rendered. */
      cp_abuf.timing = cp_debug->abuffer_timing;
      /* Likewise the running commentary on which draws are eligible: useful
       * when the question is why a draw peeled, noise on every other run. */
      cp_abuf.debug = cp_debug->abuffer_debug;
      cp_abuf.max_layers = cp_debug->abuffer_layers;
   }
   return cp_abuf.enabled == 1;
}

/*
 * Whether consecutive blended draws may share one A-buffer episode.
 *
 * The drain that decides whether the merged lists are complete is one
 * cuStreamSynchronize per eligible draw — 482 of them a frame on the capture,
 * and the largest single line in it. A batch pays one for all of them: the
 * capture's 727,372 drains become 316,413, and it replays in 185 s where it
 * took 333.
 *
 * On by default, off with CUDAPIPE_NO_ABUF_BATCH=1, which is the shape
 * CUDAPIPE_NO_BATCH and CUDAPIPE_NO_ABUFFER already have.
 */
bool
cp_abuf_batch_enabled(void)
{
   return !cp_debug->no_abuf_batch;
}

static void
cp_abuf_free(CUdeviceptr *p)
{
   if (*p)
      cuMemFree(*p);
   *p = 0;
}

static bool
cp_abuf_alloc(struct cp_abuf *ab, CUdeviceptr *p, size_t bytes,
              const char *what)
{
   CUresult e = cuMemAlloc(p, bytes);
   if (e != CUDA_SUCCESS) {
      fprintf(stderr, "abuffer: cuMemAlloc(%s, %zu) failed (%d); disabled\n",
              what, bytes, e);
      ab->disabled = true;
      return false;
   }
   return true;
}

/*
 * Allocate everything that is sized by the framebuffer. The fragment array is
 * not here: its size comes from the first frame's measured count, and nothing
 * about it depends on the framebuffer — a quad array bounded by the fragment
 * population is bounded by it at any resolution.
 *
 * **Grow-only, because the size alternates within a frame.** This used to
 * allocate once and disable itself the first time the framebuffer changed,
 * which cost nothing while the only sample taking the path rendered at one
 * size. A capture with a bloom pyramid changes size fifteen times a frame —
 * 1280x720 for the scene and 160x90 and smaller for the pyramid — so the
 * disable fired in the first second and every blended draw for the rest of the
 * run went back to the peel loop.
 *
 * Free-and-reallocate on every change is the other wrong answer: it would be
 * thirty cuMemFree/cuMemAlloc storms a frame, each of them a device-wide
 * synchronising operation. So the arrays are sized to the largest w*h and the
 * largest 2x2-block count seen so far and never shrink. Everything here is
 * indexed by pixel or by block and bounded by a count the caller passes in, so
 * an array sized for 921,600 pixels serves a 14,400-pixel pass exactly as well;
 * the kernels never read past the current n. What has to be recomputed on every
 * change is arithmetic: w, h, the three scan-level block counts on each of the
 * two ladders, and the block geometry.
 *
 * A size that does not fit the three-level scan is refused for that size alone
 * rather than disabling the path, since the next render pass is likely smaller.
 */
bool
cp_abuf_setup(struct cp_abuf *ab, unsigned w, unsigned h)
{
   if (ab->disabled)
      return false;
   if (ab->ready && ab->w == w && ab->h == h)
      return true;

   size_t n = (size_t)w * h;
   unsigned nb1 = (unsigned)((n + CP_ABUF_SCAN_BLOCK - 1) / CP_ABUF_SCAN_BLOCK);
   unsigned nb2 = (nb1 + CP_ABUF_SCAN_BLOCK - 1) / CP_ABUF_SCAN_BLOCK;
   unsigned nb3 = (nb2 + CP_ABUF_SCAN_BLOCK - 1) / CP_ABUF_SCAN_BLOCK;
   unsigned quad_width = (w + 1) / 2;
   unsigned nblocks = quad_width * ((h + 1) / 2);
   unsigned bnb1 = (nblocks + CP_ABUF_SCAN_BLOCK - 1) / CP_ABUF_SCAN_BLOCK;
   unsigned bnb2 = (bnb1 + CP_ABUF_SCAN_BLOCK - 1) / CP_ABUF_SCAN_BLOCK;
   unsigned bnb3 = (bnb2 + CP_ABUF_SCAN_BLOCK - 1) / CP_ABUF_SCAN_BLOCK;
   if (nb3 != 1 || bnb3 != 1) {
      static int said_levels = 0;
      if (!said_levels++)
         fprintf(stderr, "abuffer: %ux%u needs more than three scan levels; "
                 "draws at that size peel\n", w, h);
      return false;
   }

   /* The fixed-size counters and the events, once per process. */
   if (!ab->sum3) {
      /*
       * Every eligible draw drains the stream and then reads six words back,
       * and they used to be three allocations and therefore three separate
       * synchronous copies: 628 of them a frame, 2.8 ms, with the device idle
       * for 93% of it. One allocation in the order the readback wants makes it
       * one copy.
       *
       * The words are laid out sum3 | bsum3 | clist_count so that
       * CP_ABUF_COUNTERS covers all six; the existing derived pointers stay
       * offsets into it exactly as they were.
       */
      if (!cp_abuf_alloc(ab, &ab->counters,
                         4 * (CP_ABUF_COUNTERS + CP_PASS_MAX_SEGS + 1),
                         "sum3+bsum3+clist_count+seg counts+rec cursor") ||
          !cp_abuf_alloc(ab, &ab->list_count, 4, "list_count") ||
          !cp_abuf_alloc(ab, &ab->blk_list_count, 4, "block worklist count") ||
          !cp_abuf_alloc(ab, &ab->dbg, CP_ABUF_DBG_COUNTERS * 4,
                         "debug counters"))
         return false;

      ab->sum3 = ab->counters;              /* scan total, fill overflow, long runs */
      ab->bsum3 = ab->counters + 4 * 3;     /* quad total, quad overflow */
      ab->clist_count = ab->counters + 4 * 5;

      ab->overflow = ab->sum3 + 4;
      ab->long_runs = ab->sum3 + 8;
      ab->quad_overflow = ab->bsum3 + 4;
      ab->seg_counts = ab->counters + 4 * CP_ABUF_COUNTERS;
      ab->rec_cursor = ab->counters +
                       4 * (CP_ABUF_COUNTERS + CP_PASS_MAX_SEGS);
   }
   if (!ab->events_ready) {
      for (int i = 0; i < (int)ARRAY_SIZE(ab->ev); i++) {
         if (cuEventCreate(&ab->ev[i], CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
            fprintf(stderr, "abuffer: cuEventCreate failed; disabled\n");
            ab->disabled = true;
            return false;
         }
      }
      ab->events_ready = true;
   }

   /* The per-pixel ladder. */
   if (n > ab->cap_pixels) {
      size_t nb = n * sizeof(uint32_t);
      ab->ready = false;
      ab->cap_pixels = 0;
      cp_abuf_free(&ab->counts);
      cp_abuf_free(&ab->offsets);
      cp_abuf_free(&ab->cursor);
      cp_abuf_free(&ab->list);
      cp_abuf_free(&ab->clist);
      cp_abuf_free(&ab->sum1);
      cp_abuf_free(&ab->sum1x);
      cp_abuf_free(&ab->sum2);
      cp_abuf_free(&ab->sum2x);
      if (!cp_abuf_alloc(ab, &ab->counts, nb, "counts") ||
          !cp_abuf_alloc(ab, &ab->offsets, nb, "offsets") ||
          !cp_abuf_alloc(ab, &ab->cursor, nb, "cursor") ||
          !cp_abuf_alloc(ab, &ab->list, nb, "list") ||
          !cp_abuf_alloc(ab, &ab->sum1, nb1 * 4, "sum1") ||
          !cp_abuf_alloc(ab, &ab->sum1x, nb1 * 4, "sum1x") ||
          !cp_abuf_alloc(ab, &ab->sum2, nb2 * 4, "sum2") ||
          !cp_abuf_alloc(ab, &ab->sum2x, nb2 * 4, "sum2x"))
         return false;

      free(ab->h_counts);
      free(ab->h_offsets);
      free(ab->h_cursor);
      ab->h_counts = malloc(nb);
      ab->h_offsets = malloc(nb);
      ab->h_cursor = malloc(nb);
      if (!ab->h_counts || !ab->h_offsets || !ab->h_cursor) {
         fprintf(stderr, "abuffer: host mirrors failed; disabled\n");
         ab->disabled = true;
         return false;
      }
      ab->cap_pixels = n;
      if (ab->resizes++)
         fprintf(stderr, "abuffer: grew the per-pixel arrays to %ux%u\n", w, h);
   }

   /* The per-2x2-block ladder. 640x360 blocks for a 1280x720 framebuffer, so
    * the same three scan levels are ample. */
   if (nblocks > ab->cap_blocks) {
      size_t bb = (size_t)nblocks * sizeof(uint32_t);
      ab->ready = false;
      ab->cap_blocks = 0;
      cp_abuf_free(&ab->blk_counts);
      cp_abuf_free(&ab->blk_offsets);
      cp_abuf_free(&ab->blk_list);
      cp_abuf_free(&ab->bsum1);
      cp_abuf_free(&ab->bsum1x);
      cp_abuf_free(&ab->bsum2);
      cp_abuf_free(&ab->bsum2x);
      if (!cp_abuf_alloc(ab, &ab->blk_counts, bb, "block counts") ||
          !cp_abuf_alloc(ab, &ab->blk_offsets, bb, "block offsets") ||
          !cp_abuf_alloc(ab, &ab->blk_list, bb, "block worklist") ||
          !cp_abuf_alloc(ab, &ab->bsum1, bnb1 * 4, "block sum1") ||
          !cp_abuf_alloc(ab, &ab->bsum1x, bnb1 * 4, "block sum1x") ||
          !cp_abuf_alloc(ab, &ab->bsum2, bnb2 * 4, "block sum2") ||
          !cp_abuf_alloc(ab, &ab->bsum2x, bnb2 * 4, "block sum2x"))
         return false;
      free(ab->h_blk_counts);
      free(ab->h_blk_offsets);
      ab->h_blk_counts = malloc(bb);
      ab->h_blk_offsets = malloc(bb);
      if (!ab->h_blk_counts || !ab->h_blk_offsets) {
         fprintf(stderr, "abuffer: block mirrors failed; disabled\n");
         ab->disabled = true;
         return false;
      }
      ab->cap_blocks = nblocks;
   }

   /* The composite's worklist. Outside the growth block because the flag can be
    * turned on after the first setup — asking to verify without the kernels to
    * verify with falls back to compositing — and then this has to appear. */
   if (cp_abuf.composite) {
      if (!ab->clist &&
          !cp_abuf_alloc(ab, &ab->clist, ab->cap_pixels * sizeof(uint32_t),
                         "composite worklist"))
         return false;
      /* clist_count is carved out of ab->counters above, so that the
       * per-draw readback stays a single copy; nothing to allocate. */
   }

   if (cp_abuf.verify && n > ab->cap_log_pixels) {
      size_t logb = n * CP_ABUF_LOG_LAYERS * sizeof(uint32_t);
      size_t deepb = CP_ABUF_DEEP_PIXELS * CP_BLEND_LAYERS * sizeof(uint32_t);
      ab->cap_log_pixels = 0;
      cp_abuf_free(&ab->log);
      if (!cp_abuf_alloc(ab, &ab->log, logb, "peel log"))
         return false;
      if (!ab->deep_log &&
          (!cp_abuf_alloc(ab, &ab->deep_list,
                          CP_ABUF_DEEP_PIXELS * 4, "deep list") ||
           !cp_abuf_alloc(ab, &ab->deep_log, deepb, "deep peel log")))
         return false;
      free(ab->h_log);
      ab->h_log = malloc(logb);
      if (!ab->h_deep_log) {
         ab->h_deep_log = malloc(deepb);
         ab->h_deep_list = malloc(CP_ABUF_DEEP_PIXELS * 4);
      }
      if (!ab->h_log || !ab->h_deep_log || !ab->h_deep_list) {
         fprintf(stderr, "abuffer: host log mirrors failed; disabled\n");
         ab->disabled = true;
         return false;
      }
      ab->cap_log_pixels = n;
      fprintf(stderr, "abuffer: verification on — peel log %.1f MB, "
              "deep log %.1f MB\n", logb / (1024.0 * 1024.0),
              deepb / (1024.0 * 1024.0));
   }

   ab->w = w;
   ab->h = h;
   ab->nb1 = nb1;
   ab->nb2 = nb2;
   ab->nb3 = nb3;
   ab->quad_width = quad_width;
   ab->nblocks = nblocks;
   ab->bnb1 = bnb1;
   ab->bnb2 = bnb2;
   ab->bnb3 = bnb3;
   ab->ready = true;
   return true;
}

/*
 * What the fragment array ended up holding, against what it was sized for.
 *
 * The one number that says whether the headroom is right: an allocation sized
 * from the first draw and never checked again is how the opt-in version came
 * to be running at 87.1% of capacity without anybody knowing. Printed once, at
 * teardown, and only when asked — CUDAPIPE_ABUFFER_TIMING, which is the switch
 * for "tell me what this path did".
 */
void
cp_abuf_report(void)
{
   struct cp_abuf *ab = &cp_abuf;
   if (!cp_abuf.timing || !ab->peak)
      return;
   fprintf(stderr, "abuffer: peak population %u fragments against a capacity "
           "of %u (%.1f%%), %u growth%s\n", ab->peak, ab->capacity,
           ab->capacity ? 100.0 * ab->peak / ab->capacity : 0.0,
           ab->growths, ab->growths == 1 ? "" : "s");
}

/*
 * Make the per-fragment arrays big enough for a draw of `total` fragments, and
 * big enough that the next few draws will not have to ask again.
 *
 * Called once per eligible draw, after the count pass and before the fill —
 * which is the only moment the host knows the population and nothing has been
 * written into the arrays yet. Never called from inside the peel loop or the
 * merge. Returns false only when there are no usable arrays at all; a growth
 * that is refused leaves the existing ones in place and says so, and the
 * caller then decides whether this particular draw still fits.
 */
bool
cp_abuf_size_arrays(struct cp_abuf *ab, uint32_t total)
{
   bool grow = ab->frags &&
      (double)total > (double)ab->capacity * CP_ABUF_GROW_AT;

   if (ab->frags && !grow)
      return true;
   if (grow && (ab->grow_capped || ab->growths >= CP_ABUF_MAX_GROWTHS))
      return true;

   size_t want = MAX2((size_t)total * CP_ABUF_HEADROOM + CP_ABUF_SLACK,
                      (size_t)CP_ABUF_MIN_FRAGS);
   if (want > CP_ABUF_MAX_FRAGS) {
      if (ab->frags) {
         /* Keep what is there. Draws that fit still take the path; draws that
          * do not fall back, which is what the cap is for. */
         if (!ab->grow_capped)
            fprintf(stderr, "abuffer: %u fragments would want %zu entries, "
                    "over the %u cap — keeping %u and letting oversized draws "
                    "peel\n", total, want, CP_ABUF_MAX_FRAGS, ab->capacity);
         ab->grow_capped = true;
         return true;
      }
      fprintf(stderr, "abuffer: %u fragments needs %zu entries, over the %u "
              "cap — falling back to the peel path and not allocating\n",
              total, want, CP_ABUF_MAX_FRAGS);
      ab->disabled = true;
      return false;
   }

   unsigned was = ab->capacity;

   /* Freed before the new ones are asked for, so a grow needs the new size on
    * the card rather than the old and the new at once. Nothing in them is
    * live: the count pass writes only the per-pixel counters. */
   cp_abuf_free(&ab->frags);
   cp_abuf_free(&ab->recs);
   cp_abuf_free(&ab->quad_prim);
   cp_abuf_free(&ab->quad_mask);
   cp_abuf_free(&ab->quad_block);
   cp_abuf_free(&ab->shade_slot);
   cp_abuf_free(&ab->peel_mask);
   cp_abuf_free(&ab->colors_abuf);
   cp_abuf_free(&ab->writes_abuf);
   cp_abuf_free(&ab->colors_peel);
   cp_abuf_free(&ab->writes_peel);
   ab->capacity = ab->quad_capacity = 0;
   ab->colors_ready = false;

   if (!cp_abuf_alloc(ab, &ab->frags, want * sizeof(uint32_t), "fragments"))
      return false;
   if (!cp_debug->no_abuf_append &&
       !cp_abuf_alloc(ab, &ab->recs, want * sizeof(uint64_t),
                      "fragment records"))
      return false;
   ab->capacity = (unsigned)want;

   /*
    * The quad arrays, sized from the same number. A quad needs at least one
    * covering fragment, so there can never be more quads than fragments —
    * which is what lets these be allocated from a count the host already has,
    * instead of draining the device again to ask how many the merge produced.
    */
   if (!cp_abuf_alloc(ab, &ab->quad_prim, want * sizeof(uint32_t),
                      "quad primitives") ||
       !cp_abuf_alloc(ab, &ab->quad_mask, want, "quad masks") ||
       !cp_abuf_alloc(ab, &ab->quad_block, want * sizeof(uint32_t),
                      "quad blocks") ||
       /* Only the composite reads this one, and only the comparison against
        * the peel loop reads the other. */
       (cp_abuf.composite &&
        !cp_abuf_alloc(ab, &ab->shade_slot, want * sizeof(uint32_t),
                       "shading slots")) ||
       (cp_abuf.verify &&
        !cp_abuf_alloc(ab, &ab->peel_mask, want * sizeof(uint32_t),
                       "peel masks")))
      return false;
   ab->quad_capacity = (unsigned)want;

   /* Thirteen bytes an entry for the quads either way: primitive, mask and
    * block are common, and the fourth word is the shading slot when
    * compositing and the peel mask when checking. */
   if (grow) {
      ab->growths++;
      fprintf(stderr, "abuffer: grew the fragment array %u -> %u entries "
              "(%.1f MB, %.1f MB of quads) — a draw counted %u, which is "
              "%.0f%% of what it had; growth %u of %u\n",
              was, ab->capacity, want * (ab->recs ? 12.0 : 4.0) / (1024.0 * 1024.0),
              want * 13.0 / (1024.0 * 1024.0), total,
              was ? 100.0 * total / was : 0.0, ab->growths,
              CP_ABUF_MAX_GROWTHS);
   } else {
      fprintf(stderr, "abuffer: fragment array %u entries (%.1f MB, %.1f MB "
              "of quads) from a first count of %u\n", ab->capacity,
              want * (ab->recs ? 12.0 : 4.0) / (1024.0 * 1024.0), want * 13.0 / (1024.0 * 1024.0),
              total);
   }

   if (cp_abuf.verify) {
      free(ab->h_frags);
      free(ab->h_quad_prim);
      free(ab->h_quad_mask);
      free(ab->h_peel_mask);
      ab->h_frags = malloc(want * sizeof(uint32_t));
      ab->h_quad_prim = malloc(want * sizeof(uint32_t));
      ab->h_quad_mask = malloc(want);
      ab->h_peel_mask = malloc(want * sizeof(uint32_t));
      if (!ab->h_frags || !ab->h_quad_prim || !ab->h_quad_mask ||
          !ab->h_peel_mask) {
         fprintf(stderr, "abuffer: host mirrors failed; disabled\n");
         ab->disabled = true;
         return false;
      }

      /*
       * One float4 and one write count per A-buffer slot, for each of the two
       * paths. Sized by the same capacity because a slot is what both paths
       * address, and refused rather than allocated past a cap — 40 bytes a
       * slot is 130 MB at the measured population and grows with it.
       */
      size_t cb = want * 16, wb = want * sizeof(uint32_t);
      if (2 * (cb + wb) > CP_ABUF_MAX_COLOR_BYTES) {
         fprintf(stderr, "abuffer: colour comparison would need %.1f MB, over "
                 "the %.0f MB cap — the quad stream is still shaded and timed, "
                 "but not checked\n", 2.0 * (cb + wb) / (1024.0 * 1024.0),
                 CP_ABUF_MAX_COLOR_BYTES / (1024.0 * 1024.0));
         return true;
      }
      if (!cp_abuf_alloc(ab, &ab->colors_abuf, cb, "colours (abuf)") ||
          !cp_abuf_alloc(ab, &ab->writes_abuf, wb, "writes (abuf)") ||
          !cp_abuf_alloc(ab, &ab->colors_peel, cb, "colours (peel)") ||
          !cp_abuf_alloc(ab, &ab->writes_peel, wb, "writes (peel)"))
         return false;
      free(ab->h_colors_abuf);
      free(ab->h_colors_peel);
      free(ab->h_writes_abuf);
      free(ab->h_writes_peel);
      ab->h_colors_abuf = malloc(cb);
      ab->h_colors_peel = malloc(cb);
      ab->h_writes_abuf = malloc(wb);
      ab->h_writes_peel = malloc(wb);
      if (!ab->h_colors_abuf || !ab->h_colors_peel || !ab->h_writes_abuf ||
          !ab->h_writes_peel) {
         fprintf(stderr, "abuffer: colour mirrors failed; disabled\n");
         ab->disabled = true;
         return false;
      }
      ab->colors_ready = true;
      fprintf(stderr, "abuffer: colour comparison on — %.1f MB on the device, "
              "%.1f MB on the host\n", 2.0 * (cb + wb) / (1024.0 * 1024.0),
              2.0 * (cb + wb) / (1024.0 * 1024.0));
   }
   return true;
}

/* counts -> offsets, exclusive, three levels. The grand total is left in
 * sums[0] of the top level on the device. Used over the pixels for the
 * fragment lists and over the 2x2 blocks for the quads. */
void
cp_abuf_scan_n(struct cp_context *cp, struct cp_device *screen,
               CUdeviceptr in, CUdeviceptr out, CUdeviceptr s1, CUdeviceptr s1x,
               CUdeviceptr s2, CUdeviceptr s2x, CUdeviceptr s3,
               unsigned n, unsigned nb1, unsigned nb2, unsigned nb3,
               CUdeviceptr clamp_counts, uint32_t clamp_capacity,
               CUdeviceptr clamp_overflow)
{
   if (nb1 <= CP_ABUF_SCAN_BLOCK) {
      void *p0[] = { &in, &out, &s1, &n };
      CP_LAUNCH(screen->kernels.abuf_scan_block, nb1, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p0, NULL);

      void *p1[] = { &s1, &s1x, &s3, &nb1 };
      CP_LAUNCH(screen->kernels.abuf_scan_block, 1, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p1, NULL);

      void *pa[] = { &out, &s1x, &n, &clamp_counts, &s3,
                     &clamp_capacity, &clamp_overflow };
      CP_LAUNCH(screen->kernels.abuf_scan_add, nb1, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, pa, NULL);
      return;
   }

   struct { CUdeviceptr in, out, sums; unsigned n, grid; } lvl[3] = {
      { in, out, s1, n,   nb1 },
      { s1,  s1x, s2, nb1, nb2 },
      { s2,  s2x, s3, nb2, nb3 },
   };
   for (int i = 0; i < 3; i++) {
      void *p[] = { &lvl[i].in, &lvl[i].out, &lvl[i].sums, &lvl[i].n };
      CP_LAUNCH(screen->kernels.abuf_scan_block, lvl[i].grid, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p, NULL);
   }
   /* Add each level's scanned bases back down. */
   for (int i = 1; i >= 0; i--) {
      CUdeviceptr data = i ? s1x : out;
      CUdeviceptr sums = i ? s2x : s1x;
      CUdeviceptr counts = i ? 0 : clamp_counts;
      CUdeviceptr total = i ? 0 : s3;
      uint32_t capacity = i ? 0 : clamp_capacity;
      CUdeviceptr overflow = i ? 0 : clamp_overflow;
      unsigned cnt = i ? nb1 : n;
      unsigned grid = i ? nb2 : nb1;
      void *p[] = { &data, &sums, &cnt, &counts, &total, &capacity,
                    &overflow };
      CP_LAUNCH(screen->kernels.abuf_scan_add, grid, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p, NULL);
   }
}

void
cp_abuf_scan(struct cp_context *cp, struct cp_device *screen,
             struct cp_abuf *ab, unsigned n)
{
   cp_abuf_scan_n(cp, screen, ab->counts, ab->offsets, ab->sum1, ab->sum1x,
                  ab->sum2, ab->sum2x, ab->sum3, n, ab->nb1, ab->nb2, ab->nb3,
                  ab->counts, ab->capacity, ab->overflow);
}


int
cp_abuf_cmp_deep(const void *a, const void *b)
{
   const struct cp_abuf_deep *x = a, *y = b;
   if (x->count != y->count)
      return x->count > y->count ? -1 : 1;
   return x->pixel < y->pixel ? -1 : x->pixel > y->pixel ? 1 : 0;
}

/*
 * Compare the built lists against what the peel loop actually composited, and
 * against the invariants that hold whatever the loop did.
 *
 * `passes_run` bounds the comparison from the loop's side: the loop stops at
 * CP_BLEND_LAYERS, and the deepest pixels here are deeper than that, so their
 * last entries have no peel sequence to be compared against and are checked
 * against the invariants only. That is counted and reported, not hidden.
 */
void
cp_abuf_verify(struct cp_abuf *ab, unsigned w, unsigned h, uint32_t total,
               unsigned passes_run, unsigned deep_n, uint32_t overflow,
               uint32_t long_runs)
{
   size_t n = (size_t)w * h;
   unsigned bad = 0;
   const unsigned bad_max = 10;

   uint64_t sum_counts = 0;
   size_t covered = 0, len_bad = 0, order_bad = 0;
   for (size_t i = 0; i < n; i++) {
      sum_counts += ab->h_counts[i];
      if (ab->h_counts[i])
         covered++;
      if (ab->h_cursor[i] != ab->h_counts[i])
         len_bad++;
   }

   /* Each run strictly increasing, no duplicates. */
   for (size_t i = 0; i < n; i++) {
      uint32_t c = ab->h_counts[i];
      const uint32_t *run = ab->h_frags + ab->h_offsets[i];
      for (uint32_t k = 1; k < c; k++)
         if (run[k] <= run[k - 1]) {
            order_bad++;
            if (bad++ < bad_max)
               fprintf(stderr, "abuffer: pixel (%zu,%zu) run not increasing at "
                       "%u: %u then %u\n", i % w, i / w, k, run[k - 1], run[k]);
            break;
         }
   }

   /* The shallow comparison: every pixel, the first CP_ABUF_LOG_LAYERS peel
    * passes. */
   uint64_t cmp_entries = 0, cmp_pixels = 0, mism = 0, tail_bad = 0;
   unsigned shallow = MIN2((unsigned)CP_ABUF_LOG_LAYERS, passes_run);
   for (size_t i = 0; i < n; i++) {
      uint32_t c = ab->h_counts[i];
      const uint32_t *run = ab->h_frags + ab->h_offsets[i];
      const uint32_t *log = ab->h_log + i * CP_ABUF_LOG_LAYERS;
      unsigned k_max = MIN2(c, shallow);
      bool pixel_bad = false;
      for (unsigned k = 0; k < k_max; k++) {
         cmp_entries++;
         if (log[k] != run[k]) {
            mism++;
            pixel_bad = true;
         }
      }
      /* Past the end of the run the loop must have found nothing left. */
      for (unsigned k = c; k < shallow; k++)
         if (log[k] != 0xFFFFFFFFu) {
            tail_bad++;
            pixel_bad = true;
         }
      if (k_max)
         cmp_pixels++;
      if (pixel_bad && bad++ < bad_max) {
         fprintf(stderr, "abuffer: MISMATCH pixel (%zu,%zu) count=%u\n"
                 "   peel:", i % w, i / w, c);
         for (unsigned k = 0; k < shallow; k++)
            fprintf(stderr, " %d", (int)log[k]);
         fprintf(stderr, "\n   list:");
         for (unsigned k = 0; k < MIN2(c, shallow); k++)
            fprintf(stderr, " %u", run[k]);
         fprintf(stderr, "\n");
      }
   }

   /* The deep comparison: the deepest pixels, to the full depth the loop ran
    * to. Everything past passes_run has no peel sequence and is skipped. */
   uint64_t deep_entries = 0, deep_mism = 0, deep_unreachable = 0;
   for (unsigned d = 0; d < deep_n; d++) {
      uint32_t p = ab->h_deep_list[d];
      uint32_t c = ab->h_counts[p];
      const uint32_t *run = ab->h_frags + ab->h_offsets[p];
      const uint32_t *log = ab->h_deep_log + (size_t)d * CP_BLEND_LAYERS;
      unsigned k_max = MIN2(c, passes_run);
      bool pixel_bad = false;
      for (unsigned k = 0; k < k_max; k++) {
         deep_entries++;
         if (log[k] != run[k]) {
            deep_mism++;
            pixel_bad = true;
         }
      }
      if (c > passes_run)
         deep_unreachable += c - passes_run;
      if (pixel_bad && bad++ < bad_max) {
         fprintf(stderr, "abuffer: DEEP MISMATCH pixel (%u,%u) count=%u\n",
                 p % w, p / w, c);
         for (unsigned k = 0; k < k_max; k++)
            if (log[k] != run[k]) {
               fprintf(stderr, "   layer %u: peel %d, list %u\n", k,
                       (int)log[k], run[k]);
               break;
            }
      }
   }

   fprintf(stderr,
           "abuffer verify: total=%u sum(counts)=%" PRIu64 " covered=%zu "
           "run_len_bad=%zu not_increasing=%zu overflow=%u long_runs=%u\n"
           "  shallow: %" PRIu64 " entries over %" PRIu64 " pixels, %u layers "
           "deep, mismatches=%" PRIu64 " (tail=%" PRIu64 ")\n"
           "  deep:    %" PRIu64 " entries over %u pixels, up to %u layers, "
           "mismatches=%" PRIu64 "; %" PRIu64 " entries beyond the peel cap "
           "were checked against the invariants only\n"
           "  VERDICT: %s\n",
           total, sum_counts, covered, len_bad, order_bad, overflow, long_runs,
           cmp_entries, cmp_pixels, shallow, mism, tail_bad,
           deep_entries, deep_n, passes_run, deep_mism, deep_unreachable,
           (sum_counts == total && !len_bad && !order_bad && !overflow &&
            !mism && !tail_bad && !deep_mism)
              ? "every list matches the peel sequence"
              : "MISMATCH — see above");
}

/*
 * Check the quad stream: the (2x2 block, primitive, 4-bit coverage mask)
 * triples the merge produced, against what cp_fs_interpolate emitted over
 * every pass of the peel loop.
 *
 * Three separate claims, and they are kept separate on purpose:
 *
 *   1. The merge is the 4-way merge of the block's four pixel lists. Checked
 *      by redoing it here, on the host, from the same lists — an independent
 *      implementation rather than a restatement of the kernel.
 *   2. The invariants: no empty mask, primitives ascending and distinct within
 *      a block, and the set mask bits summing to the fragment total.
 *   3. The triples equal what the peel loop emitted. The peel side emits one
 *      quad per (block, primitive) *per pass*, and one primitive can win
 *      different pixels of a block on different passes, so what it emits is
 *      compared after ORing each block-primitive's masks together — which is
 *      what "summed over the passes" has to mean for the two to be comparable
 *      at all.
 *
 * The peel loop stops at CP_BLEND_LAYERS, so a primitive lying deeper than
 * that in a pixel's list is never selected there and its bit cannot appear on
 * the peel side. Those bits are excluded from claim 3 by rank, and counted.
 */
void
cp_abuf_verify_quads(struct cp_abuf *ab, unsigned w, unsigned h,
                     uint32_t total_frags, uint32_t total_quads,
                     unsigned passes_run, uint32_t quad_overflow,
                     const uint32_t *dbg)
{
   unsigned qw = ab->quad_width;
   unsigned bad = 0;
   const unsigned bad_max = 10;

   uint64_t host_quads = 0, blocks_covered = 0;
   uint64_t count_bad = 0, prim_bad = 0, mask_bad = 0;
   uint64_t empty_mask = 0, not_ascending = 0, bits = 0;

   uint64_t triples_compared = 0, peel_mismatch = 0;
   uint64_t quads_excluded = 0, bits_excluded = 0, peel_extra = 0;

   for (unsigned b = 0; b < ab->nblocks; b++) {
      /* The block's four pixel runs, exactly as the kernel takes them. */
      const uint32_t *run[4];
      uint32_t rn[4], cur[4];
      unsigned qx = (b % qw) * 2, qy = (b / qw) * 2;
      for (int i = 0; i < 4; i++) {
         unsigned x = qx + (i & 1), y = qy + (i >> 1);
         run[i] = NULL;
         rn[i] = 0;
         cur[i] = 0;
         if (x < w && y < h) {
            size_t p = (size_t)y * w + x;
            run[i] = ab->h_frags + ab->h_offsets[p];
            rn[i] = ab->h_counts[p];
         }
      }

      uint32_t gpu_n = ab->h_blk_counts[b];
      uint32_t gpu_at = ab->h_blk_offsets[b];
      uint32_t k = 0, prev = 0;
      bool block_bad = false;

      for (;;) {
         bool have = false;
         uint32_t best = 0;
         for (int i = 0; i < 4; i++)
            if (cur[i] < rn[i] && (!have || run[i][cur[i]] < best)) {
               best = run[i][cur[i]];
               have = true;
            }
         if (!have)
            break;

         /* The mask this primitive should carry, and the part of it the peel
          * loop could ever have seen: a pixel's k-th entry is selected on pass
          * k, so an entry at or past passes_run has no peel counterpart. */
         uint32_t mask = 0, reach = 0, deep = 0;
         for (int i = 0; i < 4; i++)
            while (cur[i] < rn[i] && run[i][cur[i]] == best) {
               mask |= 1u << i;
               if (cur[i] < passes_run)
                  reach |= 1u << i;
               else
                  deep++;
               cur[i]++;
            }

         host_quads++;
         bits += (uint32_t)__builtin_popcount(mask);
         bits_excluded += deep;
         if (!mask)
            empty_mask++;
         if (k && best <= prev)
            not_ascending++;
         prev = best;

         /* Claim 1: the kernel's array, entry for entry. */
         if (k < gpu_n) {
            if (ab->h_quad_prim[gpu_at + k] != best) {
               prim_bad++;
               block_bad = true;
            } else if (ab->h_quad_mask[gpu_at + k] != (unsigned char)mask) {
               mask_bad++;
               block_bad = true;
            }
         }

         /* Claim 3, restricted to the peel-reachable part of the mask. */
         uint32_t got = k < gpu_n ? ab->h_peel_mask[gpu_at + k] : 0;
         if (reach) {
            triples_compared++;
            if (got != reach) {
               peel_mismatch++;
               block_bad = true;
               if (bad++ < bad_max)
                  fprintf(stderr, "abuffer quads: MISMATCH block (%u,%u) "
                          "prim=%u  merged mask=0x%x reachable=0x%x  "
                          "peel emitted=0x%x\n", b % qw, b / qw, best,
                          mask, reach, got);
            }
         } else {
            quads_excluded++;
            if (got) {
               peel_extra++;
               block_bad = true;
               if (bad++ < bad_max)
                  fprintf(stderr, "abuffer quads: block (%u,%u) prim=%u lies "
                          "past the peel cap yet peel emitted mask 0x%x\n",
                          b % qw, b / qw, best, got);
            }
         }
         k++;
      }

      if (k)
         blocks_covered++;
      if (k != gpu_n) {
         count_bad++;
         block_bad = true;
      }
      if (block_bad && bad++ < bad_max)
         fprintf(stderr, "abuffer quads: block (%u,%u) host=%u quads, "
                 "kernel=%u at offset %u\n", b % qw, b / qw, k, gpu_n, gpu_at);
   }

   uint64_t peel_quads = dbg[CP_ABUF_DBG_PEEL_QUADS];
   bool ok = !count_bad && !prim_bad && !mask_bad && !empty_mask &&
             !not_ascending && !peel_mismatch && !peel_extra &&
             !quad_overflow && host_quads == total_quads &&
             bits == total_frags && !dbg[CP_ABUF_DBG_NOT_FOUND] &&
             !dbg[CP_ABUF_DBG_FULL];

   fprintf(stderr,
           "abuffer quads: %" PRIu64 " quads over %" PRIu64 " covered blocks "
           "of %u (%.1f%%); kernel says %u, overflow=%u\n"
           "  merge vs host: count_bad=%" PRIu64 " prim_bad=%" PRIu64
           " mask_bad=%" PRIu64 "\n"
           "  invariants: empty_mask=%" PRIu64 " not_ascending=%" PRIu64
           "  sum(popcount)=%" PRIu64 " vs %u fragments\n"
           "  peel emitted %" PRIu64 " quads over %u passes; not_found=%u "
           "degenerate=%u dropped_full=%u\n"
           "  triples compared=%" PRIu64 " mismatches=%" PRIu64
           " peel_extra=%" PRIu64 "\n"
           "  excluded by the %u-pass cap: %" PRIu64 " quads entirely, "
           "%" PRIu64 " of %" PRIu64 " coverage bits (%.1f%%)\n"
           "  VERDICT: %s\n",
           host_quads, blocks_covered, ab->nblocks,
           100.0 * (double)blocks_covered / (double)ab->nblocks,
           total_quads, quad_overflow,
           count_bad, prim_bad, mask_bad,
           empty_mask, not_ascending, bits, total_frags,
           peel_quads, passes_run, dbg[CP_ABUF_DBG_NOT_FOUND],
           dbg[CP_ABUF_DBG_DEGENERATE], dbg[CP_ABUF_DBG_FULL],
           triples_compared, peel_mismatch, peel_extra,
           passes_run, quads_excluded, bits_excluded, bits,
           bits ? 100.0 * (double)bits_excluded / (double)bits : 0.0,
           ok ? "the quad stream is what the peel loop emitted"
              : "MISMATCH — see above");
}

/*
 * ---------------------------------------------------------------------------
 * TEMPORARY: step 3b — shade the quad stream (CUDAPIPE_ABUFFER)
 * ---------------------------------------------------------------------------
 *
 * The merge produced one (block, primitive, 4-bit mask) triple per distinct
 * primitive per 2x2 block, for every layer at once. That is precisely what
 * cp_fs_interpolate emits, one pass at a time, so the same four parallel
 * arrays can be filled from it and the same fragment shader run over them.
 *
 * Nothing here composites: the draw is still rendered by the peel loop, and
 * the colours this produces are written into the A-buffer's own slots to be
 * compared against the ones the loop shaded, not into the framebuffer.
 *
 * The interpolation itself is cp_interp_pixel, the function the peel
 * interpolator calls — see cp_abuf_interpolate in cp_fs.cu. Two copies of
 * perspective-correct interpolation that can drift is the failure this driver
 * has already been bitten by.
 */

/* Sortable-uint form of a depth value: monotonic in the float, so the
 * rasterizer's integer compares order the same way floats would. */
uint32_t
cp_depth_to_sortable(float depth)
{
   union { float f; uint32_t u; } v = { .f = depth };
   uint32_t mask = -((int32_t)v.u >> 31) | 0x80000000u;
   return v.u ^ mask;
}

void
cp_clear_depthbuf(struct cp_context *cp, float depth)
{
   if (!cp->depthbuf)
      return;

   uint32_t value = cp_depth_to_sortable(depth);
   size_t count = (size_t)cp->depthbuf_w * cp->depthbuf_h;

   cuCtxSetCurrent(cp->screen->cuda_ctx);
   cuMemsetD32Async(cp->depthbuf, value, count * MAX2(cp->visbuf_samples, 1u), cp->stream);
   cp->depthbuf_cleared = true;
}

/* Resolve every interval recorded this draw into per-stage totals. Costs one
 * synchronisation, which is why it is debug-only. */
void
cp_stage_resolve(struct cp_context *cp, double *ms)
{
   struct cp_stage_timer *t = &cp->timer;
   if (t->num < 2) {
      t->num = 0;
      return;
   }

   cuEventSynchronize(t->events[t->num - 1]);
   for (unsigned i = 1; i < t->num; i++) {
      float dt = 0.0f;
      int stage = t->stages[i];
      if (stage >= 0 && stage < CP_NUM_STAGES &&
          cuEventElapsedTime(&dt, t->events[i - 1], t->events[i]) == CUDA_SUCCESS)
         ms[stage] += dt;
   }
   t->num = 0;
}

/*
 * Which shader I/O slot carries a given varying location, or -1 if none does.
 * gl_PointSize and gl_PointCoord reach the kernels this way like any other
 * varying, rather than through a dedicated path.
 */
int32_t
cp_slot_for_location(const unsigned *locations, unsigned count,
                     unsigned location)
{
   for (unsigned i = 0; i < count && i < CP_MAX_IO_SLOTS; i++)
      if (locations[i] == location)
         return (int32_t)i;
   return -1;
}


/* Number of triangles one draw of `count` vertices produces. */
unsigned
cp_triangles_for_draw(enum mesa_prim mode, unsigned count)
{
   if (mode == MESA_PRIM_TRIANGLE_STRIP || mode == MESA_PRIM_TRIANGLE_FAN)
      return count >= 3 ? count - 2 : 0;
   if (mode == MESA_PRIM_POINTS)
      return count;
   return count / 3;
}

/*
 * Resolve every assembled vertex once: expand the primitive topology, apply
 * the index buffer, and repeat the whole thing per instance.
 *
 * Everything downstream (positions, shader inputs, vertex ids) indexes this
 * array, so the topology and indexing rules live in exactly one place.
 */
struct cp_vertex_ref *
cp_build_vertex_refs(const struct cp_draw_call *info,
                     const struct cp_draw_range *draws,
                     unsigned num_draws, unsigned instance_count,
                     const void *ib_base, unsigned num_triangles)
{
   struct cp_vertex_ref *refs =
      MALLOC(sizeof(*refs) * num_triangles * 3);
   if (!refs)
      return NULL;

   bool indexed = info->index_size > 0;
   unsigned index_size = info->index_size;
   unsigned out_tri = 0;

   for (unsigned inst = 0; inst < instance_count; inst++) {
      for (unsigned d = 0; d < num_draws; d++) {
         unsigned count = draws[d].count;
         unsigned first = draws[d].start;
         int base_vertex = indexed ? draws[d].index_bias : 0;

         const void *ib_data = NULL;
         if (indexed && ib_base)
            ib_data = (const char *)ib_base + (size_t)first * index_size;

         unsigned draw_tris = cp_triangles_for_draw(info->mode, count);

         for (unsigned tri = 0; tri < draw_tris; tri++) {
            unsigned idx[3];
            if (info->mode == MESA_PRIM_POINTS) {
               /* Each point becomes a degenerate triangle: the rasterizer will
                * expand it into a screen-aligned quad later using point size. */
               idx[0] = tri;
               idx[1] = tri;
               idx[2] = tri;
            } else if (info->mode == MESA_PRIM_TRIANGLE_STRIP) {
               /* Odd triangles swap two vertices to keep the winding. */
               idx[0] = tri;
               idx[1] = tri + 1 + (tri & 1);
               idx[2] = tri + 2 - (tri & 1);
            } else if (info->mode == MESA_PRIM_TRIANGLE_FAN) {
               idx[0] = 0;
               idx[1] = tri + 1;
               idx[2] = tri + 2;
            } else {
               idx[0] = tri * 3 + 0;
               idx[1] = tri * 3 + 1;
               idx[2] = tri * 3 + 2;
            }

            for (unsigned vi = 0; vi < 3; vi++) {
               unsigned vertex;
               if (indexed && ib_data) {
                  unsigned raw = index_size == 2
                     ? ((const uint16_t *)ib_data)[idx[vi]]
                     : ((const uint32_t *)ib_data)[idx[vi]];
                  vertex = (unsigned)((int)raw + base_vertex);
               } else {
                  vertex = first + idx[vi];
               }
               refs[out_tri * 3 + vi].vertex = vertex;
               /* Zero-based, matching load_instance_id. The first instance
                * offset belongs to attribute fetch, not to the shader's
                * instance id. */
               refs[out_tri * 3 + vi].instance = inst;
            }
            out_tri++;
         }
      }
   }

   return refs;
}

/* 0 = keep everything, 1 = drop positive-area triangles, 2 = drop negative. */
uint32_t
cp_cull_mode(const struct cp_raster_state *rs)
{
   bool cull_back = (rs->cull_face & CP_FACE_BACK) != 0;
   bool cull_front = (rs->cull_face & CP_FACE_FRONT) != 0;

   if (!cull_back && !cull_front)
      return 0;
   if (cull_back && cull_front)
      return 0;   /* handled by skipping the draw */

   /* After the viewport transform a front face has positive area when the
    * front is counter-clockwise, since Vulkan's clip space already has y
    * running downward and the viewport scale does not flip it again. */
   if (cull_back)
      return rs->front_ccw ? 2 : 1;
   return rs->front_ccw ? 1 : 2;
}

/*
 * The draw's blend equation, in the form both kernels that evaluate it read.
 *
 * Split out for the same reason the struct is shared: the peel path's
 * writeback and the A-buffer's composite have to be blending the same draw the
 * same way, and a second place that turns a pipe_rt_blend_state into one is a
 * second place that can forget a field.
 */
struct cp_blend_desc
cp_blend_desc_for(const struct cp_context *cp)
{
   /* Resolved once when the state was bound; see cp_bind_blend_state. */
   return cp->blend_desc;
}
