/*
 * The renderer, compiled into both front ends.
 *
 * This file exists so that code the draw pipeline needs can be shared by the
 * Gallium-hosted driver and the native Vulkan one without either including
 * the other's headers. It starts with bring-up; the rest of the pipeline
 * follows it out of cp_context.c as each piece stops depending on Gallium.
 */

#include "cp_renderer.h"
#include "cp_nvtx.h"
#include "compiler/glsl_types.h"

/*
 * TEMPORARY (CUDAPIPE_ABUFFER): the merged quad array, for the one draw whose
 * peel loop is being compared against it. Zero for every other draw, and for
 * every draw once the verification budget is spent, so the instrumented
 * interpolator is not carried by frames that are only being timed.
 */
struct cp_abuf_dbg_state cp_abuf_dbg;

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

/*
 * Which vertex shader output drives each fragment shader input, and the few
 * pieces of state the interpolator needs beyond its buffers.
 *
 * Split out because the A-buffer path interpolates the same varyings from a
 * different source and must not describe them differently: a second copy of
 * this matching is a second thing that can be right about a shader the first
 * one is wrong about.
 */
static void
cp_fs_interp_setup(struct cp_context *cp, const struct cp_draw_call *info,
                   const struct cp_shader_binary *fs, unsigned num_fs_inputs,
                   unsigned num_vs_outputs, struct cp_fs_interp_args *interp)
{
   interp->num_samples = MAX2(cp->fb_samples, 1u);
   interp->point_mode = info->mode == MESA_PRIM_POINTS;
   interp->psiz_slot = cp_slot_for_location(cp->vs_shader->out_location,
                                            num_vs_outputs, VARYING_SLOT_PSIZ);
   /* gl_PointCoord is a fragment shader input that no vertex shader output
    * drives, so the match below leaves it at -1 and the interpolator fills it
    * from the pixel's position within the point. */
   interp->pntc_input = cp_slot_for_location(fs->in_location, num_fs_inputs,
                                             VARYING_SLOT_PNTC);
   /*
    * gl_FragCoord is a shader_in at VARYING_SLOT_POS, so the match below pairs
    * it with the vertex shader's gl_Position and interpolates clip space into
    * it. The window coordinate the shader is owed is what the interpolator
    * already computes for the frag_coord array, so name the slot and let it
    * write that instead. Without this every gl_FragCoord read a fragment
    * shader makes returns the clip-space position — which for the `oit`
    * geometry pass indexes its head-index image at negative coordinates.
    */
   interp->pos_input = cp_slot_for_location(fs->in_location, num_fs_inputs,
                                            VARYING_SLOT_POS);

   /* Match each fragment shader input to the vertex shader output carrying the
    * same varying location. */
   for (unsigned i = 0; i < num_fs_inputs; i++) {
      interp->input_vs_slot[i] = -1;
      unsigned location = fs->in_location[i];
      if (location == VARYING_SLOT_MAX)
         continue;
      for (unsigned o = 0; o < num_vs_outputs && o < CP_MAX_IO_SLOTS; o++) {
         if (cp->vs_shader->out_location[o] == location) {
            interp->input_vs_slot[i] = (int32_t)o;
            break;
         }
      }
   }
}

/*
 * Deciding, per fragment shader, whether capping its registers pays.
 *
 * The compiler marks a shader as a candidate when its own register count is
 * what holds its occupancy below two blocks an SM — see
 * load_shader_module_tuned(). It cannot do more than mark it, because the
 * register count of every textured fragment shader in this driver is the same
 * 195: that is the linked sampler's allocation and not the shader's, so it
 * says nothing about the shader it belongs to. Measured, the same 195 -> 126
 * cap is worth −9.0% on gltfscenerendering and +3.2% on bloom, and inside
 * bloom one capped shader gets 4% faster while the one that owns the frame
 * gets 11% slower. Nothing available before the shader runs separates those.
 *
 * What separates them is running it. A candidate is timed on the device for
 * CP_TUNE_SAMPLES launches as built and CP_TUNE_SAMPLES launches capped, and
 * the faster build is kept for the rest of the process. The comparison is
 * per shader and on this application's own draws, which is the only place the
 * answer exists.
 *
 * Four things keep it honest and cheap:
 *
 * - **The trial is the shader's, not the driver's.** Every candidate carries
 *   its own events and its own phase, so twenty-five materials settle over the
 *   same few frames instead of queueing behind one another. Nothing is shared
 *   between them, and a launch is bracketed by its own two events on one
 *   stream, so what is timed is that kernel and nothing else.
 * - **Nothing waits.** The events of a finished phase are read when the device
 *   has got to them — cuEventQuery on the last one, at whatever launch comes
 *   next — and never waited for. Draining the stream at the two phase
 *   boundaries instead, which is the obvious way to write this, cost
 *   gltfscenerendering 0.9 ms of frame across a 600-frame run: the host runs
 *   well ahead of the device here, so each of the fifty drains gives up a
 *   whole pipeline's depth. It is the single largest thing measured in this
 *   pass, and it is entirely the measurement's own cost.
 * - **The first launches of a phase are thrown away.** The launches after a
 *   build is switched in pay for cold instruction caches and the driver's
 *   first-launch work on it, which is not what is being compared.
 * - **The median is compared, not the mean.** A fragment shader's launches
 *   vary by two orders of magnitude across draws in this set, and the capped
 *   phase is a different set of draws from the uncapped one.
 *
 * **The trial is a veto, not an election.** The cap is taken unless the capped
 * build is CP_TUNE_VETO worse, rather than only when it is measurably better,
 * and the asymmetry is what the measurements are shaped like: across the sweep
 * a capped shader either swings by 12-40% or sits within a few percent of
 * where it started, and nothing lands in between. A 40% swing is a fact about
 * the shader; a 2% one is a fact about the twenty-four draws that happened to
 * be timed, and reruns of the same shader move it by that much on their own. The trial runs in the samples' warm-up, which renders frame 0 over
 * and over, so it sees one camera position of a scene the run then orbits
 * around. Requiring the cap to prove itself there leaves nine of Sponza's
 * twenty-five materials as built and measures 0.1-0.9 ms/frame worse than
 * capping them; requiring the veto to prove itself keeps them, and still
 * throws out the shader that owns bloom's frame, which is 12% worse capped.
 *
 * The whole cost is 48 event records per candidate shader, in the first
 * frames it is used in — inside the warm-up second for the sweep. Nothing is
 * timed, rebuilt or waited for after that, and a shader the compiler did not
 * mark is never touched at all.
 *
 * CUDAPIPE_TUNE_VETO overrides the threshold, and it is the knob to reach for
 * if a sample regresses: it trades what the marginal shaders are worth on the
 * samples that gain against what they cost on the samples that do not.
 * CUDAPIPE_NO_REGCAP turns the whole thing off.
 */

static int
cp_tune_cmp_float(const void *a, const void *b)
{
   float x = *(const float *)a, y = *(const float *)b;
   return x < y ? -1 : x > y ? 1 : 0;
}

/* Median of one phase's samples, in microseconds. */
static double
cp_tune_median(struct cp_shader_tune *t, int phase)
{
   float v[CP_TUNE_SAMPLES];
   memcpy(v, t->us[phase], sizeof(v));
   qsort(v, CP_TUNE_SAMPLES, sizeof(v[0]), cp_tune_cmp_float);
   return 0.5 * (v[CP_TUNE_SAMPLES / 2] + v[(CP_TUNE_SAMPLES - 1) / 2]);
}

static void
cp_tune_release(struct cp_shader_binary *fs)
{
   struct cp_shader_tune *t = &fs->tune;
   if (t->events_made) {
      for (unsigned i = 0; i < CP_TUNE_SAMPLES; i++) {
         cuEventDestroy(t->start[i]);
         cuEventDestroy(t->stop[i]);
      }
      t->events_made = false;
   }
   fs->tune_done = true;
}

/* Read a finished phase's events back, if the last of them has completed.
 * Nothing waits: a phase that is not ready yet is read at the next launch. */
static bool
cp_tune_harvest(struct cp_shader_tune *t)
{
   if (cuEventQuery(t->stop[CP_TUNE_SAMPLES - 1]) != CUDA_SUCCESS)
      return false;
   for (unsigned i = 0; i < CP_TUNE_SAMPLES; i++) {
      float ms = 0;
      cuEventElapsedTime(&ms, t->start[i], t->stop[i]);
      t->us[t->phase][i] = ms * 1000.0f;
   }
   return true;
}

/* Called just before a fragment shader launch. Returns true if this launch is
 * being timed. cp_tune_after() is called after every launch either way, and
 * is where a change of build takes effect. */
static bool
cp_tune_before(struct cp_context *cp, struct cp_shader_binary *fs)
{
   if (cp_debug->no_regcap || !fs->tune_cap || fs->tune_done)
      return false;

   struct cp_shader_tune *t = &fs->tune;

   /* A phase whose events are still in flight. Keep launching what is bound
    * — a few extra launches of either build cost nothing — and read them when
    * the device has got to them. */
   if (t->reading) {
      if (!cp_tune_harvest(t))
         return false;
      t->reading = false;

      if (t->phase == 0) {
         /* Back to the build the JIT chose, which is already loaded. */
         t->regs_capped = fs->num_regs;
         t->swap_pending = true;
         t->phase = 1;
         /* Not zero: the swap is already scheduled, and seen == 0 is what
          * schedules one. Counting from one skips CP_TUNE_SKIP launches of
          * the build being switched to, exactly as phase 0 did. */
         t->seen = 1;
         t->timed = 0;
         return false;
      }

      double capped = cp_tune_median(t, 0), as_built = cp_tune_median(t, 1);
      bool keep = capped < as_built * cp_debug->tune_veto;
      if (keep)
         t->swap_pending = true;   /* back to capped, on the next launch */

      if (cp_debug->shader_stats)
         fprintf(stderr, "cudapipe: shader trial regs %3d -> %3d (cap %d): "
                 "%.1f us -> %.1f us median of %d, %s\n",
                 fs->num_regs, t->regs_capped, fs->tune_cap, as_built, capped,
                 CP_TUNE_SAMPLES, keep ? "CAPPED" : "left as built");

      /* The events go now; a swap still pending is applied by the launch this
       * call is about to let through, which no longer times anything. */
      cp_tune_release(fs);
      return false;
   }

   if (t->seen++ == 0) {
      /* The capped build goes first, so that the two phases are the same
       * shape: each starts with a switch of build and then skips launches.
       * Swapping here rather than before this launch keeps the sampler
       * globals with the module they were written to — the swap takes effect
       * on the next launch, which is one of the skipped ones. */
      t->swap_pending = true;
      return false;
   }
   if (t->seen <= CP_TUNE_SKIP)
      return false;

   if (!t->events_made) {
      for (unsigned i = 0; i < CP_TUNE_SAMPLES; i++) {
         if (cuEventCreate(&t->start[i], CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
             cuEventCreate(&t->stop[i], CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
            fs->tune_done = true;
            return false;
         }
      }
      t->events_made = true;
   }

   cuEventRecord(t->start[t->timed], cp->stream);
   return true;
}

/* Called after every fragment shader launch: applies a pending change of
 * build, and closes a timed launch's event pair. */
static void
cp_tune_after(struct cp_context *cp, struct cp_shader_binary *fs, bool timed)
{
   struct cp_shader_tune *t = &fs->tune;

   if (t->swap_pending) {
      cp_shader_swap_build(fs);
      t->swap_pending = false;
   }
   if (!timed)
      return;

   cuEventRecord(t->stop[t->timed], cp->stream);
   if (++t->timed >= CP_TUNE_SAMPLES)
      t->reading = true;
}

/*
 * Launch the compiled fragment shader over a prepared input buffer.
 *
 * The shader reads its arguments through the same pointer-array ABI the
 * compute path uses; see cp_launch_grid(). Split out for the same reason as
 * the matching above — the A-buffer path shades the same shader over its own
 * buffers, and an ABI described in two places is an ABI that will be described
 * differently.
 */
static bool
cp_fs_launch_shader(struct cp_context *cp, struct cp_shader_binary *fs,
                    CUdeviceptr counter, CUdeviceptr fs_in,
                    unsigned fs_in_stride, CUdeviceptr fs_out,
                    CUdeviceptr frag_coord, CUdeviceptr discard_mask,
                    CUdeviceptr front_face, CUdeviceptr coverage,
                    CUdeviceptr fused_interp, unsigned num_threads,
                    CUevent ev_before,
                    CUdeviceptr batch_rows)
{
   /*
    * The argument block, and behind it the per-draw uniform table the shader
    * indexes into — one upload, because the block holds the table's device
    * address. The shape is the vertex stage's, for the same reason and with
    * the same single-draw degeneracy: one row, a zero mask and a one-word row
    * array holding zero, so a draw that is not a batch computes exactly the
    * args[18 + i] it always did. See CP_ARG_SLOT_UBO_TABLE.
    */
   /* A shader that reads no constant buffer dereferences none of this; one
    * row keeps the upload at its unbatched size instead of scaling the block
    * by the draw count for a table nothing loads. */
   const uint64_t *tbl_src = cp->fs_batch.ubos;
   unsigned rows = (tbl_src && fs->reads_const_bufs)
      ? MAX2(cp->fs_batch.ndraws, 1u) : 1;
   struct cp_sampler_info resolved_samplers[CP_MAX_TEX_DESCS];
   bool samplers_resolved = !cp_debug->no_sampler_variant &&
      fs->num_tex_descs == 1 && !fs->tex_descs_dynamic && tbl_src;
   for (unsigned i = 0; i < fs->num_tex_descs && samplers_resolved; i++) {
      const struct cp_tex_desc_ref *ref = &fs->tex_descs[i];
      bool have_state = false;
      if (ref->ubo_slot >= CP_ARG_UBO_STRIDE) {
         samplers_resolved = false;
         break;
      }
      for (unsigned r = 0; r < rows; r++) {
         const uint64_t *row = tbl_src + (size_t)r * CP_ARG_UBO_STRIDE;
         const char *base = (const char *)(uintptr_t)row[ref->ubo_slot];
         if (!base) {
            samplers_resolved = false;
            break;
         }
         unsigned index = *(const unsigned *)(base + ref->sampler_offset +
                                              CP_DESC_SAMPLER_INDEX_OFFSET);
         if (index >= cp->num_samplers) {
            samplers_resolved = false;
            break;
         }
         const struct cp_sampler_info *state =
            &cp->sampler_table_host[index];
         if (have_state && memcmp(&resolved_samplers[i], state,
                                  sizeof(*state))) {
            samplers_resolved = false;
            break;
         }
         memcpy(&resolved_samplers[i], state, sizeof(*state));
         have_state = true;
      }
   }
   struct cp_sampler_variant *sampler_variant = samplers_resolved
      ? cp_shader_find_sampler_variant(fs, resolved_samplers,
                                       fs->num_tex_descs) : NULL;
   if (samplers_resolved && !sampler_variant &&
       fs->num_sampler_variants < CP_MAX_SAMPLER_VARIANTS &&
       (!fs->tune_cap || fs->tune_done)) {
      char *sampler_ptx = cp_compile_sampler_variant(cp->screen->sm_major,
                                                cp->screen->sm_minor,
                                                     resolved_samplers);
      if (sampler_ptx) {
         cp_shader_build_sampler_variant(fs, sampler_ptx, resolved_samplers,
                                         fs->num_tex_descs);
         free(sampler_ptx);
      }
      sampler_variant = cp_shader_find_sampler_variant(
         fs, resolved_samplers, fs->num_tex_descs);
   }
   bool use_sampler_variant = sampler_variant != NULL;
   CUmodule launch_module = use_sampler_variant
      ? sampler_variant->module : fs->module;
   CUfunction launch_kernel = use_sampler_variant
      ? sampler_variant->kernel : fs->kernel;
   if (cp_debug->debug_tex && !fs->tex_descs_reported) {
      fs->tex_descs_reported = true;
      fprintf(stderr, "cudapipe: FS sampler refs=%u dynamic=%u rows=%u\n",
              fs->num_tex_descs, fs->tex_descs_dynamic, rows);
      for (unsigned r = 0; r < rows; r++) {
         const uint64_t *row = tbl_src
            ? tbl_src + (size_t)r * CP_ARG_UBO_STRIDE : NULL;
         for (unsigned i = 0; i < fs->num_tex_descs; i++) {
            const struct cp_tex_desc_ref *ref = &fs->tex_descs[i];
            const char *base = row && ref->ubo_slot < CP_ARG_UBO_STRIDE
               ? (const char *)(uintptr_t)row[ref->ubo_slot] : NULL;
            unsigned sampler = base
               ? *(const unsigned *)(base + ref->sampler_offset +
                                      CP_DESC_SAMPLER_INDEX_OFFSET) : 0;
            fprintf(stderr, "  row=%u ubo=%u offset=%u sampler=%u\n",
                    r, ref->ubo_slot, ref->sampler_offset, sampler);
         }
      }
   }
   const size_t fs_args_bytes = 64 * sizeof(void *);
   const size_t fs_scal_off = fs_args_bytes;   /* row 0, then the mask */
   const size_t fs_tbl_off = fs_args_bytes + 16;
   size_t fs_blk_bytes = fs_tbl_off +
      (size_t)rows * CP_ARG_UBO_STRIDE * sizeof(uint64_t);

   void *fs_blk = NULL;
   CUdeviceptr fs_args_dev = cp_upload_begin(cp, fs_blk_bytes, &fs_blk);
   if (!fs_args_dev)
      return false;
   memset(fs_blk, 0, fs_blk_bytes);

   void **fs_args_host = (void **)fs_blk;
   fs_args_host[0] = (void *)(uintptr_t)counter;
   fs_args_host[2] = (void *)(uintptr_t)fs_in;
   fs_args_host[3] =
      (void *)(uintptr_t)(fs_args_dev + fs_scal_off + 2 * sizeof(uint32_t));
   fs_args_host[4] = (void *)(uintptr_t)fs_out;
   fs_args_host[6] = (void *)(uintptr_t)frag_coord;
   fs_args_host[CP_ARG_SLOT_DISCARD] = (void *)(uintptr_t)discard_mask;
   fs_args_host[CP_ARG_SLOT_FRONT_FACE] = (void *)(uintptr_t)front_face;
   /* Only a shader that writes memory reads this, and only such a shader is
    * given it — see CP_ARG_SLOT_COVERAGE. */
   fs_args_host[CP_ARG_SLOT_COVERAGE] =
      fs->writes_memory ? (void *)(uintptr_t)coverage : NULL;
   fs_args_host[CP_ARG_SLOT_FUSED_INTERP] =
      (void *)(uintptr_t)fused_interp;
   fs_args_host[CP_ARG_SLOT_UBO_TABLE] =
      (void *)(uintptr_t)(fs_args_dev + fs_tbl_off);
   fs_args_host[CP_ARG_SLOT_BATCH_ROWS] = batch_rows
      ? (void *)(uintptr_t)batch_rows
      : (void *)(uintptr_t)(fs_args_dev + fs_scal_off);
   fs_args_host[CP_ARG_SLOT_BATCH_MASK] =
      (void *)(uintptr_t)(fs_args_dev + fs_scal_off + 4);

   ((uint32_t *)((char *)fs_blk + fs_scal_off))[0] = 0;
   ((uint32_t *)((char *)fs_blk + fs_scal_off))[1] =
      batch_rows ? 0xFFFFFFFFu : 0u;
   ((uint32_t *)((char *)fs_blk + fs_scal_off))[2] = fs_in_stride;

   uint64_t *fs_tbl = (uint64_t *)((char *)fs_blk + fs_tbl_off);
   for (unsigned d = 0; d < rows; d++) {
      const uint64_t *row = tbl_src ? tbl_src + (size_t)d * CP_ARG_UBO_STRIDE
                                    : NULL;
      for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
         fs_tbl[d * CP_ARG_UBO_STRIDE + i] =
            row ? row[i] : (uint64_t)(uintptr_t)cp->fs_ubos[i].buffer;
   }

   /* Still written, so that the block reads the same whichever form a stage
    * takes its bindings from — row zero, not the live binding, since a
    * deferred draw's is no longer what is bound. */
   for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
      fs_args_host[18 + i] =
         (void *)(uintptr_t)fs_tbl[i];

   cp_upload_end(cp, fs_args_dev, fs_blk, fs_blk_bytes);

   if (cp_debug->debug_tex) {
      fprintf(stderr, "cudapipe: sampler table %p (%u entries) for FS module\n",
              (void *)(uintptr_t)cp->sampler_table, cp->num_samplers);
   }

   /*
    * Hand the linked sampler the state it reads through module globals: the
    * sampler table and the fact that fragment threads are laid out four to a
    * quad, so it may take derivatives by shuffling between them.
    *
    * Resolved once per module and written only when the value changes. Both
    * used to be looked up and copied on every draw — four host API calls to
    * store sixteen bytes that are the same as last time, on a path where a
    * draw of a dozen triangles is limited by how fast the host can issue
    * calls rather than by anything the device does.
    */
   {
      CUdeviceptr sym;
      size_t sym_size;
      bool *globals_resolved = use_sampler_variant
         ? &sampler_variant->globals_resolved : &fs->globals_resolved;
      CUdeviceptr *sym_sampler_table = use_sampler_variant
         ? &sampler_variant->sym_sampler_table : &fs->sym_sampler_table;
      CUdeviceptr *sym_quad_derivs = use_sampler_variant
         ? &sampler_variant->sym_quad_derivs : &fs->sym_quad_derivs;
      uint64_t *last_sampler_table = use_sampler_variant
         ? &sampler_variant->last_sampler_table : &fs->last_sampler_table;
      int *last_quad_derivs = use_sampler_variant
         ? &sampler_variant->last_quad_derivs : &fs->last_quad_derivs;
      if (!*globals_resolved) {
         if (cuModuleGetGlobal(&sym, &sym_size, launch_module,
                               "cp_sampler_table") == CUDA_SUCCESS)
            *sym_sampler_table = sym;
         if (cuModuleGetGlobal(&sym, &sym_size, launch_module,
                               "cp_quad_derivs") == CUDA_SUCCESS)
            *sym_quad_derivs = sym;
         /* Nothing has been written yet, and zero is a value the sampler
          * table can legitimately take, so neither cache is valid until the
          * first write below. */
         *last_sampler_table = ~(uint64_t)0;
         *last_quad_derivs = -1;
         *globals_resolved = true;
      }

      if (cp->sampler_table && *sym_sampler_table &&
          *last_sampler_table != (uint64_t)cp->sampler_table) {
         uint64_t addr = (uint64_t)cp->sampler_table;
         cuMemcpyHtoD(*sym_sampler_table, &addr, sizeof(addr));
         *last_sampler_table = addr;
      }
      if (*sym_quad_derivs && *last_quad_derivs != 1) {
         int on = 1;
         cuMemcpyHtoD(*sym_quad_derivs, &on, sizeof(on));
         *last_quad_derivs = 1;
      }
   }

   void *fs_arg_ptr = (void *)(uintptr_t)fs_args_dev;
   void *fs_params[] = { &fs_arg_ptr };
   {
      /* Scoped: the launch check returns out of the middle. */
      CP_NVTX_SCOPE("fs");
      /* TEMPORARY: recorded here rather than by the caller so that a timing of
       * the shader does not also contain the two small host-to-device copies
       * above, which are the ABI rather than the shading. */
      if (ev_before)
         cuEventRecord(ev_before, cp->stream);
      /* A shader the compiler marked as a register-cap candidate is timed
       * here, both as built and capped, and the faster build kept. */
      bool timed = use_sampler_variant ? false : cp_tune_before(cp, fs);
      /* Compiled shaders grid-stride now, so the launch is capped: the count
       * is device-side and num_threads is the framebuffer's worst case, so a
       * small draw's launch was mostly scheduling idle blocks. 4096 blocks
       * of 256 is far past what fills the machine. */
      CUresult fs_err = cuLaunchKernel(launch_kernel,
                                       MIN2((num_threads + 255) / 256, 4096u),
                                       1, 1, 256, 1, 1, 0, cp->stream,
                                       fs_params, NULL);
      if (fs_err != CUDA_SUCCESS) {
         fprintf(stderr, "cudapipe: fragment shader launch failed (%d)\n",
                 fs_err);
         return false;
      }
      if (!use_sampler_variant)
         cp_tune_after(cp, fs, timed);
   }
   return true;
}

/*
 * Run the fragment shader over every pixel the rasterizer covered.
 *
 * Three launches: gather the shader's inputs (which also compacts the covered
 * pixels into a list), run the shader itself one thread per covered pixel, and
 * blend its output into the colour attachment.
 */
void
cp_shade_fragments(struct cp_context *cp, const struct cp_draw_call *info,
                   CUdeviceptr visbuf, CUdeviceptr positions,
                   CUdeviceptr vs_output_buf, unsigned num_triangles,
                   unsigned w, unsigned h, void *color_data,
                   float vp_scale_x, float vp_scale_y,
                   float vp_trans_x, float vp_trans_y,
                   CUdeviceptr reject, CUdeviceptr resolved,
                   unsigned reject_pass, CUdeviceptr seg_ranges,
                   unsigned num_seg_ranges)
{
   struct cp_device *screen = cp->screen;
   struct cp_shader_binary *fs = cp->fs_shader;

   if (!fs || !fs->kernel || !vs_output_buf || !cp->vs_shader ||
       !screen->kernels.fs_interpolate || !screen->kernels.fs_writeback) {
      if (cp_debug->debug_draw)
         fprintf(stderr, "  no fragment stage: fs=%p kernel=%p vs_out=%p vs=%p\n",
                 (void *)fs, fs ? (void *)fs->kernel : NULL,
                 (void *)(uintptr_t)vs_output_buf, (void *)cp->vs_shader);
      return;
   }

   unsigned num_fs_inputs = MIN2(fs->nir_num_inputs, CP_MAX_FS_INPUTS);
   unsigned num_vs_outputs = cp->vs_shader->nir_num_outputs
      ? cp->vs_shader->nir_num_outputs : 2;

   /* Fragment shader I/O buffers are indexed by thread, and the shader is
    * launched in whole blocks, so round up to keep the tail threads in
    * bounds. */
   /* Quads are emitted per triangle, so a block straddling a seam produces
    * more than one. Boundary blocks are a minority; twice the pixel count is
    * ample and anything past it is dropped rather than scribbling. */
   unsigned max_pixels = ALIGN_POT(w * h, 256) * 2;
   unsigned fs_in_stride = MAX2(num_fs_inputs, 1u) * 16;
   unsigned fs_out_stride = MAX2(fs->nir_num_outputs, 1u) * 16;


   /* Written by one kernel and read by the next; the host never sees them. */
   CUdeviceptr pixel_list = cp_scratch_alloc_device(cp, max_pixels * 4);
   CUdeviceptr counter = cp_scratch_alloc_device(cp, 4);
   CUdeviceptr fs_in = cp_scratch_alloc_device(cp, (size_t)max_pixels * fs_in_stride);
   CUdeviceptr fs_out = cp_scratch_alloc_device(cp, (size_t)max_pixels * fs_out_stride);
   CUdeviceptr coverage = cp_scratch_alloc_device(cp, max_pixels);
   CUdeviceptr frag_coord = cp_scratch_alloc_device(cp, (size_t)max_pixels * 16);
   /*
    * One byte per shaded pixel, set by `discard` in the fragment shader —
    * and only for a shader that has one. The writeback already reads the mask
    * conditionally, so a shader that cannot discard needs neither the
    * allocation nor the clear, which at this size is 1.8 MB a draw.
    */
   CUdeviceptr discard_mask = fs->uses_discard
      ? cp_scratch_alloc_device(cp, max_pixels) : 0;
   /* gl_FrontFacing, on the same bargain as the discard mask above. */
   CUdeviceptr front_face = fs->reads_front_face
      ? cp_scratch_alloc_device(cp, max_pixels) : 0;

   if (!pixel_list || !counter || !fs_in || !fs_out || !coverage || !frag_coord ||
       (fs->uses_discard && !discard_mask) ||
       (fs->reads_front_face && !front_face))
      return;

   /*
    * These two are read where they were not written, so they have to start
    * clean rather than inherit whatever the arena last held. A fresh
    * cuMemAllocManaged happens to be zeroed, which hid the dependency for as
    * long as every pass of a multi-pass draw got its own allocation — and
    * turned into double-blended frames the moment the passes started reusing
    * one, which they must, or a 256 layer draw asks for tens of gigabytes.
    */
   cuMemsetD32Async(counter, 0, 1, cp->stream);
   if (discard_mask)
      cuMemsetD8Async(discard_mask, 0, max_pixels, cp->stream);

   struct cp_fs_interp_args interp = {
      .visbuf = visbuf,
      .positions = positions,
      .vs_out = vs_output_buf,
      .pixel_list = pixel_list,
      .counter = counter,
      .fs_in = fs_in,
      .frag_coord = frag_coord,
      .coverage = coverage,
      .front_face = front_face,
      .front_ccw = cp->rasterizer.front_ccw,
      .width = w, .height = h,
      .vs_out_stride = num_vs_outputs * 16,
      .fs_in_stride = fs_in_stride,
      .num_fs_inputs = num_fs_inputs,
      .max_pixels = max_pixels,
      .quad_width = (w + 1) / 2,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
      /* TEMPORARY: see cp_abuf_dbg above. */
      .dbg_blk_offsets = cp_abuf_dbg.blk_offsets,
      .dbg_blk_counts = cp_abuf_dbg.blk_counts,
      .dbg_quad_prim = cp_abuf_dbg.quad_prim,
      .dbg_peel_mask = cp_abuf_dbg.peel_mask,
      .dbg_counters = cp_abuf_dbg.counters,
      .seg_ranges = seg_ranges,
      .num_seg_ranges = num_seg_ranges,
   };

   cp_fs_interp_setup(cp, info, fs, num_fs_inputs, num_vs_outputs, &interp);

   /* See the same block in cp_abuf_shade(). A blended batch reaches this path
    * only when its A-buffer merge was refused and the peel loop renders it
    * instead, which is rare and still has to be right. */
   CUdeviceptr batch_rows = 0;
   if (cp->fs_batch.ndraws > 1 && fs->reads_const_bufs) {
      if (!cp->fs_batch.slices) {
         fprintf(stderr, "cudapipe: a batch of %u has no slice table; refusing "
                 "to shade it\n", cp->fs_batch.ndraws);
         return;
      }
      batch_rows = cp_scratch_alloc_device(cp, (size_t)max_pixels * 4);
      if (!batch_rows)
         return;
      interp.out_batch_rows = batch_rows;
      interp.draw_slices = cp->fs_batch.slices;
      interp.num_draw_slices = cp->fs_batch.ndraws;
      interp.prim_shift = cp->fs_batch.prim_shift;
   }

   /*
    * TEMPORARY (CUDAPIPE_ABUFFER): where this pass's shaded colours are to be
    * deposited for the comparison, one slot per (pixel, primitive). Allocated
    * from the same arena the pass rewinds, so it costs nothing on a frame that
    * is not being verified.
    */
   CUdeviceptr dbg_slot = 0;
   if (cp_abuf_dbg.colors) {
      dbg_slot = cp_scratch_alloc_device(cp, (size_t)max_pixels * 4);
      if (dbg_slot) {
         interp.dbg_slot = dbg_slot;
         interp.abuf_frags = cp_abuf_dbg.frags;
         interp.abuf_offsets = cp_abuf_dbg.offsets;
         interp.abuf_counts = cp_abuf_dbg.counts;
      }
   }

   void *interp_params[] = { &interp };
   {
      /* Scoped rather than pushed and popped, because the launch check below
       * returns out of the middle of it. */
      CP_NVTX_SCOPE("interp");
      /* One thread per 2x2 quad, and the shader then runs four threads per
       * quad so it can difference across one. */
      unsigned num_quads = ((w + 1) / 2) * ((h + 1) / 2);
      CUresult interp_err = cuLaunchKernel(screen->kernels.fs_interpolate,
                                           (num_quads + 255) / 256, 1, 1, 256, 1, 1,
                                           0, cp->stream, interp_params, NULL);
      if (interp_err != CUDA_SUCCESS) {
         fprintf(stderr, "cudapipe: fs_interpolate launch failed (%d)\n", interp_err);
         return;
      }
   }
   cp_stage_end(cp, CP_STAGE_INTERPOLATE);

   /* Launch FS and writeback over max_pixels — each kernel reads the actual
    * pixel count from the counter (device-visible managed memory) and exits
    * early for threads beyond it. This avoids a sync just to read the count. */
   unsigned num_pixels = max_pixels;

   if (!cp_fs_launch_shader(cp, fs, counter, fs_in, fs_in_stride, fs_out,
                            frag_coord, discard_mask, front_face, coverage,
                            0, num_pixels, 0, batch_rows))
      return;
   cp_stage_end(cp, CP_STAGE_FRAGMENT);

   /* TEMPORARY (CUDAPIPE_ABUFFER): scatter this pass's colours into the
    * A-buffer slots the interpolation just resolved. */
   if (dbg_slot && screen->kernels.abuf_scatter_colors) {
      void *p[] = { &fs_out, &fs_out_stride, &dbg_slot, &counter, &num_pixels,
                    &cp_abuf_dbg.capacity, &cp_abuf_dbg.colors,
                    &cp_abuf_dbg.writes, &cp_abuf_dbg.counters };
      CP_LAUNCH(screen->kernels.abuf_scatter_colors,
                     (num_pixels + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, p, NULL);
   }

   struct cp_fs_writeback_args wb = {
      .pixel_list = pixel_list,
      .fs_out = fs_out,
      .color_out = (uint64_t)(uintptr_t)color_data,
      .visbuf = visbuf,
      .depthbuf = cp->depthbuf,
      .pixel_counter = counter,
      .discard_mask = discard_mask,
      .coverage = coverage,
      .reject = reject,
      .resolved = resolved,
      .reject_layers = CP_DISCARD_LAYERS,
      .reject_pass = reject_pass,
      .depth_write = cp->depth_stencil.depth_writemask,
      .depth_key_invert = cp->depth_stencil.depth_enabled &&
         (cp->depth_stencil.depth_func == CP_FUNC_GREATER ||
          cp->depth_stencil.depth_func == CP_FUNC_GEQUAL),
      .width = w,
      .fs_out_stride = fs_out_stride,
      .num_pixels = num_pixels,
      .color_encoding = (uint32_t)MAX2(cp->fb.color_encoding, 0),
      .blend = cp_blend_desc_for(cp),
      .num_samples = MAX2(cp->fb_samples, 1u),
      .height = h,
      .sample_stride = cp->fb.color_sample_stride,
   };

   /*
    * An attachmentless pass has no colour to write back. It reaches this
    * function at all because its fragment shader has side effects — storage
    * image and SSBO writes, which the shader launch above has already made —
    * and cp_fs_writeback does nothing else that such a pass asks for: `reject`
    * and `resolved` belong to the discard retry, which needs a colour target
    * to be enabled at all, and depth writeback never ran for a colourless
    * draw because the whole fragment stage used to be skipped for one. So the
    * launch is skipped rather than given a null `color_out` to dereference.
    */
   if (color_data) {
      void *wb_params[] = { &wb };
      cp_nvtx_push("writeback");
      /* The kernel strides, so the grid is capped: num_pixels is the
       * framebuffer's worst case and the launch was spending more time
       * scheduling idle blocks than writing pixels on small draws. */
      CP_LAUNCH(screen->kernels.fs_writeback,
                     MIN2((num_pixels + 255) / 256, 2048u), 1, 1, 256, 1, 1,
                     0, cp->stream, wb_params, NULL);
      cp_nvtx_pop();   /* writeback */
   }
   cp_stage_end(cp, CP_STAGE_WRITEBACK);

   /* How much of the shading launch does any work. Syncs, so debug only, and
    * read it on a deterministic sample. */
   if (cp_debug->debug_work) {
      uint32_t shaded = 0;
      cuStreamSynchronize(cp->stream);
      cuMemcpyDtoH(&shaded, counter, sizeof(shaded));
      fprintf(stderr, "work shaded=%u fs_threads=%u interp_threads=%u fb=%u\n",
              MIN2(shaded, max_pixels), num_pixels,
              ((w + 1) / 2) * ((h + 1) / 2), w * h);
   }

   if (cp_debug->debug_discard) {
      /* Both live in device-only memory now, so they have to be fetched
       * rather than read through the pointer. This path already synchronises,
       * which is what makes that affordable. */
      cuCtxSynchronize();
      uint32_t covered = 0;
      cuMemcpyDtoH(&covered, counter, sizeof(covered));
      covered = MIN2(covered, max_pixels);

      unsigned nd = 0;
      unsigned char *dm = covered ? malloc(covered) : NULL;
      if (dm) {
         cuMemcpyDtoH(dm, discard_mask, covered);
         for (unsigned k = 0; k < covered; k++)
            nd += dm[k] ? 1u : 0u;
         free(dm);
      }
      fprintf(stderr, "  pass %u: covered %u, discarded %u\n",
              reject_pass, covered, nd);
   }

   if (cp_debug->debug_draw)
      fprintf(stderr, "  shaded %u pixels (%u fs inputs, %u tris) "
              "blend=%u src=%u dst=%u mask=0x%x\n",
              num_pixels, num_fs_inputs, num_triangles,
              wb.blend.enable, wb.blend.rgb_src_factor,
              wb.blend.rgb_dst_factor, wb.blend.colormask);

   if (cp_debug->debug_fs) {
      /*
       * Everything printed below is device-only, so it is fetched whole
       * first. Wasteful, and correct for a path that already synchronises
       * and prints eight lines.
       */
      cuCtxSynchronize();
      size_t vs_out_bytes = (size_t)num_triangles * 3 * num_vs_outputs * 16;
      float *vs_out = malloc(vs_out_bytes);
      uint32_t *plist_buf = malloc((size_t)num_pixels * 4);
      float *fin_buf = malloc((size_t)num_pixels * fs_in_stride);
      float *fout_buf = malloc((size_t)num_pixels * fs_out_stride);
      if (!vs_out || !plist_buf || !fin_buf || !fout_buf) {
         free(vs_out); free(plist_buf); free(fin_buf); free(fout_buf);
         fprintf(stderr, "  (CUDAPIPE_DEBUG_FS: out of memory)\n");
         return;
      }
      cuMemcpyDtoH(vs_out, vs_output_buf, vs_out_bytes);
      cuMemcpyDtoH(plist_buf, pixel_list, (size_t)num_pixels * 4);
      cuMemcpyDtoH(fin_buf, fs_in, (size_t)num_pixels * fs_in_stride);
      cuMemcpyDtoH(fout_buf, fs_out, (size_t)num_pixels * fs_out_stride);

      unsigned vstep = cp_debug->debug_fs_vstep;   /* registry clamps to >= 1 */
      for (unsigned v = 0; v < num_triangles * 3 && v < 6 * vstep; v += vstep) {
         fprintf(stderr, "  vtx%u:", v);
         for (unsigned s = 0; s < num_vs_outputs; s++)
            fprintf(stderr, " slot%u=[%.3f %.3f %.3f %.3f]", s,
                    vs_out[(v * num_vs_outputs + s) * 4 + 0],
                    vs_out[(v * num_vs_outputs + s) * 4 + 1],
                    vs_out[(v * num_vs_outputs + s) * 4 + 2],
                    vs_out[(v * num_vs_outputs + s) * 4 + 3]);
         fprintf(stderr, "\n");
      }
      for (unsigned i = 0; i < num_fs_inputs; i++)
         fprintf(stderr, "  fs_in[%u] <- vs slot %d (loc %u)\n", i,
                 interp.input_vs_slot[i], fs->in_location[i]);
      const uint32_t *plist = plist_buf;
      const float *fin = fin_buf;
      const float *fout = fout_buf;
      int want_row = cp_debug->debug_fs_row;
      unsigned shown = 0;
      for (unsigned i = 0; i < num_pixels && shown < (want_row >= 0 ? 64u : 8u); i++) {
         unsigned px = plist[i];
         if (want_row >= 0 && (int)(px / w) != want_row)
            continue;
         shown++;
         const uint32_t *cb = (const uint32_t *)color_data;
         fprintf(stderr, "  px(%u,%u) in=[%.9f %.9f] out=[%.3f %.3f %.3f %.3f] "
                 "fb=0x%08x\n",
                 px % w, px / w,
                 fin[i * (fs_in_stride / 4) + 0], fin[i * (fs_in_stride / 4) + 1],
                 fout[i * (fs_out_stride / 4) + 0], fout[i * (fs_out_stride / 4) + 1],
                 fout[i * (fs_out_stride / 4) + 2], fout[i * (fs_out_stride / 4) + 3],
                 cb ? cb[px] : 0u);
      }
      free(vs_out); free(plist_buf); free(fin_buf); free(fout_buf);
   }

}


bool
cp_abuf_shade(struct cp_context *cp, const struct cp_draw_call *info,
              struct cp_abuf *ab, CUdeviceptr positions,
              CUdeviceptr vs_output_buf, unsigned w, unsigned h,
              float vp_scale_x, float vp_scale_y,
              float vp_trans_x, float vp_trans_y,
              uint32_t num_quads, uint32_t num_covered, bool record_colors,
              void *color_data, bool composite, float *t_interp,
              float *t_shade, float *t_composite,
              struct cp_abuf_seg_shade *seg)
{
   struct cp_device *screen = cp->screen;
   struct cp_shader_binary *fs = cp->fs_shader;

   *t_interp = 0.0f;
   *t_shade = 0.0f;
   *t_composite = 0.0f;
   if (!fs || !fs->kernel || !vs_output_buf || !cp->vs_shader ||
       !screen->kernels.abuf_interpolate || !num_quads)
      return false;
   if (composite && (!screen->kernels.abuf_composite || !ab->shade_slot ||
                     !ab->clist || !color_data))
      return false;

   unsigned num_fs_inputs = MIN2(fs->nir_num_inputs, CP_MAX_FS_INPUTS);
   unsigned num_vs_outputs = cp->vs_shader->nir_num_outputs
      ? cp->vs_shader->nir_num_outputs : 2;
   unsigned fs_in_stride = MAX2(num_fs_inputs, 1u) * 16;
   unsigned fs_out_stride = MAX2(fs->nir_num_outputs, 1u) * 16;

   /* Four slots a quad — no atomic, because the merge already said how many
    * quads there are and where each one is. Rounded up because the shader is
    * launched in whole blocks and the tail threads index these buffers. */
   size_t want_slots = (size_t)num_quads * 4;
   if (want_slots > CP_ABUF_MAX_SHADE_SLOTS) {
      fprintf(stderr, "abuffer: %zu shading slots is over the %u cap; this "
              "draw's quad stream is not shaded\n", want_slots,
              CP_ABUF_MAX_SHADE_SLOTS);
      return false;
   }
   unsigned num_slots = ALIGN_POT((unsigned)want_slots, 256);

   CUdeviceptr pixel_list = cp_scratch_alloc_device(cp, (size_t)num_slots * 4);
   CUdeviceptr counter = cp_scratch_alloc_device(cp, 4);
   CUdeviceptr fs_in = cp_scratch_alloc_device(cp,
                                               (size_t)num_slots * fs_in_stride);
   CUdeviceptr fs_out = cp_scratch_alloc_device(cp,
                                                (size_t)num_slots * fs_out_stride);
   CUdeviceptr coverage = cp_scratch_alloc_device(cp, num_slots);
   CUdeviceptr frag_coord = cp_scratch_alloc_device(cp, (size_t)num_slots * 16);
   CUdeviceptr discard_mask = fs->uses_discard
      ? cp_scratch_alloc_device(cp, num_slots) : 0;
   CUdeviceptr front_face = fs->reads_front_face
      ? cp_scratch_alloc_device(cp, num_slots) : 0;
   CUdeviceptr dbg_slot = record_colors
      ? cp_scratch_alloc_device(cp, (size_t)num_slots * 4) : 0;

   if (!pixel_list || !counter || !fs_in || !fs_out || !coverage ||
       !frag_coord || (fs->uses_discard && !discard_mask) ||
       (fs->reads_front_face && !front_face) ||
       (record_colors && !dbg_slot)) {
      fprintf(stderr, "abuffer: shading buffers for %u slots refused; this "
              "draw's quad stream is not shaded\n", num_slots);
      return false;
   }

   /* The shader reads its extent from the device the way it does on the peel
    * path, where an atomic put it there. */
   if (seg && seg->num_quads_dev) {
      struct cp_abuf_shade_count_args count_args = {
         .count = seg->num_quads_dev,
         .slots = counter,
      };
      void *count_params[] = { &count_args };
      CP_LAUNCH(screen->kernels.abuf_prepare_shade_count,
                1, 1, 1, 1, 1, 1, 0, cp->stream, count_params, NULL);
   } else {
      cuMemsetD32Async(counter, (unsigned)want_slots, 1, cp->stream);
   }
   if (discard_mask)
      cuMemsetD8Async(discard_mask, 0, num_slots, cp->stream);

   struct cp_fs_interp_args interp = {
      .positions = positions,
      .vs_out = vs_output_buf,
      .pixel_list = pixel_list,
      .counter = counter,
      .fs_in = fs_in,
      .frag_coord = frag_coord,
      .coverage = coverage,
      .front_face = front_face,
      .front_ccw = cp->rasterizer.front_ccw,
      .width = w, .height = h,
      .vs_out_stride = num_vs_outputs * 16,
      .fs_in_stride = fs_in_stride,
      .num_fs_inputs = num_fs_inputs,
      .max_pixels = num_slots,
      .quad_width = (w + 1) / 2,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
      .abuf_quad_prim = ab->quad_prim,
      .abuf_quad_mask = ab->quad_mask,
      .abuf_quad_block = ab->quad_block,
      .abuf_frags = ab->frags,
      .abuf_offsets = ab->offsets,
      .abuf_counts = ab->counts,
      .dbg_slot = dbg_slot,
      .abuf_num_quads = num_quads,
      .num_quads_dev = ab->bsum3,
   };
   if (seg) {
      interp.quad_list = seg->quad_list;
      interp.quad_list_base = seg->quad_list_base;
      interp.quad_list_base_dev = seg->quad_list_base_dev;
      interp.abuf_prim_base = seg->prim_base;
      interp.seg_ranges = seg->ranges;
      interp.num_seg_ranges = seg->num_ranges;
      if (seg->num_quads_dev)
         interp.num_quads_dev = seg->num_quads_dev;
   }
   cp_fs_interp_setup(cp, info, fs, num_fs_inputs, num_vs_outputs, &interp);

   /* Which merged draw's fragment bindings each shaded slot is to use. Only a
    * batch has more than one answer, and only a shader that reads a constant
    * buffer can tell. */
   CUdeviceptr batch_rows = 0;
   if (cp->fs_batch.ndraws > 1 && fs->reads_const_bufs) {
      /* A batch the interpolator cannot resolve would shade every draw of it
       * with the first one's material. Say so rather than render it. A
       * merged group carries its slice tables per range instead, where a
       * member that is not a batch legitimately has none. */
      if (!cp->fs_batch.slices && !(seg && seg->ranges)) {
         fprintf(stderr, "abuffer: a batch of %u has no slice table; refusing "
                 "to shade it\n", cp->fs_batch.ndraws);
         return false;
      }
      batch_rows = cp_scratch_alloc_device(cp, (size_t)num_slots * 4);
      if (!batch_rows)
         return false;
      interp.out_batch_rows = batch_rows;
      interp.draw_slices = cp->fs_batch.slices;
      interp.num_draw_slices = cp->fs_batch.ndraws;
      interp.prim_shift = cp->fs_batch.prim_shift;
   }

   CUdeviceptr interp_dev = cp_upload(cp, &interp, sizeof(interp));
   if (!interp_dev)
      return false;

   if (cp_debug->no_fused_abuf_interp) {
      void *interp_params[] = { &interp };
      CUfunction kernel = seg && seg->ranges
         ? screen->kernels.abuf_interpolate_ranges
         : screen->kernels.abuf_interpolate;
      CP_LAUNCH(kernel, (num_quads + 255) / 256, 1, 1, 256, 1, 1, 0,
                cp->stream, interp_params, NULL);
   }

   if (!cp_fs_launch_shader(cp, fs, counter, fs_in, fs_in_stride, fs_out,
                            frag_coord, discard_mask, front_face, coverage,
                            cp_debug->no_fused_abuf_interp ? 0 : interp_dev,
                            num_slots,
                            cp_abuf.timing ? ab->ev[13] : 0, batch_rows))
      return false;
   cp_abuf_mark(ab->ev[14], cp->stream);

   /* A segment's colours are composited once for the whole episode, through
    * the per-quad segment map — hand the arrays back instead. */
   if (seg) {
      seg->fs_out = fs_out;
      seg->coverage = coverage;
      seg->discard = discard_mask;
      seg->fs_out_stride = fs_out_stride;
      seg->num_slots = num_slots;
      return true;
   }

   if (record_colors && screen->kernels.abuf_scatter_colors) {
      void *p[] = { &fs_out, &fs_out_stride, &dbg_slot, &counter, &num_slots,
                    &ab->capacity, &ab->colors_abuf, &ab->writes_abuf,
                    &ab->dbg };
      CP_LAUNCH(screen->kernels.abuf_scatter_colors,
                     (num_slots + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, p, NULL);
   }

   /*
    * --- step 7: composite ---
    *
    * One thread per covered pixel, over the worklist built beside the sort.
    * The grid is sized to the framebuffer because the length of that worklist
    * lives on the device — the kernel reads it and the tail threads return,
    * which is what the peel path's own kernels do with their pixel counter and
    * is cheaper than draining to ask.
    */
   if (composite) {
      struct cp_abuf_composite_args ca = {
         .offsets = ab->offsets,
         .counts = ab->counts,
         .shade_slot = ab->shade_slot,
         .fs_out = fs_out,
         .coverage = coverage,
         .discard_mask = discard_mask,
         .color_out = (uint64_t)(uintptr_t)color_data,
         .list = ab->clist,
         .list_count = ab->clist_count,
         .fs_out_stride = fs_out_stride,
         .num_slots = num_slots,
         .capacity = ab->capacity,
         .color_encoding = (uint32_t)MAX2(
            cp->fb.color_encoding, 0),
         .max_layers = cp_abuf.max_layers,
         .blend = cp_blend_desc_for(cp),
      };
      void *p[] = { &ca };
      /* The worklist's own length, read in the drain that decided this path;
       * the kernel still reads it from the device, so a grid sized to it is a
       * statement about how much of the machine to use rather than a bound
       * anything depends on. */
      unsigned nwork = num_covered ? num_covered : w * h;
      cp_nvtx_push("composite");
      cp_abuf_mark(ab->ev[15], cp->stream);
      CUresult ce = cuLaunchKernel(screen->kernels.abuf_composite,
                                   (nwork + 255) / 256, 1, 1, 256, 1, 1,
                                   0, cp->stream, p, NULL);
      cp_abuf_mark(ab->ev[16], cp->stream);
      cp_nvtx_pop();   /* composite */
      if (ce != CUDA_SUCCESS) {
         fprintf(stderr, "abuffer: cp_abuf_composite launch failed (%d)\n", ce);
         return false;
      }
   }

   /*
    * Reading the events means draining, which is a real cost on a path whose
    * point is not to. Only done when the breakdown is being printed.
    */
   if (cp_abuf.timing) {
      cuStreamSynchronize(cp->stream);
      cuEventElapsedTime(t_interp, ab->ev[11], ab->ev[12]);
      cuEventElapsedTime(t_shade, ab->ev[13], ab->ev[14]);
      if (composite)
         cuEventElapsedTime(t_composite, ab->ev[15], ab->ev[16]);
   }
   return true;
}

/*
 * Compare the two paths' shaded colours, slot by slot.
 *
 * A slot is one (pixel, primitive), so this is complete rather than sampled:
 * every fragment the peel loop shaded has exactly one counterpart here, and
 * the varyings depend only on the triangle and the pixel — not on which pass
 * shaded them — so the claim is bit-identity and not proximity.
 *
 * Two populations are deliberately not compared. Helper lanes: shaded by both
 * paths, dropped by both, and named by no slot. And fragments lying at or past
 * the peel loop's cap, which the loop never reached — those are counted and
 * reported rather than quietly dropped.
 */
static void
cp_abuf_verify_colors(struct cp_abuf *ab, unsigned w, unsigned h,
                      uint32_t total, unsigned passes_run, uint32_t slot_bad)
{
   size_t n = (size_t)w * h;
   uint64_t compared = 0, matched = 0, mismatched = 0;
   uint64_t excluded = 0, peel_past_cap = 0;
   uint64_t abuf_missing = 0, peel_missing = 0;
   uint64_t abuf_dup = 0, peel_dup = 0;
   unsigned shown = 0;
   const unsigned show_max = 10;

   for (size_t p = 0; p < n; p++) {
      uint32_t c = ab->h_counts[p];
      uint32_t base = ab->h_offsets[p];
      for (uint32_t k = 0; k < c; k++) {
         uint32_t slot = base + k;
         uint32_t prim = ab->h_frags[slot];
         uint32_t aw = ab->h_writes_abuf[slot], pw = ab->h_writes_peel[slot];
         if (aw > 1)
            abuf_dup++;
         if (pw > 1)
            peel_dup++;

         /* A fragment at rank k is composited on peel pass k, so anything at
          * or past the number of passes the loop ran has no peel-side
          * counterpart at all. */
         if (k >= passes_run) {
            excluded++;
            if (pw)
               peel_past_cap++;
            continue;
         }

         if (!aw) {
            abuf_missing++;
            continue;
         }
         if (!pw) {
            peel_missing++;
            continue;
         }

         compared++;
         const uint32_t *a = (const uint32_t *)(ab->h_colors_abuf + slot * 4);
         const uint32_t *b = (const uint32_t *)(ab->h_colors_peel + slot * 4);
         if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]) {
            matched++;
            continue;
         }

         mismatched++;
         if (shown++ < show_max) {
            const float *fa = ab->h_colors_abuf + slot * 4;
            const float *fb = ab->h_colors_peel + slot * 4;
            fprintf(stderr, "abuffer colours: MISMATCH pixel (%zu,%zu) prim=%u "
                    "layer=%u\n    abuf [%.9g %.9g %.9g %.9g]\n"
                    "    peel [%.9g %.9g %.9g %.9g]\n",
                    p % w, p / w, prim, k,
                    fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], fb[2], fb[3]);
         }
      }
   }

   bool ok = !mismatched && !abuf_missing && !peel_missing && !abuf_dup &&
             !peel_dup && !peel_past_cap && !slot_bad;
   fprintf(stderr,
           "abuffer colours: %u slots; compared %" PRIu64 ", matched "
           "%" PRIu64 ", mismatched %" PRIu64 "\n"
           "  excluded by the %u-pass cap: %" PRIu64 " (%.1f%%)%s\n"
           "  unwritten: abuf %" PRIu64 ", peel %" PRIu64 "; claimed twice: "
           "abuf %" PRIu64 ", peel %" PRIu64 "; out-of-range slots %u\n"
           "  VERDICT: %s\n",
           total, compared, matched, mismatched,
           passes_run, excluded, total ? 100.0 * (double)excluded / total : 0.0,
           peel_past_cap ? "  (and the peel loop wrote past it — see above)"
                         : "",
           abuf_missing, peel_missing, abuf_dup, peel_dup, slot_bad,
           ok ? "every fragment the peel path shaded came out bit-identical"
              : "MISMATCH — see above");
}

/*
 * Run one draw, or one batch of them, through the whole pipeline.
 *
 * `batch_draws` is how many merged draws this launch stands for and
 * `vs_ubo_table` their vertex-stage uniform bindings, one row of
 * CP_ARG_UBO_STRIDE per draw. A batch of one with a null table is the
 * unbatched path, and every kernel it launches takes the arguments it took
 * before this existed — the batch fields are zero, and the code below reduces
 * to what it was.
 */
void
cp_draw_execute(struct cp_context *cp, const struct cp_draw_call *info,
                unsigned drawid_offset,
                const struct cp_draw_range *draws,
                unsigned num_draws, unsigned batch_draws,
                const uint64_t *vs_ubo_table, const uint64_t *fs_ubo_table,
                const uint32_t *draw_ids, const uint32_t *instance_counts,
                const uint64_t *vb_table,
                const struct cp_rect *scissors)
{
   struct cp_device *screen = cp->screen;
   const struct cp_fb_desc *fb = &cp->fb;

   /*
    * The fragment stage's per-draw uniform bindings.
    *
    * Set here rather than where the slice table is built, because a *deferred*
    * draw needs them even when it is a batch of one: it runs after the next
    * draw's bindings have been bound over the live ones, so reading
    * cp->fs_ubos at launch time shades it with the wrong material. That was
    * worth 57% of the pixels of a frame, on batches of one, with no merging
    * involved at all — BATCHING.md's "the hazard is deferral, not merging",
    * arrived at a second time.
    *
    * Rewritten unconditionally so nothing about the last draw survives into
    * this one. `slices` stays null until the batch actually has more than one
    * draw to tell apart; with one draw every fragment takes row zero.
    */
   cp->fs_batch.ubos = fs_ubo_table;
   cp->fs_batch.ndraws = fs_ubo_table ? batch_draws : 0;
   cp->fs_batch.slices = 0;
   cp->fs_batch.prim_shift = 0;

   if (!screen->kernels.initialized || !screen->kernels.rasterize_triangles)
      return;
   /*
    * A draw with nowhere to put a colour and nowhere to put a depth normally
    * has nothing to do. The exception is a fragment shader that writes memory:
    * Vulkan allows a subpass with no attachments at all whose fragment shader
    * exists purely for its storage-image and SSBO writes, and the `oit` sample
    * builds its per-pixel fragment lists in exactly such a pass. Those writes
    * are the draw's output, so the draw is not skippable — see `fs_side_effects`
    * further down, which carries the same condition through the rest of the
    * function.
    */
   if (!fb->nr_cbufs && !fb->has_zs &&
       !(cp->fs_shader && cp->fs_shader->writes_memory))
      do { if (cp_debug->debug_draw)
            fprintf(stderr, "  skipped: no colour or depth attachment\n");
         return; } while (0);
   if (num_draws == 0 || draws[0].count == 0)
      return;

   cuCtxSetCurrent(screen->cuda_ctx);

   bool indexed = info->index_size > 0;

   /* Every instance replays the same primitives, so it multiplies the count. */
   unsigned instance_count = MAX2(info->instance_count, 1u);

   unsigned total_triangles = 0;
   for (unsigned d = 0; d < num_draws; d++)
      total_triangles += cp_triangles_for_draw(info->mode, draws[d].count);
   total_triangles *= instance_count;

   /* What one draw of the batch contributes, which is what the instance
    * arithmetic below divides by. Only the unbatched launch-wide
    * verts_per_instance reads it — a batch resolves instancing out of its
    * slice table instead. */
   unsigned tris_per_draw = total_triangles;

   /*
    * A batch concatenates its merged draws, and they need not be the same
    * size: `draws` holds one range per merged draw. Instances multiply each
    * draw's contribution — instance_count is in the batch key, so it is one
    * number for the whole batch. That is the whole of what makes the grids
    * below bigger.
    */
   if (batch_draws > 1) {
      total_triangles = 0;
      for (unsigned d = 0; d < batch_draws; d++) {
         unsigned draw_instances = instance_counts
            ? MAX2(instance_counts[d], 1u) : instance_count;
         total_triangles += cp_triangles_for_draw(info->mode, draws[d].count) *
                            draw_instances;
      }
   }

   if (total_triangles == 0)
      do { if (cp_debug->debug_draw)
            fprintf(stderr, "  skipped: no triangles\n");
         return; } while (0);
   unsigned num_triangles = total_triangles;
   /* Near-plane clipping can split triangles, so the rasterizer grid is sized
    * for the post-clip worst case while the count itself lives on the GPU. */
   unsigned rast_num_triangles = total_triangles;

   /* Resolved when the framebuffer was bound; NULL for a depth-only pass. */
   void *color_data = fb->color;
   if (cp_debug->debug_draw && !color_data)
      fprintf(stderr, "  color=(nil) reason: nr_cbufs=%u\n", fb->nr_cbufs);

   unsigned w = fb->width;
   unsigned h = fb->height;
   unsigned fb_samples = MAX2(cp->fb_samples, 1u);

   /* The visibility buffer only ever holds this draw's triangles: its entries
    * are triangle indices into this draw's vertex arrays, so carrying it
    * across draws would shade one draw's pixels with another's geometry.
    * Occlusion between draws is carried by the depth buffer instead. */
   CUdeviceptr visbuf = cp->visbuf;
   if (!visbuf)
      do { if (cp_debug->debug_draw)
            fprintf(stderr, "  skipped: no visibility buffer\n");
         return; } while (0);

   /* Clear visbuf to VISBUF_EMPTY (all-ones). cuMemsetD32 fills 32-bit words
    * which is faster than a kernel launch for a bulk fill. A segment append
    * skips it: the A-buffer passes never touch the visibility buffer, and at
    * a segment per blended batch this was most of the frame's memset traffic.
    * The fallback path re-executes classically and clears its own. */
   if (!cp->pass.appending)
      cuMemsetD32Async(visbuf, 0xFFFFFFFF, (size_t)w * h * 2 * fb_samples,
                       cp->stream);

   if (!cp->depthbuf_cleared)
      cp_clear_depthbuf(cp, 1.0f);

   /* For now: read vertex positions directly from the first bound vertex buffer.
    * Assume positions are at offset 0 as float4 (x,y,z,w).
    * TODO: proper VS execution with compiled vertex shader */
   /* Viewport: pass raw scale/translate. The rasterizer uses:
    * screen = ndc * scale + translate (handles both Y-flip and non-flip) */
   float vp_scale_x = cp->viewport.scale[0];
   float vp_scale_y = cp->viewport.scale[1];
   float vp_trans_x = cp->viewport.translate[0];
   float vp_trans_y = cp->viewport.translate[1];
   /* For the rasterize kernel: vp_x/y/w/h format */
   float vp_w = fabsf(vp_scale_x) * 2.0f;
   float vp_h = fabsf(vp_scale_y) * 2.0f;
   float vp_x = vp_trans_x - fabsf(vp_scale_x);
   float vp_y = vp_trans_y - fabsf(vp_scale_y);

   /*
    * Where a fragment is allowed to land. Clipping to the view volume is what
    * confines a primitive to its viewport, and there is nothing in this driver
    * that clips x or y — the near plane is the only plane cp_clip_triangles
    * cuts — so the rasterizer is where it happens, as the bound on the pixels
    * it walks. Without it a sample drawing two viewports side by side spills
    * the geometry that runs past NDC +-1 into its neighbour.
    *
    * Half-pixel centres, matching lp_setup_set_viewports(); the scissor
    * arrives with an exclusive maximum, as llvmpipe's lp_setup_set_scissors()
    * has it.
    */
   int clip_x0 = MAX2(0, (int)(vp_x + 0.499f));
   int clip_y0 = MAX2(0, (int)(vp_y + 0.499f));
   int clip_x1 = MIN2((int)w - 1, (int)(vp_x + vp_w - 0.501f));
   int clip_y1 = MIN2((int)h - 1, (int)(vp_y + vp_h - 0.501f));
   /*
    * A batch whose primitives can be resolved to their draw — the same
    * stable-clip condition as the fragment tables — carries one rectangle
    * per draw instead of folding the scissor in here, so draws that disagree
    * on it merge. Everything else takes the recorded (or live, unbatched)
    * scissor exactly as before. A deferred draw's scissor is its snapshot,
    * not the live state.
    */
   bool rows_stable = cp->blend_enabled ||
      (cp->fs_shader && cp->fs_shader->reads_const_bufs);
   bool per_draw_rects = scissors && batch_draws > 1 && rows_stable &&
      cp->rasterizer.scissor;
   if (cp->rasterizer.scissor && !per_draw_rects) {
      const struct cp_rect *sc0 =
         scissors ? &scissors[0] : &cp->scissor;
      clip_x0 = MAX2(clip_x0, (int)sc0->minx);
      clip_y0 = MAX2(clip_y0, (int)sc0->miny);
      clip_x1 = MIN2(clip_x1, (int)sc0->maxx - 1);
      clip_y1 = MIN2(clip_y1, (int)sc0->maxy - 1);
   }
   struct cp_rasterize_args rast_args = {
      .framebuffer = visbuf,
      .color_buffer = (uint64_t)(uintptr_t)color_data,
      .width = w,
      .height = h,
      .num_triangles = num_triangles,
      .num_varyings = 0,
      .vp_x = vp_x, .vp_y = vp_y,
      .vp_w = vp_w, .vp_h = vp_h,
      .clip_x0 = clip_x0, .clip_y0 = clip_y0,
      .clip_x1 = clip_x1, .clip_y1 = clip_y1,
      .vp_near = 0.0f, .vp_far = 1.0f,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
      /*
       * Which winding to drop. The rasterizer's area is positive for one
       * winding after the viewport transform, so front_ccw picks which sign
       * front-facing means. Drawing what should have been culled is not merely
       * wasted work: a back face can win the depth test and hide the surface
       * in front of it, which is what cost Sponza the leaves whose backs face
       * the camera.
       */
      .cull_mode = info->mode == MESA_PRIM_POINTS ? 0
                 : cp_cull_mode(&cp->rasterizer),
      .front_face = cp->rasterizer.front_ccw,
      /* Points have no winding to cull and no edges to test; the square comes
       * from gl_PointSize, wherever the vertex shader put it. */
      .num_samples = fb_samples,
      .point_mode = info->mode == MESA_PRIM_POINTS,
      .psiz_slot = cp->vs_shader
         ? cp_slot_for_location(cp->vs_shader->out_location,
                                CP_MAX_IO_SLOTS, VARYING_SLOT_PSIZ)
         : -1,
      .depthbuf = cp->depthbuf,
      .depth_test = cp->depth_stencil.depth_enabled,
      .depth_func = cp->depth_stencil.depth_func,
      .depth_key_invert = cp->depth_stencil.depth_enabled &&
         (cp->depth_stencil.depth_func == CP_FUNC_GREATER ||
          cp->depth_stencil.depth_func == CP_FUNC_GEQUAL),
   };
   if (cp->pass.appending && cp->pass.opaque)
      rast_args.abuf_prim_base = cp->pass.next_prim;

   /* Names this draw on the timeline for the rest of the function, however it
    * leaves — see CP_NVTX_SCOPE. */
   CP_NVTX_SCOPEF("draw %u tris%s", num_triangles,
                  instance_count > 1 ? " inst" : "");

   /* Marks the start of the draw; the interval it opens is
    * attributed to nothing. */
   cp_stage_end(cp, -1);

   /* Reclaim last draw's scratch and size the arena for this one. A pass
    * episode owns the epoch instead: every segment's clipped stream has to
    * survive until the episode's shading has read it, so only the first
    * segment reclaims and the rest allocate beyond. */
   if (!cp->pass.appending || cp->pass.nsegs == 0)
      cp_scratch_begin(cp);

   /* A shader with no declared inputs needs no vertex buffer: it builds its
    * positions from gl_VertexIndex, which is how a fullscreen pass is drawn.
    * A batch carries its own snapshot of the bindings, so the live state —
    * which the next draw may have rebound over — is not consulted for one. */
   bool has_vs = cp->vs_shader && cp->vs_shader->kernel &&
                 (vb_table != NULL ||
                  (cp->num_vertex_buffers > 0 && cp->vb_base[0]) ||
                  cp->num_vertex_elements == 0);

   /* Already resolved by whoever built the draw call: under Gallium that is
    * cp_draw_vbo unwrapping a pipe_resource, natively it is the recorded
    * index buffer's device address. */
   const void *ib_base = indexed ? info->index_ptr : NULL;

   unsigned total_verts = num_triangles * 3;

   /*
    * Rewind the device-only scratch to the start of the draw.
    *
    * Every buffer taken from it — the vertex shader's output, the clipped
    * positions, the pixel list, the shader's inputs and outputs, the fragment
    * coordinates, the coverage and discard masks — is produced by one kernel
    * of this draw and consumed by another. None of it outlives the draw, and
    * nothing on the host ever touches it: cp_scratch_alloc_device() returns a
    * device address rather than a pointer precisely so that dereferencing one
    * does not compile.
    *
    * So the next draw may have the same memory, for the reason the pass loop
    * below already rewinds on: kernels on one stream are serialized, and this
    * draw's first kernel cannot start writing until the previous draw's last
    * kernel has finished reading. The managed arena is a different matter and
    * is deliberately not rewound here — the host writes into that one while
    * the device is still reading the draw before.
    *
    * Without this the arena carries a whole frame of draws, and it is sized to
    * the framebuffer rather than to coverage: max_pixels is twice the pixels
    * in the framebuffer, so one draw's shading buffers run to hundreds of
    * megabytes whatever the draw covers. instancing reached 5.1 GB in a frame
    * and spent 62% of it with the GPU idle, because passing
    * CP_SCRATCH_RECLAIM_BYTES makes cp_scratch_alloc() drain the device and
    * free the overflow arenas — and it was passing it several times a frame.
    *
    * Same episode exception as the managed arena above. */
   if (!cp->pass.appending || cp->pass.nsegs == 0)
      cp->dscratch.used = 0;

   /*
    * For TRIANGLE_LIST with a VS, the vertex fetch kernel indexes the IB
    * directly on the GPU — no CPU-side topology expansion needed.
    *
    * Instancing is included, and used not to be. An instanced draw replays one
    * index range once per instance, so the vertex and instance ids of every
    * assembled vertex follow from its position: vertex v is index v % vpi of
    * instance v / vpi. That is arithmetic the kernel can do per thread, and
    * building it on the host meant a 4.4 million entry table for one draw of
    * `instancing` — walked five times, written twice into managed memory the
    * device then had to fault back, and sized to the whole draw rather than to
    * anything about it. The device idled 62% of that frame.
    */
   bool skip_refs = has_vs && info->mode == MESA_PRIM_TRIANGLES &&
                    num_draws == 1;
   /* Assembled vertices per instance, which is what the kernel divides by.
    * Taken from the single draw's triangle count, so an unbatched draw sees
    * exactly the number it saw before batching existed; a batch is never
    * instanced and passes zero instead, having already resolved the vertex
    * out of its slice table. */
   unsigned verts_per_instance = tris_per_draw / instance_count * 3;
   struct cp_vertex_ref *refs = NULL;
   if (!skip_refs) {
      refs = cp_build_vertex_refs(info, draws, num_draws,
                                  instance_count, ib_base, num_triangles);
      if (!refs)
         return;
   }

   CUdeviceptr packed_positions = 0;
   CUdeviceptr vs_output_buf = 0;
   bool vs_ran = false;

   /* If no VS will run, pack positions from VB directly (passthrough).
    * When a VS is present, skip this — VS output provides positions. */
   if (!has_vs) {
      if (cp->num_vertex_buffers > 0 && cp->vb_base[0]) {
         {
            /* Base and offset were folded together when the buffer was
             * bound. */
            char *vb_start = (char *)(uintptr_t)cp->vb_base[0];
            unsigned stride = cp->vertex_stride ? cp->vertex_stride : 16;

            packed_positions =
               (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, (size_t)total_verts * 16);
            if (!packed_positions) {
               FREE(refs);
               return;
            }
            float *dst = (float *)(uintptr_t)packed_positions;

            for (unsigned v = 0; v < total_verts; v++) {
               const float *src = (const float *)(vb_start +
                                                  (size_t)refs[v].vertex * stride);
               memcpy(dst + v * 4, src, 16);
            }
            rast_args.positions = packed_positions;
         }
      }
      if (rast_args.positions == 0) {
         FREE(refs);
         return;
      }
   }
   cp_stage_end(cp, CP_STAGE_ASSEMBLE);

   /* If we have a compiled VS, run it to transform vertices.
    * The VS kernel reads from VB (args[2]) and writes positions+varyings (args[4]).
    * The output replaces packed_positions for the rasterizer. */
   /* VS execution */
   if (cp->vs_shader && cp->vs_shader->kernel) {
      CP_NVTX_SCOPE("vertex");
      void *vb_data2 = (cp->num_vertex_buffers > 0)
         ? (void *)(uintptr_t)cp->vb_base[0] : NULL;

      /* A vertex shader may build its positions from gl_VertexIndex alone and
       * declare no inputs at all, which is how a fullscreen pass is drawn.
       * That draw binds no vertex buffer, and skipping it loses every
       * post-processing and skybox pass. A batch's bindings are its snapshot,
       * not the live state. */
      if (vb_table != NULL || vb_data2 || cp->num_vertex_elements == 0) {
         unsigned stride;
         unsigned num_vs_outputs = cp->vs_shader->nir_num_outputs ? cp->vs_shader->nir_num_outputs : 2;
         unsigned out_stride = num_vs_outputs * 16;

         vs_output_buf = cp_scratch_alloc_device(cp, (size_t)total_verts * out_stride);

         /* Build VS input buffer on GPU: the vertex fetch kernel gathers
          * attributes in parallel, one thread per assembled vertex. */
         unsigned vs_in_stride = cp->num_vertex_elements * 16;
         CUdeviceptr vs_input_buf = vs_in_stride
            ? cp_scratch_alloc_device(cp, (size_t)total_verts * vs_in_stride)
            : 0;
         if (!vs_output_buf || (vs_in_stride && !vs_input_buf)) {
            FREE(refs);
            return;
         }

         /*
          * Vertex and instance ids: one uint32 per assembled vertex, read by
          * the shader as gl_VertexIndex and gl_InstanceIndex and by the fetch
          * kernel to resolve a divisored attribute.
          *
          * There are two ways to get them, and which applies is exactly
          * whether the host had to expand the topology. A strip, a fan, a
          * point list or several draws in one call leave the answer only in
          * `refs`, so it is uploaded. A triangle list does not: the ids follow
          * from the thread index, and the fetch kernel derives them.
          *
          * That second case used to build the arrays on the host as well, and
          * build them *twice* — once for the fetch kernel and once for the
          * shader, from the same refs, into two pairs of managed buffers. One
          * draw of `instancing` assembles 4.4 million vertices, so that was
          * four arrays of 17.7 MB a frame, each walked on the host and written
          * into memory the device then had to fault back page by page. It cost
          * a 35 MB refs table, five host passes over 4.4 million entries, and
          * enough managed traffic to trip the scratch reclaim several times a
          * frame. None of it was information: every byte is a function of v.
          */
         CUdeviceptr vfetch_vid = 0, vfetch_iid = 0;
         if (refs) {
            vfetch_vid = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, total_verts * 4);
            vfetch_iid = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, total_verts * 4);
            if (!vfetch_vid || !vfetch_iid) { FREE(refs); return; }
            uint32_t *vid_host = malloc((size_t)total_verts * 4);
            uint32_t *iid_host = malloc((size_t)total_verts * 4);
            if (!vid_host || !iid_host) { free(vid_host); free(iid_host); FREE(refs); return; }
            for (unsigned v = 0; v < total_verts; v++) {
               vid_host[v] = refs[v].vertex;
               iid_host[v] = refs[v].instance;
            }
            memcpy((void*)(uintptr_t)vfetch_vid, vid_host, (size_t)total_verts * 4);
            memcpy((void*)(uintptr_t)vfetch_iid, iid_host, (size_t)total_verts * 4);
            free(vid_host);
            free(iid_host);
         }

         /*
          * What the shader will be handed. With the ids already materialised
          * the shader reads the same arrays; otherwise the fetch kernel
          * publishes them, and only the ones the shader actually reads are
          * given anywhere to write.
          *
          * A 32-bit index buffer is already the vertex id array and can be
          * pointed at rather than copied — but only while every assembled
          * vertex reads a distinct entry of it, which instancing ends.
          */
         CUdeviceptr vid_buf = vfetch_vid, iid_buf = vfetch_iid;
         CUdeviceptr out_vid = 0, out_iid = 0;

         if (!refs) {
            /* A batch's assembled vertices span several ranges of the index
             * buffer, so the buffer is no longer the id array for them — the
             * fetch kernel has to publish one. */
            if (indexed && ib_base && info->index_size == 4 &&
                instance_count == 1 && batch_draws == 1) {
               vid_buf = (CUdeviceptr)(uintptr_t)ib_base +
                         (size_t)draws[0].start * 4;
            } else if (cp->vs_shader->reads_vertex_id) {
               out_vid = cp_scratch_alloc_device(cp, (size_t)total_verts * 4);
               if (!out_vid) { FREE(refs); return; }
               vid_buf = out_vid;
            }

            if (cp->vs_shader->reads_instance_id) {
               out_iid = cp_scratch_alloc_device(cp, (size_t)total_verts * 4);
               if (!out_iid) { FREE(refs); return; }
               iid_buf = out_iid;
            } else {
               iid_buf = 0;
            }
         }

         /*
          * The batch's slice table: where each merged draw's vertices begin in
          * the concatenated stream, and which part of the index buffer they
          * come from. This is what lets draws of different sizes merge — the
          * fetch kernel searches it per thread rather than dividing by a
          * single vertex count.
          *
          * Built only for a real batch. An unbatched draw leaves it null, and
          * every field the kernel would otherwise take from a slice keeps the
          * value it has always had, which is what makes CUDAPIPE_BATCH_MAX=1
          * bit-identical rather than merely close.
          */
         CUdeviceptr slices_dev = 0;
         CUdeviceptr batch_rows = 0;
         if (batch_draws > 1) {
            struct cp_draw_slice slices[CP_MAX_BATCH_DRAWS];
            uint32_t vbegin = 0;
            for (unsigned d = 0; d < batch_draws; d++) {
               unsigned inst = instance_counts
                  ? MAX2(instance_counts[d], 1u) : instance_count;
               unsigned dverts =
                  cp_triangles_for_draw(info->mode, draws[d].count) * 3;
               slices[d].vert_begin = vbegin;
               slices[d].index_bytes =
                  indexed ? (uint32_t)draws[d].start * info->index_size : 0;
               slices[d].first_vertex =
                  indexed ? (uint32_t)draws[d].index_bias : draws[d].start;
               /* Zero for a plain draw, so the fetch skips the division; the
                * span covers every instance either way. */
               slices[d].verts_per_instance = inst > 1 ? dverts : 0;
               vbegin += dverts * inst;
            }
            slices_dev = cp_upload(cp, slices,
                                   (size_t)batch_draws * sizeof(slices[0]));
            if (!slices_dev) { FREE(refs); return; }

            /* One row per assembled vertex, for the shader to pick its
             * uniform bindings and its draw parameters with. Only a shader
             * that reads either has any use for it. */
            if (cp->vs_shader->reads_const_bufs ||
                cp->vs_shader->reads_draw_params) {
               batch_rows = cp_scratch_alloc_device(cp, (size_t)total_verts * 4);
               if (!batch_rows) { FREE(refs); return; }
            }

            /* The fragment stage searches the same table, from the primitive
             * rather than from the vertex; see cp_fs_interp_args. */
            if (fs_ubo_table)
               cp->fs_batch.slices = slices_dev;

            /*
             * The per-draw clip rectangles, when the batch's draws may
             * disagree on the scissor: each is the batch-wide rectangle —
             * which then carries no scissor at all — intersected with that
             * draw's recorded one. The rasterizer resolves a primitive to
             * its rectangle through the slice table above.
             */
            if (per_draw_rects) {
               int32_t rects[CP_MAX_BATCH_DRAWS][4];
               for (unsigned d = 0; d < batch_draws; d++) {
                  rects[d][0] = MAX2(clip_x0, (int)scissors[d].minx);
                  rects[d][1] = MAX2(clip_y0, (int)scissors[d].miny);
                  rects[d][2] = MIN2(clip_x1, (int)scissors[d].maxx - 1);
                  rects[d][3] = MIN2(clip_y1, (int)scissors[d].maxy - 1);
               }
               CUdeviceptr rects_dev = cp_upload(cp, rects,
                  (size_t)batch_draws * sizeof(rects[0]));
               if (!rects_dev) { FREE(refs); return; }
               rast_args.rect_draw_slices = slices_dev;
               rast_args.clip_rects = rects_dev;
               rast_args.num_rect_slices = batch_draws;
               rast_args.rect_prim_shift = 0;   /* 2 once stable clip runs */
            }
         }

         /* The batch's per-draw vertex-buffer bases. Uploaded for a batch of
          * one too: a deferred draw runs after the next draw's bindings were
          * set over the live ones, so the snapshot is the only true copy. */
         CUdeviceptr elem_bases_dev = 0;
         if (vb_table) {
            elem_bases_dev = cp_upload(cp, vb_table,
               (size_t)batch_draws * CP_VB_TABLE_STRIDE * sizeof(uint64_t));
            if (!elem_bases_dev) {
               FREE(refs);
               return;
            }
         }

         struct cp_vertex_fetch_args vf_args = {
            .output = vs_input_buf,
            .num_elements = cp->num_vertex_elements,
            .num_verts = total_verts,
            .vs_in_stride = vs_in_stride,
            .index_size = refs ? 0 : info->index_size,
            .first_vertex = indexed ? (unsigned)draws[0].index_bias : draws[0].start,
            .start_instance = info->start_instance,
            .vertex_ids = vfetch_vid,
            .instance_ids = vfetch_iid,
            .verts_per_instance = (refs || batch_draws > 1) ? 0 : verts_per_instance,
            .out_vertex_ids = out_vid,
            .out_instance_ids = out_iid,
            .draw_slices = slices_dev,
            .num_draw_slices = slices_dev ? batch_draws : 0,
            .out_batch_rows = batch_rows,
            .elem_bases = elem_bases_dev,
         };

         /* Set up index buffer for direct GPU indexing (triangle list only).
          * The fetch kernel indexes from the start of what it is given, so the
          * draw's first index has to be folded into the pointer — a glTF model
          * draws every primitive out of one shared buffer this way, and
          * ignoring it draws the first primitive over and over. A batch has
          * one such offset per merged draw, so it carries them in the slice
          * table and is handed the buffer's own base. */
         if (!refs && indexed && ib_base)
            vf_args.index_buffer = (uint64_t)(uintptr_t)ib_base +
               (slices_dev ? 0 : (uint64_t)draws[0].start * info->index_size);

         for (unsigned e = 0; e < cp->num_vertex_elements && e < 16; e++) {
            const struct cp_vertex_elem *elem = &cp->velem[e];
            unsigned vb_idx = elem->vertex_buffer_index;
            if (vb_idx < cp->num_vertex_buffers && cp->vb_base[vb_idx])
               vf_args.vb_bases[vb_idx] = cp->vb_base[vb_idx];
            vf_args.elem_vb_idx[e] = vb_idx;
            vf_args.elem_src_offset[e] = elem->src_offset;
            vf_args.elem_src_stride[e] = elem->src_stride;
            vf_args.elem_attr_size[e] = elem->attr_size;
            vf_args.elem_nr_chan[e] = elem->nr_chan;
            vf_args.elem_chan_bytes[e] = elem->chan_bytes;
            vf_args.elem_conv[e] = elem->conv;
            vf_args.elem_swizzle[e] = elem->swizzle;
            vf_args.elem_fill_w[e] = elem->fill_w;
            vf_args.elem_instance_divisor[e] = elem->instance_divisor;
         }

         /* Nothing to gather when the shader declares no inputs — but it still
          * runs if the ids are wanted, since deriving those is now its job
          * too and a shader with no inputs may still read gl_VertexIndex. */
         if (vs_input_buf || out_vid || out_iid || batch_rows) {
            if (vs_input_buf)
               cuMemsetD8Async(vs_input_buf, 0, (size_t)total_verts * vs_in_stride, cp->stream);
            void *vf_params[] = { &vf_args };
            CP_LAUNCH(screen->kernels.vertex_fetch,
               (total_verts + 255) / 256, 1, 1, 256, 1, 1,
               0, cp->stream, vf_params, NULL);
         }

         stride = vs_in_stride;

         /* Dump what the GPU fetch actually gathered, which is the quickest way to
          * tell a bad attribute layout from a bad shader. Syncs, so debug only. */
         if (cp_debug->debug_vfetch && vs_input_buf) {
            /* Device-only; fetch the two vertices this prints. */
            cuCtxSynchronize();
            unsigned nfetch = MIN2(2u, total_verts);
            float *in = calloc(nfetch ? nfetch : 1, vs_in_stride);
            if (in)
               cuMemcpyDtoH(in, vs_input_buf, (size_t)nfetch * vs_in_stride);
            for (unsigned e = 0; e < cp->num_vertex_elements && e < 8; e++)
               fprintf(stderr, "  elem%u vb=%u off=%u stride=%u div=%u sz=%u\n",
                       e, cp->velem[e].vertex_buffer_index,
                       cp->velem[e].src_offset,
                       cp->velem[e].src_stride,
                       cp->velem[e].instance_divisor,
                       vf_args.elem_attr_size[e]);
            for (unsigned v = 0; in && v < nfetch; v++) {
               fprintf(stderr, "  vfetch v%u:", v);
               for (unsigned e = 0; e < cp->num_vertex_elements && e < 8; e++)
                  fprintf(stderr, " e%u=[%.3f %.3f %.3f]", e,
                          in[(v * vs_in_stride) / 4 + e * 4 + 0],
                          in[(v * vs_in_stride) / 4 + e * 4 + 1],
                          in[(v * vs_in_stride) / 4 + e * 4 + 2]);
               fprintf(stderr, "\n");
            }
            free(in);
         }

         /* vid_buf and iid_buf were decided above, alongside the fetch kernel
          * that fills them. */

         /*
          * The scalars the shader dereferences, in one block and so in one
          * upload: they are read by every thread of the launch, and three
          * separate managed allocations meant three pages to fault back from
          * the host after the host had just written them.
          */
         struct cp_vs_meta {
            uint32_t vcount;
            uint32_t stride;
         } meta = {
            .vcount = total_verts,
            .stride = stride,
         };

         CUdeviceptr meta_dev = cp_upload(cp, &meta, sizeof(meta));
         if (!meta_dev) {
            FREE(refs);
            return;
         }

         /*
          * The VS argument block, and behind it the per-draw uniform table the
          * shader indexes into — one upload, because the block has to hold the
          * table's device address and that is only known once the destination
          * is chosen. See CP_ARG_SLOT_UBO_TABLE.
          *
          * A single draw still gets a table: one row, pointed at by args[9],
          * with the mask at args[11] zero so that every thread indexes the
          * one-word row array at args[10], which holds zero. That is the same
          * pointer the shader used to read out of args[18 + i], so there is no
          * batched and unbatched form of the generated code and nothing to get
          * out of step.
          */
         const size_t vs_args_bytes = 64 * sizeof(void *);
         const size_t vs_scal_off = vs_args_bytes;   /* row 0, then the mask */
         const size_t vs_tbl_off = vs_args_bytes + 16;
         /* The per-draw parameter rows behind the uniform table — see
          * CP_ARG_DRAW_PARAM_STRIDE. Same block, same upload. */
         const size_t vs_dp_off = vs_tbl_off +
            (size_t)batch_draws * CP_ARG_UBO_STRIDE * sizeof(uint64_t);
         size_t vs_blk_bytes = vs_dp_off +
            (size_t)batch_draws * CP_ARG_DRAW_PARAM_STRIDE * sizeof(uint32_t);

         void *vs_blk = NULL;
         CUdeviceptr vs_args_dev = cp_upload_begin(cp, vs_blk_bytes, &vs_blk);
         if (!vs_args_dev) {
            FREE(refs);
            return;
         }
         memset(vs_blk, 0, vs_blk_bytes);

         void **vs_args_host = (void **)vs_blk;
         vs_args_host[0] = (void*)(uintptr_t)(meta_dev + offsetof(struct cp_vs_meta, vcount));
         vs_args_host[1] = NULL;
         vs_args_host[2] = (void*)(uintptr_t)vs_input_buf;
         vs_args_host[3] = (void*)(uintptr_t)(meta_dev + offsetof(struct cp_vs_meta, stride));
         vs_args_host[4] = (void*)(uintptr_t)vs_output_buf;
         vs_args_host[5] = (void*)(uintptr_t)vid_buf;
         vs_args_host[6] = (void*)(uintptr_t)iid_buf;
         vs_args_host[7] = (void*)(uintptr_t)(vs_args_dev + vs_dp_off);
         vs_args_host[CP_ARG_SLOT_UBO_TABLE] =
            (void*)(uintptr_t)(vs_args_dev + vs_tbl_off);
         /*
          * row = rows[tid & mask]. A batch points rows at what cp_vertex_fetch
          * wrote — one draw index per assembled vertex, resolved by the same
          * search that decided what to gather — and masks nothing off. A
          * single draw points rows at the zero word sitting in this block and
          * masks the index to zero, so every thread reads row zero without a
          * branch or a second shader variant.
          */
         vs_args_host[CP_ARG_SLOT_BATCH_ROWS] = batch_rows
            ? (void*)(uintptr_t)batch_rows
            : (void*)(uintptr_t)(vs_args_dev + vs_scal_off);
         vs_args_host[CP_ARG_SLOT_BATCH_MASK] =
            (void*)(uintptr_t)(vs_args_dev + vs_scal_off + 4);

         ((uint32_t *)((char *)vs_blk + vs_scal_off))[0] = 0;
         ((uint32_t *)((char *)vs_blk + vs_scal_off))[1] =
            batch_rows ? 0xFFFFFFFFu : 0u;

         uint64_t *vs_tbl = (uint64_t *)((char *)vs_blk + vs_tbl_off);
         for (unsigned d = 0; d < batch_draws; d++) {
            const uint64_t *row = vs_ubo_table
               ? vs_ubo_table + (size_t)d * CP_ARG_UBO_STRIDE : NULL;
            for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
               vs_tbl[d * CP_ARG_UBO_STRIDE + i] =
                  row ? row[i] : (uint64_t)(uintptr_t)cp->vs_ubos[i].buffer;
         }

         /* One row of draw parameters per merged draw, indexed by the same
          * batch row as the uniform table. gl_DrawID is the offset recorded
          * when the draw joined the batch, not an index into it; base_vertex
          * is defined to read zero for a non-indexed draw, where first_vertex
          * reads the draw's start. */
         uint32_t *vs_dp = (uint32_t *)((char *)vs_blk + vs_dp_off);
         for (unsigned d = 0; d < batch_draws; d++) {
            vs_dp[d * CP_ARG_DRAW_PARAM_STRIDE + 0] =
               indexed ? (uint32_t)draws[d].index_bias : draws[d].start;
            vs_dp[d * CP_ARG_DRAW_PARAM_STRIDE + 1] = info->start_instance;
            vs_dp[d * CP_ARG_DRAW_PARAM_STRIDE + 2] =
               draw_ids ? draw_ids[d] : drawid_offset;
            vs_dp[d * CP_ARG_DRAW_PARAM_STRIDE + 3] =
               indexed ? (uint32_t)draws[d].index_bias : 0;
         }

         /* Still written, so that the block reads the same whichever form a
          * stage takes its bindings from. */
         for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
            vs_args_host[CP_ARG_UBO_BASE + i] = cp->vs_ubos[i].buffer;

         cp_upload_end(cp, vs_args_dev, vs_blk, vs_blk_bytes);

         void *vs_arg_ptr = (void*)(uintptr_t)vs_args_dev;
         void *vs_params[] = { &vs_arg_ptr };
         /* Compiled shaders grid-stride; see the fragment launch. */
         CUresult vs_err = cuLaunchKernel(cp->vs_shader->kernel,
            MIN2((total_verts + 255) / 256, 4096u), 1, 1, 256, 1, 1,
            0, cp->stream, vs_params, NULL);

         if (vs_err == CUDA_SUCCESS) {
            vs_ran = true;
            /* The rasterizer reads positions directly from VS output */
            rast_args.positions = vs_output_buf;
            rast_args.num_varyings = num_vs_outputs - 1;

            /*
             * Cut the triangles that cross the near plane. Anything with a
             * vertex behind the eye projects to a mirrored position, so this
             * has to happen before the rasterizer divides by w. The result
             * has the same per-vertex layout, so the rasterizer and the
             * interpolator just read the clipped buffer instead.
             */
            if (screen->kernels.clip_triangles &&
                num_vs_outputs <= CP_MAX_CLIP_SLOTS) {
               unsigned max_clipped = num_triangles * CP_CLIP_MAX_OUT;
               CUdeviceptr clipped = cp_scratch_alloc_device(
                  cp, (size_t)max_clipped * 3 * out_stride);
               CUdeviceptr clip_count = cp_scratch_alloc_device(cp, 4);

               /*
                * Falling through here does not draw nothing, it draws wrong:
                * unclipped geometry crossing a depth plane divides to
                * coordinates far off screen, and if the pass writes depth
                * those triangles then occlude what is behind them. The
                * allocator says why it failed; this says what the failure
                * costs, because a silent skip looks exactly like a rasterizer
                * bug from the outside.
                */
               if (!clipped || !clip_count) {
                  static bool said;
                  if (!said) {
                     said = true;
                     fprintf(stderr, "cudapipe: no scratch for clipping %u "
                             "triangles — this draw is rasterized unclipped "
                             "and may be visibly wrong.\n", num_triangles);
                  }
               }

               if (clipped && clip_count) {
                  /*
                   * A batch of blended draws is composited in primitive
                   * order, so its primitives have to *be* in submission order
                   * — which compaction by atomicAdd does not promise. Stable
                   * mode gives every input triangle a fixed slot range and
                   * retires the ones it does not fill, so the count is the
                   * whole array and the rasterizer skips the holes on their
                   * zero area. Only for a batch: a single draw keeps the
                   * compacting path, so CUDAPIPE_BATCH_MAX=1 stays
                   * bit-identical to a build without any of this.
                   *
                   * An opaque batch whose fragment shader reads a constant
                   * buffer needs it too: the primitive index has to name the
                   * input triangle, or cp_write_batch_rows() maps fragments
                   * to the wrong draw's material. An opaque batch whose
                   * shader reads none keeps the compacting path — stable mode
                   * rasterizes the whole 4x slot array, holes and all, and
                   * multithreading paid 72% for ordering nothing consumes.
                   */
                  bool stable_clip = batch_draws > 1 &&
                     (cp->blend_enabled ||
                      (cp->fs_shader && cp->fs_shader->reads_const_bufs));

                  cuMemsetD32Async(clip_count,
                                   stable_clip ? max_clipped : 0, 1,
                                   cp->stream);

                  struct cp_clip_args clip = {
                     .vs_out = vs_output_buf,
                     .out = clipped,
                     .out_count = clip_count,
                     .num_triangles = num_triangles,
                     .num_slots = num_vs_outputs,
                     .max_triangles = max_clipped,
                     .stable = stable_clip,
                  };
                  void *clip_params[] = { &clip };
                  CUresult clip_err = cuLaunchKernel(
                     screen->kernels.clip_triangles,
                     (num_triangles + 63) / 64, 1, 1, 64, 1, 1,
                     0, cp->stream, clip_params, NULL);

                  if (clip_err == CUDA_SUCCESS) {
                     vs_output_buf = clipped;
                     rast_args.positions = clipped;
                     rast_args.tri_count = clip_count;
                     rast_num_triangles = max_clipped;
                     if (stable_clip) {
                        cp->fs_batch.prim_shift = 3;
                        rast_args.rect_prim_shift = 3;
                     }
                  } else {
                     fprintf(stderr, "cudapipe: clip launch failed (%d)\n",
                             clip_err);
                  }
               }
            }
         } else {
            fprintf(stderr, "  VS launch failed: %d\n", vs_err);
         }
      }
   }
   cp_stage_end(cp, CP_STAGE_VERTEX);

   if (cp_debug->debug_draw) {
      fprintf(stderr, "cudapipe: [samples=%u] draw %u tris (%u instances), fb=%ux%u, "
              "vp=[%.0f,%.0f,%.0f,%.0f] stride=%u scale=[%.1f,%.1f] color=%p\n",
              fb_samples, num_triangles, instance_count, w, h, vp_x, vp_y, vp_w, vp_h,
              cp->vertex_stride,
              cp->viewport.scale[0], cp->viewport.scale[1], color_data);
      for (unsigned e = 0; e < cp->num_vertex_elements && e < 4; e++)
         fprintf(stderr, "  elem[%u]: offset=%u fmt=%u vb=%u\n", e,
                 cp->velem[e].src_offset, cp->velem[e].conv,
                 cp->velem[e].vertex_buffer_index);
   }

   /* 3-stage adaptive rasterize. The queue counters are zeroed inside the pass
    * loop below, which runs for pass 0 as well — clearing them here too was
    * two host calls per draw that the first pass immediately repeated. */
   struct cp_rast_queues rast_queues = {
      .nontrivial = cp->cur_qset.nontrivial,
      .nontrivial_count = cp->cur_qset.counts,
      .huge_tiles = cp->cur_qset.huge_tiles,
      .huge_count = cp->cur_qset.counts + sizeof(uint32_t),
      .mode = CP_QUEUE_FILL,
   };

   /*
    * Alpha-tested geometry needs more than one go. Visibility resolves before
    * the shader runs, so a fragment that discards has already displaced the one
    * behind it — a leaf's transparent texel hides the leaf further back. Each
    * pass records what discarded where and repeats, letting the next fragment
    * win, until every pixel has settled or the layers run out.
    */
   bool retry = cp->fs_shader && cp->fs_shader->uses_discard &&
                cp->reject && cp->resolved && color_data;

   /*
    * A fragment shader whose output is not a colour. Vulkan allows a subpass
    * with no attachments at all, and a fragment shader that runs in it purely
    * to write a storage image or an SSBO — which is how the `oit` sample
    * builds the per-pixel linked list it sorts and blends in a second pass.
    * The fragment stage used to be skipped whenever there was nowhere to put a
    * colour, so that first pass never ran and the sample rendered its
    * background.
    */
   bool fs_side_effects = cp->fs_shader && cp->fs_shader->writes_memory;

   /*
    * Blended geometry needs every layer, not the nearest one. The visibility
    * buffer resolves a single fragment per pixel, which is what makes opaque
    * overdraw cost one shade — and exactly wrong for transparency, where
    * particlesystem's fire is tens of additive sprites deep and came out as
    * one sprite with holes punched in it.
    *
    * llvmpipe has no such problem because it never defers: it bins primitives
    * per tile and replays each tile's list in submission order, shading and
    * blending inline, so ordering falls out of the data structure. The same
    * semantics reach the same place here by peeling instead — each pass takes
    * the earliest primitive a pixel has not composited yet, blends it, and
    * steps past it. Both do one shade per fragment per pixel; llvmpipe
    * serializes them within a tile, this serializes them across passes and
    * keeps every pixel in parallel within one.
    *
    * Discard already owns the multi-pass machinery for its own reasons, so the
    * two do not combine yet and alpha-tested draws keep the retry path.
    */
   /*
    * A side-effect-only pass needs every layer for the same reason a blended
    * one does, and needs it more literally: the visibility buffer resolves one
    * fragment per pixel, so without peeling the linked list `oit` builds gets
    * a single node per pixel — the nearest — and the sort in its second pass
    * has nothing to sort. Peeling shades each covered fragment exactly once,
    * in submission order, which is what a shader with side effects is entitled
    * to. Written as a disjoint arm rather than folded into the condition
    * above, so that a draw which has a colour attachment reaches this line
    * with exactly the answer it reached it with before.
    */
   bool peel = !retry && cp->peel_next && screen->kernels.peel_advance &&
               ((color_data && cp->blend_enabled) ||
                (!color_data && fs_side_effects));
   /* A draw can never stack more layers than it has primitives, so a blended
    * draw of two triangles costs two passes rather than the cap. */
   unsigned peel_passes = MIN2((unsigned)CP_BLEND_LAYERS,
                               MAX2(num_triangles, 1u));
   unsigned passes = retry ? CP_DISCARD_LAYERS
                   : peel ? peel_passes : 1;

   /* A segment append never runs the peel loop — on episode failure the
    * segment re-executes classically, which sets this up itself — and the
    * A-buffer count must see the unfiltered population, so peel_next stays
    * out of the rasterizer arguments. */
   if (peel && !cp->pass.appending) {
      cuMemsetD32Async(cp->peel_next, 0, (size_t)w * h, cp->stream);
      rast_args.peel_next = cp->peel_next;
      rast_args.peel_any = cp->peel_any;
      rast_args.blend_peel = 1;
   }

   if (retry) {
      cuMemsetD8Async(cp->resolved, 0, (size_t)w * h, cp->stream);
      cuMemsetD32Async(cp->reject, 0xFFFFFFFF,
                  (size_t)w * h * CP_DISCARD_LAYERS, cp->stream);
      rast_args.reject = cp->reject;
      rast_args.resolved = cp->resolved;
      rast_args.reject_layers = CP_DISCARD_LAYERS;
   }

   /*
    * Every pass shades into the same scratch. Without this the arena grows by
    * a pass's worth of buffers each time round — for a 1280x720 draw with
    * five fragment inputs that is a couple of hundred megabytes a pass, which
    * a 256 layer blended draw turns into tens of gigabytes of managed memory
    * and an out-of-memory kill. Rewinding is safe because kernels in the
    * default stream are serialized: the next pass cannot start writing these
    * buffers until this pass has finished reading them.
    *
    * Both arenas, and for the same reason: the shading buffers moved to the
    * device-only one, and rewinding only the arena they had left produced a
    * particlesystem whose passes each allocated afresh until the cap refused
    * them and the stages downstream silently drew nothing.
    */
   size_t shade_mark = cp->scratch.used;
   size_t shade_dmark = cp->dscratch.used;

   /*
    * How many peel passes to launch between convergence checks.
    *
    * Reading cp->peel_any is a host read of memory a kernel just wrote, so it
    * costs a full device drain — the pipeline empties and refills. Doing that
    * once per layer is the single most expensive synchronisation in the
    * driver: particlesystem's fire is 512 additive sprites piled tens deep, it
    * runs the full CP_BLEND_LAYERS passes because it never converges early,
    * and it was paying a drain for each of them to be told so.
    *
    * The check is only ever "has this stopped compositing", and once a pass
    * composites nothing every later pass does too, because peel_next only
    * moves forward. So it is safe to ask less often, provided the flag is
    * reset once per interval rather than once per pass and the answer is read
    * as "did any pass in this interval do anything".
    *
    * Doubling from one keeps the common case exact — a blended draw that does
    * not overlap itself converges on pass 2 and is still detected on pass 2,
    * with nothing wasted — while a draw that runs to the cap pays a drain
    * roughly every CP_PEEL_CHECK_MAX passes instead of every pass. The cap
    * bounds the overshoot: a draw converging just after a check runs at most
    * CP_PEEL_CHECK_MAX-1 further passes, which against the 256 layer cap is
    * a few percent, where unbounded doubling would risk running twice the
    * passes the draw needed.
    */
   unsigned check_interval = 1;
   unsigned interval_start = 0;

   /*
    * Which primitive goes to which stage is decided from the geometry and the
    * clip rectangle alone, and a peeled draw changes neither between passes —
    * peel_next is consumed inside emit_fragment, long after coverage. So the
    * 256 passes of particlesystem's fire were each rebuilding queues identical
    * to the ones the pass before had just thrown away.
    *
    * The queues are context-lifetime cuMemAlloc buffers, not scratch, so
    * nothing the loop rewinds below can touch them: the first pass builds
    * them, the rest reuse them, and the counters are simply not reset.
    */
   bool cache_queues = peel && !cp_debug->no_bincache;

   /*
    * TEMPORARY: fragment census. One line per draw so the peeled one can be
    * picked out of the frame, and per-pixel counters wired in for pass 0 only.
    */
   static CUdeviceptr census_buf = 0, census_depth_buf = 0;
   static unsigned census_w = 0, census_h = 0;
   static unsigned census_draw_seq = 0, census_peel_seq = 0;
   unsigned this_draw_seq = census_draw_seq;
   bool census = false;
   if (cp_census_enabled()) {
      census_draw_seq++;
      fprintf(stderr, "census: draw=%u tris=%u passes=%u peel=%d retry=%d "
              "blend=%d fb=%ux%u\n", this_draw_seq, num_triangles, passes,
              peel ? 1 : 0, retry ? 1 : 0, cp->blend_enabled ? 1 : 0, w, h);
      if (peel && w && h) {
         if (census_w != w || census_h != h) {
            if (census_buf)
               cuMemFree(census_buf);
            if (census_depth_buf)
               cuMemFree(census_depth_buf);
            census_buf = census_depth_buf = 0;
            cuMemAlloc(&census_buf, (size_t)w * h * sizeof(uint32_t));
            cuMemAlloc(&census_depth_buf, (size_t)w * h * sizeof(uint32_t));
            census_w = w;
            census_h = h;
         }
         if (census_buf && census_depth_buf) {
            census = true;
            cuMemsetD32Async(census_buf, 0, (size_t)w * h, cp->stream);
            cuMemsetD32Async(census_depth_buf, 0, (size_t)w * h, cp->stream);
         }
      }
   }

   /*
    * TEMPORARY: build and check the A-buffer for this draw. Everything here
    * happens before the peel loop and writes nothing the loop reads, except
    * that the extra rasterization passes have to leave the visibility buffer
    * as they found it — they do, because emit_fragment returns before it.
    */
   struct cp_abuf *ab = &cp_abuf;
   bool abuf = false;
   bool abuf_log = false;
   unsigned abuf_deep_n = 0;
   uint32_t abuf_total = 0;
   /* The records buffer the count pass appended into, if it appended at all.
    * Compared against ab->recs again at the fill: the first-draw bootstrap
    * sizes the arrays between count and fill, and a records buffer allocated
    * or replaced there holds nothing this draw's count wrote. */
   CUdeviceptr abuf_recs_filled = 0;
   /* Whether this draw is rendered by the A-buffer instead of by the peel
    * loop, and how many quads its merge produced. Both settled after the
    * merge; see there. */
   bool abuf_prod = false;
   uint32_t abuf_quads = 0, abuf_covered = 0;
   if (cp_abuf_enabled() && peel && w && h && !ab->disabled) {
      /* Why a draw is not eligible is a question about one run, and this is a
       * path every blended draw now reaches — so it is said once, and only
       * when CUDAPIPE_ABUFFER_DEBUG asked. */
      static int said = 0;
      if (!cp_abuf.debug)
         said = 1;
      /*
       * Asking for the comparison against the peel loop without the
       * instrumentation to make it is not a state to render in: the log the
       * comparison reads is written by kernels CUDAPIPE_ABUF_COMPILE=0
       * refused, so every pixel reads as a mismatch. Say so once and render
       * the ordinary way rather than print a thousand false ones.
       */
      if (cp_abuf.verify && !screen->kernels.abuf_peel_log) {
         fprintf(stderr, "abuffer: CUDAPIPE_ABUFFER_VERIFY needs the "
                 "instrumentation CUDAPIPE_ABUF_COMPILE=0 refused — "
                 "not verifying\n");
         cp_abuf.verify = 0;
         cp_abuf.composite = 1;
      }
      if (!screen->kernels.abuf_quad_fill) {
         if (!said++)
            fprintf(stderr, "abuffer: kernels not compiled in — skipped\n");
         ab->disabled = true;
      } else if (fb_samples != 1) {
         if (!said++)
            fprintf(stderr, "abuffer: %u samples per pixel; peel path only, "
                    "single-sampled only — skipped\n", fb_samples);
      } else if (cp->depth_stencil.depth_writemask) {
         /*
          * A draw that writes depth changes the depth-test outcome between
          * peel passes, so a population counted once before the loop is not
          * the one the loop sees — and the composite has no per-layer moment
          * at which to commit a depth value either. Tested on the write mask
          * alone rather than on it and the test together, because
          * cp_fs_writeback does: a draw with the test off and the mask on
          * still writes depth there.
          */
         if (!said++)
            fprintf(stderr, "abuffer: draw writes depth (test=%d mask=%d), so "
                    "the depth-passing population changes between peel passes "
                    "— skipped\n", cp->depth_stencil.depth_enabled ? 1 : 0,
                    cp->depth_stencil.depth_writemask ? 1 : 0);
      } else if (fb->nr_cbufs != 1) {
         /* The composite writes one attachment, as the writeback does. */
         if (!said++)
            fprintf(stderr, "abuffer: %u colour attachments; one only — "
                    "skipped\n", fb->nr_cbufs);
      } else if (fb->color_encoding < 0) {
         if (!said++)
            fprintf(stderr, "abuffer: the colour format has no encoding — "
                    "skipped\n");
      } else {
         abuf = cp_abuf_setup(ab, w, h);
      }
   }

   /*
    * A growth an earlier draw asked for. It happens here, before this draw
    * touches anything, because the draw that asked for it was still reading
    * the arrays it frees — its merge and its composite were issued after the
    * count it asked on. The count pass below reads none of them.
    *
    * Rare by construction: the arrays are sized with headroom and this is
    * asked at three quarters of it. No draw on the traced workloads asks at
    * all after the first.
    */
   /*
    * A segment append can only proceed onto the shared-episode path: no
    * growth pending (handled between episodes), a fragment array to fill, and
    * compositing mode. Anything else backs out — the
    * caller finishes the episode without this segment and re-executes it
    * classically. The vertex work above is repeated then; the case is rare.
    */
   if (cp->pass.appending && !cp->pass.opaque &&
       (!abuf || ab->grow_to || !ab->frags || !cp_abuf.composite ||
        cp->pass.nsegs >= CP_PASS_MAX_SEGS)) {
      cp->pass.append_failed = true;
      return;
   }

   if (abuf && ab->grow_to) {
      uint32_t want = ab->grow_to;
      ab->grow_to = 0;
      /* cuMemFree already blocks until the device has finished; the drain is
       * written out so that this does not rest on that. */
      cuStreamSynchronize(cp->stream);
      if (!cp_abuf_size_arrays(ab, want))
         abuf = false;
   }

   if (abuf) {
      size_t n = (size_t)w * h;
      struct cp_rasterize_args aa = rast_args;
      aa.abuf_counts = ab->counts;
      aa.abuf_offsets = ab->offsets;
      aa.abuf_cursor = ab->cursor;
      aa.abuf_overflow = ab->overflow;

      /* --- step 1: count ---
       *
       * An episode's per-pixel counts accumulate across its segments, so the
       * clears run for the first segment only; every segment's fragments then
       * carry its own primitive-slot base, which is what makes the episode's
       * one sort come out in submission order. */
      /* An episode's clears run once, on the main stream in cp_pass_append(),
       * gated ahead of every segment stream. */
      if (!cp->pass.appending) {
         cuMemsetD32Async(ab->counts, 0, n, cp->stream);
         cuMemsetD32Async(ab->sum3, 0, 3, cp->stream);
         if (ab->recs)
            cuMemsetD32Async(ab->rec_cursor, 0, 1, cp->stream);
      }
      if (cp->pass.appending)
         aa.abuf_prim_base = cp->pass.next_prim;
      /* Single-pass build: this counting launch also appends the records the
       * fill will replay, so it needs the array bound the fill would have
       * used. First-draw bootstrap runs count-only — the records array is
       * sized from this draw's count, along with everything else. */
      if (ab->recs) {
         aa.abuf_recs = ab->recs;
         aa.abuf_rec_cursor = ab->rec_cursor;
         aa.abuf_capacity = ab->capacity;
         abuf_recs_filled = ab->recs;
      }
      cuMemsetD32Async(cp->cur_qset.counts, 0, 2, cp->stream);
      aa.abuf_mode = CP_ABUF_COUNT;
      rast_queues.mode = CP_QUEUE_FILL;
      cp_abuf_mark(ab->ev[0], cp->stream);
      void *ap[] = { &aa, &rast_queues };
      /* The _abuf specialisations: same rasterizer, compiled with the count
       * and fill branch live. Every other launch in this file uses the plain
       * ones, which have no A-buffer code in them at all. */
      CP_LAUNCH(screen->kernels.rasterize_stage1_abuf,
                     (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, ap, NULL);
      CP_LAUNCH(screen->kernels.rasterize_stage2_abuf,
                     CLAMP((rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                     256, 1, 1, 0, cp->stream, ap, NULL);
      CP_LAUNCH(screen->kernels.rasterize_stage3_abuf,
                     CLAMP(rast_num_triangles * 8, 512u, 2048u), 1, 1,
                     64, 1, 1, 0, cp->stream, ap, NULL);
      cp_abuf_mark(ab->ev[1], cp->stream);

      /* The segment is counted; everything from the scan on happens once,
       * at cp_pass_finish(). */
      if (cp->pass.appending) {
         cp_pass_record_segment(cp, &aa, &rast_queues, rast_num_triangles,
                                num_triangles, info, drawid_offset,
                                batch_draws, draws, instance_counts, vs_ubo_table,
                                fs_ubo_table, draw_ids, vb_table, scissors);
         return;
      }

      /* --- step 2: prefix sum --- */
      cp_abuf_scan(cp, screen, ab, (unsigned)n);
      cp_abuf_mark(ab->ev[2], cp->stream);

      /*
       * The count stays on the device.
       *
       * The host used to drain here for the total, and wanted it for three
       * things: to size the fragment array, to refuse a draw that outran it,
       * and to record the peak. None of the three has to be answered at this
       * point in the draw, and answering it here is the most expensive
       * synchronisation in the frame — 482 blended draws a frame on the
       * headless_streamer capture each stop the host until the device has
       * caught up, to re-derive a number whose answer never changes.
       *
       * What replaces it is machinery that was already there:
       *
       *  - The fill is bounded on the device by ab->capacity and counts every
       *    fragment it could not place into ab->overflow. It cannot write past
       *    the array whatever the count turns out to be.
       *  - The drain after the merge, which this path pays anyway to decide
       *    whether the peel loop runs, already reads that counter and refuses
       *    the draw when it is not zero — see the eligibility check below.
       *
       * So a draw that outruns the array is caught one step later by the test
       * that was always making the final decision, having spent a fill and a
       * merge and nothing else. Nothing has been written that the peel loop
       * reads: both A-buffer rasterization passes return before the visibility
       * buffer, and the composite has not been issued. The peel loop renders
       * that draw exactly as it did before this change.
       *
       * Two paths still need the total here and still drain for it: the first
       * eligible draw, which has no array to fill and no size to give the
       * fill, and every mode that does not composite — the verification reads
       * the per-pixel counts on the host before the fill, and there is no
       * later drain in those modes to read the total in.
       */
      bool drain_for_count = !ab->frags || !cp_abuf.composite;

      if (!drain_for_count) {
         /* The final prefix-add clamps runs against the fragment array and
          * records overflow while it already has every offset in registers. */
      } else {
         cuStreamSynchronize(cp->stream);
         cuMemcpyDtoH(&abuf_total, ab->sum3, sizeof(uint32_t));

         /* Whether the arrays are big enough for this draw, and whether they
          * should be made bigger before the next one. Both live in one place;
          * see cp_abuf_size_arrays(). */
         if (abuf_total > ab->peak)
            ab->peak = abuf_total;
         /* size_arrays may free and replace the records buffer, and a
          * replacement can land at the old address — so a drained draw always
          * refills by rasterizing. These are the bootstrap and the
          * non-compositing modes; every hot draw takes the clamp branch. */
         abuf_recs_filled = 0;
         if (!cp_abuf_size_arrays(ab, abuf_total))
            abuf = false;
         else if (abuf_total > ab->capacity) {
            /* The growth was capped or refused and the draw outran what is
             * there. Correct and slow rather than wrong: the peel loop renders
             * it. */
            static int said_over = 0;
            if (!said_over++)
               fprintf(stderr, "abuffer: %u fragments exceeds the %u allocated; "
                       "this draw falls back to the peel path\n",
                       abuf_total, ab->capacity);
            abuf = false;
         }
      }
   }

   if (abuf) {
      size_t n = (size_t)w * h;
      struct cp_rasterize_args aa = rast_args;
      aa.abuf_counts = ab->counts;
      aa.abuf_offsets = ab->offsets;
      aa.abuf_cursor = ab->cursor;
      aa.abuf_frags = ab->frags;
      aa.abuf_overflow = ab->overflow;
      aa.abuf_capacity = ab->capacity;

      /* Which pixels to compare to full depth. Chosen here because this is
       * the first moment the counts exist on the host, and the peel loop that
       * has to log them has not started. */
      if (cp_abuf.verify) {
         cuMemcpyDtoH(ab->h_counts, ab->counts, n * sizeof(uint32_t));
         struct cp_abuf_deep *d = malloc(n * sizeof(*d));
         if (d) {
            size_t m = 0;
            for (size_t i = 0; i < n; i++)
               if (ab->h_counts[i]) {
                  d[m].count = ab->h_counts[i];
                  d[m].pixel = (uint32_t)i;
                  m++;
               }
            qsort(d, m, sizeof(*d), cp_abuf_cmp_deep);
            abuf_deep_n = (unsigned)MIN2(m, (size_t)CP_ABUF_DEEP_PIXELS);
            for (unsigned i = 0; i < abuf_deep_n; i++)
               ab->h_deep_list[i] = d[i].pixel;
            free(d);
            cuMemcpyHtoD(ab->deep_list, ab->h_deep_list, abuf_deep_n * 4);
            cuMemsetD32Async(ab->log, 0xFFFFFFFF,
                             n * CP_ABUF_LOG_LAYERS, cp->stream);
            cuMemsetD32Async(ab->deep_log, 0xFFFFFFFF,
                             (size_t)CP_ABUF_DEEP_PIXELS * CP_BLEND_LAYERS,
                             cp->stream);
            abuf_log = true;
         }
      }

      /* --- step 3: fill --- */
      cuMemsetD32Async(ab->cursor, 0, n, cp->stream);
      cp_abuf_mark(ab->ev[3], cp->stream);
      if (abuf_recs_filled && screen->kernels.abuf_fill_recs) {
         /* The count pass already appended every (pixel, prim) record; the
          * fill is a linear replay instead of a second rasterization. */
         void *fp[] = { &ab->recs, &ab->rec_cursor, &ab->capacity, &ab->frags,
                        &ab->offsets, &ab->counts, &ab->cursor, &ab->overflow,
                        &ab->capacity };
         CP_LAUNCH(screen->kernels.abuf_fill_recs, 1024, 1, 1, 256, 1, 1,
                        0, cp->stream, fp, NULL);
      } else {
         cuMemsetD32Async(cp->cur_qset.counts, 0, 2, cp->stream);
         aa.abuf_mode = CP_ABUF_FILL;
         rast_queues.mode = CP_QUEUE_FILL;
         void *ap[] = { &aa, &rast_queues };
         CP_LAUNCH(screen->kernels.rasterize_stage1_abuf,
                        (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, ap, NULL);
         CP_LAUNCH(screen->kernels.rasterize_stage2_abuf,
                        CLAMP((rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                        256, 1, 1, 0, cp->stream, ap, NULL);
         CP_LAUNCH(screen->kernels.rasterize_stage3_abuf,
                        CLAMP(rast_num_triangles * 8, 512u, 2048u), 1, 1,
                        64, 1, 1, 0, cp->stream, ap, NULL);
      }
      cp_abuf_mark(ab->ev[4], cp->stream);

      /* --- step 4: sort. The worklist build is inside this measurement: it
       * is a prerequisite of the sort as written, not a separate step. --- */
      {
         unsigned nn = (unsigned)n, min2 = 2;
         cuMemsetD32Async(ab->list_count, 0, 1, cp->stream);
         void *wp[] = { &ab->counts, &nn, &min2, &ab->list, &ab->list_count };
         CP_LAUNCH(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, wp, NULL);
         void *sp[] = { &ab->frags, &ab->offsets, &ab->counts, &ab->list,
                        &ab->list_count, &ab->long_runs };
         CP_LAUNCH(screen->kernels.abuf_sort, 4096, 1, 1, 256, 1, 1,
                        0, cp->stream, sp, NULL);
      }
      cp_abuf_mark(ab->ev[5], cp->stream);

      /* The composite's worklist — every pixel with anything in it, not just
       * the ones worth sorting. Built here so it rides alongside the sort
       * rather than serialising behind the shading. */
      if (ab->clist) {
         unsigned nn = (unsigned)n, min1 = 1;
         cuMemsetD32Async(ab->clist_count, 0, 1, cp->stream);
         void *wp[] = { &ab->counts, &nn, &min1, &ab->clist, &ab->clist_count };
         CP_LAUNCH(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, wp, NULL);
      }

      /*
       * --- step 5: the quad stream ---
       *
       * The fragment shader consumes 2x2 quads of one primitive each, so the
       * per-pixel lists have to become a list of (block, primitive, coverage
       * mask). Blocks first, so that the merge visits the ~4% of the
       * framebuffer with anything in it rather than all 230,400 blocks.
       */
      unsigned nblocks = ab->nblocks, qw = ab->quad_width;
      cuMemsetD32Async(ab->blk_list_count, 0, 1, cp->stream);
      cp_abuf_mark(ab->ev[6], cp->stream);
      {
         void *p[] = { &ab->counts, &w, &h, &qw, &nblocks, &ab->blk_list,
                       &ab->blk_list_count };
         CP_LAUNCH(screen->kernels.abuf_block_worklist,
                        (nblocks + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab->ev[7], cp->stream);

      /* Counting pass, prefix sum, filling pass — the same shape as the
       * fragment lists themselves, and for the same reason: the merge has to
       * know where a block's quads go before it can write them. */
      cuMemsetD32Async(ab->blk_counts, 0, nblocks, cp->stream);
      cuMemsetD32Async(ab->bsum3, 0, 2, cp->stream);
      cuMemsetD32Async(ab->dbg, 0, CP_ABUF_DBG_COUNTERS, cp->stream);
      {
         void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                       &ab->blk_list, &ab->blk_list_count, &ab->blk_counts };
         /* 32 threads a block, not 256: there are only ~8,100 covered blocks,
          * and a wide thread block packs them into a few dozen CUDA blocks
          * that occupy a fraction of the SMs. */
         CP_LAUNCH(screen->kernels.abuf_quad_count, 1024, 1, 1, 32, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab->ev[9], cp->stream);
      cp_abuf_scan_n(cp, screen, ab->blk_counts, ab->blk_offsets, ab->bsum1,
                     ab->bsum1x, ab->bsum2, ab->bsum2x, ab->bsum3, nblocks,
                     ab->bnb1, ab->bnb2, ab->bnb3, 0, 0, 0);
      {
         void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                       &ab->blk_list, &ab->blk_list_count, &ab->blk_offsets,
                       &ab->quad_prim, &ab->quad_mask, &ab->peel_mask,
                       &ab->quad_block, &ab->shade_slot, &ab->quad_capacity,
                       &ab->quad_overflow };
         CP_LAUNCH(screen->kernels.abuf_quad_fill, 1024, 1, 1, 32, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab->ev[10], cp->stream);
      cp_abuf_mark(ab->ev[8], cp->stream);

      /*
       * Whether this draw is rendered by the A-buffer rather than merely
       * described by it.
       *
       * The host has to see three device-side numbers to say so, and this is
       * where it costs least: the merge has just been issued and the peel loop
       * has not started, so the drain replaces one the loop would have paid
       * anyway. What is being asked is whether the lists are complete —
       * whether the fill placed every fragment it counted, whether the merge
       * placed every quad, and whether there are any quads at all. A "no" to
       * any of them is a draw the peel loop renders, not a draw rendered
       * approximately.
       */
      /*
       * A draw of a couple of triangles cannot overflow anything: its
       * fragments are bounded by primitives times pixels and its quads by
       * primitives times blocks, both provably inside the arrays. So the
       * drain that exists to ask whether the lists are complete has a known
       * answer, and the shading launches size themselves to the bound with
       * the interpolator stopping at the device's own quad total. This is
       * the bloom pyramid's shape — a fullscreen filter quad per pass,
       * twenty drains a frame on the capture — and it keeps that chain fed
       * instead of emptying the device between filters. Batched draws keep
       * the drain (a garbage batch row would dereference a garbage table
       * entry), as do memory-writing shaders (their side-effect gate reads
       * coverage the interpolator never wrote for slots past the total).
       */
      bool bounded = cp_abuf.composite && !cp_abuf.verify && !cp_abuf.timing &&
                     num_triangles <= 2 && cp->fs_batch.ndraws <= 1 &&
                     cp->fs_shader && !cp->fs_shader->writes_memory &&
                     (size_t)ab->nblocks * rast_num_triangles * 4 <=
                        (size_t)(512u << 10) &&
                     (size_t)rast_num_triangles * n <= ab->capacity &&
                     (size_t)ab->nblocks * rast_num_triangles <=
                        ab->quad_capacity;
      if (bounded) {
         abuf_prod = true;
         abuf_quads = (uint32_t)((size_t)ab->nblocks * rast_num_triangles);
         abuf_covered = 0;   /* the composite covers the framebuffer */
      } else if (cp_abuf.composite) {
         uint32_t ctr[CP_ABUF_COUNTERS] = { 0 };
         cuStreamSynchronize(cp->stream);
         /* One copy: sum3, bsum3 and clist_count are contiguous. The last of
          * them is free rather than merely cheap now — the composite's grid is
          * the covered pixels rather than the framebuffer, which on a sample
          * covering 4% of it is 140 blocks instead of 3,600. */
         cuMemcpyDtoH(ctr, ab->counters, sizeof(ctr));
         const uint32_t *c3 = &ctr[0];
         const uint32_t *q = &ctr[3];
         abuf_covered = ctr[5];
         abuf_quads = q[0];
         abuf_prod = q[0] != 0 && q[1] == 0 && c3[1] == 0;
         if (!abuf_prod) {
            static int said_prod = 0;
            if (!said_prod++)
               fprintf(stderr, "abuffer: quads=%u quad-overflow=%u "
                       "fragment-overflow=%u — this draw falls back to the "
                       "peel loop\n", q[0], q[1], c3[1]);
         }

         /*
          * The population, arriving one drain later than it used to, and the
          * only place it is read now. The peak is exact; the growth it asks
          * for is served by the next eligible draw, because this one's merge
          * and composite are still reading the arrays it would free. A draw
          * that outran the array does not wait for a bigger one — the check
          * above has already sent it to the peel loop.
          */
         abuf_total = c3[0];
         if (abuf_total > ab->peak)
            ab->peak = abuf_total;
         if (ab->frags && !ab->grow_capped &&
             ab->growths < CP_ABUF_MAX_GROWTHS &&
             (double)abuf_total > (double)ab->capacity * CP_ABUF_GROW_AT)
            ab->grow_to = MAX2(ab->grow_to, abuf_total);
      }

      /* Hand the quad array to the interpolator, so that the peel loop about
       * to run records what it emits against it. Only while there is still a
       * verification to do: a frame that is only being timed must not carry
       * the instrumented interpolator. */
      if (cp_abuf.verify && ab->verified < cp_abuf.verify_max) {
         cp_abuf_dbg.blk_offsets = ab->blk_offsets;
         cp_abuf_dbg.blk_counts = ab->blk_counts;
         cp_abuf_dbg.quad_prim = ab->quad_prim;
         cp_abuf_dbg.peel_mask = ab->peel_mask;
         cp_abuf_dbg.counters = ab->dbg;

         /* Step 3b: and where to deposit each pass's shaded colours. The
          * write counts start at zero every draw; the colours themselves need
          * no clearing, since a slot nothing wrote is never read. */
         if (ab->colors_ready) {
            cuMemsetD32Async(ab->writes_peel, 0, ab->capacity, cp->stream);
            cuMemsetD32Async(ab->writes_abuf, 0, ab->capacity, cp->stream);
            cp_abuf_dbg.frags = ab->frags;
            cp_abuf_dbg.offsets = ab->offsets;
            cp_abuf_dbg.counts = ab->counts;
            cp_abuf_dbg.colors = ab->colors_peel;
            cp_abuf_dbg.writes = ab->writes_peel;
            cp_abuf_dbg.capacity = ab->capacity;
         }
      }
   }

   /* The whole point of the path: an eligible draw is counted, scanned,
    * filled, sorted, merged, interpolated, shaded and composited once, and the
    * peel loop does not run at all. */
   if (abuf_prod)
      passes = 0;

   unsigned passes_run = 0;
   for (unsigned pass = 0; pass < passes; pass++) {
      passes_run = pass + 1;
      CP_NVTX_SCOPEF("pass %u", pass);
      cp->scratch.used = shade_mark;
      cp->dscratch.used = shade_dmark;
      if (pass) {
         /* Each pass resolves visibility afresh, minus what has been rejected. */
         cuMemsetD32Async(visbuf, 0xFFFFFFFF, (size_t)w * h * 2 * fb_samples, cp->stream);
         rast_args.reject_passes = pass;
      }
      /* Cleared at the start of each interval, not each pass: the question
       * asked below is whether any pass since the last check composited. */
      if (peel && pass == interval_start)
         *(volatile uint32_t *)(uintptr_t)cp->peel_any = 0;
      rast_queues.mode = !cache_queues ? CP_QUEUE_FILL
                       : pass ? CP_QUEUE_REUSE : CP_QUEUE_BUILD;

      /* A tiled opaque episode needs the completed vertex geometry now, but
       * not the ordinary visibility pass. Save the launch description and
       * let cp_opaque_finish enqueue mutually exclusive tiled-success and
       * classic-overflow paths from one device-resident predicate. */
      if (cp->pass.appending && cp->pass.opaque &&
          cp_debug->tiled_opaque) {
         rast_queues.mode = CP_QUEUE_FILL;
         cp_pass_record_segment(cp, &rast_args, &rast_queues,
                                rast_num_triangles, num_triangles, info,
                                drawid_offset, batch_draws, draws,
                                instance_counts, vs_ubo_table, fs_ubo_table,
                                draw_ids, vb_table, scissors);
         return;
      }

      /* TEMPORARY: count the first pass only — it rasterizes the whole draw,
       * and the launch below copies these by value, so later passes see 0. */
      if (census) {
         rast_args.census = pass == 0 ? census_buf : 0;
         rast_args.census_depth = pass == 0 ? census_depth_buf : 0;
      }

      /* Both counters in one call; they are adjacent for this reason. The
       * memset is enqueued on the default stream, so it serializes properly
       * with the preceding pass's kernels. A reusing pass keeps the counts
       * the first pass arrived at — clearing them would leave stages 2 and 3
       * reading an empty queue. */
      if (rast_queues.mode != CP_QUEUE_REUSE)
         cuMemsetD32Async(cp->cur_qset.counts, 0, 2, cp->stream);

      /* Stage 1: 1 thread per triangle (small rasterize in place, others queue) */
      cp_nvtx_push("raster");
      void *s1_params[] = { &rast_args, &rast_queues };
      CUresult rast_err = cuLaunchKernel(screen->kernels.rasterize_stage1,
         (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
         0, cp->stream, s1_params, NULL);

      /*
       * Both later stages stride their queue, so any grid is correct and the
       * grid is only a statement of how much of the machine to use. That makes
       * it worth bounding by the draw as well as by the device: the queues can
       * hold at most one entry per triangle for stage 2, and a draw of a dozen
       * triangles was launching 131,072 threads at each of them to have all
       * but a handful read the counter and exit. dynamicuniformbuffer draws 625
       * cubes of twelve triangles a frame and spends most of it in launches.
       *
       * Stage 3 gets no such bound. Its entries are tiles, not primitives, and
       * how many tiles a primitive covers is only known on the device: a single
       * full-screen quad is two triangles and 510 tiles. Sizing that grid from
       * the triangle count was measured and is a clear loss — texturemipmapgen
       * 17% slower, texturecubemap and computeshader 12% — because the samples
       * with the fewest triangles are exactly the ones covering whole tiles
       * with them.
       */
      unsigned s2_blocks = CLAMP((rast_num_triangles + 7) / 8, 1u, 512u);
      unsigned s3_blocks = 2048;

      /* Stage 2: warp-cooperative, grid-strided over the nontrivial queue */
      void *s2_params[] = { &rast_args, &rast_queues };
      CP_LAUNCH(screen->kernels.rasterize_stage2,
         s2_blocks, 1, 1, 256, 1, 1,
         0, cp->stream, s2_params, NULL);

      /* Stage 3: block per tile, grid-strided over the huge-tile queue */
      void *s3_params[] = { &rast_args, &rast_queues };
      CP_LAUNCH(screen->kernels.rasterize_stage3,
         s3_blocks, 1, 1, 64, 1, 1,
         0, cp->stream, s3_params, NULL);

      if (rast_err != CUDA_SUCCESS && cp_debug->debug_draw)
         fprintf(stderr, "  rasterize launch failed: %d\n", rast_err);
      cp_nvtx_pop();   /* raster */
      cp_stage_end(cp, CP_STAGE_RASTERIZE);

      if (cp->pass.appending && cp->pass.opaque) {
         cp_pass_record_segment(cp, &rast_args, &rast_queues,
                                rast_num_triangles, num_triangles, info,
                                drawid_offset, batch_draws, draws,
                                instance_counts, vs_ubo_table, fs_ubo_table, draw_ids,
                                vb_table, scissors);
         return;
      }

      /* TEMPORARY: read back the census of this one pass and print it. */
      if (census && pass == 0) {
         size_t bytes = (size_t)w * h * sizeof(uint32_t);
         uint32_t *raw = malloc(bytes), *dep = malloc(bytes);
         cuStreamSynchronize(cp->stream);
         if (raw && dep) {
            cuMemcpyDtoH(raw, census_buf, bytes);
            cuMemcpyDtoH(dep, census_depth_buf, bytes);
            cp_census_dump("raw", this_draw_seq, census_peel_seq,
                           num_triangles, MAX2(fb_samples, 1u), raw, w, h);
            cp_census_dump("depth-passed", this_draw_seq, census_peel_seq,
                           num_triangles, MAX2(fb_samples, 1u), dep, w, h);
         }
         free(raw);
         free(dep);
         census_peel_seq++;
      }

      /*
       * TEMPORARY: record what this peel pass selected, before anything
       * downstream reads or clears the visibility buffer. This is the
       * sequence the A-buffer lists are checked against.
       */
      if (abuf_log) {
         unsigned nn = w * h;
         unsigned layers = CP_ABUF_LOG_LAYERS;
         if (pass < layers) {
            void *p[] = { &visbuf, &ab->log, &layers, &pass, &nn };
            CP_LAUNCH(screen->kernels.abuf_peel_log, (nn + 255) / 256,
                           1, 1, 256, 1, 1, 0, cp->stream, p, NULL);
         }
         unsigned dlayers = CP_BLEND_LAYERS;
         if (abuf_deep_n && pass < dlayers) {
            void *p[] = { &visbuf, &ab->deep_list, &ab->deep_log, &dlayers,
                          &pass, &abuf_deep_n };
            CP_LAUNCH(screen->kernels.abuf_peel_log_list,
                           (abuf_deep_n + 255) / 256, 1, 1, 256, 1, 1,
                           0, cp->stream, p, NULL);
         }
      }

      /* Shade every covered pixel by running the fragment shader on the GPU:
       * interpolate its inputs, launch it, then blend its output into the
       * attachment — or, for a shader that exists for its stores rather than
       * for a colour, just the first two. */
      if (color_data || fs_side_effects)
         cp_shade_fragments(cp, info, visbuf, rast_args.positions, vs_output_buf,
                            num_triangles, w, h, color_data,
                            vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y,
                            retry ? cp->reject : 0,
                            retry ? cp->resolved : 0,
                            pass, 0, 0);

      if (peel) {
         /* Step past what this pass blended, and stop once the interval finds
          * nothing left to composite — which is the common case at pass 2,
          * since most blended draws do not overlap themselves. */
         void *pa_params[] = { &visbuf, &cp->peel_next, &w, &h };
         CP_LAUNCH(screen->kernels.peel_advance,
                        (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1,
                        0, cp->stream, pa_params, NULL);

         if (pass + 1 >= interval_start + check_interval) {
            cuStreamSynchronize(cp->stream);
            if (!*(volatile uint32_t *)(uintptr_t)cp->peel_any)
               break;
            interval_start = pass + 1;
            check_interval = MIN2(check_interval * 2,
                                  (unsigned)CP_PEEL_CHECK_MAX);
         }
      }
   }

   /*
    * TEMPORARY: report what the four steps cost, and check the lists against
    * the log the loop just wrote.
    */
   if (abuf) {
      float t_count = 0, t_scan = 0, t_fill = 0, t_sort = 0;
      float t_work = 0, t_merge = 0, t_mcount = 0, t_mfill = 0;

      /* The merge's own totals, and the counters it and the instrumented
       * interpolator share. Already in hand when compositing, which read them
       * to decide; the checking path still has to fetch them. */
      uint32_t qcounters[2] = { abuf_quads, 0 };
      uint32_t dbg[CP_ABUF_DBG_COUNTERS] = { 0 };
      if (!cp_abuf.composite) {
         cuStreamSynchronize(cp->stream);
         cuMemcpyDtoH(qcounters, ab->bsum3, sizeof(qcounters));
      }

      /*
       * --- steps 6 and 7: interpolate the quad stream, shade it, composite it
       *
       * When the peel loop ran, this has to come after it, so that the arena
       * being rewound into is the one the loop has finished with — the peel
       * path's own shading buffers are rewound between passes for exactly that
       * reason, and taking the arena from underneath it would be a rendering
       * change. When it did not run, there is nothing to come after.
       */
      float t_qinterp = 0, t_qshade = 0, t_composite = 0;
      bool shaded = false;
      /*
       * Shading a quad stream nothing will composite is work for nobody: a
       * draw the check above refused is rendered by the peel loop, which
       * shades it itself. The comparison modes still want it, because what
       * they compare is the shading.
       */
      if (qcounters[0] && !qcounters[1] && (abuf_prod || !cp_abuf.composite)) {
         cp->scratch.used = shade_mark;
         cp->dscratch.used = shade_dmark;
         shaded = cp_abuf_shade(cp, info, ab, rast_args.positions,
                                vs_output_buf, w, h, vp_scale_x, vp_scale_y,
                                vp_trans_x, vp_trans_y, qcounters[0],
                                abuf_covered,
                                ab->colors_ready && cp_abuf_dbg.colors,
                                color_data, abuf_prod,
                                &t_qinterp, &t_qshade, &t_composite, NULL);
         /*
          * A composite that could not be launched has drawn nothing, and the
          * peel loop has already been skipped. Nothing can recover the draw at
          * this point, so say so loudly rather than let a frame come out
          * missing a layer that nobody counted.
          */
         if (abuf_prod && !shaded) {
            fprintf(stderr, "abuffer: the composite path failed after the peel "
                    "loop was skipped — this draw drew nothing\n");
            ab->disabled = true;
         }
      }

      /* Read after the shading pass, so its own counter is in it. */
      if (cp_abuf.timing || cp_abuf.verify)
         cuMemcpyDtoH(dbg, ab->dbg, sizeof(dbg));

      if (cp_abuf.timing) {
         cuStreamSynchronize(cp->stream);
         cuEventElapsedTime(&t_count, ab->ev[0], ab->ev[1]);
         cuEventElapsedTime(&t_scan, ab->ev[1], ab->ev[2]);
         cuEventElapsedTime(&t_fill, ab->ev[3], ab->ev[4]);
         cuEventElapsedTime(&t_sort, ab->ev[4], ab->ev[5]);
         cuEventElapsedTime(&t_work, ab->ev[6], ab->ev[7]);
         cuEventElapsedTime(&t_merge, ab->ev[7], ab->ev[8]);
         cuEventElapsedTime(&t_mcount, ab->ev[7], ab->ev[9]);
         cuEventElapsedTime(&t_mfill, ab->ev[9], ab->ev[10]);

         fprintf(stderr,
                 "abuffer[%u]: %s tris=%u frags=%u quads=%u passes_run=%u\n"
                 "  count %6.3f  scan %6.3f  fill %6.3f  sort %6.3f"
                 "  worklist %6.3f  merge %6.3f (count %.3f + scan/fill %.3f)"
                 "  total %6.3f ms\n"
                 "  quad interp %6.3f  shade %6.3f  composite %6.3f%s"
                 "  whole path %6.3f ms\n",
                 ab->seq, abuf_prod ? "COMPOSITED" : "checked",
                 num_triangles, abuf_total, qcounters[0], passes_run,
                 t_count, t_scan, t_fill, t_sort, t_work, t_merge, t_mcount,
                 t_mfill,
                 t_count + t_scan + t_fill + t_sort + t_work + t_merge,
                 t_qinterp, t_qshade, t_composite, shaded ? "" : " (not run)",
                 t_count + t_scan + t_fill + t_sort + t_work + t_merge +
                 t_qinterp + t_qshade + t_composite);
      }

      /* Nothing downstream may see the debug pointers; the next draw's
       * interpolation must be the ordinary one. */
      memset(&cp_abuf_dbg, 0, sizeof(cp_abuf_dbg));

      if (abuf_log && ab->verified < cp_abuf.verify_max) {
         size_t n = (size_t)w * h;
         uint32_t counters[3] = { 0, 0, 0 };
         cuMemcpyDtoH(counters, ab->sum3, sizeof(counters));
         cuMemcpyDtoH(ab->h_counts, ab->counts, n * sizeof(uint32_t));
         cuMemcpyDtoH(ab->h_offsets, ab->offsets, n * sizeof(uint32_t));
         cuMemcpyDtoH(ab->h_cursor, ab->cursor, n * sizeof(uint32_t));
         cuMemcpyDtoH(ab->h_frags, ab->frags,
                      (size_t)abuf_total * sizeof(uint32_t));
         cuMemcpyDtoH(ab->h_log, ab->log,
                      n * CP_ABUF_LOG_LAYERS * sizeof(uint32_t));
         cuMemcpyDtoH(ab->h_deep_log, ab->deep_log,
                      (size_t)CP_ABUF_DEEP_PIXELS * CP_BLEND_LAYERS *
                      sizeof(uint32_t));
         cp_abuf_verify(ab, w, h, abuf_total, passes_run, abuf_deep_n,
                        counters[1], counters[2]);

         if (ab->quad_prim) {
            cuMemcpyDtoH(ab->h_blk_counts, ab->blk_counts,
                         (size_t)ab->nblocks * sizeof(uint32_t));
            cuMemcpyDtoH(ab->h_blk_offsets, ab->blk_offsets,
                         (size_t)ab->nblocks * sizeof(uint32_t));
            cuMemcpyDtoH(ab->h_quad_prim, ab->quad_prim,
                         (size_t)qcounters[0] * sizeof(uint32_t));
            cuMemcpyDtoH(ab->h_quad_mask, ab->quad_mask, qcounters[0]);
            cuMemcpyDtoH(ab->h_peel_mask, ab->peel_mask,
                         (size_t)qcounters[0] * sizeof(uint32_t));
            cp_abuf_verify_quads(ab, w, h, abuf_total, qcounters[0],
                                 passes_run, qcounters[1], dbg);
         }

         /* Step 3b: the shaded colours, one slot per (pixel, primitive). */
         if (shaded && ab->colors_ready) {
            cuMemcpyDtoH(ab->h_colors_abuf, ab->colors_abuf,
                         (size_t)abuf_total * 16);
            cuMemcpyDtoH(ab->h_colors_peel, ab->colors_peel,
                         (size_t)abuf_total * 16);
            cuMemcpyDtoH(ab->h_writes_abuf, ab->writes_abuf,
                         (size_t)abuf_total * sizeof(uint32_t));
            cuMemcpyDtoH(ab->h_writes_peel, ab->writes_peel,
                         (size_t)abuf_total * sizeof(uint32_t));
            cp_abuf_verify_colors(ab, w, h, abuf_total, passes_run,
                                  dbg[CP_ABUF_DBG_SLOT_BAD]);
         }
         ab->verified++;
      }
      ab->seq++;
   }

   if (cp_timing_enabled()) {
      /* Device time between the events recorded above. This synchronises, so
       * the numbers describe a draw that ran on its own — which is the point,
       * but means a total here will not add up to a frame measured by the
       * benchmark, where draws overlap the host. */
      double ms[CP_NUM_STAGES] = {0};
      cp_stage_resolve(cp, ms);
      double total = 0.0;
      for (int i = 0; i < CP_NUM_STAGES; i++)
         total += ms[i];
      fprintf(stderr,
              "cudapipe: %5u tris  assemble %6.3f  vertex %6.3f  raster %6.3f  "
              "interp %6.3f  fragment %6.3f  writeback %6.3f  total %6.3f ms\n",
              num_triangles, ms[CP_STAGE_ASSEMBLE], ms[CP_STAGE_VERTEX],
              ms[CP_STAGE_RASTERIZE], ms[CP_STAGE_INTERPOLATE],
              ms[CP_STAGE_FRAGMENT], ms[CP_STAGE_WRITEBACK], total);
   } else {
      /* Nothing was recorded, but keep the pool empty either way so a later
       * run with timing on does not resolve against a stale mark. */
      cp->timer.num = 0;
   }


   /* No sync needed here: kernels in the default stream are serialized, and
    * the next draw's rasterizer reads the depth buffer on the GPU — not the
    * host. Sync only when the host must read GPU results (e.g., readback). */
   FREE(refs);
}

/*
 * The opaque merge condition: a result that cannot depend on the order the
 * draws arrived in, because the visibility buffer resolves it with atomicMin.
 */
bool
cp_batch_order_free(struct cp_context *cp)
{
   if (cp->blend_enabled || cp->fs_shader->uses_discard)
      return false;
   if (!cp->depth_stencil.depth_enabled || !cp->depth_stencil.depth_writemask)
      return false;
   switch (cp->depth_stencil.depth_func) {
   case CP_FUNC_LESS:
   case CP_FUNC_LEQUAL:
   case CP_FUNC_GREATER:
   case CP_FUNC_GEQUAL:
      return true;
   default:
      return false;
   }
}

/*
 * The blended merge condition: a draw the A-buffer renders.
 *
 * Order here is not free — it is *carried*, by the primitive index the
 * A-buffer sorts on. Merging is legal only because the clipper lays a batch's
 * triangles out in submission order (see cp_clip_triangles' stable mode), so a
 * later draw's primitives sort after an earlier draw's exactly as they would
 * have if the draws had run one at a time.
 *
 * The conditions are the A-buffer's own, restated: this has to agree with the
 * test in cp_draw_execute, because a batch that ends up on the peel loop
 * instead composites its merged draws in the same primitive order and is
 * equally correct — but a batch that ends up anywhere else is not.
 */
bool
cp_batch_abuf_ok(struct cp_context *cp)
{
   struct cp_device *screen = cp->screen;
   const struct cp_fb_desc *fb = &cp->fb;

   if (!cp_abuf_enabled() || cp_abuf.disabled)
      return false;
   if (!screen->kernels.abuf_quad_fill || !screen->kernels.clip_triangles ||
       !screen->kernels.peel_advance)
      return false;

   /* What makes the draw peel at all — cp_draw_execute's `peel`. A discarding
    * shader takes the retry path instead, and its fragments are not the
    * A-buffer's population. */
   if (!cp->blend_enabled || cp->fs_shader->uses_discard || !cp->peel_next)
      return false;

   /* The A-buffer's own gate. */
   if (MAX2(cp->fb_samples, 1u) != 1 || cp->depth_stencil.depth_writemask)
      return false;
   if (fb->nr_cbufs != 1 || !fb->color || fb->color_encoding < 0)
      return false;

   /*
    * The vertex shader's outputs have to fit the clipper, because the stable
    * layout the ordering rests on is the clipper's. A draw wide enough to skip
    * clipping keeps the compacting path, where a batch's primitive indices
    * would still be in submission order — but it is one condition rather than
    * two, so it is refused here and left on the single-draw path.
    */
   unsigned nout = cp->vs_shader->nir_num_outputs ? cp->vs_shader->nir_num_outputs : 2;
   if (nout > CP_MAX_CLIP_SLOTS)
      return false;

   return true;
}

/* Snapshot this draw's index range and vertex-stage bindings as the next row
 * of the batch's tables. */
void
cp_batch_record(struct cp_context *cp,
                const struct cp_draw_range *draw, unsigned tris,
                unsigned drawid_offset, unsigned instance_count)
{
   uint64_t *row = cp->batch.vs_ubos +
      (size_t)cp->batch.ndraws * CP_ARG_UBO_STRIDE;
   memset(row, 0, CP_ARG_UBO_STRIDE * sizeof(*row));
   for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
      row[i] = (uint64_t)(uintptr_t)cp->vs_ubos[i].buffer;

   /* The fragment stage's — the bindings the key stopped comparing, for every
    * batch since the opaque path adopted the per-draw table too. Recorded now,
    * because by the time the batch runs the next draw's have been bound over
    * them. */
   {
      uint64_t *frow = cp->batch.fs_ubos +
         (size_t)cp->batch.ndraws * CP_ARG_UBO_STRIDE;
      memset(frow, 0, CP_ARG_UBO_STRIDE * sizeof(*frow));
      for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
         frow[i] = (uint64_t)(uintptr_t)cp->fs_ubos[i].buffer;
   }

   /* The vertex-buffer bases, resolved per element the way the launch would
    * resolve them — snapshotted for the same reason as the uniform rows. */
   {
      uint64_t *vrow = cp->batch.vb_bases +
         (size_t)cp->batch.ndraws * CP_VB_TABLE_STRIDE;
      memset(vrow, 0, CP_VB_TABLE_STRIDE * sizeof(*vrow));
      for (unsigned e = 0; e < cp->num_vertex_elements &&
                           e < CP_VB_TABLE_STRIDE; e++) {
         unsigned vb_idx = cp->velem[e].vertex_buffer_index;
         if (vb_idx < cp->num_vertex_buffers && vb_idx < 16 &&
             cp->vb_base[vb_idx]) {
            /* Base and offset were folded together when the buffer was
             * bound, which is the same value this computed for itself. */
            vrow[e] = cp->vb_base[vb_idx];
         }
      }
   }

   cp->batch.draws[cp->batch.ndraws] = *draw;
   cp->batch.instance_counts[cp->batch.ndraws] = instance_count;
   cp->batch.draw_ids[cp->batch.ndraws] = drawid_offset;
   cp->batch.scissors[cp->batch.ndraws] = cp->scissor;
   cp->batch.tris += tris;
   cp->batch.ndraws++;
}

/*
 * ---- Pass episodes ----
 *
 * Consecutive blended batches share one A-buffer build, one drain and one
 * composite. Each flushed blended batch becomes a segment: its vertex stage
 * and count rasterization ran at append time (inside cp_draw_execute, which
 * returned after the count when cp->pass.appending was set), its fragments
 * carrying an episode-global primitive base so the one sort's ascending
 * order is submission order across segments. cp_pass_finish() then runs the
 * scan, the per-segment fills, the sort, the quad merge, buckets the quads
 * by segment, drains once, shades each segment densely over its own quads,
 * and composites the lot through the per-quad segment map.
 *
 * An episode never spans a change to anything the composite or the
 * rasterizer read episode-wide — framebuffer, viewport, blend, depth,
 * rasterizer state — because every setter of those flushes through
 * cp_batch_flush_why(), which finishes the episode. Only the four
 * per-segment changes defer: a draw whose key broke the batch, and the
 * vertex-shader, fragment-shader and vertex-elements binds.
 */

static bool
cp_pass_appendable(struct cp_context *cp)
{
   struct cp_device *screen = cp->screen;
   struct cp_abuf *ab = &cp_abuf;

   if (cp_debug->no_pass_episode || cp_debug->no_abuf_batch)
      return false;
   /* The verification, timing and census modes read per-draw state the
    * episode deliberately does not keep. */
   if (!cp_abuf_enabled() || ab->disabled || cp_abuf.verify ||
       !cp_abuf.composite || cp_abuf.timing || cp_census_enabled())
      return false;
   if (!screen->kernels.abuf_seg_count || !screen->kernels.abuf_seg_scatter ||
       !screen->kernels.abuf_composite || !screen->kernels.abuf_interpolate ||
       !screen->kernels.abuf_quad_fill)
      return false;
   /* The first eligible draw sizes the fragment array with a drain of its
    * own; episodes start once it exists. A pending growth is likewise served
    * between episodes. */
   if (!ab->frags || ab->grow_to || !ab->shade_slot || !ab->clist)
      return false;
   if (cp->pass.nsegs >= CP_PASS_MAX_SEGS ||
       cp->pass.next_prim > (1u << 30))
      return false;
   if (!cp->pass_segs) {
      cp->pass_segs = calloc(CP_PASS_MAX_SEGS, sizeof(*cp->pass_segs));
      if (!cp->pass_segs)
         return false;
   }

   /* The side streams and their queue sets, once. Failure leaves every
    * seg_streams[] entry null and the episode runs on the main stream. */
   if (!cp->pass_streams_ready) {
      cp->pass_streams_ready = true;
      bool ok = cuEventCreate(&cp->pass_gate,
                              CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS;
      for (unsigned k = 0; ok && k < CP_PASS_STREAMS; k++) {
         ok = cuStreamCreate(&cp->seg_streams[k],
                             CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
              cuEventCreate(&cp->seg_ev[k],
                            CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS &&
              cuMemAlloc(&cp->seg_qsets[k].nontrivial,
                         (size_t)CP_MAX_NONTRIVIAL * sizeof(uint32_t)) ==
                 CUDA_SUCCESS &&
              cuMemAlloc(&cp->seg_qsets[k].huge_tiles,
                         (size_t)CP_MAX_HUGE_TILES *
                         sizeof(struct cp_tile_pair)) == CUDA_SUCCESS &&
              cuMemAlloc(&cp->seg_qsets[k].counts, 256) == CUDA_SUCCESS;
      }
      if (!ok) {
         fprintf(stderr, "cudapipe: pass-episode streams unavailable; "
                 "episodes run on the main stream\n");
         memset(cp->seg_streams, 0, sizeof(cp->seg_streams));
      }
   }
   return true;
}

static bool
cp_opaque_appendable(struct cp_context *cp)
{
   const struct cp_fb_desc *fb = &cp->fb;
   if (cp_debug->no_opaque_episode || !cp_batch_order_free(cp) ||
       !cp->fs_shader || cp->fs_shader->writes_memory ||
       MAX2(cp->fb_samples, 1u) != 1 || fb->nr_cbufs != 1 ||
       !fb->color || fb->color_encoding < 0)
      return false;
   if (cp->pass.opaque &&
       (cp->pass.nsegs >= CP_PASS_MAX_SEGS ||
        cp->pass.next_prim > (1u << 30)))
      return false;
   if (!cp->pass_segs) {
      cp->pass_segs = calloc(CP_PASS_MAX_SEGS, sizeof(*cp->pass_segs));
      if (!cp->pass_segs)
         return false;
   }
   return true;
}

/* The stream a segment's own work runs on: its slice of the side streams, or
 * the main stream when they could not be created. */
static CUstream
cp_pass_seg_stream(struct cp_context *cp, unsigned s)
{
   CUstream st = cp->seg_streams[s % CP_PASS_STREAMS];
   return st ? st : cp->stream;
}

/* Join every side stream a finished phase used back into the main stream. */
static void
cp_pass_join(struct cp_context *cp, unsigned nsegs)
{
   if (!cp->seg_streams[0])
      return;
   unsigned used = MIN2(nsegs, (unsigned)CP_PASS_STREAMS);
   for (unsigned k = 0; k < used; k++) {
      cuEventRecord(cp->seg_ev[k], cp->seg_streams[k]);
      cuStreamWaitEvent(cp->stream, cp->seg_ev[k], 0);
   }
}

/* The reverse: gate every side stream behind the main stream's tail. */
static void
cp_pass_broadcast(struct cp_context *cp, unsigned nsegs)
{
   if (!cp->seg_streams[0])
      return;
   cuEventRecord(cp->pass_gate, cp->stream);
   unsigned used = MIN2(nsegs, (unsigned)CP_PASS_STREAMS);
   for (unsigned k = 0; k < used; k++)
      cuStreamWaitEvent(cp->seg_streams[k], cp->pass_gate, 0);
}

/* Everything cp_draw_execute reads from live context state that varies per
 * segment, saved and restored around the fallback and the shading loop. */
struct cp_pass_live {
   struct cp_shader_binary *vs, *fs;
   /* The resolved vertex input, which is what the draw path reads. The
    * Gallium array is not saved: nothing during a fallback re-execution
    * looks at it, and leaving the live copy alone is what keeps the next
    * batch key correct. */
   struct cp_vertex_elem velem[16];
   uint64_t vb_base[16];
   unsigned num_vertex_buffers;
   unsigned num_vertex_elements, vertex_stride;
   unsigned num_vs_ubos, num_fs_ubos;
   struct cp_fs_batch fs_batch;
};

static void
cp_pass_live_save(struct cp_context *cp, struct cp_pass_live *lv)
{
   lv->vs = cp->vs_shader;
   lv->fs = cp->fs_shader;
   memcpy(lv->velem, cp->velem, sizeof(lv->velem));
   memcpy(lv->vb_base, cp->vb_base, sizeof(lv->vb_base));
   lv->num_vertex_buffers = cp->num_vertex_buffers;
   lv->num_vertex_elements = cp->num_vertex_elements;
   lv->vertex_stride = cp->vertex_stride;
   lv->num_vs_ubos = cp->num_vs_ubos;
   lv->num_fs_ubos = cp->num_fs_ubos;
   lv->fs_batch = cp->fs_batch;
}

static void
cp_pass_live_restore(struct cp_context *cp, const struct cp_pass_live *lv)
{
   cp->vs_shader = lv->vs;
   cp->fs_shader = lv->fs;
   memcpy(cp->velem, lv->velem, sizeof(lv->velem));
   memcpy(cp->vb_base, lv->vb_base, sizeof(lv->vb_base));
   cp->num_vertex_buffers = lv->num_vertex_buffers;
   cp->num_vertex_elements = lv->num_vertex_elements;
   cp->vertex_stride = lv->vertex_stride;
   cp->num_vs_ubos = lv->num_vs_ubos;
   cp->num_fs_ubos = lv->num_fs_ubos;
   cp->fs_batch = lv->fs_batch;
}

static void
cp_pass_seg_restore(struct cp_context *cp, const struct cp_pass_seg *sg)
{
   cp->vs_shader = sg->vs;
   cp->fs_shader = sg->fs;
   memcpy(cp->velem, sg->velem, sizeof(sg->velem));
   memcpy(cp->vb_base, sg->vb_base, sizeof(sg->vb_base));
   cp->num_vertex_buffers = sg->num_vertex_buffers;
   cp->num_vertex_elements = sg->num_vertex_elements;
   cp->vertex_stride = sg->vertex_stride;
   cp->num_vs_ubos = sg->num_vs_ubos;
   cp->num_fs_ubos = sg->num_fs_ubos;
}

/* The episode could not deliver; render every segment the classic way, in
 * submission order, from its snapshot. Rasterization is idempotent — the
 * abandoned lists were never read by anything that draws. */
static void
cp_pass_fallback(struct cp_context *cp, struct cp_pass_seg *segs,
                 unsigned nsegs)
{
   /* The abandoned episode's kernels may still be in flight on the side
    * streams, writing the shared lists the re-execution is about to clear. */
   cp_pass_join(cp, nsegs);

   struct cp_pass_live lv;
   cp_pass_live_save(cp, &lv);
   for (unsigned s = 0; s < nsegs; s++) {
      struct cp_pass_seg *sg = &segs[s];
      cp_pass_seg_restore(cp, sg);
      cp_draw_execute(cp, &sg->info, sg->drawid_offset, sg->draws, 1,
                      sg->ndraws, sg->vs_ubos, sg->fs_ubos, sg->draw_ids,
                      sg->instance_counts, sg->vb_bases, sg->scissors);
   }
   cp_pass_live_restore(cp, &lv);
}

static bool
cp_opaque_tile_visibility(struct cp_context *cp, struct cp_pass_seg *segs,
                          unsigned nsegs, unsigned w, unsigned h)
{
   struct cp_device *screen = cp->screen;
   if (!cp_debug->tiled_opaque || !screen->kernels.opaque_tile_count ||
       !screen->kernels.opaque_tile_fill ||
       !screen->kernels.opaque_tile_raster)
      return false;

   uint32_t tiles_x = DIV_ROUND_UP(w, CP_OPAQUE_TILE_SIZE);
   uint32_t tiles_y = DIV_ROUND_UP(h, CP_OPAQUE_TILE_SIZE);
   uint32_t ntiles = tiles_x * tiles_y;
   uint32_t nb1 = DIV_ROUND_UP(ntiles, CP_ABUF_SCAN_BLOCK);
   uint32_t nb2 = DIV_ROUND_UP(nb1, CP_ABUF_SCAN_BLOCK);
   uint32_t nb3 = DIV_ROUND_UP(nb2, CP_ABUF_SCAN_BLOCK);
   CUdeviceptr counts = cp_scratch_alloc_device(cp, (size_t)ntiles * 4);
   CUdeviceptr offsets = cp_scratch_alloc_device(cp, (size_t)ntiles * 4);
   CUdeviceptr cursors = cp_scratch_alloc_device(cp, (size_t)ntiles * 4);
   CUdeviceptr refs = cp_scratch_alloc_device(
      cp, (size_t)CP_MAX_OPAQUE_TILE_REFS * sizeof(struct cp_opaque_tile_ref));
   CUdeviceptr overflow = cp_scratch_alloc_device(cp, 4);
   CUdeviceptr s1 = cp_scratch_alloc_device(cp, (size_t)MAX2(nb1, 1u) * 4);
   CUdeviceptr s1x = cp_scratch_alloc_device(cp, (size_t)MAX2(nb1, 1u) * 4);
   CUdeviceptr s2 = cp_scratch_alloc_device(cp, (size_t)MAX2(nb2, 1u) * 4);
   CUdeviceptr s2x = cp_scratch_alloc_device(cp, (size_t)MAX2(nb2, 1u) * 4);
   CUdeviceptr s3 = cp_scratch_alloc_device(cp, (size_t)MAX2(nb3, 1u) * 4);
   if (!counts || !offsets || !cursors || !refs || !overflow || !s1 ||
       !s1x || !s2 || !s2x || !s3)
      return false;

   cuMemsetD32Async(counts, 0, ntiles, cp->stream);
   cuMemsetD32Async(cursors, 0, ntiles, cp->stream);
   cuMemsetD32Async(overflow, 0, 1, cp->stream);
   for (unsigned s = 0; s < nsegs; s++) {
      struct cp_opaque_tile_build_args args = {
         .rast = segs[s].rast,
         .tile_counts = counts,
         .tile_offsets = offsets,
         .tile_cursors = cursors,
         .tile_refs = refs,
         .overflow = overflow,
         .tiles_x = tiles_x,
         .tiles_y = tiles_y,
         .segment = s,
         .capacity = CP_MAX_OPAQUE_TILE_REFS,
      };
      void *params[] = { &args };
      CP_LAUNCH(screen->kernels.opaque_tile_count,
                MIN2(DIV_ROUND_UP(segs[s].rast_num_triangles, 256), 1024u),
                1, 1, 256, 1, 1, 0, cp->stream, params, NULL);
   }

   cp_abuf_scan_n(cp, screen, counts, offsets, s1, s1x, s2, s2x, s3,
                  ntiles, nb1, nb2, nb3, counts,
                  CP_MAX_OPAQUE_TILE_REFS, overflow);
   cuMemsetD32Async(cursors, 0, ntiles, cp->stream);
   for (unsigned s = 0; s < nsegs; s++) {
      struct cp_opaque_tile_build_args args = {
         .rast = segs[s].rast,
         .tile_counts = counts,
         .tile_offsets = offsets,
         .tile_cursors = cursors,
         .tile_refs = refs,
         .overflow = overflow,
         .tiles_x = tiles_x,
         .tiles_y = tiles_y,
         .segment = s,
         .capacity = CP_MAX_OPAQUE_TILE_REFS,
      };
      void *params[] = { &args };
      CP_LAUNCH(screen->kernels.opaque_tile_fill,
                MIN2(DIV_ROUND_UP(segs[s].rast_num_triangles, 256), 1024u),
                1, 1, 256, 1, 1, 0, cp->stream, params, NULL);
   }

   struct cp_rasterize_args rast_args[CP_PASS_MAX_SEGS];
   for (unsigned s = 0; s < nsegs; s++)
      rast_args[s] = segs[s].rast;
   CUdeviceptr rast_dev = cp_upload(cp, rast_args,
                                    (size_t)nsegs * sizeof(rast_args[0]));
   if (!rast_dev)
      return false;
   struct cp_opaque_tile_raster_args raster = {
      .rast_args = rast_dev,
      .tile_counts = counts,
      .tile_offsets = offsets,
      .tile_refs = refs,
      .overflow = overflow,
      .num_segments = nsegs,
      .tiles_x = tiles_x,
      .tiles_y = tiles_y,
      .width = w,
      .height = h,
   };
   void *raster_params[] = { &raster };
   CP_LAUNCH(screen->kernels.opaque_tile_raster, ntiles, 1, 1,
             256, 1, 1, 0, cp->stream, raster_params, NULL);

   /* The tiled kernel returns immediately when overflow is nonzero. Enqueue
    * the ordinary raster stages behind it with the inverse predicate, so the
    * GPU executes exactly one visibility path without a host readback. */
   for (unsigned s = 0; s < nsegs; s++) {
      struct cp_rasterize_args aa = segs[s].rast;
      struct cp_rast_queues queues = segs[s].queues;
      aa.path_flag = overflow;
      aa.path_value = 1;
      queues.mode = CP_QUEUE_FILL;
      cuMemsetD32Async(queues.nontrivial_count, 0, 1, cp->stream);
      cuMemsetD32Async(queues.huge_count, 0, 1, cp->stream);
      void *params[] = { &aa, &queues };
      CP_LAUNCH(screen->kernels.rasterize_stage1,
                DIV_ROUND_UP(segs[s].rast_num_triangles, 256), 1, 1,
                256, 1, 1, 0, cp->stream, params, NULL);
      CP_LAUNCH(screen->kernels.rasterize_stage2,
                CLAMP(DIV_ROUND_UP(segs[s].rast_num_triangles, 8), 1u, 512u),
                1, 1, 256, 1, 1, 0, cp->stream, params, NULL);
      CP_LAUNCH(screen->kernels.rasterize_stage3, 2048, 1, 1,
                64, 1, 1, 0, cp->stream, params, NULL);
   }

   if (cp_debug->tiled_opaque_census) {
      uint32_t host_overflow = 0;
      cuMemcpyDtoHAsync(&host_overflow, overflow, sizeof(host_overflow),
                        cp->stream);
      cuStreamSynchronize(cp->stream);
      fprintf(stderr, "cudapipe: opaque tiles %ux%u segments=%u overflow=%u\n",
              tiles_x, tiles_y, nsegs, host_overflow);
   }
   return true;
}

/* Defined with the rest of the census, below. */
static void cp_tile_census_visbuf(struct cp_context *cp,
                                  struct cp_pass_seg *segs, unsigned nsegs,
                                  unsigned w, unsigned h);
static void cp_tile_census_quads(struct cp_context *cp,
                                 struct cp_pass_seg *segs, unsigned nsegs,
                                 unsigned w, unsigned h);

static void
cp_opaque_finish(struct cp_context *cp)
{
   unsigned nsegs = cp->pass.nsegs;
   if (!nsegs)
      return;

   struct cp_pass_seg *segs = cp->pass_segs;
   unsigned w = cp->pass.w, h = cp->pass.h;
   cp->pass.nsegs = 0;
   cp->pass.next_prim = 0;
   cp->pass.opaque = false;

   if (cp_debug->tiled_opaque &&
       !cp_opaque_tile_visibility(cp, segs, nsegs, w, h)) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   cp_tile_census_visbuf(cp, segs, nsegs, w, h);

   uint8_t seg_group[CP_PASS_MAX_SEGS];
   unsigned group_first[CP_PASS_MAX_SEGS];
   unsigned ngroups = 0;
   for (unsigned s = 0; s < nsegs; s++) {
      unsigned group = ngroups;
      if (!cp_debug->no_seg_merge) {
         for (unsigned g = 0; g < ngroups; g++) {
            struct cp_pass_seg *first = &segs[group_first[g]];
            if (first->vs == segs[s].vs && first->fs == segs[s].fs &&
                first->num_fs_ubos == segs[s].num_fs_ubos &&
                first->info.mode == segs[s].info.mode &&
                first->ndraws == segs[s].ndraws &&
                !memcmp(first->fs_ubos, segs[s].fs_ubos,
                        (size_t)first->ndraws * CP_ARG_UBO_STRIDE *
                           sizeof(uint64_t))) {
               group = g;
               break;
            }
         }
      }
      if (group == ngroups)
         group_first[ngroups++] = s;
      seg_group[s] = (uint8_t)group;
   }

   if (!cp->pass_group_ubos) {
      cp->pass_group_ubos =
         malloc((size_t)CP_PASS_MAX_SEGS * CP_MAX_BATCH_DRAWS *
                CP_ARG_UBO_STRIDE * sizeof(uint64_t));
      if (!cp->pass_group_ubos) {
         cp_pass_fallback(cp, segs, nsegs);
         return;
      }
   }

   struct cp_seg_range ranges[CP_PASS_MAX_SEGS];
   unsigned group_range_base[CP_PASS_MAX_SEGS];
   unsigned group_range_count[CP_PASS_MAX_SEGS];
   unsigned nranges = 0;
   for (unsigned g = 0; g < ngroups; g++) {
      unsigned group_rows = 0;
      group_range_base[g] = nranges;
      for (unsigned s = 0; s < nsegs; s++) {
         if (seg_group[s] != g)
            continue;
         struct cp_pass_seg *seg = &segs[s];
         ranges[nranges++] = (struct cp_seg_range) {
            .positions = seg->rast.positions,
            .draw_slices = seg->slices_dev,
            .num_draw_slices = seg->ndraws,
            .prim_base = seg->prim_base,
            .prim_end = seg->prim_base + seg->prim_slots,
            .row_base = group_rows,
            .prim_shift = seg->prim_shift,
         };
         group_rows += seg->ndraws;
      }
      group_range_count[g] = nranges - group_range_base[g];
   }
   CUdeviceptr ranges_dev = cp_upload(cp, ranges,
                                      (size_t)nranges * sizeof(ranges[0]));
   if (!ranges_dev) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   const struct cp_fb_desc *fb = &cp->fb;
   void *color_data = fb->color;
   if (!color_data) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   struct cp_pass_live live;
   cp_pass_live_save(cp, &live);
   size_t shade_mark = cp->scratch.used;
   size_t shade_dmark = cp->dscratch.used;
   for (unsigned g = 0; g < ngroups; g++) {
      struct cp_pass_seg *seg = &segs[group_first[g]];
      unsigned group_rows = 0;
      for (unsigned s = 0; s < nsegs; s++) {
         if (seg_group[s] != g)
            continue;
         struct cp_pass_seg *member = &segs[s];
         memcpy(cp->pass_group_ubos +
                   (size_t)group_rows * CP_ARG_UBO_STRIDE,
                member->fs_ubos,
                (size_t)member->ndraws * CP_ARG_UBO_STRIDE *
                   sizeof(uint64_t));
         group_rows += member->ndraws;
      }
      cp->scratch.used = shade_mark;
      cp->dscratch.used = shade_dmark;
      cp_pass_seg_restore(cp, seg);
      cp->fs_batch.ubos = cp->pass_group_ubos;
      cp->fs_batch.ndraws = group_rows;
      cp->fs_batch.slices = seg->slices_dev;
      cp->fs_batch.prim_shift = seg->prim_shift;
      cp_shade_fragments(cp, &seg->info, cp->visbuf, seg->rast.positions,
                         seg->rast.positions, seg->num_triangles, w, h,
                         color_data, seg->rast.vp_scale_x,
                         seg->rast.vp_scale_y, seg->rast.vp_trans_x,
                         seg->rast.vp_trans_y, 0, 0, 0,
                         ranges_dev + (size_t)group_range_base[g] *
                            sizeof(ranges[0]),
                         group_range_count[g]);
   }
   cp_pass_live_restore(cp, &live);
}

/*
 * Tile-bin shader census (CUDAPIPE_TILE_CENSUS=<tile edge>).
 *
 * The question it answers decides whether a tile-local rasterizer is
 * buildable here: when a tile owns its pixels and walks its primitives in
 * submission order, how many distinct fragment shaders must it be able to
 * call, and do they arrive in contiguous runs? One shader per tile means the
 * tile kernel can be specialized and statically linked, exactly as the
 * sampler already is. Many interleaved shaders mean only an indirect call or
 * a switch over every shader in the pass would keep the order, and both give
 * back what tiling is for.
 *
 * The accumulation unit is the framebuffer bind, because that is the longest
 * interval a tile could hold colour and depth on chip. Both sources of
 * shaded coverage feed it: the A-buffer's finished quad stream for blended
 * draws, and the shared visibility buffer's winners for opaque ones. A tile
 * only has to be able to call the shader of a fragment it actually shades,
 * which is why the opaque side counts winners rather than every primitive
 * that touched the tile.
 *
 * Ordering is keyed on a pass-global draw sequence, not on primitive ids,
 * which restart at every episode and cannot be compared across one.
 *
 * It changes no rendering, and when the flag is unset it is one branch.
 */
static void
cp_tile_census_reduce_pass(struct cp_context *cp);

/* The distinct fragment shader a tile kernel would have to dispatch on is
 * the compiled binary, not the shading group, so this dedups binaries over
 * the whole bind. */
static unsigned
cp_tile_census_shader_index(struct cp_context *cp, struct cp_shader_binary *fs)
{
   for (unsigned i = 0; i < cp->tile_census_nfs; i++)
      if (cp->tile_census_fs[i] == fs)
         return i;
   if (cp->tile_census_nfs == CP_TILE_CENSUS_MAX_SHADERS)
      return CP_TILE_CENSUS_MAX_SHADERS - 1;
   cp->tile_census_fs[cp->tile_census_nfs] = fs;
   return cp->tile_census_nfs++;
}

/* Open the accumulation for this bind, sizing and clearing the per-tile
 * arrays. Returns false when the census cannot run at all. */
static bool
cp_tile_census_begin(struct cp_context *cp, unsigned w, unsigned h)
{
   struct cp_device *screen = cp->screen;
   unsigned tile = cp_debug->tile_census;

   if (!tile || !w || !h || !screen->kernels.tile_census_mark ||
       !screen->kernels.tile_census_reduce)
      return false;

   if (cp->tile_census_open) {
      /* A framebuffer of a different size inside one bind is not something
       * the tile grid can follow; close the run and start another. */
      if (w == cp->tile_census_w && h == cp->tile_census_h)
         return true;
      cp_tile_census_reduce_pass(cp);
   }

   unsigned tiles_x = (w + tile - 1) / tile;
   unsigned tiles_y = (h + tile - 1) / tile;
   unsigned ntiles = tiles_x * tiles_y;
   if (!ntiles)
      return false;

   if (!cp->tile_census_hist) {
      if (cuMemAllocManaged(&cp->tile_census_hist,
                            CP_TILE_CENSUS_WORDS * sizeof(uint64_t),
                            CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS)
         return false;
      memset((void *)(uintptr_t)cp->tile_census_hist, 0,
             CP_TILE_CENSUS_WORDS * sizeof(uint64_t));
   }

   if (ntiles > cp->tile_census_alloc) {
      if (cp->tile_census_mask) {
         cuMemFree(cp->tile_census_mask);
         cuMemFree(cp->tile_census_quads);
         cuMemFree(cp->tile_census_refs);
         cuMemFree(cp->tile_census_smin);
         cuMemFree(cp->tile_census_smax);
         cp->tile_census_mask = 0;
      }
      size_t per = (size_t)ntiles * CP_TILE_CENSUS_MAX_SHADERS * 4;
      if (cuMemAlloc(&cp->tile_census_mask, (size_t)ntiles * 8) ||
          cuMemAlloc(&cp->tile_census_quads, (size_t)ntiles * 4) ||
          cuMemAlloc(&cp->tile_census_refs, (size_t)ntiles * 4) ||
          cuMemAlloc(&cp->tile_census_smin, per) ||
          cuMemAlloc(&cp->tile_census_smax, per)) {
         cp->tile_census_alloc = 0;
         return false;
      }
      cp->tile_census_alloc = ntiles;
   }

   cuMemsetD32Async(cp->tile_census_mask, 0, (size_t)ntiles * 2, cp->stream);
   cuMemsetD32Async(cp->tile_census_quads, 0, ntiles, cp->stream);
   cuMemsetD32Async(cp->tile_census_refs, 0, ntiles, cp->stream);
   cuMemsetD32Async(cp->tile_census_smin, 0xFFFFFFFFu,
                    (size_t)ntiles * CP_TILE_CENSUS_MAX_SHADERS, cp->stream);
   cuMemsetD32Async(cp->tile_census_smax, 0,
                    (size_t)ntiles * CP_TILE_CENSUS_MAX_SHADERS, cp->stream);

   cp->tile_census_tiles_x = tiles_x;
   cp->tile_census_tiles_y = tiles_y;
   cp->tile_census_w = w;
   cp->tile_census_h = h;
   cp->tile_census_seq = 0;
   cp->tile_census_nfs = 0;
   cp->tile_census_open = true;
   cp->tile_census_cut = false;
   return true;
}

/* Fill in everything both sources share, and take this run's slice of the
 * pass-global draw order. */
static bool
cp_tile_census_common(struct cp_context *cp, struct cp_pass_seg *segs,
                      unsigned nsegs, struct cp_tile_census_args *ca)
{
   uint32_t bases[CP_PASS_MAX_SEGS], seqs[CP_PASS_MAX_SEGS];
   uint8_t shaders[CP_PASS_MAX_SEGS];
   unsigned nshaders = 0;

   for (unsigned s = 0; s < nsegs; s++) {
      bases[s] = segs[s].prim_base;
      seqs[s] = cp->tile_census_seq + s;
      unsigned i = cp_tile_census_shader_index(cp, segs[s].fs);
      shaders[s] = (uint8_t)i;
      if (i + 1 > nshaders)
         nshaders = i + 1;
   }
   cp->tile_census_seq += nsegs;

   CUdeviceptr bases_dev = cp_upload(cp, bases, (size_t)nsegs * 4);
   CUdeviceptr seq_dev = cp_upload(cp, seqs, (size_t)nsegs * 4);
   CUdeviceptr shader_dev = cp_upload(cp, shaders, nsegs);
   if (!bases_dev || !seq_dev || !shader_dev)
      return false;

   *ca = (struct cp_tile_census_args) {
      .seg_prim_base = bases_dev,
      .seg_seq = seq_dev,
      .seg_shader = shader_dev,
      .tile_mask = cp->tile_census_mask,
      .tile_quads = cp->tile_census_quads,
      .tile_smin = cp->tile_census_smin,
      .tile_smax = cp->tile_census_smax,
      .tile_refs = cp->tile_census_refs,
      .hist = cp->tile_census_hist,
      .nsegs = nsegs,
      .tile = cp_debug->tile_census,
      .tiles_x = cp->tile_census_tiles_x,
      .tiles_y = cp->tile_census_tiles_y,
      .nshaders = nshaders,
   };

   cp->tile_census_marks++;
   cp->tile_census_shaders += cp->tile_census_nfs;
   for (unsigned s = 0; s < nsegs; s++)
      cp->tile_census_marked_draws += segs[s].ndraws;

   /* How large the bin itself would be, which the shaded counts cannot say. */
   if (cp->screen->kernels.tile_census_refs) {
      for (unsigned s = 0; s < nsegs; s++) {
         unsigned n = segs[s].rast_num_triangles;
         if (!n)
            continue;
         void *p[] = { &segs[s].rast, ca };
         CP_LAUNCH(cp->screen->kernels.tile_census_refs,
                   MIN2((n + 255) / 256, 1024u), 1, 1, 256, 1, 1, 0,
                   cp->stream, p, NULL);
      }
   }
   return true;
}

/* Blended source: the A-buffer's finished quad stream. */
static void
cp_tile_census_quads(struct cp_context *cp, struct cp_pass_seg *segs,
                     unsigned nsegs, unsigned w, unsigned h)
{
   struct cp_device *screen = cp->screen;
   struct cp_abuf *ab = &cp_abuf;
   struct cp_tile_census_args ca;

   if (!ab->quad_prim || !ab->quad_block ||
       !cp_tile_census_begin(cp, w, h) ||
       !cp_tile_census_common(cp, segs, nsegs, &ca))
      return;

   ca.quad_prim = ab->quad_prim;
   ca.quad_block = ab->quad_block;
   ca.num_quads_dev = ab->bsum3;
   ca.num_quads = (uint32_t)ab->quad_capacity;
   ca.quad_width = (w + 1) / 2;

   void *p[] = { &ca };
   CP_LAUNCH(screen->kernels.tile_census_mark,
             MIN2(((unsigned)ab->quad_capacity + 255) / 256, 1024u), 1, 1,
             256, 1, 1, 0, cp->stream, p, NULL);
}

/* Opaque source: the shared visibility buffer's winners. */
static void
cp_tile_census_visbuf(struct cp_context *cp, struct cp_pass_seg *segs,
                      unsigned nsegs, unsigned w, unsigned h)
{
   struct cp_device *screen = cp->screen;
   struct cp_tile_census_args ca;

   if (!cp->visbuf || !screen->kernels.tile_census_mark_vis ||
       !cp_tile_census_begin(cp, w, h) ||
       !cp_tile_census_common(cp, segs, nsegs, &ca))
      return;

   ca.visbuf = cp->visbuf;
   ca.width = w;
   ca.height = h;

   void *p[] = { &ca };
   CP_LAUNCH(screen->kernels.tile_census_mark_vis, (w + 15) / 16,
             (h + 15) / 16, 1, 16, 16, 1, 0, cp->stream, p, NULL);
}

/* Close the bind: reduce every tile into the histogram, and report. */
static void
cp_tile_census_reduce_pass(struct cp_context *cp)
{
   struct cp_device *screen = cp->screen;

   if (!cp->tile_census_open)
      return;
   cp->tile_census_open = false;
   cp->tile_census_passes++;
   if (cp->tile_census_cut)
      cp->tile_census_binds_cut++;

   unsigned ntiles = cp->tile_census_tiles_x * cp->tile_census_tiles_y;
   struct cp_tile_census_args ca = {
      .tile_mask = cp->tile_census_mask,
      .tile_quads = cp->tile_census_quads,
      .tile_smin = cp->tile_census_smin,
      .tile_smax = cp->tile_census_smax,
      .tile_refs = cp->tile_census_refs,
      .hist = cp->tile_census_hist,
      .tiles_x = cp->tile_census_tiles_x,
      .tiles_y = cp->tile_census_tiles_y,
   };
   {
      void *p[] = { &ca };
      CP_LAUNCH(screen->kernels.tile_census_reduce, (ntiles + 127) / 128, 1, 1,
                128, 1, 1, 0, cp->stream, p, NULL);
   }

   if (cp->tile_census_passes % cp_debug->tile_census_every)
      return;

   cuStreamSynchronize(cp->stream);
   const uint64_t *hist = (const uint64_t *)(uintptr_t)cp->tile_census_hist;
   uint64_t tiles_tot = 0, frags_tot = 0, split_tiles = 0, split_frags = 0;
   for (unsigned b = 1; b < CP_TILE_CENSUS_BINS; b++) {
      tiles_tot += hist[b * 4 + 0] + hist[b * 4 + 1];
      frags_tot += hist[b * 4 + 2] + hist[b * 4 + 3];
      split_tiles += hist[b * 4 + 0];
      split_frags += hist[b * 4 + 2];
   }
   if (!tiles_tot)
      return;

   fprintf(stderr,
           "tilecensus tile=%u binds=%llu marked_runs=%llu marked_draws=%llu "
           "unmarked_draws=%llu (%.2f%%) shaders/bind=%.2f "
           "nonempty_tiles=%llu shaded=%llu "
           "splittable_tiles=%.1f%% splittable_shaded=%.1f%%\n",
           cp_debug->tile_census,
           (unsigned long long)cp->tile_census_passes,
           (unsigned long long)cp->tile_census_marks,
           (unsigned long long)cp->tile_census_marked_draws,
           (unsigned long long)cp->tile_census_solo,
           100.0 * (double)cp->tile_census_solo /
              (double)MAX2(cp->tile_census_solo + cp->tile_census_marked_draws,
                           (uint64_t)1),
           (double)cp->tile_census_shaders / (double)cp->tile_census_passes,
           (unsigned long long)tiles_tot, (unsigned long long)frags_tot,
           100.0 * (double)split_tiles / (double)tiles_tot,
           100.0 * (double)split_frags / (double)frags_tot);

   /* (1) the bin's size, (2) how evenly the shading falls, (3) whether the
    * bind really was the residency interval. */
   {
      const uint64_t *gl = hist + CP_TILE_CENSUS_GLOBALS;
      double mean_refs = gl[4] ? (double)gl[2] / (double)gl[4] : 0.0;
      double mean_sh = gl[5] ? (double)gl[3] / (double)gl[5] : 0.0;
      fprintf(stderr,
              "tilecensus  bin: refs=%llu over %llu tiles, mean=%.1f max=%llu "
              "-> %.1f MB/bind at 8 B/ref\n",
              (unsigned long long)gl[2], (unsigned long long)gl[4], mean_refs,
              (unsigned long long)gl[0],
              (double)gl[2] * 8.0 / 1048576.0 /
                 (double)MAX2(cp->tile_census_passes, (uint64_t)1));
      fprintf(stderr,
              "tilecensus  load: shaded mean=%.1f max=%llu per tile, "
              "imbalance=%.1fx\n", mean_sh, (unsigned long long)gl[1],
              mean_sh > 0.0 ? (double)gl[1] / mean_sh : 0.0);
      for (int which = 0; which < 2; which++) {
         const uint64_t *h = hist + (which ? CP_TILE_CENSUS_REFS
                                           : CP_TILE_CENSUS_SHADED);
         uint64_t tot = 0;
         for (unsigned b = 0; b < CP_TILE_CENSUS_LOG; b++)
            tot += h[b];
         if (!tot)
            continue;
         uint64_t run = 0;
         unsigned p50 = 0, p90 = 0, p99 = 0;
         for (unsigned b = 0; b < CP_TILE_CENSUS_LOG; b++) {
            run += h[b];
            if (!p50 && run * 2 >= tot)
               p50 = b;
            if (!p90 && run * 10 >= tot * 9)
               p90 = b;
            if (!p99 && run * 100 >= tot * 99)
               p99 = b;
         }
         fprintf(stderr,
                 "tilecensus  %s per tile: p50<%u p90<%u p99<%u (powers of 2)\n",
                 which ? "refs " : "shaded", 1u << p50, 1u << p90, 1u << p99);
      }
      fprintf(stderr,
              "tilecensus  residency: %llu of %llu binds interrupted (%.1f%%) "
              "map=%llu copy=%llu flush=%llu compute=%llu, draws/bind=%.1f\n",
              (unsigned long long)cp->tile_census_binds_cut,
              (unsigned long long)cp->tile_census_passes,
              100.0 * (double)cp->tile_census_binds_cut /
                 (double)cp->tile_census_passes,
              (unsigned long long)cp->tile_census_cut_map,
              (unsigned long long)cp->tile_census_cut_copy,
              (unsigned long long)cp->tile_census_cut_flush,
              (unsigned long long)cp->tile_census_cut_compute,
              (double)(cp->tile_census_marked_draws + cp->tile_census_solo) /
                 (double)cp->tile_census_passes);
   }
   for (unsigned b = 1; b < CP_TILE_CENSUS_BINS; b++) {
      uint64_t t = hist[b * 4 + 0] + hist[b * 4 + 1];
      uint64_t q = hist[b * 4 + 2] + hist[b * 4 + 3];
      if (!t)
         continue;
      fprintf(stderr,
              "tilecensus   shaders=%2u tiles=%10llu (%5.1f%%) "
              "shaded=%12llu (%5.1f%%) disjoint_tiles=%5.1f%%\n",
              b, (unsigned long long)t, 100.0 * (double)t / (double)tiles_tot,
              (unsigned long long)q, 100.0 * (double)q / (double)frags_tot,
              100.0 * (double)hist[b * 4 + 0] / (double)t);
   }
}

/*
 * Was the bind really the interval a tile could have stayed resident? These
 * are the things that read or write the attachment while one is open, which
 * would force a real tiler to flush and reload the tile mid-pass. Recorded
 * by kind rather than lumped, because they have different answers: a copy
 * can often be deferred, a host map cannot.
 */
void
cp_tile_census_cut(struct cp_context *cp, enum cp_tile_census_cut_kind kind)
{
   if (!cp_debug->tile_census || !cp->tile_census_open)
      return;
   switch (kind) {
   case CP_TILE_CUT_MAP:     cp->tile_census_cut_map++;     break;
   case CP_TILE_CUT_COPY:    cp->tile_census_cut_copy++;    break;
   case CP_TILE_CUT_FLUSH:   cp->tile_census_cut_flush++;   break;
   case CP_TILE_CUT_COMPUTE: cp->tile_census_cut_compute++; break;
   }
   cp->tile_census_cut = true;
}

void
cp_tile_census_end_pass(struct cp_context *cp)
{
   if (cp_debug->tile_census)
      cp_tile_census_reduce_pass(cp);
}

static bool
cp_pass_finish_bounded_groups(struct cp_context *cp,
                              struct cp_pass_seg *segs,
                              unsigned nsegs, unsigned w, unsigned h)
{
   struct cp_abuf *ab = &cp_abuf;
   size_t quad_bound = 0;

   if (cp_debug->unsafe_no_overflow) {
      quad_bound = MIN3(ab->quad_capacity, ab->capacity / 4u,
                        (size_t)CP_ABUF_MAX_SHADE_SLOTS / 4u);
   } else {
      for (unsigned s = 0; s < nsegs; s++) {
         if (segs[s].rast_num_triangles > SIZE_MAX / ab->nblocks ||
             quad_bound > SIZE_MAX -
                (size_t)segs[s].rast_num_triangles * ab->nblocks)
            return false;
         quad_bound += (size_t)segs[s].rast_num_triangles * ab->nblocks;
      }
   }

   if (!quad_bound || quad_bound > ab->quad_capacity ||
       quad_bound > ab->capacity / 4u ||
       quad_bound > CP_ABUF_MAX_SHADE_SLOTS / 4u)
      return false;

   uint8_t seg_group[CP_PASS_MAX_SEGS];
   unsigned group_first[CP_PASS_MAX_SEGS];
   unsigned ngroups = 0;
   for (unsigned s = 0; s < nsegs; s++) {
      unsigned group = ngroups;
      for (unsigned i = 0; i < ngroups && !cp_debug->no_seg_merge &&
                           !cp_debug->unsafe_no_overflow; i++) {
         struct cp_pass_seg *first = &segs[group_first[i]];
         if (first->vs == segs[s].vs && first->fs == segs[s].fs &&
             first->num_fs_ubos == segs[s].num_fs_ubos &&
             first->info.mode == segs[s].info.mode) {
            group = i;
            break;
         }
      }
      if (group == ngroups)
         group_first[ngroups++] = s;
      seg_group[s] = group;
   }

   bool compact = cp_debug->unsafe_no_overflow;
   CUdeviceptr quad_seg = 0, quad_dense = 0, grouped = 0;
   CUdeviceptr group_base_dev = 0, group_counts_dev = 0;
   if (ngroups > 1 || compact) {
      quad_seg = cp_scratch_alloc_device(cp, quad_bound);
      uint32_t seg_prims[CP_PASS_MAX_SEGS];
      for (unsigned s = 0; s < nsegs; s++)
         seg_prims[s] = segs[s].prim_base;
      CUdeviceptr seg_prims_dev = cp_upload(cp, seg_prims, (size_t)nsegs * 4);
      if (!quad_seg || !seg_prims_dev)
         return false;
      struct cp_abuf_seg_args bucket = {
         .quad_prim = ab->quad_prim,
         .seg_prim_base = seg_prims_dev,
         .seg_counts = compact ? ab->seg_counts : 0,
         .quad_seg = quad_seg,
         .num_quads_dev = ab->bsum3,
         .nsegs = nsegs,
         .num_quads = quad_bound,
         .warp_aggregate = false,
      };
      if (compact)
         cuMemsetD32Async(ab->seg_counts, 0, CP_PASS_MAX_SEGS, cp->stream);
      void *bucket_params[] = { &bucket };
      CP_LAUNCH(cp->screen->kernels.abuf_seg_count,
                MIN2(((unsigned)quad_bound + 255) / 256, 1024u),
                1, 1, 256, 1, 1, 0, cp->stream, bucket_params, NULL);

      if (compact) {
         CUdeviceptr seg_group_dev = cp_upload(cp, seg_group, nsegs);
         CUdeviceptr seg_base_dev =
            cp_scratch_alloc_device(cp, (size_t)nsegs * 4);
         group_base_dev = cp_scratch_alloc_device(cp, (size_t)ngroups * 4);
         group_counts_dev = cp_scratch_alloc_device(cp, (size_t)ngroups * 4);
         CUdeviceptr seg_cursor =
            cp_scratch_alloc_device(cp, (size_t)nsegs * 4);
         grouped = cp_scratch_alloc_device(cp, quad_bound * 4);
         quad_dense = cp_scratch_alloc_device(cp, quad_bound * 4);
         if (!seg_group_dev || !seg_base_dev || !group_base_dev ||
             !group_counts_dev || !seg_cursor || !grouped || !quad_dense)
            return false;

         struct cp_abuf_seg_prefix_args prefix = {
            .seg_counts = ab->seg_counts,
            .seg_group = seg_group_dev,
            .seg_base = seg_base_dev,
            .group_base = group_base_dev,
            .group_counts = group_counts_dev,
            .nsegs = nsegs,
            .ngroups = ngroups,
         };
         void *prefix_params[] = { &prefix };
         CP_LAUNCH(cp->screen->kernels.abuf_seg_prefix,
                   1, 1, 1, 1, 1, 1, 0, cp->stream, prefix_params, NULL);

         cuMemsetD32Async(seg_cursor, 0, nsegs, cp->stream);
         bucket.seg_cursor = seg_cursor;
         bucket.seg_base = seg_base_dev;
         bucket.grouped = grouped;
         bucket.quad_dense = quad_dense;
         bucket.seg_group = seg_group_dev;
         bucket.group_base = group_base_dev;
         void *scatter_params[] = { &bucket };
         CP_LAUNCH(cp->screen->kernels.abuf_seg_scatter,
                   ((unsigned)quad_bound + 255) / 256,
                   1, 1, 256, 1, 1, 0, cp->stream, scatter_params, NULL);
      }
   }

   struct cp_seg_range ranges[CP_PASS_MAX_SEGS];
   struct cp_seg_desc descs[CP_PASS_MAX_SEGS] = {0};
   struct cp_abuf_seg_shade group_shades[CP_PASS_MAX_SEGS] = {0};
   struct cp_pass_live live;
   cp_pass_live_save(cp, &live);
   bool shaded = true;
   for (unsigned group = 0; group < ngroups && shaded; group++) {
      struct cp_pass_seg *first = &segs[group_first[group]];
      bool need_rows = first->fs->reads_const_bufs;
      if (need_rows && !cp->pass_group_ubos) {
         cp->pass_group_ubos =
            malloc((size_t)CP_PASS_MAX_SEGS * CP_MAX_BATCH_DRAWS *
                   CP_ARG_UBO_STRIDE * sizeof(uint64_t));
         if (!cp->pass_group_ubos) {
            shaded = false;
            break;
         }
      }

      uint32_t rows = 0;
      unsigned nranges = 0;
      for (unsigned s = 0; s < nsegs; s++) {
         if (seg_group[s] != group)
            continue;
         struct cp_pass_seg *seg = &segs[s];
         ranges[nranges++] = (struct cp_seg_range) {
            .positions = seg->rast.positions,
            .draw_slices = seg->slices_dev,
            .num_draw_slices = seg->ndraws,
            .prim_base = seg->prim_base,
            .prim_end = seg->prim_base + seg->prim_slots,
            .row_base = rows,
            .prim_shift = seg->prim_shift,
         };
         if (need_rows)
            memcpy(cp->pass_group_ubos + (size_t)rows * CP_ARG_UBO_STRIDE,
                   seg->fs_ubos,
                   (size_t)seg->ndraws * CP_ARG_UBO_STRIDE * sizeof(uint64_t));
         rows += seg->ndraws;
      }
      CUdeviceptr ranges_dev =
         cp_upload(cp, ranges, (size_t)nranges * sizeof(ranges[0]));
      if (!ranges_dev) {
         shaded = false;
         break;
      }

      cp->vs_shader = first->vs;
      cp->fs_shader = first->fs;
      cp->num_fs_ubos = first->num_fs_ubos;
      cp->fs_batch.ubos = need_rows ? cp->pass_group_ubos : first->fs_ubos;
      cp->fs_batch.ndraws = rows;
      cp->fs_batch.slices = first->slices_dev;
      cp->fs_batch.prim_shift = first->prim_shift;
      struct cp_abuf_seg_shade shade = {
         .quad_list = compact ? grouped : 0,
         .quad_list_base_dev = compact ? group_base_dev + group * 4 : 0,
         .num_quads_dev = compact ? group_counts_dev + group * 4 : 0,
         .prim_base = first->prim_base,
         .ranges = ranges_dev,
         .num_ranges = nranges,
      };
      float ti, ts, tc;
      shaded = cp_abuf_shade(
         cp, &first->info, ab, first->rast.positions, first->rast.positions,
         w, h, first->rast.vp_scale_x, first->rast.vp_scale_y,
         first->rast.vp_trans_x, first->rast.vp_trans_y,
         (uint32_t)quad_bound, 0, false, NULL, false, &ti, &ts, &tc, &shade);
      if (!shaded)
         break;
      group_shades[group] = shade;
      for (unsigned s = 0; s < nsegs; s++) {
         if (seg_group[s] != group)
            continue;
         descs[s] = (struct cp_seg_desc) {
            .fs_out = shade.fs_out,
            .coverage = shade.coverage,
            .discard = shade.discard,
            .fs_out_stride = shade.fs_out_stride,
            .num_slots = shade.num_slots,
            .global_slots = !compact,
         };
      }
   }
   cp_pass_live_restore(cp, &live);
   if (!shaded)
      return false;

   const struct cp_fb_desc *fb = &cp->fb;
   void *color_data = fb->color;
   CUdeviceptr descs_dev = ngroups > 1
      ? cp_upload(cp, descs, (size_t)nsegs * sizeof(descs[0])) : 0;
   if (!color_data || (ngroups > 1 && !descs_dev))
      return false;

   struct cp_abuf_seg_shade *direct = &group_shades[0];
   struct cp_abuf_composite_args ca = {
      .offsets = ab->offsets,
      .counts = ab->counts,
      .shade_slot = ab->shade_slot,
      .fs_out = ngroups == 1 ? direct->fs_out : 0,
      .coverage = ngroups == 1 ? direct->coverage : 0,
      .discard_mask = ngroups == 1 ? direct->discard : 0,
      .color_out = (uint64_t)(uintptr_t)color_data,
      .list = ab->clist,
      .list_count = ab->clist_count,
      .fs_out_stride = ngroups == 1 ? direct->fs_out_stride : 0,
      .num_slots = ngroups == 1 ? direct->num_slots : 0,
      .capacity = ab->capacity,
      .color_encoding = (uint32_t)MAX2(
         fb->color_encoding, 0),
      .max_layers = cp_abuf.max_layers,
      .blend = cp_blend_desc_for(cp),
      .quad_seg = ngroups > 1 ? quad_seg : 0,
      .quad_dense = ngroups > 1 ? quad_dense : 0,
      .seg_desc = ngroups > 1 ? descs_dev : 0,
   };
   void *params[] = { &ca };
   CUresult err = cuLaunchKernel(cp->screen->kernels.abuf_composite,
                                 ((size_t)w * h + 255) / 256, 1, 1,
                                 256, 1, 1, 0, cp->stream, params, NULL);
   if (err != CUDA_SUCCESS)
      fprintf(stderr, "abuffer: bounded grouped composite launch failed "
              "(%d)\n", err);
   return err == CUDA_SUCCESS;
}

void
cp_pass_finish(struct cp_context *cp)
{
   struct cp_device *screen = cp->screen;
   struct cp_abuf *ab = &cp_abuf;
   unsigned nsegs = cp->pass.nsegs;

   if (!nsegs)
      return;
   if (cp->pass.opaque) {
      cp_opaque_finish(cp);
      return;
   }

   /* Cleared first: nothing below may see the episode as still open. */
   cp->pass.nsegs = 0;
   cp->pass.next_prim = 0;

   struct cp_pass_seg *segs = cp->pass_segs;
   unsigned w = cp->pass.w, h = cp->pass.h;
   size_t n = (size_t)w * h;
   bool failed = false;

   /*
    * The fallback exists for overflow and allocation failure, which the
    * sample set never reaches and the captures reach only on the standalone
    * A-buffer path -- so the episode fallback is code the gate cannot
    * exercise. This makes it reachable on purpose: every episode takes it,
    * and the output must be identical, because re-executing the segments
    * classically is defined to produce what the episode would have.
    */
   if (cp_debug->force_pass_fallback) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   cuCtxSetCurrent(screen->cuda_ctx);
   CP_NVTX_SCOPEF("episode %u segs", nsegs);

   /* The segments' count phases ran on the side streams; the scan reads
    * across all of them. */
   cp_pass_join(cp, nsegs);

   /* --- scan the accumulated counts, clamp the runs to the array --- */
   cp_abuf_scan(cp, screen, ab, (unsigned)n);

   /* --- fill --- */
   CUstream pass_main = cp->stream;
   cuMemsetD32Async(ab->cursor, 0, n, cp->stream);
   if (ab->recs && screen->kernels.abuf_fill_recs) {
      /* Single-pass build: every segment's count already appended its
       * records (ab->recs cannot change mid-episode — a pending growth
       * refuses the append), so the whole episode's fill is one linear
       * replay on the main stream, and the fan-out/join the per-segment
       * relaunches needed disappears with them. */
      void *fp[] = { &ab->recs, &ab->rec_cursor, &ab->capacity, &ab->frags,
                     &ab->offsets, &ab->counts, &ab->cursor, &ab->overflow,
                     &ab->capacity };
      CP_LAUNCH(screen->kernels.abuf_fill_recs, 2048, 1, 1, 256, 1, 1,
                     0, cp->stream, fp, NULL);
   } else {
      /* One relaunch per segment from its saved arguments, fanned back out
       * over the side streams behind the scan. */
      cp_pass_broadcast(cp, nsegs);
      for (unsigned s = 0; s < nsegs; s++) {
         struct cp_pass_seg *sg = &segs[s];
         if (cp->seg_streams[0])
            cp->stream = cp_pass_seg_stream(cp, s);
         struct cp_rasterize_args aa = sg->rast;
         aa.abuf_frags = ab->frags;
         aa.abuf_capacity = ab->capacity;
         aa.abuf_mode = CP_ABUF_FILL;
         struct cp_rast_queues q = sg->queues;
         q.mode = CP_QUEUE_FILL;
         /* The segment's own queue set, saved with its arguments. */
         cuMemsetD32Async(q.nontrivial_count, 0, 2, cp->stream);
         void *ap[] = { &aa, &q };
         CP_LAUNCH(screen->kernels.rasterize_stage1_abuf,
                        (sg->rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, ap, NULL);
         CP_LAUNCH(screen->kernels.rasterize_stage2_abuf,
                        CLAMP((sg->rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                        256, 1, 1, 0, cp->stream, ap, NULL);
         CP_LAUNCH(screen->kernels.rasterize_stage3_abuf,
                        CLAMP(sg->rast_num_triangles * 8, 512u, 2048u), 1, 1,
                        64, 1, 1, 0, cp->stream, ap, NULL);
      }
      cp->stream = pass_main;
      cp_pass_join(cp, nsegs);
   }

   /* --- sort, both worklists --- */
   {
      unsigned nn = (unsigned)n;
      unsigned max_short = cp_debug->abuf_short_sort_max;
      unsigned min_long = cp_debug->no_abuf_short_sort ? 2 : max_short + 1;
      if (!cp_debug->no_abuf_short_sort) {
         cuMemsetD32Async(ab->clist_count, 0, 1, cp->stream);
         cuMemsetD32Async(ab->list_count, 0, 1, cp->stream);
         void *ssp[] = { &ab->frags, &ab->offsets, &ab->counts, &nn,
                         &max_short, &ab->clist, &ab->clist_count,
                         &min_long, &ab->list, &ab->list_count };
         CP_LAUNCH(screen->kernels.abuf_sort_short,
                        MIN2((nn + 255) / 256, 4096u), 1, 1, 256, 1, 1,
                        0, cp->stream, ssp, NULL);
      } else {
         cuMemsetD32Async(ab->list_count, 0, 1, cp->stream);
         void *wp[] = { &ab->counts, &nn, &min_long, &ab->list,
                        &ab->list_count };
         CP_LAUNCH(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, wp, NULL);
      }
      void *sp[] = { &ab->frags, &ab->offsets, &ab->counts, &ab->list,
                     &ab->list_count, &ab->long_runs };
      CP_LAUNCH(screen->kernels.abuf_sort, 4096, 1, 1, 256, 1, 1,
                     0, cp->stream, sp, NULL);
      if (cp_debug->no_abuf_short_sort) {
         unsigned min1 = 1;
         cuMemsetD32Async(ab->clist_count, 0, 1, cp->stream);
         void *cw[] = { &ab->counts, &nn, &min1, &ab->clist,
                        &ab->clist_count };
         CP_LAUNCH(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, cw, NULL);
      }
   }

   /* --- the quad stream --- */
   unsigned nblocks = ab->nblocks, qw = ab->quad_width;
   cuMemsetD32Async(ab->blk_list_count, 0, 1, cp->stream);
   {
      void *p[] = { &ab->counts, &w, &h, &qw, &nblocks, &ab->blk_list,
                    &ab->blk_list_count };
      CP_LAUNCH(screen->kernels.abuf_block_worklist,
                     (nblocks + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, p, NULL);
   }
   cuMemsetD32Async(ab->blk_counts, 0, nblocks, cp->stream);
   cuMemsetD32Async(ab->bsum3, 0, 2, cp->stream);
   cuMemsetD32Async(ab->dbg, 0, CP_ABUF_DBG_COUNTERS, cp->stream);
   {
      void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                    &ab->blk_list, &ab->blk_list_count, &ab->blk_counts };
      CP_LAUNCH(screen->kernels.abuf_quad_count, 1024, 1, 1, 32, 1, 1,
                     0, cp->stream, p, NULL);
   }
   cp_abuf_scan_n(cp, screen, ab->blk_counts, ab->blk_offsets, ab->bsum1,
                  ab->bsum1x, ab->bsum2, ab->bsum2x, ab->bsum3, nblocks,
                  ab->bnb1, ab->bnb2, ab->bnb3, 0, 0, 0);
   {
      void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                    &ab->blk_list, &ab->blk_list_count, &ab->blk_offsets,
                    &ab->quad_prim, &ab->quad_mask, &ab->peel_mask,
                    &ab->quad_block, &ab->shade_slot, &ab->quad_capacity,
                    &ab->quad_overflow };
      CP_LAUNCH(screen->kernels.abuf_quad_fill, 1024, 1, 1, 32, 1, 1,
                     0, cp->stream, p, NULL);
   }

   cp_tile_census_quads(cp, segs, nsegs, w, h);

   if (cp_pass_finish_bounded_groups(cp, segs, nsegs, w, h))
      return;

   /* --- bucket the quads by segment, before the drain so the counts ride
    * it --- */
   CUdeviceptr quad_seg = cp_scratch_alloc_device(cp, ab->quad_capacity);
   uint32_t seg_prims[CP_PASS_MAX_SEGS];
   for (unsigned s = 0; s < nsegs; s++)
      seg_prims[s] = segs[s].prim_base;
   CUdeviceptr seg_prims_dev = cp_upload(cp, seg_prims, (size_t)nsegs * 4);
   if (!quad_seg || !seg_prims_dev) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }
   cuMemsetD32Async(ab->seg_counts, 0, CP_PASS_MAX_SEGS, cp->stream);
   struct cp_abuf_seg_args sa = {
      .quad_prim = ab->quad_prim,
      .seg_prim_base = seg_prims_dev,
      .seg_counts = ab->seg_counts,
      .quad_seg = quad_seg,
      .num_quads_dev = ab->bsum3,
      .nsegs = nsegs,
      .num_quads = (uint32_t)ab->quad_capacity,
      .warp_aggregate = !cp_debug->no_abuf_warp_bucket,
   };
   {
      void *p[] = { &sa };
      CP_LAUNCH(screen->kernels.abuf_seg_count,
                     MIN2(((unsigned)ab->quad_capacity + 255) / 256, 1024u),
                     1, 1, 256, 1, 1, 0, cp->stream, p, NULL);
   }

   /* --- the drain: the six counters and the per-segment quad counts --- */
   uint32_t ctr[CP_ABUF_COUNTERS + CP_PASS_MAX_SEGS] = { 0 };
   cuStreamSynchronize(cp->stream);
   cuMemcpyDtoH(ctr, ab->counters,
                sizeof(uint32_t) * (CP_ABUF_COUNTERS + nsegs));

   uint32_t total = ctr[0], fill_over = ctr[1];
   uint32_t quads = ctr[3], quad_over = ctr[4], covered = ctr[5];

   if (total > ab->peak)
      ab->peak = total;
   if (ab->frags && !ab->grow_capped && ab->growths < CP_ABUF_MAX_GROWTHS &&
       (double)total > (double)ab->capacity * CP_ABUF_GROW_AT)
      ab->grow_to = MAX2(ab->grow_to, total);

   if (fill_over || quad_over) {
      static int said = 0;
      if (!said++)
         fprintf(stderr, "abuffer: episode of %u segments overflowed "
                 "(fragments=%u quads=%u); re-rendering it segment by "
                 "segment\n", nsegs, fill_over, quad_over);
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }
   if (!quads)
      return;   /* nothing covered anything; there is nothing to composite */

   /*
    * --- merge segments into shading groups ---
    *
    * The capture's big passes alternate two vertex shaders draw by draw, so
    * an episode carries several times more segments than distinct shading
    * identities. Segments whose shade would be launched with the same
    * shaders, constant-buffer count and primitive mode shade as one group
    * over one contiguous slice of the grouped quad list — the interpolator
    * resolves each quad's own vertex stream and slice table through a range
    * table — and the composite still resolves per *segment*, through descs
    * synthesized as offsets into the group's arrays. Everything before this
    * point, the fallback, and the bucketing kernels are unchanged: grouping
    * is purely how the dense bases are laid out and how many shade launch
    * groups run.
    */
   uint8_t seg_group[CP_PASS_MAX_SEGS];
   unsigned group_first[CP_PASS_MAX_SEGS];
   unsigned ngroups = 0;
   for (unsigned s = 0; s < nsegs; s++) {
      unsigned g = ngroups;
      if (!cp_debug->no_seg_merge) {
         for (unsigned i = 0; i < ngroups; i++) {
            struct cp_pass_seg *f = &segs[group_first[i]];
            if (f->vs == segs[s].vs && f->fs == segs[s].fs &&
                f->num_fs_ubos == segs[s].num_fs_ubos &&
                f->info.mode == segs[s].info.mode) {
               g = i;
               break;
            }
         }
      }
      if (g == ngroups)
         group_first[ngroups++] = s;
      seg_group[s] = (uint8_t)g;
   }

   /* --- dense bases, group-major so each group's quads are one slice --- */
   uint32_t seg_base_host[CP_PASS_MAX_SEGS];
   uint32_t group_base[CP_PASS_MAX_SEGS], group_quads[CP_PASS_MAX_SEGS];
   uint32_t running = 0;
   for (unsigned g = 0; g < ngroups; g++) {
      group_base[g] = running;
      for (unsigned s = 0; s < nsegs; s++) {
         if (seg_group[s] != g)
            continue;
         seg_base_host[s] = running;
         running += ctr[CP_ABUF_COUNTERS + s];
      }
      group_quads[g] = running - group_base[g];
   }
   if (running != quads) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }
   CUdeviceptr grouped = cp_scratch_alloc_device(cp, (size_t)quads * 4);
   CUdeviceptr quad_dense = cp_scratch_alloc_device(cp, (size_t)quads * 4);
   CUdeviceptr seg_cursor = cp_scratch_alloc_device(cp,
                                                    (size_t)nsegs * 4);
   CUdeviceptr seg_base_dev = cp_upload(cp, seg_base_host,
                                        (size_t)nsegs * 4);
   if (!grouped || !quad_dense || !seg_cursor || !seg_base_dev) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }
   cuMemsetD32Async(seg_cursor, 0, nsegs, cp->stream);
   sa.num_quads = quads;
   sa.seg_cursor = seg_cursor;
   sa.seg_base = seg_base_dev;
   sa.grouped = grouped;
   sa.quad_dense = quad_dense;
   {
      void *p[] = { &sa };
      CP_LAUNCH(screen->kernels.abuf_seg_scatter, (quads + 255) / 256, 1, 1,
                     256, 1, 1, 0, cp->stream, p, NULL);
   }

   /* --- shade each group densely over its slice of the quads, fanned out --- */
   cp_pass_broadcast(cp, nsegs);
   struct cp_pass_live lv;
   cp_pass_live_save(cp, &lv);
   struct cp_seg_desc descs[CP_PASS_MAX_SEGS];
   memset(descs, 0, sizeof(descs));
   for (unsigned g = 0; g < ngroups && !failed; g++) {
      uint32_t gq = group_quads[g];
      if (!gq)
         continue;
      struct cp_pass_seg *sg = &segs[group_first[g]];
      unsigned members = 0;
      for (unsigned s = 0; s < nsegs; s++)
         members += seg_group[s] == g;
      if (cp->seg_streams[0])
         cp->stream = cp_pass_seg_stream(cp, g);
      cp->vs_shader = sg->vs;
      cp->fs_shader = sg->fs;
      cp->num_fs_ubos = sg->num_fs_ubos;
      struct cp_abuf_seg_shade ss = {
         .quad_list = grouped,
         .quad_list_base = group_base[g],
         .prim_base = sg->prim_base,
      };
      if (members == 1) {
         cp->fs_batch.ubos = sg->fs_ubos;
         cp->fs_batch.ndraws = sg->ndraws;
         cp->fs_batch.slices = sg->slices_dev;
         cp->fs_batch.prim_shift = sg->prim_shift;
      } else {
         /*
          * The group's range table, and — when the fragment shader reads
          * constant buffers — its members' fs-UBO rows concatenated, each
          * range knowing where its rows landed. The interpolator writes
          * group-global rows, so the shader indexes the concatenated table
          * exactly as it indexes a single batch's.
          */
         struct cp_seg_range ranges[CP_PASS_MAX_SEGS];
         unsigned nr = 0;
         uint32_t rows = 0;
         bool need_rows = sg->fs->reads_const_bufs;
         if (need_rows && !cp->pass_group_ubos) {
            cp->pass_group_ubos =
               malloc((size_t)CP_PASS_MAX_SEGS * CP_MAX_BATCH_DRAWS *
                      CP_ARG_UBO_STRIDE * sizeof(uint64_t));
            if (!cp->pass_group_ubos) {
               failed = true;
               break;
            }
         }
         for (unsigned s = 0; s < nsegs; s++) {
            if (seg_group[s] != g)
               continue;
            struct cp_pass_seg *m = &segs[s];
            ranges[nr++] = (struct cp_seg_range) {
               .positions = m->rast.positions,
               .draw_slices = m->slices_dev,
               .num_draw_slices = m->ndraws,
               .prim_base = m->prim_base,
               .prim_end = m->prim_base + m->prim_slots,
               .row_base = rows,
               .prim_shift = m->prim_shift,
            };
            if (need_rows)
               memcpy(cp->pass_group_ubos + (size_t)rows * CP_ARG_UBO_STRIDE,
                      m->fs_ubos,
                      (size_t)m->ndraws * CP_ARG_UBO_STRIDE *
                      sizeof(uint64_t));
            rows += m->ndraws;
         }
         CUdeviceptr ranges_dev =
            cp_upload(cp, ranges, (size_t)nr * sizeof(ranges[0]));
         if (!ranges_dev) {
            failed = true;
            break;
         }
         ss.ranges = ranges_dev;
         ss.num_ranges = nr;
         /* The launch-wide table and slices are placeholders the ranges
          * override per quad; ndraws is the concatenated row count, which
          * is what enables the batch-rows path and sizes the upload. */
         cp->fs_batch.ubos = need_rows ? cp->pass_group_ubos : sg->fs_ubos;
         cp->fs_batch.ndraws = rows;
         cp->fs_batch.slices = sg->slices_dev;
         cp->fs_batch.prim_shift = sg->prim_shift;
      }
      float ti, ts, tc;
      if (!cp_abuf_shade(cp, &sg->info, ab, sg->rast.positions,
                         sg->rast.positions, w, h,
                         sg->rast.vp_scale_x, sg->rast.vp_scale_y,
                         sg->rast.vp_trans_x, sg->rast.vp_trans_y,
                         gq, 0, false, NULL, false, &ti, &ts, &tc, &ss)) {
         failed = true;
         break;
      }
      /* Per-segment descs, as offsets into the group's dense arrays: the
       * composite still resolves by segment, so quad_seg and the bucketing
       * kernels never learned about groups. */
      for (unsigned s = 0; s < nsegs; s++) {
         if (seg_group[s] != g)
            continue;
         uint32_t sq = ctr[CP_ABUF_COUNTERS + s];
         if (!sq)
            continue;
         uint32_t off = (seg_base_host[s] - group_base[g]) * 4u;
         descs[s].fs_out = ss.fs_out + (size_t)off * ss.fs_out_stride;
         descs[s].coverage = ss.coverage ? ss.coverage + off : 0;
         descs[s].discard = ss.discard ? ss.discard + off : 0;
         descs[s].fs_out_stride = ss.fs_out_stride;
         descs[s].num_slots = sq * 4u;
      }
   }
   cp->stream = pass_main;
   cp_pass_live_restore(cp, &lv);
   cp_pass_join(cp, nsegs);
   if (failed) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   /* --- one composite for the whole episode --- */
   const struct cp_fb_desc *fb = &cp->fb;
   void *color_data = fb->color;
   CUdeviceptr descs_dev = cp_upload(cp, descs,
                                     (size_t)nsegs * sizeof(descs[0]));
   if (!color_data || !descs_dev) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }
   struct cp_abuf_composite_args ca = {
      .offsets = ab->offsets,
      .counts = ab->counts,
      .shade_slot = ab->shade_slot,
      .color_out = (uint64_t)(uintptr_t)color_data,
      .list = ab->clist,
      .list_count = ab->clist_count,
      .capacity = ab->capacity,
      .color_encoding = (uint32_t)MAX2(
         fb->color_encoding, 0),
      .max_layers = cp_abuf.max_layers,
      .blend = cp_blend_desc_for(cp),
      .quad_seg = quad_seg,
      .quad_dense = quad_dense,
      .seg_desc = descs_dev,
   };
   void *p[] = { &ca };
   unsigned nwork = covered ? covered : (unsigned)n;
   cp_nvtx_push("composite");
   CUresult ce = cuLaunchKernel(screen->kernels.abuf_composite,
                                (nwork + 255) / 256, 1, 1, 256, 1, 1,
                                0, cp->stream, p, NULL);
   cp_nvtx_pop();
   if (ce != CUDA_SUCCESS) {
      fprintf(stderr, "abuffer: episode composite launch failed (%d); "
              "re-rendering segment by segment\n", ce);
      cp_pass_fallback(cp, segs, nsegs);
   }
}

/* Record the segment cp_draw_execute has just counted; called from inside
 * it, with the count launches already on the stream. */
void
cp_pass_record_segment(struct cp_context *cp,
                       const struct cp_rasterize_args *aa,
                       const struct cp_rast_queues *queues,
                       unsigned rast_num_triangles, unsigned num_triangles,
                       const struct cp_draw_call *info,
                       unsigned drawid_offset, unsigned ndraws,
                       const struct cp_draw_range *draws,
                       const uint32_t *instance_counts,
                       const uint64_t *vs_ubo_table,
                       const uint64_t *fs_ubo_table,
                       const uint32_t *draw_ids, const uint64_t *vb_table,
                       const struct cp_rect *scissors)
{
   struct cp_pass_seg *sg = &cp->pass_segs[cp->pass.nsegs];

   memset(sg, 0, sizeof(*sg));
   sg->rast = *aa;
   sg->queues = *queues;
   sg->rast_num_triangles = rast_num_triangles;
   sg->num_triangles = num_triangles;
   sg->prim_base = cp->pass.next_prim;
   sg->prim_slots = rast_num_triangles;
   sg->prim_shift = cp->fs_batch.prim_shift;
   sg->vs = cp->vs_shader;
   sg->fs = cp->fs_shader;
   sg->info = *info;
   sg->ndraws = ndraws;
   sg->drawid_offset = drawid_offset;
   sg->slices_dev = cp->fs_batch.slices;
   memcpy(sg->draws, draws, (size_t)ndraws * sizeof(draws[0]));
   if (instance_counts)
      memcpy(sg->instance_counts, instance_counts,
             (size_t)ndraws * sizeof(instance_counts[0]));
   else
      for (unsigned d = 0; d < ndraws; d++)
         sg->instance_counts[d] = info->instance_count;
   if (draw_ids)
      memcpy(sg->draw_ids, draw_ids, (size_t)ndraws * sizeof(draw_ids[0]));
   if (scissors)
      memcpy(sg->scissors, scissors, (size_t)ndraws * sizeof(scissors[0]));
   if (vs_ubo_table)
      memcpy(sg->vs_ubos, vs_ubo_table,
             (size_t)ndraws * CP_ARG_UBO_STRIDE * sizeof(uint64_t));
   if (fs_ubo_table)
      memcpy(sg->fs_ubos, fs_ubo_table,
             (size_t)ndraws * CP_ARG_UBO_STRIDE * sizeof(uint64_t));
   if (vb_table)
      memcpy(sg->vb_bases, vb_table,
             (size_t)ndraws * CP_VB_TABLE_STRIDE * sizeof(uint64_t));
   memcpy(sg->velem, cp->velem, sizeof(sg->velem));
   memcpy(sg->vb_base, cp->vb_base, sizeof(sg->vb_base));
   sg->num_vertex_buffers = cp->num_vertex_buffers;
   sg->num_vertex_elements = cp->num_vertex_elements;
   sg->vertex_stride = cp->vertex_stride;
   sg->num_vs_ubos = cp->num_vs_ubos;
   sg->num_fs_ubos = cp->num_fs_ubos;

   if (cp->pass.nsegs == 0) {
      cp->pass.w = aa->width;
      cp->pass.h = aa->height;
   }
   cp->pass.nsegs++;
   cp->pass.next_prim += rast_num_triangles;
}

/* Append the pending batch to the episode as a segment; on a refusal deep
 * enough that only cp_draw_execute could see it, finish the episode and
 * render the batch the classic way — its draws came after every segment's. */
static void
cp_pass_append(struct cp_context *cp, unsigned ndraws)
{
   struct cp_abuf *ab = &cp_abuf;
   const struct cp_fb_desc *fb = &cp->fb;

   if (cp->pass.nsegs && cp->pass.opaque)
      cp_pass_finish(cp);

   /*
    * Episode start: size the per-pixel arrays for this framebuffer and clear
    * the shared lists once, on the main stream, with the gate event recorded
    * behind them — every segment stream waits on it before its first work.
    */
   if (cp->pass.nsegs == 0) {
      unsigned w = fb->width, h = fb->height;
      if (!w || !h || !cp_abuf_setup(ab, w, h)) {
         cp_pass_finish(cp);
         cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                         cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                         cp->batch.fs_ubos, cp->batch.draw_ids,
                         cp->batch.instance_counts, cp->batch.vb_bases,
                         cp->batch.scissors);
         return;
      }
      cuMemsetD32Async(ab->counts, 0, (size_t)w * h, cp->stream);
      cuMemsetD32Async(ab->sum3, 0, 3, cp->stream);
      if (ab->recs)
         cuMemsetD32Async(ab->rec_cursor, 0, 1, cp->stream);
      if (cp->seg_streams[0])
         cuEventRecord(cp->pass_gate, cp->stream);
   }

   CUstream saved_stream = cp->stream;
   struct cp_queue_set saved_qset = cp->cur_qset;
   if (cp->seg_streams[0]) {
      unsigned k = cp->pass.nsegs % CP_PASS_STREAMS;
      cuStreamWaitEvent(cp->seg_streams[k], cp->pass_gate, 0);
      cp->stream = cp->seg_streams[k];
      cp->cur_qset = cp->seg_qsets[k];
   }

   cp->pass.appending = true;
   cp->pass.append_failed = false;
   cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                   cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                   cp->batch.fs_ubos, cp->batch.draw_ids,
                   cp->batch.instance_counts, cp->batch.vb_bases,
                   cp->batch.scissors);
   cp->pass.appending = false;
   cp->stream = saved_stream;
   cp->cur_qset = saved_qset;

   if (cp->pass.append_failed) {
      /* The failed append may have run vertex work and uploads on its side
       * stream before backing out; when it was the would-be first segment,
       * cp_pass_finish below returns without joining anything, and the flush
       * fence — recorded on the main stream only — would not cover it. Join
       * every side stream so it always does. */
      cp_pass_join(cp, CP_PASS_STREAMS);
      cp_pass_finish(cp);
      cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                      cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                      cp->batch.fs_ubos, cp->batch.draw_ids,
                      cp->batch.instance_counts, cp->batch.vb_bases,
                      cp->batch.scissors);
   }
}

static void
cp_opaque_append(struct cp_context *cp, unsigned ndraws)
{
   if (cp->pass.nsegs && !cp->pass.opaque)
      cp_pass_finish(cp);
   if (cp->pass.nsegs >= CP_PASS_MAX_SEGS) {
      cp_pass_finish(cp);
   }

   if (!cp->pass.nsegs) {
      const struct cp_fb_desc *fb = &cp->fb;
      cp->pass.opaque = true;
      cp->pass.w = fb->width;
      cp->pass.h = fb->height;
      cp->pass.next_prim = 0;
      cuMemsetD32Async(cp->visbuf, 0xFFFFFFFF,
                       (size_t)fb->width * fb->height * 2,
                       cp->stream);
   }

   unsigned before = cp->pass.nsegs;
   cp->pass.appending = true;
   cp->pass.append_failed = false;
   cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                   cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                   cp->batch.fs_ubos, cp->batch.draw_ids,
                   cp->batch.instance_counts, cp->batch.vb_bases,
                   cp->batch.scissors);
   cp->pass.appending = false;

   if (cp->pass.append_failed || cp->pass.nsegs == before) {
      cp_pass_finish(cp);
      cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                      cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                      cp->batch.fs_ubos, cp->batch.draw_ids,
                      cp->batch.instance_counts, cp->batch.vb_bases,
                      cp->batch.scissors);
   }
}

void
cp_batch_flush(struct cp_context *cp)
{
   cp_batch_flush_why(cp, "a readback, a clear or a flush");
}

/*
 * The deferrable flush: the pending batch either joins the pass episode as a
 * segment or executes, but a pending *episode* stays open. Only the four
 * per-segment state changes may call this — a draw whose key broke the
 * batch, and the vertex-shader, fragment-shader and vertex-elements binds.
 * Everything else goes through cp_batch_flush_why(), which also finishes the
 * episode, because everything else either observes rendering or changes
 * state the episode reads episode-wide.
 */
void
cp_batch_flush_defer_why(struct cp_context *cp, const char *why)
{
   if (!cp->batch.pending)
      return;

   if (cp_debug->debug_batch)
      fprintf(stderr, "cudapipe: batch of %u ends: %s\n", cp->batch.ndraws, why);

   unsigned ndraws = cp->batch.ndraws;
   bool blended = cp->batch.blended;
   /* Cleared first: cp_draw_execute() runs a whole frame's worth of driver
    * code and nothing in it may see a batch that is already on its way. */
   cp->batch.pending = false;
   cp->batch.ndraws = 0;
   cp->batch.tris = 0;
   cp->batch.blended = false;

   if (cp_debug->debug_draw) {
      fprintf(stderr, "cudapipe: batch of %u draws\n", ndraws);
      for (unsigned d = 0; d < MIN2(ndraws, 4u); d++) {
         fprintf(stderr, "  row %u:", d);
         for (unsigned i = 0; i < cp->batch.key.num_vs_ubos; i++)
            fprintf(stderr, " %p",
                    (void *)(uintptr_t)cp->batch.vs_ubos[d * CP_ARG_UBO_STRIDE + i]);
         fprintf(stderr, "\n");
      }
   }

   if (blended && cp_pass_appendable(cp)) {
      cp_pass_append(cp, ndraws);
      return;
   }
   if (!blended && cp_opaque_appendable(cp)) {
      cp_opaque_append(cp, ndraws);
      return;
   }

   /* Whatever the episode holds was submitted before these draws. */
   cp_pass_finish(cp);
   cp->tile_census_solo += ndraws;
   cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                   cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                   cp->batch.fs_ubos, cp->batch.draw_ids,
                   cp->batch.instance_counts, cp->batch.vb_bases,
                   cp->batch.scissors);
}

void
cp_batch_flush_why(struct cp_context *cp, const char *why)
{
   cp_batch_flush_defer_why(cp, why);
   cp_pass_finish(cp);
}

/*
 * Bind a framebuffer: the resolved attachments, the sample count, and the five
 * framebuffer-sized buffers that scale with them.
 *
 * Grow-only, and that is the point: a real capture alternates 1280x720 with a
 * 160x90 bloom pyramid many times a frame, and an equality test threw all five
 * buffers away and rebuilt them in both directions 9.5 times a frame -- 356 GB
 * of allocation churn over a replay, 3.4 ms a frame, with the device idle for
 * every microsecond of it.
 */
void
cp_context_set_framebuffer(struct cp_context *cp, const struct cp_fb_desc *fb,
                           unsigned samples)
{
   cp->fb = *fb;

   /* Coverage and depth are per sample, so the buffers scale with the sample
    * count and it has to force a reallocation the same way the size does. */
   if (samples > CP_MAX_SAMPLES)
      samples = CP_MAX_SAMPLES;
   cp->fb_samples = samples;

   /*
    * Size the five framebuffer-sized buffers, growing only.
    *
    * These used to be reallocated whenever the bound size differed from the
    * last one, which is fine for a workload that renders one size and
    * pathological for one that does not: a real capture alternates 1280x720
    * with a 160x90 bloom pyramid many times a frame, so an equality test threw
    * all five away and rebuilt them in both directions 9.5 times a frame. That
    * measured 356 GB of allocation churn over a replay and 3.4 ms a frame,
    * with the device idle for every microsecond of it.
    *
    * Keeping the largest is safe because nothing here is addressed by capacity:
    * every kernel that reads these is bounded by the width and height passed at
    * launch, and the depth clear below uses the bound size, so a buffer sized
    * for 921,600 pixels serves a 14,400-pixel pass and clears only the part in
    * use. The two counts are tracked separately because visbuf and depthbuf
    * scale with samples and the other three do not.
    */
   unsigned w = cp->fb.width, h = cp->fb.height;
   size_t px = (size_t)w * h;
   size_t px_samples = px * samples;

   /*
    * A different size means the depth contents at these addresses belong to
    * some other framebuffer, whether or not the buffer was big enough to keep.
    * The reallocation used to imply this; now that it no longer happens on
    * every change, say it directly.
    */
   if (w != cp->visbuf_w || h != cp->visbuf_h || samples != cp->visbuf_samples)
      cp->depthbuf_cleared = false;

   cp->visbuf_w = cp->depthbuf_w = w;
   cp->visbuf_h = cp->depthbuf_h = h;
   cp->visbuf_samples = samples;

   if (px > 0 && (px > cp->fb_cap_px || px_samples > cp->fb_cap_px_samples)) {
      /* Grow both to the new high-water mark, so a later pass that is wider
       * but has fewer samples does not come back here. */
      cp->fb_cap_px = MAX2(cp->fb_cap_px, px);
      cp->fb_cap_px_samples = MAX2(cp->fb_cap_px_samples, px_samples);

      if (cp->visbuf)
         cuMemFree(cp->visbuf);
      if (cp->depthbuf)
         cuMemFree(cp->depthbuf);
      if (cp->reject)
         cuMemFree(cp->reject);
      if (cp->resolved)
         cuMemFree(cp->resolved);
      if (cp->peel_next)
         cuMemFree(cp->peel_next);
      cp->visbuf = 0;
      cp->depthbuf = 0;
      cp->reject = 0;
      cp->resolved = 0;
      cp->peel_next = 0;

      cuCtxSetCurrent(cp->screen->cuda_ctx);
      CUresult e1 = cuMemAlloc(&cp->visbuf,
                               cp->fb_cap_px_samples * sizeof(uint64_t));
      CUresult e2 = cuMemAlloc(&cp->depthbuf,
                               cp->fb_cap_px_samples * sizeof(uint32_t));
      CP_CU_WARN(cuMemAlloc(&cp->reject,
                            cp->fb_cap_px * CP_DISCARD_LAYERS * sizeof(uint32_t)),
                 "cuMemAlloc(reject)");
      CP_CU_WARN(cuMemAlloc(&cp->resolved, cp->fb_cap_px), "cuMemAlloc(resolved)");
      CP_CU_WARN(cuMemAlloc(&cp->peel_next, cp->fb_cap_px * sizeof(uint32_t)),
                 "cuMemAlloc(peel_next)");
      /* Managed, because the host reads it between passes to decide
       * whether another one is worth launching. */
      if (!cp->peel_any)
         cuMemAllocManaged(&cp->peel_any, sizeof(uint32_t),
                           CU_MEM_ATTACH_GLOBAL);
      if (e1 != CUDA_SUCCESS || e2 != CUDA_SUCCESS)
         fprintf(stderr, "cudapipe: visbuf/depthbuf alloc %zu px x %u samples "
                 "failed (%d, %d)\n", cp->fb_cap_px, samples, e1, e2);

      /* Contents are new, whatever was cleared before is gone. */
      cp->depthbuf_cleared = false;
   }

   if (cp->gpu_state) {
      cp->gpu_state->visbuf = cp->visbuf;
      cp->gpu_state->depthbuf = cp->depthbuf;
      cp->gpu_state->fb_width = w;
      cp->gpu_state->fb_height = h;
      if (cp->fb.color) {
         cp->gpu_state->color_attachment = (uint64_t)(uintptr_t)cp->fb.color;
         cp->gpu_state->color_encoding =
            (uint32_t)MAX2(cp->fb.color_encoding, 0);
      } else {
         cp->gpu_state->color_attachment = 0;
      }
   }
}



/*
 * The I/O slot size both front ends must lower with.
 *
 * Shared rather than copied: the vertex and fragment stages agree on where an
 * output lands only because both were lowered by the same function, and a
 * second copy that drifted would put the fragment shader's inputs at slots the
 * vertex shader never wrote.
 */
int
cp_type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}


/*
 * Publish the driver-owned state into the device-visible copy.
 *
 * The Gallium adapter writes these fields one at a time as Gallium's state
 * setters arrive, because that is the shape of the interface it is given.
 * Vulkan hands the whole pipeline at once, so the native front end fills the
 * driver's own structs and calls this. The two must agree field for field:
 * the device kernels read only this copy, and a field left at zero is not a
 * missing optimisation but a viewport that scales every vertex to a point,
 * which is exactly how the first native draw rasterized one triangle and
 * wrote nothing.
 */
void
cp_context_publish_state(struct cp_context *cp)
{
   if (!cp->gpu_state)
      return;

   cp->gpu_state->vp_scale_x = cp->viewport.scale[0];
   cp->gpu_state->vp_scale_y = cp->viewport.scale[1];
   cp->gpu_state->vp_trans_x = cp->viewport.translate[0];
   cp->gpu_state->vp_trans_y = cp->viewport.translate[1];

   cp->gpu_state->depth_test = cp->depth_stencil.depth_enabled;
   cp->gpu_state->depth_func = cp->depth_stencil.depth_func;
   cp->gpu_state->depth_write = cp->depth_stencil.depth_writemask;
   cp->gpu_state->depth_key_invert = cp->depth_stencil.depth_enabled &&
      (cp->depth_stencil.depth_func == CP_FUNC_GREATER ||
       cp->depth_stencil.depth_func == CP_FUNC_GEQUAL);

   cp->gpu_state->blend_enable = cp->blend_desc.enable;
   cp->gpu_state->colormask = cp->blend_desc.colormask ?
      cp->blend_desc.colormask : 0xF;
   cp->gpu_state->rgb_func = cp->blend_desc.rgb_func;
   cp->gpu_state->rgb_src_factor = cp->blend_desc.rgb_src_factor;
   cp->gpu_state->rgb_dst_factor = cp->blend_desc.rgb_dst_factor;
   cp->gpu_state->alpha_func = cp->blend_desc.alpha_func;
   cp->gpu_state->alpha_src_factor = cp->blend_desc.alpha_src_factor;
   cp->gpu_state->alpha_dst_factor = cp->blend_desc.alpha_dst_factor;

   cp->gpu_state->num_elements = cp->num_vertex_elements;
   cp->gpu_state->vs_in_stride = cp->num_vertex_elements * 16;
   for (unsigned i = 0; i < cp->num_vertex_elements && i < 16; i++) {
      cp->gpu_state->elem_vb_idx[i] = cp->velem[i].vertex_buffer_index;
      cp->gpu_state->elem_src_offset[i] = cp->velem[i].src_offset;
      cp->gpu_state->elem_src_stride[i] = cp->velem[i].src_stride;
      cp->gpu_state->elem_attr_size[i] = cp->velem[i].attr_size;
      cp->gpu_state->elem_instance_divisor[i] = cp->velem[i].instance_divisor;
   }
   for (unsigned i = 0; i < CP_MAX_VERTEX_BUFFERS_VF; i++)
      cp->gpu_state->vb_bases[i] = cp->vb_base[i];

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++) {
      cp->gpu_state->vs_ubos[i] = cp->vs_ubos[i].managed_copy;
      cp->gpu_state->fs_ubos[i] = cp->fs_ubos[i].managed_copy;
   }
}
