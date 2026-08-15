#include "cp_context.h"
#include "cp_screen.h"
#include "cp_nvtx.h"
#include "cp_resource.h"
#include "nir_to_ptx/cp_nir_to_llvm.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

static int
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}
#include "kernels/cp_rast_types.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_state.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_upload_mgr.h"
#include "util/u_framebuffer.h"
#include "compiler/shader_enums.h"
#include "util/u_prim.h"

#include "gallivm/lp_bld_jit_types.h"
#include "util/format/u_format.h"

#include <string.h>
#include <math.h>
#include <stddef.h>
#include <time.h>
#include <inttypes.h>   /* TEMPORARY: fragment census printf */
#include <cuda.h>

/*
 * The sampler reads two fields out of the descriptors lavapipe builds. Pin
 * those offsets here so an upstream layout change is a build failure instead
 * of silently corrupt texturing.
 */
static_assert(offsetof(struct lp_image_descriptor, texture.base) ==
              CP_DESC_IMAGE_BASE_OFFSET,
              "lp_image_descriptor texture base offset changed");
static_assert(offsetof(struct lp_image_descriptor, functions) ==
              CP_DESC_IMAGE_FUNCTIONS_OFFSET,
              "lp_image_descriptor functions offset changed");
static_assert(offsetof(struct lp_sampler_descriptor, sampler_index) ==
              CP_DESC_SAMPLER_INDEX_OFFSET,
              "lp_sampler_descriptor sampler_index offset changed");

static void cp_scratch_destroy(struct cp_context *cp);
static void cp_abuf_report(void);
static void cp_scratch_reset(struct cp_context *cp);

static void
cp_destroy_context(struct pipe_context *ctx)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   cp_batch_flush(cp);
   cp_abuf_report();
   if (cp->visbuf)
      cuMemFree(cp->visbuf);
   if (cp->depthbuf)
      cuMemFree(cp->depthbuf);
   if (cp->reject)
      cuMemFree(cp->reject);
   if (cp->peel_any)
      cuMemFree(cp->peel_any);
   if (cp->resolved)
      cuMemFree(cp->resolved);
   if (cp->rast_nontrivial)
      cuMemFree(cp->rast_nontrivial);
   if (cp->rast_huge_tiles)
      cuMemFree(cp->rast_huge_tiles);
   /* One allocation behind both counters; the two pointers into it are not
    * separately owned. */
   if (cp->rast_counts)
      cuMemFree(cp->rast_counts);
   if (cp->sampler_table)
      cuMemFree(cp->sampler_table);
   cp_scratch_destroy(cp);
   if (ctx->stream_uploader)
      u_upload_destroy(ctx->stream_uploader);
   FREE(cp);
}

static void
cp_set_framebuffer_state(struct pipe_context *ctx,
                         const struct pipe_framebuffer_state *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;

   /* The visibility and depth buffers may be freed below, and the held-back
    * draws were recorded against the framebuffer that is going away. */
   cp_batch_flush_why(cp, "framebuffer");

   util_copy_framebuffer_state(&cp->framebuffer, state);

   /* Coverage and depth are per sample, so the buffers scale with the sample
    * count and it has to force a reallocation the same way the size does. */
   unsigned samples = 1;
   if (state->nr_cbufs && state->cbufs[0].texture)
      samples = MAX2(state->cbufs[0].texture->nr_samples, 1u);
   else if (state->zsbuf.texture)
      samples = MAX2(state->zsbuf.texture->nr_samples, 1u);
   if (samples > CP_MAX_SAMPLES)
      samples = CP_MAX_SAMPLES;
   cp->fb_samples = samples;

   /* Reallocate the visibility and depth buffers if the size changed */
   unsigned w = state->width, h = state->height;
   if (w != cp->visbuf_w || h != cp->visbuf_h || samples != cp->visbuf_samples) {
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
      cp->visbuf_w = cp->depthbuf_w = w;
      cp->visbuf_h = cp->depthbuf_h = h;
      cp->visbuf_samples = samples;
      if (w > 0 && h > 0) {
         cuCtxSetCurrent(cp->screen->cuda_ctx);
         CUresult e1 = cuMemAlloc(&cp->visbuf,
                                  (size_t)w * h * samples * sizeof(uint64_t));
         CUresult e2 = cuMemAlloc(&cp->depthbuf,
                                  (size_t)w * h * samples * sizeof(uint32_t));
         cuMemAlloc(&cp->reject,
                    (size_t)w * h * CP_DISCARD_LAYERS * sizeof(uint32_t));
         cuMemAlloc(&cp->resolved, (size_t)w * h);
         cuMemAlloc(&cp->peel_next, (size_t)w * h * sizeof(uint32_t));
         /* Managed, because the host reads it between passes to decide
          * whether another one is worth launching. */
         if (!cp->peel_any)
            cuMemAllocManaged(&cp->peel_any, sizeof(uint32_t),
                              CU_MEM_ATTACH_GLOBAL);
         if (e1 != CUDA_SUCCESS || e2 != CUDA_SUCCESS)
            fprintf(stderr, "cudapipe: visbuf/depthbuf alloc %ux%u failed "
                    "(%d, %d)\n", w, h, e1, e2);
      }
      cp->depthbuf_cleared = false;
   }

   if (cp->gpu_state) {
      cp->gpu_state->visbuf = cp->visbuf;
      cp->gpu_state->depthbuf = cp->depthbuf;
      cp->gpu_state->fb_width = w;
      cp->gpu_state->fb_height = h;
      if (state->nr_cbufs && state->cbufs[0].texture) {
         struct cp_resource *cres = cp_resource(state->cbufs[0].texture);
         cp->gpu_state->color_attachment = (uint64_t)(uintptr_t)cp_resource_data(cres);
         cp->gpu_state->color_encoding = (uint32_t)MAX2(
            cp_color_encoding_from_format(state->cbufs[0].format), 0);
      } else {
         cp->gpu_state->color_attachment = 0;
      }
   }
}

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

static void
cp_set_viewport_states(struct pipe_context *ctx, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *viewports)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (num_viewports > 0) {
      if (memcmp(&cp->viewport, &viewports[0], sizeof(cp->viewport)))
         cp_batch_flush_why(cp, "viewport");
      cp->viewport = viewports[0];
      if (cp->gpu_state) {
         cp->gpu_state->vp_scale_x = viewports[0].scale[0];
         cp->gpu_state->vp_scale_y = viewports[0].scale[1];
         cp->gpu_state->vp_trans_x = viewports[0].translate[0];
         cp->gpu_state->vp_trans_y = viewports[0].translate[1];
      }
   }
}

static void
cp_set_scissor_states(struct pipe_context *ctx, unsigned start_slot,
                      unsigned num_scissors,
                      const struct pipe_scissor_state *scissors)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (num_scissors > 0) {
      if (memcmp(&cp->scissor, &scissors[0], sizeof(cp->scissor)))
         cp_batch_flush_why(cp, "scissor");
      cp->scissor = scissors[0];
   }
}

/*
 * A backstop, not a budget. The arena accumulates across the draws of a frame
 * and only resets after a few expansions, so a frame with a dozen draws at
 * 1280x720 legitimately reaches two or three gigabytes. What this catches is
 * the runaway: a stage allocating per pass rather than reusing asks for tens
 * of gigabytes, and since the arena is managed memory it is backed by system
 * RAM, so that does not fail — it invokes the OOM killer on the whole machine.
 */
#define CP_SCRATCH_MAX_BYTES ((size_t)8 << 30)

/* Sync and reclaim once a frame's draws have run up this much. */
#define CP_SCRATCH_RECLAIM_BYTES ((size_t)1 << 30)

/*
 * Hand out a slice of the draw's scratch arena.
 *
 * Returns managed memory, so the pointer is valid on both host and device. The
 * arena is only resized between draws, so a request that doesn't fit is served
 * by a one-off allocation and the arena grows to cover it next time rather
 * than moving memory that this draw is already pointing at.
 */
static void *
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
   if (want > CP_SCRATCH_MAX_BYTES) {
      fprintf(stderr, "cudapipe: scratch arena wants %zu bytes, over the %zu "
              "cap — refusing. A stage is almost certainly allocating per "
              "pass instead of reusing.\n",
              want, (size_t)CP_SCRATCH_MAX_BYTES);
      return NULL;
   }
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
static CUdeviceptr
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
   if (want > CP_SCRATCH_MAX_BYTES) {
      fprintf(stderr, "cudapipe: device scratch wants %zu bytes, over the %zu "
              "cap — refusing.\n", want, (size_t)CP_SCRATCH_MAX_BYTES);
      return 0;
   }

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
static CUdeviceptr
cp_upload_begin(struct cp_context *cp, size_t size, void **host_out)
{
   if (!cp->arena_base || !cp->upload_host || !size)
      return 0;

   /* 256 bytes keeps every block on its own cache line and matches the
    * alignment the constant-buffer path already promises. */
   size_t dev_off = ALIGN_POT(cp->arena_offset, 256);
   size_t host_off = ALIGN_POT(cp->upload_offset, 256);

   if (dev_off + size > cp->arena_size || host_off + size > cp->upload_size) {
      /*
       * Out of room before a flush came round. Rewinding would let this draw
       * overwrite staging a previous draw's copy has not read yet, so fall
       * back to a synchronous copy from the caller's own memory, which cannot
       * be reused early because it does not return until the copy is made.
       */
      cuCtxSynchronize();
      cp->arena_offset = cp->upload_offset = 0;
      dev_off = host_off = 0;
      if (size > cp->arena_size || size > cp->upload_size)
         return 0;
   }

   cp->arena_offset = dev_off + size;
   cp->upload_offset = host_off + size;
   *host_out = (char *)cp->upload_host + host_off;
   return cp->arena_base + dev_off;
}

/* Send a block reserved above, once the caller has finished writing it. */
static void
cp_upload_end(struct cp_context *cp, CUdeviceptr dst, const void *host,
              size_t size)
{
   cuMemcpyHtoDAsync(dst, host, size, cp->stream);
}

static CUdeviceptr
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

static void
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
static void
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

   /* Callers of this have already waited for the device, so the staging the
    * uploads were copied out of is free to be written over again. */
   cp->arena_offset = 0;
   cp->upload_offset = 0;
}

static void
cp_scratch_destroy(struct cp_context *cp)
{
   cuCtxSynchronize();
   for (unsigned i = 0; i < cp->scratch.num_overflow; i++)
      cuMemFree(cp->scratch.overflow[i]);
   for (unsigned i = 0; i < 2; i++) {
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
static bool
cp_timing_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("CUDAPIPE_DEBUG_TIME") ? 1 : 0;
   return enabled;
}

/*
 * Stage timing, on CUDA events rather than on the host clock.
 *
 * This used to bracket each stage with clock_gettime. That measures how long
 * the host spent issuing the stage, which was already only loosely related to
 * how long the device spent running it and is now not related at all: every
 * launch goes on a stream and returns immediately. A stage whose kernel runs
 * for a millisecond and whose launch takes two microseconds was being
 * reported as two microseconds, and the one unlucky stage that happened to
 * follow a full queue absorbed everyone else's time.
 *
 * Events are recorded on the same stream as the work, so the interval between
 * two of them is device time between those two points. Reading them back
 * needs the stream to have reached the last one, which is a synchronisation —
 * hence only under CUDAPIPE_DEBUG_TIME, and hence a pool rather than one pair
 * per stage, because a blended draw runs the shading stages hundreds of times
 * and every interval has to be recorded before any of them can be read.
 */
enum cp_stage {
   CP_STAGE_ASSEMBLE,
   CP_STAGE_VERTEX,
   CP_STAGE_RASTERIZE,
   CP_STAGE_INTERPOLATE,
   CP_STAGE_FRAGMENT,
   CP_STAGE_WRITEBACK,
   CP_NUM_STAGES,
};

/* Record that `stage` has just finished. The first mark of a draw carries no
 * stage and only starts the clock. */
static void
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

/* Resolve every interval recorded this draw into per-stage totals. Costs one
 * synchronisation, which is why it is debug-only. */
static void
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
static int32_t
cp_slot_for_location(const unsigned *locations, unsigned count,
                     unsigned location)
{
   for (unsigned i = 0; i < count && i < CP_MAX_IO_SLOTS; i++)
      if (locations[i] == location)
         return (int32_t)i;
   return -1;
}

/* One assembled vertex: which vertex of the bound buffers it reads, and which
 * instance it belongs to. */
struct cp_vertex_ref {
   uint32_t vertex;
   uint32_t instance;
};

/* Number of triangles one draw of `count` vertices produces. */
static unsigned
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
static struct cp_vertex_ref *
cp_build_vertex_refs(const struct pipe_draw_info *info,
                     const struct pipe_draw_start_count_bias *draws,
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

/*
 * Describe a vertex format for the fetch kernel.
 *
 * Vulkan delivers every component of a vertex attribute in its own 32 bit
 * slot however narrow it is in memory, so the fetch has to widen anything that
 * is not already 32 bits per component. cp_screen.c's claim that attributes
 * are "fetched as raw bytes and reinterpreted by the shader" holds only for
 * the 32-bit-per-component formats; for an R8G8B8A8_UINT it packs all four
 * components into the first slot, which a shader indexing an array with the
 * result reads as a value up to 2^32.
 *
 * Fills nr_chan, chan_bytes and swizzle, and returns the conversion. Formats
 * whose channels are not a whole number of bytes, or not all the same width,
 * keep the old verbatim copy — they would need bitfield extraction, and
 * nothing reaching this driver uses one.
 */
static enum cp_vf_conv
cp_vertex_format(enum pipe_format format, uint32_t *nr_chan,
                 uint32_t *chan_bytes, uint32_t *swizzle)
{
   const struct util_format_description *desc =
      util_format_description(format);

   *nr_chan = 0;
   *chan_bytes = 0;
   *swizzle = 0x3210;

   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN)
      return CP_VF_CONV_COPY32;

   const struct util_format_channel_description *chan = &desc->channel[0];
   unsigned size = chan->size;

   if (size % 8 || size > 32)
      return CP_VF_CONV_COPY32;

   for (unsigned c = 1; c < desc->nr_channels; c++) {
      if (desc->channel[c].size != size ||
          desc->channel[c].type != chan->type ||
          desc->channel[c].normalized != chan->normalized ||
          desc->channel[c].pure_integer != chan->pure_integer)
         return CP_VF_CONV_COPY32;
   }

   *nr_chan = desc->nr_channels;
   *chan_bytes = size / 8;

   uint32_t swz = 0;
   for (unsigned c = 0; c < 4; c++) {
      unsigned s = c < 4 ? desc->swizzle[c] : PIPE_SWIZZLE_0;
      /* Anything that is not a plain channel reference reads as the zero fill
       * or as fill_w, so point it past the channel count and let the kernel
       * skip it. */
      swz |= (uint32_t)(s <= PIPE_SWIZZLE_W ? s : 0xf) << (c * 4);
   }
   *swizzle = swz;

   if (size == 32)
      return CP_VF_CONV_COPY32;

   switch (chan->type) {
   case UTIL_FORMAT_TYPE_FLOAT:
      return size == 16 ? CP_VF_CONV_FLOAT16 : CP_VF_CONV_COPY32;
   case UTIL_FORMAT_TYPE_UNSIGNED:
      if (chan->normalized)
         return CP_VF_CONV_UNORM;
      return chan->pure_integer ? CP_VF_CONV_UINT : CP_VF_CONV_USCALED;
   case UTIL_FORMAT_TYPE_SIGNED:
      if (chan->normalized)
         return CP_VF_CONV_SNORM;
      return chan->pure_integer ? CP_VF_CONV_SINT : CP_VF_CONV_SSCALED;
   default:
      return CP_VF_CONV_COPY32;
   }
}

/*
 * The value a vertex attribute's fourth component reads as when the format
 * doesn't supply one. Vulkan defines the missing components of a vertex
 * attribute as (0, 0, 0, 1), and the zero-filled slot already covers y and z.
 * Returns 0 when the format supplies all four components and nothing is due.
 *
 * Which one it is follows from the conversion rather than from the format:
 * every conversion that produces a float wants 1.0f, and only the ones that
 * leave an integer in the slot want an integer 1.
 */
static uint32_t
cp_vertex_fill_w(enum pipe_format format, enum cp_vf_conv conv)
{
   const struct util_format_description *desc =
      util_format_description(format);

   if (!desc || desc->nr_channels >= 4)
      return 0;

   bool is_float;
   switch (conv) {
   case CP_VF_CONV_UINT:
   case CP_VF_CONV_SINT:
      is_float = false;
      break;
   case CP_VF_CONV_COPY32:
      /* Untouched 32 bit components: float unless the format is a plain
       * integer one. */
      is_float = desc->channel[0].type == UTIL_FORMAT_TYPE_FLOAT ||
                 desc->channel[0].normalized;
      break;
   default:
      is_float = true;   /* unorm, snorm, uscaled, sscaled, half */
      break;
   }

   if (!is_float)
      return 1;

   float one = 1.0f;
   uint32_t bits;
   memcpy(&bits, &one, 4);
   return bits;
}

/* 0 = keep everything, 1 = drop positive-area triangles, 2 = drop negative. */
static uint32_t
cp_cull_mode(const struct pipe_rasterizer_state *rs)
{
   bool cull_back = (rs->cull_face & PIPE_FACE_BACK) != 0;
   bool cull_front = (rs->cull_face & PIPE_FACE_FRONT) != 0;

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

/* Which colour encoding the fragment writeback can produce, or -1 if it can't
 * write this format at all. */
int
cp_color_encoding_from_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return CP_COLOR_R8G8B8A8_UNORM;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return CP_COLOR_B8G8R8A8_UNORM;
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
      return CP_COLOR_R8G8B8A8_SRGB;
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return CP_COLOR_B8G8R8A8_SRGB;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return CP_COLOR_R32G32B32A32_FLOAT;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      return CP_COLOR_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return CP_COLOR_R11G11B10_FLOAT;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return CP_COLOR_A2B10G10R10_UNORM;
   case PIPE_FORMAT_R16_FLOAT:
      return CP_COLOR_R16_SFLOAT;
   case PIPE_FORMAT_R16G16_FLOAT:
      return CP_COLOR_R16G16_SFLOAT;
   case PIPE_FORMAT_R8_UNORM:
      return CP_COLOR_R8_UNORM;
   default:
      return -1;
   }
}

/*
 * TEMPORARY (CUDAPIPE_ABUFFER): the merged quad array, for the one draw whose
 * peel loop is being compared against it. Zero for every other draw, and for
 * every draw once the verification budget is spent, so the instrumented
 * interpolator is not carried by frames that are only being timed.
 */
static struct {
   CUdeviceptr blk_offsets, blk_counts, quad_prim, peel_mask, counters;
   /*
    * Step 3b: where the peel path deposits its shaded colours. One slot per
    * (pixel, primitive) — the A-buffer's own indexing — so the two paths can
    * be compared without either of them agreeing on an order.
    */
   CUdeviceptr frags, offsets, counts, colors, writes;
   uint32_t capacity;
} cp_abuf_dbg;

/*
 * The draw's blend equation, in the form both kernels that evaluate it read.
 *
 * Split out for the same reason the struct is shared: the peel path's
 * writeback and the A-buffer's composite have to be blending the same draw the
 * same way, and a second place that turns a pipe_rt_blend_state into one is a
 * second place that can forget a field.
 */
static struct cp_blend_desc
cp_blend_desc_for(const struct cp_context *cp)
{
   const struct pipe_rt_blend_state *rt = &cp->blend_state.rt[0];
   struct cp_blend_desc b = {
      .enable = rt->blend_enable,
      .rgb_src_factor = rt->rgb_src_factor,
      .rgb_dst_factor = rt->rgb_dst_factor,
      .rgb_func = rt->rgb_func,
      .alpha_src_factor = rt->alpha_src_factor,
      .alpha_dst_factor = rt->alpha_dst_factor,
      .alpha_func = rt->alpha_func,
      /* A zero write mask reaches here from a state object that never set
       * one, so it means "all four" rather than "none" — which is what the
       * writeback has always done with it. */
      .colormask = rt->colormask ? rt->colormask : 0xF,
   };
   return b;
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
cp_fs_interp_setup(struct cp_context *cp, const struct pipe_draw_info *info,
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
#define CP_TUNE_VETO 1.05   /* how much worse capped has to be to be refused */

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
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("CUDAPIPE_NO_REGCAP") ? 0 : 1;
   if (!enabled || !fs->tune_cap || fs->tune_done)
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
      const char *v = getenv("CUDAPIPE_TUNE_VETO");
      bool keep = capped < as_built * (v ? atof(v) : CP_TUNE_VETO);
      if (keep)
         t->swap_pending = true;   /* back to capped, on the next launch */

      if (getenv("CUDAPIPE_SHADER_STATS"))
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
                    CUdeviceptr front_face,
                    unsigned num_threads, CUevent ev_before,
                    CUdeviceptr batch_rows)
{
   /* Both blocks go to the device by DMA rather than through managed memory
    * the host has just dirtied — see cp_upload(). */
   uint32_t fs_stride_host = fs_in_stride;
   CUdeviceptr stride_dev = cp_upload(cp, &fs_stride_host,
                                      sizeof(fs_stride_host));
   if (!stride_dev)
      return false;

   /*
    * The argument block, and behind it the per-draw uniform table the shader
    * indexes into — one upload, because the block holds the table's device
    * address. The shape is the vertex stage's, for the same reason and with
    * the same single-draw degeneracy: one row, a zero mask and a one-word row
    * array holding zero, so a draw that is not a batch computes exactly the
    * args[18 + i] it always did. See CP_ARG_SLOT_UBO_TABLE.
    */
   const uint64_t *tbl_src = cp->fs_batch.ubos;
   unsigned rows = tbl_src ? MAX2(cp->fs_batch.ndraws, 1u) : 1;
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
   fs_args_host[3] = (void *)(uintptr_t)stride_dev;
   fs_args_host[4] = (void *)(uintptr_t)fs_out;
   fs_args_host[6] = (void *)(uintptr_t)frag_coord;
   fs_args_host[CP_ARG_SLOT_DISCARD] = (void *)(uintptr_t)discard_mask;
   fs_args_host[CP_ARG_SLOT_FRONT_FACE] = (void *)(uintptr_t)front_face;
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

   if (getenv("CUDAPIPE_DEBUG_TEX")) {
      fprintf(stderr, "cudapipe: sampler table %p (%u entries) for FS module\n",
              (void *)(uintptr_t)cp->sampler_table, cp->num_samplers);

      /* Walk each bound descriptor the way the sampler does, so a mismatch
       * between what the host bound and what the shader samples is visible. */
      for (unsigned b = 0; b < cp->num_fs_ubos; b++) {
         if (!cp->fs_ubos[b].buffer)
            continue;
         const char *desc = (const char *)cp->fs_ubos[b].buffer;
         const struct cp_texture_info *ti =
            *(const struct cp_texture_info *const *)(desc + CP_DESC_IMAGE_FUNCTIONS_OFFSET);
         if (!ti)
            continue;
         fprintf(stderr, "  fs_ubo[%u]: %ux%u enc=%u stride=%u levels=%u..%u "
                 "base=%p\n", b, ti->width, ti->height, ti->encoding,
                 ti->row_stride[0], ti->first_level, ti->last_level,
                 (void *)(uintptr_t)ti->base);
      }
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
      if (!fs->globals_resolved) {
         if (cuModuleGetGlobal(&sym, &sym_size, fs->module,
                               "cp_sampler_table") == CUDA_SUCCESS)
            fs->sym_sampler_table = sym;
         if (cuModuleGetGlobal(&sym, &sym_size, fs->module,
                               "cp_quad_derivs") == CUDA_SUCCESS)
            fs->sym_quad_derivs = sym;
         /* Nothing has been written yet, and zero is a value the sampler
          * table can legitimately take, so neither cache is valid until the
          * first write below. */
         fs->last_sampler_table = ~(uint64_t)0;
         fs->last_quad_derivs = -1;
         fs->globals_resolved = true;
      }

      if (cp->sampler_table && fs->sym_sampler_table &&
          fs->last_sampler_table != (uint64_t)cp->sampler_table) {
         uint64_t addr = (uint64_t)cp->sampler_table;
         cuMemcpyHtoD(fs->sym_sampler_table, &addr, sizeof(addr));
         fs->last_sampler_table = addr;
      }
      if (fs->sym_quad_derivs && fs->last_quad_derivs != 1) {
         int on = 1;
         cuMemcpyHtoD(fs->sym_quad_derivs, &on, sizeof(on));
         fs->last_quad_derivs = 1;
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
      bool timed = cp_tune_before(cp, fs);
      CUresult fs_err = cuLaunchKernel(fs->kernel, (num_threads + 255) / 256,
                                       1, 1, 256, 1, 1, 0, cp->stream,
                                       fs_params, NULL);
      if (fs_err != CUDA_SUCCESS) {
         fprintf(stderr, "cudapipe: fragment shader launch failed (%d)\n",
                 fs_err);
         return false;
      }
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
static void
cp_shade_fragments(struct cp_context *cp, const struct pipe_draw_info *info,
                   CUdeviceptr visbuf, CUdeviceptr positions,
                   CUdeviceptr vs_output_buf, unsigned num_triangles,
                   unsigned w, unsigned h, void *color_data,
                   float vp_scale_x, float vp_scale_y,
                   float vp_trans_x, float vp_trans_y,
                   CUdeviceptr reject, CUdeviceptr resolved,
                   unsigned reject_pass)
{
   struct cp_screen *screen = cp->screen;
   struct cp_shader_binary *fs = cp->fs_shader;

   if (!fs || !fs->kernel || !vs_output_buf || !cp->vs_shader ||
       !screen->kernels.fs_interpolate || !screen->kernels.fs_writeback) {
      if (getenv("CUDAPIPE_DEBUG_DRAW"))
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
                            frag_coord, discard_mask, front_face, num_pixels, 0,
                            batch_rows))
      return;
   cp_stage_end(cp, CP_STAGE_FRAGMENT);

   /* TEMPORARY (CUDAPIPE_ABUFFER): scatter this pass's colours into the
    * A-buffer slots the interpolation just resolved. */
   if (dbg_slot && screen->kernels.abuf_scatter_colors) {
      void *p[] = { &fs_out, &fs_out_stride, &dbg_slot, &counter, &num_pixels,
                    &cp_abuf_dbg.capacity, &cp_abuf_dbg.colors,
                    &cp_abuf_dbg.writes, &cp_abuf_dbg.counters };
      cuLaunchKernel(screen->kernels.abuf_scatter_colors,
                     (num_pixels + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, p, NULL);
   }

   const struct pipe_rt_blend_state *rt = &cp->blend_state.rt[0];
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
         (cp->depth_stencil.depth_func == PIPE_FUNC_GREATER ||
          cp->depth_stencil.depth_func == PIPE_FUNC_GEQUAL),
      .width = w,
      .fs_out_stride = fs_out_stride,
      .num_pixels = num_pixels,
      .color_encoding = (uint32_t)MAX2(
         cp_color_encoding_from_format(cp->framebuffer.cbufs[0].format), 0),
      .blend = cp_blend_desc_for(cp),
      .num_samples = MAX2(cp->fb_samples, 1u),
      .height = h,
      .sample_stride = cp->framebuffer.nr_cbufs &&
                       cp->framebuffer.cbufs[0].texture
         ? (uint32_t)cp_resource(cp->framebuffer.cbufs[0].texture)->lpr.sample_stride
         : 0,
   };

   void *wb_params[] = { &wb };
   cp_nvtx_push("writeback");
   cuLaunchKernel(screen->kernels.fs_writeback,
                  (num_pixels + 255) / 256, 1, 1, 256, 1, 1,
                  0, cp->stream, wb_params, NULL);
   cp_nvtx_pop();   /* writeback */
   cp_stage_end(cp, CP_STAGE_WRITEBACK);

   /* How much of the shading launch does any work. Syncs, so debug only, and
    * read it on a deterministic sample. */
   if (getenv("CUDAPIPE_DEBUG_WORK")) {
      uint32_t shaded = 0;
      cuStreamSynchronize(cp->stream);
      cuMemcpyDtoH(&shaded, counter, sizeof(shaded));
      fprintf(stderr, "work shaded=%u fs_threads=%u interp_threads=%u fb=%u\n",
              MIN2(shaded, max_pixels), num_pixels,
              ((w + 1) / 2) * ((h + 1) / 2), w * h);
   }

   if (getenv("CUDAPIPE_DEBUG_DISCARD")) {
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

   if (getenv("CUDAPIPE_DEBUG_DRAW"))
      fprintf(stderr, "  shaded %u pixels (%u fs inputs, %u tris) "
              "blend=%u src=%u dst=%u mask=0x%x\n",
              num_pixels, num_fs_inputs, num_triangles,
              rt->blend_enable, rt->rgb_src_factor, rt->rgb_dst_factor,
              wb.blend.colormask);

   if (getenv("CUDAPIPE_DEBUG_FS")) {
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

      const char *step_env = getenv("CUDAPIPE_DEBUG_FS_VSTEP");
      unsigned vstep = step_env ? (unsigned)atoi(step_env) : 1;
      if (vstep < 1)
         vstep = 1;
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
      const char *row_env = getenv("CUDAPIPE_DEBUG_FS_ROW");
      int want_row = row_env ? atoi(row_env) : -1;
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
                 cb[px]);
      }
      free(vs_out); free(plist_buf); free(fin_buf); free(fout_buf);
   }

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
static bool
cp_census_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("CUDAPIPE_FRAG_CENSUS") ? 1 : 0;
   return enabled == 1;
}

static int
cp_census_cmp_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
   return x < y ? -1 : x > y ? 1 : 0;
}

static void
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

/* Both paths' shaded colours, one float4 and one write count per slot each.
 * The measured population needs about 135 MB of this; a draw wanting more is
 * shaded and timed without being checked, rather than allocated for. */
#define CP_ABUF_MAX_COLOR_BYTES (512.0 * 1024.0 * 1024.0)

/* Slots the quad stream's own shading pass may use, four per quad. Bounds the
 * fragment shader's input and output buffers, which at five varyings are about
 * 90 bytes a slot. */
#define CP_ABUF_MAX_SHADE_SLOTS (16u * 1024u * 1024u)

/* Pixels whose whole run is compared against the peel loop, deepest first. */
#define CP_ABUF_DEEP_PIXELS 1000

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
#define CP_ABUF_GROW_AT      0.75    /* fraction of capacity that triggers a grow */
#define CP_ABUF_MAX_GROWTHS  8u

struct cp_abuf {
   int enabled;             /* -1 unknown, 0 off, 1 on */
   int verify;
   /* Whether an eligible draw is rendered by this path instead of by the peel
    * loop, rather than merely having its lists built and checked beside it. */
   int composite;
   int timing;              /* print the per-draw event breakdown */
   int debug;               /* explain on stderr why a draw is not eligible */
   /* Layers the composite stops after, 0 for all of them. Only for asking what
    * the peel loop's own CP_BLEND_LAYERS truncation was worth; the path has no
    * such cap and is not meant to acquire one. */
   unsigned max_layers;
   bool ready;              /* per-pixel buffers allocated for w x h */
   bool disabled;           /* something refused it; do not try again */
   unsigned w, h;
   /*
    * What the framebuffer-sized allocations actually hold, as against what the
    * current framebuffer needs. Grow-only: a render pass at a smaller size
    * reuses the larger arrays and only recomputes the derived counts below.
    * See cp_abuf_setup() for why it is not a realloc on every change.
    */
   size_t cap_pixels;       /* pixels the per-pixel arrays are sized for */
   unsigned cap_blocks;     /* 2x2 blocks the per-block arrays are sized for */
   size_t cap_log_pixels;   /* pixels the verification's peel log is sized for */
   unsigned resizes;        /* times the arrays have been grown */
   bool events_ready;       /* the CUevents are created once, not per size */

   /* Per-pixel, and the scan's per-level partial sums. */
   CUdeviceptr counts, offsets, cursor;
   CUdeviceptr sum1, sum1x, sum2, sum2x, sum3;
   unsigned nb1, nb2, nb3;

   /* The sort's worklist, and the two failure counters the kernels bump. */
   CUdeviceptr list, list_count, overflow, long_runs;

   /* The composite's own worklist: every pixel with at least one fragment,
    * where the sort's holds only those with more than one. Separate arrays
    * rather than one with the looser threshold, so the sort keeps costing what
    * it was measured to cost. */
   CUdeviceptr clist, clist_count;

   /* The fragment array itself: sized from the first draw's count with
    * CP_ABUF_HEADROOM to spare, and grown between draws when a later count
    * comes within CP_ABUF_GROW_AT of it. See the constants above for why the
    * growth is bounded the way it is. */
   CUdeviceptr frags;
   unsigned capacity;
   unsigned growths;        /* times it has been grown this process */
   unsigned peak;           /* largest population any draw has counted */
   bool grow_capped;        /* a growth was refused; do not ask again */
   /*
    * A population that wants the arrays bigger, recorded by the draw that
    * counted it and acted on by the next eligible draw. The count is no longer
    * read on the host until after the fill has been issued — see the drain
    * that used to be between them — so by the time a growth is known to be
    * wanted, this draw's own kernels are already reading the arrays it would
    * free. The next draw's count pass reads none of them, and is where the
    * resize is safe.
    */
   uint32_t grow_to;

   /* What the peel loop selected: CP_ABUF_LOG_LAYERS for every pixel, and the
    * full CP_BLEND_LAYERS for the deepest CP_ABUF_DEEP_PIXELS. */
   CUdeviceptr log, deep_list, deep_log;

   /*
    * Step 3a: the quad stream. Per 2x2 block, a worklist of the blocks with
    * any coverage and a count of the distinct primitives in them; per quad,
    * that primitive and the 4-bit mask of the block's pixels it covers.
    * bsum3 holds the quad total, the fill's overflow counter and the
    * interpolator's four debug counters.
    */
   CUdeviceptr blk_counts, blk_offsets, blk_list, blk_list_count;
   CUdeviceptr bsum1, bsum1x, bsum2, bsum2x, bsum3, quad_overflow, dbg;
   unsigned bnb1, bnb2, bnb3, nblocks, quad_width;
   CUdeviceptr quad_prim, quad_mask, peel_mask, quad_block;
   unsigned quad_capacity;

   /*
    * Step 4: for each A-buffer slot, the shading slot holding that fragment's
    * colour. Written by the merge, which is the only place both indices are in
    * hand at once, and read by the composite as it walks a pixel's run.
    */
   CUdeviceptr shade_slot;

   /*
    * Step 3b: the shaded colours, one slot per (pixel, primitive), written by
    * both paths and compared element-wise. `writes` counts rather than flags,
    * so a slot two fragments claimed is visible instead of passing for
    * agreement.
    */
   CUdeviceptr colors_abuf, writes_abuf, colors_peel, writes_peel;
   bool colors_ready;

   CUevent ev[17];

   uint32_t *h_counts, *h_offsets, *h_cursor, *h_frags, *h_log, *h_deep_log;
   uint32_t *h_deep_list;
   uint32_t *h_blk_counts, *h_blk_offsets, *h_quad_prim, *h_peel_mask;
   unsigned char *h_quad_mask;
   float *h_colors_abuf, *h_colors_peel;
   uint32_t *h_writes_abuf, *h_writes_peel;

   unsigned verified, verify_max;
   unsigned seq;
};

static struct cp_abuf cp_abuf = { .enabled = -1 };

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
static inline void
cp_abuf_mark(CUevent ev, CUstream stream)
{
   if (cp_abuf.timing)
      cuEventRecord(ev, stream);
}

static bool
cp_abuf_enabled(void)
{
   if (cp_abuf.enabled < 0) {
      /* On by default, off with CUDAPIPE_NO_ABUFFER=1 — the same shape as
       * CUDAPIPE_NO_BATCH and CUDAPIPE_NO_BINCACHE. CUDAPIPE_ABUFFER=1 still
       * means what it always did and is now a no-op, so a command line or a
       * script written against the opt-in version still does what it says. */
      cp_abuf.enabled = getenv("CUDAPIPE_NO_ABUFFER") ? 0 : 1;

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
      const char *v = getenv("CUDAPIPE_ABUFFER_VERIFY");
      const char *c = getenv("CUDAPIPE_ABUFFER_COMPOSITE");
      cp_abuf.verify = v ? atoi(v) != 0 : 0;
      cp_abuf.composite = c ? atoi(c) != 0 : !cp_abuf.verify;
      if (cp_abuf.composite)
         cp_abuf.verify = 0;

      const char *n = getenv("CUDAPIPE_ABUFFER_VERIFY_DRAWS");
      cp_abuf.verify_max = n ? (unsigned)atoi(n) : 8;
      /* The per-draw event breakdown costs a drain and a line of stderr per
       * draw, which nothing on the default path wants — it was on by default
       * while the path was opt-in and something being examined, and is off by
       * default now that it is how blended draws are rendered. */
      const char *t = getenv("CUDAPIPE_ABUFFER_TIMING");
      cp_abuf.timing = t ? atoi(t) != 0 : 0;
      /* Likewise the running commentary on which draws are eligible: useful
       * when the question is why a draw peeled, noise on every other run. */
      const char *d = getenv("CUDAPIPE_ABUFFER_DEBUG");
      cp_abuf.debug = d ? atoi(d) != 0 : 0;
      const char *l = getenv("CUDAPIPE_ABUFFER_LAYERS");
      cp_abuf.max_layers = l && *l ? (unsigned)atoi(l) : 0;
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
static bool
cp_abuf_batch_enabled(void)
{
   static int on = -1;
   if (on < 0)
      on = getenv("CUDAPIPE_NO_ABUF_BATCH") ? 0 : 1;
   return on == 1;
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
static bool
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
      if (!cp_abuf_alloc(ab, &ab->sum3, 4 * 3, "sum3+counters") ||
          !cp_abuf_alloc(ab, &ab->list_count, 4, "list_count") ||
          !cp_abuf_alloc(ab, &ab->blk_list_count, 4, "block worklist count") ||
          !cp_abuf_alloc(ab, &ab->bsum3, 4 * 2, "block sum3+overflow") ||
          !cp_abuf_alloc(ab, &ab->dbg, CP_ABUF_DBG_COUNTERS * 4,
                         "debug counters"))
         return false;
      /* Three words in one allocation: the scan total, the fill's overflow
       * counter and the sort's long-run counter. */
      ab->overflow = ab->sum3 + 4;
      ab->long_runs = ab->sum3 + 8;
      ab->quad_overflow = ab->bsum3 + 4;
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
      if (!ab->clist_count &&
          !cp_abuf_alloc(ab, &ab->clist_count, 4, "composite worklist count"))
         return false;
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
static void
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
static bool
cp_abuf_size_arrays(struct cp_abuf *ab, uint32_t total)
{
   bool grow = ab->frags &&
      (double)total > (double)ab->capacity * CP_ABUF_GROW_AT;

   if (ab->frags && !grow)
      return true;
   if (grow && (ab->grow_capped || ab->growths >= CP_ABUF_MAX_GROWTHS))
      return true;

   size_t want = (size_t)total * CP_ABUF_HEADROOM + CP_ABUF_SLACK;
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
              was, ab->capacity, want * 4.0 / (1024.0 * 1024.0),
              want * 13.0 / (1024.0 * 1024.0), total,
              was ? 100.0 * total / was : 0.0, ab->growths,
              CP_ABUF_MAX_GROWTHS);
   } else {
      fprintf(stderr, "abuffer: fragment array %u entries (%.1f MB, %.1f MB "
              "of quads) from a first count of %u\n", ab->capacity,
              want * 4.0 / (1024.0 * 1024.0), want * 13.0 / (1024.0 * 1024.0),
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
static void
cp_abuf_scan_n(struct cp_context *cp, struct cp_screen *screen,
               CUdeviceptr in, CUdeviceptr out, CUdeviceptr s1, CUdeviceptr s1x,
               CUdeviceptr s2, CUdeviceptr s2x, CUdeviceptr s3,
               unsigned n, unsigned nb1, unsigned nb2, unsigned nb3)
{
   struct { CUdeviceptr in, out, sums; unsigned n, grid; } lvl[3] = {
      { in, out, s1, n,   nb1 },
      { s1,  s1x, s2, nb1, nb2 },
      { s2,  s2x, s3, nb2, nb3 },
   };
   for (int i = 0; i < 3; i++) {
      void *p[] = { &lvl[i].in, &lvl[i].out, &lvl[i].sums, &lvl[i].n };
      cuLaunchKernel(screen->kernels.abuf_scan_block, lvl[i].grid, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p, NULL);
   }
   /* Add each level's scanned bases back down. */
   for (int i = 1; i >= 0; i--) {
      CUdeviceptr data = i ? s1x : out;
      CUdeviceptr sums = i ? s2x : s1x;
      unsigned cnt = i ? nb1 : n;
      unsigned grid = i ? nb2 : nb1;
      void *p[] = { &data, &sums, &cnt };
      cuLaunchKernel(screen->kernels.abuf_scan_add, grid, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p, NULL);
   }
}

static void
cp_abuf_scan(struct cp_context *cp, struct cp_screen *screen,
             struct cp_abuf *ab, unsigned n)
{
   cp_abuf_scan_n(cp, screen, ab->counts, ab->offsets, ab->sum1, ab->sum1x,
                  ab->sum2, ab->sum2x, ab->sum3, n, ab->nb1, ab->nb2, ab->nb3);
}

/* Deepest pixels first, for the full-depth half of the comparison. */
struct cp_abuf_deep { uint32_t count, pixel; };

static int
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
static void
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
static void
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
static bool
cp_abuf_shade(struct cp_context *cp, const struct pipe_draw_info *info,
              struct cp_abuf *ab, CUdeviceptr positions,
              CUdeviceptr vs_output_buf, unsigned w, unsigned h,
              float vp_scale_x, float vp_scale_y,
              float vp_trans_x, float vp_trans_y,
              uint32_t num_quads, uint32_t num_covered, bool record_colors,
              void *color_data, bool composite, float *t_interp,
              float *t_shade, float *t_composite)
{
   struct cp_screen *screen = cp->screen;
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
   cuMemsetD32Async(counter, (unsigned)want_slots, 1, cp->stream);
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
   };
   cp_fs_interp_setup(cp, info, fs, num_fs_inputs, num_vs_outputs, &interp);

   /* Which merged draw's fragment bindings each shaded slot is to use. Only a
    * batch has more than one answer, and only a shader that reads a constant
    * buffer can tell. */
   CUdeviceptr batch_rows = 0;
   if (cp->fs_batch.ndraws > 1 && fs->reads_const_bufs) {
      /* A batch the interpolator cannot resolve would shade every draw of it
       * with the first one's material. Say so rather than render it. */
      if (!cp->fs_batch.slices) {
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

   void *ip[] = { &interp };
   cp_abuf_mark(ab->ev[11], cp->stream);
   CUresult e = cuLaunchKernel(screen->kernels.abuf_interpolate,
                               (num_quads + 255) / 256, 1, 1, 256, 1, 1,
                               0, cp->stream, ip, NULL);
   cp_abuf_mark(ab->ev[12], cp->stream);
   if (e != CUDA_SUCCESS) {
      fprintf(stderr, "abuffer: cp_abuf_interpolate launch failed (%d)\n", e);
      return false;
   }

   if (!cp_fs_launch_shader(cp, fs, counter, fs_in, fs_in_stride, fs_out,
                            frag_coord, discard_mask, front_face, num_slots,
                            cp_abuf.timing ? ab->ev[13] : 0, batch_rows))
      return false;
   cp_abuf_mark(ab->ev[14], cp->stream);

   if (record_colors && screen->kernels.abuf_scatter_colors) {
      void *p[] = { &fs_out, &fs_out_stride, &dbg_slot, &counter, &num_slots,
                    &ab->capacity, &ab->colors_abuf, &ab->writes_abuf,
                    &ab->dbg };
      cuLaunchKernel(screen->kernels.abuf_scatter_colors,
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
            cp_color_encoding_from_format(cp->framebuffer.cbufs[0].format), 0),
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
static void
cp_draw_execute(struct cp_context *cp, const struct pipe_draw_info *info,
                unsigned drawid_offset,
                const struct pipe_draw_start_count_bias *draws,
                unsigned num_draws, unsigned batch_draws,
                const uint64_t *vs_ubo_table, const uint64_t *fs_ubo_table)
{
   struct cp_screen *screen = cp->screen;
   struct pipe_framebuffer_state *fb = &cp->framebuffer;

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
   if (!fb->nr_cbufs && !fb->zsbuf.texture)
      do { if (getenv("CUDAPIPE_DEBUG_DRAW"))
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
    * arithmetic below divides by. A batch is never instanced, so this is the
    * unbatched draw's own count and nothing else reads it. */
   unsigned tris_per_draw = total_triangles;

   /*
    * A batch concatenates its merged draws, and they need not be the same
    * size: `draws` holds one range per merged draw. That is the whole of what
    * makes the grids below bigger.
    */
   if (batch_draws > 1) {
      total_triangles = 0;
      for (unsigned d = 0; d < batch_draws; d++)
         total_triangles += cp_triangles_for_draw(info->mode, draws[d].count);
   }

   if (total_triangles == 0)
      do { if (getenv("CUDAPIPE_DEBUG_DRAW"))
            fprintf(stderr, "  skipped: no triangles\n");
         return; } while (0);
   unsigned num_triangles = total_triangles;
   /* Near-plane clipping can split triangles, so the rasterizer grid is sized
    * for the post-clip worst case while the count itself lives on the GPU. */
   unsigned rast_num_triangles = total_triangles;

   /* Get the color output surface (may be NULL for depth-only passes) */
   void *color_data = NULL;
   if (fb->nr_cbufs && fb->cbufs[0].texture) {
      struct cp_resource *color_res = cp_resource(fb->cbufs[0].texture);
      color_data = cp_resource_data(color_res);
   }
   if (getenv("CUDAPIPE_DEBUG_DRAW") && !color_data)
      fprintf(stderr, "  color=(nil) reason: nr_cbufs=%u tex=%p data=%p\n",
              fb->nr_cbufs, fb->nr_cbufs ? (void*)fb->cbufs[0].texture : NULL,
              fb->nr_cbufs && fb->cbufs[0].texture ?
                 cp_resource_data(cp_resource(fb->cbufs[0].texture)) : NULL);

   unsigned w = fb->width;
   unsigned h = fb->height;
   unsigned fb_samples = MAX2(cp->fb_samples, 1u);

   /* The visibility buffer only ever holds this draw's triangles: its entries
    * are triangle indices into this draw's vertex arrays, so carrying it
    * across draws would shade one draw's pixels with another's geometry.
    * Occlusion between draws is carried by the depth buffer instead. */
   CUdeviceptr visbuf = cp->visbuf;
   if (!visbuf)
      do { if (getenv("CUDAPIPE_DEBUG_DRAW"))
            fprintf(stderr, "  skipped: no visibility buffer\n");
         return; } while (0);

   /* Clear visbuf to VISBUF_EMPTY (all-ones). cuMemsetD32 fills 32-bit words
    * which is faster than a kernel launch for a bulk fill. */
   cuMemsetD32Async(visbuf, 0xFFFFFFFF, (size_t)w * h * 2 * fb_samples, cp->stream);

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
   if (cp->rasterizer.scissor) {
      clip_x0 = MAX2(clip_x0, (int)cp->scissor.minx);
      clip_y0 = MAX2(clip_y0, (int)cp->scissor.miny);
      clip_x1 = MIN2(clip_x1, (int)cp->scissor.maxx - 1);
      clip_y1 = MIN2(clip_y1, (int)cp->scissor.maxy - 1);
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
         (cp->depth_stencil.depth_func == PIPE_FUNC_GREATER ||
          cp->depth_stencil.depth_func == PIPE_FUNC_GEQUAL),
   };

   /* Names this draw on the timeline for the rest of the function, however it
    * leaves — see CP_NVTX_SCOPE. */
   CP_NVTX_SCOPEF("draw %u tris%s", num_triangles,
                  instance_count > 1 ? " inst" : "");

   /* Marks the start of the draw; the interval it opens is
    * attributed to nothing. */
   cp_stage_end(cp, -1);

   /* Reclaim last draw's scratch and size the arena for this one. */
   cp_scratch_begin(cp);

   /* A shader with no declared inputs needs no vertex buffer: it builds its
    * positions from gl_VertexIndex, which is how a fullscreen pass is drawn. */
   bool has_vs = cp->vs_shader && cp->vs_shader->kernel &&
                 ((cp->num_vertex_buffers > 0 &&
                   cp->vertex_buffers[0].buffer.resource) ||
                  cp->num_vertex_elements == 0);

   const void *ib_base = NULL;
   if (indexed && info->index.resource) {
      struct cp_resource *ib_res = cp_resource(info->index.resource);
      ib_base = cp_resource_data(ib_res);
   }

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
    */
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
      if (cp->num_vertex_buffers > 0 && cp->vertex_buffers[0].buffer.resource) {
         struct cp_resource *vb_res = cp_resource(cp->vertex_buffers[0].buffer.resource);
         void *vb_data = cp_resource_data(vb_res);
         if (vb_data) {
            char *vb_start = (char *)vb_data + cp->vertex_buffers[0].buffer_offset;
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
      void *vb_data2 = NULL;
      if (cp->num_vertex_buffers > 0 && cp->vertex_buffers[0].buffer.resource) {
         struct cp_resource *vb_res2 =
            cp_resource(cp->vertex_buffers[0].buffer.resource);
         vb_data2 = cp_resource_data(vb_res2);
      }

      /* A vertex shader may build its positions from gl_VertexIndex alone and
       * declare no inputs at all, which is how a fullscreen pass is drawn.
       * That draw binds no vertex buffer, and skipping it loses every
       * post-processing and skybox pass. */
      if (vb_data2 || cp->num_vertex_elements == 0) {
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
               slices[d].vert_begin = vbegin;
               slices[d].index_bytes =
                  indexed ? (uint32_t)draws[d].start * info->index_size : 0;
               slices[d].first_vertex =
                  indexed ? (uint32_t)draws[d].index_bias : draws[d].start;
               slices[d].pad = 0;
               vbegin += cp_triangles_for_draw(info->mode, draws[d].count) * 3;
            }
            slices_dev = cp_upload(cp, slices,
                                   (size_t)batch_draws * sizeof(slices[0]));
            if (!slices_dev) { FREE(refs); return; }

            /* One row per assembled vertex, for the shader to pick its
             * uniform bindings with. Only a shader that reads a constant
             * buffer at all has any use for it. */
            if (cp->vs_shader->reads_const_bufs) {
               batch_rows = cp_scratch_alloc_device(cp, (size_t)total_verts * 4);
               if (!batch_rows) { FREE(refs); return; }
            }

            /* The fragment stage searches the same table, from the primitive
             * rather than from the vertex; see cp_fs_interp_args. */
            if (fs_ubo_table)
               cp->fs_batch.slices = slices_dev;
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
            const struct pipe_vertex_element *elem = &cp->vertex_elements[e];
            unsigned vb_idx = elem->vertex_buffer_index;
            if (vb_idx < cp->num_vertex_buffers &&
                cp->vertex_buffers[vb_idx].buffer.resource) {
               struct cp_resource *evb = cp_resource(cp->vertex_buffers[vb_idx].buffer.resource);
               void *evb_data = cp_resource_data(evb);
               if (evb_data)
                  vf_args.vb_bases[vb_idx] = (uint64_t)(uintptr_t)evb_data +
                     cp->vertex_buffers[vb_idx].buffer_offset;
            }
            vf_args.elem_vb_idx[e] = vb_idx;
            vf_args.elem_src_offset[e] = elem->src_offset;
            vf_args.elem_src_stride[e] = elem->src_stride;
            vf_args.elem_attr_size[e] = util_format_get_blocksize(elem->src_format);

            uint32_t nr_chan, chan_bytes, swizzle;
            enum cp_vf_conv conv = cp_vertex_format(elem->src_format, &nr_chan,
                                                    &chan_bytes, &swizzle);
            vf_args.elem_nr_chan[e] = nr_chan;
            vf_args.elem_chan_bytes[e] = chan_bytes;
            vf_args.elem_conv[e] = conv;
            vf_args.elem_swizzle[e] = swizzle;

            vf_args.elem_fill_w[e] = cp_vertex_fill_w(elem->src_format, conv);
            vf_args.elem_instance_divisor[e] = elem->instance_divisor;
         }

         /* Nothing to gather when the shader declares no inputs — but it still
          * runs if the ids are wanted, since deriving those is now its job
          * too and a shader with no inputs may still read gl_VertexIndex. */
         if (vs_input_buf || out_vid || out_iid || batch_rows) {
            if (vs_input_buf)
               cuMemsetD8Async(vs_input_buf, 0, (size_t)total_verts * vs_in_stride, cp->stream);
            void *vf_params[] = { &vf_args };
            cuLaunchKernel(screen->kernels.vertex_fetch,
               (total_verts + 255) / 256, 1, 1, 256, 1, 1,
               0, cp->stream, vf_params, NULL);
         }

         stride = vs_in_stride;

         /* Dump what the GPU fetch actually gathered, which is the quickest way to
          * tell a bad attribute layout from a bad shader. Syncs, so debug only. */
         if (getenv("CUDAPIPE_DEBUG_VFETCH") && vs_input_buf) {
            /* Device-only; fetch the two vertices this prints. */
            cuCtxSynchronize();
            unsigned nfetch = MIN2(2u, total_verts);
            float *in = calloc(nfetch ? nfetch : 1, vs_in_stride);
            if (in)
               cuMemcpyDtoH(in, vs_input_buf, (size_t)nfetch * vs_in_stride);
            for (unsigned e = 0; e < cp->num_vertex_elements && e < 8; e++)
               fprintf(stderr, "  elem%u vb=%u off=%u stride=%u div=%u sz=%u\n",
                       e, cp->vertex_elements[e].vertex_buffer_index,
                       cp->vertex_elements[e].src_offset,
                       cp->vertex_elements[e].src_stride,
                       cp->vertex_elements[e].instance_divisor,
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
            uint32_t draw_params[3];
         } meta = {
            .vcount = total_verts,
            .stride = stride,
            .draw_params = {
               indexed ? (uint32_t)draws[0].index_bias : draws[0].start,
               info->start_instance,
               drawid_offset,
            },
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
         size_t vs_blk_bytes = vs_tbl_off +
            (size_t)batch_draws * CP_ARG_UBO_STRIDE * sizeof(uint64_t);

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
         vs_args_host[7] = (void*)(uintptr_t)(meta_dev + offsetof(struct cp_vs_meta, draw_params));
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

         /* Still written, so that the block reads the same whichever form a
          * stage takes its bindings from. */
         for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
            vs_args_host[CP_ARG_UBO_BASE + i] = cp->vs_ubos[i].buffer;

         cp_upload_end(cp, vs_args_dev, vs_blk, vs_blk_bytes);

         void *vs_arg_ptr = (void*)(uintptr_t)vs_args_dev;
         void *vs_params[] = { &vs_arg_ptr };
         CUresult vs_err = cuLaunchKernel(cp->vs_shader->kernel,
            (total_verts + 255) / 256, 1, 1, 256, 1, 1,
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
               /* Three planes turn one triangle into a hexagon at worst, and
                * the fan over that is four triangles. */
               unsigned max_clipped = num_triangles * 4;
               CUdeviceptr clipped = cp_scratch_alloc_device(
                  cp, (size_t)max_clipped * 3 * out_stride);
               CUdeviceptr clip_count = cp_scratch_alloc_device(cp, 4);

               if (clipped && clip_count) {
                  /*
                   * A batch of blended draws is composited in primitive
                   * order, so its primitives have to *be* in submission order
                   * — which compaction by atomicAdd does not promise. Stable
                   * mode gives every input triangle four slots of its own and
                   * retires the ones it does not fill, so the count is the
                   * whole array and the rasterizer skips the holes on their
                   * zero area. Only for a batch: a single draw keeps the
                   * compacting path, so CUDAPIPE_BATCH_MAX=1 stays
                   * bit-identical to a build without any of this.
                   */
                  bool stable_clip = batch_draws > 1 && cp->blend_enabled;

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
                     if (stable_clip)
                        cp->fs_batch.prim_shift = 2;
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

   if (getenv("CUDAPIPE_DEBUG_DRAW")) {
      fprintf(stderr, "cudapipe: [samples=%u] draw %u tris (%u instances), fb=%ux%u, "
              "vp=[%.0f,%.0f,%.0f,%.0f] stride=%u scale=[%.1f,%.1f] color=%p\n",
              fb_samples, num_triangles, instance_count, w, h, vp_x, vp_y, vp_w, vp_h,
              cp->vertex_stride,
              cp->viewport.scale[0], cp->viewport.scale[1], color_data);
      for (unsigned e = 0; e < cp->num_vertex_elements && e < 4; e++)
         fprintf(stderr, "  elem[%u]: offset=%u fmt=%u vb=%u\n", e,
                 cp->vertex_elements[e].src_offset, cp->vertex_elements[e].src_format,
                 cp->vertex_elements[e].vertex_buffer_index);
   }

   /* 3-stage adaptive rasterize. The queue counters are zeroed inside the pass
    * loop below, which runs for pass 0 as well — clearing them here too was
    * two host calls per draw that the first pass immediately repeated. */
   struct cp_rast_queues rast_queues = {
      .nontrivial = cp->rast_nontrivial,
      .nontrivial_count = cp->rast_nontrivial_count,
      .huge_tiles = cp->rast_huge_tiles,
      .huge_count = cp->rast_huge_count,
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
   bool peel = !retry && color_data && cp->blend_enabled &&
               cp->peel_next && screen->kernels.peel_advance;
   /* A draw can never stack more layers than it has primitives, so a blended
    * draw of two triangles costs two passes rather than the cap. */
   unsigned peel_passes = MIN2((unsigned)CP_BLEND_LAYERS,
                               MAX2(num_triangles, 1u));
   unsigned passes = retry ? CP_DISCARD_LAYERS
                   : peel ? peel_passes : 1;

   if (peel) {
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
   static int bincache = -1;
   if (bincache < 0)
      bincache = getenv("CUDAPIPE_NO_BINCACHE") ? 0 : 1;
   bool cache_queues = peel && bincache;

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
      } else if (cp_color_encoding_from_format(fb->cbufs[0].format) < 0) {
         if (!said++)
            fprintf(stderr, "abuffer: colour format %u has no encoding — "
                    "skipped\n", fb->cbufs[0].format);
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

      /* --- step 1: count --- */
      cuMemsetD32Async(ab->counts, 0, n, cp->stream);
      cuMemsetD32Async(ab->sum3, 0, 3, cp->stream);
      cuMemsetD32Async(cp->rast_counts, 0, 2, cp->stream);
      aa.abuf_mode = CP_ABUF_COUNT;
      rast_queues.mode = CP_QUEUE_FILL;
      cp_abuf_mark(ab->ev[0], cp->stream);
      void *ap[] = { &aa, &rast_queues };
      /* The _abuf specialisations: same rasterizer, compiled with the count
       * and fill branch live. Every other launch in this file uses the plain
       * ones, which have no A-buffer code in them at all. */
      cuLaunchKernel(screen->kernels.rasterize_stage1_abuf,
                     (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, ap, NULL);
      cuLaunchKernel(screen->kernels.rasterize_stage2_abuf,
                     CLAMP((rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                     256, 1, 1, 0, cp->stream, ap, NULL);
      cuLaunchKernel(screen->kernels.rasterize_stage3_abuf, 2048, 1, 1, 64, 1, 1,
                     0, cp->stream, ap, NULL);
      cp_abuf_mark(ab->ev[1], cp->stream);

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
      bool drain_for_count = !ab->frags || !cp_abuf.composite ||
                             !screen->kernels.abuf_clamp_runs;

      if (!drain_for_count) {
         /*
          * Since nobody looks, nothing may index the array outside it. The
          * offsets are a prefix sum, so a count larger than the array leaves
          * later pixels with runs beginning or ending past its end, and the
          * sort and the merge index frags + offsets[p] with no bound of their
          * own. See cp_abuf_clamp_runs — which also records what it cut as
          * overflow, so the draw this happens to is refused below and rendered
          * by the peel loop rather than composited short.
          */
         unsigned nn = (unsigned)n;
         void *p[] = { &ab->counts, &ab->offsets, &ab->sum3, &nn,
                       &ab->capacity, &ab->overflow };
         cuLaunchKernel(screen->kernels.abuf_clamp_runs, 1024, 1, 1, 256, 1, 1,
                        0, cp->stream, p, NULL);
      } else {
         cuStreamSynchronize(cp->stream);
         cuMemcpyDtoH(&abuf_total, ab->sum3, sizeof(uint32_t));

         /* Whether the arrays are big enough for this draw, and whether they
          * should be made bigger before the next one. Both live in one place;
          * see cp_abuf_size_arrays(). */
         if (abuf_total > ab->peak)
            ab->peak = abuf_total;
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
      cuMemsetD32Async(cp->rast_counts, 0, 2, cp->stream);
      aa.abuf_mode = CP_ABUF_FILL;
      rast_queues.mode = CP_QUEUE_FILL;
      cp_abuf_mark(ab->ev[3], cp->stream);
      void *ap[] = { &aa, &rast_queues };
      cuLaunchKernel(screen->kernels.rasterize_stage1_abuf,
                     (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                     0, cp->stream, ap, NULL);
      cuLaunchKernel(screen->kernels.rasterize_stage2_abuf,
                     CLAMP((rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                     256, 1, 1, 0, cp->stream, ap, NULL);
      cuLaunchKernel(screen->kernels.rasterize_stage3_abuf, 2048, 1, 1, 64, 1, 1,
                     0, cp->stream, ap, NULL);
      cp_abuf_mark(ab->ev[4], cp->stream);

      /* --- step 4: sort. The worklist build is inside this measurement: it
       * is a prerequisite of the sort as written, not a separate step. --- */
      {
         unsigned nn = (unsigned)n, min2 = 2;
         cuMemsetD32Async(ab->list_count, 0, 1, cp->stream);
         void *wp[] = { &ab->counts, &nn, &min2, &ab->list, &ab->list_count };
         cuLaunchKernel(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, wp, NULL);
         void *sp[] = { &ab->frags, &ab->offsets, &ab->counts, &ab->list,
                        &ab->list_count, &ab->long_runs };
         cuLaunchKernel(screen->kernels.abuf_sort, 4096, 1, 1, 256, 1, 1,
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
         cuLaunchKernel(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
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
         cuLaunchKernel(screen->kernels.abuf_block_worklist,
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
         cuLaunchKernel(screen->kernels.abuf_quad_count, 1024, 1, 1, 32, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab->ev[9], cp->stream);
      cp_abuf_scan_n(cp, screen, ab->blk_counts, ab->blk_offsets, ab->bsum1,
                     ab->bsum1x, ab->bsum2, ab->bsum2x, ab->bsum3, nblocks,
                     ab->bnb1, ab->bnb2, ab->bnb3);
      {
         /* Slots the merge cannot place stay 0xFFFFFFFF and are skipped by the
          * composite, so a fill that fell short loses a layer rather than
          * reading whatever the last draw left there. Only the run actually in
          * use is cleared — by a kernel rather than a memset, because the
          * length of that run is now only on the device. */
         if (ab->shade_slot && screen->kernels.abuf_clear_slots) {
            void *p[] = { &ab->shade_slot, &ab->sum3, &ab->capacity };
            cuLaunchKernel(screen->kernels.abuf_clear_slots, 1024, 1, 1,
                           256, 1, 1, 0, cp->stream, p, NULL);
         } else if (ab->shade_slot) {
            /* Without it, the whole array: clearing more than the draw uses is
             * only slower, and clearing less would let the composite read a
             * slot map the last draw wrote. */
            cuMemsetD32Async(ab->shade_slot, 0xFFFFFFFF,
                             abuf_total ? abuf_total : ab->capacity,
                             cp->stream);
         }
         void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                       &ab->blk_list, &ab->blk_list_count, &ab->blk_offsets,
                       &ab->quad_prim, &ab->quad_mask, &ab->peel_mask,
                       &ab->quad_block, &ab->shade_slot, &ab->quad_capacity,
                       &ab->quad_overflow };
         cuLaunchKernel(screen->kernels.abuf_quad_fill, 1024, 1, 1, 32, 1, 1,
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
      if (cp_abuf.composite) {
         uint32_t q[2] = { 0, 0 }, c3[3] = { 0, 0, 0 };
         cuStreamSynchronize(cp->stream);
         cuMemcpyDtoH(q, ab->bsum3, sizeof(q));
         cuMemcpyDtoH(c3, ab->sum3, sizeof(c3));
         /* Free, since the drain is already paid for: the composite's grid is
          * then the covered pixels rather than the framebuffer, which on a
          * sample covering 4% of it is 140 blocks instead of 3,600. */
         cuMemcpyDtoH(&abuf_covered, ab->clist_count, sizeof(abuf_covered));
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
         cuMemsetD32Async(cp->rast_counts, 0, 2, cp->stream);

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
      cuLaunchKernel(screen->kernels.rasterize_stage2,
         s2_blocks, 1, 1, 256, 1, 1,
         0, cp->stream, s2_params, NULL);

      /* Stage 3: block per tile, grid-strided over the huge-tile queue */
      void *s3_params[] = { &rast_args, &rast_queues };
      cuLaunchKernel(screen->kernels.rasterize_stage3,
         s3_blocks, 1, 1, 64, 1, 1,
         0, cp->stream, s3_params, NULL);

      if (rast_err != CUDA_SUCCESS && getenv("CUDAPIPE_DEBUG_DRAW"))
         fprintf(stderr, "  rasterize launch failed: %d\n", rast_err);
      cp_nvtx_pop();   /* raster */
      cp_stage_end(cp, CP_STAGE_RASTERIZE);

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
            cuLaunchKernel(screen->kernels.abuf_peel_log, (nn + 255) / 256,
                           1, 1, 256, 1, 1, 0, cp->stream, p, NULL);
         }
         unsigned dlayers = CP_BLEND_LAYERS;
         if (abuf_deep_n && pass < dlayers) {
            void *p[] = { &visbuf, &ab->deep_list, &ab->deep_log, &dlayers,
                          &pass, &abuf_deep_n };
            cuLaunchKernel(screen->kernels.abuf_peel_log_list,
                           (abuf_deep_n + 255) / 256, 1, 1, 256, 1, 1,
                           0, cp->stream, p, NULL);
         }
      }

      /* Shade every covered pixel by running the fragment shader on the GPU:
       * interpolate its inputs, launch it, then blend its output into the
       * attachment. */
      if (color_data)
         cp_shade_fragments(cp, info, visbuf, rast_args.positions, vs_output_buf,
                            num_triangles, w, h, color_data,
                            vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y,
                            retry ? cp->reject : 0,
                            retry ? cp->resolved : 0,
                            pass);

      if (peel) {
         /* Step past what this pass blended, and stop once the interval finds
          * nothing left to composite — which is the common case at pass 2,
          * since most blended draws do not overlap themselves. */
         void *pa_params[] = { &visbuf, &cp->peel_next, &w, &h };
         cuLaunchKernel(screen->kernels.peel_advance,
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
                                &t_qinterp, &t_qshade, &t_composite);
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
 * ---------------------------------------------------------------------------
 * Draw batching
 * ---------------------------------------------------------------------------
 *
 * Why this is allowed to reorder anything at all.
 *
 * A batch is submitted as one draw, so the fragments of its member draws reach
 * the visibility buffer interleaved rather than draw by draw. Three things
 * make that produce the same pixels:
 *
 *  - With blending off, the visibility buffer's atomicMin keeps the nearest
 *    fragment per pixel and discards the rest. That is a minimum, so it does
 *    not depend on the order the fragments arrived in — which is why no
 *    submission-order sort key is needed here and why blending, whose result
 *    does depend on order, is refused outright.
 *  - Depth has to be tested and written, with a function that agrees with that
 *    minimum. Run draw by draw, the second draw tests against the first's
 *    depth, so the pixel ends up with the nearest fragment; run as a batch,
 *    the visibility buffer selects the same fragment directly. With the test
 *    off, or the write off, or a function like ALWAYS, the draw-by-draw answer
 *    is "the last one" instead, which a batch cannot reproduce.
 *  - A shader that discards is refused, because a discarded fragment has
 *    already displaced the one behind it and the retry machinery is per draw.
 *
 * What a batch is then allowed to vary is two things, and both are carried as
 * a table with one row per merged draw:
 *
 *  - The vertex stage's uniform bindings — the same buffer at a different
 *    dynamic offset, which is what dynamicuniformbuffer's 125 cubes a frame
 *    differ in. The vertex shader picks its row out of the table at
 *    CP_ARG_SLOT_UBO_TABLE.
 *  - The index range. Draws of different sizes out of one buffer concatenate,
 *    and cp_vertex_fetch searches a table of slices to find which draw a
 *    thread's vertex came from — see struct cp_draw_slice. Without that,
 *    bloom's 154 draws a frame and vulkanscene's 19 merge none of themselves,
 *    because no two of them replay the same range.
 *
 * The *fragment* stage's bindings are still a merge condition, so a scene
 * whose meshes carry a texture each still batches nothing. That would need a
 * per-triangle draw index threaded through the clipper, which is the next
 * increment and a larger one.
 */

/* Fill in everything two draws must agree on. See struct cp_batch_key. */
static void
cp_batch_build_key(struct cp_context *cp, const struct pipe_draw_info *info,
                   unsigned drawid_offset,
                   const struct pipe_draw_start_count_bias *draws,
                   struct cp_batch_key *key, bool blended)
{
   struct pipe_framebuffer_state *fb = &cp->framebuffer;

   memset(key, 0, sizeof(*key));

   key->vs = cp->vs_shader;
   key->fs = cp->fs_shader;

   key->cbuf_texture = fb->nr_cbufs ? fb->cbufs[0].texture : NULL;
   key->zs_texture = fb->zsbuf.texture;
   key->color_data = (fb->nr_cbufs && fb->cbufs[0].texture)
      ? cp_resource_data(cp_resource(fb->cbufs[0].texture)) : NULL;
   key->visbuf = cp->visbuf;
   key->depthbuf = cp->depthbuf;
   key->fb_w = fb->width;
   key->fb_h = fb->height;
   key->fb_nr_cbufs = fb->nr_cbufs;
   key->fb_samples = cp->fb_samples;
   key->cbuf_format = fb->nr_cbufs ? (uint32_t)fb->cbufs[0].format : 0;

   key->mode = info->mode;
   key->index_size = info->index_size;
   key->instance_count = info->instance_count;
   key->start_instance = info->start_instance;
   key->drawid_offset = drawid_offset;
   key->index_resource = info->index_size ? info->index.resource : NULL;
   /*
    * The range is normally not a merge condition — a batch carries one slice
    * per draw and the fetch kernel resolves which is which. It becomes one
    * again for a shader that reads gl_BaseVertex, gl_BaseInstance or
    * gl_DrawID: those come from a single triple in the argument block, so two
    * draws that would be told different things cannot share a launch.
    */
   if (cp->vs_shader && cp->vs_shader->reads_draw_params) {
      key->draw_start = draws[0].start;
      key->draw_count = draws[0].count;
      key->draw_index_bias = draws[0].index_bias;
   }

   key->viewport = cp->viewport;
   key->scissor = cp->scissor;
   key->rasterizer = cp->rasterizer;
   key->depth_stencil = cp->depth_stencil;
   key->blend_state = cp->blend_state;
   key->blend_enabled = cp->blend_enabled;

   memcpy(key->vertex_elements, cp->vertex_elements,
          sizeof(key->vertex_elements));
   key->num_vertex_elements = cp->num_vertex_elements;
   key->vertex_stride = cp->vertex_stride;
   key->num_vertex_buffers = cp->num_vertex_buffers;
   for (unsigned i = 0; i < cp->num_vertex_buffers && i < 16; i++) {
      key->vertex_buffers[i].resource = cp->vertex_buffers[i].buffer.resource;
      key->vertex_buffers[i].offset = cp->vertex_buffers[i].buffer_offset;
   }

   key->num_vs_ubos = cp->num_vs_ubos;
   /*
    * Only where the fragment shader can see them. A shader that reads no
    * constant buffer at all — multithreading's, which takes everything it
    * needs from its varyings — cannot tell what is bound in those slots, and
    * lavapipe re-binds the push constant range for both stages on every draw.
    * Keying on bindings the shader never loads would break every batch in that
    * sample for no reason.
    */
   /*
    * A *blended* batch carries these per draw instead, in the table behind the
    * fragment shader's argument block — which is what makes 482 blended draws
    * a frame merge at all, and is why they are absent from the key here rather
    * than compared. See cp_batch_abuf_ok() and cp_fs_launch_shader().
    */
   if (cp->fs_shader && cp->fs_shader->reads_const_bufs && !blended) {
      key->num_fs_ubos = cp->num_fs_ubos;
      for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++) {
         key->fs_ubos[i] = cp->fs_ubos[i].buffer;
         key->fs_ubo_sizes[i] = cp->fs_ubos[i].buffer_size;
      }
   }
   key->sampler_table = cp->sampler_table;
   key->num_samplers = cp->num_samplers;
}

/*
 * Which fields of the key two draws disagree on.
 *
 * memcmp gives one lumped verdict — "state or geometry" — which is exactly the
 * wrong granularity when the question is what stops a sample batching. Under
 * CUDAPIPE_DEBUG_BATCHDIFF the mismatch is reported field by field, so the
 * answer can be counted rather than guessed at. Off by default and never on the
 * fast path: the caller still decides with memcmp.
 */
static void
cp_batch_key_report_diff(const struct cp_batch_key *a,
                         const struct cp_batch_key *b)
{
#define F(name) { #name, offsetof(struct cp_batch_key, name), \
                  sizeof(((struct cp_batch_key *)0)->name) }
   static const struct { const char *name; size_t off, size; } fields[] = {
      F(vs), F(fs),
      F(cbuf_texture), F(zs_texture), F(color_data), F(visbuf), F(depthbuf),
      F(fb_w), F(fb_h), F(fb_nr_cbufs), F(fb_samples), F(cbuf_format),
      F(mode), F(index_size), F(instance_count), F(start_instance),
      F(drawid_offset), F(index_resource),
      F(draw_start), F(draw_count), F(draw_index_bias),
      F(viewport), F(scissor), F(rasterizer), F(depth_stencil),
      F(blend_state), F(blend_enabled),
      F(vertex_elements), F(num_vertex_elements), F(vertex_stride),
      F(num_vertex_buffers), F(vertex_buffers),
      F(num_fs_ubos), F(num_vs_ubos), F(fs_ubos), F(fs_ubo_sizes),
      F(sampler_table), F(num_samplers),
   };
#undef F
   char line[512];
   size_t n = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(fields); i++) {
      if (!memcmp((const char *)a + fields[i].off,
                  (const char *)b + fields[i].off, fields[i].size))
         continue;
      int w = snprintf(line + n, sizeof(line) - n, "%s%s",
                       n ? "," : "", fields[i].name);
      if (w < 0 || (size_t)w >= sizeof(line) - n)
         break;
      n += w;
   }
   fprintf(stderr, "cudapipe: batchdiff %s\n", n ? line : "(none)");
}


/*
 * Whether this draw may be held back at all.
 *
 * Everything here is a property of the draw and of the state bound for it, not
 * of what came before, so a draw either can join a batch or cannot — the key
 * above then decides which batch.
 */
static bool
cp_batch_structural(struct cp_context *cp, const struct pipe_draw_info *info,
                    const struct pipe_draw_indirect_info *indirect,
                    const struct pipe_draw_start_count_bias *draws,
                    unsigned num_draws)
{
   /* The pipeline the batched path takes: a compiled vertex shader over a
    * triangle list, with the topology resolved on the device. Anything the
    * host has to expand into a refs table is left alone. */
   if (!cp->vs_shader || !cp->vs_shader->kernel || !cp->fs_shader ||
       !cp->fs_shader->kernel)
      return false;
   if (info->mode != MESA_PRIM_TRIANGLES || num_draws != 1 || indirect)
      return false;
   if (MAX2(info->instance_count, 1u) != 1)
      return false;
   if (info->has_user_indices)
      return false;
   if (info->index_size && !info->index.resource)
      return false;

   /* A vertex buffer is what the batch replays; a shader building its
    * positions from gl_VertexIndex alone has nothing to gain and is left on
    * the single-draw path. */
   if (!cp->num_vertex_buffers || !cp->vertex_buffers[0].buffer.resource)
      return false;

   /* Framebuffer and the buffers the stages need. */
   if (!cp->framebuffer.nr_cbufs || !cp->framebuffer.cbufs[0].texture ||
       !cp->visbuf || !cp->depthbuf)
      return false;

   /*
    * A binding copied out of a user pointer lands in the same device buffer
    * every time it is set, so two draws can hold the identical address and
    * mean different bytes. Nothing in the key can see that, so refuse the
    * draw rather than merge it wrongly.
    */
   for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
      if (cp->vs_ubos[i].user_copy)
         return false;
   if (cp->fs_shader->reads_const_bufs)
      for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
         if (cp->fs_ubos[i].user_copy)
            return false;

   /* One draw over the cap is worth nothing to a batch and would grow the
    * clipper's output buffer past what the arena will hand out. */
   if (cp_triangles_for_draw(info->mode, draws[0].count) == 0 ||
       cp_triangles_for_draw(info->mode, draws[0].count) > CP_MAX_BATCH_TRIS)
      return false;

   return true;
}

/*
 * The opaque merge condition: a result that cannot depend on the order the
 * draws arrived in, because the visibility buffer resolves it with atomicMin.
 */
static bool
cp_batch_order_free(struct cp_context *cp)
{
   if (cp->blend_enabled || cp->fs_shader->uses_discard)
      return false;
   if (!cp->depth_stencil.depth_enabled || !cp->depth_stencil.depth_writemask)
      return false;
   switch (cp->depth_stencil.depth_func) {
   case PIPE_FUNC_LESS:
   case PIPE_FUNC_LEQUAL:
   case PIPE_FUNC_GREATER:
   case PIPE_FUNC_GEQUAL:
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
static bool
cp_batch_abuf_ok(struct cp_context *cp)
{
   struct cp_screen *screen = cp->screen;
   struct pipe_framebuffer_state *fb = &cp->framebuffer;

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
   if (fb->nr_cbufs != 1 || !fb->cbufs[0].texture ||
       cp_color_encoding_from_format(fb->cbufs[0].format) < 0)
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

static bool
cp_batch_eligible(struct cp_context *cp, const struct pipe_draw_info *info,
                  const struct pipe_draw_indirect_info *indirect,
                  const struct pipe_draw_start_count_bias *draws,
                  unsigned num_draws, bool *blended)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("CUDAPIPE_NO_BATCH") ? 0 : 1;
   if (!enabled)
      return false;

   if (!cp_batch_structural(cp, info, indirect, draws, num_draws))
      return false;

   if (cp_batch_order_free(cp)) {
      *blended = false;
      return true;
   }
   if (cp_abuf_batch_enabled() && cp_batch_abuf_ok(cp)) {
      *blended = true;
      return true;
   }
   return false;
}

/* Snapshot this draw's index range and vertex-stage bindings as the next row
 * of the batch's tables. */
static void
cp_batch_record(struct cp_context *cp,
                const struct pipe_draw_start_count_bias *draw, unsigned tris)
{
   uint64_t *row = cp->batch.vs_ubos +
      (size_t)cp->batch.ndraws * CP_ARG_UBO_STRIDE;
   memset(row, 0, CP_ARG_UBO_STRIDE * sizeof(*row));
   for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
      row[i] = (uint64_t)(uintptr_t)cp->vs_ubos[i].buffer;

   /* The fragment stage's, for a blended batch — the bindings the key stopped
    * comparing. Recorded now, because by the time the batch runs the next
    * draw's have been bound over them. */
   if (cp->batch.blended) {
      uint64_t *frow = cp->batch.fs_ubos +
         (size_t)cp->batch.ndraws * CP_ARG_UBO_STRIDE;
      memset(frow, 0, CP_ARG_UBO_STRIDE * sizeof(*frow));
      for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
         frow[i] = (uint64_t)(uintptr_t)cp->fs_ubos[i].buffer;
   }

   cp->batch.draws[cp->batch.ndraws] = *draw;
   cp->batch.tris += tris;
   cp->batch.ndraws++;
}

void
cp_batch_flush(struct cp_context *cp)
{
   cp_batch_flush_why(cp, "a readback, a clear or a flush");
}

void
cp_batch_flush_why(struct cp_context *cp, const char *why)
{
   if (!cp->batch.pending)
      return;

   /* Resolved once: a sample with a hundred single-draw batches a frame calls
    * this on the path the batching is meant to make cheaper. */
   static int debug = -1;
   if (debug < 0)
      debug = (getenv("CUDAPIPE_DEBUG_BATCH") ? 1 : 0) |
              (getenv("CUDAPIPE_DEBUG_DRAW") ? 2 : 0);

   if (debug & 1)
      fprintf(stderr, "cudapipe: batch of %u ends: %s\n", cp->batch.ndraws, why);

   unsigned ndraws = cp->batch.ndraws;
   bool blended = cp->batch.blended;
   /* Cleared first: cp_draw_execute() runs a whole frame's worth of driver
    * code and nothing in it may see a batch that is already on its way. */
   cp->batch.pending = false;
   cp->batch.ndraws = 0;
   cp->batch.tris = 0;
   cp->batch.blended = false;

   if (debug & 2) {
      fprintf(stderr, "cudapipe: batch of %u draws\n", ndraws);
      for (unsigned d = 0; d < MIN2(ndraws, 4u); d++) {
         fprintf(stderr, "  row %u:", d);
         for (unsigned i = 0; i < cp->batch.key.num_vs_ubos; i++)
            fprintf(stderr, " %p",
                    (void *)(uintptr_t)cp->batch.vs_ubos[d * CP_ARG_UBO_STRIDE + i]);
         fprintf(stderr, "\n");
      }
   }

   cp_draw_execute(cp, &cp->batch.info, cp->batch.drawid_offset,
                   cp->batch.draws, 1, ndraws, cp->batch.vs_ubos,
                   blended ? cp->batch.fs_ubos : NULL);
}

static void
cp_draw_vbo(struct pipe_context *ctx, const struct pipe_draw_info *info,
            unsigned drawid_offset,
            const struct pipe_draw_indirect_info *indirect,
            const struct pipe_draw_start_count_bias *draws,
            unsigned num_draws)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_screen *screen = cp->screen;

   if (!screen->kernels.initialized || !screen->kernels.rasterize_triangles)
      return;
   if (num_draws == 0 || draws[0].count == 0)
      return;

   cuCtxSetCurrent(screen->cuda_ctx);

   bool blended = false;
   if (cp_batch_eligible(cp, info, indirect, draws, num_draws, &blended)) {
      struct cp_batch_key key;
      cp_batch_build_key(cp, info, drawid_offset, draws, &key, blended);

      unsigned tris = cp_triangles_for_draw(info->mode, draws[0].count);

      /* The cap, overridable so that a suspect batch can be bisected by size
       * without a rebuild — 1 exercises the whole batched path on a batch of
       * one, which is the case that has to stay bit-identical. */
      static int cap = -1;
      if (cap < 0) {
         const char *e = getenv("CUDAPIPE_BATCH_MAX");
         cap = e && *e ? CLAMP(atoi(e), 1, CP_MAX_BATCH_DRAWS)
                       : CP_MAX_BATCH_DRAWS;
      }

      if (cp->batch.pending) {
         const char *why = NULL;
         if (memcmp(&key, &cp->batch.key, sizeof(key))) {
            why = "the next draw differs in state or geometry";
            static int diffdbg = -1;
            if (diffdbg < 0)
               diffdbg = getenv("CUDAPIPE_DEBUG_BATCHDIFF") ? 1 : 0;
            if (diffdbg)
               cp_batch_key_report_diff(&key, &cp->batch.key);
         }
         else if (cp->batch.ndraws >= (unsigned)cap)
            why = "the draw cap";
         else if (cp->batch.tris + tris > CP_MAX_BATCH_TRIS)
            why = "the triangle cap";

         if (!why) {
            cp_batch_record(cp, &draws[0], tris);
            return;
         }
         cp_batch_flush_why(cp, why);
      }

      /* Whatever was pending has gone; this draw opens the next batch. */
      cp->batch.key = key;
      cp->batch.info = *info;
      cp->batch.drawid_offset = drawid_offset;
      cp->batch.pending = true;
      /* Set before the first row is recorded: cp_batch_record() reads it to
       * decide whether the fragment bindings have to be snapshotted too. */
      cp->batch.blended = blended;
      cp_batch_record(cp, &draws[0], tris);
      return;
   }

   cp_batch_flush_why(cp, "the next draw cannot be batched");
   cp_draw_execute(cp, info, drawid_offset, draws, num_draws, 1, NULL, NULL);
}

static void
cp_launch_grid(struct pipe_context *ctx, const struct pipe_grid_info *info)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_shader_binary *bin = cp->compute_shader;

   /* A dispatch may read what the held-back draws were going to write. */
   cp_batch_flush_why(cp, "a compute dispatch");

   if (!bin || !bin->kernel)
      return;

   unsigned grid[3] = { info->grid[0], info->grid[1], info->grid[2] };

   /* Handle indirect dispatch — read grid from buffer */
   if (info->indirect) {
      struct cp_resource *ind_res = cp_resource(info->indirect);
      void *ind_data = cp_resource_data(ind_res);
      if (ind_data) {
         uint32_t *dims = (uint32_t *)((char *)ind_data + info->indirect_offset);
         grid[0] = dims[0];
         grid[1] = dims[1];
         grid[2] = dims[2];
      }
   }

   if (grid[0] == 0 || grid[1] == 0 || grid[2] == 0)
      return;
   if (info->block[0] == 0 || info->block[1] == 0 || info->block[2] == 0)
      return;

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* Debug: print bound UBO/SSBO pointers */
   if (getenv("CUDAPIPE_DEBUG_LAUNCH")) {
      for (unsigned i = 0; i < cp->num_compute_ubos; i++) {
         fprintf(stderr, "  UBO[%u] = %p (size %u)", i, cp->compute_ubos[i].buffer, cp->compute_ubos[i].buffer_size);
         if (cp->compute_ubos[i].buffer && cp->compute_ubos[i].buffer_size >= 16) {
            uint64_t *addrs = (uint64_t *)cp->compute_ubos[i].buffer;
            fprintf(stderr, " u64s: [%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx]",
               addrs[0], addrs[1], addrs[2], addrs[3], addrs[4], addrs[5], addrs[6], addrs[7]);
         }
         fprintf(stderr, "\n");
      }
      for (unsigned i = 0; i < cp->num_compute_ssbos; i++)
         fprintf(stderr, "  SSBO[%u] = %p (size %u)\n", i, cp->compute_ssbos[i].buffer, cp->compute_ssbos[i].buffer_size);
   }

   /*
    * Build the argument buffer layout (array of pointers):
    *   [0]    = pointer to grid_size {gridX, gridY, gridZ}
    *   [1]    = reserved
    *   [2..17]  = SSBO pointers (16 slots)
    *   [18..33] = UBO pointers (16 slots)
    *
    * This is allocated as managed memory so the GPU can access it.
    */
   uint32_t grid_size[3] = { grid[0], grid[1], grid[2] };

   /*
    * Both blocks go into the upload arena by DMA, the way the draw path's
    * argument blocks do. They used to be a cuMemAllocManaged pair freed after
    * the dispatch, which is what the cuCtxSynchronize below them was for —
    * the memory could not be released until the kernel reading it had
    * finished. Nothing here needs the host to wait: the arena is reclaimed in
    * bulk, and a managed block the host has just written is the page-fault
    * stall cp_upload() exists to avoid.
    */
   CUdeviceptr grid_dev = cp_upload(cp, grid_size, sizeof(grid_size));
   if (!grid_dev)
      return;

   void *arg_ptrs_host[34] = {0};
   arg_ptrs_host[0] = (void *)(uintptr_t)grid_dev;
   arg_ptrs_host[1] = NULL;

   for (unsigned i = 0; i < CP_MAX_SHADER_BUFFERS; i++)
      arg_ptrs_host[2 + i] = cp->compute_ssbos[i].buffer;

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++)
      arg_ptrs_host[18 + i] = cp->compute_ubos[i].buffer;

   CUdeviceptr args_dev = cp_upload(cp, arg_ptrs_host, sizeof(arg_ptrs_host));
   if (!args_dev)
      return;

   void *args_ptr_val = (void *)(uintptr_t)args_dev;
   void *kernel_params[] = { &args_ptr_val };

   if (getenv("CUDAPIPE_DEBUG_LAUNCH")) {
      fprintf(stderr, "  args_dev=%p arg_ptrs_host[19]=%p (UBO[1])\n",
              (void*)(uintptr_t)args_dev, arg_ptrs_host[19]);
   }

   CUresult err = cuLaunchKernel(
      bin->kernel,
      grid[0], grid[1], grid[2],
      info->block[0], info->block[1], info->block[2],
      bin->shared_size, cp->stream, kernel_params, NULL);

   if (err != CUDA_SUCCESS)
      fprintf(stderr, "cudapipe: cuLaunchKernel failed (%d) grid=[%u,%u,%u] block=[%u,%u,%u]\n",
              err, info->grid[0], info->grid[1], info->grid[2],
              info->block[0], info->block[1], info->block[2]);

   /* No sync: the argument blocks live in the upload arena, which is
    * reclaimed in bulk once the work reading it has finished, rather than
    * being freed here. */
}

static void
cp_flush(struct pipe_context *ctx, struct pipe_fence_handle **fence,
         unsigned flags)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* Before the fence: a batch still being held back has not been submitted,
    * so anything waiting on this would be waiting for work that was never
    * queued. */
   cp_batch_flush(cp);

   if (fence) {
      CUevent event;
      cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
      cuEventRecord(event, 0);
      *fence = (struct pipe_fence_handle *)(uintptr_t)event;
   }

   /*
    * Sync and reclaim the scratch arenas.
    *
    * The plan asks for this drain to go, on the grounds that it is the host
    * waiting for work it could be queueing behind — and the premise is sound:
    * the GPU is no longer the "99% utilized" the original comment claimed,
    * but 66-80% on the launch-heavy samples by tests/cp_gpu_busy.sh.
    *
    * Removing it is nonetheless a regression, measured: +0.4% over the sweep,
    * pbribl +14.5%, negativeviewportheight +9.2%, texture +8.5%,
    * texturecubemap +6.4%. Correctness and memory were both fine — the peak
    * stayed at 3.4 GB of 32, because cp_scratch_alloc() still reclaims at
    * CP_SCRATCH_RECLAIM_BYTES.
    *
    * That reclaim is exactly why it loses. Resetting here costs a drain and
    * nothing else, because the arena is rewound to zero and the next frame
    * reuses the same pages. Deferring it until an arena has handed out a
    * gigabyte means the reclaim path runs instead, and that one frees and
    * reallocates the overflow buffers — cuMemAlloc and cuMemFree are tens of
    * microseconds each where a drain of an almost-idle queue is a few. Trading
    * many cheap syncs for occasional expensive reallocation is the wrong way
    * round.
    *
    * So this stays until the arena can be reclaimed without reallocating,
    * which is a change to cp_scratch_alloc() rather than to this line.
    */
   cuCtxSynchronize();
   cp_scratch_reset(cp);
   cp_nvtx_mark("flush");
}

/* Stub state functions - store state for use at draw time */

static void *
cp_create_blend_state(struct pipe_context *ctx,
                      const struct pipe_blend_state *state)
{
   struct pipe_blend_state *copy = MALLOC_STRUCT(pipe_blend_state);
   if (copy)
      *copy = *state;
   return copy;
}

static void
cp_bind_blend_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct pipe_blend_state next;
   if (state)
      next = *(struct pipe_blend_state *)state;
   else
      memset(&next, 0, sizeof(next));
   if (memcmp(&cp->blend_state, &next, sizeof(next)))
      cp_batch_flush_why(cp, "blend state");

   if (state) {
      cp->blend_state = next;
      cp->blend_enabled = cp->blend_state.rt[0].blend_enable;
      if (cp->gpu_state) {
         const struct pipe_rt_blend_state *rt = &cp->blend_state.rt[0];
         cp->gpu_state->blend_enable = rt->blend_enable;
         cp->gpu_state->rgb_src_factor = rt->rgb_src_factor;
         cp->gpu_state->rgb_dst_factor = rt->rgb_dst_factor;
         cp->gpu_state->rgb_func = rt->rgb_func;
         cp->gpu_state->alpha_src_factor = rt->alpha_src_factor;
         cp->gpu_state->alpha_dst_factor = rt->alpha_dst_factor;
         cp->gpu_state->alpha_func = rt->alpha_func;
         cp->gpu_state->colormask = rt->colormask ? rt->colormask : 0xF;
      }
   } else {
      memset(&cp->blend_state, 0, sizeof(cp->blend_state));
      cp->blend_enabled = false;
      if (cp->gpu_state) {
         cp->gpu_state->blend_enable = 0;
         cp->gpu_state->colormask = 0xF;
      }
   }
}

static void
cp_delete_blend_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

static void *
cp_create_rasterizer_state(struct pipe_context *ctx,
                           const struct pipe_rasterizer_state *state)
{
   struct pipe_rasterizer_state *copy = MALLOC_STRUCT(pipe_rasterizer_state);
   if (copy)
      *copy = *state;
   return copy;
}

static void
cp_bind_rasterizer_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct pipe_rasterizer_state next;
   if (state)
      next = *(struct pipe_rasterizer_state *)state;
   else
      memset(&next, 0, sizeof(next));
   if (memcmp(&cp->rasterizer, &next, sizeof(next)))
      cp_batch_flush_why(cp, "rasterizer state");
   cp->rasterizer = next;
}

static void
cp_delete_rasterizer_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

static void *
cp_create_depth_stencil_alpha_state(struct pipe_context *ctx,
                                    const struct pipe_depth_stencil_alpha_state *state)
{
   struct pipe_depth_stencil_alpha_state *copy =
      MALLOC_STRUCT(pipe_depth_stencil_alpha_state);
   if (copy)
      *copy = *state;
   return copy;
}

static void
cp_bind_depth_stencil_alpha_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct pipe_depth_stencil_alpha_state next;
   if (state)
      next = *(struct pipe_depth_stencil_alpha_state *)state;
   else
      memset(&next, 0, sizeof(next));
   if (memcmp(&cp->depth_stencil, &next, sizeof(next)))
      cp_batch_flush_why(cp, "depth/stencil state");

   if (state) {
      cp->depth_stencil = next;
      if (cp->gpu_state) {
         cp->gpu_state->depth_test = cp->depth_stencil.depth_enabled;
         cp->gpu_state->depth_func = cp->depth_stencil.depth_func;
         cp->gpu_state->depth_write = cp->depth_stencil.depth_writemask;
         cp->gpu_state->depth_key_invert = cp->depth_stencil.depth_enabled &&
            (cp->depth_stencil.depth_func == PIPE_FUNC_GREATER ||
             cp->depth_stencil.depth_func == PIPE_FUNC_GEQUAL);
      }
   } else {
      memset(&cp->depth_stencil, 0, sizeof(cp->depth_stencil));
      if (cp->gpu_state) {
         cp->gpu_state->depth_test = 0;
         cp->gpu_state->depth_write = 0;
      }
   }
}

static void
cp_delete_depth_stencil_alpha_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

struct cp_vertex_elements_state {
   struct pipe_vertex_element elements[16];
   unsigned num_elements;
   unsigned stride;
};

static void *
cp_create_vertex_elements_state(struct pipe_context *ctx, unsigned num_elements,
                                const struct pipe_vertex_element *elements)
{
   struct cp_vertex_elements_state *state = CALLOC_STRUCT(cp_vertex_elements_state);
   state->num_elements = num_elements;
   memcpy(state->elements, elements, num_elements * sizeof(struct pipe_vertex_element));
   if (num_elements > 0)
      state->stride = elements[0].src_stride;
   return state;
}

static void
cp_bind_vertex_elements_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (state) {
      struct cp_vertex_elements_state *ve = (struct cp_vertex_elements_state *)state;
      if (ve->num_elements != cp->num_vertex_elements ||
          ve->stride != cp->vertex_stride ||
          memcmp(cp->vertex_elements, ve->elements,
                 ve->num_elements * sizeof(struct pipe_vertex_element)))
         cp_batch_flush_why(cp, "vertex elements");
      memcpy(cp->vertex_elements, ve->elements, ve->num_elements * sizeof(struct pipe_vertex_element));
      cp->num_vertex_elements = ve->num_elements;
      cp->vertex_stride = ve->stride;

      /* Update GPU-resident state */
      if (cp->gpu_state) {
         cp->gpu_state->num_elements = ve->num_elements;
         cp->gpu_state->vs_in_stride = ve->num_elements * 16;
         for (unsigned i = 0; i < ve->num_elements && i < 16; i++) {
            cp->gpu_state->elem_vb_idx[i] = ve->elements[i].vertex_buffer_index;
            cp->gpu_state->elem_src_offset[i] = ve->elements[i].src_offset;
            cp->gpu_state->elem_src_stride[i] = ve->elements[i].src_stride;
            cp->gpu_state->elem_attr_size[i] = util_format_get_blocksize(ve->elements[i].src_format);
            cp->gpu_state->elem_instance_divisor[i] = ve->elements[i].instance_divisor;
         }
      }
   }
}

static void
cp_delete_vertex_elements_state(struct pipe_context *ctx, void *state)
{
   FREE(state);  /* frees cp_vertex_elements_state */
}

static void *
cp_create_fs_state(struct pipe_context *ctx,
                   const struct pipe_shader_state *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (state->type != PIPE_SHADER_IR_NIR)
      return MALLOC(1);

   struct nir_shader *nir = (struct nir_shader *)state->ir.nir;

   if (getenv("CUDAPIPE_DUMP_NIR")) {
      fprintf(stderr, "=== FS NIR ===\n");
      nir_print_shader(nir, stderr);
   }

   /* Lower FS I/O */
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out,
                type_size_vec4, nir_lower_io_lower_64bit_to_32);

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(nir,
      cp->screen->sm_major, cp->screen->sm_minor,
      cp->screen->kernels.sampler_ptx);
   if (!bin)
      bin = CALLOC_STRUCT(cp_shader_binary);
   return bin;
}

static void
cp_bind_fs_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (cp->fs_shader != (struct cp_shader_binary *)state)
      cp_batch_flush_why(cp, "fragment shader");
   cp->fs_shader = (struct cp_shader_binary *)state;
}

static void
cp_delete_fs_state(struct pipe_context *ctx, void *state)
{
   cp_shader_binary_destroy((struct cp_shader_binary *)state);
}

static void *
cp_create_vs_state(struct pipe_context *ctx,
                   const struct pipe_shader_state *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (state->type != PIPE_SHADER_IR_NIR)
      return MALLOC(1);

   struct nir_shader *nir = (struct nir_shader *)state->ir.nir;

   if (getenv("CUDAPIPE_DUMP_NIR")) {
      fprintf(stderr, "=== VS NIR ===\n");
      nir_print_shader(nir, stderr);
   }

   /* Lower I/O derefs to explicit load_input/store_output */
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out,
                type_size_vec4, nir_lower_io_lower_64bit_to_32);

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(nir,
      cp->screen->sm_major, cp->screen->sm_minor,
      cp->screen->kernels.sampler_ptx);
   if (!bin)
      bin = CALLOC_STRUCT(cp_shader_binary);
   return bin;
}

static void
cp_bind_vs_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (cp->vs_shader != (struct cp_shader_binary *)state) {
      /* Resolved once: this runs on every draw of every sample. A scene that
       * builds one pipeline per material — gltfscenerendering — rebinds a
       * distinct binary here between every draw, which ends the batch before
       * the key is ever consulted, and that is invisible from the key's own
       * diagnostics. */
      static int diffdbg = -1;
      if (diffdbg < 0)
         diffdbg = getenv("CUDAPIPE_DEBUG_BATCHDIFF") ? 1 : 0;
      if (diffdbg)
         fprintf(stderr, "cudapipe: batchdiff vs bind %p -> %p\n",
                 (void *)cp->vs_shader, state);
      cp_batch_flush_why(cp, "vertex shader");
   }
   cp->vs_shader = (struct cp_shader_binary *)state;
}

static void
cp_bind_gs_state(struct pipe_context *ctx, void *state) {}
static void
cp_bind_tcs_state(struct pipe_context *ctx, void *state) {}
static void
cp_bind_tes_state(struct pipe_context *ctx, void *state) {}

static void
cp_delete_vs_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

static void *
cp_create_compute_state(struct pipe_context *ctx,
                        const struct pipe_compute_state *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (state->ir_type != PIPE_SHADER_IR_NIR)
      return NULL;

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   struct nir_shader *nir = (struct nir_shader *)state->prog;
   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(nir,
      cp->screen->sm_major, cp->screen->sm_minor,
      cp->screen->kernels.sampler_ptx);
   if (!bin) {
      /* Return empty binary so lavapipe doesn't get NULL */
      bin = CALLOC_STRUCT(cp_shader_binary);
   }
   return bin;
}

static void
cp_bind_compute_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   cp->compute_shader = (struct cp_shader_binary *)state;
}

static void
cp_delete_compute_state(struct pipe_context *ctx, void *state)
{
   cp_shader_binary_destroy((struct cp_shader_binary *)state);
}

static void *
cp_create_sampler_state(struct pipe_context *ctx,
                        const struct pipe_sampler_state *state)
{
   struct pipe_sampler_state *copy = MALLOC_STRUCT(pipe_sampler_state);
   if (copy)
      *copy = *state;
   return copy;
}

static void
cp_bind_sampler_states(struct pipe_context *ctx, mesa_shader_stage shader,
                       unsigned start, unsigned count, void **states)
{
   if (getenv("CUDAPIPE_DEBUG_TEX")) {
      fprintf(stderr, "cudapipe: bind_sampler_states stage=%d start=%u count=%u\n",
              shader, start, count);
      for (unsigned i = 0; i < count; i++) {
         struct pipe_sampler_state *s = states ? states[i] : NULL;
         if (s)
            fprintf(stderr, "   samp[%u]: min=%u mag=%u mip=%u wrap=%u,%u,%u\n",
                    start + i, s->min_img_filter, s->mag_img_filter,
                    s->min_mip_filter, s->wrap_s, s->wrap_t, s->wrap_r);
      }
   }
}

static void
cp_delete_sampler_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

static struct pipe_sampler_view *
cp_create_sampler_view(struct pipe_context *ctx, struct pipe_resource *resource,
                       const struct pipe_sampler_view *templ)
{
   struct pipe_sampler_view *view = CALLOC_STRUCT(pipe_sampler_view);
   if (!view)
      return NULL;
   *view = *templ;
   view->reference.count = 1;
   view->texture = NULL;
   pipe_resource_reference(&view->texture, resource);
   view->context = ctx;
   return view;
}

static void
cp_sampler_view_destroy(struct pipe_context *ctx, struct pipe_sampler_view *view)
{
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

static void
cp_set_sampler_views(struct pipe_context *ctx, mesa_shader_stage shader,
                     unsigned start, unsigned count, unsigned unbind_num_trailing_slots,
                     struct pipe_sampler_view **views)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (getenv("CUDAPIPE_DEBUG_TEX"))
      fprintf(stderr, "cudapipe: set_sampler_views stage=%d start=%u count=%u views=%p\n",
              shader, start, count, (void *)views);
   if (shader != MESA_SHADER_FRAGMENT)
      return;

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   for (unsigned i = 0; i < count; i++) {
      unsigned idx = start + i;
      if (idx >= 32) break;

      /* Destroy old texture object */
      if (cp->tex_objects[idx]) {
         cuTexObjectDestroy(cp->tex_objects[idx]);
         cp->tex_objects[idx] = 0;
      }

      if (!views || !views[i] || !views[i]->texture)
         continue;

      struct pipe_resource *res = views[i]->texture;
      struct cp_resource *cp_res = cp_resource(res);
      void *data = cp_resource_data(cp_res);
      if (!data)
         continue;

      /* Create CUDA texture object for 2D textures */
      unsigned w = res->width0;
      unsigned h = res->height0;
      unsigned pixel_size = util_format_get_blocksize(res->format);
      unsigned row_stride = cp_res->lpr.row_stride[0];

      CUDA_RESOURCE_DESC resDesc = {0};
      resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
      resDesc.res.pitch2D.devPtr = (CUdeviceptr)(uintptr_t)data;
      resDesc.res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8;
      resDesc.res.pitch2D.numChannels = pixel_size;
      resDesc.res.pitch2D.width = w;
      resDesc.res.pitch2D.height = h;
      resDesc.res.pitch2D.pitchInBytes = row_stride;

      CUDA_TEXTURE_DESC texDesc = {0};
      texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
      texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
      texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
      texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;

      CUresult err = cuTexObjectCreate(&cp->tex_objects[idx], &resDesc, &texDesc, NULL);
      if (err != CUDA_SUCCESS)
         cp->tex_objects[idx] = 0;

      /* Store resource info for CPU-side sampling */
      cp->tex_resources[idx].data = data;
      cp->tex_resources[idx].width = w;
      cp->tex_resources[idx].height = h;
      cp->tex_resources[idx].row_stride = row_stride;
      cp->tex_resources[idx].pixel_size = pixel_size;
      cp->tex_resources[idx].format = res->format;
   }

   if (start + count > cp->num_tex_objects)
      cp->num_tex_objects = start + count;
}

static void
cp_set_constant_buffer(struct pipe_context *ctx, mesa_shader_stage shader,
                       uint index,
                       const struct pipe_constant_buffer *buf)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (index >= CP_MAX_CONST_BUFFERS)
      return;
   if (shader != MESA_SHADER_COMPUTE && shader != MESA_SHADER_FRAGMENT &&
       shader != MESA_SHADER_VERTEX)
      return;

   void *buf_ptr = NULL;
   unsigned buf_size = 0;
   bool needs_managed_copy = false;

   if (buf && buf->buffer) {
      struct cp_resource *res = cp_resource(buf->buffer);
      void *data = cp_resource_data(res);
      if (data) {
         buf_ptr = (char *)data + buf->buffer_offset;
         buf_size = buf->buffer_size;
      }
   } else if (buf && buf->user_buffer) {
      buf_ptr = (void *)buf->user_buffer;
      buf_size = buf->buffer_size;
      needs_managed_copy = true;
   }

   /* Macro to handle all three shader stages identically */
#define SET_UBO(stage) do {                                              \
      if (needs_managed_copy && buf_ptr && buf_size > 0) {              \
         if (cp->stage##_ubos[index].managed_size < buf_size) {         \
            if (cp->stage##_ubos[index].managed_copy)                   \
               cuMemFree(cp->stage##_ubos[index].managed_copy);         \
            cuMemAlloc(&cp->stage##_ubos[index].managed_copy, buf_size);\
            cp->stage##_ubos[index].managed_size = buf_size;            \
         }                                                              \
         if (cp->stage##_ubos[index].managed_copy) {                    \
            cuMemcpyHtoD(cp->stage##_ubos[index].managed_copy,          \
                         buf_ptr, buf_size);                             \
            buf_ptr = (void*)(uintptr_t)cp->stage##_ubos[index].managed_copy; \
         }                                                              \
      }                                                                 \
      cp->stage##_ubos[index].buffer = buf_ptr;                         \
      cp->stage##_ubos[index].buffer_size = buf_size;                   \
      cp->stage##_ubos[index].user_copy = needs_managed_copy;           \
      if (index + 1 > cp->num_##stage##_ubos)                           \
         cp->num_##stage##_ubos = index + 1;                            \
   } while (0)

   /*
    * A pending batch survives a *vertex* binding, which is what the per-draw
    * table exists to carry, and a compute one, which no draw reads. A fragment
    * binding it will read when it finally runs, so changing one has to submit
    * what is held back first — unless the fragment shader reads no constant
    * buffer at all, in which case what is bound there cannot reach it, or the
    * batch is a blended one, which carries a per-draw table of these too and
    * has already snapshotted the row this would overwrite.
    */
   if (shader == MESA_SHADER_FRAGMENT && cp->batch.pending &&
       !cp->batch.blended &&
       cp->fs_shader && cp->fs_shader->reads_const_bufs) {
      if (cp->fs_ubos[index].buffer != buf_ptr ||
          cp->fs_ubos[index].buffer_size != buf_size || needs_managed_copy)
         cp_batch_flush_why(cp, "a fragment uniform binding");
   }

   if (shader == MESA_SHADER_COMPUTE) {
      SET_UBO(compute);
   } else if (shader == MESA_SHADER_FRAGMENT) {
      SET_UBO(fs);
      if (cp->gpu_state)
         cp->gpu_state->fs_ubos[index] = (uint64_t)(uintptr_t)buf_ptr;
   } else {
      SET_UBO(vs);
      if (cp->gpu_state)
         cp->gpu_state->vs_ubos[index] = (uint64_t)(uintptr_t)buf_ptr;
   }
#undef SET_UBO
}

static void
cp_set_vertex_buffers(struct pipe_context *ctx, unsigned count,
                      const struct pipe_vertex_buffer *buffers)
{
   struct cp_context *cp = (struct cp_context *)ctx;

   if (cp->batch.pending) {
      bool same = count == cp->num_vertex_buffers;
      for (unsigned i = 0; same && i < count && i < 16; i++) {
         const struct pipe_vertex_buffer *nb = buffers ? &buffers[i] : NULL;
         same = nb &&
            nb->buffer.resource == cp->vertex_buffers[i].buffer.resource &&
            nb->buffer_offset == cp->vertex_buffers[i].buffer_offset;
      }
      if (!same)
         cp_batch_flush_why(cp, "vertex buffers");
   }

   for (unsigned i = 0; i < count; i++) {
      if (buffers) {
         cp->vertex_buffers[i] = buffers[i];
         /* Update GPU-resident state */
         if (cp->gpu_state && buffers[i].buffer.resource) {
            struct cp_resource *res = cp_resource(buffers[i].buffer.resource);
            void *data = cp_resource_data(res);
            cp->gpu_state->vb_bases[i] = data
               ? (uint64_t)(uintptr_t)data + buffers[i].buffer_offset : 0;
         } else if (cp->gpu_state) {
            cp->gpu_state->vb_bases[i] = 0;
         }
      } else {
         memset(&cp->vertex_buffers[i], 0, sizeof(cp->vertex_buffers[i]));
         if (cp->gpu_state)
            cp->gpu_state->vb_bases[i] = 0;
      }
   }
   cp->num_vertex_buffers = count;
}

static void
cp_set_shader_buffers(struct pipe_context *ctx, mesa_shader_stage shader,
                      unsigned start, unsigned count,
                      const struct pipe_shader_buffer *buffers,
                      unsigned writable_bitmask)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (shader != MESA_SHADER_COMPUTE)
      return;
   for (unsigned i = 0; i < count; i++) {
      unsigned idx = start + i;
      if (idx >= CP_MAX_SHADER_BUFFERS)
         break;
      if (buffers && buffers[i].buffer) {
         struct cp_resource *res = cp_resource(buffers[i].buffer);
         cp->compute_ssbos[idx].buffer = (char *)cp_resource_data(res) + buffers[i].buffer_offset;
         cp->compute_ssbos[idx].buffer_size = buffers[i].buffer_size;
      } else {
         cp->compute_ssbos[idx].buffer = NULL;
         cp->compute_ssbos[idx].buffer_size = 0;
      }
   }
   if (start + count > cp->num_compute_ssbos)
      cp->num_compute_ssbos = start + count;
}

static void
cp_set_shader_images(struct pipe_context *ctx, mesa_shader_stage shader,
                     unsigned start, unsigned count,
                     unsigned unbind_num_trailing_slots,
                     const struct pipe_image_view *images)
{
}

static void
cp_set_blend_color(struct pipe_context *ctx,
                   const struct pipe_blend_color *color)
{
}

static void
cp_set_stencil_ref(struct pipe_context *ctx,
                   const struct pipe_stencil_ref ref)
{
}

static void
cp_set_sample_mask(struct pipe_context *ctx, unsigned mask)
{
}

static void
cp_set_clip_state(struct pipe_context *ctx,
                  const struct pipe_clip_state *clip)
{
}

static void
cp_set_polygon_stipple(struct pipe_context *ctx,
                       const struct pipe_poly_stipple *stipple)
{
}

static void
cp_set_sample_locations(struct pipe_context *ctx, size_t size, const uint8_t *locations)
{
}

static void
cp_set_min_samples(struct pipe_context *ctx, unsigned min_samples)
{
}

static void
cp_render_condition(struct pipe_context *ctx, struct pipe_query *query,
                    bool condition, enum pipe_render_cond_flag mode)
{
}

struct cp_query {
   unsigned type;
};

static struct pipe_query *
cp_create_query(struct pipe_context *ctx, unsigned query_type, unsigned index)
{
   struct cp_query *q = CALLOC_STRUCT(cp_query);
   if (q)
      q->type = query_type;
   return (struct pipe_query *)q;
}

static void
cp_destroy_query(struct pipe_context *ctx, struct pipe_query *query)
{
   FREE(query);
}

static bool
cp_begin_query(struct pipe_context *ctx, struct pipe_query *query)
{
   return true;
}

static bool
cp_end_query(struct pipe_context *ctx, struct pipe_query *query)
{
   return true;
}

static bool
cp_get_query_result(struct pipe_context *ctx, struct pipe_query *query,
                    bool wait, union pipe_query_result *result)
{
   memset(result, 0, sizeof(*result));
   if (((struct cp_query *)query)->type == PIPE_QUERY_TIMESTAMP)
      result->u64 = 0;
   return true;
}

static void
cp_get_query_result_resource(struct pipe_context *ctx, struct pipe_query *query,
                             enum pipe_query_flags flags, enum pipe_query_value_type type,
                             int index, struct pipe_resource *resource,
                             unsigned offset)
{
   struct cp_resource *res = cp_resource(resource);
   void *data = cp_resource_data(res);
   if (!data)
      return;
   char *dst = (char *)data + offset;
   if (type == PIPE_QUERY_TYPE_U64) {
      uint64_t val = 0;
      memcpy(dst, &val, 8);
   } else {
      uint32_t val = 0;
      memcpy(dst, &val, 4);
   }
}

/*
 * Layout-compatible with lp_texture_handle: lavapipe reads ->functions and
 * ->sampler_index straight out of whatever create_texture_handle() returns and
 * copies them into the descriptor it builds.
 */
struct cp_texture_handle {
   void *functions;
   uint32_t sampler_index;
};

/* Translate a pipe_format into the sampler's decode path. Formats we don't
 * decode yet map to CP_TEXEL_UNSUPPORTED; the screen refuses to advertise
 * those, so reaching one here means something bypassed format checking. */
uint32_t
cp_texel_encoding_from_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
      return CP_TEXEL_R8G8B8A8_UNORM;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
      return CP_TEXEL_B8G8R8A8_UNORM;
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
      return CP_TEXEL_R8G8B8X8_UNORM;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return CP_TEXEL_B8G8R8X8_UNORM;
   case PIPE_FORMAT_A8R8G8B8_UNORM:
   case PIPE_FORMAT_A8R8G8B8_SRGB:
      return CP_TEXEL_A8R8G8B8_UNORM;
   case PIPE_FORMAT_X8R8G8B8_UNORM:
   case PIPE_FORMAT_X8R8G8B8_SRGB:
      return CP_TEXEL_X8R8G8B8_UNORM;
   case PIPE_FORMAT_R8G8B8_UNORM:
   case PIPE_FORMAT_R8G8B8_SRGB:
      return CP_TEXEL_R8G8B8_UNORM;
   case PIPE_FORMAT_R8G8_UNORM:
      return CP_TEXEL_R8G8_UNORM;
   case PIPE_FORMAT_R8_UNORM:
      return CP_TEXEL_R8_UNORM;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      return CP_TEXEL_R8G8B8A8_SNORM;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
      return CP_TEXEL_R16G16B16A16_UNORM;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      return CP_TEXEL_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return CP_TEXEL_R32G32B32A32_FLOAT;
   case PIPE_FORMAT_R32G32B32_FLOAT:
      return CP_TEXEL_R32G32B32_FLOAT;
   case PIPE_FORMAT_R32G32_FLOAT:
      return CP_TEXEL_R32G32_FLOAT;
   case PIPE_FORMAT_R32_FLOAT:
      return CP_TEXEL_R32_FLOAT;
   case PIPE_FORMAT_B5G6R5_UNORM:
      return CP_TEXEL_R5G6B5_UNORM;
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_B5G5R5X1_UNORM:
      return CP_TEXEL_B5G5R5A1_UNORM;
   case PIPE_FORMAT_A1R5G5B5_UNORM:
      return CP_TEXEL_A1R5G5B5_UNORM;
   case PIPE_FORMAT_A1B5G5R5_UNORM:
   case PIPE_FORMAT_X1B5G5R5_UNORM:
      return CP_TEXEL_A1B5G5R5_UNORM;
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_B4G4R4X4_UNORM:
      return CP_TEXEL_B4G4R4A4_UNORM;
   case PIPE_FORMAT_A4R4G4B4_UNORM:
      return CP_TEXEL_A4R4G4B4_UNORM;
   case PIPE_FORMAT_A4B4G4R4_UNORM:
      return CP_TEXEL_A4B4G4R4_UNORM;
   case PIPE_FORMAT_R4G4B4A4_UNORM:
      return CP_TEXEL_R4G4B4A4_UNORM;
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return CP_TEXEL_R11G11B10_FLOAT;
   case PIPE_FORMAT_R9G9B9E5_FLOAT:
      return CP_TEXEL_R9G9B9E5_FLOAT;
   case PIPE_FORMAT_R16_FLOAT:
      return CP_TEXEL_R16_SFLOAT;
   case PIPE_FORMAT_R16G16_FLOAT:
      return CP_TEXEL_R16G16_SFLOAT;
   case PIPE_FORMAT_R16G16_UNORM:
      return CP_TEXEL_R16G16_UNORM;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return CP_TEXEL_A2B10G10R10_UNORM;
   case PIPE_FORMAT_R32_SINT:
      return CP_TEXEL_R32_SINT;
   case PIPE_FORMAT_R16_SINT:
      return CP_TEXEL_R16_SINT;
   case PIPE_FORMAT_DXT1_RGB:
   case PIPE_FORMAT_DXT1_SRGB:
      return CP_TEXEL_DXT1_RGB;
   case PIPE_FORMAT_DXT1_RGBA:
   case PIPE_FORMAT_DXT1_SRGBA:
      return CP_TEXEL_DXT1_RGBA;
   case PIPE_FORMAT_DXT3_RGBA:
   case PIPE_FORMAT_DXT3_SRGBA:
      return CP_TEXEL_DXT3_RGBA;
   case PIPE_FORMAT_DXT5_RGBA:
   case PIPE_FORMAT_DXT5_SRGBA:
      return CP_TEXEL_DXT5_RGBA;
   default:
      return CP_TEXEL_UNSUPPORTED;
   }
}

/* Samplers are deduplicated into a device-visible table; the descriptor only
 * carries the resulting index. */
static uint32_t
cp_register_sampler(struct cp_context *cp, const struct pipe_sampler_state *state)
{
   if (!cp->sampler_table) {
      if (cuMemAlloc(&cp->sampler_table,
                     CP_MAX_SAMPLERS * sizeof(struct cp_sampler_info)) != CUDA_SUCCESS)
         return 0;
      cuMemsetD8Async(cp->sampler_table, 0,
                 CP_MAX_SAMPLERS * sizeof(struct cp_sampler_info), cp->stream);
      cp->num_samplers = 0;
      memset(cp->sampler_table_host, 0, sizeof(cp->sampler_table_host));
   }

   struct cp_sampler_info info = {
      .wrap_s = state->wrap_s,
      .wrap_t = state->wrap_t,
      .wrap_r = state->wrap_r,
      .min_img_filter = state->min_img_filter,
      .mag_img_filter = state->mag_img_filter,
      .min_mip_filter = state->min_mip_filter,
      .unnormalized_coords = state->unnormalized_coords,
      .min_lod = state->min_lod,
      .max_lod = state->max_lod,
      .lod_bias = state->lod_bias,
      .max_anisotropy = state->max_anisotropy,
   };
   memcpy(info.border_color, state->border_color.f, sizeof(info.border_color));

   for (unsigned i = 0; i < cp->num_samplers; i++) {
      if (memcmp(&cp->sampler_table_host[i], &info, sizeof(info)) == 0)
         return i;
   }

   if (cp->num_samplers >= CP_MAX_SAMPLERS)
      return 0;

   cp->sampler_table_host[cp->num_samplers] = info;
   cuMemcpyHtoD(cp->sampler_table + cp->num_samplers * sizeof(info),
                &info, sizeof(info));
   if (getenv("CUDAPIPE_DEBUG_TEX"))
      fprintf(stderr, "cudapipe: sampler[%u] wrap=%u,%u min=%u mag=%u mip=%u\n",
              cp->num_samplers, info.wrap_s, info.wrap_t,
              info.min_img_filter, info.mag_img_filter, info.min_mip_filter);
   return cp->num_samplers++;
}

static uint64_t
cp_create_texture_handle(struct pipe_context *ctx,
                         struct pipe_sampler_view *view,
                         const struct pipe_sampler_state *state)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_texture_handle *h = CALLOC_STRUCT(cp_texture_handle);
   if (!h)
      return 0;

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* lavapipe calls this once per image view (view set, sampler NULL) and once
    * per VkSampler (view NULL, sampler set), then copies whichever field it
    * needs into the descriptor. */
   if (view && view->texture) {
      CUdeviceptr info_dev;
      if (cuMemAlloc(&info_dev, sizeof(struct cp_texture_info)) == CUDA_SUCCESS) {
         struct cp_texture_info info_host;
         memset(&info_host, 0, sizeof(info_host));
         struct cp_texture_info *info = &info_host;

         struct pipe_resource *res = view->texture;
         struct cp_resource *cres = cp_resource(res);
         enum pipe_format format = view->format ? view->format : res->format;

         info->base = (uint64_t)(uintptr_t)cp_resource_data(cres);
         info->width = res->width0;
         info->height = res->height0;

         /* Layer count lives in depth0 for 3D textures and in array_size for
          * everything layered — including cube maps, whose six faces are just
          * array layers to the sampler. */
         switch (res->target) {
         case PIPE_TEXTURE_3D:
            info->depth = MAX2(res->depth0, 1);
            break;
         case PIPE_TEXTURE_CUBE:
         case PIPE_TEXTURE_CUBE_ARRAY:
         case PIPE_TEXTURE_1D_ARRAY:
         case PIPE_TEXTURE_2D_ARRAY:
            info->depth = MAX2(res->array_size, 1);
            break;
         default:
            info->depth = 1;
            break;
         }

         info->format = format;
         info->target = res->target;
         info->first_level = view->u.tex.first_level;
         info->last_level = view->u.tex.last_level;
         info->first_layer = view->u.tex.first_layer;
         info->encoding = cp_texel_encoding_from_format(format);
         info->blocksize = util_format_get_blocksize(format);
         info->is_srgb = util_format_is_srgb(format);

         for (unsigned l = 0; l <= res->last_level && l < CP_MAX_TEXTURE_LEVELS; l++) {
            info->row_stride[l] = cres->lpr.row_stride[l];
            info->img_stride[l] = cres->lpr.img_stride[l];
            info->mip_offset[l] = cres->lpr.mip_offsets[l];
         }

         cuMemcpyHtoD(info_dev, &info_host, sizeof(info_host));
         h->functions = (void *)(uintptr_t)info_dev;

         if (getenv("CUDAPIPE_DEBUG_TEX"))
            fprintf(stderr, "cudapipe: texture handle %ux%u fmt=%u enc=%u "
                    "stride=%u base=%p\n", info->width, info->height,
                    info->format, info->encoding, info->row_stride[0],
                    (void *)(uintptr_t)info->base);
      }
   }

   if (state)
      h->sampler_index = cp_register_sampler(cp, state);

   return (uint64_t)(uintptr_t)h;
}

static uint64_t
cp_create_image_handle(struct pipe_context *ctx,
                       const struct pipe_image_view *image)
{
   struct cp_texture_handle *h = CALLOC_STRUCT(cp_texture_handle);
   return (uint64_t)(uintptr_t)h;
}

static void
cp_delete_texture_handle(struct pipe_context *ctx, uint64_t handle)
{
   struct cp_texture_handle *h = (struct cp_texture_handle *)(uintptr_t)handle;
   if (!h)
      return;
   if (h->functions)
      cuMemFree((CUdeviceptr)(uintptr_t)h->functions);
   FREE(h);
}

static void
cp_delete_image_handle(struct pipe_context *ctx, uint64_t handle)
{
   FREE((void *)(uintptr_t)handle);
}

static void
cp_buffer_subdata(struct pipe_context *ctx, struct pipe_resource *resource,
                  unsigned usage, unsigned offset, unsigned size, const void *data)
{
   struct pipe_transfer *transfer = NULL;
   void *map = ctx->buffer_map(ctx, resource, 0, PIPE_MAP_WRITE, &(struct pipe_box){
      .x = offset, .width = size, .height = 1, .depth = 1
   }, &transfer);
   if (map) {
      memcpy(map, data, size);
      ctx->buffer_unmap(ctx, transfer);
   }
}

struct pipe_context *
cudapipe_create_context(struct pipe_screen *screen, void *priv, unsigned flags)
{
   struct cp_context *ctx = CALLOC_STRUCT(cp_context);
   if (!ctx)
      return NULL;

   ctx->screen = cp_screen(screen);
   ctx->base.screen = screen;
   ctx->base.priv = priv;

   ctx->base.destroy = cp_destroy_context;

   ctx->base.draw_vbo = cp_draw_vbo;
   ctx->base.launch_grid = cp_launch_grid;
   ctx->base.flush = cp_flush;

   ctx->base.create_blend_state = cp_create_blend_state;
   ctx->base.bind_blend_state = cp_bind_blend_state;
   ctx->base.delete_blend_state = cp_delete_blend_state;

   ctx->base.create_rasterizer_state = cp_create_rasterizer_state;
   ctx->base.bind_rasterizer_state = cp_bind_rasterizer_state;
   ctx->base.delete_rasterizer_state = cp_delete_rasterizer_state;

   ctx->base.create_depth_stencil_alpha_state = cp_create_depth_stencil_alpha_state;
   ctx->base.bind_depth_stencil_alpha_state = cp_bind_depth_stencil_alpha_state;
   ctx->base.delete_depth_stencil_alpha_state = cp_delete_depth_stencil_alpha_state;

   ctx->base.create_vertex_elements_state = cp_create_vertex_elements_state;
   ctx->base.bind_vertex_elements_state = cp_bind_vertex_elements_state;
   ctx->base.delete_vertex_elements_state = cp_delete_vertex_elements_state;

   ctx->base.create_fs_state = cp_create_fs_state;
   ctx->base.bind_fs_state = cp_bind_fs_state;
   ctx->base.delete_fs_state = cp_delete_fs_state;

   ctx->base.create_vs_state = cp_create_vs_state;
   ctx->base.bind_vs_state = cp_bind_vs_state;
   ctx->base.delete_vs_state = cp_delete_vs_state;
   ctx->base.bind_gs_state = cp_bind_gs_state;
   ctx->base.bind_tcs_state = cp_bind_tcs_state;
   ctx->base.bind_tes_state = cp_bind_tes_state;

   ctx->base.create_compute_state = cp_create_compute_state;
   ctx->base.bind_compute_state = cp_bind_compute_state;
   ctx->base.delete_compute_state = cp_delete_compute_state;

   ctx->base.create_sampler_state = cp_create_sampler_state;
   ctx->base.bind_sampler_states = cp_bind_sampler_states;
   ctx->base.delete_sampler_state = cp_delete_sampler_state;

   ctx->base.create_sampler_view = cp_create_sampler_view;
   ctx->base.sampler_view_destroy = cp_sampler_view_destroy;
   ctx->base.set_sampler_views = cp_set_sampler_views;

   /*
    * pipe_resource_release() calls this through the context, not the screen,
    * and a driver that leaves it null gets a jump to address zero when
    * lavapipe tears its upload manager down. Four samples were dying there
    * after rendering correctly. llvmpipe uses the same default.
    */
   ctx->base.resource_release = u_default_resource_release;
   ctx->base.set_framebuffer_state = cp_set_framebuffer_state;
   ctx->base.set_viewport_states = cp_set_viewport_states;
   ctx->base.set_scissor_states = cp_set_scissor_states;
   ctx->base.set_constant_buffer = cp_set_constant_buffer;
   ctx->base.set_vertex_buffers = cp_set_vertex_buffers;
   ctx->base.set_shader_buffers = cp_set_shader_buffers;
   ctx->base.set_shader_images = cp_set_shader_images;
   ctx->base.set_blend_color = cp_set_blend_color;
   ctx->base.set_stencil_ref = cp_set_stencil_ref;
   ctx->base.set_sample_mask = cp_set_sample_mask;
   ctx->base.set_clip_state = cp_set_clip_state;
   ctx->base.set_polygon_stipple = cp_set_polygon_stipple;
   ctx->base.buffer_subdata = cp_buffer_subdata;
   ctx->base.set_sample_locations = cp_set_sample_locations;
   ctx->base.set_min_samples = cp_set_min_samples;
   ctx->base.render_condition = cp_render_condition;
   ctx->base.create_query = cp_create_query;
   ctx->base.destroy_query = cp_destroy_query;
   ctx->base.begin_query = cp_begin_query;
   ctx->base.end_query = cp_end_query;
   ctx->base.get_query_result = cp_get_query_result;
   ctx->base.get_query_result_resource = cp_get_query_result_resource;
   ctx->base.create_texture_handle = cp_create_texture_handle;
   ctx->base.create_image_handle = cp_create_image_handle;
   ctx->base.delete_texture_handle = cp_delete_texture_handle;
   ctx->base.delete_image_handle = cp_delete_image_handle;

   ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
   ctx->base.const_uploader = ctx->base.stream_uploader;

   cudapipe_init_context_resource_funcs(&ctx->base);

   /* Allocate persistent GPU state (managed) and device-only arena */
   cuCtxSetCurrent(ctx->screen->cuda_ctx);

   /* Every frame-path launch, memset and copy goes here; see cp_context.h for
    * why it is a default-flagged stream rather than a non-blocking one. If it
    * cannot be created the field stays zero, which is the legacy NULL stream
    * and exactly the behaviour this replaces. */
   if (cuStreamCreate(&ctx->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
      ctx->stream = NULL;
   CUdeviceptr state_dev;
   if (cuMemAllocManaged(&state_dev, sizeof(struct cp_gpu_state),
                         CU_MEM_ATTACH_GLOBAL) == CUDA_SUCCESS) {
      ctx->gpu_state = (struct cp_gpu_state *)(uintptr_t)state_dev;
      memset(ctx->gpu_state, 0, sizeof(struct cp_gpu_state));
   }

   /* 256MB arena — device-only, never touched by CPU */
   cuMemAlloc(&ctx->arena_base, 256 * 1024 * 1024);
   ctx->arena_size = 256 * 1024 * 1024;
   ctx->arena_offset = 0;

   /* Staging for it. A frame of the heaviest sample in the sweep uploads
    * under a megabyte, and running out only costs a synchronisation. */
   if (cuMemAllocHost(&ctx->upload_host, 8 * 1024 * 1024) == CUDA_SUCCESS)
      ctx->upload_size = 8 * 1024 * 1024;

   /* Adaptive rasterizer queues — allocated once, reused across draws.
    *
    * The two queue counters share one allocation and sit adjacent, so the
    * pass loop zeroes both with a single cuMemsetD32 of two words rather than
    * one call each. They are zeroed once per rasterizer pass and a blended
    * draw runs hundreds of passes, so this is two host calls per pass rather
    * than two bytes of memory. Only the base is freed. */
   cuMemAlloc(&ctx->rast_nontrivial,
              (size_t)CP_MAX_NONTRIVIAL * sizeof(uint32_t));
   cuMemAlloc(&ctx->rast_counts, 256);
   ctx->rast_nontrivial_count = ctx->rast_counts;
   ctx->rast_huge_count = ctx->rast_counts + sizeof(uint32_t);
   cuMemAlloc(&ctx->rast_huge_tiles,
              (size_t)CP_MAX_HUGE_TILES * sizeof(struct cp_tile_pair));

   return &ctx->base;
}
