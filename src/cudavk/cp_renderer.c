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
 * TEMPORARY (CUDAVK_ABUFFER): the merged quad array, for the one draw whose
 * peel loop is being compared against it. Zero for every other draw, and for
 * every draw once the verification budget is spent, so the instrumented
 * interpolator is not carried by frames that are only being timed.
 */
#include "nir_to_ptx/cp_nir_to_llvm.h"

#include <inttypes.h>
#include "util/os_time.h"

#include "util/u_memory.h"
#include "util/u_atomic.h"
#include <string.h>
#include <stdlib.h>

/* Last, because it intercepts the CUDA entry points to take the iteration 26
 * census. Nothing below this line calls them unwrapped. */
#include "cp_smallop_tele.h"

/*
 * Bring a renderer up on a device.
 *
 * Everything here is CUDA and the driver's own bookkeeping -- a stream, the
 * flush generation ring, the device arena and its host staging, and the
 * rasterizer's queues. None of it needs a
 * no Gallium object at all, which is the point: a Vulkan front end calls
 * this with a cp_device and gets a renderer it can draw with.
 */
static uint64_t
cp_texture_stream_serial_alloc(struct cp_device *dev)
{
   uint_fast64_t old = atomic_load_explicit(
      &dev->next_texture_stream_serial, memory_order_relaxed);
   while (old != UINT_FAST64_MAX) {
      if (atomic_compare_exchange_weak_explicit(
             &dev->next_texture_stream_serial, &old, old + 1,
             memory_order_relaxed, memory_order_relaxed))
         return old + 1;
   }
   /* Zero permanently selects safe batch-local wait memoization. */
   return 0;
}

bool
cp_context_init(struct cp_context *cp, struct cp_device *dev)
{
   cp->dev = dev;
   cp_smallop_enabled = cp_debug->upload_stats;

   /*
    * Arm the predecessor epoch only when a PDL launch can actually happen:
    * the flag is on AND the modules were built with the waits. When it is off
    * the interception in cp_smallop_tele.h is one predictable branch, the
    * same shape as the upload census next to it. Never cleared, because a
    * second context on an older device must not switch off the checking the
    * first one relies on.
    */
   if (cp_debug->pdl && dev->kernels.pdl)
      cp_pdl_watch = true;
   cp->pdl_prev_fn = NULL;
   cp->pdl_prev_stream = NULL;
   cp->pdl_prev_epoch = 0;
   cp->pdl_taken = cp->pdl_declined = 0;
   cp->pdl_failed = false;

   /* Allocate the persistent device-only arenas and queues. */

   /*
    * How many multiprocessors this device has, for grids that are sized to
    * fill the machine rather than to cover the worst case. Asked once, of the
    * device this context runs on, because a process may hold several.
    */
   cp->sm_count = 0;
   cuDeviceGetAttribute(&cp->sm_count,
                        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                        dev->cuda_device);

   /* Every frame-path launch, memset and copy goes here; see cp_context.h for
    * why it is a default-flagged stream rather than a non-blocking one. If it
    * cannot be created the field stays zero, which is the legacy NULL stream
    * and exactly the behaviour this replaces. */
   if (cuStreamCreate(&cp->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
      goto fail;
   cp->main_stream = cp->stream;
   cp->main_stream_serial = cp_texture_stream_serial_alloc(dev);

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

   cp->abuf = calloc(1, sizeof(*cp->abuf));
   if (!cp->abuf)
      goto fail;
   cp->abuf->enabled = -1;

   /* 256MB arena — device-only, never touched by CPU */
   if (cuMemAlloc(&cp->arena_base, 256 * 1024 * 1024) != CUDA_SUCCESS)
      goto fail;
   cp->arena_size = 256 * 1024 * 1024;
   cp->arena_offset = 0;

   /* Staging for it. A frame of the heaviest sample in the sweep uploads
    * under a megabyte, and running out only costs a synchronisation. */
   if (cuMemAllocHost(&cp->upload_host, 8 * 1024 * 1024) != CUDA_SUCCESS)
      goto fail;
   cp->upload_size = 8 * 1024 * 1024;

   /* Adaptive rasterizer queues — allocated once, reused across draws.
    *
    * The two queue counters and the setup-cache count share one allocation,
    * so the pass loop zeroes all three with one cuMemsetD32 rather than one
    * call each. They are zeroed once per rasterizer pass and a blended
    * draw runs hundreds of passes, so this is two host calls per pass rather
    * than two bytes of memory. Only the base is freed. */
   if (cuMemAlloc(&cp->rast_nontrivial,
                  (size_t)CP_MAX_NONTRIVIAL * sizeof(uint32_t)) != CUDA_SUCCESS ||
       cuMemAlloc(&cp->rast_counts, 256) != CUDA_SUCCESS ||
       cuMemAlloc(&cp->rast_huge_tiles,
                  (size_t)CP_MAX_HUGE_TILES * sizeof(struct cp_tile_pair)) !=
          CUDA_SUCCESS)
      goto fail;
   cp->rast_nontrivial_count = cp->rast_counts;
   cp->rast_huge_count = cp->rast_counts + sizeof(uint32_t);
   /* The census maximum was 776 records in a queue generation. 1024 leaves
    * measured headroom and costs only 84 KiB; refusal is the classic path. */
   if (!cp_debug->no_setup_cache &&
       cuMemAlloc(&cp->rast_setup_cache,
                  (size_t)CP_SETUP_CACHE_CAPACITY *
                  sizeof(struct cp_setup_cache_entry)) != CUDA_SUCCESS)
      cp->rast_setup_cache = 0;

   /* What cp_draw_execute builds its queue struct from; a pass-episode
    * segment append swaps in its stream's own set and restores this one. */
   cp->cur_qset.nontrivial = cp->rast_nontrivial;
   cp->cur_qset.huge_tiles = cp->rast_huge_tiles;
   cp->cur_qset.setup_cache = cp->rast_setup_cache;
   cp->cur_qset.counts = cp->rast_counts;

   cp->upload_open_lo = SIZE_MAX;

   return true;

fail:
   cp_context_cleanup(cp);
   return false;
}

/*
 * Hand out a slice of the draw's scratch arena.
 *
 * Returns managed memory, so the pointer is valid on both host and device. The
 * arena is only resized between draws, so a request that doesn't fit is served
 * by a one-off allocation and the arena grows to cover it next time rather
 * than moving memory that this draw is already pointing at.
 */
static void cp_texture_cache_unpin(struct cp_context *cp);

static void
cp_renderer_texture_fatal(struct cp_context *cp)
{
   cp->hardware_texture.fatal = true;
   cp->device_fatal = true;
   if (cp->dev->texture_cache_fatal)
      cp->dev->texture_cache_fatal(cp->dev->texture_cache_private);
}

static CUresult
cp_mem_alloc_retry(struct cp_context *cp, CUdeviceptr *ptr, size_t bytes)
{
   CUresult err;
   if (cp->hardware_texture.authoritative_oom_armed) {
      cp->hardware_texture.authoritative_oom_armed = false;
      cp->hardware_texture.authoritative_oom_retries++;
      err = CUDA_ERROR_OUT_OF_MEMORY;
   } else {
      err = cuMemAlloc(ptr, bytes);
   }
   if (err == CUDA_ERROR_OUT_OF_MEMORY &&
       !cp->hardware_texture.table_pinned && cp->dev->texture_cache_purge) {
      enum cp_texture_cache_purge_result purge =
         cp->dev->texture_cache_purge(cp->dev->texture_cache_private);
      if (purge == CP_TEXTURE_CACHE_PURGE_FATAL) {
         cp_renderer_texture_fatal(cp);
         return CUDA_ERROR_UNKNOWN;
      }
      if (purge == CP_TEXTURE_CACHE_PURGE_RECLAIMED)
         err = cuMemAlloc(ptr, bytes);
   }
   if (err != CUDA_SUCCESS && err != CUDA_ERROR_OUT_OF_MEMORY)
      cp_renderer_texture_fatal(cp);
   return err;
}

static CUresult
cp_mem_alloc_managed_retry(struct cp_context *cp, CUdeviceptr *ptr,
                           size_t bytes)
{
   CUresult err = cuMemAllocManaged(ptr, bytes, CU_MEM_ATTACH_GLOBAL);
   if (err == CUDA_ERROR_OUT_OF_MEMORY &&
       !cp->hardware_texture.table_pinned && cp->dev->texture_cache_purge) {
      enum cp_texture_cache_purge_result purge =
         cp->dev->texture_cache_purge(cp->dev->texture_cache_private);
      if (purge == CP_TEXTURE_CACHE_PURGE_FATAL) {
         cp_renderer_texture_fatal(cp);
         return CUDA_ERROR_UNKNOWN;
      }
      if (purge == CP_TEXTURE_CACHE_PURGE_RECLAIMED)
         err = cuMemAllocManaged(ptr, bytes, CU_MEM_ATTACH_GLOBAL);
   }
   if (err != CUDA_SUCCESS && err != CUDA_ERROR_OUT_OF_MEMORY)
      cp_renderer_texture_fatal(cp);
   return err;
}

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

   cp->plan.scratch_grows++;

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
      fprintf(stderr, "cudavk: scratch arena wants %zu bytes, over the %zu "
              "cap — refusing. A stage is almost certainly allocating per "
              "pass instead of reusing.\n",
              end, (size_t)CP_SCRATCH_MAX_BYTES);
      return NULL;
   }
   want = MIN2(want, (size_t)CP_SCRATCH_MAX_BYTES);
   CUdeviceptr new_base;
   CUresult err = cp_mem_alloc_managed_retry(cp, &new_base, want);
   if (err != CUDA_SUCCESS) {
      /* Callers treat NULL as "skip this stage", which renders nothing and
       * looks like a shader bug, so say what actually happened. */
      fprintf(stderr, "cudavk: scratch arena grow to %zu bytes failed (%d)\n",
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
      fprintf(stderr, "cudavk: device scratch wants %zu bytes, over the %zu "
              "cap — refusing.\n", end, (size_t)CP_SCRATCH_MAX_BYTES);
      return 0;
   }
   want = MIN2(want, (size_t)CP_SCRATCH_MAX_BYTES);

   CUdeviceptr new_base;
   CUresult err = cp_mem_alloc_retry(cp, &new_base, want);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudavk: device scratch grow to %zu bytes failed (%d)\n",
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
cp_upload_begin_checked(struct cp_context *cp, size_t size, void **host_out,
                        CUresult *error)
{
   *error = CUDA_SUCCESS;
   if (!cp->arena_base || !cp->upload_host || !size)
      return 0;
   if (cp->arena_offset > SIZE_MAX - 255 ||
       cp->upload_offset > SIZE_MAX - 255)
      return 0;

   /* 256 bytes keeps every block on its own cache line. */
   size_t dev_off = ALIGN_POT(cp->arena_offset, 256);
   size_t host_off = ALIGN_POT(cp->upload_offset, 256);
   if (!cp->flush_gens || cp->scratch.current >= cp->flush_gens)
      return 0;
   size_t dev_slice = cp->arena_size / cp->flush_gens;
   size_t host_slice = cp->upload_size / cp->flush_gens;
   unsigned gen = cp->scratch.current;
   size_t dev_start = gen * dev_slice;
   size_t host_start = gen * host_slice;
   size_t dev_limit = dev_start + dev_slice;
   size_t host_limit = host_start + host_slice;

   bool dev_full = dev_off > dev_limit || size > dev_limit - dev_off;
   bool host_full = host_off > host_limit || size > host_limit - host_off;
   if (dev_full || host_full) {
      /* Reuse requires a successful whole-context drain. The native ICD never
       * rotates cp->scratch.current, so this is generation 0 wrapping, and
       * the drain is a whole-context stall the frame pays unnamed. */
      cp_smallop_hit(__FILE__, __LINE__, CP_SMALLOP_UPLOAD_WRAP, size);
      /* Owed bytes first, then the drain, then the rewind: the span names
       * offsets this is about to hand out again. */
      *error = cp_upload_flush(cp);
      if (*error != CUDA_SUCCESS)
         return 0;
      *error = cuCtxSynchronize();
      if (*error != CUDA_SUCCESS)
         return 0;
      cp->arena_offset = dev_start;
      cp->upload_offset = host_start;
      cp->upload_flushed = host_start;
      cp->arena_flushed = dev_start;
      dev_off = dev_start;
      host_off = host_start;
      if (size > dev_slice || size > host_slice)
         return 0;
   }

   /* The bounds above make both additions overflow-safe. */
   cp->arena_offset = dev_off + size;
   cp->upload_offset = host_off + size;
   /* Open until cp_upload_end(): the flush watermark may not pass it. */
   cp->upload_open++;
   cp->upload_open_lo = MIN2(cp->upload_open_lo, host_off);
   *host_out = (char *)cp->upload_host + host_off;
   return cp->arena_base + dev_off;
}

CUdeviceptr
cp_upload_begin(struct cp_context *cp, size_t size, void **host_out)
{
   CUresult ignored;
   return cp_upload_begin_checked(cp, size, host_out, &ignored);
}

/*
 * Send everything the host has written into the staging buffer and not yet
 * copied. One copy per flush point instead of one per block.
 *
 * The span is exact: cp_upload_begin_checked() advances the device and host
 * offsets by the same size from bases that are both gen*slice, so within a
 * generation the two differ by a constant and a host span maps to a device
 * span of the same length. arena_flushed is tracked alongside rather than
 * recomputed, so the two can never drift silently.
 */
CUresult
cp_upload_flush(struct cp_context *cp)
{
   if (cp_debug->no_upload_coalesce)
      return CUDA_SUCCESS;

   /* An open reservation holds the watermark back: its bytes are still being
    * written, and nothing has been launched that could refer to it. */
   size_t hi = MIN2(cp->upload_offset, cp->upload_open_lo);
   if (hi <= cp->upload_flushed) {
      cp->upload.empty++;
      return CUDA_SUCCESS;
   }

   size_t lo = cp->upload_flushed;
   size_t len = hi - lo;
   CUresult err;
   if (cp_debug->upload_flush_fail_at &&
       cp->upload.flushes + 1 == cp_debug->upload_flush_fail_at) {
      err = CUDA_ERROR_UNKNOWN;
   } else {
      cp_smallop_hit(__FILE__, __LINE__, CP_SMALLOP_HTOD_ASYNC, len);
      err = cp_smallop_htod_async_raw(cp->arena_base + cp->arena_flushed,
                                      (const char *)cp->upload_host + lo, len,
                                      cp->stream);
   }
   if (err != CUDA_SUCCESS) {
      /* A deferred copy fails later than the site that produced the block, so
       * no site-local refusal is possible any more. It is fatal, as every
       * other CUDA failure on this path is. */
      fprintf(stderr, "cudavk: coalesced upload flush of %zu bytes failed "
              "(%d); latching device loss\n", len, err);
      cp_renderer_texture_fatal(cp);
      return err;
   }
   cp->upload_flushed = hi;
   cp->arena_flushed += len;
   cp->upload_stream = cp->stream;
   cp->upload.flushes++;
   cp->upload.bytes += len;
   return CUDA_SUCCESS;
}

/* The one place a kernel is launched, and so the one place the owed span has
 * to be sent. Nothing in the driver may call cuLaunchKernel directly; the
 * cp_launch_audit test enforces that.
 *
 * `pdl_after`, when not null, is the kernel the caller believes is directly in
 * front of this one on `stream`. It is a claim, not an instruction: the three
 * conditions below are checked and the attribute is dropped if any of them
 * fails, so a site that becomes wrong later loses its overlap rather than its
 * ordering.
 *
 *   1. the loaded module carries griddepcontrol.wait in its secondaries
 *      (cp_kernels.pdl -- capability, driver version and the flag, decided
 *      once at module build);
 *   2. the previous launch through here really was `pdl_after`, on this same
 *      stream;
 *   3. nothing else has been issued on a stream since: no clear, no copy, no
 *      event, and in particular not the coalesced upload flush this function
 *      performs two lines further up, which is the one interposition the
 *      caller cannot see.
 *
 * (3) is deliberately global rather than per stream. It over-refuses when two
 * threads are submitting at once and never under-refuses, which is the right
 * way round for a scheduling hint.
 */
CUresult
cp_launch_after(struct cp_context *cp, CUfunction f,
                unsigned gx, unsigned gy, unsigned gz,
                unsigned bx, unsigned by, unsigned bz,
                unsigned shmem, CUstream stream, void **params, void **extra,
                CUfunction pdl_after, unsigned pdl_tier)
{
   cp_ctx_check("cp_launch", cp->dev->cuda_ctx);
   CUresult err = cp_upload_flush(cp);
   if (err != CUDA_SUCCESS)
      return err;
   cp->launches++;

   uint64_t epoch = cp_pdl_watch ?
      __atomic_load_n(&cp_pdl_epoch, __ATOMIC_RELAXED) : 0;
   /* CP_PDL_ANY drops the identity comparison and nothing else; see the
    * macro's comment for why that is sound only for a secondary whose wait is
    * its first instruction. */
   bool named = pdl_after == CP_PDL_ANY ? cp->pdl_prev_fn != NULL
                                        : cp->pdl_prev_fn == pdl_after;
   bool pdl = pdl_after && cp_pdl_watch && !cp->pdl_failed &&
              cp->dev->kernels.pdl >= pdl_tier && named &&
              cp->pdl_prev_stream == stream &&
              cp->pdl_prev_epoch == epoch;

   /*
    * Only count a link the running level is meant to convert. A level-1 run
    * would otherwise report every level-2 site as a decline and make the take
    * rate -- which is how a PDL measurement is checked for having happened at
    * all -- unreadable.
    */
   if (pdl_after && cp->dev->kernels.pdl >= pdl_tier) {
      if (pdl)
         cp->pdl_taken++;
      else
         cp->pdl_declined++;
   }

   /* This launch is now the predecessor of whatever comes next. */
   cp->pdl_prev_fn = f;
   cp->pdl_prev_stream = stream;
   cp->pdl_prev_epoch = epoch;

#if CUDA_VERSION >= 11080
   if (pdl) {
      CUlaunchAttribute attr = {
         .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
         .value = { .programmaticStreamSerializationAllowed = 1 },
      };
      CUlaunchConfig cfg = {
         .gridDimX = gx, .gridDimY = gy, .gridDimZ = gz,
         .blockDimX = bx, .blockDimY = by, .blockDimZ = bz,
         .sharedMemBytes = shmem, .hStream = stream,
         .attrs = &attr, .numAttrs = 1,
      };
      err = cuLaunchKernelEx(&cfg, f, params, extra);
      if (err == CUDA_SUCCESS)
         return CUDA_SUCCESS;
      /*
       * Nothing was enqueued: an extended launch that rejects its attribute
       * fails before submission. Say so once and never ask again -- the
       * ordinary launch below is the whole fallback, because the wait in the
       * kernel is inert when no dependency was declared.
       */
      fprintf(stderr, "cudavk: cuLaunchKernelEx refused the programmatic "
              "launch attribute (%d); PDL is off for this context\n", err);
      cp->pdl_failed = true;
      cp->pdl_taken--;
      cp->pdl_declined++;
   }
#endif

   return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream,
                         params, extra);
}

CUresult
cp_launch(struct cp_context *cp, CUfunction f,
          unsigned gx, unsigned gy, unsigned gz,
          unsigned bx, unsigned by, unsigned bz,
          unsigned shmem, CUstream stream, void **params, void **extra)
{
   return cp_launch_after(cp, f, gx, gy, gz, bx, by, bz, shmem, stream,
                          params, extra, NULL, 0);
}

/*
 * Switching streams closes the span: a copy issued on one stream orders
 * nothing on another, so the bytes owed are sent before the switch, on the
 * stream that will carry the readers reserved so far.
 */
void
cp_stream_set(struct cp_context *cp, CUstream stream)
{
   if (cp->stream == stream)
      return;
   cp_upload_flush(cp);
   cp->stream = stream;
}

/* Send a block reserved above, once the caller has finished writing it. */
CUresult
cp_upload_end(struct cp_context *cp, CUdeviceptr dst, const void *host,
              size_t size)
{
   /* Every upload block in the driver arrives here, so counting this line
    * would say only that uploads happen. The census wants the site that
    * asked for the block, which is this function's return address -- and it
    * wants device operations, so a block whose copy is owed rather than
    * issued is not one of them. */
   if (cp_smallop_enabled && cp_debug->no_upload_coalesce)
      cp_smallop_note(__FILE__, __LINE__, __builtin_return_address(0),
                      CP_SMALLOP_HTOD_ASYNC, size);

   cp->upload.blocks++;
   if (cp->upload_open && !--cp->upload_open)
      cp->upload_open_lo = SIZE_MAX;
   if (!cp_debug->no_upload_coalesce)
      return CUDA_SUCCESS;   /* the copy is owed, not skipped */

   return cp_smallop_htod_async_raw(dst, host, size, cp->stream);
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
      if (cuCtxSynchronize() != CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return;
      }
      cp_scratch_reset(cp);
   }
}


/* Reset scratch after all GPU work is done. Frees overflow arenas (old
 * arenas that were replaced during growth) and resets the bump pointer.
 * The current arena is kept at its grown size. */
/* One timed synchronize: the wall time the calling thread spent blocked. */
static bool
cp_sync_timed(struct cp_context *cp, CUstream stream,
              uint64_t *ns, uint64_t *n)
{
   /* Whatever is owed was owed to work this is about to wait for. */
   if (cp_upload_flush(cp) != CUDA_SUCCESS)
      return false;
   int64_t t0 = os_time_get_nano();
   CUresult err = cuStreamSynchronize(stream);
   *ns += (uint64_t)(os_time_get_nano() - t0);
   (*n)++;
   if (err != CUDA_SUCCESS) {
      cp_renderer_texture_fatal(cp);
      return false;
   }
   return true;
}

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
   /*
    * Unless a batch's uploads are still waiting to be launched. The device
    * being idle says the *previous* work has finished reading the arena; it
    * says nothing about a batch whose push blocks are already in it and whose
    * kernels have not been issued yet. Rewinding then lets this draw's own
    * uploads land on them.
    */
   if (!cp->batch_uploads_live) {
      /* The rewind hands these bytes out again, so anything still owed on
       * them has to go first. The callers have already drained the device,
       * so the copy this issues is the last reader of the old contents. */
      cp_upload_flush(cp);
      cp->arena_offset = (size_t)cp->scratch.current *
                         (cp->arena_size / cp->flush_gens);
      cp->upload_offset = (size_t)cp->scratch.current *
                          (cp->upload_size / cp->flush_gens);
      cp->upload_flushed = cp->upload_offset;
      cp->arena_flushed = cp->arena_offset;
   }
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

/*
 * What sampler specialisation actually did this process.
 *
 * Reported rather than inferred: the only other evidence that specialisation
 * stopped firing is a slower sweep, which is indistinguishable from a dozen
 * other causes.
 */
/*
 * What reconstructing batches and episodes from a stream of draws cost, when
 * the whole pass was in an array the entire time.
 */
static void
cp_plan_report(struct cp_context *cp)
{
   if (!cp_debug->plan_stats || !cp->plan.merge_tests)
      return;

   double scopes = (double)MAX2(cp->plan.scopes, 1u);
   fprintf(stderr,
           "cudavk: per-draw planning: %" PRIu64 " keys built, %" PRIu64
           " merge tests, %" PRIu64 " merged first try (%.1f%%), %" PRIu64
           " forced a flush first, %" PRIu64 " batches executed, %" PRIu64
           " episodes closed of %" PRIu64 " attempts, over %" PRIu64
           " render scopes\n",
           cp->plan.key_builds, cp->plan.merge_tests, cp->plan.merges,
           100.0 * (double)cp->plan.merges / (double)cp->plan.merge_tests,
           cp->plan.key_breaks, cp->plan.flushes, cp->plan.pass_finishes,
           cp->plan.pass_finish_calls, cp->plan.scopes);
   fprintf(stderr,
           "cudavk: per scope: %.1f keys, %.1f merge tests, %.1f batches; "
           "reactive reallocations: %" PRIu64 " scratch, %" PRIu64
           " descriptor arena, %" PRIu64 " framebuffer\n",
           cp->plan.key_builds / scopes, cp->plan.merge_tests / scopes,
           cp->plan.flushes / scopes, cp->plan.scratch_grows,
           cp->plan.arena_grows, cp->plan.fb_reallocs);
   if (cp->plan.wait_episode_n + cp->plan.wait_quads_n + cp->plan.wait_peel_n +
       cp->plan.wait_seg_n + cp->plan.wait_upload_n)
      fprintf(stderr, "cudavk: main-thread waits: episode drain %.1f ms/%"
              PRIu64 ", quad counters %.1f ms/%" PRIu64 ", peel checks %.1f "
              "ms/%" PRIu64 ", segment counters %.1f ms/%" PRIu64
              ", desc uploads %.1f ms/%" PRIu64 "\n",
              cp->plan.wait_episode_ns / 1e6, cp->plan.wait_episode_n,
              cp->plan.wait_quads_ns / 1e6, cp->plan.wait_quads_n,
              cp->plan.wait_peel_ns / 1e6, cp->plan.wait_peel_n,
              cp->plan.wait_seg_ns / 1e6, cp->plan.wait_seg_n,
              cp->plan.wait_upload_ns / 1e6, cp->plan.wait_upload_n);
   if (cp->plan.opaque_episodes)
      fprintf(stderr, "cudavk: opaque episodes: %" PRIu64 " closed, %" PRIu64
              " segments (%.2f per episode, longest %" PRIu64 "), %" PRIu64
              " episodes had more than one segment, %" PRIu64
              " segments could overlap a predecessor, %" PRIu64
              " episodes ran on the side streams\n",
              cp->plan.opaque_episodes, cp->plan.opaque_segs,
              (double)cp->plan.opaque_segs /
                 (double)MAX2(cp->plan.opaque_episodes, (uint64_t)1),
              cp->plan.opaque_max_segs, cp->plan.opaque_multiseg,
              cp->plan.opaque_concurrent, cp->plan.opaque_fanned);
   fprintf(stderr, "cudavk: kernel launches: %" PRIu64 " over %" PRIu64
           " episodes and %" PRIu64 " render scopes (%.1f per episode)\n",
           cp->launches, cp->plan.pass_finishes, cp->plan.scopes,
           (double)cp->launches / (double)MAX2(cp->plan.pass_finishes, 1u));
   /* A PDL run that converted nothing looks exactly like a run without the
    * flag, so the two counts are the first thing to read before believing
    * either result. `declined` is a refusal by the checks in
    * cp_launch_after(), not an error. */
   if (cp->pdl_taken + cp->pdl_declined)
      fprintf(stderr, "cudavk: programmatic dependent launches: %" PRIu64
              " of %" PRIu64 " offered took the attribute (%.1f%%), %" PRIu64
              " declined%s\n", cp->pdl_taken,
              cp->pdl_taken + cp->pdl_declined,
              100.0 * (double)cp->pdl_taken /
                 (double)(cp->pdl_taken + cp->pdl_declined),
              cp->pdl_declined, cp->pdl_failed ? ", driver refused" : "");
   if (cp->plan.plan_hits + cp->plan.plan_misses)
      fprintf(stderr, "cudavk: batch plan answered %" PRIu64 " of %" PRIu64
              " merge decisions (%.1f%%)\n", cp->plan.plan_hits,
              cp->plan.plan_hits + cp->plan.plan_misses,
              100.0 * (double)cp->plan.plan_hits /
              (double)(cp->plan.plan_hits + cp->plan.plan_misses));
}

static void
cp_hardware_texture_report(struct cp_context *cp)
{
   if (!cp_debug->texture_cache_stats || !cp->hardware_texture.launches)
      return;
   fprintf(stderr, "cudavk: hardware texture: %" PRIu64 "/%" PRIu64
           " fragment launches hit (%.1f%%), fallbacks shader=%" PRIu64
           " descriptor=%" PRIu64 ", direct=%" PRIu64 "/%" PRIu64
           ", abuffer=%" PRIu64 "/%" PRIu64
           ", modes inline=%" PRIu64 " fused=%" PRIu64
           ", register classes",
           cp->hardware_texture.hits, cp->hardware_texture.launches,
           100.0 * (double)cp->hardware_texture.hits /
              (double)cp->hardware_texture.launches,
           cp->hardware_texture.fallback_shader,
           cp->hardware_texture.fallback_descriptor,
           cp->hardware_texture.path_hits[0],
           cp->hardware_texture.path_launches[0],
           cp->hardware_texture.path_hits[1],
           cp->hardware_texture.path_launches[1],
           cp->hardware_texture.mode_hits[0],
           cp->hardware_texture.mode_hits[1]);
   for (unsigned r = 0; r < ARRAY_SIZE(cp->hardware_texture.hit_regs); r++)
      if (cp->hardware_texture.hit_regs[r])
         fprintf(stderr, " %u:%" PRIu64, r, cp->hardware_texture.hit_regs[r]);
   fputc('\n', stderr);
   for (unsigned m = 0; m < 2; m++) {
      if (!cp->hardware_texture.mode_hits[m])
         continue;
      fprintf(stderr, "cudavk: hardware texture %s resources: "
              "avg-spill=%.1f avg-blocks/sm=%.2f regs",
              m ? "fused" : "inline",
              (double)cp->hardware_texture.mode_spill_sum[m] /
                 cp->hardware_texture.mode_hits[m],
              (double)cp->hardware_texture.mode_blocks_sum[m] /
                 cp->hardware_texture.mode_hits[m]);
      for (unsigned rg = 0; rg < 257; rg++)
         if (cp->hardware_texture.mode_hit_regs[m][rg])
            fprintf(stderr, " %u:%" PRIu64, rg,
                    cp->hardware_texture.mode_hit_regs[m][rg]);
      fputc('\n', stderr);
   }
   static const char *reasons[] = {
      "none/other", "ineligible", "inline-footprint", "surviving-helper",
      "local-memory", "bad-texture-ptx", "jit", "other"
   };
   fprintf(stderr, "cudavk: hardware texture shader fallback reasons:");
   for (unsigned i = 0; i < ARRAY_SIZE(reasons); i++)
      if (cp->hardware_texture.fallback_shader_reason[i])
         fprintf(stderr, " %s=%" PRIu64 "/%" PRIu64 "-binaries",
                 reasons[i], cp->hardware_texture.fallback_shader_reason[i],
                 cp->hardware_texture.fallback_shader_binaries[i]);
   fputc('\n', stderr);
   if (cp->hardware_texture.preflight_calls)
      fprintf(stderr, "cudavk: hardware texture preflight: calls=%" PRIu64
              " cells=%" PRIu64 " host_ns=%" PRIu64
              " avg_us=%.3f ns_per_cell=%.1f authoritative_oom_retries=%" PRIu64
              "\n",
              cp->hardware_texture.preflight_calls,
              cp->hardware_texture.preflight_cells,
              cp->hardware_texture.preflight_ns,
              (double)cp->hardware_texture.preflight_ns /
                 cp->hardware_texture.preflight_calls / 1000.0,
              (double)cp->hardware_texture.preflight_ns /
                 cp->hardware_texture.preflight_cells,
              cp->hardware_texture.authoritative_oom_retries);
}

static void
cp_spec_report(struct cp_context *cp)
{
   if (!cp_debug->spec_stats || !cp->spec.launches)
      return;

   fprintf(stderr, "cudavk: sampler specialisation: %" PRIu64 "/%" PRIu64
           " fragment launches specialised (%.1f%%), %" PRIu64 " shaders, "
           "%" PRIu64 " with an unmatched sampler handle, %" PRIu64
           " sampling with none matched\n",
           cp->spec.specialised, cp->spec.launches,
           100.0 * (double)cp->spec.specialised / (double)cp->spec.launches,
           cp->spec.shaders, cp->spec.shaders_unmatched,
           cp->spec.shaders_unmatchable);
}

void
cp_context_cleanup(struct cp_context *cp)
{
   if (!cp || !cp->dev)
      return;

   if (cuCtxSynchronize() != CUDA_SUCCESS)
      cp_renderer_texture_fatal(cp);
   /* Release the exclusive cache-use lifetime even on a poisoned context. */
   cp_texture_cache_unpin(cp);
   cp_hardware_texture_report(cp);
   cp_spec_report(cp);
   cp_vs_census_report();
   if (cp_debug->shader_stats && cp->fs_launches)
      fprintf(stderr, "cudavk: fragment grid: %" PRIu64 " launches, %" PRIu64
              " blocks, %.1f blocks/launch, %d SMs, waves=%u\n",
              cp->fs_launches, cp->fs_blocks,
              (double)cp->fs_blocks / (double)cp->fs_launches, cp->sm_count,
              cp_debug->fs_grid_waves);
   cp_plan_report(cp);
   if (cp_debug->upload_stats)
      fprintf(stderr, "cudavk: uploads: blocks=%" PRIu64 " flushes=%" PRIu64
              " bytes=%" PRIu64 " empty=%" PRIu64 " coalesce=%d\n",
              cp->upload.blocks, cp->upload.flushes, cp->upload.bytes,
              cp->upload.empty, (int)!cp_debug->no_upload_coalesce);
   cp_smallop_report();
   cp_abuf_fuse_report();
   if (cp->fuse_check) {
      cuMemFree(cp->fuse_check);
      cp->fuse_check = 0;
   }
   cp_abuf_cleanup(cp->abuf);
   free(cp->abuf);
   cp->abuf = NULL;
   cp_scratch_destroy(cp);

   for (unsigned i = 0; i < cp->timer.cap; i++)
      if (cp->timer.events && cp->timer.events[i])
         cuEventDestroy(cp->timer.events[i]);
   free(cp->timer.events);
   free(cp->timer.stages);

   for (unsigned i = 0; i < CP_FLUSH_GENS; i++)
      if (cp->flush_retire[i])
         cuEventDestroy(cp->flush_retire[i]);
   for (unsigned i = 0; i < CP_PASS_STREAMS; i++) {
      if (cp->seg_qsets[i].setup_cache)
         cuMemFree(cp->seg_qsets[i].setup_cache);
      if (cp->seg_ev[i])
         cuEventDestroy(cp->seg_ev[i]);
      if (cp->seg_streams[i])
         cuStreamDestroy(cp->seg_streams[i]);
   }
   if (cp->pass_gate)
      cuEventDestroy(cp->pass_gate);
   if (cp->stream)
      cuStreamDestroy(cp->stream);
   if (cp->upload_host)
      cuMemFreeHost(cp->upload_host);
   if (cp->arena_base)
      cuMemFree(cp->arena_base);
   if (cp->rast_nontrivial)
      cuMemFree(cp->rast_nontrivial);
   if (cp->rast_counts)
      cuMemFree(cp->rast_counts);
   if (cp->rast_huge_tiles)
      cuMemFree(cp->rast_huge_tiles);
   if (cp->rast_setup_cache)
      cuMemFree(cp->rast_setup_cache);

   /* The framebuffer-sized buffers and the sampler table are context state,
    * not draw scratch: nothing else frees them, and a Gallium context whose
    * CUDA context outlives it would leak every one. */
   CUdeviceptr *owned[] = {
      &cp->visbuf, &cp->reject, &cp->resolved, &cp->peel_next, &cp->peel_any,
      &cp->depthbuf, &cp->sampler_table,
      &cp->tile_census_hist, &cp->tile_census_mask, &cp->tile_census_quads,
      &cp->tile_census_smin, &cp->tile_census_smax, &cp->tile_census_refs,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(owned); i++) {
      if (*owned[i])
         cuMemFree(*owned[i]);
      *owned[i] = 0;
   }

   free(cp->hardware_texture.resolve_workspace);
   cp->hardware_texture.resolve_workspace = NULL;
   free(cp->pass_segs);
   free(cp->pass_group_ubos);
}

/* Per-stage timing for a draw, printed under CUDAVK_DEBUG_TIME. See
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
 * TEMPORARY INSTRUMENTATION: fragment census (CUDAVK_FRAG_CENSUS=1)
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
 * TEMPORARY INSTRUMENTATION: A-buffer build and check (CUDAVK_ABUFFER=1)
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
 * and at most once per draw.** `CUDAVK_HANDOFF.md` records this driver
 * invoking the OOM killer with an allocator that grew inside a loop. It is
 * additionally bounded three ways: by CP_ABUF_MAX_FRAGS, by
 * CP_ABUF_MAX_GROWTHS over the process, and by the fact that every growth is
 * announced on stderr. A refused growth is not an error — the draw falls back
 * to the peel loop, which is correct and slow.
 */
#define CP_ABUF_HEADROOM     2u      /* times the count that sized it */
#define CP_ABUF_SLACK        65536u  /* plus this, so a tiny first draw is not tiny */


/*
 * Record an A-buffer stage boundary, but only when someone is going to read
 * it. cuEventElapsedTime is called only under CUDAVK_ABUFFER_TIMING, which
 * is off by default, and sixteen records per eligible draw is nothing beside
 * the kernels they bracket when the numbers are wanted.
 *
 * When they are not, it is 11.8M calls and 3% of the frame on a capture that
 * takes the A-buffer path 482 times per frame — which only became visible
 * once the A-buffer stopped disabling itself and started running.
 */
void
cp_abuf_mark(struct cp_abuf *ab, CUevent ev, CUstream stream)
{
   if (ab && ab->timing)
      cuEventRecord(ev, stream);
}

bool
cp_abuf_enabled(struct cp_abuf *ab)
{
   if (!ab)
      return false;

   if (ab->enabled < 0) {
      /* On by default, off with CUDAVK_NO_ABUFFER=1 — the same shape as
       * CUDAVK_NO_BATCH and CUDAVK_NO_BINCACHE. CUDAVK_ABUFFER=1 still
       * means what it always did and is now a no-op, so a command line or a
       * script written against the opt-in version still does what it says. */
      ab->enabled = cp_debug->no_abuffer ? 0 : 1;

      /* Compositing and verification are mutually exclusive: verification
       * compares against the peel path, while compositing skips that path. */
      ab->verify = cp_debug->abuffer_verify;
      ab->composite = cp_debug->abuffer_composite;
      ab->verify_max = cp_debug->abuffer_verify_draws;
      ab->timing = cp_debug->abuffer_timing;
      ab->debug = cp_debug->abuffer_debug;
      ab->max_layers = cp_debug->abuffer_layers;
   }
   return ab->enabled == 1;
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
 * On by default, off with CUDAVK_NO_ABUF_BATCH=1, which is the shape
 * CUDAVK_NO_BATCH and CUDAVK_NO_ABUFFER already have.
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
cp_abuf_alloc(struct cp_context *cp, struct cp_abuf *ab,
              CUdeviceptr *p, size_t bytes,
              const char *what)
{
   CUresult e = cp_mem_alloc_retry(cp, p, bytes);
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
cp_abuf_setup(struct cp_context *cp, struct cp_abuf *ab, unsigned w, unsigned h)
{
   if (!cp_abuf_enabled(ab))
      return false;
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
      /*
       * Every scalar counter the A-buffer owns, in one allocation. The three
       * that used to be allocated separately -- the two worklist counts and
       * the interpolator's debug counters -- join the block so that a draw or
       * an episode can clear all of them with one device operation instead of
       * six, which is what CUDAVK_COUNTER_BLOCK does. The per-pixel arrays
       * stay where they are: a counter whose size depends on the framebuffer
       * does not belong here.
       */
      if (!cp_abuf_alloc(cp, ab, &ab->counters, 4 * CP_ABUF_BLOCK_WORDS,
                         "counter block"))
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
      ab->list_count = ab->counters +
                       4 * (CP_ABUF_COUNTERS + CP_PASS_MAX_SEGS + 1);
      ab->blk_list_count = ab->counters +
                           4 * (CP_ABUF_COUNTERS + CP_PASS_MAX_SEGS + 2);
      ab->dbg = ab->counters +
                4 * (CP_ABUF_COUNTERS + CP_PASS_MAX_SEGS + 3);
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
      if (!cp_abuf_alloc(cp, ab, &ab->counts, nb, "counts") ||
          !cp_abuf_alloc(cp, ab, &ab->offsets, nb, "offsets") ||
          !cp_abuf_alloc(cp, ab, &ab->cursor, nb, "cursor") ||
          !cp_abuf_alloc(cp, ab, &ab->list, nb, "list") ||
          !cp_abuf_alloc(cp, ab, &ab->sum1, nb1 * 4, "sum1") ||
          !cp_abuf_alloc(cp, ab, &ab->sum1x, nb1 * 4, "sum1x") ||
          !cp_abuf_alloc(cp, ab, &ab->sum2, nb2 * 4, "sum2") ||
          !cp_abuf_alloc(cp, ab, &ab->sum2x, nb2 * 4, "sum2x"))
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
      if (!cp_abuf_alloc(cp, ab, &ab->blk_counts, bb, "block counts") ||
          !cp_abuf_alloc(cp, ab, &ab->blk_offsets, bb, "block offsets") ||
          !cp_abuf_alloc(cp, ab, &ab->blk_list, bb, "block worklist") ||
          !cp_abuf_alloc(cp, ab, &ab->bsum1, bnb1 * 4, "block sum1") ||
          !cp_abuf_alloc(cp, ab, &ab->bsum1x, bnb1 * 4, "block sum1x") ||
          !cp_abuf_alloc(cp, ab, &ab->bsum2, bnb2 * 4, "block sum2") ||
          !cp_abuf_alloc(cp, ab, &ab->bsum2x, bnb2 * 4, "block sum2x"))
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
   if (ab->composite) {
      if (!ab->clist &&
          !cp_abuf_alloc(cp, ab, &ab->clist, ab->cap_pixels * sizeof(uint32_t),
                         "composite worklist"))
         return false;
      /* clist_count is carved out of ab->counters above, so that the
       * per-draw readback stays a single copy; nothing to allocate. */
   }

   if (ab->verify && n > ab->cap_log_pixels) {
      size_t logb = n * CP_ABUF_LOG_LAYERS * sizeof(uint32_t);
      size_t deepb = CP_ABUF_DEEP_PIXELS * CP_BLEND_LAYERS * sizeof(uint32_t);
      ab->cap_log_pixels = 0;
      cp_abuf_free(&ab->log);
      if (!cp_abuf_alloc(cp, ab, &ab->log, logb, "peel log"))
         return false;
      if (!ab->deep_log &&
          (!cp_abuf_alloc(cp, ab, &ab->deep_list,
                          CP_ABUF_DEEP_PIXELS * 4, "deep list") ||
           !cp_abuf_alloc(cp, ab, &ab->deep_log, deepb, "deep peel log")))
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
 * teardown, and only when asked — CUDAVK_ABUFFER_TIMING, which is the switch
 * for "tell me what this path did".
 */
void
cp_abuf_report(struct cp_abuf *ab)
{
   if (!ab || !ab->timing || !ab->peak)
      return;
   fprintf(stderr, "abuffer: peak population %u fragments against a capacity "
           "of %u (%.1f%%), %u growth%s\n", ab->peak, ab->capacity,
           ab->capacity ? 100.0 * ab->peak / ab->capacity : 0.0,
           ab->growths, ab->growths == 1 ? "" : "s");
}

void
cp_abuf_cleanup(struct cp_abuf *ab)
{
   if (!ab)
      return;

   cp_abuf_report(ab);

   CUdeviceptr *device_allocs[] = {
      &ab->counts, &ab->offsets, &ab->cursor,
      &ab->sum1, &ab->sum1x, &ab->sum2, &ab->sum2x,
      &ab->list, &ab->list_count, &ab->clist, &ab->counters,
      &ab->blk_counts, &ab->blk_offsets, &ab->blk_list,
      &ab->blk_list_count, &ab->bsum1, &ab->bsum1x,
      &ab->bsum2, &ab->bsum2x, &ab->dbg,
      &ab->frags, &ab->recs, &ab->quad_prim, &ab->quad_mask,
      &ab->quad_block, &ab->shade_slot, &ab->peel_mask,
      &ab->log, &ab->deep_list, &ab->deep_log,
      &ab->colors_abuf, &ab->writes_abuf,
      &ab->colors_peel, &ab->writes_peel,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(device_allocs); i++)
      cp_abuf_free(device_allocs[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(ab->ev); i++)
      if (ab->ev[i])
         cuEventDestroy(ab->ev[i]);

   free(ab->h_counts);
   free(ab->h_offsets);
   free(ab->h_cursor);
   free(ab->h_frags);
   free(ab->h_log);
   free(ab->h_deep_log);
   free(ab->h_deep_list);
   free(ab->h_blk_counts);
   free(ab->h_blk_offsets);
   free(ab->h_quad_prim);
   free(ab->h_quad_mask);
   free(ab->h_peel_mask);
   free(ab->h_colors_abuf);
   free(ab->h_colors_peel);
   free(ab->h_writes_abuf);
   free(ab->h_writes_peel);

   memset(ab, 0, sizeof(*ab));
   ab->enabled = -1;
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
cp_abuf_size_arrays(struct cp_context *cp, struct cp_abuf *ab, uint32_t total)
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

   if (!cp_abuf_alloc(cp, ab, &ab->frags, want * sizeof(uint32_t), "fragments"))
      return false;
   if (!cp_debug->no_abuf_append &&
       !cp_abuf_alloc(cp, ab, &ab->recs, want * sizeof(uint64_t),
                      "fragment records"))
      return false;
   ab->capacity = (unsigned)want;

   /*
    * The quad arrays, sized from the same number. A quad needs at least one
    * covering fragment, so there can never be more quads than fragments —
    * which is what lets these be allocated from a count the host already has,
    * instead of draining the device again to ask how many the merge produced.
    */
   if (!cp_abuf_alloc(cp, ab, &ab->quad_prim, want * sizeof(uint32_t),
                      "quad primitives") ||
       !cp_abuf_alloc(cp, ab, &ab->quad_mask, want, "quad masks") ||
       !cp_abuf_alloc(cp, ab, &ab->quad_block, want * sizeof(uint32_t),
                      "quad blocks") ||
       /* Only the composite reads this one, and only the comparison against
        * the peel loop reads the other. */
       (ab->composite &&
        !cp_abuf_alloc(cp, ab, &ab->shade_slot, want * sizeof(uint32_t),
                       "shading slots")) ||
       (ab->verify &&
        !cp_abuf_alloc(cp, ab, &ab->peel_mask, want * sizeof(uint32_t),
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

   if (ab->verify) {
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
      if (!cp_abuf_alloc(cp, ab, &ab->colors_abuf, cb, "colours (abuf)") ||
          !cp_abuf_alloc(cp, ab, &ab->writes_abuf, wb, "writes (abuf)") ||
          !cp_abuf_alloc(cp, ab, &ab->colors_peel, cb, "colours (peel)") ||
          !cp_abuf_alloc(cp, ab, &ab->writes_peel, wb, "writes (peel)"))
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

/*
 * The tiling the fused scan uses, and the fused quad count with it.
 *
 *     ept  = ceil(n / (512*512))   elements per thread, at least one
 *     grid = ceil(n / (512*ept))
 *
 * `grid <= CP_ABUF_SCAN_BLOCK` follows: 512*512*ept >= n by the choice of ept,
 * so ceil(n/(512*ept)) <= 512. That is the property the fusion rests on — the
 * array of per-block sums is small enough for every block to scan all of it
 * itself — and it holds for every n, with nothing sized to a bound and no
 * buffer resized. `s1` already holds nb1 = ceil(n/512) >= grid words.
 */
void
cp_abuf_scan_tiling(unsigned n, unsigned *ept, unsigned *grid)
{
   unsigned e = DIV_ROUND_UP(n, CP_ABUF_SCAN_BLOCK * CP_ABUF_SCAN_BLOCK);
   unsigned g;
   if (!e)
      e = 1;
   g = DIV_ROUND_UP(n, CP_ABUF_SCAN_BLOCK * e);
   if (!g)
      g = 1;
   assert(g <= CP_ABUF_SCAN_BLOCK);
   *ept = e;
   *grid = g;
}

bool
cp_abuf_fuse_scan_ready(struct cp_device *screen)
{
   return !cp_debug->no_abuf_fuse_scan && screen->kernels.abuf_scan_reduce &&
          screen->kernels.abuf_scan_finish;
}

bool
cp_abuf_fuse_quad_ready(struct cp_device *screen)
{
   return !cp_debug->no_abuf_fuse_quad && screen->kernels.abuf_quad_count_all &&
          screen->kernels.abuf_quad_fill_all &&
          screen->kernels.abuf_scan_finish;
}

/*
 * The equivalence gate's census. Only ever touched when
 * CUDAVK_ABUF_FUSE_CHECK is set; the counters are read at teardown by
 * cp_abuf_fuse_report(), which is what a gate run looks at.
 */
static uint64_t cp_fuse_scans, cp_fuse_quads, cp_fuse_bad_elems;
static uint64_t cp_fuse_bad_total, cp_fuse_unwritten, cp_fuse_bad_cover;

void
cp_abuf_fuse_report(void)
{
   if (!cp_debug->abuf_fuse_check || !(cp_fuse_scans + cp_fuse_quads))
      return;
   fprintf(stderr,
           "cudavk: abuf fusion check: %" PRIu64 " scans and %" PRIu64
           " quad builds compared against the classic chain element by "
           "element; %" PRIu64 " differing elements, %" PRIu64
           " differing totals, %" PRIu64 " entries never written, %" PRIu64
           " coverage violations\n",
           cp_fuse_scans, cp_fuse_quads, cp_fuse_bad_elems, cp_fuse_bad_total,
           cp_fuse_unwritten, cp_fuse_bad_cover);
}

static CUdeviceptr
cp_abuf_fuse_counters(struct cp_context *cp)
{
   if (!cp->fuse_check) {
      if (cuMemAlloc(&cp->fuse_check, 8 * sizeof(uint32_t)) != CUDA_SUCCESS) {
         cp->fuse_check = 0;
         return 0;
      }
      cuMemsetD32(cp->fuse_check, 0, 8);
   }
   return cp->fuse_check;
}

/*
 * Read the gate's four counters, fold them into the census and clear them.
 * This synchronises, which is exactly why it is behind a flag: a gate run is
 * never a timed run.
 */
static void
cp_abuf_fuse_tally(struct cp_context *cp, const char *what)
{
   uint32_t c[4] = { 0, 0, 0, 0 };
   if (!cp->fuse_check)
      return;
   if (cuStreamSynchronize(cp->stream) != CUDA_SUCCESS ||
       cuMemcpyDtoH(c, cp->fuse_check, sizeof(c)) != CUDA_SUCCESS) {
      cp_renderer_texture_fatal(cp);
      return;
   }
   cp_fuse_bad_elems += c[0];
   cp_fuse_unwritten += c[1];
   cp_fuse_bad_cover += c[2];
   cp_fuse_bad_total += c[3];
   if (c[0] | c[1] | c[2] | c[3])
      fprintf(stderr, "cudavk: abuf fusion check FAILED (%s): %u differing "
              "elements, %u never written, %u coverage violations, %u "
              "differing totals\n", what, c[0], c[1], c[2], c[3]);
   cuMemsetD32Async(cp->fuse_check, 0, 4, cp->stream);
}

/* Count the elements of `a` that differ from `b`, and — when asked — the ones
 * still holding the sentinel, meaning nothing wrote them. */
static void
cp_abuf_fuse_cmp(struct cp_context *cp, struct cp_device *screen,
                 CUdeviceptr a, CUdeviceptr b, unsigned n, uint32_t sentinel,
                 bool check_sentinel, CUdeviceptr counters)
{
   uint32_t sent = sentinel, chk = check_sentinel;
   if (!screen->kernels.abuf_fuse_cmp || !counters)
      return;
   void *p[] = { &a, &b, &n, &sent, &chk, &counters };
   CP_LAUNCH(screen->kernels.abuf_fuse_cmp, MIN2(DIV_ROUND_UP(n, 256u), 1024u),
             1, 1, 256, 1, 1, 0, cp->stream, p, NULL);
}

static void
cp_abuf_scan_classic(struct cp_context *cp, struct cp_device *screen,
                     CUdeviceptr in, CUdeviceptr out, CUdeviceptr s1,
                     CUdeviceptr s1x, CUdeviceptr s2, CUdeviceptr s2x,
                     CUdeviceptr s3, unsigned n, unsigned nb1, unsigned nb2,
                     unsigned nb3, CUdeviceptr clamp_counts,
                     uint32_t clamp_capacity, CUdeviceptr clamp_overflow)
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

/*
 * The second half of the fused scan, on its own: the quad build's counting
 * pass has already produced the block sums, so that scan needs only this.
 *
 * `pdl_after` names the kernel the caller has just launched on this stream, or
 * is null. It is only a claim; cp_launch_after() checks it, and checks that
 * nothing else reached the stream in between, before the launch may overlap
 * that predecessor. See cp_launch_after().
 */
void
cp_abuf_scan_finish_only(struct cp_context *cp, struct cp_device *screen,
                         CUdeviceptr in, CUdeviceptr out, CUdeviceptr sums,
                         unsigned nsums, unsigned n, unsigned ept,
                         CUdeviceptr total, CUdeviceptr zero,
                         CUdeviceptr clamp_counts, uint32_t clamp_capacity,
                         CUdeviceptr clamp_overflow, CUfunction pdl_after)
{
   unsigned brk = cp_debug->abuf_fuse_break;
   void *p[] = { &in, &out, &sums, &nsums, &n, &ept, &total, &zero,
                 &clamp_counts, &clamp_capacity, &clamp_overflow, &brk };
   CP_LAUNCH_AFTER(pdl_after, CP_PDL_TIER_SCAN,
                   screen->kernels.abuf_scan_finish, nsums, 1, 1,
                   CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p, NULL);
}

/*
 * counts -> offsets, exclusive. The grand total is left on the device in
 * `s3`. Used over the pixels for the fragment lists and over the 2x2 blocks
 * for the quads.
 *
 * Two launches when the fusion is on, three or five when it is not. `zero`,
 * when given, is an array of n words cleared in the same pass — the fill
 * cursor, whose memset used to follow this call. It is only passed where no
 * reallocation can happen between the scan and the fill.
 */
void
cp_abuf_scan_n(struct cp_context *cp, struct cp_device *screen,
               CUdeviceptr in, CUdeviceptr out, CUdeviceptr s1, CUdeviceptr s1x,
               CUdeviceptr s2, CUdeviceptr s2x, CUdeviceptr s3,
               unsigned n, unsigned nb1, unsigned nb2, unsigned nb3,
               CUdeviceptr clamp_counts, uint32_t clamp_capacity,
               CUdeviceptr clamp_overflow, CUdeviceptr zero)
{
   unsigned ept, grid;

   if (!cp_abuf_fuse_scan_ready(screen)) {
      cp_abuf_scan_classic(cp, screen, in, out, s1, s1x, s2, s2x, s3, n, nb1,
                           nb2, nb3, clamp_counts, clamp_capacity,
                           clamp_overflow);
      if (zero)
         cuMemsetD32Async(zero, 0, n, cp->stream);
      return;
   }

   cp_abuf_scan_tiling(n, &ept, &grid);

   /*
    * The gate. The classic chain runs first, into shadow buffers, with its
    * clamp disabled so that it cannot disturb the counts the fused chain is
    * about to read; the fused chain then runs into the live buffers and every
    * offset and the grand total are compared on the device. The reassociation
    * is over uint32 counts, so equality is exact by construction — this is
    * not a tolerance.
    */
   CUdeviceptr sh_out = 0, sh1 = 0, sh1x = 0, sh2 = 0, sh2x = 0, sh3 = 0;
   CUdeviceptr counters = 0;
   if (cp_debug->abuf_fuse_check && n) {
      counters = cp_abuf_fuse_counters(cp);
      sh_out = cp_scratch_alloc_device(cp, (size_t)n * 4);
      sh1 = cp_scratch_alloc_device(cp, (size_t)MAX2(nb1, 1u) * 4);
      sh1x = cp_scratch_alloc_device(cp, (size_t)MAX2(nb1, 1u) * 4);
      sh2 = cp_scratch_alloc_device(cp, (size_t)MAX2(nb2, 1u) * 4);
      sh2x = cp_scratch_alloc_device(cp, (size_t)MAX2(nb2, 1u) * 4);
      sh3 = cp_scratch_alloc_device(cp, (size_t)MAX2(nb3, 1u) * 4);
      if (counters && sh_out && sh1 && sh1x && sh2 && sh2x && sh3)
         cp_abuf_scan_classic(cp, screen, in, sh_out, sh1, sh1x, sh2, sh2x,
                              sh3, n, nb1, nb2, nb3, 0, 0, 0);
      else
         sh_out = 0;
   }

   /*
    * The fill cursor is cleared inside the finish kernel, and under CUDAVK_PDL
    * that clear is hoisted in front of the programmatic wait. That is only
    * equivalent while `zero` is a distinct array: clearing it early would
    * otherwise destroy the input the scan is about to read.
    */
   assert(!zero || (zero != in && zero != out && zero != clamp_counts));

   void *pr[] = { &in, &s1, &n, &ept };
   CP_LAUNCH(screen->kernels.abuf_scan_reduce, grid, 1, 1,
                  CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, pr, NULL);
   /*
    * The one link this prototype was written for: reduce -> finish, nothing
    * between them on the stream, ~128 launches a frame on the main stream and
    * a kernel on both ends.
    */
   cp_abuf_scan_finish_only(cp, screen, in, out, s1, grid, n, ept, s3, zero,
                            clamp_counts, clamp_capacity, clamp_overflow,
                            screen->kernels.abuf_scan_reduce);

   if (sh_out) {
      cp_abuf_fuse_cmp(cp, screen, out, sh_out, n, 0, false, counters);
      cp_abuf_fuse_cmp(cp, screen, s3, sh3, 1, 0, false, counters + 3 * 4);
      cp_fuse_scans++;
      cp_abuf_fuse_tally(cp, "scan");
   }
}

void
cp_abuf_scan(struct cp_context *cp, struct cp_device *screen,
             struct cp_abuf *ab, unsigned n, CUdeviceptr zero)
{
   /* The bootstrap count runs before the fragment array exists.  Preserve its
    * counts and offsets so that, after sizing the array from the total, the
    * fill can consume the same scan.  A zero capacity would clamp every run
    * to zero and make the first eligible draw fall back forever. */
   unsigned capacity = ab->frags ? ab->capacity : ~0u;
   cp_abuf_scan_n(cp, screen, ab->counts, ab->offsets, ab->sum1, ab->sum1x,
                  ab->sum2, ab->sum2x, ab->sum3, n, ab->nb1, ab->nb2, ab->nb3,
                  ab->counts, capacity, ab->overflow, zero);
}

/*
 * The quad build: how many distinct primitives each 2x2 block holds, where
 * each block's quads go, and the merge that writes them.
 *
 * Classically that is a compaction pass, a counting pass, a three-launch
 * prefix sum and a filling pass, behind a 0.9 MB clear of blk_counts. Fused
 * it is three launches and no clear, because the counting pass takes over
 * both the compaction (it tests coverage itself, from the four counts the
 * compaction was reading anyway, and writes the zero the clear used to write)
 * and the scan's reduce (its tile is the scan's tile). Both forms leave
 * exactly the same blk_counts, blk_offsets, quad arrays and quad total.
 */
static void
cp_abuf_quad_build(struct cp_context *cp, struct cp_device *screen,
                   struct cp_abuf *ab, unsigned w, unsigned h)
{
   unsigned nblocks = ab->nblocks, qw = ab->quad_width;

   if (cp_debug->no_counter_block) {
      cuMemsetD32Async(ab->bsum3, 0, 2, cp->stream);
      cuMemsetD32Async(ab->dbg, 0, CP_ABUF_DBG_COUNTERS, cp->stream);
   }

   if (!cp_abuf_fuse_quad_ready(screen)) {
      if (cp_debug->no_counter_block)
         cuMemsetD32Async(ab->blk_list_count, 0, 1, cp->stream);
      cp_abuf_mark(ab, ab->ev[6], cp->stream);
      {
         void *p[] = { &ab->counts, &w, &h, &qw, &nblocks, &ab->blk_list,
                       &ab->blk_list_count };
         CP_LAUNCH(screen->kernels.abuf_block_worklist,
                        (nblocks + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab, ab->ev[7], cp->stream);
      cuMemsetD32Async(ab->blk_counts, 0, nblocks, cp->stream);
      {
         void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                       &ab->blk_list, &ab->blk_list_count, &ab->blk_counts };
         /* 32 threads a block, not 256: there are only ~8,100 covered blocks,
          * and a wide thread block packs them into a few dozen CUDA blocks
          * that occupy a fraction of the SMs. */
         CP_LAUNCH(screen->kernels.abuf_quad_count, 1024, 1, 1, 32, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab, ab->ev[9], cp->stream);
      cp_abuf_scan_n(cp, screen, ab->blk_counts, ab->blk_offsets, ab->bsum1,
                     ab->bsum1x, ab->bsum2, ab->bsum2x, ab->bsum3, nblocks,
                     ab->bnb1, ab->bnb2, ab->bnb3, 0, 0, 0, 0);
      {
         void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                       &ab->blk_list, &ab->blk_list_count, &ab->blk_offsets,
                       &ab->quad_prim, &ab->quad_mask, &ab->peel_mask,
                       &ab->quad_block, &ab->shade_slot, &ab->quad_capacity,
                       &ab->quad_overflow };
         CP_LAUNCH(screen->kernels.abuf_quad_fill, 1024, 1, 1, 32, 1, 1,
                        0, cp->stream, p, NULL);
      }
      cp_abuf_mark(ab, ab->ev[10], cp->stream);
      cp_abuf_mark(ab, ab->ev[8], cp->stream);
      return;
   }

   unsigned ept, grid;
   cp_abuf_scan_tiling(nblocks, &ept, &grid);

   /*
    * The gate: the classic compaction and count first, into a shadow array,
    * and the live array pre-filled with a sentinel so that an entry the fused
    * count fails to write is counted rather than inferred from the index
    * arithmetic. Retiring the memset means every one of the nblocks entries
    * must be written every episode, including the tail of the last tile.
    */
   CUdeviceptr counters = 0, sh_counts = 0;
   if (cp_debug->abuf_fuse_check && nblocks) {
      counters = cp_abuf_fuse_counters(cp);
      sh_counts = cp_scratch_alloc_device(cp, (size_t)nblocks * 4);
      if (counters && sh_counts) {
         cuMemsetD32Async(ab->blk_list_count, 0, 1, cp->stream);
         void *p[] = { &ab->counts, &w, &h, &qw, &nblocks, &ab->blk_list,
                       &ab->blk_list_count };
         CP_LAUNCH(screen->kernels.abuf_block_worklist,
                        (nblocks + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, p, NULL);
         cuMemsetD32Async(sh_counts, 0, nblocks, cp->stream);
         void *q[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                       &ab->blk_list, &ab->blk_list_count, &sh_counts };
         CP_LAUNCH(screen->kernels.abuf_quad_count, 1024, 1, 1, 32, 1, 1,
                        0, cp->stream, q, NULL);
         cuMemsetD32Async(ab->blk_counts, 0xFFFFFFFFu, nblocks, cp->stream);
      } else {
         sh_counts = 0;
      }
   }

   cp_abuf_mark(ab, ab->ev[6], cp->stream);
   {
      unsigned brk = cp_debug->abuf_fuse_break;
      void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                    &nblocks, &ab->blk_counts, &ab->bsum1, &ept, &brk };
      CP_LAUNCH(screen->kernels.abuf_quad_count_all, grid, 1, 1,
                     CP_ABUF_SCAN_BLOCK, 1, 1, 0, cp->stream, p, NULL);
   }
   cp_abuf_mark(ab, ab->ev[7], cp->stream);
   cp_abuf_mark(ab, ab->ev[9], cp->stream);

   /*
    * count_all -> finish -> fill_all, the same chain with two more links. The
    * two cp_abuf_mark() calls above are event records, which move the
    * predecessor epoch, so under CUDAVK_ABUFFER_TIMING the first of these two
    * links declines itself and the timing stays honest. That is the check
    * doing its job, not a special case.
    */
   cp_abuf_scan_finish_only(cp, screen, ab->blk_counts, ab->blk_offsets,
                            ab->bsum1, grid, nblocks, ept, ab->bsum3, 0,
                            0, 0, 0, screen->kernels.abuf_quad_count_all);
   {
      void *p[] = { &ab->frags, &ab->offsets, &ab->counts, &w, &h, &qw,
                    &nblocks, &ab->blk_counts, &ab->blk_offsets,
                    &ab->quad_prim, &ab->quad_mask, &ab->peel_mask,
                    &ab->quad_block, &ab->shade_slot, &ab->quad_capacity,
                    &ab->quad_overflow };
      CP_LAUNCH_AFTER(screen->kernels.abuf_scan_finish, CP_PDL_TIER_SCAN,
                      screen->kernels.abuf_quad_fill_all,
                      DIV_ROUND_UP(nblocks, 256u), 1, 1, 256, 1, 1,
                      0, cp->stream, p, NULL);
   }
   cp_abuf_mark(ab, ab->ev[10], cp->stream);
   cp_abuf_mark(ab, ab->ev[8], cp->stream);

   if (sh_counts) {
      cp_abuf_fuse_cmp(cp, screen, ab->blk_counts, sh_counts, nblocks,
                       0xFFFFFFFFu, true, counters);
      if (screen->kernels.abuf_fuse_cover) {
         void *p[] = { &ab->counts, &w, &h, &qw, &nblocks, &ab->blk_counts,
                       &counters };
         CP_LAUNCH(screen->kernels.abuf_fuse_cover,
                        MIN2(DIV_ROUND_UP(nblocks, 256u), 1024u), 1, 1,
                        256, 1, 1, 0, cp->stream, p, NULL);
      }
      cp_fuse_quads++;
      cp_abuf_fuse_tally(cp, "quad build");
   }
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
 * TEMPORARY: step 3b — shade the quad stream (CUDAVK_ABUFFER)
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

static bool
cp_depth_attachment_xfer(struct cp_context *cp,
                         const struct cp_render_scope *scope, bool store)
{
   const struct cp_depth_attachment *depth = &scope->depth;
   CUfunction fn = store ? cp->dev->kernels.depth_attachment_store
                         : cp->dev->kernels.depth_attachment_load;
   if (!fn || !depth->data || !cp->depthbuf)
      return false;

   if (!scope->fb.width || !scope->fb.height)
      return true;

   struct cp_depth_attachment_args args = {
      .image = depth->data,
      .depthbuf = cp->depthbuf,
      .width = scope->fb.width,
      .height = scope->fb.height,
      .row_stride = depth->row_stride,
      .sample_stride = depth->sample_stride,
      .pixel_stride = depth->pixel_stride,
      .format = depth->format,
      .samples = MAX2(scope->attachment_samples, 1u),
      .stencil_clear = store ? depth->stencil_clear : 0,
      .stencil_value = depth->stencil_value,
   };
   void *params[] = { &args };
   CUresult err = cp_launch(cp, fn,
      (args.width + 15) / 16, (args.height + 15) / 16, args.samples,
      16, 16, 1, 0, cp->stream, params, NULL);
   if (err != CUDA_SUCCESS)
      fprintf(stderr, "cudavk: depth attachment %s failed: %d\n",
              store ? "store" : "load", err);
   return err == CUDA_SUCCESS;
}

void
cp_clear_depthbuf(struct cp_context *cp, float depth)
{
   if (!cp->depthbuf)
      return;

   uint32_t value = cp_depth_to_sortable(depth);
   size_t count = (size_t)cp->depthbuf_w * cp->depthbuf_h;

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
cp_blend_desc_for(const struct cp_draw_state *state)
{
   return state->blend;
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
cp_fs_interp_setup(struct cp_context *cp, const struct cp_draw_state *state,
                   const struct cp_render_scope *scope,
                   const struct cp_draw_call *info,
                   const struct cp_shader_binary *fs, unsigned num_fs_inputs,
                   unsigned num_vs_outputs, struct cp_fs_interp_args *interp)
{
   interp->num_samples = MAX2(scope->attachment_samples, 1u);
   interp->point_mode = info->mode == MESA_PRIM_POINTS;
   interp->psiz_slot = cp_slot_for_location(state->vs->out_location,
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
         if (state->vs->out_location[o] == location) {
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
 * CUDAVK_TUNE_VETO overrides the threshold, and it is the knob to reach for
 * if a sample regresses: it trades what the marginal shaders are worth on the
 * samples that gain against what they cost on the samples that do not.
 * CUDAVK_NO_REGCAP turns the whole thing off.
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

/*
 * One register-cap trial, whoever owns it. The same state machine times a
 * base shader's two builds and a sampler variant's two builds; only where the
 * builds live and how they swap differs, which is what this view carries.
 */
struct cp_tune_ctx {
   struct cp_shader_tune *t;
   int tune_cap;
   bool *done;
   const int *num_regs;      /* the currently-bound build's register count */
   void (*swap)(void *);
   void *obj;
   const char *what;
};

static void
cp_tune_release(const struct cp_tune_ctx *c)
{
   struct cp_shader_tune *t = c->t;
   if (t->events_made) {
      for (unsigned i = 0; i < CP_TUNE_SAMPLES; i++) {
         cuEventDestroy(t->start[i]);
         cuEventDestroy(t->stop[i]);
      }
      t->events_made = false;
   }
   *c->done = true;
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
cp_tune_before_ctx(struct cp_context *cp, const struct cp_tune_ctx *c)
{
   struct cp_shader_tune *t = c->t;

   /* A phase whose events are still in flight. Keep launching what is bound
    * — a few extra launches of either build cost nothing — and read them when
    * the device has got to them. */
   if (t->reading) {
      if (!cp_tune_harvest(t))
         return false;
      t->reading = false;

      if (t->phase == 0) {
         /* Back to the build the JIT chose, which is already loaded. */
         t->regs_capped = *c->num_regs;
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
         fprintf(stderr, "cudavk: %s trial regs %3d -> %3d (cap %d): "
                 "%.1f us -> %.1f us median of %d, %s\n", c->what,
                 *c->num_regs, t->regs_capped, c->tune_cap, as_built, capped,
                 CP_TUNE_SAMPLES, keep ? "CAPPED" : "left as built");

      /* The events go now; a swap still pending is applied by the launch this
       * call is about to let through, which no longer times anything. */
      cp_tune_release(c);
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
            *c->done = true;
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
cp_tune_after_ctx(struct cp_context *cp, const struct cp_tune_ctx *c, bool timed)
{
   struct cp_shader_tune *t = c->t;

   if (t->swap_pending) {
      c->swap(c->obj);
      t->swap_pending = false;
   }
   if (!timed)
      return;

   cuEventRecord(t->stop[t->timed], cp->stream);
   if (++t->timed >= CP_TUNE_SAMPLES)
      t->reading = true;
}

static void
cp_tune_swap_exec(void *obj) { cp_shader_exec_swap_build(obj); }

static struct cp_tune_ctx
cp_tune_ctx_exec(struct cp_shader_exec *exec, const char *what)
{
   return (struct cp_tune_ctx) { &exec->tune, exec->tune_cap,
                                 &exec->tune_done, &exec->num_regs,
                                 cp_tune_swap_exec, exec, what };
}

static bool
cp_tune_before(struct cp_context *cp, struct cp_shader_exec *exec,
               const char *what)
{
   if (cp_debug->no_regcap || !exec->tune_cap || exec->tune_done)
      return false;
   struct cp_tune_ctx c = cp_tune_ctx_exec(exec, what);
   return cp_tune_before_ctx(cp, &c);
}

static void
cp_tune_after(struct cp_context *cp, struct cp_shader_exec *exec,
              const char *what, bool timed)
{
   struct cp_tune_ctx c = cp_tune_ctx_exec(exec, what);
   cp_tune_after_ctx(cp, &c, timed);
}

static void
cp_texture_attachment_written(struct cp_context *cp,
                              const struct cp_fb_desc *fb)
{
   if (!fb || !fb->texture_cookie || !cp->dev->texture_cache_written)
      return;
   cp->dev->texture_cache_written(cp->dev->texture_cache_private,
                                  fb->texture_cookie, cp->stream);
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
static const void *
cp_host_ptr(const struct cp_context *cp, uint64_t addr)
{
   for (unsigned i = 0; i < cp->num_host_maps; i++) {
      const struct cp_host_map *m = &cp->host_maps[i];
      if (addr >= m->dev && addr - m->dev < m->size)
         return (const char *)m->host + (addr - m->dev);
   }
   /* Gallium's bindings are managed allocations and remain directly host
    * visible. Native addresses not in a map use the same fallback for null
    * pages and other context-lifetime managed storage. */
   return (const void *)(uintptr_t)addr;
}

static const void *
cp_host_range(const struct cp_context *cp, uint64_t base, size_t offset,
              size_t size)
{
   if (base > UINT64_MAX - offset)
      return NULL;
   uint64_t address = base + offset;
   for (unsigned i = 0; i < cp->num_host_maps; i++) {
      const struct cp_host_map *m = &cp->host_maps[i];
      if (address >= m->dev && address - m->dev <= m->size &&
          size <= m->size - (address - m->dev))
         return (const char *)m->host + (address - m->dev);
   }
   /* Native descriptors must belong to a registered immutable arena. Gallium
    * uses managed allocations and has no native cache bridge. */
   if (cp->dev->texture_cache_private)
      return NULL;
   return (const void *)(uintptr_t)address;
}

static bool
cp_shader_exec_is_hardware(enum cp_shader_exec_mode mode)
{
   return mode == CP_SHADER_EXEC_HW_INLINE ||
          mode == CP_SHADER_EXEC_HW_FUSED;
}

static void
cp_texture_cache_unpin(struct cp_context *cp)
{
   if (cp->hardware_texture.table_pinned &&
       cp->dev->texture_cache_use_end) {
      cp->dev->texture_cache_use_end(cp->dev->texture_cache_private);
      cp->hardware_texture.table_pinned = false;
   }
}

static uint64_t
cp_texture_stream_serial(const struct cp_context *cp)
{
   if (cp->stream == cp->main_stream)
      return cp->main_stream_serial;
   for (unsigned i = 0; i < CP_PASS_STREAMS; i++)
      if (cp->stream == cp->seg_streams[i])
         return cp->seg_stream_serial[i];
   return 0; /* Unknown streams get only batch-local wait memoization. */
}

static bool
cp_texture_cache_available(struct cp_context *cp,
                           const struct cp_draw_state *state,
                           struct cp_shader_binary *fs, bool abuffer)
{
   cp_texture_cache_unpin(cp);
   cp->hardware_texture.table_dev = 0;
   cp->hardware_texture.table_rows = 0;
   cp->hardware_texture.table_sites = 0;
   if (cp->device_fatal) {
      cp->hardware_texture.fatal = true;
      return false;
   }
   cp->hardware_texture.fatal = false;
   if (!cp_debug->texture_cache ||
       !cp->dev->texture_cache_resolve)
      return false;
   if (!cp->dev->texture_cache_use_begin ||
       !cp->dev->texture_cache_use_end) {
      cp_renderer_texture_fatal(cp);
      return false;
   }

   unsigned path = abuffer ? 1 : 0;
   cp->hardware_texture.launches++;
   cp->hardware_texture.path_launches[path]++;
   cp->hardware_texture.launch_mode =
      fs->exec[CP_SHADER_EXEC_HW_INLINE].kernel
      ? CP_SHADER_EXEC_HW_INLINE : CP_SHADER_EXEC_HW_FUSED;
   struct cp_shader_exec *exec = &fs->exec[cp->hardware_texture.launch_mode];
   if (!exec->kernel || !fs->num_hw_tex_sites || fs->hw_tex_dynamic) {
      cp->hardware_texture.fallback_shader++;
      unsigned reason = MIN2(fs->hw_compile_failure,
                             ARRAY_SIZE(cp->hardware_texture.fallback_shader_reason) - 1);
      cp->hardware_texture.fallback_shader_reason[reason]++;
      if (!fs->hw_failure_counted) {
         cp->hardware_texture.fallback_shader_binaries[reason]++;
         fs->hw_failure_counted = true;
         if (cp_debug->shader_stats)
            fprintf(stderr, "cudavk: hardware texture missing binary "
                    "reason=%u sites=%u dynamic=%u fs=%p\n", reason,
                    fs->num_hw_tex_sites, fs->hw_tex_dynamic, (void *)fs);
      }
      return false;
   }

   const uint64_t *tbl = cp->fs_batch.ubos;
   unsigned rows = (tbl && fs->reads_const_bufs)
      ? MAX2(cp->fs_batch.ndraws, 1u) : 1;
   cp->hardware_texture.preflight_attempts++;
   if (cp_debug->texture_cache_fail_authoritative_alloc_at_preflight ==
       cp->hardware_texture.preflight_attempts) {
      cp->hardware_texture.authoritative_oom_armed = true;
      size_t grow = cp->dscratch.size <= CP_SCRATCH_MAX_BYTES - 256
         ? cp->dscratch.size + 256 : 0;
      size_t old_used = cp->dscratch.used;
      if (!grow || !cp_scratch_alloc_device(cp, grow)) {
         if (cp->device_fatal)
            return false;
         cp->hardware_texture.fallback_descriptor++;
         return false;
      }
      /* Keep the real grown authoritative arena, but do not consume dummy
       * bytes after the one-shot purge/retry probe. */
      cp->dscratch.used = old_used;
   }
   if (cp_debug->texture_cache_purge_at_preflight ==
          cp->hardware_texture.preflight_attempts &&
       cp->dev->texture_cache_purge) {
      enum cp_texture_cache_purge_result purge =
         cp->dev->texture_cache_purge(cp->dev->texture_cache_private);
      if (purge == CP_TEXTURE_CACHE_PURGE_FATAL) {
         cp_renderer_texture_fatal(cp);
         return false;
      }
   }

   if (!fs->num_hw_tex_sites ||
       rows > SIZE_MAX / fs->num_hw_tex_sites) {
      cp->hardware_texture.fallback_descriptor++;
      return false;
   }
   size_t count = (size_t)rows * fs->num_hw_tex_sites;
   if (count > SIZE_MAX / sizeof(uint64_t)) {
      cp->hardware_texture.fallback_descriptor++;
      return false;
   }
   if (count > cp->hardware_texture.resolve_workspace_cells) {
      if (count > SIZE_MAX / (3 * sizeof(uint64_t))) {
         cp->hardware_texture.fallback_descriptor++;
         return false;
      }
      uint64_t *grown = realloc(cp->hardware_texture.resolve_workspace,
                                count * 3 * sizeof(uint64_t));
      if (!grown) {
         cp->hardware_texture.fallback_descriptor++;
         return false;
      }
      cp->hardware_texture.resolve_workspace = grown;
      cp->hardware_texture.resolve_workspace_cells = count;
   }
   uint64_t *objects = cp->hardware_texture.resolve_workspace;
   uint64_t *image_cookies = objects + count;
   uint64_t *sampler_cookies = image_cookies + count;

   if (cp->dev->texture_cache_use_begin) {
      cp->dev->texture_cache_use_begin(cp->dev->texture_cache_private);
      cp->hardware_texture.table_pinned = true;
   }
   int64_t preflight_start = cp_debug->plan_stats ? os_time_get_nano() : 0;
   bool use = true;
   for (unsigned row_index = 0; row_index < rows && use; row_index++) {
      const uint64_t *row = tbl
         ? tbl + (size_t)row_index * CP_ARG_UBO_STRIDE : NULL;
      for (unsigned site = 0; site < fs->num_hw_tex_sites; site++) {
         const struct cp_hw_tex_site *ref = &fs->hw_tex_sites[site];
         if (ref->image.ubo_slot >= CP_ARG_UBO_STRIDE ||
             ref->sampler.ubo_slot >= CP_ARG_UBO_STRIDE) {
            use = false;
            break;
         }
         uint64_t image_base = row ? row[ref->image.ubo_slot]
            : (uint64_t)(uintptr_t)state->fs_ubos[ref->image.ubo_slot];
         uint64_t sampler_base = row ? row[ref->sampler.ubo_slot]
            : (uint64_t)(uintptr_t)state->fs_ubos[ref->sampler.ubo_slot];
         const void *image_field = cp_host_range(cp, image_base,
            (size_t)ref->image.offset +
               offsetof(struct cpvk_descriptor, image_cookie),
            sizeof(uint64_t));
         const void *sampler_field = cp_host_range(cp, sampler_base,
            (size_t)ref->sampler.offset +
               offsetof(struct cpvk_descriptor, sampler_cookie),
            sizeof(uint64_t));
         uint64_t image_cookie = 0, sampler_cookie = 0;
         if (image_field)
            memcpy(&image_cookie, image_field, sizeof(image_cookie));
         if (sampler_field)
            memcpy(&sampler_cookie, sampler_field, sizeof(sampler_cookie));
         size_t index = (size_t)row_index * fs->num_hw_tex_sites + site;
         image_cookies[index] = image_cookie;
         sampler_cookies[index] = sampler_cookie;
         if (!image_cookie || !sampler_cookie) {
            use = false;
            break;
         }
      }
   }

   enum cp_texture_cache_result resolve_result =
      CP_TEXTURE_CACHE_SOFT_FALLBACK;
   if (use && cp->dev->texture_cache_resolve_batch) {
      resolve_result = cp->dev->texture_cache_resolve_batch(
         cp->dev->texture_cache_private, image_cookies, sampler_cookies,
         count, cp->stream, cp_texture_stream_serial(cp),
         (CUtexObject *)objects);
      use = resolve_result == CP_TEXTURE_CACHE_READY;
   } else if (use) {
      resolve_result = CP_TEXTURE_CACHE_READY;
      for (size_t i = 0; i < count; i++) {
         resolve_result = cp->dev->texture_cache_resolve(
            cp->dev->texture_cache_private, image_cookies[i],
            sampler_cookies[i], cp->stream, cp_texture_stream_serial(cp),
            (CUtexObject *)&objects[i]);
         if (resolve_result != CP_TEXTURE_CACHE_READY) {
            use = false;
            break;
         }
      }
   }
   cp->hardware_texture.fatal = resolve_result == CP_TEXTURE_CACHE_FATAL;
   cp->device_fatal |= cp->hardware_texture.fatal;
   if (preflight_start) {
      cp->hardware_texture.preflight_ns +=
         os_time_get_nano() - preflight_start;
      cp->hardware_texture.preflight_calls++;
      cp->hardware_texture.preflight_cells += count;
   }
   /* FORCE_FUSED_FS is the same-binary control. It deliberately performs the
    * lazy cache/object work above, then keeps the complete software module. */
   if (!use) {
      cp_texture_cache_unpin(cp);
      cp->hardware_texture.fallback_descriptor++;
      return false;
   }

   void *upload = NULL;
   size_t bytes = count * sizeof(*objects);
   CUresult begin_err = CUDA_SUCCESS;
   CUdeviceptr table_dev =
      cp_upload_begin_checked(cp, bytes, &upload, &begin_err);
   if (!table_dev) {
      cp_texture_cache_unpin(cp);
      if (begin_err != CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return false;
      }
      cp->hardware_texture.fallback_descriptor++;
      return false;
   }
   memcpy(upload, objects, bytes);
   cp->hardware_texture.table_upload_calls++;
   CUresult upload_err =
      cp_debug->texture_cache_fail_table_upload_at ==
         cp->hardware_texture.table_upload_calls
      ? CUDA_ERROR_INVALID_VALUE
      : cp_upload_end(cp, table_dev, upload, bytes);
   if (upload_err != CUDA_SUCCESS) {
      cp_texture_cache_unpin(cp);
      if (cp_debug->texture_cache_stats)
         fprintf(stderr, "cudavk: hardware texture fatal table upload=%d "
                 "before fragment attempt, fs_attempts=%" PRIu64 "\n",
                 upload_err, cp->hardware_texture.fs_attempts);
      cp->hardware_texture.fatal = true;
      cp->device_fatal = true;
      if (cp->dev->texture_cache_fatal)
         cp->dev->texture_cache_fatal(cp->dev->texture_cache_private);
      return false;
   }

   cp->hardware_texture.table_dev = table_dev;
   cp->hardware_texture.table_rows = rows;
   cp->hardware_texture.table_sites = fs->num_hw_tex_sites;
   if (cp_debug->force_fused_fs) {
      cp_texture_cache_unpin(cp);
      return false;
   }
   cp->hardware_texture.hits++;
   cp->hardware_texture.path_hits[path]++;
   cp->hardware_texture.mode_hits[
      cp->hardware_texture.launch_mode == CP_SHADER_EXEC_HW_FUSED]++;
   unsigned hw_mode =
      cp->hardware_texture.launch_mode == CP_SHADER_EXEC_HW_FUSED;
   unsigned regs = MIN2(MAX2(exec->num_regs, 0), 256);
   cp->hardware_texture.hit_regs[regs]++;
   cp->hardware_texture.mode_hit_regs[hw_mode][regs]++;
   cp->hardware_texture.mode_spill_sum[hw_mode] += MAX2(exec->spill_bytes, 0);
   cp->hardware_texture.mode_blocks_sum[hw_mode] += MAX2(exec->blocks_per_sm, 0);
   return true;
}

static bool
cp_fs_launch_shader(struct cp_context *cp, const struct cp_draw_state *state,
                    struct cp_shader_binary *fs,
                    enum cp_shader_exec_mode mode,
                    CUdeviceptr counter, CUdeviceptr fs_in,
                    unsigned fs_in_stride, CUdeviceptr fs_out,
                    CUdeviceptr frag_coord, CUdeviceptr discard_mask,
                    CUdeviceptr front_face, CUdeviceptr coverage,
                    CUdeviceptr fused_interp, unsigned num_threads,
                    CUevent ev_before,
                    CUdeviceptr batch_rows, CUfunction *launched)
{
   /* Which kernel actually ran, for a caller that wants to name it as the
    * predecessor of its own launch. A shader has five possible executions and
    * a pending tune can retarget the launch to an alternate binary, so the
    * caller cannot work this out from the mode it asked for. */
   if (launched)
      *launched = NULL;
   struct cp_shader_exec *base_exec = &fs->exec[mode];
   if (!base_exec->kernel && mode == CP_SHADER_EXEC_CLASSIC &&
       fs->exec[CP_SHADER_EXEC_FUSED].kernel) {
      base_exec = &fs->exec[CP_SHADER_EXEC_FUSED];
      if (!fs->classic_fallback_reported) {
         fprintf(stderr, "cudavk: bare fragment binary unavailable; "
                 "classic interpolation is using the fused binary with its "
                 "helper disabled (not an isolated A/B)\n");
         fs->classic_fallback_reported = true;
      }
   }
   if (!base_exec->kernel) {
      cp_texture_cache_unpin(cp);
      return false;
   }
   enum cp_shader_exec_mode exec_mode =
      base_exec == &fs->exec[CP_SHADER_EXEC_HW_INLINE]
      ? CP_SHADER_EXEC_HW_INLINE
      : (base_exec == &fs->exec[CP_SHADER_EXEC_HW_FUSED]
         ? CP_SHADER_EXEC_HW_FUSED
         : (base_exec == &fs->exec[CP_SHADER_EXEC_INLINE]
            ? CP_SHADER_EXEC_INLINE
            : (base_exec == &fs->exec[CP_SHADER_EXEC_FUSED]
               ? CP_SHADER_EXEC_FUSED : CP_SHADER_EXEC_CLASSIC)));
   if (!cp_shader_exec_is_hardware(exec_mode))
      cp_texture_cache_unpin(cp);

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
   /*
    * Specialisation bakes one sampler state into the kernel with #defines, so
    * a shader qualifies when every sampler handle it uses resolves to the
    * same state -- not only when it uses exactly one, which is what this said
    * and which no longer costs nothing: with the handle matcher fixed,
    * gltfscenerendering's two fragment shaders each carry two matched
    * descriptors and a scene creates both samplers the same way.
    *
    * The equality check below is therefore correctness, not tidiness: two
    * descriptors with different states must fall back, or one texture is
    * filtered with the other's sampler. cpvk_sampler_two_bindings is that
    * case.
    */
   struct cp_sampler_info resolved_samplers[CP_MAX_TEX_DESCS];
   bool samplers_resolved = !cp_shader_exec_is_hardware(exec_mode) &&
      !cp_debug->no_sampler_variant && fs->num_tex_descs >= 1 && fs->num_tex_descs <= CP_MAX_TEX_DESCS &&
      !fs->tex_descs_dynamic && tbl_src;
   for (unsigned i = 0; i < fs->num_tex_descs && samplers_resolved; i++) {
      const struct cp_tex_desc_ref *ref = &fs->tex_descs[i];
      bool have_state = false;
      if (ref->ubo_slot >= CP_ARG_UBO_STRIDE) {
         samplers_resolved = false;
         break;
      }
      for (unsigned r = 0; r < rows; r++) {
         const uint64_t *row = tbl_src + (size_t)r * CP_ARG_UBO_STRIDE;
         const char *base = cp_host_ptr(cp, row[ref->ubo_slot]);
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
      /* Only one state reaches the compiled variant, so every descriptor has
       * to agree with the first. */
      if (samplers_resolved && i &&
          memcmp(&resolved_samplers[0], &resolved_samplers[i],
                 sizeof(resolved_samplers[0])))
         samplers_resolved = false;
   }
   struct cp_sampler_variant *sampler_variant = samplers_resolved
      ? cp_shader_find_sampler_variant(fs, resolved_samplers,
                                       fs->num_tex_descs) : NULL;
   bool variant_missing_exec = !sampler_variant ||
      !sampler_variant->exec[exec_mode].kernel;
   if (samplers_resolved && variant_missing_exec &&
       (sampler_variant ||
        fs->num_sampler_variants < CP_MAX_SAMPLER_VARIANTS) &&
       (!base_exec->tune_cap || base_exec->tune_done)) {
      char *sampler_ptx = cp_compile_sampler_variant(
         cp->dev->sm_major, cp->dev->sm_minor, resolved_samplers,
         (fs->uses_tex_3d ? CP_SAMPLER_3D : 0) |
         (fs->uses_tex_gather ? CP_SAMPLER_GATHER : 0));
      if (sampler_ptx) {
         cp_shader_build_sampler_variant(fs, sampler_ptx, resolved_samplers,
                                         fs->num_tex_descs, exec_mode);
         free(sampler_ptx);
      }
      sampler_variant = cp_shader_find_sampler_variant(
         fs, resolved_samplers, fs->num_tex_descs);
   }
   struct cp_shader_exec *launch_exec = sampler_variant &&
      sampler_variant->exec[exec_mode].kernel
      ? &sampler_variant->exec[exec_mode] : base_exec;
   bool use_sampler_variant = launch_exec != base_exec;
   if (cp_shader_exec_is_hardware(exec_mode))
      assert(!use_sampler_variant);

   /*
    * Specialisation is invisible when it stops working, so count it. A
    * shader whose sampler handles the specialiser could not match still
    * renders correctly and simply launches the generic kernel.
    */
   if (!cp_shader_exec_is_hardware(exec_mode)) {
      cp->spec.launches++;
      if (use_sampler_variant)
         cp->spec.specialised++;
      if (!fs->spec_counted) {
         fs->spec_counted = true;
         cp->spec.shaders++;
         if (fs->num_tex_instrs > fs->num_tex_descs || fs->tex_descs_dynamic)
            cp->spec.shaders_unmatched++;
         if (fs->num_tex_instrs && !fs->num_tex_descs)
            cp->spec.shaders_unmatchable++;
      }
   }

   CUmodule launch_module = launch_exec->module;
   CUfunction launch_kernel = launch_exec->kernel;

   if (cp_debug->debug_tex && !fs->tex_descs_reported) {
      fs->tex_descs_reported = true;
      fprintf(stderr, "cudavk: FS sampler refs=%u dynamic=%u rows=%u\n",
              fs->num_tex_descs, fs->tex_descs_dynamic, rows);
      for (unsigned r = 0; r < rows; r++) {
         const uint64_t *row = tbl_src
            ? tbl_src + (size_t)r * CP_ARG_UBO_STRIDE : NULL;
         for (unsigned i = 0; i < fs->num_tex_descs; i++) {
            const struct cp_tex_desc_ref *ref = &fs->tex_descs[i];
            const char *base = row && ref->ubo_slot < CP_ARG_UBO_STRIDE
               ? cp_host_ptr(cp, row[ref->ubo_slot]) : NULL;
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
   CUresult begin_err;
   bool inject_begin = cp_shader_exec_is_hardware(exec_mode) &&
      cp_debug->texture_cache_fail_fs_arg_begin &&
      !cp->hardware_texture.fail_fs_arg_begin_done;
   if (inject_begin)
      cp->hardware_texture.fail_fs_arg_begin_done = true;
   CUdeviceptr fs_args_dev = inject_begin ? 0 :
      cp_upload_begin_checked(cp, fs_blk_bytes, &fs_blk, &begin_err);
   if (inject_begin)
      begin_err = CUDA_ERROR_INVALID_VALUE;
   if (!fs_args_dev) {
      if (begin_err != CUDA_SUCCESS) {
         if (inject_begin && cp_debug->texture_cache_stats)
            fprintf(stderr, "cudavk: injected fatal hardware FS-argument "
                    "reservation before FS attempts=%" PRIu64 "\n",
                    cp->hardware_texture.fs_attempts);
         cp_renderer_texture_fatal(cp);
      }
      cp_texture_cache_unpin(cp);
      return false;
   }
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
   fs_args_host[CP_ARG_SLOT_HW_TEX_TABLE] =
      (void *)(uintptr_t)(cp_shader_exec_is_hardware(exec_mode)
                          ? cp->hardware_texture.table_dev : 0);
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
      for (unsigned i = 0; i < state->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
         fs_tbl[d * CP_ARG_UBO_STRIDE + i] =
            row ? row[i] : (uint64_t)(uintptr_t)(void *)(uintptr_t)state->fs_ubos[i];
   }

   /* Still written, so that the block reads the same whichever form a stage
    * takes its bindings from — row zero, not the live binding, since a
    * deferred draw's is no longer what is bound. */
   for (unsigned i = 0; i < state->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
      fs_args_host[18 + i] =
         (void *)(uintptr_t)fs_tbl[i];

   CUresult args_upload_err =
      cp_upload_end(cp, fs_args_dev, fs_blk, fs_blk_bytes);
   if (args_upload_err != CUDA_SUCCESS) {
      cp_texture_cache_unpin(cp);
      if (cp_shader_exec_is_hardware(exec_mode)) {
         cp->hardware_texture.fatal = true;
         cp->device_fatal = true;
         if (cp->dev->texture_cache_fatal)
            cp->dev->texture_cache_fatal(cp->dev->texture_cache_private);
      }
      return false;
   }

   if (cp_debug->debug_tex) {
      fprintf(stderr, "cudavk: sampler table %p (%u entries) for FS module\n",
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
   if (!cp_shader_exec_is_hardware(exec_mode)) {
      CUdeviceptr sym;
      size_t sym_size;
      bool *globals_resolved = &launch_exec->globals_resolved;
      CUdeviceptr *sym_sampler_table = &launch_exec->sym_sampler_table;
      CUdeviceptr *sym_quad_derivs = &launch_exec->sym_quad_derivs;
      uint64_t *last_sampler_table = &launch_exec->last_sampler_table;
      int *last_quad_derivs = &launch_exec->last_quad_derivs;
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
       * here, both as built and capped, and the faster build kept. A sampler
       * variant carries its own trial: the base's verdict was measured on
       * the generic sampler path and does not transfer. */
      const char *mode_name = exec_mode == CP_SHADER_EXEC_HW_INLINE
         ? "hardware inline"
         : (exec_mode == CP_SHADER_EXEC_HW_FUSED ? "hardware fused" :
            (exec_mode == CP_SHADER_EXEC_INLINE ? "inline" :
            (exec_mode == CP_SHADER_EXEC_FUSED ? "fused" : "classic")));
      char tune_what[48];
      snprintf(tune_what, sizeof(tune_what), "%s %s",
               use_sampler_variant ? "sampler variant" : "shader", mode_name);
      bool timed = cp_tune_before(cp, launch_exec, tune_what);
      /* A pending swap retargets this launch to the execution's own alternate,
       * never to the other interpolation mode. */
      launch_kernel = launch_exec->kernel;

      /* Compiled shaders grid-stride now, so the launch is capped: the count
       * is device-side and num_threads is the framebuffer's worst case, so a
       * small draw's launch was mostly scheduling idle blocks. 4096 blocks
       * of 256 is far past what fills the machine. */
      unsigned fs_blocks = MIN2((num_threads + 255) / 256, 4096u);

      /*
       * And 4096 is still far past it. The count this grid bounds itself
       * against is device-side, so the host cannot size the grid to the work
       * without a readback -- but it does not need to, because the kernel
       * grid-strides: a grid that fills the machine once covers any count at
       * all, in more iterations rather than in more blocks. NCU measured 12.05
       * waves per SM on this launch with 65% of them running under 50
       * instructions per thread, which is eleven waves of blocks scheduled to
       * discover they have nothing to do.
       *
       * The occupancy comes from the execution being launched, not from a
       * constant: a 40-register shader and a 126-register one do not fit the
       * same number of blocks, and this iteration's own census showed vertex
       * shaders spread over 3 to 6 blocks per SM for the same reason. 4096
       * stays the hard upper bound, so this can only ever shrink the grid.
       */
      if (cp_debug->fs_grid_waves && cp->sm_count > 0) {
         unsigned per_sm = launch_exec->blocks_per_sm > 0
            ? (unsigned)launch_exec->blocks_per_sm : 1u;
         unsigned fill = (unsigned)cp->sm_count * per_sm *
                         cp_debug->fs_grid_waves;
         fs_blocks = MIN2(fs_blocks, MAX2(fill, 1u));
      }

      cp->hardware_texture.fs_attempts++;
      cp->fs_launches++;
      cp->fs_blocks += fs_blocks;
      CUresult fs_err = cp_launch(cp, launch_kernel, fs_blocks,
                                       1, 1, 256, 1, 1, 0, cp->stream,
                                       fs_params, NULL);
      if (launched && fs_err == CUDA_SUCCESS)
         *launched = launch_kernel;
      cp_texture_cache_unpin(cp);
      if (fs_err != CUDA_SUCCESS) {
         fprintf(stderr, "cudavk: fragment shader launch failed (%d)\n",
                 fs_err);
         if (cp_shader_exec_is_hardware(exec_mode)) {
            cp->hardware_texture.fatal = true;
            cp->device_fatal = true;
            if (cp->dev->texture_cache_fatal)
               cp->dev->texture_cache_fatal(cp->dev->texture_cache_private);
         }
         return false;
      }
      if (cp_shader_exec_is_hardware(exec_mode) &&
          cp_debug->texture_cache_fail_after_fs_enqueue &&
          !cp->hardware_texture.fail_after_fs_done) {
         cp->hardware_texture.fail_after_fs_done = true;
         if (cuStreamSynchronize(cp->stream) != CUDA_SUCCESS) {
            cp_renderer_texture_fatal(cp);
            return false;
         }
         if (cp_debug->texture_cache_stats)
            fprintf(stderr, "cudavk: injected fatal after one hardware FS "
                    "enqueue, fs_attempts=%" PRIu64 "\n",
                    cp->hardware_texture.fs_attempts);
         cp_renderer_texture_fatal(cp);
         return false;
      }
      cp_tune_after(cp, launch_exec, tune_what, timed);
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
cp_shade_fragments(struct cp_context *cp, const struct cp_draw_state *state,
                   const struct cp_render_scope *scope,
                   const struct cp_draw_call *info,
                   CUdeviceptr visbuf, CUdeviceptr positions,
                   CUdeviceptr vs_output_buf, CUdeviceptr prim_refs,
                   unsigned num_triangles,
                   unsigned w, unsigned h, void *color_data,
                   float vp_scale_x, float vp_scale_y,
                   float vp_trans_x, float vp_trans_y,
                   float depth_scale, float depth_translate,
                   CUdeviceptr reject, CUdeviceptr resolved,
                   unsigned reject_pass, CUdeviceptr seg_ranges,
                   unsigned num_seg_ranges)
{
   struct cp_device *screen = cp->dev;
   struct cp_shader_binary *fs = state->fs;

   bool have_fs = cp_shader_has_standalone_exec(fs);
   if (!have_fs || !vs_output_buf || !state->vs ||
       !screen->kernels.fs_interpolate || !screen->kernels.fs_writeback) {
      if (cp_debug->debug_draw)
         fprintf(stderr, "  no fragment stage: fs=%p classic=%p fused=%p inline=%p "
                 "vs_out=%p vs=%p\n", (void *)fs,
                 fs ? (void *)fs->exec[CP_SHADER_EXEC_CLASSIC].kernel : NULL,
                 fs ? (void *)fs->exec[CP_SHADER_EXEC_FUSED].kernel : NULL,
                 fs ? (void *)fs->exec[CP_SHADER_EXEC_INLINE].kernel : NULL,
                 (void *)(uintptr_t)vs_output_buf, (void *)state->vs);
      return;
   }

   unsigned num_fs_inputs = MIN2(fs->nir_num_inputs, CP_MAX_FS_INPUTS);
   unsigned num_vs_outputs = state->vs->nir_num_outputs
      ? state->vs->nir_num_outputs : 2;

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
      .prim_refs = prim_refs,
      .vs_out = vs_output_buf,
      .pixel_list = pixel_list,
      .counter = counter,
      .fs_in = fs_in,
      .frag_coord = frag_coord,
      .coverage = coverage,
      .front_face = front_face,
      .front_ccw = state->raster.front_ccw,
      .width = w, .height = h,
      .vs_out_stride = num_vs_outputs * 16,
      .fs_in_stride = fs_in_stride,
      .num_fs_inputs = num_fs_inputs,
      .max_pixels = max_pixels,
      .quad_width = (w + 1) / 2,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
      /* gl_FragCoord.z is the depth the rasterizer tested with, so the
       * interpolator takes the same viewport depth transform. */
      .depth_scale = depth_scale, .depth_translate = depth_translate,
      /* TEMPORARY: see cp->abuf_dbg above. */
      .dbg_blk_offsets = cp->abuf_dbg.blk_offsets,
      .dbg_blk_counts = cp->abuf_dbg.blk_counts,
      .dbg_quad_prim = cp->abuf_dbg.quad_prim,
      .dbg_peel_mask = cp->abuf_dbg.peel_mask,
      .dbg_counters = cp->abuf_dbg.counters,
      .seg_ranges = seg_ranges,
      .num_seg_ranges = num_seg_ranges,
   };

   cp_fs_interp_setup(cp, state, scope, info, fs, num_fs_inputs, num_vs_outputs, &interp);

   /* See the same block in cp_abuf_shade(). A blended batch reaches this path
    * only when its A-buffer merge was refused and the peel loop renders it
    * instead, which is rare and still has to be right. */
   CUdeviceptr batch_rows = 0;
   if (cp->fs_batch.ndraws > 1 && fs->reads_const_bufs) {
      if (!cp->fs_batch.slices) {
         fprintf(stderr, "cudavk: a batch of %u has no slice table; refusing "
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
    * TEMPORARY (CUDAVK_ABUFFER): where this pass's shaded colours are to be
    * deposited for the comparison, one slot per (pixel, primitive). Allocated
    * from the same arena the pass rewinds, so it costs nothing on a frame that
    * is not being verified.
    */
   CUdeviceptr dbg_slot = 0;
   if (cp->abuf_dbg.colors) {
      dbg_slot = cp_scratch_alloc_device(cp, (size_t)max_pixels * 4);
      if (dbg_slot) {
         interp.dbg_slot = dbg_slot;
         interp.abuf_frags = cp->abuf_dbg.frags;
         interp.abuf_offsets = cp->abuf_dbg.offsets;
         interp.abuf_counts = cp->abuf_dbg.counts;
      }
   }

   /*
    * Fused interpolation — the direct path's version of what the A-buffer
    * shade already does. The generated shader interpolates its own slot
    * (cp_fs_direct_lane, through CP_ARG_SLOT_FUSED_INTERP) and the launch
    * below shrinks to cp_fs_compact: the visibility scan and the atomic slot
    * allocation, which the shader cannot reproduce because it does not know
    * its slot until it has one. That removes one launch per shade and keeps
    * the fs_in round trip inside the shader's own kernel.
    *
    * Declined — reverting to the classic three-launch chain — by
    * CUDAVK_NO_FUSED_INTERP, when the A-buffer instrumentation is compiled
    * in (the verification reads the peel interpolator's debug records, which
    * the compaction does not produce), or when the per-quad primitive array
    * or the argument upload is refused.
    */
   CUdeviceptr inshader_interp_dev = 0;
   enum cp_shader_exec_mode inshader_mode = CP_SHADER_EXEC_CLASSIC;
   bool compact_available = screen->kernels.fs_compact &&
      !cp_kernels_instrumented();
   bool use_hw_inline = compact_available &&
      cp_texture_cache_available(cp, state, fs, false);
   if (cp->hardware_texture.fatal)
      return;
   bool allow_inshader = compact_available &&
      (use_hw_inline || (!cp_debug->no_inline_fs &&
       (!cp_debug->no_fused_interp || cp_debug->force_fused_fs)));
   if (allow_inshader) {
      if (use_hw_inline)
         inshader_mode = cp->hardware_texture.launch_mode;
      else if (cp_debug->inline_fs && !cp_debug->force_fused_fs &&
               fs->exec[CP_SHADER_EXEC_INLINE].kernel)
         inshader_mode = CP_SHADER_EXEC_INLINE;
      else if (fs->exec[CP_SHADER_EXEC_FUSED].kernel) {
         inshader_mode = CP_SHADER_EXEC_FUSED;
         if (cp_debug->inline_fs && !cp_debug->force_fused_fs &&
             (cp_debug->shader_stats || cp_debug->dump_ir ||
              cp_debug->dump_ptx) && !fs->inline_fallback_reported) {
            fprintf(stderr, "cudavk: inline fragment binary unavailable; "
                    "using proven fused helper binary\n");
            fs->inline_fallback_reported = true;
         }
      }
   }
   if (allow_inshader && inshader_mode == CP_SHADER_EXEC_CLASSIC &&
       !fs->fused_fallback_reported) {
      fprintf(stderr, "cudavk: in-shader interpolation unavailable; "
              "using classic standalone interpolation\n");
      fs->fused_fallback_reported = true;
   }
   if (inshader_mode != CP_SHADER_EXEC_CLASSIC) {
      CUdeviceptr prim_list = cp_scratch_alloc_device(cp, max_pixels);
      if (prim_list) {
         interp.out_prim_list = prim_list;
         interp.fused_direct = 1;
         inshader_interp_dev = cp_upload(cp, &interp, sizeof(interp));
         if (!inshader_interp_dev) {
            interp.out_prim_list = 0;
            interp.fused_direct = 0;
            inshader_mode = CP_SHADER_EXEC_CLASSIC;
         }
      } else {
         inshader_mode = CP_SHADER_EXEC_CLASSIC;
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
      CUfunction interp_kernel = inshader_interp_dev
         ? screen->kernels.fs_compact : screen->kernels.fs_interpolate;
      CUresult interp_err = cp_launch(cp, interp_kernel,
                                           (num_quads + 255) / 256, 1, 1, 256, 1, 1,
                                           0, cp->stream, interp_params, NULL);
      if (interp_err != CUDA_SUCCESS) {
         fprintf(stderr, "cudavk: fs_interpolate launch failed (%d)\n", interp_err);
         cp_texture_cache_unpin(cp);
         return;
      }
   }
   cp_stage_end(cp, CP_STAGE_INTERPOLATE);

   /* Launch FS and writeback over max_pixels — each kernel reads the actual
    * pixel count from the counter (device-visible managed memory) and exits
    * early for threads beyond it. This avoids a sync just to read the count. */
   unsigned num_pixels = max_pixels;

   enum cp_shader_exec_mode fs_mode = inshader_interp_dev
      ? inshader_mode : CP_SHADER_EXEC_CLASSIC;
   CUfunction fs_kernel = NULL;
   if (!cp_fs_launch_shader(cp, state, fs, fs_mode, counter, fs_in,
                            fs_in_stride, fs_out, frag_coord, discard_mask,
                            front_face, coverage, inshader_interp_dev, num_pixels,
                            0, batch_rows, &fs_kernel))
      return;
   cp_stage_end(cp, CP_STAGE_FRAGMENT);

   /* TEMPORARY (CUDAVK_ABUFFER): scatter this pass's colours into the
    * A-buffer slots the interpolation just resolved. */
   if (dbg_slot && screen->kernels.abuf_scatter_colors) {
      void *p[] = { &fs_out, &fs_out_stride, &dbg_slot, &counter, &num_pixels,
                    &cp->abuf_dbg.capacity, &cp->abuf_dbg.colors,
                    &cp->abuf_dbg.writes, &cp->abuf_dbg.counters };
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
      .depth_write = state->depth.depth_writemask,
      .depth_key_invert = state->depth.depth_enabled &&
         (state->depth.depth_func == CP_FUNC_GREATER ||
          state->depth.depth_func == CP_FUNC_GEQUAL),
      .width = w,
      .fs_out_stride = fs_out_stride,
      .num_pixels = num_pixels,
      .color_encoding = (uint32_t)MAX2(scope->fb.color_encoding, 0),
      .blend = cp_blend_desc_for(state),
      .num_samples = MAX2(scope->attachment_samples, 1u),
      .height = h,
      .sample_stride = scope->fb.color_sample_stride,
   };

   /*
    * A pass with no colour attachment still needs this launch when it writes
    * depth, because `cp_fs_writeback` is the only thing that ever writes
    * `cp->depthbuf`: the rasterizer reads it to test against and never
    * advances it. A depth-only pass — a shadow map — that skipped the launch
    * left its attachment at the clear value, which is what made favorite2's
    * shadow map uniformly far and the surfaces sampling it black.
    *
    * `reject` and `resolved` belong to the discard retry, which needs a
    * colour target to be enabled at all, so a colourless draw still asks for
    * nothing else here; the kernel returns after the depth commit when
    * `color_out` is null rather than dereferencing it. A truly attachmentless
    * pass — a fragment shader run purely for its storage-image and SSBO
    * writes, which the shader launch above has already made — writes neither,
    * and is still skipped.
    */
   if (color_data || (wb.depth_write && scope->fb.has_zs)) {
      void *wb_params[] = { &wb };
      cp_nvtx_push("writeback");
      /* The kernel strides, so the grid is capped: num_pixels is the
       * framebuffer's worst case and the launch was spending more time
       * scheduling idle blocks than writing pixels on small draws. */
      /*
       * Tier 2: the writeback behind the shader itself. The shader is a
       * generated kernel and gets no trigger of its own, which it does not
       * need -- the trigger is implicit once its blocks exit. The name is
       * carried out of cp_fs_launch_shader() rather than guessed, so the
       * optional A-buffer colour scatter above, or a tuning event record,
       * makes this decline instead of claiming a predecessor it no longer
       * has.
       */
      CUresult wb_err = cp_launch_after(cp, screen->kernels.fs_writeback,
                     MIN2((num_pixels + 255) / 256, 2048u), 1, 1, 256, 1, 1,
                     0, cp->stream, wb_params, NULL,
                     fs_kernel, CP_PDL_TIER_FS);
      cp_nvtx_pop();   /* writeback */
      if (wb_err != CUDA_SUCCESS) {
         fprintf(stderr, "cudavk: FS writeback launch failed (%d)\n", wb_err);
         cp->device_fatal = true;
         if (cp->dev->texture_cache_fatal)
            cp->dev->texture_cache_fatal(cp->dev->texture_cache_private);
         return;
      }
      cp_texture_attachment_written(cp, &scope->fb);
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

      /*
       * Only as many pixels as are going to be printed. Fetching all of them
       * means four allocations scaled by a full frame's pixel count, and on a
       * 1280x720 draw that is where this path died -- taking the answer with
       * it, because the draw being investigated was the one that never got
       * here. The vertex dump below needs no pixels at all, and a draw that
       * shades nothing is exactly the case worth looking at.
       */
      unsigned dump_pixels = MIN2(num_pixels, 4096u);

      /*
       * Only the vertices that will be printed. The loop below stops at
       * 6 * vstep, and fetching the whole stream instead read the entire
       * vertex-output buffer of a 4,512-triangle draw -- which is where this
       * path crashed on pbribl and bloom, taking with it the answer to why
       * their draws shade nothing. An instrument that dies on the case being
       * investigated is worse than none.
       */
      unsigned dump_verts = MIN2(num_triangles * 3,
                                 6 * MAX2(cp_debug->debug_fs_vstep, 1u));
      size_t vs_out_bytes = (size_t)dump_verts * num_vs_outputs * 16;
      float *vs_out = vs_out_bytes ? malloc(vs_out_bytes) : NULL;
      uint32_t *plist_buf = dump_pixels ? malloc((size_t)dump_pixels * 4) : NULL;
      float *fin_buf = dump_pixels ? malloc((size_t)dump_pixels * fs_in_stride) : NULL;
      float *fout_buf = dump_pixels ? malloc((size_t)dump_pixels * fs_out_stride) : NULL;
      if ((vs_out_bytes && !vs_out) ||
          (dump_pixels && (!plist_buf || !fin_buf || !fout_buf))) {
         free(vs_out); free(plist_buf); free(fin_buf); free(fout_buf);
         fprintf(stderr, "  (CUDAVK_DEBUG_FS: out of memory)\n");
         return;
      }
      if (vs_out_bytes)
         cuMemcpyDtoH(vs_out, vs_output_buf, vs_out_bytes);
      if (dump_pixels) {
         cuMemcpyDtoH(plist_buf, pixel_list, (size_t)dump_pixels * 4);
         cuMemcpyDtoH(fin_buf, fs_in, (size_t)dump_pixels * fs_in_stride);
         cuMemcpyDtoH(fout_buf, fs_out, (size_t)dump_pixels * fs_out_stride);
      }
      num_pixels = dump_pixels;

      unsigned vstep = cp_debug->debug_fs_vstep;   /* registry clamps to >= 1 */
      for (unsigned v = 0; v < dump_verts; v += vstep) {
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
         /*
          * The framebuffer word, fetched rather than dereferenced.
          * color_data is a device address, and reading it on the host works
          * only when the colour target happens to be host-visible -- which it
          * is in a small test and is not in a sample. This path segfaulted on
          * pbribl and bloom for that reason, which is why neither could be
          * looked at with it.
          */
         uint32_t fb_word = 0;
         if (color_data)
            cuMemcpyDtoH(&fb_word, (CUdeviceptr)(uintptr_t)color_data +
                         (size_t)px * 4, sizeof(fb_word));
         fprintf(stderr, "  px(%u,%u) in=[%.9f %.9f] out=[%.3f %.3f %.3f %.3f] "
                 "fb=0x%08x\n",
                 px % w, px / w,
                 fin[i * (fs_in_stride / 4) + 0], fin[i * (fs_in_stride / 4) + 1],
                 fout[i * (fs_out_stride / 4) + 0], fout[i * (fs_out_stride / 4) + 1],
                 fout[i * (fs_out_stride / 4) + 2], fout[i * (fs_out_stride / 4) + 3],
                 fb_word);
      }
      free(vs_out); free(plist_buf); free(fin_buf); free(fout_buf);
   }

}


bool
cp_abuf_shade(struct cp_context *cp, const struct cp_draw_state *state,
              const struct cp_render_scope *scope,
              const struct cp_draw_call *info,
              struct cp_abuf *ab, CUdeviceptr positions,
              CUdeviceptr vs_output_buf, CUdeviceptr prim_refs,
              unsigned w, unsigned h,
              float vp_scale_x, float vp_scale_y,
              float vp_trans_x, float vp_trans_y,
              float depth_scale, float depth_translate,
              uint32_t num_quads, uint32_t num_covered, bool record_colors,
              void *color_data, bool composite, float *t_interp,
              float *t_shade, float *t_composite,
              struct cp_abuf_seg_shade *seg)
{
   struct cp_device *screen = cp->dev;
   struct cp_shader_binary *fs = state->fs;

   *t_interp = 0.0f;
   *t_shade = 0.0f;
   *t_composite = 0.0f;
   bool have_fs = cp_shader_has_standalone_exec(fs);
   if (!have_fs || !vs_output_buf || !state->vs ||
       !screen->kernels.abuf_interpolate || !num_quads)
      return false;
   if (composite && (!screen->kernels.abuf_composite || !ab->shade_slot ||
                     !ab->clist || !color_data))
      return false;

   unsigned num_fs_inputs = MIN2(fs->nir_num_inputs, CP_MAX_FS_INPUTS);
   unsigned num_vs_outputs = state->vs->nir_num_outputs
      ? state->vs->nir_num_outputs : 2;
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
      .prim_refs = prim_refs,
      .vs_out = vs_output_buf,
      .pixel_list = pixel_list,
      .counter = counter,
      .fs_in = fs_in,
      .frag_coord = frag_coord,
      .coverage = coverage,
      .front_face = front_face,
      .front_ccw = state->raster.front_ccw,
      .width = w, .height = h,
      .vs_out_stride = num_vs_outputs * 16,
      .fs_in_stride = fs_in_stride,
      .num_fs_inputs = num_fs_inputs,
      .max_pixels = num_slots,
      .quad_width = (w + 1) / 2,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
      /* gl_FragCoord.z is the depth the rasterizer tested with, so the
       * interpolator takes the same viewport depth transform. */
      .depth_scale = depth_scale, .depth_translate = depth_translate,
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
   cp_fs_interp_setup(cp, state, scope, info, fs, num_fs_inputs, num_vs_outputs, &interp);

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

   enum cp_shader_exec_mode abuf_mode = CP_SHADER_EXEC_CLASSIC;
   bool use_hw_inline = cp_texture_cache_available(cp, state, fs, true);
   if (cp->hardware_texture.fatal)
      return false;
   bool allow_inshader = use_hw_inline || (!cp_debug->no_inline_fs &&
      (!cp_debug->no_fused_abuf_interp || cp_debug->force_fused_fs));
   if (allow_inshader) {
      if (use_hw_inline)
         abuf_mode = cp->hardware_texture.launch_mode;
      else if (cp_debug->inline_fs && !cp_debug->force_fused_fs &&
               fs->exec[CP_SHADER_EXEC_INLINE].kernel)
         abuf_mode = CP_SHADER_EXEC_INLINE;
      else if (fs->exec[CP_SHADER_EXEC_FUSED].kernel) {
         abuf_mode = CP_SHADER_EXEC_FUSED;
         if (cp_debug->inline_fs && !cp_debug->force_fused_fs &&
             (cp_debug->shader_stats || cp_debug->dump_ir ||
              cp_debug->dump_ptx) && !fs->inline_fallback_reported) {
            fprintf(stderr, "cudavk: inline fragment binary unavailable; "
                    "using proven fused helper binary\n");
            fs->inline_fallback_reported = true;
         }
      }
   }
   bool use_fused_interp = abuf_mode != CP_SHADER_EXEC_CLASSIC;
   if (allow_inshader && !use_fused_interp && !fs->fused_fallback_reported) {
      fprintf(stderr, "cudavk: in-shader interpolation unavailable; "
              "using classic standalone interpolation\n");
      fs->fused_fallback_reported = true;
   }

   if (!use_fused_interp) {
      void *interp_params[] = { &interp };
      CUfunction kernel = seg && seg->ranges
         ? screen->kernels.abuf_interpolate_ranges
         : screen->kernels.abuf_interpolate;
      CP_LAUNCH(kernel, (num_quads + 255) / 256, 1, 1, 256, 1, 1, 0,
                cp->stream, interp_params, NULL);
   }

   enum cp_shader_exec_mode fs_mode = use_fused_interp
      ? abuf_mode : CP_SHADER_EXEC_CLASSIC;
   if (!cp_fs_launch_shader(cp, state, fs, fs_mode, counter, fs_in,
                            fs_in_stride, fs_out, frag_coord, discard_mask,
                            front_face, coverage,
                            use_fused_interp ? interp_dev : 0, num_slots,
                            ab->timing ? ab->ev[13] : 0, batch_rows, NULL))
      return false;
   cp_abuf_mark(ab, ab->ev[14], cp->stream);

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
            scope->fb.color_encoding, 0),
         .max_layers = ab->max_layers,
         .blend = cp_blend_desc_for(state),
      };
      void *p[] = { &ca };
      /* The worklist's own length, read in the drain that decided this path;
       * the kernel still reads it from the device, so a grid sized to it is a
       * statement about how much of the machine to use rather than a bound
       * anything depends on. */
      unsigned nwork = num_covered ? num_covered : w * h;
      cp_nvtx_push("composite");
      cp_abuf_mark(ab, ab->ev[15], cp->stream);
      CUresult ce = cp_launch(cp, screen->kernels.abuf_composite,
                                   (nwork + 255) / 256, 1, 1, 256, 1, 1,
                                   0, cp->stream, p, NULL);
      cp_abuf_mark(ab, ab->ev[16], cp->stream);
      cp_nvtx_pop();   /* composite */
      if (ce != CUDA_SUCCESS) {
         fprintf(stderr, "abuffer: cp_abuf_composite launch failed (%d)\n", ce);
         return false;
      }
      cp_texture_attachment_written(cp, &scope->fb);
   }

   /*
    * Reading the events means draining, which is a real cost on a path whose
    * point is not to. Only done when the breakdown is being printed.
    */
   if (ab->timing) {
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
 * Launch the standalone clip kernel from a deferred argument block. The
 * fallback for any path that needs the clipped buffer but cannot (or must
 * not) launch the fused clip+rasterize kernel; identical to the immediate
 * launch the non-fused path performs. Returns whether the launch succeeded —
 * under memory pressure cuLaunchKernel can refuse (the clip kernel's local
 * memory pool is real), and the caller then has to fail the way the classic
 * path fails: rasterize unclipped rather than read a buffer nothing wrote.
 */
struct cp_pending_clip {
   struct cp_clip_args args;
   bool pending;
   CUdeviceptr positions;
   uint32_t num_triangles;
   unsigned rast_num_triangles;
   uint32_t rect_prim_shift;
   unsigned fs_prim_shift;
};

static bool
cp_flush_pending_clip(struct cp_context *cp, struct cp_device *screen,
                      struct cp_pending_clip *clip)
{
   if (!clip->pending)
      return true;
   clip->pending = false;
   void *params[] = { &clip->args };
   CUresult err = cp_launch(cp, screen->kernels.clip_triangles,
                                 (clip->args.num_triangles + 63) / 64, 1, 1,
                                 64, 1, 1, 0, cp->stream, params, NULL);
   if (err != CUDA_SUCCESS)
      fprintf(stderr, "cudavk: clip launch failed (%d)\n", err);
   return err == CUDA_SUCCESS;
}

/* Report a broken fused path without flooding one line per draw: the classic
 * launch still recovers correct work, but silently losing the optimization
 * would make a bad kernel handle or resource regression hard to diagnose. */
static void
cp_warn_fused_rast_refused(CUresult err)
{
   static int warned;
   if (p_atomic_cmpxchg(&warned, 0, 1) == 0)
      fprintf(stderr, "cudavk: fused clip+raster launch failed (%d); "
              "using separate kernels\n", err);
}

/*
 * Undo the optimistic commit a deferred clip made to a rasterize-argument
 * block, so a draw whose clip launch failed rasterizes the unclipped
 * geometry — exactly the state the classic path's failure branch leaves.
 */
static void
cp_rast_args_unclip(struct cp_rasterize_args *ra, uint64_t positions,
                    uint32_t num_triangles, uint32_t rect_prim_shift)
{
   ra->positions = positions;
   ra->prim_refs = 0;
   ra->tri_count = 0;
   ra->active_ids = 0;
   ra->num_triangles = num_triangles;
   ra->rect_prim_shift = rect_prim_shift;
}

static void
cp_restore_preclip(const struct cp_pending_clip *clip,
                   struct cp_rasterize_args *ra, CUdeviceptr *vs_output_buf,
                   unsigned *rast_num_triangles, unsigned *fs_prim_shift)
{
   cp_rast_args_unclip(ra, clip->positions, clip->num_triangles,
                       clip->rect_prim_shift);
   *vs_output_buf = clip->positions;
   *rast_num_triangles = clip->rast_num_triangles;
   *fs_prim_shift = clip->fs_prim_shift;
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
cp_draw_execute_batch(struct cp_context *cp, const struct cp_draw_batch *batch)
{
   cp->plan.flushes++;
   unsigned batch_draws = batch->ndraws;
   const struct cp_draw_call *info = &batch->info;
   unsigned drawid_offset = batch->drawid_offset;
   const struct cp_draw_range *draws = batch->draws;
   unsigned num_draws = 1;
   const uint64_t *vs_ubo_table = batch->compact_rows ? batch->vs_ubos : NULL;
   const uint64_t *fs_ubo_table = batch->compact_rows ? batch->fs_ubos : NULL;
   const uint32_t *draw_ids = batch->compact_rows ? batch->draw_ids : NULL;
   const uint32_t *instance_counts = batch->compact_rows ?
                                     batch->instance_counts : NULL;
   const uint64_t *vb_table = batch->compact_rows ? batch->vb_bases : NULL;
   const struct cp_rect *scissors = batch->compact_rows ?
                                    batch->scissors : NULL;

   assert(batch_draws > 0);
   const struct cp_draw_state *state = &batch->state;
   const struct cp_rect *draw_scissor = &batch->scissors[batch_draws - 1];
   struct cp_device *screen = cp->dev;
   const struct cp_fb_desc *fb = &batch->scope.fb;

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

   if (cp_debug->debug_rows)
      fprintf(stderr, "exec: batch_draws=%u num_draws=%u fs_tbl=%d "
              "fs_ndraws=%u\n", batch_draws, num_draws, fs_ubo_table ? 1 : 0,
              cp->fs_batch.ndraws);

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
       !(state->fs && state->fs->writes_memory))
      do { if (cp_debug->debug_draw)
            fprintf(stderr, "  skipped: no colour or depth attachment\n");
         return; } while (0);
   if (num_draws == 0 || draws[0].count == 0)
      return;


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

   /*
    * A clip launch waiting to be fused into the first rasterizer launch.
    * When the fused clip+stage1 kernel is usable, the clip block below
    * records its arguments here instead of launching, and whichever
    * chain site consumes the geometry first launches the fused kernel. Any
    * path that needs the clipped buffer without launching a chain flushes
    * it as the classic standalone kernel instead.
    */
   /* The complete deferred launch and the state its optimistic commit
    * replaced. Keeping the rollback together prevents new clip-sensitive
    * raster fields from growing a second loose snapshot list. */
   struct cp_pending_clip pending_clip = {0};

   /* Resolved when the framebuffer was bound; NULL for a depth-only pass. */
   void *color_data = fb->color;
   if (cp_debug->debug_draw && !color_data)
      fprintf(stderr, "  color=(nil) reason: nr_cbufs=%u\n", fb->nr_cbufs);

   unsigned w = fb->width;
   unsigned h = fb->height;
   unsigned fb_samples = MAX2(batch->scope.attachment_samples, 1u);

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
   float vp_scale_x = state->viewport.scale[0];
   float vp_scale_y = state->viewport.scale[1];
   float vp_trans_x = state->viewport.translate[0];
   float vp_trans_y = state->viewport.translate[1];
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
   bool rows_stable = state->blend.enable ||
      (state->fs && state->fs->reads_const_bufs);
   bool per_draw_rects = scissors && batch_draws > 1 && rows_stable &&
      state->raster.scissor;
   if (state->raster.scissor && !per_draw_rects) {
      const struct cp_rect *sc0 =
         scissors ? &scissors[0] : &(*draw_scissor);
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
      /* The viewport's depth transform, which until now was computed here
       * (cpvk_cmd.c:1905) and read nowhere. */
      .depth_scale = state->viewport.scale[2],
      .depth_translate = state->viewport.translate[2],
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
                 : cp_cull_mode(&state->raster),
      .front_face = state->raster.front_ccw,
      /* Points have no winding to cull and no edges to test; the square comes
       * from gl_PointSize, wherever the vertex shader put it. */
      .num_samples = fb_samples,
      .point_mode = info->mode == MESA_PRIM_POINTS,
      .psiz_slot = state->vs
         ? cp_slot_for_location(state->vs->out_location,
                                CP_MAX_IO_SLOTS, VARYING_SLOT_PSIZ)
         : -1,
      .depthbuf = cp->depthbuf,
      .depth_test = state->depth.depth_enabled,
      .depth_func = state->depth.depth_func,
      .depth_key_invert = state->depth.depth_enabled &&
         (state->depth.depth_func == CP_FUNC_GREATER ||
          state->depth.depth_func == CP_FUNC_GEQUAL),
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
   if (cp->device_fatal)
      return;

   /*
    * The batch's uploads have survived the reclaim above; everything this
    * draw uploads from here is its own, and the next draw may rewind freely.
    */
   cp->batch_uploads_live = false;

   /* A shader with no declared inputs needs no vertex buffer: it builds its
    * positions from gl_VertexIndex, which is how a fullscreen pass is drawn.
    * A batch carries its own snapshot of the bindings, so the live state —
    * which the next draw may have rebound over — is not consulted for one. */
   bool has_vs = state->vs &&
      state->vs->exec[CP_SHADER_EXEC_CLASSIC].kernel &&
                 (vb_table != NULL ||
                  (state->num_vertex_buffers > 0 && state->vb_base[0]) ||
                  state->num_vertex_elements == 0);

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

   /* If no VS will run, pack positions from VB directly (passthrough).
    * When a VS is present, skip this — VS output provides positions. */
   if (!has_vs) {
      if (state->num_vertex_buffers > 0 && state->vb_base[0]) {
         {
            /* Base and offset were folded together when the buffer was
             * bound. */
            char *vb_start = (char *)(uintptr_t)state->vb_base[0];
            unsigned stride = state->vertex_stride ? state->vertex_stride : 16;

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
   /* Whether cp_vertex_fetch seeded this batch's counters, which decides
    * whether the launches after it still clear them. Declared here because
    * the raster passes below are outside the block that sets it. */
   bool fetch_fold = false;

   if (state->vs && state->vs->exec[CP_SHADER_EXEC_CLASSIC].kernel) {
      CP_NVTX_SCOPE("vertex");
      void *vb_data2 = (state->num_vertex_buffers > 0)
         ? (void *)(uintptr_t)state->vb_base[0] : NULL;

      /* A vertex shader may build its positions from gl_VertexIndex alone and
       * declare no inputs at all, which is how a fullscreen pass is drawn.
       * That draw binds no vertex buffer, and skipping it loses every
       * post-processing and skybox pass. A batch's bindings are its snapshot,
       * not the live state. */
      if (vb_table != NULL || vb_data2 || state->num_vertex_elements == 0) {
         unsigned stride;
         unsigned num_vs_outputs = state->vs->nir_num_outputs ? state->vs->nir_num_outputs : 2;
         unsigned out_stride = num_vs_outputs * 16;

         vs_output_buf = cp_scratch_alloc_device(cp, (size_t)total_verts * out_stride);

         /*
          * Whether this draw's vertex shader gathers its own attributes.
          *
          * The decision is per launch and everything below follows it: the
          * packed input buffer, its clear, the fetch launch, the id arrays and
          * the batch-row array all exist only for the classic form, and the
          * counter seeding moves with the launch that carries it. It selects a
          * different CUfunction from an independently linked module, which is
          * what makes CUDAVK_FUSED_VFETCH=0 a real revert rather than a
          * skipped call inside the same register allocation.
          *
          * CUDAVK_DEBUG_VFETCH reads the packed buffer back to the host, so
          * a draw being traced that way keeps the buffer and the launch.
          */
         const bool fused_vfetch =
            cp_debug->fused_vfetch && !cp_debug->no_fused_vfetch &&
            !cp_debug->debug_vfetch &&
            state->vs->exec[CP_SHADER_EXEC_VS_FETCH].kernel &&
            state->num_vertex_elements <= CP_MAX_VERTEX_ELEMENTS_VF;

         /* Build VS input buffer on GPU: the vertex fetch kernel gathers
          * attributes in parallel, one thread per assembled vertex. */
         unsigned vs_in_stride = state->num_vertex_elements * 16;
         CUdeviceptr vs_input_buf = (vs_in_stride && !fused_vfetch)
            ? cp_scratch_alloc_device(cp, (size_t)total_verts * vs_in_stride)
            : 0;
         if (!vs_output_buf || (vs_in_stride && !fused_vfetch && !vs_input_buf)) {
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

         if (!refs && !fused_vfetch) {
            /* A batch's assembled vertices span several ranges of the index
             * buffer, so the buffer is no longer the id array for them — the
             * fetch kernel has to publish one. */
            if (indexed && ib_base && info->index_size == 4 &&
                instance_count == 1 && batch_draws == 1) {
               vid_buf = (CUdeviceptr)(uintptr_t)ib_base +
                         (size_t)draws[0].start * 4;
            } else if (state->vs->reads_vertex_id) {
               out_vid = cp_scratch_alloc_device(cp, (size_t)total_verts * 4);
               if (!out_vid) { FREE(refs); return; }
               vid_buf = out_vid;
            }

            if (state->vs->reads_instance_id) {
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
          * value it has always had, which is what makes CUDAVK_BATCH_MAX=1
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
             * that reads either has any use for it — and a fused shader has
             * the row in a register, so nobody has to write it down. */
            if (!fused_vfetch &&
                (state->vs->reads_const_bufs ||
                 state->vs->reads_draw_params)) {
               batch_rows = cp_scratch_alloc_device(cp, (size_t)total_verts * 4);
               if (!batch_rows) { FREE(refs); return; }
            }

            /* The fragment stage searches the same table, from the primitive
             * rather than from the vertex; see cp_fs_interp_args. */
            if (fs_ubo_table)
               cp->fs_batch.slices = slices_dev;

            if (cp_debug->debug_rows)
               fprintf(stderr, "slices: n=%u set=%d verts=[%u %u %u]\n",
                       batch_draws, slices_dev ? 1 : 0, slices[0].vert_begin,
                       batch_draws > 1 ? slices[1].vert_begin : 0,
                       batch_draws > 2 ? slices[2].vert_begin : 0);

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
            .num_elements = state->num_vertex_elements,
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

         for (unsigned e = 0; e < state->num_vertex_elements && e < 16; e++) {
            const struct cp_vertex_elem *elem = &state->velem[e];
            unsigned vb_idx = elem->vertex_buffer_index;
            if (vb_idx < state->num_vertex_buffers && state->vb_base[vb_idx])
               vf_args.vb_bases[vb_idx] = state->vb_base[vb_idx];
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

         /*
          * The clears the launches after this one would each have issued as
          * their own device operation. The fetch runs once per executed
          * batch, on this stream, ahead of the clipper and the rasterizer, so
          * it can write them itself -- but only if it runs at all, and only
          * for a launch that has a thread zero, so the eager clears stay on
          * every other path.
          */
         const bool fetch_runs =
            (vs_input_buf || out_vid || out_iid || batch_rows) &&
            total_verts > 0;
         /*
          * A fused draw has no fetch launch to seed from, so the seeding rides
          * on the vertex shader instead — the same kernel, one launch earlier
          * in the chain, still ahead of the clipper and the rasterizer on this
          * stream. Without this move, iteration 26 S2 would be silently
          * reverted for every admitted shader.
          */
         const bool vs_runs = total_verts > 0;
         fetch_fold = !cp_debug->no_fetch_fold &&
                      (fused_vfetch ? vs_runs : fetch_runs);

         /*
          * The clipper's counter, hoisted here from the clip block below so
          * the fetch can seed it. Four bytes; its seed is not always zero,
          * and the expression that decides it is reproduced there unchanged.
          */
         const unsigned max_clipped_early = num_triangles * CP_CLIP_MAX_OUT;
         const bool clip_block_runs = screen->kernels.clip_triangles &&
                                      num_vs_outputs <= CP_MAX_CLIP_SLOTS;
         bool stable_clip_early = state->blend.enable ||
            (batch_draws > 1 && state->fs && state->fs->reads_const_bufs);
         CUdeviceptr clip_count_early = 0, active_ids_early = 0;
         if (fetch_fold && clip_block_runs) {
            clip_count_early = cp_scratch_alloc_device(cp, 4);
            if (stable_clip_early)
               active_ids_early = cp_scratch_alloc_device(
                  cp, (size_t)max_clipped_early * sizeof(uint32_t));
         }

         /*
          * The negative control for the move: an admitted fused draw that
          * does not seed, while the host still skips the clears the seeding
          * replaced. If a following batch renders correctly anyway, this gate
          * is not covering the counters it claims to.
          */
         const bool skip_seed = fused_vfetch && cp_debug->vfetch_skip_seed;

         if (fetch_fold && !skip_seed) {
            /*
             * This seeding rides on a launch existing -- the fetch's when the
             * shader declined, the fused vertex shader's when it did not. Both
             * read these fields out of the same block, and moving the job with
             * the launch is what keeps iteration 26 S2 from being silently
             * reverted here: the counters would otherwise go back to their own
             * clears without anything failing. CUDAVK_VFETCH_SKIP_SEED is
             * the control that proves it, and it makes tests fail.
             */
            /* Whichever raster pass this batch takes clears these three words
             * first; the fill relaunch inside a pass still clears its own. */
            vf_args.seed_counts = cp->cur_qset.counts;
            vf_args.seed_clip_count = clip_count_early;
            vf_args.clip_seed = stable_clip_early && !active_ids_early
               ? max_clipped_early : 0u;
         }

         /* Nothing to gather when the shader declares no inputs — but it still
          * runs if the ids are wanted, since deriving those is now its job
          * too and a shader with no inputs may still read gl_VertexIndex. */
         if (!fused_vfetch && (vs_input_buf || out_vid || out_iid || batch_rows)) {
            /*
             * Small by call count, bulk by bytes: this clears about 90 MB a
             * frame on the old capture, which cuMemsetD8Async does in one
             * kernel at DRAM speed. Folding it into the gather kernel row by
             * row was measured at 1.84 ms a frame slower even with 16-byte
             * stores, so it stays a memset. Only the counters below are worth
             * folding.
             */
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
            for (unsigned e = 0; e < state->num_vertex_elements && e < 8; e++)
               fprintf(stderr, "  elem%u vb=%u off=%u stride=%u div=%u sz=%u\n",
                       e, state->velem[e].vertex_buffer_index,
                       state->velem[e].src_offset,
                       state->velem[e].src_stride,
                       state->velem[e].instance_divisor,
                       vf_args.elem_attr_size[e]);
            for (unsigned v = 0; in && v < nfetch; v++) {
               fprintf(stderr, "  vfetch v%u:", v);
               for (unsigned e = 0; e < state->num_vertex_elements && e < 8; e++)
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

         /*
          * Or not in a block at all. The argument block below already carries
          * a scalar area for the batch row and its mask, and already hands the
          * shader device addresses into itself, so two more words there hold
          * these two scalars just as well -- and the eight-byte copy that
          * carried them was the single most frequent host-to-device operation
          * in the driver, 203.7 of them per frame on the old capture.
          *
          * The shader ABI does not move: args[0] and args[3] are still two
          * device addresses it dereferences.
          */
         const bool meta_in_block = !cp_debug->no_meta_fold;
         CUdeviceptr meta_dev = 0;
         if (!meta_in_block) {
            meta_dev = cp_upload(cp, &meta, sizeof(meta));
            if (!meta_dev) {
               FREE(refs);
               return;
            }
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
         const size_t vs_scal_off = vs_args_bytes;   /* row 0, then the mask,
                                                      * then vcount and stride
                                                      * when they are folded */
         const size_t vs_scal_bytes = meta_in_block ? 32 : 16;
         const size_t vs_tbl_off = vs_args_bytes + vs_scal_bytes;
         /* The per-draw parameter rows behind the uniform table — see
          * CP_ARG_DRAW_PARAM_STRIDE. Same block, same upload. */
         const size_t vs_dp_off = vs_tbl_off +
            (size_t)batch_draws * CP_ARG_UBO_STRIDE * sizeof(uint64_t);
         /*
          * The fetch's own argument block, behind the draw-parameter rows and
          * inside the same upload. It travels today as a by-value kernel
          * parameter of a launch that is about to stop existing; putting it in
          * its own copy would hand back the operation this iteration removes,
          * so it rides in the block the host already builds and sends once.
          */
         size_t vs_blk_bytes = vs_dp_off +
            (size_t)batch_draws * CP_ARG_DRAW_PARAM_STRIDE * sizeof(uint32_t);
         const size_t vs_vf_off = fused_vfetch
            ? (vs_blk_bytes + 15u) & ~(size_t)15u : 0;
         if (fused_vfetch)
            vs_blk_bytes = vs_vf_off + sizeof(struct cp_vertex_fetch_args);

         void *vs_blk = NULL;
         CUdeviceptr vs_args_dev = cp_upload_begin(cp, vs_blk_bytes, &vs_blk);
         if (!vs_args_dev) {
            FREE(refs);
            return;
         }
         memset(vs_blk, 0, vs_blk_bytes);

         void **vs_args_host = (void **)vs_blk;
         vs_args_host[0] = meta_in_block
            ? (void*)(uintptr_t)(vs_args_dev + vs_scal_off + 8)
            : (void*)(uintptr_t)(meta_dev +
                                 offsetof(struct cp_vs_meta, vcount));
         vs_args_host[1] = NULL;
         vs_args_host[2] = (void*)(uintptr_t)vs_input_buf;
         vs_args_host[3] = meta_in_block
            ? (void*)(uintptr_t)(vs_args_dev + vs_scal_off + 12)
            : (void*)(uintptr_t)(meta_dev +
                                 offsetof(struct cp_vs_meta, stride));
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
         if (fused_vfetch) {
            memcpy((char *)vs_blk + vs_vf_off, &vf_args, sizeof(vf_args));
            vs_args_host[CP_ARG_SLOT_VS_FETCH] =
               (void*)(uintptr_t)(vs_args_dev + vs_vf_off);
         }

         uint32_t *vs_scal = (uint32_t *)((char *)vs_blk + vs_scal_off);
         vs_scal[0] = 0;
         vs_scal[1] = batch_rows ? 0xFFFFFFFFu : 0u;
         if (meta_in_block) {
            vs_scal[2] = meta.vcount;
            vs_scal[3] = meta.stride;
         }

         uint64_t *vs_tbl = (uint64_t *)((char *)vs_blk + vs_tbl_off);
         for (unsigned d = 0; d < batch_draws; d++) {
            const uint64_t *row = vs_ubo_table
               ? vs_ubo_table + (size_t)d * CP_ARG_UBO_STRIDE : NULL;
            for (unsigned i = 0; i < state->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
               vs_tbl[d * CP_ARG_UBO_STRIDE + i] =
                  row ? row[i] : state->vs_ubos[i];
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
         for (unsigned i = 0; i < state->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
            vs_args_host[CP_ARG_UBO_BASE + i] =
               (void *)(uintptr_t)state->vs_ubos[i];

         cp_upload_end(cp, vs_args_dev, vs_blk, vs_blk_bytes);

         void *vs_arg_ptr = (void*)(uintptr_t)vs_args_dev;
         void *vs_params[] = { &vs_arg_ptr };
         /*
          * The two forms are two CUfunctions from two independently linked
          * modules; nothing about one transfers to the other. A fused kernel
          * launched with a null fetch block would gather from address zero, so
          * that is an internal error rather than a soft path — there is no
          * meaningful "fetch disabled" reading of this kernel.
          */
         const struct cp_shader_exec *vs_exec = fused_vfetch
            ? &state->vs->exec[CP_SHADER_EXEC_VS_FETCH]
            : &state->vs->exec[CP_SHADER_EXEC_CLASSIC];
         assert(!fused_vfetch || vs_args_host[CP_ARG_SLOT_VS_FETCH]);

         /* Which shader ran, and how often, so that the fused-fetch admission
          * share can be weighted by launches rather than by shader count. */
         if (state->vs->vs_census) {
            p_atomic_inc(&state->vs->vs_census->launches);
            if (fused_vfetch)
               p_atomic_inc(&state->vs->vs_census->launches_fused);
         }

         /* Compiled shaders grid-stride; see the fragment launch. */
         CUresult vs_err = cp_launch(cp, vs_exec->kernel,
            MIN2((total_verts + 255) / 256, 4096u), 1, 1, 256, 1, 1,
            0, cp->stream, vs_params, NULL);

         if (vs_err == CUDA_SUCCESS) {
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
               /* Allocated above the fetch when that launch seeded it. */
               CUdeviceptr clip_count = clip_count_early
                  ? clip_count_early : cp_scratch_alloc_device(cp, 4);
               /* Exact worst-case table, generation-owned beside the original
                * VS output and clipped scratch. Refusal keeps clip+copy. */
               CUdeviceptr prim_refs = !cp_debug->no_prim_refs &&
                                       !cp_debug->debug_fs
                  ? cp_scratch_alloc_device(
                       cp, (size_t)max_clipped * sizeof(uint64_t)) : 0;

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
                     fprintf(stderr, "cudavk: no scratch for clipping %u "
                             "triangles — this draw is rasterized unclipped "
                             "and may be visibly wrong.\n", num_triangles);
                  }
               }

               if (clipped && clip_count) {
                  /*
                   * A blended draw is composited in primitive order, so its
                   * primitives have to *be* in submission order — which
                   * compaction by atomicAdd does not promise. This applies to
                   * one draw too: the older GFXR capture draws each text label
                   * as black outline primitives followed by coloured fill
                   * primitives, and compaction randomly put the outline on top.
                   * Stable mode gives every input triangle a fixed slot range
                   * and retires the ones it does not fill, so the rasterizer
                   * skips holes without changing primitive order.
                   *
                   * An opaque batch whose fragment shader reads a constant
                   * buffer needs it too: the primitive index has to name the
                   * input triangle, or cp_write_batch_rows() maps fragments
                   * to the wrong draw's material. An opaque batch whose
                   * shader reads none keeps the compacting path — stable mode
                   * rasterizes the whole 4x slot array, holes and all, and
                   * multithreading paid 72% for ordering nothing consumes.
                   */
                  bool stable_clip = state->blend.enable ||
                     (batch_draws > 1 && state->fs &&
                      state->fs->reads_const_bufs);
                  /* Stable IDs no longer require rasterizing all seven unused
                   * slots beside the usual one-triangle output.  The clipper
                   * appends live fixed-slot IDs here; allocation failure keeps
                   * the proven hole-filled path as a correctness fallback. */
                  CUdeviceptr active_ids = active_ids_early ? active_ids_early
                     : (stable_clip
                        ? cp_scratch_alloc_device(
                             cp, (size_t)max_clipped * sizeof(uint32_t))
                        : 0);

                  if (cp_debug->debug_clip)
                     fprintf(stderr, "clip: tris=%u batch_draws=%u stable=%d "
                             "reads_cb=%d\n", num_triangles, batch_draws,
                             (int)stable_clip,
                             state->fs ? (int)state->fs->reads_const_bufs : -1);

                  /* Seeded by cp_vertex_fetch when the fold is on; the value
                   * is the same expression either way. */
                  if (!fetch_fold)
                     cuMemsetD32Async(clip_count,
                                      stable_clip && !active_ids ? max_clipped : 0,
                                      1, cp->stream);

                  struct cp_clip_args clip = {
                     .vs_out = vs_output_buf,
                     .out = clipped,
                     .out_count = clip_count,
                     .active_ids = active_ids,
                     .prim_refs = prim_refs,
                     .num_triangles = num_triangles,
                     .num_slots = num_vs_outputs,
                     .max_triangles = max_clipped,
                     .stable = stable_clip,
                  };
                  /*
                   * Fused path: hold the launch, and let the first
                   * rasterizer chain below run clip+stage1 as one kernel.
                   * The argument block is complete here — only the
                   * launch moves. Instrumented builds keep the classic
                   * sequence so every debug census sees the kernels it was
                   * written against.
                   */
                  CUresult clip_err = CUDA_SUCCESS;
                  if (screen->kernels.clip_rast_fused &&
                      !cp_debug->no_fused_rast && !cp_kernels_instrumented()) {
                     pending_clip.args = clip;
                     pending_clip.pending = true;
                     pending_clip.positions = rast_args.positions;
                     pending_clip.args.num_triangles = rast_args.num_triangles;
                     pending_clip.rast_num_triangles = rast_num_triangles;
                     pending_clip.rect_prim_shift = rast_args.rect_prim_shift;
                     pending_clip.fs_prim_shift = cp->fs_batch.prim_shift;
                  } else {
                     void *clip_params[] = { &clip };
                     clip_err = cp_launch(cp,
                        screen->kernels.clip_triangles,
                        (num_triangles + 63) / 64, 1, 1, 64, 1, 1,
                        0, cp->stream, clip_params, NULL);
                  }

                  if (clip_err == CUDA_SUCCESS) {
                     vs_output_buf = clipped;
                     rast_args.positions = clipped;
                     rast_args.prim_refs = prim_refs;
                     rast_args.tri_count = clip_count;
                     rast_args.active_ids = active_ids;
                     /* active_ids contains fixed IDs up to max_clipped - 1;
                      * num_triangles is their validity bound, while tri_count
                      * remains the compact amount of initial work. */
                     if (active_ids)
                        rast_args.num_triangles = max_clipped;
                     rast_num_triangles = max_clipped;
                     if (stable_clip) {
                        cp->fs_batch.prim_shift = CP_CLIP_PRIM_SHIFT;
                        rast_args.rect_prim_shift = CP_CLIP_PRIM_SHIFT;
                     }
                  } else {
                     fprintf(stderr, "cudavk: clip launch failed (%d)\n",
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
      fprintf(stderr, "cudavk: [samples=%u] draw %u tris (%u instances), fb=%ux%u, "
              "vp=[%.0f,%.0f,%.0f,%.0f] stride=%u scale=[%.1f,%.1f] color=%p\n",
              fb_samples, num_triangles, instance_count, w, h, vp_x, vp_y, vp_w, vp_h,
              state->vertex_stride,
              state->viewport.scale[0], state->viewport.scale[1], color_data);
      for (unsigned e = 0; e < state->num_vertex_elements && e < 4; e++)
         fprintf(stderr, "  elem[%u]: offset=%u fmt=%u vb=%u\n", e,
                 state->velem[e].src_offset, state->velem[e].conv,
                 state->velem[e].vertex_buffer_index);
   }

   /* 3-stage adaptive rasterize. The queue counters are zeroed inside the pass
    * loop below, which runs for pass 0 as well — clearing them here too was
    * two host calls per draw that the first pass immediately repeated. */
   struct cp_rast_queues rast_queues = {
      .nontrivial = cp->cur_qset.nontrivial,
      .nontrivial_count = cp->cur_qset.counts,
      .huge_tiles = cp->cur_qset.huge_tiles,
      .huge_count = cp->cur_qset.counts + sizeof(uint32_t),
      .setup_cache = cp->cur_qset.setup_cache,
      .setup_count = cp->cur_qset.counts + 2 * sizeof(uint32_t),
      .setup_capacity = cp->cur_qset.setup_cache ? CP_SETUP_CACHE_CAPACITY : 0,
      .mode = CP_QUEUE_FILL,
   };

   /*
    * Alpha-tested geometry needs more than one go. Visibility resolves before
    * the shader runs, so a fragment that discards has already displaced the one
    * behind it — a leaf's transparent texel hides the leaf further back. Each
    * pass records what discarded where and repeats, letting the next fragment
    * win, until every pixel has settled or the layers run out.
    */
   bool retry = state->fs && state->fs->uses_discard &&
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
   bool fs_side_effects = state->fs && state->fs->writes_memory;

   /*
    * A depth-only pass: a depth attachment, no colour, and a fragment shader
    * that exists only to be allowed to discard. A shadow map is the whole
    * reason such a pass exists, and this driver committed no depth for one --
    * cp_fs_writeback is the only writer of cp->depthbuf, and it was launched
    * only when there was a colour to blend, so the fragment stage was skipped
    * altogether and every draw in the pass tested against, and left behind,
    * the clear value. favorite2's 2080x2080 D16 shadow map came out uniformly
    * 65535 and the surfaces that sample it went black.
    *
    * The depth *write mask*, not the depth test, is what decides: a draw with
    * the test off and the mask on still writes depth, which is the same
    * condition cp_fs_writeback itself uses.
    */
   bool depth_commit = !color_data && fb->has_zs &&
                       state->depth.depth_writemask;

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
               ((color_data && state->blend.enable) ||
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
              peel ? 1 : 0, retry ? 1 : 0, state->blend.enable ? 1 : 0, w, h);
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
   struct cp_abuf *ab = cp->abuf;
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
   if (cp_abuf_enabled(ab) && peel && w && h && !ab->disabled) {
      /* Why a draw is not eligible is a question about one run, and this is a
       * path every blended draw now reaches — so it is said once, and only
       * when CUDAVK_ABUFFER_DEBUG asked. */
      static int said = 0;
      if (!ab->debug)
         said = 1;
      /*
       * Asking for the comparison against the peel loop without the
       * instrumentation to make it is not a state to render in: the log the
       * comparison reads is written by kernels CUDAVK_ABUF_COMPILE=0
       * refused, so every pixel reads as a mismatch. Say so once and render
       * the ordinary way rather than print a thousand false ones.
       */
      if (ab->verify && !screen->kernels.abuf_peel_log) {
         fprintf(stderr, "abuffer: CUDAVK_ABUFFER_VERIFY needs the "
                 "instrumentation CUDAVK_ABUF_COMPILE=0 refused — "
                 "not verifying\n");
         ab->verify = 0;
         ab->composite = 1;
      }
      if (!screen->kernels.abuf_quad_fill) {
         if (!said++)
            fprintf(stderr, "abuffer: kernels not compiled in — skipped\n");
         ab->disabled = true;
      } else if (fb_samples != 1) {
         if (!said++)
            fprintf(stderr, "abuffer: %u samples per pixel; peel path only, "
                    "single-sampled only — skipped\n", fb_samples);
      } else if (state->depth.depth_writemask) {
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
                    "— skipped\n", state->depth.depth_enabled ? 1 : 0,
                    state->depth.depth_writemask ? 1 : 0);
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
         abuf = cp_abuf_setup(cp, ab, w, h);
      }
   }

   /*
    * A batch too small to pay for the build.
    *
    * The A-buffer costs about ten launches whatever the draw's size: the
    * clears, three count stages, a two-level scan, the merge, the sort and the
    * composite. A draw with real depth complexity buys that back many times
    * over -- particlesystem is 11.8x slower peeling -- and a small one does
    * not, because it peels in one or two passes.
    *
    * The native front end batches blended draws whether or not the A-buffer is
    * on, which the Gallium driver does not, so it arrives here with many small
    * blended batches where that driver had few large ones. On both captures
    * the A-buffer is a net loss because of it: 8.79 ms against 6.75, and 31.12
    * against 21.89.
    *
    * So the fixed cost is asked for explicitly, from the triangle count, which
    * is known here before anything has been spent.
    *
    * CUDAVK_ABUF_MIN_TRIS is that floor and defaults to zero, which is
    * deliberate: once the bootstrap scan was fixed Crossroads measured 10.9 ms
    * with no floor against 33.0 ms at 256, and repeated 600-frame sample runs
    * were unchanged within noise. The switch stays for threshold experiments.
    */
   if (abuf && cp_debug->abuf_min_tris &&
       rast_num_triangles < cp_debug->abuf_min_tris)
      abuf = false;

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
       (!abuf || ab->grow_to || !ab->frags || !ab->composite ||
        cp->pass.nsegs >= CP_PASS_MAX_SEGS)) {
      /* The re-execution repeats the vertex work, so nothing will read this
       * clipped buffer — but flush anyway: one rare launch buys the
       * invariant that a deferred clip always runs. */
      cp_flush_pending_clip(cp, screen, &pending_clip);
      cp->pass.append_failed = true;
      return;
   }

   if (abuf && ab->grow_to) {
      uint32_t want = ab->grow_to;
      ab->grow_to = 0;
      /* cuMemFree already blocks until the device has finished; the drain is
       * written out so that this does not rest on that. */
      if (cuStreamSynchronize(cp->stream) != CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return;
      }
      if (!cp_abuf_size_arrays(cp, ab, want)) {
         if (cp->device_fatal)
            return;
         abuf = false;
      }
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
         if (!cp_debug->no_counter_block) {
            /* One clear for every scalar counter this draw will fill: the
             * worklist counts, the block counts and the debug counters are
             * written only by kernels launched further down, so clearing them
             * here is the same zero at a cheaper price. */
            cuMemsetD32Async(ab->counters, 0, CP_ABUF_BLOCK_WORDS, cp->stream);
         } else {
            cuMemsetD32Async(ab->sum3, 0, 3, cp->stream);
            if (ab->recs)
               cuMemsetD32Async(ab->rec_cursor, 0, 1, cp->stream);
         }
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
      /* The queue counters the count pass is about to fill, unless
       * cp_vertex_fetch already seeded them for this batch. */
      if (!fetch_fold &&
          cuMemsetD32Async(cp->cur_qset.counts, 0, 3, cp->stream) !=
          CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return;
      }
      aa.abuf_mode = CP_ABUF_COUNT;
      rast_queues.mode = CP_QUEUE_FILL;
      cp_abuf_mark(ab, ab->ev[0], cp->stream);
      /* The _abuf specialisations: same rasterizer, compiled with the count
       * and fill branch live. Every other launch in this file uses the plain
       * ones, which have no A-buffer code in them at all. */
      bool fused_count = false;
      if (pending_clip.pending) {
         /* Clip and stage 1 in one launch: a thread per input triangle, so
          * the grid is the clip kernel's, an eighth of the worst-case slot
          * count the classic stage 1 is sized for — and the clip kernel's
          * 64-thread blocks, which measured 0.3 ms/frame better than 256 on
          * the old capture (more blocks spread the same small draw over
          * more SMs). Stages 2 and 3 follow unchanged — see the fused
          * kernel's comment for why they keep their own grids. */
         void *fp[] = { &pending_clip.args, &aa, &rast_queues };
         CUresult fused_err = cp_launch(cp,
            screen->kernels.clip_rast_fused_abuf,
            (pending_clip.args.num_triangles + 63) / 64, 1, 1,
            64, 1, 1, 0, cp->stream, fp, NULL);
         fused_count = fused_err == CUDA_SUCCESS;
         if (fused_count) {
            pending_clip.pending = false;
         } else {
            cp_warn_fused_rast_refused(fused_err);
         }
         if (!fused_count &&
             !cp_flush_pending_clip(cp, screen, &pending_clip)) {
            cp_restore_preclip(&pending_clip, &rast_args, &vs_output_buf,
                               &rast_num_triangles, &cp->fs_batch.prim_shift);
            cp_rast_args_unclip(&aa, pending_clip.positions,
                                pending_clip.num_triangles,
                                pending_clip.rect_prim_shift);
         }
      }
      void *ap[] = { &aa, &rast_queues };
      /* Tier 4, count pass. The queue-counter clear above is skipped whenever
       * the vertex stage seeded the counters (fetch_fold), and the episode's
       * clears run for its first segment only, so this site is expected to
       * take. The clip may or may not be the kernel in front of it. */
      if (!fused_count)
         CP_LAUNCH_AFTER(CP_PDL_ANY, CP_PDL_TIER_STAGE1,
                        screen->kernels.rasterize_stage1_abuf,
                        (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, ap, NULL);
      /* Whichever of the two actually filled the queue is the predecessor;
       * naming the wrong one costs the overlap, not the ordering. Stage 1 is
       * never a secondary here -- the queue-counter clear is in front of it. */
      CP_LAUNCH_AFTER(fused_count ? screen->kernels.clip_rast_fused_abuf
                                  : screen->kernels.rasterize_stage1_abuf,
                     CP_PDL_TIER_RASTER, screen->kernels.rasterize_stage2_abuf,
                     CLAMP((rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                     256, 1, 1, 0, cp->stream, ap, NULL);
      CP_LAUNCH_AFTER(screen->kernels.rasterize_stage2_abuf,
                     CP_PDL_TIER_RASTER, screen->kernels.rasterize_stage3_abuf,
                     CLAMP(rast_num_triangles * 8, 512u, 2048u), 1, 1,
                     64, 1, 1, 0, cp->stream, ap, NULL);
      cp_abuf_mark(ab, ab->ev[1], cp->stream);

      /* The segment is counted; everything from the scan on happens once,
       * at cp_pass_finish(). */
      if (cp->pass.appending) {
         cp_pass_record_segment(cp, &aa, &rast_queues, rast_num_triangles,
                                num_triangles, batch);
         return;
      }

      /* --- step 2: prefix sum ---
       *
       * The fill cursor is not folded in here: cp_abuf_size_arrays() can
       * replace ab->cursor between this scan and this path's fill, so the
       * clear has to stay where it is, after that decision. */
      cp_abuf_scan(cp, screen, ab, (unsigned)n, 0);
      cp_abuf_mark(ab, ab->ev[2], cp->stream);

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
      bool drain_for_count = !ab->frags || !ab->composite;

      if (!drain_for_count) {
         /* The final prefix-add clamps runs against the fragment array and
          * records overflow while it already has every offset in registers. */
      } else {
         if (cuStreamSynchronize(cp->stream) != CUDA_SUCCESS ||
             cuMemcpyDtoH(&abuf_total, ab->sum3, sizeof(uint32_t)) !=
                CUDA_SUCCESS) {
            cp_renderer_texture_fatal(cp);
            return;
         }

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
         if (!cp_abuf_size_arrays(cp, ab, abuf_total)) {
            if (cp->device_fatal)
               return;
            abuf = false;
         } else if (abuf_total > ab->capacity) {
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
      if (ab->verify) {
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
      cp_abuf_mark(ab, ab->ev[3], cp->stream);
      if (abuf_recs_filled && screen->kernels.abuf_fill_recs) {
         /* The count pass already appended every (pixel, prim) record; the
          * fill is a linear replay instead of a second rasterization. */
         void *fp[] = { &ab->recs, &ab->rec_cursor, &ab->capacity, &ab->frags,
                        &ab->offsets, &ab->counts, &ab->cursor, &ab->overflow,
                        &ab->capacity };
         CP_LAUNCH(screen->kernels.abuf_fill_recs, 1024, 1, 1, 256, 1, 1,
                        0, cp->stream, fp, NULL);
      } else {
         cuMemsetD32Async(cp->cur_qset.counts, 0, 3, cp->stream);
         aa.abuf_mode = CP_ABUF_FILL;
         rast_queues.mode = CP_QUEUE_FILL;
         void *ap[] = { &aa, &rast_queues };
         /* Tier 4, fill pass. Expected to DECLINE: the counter clear above
          * is unconditional here, and the 3.7 MB cursor clear is in front of
          * that. Offered so that the counter says so rather than a comment. */
         CP_LAUNCH_AFTER(CP_PDL_ANY, CP_PDL_TIER_STAGE1,
                        screen->kernels.rasterize_stage1_abuf,
                        (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, ap, NULL);
         CP_LAUNCH_AFTER(screen->kernels.rasterize_stage1_abuf,
                        CP_PDL_TIER_RASTER,
                        screen->kernels.rasterize_stage2_abuf,
                        CLAMP((rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                        256, 1, 1, 0, cp->stream, ap, NULL);
         CP_LAUNCH_AFTER(screen->kernels.rasterize_stage2_abuf,
                        CP_PDL_TIER_RASTER,
                        screen->kernels.rasterize_stage3_abuf,
                        CLAMP(rast_num_triangles * 8, 512u, 2048u), 1, 1,
                        64, 1, 1, 0, cp->stream, ap, NULL);
      }
      cp_abuf_mark(ab, ab->ev[4], cp->stream);

      /* --- step 4: sort. The worklist build is inside this measurement: it
       * is a prerequisite of the sort as written, not a separate step. --- */
      {
         unsigned nn = (unsigned)n, min2 = 2;
         if (cp_debug->no_counter_block)
            cuMemsetD32Async(ab->list_count, 0, 1, cp->stream);
         void *wp[] = { &ab->counts, &nn, &min2, &ab->list, &ab->list_count };
         CP_LAUNCH(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, wp, NULL);
         void *sp[] = { &ab->frags, &ab->offsets, &ab->counts, &ab->list,
                        &ab->list_count, &ab->long_runs };
         CP_LAUNCH(screen->kernels.abuf_sort, 4096, 1, 1, 256, 1, 1,
                        0, cp->stream, sp, NULL);
      }
      cp_abuf_mark(ab, ab->ev[5], cp->stream);

      /* The composite's worklist — every pixel with anything in it, not just
       * the ones worth sorting. Built here so it rides alongside the sort
       * rather than serialising behind the shading. */
      if (ab->clist) {
         unsigned nn = (unsigned)n, min1 = 1;
         if (cp_debug->no_counter_block)
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
      cp_abuf_quad_build(cp, screen, ab, w, h);

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
      bool bounded = ab->composite && !ab->verify && !ab->timing &&
                     num_triangles <= 2 && cp->fs_batch.ndraws <= 1 &&
                     state->fs && !state->fs->writes_memory &&
                     (size_t)ab->nblocks * rast_num_triangles * 4 <=
                        (size_t)(512u << 10) &&
                     /* Clipping triangulates each input polygon; its pieces
                      * do not multiply covered pixels. Stable-slot capacity is
                      * therefore bounded by input triangles, not by the eight
                      * reserved output IDs per input triangle. */
                     (size_t)num_triangles * n <= ab->capacity &&
                     (size_t)ab->nblocks * rast_num_triangles <=
                        ab->quad_capacity;
      if (bounded) {
         abuf_prod = true;
         abuf_quads = (uint32_t)((size_t)ab->nblocks * rast_num_triangles);
         abuf_covered = 0;   /* the composite covers the framebuffer */
      } else if (ab->composite) {
         uint32_t ctr[CP_ABUF_COUNTERS] = { 0 };
         if (!cp_sync_timed(cp, cp->stream, &cp->plan.wait_seg_ns,
                            &cp->plan.wait_seg_n))
            return;
         /* One copy: sum3, bsum3 and clist_count are contiguous. The last of
          * them is free rather than merely cheap now — the composite's grid is
          * the covered pixels rather than the framebuffer, which on a sample
          * covering 4% of it is 140 blocks instead of 3,600. */
         if (cuMemcpyDtoH(ctr, ab->counters, sizeof(ctr)) != CUDA_SUCCESS) {
            cp_renderer_texture_fatal(cp);
            return;
         }
         const uint32_t *c3 = &ctr[0];
         const uint32_t *q = &ctr[3];
         abuf_covered = ctr[5];
         abuf_quads = q[0];
         /* An empty draw is a successful no-op, not a reason to rerun the
          * peel loop.  A nonempty fragment list with no quads still indicates
          * a broken/incomplete merge and must fall back. */
         abuf_prod = q[1] == 0 && c3[1] == 0 && (q[0] != 0 || c3[0] == 0);
         if (!abuf_prod) {
            static int said_prod = 0;
            if (!said_prod++)
               fprintf(stderr, "abuffer: fragments=%u/%u quads=%u/%u "
                       "quad-overflow=%u fragment-overflow=%u — this draw "
                       "falls back to the peel loop\n", c3[0], ab->capacity,
                       q[0], ab->quad_capacity, q[1], c3[1]);
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
      if (ab->verify && ab->verified < ab->verify_max) {
         cp->abuf_dbg.blk_offsets = ab->blk_offsets;
         cp->abuf_dbg.blk_counts = ab->blk_counts;
         cp->abuf_dbg.quad_prim = ab->quad_prim;
         cp->abuf_dbg.peel_mask = ab->peel_mask;
         cp->abuf_dbg.counters = ab->dbg;

         /* Step 3b: and where to deposit each pass's shaded colours. The
          * write counts start at zero every draw; the colours themselves need
          * no clearing, since a slot nothing wrote is never read. */
         if (ab->colors_ready) {
            cuMemsetD32Async(ab->writes_peel, 0, ab->capacity, cp->stream);
            cuMemsetD32Async(ab->writes_abuf, 0, ab->capacity, cp->stream);
            cp->abuf_dbg.frags = ab->frags;
            cp->abuf_dbg.offsets = ab->offsets;
            cp->abuf_dbg.counts = ab->counts;
            cp->abuf_dbg.colors = ab->colors_peel;
            cp->abuf_dbg.writes = ab->writes_peel;
            cp->abuf_dbg.capacity = ab->capacity;
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
         /* The segment replay reads the clipped buffer without another clip
          * launch, so a deferred clip must run now, classically. */
         if (!cp_flush_pending_clip(cp, screen, &pending_clip)) {
            cp_restore_preclip(&pending_clip, &rast_args, &vs_output_buf,
                               &rast_num_triangles, &cp->fs_batch.prim_shift);
         }
         rast_queues.mode = CP_QUEUE_FILL;
         cp_pass_record_segment(cp, &rast_args, &rast_queues,
                                rast_num_triangles, num_triangles, batch);
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
      if (rast_queues.mode != CP_QUEUE_REUSE &&
          !(fetch_fold && pass == 0))
         cuMemsetD32Async(cp->cur_qset.counts, 0, 3, cp->stream);

      /* Stage 1: 1 thread per triangle (small rasterize in place, others queue) */
      cp_nvtx_push("raster");
      CUresult rast_err = CUDA_SUCCESS;
      bool fused_rast = false;
      if (pending_clip.pending && rast_queues.mode != CP_QUEUE_REUSE) {
         /* Clip and stage 1 fused; only pass 0 can get here, since the
          * deferred clip is consumed by the first chain and a reusing pass
          * never clips. The queue appends are classic stage 1's, so stages
          * 2 and 3 below read exactly what they always read. */
         void *fp[] = { &pending_clip.args, &rast_args, &rast_queues };
         rast_err = cp_launch(cp, screen->kernels.clip_rast_fused,
            (pending_clip.args.num_triangles + 63) / 64, 1, 1, 64, 1, 1,
            0, cp->stream, fp, NULL);
         fused_rast = rast_err == CUDA_SUCCESS;
         if (fused_rast) {
            pending_clip.pending = false;
         } else {
            cp_warn_fused_rast_refused(rast_err);
         }
         if (!fused_rast &&
             !cp_flush_pending_clip(cp, screen, &pending_clip)) {
            cp_restore_preclip(&pending_clip, &rast_args, &vs_output_buf,
                               &rast_num_triangles, &cp->fs_batch.prim_shift);
         }
      }
      void *s1_params[] = { &rast_args, &rast_queues };
      /* Tier 4, plain path. The clear above is skipped for pass 0 under the
       * fetch fold and for a reusing pass, so pass 0 is expected to take and
       * every later peel pass to decline. */
      if (!fused_rast)
         rast_err = cp_launch_after(cp, screen->kernels.rasterize_stage1,
            (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
            0, cp->stream, s1_params, NULL,
            CP_PDL_ANY, CP_PDL_TIER_STAGE1);

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
      CP_LAUNCH_AFTER(fused_rast ? screen->kernels.clip_rast_fused
                                 : screen->kernels.rasterize_stage1,
         CP_PDL_TIER_RASTER, screen->kernels.rasterize_stage2,
         s2_blocks, 1, 1, 256, 1, 1,
         0, cp->stream, s2_params, NULL);

      /* Stage 3: block per tile, grid-strided over the huge-tile queue */
      void *s3_params[] = { &rast_args, &rast_queues };
      CP_LAUNCH_AFTER(screen->kernels.rasterize_stage2,
         CP_PDL_TIER_RASTER, screen->kernels.rasterize_stage3,
         s3_blocks, 1, 1, 64, 1, 1,
         0, cp->stream, s3_params, NULL);

      if (rast_err != CUDA_SUCCESS && cp_debug->debug_draw)
         fprintf(stderr, "  rasterize launch failed: %d\n", rast_err);
      cp_nvtx_pop();   /* raster */
      cp_stage_end(cp, CP_STAGE_RASTERIZE);

      if (cp->pass.appending && cp->pass.opaque) {
         cp_pass_record_segment(cp, &rast_args, &rast_queues,
                                rast_num_triangles, num_triangles, batch);
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
       * for a colour, just the first two. A depth-only pass runs it for a
       * third reason: the writeback at the end of it is what commits depth,
       * and the shader has to run first because it may discard. */
      if (color_data || fs_side_effects || depth_commit)
         cp_shade_fragments(cp, state, &batch->scope, info, visbuf,
                            rast_args.positions, vs_output_buf,
                            rast_args.prim_refs, num_triangles, w, h,
                            color_data,
                            vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y,
                            rast_args.depth_scale, rast_args.depth_translate,
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
            if (!cp_sync_timed(cp, cp->stream, &cp->plan.wait_peel_ns,
                               &cp->plan.wait_peel_n))
               return;
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
      if (!ab->composite) {
         if (!cp_sync_timed(cp, cp->stream, &cp->plan.wait_quads_ns,
                            &cp->plan.wait_quads_n))
            return;
         if (cuMemcpyDtoH(qcounters, ab->bsum3,
                          sizeof(qcounters)) != CUDA_SUCCESS) {
            cp_renderer_texture_fatal(cp);
            return;
         }
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
      if (qcounters[0] && !qcounters[1] && (abuf_prod || !ab->composite)) {
         cp->scratch.used = shade_mark;
         cp->dscratch.used = shade_dmark;
         shaded = cp_abuf_shade(cp, state, &batch->scope, info, ab,
                                rast_args.positions, vs_output_buf,
                                rast_args.prim_refs, w, h,
                                vp_scale_x, vp_scale_y,
                                vp_trans_x, vp_trans_y,
                                rast_args.depth_scale,
                                rast_args.depth_translate, qcounters[0],
                                abuf_covered,
                                ab->colors_ready && cp->abuf_dbg.colors,
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
      if (ab->timing || ab->verify)
         cuMemcpyDtoH(dbg, ab->dbg, sizeof(dbg));

      if (ab->timing) {
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
      memset(&cp->abuf_dbg, 0, sizeof(cp->abuf_dbg));

      if (abuf_log && ab->verified < ab->verify_max) {
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
              "cudavk: %5u tris  assemble %6.3f  vertex %6.3f  raster %6.3f  "
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
cp_batch_order_free(const struct cp_draw_state *state)
{
   if (state->blend.enable || state->fs->uses_discard)
      return false;
   /* The renderer's half of the CUDAVK_UNSAFE_FORCE_OPAQUE relaxation; see
    * cpvk_pipeline_order_free(), which this function has to agree with or the
    * front end and cp_opaque_appendable() disagree about the same batch. */
   if (cp_debug->unsafe_force_opaque)
      return true;
   if (!state->depth.depth_enabled || !state->depth.depth_writemask)
      return false;
   switch (state->depth.depth_func) {
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


/* Snapshot this draw's index range and vertex-stage bindings as the next row
 * of the batch's tables. */


void
cp_batch_begin_packet(struct cp_context *cp,
                      const struct cp_draw_packet *packet,
                      const struct cp_batch_key *key, bool blended)
{
   cp->batch.state = packet->state;
   cp->batch.scope = *packet->scope;
   cp->batch.key = *key;
   cp->batch.info = packet->call;
   cp->batch.drawid_offset = packet->drawid_offset;
   cp->batch.ndraws = 0;
   cp->batch.tris = 0;
   cp->batch.pending = true;
   cp->batch.compact_rows = true;
   cp->batch.blended = blended;
}

/* Snapshot this packet's range and per-draw rows into batch-owned storage. */
void
cp_batch_record_packet(struct cp_context *cp,
                       const struct cp_draw_packet *packet, unsigned tris)
{
   const struct cp_draw_state *s = &packet->state;
   unsigned n = cp->batch.ndraws;
   if (cp_debug->debug_rows) {
      fprintf(stderr, "row %u: fs slots", n);
      for (unsigned q = 0; q < 4 && q < s->num_fs_ubos; q++)
         fprintf(stderr, " [%u]=%p", q,
                 (void *)(uintptr_t)s->fs_ubos[q]);
      fprintf(stderr, "  start=%u count=%u\n",
              packet->range.start, packet->range.count);
   }

   uint64_t *row = cp->batch.vs_ubos + (size_t)n * CP_ARG_UBO_STRIDE;
   memset(row, 0, CP_ARG_UBO_STRIDE * sizeof(*row));
   memcpy(row, s->vs_ubos,
          MIN2(s->num_vs_ubos, CP_MAX_CONST_BUFFERS) * sizeof(*row));

   uint64_t *frow = cp->batch.fs_ubos + (size_t)n * CP_ARG_UBO_STRIDE;
   memset(frow, 0, CP_ARG_UBO_STRIDE * sizeof(*frow));
   memcpy(frow, s->fs_ubos,
          MIN2(s->num_fs_ubos, CP_MAX_CONST_BUFFERS) * sizeof(*frow));

   uint64_t *vrow = cp->batch.vb_bases + (size_t)n * CP_VB_TABLE_STRIDE;
   memset(vrow, 0, CP_VB_TABLE_STRIDE * sizeof(*vrow));
   for (unsigned e = 0; e < s->num_vertex_elements &&
                        e < CP_VB_TABLE_STRIDE; e++) {
      unsigned vb_idx = s->velem[e].vertex_buffer_index;
      if (vb_idx < s->num_vertex_buffers && vb_idx < 16 &&
          s->vb_base[vb_idx])
         vrow[e] = s->vb_base[vb_idx];
   }

   cp->batch.draws[n] = packet->range;
   cp->batch.instance_counts[n] = packet->call.instance_count;
   cp->batch.draw_ids[n] = packet->drawid_offset;
   cp->batch.scissors[n] = packet->scissor;
   cp->batch.tris += tris;
   cp->batch.ndraws++;
   cp->batch_uploads_live = true;
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

static void cp_pass_streams_init(struct cp_context *cp);

static bool
cp_pass_appendable(struct cp_context *cp,
                   const struct cp_draw_batch *batch)
{
   if (batch->state.fs && batch->state.fs->writes_memory)
      return false;
   struct cp_device *screen = cp->dev;
   struct cp_abuf *ab = cp->abuf;

   if (cp_debug->debug_pass) {
      static int said;
      if (said++ < 3)
         fprintf(stderr, "pass?: nopass=%d noabufbatch=%d abuf_en=%d dis=%d "
                 "verify=%d comp=%d timing=%d census=%d segcount=%d frags=%d "
                 "grow=%d shade=%d clist=%d nsegs=%u\n",
                 (int)cp_debug->no_pass_episode, (int)cp_debug->no_abuf_batch,
                 (int)cp_abuf_enabled(ab), (int)ab->disabled, (int)ab->verify,
                 (int)!!ab->composite, (int)ab->timing,
                 (int)cp_census_enabled(),
                 (int)!!screen->kernels.abuf_seg_count, (int)!!ab->frags,
                 (int)!!ab->grow_to, (int)!!ab->shade_slot, (int)!!ab->clist,
                 cp->pass.nsegs);
   }

   if (cp_debug->no_pass_episode || cp_debug->no_abuf_batch)
      return false;
   /* The verification, timing and census modes read per-draw state the
    * episode deliberately does not keep. */
   if (!cp_abuf_enabled(ab) || ab->disabled || ab->verify ||
       !ab->composite || ab->timing || cp_census_enabled())
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
       cp->pass.next_prim >= CP_PRIM_ID_LIMIT)
      return false;
   if (!cp->pass_segs) {
      cp->pass_segs = calloc(CP_PASS_MAX_SEGS, sizeof(*cp->pass_segs));
      if (!cp->pass_segs)
         return false;
   }

   cp_pass_streams_init(cp);
   return true;
}

/*
 * The side streams and their queue sets, once. Failure leaves every
 * seg_streams[] entry null and the episode runs on the main stream.
 *
 * Factored out of cp_pass_appendable() because an opaque episode under
 * CUDAVK_OPAQUE_STREAMS wants exactly the same eight streams, the same eight
 * rasterizer queue sets and the same gate event, and it never goes through the
 * blended admission test that used to be the only place they were built.
 */
static void
cp_pass_streams_init(struct cp_context *cp)
{
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
         if (cp->seg_streams[k])
            cp->seg_stream_serial[k] =
               cp_texture_stream_serial_alloc(cp->dev);
      }
      /* A cache refusal must not disable the side stream or its classic
       * queues. Each stream owns the same 84-KiB evidence-sized capacity. */
      if (ok && !cp_debug->no_setup_cache) {
         for (unsigned k = 0; k < CP_PASS_STREAMS; k++)
            if (cuMemAlloc(&cp->seg_qsets[k].setup_cache,
                           (size_t)CP_SETUP_CACHE_CAPACITY *
                           sizeof(struct cp_setup_cache_entry)) != CUDA_SUCCESS)
               cp->seg_qsets[k].setup_cache = 0;
      }
      if (!ok) {
         fprintf(stderr, "cudavk: pass-episode streams unavailable; "
                 "episodes run on the main stream\n");
         memset(cp->seg_streams, 0, sizeof(cp->seg_streams));
      }
   }
}

/*
 * Whether this opaque episode fans its segments out. On by default since the
 * fan-out was measured and gated; CUDAVK_NO_OPAQUE_STREAMS reverts to issuing
 * every segment back to back on the main stream, and with it set every
 * predicate below reduces to what it was before the fan-out existed.
 */
static bool
cp_opaque_side_streams(const struct cp_context *cp)
{
   if (cp_debug->no_opaque_streams || !cp->seg_streams[0])
      return false;
   /*
    * Two modes are excluded, not because they would be slow but because they
    * are not the thing being made concurrent.
    *
    * CUDAVK_TILED_OPAQUE routes cp_opaque_finish through one shared tile-ref
    * arena with one cursor and one overflow word, and its classic-overflow
    * relaunch loop clears the *shared* nontrivial_count between segments. It
    * also takes its per-segment record earlier in the draw, so it does not
    * even mean the same thing by a segment. It stays serial.
    *
    * CUDAVK_DEBUG_TIME records CUDA events into one flat array and differences
    * consecutive pairs. Events recorded on eight streams and differenced
    * pairwise produce numbers with no meaning, up to and including negative
    * ones. A blended episode refuses itself for the same reason (ab->timing).
    */
   if (cp_debug->tiled_opaque || cp_timing_enabled())
      return false;
   return true;
}

static bool
cp_opaque_appendable(struct cp_context *cp,
                     const struct cp_draw_batch *batch)
{
   const struct cp_draw_state *state = &batch->state;
   const struct cp_fb_desc *fb = &batch->scope.fb;
   if (cp_debug->no_opaque_episode || !cp_batch_order_free(state) ||
       !state->fs || state->fs->writes_memory ||
       MAX2(batch->scope.attachment_samples, 1u) != 1 || fb->nr_cbufs != 1 ||
       !fb->color || fb->color_encoding < 0) {
      if (cp_debug->debug_episode)
         fprintf(stderr, "no-episode: noflag=%d orderfree=%d fs=%d "
                 "writes=%d samples=%u cbufs=%u color=%d enc=%d\n",
                 (int)cp_debug->no_opaque_episode, (int)cp_batch_order_free(state),
                 (int)!!state->fs,
                 state->fs ? (int)state->fs->writes_memory : -1,
                 MAX2(batch->scope.attachment_samples, 1u), fb->nr_cbufs, (int)!!fb->color,
                 fb->color_encoding);
      return false;
   }
   if (cp->pass.opaque &&
       (cp->pass.nsegs >= CP_PASS_MAX_SEGS ||
        cp->pass.next_prim >= CP_PRIM_ID_LIMIT))
      return false;
   if (!cp->pass_segs) {
      cp->pass_segs = calloc(CP_PASS_MAX_SEGS, sizeof(*cp->pass_segs));
      if (!cp->pass_segs)
         return false;
   }
   /* An opaque episode that is going to fan out needs the same eight streams
    * a blended one uses, and blended admission may never have run. */
   if (!cp_debug->no_opaque_streams)
      cp_pass_streams_init(cp);
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
static bool
cp_pass_join(struct cp_context *cp, unsigned nsegs)
{
   /* The events below order other streams against this one, so anything owed
    * on the current stream has to be issued before they are recorded. */
   if (cp_upload_flush(cp) != CUDA_SUCCESS)
      return false;
   if (!cp->seg_streams[0])
      return true;
   unsigned used = MIN2(nsegs, (unsigned)CP_PASS_STREAMS);
   for (unsigned k = 0; k < used; k++) {
      /* An unjoined side stream is not a slow episode, it is an unordered
       * one: everything after this point reads what those streams wrote. */
      if (cuEventRecord(cp->seg_ev[k], cp->seg_streams[k]) != CUDA_SUCCESS ||
          cuStreamWaitEvent(cp->stream, cp->seg_ev[k], 0) != CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return false;
      }
   }
   return true;
}

/*
 * Gate one side stream behind everything the main stream has issued so far.
 *
 * cp_pass_broadcast() below does this for the whole fan-out at a phase
 * boundary, once. A segment needs it one stream at a time and at the moment
 * the segment is issued, because the main stream does not stand still between
 * segments: every draw the application records leaves its uniform rows owed in
 * the upload ring — cpvk_prepare_draw() (cpvk_cmd.c:2582) reserves them, and
 * for this sample that is 11 KiB per segment — and cp_stream_set() sends the
 * owed span on the stream it is *leaving*, which is the main one.
 *
 * A gate recorded once at episode start is therefore behind those bytes rather
 * than in front of them: the segment's kernels are free to run before the copy
 * that fills their argument rows has landed, and they read whatever the arena
 * held before it. That is the multithreading sample losing whole models,
 * intermittently, and it is not a race between segments — serialising the
 * segments with cuStreamSynchronize() does not fix it, because the unordered
 * pair is a main-stream copy against a side-stream kernel.
 */
static bool
cp_pass_gate_stream(struct cp_context *cp, unsigned k)
{
   /* The owed span first, then the event: recording it behind the copy is
    * what makes the wait below cover the copy. */
   if (cp_upload_flush(cp) != CUDA_SUCCESS)
      return false;
   if (cuEventRecord(cp->pass_gate, cp->stream) != CUDA_SUCCESS ||
       cuStreamWaitEvent(cp->seg_streams[k], cp->pass_gate, 0) !=
          CUDA_SUCCESS) {
      cp_renderer_texture_fatal(cp);
      return false;
   }
   return true;
}

/* The reverse: gate every side stream behind the main stream's tail. */
static bool
cp_pass_broadcast(struct cp_context *cp, unsigned nsegs)
{
   if (cp_upload_flush(cp) != CUDA_SUCCESS)
      return false;
   if (!cp->seg_streams[0])
      return true;
   if (cuEventRecord(cp->pass_gate, cp->stream) != CUDA_SUCCESS) {
      cp_renderer_texture_fatal(cp);
      return false;
   }
   unsigned used = MIN2(nsegs, (unsigned)CP_PASS_STREAMS);
   for (unsigned k = 0; k < used; k++)
      if (cuStreamWaitEvent(cp->seg_streams[k], cp->pass_gate, 0) !=
          CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return false;
      }
   return true;
}


static bool
cp_pass_can_retry(struct cp_context *cp)
{
   if (cp->device_fatal)
      return false;
   if (cp->hardware_texture.fs_attempts == cp->pass.hw_attempt_start)
      return true;
   fprintf(stderr, "cudavk: refusing episode retry after FS attempt; "
           "latching device loss\n");
   cp->device_fatal = true;
   if (cp->dev->texture_cache_fatal)
      cp->dev->texture_cache_fatal(cp->dev->texture_cache_private);
   return false;
}

/* The episode could not deliver; render every segment the classic way, in
 * submission order, from its snapshot. Rasterization is idempotent — the
 * abandoned lists were never read by anything that draws. */
static void
cp_pass_fallback(struct cp_context *cp, struct cp_pass_seg *segs,
                 unsigned nsegs)
{
   if (!cp_pass_can_retry(cp))
      return;
   /* The abandoned episode's kernels may still be in flight on the side
    * streams, writing the shared lists the re-execution is about to clear. */
   if (!cp_pass_join(cp, nsegs))
      return;

   struct cp_fs_batch saved_fs_batch = cp->fs_batch;
   for (unsigned s = 0; s < nsegs; s++) {
      struct cp_pass_seg *sg = &segs[s];
      cp_draw_execute_batch(cp, &sg->batch);
   }
   cp->fs_batch = saved_fs_batch;
}

static bool
cp_opaque_tile_visibility(struct cp_context *cp, struct cp_pass_seg *segs,
                          unsigned nsegs, unsigned w, unsigned h)
{
   struct cp_device *screen = cp->dev;
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
                  CP_MAX_OPAQUE_TILE_REFS, overflow, cursors);
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
      /* All three counters share this allocation: FILL starts a fresh
       * nontrivial queue, tile queue and setup-cache generation together. */
      cuMemsetD32Async(queues.nontrivial_count, 0, 3, cp->stream);
      void *params[] = { &aa, &queues };
      /* Tier 4, tiled-opaque fallback. Expected to decline: the clear is
       * unconditional and inside the segment loop. */
      CP_LAUNCH_AFTER(CP_PDL_ANY, CP_PDL_TIER_STAGE1,
                screen->kernels.rasterize_stage1,
                DIV_ROUND_UP(segs[s].rast_num_triangles, 256), 1, 1,
                256, 1, 1, 0, cp->stream, params, NULL);
      CP_LAUNCH_AFTER(screen->kernels.rasterize_stage1, CP_PDL_TIER_RASTER,
                screen->kernels.rasterize_stage2,
                CLAMP(DIV_ROUND_UP(segs[s].rast_num_triangles, 8), 1u, 512u),
                1, 1, 256, 1, 1, 0, cp->stream, params, NULL);
      CP_LAUNCH_AFTER(screen->kernels.rasterize_stage2, CP_PDL_TIER_RASTER,
                screen->kernels.rasterize_stage3, 2048, 1, 1,
                64, 1, 1, 0, cp->stream, params, NULL);
   }

   if (cp_debug->tiled_opaque_census) {
      uint32_t host_overflow = 0;
      cuMemcpyDtoHAsync(&host_overflow, overflow, sizeof(host_overflow),
                        cp->stream);
      cuStreamSynchronize(cp->stream);
      fprintf(stderr, "cudavk: opaque tiles %ux%u segments=%u overflow=%u\n",
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
   for (unsigned s = 0; s < nsegs; s++) {
      if (segs[s].batch.state.fs && segs[s].batch.state.fs->writes_memory) {
         cp_pass_fallback(cp, segs, nsegs);
         return;
      }
   }
   unsigned w = cp->pass.w, h = cp->pass.h;
   cp->pass.nsegs = 0;
   cp->pass.next_prim = 0;
   cp->pass.opaque = false;

   /* Counted in both arms: how much concurrency this episode had to offer. */
   cp->plan.opaque_episodes++;
   cp->plan.opaque_segs += nsegs;
   cp->plan.opaque_concurrent += nsegs - 1;
   if (nsegs > 1)
      cp->plan.opaque_multiseg++;
   if (nsegs > cp->plan.opaque_max_segs)
      cp->plan.opaque_max_segs = nsegs;
   /* Episodes that really put a segment on a side stream. A one-segment
    * episode under the flag now issues exactly what the flag-off path issues,
    * so counting it as fanned out would report a fan-out that did not
    * happen — and this counter is what says whether a sample's regression or
    * win came from the streams at all. */
   if (cp_opaque_side_streams(cp) && nsegs > 1)
      cp->plan.opaque_fanned++;

   /* The segments rasterized visibility on the side streams; everything below
    * — the tile pass, the census and the shading — reads across all of them.
    * An unjoined side stream is not a slow episode, it is an unordered one.
    *
    * nsegs - 1 side streams, because segment 0 ran on the main stream and is
    * already ordered ahead of this point. Joining a stream that carried no
    * work is not merely wasted: it is the same event record and cross-stream
    * wait a real join costs, which for a one-segment episode is the whole
    * price of the fan-out and none of its benefit. */
   if (cp_opaque_side_streams(cp) && !cp_pass_join(cp, nsegs - 1))
      return;

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
            if (first->batch.state.vs == segs[s].batch.state.vs &&
                first->batch.state.fs == segs[s].batch.state.fs &&
                first->batch.state.num_fs_ubos == segs[s].batch.state.num_fs_ubos &&
                first->batch.info.mode == segs[s].batch.info.mode &&
                first->batch.ndraws == segs[s].batch.ndraws &&
                !memcmp(first->batch.fs_ubos, segs[s].batch.fs_ubos,
                        (size_t)first->batch.ndraws * CP_ARG_UBO_STRIDE *
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
            .prim_refs = seg->rast.prim_refs,
            .draw_slices = seg->slices_dev,
            .num_draw_slices = seg->batch.ndraws,
            .prim_base = seg->prim_base,
            .prim_end = seg->prim_base + seg->prim_slots,
            .row_base = group_rows,
            .prim_shift = seg->prim_shift,
         };
         group_rows += seg->batch.ndraws;
      }
      group_range_count[g] = nranges - group_range_base[g];
   }
   CUdeviceptr ranges_dev = cp_upload(cp, ranges,
                                      (size_t)nranges * sizeof(ranges[0]));
   if (!ranges_dev) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   const struct cp_fb_desc *fb = &segs[0].batch.scope.fb;
   void *color_data = fb->color;
   if (!color_data) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   struct cp_fs_batch saved_fs_batch = cp->fs_batch;
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
                member->batch.fs_ubos,
                (size_t)member->batch.ndraws * CP_ARG_UBO_STRIDE *
                   sizeof(uint64_t));
         group_rows += member->batch.ndraws;
      }
      cp->scratch.used = shade_mark;
      cp->dscratch.used = shade_dmark;
      cp->fs_batch.ubos = cp->pass_group_ubos;
      cp->fs_batch.ndraws = group_rows;
      cp->fs_batch.slices = seg->slices_dev;
      cp->fs_batch.prim_shift = seg->prim_shift;
      cp_shade_fragments(cp, &seg->batch.state, &seg->batch.scope,
                         &seg->batch.info, cp->visbuf, seg->rast.positions,
                         seg->rast.positions, seg->rast.prim_refs,
                         seg->num_triangles, w, h,
                         color_data, seg->rast.vp_scale_x,
                         seg->rast.vp_scale_y, seg->rast.vp_trans_x,
                         seg->rast.vp_trans_y, seg->rast.depth_scale,
                         seg->rast.depth_translate, 0, 0, 0,
                         ranges_dev + (size_t)group_range_base[g] *
                            sizeof(ranges[0]),
                         group_range_count[g]);
   }
   cp->fs_batch = saved_fs_batch;
}

/*
 * Tile-bin shader census (CUDAVK_TILE_CENSUS=<tile edge>).
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
   struct cp_device *screen = cp->dev;
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
      unsigned i = cp_tile_census_shader_index(cp, segs[s].batch.state.fs);
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
      cp->tile_census_marked_draws += segs[s].batch.ndraws;

   /* How large the bin itself would be, which the shaded counts cannot say. */
   if (cp->dev->kernels.tile_census_refs) {
      for (unsigned s = 0; s < nsegs; s++) {
         unsigned n = segs[s].rast_num_triangles;
         if (!n)
            continue;
         void *p[] = { &segs[s].rast, ca };
         CP_LAUNCH(cp->dev->kernels.tile_census_refs,
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
   struct cp_device *screen = cp->dev;
   struct cp_abuf *ab = cp->abuf;
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
   struct cp_device *screen = cp->dev;
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
   struct cp_device *screen = cp->dev;

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
   struct cp_abuf *ab = cp->abuf;
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
         if (first->batch.state.vs == segs[s].batch.state.vs &&
             first->batch.state.fs == segs[s].batch.state.fs &&
             first->batch.state.num_fs_ubos == segs[s].batch.state.num_fs_ubos &&
             first->batch.info.mode == segs[s].batch.info.mode) {
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
      if (compact && cp_debug->no_counter_block)
         cuMemsetD32Async(ab->seg_counts, 0, CP_PASS_MAX_SEGS, cp->stream);
      void *bucket_params[] = { &bucket };
      CP_LAUNCH(cp->dev->kernels.abuf_seg_count,
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
         /*
          * The cursor clear moves in front of the prefix rather than behind
          * it, so that the prefix and the scatter are adjacent kernels on the
          * stream and the scatter can be a PDL secondary of the prefix. The
          * prefix reads and writes seg_counts, seg_group, seg_base,
          * group_base and group_counts, and never seg_cursor, so this is the
          * same zeros to the same words at a different point in an order that
          * already had to hold. Without the move the clear sits between them
          * and the epoch check refuses the link -- correctly.
          *
          * It is gated on PDL being on at all, and that is not fussiness.
          * CUDAVK_NO_PDL is the revert, and a revert that still reorders one
          * stream operation is not a revert: a reader bisecting a regression
          * has to get the pre-PDL driver back exactly, or level 0 is a fourth
          * behaviour rather than the original one. Nothing at level 0 wants
          * the move -- it exists only to open the level-3 link.
          */
         const bool hoist_cursor_clear =
            cp->dev->kernels.pdl >= CP_PDL_TIER_SCAN;
         if (hoist_cursor_clear)
            cuMemsetD32Async(seg_cursor, 0, nsegs, cp->stream);

         void *prefix_params[] = { &prefix };
         CP_LAUNCH(cp->dev->kernels.abuf_seg_prefix,
                   1, 1, 1, 1, 1, 1, 0, cp->stream, prefix_params, NULL);

         if (!hoist_cursor_clear)
            cuMemsetD32Async(seg_cursor, 0, nsegs, cp->stream);
         bucket.seg_cursor = seg_cursor;
         bucket.seg_base = seg_base_dev;
         bucket.grouped = grouped;
         bucket.quad_dense = quad_dense;
         bucket.seg_group = seg_group_dev;
         bucket.group_base = group_base_dev;
         void *scatter_params[] = { &bucket };
         /* The prefix is one thread walking nsegs x ngroups; the scatter's
          * two independent loads and its CTA setup are worth having in
          * flight while it does. */
         CP_LAUNCH_AFTER(cp->dev->kernels.abuf_seg_prefix, CP_PDL_TIER_FS,
                   cp->dev->kernels.abuf_seg_scatter,
                   ((unsigned)quad_bound + 255) / 256,
                   1, 1, 256, 1, 1, 0, cp->stream, scatter_params, NULL);
      }
   }

   struct cp_seg_range ranges[CP_PASS_MAX_SEGS];
   struct cp_seg_desc descs[CP_PASS_MAX_SEGS] = {0};
   struct cp_abuf_seg_shade group_shades[CP_PASS_MAX_SEGS] = {0};
   struct cp_fs_batch saved_fs_batch = cp->fs_batch;
   bool shaded = true;
   for (unsigned group = 0; group < ngroups && shaded; group++) {
      struct cp_pass_seg *first = &segs[group_first[group]];
      bool need_rows = first->batch.state.fs->reads_const_bufs;
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
            .prim_refs = seg->rast.prim_refs,
            .draw_slices = seg->slices_dev,
            .num_draw_slices = seg->batch.ndraws,
            .prim_base = seg->prim_base,
            .prim_end = seg->prim_base + seg->prim_slots,
            .row_base = rows,
            .prim_shift = seg->prim_shift,
         };
         if (need_rows)
            memcpy(cp->pass_group_ubos + (size_t)rows * CP_ARG_UBO_STRIDE,
                   seg->batch.fs_ubos,
                   (size_t)seg->batch.ndraws * CP_ARG_UBO_STRIDE * sizeof(uint64_t));
         rows += seg->batch.ndraws;
      }
      CUdeviceptr ranges_dev =
         cp_upload(cp, ranges, (size_t)nranges * sizeof(ranges[0]));
      if (!ranges_dev) {
         shaded = false;
         break;
      }

      cp->fs_batch.ubos = need_rows ? cp->pass_group_ubos : first->batch.fs_ubos;
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
         cp, &first->batch.state, &first->batch.scope,
         &first->batch.info, ab, first->rast.positions, first->rast.positions,
         first->rast.prim_refs, w, h, first->rast.vp_scale_x,
         first->rast.vp_scale_y,
         first->rast.vp_trans_x, first->rast.vp_trans_y,
         first->rast.depth_scale, first->rast.depth_translate,
         (uint32_t)quad_bound, 0, false, NULL, false, &ti, &ts, &tc, &shade);
      if (!shaded)
         break;
      if (group == 0 &&
          cp_debug->texture_cache_fail_after_bounded_group0) {
         if (cp_debug->texture_cache_stats)
            fprintf(stderr, "cudavk: injected bounded fatal after group0, "
                    "fs_attempt_delta=%" PRIu64 "\n",
                    cp->hardware_texture.fs_attempts - cp->pass.hw_attempt_start);
         cp_renderer_texture_fatal(cp);
         shaded = false;
         break;
      }
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
   cp->fs_batch = saved_fs_batch;
   if (!shaded)
      return false;

   const struct cp_fb_desc *fb = &segs[0].batch.scope.fb;
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
      .max_layers = ab->max_layers,
      .blend = cp_blend_desc_for(&segs[0].batch.state),
      .quad_seg = ngroups > 1 ? quad_seg : 0,
      .quad_dense = ngroups > 1 ? quad_dense : 0,
      .seg_desc = ngroups > 1 ? descs_dev : 0,
   };
   void *params[] = { &ca };
   CUresult err = cp_launch(cp, cp->dev->kernels.abuf_composite,
                                 ((size_t)w * h + 255) / 256, 1, 1,
                                 256, 1, 1, 0, cp->stream, params, NULL);
   if (err != CUDA_SUCCESS)
      fprintf(stderr, "abuffer: bounded grouped composite launch failed "
              "(%d)\n", err);
   else
      cp_texture_attachment_written(cp, fb);
   return err == CUDA_SUCCESS;
}

void
cp_pass_finish(struct cp_context *cp)
{
   cp->plan.pass_finish_calls++;
   struct cp_device *screen = cp->dev;
   struct cp_abuf *ab = cp->abuf;
   unsigned nsegs = cp->pass.nsegs;
   if (nsegs)
      cp->plan.pass_finishes++;

   if (nsegs && cp_debug->debug_episode)
      fprintf(stderr, "episode: nsegs=%u opaque=%d\n", nsegs,
              (int)cp->pass.opaque);

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

   CP_NVTX_SCOPEF("episode %u segs", nsegs);

   /* The segments' count phases ran on the side streams; the scan reads
    * across all of them. */
   if (!cp_pass_join(cp, nsegs))
      return;

   /* --- scan the accumulated counts, clamp the runs to the array ---
    *
    * The fill cursor's clear rides in the same pass: nothing between here and
    * the fill can replace ab->cursor, and the clear was a 3.7 MB device
    * operation of its own immediately behind a kernel that already writes one
    * word per pixel. */
   CUstream pass_main = cp->stream;
   cp_abuf_scan(cp, screen, ab, (unsigned)n, ab->cursor);

   /* --- fill --- */
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
      if (!cp_pass_broadcast(cp, nsegs))
         return;
      for (unsigned s = 0; s < nsegs; s++) {
         struct cp_pass_seg *sg = &segs[s];
         if (cp->seg_streams[0])
            cp_stream_set(cp, cp_pass_seg_stream(cp, s));
         struct cp_rasterize_args aa = sg->rast;
         aa.abuf_frags = ab->frags;
         aa.abuf_capacity = ab->capacity;
         aa.abuf_mode = CP_ABUF_FILL;
         struct cp_rast_queues q = sg->queues;
         q.mode = CP_QUEUE_FILL;
         /* The segment's own queue set, saved with its arguments. */
         cuMemsetD32Async(q.nontrivial_count, 0, 3, cp->stream);
         void *ap[] = { &aa, &q };
         /* Tier 4, pass segments. Expected to decline for the same reason as
          * the tiled fallback: one clear per segment, inside the loop. */
         CP_LAUNCH_AFTER(CP_PDL_ANY, CP_PDL_TIER_STAGE1,
                        screen->kernels.rasterize_stage1_abuf,
                        (sg->rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
                        0, cp->stream, ap, NULL);
         /* One segment's three stages are a chain on that segment's own
          * stream; the fan-out overlaps segments, never these. */
         CP_LAUNCH_AFTER(screen->kernels.rasterize_stage1_abuf,
                        CP_PDL_TIER_RASTER,
                        screen->kernels.rasterize_stage2_abuf,
                        CLAMP((sg->rast_num_triangles + 7) / 8, 1u, 512u), 1, 1,
                        256, 1, 1, 0, cp->stream, ap, NULL);
         CP_LAUNCH_AFTER(screen->kernels.rasterize_stage2_abuf,
                        CP_PDL_TIER_RASTER,
                        screen->kernels.rasterize_stage3_abuf,
                        CLAMP(sg->rast_num_triangles * 8, 512u, 2048u), 1, 1,
                        64, 1, 1, 0, cp->stream, ap, NULL);
      }
      cp_stream_set(cp, pass_main);
      if (!cp_pass_join(cp, nsegs))
         return;
   }

   /* --- sort, both worklists --- */
   {
      unsigned nn = (unsigned)n;
      unsigned max_short = cp_debug->abuf_short_sort_max;
      unsigned min_long = cp_debug->no_abuf_short_sort ? 2 : max_short + 1;
      if (!cp_debug->no_abuf_short_sort) {
         if (cp_debug->no_counter_block) {
            cuMemsetD32Async(ab->clist_count, 0, 1, cp->stream);
            cuMemsetD32Async(ab->list_count, 0, 1, cp->stream);
         }
         void *ssp[] = { &ab->frags, &ab->offsets, &ab->counts, &nn,
                         &max_short, &ab->clist, &ab->clist_count,
                         &min_long, &ab->list, &ab->list_count };
         CP_LAUNCH(screen->kernels.abuf_sort_short,
                        MIN2((nn + 255) / 256, 4096u), 1, 1, 256, 1, 1,
                        0, cp->stream, ssp, NULL);
      } else {
         if (cp_debug->no_counter_block)
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
         if (cp_debug->no_counter_block)
            cuMemsetD32Async(ab->clist_count, 0, 1, cp->stream);
         void *cw[] = { &ab->counts, &nn, &min1, &ab->clist,
                        &ab->clist_count };
         CP_LAUNCH(screen->kernels.abuf_worklist, (nn + 255) / 256, 1, 1,
                        256, 1, 1, 0, cp->stream, cw, NULL);
      }
   }

   /* --- the quad stream --- */
   cp_abuf_quad_build(cp, screen, ab, w, h);

cp_tile_census_quads(cp, segs, nsegs, w, h);

   if (cp_pass_finish_bounded_groups(cp, segs, nsegs, w, h))
      return;
   if (!cp_pass_can_retry(cp))
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
   if (cp_debug->no_counter_block)
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
   if (!cp_sync_timed(cp, cp->stream, &cp->plan.wait_episode_ns,
                      &cp->plan.wait_episode_n))
      return;
   if (cuMemcpyDtoH(ctr, ab->counters,
                    sizeof(uint32_t) * (CP_ABUF_COUNTERS + nsegs)) !=
       CUDA_SUCCESS) {
      cp_renderer_texture_fatal(cp);
      return;
   }

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
            if (f->batch.state.vs == segs[s].batch.state.vs &&
                f->batch.state.fs == segs[s].batch.state.fs &&
                f->batch.state.num_fs_ubos == segs[s].batch.state.num_fs_ubos &&
                f->batch.info.mode == segs[s].batch.info.mode) {
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
   if (!cp_pass_broadcast(cp, nsegs))
      return;
   struct cp_fs_batch saved_fs_batch = cp->fs_batch;
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
         cp_stream_set(cp, cp_pass_seg_stream(cp, g));
      struct cp_abuf_seg_shade ss = {
         .quad_list = grouped,
         .quad_list_base = group_base[g],
         .prim_base = sg->prim_base,
      };
      if (members == 1) {
         cp->fs_batch.ubos = sg->batch.fs_ubos;
         cp->fs_batch.ndraws = sg->batch.ndraws;
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
         bool need_rows = sg->batch.state.fs->reads_const_bufs;
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
               .prim_refs = m->rast.prim_refs,
               .draw_slices = m->slices_dev,
               .num_draw_slices = m->batch.ndraws,
               .prim_base = m->prim_base,
               .prim_end = m->prim_base + m->prim_slots,
               .row_base = rows,
               .prim_shift = m->prim_shift,
            };
            if (need_rows)
               memcpy(cp->pass_group_ubos + (size_t)rows * CP_ARG_UBO_STRIDE,
                      m->batch.fs_ubos,
                      (size_t)m->batch.ndraws * CP_ARG_UBO_STRIDE *
                      sizeof(uint64_t));
            rows += m->batch.ndraws;
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
         cp->fs_batch.ubos = need_rows ? cp->pass_group_ubos : sg->batch.fs_ubos;
         cp->fs_batch.ndraws = rows;
         cp->fs_batch.slices = sg->slices_dev;
         cp->fs_batch.prim_shift = sg->prim_shift;
      }
      float ti, ts, tc;
      if (!cp_abuf_shade(cp, &sg->batch.state, &sg->batch.scope,
                         &sg->batch.info, ab, sg->rast.positions,
                         sg->rast.positions, sg->rast.prim_refs, w, h,
                         sg->rast.vp_scale_x, sg->rast.vp_scale_y,
                         sg->rast.vp_trans_x, sg->rast.vp_trans_y,
                         sg->rast.depth_scale, sg->rast.depth_translate,
                         gq, 0, false, NULL, false, &ti, &ts, &tc, &ss)) {
         failed = true;
         break;
      }
      if (g == 0 && cp_debug->texture_cache_fail_after_main_group0) {
         if (cp_debug->texture_cache_stats)
            fprintf(stderr, "cudavk: injected main fatal after group0, "
                    "fs_attempt_delta=%" PRIu64 "\n",
                    cp->hardware_texture.fs_attempts - cp->pass.hw_attempt_start);
         cp_renderer_texture_fatal(cp);
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
   cp_stream_set(cp, pass_main);
   cp->fs_batch = saved_fs_batch;
   /* A failed join leaves the shading streams unordered against the composite;
    * neither compositing nor re-rendering is safe after it. */
   if (!cp_pass_join(cp, nsegs))
      return;
   if (failed) {
      cp_pass_fallback(cp, segs, nsegs);
      return;
   }

   /* --- one composite for the whole episode --- */
   const struct cp_fb_desc *fb = &segs[0].batch.scope.fb;
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
      .max_layers = ab->max_layers,
      .blend = cp_blend_desc_for(&segs[0].batch.state),
      .quad_seg = quad_seg,
      .quad_dense = quad_dense,
      .seg_desc = descs_dev,
   };
   void *p[] = { &ca };
   unsigned nwork = covered ? covered : (unsigned)n;
   cp_nvtx_push("composite");
   CUresult ce = cp_launch(cp, screen->kernels.abuf_composite,
                                (nwork + 255) / 256, 1, 1, 256, 1, 1,
                                0, cp->stream, p, NULL);
   cp_nvtx_pop();
   if (ce != CUDA_SUCCESS) {
      fprintf(stderr, "abuffer: episode composite launch failed (%d); "
              "re-rendering segment by segment\n", ce);
      cp_pass_fallback(cp, segs, nsegs);
   } else {
      cp_texture_attachment_written(cp, fb);
   }
}

/* Record the segment cp_draw_execute has just counted; called from inside
 * it, with the count launches already on the stream. */
void
cp_pass_record_segment(struct cp_context *cp,
                       const struct cp_rasterize_args *aa,
                       const struct cp_rast_queues *queues,
                       unsigned rast_num_triangles, unsigned num_triangles,
                       const struct cp_draw_batch *batch)
{
   if (!cp->pass.nsegs)
      cp->pass.hw_attempt_start = cp->hardware_texture.fs_attempts;
   struct cp_pass_seg *sg = &cp->pass_segs[cp->pass.nsegs];

   sg->rast = *aa;
   sg->queues = *queues;
   sg->rast_num_triangles = rast_num_triangles;
   sg->num_triangles = num_triangles;
   sg->prim_base = cp->pass.next_prim;
   sg->prim_slots = rast_num_triangles;
   sg->prim_shift = cp->fs_batch.prim_shift;
   sg->batch = *batch;
   sg->slices_dev = cp->fs_batch.slices;

   if (cp->pass.nsegs == 0) {
      cp->pass.w = aa->width;
      cp->pass.h = aa->height;
   }
   cp->pass.nsegs++;
   cp->pass.next_prim += rast_num_triangles;
}


/*
 * The triangle count cp_draw_execute will size its grids for, computed here so
 * the A-buffer's size test can be applied before the vertex work rather than
 * after it.
 *
 * This is the pre-clip count -- the same worst case the rasterizer grid uses --
 * so the two tests agree by construction.
 */
static unsigned
cp_batch_total_triangles(const struct cp_draw_call *info,
                         const struct cp_draw_range *draws,
                         unsigned batch_draws,
                         const uint32_t *instance_counts)
{
   unsigned instance_count = MAX2(info->instance_count, 1u);
   if (batch_draws <= 1)
      return cp_triangles_for_draw(info->mode, draws[0].count) *
             instance_count;

   unsigned total = 0;
   for (unsigned d = 0; d < batch_draws; d++) {
      unsigned draw_instances = instance_counts
         ? MAX2(instance_counts[d], 1u) : instance_count;
      total += cp_triangles_for_draw(info->mode, draws[d].count) *
               draw_instances;
   }
   return total;
}

/* Append the pending batch to the episode as a segment; on a refusal deep
 * enough that only cp_draw_execute could see it, finish the episode and
 * render the batch the classic way — its draws came after every segment's. */
static void
cp_pass_append(struct cp_context *cp, unsigned ndraws)
{
   struct cp_abuf *ab = cp->abuf;
   const struct cp_fb_desc *fb = &cp->batch.scope.fb;

   if (cp->pass.nsegs && cp->pass.opaque)
      cp_pass_finish(cp);

   /*
    * A batch too small for the A-buffer, decided before the vertex work.
    *
    * cp_draw_execute reaches the same conclusion, but only after fetching and
    * clipping -- and an append that backs out there is re-executed whole, so
    * the vertex work happens twice. The trace showed it: cp_vertex_fetch ran
    * 32.6 times a frame against 24.3 rasterizations, where the Gallium
    * driver's two are equal.
    */
   if (cp_debug->abuf_min_tris &&
       cp_batch_total_triangles(&cp->batch.info, cp->batch.draws, ndraws,
                                cp->batch.instance_counts)
          < cp_debug->abuf_min_tris) {
      cp_pass_finish(cp);
      cp_draw_execute_batch(cp, &cp->batch);
      return;
   }

   /*
    * Episode start: size the per-pixel arrays for this framebuffer and clear
    * the shared lists once, on the main stream, with the gate event recorded
    * behind them — every segment stream waits on it before its first work.
    */
   if (cp->pass.nsegs == 0) {
      unsigned w = fb->width, h = fb->height;
      if (!w || !h || !cp_abuf_setup(cp, ab, w, h)) {
         /* A refusal falls back; a latched device loss must not. */
         if (cp->device_fatal)
            return;
         cp_pass_finish(cp);
         cp_draw_execute_batch(cp, &cp->batch);
         return;
      }
      /* The episode's shared lists. Every segment below accumulates into
       * them, so a clear that never ran would be read as real counts. */
      CUresult err = cuMemsetD32Async(ab->counts, 0, (size_t)w * h,
                                      cp->stream);
      if (err == CUDA_SUCCESS) {
         if (!cp_debug->no_counter_block) {
            /* Every scalar counter this episode fills, in one clear. */
            err = cuMemsetD32Async(ab->counters, 0, CP_ABUF_BLOCK_WORDS,
                                   cp->stream);
         } else {
            err = cuMemsetD32Async(ab->sum3, 0, 3, cp->stream);
            if (err == CUDA_SUCCESS && ab->recs)
               err = cuMemsetD32Async(ab->rec_cursor, 0, 1, cp->stream);
         }
      }
      if (err == CUDA_SUCCESS && cp->seg_streams[0]) {
         /* The gate every segment stream waits on. */
         err = cp_upload_flush(cp);
         if (err == CUDA_SUCCESS)
            err = cuEventRecord(cp->pass_gate, cp->stream);
      }
      if (err != CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return;
      }
   }

   CUstream saved_stream = cp->stream;
   struct cp_queue_set saved_qset = cp->cur_qset;
   if (cp->seg_streams[0]) {
      unsigned k = cp->pass.nsegs % CP_PASS_STREAMS;
      /* The gate is what orders this segment behind the clears above. */
      if (cuStreamWaitEvent(cp->seg_streams[k], cp->pass_gate, 0) !=
          CUDA_SUCCESS) {
         cp_renderer_texture_fatal(cp);
         return;
      }
      cp_stream_set(cp, cp->seg_streams[k]);
      cp->cur_qset = cp->seg_qsets[k];
   }

   cp->pass.appending = true;
   cp->pass.append_failed = false;
   cp_draw_execute_batch(cp, &cp->batch);
   cp->pass.appending = false;
   cp_stream_set(cp, saved_stream);
   cp->cur_qset = saved_qset;

   if (cp->pass.append_failed) {
      if (cp->device_fatal)
         return;
      /* The failed append may have run vertex work and uploads on its side
       * stream before backing out; when it was the would-be first segment,
       * cp_pass_finish below returns without joining anything, and the flush
       * fence — recorded on the main stream only — would not cover it. Join
       * every side stream so it always does. */
      if (!cp_pass_join(cp, CP_PASS_STREAMS))
         return;
      cp_pass_finish(cp);
      /* The episode this rollback closed may have latched device loss; the
       * batch must not be re-executed onto a poisoned context. */
      if (cp->device_fatal)
         return;
      cp_draw_execute_batch(cp, &cp->batch);
   }
}

static void
cp_opaque_append(struct cp_context *cp, unsigned ndraws)
{
   if (cp_debug->debug_episode)
      fprintf(stderr, "append: nsegs=%u opaque=%d ndraws=%u\n",
              cp->pass.nsegs, (int)cp->pass.opaque, ndraws);

   if (cp->pass.nsegs && !cp->pass.opaque)
      cp_pass_finish(cp);
   if (cp->pass.nsegs >= CP_PASS_MAX_SEGS) {
      cp_pass_finish(cp);
   }

   bool side = cp_opaque_side_streams(cp);

   if (!cp->pass.nsegs) {
      const struct cp_fb_desc *fb = &cp->batch.scope.fb;
      cp->pass.opaque = true;
      cp->pass.w = fb->width;
      cp->pass.h = fb->height;
      cp->pass.next_prim = 0;
      cuMemsetD32Async(cp->visbuf, 0xFFFFFFFF,
                       (size_t)fb->width * fb->height * 2,
                       cp->stream);
      if (side) {
         /*
          * Two device operations the segments read and neither of them owns.
          *
          * The visibility clear above is one. The depth clear is the other,
          * and it is the one a fan-out breaks: cp_draw_execute_batch() issues
          * it lazily at the first segment that finds cp->depthbuf_cleared
          * false, which with side streams means on that segment's stream,
          * while every other segment reads cp->depthbuf for its depth test on
          * a stream that has not waited for it. Issuing it here puts it on
          * the main stream, and the latch makes the lazy call a no-op for
          * every segment.
          *
          * Both clears land on the main stream ahead of the gate each segment
          * records for itself below, so no segment can run before them.
          */
         if (!cp->depthbuf_cleared)
            cp_clear_depthbuf(cp, 1.0f);
      }
   }

   unsigned before = cp->pass.nsegs;
   CUstream saved_stream = cp->stream;
   struct cp_queue_set saved_qset = cp->cur_qset;
   /*
    * The first segment stays on the main stream, and only segment s > 0 goes
    * to a side stream — stream s-1, so eight streams still carry the eight
    * segments that can overlap.
    *
    * Segment 0 has nothing to overlap: it is the only work the episode has in
    * flight, and everything the episode did before it is on the main stream
    * ahead of it. Moving it to a side stream buys no concurrency and costs two
    * cross-stream synchronisations, the gate in and the join out, whose
    * bubbles the segment cannot hide behind anything. Measured at about 32 us
    * per episode: it is what made the samples whose opaque episodes are all
    * exactly one segment long — pbribl 0.60 -> 1.35, pushconstants 0.13 ->
    * 1.13 ms/frame — pay for a fan-out they never used, and it cost the
    * 600-frame sweep 0.8 ms of hot sum. Those episodes now issue exactly what
    * the flag-off path issues.
    *
    * This is not a special case for one-segment episodes; it is the same
    * saving on every episode, which pays for one fewer stream hand-off than it
    * did. The isolation hazard 2 requires is unchanged: segment 0 uses the
    * context-wide queue set, which no side stream ever touches, and segments
    * 1..n use their own.
    */
   bool seg_side = side && cp->pass.nsegs > 0;
   if (seg_side) {
      unsigned k = (cp->pass.nsegs - 1) % CP_PASS_STREAMS;
      /*
       * The gate is what orders this segment behind the episode's clears and
       * behind everything else the main stream has issued since the previous
       * segment — in particular the upload span this segment's own uniform
       * rows are sitting in, which cp_stream_set() below is about to send on
       * the main stream. Recorded here rather than once at episode start, for
       * the reason cp_pass_gate_stream() gives.
       */
      if (!cp_pass_gate_stream(cp, k))
         return;
      cp_stream_set(cp, cp->seg_streams[k]);
      cp->cur_qset = cp->seg_qsets[k];
   }

   cp->pass.appending = true;
   cp->pass.append_failed = false;
   cp_draw_execute_batch(cp, &cp->batch);
   cp->pass.appending = false;
   if (seg_side) {
      cp_stream_set(cp, saved_stream);
      cp->cur_qset = saved_qset;
   }

   if (cp->pass.append_failed || cp->pass.nsegs == before) {
      if (cp_debug->debug_episode)
         fprintf(stderr, "episode-cut: append failed=%d nsegs %u->%u\n",
                 (int)cp->pass.append_failed, before, cp->pass.nsegs);
      /* A failed append may have run vertex work on its side stream before
       * backing out; when it was the would-be first segment, cp_pass_finish
       * below returns without joining anything and the flush fence — recorded
       * on the main stream only — would not cover it. */
      if (side && !cp_pass_join(cp, CP_PASS_STREAMS))
         return;
      cp_pass_finish(cp);
      /* The rollback may have latched device loss; the batch must not be
       * re-executed onto a poisoned context. Scoped to the fan-out so that
       * with the flag clear this path is what it was. */
      if (side && cp->device_fatal)
         return;
      cp_draw_execute_batch(cp, &cp->batch);
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
      fprintf(stderr, "cudavk: batch of %u ends: %s\n", cp->batch.ndraws, why);

   unsigned ndraws = cp->batch.ndraws;
   bool blended = cp->batch.blended;
   /* Cleared first: cp_draw_execute_batch() runs a whole frame's worth of driver
    * code and nothing in it may see a batch that is already on its way. */
   cp->batch.pending = false;
   /* Keep the immutable payload intact while append/execute/fallback consumes
    * it. cp_batch_begin_packet overwrites counters for the next batch. */

   if (cp_debug->debug_draw) {
      fprintf(stderr, "cudavk: batch of %u draws\n", ndraws);
      for (unsigned d = 0; d < MIN2(ndraws, 4u); d++) {
         fprintf(stderr, "  vs row %u:", d);
         for (unsigned i = 0; i < cp->batch.key.num_vs_ubos; i++)
            fprintf(stderr, " %p",
                    (void *)(uintptr_t)cp->batch.vs_ubos[d * CP_ARG_UBO_STRIDE + i]);
         fprintf(stderr, "\n");
         /* And the fragment stage's, which the vertex ones do not stand in
          * for: a batch whose materials differ differs here and nowhere
          * else. */
         fprintf(stderr, "  fs row %u:", d);
         for (unsigned i = 0; i < cp->batch.key.num_fs_ubos; i++)
            fprintf(stderr, " %p",
                    (void *)(uintptr_t)cp->batch.fs_ubos[d * CP_ARG_UBO_STRIDE + i]);
         fprintf(stderr, "\n");
      }
   }

   if (blended && cp_pass_appendable(cp, &cp->batch)) {
      cp_pass_append(cp, ndraws);
      return;
   }
   if (!blended && cp_opaque_appendable(cp, &cp->batch)) {
      cp_opaque_append(cp, ndraws);
      return;
   }

   /* Whatever the episode holds was submitted before these draws. */
   cp_pass_finish(cp);
   cp->tile_census_solo += ndraws;
   cp_draw_execute_batch(cp, &cp->batch);
}

void
cp_batch_flush_why(struct cp_context *cp, const char *why)
{
   cp_batch_flush_defer_why(cp, why);
   if (cp_debug->debug_episode && cp->pass.nsegs)
      fprintf(stderr, "episode-cut: flush_why=%s nsegs=%u\n", why,
              cp->pass.nsegs);
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
cp_render_scope_begin(struct cp_context *cp, const struct cp_render_scope *scope)
{
   cp->plan.scopes++;
   cp_batch_flush_why(cp, "framebuffer");
   if (cp_debug->debug_episode)
      fprintf(stderr, "episode-cut: begin_render\n");
   cp_pass_finish(cp);
   cp->pass.scope = *scope;
   cp->pass.scope_open = true;
   cp_context_set_framebuffer(cp, &scope->fb, scope->attachment_samples);
   if (scope->fb.has_zs) {
      cp->depthbuf_cleared = false;
      if (scope->depth.load && cp_depth_attachment_xfer(cp, scope, false))
         cp->depthbuf_cleared = true;
   }
}

void
cp_render_scope_end(struct cp_context *cp)
{
   cp_batch_flush_why(cp, "render scope end");
   if (cp_debug->debug_episode)
      fprintf(stderr, "episode-cut: end_render\n");
   cp_pass_finish(cp);
   if (cp->pass.scope.depth.store)
      cp_depth_attachment_xfer(cp, &cp->pass.scope, true);
   cp->pass.scope_open = false;
}

void
cp_context_set_framebuffer(struct cp_context *cp, const struct cp_fb_desc *fb,
                           unsigned samples)
{
   /* Coverage and depth are per sample, so the buffers scale with the sample
    * count and it has to force a reallocation the same way the size does. */
   if (samples > CP_MAX_SAMPLES)
      samples = CP_MAX_SAMPLES;
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
   unsigned w = fb->width, h = fb->height;
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
      cp->plan.fb_reallocs++;
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

      CUresult e1 = cp_mem_alloc_retry(cp, &cp->visbuf,
                               cp->fb_cap_px_samples * sizeof(uint64_t));
      CUresult e2 = cp_mem_alloc_retry(cp, &cp->depthbuf,
                               cp->fb_cap_px_samples * sizeof(uint32_t));
      CP_CU_WARN(cp_mem_alloc_retry(cp, &cp->reject,
                            cp->fb_cap_px * CP_DISCARD_LAYERS * sizeof(uint32_t)),
                 "cuMemAlloc(reject)");
      CP_CU_WARN(cp_mem_alloc_retry(cp, &cp->resolved, cp->fb_cap_px), "cuMemAlloc(resolved)");
      CP_CU_WARN(cp_mem_alloc_retry(cp, &cp->peel_next, cp->fb_cap_px * sizeof(uint32_t)),
                 "cuMemAlloc(peel_next)");
      /* Managed, because the host reads it between passes to decide
       * whether another one is worth launching. */
      if (!cp->peel_any)
         cp_mem_alloc_managed_retry(cp, &cp->peel_any, sizeof(uint32_t));
      if (e1 != CUDA_SUCCESS || e2 != CUDA_SUCCESS)
         fprintf(stderr, "cudavk: visbuf/depthbuf alloc %zu px x %u samples "
                 "failed (%d, %d)\n", cp->fb_cap_px, samples, e1, e2);

      /* Contents are new, whatever was cleared before is gone. */
      cp->depthbuf_cleared = false;
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
enum cp_vf_conv
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
uint32_t
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

/*
 * Fill one rectangle of a device surface with an already-packed value, as a
 * kernel on cp->stream.
 *
 * The point of the kernel is not that it is faster than the host loop it
 * replaces -- it is that it is *ordered*. A host store into the same pages is
 * a write-after-write race against kernels that may still be running, with no
 * edge in either direction. Launching on cp->stream supplies the edge.
 *
 * `value` is taken already packed. Packing here as well would silently
 * produce a different colour.
 *
 * Returns false when the caller has to fall back: the kernels index on the
 * pixel size with no else arm, so a size they do not name writes nothing
 * rather than something wrong -- which is the worse failure of the two,
 * because nothing looks like "the clear did not run".
 */
static bool
cp_clear_rect_impl(struct cp_context *cp, void *data, uint64_t offset,
                   unsigned width, unsigned height, unsigned stride,
                   unsigned pixel_size, const uint32_t value[4],
                   const uint32_t mask[4], bool depth)
{
   struct cp_device *screen = cp->dev;
   CUfunction fn = mask  ? screen->kernels.clear_masked_kernel
                 : depth ? screen->kernels.clear_depth_kernel
                         : screen->kernels.clear_kernel;

   if (!fn || !data)
      return false;

   /* Depth is Z16/Z32F/Z24X8 only, colour excludes the three-component
    * formats R8G8B8, R16G16B16 and R32G32B32. */
   if (depth && !mask) {
      if (pixel_size != 2 && pixel_size != 4)
         return false;
   } else if (pixel_size != 1 && pixel_size != 2 && pixel_size != 4 &&
              pixel_size != 8 && pixel_size != 16) {
      return false;
   }

   if (!width || !height)
      return true;

   struct cp_clear_args args = {
      .target = (uint64_t)(uintptr_t)data + offset,
      .width = width, .height = height,
      .stride = stride,
      .pixel_size = pixel_size,
   };
   memcpy(args.clear_value, value, sizeof(args.clear_value));
   if (mask)
      memcpy(args.clear_mask, mask, sizeof(args.clear_mask));

   void *params[] = { &args };
   return cp_launch(cp, fn,
      (width + 15) / 16, (height + 15) / 16, 1,
      16, 16, 1,
      0, cp->stream, params, NULL) == CUDA_SUCCESS;
}

bool
cp_clear_rect(struct cp_context *cp, void *data, uint64_t offset,
              unsigned width, unsigned height, unsigned stride,
              unsigned pixel_size, const uint32_t value[4], bool depth)
{
   return cp_clear_rect_impl(cp, data, offset, width, height, stride,
                             pixel_size, value, NULL, depth);
}

/*
 * The same rectangle, writing only the bits `mask` names.
 *
 * vkCmdClearDepthStencilImage can name one aspect of a format whose two
 * aspects share a word -- D24_UNORM_S8_UINT does, and D32_SFLOAT_S8_UINT
 * shares an eight-byte element -- and the aspect that was not named has to
 * come out of the clear unchanged.
 */
bool
cp_clear_rect_masked(struct cp_context *cp, void *data, uint64_t offset,
                     unsigned width, unsigned height, unsigned stride,
                     unsigned pixel_size, const uint32_t value[4],
                     const uint32_t mask[4])
{
   return cp_clear_rect_impl(cp, data, offset, width, height, stride,
                             pixel_size, value, mask, false);
}
