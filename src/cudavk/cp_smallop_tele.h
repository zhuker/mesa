#ifndef CP_SMALLOP_TELE_H
#define CP_SMALLOP_TELE_H

/*
 * Iteration 26 S0: a census of the sub-4 KB device operations, taken from the
 * source side rather than from a profiler.
 *
 * CUPTI adds host cost to exactly the calls being counted, so the number of
 * small copies and clears per frame cannot be measured by tracing them. This
 * counts them where they are issued instead. It is off unless
 * CUDAVK_UPLOAD_STATS is set, and when it is off the whole mechanism is one
 * predictable branch per call.
 *
 * The counting is done by intercepting the CUDA entry points with macros, so
 * that a call site is attributed without editing forty of them. Uploads are
 * different: they all funnel through cp_upload_end(), so that one site
 * records its caller's return address instead, and the report prints an
 * offset that addr2line resolves against the shared object.
 */

#include <cuda.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The epoch every wrapper below moves. Its own header, so that a file which
 * enqueues work without wanting the census interception can move it too. */
#include "cp_devop.h"

enum cp_smallop_kind {
   CP_SMALLOP_HTOD_ASYNC,
   CP_SMALLOP_HTOD_SYNC,
   CP_SMALLOP_DTOH,
   CP_SMALLOP_MEMSET_ASYNC,
   CP_SMALLOP_MEMSET_SYNC,
   CP_SMALLOP_CTXSYNC,
   CP_SMALLOP_UPLOAD_WRAP,
   CP_SMALLOP_SUBMIT,
   CP_SMALLOP_KINDS,
};

extern bool cp_smallop_enabled;

void cp_smallop_note(const char *file, int line, const void *ra,
                     enum cp_smallop_kind kind, size_t bytes);
void cp_smallop_report(void);

/*
 * The programmatic-dependent-launch predecessor check rides on the same
 * interception, because it needs the same fact from the other direction.
 *
 * CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION is defined against
 * "the previous kernel in the stream", so it may only be set on a launch
 * whose immediate predecessor on that stream really is the kernel the caller
 * named -- not a clear, not a copy, not an event. Those are exactly the calls
 * this header already renames, so each of them bumps an epoch and cp_launch()
 * refuses the attribute when the epoch moved since the predecessor launched.
 * That turns "no memset can be sitting there" from a claim about the source
 * into something the code checks.
 *
 * The epoch itself lives in cp_devop.h and is always counted -- CUDAVK_VS_LANE
 * needs it with no debug flag set. cp_pdl_watch only arms the PDL attribute.
 */
extern bool cp_pdl_watch;

static inline void
cp_smallop_hit(const char *file, int line, enum cp_smallop_kind kind,
               size_t bytes)
{
   if (cp_smallop_enabled)
      cp_smallop_note(file, line, NULL, kind, bytes);
}

/*
 * The wrappers below are defined while the CUDA headers' own macros are still
 * in force, so each call resolves to the versioned entry point cuda.h names.
 * The interception macros are defined afterwards, so nothing recurses.
 */
static inline CUresult
cp_smallop_htod_async(const char *f, int l, CUdeviceptr dst, const void *src,
                      size_t n, CUstream s)
{
   cp_smallop_hit(f, l, CP_SMALLOP_HTOD_ASYNC, n);
   cp_devop_note();
   return cuMemcpyHtoDAsync(dst, src, n, s);
}

/* The upload funnel notes its caller itself and must not be counted twice. */
static inline CUresult
cp_smallop_htod_async_raw(CUdeviceptr dst, const void *src, size_t n,
                          CUstream s)
{
   cp_devop_note();
   return cuMemcpyHtoDAsync(dst, src, n, s);
}

static inline CUresult
cp_smallop_htod(const char *f, int l, CUdeviceptr dst, const void *src,
                size_t n)
{
   cp_smallop_hit(f, l, CP_SMALLOP_HTOD_SYNC, n);
   return cuMemcpyHtoD(dst, src, n);
}

static inline CUresult
cp_smallop_dtoh(const char *f, int l, void *dst, CUdeviceptr src, size_t n)
{
   cp_smallop_hit(f, l, CP_SMALLOP_DTOH, n);
   return cuMemcpyDtoH(dst, src, n);
}

static inline CUresult
cp_smallop_memset32_async(const char *f, int l, CUdeviceptr dst, unsigned v,
                          size_t n, CUstream s)
{
   cp_smallop_hit(f, l, CP_SMALLOP_MEMSET_ASYNC, n * 4);
   cp_devop_note();
   return cuMemsetD32Async(dst, v, n, s);
}

static inline CUresult
cp_smallop_memset8_async(const char *f, int l, CUdeviceptr dst,
                         unsigned char v, size_t n, CUstream s)
{
   cp_smallop_hit(f, l, CP_SMALLOP_MEMSET_ASYNC, n);
   cp_devop_note();
   return cuMemsetD8Async(dst, v, n, s);
}

static inline CUresult
cp_smallop_memset8(const char *f, int l, CUdeviceptr dst, unsigned char v,
                   size_t n)
{
   cp_smallop_hit(f, l, CP_SMALLOP_MEMSET_SYNC, n);
   return cuMemsetD8(dst, v, n);
}

/*
 * Not census points: nothing here counts. They exist only so that an event
 * record, a cross-stream wait or a device-to-host copy invalidates a PDL
 * predecessor the same way a clear does.
 */
static inline CUresult
cp_pdl_event_record(CUevent e, CUstream s)
{
   cp_devop_note();
   return cuEventRecord(e, s);
}

static inline CUresult
cp_pdl_stream_wait_event(CUstream s, CUevent e, unsigned int flags)
{
   cp_devop_note();
   return cuStreamWaitEvent(s, e, flags);
}

static inline CUresult
cp_pdl_dtoh_async(void *dst, CUdeviceptr src, size_t n, CUstream s)
{
   cp_devop_note();
   return cuMemcpyDtoHAsync(dst, src, n, s);
}

/* The four the interception did not cover until CUDAVK_VS_LANE needed the
 * epoch to mean every enqueue: the image copies, the buffer-to-buffer copy
 * and the host callbacks in cpvk_cmd.c, all of them issued on the renderer's
 * main stream between draws. A PDL predecessor was already wrong across one
 * of these; nothing measured it because they are rare. */
static inline CUresult
cp_devop_memcpy2d_async(const CUDA_MEMCPY2D *copy, CUstream s)
{
   cp_devop_note();
   return cuMemcpy2DAsync(copy, s);
}

static inline CUresult
cp_devop_memcpy3d_async(const CUDA_MEMCPY3D *copy, CUstream s)
{
   cp_devop_note();
   return cuMemcpy3DAsync(copy, s);
}

static inline CUresult
cp_devop_dtod_async(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s)
{
   cp_devop_note();
   return cuMemcpyDtoDAsync(dst, src, n, s);
}

static inline CUresult
cp_devop_launch_host_func(CUstream s, CUhostFn fn, void *user)
{
   cp_devop_note();
   return cuLaunchHostFunc(s, fn, user);
}

static inline CUresult
cp_smallop_ctxsync(const char *f, int l)
{
   cp_smallop_hit(f, l, CP_SMALLOP_CTXSYNC, 0);
   return cuCtxSynchronize();
}

/* cuda.h defines these names as object-like macros for its versioned
 * symbols, so each is retired before the census takes the name. */
#undef cuMemcpyHtoDAsync
#undef cuMemcpyHtoD
#undef cuMemcpyDtoH
#undef cuMemsetD32Async
#undef cuMemsetD8Async
#undef cuMemsetD8
#undef cuCtxSynchronize
#undef cuEventRecord
#undef cuStreamWaitEvent
#undef cuMemcpyDtoHAsync
#undef cuMemcpy2DAsync
#undef cuMemcpy3DAsync
#undef cuMemcpyDtoDAsync
#undef cuLaunchHostFunc

#define cuMemcpyHtoDAsync(d, s, n, st) \
   cp_smallop_htod_async(__FILE__, __LINE__, (d), (s), (n), (st))
#define cuMemcpyHtoD(d, s, n) \
   cp_smallop_htod(__FILE__, __LINE__, (d), (s), (n))
#define cuMemcpyDtoH(d, s, n) \
   cp_smallop_dtoh(__FILE__, __LINE__, (d), (s), (n))
#define cuMemsetD32Async(d, v, n, st) \
   cp_smallop_memset32_async(__FILE__, __LINE__, (d), (v), (n), (st))
#define cuMemsetD8Async(d, v, n, st) \
   cp_smallop_memset8_async(__FILE__, __LINE__, (d), (v), (n), (st))
#define cuMemsetD8(d, v, n) \
   cp_smallop_memset8(__FILE__, __LINE__, (d), (v), (n))
#define cuCtxSynchronize() cp_smallop_ctxsync(__FILE__, __LINE__)
#define cuEventRecord(e, st) cp_pdl_event_record((e), (st))
#define cuStreamWaitEvent(st, e, fl) cp_pdl_stream_wait_event((st), (e), (fl))
#define cuMemcpyDtoHAsync(d, s, n, st) cp_pdl_dtoh_async((d), (s), (n), (st))
#define cuMemcpy2DAsync(c, st) cp_devop_memcpy2d_async((c), (st))
#define cuMemcpy3DAsync(c, st) cp_devop_memcpy3d_async((c), (st))
#define cuMemcpyDtoDAsync(d, s, n, st) cp_devop_dtod_async((d), (s), (n), (st))
#define cuLaunchHostFunc(st, f, u) cp_devop_launch_host_func((st), (f), (u))

#endif /* CP_SMALLOP_TELE_H */
