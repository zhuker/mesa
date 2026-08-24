#ifndef CP_SMALLOP_TELE_H
#define CP_SMALLOP_TELE_H

/*
 * Iteration 26 S0: a census of the sub-4 KB device operations, taken from the
 * source side rather than from a profiler.
 *
 * CUPTI adds host cost to exactly the calls being counted, so the number of
 * small copies and clears per frame cannot be measured by tracing them. This
 * counts them where they are issued instead. It is off unless
 * CUDAPIPE_UPLOAD_STATS is set, and when it is off the whole mechanism is one
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
   return cuMemcpyHtoDAsync(dst, src, n, s);
}

/* The upload funnel notes its caller itself and must not be counted twice. */
static inline CUresult
cp_smallop_htod_async_raw(CUdeviceptr dst, const void *src, size_t n,
                          CUstream s)
{
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
   return cuMemsetD32Async(dst, v, n, s);
}

static inline CUresult
cp_smallop_memset8_async(const char *f, int l, CUdeviceptr dst,
                         unsigned char v, size_t n, CUstream s)
{
   cp_smallop_hit(f, l, CP_SMALLOP_MEMSET_ASYNC, n);
   return cuMemsetD8Async(dst, v, n, s);
}

static inline CUresult
cp_smallop_memset8(const char *f, int l, CUdeviceptr dst, unsigned char v,
                   size_t n)
{
   cp_smallop_hit(f, l, CP_SMALLOP_MEMSET_SYNC, n);
   return cuMemsetD8(dst, v, n);
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

#endif /* CP_SMALLOP_TELE_H */
