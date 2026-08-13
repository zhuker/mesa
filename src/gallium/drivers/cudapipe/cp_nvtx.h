#ifndef CP_NVTX_H
#define CP_NVTX_H

/*
 * Timeline ranges for Nsight Systems.
 *
 * Every shader this driver compiles is a CUDA kernel named `main`, so a trace
 * of it is a wall of identical rows and the vertex and fragment stages sum
 * into one — which is why `tests/cp_prof_kernels.py` exists to split them by
 * grid size. That works around the problem from outside. These ranges fix it
 * from inside: the host says what it is doing while it does it, so the
 * timeline names draws and stages directly, and the skill pack's
 * `report-query` can group by NVTX range instead of by kernel name.
 *
 * **A range marks when the host issued the work, not when the device ran it.**
 * Every launch is asynchronous, so a range closes as soon as the calls are
 * queued. That is the useful reading — the range tells you which draw and
 * which stage a kernel on the GPU row belongs to, and the gap between a range
 * ending and its kernels finishing is the pipelining. It is not a device
 * timing; `CUDAPIPE_DEBUG_TIME` and its CUDA events are that.
 *
 * Off unless `CUDAPIPE_NVTX` is set, because this driver issues thousands of
 * draws a frame and a range is not free. When off each call is a predictable
 * branch on a cached flag.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef CP_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

static inline bool
cp_nvtx_enabled(void)
{
#ifdef CP_HAVE_NVTX
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("CUDAPIPE_NVTX") ? 1 : 0;
   return enabled != 0;
#else
   return false;
#endif
}

static inline void
cp_nvtx_push(const char *name)
{
#ifdef CP_HAVE_NVTX
   if (cp_nvtx_enabled())
      nvtxRangePushA(name);
#else
   (void)name;
#endif
}

/* Formatted, for a range that carries a count — one per draw, not one per
 * primitive, so the snprintf is affordable at the rate it is called. */
static inline void
cp_nvtx_pushf(const char *fmt, ...)
{
#ifdef CP_HAVE_NVTX
   if (!cp_nvtx_enabled())
      return;
   char buf[128];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   nvtxRangePushA(buf);
#else
   (void)fmt;
#endif
}

static inline void
cp_nvtx_pop(void)
{
#ifdef CP_HAVE_NVTX
   if (cp_nvtx_enabled())
      nvtxRangePop();
#endif
}

static inline void
cp_nvtx_mark(const char *name)
{
#ifdef CP_HAVE_NVTX
   if (cp_nvtx_enabled())
      nvtxMarkA(name);
#else
   (void)name;
#endif
}

/*
 * Scoped range, closed on every exit from the enclosing block.
 *
 * cp_draw_vbo() returns from about twenty places — a missing vertex buffer, a
 * failed allocation, a launch error — and a range closed only at the bottom
 * would leak on each of them, which in NVTX means every later range nests one
 * level deeper until the timeline is unreadable. A range that has to be
 * balanced by hand across twenty exits will not stay balanced, so it is not
 * done by hand.
 */
static inline void
cp_nvtx_scope_end(const int *unused)
{
   (void)unused;
   cp_nvtx_pop();
}

#define CP_NVTX_SCOPE(name)                                                   \
   __attribute__((cleanup(cp_nvtx_scope_end))) int                            \
   cp_nvtx_scope_##__LINE__ = (cp_nvtx_push(name), 0)

#define CP_NVTX_SCOPEF(...)                                                   \
   __attribute__((cleanup(cp_nvtx_scope_end))) int                            \
   cp_nvtx_scopef_##__LINE__ = (cp_nvtx_pushf(__VA_ARGS__), 0)

#endif /* CP_NVTX_H */
