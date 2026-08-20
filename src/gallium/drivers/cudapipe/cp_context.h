#ifndef CP_CONTEXT_H
#define CP_CONTEXT_H

#include "cp_renderer.h"

#include "pipe/p_context.h"
#include "pipe/p_state.h"

struct cp_gallium {
   struct pipe_context base;   /* first: entry points cast pipe_context to this */
   struct cp_context cp;

   /*
    * State this front end keeps for itself. None of it is read by the
    * pipeline: the framebuffer, the CSO copies and the element array exist to
    * decide whether a held-back batch must be flushed and to fill the batch
    * key's state blob, which is the front end's business by construction.
    */
   struct pipe_framebuffer_state framebuffer;
   struct pipe_viewport_state viewport_cso;
   struct pipe_rasterizer_state rasterizer_cso;
   struct pipe_depth_stencil_alpha_state depth_stencil_cso;
   struct pipe_blend_state blend_state;
   struct pipe_vertex_element vertex_elements[16];
};

/* The Gallium object around a renderer, for the adapter's own state. */
static inline struct cp_gallium *
cp_gallium_of(struct cp_context *cp)
{
   return (struct cp_gallium *)((char *)cp - offsetof(struct cp_gallium, cp));
}

/*
 * The one place a pipe_context becomes the renderer.
 *
 * Written as a function because the hand-written cast is silently wrong the
 * moment the layout changes, and a wrong one compiles: making pipe_context a
 * member rather than the head of cp_context left ten casts in cp_resource.c
 * pointing at the wrong offset, and every sample still exited 0 while the
 * capture rendered 28% wrong.
 */
static inline struct cp_context *
cp_ctx(struct pipe_context *ctx)
{
   return &((struct cp_gallium *)ctx)->cp;
}


struct pipe_context *
cudapipe_create_context(struct pipe_screen *screen, void *priv, unsigned flags);


/*
 * Submit whatever draws are being held back for merging, if any.
 *
 * Anything that observes the framebuffer — a flush, a map, a clear, a blit, a
 * copy, a dispatch — has to call this first, or it looks at a frame with draws
 * missing from it. Cheap and idempotent when nothing is pending.
 */

/*
 * The same, naming what ended the batch. Every state change that a batch
 * cannot survive goes through this, and CUDAPIPE_DEBUG_BATCH prints the
 * reason — which is the only practical way to find out why a sample that
 * looks batchable is producing batches of one.
 */
/* The deferrable variant: the pending batch may become a pass-episode
 * segment, and a pending episode stays open. Only for state changes an
 * episode carries per segment; see the definition. */

/* Bring a renderer up on a device, and tear it down. No pipe_screen and no
 * pipe_context are involved: this is the entry point a Vulkan front end uses. */

/*
 * How the sampler and the fragment writeback decode and encode a format, or
 * CP_TEXEL_UNSUPPORTED / a negative result when they can't handle it at all.
 *
 * The screen reports format support from these, so that what the driver claims
 * to support and what its kernels can actually decode cannot drift apart.
 */
uint32_t cp_texel_encoding_from_format(enum pipe_format format);
int cp_color_encoding_from_format(enum pipe_format format);

/*
 * Say so when a CUDA call fails, once per site.
 *
 * This driver's failure mode is silence, and every expensive bug found in it
 * so far cost what it did for that reason rather than for its own difficulty.
 * A NULL from cp_allocate_memory became a segfault inside lavapipe, three
 * frames from the kernel that actually faulted. A clip pass was skipped
 * because its scratch allocation returned NULL. The A-buffer switched itself
 * off. A shader read an intrinsic the backend did not implement and got undef.
 * In each case the driver knew, and did not say.
 *
 * Once per site rather than once per failure, because these sit on paths that
 * run hundreds of times a frame: a real fault would otherwise bury its own
 * first line, which is the one worth reading. Nothing here changes behaviour —
 * a caller that can carry on still carries on.
 */
#define CP_CU_WARN(err, what)                                                 \
   do {                                                                       \
      CUresult _cp_e = (err);                                                 \
      if (_cp_e != CUDA_SUCCESS) {                                            \
         static bool _cp_said;                                                \
         if (!_cp_said) {                                                     \
            const char *_cp_n = NULL, *_cp_s = NULL;                          \
            _cp_said = true;                                                  \
            cuGetErrorName(_cp_e, &_cp_n);                                    \
            cuGetErrorString(_cp_e, &_cp_s);                                  \
            fprintf(stderr, "cudapipe: %s failed at %s:%d — %s (%d)%s%s\n",   \
                    (what), __func__, __LINE__,                               \
                    _cp_n ? _cp_n : "unknown", (int)_cp_e,                    \
                    _cp_s ? ": " : "", _cp_s ? _cp_s : "");                   \
            if (_cp_e == CUDA_ERROR_ILLEGAL_ADDRESS ||                        \
                _cp_e == CUDA_ERROR_LAUNCH_FAILED)                            \
               fprintf(stderr, "cudapipe:   this error is sticky — every "    \
                       "later CUDA call fails too, so the first report is "   \
                       "the one that names the cause. Re-run with "           \
                       "CUDA_LAUNCH_BLOCKING=1, or under compute-sanitizer "  \
                       "to name the kernel and the address.\n");              \
         }                                                                    \
      }                                                                       \
   } while (0)

/*
 * A launch that says so when it fails.
 *
 * Worth knowing what this can and cannot tell you. A launch is asynchronous,
 * so the error it returns is rarely its own: it is whatever sticky error a
 * previous kernel left behind, and the first launch to report is the first one
 * issued after the fault, not the one that caused it. That is why the driver
 * reported "VS launch failed: 700" for a fault in vertex fetch.
 *
 * What it is good for is the moment of transition — turning "a segfault
 * somewhere in libc, three frames later" into a line naming a CUDA error and
 * the tools that find the kernel. Synchronous failures, a bad grid or too much
 * shared memory, it does report exactly.
 */
#define CP_LAUNCH(...) CP_CU_WARN(cuLaunchKernel(__VA_ARGS__), "cuLaunchKernel")

#endif
