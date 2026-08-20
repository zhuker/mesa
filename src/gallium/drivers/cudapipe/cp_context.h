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


#endif
