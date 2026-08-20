#include "cp_context.h"
#include "cp_screen.h"
#include "cp_draw_types.h"
#include "cp_kernels.h"
#include "cp_nvtx.h"
#include "cp_resource.h"
#include "nir_to_ptx/cp_nir_to_llvm.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

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

void cp_pass_finish(struct cp_context *cp);

static void
cp_destroy_context(struct pipe_context *ctx)
{
   struct cp_context *cp = cp_ctx(ctx);
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
   free(cp->pass_segs);
   free(cp->pass_group_ubos);
   for (unsigned k = 0; k < CP_PASS_STREAMS; k++) {
      if (cp->seg_streams[k])
         cuStreamDestroy(cp->seg_streams[k]);
      if (cp->seg_ev[k])
         cuEventDestroy(cp->seg_ev[k]);
      if (cp->seg_qsets[k].nontrivial)
         cuMemFree(cp->seg_qsets[k].nontrivial);
      if (cp->seg_qsets[k].huge_tiles)
         cuMemFree(cp->seg_qsets[k].huge_tiles);
      if (cp->seg_qsets[k].counts)
         cuMemFree(cp->seg_qsets[k].counts);
   }
   if (cp->pass_gate)
      cuEventDestroy(cp->pass_gate);
   for (unsigned i = 0; i < CP_FLUSH_GENS; i++)
      if (cp->flush_retire[i])
         cuEventDestroy(cp->flush_retire[i]);
   if (ctx->stream_uploader)
      u_upload_destroy(ctx->stream_uploader);
   FREE((struct cp_gallium *)ctx);
}

static void
cp_set_framebuffer_state(struct pipe_context *ctx,
                         const struct pipe_framebuffer_state *state)
{
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);

   /* The visibility and depth buffers may be freed below, and the held-back
    * draws were recorded against the framebuffer that is going away. */
   cp_batch_flush_why(cp, "framebuffer");

   /* The bind is the census's accumulation unit: everything drawn to this
    * framebuffer could have shared one tile's on-chip colour and depth. */
   cp_tile_census_end_pass(cp);

   if (cp_debug->debug_passseq)
      fprintf(stderr, "passseq fb %ux%u cbuf=%p zs=%p\n",
              state->width, state->height,
              state->nr_cbufs ? (void *)state->cbufs[0].texture : NULL,
              (void *)state->zsbuf.texture);

   util_copy_framebuffer_state(&g->framebuffer, state);

   /*
    * Resolve the attachments once. The draw path used to unwrap the colour
    * resource and call cp_color_encoding_from_format on every draw to answer
    * the same four questions.
    */
   struct cp_resource *cres = (state->nr_cbufs && state->cbufs[0].texture)
      ? cp_resource(state->cbufs[0].texture) : NULL;
   struct cp_fb_desc fb = (struct cp_fb_desc) {
      .width = state->width,
      .height = state->height,
      .nr_cbufs = state->nr_cbufs,
      .color = cres ? cp_resource_data(cres) : NULL,
      .color_encoding = state->nr_cbufs
         ? cp_color_encoding_from_format(state->cbufs[0].format) : -1,
      .color_sample_stride = cres ? (unsigned)cres->lpr.sample_stride : 0,
      .has_zs = state->zsbuf.texture != NULL,
   };

   /* Coverage and depth are per sample, so the buffers scale with the sample
    * count and it has to force a reallocation the same way the size does. */
   unsigned samples = 1;
   if (state->nr_cbufs && state->cbufs[0].texture)
      samples = MAX2(state->cbufs[0].texture->nr_samples, 1u);
   else if (state->zsbuf.texture)
      samples = MAX2(state->zsbuf.texture->nr_samples, 1u);

   cp_context_set_framebuffer(cp, &fb, samples);
}


static void
cp_set_viewport_states(struct pipe_context *ctx, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *viewports)
{
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);
   if (num_viewports > 0) {
      if (memcmp(&g->viewport_cso, &viewports[0], sizeof(g->viewport_cso)))
         cp_batch_flush_why(cp, "viewport");
      g->viewport_cso = viewports[0];
      memcpy(cp->viewport.scale, viewports[0].scale, sizeof(cp->viewport.scale));
      memcpy(cp->viewport.translate, viewports[0].translate,
             sizeof(cp->viewport.translate));
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
   struct cp_context *cp = cp_ctx(ctx);
   if (num_scissors > 0) {
      /* A batch on the stable clipper records the scissor per draw, so a
       * pending one survives the change; one that cannot resolve a primitive
       * to its draw still has to go. Same condition as the key builder's. */
      if (cp->batch.pending &&
          !(cp->batch.blended ||
            (cp->fs_shader && cp->fs_shader->reads_const_bufs)) &&
          memcmp(&cp->scissor, &scissors[0], sizeof(cp->scissor)))
         cp_batch_flush_why(cp, "scissor");
      /* Field-wise rather than a cast: the two structs happen to agree today
       * and nothing should quietly depend on that. */
      cp->scissor = (struct cp_rect) {
         .minx = scissors[0].minx, .miny = scissors[0].miny,
         .maxx = scissors[0].maxx, .maxy = scissors[0].maxy,
      };
   }
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



/* The one place a pipe_rt_blend_state becomes the driver's own description. */
static struct cp_blend_desc
cp_blend_desc_from_gallium(const struct pipe_rt_blend_state *rt)
{
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
 * What a batch is then allowed to vary is carried as tables with one row per
 * merged draw:
 *
 *  - Both stages' uniform bindings — the same buffer at a different dynamic
 *    offset, which is what dynamicuniformbuffer's 125 cubes a frame differ
 *    in, and the per-draw fragment material the capture rebinds on every
 *    draw. Each shader picks its row out of the table at
 *    CP_ARG_SLOT_UBO_TABLE; the fragment stage's row is resolved by the
 *    interpolator from the primitive index, which the stable clipper keeps
 *    in submission order.
 *  - The index range. Draws of different sizes out of one buffer concatenate,
 *    and cp_vertex_fetch searches a table of slices to find which draw a
 *    thread's vertex came from — see struct cp_draw_slice. Without that,
 *    bloom's 154 draws a frame and vulkanscene's 19 merge none of themselves,
 *    because no two of them replay the same range.
 *  - The draw parameters — gl_BaseVertex, gl_BaseInstance, gl_DrawID — one
 *    CP_ARG_DRAW_PARAM_STRIDE row per draw at args[7], by the same row.
 */

/*
 * What this front end means by "the same pipeline state": the CSO structs
 * whole. The native driver's answer will be a pipeline handle and its dynamic
 * state, and the batcher will not know the difference.
 */
struct cp_gallium_batch_state {
   struct pipe_viewport_state viewport;
   struct pipe_rasterizer_state rasterizer;
   struct pipe_depth_stencil_alpha_state depth_stencil;
   struct pipe_blend_state blend_state;
   struct pipe_vertex_element vertex_elements[16];
};

const struct cp_batch_state_field *
cp_batch_state_fields(unsigned *count)
{
#define S(name) { #name, offsetof(struct cp_gallium_batch_state, name), \
                  sizeof(((struct cp_gallium_batch_state *)0)->name) }
   static const struct cp_batch_state_field fields[] = {
      S(viewport), S(rasterizer), S(depth_stencil), S(blend_state),
      S(vertex_elements),
   };
#undef S
   *count = ARRAY_SIZE(fields);
   return fields;
}

/* Fill in everything two draws must agree on. See struct cp_batch_key. */
static void
cp_batch_build_key(struct cp_gallium *g, const struct cp_draw_call *info,
                   const struct cp_draw_range *draws,
                   struct cp_batch_key *key, bool blended)
{
   struct cp_context *cp = &g->cp;
   struct pipe_framebuffer_state *fb = &g->framebuffer;

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
   key->start_instance = info->start_instance;
   key->index_resource = info->index_size ? info->index_ptr : NULL;
   /*
    * The range is not a merge condition, and neither are the draw parameters
    * any more: a batch carries one slice per draw for the fetch kernel and
    * one CP_ARG_DRAW_PARAM_STRIDE row per draw at args[7] for the shader, so
    * gl_BaseVertex, gl_BaseInstance and gl_DrawID all resolve per draw
    * through the batch row. start_instance stays keyed above: the fetch
    * kernel's instance-divisor gather still reads it as one scalar.
    */

   /*
    * The state blob. Zeroed whole first so that padding inside the structs
    * cannot make two identical states compare different.
    */
   struct cp_gallium_batch_state gs;
   memset(&gs, 0, sizeof(gs));
   gs.viewport = g->viewport_cso;
   /*
    * The scissor is a merge condition only where a primitive cannot be
    * resolved to its draw: a batch on the stable clipper carries one clip
    * rectangle per draw instead (see cp_rasterize_args.clip_rects), and with
    * the scissor test off in the rasterizer state the value is unread. The
    * rasterizer state itself stays keyed, so the enable bit agrees across
    * any batch.
    */
   if (cp->rasterizer.scissor &&
       !(blended || (cp->fs_shader && cp->fs_shader->reads_const_bufs)))
      key->scissor = cp->scissor;
   gs.rasterizer = g->rasterizer_cso;
   gs.depth_stencil = g->depth_stencil_cso;
   gs.blend_state = g->blend_state;
   key->blend_enabled = cp->blend_enabled;

   memcpy(gs.vertex_elements, g->vertex_elements, sizeof(gs.vertex_elements));
   static_assert(sizeof(gs) <= CP_BATCH_STATE_BYTES,
                 "the adapter's batch state does not fit the key");
   memcpy(key->state, &gs, sizeof(gs));
   key->num_vertex_elements = cp->num_vertex_elements;
   key->vertex_stride = cp->vertex_stride;
   key->num_vertex_buffers = cp->num_vertex_buffers;

   key->num_vs_ubos = cp->num_vs_ubos;
   /*
    * The fragment binding *pointers* are deliberately absent for every batch
    * now, not only a blended one: an opaque batch resolves visibility by
    * atomicMin before anything is shaded, so per-draw fragment bindings never
    * had anything to do with the ordering argument, and the same per-draw
    * table that carries them for a blended batch carries them here. The
    * *count* is newly keyed for both kinds: the launch fills each table row
    * cp->num_fs_ubos wide at flush time, so two draws that disagree on it
    * cannot share one table. See cp_batch_record() and cp_fs_launch_shader().
    */
   if (cp->fs_shader && cp->fs_shader->reads_const_bufs)
      key->num_fs_ubos = cp->num_fs_ubos;
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
      F(mode), F(index_size), F(start_instance),
      F(index_resource),
      F(scissor), F(blend_enabled),
      F(num_vertex_elements), F(vertex_stride),
      F(num_vertex_buffers),
      F(num_fs_ubos), F(num_vs_ubos),
      F(sampler_table), F(num_samplers),
   };
#undef F
   char line[512];
   size_t n = 0;
   /* The front end's own state, named piece by piece so the report still says
    * which one broke the batch rather than just "state". */
   unsigned num_state_fields;
   const struct cp_batch_state_field *sf =
      cp_batch_state_fields(&num_state_fields);
   for (unsigned i = 0; i < num_state_fields; i++) {
      const size_t off = offsetof(struct cp_batch_key, state) + sf[i].off;
      if (!memcmp((const char *)a + off, (const char *)b + off, sf[i].size))
         continue;
      int w = snprintf(line + n, sizeof(line) - n, "%s%s",
                       n ? "," : "", sf[i].name);
      if (w < 0 || (size_t)w >= sizeof(line) - n)
         break;
      n += w;
   }
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
cp_batch_structural(struct cp_context *cp, const struct cp_draw_call *info,
                    const struct pipe_draw_indirect_info *indirect,
                    const struct cp_draw_range *draws,
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
   if (info->has_user_indices)
      return false;
   if (info->index_size && !info->index_ptr)
      return false;

   /* A vertex buffer is what the batch replays; a shader building its
    * positions from gl_VertexIndex alone has nothing to gain and is left on
    * the single-draw path. */
   if (!cp->num_vertex_buffers || !cp->vb_base[0])
      return false;

   /* Framebuffer and the buffers the stages need. */
   if (!cp->fb.nr_cbufs || !cp->fb.color ||
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
    * clipper's output buffer past what the arena will hand out. Instances
    * multiply the assembled stream, so they count here — silently admitting
    * a small draw with many instances is exactly how the cap would fail. */
   {
      uint64_t tris = (uint64_t)cp_triangles_for_draw(info->mode,
                                                      draws[0].count) *
                      MAX2(info->instance_count, 1u);
      if (tris == 0 || tris > CP_MAX_BATCH_TRIS)
         return false;
   }

   return true;
}


static bool
cp_batch_eligible(struct cp_context *cp, const struct cp_draw_call *info,
                  const struct pipe_draw_indirect_info *indirect,
                  const struct cp_draw_range *draws,
                  unsigned num_draws, bool *blended)
{
   if (cp_debug->no_batch)
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


static void
cp_draw_vbo(struct pipe_context *ctx, const struct pipe_draw_info *gallium_info,
            unsigned drawid_offset,
            const struct pipe_draw_indirect_info *indirect,
            const struct pipe_draw_start_count_bias *gallium_draws,
            unsigned num_draws)
{
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);
   struct cp_device *screen = cp->screen;

   /*
    * The Gallium adapter, and the only place in the driver that sees
    * Gallium's draw types. Everything below takes the driver's own
    * description, which the native Vulkan front end fills in from a recorded
    * command buffer instead. pipe_draw_start_count_bias and cp_draw_range
    * are layout-compatible, so the array passes straight through.
    */
   struct cp_draw_call call = {
      .mode = gallium_info->mode,
      .index_size = gallium_info->index_size,
      .instance_count = gallium_info->instance_count,
      .start_instance = gallium_info->start_instance,
      .has_user_indices = gallium_info->has_user_indices,
      .index_ptr = (gallium_info->index_size && !gallium_info->has_user_indices &&
                    gallium_info->index.resource)
         ? cp_resource_data(cp_resource(gallium_info->index.resource)) : NULL,
   };
   const struct cp_draw_call *info = &call;
   const struct cp_draw_range *draws =
      (const struct cp_draw_range *)gallium_draws;

   if (!screen->kernels.initialized || !screen->kernels.rasterize_triangles)
      return;
   if (num_draws == 0 || draws[0].count == 0)
      return;

   /*
    * Culling both faces draws nothing. cp_cull_mode() returns 0 for that case
    * — "keep everything" — with a comment saying the draw is skipped instead,
    * and nothing skipped it, so a draw that should have rendered nothing
    * rendered in full. Points have no winding, and the rasterizer already
    * ignores culling for them, so they are exempt here too.
    */
   if (info->mode != MESA_PRIM_POINTS &&
       (cp->rasterizer.cull_face & CP_FACE_FRONT_AND_BACK) ==
       CP_FACE_FRONT_AND_BACK)
      return;

   cuCtxSetCurrent(screen->cuda_ctx);

   bool blended = false;
   bool eligible = cp_batch_eligible(cp, info, indirect, draws, num_draws,
                                     &blended);

   if (cp_debug->debug_passseq) {
      /* Mirrors the eligibility tests, so an ineligible draw says which one
       * refused it — statistics only, never consulted for a decision. */
      const char *why = "";
      unsigned t = cp_triangles_for_draw(info->mode, draws[0].count);
      if (!eligible) {
         if (cp_debug->no_batch) why = ":nobatch";
         else if (!cp->vs_shader || !cp->vs_shader->kernel ||
                  !cp->fs_shader || !cp->fs_shader->kernel) why = ":noshader";
         else if (info->mode != MESA_PRIM_TRIANGLES) why = ":topology";
         else if (num_draws != 1 || indirect) why = ":multidraw";
         else if (MAX2(info->instance_count, 1u) != 1) why = ":instanced";
         else if (info->has_user_indices) why = ":userindex";
         else if (!cp->num_vertex_buffers ||
                  !cp->vb_base[0]) why = ":novb";
         else if (!g->framebuffer.nr_cbufs ||
                  !g->framebuffer.cbufs[0].texture ||
                  !cp->visbuf || !cp->depthbuf) why = ":nofb";
         else if (t == 0 || t > CP_MAX_BATCH_TRIS) why = ":toobig";
         else if (!cp->blend_enabled) {
            why = cp->fs_shader->uses_discard ? ":discard" : ":depthfunc";
         } else {
            if (cp->fs_shader->uses_discard) why = ":discard";
            else if (!cp->peel_next) why = ":nopeel";
            else if (MAX2(cp->fb_samples, 1u) != 1 ||
                     cp->depth_stencil.depth_writemask) why = ":abufgate";
            else why = ":abufmisc";
         }
      }
      fprintf(stderr, "passseq draw vs=%p fs=%p %s%s tris=%u vp=%.0fx%.0f\n",
              (void *)cp->vs_shader, (void *)cp->fs_shader,
              !eligible ? "inelig" : blended ? "blended" : "opaque", why, t,
              cp->viewport.scale[0] * 2.0f, cp->viewport.scale[1] * 2.0f);
   }

   if (eligible) {
      struct cp_batch_key key;
      cp_batch_build_key(cp_gallium_of(cp), info, draws, &key, blended);

      /* Instances included: this is what accumulates into batch.tris, which
       * the triangle cap and every grid downstream are sized from. The
       * structural gate above already refused anything whose product
       * overflows the cap. */
      unsigned tris = cp_triangles_for_draw(info->mode, draws[0].count) *
                      MAX2(info->instance_count, 1u);

      /* The cap, overridable so that a suspect batch can be bisected by size
       * without a rebuild — 1 exercises the whole batched path on a batch of
       * one, which is the case that has to stay bit-identical. */
      const unsigned cap = cp_debug->batch_max;

      if (cp->batch.pending) {
         const char *why = NULL;
         if (memcmp(&key, &cp->batch.key, sizeof(key))) {
            why = "the next draw differs in state or geometry";
            if (cp_debug->debug_batchdiff)
               cp_batch_key_report_diff(&key, &cp->batch.key);
         }
         else if (cp->batch.ndraws >= (unsigned)cap)
            why = "the draw cap";
         else if (cp->batch.tris + tris > CP_MAX_BATCH_TRIS)
            why = "the triangle cap";

         if (!why) {
            cp_batch_record(cp, &draws[0], tris, drawid_offset,
                            info->instance_count);
            return;
         }
         cp_batch_flush_defer_why(cp, why);
      }

      /* Whatever was pending has gone; this draw opens the next batch. */
      cp->batch.key = key;
      cp->batch.info = *info;
      cp->batch.drawid_offset = drawid_offset;
      cp->batch.pending = true;
      /* Set before the first row is recorded: cp_batch_record() reads it to
       * decide whether the fragment bindings have to be snapshotted too. */
      cp->batch.blended = blended;
      cp_batch_record(cp, &draws[0], tris, drawid_offset,
                      info->instance_count);
      return;
   }

   cp_batch_flush_why(cp, "the next draw cannot be batched");
   cp_draw_execute(cp, info, drawid_offset, draws, num_draws, 1, NULL, NULL,
                   NULL, NULL, NULL, NULL);
}

static void
cp_launch_grid(struct pipe_context *ctx, const struct pipe_grid_info *info)
{
   struct cp_context *cp = cp_ctx(ctx);
   struct cp_shader_binary *bin = cp->compute_shader;

   /* A dispatch may read what the held-back draws were going to write. */
   cp_batch_flush_why(cp, "a compute dispatch");
   cp_tile_census_cut(cp, CP_TILE_CUT_COMPUTE);

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
   if (cp_debug->debug_launch) {
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
   /*
    * The block goes into the upload arena by DMA, the way the draw path's
    * argument blocks do. It used to be a cuMemAllocManaged pair freed after
    * the dispatch, which is what the cuCtxSynchronize below them was for —
    * the memory could not be released until the kernel reading it had
    * finished. Nothing here needs the host to wait: the arena is reclaimed in
    * bulk, and a managed block the host has just written is the page-fault
    * stall cp_upload() exists to avoid.
    */
   const size_t args_bytes = 34 * sizeof(void *);
   const size_t grid_off = args_bytes;
   void *block_host;
   CUdeviceptr args_dev = cp_upload_begin(cp, grid_off + 3 * sizeof(uint32_t),
                                          &block_host);
   if (!args_dev)
      return;

   memset(block_host, 0, grid_off + 3 * sizeof(uint32_t));
   void **arg_ptrs_host = block_host;
   arg_ptrs_host[0] = (void *)(uintptr_t)(args_dev + grid_off);
   arg_ptrs_host[1] = NULL;

   for (unsigned i = 0; i < CP_MAX_SHADER_BUFFERS; i++)
      arg_ptrs_host[2 + i] = cp->compute_ssbos[i].buffer;

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++)
      arg_ptrs_host[18 + i] = cp->compute_ubos[i].buffer;

   uint32_t *grid_size = (uint32_t *)((char *)block_host + grid_off);
   grid_size[0] = grid[0];
   grid_size[1] = grid[1];
   grid_size[2] = grid[2];
   cp_upload_end(cp, args_dev, block_host, grid_off + 3 * sizeof(uint32_t));

   void *args_ptr_val = (void *)(uintptr_t)args_dev;
   void *kernel_params[] = { &args_ptr_val };

   if (cp_debug->debug_launch) {
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
   struct cp_context *cp = cp_ctx(ctx);
   cuCtxSetCurrent(cp->screen->cuda_ctx);

   /* Before the fence: a batch still being held back has not been submitted,
    * so anything waiting on this would be waiting for work that was never
    * queued. */
   cp_batch_flush(cp);
   cp_tile_census_cut(cp, CP_TILE_CUT_FLUSH);

   /* On cp->stream, after cp_batch_flush: every side stream's episode work
    * has been joined into the main stream by cp_pass_finish, so this event
    * really does cover everything the context has queued — unlike the legacy
    * NULL stream it used to be recorded on, which orders against nothing
    * because cp->stream is CU_STREAM_NON_BLOCKING. cp_fence_finish waits on
    * exactly this event. */
   if (fence) {
      struct cp_fence *f = malloc(sizeof(*f));
      if (f) {
         f->refcount = 1;
         cuEventCreate(&f->event, CU_EVENT_DISABLE_TIMING);
         cuEventRecord(f->event, cp->stream);
      }
      *fence = (struct pipe_fence_handle *)f;
   }

   /*
    * Reclaim the scratch arenas.
    *
    * An earlier attempt simply deleted the drain here and let the threshold
    * reclaim in cp_scratch_alloc() cope. Measured regression: +0.4% over the
    * sweep, pbribl +14.5%, negativeviewportheight +9.2%, texture +8.5%,
    * texturecubemap +6.4% — because rewinding at flush is what lets the same
    * pages be reused without reallocating, and deferring it swapped many
    * cheap syncs for occasional expensive cuMemFree/cuMemAlloc churn.
    *
    * This is the reclaim-without-reallocating that comment asked for: the
    * arenas are split into CP_FLUSH_GENS generations, the flush records a
    * retire event behind the generation it is leaving and rewinds into the
    * next one, waiting that generation's own event from CP_FLUSH_GENS-1
    * flushes ago — by then almost always signalled, so the flush stops
    * draining the pipeline it just fed. The overflow arenas, whose cuMemFree
    * genuinely needs an idle device, wait for the threshold reclaim or
    * teardown, both of which still drain first.
    *
    * CUDAPIPE_FLUSH_DRAIN=1 restores the old full drain (flush_gens == 1).
    */
   if (cp->flush_gens <= 1 || !cp->flush_retire[0]) {
      cuCtxSynchronize();
      cp_scratch_reset(cp);
   } else {
      unsigned gen = cp->scratch.current;
      cuEventRecord(cp->flush_retire[gen], cp->stream);
      cp->flush_retire_recorded[gen] = true;
      gen = (gen + 1) % cp->flush_gens;
      cp->scratch.current = gen;
      if (cp->flush_retire_recorded[gen])
         cuEventSynchronize(cp->flush_retire[gen]);
      cp->scratch.used = 0;
      cp->arena_offset = (size_t)gen * (cp->arena_size / cp->flush_gens);
      cp->upload_offset = (size_t)gen * (cp->upload_size / cp->flush_gens);
   }
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
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);
   struct pipe_blend_state next;
   if (state)
      next = *(struct pipe_blend_state *)state;
   else
      memset(&next, 0, sizeof(next));
   if (memcmp(&g->blend_state, &next, sizeof(next)))
      cp_batch_flush_why(cp, "blend state");

   if (state) {
      g->blend_state = next;
      cp->blend_desc = cp_blend_desc_from_gallium(&next.rt[0]);
      cp->blend_enabled = g->blend_state.rt[0].blend_enable;
      if (cp->gpu_state) {
         const struct pipe_rt_blend_state *rt = &g->blend_state.rt[0];
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
      memset(&g->blend_state, 0, sizeof(g->blend_state));
      memset(&cp->blend_desc, 0, sizeof(cp->blend_desc));
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
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);
   struct pipe_rasterizer_state next;
   if (state)
      next = *(struct pipe_rasterizer_state *)state;
   else
      memset(&next, 0, sizeof(next));
   if (memcmp(&g->rasterizer_cso, &next, sizeof(next)))
      cp_batch_flush_why(cp, "rasterizer state");
   g->rasterizer_cso = next;
   cp->rasterizer = (struct cp_raster_state) {
      .cull_face = next.cull_face,
      .front_ccw = next.front_ccw,
      .scissor = next.scissor,
   };
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
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);
   struct pipe_depth_stencil_alpha_state next;
   if (state)
      next = *(struct pipe_depth_stencil_alpha_state *)state;
   else
      memset(&next, 0, sizeof(next));
   if (memcmp(&g->depth_stencil_cso, &next, sizeof(next)))
      cp_batch_flush_why(cp, "depth/stencil state");

   if (state) {
      g->depth_stencil_cso = next;
      static_assert((int)PIPE_FUNC_LESS == (int)CP_FUNC_LESS &&
                    (int)PIPE_FUNC_LEQUAL == (int)CP_FUNC_LEQUAL &&
                    (int)PIPE_FUNC_GREATER == (int)CP_FUNC_GREATER &&
                    (int)PIPE_FUNC_GEQUAL == (int)CP_FUNC_GEQUAL,
                    "the driver's compare functions must match Gallium's");
      cp->depth_stencil = (struct cp_depth_state) {
         .depth_enabled = next.depth_enabled,
         .depth_writemask = next.depth_writemask,
         .depth_func = next.depth_func,
      };
      if (cp->gpu_state) {
         cp->gpu_state->depth_test = cp->depth_stencil.depth_enabled;
         cp->gpu_state->depth_func = cp->depth_stencil.depth_func;
         cp->gpu_state->depth_write = cp->depth_stencil.depth_writemask;
         cp->gpu_state->depth_key_invert = cp->depth_stencil.depth_enabled &&
            (cp->depth_stencil.depth_func == CP_FUNC_GREATER ||
             cp->depth_stencil.depth_func == CP_FUNC_GEQUAL);
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
   struct cp_gallium *g = (struct cp_gallium *)ctx;
   struct cp_context *cp = cp_ctx(ctx);
   if (state) {
      struct cp_vertex_elements_state *ve = (struct cp_vertex_elements_state *)state;
      if (ve->num_elements != cp->num_vertex_elements ||
          ve->stride != cp->vertex_stride ||
          memcmp(g->vertex_elements, ve->elements,
                 ve->num_elements * sizeof(struct pipe_vertex_element)))
         cp_batch_flush_defer_why(cp, "vertex elements");
      memcpy(g->vertex_elements, ve->elements, ve->num_elements * sizeof(struct pipe_vertex_element));
      cp->num_vertex_elements = ve->num_elements;
      cp->vertex_stride = ve->stride;

      /* The format's consequences, worked out once. The fetch kernel reads
       * these per draw and they do not vary with one. */
      memset(cp->velem, 0, sizeof(cp->velem));
      for (unsigned e = 0; e < ve->num_elements && e < 16; e++) {
         const struct pipe_vertex_element *el = &ve->elements[e];
         uint32_t nr_chan, chan_bytes, swizzle;
         enum cp_vf_conv conv =
            cp_vertex_format(el->src_format, &nr_chan, &chan_bytes, &swizzle);
         cp->velem[e] = (struct cp_vertex_elem) {
            .vertex_buffer_index = el->vertex_buffer_index,
            .src_offset = el->src_offset,
            .src_stride = el->src_stride,
            .instance_divisor = el->instance_divisor,
            .attr_size = util_format_get_blocksize(el->src_format),
            .nr_chan = nr_chan,
            .chan_bytes = chan_bytes,
            .swizzle = swizzle,
            .conv = conv,
            .fill_w = cp_vertex_fill_w(el->src_format, conv),
         };
      }

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
   /* A pass-episode segment may still name this state; render it first. */
   cp_batch_flush(cp_ctx(ctx));
   FREE(state);  /* frees cp_vertex_elements_state */
}

static void *
cp_create_fs_state(struct pipe_context *ctx,
                   const struct pipe_shader_state *state)
{
   struct cp_context *cp = cp_ctx(ctx);
   if (state->type != PIPE_SHADER_IR_NIR)
      return MALLOC(1);

   struct nir_shader *nir = (struct nir_shader *)state->ir.nir;

   if (cp_debug->dump_nir) {
      fprintf(stderr, "=== FS NIR ===\n");
      nir_print_shader(nir, stderr);
   }

   /* Lower FS I/O */
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out,
                cp_type_size_vec4, nir_lower_io_lower_64bit_to_32);

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(nir,
      cp->screen->sm_major, cp->screen->sm_minor,
      cp->screen->kernels.sampler_ptx, cp->screen->kernels.fs_helper_ptx);
   if (!bin)
      bin = CALLOC_STRUCT(cp_shader_binary);
   return bin;
}

static void
cp_bind_fs_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = cp_ctx(ctx);
   if (cp->fs_shader != (struct cp_shader_binary *)state)
      cp_batch_flush_defer_why(cp, "fragment shader");
   cp->fs_shader = (struct cp_shader_binary *)state;
}

static void
cp_delete_fs_state(struct pipe_context *ctx, void *state)
{
   /* A pass-episode segment may still name this shader; render it first. */
   cp_batch_flush(cp_ctx(ctx));
   cp_shader_binary_destroy((struct cp_shader_binary *)state);
}

static void *
cp_create_vs_state(struct pipe_context *ctx,
                   const struct pipe_shader_state *state)
{
   struct cp_context *cp = cp_ctx(ctx);
   if (state->type != PIPE_SHADER_IR_NIR)
      return MALLOC(1);

   struct nir_shader *nir = (struct nir_shader *)state->ir.nir;

   if (cp_debug->dump_nir) {
      fprintf(stderr, "=== VS NIR ===\n");
      nir_print_shader(nir, stderr);
   }

   /* Lower I/O derefs to explicit load_input/store_output */
   nir_lower_io(nir, nir_var_shader_in | nir_var_shader_out,
                cp_type_size_vec4, nir_lower_io_lower_64bit_to_32);

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(nir,
      cp->screen->sm_major, cp->screen->sm_minor,
      cp->screen->kernels.sampler_ptx, NULL);
   if (!bin)
      bin = CALLOC_STRUCT(cp_shader_binary);
   return bin;
}

static void
cp_bind_vs_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = cp_ctx(ctx);
   if (cp->vs_shader != (struct cp_shader_binary *)state) {
      /* Resolved once: this runs on every draw of every sample. A scene that
       * builds one pipeline per material — gltfscenerendering — rebinds a
       * distinct binary here between every draw, which ends the batch before
       * the key is ever consulted, and that is invisible from the key's own
       * diagnostics. */
      if (cp_debug->debug_batchdiff)
         fprintf(stderr, "cudapipe: batchdiff vs bind %p -> %p\n",
                 (void *)cp->vs_shader, state);
      cp_batch_flush_defer_why(cp, "vertex shader");
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
   /* A pass-episode segment may still name this shader; render it first. */
   cp_batch_flush(cp_ctx(ctx));
   FREE(state);
}

static void *
cp_create_compute_state(struct pipe_context *ctx,
                        const struct pipe_compute_state *state)
{
   struct cp_context *cp = cp_ctx(ctx);
   if (state->ir_type != PIPE_SHADER_IR_NIR)
      return NULL;

   cuCtxSetCurrent(cp->screen->cuda_ctx);

   struct nir_shader *nir = (struct nir_shader *)state->prog;
   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(nir,
      cp->screen->sm_major, cp->screen->sm_minor,
      cp->screen->kernels.sampler_ptx, NULL);
   if (!bin) {
      /* Return empty binary so lavapipe doesn't get NULL */
      bin = CALLOC_STRUCT(cp_shader_binary);
   }
   return bin;
}

static void
cp_bind_compute_state(struct pipe_context *ctx, void *state)
{
   struct cp_context *cp = cp_ctx(ctx);
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
   if (cp_debug->debug_tex) {
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
   struct cp_context *cp = cp_ctx(ctx);
   if (cp_debug->debug_tex)
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
   }

   if (start + count > cp->num_tex_objects)
      cp->num_tex_objects = start + count;
}

static void
cp_set_constant_buffer(struct pipe_context *ctx, mesa_shader_stage shader,
                       uint index,
                       const struct pipe_constant_buffer *buf)
{
   struct cp_context *cp = cp_ctx(ctx);
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
    * A pending batch survives any uniform binding now. The vertex and the
    * fragment stage both carry a per-draw table of pointers snapshotted at
    * cp_batch_record() time, so nothing the batch will read is live state; a
    * compute binding no draw reads. The one write that could still corrupt a
    * recorded row — rebinding a user_copy buffer, whose managed shadow is
    * reused in place — cannot reach one, because cp_batch_structural()
    * refuses to record a draw while any user_copy binding is live.
    */
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
   struct cp_context *cp = cp_ctx(ctx);

   /* A pending batch survives this: it snapshotted one row of resolved
    * per-element bases per draw at cp_batch_record() time, so nothing it will
    * read is live state. A count change breaks the key on its own. */

   for (unsigned i = 0; i < count; i++) {
      if (buffers) {
         void *data = buffers[i].buffer.resource
            ? cp_resource_data(cp_resource(buffers[i].buffer.resource)) : NULL;
         cp->vb_base[i] = data
            ? (uint64_t)(uintptr_t)data + buffers[i].buffer_offset : 0;
         if (cp->gpu_state)
            cp->gpu_state->vb_bases[i] = cp->vb_base[i];
      } else {
         cp->vb_base[i] = 0;
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
   struct cp_context *cp = cp_ctx(ctx);
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
   if (cp_debug->debug_tex)
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
   struct cp_context *cp = cp_ctx(ctx);
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

         if (cp_debug->debug_tex)
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
   struct cp_gallium *g = CALLOC_STRUCT(cp_gallium);
   if (!g)
      return NULL;
   struct cp_context *ctx = &g->cp;

   if (!cp_context_init(ctx, &cp_screen(screen)->dev)) {
      FREE(g);
      return NULL;
   }
   g->base.screen = screen;
   g->base.priv = priv;

   g->base.destroy = cp_destroy_context;

   g->base.draw_vbo = cp_draw_vbo;
   g->base.launch_grid = cp_launch_grid;
   g->base.flush = cp_flush;

   g->base.create_blend_state = cp_create_blend_state;
   g->base.bind_blend_state = cp_bind_blend_state;
   g->base.delete_blend_state = cp_delete_blend_state;

   g->base.create_rasterizer_state = cp_create_rasterizer_state;
   g->base.bind_rasterizer_state = cp_bind_rasterizer_state;
   g->base.delete_rasterizer_state = cp_delete_rasterizer_state;

   g->base.create_depth_stencil_alpha_state = cp_create_depth_stencil_alpha_state;
   g->base.bind_depth_stencil_alpha_state = cp_bind_depth_stencil_alpha_state;
   g->base.delete_depth_stencil_alpha_state = cp_delete_depth_stencil_alpha_state;

   g->base.create_vertex_elements_state = cp_create_vertex_elements_state;
   g->base.bind_vertex_elements_state = cp_bind_vertex_elements_state;
   g->base.delete_vertex_elements_state = cp_delete_vertex_elements_state;

   g->base.create_fs_state = cp_create_fs_state;
   g->base.bind_fs_state = cp_bind_fs_state;
   g->base.delete_fs_state = cp_delete_fs_state;

   g->base.create_vs_state = cp_create_vs_state;
   g->base.bind_vs_state = cp_bind_vs_state;
   g->base.delete_vs_state = cp_delete_vs_state;
   g->base.bind_gs_state = cp_bind_gs_state;
   g->base.bind_tcs_state = cp_bind_tcs_state;
   g->base.bind_tes_state = cp_bind_tes_state;

   g->base.create_compute_state = cp_create_compute_state;
   g->base.bind_compute_state = cp_bind_compute_state;
   g->base.delete_compute_state = cp_delete_compute_state;

   g->base.create_sampler_state = cp_create_sampler_state;
   g->base.bind_sampler_states = cp_bind_sampler_states;
   g->base.delete_sampler_state = cp_delete_sampler_state;

   g->base.create_sampler_view = cp_create_sampler_view;
   g->base.sampler_view_destroy = cp_sampler_view_destroy;
   g->base.set_sampler_views = cp_set_sampler_views;

   /*
    * pipe_resource_release() calls this through the context, not the screen,
    * and a driver that leaves it null gets a jump to address zero when
    * lavapipe tears its upload manager down. Four samples were dying there
    * after rendering correctly. llvmpipe uses the same default.
    */
   g->base.resource_release = u_default_resource_release;
   g->base.set_framebuffer_state = cp_set_framebuffer_state;
   g->base.set_viewport_states = cp_set_viewport_states;
   g->base.set_scissor_states = cp_set_scissor_states;
   g->base.set_constant_buffer = cp_set_constant_buffer;
   g->base.set_vertex_buffers = cp_set_vertex_buffers;
   g->base.set_shader_buffers = cp_set_shader_buffers;
   g->base.set_shader_images = cp_set_shader_images;
   g->base.set_blend_color = cp_set_blend_color;
   g->base.set_stencil_ref = cp_set_stencil_ref;
   g->base.set_sample_mask = cp_set_sample_mask;
   g->base.set_clip_state = cp_set_clip_state;
   g->base.set_polygon_stipple = cp_set_polygon_stipple;
   g->base.buffer_subdata = cp_buffer_subdata;
   g->base.set_sample_locations = cp_set_sample_locations;
   g->base.set_min_samples = cp_set_min_samples;
   g->base.render_condition = cp_render_condition;
   g->base.create_query = cp_create_query;
   g->base.destroy_query = cp_destroy_query;
   g->base.begin_query = cp_begin_query;
   g->base.end_query = cp_end_query;
   g->base.get_query_result = cp_get_query_result;
   g->base.get_query_result_resource = cp_get_query_result_resource;
   g->base.create_texture_handle = cp_create_texture_handle;
   g->base.create_image_handle = cp_create_image_handle;
   g->base.delete_texture_handle = cp_delete_texture_handle;
   g->base.delete_image_handle = cp_delete_image_handle;

   g->base.stream_uploader = u_upload_create_default(&g->base);
   g->base.const_uploader = g->base.stream_uploader;

   cudapipe_init_context_resource_funcs(&g->base);


   return &g->base;
}
