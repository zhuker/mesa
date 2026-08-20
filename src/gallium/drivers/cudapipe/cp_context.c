#include "cp_context.h"
#include "cp_screen.h"
#include "cp_draw_types.h"
#include "cp_kernels.h"
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
   cp->fb = (struct cp_fb_desc) {
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
   unsigned w = state->width, h = state->height;
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
static bool
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

/* Snapshot this draw's index range and vertex-stage bindings as the next row
 * of the batch's tables. */
static void
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
                type_size_vec4, nir_lower_io_lower_64bit_to_32);

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
                type_size_vec4, nir_lower_io_lower_64bit_to_32);

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
