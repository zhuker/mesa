#include "cp_context.h"
#include "cp_screen.h"
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
static void cp_scratch_reset(struct cp_context *cp);

static void
cp_destroy_context(struct pipe_context *ctx)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (cp->visbuf)
      cuMemFree(cp->visbuf);
   if (cp->depthbuf)
      cuMemFree(cp->depthbuf);
   if (cp->reject)
      cuMemFree(cp->reject);
   if (cp->resolved)
      cuMemFree(cp->resolved);
   if (cp->rast_nontrivial)
      cuMemFree(cp->rast_nontrivial);
   if (cp->rast_nontrivial_count)
      cuMemFree(cp->rast_nontrivial_count);
   if (cp->rast_huge_tiles)
      cuMemFree(cp->rast_huge_tiles);
   if (cp->rast_huge_count)
      cuMemFree(cp->rast_huge_count);
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
   util_copy_framebuffer_state(&cp->framebuffer, state);

   /* Reallocate the visibility and depth buffers if the size changed */
   unsigned w = state->width, h = state->height;
   if (w != cp->visbuf_w || h != cp->visbuf_h) {
      if (cp->visbuf)
         cuMemFree(cp->visbuf);
      if (cp->depthbuf)
         cuMemFree(cp->depthbuf);
      if (cp->reject)
         cuMemFree(cp->reject);
      if (cp->resolved)
         cuMemFree(cp->resolved);
      cp->visbuf = 0;
      cp->depthbuf = 0;
      cp->reject = 0;
      cp->resolved = 0;
      cp->visbuf_w = cp->depthbuf_w = w;
      cp->visbuf_h = cp->depthbuf_h = h;
      if (w > 0 && h > 0) {
         cuCtxSetCurrent(cp->screen->cuda_ctx);
         CUresult e1 = cuMemAlloc(&cp->visbuf, (size_t)w * h * sizeof(uint64_t));
         CUresult e2 = cuMemAlloc(&cp->depthbuf, (size_t)w * h * sizeof(uint32_t));
         cuMemAlloc(&cp->reject,
                    (size_t)w * h * CP_DISCARD_LAYERS * sizeof(uint32_t));
         cuMemAlloc(&cp->resolved, (size_t)w * h);
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
   cuMemsetD32(cp->depthbuf, value, count);
   cp->depthbuf_cleared = true;
}

static void
cp_set_viewport_states(struct pipe_context *ctx, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *viewports)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (num_viewports > 0) {
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
   if (num_scissors > 0)
      cp->scissor = scissors[0];
}

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

static void
cp_scratch_begin(struct cp_context *cp)
{
   /* Cap memory growth: sync and reclaim after a few arena expansions.
    * With each expansion doubling (1MB→2→4→8→16→32), 5 expansions = 32MB
    * which is enough to pipeline many draws without exhausting GPU memory. */
   if (cp->scratch.num_overflow >= 5) {
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
}

/*
 * Per-stage timing for a draw, printed under CUDAPIPE_DEBUG_TIME.
 *
 * Every stage already synchronises, so wall clock around each one is an honest
 * measure of where a draw's time goes — which is worth knowing before
 * optimising anything.
 */
struct cp_draw_timing {
   double assemble_ms;
   double vertex_ms;
   double rasterize_ms;
   double interpolate_ms;
   double fragment_ms;
   double writeback_ms;
};

static bool
cp_timing_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("CUDAPIPE_DEBUG_TIME") ? 1 : 0;
   return enabled;
}

static double
cp_now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* Returns the elapsed time since *since and resets it, so stages can be timed
 * one after another without repeating the bookkeeping. */
static double
cp_lap(double *since)
{
   if (!cp_timing_enabled())
      return 0.0;
   double now = cp_now_ms();
   double elapsed = now - *since;
   *since = now;
   return elapsed;
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
 * The value a vertex attribute's fourth component reads as when the format
 * doesn't supply one. Vulkan defines the missing components of a vertex
 * attribute as (0, 0, 0, 1), and the zero-filled slot already covers y and z.
 * Returns 0 when the format supplies all four components and nothing is due.
 */
static uint32_t
cp_vertex_fill_w(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   if (!desc || desc->nr_channels >= 4)
      return 0;

   if (desc->channel[0].type == UTIL_FORMAT_TYPE_FLOAT) {
      float one = 1.0f;
      uint32_t bits;
      memcpy(&bits, &one, 4);
      return bits;
   }

   /* Normalised formats also read as 1.0, and this driver hands the shader
    * floats for them, so only the plain integer formats want an integer 1. */
   if (desc->channel[0].normalized) {
      float one = 1.0f;
      uint32_t bits;
      memcpy(&bits, &one, 4);
      return bits;
   }

   return 1;
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
                   unsigned reject_pass,
                   struct cp_draw_timing *timing)
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
   unsigned max_pixels = ALIGN_POT(w * h, 256);
   unsigned fs_in_stride = MAX2(num_fs_inputs, 1u) * 16;
   unsigned fs_out_stride = MAX2(fs->nir_num_outputs, 1u) * 16;

   unsigned fs_deriv_stride = MAX2(num_fs_inputs, 1u) * 16;

   CUdeviceptr pixel_list = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, max_pixels * 4);
   CUdeviceptr counter = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 4);
   CUdeviceptr fs_in = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, (size_t)max_pixels * fs_in_stride);
   CUdeviceptr fs_out = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, (size_t)max_pixels * fs_out_stride);
   CUdeviceptr fs_deriv = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, (size_t)max_pixels * fs_deriv_stride);
   CUdeviceptr frag_coord = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, (size_t)max_pixels * 16);
   /* One byte per shaded pixel, set by `discard` in the fragment shader. */
   CUdeviceptr discard_mask = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, max_pixels);
   CUdeviceptr fs_args_dev = 0, count_dev = 0, stride_dev = 0;

   if (!pixel_list || !counter || !fs_in || !fs_out || !fs_deriv || !frag_coord ||
       !discard_mask)
      return;

   cuMemsetD32(counter, 0, 1);
   cuMemsetD8(discard_mask, 0, max_pixels);

   struct cp_fs_interp_args interp = {
      .visbuf = visbuf,
      .positions = positions,
      .vs_out = vs_output_buf,
      .pixel_list = pixel_list,
      .counter = counter,
      .fs_in = fs_in,
      .frag_coord = frag_coord,
      .fs_deriv = fs_deriv,
      .width = w, .height = h,
      .vs_out_stride = num_vs_outputs * 16,
      .fs_in_stride = fs_in_stride,
      .num_fs_inputs = num_fs_inputs,
      .max_pixels = max_pixels,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
   };

   /* Match each fragment shader input to the vertex shader output carrying the
    * same varying location. */
   for (unsigned i = 0; i < num_fs_inputs; i++) {
      interp.input_vs_slot[i] = -1;
      unsigned location = fs->in_location[i];
      if (location == VARYING_SLOT_MAX)
         continue;
      for (unsigned o = 0; o < num_vs_outputs && o < CP_MAX_IO_SLOTS; o++) {
         if (cp->vs_shader->out_location[o] == location) {
            interp.input_vs_slot[i] = (int32_t)o;
            break;
         }
      }
   }

   double mark = cp_timing_enabled() ? cp_now_ms() : 0.0;

   void *interp_params[] = { &interp };
   CUresult interp_err = cuLaunchKernel(screen->kernels.fs_interpolate,
                                        (w * h + 255) / 256, 1, 1, 256, 1, 1,
                                        0, NULL, interp_params, NULL);
   if (interp_err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: fs_interpolate launch failed (%d)\n", interp_err);
      return;
   }
   timing->interpolate_ms = cp_lap(&mark);

   /* Launch FS and writeback over max_pixels — each kernel reads the actual
    * pixel count from the counter (device-visible managed memory) and exits
    * early for threads beyond it. This avoids a sync just to read the count. */
   unsigned num_pixels = max_pixels;

   /* The shader reads its arguments through the same pointer-array ABI the
    * compute path uses; see cp_launch_grid(). */
   fs_args_dev = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 64 * sizeof(void *));
   stride_dev = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 4);
   if (!fs_args_dev || !stride_dev)
      return;

   uint32_t fs_stride_host = fs_in_stride;
   *(uint32_t*)(uintptr_t)stride_dev = fs_stride_host;

   void *fs_args_host[64] = {0};
   fs_args_host[0] = (void *)(uintptr_t)counter;
   fs_args_host[2] = (void *)(uintptr_t)fs_in;
   fs_args_host[3] = (void *)(uintptr_t)stride_dev;
   fs_args_host[4] = (void *)(uintptr_t)fs_out;
   fs_args_host[6] = (void *)(uintptr_t)frag_coord;
   fs_args_host[CP_ARG_SLOT_DISCARD] = (void *)(uintptr_t)discard_mask;
   for (unsigned i = 0; i < cp->num_fs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
      fs_args_host[18 + i] = cp->fs_ubos[i].buffer;

   memcpy((void*)(uintptr_t)fs_args_dev, fs_args_host, 64 * sizeof(void*));

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

   /* Hand the linked sampler the state it reads through module globals: the
    * sampler table and the varying derivatives it needs for mip selection. */
   {
      CUdeviceptr sym;
      size_t sym_size;
      if (cp->sampler_table &&
          cuModuleGetGlobal(&sym, &sym_size, fs->module,
                            "cp_sampler_table") == CUDA_SUCCESS) {
         uint64_t addr = (uint64_t)cp->sampler_table;
         cuMemcpyHtoD(sym, &addr, sizeof(addr));
      }
      if (cuModuleGetGlobal(&sym, &sym_size, fs->module,
                            "cp_fs_deriv") == CUDA_SUCCESS) {
         uint64_t addr = (uint64_t)fs_deriv;
         cuMemcpyHtoD(sym, &addr, sizeof(addr));
      }
      if (cuModuleGetGlobal(&sym, &sym_size, fs->module,
                            "cp_fs_deriv_stride") == CUDA_SUCCESS)
         cuMemcpyHtoD(sym, &fs_deriv_stride, sizeof(fs_deriv_stride));
   }

   void *fs_arg_ptr = (void *)(uintptr_t)fs_args_dev;
   void *fs_params[] = { &fs_arg_ptr };
   CUresult fs_err = cuLaunchKernel(fs->kernel, (num_pixels + 255) / 256, 1, 1,
                                    256, 1, 1, 0, NULL, fs_params, NULL);
   if (fs_err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: fragment shader launch failed (%d)\n", fs_err);
      return;
   }
   timing->fragment_ms = cp_lap(&mark);

   const struct pipe_rt_blend_state *rt = &cp->blend_state.rt[0];
   struct cp_fs_writeback_args wb = {
      .pixel_list = pixel_list,
      .fs_out = fs_out,
      .color_out = (uint64_t)(uintptr_t)color_data,
      .visbuf = visbuf,
      .depthbuf = cp->depthbuf,
      .pixel_counter = counter,
      .discard_mask = discard_mask,
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
      .blend_enable = rt->blend_enable,
      .rgb_src_factor = rt->rgb_src_factor,
      .rgb_dst_factor = rt->rgb_dst_factor,
      .rgb_func = rt->rgb_func,
      .alpha_src_factor = rt->alpha_src_factor,
      .alpha_dst_factor = rt->alpha_dst_factor,
      .alpha_func = rt->alpha_func,
      .colormask = rt->colormask ? rt->colormask : 0xF,
   };

   void *wb_params[] = { &wb };
   cuLaunchKernel(screen->kernels.fs_writeback,
                  (num_pixels + 255) / 256, 1, 1, 256, 1, 1,
                  0, NULL, wb_params, NULL);
   timing->writeback_ms = cp_lap(&mark);

   if (getenv("CUDAPIPE_DEBUG_DISCARD")) {
      cuCtxSynchronize();
      unsigned covered = *(const uint32_t *)(uintptr_t)counter;
      const unsigned char *dm = (const unsigned char *)(uintptr_t)discard_mask;
      unsigned nd = 0;
      for (unsigned k = 0; k < covered && k < max_pixels; k++)
         nd += dm[k] ? 1u : 0u;
      fprintf(stderr, "  pass %u: covered %u, discarded %u\n",
              reject_pass, covered, nd);
   }

   if (getenv("CUDAPIPE_DEBUG_DRAW"))
      fprintf(stderr, "  shaded %u pixels (%u fs inputs, %u tris) "
              "blend=%u src=%u dst=%u mask=0x%x\n",
              num_pixels, num_fs_inputs, num_triangles,
              rt->blend_enable, rt->rgb_src_factor, rt->rgb_dst_factor,
              wb.colormask);

   if (getenv("CUDAPIPE_DEBUG_FS")) {
      const float *vs_out = (const float *)(uintptr_t)vs_output_buf;
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
      const uint32_t *plist = (const uint32_t *)(uintptr_t)pixel_list;
      const float *fin = (const float *)(uintptr_t)fs_in;
      const float *fout = (const float *)(uintptr_t)fs_out;
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
   }

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
   struct pipe_framebuffer_state *fb = &cp->framebuffer;

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
   cuMemsetD32(visbuf, 0xFFFFFFFF, (size_t)w * h * 2);

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

   struct cp_rasterize_args rast_args = {
      .framebuffer = visbuf,
      .color_buffer = (uint64_t)(uintptr_t)color_data,
      .width = w,
      .height = h,
      .num_triangles = num_triangles,
      .num_varyings = 0,
      .vp_x = vp_x, .vp_y = vp_y,
      .vp_w = vp_w, .vp_h = vp_h,
      .vp_near = 0.0f, .vp_far = 1.0f,
      .vp_scale_x = vp_scale_x, .vp_scale_y = vp_scale_y,
      .vp_trans_x = vp_trans_x, .vp_trans_y = vp_trans_y,
      .cull_mode = 0,
      .front_face = 0,
      .depthbuf = cp->depthbuf,
      .depth_test = cp->depth_stencil.depth_enabled,
      .depth_func = cp->depth_stencil.depth_func,
      .depth_key_invert = cp->depth_stencil.depth_enabled &&
         (cp->depth_stencil.depth_func == PIPE_FUNC_GREATER ||
          cp->depth_stencil.depth_func == PIPE_FUNC_GEQUAL),
   };

   struct cp_draw_timing timing = {0};
   double mark = cp_timing_enabled() ? cp_now_ms() : 0.0;

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

   /* For TRIANGLE_LIST with a VS and single instance, the vertex fetch kernel
    * indexes the IB directly on GPU — no CPU-side topology expansion needed. */
   bool skip_refs = has_vs && info->mode == MESA_PRIM_TRIANGLES &&
                    instance_count == 1 && num_draws == 1;
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
   timing.assemble_ms = cp_lap(&mark);

   /* If we have a compiled VS, run it to transform vertices.
    * The VS kernel reads from VB (args[2]) and writes positions+varyings (args[4]).
    * The output replaces packed_positions for the rasterizer. */
   /* VS execution */
   if (cp->vs_shader && cp->vs_shader->kernel) {
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

         vs_output_buf = (CUdeviceptr)(uintptr_t)
            cp_scratch_alloc(cp, (size_t)total_verts * out_stride);

         /* Build VS input buffer on GPU: the vertex fetch kernel gathers
          * attributes in parallel, one thread per assembled vertex. */
         unsigned vs_in_stride = cp->num_vertex_elements * 16;
         CUdeviceptr vs_input_buf = vs_in_stride
            ? (CUdeviceptr)(uintptr_t)cp_scratch_alloc(
                 cp, (size_t)total_verts * vs_in_stride)
            : 0;
         if (!vs_output_buf || (vs_in_stride && !vs_input_buf)) {
            FREE(refs);
            return;
         }

         /* Upload vertex_id and instance_id arrays for topology-expanded draws */
         CUdeviceptr vfetch_vid = 0, vfetch_iid = 0;
         bool need_refs_on_gpu = (info->mode != MESA_PRIM_TRIANGLES) || instance_count > 1;
         if (need_refs_on_gpu) {
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

         struct cp_vertex_fetch_args vf_args = {
            .output = vs_input_buf,
            .num_elements = cp->num_vertex_elements,
            .num_verts = total_verts,
            .vs_in_stride = vs_in_stride,
            .index_size = need_refs_on_gpu ? 0 : info->index_size,
            .first_vertex = indexed ? (unsigned)draws[0].index_bias : draws[0].start,
            .start_instance = info->start_instance,
            .vertex_ids = vfetch_vid,
            .instance_ids = vfetch_iid,
         };

         /* Set up index buffer for direct GPU indexing (triangle list only).
          * The fetch kernel indexes from the start of what it is given, so the
          * draw's first index has to be folded into the pointer — a glTF model
          * draws every primitive out of one shared buffer this way, and
          * ignoring it draws the first primitive over and over. */
         if (!need_refs_on_gpu && indexed && ib_base)
            vf_args.index_buffer = (uint64_t)(uintptr_t)ib_base +
                                   (uint64_t)draws[0].start * info->index_size;

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
            vf_args.elem_fill_w[e] = cp_vertex_fill_w(elem->src_format);
            vf_args.elem_instance_divisor[e] = elem->instance_divisor;
         }

         /* Nothing to gather when the shader declares no inputs. */
         if (vs_input_buf) {
            cuMemsetD8(vs_input_buf, 0, (size_t)total_verts * vs_in_stride);
            void *vf_params[] = { &vf_args };
            cuLaunchKernel(screen->kernels.vertex_fetch,
               (total_verts + 255) / 256, 1, 1, 256, 1, 1,
               0, NULL, vf_params, NULL);
         }

         stride = vs_in_stride;

         /* Dump what the GPU fetch actually gathered, which is the quickest way to
          * tell a bad attribute layout from a bad shader. Syncs, so debug only. */
         if (getenv("CUDAPIPE_DEBUG_VFETCH") && vs_input_buf) {
            cuCtxSynchronize();
            const float *in = (const float *)(uintptr_t)vs_input_buf;
            for (unsigned e = 0; e < cp->num_vertex_elements && e < 8; e++)
               fprintf(stderr, "  elem%u vb=%u off=%u stride=%u div=%u sz=%u\n",
                       e, cp->vertex_elements[e].vertex_buffer_index,
                       cp->vertex_elements[e].src_offset,
                       cp->vertex_elements[e].src_stride,
                       cp->vertex_elements[e].instance_divisor,
                       vf_args.elem_attr_size[e]);
            for (unsigned v = 0; v < 2 && v < total_verts; v++) {
               fprintf(stderr, "  vfetch v%u:", v);
               for (unsigned e = 0; e < cp->num_vertex_elements && e < 8; e++)
                  fprintf(stderr, " e%u=[%.3f %.3f %.3f]", e,
                          in[(v * vs_in_stride) / 4 + e * 4 + 0],
                          in[(v * vs_in_stride) / 4 + e * 4 + 1],
                          in[(v * vs_in_stride) / 4 + e * 4 + 2]);
               fprintf(stderr, "\n");
            }
         }

         /* Allocate device-side buffers for VS kernel args and metadata */
         CUdeviceptr vs_args_dev = (CUdeviceptr)(uintptr_t)
            cp_scratch_alloc(cp, 64 * sizeof(void *));
         CUdeviceptr stride_dev = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 4);
         CUdeviceptr vcount_dev = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 4);
         CUdeviceptr vid_buf = 0, iid_buf = 0;
         CUdeviceptr draw_params = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 3 * 4);

         if (skip_refs) {
            /* TRIANGLE_LIST fast path: VS reads vertex IDs from the index buffer
             * directly (or sequentially for non-indexed). No CPU loop needed. */
            if (indexed && ib_base) {
               /* Point vid at the IB data — VS reads IB[thread_id] as vertex_id.
                * Note: this only works for 32-bit indices. For 16-bit we still
                * need a conversion (the VS expects uint32 per vertex). */
               if (info->index_size == 4) {
                  vid_buf = (CUdeviceptr)(uintptr_t)ib_base +
                            (size_t)draws[0].start * 4;
               } else {
                  /* 16-bit IB: allocate and let the vertex_fetch kernel handle it */
                  vid_buf = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, total_verts * 4);
               }
            } else if (cp->vs_shader->reads_vertex_id) {
               /* Non-indexed: vertex_id is just thread_id + first, but the
                * shader reads it out of this array rather than computing it,
                * so it has to exist. Only materialised for the shaders that
                * ask — a fullscreen pass building its corners from
                * gl_VertexIndex is the usual one. */
               vid_buf = (CUdeviceptr)(uintptr_t)
                  cp_scratch_alloc(cp, (size_t)total_verts * 4);
               if (vid_buf) {
                  uint32_t *ids = (uint32_t *)(uintptr_t)vid_buf;
                  for (unsigned v = 0; v < total_verts; v++)
                     ids[v] = draws[0].start + v;
               }
            } else {
               vid_buf = 0;
            }
            /* instance_id = 0 for all vertices (single instance) */
            iid_buf = 0;
         } else {
            vid_buf = (CUdeviceptr)(uintptr_t)
               cp_scratch_alloc(cp, (size_t)total_verts * 4);
            iid_buf = (CUdeviceptr)(uintptr_t)
               cp_scratch_alloc(cp, (size_t)total_verts * 4);
            if (!vid_buf || !iid_buf) { FREE(refs); return; }

            uint32_t *vid_host = malloc((size_t)total_verts * 4);
            uint32_t *iid_host = malloc((size_t)total_verts * 4);
            if (!vid_host || !iid_host) {
               free(vid_host); free(iid_host); FREE(refs); return;
            }
            for (unsigned v = 0; v < total_verts; v++) {
               vid_host[v] = refs[v].vertex;
               iid_host[v] = refs[v].instance;
            }
            memcpy((void*)(uintptr_t)vid_buf, vid_host, (size_t)total_verts * 4);
            memcpy((void*)(uintptr_t)iid_buf, iid_host, (size_t)total_verts * 4);
            free(vid_host);
            free(iid_host);
         }

         if (!vs_args_dev || !stride_dev || !vcount_dev || !draw_params) {
            FREE(refs);
            return;
         }

         /* Build the VS args table on the stack, upload in one shot */
         void *vs_args_host[64] = {0};
         uint32_t stride_host = stride;
         uint32_t vcount_host = total_verts;
         uint32_t draw_params_host[3] = {
            indexed ? (uint32_t)draws[0].index_bias : draws[0].start,
            info->start_instance,
            drawid_offset,
         };

         *(uint32_t*)(uintptr_t)stride_dev = stride_host;
         *(uint32_t*)(uintptr_t)vcount_dev = vcount_host;
         memcpy((void*)(uintptr_t)draw_params, draw_params_host, 12);

         vs_args_host[0] = (void*)(uintptr_t)vcount_dev;
         vs_args_host[1] = NULL;
         vs_args_host[2] = (void*)(uintptr_t)vs_input_buf;
         vs_args_host[3] = (void*)(uintptr_t)stride_dev;
         vs_args_host[4] = (void*)(uintptr_t)vs_output_buf;
         vs_args_host[5] = (void*)(uintptr_t)vid_buf;
         vs_args_host[6] = (void*)(uintptr_t)iid_buf;
         vs_args_host[7] = (void*)(uintptr_t)draw_params;

         for (unsigned i = 0; i < cp->num_vs_ubos && i < CP_MAX_CONST_BUFFERS; i++)
            vs_args_host[18 + i] = cp->vs_ubos[i].buffer;

         memcpy((void*)(uintptr_t)vs_args_dev, vs_args_host, 64 * sizeof(void*));

         void *vs_arg_ptr = (void*)(uintptr_t)vs_args_dev;
         void *vs_params[] = { &vs_arg_ptr };
         CUresult vs_err = cuLaunchKernel(cp->vs_shader->kernel,
            (total_verts + 255) / 256, 1, 1, 256, 1, 1,
            0, NULL, vs_params, NULL);

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
               /* Two planes turn one triangle into at most three. */
               unsigned max_clipped = num_triangles * 3;
               CUdeviceptr clipped = (CUdeviceptr)(uintptr_t)cp_scratch_alloc(
                  cp, (size_t)max_clipped * 3 * out_stride);
               CUdeviceptr clip_count =
                  (CUdeviceptr)(uintptr_t)cp_scratch_alloc(cp, 4);

               if (clipped && clip_count) {
                  cuMemsetD32(clip_count, 0, 1);

                  struct cp_clip_args clip = {
                     .vs_out = vs_output_buf,
                     .out = clipped,
                     .out_count = clip_count,
                     .num_triangles = num_triangles,
                     .num_slots = num_vs_outputs,
                     .max_triangles = max_clipped,
                  };
                  void *clip_params[] = { &clip };
                  CUresult clip_err = cuLaunchKernel(
                     screen->kernels.clip_triangles,
                     (num_triangles + 63) / 64, 1, 1, 64, 1, 1,
                     0, NULL, clip_params, NULL);

                  if (clip_err == CUDA_SUCCESS) {
                     vs_output_buf = clipped;
                     rast_args.positions = clipped;
                     rast_args.tri_count = clip_count;
                     rast_num_triangles = max_clipped;
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
   timing.vertex_ms = cp_lap(&mark);

   if (getenv("CUDAPIPE_DEBUG_DRAW")) {
      fprintf(stderr, "cudapipe: draw %u tris (%u instances), fb=%ux%u, "
              "vp=[%.0f,%.0f,%.0f,%.0f] stride=%u scale=[%.1f,%.1f] color=%p\n",
              num_triangles, instance_count, w, h, vp_x, vp_y, vp_w, vp_h,
              cp->vertex_stride,
              cp->viewport.scale[0], cp->viewport.scale[1], color_data);
      for (unsigned e = 0; e < cp->num_vertex_elements && e < 4; e++)
         fprintf(stderr, "  elem[%u]: offset=%u fmt=%u vb=%u\n", e,
                 cp->vertex_elements[e].src_offset, cp->vertex_elements[e].src_format,
                 cp->vertex_elements[e].vertex_buffer_index);
   }

   /* 3-stage adaptive rasterize — cuMemsetD32 is enqueued on the default
    * stream, so it serializes properly with the preceding draw's kernels. */
   cuMemsetD32(cp->rast_nontrivial_count, 0, 1);
   cuMemsetD32(cp->rast_huge_count, 0, 1);

   struct cp_rast_queues rast_queues = {
      .nontrivial = cp->rast_nontrivial,
      .nontrivial_count = cp->rast_nontrivial_count,
      .huge_tiles = cp->rast_huge_tiles,
      .huge_count = cp->rast_huge_count,
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
   unsigned passes = retry ? CP_DISCARD_LAYERS : 1;

   if (retry) {
      cuMemsetD8(cp->resolved, 0, (size_t)w * h);
      cuMemsetD32(cp->reject, 0xFFFFFFFF,
                  (size_t)w * h * CP_DISCARD_LAYERS);
      rast_args.reject = cp->reject;
      rast_args.resolved = cp->resolved;
      rast_args.reject_layers = CP_DISCARD_LAYERS;
   }

   for (unsigned pass = 0; pass < passes; pass++) {
      if (pass) {
         /* Each pass resolves visibility afresh, minus what has been rejected. */
         cuMemsetD32(visbuf, 0xFFFFFFFF, (size_t)w * h * 2);
         rast_args.reject_passes = pass;
      }
      cuMemsetD32(cp->rast_nontrivial_count, 0, 1);
      cuMemsetD32(cp->rast_huge_count, 0, 1);

      /* Stage 1: 1 thread per triangle (small rasterize in place, others queue) */
      void *s1_params[] = { &rast_args, &rast_queues };
      CUresult rast_err = cuLaunchKernel(screen->kernels.rasterize_stage1,
         (rast_num_triangles + 255) / 256, 1, 1, 256, 1, 1,
         0, NULL, s1_params, NULL);

      /* Stage 2: warp-cooperative, fixed grid self-bounding from counter */
      void *s2_params[] = { &rast_args, &rast_queues };
      cuLaunchKernel(screen->kernels.rasterize_stage2,
         512, 1, 1, 32, 1, 1,
         0, NULL, s2_params, NULL);

      /* Stage 3: block per tile, fixed grid self-bounding from counter */
      void *s3_params[] = { &rast_args, &rast_queues };
      cuLaunchKernel(screen->kernels.rasterize_stage3,
         2048, 1, 1, 64, 1, 1,
         0, NULL, s3_params, NULL);

      if (rast_err != CUDA_SUCCESS && getenv("CUDAPIPE_DEBUG_DRAW"))
         fprintf(stderr, "  rasterize launch failed: %d\n", rast_err);
      timing.rasterize_ms += cp_lap(&mark);

      /* Shade every covered pixel by running the fragment shader on the GPU:
       * interpolate its inputs, launch it, then blend its output into the
       * attachment. */
      if (color_data)
         cp_shade_fragments(cp, info, visbuf, rast_args.positions, vs_output_buf,
                            num_triangles, w, h, color_data,
                            vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y,
                            retry ? cp->reject : 0,
                            retry ? cp->resolved : 0,
                            pass, &timing);
   }

   if (cp_timing_enabled()) {
      double total = timing.assemble_ms + timing.vertex_ms +
                     timing.rasterize_ms + timing.interpolate_ms +
                     timing.fragment_ms + timing.writeback_ms;
      fprintf(stderr,
              "cudapipe: %5u tris  assemble %6.2f  vertex %6.2f  raster %6.2f  "
              "interp %6.2f  fragment %6.2f  writeback %6.2f  total %6.2f ms\n",
              num_triangles, timing.assemble_ms, timing.vertex_ms,
              timing.rasterize_ms, timing.interpolate_ms, timing.fragment_ms,
              timing.writeback_ms, total);
   }


   /* No sync needed here: kernels in the default stream are serialized, and
    * the next draw's rasterizer reads the depth buffer on the GPU — not the
    * host. Sync only when the host must read GPU results (e.g., readback). */
   FREE(refs);
}

static void
cp_launch_grid(struct pipe_context *ctx, const struct pipe_grid_info *info)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   struct cp_shader_binary *bin = cp->compute_shader;
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

   CUdeviceptr args_dev = 0;
   CUdeviceptr grid_dev = 0;
   if (cuMemAllocManaged(&args_dev, 34 * sizeof(void *), CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS ||
       cuMemAllocManaged(&grid_dev, sizeof(grid_size), CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
      if (args_dev) cuMemFree(args_dev);
      if (grid_dev) cuMemFree(grid_dev);
      return;
   }
   memcpy((void*)(uintptr_t)grid_dev, grid_size, sizeof(grid_size));

   void *arg_ptrs_host[34] = {0};
   arg_ptrs_host[0] = (void *)(uintptr_t)grid_dev;
   arg_ptrs_host[1] = NULL;

   for (unsigned i = 0; i < CP_MAX_SHADER_BUFFERS; i++)
      arg_ptrs_host[2 + i] = cp->compute_ssbos[i].buffer;

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++)
      arg_ptrs_host[18 + i] = cp->compute_ubos[i].buffer;

   memcpy((void*)(uintptr_t)args_dev, arg_ptrs_host, 34 * sizeof(void*));

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
      bin->shared_size, NULL, kernel_params, NULL);

   if (err != CUDA_SUCCESS)
      fprintf(stderr, "cudapipe: cuLaunchKernel failed (%d) grid=[%u,%u,%u] block=[%u,%u,%u]\n",
              err, info->grid[0], info->grid[1], info->grid[2],
              info->block[0], info->block[1], info->block[2]);

   cuCtxSynchronize();
   cuMemFree(args_dev);
   cuMemFree(grid_dev);
}

static void
cp_flush(struct pipe_context *ctx, struct pipe_fence_handle **fence,
         unsigned flags)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   cuCtxSetCurrent(cp->screen->cuda_ctx);

   if (fence) {
      CUevent event;
      cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
      cuEventRecord(event, 0);
      *fence = (struct pipe_fence_handle *)(uintptr_t)event;
   }

   /* Sync and reclaim scratch — keeps memory bounded. The sync is cheap
    * here because the GPU is typically already caught up (99% utilized). */
   cuCtxSynchronize();
   cp_scratch_reset(cp);
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
   if (state) {
      cp->blend_state = *(struct pipe_blend_state *)state;
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
   if (state) {
      cp->depth_stencil = *(struct pipe_depth_stencil_alpha_state *)state;
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
      if (index + 1 > cp->num_##stage##_ubos)                           \
         cp->num_##stage##_ubos = index + 1;                            \
   } while (0)

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
      cuMemsetD8(cp->sampler_table, 0,
                 CP_MAX_SAMPLERS * sizeof(struct cp_sampler_info));
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

   /* Adaptive rasterizer queues — allocated once, reused across draws.
    * Counters are allocated as 256 bytes each (minimum practical GPU alloc)
    * and zeroed per draw via cuMemsetD32. */
   cuMemAlloc(&ctx->rast_nontrivial,
              (size_t)CP_MAX_NONTRIVIAL * sizeof(uint32_t));
   cuMemAlloc(&ctx->rast_nontrivial_count, 256);
   cuMemAlloc(&ctx->rast_huge_tiles,
              (size_t)CP_MAX_HUGE_TILES * sizeof(struct cp_tile_pair));
   cuMemAlloc(&ctx->rast_huge_count, 256);

   return &ctx->base;
}
