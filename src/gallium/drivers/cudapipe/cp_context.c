#include "cp_context.h"
#include "cp_screen.h"
#include "cp_resource.h"
#include "nir_to_ptx/cp_nir_to_llvm.h"
#include "kernels/cp_rast_types.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_state.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_upload_mgr.h"
#include "util/u_framebuffer.h"
#include "compiler/shader_enums.h"

#include <string.h>
#include <math.h>
#include <cuda.h>

static void
cp_destroy_context(struct pipe_context *ctx)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   /* TODO: cuStreamDestroy */
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
}

static void
cp_set_viewport_states(struct pipe_context *ctx, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *viewports)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (num_viewports > 0)
      cp->viewport = viewports[0];
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
   if (!fb->nr_cbufs || !fb->cbufs[0].texture)
      return;
   if (num_draws == 0 || draws[0].count == 0)
      return;

   cuCtxSetCurrent(screen->cuda_ctx);

   /* For now: only handle triangle lists without index buffers as a simple case */
   unsigned vertex_count = draws[0].count;
   unsigned num_triangles = vertex_count / 3;
   if (num_triangles == 0)
      return;

   /* Get the color output surface */
   struct cp_resource *color_res = cp_resource(fb->cbufs[0].texture);
   void *color_data = cp_resource_data(color_res);
   if (!color_data)
      return;

   unsigned w = fb->width;
   unsigned h = fb->height;

   /* Allocate visibility buffer (temporary) */
   CUdeviceptr visbuf;
   cuMemAllocManaged(&visbuf, w * h * sizeof(uint64_t), CU_MEM_ATTACH_GLOBAL);

   /* Clear visbuf */
   uint32_t vw = w, vh = h;
   uint64_t visbuf_ptr = visbuf;
   void *cv_params[] = { &visbuf_ptr, &vw, &vh };
   cuLaunchKernel(screen->kernels.clear_visbuf,
      (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1,
      0, NULL, cv_params, NULL);

   /* For now: read vertex positions directly from the first bound vertex buffer.
    * Assume positions are at offset 0 as float4 (x,y,z,w).
    * TODO: proper VS execution with compiled vertex shader */
   /* Viewport: pass raw scale/translate for proper Vulkan Y-flip handling.
    * screen = ndc * scale + translate (where scale[1] is negative for Y-down) */
   float vp_w = fabsf(cp->viewport.scale[0]) * 2.0f;
   float vp_h = fabsf(cp->viewport.scale[1]) * 2.0f;
   float vp_x = cp->viewport.translate[0] - fabsf(cp->viewport.scale[0]);
   float vp_y = cp->viewport.translate[1] - fabsf(cp->viewport.scale[1]);

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
      .cull_mode = 0,
      .front_face = 0,
   };

   /* Extract positions from vertex buffer into a packed float4 array.
    * The VB is interleaved (stride != sizeof(float4)), so we copy out positions.
    * TODO: run compiled vertex shader instead of passthrough copy. */
   CUdeviceptr packed_positions = 0;
   if (cp->num_vertex_buffers > 0 && cp->vertex_buffers[0].buffer.resource) {
      struct cp_resource *vb_res = cp_resource(cp->vertex_buffers[0].buffer.resource);
      void *vb_data = cp_resource_data(vb_res);
      if (vb_data) {
         char *vb_start = (char *)vb_data + cp->vertex_buffers[0].buffer_offset;
         unsigned stride = cp->vertex_stride ? cp->vertex_stride : 16;

         /* Allocate packed positions (float4 per vertex) */
         cuMemAllocManaged(&packed_positions, vertex_count * 16, CU_MEM_ATTACH_GLOBAL);
         float *dst = (float *)(uintptr_t)packed_positions;

         for (unsigned v = 0; v < vertex_count; v++) {
            float *src_pos = (float *)(vb_start + v * stride);
            dst[v * 4 + 0] = src_pos[0];
            dst[v * 4 + 1] = src_pos[1];
            dst[v * 4 + 2] = src_pos[2];
            dst[v * 4 + 3] = src_pos[3];
         }
         rast_args.positions = packed_positions;
      }
   }

   if (rast_args.positions == 0) {
      cuMemFree(visbuf);
      return;
   }

   if (getenv("CUDAPIPE_DEBUG_DRAW")) {
      fprintf(stderr, "cudapipe: draw %u tris, fb=%ux%u, vp=[%.0f,%.0f,%.0f,%.0f] stride=%u\n",
              num_triangles, w, h, vp_x, vp_y, vp_w, vp_h, cp->vertex_stride);
      for (unsigned e = 0; e < cp->num_vertex_elements && e < 4; e++)
         fprintf(stderr, "  elem[%u]: offset=%u fmt=%u vb=%u\n", e,
                 cp->vertex_elements[e].src_offset, cp->vertex_elements[e].src_format,
                 cp->vertex_elements[e].vertex_buffer_index);
   }

   /* Rasterize */
   void *rast_params[] = { &rast_args };
   CUresult rast_err = cuLaunchKernel(screen->kernels.rasterize_triangles,
      (num_triangles + 255) / 256, 1, 1, 256, 1, 1,
      0, NULL, rast_params, NULL);
   if (rast_err != CUDA_SUCCESS && getenv("CUDAPIPE_DEBUG_DRAW"))
      fprintf(stderr, "  rasterize launch failed: %d\n", rast_err);

   cuCtxSynchronize();

   /* Resolve — interpolate vertex colors */
   struct cp_resolve_args resolve_args = {
      .visbuf = visbuf,
      .positions = packed_positions,
      .colors = 0,
      .color_out = (uint64_t)(uintptr_t)color_data,
      .width = w, .height = h,
      .vp_x = vp_x, .vp_y = vp_y, .vp_w = vp_w, .vp_h = vp_h,
      .color_stride = 16,
   };

   /* Extract per-vertex colors (float4 at offset 16 in each vertex) */
   CUdeviceptr packed_colors = 0;
   if (cp->num_vertex_buffers > 0 && cp->vertex_buffers[0].buffer.resource &&
       cp->num_vertex_elements >= 2) {
      struct cp_resource *vb_res = cp_resource(cp->vertex_buffers[0].buffer.resource);
      void *vb_data = cp_resource_data(vb_res);
      if (vb_data) {
         char *vb_start = (char *)vb_data + cp->vertex_buffers[0].buffer_offset;
         unsigned stride = cp->vertex_stride ? cp->vertex_stride : 16;
         unsigned color_offset = cp->vertex_elements[1].src_offset;

         cuMemAllocManaged(&packed_colors, vertex_count * 16, CU_MEM_ATTACH_GLOBAL);
         float *cdst = (float *)(uintptr_t)packed_colors;
         for (unsigned v = 0; v < vertex_count; v++) {
            float *src_color = (float *)(vb_start + v * stride + color_offset);
            cdst[v * 4 + 0] = src_color[0];
            cdst[v * 4 + 1] = src_color[1];
            cdst[v * 4 + 2] = src_color[2];
            cdst[v * 4 + 3] = src_color[3];
         }
         resolve_args.colors = packed_colors;
         if (getenv("CUDAPIPE_DEBUG_DRAW")) {
            float *c = (float*)(uintptr_t)packed_colors;
            fprintf(stderr, "  packed colors: v0=[%.2f,%.2f,%.2f,%.2f] v1=[%.2f,%.2f,%.2f,%.2f] v2=[%.2f,%.2f,%.2f,%.2f]\n",
                    c[0],c[1],c[2],c[3], c[4],c[5],c[6],c[7], c[8],c[9],c[10],c[11]);
            /* Raw VB color data */
            float *raw0 = (float*)(vb_start + 0*stride + color_offset);
            float *raw1 = (float*)(vb_start + 1*stride + color_offset);
            float *raw2 = (float*)(vb_start + 2*stride + color_offset);
            fprintf(stderr, "  raw colors: v0=[%.2f,%.2f,%.2f,%.2f] v1=[%.2f,%.2f,%.2f,%.2f] v2=[%.2f,%.2f,%.2f,%.2f]\n",
                    raw0[0],raw0[1],raw0[2],raw0[3], raw1[0],raw1[1],raw1[2],raw1[3], raw2[0],raw2[1],raw2[2],raw2[3]);
         }
      }
   }

   /* CPU-side resolve for now (GPU resolve has parameter issues) */
   {
      uint64_t *vis = (uint64_t *)(uintptr_t)visbuf;
      uint32_t *col = (uint32_t *)color_data;
      float *pos = (float *)(uintptr_t)packed_positions;
      float *colors_arr = packed_colors ? (float *)(uintptr_t)packed_colors : NULL;

      for (unsigned py = 0; py < h; py++) {
         for (unsigned px = 0; px < w; px++) {
            uint64_t entry = vis[py * w + px];
            if (entry == 0xFFFFFFFFFFFFFFFFULL)
               continue;
            uint32_t tri_id = (uint32_t)(entry & 0xFFFFFFFF);

            /* Re-fetch positions */
            float *v0p = pos + (tri_id*3+0)*4;
            float *v1p = pos + (tri_id*3+1)*4;
            float *v2p = pos + (tri_id*3+2)*4;

            float sx0 = (v0p[0]/v0p[3]*0.5f+0.5f)*vp_w+vp_x;
            float sy0 = (0.5f-v0p[1]/v0p[3]*0.5f)*vp_h+vp_y;
            float sx1 = (v1p[0]/v1p[3]*0.5f+0.5f)*vp_w+vp_x;
            float sy1 = (0.5f-v1p[1]/v1p[3]*0.5f)*vp_h+vp_y;
            float sx2 = (v2p[0]/v2p[3]*0.5f+0.5f)*vp_w+vp_x;
            float sy2 = (0.5f-v2p[1]/v2p[3]*0.5f)*vp_h+vp_y;

            float cx = (float)px + 0.5f;
            float cy = (float)py + 0.5f;
            float area = (sx1-sx0)*(sy2-sy0)-(sy1-sy0)*(sx2-sx0);
            if (area == 0) continue;
            float inv_a = 1.0f / area;
            float w0 = ((sx1-cx)*(sy2-cy)-(sy1-cy)*(sx2-cx)) * inv_a;
            float w1 = ((sx2-cx)*(sy0-cy)-(sy2-cy)*(sx0-cx)) * inv_a;
            float w2 = 1.0f - w0 - w1;

            float r=1,g=1,b=1,a=1;
            if (colors_arr) {
               float *c0=colors_arr+(tri_id*3+0)*4;
               float *c1=colors_arr+(tri_id*3+1)*4;
               float *c2=colors_arr+(tri_id*3+2)*4;
               r = w0*c0[0]+w1*c1[0]+w2*c2[0];
               g = w0*c0[1]+w1*c1[1]+w2*c2[1];
               b = w0*c0[2]+w1*c1[2]+w2*c2[2];
               a = w0*c0[3]+w1*c1[3]+w2*c2[3];
            }
            if(r<0)r=0; if(r>1)r=1;
            if(g<0)g=0; if(g>1)g=1;
            if(b<0)b=0; if(b>1)b=1;
            if(a<0)a=0; if(a>1)a=1;
            uint32_t ri=(uint32_t)(r*255+0.5f);
            uint32_t gi=(uint32_t)(g*255+0.5f);
            uint32_t bi=(uint32_t)(b*255+0.5f);
            uint32_t ai=(uint32_t)(a*255+0.5f);
            col[py*w+px] = ri|(gi<<8)|(bi<<16)|(ai<<24);
         }
      }
   }

   cuCtxSynchronize();
   cuMemFree(visbuf);
   if (packed_positions)
      cuMemFree(packed_positions);
   if (packed_colors)
      cuMemFree(packed_colors);
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

   CUdeviceptr args_dev;
   cuMemAllocManaged(&args_dev, 34 * sizeof(void *), CU_MEM_ATTACH_GLOBAL);
   void **arg_ptrs = (void **)(uintptr_t)args_dev;

   CUdeviceptr grid_dev;
   cuMemAllocManaged(&grid_dev, sizeof(grid_size), CU_MEM_ATTACH_GLOBAL);
   memcpy((void *)(uintptr_t)grid_dev, grid_size, sizeof(grid_size));

   arg_ptrs[0] = (void *)(uintptr_t)grid_dev;
   arg_ptrs[1] = NULL;

   for (unsigned i = 0; i < CP_MAX_SHADER_BUFFERS; i++)
      arg_ptrs[2 + i] = cp->compute_ssbos[i].buffer;

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++)
      arg_ptrs[18 + i] = cp->compute_ubos[i].buffer;

   void *args_ptr_val = (void *)(uintptr_t)args_dev;
   void *kernel_params[] = { &args_ptr_val };

   if (getenv("CUDAPIPE_DEBUG_LAUNCH")) {
      fprintf(stderr, "  args_dev=%p arg_ptrs[19]=%p (UBO[1])\n",
              (void*)(uintptr_t)args_dev, arg_ptrs[19]);
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
   /* TODO: cuStreamSynchronize + create CUevent fence */
   if (fence)
      *fence = NULL;
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
   /* TODO: Phase 4 - NIR -> PTX device function */
   return MALLOC(1);
}

static void
cp_bind_fs_state(struct pipe_context *ctx, void *state)
{
}

static void
cp_delete_fs_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

static void *
cp_create_vs_state(struct pipe_context *ctx,
                   const struct pipe_shader_state *state)
{
   /* TODO: Phase 4 - NIR -> PTX device function */
   return MALLOC(1);
}

static void
cp_bind_vs_state(struct pipe_context *ctx, void *state)
{
}

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
      cp->screen->sm_major, cp->screen->sm_minor);
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
}

static void
cp_set_constant_buffer(struct pipe_context *ctx, mesa_shader_stage shader,
                       uint index,
                       const struct pipe_constant_buffer *buf)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   if (shader != MESA_SHADER_COMPUTE || index >= CP_MAX_CONST_BUFFERS)
      return;
   if (buf && buf->buffer) {
      struct cp_resource *res = cp_resource(buf->buffer);
      cp->compute_ubos[index].buffer = (char *)cp_resource_data(res) + buf->buffer_offset;
      cp->compute_ubos[index].buffer_size = buf->buffer_size;
   } else if (buf && buf->user_buffer) {
      cp->compute_ubos[index].buffer = (void *)buf->user_buffer;
      cp->compute_ubos[index].buffer_size = buf->buffer_size;
   } else {
      cp->compute_ubos[index].buffer = NULL;
      cp->compute_ubos[index].buffer_size = 0;
   }
   if (index + 1 > cp->num_compute_ubos)
      cp->num_compute_ubos = index + 1;
}

static void
cp_set_vertex_buffers(struct pipe_context *ctx, unsigned count,
                      const struct pipe_vertex_buffer *buffers)
{
   struct cp_context *cp = (struct cp_context *)ctx;
   for (unsigned i = 0; i < count; i++) {
      if (buffers) {
         cp->vertex_buffers[i] = buffers[i];
      } else {
         memset(&cp->vertex_buffers[i], 0, sizeof(cp->vertex_buffers[i]));
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

struct cp_texture_handle {
   void *functions;
   uint32_t sampler_index;
};

static uint64_t
cp_create_texture_handle(struct pipe_context *ctx,
                         struct pipe_sampler_view *view,
                         const struct pipe_sampler_state *state)
{
   struct cp_texture_handle *h = CALLOC_STRUCT(cp_texture_handle);
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
   FREE((void *)(uintptr_t)handle);
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
   ctx->base.create_texture_handle = cp_create_texture_handle;
   ctx->base.create_image_handle = cp_create_image_handle;
   ctx->base.delete_texture_handle = cp_delete_texture_handle;
   ctx->base.delete_image_handle = cp_delete_image_handle;

   ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
   ctx->base.const_uploader = ctx->base.stream_uploader;

   cudapipe_init_context_resource_funcs(&ctx->base);

   return &ctx->base;
}
