#ifndef CP_CONTEXT_H
#define CP_CONTEXT_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include <cuda.h>

struct cp_shader_binary;

#define CP_MAX_SHADER_BUFFERS 16
#define CP_MAX_CONST_BUFFERS  16
#define CP_MAX_SAMPLERS       256

struct cp_context {
   struct pipe_context base;

   struct cp_screen *screen;

   struct pipe_framebuffer_state framebuffer;
   struct pipe_viewport_state viewport;
   struct pipe_scissor_state scissor;

   /* Visibility buffer, rebuilt per draw: it resolves which triangle of the
    * current draw wins each pixel. */
   CUdeviceptr visbuf;
   unsigned visbuf_w, visbuf_h;

   /* Depth buffer for the render pass, one sortable uint32 per pixel. This is
    * what carries occlusion across draws. */
   CUdeviceptr depthbuf;
   unsigned depthbuf_w, depthbuf_h;
   bool depthbuf_cleared;

   struct pipe_depth_stencil_alpha_state depth_stencil;

   struct cp_shader_binary *compute_shader;
   struct cp_shader_binary *vs_shader;
   struct cp_shader_binary *fs_shader;

   /* Blend state */
   struct pipe_blend_state blend_state;
   bool blend_enabled;

   /* Texture state for FS */
   CUtexObject tex_objects[32];
   unsigned num_tex_objects;
   struct {
      void *data;
      unsigned width, height;
      unsigned row_stride;
      unsigned pixel_size;
      enum pipe_format format;
   } tex_resources[32];

   /* Vertex buffers and elements */
   struct pipe_vertex_buffer vertex_buffers[16];
   unsigned num_vertex_buffers;

   struct pipe_vertex_element vertex_elements[16];
   unsigned num_vertex_elements;
   unsigned vertex_stride;

   struct {
      void *buffer;
      unsigned buffer_size;
   } compute_ssbos[CP_MAX_SHADER_BUFFERS];
   unsigned num_compute_ssbos;

   struct {
      void *buffer;
      unsigned buffer_size;
   } compute_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_compute_ubos;

   struct {
      void *buffer;
      unsigned buffer_size;
   } fs_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_fs_ubos;

   struct {
      void *buffer;
      unsigned buffer_size;
   } vs_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_vs_ubos;

   /* Device-visible table of deduplicated sampler states. Descriptors refer to
    * entries by index; see cp_register_sampler(). */
   CUdeviceptr sampler_table;
   unsigned num_samplers;
};

struct pipe_context *
cudapipe_create_context(struct pipe_screen *screen, void *priv, unsigned flags);

uint32_t cp_depth_to_sortable(float depth);
void cp_clear_depthbuf(struct cp_context *cp, float depth);

#endif
