#ifndef CP_CONTEXT_H
#define CP_CONTEXT_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include <cuda.h>

struct cp_shader_binary;

#define CP_MAX_SHADER_BUFFERS 16
#define CP_MAX_CONST_BUFFERS  16

struct cp_context {
   struct pipe_context base;

   struct cp_screen *screen;

   struct pipe_framebuffer_state framebuffer;
   struct pipe_viewport_state viewport;
   struct pipe_scissor_state scissor;

   /* Persistent visibility buffer for the current render pass */
   CUdeviceptr visbuf;
   unsigned visbuf_w, visbuf_h;
   bool visbuf_cleared;

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
};

struct pipe_context *
cudapipe_create_context(struct pipe_screen *screen, void *priv, unsigned flags);

#endif
