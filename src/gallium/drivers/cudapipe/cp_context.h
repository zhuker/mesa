#ifndef CP_CONTEXT_H
#define CP_CONTEXT_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include <cuda.h>
#include "kernels/cp_rast_types.h"

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
      CUdeviceptr managed_copy; /* Device buffer for user_buffer data */
      unsigned managed_size;
   } compute_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_compute_ubos;

   struct {
      void *buffer;
      unsigned buffer_size;
      CUdeviceptr managed_copy;
      unsigned managed_size;
   } fs_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_fs_ubos;

   struct {
      void *buffer;
      unsigned buffer_size;
      CUdeviceptr managed_copy;
      unsigned managed_size;
   } vs_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_vs_ubos;

   /* Device-visible table of deduplicated sampler states. Descriptors refer to
    * entries by index; see cp_register_sampler(). */
   CUdeviceptr sampler_table;
   struct cp_sampler_info sampler_table_host[CP_MAX_SAMPLERS];
   unsigned num_samplers;

   /*
    * Scratch arena for the buffers a draw needs.
    *
    * A draw wants a dozen or so short-lived device allocations, and managed
    * allocation is expensive enough that doing it per draw dominated frame
    * time. Bump-allocate from one buffer instead and reset it per draw; it is
    * grown between draws, when nothing points into it.
    */
   struct {
      CUdeviceptr base[2];
      size_t size[2];
      size_t used;
      size_t peak;
      unsigned current;  /* 0 or 1: which arena is active */
      CUdeviceptr overflow[64];
      unsigned num_overflow;
   } scratch;
};

struct pipe_context *
cudapipe_create_context(struct pipe_screen *screen, void *priv, unsigned flags);

uint32_t cp_depth_to_sortable(float depth);
void cp_clear_depthbuf(struct cp_context *cp, float depth);

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
