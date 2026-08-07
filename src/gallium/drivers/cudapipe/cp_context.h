#ifndef CP_CONTEXT_H
#define CP_CONTEXT_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"

struct cp_shader_binary;

#define CP_MAX_SHADER_BUFFERS 16
#define CP_MAX_CONST_BUFFERS  16

struct cp_context {
   struct pipe_context base;

   struct cp_screen *screen;

   struct pipe_framebuffer_state framebuffer;
   struct pipe_viewport_state viewport;
   struct pipe_scissor_state scissor;

   struct cp_shader_binary *compute_shader;

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
