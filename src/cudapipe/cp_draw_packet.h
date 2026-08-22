/* Immutable execution structures shared by the native frontend and renderer. */
#ifndef CP_DRAW_PACKET_H
#define CP_DRAW_PACKET_H

#include "cp_draw_types.h"
#include "kernels/cp_rast_types.h"
#include <stdint.h>

/* One vkCmdBeginRendering boundary, owned by its recording command buffer. */
struct cp_render_scope {
   struct cp_fb_desc fb;
   uint32_t attachment_samples;
   uint32_t serial;
};

#define CP_RENDER_SCOPE_NONE      UINT32_MAX
#define CP_RENDER_SCOPE_INHERITED (UINT32_MAX - 1u)

struct cp_shader_binary;

struct cp_draw_state {
   struct cp_shader_binary *vs, *fs;
   struct cp_viewport_state viewport;
   struct cp_raster_state raster;
   struct cp_depth_state depth;
   struct cp_blend_desc blend;
   uint32_t pipeline_samples;

   struct cp_vertex_elem velem[16];
   uint64_t vb_base[16];
   uint64_t vs_ubos[16];
   uint64_t fs_ubos[16];
   uint32_t num_vertex_buffers;
   uint32_t num_vertex_elements, vertex_stride;
   uint32_t num_vs_ubos, num_fs_ubos;
};

struct cp_draw_packet {
   const struct cp_render_scope *scope;
   struct cp_draw_state state;
   struct cp_draw_call call;
   struct cp_draw_range range;
   struct cp_rect scissor;
   uint32_t drawid_offset;
};

#endif
