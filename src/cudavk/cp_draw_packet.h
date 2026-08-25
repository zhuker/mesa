/* Immutable execution structures shared by the native frontend and renderer. */
#ifndef CP_DRAW_PACKET_H
#define CP_DRAW_PACKET_H

#include "cp_draw_types.h"
#include "kernels/cp_rast_types.h"
#include <stdint.h>

/* One VkRenderingAttachmentInfo resolved to its exact image subresource. */
struct cp_depth_attachment {
   uint64_t data;
   uint32_t row_stride;
   uint32_t sample_stride;
   uint32_t pixel_stride;
   uint32_t format;
   uint32_t load;
   uint32_t store;
   /* A stencil LOAD_OP_CLEAR on the shared aspect: the store writes this
    * value instead of preserving the image's stencil bits. */
   uint32_t stencil_clear;
   uint32_t stencil_value;
};

/* One vkCmdBeginRendering boundary, owned by its recording command buffer. */
struct cp_render_scope {
   struct cp_fb_desc fb;
   struct cp_depth_attachment depth;
   uint32_t attachment_samples;
   uint32_t serial;

   /*
    * What the recorded pass will ask of this scope, computed by
    * cpvk_plan_batches() when recording ends. Zero until then. This is the
    * per-scope half of the record-time plan: a resource decision -- sizing,
    * path choice, a captured launch sequence -- can read it before the first
    * draw executes instead of discovering it draw by draw.
    */
   uint32_t planned_draws;
   uint32_t planned_blended_draws;
   uint64_t planned_tris;
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
