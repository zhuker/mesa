/* Immutable execution structures shared by the native frontend and renderer. */
#ifndef CP_DRAW_PACKET_H
#define CP_DRAW_PACKET_H

#include "cp_draw_types.h"
#include <stdint.h>

/* One vkCmdBeginRendering boundary, owned by its recording command buffer. */
struct cp_render_scope {
   struct cp_fb_desc fb;
   uint32_t attachment_samples;
   uint32_t serial;
};

#define CP_RENDER_SCOPE_NONE      UINT32_MAX
#define CP_RENDER_SCOPE_INHERITED (UINT32_MAX - 1u)

#endif
