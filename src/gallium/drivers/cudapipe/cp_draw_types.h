/*
 * What the draw pipeline needs to know about a draw.
 *
 * The pipeline itself -- cp_draw_execute(), 1,973 lines of it -- touches
 * Gallium on sixteen lines, and reads exactly eight fields of
 * pipe_draw_info. This is those eight, so the body can stay as it is while
 * the front end above it changes: the Gallium adapter fills one in from
 * pipe_draw_info, and the native Vulkan driver fills the same one in from a
 * recorded command buffer.
 *
 * The field names are deliberately those of pipe_draw_info, so that
 * introducing this changed no line of the pipeline that reads them.
 */

#ifndef CP_DRAW_TYPES_H
#define CP_DRAW_TYPES_H

#include <cuda.h>
#include <stdbool.h>
#include "util/format/u_formats.h"
#include "util/mesa-blake3.h"
#include "compiler/shader_enums.h"

struct cp_draw_call {
   enum mesa_prim mode;
   unsigned index_size;             /* 0, 1, 2 or 4 bytes */
   unsigned instance_count;
   unsigned start_instance;
   bool has_user_indices;
   /* The index buffer, already resolved to something a kernel can read.
    * Under Gallium this was a pipe_resource the pipeline had to unwrap. */
   const void *index_ptr;
   const void *indirect;            /* opaque; the pipeline only tests it */
   unsigned indirect_offset;
};

/* Layout-compatible with pipe_draw_start_count_bias on purpose: the batch and
 * episode snapshots copy arrays of these around. */
struct cp_draw_range {
   unsigned start;
   unsigned count;
   int index_bias;
};

#endif /* CP_DRAW_TYPES_H */
