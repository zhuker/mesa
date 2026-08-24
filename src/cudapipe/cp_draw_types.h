/*
 * What the draw pipeline needs to know about a draw.
 *
 * The field names date from the port, when they were chosen to match
 * pipe_draw_info so extraction changed no line of the pipeline. Nothing
 * enforces that correspondence any more -- this driver's recorded command
 * buffer is the only producer -- so they are just names now, renameable
 * whenever a better one exists.
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

/* The scissor rectangle. Layout-compatible with pipe_scissor_state, which is
 * four plain unsigneds and contains nothing the pipeline does not use -- so
 * unlike the other state, converting the batch key's copy too changes nothing
 * about what merges. */
struct cp_rect {
   unsigned minx;
   unsigned miny;
   unsigned maxx;
   unsigned maxy;
};

/* Layout-compatible with pipe_draw_start_count_bias on purpose: the batch and
 * episode snapshots copy arrays of these around. */
struct cp_draw_range {
   unsigned start;
   unsigned count;
   int index_bias;
};

/*
 * The pipeline's view of the pipeline state.
 *
 * Only the fields the rasterizer and shading stages actually read: three of
 * pipe_depth_stencil_alpha_state, three of pipe_rasterizer_state, and the
 * viewport's scale and translate. Field names again match Gallium's so that
 * introducing these changed nothing that reads them.
 *
 * The comparison that decides whether a held-back batch must be flushed still
 * happens on the full Gallium state in the adapter, deliberately: narrowing it
 * to the fields the pipeline reads would make batches larger than they are
 * today, and batch size is what makes the clipper's unstable primitive order
 * visible (handoff gaps 15 and 16). A factoring must not change what merges.
 */

/*
 * The depth comparison uses enum cp_compare_func from the kernel ABI header,
 * which is where it has always lived -- the kernels read it. The host was
 * passing Gallium's PIPE_FUNC_* values into those fields and relying on the
 * two agreeing; the adapter now static_asserts that they do, and the renderer
 * names only the driver's own constants.
 */
struct cp_depth_state {
   bool depth_enabled;
   bool depth_writemask;
   unsigned depth_func;             /* enum cp_compare_func */
};

/* Face bits, the driver's own. Gallium's values, asserted by the adapter. */
enum cp_face {
   CP_FACE_NONE = 0,
   CP_FACE_FRONT = 1,
   CP_FACE_BACK = 2,
   CP_FACE_FRONT_AND_BACK = 3,
};

struct cp_raster_state {
   unsigned cull_face;              /* enum cp_face */
   bool front_ccw;
   bool scissor;
};

struct cp_viewport_state {
   float scale[3];
   float translate[3];
};

/*
 * The attachments, resolved when the framebuffer was bound.
 *
 * The pipeline asked four separate questions of a pipe_framebuffer_state on
 * every draw -- is there a colour buffer, where is its memory, what encoding
 * does the writeback use, what is its sample stride -- and answered them by
 * unwrapping a pipe_resource and calling cp_color_encoding_from_format each
 * time. They are answered once here instead.
 *
 * The Gallium state stays beside this for the batch key, which compares
 * texture identity and format: see cp_draw_types.h's note on why the key did
 * not move with the fields.
 */
struct cp_fb_desc {
   unsigned width, height;
   unsigned nr_cbufs;
   void *color;                     /* base address, NULL for depth-only */
   int color_encoding;              /* enum cp_color_encoding, -1 = unsupported */
   unsigned color_sample_stride;
   uint64_t texture_cookie;          /* native cpvk_image_view, Gallium zero */
   bool has_zs;
};

/*
 * Vertex input, resolved when the state was bound.
 *
 * The fetch kernel wants a conversion, a channel count and a channel size per
 * element; the draw path was deriving all three from a pipe_format on every
 * draw, along with the attribute's block size and its fill-w rule, and
 * unwrapping a pipe_resource for each buffer base. None of that varies with
 * the draw.
 */
struct cp_vertex_elem {
   unsigned vertex_buffer_index;
   unsigned src_offset;
   unsigned src_stride;
   unsigned instance_divisor;
   unsigned attr_size;              /* bytes in memory */
   unsigned nr_chan;
   unsigned chan_bytes;
   unsigned swizzle;
   unsigned conv;                   /* enum cp_vf_conv */
   unsigned fill_w;
};

/*
 * The front end's statement that two draws have identical pipeline state.
 *
 * The batcher only ever memcmps this, so what it contains is the front end's
 * business: the Gallium adapter puts whole CSO structs in, and the native
 * Vulkan driver will put a pipeline handle and its dynamic state in. Neither
 * changes what merges for the other, and the comparison stays exactly as wide
 * as it is today -- which matters, because widening or narrowing it changes
 * batch sizes, and batch size is what makes the clipper's unstable primitive
 * order visible (gaps 15 and 16).
 *
 * The field table exists so that CUDAPIPE_DEBUG_BATCHDIFF can still name the
 * piece of state that broke a batch. That report is not a nicety: it is what
 * turned "why does nothing merge" into "fs_ubos breaks 97% of runs" and then
 * into the per-draw binding tables.
 */
#define CP_BATCH_STATE_BYTES 768

struct cp_batch_state_field {
   const char *name;
   unsigned off, size;
};

/* Implemented by whichever front end is built. */
const struct cp_batch_state_field *cp_batch_state_fields(unsigned *count);

#endif /* CP_DRAW_TYPES_H */
