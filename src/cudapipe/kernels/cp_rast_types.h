/*
 * Shared types between host code and CUDA kernels.
 * This header is included by both C host code and NVRTC-compiled .cu kernels.
 */

#ifndef CP_RAST_TYPES_H
#define CP_RAST_TYPES_H

#ifndef __CUDACC__
#include <stdint.h>
#else
/* NVRTC doesn't have stdint.h — define what we need */
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned long long uintptr_t;
#endif

struct cp_framebuffer_info {
   uint64_t color[4];       /* Device pointers to color attachments */
   uint64_t depth;          /* Device pointer to depth buffer (packed depth|triID or float) */
   uint64_t stencil;        /* Device pointer to stencil buffer */
   uint32_t width;
   uint32_t height;
   uint32_t color_format[4]; /* PIPE_FORMAT_* for each color attachment */
   uint32_t depth_format;
   uint32_t num_color_attachments;
};

struct cp_cache_convert_args {
   uint64_t src;
   uint64_t surface;
   uint64_t src_pitch;
   uint64_t src_slice;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t target; /* 0=2D, 1=2D layered/cube, 2=3D */
   uint32_t format; /* 1=R11G11B10 -> half4, 2=BC1, 3=BC3 */
};

struct cp_clear_args {
   uint64_t target;         /* Device pointer to buffer to clear */
   uint32_t width;
   uint32_t height;
   uint32_t stride;         /* Row stride in bytes */
   uint32_t clear_value[4]; /* Clear color as 4x u32 */
   uint32_t pixel_size;     /* Bytes per pixel */
};

struct cp_depth_attachment_args {
   uint64_t image;
   uint64_t depthbuf;
   uint32_t width;
   uint32_t height;
   uint32_t row_stride;
   uint32_t sample_stride;
   uint32_t pixel_stride;
   uint32_t format;
   uint32_t samples;
   uint32_t stencil_clear;
   uint32_t stencil_value;
};

struct cp_vertex_args {
   uint64_t positions_out;  /* Output: clip-space positions (float4 per vertex) */
   uint64_t varyings_out;   /* Output: interpolated varyings */
   uint64_t vertex_buffers[16]; /* Input vertex buffers */
   uint32_t vertex_strides[16];
   uint64_t index_buffer;
   uint32_t index_type;     /* 0=none, 2=uint16, 4=uint32 */
   uint32_t vertex_count;
   uint32_t first_vertex;
   uint32_t num_varyings;   /* Number of output floats per vertex (beyond position) */
};

struct cp_rasterize_args {
   /* Optional device-side launch predicate. Zero leaves the ordinary path
    * unchanged; otherwise every raster stage returns unless *path_flag equals
    * path_value. Opaque tiled episodes use this to run classic rasterization
    * only when device-side binning reports overflow. */
   uint64_t path_flag;
   uint32_t path_value;
   uint64_t positions;      /* Input: contiguous vertex stream fallback */
   /* Optional uint64_t[max output primitives]. Each nonzero entry is the
    * device address of that primitive's first vertex slot. Clipped wholly
    * inside primitives can therefore keep reading immutable VS output. */
   uint64_t prim_refs;
   uint64_t varyings;       /* Input: varyings from VS */
   uint64_t framebuffer;    /* Output: visibility buffer (uint64 per pixel) */
   uint64_t color_buffer;   /* Output: color buffer (uint32 per pixel, RGBA8) */
   uint32_t width;
   uint32_t height;
   uint32_t num_triangles;
   uint32_t num_varyings;
   /* Viewport (raw scale/translate for proper Vulkan Y handling) */
   float vp_x, vp_y, vp_w, vp_h;
   float vp_near, vp_far;
   float vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y;
   /*
    * The rectangle of pixels a fragment may land in, inclusive on both ends:
    * the framebuffer intersected with the viewport rectangle and, when the
    * rasterizer state asks for it, the scissor. Vulkan clips primitives to the
    * view volume, which after the viewport transform is exactly the viewport
    * rectangle, so geometry running past NDC +-1 must not reach the pixels
    * beside the viewport — with two viewports side by side those pixels belong
    * to the other one. Computed on the host in the same half-pixel convention
    * llvmpipe uses in lp_setup_set_viewports(); an empty rectangle (x1 < x0)
    * draws nothing.
    */
   int32_t clip_x0, clip_y0, clip_x1, clip_y1;
   /*
    * Per-draw clip rectangles, for a batch whose draws disagree on the
    * scissor. clip_rects points at one int4 {x0,y0,x1,y1} per merged draw —
    * already intersected with the batch-wide rectangle above, which then
    * carries only framebuffer ∩ viewport. setup_triangle() resolves the
    * primitive to its draw through draw_slices, the same search
    * cp_write_batch_rows() does. Zero means every primitive takes clip_*
    * above, which is the unbatched path unchanged.
    */
   uint64_t rect_draw_slices;  /* const struct cp_draw_slice * */
   uint64_t clip_rects;        /* const int4 *, one per draw */
   uint32_t num_rect_slices;
   uint32_t rect_prim_shift;
   /* Rasterizer state */
   uint32_t cull_mode;      /* 0=none, 1=front, 2=back */
   uint32_t front_face;     /* 0=CCW, 1=CW */
   /* Depth buffer for the render pass, one sortable uint32 per pixel. The
    * visibility buffer only resolves depth within a single draw, so the test
    * against earlier draws happens here. */
   uint64_t depthbuf;
   uint32_t depth_test;     /* Enable the comparison below */
   uint32_t depth_func;     /* enum pipe_compare_func */
   uint32_t depth_key_invert; /* Depth function prefers the farthest fragment */
   /* Device pointer to the triangle count produced by near-plane clipping.
    * Zero means the count is not known on the GPU and num_triangles applies. */
   uint64_t tri_count;
   /*
    * Stable clipping keeps the primitive ID tied to the input triangle so
    * ordered blending can sort by submission order.  active_ids compacts the
    * actual output into work indices without renumbering those primitives:
    * stage 1 maps work item i through active_ids[i], while later queues and
    * visibility records continue to carry the stable ID.  num_triangles is
    * the stable-ID limit and *tri_count is the compact work count.
    */
   uint64_t active_ids;      /* const uint32_t *, zero on the compact path */
   /*
    * Alpha-tested geometry. Visibility is resolved before the shader runs, so
    * a fragment that turns out to discard has already displaced the one behind
    * it. The draw is repeated: each pass records the triangle that discarded
    * at a pixel here, and later passes skip it so the next fragment can win.
    */
   uint64_t reject;         /* uint32[reject_layers] per pixel, 0 if unused */
   uint64_t resolved;       /* one byte per pixel, set once a pixel is written */
   uint32_t reject_layers;
   uint32_t reject_passes;  /* how many layers hold a triangle so far */
   /*
    * Point rasterization. A POINT_LIST draw is expanded on the host into one
    * degenerate triangle per point — all three vertices the same — and the
    * square it covers is built here from the vertex shader's gl_PointSize.
    */
   uint32_t point_mode;
   int32_t psiz_slot;       /* VS output slot holding VARYING_SLOT_PSIZ, -1 if none */
   /*
    * Ordered blending. A blended draw cannot resolve to one fragment per
    * pixel: every layer has to be composited, in the order the primitives were
    * submitted. Instead of the nearest fragment, the visibility buffer then
    * selects the lowest numbered primitive at or after peel_next, and the host
    * repeats the draw, advancing peel_next past whatever was blended, until
    * nothing is left.
    */
   uint64_t peel_next;      /* uint32 per pixel: first primitive not yet blended */
   uint64_t peel_any;       /* uint32: set when any pixel still had a layer */
   uint32_t blend_peel;
   /* Samples per pixel, 1 or CP_MAX_SAMPLES. Coverage and depth are resolved
    * per sample; shading stays per pixel. */
   uint32_t num_samples;
   /*
    * TEMPORARY INSTRUMENTATION (CUDAPIPE_FRAG_CENSUS). Both are uint32 per
    * pixel, sample planes folded together, and both are zero unless the census
    * is on. `census` counts every fragment emit_fragment is called with, i.e.
    * raw coverage before any visibility resolution; `census_depth` counts the
    * subset that survives the depth test against earlier draws. Nothing reads
    * them on the device, so a set pointer cannot change what is rendered.
    */
   uint64_t census;
   uint64_t census_depth;
   /*
    * TEMPORARY (CUDAPIPE_ABUFFER). A counted per-pixel fragment list, built by
    * two extra rasterization passes over the same geometry the peel loop is
    * about to re-rasterize CP_BLEND_LAYERS times. Nothing downstream reads it:
    * the draw is still rendered by the peel loop, and this exists only to be
    * checked against what that loop composites.
    *
    * abuf_mode selects which pass this launch is. In either the fragment is
    * counted or recorded *after* the depth test and *instead of* the
    * visibility buffer, so a counting or filling launch writes nothing the
    * renderer reads.
    */
   uint64_t abuf_counts;    /* uint32 per pixel: depth-passing fragments */
   uint64_t abuf_offsets;   /* uint32 per pixel: exclusive scan of the above */
   uint64_t abuf_cursor;    /* uint32 per pixel: fill position within the run */
   uint64_t abuf_frags;     /* uint32 per fragment: primitive id */
   uint64_t abuf_overflow;  /* uint32: fragments the fill could not place */
   uint32_t abuf_capacity;  /* entries in abuf_frags */
   uint32_t abuf_mode;      /* CP_ABUF_* below */
   /*
    * Added to the local triangle id in the fragment the A-buffer records —
    * and nowhere else. A pass episode rasterizes each segment from its own
    * clipped buffer, whose triangle ids start at zero, but the merged
    * per-pixel lists sort on the recorded id, so each segment offsets its
    * ids by the slots of every segment before it. Zero outside an episode.
    */
   uint32_t abuf_prim_base;
   /*
    * Single-pass build. When abuf_recs is non-null a counting launch also
    * appends one 64-bit (pixel << 32 | prim) record per fragment through the
    * one global cursor, so the fill never has to rasterize a second time —
    * cp_abuf_fill_recs replays the records instead. Appends past
    * abuf_capacity land in abuf_overflow, exactly as a fill that ran out of
    * room would.
    */
   uint64_t abuf_recs;       /* uint64 per fragment: pixel << 32 | prim */
   uint64_t abuf_rec_cursor; /* uint32: the records' one append cursor */
};

#define CP_ABUF_OFF     0u
#define CP_ABUF_COUNT   1u
#define CP_ABUF_FILL    2u

/*
 * Whether the device code above is compiled at all.
 *
 * The fields stay in the struct unconditionally — the host writes them and the
 * two sides have to agree on the layout — but the branches that read them sit
 * in emit_fragment, which runs about 960 million times a frame on
 * particlesystem. Three branches on a pointer that is uniformly null still
 * cost 1.6% there, and that is a tax on every future A/B of a tree carrying
 * this instrumentation. NVRTC compiles at run time, so the host defines this
 * to 1 only when CUDAPIPE_ABUFFER or CUDAPIPE_FRAG_CENSUS is set, exactly the
 * way CP_SMALL_THRESHOLD and friends are handed over; see cp_kernels.c.
 */
#ifndef CP_ABUF_INSTRUMENT
#define CP_ABUF_INSTRUMENT 0
#endif

/* Elements one block of the prefix sum scans. One thread per element, two
 * shared buffers, so the shared cost is 2 * this * 4 bytes. */
#define CP_ABUF_SCAN_BLOCK 512

/* Longest per-pixel run the sort handles in shared memory. Measured maximum
 * depth on particlesystem is 406-411; anything above this falls back to a
 * single-threaded insertion sort in global memory, which is correct and slow
 * rather than wrong. */
#define CP_ABUF_SORT_MAX 1024

/* Peel layers the verification log records for every pixel. */
#define CP_ABUF_LOG_LAYERS 32

/*
 * Counters the quad merge and the instrumented interpolator share, in one
 * allocation so a single copy back reads all of them.
 */
#define CP_ABUF_DBG_PEEL_QUADS  0  /* quads cp_fs_interpolate emitted, all passes */
#define CP_ABUF_DBG_NOT_FOUND   1  /* ... whose (block, primitive) is not in the merge */
#define CP_ABUF_DBG_DEGENERATE  2  /* ... whose interpolation refused a covered pixel */
#define CP_ABUF_DBG_FULL        3  /* ... dropped because the fragment buffer filled */
#define CP_ABUF_DBG_SLOT_BAD    4  /* shaded fragments whose A-buffer slot was out of range */
/*
 * The six words the host reads back after each A-buffer drain, in one
 * allocation and in this order: sum3 (scan total, fill overflow, long runs),
 * bsum3 (quad total, quad overflow), then clist_count. Contiguous so the
 * readback is a single copy — it happens once per eligible draw, which is
 * hundreds of times a frame.
 */
#define CP_ABUF_COUNTERS        6
/* Behind the six: one quad count per pass-episode segment, in the same
 * allocation so the episode's one drain reads everything in one copy. */
#define CP_PASS_MAX_SEGS        64

#define CP_ABUF_DBG_COUNTERS    5

/*
 * Everything the blend equation is, in the form both the peel path's writeback
 * and the A-buffer's composite read it.
 *
 * One struct rather than two sets of flat fields, because the two kernels
 * evaluate the same equation on the same draw and a second copy of it is a
 * second thing that can be right about a draw the first one is wrong about —
 * which is the failure mode gap 12 in CUDAPIPE_HANDOFF.md already is.
 */
struct cp_blend_desc {
   uint32_t enable;
   /* pipe_blend_state factors/functions for the colour and alpha channels. */
   uint32_t rgb_src_factor, rgb_dst_factor, rgb_func;
   uint32_t alpha_src_factor, alpha_dst_factor, alpha_func;
   uint32_t colormask;
};

/* Passes an alpha-tested draw gets to find a fragment that survives. Each one
 * is a full rasterize and shade of the draw, so this is bought with time:
 * dropping it to 4 costs Sponza 0.14% of its pixels. */
#define CP_DISCARD_LAYERS 8

/*
 * Near-plane clipping. A triangle crossing the plane has a vertex with w <= 0,
 * whose perspective divide produces a position that is not merely wrong but
 * mirrored, so it has to be cut before projection. Clipping one plane splits a
 * triangle into at most two, so the output buffer holds 2x the input.
 */
struct cp_clip_args {
   uint64_t vs_out;         /* Input: 3 vertices per triangle, num_slots float4 each */
   uint64_t out;            /* Output: same layout, compacted */
   uint64_t out_count;      /* Output: uint32 triangle counter */
   /* Stable mode optionally appends each live fixed-slot ID here.  This keeps
    * primitive IDs ordered while letting rasterization skip retired holes. */
   uint64_t active_ids;      /* Output: uint32[max_triangles], or zero */
   /* Optional output uint64_t[max_triangles]. Publication happens only after
    * a crossing output is complete; inside outputs point at vs_out directly. */
   uint64_t prim_refs;
   uint32_t num_triangles;
   uint32_t num_slots;      /* Position plus varyings, i.e. num_varyings + 1 */
   uint32_t max_triangles;  /* Capacity of `out`, in triangles */
   /*
    * Lay the output out by input triangle rather than compacting it: input
    * triangle t owns slots 4t..4t+3, and the ones it does not fill are marked
    * degenerate. A batch's primitive index is then monotone in submission
    * order, which is what the A-buffer sorts on and so what makes merging
    * blended draws into one episode legal at all. Compaction with atomicAdd
    * cannot promise that — its output order is whichever thread got there
    * first, so a later draw's fragment can sort before an earlier draw's.
    *
    * Costs nothing in the rasterizer's grid, which is already sized for the
    * 4x worst case; the threads that would have exited on the count now exit
    * on a zero area instead.
    */
   uint32_t stable;
};

#define CP_CLIP_MAX_OUT 8
#define CP_CLIP_PRIM_SHIFT 3
#if CP_CLIP_MAX_OUT != (1u << CP_CLIP_PRIM_SHIFT)
#error "stable clip IDs require CP_CLIP_MAX_OUT == 1 << CP_CLIP_PRIM_SHIFT"
#endif

#define CP_MAX_CLIP_SLOTS 16

/* Shader kernel argument slots. 0..7 are the fixed stage inputs and 18.. are
 * the uniform/descriptor buffers; the gap in between is free. */
#define CP_ARG_SLOT_DISCARD 8

/*
 * Draw batching. Several consecutive draws that differ in nothing the result
 * can depend on are submitted as one, so that the grids are sized to the batch
 * rather than to a draw of a dozen triangles. The vertex shader then has to
 * pick its own draw's bindings out of a table:
 *
 *   row  = ((const uint32_t *)args[CP_ARG_SLOT_BATCH_ROWS])
 *             [thread_id & *(uint32_t *)args[CP_ARG_SLOT_BATCH_MASK]]
 *   base = ((void **)args[CP_ARG_SLOT_UBO_TABLE])[row * CP_ARG_UBO_STRIDE + i]
 *
 * All three slots are always filled, so the generated code has no branch and
 * no batched/unbatched variant. A single draw sets the table to `&args[18]`,
 * the mask to zero and the row array to one word holding zero, which makes the
 * expression above compute exactly the args[18 + i] the shader used to load.
 *
 * The row is a lookup rather than arithmetic on the thread id because a batch
 * no longer has to replay one index range: draws of different vertex counts
 * merge, so which draw a thread belongs to is a search over the batch's
 * offsets. cp_vertex_fetch does that search already to know what to gather,
 * and writes the answer here instead of the shader repeating it.
 *
 * Both drawing stages read them: the vertex stage's rows are written by
 * cp_vertex_fetch, the fragment stage's by the interpolator (see
 * cp_write_batch_rows), and every batch — blended or opaque — carries its
 * fragment bindings per draw through the same table.
 */
#define CP_ARG_SLOT_UBO_TABLE  9
#define CP_ARG_SLOT_BATCH_ROWS 10
#define CP_ARG_SLOT_BATCH_MASK 11

/*
 * gl_FrontFacing: one byte per shaded slot, written by the interpolator, read
 * by the fragment stage at its own thread id. Fragment stage only; the slot is
 * null everywhere else and no other stage emits load_front_face.
 */
#define CP_ARG_SLOT_FRONT_FACE 12

/*
 * Coverage: one byte per shaded slot, written by the interpolator, non-zero
 * for a lane the primitive actually covers.
 *
 * The fragment stage shades whole 2x2 quads so that derivatives can be taken
 * across one, which means a lane outside the primitive is shaded too. Vulkan
 * calls those helper invocations, and says their stores and atomics have no
 * effect — cp_fs_writeback enforces that for colour and depth by consulting
 * this mask, and for everything else it did not need enforcing, because a
 * fragment shader had no other way to write anything.
 *
 * A shader with side effects does. So the slot is filled, and read, only for
 * a shader the compiler saw write memory: it returns before its body if its
 * lane is a helper. Filled with null for every other shader, which is what
 * keeps the generated code for one identical.
 */
#define CP_ARG_SLOT_COVERAGE 13

/* Optional device pointer to cp_fs_interp_args.  When non-null, a fragment
 * kernel prepares its own A-buffer input before executing the generated
 * shader body, eliminating the separate interpolation launch. */
#define CP_ARG_SLOT_FUSED_INTERP 14
/* Immutable uint64_t[row_count * site_count] CUtexObject table. It exists
 * only for CP_SHADER_EXEC_HW_INLINE launches. */
#define CP_ARG_SLOT_HW_TEX_TABLE 15

/*
 * One merged draw's slice of the assembled vertex stream.
 *
 * A batch concatenates its draws, so vertex v of the launch belongs to the
 * last draw whose vert_begin is not past it, and gathers from that draw's
 * index range. The table is bounded by CP_MAX_BATCH_DRAWS, so the search is a
 * handful of steps over something entirely in cache.
 */
struct cp_draw_slice {
   uint32_t vert_begin;    /* first assembled vertex of this draw */
   uint32_t index_bytes;   /* byte offset of its first index into the IB */
   uint32_t first_vertex;  /* index_bias when indexed, draw start when not */
   /* Assembled vertices per instance for an instanced draw, 0 for a plain
    * one — the same convention as the launch-wide field, which a batch
    * ignores in favour of this. vert_begin spans count that includes every
    * instance, so the fetch's slice search needs no other change. */
   uint32_t verts_per_instance;
};
/* Entries per draw in the table at CP_ARG_SLOT_UBO_TABLE; matches
 * CP_MAX_CONST_BUFFERS and the 18.. layout it stands in for. */
#define CP_ARG_UBO_STRIDE 16
#define CP_ARG_UBO_BASE 18

/* uint32 words per draw in the draw-parameter table at args[7]:
 * [0] first_vertex, [1] base_instance, [2] draw_id, [3] base_vertex
 * (0 for a non-indexed draw, where first_vertex is the draw's start).
 * The vertex shader indexes it by its batch row, so an unbatched draw's
 * single row reads exactly as the flat triple this used to be. */
#define CP_ARG_DRAW_PARAM_STRIDE 4

/* enum pipe_compare_func */
enum cp_compare_func {
   CP_FUNC_NEVER = 0,
   CP_FUNC_LESS,
   CP_FUNC_EQUAL,
   CP_FUNC_LEQUAL,
   CP_FUNC_GREATER,
   CP_FUNC_NOTEQUAL,
   CP_FUNC_GEQUAL,
   CP_FUNC_ALWAYS,
};

struct cp_resolve_args {
   uint64_t visbuf;
   uint64_t positions;
   uint64_t colors;
   uint64_t color_out;
   uint32_t width, height;
   float vp_x, vp_y, vp_w, vp_h;
   float vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y;
   uint32_t color_stride;
};

#define CP_MAX_TEXTURE_LEVELS 16
#define CP_MAX_FS_INPUTS 16

/* How a colour attachment's bytes are laid out, for the writeback kernel. */
enum cp_color_encoding {
   CP_COLOR_R8G8B8A8_UNORM = 0,
   CP_COLOR_B8G8R8A8_UNORM,
   CP_COLOR_R8G8B8A8_SRGB,
   CP_COLOR_B8G8R8A8_SRGB,
   CP_COLOR_R32G32B32A32_FLOAT,
   CP_COLOR_R16G16B16A16_FLOAT,
   CP_COLOR_R11G11B10_FLOAT,
   CP_COLOR_A2B10G10R10_UNORM,
   CP_COLOR_R16_SFLOAT,
   CP_COLOR_R16G16_SFLOAT,
   CP_COLOR_R8_UNORM,
   CP_COLOR_R8G8_UNORM,
};

/*
 * Gathers the fragment shader's inputs for every covered pixel.
 *
 * Covered pixels are compacted into pixel_list so the fragment shader can be
 * launched over just those, one thread per pixel.
 */
struct cp_fs_interp_args {
   uint64_t visbuf;
   uint64_t positions;      /* Contiguous clip-space stream fallback */
   uint64_t prim_refs;      /* const uint64_t * primitive bases, or zero */
   uint64_t vs_out;         /* Vertex shader output buffer */
   uint64_t pixel_list;     /* Out: y * width + x for each covered pixel */
   uint64_t counter;        /* Out: number of covered pixels */
   uint64_t fs_in;          /* Out: interpolated varyings, per shaded pixel */
   uint64_t frag_coord;     /* Out: float4 (x, y, z, 1/w) per shaded pixel */
   /*
    * Out: one byte per slot, zero for a helper lane. Pixels are compacted in
    * 2x2 quads so the shader can take screen-space derivatives across one, and
    * an uncovered corner still has to be shaded to supply them — it just must
    * not reach the framebuffer.
    */
   uint64_t coverage;
   /*
    * Out: one byte per slot, non-zero when the primitive that won the slot
    * faces the viewer. gl_FrontFacing is not a varying — it is a property of
    * the primitive — and this is the only place both the primitive and the
    * slot it shades are in hand. front_ccw is the rasterizer state that turns
    * a signed screen-space area into a facing, and has to be carried here
    * because setup_triangle discards the sign.
    */
   uint64_t front_face;
   uint32_t front_ccw;
   uint32_t width, height;
   uint32_t vs_out_stride;  /* Bytes per vertex in vs_out */
   uint32_t fs_in_stride;   /* Bytes per pixel in fs_in */
   uint32_t num_fs_inputs;
   uint32_t max_pixels;
   /* Which vertex shader output slot feeds each fragment shader input slot,
    * matched by varying location on the host. -1 means nothing drives it. */
   int32_t input_vs_slot[CP_MAX_FS_INPUTS];
   uint32_t quad_width;     /* Quads across the framebuffer */
   float vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y;
   /* Point rasterization, matching cp_rasterize_args. pntc_input is the
    * fragment shader input slot that gl_PointCoord feeds, which no vertex
    * shader output drives, so the interpolator writes it directly. */
   uint32_t point_mode;
   int32_t psiz_slot;
   int32_t pntc_input;
   /*
    * The fragment shader input slot gl_FragCoord occupies, or -1.
    *
    * It is a shader_in variable at VARYING_SLOT_POS, so the host's match by
    * varying location pairs it with the vertex shader's gl_Position output and
    * the loop below would interpolate *clip space* into it. What the shader is
    * owed is the window coordinate, which is the `fc` this function already
    * computes for the frag_coord array — so the slot is named here and written
    * from that instead, exactly as pntc_input is.
    */
   int32_t pos_input;
   uint32_t num_samples;
   /*
    * TEMPORARY (CUDAPIPE_ABUFFER). What this interpolation emitted, folded
    * into the merged quad array so the peel path and the A-buffer path can be
    * compared without keeping a record per pass.
    *
    * A quad is a (2x2 block, primitive, 4-bit coverage mask) triple, and the
    * merge already holds every such triple the A-buffer implies, one per
    * distinct primitive per block, ascending. So the interpolator looks its
    * own triple up in that array and ORs its mask into dbg_peel_mask; over the
    * peel loop's passes the accumulated mask is what the merged mask has to
    * equal. A triple the merge does not contain is counted rather than
    * written, which is the mismatch this exists to find.
    *
    * All five are zero for every draw but the one being verified, and the code
    * that reads them compiles out entirely unless CP_ABUF_INSTRUMENT.
    */
   uint64_t dbg_blk_offsets;  /* uint32 per block: first quad of the block */
   uint64_t dbg_blk_counts;   /* uint32 per block: quads in the block */
   uint64_t dbg_quad_prim;    /* uint32 per quad: primitive id, ascending */
   uint64_t dbg_peel_mask;    /* uint32 per quad: OR of the masks emitted */
   uint64_t dbg_counters;     /* uint32[CP_ABUF_DBG_COUNTERS] */
   /*
    * TEMPORARY (CUDAPIPE_ABUFFER), step 3b. The quad stream as the *source* of
    * an interpolation rather than something to check one against, and the map
    * from a shaded slot back to the A-buffer slot it belongs to.
    *
    * cp_abuf_interpolate reads the first four and fills the same four output
    * arrays cp_fs_interpolate does, through the same cp_interp_pixel; the
    * A-buffer lists are read by both, to turn a (pixel, primitive) into the
    * one slot where both paths deposit their shaded colour.
    */
   uint64_t abuf_quad_prim;   /* uint32 per quad: primitive */
   uint64_t abuf_quad_mask;   /* uint8 per quad: 4-bit pixel coverage */
   uint64_t abuf_quad_block;  /* uint32 per quad: 2x2 block index */
   uint64_t abuf_frags;       /* uint32 per A-buffer slot: primitive, sorted */
   uint64_t abuf_offsets;     /* uint32 per pixel: first slot of its run */
   uint64_t abuf_counts;      /* uint32 per pixel: length of its run */
   uint64_t dbg_slot;         /* Out: uint32 per shaded slot, ~0 for no slot */
   uint32_t abuf_num_quads;
   /*
    * Which merged draw each shaded slot belongs to, for a batch whose draws
    * differ in their *fragment* uniform bindings.
    *
    * The vertex stage answers this from the vertex id, which is what
    * cp_vertex_fetch's slice search does. The fragment stage cannot: a shaded
    * slot is a pixel, and which draw covered it is a property of the primitive
    * that won it. So the interpolator — the one kernel that knows both the
    * slot and the primitive — writes the row here, and the shader reads it at
    * CP_ARG_SLOT_BATCH_ROWS exactly as the vertex shader does.
    *
    * The primitive index is a *stable* clipper output (see cp_clip_args), so
    * `prim >> 2` is the input triangle and `3 * (prim >> 2)` the assembled
    * vertex the slice table is keyed on. Null for a draw that is not a batch,
    * where the mask at CP_ARG_SLOT_BATCH_MASK sends every thread to row zero.
    */
   uint64_t draw_slices;      /* struct cp_draw_slice[num_draw_slices] */
   uint32_t num_draw_slices;
   /* log2 of the clipper's output slots per input triangle: 2 when it ran in
    * stable mode, 0 when it did not run at all. */
   uint32_t prim_shift;
   uint64_t out_batch_rows;   /* Out: uint32 per shaded slot */
   /*
    * Pass-episode mode: shade one segment's quads, densely. quad_list holds
    * the episode's quad indices grouped by segment; this launch covers
    * abuf_num_quads entries starting at quad_list_base, thread i shading
    * quad quad_list[quad_list_base + i] into dense slots 4i..4i+3. The quad
    * stream's primitive ids are episode-global, so abuf_prim_base is
    * subtracted before this segment's own vertex stream and slice table are
    * addressed. quad_list null outside an episode, and everything above
    * reads as it always did.
    */
   uint64_t quad_list;
   uint32_t quad_list_base;
   uint32_t abuf_prim_base;
   /* The device's own quad total, when abuf_num_quads is only a bound: a
    * drainless draw sizes its launches to what cannot be exceeded and the
    * interpolator stops here. Zero means abuf_num_quads is exact. */
   uint64_t num_quads_dev;
   /*
    * Merged shading groups: when set, this launch spans several segments and
    * each quad resolves its own positions, slice table, prim base and row
    * base through this table (struct cp_seg_range[num_seg_ranges], sorted by
    * prim_base) instead of the launch-wide fields above. row_base is what
    * cp_write_batch_rows adds to the row it finds, landing it in the group's
    * concatenated fs-UBO table; zero everywhere else.
    */
   uint64_t seg_ranges;
   uint32_t num_seg_ranges;
   uint32_t row_base;
   uint64_t quad_list_base_dev;
   /*
    * Fused direct shading. When fused_direct is set, this argument block is
    * the one handed to the generated fragment shader at
    * CP_ARG_SLOT_FUSED_INTERP, and the launch that would have been
    * cp_fs_interpolate was cp_fs_compact instead: it allocated the slots and
    * wrote pixel_list, coverage and out_batch_rows, and recorded each quad's
    * primitive here -- one uint32 per four slots -- because the primitive is
    * the one input cp_fs_direct_lane cannot recover from the slot alone. The
    * shader then interpolates its own slot before running its body.
    */
   uint64_t out_prim_list;    /* uint32 per slot quad: global primitive id */
   uint32_t fused_direct;
   uint32_t pad_fused;
};

struct cp_fs_writeback_args {
   uint64_t pixel_list;
   uint64_t fs_out;         /* Fragment shader colour output, per covered pixel */
   uint64_t color_out;
   uint64_t visbuf;         /* Source of the depth to commit */
   uint64_t depthbuf;
   uint64_t pixel_counter;  /* Device pointer to actual pixel count (0 = use num_pixels) */
   uint64_t discard_mask;   /* One byte per shaded pixel, set by `discard` (0 = none) */
   uint64_t coverage;       /* One byte per slot; helper lanes are zero */
   uint64_t reject;         /* Out: triangle that discarded, per pixel (0 = unused) */
   uint64_t resolved;       /* Out: marks pixels that have been written */
   uint32_t reject_layers;
   uint32_t reject_pass;    /* Which reject slot this pass writes */
   uint32_t depth_write;
   uint32_t depth_key_invert;
   uint32_t width;
   uint32_t fs_out_stride;
   uint32_t num_pixels;
   uint32_t color_encoding; /* enum cp_color_encoding */
   struct cp_blend_desc blend;
   /* Samples per pixel, and the distance between one sample's plane of the
    * colour attachment and the next, in bytes. */
   uint32_t num_samples;
   uint32_t sample_stride;   /* bytes between colour sample planes */
   uint32_t height;          /* with width, the stride between visbuf planes */
};

/*
 * Composite a whole A-buffer into the colour attachment (CUDAPIPE_ABUFFER).
 *
 * One thread per covered pixel. The pixel's run of fragments is already sorted
 * ascending by primitive, which is the order the peel loop composites in, so
 * the thread reads the attachment once, blends the run in place and writes it
 * back once. Reading and writing the attachment once per pixel rather than
 * once per layer is most of what this replaces the peel loop to get.
 *
 * `shade_slot` is the map the merge left behind: for A-buffer slot s, which of
 * the quad stream's shading slots holds that fragment. 0xFFFFFFFF for a slot
 * the merge could not place, which is skipped rather than read.
 */
struct cp_abuf_composite_args {
   uint64_t offsets;        /* uint32 per pixel: first A-buffer slot */
   uint64_t counts;         /* uint32 per pixel: length of the run */
   uint64_t shade_slot;     /* uint32 per A-buffer slot: shading slot */
   uint64_t fs_out;         /* Fragment shader colour output, per shading slot */
   uint64_t coverage;       /* uint8 per shading slot; helper lanes are zero */
   uint64_t discard_mask;   /* uint8 per shading slot, set by `discard` (0 = none) */
   uint64_t color_out;
   uint64_t list;           /* uint32 per covered pixel */
   uint64_t list_count;     /* uint32: entries in `list` */
   uint32_t fs_out_stride;
   uint32_t num_slots;      /* bound on a shading slot index */
   uint32_t capacity;       /* bound on an A-buffer slot index */
   uint32_t color_encoding; /* enum cp_color_encoding */
   /* Layers to composite, 0 for all of them. Only for reproducing the peel
    * loop's own CP_BLEND_LAYERS truncation when something needs comparing
    * against it; the point of this path is that it has no such cap. */
   uint32_t max_layers;
   struct cp_blend_desc blend;
   /*
    * Pass-episode resolution. Shading ran per segment into dense per-segment
    * arrays, so a global shading slot out of shade_slot — quad * 4 + lane —
    * resolves through the quad's segment and its dense position to that
    * segment's own buffers. All null outside an episode, where fs_out,
    * coverage and discard_mask above are the single segment's arrays.
    */
   uint64_t quad_seg;       /* uint8 per quad: segment index */
   uint64_t quad_dense;     /* uint32 per quad: dense position in segment */
   uint64_t seg_desc;       /* struct cp_seg_desc[segments] */
};

#ifdef __CUDACC__
/* Resolve one primitive once, then index its three vertices and slots from the
 * returned base. `refs` and every entry are device addresses; the host never
 * dereferences them. Null refs are retired stable slots and are rejected by
 * callers before any vertex load. */
static __device__ __forceinline__ const unsigned char *
cp_primitive_base(uint64_t refs, uint64_t contiguous, uint32_t primitive,
                  uint32_t vertex_stride)
{
   if (refs) {
      uint64_t base = ((const uint64_t *)(uintptr_t)refs)[primitive];
      return (const unsigned char *)(uintptr_t)base;
   }
   return (const unsigned char *)(uintptr_t)contiguous +
          (size_t)primitive * 3u * vertex_stride;
}
#endif

/* One pass-episode segment's shading arrays, for the composite. */
struct cp_seg_desc {
   uint64_t fs_out;
   uint64_t coverage;       /* uint8 per dense slot; 0 = shader has none */
   uint64_t discard;        /* uint8 per dense slot; 0 = discards nothing */
   uint32_t fs_out_stride;
   uint32_t num_slots;      /* bound on a dense slot index */
   uint32_t global_slots;   /* arrays use the original quad slot directly */
   uint32_t pad;
};

/*
 * One segment of a merged shading group, for the interpolator. Segments with
 * the same shading identity — same shaders, same constant-buffer count —
 * shade in one launch over the group's contiguous slice of the grouped quad
 * list, and the interpolator resolves each quad's own segment by its global
 * primitive id through this table (sorted by prim_base, the same search the
 * quad bucketing uses). The fields are exactly what the launch-wide
 * arguments can no longer be when one launch spans segments: the segment's
 * clipped vertex stream, its slice table, and where its fs-UBO rows landed
 * in the group's concatenated table.
 */
struct cp_seg_range {
   uint64_t positions;      /* the segment's contiguous fallback stream */
   uint64_t prim_refs;      /* the segment's primitive-reference table */
   uint64_t draw_slices;    /* struct cp_draw_slice[num_draw_slices], or 0 */
   uint32_t num_draw_slices;
   uint32_t prim_base;      /* first episode-global primitive slot */
   uint32_t prim_end;       /* exclusive end of this segment's slot range */
   uint32_t row_base;       /* first row in the group's concatenated tables */
   uint32_t prim_shift;
};

/*
 * Bucketing the episode's quad stream by segment: `count` walks the quads,
 * resolves each to its segment by the global primitive id, counts per
 * segment and records the segment per quad; `scatter` then places each
 * quad's index into the grouped list at its segment's base. Between the two
 * the host has read the counts back (in the episode's one drain) and
 * computed the bases.
 */
struct cp_abuf_seg_args {
   uint64_t quad_prim;      /* uint32 per quad: episode-global primitive */
   uint64_t seg_prim_base;  /* uint32 per segment: first primitive slot */
   uint64_t seg_counts;     /* uint32 per segment: quads (atomic) */
   uint64_t quad_seg;       /* uint8 per quad: out (count) / in (scatter) */
   uint64_t num_quads_dev;  /* uint32*: the quad total (count reads it) */
   uint32_t nsegs;
   uint32_t num_quads;      /* count: grid bound; scatter: exact total */
   uint32_t warp_aggregate;
   uint32_t pad;
   uint64_t seg_cursor;     /* uint32 per segment: atomic (scatter) */
   uint64_t seg_base;       /* uint32 per segment: dense base (scatter) */
   uint64_t grouped;        /* uint32 per quad: quad indices by segment */
   uint64_t quad_dense;     /* uint32 per quad: dense position in segment */
   uint64_t seg_group;      /* uint8 per segment, optional */
   uint64_t group_base;     /* uint32 per group, optional */
};

/*
 * Tile shader census: how many distinct fragment shaders would land in one
 * tile's bin, and whether they arrive in contiguous runs of submission order.
 *
 * This changes no rendering. It reads the A-buffer's own quad stream, which
 * already carries every depth-passing fragment's 2x2 block and its
 * episode-global primitive id, so the answer is exact coverage rather than a
 * bounding-box estimate. A tile renderer's bin for this episode is exactly
 * the set of primitives whose quads land in that tile.
 *
 * Two numbers per tile, because they decide different halves of the design:
 * the distinct shader count says whether one kernel would have to be able to
 * call many shaders, and the disjointness of each shader's [min, max]
 * primitive range says whether the tile could instead be split into that many
 * ordered passes without breaking blend order.
 */
#define CP_TILE_CENSUS_MAX_SHADERS 64
#define CP_TILE_CENSUS_BINS        65   /* 0..63 shaders, 64 = more */
#define CP_TILE_CENSUS_LOG         32   /* log2 buckets for the two spreads */
/*
 * Histogram layout, all unsigned long long, all accumulated on the device:
 *   [0]                      shaders-per-tile, four words per bin
 *   [SHADED]                 log2(shaded fragments per tile), tiles
 *   [REFS]                   log2(primitive references per tile), tiles
 *   [GLOBALS + 0..5]         max refs/tile, max shaded/tile, total refs,
 *                            total shaded, tiles with refs, tiles shaded
 */
#define CP_TILE_CENSUS_SHADED  (CP_TILE_CENSUS_BINS * 4)
#define CP_TILE_CENSUS_REFS    (CP_TILE_CENSUS_SHADED + CP_TILE_CENSUS_LOG)
#define CP_TILE_CENSUS_GLOBALS (CP_TILE_CENSUS_REFS + CP_TILE_CENSUS_LOG)
#define CP_TILE_CENSUS_WORDS   (CP_TILE_CENSUS_GLOBALS + 8)

struct cp_tile_census_args {
   uint64_t quad_prim;      /* uint32 per quad: episode-global primitive */
   uint64_t quad_block;     /* uint32 per quad: 2x2 block index */
   uint64_t num_quads_dev;  /* uint32*: the exact quad total */
   uint64_t seg_prim_base;  /* uint32 per segment: first primitive slot */
   uint64_t seg_shader;     /* uint8 per segment: distinct-fs index */
   uint64_t tile_mask;      /* uint64 per tile: shaders present */
   uint64_t tile_quads;     /* uint32 per tile */
   uint64_t tile_smin;      /* uint32 per (tile, shader): lowest primitive */
   uint64_t tile_smax;      /* uint32 per (tile, shader): highest */
   uint64_t tile_refs;      /* uint32 per tile: primitive references binned */
   uint64_t hist;           /* unsigned long long[CP_TILE_CENSUS_WORDS] */
   uint64_t seg_seq;        /* uint32 per segment: pass-global draw order */
   uint64_t visbuf;         /* the opaque source: one winner per pixel */
   uint32_t width;
   uint32_t height;
   uint32_t num_quads;      /* bound on the quad array */
   uint32_t nsegs;
   uint32_t quad_width;     /* 2x2 blocks per framebuffer row */
   uint32_t tile;           /* tile edge in pixels */
   uint32_t tiles_x;
   uint32_t tiles_y;
   uint32_t nshaders;
   uint32_t pad;
};

struct cp_abuf_seg_prefix_args {
   uint64_t seg_counts;
   uint64_t seg_group;
   uint64_t seg_base;
   uint64_t group_base;
   uint64_t group_counts;
   uint32_t nsegs;
   uint32_t ngroups;
};

struct cp_abuf_shade_count_args {
   uint64_t count;
   uint64_t slots;
};

#define CP_OPAQUE_TILE_SIZE 32u
#define CP_MAX_OPAQUE_TILE_REFS 2000000u

struct cp_opaque_tile_ref {
   uint32_t global_prim;
   uint16_t segment;
   uint16_t flags;
};

struct cp_opaque_tile_build_args {
   struct cp_rasterize_args rast;
   uint64_t tile_counts;
   uint64_t tile_offsets;
   uint64_t tile_cursors;
   uint64_t tile_refs;
   uint64_t overflow;
   uint32_t tiles_x;
   uint32_t tiles_y;
   uint32_t segment;
   uint32_t capacity;
};

struct cp_opaque_tile_raster_args {
   uint64_t rast_args;
   uint64_t tile_counts;
   uint64_t tile_offsets;
   uint64_t tile_refs;
   uint64_t overflow;
   uint32_t num_segments;
   uint32_t tiles_x;
   uint32_t tiles_y;
   uint32_t width;
   uint32_t height;
};

/*
 * Everything the sampler needs to know about one texture.
 *
 * lavapipe hands the driver an lp_image_descriptor whose `functions` field is
 * whatever this driver's create_texture_handle() returned, so we point it at
 * one of these instead of at llvmpipe's JIT-compiled sample functions. That
 * keeps us out of llvmpipe's internal descriptor layout entirely.
 */
struct cp_texture_info {
   uint64_t base;           /* Texture data (level 0) */
   uint32_t width, height, depth;
   uint32_t format;         /* enum pipe_format */
   uint32_t target;         /* enum pipe_texture_target */
   uint32_t first_level, last_level;
   uint32_t first_layer;    /* Views can start partway into an array */
   uint32_t row_stride[CP_MAX_TEXTURE_LEVELS];
   uint32_t img_stride[CP_MAX_TEXTURE_LEVELS];
   uint32_t mip_offset[CP_MAX_TEXTURE_LEVELS];
   /* Decoded from `format` on the host so the kernel doesn't need a format
    * table: see enum cp_texel_encoding. */
   uint32_t encoding;
   uint32_t blocksize;      /* Bytes per texel (uncompressed) or per block */
   uint32_t is_srgb;
};

/* How the sampler should turn raw bytes into an RGBA float. */
enum cp_texel_encoding {
   CP_TEXEL_UNSUPPORTED = 0,
   CP_TEXEL_R8G8B8A8_UNORM,
   CP_TEXEL_B8G8R8A8_UNORM,
   CP_TEXEL_R8G8B8X8_UNORM,
   CP_TEXEL_B8G8R8X8_UNORM,
   CP_TEXEL_R8G8_UNORM,
   CP_TEXEL_R8_UNORM,
   CP_TEXEL_R8G8B8A8_SNORM,
   CP_TEXEL_R16G16B16A16_UNORM,
   CP_TEXEL_R16G16B16A16_FLOAT,
   CP_TEXEL_R32G32B32A32_FLOAT,
   CP_TEXEL_R32G32B32_FLOAT,
   CP_TEXEL_R32G32_FLOAT,
   CP_TEXEL_R32_FLOAT,
   CP_TEXEL_R5G6B5_UNORM,
   CP_TEXEL_B5G5R5A1_UNORM,
   CP_TEXEL_A1R5G5B5_UNORM,
   CP_TEXEL_A1B5G5R5_UNORM,
   CP_TEXEL_B4G4R4A4_UNORM,
   CP_TEXEL_A4R4G4B4_UNORM,
   CP_TEXEL_A4B4G4R4_UNORM,
   CP_TEXEL_R4G4B4A4_UNORM,
   CP_TEXEL_R11G11B10_FLOAT,
   CP_TEXEL_R9G9B9E5_FLOAT,
   CP_TEXEL_A8R8G8B8_UNORM,
   CP_TEXEL_X8R8G8B8_UNORM,
   CP_TEXEL_R8G8B8_UNORM,
   CP_TEXEL_R16_SFLOAT,
   CP_TEXEL_R16G16_SFLOAT,
   CP_TEXEL_R16G16_UNORM,
   CP_TEXEL_A2B10G10R10_UNORM,
   CP_TEXEL_R32_SINT,
   CP_TEXEL_R16_SINT,
   CP_TEXEL_DXT1_RGB,
   CP_TEXEL_DXT1_RGBA,
   CP_TEXEL_DXT3_RGBA,
   CP_TEXEL_DXT5_RGBA,
};

/* What kind of texture a sample targets, and how to sample it. Passed to the
 * sampler as the `flags` argument. */
enum cp_tex_target {
   CP_TEX_1D = 0,
   CP_TEX_2D,
   CP_TEX_3D,
   CP_TEX_CUBE,
   CP_TEX_1D_ARRAY,
   CP_TEX_2D_ARRAY,
   CP_TEX_CUBE_ARRAY,
};

#define CP_TEX_TARGET_MASK 0xF
/* Coordinates are integer texels and the level is explicit (texelFetch). */
#define CP_TEX_FETCH       0x10
/* textureLod: explicit_lod is the level, derivatives are not consulted. */
#define CP_TEX_LOD         0x20
/* texture(..., bias): explicit_lod is added to the computed level. */
#define CP_TEX_BIAS        0x40

/* Mirrors the subset of pipe_sampler_state the sampler actually uses. */
struct cp_sampler_info {
   uint32_t wrap_s, wrap_t, wrap_r;
   uint32_t min_img_filter, mag_img_filter, min_mip_filter;
   uint32_t unnormalized_coords;
   float min_lod, max_lod, lod_bias;
   /* Ratio of the longest to the shortest footprint axis the sampler may take
    * separate samples along. 1 (or 0) means isotropic filtering. */
   float max_anisotropy;
   float border_color[4];
   uint32_t compare_enable;
   uint32_t reduction_mode;
   uint32_t non_seamless_cube;
};

/*
 * The three fields of a descriptor row that CUDA code reads: cp_sampler.cu and
 * the generated shader PTX both address them by these byte offsets.
 *
 * They are this driver's own layout, defined and asserted against
 * struct cpvk_descriptor in cp_shader_abi.h, which is the only place to change
 * them. The numbers were originally lavapipe's, because lavapipe used to be
 * the Vulkan front end that produced these descriptors and the rows were its
 * structures; nothing outside this directory produces one now.
 *
 * They are here rather than in cp_shader_abi.h because this header is
 * stringified into the kernel sources at build time and NVRTC compiles it with
 * nothing else available.
 */
#define CP_DESC_IMAGE_BASE_OFFSET      0   /* cpvk_descriptor.base */
#define CP_DESC_IMAGE_FUNCTIONS_OFFSET 48  /* cpvk_descriptor.texture_info */
#define CP_DESC_SAMPLER_INDEX_OFFSET   28  /* .sampler_index_or_img_stride */

/*
 * Persistent GPU-visible state. Written by CPU on pipe state changes
 * (between draws), read by all GPU kernels during draws. Lives in
 * managed memory for the lifetime of the context.
 */
/*
 * Adaptive rasterizer thresholds and queue sizes.
 *
 * A triangle goes to the stage that can afford its bounding box: one thread
 * up to CP_SMALL_THRESHOLD pixels, one warp up to CP_MEDIUM_THRESHOLD, and a
 * block per tile above that. The thresholds are what decides how much of the
 * machine a triangle gets, so they are the rasterizer's main tuning knob and
 * both can be overridden with -D at NVRTC time; see cp_kernels.c.
 *
 * Getting the small one wrong is expensive in one direction only. Too low
 * costs a queue round trip on a triangle a single thread could have finished;
 * too high runs a whole triangle on one lane while the other 127 SMs idle,
 * and since a warp cannot retire until its slowest lane does, one large
 * triangle also holds up the 31 small ones sharing its warp.
 */
#ifndef CP_SMALL_THRESHOLD
#define CP_SMALL_THRESHOLD   128
#endif
#ifndef CP_MEDIUM_THRESHOLD
#define CP_MEDIUM_THRESHOLD  1536
#endif

/*
 * Points take a lower medium threshold than triangles, and the reason is a
 * measurement rather than a preference.
 *
 * A sprite between the two thresholds gets one warp in stage 2 however many
 * pixels it covers, which is the defect the large-point path already fixed
 * above CP_MEDIUM_THRESHOLD. particlesystem's fire spends most of its frames
 * with about thirty sprites sitting in that gap, and they cost ~60 us a peel
 * pass while the other 1500 warps of stage 2's grid have nothing to do, since
 * a kernel retires with its slowest warp. Sending them to stage 3 instead is
 * worth 28% of the sample.
 *
 * Why points and not triangles: a point is resolved by the same four-comparison
 * square test in both stages and carries a single depth, so which stage takes
 * it cannot change a pixel. A triangle's depth goes through the interpolation
 * each stage writes separately, and the two are not bit-identical -- moving
 * triangles across the same boundary flips two pixels of gltfscenerendering,
 * where an atomicMin tie resolves the other way. That difference is worth
 * fixing on its own, but it is not this knob's to carry.
 *
 * At CP_SMALL_THRESHOLD every point reaching stage 2 goes on to stage 3, which
 * is where the sweep flattens; it is a separate constant so the gap can be
 * reopened without touching the triangle path.
 */
#ifndef CP_POINT_THRESHOLD
#define CP_POINT_THRESHOLD   CP_SMALL_THRESHOLD
#endif
#define CP_TILE_SIZE         64

/*
 * Whether stage 3 walks the whole tile or only the part of it the primitive's
 * bounding box reaches.
 *
 * A stage 3 block is a (primitive, tile) pair, and the tiles were enumerated
 * from the primitive's bounding box — so the last tile of each row and column
 * is usually only partly covered, and a sprite two tiles across covers a
 * quarter of each of its four tiles on average. Walking all 64x64 of them
 * tests coverage on pixels the box already excludes.
 *
 * The bounding box in struct tri_setup is a superset of coverage by
 * construction and is already clamped to the clip rectangle, for points and
 * triangles alike, so intersecting it with the tile costs no new arithmetic
 * and cannot drop a covered sample. Defined as a switch rather than assumed so
 * that one binary can be A/B'd: CUDAPIPE_TILE_BOUND=0 in the environment
 * compiles the full-tile walk back, the same way the thresholds above are
 * swept. See cp_kernels.c.
 */
#ifndef CP_TILE_BOUND
#define CP_TILE_BOUND        1
#endif

/* Bound on the per-point rasterization loop, and the point size clamp. */
#define CP_MAX_POINT_SIZE    256.0f

/* How deep a pile of blended fragments one draw will composite. */
#define CP_BLEND_LAYERS      256

/*
 * How many peel passes may be launched between convergence checks.
 *
 * Asking whether a pass composited anything means the host reads memory a
 * kernel just wrote, which drains the device. The check interval doubles from
 * one so that a draw converging on pass 2 is still caught on pass 2, and this
 * caps it so that a draw converging just after a check wastes at most this
 * many further passes rather than up to as many as it has already run.
 */
#define CP_PEEL_CHECK_MAX    16

/* Multisampling. 1x, 4x and 8x are advertised, and the visibility, depth and
 * colour buffers all hold the samples plane after plane: sample s of pixel p
 * lives at s * width * height + p. The coverage byte carries one bit per
 * sample, so eight is the most this layout takes without widening it. */
#define CP_MAX_SAMPLES       8

/* Distinct primitives one 2x2 block will shade. With multisampling a block
 * covers 16 samples, so more triangles can meet inside it than without. */
#define CP_MAX_BLOCK_TRIS    8

struct cp_resolve_msaa_args {
   uint64_t src;
   uint64_t dst;
   uint32_t width, height;
   uint32_t src_stride, dst_stride;
   uint32_t sample_stride;
   uint32_t num_samples;
   int32_t encoding;
};

struct cp_blit_linear_args {
   uint64_t src;
   uint64_t dst;
   uint32_t src_width;
   uint32_t src_height;
   uint32_t dst_width;
   uint32_t dst_height;
   uint32_t src_stride;
   uint32_t dst_stride;
   uint32_t src_layer_stride;
   uint32_t dst_layer_stride;
   uint32_t layers;
   int32_t src_encoding;
   int32_t dst_encoding;
   uint32_t filter_linear;
};
#define CP_MAX_NONTRIVIAL    1000000
#define CP_MAX_HUGE_TILES    2000000

/*
 * Immutable screen-space setup made once by stage 2 and consumed by every
 * stage-3 tile of the same huge primitive. uint8_t, rather than C/C++ bool,
 * keeps the host allocation ABI explicit while preserving the old layout.
 */
struct cp_tri_setup {
   float sx0, sy0, sx1, sy1, sx2, sy2;
   float ndc_z0, ndc_z1, ndc_z2;
   float inv_area;
   uint8_t e0_top_left, e1_top_left, e2_top_left;
   int32_t ix_min, iy_min, ix_max, iy_max;
   uint8_t is_point;
   float pt_x0, pt_y0, pt_x1, pt_y1;
};

struct cp_setup_cache_entry {
   uint32_t tri_id;             /* original primitive id for fragment output */
   struct cp_tri_setup setup;
};

struct cp_tile_pair {
   uint32_t tri_id;             /* primitive id, or CP_TILE_SETUP_TAG | index */
   uint16_t tile_x;
   uint16_t tile_y;
};

#define CP_PRIM_ID_LIMIT          (1u << 30)
#define CP_TILE_SETUP_TAG         0x80000000u
#define CP_TILE_SETUP_INDEX_MASK  0x7fffffffu
#define CP_SETUP_CACHE_CAPACITY   1024u
#define CP_SETUP_CACHE_MIN_TILES  4u

#ifdef __CUDACC__
static_assert(sizeof(struct cp_tri_setup) == 80, "cp_tri_setup ABI");
static_assert(sizeof(struct cp_setup_cache_entry) == 84, "setup-cache ABI");
static_assert(CP_PRIM_ID_LIMIT <= CP_TILE_SETUP_TAG, "primitive/tag range collision");
static_assert(CP_SETUP_CACHE_CAPACITY <= CP_TILE_SETUP_INDEX_MASK,
              "setup-cache index does not fit tile tag");
#else
_Static_assert(sizeof(struct cp_tri_setup) == 80, "cp_tri_setup ABI");
_Static_assert(sizeof(struct cp_setup_cache_entry) == 84, "setup-cache ABI");
_Static_assert(CP_PRIM_ID_LIMIT <= CP_TILE_SETUP_TAG, "primitive/tag range collision");
_Static_assert(CP_SETUP_CACHE_CAPACITY <= CP_TILE_SETUP_INDEX_MASK,
               "setup-cache index does not fit tile tag");
#endif

/*
 * What a pass is allowed to assume about the queues below.
 *
 * Which primitives land in which queue is a pure function of the geometry and
 * the clip rectangle, and a peeled draw changes neither between its passes —
 * only peel_next, which is read inside emit_fragment after coverage is already
 * decided. So a draw that runs the full CP_BLEND_LAYERS passes rebuilds
 * byte-identical queues 256 times.
 *
 * The first pass of such a draw therefore builds them and marks each queue
 * entry it hands to stage 3, and every later pass reuses what is there: the
 * counters are not reset, stage 1 stops appending, and stage 2 skips a marked
 * entry without even doing its setup. CP_QUEUE_FILL is the unmarked build the
 * kill switch and every non-peeled draw still take.
 */
#define CP_QUEUE_FILL    0u   /* build the queues, mark nothing */
#define CP_QUEUE_BUILD   1u   /* build them and mark the stage 3 entries */
#define CP_QUEUE_REUSE   2u   /* already valid; skip everything that fills them */

/* Set on a nontrivial-queue entry that stage 2 decomposed into tiles, so that
 * a reusing pass can skip it. Triangle ids are indices into a draw's clipped
 * primitive list, so the top bit is free. The same central tag marks a setup
 * index in cp_tile_pair, but the words live in disjoint queue formats: stage 2
 * consumes CP_NT_HUGE and writes CP_TILE_SETUP_TAG for stage 3. */
#define CP_NT_HUGE       CP_TILE_SETUP_TAG

struct cp_rast_queues {
   uint64_t nontrivial;        /* Device ptr to uint32_t[CP_MAX_NONTRIVIAL] */
   uint64_t nontrivial_count;  /* Device ptr to atomic uint32_t */
   uint64_t huge_tiles;        /* Device ptr to cp_tile_pair[CP_MAX_HUGE_TILES] */
   uint64_t huge_count;        /* Device ptr to atomic uint32_t */
   uint64_t setup_cache;       /* Device ptr to cp_setup_cache_entry[]; 0 = classic */
   uint64_t setup_count;       /* Device ptr to atomic uint32_t */
   uint32_t setup_capacity;    /* zero selects the complete classic fallback */
   uint32_t mode;              /* CP_QUEUE_* above */
};

#ifdef __CUDACC__
static_assert(sizeof(struct cp_rast_queues) == 56, "raster queue launch ABI");
#else
_Static_assert(sizeof(struct cp_rast_queues) == 56, "raster queue launch ABI");
#endif

#define CP_MAX_VERTEX_ELEMENTS_VF 16
#define CP_MAX_VERTEX_BUFFERS_VF 16

/*
 * How a vertex attribute's components are converted on the way into the
 * shader's 16 byte input slot.
 *
 * Vulkan hands the shader every component in its own 32 bit slot however
 * narrow it is in memory, so an R8G8B8A8_UINT attribute arrives as four uints
 * and not as one packed word. Anything that is not already 32 bits per
 * component therefore has to be expanded during the fetch — copying the raw
 * bytes leaves all four components in the first slot, which reads as a value
 * up to 2^32 where a small integer was meant.
 */
enum cp_vf_conv {
   CP_VF_CONV_COPY32 = 0,   /* 32 bits per component: the bytes are already right */
   CP_VF_CONV_UINT,         /* integer, zero extended */
   CP_VF_CONV_SINT,         /* integer, sign extended */
   CP_VF_CONV_UNORM,        /* unsigned normalised -> float */
   CP_VF_CONV_SNORM,        /* signed normalised -> float */
   CP_VF_CONV_USCALED,      /* unsigned integer -> float */
   CP_VF_CONV_SSCALED,      /* signed integer -> float */
   CP_VF_CONV_FLOAT16,      /* half -> float */
};

struct cp_vertex_fetch_args {
   uint64_t output;
   uint64_t index_buffer;
   uint64_t vb_bases[CP_MAX_VERTEX_BUFFERS_VF];
   uint32_t elem_vb_idx[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_src_offset[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_src_stride[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_attr_size[CP_MAX_VERTEX_ELEMENTS_VF];
   /* The format as the fetch has to expand it: how many components it
    * supplies, how wide each one is in the vertex buffer, one of
    * enum cp_vf_conv, and where each destination component reads from in
    * memory (four nibbles, low nibble first) for formats whose channels are
    * not in RGBA order. CP_VF_CONV_COPY32 ignores all of these and takes the
    * verbatim-copy path. */
   uint32_t elem_nr_chan[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_chan_bytes[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_conv[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_swizzle[CP_MAX_VERTEX_ELEMENTS_VF];
   /* Vulkan fills the components a vertex format does not supply with
    * (0, 0, 0, 1), so an attribute with fewer than four components needs a
    * one written into its w slot. Holds the bit pattern of that one, which is
    * 1.0f for float formats and integer 1 for the rest, or zero when the
    * format already supplies all four components. */
   uint32_t elem_fill_w[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_instance_divisor[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t num_elements;
   uint32_t num_verts;
   uint32_t vs_in_stride;
   uint32_t index_size;
   uint32_t first_vertex;
   uint32_t start_instance;
   uint64_t vertex_ids;
   uint64_t instance_ids;
   /*
    * Assembled vertices in one instance, when the ids are derived here rather
    * than read from `vertex_ids`. An instanced draw replays the same index
    * range once per instance, so vertex v of the draw is vertex v % this of
    * instance v / this — which is the whole of what the host used to compute
    * and upload per vertex. Zero means the caller supplied the arrays.
    */
   uint32_t verts_per_instance;
   /*
    * Where to publish the ids for the vertex shader, which reads gl_VertexIndex
    * and gl_InstanceIndex out of arrays indexed by thread. Either may be zero
    * when the shader does not read that one.
    */
   uint64_t out_vertex_ids;
   uint64_t out_instance_ids;
   /*
    * The batch's draws, one slice each, and how many there are. Zero means
    * this is not a batch and every line above reads exactly as it did before
    * batching existed — which is what makes CUDAPIPE_BATCH_MAX=1 a real
    * check rather than a different code path that happens to agree.
    */
   uint64_t draw_slices;   /* const struct cp_draw_slice * */
   uint32_t num_draw_slices;
   /*
    * Where to publish each assembled vertex's row in the batch's per-draw
    * tables, for the vertex shader to pick its uniform bindings out of. The
    * search below already knows the answer, so the shader does not repeat it.
    * Zero when the shader has no use for it.
    */
   uint64_t out_batch_rows;
   /*
    * Per-draw vertex-buffer bases: CP_VB_TABLE_STRIDE uint64 per merged draw,
    * one resolved base address per vertex *element* (buffer data plus
    * buffer_offset — the element's own src_offset stays in elem_src_offset).
    * Zero for an unbatched draw, where vb_bases above stands. A batch always
    * carries it, because a deferred draw's bindings may be rebound before the
    * batch runs — deferral is the hazard, not merging.
    */
   uint64_t elem_bases;    /* const uint64_t *, rows of CP_VB_TABLE_STRIDE */
};

/* uint64 entries per draw in the per-draw vertex-buffer base table. */
#define CP_VB_TABLE_STRIDE 16

#endif /* CP_RAST_TYPES_H */
