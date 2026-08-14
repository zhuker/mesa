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

struct cp_clear_args {
   uint64_t target;         /* Device pointer to buffer to clear */
   uint32_t width;
   uint32_t height;
   uint32_t stride;         /* Row stride in bytes */
   uint32_t clear_value[4]; /* Clear color as 4x u32 */
   uint32_t pixel_size;     /* Bytes per pixel */
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
   uint64_t positions;      /* Input: screen-space positions (float4 per vertex) */
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
   uint32_t num_triangles;
   uint32_t num_slots;      /* Position plus varyings, i.e. num_varyings + 1 */
   uint32_t max_triangles;  /* Capacity of `out`, in triangles */
   uint32_t pad;
};

#define CP_MAX_CLIP_SLOTS 16

/* Shader kernel argument slots. 0..7 are the fixed stage inputs and 18.. are
 * the uniform/descriptor buffers; the gap in between is free. */
#define CP_ARG_SLOT_DISCARD 8

/*
 * Draw batching. Several consecutive draws that differ in nothing but their
 * vertex-stage uniform bindings are submitted as one, so that the grids are
 * sized to the batch rather than to a draw of a dozen triangles. The vertex
 * shader then has to pick its own draw's bindings out of a table:
 *
 *   draw  = thread_id / *(uint32_t *)args[CP_ARG_SLOT_BATCH_DIV]
 *   base  = ((void **)args[CP_ARG_SLOT_UBO_TABLE])[draw * CP_ARG_UBO_STRIDE + i]
 *
 * Both slots are always filled, so the generated code has no branch and no
 * batched/unbatched variant. A single draw sets the table to `&args[18]` and
 * the divisor to 0xFFFFFFFF, which makes the expression above compute exactly
 * the args[18 + i] the shader used to load.
 *
 * Only the vertex stage reads them. Draws whose *fragment* bindings differ are
 * not merged at all, so the fragment shader keeps loading args[18 + i] and its
 * generated code is untouched.
 */
#define CP_ARG_SLOT_UBO_TABLE 9
#define CP_ARG_SLOT_BATCH_DIV 10
/* Entries per draw in the table at CP_ARG_SLOT_UBO_TABLE; matches
 * CP_MAX_CONST_BUFFERS and the 18.. layout it stands in for. */
#define CP_ARG_UBO_STRIDE 16
#define CP_ARG_UBO_BASE 18

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
};

/*
 * Gathers the fragment shader's inputs for every covered pixel.
 *
 * Covered pixels are compacted into pixel_list so the fragment shader can be
 * launched over just those, one thread per pixel.
 */
struct cp_fs_interp_args {
   uint64_t visbuf;
   uint64_t positions;      /* Clip-space float4, 3 per triangle */
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
};

/*
 * Offsets into lavapipe's descriptors. These are the only two things we read
 * out of structures we don't own, and cp_context.c static-asserts both against
 * offsetof() so a layout change upstream breaks the build rather than the
 * rendering.
 */
#define CP_DESC_IMAGE_BASE_OFFSET      0   /* lp_image_descriptor.texture.base */
#define CP_DESC_IMAGE_FUNCTIONS_OFFSET 48  /* lp_image_descriptor.functions */
#define CP_DESC_SAMPLER_INDEX_OFFSET   28  /* lp_sampler_descriptor.sampler_index */

/*
 * Persistent GPU-visible state. Written by CPU on pipe state changes
 * (between draws), read by all GPU kernels during draws. Lives in
 * managed memory for the lifetime of the context.
 */
struct cp_gpu_state {
   /* Vertex fetch */
   uint64_t vb_bases[16];
   uint32_t elem_vb_idx[16];
   uint32_t elem_src_offset[16];
   uint32_t elem_src_stride[16];
   uint32_t elem_attr_size[16];
   uint32_t elem_instance_divisor[16];
   uint32_t num_elements;
   uint32_t vs_in_stride;
   uint64_t index_buffer;
   uint32_t index_size;

   /* Shader UBOs */
   uint64_t vs_ubos[16];
   uint64_t fs_ubos[16];

   /* Viewport */
   float vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y;

   /* Rasterizer */
   uint32_t cull_mode;
   uint32_t front_face;

   /* Depth */
   uint32_t depth_test;
   uint32_t depth_func;
   uint32_t depth_write;
   uint32_t depth_key_invert;

   /* Blend */
   uint32_t blend_enable;
   uint32_t rgb_src_factor, rgb_dst_factor, rgb_func;
   uint32_t alpha_src_factor, alpha_dst_factor, alpha_func;
   uint32_t colormask;

   /* Framebuffer */
   uint64_t color_attachment;
   uint64_t visbuf;
   uint64_t depthbuf;
   uint32_t fb_width, fb_height;
   uint32_t color_encoding;

   /* Sampler */
   uint64_t sampler_table;

   /* VS/FS layout */
   uint32_t num_vs_outputs;
   uint32_t num_fs_inputs;
   int32_t fs_input_vs_slot[16];
};

/* Per-draw parameters passed as kernel arguments (by value, not device memory) */
struct cp_draw_params {
   uint64_t gpu_state;      /* Pointer to cp_gpu_state */
   uint64_t vs_input;       /* Arena offset: packed vertex attributes */
   uint64_t vs_output;      /* Arena offset: VS output buffer */
   uint64_t pixel_list;     /* Arena offset: covered pixel indices */
   uint64_t pixel_counter;  /* Arena offset: atomic pixel count */
   uint64_t fs_input;       /* Arena offset: interpolated FS inputs */
   uint64_t fs_output;      /* Arena offset: FS color output */
   uint64_t fs_deriv;       /* Arena offset: screen-space derivatives */
   uint64_t frag_coord;     /* Arena offset: fragment coordinates */
   uint32_t total_verts;
   uint32_t first_vertex;
   int32_t  index_bias;
   uint32_t start_instance;
   uint32_t instance_count;
   uint32_t num_vs_outputs;
   uint32_t fs_in_stride;
   uint32_t fs_out_stride;
   uint32_t max_pixels;
};

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
#define CP_MEDIUM_THRESHOLD  4096
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
#define CP_MAX_NONTRIVIAL    1000000
#define CP_MAX_HUGE_TILES    2000000

struct cp_tile_pair {
   uint32_t tri_id;
   uint16_t tile_x;
   uint16_t tile_y;
};

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
 * primitive list, so the top bit is free. */
#define CP_NT_HUGE       0x80000000u

struct cp_rast_queues {
   uint64_t nontrivial;        /* Device ptr to uint32_t[CP_MAX_NONTRIVIAL] */
   uint64_t nontrivial_count;  /* Device ptr to atomic uint32_t */
   uint64_t huge_tiles;        /* Device ptr to cp_tile_pair[CP_MAX_HUGE_TILES] */
   uint64_t huge_count;        /* Device ptr to atomic uint32_t */
   uint32_t mode;              /* CP_QUEUE_* above */
   uint32_t pad;
};

#define CP_MAX_VERTEX_ELEMENTS_VF 16
#define CP_MAX_VERTEX_BUFFERS_VF 16

struct cp_vertex_fetch_args {
   uint64_t output;
   uint64_t index_buffer;
   uint64_t vb_bases[CP_MAX_VERTEX_BUFFERS_VF];
   uint32_t elem_vb_idx[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_src_offset[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_src_stride[CP_MAX_VERTEX_ELEMENTS_VF];
   uint32_t elem_attr_size[CP_MAX_VERTEX_ELEMENTS_VF];
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
    * Assembled vertices in one draw of a batch. Batched draws share their
    * geometry entirely — same index range, same buffers — and differ only in
    * their vertex-stage uniforms, so vertex v of the launch is vertex
    * v % this of draw v / this and gathers exactly what the draw before it
    * did. Zero means this is not a batch, which is the single-draw path
    * unchanged.
    */
   uint32_t verts_per_draw;
};

#endif /* CP_RAST_TYPES_H */
