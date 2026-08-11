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
};

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
   uint64_t fs_in;          /* Out: interpolated varyings, per covered pixel */
   uint64_t frag_coord;     /* Out: float4 (x, y, z, 1/w) per covered pixel */
   /* Out: screen-space derivatives of each input, float4 per input slot per
    * pixel as (du/dx, dv/dx, du/dy, dv/dy). The sampler needs these to pick a
    * mip level, and this stage can compute them from the triangle directly. */
   uint64_t fs_deriv;
   uint32_t width, height;
   uint32_t vs_out_stride;  /* Bytes per vertex in vs_out */
   uint32_t fs_in_stride;   /* Bytes per pixel in fs_in */
   uint32_t num_fs_inputs;
   uint32_t max_pixels;
   /* Which vertex shader output slot feeds each fragment shader input slot,
    * matched by varying location on the host. -1 means nothing drives it. */
   int32_t input_vs_slot[CP_MAX_FS_INPUTS];
   float vp_scale_x, vp_scale_y, vp_trans_x, vp_trans_y;
};

struct cp_fs_writeback_args {
   uint64_t pixel_list;
   uint64_t fs_out;         /* Fragment shader colour output, per covered pixel */
   uint64_t color_out;
   uint64_t visbuf;         /* Source of the depth to commit */
   uint64_t depthbuf;
   uint64_t pixel_counter;  /* Device pointer to actual pixel count (0 = use num_pixels) */
   uint64_t discard_mask;   /* One byte per shaded pixel, set by `discard` (0 = none) */
   uint32_t depth_write;
   uint32_t depth_key_invert;
   uint32_t width;
   uint32_t fs_out_stride;
   uint32_t num_pixels;
   uint32_t color_encoding; /* enum cp_color_encoding */
   uint32_t blend_enable;
   /* pipe_blend_state factors/functions for the colour and alpha channels. */
   uint32_t rgb_src_factor, rgb_dst_factor, rgb_func;
   uint32_t alpha_src_factor, alpha_dst_factor, alpha_func;
   uint32_t colormask;
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

/* Mirrors the subset of pipe_sampler_state the sampler actually uses. */
struct cp_sampler_info {
   uint32_t wrap_s, wrap_t, wrap_r;
   uint32_t min_img_filter, mag_img_filter, min_mip_filter;
   uint32_t unnormalized_coords;
   float min_lod, max_lod, lod_bias;
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

/* Adaptive rasterizer thresholds and queue sizes */
#define CP_SMALL_THRESHOLD   999999
#define CP_MEDIUM_THRESHOLD  4096
#define CP_TILE_SIZE         64
#define CP_MAX_NONTRIVIAL    1000000
#define CP_MAX_HUGE_TILES    2000000

struct cp_tile_pair {
   uint32_t tri_id;
   uint16_t tile_x;
   uint16_t tile_y;
};

struct cp_rast_queues {
   uint64_t nontrivial;        /* Device ptr to uint32_t[CP_MAX_NONTRIVIAL] */
   uint64_t nontrivial_count;  /* Device ptr to atomic uint32_t */
   uint64_t huge_tiles;        /* Device ptr to cp_tile_pair[CP_MAX_HUGE_TILES] */
   uint64_t huge_count;        /* Device ptr to atomic uint32_t */
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
};

#endif /* CP_RAST_TYPES_H */
