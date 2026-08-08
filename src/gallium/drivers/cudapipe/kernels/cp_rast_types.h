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
   /* Viewport */
   float vp_x, vp_y, vp_w, vp_h;
   float vp_near, vp_far;
   /* Rasterizer state */
   uint32_t cull_mode;      /* 0=none, 1=front, 2=back */
   uint32_t front_face;     /* 0=CCW, 1=CW */
};

struct cp_resolve_args {
   uint64_t visbuf;
   uint64_t positions;
   uint64_t colors;
   uint64_t color_out;
   uint32_t width, height;
   float vp_x, vp_y, vp_w, vp_h;
   uint32_t color_stride;
};

#endif /* CP_RAST_TYPES_H */
