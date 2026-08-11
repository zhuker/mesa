/*
 * cudapipe fragment stage.
 *
 * The rasterizer leaves a visibility buffer holding the winning triangle per
 * pixel. These two kernels bracket the compiled fragment shader:
 *
 *   cp_fs_interpolate  compacts covered pixels and interpolates the vertex
 *                      shader's varyings at each one, producing the fragment
 *                      shader's input buffer
 *   <fragment shader>  runs as its own kernel, one thread per covered pixel
 *   cp_fs_writeback    blends the shader's colour output into the attachment
 */
#include "cp_rast_types.h"

#define VISBUF_EMPTY 0xFFFFFFFFFFFFFFFFULL

static __device__ __forceinline__ float
cp_edge(float ax, float ay, float bx, float by, float cx, float cy)
{
   return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

extern "C" __global__ void
cp_fs_interpolate(struct cp_fs_interp_args args)
{
   uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
   if (pixel >= args.width * args.height)
      return;

   const uint64_t *visbuf = (const uint64_t *)(uintptr_t)args.visbuf;
   uint64_t entry = visbuf[pixel];
   if (entry == VISBUF_EMPTY)
      return;

   /* Triangle index is stored complemented so atomicMin favours the last
    * primitive on ties; see PACK_VISBUF in cp_rasterize.cu. */
   uint32_t tri_id = ~(uint32_t)(entry & 0xFFFFFFFFu);

   const float4 *positions = (const float4 *)(uintptr_t)args.positions;
   uint32_t pos_stride = args.vs_out_stride / 16;
   if (pos_stride == 0) pos_stride = 1;
   float4 v0 = positions[(tri_id * 3 + 0) * pos_stride];
   float4 v1 = positions[(tri_id * 3 + 1) * pos_stride];
   float4 v2 = positions[(tri_id * 3 + 2) * pos_stride];

   float inv_w0 = 1.0f / v0.w;
   float inv_w1 = 1.0f / v1.w;
   float inv_w2 = 1.0f / v2.w;

   float sx0 = v0.x * inv_w0 * args.vp_scale_x + args.vp_trans_x;
   float sy0 = v0.y * inv_w0 * args.vp_scale_y + args.vp_trans_y;
   float sx1 = v1.x * inv_w1 * args.vp_scale_x + args.vp_trans_x;
   float sy1 = v1.y * inv_w1 * args.vp_scale_y + args.vp_trans_y;
   float sx2 = v2.x * inv_w2 * args.vp_scale_x + args.vp_trans_x;
   float sy2 = v2.y * inv_w2 * args.vp_scale_y + args.vp_trans_y;

   float ndc_z0 = v0.z * inv_w0;
   float ndc_z1 = v1.z * inv_w1;
   float ndc_z2 = v2.z * inv_w2;

   /* The rasterizer flips clockwise triangles so barycentrics come out
    * positive; mirror that here, and carry the swap through to the vertex
    * indices so varyings are fetched in the matching order. */
   int vidx[3] = { 0, 1, 2 };
   float area = cp_edge(sx0, sy0, sx1, sy1, sx2, sy2);
   if (area < 0.0f) {
      float t;
      t = sx1; sx1 = sx2; sx2 = t;
      t = sy1; sy1 = sy2; sy2 = t;
      t = ndc_z1; ndc_z1 = ndc_z2; ndc_z2 = t;
      t = inv_w1; inv_w1 = inv_w2; inv_w2 = t;
      vidx[1] = 2; vidx[2] = 1;
      area = -area;
   }
   if (area == 0.0f)
      return;

   float inv_area = 1.0f / area;
   float cx = (float)(pixel % args.width) + 0.5f;
   float cy = (float)(pixel / args.width) + 0.5f;

   float b0 = cp_edge(sx1, sy1, sx2, sy2, cx, cy) * inv_area;
   float b1 = cp_edge(sx2, sy2, sx0, sy0, cx, cy) * inv_area;
   float b2 = 1.0f - b0 - b1;

   uint32_t slot = atomicAdd((unsigned int *)(uintptr_t)args.counter, 1u);
   if (slot >= args.max_pixels)
      return;

   ((uint32_t *)(uintptr_t)args.pixel_list)[slot] = pixel;

   /* Perspective-correct weights: interpolate attribute/w, then divide by the
    * interpolated 1/w. */
   float persp0 = b0 * inv_w0;
   float persp1 = b1 * inv_w1;
   float persp2 = b2 * inv_w2;
   float inv_persp = 1.0f / (persp0 + persp1 + persp2);

   if (args.frag_coord) {
      float4 fc;
      fc.x = cx;
      fc.y = cy;
      fc.z = (b0 * ndc_z0 + b1 * ndc_z1 + b2 * ndc_z2) * 0.5f + 0.5f;
      fc.w = persp0 + persp1 + persp2;
      ((float4 *)(uintptr_t)args.frag_coord)[slot] = fc;
   }

   const char *vs_out = (const char *)(uintptr_t)args.vs_out;
   char *fs_in = (char *)(uintptr_t)args.fs_in + (size_t)slot * args.fs_in_stride;
   char *fs_deriv = args.fs_deriv
      ? (char *)(uintptr_t)args.fs_deriv +
        (size_t)slot * args.num_fs_inputs * 16 : 0;

   /* Perspective-correct weights one pixel to the right and one down, so the
    * varyings' screen-space derivatives fall out as plain differences. */
   float dx_b0 = cp_edge(sx1, sy1, sx2, sy2, cx + 1.0f, cy) * inv_area;
   float dx_b1 = cp_edge(sx2, sy2, sx0, sy0, cx + 1.0f, cy) * inv_area;
   float dx_p0 = dx_b0 * inv_w0, dx_p1 = dx_b1 * inv_w1;
   float dx_p2 = (1.0f - dx_b0 - dx_b1) * inv_w2;
   float dx_inv = 1.0f / (dx_p0 + dx_p1 + dx_p2);

   float dy_b0 = cp_edge(sx1, sy1, sx2, sy2, cx, cy + 1.0f) * inv_area;
   float dy_b1 = cp_edge(sx2, sy2, sx0, sy0, cx, cy + 1.0f) * inv_area;
   float dy_p0 = dy_b0 * inv_w0, dy_p1 = dy_b1 * inv_w1;
   float dy_p2 = (1.0f - dy_b0 - dy_b1) * inv_w2;
   float dy_inv = 1.0f / (dy_p0 + dy_p1 + dy_p2);

   for (uint32_t i = 0; i < args.num_fs_inputs && i < CP_MAX_FS_INPUTS; i++) {
      int32_t src = args.input_vs_slot[i];
      float4 value = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
      float4 deriv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

      if (src >= 0) {
         const float4 *a0 = (const float4 *)(vs_out +
            (size_t)(tri_id * 3 + vidx[0]) * args.vs_out_stride + src * 16);
         const float4 *a1 = (const float4 *)(vs_out +
            (size_t)(tri_id * 3 + vidx[1]) * args.vs_out_stride + src * 16);
         const float4 *a2 = (const float4 *)(vs_out +
            (size_t)(tri_id * 3 + vidx[2]) * args.vs_out_stride + src * 16);

         value.x = (a0->x * persp0 + a1->x * persp1 + a2->x * persp2) * inv_persp;
         value.y = (a0->y * persp0 + a1->y * persp1 + a2->y * persp2) * inv_persp;
         value.z = (a0->z * persp0 + a1->z * persp1 + a2->z * persp2) * inv_persp;
         value.w = (a0->w * persp0 + a1->w * persp1 + a2->w * persp2) * inv_persp;

         if (fs_deriv) {
            float ux = (a0->x * dx_p0 + a1->x * dx_p1 + a2->x * dx_p2) * dx_inv;
            float vx = (a0->y * dx_p0 + a1->y * dx_p1 + a2->y * dx_p2) * dx_inv;
            float uy = (a0->x * dy_p0 + a1->x * dy_p1 + a2->x * dy_p2) * dy_inv;
            float vy = (a0->y * dy_p0 + a1->y * dy_p1 + a2->y * dy_p2) * dy_inv;
            deriv = make_float4(ux - value.x, vx - value.y,
                                uy - value.x, vy - value.y);
         }
      }

      *(float4 *)(fs_in + i * 16) = value;
      if (fs_deriv)
         *(float4 *)(fs_deriv + i * 16) = deriv;
   }
}

/* pipe_blendfactor */
enum {
   CP_BLENDFACTOR_ONE = 1,
   CP_BLENDFACTOR_SRC_COLOR = 2,
   CP_BLENDFACTOR_SRC_ALPHA = 3,
   CP_BLENDFACTOR_DST_ALPHA = 4,
   CP_BLENDFACTOR_DST_COLOR = 5,
   CP_BLENDFACTOR_SRC_ALPHA_SATURATE = 6,
   CP_BLENDFACTOR_CONST_COLOR = 7,
   CP_BLENDFACTOR_CONST_ALPHA = 8,
   CP_BLENDFACTOR_ZERO = 0x11,
   CP_BLENDFACTOR_INV_SRC_COLOR = 0x12,
   CP_BLENDFACTOR_INV_SRC_ALPHA = 0x13,
   CP_BLENDFACTOR_INV_DST_ALPHA = 0x14,
   CP_BLENDFACTOR_INV_DST_COLOR = 0x15,
   CP_BLENDFACTOR_INV_CONST_COLOR = 0x17,
   CP_BLENDFACTOR_INV_CONST_ALPHA = 0x18,
};

/* pipe_blend_func */
enum {
   CP_BLEND_ADD = 0,
   CP_BLEND_SUBTRACT,
   CP_BLEND_REVERSE_SUBTRACT,
   CP_BLEND_MIN,
   CP_BLEND_MAX,
};

static __device__ __forceinline__ float
cp_blend_factor(uint32_t factor, float src, float src_a, float dst, float dst_a)
{
   switch (factor) {
   case CP_BLENDFACTOR_ONE:           return 1.0f;
   case CP_BLENDFACTOR_SRC_COLOR:     return src;
   case CP_BLENDFACTOR_SRC_ALPHA:     return src_a;
   case CP_BLENDFACTOR_DST_ALPHA:     return dst_a;
   case CP_BLENDFACTOR_DST_COLOR:     return dst;
   case CP_BLENDFACTOR_SRC_ALPHA_SATURATE: return fminf(src_a, 1.0f - dst_a);
   case CP_BLENDFACTOR_ZERO:          return 0.0f;
   case CP_BLENDFACTOR_INV_SRC_COLOR: return 1.0f - src;
   case CP_BLENDFACTOR_INV_SRC_ALPHA: return 1.0f - src_a;
   case CP_BLENDFACTOR_INV_DST_ALPHA: return 1.0f - dst_a;
   case CP_BLENDFACTOR_INV_DST_COLOR: return 1.0f - dst;
   default:                           return 1.0f;
   }
}

static __device__ __forceinline__ float
cp_blend_combine(uint32_t func, float src, float dst)
{
   switch (func) {
   case CP_BLEND_SUBTRACT:         return src - dst;
   case CP_BLEND_REVERSE_SUBTRACT: return dst - src;
   case CP_BLEND_MIN:              return fminf(src, dst);
   case CP_BLEND_MAX:              return fmaxf(src, dst);
   default:                        return src + dst;
   }
}

static __device__ __forceinline__ float
cp_linear_to_srgb(float c)
{
   if (c <= 0.0031308f)
      return c * 12.92f;
   return 1.055f * __powf(c, 1.0f / 2.4f) - 0.055f;
}

static __device__ __forceinline__ float
cp_srgb_to_linear_wb(float c)
{
   if (c <= 0.04045f)
      return c * (1.0f / 12.92f);
   return __powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

static __device__ __forceinline__ float
cp_unorm8_to_float(uint32_t v)
{
   return (float)v * (1.0f / 255.0f);
}

static __device__ __forceinline__ uint32_t
cp_float_to_unorm8(float v)
{
   v = fminf(fmaxf(v, 0.0f), 1.0f);
   return (uint32_t)(v * 255.0f + 0.5f);
}

static __device__ __forceinline__ float
cp_half_to_float_wb(unsigned short h)
{
   unsigned sign = (unsigned)(h >> 15) << 31;
   unsigned exp = (h >> 10) & 0x1F;
   unsigned mant = h & 0x3FF;
   unsigned bits;
   if (exp == 0)
      bits = sign;
   else if (exp == 0x1F)
      bits = sign | 0x7F800000u | (mant << 13);
   else
      bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
   return __int_as_float((int)bits);
}

static __device__ void
cp_load_dst(const void *ptr, uint32_t encoding, float *out)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT: {
      const float4 v = *(const float4 *)ptr;
      out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
      break;
   }
   case CP_COLOR_R16G16B16A16_FLOAT: {
      const unsigned short *h = (const unsigned short *)ptr;
      out[0] = cp_half_to_float_wb(h[0]); out[1] = cp_half_to_float_wb(h[1]);
      out[2] = cp_half_to_float_wb(h[2]); out[3] = cp_half_to_float_wb(h[3]);
      break;
   }
   case CP_COLOR_R11G11B10_FLOAT: {
      unsigned p = *(const uint32_t *)ptr;
      out[0] = cp_half_to_float_wb((unsigned short)((p & 0x7FF) << 4));
      out[1] = cp_half_to_float_wb((unsigned short)(((p >> 11) & 0x7FF) << 4));
      out[2] = cp_half_to_float_wb((unsigned short)(((p >> 22) & 0x3FF) << 5));
      out[3] = 1.0f;
      break;
   }
   case CP_COLOR_A2B10G10R10_UNORM: {
      uint32_t p = *(const uint32_t *)ptr;
      out[0] = (float)(p & 0x3FF) * (1.0f / 1023.0f);
      out[1] = (float)((p >> 10) & 0x3FF) * (1.0f / 1023.0f);
      out[2] = (float)((p >> 20) & 0x3FF) * (1.0f / 1023.0f);
      out[3] = (float)((p >> 30) & 0x3) * (1.0f / 3.0f);
      break;
   }
   case CP_COLOR_R16_SFLOAT: {
      out[0] = cp_half_to_float_wb(*(const unsigned short *)ptr);
      out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   case CP_COLOR_R16G16_SFLOAT: {
      const unsigned short *h = (const unsigned short *)ptr;
      out[0] = cp_half_to_float_wb(h[0]); out[1] = cp_half_to_float_wb(h[1]);
      out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   case CP_COLOR_R8_UNORM: {
      out[0] = cp_unorm8_to_float(*(const uint8_t *)ptr);
      out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
      break;
   }
   default: {
      uint32_t p = *(const uint32_t *)ptr;
      float c0 = cp_unorm8_to_float(p & 0xFF);
      float c1 = cp_unorm8_to_float((p >> 8) & 0xFF);
      float c2 = cp_unorm8_to_float((p >> 16) & 0xFF);
      float c3 = cp_unorm8_to_float((p >> 24) & 0xFF);
      if (encoding == CP_COLOR_B8G8R8A8_UNORM || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         out[0] = c2; out[1] = c1; out[2] = c0; out[3] = c3;
      } else {
         out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
      }
      if (encoding == CP_COLOR_R8G8B8A8_SRGB || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         out[0] = cp_srgb_to_linear_wb(out[0]);
         out[1] = cp_srgb_to_linear_wb(out[1]);
         out[2] = cp_srgb_to_linear_wb(out[2]);
      }
      break;
   }
   }
}

static __device__ __forceinline__ unsigned short
cp_float_to_half(float f)
{
   unsigned bits = __float_as_int(f);
   unsigned sign = (bits >> 16) & 0x8000;
   int exp = ((bits >> 23) & 0xFF) - 127 + 15;
   unsigned mant = bits & 0x7FFFFF;

   if (exp <= 0) {
      return (unsigned short)sign;
   } else if (exp >= 0x1F) {
      return (unsigned short)(sign | 0x7C00);
   }
   return (unsigned short)(sign | (exp << 10) | (mant >> 13));
}

static __device__ void
cp_store_dst(void *ptr, uint32_t encoding, const float *c)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT:
      *(float4 *)ptr = make_float4(c[0], c[1], c[2], c[3]);
      break;
   case CP_COLOR_R16G16B16A16_FLOAT: {
      unsigned short *h = (unsigned short *)ptr;
      h[0] = cp_float_to_half(c[0]); h[1] = cp_float_to_half(c[1]);
      h[2] = cp_float_to_half(c[2]); h[3] = cp_float_to_half(c[3]);
      break;
   }
   case CP_COLOR_R11G11B10_FLOAT: {
      /* 11-bit float: 5-bit exp, 6-bit mantissa; 10-bit: 5-bit exp, 5-bit mantissa */
      unsigned short hr = cp_float_to_half(c[0]);
      unsigned short hg = cp_float_to_half(c[1]);
      unsigned short hb = cp_float_to_half(c[2]);
      unsigned r11 = (hr >> 4) & 0x7FF;
      unsigned g11 = (hg >> 4) & 0x7FF;
      unsigned b10 = (hb >> 5) & 0x3FF;
      *(uint32_t *)ptr = r11 | (g11 << 11) | (b10 << 22);
      break;
   }
   case CP_COLOR_A2B10G10R10_UNORM: {
      float r = fminf(fmaxf(c[0], 0.0f), 1.0f);
      float g = fminf(fmaxf(c[1], 0.0f), 1.0f);
      float b = fminf(fmaxf(c[2], 0.0f), 1.0f);
      float a = fminf(fmaxf(c[3], 0.0f), 1.0f);
      uint32_t p = ((uint32_t)(r * 1023.0f + 0.5f)) |
                   ((uint32_t)(g * 1023.0f + 0.5f) << 10) |
                   ((uint32_t)(b * 1023.0f + 0.5f) << 20) |
                   ((uint32_t)(a * 3.0f + 0.5f) << 30);
      *(uint32_t *)ptr = p;
      break;
   }
   case CP_COLOR_R16_SFLOAT: {
      *(unsigned short *)ptr = cp_float_to_half(c[0]);
      break;
   }
   case CP_COLOR_R16G16_SFLOAT: {
      unsigned short *h = (unsigned short *)ptr;
      h[0] = cp_float_to_half(c[0]); h[1] = cp_float_to_half(c[1]);
      break;
   }
   case CP_COLOR_R8_UNORM: {
      *(uint8_t *)ptr = (uint8_t)cp_float_to_unorm8(c[0]);
      break;
   }
   default: {
      float r = c[0], g = c[1], b = c[2];
      if (encoding == CP_COLOR_R8G8B8A8_SRGB || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         r = cp_linear_to_srgb(r);
         g = cp_linear_to_srgb(g);
         b = cp_linear_to_srgb(b);
      }
      uint32_t p;
      if (encoding == CP_COLOR_B8G8R8A8_UNORM || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         p = cp_float_to_unorm8(b) | (cp_float_to_unorm8(g) << 8) |
             (cp_float_to_unorm8(r) << 16) | (cp_float_to_unorm8(c[3]) << 24);
      } else {
         p = cp_float_to_unorm8(r) | (cp_float_to_unorm8(g) << 8) |
             (cp_float_to_unorm8(b) << 16) | (cp_float_to_unorm8(c[3]) << 24);
      }
      *(uint32_t *)ptr = p;
      break;
   }
   }
}

extern "C" __global__ void
cp_fs_writeback(struct cp_fs_writeback_args args)
{
   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t limit = args.pixel_counter
      ? *(const uint32_t *)(uintptr_t)args.pixel_counter
      : args.num_pixels;
   if (i >= limit)
      return;

   /* A discarded fragment contributes neither colour nor depth. */
   if (args.discard_mask && ((const unsigned char *)(uintptr_t)args.discard_mask)[i])
      return;

   uint32_t pixel = ((const uint32_t *)(uintptr_t)args.pixel_list)[i];

   /* This fragment survived the depth test during rasterization, so commit its
    * depth before the next draw tests against it. */
   if (args.depth_write && args.depthbuf && args.visbuf) {
      uint64_t entry = ((const uint64_t *)(uintptr_t)args.visbuf)[pixel];
      uint32_t key = (uint32_t)(entry >> 32);
      ((uint32_t *)(uintptr_t)args.depthbuf)[pixel] =
         args.depth_key_invert ? ~key : key;
   }

   const float4 *fs_out =
      (const float4 *)((const char *)(uintptr_t)args.fs_out +
                       (size_t)i * args.fs_out_stride);
   float src[4] = { fs_out->x, fs_out->y, fs_out->z, fs_out->w };

   uint32_t bpp;
   switch (args.color_encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT: bpp = 16; break;
   case CP_COLOR_R16G16B16A16_FLOAT: bpp = 8; break;
   case CP_COLOR_R16G16_SFLOAT: bpp = 4; break;
   case CP_COLOR_R16_SFLOAT: bpp = 2; break;
   case CP_COLOR_R8_UNORM: bpp = 1; break;
   default: bpp = 4; break;
   }
   void *dst_ptr = (char *)(uintptr_t)args.color_out + (size_t)pixel * bpp;

   float out[4];
   if (args.blend_enable) {
      float dst[4];
      cp_load_dst(dst_ptr, args.color_encoding, dst);

      for (int c = 0; c < 3; c++) {
         float sf = cp_blend_factor(args.rgb_src_factor, src[c], src[3], dst[c], dst[3]);
         float df = cp_blend_factor(args.rgb_dst_factor, src[c], src[3], dst[c], dst[3]);
         out[c] = cp_blend_combine(args.rgb_func, src[c] * sf, dst[c] * df);
      }
      float sfa = cp_blend_factor(args.alpha_src_factor, src[3], src[3], dst[3], dst[3]);
      float dfa = cp_blend_factor(args.alpha_dst_factor, src[3], src[3], dst[3], dst[3]);
      out[3] = cp_blend_combine(args.alpha_func, src[3] * sfa, dst[3] * dfa);

      /* Channels masked out keep the destination value. */
      for (int c = 0; c < 4; c++) {
         if (!(args.colormask & (1u << c)))
            out[c] = dst[c];
      }
   } else if (args.colormask != 0xF) {
      float dst[4];
      cp_load_dst(dst_ptr, args.color_encoding, dst);
      for (int c = 0; c < 4; c++)
         out[c] = (args.colormask & (1u << c)) ? src[c] : dst[c];
   } else {
      for (int c = 0; c < 4; c++)
         out[c] = src[c];
   }

   cp_store_dst(dst_ptr, args.color_encoding, out);
}
