/*
 * cudapipe texture sampler.
 *
 * Compiled to relocatable PTX by NVRTC and linked into each fragment shader's
 * PTX, so shaders can simply call cp_tex_sample_2d() for a nir_tex instruction.
 *
 * The two handles a shader passes in are addresses of the descriptors lavapipe
 * built. From the image descriptor we only read the `functions` field, which
 * this driver's create_texture_handle() pointed at a cp_texture_info; from the
 * sampler descriptor we only read `sampler_index`, which indexes the driver's
 * own sampler table. Everything else about llvmpipe's descriptor layout is
 * none of our business.
 */
#include "cp_rast_types.h"

/* Base address of the driver's cp_sampler_info table. The host writes this
 * into each linked module before launching. */
__device__ unsigned long long cp_sampler_table;

/* Screen-space derivatives of the fragment shader's inputs, produced by
 * cp_fs_interpolate and indexed by the same thread id the shader runs under.
 * Used to pick a mip level when the coordinate comes straight from a varying. */
__device__ unsigned long long cp_fs_deriv;
__device__ unsigned int cp_fs_deriv_stride;

enum {
   CP_WRAP_REPEAT = 0,
   CP_WRAP_CLAMP,
   CP_WRAP_CLAMP_TO_EDGE,
   CP_WRAP_CLAMP_TO_BORDER,
   CP_WRAP_MIRROR_REPEAT,
   CP_WRAP_MIRROR_CLAMP,
   CP_WRAP_MIRROR_CLAMP_TO_EDGE,
   CP_WRAP_MIRROR_CLAMP_TO_BORDER,
};

enum {
   CP_FILTER_NEAREST = 0,
   CP_FILTER_LINEAR,
};

/* pipe_tex_mipfilter */
enum {
   CP_MIPFILTER_NEAREST = 0,
   CP_MIPFILTER_LINEAR,
   CP_MIPFILTER_NONE,
};

struct cp_rgba {
   float r, g, b, a;
};

static __device__ inline float
cp_srgb_to_linear(float c)
{
   return c <= 0.04045f ? c * (1.0f / 12.92f)
                        : __powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

static __device__ inline float
cp_unorm8(unsigned char v)
{
   return (float)v * (1.0f / 255.0f);
}

/* NVRTC compiles without the CUDA headers on its include path, so decode
 * half floats by hand rather than pulling in cuda_fp16.h. */
static __device__ inline float
cp_half_to_float(unsigned short h)
{
   unsigned sign = (unsigned)(h >> 15) << 31;
   unsigned exp = (h >> 10) & 0x1F;
   unsigned mant = h & 0x3FF;
   unsigned bits;

   if (exp == 0) {
      if (mant == 0) {
         bits = sign;
      } else {
         /* Subnormal: renormalise into a float exponent. */
         exp = 127 - 15 + 1;
         while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
         }
         mant &= 0x3FF;
         bits = sign | (exp << 23) | (mant << 13);
      }
   } else if (exp == 0x1F) {
      bits = sign | 0x7F800000u | (mant << 13);
   } else {
      bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
   }
   return __int_as_float((int)bits);
}

/* Decode one texel at (x, y) of the given mip level into RGBA float. */
static __device__ struct cp_rgba
cp_fetch_texel(const struct cp_texture_info *tex, unsigned level,
               int x, int y, int z)
{
   struct cp_rgba out = { 0.0f, 0.0f, 0.0f, 1.0f };

   unsigned row_stride = tex->row_stride[level];
   unsigned img_stride = tex->img_stride[level];
   const unsigned char *base =
      (const unsigned char *)tex->base + tex->mip_offset[level];

   if (tex->encoding >= CP_TEXEL_DXT1_RGB) {
      /* Block-compressed: 4x4 blocks, 8 or 16 bytes each. */
      unsigned block_size = (tex->encoding == CP_TEXEL_DXT1_RGB ||
                             tex->encoding == CP_TEXEL_DXT1_RGBA) ? 8 : 16;
      const unsigned char *blk = base + (unsigned)z * img_stride +
                                 (unsigned)(y >> 2) * row_stride +
                                 (unsigned)(x >> 2) * block_size;
      /* DXT3/DXT5 keep alpha in the first 8 bytes and colour in the last 8. */
      const unsigned char *colour = (block_size == 16) ? blk + 8 : blk;

      unsigned c0 = colour[0] | ((unsigned)colour[1] << 8);
      unsigned c1 = colour[2] | ((unsigned)colour[3] << 8);
      unsigned bits = colour[4] | ((unsigned)colour[5] << 8) |
                      ((unsigned)colour[6] << 16) | ((unsigned)colour[7] << 24);

      float r0 = (float)((c0 >> 11) & 0x1F) * (1.0f / 31.0f);
      float g0 = (float)((c0 >> 5) & 0x3F) * (1.0f / 63.0f);
      float b0 = (float)(c0 & 0x1F) * (1.0f / 31.0f);
      float r1 = (float)((c1 >> 11) & 0x1F) * (1.0f / 31.0f);
      float g1 = (float)((c1 >> 5) & 0x3F) * (1.0f / 63.0f);
      float b1 = (float)(c1 & 0x1F) * (1.0f / 31.0f);

      unsigned sel = (bits >> (2 * ((y & 3) * 4 + (x & 3)))) & 3;
      bool punchthrough = (tex->encoding == CP_TEXEL_DXT1_RGB ||
                           tex->encoding == CP_TEXEL_DXT1_RGBA) && c0 <= c1;

      if (sel == 0) { out.r = r0; out.g = g0; out.b = b0; }
      else if (sel == 1) { out.r = r1; out.g = g1; out.b = b1; }
      else if (sel == 2) {
         if (punchthrough) {
            out.r = (r0 + r1) * 0.5f; out.g = (g0 + g1) * 0.5f; out.b = (b0 + b1) * 0.5f;
         } else {
            out.r = (2.0f * r0 + r1) / 3.0f;
            out.g = (2.0f * g0 + g1) / 3.0f;
            out.b = (2.0f * b0 + b1) / 3.0f;
         }
      } else {
         if (punchthrough) {
            out.r = out.g = out.b = 0.0f;
            if (tex->encoding == CP_TEXEL_DXT1_RGBA)
               out.a = 0.0f;
         } else {
            out.r = (r0 + 2.0f * r1) / 3.0f;
            out.g = (g0 + 2.0f * g1) / 3.0f;
            out.b = (b0 + 2.0f * b1) / 3.0f;
         }
      }

      if (tex->encoding == CP_TEXEL_DXT3_RGBA) {
         unsigned idx = (y & 3) * 4 + (x & 3);
         unsigned nib = blk[idx >> 1];
         unsigned a = (idx & 1) ? (nib >> 4) : (nib & 0xF);
         out.a = (float)a * (1.0f / 15.0f);
      } else if (tex->encoding == CP_TEXEL_DXT5_RGBA) {
         float a0 = cp_unorm8(blk[0]);
         float a1 = cp_unorm8(blk[1]);
         unsigned long long abits = 0;
         for (int i = 0; i < 6; i++)
            abits |= (unsigned long long)blk[2 + i] << (8 * i);
         unsigned asel = (unsigned)((abits >> (3 * ((y & 3) * 4 + (x & 3)))) & 7);
         if (asel == 0) out.a = a0;
         else if (asel == 1) out.a = a1;
         else if (blk[0] > blk[1])
            out.a = ((float)(8 - asel) * a0 + (float)(asel - 1) * a1) / 7.0f;
         else if (asel < 6)
            out.a = ((float)(6 - asel) * a0 + (float)(asel - 1) * a1) / 5.0f;
         else
            out.a = (asel == 6) ? 0.0f : 1.0f;
      }
   } else {
      const unsigned char *t = base + (unsigned)z * img_stride +
                               (unsigned)y * row_stride +
                               (unsigned)x * tex->blocksize;

      switch (tex->encoding) {
      case CP_TEXEL_R8G8B8A8_UNORM:
         out.r = cp_unorm8(t[0]); out.g = cp_unorm8(t[1]);
         out.b = cp_unorm8(t[2]); out.a = cp_unorm8(t[3]);
         break;
      case CP_TEXEL_B8G8R8A8_UNORM:
         out.b = cp_unorm8(t[0]); out.g = cp_unorm8(t[1]);
         out.r = cp_unorm8(t[2]); out.a = cp_unorm8(t[3]);
         break;
      case CP_TEXEL_R8G8B8X8_UNORM:
         out.r = cp_unorm8(t[0]); out.g = cp_unorm8(t[1]);
         out.b = cp_unorm8(t[2]); out.a = 1.0f;
         break;
      case CP_TEXEL_B8G8R8X8_UNORM:
         out.b = cp_unorm8(t[0]); out.g = cp_unorm8(t[1]);
         out.r = cp_unorm8(t[2]); out.a = 1.0f;
         break;
      case CP_TEXEL_A8R8G8B8_UNORM:
         out.a = cp_unorm8(t[0]); out.r = cp_unorm8(t[1]);
         out.g = cp_unorm8(t[2]); out.b = cp_unorm8(t[3]);
         break;
      case CP_TEXEL_X8R8G8B8_UNORM:
         out.a = 1.0f; out.r = cp_unorm8(t[1]);
         out.g = cp_unorm8(t[2]); out.b = cp_unorm8(t[3]);
         break;
      case CP_TEXEL_R8G8B8_UNORM:
         out.r = cp_unorm8(t[0]); out.g = cp_unorm8(t[1]);
         out.b = cp_unorm8(t[2]); out.a = 1.0f;
         break;
      case CP_TEXEL_R8G8_UNORM:
         out.r = cp_unorm8(t[0]); out.g = cp_unorm8(t[1]);
         out.b = 0.0f; out.a = 1.0f;
         break;
      case CP_TEXEL_R8_UNORM:
         out.r = cp_unorm8(t[0]); out.g = 0.0f; out.b = 0.0f; out.a = 1.0f;
         break;
      case CP_TEXEL_R8G8B8A8_SNORM: {
         const signed char *s = (const signed char *)t;
         out.r = fmaxf((float)s[0] * (1.0f / 127.0f), -1.0f);
         out.g = fmaxf((float)s[1] * (1.0f / 127.0f), -1.0f);
         out.b = fmaxf((float)s[2] * (1.0f / 127.0f), -1.0f);
         out.a = fmaxf((float)s[3] * (1.0f / 127.0f), -1.0f);
         break;
      }
      case CP_TEXEL_R16G16B16A16_UNORM: {
         const unsigned short *s = (const unsigned short *)t;
         out.r = (float)s[0] * (1.0f / 65535.0f);
         out.g = (float)s[1] * (1.0f / 65535.0f);
         out.b = (float)s[2] * (1.0f / 65535.0f);
         out.a = (float)s[3] * (1.0f / 65535.0f);
         break;
      }
      case CP_TEXEL_R16G16B16A16_FLOAT: {
         const unsigned short *s = (const unsigned short *)t;
         out.r = cp_half_to_float(s[0]); out.g = cp_half_to_float(s[1]);
         out.b = cp_half_to_float(s[2]); out.a = cp_half_to_float(s[3]);
         break;
      }
      case CP_TEXEL_R32G32B32A32_FLOAT: {
         const float *s = (const float *)t;
         out.r = s[0]; out.g = s[1]; out.b = s[2]; out.a = s[3];
         break;
      }
      case CP_TEXEL_R32G32B32_FLOAT: {
         const float *s = (const float *)t;
         out.r = s[0]; out.g = s[1]; out.b = s[2]; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R32G32_FLOAT: {
         const float *s = (const float *)t;
         out.r = s[0]; out.g = s[1]; out.b = 0.0f; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R32_FLOAT: {
         const float *s = (const float *)t;
         out.r = s[0]; out.g = 0.0f; out.b = 0.0f; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R5G6B5_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.b = (float)(p & 0x1F) * (1.0f / 31.0f);
         out.g = (float)((p >> 5) & 0x3F) * (1.0f / 63.0f);
         out.r = (float)((p >> 11) & 0x1F) * (1.0f / 31.0f);
         out.a = 1.0f;
         break;
      }
      case CP_TEXEL_B5G5R5A1_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.b = (float)(p & 0x1F) * (1.0f / 31.0f);
         out.g = (float)((p >> 5) & 0x1F) * (1.0f / 31.0f);
         out.r = (float)((p >> 10) & 0x1F) * (1.0f / 31.0f);
         out.a = (float)((p >> 15) & 0x1);
         break;
      }
      case CP_TEXEL_A1R5G5B5_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.a = (float)(p & 0x1);
         out.r = (float)((p >> 1) & 0x1F) * (1.0f / 31.0f);
         out.g = (float)((p >> 6) & 0x1F) * (1.0f / 31.0f);
         out.b = (float)((p >> 11) & 0x1F) * (1.0f / 31.0f);
         break;
      }
      case CP_TEXEL_A1B5G5R5_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.a = (float)(p & 0x1);
         out.b = (float)((p >> 1) & 0x1F) * (1.0f / 31.0f);
         out.g = (float)((p >> 6) & 0x1F) * (1.0f / 31.0f);
         out.r = (float)((p >> 11) & 0x1F) * (1.0f / 31.0f);
         break;
      }
      case CP_TEXEL_B4G4R4A4_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.b = (float)(p & 0xF) * (1.0f / 15.0f);
         out.g = (float)((p >> 4) & 0xF) * (1.0f / 15.0f);
         out.r = (float)((p >> 8) & 0xF) * (1.0f / 15.0f);
         out.a = (float)((p >> 12) & 0xF) * (1.0f / 15.0f);
         break;
      }
      case CP_TEXEL_A4R4G4B4_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.a = (float)(p & 0xF) * (1.0f / 15.0f);
         out.r = (float)((p >> 4) & 0xF) * (1.0f / 15.0f);
         out.g = (float)((p >> 8) & 0xF) * (1.0f / 15.0f);
         out.b = (float)((p >> 12) & 0xF) * (1.0f / 15.0f);
         break;
      }
      case CP_TEXEL_A4B4G4R4_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.a = (float)(p & 0xF) * (1.0f / 15.0f);
         out.b = (float)((p >> 4) & 0xF) * (1.0f / 15.0f);
         out.g = (float)((p >> 8) & 0xF) * (1.0f / 15.0f);
         out.r = (float)((p >> 12) & 0xF) * (1.0f / 15.0f);
         break;
      }
      case CP_TEXEL_R4G4B4A4_UNORM: {
         unsigned p = t[0] | ((unsigned)t[1] << 8);
         out.r = (float)(p & 0xF) * (1.0f / 15.0f);
         out.g = (float)((p >> 4) & 0xF) * (1.0f / 15.0f);
         out.b = (float)((p >> 8) & 0xF) * (1.0f / 15.0f);
         out.a = (float)((p >> 12) & 0xF) * (1.0f / 15.0f);
         break;
      }
      case CP_TEXEL_R11G11B10_FLOAT: {
         unsigned p = *(const unsigned *)t;
         /* 11/11/10-bit floats: no sign, 5-bit exponent, so widen the mantissa
          * into a half and reuse the half decode. */
         out.r = cp_half_to_float((unsigned short)((p & 0x7FF) << 4));
         out.g = cp_half_to_float((unsigned short)(((p >> 11) & 0x7FF) << 4));
         out.b = cp_half_to_float((unsigned short)(((p >> 22) & 0x3FF) << 5));
         out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R9G9B9E5_FLOAT: {
         unsigned p = *(const unsigned *)t;
         int exp = (int)((p >> 27) & 0x1F);
         float scale = exp2f((float)(exp - 15 - 9));
         out.r = (float)(p & 0x1FF) * scale;
         out.g = (float)((p >> 9) & 0x1FF) * scale;
         out.b = (float)((p >> 18) & 0x1FF) * scale;
         out.a = 1.0f;
         break;
      }
      default:
         /* Format we don't decode yet: opaque black is at least deterministic. */
         out.r = out.g = out.b = 0.0f; out.a = 1.0f;
         break;
      }
   }

   if (tex->is_srgb) {
      out.r = cp_srgb_to_linear(out.r);
      out.g = cp_srgb_to_linear(out.g);
      out.b = cp_srgb_to_linear(out.b);
   }
   return out;
}

/* Map a texel coordinate into [0, size) according to the wrap mode.
 * Returns false when the sample falls outside and should use the border. */
static __device__ inline bool
cp_wrap_texel(int *coord, int size, unsigned mode)
{
   int c = *coord;
   switch (mode) {
   case CP_WRAP_REPEAT:
      c %= size;
      if (c < 0)
         c += size;
      break;
   case CP_WRAP_MIRROR_REPEAT: {
      int period = 2 * size;
      c %= period;
      if (c < 0)
         c += period;
      if (c >= size)
         c = period - 1 - c;
      break;
   }
   case CP_WRAP_CLAMP_TO_BORDER:
   case CP_WRAP_MIRROR_CLAMP_TO_BORDER:
      if (c < 0 || c >= size)
         return false;
      break;
   default: /* CLAMP / CLAMP_TO_EDGE and the mirrored clamp variants */
      c = c < 0 ? 0 : (c >= size ? size - 1 : c);
      break;
   }
   *coord = c;
   return true;
}

/* Sample one mip level with the given in-level filter. */
static __device__ struct cp_rgba
cp_sample_level(const struct cp_texture_info *tex,
                const struct cp_sampler_info *samp, unsigned level,
                float u, float v, unsigned filter)
{
   struct cp_rgba c;

   int w = (int)(tex->width >> level);
   int h = (int)(tex->height >> level);
   w = w < 1 ? 1 : w;
   h = h < 1 ? 1 : h;

   /* Texel-space coordinates; normalized coords scale by the level size. */
   float su = samp->unnormalized_coords ? u : u * (float)w;
   float sv = samp->unnormalized_coords ? v : v * (float)h;

   if (filter == CP_FILTER_LINEAR) {
      /* Bilinear: sample the four texels around the sample point, which sits
       * half a texel in from the texel centre. */
      float fu = su - 0.5f;
      float fv = sv - 0.5f;
      int x0 = (int)floorf(fu);
      int y0 = (int)floorf(fv);
      float au = fu - (float)x0;
      float av = fv - (float)y0;

      float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
      for (int j = 0; j < 2; j++) {
         for (int i = 0; i < 2; i++) {
            int x = x0 + i;
            int y = y0 + j;
            float weight = (i ? au : 1.0f - au) * (j ? av : 1.0f - av);
            struct cp_rgba t;
            if (cp_wrap_texel(&x, w, samp->wrap_s) &&
                cp_wrap_texel(&y, h, samp->wrap_t)) {
               t = cp_fetch_texel(tex, level, x, y, 0);
            } else {
               t.r = samp->border_color[0]; t.g = samp->border_color[1];
               t.b = samp->border_color[2]; t.a = samp->border_color[3];
            }
            acc_r += t.r * weight; acc_g += t.g * weight;
            acc_b += t.b * weight; acc_a += t.a * weight;
         }
      }
      c.r = acc_r; c.g = acc_g; c.b = acc_b; c.a = acc_a;
   } else {
      int x = (int)floorf(su);
      int y = (int)floorf(sv);
      if (cp_wrap_texel(&x, w, samp->wrap_s) &&
          cp_wrap_texel(&y, h, samp->wrap_t)) {
         c = cp_fetch_texel(tex, level, x, y, 0);
      } else {
         c.r = samp->border_color[0]; c.g = samp->border_color[1];
         c.b = samp->border_color[2]; c.a = samp->border_color[3];
      }
   }
   return c;
}

/*
 * Sample a 2D texture. `tex_handle` and `samp_handle` are the descriptor
 * addresses the shader loaded; see the file comment.
 *
 * `coord_slot` is the fragment shader input slot the coordinate came from, or
 * -1 if the compiler couldn't trace it to one. With a slot we can look up the
 * coordinate's screen-space derivatives and pick a mip level; without one we
 * sample the base level.
 */
extern "C" __device__ float4
cp_tex_sample_2d(unsigned long long tex_handle, unsigned long long samp_handle,
                 float u, float v, int coord_slot)
{
   float4 result = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

   if (!tex_handle)
      return result;

   const struct cp_texture_info *tex =
      *(const struct cp_texture_info *const *)(tex_handle +
                                               CP_DESC_IMAGE_FUNCTIONS_OFFSET);
   if (!tex || !tex->base || !tex->width || !tex->height)
      return result;

   struct cp_sampler_info samp;
   if (samp_handle && cp_sampler_table) {
      unsigned idx = *(const unsigned *)(samp_handle + CP_DESC_SAMPLER_INDEX_OFFSET);
      samp = ((const struct cp_sampler_info *)cp_sampler_table)[idx];
   } else {
      samp.wrap_s = samp.wrap_t = samp.wrap_r = CP_WRAP_REPEAT;
      samp.min_img_filter = samp.mag_img_filter = CP_FILTER_NEAREST;
      samp.min_mip_filter = CP_MIPFILTER_NONE;
      samp.unnormalized_coords = 0;
      samp.min_lod = 0.0f; samp.max_lod = 0.0f; samp.lod_bias = 0.0f;
      samp.border_color[0] = samp.border_color[1] = 0.0f;
      samp.border_color[2] = 0.0f; samp.border_color[3] = 0.0f;
   }

   /* Level of detail from how fast the coordinate moves across the screen. */
   float lod = 0.0f;
   bool have_lod = false;
   if (coord_slot >= 0 && cp_fs_deriv) {
      unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
      const float4 *d = (const float4 *)(cp_fs_deriv +
                                         (size_t)tid * cp_fs_deriv_stride +
                                         (size_t)coord_slot * 16);
      float w0 = (float)tex->width;
      float h0 = (float)tex->height;
      float dudx = d->x * w0, dvdx = d->y * h0;
      float dudy = d->z * w0, dvdy = d->w * h0;
      float rho = fmaxf(sqrtf(dudx * dudx + dvdx * dvdx),
                        sqrtf(dudy * dudy + dvdy * dvdy));
      if (rho > 0.0f) {
         lod = __log2f(rho);
         have_lod = true;
      }
   }
   lod += samp.lod_bias;

   /* Minification uses the min filter and may cross mip levels; magnification
    * always samples the base level with the mag filter. */
   bool minifying = have_lod && lod > 0.0f;
   unsigned filter = minifying ? samp.min_img_filter : samp.mag_img_filter;

   unsigned base_level = tex->first_level;
   unsigned max_level = tex->last_level > base_level ? tex->last_level : base_level;

   if (!minifying || samp.min_mip_filter == CP_MIPFILTER_NONE) {
      struct cp_rgba c = cp_sample_level(tex, &samp, base_level, u, v, filter);
      return make_float4(c.r, c.g, c.b, c.a);
   }

   lod = fmaxf(lod, samp.min_lod);
   lod = fminf(lod, samp.max_lod);
   lod = fmaxf(lod, 0.0f);

   if (samp.min_mip_filter == CP_MIPFILTER_NEAREST) {
      /* Round to nearest level, matching the usual ceil(lod - 0.5) rule. */
      unsigned level = base_level + (unsigned)ceilf(lod - 0.5f);
      if (level > max_level)
         level = max_level;
      struct cp_rgba c = cp_sample_level(tex, &samp, level, u, v, filter);
      return make_float4(c.r, c.g, c.b, c.a);
   }

   /* Trilinear: blend the two levels bracketing the LOD. */
   unsigned lo = base_level + (unsigned)floorf(lod);
   unsigned hi = lo + 1;
   float frac = lod - floorf(lod);
   if (lo > max_level) lo = max_level;
   if (hi > max_level) hi = max_level;

   struct cp_rgba a = cp_sample_level(tex, &samp, lo, u, v, filter);
   struct cp_rgba b = cp_sample_level(tex, &samp, hi, u, v, filter);
   return make_float4(a.r + (b.r - a.r) * frac,
                      a.g + (b.g - a.g) * frac,
                      a.b + (b.b - a.b) * frac,
                      a.a + (b.a - a.a) * frac);
}
