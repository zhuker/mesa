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

/*
 * Non-zero while a fragment shader is running, where threads are laid out four
 * to a 2x2 quad and screen-space derivatives can be taken by shuffling between
 * them. Compute shaders have no such neighbourhood and sample the base level.
 */
__device__ int cp_quad_derivs;

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

/* Upper bound on separate samples taken along an anisotropic footprint. */
#define CP_MAX_ANISO_TAPS 16

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

   /* A view can start partway into an array, so layers are relative to it. */
   z += (int)tex->first_layer;

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
      case CP_TEXEL_R16_SFLOAT: {
         const unsigned short *s = (const unsigned short *)t;
         out.r = cp_half_to_float(s[0]);
         out.g = 0.0f; out.b = 0.0f; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R16G16_SFLOAT: {
         const unsigned short *s = (const unsigned short *)t;
         out.r = cp_half_to_float(s[0]); out.g = cp_half_to_float(s[1]);
         out.b = 0.0f; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R16G16_UNORM: {
         const unsigned short *s = (const unsigned short *)t;
         out.r = (float)s[0] * (1.0f / 65535.0f);
         out.g = (float)s[1] * (1.0f / 65535.0f);
         out.b = 0.0f; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_A2B10G10R10_UNORM: {
         unsigned p = *(const unsigned *)t;
         out.r = (float)(p & 0x3FF) * (1.0f / 1023.0f);
         out.g = (float)((p >> 10) & 0x3FF) * (1.0f / 1023.0f);
         out.b = (float)((p >> 20) & 0x3FF) * (1.0f / 1023.0f);
         out.a = (float)((p >> 30) & 0x3) * (1.0f / 3.0f);
         break;
      }
      case CP_TEXEL_R32_SINT: {
         int v = *(const int *)t;
         out.r = __int_as_float(v); out.g = 0.0f; out.b = 0.0f; out.a = 1.0f;
         break;
      }
      case CP_TEXEL_R16_SINT: {
         short v = *(const short *)t;
         out.r = __int_as_float((int)v); out.g = 0.0f; out.b = 0.0f; out.a = 1.0f;
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

/*
 * Map a cube direction to a face and the 2D coordinates within it.
 *
 * The major axis picks the face; the other two components, divided by its
 * magnitude, give coordinates in [-1, 1] which then map to [0, 1]. Face order
 * and the sign conventions follow the usual +X -X +Y -Y +Z -Z layout.
 */
static __device__ unsigned
cp_cube_face(float x, float y, float z, float *out_u, float *out_v)
{
   float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
   float ma, uc, vc;
   unsigned face;

   if (ax >= ay && ax >= az) {
      ma = ax;
      face = x > 0.0f ? 0 : 1;
      uc = x > 0.0f ? -z : z;
      vc = -y;
   } else if (ay >= az) {
      ma = ay;
      face = y > 0.0f ? 2 : 3;
      uc = x;
      vc = y > 0.0f ? z : -z;
   } else {
      ma = az;
      face = z > 0.0f ? 4 : 5;
      uc = z > 0.0f ? x : -x;
      vc = -y;
   }

   if (ma == 0.0f)
      ma = 1.0f;
   *out_u = 0.5f * (uc / ma + 1.0f);
   *out_v = 0.5f * (vc / ma + 1.0f);
   return face;
}

/*
 * Screen-space derivative of a cube face's u and v, from the derivative of the
 * direction vector. The face selection and the sign conventions have to match
 * cp_cube_face() above exactly.
 *
 * Differencing u and v across the quad the way every other target does is
 * wrong wherever a quad straddles a face boundary: its two lanes are then
 * parameterised on different faces, so the difference is a jump between two
 * unrelated coordinate systems rather than a rate of change. The level of
 * detail derived from it collapses to the coarsest mip, and every face seam
 * draws itself as a line across the reflection — the wireframe over the sphere
 * in texturecubemap and pbribl.
 *
 * The direction vector is continuous across the seam, so differentiate that
 * and push it through the projection
 *
 *     u = 1/2 (uc / ma + 1)
 *
 * by the quotient rule
 *
 *     du = 1/2 (duc - uc dma / ma) / ma
 *
 * which is what lp_build_cube_lookup() does in llvmpipe.
 */
static __device__ void
cp_cube_derivs(float x, float y, float z, float dx, float dy, float dz,
               float *out_du, float *out_dv)
{
   float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
   float ma, uc, vc, dma, duc, dvc;

   if (ax >= ay && ax >= az) {
      ma = ax;                  dma = x > 0.0f ? dx : -dx;
      uc = x > 0.0f ? -z : z;   duc = x > 0.0f ? -dz : dz;
      vc = -y;                  dvc = -dy;
   } else if (ay >= az) {
      ma = ay;                  dma = y > 0.0f ? dy : -dy;
      uc = x;                   duc = dx;
      vc = y > 0.0f ? z : -z;   dvc = y > 0.0f ? dz : -dz;
   } else {
      ma = az;                  dma = z > 0.0f ? dz : -dz;
      uc = z > 0.0f ? x : -x;   duc = z > 0.0f ? dx : -dx;
      vc = -y;                  dvc = -dy;
   }

   if (ma == 0.0f)
      ma = 1.0f;

   float ima = 1.0f / ma;
   *out_du = 0.5f * (duc - uc * dma * ima) * ima;
   *out_dv = 0.5f * (dvc - vc * dma * ima) * ima;
}

/* Sample one mip level with the given in-level filter. `layer` selects the
 * array slice or cube face, or the 3D slice. */
static __device__ struct cp_rgba
cp_sample_level_layer(const struct cp_texture_info *tex,
                      const struct cp_sampler_info *samp, unsigned level,
                      float u, float v, int layer, unsigned filter,
                      bool layer_is_normalized, float layer_coord)
{
   struct cp_rgba c;

   int w = (int)(tex->width >> level);
   int h = (int)(tex->height >> level);
   w = w < 1 ? 1 : w;
   h = h < 1 ? 1 : h;

   /* A 3D texture's slices shrink with the mip level, so the slice has to be
    * derived from the normalized coordinate at the level being sampled —
    * unlike array layers, which are the same at every level. */
   int depth = (int)tex->depth;
   if (layer_is_normalized) {
      depth = depth >> level;
      if (depth < 1)
         depth = 1;
      layer = (int)floorf(layer_coord * (float)depth);
      if (!cp_wrap_texel(&layer, depth, samp->wrap_r)) {
         c.r = samp->border_color[0]; c.g = samp->border_color[1];
         c.b = samp->border_color[2]; c.a = samp->border_color[3];
         return c;
      }
   } else {
      if (depth < 1)
         depth = 1;
      layer = layer < 0 ? 0 : (layer >= depth ? depth - 1 : layer);
   }

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
               t = cp_fetch_texel(tex, level, x, y, layer);
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
         c = cp_fetch_texel(tex, level, x, y, layer);
      } else {
         c.r = samp->border_color[0]; c.g = samp->border_color[1];
         c.b = samp->border_color[2]; c.a = samp->border_color[3];
      }
   }
   return c;
}

/*
 * Sample a texture. `tex_handle` and `samp_handle` are the descriptor
 * addresses the shader loaded; see the file comment.
 *
 * `c0..c2` are the coordinate, interpreted per `flags` (see enum
 * cp_tex_target): the third component is the array layer, the 3D slice, or
 * part of the cube direction. `explicit_lod` is used for texel fetches and
 * explicit-LOD samples.
 *
 * `coord_slot` is the fragment shader input slot the coordinate came from, or
 * -1 if the compiler couldn't trace it to one. With a slot we can look up the
 * coordinate's screen-space derivatives and pick a mip level; without one we
 * sample the base level.
 */
/*
 * textureSize(). A blur kernel derives its tap offsets from one over this, so
 * returning zero does not merely lose detail — it sends every tap to an
 * infinite coordinate and the whole pass comes out black.
 *
 * `component` picks width, height or depth so the caller needs no struct
 * layout knowledge, and the size is that of the requested mip level.
 */
extern "C" __device__ int
cp_tex_size(unsigned long long tex_handle, int lod, int component)
{
   if (!tex_handle)
      return 0;

   const struct cp_texture_info *tex =
      *(const struct cp_texture_info *const *)(tex_handle +
                                               CP_DESC_IMAGE_FUNCTIONS_OFFSET);
   if (!tex)
      return 0;

   unsigned level = (unsigned)lod + tex->first_level;
   unsigned size;
   switch (component) {
   case 0:  size = tex->width;  break;
   case 1:  size = tex->height; break;
   default: size = tex->depth;  break;
   }

   /* Layer counts do not halve with the mip chain; the spatial axes do. */
   if (component < 2 || tex->target == CP_TEX_3D) {
      size >>= level;
      if (!size)
         size = 1;
   }
   return (int)size;
}

extern "C" __device__ float4
cp_tex_sample(unsigned long long tex_handle, unsigned long long samp_handle,
              float c0, float c1, float c2, float explicit_lod,
              int coord_slot, int flags)
{
   float4 result = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

   if (!tex_handle)
      return result;

   const struct cp_texture_info *tex =
      *(const struct cp_texture_info *const *)(tex_handle +
                                               CP_DESC_IMAGE_FUNCTIONS_OFFSET);
   if (!tex || !tex->base || !tex->width || !tex->height)
      return result;

   unsigned target = (unsigned)flags & CP_TEX_TARGET_MASK;

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

   unsigned base_level = tex->first_level;
   unsigned max_level = tex->last_level > base_level ? tex->last_level : base_level;

   /* Resolve the coordinate into 2D-plus-layer form. */
   float u = c0, v = c1;
   int layer = 0;

   switch (target) {
   case CP_TEX_CUBE:
   case CP_TEX_CUBE_ARRAY: {
      /* The cube direction picks a face, which is just another layer. */
      float fu, fv;
      unsigned face = cp_cube_face(c0, c1, c2, &fu, &fv);
      u = fu;
      v = fv;
      layer = (int)face;
      break;
   }
   case CP_TEX_1D_ARRAY:
      layer = (int)(c1 + 0.5f);
      v = 0.0f;
      break;
   case CP_TEX_2D_ARRAY:
      layer = (int)(c2 + 0.5f);
      break;
   case CP_TEX_3D:
      /* Slice resolved per mip level inside cp_sample_level_layer(); filtering
       * between slices is not implemented. */
      break;
   default:
      break;
   }

   /* A texel fetch bypasses the sampler entirely: integer coordinates, an
    * explicit level, and no filtering or wrapping. */
   if (flags & CP_TEX_FETCH) {
      unsigned level = (unsigned)explicit_lod + base_level;
      if (level > max_level)
         level = max_level;

      int w = (int)(tex->width >> level);
      int h = (int)(tex->height >> level);
      w = w < 1 ? 1 : w;
      h = h < 1 ? 1 : h;

      int x = (int)c0;
      int y = (int)c1;
      if (target == CP_TEX_1D_ARRAY)
         layer = (int)c1, y = 0;
      else if (target == CP_TEX_2D_ARRAY || target == CP_TEX_3D)
         layer = (int)c2;

      if (x < 0 || y < 0 || x >= w || y >= h)
         return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

      struct cp_rgba c = cp_fetch_texel(tex, level, x, y, layer);
      return make_float4(c.r, c.g, c.b, c.a);
   }

   /* Level of detail from how fast the coordinate moves across the screen. */
   float lod = 0.0f;
   bool have_lod = false;
   int aniso_taps = 1;
   float aniso_du = 0.0f, aniso_dv = 0.0f;
   if (flags & CP_TEX_LOD) {
      /* textureLod names the level outright — a prefiltered environment map
       * indexed by roughness depends on it, and deriving one from the
       * coordinate instead ignores what the shader asked for. */
      lod = explicit_lod;
      have_lod = true;
   } else if (cp_quad_derivs) {
      /*
       * Derivatives by shuffling across the 2x2 quad, which is how hardware
       * does it. Differencing the *final* texture coordinate — after the cube
       * face projection, after any array or 3D handling — means every target
       * is covered by the same code, and a coordinate the shader computed for
       * itself, such as a reflection vector, gets a level of detail like any
       * other. The lane's parity decides the sign of the difference; only
       * magnitudes and a symmetric tap direction are used, so it does not
       * matter.
       */
      const unsigned full = 0xFFFFFFFFu;
      float du_dx, dv_dx, du_dy, dv_dy;

      if (target == CP_TEX_CUBE || target == CP_TEX_CUBE_ARRAY) {
         /* Except on a cube, where u and v are discontinuous at a face
          * boundary and the direction vector is not. */
         float xdx = __shfl_xor_sync(full, c0, 1) - c0;
         float ydx = __shfl_xor_sync(full, c1, 1) - c1;
         float zdx = __shfl_xor_sync(full, c2, 1) - c2;
         float xdy = __shfl_xor_sync(full, c0, 2) - c0;
         float ydy = __shfl_xor_sync(full, c1, 2) - c1;
         float zdy = __shfl_xor_sync(full, c2, 2) - c2;

         cp_cube_derivs(c0, c1, c2, xdx, ydx, zdx, &du_dx, &dv_dx);
         cp_cube_derivs(c0, c1, c2, xdy, ydy, zdy, &du_dy, &dv_dy);
      } else {
         du_dx = __shfl_xor_sync(full, u, 1) - u;
         dv_dx = __shfl_xor_sync(full, v, 1) - v;
         du_dy = __shfl_xor_sync(full, u, 2) - u;
         dv_dy = __shfl_xor_sync(full, v, 2) - v;
      }

      float w0 = (float)tex->width;
      float h0 = (float)tex->height;
      float dudx = du_dx * w0, dvdx = dv_dx * h0;
      float dudy = du_dy * w0, dvdy = dv_dy * h0;
      float rho = fmaxf(sqrtf(dudx * dudx + dvdx * dvdx),
                        sqrtf(dudy * dudy + dvdy * dvdy));

      /*
       * Anisotropic filtering. A surface seen at a grazing angle has a
       * footprint far longer in one direction than the other, and an isotropic
       * level of detail has to cover the longer one — which is what blurs a
       * ground plane running to the horizon. Instead pick the level that suits
       * the shorter axis and take several samples spread along the longer one.
       *
       * The footprint is the parallelogram spanned by the two derivative
       * vectors; the ellipse through it is
       *
       *     ec_a u^2 + ec_b u v + ec_c v^2 = ec_f
       *
       * whose axes are what the filter needs. Taking the longer of the two
       * derivative vectors instead is the cheap approximation, and it only
       * corrects blur along whichever of them is picked — on a surface whose
       * stretch direction does not line up with either, such as a tunnel
       * receding around the view axis, the perpendicular blur survives.
       */
      /*
       * Anisotropic filtering, following llvmpipe's lp_build_rho_aniso() and
       * lp_apply_ellipse_transform() in gallivm/lp_bld_sample.c.
       *
       * A surface at a grazing angle has a footprint far longer in one
       * direction than the other, and an isotropic level of detail has to
       * cover the longer one — which is what blurs a plane running to the
       * horizon. Take the level that suits the short axis instead, and several
       * samples spread along the long one.
       *
       * The footprint is the parallelogram spanned by the two derivative
       * vectors, and its long axis generally lines up with neither. Rather
       * than find that axis by angle, rewrite the pair into an equivalent one
       * aligned to the ellipse's own axes; then simply taking the longer of
       * the two is correct, and no trigonometry is involved — an angle from
       * atan2 is pure noise on a near-isotropic footprint, which scatters the
       * taps on surfaces facing the viewer.
       */
      if (samp.max_anisotropy > 1.0f) {
         float dx_s = dudx, dx_t = dvdx;
         float dy_s = dudy, dy_t = dvdy;

         float len2_dx = dx_s * dx_s + dx_t * dx_t;
         float len2_dy = dy_s * dy_s + dy_t * dy_t;
         float det = dx_s * dy_t - dy_s * dx_t;
         float dot = dx_s * dy_s + dx_t * dy_t;

         float ax_s2 = dx_s * dx_s, ax_t2 = dx_t * dx_t;
         float ay_s2 = dy_s * dy_s, ay_t2 = dy_t * dy_t;

         /*
          * Three degenerate cases must be excluded before transforming: a
          * zero-length derivative, parallel derivatives (zero determinant),
          * and derivatives already perpendicular, for which the pair is its
          * own ellipse frame and the transform would divide by zero.
          */
         const float eps = 1e-6f, eps2 = 1e-12f;
         if (len2_dx >= eps2 && len2_dy >= eps2 &&
             det * det >= eps2 && fabsf(dot) >= eps) {
            float ec_A = ax_t2 + ay_t2;
            float ec_C = ax_s2 + ay_s2;
            float ec_B = -2.0f * (dx_s * dx_t + dy_s * dy_t);
            float ec_F = det * det;

            float p = ec_A - ec_C;
            float q = ec_A + ec_C;
            float t = sqrtf(p * p + ec_B * ec_B);

            float tp = t * (q + t), tm = t * (q - t);
            if (tp > 0.0f && tm > 0.0f) {
               float Fp = ec_F * (t + p), Fm = ec_F * (t - p);
               ax_s2 = Fp / tp; ax_t2 = Fm / tp;
               ay_s2 = Fm / tm; ay_t2 = Fp / tm;
            }
         }

         float rho_x2 = ax_s2 + ax_t2;
         float rho_y2 = ay_s2 + ay_t2;
         float rho_max2 = fmaxf(rho_x2, rho_y2);
         float rho_min2 = fminf(rho_x2, rho_y2);

         if (rho_min2 > 0.0f) {
            float max_aniso2 = samp.max_anisotropy * samp.max_anisotropy;
            float eta2 = rho_max2 / rho_min2;
            if (!(eta2 >= 1.0f))
               eta2 = 1.0f;
            if (eta2 > max_aniso2)
               eta2 = max_aniso2;

            int rate = (int)ceilf(sqrtf(eta2));
            if (rate > CP_MAX_ANISO_TAPS)
               rate = CP_MAX_ANISO_TAPS;

            if (rate > 1) {
               aniso_taps = rate;

               /* Step along the longer of the transformed axes. Clamping eta2
                * raises this basis, so the loop never skips texels. */
               float major2 = rho_max2;
               float axis = sqrtf(major2);
               if (rho_x2 >= rho_y2) {
                  aniso_du = (dudx / fmaxf(sqrtf(len2_dx), eps)) * axis / w0;
                  aniso_dv = (dvdx / fmaxf(sqrtf(len2_dx), eps)) * axis / h0;
               } else {
                  aniso_du = (dudy / fmaxf(sqrtf(len2_dy), eps)) * axis / w0;
                  aniso_dv = (dvdy / fmaxf(sqrtf(len2_dy), eps)) * axis / h0;
               }

               /* The level of detail divides by the unrounded ratio: ceil()
                * would drop it below the minor axis and render sharper than
                * the hardware. */
               rho = sqrtf(rho_max2 / eta2);
            }
         }
      }

      if (rho > 0.0f) {
         lod = __log2f(rho);
         have_lod = true;
      }
   }
   if (flags & CP_TEX_BIAS)
      lod += explicit_lod;
   lod += samp.lod_bias;

   /* Minification uses the min filter and may cross mip levels; magnification
    * always samples the base level with the mag filter. */
   bool minifying = have_lod && lod > 0.0f;
   unsigned filter = minifying ? samp.min_img_filter : samp.mag_img_filter;

   if (!minifying || samp.min_mip_filter == CP_MIPFILTER_NONE) {
      struct cp_rgba c =
         cp_sample_level_layer(tex, &samp, base_level, u, v, layer, filter,
                               target == CP_TEX_3D, c2);
      return make_float4(c.r, c.g, c.b, c.a);
   }

   lod = fmaxf(lod, samp.min_lod);
   lod = fminf(lod, samp.max_lod);
   lod = fmaxf(lod, 0.0f);

   /* Offsets of the anisotropic taps, spread symmetrically about the centre
    * of the footprint's long axis. One tap degenerates to the centre. */
   float tap_scale = aniso_taps > 1 ? 1.0f / (float)aniso_taps : 0.0f;
   float tap_base = -0.5f * (float)(aniso_taps - 1) * tap_scale;

   if (samp.min_mip_filter == CP_MIPFILTER_NEAREST) {
      /* Round to nearest level, matching the usual ceil(lod - 0.5) rule. */
      unsigned level = base_level + (unsigned)ceilf(lod - 0.5f);
      if (level > max_level)
         level = max_level;
      struct cp_rgba acc = { 0.0f, 0.0f, 0.0f, 0.0f };
      for (int t = 0; t < aniso_taps; t++) {
         float off = tap_base + (float)t * tap_scale;
         struct cp_rgba c =
            cp_sample_level_layer(tex, &samp, level, u + aniso_du * off,
                                  v + aniso_dv * off, layer, filter,
                                  target == CP_TEX_3D, c2);
         acc.r += c.r; acc.g += c.g; acc.b += c.b; acc.a += c.a;
      }
      float inv = 1.0f / (float)aniso_taps;
      return make_float4(acc.r * inv, acc.g * inv, acc.b * inv, acc.a * inv);
   }

   /* Trilinear: blend the two levels bracketing the LOD. */
   unsigned lo = base_level + (unsigned)floorf(lod);
   unsigned hi = lo + 1;
   float frac = lod - floorf(lod);
   if (lo > max_level) lo = max_level;
   if (hi > max_level) hi = max_level;

   struct cp_rgba acc = { 0.0f, 0.0f, 0.0f, 0.0f };
   for (int t = 0; t < aniso_taps; t++) {
      float off = tap_base + (float)t * tap_scale;
      float tu = u + aniso_du * off, tv = v + aniso_dv * off;
      struct cp_rgba a = cp_sample_level_layer(tex, &samp, lo, tu, tv, layer,
                                               filter, target == CP_TEX_3D, c2);
      struct cp_rgba b = cp_sample_level_layer(tex, &samp, hi, tu, tv, layer,
                                               filter, target == CP_TEX_3D, c2);
      acc.r += a.r + (b.r - a.r) * frac;
      acc.g += a.g + (b.g - a.g) * frac;
      acc.b += a.b + (b.b - a.b) * frac;
      acc.a += a.a + (b.a - a.a) * frac;
   }
   float inv = 1.0f / (float)aniso_taps;
   return make_float4(acc.r * inv, acc.g * inv, acc.b * inv, acc.a * inv);
}

/*
 * Precise transcendentals for shaders.
 *
 * The NVVM sin/cos approximations the backend would otherwise emit are fine for
 * shading but not for geometry. An instanced draw that rotates each object's
 * *position* by a per-instance angle multiplies the error by the orbit radius,
 * and in the instancing sample that lever — a rock 7 units out, whose own
 * vertices span 0.06 — displaced every asteroid by a visible pixel or two while
 * the same sin in its local rotation was harmless. NVRTC gives us the CUDA
 * library versions, which range-reduce properly.
 */
extern "C" __device__ float cp_sinf(float x) { return sinf(x); }
extern "C" __device__ float cp_cosf(float x) { return cosf(x); }
