/*
 * The fragment stage's writeback, shared by two compilers.
 *
 * `kernels/cp_fs.cu` compiles it with NVRTC as the body of the standalone
 * cp_fs_writeback kernel; `kernels/cp_fs_inline.c` compiles it with clang to
 * NVPTX bitcode that is linked into the generated fragment shader and inlined
 * away, so a draw whose shader admits the fusion writes its own pixel and the
 * separate launch does not exist. Both must produce identical bytes for the
 * same input, which is what the shims below are for: everything the two
 * compilers spell differently is a macro, and nothing else in this file knows
 * which compiler it is.
 *
 * This is a move out of cp_fs.cu, not a rewrite. Every expression is the one
 * that was there; only the inline qualifier, the four builtins and the
 * writeback's parameter (a C++ reference became a pointer, because the bitcode
 * build is C) changed.
 *
 * Fusing is enabled only for the direct colour path. The A-buffer path,
 * where several fragments per pixel exist, always uses its composite and is
 * never handed this block.
 */
#ifndef CP_FS_WRITE_H
#define CP_FS_WRITE_H

#include "cp_rast_types.h"

/*
 * What the two compilers spell differently. The NVRTC defaults are the CUDA
 * builtins the code always used; cp_fs_inline.c overrides them with the clang
 * builtins and the NVVM intrinsics that lower to the same instructions.
 */
#ifndef CP_FSW_INLINE
#define CP_FSW_INLINE __device__ static inline
#endif
#ifndef CP_FSW_FMINF
#define CP_FSW_FMINF(a, b) fminf((a), (b))
#endif
#ifndef CP_FSW_FMAXF
#define CP_FSW_FMAXF(a, b) fmaxf((a), (b))
#endif
#ifndef CP_FSW_POWF
#define CP_FSW_POWF(a, b) __powf((a), (b))
#endif
#ifndef CP_FSW_FFS
#define CP_FSW_FFS(a) __ffs(a)
#endif
#ifndef CP_FSW_FLOAT_AS_INT
#define CP_FSW_FLOAT_AS_INT(f) __float_as_int(f)
#endif
#ifndef CP_FSW_INT_AS_FLOAT
#define CP_FSW_INT_AS_FLOAT(i) __int_as_float(i)
#endif
#ifndef CP_FSW_UINT_AS_FLOAT
#define CP_FSW_UINT_AS_FLOAT(i) __uint_as_float(i)
#endif
#ifndef CP_FSW_FLOAT_AS_UINT
#define CP_FSW_FLOAT_AS_UINT(f) __float_as_uint(f)
#endif
#ifndef CP_FSW_MAKE_FLOAT4
#define CP_FSW_MAKE_FLOAT4(x, y, z, w) make_float4((x), (y), (z), (w))
#endif
#ifndef CP_FSW_UNROLL
#define CP_FSW_UNROLL _Pragma("unroll")
#endif

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

CP_FSW_INLINE float
cp_blend_factor(uint32_t factor, float src, float src_a, float dst, float dst_a)
{
   switch (factor) {
   case CP_BLENDFACTOR_ONE:           return 1.0f;
   case CP_BLENDFACTOR_SRC_COLOR:     return src;
   case CP_BLENDFACTOR_SRC_ALPHA:     return src_a;
   case CP_BLENDFACTOR_DST_ALPHA:     return dst_a;
   case CP_BLENDFACTOR_DST_COLOR:     return dst;
   case CP_BLENDFACTOR_SRC_ALPHA_SATURATE: return CP_FSW_FMINF(src_a, 1.0f - dst_a);
   case CP_BLENDFACTOR_ZERO:          return 0.0f;
   case CP_BLENDFACTOR_INV_SRC_COLOR: return 1.0f - src;
   case CP_BLENDFACTOR_INV_SRC_ALPHA: return 1.0f - src_a;
   case CP_BLENDFACTOR_INV_DST_ALPHA: return 1.0f - dst_a;
   case CP_BLENDFACTOR_INV_DST_COLOR: return 1.0f - dst;
   default:                           return 1.0f;
   }
}

CP_FSW_INLINE float
cp_blend_combine(uint32_t func, float src, float dst)
{
   switch (func) {
   case CP_BLEND_SUBTRACT:         return src - dst;
   case CP_BLEND_REVERSE_SUBTRACT: return dst - src;
   case CP_BLEND_MIN:              return CP_FSW_FMINF(src, dst);
   case CP_BLEND_MAX:              return CP_FSW_FMAXF(src, dst);
   default:                        return src + dst;
   }
}

CP_FSW_INLINE float
cp_linear_to_srgb(float c)
{
   if (c <= 0.0031308f)
      return c * 12.92f;
   return 1.055f * CP_FSW_POWF(c, 1.0f / 2.4f) - 0.055f;
}

CP_FSW_INLINE float
cp_srgb_to_linear_wb(float c)
{
   if (c <= 0.04045f)
      return c * (1.0f / 12.92f);
   return CP_FSW_POWF((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

CP_FSW_INLINE float
cp_unorm8_to_float(uint32_t v)
{
   return (float)v * (1.0f / 255.0f);
}

CP_FSW_INLINE uint32_t
cp_float_to_unorm8(float v)
{
   v = CP_FSW_FMINF(CP_FSW_FMAXF(v, 0.0f), 1.0f);
   return (uint32_t)(v * 255.0f + 0.5f);
}

CP_FSW_INLINE float
cp_unorm16_to_float(uint32_t v)
{
   return (float)v * (1.0f / 65535.0f);
}

CP_FSW_INLINE uint32_t
cp_float_to_unorm16(float v)
{
   v = CP_FSW_FMINF(CP_FSW_FMAXF(v, 0.0f), 1.0f);
   return (uint32_t)(v * 65535.0f + 0.5f);
}

CP_FSW_INLINE float
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
   return CP_FSW_INT_AS_FLOAT((int)bits);
}

/*
 * The blend equation, once.
 *
 * Both cp_fs_writeback and cp_abuf_composite call this, on the same
 * struct cp_blend_desc the host filled from the draw's pipe_rt_blend_state.
 * Neither of them may hold its own copy of the arithmetic: this driver already
 * carries one pair of routines that drifted apart (CUDAVK_HANDOFF.md gap 12)
 * and the whole point of the A-buffer path is that it composites what the peel
 * path composited.
 *
 * `dst` is the attachment's decoded value; `out` may alias neither.
 */
CP_FSW_INLINE float4
cp_blend_resolve4(const struct cp_blend_desc *b, float4 src, float4 dst)
{
   /*
    * By value, in four scalars, rather than through three float[4] buffers.
    * The arrays were the shape this always had and they are fine in a kernel
    * of their own -- but inlined into a fragment shader they become a
    * function-entry alloca, LLVM merges two adjacent float stores into one
    * i64, and SROA will not split a partition it sees at two widths. The
    * result was 48 bytes of __local_depot in every fused build. Scalars cannot
    * acquire a second width. The array-shaped wrappers below keep the other
    * callers unchanged.
    */
   float4 out;
   if (b->enable) {
      float sfr = cp_blend_factor(b->rgb_src_factor, src.x, src.w, dst.x, dst.w);
      float dfr = cp_blend_factor(b->rgb_dst_factor, src.x, src.w, dst.x, dst.w);
      out.x = cp_blend_combine(b->rgb_func, src.x * sfr, dst.x * dfr);
      float sfg = cp_blend_factor(b->rgb_src_factor, src.y, src.w, dst.y, dst.w);
      float dfg = cp_blend_factor(b->rgb_dst_factor, src.y, src.w, dst.y, dst.w);
      out.y = cp_blend_combine(b->rgb_func, src.y * sfg, dst.y * dfg);
      float sfb = cp_blend_factor(b->rgb_src_factor, src.z, src.w, dst.z, dst.w);
      float dfb = cp_blend_factor(b->rgb_dst_factor, src.z, src.w, dst.z, dst.w);
      out.z = cp_blend_combine(b->rgb_func, src.z * sfb, dst.z * dfb);

      float sfa = cp_blend_factor(b->alpha_src_factor, src.w, src.w, dst.w, dst.w);
      float dfa = cp_blend_factor(b->alpha_dst_factor, src.w, src.w, dst.w, dst.w);
      out.w = cp_blend_combine(b->alpha_func, src.w * sfa, dst.w * dfa);

      /* Channels masked out keep the destination value. */
      if (!(b->colormask & 1u)) out.x = dst.x;
      if (!(b->colormask & 2u)) out.y = dst.y;
      if (!(b->colormask & 4u)) out.z = dst.z;
      if (!(b->colormask & 8u)) out.w = dst.w;
   } else {
      out.x = (b->colormask & 1u) ? src.x : dst.x;
      out.y = (b->colormask & 2u) ? src.y : dst.y;
      out.z = (b->colormask & 4u) ? src.z : dst.z;
      out.w = (b->colormask & 8u) ? src.w : dst.w;
   }
   return out;
}

/* The array-shaped call the A-buffer composite and the peel path already
 * make. One implementation, two spellings of its arguments. */
CP_FSW_INLINE void
cp_blend_resolve(const struct cp_blend_desc *b, const float *src,
                 const float *dst, float *out)
{
   float4 v = cp_blend_resolve4(b, CP_FSW_MAKE_FLOAT4(src[0], src[1], src[2],
                                                      src[3]),
                                CP_FSW_MAKE_FLOAT4(dst[0], dst[1], dst[2],
                                                   dst[3]));
   out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

CP_FSW_INLINE void
cp_load_dst(const void *ptr, uint32_t encoding, float *out);

CP_FSW_INLINE uint32_t
cp_bytes_per_pixel(uint32_t encoding)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT: return 16;
   case CP_COLOR_R16G16B16A16_FLOAT: return 8;
   case CP_COLOR_R16G16_SFLOAT:      return 4;
   case CP_COLOR_R16G16_UNORM:       return 4;
   case CP_COLOR_R16_SFLOAT:         return 2;
   case CP_COLOR_R8G8_UNORM:         return 2;
   case CP_COLOR_R8_UNORM:           return 1;
   case CP_COLOR_R8G8B8A8_UINT:      return 4;
   default:                          return 4;
   }
}

CP_FSW_INLINE float4
cp_load_dst4(const void *ptr, uint32_t encoding)
{
   /* Four scalars, returned by value: see cp_blend_resolve4 for why a
    * float[4] here is what put 48 bytes of local memory in every fused
    * fragment build. Every case writes all four, as they always did. */
   float4 out = CP_FSW_MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT: {
      const float4 v = *(const float4 *)ptr;
      out.x = v.x; out.y = v.y; out.z = v.z; out.w = v.w;
      break;
   }
   case CP_COLOR_R16G16B16A16_FLOAT: {
      const unsigned short *h = (const unsigned short *)ptr;
      out.x = cp_half_to_float_wb(h[0]); out.y = cp_half_to_float_wb(h[1]);
      out.z = cp_half_to_float_wb(h[2]); out.w = cp_half_to_float_wb(h[3]);
      break;
   }
   case CP_COLOR_R11G11B10_FLOAT: {
      unsigned p = *(const uint32_t *)ptr;
      out.x = cp_half_to_float_wb((unsigned short)((p & 0x7FF) << 4));
      out.y = cp_half_to_float_wb((unsigned short)(((p >> 11) & 0x7FF) << 4));
      out.z = cp_half_to_float_wb((unsigned short)(((p >> 22) & 0x3FF) << 5));
      out.w = 1.0f;
      break;
   }
   case CP_COLOR_A2B10G10R10_UNORM: {
      uint32_t p = *(const uint32_t *)ptr;
      out.x = (float)(p & 0x3FF) * (1.0f / 1023.0f);
      out.y = (float)((p >> 10) & 0x3FF) * (1.0f / 1023.0f);
      out.z = (float)((p >> 20) & 0x3FF) * (1.0f / 1023.0f);
      out.w = (float)((p >> 30) & 0x3) * (1.0f / 3.0f);
      break;
   }
   case CP_COLOR_R16_SFLOAT: {
      out.x = cp_half_to_float_wb(*(const unsigned short *)ptr);
      out.y = 0.0f; out.z = 0.0f; out.w = 1.0f;
      break;
   }
   case CP_COLOR_R16G16_SFLOAT: {
      const unsigned short *h = (const unsigned short *)ptr;
      out.x = cp_half_to_float_wb(h[0]); out.y = cp_half_to_float_wb(h[1]);
      out.z = 0.0f; out.w = 1.0f;
      break;
   }
   case CP_COLOR_R8_UNORM: {
      out.x = cp_unorm8_to_float(*(const uint8_t *)ptr);
      out.y = 0.0f; out.z = 0.0f; out.w = 1.0f;
      break;
   }
   case CP_COLOR_R8G8_UNORM: {
      const uint8_t *v = (const uint8_t *)ptr;
      out.x = cp_unorm8_to_float(v[0]);
      out.y = cp_unorm8_to_float(v[1]);
      out.z = 0.0f; out.w = 1.0f;
      break;
   }
   case CP_COLOR_R16G16_UNORM: {
      const unsigned short *v = (const unsigned short *)ptr;
      out.x = cp_unorm16_to_float(v[0]);
      out.y = cp_unorm16_to_float(v[1]);
      out.z = 0.0f; out.w = 1.0f;
      break;
   }
   case CP_COLOR_R8G8B8A8_UINT: {
      uint32_t p = *(const uint32_t *)ptr;
      out.x = CP_FSW_UINT_AS_FLOAT(p & 0xFFu);
      out.y = CP_FSW_UINT_AS_FLOAT((p >> 8) & 0xFFu);
      out.z = CP_FSW_UINT_AS_FLOAT((p >> 16) & 0xFFu);
      out.w = CP_FSW_UINT_AS_FLOAT((p >> 24) & 0xFFu);
      break;
   }
   default: {
      uint32_t p = *(const uint32_t *)ptr;
      float c0 = cp_unorm8_to_float(p & 0xFF);
      float c1 = cp_unorm8_to_float((p >> 8) & 0xFF);
      float c2 = cp_unorm8_to_float((p >> 16) & 0xFF);
      float c3 = cp_unorm8_to_float((p >> 24) & 0xFF);
      if (encoding == CP_COLOR_B8G8R8A8_UNORM || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         out.x = c2; out.y = c1; out.z = c0; out.w = c3;
      } else {
         out.x = c0; out.y = c1; out.z = c2; out.w = c3;
      }
      if (encoding == CP_COLOR_R8G8B8A8_SRGB || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         out.x = cp_srgb_to_linear_wb(out.x);
         out.y = cp_srgb_to_linear_wb(out.y);
         out.z = cp_srgb_to_linear_wb(out.z);
      }
      break;
   }
   }
   return out;
}

/* The array-shaped call the other kernels already make. */
CP_FSW_INLINE void
cp_load_dst(const void *ptr, uint32_t encoding, float *out)
{
   float4 v = cp_load_dst4(ptr, encoding);
   out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

CP_FSW_INLINE unsigned short
cp_float_to_half(float f)
{
   unsigned bits = CP_FSW_FLOAT_AS_INT(f);
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

CP_FSW_INLINE void
cp_store_dst4(void *ptr, uint32_t encoding, float4 c)
{
   switch (encoding) {
   case CP_COLOR_R32G32B32A32_FLOAT:
      *(float4 *)ptr = CP_FSW_MAKE_FLOAT4(c.x, c.y, c.z, c.w);
      break;
   case CP_COLOR_R16G16B16A16_FLOAT: {
      unsigned short *h = (unsigned short *)ptr;
      h[0] = cp_float_to_half(c.x); h[1] = cp_float_to_half(c.y);
      h[2] = cp_float_to_half(c.z); h[3] = cp_float_to_half(c.w);
      break;
   }
   case CP_COLOR_R11G11B10_FLOAT: {
      /* 11-bit float: 5-bit exp, 6-bit mantissa; 10-bit: 5-bit exp, 5-bit mantissa */
      unsigned short hr = cp_float_to_half(c.x);
      unsigned short hg = cp_float_to_half(c.y);
      unsigned short hb = cp_float_to_half(c.z);
      unsigned r11 = (hr >> 4) & 0x7FF;
      unsigned g11 = (hg >> 4) & 0x7FF;
      unsigned b10 = (hb >> 5) & 0x3FF;
      *(uint32_t *)ptr = r11 | (g11 << 11) | (b10 << 22);
      break;
   }
   case CP_COLOR_A2B10G10R10_UNORM: {
      float r = CP_FSW_FMINF(CP_FSW_FMAXF(c.x, 0.0f), 1.0f);
      float g = CP_FSW_FMINF(CP_FSW_FMAXF(c.y, 0.0f), 1.0f);
      float b = CP_FSW_FMINF(CP_FSW_FMAXF(c.z, 0.0f), 1.0f);
      float a = CP_FSW_FMINF(CP_FSW_FMAXF(c.w, 0.0f), 1.0f);
      uint32_t p = ((uint32_t)(r * 1023.0f + 0.5f)) |
                   ((uint32_t)(g * 1023.0f + 0.5f) << 10) |
                   ((uint32_t)(b * 1023.0f + 0.5f) << 20) |
                   ((uint32_t)(a * 3.0f + 0.5f) << 30);
      *(uint32_t *)ptr = p;
      break;
   }
   case CP_COLOR_R16_SFLOAT: {
      *(unsigned short *)ptr = cp_float_to_half(c.x);
      break;
   }
   case CP_COLOR_R16G16_SFLOAT: {
      unsigned short *h = (unsigned short *)ptr;
      h[0] = cp_float_to_half(c.x); h[1] = cp_float_to_half(c.y);
      break;
   }
   case CP_COLOR_R8_UNORM: {
      *(uint8_t *)ptr = (uint8_t)cp_float_to_unorm8(c.x);
      break;
   }
   case CP_COLOR_R8G8_UNORM: {
      uint8_t *v = (uint8_t *)ptr;
      v[0] = (uint8_t)cp_float_to_unorm8(c.x);
      v[1] = (uint8_t)cp_float_to_unorm8(c.y);
      break;
   }
   case CP_COLOR_R16G16_UNORM: {
      unsigned short *v = (unsigned short *)ptr;
      v[0] = (unsigned short)cp_float_to_unorm16(c.x);
      v[1] = (unsigned short)cp_float_to_unorm16(c.y);
      break;
   }
   case CP_COLOR_R8G8B8A8_UINT: {
      *(uint32_t *)ptr = (CP_FSW_FLOAT_AS_UINT(c.x) & 0xFFu) |
                         ((CP_FSW_FLOAT_AS_UINT(c.y) & 0xFFu) << 8) |
                         ((CP_FSW_FLOAT_AS_UINT(c.z) & 0xFFu) << 16) |
                         ((CP_FSW_FLOAT_AS_UINT(c.w) & 0xFFu) << 24);
      break;
   }
   default: {
      float r = c.x, g = c.y, b = c.z;
      if (encoding == CP_COLOR_R8G8B8A8_SRGB || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         r = cp_linear_to_srgb(r);
         g = cp_linear_to_srgb(g);
         b = cp_linear_to_srgb(b);
      }
      uint32_t p;
      if (encoding == CP_COLOR_B8G8R8A8_UNORM || encoding == CP_COLOR_B8G8R8A8_SRGB) {
         p = cp_float_to_unorm8(b) | (cp_float_to_unorm8(g) << 8) |
             (cp_float_to_unorm8(r) << 16) | (cp_float_to_unorm8(c.w) << 24);
      } else {
         p = cp_float_to_unorm8(r) | (cp_float_to_unorm8(g) << 8) |
             (cp_float_to_unorm8(b) << 16) | (cp_float_to_unorm8(c.w) << 24);
      }
      *(uint32_t *)ptr = p;
      break;
   }
   }
}

/* The array-shaped call the other kernels already make. */
CP_FSW_INLINE void
cp_store_dst(void *ptr, uint32_t encoding, const float *c)
{
   cp_store_dst4(ptr, encoding, CP_FSW_MAKE_FLOAT4(c[0], c[1], c[2], c[3]));
}

CP_FSW_INLINE void
cp_fs_writeback_one_color(const struct cp_fs_writeback_args *ap, uint32_t i,
                          float4 supplied_src, uint32_t source_supplied)
{

   /* A helper lane exists only to supply derivatives to its quad. Without
    * multisampling the mask is 0 or 1; with it, one bit per sample won. */
   uint32_t cov = ap->coverage
      ? ((const unsigned char *)(uintptr_t)ap->coverage)[i] : 1u;
   if (!cov)
      return;

   uint32_t samples = ap->num_samples ? ap->num_samples : 1u;
   uint32_t pixel = ((const uint32_t *)(uintptr_t)ap->pixel_list)[i];

   uint32_t plane = ap->width * ap->height;
   /* The lowest sample this fragment won, for the per-pixel bookkeeping that
    * has no per-sample equivalent. */
   uint32_t first_sample = CP_FSW_FFS((int)cov) - 1;

   /* Once a pixel has taken a fragment, later passes of an alpha-tested draw
    * must leave it alone rather than blend into it again. */
   unsigned char *resolved = (unsigned char *)(uintptr_t)ap->resolved;
   if (resolved && resolved[pixel])
      return;

   /* A discarded fragment contributes neither colour nor depth. Record which
    * triangle it was so the next pass can pick the one behind it. */
   if (ap->discard_mask && ((const unsigned char *)(uintptr_t)ap->discard_mask)[i]) {
      if (ap->reject && ap->reject_pass < ap->reject_layers && ap->visbuf) {
         uint64_t entry = ((const uint64_t *)(uintptr_t)ap->visbuf)
            [(size_t)first_sample * plane + pixel];
         uint32_t tri = ~(uint32_t)(entry & 0xFFFFFFFFu);
         ((uint32_t *)(uintptr_t)ap->reject)[(size_t)pixel * ap->reject_layers +
                                              ap->reject_pass] = tri;
      }
      return;
   }

   if (resolved)
      resolved[pixel] = 1;

   /* Commit the depth the rasterizer tested. Ordered-blending peel passes
    * key the visibility buffer on primitive index, so those take the
    * interpolated gl_FragCoord.z instead. */
   if (ap->depth_write && ap->depthbuf &&
       (ap->visbuf || ap->depth_from_frag_coord)) {
      uint32_t fc_key = 0;
      if (ap->depth_from_frag_coord && ap->frag_coord)
         fc_key = cp_float_to_sortable_uint(
            ((const float4 *)(uintptr_t)ap->frag_coord)[i].z);
      CP_FSW_UNROLL
      for (uint32_t sm = 0; sm < CP_MAX_SAMPLES; sm++) {
         if (sm >= samples || !(cov & (1u << sm)))
            continue;
         size_t at = (size_t)sm * plane + pixel;
         if (ap->depth_from_frag_coord) {
            if (ap->frag_coord)
               ((uint32_t *)(uintptr_t)ap->depthbuf)[at] = fc_key;
            continue;
         }
         uint32_t key = (uint32_t)(((const uint64_t *)(uintptr_t)ap->visbuf)[at] >> 32);
         ((uint32_t *)(uintptr_t)ap->depthbuf)[at] =
            ap->depth_key_invert ? ~key : key;
      }
   }

   if (!ap->color_out)
      return;

   float4 src = supplied_src;
   if (!source_supplied) {
      const float4 *fs_out =
         (const float4 *)((const char *)(uintptr_t)ap->fs_out +
                          (size_t)i * ap->fs_out_stride);
      src = CP_FSW_MAKE_FLOAT4(fs_out->x, fs_out->y, fs_out->z, fs_out->w);
   }

   uint32_t bpp;
   bpp = cp_bytes_per_pixel(ap->color_encoding);
   /*
    * The shader ran once for the pixel, so every sample it covers takes the
    * same colour — that is what per-fragment shading means. Blending reads and
    * writes each sample's own plane, so a fragment covering two of four
    * samples blends into two of them and the resolve weights it accordingly.
    */
   CP_FSW_UNROLL
   for (uint32_t sm = 0; sm < CP_MAX_SAMPLES; sm++) {
   if (sm >= samples || !(cov & (1u << sm)))
      continue;
   void *dst_ptr = (char *)(uintptr_t)ap->color_out +
      (size_t)sm * ap->sample_stride + (size_t)pixel * bpp;

   /* The destination is only read when the equation needs it — an unblended,
    * unmasked write does not, and that is the common case. Values travel as
    * four scalars from here to the store, so that inlining this into a
    * fragment shader adds registers and not a local-memory buffer. */
   float4 out = src;
   if (ap->blend.enable || ap->blend.colormask != 0xF)
      out = cp_blend_resolve4(&ap->blend, src,
                              cp_load_dst4(dst_ptr, ap->color_encoding));


   cp_store_dst4(dst_ptr, ap->color_encoding, out);
   }
}

/* Standalone writeback keeps the old buffer ABI and loads colour only after
 * coverage, discard, depth and attachment checks have admitted the slot. */
CP_FSW_INLINE void
cp_fs_writeback_one(const struct cp_fs_writeback_args *ap, uint32_t i)
{
   cp_fs_writeback_one_color(ap, i, CP_FSW_MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f),
                             0u);
}

#endif /* CP_FS_WRITE_H */
