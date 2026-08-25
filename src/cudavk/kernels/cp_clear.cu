/*
 * cudavk clear kernel — fills a buffer with a constant value.
 */
#include "cp_rast_types.h"

extern "C" __global__ void
cp_clear_kernel(struct cp_clear_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= args.width || y >= args.height)
      return;

   uint8_t *base = (uint8_t *)(uintptr_t)args.target;
   uint8_t *pixel = base + y * args.stride + x * args.pixel_size;

   if (args.pixel_size == 4) {
      *(uint32_t *)pixel = args.clear_value[0];
   } else if (args.pixel_size == 8) {
      ((uint32_t *)pixel)[0] = args.clear_value[0];
      ((uint32_t *)pixel)[1] = args.clear_value[1];
   } else if (args.pixel_size == 16) {
      ((uint32_t *)pixel)[0] = args.clear_value[0];
      ((uint32_t *)pixel)[1] = args.clear_value[1];
      ((uint32_t *)pixel)[2] = args.clear_value[2];
      ((uint32_t *)pixel)[3] = args.clear_value[3];
   } else if (args.pixel_size == 1) {
      *pixel = (uint8_t)args.clear_value[0];
   } else if (args.pixel_size == 2) {
      *(uint16_t *)pixel = (uint16_t)args.clear_value[0];
   }
}

extern "C" __global__ void
cp_clear_depth_kernel(struct cp_clear_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= args.width || y >= args.height)
      return;

   uint8_t *base = (uint8_t *)(uintptr_t)args.target;
   uint8_t *pixel = base + y * args.stride + x * args.pixel_size;

   if (args.pixel_size == 4) {
      /* D32F or D24X8 */
      *(uint32_t *)pixel = args.clear_value[0];
   } else if (args.pixel_size == 2) {
      /* D16 */
      *(uint16_t *)pixel = (uint16_t)args.clear_value[0];
   }
}



extern "C" __global__ void
cp_depth_attachment_load(struct cp_depth_attachment_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   uint32_t sample = blockIdx.z;
   if (x >= args.width || y >= args.height || sample >= args.samples)
      return;
   const uint8_t *src = (const uint8_t *)(uintptr_t)args.image +
                        (uint64_t)sample * args.sample_stride +
                        (uint64_t)y * args.row_stride +
                        (uint64_t)x * args.pixel_stride;
   uint32_t *dst = (uint32_t *)(uintptr_t)args.depthbuf;
   uint32_t packed = *(const uint32_t *)src;
   uint32_t bits;
   if (args.format == 2) {
      float depth = (float)(packed & 0x00ffffffu) * (1.0f / 16777215.0f);
      bits = __float_as_uint(depth);
   } else {
      bits = packed;
   }
   dst[((uint64_t)sample * args.height + y) * args.width + x] =
      bits ^ 0x80000000u;
}

extern "C" __global__ void
cp_depth_attachment_store(struct cp_depth_attachment_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   uint32_t sample = blockIdx.z;
   if (x >= args.width || y >= args.height || sample >= args.samples)
      return;
   const uint32_t *src = (const uint32_t *)(uintptr_t)args.depthbuf;
   uint8_t *dst = (uint8_t *)(uintptr_t)args.image +
                  (uint64_t)sample * args.sample_stride +
                  (uint64_t)y * args.row_stride +
                  (uint64_t)x * args.pixel_stride;
   uint32_t bits =
      src[((uint64_t)sample * args.height + y) * args.width + x] ^ 0x80000000u;
   if (args.format == 2) {
      float depth = __uint_as_float(bits);
      depth = depth < 0.0f ? 0.0f : (depth > 1.0f ? 1.0f : depth);
      uint32_t old = *(uint32_t *)dst;
      uint32_t stencil = args.stencil_clear ? (args.stencil_value & 0xffu) << 24
                                            : (old & 0xff000000u);
      uint32_t d24 = __float2uint_rn(depth * 16777215.0f);
      *(uint32_t *)dst = stencil | (d24 & 0x00ffffffu);
   } else {
      *(uint32_t *)dst = bits;
      /* D32_SFLOAT_S8_UINT keeps its stencil byte in the word after the
       * depth; the depth store leaves it alone unless it was cleared. */
      if (args.format == 1 && args.stencil_clear && args.pixel_stride >= 8)
         dst[4] = (uint8_t)args.stencil_value;
   }
}


static __device__ __forceinline__ uint8_t
cp_expand5(uint32_t v) { return (uint8_t)((v << 3) | (v >> 2)); }
static __device__ __forceinline__ uint8_t
cp_expand6(uint32_t v) { return (uint8_t)((v << 2) | (v >> 4)); }

static __device__ __forceinline__ uint32_t
cp_bc_color(uint16_t c)
{
   uint32_t r = cp_expand5((c >> 11) & 31);
   uint32_t g = cp_expand6((c >> 5) & 63);
   uint32_t b = cp_expand5(c & 31);
   return r | (g << 8) | (b << 16) | 0xff000000u;
}

static __device__ __forceinline__ uint32_t
cp_bc1_decode(const uint8_t *block, unsigned pixel, bool force_four)
{
   uint16_t c0 = *(const uint16_t *)(block + 0);
   uint16_t c1 = *(const uint16_t *)(block + 2);
   uint32_t p[4] = { cp_bc_color(c0), cp_bc_color(c1), 0, 0 };
   uint32_t mode = *(const uint32_t *)(block + 4);
   for (unsigned ch = 0; ch < 3; ch++) {
      unsigned shift = ch * 8;
      unsigned a = (p[0] >> shift) & 255;
      unsigned b = (p[1] >> shift) & 255;
      if (force_four || c0 > c1) {
         p[2] |= ((2 * a + b) / 3) << shift;
         p[3] |= ((a + 2 * b) / 3) << shift;
      } else {
         p[2] |= ((a + b) / 2) << shift;
      }
   }
   p[2] |= 0xff000000u;
   p[3] |= 0xff000000u; /* BC1 RGB is opaque even in the three-colour mode. */
   return p[(mode >> (2 * pixel)) & 3];
}

static __device__ __forceinline__ uint8_t
cp_bc3_alpha(const uint8_t *block, unsigned pixel)
{
   uint8_t a[8] = { block[0], block[1], 0, 0, 0, 0, 0, 0 };
   if (a[0] > a[1]) {
      for (unsigned i = 1; i <= 6; i++)
         a[i + 1] = (uint8_t)(((7 - i) * a[0] + i * a[1]) / 7);
   } else {
      for (unsigned i = 1; i <= 4; i++)
         a[i + 1] = (uint8_t)(((5 - i) * a[0] + i * a[1]) / 5);
      a[6] = 0;
      a[7] = 255;
   }
   uint64_t bits = 0;
   for (unsigned i = 0; i < 6; i++)
      bits |= (uint64_t)block[2 + i] << (8 * i);
   return a[(bits >> (3 * pixel)) & 7];
}

extern "C" __global__ void
cp_cache_convert(struct cp_cache_convert_args args)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   uint32_t z = blockIdx.z;
   if (x >= args.width || y >= args.height || z >= args.depth)
      return;
   const uint8_t *src = (const uint8_t *)(uintptr_t)args.src;
   uint64_t surface = args.surface;
   if (args.format == 1) {
      uint32_t packed = *(const uint32_t *)(src + (uint64_t)z * args.src_slice +
                           (uint64_t)y * args.src_pitch + x * 4);
      ushort4 value = make_ushort4((packed & 0x7ffu) << 4,
                                   ((packed >> 11) & 0x7ffu) << 4,
                                   ((packed >> 22) & 0x3ffu) << 5,
                                   0x3c00u);
      if (args.target == 0)
         surf2Dwrite(value, surface, x * sizeof(value), y);
      else if (args.target == 1)
         surf2DLayeredwrite(value, surface, x * sizeof(value), y, z);
      else if (args.target == 3)
         surfCubemapwrite(value, surface, x * sizeof(value), y, z);
      else
         surf3Dwrite(value, surface, x * sizeof(value), y, z);
      return;
   }

   unsigned block_bytes = args.format == 2 ? 8 : 16;
   const uint8_t *block = src + (uint64_t)z * args.src_slice +
      (uint64_t)(y >> 2) * args.src_pitch + (uint64_t)(x >> 2) * block_bytes;
   unsigned pixel = (y & 3) * 4 + (x & 3);
   uint32_t value = args.format == 2
      ? cp_bc1_decode(block, pixel, false)
      : cp_bc1_decode(block + 8, pixel, true);
   if (args.format == 3)
      value = (value & 0x00ffffffu) |
              ((uint32_t)cp_bc3_alpha(block, pixel) << 24);
   if (args.target == 0)
      surf2Dwrite(value, surface, x * sizeof(value), y);
   else if (args.target == 1)
      surf2DLayeredwrite(value, surface, x * sizeof(value), y, z);
   else if (args.target == 3)
      surfCubemapwrite(value, surface, x * sizeof(value), y, z);
   else
      surf3Dwrite(value, surface, x * sizeof(value), y, z);
}
