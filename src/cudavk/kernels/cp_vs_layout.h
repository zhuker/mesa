/* Uniform VS output layout. S is always the logical byte stride.
 * A first-vertex pointer is a real slot0 address, never a tagged pointer.
 * Tile origins must be 32-byte aligned; callers check capacity before use. */
#ifndef CP_VS_LAYOUT_H
#define CP_VS_LAYOUT_H
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#include <stddef.h>
#endif
#ifndef CP_VS_TILE2
#define CP_VS_TILE2 0
#endif
#if defined(__CUDACC__)
#define CP_VS_INLINE static __device__ __forceinline__
#else
#define CP_VS_INLINE static inline
#endif
/* Applies to allocation origins, NOT valid odd-phase first-vertex slices. */
CP_VS_INLINE int
cp_vs_tile2_origin_valid(uint64_t origin)
{
   return origin != 0 && (origin & 31u) == 0;
}
CP_VS_INLINE uint64_t
cp_vs_tile2_address(uint64_t p, uint64_t stride, uint64_t vertex,
                    uint64_t slot, uint64_t byte)
{
   uint64_t phase = (p >> 4) & 1u;
   uint64_t v = phase + vertex;
   return p - phase * 16u + (v / 2u) * (2u * stride) +
          slot * 32u + (v & 1u) * 16u + byte;
}
CP_VS_INLINE uint64_t
cp_vs_slot_address(uint64_t p, uint64_t stride, uint64_t vertex,
                   uint64_t slot)
{
#if CP_VS_TILE2
   return cp_vs_tile2_address(p, stride, vertex, slot, 0);
#else
   return p + vertex * stride + slot * 16u;
#endif
}
/* These helpers also compile in NVRTC and freestanding helper bitcode.
 * They do not perform allocation or derive a phase from a null pointer. */
CP_VS_INLINE int
cp_vs_tile2_capacity(size_t n, size_t stride, size_t *bytes)
{
   const size_t limit = (size_t)-1;
   if (!stride || (stride & 15u) || stride > limit / 2u)
      return 0;
   size_t tile_bytes = 2u * stride;
   size_t tiles = n / 2u + (n % 2u != 0);
   if (tiles > limit / tile_bytes)
      return 0;
   *bytes = tiles * tile_bytes;
   return 1;
}
CP_VS_INLINE int
cp_vs_tile2_triangle_capacity(size_t prim, size_t stride, size_t *bytes)
{
   if (prim > (size_t)-1 / 3u)
      return 0;
   return cp_vs_tile2_capacity(prim * 3u, stride, bytes);
}
CP_VS_INLINE int
cp_vs_checked_append(size_t *off, size_t bytes, size_t alignment, size_t *start)
{
   size_t limit = (size_t)-1;
   if (!alignment || (alignment & (alignment - 1u)))
      return 0;
   size_t mask = alignment - 1u;
   if (*off > limit - mask)
      return 0;
   size_t aligned = (*off + mask) & ~mask;
   if (bytes > limit - aligned)
      return 0;
   *start = aligned;
   *off = aligned + bytes;
   return 1;
}
CP_VS_INLINE int
cp_vs_tile2_debug_span(uint64_t p, size_t n, size_t stride,
                      uint64_t *origin, size_t *bytes)
{
   if (!n) { *origin = p; *bytes = 0; return 1; }
   if (!p || (p & 15u)) return 0;
   size_t phase = (size_t)((p >> 4) & 1u);
   if (n > (size_t)-1 - phase) return 0;
   if (!cp_vs_tile2_capacity(n + phase, stride, bytes)) return 0;
   *origin = p - phase * 16u;
   return *bytes <= (uint64_t)-1 - *origin;
}
#undef CP_VS_INLINE
#endif
