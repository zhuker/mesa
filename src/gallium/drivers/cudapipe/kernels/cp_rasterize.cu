/*
 * cudapipe triangle rasterizer — visibility buffer approach.
 *
 * Stage 1: 1 thread per triangle. Transform vertices to screen space,
 *           rasterize within bounding box using edge functions,
 *           write depth|triID to visibility buffer via atomicMin.
 *
 * Stage 2 (resolve): 1 thread per pixel. Read winning triID from
 *           visibility buffer, recompute barycentrics, output color.
 *
 * Inspired by CuRast's adaptive rasterizer, simplified for initial bring-up.
 */
#include "cp_rast_types.h"

/* Packed visibility buffer entry: upper 32 bits = depth (as uint for comparison),
 * lower 32 bits = triangle ID. atomicMin gives closest triangle. */
#define PACK_VISBUF(depth_uint, tri_id) (((uint64_t)(depth_uint) << 32) | (uint64_t)(tri_id))
#define VISBUF_DEPTH(packed) ((uint32_t)((packed) >> 32))
#define VISBUF_TRIID(packed) ((uint32_t)((packed) & 0xFFFFFFFF))
#define VISBUF_EMPTY 0xFFFFFFFFFFFFFFFFULL

static __device__ __forceinline__ float
edge_function(float ax, float ay, float bx, float by, float cx, float cy)
{
   return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

static __device__ __forceinline__ uint32_t
float_to_sortable_uint(float f)
{
   uint32_t u = __float_as_uint(f);
   uint32_t mask = -((int32_t)u >> 31) | 0x80000000;
   return u ^ mask;
}

/*
 * Simple rasterizer: 1 thread per triangle, iterates over bounding box.
 * Good for small triangles. For large triangles this is inefficient
 * but correct — optimize later with tile-based approach.
 */
extern "C" __global__ void
cp_rasterize_triangles(struct cp_rasterize_args args)
{
   uint32_t tri_id = blockIdx.x * blockDim.x + threadIdx.x;
   if (tri_id >= args.num_triangles)
      return;

   float4 *positions = (float4 *)(uintptr_t)args.positions;
   uint64_t *visbuf = (uint64_t *)(uintptr_t)args.framebuffer;

   /* Load triangle vertices (already in clip space from VS) */
   float4 v0 = positions[tri_id * 3 + 0];
   float4 v1 = positions[tri_id * 3 + 1];
   float4 v2 = positions[tri_id * 3 + 2];

   /* Perspective divide: clip → NDC */
   float inv_w0 = 1.0f / v0.w;
   float inv_w1 = 1.0f / v1.w;
   float inv_w2 = 1.0f / v2.w;

   float ndc_x0 = v0.x * inv_w0, ndc_y0 = v0.y * inv_w0, ndc_z0 = v0.z * inv_w0;
   float ndc_x1 = v1.x * inv_w1, ndc_y1 = v1.y * inv_w1, ndc_z1 = v1.z * inv_w1;
   float ndc_x2 = v2.x * inv_w2, ndc_y2 = v2.y * inv_w2, ndc_z2 = v2.z * inv_w2;

   /* NDC → screen space */
   float sx0 = (ndc_x0 * 0.5f + 0.5f) * args.vp_w + args.vp_x;
   float sy0 = (ndc_y0 * 0.5f + 0.5f) * args.vp_h + args.vp_y;
   float sx1 = (ndc_x1 * 0.5f + 0.5f) * args.vp_w + args.vp_x;
   float sy1 = (ndc_y1 * 0.5f + 0.5f) * args.vp_h + args.vp_y;
   float sx2 = (ndc_x2 * 0.5f + 0.5f) * args.vp_w + args.vp_x;
   float sy2 = (ndc_y2 * 0.5f + 0.5f) * args.vp_h + args.vp_y;

   /* Backface culling */
   float area = edge_function(sx0, sy0, sx1, sy1, sx2, sy2);
   if (args.cull_mode == 1 && area > 0) return; /* cull front (CW positive) */
   if (args.cull_mode == 2 && area < 0) return; /* cull back (CCW negative) */
   if (area == 0.0f) return; /* degenerate */

   float inv_area = 1.0f / area;

   /* Bounding box (clipped to viewport) */
   float min_x = fminf(fminf(sx0, sx1), sx2);
   float min_y = fminf(fminf(sy0, sy1), sy2);
   float max_x = fmaxf(fmaxf(sx0, sx1), sx2);
   float max_y = fmaxf(fmaxf(sy0, sy1), sy2);

   int ix_min = max((int)floorf(min_x), 0);
   int iy_min = max((int)floorf(min_y), 0);
   int ix_max = min((int)ceilf(max_x), (int)args.width - 1);
   int iy_max = min((int)ceilf(max_y), (int)args.height - 1);

   /* Iterate over bounding box */
   for (int py = iy_min; py <= iy_max; py++) {
      for (int px = ix_min; px <= ix_max; px++) {
         float cx = (float)px + 0.5f;
         float cy = (float)py + 0.5f;

         /* Barycentric coordinates */
         float w0 = edge_function(sx1, sy1, sx2, sy2, cx, cy) * inv_area;
         float w1 = edge_function(sx2, sy2, sx0, sy0, cx, cy) * inv_area;
         float w2 = 1.0f - w0 - w1;

         /* Inside test */
         if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
            continue;

         /* Interpolate depth */
         float depth = w0 * ndc_z0 + w1 * ndc_z1 + w2 * ndc_z2;
         depth = depth * 0.5f + 0.5f; /* NDC [-1,1] → [0,1] */

         /* Pack and atomicMin into visibility buffer */
         uint32_t depth_uint = float_to_sortable_uint(depth);
         uint64_t packed = PACK_VISBUF(depth_uint, tri_id);
         atomicMin(&visbuf[py * args.width + px], packed);
      }
   }
}

/*
 * Clear visibility buffer to "empty" (max depth, invalid triID)
 */
extern "C" __global__ void
cp_clear_visbuf(uint64_t *visbuf, uint32_t width, uint32_t height)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x < width && y < height)
      visbuf[y * width + x] = VISBUF_EMPTY;
}

/*
 * Resolve: read visibility buffer, output flat white for any hit pixel.
 * TODO: call compiled fragment shader, interpolate varyings.
 */
extern "C" __global__ void
cp_resolve_visbuf(uint64_t *visbuf, uint32_t *color_out,
                  uint32_t width, uint32_t height)
{
   uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height)
      return;

   uint64_t entry = visbuf[y * width + x];
   if (entry == VISBUF_EMPTY) {
      color_out[y * width + x] = 0xFF000000; /* black, opaque */
   } else {
      /* For now, output white for any triangle hit */
      color_out[y * width + x] = 0xFFFFFFFF; /* white, opaque */
   }
}
