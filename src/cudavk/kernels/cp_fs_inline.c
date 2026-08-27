/* Freestanding LLVM-bitcode entry for same-module fragment interpolation. */
#include <stddef.h>
#include <stdbool.h>
#define static_assert _Static_assert

#define __CUDACC__ 1
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
#include "cp_rast_types.h"
#undef __forceinline__
#undef __device__

/* This bitcode is embedded and linked later, so an ABI drift cannot be left
 * for runtime pointer decoding to discover. Keep sentinel offsets across the
 * early inputs, point state, A-buffer state, segment table and fused tail. */
_Static_assert(sizeof(struct cp_fs_interp_args) == 416,
               "cp_fs_interp_args ABI size");
_Static_assert(_Alignof(struct cp_fs_interp_args) == 8,
               "cp_fs_interp_args ABI alignment");
_Static_assert(offsetof(struct cp_fs_interp_args, fs_in) == 48,
               "cp_fs_interp_args.fs_in ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, input_vs_slot) == 108,
               "cp_fs_interp_args.input_vs_slot ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, point_mode) == 192,
               "cp_fs_interp_args.point_mode ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, abuf_quad_prim) == 256,
               "cp_fs_interp_args.abuf_quad_prim ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, draw_slices) == 320,
               "cp_fs_interp_args.draw_slices ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, seg_ranges) == 368,
               "cp_fs_interp_args.seg_ranges ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, out_prim_list) == 392,
               "cp_fs_interp_args.out_prim_list ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, fused_direct) == 400,
               "cp_fs_interp_args.fused_direct ABI");
_Static_assert(offsetof(struct cp_fs_interp_args, depth_scale) == 404,
               "cp_fs_interp_args.depth_scale ABI");

_Static_assert(sizeof(struct cp_seg_range) == 48, "cp_seg_range ABI size");
_Static_assert(_Alignof(struct cp_seg_range) == 8,
               "cp_seg_range ABI alignment");
_Static_assert(offsetof(struct cp_seg_range, positions) == 0,
               "cp_seg_range.positions ABI");
_Static_assert(offsetof(struct cp_seg_range, draw_slices) == 16,
               "cp_seg_range.draw_slices ABI");
_Static_assert(offsetof(struct cp_seg_range, num_draw_slices) == 24,
               "cp_seg_range.num_draw_slices ABI");
_Static_assert(offsetof(struct cp_seg_range, prim_base) == 28,
               "cp_seg_range.prim_base ABI");
_Static_assert(offsetof(struct cp_seg_range, row_base) == 36,
               "cp_seg_range.row_base ABI");
_Static_assert(offsetof(struct cp_seg_range, prim_shift) == 40,
               "cp_seg_range.prim_shift ABI");

/* CUDA's float4 ABI is 16-byte sized and aligned. */
typedef struct __attribute__((aligned(16))) float4 {
   float x, y, z, w;
} float4;

static inline __attribute__((always_inline)) float4
cp_make_float4(float x, float y, float z, float w)
{
   float4 v = { x, y, z, w };
   return v;
}

#define CP_INTERP_INLINE static inline __attribute__((always_inline))
#define CP_INTERP_FIXED_SLOTS 1
#define CP_FMUL_RN(a, b) ((a) * (b))
#define CP_FSUB_RN(a, b) ((a) - (b))
#define CP_MAKE_FLOAT4(x, y, z, w) cp_make_float4((x), (y), (z), (w))
#include "cp_fs_interp.h"
#undef CP_MAKE_FLOAT4
#undef CP_FSUB_RN
#undef CP_FMUL_RN
#undef CP_INTERP_FIXED_SLOTS
#undef CP_INTERP_INLINE

extern uint32_t cp_nvvm_tid_x(void)
   __asm("llvm.nvvm.read.ptx.sreg.tid.x");
extern uint32_t cp_nvvm_shfl_idx(uint32_t, uint32_t, uint32_t, uint32_t)
   __asm("llvm.nvvm.shfl.sync.idx.i32");

static inline __attribute__((always_inline)) uint32_t
cp_quad_broadcast_u32(uint32_t value, uint32_t src_lane)
{
   return cp_nvvm_shfl_idx(0xFFFFFFFFu, value, src_lane, 0x1Fu);
}

static inline __attribute__((always_inline)) float
cp_quad_broadcast_f32(float value, uint32_t src_lane)
{
   union { float f; uint32_t u; } bits = { .f = value };
   bits.u = cp_quad_broadcast_u32(bits.u, src_lane);
   return bits.f;
}

/* This is linked into the generated module and must disappear after the
 * always-inline pass. local_fs_in is a function-entry alloca owned by main. */
__attribute__((always_inline)) int
cp_fs_inline_lane(const struct cp_fs_interp_args *source, uint32_t slot,
                  float4 *local_fs_in, uint32_t num_inputs,
                  uint32_t live_slots, int32_t pntc_input,
                  int32_t pos_input)
{
   struct cp_fs_interp_args args = *source;
   if (num_inputs > CP_MAX_FS_INPUTS)
      num_inputs = CP_MAX_FS_INPUTS;

   if (source->fused_direct) {
      if (slot >= args.max_pixels)
         return 0;
      uint32_t pixel = ((const uint32_t *)(uintptr_t)args.pixel_list)[slot];
      uint32_t gprim =
         ((const uint32_t *)(uintptr_t)args.out_prim_list)[slot >> 2];
      unsigned char *coverage = (unsigned char *)(uintptr_t)args.coverage;
      bool ok = cp_resolve_seg_range(&args, gprim);
      if (ok) {
         uint32_t prim = gprim - args.abuf_prim_base;
         struct cp_interp_tri tri;
         ok = cp_interp_setup(&args, prim, &tri) &&
              cp_interp_pixel_prepared(&args, prim, pixel, slot, &tri,
                                       local_fs_in, live_slots, num_inputs,
                                       pntc_input, pos_input);
      }
      if (!ok && coverage)
         coverage[slot] = 0;
      return ok ? 1 : 0;
   }

   uint32_t iq = slot >> 2;
   uint32_t lane = slot & 3u;
   uint32_t nq = args.abuf_num_quads;
   if (args.num_quads_dev) {
      uint32_t exact = *(const uint32_t *)(uintptr_t)args.num_quads_dev;
      if (exact < nq)
         nq = exact;
   }
   if (iq >= nq)
      return 0;

   uint32_t list_base = args.quad_list_base_dev
      ? *(const uint32_t *)(uintptr_t)args.quad_list_base_dev
      : args.quad_list_base;
   uint32_t q = args.quad_list
      ? ((const uint32_t *)(uintptr_t)args.quad_list)[list_base + iq] : iq;
   uint32_t block = ((const uint32_t *)(uintptr_t)args.abuf_quad_block)[q];
   uint32_t gprim = ((const uint32_t *)(uintptr_t)args.abuf_quad_prim)[q];
   if (!cp_resolve_seg_range(&args, gprim))
      return 0;

   uint32_t prim = gprim - args.abuf_prim_base;
   uint32_t qx = (block % args.quad_width) * 2;
   uint32_t qy = (block / args.quad_width) * 2;
   uint32_t x = qx + (lane & 1u);
   uint32_t y = qy + (lane >> 1);
   bool in_fb = x < args.width && y < args.height;
   uint32_t pixel = (y < args.height ? y : args.height - 1) * args.width +
                    (x < args.width ? x : args.width - 1);
   uint32_t mask = ((const unsigned char *)(uintptr_t)args.abuf_quad_mask)[q];

   uint32_t *rows = (uint32_t *)(uintptr_t)args.out_batch_rows;
   if (rows) {
      uint32_t row = 0;
      if (args.num_draw_slices > 1) {
         const struct cp_draw_slice *s =
            (const struct cp_draw_slice *)(uintptr_t)args.draw_slices;
         uint32_t vert = (prim >> args.prim_shift) * 3u;
         uint32_t lo = 0, hi = args.num_draw_slices - 1;
         while (lo < hi) {
            uint32_t mid = (lo + hi + 1u) >> 1;
            if (s[mid].vert_begin <= vert)
               lo = mid;
            else
               hi = mid - 1;
         }
         row = lo;
      }
      rows[slot] = row + args.row_base;
   }

   struct cp_interp_tri tri = {0};
   bool tri_ok = false;
   if (lane == 0)
      tri_ok = cp_interp_setup(&args, prim, &tri);
   uint32_t src_lane = cp_nvvm_tid_x() & ~3u;
#define CP_BCAST_F(field) tri.field = cp_quad_broadcast_f32(tri.field, src_lane)
#define CP_BCAST_I(field) tri.field = (int)cp_quad_broadcast_u32((uint32_t)tri.field, src_lane)
   CP_BCAST_F(sx0); CP_BCAST_F(sy0); CP_BCAST_F(sx1); CP_BCAST_F(sy1);
   CP_BCAST_F(sx2); CP_BCAST_F(sy2);
   CP_BCAST_F(ndc_z0); CP_BCAST_F(ndc_z1); CP_BCAST_F(ndc_z2);
   CP_BCAST_F(inv_w0); CP_BCAST_F(inv_w1); CP_BCAST_F(inv_w2);
   CP_BCAST_F(inv_area);
   CP_BCAST_I(vidx1); CP_BCAST_I(vidx2);
   uint32_t base_lo = cp_quad_broadcast_u32((uint32_t)tri.base, src_lane);
   uint32_t base_hi = cp_quad_broadcast_u32((uint32_t)(tri.base >> 32), src_lane);
   tri.base = (uint64_t)base_lo | ((uint64_t)base_hi << 32);
   tri.front = cp_quad_broadcast_u32((uint32_t)tri.front, src_lane) != 0;
   tri_ok = cp_quad_broadcast_u32((uint32_t)tri_ok, src_lane) != 0;
#undef CP_BCAST_I
#undef CP_BCAST_F
   bool ok = tri_ok &&
             cp_interp_pixel_prepared(&args, prim, pixel, slot, &tri,
                                      local_fs_in, live_slots, num_inputs,
                                       pntc_input, pos_input);
   if (args.coverage)
      ((unsigned char *)(uintptr_t)args.coverage)[slot] =
         (ok && in_fb && (mask & (1u << lane))) ? 1u : 0u;
   return ok ? 1 : 0;
}
