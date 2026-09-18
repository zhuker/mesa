/* Freestanding LLVM-bitcode entry for same-module vertex fetch. */
#include <stddef.h>
#include <stdbool.h>
#define static_assert _Static_assert

#define __CUDACC__ 1
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
#include "cp_rast_types.h"
#undef __forceinline__
#undef __device__

/*
 * This bitcode is embedded and linked into the generated vertex shader, so an
 * ABI drift between the host that fills the block and the code that decodes it
 * cannot be left for runtime to discover. Sentinel offsets across the element
 * tables, the batch tables and the iteration-26 counter seeds.
 */
_Static_assert(sizeof(struct cp_vertex_fetch_args) == 944,
               "cp_vertex_fetch_args ABI size");
_Static_assert(_Alignof(struct cp_vertex_fetch_args) == 8,
               "cp_vertex_fetch_args ABI alignment");
_Static_assert(offsetof(struct cp_vertex_fetch_args, output) == 0,
               "cp_vertex_fetch_args.output ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, elem_conv) == 528,
               "cp_vertex_fetch_args.elem_conv ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, elem_instance_divisor) == 720,
               "cp_vertex_fetch_args.elem_instance_divisor ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, num_verts) == 788,
               "cp_vertex_fetch_args.num_verts ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, vs_in_stride) == 792,
               "cp_vertex_fetch_args.vs_in_stride ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, draw_slices) == 848,
               "cp_vertex_fetch_args.draw_slices ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, num_draw_slices) == 856,
               "cp_vertex_fetch_args.num_draw_slices ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, out_batch_rows) == 864,
               "cp_vertex_fetch_args.out_batch_rows ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, elem_bases) == 872,
               "cp_vertex_fetch_args.elem_bases ABI");
_Static_assert(offsetof(struct cp_vertex_fetch_args, seed_counts) == 880,
               "cp_vertex_fetch_args.seed_counts ABI");

_Static_assert(sizeof(struct cp_draw_slice) == 24, "cp_draw_slice ABI size");
_Static_assert(_Alignof(struct cp_draw_slice) == 4,
               "cp_draw_slice ABI alignment");

/* CUDA's float4 ABI is 16-byte sized and aligned. */
typedef struct __attribute__((aligned(16))) float4 {
   float x, y, z, w;
} float4;

static inline __attribute__((always_inline)) uint32_t
cp_vs_float_as_uint(float f)
{
   union { float f; uint32_t u; } bits = { .f = f };
   return bits.u;
}

#define __float_as_uint(f) cp_vs_float_as_uint(f)
#define CP_VF_INLINE static inline __attribute__((always_inline))
/* The gather's destination is registers here, not a buffer: nothing memset it
 * and nothing may be written to it twice under two different types. The
 * standalone kernel keeps the shape it always had. */
#define CP_VF_FIXED_SLOTS 1
#define CP_VF_UNROLL _Pragma("clang loop unroll(full)")
#include "cp_vf_lane.h"
#undef CP_VF_UNROLL
#undef CP_VF_FIXED_SLOTS
#undef CP_VF_INLINE
#undef __float_as_uint

/*
 * Linked into the generated vertex shader's module and inlined away before
 * optimisation. The shader owns the destination -- a function-entry alloca
 * for the slots, three more for the ids and the row -- which is what lets
 * SROA promote the gathered attributes into registers and delete the packed
 * input buffer entirely. A surviving call in the emitted PTX is a build
 * failure, not a fallback: that was iteration 4's failure mode.
 *
 * Three entry points rather than one, because the shape of the call is what
 * keeps the array promotable. The identity runs once; the gather runs once
 * per input slot the shader reads, with the element index a constant supplied
 * by the backend, so nothing indexes the array dynamically.
 */
__attribute__((always_inline)) int
cp_vs_fetch_ids(const struct cp_vertex_fetch_args *source, uint32_t v,
                uint32_t *out_vertex_id, uint32_t *out_instance_id,
                uint32_t *out_row)
{
   *out_vertex_id = 0;
   *out_instance_id = 0;
   *out_row = 0;
   return cp_vf_lane_ids(source, v, out_vertex_id, out_instance_id, out_row);
}

__attribute__((always_inline)) void
cp_vs_fetch_element(const struct cp_vertex_fetch_args *source, uint32_t e,
                    uint32_t vertex_id, uint32_t instance_id, uint32_t row,
                    unsigned char *slot)
{
   cp_vf_lane_element(source, e, vertex_id, instance_id, row, slot);
}

/*
 * The counter seeds iteration 26 moved onto the fetch launch (cp_vertex_fetch
 * seeds the clipper's output counter and the three raster queue counters). A
 * fused draw has no fetch launch, so the seeding rides on this kernel
 * instead -- one thread of the grid, before the bounds check, so that it does
 * not depend on that thread having a vertex to shade. Without this, iteration
 * 26 S2 would be silently reverted for every admitted shader: the counters
 * would go back to their own clears and nothing would fail.
 */
__attribute__((always_inline)) void
cp_vs_fetch_seed(const struct cp_vertex_fetch_args *source, uint32_t gtid)
{
   if (gtid != 0)
      return;
   if (source->seed_counts) {
      uint32_t *counts = (uint32_t *)(uintptr_t)source->seed_counts;
      counts[0] = 0;
      counts[1] = 0;
      counts[2] = 0;
   }
   if (source->seed_clip_count)
      *(uint32_t *)(uintptr_t)source->seed_clip_count = source->clip_seed;
}

/* Static layout is scalar: no aggregate alloca or pointer ABI. */
__attribute__((always_inline)) void
cp_vs_fetch_element_layout(const struct cp_vertex_fetch_args *source, uint32_t e,
   uint32_t vertex_id, uint32_t instance_id, uint32_t row, unsigned char *slot,
   uint32_t num_elements, uint32_t vertex_buffer_index, uint32_t src_offset, uint32_t src_stride,
                   uint32_t instance_divisor, uint32_t attr_size, uint32_t nr_chan,
                   uint32_t chan_bytes, uint32_t swizzle, uint32_t conv,
                   uint32_t fill_w)
{
   cp_vf_lane_element_layout(source, e, vertex_id, instance_id, row, slot,
      num_elements, vertex_buffer_index, src_offset, src_stride, instance_divisor, attr_size, nr_chan, chan_bytes, swizzle, conv, fill_w);
}

/* These entry points are emitted only for the instrumented module. They never
 * run the shader or recompute its gather; slot points at its actual alloca. */
#include "cp_vfetch_eq.h"
_Static_assert(offsetof(struct cp_vertex_fetch_args, eq_rows) == 904,
               "fetch validation trailer ABI");
_Static_assert(sizeof(struct cp_vfetch_eq_row) == 528, "oracle row ABI");
_Static_assert(sizeof(struct cp_vfetch_eq_report) == 136, "report ABI");
#define EQ_INLINE static inline __attribute__((always_inline))
EQ_INLINE void cp_eq_add(uint64_t *p, uint64_t n)
{
   __atomic_fetch_add(p, n, __ATOMIC_RELAXED);
}
EQ_INLINE void cp_eq_first(const struct cp_vertex_fetch_args *a, uint32_t v,
                           uint32_t e, uint32_t w, uint32_t ref, uint32_t got)
{
   struct cp_vfetch_eq_report *r = (void *)(uintptr_t)a->eq_report;
   uint64_t zero = 0;
   if (__atomic_compare_exchange_n(&r->first_claim, &zero, 1, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
      r->first_dispatch = a->eq_dispatch;
      r->first_invocation = v;
      r->first_slot = e;
      r->first_word = w;
      r->first_expected = ref;
      r->first_actual = got;
   }
}
__attribute__((always_inline)) uint32_t
cp_vs_fetch_eq_ids(const struct cp_vertex_fetch_args *a, uint32_t v,
                   uint32_t valid, uint32_t vertex, uint32_t instance,
                   uint32_t row)
{
   struct cp_vfetch_eq_report *r = (void *)(uintptr_t)a->eq_report;
   const struct cp_vfetch_eq_row *q =
      (const void *)(uintptr_t)a->eq_rows;
   q += v;
   cp_eq_add(&r->invocations, 1);
   if (valid) cp_eq_add(&r->valid, 1);
   uint32_t dv = !!valid != !!q->valid;
   uint32_t di = valid && q->valid &&
      (vertex != q->vertex_id || instance != q->instance_id || row != q->row);
   if (dv) {
      cp_eq_add(&r->diff_validity, 1);
      cp_eq_first(a, v, UINT32_MAX, 0, q->valid, valid);
   }
   if (di) {
      cp_eq_add(&r->diff_ids, 1);
      if (vertex != q->vertex_id)
         cp_eq_first(a, v, UINT32_MAX, 1, q->vertex_id, vertex);
      else if (instance != q->instance_id)
         cp_eq_first(a, v, UINT32_MAX, 2, q->instance_id, instance);
      else cp_eq_first(a, v, UINT32_MAX, 3, q->row, row);
   }
   /* A rejected candidate never reaches the slot or final hook. */
   if (!valid && (dv || di)) cp_eq_add(&r->differing_invocations, 1);
   return dv || di;
}
__attribute__((always_inline)) uint32_t
cp_vs_fetch_eq_slot(const struct cp_vertex_fetch_args *a, uint32_t v,
                    uint32_t e, unsigned char *slot)
{
   struct cp_vfetch_eq_report *r = (void *)(uintptr_t)a->eq_report;
   const struct cp_vfetch_eq_row *q =
      (const void *)(uintptr_t)a->eq_rows;
   /* Fault the LIVE result, not an oracle or a mismatch counter. Restore it
    * after comparison so the detector control can finish a complete replay.
    * Character accesses alias the float/int slot representation legally. */
   volatile unsigned char *live = slot;
   if (a->eq_break) {
      live[0] ^= 1;
      cp_eq_add(&r->injections, 1);
   }
   uint32_t bits[4];
   __builtin_memcpy(bits, slot, sizeof(bits));
   uint32_t n = 0;
   for (unsigned w = 0; w < 4; w++) {
      if (bits[w] != q[v].slots[e][w]) {
         n++;
         cp_eq_first(a, v, e, w, q[v].slots[e][w], bits[w]);
      }
   }
   if (a->eq_break) live[0] ^= 1;
   cp_eq_add(&r->slots, 1);
   cp_eq_add(&r->words, 4);
   if (n) {
      cp_eq_add(&r->diff_slots, 1);
      cp_eq_add(&r->diff_words, n);
   }
   return n != 0;
}
__attribute__((always_inline)) void
cp_vs_fetch_eq_finish(const struct cp_vertex_fetch_args *a, uint32_t different)
{
   struct cp_vfetch_eq_report *r = (void *)(uintptr_t)a->eq_report;
   if (different) cp_eq_add(&r->differing_invocations, 1);
}
