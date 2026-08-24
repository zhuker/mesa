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
_Static_assert(sizeof(struct cp_vertex_fetch_args) == 904,
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

_Static_assert(sizeof(struct cp_draw_slice) == 16, "cp_draw_slice ABI size");
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
#include "cp_vf_lane.h"
#undef CP_VF_INLINE
#undef __float_as_uint

/*
 * Linked into the generated vertex shader's module and inlined away before
 * optimisation; `slots` is a function-entry alloca owned by main, which is
 * what lets SROA promote the gathered attributes into registers and delete
 * the packed input buffer entirely. A surviving call in the emitted PTX is a
 * build failure, not a fallback -- that was iteration 4's failure mode.
 */
__attribute__((always_inline)) int
cp_vs_fetch_lane(const struct cp_vertex_fetch_args *source, uint32_t v,
                 unsigned char *slots, uint32_t *out_vertex_id,
                 uint32_t *out_instance_id, uint32_t *out_row)
{
   return cp_vf_lane(source, v, slots, out_vertex_id, out_instance_id,
                     out_row);
}
