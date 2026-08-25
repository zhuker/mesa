/*
 * The shader ABI: how a compiled kernel reaches the resources a draw or a
 * dispatch bound.
 *
 * This driver's answer, stated once. It was previously three answers: the same
 * descriptor size defined in the Vulkan front end and again in the backend,
 * the same argument-block base defined beside the kernels and again in the
 * command recorder, and a set of field offsets whose only documentation was
 * the name of the lavapipe structure they were originally copied from. None of
 * the four duplications was asserted against any of the others.
 *
 * The layout is unchanged by this header. What changes is that there is now
 * one place to change it, and a static assertion for every number a CUDA
 * kernel depends on, so a future edit either agrees with the kernels or fails
 * the build.
 *
 * The model itself, for the reader who has not met it:
 *
 *   A kernel takes one argument block. Slots 0..17 are fixed state; from
 *   CP_ARG_UBO_BASE there is one slot per constant buffer, and a descriptor
 *   set occupies exactly one of those slots -- not one per binding, which ran
 *   out on a real pipeline layout. Slot CPVK_UBO_PUSH_SLOT of that range is
 *   the push-constant block, always, whether or not a pipeline has one.
 *
 *   A set's slot holds the address of a flat array of descriptor rows. A
 *   binding is a byte offset into that array, computed at compile time by the
 *   descriptor lowering, and an array element steps by one row.
 *
 *   Graphics adds one indirection the kernels do not: when draws are merged,
 *   the slot array is per draw and the shader selects its own row. See
 *   CP_ARG_SLOT_UBO_TABLE and emit_const_buf_base().
 *
 * The three offsets marked below are read by CUDA code, in cp_sampler.cu and
 * in the generated shader PTX. Everything else in a row is this driver's to
 * arrange.
 */

#ifndef CP_SHADER_ABI_H
#define CP_SHADER_ABI_H

#include "kernels/cp_rast_types.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

/* Constant-buffer slots, and therefore the most descriptor sets a pipeline
 * layout can use: one slot is spent on push constants. */
#define CP_MAX_CONST_BUFFERS  16

/* The push-constant block's slot. Zero because that is where the backend's
 * index form already looked; a layout that numbered sets from zero would have
 * every set's first binding shadowed by it. */
#define CPVK_UBO_PUSH_SLOT  0

/* Names the Vulkan front end uses for the same two kernel-side numbers. */
#define CPVK_ARG_UBO_BASE   CP_ARG_UBO_BASE
#define CPVK_MAX_ARG_BUFS   CP_MAX_CONST_BUFFERS

#define CPVK_DESCRIPTOR_SIZE 64

/*
 * One descriptor.
 *
 * A row is one kind of thing or another -- a buffer, a sampled texture, a
 * storage image -- and the kinds overlap in it rather than each having their
 * own space, because the kernels index rows by a compile-time offset and a
 * uniform stride. The padding is where a kind's fields do not reach.
 */
struct cpvk_descriptor {
   uint64_t base;                     /* +0  buffer base, or image base */
   /*
    * +8 is the extent for a storage image and the bound range in bytes for a
    * buffer. The image paths clamp against it before touching memory --
    * leaving it zero made every coordinate out of bounds and robustness
    * returned zero for a whole image -- and get_ssbo_size reads it as the
    * size that `buffer.length()` divides by its array stride.
    */
   uint32_t width_or_range;           /* +8 */
   uint16_t height;                   /* +12 storage image only */
   uint16_t depth;                    /* +14 storage image only */
   uint8_t  pad0[8];
   uint32_t row_stride;               /* +24 storage image only */
   /*
    * +28 is two things because a row is one kind or the other: a sampler's
    * index into cp_sampler_table, or a storage image's layer stride.
    */
   uint32_t sampler_index_or_img_stride;
   /* Host-only immutable sampler cookie. This overlaps storage-image padding
    * and is ignored by every software/CUDA descriptor consumer. */
   uint64_t sampler_cookie;            /* +32 struct cpvk_sampler * */
   uint32_t base_offset;              /* +40 storage image only */
   uint8_t  pad2[4];
   uint64_t texture_info;             /* +48 struct cp_texture_info * */
   /* Host-only immutable image-view cookie. CUDA kernels never read it. */
   uint64_t image_cookie;              /* +56, 1-based stable device view ID */
};

/* The row's size and the three offsets CUDA code reads. A kernel indexes rows
 * by CPVK_DESCRIPTOR_SIZE and reads these fields at these offsets; the
 * descriptor lowering bakes the same stride into every handle it builds. */
static_assert(sizeof(struct cpvk_descriptor) == CPVK_DESCRIPTOR_SIZE,
              "the kernels index descriptor rows by this stride");
static_assert(offsetof(struct cpvk_descriptor, base) ==
              CP_DESC_IMAGE_BASE_OFFSET, "read by CUDA");
static_assert(offsetof(struct cpvk_descriptor, sampler_index_or_img_stride) ==
              CP_DESC_SAMPLER_INDEX_OFFSET, "read by CUDA");
static_assert(offsetof(struct cpvk_descriptor, texture_info) ==
              CP_DESC_IMAGE_FUNCTIONS_OFFSET, "read by CUDA");

/* The fields only this driver's own host and device code share. Asserted
 * because they are spelled as byte offsets in the backend rather than as
 * member accesses. */
static_assert(offsetof(struct cpvk_descriptor, width_or_range) == 8,
              "get_ssbo_size and the storage image paths read +8");
static_assert(offsetof(struct cpvk_descriptor, height) == 12, "descriptor ABI");
static_assert(offsetof(struct cpvk_descriptor, depth) == 14, "descriptor ABI");
static_assert(offsetof(struct cpvk_descriptor, row_stride) == 24,
              "descriptor ABI");
static_assert(offsetof(struct cpvk_descriptor, sampler_cookie) == 32,
              "hardware texture host metadata");
static_assert(offsetof(struct cpvk_descriptor, base_offset) == 40,
              "descriptor ABI");
static_assert(offsetof(struct cpvk_descriptor, image_cookie) == 56,
              "hardware texture host metadata");

/* The argument block and the per-draw table must describe the same number of
 * slots, or a batched draw reads a neighbouring draw's descriptors. */
static_assert(CP_ARG_UBO_STRIDE == CP_MAX_CONST_BUFFERS,
              "the per-draw UBO table's stride is the slot count");
static_assert(CPVK_UBO_PUSH_SLOT < CP_MAX_CONST_BUFFERS,
              "push constants occupy a constant-buffer slot");
static_assert(CP_ARG_SLOT_UBO_TABLE < CP_ARG_UBO_BASE,
              "the table pointer is fixed state, below the buffer slots");

/* The fused vertex fetch's argument block, likewise fixed state: a slot that
 * collided with a constant buffer would hand the gather a UBO pointer. */
static_assert(CP_ARG_SLOT_VS_FETCH < CP_ARG_UBO_BASE,
              "the fetch block is fixed state, below the buffer slots");
static_assert(CP_ARG_SLOT_VS_FETCH != CP_ARG_SLOT_HW_TEX_TABLE,
              "every fixed argument slot is used by exactly one thing");

#endif /* CP_SHADER_ABI_H */
