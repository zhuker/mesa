/*
 * The NIR options the CUDA backend needs, in one place.
 *
 * Shared between the Gallium-hosted screen and the native Vulkan driver:
 * two copies of this list would drift, and every entry below is a decision
 * with a measurement behind it (see the comments) rather than a default.
 */

#ifndef CP_NIR_OPTIONS_H
#define CP_NIR_OPTIONS_H

#include "nir.h"

static const struct nir_shader_compiler_options cp_nir_options = {
   .lower_scmp = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_flrp16 = true,
   .lower_fsat = true,
   .lower_bitfield_insert = true,
   .lower_bitfield_extract = true,
   .lower_fdph = true,
   .lower_fmod = true,
   /* NVPTX has no libcall for pow or the trig functions, so llvm.pow and
    * llvm.sin fail instruction selection and take the backend down with them.
    * Let NIR express them in terms of exp2/log2 and the fractional-turn
    * reductions, which do select. */
   /* Unlike sincos, pow stays lowered. Routing it to the CUDA library version
    * was tried and changed the image without moving it closer to the reference,
    * so the approximation is accurate enough here and is not worth a call. */
   .lower_fpow = true,
   /* Keep fsin/fcos: the backend routes them to the CUDA library versions,
    * which are far more accurate than NIR's lowered polynomial. That accuracy
    * matters where a shader rotates a position rather than a direction — the
    * instancing sample's asteroids orbit at radius 7 with vertices spanning
    * 0.06, an 80x lever that turned the polynomial's error into a visible
    * displacement of every rock. llvmpipe leaves this off for the same reason. */
   .lower_sincos = false,
   .lower_hadd = true,
   .lower_uadd_sat = true,
   .lower_usub_sat = true,
   .lower_iadd_sat = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_half_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_half_2x16 = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_2x32_64 = true,
   .lower_ifind_msb = true,
   .max_unroll_iterations = 32,
   .lower_to_scalar = true,
   .lower_uniforms_to_ubo = true,
   .lower_device_index_to_zero = true,
   .support_16bit_alu = false,
};

#endif /* CP_NIR_OPTIONS_H */
