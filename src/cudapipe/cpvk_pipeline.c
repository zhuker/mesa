/*
 * Shader modules, layouts and the compute pipeline for the native driver.
 *
 * The point of this file is what it does *not* contain: the compiler. NIR to
 * PTX is `cp_compile_nir_to_ptx()` from the Gallium-hosted driver, compiled
 * into this target unchanged, because it never depended on Gallium in the
 * first place -- its only "gallium" include is util/u_memory.h. So the front
 * end changes and the 3,277-line backend does not, which is the whole claim
 * this port rests on, tested here rather than asserted.
 */

#include "vk_format.h"
#include "util/blend.h"
#include "util/format/u_format.h"
#include "util/mesa-blake3.h"
#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_util.h"

#include "cp_nir_options.h"
#include "cp_nir_to_llvm.h"

#include "nir.h"
#include "nir_builder.h"
#include "compiler/spirv/nir_spirv.h"

static const struct spirv_to_nir_options cpvk_spirv_options = {
   .environment = NIR_SPIRV_VULKAN,
   /*
    * lavapipe's formats, and they have to be: this is what decides the width
    * of the deref_cast SPIR-V builds for a buffer variable, and
    * nir_lower_explicit_io asserts that a vec2_index_32bit_offset address is
    * three components. Declaring the two-component format here and returning
    * a three-component address from vulkan_resource_index produced a
    * `32x2 deref_cast` over a `32x3 vec3` and an assertion inside the
    * lowering, which reads as a bug in the lowering and is not one.
    */
   .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
   .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
   .phys_ssbo_addr_format = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_logical,
   .shared_addr_format = nir_address_format_32bit_offset,
   .global_addr_format = nir_address_format_64bit_global,
};

/*
 * The lowering lavapipe used to do on the way down.
 *
 * The backend's own unconditional "intrinsic is not implemented" warning --
 * the one added after gl_FrontFacing spent weeks as an undef -- named the gap
 * on the first run: load_deref, store_deref, vulkan_resource_index and
 * load_vulkan_descriptor arrived intact because nothing had lowered them.
 * Variables and explicit-io lowering are generic and go here; the two
 * descriptor intrinsics need this driver's own descriptor model and are the
 * next milestone, so they still warn rather than silently computing on undef.
 */

/*
 * Which ALU operations still have to be scalarised.
 *
 * emit_alu() now applies the swizzle at any width, so in principle none do.
 * In practice twelve samples render wrong without scalarisation while all
 * fourteen unit tests pass, so something narrower than "vectors" is broken.
 * CPVK_SCALARIZE selects what to keep scalarising, to find out which:
 *
 *   (unset) everything, which is the behaviour before this existed
 *   none    nothing
 *   alu     the arithmetic (fadd/fmul/ffma/...) only
 *   sel     bcsel and the comparisons only
 *   move    mov and the vecN constructors only
 */
static bool
cpvk_scalarize_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   const char *which = data;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);

   if (!strcmp(which, "none"))
      return false;

   switch (alu->op) {
   case nir_op_bcsel:
   case nir_op_flt: case nir_op_fge: case nir_op_feq: case nir_op_fneu:
   case nir_op_ilt: case nir_op_ige: case nir_op_ieq: case nir_op_ine:
      return !strcmp(which, "sel") || !strcmp(which, "alu");

   case nir_op_mov:
   case nir_op_vec2: case nir_op_vec3: case nir_op_vec4:
      return !strcmp(which, "move") || !strcmp(which, "alu");

   /* Everything that lowers to an LLVM intrinsic, where build_intrinsic()
    * overloads on the type of the first argument only. */
   case nir_op_fsqrt: case nir_op_frsq: case nir_op_frcp:
   case nir_op_fexp2: case nir_op_flog2: case nir_op_fsin: case nir_op_fcos:
   case nir_op_fpow: case nir_op_ffma: case nir_op_fabs:
   case nir_op_fmin: case nir_op_fmax: case nir_op_ffloor: case nir_op_fceil:
   case nir_op_ftrunc: case nir_op_fround_even: case nir_op_ffract:
      return !strcmp(which, "intr") || !strcmp(which, "alu");

   /* Plain LLVM binary operators. */
   case nir_op_fadd: case nir_op_fsub: case nir_op_fmul: case nir_op_fneg:
   case nir_op_fdiv:
   case nir_op_iadd: case nir_op_isub: case nir_op_imul: case nir_op_ineg:
   case nir_op_iand: case nir_op_ior: case nir_op_ixor: case nir_op_inot:
   case nir_op_ishl: case nir_op_ishr: case nir_op_ushr:
      return !strcmp(which, "basic") || !strcmp(which, "alu");

   default:
      return !strcmp(which, "rest") || !strcmp(which, "alu");
   }
}

static bool
cpvk_cf_has_loop(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cpvk_cf_has_loop(&nif->then_list) ||
             cpvk_cf_has_loop(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
cpvk_nir_has_loop(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      if (cpvk_cf_has_loop(&impl->body))
         return true;
   }
   return false;
}

static void
cpvk_optimize_nir(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_flrp, 16 | 32 | 64, true);
      NIR_PASS(progress, nir, nir_split_array_vars, nir_var_function_temp);
      NIR_PASS(progress, nir, nir_shrink_vec_array_vars, nir_var_function_temp);
      NIR_PASS(progress, nir, nir_opt_deref);
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_memcpy);
      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      nir_opt_peephole_select_options ps = {
         .limit = 8, .indirect_load_ok = true, .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &ps);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      bool loop = false;
      NIR_PASS(loop, nir, nir_opt_loop);
      progress |= loop;
      if (loop) {
         NIR_PASS(progress, nir, nir_opt_copy_prop);
         NIR_PASS(progress, nir, nir_opt_dce);
         NIR_PASS(progress, nir, nir_opt_remove_phis);
      }
      NIR_PASS(progress, nir, nir_opt_if,
               nir_opt_if_optimize_phi_true_false);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      nir_opt_peephole_select_options discard = {
         .limit = 0, .discard_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &discard);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_deref);
      const char *which = getenv("CPVK_SCALARIZE");
      if (!which)
         NIR_PASS(progress, nir, nir_lower_alu_to_scalar, NULL, NULL);
      else
         NIR_PASS(progress, nir, nir_lower_alu_to_scalar,
                  cpvk_scalarize_filter, (void *)which);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);

   NIR_PASS(_, nir, nir_opt_algebraic_late);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, nir, nir_opt_dce);
   nir_sweep(nir);
}

static void
cpvk_lower_nir(nir_shader *nir)
{
   /*
    * Registers to SSA first, before anything introduces a phi.
    *
    * spirv_to_nir emits a register for a value an if/else assigns and later
    * code reads. Left alone it becomes an alloca in the LLVM the backend
    * builds and the NVPTX backend gives the kernel a __local_depot for it --
    * off-chip memory, in a loop, which the Gallium path never had because
    * Mesa's common pipeline ran this before lavapipe handed the shader over.
    *
    * It has to be here rather than later: nir_lower_reg_intrinsics_to_ssa
    * finds nothing once the register has been through other passes, and
    * nir_trivialize_registers, which would restore the form it wants, asserts
    * that no phi exists yet and aborts seven samples if run after one does.
    */
   NIR_PASS(_, nir, nir_lower_reg_intrinsics_to_ssa);

   /* Use the same fixed-point optimization sequence that prepares lavapipe's
    * loop NIR before it reaches this backend.  Unoptimized structured loops
    * made pbribl's irradiance shader take 33 seconds instead of 26 ms.  Keep
    * straight-line shaders on their established path: running the loop
    * optimizer over them regressed negativeviewportheight by 32%.  Native
    * descriptor handles are lowered below, so lavapipe's indirect-texture
    * fixup does not apply. */
   if (cpvk_nir_has_loop(nir))
      cpvk_optimize_nir(nir);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   NIR_PASS(_, nir, nir_lower_explicit_io,
            nir_var_mem_shared, nir_address_format_32bit_offset);

   /* Push constants become load_push_constant here and a read of buffer slot
    * zero in cpvk_lower_descriptors; left as derefs they reached the backend
    * intact, which said so and computed on undef. */
   NIR_PASS(_, nir, nir_lower_explicit_io,
            nir_var_mem_push_const, nir_address_format_32bit_offset);

   /* gl_GlobalInvocationID lowers into a base plus workgroup arithmetic, and
    * the base is only nonzero for a based dispatch, which this driver does not
    * expose. Saying so explicitly folds it away; leaving it to the default
    * left load_base_global_invocation_id in the shader, which the backend has
    * no case for and would have computed on undef. */
   NIR_PASS(_, nir, nir_lower_system_values);
   struct nir_lower_compute_system_values_options csv = {
      .has_base_global_invocation_id = false,
      .has_base_workgroup_id = false,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);

   /*
    * Scalarize, which is the shape the backend has only ever been given.
    *
    * emit_alu() applies an operand's swizzle only when the destination is
    * scalar; for a vector destination it passes the source through at its own
    * width. Under Gallium that is invisible, because lavapipe scalarizes for
    * llvmpipe before cudapipe ever sees the shader. Reaching the backend with
    * `pos.xy + ubo.d.xy` intact produced an LLVM module that failed
    * verification with `fadd <3 x float>, <4 x float>` -- the two sources at
    * their own widths, the swizzles dropped.
    *
    * Fixing emit_alu to build a shuffle would be the deeper repair, and it
    * would put the backend on a path no shader has ever taken. This puts the
    * native front end on the path every shader has taken instead.
    */
   /* flrp, which the backend has no case for and lavapipe lowers before
    * cudapipe ever sees a shader. The capture's shaders use mix(). */
   NIR_PASS(_, nir, nir_lower_flrp, 16 | 32 | 64, true);

   {
      const char *which = getenv("CPVK_SCALARIZE");
      if (!which)
         NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
      else
         NIR_PASS(_, nir, nir_lower_alu_to_scalar, cpvk_scalarize_filter,
                  (void *)which);
   }

   NIR_PASS(_, nir, nir_opt_dce);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

static bool
lower_descriptors(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct cpvk_pipeline_layout *layout = data;

   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      unsigned set = nir_intrinsic_desc_set(intr);
      unsigned binding = nir_intrinsic_binding(intr);
      const struct cpvk_descriptor_set_layout *sl =
         (const struct cpvk_descriptor_set_layout *)layout->vk.set_layouts[set];
      unsigned flat = (sl && binding < sl->num_bindings)
         ? sl->bindings[binding].flat : binding;

      b->cursor = nir_before_instr(&intr->instr);
      /* (slot, byte offset in the set's buffer, 0) -- three components,
       * which is what the format wants despite its name. The array index
       * steps by one descriptor. */
      nir_def *off =
         nir_iadd_imm(b, nir_imul_imm(b, intr->src[0].ssa, CPVK_DESCRIPTOR_SIZE),
                      flat * CPVK_DESCRIPTOR_SIZE);
      nir_def *addr = nir_vec3(b, nir_imm_int(b, layout->set_slot[set]), off,
                               nir_imm_int(b, 0));
      nir_def_replace(&intr->def, addr);
      return true;
   }

   case nir_intrinsic_vulkan_resource_reindex: {
      /*
       * The array index, applied to an address this pass already built.
       *
       * NIR emits vulkan_resource_index for the binding and then a reindex
       * for each step through a descriptor array, so a shader indexing
       * `sampler samplers[3]` arrives here with the element in src[1].
       * Dropping it left every access on element 0: texturemipmapgen chose
       * between three samplers with a uniform and always got the first, which
       * is the one built with lod 0.0..0.0, so nothing it drew ever used a
       * mip level.
       *
       * The address is (slot, byte offset, 0), so the step is on the second
       * component and is the same descriptor stride the index above uses.
       */
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *addr = intr->src[0].ssa;
      nir_def *step = nir_imul_imm(b, intr->src[1].ssa, CPVK_DESCRIPTOR_SIZE);
      nir_def *out = nir_vec3(b, nir_channel(b, addr, 0),
                              nir_iadd(b, nir_channel(b, addr, 1), step),
                              nir_channel(b, addr, 2));
      nir_def_replace(&intr->def, out);
      return true;
   }

   /*
    * A buffer access still carrying the (slot, offset) pair becomes one
    * against the descriptor: the set's buffer base plus the offset. The
    * backend's emit_buffer_base dereferences a 64-bit source as a descriptor
    * and reads the buffer pointer out of its first field, which is where
    * cpvk_descriptor keeps it.
    */
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: {
      unsigned si = intr->intrinsic == nir_intrinsic_store_ssbo ? 1 : 0;
      if (nir_src_num_components(intr->src[si]) == 1)
         return false;

      b->cursor = nir_before_instr(&intr->instr);
      nir_def *slot = nir_channel(b, intr->src[si].ssa, 0);
      nir_def *offset = nir_channel(b, intr->src[si].ssa, 1);
      nir_def *desc = nir_iadd(b, nir_load_const_buf_base_addr_lvp(b, slot),
                               nir_u2u64(b, offset));
      nir_src_rewrite(&intr->src[si], desc);
      return true;
   }
   case nir_intrinsic_load_vulkan_descriptor:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, intr->src[0].ssa);
      return true;

   case nir_intrinsic_load_push_constant: {
      /* A read of the push constant block is a read of buffer slot 0 at the
       * same offset. The backend has no push-constant case and does not need
       * one; this is the form lavapipe hands it. */
      b->cursor = nir_before_instr(&intr->instr);
      nir_def *val = nir_load_ubo(b, intr->def.num_components,
                                  intr->def.bit_size,
                                  nir_imm_int(b, CPVK_UBO_PUSH_SLOT),
                                  intr->src[0].ssa,
                                  .align_mul = 4, .align_offset = 0,
                                  .range = ~0);
      nir_def_replace(&intr->def, val);
      return true;
   }
   default:
      return false;
   }
}

/*
 * Texture and sampler derefs become the handles the backend expects.
 *
 * A handle is `set_base + binding * sizeof(struct cpvk_descriptor)`, computed
 * here and never loaded: the kernel dereferences it to reach the
 * cp_texture_info and the sampler index. load_const_buf_base_addr_lvp is the
 * intrinsic the backend already implements for exactly this -- it resolves to
 * the same constant-buffer slot lookup a uniform read uses -- so the set's
 * buffer address goes in its own slot and the offset is added on top.
 */
/* The descriptor handle for a set/binding, in the form the kernels read. */
static nir_def *
cpvk_descriptor_handle(nir_builder *b,
                       const struct cpvk_pipeline_layout *layout,
                       unsigned set, unsigned binding)
{
   if (set >= MESA_VK_MAX_DESCRIPTOR_SETS)
      return NULL;

   const struct cpvk_descriptor_set_layout *sl =
      (const struct cpvk_descriptor_set_layout *)layout->vk.set_layouts[set];
   unsigned flat = (sl && binding < sl->num_bindings)
      ? sl->bindings[binding].flat : binding;

   nir_def *base =
      nir_load_const_buf_base_addr_lvp(b, nir_imm_int(b, layout->set_slot[set]));
   return nir_iadd_imm(b, base, (uint64_t)flat * CPVK_DESCRIPTOR_SIZE);
}

/*
 * Storage images, to the bindless form the backend implements.
 *
 * It has bindless_image_load, _store and _atomic and no deref-based case at
 * all, so a shader that stores to an image arrived saying image_deref_store
 * and computed on undef. The handle is the same descriptor address a texture
 * uses, which is why this sits beside it.
 */
static bool
lower_image(nir_builder *b, nir_intrinsic_instr *intr,
            const struct cpvk_pipeline_layout *layout)
{
   nir_intrinsic_op op;
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:   op = nir_intrinsic_bindless_image_load; break;
   case nir_intrinsic_image_deref_store:  op = nir_intrinsic_bindless_image_store; break;
   case nir_intrinsic_image_deref_atomic: op = nir_intrinsic_bindless_image_atomic; break;
   default:
      return false;
   }

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *handle = cpvk_descriptor_handle(b, layout, var->data.descriptor_set,
                                            var->data.binding);
   if (!handle)
      return false;

   nir_intrinsic_instr *new = nir_intrinsic_instr_create(b->shader, op);
   new->num_components = intr->num_components;
   new->src[0] = nir_src_for_ssa(handle);
   for (unsigned i = 1; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; i++)
      new->src[i] = nir_src_for_ssa(intr->src[i].ssa);
   nir_intrinsic_copy_const_indices(new, intr);

   if (nir_intrinsic_infos[op].has_dest) {
      nir_def_init(&new->instr, &new->def, intr->def.num_components,
                   intr->def.bit_size);
      nir_builder_instr_insert(b, &new->instr);
      nir_def_replace(&intr->def, &new->def);
   } else {
      nir_builder_instr_insert(b, &new->instr);
      nir_instr_remove(&intr->instr);
   }
   return true;
}

static bool
lower_tex(nir_builder *b, nir_instr *instr, void *data)
{
   const struct cpvk_pipeline_layout *layout = data;

   if (instr->type == nir_instr_type_intrinsic)
      return lower_image(b, nir_instr_as_intrinsic(instr), layout);

   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);

   b->cursor = nir_before_instr(instr);

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      nir_tex_src_type want;
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref: want = nir_tex_src_texture_handle; break;
      case nir_tex_src_sampler_deref: want = nir_tex_src_sampler_handle; break;
      default: continue;
      }

      nir_deref_instr *deref = nir_src_as_deref(tex->src[i].src);
      nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
      if (!var)
         continue;

      nir_def *handle = cpvk_descriptor_handle(b, layout,
                                               var->data.descriptor_set,
                                               var->data.binding);
      if (!handle)
         continue;

      /*
       * The array element, which the deref carries and the handle did not.
       *
       * `uniform sampler samplers[3]` indexed by anything -- a constant or a
       * uniform -- reaches here as a deref_array over the variable, and only
       * the variable's binding was being used. Every element therefore
       * resolved to the first: texturemipmapgen chose among three samplers
       * and always got samplers[0], the one built with lod 0.0..0.0, so
       * nothing it drew ever read a mip level.
       *
       * The index steps by one descriptor, the same stride
       * cpvk_descriptor_handle uses for the binding.
       */
      if (deref->deref_type == nir_deref_type_array) {
         /* The multiply stays in 32 bits and the result widens. Doing it in
          * 64 let NIR fold it to `shl i64 %x, i32 6`, whose operands differ in
          * width, and the module failed LLVM verification. */
         nir_def *step = nir_imul_imm(b, deref->arr.index.ssa,
                                      CPVK_DESCRIPTOR_SIZE);
         handle = nir_iadd(b, handle, nir_u2u64(b, step));
      }

      nir_tex_instr_remove_src(tex, i);
      nir_tex_instr_add_src(tex, want, handle);
      i = -1;   /* sources shifted; rescan */
   }

   return true;
}

/* Which pipeline is being lowered, so a dump can be matched to the one a
 * failing dispatch names. Set immediately before the call; nothing else
 * reads it. */
static const void *cpvk_lower_descriptors_dump;

static void
cpvk_lower_descriptors(nir_shader *nir,
                       const struct cpvk_pipeline_layout *layout)
{
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_descriptors,
            nir_metadata_control_flow, (void *)layout);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_tex,
            nir_metadata_control_flow, (void *)layout);
   NIR_PASS(_, nir, nir_lower_explicit_io,
            nir_var_mem_ubo | nir_var_mem_ssbo,
            nir_address_format_vec2_index_32bit_offset);
   /* Again, for the loads that lowering just built: they carry the pair and
    * have to become descriptor addresses too. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_descriptors,
            nir_metadata_control_flow, (void *)layout);
   NIR_PASS(_, nir, nir_opt_dce);

   /* Registers to SSA here as well as before the backend, because both the
    * graphics and the compute path come through this function and only the
    * graphics one reaches the later call. A register becomes an alloca, and
    * the shaders that still carried one were the two <3 x i32> allocas in
    * instancing's IR that the Gallium path did not have. */
   NIR_PASS(_, nir, nir_lower_reg_intrinsics_to_ssa);

   if (cp_debug->dump_nir) {
      fprintf(stderr, "=== %s NIR after descriptor lowering, pipeline %p ===\n",
              _mesa_shader_stage_to_string(nir->info.stage),
              cpvk_lower_descriptors_dump);
      nir_print_shader(nir, stderr);
   }
   cpvk_lower_descriptors_dump = NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorSetLayout(
   VkDevice _device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_descriptor_set_layout *layout =
      vk_descriptor_set_layout_zalloc(&dev->vk, sizeof(*layout), pCreateInfo);
   if (!layout)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Bindings are stored in binding-number order, so the flat index a shader
    * ends up using is stable and computable at compile time. */
   unsigned flat = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &pCreateInfo->pBindings[i];
      if (b->binding >= CPVK_MAX_BINDINGS) {
         vk_descriptor_set_layout_destroy(&dev->vk, &layout->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      layout->bindings[b->binding].type = b->descriptorType;
      layout->bindings[b->binding].count = MAX2(b->descriptorCount, 1u);
      if (b->binding + 1 > layout->num_bindings)
         layout->num_bindings = b->binding + 1;
   }
   for (unsigned b = 0; b < layout->num_bindings; b++) {
      layout->bindings[b].flat = flat;
      flat += layout->bindings[b].count;
   }
   layout->num_descriptors = flat;

   *pSetLayout = cpvk_descriptor_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreatePipelineLayout(VkDevice _device,
                          const VkPipelineLayoutCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_pipeline_layout *layout =
      vk_pipeline_layout_zalloc(&dev->vk, sizeof(*layout), pCreateInfo);
   if (!layout)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /*
    * Slot 0 is the push constant block, always, whether or not the pipeline
    * has one. That is where the backend already looks: emit_const_buf_base's
    * index form is what push constants arrive as, with index 0. Descriptors
    * therefore start at 1, and a layout that numbered them from 0 would have
    * every set's first binding shadowed by the push constants.
    */
   /*
    * One constant-buffer slot per descriptor set, with slot 0 the push
    * constants. lavapipe's numbering, and it has to be: a slot per binding
    * ran out. A capture's layouts wanted 17, 18, 22 and 23 slots where there
    * are 16, and the bindings past the limit reached the shader as an address
    * of zero and faulted a compute dispatch. A binding is an offset inside
    * its set's buffer, so the count follows the number of sets and not the
    * number of bindings.
    */
   for (uint32_t s = 0; s < layout->vk.set_count; s++) {
      layout->set_base[s] = 0;
      layout->set_slot[s] = CPVK_UBO_PUSH_SLOT + 1 + s;
   }
   layout->num_descriptors = layout->vk.set_count;

   if (layout->vk.set_count + 1 > CP_MAX_CONST_BUFFERS)
      fprintf(stderr, "cudapipe: pipeline layout has %u descriptor sets and "
              "there are %u slots\n", layout->vk.set_count,
              CP_MAX_CONST_BUFFERS - 1);

   *pPipelineLayout = cpvk_pipeline_layout_to_handle(layout);
   return VK_SUCCESS;
}

/*
 * vulkan_resource_index -> the flat buffer index the backend expects.
 *
 * The backend's emit_buffer_base() takes a 32-bit source as an index into
 * args[18..], which is exactly the ABI a compute dispatch wants and needs no
 * descriptor memory at all: the host resolves the set at bind time and writes
 * the addresses into the argument block. A 64-bit source would instead be a
 * pointer to a descriptor struct, which is the lavapipe shape and the one
 * that made every descriptor set its own allocation.
 */

static void
cpvk_pipeline_init_ref(struct cpvk_device *dev, struct cpvk_pipeline *pipeline,
                       const VkAllocationCallbacks *pAllocator)
{
   pipeline->dev = dev;
   pipeline->alloc = pAllocator ? *pAllocator : dev->vk.alloc;
   atomic_init(&pipeline->refcnt, 1);
}

void
cpvk_pipeline_ref(struct cpvk_pipeline *pipeline)
{
   if (pipeline)
      atomic_fetch_add_explicit(&pipeline->refcnt, 1, memory_order_relaxed);
}

void
cpvk_pipeline_unref(struct cpvk_pipeline *pipeline)
{
   if (!pipeline ||
       atomic_fetch_sub_explicit(&pipeline->refcnt, 1,
                                 memory_order_acq_rel) != 1)
      return;

   if (pipeline->bin)
      cp_shader_binary_destroy(pipeline->bin);
   vk_object_free(&pipeline->dev->vk, &pipeline->alloc, pipeline);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyPipeline(VkDevice _device, VkPipeline _pipeline,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_pipeline, pipeline, _pipeline);
   cpvk_pipeline_unref(pipeline);
}

static bool
cpvk_nir_uses_tex(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return true;
         }
      }
   }
   return false;
}

static bool
cpvk_nir_uses_tex_3d(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->sampler_dim == GLSL_SAMPLER_DIM_3D &&
                (tex->op == nir_texop_tex || tex->op == nir_texop_txl ||
                 tex->op == nir_texop_txb))
               return true;
         }
      }
   }
   return false;
}

static const char *
cpvk_sampler_ptx(struct cpvk_device *dev, bool enable_3d)
{
   struct cp_kernels *kernels = &dev->cp_dev.kernels;
   if (!enable_3d)
      return kernels->sampler_ptx;

   /* The z-filtering entry point nearly doubles the sampler module and costs
    * about two seconds of NVRTC/JIT work. Compile it only for a pipeline that
    * actually contains a 3D texture instruction, rather than charging every
    * Vulkan process at device creation. */
   simple_mtx_lock(&dev->shader_cache_lock);
   const char *ptx = kernels->sampler_3d_ptx;
   simple_mtx_unlock(&dev->shader_cache_lock);
   if (ptx)
      return ptx;

   /* NVRTC is slow; do not serialize unrelated pipeline-cache operations
    * behind it. Two racing first users may compile the same source, but only
    * one result is retained. */
   char *compiled = cp_compile_sampler_3d(dev->pdev->sm_major,
                                          dev->pdev->sm_minor);
   simple_mtx_lock(&dev->shader_cache_lock);
   if (!kernels->sampler_3d_ptx)
      kernels->sampler_3d_ptx = compiled;
   else
      free(compiled);
   ptx = kernels->sampler_3d_ptx;
   simple_mtx_unlock(&dev->shader_cache_lock);
   return ptx;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateComputePipelines(VkDevice _device, VkPipelineCache pipelineCache,
                            uint32_t count,
                            const VkComputePipelineCreateInfo *pCreateInfos,
                            const VkAllocationCallbacks *pAllocator,
                            VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VkResult first_error = VK_SUCCESS;

   for (uint32_t i = 0; i < count; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   for (uint32_t i = 0; i < count; i++) {
      struct cpvk_pipeline *pipeline =
         vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pipeline),
                          VK_OBJECT_TYPE_PIPELINE);
      if (!pipeline) {
         first_error = VK_ERROR_OUT_OF_HOST_MEMORY;
         break;
      }
      cpvk_pipeline_init_ref(dev, pipeline, pAllocator);
      pipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;

      void *mem_ctx = ralloc_context(NULL);
      nir_shader *nir = NULL;
      VkResult result = vk_pipeline_shader_stage_to_nir(
         &dev->vk, pCreateInfos[i].flags, &pCreateInfos[i].stage,
         &cpvk_spirv_options, &cp_nir_options, mem_ctx, &nir);
      if (result != VK_SUCCESS) {
         ralloc_free(mem_ctx);
         cpvk_pipeline_unref(pipeline);
         if (first_error == VK_SUCCESS)
            first_error = result;
         continue;
      }

      cpvk_lower_nir(nir);
      cpvk_lower_descriptors_dump = pipeline;
      cpvk_lower_descriptors(nir, cpvk_pipeline_layout_from_handle(
                                     pCreateInfos[i].layout));

      pipeline->local_size[0] = nir->info.workgroup_size[0];
      pipeline->local_size[1] = nir->info.workgroup_size[1];
      pipeline->local_size[2] = nir->info.workgroup_size[2];

      bool uses_tex = cpvk_nir_uses_tex(nir);
      bool uses_tex_3d = cpvk_nir_uses_tex_3d(nir);
      const char *sampler_ptx = cpvk_sampler_ptx(dev, uses_tex_3d);
      cuCtxSetCurrent(dev->cu_ctx);
      pipeline->bin = cp_compile_nir_to_ptx(
         nir, dev->pdev->sm_major, dev->pdev->sm_minor,
         uses_tex ? sampler_ptx : NULL, NULL);
      if (pipeline->bin)
         pipeline->bin->uses_tex_3d = uses_tex_3d;
      ralloc_free(mem_ctx);

      if (!pipeline->bin || !pipeline->bin->kernel) {
         cpvk_pipeline_unref(pipeline);
         if (first_error == VK_SUCCESS)
            first_error = vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
         continue;
      }

      pPipelines[i] = cpvk_pipeline_to_handle(pipeline);
   }

   return first_error;
}

/*
 * The graphics pipeline, and a draw.
 *
 * Vulkan hands the whole pipeline state at creation, which is where the
 * renderer wants it: the Gallium adapter had to reconstruct this from a
 * stream of state setters and hold a batch back to see the next draw's state
 * before it could execute the previous one. Here it is one struct, resolved
 * once, and vkCmdDraw fills a cp_draw_call and calls the same
 * cp_draw_execute the Gallium-hosted driver runs.
 */

/*
 * Blend factors and functions, in the values the kernels read: cp_blend_desc
 * says "pipe_blend_state factors/functions", so these are Gallium's numbers
 * and the mapping is written out rather than assumed to coincide with
 * Vulkan's.
 */
static uint32_t
cpvk_blend_factor(VkBlendFactor f)
{
   switch (f) {
   case VK_BLEND_FACTOR_ZERO:                     return PIPE_BLENDFACTOR_ZERO;
   case VK_BLEND_FACTOR_ONE:                      return PIPE_BLENDFACTOR_ONE;
   case VK_BLEND_FACTOR_SRC_COLOR:                return PIPE_BLENDFACTOR_SRC_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:      return PIPE_BLENDFACTOR_INV_SRC_COLOR;
   case VK_BLEND_FACTOR_DST_COLOR:                return PIPE_BLENDFACTOR_DST_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR:      return PIPE_BLENDFACTOR_INV_DST_COLOR;
   case VK_BLEND_FACTOR_SRC_ALPHA:                return PIPE_BLENDFACTOR_SRC_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:      return PIPE_BLENDFACTOR_INV_SRC_ALPHA;
   case VK_BLEND_FACTOR_DST_ALPHA:                return PIPE_BLENDFACTOR_DST_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:      return PIPE_BLENDFACTOR_INV_DST_ALPHA;
   case VK_BLEND_FACTOR_CONSTANT_COLOR:           return PIPE_BLENDFACTOR_CONST_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return PIPE_BLENDFACTOR_INV_CONST_COLOR;
   case VK_BLEND_FACTOR_CONSTANT_ALPHA:           return PIPE_BLENDFACTOR_CONST_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return PIPE_BLENDFACTOR_INV_CONST_ALPHA;
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:       return PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE;
   default:                                       return PIPE_BLENDFACTOR_ONE;
   }
}

static uint32_t
cpvk_blend_op(VkBlendOp op)
{
   switch (op) {
   case VK_BLEND_OP_SUBTRACT:         return PIPE_BLEND_SUBTRACT;
   case VK_BLEND_OP_REVERSE_SUBTRACT: return PIPE_BLEND_REVERSE_SUBTRACT;
   case VK_BLEND_OP_MIN:              return PIPE_BLEND_MIN;
   case VK_BLEND_OP_MAX:              return PIPE_BLEND_MAX;
   default:                           return PIPE_BLEND_ADD;
   }
}

static enum mesa_prim
cpvk_prim(VkPrimitiveTopology t)
{
   switch (t) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:     return MESA_PRIM_POINTS;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: return MESA_PRIM_TRIANGLE_STRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:   return MESA_PRIM_TRIANGLE_FAN;
   default:                                   return MESA_PRIM_TRIANGLES;
   }
}

/*
 * A compiled shader for this stage, compiled once.
 *
 * Keyed on the runtime's own stage hash, which covers the SPIR-V, the entry
 * point and the specialisation constants -- everything that decides what the
 * NIR will be. The descriptor lowering depends on the pipeline layout as
 * well, so the layout's set numbering goes in beside it.
 */
static struct cp_shader_binary *
cpvk_compile_stage(struct cpvk_device *dev,
                   VkPipelineCreateFlags2KHR flags,
                   const VkPipelineShaderStageCreateInfo *stage,
                   const struct cpvk_pipeline_layout *layout,
                   VkResult *result)
{
   /* Robustness is not implemented here, so the state that feeds the hash is
    * the disabled one -- zeroed rather than filled from a device that has no
    * robustness features to report. */
   struct vk_pipeline_robustness_state rstate = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs   = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images          = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };

   unsigned char hash[BLAKE3_OUT_LEN];
   vk_pipeline_hash_shader_stage(flags, stage, &rstate, hash);

   void *mem_ctx = ralloc_context(NULL);
   nir_shader *nir = NULL;
   *result = vk_pipeline_shader_stage_to_nir(&dev->vk, flags, stage,
                                             &cpvk_spirv_options,
                                             &cp_nir_options, mem_ctx, &nir);
   if (*result != VK_SUCCESS) {
      ralloc_free(mem_ctx);
      return NULL;
   }

   /* Descriptor lowering bakes the constant-buffer slot and flat descriptor
    * offset of every resource the shader uses into NIR.  Include exactly those
    * mappings in the cache key.  Hashing only set slots reused a sky shader
    * compiled with binding 1 at flat offset 0 where it was at offset 1; hashing
    * whole layouts fixed that but made pbribl compile the same shader for many
    * layouts that differed only in unused bindings. */
   if (layout) {
      bool used[MESA_VK_MAX_DESCRIPTOR_SETS][CPVK_MAX_BINDINGS] = {{ false }};
      nir_foreach_variable_with_modes(var, nir,
                                      nir_var_uniform | nir_var_mem_ubo |
                                      nir_var_mem_ssbo | nir_var_image) {
         if (var->data.descriptor_set < MESA_VK_MAX_DESCRIPTOR_SETS &&
             var->data.binding < CPVK_MAX_BINDINGS)
            used[var->data.descriptor_set][var->data.binding] = true;
      }

      struct mesa_blake3 ctx;
      _mesa_blake3_init(&ctx);
      _mesa_blake3_update(&ctx, hash, sizeof(hash));
      for (unsigned s = 0; s < layout->vk.set_count; s++) {
         const struct cpvk_descriptor_set_layout *sl =
            (const struct cpvk_descriptor_set_layout *)
            layout->vk.set_layouts[s];
         for (unsigned b = 0; b < CPVK_MAX_BINDINGS; b++) {
            if (!used[s][b])
               continue;
            unsigned flat = sl && b < sl->num_bindings
               ? sl->bindings[b].flat : b;
            _mesa_blake3_update(&ctx, &s, sizeof(s));
            _mesa_blake3_update(&ctx, &b, sizeof(b));
            _mesa_blake3_update(&ctx, &layout->set_slot[s],
                                sizeof(layout->set_slot[s]));
            _mesa_blake3_update(&ctx, &flat, sizeof(flat));
         }
      }
      _mesa_blake3_final(&ctx, hash);
   }

   simple_mtx_lock(&dev->shader_cache_lock);
   for (unsigned i = 0; i < dev->num_shaders; i++) {
      if (!memcmp(dev->shader_cache[i].hash, hash, sizeof(hash))) {
         struct cp_shader_binary *bin = dev->shader_cache[i].bin;
         simple_mtx_unlock(&dev->shader_cache_lock);
         ralloc_free(mem_ctx);
         *result = VK_SUCCESS;
         return bin;
      }
   }
   simple_mtx_unlock(&dev->shader_cache_lock);

   if (cp_debug->dump_nir) {
      fprintf(stderr, "=== %s NIR (native) ===\n",
              _mesa_shader_stage_to_string(nir->info.stage));
      nir_print_shader(nir, stderr);
   }

   cpvk_lower_nir(nir);
   cpvk_lower_descriptors(nir, layout);

   /*
    * gl_PointCoord as a varying, which is how the renderer already carries it:
    * the rasteriser writes it into the fragment stage's input slots like any
    * other varying, and the backend has no system value for it. Without this
    * particlesystem's fragment shader read undef and the driver said so.
    *
    * The renderer supplies point coordinates and window-space fragment
    * coordinates as ordinary fragment inputs. Leaving frag_coord as a system
    * value reaches the backend as an unimplemented load_frag_coord and makes
    * screen-space effects compute on undef.
    */
   {
      const nir_lower_sysvals_to_varyings_options sv = {
         .point_coord = true,
         .frag_coord = true,
      };
      NIR_PASS(_, nir, nir_lower_sysvals_to_varyings, &sv);
   }

   /*
    * Outputs become temporaries, so nothing reads one back.
    *
    * A shader that computes an output and then uses it -- `outWorldPos =
    * locPos + objPos; gl_Position = proj * view * vec4(outWorldPos, 1)` --
    * leaves a load_output in the NIR, and the backend has no case for it and
    * says so. It computed on undef, so every one of pbribl's sphere vertices
    * came out at the origin with w = 0, every triangle was degenerate, and
    * the spheres never appeared. lavapipe runs this pass; this driver did
    * not.
    */
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /*
       * A vertex shader's inputs are numbered by their attribute location,
       * not compacted: the fetch kernel writes attribute N into slot N
       * because cpvk_pipeline::velem is indexed by
       * VkVertexInputAttributeDescription::location. Letting
       * nir_assign_io_var_locations renumber them means the shader reads a
       * slot the fetch never wrote whenever the locations are not exactly
       * 0..n-1 in declaration order -- which is how pushconstants' spheres
       * came out shaded by their normals instead of their colour.
       */
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_in)
         var->data.driver_location = var->data.location >= VERT_ATTRIB_GENERIC0
            ? var->data.location - VERT_ATTRIB_GENERIC0 : var->data.location;
   } else {
      nir_assign_io_var_locations(nir, nir_var_shader_in);
   }
   nir_assign_io_var_locations(nir, nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            cp_type_size_vec4, nir_lower_io_lower_64bit_to_32);

   /* Hoist vertex-input loads before the arithmetic which consumes them.
    *
    * The instancing sample's rock shader runs over 4.42 million vertices.  In
    * this frontend's NIR the loads were interleaved with sin/cos and matrix
    * chains; the CUDA JIT then kept only 104 registers live and serialized
    * global loads (94.4% long-scoreboard stalls, 21% DRAM throughput).  Mesa's
    * existing pass exposes those independent loads just as the Gallium path's
    * lowering does: 6.14 instead of 8.63 ms/frame on that sample. */
   if (!getenv("CPVK_NO_HOIST_INPUTS") &&
       nir->info.stage == MESA_SHADER_VERTEX)
      NIR_PASS(_, nir, nir_opt_move_to_top,
               nir_move_to_top_input_loads_simple);

   /*
    * Registers back to SSA, immediately before the backend.
    *
    * spirv_to_nir emits a register for a value an if/else assigns and later
    * code reads -- a phi written the old way -- and this driver's fragment
    * shaders reached the backend still carrying one where the Gallium path's
    * did not, because Mesa's common pipeline had already run this before
    * lavapipe handed the shader over. A register becomes an alloca in the
    * LLVM the backend builds.
    */
   if (!getenv("CPVK_NO_REG_SSA")) {
      NIR_PASS(_, nir, nir_lower_reg_intrinsics_to_ssa);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   bool uses_tex_3d = cpvk_nir_uses_tex_3d(nir);
   const char *sampler_ptx = cpvk_sampler_ptx(dev, uses_tex_3d);
   cuCtxSetCurrent(dev->cu_ctx);
   bool frag = stage->stage == VK_SHADER_STAGE_FRAGMENT_BIT;
   struct cp_shader_binary *bin =
      cp_compile_nir_to_ptx(nir, dev->pdev->sm_major, dev->pdev->sm_minor,
                            sampler_ptx,
                            frag ? dev->cp_dev.kernels.fs_helper_ptx : NULL);
   if (bin)
      bin->uses_tex_3d = uses_tex_3d;
   ralloc_free(mem_ctx);

   /* A small shader which calls the dynamic sampler helper cannot amortise a
    * capped build's spills. The warm-up tuner sees one fixed camera position
    * and chooses the cap there, then texturemipmapgen's orbit makes it 2x
    * slower. Keep larger dynamic shaders eligible: multisampling crosses this
    * boundary and benefits from the extra resident block. */
   if (bin && frag && !getenv("CPVK_KEEP_SMALL_DYNAMIC_REGCAP") &&
       bin->tune_cap && bin->tex_descs_dynamic && !bin->num_tex_descs &&
       bin->ptx_size < 6 * 1024)
      bin->tune_cap = 0;

   if (!bin || !bin->kernel) {
      cp_shader_binary_destroy(bin);
      *result = VK_ERROR_INITIALIZATION_FAILED;
      return NULL;
   }

   simple_mtx_lock(&dev->shader_cache_lock);
   /* Another thread may have compiled the same stage while this one was
    * outside the lock. Preserve the cache's pointer-identity invariant and
    * discard the duplicate build. */
   for (unsigned i = 0; i < dev->num_shaders; i++) {
      if (!memcmp(dev->shader_cache[i].hash, hash, sizeof(hash))) {
         struct cp_shader_binary *cached = dev->shader_cache[i].bin;
         simple_mtx_unlock(&dev->shader_cache_lock);
         cp_shader_binary_destroy(bin);
         *result = VK_SUCCESS;
         return cached;
      }
   }
   if (dev->num_shaders >= dev->max_shaders) {
      unsigned want = dev->max_shaders ? dev->max_shaders * 2 : 64;
      void *p = realloc(dev->shader_cache, want * sizeof(*dev->shader_cache));
      if (!p) {
         simple_mtx_unlock(&dev->shader_cache_lock);
         cp_shader_binary_destroy(bin);
         *result = VK_ERROR_OUT_OF_HOST_MEMORY;
         return NULL;
      }
      dev->shader_cache = p;
      dev->max_shaders = want;
   }
   memcpy(dev->shader_cache[dev->num_shaders].hash, hash, sizeof(hash));
   dev->shader_cache[dev->num_shaders].bin = bin;
   dev->num_shaders++;
   simple_mtx_unlock(&dev->shader_cache_lock);

   *result = VK_SUCCESS;
   return bin;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateGraphicsPipelines(VkDevice _device, VkPipelineCache cache,
                             uint32_t count,
                             const VkGraphicsPipelineCreateInfo *pCreateInfos,
                             const VkAllocationCallbacks *pAllocator,
                             VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VkResult first_error = VK_SUCCESS;

   for (uint32_t i = 0; i < count; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   for (uint32_t i = 0; i < count; i++) {
      const VkGraphicsPipelineCreateInfo *info = &pCreateInfos[i];
      if (info->pColorBlendState &&
          info->pColorBlendState->attachmentCount > 1) {
         if (first_error == VK_SUCCESS)
            first_error = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
         continue;
      }
      struct cpvk_pipeline *pipeline =
         vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pipeline),
                          VK_OBJECT_TYPE_PIPELINE);
      if (!pipeline) {
         first_error = VK_ERROR_OUT_OF_HOST_MEMORY;
         break;
      }
      cpvk_pipeline_init_ref(dev, pipeline, pAllocator);
      pipeline->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;

      /* Both stages through the cache, so identical SPIR-V compiles once and
       * two pipelines built from it share one binary -- which is what lets
       * their draws batch. */
      VkResult result = VK_SUCCESS;
      const struct cpvk_pipeline_layout *layout =
         cpvk_pipeline_layout_from_handle(info->layout);

      for (uint32_t s = 0; s < info->stageCount; s++) {
         const VkPipelineShaderStageCreateInfo *stage = &info->pStages[s];
         struct cp_shader_binary *bin =
            cpvk_compile_stage(dev, info->flags, stage, layout, &result);
         if (!bin)
            break;
         if (stage->stage == VK_SHADER_STAGE_FRAGMENT_BIT)
            pipeline->fs = bin;
         else
            pipeline->vs = bin;
      }
      if (result != VK_SUCCESS || !pipeline->vs || !pipeline->fs) {
         cpvk_pipeline_unref(pipeline);
         if (first_error == VK_SUCCESS) {
            VkResult error = result != VK_SUCCESS
               ? result : VK_ERROR_INITIALIZATION_FAILED;
            first_error = vk_error(dev, error);
         }
         continue;
      }

      /* The fixed-function state the renderer reads. */
      const VkPipelineRasterizationStateCreateInfo *rs = info->pRasterizationState;
      pipeline->raster = (struct cp_raster_state) {
         .cull_face = rs ? (unsigned)rs->cullMode : 0,
         .front_ccw = rs && rs->frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE,
         .scissor = true,
      };
      const VkPipelineDepthStencilStateCreateInfo *ds = info->pDepthStencilState;
      pipeline->depth = (struct cp_depth_state) {
         .depth_enabled = ds && ds->depthTestEnable,
         .depth_writemask = ds && ds->depthWriteEnable,
         /* VkCompareOp and cp_compare_func agree; the values are Gallium's
          * and Vulkan's alike. */
         .depth_func = ds ? (unsigned)ds->depthCompareOp : CP_FUNC_ALWAYS,
      };
      const VkPipelineColorBlendStateCreateInfo *cb = info->pColorBlendState;
      if (cb && cb->attachmentCount) {
         const VkPipelineColorBlendAttachmentState *at = &cb->pAttachments[0];
         pipeline->blend = (struct cp_blend_desc) {
            .enable = at->blendEnable,
            .colormask = at->colorWriteMask ? at->colorWriteMask : 0xF,
            /*
             * The equation itself, which was missing: only `enable` and the
             * write mask were being resolved, so every blended draw combined
             * its fragments with whatever factors happened to be zero. That
             * is one line of state and four samples' worth of wrong pixels.
             */
            .rgb_src_factor = cpvk_blend_factor(at->srcColorBlendFactor),
            .rgb_dst_factor = cpvk_blend_factor(at->dstColorBlendFactor),
            .rgb_func = cpvk_blend_op(at->colorBlendOp),
            .alpha_src_factor = cpvk_blend_factor(at->srcAlphaBlendFactor),
            .alpha_dst_factor = cpvk_blend_factor(at->dstAlphaBlendFactor),
            .alpha_func = cpvk_blend_op(at->alphaBlendOp),
         };
      } else {
         pipeline->blend.colormask = 0xF;
      }

      /* The sample count, which was pinned at one: multisampling rendered
       * every pixel from one sample and differed from the Gallium driver on
       * 98% of them. */
      const VkPipelineMultisampleStateCreateInfo *ms = info->pMultisampleState;
      pipeline->samples = ms ? MAX2((unsigned)ms->rasterizationSamples, 1u) : 1;

      const VkPipelineInputAssemblyStateCreateInfo *ia = info->pInputAssemblyState;
      pipeline->topology = ia ? cpvk_prim(ia->topology) : MESA_PRIM_TRIANGLES;
      const VkPipelineVertexInputStateCreateInfo *vi = info->pVertexInputState;
      if (vi) {
         for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount && a < 16; a++) {
            const VkVertexInputAttributeDescription *ad =
               &vi->pVertexAttributeDescriptions[a];
            unsigned stride = 0, divisor = 0;
            for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++)
               if (vi->pVertexBindingDescriptions[b].binding == ad->binding) {
                  stride = vi->pVertexBindingDescriptions[b].stride;
                  /* Per-instance attributes step once per instance. */
                  divisor = vi->pVertexBindingDescriptions[b].inputRate ==
                            VK_VERTEX_INPUT_RATE_INSTANCE ? 1 : 0;
               }
            /* The format's consequences, worked out by the same function
             * the Gallium front end calls: every component arrives in its own
             * 32-bit slot however narrow it is in memory, so anything that is
             * not already 32 bits per component has to be widened. The
             * hardcoded four-float attribute this replaces read an R8G8B8A8
             * as one number up to 2^32. */
            enum pipe_format pfmt = vk_format_to_pipe_format(ad->format);
            uint32_t nr_chan, chan_bytes, swizzle;
            enum cp_vf_conv conv =
               cp_vertex_format(pfmt, &nr_chan, &chan_bytes, &swizzle);
            pipeline->velem[ad->location] = (struct cp_vertex_elem) {
               .vertex_buffer_index = ad->binding,
               .src_offset = ad->offset,
               .src_stride = stride,
               .instance_divisor = divisor,
               .attr_size = util_format_get_blocksize(pfmt),
               .nr_chan = nr_chan,
               .chan_bytes = chan_bytes,
               .swizzle = swizzle,
               .conv = conv,
               .fill_w = cp_vertex_fill_w(pfmt, conv),
            };
            if (ad->location + 1 > pipeline->num_velem)
               pipeline->num_velem = ad->location + 1;
            pipeline->vertex_stride = stride;
         }
      }

      pPipelines[i] = cpvk_pipeline_to_handle(pipeline);
   }

   return first_error;
}
