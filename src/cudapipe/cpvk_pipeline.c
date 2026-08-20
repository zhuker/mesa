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
static void
cpvk_lower_nir(nir_shader *nir)
{
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

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);

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
cpvk_pipeline_destroy(struct cpvk_device *dev, struct cpvk_pipeline *pipeline,
                      const VkAllocationCallbacks *pAllocator)
{
   if (pipeline->bin)
      cp_shader_binary_destroy(pipeline->bin);
   vk_object_free(&dev->vk, pAllocator, pipeline);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyPipeline(VkDevice _device, VkPipeline _pipeline,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_pipeline, pipeline, _pipeline);

   if (pipeline)
      cpvk_pipeline_destroy(dev, pipeline, pAllocator);
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
      pipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;

      void *mem_ctx = ralloc_context(NULL);
      nir_shader *nir = NULL;
      VkResult result = vk_pipeline_shader_stage_to_nir(
         &dev->vk, pCreateInfos[i].flags, &pCreateInfos[i].stage,
         &cpvk_spirv_options, &cp_nir_options, mem_ctx, &nir);
      if (result != VK_SUCCESS) {
         ralloc_free(mem_ctx);
         cpvk_pipeline_destroy(dev, pipeline, pAllocator);
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

      cuCtxSetCurrent(dev->cu_ctx);
      pipeline->bin = cp_compile_nir_to_ptx(nir, dev->pdev->sm_major,
                                            dev->pdev->sm_minor, NULL, NULL);
      ralloc_free(mem_ctx);

      if (!pipeline->bin || !pipeline->bin->kernel) {
         cpvk_pipeline_destroy(dev, pipeline, pAllocator);
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

   /* The layout decides which slot each set lands in, so a shader compiled
    * against one layout cannot be reused for another. */
   if (layout) {
      unsigned n = MIN2(layout->vk.set_count, BLAKE3_OUT_LEN / 2);
      for (unsigned i = 0; i < n; i++)
         hash[i] ^= (unsigned char)(layout->set_slot[i] + 1);
   }

   simple_mtx_lock(&dev->shader_cache_lock);
   for (unsigned i = 0; i < dev->num_shaders; i++) {
      if (!memcmp(dev->shader_cache[i].hash, hash, sizeof(hash))) {
         struct cp_shader_binary *bin = dev->shader_cache[i].bin;
         simple_mtx_unlock(&dev->shader_cache_lock);
         *result = VK_SUCCESS;
         return bin;
      }
   }
   simple_mtx_unlock(&dev->shader_cache_lock);

   void *mem_ctx = ralloc_context(NULL);
   nir_shader *nir = NULL;
   *result = vk_pipeline_shader_stage_to_nir(&dev->vk, flags, stage,
                                             &cpvk_spirv_options,
                                             &cp_nir_options, mem_ctx, &nir);
   if (*result != VK_SUCCESS) {
      ralloc_free(mem_ctx);
      return NULL;
   }

   if (cp_debug->dump_nir) {
      fprintf(stderr, "=== %s NIR (native) ===\n",
              _mesa_shader_stage_to_string(nir->info.stage));
      nir_print_shader(nir, stderr);
   }

   cpvk_lower_nir(nir);
   cpvk_lower_descriptors(nir, layout);

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
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   cuCtxSetCurrent(dev->cu_ctx);
   bool frag = stage->stage == VK_SHADER_STAGE_FRAGMENT_BIT;
   struct cp_shader_binary *bin =
      cp_compile_nir_to_ptx(nir, dev->pdev->sm_major, dev->pdev->sm_minor,
                            dev->cp_dev.kernels.sampler_ptx,
                            frag ? dev->cp_dev.kernels.fs_helper_ptx : NULL);
   ralloc_free(mem_ctx);

   if (!bin || !bin->kernel) {
      *result = VK_ERROR_INITIALIZATION_FAILED;
      return NULL;
   }

   simple_mtx_lock(&dev->shader_cache_lock);
   if (dev->num_shaders >= dev->max_shaders) {
      unsigned want = dev->max_shaders ? dev->max_shaders * 2 : 64;
      void *p = realloc(dev->shader_cache, want * sizeof(*dev->shader_cache));
      if (p) {
         dev->shader_cache = p;
         dev->max_shaders = want;
      }
   }
   if (dev->num_shaders < dev->max_shaders) {
      memcpy(dev->shader_cache[dev->num_shaders].hash, hash, sizeof(hash));
      dev->shader_cache[dev->num_shaders].bin = bin;
      dev->num_shaders++;
   }
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
      struct cpvk_pipeline *pipeline =
         vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pipeline),
                          VK_OBJECT_TYPE_PIPELINE);
      if (!pipeline) {
         first_error = VK_ERROR_OUT_OF_HOST_MEMORY;
         break;
      }
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
         cpvk_pipeline_destroy(dev, pipeline, pAllocator);
         if (first_error == VK_SUCCESS)
            first_error = vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
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
