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
   .ubo_addr_format = nir_address_format_32bit_index_offset,
   .ssbo_addr_format = nir_address_format_32bit_index_offset,
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
      unsigned base = layout->set_base[set];
      const struct cpvk_descriptor_set_layout *sl =
         (const struct cpvk_descriptor_set_layout *)layout->vk.set_layouts[set];
      if (sl && binding < sl->num_bindings)
         base += sl->bindings[binding].flat;

      b->cursor = nir_before_instr(&intr->instr);
      nir_def *index = nir_iadd_imm(b, intr->src[0].ssa, base);
      /* (index, offset) is the 32bit_index_offset address format. */
      nir_def *addr = nir_vec2(b, index, nir_imm_int(b, 0));
      nir_def_replace(&intr->def, addr);
      return true;
   }
   case nir_intrinsic_load_vulkan_descriptor:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, intr->src[0].ssa);
      return true;
   default:
      return false;
   }
}

static void
cpvk_lower_descriptors(nir_shader *nir,
                       const struct cpvk_pipeline_layout *layout)
{
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_descriptors,
            nir_metadata_control_flow, (void *)layout);
   NIR_PASS(_, nir, nir_lower_explicit_io,
            nir_var_mem_ubo | nir_var_mem_ssbo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_opt_dce);
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

   unsigned flat = 0;
   for (uint32_t s = 0; s < layout->vk.set_count; s++) {
      layout->set_base[s] = flat;
      struct cpvk_descriptor_set_layout *set =
         (struct cpvk_descriptor_set_layout *)layout->vk.set_layouts[s];
      if (set)
         flat += set->num_descriptors;
   }
   layout->num_descriptors = flat;

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
