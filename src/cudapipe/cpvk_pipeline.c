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

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorSetLayout(
   VkDevice _device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct vk_descriptor_set_layout *layout =
      vk_descriptor_set_layout_zalloc(&dev->vk, sizeof(*layout),
                                      pCreateInfo);
   if (!layout)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pSetLayout = vk_descriptor_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreatePipelineLayout(VkDevice _device,
                          const VkPipelineLayoutCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct vk_pipeline_layout *layout =
      vk_pipeline_layout_zalloc(&dev->vk, sizeof(*layout), pCreateInfo);
   if (!layout)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pPipelineLayout = vk_pipeline_layout_to_handle(layout);
   return VK_SUCCESS;
}

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
