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

   /*
    * Slot 0 is the push constant block, always, whether or not the pipeline
    * has one. That is where the backend already looks: emit_const_buf_base's
    * index form is what push constants arrive as, with index 0. Descriptors
    * therefore start at 1, and a layout that numbered them from 0 would have
    * every set's first binding shadowed by the push constants.
    */
   unsigned flat = CPVK_UBO_PUSH_SLOT + 1;
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

      /* Both stages through the shared compiler. */
      VkResult result = VK_SUCCESS;
      for (uint32_t s = 0; s < info->stageCount; s++) {
         const VkPipelineShaderStageCreateInfo *stage = &info->pStages[s];
         void *mem_ctx = ralloc_context(NULL);
         nir_shader *nir = NULL;
         result = vk_pipeline_shader_stage_to_nir(&dev->vk, info->flags, stage,
                                                  &cpvk_spirv_options,
                                                  &cp_nir_options, mem_ctx,
                                                  &nir);
         if (result != VK_SUCCESS) {
            ralloc_free(mem_ctx);
            break;
         }
         if (cp_debug->dump_nir) {
            fprintf(stderr, "=== %s NIR (native) ===\n",
                    _mesa_shader_stage_to_string(nir->info.stage));
            nir_print_shader(nir, stderr);
         }

         cpvk_lower_nir(nir);
         cpvk_lower_descriptors(nir, cpvk_pipeline_layout_from_handle(info->layout));

         /*
          * Slots first, then the lowering that uses them. nir_lower_io reads
          * var->data.driver_location, which is zero on every variable until
          * something assigns it -- so the first native draw put the fragment
          * colour and gl_Position in the same slot and rasterized a triangle
          * whose position was its colour. The backend's capture_io_locations
          * reads the same field to match the fragment shader's inputs to the
          * vertex shader's outputs, so this is what makes the stages agree.
          */
         nir_assign_io_var_locations(nir, nir_var_shader_in);
         nir_assign_io_var_locations(nir, nir_var_shader_out);

         /* The same lowering the Gallium front end does, with the same slot
          * size function: the two stages agree on where an output lands only
          * because one function decides it for both. */
         NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
                  cp_type_size_vec4, nir_lower_io_lower_64bit_to_32);

         /* Gather after the lowering, not before it: the backend reads the
          * input and output masks off shader_info, and cpvk_lower_nir's
          * gather ran while the I/O was still derefs -- which is how a
          * fragment shader with one input arrived claiming none. */
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

         cuCtxSetCurrent(dev->cu_ctx);
         bool frag = stage->stage == VK_SHADER_STAGE_FRAGMENT_BIT;
         struct cp_shader_binary *bin =
            cp_compile_nir_to_ptx(nir, dev->pdev->sm_major, dev->pdev->sm_minor,
                                  dev->cp_dev.kernels.sampler_ptx,
                                  frag ? dev->cp_dev.kernels.fs_helper_ptx : NULL);
         ralloc_free(mem_ctx);
         if (!bin || !bin->kernel) {
            result = VK_ERROR_INITIALIZATION_FAILED;
            break;
         }
         if (!frag)
            pipeline->vs = bin;
         else
            pipeline->fs = bin;
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
         };
      } else {
         pipeline->blend.colormask = 0xF;
      }

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
