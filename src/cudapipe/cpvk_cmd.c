/*
 * Descriptor sets, command buffers and dispatch.
 *
 * The compute argument ABI is the Gallium-hosted driver's, unchanged, because
 * the generated kernel is the same kernel: args[0] points at the grid size,
 * and args[18 + i] is the base address of buffer i, which is what
 * emit_const_buf_base() dereferences for a 32-bit-index load_ubo or
 * load_ssbo. So binding a descriptor set is filling in those slots, and a
 * descriptor set is an array of device addresses -- no descriptor memory, no
 * VkDeviceMemory per set, and none of the fault storm that arrangement caused.
 */

#include "vk_format.h"
#include "util/format/u_format.h"
#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"

#define CPVK_ARG_SLOTS      34
#define CPVK_ARG_UBO_BASE   18
#define CPVK_MAX_ARG_BUFS   16

/* ---------------------------------------------------------------- pools */

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorPool(VkDevice _device,
                          const VkDescriptorPoolCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_descriptor_pool *pool =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                       VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pDescriptorPool = cpvk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool _pool,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_descriptor_pool, pool, _pool);

   if (pool)
      vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_AllocateDescriptorSets(VkDevice _device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);

      struct cpvk_descriptor_set *set =
         vk_object_zalloc(&dev->vk, NULL, sizeof(*set),
                          VK_OBJECT_TYPE_DESCRIPTOR_SET);
      if (!set) {
         for (uint32_t j = 0; j < i; j++) {
            VK_FROM_HANDLE(cpvk_descriptor_set, s, pDescriptorSets[j]);
            vk_object_free(&dev->vk, NULL, s);
            pDescriptorSets[j] = VK_NULL_HANDLE;
         }
         return vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);
      }
      set->layout = layout;

      /* The set as one buffer: a texture handle is an offset into it. */
      if (layout && layout->num_descriptors) {
         cuCtxSetCurrent(dev->cu_ctx);
         size_t size = (size_t)layout->num_descriptors *
                       sizeof(struct cpvk_descriptor);
         if (cuMemAllocManaged(&set->buf, size, CU_MEM_ATTACH_GLOBAL) ==
             CUDA_SUCCESS) {
            set->host = (struct cpvk_descriptor *)(uintptr_t)set->buf;
            memset(set->host, 0, size);
         }
      }

      pDescriptorSets[i] = cpvk_descriptor_set_to_handle(set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_FreeDescriptorSets(VkDevice _device, VkDescriptorPool pool,
                        uint32_t count, const VkDescriptorSet *pSets)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set, set, pSets[i]);
      if (set)
         vk_object_free(&dev->vk, NULL, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_ResetDescriptorPool(VkDevice _device, VkDescriptorPool pool,
                         VkDescriptorPoolResetFlags flags)
{
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_UpdateDescriptorSets(VkDevice _device, uint32_t writeCount,
                          const VkWriteDescriptorSet *pWrites,
                          uint32_t copyCount,
                          const VkCopyDescriptorSet *pCopies)
{
   for (uint32_t w = 0; w < writeCount; w++) {
      const VkWriteDescriptorSet *write = &pWrites[w];
      VK_FROM_HANDLE(cpvk_descriptor_set, set, write->dstSet);
      if (!set)
         continue;

      for (uint32_t e = 0; e < write->descriptorCount; e++) {
         unsigned binding = write->dstBinding;
         if (binding >= CPVK_MAX_BINDINGS)
            continue;
         unsigned flat = set->layout->bindings[binding].flat +
                         write->dstArrayElement + e;
         if (flat >= CPVK_MAX_BINDINGS)
            continue;

         switch (write->descriptorType) {
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: {
            VK_FROM_HANDLE(cpvk_buffer, buffer, write->pBufferInfo[e].buffer);
            set->addrs[flat] = buffer && buffer->mem
               ? buffer->mem->dev_ptr + buffer->offset +
                 write->pBufferInfo[e].offset
               : 0;
            if (set->host)
               set->host[flat].buffer_base = set->addrs[flat];
            break;
         }

         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_SAMPLER: {
            if (!set->host)
               break;
            const VkDescriptorImageInfo *ii = &write->pImageInfo[e];
            if (ii->imageView) {
               VK_FROM_HANDLE(cpvk_image_view, view, ii->imageView);
               set->host[flat].texture_info = view ? view->tex_info : 0;
            }
            if (ii->sampler) {
               VK_FROM_HANDLE(cpvk_sampler, samp, ii->sampler);
               set->host[flat].sampler_index = samp ? samp->index : 0;
            }
            break;
         }

         default:
            break;
         }
      }
   }
}

/* -------------------------------------------------------- command buffers */

static void
cpvk_cmd_buffer_reset(struct vk_command_buffer *vk_cmd,
                      VkCommandBufferResetFlags flags)
{
   struct cpvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct cpvk_cmd_buffer, vk);

   vk_command_buffer_reset(&cmd->vk);
   cmd->num_dispatches = 0;
   cmd->num_ops = 0;
   cmd->pipeline = NULL;
   memset(cmd->addrs, 0, sizeof(cmd->addrs));
   memset(cmd->vb_base, 0, sizeof(cmd->vb_base));
   cmd->num_vb = 0;
   cmd->has_fb = false;
   cmd->index_ptr = NULL;
   cmd->index_size = 0;
   cmd->push_size = 0;
}

static void
cpvk_cmd_buffer_destroy(struct vk_command_buffer *vk_cmd)
{
   struct cpvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct cpvk_cmd_buffer, vk);

   vk_command_buffer_finish(&cmd->vk);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

static VkResult
cpvk_cmd_buffer_create(struct vk_command_pool *pool,
                       VkCommandBufferLevel level,
                       struct vk_command_buffer **out)
{
   struct cpvk_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &cpvk_cmd_buffer_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }

   *out = &cmd->vk;
   return VK_SUCCESS;
}

const struct vk_command_buffer_ops cpvk_cmd_buffer_ops = {
   .create = cpvk_cmd_buffer_create,
   .reset = cpvk_cmd_buffer_reset,
   .destroy = cpvk_cmd_buffer_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                        const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   cmd->num_dispatches = 0;
   cmd->num_ops = 0;
   cmd->pipeline = NULL;
   memset(cmd->addrs, 0, sizeof(cmd->addrs));
   memset(cmd->vb_base, 0, sizeof(cmd->vb_base));
   cmd->num_vb = 0;
   cmd->has_fb = false;
   cmd->index_ptr = NULL;
   cmd->index_size = 0;
   cmd->push_size = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   return vk_command_buffer_end(&cmd->vk);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindPipeline(VkCommandBuffer commandBuffer,
                     VkPipelineBindPoint pipelineBindPoint,
                     VkPipeline _pipeline)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_pipeline, pipeline, _pipeline);

   cmd->pipeline = pipeline;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindDescriptorSets2(VkCommandBuffer commandBuffer,
                            const VkBindDescriptorSetsInfo *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_pipeline_layout, layout, pInfo->layout);

   for (uint32_t i = 0; i < pInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set, set, pInfo->pDescriptorSets[i]);
      if (!set)
         continue;
      unsigned base = layout->set_base[pInfo->firstSet + i];
      for (unsigned d = 0; d < set->layout->num_descriptors; d++) {
         if (base + d < CPVK_MAX_ARG_BUFS)
            cmd->addrs[base + d] = set->addrs[d];
      }

      /* And the set itself, which is what a texture handle is an offset
       * into. Buffers keep their own slot; this one is for images. */
      unsigned slot = layout->set_slot[pInfo->firstSet + i];
      if (slot < CPVK_MAX_ARG_BUFS)
         cmd->addrs[slot] = set->buf;
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                     uint32_t baseGroupY, uint32_t baseGroupZ,
                     uint32_t groupCountX, uint32_t groupCountY,
                     uint32_t groupCountZ)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   if (cmd->num_dispatches >= CPVK_MAX_DISPATCHES || !cmd->pipeline)
      return;

   struct cpvk_dispatch *d = &cmd->dispatches[cmd->num_dispatches++];
   d->pipeline = cmd->pipeline;
   d->grid[0] = groupCountX;
   d->grid[1] = groupCountY;
   d->grid[2] = groupCountZ;
   memcpy(d->addrs, cmd->addrs, sizeof(d->addrs));
}

/* ------------------------------------------------------------- execution */

VkResult
cpvk_execute_cmd_buffer(struct cpvk_device *dev, struct cpvk_cmd_buffer *cmd)
{
   for (unsigned i = 0; i < cmd->num_dispatches; i++) {
      const struct cpvk_dispatch *d = &cmd->dispatches[i];
      const struct cp_shader_binary *bin = d->pipeline->bin;
      if (!bin || !bin->kernel)
         continue;

      /* The argument block: an array of pointers, with the grid behind it,
       * exactly as cp_launch_grid() builds it. */
      const size_t args_bytes = CPVK_ARG_SLOTS * sizeof(void *);
      const size_t total = args_bytes + 3 * sizeof(uint32_t);

      CUdeviceptr block;
      if (cuMemAllocManaged(&block, total, CU_MEM_ATTACH_GLOBAL) !=
          CUDA_SUCCESS)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      void **slots = (void **)(uintptr_t)block;
      memset(slots, 0, total);
      slots[0] = (void *)(uintptr_t)(block + args_bytes);
      for (unsigned s = 0; s < CPVK_MAX_ARG_BUFS; s++)
         slots[CPVK_ARG_UBO_BASE + s] = (void *)(uintptr_t)d->addrs[s];

      uint32_t *grid = (uint32_t *)((char *)slots + args_bytes);
      grid[0] = d->grid[0];
      grid[1] = d->grid[1];
      grid[2] = d->grid[2];

      void *kernel_args[] = { &block };
      unsigned bx = MAX2(d->pipeline->local_size[0], (uint16_t)1);
      unsigned by = MAX2(d->pipeline->local_size[1], (uint16_t)1);
      unsigned bz = MAX2(d->pipeline->local_size[2], (uint16_t)1);
      CUresult err = cuLaunchKernel(bin->kernel, d->grid[0], d->grid[1],
                                    d->grid[2], bx, by, bz, 0, dev->stream,
                                    kernel_args, NULL);
      if (err != CUDA_SUCCESS) {
         cuMemFree(block);
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      }

      /* The block is read by the kernel, so it cannot be released until the
       * launch has run. One sync per dispatch is the wrong answer and is
       * replaced by the upload arena when there is a frame to amortise it
       * over; at one dispatch it is honest and obvious. */
      cuStreamSynchronize(dev->stream);
      cuMemFree(block);
   }
   return VK_SUCCESS;
}

/* ------------------------------------------------------- rendering + draw */

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                       const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   struct cp_fb_desc fb = {
      .width = pRenderingInfo->renderArea.offset.x +
               pRenderingInfo->renderArea.extent.width,
      .height = pRenderingInfo->renderArea.offset.y +
                pRenderingInfo->renderArea.extent.height,
      .nr_cbufs = pRenderingInfo->colorAttachmentCount,
      .color_encoding = -1,
      .has_zs = pRenderingInfo->pDepthAttachment != NULL &&
                pRenderingInfo->pDepthAttachment->imageView != VK_NULL_HANDLE,
   };

   const VkRenderingAttachmentInfo *cat =
      pRenderingInfo->colorAttachmentCount ?
      &pRenderingInfo->pColorAttachments[0] : NULL;
   struct cpvk_image *cimg = NULL;

   if (cat && cat->imageView) {
      VK_FROM_HANDLE(cpvk_image_view, view, cat->imageView);
      if (view && view->image && view->image->mem) {
         cimg = view->image;
         fb.color = (void *)(uintptr_t)(cimg->mem->dev_ptr + cimg->offset);
         fb.color_encoding = cimg->color;
      }
   }

   cmd->fb = fb;
   cmd->has_fb = true;

   /*
    * Bind the framebuffer here, as its own op, and not at the first draw.
    *
    * cp_context_set_framebuffer is what allocates the renderer's depth
    * buffer, and cp_clear_depthbuf clears whatever is allocated now. Doing
    * the bind at draw time meant the depth clear ran against the previous
    * framebuffer's buffer and the new one arrived uncleared, so every
    * fragment failed the depth test: renderheadless rendered its clear colour
    * and all three triangles vanished, with the rasterizer reporting them
    * shaded.
    */
   if (cmd->num_ops < CPVK_MAX_DISPATCHES) {
      struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
      *op = (struct cpvk_op) { .kind = CPVK_OP_BEGIN_RENDER };
      op->fb = fb;
   }

   /* LOAD_OP_CLEAR, recorded in order with the draws that follow it. */
   if (cat && cimg && cat->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR &&
       cmd->num_ops < CPVK_MAX_DISPATCHES) {
      enum pipe_format pfmt = vk_format_to_pipe_format(cimg->vk.format);
      struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
      *op = (struct cpvk_op) { .kind = CPVK_OP_CLEAR };
      op->clear = (struct cpvk_clear) {
         .data = fb.color,
         .width = fb.width,
         .height = fb.height,
         .stride = cimg->row_stride[0],
         .pixel_size = util_format_get_blocksize(pfmt),
      };
      /* Packed here rather than in the kernel: cp_clear_rect takes the value
       * already packed, and packing it twice produces a plausible wrong
       * colour rather than an obvious one. */
      util_format_pack_rgba(pfmt, op->clear.value,
                            cat->clearValue.color.float32, 1);
   }

   const VkRenderingAttachmentInfo *dat = pRenderingInfo->pDepthAttachment;
   if (dat && dat->imageView && dat->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR &&
       cmd->num_ops < CPVK_MAX_DISPATCHES) {
      struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
      *op = (struct cpvk_op) { .kind = CPVK_OP_CLEAR };
      op->clear = (struct cpvk_clear) {
         .depth = true,
         .depth_value = dat->clearValue.depthStencil.depth,
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                           uint32_t bindingCount, const VkBuffer *pBuffers,
                           const VkDeviceSize *pOffsets,
                           const VkDeviceSize *pSizes,
                           const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   for (uint32_t i = 0; i < bindingCount; i++) {
      unsigned b = firstBinding + i;
      if (b >= 16)
         continue;
      VK_FROM_HANDLE(cpvk_buffer, buf, pBuffers[i]);
      cmd->vb_base[b] = (buf && buf->mem)
         ? buf->mem->dev_ptr + buf->offset + pOffsets[i] : 0;
      cmd->num_vb = MAX2(cmd->num_vb, b + 1);
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPushConstants2(VkCommandBuffer commandBuffer,
                       const VkPushConstantsInfo *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   if (pInfo->offset + pInfo->size > CPVK_MAX_PUSH_BYTES)
      return;

   memcpy(cmd->push + pInfo->offset, pInfo->pValues, pInfo->size);
   if (pInfo->offset + pInfo->size > cmd->push_size)
      cmd->push_size = pInfo->offset + pInfo->size;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                      VkShaderStageFlags stageFlags, uint32_t offset,
                      uint32_t size, const void *pValues)
{
   VkPushConstantsInfo info = {
      .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
      .layout = layout, .stageFlags = stageFlags,
      .offset = offset, .size = size, .pValues = pValues,
   };
   cpvk_CmdPushConstants2(commandBuffer, &info);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindIndexBuffer2(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                         VkDeviceSize offset, VkDeviceSize size,
                         VkIndexType indexType)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, _buffer);

   cmd->index_ptr = (buf && buf->mem)
      ? (const void *)(uintptr_t)(buf->mem->dev_ptr + buf->offset + offset)
      : NULL;
   cmd->index_size = indexType == VK_INDEX_TYPE_UINT16 ? 2 :
                     indexType == VK_INDEX_TYPE_UINT8 ? 1 : 4;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer,
                        VkDeviceSize offset, VkIndexType indexType)
{
   cpvk_CmdBindIndexBuffer2(commandBuffer, buffer, offset, VK_WHOLE_SIZE,
                            indexType);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetViewportWithCount(VkCommandBuffer commandBuffer, uint32_t count,
                             const VkViewport *pViewports)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   if (!count)
      return;
   /* Vulkan's clip space already has y running downward, which is why the
    * scale is not flipped again here; see cp_cull_mode's note on winding. */
   cmd->viewport.scale[0] = pViewports[0].width * 0.5f;
   cmd->viewport.scale[1] = pViewports[0].height * 0.5f;
   cmd->viewport.scale[2] = pViewports[0].maxDepth - pViewports[0].minDepth;
   cmd->viewport.translate[0] = pViewports[0].x + pViewports[0].width * 0.5f;
   cmd->viewport.translate[1] = pViewports[0].y + pViewports[0].height * 0.5f;
   cmd->viewport.translate[2] = pViewports[0].minDepth;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport,
                    uint32_t count, const VkViewport *pViewports)
{
   if (firstViewport == 0)
      cpvk_CmdSetViewportWithCount(commandBuffer, count, pViewports);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetScissorWithCount(VkCommandBuffer commandBuffer, uint32_t count,
                            const VkRect2D *pScissors)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   if (!count)
      return;
   cmd->scissor = (struct cp_rect) {
      .minx = pScissors[0].offset.x,
      .miny = pScissors[0].offset.y,
      .maxx = pScissors[0].offset.x + pScissors[0].extent.width,
      .maxy = pScissors[0].offset.y + pScissors[0].extent.height,
   };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor,
                   uint32_t count, const VkRect2D *pScissors)
{
   if (firstScissor == 0)
      cpvk_CmdSetScissorWithCount(commandBuffer, count, pScissors);
}

/* One recorded draw, shared by vkCmdDraw and vkCmdDrawIndexed. */
static void
cpvk_record_draw(struct cpvk_cmd_buffer *cmd, unsigned count, unsigned first,
                 unsigned instance_count, unsigned first_instance,
                 int vertex_offset, bool indexed)
{
   if (cmd->num_ops >= CPVK_MAX_DISPATCHES || !cmd->pipeline)
      return;

   struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
   *op = (struct cpvk_op) { .kind = CPVK_OP_DRAW };
   struct cpvk_draw *d = &op->draw;

   d->pipeline = cmd->pipeline;
   d->fb = cmd->fb;
   d->viewport = cmd->viewport;
   d->scissor = cmd->scissor;
   d->range = (struct cp_draw_range) {
      .start = first,
      .count = count,
      .index_bias = vertex_offset,
   };
   d->call = (struct cp_draw_call) {
      .mode = cmd->pipeline->topology,
      .instance_count = MAX2(instance_count, 1u),
      .start_instance = first_instance,
      .index_size = indexed ? cmd->index_size : 0,
      .index_ptr = indexed ? cmd->index_ptr : NULL,
   };
   memcpy(d->vb_base, cmd->vb_base, sizeof(d->vb_base));
   d->num_vb = cmd->num_vb;
   memcpy(d->addrs, cmd->addrs, sizeof(d->addrs));
   memcpy(d->push, cmd->push, sizeof(d->push));
   d->push_size = cmd->push_size;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
             uint32_t instanceCount, uint32_t firstVertex,
             uint32_t firstInstance)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   cpvk_record_draw(cmd, vertexCount, firstVertex, instanceCount,
                    firstInstance, 0, false);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                    uint32_t instanceCount, uint32_t firstIndex,
                    int32_t vertexOffset, uint32_t firstInstance)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   cpvk_record_draw(cmd, indexCount, firstIndex, instanceCount, firstInstance,
                    vertexOffset, true);
}

/* A recorded clear, at submit. */
void
cpvk_execute_clear(struct cpvk_device *dev, const struct cpvk_clear *c)
{
   struct cp_context *cp = &dev->renderer;

   if (c->depth) {
      cp_clear_depthbuf(cp, c->depth_value);
      return;
   }

   cp_clear_rect(cp, c->data, c->offset, c->width, c->height, c->stride,
                 c->pixel_size, c->value, false);
}

/* Run one recorded draw through the renderer. */
void
cpvk_execute_draw(struct cpvk_device *dev, const struct cpvk_draw *d)
{
   struct cp_context *cp = &dev->renderer;
   struct cpvk_pipeline *p = d->pipeline;

   cp->viewport = d->viewport;
   cp->scissor = d->scissor;
   cp->rasterizer = p->raster;
   cp->depth_stencil = p->depth;
   cp->blend_desc = p->blend;
   cp->blend_enabled = p->blend.enable;
   cp->vs_shader = p->vs;
   cp->fs_shader = p->fs;
   memcpy(cp->velem, p->velem, sizeof(cp->velem));
   cp->num_vertex_elements = p->num_velem;
   cp->vertex_stride = p->vertex_stride;
   memcpy(cp->vb_base, d->vb_base, sizeof(cp->vb_base));
   cp->num_vertex_buffers = d->num_vb;

   /*
    * The descriptor sets the command buffer resolved become the shaders'
    * constant buffers. A cudapipe descriptor set is an array of device
    * addresses, which is exactly what a UBO binding is here.
    *
    * It goes in `buffer`, which is the binding's device address, and not in
    * `managed_copy`, which is where the Gallium adapter stages a binding that
    * came from a user pointer. The single-draw path builds its uniform table
    * out of `buffer`; filling the other field left the table full of nulls
    * and the vertex shader read address zero.
    */
   /* Slot 0 is the push constant block, staged into the upload arena so the
    * kernels read it from device memory like any other binding. */
   CUdeviceptr push_dev = 0;
   if (d->push_size) {
      void *host = NULL;
      push_dev = cp_upload_begin(cp, d->push_size, &host);
      if (push_dev) {
         memcpy(host, d->push, d->push_size);
         /* Reserving the block does not send it. Without this the shaders read
          * whatever the arena held. */
         cp_upload_end(cp, push_dev, host, d->push_size);
      }
   }

   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++) {
      cp->vs_ubos[i].buffer = (void *)(uintptr_t)d->addrs[i];
      cp->vs_ubos[i].managed_copy = 0;
      cp->vs_ubos[i].user_copy = false;
      cp->fs_ubos[i].buffer = (void *)(uintptr_t)d->addrs[i];
      cp->fs_ubos[i].managed_copy = 0;
      cp->fs_ubos[i].user_copy = false;
   }
   cp->vs_ubos[CPVK_UBO_PUSH_SLOT].buffer = (void *)(uintptr_t)push_dev;
   cp->fs_ubos[CPVK_UBO_PUSH_SLOT].buffer = (void *)(uintptr_t)push_dev;
   cp->num_vs_ubos = cp->num_fs_ubos = CP_MAX_CONST_BUFFERS;

   cp_context_publish_state(cp);

   /* The single-draw call: every batch table NULL, which is the convention
    * cp_draw_vbo uses when a draw is not merged. Batching on the native side
    * comes later, and until it does this is the path that has to be right. */
   cp_draw_execute(cp, &d->call, 0, &d->range, 1, 1, NULL, NULL, NULL, NULL,
                   NULL, NULL);
}

/* ----------------------------------------------------------- transfers */

static struct cpvk_copy *
cpvk_record_copy(struct cpvk_cmd_buffer *cmd)
{
   if (cmd->num_ops >= CPVK_MAX_DISPATCHES)
      return NULL;
   struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
   *op = (struct cpvk_op) { .kind = CPVK_OP_COPY };
   return &op->copy;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyBuffer2(VkCommandBuffer commandBuffer,
                    const VkCopyBufferInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, src, pInfo->srcBuffer);
   VK_FROM_HANDLE(cpvk_buffer, dst, pInfo->dstBuffer);

   if (!src || !dst || !src->mem || !dst->mem)
      return;

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferCopy2 *r = &pInfo->pRegions[i];
      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      *c = (struct cpvk_copy) {
         .src = src->mem->dev_ptr + src->offset + r->srcOffset,
         .dst = dst->mem->dev_ptr + dst->offset + r->dstOffset,
         .width_bytes = r->size,
         .rows = 1,
      };
   }
}

/* The byte offset of one level of an image, and its row pitch. */
static bool
cpvk_image_plane(const struct cpvk_image *img, unsigned level,
                 CUdeviceptr *base, size_t *pitch, unsigned *bpp)
{
   if (!img || !img->mem || level >= CPVK_MAX_MIP_LEVELS)
      return false;
   *base = img->mem->dev_ptr + img->offset + img->level_offset[level];
   *pitch = img->row_stride[level];
   *bpp = util_format_get_blocksize(vk_format_to_pipe_format(img->vk.format));
   return true;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyImage2(VkCommandBuffer commandBuffer,
                   const VkCopyImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, src, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_image, dst, pInfo->dstImage);

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageCopy2 *r = &pInfo->pRegions[i];
      CUdeviceptr sb, db;
      size_t sp, dp;
      unsigned sbpp, dbpp;
      if (!cpvk_image_plane(src, r->srcSubresource.mipLevel, &sb, &sp, &sbpp) ||
          !cpvk_image_plane(dst, r->dstSubresource.mipLevel, &db, &dp, &dbpp))
         return;

      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      *c = (struct cpvk_copy) {
         .src = sb + (size_t)r->srcOffset.y * sp + (size_t)r->srcOffset.x * sbpp,
         .dst = db + (size_t)r->dstOffset.y * dp + (size_t)r->dstOffset.x * dbpp,
         .src_pitch = sp,
         .dst_pitch = dp,
         .width_bytes = (size_t)r->extent.width * sbpp,
         .rows = r->extent.height,
      };
   }
}

/*
 * Barriers are recorded and ignored, deliberately.
 *
 * Everything this driver submits runs on one CUDA stream, and stream order is
 * program order, so the dependency a barrier expresses already holds. This is
 * not a stub to fill in later: it is the same reasoning that lets the Gallium
 * driver launch a clear on cp->stream instead of synchronising.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                         const VkDependencyInfo *pDependencyInfo)
{
}

void
cpvk_execute_begin_render(struct cpvk_device *dev, const struct cp_fb_desc *fb)
{
   cp_context_set_framebuffer(&dev->renderer, fb, 1);
}

void
cpvk_execute_copy(struct cpvk_device *dev, const struct cpvk_copy *c)
{
   struct cp_context *cp = &dev->renderer;

   cuCtxSetCurrent(dev->cu_ctx);

   if (c->rows <= 1 && !c->src_pitch && !c->dst_pitch) {
      cuMemcpyDtoDAsync(c->dst, c->src, c->width_bytes, cp->stream);
      return;
   }

   CUDA_MEMCPY2D m = {
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = c->src,
      .srcPitch = c->src_pitch ? c->src_pitch : c->width_bytes,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = c->dst,
      .dstPitch = c->dst_pitch ? c->dst_pitch : c->width_bytes,
      .WidthInBytes = c->width_bytes,
      .Height = c->rows,
   };
   cuMemcpy2DAsync(&m, cp->stream);
}
