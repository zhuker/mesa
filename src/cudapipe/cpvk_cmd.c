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
#include <time.h>
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

            /*
             * A descriptor nothing ever writes still gets read, by a shader
             * that declares a binding the application does not use on this
             * path. Its base pointed at zero and took the device down; it
             * points at the null page instead, so the read lands in zeroes
             * and the frame is merely wrong. The capture's first compute
             * dispatch died on exactly this: descriptor two of a bound set,
             * never written, read eight bytes at 0x80.
             */
            for (unsigned d = 0; d < layout->num_descriptors; d++)
               set->host[d].base = dev->null_data;
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

/*
 * One descriptor written, shared by vkUpdateDescriptorSets and the template
 * path. They differ only in where the VkDescriptorImageInfo and
 * VkDescriptorBufferInfo come from -- an array, or a stride into a blob of
 * application memory -- and a second copy of this switch would be a second
 * thing to be right about a descriptor the first one is wrong about.
 */
static void
cpvk_write_descriptor(struct cpvk_descriptor_set *set, unsigned flat,
                      VkDescriptorType type,
                      const VkDescriptorImageInfo *ii,
                      const VkDescriptorBufferInfo *bi)
{
   if (getenv("CPVK_DEBUG_RT"))
      fprintf(stderr, "descw flat=%u type=%u ii=%p bi=%p\n", flat, type,
              (const void *)ii, (const void *)bi);

   if (flat >= CPVK_MAX_BINDINGS)
      return;

   switch (type) {
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: {
      if (!bi)
         break;
      VK_FROM_HANDLE(cpvk_buffer, buffer, bi->buffer);
      set->addrs[flat] = buffer && buffer->mem
         ? buffer->mem->dev_ptr + buffer->offset + bi->offset : 0;
      if (set->host)
         set->host[flat].base = set->addrs[flat];
      break;
   }

   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_SAMPLER: {
      if (!set->host || !ii)
         break;
      if (ii->imageView) {
         VK_FROM_HANDLE(cpvk_image_view, view, ii->imageView);
         set->host[flat].texture_info = view ? view->tex_info : 0;
      }
      if (ii->sampler) {
         VK_FROM_HANDLE(cpvk_sampler, samp, ii->sampler);
         set->host[flat].sampler_index_or_img_stride = samp ? samp->index : 0;
      }

      if (getenv("CPVK_DEBUG_RT")) {
         VK_FROM_HANDLE(cpvk_image_view, dv, ii->imageView);
         fprintf(stderr, "desc flat=%u type=%u tex=%p "
                 "samp=%u img=%ux%u\n", flat, type,
                 (void *)(uintptr_t)set->host[flat].texture_info,
                 set->host[flat].sampler_index_or_img_stride,
                 (dv && dv->image) ? dv->image->vk.extent.width : 0,
                 (dv && dv->image) ? dv->image->vk.extent.height : 0);
      }

      /*
       * A storage image is addressed directly by the shader rather than
       * sampled, so it needs the four fields the backend's
       * bindless_image_load and _store read out of the descriptor.
       */
      if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && ii->imageView) {
         VK_FROM_HANDLE(cpvk_image_view, view, ii->imageView);
         struct cpvk_image *img = view ? view->image : NULL;
         if (img && img->mem) {
            unsigned level = view->vk.base_mip_level;
            set->host[flat].base = img->mem->dev_ptr + img->offset;
            /*
             * The extent at this level, which the load and store paths clamp
             * against. Without it width read zero, every coordinate was out
             * of bounds, and robustness returned zero for the whole image.
             */
            set->host[flat].width = MAX2(img->vk.extent.width >> level, 1u);
            set->host[flat].height = MAX2(img->vk.extent.height >> level, 1u);
            set->host[flat].depth = MAX2(img->vk.extent.depth >> level, 1u);
            set->host[flat].row_stride = img->row_stride[level];
            set->host[flat].sampler_index_or_img_stride = img->level_size[level];
            set->host[flat].base_offset = img->level_offset[level];
         }
      }
      break;
   }

   default:
      break;
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_UpdateDescriptorSets(VkDevice _device, uint32_t writeCount,
                          const VkWriteDescriptorSet *pWrites,
                          uint32_t copyCount,
                          const VkCopyDescriptorSet *pCopies)
{
   if (getenv("CPVK_DEBUG_RT"))
      fprintf(stderr, "updsets n=%u\n", writeCount);

   for (uint32_t w = 0; w < writeCount; w++) {
      const VkWriteDescriptorSet *write = &pWrites[w];
      VK_FROM_HANDLE(cpvk_descriptor_set, set, write->dstSet);
      if (getenv("CPVK_DEBUG_RT"))
         fprintf(stderr, "  w=%u bind=%u type=%u set=%p n=%u\n", w,
                 write->dstBinding, write->descriptorType, (void *)set,
                 write->descriptorCount);
      if (!set || write->dstBinding >= CPVK_MAX_BINDINGS)
         continue;

      for (uint32_t e = 0; e < write->descriptorCount; e++) {
         unsigned flat = set->layout->bindings[write->dstBinding].flat +
                         write->dstArrayElement + e;
         cpvk_write_descriptor(set, flat, write->descriptorType,
                               write->pImageInfo ? &write->pImageInfo[e] : NULL,
                               write->pBufferInfo ? &write->pBufferInfo[e] : NULL);
      }
   }
}

/* ------------------------------------------------- update templates */

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDescriptorUpdateTemplate(
   VkDevice _device, const VkDescriptorUpdateTemplateCreateInfo *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkDescriptorUpdateTemplate *pTemplate)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_descriptor_update_template *tmpl =
      vk_object_zalloc(&dev->vk, pAllocator,
                       sizeof(*tmpl) + pCreateInfo->descriptorUpdateEntryCount *
                                       sizeof(*tmpl->entries),
                       VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE);
   if (!tmpl)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   tmpl->entry_count = pCreateInfo->descriptorUpdateEntryCount;
   memcpy(tmpl->entries, pCreateInfo->pDescriptorUpdateEntries,
          tmpl->entry_count * sizeof(*tmpl->entries));

   *pTemplate = cpvk_descriptor_update_template_to_handle(tmpl);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDescriptorUpdateTemplate(VkDevice _device,
                                     VkDescriptorUpdateTemplate _tmpl,
                                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_descriptor_update_template, tmpl, _tmpl);

   if (tmpl)
      vk_object_free(&dev->vk, pAllocator, tmpl);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_UpdateDescriptorSetWithTemplate(VkDevice _device, VkDescriptorSet _set,
                                     VkDescriptorUpdateTemplate _tmpl,
                                     const void *pData)
{
   VK_FROM_HANDLE(cpvk_descriptor_set, set, _set);
   VK_FROM_HANDLE(cpvk_descriptor_update_template, tmpl, _tmpl);

   if (!set || !tmpl)
      return;

   for (uint32_t i = 0; i < tmpl->entry_count; i++) {
      const VkDescriptorUpdateTemplateEntry *e = &tmpl->entries[i];
      if (e->dstBinding >= CPVK_MAX_BINDINGS)
         continue;

      for (uint32_t j = 0; j < e->descriptorCount; j++) {
         const char *src = (const char *)pData + e->offset + j * e->stride;
         unsigned flat = set->layout->bindings[e->dstBinding].flat +
                         e->dstArrayElement + j;
         cpvk_write_descriptor(set, flat, e->descriptorType,
                               (const VkDescriptorImageInfo *)src,
                               (const VkDescriptorBufferInfo *)src);
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
   for (unsigned i = 0; i < cmd->num_desc_retired; i++)
      cuMemFree(cmd->desc_retired[i]);
   cmd->num_desc_retired = 0;
   cmd->desc_arena_used = 0;
}

static void
cpvk_cmd_buffer_destroy(struct vk_command_buffer *vk_cmd)
{
   struct cpvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct cpvk_cmd_buffer, vk);

   vk_command_buffer_finish(&cmd->vk);
   for (unsigned i = 0; i < cmd->num_desc_retired; i++)
      cuMemFree(cmd->desc_retired[i]);
   free(cmd->desc_retired);
   if (cmd->desc_arena)
      cuMemFree(cmd->desc_arena);
   free(cmd->ops);
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
   for (unsigned i = 0; i < cmd->num_desc_retired; i++)
      cuMemFree(cmd->desc_retired[i]);
   cmd->num_desc_retired = 0;
   cmd->desc_arena_used = 0;
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

/* Keep an outgrown arena alive until the command buffer is reset. */
static void
cpvk_arena_retire(struct cpvk_cmd_buffer *cmd, CUdeviceptr arena)
{
   if (cmd->num_desc_retired >= cmd->max_desc_retired) {
      unsigned want = cmd->max_desc_retired ? cmd->max_desc_retired * 2 : 8;
      CUdeviceptr *p = realloc(cmd->desc_retired, want * sizeof(*p));
      if (!p)
         return;   /* leaked until the device goes away; better than a crash */
      cmd->desc_retired = p;
      cmd->max_desc_retired = want;
   }
   cmd->desc_retired[cmd->num_desc_retired++] = arena;
}

/*
 * Copy a descriptor set into memory this command buffer owns and return its
 * device address. Managed, because the renderer reads descriptors on the host
 * when it specialises a shader on its sampler state.
 */
static CUdeviceptr
cpvk_snapshot_set(struct cpvk_cmd_buffer *cmd, struct cpvk_descriptor_set *set)
{
   if (!set->host || !set->layout->num_descriptors) {
      /* Returning zero here is an address of zero in a shader, which is the
       * fault compute-sanitizer reports as a read at 0x80 -- descriptor two
       * of a set that is not there. Say so at the point it happens. */
      fprintf(stderr, "cudapipe: descriptor set has no buffer (%u descriptors, "
              "host=%p); shaders reading it will fault\n",
              set->layout ? set->layout->num_descriptors : 0,
              (void *)set->host);
      return 0;
   }

   size_t bytes = (size_t)set->layout->num_descriptors *
                  sizeof(struct cpvk_descriptor);

   if (cmd->desc_arena_used + bytes > cmd->desc_arena_size) {
      size_t want = MAX2(cmd->desc_arena_size * 2,
                         cmd->desc_arena_used + bytes);
      want = MAX2(want, (size_t)64 * 1024);
      CUdeviceptr fresh;
      if (cuMemAllocManaged(&fresh, want, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS)
         return set->buf;

      /* The old arena is still referenced by the draws already recorded, so
       * it is kept until the command buffer is reset rather than freed here.
       * One leak per growth, bounded by the buffer's lifetime. */
      if (cmd->desc_arena)
         cpvk_arena_retire(cmd, cmd->desc_arena);
      cmd->desc_arena = fresh;
      cmd->desc_arena_host = (struct cpvk_descriptor *)(uintptr_t)fresh;
      cmd->desc_arena_size = want;
      cmd->desc_arena_used = 0;
   }

   struct cpvk_descriptor *dst =
      (struct cpvk_descriptor *)((char *)cmd->desc_arena_host +
                                 cmd->desc_arena_used);
   CUdeviceptr addr = cmd->desc_arena + cmd->desc_arena_used;
   memcpy(dst, set->host, bytes);
   cmd->desc_arena_used += bytes;
   return addr;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBindDescriptorSets2(VkCommandBuffer commandBuffer,
                            const VkBindDescriptorSetsInfo *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_pipeline_layout, layout, pInfo->layout);
   unsigned dyn = 0;

   for (uint32_t i = 0; i < pInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(cpvk_descriptor_set, set, pInfo->pDescriptorSets[i]);
      if (!set)
         continue;

      /*
       * The set's buffer goes in the set's slot -- a copy of it, so that a
       * later rebind cannot reach back into the draws already recorded.
       * Every binding in it, buffer or texture or storage image alike, is an
       * offset from here.
       */
      unsigned slot = layout->set_slot[pInfo->firstSet + i];
      CUdeviceptr snap_addr = cpvk_snapshot_set(cmd, set);
      if (slot < CPVK_MAX_ARG_BUFS) {
         cmd->addrs[slot] = snap_addr;

         /* What the set contains, so two draws binding equal descriptors can
          * merge even though each bind has its own copy. FNV-1a over the
          * snapshot; a batch key is a merge decision, not a security
          * boundary. */
         uint64_t h = 0xcbf29ce484222325ull;
         const uint8_t *bytes = (const uint8_t *)(uintptr_t)snap_addr;
         size_t n = (size_t)set->layout->num_descriptors *
                    sizeof(struct cpvk_descriptor);
         for (size_t b = 0; bytes && b < n; b++) {
            h ^= bytes[b];
            h *= 0x100000001b3ull;
         }
         cmd->desc_hash[slot] = h;
      }

      /*
       * Dynamic offsets: a dynamic uniform buffer binding names one buffer
       * and the draw picks the element out of it with an offset given at
       * bind time. They are consumed in binding order over the dynamic
       * descriptors of each set, which is what the spec says and what the
       * sample relies on -- and they are written into the snapshot rather
       * than the set, because the next bind of the same set carries a
       * different offset and must not rewrite this draw's.
       */
      struct cpvk_descriptor *snap =
         (struct cpvk_descriptor *)(uintptr_t)snap_addr;

      for (unsigned b = 0; b < set->layout->num_bindings &&
                           dyn < pInfo->dynamicOffsetCount; b++) {
         VkDescriptorType ty = set->layout->bindings[b].type;
         if (ty != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
             ty != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
            continue;
         for (unsigned e = 0; e < set->layout->bindings[b].count &&
                              dyn < pInfo->dynamicOffsetCount; e++) {
            unsigned flat = set->layout->bindings[b].flat + e;
            uint32_t off = pInfo->pDynamicOffsets[dyn++];
            if (snap && flat < CPVK_MAX_BINDINGS && set->addrs[flat])
               snap[flat].base = set->addrs[flat] + off;
         }
      }
   }
}

static struct cpvk_op *
cpvk_op_alloc(struct cpvk_cmd_buffer *cmd, enum cpvk_op_kind kind);

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                     uint32_t baseGroupY, uint32_t baseGroupZ,
                     uint32_t groupCountX, uint32_t groupCountY,
                     uint32_t groupCountZ)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   if (!cmd->pipeline)
      return;

   /* In the op list, so that it runs where it was recorded. */
   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_DISPATCH);
   if (!op)
      return;
   struct cpvk_dispatch *d = &op->dispatch;
   d->pipeline = cmd->pipeline;
   d->grid[0] = groupCountX;
   d->grid[1] = groupCountY;
   d->grid[2] = groupCountZ;
   memcpy(d->addrs, cmd->addrs, sizeof(d->addrs));
   memcpy(d->push, cmd->push, sizeof(d->push));
   d->push_size = cmd->push_size;
}

/* ------------------------------------------------------------- execution */

VkResult
cpvk_execute_dispatch(struct cpvk_device *dev,
                       const struct cpvk_dispatch *d)
{
   const struct cp_shader_binary *bin = d->pipeline->bin;
   if (!bin || !bin->kernel)
      return VK_SUCCESS;

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
      slots[CPVK_ARG_UBO_BASE + s] = (void *)(uintptr_t)
         (d->addrs[s] ? d->addrs[s] : dev->null_desc);

   /*
    * The push constant block, in slot 0, which the graphics path staged
    * and this one did not. A compute shader reading push constants
    * therefore read address zero: compute-sanitizer reported an 8-byte
    * read at 0x80, which is 128 bytes into a block that was not there.
    */
   CUdeviceptr push_block = 0;
   if (d->push_size) {
      if (cuMemAllocManaged(&push_block, d->push_size,
                            CU_MEM_ATTACH_GLOBAL) == CUDA_SUCCESS) {
         memcpy((void *)(uintptr_t)push_block, d->push, d->push_size);
         slots[CPVK_ARG_UBO_BASE + CPVK_UBO_PUSH_SLOT] =
            (void *)(uintptr_t)push_block;
      }
   }

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
      /* Name it. A submit that returns DEVICE_LOST and says nothing else
       * is indistinguishable from every other way a replay can stop. */
      const char *name = NULL;
      cuGetErrorName(err, &name);
      fprintf(stderr, "cudapipe: compute dispatch %ux%ux%u failed: %s (%d)\n",
              d->grid[0], d->grid[1], d->grid[2], name ? name : "?", err);
      fprintf(stderr, "cudapipe:   pipeline %p, push=%u bytes, buffer slots:",
              (void *)d->pipeline, d->push_size);
      for (unsigned s = 0; s < CPVK_MAX_ARG_BUFS; s++)
         if (d->addrs[s])
            fprintf(stderr, " [%u]=%p", s, (void *)(uintptr_t)d->addrs[s]);
      fprintf(stderr, "\n");
      cuMemFree(block);
      return vk_error(dev, VK_ERROR_DEVICE_LOST);
   }

   /* The blocks are read by the kernel, so they cannot be released until
    * the launch has run. One sync per dispatch is the wrong answer and is
    * replaced by the upload arena when there is a frame to amortise it
    * over; at one dispatch it is honest and obvious. */
   cuStreamSynchronize(dev->stream);
   cuMemFree(block);
   if (push_block)
      cuMemFree(push_block);

   return VK_SUCCESS;
}

/*
 * Secondary command buffers, replayed into the primary.
 *
 * A secondary records the same ops a primary does, so executing one is
 * appending its list. The alternative -- keeping a reference and walking into
 * it at submit -- would have to answer what happens when the secondary is
 * reset before the primary is submitted, and the answer would be a
 * use-after-free.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t count,
                        const VkCommandBuffer *pCommandBuffers)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(cpvk_cmd_buffer, sec, pCommandBuffers[i]);
      if (!sec || !sec->num_ops)
         continue;

      if (cmd->num_ops + sec->num_ops > cmd->max_ops) {
         unsigned want = MAX2(cmd->max_ops ? cmd->max_ops * 2 : 64,
                              cmd->num_ops + sec->num_ops);
         struct cpvk_op *ops = realloc(cmd->ops, want * sizeof(*ops));
         if (!ops)
            return;
         cmd->ops = ops;
         cmd->max_ops = want;
      }

      memcpy(cmd->ops + cmd->num_ops, sec->ops,
             sec->num_ops * sizeof(*sec->ops));
      cmd->num_ops += sec->num_ops;
   }
}

/* Room for one more operation, or NULL if it cannot be had. */
static struct cpvk_op *
cpvk_op_alloc(struct cpvk_cmd_buffer *cmd, enum cpvk_op_kind kind)
{
   if (cmd->num_ops >= cmd->max_ops) {
      unsigned want = cmd->max_ops ? cmd->max_ops * 2 : 64;
      struct cpvk_op *ops = realloc(cmd->ops, want * sizeof(*ops));
      if (!ops)
         return NULL;
      cmd->ops = ops;
      cmd->max_ops = want;
   }

   struct cpvk_op *op = &cmd->ops[cmd->num_ops++];
   memset(op, 0, sizeof(*op));
   op->kind = kind;
   return op;
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
         /*
          * The view's subresource, not the image's base. A cube map is
          * rendered one face at a time through a view whose baseArrayLayer
          * selects the face, and a mip chain likewise through baseMipLevel.
          * Ignoring both meant every face of pbribl's environment cube was
          * rendered over face zero, so its spheres reflected nothing.
          */
         unsigned rlevel = MIN2(view->vk.base_mip_level, CPVK_MAX_MIP_LEVELS - 1);
         fb.color = (void *)(uintptr_t)(cimg->mem->dev_ptr + cimg->offset +
                                        cimg->level_offset[rlevel] +
                                        (uint64_t)view->vk.base_array_layer *
                                        cimg->level_size[rlevel]);
         /*
          * The view's format decides the encoding, not the image's. They are
          * usually the same and were assumed to be; when they are not, the
          * difference is a channel order, and the sample suite's triangle came
          * out with red and blue exchanged and every pixel otherwise exact.
          */
         const struct cpvk_format_info *vf = cpvk_format_info(view->vk.format);
         fb.color_encoding = vf ? vf->color : cimg->color;
         /* How far apart the samples are, which the renderer needs in order
          * to write them and the resolve needs in order to find them. */
         fb.color_sample_stride = (unsigned)cimg->sample_stride;

         if (getenv("CPVK_DEBUG_RT"))
            fprintf(stderr, "rt: %ux%u layer=%u level=%u layers=%u img=%ux%u "
                    "fmt=%u base=%p\n", fb.width, fb.height,
                    view->vk.base_array_layer, view->vk.base_mip_level,
                    pRenderingInfo->layerCount, cimg->vk.extent.width,
                    cimg->vk.extent.height, view->vk.format, fb.color);
      }
   }

   /*
    * Layered rendering is not implemented: every draw goes to one layer, the
    * one the colour view names. A pass that asks for more silently rendered
    * its layers on top of each other, which is the failure mode this driver
    * spends most of its warnings avoiding.
    */
   if (pRenderingInfo->layerCount > 1) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr, "cudapipe: vkCmdBeginRendering with layerCount=%u; "
                 "layered rendering is not implemented and every layer goes "
                 "to the first\n", pRenderingInfo->layerCount);
      }
   }

   cmd->fb = fb;
   cmd->has_fb = true;

   cmd->resolve_src = NULL;
   cmd->resolve_dst = NULL;
   if (cat && cimg && cat->resolveImageView != VK_NULL_HANDLE &&
       cat->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(cpvk_image_view, rview, cat->resolveImageView);
      if (rview && rview->image && rview->image->mem) {
         cmd->resolve_src = cimg;
         cmd->resolve_dst = rview->image;
         cmd->resolve_area = pRenderingInfo->renderArea;
      }
   }

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
   {
      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_BEGIN_RENDER);
      if (!op)
         return;
      op->fb = fb;
      /* The attachment's sample count, taken from the image the rendering
       * binds rather than from the pipeline, because it is what the
       * framebuffer-sized buffers have to be sized for. */
      cmd->fb_samples = cimg ? MAX2(cimg->vk.samples, 1u) : 1;
      op->fb_samples = cmd->fb_samples;
   }

   /* LOAD_OP_CLEAR, recorded in order with the draws that follow it. */
   if (cat && cimg && cat->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      enum pipe_format pfmt = vk_format_to_pipe_format(cimg->vk.format);
      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
      if (!op)
         return;
      op->clear = (struct cpvk_clear) {
         .data = fb.color,
         .width = fb.width,
         .height = fb.height,
         .stride = cimg->row_stride[0],
         .pixel_size = util_format_get_blocksize(pfmt),
         .samples = cmd->fb_samples,
         .sample_stride = fb.color_sample_stride,
      };
      /* Packed here rather than in the kernel: cp_clear_rect takes the value
       * already packed, and packing it twice produces a plausible wrong
       * colour rather than an obvious one. */
      util_format_pack_rgba(pfmt, op->clear.value,
                            cat->clearValue.color.float32, 1);
   }

   const VkRenderingAttachmentInfo *dat = pRenderingInfo->pDepthAttachment;
   if (dat && dat->imageView && dat->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_CLEAR);
      if (!op)
         return;
      op->clear = (struct cpvk_clear) {
         .depth = true,
         .depth_value = dat->clearValue.depthStencil.depth,
      };
   }
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
   if (getenv("CPVK_DEBUG_PUSH")) {
      const float *f = (const float *)(const void *)cmd->push;
      fprintf(stderr, "push off=%u size=%u -> now %u: ", pInfo->offset,
              pInfo->size, cmd->push_size);
      for (unsigned k = 0; k < 9; k++)
         fprintf(stderr, "%.3f ", f[k]);
      fprintf(stderr, "\n");
   }
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
   if (!cmd->pipeline)
      return;

   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_DRAW);
   if (!op)
      return;
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
   memcpy(d->desc_hash, cmd->desc_hash, sizeof(d->desc_hash));
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

   /* So does a clear. */
   cp_batch_flush(cp);

   if (c->depth) {
      cp_clear_depthbuf(cp, c->depth_value);
      return;
   }

   /* Once per sample plane. */
   for (unsigned s = 0; s < MAX2(c->samples, 1u); s++)
      cp_clear_rect(cp, (char *)c->data + s * c->sample_stride, c->offset,
                    c->width, c->height, c->stride, c->pixel_size, c->value,
                    false);
}

/* ------------------------------------------------------------- batching */

/*
 * The pipeline state two draws must share to merge, in the form this front
 * end has it.
 *
 * A VkPipeline is immutable and holds the rasterizer, depth, blend, vertex
 * layout and topology, so its address covers all of them at once -- which is
 * the shape cp_draw_types.h anticipated for this driver. Only what a command
 * buffer can change without a new pipeline goes beside it.
 */
struct cpvk_batch_state {
   const void *pipeline;
   struct cp_viewport_state viewport;
   unsigned fb_samples;
   /*
    * The fragment stage's bindings. The renderer carries them per draw and
    * says so, so in principle they need not be a merge condition -- but
    * taking them out moved pushconstants from 5.098 to 7.336 and bought
    * nothing measurable (23.32 ms against 24.01 on Crossroads), so they stay
    * until something explains that.
    */
   uint64_t fs_ubos[CP_MAX_CONST_BUFFERS];
};

static const struct cp_batch_state_field cpvk_batch_state_field_table[] = {
   { "pipeline",   offsetof(struct cpvk_batch_state, pipeline),
                   sizeof(((struct cpvk_batch_state *)0)->pipeline) },
   { "viewport",   offsetof(struct cpvk_batch_state, viewport),
                   sizeof(((struct cpvk_batch_state *)0)->viewport) },
   { "fb_samples", offsetof(struct cpvk_batch_state, fb_samples),
                   sizeof(((struct cpvk_batch_state *)0)->fb_samples) },
   { "fs_ubos",    offsetof(struct cpvk_batch_state, fs_ubos),
                   sizeof(((struct cpvk_batch_state *)0)->fs_ubos) },
};

const struct cp_batch_state_field *
cp_batch_state_fields(unsigned *count)
{
   *count = ARRAY_SIZE(cpvk_batch_state_field_table);
   return cpvk_batch_state_field_table;
}

static_assert(sizeof(struct cpvk_batch_state) <= CP_BATCH_STATE_BYTES,
              "the batch key's state blob is too small for this front end");

/*
 * Everything that has to match for two draws to merge, filled from the
 * context the draw was just staged into. Compared with memcmp and nothing
 * else, so a field left out is simply not a merge condition -- which is why
 * this fills a zeroed struct and copies whole values rather than deriving
 * any of them.
 */
static void
cpvk_build_batch_key(struct cpvk_device *dev, const struct cpvk_draw *d,
                     struct cp_batch_key *key, bool blended)
{
   struct cp_context *cp = &dev->renderer;

   memset(key, 0, sizeof(*key));

   key->vs = cp->vs_shader;
   key->fs = cp->fs_shader;

   /* The colour target's identity. Natively an image view names it, and its
    * memory is what the kernels write, so both go in. */
   key->cbuf_texture = d->fb.color;
   key->color_data = d->fb.color;
   key->zs_texture = d->fb.has_zs ? (const void *)(uintptr_t)1 : NULL;
   key->visbuf = cp->visbuf;
   key->depthbuf = cp->depthbuf;
   key->fb_w = d->fb.width;
   key->fb_h = d->fb.height;
   key->fb_nr_cbufs = d->fb.nr_cbufs;
   key->fb_samples = cp->fb_samples;
   key->cbuf_format = (uint32_t)d->fb.color_encoding;

   key->mode = d->call.mode;
   key->index_size = d->call.index_size;
   key->start_instance = d->call.start_instance;
   key->index_resource = d->call.index_ptr;

   key->scissor = d->scissor;
   key->blend_enabled = blended;

   struct cpvk_batch_state state;
   memset(&state, 0, sizeof(state));
   state.pipeline = d->pipeline;
   state.viewport = d->viewport;
   state.fb_samples = cp->fb_samples;
   for (unsigned i = 0; i < CP_MAX_CONST_BUFFERS; i++)
      state.fs_ubos[i] = (uint64_t)(uintptr_t)cp->fs_ubos[i].buffer;
   memcpy(key->state, &state, sizeof(state));

   key->num_vertex_elements = cp->num_vertex_elements;
   key->vertex_stride = cp->vertex_stride;
   key->num_vertex_buffers = cp->num_vertex_buffers;
   key->num_fs_ubos = cp->num_fs_ubos;
   key->num_vs_ubos = cp->num_vs_ubos;
   key->sampler_table = cp->sampler_table;
   key->num_samplers = cp->num_samplers;
}

/*
 * Whether this draw may be held back at all: a property of the draw and the
 * state bound for it, never of what came before. Mirrors cp_batch_structural
 * in the Gallium adapter, which is the worked example.
 */
static bool
cpvk_batch_structural(struct cpvk_device *dev, const struct cpvk_draw *d)
{
   struct cp_context *cp = &dev->renderer;

   if (!cp->vs_shader || !cp->vs_shader->kernel ||
       !cp->fs_shader || !cp->fs_shader->kernel)
      return false;
   if (d->call.mode != MESA_PRIM_TRIANGLES)
      return false;
   if (d->call.index_size && !d->call.index_ptr)
      return false;

   /* A batch replays vertex buffers; a shader building positions from
    * gl_VertexIndex has nothing to gain and stays on the single-draw path. */
   if (!cp->num_vertex_buffers || !cp->vb_base[0])
      return false;

   if (!cp->fb.nr_cbufs || !cp->fb.color || !cp->visbuf || !cp->depthbuf)
      return false;

   uint64_t tris = (uint64_t)cp_triangles_for_draw(d->call.mode,
                                                   d->range.count) *
                   MAX2(d->call.instance_count, 1u);
   if (tris == 0 || tris > CP_MAX_BATCH_TRIS)
      return false;

   return true;
}

static bool cpvk_batch_eligible(struct cpvk_device *dev,
                                const struct cpvk_draw *d, bool *blended);

/*
 * Whether this draw can join whatever is pending -- answered without touching
 * the context, because the answer decides whether the context may be written.
 * The key is built from the draw and its pipeline; the few context fields it
 * needs (the framebuffer's buffers, the sampler table) belong to the pass and
 * are already current.
 */
/*
 * Whether two draws may be merged, decided by comparing the draws themselves.
 *
 * Everything a draw carries that must match is here, and everything a batch is
 * allowed to differ in -- the index range, the vertex-stage bindings, the
 * per-draw scissor -- is deliberately absent, exactly as cp_batch_key
 * documents for the Gallium side. The comparison touches no driver state at
 * all, which is what makes it safe to run before the draw is staged.
 */
static bool
cpvk_draws_mergeable(const struct cpvk_draw *a, const struct cpvk_draw *b)
{
#define CPVK_DIFF(cond, what)                                   \
   do {                                                         \
      if (cond) {                                               \
         if (cp_debug->debug_batchdiff)                          \
            fprintf(stderr, "batchdiff: %s\n", what);           \
         return false;                                          \
      }                                                         \
   } while (0)

   /*
    * The pipeline's state, not its identity. gltfscenerendering builds one
    * pipeline per material and draws them back to back; comparing addresses
    * made every batch a single draw, where the Gallium adapter compares the
    * state itself and merges them.
    */
   if (a->pipeline != b->pipeline) {
      const struct cpvk_pipeline *pa = a->pipeline, *pb = b->pipeline;
      CPVK_DIFF(!pa || !pb, "pipeline");
      CPVK_DIFF(pa->vs != pb->vs, "vertex shader");
      CPVK_DIFF(pa->fs != pb->fs, "fragment shader");
      CPVK_DIFF(memcmp(&pa->raster, &pb->raster, sizeof(pa->raster)), "rasterizer");
      CPVK_DIFF(memcmp(&pa->depth, &pb->depth, sizeof(pa->depth)), "depth state");
      CPVK_DIFF(memcmp(&pa->blend, &pb->blend, sizeof(pa->blend)), "blend state");
      CPVK_DIFF(pa->topology != pb->topology, "topology");
      CPVK_DIFF(pa->num_velem != pb->num_velem ||
                pa->vertex_stride != pb->vertex_stride ||
                memcmp(pa->velem, pb->velem,
                       pa->num_velem * sizeof(pa->velem[0])), "vertex layout");
      CPVK_DIFF(pa->samples != pb->samples, "sample count");
   }
   CPVK_DIFF(memcmp(&a->fb, &b->fb, sizeof(a->fb)), "framebuffer");
   CPVK_DIFF(memcmp(&a->viewport, &b->viewport, sizeof(a->viewport)), "viewport");
   CPVK_DIFF(memcmp(&a->scissor, &b->scissor, sizeof(a->scissor)), "scissor");
   CPVK_DIFF(a->call.mode != b->call.mode, "topology");
   CPVK_DIFF(a->call.index_size != b->call.index_size, "index size");
   CPVK_DIFF(a->call.index_ptr != b->call.index_ptr, "index buffer");
   CPVK_DIFF(a->call.start_instance != b->call.start_instance, "start instance");
   /*
    * The vertex offset, until the batched path stops taking it from the first
    * draw. cp_draw_execute builds its vertex fetch with
    * `first_vertex = draws[0].index_bias` for the whole batch, so draws whose
    * vertexOffset differs must not merge -- which is what gltfscenerendering's
    * per-primitive offsets are.
    */
   CPVK_DIFF(a->range.index_bias != b->range.index_bias, "vertex offset");
   CPVK_DIFF(a->call.instance_count != b->call.instance_count, "instance count");
   /*
    * The descriptors, by content.
    *
    * cpvk_batch and cpvk_batchtex show two and three draws differing only in
    * their descriptor set -- uniform buffers in one, textures in the other --
    * merging correctly without this, so in principle the per-draw binding
    * rows make it unnecessary. In practice removing it still costs
    * gltfscenerendering its correctness (0.000 to 20.768) even with the
    * vertex offset now a merge condition, and one measurement to the contrary
    * in the same turn did not reproduce. It stays until that is settled.
    */
   /*
    * The descriptors, by content.
    *
    * Required, measured three runs each way: gltfscenerendering is 0.000 with
    * this and 20.768 without. Why is still open -- every property of those
    * draws that could plausibly need it has been reproduced in cpvk_batch and
    * cpvk_batchtex and merges correctly without it: three draws, three
    * descriptor sets, three textures, overlapping geometry at three depths
    * with the depth test on, and a discarding fragment shader.
    *
    * What is left untested is the size of the batch (nine draws there, three
    * here) and a layout with more than one descriptor set. CPVK_NO_DESC_KEY
    * turns it off for the next person who wants to bisect that.
    */
   if (!getenv("CPVK_NO_DESC_KEY"))
      CPVK_DIFF(memcmp(a->desc_hash, b->desc_hash, sizeof(a->desc_hash)),
                "descriptors");
   CPVK_DIFF(a->num_vb != b->num_vb, "vertex buffer count");
   CPVK_DIFF(memcmp(a->vb_base, b->vb_base, sizeof(a->vb_base)), "vertex buffers");
   CPVK_DIFF(a->push_size != b->push_size, "push constant size");
   CPVK_DIFF(memcmp(a->push, b->push, a->push_size), "push constants");
   return true;
#undef CPVK_DIFF
}

static bool
cpvk_batch_can_join(struct cpvk_device *dev, const struct cpvk_draw *d,
                    bool *out_blended)
{
   struct cp_context *cp = &dev->renderer;
   bool blended = false;

   if (!cpvk_batch_eligible(dev, d, &blended))
      return false;

   /* Out to the caller, which stages it on the batch: it decides at flush
    * whether the batch appends to a blended pass episode or an opaque one. */
   *out_blended = blended;

   unsigned tris = cp_triangles_for_draw(d->call.mode, d->range.count) *
                   MAX2(d->call.instance_count, 1u);

   if (!cp->batch.pending)
      return true;

   if (!dev->prev_draw_valid || !dev->prev_draw ||
       !cpvk_draws_mergeable(d, dev->prev_draw)) {
      if (cp_debug->debug_batchdiff)
         fprintf(stderr, "batchdiff: the draws differ\n");
      return false;
   }
   if (cp->batch.ndraws >= (unsigned)cp_debug->batch_max)
      return false;
   if (cp->batch.tris + tris > CP_MAX_BATCH_TRIS)
      return false;

   return true;
}

static bool
cpvk_batch_eligible(struct cpvk_device *dev, const struct cpvk_draw *d,
                    bool *blended)
{
   /*
    * Off, and measured rather than abandoned.
    *
    * Batching is worth a great deal here: with it the Crossroads capture
    * replays at 5.63 ms against 24.28 without, and the old capture at 12.71
    * against 78.90 -- both faster than the Gallium-hosted driver's 7.13 and
    * 25.20. The machinery is the renderer's and already linked in.
    *
    * But this front end's key is not yet a complete statement of what has to
    * match. With it on, computeshader goes from 4.427 to 83.474 against the
    * Gallium driver, bloom from 17.386 to 26.071 and instancing from 11.851
    * to 14.467, while particlesystem improves from 41.036 to 6.785. Adding
    * the fragment stage's bindings to the state blob fixed multithreading
    * (0.641 to 0.012) and moved none of the others, and the blended path is
    * not implicated: disabling it changes nothing, so the fault is in the
    * opaque half.
    *
    * Something a draw carries and this key does not is still varying inside a
    * batch. Finding it is the remaining work, and CUDAPIPE_DEBUG_BATCHDIFF
    * against the Gallium driver's key on the same sample is the way in.
    */
   /*
    * Off. It is correct now -- every sample matches the unbatched result --
    * and it is worth nothing measurable, because almost nothing merges: the
    * key holds the pipeline's *address*, so two pipelines with identical
    * state never merge, where the Gallium adapter compares the state itself
    * and merges them. Keying on the resolved state instead is the work that
    * would make this pay, and until then an always-false batcher is honest
    * about what it does.
    */
   if (!getenv("CPVK_BATCH"))
      return false;

   if (cp_debug->no_batch)
      return false;
   if (!cpvk_batch_structural(dev, d))
      return false;

   if (cp_batch_order_free(&dev->renderer)) {
      *blended = false;
      return true;
   }
   /*
    * The blended half, which merges through the A-buffer: the order fragments
    * composite in is decided per pixel rather than by submission order, so
    * merging is sound in principle. It was left off for want of a test.
    *
    * cpvk_batchblend is that test -- cpvk_batchtex with blending on and depth
    * writes off, nine draws over nine textures at nine depths -- and it is
    * what this flag is validated against.
    *
    * It matters far more than the opaque half. A blended draw that does not
    * join a batch never reaches an A-buffer pass episode, because the episode
    * is entered from a batch flush; and without episodes every blended draw
    * builds, sorts and peels its own A-buffer. Measured on the Crossroads
    * capture that is 3,381 kernel launches a frame against the Gallium
    * driver's 245, with cp_peel_advance alone running 109 times a frame
    * against 0.1.
    */
   if (getenv("CPVK_BATCH_BLEND")) {
      *blended = true;
      return true;
   }
   return false;
}

/* Run one recorded draw through the renderer. */
void
cpvk_execute_draw(struct cpvk_device *dev, const struct cpvk_draw *d)
{
   struct cp_context *cp = &dev->renderer;
   struct cpvk_pipeline *p = d->pipeline;

   /*
    * Decide about the batch *before* staging this draw's state.
    *
    * A flush renders what is already held back, and it reads the context to
    * do it -- shaders, descriptors, vertex bindings. Staging first and
    * deciding afterwards meant every flush rendered the previous batch with
    * this draw's state. It showed up as computeshader being wrong even at
    * CUDAPIPE_BATCH_MAX=1, where each draw is its own batch and the result is
    * supposed to be bit-identical to not batching at all -- which is exactly
    * what that flag is for.
    */
   bool batch_blended = false;
   bool batch_ok = cpvk_batch_can_join(dev, d, &batch_blended);
   if (!batch_ok)
      cp_batch_flush_why(cp, "the next draw cannot join");

   if (!dev->prev_draw)
      dev->prev_draw = malloc(sizeof(*dev->prev_draw));
   if (dev->prev_draw) {
      *dev->prev_draw = *d;
      dev->prev_draw_valid = true;
   }

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
      cp->vs_ubos[i].buffer = (void *)(uintptr_t)
         (d->addrs[i] ? d->addrs[i] : dev->null_desc);
      cp->vs_ubos[i].managed_copy = 0;
      cp->vs_ubos[i].user_copy = false;
      cp->fs_ubos[i].buffer = (void *)(uintptr_t)
         (d->addrs[i] ? d->addrs[i] : dev->null_desc);
      cp->fs_ubos[i].managed_copy = 0;
      cp->fs_ubos[i].user_copy = false;
   }
   /*
    * Slot zero, and never a null. This assignment used to overwrite the null
    * descriptor the loop above had just put there, so a draw with no push
    * constants -- or one whose upload came back empty because the arena was
    * exhausted -- pointed a shader at address zero. compute-sanitizer put it
    * at `main+0x230 reading 4 bytes at 0x0`, in the second render pass of a
    * command buffer, which is what stopped pbribl's spheres.
    */
   CUdeviceptr push_slot = push_dev ? push_dev : dev->null_desc;
   cp->vs_ubos[CPVK_UBO_PUSH_SLOT].buffer = (void *)(uintptr_t)push_slot;
   cp->fs_ubos[CPVK_UBO_PUSH_SLOT].buffer = (void *)(uintptr_t)push_slot;
   cp->num_vs_ubos = cp->num_fs_ubos = CP_MAX_CONST_BUFFERS;



   cp_context_publish_state(cp);

   /*
    * Hold the draw back if it can join the one before it. The machinery is
    * the renderer's -- cp_batch_record, the key, the flush discipline -- and
    * what this front end supplies is the key and the decision, exactly as
    * cp_draw_vbo does for Gallium.
    *
    * Measured worth: with batching off the Gallium driver replays Crossroads
    * at 27.88 ms and with it at 7.13, and this driver had no batching at all.
    */
   if (batch_ok) {
      unsigned tris = cp_triangles_for_draw(d->call.mode, d->range.count) *
                      MAX2(d->call.instance_count, 1u);
      if (!cp->batch.pending) {
         cpvk_build_batch_key(dev, d, &cp->batch.key, batch_blended);
         cp->batch.info = d->call;
         cp->batch.drawid_offset = 0;
         cp->batch.pending = true;
         /*
          * Whether this batch is blended, which decides at flush whether it
          * appends to an A-buffer pass episode or to the opaque one. It was
          * hardcoded false, so a blended batch -- once the front end merged
          * one at all -- still took the opaque branch and no episode was ever
          * entered.
          */
         cp->batch.blended = batch_blended;
      }
      cp_batch_record(cp, &d->range, tris, 0, d->call.instance_count);
      return;
   }

   cp_draw_execute(cp, &d->call, 0, &d->range, 1, 1, NULL, NULL, NULL, NULL,
                   NULL, NULL);
}

/* ----------------------------------------------------------- transfers */

static struct cpvk_copy *
cpvk_record_copy(struct cpvk_cmd_buffer *cmd)
{
   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_COPY);
   return op ? &op->copy : NULL;
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
         .src_end = src->mem->dev_ptr + src->offset + src->vk.size,
         .dst_end = dst->mem->dev_ptr + dst->offset + dst->vk.size,
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

/* One past the last byte an image's allocation covers. */
static uint64_t
cpvk_image_end(const struct cpvk_image *img)
{
   return (img && img->mem) ? img->mem->dev_ptr + img->offset + img->size : 0;
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
      bool sok = cpvk_image_plane(src, r->srcSubresource.mipLevel, &sb, &sp, &sbpp);
      bool dok = cpvk_image_plane(dst, r->dstSubresource.mipLevel, &db, &dp, &dbpp);
      if (getenv("CPVK_DEBUG_RT"))
         fprintf(stderr, "copyimg: %ux%u src(l=%u lay=%u ok=%d) dst(l=%u lay=%u "
                 "ok=%d) dstimg=%ux%u layers=%u mips=%u\n", r->extent.width,
                 r->extent.height, r->srcSubresource.mipLevel,
                 r->srcSubresource.baseArrayLayer, sok,
                 r->dstSubresource.mipLevel, r->dstSubresource.baseArrayLayer,
                 dok, dst->vk.extent.width, dst->vk.extent.height,
                 dst->vk.array_layers, dst->vk.mip_levels);
      if (!sok || !dok)
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
 * Buffer <-> image, which is how every texture in the sample suite is
 * uploaded: staged into a buffer, then copied in. bufferRowLength of zero
 * means tightly packed, which is not the same as the image's row pitch and is
 * the whole reason this cannot be one flat memcpy.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer,
                           const VkCopyBufferToImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_buffer, buf, pInfo->srcBuffer);
   VK_FROM_HANDLE(cpvk_image, img, pInfo->dstImage);

   if (!buf || !buf->mem)
      return;

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pInfo->pRegions[i];
      CUdeviceptr ib;
      size_t ip;
      unsigned bpp;
      if (!cpvk_image_plane(img, r->imageSubresource.mipLevel, &ib, &ip, &bpp))
         return;

      /*
       * In blocks, not texels. A block-compressed format has a blocksize per
       * 4x4 block, so measuring a row as width * blocksize overstates it
       * sixteenfold: the capture's 1024-wide BC image wanted a two-megabyte
       * read out of a 128 KB staging buffer, and cuMemcpy2DAsync answers that
       * by faulting inside libcuda. For an uncompressed format the block is
       * one texel and this is the same arithmetic as before.
       */
      enum pipe_format pfmt = vk_format_to_pipe_format(img->vk.format);
      unsigned row_texels = r->bufferRowLength ? r->bufferRowLength
                                               : r->imageExtent.width;
      unsigned img_rows = r->bufferImageHeight ? r->bufferImageHeight
                                               : r->imageExtent.height;
      size_t src_row = (size_t)util_format_get_nblocksx(pfmt, row_texels) * bpp;
      size_t copy_row = (size_t)util_format_get_nblocksx(pfmt,
                                                        r->imageExtent.width) * bpp;
      unsigned copy_rows = util_format_get_nblocksy(pfmt, r->imageExtent.height);
      size_t layer_bytes = src_row *
                           util_format_get_nblocksy(pfmt, img_rows);
      unsigned layers = MAX2(r->imageSubresource.layerCount, 1u);

      for (unsigned l = 0; l < layers; l++) {
         struct cpvk_copy *c = cpvk_record_copy(cmd);
         if (!c)
            return;
         *c = (struct cpvk_copy) {
            .src = buf->mem->dev_ptr + buf->offset + r->bufferOffset +
                   l * layer_bytes,
            .dst = ib + (size_t)(r->imageSubresource.baseArrayLayer + l) *
                        img->level_size[r->imageSubresource.mipLevel] +
                   (size_t)r->imageOffset.y * ip +
                   (size_t)r->imageOffset.x * bpp,
            .src_pitch = src_row,
            .dst_pitch = ip,
            .width_bytes = copy_row,
            .rows = copy_rows,
            /* Both ends bounded: this is the path that uploads every texture
             * and every mip level in a capture, and it was the one with no
             * limits on it. */
            .src_end = buf->mem->dev_ptr + buf->offset + buf->vk.size,
            .dst_end = cpvk_image_end(img),
         };
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer,
                           const VkCopyImageToBufferInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, img, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_buffer, buf, pInfo->dstBuffer);

   if (!buf || !buf->mem)
      return;

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkBufferImageCopy2 *r = &pInfo->pRegions[i];
      CUdeviceptr ib;
      size_t ip;
      unsigned bpp;
      if (!cpvk_image_plane(img, r->imageSubresource.mipLevel, &ib, &ip, &bpp))
         return;

      enum pipe_format pfmt = vk_format_to_pipe_format(img->vk.format);
      unsigned row_texels = r->bufferRowLength ? r->bufferRowLength
                                               : r->imageExtent.width;
      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      *c = (struct cpvk_copy) {
         .src = ib + (size_t)r->imageSubresource.baseArrayLayer *
                     img->level_size[r->imageSubresource.mipLevel] +
                (size_t)r->imageOffset.y * ip +
                (size_t)r->imageOffset.x * bpp,
         .dst = buf->mem->dev_ptr + buf->offset + r->bufferOffset,
         .src_pitch = ip,
         .dst_pitch = (size_t)util_format_get_nblocksx(pfmt, row_texels) * bpp,
         .width_bytes = (size_t)util_format_get_nblocksx(pfmt,
                                                        r->imageExtent.width) * bpp,
         .rows = util_format_get_nblocksy(pfmt, r->imageExtent.height),
         .src_end = cpvk_image_end(img),
         .dst_end = buf->mem->dev_ptr + buf->offset + buf->vk.size,
      };
   }
}

static bool
cpvk_is_bgra(VkFormat f)
{
   switch (f) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_SNORM:
      return true;
   default:
      return false;
   }
}

/*
 * A blit, for the case this driver actually sees: same extent, same texel
 * size, no scaling and no format conversion. That is what an offscreen app
 * does to get a rendered image into a linear buffer it can save.
 *
 * A scaling or converting blit is refused rather than approximated, because a
 * silently wrong image is the failure this driver's history is made of. It
 * will be a kernel when something needs it.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBlitImage2(VkCommandBuffer commandBuffer,
                   const VkBlitImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, src, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_image, dst, pInfo->dstImage);

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageBlit2 *r = &pInfo->pRegions[i];
      CUdeviceptr sb, db;
      size_t sp, dp;
      unsigned sbpp, dbpp;
      if (!cpvk_image_plane(src, r->srcSubresource.mipLevel, &sb, &sp, &sbpp) ||
          !cpvk_image_plane(dst, r->dstSubresource.mipLevel, &db, &dp, &dbpp))
         return;

      int sw = r->srcOffsets[1].x - r->srcOffsets[0].x;
      int sh = r->srcOffsets[1].y - r->srcOffsets[0].y;
      int dw = r->dstOffsets[1].x - r->dstOffsets[0].x;
      int dh = r->dstOffsets[1].y - r->dstOffsets[0].y;

      if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || sbpp != dbpp) {
         fprintf(stderr, "cudapipe: vkCmdBlitImage %dx%d -> %dx%d (%u/%u bpp) "
                 "changes texel size or inverts, which is not implemented\n",
                 sw, sh, dw, dh, sbpp, dbpp);
         return;
      }

      bool scaling = (sw != dw || sh != dh);
      if (scaling && sbpp != 4) {
         fprintf(stderr, "cudapipe: vkCmdBlitImage scales a %u-byte texel, "
                 "which is not implemented\n", sbpp);
         return;
      }

      /* Same size and same texel width, so the only conversion a blit can be
       * asked for here is a channel order. Anything else is refused rather
       * than approximated. */
      bool swap_rb = false;
      if (src->vk.format != dst->vk.format) {
         if (sbpp == 4 && cpvk_is_bgra(src->vk.format) !=
                          cpvk_is_bgra(dst->vk.format)) {
            swap_rb = true;
         } else {
            fprintf(stderr, "cudapipe: vkCmdBlitImage %u -> %u is a format "
                    "conversion that is not implemented\n",
                    src->vk.format, dst->vk.format);
            return;
         }
      }

      struct cpvk_copy *c = cpvk_record_copy(cmd);
      if (!c)
         return;
      *c = (struct cpvk_copy) {
         .src = sb + (size_t)r->srcOffsets[0].y * sp +
                (size_t)r->srcOffsets[0].x * sbpp,
         .dst = db + (size_t)r->dstOffsets[0].y * dp +
                (size_t)r->dstOffsets[0].x * dbpp,
         .src_pitch = sp,
         .dst_pitch = dp,
         .width_bytes = (size_t)sw * sbpp,
         .rows = sh,
         .src_end = cpvk_image_end(src),
         .dst_end = cpvk_image_end(dst),
         .swap_rb = swap_rb,
         .src_w = scaling ? (unsigned)sw : 0,
         .src_h = scaling ? (unsigned)sh : 0,
         .dst_w = scaling ? (unsigned)dw : 0,
         .dst_h = scaling ? (unsigned)dh : 0,
         .bpp = sbpp,
         .filter_linear = pInfo->filter == VK_FILTER_LINEAR,
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);

   /* The render pass's own resolve, recorded where it happens: at the end of
    * the rendering, after the draws and before whatever reads the result. */
   struct cpvk_image *src = cmd->resolve_src, *dst = cmd->resolve_dst;
   if (!src || !dst)
      return;

   CUdeviceptr sb, db;
   size_t sp, dp;
   unsigned sbpp, dbpp;
   if (!cpvk_image_plane(src, 0, &sb, &sp, &sbpp) ||
       !cpvk_image_plane(dst, 0, &db, &dp, &dbpp) || sbpp != dbpp || sbpp != 4)
      return;

   struct cpvk_copy *c = cpvk_record_copy(cmd);
   if (!c)
      return;
   *c = (struct cpvk_copy) {
      .src = sb,
      .dst = db,
      .src_pitch = sp,
      .dst_pitch = dp,
      .width_bytes = (size_t)cmd->resolve_area.extent.width * sbpp,
      .rows = cmd->resolve_area.extent.height,
      .src_end = cpvk_image_end(src),
      .dst_end = cpvk_image_end(dst),
      .samples = MAX2(src->vk.samples, 1u),
      .sample_stride = src->sample_stride,
   };

   cmd->resolve_src = NULL;
   cmd->resolve_dst = NULL;
}

/*
 * A multisample resolve: the average of the samples, which are planes
 * cpvk_image::sample_stride apart.
 *
 * It took sample zero and said so out loud until images were sized for their
 * samples, because until then there was nothing else in memory to average.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResolveImage2(VkCommandBuffer commandBuffer,
                      const VkResolveImageInfo2 *pInfo)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_image, src, pInfo->srcImage);
   VK_FROM_HANDLE(cpvk_image, dst, pInfo->dstImage);

   for (uint32_t i = 0; i < pInfo->regionCount; i++) {
      const VkImageResolve2 *r = &pInfo->pRegions[i];
      CUdeviceptr sb, db;
      size_t sp, dp;
      unsigned sbpp, dbpp;
      if (!cpvk_image_plane(src, r->srcSubresource.mipLevel, &sb, &sp, &sbpp) ||
          !cpvk_image_plane(dst, r->dstSubresource.mipLevel, &db, &dp, &dbpp))
         return;
      if (sbpp != dbpp || sbpp != 4) {
         fprintf(stderr, "cudapipe: vkCmdResolveImage of a %u-byte texel is "
                 "not implemented\n", sbpp);
         return;
      }

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
         .src_end = cpvk_image_end(src),
         .dst_end = cpvk_image_end(dst),
         .samples = MAX2(src->vk.samples, 1u),
         .sample_stride = src->sample_stride,
      };
   }
}


/* ------------------------------------------------------------------ events
 *
 * The device-side half is empty on purpose. One stream, program order: a
 * vkCmdWaitEvents2 recorded after the vkCmdSetEvent2 that satisfies it is
 * already ordered behind it, and a wait on an event set by the host cannot be
 * reached until that host call has happened. The host-side half is real,
 * because vkSetEvent and vkGetEventStatus are questions with answers.
 */
VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateEvent(VkDevice _device, const VkEventCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_event *event =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event),
                       VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pEvent = cpvk_event_to_handle(event);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyEvent(VkDevice _device, VkEvent _event,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_event, event, _event);

   if (event)
      vk_object_free(&dev->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_GetEventStatus(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   return (event && event->signaled) ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_SetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   if (event)
      event->signaled = true;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_ResetEvent(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   if (event)
      event->signaled = false;
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   if (event)
      event->signaled = true;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                    VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(cpvk_event, event, _event);
   if (event)
      event->signaled = false;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                    const VkEvent *pEvents,
                    const VkDependencyInfo *pDependencyInfos)
{
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
cpvk_execute_begin_render(struct cpvk_device *dev, const struct cp_fb_desc *fb,
                          unsigned samples)
{
   /*
    * End whatever the previous pass left open before binding the next
    * framebuffer, which is what the Gallium adapter does at the same point:
    * "the visibility and depth buffers may be freed below, and the held-back
    * draws were recorded against the framebuffer that is going away."
    *
    * The native path did neither, and a second vkCmdBeginRendering in one
    * command buffer made the draws after it fault in the rasterizer -- which
    * is what stopped pbribl's spheres, after three offscreen passes.
    */
   cp_batch_flush_why(&dev->renderer, "framebuffer");
   if (getenv("CPVK_DEBUG_EPISODE"))
      fprintf(stderr, "episode-cut: begin_render\n");
   cp_pass_finish(&dev->renderer);

   cp_context_set_framebuffer(&dev->renderer, fb, MAX2(samples, 1u));
}

void
cpvk_execute_copy(struct cpvk_device *dev, const struct cpvk_copy *c)
{
   struct cp_context *cp = &dev->renderer;

   /* A copy observes rendering, so whatever is held back has to run first. */
   cp_batch_flush(cp);

   cuCtxSetCurrent(dev->cu_ctx);

   /*
    * Refuse a copy that cannot be one. A zero endpoint is an image or buffer
    * whose memory was never bound, and a pitch narrower than the row is a
    * region computed from the wrong subresource; CUDA takes both as an
    * invitation to walk off the end.
    */
   /*
    * A scaling blit describes its geometry in src_w/src_h and dst_w/dst_h and
    * carries the source's row width in width_bytes, so the pitch and reach
    * tests below do not apply to it -- they rejected three perfectly good mip
    * downscales before this line existed.
    */
   uint64_t src_reach = c->src_w ? 0 :
                        c->src + (uint64_t)(c->rows ? c->rows - 1 : 0) *
                        (c->src_pitch ? c->src_pitch : c->width_bytes) +
                        c->width_bytes;
   uint64_t dst_reach = c->src_w ? 0 :
                        c->dst + (uint64_t)(c->rows ? c->rows - 1 : 0) *
                        (c->dst_pitch ? c->dst_pitch : c->width_bytes) +
                        c->width_bytes;

   if (!c->src || !c->dst || !c->width_bytes || !c->rows ||
       (!c->src_w && c->src_pitch && c->src_pitch < c->width_bytes) ||
       (!c->src_w && c->dst_pitch && c->dst_pitch < c->width_bytes) ||
       (c->src_end && src_reach > c->src_end) ||
       (c->dst_end && dst_reach > c->dst_end)) {
      fprintf(stderr, "cudapipe: refusing copy src=%p dst=%p %zux%zu "
              "pitch %zu->%zu reach %p/%p ends %p/%p\n",
              (void *)(uintptr_t)c->src, (void *)(uintptr_t)c->dst,
              c->width_bytes, c->rows, c->src_pitch, c->dst_pitch,
              (void *)(uintptr_t)src_reach, (void *)(uintptr_t)dst_reach,
              (void *)(uintptr_t)c->src_end, (void *)(uintptr_t)c->dst_end);
      return;
   }

   if (c->rows <= 1 && !c->src_pitch && !c->dst_pitch) {
      cuMemcpyDtoDAsync(c->dst, c->src, c->width_bytes, cp->stream);
      return;
   }

   if (c->samples > 1) {
      /*
       * Resolve on the host, once per render pass. Every sample plane is read
       * and averaged into the destination; this is not on a frame's critical
       * path and a kernel can replace it when something resolves per draw.
       */
      size_t bytes = c->width_bytes * c->rows;
      uint8_t *acc_src = malloc(bytes);
      uint32_t *acc = calloc(bytes / 4 * 4, sizeof(uint32_t));
      uint8_t *out = malloc(bytes);
      if (!acc_src || !acc || !out) {
         free(acc_src); free(acc); free(out);
         return;
      }

      cuStreamSynchronize(cp->stream);
      for (unsigned s = 0; s < c->samples; s++) {
         CUDA_MEMCPY2D d2h = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .srcDevice = c->src + s * c->sample_stride,
            .srcPitch = c->src_pitch,
            .dstMemoryType = CU_MEMORYTYPE_HOST, .dstHost = acc_src,
            .dstPitch = c->width_bytes,
            .WidthInBytes = c->width_bytes, .Height = c->rows,
         };
         if (cuMemcpy2D(&d2h) != CUDA_SUCCESS)
            break;
         for (size_t i = 0; i < bytes; i++)
            acc[i] += acc_src[i];
      }

      for (size_t i = 0; i < bytes; i++)
         out[i] = (uint8_t)((acc[i] + c->samples / 2) / c->samples);

      CUDA_MEMCPY2D h2d = {
         .srcMemoryType = CU_MEMORYTYPE_HOST, .srcHost = out,
         .srcPitch = c->width_bytes,
         .dstMemoryType = CU_MEMORYTYPE_DEVICE, .dstDevice = c->dst,
         .dstPitch = c->dst_pitch,
         .WidthInBytes = c->width_bytes, .Height = c->rows,
      };
      cuMemcpy2D(&h2d);
      free(acc_src); free(acc); free(out);
      return;
   }

   if (getenv("CPVK_DEBUG_RT")) {
      /* The first texels of the source, as halves: what the pass just
       * rendered, before this copy places it. */
      uint16_t h[8] = { 0 };
      cuStreamSynchronize(cp->stream);
      if (cuMemcpyDtoH(h, c->src, sizeof(h)) == CUDA_SUCCESS) {
         fprintf(stderr, "copysrc halfs:");
         for (int k = 0; k < 8; k++)
            fprintf(stderr, " %04x", h[k]);
         fprintf(stderr, "\n");
      }
   }

   if (c->src_w) {
      /*
       * A scaling blit, on the host and synchronously, for the same reason
       * the converting one is: this builds a mip chain at load time and is
       * not on a frame's critical path. Box-filtered when the caller asked
       * for LINEAR, which is what a downscale by two wants and what every
       * mip generator asks for; point-sampled otherwise.
       */
      size_t src_bytes = (size_t)c->src_w * c->src_h * 4;
      size_t dst_bytes = (size_t)c->dst_w * c->dst_h * 4;
      uint8_t *src = malloc(src_bytes), *dst = malloc(dst_bytes);
      if (!src || !dst) {
         free(src); free(dst);
         return;
      }

      CUDA_MEMCPY2D d2h = {
         .srcMemoryType = CU_MEMORYTYPE_DEVICE, .srcDevice = c->src,
         .srcPitch = c->src_pitch,
         .dstMemoryType = CU_MEMORYTYPE_HOST, .dstHost = src,
         .dstPitch = (size_t)c->src_w * 4,
         .WidthInBytes = (size_t)c->src_w * 4, .Height = c->src_h,
      };
      cuStreamSynchronize(cp->stream);
      cuMemcpy2D(&d2h);

      for (unsigned y = 0; y < c->dst_h; y++) {
         for (unsigned x = 0; x < c->dst_w; x++) {
            uint8_t *o = dst + ((size_t)y * c->dst_w + x) * 4;
            if (c->filter_linear) {
               /* The source footprint of this destination texel, averaged.
                * Exact for the power-of-two halving a mip chain does. */
               unsigned x0 = x * c->src_w / c->dst_w;
               unsigned x1 = MAX2((x + 1) * c->src_w / c->dst_w, x0 + 1);
               unsigned y0 = y * c->src_h / c->dst_h;
               unsigned y1 = MAX2((y + 1) * c->src_h / c->dst_h, y0 + 1);
               unsigned acc[4] = { 0, 0, 0, 0 }, n = 0;
               for (unsigned sy = y0; sy < y1 && sy < c->src_h; sy++)
                  for (unsigned sx = x0; sx < x1 && sx < c->src_w; sx++) {
                     const uint8_t *s =
                        src + ((size_t)sy * c->src_w + sx) * 4;
                     for (int k = 0; k < 4; k++)
                        acc[k] += s[k];
                     n++;
                  }
               for (int k = 0; k < 4; k++)
                  o[k] = n ? (uint8_t)((acc[k] + n / 2) / n) : 0;
            } else {
               unsigned sx = x * c->src_w / c->dst_w;
               unsigned sy = y * c->src_h / c->dst_h;
               memcpy(o, src + ((size_t)sy * c->src_w + sx) * 4, 4);
            }

            if (c->swap_rb) {
               uint8_t r = o[0];
               o[0] = o[2];
               o[2] = r;
            }
         }
      }

      CUDA_MEMCPY2D h2d = {
         .srcMemoryType = CU_MEMORYTYPE_HOST, .srcHost = dst,
         .srcPitch = (size_t)c->dst_w * 4,
         .dstMemoryType = CU_MEMORYTYPE_DEVICE, .dstDevice = c->dst,
         .dstPitch = c->dst_pitch,
         .WidthInBytes = (size_t)c->dst_w * 4, .Height = c->dst_h,
      };
      cuMemcpy2D(&h2d);
      free(src);
      free(dst);
      return;
   }

   if (c->swap_rb) {
      /*
       * The converting blit, on the host and synchronously.
       *
       * This is the path an offscreen app takes once per saved frame, and it
       * is not on any frame's critical path. Doing it here rather than as a
       * kernel keeps the kernel set as it is; if something ever blits per
       * frame, this is the line that says it needs one.
       */
      size_t bytes = c->width_bytes * c->rows;
      uint32_t *tmp = malloc(bytes);
      if (!tmp)
         return;

      CUDA_MEMCPY2D d2h = {
         .srcMemoryType = CU_MEMORYTYPE_DEVICE, .srcDevice = c->src,
         .srcPitch = c->src_pitch,
         .dstMemoryType = CU_MEMORYTYPE_HOST, .dstHost = tmp,
         .dstPitch = c->width_bytes,
         .WidthInBytes = c->width_bytes, .Height = c->rows,
      };
      cuStreamSynchronize(cp->stream);
      cuMemcpy2D(&d2h);

      for (size_t i = 0; i < bytes / 4; i++) {
         uint32_t v = tmp[i];
         tmp[i] = (v & 0xFF00FF00u) | ((v & 0x00FF0000u) >> 16) |
                  ((v & 0x000000FFu) << 16);
      }

      CUDA_MEMCPY2D h2d = {
         .srcMemoryType = CU_MEMORYTYPE_HOST, .srcHost = tmp,
         .srcPitch = c->width_bytes,
         .dstMemoryType = CU_MEMORYTYPE_DEVICE, .dstDevice = c->dst,
         .dstPitch = c->dst_pitch,
         .WidthInBytes = c->width_bytes, .Height = c->rows,
      };
      cuMemcpy2D(&h2d);
      free(tmp);
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

   if (getenv("CPVK_DEBUG_RT")) {
      /* And what landed, read back from the destination this copy just
       * wrote: the face of the cube level, not the offscreen it came from. */
      uint16_t hd[8] = { 0 };
      cuStreamSynchronize(cp->stream);
      if (cuMemcpyDtoH(hd, c->dst, sizeof(hd)) == CUDA_SUCCESS) {
         fprintf(stderr, "copydst rows=%zu wb=%zu halfs:", (size_t)c->rows,
                 (size_t)c->width_bytes);
         for (int k = 0; k < 8; k++)
            fprintf(stderr, " %04x", hd[k]);
         fprintf(stderr, "\n");
      }
   }
}


/* ------------------------------------------------------------- queries */

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateQueryPool(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_query_pool *pool =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                       VK_OBJECT_TYPE_QUERY_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->type = pCreateInfo->queryType;
   pool->count = pCreateInfo->queryCount;
   pool->results = calloc(pool->count, sizeof(*pool->results));
   pool->available = calloc(pool->count, sizeof(*pool->available));
   if (!pool->results || !pool->available) {
      free(pool->results);
      free(pool->available);
      vk_object_free(&dev->vk, pAllocator, pool);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *pQueryPool = cpvk_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyQueryPool(VkDevice _device, VkQueryPool _pool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);

   if (!pool)
      return;
   free(pool->results);
   free(pool->available);
   vk_object_free(&dev->vk, pAllocator, pool);
}

static void
cpvk_record_query(struct cpvk_cmd_buffer *cmd, struct cpvk_query_pool *pool,
                  uint32_t first, uint32_t count, bool reset)
{
   if (!pool)
      return;
   struct cpvk_op *op = cpvk_op_alloc(cmd, CPVK_OP_QUERY);
   if (!op)
      return;
   op->query = (struct cpvk_query_op) {
      .pool = pool, .first = first, .count = count, .reset = reset,
   };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                       uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   cpvk_record_query(cmd, pool, firstQuery, queryCount, true);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWriteTimestamp2(VkCommandBuffer commandBuffer,
                        VkPipelineStageFlags2 stage, VkQueryPool _pool,
                        uint32_t query)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   cpvk_record_query(cmd, pool, query, 1, false);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdWriteTimestamp(VkCommandBuffer commandBuffer,
                       VkPipelineStageFlagBits stage, VkQueryPool pool,
                       uint32_t query)
{
   cpvk_CmdWriteTimestamp2(commandBuffer, stage, pool, query);
}

/* Occlusion and statistics queries: recorded so the pool becomes available,
 * counted as zero because nothing counts them. */
VKAPI_ATTR void VKAPI_CALL
cpvk_CmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool pool,
                   uint32_t query, VkQueryControlFlags flags)
{
}

VKAPI_ATTR void VKAPI_CALL
cpvk_CmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                 uint32_t query)
{
   VK_FROM_HANDLE(cpvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);
   cpvk_record_query(cmd, pool, query, 1, false);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_GetQueryPoolResults(VkDevice _device, VkQueryPool _pool,
                         uint32_t firstQuery, uint32_t queryCount,
                         size_t dataSize, void *pData, VkDeviceSize stride,
                         VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(cpvk_query_pool, pool, _pool);

   if (!pool)
      return VK_ERROR_UNKNOWN;

   char *out = pData;
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < queryCount; i++) {
      uint32_t q = firstQuery + i;
      bool avail = q < pool->count && pool->available[q];
      uint64_t value = avail ? pool->results[q] : 0;

      if (!avail)
         result = VK_NOT_READY;

      char *slot = out + (size_t)i * stride;
      if (flags & VK_QUERY_RESULT_64_BIT) {
         uint64_t *p = (uint64_t *)slot;
         p[0] = value;
         if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
            p[1] = avail;
      } else {
         uint32_t *p = (uint32_t *)slot;
         p[0] = (uint32_t)value;
         if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
            p[1] = avail;
      }
   }

   return result;
}

void
cpvk_execute_query(struct cpvk_device *dev, const struct cpvk_query_op *q)
{
   struct cpvk_query_pool *pool = q->pool;

   if (q->reset) {
      for (uint32_t i = 0; i < q->count; i++)
         if (q->first + i < pool->count) {
            pool->results[q->first + i] = 0;
            pool->available[q->first + i] = false;
         }
      return;
   }

   if (q->first >= pool->count)
      return;

   uint64_t value = 0;
   if (pool->type == VK_QUERY_TYPE_TIMESTAMP) {
      /* Nanoseconds, which is what timestampPeriod says a tick is. The
       * stream has reached this point because the ops before it have already
       * been issued and a submit synchronises before returning. */
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      value = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
   }

   pool->results[q->first] = value;
   pool->available[q->first] = true;
}


