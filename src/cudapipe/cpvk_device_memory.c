/*
 * Device, queue, memory and buffers for the native driver.
 *
 * The memory types are the point of this file. Under lavapipe every
 * VkDeviceMemory became a pipe_screen::allocate_memory, which meant a 64-byte
 * descriptor set became a page-granular managed allocation the host wrote and
 * every launch then faulted on -- 22.4 ms of a 122.6 ms frame at its worst,
 * and two optimisation passes (a small-allocation arena, then a bounded
 * allocation cache) to claw back. Here the driver owns the allocator, and the
 * three memory types session 13 had to add hooks to lavapipe for are just
 * three branches.
 */

#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"

static VkResult
cpvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct cpvk_device *dev =
      container_of(vk_queue->base.device, struct cpvk_device, vk);

   cuCtxSetCurrent(dev->cu_ctx);

   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct cpvk_cmd_buffer *cmd =
         container_of(submit->command_buffers[i], struct cpvk_cmd_buffer, vk);
      VkResult result = cpvk_execute_cmd_buffer(dev, cmd);
      if (result != VK_SUCCESS)
         return result;

      /* In record order: a clear after a draw must not run before it. */
      for (unsigned o = 0; o < cmd->num_ops; o++) {
         switch (cmd->ops[o].kind) {
         case CPVK_OP_BEGIN_RENDER:
            cpvk_execute_begin_render(dev, &cmd->ops[o].fb);
            break;
         case CPVK_OP_CLEAR:
            cpvk_execute_clear(dev, &cmd->ops[o].clear);
            break;
         case CPVK_OP_COPY:
            cpvk_execute_copy(dev, &cmd->ops[o].copy);
            break;
         case CPVK_OP_DRAW:
            cpvk_execute_draw(dev, &cmd->ops[o].draw);
            break;
         }
      }
   }

   /* Ordering is the stream's; a submit promises only that everything it
    * recorded has run before any signal is observed. */
   cuStreamSynchronize(dev->stream);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDevice(VkPhysicalDevice physicalDevice,
                  const VkDeviceCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(cpvk_physical_device, pdev, physicalDevice);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &pdev->vk.instance->alloc;

   struct cpvk_device *dev = vk_zalloc(alloc, sizeof(*dev), 8,
                                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &cpvk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_device_entrypoints, false);

   VkResult result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table,
                                    pCreateInfo, alloc);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   /* After vk_device_init, which zeroes the device: the common command pool
    * dereferences this the moment a pool is created. */
   dev->vk.command_buffer_ops = &cpvk_cmd_buffer_ops;
   dev->pdev = pdev;

   /* CUDA 12.8: cuCtxCreate takes three arguments. Code written against
    * CUDA 13's four-argument form does not compile here. */
   if (cuCtxCreate(&dev->cu_ctx, 0, pdev->cu_dev) != CUDA_SUCCESS) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_device;
   }
   if (cuStreamCreate(&dev->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_ctx;
   }

   /*
    * The device the renderer runs on, and then the renderer: the same
    * NVRTC-compiled kernels and the same cp_context_init() the Gallium-hosted
    * driver uses, neither of which needs a pipe_screen or a pipe_context.
    */
   dev->cp_dev.cuda_device = pdev->cu_dev;
   dev->cp_dev.cuda_ctx = dev->cu_ctx;
   dev->cp_dev.sm_major = pdev->sm_major;
   dev->cp_dev.sm_minor = pdev->sm_minor;
   if (!cp_kernels_init(&dev->cp_dev.kernels, pdev->sm_major, pdev->sm_minor)) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_stream;
   }
   if (!cp_context_init(&dev->renderer, &dev->cp_dev)) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_stream;
   }

   result = vk_queue_init(&dev->queue, &dev->vk,
                          &pCreateInfo->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS)
      goto fail_stream;
   dev->queue.driver_submit = cpvk_queue_submit;

   *pDevice = cpvk_device_to_handle(dev);
   return VK_SUCCESS;

fail_stream:
   cuStreamDestroy(dev->stream);
fail_ctx:
   cuCtxDestroy(dev->cu_ctx);
fail_device:
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   if (!dev)
      return;

   vk_queue_finish(&dev->queue);
   cuStreamDestroy(dev->stream);
   cuCtxDestroy(dev->cu_ctx);

   const VkAllocationCallbacks *alloc = &dev->vk.alloc;
   vk_device_finish(&dev->vk);
   vk_free(alloc, dev);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_DeviceWaitIdle(VkDevice _device)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   cuCtxSetCurrent(dev->cu_ctx);
   return cuCtxSynchronize() == CUDA_SUCCESS
             ? VK_SUCCESS
             : vk_error(dev, VK_ERROR_DEVICE_LOST);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_AllocateMemory(VkDevice _device,
                    const VkMemoryAllocateInfo *pAllocateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_device_memory *mem =
      vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator,
                              sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->kind = pAllocateInfo->memoryTypeIndex;
   size_t size = pAllocateInfo->allocationSize;
   CUresult err;

   cuCtxSetCurrent(dev->cu_ctx);
   switch (mem->kind) {
   case CPVK_MEM_DEVICE:
      err = cuMemAlloc(&mem->dev_ptr, size);
      mem->host_ptr = NULL;
      break;
   case CPVK_MEM_HOST:
      err = cuMemHostAlloc(&mem->host_ptr, size, CU_MEMHOSTALLOC_DEVICEMAP);
      if (err == CUDA_SUCCESS)
         err = cuMemHostGetDevicePointer(&mem->dev_ptr, mem->host_ptr, 0);
      break;
   default:
      err = cuMemAllocManaged(&mem->dev_ptr, size, CU_MEM_ATTACH_GLOBAL);
      mem->host_ptr = (void *)(uintptr_t)mem->dev_ptr;
      break;
   }

   if (err != CUDA_SUCCESS) {
      vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   /*
    * Vulkan defines ordinary allocation contents as undefined, and clearing
    * them anyway cost the Gallium-hosted driver 1,571 blocking cuMemsetD8
    * calls and 830 ms of CUDA API time per replay before session 11 removed
    * it. Only the explicit request is honoured.
    */
   if (mem->vk.alloc_flags & VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT) {
      if (mem->host_ptr)
         memset(mem->host_ptr, 0, size);
      else
         cuMemsetD8(mem->dev_ptr, 0, size);
   }

   *pMem = cpvk_device_memory_to_handle(mem);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_FreeMemory(VkDevice _device, VkDeviceMemory _mem,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_device_memory, mem, _mem);

   if (!mem)
      return;

   cuCtxSetCurrent(dev->cu_ctx);
   switch (mem->kind) {
   case CPVK_MEM_HOST:
      cuMemFreeHost(mem->host_ptr);
      break;
   default:
      cuMemFree(mem->dev_ptr);
      break;
   }

   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_MapMemory2(VkDevice _device, const VkMemoryMapInfo *pMemoryMapInfo,
                void **ppData)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_device_memory, mem, pMemoryMapInfo->memory);

   /* A device-local allocation is deliberately not mappable: staging through
    * a host-visible allocation is the caller's job, and hiding that behind a
    * silent download and upload is what made cp_buffer_map copy an entire
    * Vulkan allocation around every partial write. */
   if (!mem->host_ptr)
      return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);

   *ppData = (char *)mem->host_ptr + pMemoryMapInfo->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_UnmapMemory2(VkDevice _device, const VkMemoryUnmapInfo *pMemoryUnmapInfo)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_FlushMappedMemoryRanges(VkDevice _device, uint32_t count,
                             const VkMappedMemoryRange *pRanges)
{
   return VK_SUCCESS;   /* every mappable type here is coherent */
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t count,
                                  const VkMappedMemoryRange *pRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateBuffer(VkDevice _device, const VkBufferCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   struct cpvk_buffer *buffer =
      vk_buffer_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*buffer));
   if (!buffer)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pBuffer = cpvk_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyBuffer(VkDevice _device, VkBuffer _buffer,
                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_buffer, buffer, _buffer);

   if (!buffer)
      return;

   vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetBufferMemoryRequirements2(VkDevice _device,
                                  const VkBufferMemoryRequirementsInfo2 *pInfo,
                                  VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(cpvk_buffer, buffer, pInfo->buffer);

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements) {
      .size = buffer->vk.size,
      .alignment = 256,
      /* Every type can back a buffer; which one the application picks is what
       * decides whether the host can see it. */
      .memoryTypeBits = 0x7,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_BindBufferMemory2(VkDevice _device, uint32_t bindInfoCount,
                       const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(cpvk_buffer, buffer, pBindInfos[i].buffer);
      VK_FROM_HANDLE(cpvk_device_memory, mem, pBindInfos[i].memory);

      buffer->mem = mem;
      buffer->offset = pBindInfos[i].memoryOffset;

      /* The runtime's own vk_buffer_address() asserts on this, and every
       * common entrypoint that takes a buffer goes through it. A CUDA device
       * pointer is a flat address, so the buffer's address is simply where it
       * was bound -- there is nothing to opt into. */
      buffer->vk.device_address = mem ?
         mem->dev_ptr + pBindInfos[i].memoryOffset : 0;
   }
   return VK_SUCCESS;
}
