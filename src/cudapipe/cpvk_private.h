/*
 * cudapipe, native Vulkan driver.
 *
 * This is the same CUDA rasterizer backend as the Gallium-hosted driver in
 * src/gallium/drivers/cudapipe, with Mesa's common Vulkan runtime in front of
 * it instead of lavapipe and Gallium. It exists because a large fraction of
 * the Gallium-hosted driver is machinery that reconstructs, by watching a
 * stream of state setters, what a Vulkan command buffer already states
 * explicitly: draw batching and its flush discipline, pass episodes and their
 * segment snapshots, the live save/restore around a deferred execute, the
 * fallback replay, and the small-allocation arenas that exist because every
 * descriptor set is backed by its own device allocation.
 *
 * See CUDAPIPE_VK_NATIVE.md for the plan, the evidence, and the staging.
 *
 * Nothing here is wired into rendering yet. The first milestone is the one
 * the original plan set for the Gallium driver: enumerate a device.
 */

#ifndef CPVK_PRIVATE_H
#define CPVK_PRIVATE_H

#include "vk_device.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_buffer.h"
#include "vk_device_memory.h"
#include "vk_log.h"

#include <cuda.h>

#include "cpvk_entrypoints.h"

struct cpvk_instance {
   struct vk_instance vk;
};

struct cpvk_physical_device {
   struct vk_physical_device vk;

   CUdevice cu_dev;
   int sm_major, sm_minor;
   char name[256];
   size_t vram;
};

struct cpvk_device {
   struct vk_device vk;
   struct cpvk_physical_device *pdev;

   CUcontext cu_ctx;
   CUstream stream;
   struct vk_queue queue;
};

/*
 * A memory type index picks the allocator, and the three of them are the split
 * session 13 had to negotiate with lavapipe through two new pipe_screen hooks.
 * Here they are simply what the driver does.
 */
enum cpvk_memory_kind {
   CPVK_MEM_DEVICE = 0,     /* cuMemAlloc: never host-mapped */
   CPVK_MEM_HOST = 1,       /* cuMemHostAlloc: pinned, host-cached */
   CPVK_MEM_MANAGED = 2,    /* cuMemAllocManaged: both, and migrating */
};

struct cpvk_device_memory {
   struct vk_device_memory vk;
   enum cpvk_memory_kind kind;
   CUdeviceptr dev_ptr;     /* what a kernel dereferences */
   void *host_ptr;          /* what vkMapMemory returns, or NULL */
};

struct cpvk_buffer {
   struct vk_buffer vk;
   struct cpvk_device_memory *mem;
   VkDeviceSize offset;
};

VK_DEFINE_HANDLE_CASTS(cpvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(cpvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(cpvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_buffer, vk.base, VkBuffer,
                               VK_OBJECT_TYPE_BUFFER)

#endif /* CPVK_PRIVATE_H */
