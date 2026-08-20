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
#include "vk_command_buffer.h"
#include "cp_nir_to_llvm.h"
#include "cp_kernels.h"
#include "cp_debug.h"
#include "cp_device.h"
#include "vk_buffer.h"
#include "vk_image.h"
#include "vk_descriptor_set_layout.h"
#include "vk_device_memory.h"
#include "vk_pipeline_layout.h"
#include "vk_log.h"

#include <cuda.h>

#include "cpvk_entrypoints.h"

struct cpvk_instance {
   struct vk_instance vk;
};

struct cpvk_physical_device {
   struct vk_physical_device vk;

   CUdevice cu_dev;
   struct cp_kernels kernels;      /* built once the device has a context */
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

   /* The device the renderer will run on. cp_context itself is still declared
    * in a header that pulls Gallium in; splitting that is the next slice. */
   struct cp_device cp_dev;
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

/*
 * Descriptors, resolved on the host.
 *
 * The generated kernel takes one argument: a pointer to an array of pointers,
 * where slot 18 + i is the base address of buffer i ("compute reads
 * args[18 + i] directly; it has no batch", cp_nir_to_llvm.c). So a descriptor
 * set here is an array of device addresses and nothing else, and binding one
 * is filling in argument slots -- there is no descriptor memory, no
 * VkDeviceMemory per set, and therefore none of the page-fault storm that
 * arrangement caused under lavapipe.
 */
#define CPVK_MAX_BINDINGS 32

struct cpvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   unsigned num_bindings;
   struct {
      VkDescriptorType type;
      unsigned count;
      unsigned flat;          /* index within this set */
   } bindings[CPVK_MAX_BINDINGS];
   unsigned num_descriptors;
};

struct cpvk_pipeline_layout {
   struct vk_pipeline_layout vk;
   unsigned set_base[MESA_VK_MAX_DESCRIPTOR_SETS];  /* flat base per set */
   unsigned num_descriptors;
};

struct cpvk_descriptor_set {
   struct vk_object_base base;
   struct cpvk_descriptor_set_layout *layout;
   CUdeviceptr addrs[CPVK_MAX_BINDINGS];
};

struct cpvk_descriptor_pool {
   struct vk_object_base base;
};

#define CPVK_MAX_DISPATCHES 64

struct cpvk_dispatch {
   struct cpvk_pipeline *pipeline;
   uint32_t grid[3];
   CUdeviceptr addrs[16];
};

struct cpvk_cmd_buffer {
   struct vk_command_buffer vk;
   struct cpvk_pipeline *pipeline;
   CUdeviceptr addrs[16];
   struct cpvk_dispatch dispatches[CPVK_MAX_DISPATCHES];
   unsigned num_dispatches;
};

extern const struct vk_command_buffer_ops cpvk_cmd_buffer_ops;
extern const struct vk_sync_type *const cpvk_sync_types[];

VkResult cpvk_execute_cmd_buffer(struct cpvk_device *dev,
                                 struct cpvk_cmd_buffer *cmd);

struct cpvk_pipeline {
   struct vk_object_base base;
   VkPipelineBindPoint bind_point;
   struct cp_shader_binary *bin;   /* the compiled CUDA kernel */
   /* The block size. Under Gallium this arrived per dispatch from lavapipe,
    * which had read it out of the shader; natively the pipeline is where it
    * belongs, since it is fixed at compile time for anything but
    * VK_EXT_subgroup_size_control. */
   uint16_t local_size[3];
};

struct cpvk_buffer {
   struct vk_buffer vk;
   struct cpvk_device_memory *mem;
   VkDeviceSize offset;
};

#define CPVK_MAX_MIP_LEVELS 16

struct cpvk_image {
   struct vk_image vk;
   struct cpvk_device_memory *mem;
   VkDeviceSize offset;
   /* Every level at its own offset. Folding that into one base is the bug
    * that wrote level 0 over and over while the rest stayed untouched, and
    * stayed invisible until textureLod was honoured. */
   uint64_t level_offset[CPVK_MAX_MIP_LEVELS];
   uint64_t level_size[CPVK_MAX_MIP_LEVELS];
   uint32_t row_stride[CPVK_MAX_MIP_LEVELS];
   uint64_t size;
   uint32_t texel;                 /* enum cp_texel_format */
   int color;                      /* enum cp_color_encoding, -1 if none */
};

struct cpvk_image_view {
   struct vk_image_view vk;
   struct cpvk_image *image;
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
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_pipeline, base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_set_layout, vk.base,
                               VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_pipeline_layout, vk.base, VkPipelineLayout,
                               VK_OBJECT_TYPE_PIPELINE_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_DEFINE_HANDLE_CASTS(cpvk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_image, vk.base, VkImage,
                               VK_OBJECT_TYPE_IMAGE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)

#endif /* CPVK_PRIVATE_H */
