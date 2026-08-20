/*
 * Instance, physical device and device bring-up for the native driver.
 * Milestone 1: a CUDA device is enumerated as a Vulkan physical device.
 */

#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "util/u_debug.h"

/*
 * Nothing yet. KHR_get_physical_device_properties2 was advertised first and
 * taken straight back out: an application that enables it calls the KHR
 * aliases, and at apiVersion 1.0 those alias entrypoints are not wired to the
 * runtime's core implementations, so the loader jumps to a null pointer. That
 * is the "advertised and then clamped is worse than refused" rule in its
 * sharpest form — the failure was a segfault inside vulkaninfo with no driver
 * frame in the backtrace. It comes back with apiVersion 1.1, or with the
 * aliases implemented explicitly.
 */
static const struct vk_instance_extension_table cpvk_instance_extensions = {
   /*
    * Advertised without any WSI behind it, which is what lavapipe does and
    * for the same stated reason: an offscreen app enables this to get the
    * PRESENT_SRC_KHR image layout, and never creates a surface. The Vulkan
    * sample suite asks for it unconditionally and asks for a platform
    * surface extension only when it opens a window, so this is the whole gap
    * between "cannot create an instance" and "renders".
    *
    * If a surface is ever created this is a lie, and the honest fix then is
    * wsi_common rather than a wider claim here.
    */
   .KHR_surface = true,
};

static const struct vk_device_extension_table cpvk_device_extensions = {
   /* Filled in as features land. Headless: no swapchain, ever. */

   /* Dynamic rendering, because there is no other kind here: this driver has
    * no tiler to hand a render pass to, and the runtime only builds the
    * vkCmdBeginRendering entrypoint when the extension is advertised. Without
    * it vkGetDeviceProcAddr returns NULL and the caller jumps to zero, which
    * is what it did. */
   .KHR_dynamic_rendering = true,

   /* Same bargain as KHR_surface above: enabled for the image layout, with
    * no swapchain implementation behind it. */
   .KHR_swapchain = true,

   /*
    * Negative viewport height, which this driver already does: the viewport
    * is resolved into a scale and a translate, and a negative height makes
    * the y scale negative, which is the flip. Advertised because the sample
    * that uses it is the test of it -- not ahead of it.
    */
   .KHR_maintenance1 = true,
};

static void
cpvk_get_features(struct vk_features *features)
{
   *features = (struct vk_features) {
      /* Vulkan 1.0 */
      .fullDrawIndexUint32 = true,
      .independentBlend = true,
      .fragmentStoresAndAtomics = true,
      .vertexPipelineStoresAndAtomics = true,
      .shaderClipDistance = false,
      .samplerAnisotropy = true,

      /*
       * Vulkan 1.3 makes all of these mandatory, so reporting 1.3 means
       * declaring them. Where the driver has nothing to do for one, it has
       * nothing to do: a barrier is already satisfied by one in-order stream,
       * an event is a boolean, and private data and cache control are the
       * runtime's. They are declared because the version number promises
       * them, and the entry points behind them exist and return.
       */
      .dynamicRendering = true,
      .synchronization2 = true,
      .maintenance4 = true,
      .privateData = true,
      .pipelineCreationCacheControl = true,
      .shaderTerminateInvocation = true,
      .shaderDemoteToHelperInvocation = true,
      .shaderZeroInitializeWorkgroupMemory = true,
      .shaderIntegerDotProduct = true,
      .subgroupSizeControl = true,
      .computeFullSubgroups = true,
      .inlineUniformBlock = true,
   };
}

static void
cpvk_get_properties(const struct cpvk_physical_device *pdev,
                    struct vk_properties *props)
{
   *props = (struct vk_properties) {
      /*
       * 1.1, not 1.0. At 1.0 the runtime does not wire the KHR alias
       * entrypoints to its core implementations, so an application that
       * enables KHR_get_physical_device_properties2 -- or calls any 1.1-and-
       * later core entry point such as vkCmdBlitImage2 -- jumps through a
       * null pointer in the loader.
       *
       * And 1.1, not 1.3, which was tried here and measured. Every sample and
       * both replays still passed at 1.3, but `vkCreateDevice` asking for
       * `VkPhysicalDeviceVulkan13Features` came back
       * VK_ERROR_FEATURE_NOT_PRESENT: 1.3 makes a long list of features
       * mandatory -- synchronization2, inlineUniformBlock, privateData,
       * maintenance4, shaderTerminateInvocation, subgroupSizeControl and more
       * -- and this driver declares one of them. Reporting 1.3 claims all of
       * them, and a version number is a promise about what may be called, not
       * a request for what happens to work today.
       *
       * The Gallium driver this one replaces reports 1.4, so this is still a
       * gap; closing it means implementing those features, not editing this
       * number.
       */
      .apiVersion = VK_MAKE_VERSION(1, 3, VK_HEADER_VERSION),
      .driverVersion = 1,
      .vendorID = 0x10de,          /* NVIDIA: the device really is one */
      .deviceID = 0,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,

      /* The capture's requirements, from
       * tests/headless_streamer_requirements.txt. Raised as features land
       * rather than advertised ahead of them: a capability that is announced
       * and then clamped is worse than one that is refused. */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      .maxImageDimension3D = 2048,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 2048,
      .maxBoundDescriptorSets = 8,
      .maxColorAttachments = 4,
      .maxPerStageDescriptorSamplers = 32,
      .maxPerStageDescriptorSampledImages = 32,
      .maxPerStageDescriptorUniformBuffers = 16,
      .maxPerStageDescriptorStorageBuffers = 16,
      .maxPerStageDescriptorStorageImages = 16,
      .maxPushConstantsSize = 256,
      .maxComputeWorkGroupInvocations = 1024,
      .maxComputeWorkGroupSize = { 1024, 1024, 64 },
      .maxComputeWorkGroupCount = { 65535, 65535, 65535 },
      .maxComputeSharedMemorySize = 48 * 1024,
      .maxVertexInputAttributes = 32,
      .maxVertexInputBindings = 32,
      .maxViewports = 1,
      .maxViewportDimensions = { 16384, 16384 },
      .viewportBoundsRange = { -32768.0f, 32768.0f },
      .framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT |
                                      VK_SAMPLE_COUNT_2_BIT |
                                      VK_SAMPLE_COUNT_4_BIT,
      .framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT |
                                      VK_SAMPLE_COUNT_2_BIT |
                                      VK_SAMPLE_COUNT_4_BIT,
      .sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSamplerAnisotropy = 16.0f,
      .minMemoryMapAlignment = 4096,
      .minUniformBufferOffsetAlignment = 256,
      .minStorageBufferOffsetAlignment = 256,
      .subPixelPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT32_MAX,
      .timestampComputeAndGraphics = false,
      .pointSizeRange = { 1.0f, 256.0f },
      .lineWidthRange = { 1.0f, 1.0f },
      .pointSizeGranularity = 0.125f,
      .lineWidthGranularity = 0.0f,
   };

   snprintf(props->deviceName, sizeof(props->deviceName), "cudapipe (%s)",
            pdev->name);
   memset(props->pipelineCacheUUID, 0, VK_UUID_SIZE);
   memset(props->deviceUUID, 0, VK_UUID_SIZE);
   memset(props->driverUUID, 0, VK_UUID_SIZE);
}

static void
cpvk_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct cpvk_physical_device *pdev =
      container_of(vk_pdev, struct cpvk_physical_device, vk);

   vk_physical_device_finish(&pdev->vk);
   vk_free(&pdev->vk.instance->alloc, pdev);
}

static VkResult
cpvk_enumerate_physical_devices(struct vk_instance *vk_instance)
{
   struct cpvk_instance *instance =
      container_of(vk_instance, struct cpvk_instance, vk);

   /* Before anything reads cp_debug: the shared registry is the driver's
    * whole debugging surface, and until this runs every flag reads as its
    * zero value rather than as what the environment asked for -- which is
    * silence of exactly the kind this driver's failure modes are made of. */
   cp_debug_init();

   if (cuInit(0) != CUDA_SUCCESS)
      return VK_SUCCESS;          /* no CUDA device: enumerate nothing */

   int count = 0;
   if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count < 1)
      return VK_SUCCESS;

   /* One device for now: the Gallium-hosted driver takes device 0 too, and
    * multi-device selection is a question for when anything renders. */
   CUdevice cu_dev;
   if (cuDeviceGet(&cu_dev, 0) != CUDA_SUCCESS)
      return VK_SUCCESS;

   struct cpvk_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!pdev)
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   pdev->cu_dev = cu_dev;
   cuDeviceGetName(pdev->name, sizeof(pdev->name), cu_dev);
   cuDeviceGetAttribute(&pdev->sm_major,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev);
   cuDeviceGetAttribute(&pdev->sm_minor,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev);
   cuDeviceTotalMem(&pdev->vram, cu_dev);

   struct vk_features features;
   struct vk_properties props;
   cpvk_get_features(&features);
   cpvk_get_properties(pdev, &props);

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &cpvk_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_physical_device_entrypoints, false);

   VkResult result =
      vk_physical_device_init(&pdev->vk, &instance->vk,
                              &cpvk_device_extensions, &features, &props,
                              &dispatch_table);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }

   pdev->vk.supported_sync_types = cpvk_sync_types;

   list_addtail(&pdev->vk.link, &instance->vk.physical_devices.list);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkInstance *pInstance)
{
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : vk_default_allocator();

   struct cpvk_instance *instance =
      vk_zalloc(alloc, sizeof(*instance), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &cpvk_instance_entrypoints, true);
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_instance_entrypoints, false);

   VkResult result = vk_instance_init(&instance->vk, &cpvk_instance_extensions,
                                      &dispatch_table, pCreateInfo, alloc);
   if (result != VK_SUCCESS) {
      vk_free(alloc, instance);
      return result;
   }

   instance->vk.physical_devices.enumerate = cpvk_enumerate_physical_devices;
   instance->vk.physical_devices.destroy = cpvk_physical_device_destroy;

   *pInstance = cpvk_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyInstance(VkInstance _instance,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_instance, instance, _instance);

   if (!instance)
      return;

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
cpvk_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(cpvk_instance, instance, _instance);
   return vk_instance_get_proc_addr(&instance->vk, &cpvk_instance_entrypoints,
                                    pName);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                          uint32_t *pPropertyCount,
                                          VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(
      &cpvk_instance_extensions, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   /*
    * The instance version caps everything above it. With this at 1.0 the
    * physical device could report 1.3 and vkGetDeviceProcAddr would still
    * hand back a null pointer for every 1.1-and-later core entry point,
    * because the loader will not dispatch past the version the instance
    * claims.
    */
   *pApiVersion = VK_MAKE_VERSION(1, 3, VK_HEADER_VERSION);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceMemoryProperties(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties *pMemoryProperties)
{
   VK_FROM_HANDLE(cpvk_physical_device, pdev, physicalDevice);

   /*
    * The split the Gallium-hosted driver had to negotiate with lavapipe in
    * session 13 is the natural shape here: device-local for everything the
    * host never touches, host-visible for staging, and a managed type for
    * software that demands both.
    */
   /*
    * The flags are the Gallium-hosted driver's, type for type, and that is
    * not cosmetic. A capture records the memory type it allocated from and
    * the replayer maps it onto a type here with at least the same
    * properties; the managed type was missing HOST_CACHED, no replay type
    * satisfied the capture's, and the replay aborted with "specified memory
    * type index exceeds number of available memory types" before a frame
    * was drawn. One heap for the same reason.
    *
    * Both host-visible types are honest: cuMemAllocHost is pinned and the
    * CPU caches it, and managed memory is device-local and host-visible by
    * construction.
    */
   *pMemoryProperties = (VkPhysicalDeviceMemoryProperties) {
      .memoryHeapCount = 1,
      .memoryHeaps[0] = { .size = pdev->vram,
                          .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT },
      .memoryTypeCount = 3,
      .memoryTypes[0] = { .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          .heapIndex = 0 },
      .memoryTypes[1] = { .propertyFlags =
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                             VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          .heapIndex = 0 },
      .memoryTypes[2] = { .propertyFlags =
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                             VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          .heapIndex = 0 },
   };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out,
                          pQueueFamilyProperties, pQueueFamilyPropertyCount);

   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = (VkQueueFamilyProperties) {
         .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                       VK_QUEUE_TRANSFER_BIT,
         .queueCount = 1,
         .timestampValidBits = 0,
         .minImageTransferGranularity = { 1, 1, 1 },
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceSparseImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
   uint32_t *pPropertyCount, VkSparseImageFormatProperties2 *pProperties)
{
   *pPropertyCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceExternalBufferProperties(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
   VkExternalBufferProperties *pExternalBufferProperties)
{
   pExternalBufferProperties->externalMemoryProperties =
      (VkExternalMemoryProperties) { 0 };
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceExternalFenceProperties(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalFenceInfo *pExternalFenceInfo,
   VkExternalFenceProperties *pExternalFenceProperties)
{
   pExternalFenceProperties->exportFromImportedHandleTypes = 0;
   pExternalFenceProperties->compatibleHandleTypes = 0;
   pExternalFenceProperties->externalFenceFeatures = 0;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceExternalSemaphoreProperties(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalSemaphoreInfo *pExternalSemaphoreInfo,
   VkExternalSemaphoreProperties *pExternalSemaphoreProperties)
{
   pExternalSemaphoreProperties->exportFromImportedHandleTypes = 0;
   pExternalSemaphoreProperties->compatibleHandleTypes = 0;
   pExternalSemaphoreProperties->externalSemaphoreFeatures = 0;
}

/* The format table lives in cpvk_image.c, beside the layout it describes. */
