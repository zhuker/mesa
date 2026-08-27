/*
 * Instance, physical device and device bring-up for the native driver.
 * Milestone 1: a CUDA device is enumerated as a Vulkan physical device.
 */

#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "util/build_id.h"
#include "util/disk_cache.h"
#include "util/mesa-blake3.h"
#include "util/u_debug.h"

/*
 * KHR_get_physical_device_properties2 was advertised once, removed after
 * vulkaninfo segfaulted, and is back. The segfault was real; the reason
 * recorded for it was wrong, and it cost this extension a long exile.
 *
 * The old comment blamed unwired KHR aliases at apiVersion 1.0. They were
 * never unwired. vk_dispatch_table.h stores each core entrypoint and its KHR
 * alias in a union, and physical_device_compaction_table[] maps both
 * entrypoint indices onto that one shared dispatch slot, so filling the core
 * slot resolves both names. The extension check in
 * vk_physical_device_entrypoint_is_enabled is only a gate: passing it hands
 * back the core implementation.
 *
 * The actual crash was a missing *core* entrypoint,
 * cpvk_GetPhysicalDeviceMemoryProperties2 -- see the comment on it below. It
 * segfaulted at apiVersion 1.1 with no extension requested at all; enabling
 * the extension merely widened the blast radius to more callers. Implementing
 * it fixed both.
 *
 * Do NOT add KHR alias forwarders here to "help". Because core and alias share
 * one dispatch slot, a driver that defines both trips
 * assert(disp[disp_index] == NULL) in
 * vk_physical_device_dispatch_table_from_entrypoints, which is a hard abort in
 * this build. Fourteen Mesa drivers advertise this extension and not one of
 * them defines an alias forwarder; lavapipe is the model.
 */
static const struct vk_instance_extension_table cpvk_instance_extensions = {
   /* Mesa WSI common supplies a real headless surface implementation. No X11,
    * Wayland or display platform extension is exposed by this build. */
   .KHR_surface = true,
   .EXT_headless_surface = true,
   .KHR_get_physical_device_properties2 = true,
};

static const struct vk_device_extension_table cpvk_device_extensions = {
   /* Filled in as features land. */

   /* Dynamic rendering, because there is no other kind here: this driver has
    * no tiler to hand a render pass to, and the runtime only builds the
    * vkCmdBeginRendering entrypoint when the extension is advertised. Without
    * it vkGetDeviceProcAddr returns NULL and the caller jumps to zero, which
    * is what it did. */
   .KHR_dynamic_rendering = true,

   /* Real CPU-visible headless swapchains through Mesa WSI common. */
   .KHR_swapchain = true,

   /*
    * Export a VkDeviceMemory as a file descriptor a CUDA process can import.
    * Only the _fd half is named: VK_KHR_external_memory and
    * VK_KHR_external_memory_capabilities are promoted to Vulkan 1.1, which
    * this driver already advertises, so their structs and the capability
    * query are core here and listing them would claim nothing new. The
    * transport is the part that is platform-specific and therefore still an
    * extension. See docs/cudavk/CUDA_INTEROP.md.
    */
   .KHR_external_memory_fd = true,

   /*
    * Negative viewport height, which this driver already does: the viewport
    * is resolved into a scale and a translate, and a negative height makes
    * the y scale negative, which is the flip. Advertised because the sample
    * that uses it is the test of it -- not ahead of it.
    */
   .KHR_maintenance1 = true,

   /*
    * Promoted to Vulkan 1.1, which this driver advertises, so the entrypoints
    * behind these two already exist and are already what the runtime
    * dispatches: cpvk_BindBufferMemory2 and cpvk_GetBufferMemoryRequirements2
    * in cpvk_device_memory.c, cpvk_BindImageMemory2 and
    * cpvk_GetImageMemoryRequirements2 in cpvk_image.c. Only the extension
    * names were missing, and a name is what vkCreateDevice checks.
    *
    * That mattered. The headless streamer capture this driver is measured
    * against enables seven device extensions, and these were two of the six
    * that went unadvertised, so vkCreateDevice returned
    * VK_ERROR_EXTENSION_NOT_PRESENT and the replay ran at all only because
    * gfxrecon's --remove-unsupported stripped them first. A client that
    * enables them unconditionally could not create a device here.
    *
    * As with KHR_get_physical_device_properties2 above, do not add KHR alias
    * forwarders: core and alias share one dispatch slot and defining both
    * trips an assert in the runtime.
    */
   .KHR_bind_memory2 = true,
   .KHR_get_memory_requirements2 = true,

   /* The capture uses the KHR aliases even though these commands are core in
    * 1.1. The implementation below is shared with the core entrypoints. */
   .KHR_descriptor_update_template = true,

   /*
    * KHR_dynamic_rendering's dependency chain, which an application must be
    * able to enable alongside it: without these an otherwise legal
    * vkCreateDevice naming dynamic rendering is invalid usage. Render passes
    * are the runtime's common implementation over this driver's dynamic
    * rendering, and sample-zero depth/stencil resolve is implemented as the
    * first sample plane's copy.
    */
   .KHR_create_renderpass2 = true,
   .KHR_depth_stencil_resolve = true,

   /*
    * Timeline semaphores, implemented as a 64-bit counter on the driver's one
    * vk_sync type (cpvk_sync.c); the runtime's vk_semaphore.c supplies
    * vkWaitSemaphores, vkSignalSemaphore and vkGetSemaphoreCounterValue on top
    * of it, so there is no entrypoint of this here.
    *
    * The extension form, not core: apiVersion stays 1.1, which is what the
    * driver implements, and promoting it to 1.2 would claim everything else in
    * that version too. Note that this is the whole of what is possible here --
    * an exportable timeline is not. CUDA 12.8 has no call that creates or
    * exports a semaphore, so VK_KHR_external_semaphore_fd stays unadvertised;
    * see docs/cudavk/CUDA_INTEROP.md.
    */
   .KHR_timeline_semaphore = true,
};

static void
cpvk_get_features(struct vk_features *features)
{
   *features = (struct vk_features) {
      /* Vulkan 1.0 */
      .fullDrawIndexUint32 = true,
      .independentBlend = false,
      .fragmentStoresAndAtomics = true,
      /*
       * Dynamically indexed descriptor arrays. The lowering resolves a
       * non-constant array index through vulkan_resource_reindex and the
       * kernels address the element at run time, which `cpvk_desc_array`
       * checks against lavapipe and NVIDIA across two sets and two
       * dispatches; `cpvk_sampler_array` does the same for a combined image
       * sampler array indexed by a push constant. These were false while the
       * implementation worked, which is the mirror image of the usual mistake
       * and costs an application that checks features before using them.
       *
       * Storage *image* array dynamic indexing is deliberately still false:
       * the same lowering would serve it, but nothing here tests it, and an
       * untested feature bit is the thing this pass exists to remove.
       */
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .vertexPipelineStoresAndAtomics = true,
      .shaderClipDistance = false,
      .samplerAnisotropy = true,

      /* KHR_dynamic_rendering is the one advertised post-1.1 feature. Keep
       * newer feature structs zero unless the matching extension and behavior
       * are both implemented; the old Vulkan-1.3 experiment overclaimed all
       * of these merely because that core version made them mandatory. */
      .dynamicRendering = true,

      /* Vulkan 1.2 / KHR_timeline_semaphore. The counter, the monotonic
       * signal and the host wait are the vk_sync type's; wait-before-signal
       * is the runtime's ASSISTED timeline mode over this driver's
       * VK_SYNC_WAIT_PENDING. See cpvk_sync.c. */
      .timelineSemaphore = true,
   };
}

/*
 * The three UUIDs, which were all zero until CUDA interop needed them.
 *
 * deviceUUID is the load-bearing one: a CUDA consumer that receives an
 * exported handle has to decide which of its CUDA devices the memory belongs
 * to, and the documented way is to match VkPhysicalDeviceIDProperties against
 * cuDeviceGetUuid. Against the proprietary driver that comparison is exact,
 * which is what makes it usable as an identity test rather than a hint. Note
 * that this is cuDeviceGetUuid and not cuDeviceGetUuid_v2: the plain form is
 * the one measured to match, and _v2 only differs under MIG, which this driver
 * does not enumerate.
 *
 * driverUUID identifies the driver build, so two processes can agree that they
 * are talking to the same implementation before sharing anything.
 *
 * pipelineCacheUUID decides when a serialized pipeline cache is stale. Leaving
 * it zero meant a cache written by one build was accepted by the next one.
 */
static VkResult
cpvk_init_uuids(struct cpvk_physical_device *pdev,
                struct cpvk_instance *instance)
{
   CUuuid cu_uuid;
   if (cuDeviceGetUuid(&cu_uuid, pdev->cu_dev) != CUDA_SUCCESS)
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                       "cuDeviceGetUuid failed");
   STATIC_ASSERT(sizeof(cu_uuid.bytes) == VK_UUID_SIZE);
   memcpy(pdev->device_uuid, cu_uuid.bytes, VK_UUID_SIZE);

   const struct build_id_note *note =
      build_id_find_nhdr_for_addr(cpvk_init_uuids);
   if (!note)
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                       "failed to find build-id");
   unsigned build_id_len = build_id_length(note);
   if (build_id_len < BUILD_ID_EXPECTED_HASH_LENGTH)
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                       "build-id too short; it needs to be a SHA");

   memcpy(pdev->driver_uuid, build_id_data(note), VK_UUID_SIZE);

   /* Build plus device: a cache is invalid if either changes. The compute
    * capability is in it because the kernels are compiled for it. */
   blake3_hasher blake3_ctx;
   uint8_t blake3[BLAKE3_OUT_LEN];
   STATIC_ASSERT(VK_UUID_SIZE <= sizeof(blake3));
   _mesa_blake3_init(&blake3_ctx);
   _mesa_blake3_update(&blake3_ctx, build_id_data(note), build_id_len);
   _mesa_blake3_update(&blake3_ctx, pdev->device_uuid, VK_UUID_SIZE);
   _mesa_blake3_update(&blake3_ctx, &pdev->sm_major, sizeof(pdev->sm_major));
   _mesa_blake3_update(&blake3_ctx, &pdev->sm_minor, sizeof(pdev->sm_minor));
   _mesa_blake3_final(&blake3_ctx, blake3);
   memcpy(pdev->cache_uuid, blake3, VK_UUID_SIZE);

   return VK_SUCCESS;
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
      .apiVersion = VK_MAKE_VERSION(1, 1, VK_HEADER_VERSION),
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
      .maxTexelBufferElements = 65536,
      .maxUniformBufferRange = 1u << 27,
      .maxStorageBufferRange = 1u << 30,
      .bufferImageGranularity = 1,
      .maxBoundDescriptorSets = 8,
      .maxColorAttachments = 1,
      .maxPerStageDescriptorSamplers = 32,
      .maxPerStageDescriptorSampledImages = 32,
      .maxPerStageDescriptorUniformBuffers = 16,
      .maxPerStageDescriptorStorageBuffers = 16,
      .maxPerStageDescriptorStorageImages = 16,
      .maxPerStageResources = 128,
      /*
       * The Vulkan 1.1 per-set minima. A set's storage is sized by its
       * layout rather than by a constant, so these are what the descriptor
       * code can hold rather than a number chosen to look large; a legal set
       * is either stored completely or refused at layout creation.
       */
      .maxDescriptorSetSamplers = 96,
      .maxDescriptorSetSampledImages = 96,
      .maxDescriptorSetUniformBuffers = 72,
      .maxDescriptorSetUniformBuffersDynamic = 8,
      .maxDescriptorSetStorageBuffers = 24,
      .maxDescriptorSetStorageBuffersDynamic = 4,
      .maxDescriptorSetStorageImages = 24,
      /* Input attachments are not implemented, and a nonzero limit here
       * would be the promise that they are. */
      .maxDescriptorSetInputAttachments = 0,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 4000,
      .maxMemoryAllocationSize = 1u << 31,
      .maxPushConstantsSize = 256,
      .maxComputeWorkGroupInvocations = 1024,
      .maxComputeWorkGroupSize = { 1024, 1024, 64 },
      .maxComputeWorkGroupCount = { 65535, 65535, 65535 },
      .maxComputeSharedMemorySize = 48 * 1024,
      .maxVertexInputAttributes = 32,
      .maxVertexInputBindings = 32,
      .maxVertexInputAttributeOffset = 2047,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 64,
      .maxFragmentInputComponents = 64,
      .maxFragmentOutputAttachments = 1,
      .maxFragmentDualSrcAttachments = 0,
      .maxFragmentCombinedOutputResources = 17,
      .maxViewports = 1,
      .maxViewportDimensions = { 16384, 16384 },
      .viewportBoundsRange = { -32768.0f, 32768.0f },
      .viewportSubPixelBits = 8,
      .maxFramebufferWidth = 16384,
      .maxFramebufferHeight = 16384,
      .maxFramebufferLayers = 1,
      /*
       * 1, 4 and 8, which is what the Gallium driver this replaces reports and
       * what the rasteriser's sample-position table actually holds -- it has
       * entries for one, four and eight.
       *
       * Advertising 2 and withholding 8 made `multisampling` pick 4x where the
       * reference picks 8x, since the sample asks for the highest count the
       * device offers. The two images then differ on every silhouette by the
       * difference between four samples and eight, which is what its 0.553
       * was.
       */
      .framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT |
                                      VK_SAMPLE_COUNT_4_BIT |
                                      VK_SAMPLE_COUNT_8_BIT,
      .framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT |
                                      VK_SAMPLE_COUNT_4_BIT |
                                      VK_SAMPLE_COUNT_8_BIT,
      .framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT |
                                        VK_SAMPLE_COUNT_4_BIT |
                                        VK_SAMPLE_COUNT_8_BIT,
      .framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSampleMaskWords = 1,
      .maxSamplerLodBias = 16.0f,
      .maxSamplerAnisotropy = 16.0f,
      .minMemoryMapAlignment = 4096,
      .minTexelBufferOffsetAlignment = 16,
      .minUniformBufferOffsetAlignment = 256,
      .minStorageBufferOffsetAlignment = 256,
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT32_MAX,
      .timestampComputeAndGraphics = false,
      .timestampPeriod = 1.0f,
      .maxClipDistances = 0,
      .maxCullDistances = 0,
      .maxCombinedClipAndCullDistances = 0,
      .discreteQueuePriorities = 2,
      .pointSizeRange = { 1.0f, 256.0f },
      .lineWidthRange = { 1.0f, 1.0f },
      .pointSizeGranularity = 0.125f,
      .lineWidthGranularity = 0.0f,
      /* Sample zero only: it is a plane copy and needs no averaging kernel.
       * Depth and stencil must use the same mode, which the packed formats
       * make automatic. */
      .supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .independentResolveNone = false,
      .independentResolve = false,
      .strictLines = true,
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .nonCoherentAtomSize = 256,
   };

   snprintf(props->deviceName, sizeof(props->deviceName), "cudavk (%s)",
            pdev->name);
   memcpy(props->pipelineCacheUUID, pdev->cache_uuid, VK_UUID_SIZE);
   memcpy(props->deviceUUID, pdev->device_uuid, VK_UUID_SIZE);
   memcpy(props->driverUUID, pdev->driver_uuid, VK_UUID_SIZE);
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
cpvk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *name)
{
   VK_FROM_HANDLE(cpvk_physical_device, pdev, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(pdev->vk.instance, name);
}

static void
cpvk_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct cpvk_physical_device *pdev =
      container_of(vk_pdev, struct cpvk_physical_device, vk);

   if (pdev->wsi_initialized) {
      pdev->vk.wsi_device = NULL;
      wsi_device_finish(&pdev->wsi_device, &pdev->vk.instance->alloc);
      pdev->wsi_initialized = false;
   }
   if (pdev->vk.disk_cache) {
      disk_cache_destroy(pdev->vk.disk_cache);
      pdev->vk.disk_cache = NULL;
   }
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

   VkResult result = cpvk_init_uuids(pdev, instance);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }

   struct vk_features features;
   struct vk_properties props;
   cpvk_get_features(&features);
   cpvk_get_properties(pdev, &props);

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &cpvk_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_physical_device_entrypoints, false);
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_physical_device_entrypoints, false);

   result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                    &cpvk_device_extensions, &features, &props,
                                    &dispatch_table);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }

   result = wsi_device_init(&pdev->wsi_device,
                            cpvk_physical_device_to_handle(pdev),
                            cpvk_wsi_proc_addr, &instance->vk.alloc,
                            -1, NULL,
                            &(struct wsi_device_options){ .sw_device = true });
   if (result != VK_SUCCESS) {
      vk_physical_device_finish(&pdev->vk);
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }
   pdev->wsi_initialized = true;
   pdev->wsi_device.wants_linear = true;
   pdev->vk.wsi_device = &pdev->wsi_device;

   pdev->vk.supported_sync_types = cpvk_sync_types;
   pdev->vk.disk_cache = disk_cache_create(
      pdev->name, "cudavk-nvrtc-v1", 0);

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
      &dispatch_table, &wsi_instance_entrypoints, false);
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
   *pApiVersion = VK_MAKE_VERSION(1, 1, VK_HEADER_VERSION);
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
    * Both host-visible types use managed memory, which is directly host
    * mapped and can migrate to the GPU.  Type 2 additionally advertises that
    * device locality; type 1 exists so captured non-device-local requirements
    * still have an exact property match.
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

/*
 * Mesa's convention runs the other way from the rest of the 1.0 entrypoints in
 * this file: the driver implements the *2 form and vk_common builds the 1.0
 * call on top of it (vk_physical_device.c:171-186). There is no
 * vk_common_GetPhysicalDeviceMemoryProperties2, so defining only the 1.0 form
 * left the shared dispatch slot NULL and the core 1.1 call jumped through it.
 * That was a live segfault at apiVersion 1.1 with no extension requested.
 *
 * The pNext chain is ignored on purpose: VK_EXT_memory_budget is not
 * advertised, so no chained struct is legal here yet.
 */
VKAPI_ATTR void VKAPI_CALL
cpvk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   cpvk_GetPhysicalDeviceMemoryProperties(physicalDevice,
                                          &pMemoryProperties->memoryProperties);
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
   /*
    * OPAQUE_FD only, and only for buffers. The fd is a CUDA VMM shareable
    * handle (cpvk_allocate_exportable), so it is opaque in the strict sense:
    * it is not a dma-buf, it cannot be mmap'd or sized from outside, and only
    * a CUDA importer on the same device understands it. That is exactly what
    * OPAQUE_FD promises, and dma-buf is not offered because this hardware
    * reports DMA_BUF_SUPPORTED=0 (docs/cudavk/notes/CUDA13_UPGRADE.md).
    *
    * DEDICATED_ONLY is deliberately not set: one VkDeviceMemory is one CUDA
    * allocation here and several buffers may be bound into it at offsets,
    * which is what the interop sample does with its frame and result buffers.
    *
    * No usage is refused. Every buffer in this driver is the same flat device
    * memory whatever it is used for, so there is no usage this could export
    * and that one it could not.
    */
   if (pExternalBufferInfo->handleType ==
       VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
      pExternalBufferProperties->externalMemoryProperties =
         (VkExternalMemoryProperties) {
            .externalMemoryFeatures =
               VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
            .exportFromImportedHandleTypes =
               VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            .compatibleHandleTypes =
               VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
         };
      return;
   }

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
