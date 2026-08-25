/*
 * Descriptor arrays and two descriptor sets, resolved per element.
 *
 * One compute pipeline, one pipeline layout, two sets:
 *
 *    set 0 binding 0   uniform buffer   [3]   three separate VkBuffers
 *    set 0 binding 7   storage buffer   [1]   the output
 *    set 1 binding 2   storage buffer   [2]   two separate VkBuffers
 *
 * Every one of those five source buffers holds one distinct uint, so the
 * question the test asks is whether a descriptor array element reaches its own
 * buffer or collapses onto element zero. It is asked three ways over the same
 * arrays:
 *
 *    constant index        src[0] src[1] src[2] aux[0] aux[1]
 *    index read from memory   src[m] aux[m], m loaded out of the output SSBO
 *    index from a push constant, into an output slot that is also pushed
 *
 * The last two are what force the array step through NIR's
 * vulkan_resource_reindex rather than folding it into the binding: neither
 * index is a compile-time constant, and the pushed one is not even fixed for
 * the pipeline, because the same command buffer dispatches twice with
 * different push constants and the two dispatches must land different values
 * in different slots.
 *
 * The output SSBO is 32 uints, pre-filled with 0xdeadbeef and with the memory
 * index seeded in its last slot. Slots the shader never writes are checked to
 * be untouched, so a driver that writes through the wrong descriptor is caught
 * whether it reads the wrong buffer or writes the wrong one.
 *
 * Nothing here renders. It is a compute dispatch, host-visible buffers and a
 * readback, so it runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. SPIR-V is embedded; nothing is read from disk.
 *
 * Dynamic indexing of a descriptor array is gated by the Vulkan 1.0 features
 * shaderUniformBufferArrayDynamicIndexing and
 * shaderStorageBufferArrayDynamicIndexing, so both are queried and enabled
 * when the device has them; a device that does not is named in the output.
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- da.comp --
 *   #version 450
 *   layout(local_size_x = 1) in;
 *   layout(set = 0, binding = 0, std140) uniform Src { uint value; } src[3];
 *   layout(set = 0, binding = 7, std430) buffer Out { uint slot[]; } dst;
 *   layout(set = 1, binding = 2, std430) buffer Aux { uint value; } aux[2];
 *   layout(push_constant) uniform Push {
 *      uint src_index;
 *      uint aux_index;
 *      uint out_index;
 *   } pc;
 *   void main()
 *   {
 *      dst.slot[0] = src[0].value;
 *      dst.slot[1] = src[1].value;
 *      dst.slot[2] = src[2].value;
 *      dst.slot[3] = aux[0].value;
 *      dst.slot[4] = aux[1].value;
 *      uint m = dst.slot[31];
 *      dst.slot[5] = src[m].value;
 *      dst.slot[6] = aux[m].value;
 *      dst.slot[pc.out_index]     = src[pc.src_index].value;
 *      dst.slot[pc.out_index + 1] = aux[pc.aux_index].value;
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* The output SSBO, in uints. */
#define SLOTS 32
/* The shader loads its memory index out of the last slot. */
#define MEM_SLOT (SLOTS - 1)
#define MEM_INDEX 1u
/* Anything the shader does not write must still read back as this. */
#define SENTINEL 0xdeadbeefu

/* One distinct value per source buffer, so an element that resolved to the
 * wrong buffer cannot look like an element that resolved to the right one. */
#define SRC_COUNT 3
#define AUX_COUNT 2
#define SRC_VALUE(i) (0x1000u + (unsigned)(i))
#define AUX_VALUE(i) (0x2000u + (unsigned)(i))

/* Two dispatches of the one pipeline, differing only in push constants. */
struct push {
   uint32_t src_index;
   uint32_t aux_index;
   uint32_t out_index;
};
static const struct push pass_a = { 2, 1, 8 };
static const struct push pass_b = { 0, 0, 16 };

/* da.comp */
static const uint32_t comp_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000050u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0005000fu, 0x00000005u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00060010u, 0x00000004u, 0x00000011u,
   0x00000001u, 0x00000001u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u,
   0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000008u,
   0x0074754fu, 0x00050006u, 0x00000008u, 0x00000000u, 0x746f6c73u, 0x00000000u,
   0x00030005u, 0x0000000au, 0x00747364u, 0x00030005u, 0x0000000du, 0x00637253u,
   0x00050006u, 0x0000000du, 0x00000000u, 0x756c6176u, 0x00000065u, 0x00030005u,
   0x00000011u, 0x00637273u, 0x00030005u, 0x0000001fu, 0x00787541u, 0x00050006u,
   0x0000001fu, 0x00000000u, 0x756c6176u, 0x00000065u, 0x00030005u, 0x00000023u,
   0x00787561u, 0x00030005u, 0x0000002cu, 0x0000006du, 0x00040005u, 0x0000003au,
   0x68737550u, 0x00000000u, 0x00060006u, 0x0000003au, 0x00000000u, 0x5f637273u,
   0x65646e69u, 0x00000078u, 0x00060006u, 0x0000003au, 0x00000001u, 0x5f787561u,
   0x65646e69u, 0x00000078u, 0x00060006u, 0x0000003au, 0x00000002u, 0x5f74756fu,
   0x65646e69u, 0x00000078u, 0x00030005u, 0x0000003cu, 0x00006370u, 0x00040047u,
   0x00000007u, 0x00000006u, 0x00000004u, 0x00030047u, 0x00000008u, 0x00000003u,
   0x00050048u, 0x00000008u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00040047u,
   0x0000000au, 0x00000021u, 0x00000007u, 0x00040047u, 0x0000000au, 0x00000022u,
   0x00000000u, 0x00030047u, 0x0000000du, 0x00000002u, 0x00050048u, 0x0000000du,
   0x00000000u, 0x00000023u, 0x00000000u, 0x00040047u, 0x00000011u, 0x00000021u,
   0x00000000u, 0x00040047u, 0x00000011u, 0x00000022u, 0x00000000u, 0x00030047u,
   0x0000001fu, 0x00000003u, 0x00050048u, 0x0000001fu, 0x00000000u, 0x00000023u,
   0x00000000u, 0x00040047u, 0x00000023u, 0x00000021u, 0x00000002u, 0x00040047u,
   0x00000023u, 0x00000022u, 0x00000001u, 0x00030047u, 0x0000003au, 0x00000002u,
   0x00050048u, 0x0000003au, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u,
   0x0000003au, 0x00000001u, 0x00000023u, 0x00000004u, 0x00050048u, 0x0000003au,
   0x00000002u, 0x00000023u, 0x00000008u, 0x00040047u, 0x0000004fu, 0x0000000bu,
   0x00000019u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00040015u, 0x00000006u, 0x00000020u, 0x00000000u, 0x0003001du, 0x00000007u,
   0x00000006u, 0x0003001eu, 0x00000008u, 0x00000007u, 0x00040020u, 0x00000009u,
   0x00000002u, 0x00000008u, 0x0004003bu, 0x00000009u, 0x0000000au, 0x00000002u,
   0x00040015u, 0x0000000bu, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000bu,
   0x0000000cu, 0x00000000u, 0x0003001eu, 0x0000000du, 0x00000006u, 0x0004002bu,
   0x00000006u, 0x0000000eu, 0x00000003u, 0x0004001cu, 0x0000000fu, 0x0000000du,
   0x0000000eu, 0x00040020u, 0x00000010u, 0x00000002u, 0x0000000fu, 0x0004003bu,
   0x00000010u, 0x00000011u, 0x00000002u, 0x00040020u, 0x00000012u, 0x00000002u,
   0x00000006u, 0x0004002bu, 0x0000000bu, 0x00000016u, 0x00000001u, 0x0004002bu,
   0x0000000bu, 0x0000001au, 0x00000002u, 0x0004002bu, 0x0000000bu, 0x0000001eu,
   0x00000003u, 0x0003001eu, 0x0000001fu, 0x00000006u, 0x0004002bu, 0x00000006u,
   0x00000020u, 0x00000002u, 0x0004001cu, 0x00000021u, 0x0000001fu, 0x00000020u,
   0x00040020u, 0x00000022u, 0x00000002u, 0x00000021u, 0x0004003bu, 0x00000022u,
   0x00000023u, 0x00000002u, 0x0004002bu, 0x0000000bu, 0x00000027u, 0x00000004u,
   0x00040020u, 0x0000002bu, 0x00000007u, 0x00000006u, 0x0004002bu, 0x0000000bu,
   0x0000002du, 0x0000001fu, 0x0004002bu, 0x0000000bu, 0x00000030u, 0x00000005u,
   0x0004002bu, 0x0000000bu, 0x00000035u, 0x00000006u, 0x0005001eu, 0x0000003au,
   0x00000006u, 0x00000006u, 0x00000006u, 0x00040020u, 0x0000003bu, 0x00000009u,
   0x0000003au, 0x0004003bu, 0x0000003bu, 0x0000003cu, 0x00000009u, 0x00040020u,
   0x0000003du, 0x00000009u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x00000047u,
   0x00000001u, 0x00040017u, 0x0000004eu, 0x00000006u, 0x00000003u, 0x0006002cu,
   0x0000004eu, 0x0000004fu, 0x00000047u, 0x00000047u, 0x00000047u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x0004003bu, 0x0000002bu, 0x0000002cu, 0x00000007u, 0x00060041u, 0x00000012u,
   0x00000013u, 0x00000011u, 0x0000000cu, 0x0000000cu, 0x0004003du, 0x00000006u,
   0x00000014u, 0x00000013u, 0x00060041u, 0x00000012u, 0x00000015u, 0x0000000au,
   0x0000000cu, 0x0000000cu, 0x0003003eu, 0x00000015u, 0x00000014u, 0x00060041u,
   0x00000012u, 0x00000017u, 0x00000011u, 0x00000016u, 0x0000000cu, 0x0004003du,
   0x00000006u, 0x00000018u, 0x00000017u, 0x00060041u, 0x00000012u, 0x00000019u,
   0x0000000au, 0x0000000cu, 0x00000016u, 0x0003003eu, 0x00000019u, 0x00000018u,
   0x00060041u, 0x00000012u, 0x0000001bu, 0x00000011u, 0x0000001au, 0x0000000cu,
   0x0004003du, 0x00000006u, 0x0000001cu, 0x0000001bu, 0x00060041u, 0x00000012u,
   0x0000001du, 0x0000000au, 0x0000000cu, 0x0000001au, 0x0003003eu, 0x0000001du,
   0x0000001cu, 0x00060041u, 0x00000012u, 0x00000024u, 0x00000023u, 0x0000000cu,
   0x0000000cu, 0x0004003du, 0x00000006u, 0x00000025u, 0x00000024u, 0x00060041u,
   0x00000012u, 0x00000026u, 0x0000000au, 0x0000000cu, 0x0000001eu, 0x0003003eu,
   0x00000026u, 0x00000025u, 0x00060041u, 0x00000012u, 0x00000028u, 0x00000023u,
   0x00000016u, 0x0000000cu, 0x0004003du, 0x00000006u, 0x00000029u, 0x00000028u,
   0x00060041u, 0x00000012u, 0x0000002au, 0x0000000au, 0x0000000cu, 0x00000027u,
   0x0003003eu, 0x0000002au, 0x00000029u, 0x00060041u, 0x00000012u, 0x0000002eu,
   0x0000000au, 0x0000000cu, 0x0000002du, 0x0004003du, 0x00000006u, 0x0000002fu,
   0x0000002eu, 0x0003003eu, 0x0000002cu, 0x0000002fu, 0x0004003du, 0x00000006u,
   0x00000031u, 0x0000002cu, 0x00060041u, 0x00000012u, 0x00000032u, 0x00000011u,
   0x00000031u, 0x0000000cu, 0x0004003du, 0x00000006u, 0x00000033u, 0x00000032u,
   0x00060041u, 0x00000012u, 0x00000034u, 0x0000000au, 0x0000000cu, 0x00000030u,
   0x0003003eu, 0x00000034u, 0x00000033u, 0x0004003du, 0x00000006u, 0x00000036u,
   0x0000002cu, 0x00060041u, 0x00000012u, 0x00000037u, 0x00000023u, 0x00000036u,
   0x0000000cu, 0x0004003du, 0x00000006u, 0x00000038u, 0x00000037u, 0x00060041u,
   0x00000012u, 0x00000039u, 0x0000000au, 0x0000000cu, 0x00000035u, 0x0003003eu,
   0x00000039u, 0x00000038u, 0x00050041u, 0x0000003du, 0x0000003eu, 0x0000003cu,
   0x0000001au, 0x0004003du, 0x00000006u, 0x0000003fu, 0x0000003eu, 0x00050041u,
   0x0000003du, 0x00000040u, 0x0000003cu, 0x0000000cu, 0x0004003du, 0x00000006u,
   0x00000041u, 0x00000040u, 0x00060041u, 0x00000012u, 0x00000042u, 0x00000011u,
   0x00000041u, 0x0000000cu, 0x0004003du, 0x00000006u, 0x00000043u, 0x00000042u,
   0x00060041u, 0x00000012u, 0x00000044u, 0x0000000au, 0x0000000cu, 0x0000003fu,
   0x0003003eu, 0x00000044u, 0x00000043u, 0x00050041u, 0x0000003du, 0x00000045u,
   0x0000003cu, 0x0000001au, 0x0004003du, 0x00000006u, 0x00000046u, 0x00000045u,
   0x00050080u, 0x00000006u, 0x00000048u, 0x00000046u, 0x00000047u, 0x00050041u,
   0x0000003du, 0x00000049u, 0x0000003cu, 0x00000016u, 0x0004003du, 0x00000006u,
   0x0000004au, 0x00000049u, 0x00060041u, 0x00000012u, 0x0000004bu, 0x00000023u,
   0x0000004au, 0x0000000cu, 0x0004003du, 0x00000006u, 0x0000004cu, 0x0000004bu,
   0x00060041u, 0x00000012u, 0x0000004du, 0x0000000au, 0x0000000cu, 0x00000048u,
   0x0003003eu, 0x0000004du, 0x0000004cu, 0x000100fdu, 0x00010038u,
};

static uint32_t
pick_memory(VkPhysicalDevice pdev, uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint32_t *map;
   VkDeviceSize size;
};

static int
make_buffer(VkPhysicalDevice pdev, VkDevice dev, VkDeviceSize size,
            VkBufferUsageFlags usage, struct buffer *out)
{
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = size, .usage = usage,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
   CHECK(vkCreateBuffer(dev, &bci, NULL, &out->buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, out->buffer, &req);
   uint32_t type = pick_memory(pdev, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type == UINT32_MAX)
      type = pick_memory(pdev, req.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (type == UINT32_MAX) {
      fprintf(stderr, "no host-visible memory type\n");
      return 1;
   }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, &out->memory));
   CHECK(vkBindBufferMemory(dev, out->buffer, out->memory, 0));
   CHECK(vkMapMemory(dev, out->memory, 0, VK_WHOLE_SIZE, 0,
                     (void **)&out->map));
   out->size = size;
   return 0;
}

static int
flush(VkDevice dev, const struct buffer *b)
{
   VkMappedMemoryRange r = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                             .memory = b->memory, .size = VK_WHOLE_SIZE };
   CHECK(vkFlushMappedMemoryRanges(dev, 1, &r));
   return 0;
}

static void
destroy_buffer(VkDevice dev, struct buffer *b)
{
   vkUnmapMemory(dev, b->memory);
   vkDestroyBuffer(dev, b->buffer, NULL);
   vkFreeMemory(dev, b->memory, NULL);
}

/* One expected slot of the output. */
struct want {
   int slot;
   uint32_t value;
   const char *what;
};

int
main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t n = 1;
   VkPhysicalDevice pdev;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, &pdev));

   VkPhysicalDeviceProperties pprops;
   vkGetPhysicalDeviceProperties(pdev, &pprops);
   printf("device: %s\n", pprops.deviceName);

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; i++)
      if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
   free(families);
   if (family == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }

   if (pprops.limits.maxBoundDescriptorSets < 2) {
      fprintf(stderr, "maxBoundDescriptorSets %u < 2\n",
              pprops.limits.maxBoundDescriptorSets);
      return 1;
   }
   if (pprops.limits.maxPushConstantsSize < sizeof(struct push)) {
      fprintf(stderr, "maxPushConstantsSize %u < %zu\n",
              pprops.limits.maxPushConstantsSize, sizeof(struct push));
      return 1;
   }

   /*
    * Dynamic rendering is core from 1.3, but the native driver reports less
    * than that and means it, so ask for the extension when it is advertised --
    * and name its dependency chain as well, because on a device below 1.3
    * nothing else supplies it and the validation layer refuses the device
    * otherwise: VUID-vkCreateDevice-ppEnabledExtensionNames-01387. Above 1.3
    * the chain is core, so only the extension itself is asked for. This test
    * never renders and does not need any of it; the preamble is shared with
    * every other cpvk test so that device creation is one thing to get right
    * rather than a per-test decision.
    */
   static const char *const wanted[] = {
      VK_KHR_MULTIVIEW_EXTENSION_NAME,
      VK_KHR_MAINTENANCE2_EXTENSION_NAME,
      VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
      VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
      VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
   };
   const uint32_t wanted_count = sizeof(wanted) / sizeof(wanted[0]);
   const int core_dynrend = pprops.apiVersion >= VK_API_VERSION_1_3;

   uint32_t ext_count = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, NULL));
   VkExtensionProperties *exts = calloc(ext_count, sizeof(*exts));
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, exts));
   const char *dev_exts[sizeof(wanted) / sizeof(wanted[0])];
   uint32_t dev_ext_count = 0;
   int have_dynrend = 0;
   for (uint32_t w = 0; w < wanted_count; w++) {
      const int is_dynrend = w == wanted_count - 1;
      if (core_dynrend && !is_dynrend)
         continue;
      for (uint32_t i = 0; i < ext_count; i++) {
         if (strcmp(exts[i].extensionName, wanted[w]))
            continue;
         dev_exts[dev_ext_count++] = wanted[w];
         have_dynrend |= is_dynrend;
         break;
      }
   }
   free(exts);
   if (!have_dynrend && !core_dynrend) {
      fprintf(stderr, "no dynamic rendering\n");
      return 1;
   }

   /*
    * The features this test is actually about. Indexing a descriptor array
    * with anything but a constant needs them, so they are enabled where they
    * exist and named where they do not -- a device without them cannot run the
    * dynamic half of this legally, and saying so is more use than a silent
    * pass.
    */
   VkPhysicalDeviceFeatures supported;
   vkGetPhysicalDeviceFeatures(pdev, &supported);
   VkPhysicalDeviceFeatures enabled = {
      .shaderUniformBufferArrayDynamicIndexing =
         supported.shaderUniformBufferArrayDynamicIndexing,
      .shaderStorageBufferArrayDynamicIndexing =
         supported.shaderStorageBufferArrayDynamicIndexing,
   };
   printf("  shaderUniformBufferArrayDynamicIndexing %s, "
          "shaderStorageBufferArrayDynamicIndexing %s\n",
          enabled.shaderUniformBufferArrayDynamicIndexing ? "yes" : "NO",
          enabled.shaderStorageBufferArrayDynamicIndexing ? "yes" : "NO");

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &prio };
   VkPhysicalDeviceDynamicRenderingFeatures dyn = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &dyn,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
      .enabledExtensionCount = dev_ext_count,
      .ppEnabledExtensionNames = dev_exts,
      .pEnabledFeatures = &enabled };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   /*
    * Five separate buffers behind the two arrays, each with its own
    * allocation. Sharing one allocation at different offsets would test the
    * descriptor's offset field; separate buffers test the element's pointer,
    * which is the thing that collapses when the array index is dropped.
    */
   struct buffer src[SRC_COUNT], aux[AUX_COUNT], out;
   for (int i = 0; i < SRC_COUNT; i++) {
      if (make_buffer(pdev, dev, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      &src[i]))
         return 1;
      src[i].map[0] = SRC_VALUE(i);
      if (flush(dev, &src[i]))
         return 1;
   }
   for (int i = 0; i < AUX_COUNT; i++) {
      if (make_buffer(pdev, dev, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      &aux[i]))
         return 1;
      aux[i].map[0] = AUX_VALUE(i);
      if (flush(dev, &aux[i]))
         return 1;
   }
   if (make_buffer(pdev, dev, SLOTS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   &out))
      return 1;
   for (int i = 0; i < SLOTS; i++)
      out.map[i] = SENTINEL;
   out.map[MEM_SLOT] = MEM_INDEX;
   if (flush(dev, &out))
      return 1;

   /* set 0: the uniform buffer array, and the output at a binding well away
    * from it, so the two are not adjacent by accident. */
   VkDescriptorSetLayoutBinding set0[2] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = SRC_COUNT,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 7, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   /* set 1: the storage buffer array, on its own set. */
   VkDescriptorSetLayoutBinding set1[1] = {
      { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = AUX_COUNT,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   VkDescriptorSetLayoutCreateInfo sl0 = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = set0 };
   VkDescriptorSetLayoutCreateInfo sl1 = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = set1 };
   VkDescriptorSetLayout layouts[2];
   CHECK(vkCreateDescriptorSetLayout(dev, &sl0, NULL, &layouts[0]));
   CHECK(vkCreateDescriptorSetLayout(dev, &sl1, NULL, &layouts[1]));

   VkPushConstantRange range = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                 .offset = 0,
                                 .size = (uint32_t)sizeof(struct push) };
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 2, .pSetLayouts = layouts,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   VkDescriptorPoolSize sizes[2] = {
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SRC_COUNT },
      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, AUX_COUNT + 1 },
   };
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 2, .pPoolSizes = sizes };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 2,
      .pSetLayouts = layouts };
   VkDescriptorSet sets[2];
   CHECK(vkAllocateDescriptorSets(dev, &dsai, sets));

   VkDescriptorBufferInfo src_info[SRC_COUNT];
   for (int i = 0; i < SRC_COUNT; i++)
      src_info[i] = (VkDescriptorBufferInfo){ src[i].buffer, 0, VK_WHOLE_SIZE };
   VkDescriptorBufferInfo aux_info[AUX_COUNT];
   for (int i = 0; i < AUX_COUNT; i++)
      aux_info[i] = (VkDescriptorBufferInfo){ aux[i].buffer, 0, VK_WHOLE_SIZE };
   VkDescriptorBufferInfo out_info = { out.buffer, 0, VK_WHOLE_SIZE };

   /*
    * The uniform array goes in as one write of three elements; the storage
    * array goes in as two writes of one, at dstArrayElement 0 and 1. Both
    * spellings are legal and a driver can get one right and the other wrong.
    */
   VkWriteDescriptorSet writes[4] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = sets[0], .dstBinding = 0, .dstArrayElement = 0,
        .descriptorCount = SRC_COUNT,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = src_info },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = sets[0], .dstBinding = 7, .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &out_info },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = sets[1], .dstBinding = 2, .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &aux_info[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = sets[1], .dstBinding = 2, .dstArrayElement = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &aux_info[1] },
   };
   vkUpdateDescriptorSets(dev, 4, writes, 0, NULL);

   VkShaderModuleCreateInfo smi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(comp_spv), .pCode = comp_spv };
   VkShaderModule cs;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &cs));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs,
                 .pName = "main" },
      .layout = layout };
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   /* Both sets in one call, so set 1 is reached through firstSet arithmetic
    * rather than by being bound on its own. */
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2,
                           sets, 0, NULL);

   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                      (uint32_t)sizeof(pass_a), &pass_a);
   vkCmdDispatch(cmd, 1, 1, 1);

   /* The second dispatch reads the slot the first one wrote, and writes the
    * same buffer, so the two are separated. */
   VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                           VK_ACCESS_SHADER_WRITE_BIT };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        1, &mb, 0, NULL, 0, NULL);

   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                      (uint32_t)sizeof(pass_b), &pass_b);
   vkCmdDispatch(cmd, 1, 1, 1);

   VkMemoryBarrier to_host = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                               .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0,
                        1, &to_host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   VkMappedMemoryRange invalidate = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = out.memory, .size = VK_WHOLE_SIZE };
   CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &invalidate));

   /*
    * What each written slot has to be. Everything else in the buffer is the
    * sentinel, checked below, so a stray store is caught too.
    */
   const struct want wants[] = {
      { 0,  SRC_VALUE(0), "src[0] constant index" },
      { 1,  SRC_VALUE(1), "src[1] constant index" },
      { 2,  SRC_VALUE(2), "src[2] constant index" },
      { 3,  AUX_VALUE(0), "aux[0] constant index, set 1" },
      { 4,  AUX_VALUE(1), "aux[1] constant index, set 1" },
      { 5,  SRC_VALUE(MEM_INDEX), "src[m] index loaded from memory" },
      { 6,  AUX_VALUE(MEM_INDEX), "aux[m] index loaded from memory" },
      { (int)pass_a.out_index,     SRC_VALUE(pass_a.src_index),
        "src[push] dispatch 1" },
      { (int)pass_a.out_index + 1, AUX_VALUE(pass_a.aux_index),
        "aux[push] dispatch 1" },
      { (int)pass_b.out_index,     SRC_VALUE(pass_b.src_index),
        "src[push] dispatch 2" },
      { (int)pass_b.out_index + 1, AUX_VALUE(pass_b.aux_index),
        "aux[push] dispatch 2" },
   };
   const int want_count = (int)(sizeof(wants) / sizeof(wants[0]));

   int fail = 0;
   for (int i = 0; i < want_count; i++) {
      const uint32_t got = out.map[wants[i].slot];
      const int ok = got == wants[i].value;
      printf("  slot %-2d  %-34s want 0x%04x  got 0x%08x  %s\n",
             wants[i].slot, wants[i].what, wants[i].value, got,
             ok ? "ok" : "WRONG");
      if (!ok)
         fail = 1;
   }

   /* Slots nothing should have touched. */
   int written[SLOTS] = { 0 };
   for (int i = 0; i < want_count; i++)
      written[wants[i].slot] = 1;
   written[MEM_SLOT] = 1;
   unsigned scribbled = 0;
   for (int i = 0; i < SLOTS; i++)
      if (!written[i] && out.map[i] != SENTINEL) {
         printf("  slot %-2d  untouched                          "
                "want 0x%08x  got 0x%08x  WRONG\n",
                i, SENTINEL, out.map[i]);
         scribbled++;
      }
   if (out.map[MEM_SLOT] != MEM_INDEX) {
      printf("  slot %-2d  the seeded index                   "
             "want 0x%08x  got 0x%08x  WRONG\n",
             MEM_SLOT, MEM_INDEX, out.map[MEM_SLOT]);
      scribbled++;
   }
   if (scribbled) {
      printf("FAIL %u slot(s) the shader never writes came back changed\n",
             scribbled);
      fail = 1;
   }

   /*
    * The failure this test exists for, said plainly: every array element
    * resolving to element zero passes nothing above except the slots that ask
    * for element zero, so name it when the values collapse.
    */
   if (out.map[0] == out.map[1] && out.map[1] == out.map[2]) {
      printf("FAIL the three uniform buffer array elements all read the "
             "same buffer\n");
      fail = 1;
   }
   if (out.map[3] == out.map[4]) {
      printf("FAIL both storage buffer array elements read the same buffer\n");
      fail = 1;
   }
   if (out.map[5] == SRC_VALUE(0) && MEM_INDEX != 0) {
      printf("FAIL the memory-loaded index landed on uniform element 0\n");
      fail = 1;
   }
   if (out.map[pass_a.out_index] == out.map[pass_b.out_index]) {
      printf("FAIL both dispatches read the same uniform array element, so "
             "the pushed index was not read per dispatch\n");
      fail = 1;
   }

   if (!fail)
      printf("PASS every descriptor array element resolved to its own buffer, "
             "across two sets, constant and dynamic indices, and two "
             "dispatches\n");

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, cs, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, layouts[1], NULL);
   vkDestroyDescriptorSetLayout(dev, layouts[0], NULL);
   destroy_buffer(dev, &out);
   for (int i = 0; i < AUX_COUNT; i++)
      destroy_buffer(dev, &aux[i]);
   for (int i = 0; i < SRC_COUNT; i++)
      destroy_buffer(dev, &src[i]);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fail;
}
