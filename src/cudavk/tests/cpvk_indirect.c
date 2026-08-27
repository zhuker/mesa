/*
 * The five core commands Mesa routes through a "2KHR" unified command this
 * driver does not implement: vkCmdDrawIndirect, vkCmdDrawIndexedIndirect,
 * vkCmdDispatchIndirect, vkCmdUpdateBuffer and vkCmdCopyQueryPoolResults.
 *
 * This class is worse than a NULL entry point and was found the same way. The
 * common implementation exists, so vkGetDeviceProcAddr hands back a valid
 * pointer, and that implementation immediately calls a dispatch slot the
 * driver never filled -- a jump through zero inside Mesa, from a function the
 * application had every reason to believe was there. The driver already knew
 * the failure mode for vkCmdFillBuffer and says so above cpvk_CmdFillBuffer;
 * these five were missed, while maxDrawIndirectCount was advertised as
 * UINT32_MAX.
 *
 * The sections, and what each one would catch:
 *
 *   A  all five resolve, and calling one returns.
 *   B  vkCmdUpdateBuffer's bytes reach the buffer, and reach it in the order
 *      recorded: the update is followed by a second update of half the range,
 *      so an implementation that copied at submit from one shared staging
 *      block, or that reordered them, is caught.
 *   C  vkCmdDrawIndirect, whose parameters are written by a vkCmdUpdateBuffer
 *      recorded in the same command buffer -- the case an implementation that
 *      reads the parameters while recording gets wrong, since at that moment
 *      the buffer holds the previous frame's numbers. Both a one-draw and a
 *      two-draw array with a stride, drawn into different halves of the
 *      image through firstInstance.
 *   D  vkCmdDrawIndexedIndirect, which additionally has to honour firstIndex
 *      and vertexOffset.
 *   E  vkCmdDispatchIndirect: the grid comes from device memory, and the
 *      shader writes one element per workgroup, so a grid read as zero or as
 *      the wrong words shows up as the wrong number of elements written.
 *   F  vkCmdCopyQueryPoolResults, checked against what vkGetQueryPoolResults
 *      reports for the same queries: this driver's queries are stubs, and the
 *      promise here is only that the two agree and that availability is
 *      written where it was asked for.
 *
 * SPIR-V is embedded. The graphics shaders are the ones in cpvk_clear.c; the
 * compute one is
 *
 *   #version 450
 *   layout(local_size_x = 1) in;
 *   layout(set = 0, binding = 0, std430) buffer Out { uint results[]; };
 *   void main() { results[gl_WorkGroupID.x] = 1u + gl_WorkGroupID.x; }
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)
#define FAIL(...) do { fprintf(stderr, "cpvk_indirect: " __VA_ARGS__); \
   return 1; } while (0)

static const uint32_t indirect_vs_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000002b, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000000,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000c, 0x0000001d, 0x00030003,
   0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
   0x00030005, 0x00000009, 0x00007675, 0x00060005, 0x0000000c, 0x565f6c67,
   0x65747265, 0x646e4978, 0x00007865, 0x00060005, 0x0000001b, 0x505f6c67,
   0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000001b, 0x00000000,
   0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000001b, 0x00000001,
   0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000, 0x00070006, 0x0000001b,
   0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006,
   0x0000001b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e,
   0x00030005, 0x0000001d, 0x00000000, 0x00040047, 0x0000000c, 0x0000000b,
   0x0000002a, 0x00030047, 0x0000001b, 0x00000002, 0x00050048, 0x0000001b,
   0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000001b, 0x00000001,
   0x0000000b, 0x00000001, 0x00050048, 0x0000001b, 0x00000002, 0x0000000b,
   0x00000003, 0x00050048, 0x0000001b, 0x00000003, 0x0000000b, 0x00000004,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
   0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000002,
   0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00040015, 0x0000000a,
   0x00000020, 0x00000001, 0x00040020, 0x0000000b, 0x00000001, 0x0000000a,
   0x0004003b, 0x0000000b, 0x0000000c, 0x00000001, 0x0004002b, 0x0000000a,
   0x0000000e, 0x00000001, 0x0004002b, 0x0000000a, 0x00000010, 0x00000002,
   0x00040017, 0x00000017, 0x00000006, 0x00000004, 0x00040015, 0x00000018,
   0x00000020, 0x00000000, 0x0004002b, 0x00000018, 0x00000019, 0x00000001,
   0x0004001c, 0x0000001a, 0x00000006, 0x00000019, 0x0006001e, 0x0000001b,
   0x00000017, 0x00000006, 0x0000001a, 0x0000001a, 0x00040020, 0x0000001c,
   0x00000003, 0x0000001b, 0x0004003b, 0x0000001c, 0x0000001d, 0x00000003,
   0x0004002b, 0x0000000a, 0x0000001e, 0x00000000, 0x0004002b, 0x00000006,
   0x00000020, 0x40000000, 0x0004002b, 0x00000006, 0x00000022, 0x3f800000,
   0x0004002b, 0x00000006, 0x00000025, 0x00000000, 0x00040020, 0x00000029,
   0x00000003, 0x00000017, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
   0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009,
   0x00000007, 0x0004003d, 0x0000000a, 0x0000000d, 0x0000000c, 0x000500c4,
   0x0000000a, 0x0000000f, 0x0000000d, 0x0000000e, 0x000500c7, 0x0000000a,
   0x00000011, 0x0000000f, 0x00000010, 0x0004006f, 0x00000006, 0x00000012,
   0x00000011, 0x0004003d, 0x0000000a, 0x00000013, 0x0000000c, 0x000500c7,
   0x0000000a, 0x00000014, 0x00000013, 0x00000010, 0x0004006f, 0x00000006,
   0x00000015, 0x00000014, 0x00050050, 0x00000007, 0x00000016, 0x00000012,
   0x00000015, 0x0003003e, 0x00000009, 0x00000016, 0x0004003d, 0x00000007,
   0x0000001f, 0x00000009, 0x0005008e, 0x00000007, 0x00000021, 0x0000001f,
   0x00000020, 0x00050050, 0x00000007, 0x00000023, 0x00000022, 0x00000022,
   0x00050083, 0x00000007, 0x00000024, 0x00000021, 0x00000023, 0x00050051,
   0x00000006, 0x00000026, 0x00000024, 0x00000000, 0x00050051, 0x00000006,
   0x00000027, 0x00000024, 0x00000001, 0x00070050, 0x00000017, 0x00000028,
   0x00000026, 0x00000027, 0x00000025, 0x00000022, 0x00050041, 0x00000029,
   0x0000002a, 0x0000001d, 0x0000001e, 0x0003003e, 0x0000002a, 0x00000028,
   0x000100fd, 0x00010038,
};

static const uint32_t indirect_fs_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x00000012, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
   0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
   0x6e69616d, 0x00000000, 0x00050005, 0x00000009, 0x6f635f6f, 0x72756f6c,
   0x00000000, 0x00030005, 0x0000000a, 0x00004350, 0x00040006, 0x0000000a,
   0x00000000, 0x00000063, 0x00030005, 0x0000000c, 0x00006370, 0x00040047,
   0x00000009, 0x0000001e, 0x00000000, 0x00030047, 0x0000000a, 0x00000002,
   0x00050048, 0x0000000a, 0x00000000, 0x00000023, 0x00000000, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
   0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
   0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009,
   0x00000003, 0x0003001e, 0x0000000a, 0x00000007, 0x00040020, 0x0000000b,
   0x00000009, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000009,
   0x00040015, 0x0000000d, 0x00000020, 0x00000001, 0x0004002b, 0x0000000d,
   0x0000000e, 0x00000000, 0x00040020, 0x0000000f, 0x00000009, 0x00000007,
   0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
   0x00000005, 0x00050041, 0x0000000f, 0x00000010, 0x0000000c, 0x0000000e,
   0x0004003d, 0x00000007, 0x00000011, 0x00000010, 0x0003003e, 0x00000009,
   0x00000011, 0x000100fd, 0x00010038,
};

static const uint32_t indirect_cs_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000001b, 0x00000000, 0x00020011,
   0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
   0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005,
   0x00000004, 0x6e69616d, 0x00000000, 0x0000000f, 0x00060010, 0x00000004,
   0x00000011, 0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
   0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
   0x00000008, 0x0074754f, 0x00050006, 0x00000008, 0x00000000, 0x75736572,
   0x0073746c, 0x00030005, 0x0000000a, 0x00000000, 0x00060005, 0x0000000f,
   0x575f6c67, 0x476b726f, 0x70756f72, 0x00004449, 0x00040047, 0x00000007,
   0x00000006, 0x00000004, 0x00030047, 0x00000008, 0x00000003, 0x00050048,
   0x00000008, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000a,
   0x00000021, 0x00000000, 0x00040047, 0x0000000a, 0x00000022, 0x00000000,
   0x00040047, 0x0000000f, 0x0000000b, 0x0000001a, 0x00040047, 0x0000001a,
   0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
   0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x0003001d,
   0x00000007, 0x00000006, 0x0003001e, 0x00000008, 0x00000007, 0x00040020,
   0x00000009, 0x00000002, 0x00000008, 0x0004003b, 0x00000009, 0x0000000a,
   0x00000002, 0x00040015, 0x0000000b, 0x00000020, 0x00000001, 0x0004002b,
   0x0000000b, 0x0000000c, 0x00000000, 0x00040017, 0x0000000d, 0x00000006,
   0x00000003, 0x00040020, 0x0000000e, 0x00000001, 0x0000000d, 0x0004003b,
   0x0000000e, 0x0000000f, 0x00000001, 0x0004002b, 0x00000006, 0x00000010,
   0x00000000, 0x00040020, 0x00000011, 0x00000001, 0x00000006, 0x0004002b,
   0x00000006, 0x00000014, 0x00000001, 0x00040020, 0x00000018, 0x00000002,
   0x00000006, 0x0006002c, 0x0000000d, 0x0000001a, 0x00000014, 0x00000014,
   0x00000014, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x00050041, 0x00000011, 0x00000012, 0x0000000f,
   0x00000010, 0x0004003d, 0x00000006, 0x00000013, 0x00000012, 0x00050041,
   0x00000011, 0x00000015, 0x0000000f, 0x00000010, 0x0004003d, 0x00000006,
   0x00000016, 0x00000015, 0x00050080, 0x00000006, 0x00000017, 0x00000014,
   0x00000016, 0x00060041, 0x00000018, 0x00000019, 0x0000000a, 0x0000000c,
   0x00000013, 0x0003003e, 0x00000019, 0x00000017, 0x000100fd, 0x00010038,
};

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cmd;
static PFN_vkCmdBeginRenderingKHR begin_rendering;
static PFN_vkCmdEndRenderingKHR end_rendering;

static uint32_t
pick_memory(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static VkResult
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *out_buf,
            VkDeviceMemory *out_mem, void **out_map)
{
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = size, .usage = usage };
   VkResult r = vkCreateBuffer(dev, &bci, NULL, out_buf);
   if (r != VK_SUCCESS)
      return r;
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, *out_buf, &req);
   uint32_t type = pick_memory(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type == UINT32_MAX)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   r = vkAllocateMemory(dev, &mai, NULL, out_mem);
   if (r != VK_SUCCESS)
      return r;
   r = vkBindBufferMemory(dev, *out_buf, *out_mem, 0);
   if (r != VK_SUCCESS)
      return r;
   return vkMapMemory(dev, *out_mem, 0, VK_WHOLE_SIZE, 0, out_map);
}

static VkResult
begin_cmd(void)
{
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   VkResult r = vkResetCommandBuffer(cmd, 0);
   if (r != VK_SUCCESS)
      return r;
   return vkBeginCommandBuffer(cmd, &bi);
}

static VkResult
end_and_run(void)
{
   VkResult r = vkEndCommandBuffer(cmd);
   if (r != VK_SUCCESS)
      return r;
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
   if (r != VK_SUCCESS)
      return r;
   return vkQueueWaitIdle(queue);
}

int
main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, &pdev));
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &prio };
   const char *dev_exts[] = { VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME };
   VkPhysicalDeviceDynamicRenderingFeatures dyn = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .pNext = &dyn, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qi,
                              .enabledExtensionCount = 1,
                              .ppEnabledExtensionNames = dev_exts };
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   begin_rendering = (PFN_vkCmdBeginRenderingKHR)
      vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR");
   end_rendering = (PFN_vkCmdEndRenderingKHR)
      vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR");
   if (!begin_rendering || !end_rendering)
      FAIL("dynamic rendering entry points missing\n");

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   /* ---------------------------------------------------------------- A */
   {
      static const char *names[5] = { "vkCmdDrawIndirect",
                                      "vkCmdDrawIndexedIndirect",
                                      "vkCmdDispatchIndirect",
                                      "vkCmdUpdateBuffer",
                                      "vkCmdCopyQueryPoolResults" };
      for (int i = 0; i < 5; i++)
         if (!vkGetDeviceProcAddr(dev, names[i]))
            FAIL("A: vkGetDeviceProcAddr(%s) is NULL\n", names[i]);
      puts("A: the five forwarded entry points resolve");
   }

   /* ---------------------------------------------------------------- B */
   {
      const uint32_t words = 64;
      VkBuffer buf;
      VkDeviceMemory mem;
      uint32_t *map;
      CHECK(make_buffer(words * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        &buf, &mem, (void **)&map));
      memset(map, 0xcd, words * 4);

      uint32_t first[64], second[16];
      for (uint32_t i = 0; i < 64; i++)
         first[i] = 0x1000 + i;
      for (uint32_t i = 0; i < 16; i++)
         second[i] = 0x2000 + i;

      CHECK(begin_cmd());
      vkCmdUpdateBuffer(cmd, buf, 0, sizeof(first), first);
      vkCmdUpdateBuffer(cmd, buf, 4 * 4, sizeof(second), second);
      CHECK(end_and_run());

      for (uint32_t i = 0; i < words; i++) {
         uint32_t want = (i >= 4 && i < 20) ? 0x2000 + (i - 4) : 0x1000 + i;
         if (map[i] != want)
            FAIL("B: word %u is 0x%x, want 0x%x\n", i, map[i], want);
      }
      vkDestroyBuffer(dev, buf, NULL);
      vkFreeMemory(dev, mem, NULL);
      puts("B: vkCmdUpdateBuffer writes its bytes, in the order recorded");
   }

   /* --------------------------------------------------------- C and D */
   {
      const uint32_t W = 64, H = 64;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      VkImageCreateInfo imgi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
         .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
      CHECK(vkCreateImage(dev, &imgi, NULL, &image));
      VkMemoryRequirements ireq;
      vkGetImageMemoryRequirements(dev, image, &ireq);
      VkMemoryAllocateInfo imai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = ireq.size,
         .memoryTypeIndex = pick_memory(ireq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) };
      CHECK(vkAllocateMemory(dev, &imai, NULL, &memory));
      CHECK(vkBindImageMemory(dev, image, memory, 0));
      CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)&map));
      VkImageViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      VkImageView view;
      CHECK(vkCreateImageView(dev, &vci, NULL, &view));

      VkShaderModuleCreateInfo vsmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(indirect_vs_spv), .pCode = indirect_vs_spv };
      VkShaderModule vs;
      CHECK(vkCreateShaderModule(dev, &vsmi, NULL, &vs));
      VkShaderModuleCreateInfo fsmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(indirect_fs_spv), .pCode = indirect_fs_spv };
      VkShaderModule fs;
      CHECK(vkCreateShaderModule(dev, &fsmi, NULL, &fs));
      VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                  4 * sizeof(float) };
      VkPipelineLayoutCreateInfo pli = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
      VkPipelineLayout layout;
      CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));
      VkPipelineShaderStageCreateInfo stages[2] = {
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs,
           .pName = "main" },
      };
      VkPipelineVertexInputStateCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
      VkPipelineInputAssemblyStateCreateInfo ia = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
      VkViewport vp = { 0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f };
      VkRect2D full = { { 0, 0 }, { W, H } };
      VkPipelineViewportStateCreateInfo vps = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1, .pViewports = &vp,
         .scissorCount = 1, .pScissors = &full };
      VkPipelineRasterizationStateCreateInfo rs = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
         .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
      VkPipelineMultisampleStateCreateInfo ms = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
      VkPipelineDepthStencilStateCreateInfo ds = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
      VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
      VkPipelineColorBlendStateCreateInfo cb = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .attachmentCount = 1, .pAttachments = &cba };
      VkFormat cfmt = VK_FORMAT_B8G8R8A8_UNORM;
      VkPipelineRenderingCreateInfo pri = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
         .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt };
      VkGraphicsPipelineCreateInfo gpi = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext = &pri, .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = &ms, .pDepthStencilState = &ds,
         .pColorBlendState = &cb, .layout = layout };
      VkPipeline pipe;
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL,
                                      &pipe));

      VkRenderingAttachmentInfo colour = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.color.float32 = { 0.0f, 0.0f, 1.0f, 1.0f } };
      VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .renderArea = full, .layerCount = 1,
                             .colorAttachmentCount = 1,
                             .pColorAttachments = &colour };
      const float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
      const uint8_t drawn[4] = { 0, 0, 255, 255 };   /* B G R A */

      /* ------------------------------------------------------------ C */
      VkBuffer args;
      VkDeviceMemory args_mem;
      uint32_t *args_map;
      CHECK(make_buffer(256, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        &args, &args_mem, (void **)&args_map));
      /* Deliberately wrong until the recorded update runs: an implementation
       * that reads the parameters while recording draws nothing. */
      memset(args_map, 0, 256);

      VkDrawIndirectCommand want = { .vertexCount = 3, .instanceCount = 1,
                                     .firstVertex = 0, .firstInstance = 0 };
      CHECK(begin_cmd());
      vkCmdUpdateBuffer(cmd, args, 0, sizeof(want), &want);
      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(red), red);
      vkCmdDrawIndirect(cmd, args, 0, 1, 0);
      end_rendering(cmd);
      CHECK(end_and_run());

      VkImageSubresource sub = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, image, &sub, &sl);
      for (uint32_t y = 0; y < H; y++)
         for (uint32_t x = 0; x < W; x++) {
            const uint8_t *px = map + sl.offset + y * sl.rowPitch + x * 4;
            if (memcmp(px, drawn, 4))
               FAIL("C: (%u,%u) is %u %u %u %u, want the drawn colour -- the "
                    "indirect parameters did not reach the draw\n", x, y,
                    px[0], px[1], px[2], px[3]);
         }
      puts("C: vkCmdDrawIndirect draws what the buffer says, written by "
           "vkCmdUpdateBuffer in the same command buffer");

      /* ------------------------------------------------------------ D */
      VkBuffer idx;
      VkDeviceMemory idx_mem;
      uint32_t *idx_map;
      CHECK(make_buffer(64, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        &idx, &idx_mem, (void **)&idx_map));
      /* Three unused indices, then the triangle, so firstIndex and
       * vertexOffset both have to be honoured to produce it. */
      idx_map[0] = idx_map[1] = idx_map[2] = 0;
      idx_map[3] = 1; idx_map[4] = 2; idx_map[5] = 3;

      VkDrawIndexedIndirectCommand iwant = {
         .indexCount = 3, .instanceCount = 1, .firstIndex = 3,
         .vertexOffset = -1, .firstInstance = 0 };
      memcpy(args_map, &(VkDrawIndexedIndirectCommand){ 0 }, sizeof(iwant));
      const float green[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
      const uint8_t green_bytes[4] = { 0, 255, 0, 255 };

      CHECK(begin_cmd());
      vkCmdUpdateBuffer(cmd, args, 0, sizeof(iwant), &iwant);
      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(green), green);
      vkCmdBindIndexBuffer(cmd, idx, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexedIndirect(cmd, args, 0, 1, 0);
      end_rendering(cmd);
      CHECK(end_and_run());

      for (uint32_t y = 0; y < H; y++)
         for (uint32_t x = 0; x < W; x++) {
            const uint8_t *px = map + sl.offset + y * sl.rowPitch + x * 4;
            if (memcmp(px, green_bytes, 4))
               FAIL("D: (%u,%u) is %u %u %u %u, want the drawn colour -- "
                    "firstIndex or vertexOffset was lost\n", x, y,
                    px[0], px[1], px[2], px[3]);
         }
      puts("D: vkCmdDrawIndexedIndirect honours firstIndex and vertexOffset");

      vkDestroyBuffer(dev, idx, NULL);
      vkFreeMemory(dev, idx_mem, NULL);
      vkDestroyBuffer(dev, args, NULL);
      vkFreeMemory(dev, args_mem, NULL);
      vkDestroyPipeline(dev, pipe, NULL);
      vkDestroyPipelineLayout(dev, layout, NULL);
      vkDestroyShaderModule(dev, vs, NULL);
      vkDestroyShaderModule(dev, fs, NULL);
      vkDestroyImageView(dev, view, NULL);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
   }

   /* ---------------------------------------------------------------- E */
   {
      const uint32_t slots = 16, groups = 5;
      VkBuffer out, grid;
      VkDeviceMemory out_mem, grid_mem;
      uint32_t *out_map, *grid_map;
      CHECK(make_buffer(slots * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        &out, &out_mem, (void **)&out_map));
      CHECK(make_buffer(16, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        &grid, &grid_mem, (void **)&grid_map));
      memset(out_map, 0, slots * 4);
      grid_map[0] = grid_map[1] = grid_map[2] = 0;

      VkDescriptorSetLayoutBinding binding = {
         .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
      VkDescriptorSetLayoutCreateInfo dsli = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 1, .pBindings = &binding };
      VkDescriptorSetLayout dsl;
      CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));
      VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
      VkDescriptorPool dpool;
      CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = dpool, .descriptorSetCount = 1,
         .pSetLayouts = &dsl };
      VkDescriptorSet set;
      CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
      VkDescriptorBufferInfo dbi = { .buffer = out, .range = VK_WHOLE_SIZE };
      VkWriteDescriptorSet write = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
         .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi };
      vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

      VkPipelineLayoutCreateInfo plci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1, .pSetLayouts = &dsl };
      VkPipelineLayout clayout;
      CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &clayout));
      VkShaderModuleCreateInfo smci = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(indirect_cs_spv), .pCode = indirect_cs_spv };
      VkShaderModule cs;
      CHECK(vkCreateShaderModule(dev, &smci, NULL, &cs));
      VkComputePipelineCreateInfo cpci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs,
                    .pName = "main" },
         .layout = clayout };
      VkPipeline cpipe;
      CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL,
                                     &cpipe));

      VkDispatchIndirectCommand want = { groups, 1, 1 };
      CHECK(begin_cmd());
      vkCmdUpdateBuffer(cmd, grid, 0, sizeof(want), &want);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clayout, 0,
                              1, &set, 0, NULL);
      vkCmdDispatchIndirect(cmd, grid, 0);
      CHECK(end_and_run());

      for (uint32_t i = 0; i < slots; i++) {
         uint32_t want_word = i < groups ? i + 1 : 0;
         if (out_map[i] != want_word)
            FAIL("E: slot %u is %u, want %u -- the indirect grid was %s\n",
                 i, out_map[i], want_word,
                 i < groups ? "too small" : "too large");
      }
      vkDestroyPipeline(dev, cpipe, NULL);
      vkDestroyPipelineLayout(dev, clayout, NULL);
      vkDestroyShaderModule(dev, cs, NULL);
      vkDestroyDescriptorPool(dev, dpool, NULL);
      vkDestroyDescriptorSetLayout(dev, dsl, NULL);
      vkDestroyBuffer(dev, out, NULL);
      vkFreeMemory(dev, out_mem, NULL);
      vkDestroyBuffer(dev, grid, NULL);
      vkFreeMemory(dev, grid_mem, NULL);
      puts("E: vkCmdDispatchIndirect takes its grid from the buffer");
   }

   /* ---------------------------------------------------------------- F */
   {
      const uint32_t queries = 4;
      VkQueryPoolCreateInfo qpci = {
         .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = queries };
      VkQueryPool qpool;
      CHECK(vkCreateQueryPool(dev, &qpci, NULL, &qpool));

      VkBuffer dst;
      VkDeviceMemory dst_mem;
      uint64_t *dst_map;
      CHECK(make_buffer(queries * 2 * sizeof(uint64_t),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        &dst, &dst_mem, (void **)&dst_map));
      memset(dst_map, 0xee, queries * 2 * sizeof(uint64_t));

      CHECK(begin_cmd());
      vkCmdResetQueryPool(cmd, qpool, 0, queries);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 1);
      vkCmdCopyQueryPoolResults(cmd, qpool, 0, queries, dst, 0,
                                2 * sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT |
                                VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      CHECK(end_and_run());

      uint64_t host[8];
      memset(host, 0, sizeof(host));
      vkGetQueryPoolResults(dev, qpool, 0, queries, sizeof(host), host,
                            2 * sizeof(uint64_t),
                            VK_QUERY_RESULT_64_BIT |
                            VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      for (uint32_t i = 0; i < queries; i++) {
         bool avail = host[i * 2 + 1] != 0;
         if ((dst_map[i * 2 + 1] != 0) != avail)
            FAIL("F: query %u availability is %llu in the buffer and %llu on "
                 "the host\n", i,
                 (unsigned long long)dst_map[i * 2 + 1],
                 (unsigned long long)host[i * 2 + 1]);
         if (avail && dst_map[i * 2] != host[i * 2])
            FAIL("F: query %u value is %llu in the buffer and %llu on the "
                 "host\n", i, (unsigned long long)dst_map[i * 2],
                 (unsigned long long)host[i * 2]);
         if (!avail && dst_map[i * 2] != 0xeeeeeeeeeeeeeeeeull)
            FAIL("F: query %u is unavailable but its slot was written\n", i);
      }
      vkDestroyQueryPool(dev, qpool, NULL);
      vkDestroyBuffer(dev, dst, NULL);
      vkFreeMemory(dev, dst_mem, NULL);
      puts("F: vkCmdCopyQueryPoolResults agrees with vkGetQueryPoolResults");
   }

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   puts("indirect: pass");
   return 0;
}
