/*
 * The three core clear commands: vkCmdClearColorImage,
 * vkCmdClearDepthStencilImage and vkCmdClearAttachments.
 *
 * All three were NULL dispatch slots until this test was written -- nothing in
 * the driver and no vk_common_* fallback in Mesa's runtime -- so an
 * application that looked one up got NULL from vkGetDeviceProcAddr and an
 * application that called one through the loader jumped to address zero. They
 * are core Vulkan 1.0 with no feature or extension gate.
 *
 * What each section covers, and what would break it:
 *
 *   A  the three entry points resolve at all. This is the regression that
 *      started the work: a NULL here is a segfault in the caller with no
 *      driver output of any kind.
 *   B  vkCmdClearColorImage against a subresource range: two mip levels and
 *      two array layers, cleared whole and then one level of one layer
 *      cleared again. A clear that ignored baseMipLevel or baseArrayLayer
 *      passes the first half and fails the second. The offsets come from
 *      vkGetImageSubresourceLayout, so the test does not restate the
 *      driver's layout.
 *   C  vkCmdClearDepthStencilImage on D24_UNORM_S8_UINT, which packs both
 *      aspects into one 32-bit word: both aspects, then depth alone, then
 *      stencil alone. An implementation that wrote the whole word would
 *      destroy the aspect the caller did not name, and this is the case a
 *      masked write exists for.
 *   D  the same on D32_SFLOAT_S8_UINT, where the stencil byte follows the
 *      depth float inside an eight-byte element.
 *   E  vkCmdClearAttachments *after* a draw in the same pass. This is the
 *      ordering the driver records operations for: the clear must land where
 *      it was recorded, so the rectangle it names comes out cleared and the
 *      rest of the attachment keeps the draw. A clear performed at record
 *      time, or hoisted to the start of the pass, leaves the draw's colour
 *      everywhere and fails here.
 *   F  an in-pass depth clear over a sub-rectangle, read back through the
 *      pass's depth store. The depth test in this driver reads a renderer-side
 *      buffer rather than the image, so this proves the clear reached the
 *      buffer the test would read and the store carried it out.
 *   G  an in-pass stencil clear of the whole render area, which the pass's
 *      depth store is the only thing that can honour -- the same route
 *      LOAD_OP_CLEAR's stencil value takes.
 *   I  vkCmdClearDepthStencilImage on D16_UNORM, whose texel is two bytes
 *      wide -- half of every other depth format here. A clear that wrote a
 *      32-bit word would take the neighbouring texel with it, and the last
 *      one would be written past the end of the row. The stencil aspect must
 *      be refused for it, the way it already is for D32_SFLOAT.
 *   H  the refusals, which must be refusals and not crashes or silent
 *      no-ops: a layered clear rectangle, a stencil clear of part of the
 *      render area, and a vkCmdClearAttachments outside a render pass. Each
 *      must come back from vkEndCommandBuffer as an error. A gate that
 *      cannot fail is not covering anything, and "advertised and then
 *      quietly clamped" is worse here than "refused".
 *
 * SPIR-V is embedded; nothing is read from disk. Generated with
 * glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- cl.vert --
 *   #version 450
 *   void main()
 *   {
 *      vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
 *      gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
 *   }
 *
 *   -- cl.frag --
 *   #version 450
 *   layout(location = 0) out vec4 o_colour;
 *   layout(push_constant) uniform PC { vec4 c; } pc;
 *   void main() { o_colour = pc.c; }
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

#define FAIL(...) do { fprintf(stderr, "cpvk_clear: " __VA_ARGS__); \
   return 1; } while (0)

static const uint32_t clear_vs_spv[] = {
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

static const uint32_t clear_fs_spv[] = {
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

/* A linear, host-visible image: the memory the driver writes is the memory
 * this test maps, which is what makes a clear checkable byte for byte. */
static VkResult
make_image(VkFormat format, uint32_t w, uint32_t h, uint32_t mips,
           uint32_t layers, VkImageUsageFlags usage, VkImage *out_image,
           VkDeviceMemory *out_memory, void **out_map)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { w, h, 1 }, .mipLevels = mips, .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = usage };
   VkResult r = vkCreateImage(dev, &ici, NULL, out_image);
   if (r != VK_SUCCESS)
      return r;

   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, *out_image, &req);
   uint32_t type = pick_memory(req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (type == UINT32_MAX)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   r = vkAllocateMemory(dev, &mai, NULL, out_memory);
   if (r != VK_SUCCESS)
      return r;
   r = vkBindImageMemory(dev, *out_image, *out_memory, 0);
   if (r != VK_SUCCESS)
      return r;
   return vkMapMemory(dev, *out_memory, 0, VK_WHOLE_SIZE, 0, out_map);
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
      static const char *names[3] = { "vkCmdClearAttachments",
                                      "vkCmdClearColorImage",
                                      "vkCmdClearDepthStencilImage" };
      for (int i = 0; i < 3; i++) {
         if (!vkGetDeviceProcAddr(dev, names[i]))
            FAIL("A: vkGetDeviceProcAddr(%s) is NULL\n", names[i]);
      }
      puts("A: the three clear entry points resolve");
   }

   /* ---------------------------------------------------------------- B */
   {
      const uint32_t W = 16, H = 16;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_R8G8B8A8_UNORM, W, H, 2, 2,
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));

      /* Values that land exactly on a byte, so the check is equality. */
      VkClearColorValue whole = { .float32 = { 64.0f / 255.0f, 128.0f / 255.0f,
                                               192.0f / 255.0f, 1.0f } };
      VkClearColorValue one = { .float32 = { 1.0f, 0.0f, 32.0f / 255.0f,
                                             16.0f / 255.0f } };
      const uint8_t whole_bytes[4] = { 64, 128, 192, 255 };
      const uint8_t one_bytes[4] = { 255, 0, 32, 16 };

      VkImageSubresourceRange all = { VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                      VK_REMAINING_MIP_LEVELS, 0,
                                      VK_REMAINING_ARRAY_LAYERS };
      VkImageSubresourceRange one_sub = { VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                                          1, 1 };
      CHECK(begin_cmd());
      vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &whole, 1,
                           &all);
      vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &one, 1,
                           &one_sub);
      CHECK(end_and_run());

      for (uint32_t level = 0; level < 2; level++) {
         for (uint32_t layer = 0; layer < 2; layer++) {
            VkImageSubresource sub = { VK_IMAGE_ASPECT_COLOR_BIT, level,
                                       layer };
            VkSubresourceLayout sl;
            vkGetImageSubresourceLayout(dev, image, &sub, &sl);
            const uint8_t *want = (level == 1 && layer == 1) ? one_bytes
                                                             : whole_bytes;
            uint32_t lw = W >> level, lh = H >> level;
            for (uint32_t y = 0; y < lh; y++) {
               for (uint32_t x = 0; x < lw; x++) {
                  const uint8_t *px = map + sl.offset + y * sl.rowPitch + x * 4;
                  if (memcmp(px, want, 4))
                     FAIL("B: level %u layer %u (%u,%u) is %u %u %u %u, "
                          "want %u %u %u %u\n", level, layer, x, y,
                          px[0], px[1], px[2], px[3],
                          want[0], want[1], want[2], want[3]);
               }
            }
         }
      }
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("B: vkCmdClearColorImage honours mip level and array layer");
   }

   /* ---------------------------------------------------------------- C */
   {
      const uint32_t W = 8, H = 8;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_D24_UNORM_S8_UINT, W, H, 1, 1,
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));
      VkImageSubresource sub = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0 };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, image, &sub, &sl);

      const uint32_t d_quarter = (uint32_t)lrintf(0.25f * 16777215.0f);
      const uint32_t d_one = 16777215u;

      struct { VkImageAspectFlags aspects; float depth; uint32_t stencil;
               uint32_t want; const char *what; } steps[] = {
         { VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
           0.25f, 0xab, (0xabu << 24) | d_quarter, "both aspects" },
         { VK_IMAGE_ASPECT_DEPTH_BIT,
           1.0f, 0x11, (0xabu << 24) | d_one, "depth alone keeps stencil" },
         { VK_IMAGE_ASPECT_STENCIL_BIT,
           0.0f, 0x12, (0x12u << 24) | d_one, "stencil alone keeps depth" },
      };
      for (unsigned s = 0; s < 3; s++) {
         VkClearDepthStencilValue v = { steps[s].depth, steps[s].stencil };
         VkImageSubresourceRange range = { steps[s].aspects, 0, 1, 0, 1 };
         CHECK(begin_cmd());
         vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &v,
                                     1, &range);
         CHECK(end_and_run());
         for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
               uint32_t got;
               memcpy(&got, map + sl.offset + y * sl.rowPitch + x * 4, 4);
               if (got != steps[s].want)
                  FAIL("C: %s at (%u,%u): 0x%08x, want 0x%08x\n",
                       steps[s].what, x, y, got, steps[s].want);
            }
         }
      }
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("C: vkCmdClearDepthStencilImage packs D24S8 per aspect");
   }

   /* ---------------------------------------------------------------- D */
   {
      const uint32_t W = 8, H = 8;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_D32_SFLOAT_S8_UINT, W, H, 1, 1,
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));
      VkImageSubresource sub = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0 };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, image, &sub, &sl);

      VkClearDepthStencilValue both = { 0.5f, 0x5a };
      VkImageSubresourceRange range_both = {
         VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 };
      CHECK(begin_cmd());
      vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &both,
                                  1, &range_both);
      CHECK(end_and_run());
      for (uint32_t i = 0; i < W * H; i++) {
         const uint8_t *e = map + sl.offset + (i / W) * sl.rowPitch +
                            (i % W) * 8;
         float d;
         memcpy(&d, e, 4);
         if (d != 0.5f || e[4] != 0x5a)
            FAIL("D: element %u is depth %f stencil %u, want 0.5 90\n",
                 i, d, e[4]);
      }

      VkClearDepthStencilValue depth_only = { 0.125f, 0xff };
      VkImageSubresourceRange range_depth = { VK_IMAGE_ASPECT_DEPTH_BIT,
                                              0, 1, 0, 1 };
      CHECK(begin_cmd());
      vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL,
                                  &depth_only, 1, &range_depth);
      CHECK(end_and_run());
      for (uint32_t i = 0; i < W * H; i++) {
         const uint8_t *e = map + sl.offset + (i / W) * sl.rowPitch +
                            (i % W) * 8;
         float d;
         memcpy(&d, e, 4);
         if (d != 0.125f || e[4] != 0x5a)
            FAIL("D: after a depth-only clear element %u is depth %f "
                 "stencil %u, want 0.125 90\n", i, d, e[4]);
      }
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("D: vkCmdClearDepthStencilImage keeps the D32S8 stencil byte");
   }

   /* ---------------------------------------------------------------- E */
   {
      const uint32_t W = 64, H = 64;
      const int32_t RX = 16, RY = 16, RW = 32, RH = 32;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_B8G8R8A8_UNORM, W, H, 1, 1,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                       &image, &memory, (void **)&map));
      VkImageViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      VkImageView view;
      CHECK(vkCreateImageView(dev, &vci, NULL, &view));

      VkShaderModuleCreateInfo vsmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(clear_vs_spv), .pCode = clear_vs_spv };
      VkShaderModule vs;
      CHECK(vkCreateShaderModule(dev, &vsmi, NULL, &vs));
      VkShaderModuleCreateInfo fsmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(clear_fs_spv), .pCode = clear_fs_spv };
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
           .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs,
           .pName = "main" },
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

      const float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
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

      VkClearAttachment ca = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
         .clearValue.color.float32 = { 0.0f, 1.0f, 0.0f, 1.0f } };
      VkClearRect cr = { { { RX, RY }, { RW, RH } }, 0, 1 };

      CHECK(begin_cmd());
      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(red), red);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      /* After the draw, on purpose. */
      vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
      end_rendering(cmd);
      CHECK(end_and_run());

      /* B, G, R, A in memory. */
      const uint8_t drawn[4] = { 0, 0, 255, 255 };
      const uint8_t cleared[4] = { 0, 255, 0, 255 };
      VkImageSubresource sub = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, image, &sub, &sl);
      for (uint32_t y = 0; y < H; y++) {
         for (uint32_t x = 0; x < W; x++) {
            bool inside = (int32_t)x >= RX && (int32_t)x < RX + RW &&
                          (int32_t)y >= RY && (int32_t)y < RY + RH;
            const uint8_t *want = inside ? cleared : drawn;
            const uint8_t *px = map + sl.offset + y * sl.rowPitch + x * 4;
            if (memcmp(px, want, 4))
               FAIL("E: (%u,%u) is %u %u %u %u, want %u %u %u %u -- a clear "
                    "recorded after a draw did not land after it\n", x, y,
                    px[0], px[1], px[2], px[3],
                    want[0], want[1], want[2], want[3]);
         }
      }
      vkDestroyPipeline(dev, pipe, NULL);
      vkDestroyPipelineLayout(dev, layout, NULL);
      vkDestroyShaderModule(dev, vs, NULL);
      vkDestroyShaderModule(dev, fs, NULL);
      vkDestroyImageView(dev, view, NULL);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("E: vkCmdClearAttachments lands after the draws it follows");
   }

   /* ---------------------------------------------------------------- F */
   {
      const uint32_t W = 32, H = 32;
      const int32_t RX = 8, RY = 8, RW = 16, RH = 16;
      VkImage image;
      VkDeviceMemory memory;
      float *map;
      CHECK(make_image(VK_FORMAT_D32_SFLOAT, W, H, 1, 1,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));
      VkImageViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_D32_SFLOAT,
         .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 } };
      VkImageView view;
      CHECK(vkCreateImageView(dev, &vci, NULL, &view));

      VkRenderingAttachmentInfo depth = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.depthStencil.depth = 0.25f };
      VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .renderArea = { { 0, 0 }, { W, H } },
                             .layerCount = 1, .pDepthAttachment = &depth };
      VkClearAttachment ca = { .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                               .clearValue.depthStencil.depth = 0.75f };
      VkClearRect cr = { { { RX, RY }, { RW, RH } }, 0, 1 };

      CHECK(begin_cmd());
      begin_rendering(cmd, &ri);
      vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
      end_rendering(cmd);
      CHECK(end_and_run());

      for (uint32_t y = 0; y < H; y++) {
         for (uint32_t x = 0; x < W; x++) {
            bool inside = (int32_t)x >= RX && (int32_t)x < RX + RW &&
                          (int32_t)y >= RY && (int32_t)y < RY + RH;
            float want = inside ? 0.75f : 0.25f;
            float got = map[y * W + x];
            if (fabsf(got - want) > 1e-6f)
               FAIL("F: depth (%u,%u) is %f, want %f\n", x, y, got, want);
         }
      }
      vkDestroyImageView(dev, view, NULL);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("F: an in-pass depth clear reaches the pass's depth store");
   }

   /* ---------------------------------------------------------------- G */
   {
      const uint32_t W = 16, H = 16;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_D24_UNORM_S8_UINT, W, H, 1, 1,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));
      VkImageViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_D24_UNORM_S8_UINT,
         .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT |
                               VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 } };
      VkImageView view;
      CHECK(vkCreateImageView(dev, &vci, NULL, &view));

      VkRenderingAttachmentInfo depth = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.depthStencil.depth = 1.0f };
      VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .renderArea = { { 0, 0 }, { W, H } },
                             .layerCount = 1, .pDepthAttachment = &depth };
      VkClearAttachment ca = { .aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT,
                               .clearValue.depthStencil.stencil = 0x7f };
      VkClearRect cr = { { { 0, 0 }, { W, H } }, 0, 1 };

      CHECK(begin_cmd());
      begin_rendering(cmd, &ri);
      vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
      end_rendering(cmd);
      CHECK(end_and_run());

      VkImageSubresource sub = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0 };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, image, &sub, &sl);
      uint32_t want = (0x7fu << 24) | 16777215u;
      for (uint32_t y = 0; y < H; y++) {
         for (uint32_t x = 0; x < W; x++) {
            uint32_t got;
            memcpy(&got, map + sl.offset + y * sl.rowPitch + x * 4, 4);
            if (got != want)
               FAIL("G: (%u,%u) is 0x%08x, want 0x%08x\n", x, y, got, want);
         }
      }
      vkDestroyImageView(dev, view, NULL);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("G: an in-pass stencil clear is carried by the depth store");
   }

   /* ---------------------------------------------------------------- H */
   {
      const uint32_t W = 16, H = 16;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_B8G8R8A8_UNORM, W, H, 1, 1,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));
      VkImageViewCreateInfo vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      VkImageView view;
      CHECK(vkCreateImageView(dev, &vci, NULL, &view));
      VkRenderingAttachmentInfo colour = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.color.float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
      VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                             .renderArea = { { 0, 0 }, { W, H } },
                             .layerCount = 1, .colorAttachmentCount = 1,
                             .pColorAttachments = &colour };

      /* A layered rectangle: this driver binds one layer per pass. */
      VkClearAttachment ca_colour = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0 };
      VkClearRect layered = { { { 0, 0 }, { W, H } }, 1, 1 };
      CHECK(begin_cmd());
      begin_rendering(cmd, &ri);
      vkCmdClearAttachments(cmd, 1, &ca_colour, 1, &layered);
      end_rendering(cmd);
      if (vkEndCommandBuffer(cmd) == VK_SUCCESS)
         FAIL("H: a layered clear rectangle was accepted\n");

      /*
       * A stencil clear of part of the render area, in a pass that really has
       * a stencil aspect -- otherwise this would be refused for the other
       * reason and prove nothing about the rectangle.
       */
      VkImage ds_image;
      VkDeviceMemory ds_memory;
      uint8_t *ds_map;
      CHECK(make_image(VK_FORMAT_D24_UNORM_S8_UINT, W, H, 1, 1,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &ds_image, &ds_memory, (void **)&ds_map));
      VkImageViewCreateInfo ds_vci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = ds_image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_D24_UNORM_S8_UINT,
         .subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT |
                               VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 } };
      VkImageView ds_view;
      CHECK(vkCreateImageView(dev, &ds_vci, NULL, &ds_view));
      VkRenderingAttachmentInfo ds_att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = ds_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.depthStencil.depth = 1.0f };
      VkRenderingInfo ds_ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                .renderArea = { { 0, 0 }, { W, H } },
                                .layerCount = 1,
                                .pDepthAttachment = &ds_att };
      VkClearAttachment ca_stencil = {
         .aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT };
      VkClearRect part = { { { 2, 2 }, { 4, 4 } }, 0, 1 };
      CHECK(begin_cmd());
      begin_rendering(cmd, &ds_ri);
      vkCmdClearAttachments(cmd, 1, &ca_stencil, 1, &part);
      end_rendering(cmd);
      if (vkEndCommandBuffer(cmd) == VK_SUCCESS)
         FAIL("H: a stencil clear of part of the render area was accepted\n");

      /* The same pass and the same aspect over the whole render area is
       * accepted, so the refusal above is about the rectangle and not about
       * the aspect. */
      VkClearRect whole_area = { { { 0, 0 }, { W, H } }, 0, 1 };
      CHECK(begin_cmd());
      begin_rendering(cmd, &ds_ri);
      vkCmdClearAttachments(cmd, 1, &ca_stencil, 1, &whole_area);
      end_rendering(cmd);
      CHECK(end_and_run());
      vkDestroyImageView(dev, ds_view, NULL);
      vkDestroyImage(dev, ds_image, NULL);
      vkFreeMemory(dev, ds_memory, NULL);

      /* And outside a render pass entirely: a refusal, not a crash. */
      VkClearRect whole = { { { 0, 0 }, { W, H } }, 0, 1 };
      CHECK(begin_cmd());
      vkCmdClearAttachments(cmd, 1, &ca_colour, 1, &whole);
      if (vkEndCommandBuffer(cmd) == VK_SUCCESS)
         FAIL("H: vkCmdClearAttachments outside a render pass was accepted\n");

      /* The command buffer still resets and still records afterwards. */
      CHECK(begin_cmd());
      VkClearColorValue grey = { .float32 = { 0.5f, 0.5f, 0.5f, 1.0f } };
      VkImageSubresourceRange all = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &grey, 1,
                           &all);
      CHECK(end_and_run());

      vkDestroyImageView(dev, view, NULL);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("H: the cases this driver cannot serve are refused, not crashed");
   }

   /* ---------------------------------------------------------------- I */
   {
      const uint32_t W = 8, H = 8;
      VkImage image;
      VkDeviceMemory memory;
      uint8_t *map;
      CHECK(make_image(VK_FORMAT_D16_UNORM, W, H, 1, 1,
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &image, &memory, (void **)&map));
      VkImageSubresource sub = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0 };
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, image, &sub, &sl);

      /* Two clears, so that the second has to overwrite the first: a
       * two-byte write that was really four bytes wide would leave half of
       * the previous value in every other texel. */
      const struct { float depth; uint32_t want; } steps[] = {
         { 0.25f, 16384u },   /* round(0.25 * 65535) */
         { 1.0f,  65535u },
      };
      for (unsigned s = 0; s < 2; s++) {
         VkClearDepthStencilValue v = { steps[s].depth, 0 };
         VkImageSubresourceRange range = { VK_IMAGE_ASPECT_DEPTH_BIT,
                                           0, 1, 0, 1 };
         CHECK(begin_cmd());
         vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &v,
                                     1, &range);
         CHECK(end_and_run());
         for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
               uint16_t got;
               memcpy(&got, map + sl.offset + y * sl.rowPitch + x * 2, 2);
               if (got != steps[s].want)
                  FAIL("I: D16 depth %.2f at (%u,%u): %u, want %u\n",
                       steps[s].depth, x, y, got, steps[s].want);
            }
         }
      }

      /* D16 has no stencil aspect, and naming one is the same mistake as
       * naming it on D32_SFLOAT. */
      VkClearDepthStencilValue v = { 0.5f, 0x7f };
      VkImageSubresourceRange stencil_range = { VK_IMAGE_ASPECT_STENCIL_BIT,
                                                0, 1, 0, 1 };
      CHECK(begin_cmd());
      vkCmdClearDepthStencilImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &v, 1,
                                  &stencil_range);
      if (vkEndCommandBuffer(cmd) == VK_SUCCESS)
         FAIL("I: a stencil clear of a D16 image was accepted\n");

      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);
      puts("I: vkCmdClearDepthStencilImage writes D16 two bytes at a time");
   }

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   puts("clear: pass");
   return 0;
}
