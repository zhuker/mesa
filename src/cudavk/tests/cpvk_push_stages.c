/*
 * Per-stage push constants: one pipeline layout, two ranges, one draw, and
 * both stages must come out with their own values.
 *
 * The layout declares 32 bytes split between the stages:
 *
 *    bytes  0..15   VK_SHADER_STAGE_VERTEX_BIT     a position offset
 *    bytes 16..31   VK_SHADER_STAGE_FRAGMENT_BIT   an output colour
 *
 * and each stage's push block is declared at its own byte offset, so the two
 * vkCmdPushConstants calls carry different 16-byte payloads, name different
 * stages, and must not disturb each other. Then one vkCmdDraw, and the
 * readback has to agree on both halves at once: where the triangle is says
 * which bytes the vertex stage read, what colour it is says which bytes the
 * fragment stage read.
 *
 *    push VERTEX   at offset  0 = vec4(-0.5, -0.5, 0, 0)
 *    push FRAGMENT at offset 16 = vec4( 0.5,  0.5, 1, 1)
 *
 *    vertex read bytes  0..15  triangle at pixels (8,8) (24,8) (16,24)
 *    vertex read bytes 16..31  triangle at pixels (40,40) (56,40) (48,56)
 *    vertex read nothing       triangle at pixels (24,24) (40,24) (32,40)
 *    fragment read bytes 16..31   RGBA 128,128,255,255
 *    fragment read bytes  0..15   RGBA 0,0,0,0 -- the negatives clamp to black
 *
 * The three possible positions are disjoint, so one probe each plus a bounding
 * box over every lit pixel separates them, and the colour is checked on every
 * lit pixel rather than on one sample. A driver that flattens both stages'
 * blocks onto one 16-byte window fails on the colour; one that ignores the
 * per-stage split and lets the second push overwrite the first fails on the
 * position.
 *
 * The obvious sharper test -- push 16 bytes at offset 0 for the vertex stage
 * and 16 different bytes at offset 0 again for the fragment stage -- is not
 * written here because it is not legal Vulkan.  Push constants are one block
 * of memory that the ranges carve up per stage, not per-stage storage, so
 * VUID-vkCmdPushConstants-offset-01796 requires that a push touching a byte
 * name every stage of every range overlapping that byte. Both lavapipe and
 * NVIDIA report that VUID for the overlapping form and both then render the
 * geometry at the *colour's* position, because the second push overwrote the
 * first. Independence is only ever between disjoint byte ranges.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, so this runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. SPIR-V is embedded; nothing is read from disk.
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- ps.vert --
 *   #version 450
 *   layout(push_constant) uniform Push { layout(offset = 0) vec4 offset; } vsp;
 *   const vec2 base[3] = vec2[3](vec2(-0.25, -0.25),
 *                                vec2( 0.25, -0.25),
 *                                vec2( 0.00,  0.25));
 *   void main()
 *   {
 *      gl_Position = vec4(base[gl_VertexIndex] + vsp.offset.xy, 0.0, 1.0);
 *   }
 *
 *   -- ps.frag --
 *   #version 450
 *   layout(push_constant) uniform Push { layout(offset = 16) vec4 colour; } fsp;
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      out_colour = fsp.colour;
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

#define VS_OFFSET 0
#define FS_OFFSET 16

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* ps.vert */
static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000030u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000000u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x0000001bu, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
   0x00060005u, 0x0000000bu, 0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u,
   0x00060006u, 0x0000000bu, 0x00000000u, 0x505f6c67u, 0x7469736fu, 0x006e6f69u,
   0x00070006u, 0x0000000bu, 0x00000001u, 0x505f6c67u, 0x746e696fu, 0x657a6953u,
   0x00000000u, 0x00070006u, 0x0000000bu, 0x00000002u, 0x435f6c67u, 0x4470696cu,
   0x61747369u, 0x0065636eu, 0x00070006u, 0x0000000bu, 0x00000003u, 0x435f6c67u,
   0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u, 0x0000000du, 0x00000000u,
   0x00060005u, 0x0000001bu, 0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u,
   0x00050005u, 0x0000001eu, 0x65646e69u, 0x6c626178u, 0x00000065u, 0x00040005u,
   0x00000022u, 0x68737550u, 0x00000000u, 0x00050006u, 0x00000022u, 0x00000000u,
   0x7366666fu, 0x00007465u, 0x00030005u, 0x00000024u, 0x00707376u, 0x00030047u,
   0x0000000bu, 0x00000002u, 0x00050048u, 0x0000000bu, 0x00000000u, 0x0000000bu,
   0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u, 0x0000000bu, 0x00000001u,
   0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u,
   0x0000000bu, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00040047u, 0x0000001bu,
   0x0000000bu, 0x0000002au, 0x00030047u, 0x00000022u, 0x00000002u, 0x00050048u,
   0x00000022u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u, 0x00000009u, 0x00000001u,
   0x0004001cu, 0x0000000au, 0x00000006u, 0x00000009u, 0x0006001eu, 0x0000000bu,
   0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu,
   0x00000003u, 0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u,
   0x00040015u, 0x0000000eu, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu,
   0x0000000fu, 0x00000000u, 0x00040017u, 0x00000010u, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000008u, 0x00000011u, 0x00000003u, 0x0004001cu, 0x00000012u,
   0x00000010u, 0x00000011u, 0x0004002bu, 0x00000006u, 0x00000013u, 0xbe800000u,
   0x0005002cu, 0x00000010u, 0x00000014u, 0x00000013u, 0x00000013u, 0x0004002bu,
   0x00000006u, 0x00000015u, 0x3e800000u, 0x0005002cu, 0x00000010u, 0x00000016u,
   0x00000015u, 0x00000013u, 0x0004002bu, 0x00000006u, 0x00000017u, 0x00000000u,
   0x0005002cu, 0x00000010u, 0x00000018u, 0x00000017u, 0x00000015u, 0x0006002cu,
   0x00000012u, 0x00000019u, 0x00000014u, 0x00000016u, 0x00000018u, 0x00040020u,
   0x0000001au, 0x00000001u, 0x0000000eu, 0x0004003bu, 0x0000001au, 0x0000001bu,
   0x00000001u, 0x00040020u, 0x0000001du, 0x00000007u, 0x00000012u, 0x00040020u,
   0x0000001fu, 0x00000007u, 0x00000010u, 0x0003001eu, 0x00000022u, 0x00000007u,
   0x00040020u, 0x00000023u, 0x00000009u, 0x00000022u, 0x0004003bu, 0x00000023u,
   0x00000024u, 0x00000009u, 0x00040020u, 0x00000025u, 0x00000009u, 0x00000007u,
   0x0004002bu, 0x00000006u, 0x0000002au, 0x3f800000u, 0x00040020u, 0x0000002eu,
   0x00000003u, 0x00000007u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000001du, 0x0000001eu,
   0x00000007u, 0x0004003du, 0x0000000eu, 0x0000001cu, 0x0000001bu, 0x0003003eu,
   0x0000001eu, 0x00000019u, 0x00050041u, 0x0000001fu, 0x00000020u, 0x0000001eu,
   0x0000001cu, 0x0004003du, 0x00000010u, 0x00000021u, 0x00000020u, 0x00050041u,
   0x00000025u, 0x00000026u, 0x00000024u, 0x0000000fu, 0x0004003du, 0x00000007u,
   0x00000027u, 0x00000026u, 0x0007004fu, 0x00000010u, 0x00000028u, 0x00000027u,
   0x00000027u, 0x00000000u, 0x00000001u, 0x00050081u, 0x00000010u, 0x00000029u,
   0x00000021u, 0x00000028u, 0x00050051u, 0x00000006u, 0x0000002bu, 0x00000029u,
   0x00000000u, 0x00050051u, 0x00000006u, 0x0000002cu, 0x00000029u, 0x00000001u,
   0x00070050u, 0x00000007u, 0x0000002du, 0x0000002bu, 0x0000002cu, 0x00000017u,
   0x0000002au, 0x00050041u, 0x0000002eu, 0x0000002fu, 0x0000000du, 0x0000000fu,
   0x0003003eu, 0x0000002fu, 0x0000002du, 0x000100fdu, 0x00010038u,
};

/* ps.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000012u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00040005u, 0x0000000au, 0x68737550u, 0x00000000u, 0x00050006u,
   0x0000000au, 0x00000000u, 0x6f6c6f63u, 0x00007275u, 0x00030005u, 0x0000000cu,
   0x00707366u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00030047u,
   0x0000000au, 0x00000002u, 0x00050048u, 0x0000000au, 0x00000000u, 0x00000023u,
   0x00000010u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u,
   0x00000004u, 0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu,
   0x00000008u, 0x00000009u, 0x00000003u, 0x0003001eu, 0x0000000au, 0x00000007u,
   0x00040020u, 0x0000000bu, 0x00000009u, 0x0000000au, 0x0004003bu, 0x0000000bu,
   0x0000000cu, 0x00000009u, 0x00040015u, 0x0000000du, 0x00000020u, 0x00000001u,
   0x0004002bu, 0x0000000du, 0x0000000eu, 0x00000000u, 0x00040020u, 0x0000000fu,
   0x00000009u, 0x00000007u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x00050041u, 0x0000000fu, 0x00000010u,
   0x0000000cu, 0x0000000eu, 0x0004003du, 0x00000007u, 0x00000011u, 0x00000010u,
   0x0003003eu, 0x00000009u, 0x00000011u, 0x000100fdu, 0x00010038u,
};

/* The vertex stage's block, bytes 0..15: only .xy is used, as an offset. */
static const float vs_push[4] = { -0.5f, -0.5f, 0.0f, 0.0f };
/* The fragment stage's block, bytes 16..31: the output colour. Read as an
 * offset it would move the triangle; the vertex values read as a colour clamp
 * to transparent black. Neither mistake can hide. */
static const float fs_push[4] = {  0.5f,  0.5f, 1.0f, 1.0f };

static const unsigned char clear_rgba[4]    = {  26,  26,  38, 255 };
static const unsigned char expect_rgba[4]   = { 128, 128, 255, 255 };

/* Centroids of the three triangles the draw could produce, in pixels. */
#define PROBE_VERTEX_OWN_X   16
#define PROBE_VERTEX_OWN_Y   13
#define PROBE_VERTEX_STOLE_X 48
#define PROBE_VERTEX_STOLE_Y 45
#define PROBE_VERTEX_NONE_X  32
#define PROBE_VERTEX_NONE_Y  29

/* Bounding box of the correct triangle, half open. */
#define BOX_X0 8
#define BOX_X1 24
#define BOX_Y0 8
#define BOX_Y1 24
/* 0.5 * 16 * 16 pixels, before fill-rule rounding. */
#define AREA_LO 100
#define AREA_HI 160

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

static int
same(const unsigned char *px, const unsigned char *want)
{
   for (int i = 0; i < 4; i++) {
      int d = (int)px[i] - (int)want[i];
      if (d < -2 || d > 2)
         return 0;
   }
   return 1;
}

static void
describe(const char *what, int x, int y, const unsigned char *px)
{
   printf("  %-28s (%2d,%2d) = %3u %3u %3u %3u\n", what, x, y,
          px[0], px[1], px[2], px[3]);
}

int
main(int argc, char **argv)
{
   const char *ppm = argc > 1 ? argv[1] : NULL;

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
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }
   free(families);
   if (family == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 1; }

   /* Dynamic rendering is core from 1.3, but the native driver reports 1.0 and
    * means it, so ask for the extension when it is advertised. */
   uint32_t ext_count = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, NULL));
   VkExtensionProperties *exts = calloc(ext_count, sizeof(*exts));
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, exts));
   int have_dynrend = 0;
   for (uint32_t i = 0; i < ext_count; i++)
      if (!strcmp(exts[i].extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
         have_dynrend = 1;
   free(exts);
   if (!have_dynrend && pprops.apiVersion < VK_API_VERSION_1_3) {
      fprintf(stderr, "no dynamic rendering\n");
      return 1;
   }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &prio };
   const char *dev_exts[] = { VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME };
   VkPhysicalDeviceDynamicRenderingFeatures dyn = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &dyn,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
      .enabledExtensionCount = have_dynrend ? 1 : 0,
      .ppEnabledExtensionNames = dev_exts };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   PFN_vkCmdBeginRenderingKHR begin_rendering =
      (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR");
   PFN_vkCmdEndRenderingKHR end_rendering =
      (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR");
   if (!begin_rendering || !end_rendering) {
      fprintf(stderr, "dynamic rendering entrypoints missing\n");
      return 1;
   }

   /* Optimal tiling and a copy back through a buffer, so that a device which
    * cannot render into linear host memory still runs this. */
   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkImageCreateInfo imgi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = cfmt,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   CHECK(vkCreateImage(dev, &imgi, NULL, &img));
   VkMemoryRequirements ireq;
   vkGetImageMemoryRequirements(dev, img, &ireq);
   uint32_t itype = pick_memory(pdev, ireq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (itype == UINT32_MAX)
      itype = pick_memory(pdev, ireq.memoryTypeBits, 0);
   if (itype == UINT32_MAX) { fprintf(stderr, "no image memory type\n"); return 1; }
   VkMemoryAllocateInfo imai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = ireq.size,
                                 .memoryTypeIndex = itype };
   VkDeviceMemory imem;
   CHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   CHECK(vkBindImageMemory(dev, img, imem, 0));

   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = cfmt,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &vci, NULL, &view));

   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = (VkDeviceSize)W * H * 4,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   VkBuffer readback;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &readback));
   VkMemoryRequirements breq;
   vkGetBufferMemoryRequirements(dev, readback, &breq);
   uint32_t btype = pick_memory(pdev, breq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (btype == UINT32_MAX) { fprintf(stderr, "no host-visible type\n"); return 1; }
   VkMemoryAllocateInfo bmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = breq.size,
                                 .memoryTypeIndex = btype };
   VkDeviceMemory bmem;
   CHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   CHECK(vkBindBufferMemory(dev, readback, bmem, 0));

   VkShaderModuleCreateInfo vsmi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv), .pCode = vert_spv };
   VkShaderModule vs;
   CHECK(vkCreateShaderModule(dev, &vsmi, NULL, &vs));
   VkShaderModuleCreateInfo fsmi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(frag_spv), .pCode = frag_spv };
   VkShaderModule fs;
   CHECK(vkCreateShaderModule(dev, &fsmi, NULL, &fs));

   /*
    * The point of the test: two ranges over the same bytes, one per stage.
    * Declaring both is what makes each vkCmdPushConstants below legal.
    */
   VkPushConstantRange ranges[2] = {
      { .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = VS_OFFSET, .size = 16 },
      { .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = FS_OFFSET, .size = 16 },
   };
   if (pprops.limits.maxPushConstantsSize < FS_OFFSET + 16) {
      fprintf(stderr, "maxPushConstantsSize %u < %d\n",
              pprops.limits.maxPushConstantsSize, FS_OFFSET + 16);
      return 1;
   }
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 2, .pPushConstantRanges = ranges };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkViewport vp = { 0, 0, W, H, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   /*
    * Viewport and scissor are dynamic. The native driver rasterizes nothing
    * from a pipeline's static viewport, so they have to come from
    * vkCmdSetViewport/vkCmdSetScissor -- and declaring them here is what makes
    * those calls legal: VUID-vkCmdDraw-None-08608 forbids setting dynamic
    * state that the bound pipeline specified statically.
    */
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1 };
   VkDynamicState dynamic[2] = { VK_DYNAMIC_STATE_VIEWPORT,
                                 VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dyns = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dynamic };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
   VkPipelineRenderingCreateInfo pri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt };
   VkGraphicsPipelineCreateInfo gpi = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &pri, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .pDynamicState = &dyns, .layout = layout };
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe));

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

   VkImageMemoryBarrier to_colour = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_colour);

   VkRenderingAttachmentInfo at = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = { 0.1f, 0.1f, 0.15f, 1.0f } };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &at };

   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &sc);

   /*
    * Two pushes, two stages, two disjoint 16-byte windows. The fragment push
    * is second, so a driver that ignores either the stage or the byte offset
    * shows it: the geometry moves, or the colour goes black.
    */
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, VS_OFFSET,
                      (uint32_t)sizeof(vs_push), vs_push);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, FS_OFFSET,
                      (uint32_t)sizeof(fs_push), fs_push);

   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   VkImageMemoryBarrier to_src = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_src);

   VkBufferImageCopy copy = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback, 1, &copy);

   VkBufferMemoryBarrier to_host = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback, .size = VK_WHOLE_SIZE };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0,
                        0, NULL, 1, &to_host, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   unsigned char *px = NULL;
   CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&px));
   VkMappedMemoryRange invalidate = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = bmem, .size = VK_WHOLE_SIZE };
   CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &invalidate));

   unsigned lit = 0, wrong_colour = 0, outside = 0;
   int minx = W, maxx = -1, miny = H, maxy = -1;
   for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
         const unsigned char *p = px + ((size_t)y * W + x) * 4;
         if (same(p, clear_rgba))
            continue;
         lit++;
         if (!same(p, expect_rgba))
            wrong_colour++;
         if (x < minx) minx = x;
         if (x > maxx) maxx = x;
         if (y < miny) miny = y;
         if (y > maxy) maxy = y;
         if (x < BOX_X0 || x >= BOX_X1 || y < BOX_Y0 || y >= BOX_Y1)
            outside++;
      }
   }

   const unsigned char *own   = px + ((size_t)PROBE_VERTEX_OWN_Y * W +
                                      PROBE_VERTEX_OWN_X) * 4;
   const unsigned char *stole = px + ((size_t)PROBE_VERTEX_STOLE_Y * W +
                                      PROBE_VERTEX_STOLE_X) * 4;
   const unsigned char *none  = px + ((size_t)PROBE_VERTEX_NONE_Y * W +
                                      PROBE_VERTEX_NONE_X) * 4;

   printf("  push VERTEX   at offset %2d = %.2f %.2f %.2f %.2f\n", VS_OFFSET,
          vs_push[0], vs_push[1], vs_push[2], vs_push[3]);
   printf("  push FRAGMENT at offset %2d = %.2f %.2f %.2f %.2f\n", FS_OFFSET,
          fs_push[0], fs_push[1], fs_push[2], fs_push[3]);
   describe("vertex read bytes 0..15", PROBE_VERTEX_OWN_X,
            PROBE_VERTEX_OWN_Y, own);
   describe("vertex read bytes 16..31", PROBE_VERTEX_STOLE_X,
            PROBE_VERTEX_STOLE_Y, stole);
   describe("vertex read nothing", PROBE_VERTEX_NONE_X,
            PROBE_VERTEX_NONE_Y, none);
   printf("  %u lit pixels, %u off colour, %u outside the box, "
          "bbox x %d..%d y %d..%d\n",
          lit, wrong_colour, outside, minx, maxx, miny, maxy);

   int fail = 0;
   if (!same(own, expect_rgba)) {
      printf("FAIL the triangle is not where the vertex push puts it, "
             "or it is the wrong colour\n");
      fail = 1;
   }
   if (!same(stole, clear_rgba)) {
      printf("FAIL the vertex stage read the fragment stage's bytes\n");
      fail = 1;
   }
   if (!same(none, clear_rgba)) {
      printf("FAIL the vertex stage read no push constants at all\n");
      fail = 1;
   }
   if (wrong_colour) {
      printf("FAIL %u lit pixels are not the fragment stage's colour "
             "(black means it read the vertex stage's bytes)\n",
             wrong_colour);
      fail = 1;
   }
   if (outside) {
      printf("FAIL %u lit pixels fall outside the expected triangle\n", outside);
      fail = 1;
   }
   if (lit < AREA_LO || lit > AREA_HI) {
      printf("FAIL %u lit pixels, expected %d..%d\n", lit, AREA_LO, AREA_HI);
      fail = 1;
   }
   if (!fail)
      printf("PASS the vertex stage kept bytes 0..15 and the fragment stage "
             "kept bytes 16..31 across one draw\n");

   if (ppm) {
      FILE *f = fopen(ppm, "wb");
      if (f) {
         fprintf(f, "P6\n%d %d\n255\n", W, H);
         for (int i = 0; i < W * H; i++) {
            fputc(px[i * 4 + 0], f);
            fputc(px[i * 4 + 1], f);
            fputc(px[i * 4 + 2], f);
         }
         fclose(f);
         printf("  %s written\n", ppm);
      }
   }

   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fail;
}
