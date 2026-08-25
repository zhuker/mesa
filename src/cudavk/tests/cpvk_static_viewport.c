/*
 * Static viewport and scissor: a pipeline that names its viewport and its
 * scissor in VkPipelineViewportStateCreateInfo, declares neither of them
 * dynamic, and is drawn with no vkCmdSetViewport and no vkCmdSetScissor at
 * all. The pixels have to land where the pipeline said.
 *
 * Every other cpvk test takes the easy road -- VK_DYNAMIC_STATE_VIEWPORT and
 * VK_DYNAMIC_STATE_SCISSOR plus vkCmdSetViewport/vkCmdSetScissor -- so nothing
 * in the suite notices a driver that carries the pipeline's pViewports and
 * pScissors around and never reads them. This one takes the other road and
 * only that road: if the static state is dropped the picture is wrong, because
 * there is no command to fall back on.
 *
 * The target is 64x64 and the static viewport is the middle 32x32 of it, so
 * the viewport transform is not the identity in either offset or scale:
 *
 *    viewport   x 16 y 16   w 32 h 32
 *
 * One triangle at NDC (-0.5,-0.5) (0.5,-0.5) (-0.5,0.5), which that viewport
 * puts at pixels (24,24) (40,24) (24,40) -- a right triangle filling the
 * upper-left half of the box x 24..39, y 24..39. It is well inside the
 * viewport rectangle, so nothing here depends on clipping at the viewport
 * edge, only on the transform. Ignoring the viewport and rasterizing through
 * the whole framebuffer instead would put the same triangle at (16,16)
 * (48,16) (16,48): four times the area, and a bounding box that starts eight
 * pixels earlier, which is what the probe at (18,18) watches for.
 *
 * Then the same triangle again through a second pipeline that differs in one
 * field: its static scissor is the 8x8 box at (24,24) instead of the whole
 * framebuffer. That box lies wholly inside the triangle, so a scissor that is
 * obeyed leaves exactly 64 lit pixels and a scissor that is dropped leaves the
 * whole triangle. The two draws go to the same image, one after the other,
 * each cleared and copied out on its own.
 *
 *    draw 1  viewport 16,16 32x32   scissor 0,0 64x64   ~128 px, box 24..39
 *    draw 2  viewport 16,16 32x32   scissor 24,24 8x8     64 px, box 24..31
 *
 * Both pipelines pass pDynamicState = NULL. That is what makes the absence of
 * vkCmdSetViewport/vkCmdSetScissor legal rather than undefined:
 * VUID-vkCmdDraw-None-07831 and -07832 only require those commands for state a
 * pipeline declared dynamic, and VUID-vkCmdDraw-None-08608 forbids them for
 * state a pipeline declared statically. Static and dynamic are exclusive, and
 * this test is the static half.
 *
 * Draw 1's lit-pixel count is given a range rather than a number, because its
 * hypotenuse runs exactly through pixel centres and the fill rule decides
 * sixteen of them. Its bounding box is not a range: the two edges that set the
 * box origin are axis aligned and half a pixel from the nearest centre, so a
 * correct viewport transform puts minx and miny at 24 exactly. Draw 2 is exact
 * in both, having no triangle edge inside its scissor box.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, so this runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. SPIR-V is embedded; nothing is read from disk.
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- sv.vert --
 *   #version 450
 *   const vec2 pos[3] = vec2[3](vec2(-0.5, -0.5),
 *                               vec2( 0.5, -0.5),
 *                               vec2(-0.5,  0.5));
 *   void main()
 *   {
 *      gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
 *   }
 *
 *   -- sv.frag --
 *   #version 450
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      out_colour = vec4(0.2, 0.9, 0.4, 1.0);
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/* The static viewport: the middle 32x32 of the target. */
#define VP_X 16
#define VP_Y 16
#define VP_W 32
#define VP_H 32

/* The static scissor of the second pipeline, wholly inside the triangle. */
#define SC_X 24
#define SC_Y 24
#define SC_W  8
#define SC_H  8

/* Where that viewport puts NDC -0.5 and +0.5, as a half open pixel box. */
#define TRI_X0 (VP_X + VP_W / 4)
#define TRI_X1 (VP_X + (3 * VP_W) / 4)
#define TRI_Y0 (VP_Y + VP_H / 4)
#define TRI_Y1 (VP_Y + (3 * VP_H) / 4)

/* Half of a 16x16 box, before the fill rule decides the hypotenuse. */
#define AREA_LO 110
#define AREA_HI 145

/* Inside the triangle and inside the small scissor: lit by both draws. */
#define PROBE_KEPT_X 27
#define PROBE_KEPT_Y 27
/* Inside the triangle and outside the small scissor: draw 2 must lose it. */
#define PROBE_CLIPPED_X 34
#define PROBE_CLIPPED_Y 26
/* Inside the triangle only if the static viewport was ignored: never lit. */
#define PROBE_NO_VIEWPORT_X 18
#define PROBE_NO_VIEWPORT_Y 18

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* sv.vert */
static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000028u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000000u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x0000001au, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
   0x00060005u, 0x0000000bu, 0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u,
   0x00060006u, 0x0000000bu, 0x00000000u, 0x505f6c67u, 0x7469736fu, 0x006e6f69u,
   0x00070006u, 0x0000000bu, 0x00000001u, 0x505f6c67u, 0x746e696fu, 0x657a6953u,
   0x00000000u, 0x00070006u, 0x0000000bu, 0x00000002u, 0x435f6c67u, 0x4470696cu,
   0x61747369u, 0x0065636eu, 0x00070006u, 0x0000000bu, 0x00000003u, 0x435f6c67u,
   0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u, 0x0000000du, 0x00000000u,
   0x00060005u, 0x0000001au, 0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u,
   0x00050005u, 0x0000001du, 0x65646e69u, 0x6c626178u, 0x00000065u, 0x00030047u,
   0x0000000bu, 0x00000002u, 0x00050048u, 0x0000000bu, 0x00000000u, 0x0000000bu,
   0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u, 0x0000000bu, 0x00000001u,
   0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u,
   0x0000000bu, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00040047u, 0x0000001au,
   0x0000000bu, 0x0000002au, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
   0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
   0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u,
   0x0004002bu, 0x00000008u, 0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au,
   0x00000006u, 0x00000009u, 0x0006001eu, 0x0000000bu, 0x00000007u, 0x00000006u,
   0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000003u, 0x0000000bu,
   0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u, 0x00040015u, 0x0000000eu,
   0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u,
   0x00040017u, 0x00000010u, 0x00000006u, 0x00000002u, 0x0004002bu, 0x00000008u,
   0x00000011u, 0x00000003u, 0x0004001cu, 0x00000012u, 0x00000010u, 0x00000011u,
   0x0004002bu, 0x00000006u, 0x00000013u, 0xbf000000u, 0x0005002cu, 0x00000010u,
   0x00000014u, 0x00000013u, 0x00000013u, 0x0004002bu, 0x00000006u, 0x00000015u,
   0x3f000000u, 0x0005002cu, 0x00000010u, 0x00000016u, 0x00000015u, 0x00000013u,
   0x0005002cu, 0x00000010u, 0x00000017u, 0x00000013u, 0x00000015u, 0x0006002cu,
   0x00000012u, 0x00000018u, 0x00000014u, 0x00000016u, 0x00000017u, 0x00040020u,
   0x00000019u, 0x00000001u, 0x0000000eu, 0x0004003bu, 0x00000019u, 0x0000001au,
   0x00000001u, 0x00040020u, 0x0000001cu, 0x00000007u, 0x00000012u, 0x00040020u,
   0x0000001eu, 0x00000007u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000021u,
   0x00000000u, 0x0004002bu, 0x00000006u, 0x00000022u, 0x3f800000u, 0x00040020u,
   0x00000026u, 0x00000003u, 0x00000007u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000001cu,
   0x0000001du, 0x00000007u, 0x0004003du, 0x0000000eu, 0x0000001bu, 0x0000001au,
   0x0003003eu, 0x0000001du, 0x00000018u, 0x00050041u, 0x0000001eu, 0x0000001fu,
   0x0000001du, 0x0000001bu, 0x0004003du, 0x00000010u, 0x00000020u, 0x0000001fu,
   0x00050051u, 0x00000006u, 0x00000023u, 0x00000020u, 0x00000000u, 0x00050051u,
   0x00000006u, 0x00000024u, 0x00000020u, 0x00000001u, 0x00070050u, 0x00000007u,
   0x00000025u, 0x00000023u, 0x00000024u, 0x00000021u, 0x00000022u, 0x00050041u,
   0x00000026u, 0x00000027u, 0x0000000du, 0x0000000fu, 0x0003003eu, 0x00000027u,
   0x00000025u, 0x000100fdu, 0x00010038u
};

/* sv.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000000fu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00020013u,
   0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u,
   0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u,
   0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u,
   0x00000003u, 0x0004002bu, 0x00000006u, 0x0000000au, 0x3e4ccccdu, 0x0004002bu,
   0x00000006u, 0x0000000bu, 0x3f666666u, 0x0004002bu, 0x00000006u, 0x0000000cu,
   0x3ecccccdu, 0x0004002bu, 0x00000006u, 0x0000000du, 0x3f800000u, 0x0007002cu,
   0x00000007u, 0x0000000eu, 0x0000000au, 0x0000000bu, 0x0000000cu, 0x0000000du,
   0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
   0x00000005u, 0x0003003eu, 0x00000009u, 0x0000000eu, 0x000100fdu, 0x00010038u
};

static const unsigned char clear_rgba[4]  = {  26,  26,  38, 255 };
static const unsigned char expect_rgba[4] = {  51, 230, 102, 255 };

/* What one readback is worth looking at. */
struct scan {
   unsigned lit;          /* pixels that are not the clear colour */
   unsigned wrong_colour; /* of those, ones that are not the shader's colour */
   unsigned outside;      /* of those, ones outside the expected box */
   int minx, maxx, miny, maxy;
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

/* Every lit pixel of one image, against the half open box it should be in. */
static void
scan_image(const unsigned char *px, int x0, int x1, int y0, int y1,
           struct scan *s)
{
   s->lit = s->wrong_colour = s->outside = 0;
   s->minx = W; s->maxx = -1; s->miny = H; s->maxy = -1;
   for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
         const unsigned char *p = px + ((size_t)y * W + x) * 4;
         if (same(p, clear_rgba))
            continue;
         s->lit++;
         if (!same(p, expect_rgba))
            s->wrong_colour++;
         if (x < s->minx) s->minx = x;
         if (x > s->maxx) s->maxx = x;
         if (y < s->miny) s->miny = y;
         if (y > s->maxy) s->maxy = y;
         if (x < x0 || x >= x1 || y < y0 || y >= y1)
            s->outside++;
      }
   }
}

static void
describe(const char *what, int x, int y, const unsigned char *px)
{
   printf("  %-30s (%2d,%2d) = %3u %3u %3u %3u\n", what, x, y,
          px[0], px[1], px[2], px[3]);
}

static const unsigned char *
pixel(const unsigned char *px, int x, int y)
{
   return px + ((size_t)y * W + x) * 4;
}

static void
write_ppm(const char *path, const unsigned char *px)
{
   FILE *f = fopen(path, "wb");
   if (!f)
      return;
   fprintf(f, "P6\n%d %d\n255\n", W, H);
   for (int i = 0; i < W * H; i++) {
      fputc(px[i * 4 + 0], f);
      fputc(px[i * 4 + 1], f);
      fputc(px[i * 4 + 2], f);
   }
   fclose(f);
   printf("  %s written\n", path);
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

   /*
    * Dynamic rendering is core from 1.3, but the native driver reports less
    * than that and means it, so ask for the extension when it is advertised --
    * and name its dependency chain as well, because on a device below 1.3
    * nothing else supplies it and the validation layer refuses the device
    * otherwise: VUID-vkCreateDevice-ppEnabledExtensionNames-01387. Above 1.3
    * the chain is core, so only the extension itself is asked for, which is
    * what keeps vkGetDeviceProcAddr of the KHR entrypoints legal everywhere.
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
      .ppEnabledExtensionNames = dev_exts };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   /* The KHR entrypoints exist when the extension was enabled above; a 1.3
    * device that does not advertise it at all still has the core ones. */
   PFN_vkCmdBeginRenderingKHR begin_rendering =
      (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(dev, have_dynrend ?
         "vkCmdBeginRenderingKHR" : "vkCmdBeginRendering");
   PFN_vkCmdEndRenderingKHR end_rendering =
      (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(dev, have_dynrend ?
         "vkCmdEndRenderingKHR" : "vkCmdEndRendering");
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

   /* One buffer, two frames: the wide-scissor draw then the clipped one. */
   const VkDeviceSize frame_bytes = (VkDeviceSize)W * H * 4;
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = 2 * frame_bytes,
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

   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
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

   /*
    * The whole point. Both pipelines carry a viewport and a scissor here and
    * name no dynamic state at all, so these two structures are the only place
    * the rasterizer can learn where to put the triangle: there is no
    * vkCmdSetViewport and no vkCmdSetScissor anywhere below.
    */
   VkViewport vp = { .x = VP_X, .y = VP_Y, .width = VP_W, .height = VP_H,
                     .minDepth = 0.0f, .maxDepth = 1.0f };
   VkRect2D wide_scissor = { { 0, 0 }, { W, H } };
   VkRect2D tight_scissor = { { SC_X, SC_Y }, { SC_W, SC_H } };
   VkPipelineViewportStateCreateInfo wide_vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &wide_scissor };
   VkPipelineViewportStateCreateInfo tight_vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &tight_scissor };

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

   /* Identical but for pViewportState, and pDynamicState is NULL in both. */
   VkGraphicsPipelineCreateInfo gpi[2] = {
      { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pri, .stageCount = 2, .pStages = stages,
        .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &wide_vps, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds,
        .pColorBlendState = &cb, .pDynamicState = NULL, .layout = layout },
      { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pri, .stageCount = 2, .pStages = stages,
        .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &tight_vps, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds,
        .pColorBlendState = &cb, .pDynamicState = NULL, .layout = layout },
   };
   VkPipeline pipes[2];
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 2, gpi, NULL, pipes));

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
   VkImageMemoryBarrier back_to_colour = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };

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

   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_colour);

   for (int pass = 0; pass < 2; pass++) {
      if (pass)
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                              0, NULL, 0, NULL, 1, &back_to_colour);

      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[pass]);
      /* No vkCmdSetViewport. No vkCmdSetScissor. That is the test. */
      vkCmdDraw(cmd, 3, 1, 0, 0);
      end_rendering(cmd);

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           0, NULL, 0, NULL, 1, &to_src);

      VkBufferImageCopy copy = {
         .bufferOffset = (VkDeviceSize)pass * frame_bytes,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { W, H, 1 } };
      vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback, 1, &copy);
   }

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

   unsigned char *mapped = NULL;
   CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   VkMappedMemoryRange invalidate = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = bmem, .size = VK_WHOLE_SIZE };
   CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &invalidate));

   const unsigned char *wide = mapped;
   const unsigned char *tight = mapped + frame_bytes;

   struct scan ws, ts;
   scan_image(wide, TRI_X0, TRI_X1, TRI_Y0, TRI_Y1, &ws);
   scan_image(tight, SC_X, SC_X + SC_W, SC_Y, SC_Y + SC_H, &ts);

   printf("  static viewport   x %d y %d  %dx%d\n", VP_X, VP_Y, VP_W, VP_H);
   printf("  static scissor    draw 1  x 0 y 0  %dx%d\n", W, H);
   printf("  static scissor    draw 2  x %d y %d  %dx%d\n",
          SC_X, SC_Y, SC_W, SC_H);
   printf("  draw 1: %u lit, %u off colour, %u outside x %d..%d y %d..%d, "
          "bbox x %d..%d y %d..%d\n",
          ws.lit, ws.wrong_colour, ws.outside,
          TRI_X0, TRI_X1 - 1, TRI_Y0, TRI_Y1 - 1,
          ws.minx, ws.maxx, ws.miny, ws.maxy);
   printf("  draw 2: %u lit, %u off colour, %u outside x %d..%d y %d..%d, "
          "bbox x %d..%d y %d..%d\n",
          ts.lit, ts.wrong_colour, ts.outside,
          SC_X, SC_X + SC_W - 1, SC_Y, SC_Y + SC_H - 1,
          ts.minx, ts.maxx, ts.miny, ts.maxy);
   describe("draw 1 inside the scissor box",
            PROBE_KEPT_X, PROBE_KEPT_Y, pixel(wide, PROBE_KEPT_X, PROBE_KEPT_Y));
   describe("draw 2 inside the scissor box",
            PROBE_KEPT_X, PROBE_KEPT_Y, pixel(tight, PROBE_KEPT_X, PROBE_KEPT_Y));
   describe("draw 1 outside it",
            PROBE_CLIPPED_X, PROBE_CLIPPED_Y,
            pixel(wide, PROBE_CLIPPED_X, PROBE_CLIPPED_Y));
   describe("draw 2 outside it",
            PROBE_CLIPPED_X, PROBE_CLIPPED_Y,
            pixel(tight, PROBE_CLIPPED_X, PROBE_CLIPPED_Y));
   describe("draw 1 where no viewport puts it",
            PROBE_NO_VIEWPORT_X, PROBE_NO_VIEWPORT_Y,
            pixel(wide, PROBE_NO_VIEWPORT_X, PROBE_NO_VIEWPORT_Y));

   int fail = 0;

   /* Draw 1: the static viewport placed and scaled the triangle. */
   if (ws.lit < AREA_LO || ws.lit > AREA_HI) {
      printf("FAIL draw 1 lit %u pixels, expected %d..%d -- the static "
             "viewport did not scale the triangle\n",
             ws.lit, AREA_LO, AREA_HI);
      fail = 1;
   }
   if (ws.outside) {
      printf("FAIL draw 1 put %u pixels outside x %d..%d y %d..%d -- the "
             "static viewport did not place the triangle\n",
             ws.outside, TRI_X0, TRI_X1 - 1, TRI_Y0, TRI_Y1 - 1);
      fail = 1;
   }
   if (ws.minx != TRI_X0 || ws.miny != TRI_Y0) {
      printf("FAIL draw 1 bbox starts at (%d,%d), expected (%d,%d)\n",
             ws.minx, ws.miny, TRI_X0, TRI_Y0);
      fail = 1;
   }
   /* The hypotenuse crosses centres, so the far edge may lose one column. */
   if (ws.maxx < TRI_X1 - 2 || ws.maxy < TRI_Y1 - 2) {
      printf("FAIL draw 1 bbox ends at (%d,%d), expected (%d,%d)\n",
             ws.maxx, ws.maxy, TRI_X1 - 1, TRI_Y1 - 1);
      fail = 1;
   }
   if (ws.wrong_colour) {
      printf("FAIL draw 1 has %u pixels that are neither the clear colour nor "
             "the shader's\n", ws.wrong_colour);
      fail = 1;
   }
   if (!same(pixel(wide, PROBE_KEPT_X, PROBE_KEPT_Y), expect_rgba)) {
      printf("FAIL draw 1 left (%d,%d) unlit\n", PROBE_KEPT_X, PROBE_KEPT_Y);
      fail = 1;
   }
   if (!same(pixel(wide, PROBE_NO_VIEWPORT_X, PROBE_NO_VIEWPORT_Y), clear_rgba)) {
      printf("FAIL draw 1 lit (%d,%d), which is only inside the triangle if "
             "the static viewport was ignored\n",
             PROBE_NO_VIEWPORT_X, PROBE_NO_VIEWPORT_Y);
      fail = 1;
   }

   /* Draw 2: the static scissor clipped it, and nothing else changed. */
   if (ts.lit != SC_W * SC_H) {
      printf("FAIL draw 2 lit %u pixels, expected exactly %d -- the static "
             "scissor did not clip\n", ts.lit, SC_W * SC_H);
      fail = 1;
   }
   if (ts.outside) {
      printf("FAIL draw 2 put %u pixels outside its %dx%d static scissor\n",
             ts.outside, SC_W, SC_H);
      fail = 1;
   }
   if (ts.minx != SC_X || ts.miny != SC_Y ||
       ts.maxx != SC_X + SC_W - 1 || ts.maxy != SC_Y + SC_H - 1) {
      printf("FAIL draw 2 bbox x %d..%d y %d..%d, expected x %d..%d y %d..%d\n",
             ts.minx, ts.maxx, ts.miny, ts.maxy,
             SC_X, SC_X + SC_W - 1, SC_Y, SC_Y + SC_H - 1);
      fail = 1;
   }
   if (ts.wrong_colour) {
      printf("FAIL draw 2 has %u pixels that are neither the clear colour nor "
             "the shader's\n", ts.wrong_colour);
      fail = 1;
   }
   if (!same(pixel(tight, PROBE_KEPT_X, PROBE_KEPT_Y), expect_rgba)) {
      printf("FAIL draw 2 lost (%d,%d), which its scissor keeps\n",
             PROBE_KEPT_X, PROBE_KEPT_Y);
      fail = 1;
   }
   if (!same(pixel(wide, PROBE_CLIPPED_X, PROBE_CLIPPED_Y), expect_rgba)) {
      printf("FAIL draw 1 lost (%d,%d), so draw 2 losing it proves nothing\n",
             PROBE_CLIPPED_X, PROBE_CLIPPED_Y);
      fail = 1;
   }
   if (!same(pixel(tight, PROBE_CLIPPED_X, PROBE_CLIPPED_Y), clear_rgba)) {
      printf("FAIL draw 2 kept (%d,%d), which is outside its static scissor\n",
             PROBE_CLIPPED_X, PROBE_CLIPPED_Y);
      fail = 1;
   }
   if (ts.lit >= ws.lit) {
      printf("FAIL the smaller static scissor did not remove anything: "
             "%u lit against %u\n", ts.lit, ws.lit);
      fail = 1;
   }

   if (!fail)
      printf("PASS the static viewport placed the triangle and the static "
             "scissor clipped it, with no vkCmdSetViewport or "
             "vkCmdSetScissor\n");

   if (ppm) {
      char clipped[1024];
      write_ppm(ppm, wide);
      snprintf(clipped, sizeof(clipped), "%.*s.clipped.ppm",
               (int)(sizeof(clipped) - 32), ppm);
      write_ppm(clipped, tight);
   }

   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipes[1], NULL);
   vkDestroyPipeline(dev, pipes[0], NULL);
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
