/*
 * Deterministic fixed-edge shared-ownership oracle.
 *
 * Two triangles form a 176x176 quad. Their shared diagonal passes through pixel
 * centres. Additive blending writes 1/4 per covered fragment, so the CPU Q8
 * oracle detects both cracks (0) and double ownership (1/2). A second render
 * uses a one-pixel scissor exactly on that diagonal.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 192
#define H 192
#define VP_X 0
#define VP_Y 0
#define VP_W 192
#define VP_H 192
#define SC_X 96
#define SC_Y 96
#define SC_W 1
#define SC_H 1
#define TRI_X0 8
#define TRI_X1 184
#define TRI_Y0 8
#define TRI_Y1 184

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* fixed.vert */
static const uint32_t vert_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x00000029, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
   0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x0000001b, 0x00030003,
   0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005, 0x0000000b,
   0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000000b, 0x00000000, 0x505f6c67,
   0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
   0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e,
   0x00070006, 0x0000000b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e, 0x00030005,
   0x0000000d, 0x00000000, 0x00060005, 0x0000001b, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865,
   0x00050005, 0x0000001e, 0x65646e69, 0x6c626178, 0x00000065, 0x00030047, 0x0000000b, 0x00000002,
   0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001,
   0x0000000b, 0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
   0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x0000001b, 0x0000000b, 0x0000002a,
   0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
   0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015, 0x00000008, 0x00000020, 0x00000000,
   0x0004002b, 0x00000008, 0x00000009, 0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009,
   0x0006001e, 0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c,
   0x00000003, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e,
   0x00000020, 0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040017, 0x00000010,
   0x00000006, 0x00000002, 0x0004002b, 0x00000008, 0x00000011, 0x00000006, 0x0004001c, 0x00000012,
   0x00000010, 0x00000011, 0x0004002b, 0x00000006, 0x00000013, 0xbf6aaaab, 0x0005002c, 0x00000010,
   0x00000014, 0x00000013, 0x00000013, 0x0004002b, 0x00000006, 0x00000015, 0x3f6aaaab, 0x0005002c,
   0x00000010, 0x00000016, 0x00000015, 0x00000013, 0x0005002c, 0x00000010, 0x00000017, 0x00000015,
   0x00000015, 0x0005002c, 0x00000010, 0x00000018, 0x00000013, 0x00000015, 0x0009002c, 0x00000012,
   0x00000019, 0x00000014, 0x00000016, 0x00000017, 0x00000014, 0x00000017, 0x00000018, 0x00040020,
   0x0000001a, 0x00000001, 0x0000000e, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000001, 0x00040020,
   0x0000001d, 0x00000007, 0x00000012, 0x00040020, 0x0000001f, 0x00000007, 0x00000010, 0x0004002b,
   0x00000006, 0x00000022, 0x00000000, 0x0004002b, 0x00000006, 0x00000023, 0x3f800000, 0x00040020,
   0x00000027, 0x00000003, 0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
   0x000200f8, 0x00000005, 0x0004003b, 0x0000001d, 0x0000001e, 0x00000007, 0x0004003d, 0x0000000e,
   0x0000001c, 0x0000001b, 0x0003003e, 0x0000001e, 0x00000019, 0x00050041, 0x0000001f, 0x00000020,
   0x0000001e, 0x0000001c, 0x0004003d, 0x00000010, 0x00000021, 0x00000020, 0x00050051, 0x00000006,
   0x00000024, 0x00000021, 0x00000000, 0x00050051, 0x00000006, 0x00000025, 0x00000021, 0x00000001,
   0x00070050, 0x00000007, 0x00000026, 0x00000024, 0x00000025, 0x00000022, 0x00000023, 0x00050041,
   0x00000027, 0x00000028, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000028, 0x00000026, 0x000100fd,
   0x00010038,
};

/* fixed.frag */
static const uint32_t frag_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
   0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
   0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
   0x00030005, 0x00000009, 0x00000063, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013,
   0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017,
   0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b,
   0x00000008, 0x00000009, 0x00000003, 0x0004002b, 0x00000006, 0x0000000a, 0x3e800000, 0x0004002b,
   0x00000006, 0x0000000b, 0x00000000, 0x0007002c, 0x00000007, 0x0000000c, 0x0000000a, 0x0000000b,
   0x0000000b, 0x0000000a, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
   0x00000005, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
};

static const unsigned char clear_rgba[4]  = { 0, 0, 0, 0 };
static const unsigned char expect_rgba[4] = { 64, 0, 0, 64 };

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

static int32_t
round_even_q8(double v)
{
   double q = v * 256.0;
   int64_t lo = (int64_t)q;
   if ((double)lo > q) lo--;
   double f = q - (double)lo;
   if (f > 0.5 || (f == 0.5 && (lo & 1))) lo++;
   return (int32_t)lo;
}

static int64_t
oracle_edge(int32_t ax, int32_t ay, int32_t bx, int32_t by,
            int32_t px, int32_t py)
{
   return (int64_t)(bx - px) * (ay - py) -
          (int64_t)(by - py) * (ax - px);
}

static int
oracle_triangle(const int32_t input[3][2], int x, int y)
{
   int32_t v[3][2];
   memcpy(v, input, sizeof(v));
   int64_t area = oracle_edge(v[0][0],v[0][1],v[1][0],v[1][1],v[2][0],v[2][1]);
   if (!area) return 0;
   if (area < 0) {
      int32_t tx=v[1][0], ty=v[1][1];
      v[1][0]=v[2][0]; v[1][1]=v[2][1]; v[2][0]=tx; v[2][1]=ty;
   }
   int32_t p[2] = { x * 256 + 128, y * 256 + 128 };
   for (int e = 0; e < 3; e++) {
      int a = (e + 1) % 3, b = (e + 2) % 3;
      int32_t dx = v[b][0] - v[a][0], dy = v[b][1] - v[a][1];
      int bias = (dy > 0 || (dy == 0 && dx < 0)) ? 0 : -1;
      if (oracle_edge(v[a][0], v[a][1], v[b][0], v[b][1], p[0], p[1]) + bias < 0)
         return 0;
   }
   return 1;
}

static int
oracle_owners(int x, int y)
{
   int32_t x0 = round_even_q8(8.0), x1 = round_even_q8(184.0);
   int32_t y0 = round_even_q8(8.0), y1 = round_even_q8(184.0);
   const int32_t t0[3][2] = { {x0,y0}, {x1,y0}, {x1,y1} };
   const int32_t t1[3][2] = { {x0,y0}, {x1,y1}, {x0,y1} };
   return oracle_triangle(t0,x,y) + oracle_triangle(t1,x,y);
}

static int
check_oracle(const char *name, const unsigned char *image, int scissor)
{
   unsigned cracks = 0, doubles = 0, wrong = 0, covered = 0;
   for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      int owners = oracle_owners(x,y);
      if (scissor && !(x == SC_X && y == SC_Y)) owners = 0;
      const unsigned char *p = pixel(image,x,y);
      unsigned want = owners ? 64u * (unsigned)owners : 0u;
      if (owners == 0 && (p[0] || p[1] || p[2] || p[3])) wrong++;
      if (owners > 0) {
         covered++;
         if (p[0] + 2 < want || p[0] > want + 2 || p[3] + 2 < want || p[3] > want + 2 || p[1] || p[2]) {
            if (p[0] < 32) cracks++;
            else if (p[0] > 96) doubles++;
            else wrong++;
         }
      }
   }
   printf("  %s: %u expected covered, cracks=%u doubles=%u other=%u; center=%u,%u,%u,%u\n",
          name, covered, cracks, doubles, wrong,
          pixel(image,96,96)[0], pixel(image,96,96)[1],
          pixel(image,96,96)[2], pixel(image,96,96)[3]);
   return cracks || doubles || wrong;
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
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = 0xF };
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
      .clearValue.color.float32 = { 0.0f, 0.0f, 0.0f, 0.0f } };
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
      vkCmdDraw(cmd, 6, 1, 0, 0);
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
   int fail = 0;
   fail |= check_oracle("full quad", wide, 0);
   fail |= check_oracle("one-pixel shared-edge scissor", tight, 1);
   if (!fail)
      printf("PASS fixed Q8 oracle: every shared-edge sample has exactly one owner\n");

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
