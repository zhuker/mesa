/*
 * Two elements of one combined-image-sampler array, two different samplers,
 * one shader, one pipeline: the sampler state the driver bakes must follow
 * the element that was actually indexed.
 *
 * The native driver specialises fragment shaders on sampler state. It walks
 * the NIR looking for a sampler descriptor it can identify, reads that
 * sampler's filter, wrap and border out of the descriptor set and compiles
 * them into the kernel as constants, so the texture fetch becomes straight
 * line code instead of a state lookup. That is only sound while the sampler
 * it identified is the sampler the shader will use. An array of two
 * descriptors indexed by a push constant is the case where it is not: both
 * elements are named by the same binding, only one of them is reached by any
 * given draw, and which one is not known until the command buffer runs.
 *
 *    layout(set = 0, binding = 0) uniform sampler2D tex[2];
 *
 *    tex[0]   VK_FILTER_NEAREST      same 2x2 image
 *    tex[1]   VK_FILTER_LINEAR       same 2x2 image
 *
 * Both elements view one image, so the image cannot be what separates the
 * two draws -- only the sampler can. The image is 2x2 and constant down each
 * column:
 *
 *    column 0   RGBA  32 200  64 255
 *    column 1   RGBA 224  40 192 255
 *
 * and the shader samples at a fixed (0.375, 0.5), which is a quarter of the
 * way from the centre of column 0 to the centre of column 1:
 *
 *    nearest   floor(0.375 * 2) = 0            ->  32 200  64 255
 *    linear    0.75 * col0 + 0.25 * col1       ->  80 160  96 255
 *
 * Those weights are 3/4 and 1/4 exactly, which every implementation can hold
 * in its subtexel fixed point, and the blends land on whole 8-bit values, so
 * the expected pixel is exact and one LSB of slack is enough. The vertical
 * coordinate is deliberately dead: v = 0.5 falls on the boundary between the
 * two rows, where nearest is free to pick either one and linear blends them
 * half and half, and the two rows are identical, so no filter can tell them
 * apart. Only the horizontal axis carries the answer.
 *
 * The sample coordinate is a constant, so the fragment's derivatives are zero,
 * the level of detail is zero, every fetch is a magnification, and magFilter
 * is the only filter that runs. One full-viewport triangle then paints all
 * 64x64 pixels with that one colour, and every pixel is checked, not a probe.
 *
 * Two draws, same pipeline, same descriptor set, differing only in the push
 * constant that indexes the array:
 *
 *    push index 0 -> whole frame  32 200  64 255
 *    push index 1 -> whole frame  80 160  96 255
 *
 * A driver that bakes element 0's sampler for both draws paints both frames
 * 32 200 64. One that bakes element 1's paints both 80 160 96. Either way the
 * two frames come out equal, and equal frames are the failure this test
 * exists to catch -- so the last assertion is that they differ, in the
 * direction the filters demand: red and blue up, green down.
 *
 * Note what the driver says about this shader before it runs it. The native
 * ICD reports shaderSampledImageArrayDynamicIndexing = VK_FALSE, so the
 * feature is only requested where it is advertised and the run prints what
 * each device reported. glslang emits no SampledImageArrayDynamicIndexing
 * capability for a push-constant index, so nothing rejects the shader, but a
 * device answering VK_FALSE is stating that it does not support this shape.
 *
 * What the native driver does today is decline to specialise at all here:
 *
 *   CUDAPIPE_SPEC_STATS=1 ... cpvk_sampler_array
 *   cudapipe: sampler specialisation: 0/2 fragment launches specialised
 *   (0.0%), 1 shaders, 1 with an unmatched sampler handle, ...
 *
 * The array index is not a constant, the handle does not match the shape the
 * specialiser recognises, and the generic path runs instead -- which is the
 * right answer and is why this passes. So what this test guards is the
 * fallback: the day the specialiser learns to match a handle like this one,
 * a wrong match stops being a missed optimisation and starts being wrong
 * pixels, silently, with the counter reading 100%.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, so this runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. The sampled image is linear-tiled and host-written, which all three
 * report as VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT capable.
 * SPIR-V is embedded; nothing is read from disk.
 *
 *   cc -std=c11 -Wall -Wextra -Werror src/cudapipe/tests/cpvk_sampler_array.c \
 *      -o /tmp/cpvk-tests/cpvk_sampler_array -lvulkan
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- sa.vert --
 *   #version 450
 *   const vec2 base[3] = vec2[3](vec2(-1.0, -1.0),
 *                                vec2( 3.0, -1.0),
 *                                vec2(-1.0,  3.0));
 *   void main()
 *   {
 *      gl_Position = vec4(base[gl_VertexIndex], 0.0, 1.0);
 *   }
 *
 *   -- sa.frag --
 *   #version 450
 *   layout(set = 0, binding = 0) uniform sampler2D tex[2];
 *   layout(push_constant) uniform Push { int index; } pc;
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      out_colour = texture(tex[pc.index], vec2(0.375, 0.5));
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/* The sampled image, and the one coordinate the shader reads it at. */
#define TEX_W 2
#define TEX_H 2
#define SAMPLE_U 0.375f
#define SAMPLE_V 0.5f

#define NELEM 2          /* sampler2D tex[2] */

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* sa.vert */
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
   0x0004002bu, 0x00000006u, 0x00000013u, 0xbf800000u, 0x0005002cu, 0x00000010u,
   0x00000014u, 0x00000013u, 0x00000013u, 0x0004002bu, 0x00000006u, 0x00000015u,
   0x40400000u, 0x0005002cu, 0x00000010u, 0x00000016u, 0x00000015u, 0x00000013u,
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

/* sa.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000021u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x00000010u, 0x00786574u, 0x00040005u, 0x00000012u,
   0x68737550u, 0x00000000u, 0x00050006u, 0x00000012u, 0x00000000u, 0x65646e69u,
   0x00000078u, 0x00030005u, 0x00000014u, 0x00006370u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x00000010u, 0x00000021u, 0x00000000u,
   0x00040047u, 0x00000010u, 0x00000022u, 0x00000000u, 0x00030047u, 0x00000012u,
   0x00000002u, 0x00050048u, 0x00000012u, 0x00000000u, 0x00000023u, 0x00000000u,
   0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
   0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u,
   0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u,
   0x00000009u, 0x00000003u, 0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u,
   0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu,
   0x0000000bu, 0x0000000au, 0x00040015u, 0x0000000cu, 0x00000020u, 0x00000000u,
   0x0004002bu, 0x0000000cu, 0x0000000du, 0x00000002u, 0x0004001cu, 0x0000000eu,
   0x0000000bu, 0x0000000du, 0x00040020u, 0x0000000fu, 0x00000000u, 0x0000000eu,
   0x0004003bu, 0x0000000fu, 0x00000010u, 0x00000000u, 0x00040015u, 0x00000011u,
   0x00000020u, 0x00000001u, 0x0003001eu, 0x00000012u, 0x00000011u, 0x00040020u,
   0x00000013u, 0x00000009u, 0x00000012u, 0x0004003bu, 0x00000013u, 0x00000014u,
   0x00000009u, 0x0004002bu, 0x00000011u, 0x00000015u, 0x00000000u, 0x00040020u,
   0x00000016u, 0x00000009u, 0x00000011u, 0x00040020u, 0x00000019u, 0x00000000u,
   0x0000000bu, 0x00040017u, 0x0000001cu, 0x00000006u, 0x00000002u, 0x0004002bu,
   0x00000006u, 0x0000001du, 0x3ec00000u, 0x0004002bu, 0x00000006u, 0x0000001eu,
   0x3f000000u, 0x0005002cu, 0x0000001cu, 0x0000001fu, 0x0000001du, 0x0000001eu,
   0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
   0x00000005u, 0x00050041u, 0x00000016u, 0x00000017u, 0x00000014u, 0x00000015u,
   0x0004003du, 0x00000011u, 0x00000018u, 0x00000017u, 0x00050041u, 0x00000019u,
   0x0000001au, 0x00000010u, 0x00000018u, 0x0004003du, 0x0000000bu, 0x0000001bu,
   0x0000001au, 0x00050057u, 0x00000007u, 0x00000020u, 0x0000001bu, 0x0000001fu,
   0x0003003eu, 0x00000009u, 0x00000020u, 0x000100fdu, 0x00010038u
};

/* The two texel columns, constant down each column so that v cannot matter. */
static const unsigned char col0_rgba[4] = {  32, 200,  64, 255 };
static const unsigned char col1_rgba[4] = { 224,  40, 192, 255 };

/* tex[0] is NEAREST: the fetch lands in column 0 and comes back untouched. */
static const unsigned char expect_nearest[4] = {  32, 200,  64, 255 };
/* tex[1] is LINEAR: 3/4 of column 0 plus 1/4 of column 1, exactly. */
static const unsigned char expect_linear[4]  = {  80, 160,  96, 255 };

/* Nothing in the frame should ever be this; a missing draw shows up as it. */
static const unsigned char clear_rgba[4] = { 26, 26, 38, 255 };

struct scan {
   unsigned matched;      /* pixels equal to the expected colour, within 1 */
   unsigned clear;        /* pixels still the clear colour: nothing drew */
   unsigned other;        /* pixels that are neither */
   unsigned char worst[4];   /* the first pixel that was neither */
   int worst_x, worst_y;
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

/* One LSB of slack, and no more: both expected colours are exact 8-bit values. */
static int
same(const unsigned char *px, const unsigned char *want)
{
   for (int i = 0; i < 4; i++) {
      int d = (int)px[i] - (int)want[i];
      if (d < -1 || d > 1)
         return 0;
   }
   return 1;
}

static const unsigned char *
pixel(const unsigned char *px, int x, int y)
{
   return px + ((size_t)y * W + x) * 4;
}

/* Every pixel of one frame, against the one colour the whole frame must be. */
static void
scan_image(const unsigned char *px, const unsigned char *want, struct scan *s)
{
   memset(s, 0, sizeof(*s));
   s->worst_x = s->worst_y = -1;
   for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
         const unsigned char *p = pixel(px, x, y);
         if (same(p, want)) {
            s->matched++;
         } else if (same(p, clear_rgba)) {
            s->clear++;
         } else {
            s->other++;
            if (s->worst_x < 0) {
               s->worst_x = x;
               s->worst_y = y;
               memcpy(s->worst, p, 4);
            }
         }
      }
   }
}

static void
describe(const char *what, const unsigned char *px)
{
   printf("  %-34s = %3u %3u %3u %3u\n", what, px[0], px[1], px[2], px[3]);
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

   /*
    * Indexing sampler2D tex[2] with a push constant is a dynamically uniform
    * index, which is what shaderSampledImageArrayDynamicIndexing governs. Ask
    * for it only where it is advertised: requesting an unsupported feature is
    * VK_ERROR_FEATURE_NOT_PRESENT, and this shape is exactly the one the
    * driver under test has to get right whatever its limits say.
    */
   VkPhysicalDeviceFeatures supported;
   vkGetPhysicalDeviceFeatures(pdev, &supported);
   const int dynamic_indexing = supported.shaderSampledImageArrayDynamicIndexing;
   printf("  shaderSampledImageArrayDynamicIndexing = %s\n",
          dynamic_indexing ? "VK_TRUE" : "VK_FALSE  (asking for it anyway "
          "would fail; the shader declares no capability, so nothing "
          "rejects it)");
   VkPhysicalDeviceFeatures enabled = {
      .shaderSampledImageArrayDynamicIndexing = dynamic_indexing ?
         VK_TRUE : VK_FALSE };

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

   /* One buffer, two frames: the nearest draw then the linear one. */
   const VkDeviceSize frame_bytes = (VkDeviceSize)W * H * 4;
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = NELEM * frame_bytes,
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

   /*
    * The sampled image: 2x2, linear-tiled and written by the host, which all
    * three drivers report as filterable. Its rows are identical and its two
    * columns differ, so the horizontal filter is the only thing that can
    * change the answer.
    */
   VkImageCreateInfo timgi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { TEX_W, TEX_H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED };
   VkImage timg;
   CHECK(vkCreateImage(dev, &timgi, NULL, &timg));
   VkMemoryRequirements treq;
   vkGetImageMemoryRequirements(dev, timg, &treq);
   uint32_t ttype = pick_memory(pdev, treq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (ttype == UINT32_MAX)
      ttype = pick_memory(pdev, treq.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (ttype == UINT32_MAX) { fprintf(stderr, "no texture memory type\n"); return 1; }
   VkMemoryAllocateInfo tmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = treq.size,
                                 .memoryTypeIndex = ttype };
   VkDeviceMemory tmem;
   CHECK(vkAllocateMemory(dev, &tmai, NULL, &tmem));
   CHECK(vkBindImageMemory(dev, timg, tmem, 0));
   {
      /* The row pitch is the driver's, not two texels: a linear image may pad
       * its rows, and assuming it does not puts rows where no sampler looks. */
      VkImageSubresource sub = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
      VkSubresourceLayout lay;
      vkGetImageSubresourceLayout(dev, timg, &sub, &lay);

      void *p;
      CHECK(vkMapMemory(dev, tmem, 0, VK_WHOLE_SIZE, 0, &p));
      unsigned char *t = (unsigned char *)p + lay.offset;
      for (int y = 0; y < TEX_H; y++)
         for (int x = 0; x < TEX_W; x++)
            memcpy(t + (size_t)y * lay.rowPitch + (size_t)x * 4,
                   x ? col1_rgba : col0_rgba, 4);
      VkMappedMemoryRange flush = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = tmem, .size = VK_WHOLE_SIZE };
      CHECK(vkFlushMappedMemoryRanges(dev, 1, &flush));
      vkUnmapMemory(dev, tmem);
   }
   VkImageViewCreateInfo tvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = timg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = timgi.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView tview;
   CHECK(vkCreateImageView(dev, &tvci, NULL, &tview));

   /*
    * The whole point: two samplers that differ in exactly one field. Every
    * other field is identical, so a driver that gets the two draws confused
    * cannot blame wrap mode, border or mip state for the difference.
    */
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
      .maxLod = 0.0f };
   VkSampler samplers[NELEM];
   CHECK(vkCreateSampler(dev, &sci, NULL, &samplers[0]));
   sci.magFilter = VK_FILTER_LINEAR;
   sci.minFilter = VK_FILTER_LINEAR;
   CHECK(vkCreateSampler(dev, &sci, NULL, &samplers[1]));

   /* One binding, two array elements, same view, different samplers. */
   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = NELEM,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
   VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                NELEM };
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet dset;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

   VkDescriptorImageInfo dii[NELEM] = {
      { .sampler = samplers[0], .imageView = tview,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { .sampler = samplers[1], .imageView = tview,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
   };
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
      .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = NELEM,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = dii };
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

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

   /* Four bytes of push constant: the array element this draw indexes. */
   VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(int32_t) };
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
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
   VkViewport vp = { 0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f };
   VkRect2D scissor = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp,
      .scissorCount = 1, .pScissors = &scissor };
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

   /* One pipeline. Both draws compile through it; only the push differs. */
   VkGraphicsPipelineCreateInfo gpi = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &pri, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .layout = layout };
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

   /* The host wrote the texels; hand the image to the fragment stage. */
   VkImageMemoryBarrier tex_to_read = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = timg,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                        0, NULL, 0, NULL, 1, &tex_to_read);

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
      .clearValue.color.float32 = { 26.0f / 255.0f, 26.0f / 255.0f,
                                    38.0f / 255.0f, 1.0f } };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &at };

   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_colour);

   for (int pass = 0; pass < NELEM; pass++) {
      if (pass)
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                              0, NULL, 0, NULL, 1, &back_to_colour);

      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                              0, 1, &dset, 0, NULL);
      /* The only difference between the two draws. */
      const int32_t index = pass;
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(index), &index);
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

   const unsigned char *near_frame = mapped;
   const unsigned char *lin_frame = mapped + frame_bytes;

   struct scan ns, ls;
   scan_image(near_frame, expect_nearest, &ns);
   scan_image(lin_frame, expect_linear, &ls);

   const unsigned char *np = pixel(near_frame, W / 2, H / 2);
   const unsigned char *lp = pixel(lin_frame, W / 2, H / 2);

   printf("  texture %dx%d at (%.3f, %.3f), one image, two samplers\n",
          TEX_W, TEX_H, (double)SAMPLE_U, (double)SAMPLE_V);
   describe("texel column 0", col0_rgba);
   describe("texel column 1", col1_rgba);
   describe("tex[0] NEAREST expected", expect_nearest);
   describe("tex[1] LINEAR  expected", expect_linear);
   describe("push index 0 -> got", np);
   describe("push index 1 -> got", lp);
   printf("  push index 0: %u of %d pixels on the nearest value, "
          "%u still the clear colour, %u neither\n",
          ns.matched, W * H, ns.clear, ns.other);
   if (ns.other)
      printf("    first stray at (%d,%d) = %3u %3u %3u %3u\n",
             ns.worst_x, ns.worst_y,
             ns.worst[0], ns.worst[1], ns.worst[2], ns.worst[3]);
   printf("  push index 1: %u of %d pixels on the linear value, "
          "%u still the clear colour, %u neither\n",
          ls.matched, W * H, ls.clear, ls.other);
   if (ls.other)
      printf("    first stray at (%d,%d) = %3u %3u %3u %3u\n",
             ls.worst_x, ls.worst_y,
             ls.worst[0], ls.worst[1], ls.worst[2], ls.worst[3]);

   int fail = 0;

   if (ns.clear == (unsigned)(W * H) || ls.clear == (unsigned)(W * H)) {
      printf("FAIL a frame is entirely the clear colour: the draw did not "
             "cover the viewport\n");
      fail = 1;
   }

   /* Each element on its own: the whole frame is that element's filter. */
   if (ns.matched != (unsigned)(W * H)) {
      printf("FAIL push index 0 painted %u of %d pixels the NEAREST value "
             "%u %u %u %u\n", ns.matched, W * H, expect_nearest[0],
             expect_nearest[1], expect_nearest[2], expect_nearest[3]);
      fail = 1;
   }
   if (ls.matched != (unsigned)(W * H)) {
      printf("FAIL push index 1 painted %u of %d pixels the LINEAR value "
             "%u %u %u %u\n", ls.matched, W * H, expect_linear[0],
             expect_linear[1], expect_linear[2], expect_linear[3]);
      fail = 1;
   }

   /* Nearest returns a texel, untouched. Linear returns neither texel. */
   if (!same(np, col0_rgba)) {
      printf("FAIL push index 0 did not return a texel of the image: "
             "nearest must be column 0 exactly\n");
      fail = 1;
   }
   if (same(lp, col0_rgba) || same(lp, col1_rgba)) {
      printf("FAIL push index 1 returned a whole texel, not a blend -- "
             "tex[1] filtered as if it were NEAREST\n");
      fail = 1;
   }

   /*
    * The direction, which is the part a driver cannot fake by getting one
    * frame right: moving a quarter of the way towards column 1 raises red and
    * blue and lowers green, and does so by far more than a rounding LSB.
    */
   if ((int)lp[0] - (int)np[0] < 8) {
      printf("FAIL linear red %u is not above nearest red %u\n", lp[0], np[0]);
      fail = 1;
   }
   if ((int)np[1] - (int)lp[1] < 8) {
      printf("FAIL linear green %u is not below nearest green %u\n",
             lp[1], np[1]);
      fail = 1;
   }
   if ((int)lp[2] - (int)np[2] < 8) {
      printf("FAIL linear blue %u is not above nearest blue %u\n", lp[2], np[2]);
      fail = 1;
   }

   /* And the failure this test exists for: one sampler baked for both draws. */
   if (same(np, lp)) {
      const char *which =
         same(np, expect_nearest) ? "tex[0]'s VK_FILTER_NEAREST" :
         same(np, expect_linear)  ? "tex[1]'s VK_FILTER_LINEAR" :
                                    "one sampler this test does not recognise";
      printf("FAIL both draws produced the same pixel, so the array element "
             "the push constant chose did not select the sampler: %s was "
             "used for both\n", which);
      fail = 1;
   }

   if (!fail)
      printf("PASS tex[0] filtered NEAREST and tex[1] filtered LINEAR through "
             "one pipeline, chosen by a push constant\n");

   if (ppm) {
      char linear_path[1024];
      write_ppm(ppm, near_frame);
      snprintf(linear_path, sizeof(linear_path), "%.*s.linear.ppm",
               (int)(sizeof(linear_path) - 32), ppm);
      write_ppm(linear_path, lin_frame);
   }

   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroySampler(dev, samplers[1], NULL);
   vkDestroySampler(dev, samplers[0], NULL);
   vkDestroyImageView(dev, tview, NULL);
   vkDestroyImage(dev, timg, NULL);
   vkFreeMemory(dev, tmem, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fail;
}
