/*
 * An integer colour attachment: VK_FORMAT_R8G8B8A8_UINT, rendered to and read
 * back byte for byte.
 *
 * Every other colour test in this suite renders a colour -- a number between
 * zero and one that an encoding is free to round. This one renders numbers.
 * A fragment shader whose output is a `uvec4` must reach the attachment
 * unchanged: no sRGB curve, no clamp to 0..1, no rounding, and above all no
 * float conversion anywhere on the way. The check is equality on all four
 * bytes of all 4096 pixels, because for an integer format there is no other
 * kind of correct.
 *
 * The path this covers, and what would break it:
 *
 *   - the fragment output itself. The writeback ABI carries a fragment's
 *     colour as float[4], and the NIR->PTX backend stores an output with its
 *     own LLVM type -- `st.v4.u32` for a uint output, so the four words in
 *     the colour buffer are the shader's integers and the float[4] they are
 *     read as is a carrier. A backend that converted instead (`cvt.rn.f32`)
 *     would put 200.0f where 200 belongs and this test would see 255 after
 *     the clamp, or 0 for every value below one.
 *   - loadOp CLEAR. A VkClearColorValue for an integer format is its uint32
 *     member, and packing it as floats turns {17,250,3,9} into four zeroes.
 *     The right-hand third of the image is cleared and never drawn over.
 *   - the write mask, which is the only reason the destination is read back
 *     into the blend equation at all here. The middle draw writes R and A
 *     through a mask of R|A, so G and B must still hold the first draw's
 *     exact bytes -- a destination load that decoded and re-encoded through
 *     floats would not return them.
 *   - blending, which Vulkan forbids on an integer attachment and this
 *     driver therefore does not advertise (no COLOR_ATTACHMENT_BLEND_BIT)
 *     and does not perform. The last draw asks for it anyway, deliberately,
 *     which is invalid usage on purpose: the driver says so on stderr and
 *     writes the shader's bits rather than a float sum of them. Without that
 *     refusal 0..255 held as float bit patterns are denormals, and one
 *     multiply flushes every one of them to zero.
 *
 * Three passes over one 64x64 attachment, then one readback:
 *
 *   pass 0  clear {17,250,3,9}, then k=1   over x 0..31, mask RGBA
 *   pass 1  k=200 over x 0..15, mask R|A
 *           k=7   over x 16..31, mask RGBA, blendEnable VK_TRUE
 *
 * so the finished image is three vertical bands: k=200's R and A over k=1's
 * G and B, then k=7 whole, then the clear.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, so this runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. SPIR-V is embedded; nothing is read from disk.
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- uc.vert --
 *   #version 450
 *   void main()
 *   {
 *      vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
 *      gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
 *   }
 *
 *   -- uc.frag --
 *   #version 450
 *   layout(location = 0) out uvec4 o_colour;
 *   layout(push_constant) uniform PC { uint k; } pc;
 *   void main()
 *   {
 *      uint x = uint(gl_FragCoord.x);
 *      uint y = uint(gl_FragCoord.y);
 *      o_colour = uvec4((x + pc.k)      & 255u,
 *                       (y * 3u + pc.k) & 255u,
 *                       (x ^ y)         & 255u,
 *                       (255u - pc.k)   & 255u);
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/* The three bands, as half-open pixel ranges in x. */
#define BAND_MASK_X0   0
#define BAND_MASK_X1  16
#define BAND_BLEND_X0 16
#define BAND_BLEND_X1 32
#define BAND_CLEAR_X0 32
#define BAND_CLEAR_X1 64

/* The push constant of each draw. 200 and 7 are chosen so that no channel of
 * one draw equals the same channel of another at the same pixel. */
#define K_BASE   1u
#define K_MASK 200u
#define K_BLEND  7u

/* What the whole attachment is cleared to, as integers. */
static const uint32_t clear_uint[4] = { 17, 250, 3, 9 };

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* uc.vert */
static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002bu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000000u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x0000001du, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
   0x00030005u, 0x00000009u, 0x00007675u, 0x00060005u, 0x0000000cu, 0x565f6c67u,
   0x65747265u, 0x646e4978u, 0x00007865u, 0x00060005u, 0x0000001bu, 0x505f6c67u,
   0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u, 0x0000001bu, 0x00000000u,
   0x505f6c67u, 0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000001bu, 0x00000001u,
   0x505f6c67u, 0x746e696fu, 0x657a6953u, 0x00000000u, 0x00070006u, 0x0000001bu,
   0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u, 0x0065636eu, 0x00070006u,
   0x0000001bu, 0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu,
   0x00030005u, 0x0000001du, 0x00000000u, 0x00040047u, 0x0000000cu, 0x0000000bu,
   0x0000002au, 0x00030047u, 0x0000001bu, 0x00000002u, 0x00050048u, 0x0000001bu,
   0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x0000001bu, 0x00000001u,
   0x0000000bu, 0x00000001u, 0x00050048u, 0x0000001bu, 0x00000002u, 0x0000000bu,
   0x00000003u, 0x00050048u, 0x0000001bu, 0x00000003u, 0x0000000bu, 0x00000004u,
   0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
   0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000002u,
   0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u, 0x00040015u, 0x0000000au,
   0x00000020u, 0x00000001u, 0x00040020u, 0x0000000bu, 0x00000001u, 0x0000000au,
   0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000001u, 0x0004002bu, 0x0000000au,
   0x0000000eu, 0x00000001u, 0x0004002bu, 0x0000000au, 0x00000010u, 0x00000002u,
   0x00040017u, 0x00000017u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000018u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000018u, 0x00000019u, 0x00000001u,
   0x0004001cu, 0x0000001au, 0x00000006u, 0x00000019u, 0x0006001eu, 0x0000001bu,
   0x00000017u, 0x00000006u, 0x0000001au, 0x0000001au, 0x00040020u, 0x0000001cu,
   0x00000003u, 0x0000001bu, 0x0004003bu, 0x0000001cu, 0x0000001du, 0x00000003u,
   0x0004002bu, 0x0000000au, 0x0000001eu, 0x00000000u, 0x0004002bu, 0x00000006u,
   0x00000020u, 0x40000000u, 0x0004002bu, 0x00000006u, 0x00000022u, 0x3f800000u,
   0x0004002bu, 0x00000006u, 0x00000025u, 0x00000000u, 0x00040020u, 0x00000029u,
   0x00000003u, 0x00000017u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000008u, 0x00000009u,
   0x00000007u, 0x0004003du, 0x0000000au, 0x0000000du, 0x0000000cu, 0x000500c4u,
   0x0000000au, 0x0000000fu, 0x0000000du, 0x0000000eu, 0x000500c7u, 0x0000000au,
   0x00000011u, 0x0000000fu, 0x00000010u, 0x0004006fu, 0x00000006u, 0x00000012u,
   0x00000011u, 0x0004003du, 0x0000000au, 0x00000013u, 0x0000000cu, 0x000500c7u,
   0x0000000au, 0x00000014u, 0x00000013u, 0x00000010u, 0x0004006fu, 0x00000006u,
   0x00000015u, 0x00000014u, 0x00050050u, 0x00000007u, 0x00000016u, 0x00000012u,
   0x00000015u, 0x0003003eu, 0x00000009u, 0x00000016u, 0x0004003du, 0x00000007u,
   0x0000001fu, 0x00000009u, 0x0005008eu, 0x00000007u, 0x00000021u, 0x0000001fu,
   0x00000020u, 0x00050050u, 0x00000007u, 0x00000023u, 0x00000022u, 0x00000022u,
   0x00050083u, 0x00000007u, 0x00000024u, 0x00000021u, 0x00000023u, 0x00050051u,
   0x00000006u, 0x00000026u, 0x00000024u, 0x00000000u, 0x00050051u, 0x00000006u,
   0x00000027u, 0x00000024u, 0x00000001u, 0x00070050u, 0x00000017u, 0x00000028u,
   0x00000026u, 0x00000027u, 0x00000025u, 0x00000022u, 0x00050041u, 0x00000029u,
   0x0000002au, 0x0000001du, 0x0000001eu, 0x0003003eu, 0x0000002au, 0x00000028u,
   0x000100fdu, 0x00010038u
};

/* uc.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000036u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x00000019u, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000008u, 0x00000078u,
   0x00060005u, 0x0000000cu, 0x465f6c67u, 0x43676172u, 0x64726f6fu, 0x00000000u,
   0x00030005u, 0x00000012u, 0x00000079u, 0x00050005u, 0x00000019u, 0x6f635f6fu,
   0x72756f6cu, 0x00000000u, 0x00030005u, 0x0000001bu, 0x00004350u, 0x00040006u,
   0x0000001bu, 0x00000000u, 0x0000006bu, 0x00030005u, 0x0000001du, 0x00006370u,
   0x00040047u, 0x0000000cu, 0x0000000bu, 0x0000000fu, 0x00040047u, 0x00000019u,
   0x0000001eu, 0x00000000u, 0x00030047u, 0x0000001bu, 0x00000002u, 0x00050048u,
   0x0000001bu, 0x00000000u, 0x00000023u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u,
   0x00000000u, 0x00040020u, 0x00000007u, 0x00000007u, 0x00000006u, 0x00030016u,
   0x00000009u, 0x00000020u, 0x00040017u, 0x0000000au, 0x00000009u, 0x00000004u,
   0x00040020u, 0x0000000bu, 0x00000001u, 0x0000000au, 0x0004003bu, 0x0000000bu,
   0x0000000cu, 0x00000001u, 0x0004002bu, 0x00000006u, 0x0000000du, 0x00000000u,
   0x00040020u, 0x0000000eu, 0x00000001u, 0x00000009u, 0x0004002bu, 0x00000006u,
   0x00000013u, 0x00000001u, 0x00040017u, 0x00000017u, 0x00000006u, 0x00000004u,
   0x00040020u, 0x00000018u, 0x00000003u, 0x00000017u, 0x0004003bu, 0x00000018u,
   0x00000019u, 0x00000003u, 0x0003001eu, 0x0000001bu, 0x00000006u, 0x00040020u,
   0x0000001cu, 0x00000009u, 0x0000001bu, 0x0004003bu, 0x0000001cu, 0x0000001du,
   0x00000009u, 0x00040015u, 0x0000001eu, 0x00000020u, 0x00000001u, 0x0004002bu,
   0x0000001eu, 0x0000001fu, 0x00000000u, 0x00040020u, 0x00000020u, 0x00000009u,
   0x00000006u, 0x0004002bu, 0x00000006u, 0x00000024u, 0x000000ffu, 0x0004002bu,
   0x00000006u, 0x00000027u, 0x00000003u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000007u,
   0x00000008u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000012u, 0x00000007u,
   0x00050041u, 0x0000000eu, 0x0000000fu, 0x0000000cu, 0x0000000du, 0x0004003du,
   0x00000009u, 0x00000010u, 0x0000000fu, 0x0004006du, 0x00000006u, 0x00000011u,
   0x00000010u, 0x0003003eu, 0x00000008u, 0x00000011u, 0x00050041u, 0x0000000eu,
   0x00000014u, 0x0000000cu, 0x00000013u, 0x0004003du, 0x00000009u, 0x00000015u,
   0x00000014u, 0x0004006du, 0x00000006u, 0x00000016u, 0x00000015u, 0x0003003eu,
   0x00000012u, 0x00000016u, 0x0004003du, 0x00000006u, 0x0000001au, 0x00000008u,
   0x00050041u, 0x00000020u, 0x00000021u, 0x0000001du, 0x0000001fu, 0x0004003du,
   0x00000006u, 0x00000022u, 0x00000021u, 0x00050080u, 0x00000006u, 0x00000023u,
   0x0000001au, 0x00000022u, 0x000500c7u, 0x00000006u, 0x00000025u, 0x00000023u,
   0x00000024u, 0x0004003du, 0x00000006u, 0x00000026u, 0x00000012u, 0x00050084u,
   0x00000006u, 0x00000028u, 0x00000026u, 0x00000027u, 0x00050041u, 0x00000020u,
   0x00000029u, 0x0000001du, 0x0000001fu, 0x0004003du, 0x00000006u, 0x0000002au,
   0x00000029u, 0x00050080u, 0x00000006u, 0x0000002bu, 0x00000028u, 0x0000002au,
   0x000500c7u, 0x00000006u, 0x0000002cu, 0x0000002bu, 0x00000024u, 0x0004003du,
   0x00000006u, 0x0000002du, 0x00000008u, 0x0004003du, 0x00000006u, 0x0000002eu,
   0x00000012u, 0x000500c6u, 0x00000006u, 0x0000002fu, 0x0000002du, 0x0000002eu,
   0x000500c7u, 0x00000006u, 0x00000030u, 0x0000002fu, 0x00000024u, 0x00050041u,
   0x00000020u, 0x00000031u, 0x0000001du, 0x0000001fu, 0x0004003du, 0x00000006u,
   0x00000032u, 0x00000031u, 0x00050082u, 0x00000006u, 0x00000033u, 0x00000024u,
   0x00000032u, 0x000500c7u, 0x00000006u, 0x00000034u, 0x00000033u, 0x00000024u,
   0x00070050u, 0x00000017u, 0x00000035u, 0x00000025u, 0x0000002cu, 0x00000030u,
   0x00000034u, 0x0003003eu, 0x00000019u, 0x00000035u, 0x000100fdu, 0x00010038u
};

/* The shader, written a second time on the host: this is the oracle. */
static void
shader_colour(unsigned x, unsigned y, unsigned k, uint8_t out[4])
{
   out[0] = (uint8_t)((x + k) & 255u);
   out[1] = (uint8_t)((y * 3u + k) & 255u);
   out[2] = (uint8_t)((x ^ y) & 255u);
   out[3] = (uint8_t)((255u - k) & 255u);
}

static void
expected_pixel(unsigned x, unsigned y, uint8_t out[4])
{
   if (x >= BAND_CLEAR_X0) {
      for (int c = 0; c < 4; c++)
         out[c] = (uint8_t)clear_uint[c];
   } else if (x >= BAND_BLEND_X0) {
      /* Written whole by the draw whose blending the driver must ignore. */
      shader_colour(x, y, K_BLEND, out);
   } else {
      /* R and A from the masked draw, G and B still the first draw's. */
      uint8_t base[4], masked[4];
      shader_colour(x, y, K_BASE, base);
      shader_colour(x, y, K_MASK, masked);
      out[0] = masked[0];
      out[1] = base[1];
      out[2] = base[2];
      out[3] = masked[3];
   }
}

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

   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UINT;

   /*
    * What an application asks before it commits to the format. The sample in
    * samples/interop needs exactly these two bits and stops without them.
    */
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pdev, cfmt, &fp);
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   printf("  R8G8B8A8_UINT optimalTilingFeatures 0x%x, needed 0x%x\n",
          fp.optimalTilingFeatures, need);
   if ((fp.optimalTilingFeatures & need) != need) {
      fprintf(stderr, "R8G8B8A8_UINT is not a colour attachment here\n");
      return 1;
   }
   /* And what it must not claim: blending an integer attachment. */
   if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) {
      fprintf(stderr, "R8G8B8A8_UINT advertises COLOR_ATTACHMENT_BLEND_BIT, "
              "which no integer format may\n");
      return 1;
   }

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; i++)
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }
   free(families);
   if (family == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 1; }

   /* Dynamic rendering as the extension where it is one, as core where the
    * device is 1.3, exactly as cpvk_static_viewport does it. */
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

   const VkDeviceSize frame_bytes = (VkDeviceSize)W * H * 4;
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = frame_bytes,
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

   VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(uint32_t) };
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
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
   VkPipelineRenderingCreateInfo pri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt };
   VkDynamicState dyn_states[2] = { VK_DYNAMIC_STATE_VIEWPORT,
                                    VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dsi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn_states };

   /*
    * Three pipelines differing only in their one blend attachment state.
    *
    * The third one enables blending, which for an integer attachment is
    * invalid usage -- VUID-VkGraphicsPipelineCreateInfo-renderPass-06041,
    * since the format does not carry COLOR_ATTACHMENT_BLEND_BIT and the
    * check above proved it does not. It is here on purpose: an application
    * that gets this wrong must still get its integers, not a float sum of
    * two denormals. The driver says so on stderr and drops the request.
    */
   VkPipelineColorBlendAttachmentState cba_plain = { .colorWriteMask = 0xF };
   VkPipelineColorBlendAttachmentState cba_mask = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_A_BIT };
   VkPipelineColorBlendAttachmentState cba_blend = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = 0xF };
   VkPipelineColorBlendStateCreateInfo cb[3] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba_plain },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba_mask },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba_blend },
   };

   VkGraphicsPipelineCreateInfo gpi[3];
   for (int i = 0; i < 3; i++) {
      gpi[i] = (VkGraphicsPipelineCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext = &pri, .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = &ms, .pDepthStencilState = &ds,
         .pColorBlendState = &cb[i], .pDynamicState = &dsi,
         .layout = layout };
   }
   VkPipeline pipes[3];
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 3, gpi, NULL, pipes));

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

   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_colour);

   VkRenderingAttachmentInfo at = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   /* An integer format clears through the uint32 member of the union, and
    * that is the whole point of the right-hand band. */
   memcpy(at.clearValue.color.uint32, clear_uint, sizeof(clear_uint));
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &at };

   const VkRect2D band_base = { { BAND_MASK_X0, 0 },
                                { BAND_BLEND_X1 - BAND_MASK_X0, H } };
   const VkRect2D band_mask = { { BAND_MASK_X0, 0 },
                                { BAND_MASK_X1 - BAND_MASK_X0, H } };
   const VkRect2D band_blend = { { BAND_BLEND_X0, 0 },
                                 { BAND_BLEND_X1 - BAND_BLEND_X0, H } };

   /* Pass 0: the clear, then the base pattern over the left two bands. */
   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[0]);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &band_base);
   uint32_t k = K_BASE;
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(k), &k);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   /* Pass 1: the masked draw, then the one that asks to blend. */
   at.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[1]);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &band_mask);
   k = K_MASK;
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(k), &k);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[2]);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &band_blend);
   k = K_BLEND;
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(k), &k);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

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

   unsigned char *mapped = NULL;
   CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   VkMappedMemoryRange invalidate = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = bmem, .size = VK_WHOLE_SIZE };
   CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &invalidate));

   /* Exact bytes, every pixel. The first three disagreements are printed
    * with their band, because which band fails says which mechanism did. */
   unsigned bad = 0, bad_clear = 0, bad_mask = 0, bad_blend = 0;
   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const unsigned char *got = mapped + ((size_t)y * W + x) * 4;
         uint8_t want[4];
         expected_pixel(x, y, want);
         if (!memcmp(got, want, 4))
            continue;
         bad++;
         if (x >= BAND_CLEAR_X0) bad_clear++;
         else if (x >= BAND_BLEND_X0) bad_blend++;
         else bad_mask++;
         if (bad <= 3)
            printf("  (%u,%u) got %u %u %u %u  want %u %u %u %u\n", x, y,
                   got[0], got[1], got[2], got[3],
                   want[0], want[1], want[2], want[3]);
      }
   }

   printf("  masked band  x %d..%d   %s\n", BAND_MASK_X0, BAND_MASK_X1 - 1,
          bad_mask ? "WRONG" : "exact");
   printf("  blend band   x %d..%d  %s\n", BAND_BLEND_X0, BAND_BLEND_X1 - 1,
          bad_blend ? "WRONG" : "exact (blending ignored, as required)");
   printf("  cleared band x %d..%d  %s\n", BAND_CLEAR_X0, BAND_CLEAR_X1 - 1,
          bad_clear ? "WRONG" : "exact");
   printf("%s: %u of %d pixels differ\n", bad ? "FAIL" : "PASS", bad, W * H);

   vkUnmapMemory(dev, bmem);
   for (int i = 0; i < 3; i++)
      vkDestroyPipeline(dev, pipes[i], NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return bad ? 1 : 0;
}
