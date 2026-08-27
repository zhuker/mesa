/*
 * VK_FORMAT_R16G16_UNORM as a colour attachment: rendered to, blended into,
 * multisample-resolved, and read back as sixteen-bit integers.
 *
 * The format was in cpvk_formats[] with a texel decode and colour encoding
 * -1 -- sampled but not renderable. Nothing refused a draw into it: the four
 * draw paths resolve the encoding as MAX2(fb.color_encoding, 0) and 0 is
 * CP_COLOR_R8G8B8A8_UNORM, so a shader writing (1,0,0,1) put the four bytes
 * ff 00 00 ff into two 16-bit texels -- R16=255, G16=65280 -- and returned
 * VK_SUCCESS. The LOAD_OP_CLEAR before it packed the real format correctly,
 * so a cleared-but-not-drawn target looked right and a drawn one did not.
 *
 * Every value checked here is exact, and none of them is representable in
 * eight bits: the shader writes (x + k) and (y * 257 + k) divided by 65535,
 * so an encoder that rounded to 8 bits and back would return multiples of
 * 257 where consecutive integers belong. That is the point of the numbers.
 *
 * Four things are covered, and each fails differently:
 *
 *   - the store arm (cp_store_dst). Left band, plain draw.
 *   - the load arm (cp_load_dst), reached two ways: a write mask, which must
 *     leave G holding the earlier draw's exact bits, and a real blend, which
 *     must read the destination back as a number and add to it. A load that
 *     decoded 16-bit texels as 8-bit bytes would return a different value
 *     for both.
 *   - the multisample resolve (cp_resolve_samples), which decodes every
 *     sample plane, averages and re-encodes. The format's blocksize is 4 and
 *     it is not pure-integer, so giving it a colour encoding also makes
 *     vkGetPhysicalDeviceImageFormatProperties2 advertise 4x and 8x for it;
 *     this checks the advertisement is honest. The draw is scissored on a
 *     pixel boundary, so every sample of a pixel holds the same value and
 *     the average of them is exactly that value -- an exact oracle for an
 *     average.
 *   - the refusal. A colour attachment whose format has no encoding is now
 *     rejected at vkCmdBeginRendering with VK_ERROR_FEATURE_NOT_PRESENT,
 *     the way an unsupported depth format already was. R32G32_SFLOAT is the
 *     probe: the driver honestly reports no COLOR_ATTACHMENT_BIT for it, so
 *     rendering into it is an application ignoring the answer -- which is
 *     precisely the case that used to corrupt.
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- u16.vert --
 *   #version 450
 *   void main()
 *   {
 *      vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
 *      gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
 *   }
 *
 *   -- u16.frag --
 *   #version 450
 *   layout(location = 0) out vec4 o_colour;
 *   layout(push_constant) uniform PC { uint k; } pc;
 *   void main()
 *   {
 *      uint x = uint(gl_FragCoord.x);
 *      uint y = uint(gl_FragCoord.y);
 *      o_colour = vec4(float((x + pc.k) & 65535u) / 65535.0,
 *                      float((y * 257u + pc.k) & 65535u) / 65535.0,
 *                      0.0, 1.0);
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/* The three bands of the single-sample image, as half-open ranges in x. */
#define BAND_MASK_X0   0
#define BAND_MASK_X1  16
#define BAND_BLEND_X0 16
#define BAND_BLEND_X1 32
#define BAND_CLEAR_X0 32
#define BAND_CLEAR_X1 64

/* The push constant of each draw, chosen so no two draws agree on a channel. */
#define K_BASE     1u
#define K_MASK   200u
#define K_BLEND    7u
#define K_MSAA    23u

/* (0.25, 0.75) packed as unorm16, which is what the clear must produce. */
#define CLEAR_R 16384u
#define CLEAR_G 49151u
/* The clear is packed on the host by util_format_pack_rgba rather than by the
 * writeback kernel, so it is allowed to round the other way; one part in
 * 65535 is still four hundred times finer than an 8-bit encoder could be. */
#define CLEAR_TOL 1u

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* u16.vert */
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

/* u16.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000034u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x00000018u, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000008u, 0x00000078u,
   0x00060005u, 0x0000000cu, 0x465f6c67u, 0x43676172u, 0x64726f6fu, 0x00000000u,
   0x00030005u, 0x00000012u, 0x00000079u, 0x00050005u, 0x00000018u, 0x6f635f6fu,
   0x72756f6cu, 0x00000000u, 0x00030005u, 0x0000001au, 0x00004350u, 0x00040006u,
   0x0000001au, 0x00000000u, 0x0000006bu, 0x00030005u, 0x0000001cu, 0x00006370u,
   0x00040047u, 0x0000000cu, 0x0000000bu, 0x0000000fu, 0x00040047u, 0x00000018u,
   0x0000001eu, 0x00000000u, 0x00030047u, 0x0000001au, 0x00000002u, 0x00050048u,
   0x0000001au, 0x00000000u, 0x00000023u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u,
   0x00000000u, 0x00040020u, 0x00000007u, 0x00000007u, 0x00000006u, 0x00030016u,
   0x00000009u, 0x00000020u, 0x00040017u, 0x0000000au, 0x00000009u, 0x00000004u,
   0x00040020u, 0x0000000bu, 0x00000001u, 0x0000000au, 0x0004003bu, 0x0000000bu,
   0x0000000cu, 0x00000001u, 0x0004002bu, 0x00000006u, 0x0000000du, 0x00000000u,
   0x00040020u, 0x0000000eu, 0x00000001u, 0x00000009u, 0x0004002bu, 0x00000006u,
   0x00000013u, 0x00000001u, 0x00040020u, 0x00000017u, 0x00000003u, 0x0000000au,
   0x0004003bu, 0x00000017u, 0x00000018u, 0x00000003u, 0x0003001eu, 0x0000001au,
   0x00000006u, 0x00040020u, 0x0000001bu, 0x00000009u, 0x0000001au, 0x0004003bu,
   0x0000001bu, 0x0000001cu, 0x00000009u, 0x00040015u, 0x0000001du, 0x00000020u,
   0x00000001u, 0x0004002bu, 0x0000001du, 0x0000001eu, 0x00000000u, 0x00040020u,
   0x0000001fu, 0x00000009u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x00000023u,
   0x0000ffffu, 0x0004002bu, 0x00000009u, 0x00000026u, 0x477fff00u, 0x0004002bu,
   0x00000006u, 0x00000029u, 0x00000101u, 0x0004002bu, 0x00000009u, 0x00000031u,
   0x00000000u, 0x0004002bu, 0x00000009u, 0x00000032u, 0x3f800000u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x0004003bu, 0x00000007u, 0x00000008u, 0x00000007u, 0x0004003bu, 0x00000007u,
   0x00000012u, 0x00000007u, 0x00050041u, 0x0000000eu, 0x0000000fu, 0x0000000cu,
   0x0000000du, 0x0004003du, 0x00000009u, 0x00000010u, 0x0000000fu, 0x0004006du,
   0x00000006u, 0x00000011u, 0x00000010u, 0x0003003eu, 0x00000008u, 0x00000011u,
   0x00050041u, 0x0000000eu, 0x00000014u, 0x0000000cu, 0x00000013u, 0x0004003du,
   0x00000009u, 0x00000015u, 0x00000014u, 0x0004006du, 0x00000006u, 0x00000016u,
   0x00000015u, 0x0003003eu, 0x00000012u, 0x00000016u, 0x0004003du, 0x00000006u,
   0x00000019u, 0x00000008u, 0x00050041u, 0x0000001fu, 0x00000020u, 0x0000001cu,
   0x0000001eu, 0x0004003du, 0x00000006u, 0x00000021u, 0x00000020u, 0x00050080u,
   0x00000006u, 0x00000022u, 0x00000019u, 0x00000021u, 0x000500c7u, 0x00000006u,
   0x00000024u, 0x00000022u, 0x00000023u, 0x00040070u, 0x00000009u, 0x00000025u,
   0x00000024u, 0x00050088u, 0x00000009u, 0x00000027u, 0x00000025u, 0x00000026u,
   0x0004003du, 0x00000006u, 0x00000028u, 0x00000012u, 0x00050084u, 0x00000006u,
   0x0000002au, 0x00000028u, 0x00000029u, 0x00050041u, 0x0000001fu, 0x0000002bu,
   0x0000001cu, 0x0000001eu, 0x0004003du, 0x00000006u, 0x0000002cu, 0x0000002bu,
   0x00050080u, 0x00000006u, 0x0000002du, 0x0000002au, 0x0000002cu, 0x000500c7u,
   0x00000006u, 0x0000002eu, 0x0000002du, 0x00000023u, 0x00040070u, 0x00000009u,
   0x0000002fu, 0x0000002eu, 0x00050088u, 0x00000009u, 0x00000030u, 0x0000002fu,
   0x00000026u, 0x00070050u, 0x0000000au, 0x00000033u, 0x00000027u, 0x00000030u,
   0x00000031u, 0x00000032u, 0x0003003eu, 0x00000018u, 0x00000033u, 0x000100fdu,
   0x00010038u
};

/* The shader, written a second time on the host: this is the oracle. */
static void
shader_texel(unsigned x, unsigned y, unsigned k, uint16_t out[2])
{
   out[0] = (uint16_t)((x + k) & 65535u);
   out[1] = (uint16_t)((y * 257u + k) & 65535u);
}

static void
expected_texel(unsigned x, unsigned y, uint16_t out[2])
{
   if (x >= BAND_CLEAR_X0) {
      out[0] = CLEAR_R;
      out[1] = CLEAR_G;
   } else if (x >= BAND_BLEND_X0) {
      /* dst + src, both channels, with the destination the base draw left. */
      uint16_t base[2], add[2];
      shader_texel(x, y, K_BASE, base);
      shader_texel(x, y, K_BLEND, add);
      out[0] = (uint16_t)(base[0] + add[0]);
      out[1] = (uint16_t)(base[1] + add[1]);
   } else {
      /* R from the masked draw, G still the base draw's. */
      uint16_t base[2], masked[2];
      shader_texel(x, y, K_BASE, base);
      shader_texel(x, y, K_MASK, masked);
      out[0] = masked[0];
      out[1] = base[1];
   }
}

/* An exact comparison everywhere except the cleared band. */
static int
texel_ok(unsigned x, const uint16_t got[2], const uint16_t want[2])
{
   if (x >= BAND_CLEAR_X0) {
      unsigned dr = got[0] > want[0] ? got[0] - want[0] : want[0] - got[0];
      unsigned dg = got[1] > want[1] ? got[1] - want[1] : want[1] - got[1];
      return dr <= CLEAR_TOL && dg <= CLEAR_TOL;
   }
   return got[0] == want[0] && got[1] == want[1];
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

static VkPhysicalDevice pdev;
static VkDevice dev;

static int
make_image(VkFormat format, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
           VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   CHECK(vkCreateImage(dev, &ici, NULL, img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, *img, &req);
   uint32_t type = pick_memory(pdev, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (type == UINT32_MAX)
      type = pick_memory(pdev, req.memoryTypeBits, 0);
   if (type == UINT32_MAX) { fprintf(stderr, "no image memory type\n"); return 1; }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   CHECK(vkBindImageMemory(dev, *img, *mem, 0));
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   CHECK(vkCreateImageView(dev, &vci, NULL, view));
   return 0;
}

static void
colour_barrier(VkCommandBuffer cmd, VkImage img, int to_transfer_src)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = to_transfer_src ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0,
      .dstAccessMask = to_transfer_src ? VK_ACCESS_TRANSFER_READ_BIT
                                       : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = to_transfer_src ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                   : VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = to_transfer_src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                   : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cmd,
                        to_transfer_src ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        to_transfer_src ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                        : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0, 0, NULL, 0, NULL, 1, &b);
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
   CHECK(vkEnumeratePhysicalDevices(inst, &n, &pdev));
   VkPhysicalDeviceProperties pprops;
   vkGetPhysicalDeviceProperties(pdev, &pprops);
   printf("device: %s\n", pprops.deviceName);

   const VkFormat cfmt = VK_FORMAT_R16G16_UNORM;

   /* What an application asks before it commits to the format. */
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pdev, cfmt, &fp);
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                     VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   printf("  R16G16_UNORM optimalTilingFeatures 0x%x, needed 0x%x\n",
          fp.optimalTilingFeatures, need);
   if ((fp.optimalTilingFeatures & need) != need) {
      fprintf(stderr, "R16G16_UNORM is not a blendable colour attachment here\n");
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
    * device is 1.3, exactly as cpvk_uint_attachment does it. */
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

   /* How many samples the driver says this format can be rendered with. */
   VkPhysicalDeviceImageFormatInfo2 ifi = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = cfmt, .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
   VkImageFormatProperties2 ifp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
   CHECK(vkGetPhysicalDeviceImageFormatProperties2(pdev, &ifi, &ifp));
   const int msaa4 =
      (ifp.imageFormatProperties.sampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0;
   printf("  sampleCounts for a colour attachment 0x%x (4x %s)\n",
          ifp.imageFormatProperties.sampleCounts, msaa4 ? "yes" : "no");

   VkImage img, msaa_img = VK_NULL_HANDLE, resolve_img = VK_NULL_HANDLE;
   VkDeviceMemory imem, msaa_mem = VK_NULL_HANDLE, resolve_mem = VK_NULL_HANDLE;
   VkImageView view, msaa_view = VK_NULL_HANDLE, resolve_view = VK_NULL_HANDLE;
   if (make_image(cfmt, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &img, &imem, &view))
      return 1;
   if (msaa4) {
      if (make_image(cfmt, VK_SAMPLE_COUNT_4_BIT,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                     &msaa_img, &msaa_mem, &msaa_view))
         return 1;
      if (make_image(cfmt, VK_SAMPLE_COUNT_1_BIT,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     &resolve_img, &resolve_mem, &resolve_view))
         return 1;
   }

   const VkDeviceSize frame_bytes = (VkDeviceSize)W * H * 4;
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = frame_bytes * 2,
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
   VkPipelineMultisampleStateCreateInfo ms1 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineMultisampleStateCreateInfo ms4 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_4_BIT };
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

   /* Three single-sample pipelines, plus a 4x one for the resolve. */
   VkGraphicsPipelineCreateInfo gpi[4];
   for (int i = 0; i < 4; i++) {
      gpi[i] = (VkGraphicsPipelineCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext = &pri, .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = i == 3 ? &ms4 : &ms1, .pDepthStencilState = &ds,
         .pColorBlendState = &cb[i == 3 ? 0 : i], .pDynamicState = &dsi,
         .layout = layout };
   }
   VkPipeline pipes[4];
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, msaa4 ? 4 : 3, gpi,
                                   NULL, pipes));

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
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
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   colour_barrier(cmd, img, 0);

   VkRenderingAttachmentInfo at = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   at.clearValue.color.float32[0] = 0.25f;
   at.clearValue.color.float32[1] = 0.75f;
   at.clearValue.color.float32[3] = 1.0f;
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
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(k), &k);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   /* Pass 1: the masked draw, then the blended one. */
   at.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[1]);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &band_mask);
   k = K_MASK;
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(k), &k);
   vkCmdDraw(cmd, 3, 1, 0, 0);

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[2]);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &band_blend);
   k = K_BLEND;
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(k), &k);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   colour_barrier(cmd, img, 1);
   VkBufferImageCopy copy = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback, 1, &copy);

   /* Pass 2, when the driver offers it: 4x samples resolved by averaging. The
    * scissor is on a pixel boundary, so a pixel's four samples are either all
    * drawn or all clear and the average of them is exact either way. */
   if (msaa4) {
      colour_barrier(cmd, msaa_img, 0);
      colour_barrier(cmd, resolve_img, 0);
      VkRenderingAttachmentInfo mat = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = msaa_view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT,
         .resolveImageView = resolve_view,
         .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
      mat.clearValue.color.float32[0] = 0.25f;
      mat.clearValue.color.float32[1] = 0.75f;
      mat.clearValue.color.float32[3] = 1.0f;
      VkRenderingInfo mri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
         .colorAttachmentCount = 1, .pColorAttachments = &mat };
      const VkRect2D half = { { 0, 0 }, { BAND_CLEAR_X0, H } };
      begin_rendering(cmd, &mri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[3]);
      vkCmdSetViewport(cmd, 0, 1, &vp);
      vkCmdSetScissor(cmd, 0, 1, &half);
      k = K_MSAA;
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(k), &k);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      end_rendering(cmd);

      colour_barrier(cmd, resolve_img, 1);
      VkBufferImageCopy rcopy = {
         .bufferOffset = frame_bytes,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { W, H, 1 } };
      vkCmdCopyImageToBuffer(cmd, resolve_img,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback, 1, &rcopy);
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

   const uint16_t *texels = (const uint16_t *)mapped;
   unsigned bad = 0, bad_clear = 0, bad_mask = 0, bad_blend = 0;
   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const uint16_t *got = texels + ((size_t)y * W + x) * 2;
         uint16_t want[2];
         expected_texel(x, y, want);
         if (texel_ok(x, got, want))
            continue;
         bad++;
         if (x >= BAND_CLEAR_X0) bad_clear++;
         else if (x >= BAND_BLEND_X0) bad_blend++;
         else bad_mask++;
         if (bad <= 3)
            printf("  (%u,%u) got %u %u  want %u %u\n", x, y,
                   got[0], got[1], want[0], want[1]);
      }
   }
   printf("  masked band  x %d..%d   %s\n", BAND_MASK_X0, BAND_MASK_X1 - 1,
          bad_mask ? "WRONG" : "exact");
   printf("  blend band   x %d..%d  %s\n", BAND_BLEND_X0, BAND_BLEND_X1 - 1,
          bad_blend ? "WRONG" : "exact");
   printf("  cleared band x %d..%d  %s\n", BAND_CLEAR_X0, BAND_CLEAR_X1 - 1,
          bad_clear ? "WRONG" : "exact");

   unsigned bad_msaa = 0;
   if (msaa4) {
      const uint16_t *rtex = (const uint16_t *)(mapped + frame_bytes);
      for (unsigned y = 0; y < H; y++) {
         for (unsigned x = 0; x < W; x++) {
            const uint16_t *got = rtex + ((size_t)y * W + x) * 2;
            uint16_t want[2];
            if (x < BAND_CLEAR_X0) {
               shader_texel(x, y, K_MSAA, want);
               if (got[0] == want[0] && got[1] == want[1])
                  continue;
            } else {
               want[0] = CLEAR_R; want[1] = CLEAR_G;
               unsigned dr = got[0] > want[0] ? got[0] - want[0] : want[0] - got[0];
               unsigned dg = got[1] > want[1] ? got[1] - want[1] : want[1] - got[1];
               if (dr <= CLEAR_TOL && dg <= CLEAR_TOL)
                  continue;
            }
            if (bad_msaa < 3)
               printf("  resolve (%u,%u) got %u %u  want %u %u\n", x, y,
                      got[0], got[1], want[0], want[1]);
            bad_msaa++;
         }
      }
      printf("  4x resolve                %s\n", bad_msaa ? "WRONG" : "exact");
   }

   /*
    * The refusal. R32G32_SFLOAT is in the format table with a texel decode
    * and no colour encoding, and the driver says so: no COLOR_ATTACHMENT_BIT.
    * An application that renders into it anyway used to get RGBA8 bytes and
    * VK_SUCCESS; it must now get an error out of vkEndCommandBuffer.
    */
   VkFormatProperties bad_fp;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R32G32_SFLOAT, &bad_fp);
   int refused = 1;
   if (bad_fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) {
      printf("  R32G32_SFLOAT now claims COLOR_ATTACHMENT; skipping the "
             "refusal check (pick another unencodable format)\n");
   } else {
      VkImage bimg; VkDeviceMemory bimem; VkImageView bview;
      if (make_image(VK_FORMAT_R32G32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &bimg, &bimem, &bview))
         return 1;
      VkRenderingAttachmentInfo bat = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = bview,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
      VkRenderingInfo bri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
         .colorAttachmentCount = 1, .pColorAttachments = &bat };
      CHECK(vkResetCommandBuffer(cmd, 0));
      CHECK(vkBeginCommandBuffer(cmd, &bi));
      begin_rendering(cmd, &bri);
      end_rendering(cmd);
      VkResult end = vkEndCommandBuffer(cmd);
      printf("  unencodable colour attachment -> vkEndCommandBuffer %d "
             "(want %d)\n", end, VK_ERROR_FEATURE_NOT_PRESENT);
      refused = end == VK_ERROR_FEATURE_NOT_PRESENT;
      vkDestroyImageView(dev, bview, NULL);
      vkDestroyImage(dev, bimg, NULL);
      vkFreeMemory(dev, bimem, NULL);
   }

   printf("%s: %u of %d texels differ, %u of %d resolved texels differ, "
          "refusal %s\n", (bad || bad_msaa || !refused) ? "FAIL" : "PASS",
          bad, W * H, bad_msaa, msaa4 ? W * H : 0,
          refused ? "ok" : "MISSING");

   vkUnmapMemory(dev, bmem);
   for (int i = 0; i < (msaa4 ? 4 : 3); i++)
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
   if (msaa4) {
      vkDestroyImageView(dev, msaa_view, NULL);
      vkDestroyImage(dev, msaa_img, NULL);
      vkFreeMemory(dev, msaa_mem, NULL);
      vkDestroyImageView(dev, resolve_view, NULL);
      vkDestroyImage(dev, resolve_img, NULL);
      vkFreeMemory(dev, resolve_mem, NULL);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   return (bad || bad_msaa || !refused) ? 1 : 0;
}
