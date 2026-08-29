/*
 * VK_FORMAT_D16_UNORM as a depth attachment: cleared, tested against, written
 * to, loaded back in a second pass, multisample-resolved, and read back as
 * sixteen-bit integers.
 *
 * The format was not in cpvk_formats[] at all, so it reported no features and
 * vkCmdBeginRendering refused it -- "unsupported depth attachment format 124",
 * VK_ERROR_FEATURE_NOT_PRESENT out of vkEndCommandBuffer, and a captured
 * application that uses a D16 depth buffer could not be replayed past the
 * frame that creates one.
 *
 * Adding the row is not the whole fix, and this test is shaped around the
 * part that is not. The driver's working depth buffer is always a 32-bit
 * sortable uint, whatever the attachment format; only the load and store
 * kernels touch the image. Both of them read and wrote a whole 32-bit word
 * per texel unconditionally, which for a two-byte texel is a misaligned
 * access that takes the neighbouring texel with it -- and, on the last texel
 * of the image, memory that is not the image at all. So the checks here are
 * per-texel and neighbour-sensitive:
 *
 *   - the store arm. Pass A clears to 0.75 over the whole attachment and
 *     draws a depth ramp over the left band. The band nothing ever draws
 *     into must hold the clear value exactly, and the ramp must be monotonic
 *     texel by texel: a 32-bit store would put its high half into the next
 *     texel and race the thread that owns it, which shows up as either a
 *     zero or a copy of a neighbour.
 *   - the load arm. Pass B has loadOp LOAD, so the attachment is decoded back
 *     into the working buffer, and draws a second ramp that is *above* the
 *     first one where pass A drew and *below* the 0.75 clear where it did
 *     not. The depth test therefore rejects every fragment of the left band
 *     and accepts every fragment of the middle one, which is a decision that
 *     can only come from the loaded sixteen-bit values. A load that read 32
 *     bits per texel would decode a number that is not a depth at all.
 *   - the quantisation. The stored integers are compared against
 *     round(depth * 65535) directly, not through a float, so a store that
 *     kept the D24 or D32 packing is off by orders of magnitude rather than
 *     by a rounding step.
 *   - the multisample advertisement. cpvk_image.c offers 4x and 8x for every
 *     format whose row says `depth`, so adding D16 offers them for D16 too.
 *     The 4x pass here renders at a constant window depth -- minDepth equal
 *     to maxDepth -- so every sample of every pixel holds the same value and
 *     VK_RESOLVE_MODE_SAMPLE_ZERO has an exact oracle that does not depend on
 *     where the sample positions are.
 *
 * The colour attachment carries gl_FragCoord.z, which says which pass won
 * each pixel; it is what separates "the depth test used the loaded value"
 * from "the depth test happened not to run".
 *
 * Optimal-tiled device-local images copied back through buffers, so this runs
 * unchanged on lavapipe as the oracle, and on NVIDIA. SPIR-V is embedded, and
 * is the same pair cpvk_depth_convention.c uses.
 *
 * Generated with glslangValidator from:
 *
 *   -- ramp.vert --
 *   #version 450
 *   layout(location=0) out vec2 uv;
 *   void main(){
 *     vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
 *     uv = p;
 *     gl_Position = vec4(p * 2.0 - 1.0, p.x * 0.25, 1.0);
 *   }
 *
 *   -- fragz.frag --
 *   #version 450
 *   layout(location=0) in vec2 uv;
 *   layout(location=0) out vec4 o;
 *   void main(){ o = vec4(gl_FragCoord.z, gl_FragCoord.w, gl_FragCoord.x, 1.0); }
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/*
 * Three vertical bands, so that one image holds all three answers:
 *   x <  SPLIT      pass A's ramp survives (pass B is behind it and fails)
 *   x <  B_RIGHT    pass B's ramp wins (pass A never drew here)
 *   x >= B_RIGHT    nothing drew: the LOAD_OP_CLEAR value, exactly
 */
#define SPLIT 32
#define B_RIGHT 48

/* z_ndc = RAMP_SLOPE * (x + 0.5) / W, from the vertex shader above. */
#define RAMP_SLOPE 0.25

#define CLEAR_DEPTH 0.75f

/* Pass B's viewport depth range: above pass A's ramp everywhere, below the
 * clear everywhere. 0.125 + 0.0625*u against 0.25*u -- they would cross at
 * u = 2/3, which is inside the half pass A never drew. */
#define B_MIN 0.125f
#define B_MAX 0.375f

/* The constant depth the 4x pass renders at, chosen so that its quantised
 * form is not a round number and is nowhere near any other value here. */
#define MSAA_DEPTH 0.3125f

/*
 * How far a stored integer may sit from the model. The value itself is exact
 * -- it is an integer compared with an integer -- so this is entirely the
 * slack in the interpolated float that produced it, and it is the same slack
 * cpvk_depth_convention.c allows (1e-4) expressed in sixteen-bit steps.
 * Values that are not interpolated (the clear, and the 4x constant) are
 * required to be exact.
 */
#define TOL_LSB 8

/* The ramp moves by 65535 * 0.25 / 64 = 256 codes per texel, so a neighbour
 * that leaked into this one is not a rounding step. */
#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* ramp.vert */
static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000033u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0008000fu, 0x00000000u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x00000018u, 0x00000020u,
   0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
   0x00000000u, 0x00030005u, 0x00000009u, 0x00000070u, 0x00060005u, 0x0000000cu,
   0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u, 0x00030005u, 0x00000018u,
   0x00007675u, 0x00060005u, 0x0000001eu, 0x505f6c67u, 0x65567265u, 0x78657472u,
   0x00000000u, 0x00060006u, 0x0000001eu, 0x00000000u, 0x505f6c67u, 0x7469736fu,
   0x006e6f69u, 0x00070006u, 0x0000001eu, 0x00000001u, 0x505f6c67u, 0x746e696fu,
   0x657a6953u, 0x00000000u, 0x00070006u, 0x0000001eu, 0x00000002u, 0x435f6c67u,
   0x4470696cu, 0x61747369u, 0x0065636eu, 0x00070006u, 0x0000001eu, 0x00000003u,
   0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u, 0x00000020u,
   0x00000000u, 0x00040047u, 0x0000000cu, 0x0000000bu, 0x0000002au, 0x00040047u,
   0x00000018u, 0x0000001eu, 0x00000000u, 0x00030047u, 0x0000001eu, 0x00000002u,
   0x00050048u, 0x0000001eu, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u,
   0x0000001eu, 0x00000001u, 0x0000000bu, 0x00000001u, 0x00050048u, 0x0000001eu,
   0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u, 0x0000001eu, 0x00000003u,
   0x0000000bu, 0x00000004u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
   0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
   0x00000006u, 0x00000002u, 0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u,
   0x00040015u, 0x0000000au, 0x00000020u, 0x00000001u, 0x00040020u, 0x0000000bu,
   0x00000001u, 0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000001u,
   0x0004002bu, 0x0000000au, 0x0000000eu, 0x00000001u, 0x0004002bu, 0x0000000au,
   0x00000010u, 0x00000002u, 0x00040020u, 0x00000017u, 0x00000003u, 0x00000007u,
   0x0004003bu, 0x00000017u, 0x00000018u, 0x00000003u, 0x00040017u, 0x0000001au,
   0x00000006u, 0x00000004u, 0x00040015u, 0x0000001bu, 0x00000020u, 0x00000000u,
   0x0004002bu, 0x0000001bu, 0x0000001cu, 0x00000001u, 0x0004001cu, 0x0000001du,
   0x00000006u, 0x0000001cu, 0x0006001eu, 0x0000001eu, 0x0000001au, 0x00000006u,
   0x0000001du, 0x0000001du, 0x00040020u, 0x0000001fu, 0x00000003u, 0x0000001eu,
   0x0004003bu, 0x0000001fu, 0x00000020u, 0x00000003u, 0x0004002bu, 0x0000000au,
   0x00000021u, 0x00000000u, 0x0004002bu, 0x00000006u, 0x00000023u, 0x40000000u,
   0x0004002bu, 0x00000006u, 0x00000025u, 0x3f800000u, 0x0004002bu, 0x0000001bu,
   0x00000028u, 0x00000000u, 0x00040020u, 0x00000029u, 0x00000007u, 0x00000006u,
   0x0004002bu, 0x00000006u, 0x0000002cu, 0x3e800000u, 0x00040020u, 0x00000031u,
   0x00000003u, 0x0000001au, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000008u, 0x00000009u,
   0x00000007u, 0x0004003du, 0x0000000au, 0x0000000du, 0x0000000cu, 0x000500c4u,
   0x0000000au, 0x0000000fu, 0x0000000du, 0x0000000eu, 0x000500c7u, 0x0000000au,
   0x00000011u, 0x0000000fu, 0x00000010u, 0x0004006fu, 0x00000006u, 0x00000012u,
   0x00000011u, 0x0004003du, 0x0000000au, 0x00000013u, 0x0000000cu, 0x000500c7u,
   0x0000000au, 0x00000014u, 0x00000013u, 0x00000010u, 0x0004006fu, 0x00000006u,
   0x00000015u, 0x00000014u, 0x00050050u, 0x00000007u, 0x00000016u, 0x00000012u,
   0x00000015u, 0x0003003eu, 0x00000009u, 0x00000016u, 0x0004003du, 0x00000007u,
   0x00000019u, 0x00000009u, 0x0003003eu, 0x00000018u, 0x00000019u, 0x0004003du,
   0x00000007u, 0x00000022u, 0x00000009u, 0x0005008eu, 0x00000007u, 0x00000024u,
   0x00000022u, 0x00000023u, 0x00050050u, 0x00000007u, 0x00000026u, 0x00000025u,
   0x00000025u, 0x00050083u, 0x00000007u, 0x00000027u, 0x00000024u, 0x00000026u,
   0x00050041u, 0x00000029u, 0x0000002au, 0x00000009u, 0x00000028u, 0x0004003du,
   0x00000006u, 0x0000002bu, 0x0000002au, 0x00050085u, 0x00000006u, 0x0000002du,
   0x0000002bu, 0x0000002cu, 0x00050051u, 0x00000006u, 0x0000002eu, 0x00000027u,
   0x00000000u, 0x00050051u, 0x00000006u, 0x0000002fu, 0x00000027u, 0x00000001u,
   0x00070050u, 0x0000001au, 0x00000030u, 0x0000002eu, 0x0000002fu, 0x0000002du,
   0x00000025u, 0x00050041u, 0x00000031u, 0x00000032u, 0x00000020u, 0x00000021u,
   0x0003003eu, 0x00000032u, 0x00000030u, 0x000100fdu, 0x00010038u
};

/* fragz.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000001cu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0008000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x0000000bu, 0x0000001bu,
   0x00030010u, 0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u,
   0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000009u,
   0x0000006fu, 0x00060005u, 0x0000000bu, 0x465f6c67u, 0x43676172u, 0x64726f6fu,
   0x00000000u, 0x00030005u, 0x0000001bu, 0x00007675u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000bu, 0x0000000bu, 0x0000000fu,
   0x00040047u, 0x0000001bu, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00040020u, 0x0000000au, 0x00000001u, 0x00000007u, 0x0004003bu, 0x0000000au,
   0x0000000bu, 0x00000001u, 0x00040015u, 0x0000000cu, 0x00000020u, 0x00000000u,
   0x0004002bu, 0x0000000cu, 0x0000000du, 0x00000002u, 0x00040020u, 0x0000000eu,
   0x00000001u, 0x00000006u, 0x0004002bu, 0x0000000cu, 0x00000011u, 0x00000003u,
   0x0004002bu, 0x0000000cu, 0x00000014u, 0x00000000u, 0x0004002bu, 0x00000006u,
   0x00000017u, 0x3f800000u, 0x00040017u, 0x00000019u, 0x00000006u, 0x00000002u,
   0x00040020u, 0x0000001au, 0x00000001u, 0x00000019u, 0x0004003bu, 0x0000001au,
   0x0000001bu, 0x00000001u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x00050041u, 0x0000000eu, 0x0000000fu,
   0x0000000bu, 0x0000000du, 0x0004003du, 0x00000006u, 0x00000010u, 0x0000000fu,
   0x00050041u, 0x0000000eu, 0x00000012u, 0x0000000bu, 0x00000011u, 0x0004003du,
   0x00000006u, 0x00000013u, 0x00000012u, 0x00050041u, 0x0000000eu, 0x00000015u,
   0x0000000bu, 0x00000014u, 0x0004003du, 0x00000006u, 0x00000016u, 0x00000015u,
   0x00070050u, 0x00000007u, 0x00000018u, 0x00000010u, 0x00000013u, 0x00000016u,
   0x00000017u, 0x0003003eu, 0x00000009u, 0x00000018u, 0x000100fdu, 0x00010038u
};

static VkPhysicalDevice pdev;
static VkDevice dev;

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

static int
make_image(VkFormat format, VkSampleCountFlagBits samples,
           VkImageUsageFlags usage, VkImageAspectFlags aspect,
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
   uint32_t type = pick_memory(req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (type == UINT32_MAX)
      type = pick_memory(req.memoryTypeBits, 0);
   if (type == UINT32_MAX) { fprintf(stderr, "no image memory type\n"); return 1; }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   CHECK(vkBindImageMemory(dev, *img, *mem, 0));
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { aspect, 0, 1, 0, 1 } };
   CHECK(vkCreateImageView(dev, &vci, NULL, view));
   return 0;
}

static int
make_buffer(VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory)
{
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = size,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   CHECK(vkCreateBuffer(dev, &bci, NULL, buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, *buffer, &req);
   uint32_t type = pick_memory(req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type == UINT32_MAX) { fprintf(stderr, "no host-visible type\n"); return 1; }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, memory));
   CHECK(vkBindBufferMemory(dev, *buffer, *memory, 0));
   return 0;
}

static void
barrier(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspect,
        VkImageLayout from, VkImageLayout to, VkPipelineStageFlags src_stage,
        VkPipelineStageFlags dst_stage, VkAccessFlags src, VkAccessFlags dst)
{
   VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src, .dstAccessMask = dst,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { aspect, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

/* What a D16_UNORM texel holds for a window depth: the Vulkan UNORM
 * encoding, round(d * 65535), which is what the store kernel computes. */
static unsigned
quantise(double d)
{
   if (d < 0.0) d = 0.0;
   if (d > 1.0) d = 1.0;
   return (unsigned)lrint(d * 65535.0);
}

static double
ramp_ndc(unsigned x)
{
   return RAMP_SLOPE * ((double)x + 0.5) / (double)W;
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

   const VkFormat dfmt = VK_FORMAT_D16_UNORM;

   /*
    * What an application asks before it commits to the format. This is the
    * question the driver used to answer "nothing at all" for D16, which is
    * what a well-behaved client would obey -- and the captured one does not
    * ask, which is why the refusal happened at vkCmdBeginRendering instead.
    */
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pdev, dfmt, &fp);
   const VkFormatFeatureFlags need =
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   printf("  D16_UNORM optimalTilingFeatures 0x%x, needed 0x%x\n",
          fp.optimalTilingFeatures, need);
   if ((fp.optimalTilingFeatures & need) != need) {
      fprintf(stderr, "D16_UNORM is not a depth attachment here\n");
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
    * device is 1.3, exactly as cpvk_unorm16_attachment does it. */
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

   /* How many samples the driver says a D16 depth attachment can have. The
    * answer has to be honest: cpvk_image.c derives it from the format row. */
   VkPhysicalDeviceImageFormatInfo2 ifi = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = dfmt, .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT };
   VkImageFormatProperties2 ifp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
   CHECK(vkGetPhysicalDeviceImageFormatProperties2(pdev, &ifi, &ifp));
   const int msaa4 =
      (ifp.imageFormatProperties.sampleCounts & VK_SAMPLE_COUNT_4_BIT) != 0;
   printf("  sampleCounts for a D16 depth attachment 0x%x (4x %s)\n",
          ifp.imageFormatProperties.sampleCounts, msaa4 ? "yes" : "no");

   const VkFormat cfmt = VK_FORMAT_R32G32B32A32_SFLOAT;
   const VkFormat mfmt = VK_FORMAT_B8G8R8A8_UNORM;

   VkImage cimg, dimg;
   VkDeviceMemory cmem, dmem;
   VkImageView cview, dview;
   if (make_image(cfmt, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, &cimg, &cmem, &cview))
      return 1;
   if (make_image(dfmt, VK_SAMPLE_COUNT_1_BIT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, &dimg, &dmem, &dview))
      return 1;

   VkImage mcimg = VK_NULL_HANDLE, mdimg = VK_NULL_HANDLE,
           rdimg = VK_NULL_HANDLE;
   VkDeviceMemory mcmem = VK_NULL_HANDLE, mdmem = VK_NULL_HANDLE,
                  rdmem = VK_NULL_HANDLE;
   VkImageView mcview = VK_NULL_HANDLE, mdview = VK_NULL_HANDLE,
               rdview = VK_NULL_HANDLE;
   if (msaa4) {
      if (make_image(mfmt, VK_SAMPLE_COUNT_4_BIT,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, &mcimg, &mcmem, &mcview))
         return 1;
      if (make_image(dfmt, VK_SAMPLE_COUNT_4_BIT,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT, &mdimg, &mdmem, &mdview))
         return 1;
      if (make_image(dfmt, VK_SAMPLE_COUNT_1_BIT,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT, &rdimg, &rdmem, &rdview))
         return 1;
   }

   VkBuffer cbuf, dbuf, rbuf;
   VkDeviceMemory cbmem, dbmem, rbmem;
   if (make_buffer((VkDeviceSize)W * H * 16, &cbuf, &cbmem)) return 1;
   if (make_buffer((VkDeviceSize)W * H * 2, &dbuf, &dbmem)) return 1;
   if (make_buffer((VkDeviceSize)W * H * 2, &rbuf, &rbmem)) return 1;

   VkShaderModuleCreateInfo vsmi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv), .pCode = vert_spv };
   VkShaderModuleCreateInfo fsmi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(frag_spv), .pCode = frag_spv };
   VkShaderModule vs, fs;
   CHECK(vkCreateShaderModule(dev, &vsmi, NULL, &vs));
   CHECK(vkCreateShaderModule(dev, &fsmi, NULL, &fs));

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
   VkViewport vp_a = { 0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f };
   VkRect2D full = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp_a,
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
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS, .maxDepthBounds = 1.0f };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkDynamicState dyn_states[2] = { VK_DYNAMIC_STATE_VIEWPORT,
                                    VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dsi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn_states };
   VkPipelineRenderingCreateInfo pri1 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt,
      .depthAttachmentFormat = dfmt };
   VkPipelineRenderingCreateInfo pri4 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &mfmt,
      .depthAttachmentFormat = dfmt };
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   VkGraphicsPipelineCreateInfo gpi[2];
   for (int i = 0; i < 2; i++)
      gpi[i] = (VkGraphicsPipelineCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext = i ? (const void *)&pri4 : (const void *)&pri1,
         .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = i ? &ms4 : &ms1, .pDepthStencilState = &ds,
         .pColorBlendState = &cb, .pDynamicState = &dsi, .layout = layout };
   VkPipeline pipes[2];
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, msaa4 ? 2 : 1, gpi,
                                   NULL, pipes));

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
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &bi));

   barrier(cmd, cimg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
   barrier(cmd, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

   VkRenderingAttachmentInfo cat = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = cview,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   VkRenderingAttachmentInfo dat = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = dview,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   dat.clearValue.depthStencil.depth = CLEAR_DEPTH;
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = full, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &cat,
      .pDepthAttachment = &dat };

   /* Pass A: clear the whole attachment, draw the ramp over the left half. */
   const VkRect2D left = { { 0, 0 }, { SPLIT, H } };
   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[0]);
   vkCmdSetViewport(cmd, 0, 1, &vp_a);
   vkCmdSetScissor(cmd, 0, 1, &left);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   /* Pass B: load what pass A stored, draw the whole width at a depth that
    * is above the left half's ramp and below the right half's clear. */
   VkViewport vp_b = { 0.0f, 0.0f, (float)W, (float)H, B_MIN, B_MAX };
   const VkRect2D b_area = { { 0, 0 }, { B_RIGHT, H } };
   cat.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
   dat.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[0]);
   vkCmdSetViewport(cmd, 0, 1, &vp_b);
   vkCmdSetScissor(cmd, 0, 1, &b_area);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   barrier(cmd, cimg, VK_IMAGE_ASPECT_COLOR_BIT,
           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
   barrier(cmd, dimg, VK_IMAGE_ASPECT_DEPTH_BIT,
           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
           VK_ACCESS_TRANSFER_READ_BIT);
   VkBufferImageCopy ccopy = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, cimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          cbuf, 1, &ccopy);
   VkBufferImageCopy dcopy = {
      .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, dimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          dbuf, 1, &dcopy);

   /*
    * Pass C: 4x samples at one constant window depth, resolved with
    * SAMPLE_ZERO into a single-sample D16 image. minDepth == maxDepth, so
    * every sample of every pixel holds MSAA_DEPTH whatever the sample
    * positions are, and the resolve has an exact answer.
    */
   if (msaa4) {
      barrier(cmd, mcimg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
      barrier(cmd, mdimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
      barrier(cmd, rdimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

      VkRenderingAttachmentInfo mcat = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = mcview,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
      VkRenderingAttachmentInfo mdat = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = mdview,
         .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
         .resolveImageView = rdview,
         .resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
      mdat.clearValue.depthStencil.depth = CLEAR_DEPTH;
      VkRenderingInfo mri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = full, .layerCount = 1,
         .colorAttachmentCount = 1, .pColorAttachments = &mcat,
         .pDepthAttachment = &mdat };
      VkViewport vp_c = { 0.0f, 0.0f, (float)W, (float)H,
                          MSAA_DEPTH, MSAA_DEPTH };
      begin_rendering(cmd, &mri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[1]);
      vkCmdSetViewport(cmd, 0, 1, &vp_c);
      vkCmdSetScissor(cmd, 0, 1, &left);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      end_rendering(cmd);

      barrier(cmd, rdimg, VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT);
      vkCmdCopyImageToBuffer(cmd, rdimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             rbuf, 1, &dcopy);
   }

   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   uint16_t *depth = malloc((size_t)W * H * 2);
   uint16_t *resolved = malloc((size_t)W * H * 2);
   float *colour = malloc((size_t)W * H * 16);
   if (!depth || !resolved || !colour) return 1;
   void *mapped;
   CHECK(vkMapMemory(dev, dbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(depth, mapped, (size_t)W * H * 2);
   vkUnmapMemory(dev, dbmem);
   CHECK(vkMapMemory(dev, cbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(colour, mapped, (size_t)W * H * 16);
   vkUnmapMemory(dev, cbmem);
   if (msaa4) {
      CHECK(vkMapMemory(dev, rbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
      memcpy(resolved, mapped, (size_t)W * H * 2);
      vkUnmapMemory(dev, rbmem);
   }

   int failures = 0;
   const unsigned clear_code = quantise(CLEAR_DEPTH);
   long worst_ramp = 0, worst_b = 0;
   unsigned worst_ramp_x = 0, worst_b_x = 0;

   for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         unsigned got = depth[y * W + x];
         double z_a = ramp_ndc(x);                       /* range 0..1 */
         double z_b = B_MIN + (B_MAX - B_MIN) * ramp_ndc(x);
         double want_z = x < SPLIT ? z_a : x < B_RIGHT ? z_b : CLEAR_DEPTH;
         long err = (long)got - (long)quantise(want_z);
         if (err < 0) err = -err;
         /* Nothing interpolated the clear, so it has to be exact. */
         long tol = x < B_RIGHT ? TOL_LSB : 0;
         if (x < SPLIT) {
            if (err > worst_ramp) { worst_ramp = err; worst_ramp_x = x; }
         } else if (x < B_RIGHT) {
            if (err > worst_b) { worst_b = err; worst_b_x = x; }
         }
         if (err > tol && failures++ < 8)
            fprintf(stderr,
                    "depth (%u,%u): stored %u, wanted %u for z %.6f\n",
                    x, y, got, quantise(want_z), want_z);

         /* gl_FragCoord.z of whichever fragment survived: pass A's on the
          * left band, pass B's in the middle one. Anything else means the
          * depth test did not read back what the store wrote. The right
          * band was never drawn, so it holds the colour clear. */
         if (x < B_RIGHT) {
            double frag_z = colour[(y * W + x) * 4];
            if (fabs(frag_z - want_z) > 1e-4 && failures++ < 8)
               fprintf(stderr,
                       "colour (%u,%u): gl_FragCoord.z %.6f, wanted %.6f -- "
                       "the wrong pass won this pixel\n", x, y, frag_z,
                       want_z);
         }
      }
      /* Neighbour damage shows up as a break in a monotonic ramp. */
      for (unsigned x = 1; x < B_RIGHT; x++)
         if (x != SPLIT && depth[y * W + x] <= depth[y * W + x - 1] &&
             failures++ < 8)
            fprintf(stderr, "depth row %u is not monotonic at x=%u: "
                    "%u then %u\n", y, x, depth[y * W + x - 1],
                    depth[y * W + x]);
   }
   printf("  pass A ramp worst %ld codes (x=%u), pass B ramp worst %ld codes "
          "(x=%u), tolerance %d; clear code %u\n", worst_ramp, worst_ramp_x,
          worst_b, worst_b_x, TOL_LSB, clear_code);

   if (msaa4) {
      const unsigned want = quantise(MSAA_DEPTH);
      for (unsigned y = 0; y < H; y++)
         for (unsigned x = 0; x < W; x++) {
            unsigned got = resolved[y * W + x];
            unsigned expect = x < SPLIT ? want : clear_code;
            if (got != expect && failures++ < 8)
               fprintf(stderr,
                       "4x resolve (%u,%u): stored %u, wanted %u\n",
                       x, y, got, expect);
         }
      printf("  4x SAMPLE_ZERO resolve: drawn %u, cleared %u\n",
             want, clear_code);
   }

   free(depth);
   free(resolved);
   free(colour);
   vkDestroyCommandPool(dev, pool, NULL);
   for (int i = 0; i < (msaa4 ? 2 : 1); i++)
      vkDestroyPipeline(dev, pipes[i], NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyImageView(dev, cview, NULL);
   vkDestroyImageView(dev, dview, NULL);
   vkDestroyImage(dev, cimg, NULL);
   vkDestroyImage(dev, dimg, NULL);
   vkFreeMemory(dev, cmem, NULL);
   vkFreeMemory(dev, dmem, NULL);
   if (msaa4) {
      vkDestroyImageView(dev, mcview, NULL);
      vkDestroyImageView(dev, mdview, NULL);
      vkDestroyImageView(dev, rdview, NULL);
      vkDestroyImage(dev, mcimg, NULL);
      vkDestroyImage(dev, mdimg, NULL);
      vkDestroyImage(dev, rdimg, NULL);
      vkFreeMemory(dev, mcmem, NULL);
      vkFreeMemory(dev, mdmem, NULL);
      vkFreeMemory(dev, rdmem, NULL);
   }
   vkDestroyBuffer(dev, cbuf, NULL);
   vkDestroyBuffer(dev, dbuf, NULL);
   vkDestroyBuffer(dev, rbuf, NULL);
   vkFreeMemory(dev, cbmem, NULL);
   vkFreeMemory(dev, dbmem, NULL);
   vkFreeMemory(dev, rbmem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failures) {
      fprintf(stderr, "d16 attachment: %d failure%s\n", failures,
              failures == 1 ? "" : "s");
      return 1;
   }
   puts("d16 attachment: pass");
   return 0;
}
