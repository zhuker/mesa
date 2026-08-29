/*
 * VK_FORMAT_D16_UNORM as a *sampled* image: rendered as a depth attachment in
 * one pass, then read back by texture() in a second one.
 *
 * cpvk_d16_attachment.c next door proves the other half -- that the driver can
 * clear, test, store and load a two-byte depth attachment. It never binds one
 * to a descriptor, and that is the hole this test exists for. The format row
 * carried no texel decode, so a D16 image bound as a combined image sampler
 * fell through the sampler's format switch to its default arm and every lookup
 * returned opaque black. Nothing failed: no error, no validation message, just
 * (0,0,0,1) out of every texture() -- which in a shadow-mapping application is
 * "everything is at depth zero", so whole faces render near-black with hard
 * polygon edges and it looks like a shadow bug rather than a missing decode.
 *
 * The arms:
 *
 *   - pass A renders a depth ramp into a D16 attachment created with
 *     DEPTH_STENCIL_ATTACHMENT | SAMPLED | TRANSFER_SRC, over a clear that is
 *     above the ramp everywhere. The left band holds the ramp, the right band
 *     holds the clear exactly. The raw sixteen-bit values are copied out
 *     first, so the sampled result is checked against what is really in the
 *     image rather than against a model of it -- a store bug and a decode bug
 *     cannot cancel here.
 *   - pass B transitions the image to DEPTH_STENCIL_READ_ONLY_OPTIMAL, binds
 *     it as VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER with a NEAREST,
 *     CLAMP_TO_EDGE, compareEnable = VK_FALSE sampler over
 *     VK_IMAGE_ASPECT_DEPTH_BIT, and draws a full-screen quad whose fragment
 *     shader writes the raw texture() result -- all four components -- into an
 *     R32G32B32A32_SFLOAT attachment. The quad is one texel per pixel, so
 *     texel (x,y) of the depth image lands on pixel (x,y) and the whole image
 *     is checked, not a sample of it.
 *
 *     .r must be the stored code over 65535, within one quantisation step.
 *     .g .b .a are the depth-format component fill, which both oracles agree
 *     is (0, 0, 1): Vulkan's texel output conversion supplies zero for the
 *     missing G and B and one for the missing A. That is also exactly what a
 *     driver with no decode at all returns, which is why .r is the check that
 *     detects the bug and .g .b .a are only there to catch a decode that
 *     scatters the value into the wrong component.
 *
 *   - the LINEAR arm, kept separate and deletable, samples half a texel to the
 *     right of each pixel centre over the smooth part of the ramp and reports
 *     whether D16 filtering interpolates. It only runs if the implementation
 *     advertises VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT for D16 --
 *     creating a LINEAR sampler for a view of a format without that bit is
 *     invalid usage, not a failure -- and whether it can fail the test at all
 *     is LINEAR_ARM_FATAL below. The D16 optimalTilingFeatures word is printed
 *     unconditionally, decoded, because whether this driver may advertise the
 *     linear bit is decided from what the oracles do here.
 *
 * Optimal-tiled device-local images copied back through buffers, so this runs
 * unchanged on lavapipe as the oracle, and on NVIDIA. SPIR-V is embedded.
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
 *
 *   -- sample.frag --
 *   #version 450
 *   layout(location=0) in vec2 uv;
 *   layout(location=0) out vec4 o;
 *   layout(set=0, binding=0) uniform sampler2D depthTex;
 *   layout(push_constant) uniform Push { vec2 off; } pc;
 *   void main(){ o = texture(depthTex, uv + pc.off); }
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
 * Two vertical bands, so one image holds an interpolated value and an exact
 * one:
 *   x <  SPLIT   pass A's ramp, 256 codes apart texel to texel
 *   x >= SPLIT   the clear, exactly quantise(CLEAR_DEPTH)
 */
#define SPLIT 32

/* z_ndc = RAMP_SLOPE * (x + 0.5) / W, from the vertex shader above. */
#define RAMP_SLOPE 0.25

#define CLEAR_DEPTH 0.75f

/*
 * How far a stored integer may sit from the model, in sixteen-bit codes. The
 * same slack cpvk_d16_attachment.c allows, and for the same reason: the value
 * is an integer compared with an integer, so this is entirely the slack in the
 * interpolated float that produced it.
 */
#define TOL_LSB 8

/*
 * How far the sampled float may sit from the code that is actually in the
 * image, in units of 1/65535. One step is the quantisation of the decode
 * itself; the black a missing decode returns is 49151 steps away in the clear
 * band, so nothing about this tolerance is delicate.
 */
#define TOL_SAMPLED_LSB 1.5

/* ---------------------------------------------------------------------- */
/* LINEAR ARM. Delete from here to the matching marker to remove it.       */
/*                                                                        */
/* 0: the linear-filter arm measures and reports, and cannot fail the      */
/*    test. 1: it fails the test when D16 is advertised as linear-         */
/*    filterable and does not interpolate.                                 */
#define LINEAR_ARM_FATAL 0
/* How far the interpolated sample may sit from the average of the two     */
/* neighbouring codes, in units of 1/65535.                                */
#define TOL_LINEAR_LSB 2.0
/* ---------------------------------------------------------------------- */

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
   0x0003003eu, 0x00000032u, 0x00000030u, 0x000100fdu, 0x00010038u,
};

/* fragz.frag */
static const uint32_t fragz_spv[] = {
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
   0x00000017u, 0x0003003eu, 0x00000009u, 0x00000018u, 0x000100fdu, 0x00010038u,
};

/* sample.frag */
static const uint32_t samp_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000001du, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00000011u, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000009u, 0x0000006fu,
   0x00050005u, 0x0000000du, 0x74706564u, 0x78655468u, 0x00000000u, 0x00030005u,
   0x00000011u, 0x00007675u, 0x00040005u, 0x00000013u, 0x68737550u, 0x00000000u,
   0x00040006u, 0x00000013u, 0x00000000u, 0x0066666fu, 0x00030005u, 0x00000015u,
   0x00006370u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u,
   0x0000000du, 0x00000021u, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000022u,
   0x00000000u, 0x00040047u, 0x00000011u, 0x0000001eu, 0x00000000u, 0x00030047u,
   0x00000013u, 0x00000002u, 0x00050048u, 0x00000013u, 0x00000000u, 0x00000023u,
   0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u,
   0x00000004u, 0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu,
   0x00000008u, 0x00000009u, 0x00000003u, 0x00090019u, 0x0000000au, 0x00000006u,
   0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u,
   0x0003001bu, 0x0000000bu, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000000u,
   0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000000u, 0x00040017u,
   0x0000000fu, 0x00000006u, 0x00000002u, 0x00040020u, 0x00000010u, 0x00000001u,
   0x0000000fu, 0x0004003bu, 0x00000010u, 0x00000011u, 0x00000001u, 0x0003001eu,
   0x00000013u, 0x0000000fu, 0x00040020u, 0x00000014u, 0x00000009u, 0x00000013u,
   0x0004003bu, 0x00000014u, 0x00000015u, 0x00000009u, 0x00040015u, 0x00000016u,
   0x00000020u, 0x00000001u, 0x0004002bu, 0x00000016u, 0x00000017u, 0x00000000u,
   0x00040020u, 0x00000018u, 0x00000009u, 0x0000000fu, 0x00050036u, 0x00000002u,
   0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
   0x0000000bu, 0x0000000eu, 0x0000000du, 0x0004003du, 0x0000000fu, 0x00000012u,
   0x00000011u, 0x00050041u, 0x00000018u, 0x00000019u, 0x00000015u, 0x00000017u,
   0x0004003du, 0x0000000fu, 0x0000001au, 0x00000019u, 0x00050081u, 0x0000000fu,
   0x0000001bu, 0x00000012u, 0x0000001au, 0x00050057u, 0x00000007u, 0x0000001cu,
   0x0000000eu, 0x0000001bu, 0x0003003eu, 0x00000009u, 0x0000001cu, 0x000100fdu,
   0x00010038u,
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
make_image(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
           VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
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

/* The format feature bits this test has an opinion about, named. */
static void
print_features(VkFormatFeatureFlags f)
{
   static const struct { VkFormatFeatureFlags bit; const char *name; } names[] = {
      { VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT, "SAMPLED_IMAGE" },
      { VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
        "SAMPLED_IMAGE_FILTER_LINEAR" },
      { VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        "DEPTH_STENCIL_ATTACHMENT" },
      { VK_FORMAT_FEATURE_TRANSFER_SRC_BIT, "TRANSFER_SRC" },
      { VK_FORMAT_FEATURE_TRANSFER_DST_BIT, "TRANSFER_DST" },
      { VK_FORMAT_FEATURE_BLIT_SRC_BIT, "BLIT_SRC" },
      { VK_FORMAT_FEATURE_BLIT_DST_BIT, "BLIT_DST" },
      { VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, "STORAGE_IMAGE" },
   };
   for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++)
      fprintf(stderr, "%s%s%s", i ? " " : "", (f & names[i].bit) ? "+" : "-",
              names[i].name);
   fprintf(stderr, "\n");
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
   int failures = 0;

   /*
    * The advertisement, printed on stderr whatever it says, because the
    * question "may this driver claim the linear-filter bit for D16" is
    * answered by running this test on the implementations that already do.
    */
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pdev, dfmt, &fp);
   fprintf(stderr, "D16_UNORM optimalTilingFeatures 0x%08x: ",
           fp.optimalTilingFeatures);
   print_features(fp.optimalTilingFeatures);
   const int adv_sampled =
      (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
   const int adv_linear =
      (fp.optimalTilingFeatures &
       VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
   const VkFormatFeatureFlags need =
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if ((fp.optimalTilingFeatures & need) != need) {
      fprintf(stderr, "FAIL D16_UNORM is not a depth attachment here\n");
      return 1;
   }
   /*
    * Not a reason to stop: the image is created with SAMPLED usage anyway,
    * exactly as the captured application does, so that a driver which
    * refuses the advertisement but serves the descriptor is still measured
    * on what it returns.
    */
   if (!adv_sampled) {
      fprintf(stderr, "FAIL D16_UNORM does not advertise SAMPLED_IMAGE; "
              "sampling it anyway to see what comes back\n");
      failures++;
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
    * device is 1.3, exactly as cpvk_d16_attachment.c does it. */
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

   const VkFormat cfmt = VK_FORMAT_R32G32B32A32_SFLOAT;

   /* The colour attachment pass A writes gl_FragCoord.z into, the depth
    * attachment it writes the ramp into and pass B samples, and one colour
    * attachment per sampled arm. */
   VkImage aimg, dimg, nimg, limg;
   VkDeviceMemory amem, dmem, nmem, lmem;
   VkImageView aview, dview, nview, lview;
   const VkImageUsageFlags cusage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   if (make_image(cfmt, cusage, VK_IMAGE_ASPECT_COLOR_BIT,
                  &aimg, &amem, &aview))
      return 1;
   if (make_image(cfmt, cusage, VK_IMAGE_ASPECT_COLOR_BIT,
                  &nimg, &nmem, &nview))
      return 1;
   if (make_image(cfmt, cusage, VK_IMAGE_ASPECT_COLOR_BIT,
                  &limg, &lmem, &lview))
      return 1;
   /* The usage under test: an attachment that is also a texture. */
   if (make_image(dfmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, &dimg, &dmem, &dview))
      return 1;

   VkBuffer abuf, dbuf, nbuf, lbuf;
   VkDeviceMemory abmem, dbmem, nbmem, lbmem;
   if (make_buffer((VkDeviceSize)W * H * 16, &abuf, &abmem)) return 1;
   if (make_buffer((VkDeviceSize)W * H * 16, &nbuf, &nbmem)) return 1;
   if (make_buffer((VkDeviceSize)W * H * 16, &lbuf, &lbmem)) return 1;
   if (make_buffer((VkDeviceSize)W * H * 2, &dbuf, &dbmem)) return 1;

   VkShaderModule vs, fz, ft;
   VkShaderModuleCreateInfo smi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv), .pCode = vert_spv };
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &vs));
   smi.codeSize = sizeof(fragz_spv); smi.pCode = fragz_spv;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &fz));
   smi.codeSize = sizeof(samp_spv); smi.pCode = samp_spv;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &ft));

   /* One combined image sampler in the fragment stage, and a vec2 of push
    * constants that moves the lookup off the texel centre for the linear
    * arm. */
   VkDescriptorSetLayoutBinding bind = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
   VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &bind };
   VkDescriptorSetLayout set_layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &set_layout));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               2 * sizeof(float) };

   VkPipelineLayoutCreateInfo pli_a = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   VkPipelineLayoutCreateInfo pli_b = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
   VkPipelineLayout layout_a, layout_b;
   CHECK(vkCreatePipelineLayout(dev, &pli_a, NULL, &layout_a));
   CHECK(vkCreatePipelineLayout(dev, &pli_b, NULL, &layout_b));

   VkPipelineShaderStageCreateInfo stages_a[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fz, .pName = "main" },
   };
   VkPipelineShaderStageCreateInfo stages_b[2] = {
      stages_a[0],
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = ft, .pName = "main" },
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
   VkPipelineDepthStencilStateCreateInfo ds_a = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS, .maxDepthBounds = 1.0f };
   VkPipelineDepthStencilStateCreateInfo ds_b = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .maxDepthBounds = 1.0f };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xF };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkDynamicState dyn_states[2] = { VK_DYNAMIC_STATE_VIEWPORT,
                                    VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dsi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn_states };
   VkPipelineRenderingCreateInfo pri_a = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt,
      .depthAttachmentFormat = dfmt };
   /* Pass B has no depth attachment at all: the D16 image is a texture in it,
    * and an image cannot be both in one render pass. */
   VkPipelineRenderingCreateInfo pri_b = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt,
      .depthAttachmentFormat = VK_FORMAT_UNDEFINED };

   VkGraphicsPipelineCreateInfo gpi[2] = {
      { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pri_a, .stageCount = 2, .pStages = stages_a,
        .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vps, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds_a,
        .pColorBlendState = &cb, .pDynamicState = &dsi, .layout = layout_a },
      { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pri_b, .stageCount = 2, .pStages = stages_b,
        .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vps, .pRasterizationState = &rs,
        .pMultisampleState = &ms, .pDepthStencilState = &ds_b,
        .pColorBlendState = &cb, .pDynamicState = &dsi, .layout = layout_b },
   };
   VkPipeline pipes[2];
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 2, gpi, NULL, pipes));

   /*
    * NEAREST is the arm that decides the test: one texel per pixel, no
    * filtering to blur a wrong decode into a plausible one. compareEnable is
    * false, so this is a plain depth read and not a shadow comparison -- the
    * distinction the black result hides, because a comparison against a
    * missing decode also returns zero.
    */
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .compareEnable = VK_FALSE, .maxLod = 0.25f,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE };
   VkSampler samp_nearest, samp_linear = VK_NULL_HANDLE;
   CHECK(vkCreateSampler(dev, &sci, NULL, &samp_nearest));
   /* ---- LINEAR ARM ---- */
   const int run_linear = adv_linear;
   if (run_linear) {
      sci.magFilter = VK_FILTER_LINEAR;
      sci.minFilter = VK_FILTER_LINEAR;
      CHECK(vkCreateSampler(dev, &sci, NULL, &samp_linear));
   } else {
      fprintf(stderr, "linear arm skipped: D16_UNORM does not advertise "
              "SAMPLED_IMAGE_FILTER_LINEAR here\n");
   }
   /* ---- end LINEAR ARM ---- */

   VkDescriptorPoolSize psize = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 };
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &psize };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
   VkDescriptorSetLayout set_layouts[2] = { set_layout, set_layout };
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = run_linear ? 2u : 1u,
      .pSetLayouts = set_layouts };
   VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
   CHECK(vkAllocateDescriptorSets(dev, &dsai, sets));

   /* The layout the depth image is sampled in, and the one the descriptor
    * has to name: a depth aspect read that is not a write. */
   const VkImageLayout read_layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
   VkDescriptorImageInfo dii[2] = {
      { samp_nearest, dview, read_layout },
      { samp_linear, dview, read_layout },
   };
   VkWriteDescriptorSet writes[2];
   for (int i = 0; i < 2; i++)
      writes[i] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = sets[i], .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &dii[i] };
   vkUpdateDescriptorSets(dev, run_linear ? 2 : 1, writes, 0, NULL);

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

   barrier(cmd, aimg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
   barrier(cmd, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

   /* Pass A: clear the whole attachment to CLEAR_DEPTH, draw the ramp over
    * the left band. The right band keeps the clear, exactly. */
   VkRenderingAttachmentInfo cat = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = aview,
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
   const VkRect2D left = { { 0, 0 }, { SPLIT, H } };
   begin_rendering(cmd, &ri);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[0]);
   vkCmdSetViewport(cmd, 0, 1, &vp);
   vkCmdSetScissor(cmd, 0, 1, &left);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   end_rendering(cmd);

   /* The raw sixteen-bit contents, before anything samples them: the oracle
    * the sampled arm is checked against. */
   barrier(cmd, dimg, VK_IMAGE_ASPECT_DEPTH_BIT,
           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
           VK_ACCESS_TRANSFER_READ_BIT);
   VkBufferImageCopy dcopy = {
      .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, dimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          dbuf, 1, &dcopy);
   barrier(cmd, aimg, VK_IMAGE_ASPECT_COLOR_BIT,
           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
   VkBufferImageCopy ccopy = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cmd, aimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          abuf, 1, &ccopy);

   /* Pass B: the same image, now a texture. */
   barrier(cmd, dimg, VK_IMAGE_ASPECT_DEPTH_BIT,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, read_layout,
           VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
           VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);

   const float off_zero[2] = { 0.0f, 0.0f };
   /* Half a texel to the right: the midpoint of two neighbouring texels, so
    * a filter that interpolates returns their average and one that does not
    * returns one of them. */
   const float off_half[2] = { 0.5f / (float)W, 0.0f };
   VkImageView arm_view[2] = { nview, lview };
   VkImage arm_img[2] = { nimg, limg };
   VkBuffer arm_buf[2] = { nbuf, lbuf };
   const float *arm_off[2] = { off_zero, off_half };

   for (int arm = 0; arm < (run_linear ? 2 : 1); arm++) {
      barrier(cmd, arm_img[arm], VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
      VkRenderingAttachmentInfo bcat = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = arm_view[arm],
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
      VkRenderingInfo bri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = full, .layerCount = 1,
         .colorAttachmentCount = 1, .pColorAttachments = &bcat };
      begin_rendering(cmd, &bri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[1]);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_b,
                              0, 1, &sets[arm], 0, NULL);
      vkCmdPushConstants(cmd, layout_b, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         2 * sizeof(float), arm_off[arm]);
      vkCmdSetViewport(cmd, 0, 1, &vp);
      vkCmdSetScissor(cmd, 0, 1, &full);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      end_rendering(cmd);

      barrier(cmd, arm_img[arm], VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT);
      vkCmdCopyImageToBuffer(cmd, arm_img[arm],
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             arm_buf[arm], 1, &ccopy);
   }

   CHECK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   uint16_t *depth = malloc((size_t)W * H * 2);
   float *fragz = malloc((size_t)W * H * 16);
   float *nearest = malloc((size_t)W * H * 16);
   float *linear = malloc((size_t)W * H * 16);
   if (!depth || !fragz || !nearest || !linear) return 1;
   void *mapped;
   CHECK(vkMapMemory(dev, dbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(depth, mapped, (size_t)W * H * 2);
   vkUnmapMemory(dev, dbmem);
   CHECK(vkMapMemory(dev, abmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(fragz, mapped, (size_t)W * H * 16);
   vkUnmapMemory(dev, abmem);
   CHECK(vkMapMemory(dev, nbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
   memcpy(nearest, mapped, (size_t)W * H * 16);
   vkUnmapMemory(dev, nbmem);
   if (run_linear) {
      CHECK(vkMapMemory(dev, lbmem, 0, VK_WHOLE_SIZE, 0, &mapped));
      memcpy(linear, mapped, (size_t)W * H * 16);
      vkUnmapMemory(dev, lbmem);
   }

   /*
    * Pass A first: if the ramp is not in the image, nothing the sampler
    * returns means anything. This is cpvk_d16_attachment.c's check in one
    * line per band, and it is here to keep a store regression from being
    * reported as a decode regression.
    */
   const unsigned clear_code = quantise(CLEAR_DEPTH);
   long worst_store = 0;
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++) {
         double want_z = x < SPLIT ? ramp_ndc(x) : CLEAR_DEPTH;
         long err = (long)depth[y * W + x] - (long)quantise(want_z);
         if (err < 0) err = -err;
         if (err > worst_store) worst_store = err;
         if (err > (x < SPLIT ? TOL_LSB : 0) && failures++ < 8)
            fprintf(stderr, "FAIL stored depth (%u,%u): %u, wanted %u for "
                    "z %.6f\n", x, y, depth[y * W + x], quantise(want_z),
                    want_z);
         if (x < SPLIT) {
            double got = fragz[(y * W + x) * 4];
            if (fabs(got - want_z) > 1e-4 && failures++ < 8)
               fprintf(stderr, "FAIL pass A (%u,%u): gl_FragCoord.z %.6f, "
                       "wanted %.6f\n", x, y, got, want_z);
         }
      }
   printf("  pass A: stored ramp worst %ld codes of %d allowed, clear code "
          "%u\n", worst_store, TOL_LSB, clear_code);

   /*
    * The arm that decides it. Every texel of the image, against the code
    * that is actually in the image.
    *
    * .g .b .a are the depth-format fill both oracles produce -- Vulkan's
    * texel output conversion, zero for the components the format does not
    * have and one for alpha. A driver returning opaque black satisfies these
    * three and fails .r, which is the point: the missing decode is invisible
    * in every component except the one that carries the value.
    */
   const double tol = TOL_SAMPLED_LSB / 65535.0;
   double worst_r = 0.0;
   unsigned worst_x = 0, worst_y = 0;
   int black = 0;
   /* Three texels named on stderr whatever the verdict is -- one near the
    * start of the ramp, one in the middle of it, one in the untouched clear
    * band -- so that two implementations can be compared component by
    * component without re-running either. */
   static const unsigned probe[3] = { 0, 16, W - 1 };
   for (int i = 0; i < 3; i++) {
      const unsigned x = probe[i];
      const float *px = &nearest[x * 4];
      fprintf(stderr, "sampled texel (%u,0) = (%.6f, %.6f, %.6f, %.6f), "
              "the image holds code %u = %.6f\n", x, px[0], px[1], px[2],
              px[3], depth[x], depth[x] / 65535.0);
   }
   for (unsigned y = 0; y < H; y++)
      for (unsigned x = 0; x < W; x++) {
         const float *px = &nearest[(y * W + x) * 4];
         double want = depth[y * W + x] / 65535.0;
         double err = fabs((double)px[0] - want);
         if (err > worst_r) { worst_r = err; worst_x = x; worst_y = y; }
         if (err > tol) {
            if (px[0] == 0.0f) black++;
            if (failures++ < 8)
               fprintf(stderr, "FAIL sampled (%u,%u): texture() returned "
                       "(%.6f, %.6f, %.6f, %.6f), the image holds code %u = "
                       "%.6f%s\n", x, y, px[0], px[1], px[2], px[3],
                       depth[y * W + x], want,
                       px[0] == 0.0f && px[1] == 0.0f && px[2] == 0.0f &&
                       px[3] == 1.0f ? " -- opaque black, the no-decode "
                       "result" : "");
         }
         if ((px[1] != 0.0f || px[2] != 0.0f || px[3] != 1.0f) &&
             failures++ < 8)
            fprintf(stderr, "FAIL sampled (%u,%u): .gba (%.6f, %.6f, %.6f), "
                    "wanted the depth fill (0, 0, 1)\n", x, y, px[1], px[2],
                    px[3]);
      }
   printf("  NEAREST sample: worst %.1f codes at (%u,%u), %d allowed\n",
          worst_r * 65535.0, worst_x, worst_y, (int)TOL_SAMPLED_LSB);
   if (black)
      fprintf(stderr, "FAIL %d of %d texels came back opaque black: the "
              "sampler has no decode for D16_UNORM\n", black, W * H);

   /* ------------------------------------------------------------------ */
   /* LINEAR ARM. Delete from here to the matching marker to remove it.   */
   if (run_linear) {
      /* Only the smooth interior of the ramp: the band edge at SPLIT is a
       * step, and the image edge is where CLAMP_TO_EDGE takes over. */
      double worst_interp = 0.0, worst_nearest = 0.0;
      unsigned wi_x = 0;
      int interp_fail = 0;
      for (unsigned y = 0; y < H; y++)
         for (unsigned x = 2; x + 2 < SPLIT; x++) {
            double got = linear[(y * W + x) * 4];
            double a = depth[y * W + x] / 65535.0;
            double b = depth[y * W + x + 1] / 65535.0;
            double mid = 0.5 * (a + b);
            double e_mid = fabs(got - mid);
            double e_near = fabs(got - a);
            if (e_mid > worst_interp) { worst_interp = e_mid; wi_x = x; }
            if (e_near > worst_nearest) worst_nearest = e_near;
            if (e_mid > TOL_LINEAR_LSB / 65535.0)
               interp_fail++;
         }
      printf("  LINEAR sample at the texel midpoint: worst %.1f codes from "
             "the interpolated value, %.1f codes from the nearer texel -- "
             "D16 filtering %s\n", worst_interp * 65535.0,
             worst_nearest * 65535.0,
             interp_fail ? "does NOT interpolate" : "interpolates");
      if (interp_fail) {
         fprintf(stderr, "%s D16_UNORM advertises "
                 "SAMPLED_IMAGE_FILTER_LINEAR but %d midpoint samples are "
                 "not the average of their two texels (worst at x=%u)\n",
                 LINEAR_ARM_FATAL ? "FAIL" : "note:", interp_fail, wi_x);
         if (LINEAR_ARM_FATAL)
            failures++;
      }
   }
   /* ---- end LINEAR ARM ---- */
   /* ------------------------------------------------------------------ */

   free(depth);
   free(fragz);
   free(nearest);
   free(linear);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroySampler(dev, samp_nearest, NULL);
   if (samp_linear)
      vkDestroySampler(dev, samp_linear, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   for (int i = 0; i < 2; i++)
      vkDestroyPipeline(dev, pipes[i], NULL);
   vkDestroyPipelineLayout(dev, layout_a, NULL);
   vkDestroyPipelineLayout(dev, layout_b, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fz, NULL);
   vkDestroyShaderModule(dev, ft, NULL);
   vkDestroyImageView(dev, aview, NULL);
   vkDestroyImageView(dev, dview, NULL);
   vkDestroyImageView(dev, nview, NULL);
   vkDestroyImageView(dev, lview, NULL);
   vkDestroyImage(dev, aimg, NULL);
   vkDestroyImage(dev, dimg, NULL);
   vkDestroyImage(dev, nimg, NULL);
   vkDestroyImage(dev, limg, NULL);
   vkFreeMemory(dev, amem, NULL);
   vkFreeMemory(dev, dmem, NULL);
   vkFreeMemory(dev, nmem, NULL);
   vkFreeMemory(dev, lmem, NULL);
   vkDestroyBuffer(dev, abuf, NULL);
   vkDestroyBuffer(dev, dbuf, NULL);
   vkDestroyBuffer(dev, nbuf, NULL);
   vkDestroyBuffer(dev, lbuf, NULL);
   vkFreeMemory(dev, abmem, NULL);
   vkFreeMemory(dev, dbmem, NULL);
   vkFreeMemory(dev, nbmem, NULL);
   vkFreeMemory(dev, lbmem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   if (failures) {
      fprintf(stderr, "d16 sampled: %d failure%s\n", failures,
              failures == 1 ? "" : "s");
      return 1;
   }
   puts("d16 sampled: pass");
   return 0;
}
