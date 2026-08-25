/*
 * Texture-cache descriptor coverage for the standalone native Vulkan ICD.
 *
 * The fragment shader deliberately keeps the image and samplers separate:
 *
 *   binding 0  texture2D / VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
 *   binding 1  sampler   / mutable VK_FILTER_LINEAR sampler
 *   binding 2  sampler   / immutable VK_FILTER_NEAREST sampler
 *
 * It fetches the same 2x2 image through both sampler bindings.  The left half
 * must be the exact linear result (80,160,96,255), and the right half must be
 * the exact nearest texel (32,200,64,255).  Every output pixel is checked.
 *
 * A 64-element function-local lookup table selects the half.  The indirect
 * load intentionally survives as PTX local memory.  That is a known shape for
 * which cudapipe's strict HW_INLINE admission gate rejects the binary, while
 * the resource-isolated HW_FUSED texture binary is safe.  The outer process
 * enables CUDAPIPE_TEXTURE_CACHE_STATS (the hardware path is the default), captures
 * stderr, and on a cudapipe device requires modes inline=0 and fused=all hits.
 * NVIDIA and llvmpipe ignore those variables and still run the exact pixels.
 *
 * Two valid submissions exercise image-view lifetime without stale Vulkan
 * use.  After submission 0 is idle, the sampled-image descriptor is updated
 * from view A to the equivalent view B, and only then is view A destroyed.
 * The second submission must remain exact and create a fresh cache object.
 * Vulkan offers no valid way to consume a descriptor whose image view has
 * already been destroyed.  This test therefore uses the only portable
 * lifetime order: update the descriptor, wait for the old use, then destroy
 * view A.  That leaves a stale destroyed-view cookie in the driver's cache
 * registry but never makes invalid Vulkan consume it.  A destroy-then-draw
 * variant would only prove host preflight refusal; its software fallback would
 * still receive invalid application state, so this test deliberately has no
 * such mode.
 *
 * SPIR-V is embedded; nothing is read from disk.
 *
 *   cc -std=c11 -Wall -Wextra -Werror \
 *      src/cudapipe/tests/cpvk_texture_cache_descriptors.c \
 *      -o /tmp/cpvk_texture_cache_descriptors -lvulkan
 *   /tmp/cpvk_texture_cache_descriptors
 *   VK_ICD_FILENAMES=.../lvp_devenv_icd.x86_64.json \
 *      /tmp/cpvk_texture_cache_descriptors
 *   VK_ICD_FILENAMES=.../cudapipe_native_devenv_icd.x86_64.json \
 *      /tmp/cpvk_texture_cache_descriptors
 *
 * Generated with glslangValidator 11:16.4.0 from:
 *
 *   #version 450
 *   layout(set=0,binding=0) uniform texture2D tex_image;
 *   layout(set=0,binding=1) uniform sampler linear_sampler;
 *   layout(set=0,binding=2) uniform sampler immutable_nearest_sampler;
 *   layout(location=0) out vec4 out_colour;
 *   void main() {
 *      vec4 l = texture(sampler2D(tex_image, linear_sampler),
 *                       vec2(0.375, 0.5));
 *      vec4 n = texture(sampler2D(tex_image, immutable_nearest_sampler),
 *                       vec2(0.375, 0.5));
 *      const float choose_n[64] = float[64](32 zeroes, 32 ones);
 *      int x = clamp(int(gl_FragCoord.x), 0, 63);
 *      out_colour = mix(l, n, choose_n[x]);
 *   }
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64
#define TEX_W 2
#define TEX_H 2
#define VALID_FRAMES 2

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

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
   0x00000025u, 0x000100fdu, 0x00010038u,
};

static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000003eu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000023u, 0x0000002eu, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00060005u, 0x00000009u, 0x656e696cu,
   0x635f7261u, 0x756f6c6fu, 0x00000072u, 0x00050005u, 0x0000000cu, 0x5f786574u,
   0x67616d69u, 0x00000065u, 0x00060005u, 0x00000010u, 0x656e696cu, 0x735f7261u,
   0x6c706d61u, 0x00007265u, 0x00060005u, 0x00000019u, 0x7261656eu, 0x5f747365u,
   0x6f6c6f63u, 0x00007275u, 0x00090005u, 0x0000001bu, 0x756d6d69u, 0x6c626174u,
   0x656e5f65u, 0x73657261u, 0x61735f74u, 0x656c706du, 0x00000072u, 0x00030005u,
   0x00000021u, 0x00000078u, 0x00060005u, 0x00000023u, 0x465f6c67u, 0x43676172u,
   0x64726f6fu, 0x00000000u, 0x00050005u, 0x0000002eu, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00050005u, 0x00000038u, 0x65646e69u, 0x6c626178u, 0x00000065u,
   0x00040047u, 0x0000000cu, 0x00000021u, 0x00000000u, 0x00040047u, 0x0000000cu,
   0x00000022u, 0x00000000u, 0x00040047u, 0x00000010u, 0x00000021u, 0x00000001u,
   0x00040047u, 0x00000010u, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000001bu,
   0x00000021u, 0x00000002u, 0x00040047u, 0x0000001bu, 0x00000022u, 0x00000000u,
   0x00040047u, 0x00000023u, 0x0000000bu, 0x0000000fu, 0x00040047u, 0x0000002eu,
   0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u,
   0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
   0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x00040020u, 0x0000000bu, 0x00000000u,
   0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000000u, 0x0002001au,
   0x0000000eu, 0x00040020u, 0x0000000fu, 0x00000000u, 0x0000000eu, 0x0004003bu,
   0x0000000fu, 0x00000010u, 0x00000000u, 0x0003001bu, 0x00000012u, 0x0000000au,
   0x00040017u, 0x00000014u, 0x00000006u, 0x00000002u, 0x0004002bu, 0x00000006u,
   0x00000015u, 0x3ec00000u, 0x0004002bu, 0x00000006u, 0x00000016u, 0x3f000000u,
   0x0005002cu, 0x00000014u, 0x00000017u, 0x00000015u, 0x00000016u, 0x0004003bu,
   0x0000000fu, 0x0000001bu, 0x00000000u, 0x00040015u, 0x0000001fu, 0x00000020u,
   0x00000001u, 0x00040020u, 0x00000020u, 0x00000007u, 0x0000001fu, 0x00040020u,
   0x00000022u, 0x00000001u, 0x00000007u, 0x0004003bu, 0x00000022u, 0x00000023u,
   0x00000001u, 0x00040015u, 0x00000024u, 0x00000020u, 0x00000000u, 0x0004002bu,
   0x00000024u, 0x00000025u, 0x00000000u, 0x00040020u, 0x00000026u, 0x00000001u,
   0x00000006u, 0x0004002bu, 0x0000001fu, 0x0000002au, 0x00000000u, 0x0004002bu,
   0x0000001fu, 0x0000002bu, 0x0000003fu, 0x00040020u, 0x0000002du, 0x00000003u,
   0x00000007u, 0x0004003bu, 0x0000002du, 0x0000002eu, 0x00000003u, 0x0004002bu,
   0x00000024u, 0x00000031u, 0x00000040u, 0x0004001cu, 0x00000032u, 0x00000006u,
   0x00000031u, 0x0004002bu, 0x00000006u, 0x00000033u, 0x00000000u, 0x0004002bu,
   0x00000006u, 0x00000034u, 0x3f800000u, 0x0043002cu, 0x00000032u, 0x00000035u,
   0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u,
   0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u,
   0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u,
   0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u,
   0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u, 0x00000033u,
   0x00000033u, 0x00000033u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u,
   0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u,
   0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u,
   0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u,
   0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u,
   0x00000034u, 0x00000034u, 0x00000034u, 0x00000034u, 0x00040020u, 0x00000037u,
   0x00000007u, 0x00000032u, 0x00040020u, 0x00000039u, 0x00000007u, 0x00000006u,
   0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
   0x00000005u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000007u, 0x0004003bu,
   0x00000008u, 0x00000019u, 0x00000007u, 0x0004003bu, 0x00000020u, 0x00000021u,
   0x00000007u, 0x0004003bu, 0x00000037u, 0x00000038u, 0x00000007u, 0x0004003du,
   0x0000000au, 0x0000000du, 0x0000000cu, 0x0004003du, 0x0000000eu, 0x00000011u,
   0x00000010u, 0x00050056u, 0x00000012u, 0x00000013u, 0x0000000du, 0x00000011u,
   0x00050057u, 0x00000007u, 0x00000018u, 0x00000013u, 0x00000017u, 0x0003003eu,
   0x00000009u, 0x00000018u, 0x0004003du, 0x0000000au, 0x0000001au, 0x0000000cu,
   0x0004003du, 0x0000000eu, 0x0000001cu, 0x0000001bu, 0x00050056u, 0x00000012u,
   0x0000001du, 0x0000001au, 0x0000001cu, 0x00050057u, 0x00000007u, 0x0000001eu,
   0x0000001du, 0x00000017u, 0x0003003eu, 0x00000019u, 0x0000001eu, 0x00050041u,
   0x00000026u, 0x00000027u, 0x00000023u, 0x00000025u, 0x0004003du, 0x00000006u,
   0x00000028u, 0x00000027u, 0x0004006eu, 0x0000001fu, 0x00000029u, 0x00000028u,
   0x0008000cu, 0x0000001fu, 0x0000002cu, 0x00000001u, 0x0000002du, 0x00000029u,
   0x0000002au, 0x0000002bu, 0x0003003eu, 0x00000021u, 0x0000002cu, 0x0004003du,
   0x00000007u, 0x0000002fu, 0x00000009u, 0x0004003du, 0x00000007u, 0x00000030u,
   0x00000019u, 0x0004003du, 0x0000001fu, 0x00000036u, 0x00000021u, 0x0003003eu,
   0x00000038u, 0x00000035u, 0x00050041u, 0x00000039u, 0x0000003au, 0x00000038u,
   0x00000036u, 0x0004003du, 0x00000006u, 0x0000003bu, 0x0000003au, 0x00070050u,
   0x00000007u, 0x0000003cu, 0x0000003bu, 0x0000003bu, 0x0000003bu, 0x0000003bu,
   0x0008000cu, 0x00000007u, 0x0000003du, 0x00000001u, 0x0000002eu, 0x0000002fu,
   0x00000030u, 0x0000003cu, 0x0003003eu, 0x0000002eu, 0x0000003du, 0x000100fdu,
   0x00010038u,
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
assert_native_stats(const char *log)
{
   const char *device = strstr(log, "cpvk-test-device: ");
   if (!device || strncmp(device + strlen("cpvk-test-device: "),
                          "cudapipe (", strlen("cudapipe (")))
      return 0;

   const char *hw = strstr(log, "cudapipe: hardware texture: ");
   const char *fallbacks = hw ? strstr(hw, " fallbacks shader=") : NULL;
   const char *modes = hw ? strstr(hw, " modes inline=") : NULL;
   unsigned long long hits = 0, launches = 0, shader_fb = 0, descriptor_fb = 0;
   unsigned long long inline_hits = 0, fused_hits = 0;
   if (!hw || !fallbacks || !modes ||
       sscanf(hw, "cudapipe: hardware texture: %llu/%llu fragment launches hit",
              &hits, &launches) != 2 ||
       sscanf(fallbacks, " fallbacks shader=%llu descriptor=%llu",
              &shader_fb, &descriptor_fb) != 2 ||
       sscanf(modes, " modes inline=%llu fused=%llu",
              &inline_hits, &fused_hits) != 2) {
      fprintf(stderr, "FAIL native run: cannot parse hardware texture stats\n");
      return 1;
   }

   if (inline_hits || fused_hits != hits || hits < VALID_FRAMES) {
      fprintf(stderr, "FAIL native run: expected only HW_FUSED hits, got "
              "hits=%llu inline=%llu fused=%llu\n",
              hits, inline_hits, fused_hits);
      return 1;
   }
   if (hits != launches || shader_fb || descriptor_fb) {
      fprintf(stderr, "FAIL native run: valid descriptors did not hit the "
              "cache on every launch (%llu/%llu, shader=%llu descriptor=%llu)\n",
              hits, launches, shader_fb, descriptor_fb);
      return 1;
   }

   const char *resources = strstr(log, "cudapipe: texture cache resources: ");
   unsigned long long resource_hits = 0, resource_fb = 0, rebuilds = 0;
   unsigned long long arrays = 0, objects = 0;
   const char *array_stats = resources ? strstr(resources, " arrays=") : NULL;
   if (!resources || !array_stats ||
       sscanf(resources,
              "cudapipe: texture cache resources: hits=%llu fallbacks=%llu "
              "rebuilds=%llu", &resource_hits, &resource_fb, &rebuilds) != 3 ||
       sscanf(array_stats, " arrays=%llu objects=%llu",
              &arrays, &objects) != 2) {
      fprintf(stderr, "FAIL native run: cannot parse texture resource stats\n");
      return 1;
   }
   if (objects < 4 || arrays < 1 || resource_hits < hits || resource_fb) {
      fprintf(stderr, "FAIL native run: unexpected resource stats: hits=%llu "
              "fallbacks=%llu rebuilds=%llu arrays=%llu objects=%llu\n",
              resource_hits, resource_fb, rebuilds, arrays, objects);
      return 1;
   }

   printf("PASS native cache used HW_FUSED only (%llu/%llu launches; "
          "%llu texture objects)\n", hits, launches, objects);
   return 0;
}

static int
run_wrapped(const char *self)
{
   int fds[2];
   if (pipe(fds)) {
      fprintf(stderr, "pipe failed: %s\n", strerror(errno));
      return 1;
   }
   pid_t pid = fork();
   if (pid < 0) {
      fprintf(stderr, "fork failed: %s\n", strerror(errno));
      return 1;
   }
   if (pid == 0) {
      close(fds[0]);
      if (dup2(fds[1], STDERR_FILENO) < 0)
         _exit(126);
      close(fds[1]);
      unsetenv("CUDAPIPE_NO_TEXTURE_CACHE");
      setenv("CUDAPIPE_TEXTURE_CACHE_STATS", "1", 1);
      char *const args[] = { (char *)self, "--child", NULL };
      execvp(self, args);
      _exit(127);
   }

   close(fds[1]);
   size_t used = 0, cap = 4096;
   char *log = malloc(cap);
   if (!log)
      return 1;
   for (;;) {
      if (used + 2048 + 1 > cap) {
         cap *= 2;
         char *grown = realloc(log, cap);
         if (!grown) {
            free(log);
            return 1;
         }
         log = grown;
      }
      ssize_t got = read(fds[0], log + used, cap - used - 1);
      if (got > 0) {
         used += (size_t)got;
         continue;
      }
      if (got < 0 && errno == EINTR)
         continue;
      break;
   }
   close(fds[0]);
   log[used] = 0;
   if (used)
      fwrite(log, 1, used, stderr);

   int status = 0;
   if (waitpid(pid, &status, 0) < 0) {
      fprintf(stderr, "waitpid failed: %s\n", strerror(errno));
      free(log);
      return 1;
   }
   if (!WIFEXITED(status) || WEXITSTATUS(status)) {
      int ret = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
      free(log);
      return ret;
   }
   int ret = assert_native_stats(log);
   free(log);
   return ret;
}

static int
same_exact(const unsigned char *a, const unsigned char *b)
{
   return !memcmp(a, b, 4);
}


static int
run_child(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t ndev = 1;
   VkPhysicalDevice pdev;
   CHECK(vkEnumeratePhysicalDevices(inst, &ndev, &pdev));
   if (!ndev) {
      fprintf(stderr, "no Vulkan physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   fprintf(stderr, "cpvk-test-device: %s\n", props.deviceName);
   printf("device: %s\n", props.deviceName);

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   if (!families)
      return 1;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; i++)
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         family = i;
         break;
      }
   free(families);
   if (family == UINT32_MAX) {
      fprintf(stderr, "no graphics queue\n");
      return 1;
   }

   static const char *const wanted[] = {
      VK_KHR_MULTIVIEW_EXTENSION_NAME,
      VK_KHR_MAINTENANCE2_EXTENSION_NAME,
      VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
      VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
      VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
   };
   const uint32_t wanted_count = sizeof(wanted) / sizeof(wanted[0]);
   const int core_dynrend = props.apiVersion >= VK_API_VERSION_1_3;
   uint32_t ext_count = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, NULL));
   VkExtensionProperties *exts = calloc(ext_count, sizeof(*exts));
   if (!exts)
      return 1;
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, exts));
   const char *dev_exts[sizeof(wanted) / sizeof(wanted[0])];
   uint32_t dev_ext_count = 0;
   int have_dynrend = 0;
   for (uint32_t w = 0; w < wanted_count; w++) {
      const int is_dynrend = w == wanted_count - 1;
      if (core_dynrend && !is_dynrend)
         continue;
      for (uint32_t i = 0; i < ext_count; i++) {
         if (!strcmp(exts[i].extensionName, wanted[w])) {
            dev_exts[dev_ext_count++] = wanted[w];
            have_dynrend |= is_dynrend;
            break;
         }
      }
   }
   free(exts);
   if (!have_dynrend && !core_dynrend) {
      fprintf(stderr, "no dynamic rendering\n");
      return 1;
   }

   VkFormatProperties tfp;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R8G8B8A8_UNORM, &tfp);
   const VkFormatFeatureFlags texture_need =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((tfp.optimalTilingFeatures & texture_need) != texture_need) {
      fprintf(stderr, "R8G8B8A8 optimal image is not linearly filterable\n");
      return 1;
   }

   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family, .queueCount = 1,
      .pQueuePriorities = &priority };
   VkPhysicalDeviceDynamicRenderingFeatures dyn = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &dyn,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
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

   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkImageCreateInfo color_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = cfmt,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage color;
   CHECK(vkCreateImage(dev, &color_ci, NULL, &color));
   VkMemoryRequirements creq;
   vkGetImageMemoryRequirements(dev, color, &creq);
   uint32_t ctype = pick_memory(pdev, creq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (ctype == UINT32_MAX)
      ctype = pick_memory(pdev, creq.memoryTypeBits, 0);
   if (ctype == UINT32_MAX)
      return 1;
   VkMemoryAllocateInfo cmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = creq.size,
                                 .memoryTypeIndex = ctype };
   VkDeviceMemory color_mem;
   CHECK(vkAllocateMemory(dev, &cmai, NULL, &color_mem));
   CHECK(vkBindImageMemory(dev, color, color_mem, 0));
   VkImageViewCreateInfo cvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = color, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = cfmt,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView color_view;
   CHECK(vkCreateImageView(dev, &cvci, NULL, &color_view));

   const VkDeviceSize frame_bytes = (VkDeviceSize)W * H * 4;
   VkBufferCreateInfo rbci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = VALID_FRAMES * frame_bytes,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   VkBuffer readback;
   CHECK(vkCreateBuffer(dev, &rbci, NULL, &readback));
   VkMemoryRequirements rbreq;
   vkGetBufferMemoryRequirements(dev, readback, &rbreq);
   uint32_t rbtype = pick_memory(pdev, rbreq.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (rbtype == UINT32_MAX)
      return 1;
   VkMemoryAllocateInfo rbmai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = rbreq.size, .memoryTypeIndex = rbtype };
   VkDeviceMemory readback_mem;
   CHECK(vkAllocateMemory(dev, &rbmai, NULL, &readback_mem));
   CHECK(vkBindBufferMemory(dev, readback, readback_mem, 0));

   VkImageCreateInfo texture_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { TEX_W, TEX_H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage texture;
   CHECK(vkCreateImage(dev, &texture_ci, NULL, &texture));
   VkMemoryRequirements treq;
   vkGetImageMemoryRequirements(dev, texture, &treq);
   uint32_t ttype = pick_memory(pdev, treq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (ttype == UINT32_MAX)
      return 1;
   VkMemoryAllocateInfo tmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = treq.size,
                                 .memoryTypeIndex = ttype };
   VkDeviceMemory texture_mem;
   CHECK(vkAllocateMemory(dev, &tmai, NULL, &texture_mem));
   CHECK(vkBindImageMemory(dev, texture, texture_mem, 0));
   VkImageViewCreateInfo tvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = texture, .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = texture_ci.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView texture_view[2];
   CHECK(vkCreateImageView(dev, &tvci, NULL, &texture_view[0]));
   CHECK(vkCreateImageView(dev, &tvci, NULL, &texture_view[1]));

   VkBufferCreateInfo sbci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = TEX_W * TEX_H * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT };
   VkBuffer staging;
   CHECK(vkCreateBuffer(dev, &sbci, NULL, &staging));
   VkMemoryRequirements sreq;
   vkGetBufferMemoryRequirements(dev, staging, &sreq);
   uint32_t stype = pick_memory(pdev, sreq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (stype == UINT32_MAX)
      return 1;
   VkMemoryAllocateInfo smai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = sreq.size,
                                 .memoryTypeIndex = stype };
   VkDeviceMemory staging_mem;
   CHECK(vkAllocateMemory(dev, &smai, NULL, &staging_mem));
   CHECK(vkBindBufferMemory(dev, staging, staging_mem, 0));
   unsigned char *stage_map = NULL;
   CHECK(vkMapMemory(dev, staging_mem, 0, VK_WHOLE_SIZE, 0,
                     (void **)&stage_map));
   static const unsigned char col0[4] = { 32, 200, 64, 255 };
   static const unsigned char col1[4] = { 224, 40, 192, 255 };
   for (int y = 0; y < TEX_H; y++)
      for (int x = 0; x < TEX_W; x++)
         memcpy(stage_map + (y * TEX_W + x) * 4, x ? col1 : col0, 4);
   VkMappedMemoryRange sflush = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = staging_mem, .size = VK_WHOLE_SIZE };
   CHECK(vkFlushMappedMemoryRanges(dev, 1, &sflush));
   vkUnmapMemory(dev, staging_mem);

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
      .maxLod = 0.0f };
   VkSampler immutable_nearest, mutable_linear;
   CHECK(vkCreateSampler(dev, &sci, NULL, &immutable_nearest));
   sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
   CHECK(vkCreateSampler(dev, &sci, NULL, &mutable_linear));

   VkDescriptorSetLayoutBinding bindings[3] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = &immutable_nearest },
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = bindings };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkDescriptorPoolSize pool_sizes[2] = {
      { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1 },
      { VK_DESCRIPTOR_TYPE_SAMPLER, 2 },
   };
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = pool_sizes };
   VkDescriptorPool descriptor_pool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &descriptor_pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool, .descriptorSetCount = 1,
      .pSetLayouts = &dsl };
   VkDescriptorSet descriptor_set;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &descriptor_set));
   VkDescriptorImageInfo image_info = {
      .imageView = texture_view[0],
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
   VkDescriptorImageInfo sampler_info = { .sampler = mutable_linear };
   VkWriteDescriptorSet descriptor_writes[2] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = &image_info },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set, .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = &sampler_info },
   };
   vkUpdateDescriptorSets(dev, 2, descriptor_writes, 0, NULL);

   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv), .pCode = vert_spv };
   VkShaderModule vs, fs;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &vs));
   smci.codeSize = sizeof(frag_spv);
   smci.pCode = frag_spv;
   CHECK(vkCreateShaderModule(dev, &smci, NULL, &fs));
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl };
   VkPipelineLayout pipeline_layout;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pipeline_layout));

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
   VkViewport viewport = { 0, 0, W, H, 0, 1 };
   VkRect2D scissor = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1 };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
   VkPipelineRenderingCreateInfo rendering_ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_ci, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .layout = pipeline_layout };
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL,
                                   &pipeline));

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
   VkCommandPool command_pool;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &command_pool));
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color_view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = { 0.1f, 0.1f, 0.15f, 1.0f } };
   VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &attachment };

   for (int pass = 0; pass < VALID_FRAMES; pass++) {
      if (pass == 1) {
         /* Valid lifetime order: replace the descriptor first, then destroy
          * view A after all work which used it has completed. */
         image_info.imageView = texture_view[1];
         vkUpdateDescriptorSets(dev, 1, &descriptor_writes[0], 0, NULL);
         vkDestroyImageView(dev, texture_view[0], NULL);
         texture_view[0] = VK_NULL_HANDLE;
      }

      CHECK(vkResetCommandBuffer(cmd, 0));
      VkCommandBufferBeginInfo begin = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
      CHECK(vkBeginCommandBuffer(cmd, &begin));

      if (pass == 0) {
         VkImageMemoryBarrier tex_to_dst = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = texture,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, NULL, 0, NULL, 1, &tex_to_dst);
         VkBufferImageCopy upload = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { TEX_W, TEX_H, 1 } };
         vkCmdCopyBufferToImage(cmd, staging, texture,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
         VkImageMemoryBarrier tex_to_read = tex_to_dst;
         tex_to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
         tex_to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
         tex_to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
         tex_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                              0, NULL, 0, NULL, 1, &tex_to_read);
      }

      VkImageMemoryBarrier color_to_attachment = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = pass ? VK_ACCESS_TRANSFER_READ_BIT : 0,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = pass ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                           : VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = color,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      vkCmdPipelineBarrier(cmd,
         pass ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
         0, NULL, 0, NULL, 1, &color_to_attachment);

      begin_rendering(cmd, &rendering);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      end_rendering(cmd);

      VkImageMemoryBarrier color_to_src = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = color,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           0, NULL, 0, NULL, 1, &color_to_src);
      VkBufferImageCopy copy = {
         .bufferOffset = (VkDeviceSize)pass * frame_bytes,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
         .imageExtent = { W, H, 1 } };
      vkCmdCopyImageToBuffer(cmd, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback, 1, &copy);
      VkBufferMemoryBarrier to_host = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = readback,
         .offset = (VkDeviceSize)pass * frame_bytes,
         .size = frame_bytes };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0,
                           0, NULL, 1, &to_host, 0, NULL);
      CHECK(vkEndCommandBuffer(cmd));
      VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                              .commandBufferCount = 1,
                              .pCommandBuffers = &cmd };
      CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
      CHECK(vkQueueWaitIdle(queue));
   }

   unsigned char *mapped = NULL;
   CHECK(vkMapMemory(dev, readback_mem, 0, VK_WHOLE_SIZE, 0,
                     (void **)&mapped));
   VkMappedMemoryRange invalidate = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = readback_mem, .size = VK_WHOLE_SIZE };
   CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &invalidate));
   static const unsigned char expect_linear[4] = { 80, 160, 96, 255 };
   static const unsigned char expect_nearest[4] = { 32, 200, 64, 255 };
   int fail = 0;
   for (int pass = 0; pass < VALID_FRAMES; pass++) {
      unsigned good_linear = 0, good_nearest = 0;
      int bad_x = -1, bad_y = -1;
      unsigned char bad[4] = {0};
      const unsigned char *frame = mapped + (size_t)pass * frame_bytes;
      for (int y = 0; y < H; y++) {
         for (int x = 0; x < W; x++) {
            const unsigned char *pixel = frame + ((size_t)y * W + x) * 4;
            const unsigned char *want = x < W / 2 ? expect_linear
                                                  : expect_nearest;
            if (same_exact(pixel, want)) {
               if (x < W / 2)
                  good_linear++;
               else
                  good_nearest++;
            } else if (bad_x < 0) {
               bad_x = x;
               bad_y = y;
               memcpy(bad, pixel, 4);
            }
         }
      }
      printf("pass %d (%s view): linear exact %u/%u, immutable nearest exact "
             "%u/%u\n", pass, pass ? "replacement" : "original",
             good_linear, W * H / 2, good_nearest, W * H / 2);
      if (good_linear != W * H / 2 || good_nearest != W * H / 2) {
         fprintf(stderr, "FAIL pass %d first mismatch at (%d,%d): "
                 "%u %u %u %u\n", pass, bad_x, bad_y,
                 bad[0], bad[1], bad[2], bad[3]);
         fail = 1;
      }
   }
   if (!fail)
      printf("PASS separate sampled-image + mutable sampler + immutable sampler "
             "exact pixels across valid view replacement\n");
   vkUnmapMemory(dev, readback_mem);

   vkDestroyCommandPool(dev, command_pool, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyPipelineLayout(dev, pipeline_layout, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyDescriptorPool(dev, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroySampler(dev, mutable_linear, NULL);
   vkDestroySampler(dev, immutable_nearest, NULL);
   if (texture_view[0])
      vkDestroyImageView(dev, texture_view[0], NULL);
   vkDestroyImageView(dev, texture_view[1], NULL);
   vkDestroyBuffer(dev, staging, NULL);
   vkFreeMemory(dev, staging_mem, NULL);
   vkDestroyImage(dev, texture, NULL);
   vkFreeMemory(dev, texture_mem, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, readback_mem, NULL);
   vkDestroyImageView(dev, color_view, NULL);
   vkDestroyImage(dev, color, NULL);
   vkFreeMemory(dev, color_mem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fail;
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--child"))
      return run_child();
   if (argc != 1) {
      fprintf(stderr, "usage: %s\n", argv[0]);
      return 2;
   }
   return run_wrapped(argv[0]);
}
