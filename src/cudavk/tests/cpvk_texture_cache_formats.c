/*
 * Exact-pixel sampled-image test for the CUDA hardware texture-cache path.
 *
 * Build: cc -std=c11 -O2 -Wall -Wextra cpvk_texture_cache_formats.c \
 *           $(pkg-config --cflags --libs vulkan) -o /tmp/cpvk_texture_cache_formats
 * Run:   /tmp/cpvk_texture_cache_formats
 *
 * The SPIR-V below is embedded so this source is a standalone test.  It draws
 * eight 8-pixel strips into R32G32B32A32_SFLOAT and compares the copied float
 * bit patterns, not an 8-bit screenshot.  All sampled images are optimal tiled
 * and device-local.  Uploads use TRANSFER_DST_OPTIMAL followed by a
 * TRANSFER_WRITE -> SHADER_READ barrier.  The result uses a render-pass final
 * transition/dependency to TRANSFER_SRC_OPTIMAL before buffer readback.
 *
 * Strip 5 deliberately uses a view with baseMipLevel=1.  It is Vulkan-valid,
 * but the current cudavk hardware cache rejects nonzero base mip views and
 * should use its software fallback for that draw.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define STRIP 8u
#define CASES 8u
#define WIDTH (STRIP * CASES)

static void
vk_ok(VkResult r, const char *what)
{
   if (r != VK_SUCCESS) {
      fprintf(stderr, "%s failed: %d\n", what, r);
      exit(2);
   }
}
#define VK_OK(x) vk_ok((x), #x)

static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000028u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0007000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000018u, 0x0000001cu, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x0000000cu,
   0x00000070u, 0x00060005u, 0x00000016u, 0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u,
   0x00000016u, 0x00000000u, 0x505f6c67u, 0x7469736fu, 0x006e6f69u, 0x00070006u, 0x00000016u, 0x00000001u,
   0x505f6c67u, 0x746e696fu, 0x657a6953u, 0x00000000u, 0x00070006u, 0x00000016u, 0x00000002u, 0x435f6c67u,
   0x4470696cu, 0x61747369u, 0x0065636eu, 0x00070006u, 0x00000016u, 0x00000003u, 0x435f6c67u, 0x446c6c75u,
   0x61747369u, 0x0065636eu, 0x00030005u, 0x00000018u, 0x00000000u, 0x00060005u, 0x0000001cu, 0x565f6c67u,
   0x65747265u, 0x646e4978u, 0x00007865u, 0x00030047u, 0x00000016u, 0x00000002u, 0x00050048u, 0x00000016u,
   0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x00000016u, 0x00000001u, 0x0000000bu, 0x00000001u,
   0x00050048u, 0x00000016u, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u, 0x00000016u, 0x00000003u,
   0x0000000bu, 0x00000004u, 0x00040047u, 0x0000001cu, 0x0000000bu, 0x0000002au, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u,
   0x00000006u, 0x00000002u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u, 0x0004002bu, 0x00000008u,
   0x00000009u, 0x00000003u, 0x0004001cu, 0x0000000au, 0x00000007u, 0x00000009u, 0x00040020u, 0x0000000bu,
   0x00000007u, 0x0000000au, 0x0004002bu, 0x00000006u, 0x0000000du, 0xbf800000u, 0x0005002cu, 0x00000007u,
   0x0000000eu, 0x0000000du, 0x0000000du, 0x0004002bu, 0x00000006u, 0x0000000fu, 0x40400000u, 0x0005002cu,
   0x00000007u, 0x00000010u, 0x0000000fu, 0x0000000du, 0x0005002cu, 0x00000007u, 0x00000011u, 0x0000000du,
   0x0000000fu, 0x0006002cu, 0x0000000au, 0x00000012u, 0x0000000eu, 0x00000010u, 0x00000011u, 0x00040017u,
   0x00000013u, 0x00000006u, 0x00000004u, 0x0004002bu, 0x00000008u, 0x00000014u, 0x00000001u, 0x0004001cu,
   0x00000015u, 0x00000006u, 0x00000014u, 0x0006001eu, 0x00000016u, 0x00000013u, 0x00000006u, 0x00000015u,
   0x00000015u, 0x00040020u, 0x00000017u, 0x00000003u, 0x00000016u, 0x0004003bu, 0x00000017u, 0x00000018u,
   0x00000003u, 0x00040015u, 0x00000019u, 0x00000020u, 0x00000001u, 0x0004002bu, 0x00000019u, 0x0000001au,
   0x00000000u, 0x00040020u, 0x0000001bu, 0x00000001u, 0x00000019u, 0x0004003bu, 0x0000001bu, 0x0000001cu,
   0x00000001u, 0x00040020u, 0x0000001eu, 0x00000007u, 0x00000007u, 0x0004002bu, 0x00000006u, 0x00000021u,
   0x00000000u, 0x0004002bu, 0x00000006u, 0x00000022u, 0x3f800000u, 0x00040020u, 0x00000026u, 0x00000003u,
   0x00000013u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000007u, 0x0003003eu, 0x0000000cu, 0x00000012u, 0x0004003du,
   0x00000019u, 0x0000001du, 0x0000001cu, 0x00050041u, 0x0000001eu, 0x0000001fu, 0x0000000cu, 0x0000001du,
   0x0004003du, 0x00000007u, 0x00000020u, 0x0000001fu, 0x00050051u, 0x00000006u, 0x00000023u, 0x00000020u,
   0x00000000u, 0x00050051u, 0x00000006u, 0x00000024u, 0x00000020u, 0x00000001u, 0x00070050u, 0x00000013u,
   0x00000025u, 0x00000023u, 0x00000024u, 0x00000021u, 0x00000022u, 0x00050041u, 0x00000026u, 0x00000027u,
   0x00000018u, 0x0000001au, 0x0003003eu, 0x00000027u, 0x00000025u, 0x000100fdu, 0x00010038u,
};

static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x000000b4u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
   0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
   0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu, 0x00000026u, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du,
   0x00000000u, 0x00030005u, 0x00000008u, 0x00000069u, 0x00060005u, 0x0000000cu, 0x465f6c67u, 0x43676172u,
   0x64726f6fu, 0x00000000u, 0x00030005u, 0x00000013u, 0x00004350u, 0x00050006u, 0x00000013u, 0x00000000u,
   0x646e696bu, 0x00000000u, 0x00030005u, 0x00000015u, 0x00006370u, 0x00040005u, 0x00000026u, 0x6f6c6f63u,
   0x00000072u, 0x00030005u, 0x0000002au, 0x00003274u, 0x00030005u, 0x00000081u, 0x00000064u, 0x00030005u,
   0x000000a2u, 0x00006374u, 0x00030005u, 0x000000aau, 0x00003374u, 0x00040047u, 0x0000000cu, 0x0000000bu,
   0x0000000fu, 0x00030047u, 0x00000013u, 0x00000002u, 0x00050048u, 0x00000013u, 0x00000000u, 0x00000023u,
   0x00000000u, 0x00040047u, 0x00000026u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x0000002au, 0x00000021u,
   0x00000000u, 0x00040047u, 0x0000002au, 0x00000022u, 0x00000000u, 0x00040047u, 0x000000a2u, 0x00000021u,
   0x00000001u, 0x00040047u, 0x000000a2u, 0x00000022u, 0x00000000u, 0x00040047u, 0x000000aau, 0x00000021u,
   0x00000002u, 0x00040047u, 0x000000aau, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u,
   0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u, 0x00000001u, 0x00040020u, 0x00000007u,
   0x00000007u, 0x00000006u, 0x00030016u, 0x00000009u, 0x00000020u, 0x00040017u, 0x0000000au, 0x00000009u,
   0x00000004u, 0x00040020u, 0x0000000bu, 0x00000001u, 0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu,
   0x00000001u, 0x00040015u, 0x0000000du, 0x00000020u, 0x00000000u, 0x0004002bu, 0x0000000du, 0x0000000eu,
   0x00000000u, 0x00040020u, 0x0000000fu, 0x00000001u, 0x00000009u, 0x0003001eu, 0x00000013u, 0x00000006u,
   0x00040020u, 0x00000014u, 0x00000009u, 0x00000013u, 0x0004003bu, 0x00000014u, 0x00000015u, 0x00000009u,
   0x0004002bu, 0x00000006u, 0x00000016u, 0x00000000u, 0x00040020u, 0x00000017u, 0x00000009u, 0x00000006u,
   0x0004002bu, 0x00000006u, 0x0000001au, 0x00000008u, 0x0004002bu, 0x00000006u, 0x0000001du, 0x00000007u,
   0x00020014u, 0x00000021u, 0x00040020u, 0x00000025u, 0x00000003u, 0x0000000au, 0x0004003bu, 0x00000025u,
   0x00000026u, 0x00000003u, 0x00090019u, 0x00000027u, 0x00000009u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x00000028u, 0x00000027u, 0x00040020u, 0x00000029u,
   0x00000000u, 0x00000028u, 0x0004003bu, 0x00000029u, 0x0000002au, 0x00000000u, 0x0004002bu, 0x00000006u,
   0x0000002du, 0x00000003u, 0x0004002bu, 0x00000009u, 0x00000030u, 0x3f000000u, 0x0004002bu, 0x00000009u,
   0x00000032u, 0x40800000u, 0x00040017u, 0x00000034u, 0x00000009u, 0x00000002u, 0x0004002bu, 0x00000009u,
   0x00000036u, 0x00000000u, 0x0004002bu, 0x00000006u, 0x0000003bu, 0x00000001u, 0x0004002bu, 0x00000009u,
   0x00000044u, 0x40000000u, 0x0004002bu, 0x00000006u, 0x0000004bu, 0x00000002u, 0x0004002bu, 0x00000009u,
   0x0000005cu, 0x3e000000u, 0x0004002bu, 0x00000006u, 0x00000062u, 0x00000004u, 0x0004002bu, 0x00000006u,
   0x00000069u, 0x00000005u, 0x0005002cu, 0x00000034u, 0x0000006fu, 0x00000030u, 0x00000030u, 0x0004002bu,
   0x00000006u, 0x00000077u, 0x00000006u, 0x00040017u, 0x0000007fu, 0x00000009u, 0x00000003u, 0x00040020u,
   0x00000080u, 0x00000007u, 0x0000007fu, 0x0004002bu, 0x00000009u, 0x00000082u, 0x3f800000u, 0x0006002cu,
   0x0000007fu, 0x00000083u, 0x00000082u, 0x00000036u, 0x00000036u, 0x0004002bu, 0x00000009u, 0x00000089u,
   0xbf800000u, 0x0006002cu, 0x0000007fu, 0x0000008au, 0x00000089u, 0x00000036u, 0x00000036u, 0x0006002cu,
   0x0000007fu, 0x00000090u, 0x00000036u, 0x00000082u, 0x00000036u, 0x0006002cu, 0x0000007fu, 0x00000096u,
   0x00000036u, 0x00000089u, 0x00000036u, 0x0006002cu, 0x0000007fu, 0x0000009cu, 0x00000036u, 0x00000036u,
   0x00000082u, 0x0006002cu, 0x0000007fu, 0x0000009eu, 0x00000036u, 0x00000036u, 0x00000089u, 0x00090019u,
   0x0000009fu, 0x00000009u, 0x00000003u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u,
   0x0003001bu, 0x000000a0u, 0x0000009fu, 0x00040020u, 0x000000a1u, 0x00000000u, 0x000000a0u, 0x0004003bu,
   0x000000a1u, 0x000000a2u, 0x00000000u, 0x00090019u, 0x000000a7u, 0x00000009u, 0x00000002u, 0x00000000u,
   0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x000000a8u, 0x000000a7u, 0x00040020u,
   0x000000a9u, 0x00000000u, 0x000000a8u, 0x0004003bu, 0x000000a9u, 0x000000aau, 0x00000000u, 0x0004002bu,
   0x00000009u, 0x000000afu, 0x3f400000u, 0x0004002bu, 0x00000009u, 0x000000b0u, 0x3e800000u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000007u,
   0x00000008u, 0x00000007u, 0x0004003bu, 0x00000080u, 0x00000081u, 0x00000007u, 0x00050041u, 0x0000000fu,
   0x00000010u, 0x0000000cu, 0x0000000eu, 0x0004003du, 0x00000009u, 0x00000011u, 0x00000010u, 0x0004006eu,
   0x00000006u, 0x00000012u, 0x00000011u, 0x00050041u, 0x00000017u, 0x00000018u, 0x00000015u, 0x00000016u,
   0x0004003du, 0x00000006u, 0x00000019u, 0x00000018u, 0x00050084u, 0x00000006u, 0x0000001bu, 0x00000019u,
   0x0000001au, 0x00050082u, 0x00000006u, 0x0000001cu, 0x00000012u, 0x0000001bu, 0x000500c7u, 0x00000006u,
   0x0000001eu, 0x0000001cu, 0x0000001du, 0x0003003eu, 0x00000008u, 0x0000001eu, 0x00050041u, 0x00000017u,
   0x0000001fu, 0x00000015u, 0x00000016u, 0x0004003du, 0x00000006u, 0x00000020u, 0x0000001fu, 0x000500aau,
   0x00000021u, 0x00000022u, 0x00000020u, 0x00000016u, 0x000300f7u, 0x00000024u, 0x00000000u, 0x000400fau,
   0x00000022u, 0x00000023u, 0x00000038u, 0x000200f8u, 0x00000023u, 0x0004003du, 0x00000028u, 0x0000002bu,
   0x0000002au, 0x0004003du, 0x00000006u, 0x0000002cu, 0x00000008u, 0x000500c7u, 0x00000006u, 0x0000002eu,
   0x0000002cu, 0x0000002du, 0x0004006fu, 0x00000009u, 0x0000002fu, 0x0000002eu, 0x00050081u, 0x00000009u,
   0x00000031u, 0x0000002fu, 0x00000030u, 0x00050088u, 0x00000009u, 0x00000033u, 0x00000031u, 0x00000032u,
   0x00050050u, 0x00000034u, 0x00000035u, 0x00000033u, 0x00000030u, 0x00070058u, 0x0000000au, 0x00000037u,
   0x0000002bu, 0x00000035u, 0x00000002u, 0x00000036u, 0x0003003eu, 0x00000026u, 0x00000037u, 0x000200f9u,
   0x00000024u, 0x000200f8u, 0x00000038u, 0x00050041u, 0x00000017u, 0x00000039u, 0x00000015u, 0x00000016u,
   0x0004003du, 0x00000006u, 0x0000003au, 0x00000039u, 0x000500aau, 0x00000021u, 0x0000003cu, 0x0000003au,
   0x0000003bu, 0x000300f7u, 0x0000003eu, 0x00000000u, 0x000400fau, 0x0000003cu, 0x0000003du, 0x00000048u,
   0x000200f8u, 0x0000003du, 0x0004003du, 0x00000028u, 0x0000003fu, 0x0000002au, 0x0004003du, 0x00000006u,
   0x00000040u, 0x00000008u, 0x000500c7u, 0x00000006u, 0x00000041u, 0x00000040u, 0x0000003bu, 0x0004006fu,
   0x00000009u, 0x00000042u, 0x00000041u, 0x00050081u, 0x00000009u, 0x00000043u, 0x00000042u, 0x00000030u,
   0x00050088u, 0x00000009u, 0x00000045u, 0x00000043u, 0x00000044u, 0x00050050u, 0x00000034u, 0x00000046u,
   0x00000045u, 0x00000030u, 0x00070058u, 0x0000000au, 0x00000047u, 0x0000003fu, 0x00000046u, 0x00000002u,
   0x00000036u, 0x0003003eu, 0x00000026u, 0x00000047u, 0x000200f9u, 0x0000003eu, 0x000200f8u, 0x00000048u,
   0x00050041u, 0x00000017u, 0x00000049u, 0x00000015u, 0x00000016u, 0x0004003du, 0x00000006u, 0x0000004au,
   0x00000049u, 0x000500aau, 0x00000021u, 0x0000004cu, 0x0000004au, 0x0000004bu, 0x000400a8u, 0x00000021u,
   0x0000004du, 0x0000004cu, 0x000300f7u, 0x0000004fu, 0x00000000u, 0x000400fau, 0x0000004du, 0x0000004eu,
   0x0000004fu, 0x000200f8u, 0x0000004eu, 0x00050041u, 0x00000017u, 0x00000050u, 0x00000015u, 0x00000016u,
   0x0004003du, 0x00000006u, 0x00000051u, 0x00000050u, 0x000500aau, 0x00000021u, 0x00000052u, 0x00000051u,
   0x0000002du, 0x000200f9u, 0x0000004fu, 0x000200f8u, 0x0000004fu, 0x000700f5u, 0x00000021u, 0x00000053u,
   0x0000004cu, 0x00000048u, 0x00000052u, 0x0000004eu, 0x000300f7u, 0x00000055u, 0x00000000u, 0x000400fau,
   0x00000053u, 0x00000054u, 0x0000005fu, 0x000200f8u, 0x00000054u, 0x0004003du, 0x00000028u, 0x00000056u,
   0x0000002au, 0x0004003du, 0x00000006u, 0x00000057u, 0x00000008u, 0x000500c7u, 0x00000006u, 0x00000058u,
   0x00000057u, 0x0000002du, 0x0004006fu, 0x00000009u, 0x00000059u, 0x00000058u, 0x00050081u, 0x00000009u,
   0x0000005au, 0x00000059u, 0x00000030u, 0x00050088u, 0x00000009u, 0x0000005bu, 0x0000005au, 0x00000032u,
   0x00050050u, 0x00000034u, 0x0000005du, 0x0000005bu, 0x0000005cu, 0x00070058u, 0x0000000au, 0x0000005eu,
   0x00000056u, 0x0000005du, 0x00000002u, 0x00000036u, 0x0003003eu, 0x00000026u, 0x0000005eu, 0x000200f9u,
   0x00000055u, 0x000200f8u, 0x0000005fu, 0x00050041u, 0x00000017u, 0x00000060u, 0x00000015u, 0x00000016u,
   0x0004003du, 0x00000006u, 0x00000061u, 0x00000060u, 0x000500aau, 0x00000021u, 0x00000063u, 0x00000061u,
   0x00000062u, 0x000400a8u, 0x00000021u, 0x00000064u, 0x00000063u, 0x000300f7u, 0x00000066u, 0x00000000u,
   0x000400fau, 0x00000064u, 0x00000065u, 0x00000066u, 0x000200f8u, 0x00000065u, 0x00050041u, 0x00000017u,
   0x00000067u, 0x00000015u, 0x00000016u, 0x0004003du, 0x00000006u, 0x00000068u, 0x00000067u, 0x000500aau,
   0x00000021u, 0x0000006au, 0x00000068u, 0x00000069u, 0x000200f9u, 0x00000066u, 0x000200f8u, 0x00000066u,
   0x000700f5u, 0x00000021u, 0x0000006bu, 0x00000063u, 0x0000005fu, 0x0000006au, 0x00000065u, 0x000300f7u,
   0x0000006du, 0x00000000u, 0x000400fau, 0x0000006bu, 0x0000006cu, 0x00000074u, 0x000200f8u, 0x0000006cu,
   0x0004003du, 0x00000028u, 0x0000006eu, 0x0000002au, 0x0004003du, 0x00000006u, 0x00000070u, 0x00000008u,
   0x0005008bu, 0x00000006u, 0x00000071u, 0x00000070u, 0x0000002du, 0x0004006fu, 0x00000009u, 0x00000072u,
   0x00000071u, 0x00070058u, 0x0000000au, 0x00000073u, 0x0000006eu, 0x0000006fu, 0x00000002u, 0x00000072u,
   0x0003003eu, 0x00000026u, 0x00000073u, 0x000200f9u, 0x0000006du, 0x000200f8u, 0x00000074u, 0x00050041u,
   0x00000017u, 0x00000075u, 0x00000015u, 0x00000016u, 0x0004003du, 0x00000006u, 0x00000076u, 0x00000075u,
   0x000500aau, 0x00000021u, 0x00000078u, 0x00000076u, 0x00000077u, 0x000300f7u, 0x0000007au, 0x00000000u,
   0x000400fau, 0x00000078u, 0x00000079u, 0x000000a6u, 0x000200f8u, 0x00000079u, 0x0004003du, 0x00000006u,
   0x0000007bu, 0x00000008u, 0x000500aau, 0x00000021u, 0x0000007cu, 0x0000007bu, 0x00000016u, 0x000300f7u,
   0x0000007eu, 0x00000000u, 0x000400fau, 0x0000007cu, 0x0000007du, 0x00000084u, 0x000200f8u, 0x0000007du,
   0x0003003eu, 0x00000081u, 0x00000083u, 0x000200f9u, 0x0000007eu, 0x000200f8u, 0x00000084u, 0x0004003du,
   0x00000006u, 0x00000085u, 0x00000008u, 0x000500aau, 0x00000021u, 0x00000086u, 0x00000085u, 0x0000003bu,
   0x000300f7u, 0x00000088u, 0x00000000u, 0x000400fau, 0x00000086u, 0x00000087u, 0x0000008bu, 0x000200f8u,
   0x00000087u, 0x0003003eu, 0x00000081u, 0x0000008au, 0x000200f9u, 0x00000088u, 0x000200f8u, 0x0000008bu,
   0x0004003du, 0x00000006u, 0x0000008cu, 0x00000008u, 0x000500aau, 0x00000021u, 0x0000008du, 0x0000008cu,
   0x0000004bu, 0x000300f7u, 0x0000008fu, 0x00000000u, 0x000400fau, 0x0000008du, 0x0000008eu, 0x00000091u,
   0x000200f8u, 0x0000008eu, 0x0003003eu, 0x00000081u, 0x00000090u, 0x000200f9u, 0x0000008fu, 0x000200f8u,
   0x00000091u, 0x0004003du, 0x00000006u, 0x00000092u, 0x00000008u, 0x000500aau, 0x00000021u, 0x00000093u,
   0x00000092u, 0x0000002du, 0x000300f7u, 0x00000095u, 0x00000000u, 0x000400fau, 0x00000093u, 0x00000094u,
   0x00000097u, 0x000200f8u, 0x00000094u, 0x0003003eu, 0x00000081u, 0x00000096u, 0x000200f9u, 0x00000095u,
   0x000200f8u, 0x00000097u, 0x0004003du, 0x00000006u, 0x00000098u, 0x00000008u, 0x000500aau, 0x00000021u,
   0x00000099u, 0x00000098u, 0x00000062u, 0x000300f7u, 0x0000009bu, 0x00000000u, 0x000400fau, 0x00000099u,
   0x0000009au, 0x0000009du, 0x000200f8u, 0x0000009au, 0x0003003eu, 0x00000081u, 0x0000009cu, 0x000200f9u,
   0x0000009bu, 0x000200f8u, 0x0000009du, 0x0003003eu, 0x00000081u, 0x0000009eu, 0x000200f9u, 0x0000009bu,
   0x000200f8u, 0x0000009bu, 0x000200f9u, 0x00000095u, 0x000200f8u, 0x00000095u, 0x000200f9u, 0x0000008fu,
   0x000200f8u, 0x0000008fu, 0x000200f9u, 0x00000088u, 0x000200f8u, 0x00000088u, 0x000200f9u, 0x0000007eu,
   0x000200f8u, 0x0000007eu, 0x0004003du, 0x000000a0u, 0x000000a3u, 0x000000a2u, 0x0004003du, 0x0000007fu,
   0x000000a4u, 0x00000081u, 0x00070058u, 0x0000000au, 0x000000a5u, 0x000000a3u, 0x000000a4u, 0x00000002u,
   0x00000036u, 0x0003003eu, 0x00000026u, 0x000000a5u, 0x000200f9u, 0x0000007au, 0x000200f8u, 0x000000a6u,
   0x0004003du, 0x000000a8u, 0x000000abu, 0x000000aau, 0x0004003du, 0x00000006u, 0x000000acu, 0x00000008u,
   0x000500c7u, 0x00000006u, 0x000000adu, 0x000000acu, 0x0000003bu, 0x000500abu, 0x00000021u, 0x000000aeu,
   0x000000adu, 0x00000016u, 0x000600a9u, 0x00000009u, 0x000000b1u, 0x000000aeu, 0x000000afu, 0x000000b0u,
   0x00060050u, 0x0000007fu, 0x000000b2u, 0x00000030u, 0x00000030u, 0x000000b1u, 0x00070058u, 0x0000000au,
   0x000000b3u, 0x000000abu, 0x000000b2u, 0x00000002u, 0x00000036u, 0x0003003eu, 0x00000026u, 0x000000b3u,
   0x000200f9u, 0x0000007au, 0x000200f8u, 0x0000007au, 0x000200f9u, 0x0000006du, 0x000200f8u, 0x0000006du,
   0x000200f9u, 0x00000055u, 0x000200f8u, 0x00000055u, 0x000200f9u, 0x0000003eu, 0x000200f8u, 0x0000003eu,
   0x000200f9u, 0x00000024u, 0x000200f8u, 0x00000024u, 0x000100fdu, 0x00010038u,
};

struct image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
   VkFormat format;
   uint32_t mips, layers;
};

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceSize size;
};

static uint32_t
pick_memory(VkPhysicalDevice pdev, uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties p;
   vkGetPhysicalDeviceMemoryProperties(pdev, &p);
   for (uint32_t i = 0; i < p.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (p.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type: bits=0x%x want=0x%x\n", bits, want);
   exit(2);
}

static struct buffer
make_buffer(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size,
            VkBufferUsageFlags usage, VkMemoryPropertyFlags props)
{
   struct buffer b = { .size = size };
   VkBufferCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
   VK_OK(vkCreateBuffer(dev, &ci, NULL, &b.buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, b.buffer, &req);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = pick_memory(pdev, req.memoryTypeBits, props) };
   VK_OK(vkAllocateMemory(dev, &ai, NULL, &b.memory));
   VK_OK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
   return b;
}

static VkImageView
make_view(VkDevice dev, VkImage image, VkFormat format, VkImageViewType type,
          uint32_t base_mip, uint32_t levels, uint32_t layers)
{
   VkImageViewCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image, .viewType = type, .format = format,
      .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, base_mip, levels, 0, layers } };
   VkImageView view;
   VK_OK(vkCreateImageView(dev, &ci, NULL, &view));
   return view;
}

static struct image
make_sampled_image(VkDevice dev, VkPhysicalDevice pdev, VkFormat format,
                   VkImageType image_type, VkImageViewType view_type,
                   VkExtent3D extent, uint32_t mips, uint32_t layers,
                   VkImageCreateFlags flags)
{
   struct image im = { .format = format, .mips = mips, .layers = layers };
   VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags, .imageType = image_type, .format = format,
      .extent = extent, .mipLevels = mips, .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VK_OK(vkCreateImage(dev, &ci, NULL, &im.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, im.image, &req);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = pick_memory(pdev, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VK_OK(vkAllocateMemory(dev, &ai, NULL, &im.memory));
   VK_OK(vkBindImageMemory(dev, im.image, im.memory, 0));
   im.view = make_view(dev, im.image, format, view_type, 0, mips, layers);
   return im;
}

static void
upload(VkDevice dev, VkQueue q, VkCommandBuffer cmd, struct buffer staging,
       const struct image *im, const void *data, size_t size,
       const VkBufferImageCopy *regions, uint32_t nregions)
{
   void *map;
   VK_OK(vkMapMemory(dev, staging.memory, 0, size, 0, &map));
   memcpy(map, data, size);
   vkUnmapMemory(dev, staging.memory);
   VK_OK(vkResetCommandBuffer(cmd, 0));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   VK_OK(vkBeginCommandBuffer(cmd, &bi));
   VkImageMemoryBarrier before = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im->image,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, im->mips, 0, im->layers } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &before);
   vkCmdCopyBufferToImage(cmd, staging.buffer, im->image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, nregions, regions);
   VkImageMemoryBarrier after = before;
   after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   after.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   after.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL,
                        0, NULL, 1, &after);
   VK_OK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &cmd };
   VK_OK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE));
   VK_OK(vkQueueWaitIdle(q));
}

static VkShaderModule
shader(VkDevice dev, const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = size, .pCode = code };
   VkShaderModule m;
   VK_OK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

int
main(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "cpvk_texture_cache_formats",
                             .apiVersion = VK_API_VERSION_1_0 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VkInstance instance;
   VK_OK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 0;
   VK_OK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   if (!count) { fprintf(stderr, "no Vulkan physical device\n"); return 2; }
   VkPhysicalDevice *pdevs = calloc(count, sizeof(*pdevs));
   VK_OK(vkEnumeratePhysicalDevices(instance, &count, pdevs));
   VkPhysicalDevice pdev = pdevs[0];
   free(pdevs);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   printf("device: %s\n", props.deviceName);

   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, NULL);
   VkQueueFamilyProperties *qps = calloc(nq, sizeof(*qps));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qps);
   uint32_t qfamily = UINT32_MAX;
   for (uint32_t i = 0; i < nq; i++)
      if (qps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfamily = i; break; }
   free(qps);
   if (qfamily == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 2; }
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qfamily, .queueCount = 1, .pQueuePriorities = &priority };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
   VkDevice dev;
   VK_OK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qfamily, 0, &queue);
   VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qfamily };
   VkCommandPool pool;
   VK_OK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   VK_OK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   const VkFormat sample_formats[] = {
      VK_FORMAT_A2B10G10R10_UNORM_PACK32,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK,
      VK_FORMAT_R8G8B8A8_UNORM };
   for (unsigned i = 0; i < sizeof(sample_formats) / sizeof(sample_formats[0]); i++) {
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pdev, sample_formats[i], &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
         fprintf(stderr, "required sampled optimal format unsupported: %u\n",
                 sample_formats[i]);
         return 2;
      }
   }
   VkFormatProperties ofp;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R32G32B32A32_SFLOAT, &ofp);
   if (!(ofp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
      fprintf(stderr, "R32G32B32A32_SFLOAT color attachment unsupported\n");
      return 2;
   }

   struct buffer staging = make_buffer(dev, pdev, 4096,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkBufferImageCopy one = { .imageSubresource = {
      VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { 1, 1, 1 } };

   struct image a2 = make_sampled_image(dev, pdev,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D, (VkExtent3D){4,1,1}, 1, 1, 0);
   const uint32_t a2_data[4] = {
      0xc00003ffu, 0x000ffc00u, 0x7ff00000u, 0x801aa955u };
   one.imageExtent = (VkExtent3D){4,1,1};
   upload(dev, queue, cmd, staging, &a2, a2_data, sizeof(a2_data), &one, 1);

   struct image uf = make_sampled_image(dev, pdev,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D, (VkExtent3D){2,1,1}, 1, 1, 0);
   const uint32_t uf_data[2] = { 0x781e03c0u, 0x88200380u };
   one.imageExtent = (VkExtent3D){2,1,1};
   upload(dev, queue, cmd, staging, &uf, uf_data, sizeof(uf_data), &one, 1);

   struct image bc1 = make_sampled_image(dev, pdev, VK_FORMAT_BC1_RGB_UNORM_BLOCK,
      VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, (VkExtent3D){4,4,1}, 1, 1, 0);
   /* Red and black keep all selector results exact across conformant BC
    * decoder precision choices while still distinguishing selectors 0..3. */
   const uint8_t bc1_data[8] = { 0x00,0xf8, 0x00,0x00, 0xe4,0xe4,0xe4,0xe4 };
   one.imageExtent = (VkExtent3D){4,4,1};
   upload(dev, queue, cmd, staging, &bc1, bc1_data, sizeof(bc1_data), &one, 1);

   struct image bc3 = make_sampled_image(dev, pdev, VK_FORMAT_BC3_UNORM_BLOCK,
      VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, (VkExtent3D){4,4,1}, 1, 1, 0);
   /* alpha0=255 and alpha1=3 make selectors 2 and 7 exact integers
    * (219 and 39), avoiding interpolation precision differences. */
   const uint8_t bc3_data[16] = {
      255,3, 0x88,0x8e,0xe8,0x88,0x8e,0xe8,
      0x00,0xf8, 0x00,0x00, 0xe4,0xe4,0xe4,0xe4 };
   upload(dev, queue, cmd, staging, &bc3, bc3_data, sizeof(bc3_data), &one, 1);

   struct image mip = make_sampled_image(dev, pdev, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, (VkExtent3D){4,4,1}, 3, 1, 0);
   uint8_t mip_data[84];
   for (unsigned i = 0; i < 16; i++) memcpy(mip_data + i*4, (uint8_t[]){255,0,0,255}, 4);
   for (unsigned i = 0; i < 4; i++) memcpy(mip_data + 64+i*4, (uint8_t[]){0,255,0,255}, 4);
   memcpy(mip_data + 80, (uint8_t[]){0,0,255,255}, 4);
   VkBufferImageCopy mip_regions[3] = {
      { .bufferOffset=0, .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
        .imageExtent={4,4,1} },
      { .bufferOffset=64, .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,1,0,1},
        .imageExtent={2,2,1} },
      { .bufferOffset=80, .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,2,0,1},
        .imageExtent={1,1,1} } };
   upload(dev, queue, cmd, staging, &mip, mip_data, sizeof(mip_data), mip_regions, 3);
   VkImageView mip_base = make_view(dev, mip.image, mip.format,
                                    VK_IMAGE_VIEW_TYPE_2D, 1, 2, 1);

   struct image cube = make_sampled_image(dev, pdev, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, (VkExtent3D){1,1,1}, 1, 6,
      VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
   const uint8_t cube_data[24] = {
      255,0,0,255, 0,255,0,255, 0,0,255,255,
      255,255,0,255, 255,0,255,255, 0,255,255,255 };
   one.imageSubresource.layerCount = 6;
   one.imageExtent = (VkExtent3D){1,1,1};
   upload(dev, queue, cmd, staging, &cube, cube_data, sizeof(cube_data), &one, 1);
   one.imageSubresource.layerCount = 1;

   struct image tex3 = make_sampled_image(dev, pdev, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, (VkExtent3D){1,1,2}, 1, 1, 0);
   const uint8_t tex3_data[8] = {255,0,0,255, 0,0,255,255};
   one.imageExtent = (VkExtent3D){1,1,2};
   upload(dev, queue, cmd, staging, &tex3, tex3_data, sizeof(tex3_data), &one, 1);

   VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .minLod = 0.0f, .maxLod = 8.0f, .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK };
   VkSampler sampler;
   VK_OK(vkCreateSampler(dev, &sci, NULL, &sampler));

   VkDescriptorSetLayoutBinding bindings[3];
   memset(bindings, 0, sizeof(bindings));
   for (uint32_t i = 0; i < 3; i++) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
   }
   VkDescriptorSetLayoutCreateInfo dlci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3, .pBindings = bindings };
   VkDescriptorSetLayout dlayout;
   VK_OK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dlayout));
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, CASES*3 };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = CASES, .poolSizeCount = 1, .pPoolSizes = &dps };
   VkDescriptorPool dpool;
   VK_OK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetLayout layouts[CASES];
   for (unsigned i=0;i<CASES;i++) layouts[i]=dlayout;
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = CASES, .pSetLayouts = layouts };
   VkDescriptorSet sets[CASES];
   VK_OK(vkAllocateDescriptorSets(dev, &dsai, sets));
   VkImageView two_d[CASES] = { a2.view, uf.view, bc1.view, bc3.view,
                                mip.view, mip_base, a2.view, a2.view };
   for (unsigned s = 0; s < CASES; s++) {
      VkDescriptorImageInfo ii[3] = {
         { sampler, two_d[s], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
         { sampler, cube.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
         { sampler, tex3.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } };
      VkWriteDescriptorSet writes[3];
      memset(writes, 0, sizeof(writes));
      for (uint32_t i=0;i<3;i++) writes[i] = (VkWriteDescriptorSet) {
         .sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet=sets[s],
         .dstBinding=i, .descriptorCount=1,
         .descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo=&ii[i] };
      vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);
   }

   VkImage out_image;
   VkDeviceMemory out_memory;
   VkImageCreateInfo oici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .extent = { WIDTH, 1, 1 }, .mipLevels=1, .arrayLayers=1,
      .samples=VK_SAMPLE_COUNT_1_BIT, .tiling=VK_IMAGE_TILING_OPTIMAL,
      .usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode=VK_SHARING_MODE_EXCLUSIVE, .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED };
   VK_OK(vkCreateImage(dev, &oici, NULL, &out_image));
   VkMemoryRequirements oreq;
   vkGetImageMemoryRequirements(dev, out_image, &oreq);
   VkMemoryAllocateInfo oai = { .sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize=oreq.size,
      .memoryTypeIndex=pick_memory(pdev, oreq.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VK_OK(vkAllocateMemory(dev, &oai, NULL, &out_memory));
   VK_OK(vkBindImageMemory(dev, out_image, out_memory, 0));
   VkImageView out_view = make_view(dev, out_image, oici.format,
                                    VK_IMAGE_VIEW_TYPE_2D, 0, 1, 1);
   struct buffer readback = make_buffer(dev, pdev, WIDTH * 16,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   VkAttachmentDescription ad = { .format=oici.format, .samples=VK_SAMPLE_COUNT_1_BIT,
      .loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp=VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
   VkAttachmentReference ar = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = { .pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
                                .colorAttachmentCount=1, .pColorAttachments=&ar };
   VkSubpassDependency deps[2] = {
      { VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0 },
      { 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT, 0 } };
   VkRenderPassCreateInfo rpci = { .sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount=1, .pAttachments=&ad, .subpassCount=1, .pSubpasses=&sub,
      .dependencyCount=2, .pDependencies=deps };
   VkRenderPass rp;
   VK_OK(vkCreateRenderPass(dev, &rpci, NULL, &rp));
   VkFramebufferCreateInfo fbci = { .sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass=rp, .attachmentCount=1, .pAttachments=&out_view,
      .width=WIDTH, .height=1, .layers=1 };
   VkFramebuffer fb;
   VK_OK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 };
   VkPipelineLayoutCreateInfo plci = { .sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount=1, .pSetLayouts=&dlayout,
      .pushConstantRangeCount=1, .pPushConstantRanges=&pcr };
   VkPipelineLayout playout;
   VK_OK(vkCreatePipelineLayout(dev, &plci, NULL, &playout));
   VkShaderModule vs = shader(dev, vert_spv, sizeof(vert_spv));
   VkShaderModule fs = shader(dev, frag_spv, sizeof(frag_spv));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage=VK_SHADER_STAGE_VERTEX_BIT, .module=vs, .pName="main" },
      { .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage=VK_SHADER_STAGE_FRAGMENT_BIT, .module=fs, .pName="main" } };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkPipelineViewportStateCreateInfo vp = {
      .sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount=1, .scissorCount=1 };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode=VK_POLYGON_MODE_FILL, .cullMode=VK_CULL_MODE_NONE,
      .frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth=1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples=VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask=0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount=1, .pAttachments=&cba };
   VkDynamicState dynamics[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dyn = { .sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                            .dynamicStateCount=2, .pDynamicStates=dynamics };
   VkGraphicsPipelineCreateInfo gpci = { .sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount=2, .pStages=stages, .pVertexInputState=&vi,
      .pInputAssemblyState=&ia, .pViewportState=&vp, .pRasterizationState=&rs,
      .pMultisampleState=&ms, .pColorBlendState=&cb, .pDynamicState=&dyn,
      .layout=playout, .renderPass=rp, .subpass=0 };
   VkPipeline pipeline, base_mip_pipeline;
   VK_OK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline));
   VK_OK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL,
                                  &base_mip_pipeline));

   VK_OK(vkResetCommandBuffer(cmd, 0));
   VkCommandBufferBeginInfo begin = { .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   VK_OK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier ob = { .sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED, .newLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED, .image=out_image,
      .dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1} };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0,NULL,0,NULL,1,&ob);
   VkClearValue clear = { .color.float32={0,0,0,0} };
   VkRenderPassBeginInfo rbi = { .sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass=rp, .framebuffer=fb, .renderArea={{0,0},{WIDTH,1}},
      .clearValueCount=1, .pClearValues=&clear };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   VkViewport viewport = {0,0,WIDTH,1,0,1};
   vkCmdSetViewport(cmd, 0, 1, &viewport);
   for (uint32_t kind=0; kind<CASES; kind++) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        kind == 5 ? base_mip_pipeline : pipeline);
      VkRect2D scissor = {{(int32_t)(kind*STRIP),0},{STRIP,1}};
      vkCmdSetScissor(cmd, 0, 1, &scissor);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, playout,
                              0, 1, &sets[kind], 0, NULL);
      vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &kind);
      vkCmdDraw(cmd, 3, 1, 0, 0);
   }
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy outcopy = { .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
                                 .imageExtent={WIDTH,1,1} };
   vkCmdCopyImageToBuffer(cmd, out_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback.buffer, 1, &outcopy);
   VkBufferMemoryBarrier host = { .sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask=VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
      .buffer=readback.buffer, .offset=0, .size=VK_WHOLE_SIZE };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0,NULL,1,&host,0,NULL);
   VK_OK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = { .sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount=1, .pCommandBuffers=&cmd };
   VK_OK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_OK(vkQueueWaitIdle(queue));

   static const char *names[CASES] = {
      "A2B10G10R10", "B10G11R11", "BC1", "BC3",
      "mip-full-view", "base-mip-view", "cube", "3D" };
   static const uint32_t R=0x3f800000u, Z=0x00000000u,
                         T=0x3eaaaaabu, U=0x3f2aaaabu;
   uint32_t expect[WIDTH][4];
   const uint32_t a2e[4][4] = {
      {R,Z,Z,R},{Z,R,Z,Z},{Z,Z,R,T},{T,U,0x3a802008u,U} };
   const uint32_t ufe[2][4] = {{R,R,R,R},{0x3f000000u,0x40000000u,0x40800000u,R}};
   const uint32_t bc1e[4][4] = {
      {R,Z,Z,R},{Z,Z,Z,R},{U,Z,Z,R},{T,Z,Z,R} };
   const uint32_t bc3e[4][4] = {
      {R,Z,Z,R},{Z,Z,Z,0x3c40c0c1u},{U,Z,Z,0x3f5bdbdcu},
      {T,Z,Z,0x3e1c9c9du} };
   const uint32_t mip_e[3][4] = {{R,Z,Z,R},{Z,R,Z,R},{Z,Z,R,R}};
   const uint32_t base_e[3][4] = {{Z,R,Z,R},{Z,Z,R,R},{Z,Z,R,R}};
   const uint32_t cube_e[6][4] = {
      {R,Z,Z,R},{Z,R,Z,R},{Z,Z,R,R},{R,R,Z,R},{R,Z,R,R},{Z,R,R,R} };
   for (uint32_t i=0;i<WIDTH;i++) {
      uint32_t kind=i/STRIP, j=i&7;
      const uint32_t *e = kind==0 ? a2e[j&3] : kind==1 ? ufe[j&1] :
         kind==2 ? bc1e[j&3] : kind==3 ? bc3e[j&3] :
         kind==4 ? mip_e[j%3] : kind==5 ? base_e[j%3] :
         kind==6 ? cube_e[j<6?j:5] : mip_e[(j&1)?2:0];
      memcpy(expect[i], e, 16);
   }
   uint32_t (*got)[4];
   VK_OK(vkMapMemory(dev, readback.memory, 0, VK_WHOLE_SIZE, 0, (void **)&got));
   unsigned errors=0;
   for (uint32_t i=0;i<WIDTH;i++)
      if (memcmp(got[i], expect[i], 16)) {
         fprintf(stderr, "%s pixel %u: got %08x %08x %08x %08x; expected %08x %08x %08x %08x\n",
            names[i/STRIP], i&7, got[i][0],got[i][1],got[i][2],got[i][3],
            expect[i][0],expect[i][1],expect[i][2],expect[i][3]);
         errors++;
      }
   vkUnmapMemory(dev, readback.memory);
   int status = errors ? 1 : 0;
   if (errors)
      fprintf(stderr, "FAIL: %u/%u exact pixels differ\n", errors, WIDTH);
   else
      printf("PASS: %u exact float pixels: A2B10G10R10, B10G11R11, BC1, BC3, "
             "mips, base-mip view, cube, 3D\n", WIDTH);

   VK_OK(vkDeviceWaitIdle(dev));
   vkDestroyPipeline(dev, base_mip_pipeline, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyPipelineLayout(dev, playout, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, out_view, NULL);
   vkDestroyImage(dev, out_image, NULL);
   vkFreeMemory(dev, out_memory, NULL);
   vkDestroyBuffer(dev, readback.buffer, NULL);
   vkFreeMemory(dev, readback.memory, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dlayout, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyImageView(dev, mip_base, NULL);
   struct image *images[] = { &tex3, &cube, &mip, &bc3, &bc1, &uf, &a2 };
   for (unsigned i = 0; i < sizeof(images)/sizeof(images[0]); i++) {
      vkDestroyImageView(dev, images[i]->view, NULL);
      vkDestroyImage(dev, images[i]->image, NULL);
      vkFreeMemory(dev, images[i]->memory, NULL);
   }
   vkDestroyBuffer(dev, staging.buffer, NULL);
   vkFreeMemory(dev, staging.memory, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   return status;
}
