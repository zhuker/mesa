/*
 * Exact texture-cache operation coverage not exercised by
 * cpvk_texture_cache_formats.c:
 *
 *   row 0: B10G11R11_UFLOAT_PACK32 cube conversion on +X,-X,+Y,-Y,+Z,-Z;
 *   row 1: BC3 alpha0 <= alpha1 selectors 0..7 (endpoints, four
 *          interpolants, explicit zero, explicit one);
 *   row 2: implicit-derivative texture() selects mip 2;
 *   row 3: textureGrad() selects mip 1 and mip 3 from explicit gradients;
 *   row 4: texture() with runtime bias 1/1.25 selects mip 3 (txb/NVVM exp2 lowering).
 *
 * Every sampled image is optimal tiled and device-local.  Uploads transition
 * UNDEFINED -> TRANSFER_DST_OPTIMAL, copy, then transition with a
 * TRANSFER_WRITE -> SHADER_READ dependency to SHADER_READ_ONLY_OPTIMAL.  The
 * device-local float output uses a render-pass final transition/dependency to
 * TRANSFER_SRC_OPTIMAL before readback.  Embedded SPIR-V makes the test fully
 * standalone.
 *
 * The shader's 64-element indirectly indexed function-local table deliberately
 * survives as local memory.  That rejects cudapipe's strict HW_INLINE gate but
 * is resource-isolated and admitted as HW_FUSED.  The wrapper enables texture
 * cache statistics and requires every native cudapipe fragment launch to use
 * HW_FUSED.  Other Vulkan drivers ignore the two environment variables.
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -Werror \
 *      src/cudapipe/tests/cpvk_texture_cache_ops.c -lvulkan \
 *      -o /tmp/cpvk_texture_cache_ops
 * Run:
 *   /tmp/cpvk_texture_cache_ops
 *
 * Fragment SPIR-V was generated with glslangValidator 11:16.4.0 from:
 *
 * #version 450
 * layout(set=0,binding=0) uniform samplerCube cube_tex;
 * layout(set=0,binding=1) uniform sampler2D bc3_tex;
 * layout(set=0,binding=2) uniform sampler2D mip_tex;
 * layout(push_constant) uniform Push { int kind; } pc;
 * layout(location=0) out vec4 out_colour;
 * void main() {
 *   const int lane[64] = int[64](0,1,2,3,4,5,6,7 repeated eight times);
 *   int x=clamp(int(gl_FragCoord.x),0,63), j=lane[x];
 *   if (pc.kind == 0) {
 *      vec3 d = the_cube_axis_for(j % 6);
 *      out_colour = textureLod(cube_tex,d,0.0);
 *   } else if (pc.kind == 1) {
 *      vec2 uv=(vec2(j&3,j>>2)+vec2(0.5))/4.0;
 *      out_colour=textureLod(bc3_tex,uv,0.0);
 *   } else if (pc.kind == 2) {
 *      out_colour=texture(mip_tex,gl_FragCoord.xy*0.25);
 *   } else if (pc.kind == 3) {
 *      float g=(j&1)==0 ? 0.125 : 0.5;
 *      out_colour=textureGrad(mip_tex,vec2(0.5),vec2(g,0),vec2(0,g));
 *   } else {
 *      float bias=(j&1)==0 ? 1.0 : 1.25;
 *      out_colour=texture(mip_tex,gl_FragCoord.xy*0.25,bias);
 *   }
 * }
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

#define W 64u
#define H 5u
#define PIXELS (W * H)
#define DRAWS 5u

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
   0x0007000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x0000001au, 0x00030003u,
   0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00060005u, 0x0000000bu,
   0x505f6c67u, 0x65567265u, 0x78657472u, 0x00000000u, 0x00060006u, 0x0000000bu, 0x00000000u, 0x505f6c67u,
   0x7469736fu, 0x006e6f69u, 0x00070006u, 0x0000000bu, 0x00000001u, 0x505f6c67u, 0x746e696fu, 0x657a6953u,
   0x00000000u, 0x00070006u, 0x0000000bu, 0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u, 0x0065636eu,
   0x00070006u, 0x0000000bu, 0x00000003u, 0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u,
   0x0000000du, 0x00000000u, 0x00060005u, 0x0000001au, 0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u,
   0x00050005u, 0x0000001du, 0x65646e69u, 0x6c626178u, 0x00000065u, 0x00030047u, 0x0000000bu, 0x00000002u,
   0x00050048u, 0x0000000bu, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u,
   0x0000000bu, 0x00000001u, 0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u,
   0x0000000bu, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00040047u, 0x0000001au, 0x0000000bu, 0x0000002au,
   0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u,
   0x0004002bu, 0x00000008u, 0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au, 0x00000006u, 0x00000009u,
   0x0006001eu, 0x0000000bu, 0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu,
   0x00000003u, 0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u, 0x00040015u, 0x0000000eu,
   0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u, 0x00040017u, 0x00000010u,
   0x00000006u, 0x00000002u, 0x0004002bu, 0x00000008u, 0x00000011u, 0x00000003u, 0x0004001cu, 0x00000012u,
   0x00000010u, 0x00000011u, 0x0004002bu, 0x00000006u, 0x00000013u, 0xbf800000u, 0x0005002cu, 0x00000010u,
   0x00000014u, 0x00000013u, 0x00000013u, 0x0004002bu, 0x00000006u, 0x00000015u, 0x40400000u, 0x0005002cu,
   0x00000010u, 0x00000016u, 0x00000015u, 0x00000013u, 0x0005002cu, 0x00000010u, 0x00000017u, 0x00000013u,
   0x00000015u, 0x0006002cu, 0x00000012u, 0x00000018u, 0x00000014u, 0x00000016u, 0x00000017u, 0x00040020u,
   0x00000019u, 0x00000001u, 0x0000000eu, 0x0004003bu, 0x00000019u, 0x0000001au, 0x00000001u, 0x00040020u,
   0x0000001cu, 0x00000007u, 0x00000012u, 0x00040020u, 0x0000001eu, 0x00000007u, 0x00000010u, 0x0004002bu,
   0x00000006u, 0x00000021u, 0x00000000u, 0x0004002bu, 0x00000006u, 0x00000022u, 0x3f800000u, 0x00040020u,
   0x00000026u, 0x00000003u, 0x00000007u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,
   0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000001cu, 0x0000001du, 0x00000007u, 0x0004003du, 0x0000000eu,
   0x0000001bu, 0x0000001au, 0x0003003eu, 0x0000001du, 0x00000018u, 0x00050041u, 0x0000001eu, 0x0000001fu,
   0x0000001du, 0x0000001bu, 0x0004003du, 0x00000010u, 0x00000020u, 0x0000001fu, 0x00050051u, 0x00000006u,
   0x00000023u, 0x00000020u, 0x00000000u, 0x00050051u, 0x00000006u, 0x00000024u, 0x00000020u, 0x00000001u,
   0x00070050u, 0x00000007u, 0x00000025u, 0x00000023u, 0x00000024u, 0x00000021u, 0x00000022u, 0x00050041u,
   0x00000026u, 0x00000027u, 0x0000000du, 0x0000000fu, 0x0003003eu, 0x00000027u, 0x00000025u, 0x000100fdu,
   0x00010038u,
};

static const uint32_t frag_spv[] = {
   0x07230203, 0x00010000, 0x0008000b, 0x0000009f, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
   0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
   0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000c, 0x0000004d, 0x00030010,
   0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
   0x00000000, 0x00030005, 0x00000008, 0x00000078, 0x00060005, 0x0000000c, 0x465f6c67, 0x43676172,
   0x64726f6f, 0x00000000, 0x00030005, 0x00000016, 0x0000006a, 0x00050005, 0x00000023, 0x65646e69,
   0x6c626178, 0x00000065, 0x00040005, 0x00000026, 0x68737550, 0x00000000, 0x00050006, 0x00000026,
   0x00000000, 0x646e696b, 0x00000000, 0x00030005, 0x00000028, 0x00006370, 0x00030005, 0x0000003b,
   0x00000064, 0x00050005, 0x0000004d, 0x5f74756f, 0x6f6c6f63, 0x00007275, 0x00050005, 0x00000051,
   0x65627563, 0x7865745f, 0x00000000, 0x00030005, 0x0000005d, 0x00007675, 0x00040005, 0x0000006e,
   0x5f336362, 0x00786574, 0x00040005, 0x00000078, 0x5f70696d, 0x00786574, 0x00030005, 0x00000086,
   0x00000067, 0x00040005, 0x00000093, 0x73616962, 0x00000000, 0x00040047, 0x0000000c, 0x0000000b,
   0x0000000f, 0x00030047, 0x00000026, 0x00000002, 0x00050048, 0x00000026, 0x00000000, 0x00000023,
   0x00000000, 0x00040047, 0x0000004d, 0x0000001e, 0x00000000, 0x00040047, 0x00000051, 0x00000021,
   0x00000000, 0x00040047, 0x00000051, 0x00000022, 0x00000000, 0x00040047, 0x0000006e, 0x00000021,
   0x00000001, 0x00040047, 0x0000006e, 0x00000022, 0x00000000, 0x00040047, 0x00000078, 0x00000021,
   0x00000002, 0x00040047, 0x00000078, 0x00000022, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
   0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000001, 0x00040020, 0x00000007,
   0x00000007, 0x00000006, 0x00030016, 0x00000009, 0x00000020, 0x00040017, 0x0000000a, 0x00000009,
   0x00000004, 0x00040020, 0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c,
   0x00000001, 0x00040015, 0x0000000d, 0x00000020, 0x00000000, 0x0004002b, 0x0000000d, 0x0000000e,
   0x00000000, 0x00040020, 0x0000000f, 0x00000001, 0x00000009, 0x0004002b, 0x00000006, 0x00000013,
   0x00000000, 0x0004002b, 0x00000006, 0x00000014, 0x0000003f, 0x0004002b, 0x0000000d, 0x00000017,
   0x00000040, 0x0004001c, 0x00000018, 0x00000006, 0x00000017, 0x0004002b, 0x00000006, 0x00000019,
   0x00000001, 0x0004002b, 0x00000006, 0x0000001a, 0x00000002, 0x0004002b, 0x00000006, 0x0000001b,
   0x00000003, 0x0004002b, 0x00000006, 0x0000001c, 0x00000004, 0x0004002b, 0x00000006, 0x0000001d,
   0x00000005, 0x0004002b, 0x00000006, 0x0000001e, 0x00000006, 0x0004002b, 0x00000006, 0x0000001f,
   0x00000007, 0x0043002c, 0x00000018, 0x00000020, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00000013, 0x00000019, 0x0000001a, 0x0000001b,
   0x0000001c, 0x0000001d, 0x0000001e, 0x0000001f, 0x00040020, 0x00000022, 0x00000007, 0x00000018,
   0x0003001e, 0x00000026, 0x00000006, 0x00040020, 0x00000027, 0x00000009, 0x00000026, 0x0004003b,
   0x00000027, 0x00000028, 0x00000009, 0x00040020, 0x00000029, 0x00000009, 0x00000006, 0x00020014,
   0x0000002c, 0x00040017, 0x00000039, 0x00000009, 0x00000003, 0x00040020, 0x0000003a, 0x00000007,
   0x00000039, 0x0004002b, 0x00000009, 0x0000003c, 0x3f800000, 0x0004002b, 0x00000009, 0x0000003d,
   0x00000000, 0x0006002c, 0x00000039, 0x0000003e, 0x0000003c, 0x0000003d, 0x0000003d, 0x0004002b,
   0x00000009, 0x00000040, 0xbf800000, 0x0006002c, 0x00000039, 0x00000041, 0x00000040, 0x0000003d,
   0x0000003d, 0x0006002c, 0x00000039, 0x00000043, 0x0000003d, 0x0000003c, 0x0000003d, 0x0006002c,
   0x00000039, 0x00000045, 0x0000003d, 0x00000040, 0x0000003d, 0x0006002c, 0x00000039, 0x00000047,
   0x0000003d, 0x0000003d, 0x0000003c, 0x0006002c, 0x00000039, 0x00000049, 0x0000003d, 0x0000003d,
   0x00000040, 0x00040020, 0x0000004c, 0x00000003, 0x0000000a, 0x0004003b, 0x0000004c, 0x0000004d,
   0x00000003, 0x00090019, 0x0000004e, 0x00000009, 0x00000003, 0x00000000, 0x00000000, 0x00000000,
   0x00000001, 0x00000000, 0x0003001b, 0x0000004f, 0x0000004e, 0x00040020, 0x00000050, 0x00000000,
   0x0000004f, 0x0004003b, 0x00000050, 0x00000051, 0x00000000, 0x00040017, 0x0000005b, 0x00000009,
   0x00000002, 0x00040020, 0x0000005c, 0x00000007, 0x0000005b, 0x0004002b, 0x00000009, 0x00000065,
   0x3f000000, 0x0005002c, 0x0000005b, 0x00000066, 0x00000065, 0x00000065, 0x0004002b, 0x00000009,
   0x00000068, 0x40800000, 0x00090019, 0x0000006b, 0x00000009, 0x00000001, 0x00000000, 0x00000000,
   0x00000000, 0x00000001, 0x00000000, 0x0003001b, 0x0000006c, 0x0000006b, 0x00040020, 0x0000006d,
   0x00000000, 0x0000006c, 0x0004003b, 0x0000006d, 0x0000006e, 0x00000000, 0x0004003b, 0x0000006d,
   0x00000078, 0x00000000, 0x0004002b, 0x00000009, 0x0000007c, 0x3e800000, 0x00040020, 0x00000085,
   0x00000007, 0x00000009, 0x0004002b, 0x00000009, 0x0000008a, 0x3e000000, 0x0004002b, 0x00000009,
   0x00000097, 0x3fa00000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
   0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x00000007, 0x00000016,
   0x00000007, 0x0004003b, 0x00000022, 0x00000023, 0x00000007, 0x0004003b, 0x0000003a, 0x0000003b,
   0x00000007, 0x0004003b, 0x0000005c, 0x0000005d, 0x00000007, 0x0004003b, 0x00000085, 0x00000086,
   0x00000007, 0x0004003b, 0x00000085, 0x00000093, 0x00000007, 0x00050041, 0x0000000f, 0x00000010,
   0x0000000c, 0x0000000e, 0x0004003d, 0x00000009, 0x00000011, 0x00000010, 0x0004006e, 0x00000006,
   0x00000012, 0x00000011, 0x0008000c, 0x00000006, 0x00000015, 0x00000001, 0x0000002d, 0x00000012,
   0x00000013, 0x00000014, 0x0003003e, 0x00000008, 0x00000015, 0x0004003d, 0x00000006, 0x00000021,
   0x00000008, 0x0003003e, 0x00000023, 0x00000020, 0x00050041, 0x00000007, 0x00000024, 0x00000023,
   0x00000021, 0x0004003d, 0x00000006, 0x00000025, 0x00000024, 0x0003003e, 0x00000016, 0x00000025,
   0x00050041, 0x00000029, 0x0000002a, 0x00000028, 0x00000013, 0x0004003d, 0x00000006, 0x0000002b,
   0x0000002a, 0x000500aa, 0x0000002c, 0x0000002d, 0x0000002b, 0x00000013, 0x000300f7, 0x0000002f,
   0x00000000, 0x000400fa, 0x0000002d, 0x0000002e, 0x00000055, 0x000200f8, 0x0000002e, 0x0004003d,
   0x00000006, 0x00000030, 0x00000016, 0x0005008b, 0x00000006, 0x00000031, 0x00000030, 0x0000001e,
   0x000300f7, 0x00000038, 0x00000000, 0x000d00fb, 0x00000031, 0x00000037, 0x00000000, 0x00000032,
   0x00000001, 0x00000033, 0x00000002, 0x00000034, 0x00000003, 0x00000035, 0x00000004, 0x00000036,
   0x000200f8, 0x00000037, 0x0003003e, 0x0000003b, 0x00000049, 0x000200f9, 0x00000038, 0x000200f8,
   0x00000032, 0x0003003e, 0x0000003b, 0x0000003e, 0x000200f9, 0x00000038, 0x000200f8, 0x00000033,
   0x0003003e, 0x0000003b, 0x00000041, 0x000200f9, 0x00000038, 0x000200f8, 0x00000034, 0x0003003e,
   0x0000003b, 0x00000043, 0x000200f9, 0x00000038, 0x000200f8, 0x00000035, 0x0003003e, 0x0000003b,
   0x00000045, 0x000200f9, 0x00000038, 0x000200f8, 0x00000036, 0x0003003e, 0x0000003b, 0x00000047,
   0x000200f9, 0x00000038, 0x000200f8, 0x00000038, 0x0004003d, 0x0000004f, 0x00000052, 0x00000051,
   0x0004003d, 0x00000039, 0x00000053, 0x0000003b, 0x00070058, 0x0000000a, 0x00000054, 0x00000052,
   0x00000053, 0x00000002, 0x0000003d, 0x0003003e, 0x0000004d, 0x00000054, 0x000200f9, 0x0000002f,
   0x000200f8, 0x00000055, 0x00050041, 0x00000029, 0x00000056, 0x00000028, 0x00000013, 0x0004003d,
   0x00000006, 0x00000057, 0x00000056, 0x000500aa, 0x0000002c, 0x00000058, 0x00000057, 0x00000019,
   0x000300f7, 0x0000005a, 0x00000000, 0x000400fa, 0x00000058, 0x00000059, 0x00000072, 0x000200f8,
   0x00000059, 0x0004003d, 0x00000006, 0x0000005e, 0x00000016, 0x000500c7, 0x00000006, 0x0000005f,
   0x0000005e, 0x0000001b, 0x0004006f, 0x00000009, 0x00000060, 0x0000005f, 0x0004003d, 0x00000006,
   0x00000061, 0x00000016, 0x000500c3, 0x00000006, 0x00000062, 0x00000061, 0x0000001a, 0x0004006f,
   0x00000009, 0x00000063, 0x00000062, 0x00050050, 0x0000005b, 0x00000064, 0x00000060, 0x00000063,
   0x00050081, 0x0000005b, 0x00000067, 0x00000064, 0x00000066, 0x00050050, 0x0000005b, 0x00000069,
   0x00000068, 0x00000068, 0x00050088, 0x0000005b, 0x0000006a, 0x00000067, 0x00000069, 0x0003003e,
   0x0000005d, 0x0000006a, 0x0004003d, 0x0000006c, 0x0000006f, 0x0000006e, 0x0004003d, 0x0000005b,
   0x00000070, 0x0000005d, 0x00070058, 0x0000000a, 0x00000071, 0x0000006f, 0x00000070, 0x00000002,
   0x0000003d, 0x0003003e, 0x0000004d, 0x00000071, 0x000200f9, 0x0000005a, 0x000200f8, 0x00000072,
   0x00050041, 0x00000029, 0x00000073, 0x00000028, 0x00000013, 0x0004003d, 0x00000006, 0x00000074,
   0x00000073, 0x000500aa, 0x0000002c, 0x00000075, 0x00000074, 0x0000001a, 0x000300f7, 0x00000077,
   0x00000000, 0x000400fa, 0x00000075, 0x00000076, 0x0000007f, 0x000200f8, 0x00000076, 0x0004003d,
   0x0000006c, 0x00000079, 0x00000078, 0x0004003d, 0x0000000a, 0x0000007a, 0x0000000c, 0x0007004f,
   0x0000005b, 0x0000007b, 0x0000007a, 0x0000007a, 0x00000000, 0x00000001, 0x0005008e, 0x0000005b,
   0x0000007d, 0x0000007b, 0x0000007c, 0x00050057, 0x0000000a, 0x0000007e, 0x00000079, 0x0000007d,
   0x0003003e, 0x0000004d, 0x0000007e, 0x000200f9, 0x00000077, 0x000200f8, 0x0000007f, 0x00050041,
   0x00000029, 0x00000080, 0x00000028, 0x00000013, 0x0004003d, 0x00000006, 0x00000081, 0x00000080,
   0x000500aa, 0x0000002c, 0x00000082, 0x00000081, 0x0000001b, 0x000300f7, 0x00000084, 0x00000000,
   0x000400fa, 0x00000082, 0x00000083, 0x00000092, 0x000200f8, 0x00000083, 0x0004003d, 0x00000006,
   0x00000087, 0x00000016, 0x000500c7, 0x00000006, 0x00000088, 0x00000087, 0x00000019, 0x000500aa,
   0x0000002c, 0x00000089, 0x00000088, 0x00000013, 0x000600a9, 0x00000009, 0x0000008b, 0x00000089,
   0x0000008a, 0x00000065, 0x0003003e, 0x00000086, 0x0000008b, 0x0004003d, 0x0000006c, 0x0000008c,
   0x00000078, 0x0004003d, 0x00000009, 0x0000008d, 0x00000086, 0x00050050, 0x0000005b, 0x0000008e,
   0x0000008d, 0x0000003d, 0x0004003d, 0x00000009, 0x0000008f, 0x00000086, 0x00050050, 0x0000005b,
   0x00000090, 0x0000003d, 0x0000008f, 0x00080058, 0x0000000a, 0x00000091, 0x0000008c, 0x00000066,
   0x00000004, 0x0000008e, 0x00000090, 0x0003003e, 0x0000004d, 0x00000091, 0x000200f9, 0x00000084,
   0x000200f8, 0x00000092, 0x0004003d, 0x00000006, 0x00000094, 0x00000016, 0x000500c7, 0x00000006,
   0x00000095, 0x00000094, 0x00000019, 0x000500aa, 0x0000002c, 0x00000096, 0x00000095, 0x00000013,
   0x000600a9, 0x00000009, 0x00000098, 0x00000096, 0x0000003c, 0x00000097, 0x0003003e, 0x00000093,
   0x00000098, 0x0004003d, 0x0000006c, 0x00000099, 0x00000078, 0x0004003d, 0x0000000a, 0x0000009a,
   0x0000000c, 0x0007004f, 0x0000005b, 0x0000009b, 0x0000009a, 0x0000009a, 0x00000000, 0x00000001,
   0x0005008e, 0x0000005b, 0x0000009c, 0x0000009b, 0x0000007c, 0x0004003d, 0x00000009, 0x0000009d,
   0x00000093, 0x00070057, 0x0000000a, 0x0000009e, 0x00000099, 0x0000009c, 0x00000001, 0x0000009d,
   0x0003003e, 0x0000004d, 0x0000009e, 0x000200f9, 0x00000084, 0x000200f8, 0x00000084, 0x000200f9,
   0x00000077, 0x000200f8, 0x00000077, 0x000200f9, 0x0000005a, 0x000200f8, 0x0000005a, 0x000200f9,
   0x0000002f, 0x000200f8, 0x0000002f, 0x000100fd, 0x00010038,
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
      .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
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
          uint32_t levels, uint32_t layers)
{
   VkImageViewCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image, .viewType = type, .format = format,
      .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                      VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers } };
   VkImageView view;
   VK_OK(vkCreateImageView(dev, &ci, NULL, &view));
   return view;
}

static struct image
make_sampled_image(VkDevice dev, VkPhysicalDevice pdev, VkFormat format,
                   VkExtent3D extent, uint32_t mips, uint32_t layers,
                   VkImageViewType view_type, VkImageCreateFlags flags)
{
   struct image im = { .format = format, .mips = mips, .layers = layers };
   VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags, .imageType = VK_IMAGE_TYPE_2D, .format = format,
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
   im.view = make_view(dev, im.image, format, view_type, mips, layers);
   return im;
}

static void
upload(VkDevice dev, VkQueue queue, VkCommandBuffer cmd, struct buffer staging,
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
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                        0, NULL, 0, NULL, 1, &after);
   VK_OK(vkEndCommandBuffer(cmd));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd };
   VK_OK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   VK_OK(vkQueueWaitIdle(queue));
}

static VkShaderModule
make_shader(VkDevice dev, const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code };
   VkShaderModule module;
   VK_OK(vkCreateShaderModule(dev, &ci, NULL, &module));
   return module;
}

static void
destroy_image(VkDevice dev, struct image *im)
{
   vkDestroyImageView(dev, im->view, NULL);
   vkDestroyImage(dev, im->image, NULL);
   vkFreeMemory(dev, im->memory, NULL);
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
   if (hits != launches || hits < DRAWS || shader_fb || descriptor_fb ||
       inline_hits || fused_hits != hits) {
      fprintf(stderr, "FAIL native run: expected every launch in HW_FUSED; "
              "hits=%llu/%llu shader=%llu descriptor=%llu inline=%llu fused=%llu\n",
              hits, launches, shader_fb, descriptor_fb, inline_hits, fused_hits);
      return 1;
   }
   printf("PASS native cache used HW_FUSED only (%llu/%llu launches)\n",
          hits, launches);
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
   if (!log) {
      close(fds[0]);
      return 1;
   }
   for (;;) {
      if (used + 2048 + 1 > cap) {
         cap *= 2;
         char *grown = realloc(log, cap);
         if (!grown) {
            close(fds[0]);
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
run_child(void)
{

   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "cpvk_texture_cache_ops", .apiVersion = VK_API_VERSION_1_0 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app };
   VkInstance instance;
   VK_OK(vkCreateInstance(&ici, NULL, &instance));

   uint32_t ndev = 0;
   VK_OK(vkEnumeratePhysicalDevices(instance, &ndev, NULL));
   if (!ndev) {
      fprintf(stderr, "no Vulkan physical device\n");
      return 2;
   }
   VkPhysicalDevice *pdevs = calloc(ndev, sizeof(*pdevs));
   if (!pdevs)
      return 2;
   VK_OK(vkEnumeratePhysicalDevices(instance, &ndev, pdevs));
   VkPhysicalDevice pdev = pdevs[0];
   free(pdevs);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   fprintf(stderr, "cpvk-test-device: %s\n", props.deviceName);
   printf("device: %s\n", props.deviceName);

   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, NULL);
   VkQueueFamilyProperties *qprops = calloc(nq, sizeof(*qprops));
   if (!qprops)
      return 2;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qprops);
   uint32_t qfamily = UINT32_MAX;
   for (uint32_t i = 0; i < nq; i++)
      if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         qfamily = i;
         break;
      }
   free(qprops);
   if (qfamily == UINT32_MAX) {
      fprintf(stderr, "no graphics queue\n");
      return 2;
   }

   const VkFormat sampled_formats[] = {
      VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_BC3_UNORM_BLOCK,
      VK_FORMAT_R8G8B8A8_UNORM
   };
   for (unsigned i = 0; i < sizeof(sampled_formats) / sizeof(sampled_formats[0]); i++) {
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pdev, sampled_formats[i], &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
         fprintf(stderr, "required sampled optimal format unsupported: %u\n",
                 sampled_formats[i]);
         return 2;
      }
   }
   VkFormatProperties out_fp;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_R32G32B32A32_SFLOAT,
                                       &out_fp);
   if (!(out_fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
      fprintf(stderr, "R32G32B32A32_SFLOAT color attachment unsupported\n");
      return 2;
   }

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
   VkCommandPool command_pool;
   VK_OK(vkCreateCommandPool(dev, &cpci, NULL, &command_pool));
   VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   VK_OK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   struct buffer staging = make_buffer(dev, pdev, 4096,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   struct image cube = make_sampled_image(dev, pdev,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32, (VkExtent3D){1, 1, 1}, 1, 6,
      VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
   /* Vulkan cube layer order: +X, -X, +Y, -Y, +Z, -Z.  Each word is an exact
    * unsigned-float tuple: (1,0,0), (0,1,0), (0,0,1), (0.5,2,4),
    * (8,0.25,0.125), (16,32,64). */
   static const uint32_t cube_data[6] = {
      0x000003c0u, 0x001e0000u, 0x78000000u,
      0x88200380u, 0x601a0480u, 0xa82804c0u
   };
   VkBufferImageCopy cube_region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6 },
      .imageExtent = {1, 1, 1}
   };
   upload(dev, queue, cmd, staging, &cube, cube_data, sizeof(cube_data),
          &cube_region, 1);

   struct image bc3 = make_sampled_image(dev, pdev, VK_FORMAT_BC3_UNORM_BLOCK,
      (VkExtent3D){4, 4, 1}, 1, 1, VK_IMAGE_VIEW_TYPE_2D, 0);
   /* alpha0=5 <= alpha1=10.  Selectors 0..7 appear in order twice.  The
    * close endpoints make both Vulkan NVIDIA and native CUDA texture decoding
    * produce exact bytes 5,10,6,7,8,9,0,255 despite BC decoder precision
    * latitude, while all endpoint/interpolant/explicit cases remain distinct. */
   static const uint8_t bc3_data[16] = {
      5, 10, 0x88, 0xc6, 0xfa, 0x88, 0xc6, 0xfa,
      0, 0, 0, 0, 0, 0, 0, 0
   };
   VkBufferImageCopy bc3_region = {
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = {4, 4, 1}
   };
   upload(dev, queue, cmd, staging, &bc3, bc3_data, sizeof(bc3_data),
          &bc3_region, 1);

   struct image mip = make_sampled_image(dev, pdev, VK_FORMAT_R8G8B8A8_UNORM,
      (VkExtent3D){16, 16, 1}, 5, 1, VK_IMAGE_VIEW_TYPE_2D, 0);
   uint8_t mip_data[1364];
   static const uint32_t mip_offsets[5] = {0, 1024, 1280, 1344, 1360};
   static const uint32_t mip_texels[5] = {256, 64, 16, 4, 1};
   static const uint8_t mip_colours[5][4] = {
      {255,0,0,255}, {0,255,0,255}, {0,0,255,255},
      {255,255,0,255}, {255,0,255,255}
   };
   VkBufferImageCopy mip_regions[5];
   uint32_t mw = 16;
   for (uint32_t level = 0; level < 5; level++) {
      for (uint32_t i = 0; i < mip_texels[level]; i++)
         memcpy(mip_data + mip_offsets[level] + i * 4, mip_colours[level], 4);
      mip_regions[level] = (VkBufferImageCopy) {
         .bufferOffset = mip_offsets[level],
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
         .imageExtent = {mw, mw, 1}
      };
      mw >>= 1;
   }
   upload(dev, queue, cmd, staging, &mip, mip_data, sizeof(mip_data),
          mip_regions, 5);

   VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .minLod = 0.0f, .maxLod = 4.0f,
      .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK };
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
   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps };
   VkDescriptorPool dpool;
   VK_OK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dlayout };
   VkDescriptorSet set;
   VK_OK(vkAllocateDescriptorSets(dev, &dsai, &set));
   VkDescriptorImageInfo infos[3] = {
      { sampler, cube.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { sampler, bc3.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
      { sampler, mip.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }
   };
   VkWriteDescriptorSet writes[3];
   memset(writes, 0, sizeof(writes));
   for (uint32_t i = 0; i < 3; i++)
      writes[i] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
         .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &infos[i]
      };
   vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);

   VkImage out_image;
   VkDeviceMemory out_memory;
   VkImageCreateInfo oici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VK_OK(vkCreateImage(dev, &oici, NULL, &out_image));
   VkMemoryRequirements oreq;
   vkGetImageMemoryRequirements(dev, out_image, &oreq);
   VkMemoryAllocateInfo oai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = oreq.size,
      .memoryTypeIndex = pick_memory(pdev, oreq.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VK_OK(vkAllocateMemory(dev, &oai, NULL, &out_memory));
   VK_OK(vkBindImageMemory(dev, out_image, out_memory, 0));
   VkImageView out_view = make_view(dev, out_image, oici.format,
                                    VK_IMAGE_VIEW_TYPE_2D, 1, 1);
   struct buffer readback = make_buffer(dev, pdev, PIXELS * 16,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   VkAttachmentDescription ad = { .format = oici.format,
      .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
   VkAttachmentReference ar = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1, .pColorAttachments = &ar };
   VkSubpassDependency deps[2] = {
      { VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0 },
      { 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT, 0 }
   };
   VkRenderPassCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ad, .subpassCount = 1,
      .pSubpasses = &sub, .dependencyCount = 2, .pDependencies = deps };
   VkRenderPass render_pass;
   VK_OK(vkCreateRenderPass(dev, &rpci, NULL, &render_pass));
   VkFramebufferCreateInfo fbci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass, .attachmentCount = 1, .pAttachments = &out_view,
      .width = W, .height = H, .layers = 1 };
   VkFramebuffer framebuffer;
   VK_OK(vkCreateFramebuffer(dev, &fbci, NULL, &framebuffer));

   VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dlayout,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
   VkPipelineLayout pipeline_layout;
   VK_OK(vkCreatePipelineLayout(dev, &plci, NULL, &pipeline_layout));
   VkShaderModule vs = make_shader(dev, vert_spv, sizeof(vert_spv));
   VkShaderModule fs = make_shader(dev, frag_spv, sizeof(frag_spv));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" }
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1 };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkDynamicState dynamics[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
   VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dynamics };
   VkGraphicsPipelineCreateInfo gpci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .pDynamicState = &dyn,
      .layout = pipeline_layout, .renderPass = render_pass, .subpass = 0 };
   VkPipeline pipeline;
   VK_OK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipeline));

   VK_OK(vkResetCommandBuffer(cmd, 0));
   VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   VK_OK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier out_before = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = out_image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &out_before);
   VkClearValue clear = { .color.float32 = {0, 0, 0, 0} };
   VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass, .framebuffer = framebuffer,
      .renderArea = {{0, 0}, {W, H}}, .clearValueCount = 1,
      .pClearValues = &clear };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                           0, 1, &set, 0, NULL);
   VkViewport viewport = {0, 0, W, H, 0, 1};
   vkCmdSetViewport(cmd, 0, 1, &viewport);
   for (uint32_t kind = 0; kind < DRAWS; kind++) {
      VkRect2D scissor = {{0, (int32_t)kind}, {W, 1}};
      vkCmdSetScissor(cmd, 0, 1, &scissor);
      vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, 4, &kind);
      vkCmdDraw(cmd, 3, 1, 0, 0);
   }
   vkCmdEndRenderPass(cmd);
   VkBufferImageCopy out_copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {W, H, 1}
   };
   vkCmdCopyImageToBuffer(cmd, out_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback.buffer, 1, &out_copy);
   VkBufferMemoryBarrier host = { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback.buffer, .offset = 0, .size = VK_WHOLE_SIZE };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0,
                        0, NULL, 1, &host, 0, NULL);
   VK_OK(vkEndCommandBuffer(cmd));
   VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cmd };
   VK_OK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_OK(vkQueueWaitIdle(queue));

   static const uint32_t cube_expect[6][4] = {
      {0x3f800000,0x00000000,0x00000000,0x3f800000},
      {0x00000000,0x3f800000,0x00000000,0x3f800000},
      {0x00000000,0x00000000,0x3f800000,0x3f800000},
      {0x3f000000,0x40000000,0x40800000,0x3f800000},
      {0x41000000,0x3e800000,0x3e000000,0x3f800000},
      {0x41800000,0x42000000,0x42800000,0x3f800000}
   };
   static const uint32_t bc3_alpha_expect[8] = {
      0x3ca0a0a1, 0x3d20a0a1, 0x3cc0c0c1, 0x3ce0e0e1,
      0x3d008081, 0x3d109091, 0x00000000, 0x3f800000
   };
   static const uint32_t mip2_expect[4] = {
      0x00000000,0x00000000,0x3f800000,0x3f800000
   };
   static const uint32_t mip1_expect[4] = {
      0x00000000,0x3f800000,0x00000000,0x3f800000
   };
   static const uint32_t mip3_expect[4] = {
      0x3f800000,0x3f800000,0x00000000,0x3f800000
   };
   uint32_t (*got)[4];
   VK_OK(vkMapMemory(dev, readback.memory, 0, VK_WHOLE_SIZE, 0, (void **)&got));
   unsigned errors = 0;
   for (uint32_t y = 0; y < H; y++) {
      for (uint32_t x = 0; x < W; x++) {
         uint32_t j = x & 7;
         const uint32_t *expect;
         uint32_t bc3_expect[4] = {0, 0, 0, bc3_alpha_expect[j]};
         if (y == 0)
            expect = cube_expect[j % 6];
         else if (y == 1)
            expect = bc3_expect;
         else if (y == 2)
            expect = mip2_expect;
         else if (y == 3)
            expect = (j & 1) ? mip3_expect : mip1_expect;
         else
            expect = mip3_expect;
         if (memcmp(got[y * W + x], expect, 16)) {
            fprintf(stderr, "row %u pixel %u: got %08x %08x %08x %08x; "
                    "expected %08x %08x %08x %08x\n", y, x,
                    got[y * W + x][0], got[y * W + x][1],
                    got[y * W + x][2], got[y * W + x][3],
                    expect[0], expect[1], expect[2], expect[3]);
            errors++;
         }
      }
   }
   vkUnmapMemory(dev, readback.memory);

   int fail = errors != 0;
   if (fail) {
      fprintf(stderr, "FAIL: %u/%u exact float pixels differ\n", errors, PIXELS);
   } else {
      printf("PASS cube R11G11B10 exact faces: +X 16, -X 16, +Y 8, -Y 8, +Z 8, -Z 8\n");
      printf("PASS BC3 alpha0<=alpha1 selectors exact (8 each): "
             "0=3ca0a0a1 1=3d20a0a1 2=3cc0c0c1 3=3ce0e0e1 "
             "4=3d008081 5=3d109091 6=00000000 7=3f800000\n");
      printf("PASS implicit texture() derivative (rho=4) selected mip2: 64/64 exact\n");
      printf("PASS textureGrad() gradients selected mip1 32/32 and mip3 32/32 exact\n");
      printf("PASS runtime texture bias 1/1.25 selected mip3 through txb: 64/64 exact\n");
      printf("PASS: %u exact float pixels across texture-cache operations\n", PIXELS);
   }

   VK_OK(vkDeviceWaitIdle(dev));
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyPipelineLayout(dev, pipeline_layout, NULL);
   vkDestroyFramebuffer(dev, framebuffer, NULL);
   vkDestroyRenderPass(dev, render_pass, NULL);
   vkDestroyImageView(dev, out_view, NULL);
   vkDestroyImage(dev, out_image, NULL);
   vkFreeMemory(dev, out_memory, NULL);
   vkDestroyBuffer(dev, readback.buffer, NULL);
   vkFreeMemory(dev, readback.memory, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dlayout, NULL);
   vkDestroySampler(dev, sampler, NULL);
   destroy_image(dev, &mip);
   destroy_image(dev, &bc3);
   destroy_image(dev, &cube);
   vkDestroyBuffer(dev, staging.buffer, NULL);
   vkFreeMemory(dev, staging.memory, NULL);
   vkDestroyCommandPool(dev, command_pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
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
