/*
 * textureGrad(): the level of detail a shader names as a footprint.
 *
 * textureLod() names a mip level outright, and cpvk_lod covers it. The other
 * explicit form hands over the derivatives of the coordinate instead and asks
 * the implementation to work the level out:
 *
 *    lambda = log2(max(|dPdx * size|, |dPdy * size|))
 *
 * Every normal-mapped surface shader in the Roblox capture samples that way,
 * and cudavk did not implement it: `nir_texop_txd` was not in emit_tex()'s
 * supported set in `cp_nir_to_llvm.c`, so every textureGrad() in every shader
 * returned the placeholder constant (0, 0, 0, 1) -- silently, with no warning
 * and no counter. Decoded as a tangent-space normal that constant is
 * (1, -1, 0), whose z is sqrt(clamp(1 - 2, 0, 1)) = 0, and normalize() of the
 * zero vector is NaN, which reaches the tonemap and clamps to black. The
 * capture's park bench was painted (0, 0, 0) where NVIDIA paints (23, 39, 49)
 * (`.audit/favorite2_shading.md`).
 *
 * The texture is 16x16 with five mip levels and each level is one flat
 * colour, so a sample names the level it came from and a wrong level is a
 * wrong colour rather than a subtle blend:
 *
 *    level 0  16x16  200  20  20      level 3  2x2  230 230  40
 *    level 1   8x8    20 200  20      level 4  1x1  240 240 240
 *    level 2   4x4    20  20 200
 *
 * With a gradient of (g, 0) in x and (0, g) in y the footprint is square and
 * lambda is exactly log2(16g), so the six gradient passes are
 *
 *    g = 1/16 -> lambda 0    g = 4/16 -> lambda 2    g = 16/16 -> lambda 4
 *    g = 2/16 -> lambda 1    g = 8/16 -> lambda 3
 *
 * and a sixth, ddx = (4/16, 0) with ddy = (0, 1/16), whose two axes disagree.
 * That one is the rule itself: lambda is the *larger* of the two lengths, so
 * the answer is level 2 and not level 1, level 0 or a blend of them. Every
 * lambda is a whole number and the mip mode is NEAREST, so each answer sits
 * half a level away from the nearest rounding boundary -- no implementation's
 * log2 can move it, and nothing here depends on a tolerance.
 *
 * The sampler is NEAREST in every axis and anisotropy is off, so the levels
 * are flat colours all the way through and no filter can blend two of them.
 * That matters for what this test does *not* claim: cudavk serves
 * textureGrad() by collapsing the footprint to a scalar level, which loses
 * the footprint's shape, so an anisotropic gradient would be filtered
 * differently from the hardware. Both gradients here are isotropic or
 * compared on their maximum, which is the part every implementation agrees
 * on.
 *
 * The first pass is the control: textureLod(tex, uv, 2.0), the same texture,
 * the same sampler, the same descriptor set, the same pipeline layout, and an
 * op the driver has always served. If the control paints level 2's colour
 * then the mip chain reached the device, the descriptor was bound and the
 * triangle covered the viewport, so anything wrong in the six gradient passes
 * is textureGrad() and nothing else.
 *
 * The test names the two specific failures it was written against. If a
 * gradient pass returns (0, 0, 0, 255) it says so: that is the
 * unsupported-texture-op constant, and the op was never emitted. If a
 * gradient pass returns level 0's colour for every gradient it says that too:
 * that is a driver which serves textureGrad() by ignoring the gradients, and
 * it is what a correct emitter still produces on this driver wherever the
 * *shader* computed its gradients with dFdx/dFdy, because those return zero
 * (`cp_nir_to_llvm.c`, nir_intrinsic_ddx). Constant gradients are used here
 * precisely so that this test measures the texture op and not the derivative.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, and the mip chain is uploaded from a staging buffer, so this runs
 * unchanged on NVIDIA, on lavapipe and on the native driver. SPIR-V is
 * embedded; nothing is read from disk.
 *
 *   cc -std=c11 -Wall -Wextra -Werror src/cudavk/tests/cpvk_texgrad.c \
 *      -o /tmp/cpvk-tests/cpvk_texgrad -lvulkan
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- g.vert --
 *   #version 450
 *   const vec2 base[3] = vec2[3](vec2(-1.0, -1.0),
 *                                vec2( 3.0, -1.0),
 *                                vec2(-1.0,  3.0));
 *   void main()
 *   {
 *      gl_Position = vec4(base[gl_VertexIndex], 0.0, 1.0);
 *   }
 *
 *   -- ctl.frag --  the control
 *   #version 450
 *   layout(set = 0, binding = 0) uniform sampler2D tex;
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      out_colour = textureLod(tex, vec2(0.5, 0.5), 2.0);
 *   }
 *
 *   -- g0.frag .. g4.frag --  one per lambda, GX = GY = 0.0625, 0.125,
 *                             0.25, 0.5, 1.0
 *   #version 450
 *   layout(set = 0, binding = 0) uniform sampler2D tex;
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      out_colour = textureGrad(tex, vec2(0.5, 0.5),
 *                               vec2(GX, 0.0), vec2(0.0, GY));
 *   }
 *
 *   -- gm.frag --  the two axes disagree; the larger one wins
 *   #version 450
 *   layout(set = 0, binding = 0) uniform sampler2D tex;
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      out_colour = textureGrad(tex, vec2(0.5, 0.5),
 *                               vec2(0.25, 0.0), vec2(0.0, 0.0625));
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

#define TEX_DIM 16         /* level 0 is TEX_DIM x TEX_DIM */
#define LEVELS  5          /* 16, 8, 4, 2, 1 */

#define NPASS 7            /* the control, then six gradients */

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* One flat colour per mip level; no two are within 20 of each other on any
 * channel, so a blend of two levels is not a third level's colour. */
static const unsigned char level_rgba[LEVELS][4] = {
   { 200,  20,  20, 255 },
   {  20, 200,  20, 255 },
   {  20,  20, 200, 255 },
   { 230, 230,  40, 255 },
   { 240, 240, 240, 255 },
};

/* What emit_tex()'s unsupported branch returns, in 8-bit UNORM. */
static const unsigned char unsupported_rgba[4] = { 0, 0, 0, 255 };


/* g.vert */
static const uint32_t vert_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000028u, 0x00000000u, 0x00020011u,
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

/* ctl.frag -- textureLod(tex, uv, 2.0), the control */
static const uint32_t frag_lod_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000014u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x40000000u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,
   0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu, 0x0000000eu, 0x0000000du,
   0x00070058u, 0x00000007u, 0x00000013u, 0x0000000eu, 0x00000011u, 0x00000002u,
   0x00000012u, 0x0003003eu, 0x00000009u, 0x00000013u, 0x000100fdu, 0x00010038u,
};

/* g0.frag -- ddx = ddy = 1/16, lambda 0 */
static const uint32_t frag_g0_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000017u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x3d800000u, 0x0004002bu, 0x00000006u, 0x00000013u, 0x00000000u, 0x0005002cu,
   0x0000000fu, 0x00000014u, 0x00000012u, 0x00000013u, 0x0005002cu, 0x0000000fu,
   0x00000015u, 0x00000013u, 0x00000012u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu,
   0x0000000eu, 0x0000000du, 0x00080058u, 0x00000007u, 0x00000016u, 0x0000000eu,
   0x00000011u, 0x00000004u, 0x00000014u, 0x00000015u, 0x0003003eu, 0x00000009u,
   0x00000016u, 0x000100fdu, 0x00010038u,
};

/* g1.frag -- ddx = ddy = 2/16, lambda 1 */
static const uint32_t frag_g1_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000017u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x3e000000u, 0x0004002bu, 0x00000006u, 0x00000013u, 0x00000000u, 0x0005002cu,
   0x0000000fu, 0x00000014u, 0x00000012u, 0x00000013u, 0x0005002cu, 0x0000000fu,
   0x00000015u, 0x00000013u, 0x00000012u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu,
   0x0000000eu, 0x0000000du, 0x00080058u, 0x00000007u, 0x00000016u, 0x0000000eu,
   0x00000011u, 0x00000004u, 0x00000014u, 0x00000015u, 0x0003003eu, 0x00000009u,
   0x00000016u, 0x000100fdu, 0x00010038u,
};

/* g2.frag -- ddx = ddy = 4/16, lambda 2 */
static const uint32_t frag_g2_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000017u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x3e800000u, 0x0004002bu, 0x00000006u, 0x00000013u, 0x00000000u, 0x0005002cu,
   0x0000000fu, 0x00000014u, 0x00000012u, 0x00000013u, 0x0005002cu, 0x0000000fu,
   0x00000015u, 0x00000013u, 0x00000012u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu,
   0x0000000eu, 0x0000000du, 0x00080058u, 0x00000007u, 0x00000016u, 0x0000000eu,
   0x00000011u, 0x00000004u, 0x00000014u, 0x00000015u, 0x0003003eu, 0x00000009u,
   0x00000016u, 0x000100fdu, 0x00010038u,
};

/* g3.frag -- ddx = ddy = 8/16, lambda 3 */
static const uint32_t frag_g3_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000016u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x00000000u, 0x0005002cu, 0x0000000fu, 0x00000013u, 0x00000010u, 0x00000012u,
   0x0005002cu, 0x0000000fu, 0x00000014u, 0x00000012u, 0x00000010u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x0004003du, 0x0000000bu, 0x0000000eu, 0x0000000du, 0x00080058u, 0x00000007u,
   0x00000015u, 0x0000000eu, 0x00000011u, 0x00000004u, 0x00000013u, 0x00000014u,
   0x0003003eu, 0x00000009u, 0x00000015u, 0x000100fdu, 0x00010038u,
};

/* g4.frag -- ddx = ddy = 16/16, lambda 4 */
static const uint32_t frag_g4_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000017u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x3f800000u, 0x0004002bu, 0x00000006u, 0x00000013u, 0x00000000u, 0x0005002cu,
   0x0000000fu, 0x00000014u, 0x00000012u, 0x00000013u, 0x0005002cu, 0x0000000fu,
   0x00000015u, 0x00000013u, 0x00000012u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu,
   0x0000000eu, 0x0000000du, 0x00080058u, 0x00000007u, 0x00000016u, 0x0000000eu,
   0x00000011u, 0x00000004u, 0x00000014u, 0x00000015u, 0x0003003eu, 0x00000009u,
   0x00000016u, 0x000100fdu, 0x00010038u,
};

/* gm.frag -- ddx = 4/16, ddy = 1/16; the larger axis wins, lambda 2 */
static const uint32_t frag_gm_spv[] = {
   0x07230203u, 0x00010300u, 0x0008000bu, 0x00000018u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000du, 0x00786574u, 0x00040047u, 0x00000009u,
   0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u,
   0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
   0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u,
   0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000bu, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu, 0x00000006u, 0x00000002u,
   0x0004002bu, 0x00000006u, 0x00000010u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000011u, 0x00000010u, 0x00000010u, 0x0004002bu, 0x00000006u, 0x00000012u,
   0x3e800000u, 0x0004002bu, 0x00000006u, 0x00000013u, 0x00000000u, 0x0005002cu,
   0x0000000fu, 0x00000014u, 0x00000012u, 0x00000013u, 0x0004002bu, 0x00000006u,
   0x00000015u, 0x3d800000u, 0x0005002cu, 0x0000000fu, 0x00000016u, 0x00000013u,
   0x00000015u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,
   0x000200f8u, 0x00000005u, 0x0004003du, 0x0000000bu, 0x0000000eu, 0x0000000du,
   0x00080058u, 0x00000007u, 0x00000017u, 0x0000000eu, 0x00000011u, 0x00000004u,
   0x00000014u, 0x00000016u, 0x0003003eu, 0x00000009u, 0x00000017u, 0x000100fdu,
   0x00010038u,
};

struct scan {
   unsigned matched;      /* pixels equal to the expected colour */
   unsigned clear;        /* pixels still the clear colour */
   unsigned other;        /* pixels that are neither */
   int worst_x, worst_y;
   unsigned char worst[4];
};

static const unsigned char clear_rgba[4] = { 26, 26, 38, 255 };

static const unsigned char *
pixel(const unsigned char *frame, int x, int y)
{
   return frame + ((size_t)y * W + x) * 4;
}

static int
same(const unsigned char *a, const unsigned char *b)
{
   /* Every expected value is a whole 8-bit level fetched rather than
    * filtered, so one LSB of slack is a courtesy and not a tolerance. */
   for (int c = 0; c < 4; c++) {
      int d = (int)a[c] - (int)b[c];
      if (d < -1 || d > 1)
         return 0;
   }
   return 1;
}

static void
scan_image(const unsigned char *frame, const unsigned char *expect,
           struct scan *s)
{
   memset(s, 0, sizeof(*s));
   s->worst_x = s->worst_y = -1;
   for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
         const unsigned char *px = pixel(frame, x, y);
         if (same(px, expect)) {
            s->matched++;
         } else if (same(px, clear_rgba)) {
            s->clear++;
         } else {
            if (s->worst_x < 0) {
               s->worst_x = x;
               s->worst_y = y;
               memcpy(s->worst, px, 4);
            }
            s->other++;
         }
      }
   }
}

static void
describe(const char *what, const unsigned char *rgba)
{
   printf("  %-34s %3u %3u %3u %3u\n", what, rgba[0], rgba[1], rgba[2],
          rgba[3]);
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
    * than that and means it, so ask for the extension when it is advertised,
    * with its dependency chain, exactly as cpvk_gather does.
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
    * There is no feature to ask for and nothing a device may decline.
    * OpImageSampleExplicitLod with Grad is core SPIR-V and textureGrad is core
    * GLSL 1.30, so a driver that cannot serve it has nowhere to say so -- and
    * this one did not say so. No feature is enabled here on purpose:
    * samplerAnisotropy stays off, so no implementation's anisotropic filter
    * can enter and the six answers are the isotropic rule and only that.
    */
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

   /* One buffer, one frame per pass. */
   const VkDeviceSize frame_bytes = (VkDeviceSize)W * H * 4;
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = NPASS * frame_bytes,
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
    * The sampled image: optimal-tiled with a full mip chain, filled from a
    * staging buffer. Linear tiling is not used here even though other tests
    * do: a linear image is only required to support one mip level, and the
    * whole point of this one is the chain.
    */
   VkImageCreateInfo timgi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { TEX_DIM, TEX_DIM, 1 }, .mipLevels = LEVELS, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage timg;
   CHECK(vkCreateImage(dev, &timgi, NULL, &timg));
   VkMemoryRequirements treq;
   vkGetImageMemoryRequirements(dev, timg, &treq);
   uint32_t ttype = pick_memory(pdev, treq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (ttype == UINT32_MAX)
      ttype = pick_memory(pdev, treq.memoryTypeBits, 0);
   if (ttype == UINT32_MAX) { fprintf(stderr, "no texture memory type\n"); return 1; }
   VkMemoryAllocateInfo tmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = treq.size,
                                 .memoryTypeIndex = ttype };
   VkDeviceMemory tmem;
   CHECK(vkAllocateMemory(dev, &tmai, NULL, &tmem));
   CHECK(vkBindImageMemory(dev, timg, tmem, 0));

   /* Every level's texels, back to back, in a staging buffer. */
   VkDeviceSize level_offset[LEVELS];
   VkDeviceSize staging_bytes = 0;
   for (int l = 0; l < LEVELS; l++) {
      unsigned dim = (unsigned)TEX_DIM >> l;
      level_offset[l] = staging_bytes;
      staging_bytes += (VkDeviceSize)dim * dim * 4;
   }
   VkBufferCreateInfo sbci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = staging_bytes,
                               .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT };
   VkBuffer staging;
   CHECK(vkCreateBuffer(dev, &sbci, NULL, &staging));
   VkMemoryRequirements sreq;
   vkGetBufferMemoryRequirements(dev, staging, &sreq);
   uint32_t stype = pick_memory(pdev, sreq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (stype == UINT32_MAX) { fprintf(stderr, "no staging type\n"); return 1; }
   VkMemoryAllocateInfo smai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = sreq.size,
                                 .memoryTypeIndex = stype };
   VkDeviceMemory smem;
   CHECK(vkAllocateMemory(dev, &smai, NULL, &smem));
   CHECK(vkBindBufferMemory(dev, staging, smem, 0));
   {
      unsigned char *p;
      CHECK(vkMapMemory(dev, smem, 0, VK_WHOLE_SIZE, 0, (void **)&p));
      for (int l = 0; l < LEVELS; l++) {
         unsigned dim = (unsigned)TEX_DIM >> l;
         unsigned char *t = p + level_offset[l];
         for (unsigned i = 0; i < dim * dim; i++)
            memcpy(t + i * 4, level_rgba[l], 4);
      }
      VkMappedMemoryRange flush = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = smem, .size = VK_WHOLE_SIZE };
      CHECK(vkFlushMappedMemoryRanges(dev, 1, &flush));
      vkUnmapMemory(dev, smem);
   }

   VkImageViewCreateInfo tvci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = timg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = timgi.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, 1 } };
   VkImageView tview;
   CHECK(vkCreateImageView(dev, &tvci, NULL, &tview));

   /*
    * NEAREST everywhere. A level is a flat colour, so the image filter cannot
    * matter within one; NEAREST mip mode is what makes the *answer* a single
    * level rather than a blend of two, which is what lets the expected pixel
    * be an exact table row. maxLod is the last level: nothing here should be
    * clamped, and if a driver's lambda ran away the clamp is what would hide
    * it, so it is set where the chain ends and no further.
    */
   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
      .minLod = 0.0f, .maxLod = (float)(LEVELS - 1) };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
   VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
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

   VkDescriptorImageInfo dii = {
      .sampler = sampler, .imageView = tview,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
      .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &dii };
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

   VkShaderModuleCreateInfo vsmi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv), .pCode = vert_spv };
   VkShaderModule vs;
   CHECK(vkCreateShaderModule(dev, &vsmi, NULL, &vs));

   /*
    * One fragment shader per pass rather than one with a uniform gradient:
    * the point is to compare a driver that folds nothing against one that
    * does, and a constant gradient is the shape a real shader's dFdx pair
    * reduces to after NIR has finished with it.
    */
   const uint32_t *frag_code[NPASS] = {
      frag_lod_spv, frag_g0_spv, frag_g1_spv, frag_g2_spv, frag_g3_spv,
      frag_g4_spv, frag_gm_spv };
   const size_t frag_size[NPASS] = {
      sizeof(frag_lod_spv), sizeof(frag_g0_spv), sizeof(frag_g1_spv),
      sizeof(frag_g2_spv), sizeof(frag_g3_spv), sizeof(frag_g4_spv),
      sizeof(frag_gm_spv) };
   static const char *const pass_name[NPASS] = {
      "textureLod(uv, 2.0)   control",
      "textureGrad  d = 1/16 ",
      "textureGrad  d = 2/16 ",
      "textureGrad  d = 4/16 ",
      "textureGrad  d = 8/16 ",
      "textureGrad  d = 16/16",
      "textureGrad  dx 4/16 dy 1/16",
   };
   /* The level each pass must come back with. */
   static const int pass_level[NPASS] = { 2, 0, 1, 2, 3, 4, 2 };

   VkShaderModule fs[NPASS];
   for (int p = 0; p < NPASS; p++) {
      VkShaderModuleCreateInfo fsmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = frag_size[p], .pCode = frag_code[p] };
      CHECK(vkCreateShaderModule(dev, &fsmi, NULL, &fs[p]));
   }

   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

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

   VkPipeline pipe[NPASS];
   for (int p = 0; p < NPASS; p++) {
      VkPipelineShaderStageCreateInfo stages[2] = {
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs,
           .pName = "main" },
         { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
           .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs[p],
           .pName = "main" },
      };
      VkGraphicsPipelineCreateInfo gpi = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext = &pri, .stageCount = 2, .pStages = stages,
         .pVertexInputState = &vi, .pInputAssemblyState = &ia,
         .pViewportState = &vps, .pRasterizationState = &rs,
         .pMultisampleState = &ms, .pDepthStencilState = &ds,
         .pColorBlendState = &cb, .layout = layout };
      CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL,
                                      &pipe[p]));
   }

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

   /* Fill the mip chain, then hand the whole image to the fragment stage. */
   VkImageMemoryBarrier tex_to_dst = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = timg,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, 1 } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, NULL, 0, NULL, 1, &tex_to_dst);

   VkBufferImageCopy level_copy[LEVELS];
   for (int l = 0; l < LEVELS; l++) {
      unsigned dim = (unsigned)TEX_DIM >> l;
      level_copy[l] = (VkBufferImageCopy){
         .bufferOffset = level_offset[l],
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)l, 0, 1 },
         .imageExtent = { dim, dim, 1 } };
   }
   vkCmdCopyBufferToImage(cmd, staging, timg,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          LEVELS, level_copy);

   VkImageMemoryBarrier tex_to_read = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = timg,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, 1 } };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
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

   for (int pass = 0; pass < NPASS; pass++) {
      if (pass)
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                              0, NULL, 0, NULL, 1, &back_to_colour);

      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe[pass]);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                              0, 1, &dset, 0, NULL);
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

   const unsigned char *frame[NPASS];
   const unsigned char *got[NPASS];
   struct scan s[NPASS];
   for (int p = 0; p < NPASS; p++) {
      frame[p] = mapped + (size_t)p * frame_bytes;
      got[p] = pixel(frame[p], W / 2, H / 2);
      scan_image(frame[p], level_rgba[pass_level[p]], &s[p]);
   }

   printf("  texture %dx%d, %d levels, sampled at (0.5, 0.5), mip mode "
          "NEAREST\n", TEX_DIM, TEX_DIM, LEVELS);
   for (int l = 0; l < LEVELS; l++) {
      char what[64];
      snprintf(what, sizeof(what), "level %d  %ux%u", l,
               (unsigned)TEX_DIM >> l, (unsigned)TEX_DIM >> l);
      describe(what, level_rgba[l]);
   }
   for (int p = 0; p < NPASS; p++) {
      char what[80];
      snprintf(what, sizeof(what), "%s -> level %d", pass_name[p],
               pass_level[p]);
      describe(what, level_rgba[pass_level[p]]);
      snprintf(what, sizeof(what), "%s    got", pass_name[p]);
      describe(what, got[p]);
      printf("    %u of %d pixels on it, %u still the clear colour, "
             "%u neither\n", s[p].matched, W * H, s[p].clear, s[p].other);
      if (s[p].other)
         printf("    first stray at (%d,%d) = %3u %3u %3u %3u\n",
                s[p].worst_x, s[p].worst_y, s[p].worst[0], s[p].worst[1],
                s[p].worst[2], s[p].worst[3]);
   }

   int fail = 0;

   for (int p = 0; p < NPASS; p++) {
      if (s[p].clear == (unsigned)(W * H)) {
         printf("FAIL %s: the frame is entirely the clear colour, so the draw "
                "did not cover the viewport\n", pass_name[p]);
         fail = 1;
      }
   }

   /*
    * The control first, because it decides what a gradient failure can mean.
    * If this one is wrong then the mip chain, the descriptor or the draw is
    * wrong and nothing below is about textureGrad().
    */
   if (s[0].matched != (unsigned)(W * H)) {
      printf("FAIL the control textureLod(uv, 2.0) did not return level 2 "
             "%u %u %u %u, so nothing below is about the gradient: the mip "
             "chain, the sampler or the descriptor is wrong first\n",
             level_rgba[2][0], level_rgba[2][1], level_rgba[2][2],
             level_rgba[2][3]);
      fail = 1;
   }

   int all_placeholder = 1, all_base = 1;
   for (int p = 1; p < NPASS; p++) {
      all_placeholder &= same(got[p], unsupported_rgba);
      all_base &= same(got[p], level_rgba[0]);
   }

   for (int p = 1; p < NPASS; p++) {
      if (s[p].matched == (unsigned)(W * H))
         continue;
      fail = 1;

      if (same(got[p], unsupported_rgba)) {
         printf("FAIL %s returned %u %u %u %u, which is the constant the "
                "unsupported-texture-op branch of cp_nir_to_llvm.c returns: "
                "nir_texop_txd is not in its supported set, so the sample was "
                "never emitted\n", pass_name[p], got[p][0], got[p][1],
                got[p][2], got[p][3]);
         continue;
      }
      if (same(got[p], level_rgba[0])) {
         printf("FAIL %s returned level 0 %u %u %u %u: the sample was emitted "
                "but the gradients did not reach the level selection, so "
                "every footprint reads the base level\n", pass_name[p],
                got[p][0], got[p][1], got[p][2], got[p][3]);
         continue;
      }
      printf("FAIL %s returned %u %u %u %u, expected level %d "
             "%u %u %u %u on all %d pixels; %u matched\n", pass_name[p],
             got[p][0], got[p][1], got[p][2], got[p][3], pass_level[p],
             level_rgba[pass_level[p]][0], level_rgba[pass_level[p]][1],
             level_rgba[pass_level[p]][2], level_rgba[pass_level[p]][3],
             W * H, s[p].matched);
   }

   if (all_placeholder)
      printf("  every gradient pass returned the placeholder constant: this "
             "driver does not implement textureGrad() at all\n");
   else if (all_base)
      printf("  every gradient pass returned the base level: this driver "
             "samples textureGrad() but ignores the gradients\n");

   if (ppm) {
      FILE *f = fopen(ppm, "wb");
      if (f) {
         fprintf(f, "P6\n%d %d\n255\n", W, H * NPASS);
         for (int p = 0; p < NPASS; p++)
            for (int i = 0; i < W * H; i++)
               fwrite(frame[p] + (size_t)i * 4, 1, 3, f);
         fclose(f);
         printf("  wrote %s (%d frames stacked)\n", ppm, NPASS);
      }
   }

   vkUnmapMemory(dev, bmem);

   for (int p = 0; p < NPASS; p++) {
      vkDestroyPipeline(dev, pipe[p], NULL);
      vkDestroyShaderModule(dev, fs[p], NULL);
   }
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyImageView(dev, tview, NULL);
   vkDestroyImage(dev, timg, NULL);
   vkFreeMemory(dev, tmem, NULL);
   vkDestroyBuffer(dev, staging, NULL);
   vkFreeMemory(dev, smem, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   printf("%s\n", fail ? "FAIL" : "PASS");
   return fail;
}
