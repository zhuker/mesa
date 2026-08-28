/*
 * step(): the one comparison in GLSL that returns a number instead of a bool,
 * and the one this driver got wrong on every shader that used it.
 *
 * `step(edge, x)` is defined by the spec as
 *
 *    0.0 if x < edge, otherwise 1.0
 *
 * which is the same question `x < edge ? 0.0 : 1.0` asks -- except that the
 * two travel through the compiler by completely different routes. The ternary
 * becomes an ordinary SPIR-V `OpFOrdLessThan` and an `OpSelect`, so it lands
 * on NIR's `flt` and `bcsel`, both of which every backend implements because
 * every `if` in every shader needs them. `step()` becomes
 * `OpExtInst GLSL.std.450 Step`, which `vtn_glsl450.c` lowers to
 *
 *    1.0 - slt(x, edge)          with preserve-nan and preserve-inf set
 *
 * and which `nir_opt_algebraic` then rewrites twice -- `slt` to `b2f(flt)`
 * under `options->lower_scmp`, and `1.0 - b2f(c)` to `b2f(!c)` -- until what
 * a backend is finally handed is
 *
 *    b2f32(inot(flt(x, edge)))
 *
 * The `inot` is the difference, and it is the only one. It is a *logical* not
 * of a boolean, written as NIR's bitwise operator at bit_size 1, and nothing
 * an ordinary comparison compiles to ever needs it. A backend that has one
 * representation for a boolean and negates it in another gets `step()` wrong
 * and gets `<` right, which is why nothing else in this suite noticed.
 *
 * Four passes, one draw each, every pixel of every frame checked. All of them
 * read the same uniform buffer, so nothing here is a constant the compiler can
 * fold away before the comparison is emitted:
 *
 *    edge = 0.5   below = 0.25   at = 0.5   above = 0.75
 *
 * `at` is a separate uniform that happens to hold the edge value. The
 * comparison `at < edge` is false, so `step(edge, at) == 1.0` -- the boundary
 * belongs to the upper side, and a driver that implements step with `<=`
 * instead of `<` fails on that channel alone.
 *
 * | pass | what it computes                        | R      G      B      A   |
 * |------|-----------------------------------------|--------------------------|
 * | 0    | `x < edge ? 1.0 : 0.0`, the control     | 1.0    0.0    0.0    1.0 |
 * | 1    | `step(edge, x)`, scalar                 | 0.0    1.0    1.0    1.0 |
 * | 2    | `step(vec3(edge), vec3(x))`             | 0.0    1.0    1.0    1.0 |
 * | 3    | the capture's own inside-the-volume test | 0.0   1.0    0.0    1.0 |
 *
 * with R fed `below`, G fed `at` and B fed `above` in passes 0 to 2. Note that
 * the control's R and step's R are opposite by construction: `step` is 0 where
 * `<` is true. Every value is exactly 0.0 or 1.0 and the attachment is
 * R8G8B8A8_UNORM, so every expected byte is exactly 0 or 255; the one LSB of
 * slack below is a courtesy, not a tolerance.
 *
 * Pass 0 is the control and it is what makes a failure mean something. It asks
 * the same question of the same three uniforms through the same pipeline
 * layout and the same descriptor set. If the control is right and the step
 * frames are wrong, then the uniform buffer arrived, the descriptor was bound,
 * the triangle covered the viewport and the float compare works -- so what is
 * left is the lowering of `step` and nothing else.
 *
 * Pass 3 is the shape the bug was found in, kept because it is the one that
 * costs a frame its shadows. `/home/alexzhukov/gather-validation/DRAW_BISECT.md`
 * traces a whole capture's missing shadows to fragment module 348, which asks
 *
 *    outside = step(half_extent, abs(pos - centre));
 *    any_outside = clamp(dot(outside, vec3(1.0)), 0.0, 1.0);
 *
 * and then uses `any_outside` to `mix()` away the baked sun-shadow and
 * ambient-occlusion lookup. Every axis of that draw is inside its half extent,
 * so the answer is 0.0 and the volume is read; a driver that answers 1.0 skips
 * the lookup and leaves every lit surface fully lit. `Step` appears in 352 of
 * that capture's 1066 shader modules, so this is not one shader's problem.
 *
 * Two neighbours were checked and are *not* in this family, which is why they
 * are not tested here:
 *
 *   - `smoothstep()` lowers through `nir_smoothstep()` in
 *     `nir_builtin_builder.c` -- fsub, fdiv, fsat and two fmuls. No `slt`.
 *   - `sign()` lowers to `nir_op_fsign`, which `cp_nir_to_llvm.c` implements
 *     directly (the -1/0/+1 select at `case nir_op_fsign`).
 *
 * So `step()` is the shortest route in the language to a boolean `inot`, and
 * this test is the shortest route to `step()`.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, so this runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. The uniform buffer is host-visible and written once before the
 * submit. SPIR-V is embedded; nothing is read from disk.
 *
 *   cc -std=c11 -Wall -Wextra -Werror src/cudavk/tests/cpvk_step.c \
 *      -o /tmp/cpvk-tests/cpvk_step -lvulkan
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- s.vert --
 *   #version 450
 *   const vec2 base[3] = vec2[3](vec2(-1.0, -1.0),
 *                                vec2( 3.0, -1.0),
 *                                vec2(-1.0,  3.0));
 *   void main()
 *   {
 *      gl_Position = vec4(base[gl_VertexIndex], 0.0, 1.0);
 *   }
 *
 *   -- the uniform block, shared by all four fragment shaders --
 *   layout(set = 0, binding = 0) uniform U {
 *      vec4 edge;
 *      vec4 below;
 *      vec4 at;
 *      vec4 above;
 *   } u;
 *
 *   -- s0.frag --  the control: an ordinary float compare and a select
 *   void main()
 *   {
 *      out_colour = vec4(u.below.x < u.edge.x ? 1.0 : 0.0,
 *                        u.at.x    < u.edge.x ? 1.0 : 0.0,
 *                        u.above.x < u.edge.x ? 1.0 : 0.0,
 *                        1.0);
 *   }
 *
 *   -- s1.frag --  scalar step, the same three questions
 *   void main()
 *   {
 *      out_colour = vec4(step(u.edge.x, u.below.x),
 *                        step(u.edge.x, u.at.x),
 *                        step(u.edge.x, u.above.x),
 *                        1.0);
 *   }
 *
 *   -- s2.frag --  vector step, which is what a shader actually writes
 *   void main()
 *   {
 *      vec3 x = vec3(u.below.x, u.at.x, u.above.x);
 *      out_colour = vec4(step(u.edge.xyz, x), 1.0);
 *   }
 *
 *   -- s3.frag --  the capture's inside-the-volume test, every axis inside
 *   void main()
 *   {
 *      vec3 outside = step(u.edge.xyz, u.below.xyz);
 *      float sum = dot(outside, vec3(1.0));
 *      float any_outside = clamp(sum, 0.0, 1.0);
 *      out_colour = vec4(any_outside, 1.0 - any_outside, sum / 3.0, 1.0);
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

#define NPASS 4      /* control, scalar step, vec3 step, the capture's shape */

/* The four uniforms, and the only numbers this test depends on. `at` holds the
 * edge value but lives at its own offset, so no amount of constant folding can
 * turn the comparison into a literal before the backend sees it. */
#define EDGE   0.5f
#define BELOW  0.25f
#define AT     0.5f
#define ABOVE  0.75f

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* std140: four vec4s, sixteen bytes each. */
struct ubo {
   float edge[4];
   float below[4];
   float at[4];
   float above[4];
};

/* s.vert -- one triangle over the whole viewport. */
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

/* s0.frag -- the control: `x < edge ? 1.0 : 0.0`, three times. */
static const uint32_t frag_cmp_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002bu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000au, 0x00000055u, 0x00050006u, 0x0000000au,
   0x00000000u, 0x65676465u, 0x00000000u, 0x00050006u, 0x0000000au, 0x00000001u,
   0x6f6c6562u, 0x00000077u, 0x00040006u, 0x0000000au, 0x00000002u, 0x00007461u,
   0x00050006u, 0x0000000au, 0x00000003u, 0x766f6261u, 0x00000065u, 0x00030005u,
   0x0000000cu, 0x00000075u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u,
   0x00030047u, 0x0000000au, 0x00000002u, 0x00050048u, 0x0000000au, 0x00000000u,
   0x00000023u, 0x00000000u, 0x00050048u, 0x0000000au, 0x00000001u, 0x00000023u,
   0x00000010u, 0x00050048u, 0x0000000au, 0x00000002u, 0x00000023u, 0x00000020u,
   0x00050048u, 0x0000000au, 0x00000003u, 0x00000023u, 0x00000030u, 0x00040047u,
   0x0000000cu, 0x00000021u, 0x00000000u, 0x00040047u, 0x0000000cu, 0x00000022u,
   0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u,
   0x00000004u, 0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu,
   0x00000008u, 0x00000009u, 0x00000003u, 0x0006001eu, 0x0000000au, 0x00000007u,
   0x00000007u, 0x00000007u, 0x00000007u, 0x00040020u, 0x0000000bu, 0x00000002u,
   0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000002u, 0x00040015u,
   0x0000000du, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000du, 0x0000000eu,
   0x00000001u, 0x00040015u, 0x0000000fu, 0x00000020u, 0x00000000u, 0x0004002bu,
   0x0000000fu, 0x00000010u, 0x00000000u, 0x00040020u, 0x00000011u, 0x00000002u,
   0x00000006u, 0x0004002bu, 0x0000000du, 0x00000014u, 0x00000000u, 0x00020014u,
   0x00000017u, 0x0004002bu, 0x00000006u, 0x00000019u, 0x3f800000u, 0x0004002bu,
   0x00000006u, 0x0000001au, 0x00000000u, 0x0004002bu, 0x0000000du, 0x0000001cu,
   0x00000002u, 0x0004002bu, 0x0000000du, 0x00000023u, 0x00000003u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x00060041u, 0x00000011u, 0x00000012u, 0x0000000cu, 0x0000000eu, 0x00000010u,
   0x0004003du, 0x00000006u, 0x00000013u, 0x00000012u, 0x00060041u, 0x00000011u,
   0x00000015u, 0x0000000cu, 0x00000014u, 0x00000010u, 0x0004003du, 0x00000006u,
   0x00000016u, 0x00000015u, 0x000500b8u, 0x00000017u, 0x00000018u, 0x00000013u,
   0x00000016u, 0x000600a9u, 0x00000006u, 0x0000001bu, 0x00000018u, 0x00000019u,
   0x0000001au, 0x00060041u, 0x00000011u, 0x0000001du, 0x0000000cu, 0x0000001cu,
   0x00000010u, 0x0004003du, 0x00000006u, 0x0000001eu, 0x0000001du, 0x00060041u,
   0x00000011u, 0x0000001fu, 0x0000000cu, 0x00000014u, 0x00000010u, 0x0004003du,
   0x00000006u, 0x00000020u, 0x0000001fu, 0x000500b8u, 0x00000017u, 0x00000021u,
   0x0000001eu, 0x00000020u, 0x000600a9u, 0x00000006u, 0x00000022u, 0x00000021u,
   0x00000019u, 0x0000001au, 0x00060041u, 0x00000011u, 0x00000024u, 0x0000000cu,
   0x00000023u, 0x00000010u, 0x0004003du, 0x00000006u, 0x00000025u, 0x00000024u,
   0x00060041u, 0x00000011u, 0x00000026u, 0x0000000cu, 0x00000014u, 0x00000010u,
   0x0004003du, 0x00000006u, 0x00000027u, 0x00000026u, 0x000500b8u, 0x00000017u,
   0x00000028u, 0x00000025u, 0x00000027u, 0x000600a9u, 0x00000006u, 0x00000029u,
   0x00000028u, 0x00000019u, 0x0000001au, 0x00070050u, 0x00000007u, 0x0000002au,
   0x0000001bu, 0x00000022u, 0x00000029u, 0x00000019u, 0x0003003eu, 0x00000009u,
   0x0000002au, 0x000100fdu, 0x00010038u
};

/* s1.frag -- scalar step(), the same three questions. */
static const uint32_t frag_step_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000026u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00050005u, 0x00000009u, 0x5f74756fu, 0x6f6c6f63u,
   0x00007275u, 0x00030005u, 0x0000000au, 0x00000055u, 0x00050006u, 0x0000000au,
   0x00000000u, 0x65676465u, 0x00000000u, 0x00050006u, 0x0000000au, 0x00000001u,
   0x6f6c6562u, 0x00000077u, 0x00040006u, 0x0000000au, 0x00000002u, 0x00007461u,
   0x00050006u, 0x0000000au, 0x00000003u, 0x766f6261u, 0x00000065u, 0x00030005u,
   0x0000000cu, 0x00000075u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u,
   0x00030047u, 0x0000000au, 0x00000002u, 0x00050048u, 0x0000000au, 0x00000000u,
   0x00000023u, 0x00000000u, 0x00050048u, 0x0000000au, 0x00000001u, 0x00000023u,
   0x00000010u, 0x00050048u, 0x0000000au, 0x00000002u, 0x00000023u, 0x00000020u,
   0x00050048u, 0x0000000au, 0x00000003u, 0x00000023u, 0x00000030u, 0x00040047u,
   0x0000000cu, 0x00000021u, 0x00000000u, 0x00040047u, 0x0000000cu, 0x00000022u,
   0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u,
   0x00000004u, 0x00040020u, 0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu,
   0x00000008u, 0x00000009u, 0x00000003u, 0x0006001eu, 0x0000000au, 0x00000007u,
   0x00000007u, 0x00000007u, 0x00000007u, 0x00040020u, 0x0000000bu, 0x00000002u,
   0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000002u, 0x00040015u,
   0x0000000du, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000du, 0x0000000eu,
   0x00000000u, 0x00040015u, 0x0000000fu, 0x00000020u, 0x00000000u, 0x0004002bu,
   0x0000000fu, 0x00000010u, 0x00000000u, 0x00040020u, 0x00000011u, 0x00000002u,
   0x00000006u, 0x0004002bu, 0x0000000du, 0x00000014u, 0x00000001u, 0x0004002bu,
   0x0000000du, 0x0000001au, 0x00000002u, 0x0004002bu, 0x0000000du, 0x00000020u,
   0x00000003u, 0x0004002bu, 0x00000006u, 0x00000024u, 0x3f800000u, 0x00050036u,
   0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u,
   0x00060041u, 0x00000011u, 0x00000012u, 0x0000000cu, 0x0000000eu, 0x00000010u,
   0x0004003du, 0x00000006u, 0x00000013u, 0x00000012u, 0x00060041u, 0x00000011u,
   0x00000015u, 0x0000000cu, 0x00000014u, 0x00000010u, 0x0004003du, 0x00000006u,
   0x00000016u, 0x00000015u, 0x0007000cu, 0x00000006u, 0x00000017u, 0x00000001u,
   0x00000030u, 0x00000013u, 0x00000016u, 0x00060041u, 0x00000011u, 0x00000018u,
   0x0000000cu, 0x0000000eu, 0x00000010u, 0x0004003du, 0x00000006u, 0x00000019u,
   0x00000018u, 0x00060041u, 0x00000011u, 0x0000001bu, 0x0000000cu, 0x0000001au,
   0x00000010u, 0x0004003du, 0x00000006u, 0x0000001cu, 0x0000001bu, 0x0007000cu,
   0x00000006u, 0x0000001du, 0x00000001u, 0x00000030u, 0x00000019u, 0x0000001cu,
   0x00060041u, 0x00000011u, 0x0000001eu, 0x0000000cu, 0x0000000eu, 0x00000010u,
   0x0004003du, 0x00000006u, 0x0000001fu, 0x0000001eu, 0x00060041u, 0x00000011u,
   0x00000021u, 0x0000000cu, 0x00000020u, 0x00000010u, 0x0004003du, 0x00000006u,
   0x00000022u, 0x00000021u, 0x0007000cu, 0x00000006u, 0x00000023u, 0x00000001u,
   0x00000030u, 0x0000001fu, 0x00000022u, 0x00070050u, 0x00000007u, 0x00000025u,
   0x00000017u, 0x0000001du, 0x00000023u, 0x00000024u, 0x0003003eu, 0x00000009u,
   0x00000025u, 0x000100fdu, 0x00010038u
};

/* s2.frag -- vec3 step(), one instruction for the three. */
static const uint32_t frag_step3_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002au, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x0000001du, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00030005u, 0x00000009u, 0x00000078u, 0x00030005u,
   0x0000000bu, 0x00000055u, 0x00050006u, 0x0000000bu, 0x00000000u, 0x65676465u,
   0x00000000u, 0x00050006u, 0x0000000bu, 0x00000001u, 0x6f6c6562u, 0x00000077u,
   0x00040006u, 0x0000000bu, 0x00000002u, 0x00007461u, 0x00050006u, 0x0000000bu,
   0x00000003u, 0x766f6261u, 0x00000065u, 0x00030005u, 0x0000000du, 0x00000075u,
   0x00050005u, 0x0000001du, 0x5f74756fu, 0x6f6c6f63u, 0x00007275u, 0x00030047u,
   0x0000000bu, 0x00000002u, 0x00050048u, 0x0000000bu, 0x00000000u, 0x00000023u,
   0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u, 0x00000023u, 0x00000010u,
   0x00050048u, 0x0000000bu, 0x00000002u, 0x00000023u, 0x00000020u, 0x00050048u,
   0x0000000bu, 0x00000003u, 0x00000023u, 0x00000030u, 0x00040047u, 0x0000000du,
   0x00000021u, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u,
   0x00040047u, 0x0000001du, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000003u, 0x00040020u, 0x00000008u,
   0x00000007u, 0x00000007u, 0x00040017u, 0x0000000au, 0x00000006u, 0x00000004u,
   0x0006001eu, 0x0000000bu, 0x0000000au, 0x0000000au, 0x0000000au, 0x0000000au,
   0x00040020u, 0x0000000cu, 0x00000002u, 0x0000000bu, 0x0004003bu, 0x0000000cu,
   0x0000000du, 0x00000002u, 0x00040015u, 0x0000000eu, 0x00000020u, 0x00000001u,
   0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000001u, 0x00040015u, 0x00000010u,
   0x00000020u, 0x00000000u, 0x0004002bu, 0x00000010u, 0x00000011u, 0x00000000u,
   0x00040020u, 0x00000012u, 0x00000002u, 0x00000006u, 0x0004002bu, 0x0000000eu,
   0x00000015u, 0x00000002u, 0x0004002bu, 0x0000000eu, 0x00000018u, 0x00000003u,
   0x00040020u, 0x0000001cu, 0x00000003u, 0x0000000au, 0x0004003bu, 0x0000001cu,
   0x0000001du, 0x00000003u, 0x0004002bu, 0x0000000eu, 0x0000001eu, 0x00000000u,
   0x00040020u, 0x0000001fu, 0x00000002u, 0x0000000au, 0x0004002bu, 0x00000006u,
   0x00000025u, 0x3f800000u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u,
   0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000008u, 0x00000009u,
   0x00000007u, 0x00060041u, 0x00000012u, 0x00000013u, 0x0000000du, 0x0000000fu,
   0x00000011u, 0x0004003du, 0x00000006u, 0x00000014u, 0x00000013u, 0x00060041u,
   0x00000012u, 0x00000016u, 0x0000000du, 0x00000015u, 0x00000011u, 0x0004003du,
   0x00000006u, 0x00000017u, 0x00000016u, 0x00060041u, 0x00000012u, 0x00000019u,
   0x0000000du, 0x00000018u, 0x00000011u, 0x0004003du, 0x00000006u, 0x0000001au,
   0x00000019u, 0x00060050u, 0x00000007u, 0x0000001bu, 0x00000014u, 0x00000017u,
   0x0000001au, 0x0003003eu, 0x00000009u, 0x0000001bu, 0x00050041u, 0x0000001fu,
   0x00000020u, 0x0000000du, 0x0000001eu, 0x0004003du, 0x0000000au, 0x00000021u,
   0x00000020u, 0x0008004fu, 0x00000007u, 0x00000022u, 0x00000021u, 0x00000021u,
   0x00000000u, 0x00000001u, 0x00000002u, 0x0004003du, 0x00000007u, 0x00000023u,
   0x00000009u, 0x0007000cu, 0x00000007u, 0x00000024u, 0x00000001u, 0x00000030u,
   0x00000022u, 0x00000023u, 0x00050051u, 0x00000006u, 0x00000026u, 0x00000024u,
   0x00000000u, 0x00050051u, 0x00000006u, 0x00000027u, 0x00000024u, 0x00000001u,
   0x00050051u, 0x00000006u, 0x00000028u, 0x00000024u, 0x00000002u, 0x00070050u,
   0x0000000au, 0x00000029u, 0x00000026u, 0x00000027u, 0x00000028u, 0x00000025u,
   0x0003003eu, 0x0000001du, 0x00000029u, 0x000100fdu, 0x00010038u
};

/* s3.frag -- the capture's inside-the-volume test, every axis inside. */
static const uint32_t frag_volume_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002cu, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000024u, 0x00030010u, 0x00000004u,
   0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u,
   0x6e69616du, 0x00000000u, 0x00040005u, 0x00000009u, 0x7374756fu, 0x00656469u,
   0x00030005u, 0x0000000bu, 0x00000055u, 0x00050006u, 0x0000000bu, 0x00000000u,
   0x65676465u, 0x00000000u, 0x00050006u, 0x0000000bu, 0x00000001u, 0x6f6c6562u,
   0x00000077u, 0x00040006u, 0x0000000bu, 0x00000002u, 0x00007461u, 0x00050006u,
   0x0000000bu, 0x00000003u, 0x766f6261u, 0x00000065u, 0x00030005u, 0x0000000du,
   0x00000075u, 0x00030005u, 0x0000001au, 0x006d7573u, 0x00050005u, 0x0000001fu,
   0x5f796e61u, 0x7374756fu, 0x00656469u, 0x00050005u, 0x00000024u, 0x5f74756fu,
   0x6f6c6f63u, 0x00007275u, 0x00030047u, 0x0000000bu, 0x00000002u, 0x00050048u,
   0x0000000bu, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x0000000bu,
   0x00000001u, 0x00000023u, 0x00000010u, 0x00050048u, 0x0000000bu, 0x00000002u,
   0x00000023u, 0x00000020u, 0x00050048u, 0x0000000bu, 0x00000003u, 0x00000023u,
   0x00000030u, 0x00040047u, 0x0000000du, 0x00000021u, 0x00000000u, 0x00040047u,
   0x0000000du, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000024u, 0x0000001eu,
   0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u,
   0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u,
   0x00000003u, 0x00040020u, 0x00000008u, 0x00000007u, 0x00000007u, 0x00040017u,
   0x0000000au, 0x00000006u, 0x00000004u, 0x0006001eu, 0x0000000bu, 0x0000000au,
   0x0000000au, 0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000002u,
   0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000002u, 0x00040015u,
   0x0000000eu, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu,
   0x00000000u, 0x00040020u, 0x00000010u, 0x00000002u, 0x0000000au, 0x0004002bu,
   0x0000000eu, 0x00000014u, 0x00000001u, 0x00040020u, 0x00000019u, 0x00000007u,
   0x00000006u, 0x0004002bu, 0x00000006u, 0x0000001cu, 0x3f800000u, 0x0006002cu,
   0x00000007u, 0x0000001du, 0x0000001cu, 0x0000001cu, 0x0000001cu, 0x0004002bu,
   0x00000006u, 0x00000021u, 0x00000000u, 0x00040020u, 0x00000023u, 0x00000003u,
   0x0000000au, 0x0004003bu, 0x00000023u, 0x00000024u, 0x00000003u, 0x0004002bu,
   0x00000006u, 0x00000029u, 0x40400000u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000008u,
   0x00000009u, 0x00000007u, 0x0004003bu, 0x00000019u, 0x0000001au, 0x00000007u,
   0x0004003bu, 0x00000019u, 0x0000001fu, 0x00000007u, 0x00050041u, 0x00000010u,
   0x00000011u, 0x0000000du, 0x0000000fu, 0x0004003du, 0x0000000au, 0x00000012u,
   0x00000011u, 0x0008004fu, 0x00000007u, 0x00000013u, 0x00000012u, 0x00000012u,
   0x00000000u, 0x00000001u, 0x00000002u, 0x00050041u, 0x00000010u, 0x00000015u,
   0x0000000du, 0x00000014u, 0x0004003du, 0x0000000au, 0x00000016u, 0x00000015u,
   0x0008004fu, 0x00000007u, 0x00000017u, 0x00000016u, 0x00000016u, 0x00000000u,
   0x00000001u, 0x00000002u, 0x0007000cu, 0x00000007u, 0x00000018u, 0x00000001u,
   0x00000030u, 0x00000013u, 0x00000017u, 0x0003003eu, 0x00000009u, 0x00000018u,
   0x0004003du, 0x00000007u, 0x0000001bu, 0x00000009u, 0x00050094u, 0x00000006u,
   0x0000001eu, 0x0000001bu, 0x0000001du, 0x0003003eu, 0x0000001au, 0x0000001eu,
   0x0004003du, 0x00000006u, 0x00000020u, 0x0000001au, 0x0008000cu, 0x00000006u,
   0x00000022u, 0x00000001u, 0x0000002bu, 0x00000020u, 0x00000021u, 0x0000001cu,
   0x0003003eu, 0x0000001fu, 0x00000022u, 0x0004003du, 0x00000006u, 0x00000025u,
   0x0000001fu, 0x0004003du, 0x00000006u, 0x00000026u, 0x0000001fu, 0x00050083u,
   0x00000006u, 0x00000027u, 0x0000001cu, 0x00000026u, 0x0004003du, 0x00000006u,
   0x00000028u, 0x0000001au, 0x00050088u, 0x00000006u, 0x0000002au, 0x00000028u,
   0x00000029u, 0x00070050u, 0x0000000au, 0x0000002bu, 0x00000025u, 0x00000027u,
   0x0000002au, 0x0000001cu, 0x0003003eu, 0x00000024u, 0x0000002bu, 0x000100fdu,
   0x00010038u
};

/*
 * What each frame must be, in the order the passes run. Every channel is a
 * whole 0.0 or 1.0, so every byte is 0 or 255.
 *
 *   R  fed `below` (0.25), which is under the edge
 *   G  fed `at`    (0.50), which is exactly the edge
 *   B  fed `above` (0.75), which is over the edge
 *
 * `step` is 0 where `<` is true, so pass 0's R is 255 and pass 1's R is 0.
 * That opposition is the assertion: one of them is the language's answer
 * spelled the easy way, the other is the same answer spelled the way that
 * reaches `slt`.
 */
static const unsigned char expect[NPASS][4] = {
   { 255,   0,   0, 255 },   /* 0: below < edge, at !< edge, above !< edge */
   {   0, 255, 255, 255 },   /* 1: step(edge, below/at/above) */
   {   0, 255, 255, 255 },   /* 2: the same, one vec3 at a time */
   {   0, 255,   0, 255 },   /* 3: nothing is outside, so nothing is masked */
};

static const char *const pass_name[NPASS] = {
   "x < edge ? 1 : 0   control",
   "step() scalar",
   "step() vec3",
   "step() volume test",
};

/*
 * The wrong frame this bug produces, named so a failure reads as a diagnosis
 * rather than as four numbers. `step` stuck at 1.0 paints every channel white
 * in passes 1 and 2, and in pass 3 it reports "outside" -- R 255, G 0, B 255,
 * which is the frame that costs a capture its shadows.
 */
static const unsigned char stuck_high[NPASS][4] = {
   { 255,   0,   0, 255 },   /* the control cannot show this failure */
   { 255, 255, 255, 255 },
   { 255, 255, 255, 255 },
   { 255,   0, 255, 255 },
};

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

/* One LSB of slack, and no more: every expected channel is exactly 0.0 or 1.0
 * written into a UNORM8 attachment. */
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
    * There is no feature, extension or limit to ask about here. step() is core
    * GLSL 1.10 and OpExtInst Step is core GLSL.std.450, every implementation
    * must serve it, and nothing in Vulkan lets a device say it cannot -- which
    * is exactly why a driver that gets it wrong gets it wrong silently.
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

   /* One buffer, four frames. */
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
    * The uniform buffer. The four values are written once, from the host,
    * before the submit -- the point of putting them here rather than in the
    * shader is that the compiler cannot see them, so `slt` survives constant
    * folding and the backend has to emit it.
    */
   struct ubo uvals;
   for (int i = 0; i < 4; i++) {
      uvals.edge[i] = EDGE;
      uvals.below[i] = BELOW;
      uvals.at[i] = AT;
      uvals.above[i] = ABOVE;
   }
   VkBufferCreateInfo ubci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = sizeof(uvals),
                               .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
   VkBuffer ubuf;
   CHECK(vkCreateBuffer(dev, &ubci, NULL, &ubuf));
   VkMemoryRequirements ureq;
   vkGetBufferMemoryRequirements(dev, ubuf, &ureq);
   uint32_t utype = pick_memory(pdev, ureq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (utype == UINT32_MAX)
      utype = pick_memory(pdev, ureq.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (utype == UINT32_MAX) { fprintf(stderr, "no uniform memory type\n"); return 1; }
   VkMemoryAllocateInfo umai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = ureq.size,
                                 .memoryTypeIndex = utype };
   VkDeviceMemory umem;
   CHECK(vkAllocateMemory(dev, &umai, NULL, &umem));
   CHECK(vkBindBufferMemory(dev, ubuf, umem, 0));
   {
      void *p;
      CHECK(vkMapMemory(dev, umem, 0, VK_WHOLE_SIZE, 0, &p));
      memcpy(p, &uvals, sizeof(uvals));
      VkMappedMemoryRange flush = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = umem, .size = VK_WHOLE_SIZE };
      CHECK(vkFlushMappedMemoryRanges(dev, 1, &flush));
      vkUnmapMemory(dev, umem);
   }

   VkDescriptorSetLayoutBinding dslb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
   VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &dslb };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
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

   VkDescriptorBufferInfo dbi = { .buffer = ubuf, .offset = 0,
                                  .range = sizeof(uvals) };
   VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset,
      .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &dbi };
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

   VkShaderModuleCreateInfo vsmi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vert_spv), .pCode = vert_spv };
   VkShaderModule vs;
   CHECK(vkCreateShaderModule(dev, &vsmi, NULL, &vs));

   /*
    * Four fragment shaders over one layout, one descriptor set and one uniform
    * buffer: the only thing that changes between the passes is how the shader
    * spells the comparison.
    */
   const uint32_t *frag_code[NPASS] = {
      frag_cmp_spv, frag_step_spv, frag_step3_spv, frag_volume_spv };
   const size_t frag_size[NPASS] = {
      sizeof(frag_cmp_spv), sizeof(frag_step_spv), sizeof(frag_step3_spv),
      sizeof(frag_volume_spv) };
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

   /* The host wrote the uniforms; hand them to the fragment stage. */
   VkMemoryBarrier host_to_shader = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                        1, &host_to_shader, 0, NULL, 0, NULL);

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
      scan_image(frame[p], expect[p], &s[p]);
   }

   printf("  edge = %.2f, and the three x it is compared against: "
          "below = %.2f, at = %.2f, above = %.2f\n",
          (double)EDGE, (double)BELOW, (double)AT, (double)ABOVE);
   printf("  R is fed below, G is fed at, B is fed above\n");
   for (int p = 0; p < NPASS; p++) {
      char what[64];
      snprintf(what, sizeof(what), "%s expected", pass_name[p]);
      describe(what, expect[p]);
      snprintf(what, sizeof(what), "%s got", pass_name[p]);
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
    * The control first, because it decides what a step failure can mean. If
    * this one is wrong then the uniform buffer, the descriptor or the draw is
    * wrong, and nothing below is about step().
    */
   if (s[0].matched != (unsigned)(W * H)) {
      printf("FAIL the control `x < edge ? 1 : 0` returned %u %u %u %u instead "
             "of %u %u %u %u, so nothing below is about step(): the uniform "
             "buffer, the descriptor or the float compare is wrong first\n",
             got[0][0], got[0][1], got[0][2], got[0][3],
             expect[0][0], expect[0][1], expect[0][2], expect[0][3]);
      fail = 1;
   }

   for (int p = 1; p < NPASS; p++) {
      if (s[p].matched == (unsigned)(W * H))
         continue;
      fail = 1;

      if (same(got[p], stuck_high[p])) {
         printf("FAIL %s returned %u %u %u %u, which is step() stuck at 1.0: "
                "the answer for x = %.2f, below the edge, must be 0.0, and the "
                "control got that same comparison right. What the backend is "
                "handed is `b2f32(inot(flt(x, edge)))`; a boolean `inot` that "
                "does not produce a false the consumer recognises makes every "
                "step() 1.0, exactly this frame\n",
                pass_name[p], got[p][0], got[p][1], got[p][2], got[p][3],
                (double)BELOW);
         continue;
      }
      printf("FAIL %s painted %u of %d pixels %u %u %u %u; it should be "
             "%u %u %u %u\n", pass_name[p], s[p].matched, W * H,
             got[p][0], got[p][1], got[p][2], got[p][3],
             expect[p][0], expect[p][1], expect[p][2], expect[p][3]);
   }

   /*
    * And the failure that survives getting the numbers right: step() and the
    * ternary must disagree on R and agree everywhere else. If the two frames
    * are identical, one of them is not being compiled from what it says.
    */
   if (!fail && same(got[0], got[1])) {
      printf("FAIL the control and step() returned the same pixel %u %u %u %u; "
             "step() is 0 exactly where `<` is 1, so they cannot match\n",
             got[0][0], got[0][1], got[0][2], got[0][3]);
      fail = 1;
   }

   if (!fail)
      printf("PASS step() returned 0.0 below the edge and 1.0 at and above it, "
             "scalar and vec3, and the volume test agreed with the control\n");

   if (ppm) {
      static const char *const suffix[NPASS] = {
         "", ".step", ".step3", ".volume" };
      for (int p = 0; p < NPASS; p++) {
         char path[1024];
         snprintf(path, sizeof(path), "%.*s%s", (int)(sizeof(path) - 32), ppm,
                  suffix[p]);
         write_ppm(p ? path : ppm, frame[p]);
      }
   }

   vkUnmapMemory(dev, bmem);
   vkDestroyCommandPool(dev, pool, NULL);
   for (int p = 0; p < NPASS; p++) {
      vkDestroyPipeline(dev, pipe[p], NULL);
      vkDestroyShaderModule(dev, fs[p], NULL);
   }
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroyBuffer(dev, ubuf, NULL);
   vkFreeMemory(dev, umem, NULL);
   vkDestroyBuffer(dev, readback, NULL);
   vkFreeMemory(dev, bmem, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, imem, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fail;
}
