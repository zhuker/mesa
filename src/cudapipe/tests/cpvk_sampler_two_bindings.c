/*
 * Two combined-image-sampler bindings, two different samplers, one fragment
 * shader, one draw: the sampler state a driver bakes into the kernel must
 * follow the binding the fetch names, not whichever descriptor it saw first.
 *
 * The native driver specialises fragment shaders on sampler state. It walks
 * the NIR for a sampler descriptor it can identify, reads that sampler's
 * filter, wrap and border out of the descriptor set, and compiles them into
 * the kernel as constants, so the texture fetch becomes straight line code
 * instead of a state lookup. One baked state is only sound while every fetch
 * in the shader wants that state. Today the specialiser declines any shader
 * with more than one sampler descriptor; relaxing that to "N descriptors as
 * long as they all resolve to the same state" is safe exactly to the extent
 * that the "all the same" test is real. This is the shader that test has to
 * reject:
 *
 *    layout(set = 0, binding = 0) uniform sampler2D tex0;   VK_FILTER_NEAREST
 *    layout(set = 0, binding = 1) uniform sampler2D tex1;   VK_FILTER_LINEAR
 *
 * Two separate bindings, not an array -- both are named statically, both are
 * fetched in every invocation, and the two descriptors resolve to different
 * sampler states. If one of them is baked for both, the frame is wrong and
 * nothing reports it.
 *
 * Both bindings view one image, so the image cannot be what separates the
 * two fetches -- only the sampler can. The image is 2x2 and constant down
 * each column:
 *
 *    column 0   RGBA  32 200  64 255
 *    column 1   RGBA 224  40 192 255
 *
 * and both fetches are at a fixed (0.375, 0.5), a quarter of the way from the
 * centre of column 0 to the centre of column 1:
 *
 *    nearest   floor(0.375 * 2) = 0            ->  32 200  64 255
 *    linear    0.75 * col0 + 0.25 * col1       ->  80 160  96 255
 *
 * Those weights are 3/4 and 1/4 exactly, which every implementation can hold
 * in its subtexel fixed point, and the blends land on whole 8-bit values, so
 * both expected pixels are exact and one LSB of slack is enough. The vertical
 * coordinate is deliberately dead: v = 0.5 falls on the boundary between two
 * identical rows, where nearest may pick either and linear blends them half
 * and half, and either way the answer is the same. Only the horizontal axis
 * carries the result.
 *
 * The sample coordinates are constants, so derivatives are zero, the level of
 * detail is zero, every fetch is a magnification and magFilter is the only
 * filter that runs.
 *
 * One draw shows both answers at once. The shader fetches both bindings
 * unconditionally and then picks between the two loaded values by position --
 * glslang emits OpSelect, not a branch, so neither fetch is predicated away:
 *
 *    out_colour = gl_FragCoord.x < 32.0 ? texture(tex0, C) : texture(tex1, C);
 *
 *    left  half (x <  32)   binding 0's sampler
 *    right half (x >= 32)   binding 1's sampler
 *
 * so a single 64x64 readback carries both values, every pixel of both halves
 * is checked, and the two halves come from the same invocation set of the
 * same draw of the same pipeline. A driver that bakes binding 0's state for
 * both paints the whole frame the nearest value; one that bakes binding 1's
 * paints it all the linear value. Either way the halves come out equal, and
 * equal halves are the failure this test exists to catch.
 *
 * Four passes run that draw against four descriptor sets, same pipeline, same
 * image, same shader, differing only in which sampler each binding holds:
 *
 *    pass 0   binding 0 NEAREST   binding 1 LINEAR    left N   right L
 *    pass 1   binding 0 LINEAR    binding 1 NEAREST   left L   right N
 *    pass 2   binding 0 NEAREST   binding 1 NEAREST   both  N
 *    pass 3   binding 0 LINEAR    binding 1 LINEAR    both  L
 *
 * Passes 0 and 1 are the mixed case and are each other's mirror, so "always
 * bake binding 0" and "always bake binding 1" both fail one of them; a driver
 * cannot pass by preferring a fixed binding. Passes 2 and 3 are the uniform
 * case -- two descriptors that really do resolve to one state, which is the
 * case the relaxation is meant to specialise. They are the ones that must go
 * fast, and running them after the mixed passes also catches a state baked
 * once for the shader and then reused after the descriptor set changed: pass
 * 2 must be entirely nearest and pass 3 entirely linear, with the pipeline
 * and the shader identical in both.
 *
 * What the native driver reported when this test was written:
 *
 *   CUDAPIPE_SPEC_STATS=1 ... cpvk_sampler_two_bindings
 *   cudapipe: sampler specialisation: 0/4 fragment launches specialised
 *   (0.0%), 1 shaders, 0 with an unmatched sampler handle, 0 sampling with
 *   none matched
 *
 * -- one shader, every handle matchable, and not one launch specialised:
 * the two sampler descriptors are what the specialiser refused, which is why
 * this test passes on it today. After the relaxation the arithmetic to watch
 * is that passes 2 and 3 are the ones whose descriptors agree, so the counter
 * may rise to 2/4 and must never reach 4/4. If it does reach 4/4 the pixel
 * assertions below are what turns a silent miscompile into a failure, and if
 * it stays at 0/4 the relaxation simply did not fire.
 *
 * The target is an optimal-tiled device-local image copied back through a
 * buffer, so this runs unchanged on NVIDIA, on lavapipe and on the native
 * driver. The sampled image is linear-tiled and host-written, which all three
 * report as VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT capable.
 * SPIR-V is embedded; nothing is read from disk.
 *
 *   cc -std=c11 -Wall -Wextra -Werror \
 *      src/cudapipe/tests/cpvk_sampler_two_bindings.c \
 *      -o /tmp/cpvk-tests/cpvk_sampler_two_bindings -lvulkan
 *
 * Generated with glslangValidator (Glslang 11:16.4.0) from:
 *
 *   -- tb.vert --
 *   #version 450
 *   const vec2 base[3] = vec2[3](vec2(-1.0, -1.0),
 *                                vec2( 3.0, -1.0),
 *                                vec2(-1.0,  3.0));
 *   void main()
 *   {
 *      gl_Position = vec4(base[gl_VertexIndex], 0.0, 1.0);
 *   }
 *
 *   -- tb.frag --
 *   #version 450
 *   layout(set = 0, binding = 0) uniform sampler2D tex0;
 *   layout(set = 0, binding = 1) uniform sampler2D tex1;
 *   layout(location = 0) out vec4 out_colour;
 *   void main()
 *   {
 *      vec4 a = texture(tex0, vec2(0.375, 0.5));
 *      vec4 b = texture(tex1, vec2(0.375, 0.5));
 *      out_colour = gl_FragCoord.x < 32.0 ? a : b;
 *   }
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define W 64
#define H 64

/* The fragment shader splits the frame with a literal 32.0. */
#define SPLIT_X (W / 2)
_Static_assert(W == 64, "the embedded fragment shader hardcodes x < 32.0");

/* The sampled image, and the one coordinate both fetches read it at. */
#define TEX_W 2
#define TEX_H 2
#define SAMPLE_U 0.375f
#define SAMPLE_V 0.5f

#define NBIND 2          /* binding 0 and binding 1, two descriptors */
#define NPASS 4          /* four descriptor sets, four frames */

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
   fprintf(stderr, "%s failed: %d\n", #x, _r); return 1; } } while (0)

/* tb.vert */
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

/* tb.frag */
static const uint32_t frag_spv[] = {
   0x07230203u, 0x00010000u, 0x0008000bu, 0x00000029u, 0x00000000u, 0x00020011u,
   0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
   0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00000019u, 0x0000001bu, 0x00030010u,
   0x00000004u, 0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u,
   0x00000004u, 0x6e69616du, 0x00000000u, 0x00030005u, 0x00000009u, 0x00000061u,
   0x00040005u, 0x0000000du, 0x30786574u, 0x00000000u, 0x00030005u, 0x00000014u,
   0x00000062u, 0x00040005u, 0x00000015u, 0x31786574u, 0x00000000u, 0x00050005u,
   0x00000019u, 0x5f74756fu, 0x6f6c6f63u, 0x00007275u, 0x00060005u, 0x0000001bu,
   0x465f6c67u, 0x43676172u, 0x64726f6fu, 0x00000000u, 0x00040047u, 0x0000000du,
   0x00000021u, 0x00000000u, 0x00040047u, 0x0000000du, 0x00000022u, 0x00000000u,
   0x00040047u, 0x00000015u, 0x00000021u, 0x00000001u, 0x00040047u, 0x00000015u,
   0x00000022u, 0x00000000u, 0x00040047u, 0x00000019u, 0x0000001eu, 0x00000000u,
   0x00040047u, 0x0000001bu, 0x0000000bu, 0x0000000fu, 0x00020013u, 0x00000002u,
   0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
   0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
   0x00000007u, 0x00000007u, 0x00090019u, 0x0000000au, 0x00000006u, 0x00000001u,
   0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu,
   0x0000000bu, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000000u, 0x0000000bu,
   0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000000u, 0x00040017u, 0x0000000fu,
   0x00000006u, 0x00000002u, 0x0004002bu, 0x00000006u, 0x00000010u, 0x3ec00000u,
   0x0004002bu, 0x00000006u, 0x00000011u, 0x3f000000u, 0x0005002cu, 0x0000000fu,
   0x00000012u, 0x00000010u, 0x00000011u, 0x0004003bu, 0x0000000cu, 0x00000015u,
   0x00000000u, 0x00040020u, 0x00000018u, 0x00000003u, 0x00000007u, 0x0004003bu,
   0x00000018u, 0x00000019u, 0x00000003u, 0x00040020u, 0x0000001au, 0x00000001u,
   0x00000007u, 0x0004003bu, 0x0000001au, 0x0000001bu, 0x00000001u, 0x00040015u,
   0x0000001cu, 0x00000020u, 0x00000000u, 0x0004002bu, 0x0000001cu, 0x0000001du,
   0x00000000u, 0x00040020u, 0x0000001eu, 0x00000001u, 0x00000006u, 0x0004002bu,
   0x00000006u, 0x00000021u, 0x42000000u, 0x00020014u, 0x00000022u, 0x00040017u,
   0x00000026u, 0x00000022u, 0x00000004u, 0x00050036u, 0x00000002u, 0x00000004u,
   0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000008u,
   0x00000009u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000014u, 0x00000007u,
   0x0004003du, 0x0000000bu, 0x0000000eu, 0x0000000du, 0x00050057u, 0x00000007u,
   0x00000013u, 0x0000000eu, 0x00000012u, 0x0003003eu, 0x00000009u, 0x00000013u,
   0x0004003du, 0x0000000bu, 0x00000016u, 0x00000015u, 0x00050057u, 0x00000007u,
   0x00000017u, 0x00000016u, 0x00000012u, 0x0003003eu, 0x00000014u, 0x00000017u,
   0x00050041u, 0x0000001eu, 0x0000001fu, 0x0000001bu, 0x0000001du, 0x0004003du,
   0x00000006u, 0x00000020u, 0x0000001fu, 0x000500b8u, 0x00000022u, 0x00000023u,
   0x00000020u, 0x00000021u, 0x0004003du, 0x00000007u, 0x00000024u, 0x00000009u,
   0x0004003du, 0x00000007u, 0x00000025u, 0x00000014u, 0x00070050u, 0x00000026u,
   0x00000027u, 0x00000023u, 0x00000023u, 0x00000023u, 0x00000023u, 0x000600a9u,
   0x00000007u, 0x00000028u, 0x00000027u, 0x00000024u, 0x00000025u, 0x0003003eu,
   0x00000019u, 0x00000028u, 0x000100fdu, 0x00010038u
};

/* The two texel columns, constant down each column so that v cannot matter. */
static const unsigned char col0_rgba[4] = {  32, 200,  64, 255 };
static const unsigned char col1_rgba[4] = { 224,  40, 192, 255 };

/* NEAREST: the fetch lands in column 0 and comes back untouched. */
static const unsigned char expect_nearest[4] = {  32, 200,  64, 255 };
/* LINEAR: 3/4 of column 0 plus 1/4 of column 1, exactly. */
static const unsigned char expect_linear[4]  = {  80, 160,  96, 255 };

/* Nothing in the frame should ever be this; a missing draw shows up as it. */
static const unsigned char clear_rgba[4] = { 26, 26, 38, 255 };

/* Which sampler each binding holds in each pass: 0 = NEAREST, 1 = LINEAR. */
static const struct {
   const char *what;
   int binding[NBIND];
} passes[NPASS] = {
   { "binding 0 NEAREST, binding 1 LINEAR ", { 0, 1 } },
   { "binding 0 LINEAR,  binding 1 NEAREST", { 1, 0 } },
   { "binding 0 NEAREST, binding 1 NEAREST", { 0, 0 } },
   { "binding 0 LINEAR,  binding 1 LINEAR ", { 1, 1 } },
};

static const char *const filter_name[2] = { "NEAREST", "LINEAR " };

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

/* Every pixel of one half of one frame, against the colour that half must be. */
static void
scan_half(const unsigned char *px, int x0, int x1, const unsigned char *want,
          struct scan *s)
{
   memset(s, 0, sizeof(*s));
   s->worst_x = s->worst_y = -1;
   for (int y = 0; y < H; y++) {
      for (int x = x0; x < x1; x++) {
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
   printf("  %-36s = %3u %3u %3u %3u\n", what, px[0], px[1], px[2], px[3]);
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
    * Nothing optional is needed here: two separate bindings are plain Vulkan
    * 1.0, with no array indexing and so no shaderSampledImageArrayDynamic-
    * Indexing to ask for. maxPerStageDescriptorSampledImages is at least 16
    * on every device, and this shader wants two.
    */
   printf("  maxPerStageDescriptorSampledImages = %u (need %d)\n",
          pprops.limits.maxPerStageDescriptorSampledImages, NBIND);

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
    * The sampled image: 2x2, linear-tiled and written by the host, which all
    * three drivers report as filterable. Its rows are identical and its two
    * columns differ, so the horizontal filter is the only thing that can
    * change the answer. One image, and both bindings view it.
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
    * other field is identical, so a driver that confuses the two bindings
    * cannot blame wrap mode, border or mip state for the difference -- and,
    * just as importantly, "these two descriptors resolve to the same state"
    * is a question only the filter can answer here.
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
   VkSampler samplers[2];
   CHECK(vkCreateSampler(dev, &sci, NULL, &samplers[0]));   /* NEAREST */
   sci.magFilter = VK_FILTER_LINEAR;
   sci.minFilter = VK_FILTER_LINEAR;
   CHECK(vkCreateSampler(dev, &sci, NULL, &samplers[1]));   /* LINEAR */

   /* Two bindings, one descriptor each. Not an array: binding 0 and binding
    * 1 are separate declarations that the shader names separately. */
   VkDescriptorSetLayoutBinding dslb[NBIND] = {
      { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
      { .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
   };
   VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = NBIND, .pBindings = dslb };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

   VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                NBIND * NPASS };
   VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = NPASS, .poolSizeCount = 1, .pPoolSizes = &dps };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));

   VkDescriptorSetLayout set_layouts[NPASS];
   for (int i = 0; i < NPASS; i++)
      set_layouts[i] = dsl;
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = NPASS,
      .pSetLayouts = set_layouts };
   VkDescriptorSet dsets[NPASS];
   CHECK(vkAllocateDescriptorSets(dev, &dsai, dsets));

   /* One write per binding per pass: same view every time, sampler per the
    * table. All four sets are filled before anything is recorded, so no pass
    * can be said to have rewritten another's descriptors underneath it. */
   VkDescriptorImageInfo dii[NPASS][NBIND];
   VkWriteDescriptorSet writes[NPASS * NBIND];
   for (int p = 0; p < NPASS; p++) {
      for (int b = 0; b < NBIND; b++) {
         dii[p][b] = (VkDescriptorImageInfo){
            .sampler = samplers[passes[p].binding[b]],
            .imageView = tview,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
         writes[p * NBIND + b] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = dsets[p], .dstBinding = (uint32_t)b,
            .dstArrayElement = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &dii[p][b] };
      }
   }
   vkUpdateDescriptorSets(dev, NPASS * NBIND, writes, 0, NULL);

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

   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl };
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

   /* One pipeline. All four passes draw through it; only the set differs. */
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

   for (int pass = 0; pass < NPASS; pass++) {
      if (pass)
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                              0, NULL, 0, NULL, 1, &back_to_colour);

      begin_rendering(cmd, &ri);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      /* The only difference between the four draws. */
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                              0, 1, &dsets[pass], 0, NULL);
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

   const unsigned half_pixels = (unsigned)SPLIT_X * H;

   printf("  texture %dx%d, one image, both bindings, fetched at (%.3f, %.3f)\n",
          TEX_W, TEX_H, (double)SAMPLE_U, (double)SAMPLE_V);
   describe("texel column 0", col0_rgba);
   describe("texel column 1", col1_rgba);
   describe("NEAREST expected", expect_nearest);
   describe("LINEAR  expected", expect_linear);
   printf("  left half x < %d is binding 0, right half is binding 1\n",
          SPLIT_X);

   int fail = 0;
   unsigned char centre[NPASS][NBIND][4];

   for (int p = 0; p < NPASS; p++) {
      const unsigned char *frame = mapped + (size_t)p * frame_bytes;
      const unsigned char *want[NBIND] = {
         passes[p].binding[0] ? expect_linear : expect_nearest,
         passes[p].binding[1] ? expect_linear : expect_nearest,
      };
      struct scan s[NBIND];
      scan_half(frame, 0, SPLIT_X, want[0], &s[0]);
      scan_half(frame, SPLIT_X, W, want[1], &s[1]);

      const unsigned char *px[NBIND] = {
         pixel(frame, SPLIT_X / 2, H / 2),
         pixel(frame, SPLIT_X + SPLIT_X / 2, H / 2),
      };
      for (int b = 0; b < NBIND; b++)
         memcpy(centre[p][b], px[b], 4);

      printf("\n  pass %d: %s\n", p, passes[p].what);
      for (int b = 0; b < NBIND; b++) {
         printf("    binding %d %s got %3u %3u %3u %3u  want %3u %3u %3u %3u"
                "  (%u/%u pixels, %u clear, %u neither)\n",
                b, filter_name[passes[p].binding[b]],
                px[b][0], px[b][1], px[b][2], px[b][3],
                want[b][0], want[b][1], want[b][2], want[b][3],
                s[b].matched, half_pixels, s[b].clear, s[b].other);
         if (s[b].other)
            printf("      first stray at (%d,%d) = %3u %3u %3u %3u\n",
                   s[b].worst_x, s[b].worst_y, s[b].worst[0], s[b].worst[1],
                   s[b].worst[2], s[b].worst[3]);
      }

      for (int b = 0; b < NBIND; b++) {
         if (s[b].clear == half_pixels) {
            printf("FAIL pass %d binding %d: that half is entirely the clear "
                   "colour, so the draw did not cover it\n", p, b);
            fail = 1;
            continue;
         }
         /* Every pixel of the half, not a probe. */
         if (s[b].matched != half_pixels) {
            printf("FAIL pass %d binding %d (%s): %u of %u pixels are the "
                   "expected %u %u %u %u\n", p, b,
                   filter_name[passes[p].binding[b]], s[b].matched,
                   half_pixels, want[b][0], want[b][1], want[b][2], want[b][3]);
            fail = 1;
         }
         /*
          * And what the value is, independently of the table: NEAREST must
          * hand back a texel of the image untouched, LINEAR must hand back
          * something that is neither texel, because it is a blend of both.
          */
         if (passes[p].binding[b] == 0 && !same(px[b], col0_rgba)) {
            printf("FAIL pass %d binding %d is VK_FILTER_NEAREST but did not "
                   "return a texel of the image: it must be column 0 exactly\n",
                   p, b);
            fail = 1;
         }
         if (passes[p].binding[b] == 1 &&
             (same(px[b], col0_rgba) || same(px[b], col1_rgba))) {
            printf("FAIL pass %d binding %d is VK_FILTER_LINEAR but returned a "
                   "whole texel, not a blend: it filtered as if NEAREST\n",
                   p, b);
            fail = 1;
         }
      }

      if (passes[p].binding[0] != passes[p].binding[1]) {
         const int nb = passes[p].binding[0] ? 1 : 0;   /* the nearest half */
         const int lb = 1 - nb;                          /* the linear half */
         const unsigned char *np = px[nb], *lp = px[lb];

         /*
          * The direction, which is the part a driver cannot fake by getting
          * one half right: a quarter of the way towards column 1 raises red
          * and blue and lowers green, by far more than a rounding LSB.
          */
         if ((int)lp[0] - (int)np[0] < 8) {
            printf("FAIL pass %d: linear red %u is not above nearest red %u\n",
                   p, lp[0], np[0]);
            fail = 1;
         }
         if ((int)np[1] - (int)lp[1] < 8) {
            printf("FAIL pass %d: linear green %u is not below nearest green "
                   "%u\n", p, lp[1], np[1]);
            fail = 1;
         }
         if ((int)lp[2] - (int)np[2] < 8) {
            printf("FAIL pass %d: linear blue %u is not above nearest blue "
                   "%u\n", p, lp[2], np[2]);
            fail = 1;
         }

         /* The failure this test exists for: one state baked for both. */
         if (same(px[0], px[1])) {
            const char *which =
               same(px[0], expect_nearest) ? "the VK_FILTER_NEAREST sampler" :
               same(px[0], expect_linear)  ? "the VK_FILTER_LINEAR sampler" :
                                             "a sampler this test does not "
                                             "recognise";
            printf("FAIL pass %d: binding 0 and binding 1 hold different "
                   "sampler state but produced the same pixel, so one state "
                   "was used for both fetches: %s\n", p, which);
            fail = 1;
         }
      } else {
         /* The uniform case: both halves must agree, since both really do
          * resolve to one state. This is the case a specialiser may bake. */
         if (!same(px[0], px[1])) {
            printf("FAIL pass %d: both bindings hold the same sampler state "
                   "but the two halves differ\n", p);
            fail = 1;
         }
      }
   }

   /*
    * Across passes, with the pipeline and the shader identical throughout:
    * the same binding read through a different descriptor set must change.
    * A state baked once for the shader and reused shows up here even if
    * every individual pass looked self-consistent.
    */
   if (same(centre[0][0], centre[1][0])) {
      printf("FAIL binding 0 gave the same pixel in pass 0 (NEAREST) and "
             "pass 1 (LINEAR): the descriptor set change did not reach the "
             "sampler state\n");
      fail = 1;
   }
   if (same(centre[2][0], centre[3][0])) {
      printf("FAIL the uniform passes agree with each other: pass 2 is both "
             "bindings NEAREST and pass 3 is both LINEAR, so they cannot\n");
      fail = 1;
   }

   if (!fail)
      printf("\nPASS binding 0 and binding 1 each filtered by their own "
             "sampler, in one shader, through one pipeline, in all %d "
             "descriptor sets\n", NPASS);

   if (ppm) {
      for (int p = 0; p < NPASS; p++) {
         char path[1024];
         snprintf(path, sizeof(path), "%.*s.pass%d.ppm",
                  (int)(sizeof(path) - 32), ppm, p);
         write_ppm(path, mapped + (size_t)p * frame_bytes);
      }
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
