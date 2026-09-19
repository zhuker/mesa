/* CPU-only production admission truth table. No CUDA execution. */
#include "cp_clip_order.h"
#include "cpvk_stencil_metadata.h"
#include <assert.h>
#include <stdio.h>
/* Feed actual pipeline declarations through the production classifier and
 * accumulation helper, then through production admission. Static LE/GE with
 * test/write enabled must not hide ANY unsupported dynamic depth declaration. */
static unsigned dynamic_depth_cases(void)
{
   const VkDynamicState states[] = {
      VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
      VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
      VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
      VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
      VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
      VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
   };
   unsigned cases = 0;
   for (unsigned op = 0; op < 2; op++) {
      VkPipelineDepthStencilStateCreateInfo ds = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = VK_TRUE,
         .depthWriteEnable = VK_TRUE,
         .depthCompareOp = op ? VK_COMPARE_OP_GREATER_OR_EQUAL :
                                VK_COMPARE_OP_LESS_OR_EQUAL,
      };
      bool static_exclusion = cpvk_unsupported_raster_order(&ds, NULL);
      assert(!static_exclusion);
      for (unsigned i = 0; i < sizeof(states) / sizeof(states[0]); i++) {
         VkPipelineDynamicStateCreateInfo dyn = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 1, .pDynamicStates = &states[i],
         };
         bool exclusion = cpvk_unsupported_raster_order(&ds, &dyn);
         assert(exclusion);
         /* Put the excluded draw at every position of one-/multi-draw
          * batches. Test both legacy stable reasons independently as well. */
         for (unsigned draws = 1; draws <= 4; draws++)
            for (unsigned at = 0; at < draws; at++)
               for (unsigned legacy = 0; legacy < 4; legacy++) {
                  struct cp_clip_order_policy p = {
                     .blend = legacy & 1, .reads_const_bufs = legacy & 2,
                     .batch_draws = draws,
                     .depth_test = ds.depthTestEnable,
                     .depth_write = ds.depthWriteEnable,
                     .inclusive = ds.depthCompareOp == VK_COMPARE_OP_LESS_OR_EQUAL ||
                                  ds.depthCompareOp == VK_COMPARE_OP_GREATER_OR_EQUAL,
                  };
                  bool old = p.blend || (draws > 1 && p.reads_const_bufs);
                  assert(cp_clip_order_stable(true, &p));
                  for (unsigned member = 0; member < draws; member++)
                     p.unsupported_raster_order = cp_clip_order_accumulate_exclusion(
                        p.unsupported_raster_order,
                        member == at ? exclusion : static_exclusion);
                  assert(p.unsupported_raster_order);
                  assert(cp_clip_order_stable(true, &p) == old);
                  assert(cp_clip_order_stable(false, &p) == old);
                  cases++;
               }
      }
   }
   return cases;
}

int main(void)
{
   unsigned cases = 0;
   for (unsigned bits = 0; bits < 256; bits++)
      for (unsigned draws = 1; draws <= 3; draws++) {
         struct cp_clip_order_policy p = {
            .blend = bits & 1, .reads_const_bufs = bits & 2,
            .batch_draws = draws, .depth_test = bits & 4,
            .depth_write = bits & 8, .inclusive = bits & 16,
            .unsupported_raster_order = bits & 32,
            .shader_side_effects = bits & 64, .discard = bits & 128,
         };
         bool old = p.blend || (draws > 1 && p.reads_const_bufs);
         assert(cp_clip_order_stable(false, &p) == old);
         bool extra = p.depth_test && p.depth_write && p.inclusive &&
                      !p.unsupported_raster_order && !p.shader_side_effects && !p.discard;
         assert(cp_clip_order_stable(true, &p) == (old || extra));
         cases++;
      }
   printf("clip policy CPU: PASS %u truth-table cases, %u dynamic depth cases, flag off/on\n",
          cases, dynamic_depth_cases());
}
