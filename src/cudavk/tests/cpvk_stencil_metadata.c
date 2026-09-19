/* CPU-only test of the exact metadata classifier used by pipeline creation. */
#include "../cpvk_stencil_metadata.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .front = { .compareOp = VK_COMPARE_OP_ALWAYS },
      .back = { .compareOp = VK_COMPARE_OP_ALWAYS },
   };
   assert(!cpvk_stencil_may_observe(&ds, NULL));
   ds.stencilTestEnable = VK_TRUE;
   assert(cpvk_stencil_may_observe(&ds, NULL)); /* even ALWAYS/KEEP */
   ds.back.compareOp = VK_COMPARE_OP_NEVER;
   assert(cpvk_stencil_may_observe(&ds, NULL));
   ds.stencilTestEnable = VK_FALSE;
   assert(!cpvk_stencil_may_observe(&ds, NULL)); /* ignored face state */
   assert(cpvk_stencil_may_observe(NULL, NULL));
   ds.pNext = &ds;
   assert(cpvk_stencil_may_observe(&ds, NULL));
   ds.pNext = NULL;
   VkDynamicState states[] = {
      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
      VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
      VK_DYNAMIC_STATE_STENCIL_REFERENCE,
      VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
      VK_DYNAMIC_STATE_STENCIL_OP,
   };
   VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1,
   };
   for (unsigned i = 0; i < sizeof(states) / sizeof(states[0]); i++) {
      dyn.pDynamicStates = &states[i];
      assert(cpvk_stencil_may_observe(&ds, &dyn));
   }
   VkDynamicState viewport = VK_DYNAMIC_STATE_VIEWPORT;
   dyn.pDynamicStates = &viewport;
   assert(!cpvk_stencil_may_observe(&ds, &dyn));
   dyn.pDynamicStates = NULL;
   assert(cpvk_stencil_may_observe(&ds, &dyn));
   dyn.dynamicStateCount = 0;
   assert(!cpvk_stencil_may_observe(&ds, &dyn));
   dyn.pNext = &dyn;
   assert(cpvk_stencil_may_observe(&ds, &dyn));
   puts("stencil metadata: PASS (static, dynamic, missing and unknown state)");
   return 0;
}
