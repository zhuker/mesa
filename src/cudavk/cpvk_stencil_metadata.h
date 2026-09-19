/* Host-only raster-order exclusions; no dynamic depth or stencil execution. */
#ifndef CPVK_STENCIL_METADATA_H
#define CPVK_STENCIL_METADATA_H

#include <stdbool.h>
#include <vulkan/vulkan_core.h>

/* False proves static depth ordering controls and inert stencil, not execution support.
 * Dynamic depth and stencil are unknown: the native command path does not
 * track their setters. Never infer eligibility from ignored static values. */
static inline bool
cpvk_unsupported_raster_order(const VkPipelineDepthStencilStateCreateInfo *ds,
                              const VkPipelineDynamicStateCreateInfo *dyn)
{
   if (dyn) {
      if (dyn->pNext || (dyn->dynamicStateCount && !dyn->pDynamicStates))
         return true;
      for (uint32_t i = 0; i < dyn->dynamicStateCount; i++) {
         switch (dyn->pDynamicStates[i]) {
         /* EXT aliases have the same enum values; no duplicate cases. */
         case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE:
         case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE:
         case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP:
         case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK:
         case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK:
         case VK_DYNAMIC_STATE_STENCIL_REFERENCE:
         case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE:
         case VK_DYNAMIC_STATE_STENCIL_OP:
            return true;
         default:
            break;
         }
      }
   }
   /* Absent or extended state is not a proof. This can over-exclude pipelines
    * which do not use a stencil attachment; it cannot admit an unknown one. */
   if (!ds || ds->pNext)
      return true;
   /* Even ALWAYS/no-write enabled stencil stays excluded. Do not make the
    * new path depend on a semantic shortcut for an unimplemented feature. */
   return ds->stencilTestEnable != VK_FALSE;
}

#endif
