/* Production classifier, actual shader_info. Pure CPU; no CUDA linkage. */
#include "cp_shader_clip_metadata.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
   unsigned cases = 0;
   for (unsigned reasons = 0; reasons < 8; reasons++) {
      for (unsigned outputs = 0; outputs < 3; outputs++) {
         struct shader_info info = { .stage = MESA_SHADER_FRAGMENT };
         if (outputs == 1)
            info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
         if (outputs == 2)
            info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_STENCIL);
         if (reasons & CP_CLIP_FS_DEPTH)
            info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
         if (reasons & CP_CLIP_FS_SAMPLE_MASK)
            info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
         info.fs.early_fragment_tests = !!(reasons & CP_CLIP_FS_EARLY_TESTS);
         assert(cp_shader_clip_exclusions(&info) == reasons);
         assert(cp_shader_clip_excluded(cp_shader_clip_exclusions(&info)) ==
                (reasons != 0));
         cases++;
         /* Deliberately preserve overlapping union/output bits. */
         for (unsigned stage = 0; stage < MESA_SHADER_STAGES; stage++) {
            if (stage == MESA_SHADER_FRAGMENT)
               continue;
            info.stage = stage;
            assert(cp_shader_clip_exclusions(&info) == 0);
            cases++;
         }
      }
   }
   puts("shader clip metadata: PASS (all reasons, combinations, unrelated outputs, non-FS stages)");
   printf("%u CPU classifier cases; not compilation/cache validation\n", cases);
   return 0;
}
