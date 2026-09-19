/* Host-only ordered-clip scope exclusions, not shader execution support. */
#ifndef CP_SHADER_CLIP_METADATA_H
#define CP_SHADER_CLIP_METADATA_H
#include "compiler/shader_info.h"
#include "util/macros.h"

enum cp_clip_fs_exclusion {
   CP_CLIP_FS_DEPTH = 1 << 0,
   CP_CLIP_FS_SAMPLE_MASK = 1 << 1,
   CP_CLIP_FS_EARLY_TESTS = 1 << 2,
};

/* Fragment callers supply gathered info, before backend lowering. Do not
 * interpret another stage's output locations or stage-specific union as FS. */
static inline uint8_t
cp_shader_clip_exclusions(const struct shader_info *info)
{
   if (info->stage != MESA_SHADER_FRAGMENT)
      return 0;
   return ((info->outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)) ?
           CP_CLIP_FS_DEPTH : 0) |
          ((info->outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)) ?
           CP_CLIP_FS_SAMPLE_MASK : 0) |
          (info->fs.early_fragment_tests ? CP_CLIP_FS_EARLY_TESTS : 0);
}

static inline bool
cp_shader_clip_excluded(uint8_t reasons)
{
   return reasons != 0;
}
#endif
