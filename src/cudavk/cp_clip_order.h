/* Host-only stable clip admission. No raster key or GPU ABI changes. */
#ifndef CP_CLIP_ORDER_H
#define CP_CLIP_ORDER_H
#include <stdbool.h>

struct cp_clip_order_policy {
   bool blend, reads_const_bufs;
   unsigned batch_draws;
   bool depth_test, depth_write, inclusive;
   bool unsupported_raster_order, shader_side_effects, discard;
};

/* Metadata is not a legacy merge key. Accumulate every member's exclusion. */
static inline bool
cp_clip_order_accumulate_exclusion(bool retained, bool incoming)
{
   return retained || incoming;
}

static inline bool
cp_clip_order_stable(bool enabled, const struct cp_clip_order_policy *p)
{
   /* Preserve the old condition exactly, even for excluded new cases. */
   return p->blend || (p->batch_draws > 1 && p->reads_const_bufs) ||
          (enabled && p->depth_test && p->depth_write && p->inclusive &&
           !p->unsupported_raster_order && !p->shader_side_effects && !p->discard);
}
#endif
