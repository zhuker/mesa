/* Shade-state grouping for visibility-complete opaque segments. */
#ifndef CP_SHADE_EQUIV_H
#define CP_SHADE_EQUIV_H

#include "cp_renderer.h"
#include <string.h>

/*
 * Visibility has already resolved clipping, scissor, culling and the depth
 * comparison.  Group only on state that the later interpolate/FS/writeback
 * chain still reads.  Keep this separate from cp_batch_key: that key also
 * protects geometry and visibility batching and must remain unchanged.
 */
static inline bool
cp_opaque_shade_equivalent(const struct cp_pass_seg *a,
                           const struct cp_pass_seg *b)
{
   const struct cp_draw_batch *ab = &a->batch;
   const struct cp_draw_batch *bb = &b->batch;
   const struct cp_fb_desc *af = &ab->scope.fb;
   const struct cp_fb_desc *bf = &bb->scope.fb;

   bool a_depth_invert = ab->state.depth.depth_enabled &&
      (ab->state.depth.depth_func == CP_FUNC_GREATER ||
       ab->state.depth.depth_func == CP_FUNC_GEQUAL);
   bool b_depth_invert = bb->state.depth.depth_enabled &&
      (bb->state.depth.depth_func == CP_FUNC_GREATER ||
       bb->state.depth.depth_func == CP_FUNC_GEQUAL);

   return ab->state.vs == bb->state.vs &&
          ab->state.fs == bb->state.fs &&
          ab->state.num_fs_ubos == bb->state.num_fs_ubos &&
          ab->info.mode == bb->info.mode &&
          ab->ndraws == bb->ndraws &&
          !memcmp(ab->fs_ubos, bb->fs_ubos,
                  (size_t)ab->ndraws * CP_ARG_UBO_STRIDE * sizeof(uint64_t)) &&
          ab->state.raster.front_ccw == bb->state.raster.front_ccw &&
          ab->state.depth.depth_writemask == bb->state.depth.depth_writemask &&
          a_depth_invert == b_depth_invert &&
          !memcmp(&ab->state.blend, &bb->state.blend,
                  sizeof(ab->state.blend)) &&
          !memcmp(&a->rast.depth_scale, &b->rast.depth_scale,
                  6 * sizeof(float)) &&
          ab->key.sampler_table == bb->key.sampler_table &&
          ab->key.num_samplers == bb->key.num_samplers &&
          ab->key.visbuf == bb->key.visbuf &&
          ab->key.depthbuf == bb->key.depthbuf &&
          ab->key.cbuf_texture == bb->key.cbuf_texture &&
          ab->key.zs_texture == bb->key.zs_texture &&
          ab->key.color_data == bb->key.color_data &&
          af->width == bf->width && af->height == bf->height &&
          af->nr_cbufs == bf->nr_cbufs && af->color == bf->color &&
          af->color_encoding == bf->color_encoding &&
          af->color_sample_stride == bf->color_sample_stride &&
          af->texture_cookie == bf->texture_cookie &&
          af->has_zs == bf->has_zs &&
          ab->scope.attachment_samples == bb->scope.attachment_samples;
}

#endif /* CP_SHADE_EQUIV_H */
