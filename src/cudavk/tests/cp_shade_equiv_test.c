#include "cp_renderer.h"
#include "cp_shade_equiv.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void
expect_boundary(struct cp_pass_seg *a, struct cp_pass_seg *b)
{
   assert(!cp_opaque_shade_equivalent(a, b));
   memcpy(b, a, sizeof(*b));
}

int
main(void)
{
   struct cp_pass_seg *a = calloc(1, sizeof(*a));
   struct cp_pass_seg *b = calloc(1, sizeof(*b));
   assert(a && b);

   a->batch.state.vs = (void *)1;
   a->batch.state.fs = (void *)2;
   a->batch.state.num_fs_ubos = 1;
   a->batch.info.mode = MESA_PRIM_TRIANGLES;
   a->batch.scope.fb.width = 64;
   a->batch.scope.fb.height = 32;
   a->batch.scope.fb.nr_cbufs = 1;
   a->batch.scope.fb.color = (void *)3;
   a->batch.scope.fb.color_encoding = 4;
   a->batch.scope.fb.color_sample_stride = 256;
   a->batch.scope.fb.texture_cookie = 5;
   a->batch.scope.attachment_samples = 1;
   a->batch.key.cbuf_texture = (void *)3;
   a->batch.key.color_data = (void *)3;
   a->batch.key.visbuf = 6;
   a->batch.key.depthbuf = 7;
   a->batch.key.sampler_table = 8;
   a->batch.key.num_samplers = 1;
   a->rast.depth_scale = 1.0f;
   a->rast.depth_translate = 0.0f;
   a->rast.vp_scale_x = 32.0f;
   a->rast.vp_scale_y = 16.0f;
   a->rast.vp_trans_x = 32.0f;
   a->rast.vp_trans_y = 16.0f;
   memcpy(b, a, sizeof(*b));
   assert(cp_opaque_shade_equivalent(a, b));

   /* Visibility-only differences must not split a completed run. */
   b->batch.key.scissor.minx = 9;
   b->batch.state.raster.cull_face = CP_FACE_BACK;
   b->batch.state.raster.scissor = true;
   b->batch.key.state[0] = 0x55;
   b->batch.state.depth.depth_enabled = true;
   b->batch.state.depth.depth_func = CP_FUNC_LEQUAL;
   assert(cp_opaque_shade_equivalent(a, b));
   memcpy(b, a, sizeof(*b));

   b->batch.state.vs = (void *)10;
   expect_boundary(a, b);
   b->batch.state.fs = (void *)11;
   expect_boundary(a, b);
   b->batch.state.num_fs_ubos++;
   expect_boundary(a, b);
   b->batch.info.mode = MESA_PRIM_POINTS;
   expect_boundary(a, b);
   a->batch.ndraws = 1;
   memcpy(b, a, sizeof(*b));
   b->batch.fs_ubos[0] = 12;
   expect_boundary(a, b);
   b->rast.vp_scale_x += 1.0f;
   expect_boundary(a, b);
   b->rast.depth_translate = 0.25f;
   expect_boundary(a, b);
   b->batch.state.raster.front_ccw = true;
   expect_boundary(a, b);
   b->batch.key.sampler_table++;
   expect_boundary(a, b);
   b->batch.scope.fb.texture_cookie++;
   expect_boundary(a, b);
   b->batch.scope.fb.color = (void *)9;
   expect_boundary(a, b);

   /* The exact compare function is visibility-only; only the writeback's
    * derived depth-key direction remains shade state. */
   a->batch.state.depth.depth_enabled = true;
   a->batch.state.depth.depth_func = CP_FUNC_LESS;
   memcpy(b, a, sizeof(*b));
   b->batch.state.depth.depth_func = CP_FUNC_LEQUAL;
   assert(cp_opaque_shade_equivalent(a, b));
   b->batch.state.depth.depth_func = CP_FUNC_GREATER;
   expect_boundary(a, b);

   free(a);
   free(b);
   return 0;
}
