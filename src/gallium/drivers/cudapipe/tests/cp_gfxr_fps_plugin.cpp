/*
 * Per-frame timing for a capture that never presents.
 *
 * gfxrecon's own FPS measurement counts vkQueuePresentKHR and this capture has
 * none, so --measurement-file writes nothing at all. The replay event plugin
 * API reports queue submits instead, with gfxrecon's own timestamps, and this
 * application submits twice per frame — so submit boundaries are frame
 * boundaries. Driver-agnostic: the same plugin measures cudapipe, llvmpipe and
 * NVIDIA without any of them knowing.
 */
#include "gfxr/replay_event_plugin.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct { GfxrReplayPluginV1 base; FILE *f; } plug;

static void destroy(GfxrReplayPluginV1 *self)
{
   plug *p = (plug *)self;
   if (p->f) fclose(p->f);
   free(p);
}

static GfxrReplayPluginResult on_event(GfxrReplayPluginV1 *self,
                                       const GfxrReplayEventHeader *e)
{
   plug *p = (plug *)self;
   if (e->type == GFXR_REPLAY_EVENT_QUEUE_SUBMIT_END && p->f) {
      const GfxrReplayQueueSubmitEndEvent *s = (const GfxrReplayQueueSubmitEndEvent *)e;
      fprintf(p->f, "%llu %llu\n", (unsigned long long)e->timestamp_ns,
              (unsigned long long)s->submit_index);
   }
   return GFXR_REPLAY_PLUGIN_RESULT_OK;
}

extern "C" GFXR_REPLAY_PLUGIN_EXPORT GfxrReplayPluginV1 *
gfxrCreateReplayPluginV1(const GfxrReplayPluginCreateInfo *ci)
{
   plug *p = (plug *)calloc(1, sizeof *p);
   if (!p) return NULL;
   p->base.abi_version = GFXR_REPLAY_PLUGIN_ABI_VERSION;
   p->base.struct_size = sizeof(GfxrReplayPluginV1);
   p->base.destroy = destroy;
   p->base.on_event = on_event;
   const char *out = (ci && ci->plugin_params && *ci->plugin_params)
                   ? ci->plugin_params : "submits.txt";
   p->f = fopen(out, "w");
   return &p->base;
}
