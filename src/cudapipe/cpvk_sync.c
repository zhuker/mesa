/*
 * Synchronization.
 *
 * Every submit here drains the stream before returning, so by the time
 * anything can be signalled the work has already run and a wait has nothing
 * to wait for. That makes an always-signalled sync type *correct* rather than
 * a stub -- but it is correct only for as long as that remains true, which is
 * why it says so here: the moment submits stop draining, this becomes a
 * CUevent recorded on the stream, and the Gallium-hosted driver's cp_fence
 * already shows the shape (a refcounted event, because one submit's fence
 * lands in several vk_syncs, and a bare destroy double-freed under the first
 * triangle).
 */

#include "cpvk_private.h"

#include "vk_sync.h"

static VkResult
cpvk_sync_init(struct vk_device *device, struct vk_sync *sync,
               uint64_t initial_value)
{
   return VK_SUCCESS;
}

static void
cpvk_sync_finish(struct vk_device *device, struct vk_sync *sync)
{
}

static VkResult
cpvk_sync_signal(struct vk_device *device, struct vk_sync *sync,
                 uint64_t value)
{
   return VK_SUCCESS;
}

static VkResult
cpvk_sync_reset(struct vk_device *device, struct vk_sync *sync)
{
   return VK_SUCCESS;
}

static VkResult
cpvk_sync_wait(struct vk_device *device, struct vk_sync *sync,
               uint64_t wait_value, enum vk_sync_wait_flags wait_flags,
               uint64_t abs_timeout_ns)
{
   struct cpvk_device *dev = container_of(device, struct cpvk_device, vk);

   /* Cheap insurance rather than a bare return: if a future submit stops
    * draining, this still means what it says. */
   cuCtxSetCurrent(dev->cu_ctx);
   cuStreamSynchronize(dev->stream);
   return VK_SUCCESS;
}

const struct vk_sync_type cpvk_sync_type = {
   .size = sizeof(struct vk_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL | VK_SYNC_FEATURE_WAIT_PENDING,
   .init = cpvk_sync_init,
   .finish = cpvk_sync_finish,
   .signal = cpvk_sync_signal,
   .reset = cpvk_sync_reset,
   .wait = cpvk_sync_wait,
};

const struct vk_sync_type *const cpvk_sync_types[] = {
   &cpvk_sync_type,
   NULL,
};
