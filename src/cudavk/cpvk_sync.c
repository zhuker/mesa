/*
 * Binary Vulkan synchronization for the currently synchronous native queue.
 *
 * Queue submission still drains renderer.stream before publishing its signal
 * syncs.  That makes completion state a host condition today, but it must be
 * real state: a fresh/reset fence is unsignalled, a zero-time wait times out,
 * and a submitted fence/semaphore becomes signalled only after the drain.
 * The state object is intentionally suitable for replacing the boolean with a
 * refcounted CUDA completion event when submissions become asynchronous.
 */

#include "cpvk_private.h"

#include "vk_sync.h"
#include "util/timespec.h"

struct cpvk_sync {
   struct vk_sync vk;
   mtx_t lock;
   cnd_t changed;
   bool signaled;
};

static struct cpvk_sync *
cpvk_sync_from_vk(struct vk_sync *vk_sync)
{
   return container_of(vk_sync, struct cpvk_sync, vk);
}

static VkResult
cpvk_sync_init(struct vk_device *device, struct vk_sync *vk_sync,
               uint64_t initial_value)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   if (mtx_init(&sync->lock, mtx_plain) != thrd_success)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   if (cnd_init(&sync->changed) != thrd_success) {
      mtx_destroy(&sync->lock);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   sync->signaled = initial_value != 0;
   return VK_SUCCESS;
}

static void
cpvk_sync_finish(struct vk_device *device, struct vk_sync *vk_sync)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   cnd_destroy(&sync->changed);
   mtx_destroy(&sync->lock);
}

static VkResult
cpvk_sync_signal(struct vk_device *device, struct vk_sync *vk_sync,
                 uint64_t value)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   mtx_lock(&sync->lock);
   sync->signaled = true;
   cnd_broadcast(&sync->changed);
   mtx_unlock(&sync->lock);
   return VK_SUCCESS;
}

static VkResult
cpvk_sync_reset(struct vk_device *device, struct vk_sync *vk_sync)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   mtx_lock(&sync->lock);
   sync->signaled = false;
   cnd_broadcast(&sync->changed);
   mtx_unlock(&sync->lock);
   return VK_SUCCESS;
}

static VkResult
cpvk_sync_wait(struct vk_device *device, struct vk_sync *vk_sync,
               uint64_t wait_value, enum vk_sync_wait_flags wait_flags,
               uint64_t abs_timeout_ns)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   assert(!(wait_flags & VK_SYNC_WAIT_ANY));

   mtx_lock(&sync->lock);
   uint64_t now_ns = os_time_get_nano();
   while (!sync->signaled) {
      if (now_ns >= abs_timeout_ns) {
         mtx_unlock(&sync->lock);
         return VK_TIMEOUT;
      }

      int ret;
      if (abs_timeout_ns >= INT64_MAX) {
         ret = cnd_wait(&sync->changed, &sync->lock);
      } else {
         uint64_t rel_timeout_ns = abs_timeout_ns - now_ns;
         struct timespec now_ts, abs_timeout_ts;
         timespec_get(&now_ts, TIME_UTC);
         if (timespec_add_nsec(&abs_timeout_ts, &now_ts, rel_timeout_ns))
            ret = cnd_wait(&sync->changed, &sync->lock);
         else
            ret = cnd_timedwait(&sync->changed, &sync->lock, &abs_timeout_ts);
      }
      if (ret == thrd_error) {
         mtx_unlock(&sync->lock);
         return vk_errorf(device, VK_ERROR_UNKNOWN, "sync condition wait failed");
      }
      now_ns = os_time_get_nano();
   }
   mtx_unlock(&sync->lock);
   return VK_SUCCESS;
}

const struct vk_sync_type cpvk_sync_type = {
   .size = sizeof(struct cpvk_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL,
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
