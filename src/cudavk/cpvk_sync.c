/*
 * Vulkan synchronization for the native queue: binary fences and semaphores,
 * and timeline semaphores, on one vk_sync type.
 *
 * Queue submission does not drain renderer.stream: it records a CUDA event and
 * the completion worker publishes the signals behind it.  A sync therefore
 * carries three things.
 *
 *   value    the host-visible counter a fence query, vkGetSemaphoreCounterValue
 *            and a CPU wait read.  Real state: a fresh or reset fence is
 *            unsignalled, a zero-time wait times out, and a submitted
 *            fence/semaphore advances only when the worker has seen its event.
 *            A binary sync uses 0 and 1 and nothing else.
 *   pending  the value work already submitted will reach.  This is what
 *            VK_SYNC_WAIT_PENDING answers, and it is the whole reason the
 *            runtime can put this device in ASSISTED timeline mode and support
 *            wait-before-signal on top of a driver that cannot.
 *   event    the same completion expressed as device state, refcounted and
 *            shared with the pending submit that recorded it.  It is what lets
 *            a later submit wait on the GPU instead of on the application
 *            thread.
 *
 * The event is refcounted because it has more than one owner and no single
 * point of death: the pending-submit record drops its reference once the worker
 * has synchronized on it, a sync holds one until it is reset, re-armed or
 * destroyed, and the last one out destroys it.
 *
 * `epoch` is what keeps an asynchronous signal honest.  A binary semaphore
 * waited on by a submit is reset by the runtime on the spot, while the submit
 * that will signal it may still be running; publishing that signal afterwards
 * would re-signal a semaphore whose signal has already been consumed.  Each
 * arming records the epoch it armed at, a reset or a move bumps it, and a
 * completion whose epoch no longer matches is dropped.
 *
 * Two refusals are deliberate and are recorded rather than worked around.
 *
 * No external semaphore of any kind.  CUDA 12.8 can import one but cannot
 * create or export one, and that is measured, not inferred: of every fd this
 * driver could plausibly produce -- DRM binary and timeline syncobjs, a
 * syncobj exported with the TIMELINE flag, a Linux sync_file -- every one is
 * refused by cuImportExternalSemaphore with CUDA_ERROR_UNKNOWN, exactly as a
 * plain file, an eventfd and /dev/null are.  The only fd it accepts is an
 * NVIDIA RM object, which CUDA has no call to make.  So the timeline
 * semaphores here are for Vulkan, not for handing a frame to CUDA; the
 * supported mechanism for that is still the host handshake.  See
 * docs/cudavk/CUDA_INTEROP.md and TODO.md item 11.
 *
 * And no VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL.  A CUDA stream cannot be told to
 * wait for a value nothing has promised yet, so claiming it would put the
 * runtime in NATIVE timeline mode and hand it a submit-and-forget guarantee
 * this driver cannot keep.  WAIT_PENDING is claimed instead, because it is
 * true, and the runtime's ASSISTED mode does the holding.  See the features
 * comment on cpvk_sync_type below.
 */

#include "cpvk_private.h"

#include "vk_sync.h"
#include "util/timespec.h"

/* The enqueue epoch only, not the census interception: this file's one stream
 * operation has to be visible to CUDAVK_VS_LANE's "nothing was issued in
 * between" check, because a semaphore wait enqueued between two batches is
 * exactly the ordering a run-ahead vertex shader must not step over. */
#include "cp_devop.h"

struct cpvk_sync {
   struct vk_sync vk;
   mtx_t lock;
   cnd_t changed;
   uint64_t value;
   uint64_t pending;
   uint64_t epoch;
   /* The completion event that will take `value` to `pending`, or NULL when
    * nothing was ever recorded into this sync: a host signal, a reset, or a
    * fence created signalled. */
   struct cpvk_cuevent *event;
};

static struct cpvk_sync *
cpvk_sync_from_vk(struct vk_sync *vk_sync)
{
   return container_of(vk_sync, struct cpvk_sync, vk);
}

static struct cpvk_device *
cpvk_device_from_vk(struct vk_device *device)
{
   return container_of(device, struct cpvk_device, vk);
}

/*
 * The value a wait is actually asking for. A binary sync is a timeline of two
 * points, and every wait on it is a wait for point one; the Vulkan wait value
 * for a binary sync is required to be zero and carries no information.
 */
static uint64_t
cpvk_sync_target(const struct cpvk_sync *sync, uint64_t wait_value)
{
   return (sync->vk.flags & VK_SYNC_IS_TIMELINE) ? wait_value : 1;
}

struct cpvk_cuevent *
cpvk_cuevent_create(CUevent event)
{
   struct cpvk_cuevent *ev = malloc(sizeof(*ev));
   if (!ev)
      return NULL;
   ev->event = event;
   atomic_init(&ev->refcnt, 1);
   return ev;
}

struct cpvk_cuevent *
cpvk_cuevent_ref(struct cpvk_cuevent *ev)
{
   if (ev)
      atomic_fetch_add_explicit(&ev->refcnt, 1, memory_order_relaxed);
   return ev;
}

void
cpvk_cuevent_unref(struct cpvk_device *dev, struct cpvk_cuevent *ev)
{
   if (!ev)
      return;
   if (atomic_fetch_sub_explicit(&ev->refcnt, 1, memory_order_acq_rel) != 1)
      return;
   /* The last reference destroys it, on whichever thread got here, so the
    * context is made current rather than assumed. */
   CPVK_CTX_SCOPE(dev);
   cuEventDestroy(ev->event);
   free(ev);
}

/*
 * Arm a sync for a submit that will signal it: promise the value, take the
 * submit's completion event, and return the epoch this promise was made at.
 *
 * Called on the submitting thread before the worker can publish anything, so
 * that a following submit finds the event while the sync is still unsignalled,
 * and so that VK_SYNC_WAIT_PENDING is true from the moment the work is issued.
 */
uint64_t
cpvk_sync_arm(struct vk_device *device, struct vk_sync *vk_sync,
              uint64_t value, struct cpvk_cuevent *ev)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   uint64_t target = cpvk_sync_target(sync, value);

   mtx_lock(&sync->lock);
   struct cpvk_cuevent *old = sync->event;
   sync->event = cpvk_cuevent_ref(ev);
   if (target > sync->pending)
      sync->pending = target;
   uint64_t epoch = sync->epoch;
   cnd_broadcast(&sync->changed);
   mtx_unlock(&sync->lock);

   /* Outside the lock: the last unref reaches CUDA. */
   cpvk_cuevent_unref(cpvk_device_from_vk(device), old);
   return epoch;
}

/*
 * Publish a completed submit's signal, from the completion worker.
 *
 * Not vk_sync_signal(): a signal that arrives after the sync was reset or
 * moved belongs to a promise that no longer exists and must be dropped, which
 * is what the epoch decides.
 */
void
cpvk_sync_signal_completion(struct vk_device *device, struct vk_sync *vk_sync,
                            uint64_t value, uint64_t epoch)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   uint64_t target = cpvk_sync_target(sync, value);

   mtx_lock(&sync->lock);
   if (sync->epoch == epoch && target > sync->value) {
      sync->value = target;
      if (target > sync->pending)
         sync->pending = target;
      cnd_broadcast(&sync->changed);
   }
   mtx_unlock(&sync->lock);
}

/*
 * Make the stream wait for a sync's completion, instead of the calling thread.
 *
 * Returns true when the wait is satisfied on the device -- either the event was
 * enqueued on the stream, or the sync has already reached the value and there
 * is nothing to wait for.  Returns false when the sync carries no device work
 * that would reach the value, which is the case a host wait still has to cover:
 * a semaphore whose signalling submit has not been issued yet, or one signalled
 * from another thread by vkSignalSemaphore.
 *
 * Waiting on the newest event when an older one would do is deliberate and
 * safe: every submit records its event on the same stream, so a later event
 * completing implies every earlier one has.
 */
bool
cpvk_sync_gpu_wait(struct vk_device *device, struct vk_sync *vk_sync,
                   uint64_t wait_value, CUstream stream)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);

   mtx_lock(&sync->lock);
   uint64_t target = cpvk_sync_target(sync, wait_value);
   if (target <= sync->value) {
      mtx_unlock(&sync->lock);
      return true;
   }
   struct cpvk_cuevent *ev =
      target <= sync->pending ? cpvk_cuevent_ref(sync->event) : NULL;
   mtx_unlock(&sync->lock);

   if (!ev)
      return false;

   /* The reference above is what makes this safe outside the lock: the sync
    * may be reset or destroyed by another thread while the event is being
    * enqueued, and the event still exists until this unref. */
   cp_devop_note();
   CUresult r = cuStreamWaitEvent(stream, ev->event, CU_EVENT_WAIT_DEFAULT);
   cpvk_cuevent_unref(cpvk_device_from_vk(device), ev);
   return r == CUDA_SUCCESS;
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
   /* A binary sync's initial value is a flag; a timeline's is its first
    * point, and a timeline may legitimately start at zero. */
   sync->value = (vk_sync->flags & VK_SYNC_IS_TIMELINE) ? initial_value
                                                        : (initial_value != 0);
   sync->pending = sync->value;
   sync->epoch = 0;
   sync->event = NULL;
   return VK_SUCCESS;
}

static void
cpvk_sync_finish(struct vk_device *device, struct vk_sync *vk_sync)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   cpvk_cuevent_unref(cpvk_device_from_vk(device), sync->event);
   sync->event = NULL;
   cnd_destroy(&sync->changed);
   mtx_destroy(&sync->lock);
}

static VkResult
cpvk_sync_signal(struct vk_device *device, struct vk_sync *vk_sync,
                 uint64_t value)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   uint64_t target = cpvk_sync_target(sync, value);

   mtx_lock(&sync->lock);
   /*
    * Monotone, always. Signalling a timeline to a value at or below the
    * current one is invalid usage (VUID-VkSemaphoreSignalInfo-value-03258)
    * and the one thing a timeline must never do is go backwards, so the
    * lower value is dropped rather than obeyed.
    */
   if (target > sync->value) {
      sync->value = target;
      if (target > sync->pending)
         sync->pending = target;
      cnd_broadcast(&sync->changed);
   }
   mtx_unlock(&sync->lock);
   return VK_SUCCESS;
}

static VkResult
cpvk_sync_get_value(struct vk_device *device, struct vk_sync *vk_sync,
                    uint64_t *value)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   mtx_lock(&sync->lock);
   *value = sync->value;
   mtx_unlock(&sync->lock);
   return VK_SUCCESS;
}

static VkResult
cpvk_sync_reset(struct vk_device *device, struct vk_sync *vk_sync)
{
   struct cpvk_sync *sync = cpvk_sync_from_vk(vk_sync);
   assert(!(vk_sync->flags & VK_SYNC_IS_TIMELINE));

   mtx_lock(&sync->lock);
   sync->value = 0;
   sync->pending = 0;
   /* A reset sync has no completion to point at: the event it carried
    * described work whose signal has just been thrown away, and the epoch
    * bump is what stops that signal being published later. */
   struct cpvk_cuevent *old = sync->event;
   sync->event = NULL;
   sync->epoch++;
   cnd_broadcast(&sync->changed);
   mtx_unlock(&sync->lock);

   cpvk_cuevent_unref(cpvk_device_from_vk(device), old);
   return VK_SUCCESS;
}

/*
 * Move a binary sync's payload to another one and reset the source.
 *
 * Required of every binary type on a device whose timeline mode is ASSISTED:
 * the runtime steals a waited-on semaphore's payload into a temporary so that
 * the semaphore itself reads as unsignalled from the moment the wait is
 * recorded, which is what its wait-before-signal bookkeeping depends on.
 */
static VkResult
cpvk_sync_move(struct vk_device *device, struct vk_sync *vk_dst,
               struct vk_sync *vk_src)
{
   struct cpvk_sync *dst = cpvk_sync_from_vk(vk_dst);
   struct cpvk_sync *src = cpvk_sync_from_vk(vk_src);
   assert(!(vk_dst->flags & VK_SYNC_IS_TIMELINE));
   assert(!(vk_src->flags & VK_SYNC_IS_TIMELINE));

   mtx_lock(&src->lock);
   uint64_t value = src->value;
   uint64_t pending = src->pending;
   struct cpvk_cuevent *ev = src->event;
   src->value = 0;
   src->pending = 0;
   src->event = NULL;
   src->epoch++;
   cnd_broadcast(&src->changed);
   mtx_unlock(&src->lock);

   mtx_lock(&dst->lock);
   struct cpvk_cuevent *old = dst->event;
   dst->value = value;
   dst->pending = pending;
   dst->event = ev;
   dst->epoch++;
   cnd_broadcast(&dst->changed);
   mtx_unlock(&dst->lock);

   cpvk_cuevent_unref(cpvk_device_from_vk(device), old);
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
   uint64_t target = cpvk_sync_target(sync, wait_value);
   uint64_t now_ns = os_time_get_nano();
   for (;;) {
      /* PENDING asks whether the work that will get there has been issued,
       * not whether it has finished. */
      uint64_t reached =
         (wait_flags & VK_SYNC_WAIT_PENDING) ? sync->pending : sync->value;
      if (target <= reached)
         break;

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
   /*
    * One type serves both kinds; VK_SYNC_IS_TIMELINE on the object picks.
    *
    * WAIT_BEFORE_SIGNAL is deliberately absent. This queue has one CUDA
    * stream and no way to enqueue a wait for a value nothing has promised
    * yet, so claiming it would put the runtime in NATIVE timeline mode and
    * hand it a submit-and-forget guarantee the driver cannot keep. What is
    * claimed instead is WAIT_PENDING, which is honest -- a submit marks its
    * signals pending before it returns -- and which puts the device in
    * ASSISTED mode, where the runtime holds a wait-before-signal submit on
    * its own thread until the promise exists. Wait-before-signal then works,
    * with the runtime's help rather than by pretending.
    */
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE |
               VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL | VK_SYNC_FEATURE_WAIT_PENDING,
   .init = cpvk_sync_init,
   .finish = cpvk_sync_finish,
   .signal = cpvk_sync_signal,
   .get_value = cpvk_sync_get_value,
   .reset = cpvk_sync_reset,
   .move = cpvk_sync_move,
   .wait = cpvk_sync_wait,
};

const struct vk_sync_type *const cpvk_sync_types[] = {
   &cpvk_sync_type,
   NULL,
};
