/*
 * Device, queue, memory and buffers for the native driver.
 *
 * The memory types are the point of this file. Under lavapipe every
 * VkDeviceMemory became a pipe_screen::allocate_memory, which meant a 64-byte
 * descriptor set became a page-granular managed allocation the host wrote and
 * every launch then faulted on -- 22.4 ms of a 122.6 ms frame at its worst,
 * and two optimisation passes (a small-allocation arena, then a bounded
 * allocation cache) to claw back. Here the driver owns the allocator, and the
 * three memory types session 13 had to add hooks to lavapipe for are just
 * three branches.
 */

#include "cpvk_private.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_sync.h"
#include "vk_util.h"
#include "util/os_time.h"

/* Last: intercepts the CUDA entry points for the iteration 26 census. */
#include "cp_smallop_tele.h"

/*
 * One signal a completed submit owes, with the epoch the sync was armed at:
 * the sync may have been reset while the submit was still running, and a
 * signal whose epoch is stale has already been consumed. See cpvk_sync.c.
 */
struct cpvk_pending_signal {
   struct vk_sync *sync;
   uint64_t value;
   uint64_t epoch;
};

struct cpvk_pending_submit {
   struct cpvk_pending_submit *next;
   /* The submit's completion event, shared with every sync this submit
    * signals; this record holds one reference and drops it below. */
   struct cpvk_cuevent *done;
   uint32_t signal_count;
   struct cpvk_pending_signal signals[];
};

static int
cpvk_submit_worker(void *data)
{
   struct cpvk_device *dev = data;
   cuCtxSetCurrent(dev->cu_ctx);
   for (;;) {
      mtx_lock(&dev->submit_lock);
      while (!dev->submit_head && !dev->submit_worker_stop)
         cnd_wait(&dev->submit_changed, &dev->submit_lock);
      if (!dev->submit_head && dev->submit_worker_stop) {
         mtx_unlock(&dev->submit_lock);
         return 0;
      }
      struct cpvk_pending_submit *pending = dev->submit_head;
      mtx_unlock(&dev->submit_lock);

      CUresult status = cuEventSynchronize(pending->done->event);
      if (status != CUDA_SUCCESS)
         atomic_store_explicit(&dev->device_lost, true, memory_order_release);
      /* Always unblock Vulkan waiters. They observe DEVICE_LOST on the next
       * queue/device call instead of hanging behind a failed CUDA event. */
      for (uint32_t i = 0; i < pending->signal_count; i++) {
         cpvk_sync_signal_completion(&dev->vk, pending->signals[i].sync,
                                     pending->signals[i].value,
                                     pending->signals[i].epoch);
      }
      /* Not destroyed here: the syncs this submit signalled hold their own
       * references, and a wait on one of them becomes a device wait on this
       * event. The last holder destroys it. */
      cpvk_cuevent_unref(dev, pending->done);

      mtx_lock(&dev->submit_lock);
      assert(dev->submit_head == pending);
      dev->submit_head = pending->next;
      if (!dev->submit_head)
         dev->submit_tail = NULL;
      cnd_broadcast(&dev->submit_changed);
      mtx_unlock(&dev->submit_lock);
      free(pending);
   }
}

static VkResult
cpvk_submit_worker_init(struct cpvk_device *dev)
{
   if (mtx_init(&dev->submit_lock, mtx_plain) != thrd_success)
      return VK_ERROR_INITIALIZATION_FAILED;
   if (cnd_init(&dev->submit_changed) != thrd_success) {
      mtx_destroy(&dev->submit_lock);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   if (thrd_create(&dev->submit_thread, cpvk_submit_worker, dev) !=
       thrd_success) {
      cnd_destroy(&dev->submit_changed);
      mtx_destroy(&dev->submit_lock);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   dev->submit_worker_initialized = true;
   return VK_SUCCESS;
}

static void
cpvk_submit_worker_finish(struct cpvk_device *dev)
{
   if (!dev->submit_worker_initialized)
      return;
   mtx_lock(&dev->submit_lock);
   dev->submit_worker_stop = true;
   cnd_broadcast(&dev->submit_changed);
   mtx_unlock(&dev->submit_lock);
   thrd_join(dev->submit_thread, NULL);
   cnd_destroy(&dev->submit_changed);
   mtx_destroy(&dev->submit_lock);
   dev->submit_worker_initialized = false;
}

static void
cpvk_submit_wait_pending(struct cpvk_device *dev)
{
   if (!dev->submit_worker_initialized)
      return;
   mtx_lock(&dev->submit_lock);
   while (dev->submit_head)
      cnd_wait(&dev->submit_changed, &dev->submit_lock);
   mtx_unlock(&dev->submit_lock);
}


static VkResult
cpvk_submit_abort(struct cpvk_device *dev, VkResult result)
{
   /* An aborted submit leaves recorded work partially translated, so the
    * queue's contract is broken from here on: latch it. */
   if (result == VK_ERROR_DEVICE_LOST)
      atomic_store_explicit(&dev->device_lost, true, memory_order_release);
   cp_batch_flush(&dev->renderer);
   cuStreamSynchronize(dev->renderer.stream);
   cpvk_submit_wait_pending(dev);
   cp_scratch_reset(&dev->renderer);
   dev->prev_draw = NULL;
   dev->prev_scope = NULL;
   dev->prev_draw_valid = false;
   return result;
}

static VkResult
cpvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct cpvk_device *dev =
      container_of(vk_queue->base.device, struct cpvk_device, vk);

   /* The census divides by this: these captures submit twice per frame, which
    * is the same paired-submit convention every timing number here uses. */
   cp_smallop_hit(__FILE__, __LINE__, CP_SMALLOP_SUBMIT, 0);

   if (atomic_load_explicit(&dev->device_lost, memory_order_acquire))
      return vk_error(dev, VK_ERROR_DEVICE_LOST);

   CPVK_CTX_SCOPE(dev);

   /*
    * Wait semaphores, on the device where the sync carries a completion event
    * and on the host where it does not.
    *
    * A real driver makes the GPU wait here; this used to block the application
    * thread in vk_sync_wait_many before a single command was translated, which
    * on a queue whose signals are already CUDA events is an unnecessary round
    * trip. The fallback is not a formality: a sync with no event has had
    * nothing recorded into it -- a fence or semaphore signalled from the host,
    * or one whose signalling submit has not been issued yet -- and there is no
    * device object to wait on.
    */
   for (uint32_t i = 0; i < submit->wait_count; i++) {
      if (!cp_debug->no_gpu_sem_wait &&
          cpvk_sync_gpu_wait(&dev->vk, submit->waits[i].sync,
                             submit->waits[i].wait_value,
                             dev->renderer.stream))
         continue;
      VkResult result = vk_sync_wait(&dev->vk, submit->waits[i].sync,
                                     submit->waits[i].wait_value,
                                     VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
   }
   mtx_lock(&dev->submit_lock);
   bool retired = dev->submit_head == NULL;
   mtx_unlock(&dev->submit_lock);
   /* Keep asynchronous bursts asynchronous, but do not let a producer which
    * never waits grow the retired scratch list without bound. */
   if (!retired && dev->renderer.scratch.used >= 256ull * 1024 * 1024) {
      cpvk_submit_wait_pending(dev);
      retired = true;
   }
   if (retired)
      cp_scratch_reset(&dev->renderer);
   dev->renderer.num_host_maps = 0;
   for (uint32_t i = 0; i < submit->command_buffer_count; i++)
      {
         /* Consume rather than read: a command buffer recorded once and
          * submitted N times grew its arena once, not N times. The first
          * version of this counter read it repeatedly and manufactured the
          * finding that the arena grows once per render scope. */
         struct cpvk_cmd_buffer *c =
            container_of(submit->command_buffers[i],
                         struct cpvk_cmd_buffer, vk);
         dev->renderer.plan.arena_grows += c->arena_grows;
         c->arena_grows = 0;
      }

   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct cpvk_cmd_buffer *cmd =
         container_of(submit->command_buffers[i], struct cpvk_cmd_buffer, vk);

      /* Publish the host mirrors for sampler specialization. Shader UBO rows
       * contain device addresses; this is the bounded translation used by the
       * one renderer path which inspects descriptors on the CPU. */
      unsigned needed = cmd->num_desc_retired + !!cmd->desc_arena;
      if (dev->renderer.num_host_maps + needed > CP_MAX_HOST_MAPS)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      for (unsigned a = 0; a < cmd->num_desc_retired; a++)
         dev->renderer.host_maps[dev->renderer.num_host_maps++] =
            (struct cp_host_map) {
               cmd->desc_retired[a].dev, cmd->desc_retired[a].host,
               cmd->desc_retired[a].used,
            };
      if (cmd->desc_arena)
         dev->renderer.host_maps[dev->renderer.num_host_maps++] =
            (struct cp_host_map) {
               cmd->desc_arena, cmd->desc_arena_host,
               cmd->desc_arena_used,
            };

      /* Descriptor snapshots are written by the CPU while recording and read
       * by every shader. A managed arena made that one page ping-pong on each
       * frame: the first tiny vertex launch paid hundreds of GPU page faults.
       * Upload each recorded arena once instead. The synchronous copy also
       * orders it before both CUDA streams used below. */
      if (cmd->desc_arena_dirty) {
         cp_ctx_check("cpvk_queue_submit/desc-upload", dev->cu_ctx);
         int64_t upload_t0 = os_time_get_nano();
         for (unsigned a = 0; a < cmd->num_desc_retired; a++) {
            if (cuMemcpyHtoD(cmd->desc_retired[a].dev,
                             cmd->desc_retired[a].host,
                             cmd->desc_retired[a].used) != CUDA_SUCCESS)
               return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
         }
         if (cmd->desc_arena_used &&
             cuMemcpyHtoD(cmd->desc_arena, cmd->desc_arena_host,
                          cmd->desc_arena_used) != CUDA_SUCCESS)
            return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
         cmd->desc_arena_dirty = false;
         dev->renderer.plan.wait_upload_ns +=
            (uint64_t)(os_time_get_nano() - upload_t0);
         dev->renderer.plan.wait_upload_n++;
      }

      /* Earliest point on the submit path that reaches CUDA, and so where a
       * missing entry-point scope shows up first. */
      cp_ctx_check("cpvk_queue_submit/execute", dev->cu_ctx);

      /* In record order: a clear after a draw must not run before it. */
      for (unsigned o = 0; o < cmd->num_ops; o++) {
         switch (cmd->ops[o].kind) {
         case CPVK_OP_BEGIN_RENDER: {
            uint32_t s = cmd->ops[o].scope_index;
            if (s >= cmd->num_scopes)
               return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
            const struct cp_render_scope *scope = &cmd->scopes[s];
            cp_render_scope_begin(&dev->renderer, scope);
            break;
         }
         case CPVK_OP_END_RENDER:
            cp_render_scope_end(&dev->renderer);
            break;
         case CPVK_OP_CLEAR:
            if (!cpvk_execute_clear(dev, &cmd->ops[o].clear))
               return cpvk_submit_abort(dev,
                  vk_error(dev, VK_ERROR_DEVICE_LOST));
            if (cmd->ops[o].clear.image &&
                !cpvk_texture_cache_image_written(cmd->ops[o].clear.image,
                                                  dev->renderer.stream))
               return cpvk_submit_abort(dev,
                  vk_error(dev, VK_ERROR_DEVICE_LOST));
            break;
         case CPVK_OP_QUERY: {
            VkResult r = cpvk_execute_query(dev, &cmd->ops[o].query);
            if (r != VK_SUCCESS)
               return cpvk_submit_abort(dev, r);
            break;
         }
         case CPVK_OP_FILL: {
            /* A fill observes rendering the same way a copy does, so whatever
             * is held back has to run first. */
            cp_batch_flush(&dev->renderer);
            const struct cpvk_fill *fl = &cmd->ops[o].fill;
            if (cuMemsetD32Async(fl->dst, fl->value, fl->words,
                                 dev->renderer.stream) != CUDA_SUCCESS)
               return cpvk_submit_abort(dev,
                  vk_error(dev, VK_ERROR_DEVICE_LOST));
            break;
         }
         case CPVK_OP_COPY: {
            VkResult r = cpvk_execute_copy(dev, &cmd->ops[o].copy);
            if (r != VK_SUCCESS)
               return cpvk_submit_abort(dev, r);
            if (cmd->ops[o].copy.dst_image &&
                !cpvk_texture_cache_image_written(cmd->ops[o].copy.dst_image,
                                                  dev->renderer.stream))
               return cpvk_submit_abort(dev,
                  vk_error(dev, VK_ERROR_DEVICE_LOST));
            break;
         }
         case CPVK_OP_DRAW: {
            uint32_t s = cmd->ops[o].scope_index;
            if (s >= cmd->num_scopes)
               return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
            assert(cmd->ops[o].draw_cmd.scope_index == s);
            cpvk_execute_draw_cmd(dev, &cmd->scopes[s],
                                  &cmd->ops[o].draw_cmd);
            break;
         }
         case CPVK_OP_DISPATCH: {
            /* Here, in record order, and not before the copy that fills what
             * it reads. */
            VkResult r = cpvk_execute_dispatch(dev, &cmd->ops[o].dispatch);
            if (r != VK_SUCCESS)
               return cpvk_submit_abort(dev, r);
            break;
         }
         case CPVK_OP_BARRIER:
         case CPVK_OP_EVENT_SET:
         case CPVK_OP_EVENT_RESET:
         case CPVK_OP_EVENT_WAIT: {
            VkResult r = cpvk_execute_order_op(dev, &cmd->ops[o]);
            if (r != VK_SUCCESS)
               return cpvk_submit_abort(dev, r);
            break;
         }
         }
         if (atomic_load_explicit(&dev->device_lost, memory_order_acquire))
            return cpvk_submit_abort(dev,
               vk_error(dev, VK_ERROR_DEVICE_LOST));
      }
   }

   /* Finish command translation, then mark completion on the same ordered
    * stream. The worker publishes Vulkan sync state only after this event;
    * queue submission itself no longer drains CUDA. */
   cp_batch_flush(&dev->renderer);
   /* No owed upload may outlive the submit that produced it: the event
    * recorded below is what the queue reports completion on. */
   if (cp_upload_flush(&dev->renderer) != CUDA_SUCCESS)
      return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
   if (atomic_load_explicit(&dev->device_lost, memory_order_acquire))
      return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
   dev->prev_draw = NULL;
   dev->prev_scope = NULL;
   dev->prev_draw_valid = false;

   size_t pending_size = sizeof(struct cpvk_pending_submit) +
                         (size_t)submit->signal_count *
                         sizeof(struct cpvk_pending_signal);
   struct cpvk_pending_submit *pending = calloc(1, pending_size);
   if (!pending)
      return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
   pending->signal_count = submit->signal_count;
   CUevent done = NULL;
   if (cuEventCreate(&done, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS ||
       cuEventRecord(done, dev->renderer.stream) != CUDA_SUCCESS) {
      if (done)
         cuEventDestroy(done);
      free(pending);
      return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_DEVICE_LOST));
   }
   pending->done = cpvk_cuevent_create(done);
   if (!pending->done) {
      cuEventDestroy(done);
      free(pending);
      return cpvk_submit_abort(dev, vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
   }

   /*
    * Arm the syncs this submit signals before the worker can publish anything.
    * The sync is still unsignalled from here until the worker gets to it; what
    * it has gained is the device-side half of that completion, which is what a
    * later submit waits on, and the promise that it will get there, which is
    * what VK_SYNC_WAIT_PENDING answers.
    */
   for (uint32_t i = 0; i < submit->signal_count; i++) {
      pending->signals[i].sync = submit->signals[i].sync;
      pending->signals[i].value = submit->signals[i].signal_value;
      pending->signals[i].epoch =
         cpvk_sync_arm(&dev->vk, submit->signals[i].sync,
                       submit->signals[i].signal_value, pending->done);
   }

   mtx_lock(&dev->submit_lock);
   if (dev->submit_tail)
      dev->submit_tail->next = pending;
   else
      dev->submit_head = pending;
   dev->submit_tail = pending;
   cnd_signal(&dev->submit_changed);
   mtx_unlock(&dev->submit_lock);
   return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
cpvk_GetDeviceProcAddr(VkDevice device, const char *name)
{
   /* GFXReconstruct normalizes the KHR descriptor-template commands to their
    * promoted core names, but creates an application-less Vulkan 1.0 instance
    * and does not enable KHR_descriptor_update_template on the replay device.
    * Mesa's common lookup therefore returns NULL and GFXReconstruct calls
    * through it.  Bridge just this promoted trio unconditionally.  This is a
    * deliberate compatibility exception to normal version/extension gating;
    * gating it was tested and made both supported captures crash at startup. */
   if (!strcmp(name, "vkCreateDescriptorUpdateTemplate"))
      return (PFN_vkVoidFunction)cpvk_CreateDescriptorUpdateTemplateKHR;
   if (!strcmp(name, "vkDestroyDescriptorUpdateTemplate"))
      return (PFN_vkVoidFunction)cpvk_DestroyDescriptorUpdateTemplateKHR;
   if (!strcmp(name, "vkUpdateDescriptorSetWithTemplate"))
      return (PFN_vkVoidFunction)cpvk_UpdateDescriptorSetWithTemplateKHR;
   return vk_common_GetDeviceProcAddr(device, name);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateDevice(VkPhysicalDevice physicalDevice,
                  const VkDeviceCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(cpvk_physical_device, pdev, physicalDevice);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &pdev->vk.instance->alloc;

   struct cpvk_device *dev = vk_zalloc(alloc, sizeof(*dev), 8,
                                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   bool kernels_initialized = false;
   bool renderer_initialized = false;
   bool shader_lock_initialized = false;

   struct vk_device_dispatch_table dispatch_table = { 0 };
   /* Core and promoted KHR aliases compact to the same dispatch slot. The
    * non-overwrite path coalesces them; the strict path asserts when a driver
    * implements both names. */
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &cpvk_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_device_entrypoints, false);

   VkResult result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table,
                                    pCreateInfo, alloc);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   /* After vk_device_init, which zeroes the device: the common command pool
    * dereferences this the moment a pool is created. */
   dev->vk.command_buffer_ops = &cpvk_cmd_buffer_ops;
   dev->pdev = pdev;
   atomic_init(&dev->device_lost, false);

   /*
    * CUDA 12.8: cuCtxCreate takes three arguments. Code written against
    * CUDA 13's four-argument form does not compile here.
    *
    * Flags zero is CU_CTX_SCHED_AUTO, which is what this has always used and
    * stays the default. CUDAVK_CTX_SCHED overrides it; see cp_debug.h for why
    * the choice matters once anything else on the machine wants the CPU.
    */
   unsigned ctx_flags = 0;
   switch (cp_debug->ctx_sched) {
   case CP_CTX_SCHED_SPIN:     ctx_flags = CU_CTX_SCHED_SPIN; break;
   case CP_CTX_SCHED_YIELD:    ctx_flags = CU_CTX_SCHED_YIELD; break;
   case CP_CTX_SCHED_BLOCKING: ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC; break;
   default:                    ctx_flags = CU_CTX_SCHED_AUTO; break;
   }
   if (cuCtxCreate(&dev->cu_ctx, ctx_flags, pdev->cu_dev) != CUDA_SUCCESS) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_device;
   }
   /*
    * The device the renderer runs on, and then the renderer: the same
    * NVRTC-compiled kernels and the same cp_context_init() the Gallium-hosted
    * driver uses, neither of which needs a pipe_screen or a pipe_context.
    */
   dev->cp_dev.cuda_device = pdev->cu_dev;
   dev->cp_dev.cuda_ctx = dev->cu_ctx;
   dev->cp_dev.sm_major = pdev->sm_major;
   dev->cp_dev.sm_minor = pdev->sm_minor;
   atomic_init(&dev->cp_dev.next_texture_stream_serial, 0);
   if (!cp_kernels_init(&dev->cp_dev.kernels, pdev->sm_major, pdev->sm_minor,
                         pdev->vk.disk_cache)) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_stream;
   }
   kernels_initialized = true;
   /* Zeroed data, and a descriptor page whose every descriptor's base points
    * at it. Only the base fields are pointers; everything a shader reads as a
    * number reads as zero, so a loop bounded by one terminates. */
   if (cuMemAllocManaged(&dev->null_data, 1024 * 1024,
                         CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS ||
       cuMemAllocManaged(&dev->null_desc, 64 * 1024,
                         CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
      result = vk_error(pdev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      goto fail_stream;
   }
   memset((void *)(uintptr_t)dev->null_data, 0, 1024 * 1024);
   memset((void *)(uintptr_t)dev->null_desc, 0, 64 * 1024);
   struct cpvk_descriptor *d =
      (struct cpvk_descriptor *)(uintptr_t)dev->null_desc;
   for (unsigned i = 0; i < 64 * 1024 / sizeof(*d); i++)
      d[i].base = dev->null_data;

   simple_mtx_init(&dev->shader_cache_lock, mtx_plain);
   simple_mtx_init(&dev->view_lock, mtx_plain);
   simple_mtx_init(&dev->texture_cache_lock, mtx_plain);
   simple_mtx_init(&dev->texture_cache_use_lock, mtx_plain);
   dev->texture_cache_images = NULL;
   dev->texture_cache_views = NULL;
   dev->texture_cache_view_count = dev->texture_cache_view_cap = 0;
   dev->texture_batch_workspace = NULL;
   dev->texture_batch_workspace_size = 0;
   dev->texture_cache_bytes = 0;
   memset(&dev->texture_cache_stats, 0, sizeof(dev->texture_cache_stats));
   dev->cp_dev.texture_cache_private = dev;
   dev->cp_dev.texture_cache_resolve = cpvk_texture_cache_resolve;
   dev->cp_dev.texture_cache_resolve_batch = cpvk_texture_cache_resolve_batch;
   dev->cp_dev.texture_cache_written = cpvk_texture_cache_written;
   dev->cp_dev.texture_cache_fatal = cpvk_texture_cache_fatal;
   dev->cp_dev.texture_cache_purge = cpvk_texture_cache_purge;
   dev->cp_dev.texture_cache_use_begin = cpvk_texture_cache_use_begin;
   dev->cp_dev.texture_cache_use_end = cpvk_texture_cache_use_end;
   shader_lock_initialized = true;

   if (!cp_context_init(&dev->renderer, &dev->cp_dev)) {
      result = vk_error(pdev, VK_ERROR_INITIALIZATION_FAILED);
      goto fail_stream;
   }
   renderer_initialized = true;

   result = vk_queue_init(&dev->queue, &dev->vk,
                          &pCreateInfo->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS)
      goto fail_stream;
   dev->queue.driver_submit = cpvk_queue_submit;
   result = cpvk_submit_worker_init(dev);
   if (result != VK_SUCCESS)
      goto fail_queue;

   /* cuCtxCreate above pushed the new context onto this thread's stack and
    * left it current. Everything since needed that; the application does not,
    * so hand the thread back the way it was found. */
   cuCtxPopCurrent(NULL);

   *pDevice = cpvk_device_to_handle(dev);
   return VK_SUCCESS;

fail_queue:
   vk_queue_finish(&dev->queue);
fail_stream:
   if (renderer_initialized)
      cp_context_cleanup(&dev->renderer);
   if (shader_lock_initialized) {
      simple_mtx_destroy(&dev->shader_cache_lock);
      simple_mtx_destroy(&dev->view_lock);
      simple_mtx_destroy(&dev->texture_cache_lock);
      simple_mtx_destroy(&dev->texture_cache_use_lock);
   }
   if (kernels_initialized)
      cp_kernels_destroy(&dev->cp_dev.kernels);
   if (dev->null_desc)
      cuMemFree(dev->null_desc);
   if (dev->null_data)
      cuMemFree(dev->null_data);
   cuCtxPopCurrent(NULL);
   cuCtxDestroy(dev->cu_ctx);
fail_device:
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);

   if (!dev)
      return;

   /* Not CPVK_CTX_SCOPE: the scope's pop would run after cuCtxDestroy below
    * has freed the very context it would pop. Push here, pop by hand at the
    * one point the context is still alive. */
   cuCtxPushCurrent(dev->cu_ctx);
   cuCtxSynchronize();
   cpvk_submit_wait_pending(dev);
   cpvk_submit_worker_finish(dev);
   vk_queue_finish(&dev->queue);

   cpvk_batch_break_report();
   cp_context_cleanup(&dev->renderer);
   cpvk_texture_cache_report(dev);
   for (unsigned i = 0; i < dev->num_shaders; i++)
      cp_shader_binary_destroy(dev->shader_cache[i].bin);
   free(dev->shader_cache);
   cp_kernels_destroy(&dev->cp_dev.kernels);
   if (dev->null_desc)
      cuMemFree(dev->null_desc);
   if (dev->null_data)
      cuMemFree(dev->null_data);
   free(dev->texture_cache_views);
   dev->texture_cache_views = NULL;
   free(dev->texture_batch_workspace);
   dev->texture_batch_workspace = NULL;
   simple_mtx_destroy(&dev->shader_cache_lock);
   simple_mtx_destroy(&dev->view_lock);
   simple_mtx_destroy(&dev->texture_cache_lock);
   simple_mtx_destroy(&dev->texture_cache_use_lock);

   /* Restore the caller's context before the push target stops existing. */
   cuCtxPopCurrent(NULL);
   cuCtxDestroy(dev->cu_ctx);

   const VkAllocationCallbacks *alloc = &dev->vk.alloc;
   vk_device_finish(&dev->vk);
   vk_free(alloc, dev);
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_DeviceWaitIdle(VkDevice _device)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   if (cuCtxSynchronize() != CUDA_SUCCESS)
      atomic_store_explicit(&dev->device_lost, true, memory_order_release);
   cpvk_submit_wait_pending(dev);
   return atomic_load_explicit(&dev->device_lost, memory_order_acquire)
      ? vk_error(dev, VK_ERROR_DEVICE_LOST) : VK_SUCCESS;
}

/*
 * Build an exportable allocation out of the CUDA virtual memory API.
 *
 * Four calls where cuMemAlloc is one, because the two things cuMemAlloc fuses
 * are exactly the two this has to keep apart: cuMemCreate makes the physical
 * allocation and hands back a handle that can become a file descriptor, and
 * the virtual address it answers to is reserved and mapped separately.
 *
 * The rounding is not optional. VMM allocations are made in units of the
 * device's minimum granularity -- 2 MiB on this hardware -- and cuMemCreate
 * refuses a size that is not a multiple of it. Vulkan lets an application ask
 * for any allocationSize, so the request is rounded up and the surplus simply
 * belongs to the allocation; every later VMM call has to be given the rounded
 * size rather than the requested one, which is why it is stored.
 *
 * cuMemSetAccess is the step with no cuMemAlloc counterpart and the easiest to
 * forget: a freshly mapped range is readable and writable by nobody, so
 * without it every kernel touching this memory faults.
 */
static VkResult
cpvk_allocate_exportable(struct cpvk_device *dev,
                         struct cpvk_device_memory *mem, size_t size)
{
   CUmemAllocationProp prop = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location = {
         .type = CU_MEM_LOCATION_TYPE_DEVICE,
         .id = dev->pdev->cu_dev,
      },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
   };

   size_t granularity = 0;
   if (cuMemGetAllocationGranularity(&granularity, &prop,
                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM)
       != CUDA_SUCCESS || granularity == 0)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   size_t padded = (size + granularity - 1) / granularity * granularity;

   CUmemGenericAllocationHandle handle = 0;
   if (cuMemCreate(&handle, padded, &prop, 0) != CUDA_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   CUdeviceptr ptr = 0;
   if (cuMemAddressReserve(&ptr, padded, granularity, 0, 0) != CUDA_SUCCESS) {
      cuMemRelease(handle);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   if (cuMemMap(ptr, padded, 0, handle, 0) != CUDA_SUCCESS) {
      cuMemAddressFree(ptr, padded);
      cuMemRelease(handle);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   CUmemAccessDesc access = {
      .location = prop.location,
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
   };
   if (cuMemSetAccess(ptr, padded, &access, 1) != CUDA_SUCCESS) {
      cuMemUnmap(ptr, padded);
      cuMemAddressFree(ptr, padded);
      cuMemRelease(handle);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   mem->exportable = true;
   mem->vmm_handle = handle;
   mem->vmm_size = padded;
   mem->dev_ptr = ptr;
   /* Device memory in the VMM sense: there is no host mapping to hand out,
    * and CPVK_MEM_DEVICE is already the unmappable kind. */
   mem->host_ptr = NULL;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_AllocateMemory(VkDevice _device,
                    const VkMemoryAllocateInfo *pAllocateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_device_memory *mem =
      vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator,
                              sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->kind = pAllocateInfo->memoryTypeIndex;
   mem->bindings = NULL;
   mem->binding_ledger_failed = false;
   mem->exportable = false;
   mem->vmm_handle = 0;
   mem->vmm_size = 0;
   size_t size = pAllocateInfo->allocationSize;
   CUresult err;

   cp_ctx_check("cpvk_AllocateMemory", dev->cu_ctx);

   /*
    * An allocation the application intends to export cannot come from
    * cuMemAlloc: that returns a bare address with no handle behind it, and
    * there is nothing to turn into a file descriptor. The CUDA virtual memory
    * API splits the two halves apart -- cuMemCreate makes a physical handle
    * that can be exported, and the address it is mapped at is chosen
    * separately -- which is exactly the shape Vulkan's export model wants.
    *
    * Deciding this here, at allocation time, is not an implementation detail
    * leaking upward. It is why VkExportMemoryAllocateInfo has to be supplied
    * before the memory exists in every Vulkan implementation.
    */
   const VkExportMemoryAllocateInfo *export_info =
      vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   if (export_info && export_info->handleTypes) {
      if (export_info->handleTypes !=
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
         vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
         return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      }
      VkResult result = cpvk_allocate_exportable(dev, mem, size);
      if (result != VK_SUCCESS) {
         vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
         return result;
      }
      goto allocated;
   }

   bool retried_after_purge = false;
retry_cuda_allocation:
   switch (mem->kind) {
   case CPVK_MEM_DEVICE:
      err = cuMemAlloc(&mem->dev_ptr, size);
      mem->host_ptr = NULL;
      break;
   case CPVK_MEM_HOST:
      /* Host-visible Vulkan allocations can contain hot UBO/SSBO data. A
       * cache purge is allowed to reclaim optional device arrays first. */
      err = cuMemAllocManaged(&mem->dev_ptr, size, CU_MEM_ATTACH_GLOBAL);
      mem->host_ptr = err == CUDA_SUCCESS
         ? (void *)(uintptr_t)mem->dev_ptr : NULL;
      break;
   default:
      err = cuMemAllocManaged(&mem->dev_ptr, size, CU_MEM_ATTACH_GLOBAL);
      mem->host_ptr = err == CUDA_SUCCESS
         ? (void *)(uintptr_t)mem->dev_ptr : NULL;
      break;
   }
   if (err == CUDA_ERROR_OUT_OF_MEMORY && !retried_after_purge) {
      enum cp_texture_cache_purge_result purge =
         cpvk_texture_cache_purge(dev);
      if (purge == CP_TEXTURE_CACHE_PURGE_FATAL) {
         vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      }
      if (purge == CP_TEXTURE_CACHE_PURGE_RECLAIMED) {
         retried_after_purge = true;
         goto retry_cuda_allocation;
      }
   }

   if (err != CUDA_SUCCESS) {
      vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
      if (err != CUDA_ERROR_OUT_OF_MEMORY)
         cpvk_texture_cache_fatal(dev);
      if (atomic_load_explicit(&dev->device_lost, memory_order_acquire))
         return vk_error(dev, VK_ERROR_DEVICE_LOST);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

allocated:
   /*
    * Vulkan defines ordinary allocation contents as undefined, and clearing
    * them anyway cost the Gallium-hosted driver 1,571 blocking cuMemsetD8
    * calls and 830 ms of CUDA API time per replay before session 11 removed
    * it. Only the explicit request is honoured.
    */
   if (mem->vk.alloc_flags & VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT) {
      if (mem->host_ptr)
         memset(mem->host_ptr, 0, size);
      else
         cuMemsetD8(mem->dev_ptr, 0, size);
   }

   *pMem = cpvk_device_memory_to_handle(mem);
   return VK_SUCCESS;
}

void
cpvk_memory_note_bind(struct cpvk_device *dev, struct cpvk_device_memory *mem,
                      VkDeviceSize offset, VkDeviceSize size,
                      void *resource, struct cpvk_image *image)
{
   if (!cp_debug->texture_cache || !mem || !size)
      return;
   simple_mtx_lock(&dev->texture_cache_lock);
   bool overflow = offset > UINT64_MAX - size;
   VkDeviceSize end = overflow ? UINT64_MAX : offset + size;
   if (image && (overflow || mem->binding_ledger_failed))
      image->cache_alias = true;
   /* A legal rebind of the same resource replaces its old interval; it is not
    * an alias with itself. */
   struct cpvk_memory_binding **old_link = &mem->bindings;
   while (*old_link) {
      if ((*old_link)->resource == resource) {
         struct cpvk_memory_binding *dead = *old_link;
         *old_link = dead->next;
         free(dead);
         continue;
      }
      old_link = &(*old_link)->next;
   }
   for (struct cpvk_memory_binding *b = mem->bindings; b; b = b->next) {
      VkDeviceSize bend = b->offset > UINT64_MAX - b->size
         ? UINT64_MAX : b->offset + b->size;
      if (offset < bend && b->offset < end) {
         if (image)
            image->cache_alias = true;
         if (b->image)
            b->image->cache_alias = true;
      }
   }
   struct cpvk_memory_binding *binding = calloc(1, sizeof(*binding));
   if (!binding) {
      mem->binding_ledger_failed = true;
      if (image)
         image->cache_alias = true;
      for (struct cpvk_memory_binding *b = mem->bindings; b; b = b->next)
         if (b->image)
            b->image->cache_alias = true;
   } else {
      binding->offset = offset;
      binding->size = size;
      binding->resource = resource;
      binding->image = image;
      binding->next = mem->bindings;
      mem->bindings = binding;
   }
   simple_mtx_unlock(&dev->texture_cache_lock);
}

void
cpvk_memory_note_unbind(struct cpvk_device *dev,
                        struct cpvk_device_memory *mem, void *resource)
{
   if (!cp_debug->texture_cache || !mem || !resource)
      return;
   simple_mtx_lock(&dev->texture_cache_lock);
   struct cpvk_memory_binding **link = &mem->bindings;
   while (*link) {
      if ((*link)->resource == resource) {
         struct cpvk_memory_binding *dead = *link;
         *link = dead->next;
         free(dead);
         continue;
      }
      link = &(*link)->next;
   }
   simple_mtx_unlock(&dev->texture_cache_lock);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_FreeMemory(VkDevice _device, VkDeviceMemory _mem,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_device_memory, mem, _mem);
   CPVK_CTX_SCOPE(dev);

   if (!mem)
      return;

   if (cp_debug->texture_cache && mem->bindings)
      cpvk_DeviceWaitIdle(_device);
   while (mem->bindings) {
      struct cpvk_memory_binding *binding = mem->bindings;
      struct cpvk_memory_binding *next = binding->next;
      if (binding->image) {
         cpvk_texture_cache_image_destroy(binding->image);
         binding->image->mem = NULL;
      } else if (binding->resource) {
         ((struct cpvk_buffer *)binding->resource)->mem = NULL;
      }
      free(binding);
      mem->bindings = next;
   }
   if (mem->exportable) {
      /* Unwind cpvk_allocate_exportable in reverse. cuMemFree does not apply:
       * the address was reserved rather than allocated, and the physical
       * handle outlives the mapping until it is released. Any fd handed out
       * by vkGetMemoryFdKHR holds its own reference, so a consumer that still
       * has one keeps the memory alive past this point. */
      cuMemUnmap(mem->dev_ptr, mem->vmm_size);
      cuMemAddressFree(mem->dev_ptr, mem->vmm_size);
      cuMemRelease(mem->vmm_handle);
   } else {
      cuMemFree(mem->dev_ptr);
   }

   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

/*
 * Mint a file descriptor for an exportable allocation.
 *
 * Vulkan transfers ownership to the application: it closes the fd, or hands it
 * to something that does. Calling this twice yields two independent
 * descriptors for the same memory, which is why the handle is exported afresh
 * each time rather than cached.
 */
VKAPI_ATTR VkResult VKAPI_CALL
cpvk_GetMemoryFdKHR(VkDevice _device, const VkMemoryGetFdInfoKHR *pGetFdInfo,
                    int *pFd)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_device_memory, mem, pGetFdInfo->memory);
   CPVK_CTX_SCOPE(dev);

   if (pGetFdInfo->handleType !=
       VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   /* Not an error the application can recover from, but it is the one it will
    * hit if it forgot VkExportMemoryAllocateInfo, so say which. */
   if (!mem || !mem->exportable)
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "this VkDeviceMemory was not allocated with "
                       "VkExportMemoryAllocateInfo naming OPAQUE_FD");

   int fd = -1;
   if (cuMemExportToShareableHandle(&fd, mem->vmm_handle,
                                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                    0) != CUDA_SUCCESS)
      return vk_error(dev, VK_ERROR_TOO_MANY_OBJECTS);

   *pFd = fd;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_MapMemory2(VkDevice _device, const VkMemoryMapInfo *pMemoryMapInfo,
                void **ppData)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_device_memory, mem, pMemoryMapInfo->memory);
   CPVK_CTX_SCOPE(dev);

   /* A device-local allocation is deliberately not mappable: staging through
    * a host-visible allocation is the caller's job, and hiding that behind a
    * silent download and upload is what made cp_buffer_map copy an entire
    * Vulkan allocation around every partial write. */
   if (!mem->host_ptr)
      return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);

   *ppData = (char *)mem->host_ptr + pMemoryMapInfo->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_UnmapMemory2(VkDevice _device, const VkMemoryUnmapInfo *pMemoryUnmapInfo)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_FlushMappedMemoryRanges(VkDevice _device, uint32_t count,
                             const VkMappedMemoryRange *pRanges)
{
   return VK_SUCCESS;   /* every mappable type here is coherent */
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t count,
                                  const VkMappedMemoryRange *pRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_CreateBuffer(VkDevice _device, const VkBufferCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   CPVK_CTX_SCOPE(dev);

   struct cpvk_buffer *buffer =
      vk_buffer_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*buffer));
   if (!buffer)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pBuffer = cpvk_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
cpvk_DestroyBuffer(VkDevice _device, VkBuffer _buffer,
                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(cpvk_device, dev, _device);
   VK_FROM_HANDLE(cpvk_buffer, buffer, _buffer);
   CPVK_CTX_SCOPE(dev);

   if (!buffer)
      return;

   cpvk_memory_note_unbind(dev, buffer->mem, buffer);
   vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
}

VKAPI_ATTR void VKAPI_CALL
cpvk_GetBufferMemoryRequirements2(VkDevice _device,
                                  const VkBufferMemoryRequirementsInfo2 *pInfo,
                                  VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(cpvk_buffer, buffer, pInfo->buffer);

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements) {
      .size = buffer->vk.size,
      .alignment = 256,
      /* Every type can back a buffer; which one the application picks is what
       * decides whether the host can see it. */
      .memoryTypeBits = 0x7,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
cpvk_BindBufferMemory2(VkDevice _device, uint32_t bindInfoCount,
                       const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(cpvk_buffer, buffer, pBindInfos[i].buffer);
      VK_FROM_HANDLE(cpvk_device_memory, mem, pBindInfos[i].memory);

      buffer->mem = mem;
      buffer->offset = pBindInfos[i].memoryOffset;
      cpvk_memory_note_bind(cpvk_device_from_handle(_device), mem,
                            buffer->offset, buffer->vk.size, buffer, NULL);
      if (mem && pBindInfos[i].memoryOffset + buffer->vk.size > mem->vk.size)
         fprintf(stderr, "cudavk: buffer of %llu bytes bound at %llu into an "
                 "allocation of %llu -- it does not fit\n",
                 (unsigned long long)buffer->vk.size,
                 (unsigned long long)pBindInfos[i].memoryOffset,
                 (unsigned long long)mem->vk.size);

      /* The runtime's own vk_buffer_address() asserts on this, and every
       * common entrypoint that takes a buffer goes through it. A CUDA device
       * pointer is a flat address, so the buffer's address is simply where it
       * was bound -- there is nothing to opt into. */
      buffer->vk.device_address = mem ?
         mem->dev_ptr + pBindInfos[i].memoryOffset : 0;
   }
   return VK_SUCCESS;
}
