/*
 * cudapipe, native Vulkan driver.
 *
 * This is the same CUDA rasterizer backend as the Gallium-hosted driver in
 * src/gallium/drivers/cudapipe, with Mesa's common Vulkan runtime in front of
 * it instead of lavapipe and Gallium. It exists because a large fraction of
 * the Gallium-hosted driver is machinery that reconstructs, by watching a
 * stream of state setters, what a Vulkan command buffer already states
 * explicitly: draw batching and its flush discipline, pass episodes and their
 * segment snapshots, the live save/restore around a deferred execute, the
 * fallback replay, and the small-allocation arenas that exist because every
 * descriptor set is backed by its own device allocation.
 *
 * See CUDAPIPE_VK_NATIVE.md for the plan, the evidence, and the staging.
 *
 * Nothing here is wired into rendering yet. The first milestone is the one
 * the original plan set for the Gallium driver: enumerate a device.
 */

#ifndef CPVK_PRIVATE_H
#define CPVK_PRIVATE_H

#include "vk_device.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vulkan/wsi/wsi_common.h"
#include "vk_command_buffer.h"
#include "cp_nir_to_llvm.h"
#include "cp_kernels.h"
#include "cp_debug.h"
#include <stdatomic.h>
#include <stddef.h>
#include "cp_device.h"
#include "cp_renderer.h"
#include "cp_shader_abi.h"
#include "cp_draw_packet.h"
#include "vk_buffer.h"
#include "vk_image.h"
#include "vk_descriptor_set_layout.h"
#include "vk_device_memory.h"
#include "vk_pipeline_layout.h"
#include "vk_log.h"

#include <cuda.h>

#include "cpvk_entrypoints.h"

struct cpvk_instance {
   struct vk_instance vk;
};

struct cpvk_physical_device {
   struct vk_physical_device vk;
   struct wsi_device wsi_device;
   bool wsi_initialized;

   CUdevice cu_dev;
   int sm_major, sm_minor;
   char name[256];
   size_t vram;
};

struct cpvk_pending_submit;

struct cpvk_device {
   struct vk_device vk;
   struct cpvk_physical_device *pdev;

   CUcontext cu_ctx;
   struct vk_queue queue;

   mtx_t submit_lock;
   cnd_t submit_changed;
   thrd_t submit_thread;
   bool submit_worker_initialized;
   bool submit_worker_stop;
   atomic_bool device_lost;
   struct cpvk_pending_submit *submit_head, *submit_tail;

   /* The renderer, and the device it runs on: the same CUDA draw pipeline the
    * Gallium-hosted driver uses, reached through a header with no Gallium in
    * it. Brought up by cp_context_init(). */
   struct cp_device cp_dev;
   struct cp_context renderer;

   /*
    * Two pages for reads that have nowhere to land: a descriptor-shaped one
    * whose base points at a data one, and the data one, which is all zeroes.
    *
    * They must be two. The first attempt was a single page filled with its
    * own address so that a descriptor read out of it yielded a valid
    * pointer -- and every *integer* read out of it then yielded about ten to
    * the fourteen. A compute shader took its loop bound from one and ran
    * until the replay was killed at forty minutes, 100% of it inside
    * cuStreamSynchronize waiting for a kernel that was never going to end.
    * A wrong frame is a useful failure; a hang is not.
    *
    * This is a bring-up aid, not a fix: the slot being unbound, or the
    * descriptor unwritten, is the bug.
    */
   CUdeviceptr null_desc;
   CUdeviceptr null_data;

   /* Guards each image's list of views. An application externally
    * synchronizes an image and a view separately, so binding memory on one
    * thread can race view creation/destruction on another. */
   simple_mtx_t view_lock;

   /* The last command draw staged into a pending renderer batch. Merge must be
    * decided before the incoming draw mutates live renderer state. The pointer
    * refers into an immutable command-buffer op array, whose storage is stable
    * for the complete host-side submit translation walk, and is never retained
    * after the callback clears prev_draw_valid. */
   const struct cpvk_draw_cmd *prev_draw;
   const struct cp_render_scope *prev_scope;
   bool prev_draw_valid;

   /*
    * Compiled shaders, by the hash of the stage that produced them.
    *
    * Two pipelines built from the same SPIR-V used to compile it twice and
    * hold two cp_shader_binary pointers, and the batcher compares those
    * pointers -- so gltfscenerendering, which builds a pipeline per material,
    * merged nothing at all. Sharing the binary is also the difference between
    * compiling a scene's shaders once and once per material.
    */
   struct cpvk_shader_cache_entry {
      unsigned char hash[BLAKE3_OUT_LEN];
      struct cp_shader_binary *bin;
   } *shader_cache;
   unsigned num_shaders, max_shaders;
   simple_mtx_t shader_cache_lock;

};

/*
 * A memory type index picks the allocator, and the three of them are the split
 * session 13 had to negotiate with lavapipe through two new pipe_screen hooks.
 * Here they are simply what the driver does.
 */
enum cpvk_memory_kind {
   CPVK_MEM_DEVICE = 0,     /* cuMemAlloc: never host-mapped */
   CPVK_MEM_HOST = 1,       /* cuMemHostAlloc: pinned, host-cached */
   CPVK_MEM_MANAGED = 2,    /* cuMemAllocManaged: both, and migrating */
};

struct cpvk_device_memory {
   struct vk_device_memory vk;
   enum cpvk_memory_kind kind;
   CUdeviceptr dev_ptr;     /* what a kernel dereferences */
   void *host_ptr;          /* what vkMapMemory returns, or NULL */
};

/*
 * Descriptors, resolved on the host.
 *
 * The generated kernel takes one argument: a pointer to an array of pointers,
 * where slot 18 + i is the base address of buffer i ("compute reads
 * args[18 + i] directly; it has no batch", cp_nir_to_llvm.c). So a descriptor
 * set here is an array of device addresses and nothing else, and binding one
 * is filling in argument slots -- there is no descriptor memory, no
 * VkDeviceMemory per set, and therefore none of the page-fault storm that
 * arrangement caused under lavapipe.
 */
/*
 * Bindings and descriptors are sized by the layout, not by a constant. A
 * fixed cap silently dropped writes at larger flat indices, and any cap large
 * enough for the Vulkan 1.1 per-set minima would be mostly waste in every set
 * an application actually creates.
 */
struct cpvk_descriptor_binding {
   uint32_t binding;          /* application-visible, may be sparse */
   VkDescriptorType type;
   uint32_t count;
   uint32_t flat;             /* dense index within this set */
};

struct cpvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   unsigned num_bindings;
   unsigned num_descriptors;
   struct cpvk_descriptor_binding *bindings;   /* num_bindings */
   uint32_t *immutable_sampler;                /* num_descriptors */
   bool *immutable;                            /* num_descriptors */
};

struct cpvk_pipeline_layout {
   struct vk_pipeline_layout vk;
   unsigned set_base[MESA_VK_MAX_DESCRIPTOR_SETS];  /* flat base per set */
   /* The buffer slot holding this set's descriptor buffer address. Textures
    * need the set, not the binding; buffers keep using their own slot. */
   unsigned set_slot[MESA_VK_MAX_DESCRIPTOR_SETS];
   unsigned num_descriptors;
};

/* The descriptor row, the constant-buffer slots and their assertions all
 * live in the shader ABI header now; see cp_shader_abi.h. */

#define CPVK_DESCRIPTOR_TYPE_COUNT (VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1)

struct cpvk_descriptor_set {
   struct vk_object_base base;
   unsigned num_bindings;
   unsigned num_descriptors;
   struct cpvk_descriptor_binding *bindings;   /* num_bindings */
   CUdeviceptr *addrs;                         /* num_descriptors */
   bool *immutable;                            /* num_descriptors */
   /*
    * The view each image descriptor names. A storage image caches raw
    * addresses and strides, and vkBindImageMemory2 may legally follow the
    * descriptor write, so the row is re-resolved from the view when the set is
    * snapshotted rather than only when it is written.
    */
   struct cpvk_image_view **views;             /* num_descriptors */
   /* Whether any of those entries is set, so a snapshot of a set without
    * storage images does not walk its descriptors at all. */
   bool has_storage_views;

   /* Descriptor updates are host-only.  Recording copies this array into the
    * command buffer's device arena, so a set itself needs no CUDA allocation. */
   struct cpvk_descriptor *host;
   struct cpvk_descriptor_pool *pool;
   struct cpvk_descriptor_set *pool_next;
   uint32_t pool_counts[CPVK_DESCRIPTOR_TYPE_COUNT];
};

static inline const struct cpvk_descriptor_binding *
cpvk_find_binding(const struct cpvk_descriptor_binding *bindings,
                  unsigned count, uint32_t binding)
{
   for (unsigned i = 0; i < count; i++)
      if (bindings[i].binding == binding)
         return &bindings[i];
   return NULL;
}

/*
 * A query pool. Timestamps are what the captures use; they are taken on the
 * host when the recorded operation runs, which is the moment the stream has
 * reached that point, because a submit synchronises before it returns.
 *
 * Occlusion and pipeline-statistics queries read zero and say so: the
 * rasterizer counts nothing today, and a fabricated count is worse than an
 * obvious one.
 */
struct cpvk_query_pool {
   struct vk_object_base base;
   struct cpvk_device *dev;
   VkAllocationCallbacks alloc;
   atomic_uint refcnt;
   VkQueryType type;
   uint32_t count;
   uint64_t *results;
   atomic_bool *available;
   /* VK_QUERY_RESULT_WAIT_BIT waits for exactly the requested queries. A
    * device-wide drain would block behind unrelated later work, which the
    * asynchronous submit path makes easy to reach. */
   mtx_t lock;
   cnd_t changed;
};

struct cpvk_descriptor_update_template {
   struct vk_object_base base;
   uint32_t entry_count;
   VkDescriptorUpdateTemplateEntry entries[];
};

struct cpvk_sampler {
   struct vk_object_base base;
   unsigned index;                    /* into cp_sampler_table */
};


struct cpvk_descriptor_pool {
   struct vk_object_base base;
   struct cpvk_descriptor_set *sets;
   uint32_t max_sets, allocated_sets;
   uint64_t capacity[CPVK_DESCRIPTOR_TYPE_COUNT];
   uint64_t used[CPVK_DESCRIPTOR_TYPE_COUNT];
};

/*
 * Binary host/device event state. Device set, reset and wait commands retain
 * this object and execute as ordered command-buffer operations; host access is
 * guarded by the same mutex and condition variable.
 */
struct cpvk_event {
   struct vk_object_base base;
   struct cpvk_device *dev;
   VkAllocationCallbacks alloc;
   atomic_uint refcnt;
   mtx_t lock;
   cnd_t changed;
   bool signaled;
};

#define CPVK_MAX_DISPATCHES 64

/* Buffer slot 0 is the push constant block; descriptors start after it. */
#define CPVK_MAX_PUSH_BYTES 256

struct cpvk_dispatch {
   struct cpvk_pipeline *pipeline;
   uint32_t grid[3];
   CUdeviceptr addrs[16];
   /* Slot 0 is the push constant block for a compute shader exactly as it is
    * for a graphics one. Leaving it empty is an address of zero, which is
    * what a capture's dispatch was reading 128 bytes into. */
   unsigned char push[CPVK_MAX_PUSH_BYTES];
   unsigned push_size;
};


struct cpvk_draw_cmd {
   /*
    * The batch partition, decided when recording ends rather than while the
    * submit walk happens to go past. plan_prev names the draw this decision
    * was computed against -- the immediately preceding op, when that op is a
    * draw in the same scope -- and plan_mergeable is cpvk_draws_mergeable()
    * of the pair, computed once per recording instead of once per submission.
    * The submit path uses it only when dev->prev_draw is exactly plan_prev;
    * anything else falls back to the dynamic comparison, so behaviour is
    * identical by construction.
    */
   const struct cpvk_draw_cmd *plan_prev;
   bool plan_mergeable;
   struct cpvk_pipeline *pipeline;
   uint32_t scope_index;
   struct cp_viewport_state viewport;
   struct cp_rect scissor;
   struct cp_draw_call call;
   struct cp_draw_range range;
   uint64_t vb_base[16];
   unsigned num_vb;
   CUdeviceptr addrs[16];
   unsigned char vs_push[CPVK_MAX_PUSH_BYTES];
   unsigned char fs_push[CPVK_MAX_PUSH_BYTES];
   unsigned vs_push_size, fs_push_size;
};

/* A recorded attachment clear. The colour one is a fill of the image itself;
 * the depth one clears the renderer's own depth buffer, which is where the
 * depth test reads from -- the depth image is never written, exactly as under
 * Gallium, where lavapipe's zsbuf only ever set has_zs. */
struct cpvk_clear {
   bool depth;
   float depth_value;
   void *data;
   uint64_t offset;
   unsigned width, height, stride, pixel_size;
   uint32_t value[4];
   /*
    * Every sample, not just the first. A multisampled image keeps its samples
    * as planes, and clearing only plane zero left the other three black: the
    * resolve then averaged one cleared sample with three that were not, and
    * multisampling's white background came out 64, which is 255/4 rounded.
    */
   unsigned samples;
   uint64_t sample_stride;
};

/*
 * Command buffers record operations in order and replay them at submit.
 * Ordering is the whole point: a clear recorded after a draw must not run
 * before it, which a pair of separate arrays cannot express.
 */
/* A recorded transfer. Buffers and linear images are both flat device memory
 * here, so one 2D copy covers every case: a buffer is the degenerate one with
 * a single row. */
struct cpvk_copy {
   CUdeviceptr src, dst;
   size_t src_pitch, dst_pitch;
   size_t width_bytes, rows;
   /* A blit between two 32-bit formats of opposite channel order. Vulkan
    * requires a blit to convert, and an offscreen app relies on it: the
    * sample suite blits its BGRA render target into an RGBA staging image
    * precisely so it does not have to swizzle on the CPU. */
   bool swap_rb;

   /*
    * A scaling blit, which is how every mip chain in a capture is built:
    * level n-1 blitted down to level n. Zero means no scaling and the fast
    * device-to-device path applies.
    */
   unsigned src_w, src_h, dst_w, dst_h, bpp;
   bool filter_linear;

   /*
    * The last byte each end may touch. A region computed from the wrong
    * subresource -- a mip level's extent against another level's pitch, or a
    * multisample image whose allocation holds one sample -- runs past the
    * allocation, and cuMemcpy2DAsync answers that by faulting inside libcuda
    * rather than returning an error.
    */
   uint64_t src_end, dst_end;

   /* A multisample resolve: average `samples` planes `sample_stride` apart.
    * `encoding` is the source's cp_color_encoding, which the resolve kernel
    * needs to decode and re-encode the texels; -1 when it is not known and
    * the host path has to run. */
   unsigned samples;
   uint64_t sample_stride;
   int encoding;
   int dst_encoding;
};

struct cpvk_query_op {
   struct cpvk_query_pool *pool;
   uint32_t first, count;
   bool reset;                       /* else: write a timestamp at `first` */
};

enum cpvk_op_kind {
   CPVK_OP_BEGIN_RENDER,
   CPVK_OP_END_RENDER,
   CPVK_OP_QUERY,
   CPVK_OP_DRAW,
   CPVK_OP_CLEAR,
   CPVK_OP_COPY,
   /*
    * A dispatch is an op like any other, because it must run where it was
    * recorded. Held in its own array it ran before every copy and draw in the
    * command buffer, so computeshader's compute stage read the image its own
    * recorded copy had not filled yet and embossed a constant.
    */
   CPVK_OP_DISPATCH,
   CPVK_OP_BARRIER,
   CPVK_OP_EVENT_SET,
   CPVK_OP_EVENT_RESET,
   CPVK_OP_EVENT_WAIT,
};

struct cpvk_event_op {
   struct cpvk_event *event;
};

struct cpvk_op {
   enum cpvk_op_kind kind;
   /* BEGIN_RENDER/DRAW only; remapped when secondary ops are imported. */
   uint32_t scope_index;
   union {
      struct cpvk_draw_cmd draw_cmd;
      struct cpvk_clear clear;
      struct cpvk_copy copy;
      struct cpvk_query_op query;
      struct cpvk_dispatch dispatch;
      struct cpvk_event_op event;
   };
};

struct cpvk_cmd_buffer {
   struct vk_command_buffer vk;
   struct cpvk_pipeline *graphics_pipeline;
   struct cpvk_pipeline *compute_pipeline;
   struct cpvk_pipeline **retained_pipelines;
   unsigned num_retained_pipelines, max_retained_pipelines;
   struct cpvk_query_pool **retained_queries;
   unsigned num_retained_queries, max_retained_queries;
   struct cpvk_event **retained_events;
   unsigned num_retained_events, max_retained_events;
   CUdeviceptr graphics_addrs[16];
   CUdeviceptr compute_addrs[16];
   struct cpvk_dispatch dispatches[CPVK_MAX_DISPATCHES];
   unsigned num_dispatches;

   /* Graphics: the state the command buffer has accumulated, and the draws. */
   struct cp_fb_desc fb;
   bool has_fb;
   /*
    * A render pass can name a single-sample attachment to resolve into when
    * it ends, and the runtime's render-pass emulation passes it through as
    * VkRenderingAttachmentInfo::resolveImageView. Ignoring it is why
    * multisampling drew 20,000 triangles into an image nothing ever read.
    */
   struct cpvk_copy resolve;
   bool resolve_valid;
   /* Depth resolve runs after the scope's depth store, so it is recorded
    * after the end marker rather than before it. */
   struct cpvk_copy depth_resolve;
   bool depth_resolve_valid;
   struct cp_viewport_state viewport;
   struct cp_rect scissor;
   uint64_t vb_base[16];
   unsigned num_vb;
   unsigned fb_samples;
   const void *index_ptr;
   unsigned index_size;
   /* Push constants are per stage: an application may write overlapping
    * bytes for VERTEX and for FRAGMENT and each stage must observe its own
    * write. The renderer binds one uniform slot per stage, so keep one image
    * per stage this driver actually runs. */
   unsigned char vs_push[CPVK_MAX_PUSH_BYTES];
   unsigned char fs_push[CPVK_MAX_PUSH_BYTES];
   unsigned char compute_push[CPVK_MAX_PUSH_BYTES];
   unsigned vs_push_size, fs_push_size, compute_push_size;
   /*
    * Grown rather than capped. A fixed array silently dropped everything past
    * its end, which is the failure that looks like a driver bug in whatever
    * came after: bloom records well over sixty operations and rendered a
    * completely black frame.
    */
   struct cpvk_op *ops;
   unsigned num_ops, max_ops;
   struct cp_render_scope *scopes;
   unsigned num_scopes, max_scopes;
   uint32_t active_scope;
   uint32_t next_scope_serial;

   /*
    * Descriptor sets are snapshotted at bind time into memory this command
    * buffer owns, and the draws point at the copy.
    *
    * They have to be. Execution is deferred to submit, and an application
    * rebinds the same set with a different dynamic offset between draws --
    * dynamicuniformbuffer does exactly that, once per object. Pointing every
    * recorded draw at the live set means they all see whatever the last bind
    * left behind, which is one object drawn twenty times.
    */
   unsigned arena_grows;
   CUdeviceptr desc_arena;
   struct cpvk_descriptor *desc_arena_host;
   size_t desc_arena_size, desc_arena_used;
   bool desc_arena_dirty;
   /* Arenas outgrown while draws still point into them; copied at submit and
    * freed at reset.  Host and device storage stay separate so a CPU rewrite
    * cannot fault the page back while a shader is reading it. */
   struct {
      CUdeviceptr dev;
      void *host;
      size_t used;
   } *desc_retired;
   unsigned num_desc_retired, max_desc_retired;
};

void cpvk_pipeline_ref(struct cpvk_pipeline *pipeline);
void cpvk_pipeline_unref(struct cpvk_pipeline *pipeline);
void cpvk_query_pool_ref(struct cpvk_query_pool *pool);
void cpvk_query_pool_unref(struct cpvk_query_pool *pool);
void cpvk_event_ref(struct cpvk_event *event);
void cpvk_event_unref(struct cpvk_event *event);

void cpvk_execute_draw_cmd(struct cpvk_device *dev,
                           const struct cp_render_scope *scope,
                           const struct cpvk_draw_cmd *d);
void cpvk_execute_clear(struct cpvk_device *dev, const struct cpvk_clear *c);
VkResult cpvk_execute_copy(struct cpvk_device *dev, const struct cpvk_copy *c);
void cpvk_batch_break_report(void);
VkResult cpvk_execute_query(struct cpvk_device *dev,
                            const struct cpvk_query_op *q);

extern const struct vk_command_buffer_ops cpvk_cmd_buffer_ops;
extern const struct vk_sync_type *const cpvk_sync_types[];

VkResult cpvk_execute_dispatch(struct cpvk_device *dev,
                               const struct cpvk_dispatch *d);
VkResult cpvk_execute_order_op(struct cpvk_device *dev,
                              const struct cpvk_op *op);
VkResult cpvk_execute_cmd_buffer(struct cpvk_device *dev,
                                 struct cpvk_cmd_buffer *cmd);

struct cpvk_pipeline {
   struct vk_object_base base;
   struct cpvk_device *dev;
   VkAllocationCallbacks alloc;
   atomic_uint refcnt;
   VkPipelineBindPoint bind_point;
   struct cp_shader_binary *bin;   /* the compiled CUDA kernel */
   struct cp_shader_binary *vs;    /* graphics: the two stages */
   struct cp_shader_binary *fs;
   /* The pipeline state the renderer reads, resolved at creation, which is
    * where Vulkan puts it and where the Gallium adapter could never have it. */
   struct cp_raster_state raster;
   struct cp_depth_state depth;
   struct cp_blend_desc blend;
   struct cp_vertex_elem velem[16];
   unsigned num_velem, vertex_stride;
   unsigned samples;
   enum mesa_prim topology;
   /*
    * Static viewport/scissor. A pipeline that does not declare these dynamic
    * supplies them itself and an application then never calls the setters;
    * taking them only from the command buffer left the scissor empty and the
    * draw rasterized nothing.
    */
   bool static_viewport, static_scissor;
   struct cp_viewport_state viewport;
   struct cp_rect scissor;
   /* The block size. Under Gallium this arrived per dispatch from lavapipe,
    * which had read it out of the shader; natively the pipeline is where it
    * belongs, since it is fixed at compile time for anything but
    * VK_EXT_subgroup_size_control. */
   uint16_t local_size[3];
};

struct cpvk_buffer {
   struct vk_buffer vk;
   struct cpvk_device_memory *mem;
   VkDeviceSize offset;
};

#define CPVK_MAX_MIP_LEVELS 16

struct cpvk_image_view;

struct cpvk_image {
   struct vk_image vk;
   struct cpvk_device_memory *mem;
   VkDeviceSize offset;
   /* Every level at its own offset. Folding that into one base is the bug
    * that wrote level 0 over and over while the rest stayed untouched, and
    * stayed invisible until textureLod was honoured. */
   uint64_t level_offset[CPVK_MAX_MIP_LEVELS];
   uint64_t level_size[CPVK_MAX_MIP_LEVELS];
   uint32_t row_stride[CPVK_MAX_MIP_LEVELS];
   uint64_t size;
   /* One sample plane's worth; sample n lives at n * sample_stride. */
   uint64_t sample_stride;
   uint32_t texel;                 /* enum cp_texel_format */
   int color;                      /* enum cp_color_encoding, -1 if none */
   struct cpvk_image_view *views;
};

/* The format table's row; cpvk_image.c owns the table. */
struct cpvk_format_info {
   VkFormat vk;
   uint32_t texel;              /* enum cp_texel_format, 0 = not sampleable */
   int color;                   /* enum cp_color_encoding, -1 = not renderable */
   bool depth;
};

const struct cpvk_format_info *cpvk_format_info(VkFormat format);

struct cpvk_image_view {
   struct vk_image_view vk;
   struct cpvk_image *image;
   /* Managed cp_texture_info: what a texture handle ultimately points at. */
   CUdeviceptr tex_info;
   struct cp_texture_info *tex_info_host;
   struct cpvk_image_view *image_next;
};

VK_DEFINE_HANDLE_CASTS(cpvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(cpvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(cpvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_update_template, base,
                               VkDescriptorUpdateTemplate,
                               VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_query_pool, base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_event, base, VkEvent,
                               VK_OBJECT_TYPE_EVENT)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_sampler, base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_buffer, vk.base, VkBuffer,
                               VK_OBJECT_TYPE_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_pipeline, base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_set_layout, vk.base,
                               VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_pipeline_layout, vk.base, VkPipelineLayout,
                               VK_OBJECT_TYPE_PIPELINE_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_DEFINE_HANDLE_CASTS(cpvk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_image, vk.base, VkImage,
                               VK_OBJECT_TYPE_IMAGE)
VK_DEFINE_NONDISP_HANDLE_CASTS(cpvk_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)

#endif /* CPVK_PRIVATE_H */
