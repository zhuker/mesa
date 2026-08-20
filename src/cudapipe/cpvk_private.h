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
#include "vk_command_buffer.h"
#include "cp_nir_to_llvm.h"
#include "cp_kernels.h"
#include "cp_debug.h"
#include "cp_device.h"
#include "cp_renderer.h"
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

   CUdevice cu_dev;
   int sm_major, sm_minor;
   char name[256];
   size_t vram;
};

struct cpvk_device {
   struct vk_device vk;
   struct cpvk_physical_device *pdev;

   CUcontext cu_ctx;
   CUstream stream;
   struct vk_queue queue;

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

   /*
    * The last draw staged into the renderer, for deciding whether the next
    * one may join its batch. A copy of the draw rather than anything read
    * from the context, because the decision has to be made before the
    * incoming draw is staged -- a flush renders what is held back and reads
    * the context to do it -- and the context at that moment still describes
    * the previous draw. Allocated once; struct cpvk_draw is declared later.
    */
   struct cpvk_draw *prev_draw;
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
#define CPVK_MAX_BINDINGS 32

struct cpvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   unsigned num_bindings;
   struct {
      VkDescriptorType type;
      unsigned count;
      unsigned flat;          /* index within this set */
   } bindings[CPVK_MAX_BINDINGS];
   unsigned num_descriptors;
};

struct cpvk_pipeline_layout {
   struct vk_pipeline_layout vk;
   unsigned set_base[MESA_VK_MAX_DESCRIPTOR_SETS];  /* flat base per set */
   /* The buffer slot holding this set's descriptor buffer address. Textures
    * need the set, not the binding; buffers keep using their own slot. */
   unsigned set_slot[MESA_VK_MAX_DESCRIPTOR_SETS];
   unsigned num_descriptors;
};

/*
 * One descriptor, laid out so the kernels find what they read where they read
 * it. Only three offsets are fixed and they are fixed by the kernels, not by
 * this struct: the buffer base at 0, the sampler index at
 * CP_DESC_SAMPLER_INDEX_OFFSET, the texture info pointer at
 * CP_DESC_IMAGE_FUNCTIONS_OFFSET. Everything between them is padding this
 * driver owns.
 */
struct cpvk_descriptor {
   uint64_t base;                     /* +0  buffer base, or image base */
   uint8_t  pad0[16];
   uint32_t row_stride;               /* +24 storage image only */
   /*
    * +28 is two things, because a descriptor is one kind or the other and
    * lavapipe's lp_descriptor overlaps them the same way: a sampler's table
    * index, or a storage image's layer stride.
    */
   uint32_t sampler_index_or_img_stride;
   uint8_t  pad1[8];
   uint32_t base_offset;              /* +40 storage image only */
   uint8_t  pad2[4];
   uint64_t texture_info;             /* +48 struct cp_texture_info * */
   uint8_t  pad3[8];
};
static_assert(sizeof(struct cpvk_descriptor) == 64,
              "the kernels read fixed offsets into this");

struct cpvk_descriptor_set {
   struct vk_object_base base;
   struct cpvk_descriptor_set_layout *layout;
   CUdeviceptr addrs[CPVK_MAX_BINDINGS];

   /*
    * The set as one buffer, which is what a texture handle points into: the
    * shader computes `set_base + binding * sizeof(struct cpvk_descriptor)`
    * and never loads it, so there is nowhere to put a per-binding address.
    * Managed, because cp_renderer.c's sampler-variant specialisation reads
    * these on the host.
    */
   CUdeviceptr buf;
   struct cpvk_descriptor *host;
};

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
   VkQueryType type;
   uint32_t count;
   uint64_t *results;
   bool *available;
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
};

#define CPVK_MAX_DISPATCHES 64

/* Buffer slot 0 is the push constant block; descriptors start after it. */
#define CPVK_UBO_PUSH_SLOT  0
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

#define CPVK_DESCRIPTOR_SIZE 64

struct cpvk_draw {
   struct cpvk_pipeline *pipeline;
   struct cp_fb_desc fb;
   struct cp_viewport_state viewport;
   struct cp_rect scissor;
   struct cp_draw_call call;
   struct cp_draw_range range;
   uint64_t vb_base[16];
   unsigned num_vb;
   CUdeviceptr addrs[16];
   /*
    * A hash of each bound set's descriptors, taken when it was bound.
    *
    * The addresses cannot be compared -- every bind is snapshotted into fresh
    * memory, so two draws binding the identical set never share one -- and
    * they cannot be ignored either, because merging draws that sample
    * different textures shades the batch with one of them. gltfscenerendering
    * went from exact to 20.768 that way. The contents are what matters and
    * this is them.
    */
   uint64_t desc_hash[16];
   unsigned char push[CPVK_MAX_PUSH_BYTES];
   unsigned push_size;
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

   /* A multisample resolve: average `samples` planes `sample_stride` apart. */
   unsigned samples;
   uint64_t sample_stride;
};

struct cpvk_query_op {
   struct cpvk_query_pool *pool;
   uint32_t first, count;
   bool reset;                       /* else: write a timestamp at `first` */
};

enum cpvk_op_kind {
   CPVK_OP_BEGIN_RENDER,
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
};

struct cpvk_op {
   enum cpvk_op_kind kind;
   /* BEGIN_RENDER only: the attachment's sample count, which cp_fb_desc does
    * not carry because the Gallium adapter passes it beside the desc. */
   unsigned fb_samples;
   union {
      struct cpvk_draw draw;
      struct cpvk_clear clear;
      struct cpvk_copy copy;
      struct cp_fb_desc fb;
      struct cpvk_query_op query;
      struct cpvk_dispatch dispatch;
   };
};

struct cpvk_cmd_buffer {
   struct vk_command_buffer vk;
   struct cpvk_pipeline *pipeline;
   CUdeviceptr addrs[16];
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
   struct cpvk_image *resolve_src, *resolve_dst;
   VkRect2D resolve_area;
   struct cp_viewport_state viewport;
   struct cp_rect scissor;
   uint64_t vb_base[16];
   unsigned num_vb;
   uint64_t desc_hash[16];
   unsigned fb_samples;
   const void *index_ptr;
   unsigned index_size;
   unsigned char push[CPVK_MAX_PUSH_BYTES];
   unsigned push_size;
   /*
    * Grown rather than capped. A fixed array silently dropped everything past
    * its end, which is the failure that looks like a driver bug in whatever
    * came after: bloom records well over sixty operations and rendered a
    * completely black frame.
    */
   struct cpvk_op *ops;
   unsigned num_ops, max_ops;

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
   CUdeviceptr desc_arena;
   struct cpvk_descriptor *desc_arena_host;
   size_t desc_arena_size, desc_arena_used;
   /* Arenas outgrown while draws still point into them; freed at reset. */
   CUdeviceptr *desc_retired;
   unsigned num_desc_retired, max_desc_retired;
};

void cpvk_execute_draw(struct cpvk_device *dev, const struct cpvk_draw *d);
void cpvk_execute_clear(struct cpvk_device *dev, const struct cpvk_clear *c);
void cpvk_execute_copy(struct cpvk_device *dev, const struct cpvk_copy *c);
void cpvk_execute_query(struct cpvk_device *dev, const struct cpvk_query_op *q);
void cpvk_execute_begin_render(struct cpvk_device *dev, const struct cp_fb_desc *fb,
                               unsigned samples);

extern const struct vk_command_buffer_ops cpvk_cmd_buffer_ops;
extern const struct vk_sync_type *const cpvk_sync_types[];

VkResult cpvk_execute_dispatch(struct cpvk_device *dev,
                               const struct cpvk_dispatch *d);
VkResult cpvk_execute_cmd_buffer(struct cpvk_device *dev,
                                 struct cpvk_cmd_buffer *cmd);

struct cpvk_pipeline {
   struct vk_object_base base;
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
