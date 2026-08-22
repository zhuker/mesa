/*
 * The renderer: the CUDA draw pipeline, its state, its batching and its
 * episode machinery, with no Gallium in it.
 *
 * This file and cp_context.h were one file until the front end was separated
 * from what it drives. Everything here is reachable from a Vulkan driver that
 * has a cp_device; everything Gallium-shaped stayed behind in cp_context.h
 * with struct cp_gallium.
 */

#ifndef CP_RENDERER_H
#define CP_RENDERER_H

#include "cp_debug.h"
#include "cp_device.h"
#include "cp_draw_types.h"
#include "cp_draw_packet.h"

#include <cuda.h>
#include <stdbool.h>
#include <stdio.h>

#include "kernels/cp_rast_types.h"

#include "cp_shader_abi.h"

#define CP_MAX_SHADER_BUFFERS 16
/* CP_MAX_CONST_BUFFERS is the shader ABI's, in cp_shader_abi.h. */
/*
 * Distinct sampler states, not VkSampler objects: identical states share one
 * entry because shader specialization compares the state, not the index. The
 * count matches the advertised maxSamplerAllocationCount so a legal
 * application cannot exhaust the table before that limit.
 */
#define CP_MAX_SAMPLERS       4096

/*
 * Draw batching.
 *
 * Consecutive draws that would produce the same pixels in any order are held
 * back and submitted as one. What makes that safe is the visibility buffer:
 * with blending off it resolves the nearest fragment per pixel by atomicMin,
 * which does not care in what order the fragments arrived, so a batch needs no
 * submission-order sort key. See cp_batch_eligible() and struct cp_batch_key for the
 * whole list of conditions, all of which have to hold.
 *
 * The point is grid size. dynamicuniformbuffer draws 125 twelve-triangle cubes
 * a frame, and every stage is sized to one of them: cp_rasterize_stage2 runs on
 * five blocks of a 170-SM card and owns a quarter of the frame. A batch gives
 * the same kernels a grid worth launching, and pays the framebuffer-sized fixed
 * costs — the visibility clear, the interpolator, the fragment grid — once for
 * the batch instead of once per draw.
 *
 * The cap is on draws and on triangles both, because the buffers a draw sizes
 * from its triangle count (the clipper's output, four per input triangle)
 * scale with the batch and the arena has a hard limit.
 */
#define CP_MAX_BATCH_TRIS  (256 * 1024)

/*
 * Everything that has to be the same for two consecutive draws to be merged.
 *
 * Built by cp_batch_build_key() and compared with memcmp, so that adding a
 * piece of state to the driver and forgetting it here is the one mistake this
 * cannot make quietly: the key is filled from a zeroed struct by a single
 * function, and anything it does not copy is simply not a merge condition.
 * What is deliberately *absent* is the vertex stage's uniform bindings and the
 * index range — those are what a batch is allowed to differ in, and what the
 * tables behind the shader's argument block exist to carry.
 *
 * CUDAPIPE_DEBUG_BATCHDIFF names the fields two draws disagree on, which is
 * the only way to tell which of them is worth attacking next.
 */
struct cp_batch_key {
   const void *vs, *fs;

   /* Framebuffer and the driver's own per-pass buffers. */
   const void *cbuf_texture, *zs_texture, *color_data;
   uint64_t visbuf, depthbuf;
   uint32_t fb_w, fb_h, fb_nr_cbufs, fb_samples, cbuf_format;

   /* The draw itself. The index *range* is not here: a batch carries a table
    * of them and cp_vertex_fetch searches it, which is what lets a scene of
    * different meshes out of one buffer merge at all. Neither are the draw
    * parameters — gl_BaseVertex, gl_BaseInstance, gl_DrawID resolve per draw
    * through the parameter rows at args[7]. start_instance stays: the fetch
    * kernel's instance-divisor gather reads it as one scalar. */
   uint32_t mode, index_size, start_instance;
   const void *index_resource;

   struct cp_rect scissor;
   uint32_t blend_enabled;
   /* Pipeline state, as the front end describes it; see cp_draw_types.h.
    * Whole structs go in here, so a field added upstream is covered without
    * anything naming it. */
   uint8_t state[CP_BATCH_STATE_BYTES];

   /* Vertex input. The buffer *bindings* are absent: a batch carries one row
    * of resolved per-element base addresses per draw, so draws bound to
    * different vertex buffers merge. The element layout is inside `state`;
    * the counts stay here — they shape the gather itself. */
   uint32_t num_vertex_elements, vertex_stride, num_vertex_buffers;

   /* Binding *counts* only: both stages carry their per-draw pointer tables,
    * but each launch fills its table's rows this many entries wide, so the
    * width itself has to agree across a batch. */
   uint32_t num_fs_ubos, num_vs_ubos;
   uint64_t sampler_table;
   uint32_t num_samplers;
};

/*
 * The Gallium object, and the renderer inside it.
 *
 * pipe_context comes first because the driver's entry points cast a
 * pipe_context straight to this. Everything the pipeline itself uses lives in
 * cp_context beside it, which is what the native Vulkan driver will construct
 * without a pipe_context existing at all.
 */
struct cp_gallium;
struct cp_abuf;

/* A native frontend may stage CPU-written descriptor snapshots separately
 * from the device addresses stored in shader UBO rows.  The fragment sampler
 * specializer is the only renderer code which dereferences such an address
 * on the CPU, so queue submission publishes the active mappings here. */
struct cp_host_map {
   CUdeviceptr dev;
   const void *host;
   size_t size;
};
#define CP_MAX_HOST_MAPS 1024

struct cp_abuf_dbg_state {
   CUdeviceptr blk_offsets, blk_counts, quad_prim, peel_mask, counters;
   /*
    * Step 3b: where the peel path deposits its shaded colours. One slot per
    * (pixel, primitive) — the A-buffer's own indexing — so the two paths can
    * be compared without either of them agreeing on an order.
    */
   CUdeviceptr frags, offsets, counts, colors, writes;
   uint32_t capacity;
};


struct cp_draw_batch {
   struct cp_draw_state state;
   struct cp_render_scope scope;
      bool pending;
      bool compact_rows;
      unsigned ndraws;
      /* Triangles over the whole batch, which is what the clipper's output
       * buffer is sized from and so what CP_MAX_BATCH_TRIS caps. */
      unsigned tris;
      struct cp_batch_key key;
      struct cp_draw_call info;
      /* One index range per merged draw; cp_draw_execute_batch() turns these into
       * the slice table cp_vertex_fetch searches. */
      struct cp_draw_range draws[CP_MAX_BATCH_DRAWS];
      uint32_t instance_counts[CP_MAX_BATCH_DRAWS];
      unsigned drawid_offset;
      /* gl_DrawID per merged draw — the offset recorded when it joined, for
       * the per-draw parameter rows at args[7]. */
      uint32_t draw_ids[CP_MAX_BATCH_DRAWS];
      /* One row of vertex-stage uniform pointers per draw, in the layout the
       * shader indexes: row * CP_ARG_UBO_STRIDE + binding. */
      uint64_t vs_ubos[CP_MAX_BATCH_DRAWS * CP_MAX_CONST_BUFFERS];
      /*
       * The same for the fragment stage, and only a blended batch fills it.
       * A blended draw is rendered by the A-buffer, which sorts on the
       * primitive index — so merging several of them is legal where merging
       * opaque draws with different fragment bindings still is not, because
       * an opaque batch has no ordering to carry and nothing to hang a
       * per-primitive row off. See cp_batch_abuf_ok().
       */
      bool blended;
      uint64_t fs_ubos[CP_MAX_BATCH_DRAWS * CP_MAX_CONST_BUFFERS];
      /* One row of resolved per-element vertex-buffer bases per draw, so
       * draws bound to different vertex buffers merge — see
       * cp_vertex_fetch_args.elem_bases. */
      uint64_t vb_bases[CP_MAX_BATCH_DRAWS * CP_VB_TABLE_STRIDE];
      /* The scissor each draw was recorded under, for the per-draw clip
       * rectangles — see cp_rasterize_args.clip_rects. */
      struct cp_rect scissors[CP_MAX_BATCH_DRAWS];
};

struct cp_context {

   /* Per-renderer A-buffer storage: every CUDA pointer belongs to this
    * context's device instead of to process-global state. */
   struct cp_abuf *abuf;
   struct cp_abuf_dbg_state abuf_dbg;

   /* The device, not the screen: four fields the pipeline reads, and no
    * pipe_screen behind them. A Vulkan front end supplies one directly. */
   struct cp_device *screen;

   /*
    * Draws held back for merging. `pending` means one or more draws have been
    * accepted and nothing has run yet, so every path that observes rendering —
    * a flush, a readback, a clear, a blit, a dispatch — has to call
    * cp_batch_flush() before it looks.
    */
   struct cp_draw_batch batch;

   /*
    * A pass episode: consecutive blended batches sharing one A-buffer build,
    * one drain and one composite. Each flushed blended batch becomes a
    * *segment* — its vertex stage and A-buffer count rasterization run at
    * append time, into per-pixel lists shared by the whole episode, with its
    * primitive ids offset so that the sort's ascending order is submission
    * order across segments. Everything else — scan, fill, sort, quad merge,
    * the drain, per-segment shading, the composite — happens at
    * cp_pass_finish(). See cp_batch_flush_why() for what may defer and what
    * must finish.
    */
   struct cp_pass_seg {
      /* Saved as of the end of the vertex stage; the finish patches the
       * A-buffer fields and relaunches the fill stages from it. */
      struct cp_rasterize_args rast;
      struct cp_rast_queues queues;
      unsigned rast_num_triangles;
      unsigned num_triangles;
      uint32_t prim_base;           /* first episode-global primitive slot */
      uint32_t prim_slots;          /* slots this segment occupies */
      unsigned prim_shift;
      /* Complete immutable shading/fallback snapshot. */
      struct cp_draw_batch batch;
      CUdeviceptr slices_dev;
   } *pass_segs;                    /* [CP_PASS_MAX_SEGS], at context create */

   struct {
      struct cp_render_scope scope;
      bool scope_open;
      unsigned nsegs;
      unsigned total_draws;
      uint32_t next_prim;           /* running global slot base */
      bool appending;               /* a segment append is inside execute */
      bool append_failed;           /* the append could not take the path */
      bool opaque;                  /* shared-visbuf opaque run, not A-buffer */
      unsigned w, h;                /* the episode's framebuffer */
   } pass;

   /* A merged shading group's concatenated fs-UBO rows, staged here before
    * the upload; sized for the worst episode, allocated on first use. */
   uint64_t *pass_group_ubos;

   /*
    * CUDAPIPE_TILE_CENSUS. The accumulation runs for a whole framebuffer
    * bind — the interval a tile renderer could hold colour and depth on
    * chip — so the per-tile arrays are persistent and reset at the bind
    * rather than allocated per episode, and the ordering key is a
    * pass-global draw sequence rather than an episode-local primitive id.
    * The histogram is managed memory the device accumulates into, so no
    * per-pass readback exists.
    */
   CUdeviceptr tile_census_hist;
   CUdeviceptr tile_census_mask, tile_census_quads;
   CUdeviceptr tile_census_smin, tile_census_smax;
   unsigned tile_census_alloc;      /* tiles the arrays are sized for */
   unsigned tile_census_tiles_x, tile_census_tiles_y;
   unsigned tile_census_w, tile_census_h;
   bool tile_census_open;           /* a bind is being accumulated */
   unsigned tile_census_seq;        /* pass-global draw order */
   unsigned tile_census_nfs;
   struct cp_shader_binary *tile_census_fs[64];
   CUdeviceptr tile_census_refs;
   uint64_t tile_census_marks;      /* episodes and opaque runs marked */
   uint64_t tile_census_shaders;    /* their shaders, summed */
   uint64_t tile_census_passes;
   uint64_t tile_census_solo;       /* draws that took neither marked path */
   uint64_t tile_census_marked_draws;
   /* Whether a bind really is the interval a tile could stay resident: what
    * touched the attachment while one was open, by kind. */
   uint64_t tile_census_binds_cut;
   uint64_t tile_census_cut_map, tile_census_cut_copy;
   uint64_t tile_census_cut_flush, tile_census_cut_compute;
   uint64_t tile_census_bind_draws; /* draws inside binds, summed */
   bool tile_census_cut;            /* this bind has been interrupted */

   /*
    * The episode's side streams. Segment counts, fill relaunches and shades
    * are mutually independent — the per-pixel counts and cursors are
    * order-free atomics whose order the sort erases, and the dense shading
    * slots are disjoint — so they fan out across these and join at the
    * phases that read across segments. Stream k also owns its own rasterizer
    * queue set, since the queues are rebuilt per launch group and two
    * streams may be in one concurrently.
    */
#define CP_PASS_STREAMS 8
/* Arena generations cp_flush ping-pongs through; see the scratch struct. */
#define CP_FLUSH_GENS 8
   CUstream seg_streams[CP_PASS_STREAMS];
   CUevent seg_ev[CP_PASS_STREAMS];
   CUevent pass_gate;
   bool pass_streams_ready;
   struct cp_queue_set {
      CUdeviceptr nontrivial, huge_tiles, counts;
   } seg_qsets[CP_PASS_STREAMS];
   /* What cp_draw_execute builds its queue struct from: the context-wide set
    * normally, a segment stream's own during an append. */
   struct cp_queue_set cur_qset;

   /*
    * What the fragment shader launches of the draw now running should hand to
    * CP_ARG_SLOT_UBO_TABLE. Set once at the top of cp_draw_execute_batch() so that
    * it cannot carry from one draw to the next, and read by
    * cp_fs_launch_shader() — which both shading paths go through, and which is
    * three call frames below where the batch is known.
    */
   struct cp_host_map host_maps[CP_MAX_HOST_MAPS];
   unsigned num_host_maps;

   struct cp_fs_batch {
      const uint64_t *ubos;   /* rows of CP_ARG_UBO_STRIDE, or NULL */
      unsigned ndraws;
      /* Where the interpolator looks a primitive's draw up; the same table
       * cp_vertex_fetch searches, uploaded once per draw. */
      CUdeviceptr slices;
      /* How to get from a primitive index back to an input triangle: 2 when
       * the clipper laid its output out stably, 0 when it did not run. Taken
       * from what was launched, not from what was intended. */
      unsigned prim_shift;
   } fs_batch;

   /*
    * The stream every launch, memset and copy in the frame path goes on.
    *
    * Created with CU_STREAM_DEFAULT rather than CU_STREAM_NON_BLOCKING on
    * purpose: a default-flagged stream still synchronises implicitly with the
    * legacy NULL stream, so anything left on NULL — an allocation-time clear,
    * or a path added later that forgets to thread this through — stays
    * correctly ordered against the rest instead of racing it. The point of
    * having a real stream is to be able to express overlap at all; nothing
    * yet depends on it being isolated.
    */
   CUstream stream;

   /*
    * Per-stage draw timing under CUDAPIPE_DEBUG_TIME, on CUDA events recorded
    * on the stream above. A growable pool because every interval of a draw has
    * to be recorded before any can be read, and a blended draw runs the
    * shading stages once per layer. Empty unless the variable is set.
    */
   struct cp_stage_timer {
      CUevent  *events;
      int      *stages;   /* which stage the interval ending here belongs to */
      unsigned  num, cap;
   } timer;

   /* Visibility buffer, rebuilt per draw: it resolves which triangle of the
    * current draw wins each pixel. */
   CUdeviceptr visbuf;
   CUdeviceptr reject;      /* per-pixel discarded triangles, CP_DISCARD_LAYERS deep */
   CUdeviceptr resolved;    /* per-pixel byte: a fragment has been written */
   unsigned visbuf_samples; /* what the visibility and depth buffers were sized for */
   CUdeviceptr peel_next;   /* per-pixel: first primitive not yet blended */
   CUdeviceptr peel_any;    /* one uint32, managed: a pass found work to do */
   unsigned visbuf_w, visbuf_h;
   /*
    * What the five framebuffer-sized buffers were actually allocated for, as
    * opposed to what is bound now. They are kept at the largest size seen and
    * only reallocated when that grows, because an application renders more
    * than one size: this capture runs fifteen render passes a frame at
    * 1280x720 and 160x90, and an equality test against the bound size
    * reallocated all five in both directions 9.5 times a frame — 356 GB of
    * churn over a replay, 3.4 ms a frame, every microsecond of it with the
    * device idle.
    *
    * Two counts because they do not scale together: visbuf and depthbuf are
    * per sample, the other three are per pixel.
    */
   size_t fb_cap_px;
   size_t fb_cap_px_samples;

   /* Depth buffer for the render pass, one sortable uint32 per pixel. This is
    * what carries occlusion across draws. */
   CUdeviceptr depthbuf;
   unsigned depthbuf_w, depthbuf_h;
   bool depthbuf_cleared;

   /* Adaptive rasterizer queues (allocated once, reused across draws) */
   CUdeviceptr rast_nontrivial;       /* uint32_t[CP_MAX_NONTRIVIAL] */
   CUdeviceptr rast_huge_tiles;       /* cp_tile_pair[CP_MAX_HUGE_TILES] */
   /* Both queue counters, adjacent in one allocation so a pass zeroes them
    * with one cuMemsetD32. rast_counts owns the memory; the two below point
    * into it and are not freed. */
   CUdeviceptr rast_counts;
   CUdeviceptr rast_nontrivial_count; /* atomic uint32_t, = rast_counts[0] */
   CUdeviceptr rast_huge_count;       /* atomic uint32_t, = rast_counts[1] */



   struct cp_shader_binary *compute_shader;

   /* Texture state for FS */
   CUtexObject tex_objects[32];
   unsigned num_tex_objects;
   struct {
      void *data;
      unsigned width, height;
      unsigned row_stride;
      unsigned pixel_size;
   } tex_resources[32];

   struct {
      void *buffer;
      unsigned buffer_size;
   } compute_ssbos[CP_MAX_SHADER_BUFFERS];
   unsigned num_compute_ssbos;

   struct {
      void *buffer;
      unsigned buffer_size;
      CUdeviceptr managed_copy; /* Device buffer for user_buffer data */
      unsigned managed_size;
      bool user_copy;
   } compute_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_compute_ubos;


   /* Device-visible table of deduplicated sampler states. Descriptors refer to
    * entries by index; see cp_register_sampler(). */
   CUdeviceptr sampler_table;
   struct cp_sampler_info sampler_table_host[CP_MAX_SAMPLERS];
   unsigned num_samplers;

   /*
    * Sampler-specialisation counters, reported at teardown under
    * CUDAPIPE_SPEC_STATS. The specialiser recognises an exact IR shape, so a
    * lowering change can stop it firing with no error at all; these are what
    * make that visible before the next sweep blames something else.
    */
   struct {
      uint64_t launches;             /* fragment launches decided */
      uint64_t specialised;          /* of those, using a specialised kernel */
      uint64_t shaders;              /* distinct fragment shaders launched */
      uint64_t shaders_unmatched;    /* some sampler handle was not matchable */
      uint64_t shaders_unmatchable;  /* samples textures, matched none at all */
   } spec;

   /* GPU-resident pipeline state — managed memory, written by CPU on state
    * changes, read by GPU kernels during draws. */
   /* Device-only arena for per-draw scratch buffers. CPU never touches this
    * memory — just tracks offsets as integers. See cp_upload(). */
   CUdeviceptr arena_base;
   size_t arena_size;
   size_t arena_offset;

   /* Pinned host staging for cp_upload(). Pinned rather than ordinary malloc
    * because only a pinned source lets cuMemcpyHtoDAsync be genuinely
    * asynchronous; from pageable memory the driver synchronises the stream
    * first, which per draw would cost more than the upload saves. */
   void *upload_host;
   size_t upload_size;
   size_t upload_offset;

   /*
    * The arenas rewind at every cp_flush, and the flush no longer drains the
    * device first: the managed scratch and the upload ring are split into
    * CP_FLUSH_GENS generations, the flush records a retire event behind the
    * generation it is leaving and rewinds into the one whose event — from
    * CP_FLUSH_GENS-1 flushes ago — it waits, which is almost always already
    * signalled. Reuse without reallocation, which is the property the old
    * flush-time drain existed to keep (see the measured regression quoted at
    * the drain it replaced). flush_gens is 1 under CUDAPIPE_FLUSH_DRAIN,
    * which restores the drain and the old single-generation behaviour.
    */
   unsigned flush_gens;
   CUevent flush_retire[CP_FLUSH_GENS];
   bool flush_retire_recorded[CP_FLUSH_GENS];

   struct {
      CUdeviceptr base[CP_FLUSH_GENS];
      size_t size[CP_FLUSH_GENS];
      size_t used;
      size_t peak;
      unsigned current;
      CUdeviceptr overflow[64];
      unsigned num_overflow;
   } scratch;

   /*
    * The same, in memory only the device can reach.
    *
    * Everything a draw shades through — the vertex shader's output, the
    * interpolated fragment inputs, the shaded results — is written by one
    * kernel and read by the next, and never touched by the host at all. Out of
    * managed memory those buffers still cost as if it did: the arena's bump
    * pointer climbs, so each draw lands on addresses whose pages are not yet on
    * the device, and the first kernel to write one pays the migration. That
    * cost lands on the vertex shader, which is the first to touch its output.
    *
    * cuMemAlloc memory has nowhere else to be, so there is nothing to fault.
    * What the host does write — assembled vertex ids, packed positions — goes
    * through cp_upload() instead, and the arena below is for that.
    */
   struct {
      CUdeviceptr base;
      size_t size;
      size_t used;
      size_t peak;
      CUdeviceptr overflow[64];
      unsigned num_overflow;
   } dscratch;

   /*
    * A batch's per-draw uploads are live until its launches are issued.
    *
    * Each staged draw puts its push block in the upload arena and records the
    * address in its uniform row. cp_scratch_reset() rewinds that arena, and
    * cp_draw_execute_batch() calls cp_scratch_begin() before it uploads anything of
    * its own -- so a reclaim there hands the batch's own slice table and
    * argument block the addresses the push blocks are sitting at, and every
    * merged draw reads whatever landed on top of it.
    *
    * Set when a draw joins a batch, cleared when cp_draw_execute_batch() returns.
    */
   bool batch_uploads_live;
};



void cp_batch_flush(struct cp_context *cp);

void cp_batch_flush_why(struct cp_context *cp, const char *why);

void cp_batch_flush_defer_why(struct cp_context *cp, const char *why);

void cp_pass_finish(struct cp_context *cp);

void cp_tile_census_end_pass(struct cp_context *cp);

/*
 * Say so when a CUDA call fails, once per site.
 *
 * This driver's failure mode is silence, and every expensive bug found in it
 * so far cost what it did for that reason rather than for its own difficulty.
 * A NULL from cp_allocate_memory became a segfault inside lavapipe, three
 * frames from the kernel that actually faulted. A clip pass was skipped
 * because its scratch allocation returned NULL. The A-buffer switched itself
 * off. A shader read an intrinsic the backend did not implement and got undef.
 * In each case the driver knew, and did not say.
 *
 * Once per site rather than once per failure, because these sit on paths that
 * run hundreds of times a frame: a real fault would otherwise bury its own
 * first line, which is the one worth reading. Nothing here changes behaviour —
 * a caller that can carry on still carries on.
 */
#define CP_CU_WARN(err, what)                                                 \
   do {                                                                       \
      CUresult _cp_e = (err);                                                 \
      if (_cp_e != CUDA_SUCCESS) {                                            \
         static bool _cp_said;                                                \
         if (!_cp_said) {                                                     \
            const char *_cp_n = NULL, *_cp_s = NULL;                          \
            _cp_said = true;                                                  \
            cuGetErrorName(_cp_e, &_cp_n);                                    \
            cuGetErrorString(_cp_e, &_cp_s);                                  \
            fprintf(stderr, "cudapipe: %s failed at %s:%d — %s (%d)%s%s\n",   \
                    (what), __func__, __LINE__,                               \
                    _cp_n ? _cp_n : "unknown", (int)_cp_e,                    \
                    _cp_s ? ": " : "", _cp_s ? _cp_s : "");                   \
            if (_cp_e == CUDA_ERROR_ILLEGAL_ADDRESS ||                        \
                _cp_e == CUDA_ERROR_LAUNCH_FAILED)                            \
               fprintf(stderr, "cudapipe:   this error is sticky — every "    \
                       "later CUDA call fails too, so the first report is "   \
                       "the one that names the cause. Re-run with "           \
                       "CUDA_LAUNCH_BLOCKING=1, or under compute-sanitizer "  \
                       "to name the kernel and the address.\n");              \
         }                                                                    \
      }                                                                       \
   } while (0)

/*
 * A launch that says so when it fails.
 *
 * Worth knowing what this can and cannot tell you. A launch is asynchronous,
 * so the error it returns is rarely its own: it is whatever sticky error a
 * previous kernel left behind, and the first launch to report is the first one
 * issued after the fault, not the one that caused it. That is why the driver
 * reported "VS launch failed: 700" for a fault in vertex fetch.
 *
 * What it is good for is the moment of transition — turning "a segfault
 * somewhere in libc, three frames later" into a line naming a CUDA error and
 * the tools that find the kernel. Synchronous failures, a bad grid or too much
 * shared memory, it does report exactly.
 */
#define CP_LAUNCH(...) CP_CU_WARN(cuLaunchKernel(__VA_ARGS__), "cuLaunchKernel")

/* Slots the quad stream's own shading pass may use, four per quad. Bounds the
 * fragment shader's input and output buffers, which at five varyings are about
 * 90 bytes a slot. */
#define CP_ABUF_MAX_SHADE_SLOTS (16u * 1024u * 1024u)

/*
 * A pass episode's per-segment shading: which slice of the grouped quad list
 * this segment shades, and — filled in on success — the dense arrays its
 * shader produced, for the episode's one composite to resolve through.
 */
struct cp_abuf_seg_shade {
   CUdeviceptr quad_list;      /* the episode's grouped quad indices */
   uint32_t quad_list_base;    /* this segment's first entry */
   CUdeviceptr quad_list_base_dev;
   CUdeviceptr num_quads_dev;
   uint32_t prim_base;         /* subtracted from global primitive ids */
   /* A merged group: the launch spans several segments, each quad resolving
    * its own vertex stream and slice table through this table; prim_base
    * above is then only the placeholder the table overrides. */
   CUdeviceptr ranges;         /* struct cp_seg_range[num_ranges] */
   uint32_t num_ranges;
   /* out */
   CUdeviceptr fs_out, coverage, discard;
   uint32_t fs_out_stride, num_slots;
};

/* Deepest pixels first, for the full-depth half of the comparison. */
struct cp_abuf_deep { uint32_t count, pixel; };

#define CP_ABUF_DEEP_PIXELS 1000
#define CP_ABUF_MAX_GROWTHS  8u
#define CP_ABUF_GROW_AT      0.75    /* fraction of capacity that triggers a grow */
int cp_abuf_cmp_deep(const void *a, const void *b);

/* The A-buffer: its arrays, their sizing, the prefix scans over them, and
 * the verification the peel loop is compared against. One per process, like
 * the arrays it owns. */
struct cp_abuf {
   int enabled;             /* -1 unknown, 0 off, 1 on */
   int verify;
   /* Whether an eligible draw is rendered by this path instead of by the peel
    * loop, rather than merely having its lists built and checked beside it. */
   int composite;
   int timing;              /* print the per-draw event breakdown */
   int debug;               /* explain on stderr why a draw is not eligible */
   /* Layers the composite stops after, 0 for all of them. Only for asking what
    * the peel loop's own CP_BLEND_LAYERS truncation was worth; the path has no
    * such cap and is not meant to acquire one. */
   unsigned max_layers;
   bool ready;              /* per-pixel buffers allocated for w x h */
   bool disabled;           /* something refused it; do not try again */
   unsigned w, h;
   /*
    * What the framebuffer-sized allocations actually hold, as against what the
    * current framebuffer needs. Grow-only: a render pass at a smaller size
    * reuses the larger arrays and only recomputes the derived counts below.
    * See cp_abuf_setup() for why it is not a realloc on every change.
    */
   size_t cap_pixels;       /* pixels the per-pixel arrays are sized for */
   unsigned cap_blocks;     /* 2x2 blocks the per-block arrays are sized for */
   size_t cap_log_pixels;   /* pixels the verification's peel log is sized for */
   unsigned resizes;        /* times the arrays have been grown */
   bool events_ready;       /* the CUevents are created once, not per size */

   /* Per-pixel, and the scan's per-level partial sums. */
   CUdeviceptr counts, offsets, cursor;
   CUdeviceptr sum1, sum1x, sum2, sum2x, sum3;
   unsigned nb1, nb2, nb3;

   /* The sort's worklist, and the two failure counters the kernels bump. */
   CUdeviceptr list, list_count, overflow, long_runs;

   /* Pass-episode segment quad counts, carved out of `counters` behind the
    * six words so the episode's one drain reads everything in one copy. */
   CUdeviceptr seg_counts;

   /* The composite's own worklist: every pixel with at least one fragment,
    * where the sort's holds only those with more than one. Separate arrays
    * rather than one with the looser threshold, so the sort keeps costing what
    * it was measured to cost. */
   CUdeviceptr clist, clist_count;

   /* The fragment array itself: sized from the first draw's count with
    * CP_ABUF_HEADROOM to spare, and grown between draws when a later count
    * comes within CP_ABUF_GROW_AT of it. See the constants above for why the
    * growth is bounded the way it is. */
   CUdeviceptr frags;
   /*
    * The single-pass build's records: one (pixel << 32 | prim) uint64 per
    * fragment, appended by the count pass through rec_cursor (one uint32,
    * carved out of `counters` past the segment quad counts) and replayed by
    * cp_abuf_fill_recs in place of the second rasterization. Sized and freed
    * with `frags`, and the same entry capacity bounds both.
    */
   CUdeviceptr recs, rec_cursor;
   unsigned capacity;
   unsigned growths;        /* times it has been grown this process */
   unsigned peak;           /* largest population any draw has counted */
   bool grow_capped;        /* a growth was refused; do not ask again */
   /*
    * A population that wants the arrays bigger, recorded by the draw that
    * counted it and acted on by the next eligible draw. The count is no longer
    * read on the host until after the fill has been issued — see the drain
    * that used to be between them — so by the time a growth is known to be
    * wanted, this draw's own kernels are already reading the arrays it would
    * free. The next draw's count pass reads none of them, and is where the
    * resize is safe.
    */
   uint32_t grow_to;

   /* What the peel loop selected: CP_ABUF_LOG_LAYERS for every pixel, and the
    * full CP_BLEND_LAYERS for the deepest CP_ABUF_DEEP_PIXELS. */
   CUdeviceptr log, deep_list, deep_log;

   /*
    * Step 3a: the quad stream. Per 2x2 block, a worklist of the blocks with
    * any coverage and a count of the distinct primitives in them; per quad,
    * that primitive and the 4-bit mask of the block's pixels it covers.
    * bsum3 holds the quad total, the fill's overflow counter and the
    * interpolator's four debug counters.
    */
   CUdeviceptr blk_counts, blk_offsets, blk_list, blk_list_count;
   CUdeviceptr bsum1, bsum1x, bsum2, bsum2x, bsum3, quad_overflow, dbg;
   /* The single allocation sum3, bsum3 and clist_count are carved out of,
    * so that the per-draw readback is one copy rather than three. */
   CUdeviceptr counters;
   unsigned bnb1, bnb2, bnb3, nblocks, quad_width;
   CUdeviceptr quad_prim, quad_mask, peel_mask, quad_block;
   unsigned quad_capacity;

   /*
    * Step 4: for each A-buffer slot, the shading slot holding that fragment's
    * colour. Written by the merge, which is the only place both indices are in
    * hand at once, and read by the composite as it walks a pixel's run.
    */
   CUdeviceptr shade_slot;

   /*
    * Step 3b: the shaded colours, one slot per (pixel, primitive), written by
    * both paths and compared element-wise. `writes` counts rather than flags,
    * so a slot two fragments claimed is visible instead of passing for
    * agreement.
    */
   CUdeviceptr colors_abuf, writes_abuf, colors_peel, writes_peel;
   bool colors_ready;

   CUevent ev[17];

   uint32_t *h_counts, *h_offsets, *h_cursor, *h_frags, *h_log, *h_deep_log;
   uint32_t *h_deep_list;
   uint32_t *h_blk_counts, *h_blk_offsets, *h_quad_prim, *h_peel_mask;
   unsigned char *h_quad_mask;
   float *h_colors_abuf, *h_colors_peel;
   uint32_t *h_writes_abuf, *h_writes_peel;

   unsigned verified, verify_max;
   unsigned seq;
};
bool cp_abuf_batch_enabled(void);
bool cp_abuf_enabled(struct cp_abuf *ab);
void cp_abuf_mark(struct cp_abuf *ab, CUevent ev, CUstream stream);
void cp_abuf_report(struct cp_abuf *ab);
void cp_abuf_cleanup(struct cp_abuf *ab);
void cp_abuf_scan(struct cp_context *cp, struct cp_device *screen, struct cp_abuf *ab, unsigned n);
void cp_abuf_scan_n(struct cp_context *cp, struct cp_device *screen, CUdeviceptr in, CUdeviceptr out, CUdeviceptr s1, CUdeviceptr s1x, CUdeviceptr s2, CUdeviceptr s2x, CUdeviceptr s3, unsigned n, unsigned nb1, unsigned nb2, unsigned nb3, CUdeviceptr clamp_counts, uint32_t clamp_capacity, CUdeviceptr clamp_overflow);
bool cp_abuf_setup(struct cp_abuf *ab, unsigned w, unsigned h);
bool cp_abuf_size_arrays(struct cp_abuf *ab, uint32_t total);
void cp_abuf_verify(struct cp_abuf *ab, unsigned w, unsigned h, uint32_t total, unsigned passes_run, unsigned deep_n, uint32_t overflow, uint32_t long_runs);
void cp_abuf_verify_quads(struct cp_abuf *ab, unsigned w, unsigned h, uint32_t total_frags, uint32_t total_quads, unsigned passes_run, uint32_t quad_overflow, const uint32_t *dbg);
void cp_census_dump(const char *what, unsigned draw_seq, unsigned peel_seq, unsigned num_triangles, unsigned num_samples, const uint32_t *counts, unsigned w, unsigned h);
bool cp_census_enabled(void);

/* One assembled vertex: which vertex of the bound buffers it reads, and which
 * instance it belongs to. */
struct cp_vertex_ref {
   uint32_t vertex;
   uint32_t instance;
};

uint32_t cp_depth_to_sortable(float depth);
void cp_clear_depthbuf(struct cp_context *cp, float depth);
void cp_stage_resolve(struct cp_context *cp, double *ms);
int32_t cp_slot_for_location(const unsigned *locations, unsigned count, unsigned location);
unsigned cp_triangles_for_draw(enum mesa_prim mode, unsigned count);
struct cp_vertex_ref * cp_build_vertex_refs(const struct cp_draw_call *info, const struct cp_draw_range *draws, unsigned num_draws, unsigned instance_count, const void *ib_base, unsigned num_triangles);
uint32_t cp_cull_mode(const struct cp_raster_state *rs);
struct cp_blend_desc cp_blend_desc_for(const struct cp_draw_state *state);

bool cp_abuf_shade(struct cp_context *cp,
                   const struct cp_draw_state *state,
                   const struct cp_render_scope *scope,
                   const struct cp_draw_call *info, struct cp_abuf *ab,
                   CUdeviceptr positions, CUdeviceptr vs_output_buf,
                   unsigned w, unsigned h, float vp_scale_x, float vp_scale_y,
                   float vp_trans_x, float vp_trans_y, uint32_t num_quads,
                   uint32_t num_covered, bool record_colors, void *color_data,
                   bool composite, float *t_interp, float *t_shade,
                   float *t_composite, struct cp_abuf_seg_shade *seg);

/*
 * TEMPORARY (CUDAPIPE_ABUFFER): the merged quad array, for the one draw whose
 * peel loop is being compared against it. Zero for every other draw, and for
 * every draw once the verification budget is spent, so the instrumented
 * interpolator is not carried by frames that are only being timed.
 */


void cp_draw_execute_batch(struct cp_context *cp,
                           const struct cp_draw_batch *batch);
void cp_shade_fragments(struct cp_context *cp,
                        const struct cp_draw_state *state,
                        const struct cp_render_scope *scope,
                        const struct cp_draw_call *info, CUdeviceptr visbuf,
                        CUdeviceptr positions, CUdeviceptr vs_output_buf,
                        unsigned num_triangles, unsigned w, unsigned h,
                        void *color_data, float vp_scale_x, float vp_scale_y,
                        float vp_trans_x, float vp_trans_y, CUdeviceptr reject,
                        CUdeviceptr resolved, unsigned reject_pass,
                        CUdeviceptr seg_ranges, unsigned num_seg_ranges);


bool cp_batch_order_free(const struct cp_draw_state *state);

void cp_batch_begin_packet(struct cp_context *cp,
                           const struct cp_draw_packet *packet,
                           const struct cp_batch_key *key, bool blended);
void cp_batch_record_packet(struct cp_context *cp,
                            const struct cp_draw_packet *packet,
                            unsigned tris);
void cp_pass_record_segment(struct cp_context *cp,
                            const struct cp_rasterize_args *aa,
                            const struct cp_rast_queues *queues,
                            unsigned rast_num_triangles,
                            unsigned num_triangles,
                            const struct cp_draw_batch *batch);

void cp_context_set_framebuffer(struct cp_context *cp,
                                const struct cp_fb_desc *fb,
                                unsigned samples);
void cp_render_scope_begin(struct cp_context *cp,
                           const struct cp_render_scope *scope);
void cp_render_scope_end(struct cp_context *cp);

struct glsl_type;
int cp_type_size_vec4(const struct glsl_type *type, bool bindless);


/* The vertex-format description both front ends fetch with; see the
 * definition for why a narrow attribute cannot be copied verbatim. */
enum cp_vf_conv cp_vertex_format(enum pipe_format format, uint32_t *nr_chan,
                                 uint32_t *chan_bytes, uint32_t *swizzle);
uint32_t cp_vertex_fill_w(enum pipe_format format, enum cp_vf_conv conv);

bool cp_clear_rect(struct cp_context *cp, void *data, uint64_t offset,
                   unsigned width, unsigned height, unsigned stride,
                   unsigned pixel_size, const uint32_t value[4], bool depth);

bool cp_context_init(struct cp_context *cp, struct cp_device *dev);
void cp_context_cleanup(struct cp_context *cp);

/*
 * A backstop, not a budget. The arena accumulates across the draws of a frame
 * and only resets after a few expansions, so a frame with a dozen draws at
 * 1280x720 legitimately reaches two or three gigabytes. What this catches is
 * the runaway: a stage allocating per pass rather than reusing asks for tens
 * of gigabytes, and since the arena is managed memory it is backed by system
 * RAM, so that does not fail — it invokes the OOM killer on the whole machine.
 */
#define CP_SCRATCH_MAX_BYTES ((size_t)8 << 30)

/* Sync and reclaim once a frame's draws have run up this much. */
#define CP_SCRATCH_RECLAIM_BYTES ((size_t)1 << 30)

/*
 * Stage timing, on CUDA events rather than on the host clock.
 *
 * This used to bracket each stage with clock_gettime. That measures how long
 * the host spent issuing the stage, which was already only loosely related to
 * how long the device spent running it and is now not related at all: every
 * launch goes on a stream and returns immediately. A stage whose kernel runs
 * for a millisecond and whose launch takes two microseconds was being
 * reported as two microseconds, and the one unlucky stage that happened to
 * follow a full queue absorbed everyone else's time.
 *
 * Events are recorded on the same stream as the work, so the interval between
 * two of them is device time between those two points. Reading them back
 * needs the stream to have reached the last one, which is a synchronisation —
 * hence only under CUDAPIPE_DEBUG_TIME, and hence a pool rather than one pair
 * per stage, because a blended draw runs the shading stages hundreds of times
 * and every interval has to be recorded before any of them can be read.
 */
enum cp_stage {
   CP_STAGE_ASSEMBLE,
   CP_STAGE_VERTEX,
   CP_STAGE_RASTERIZE,
   CP_STAGE_INTERPOLATE,
   CP_STAGE_FRAGMENT,
   CP_STAGE_WRITEBACK,
   CP_NUM_STAGES,
};

/* The scratch and upload arenas, and the per-stage timing they share. Used
 * by every stage of the pipeline, so they moved to the renderer first. */
void *cp_scratch_alloc(struct cp_context *cp, size_t size);
CUdeviceptr cp_scratch_alloc_device(struct cp_context *cp, size_t size);
CUdeviceptr cp_upload_begin(struct cp_context *cp, size_t size, void **host);
void cp_upload_end(struct cp_context *cp, CUdeviceptr dst, const void *host,
                   size_t size);
CUdeviceptr cp_upload(struct cp_context *cp, const void *data, size_t size);
void cp_scratch_begin(struct cp_context *cp);
void cp_scratch_reset(struct cp_context *cp);
void cp_scratch_destroy(struct cp_context *cp);
bool cp_timing_enabled(void);
void cp_stage_end(struct cp_context *cp, int stage);

enum cp_tile_census_cut_kind {
   CP_TILE_CUT_MAP, CP_TILE_CUT_COPY, CP_TILE_CUT_FLUSH, CP_TILE_CUT_COMPUTE
};
void cp_tile_census_cut(struct cp_context *cp, enum cp_tile_census_cut_kind k);

#endif /* CP_RENDERER_H */
