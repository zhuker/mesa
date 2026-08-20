#ifndef CP_CONTEXT_H
#define CP_CONTEXT_H

#include "cp_debug.h"
#include "cp_draw_types.h"   /* CP_MAX_BATCH_DRAWS and the rest of the tunables */
#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include <cuda.h>
#include <stdbool.h>
#include <stdio.h>
#include "kernels/cp_rast_types.h"

struct cp_shader_binary;

#define CP_MAX_SHADER_BUFFERS 16
#define CP_MAX_CONST_BUFFERS  16
#define CP_MAX_SAMPLERS       256

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

struct cp_context {

   struct cp_screen *screen;

   /*
    * Draws held back for merging. `pending` means one or more draws have been
    * accepted and nothing has run yet, so every path that observes rendering —
    * a flush, a readback, a clear, a blit, a dispatch — has to call
    * cp_batch_flush() before it looks.
    */
   struct {
      bool pending;
      unsigned ndraws;
      /* Triangles over the whole batch, which is what the clipper's output
       * buffer is sized from and so what CP_MAX_BATCH_TRIS caps. */
      unsigned tris;
      struct cp_batch_key key;
      struct cp_draw_call info;
      /* One index range per merged draw; cp_draw_execute() turns these into
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
   } batch;

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
      /* Shading state. */
      struct cp_shader_binary *vs, *fs;
      struct cp_draw_call info;
      unsigned ndraws;
      unsigned drawid_offset;
      CUdeviceptr slices_dev;
      /* The batch snapshot, both for the per-segment shade (fs rows) and for
       * re-executing the segment classically when the episode falls back. */
      struct cp_draw_range draws[CP_MAX_BATCH_DRAWS];
      uint32_t instance_counts[CP_MAX_BATCH_DRAWS];
      uint32_t draw_ids[CP_MAX_BATCH_DRAWS];
      struct cp_rect scissors[CP_MAX_BATCH_DRAWS];
      uint64_t vs_ubos[CP_MAX_BATCH_DRAWS * CP_ARG_UBO_STRIDE];
      uint64_t fs_ubos[CP_MAX_BATCH_DRAWS * CP_ARG_UBO_STRIDE];
      uint64_t vb_bases[CP_MAX_BATCH_DRAWS * CP_VB_TABLE_STRIDE];
      /* Live state the fallback restores before re-executing. */
      struct pipe_vertex_element vertex_elements[16];
      unsigned num_vertex_elements, vertex_stride;
      unsigned num_vs_ubos, num_fs_ubos;
   } *pass_segs;                    /* [CP_PASS_MAX_SEGS], at context create */

   struct {
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
    * CP_ARG_SLOT_UBO_TABLE. Set once at the top of cp_draw_execute() so that
    * it cannot carry from one draw to the next, and read by
    * cp_fs_launch_shader() — which both shading paths go through, and which is
    * three call frames below where the batch is known.
    */
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

   struct pipe_framebuffer_state framebuffer;   /* adapter: the key, the
                                                 * clears and the blits */
   /* What the draw path reads, resolved when the framebuffer was bound. */
   struct cp_fb_desc fb;
   struct cp_rect scissor;
   /*
    * What the pipeline reads, in the driver's own types: the viewport's scale
    * and translate, three fields of the rasterizer and three of the depth
    * state. Field names match Gallium's, so nothing that reads them changed.
    */
   struct cp_viewport_state viewport;
   struct cp_raster_state rasterizer;
   struct cp_depth_state depth_stencil;
   /*
    * And what the Gallium adapter compares to decide whether a held-back
    * batch must be flushed, and what the batch key carries. Adapter-only, and
    * gone with Gallium. The comparison deliberately did not move with the
    * fields: narrowing it to what the pipeline reads would make batches
    * larger than they are today, and batch size is what makes the clipper's
    * unstable primitive order visible (gaps 15 and 16).
    */
   struct pipe_viewport_state viewport_cso;
   struct pipe_rasterizer_state rasterizer_cso;
   struct pipe_depth_stencil_alpha_state depth_stencil_cso;

   /* Visibility buffer, rebuilt per draw: it resolves which triangle of the
    * current draw wins each pixel. */
   CUdeviceptr visbuf;
   CUdeviceptr reject;      /* per-pixel discarded triangles, CP_DISCARD_LAYERS deep */
   CUdeviceptr resolved;    /* per-pixel byte: a fragment has been written */
   unsigned fb_samples;     /* samples per pixel of the bound framebuffer */
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
   struct cp_shader_binary *vs_shader;
   struct cp_shader_binary *fs_shader;

   /* Blend state */
   struct pipe_blend_state blend_state;   /* adapter-only: the flush
                                           * comparison and the batch key */
   /* What both kernels that evaluate a blend actually read, resolved once
    * when the state was bound rather than rebuilt per draw. */
   struct cp_blend_desc blend_desc;
   bool blend_enabled;

   /* Texture state for FS */
   CUtexObject tex_objects[32];
   unsigned num_tex_objects;
   struct {
      void *data;
      unsigned width, height;
      unsigned row_stride;
      unsigned pixel_size;
      enum pipe_format format;
   } tex_resources[32];

   /* Vertex buffers and elements */
   struct pipe_vertex_buffer vertex_buffers[16];
   /* Resolved when the buffers were bound: base address plus offset, which
    * is all the draw path ever wanted from them. */
   uint64_t vb_base[16];
   /* And the elements, with the format already turned into the fetch
    * kernel's conversion. The Gallium array beside this stays for the
    * batch key and the pass-segment snapshot. */
   struct cp_vertex_elem velem[16];
   unsigned num_vertex_buffers;

   struct pipe_vertex_element vertex_elements[16];
   unsigned num_vertex_elements;
   unsigned vertex_stride;

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

   struct {
      void *buffer;
      unsigned buffer_size;
      CUdeviceptr managed_copy;
      unsigned managed_size;
      /*
       * The binding came from a user pointer and was copied into
       * `managed_copy`, which is reused for every update. Two draws then see
       * the same device address with different contents, so a batch that
       * deferred either of them would shade both with whichever value landed
       * last. cp_batch_eligible() refuses such a draw outright.
       */
      bool user_copy;
   } fs_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_fs_ubos;

   struct {
      void *buffer;
      unsigned buffer_size;
      CUdeviceptr managed_copy;
      unsigned managed_size;
      bool user_copy;
   } vs_ubos[CP_MAX_CONST_BUFFERS];
   unsigned num_vs_ubos;

   /* Device-visible table of deduplicated sampler states. Descriptors refer to
    * entries by index; see cp_register_sampler(). */
   CUdeviceptr sampler_table;
   struct cp_sampler_info sampler_table_host[CP_MAX_SAMPLERS];
   unsigned num_samplers;

   /* GPU-resident pipeline state — managed memory, written by CPU on state
    * changes, read by GPU kernels during draws. */
   struct cp_gpu_state *gpu_state;

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
};
struct cp_gallium {
   struct pipe_context base;   /* first: entry points cast pipe_context to this */
   struct cp_context cp;
};

/*
 * The one place a pipe_context becomes the renderer.
 *
 * Written as a function because the hand-written cast is silently wrong the
 * moment the layout changes, and a wrong one compiles: making pipe_context a
 * member rather than the head of cp_context left ten casts in cp_resource.c
 * pointing at the wrong offset, and every sample still exited 0 while the
 * capture rendered 28% wrong.
 */
static inline struct cp_context *
cp_ctx(struct pipe_context *ctx)
{
   return &((struct cp_gallium *)ctx)->cp;
}


struct pipe_context *
cudapipe_create_context(struct pipe_screen *screen, void *priv, unsigned flags);

uint32_t cp_depth_to_sortable(float depth);
void cp_clear_depthbuf(struct cp_context *cp, float depth);

/*
 * Submit whatever draws are being held back for merging, if any.
 *
 * Anything that observes the framebuffer — a flush, a map, a clear, a blit, a
 * copy, a dispatch — has to call this first, or it looks at a frame with draws
 * missing from it. Cheap and idempotent when nothing is pending.
 */
void cp_batch_flush(struct cp_context *cp);

/*
 * The same, naming what ended the batch. Every state change that a batch
 * cannot survive goes through this, and CUDAPIPE_DEBUG_BATCH prints the
 * reason — which is the only practical way to find out why a sample that
 * looks batchable is producing batches of one.
 */
void cp_batch_flush_why(struct cp_context *cp, const char *why);
/* The deferrable variant: the pending batch may become a pass-episode
 * segment, and a pending episode stays open. Only for state changes an
 * episode carries per segment; see the definition. */
void cp_batch_flush_defer_why(struct cp_context *cp, const char *why);
void cp_pass_finish(struct cp_context *cp);
enum cp_tile_census_cut_kind {
   CP_TILE_CUT_MAP, CP_TILE_CUT_COPY, CP_TILE_CUT_FLUSH, CP_TILE_CUT_COMPUTE
};
void cp_tile_census_end_pass(struct cp_context *cp);
void cp_tile_census_cut(struct cp_context *cp, enum cp_tile_census_cut_kind k);

/*
 * How the sampler and the fragment writeback decode and encode a format, or
 * CP_TEXEL_UNSUPPORTED / a negative result when they can't handle it at all.
 *
 * The screen reports format support from these, so that what the driver claims
 * to support and what its kernels can actually decode cannot drift apart.
 */
uint32_t cp_texel_encoding_from_format(enum pipe_format format);
int cp_color_encoding_from_format(enum pipe_format format);

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

#endif
