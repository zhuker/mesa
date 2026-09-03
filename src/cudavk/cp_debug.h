/*
 * The driver's environment switches, in one place.
 *
 * There are 43 of them across five files, and before this there was no list —
 * `ABUFFER.md` documented six, `BATCHING.md` two, and everything else was
 * discoverable only by grep. That matters more here than in most drivers
 * because this driver's failure mode is silence, and these flags are its
 * debugging surface: CUDAVK_NO_ABUFFER and CUDAVK_NO_BATCH have each
 * bisected a rendering bug in one command by eliminating a whole subsystem as
 * a suspect.
 *
 * So the array in cp_debug.c is the single source of truth. It carries the
 * name, the parse, the default and a one-line meaning, `CUDAVK_HELP=1`
 * prints it, and the doc tables are generated from it. A hand-maintained table
 * drifts — ABUFFER.md's said CUDAVK_ABUFFER_TIMING defaulted on when the
 * code had defaulted it off for a month — and a generated one cannot.
 *
 * This is not a performance change. getenv is 31 ns at this environment size
 * and the per-draw sites cost 0.199 ms/frame, 0.24% of a replay. Reading once
 * is a side effect of doing this properly, not the reason to.
 *
 * Filled once, by cp_debug_init(), at the top of cudavk_create_screen() —
 * the earliest point that runs once per process and before any draw, and
 * before cp_kernels_init(), which reads four of these during NVRTC setup.
 * Read-only afterwards: a flag that can change mid-run is a flag whose
 * behaviour cannot be reasoned about, and nothing here needs it.
 */

#ifndef CP_DEBUG_H
#define CP_DEBUG_H

#include <stdbool.h>
#include <stdint.h>

#include <cuda.h>

/* The default programmatic-dependent-launch level. Here rather than in the
 * registry entry so that tests/cp_debug_doc.py resolves it into FLAGS.md
 * instead of printing a name nobody can look up. */
#define CP_PDL_LEVEL_DEFAULT 3

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Four of these are genuinely tri-state: unset, or set to a value. The
 * rasterizer thresholds turn into an -D on the NVRTC command line only when
 * they are set at all, so "0" and "not given" have to stay distinguishable —
 * CUDAVK_TILE_BOUND=0 selects the full-tile walk, and leaving it out selects
 * whatever the kernel source defaults to.
 */
struct cp_opt_int {
   bool set;
   int  value;
};

/*
 * How a thread that is waiting for the GPU behaves. This is the three-way CUDA
 * context scheduling choice, kept whole rather than reduced to a bool, because
 * the interesting comparison is three-armed and `auto` is not a fourth
 * behaviour but a heuristic that picks one of the others.
 *
 * `auto` is CUDA's default, what this driver has always used, and what it
 * keeps: with one context and more logical processors than contexts it
 * resolves to SPIN, which is what makes this driver's ~17 waits a frame cheap.
 *
 * Measured on both captures, arms alternated in one session, pixels identical:
 *
 *   arm       old frame   Crossroads   host CPU (old)
 *   auto      15.7446     5.8608       36.37 s, 103%   <- default
 *   blocking  16.5989     6.0260       17.66 s,  48%
 *   yield     15.8460     5.8588       36.32 s, 103%
 *
 * So on an idle machine `blocking` costs 5.43% of the frame and gives back 51%
 * of the host CPU. The obvious next thought -- that this is the trade to make
 * when a co-located consumer wants those cores -- was measured on this same
 * machine with 16 and then 32 concurrent CPU-side PyTorch trainers, and it is
 * **wrong in both directions**:
 *
 *   saturated, old capture:  auto 42.94 ms   blocking 84.95 ms  (+97.8%)
 *   saturated, Crossroads:   auto 24.26 ms   blocking 46.55 ms  (+91.9%)
 *
 * `blocking` roughly doubles the frame under contention, with disjoint ranges,
 * because each of the ~17 waits a frame now ends in a kernel wake-up behind a
 * full runqueue. And it saves no CPU there: the CPU *rate* falls but the replay
 * runs 62% longer, so total CPU-seconds are equal or worse. The "half a core
 * given back" is an idle-machine artefact. Nor does the competitor benefit --
 * best case +2.6%, negative on the other capture -- because one spinning wait
 * thread is one core of thirty-two.
 *
 * So: `blocking` is for an otherwise idle machine only. Under contention the
 * arm that actually returns CPU is `yield`, which is frame-neutral to faster
 * there (42.59 ms vs 42.94 old, 18.08 vs 24.26 Crossroads) with CPU per frame
 * down 17-20% -- the exact opposite of its idle behaviour, where it is a spin
 * with syscall overhead that saves nothing. Neither becomes the default: each
 * wins only in the condition the other loses.
 *
 * Full record and the caveats, including that "moderate" load already fills
 * every physical core on this 16-core SMT2 part: DEAD_ENDS.md entry 16.
 */
enum cp_ctx_sched {
   CP_CTX_SCHED_AUTO = 0,
   CP_CTX_SCHED_SPIN,
   CP_CTX_SCHED_YIELD,
   CP_CTX_SCHED_BLOCKING,
};

/*
 * Which ALU operations nir_lower_alu_to_scalar still has to break apart.
 * `all` is the default and is not a filter at all: it runs the pass with a
 * NULL filter, which is what this driver did before the switch existed and
 * what every sample has been rendered with. The named classes exist to
 * bisect which of them a vector-width backend bug lives in — see
 * cpvk_scalarize_filter in cpvk_pipeline.c. An unrecognised value falls back
 * to `none`, the disabled mode, as every enum flag here does.
 */
enum cp_scalarize {
   CP_SCALARIZE_NONE = 0,
   CP_SCALARIZE_BASIC,
   CP_SCALARIZE_SEL,
   CP_SCALARIZE_ALU,
   CP_SCALARIZE_MOVE,
   CP_SCALARIZE_INTR,
   CP_SCALARIZE_REST,
   CP_SCALARIZE_ALL,
};

/* Residency scheme for the small-allocation arena. Kept as an enum rather than
 * collapsed to a bool because having the three arms in one binary is the
 * reason that work could be measured at all — it removed build-to-build
 * variance from the comparison. */
enum cp_arena_mode {
   CP_ARENA_OFF = 0,
   CP_ARENA_PINNED,
   CP_ARENA_MANAGED,
   CP_ARENA_ADVISE,
   /* Diagnostic: open the blocks and advise them exactly as `advise` does, but
    * serve nothing out of them, so that "the blocks exist" and "the small
    * allocations moved into them" can be told apart by measurement. */
   CP_ARENA_BLOCKSONLY,
};

#define CP_ARENA_MIN_SHIFT   6
#define CP_ARENA_CLASSES     5
#define CP_ARENA_MAX_SIZE    ((uint64_t)1 << (CP_ARENA_MIN_SHIFT + CP_ARENA_CLASSES - 1))
#define CP_ARENA_DEFAULT_MAX ((uint64_t)128)
#define CP_ARENA_DEFAULT_WARMUP ((uint64_t)64)

#define CP_MAX_BATCH_DRAWS 128
#define CP_TUNE_VETO 1.0    /* retain a cap only when its trial is faster */

/*
 * Field names are the variable name minus the CUDAVK_ prefix, lowercased,
 * with no exceptions. The variable names themselves have no rule — one
 * subsystem spells itself ABUFFER, ABUF and NO_ABUF across three prefixes and
 * two polarities — but the mapping from variable to field does, so that
 * knowing one gives you the other.
 *
 * The no_* fields keep the polarity of the variable rather than the sense of
 * the feature: `no_abuffer` is true when CUDAVK_NO_ABUFFER is set. Reading
 * `if (!cp_debug->no_abuffer)` is worse English than `if (abuffer_enabled)`
 * and better traceability, which is the trade this file exists to make.
 */
struct cp_debug {
   /* Tracing. All presence-only: `=0` switches these ON, because that is what
    * they have always done and someone's script relies on it. */
   bool debug_draw;
   bool debug_tex;
   bool spec_stats;
   bool texture_cache_stats;
   bool plan_stats;
   bool debug_vfetch;
   bool debug_work;
   bool debug_discard;
   bool debug_fs;
   bool debug_launch;
   bool debug_time;
   bool debug_batch;
   bool debug_batchdiff;
   bool debug_passseq;
   bool frag_census;
   bool nvtx;
   bool debug_rows;
   bool debug_clip;
   bool debug_pass;
   bool debug_episode;
   bool debug_rt;
   bool debug_faces;

   /* Tracing, with a value. */
   unsigned debug_fs_vstep;   /* print every Nth pixel; clamped to >= 1 */
   int      debug_fs_row;     /* only this framebuffer row, or -1 for all */

   /* Subsystem switches. */
   bool no_abuffer;
   bool no_abuf_batch;
   bool no_pass_episode;
   bool upload_stats;
   bool no_meta_fold;
   bool no_fetch_fold;
   bool fused_vfetch;
   bool no_fused_vfetch;
   unsigned fs_grid_waves;
   unsigned vfetch_decline_nth;
   bool vfetch_skip_seed;
   bool no_upload_coalesce;
   bool no_counter_block;
   unsigned upload_flush_fail_at;
   bool no_opaque_episode;
   bool no_opaque_streams;
   bool no_episode_gate_once;
   /*
    * Programmatic dependent launch. `pdl` is a level, not a boolean: 0 off,
    * 1 the A-buffer scan chain, 2 the rasterizer stage links, 3 the fragment
    * writeback and the segment scatter. See CP_PDL_TIER_* in cp_kernels.h.
    * `no_pdl` is the house-convention revert and forces the level to 0 in
    * apply_couplings().
    */
   unsigned pdl;
   unsigned texture_cache_budget_mb;
   bool no_pdl;
   bool no_sampler_variant;
   /* Revert switch for the software textureGather. Set, every tg4 goes back to
    * the unsupported branch's (0, 0, 0, 1) -- which is a wrong image, and is
    * only there so that a shader can be built both ways in one session. */
   bool no_texture_gather;
   bool no_gpu_sem_wait;
   /* Derived in apply_couplings() from no_texture_cache: the hardware texture
    * path is the default, and the registry carries the revert. Every use site
    * still reads texture_cache, so nothing else had to be inverted. */
   bool texture_cache;
   bool no_texture_cache;
   unsigned texture_cache_fail_table_upload_at;
   unsigned texture_cache_purge_at_preflight;
   unsigned texture_cache_fail_authoritative_alloc_at_preflight;
   bool texture_cache_fail_after_bounded_group0;
   bool texture_cache_fail_after_main_group0;
   bool texture_cache_fail_after_fs_enqueue;
   unsigned texture_cache_fail_array_alloc_at;
   unsigned texture_cache_fail_object_create_at;
   unsigned texture_cache_fail_conversion_enqueue_at;
   bool texture_cache_fail_fs_arg_begin;
   unsigned texture_cache_fail_create_stage;
   bool no_abuf_short_sort;
   bool no_abuf_warp_bucket;
   unsigned abuf_short_sort_max;
   bool no_abuf_fuse_scan;
   bool no_abuf_fuse_quad;
   bool abuf_fuse_check;
   unsigned abuf_fuse_break;
   bool flush_drain;
   bool no_seg_merge;
   bool no_abuf_append;
   bool no_fused_abuf_interp;
   bool no_fused_interp;
   bool no_counter_pool;
   bool interp_inline;
   bool compact_pdl;
   bool no_inline_fs;
   bool inline_fs;
   bool force_fused_fs;
   bool no_fused_rast;
   bool no_prim_refs;
   bool no_setup_cache;
   bool force_pass_fallback;
   unsigned tile_census;
   unsigned tile_census_every;
   bool tiled_opaque;
   bool tiled_opaque_census;
   bool unsafe_no_overflow;
   bool unsafe_force_opaque;
   bool no_batch;
   bool no_batch_blend;
   bool no_merge_scissor;
   bool keep_voff;
   bool keep_instkey;
   bool keep_vbkey;
   bool keep_ibkey;
   unsigned wait_spin_us;
   unsigned wait_spin_before_us;
   bool keep_pushkey;
   bool no_bincache;
   bool no_regcap;
   bool regcap_static;

   /* A-buffer. */
   bool     abuffer_verify;
   bool     abuffer_composite;
   unsigned abuffer_verify_draws;
   bool     abuffer_timing;
   bool     abuffer_debug;
   unsigned abuffer_layers;      /* 0 = unset */
   unsigned abuf_min_tris;       /* 0 = no floor, and the default */
   struct cp_opt_int abuf_compile;

   /* Draw batching. */
   unsigned batch_max;

   /* CUDA context. */
   int      ctx_sched;           /* enum cp_ctx_sched */
   bool     ctx_check;
   bool     no_ctx_scope_trim;

   /* Small-allocation arena. */
   int      small_alloc;         /* enum cp_arena_mode */
   uint64_t small_alloc_max;
   uint64_t small_alloc_warmup;
   bool     small_alloc_stats;

   /* Shader compilation. */
   unsigned max_registers;
   unsigned launch_bounds;
   double   tune_veto;
   bool     shader_stats;
   bool     dump_nir;
   bool     dump_ir;
   bool     dump_ptx;
   int      scalarize;           /* enum cp_scalarize */
   bool     no_hoist_inputs;
   bool     no_reg_ssa;
   bool     keep_small_dynamic_regcap;

   /* Rasterizer tuning. Tri-state: these become NVRTC -D options only when
    * set. Unrelated to the arena despite CUDAVK_SMALL_* naming both. */
   struct cp_opt_int small_threshold;
   struct cp_opt_int medium_threshold;
   struct cp_opt_int point_threshold;
   struct cp_opt_int tile_bound;
   bool no_host_pin_readback;
   bool no_append_prefilter;
   bool no_layered_copy3d;
};

/*
 * Always non-NULL, so a read that somehow happens before init sees a zeroed
 * struct rather than a crash. Everything real runs after screen creation.
 */
extern const struct cp_debug *cp_debug;

/* Idempotent. Call before anything reads cp_debug. */
void cp_debug_init(void);

/*
 * Context discipline, checkable rather than assumed.
 *
 * Every CUDA call in this driver acts on the *current* context: of 606 driver
 * API entry points in CUDA 12.8 only 36 take a CUcontext, and none of the ones
 * used here do. So an entry point must make this device's context current, and
 * -- because that is thread state shared with the caller -- must put the
 * caller's back. CPVK_CTX_SCOPE in cpvk_private.h does both.
 *
 * That works only if every path into CUDA passes through a scoped entry point,
 * which is a whole-call-graph property no compiler checks. CUDAVK_CTX_CHECK
 * turns it into a test: the chokepoints call this, and anything reached
 * without a scope names itself instead of silently allocating in, or
 * launching into, whatever context the application left current.
 *
 * Enable it for a full test run after touching entry points, not in
 * production: it is a cuCtxGetCurrent per launch.
 */
void cp_ctx_check_failed(const char *where, const void *got, const void *want);

static inline void
cp_ctx_check(const char *where, void *want)
{
   if (!cp_debug->ctx_check)
      return;
   CUcontext got = NULL;
   cuCtxGetCurrent(&got);
   if (got != (CUcontext)want)
      cp_ctx_check_failed(where, got, want);
}

#ifdef __cplusplus
}
#endif

#endif /* CP_DEBUG_H */
