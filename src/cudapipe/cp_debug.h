/*
 * The driver's environment switches, in one place.
 *
 * There are 43 of them across five files, and before this there was no list —
 * `ABUFFER.md` documented six, `BATCHING.md` two, and everything else was
 * discoverable only by grep. That matters more here than in most drivers
 * because this driver's failure mode is silence, and these flags are its
 * debugging surface: CUDAPIPE_NO_ABUFFER and CUDAPIPE_NO_BATCH have each
 * bisected a rendering bug in one command by eliminating a whole subsystem as
 * a suspect.
 *
 * So the array in cp_debug.c is the single source of truth. It carries the
 * name, the parse, the default and a one-line meaning, `CUDAPIPE_HELP=1`
 * prints it, and the doc tables are generated from it. A hand-maintained table
 * drifts — ABUFFER.md's said CUDAPIPE_ABUFFER_TIMING defaulted on when the
 * code had defaulted it off for a month — and a generated one cannot.
 *
 * This is not a performance change. getenv is 31 ns at this environment size
 * and the per-draw sites cost 0.199 ms/frame, 0.24% of a replay. Reading once
 * is a side effect of doing this properly, not the reason to.
 *
 * Filled once, by cp_debug_init(), at the top of cudapipe_create_screen() —
 * the earliest point that runs once per process and before any draw, and
 * before cp_kernels_init(), which reads four of these during NVRTC setup.
 * Read-only afterwards: a flag that can change mid-run is a flag whose
 * behaviour cannot be reasoned about, and nothing here needs it.
 */

#ifndef CP_DEBUG_H
#define CP_DEBUG_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Four of these are genuinely tri-state: unset, or set to a value. The
 * rasterizer thresholds turn into an -D on the NVRTC command line only when
 * they are set at all, so "0" and "not given" have to stay distinguishable —
 * CUDAPIPE_TILE_BOUND=0 selects the full-tile walk, and leaving it out selects
 * whatever the kernel source defaults to.
 */
struct cp_opt_int {
   bool set;
   int  value;
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
 * Field names are the variable name minus the CUDAPIPE_ prefix, lowercased,
 * with no exceptions. The variable names themselves have no rule — one
 * subsystem spells itself ABUFFER, ABUF and NO_ABUF across three prefixes and
 * two polarities — but the mapping from variable to field does, so that
 * knowing one gives you the other.
 *
 * The no_* fields keep the polarity of the variable rather than the sense of
 * the feature: `no_abuffer` is true when CUDAPIPE_NO_ABUFFER is set. Reading
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

   /* Tracing, with a value. */
   unsigned debug_fs_vstep;   /* print every Nth pixel; clamped to >= 1 */
   int      debug_fs_row;     /* only this framebuffer row, or -1 for all */

   /* Subsystem switches. */
   bool no_abuffer;
   bool no_abuf_batch;
   bool no_pass_episode;
   bool upload_stats;
   bool meta_fold;
   bool no_opaque_episode;
   bool no_sampler_variant;
   bool texture_cache;
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
   bool flush_drain;
   bool no_seg_merge;
   bool no_abuf_append;
   bool no_fused_abuf_interp;
   bool no_fused_interp;
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
   bool no_batch;
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
   struct cp_opt_int abuf_compile;

   /* Draw batching. */
   unsigned batch_max;

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

   /* Rasterizer tuning. Tri-state: these become NVRTC -D options only when
    * set. Unrelated to the arena despite CUDAPIPE_SMALL_* naming both. */
   struct cp_opt_int small_threshold;
   struct cp_opt_int medium_threshold;
   struct cp_opt_int point_threshold;
   struct cp_opt_int tile_bound;
};

/*
 * Always non-NULL, so a read that somehow happens before init sees a zeroed
 * struct rather than a crash. Everything real runs after screen creation.
 */
extern const struct cp_debug *cp_debug;

/* Idempotent. Call before anything reads cp_debug. */
void cp_debug_init(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_DEBUG_H */
