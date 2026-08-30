/*
 * The registry. See cp_debug.h for why this exists.
 *
 * Every entry states how the variable is parsed today, not how it would be
 * parsed if the naming and the parsing had been designed together. Both
 * boolean kinds are real and both have to survive: presence-only flags treat
 * `=0` as ON, and changing that would silently switch tracing off for anyone
 * who wrote CUDAVK_DEBUG_DRAW=0 in a script and has been getting tracing
 * ever since.
 */

#include "cp_debug.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/macros.h"

/*
 * Loud on purpose, and once per site: a context escape is a correctness bug
 * whose symptom appears in someone else's library, so it must not scroll past.
 */
void
cp_ctx_check_failed(const char *where, const void *got, const void *want)
{
   fprintf(stderr,
           "cudavk: CONTEXT ESCAPE in %s: current is %p, expected %p. "
           "That path reaches CUDA without an entry-point CPVK_CTX_SCOPE.\n",
           where, got, want);
}

enum cp_flag_type {
   CP_FLAG_BOOL_PRESENCE,  /* set if the variable exists at all, whatever its value */
   CP_FLAG_BOOL_VALUE,     /* atoi(v) != 0 */
   CP_FLAG_OPT_BOOL,       /* unset, or atoi(v) != 0 */
   CP_FLAG_OPT_INT,        /* unset, or atoi(v) */
   CP_FLAG_UINT,
   CP_FLAG_U64,
   CP_FLAG_INT,
   CP_FLAG_FLOAT,
   CP_FLAG_ENUM,
};

struct cp_flag_value {
   const char *name;
   int64_t     value;
};

struct cp_flag_def {
   const char       *name;
   enum cp_flag_type type;
   size_t            offset;
   const char       *help;

   int64_t dflt;         /* integer and enum default; unused for FLOAT */
   double  fdflt;

   /* Clamp applied after parsing, when has_range. Preserved per flag: only
    * three of these clamp today and each clamps for its own reason. */
   bool    has_range;
   int64_t lo, hi;

   /* Empty string counts as unset. Some flags check `v && *v` and some check
    * only `v`; the difference is visible to anyone who exports the variable
    * empty, so it is recorded rather than unified. */
   bool    empty_is_unset;

   const struct cp_flag_value *values;   /* CP_FLAG_ENUM */
};

static const struct cp_flag_value ctx_scheds[] = {
   { "auto",     CP_CTX_SCHED_AUTO },
   { "spin",     CP_CTX_SCHED_SPIN },
   { "yield",    CP_CTX_SCHED_YIELD },
   { "blocking", CP_CTX_SCHED_BLOCKING },
   { NULL, 0 },
};

static const struct cp_flag_value scalarize_classes[] = {
   { "all",   CP_SCALARIZE_ALL },
   { "none",  CP_SCALARIZE_NONE },
   { "basic", CP_SCALARIZE_BASIC },
   { "sel",   CP_SCALARIZE_SEL },
   { "alu",   CP_SCALARIZE_ALU },
   { "move",  CP_SCALARIZE_MOVE },
   { "intr",  CP_SCALARIZE_INTR },
   { "rest",  CP_SCALARIZE_REST },
   { NULL, 0 },
};

static const struct cp_flag_value arena_modes[] = {
   { "advise",     CP_ARENA_ADVISE },
   { "pinned",     CP_ARENA_PINNED },
   { "managed",    CP_ARENA_MANAGED },
   { "blocksonly", CP_ARENA_BLOCKSONLY },
   { "off",        CP_ARENA_OFF },
   { NULL, 0 },
};

#define F(field) offsetof(struct cp_debug, field)

/*
 * Grouped as the subsystems are, and in the order CUDAVK_HELP prints them.
 * The help text is one line and says what the flag is *for* — "trace every
 * draw" rather than "sets cp->debug_draw" — because the reader is someone who
 * has a bug and no idea which of these to reach for.
 */
static const struct cp_flag_def flags[] = {
   /* ---- tracing ---- */
   { "CUDAVK_DEBUG_DRAW", CP_FLAG_BOOL_PRESENCE, F(debug_draw),
     "trace every draw: geometry, attachments, and why a draw was skipped" },
   { "CUDAVK_DEBUG_TEX", CP_FLAG_BOOL_PRESENCE, F(debug_tex),
     "trace sampler and texture-handle setup" },
   { "CUDAVK_PLAN_STATS", CP_FLAG_BOOL_VALUE, F(plan_stats),
     "report at teardown what deciding one draw at a time costs: batch keys "
     "built, pairwise merge tests, flushes, pass episodes closed and every "
     "reactive reallocation, with per-scope averages" },
   { "CUDAVK_SPEC_STATS", CP_FLAG_BOOL_VALUE, F(spec_stats),
     "report at teardown how many fragment launches used a sampler-specialised "
     "kernel, and which shaders sample textures the specialiser could not "
     "match; a silent drop to zero is a performance regression with no error" },
   { "CUDAVK_TEXTURE_CACHE_STATS", CP_FLAG_BOOL_VALUE, F(texture_cache_stats),
     "report hardware-texture fragment execution hits and software fallbacks" },
   { "CUDAVK_DEBUG_VFETCH", CP_FLAG_BOOL_PRESENCE, F(debug_vfetch),
     "dump what the GPU vertex fetch gathered; syncs, so debug-only" },
   { "CUDAVK_DEBUG_WORK", CP_FLAG_BOOL_PRESENCE, F(debug_work),
     "report how much of the shading launch did work; syncs" },
   { "CUDAVK_DEBUG_DISCARD", CP_FLAG_BOOL_PRESENCE, F(debug_discard),
     "report covered and discarded fragment counts per pass; syncs" },
   { "CUDAVK_DEBUG_FS", CP_FLAG_BOOL_PRESENCE, F(debug_fs),
     "dump fragment-shader inputs and outputs per pixel; syncs" },
   { "CUDAVK_DEBUG_FS_VSTEP", CP_FLAG_UINT, F(debug_fs_vstep),
     "with DEBUG_FS, print every Nth pixel", .dflt = 1, .has_range = true, .lo = 1, .hi = UINT32_MAX },
   { "CUDAVK_DEBUG_FS_ROW", CP_FLAG_INT, F(debug_fs_row),
     "with DEBUG_FS, restrict the dump to one framebuffer row", .dflt = -1 },
   { "CUDAVK_DEBUG_LAUNCH", CP_FLAG_BOOL_PRESENCE, F(debug_launch),
     "trace compute dispatch: bound UBO and SSBO pointers" },
   { "CUDAVK_DEBUG_TIME", CP_FLAG_BOOL_PRESENCE, F(debug_time),
     "time the pipeline stages" },
   { "CUDAVK_DEBUG_BATCH", CP_FLAG_BOOL_PRESENCE, F(debug_batch),
     "report why each draw batch ended" },
   { "CUDAVK_DEBUG_BATCHDIFF", CP_FLAG_BOOL_PRESENCE, F(debug_batchdiff),
     "report which state field broke a batch, field by field" },
   { "CUDAVK_DEBUG_PASSSEQ", CP_FLAG_BOOL_VALUE, F(debug_passseq),
     "log one line per framebuffer bind and per draw: shaders, blendedness, "
     "eligibility — the raw material for pass-structure statistics" },
   { "CUDAVK_FRAG_CENSUS", CP_FLAG_BOOL_PRESENCE, F(frag_census),
     "count fragments per draw; also compiles the instrumented kernels in" },
   { "CUDAVK_NVTX", CP_FLAG_BOOL_PRESENCE, F(nvtx),
     "push an NVTX range around each draw and stage, for nsys" },
   { "CUDAVK_DEBUG_ROWS", CP_FLAG_BOOL_PRESENCE, F(debug_rows),
     "trace the per-draw tables a batch builds: fragment UBO slots and the "
     "vertex slice table the fragment stage searches" },
   { "CUDAVK_DEBUG_CLIP", CP_FLAG_BOOL_PRESENCE, F(debug_clip),
     "report what each batch hands the clip stage: triangles, draws, whether "
     "the stable slot mode is on" },
   { "CUDAVK_DEBUG_PASS", CP_FLAG_BOOL_PRESENCE, F(debug_pass),
     "report, for the first three batches only, every condition that decides "
     "whether a blended batch may join the open pass episode" },
   { "CUDAVK_DEBUG_EPISODE", CP_FLAG_BOOL_PRESENCE, F(debug_episode),
     "trace episode boundaries: what appended to an episode and what cut it" },
   { "CUDAVK_DEBUG_RT", CP_FLAG_BOOL_PRESENCE, F(debug_rt),
     "trace descriptor writes, the attachments a render pass resolved to, and "
     "the first texels either side of an image copy; the copies sync" },
   { "CUDAVK_DEBUG_FACES", CP_FLAG_BOOL_PRESENCE, F(debug_faces),
     "when a cube descriptor is written, read back the first texel of each of "
     "its six faces from where the sampler will look; syncs" },

   /* ---- subsystem switches ---- */
   { "CUDAVK_NO_ABUFFER", CP_FLAG_BOOL_PRESENCE, F(no_abuffer),
     "disable the A-buffer; blended draws go back to the direct path" },
   { "CUDAVK_NO_ABUF_BATCH", CP_FLAG_BOOL_PRESENCE, F(no_abuf_batch),
     "disable batching of A-buffer draws" },
   { "CUDAVK_NO_PASS_EPISODE", CP_FLAG_BOOL_VALUE, F(no_pass_episode),
     "disable pass episodes: consecutive blended batches stop sharing one "
     "A-buffer build and drain" },
   { "CUDAVK_UPLOAD_STATS", CP_FLAG_BOOL_VALUE, F(upload_stats),
     "count small host-to-device copies, clears, context syncs and upload "
     "ring wraps per call site, and report them at teardown" },
   { "CUDAVK_NO_META_FOLD", CP_FLAG_BOOL_VALUE, F(no_meta_fold),
     "upload the vertex stage's vcount and stride as their own block again "
     "instead of carrying them in the argument block's scalar area" },
   { "CUDAVK_NO_COUNTER_BLOCK", CP_FLAG_BOOL_VALUE, F(no_counter_block),
     "clear the A-buffer's scalar counters one at a time again instead of "
     "clearing the whole counter block once per draw or episode" },
   { "CUDAVK_NO_UPLOAD_COALESCE", CP_FLAG_BOOL_VALUE, F(no_upload_coalesce),
     "send one host-to-device copy per upload block again instead of one per "
     "launch boundary" },
   { "CUDAVK_UPLOAD_FLUSH_FAIL_AT", CP_FLAG_UINT, F(upload_flush_fail_at),
     "fault injection: fail the Nth coalesced upload flush" },
   { "CUDAVK_NO_FETCH_FOLD", CP_FLAG_BOOL_VALUE, F(no_fetch_fold),
     "clear the clip and raster queue counters with their own device clears "
     "again instead of seeding them inside cp_vertex_fetch" },
   { "CUDAVK_FUSED_VFETCH", CP_FLAG_BOOL_VALUE, F(fused_vfetch),
     "gather vertex attributes inside the vertex shader, from bitcode inlined "
     "into it, instead of launching cp_vertex_fetch before it", .dflt = 1 },
   { "CUDAVK_NO_FUSED_VFETCH", CP_FLAG_BOOL_VALUE, F(no_fused_vfetch),
     "do not even build the fused vertex execution: the resource-isolated "
     "control, and the revert once the fusion is the default" },
   { "CUDAVK_FS_GRID_WAVES", CP_FLAG_UINT, F(fs_grid_waves),
     "size the fragment grid to fill the machine this many times over and let "
     "it grid-stride, instead of covering the framebuffer's worst case; 0 "
     "keeps the 4096-block cap, which is the only bound either way" },
   { "CUDAVK_VFETCH_DECLINE_NTH", CP_FLAG_UINT, F(vfetch_decline_nth),
     "fault injection: refuse the fused execution for the Nth vertex shader "
     "compiled, or for every one at 4294967295, after building it either way" },
   { "CUDAVK_VFETCH_SKIP_SEED", CP_FLAG_BOOL_VALUE, F(vfetch_skip_seed),
     "fault injection: a fused draw does not seed the clip and raster "
     "counters, and the host still skips their clears -- the negative control "
     "for the seeding moving into the fused vertex shader" },
   { "CUDAVK_NO_OPAQUE_EPISODE", CP_FLAG_BOOL_VALUE,
     F(no_opaque_episode),
     "disable consecutive opaque-run visibility deferral" },
   { "CUDAVK_NO_OPAQUE_STREAMS", CP_FLAG_BOOL_VALUE, F(no_opaque_streams),
     "issue an opaque episode's segments back to back on the main stream "
     "instead of fanning them over the pass side streams" },
   { "CUDAVK_NO_PDL", CP_FLAG_BOOL_VALUE, F(no_pdl),
     "issue every kernel with an ordinary launch again, so that no dependent "
     "kernel starts before its predecessor in the same stream has drained; "
     "the revert for the programmatic dependent launch default" },
   { "CUDAVK_PDL", CP_FLAG_UINT, F(pdl),
     "how much of the driver runs its dependent kernels with the programmatic "
     "stream serialization attribute, so that a kernel may start before its "
     "predecessor on the same stream has drained: 0 none, 1 the A-buffer scan "
     "chain, 2 also the rasterizer stage links, 3 also the fragment writeback "
     "and the segment scatter, 4 also offers stage 1, which is a diagnostic "
     "rather than a default: most of those links are expected to refuse "
     "themselves and the counters are the measurement. The default is 3; "
     "lower it to "
     "bisect a regression, raise it to 4 to measure stage 1, and see "
     "CUDAVK_NO_PDL for the plain revert. Needs "
     "compute capability 9.0 and CUDA 11.8, and any link whose predecessor "
     "turns out not to be the named kernel falls back to an ordinary launch",
     .dflt = CP_PDL_LEVEL_DEFAULT },
   { "CUDAVK_NO_SAMPLER_VARIANT", CP_FLAG_BOOL_VALUE,
     F(no_sampler_variant),
     "disable literal-state fragment sampler variants" },
   { "CUDAVK_NO_TEXTURE_GATHER", CP_FLAG_BOOL_VALUE, F(no_texture_gather),
     "refuse every textureGather() again, so that it returns the constant "
     "0 0 0 1 the driver returned before the sampler could serve one" },
   { "CUDAVK_NO_GPU_SEM_WAIT", CP_FLAG_BOOL_VALUE, F(no_gpu_sem_wait),
     "block the submitting thread on a queue-submit wait semaphore instead of "
     "making the renderer stream wait on its completion event" },
   { "CUDAVK_NO_TEXTURE_CACHE", CP_FLAG_BOOL_VALUE, F(no_texture_cache),
     "disable direct CUDA hardware-texture fragment execution; fall back to "
     "software sampling" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_TABLE_UPLOAD_AT", CP_FLAG_UINT,
     F(texture_cache_fail_table_upload_at),
     "fault injection: fail the Nth hardware texture table upload enqueue" },
   { "CUDAVK_TEXTURE_CACHE_PURGE_AT_PREFLIGHT", CP_FLAG_UINT,
     F(texture_cache_purge_at_preflight),
     "fault injection: purge derived textures before the Nth HW preflight" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_AUTHORITATIVE_ALLOC_AT_PREFLIGHT",
     CP_FLAG_UINT, F(texture_cache_fail_authoritative_alloc_at_preflight),
     "fault injection: OOM one renderer allocation at the Nth HW preflight" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_AFTER_BOUNDED_GROUP0", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_after_bounded_group0),
     "fault injection: latch fatal after bounded episode FS group zero" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_AFTER_MAIN_GROUP0", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_after_main_group0),
     "fault injection: latch fatal after main episode FS group zero" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_AFTER_FS_ENQUEUE", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_after_fs_enqueue),
     "fault injection: sync then latch fatal after one hardware FS enqueue" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_ARRAY_ALLOC_AT", CP_FLAG_UINT,
     F(texture_cache_fail_array_alloc_at),
     "fault injection: OOM the Nth derived CUDA array allocation" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_OBJECT_CREATE_AT", CP_FLAG_UINT,
     F(texture_cache_fail_object_create_at),
     "fault injection: OOM the Nth CUDA texture-object creation" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_CONVERSION_ENQUEUE_AT", CP_FLAG_UINT,
     F(texture_cache_fail_conversion_enqueue_at),
     "fault injection: fail the Nth converted-cache kernel enqueue fatally" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_FS_ARG_BEGIN", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_fs_arg_begin),
     "fault injection: fail first hardware FS-argument reservation fatally" },
   { "CUDAVK_TEXTURE_CACHE_FAIL_CREATE_STAGE", CP_FLAG_UINT,
     F(texture_cache_fail_create_stage),
     "fault injection: fail cache create stage 1=event or 2=surface" },
   { "CUDAVK_NO_ABUF_SHORT_SORT", CP_FLAG_BOOL_VALUE,
     F(no_abuf_short_sort),
     "disable the single-thread short A-buffer run sorter" },
   { "CUDAVK_NO_ABUF_WARP_BUCKET", CP_FLAG_BOOL_VALUE,
     F(no_abuf_warp_bucket),
     "disable warp-aggregated A-buffer segment bucketing atomics" },
   { "CUDAVK_ABUF_SHORT_SORT_MAX", CP_FLAG_UINT,
     F(abuf_short_sort_max),
     "largest A-buffer run sorted by one thread", .dflt = 16,
     .has_range = true, .lo = 2, .hi = 64 },
   { "CUDAVK_NO_ABUF_FUSE_SCAN", CP_FLAG_BOOL_VALUE, F(no_abuf_fuse_scan),
     "restore the three- or five-launch A-buffer prefix sum instead of the "
     "fused reduce plus finish, and its separate fill-cursor clear" },
   { "CUDAVK_NO_ABUF_FUSE_QUAD", CP_FLAG_BOOL_VALUE, F(no_abuf_fuse_quad),
     "restore the A-buffer quad build's separate block worklist, counting "
     "pass, three-launch scan and blk_counts clear" },
   { "CUDAVK_ABUF_FUSE_CHECK", CP_FLAG_BOOL_VALUE, F(abuf_fuse_check),
     "run the classic A-buffer scan and quad count beside the fused ones and "
     "compare every element on the device; gate only, never timed" },
   { "CUDAVK_ABUF_FUSE_BREAK", CP_FLAG_UINT, F(abuf_fuse_break),
     "negative control for the fusion check: 1 drops the block base in the "
     "fused scan, 2 stops the fused quad count writing an uncovered zero",
     .has_range = true, .lo = 0, .hi = 2 },
   { "CUDAVK_FLUSH_DRAIN", CP_FLAG_BOOL_VALUE, F(flush_drain),
     "restore the draining flush: cp_flush waits for the whole device and "
     "rewinds the arenas in place instead of ping-ponging generations" },
   { "CUDAVK_NO_SEG_MERGE", CP_FLAG_BOOL_VALUE, F(no_seg_merge),
     "disable merged shading groups: every episode segment shades in its "
     "own launch group, as before" },
   { "CUDAVK_NO_ABUF_APPEND", CP_FLAG_BOOL_VALUE, F(no_abuf_append),
     "disable the single-pass A-buffer build: the count pass stops appending "
     "(pixel, prim) records and the fill rasterizes a second time" },
   { "CUDAVK_NO_FUSED_ABUF_INTERP", CP_FLAG_BOOL_VALUE,
     F(no_fused_abuf_interp),
     "launch A-buffer interpolation separately instead of calling it from "
     "the generated fragment shader; performance experiment only" },
   { "CUDAVK_NO_FUSED_INTERP", CP_FLAG_BOOL_VALUE,
     F(no_fused_interp),
     "restore the direct shade path's separate cp_fs_interpolate launch "
     "instead of the slim compaction plus in-shader interpolation" },
   { "CUDAVK_NO_INLINE_FS", CP_FLAG_BOOL_VALUE, F(no_inline_fs),
     "restore separate interpolation plus the resource-isolated classic "
     "fragment binary instead of same-LLVM inline interpolation" },
   { "CUDAVK_INLINE_FS", CP_FLAG_BOOL_VALUE, F(inline_fs),
     "opt in to same-LLVM fragment interpolation; default stays on the "
     "pre-inline fused path because dual module ownership is expensive" },
   { "CUDAVK_FORCE_FUSED_FS", CP_FLAG_BOOL_VALUE, F(force_fused_fs),
     "force pre-inline fused interpolation execution with its isolated tuner "
     "(overrides NO_FUSED flags; "
     "NO_INLINE_FS takes precedence)" },
   { "CUDAVK_NO_FUSED_RAST", CP_FLAG_BOOL_VALUE,
     F(no_fused_rast),
     "restore the separate clip and rasterize-stage1 launches instead of "
     "the fused clip+classify kernel; stages 2 and 3 are separate either "
     "way" },
   { "CUDAVK_NO_PRIM_REFS", CP_FLAG_BOOL_VALUE, F(no_prim_refs),
     "copy every accepted clipped primitive into contiguous scratch instead "
     "of publishing a reference to immutable vertex-shader output" },
   { "CUDAVK_NO_SETUP_CACHE", CP_FLAG_BOOL_VALUE, F(no_setup_cache),
     "recompute huge-primitive setup independently in every stage-3 tile" },
   { "CUDAVK_FORCE_PASS_FALLBACK", CP_FLAG_BOOL_VALUE,
     F(force_pass_fallback),
     "make every blended episode take the classic re-execution fallback; the "
     "output must be identical, which is what makes it a test" },
   { "CUDAVK_TILE_CENSUS", CP_FLAG_UINT, F(tile_census),
     "tile edge in pixels for the blended tile-bin shader census; 0 is off, "
     "and it changes no rendering", .dflt = 0, .has_range = true,
     .lo = 0, .hi = 256 },
   { "CUDAVK_TILE_CENSUS_EVERY", CP_FLAG_UINT, F(tile_census_every),
     "report the tile census this often, in episodes", .dflt = 2000,
     .has_range = true, .lo = 1, .hi = 1000000 },
   { "CUDAVK_TILED_OPAQUE", CP_FLAG_BOOL_VALUE, F(tiled_opaque),
     "use the experimental opaque episode sort-middle tile rasterizer" },
   { "CUDAVK_TILED_OPAQUE_CENSUS", CP_FLAG_BOOL_VALUE,
     F(tiled_opaque_census),
     "collect opaque tile population statistics without changing rendering" },
   { "CUDAVK_UNSAFE_NO_OVERFLOW", CP_FLAG_BOOL_VALUE,
     F(unsafe_no_overflow),
     "skip episode overflow readback and assume fragment/quad arrays fit; "
     "unsafe diagnostic only" },
   { "CUDAVK_UNSAFE_FORCE_OPAQUE", CP_FLAG_BOOL_VALUE,
     F(unsafe_force_opaque),
     "force every draw's blend state to disabled and let the result reach an "
     "opaque episode; renders wrong output, diagnostic upper bound for the "
     "cost of the blended path" },
   { "CUDAVK_NO_BATCH", CP_FLAG_BOOL_PRESENCE, F(no_batch),
     "disable draw batching entirely" },
   { "CUDAVK_NO_BATCH_BLEND", CP_FLAG_BOOL_PRESENCE, F(no_batch_blend),
     "batch opaque draws only: a blended draw never joins a batch, so it "
     "never reaches a pass episode and builds, sorts and peels its own "
     "A-buffer (23.49 ms against 8.79 on Crossroads)" },
   { "CUDAVK_NO_MERGE_SCISSOR", CP_FLAG_BOOL_PRESENCE, F(no_merge_scissor),
     "require an equal scissor before two blended draws merge, as the front "
     "end did before it trusted the renderer's per-draw rectangles" },
   { "CUDAVK_KEEP_VOFF", CP_FLAG_BOOL_PRESENCE, F(keep_voff),
     "put the vertex offset back in the merge key; it is the largest merge "
     "blocker there is, 13,281 separations of 38,155 on Crossroads" },
   { "CUDAVK_KEEP_INSTKEY", CP_FLAG_BOOL_PRESENCE, F(keep_instkey),
     "put the instance count back in the merge key, which the renderer's "
     "per-draw instance_counts[] row makes unnecessary" },
   { "CUDAVK_KEEP_PUSHKEY", CP_FLAG_BOOL_PRESENCE, F(keep_pushkey),
     "put the push-constant block back in the merge key, so draws that push "
     "different constants stop merging" },
   { "CUDAVK_WAIT_SPIN_US", CP_FLAG_OPT_INT, F(wait_spin_us),
     "conversion probe: busy-wait this many microseconds after each episode "
     "drain returns; the slope of frame time against injected time is the "
     "site's conversion factor" },
   { "CUDAVK_WAIT_SPIN_BEFORE_US", CP_FLAG_OPT_INT, F(wait_spin_before_us),
     "conversion probe: busy-wait this many microseconds before each episode "
     "drain's sync; time absorbed by the wait rather than added to the frame "
     "is what a deferral mechanism could relocate for free" },
   { "CUDAVK_KEEP_IBKEY", CP_FLAG_BOOL_PRESENCE, F(keep_ibkey),
     "put the index buffer back in the merge key, which the slice table's "
     "per-draw base makes unnecessary; 23,276 separations on the occlusion "
     "capture" },
   { "CUDAVK_KEEP_VBKEY", CP_FLAG_BOOL_PRESENCE, F(keep_vbkey),
     "put the vertex-buffer bindings back in the merge key; the per-draw "
     "elem_bases rows carry them, and this was the largest merge blocker on "
     "the occlusion capture, 144,008 separations of 332,027" },
   { "CUDAVK_NO_BINCACHE", CP_FLAG_BOOL_PRESENCE, F(no_bincache),
     "disable the compiled-kernel binary cache" },

   /* ---- A-buffer ---- */
   { "CUDAVK_ABUFFER_VERIFY", CP_FLAG_BOOL_VALUE, F(abuffer_verify),
     "check A-buffer output against the direct path; excludes compositing" },
   { "CUDAVK_ABUFFER_VERIFY_DRAWS", CP_FLAG_UINT, F(abuffer_verify_draws),
     "how many draws to verify before giving up", .dflt = 8 },
   { "CUDAVK_ABUFFER_COMPOSITE", CP_FLAG_BOOL_VALUE, F(abuffer_composite),
     "composite the A-buffer; defaults on unless VERIFY is set", .dflt = 1 },
   { "CUDAVK_ABUFFER_TIMING", CP_FLAG_BOOL_VALUE, F(abuffer_timing),
     "per-draw CUDA-event breakdown; costs a drain per draw" },
   { "CUDAVK_ABUFFER_DEBUG", CP_FLAG_BOOL_VALUE, F(abuffer_debug),
     "report which draws were eligible for the A-buffer, and why not" },
   { "CUDAVK_ABUFFER_LAYERS", CP_FLAG_UINT, F(abuffer_layers),
     "cap A-buffer layers per pixel; 0 uses the built-in limit",
     .empty_is_unset = true },
   { "CUDAVK_ABUF_MIN_TRIS", CP_FLAG_UINT, F(abuf_min_tris),
     "triangles a blended batch must have before the A-buffer's fixed cost is "
     "worth paying; 0, the default, means always, and is deliberate — with "
     "the bootstrap scan fixed Crossroads measured 10.9 ms at 0 against 33.0 "
     "at 256", .dflt = 0, .has_range = true, .lo = 0, .hi = UINT32_MAX },
   { "CUDAVK_ABUF_COMPILE", CP_FLAG_OPT_BOOL, F(abuf_compile),
     "force the A-buffer branches in (1) or out (0) of the NVRTC build",
     .empty_is_unset = true },

   /* ---- draw batching ---- */
   { "CUDAVK_BATCH_MAX", CP_FLAG_UINT, F(batch_max),
     "cap draws per batch; 1 must stay bit-identical to NO_BATCH",
     .dflt = CP_MAX_BATCH_DRAWS, .has_range = true, .lo = 1,
     .hi = CP_MAX_BATCH_DRAWS, .empty_is_unset = true },

   /* ---- CUDA context ---- */
   { "CUDAVK_CTX_SCHED", CP_FLAG_ENUM, F(ctx_sched),
     "GPU-wait behaviour: auto | spin | yield | blocking; blocking only helps on "
     "an idle machine, yield only under CPU contention",
     .dflt = CP_CTX_SCHED_AUTO, .values = ctx_scheds },
   { "CUDAVK_CTX_CHECK", CP_FLAG_BOOL_PRESENCE, F(ctx_check),
     "verify every CUDA call runs in this device's context; names the caller "
     "that escaped an entry-point scope" },

   /* ---- small-allocation arena ---- */
   { "CUDAVK_SMALL_ALLOC", CP_FLAG_ENUM, F(small_alloc),
     "arena residency: advise | pinned | managed | blocksonly | off",
     .dflt = CP_ARENA_ADVISE, .values = arena_modes },
   { "CUDAVK_SMALL_ALLOC_MAX", CP_FLAG_U64, F(small_alloc_max),
     "allocations up to this many bytes come from the arena",
     .dflt = CP_ARENA_DEFAULT_MAX, .has_range = true, .lo = 0,
     .hi = CP_ARENA_MAX_SIZE },
   { "CUDAVK_SMALL_ALLOC_WARMUP", CP_FLAG_U64, F(small_alloc_warmup),
     "allocations to let past before the arena opens",
     .dflt = CP_ARENA_DEFAULT_WARMUP },
   { "CUDAVK_SMALL_ALLOC_STATS", CP_FLAG_BOOL_PRESENCE, F(small_alloc_stats),
     "dump the allocation mix and arena hit rates at exit" },

   /* ---- shader compilation ---- */
   { "CUDAVK_NO_REGCAP", CP_FLAG_BOOL_PRESENCE, F(no_regcap),
     "disable the register cap and the occupancy trial that tunes it" },
   { "CUDAVK_REGCAP_STATIC", CP_FLAG_BOOL_PRESENCE, F(regcap_static),
     "cap registers from a static estimate instead of the trial" },
   { "CUDAVK_MAX_REGISTERS", CP_FLAG_UINT, F(max_registers),
     "force a register cap on every shader; 0 leaves it to the driver" },
   { "CUDAVK_LAUNCH_BOUNDS", CP_FLAG_UINT, F(launch_bounds),
     "emit maxntidx metadata with this block size; 0 emits none" },
   { "CUDAVK_TUNE_VETO", CP_FLAG_FLOAT, F(tune_veto),
     "how much worse the capped build may be before it is refused",
     .fdflt = CP_TUNE_VETO },
   { "CUDAVK_SHADER_STATS", CP_FLAG_BOOL_PRESENCE, F(shader_stats),
     "report register counts and occupancy-trial outcomes" },
   { "CUDAVK_DUMP_NIR", CP_FLAG_BOOL_PRESENCE, F(dump_nir),
     "print each shader's NIR" },
   { "CUDAVK_DUMP_IR", CP_FLAG_BOOL_PRESENCE, F(dump_ir),
     "print each shader's LLVM IR" },
   { "CUDAVK_DUMP_PTX", CP_FLAG_BOOL_PRESENCE, F(dump_ptx),
     "print each shader's generated PTX" },
   { "CUDAVK_SCALARIZE", CP_FLAG_ENUM, F(scalarize),
     "which ALU operations to scalarise: all | none | basic | sel | alu | "
     "move | intr | rest; all is the NULL-filter pass every sample is "
     "rendered with, and the classes are there to bisect a backend bug",
     .dflt = CP_SCALARIZE_ALL, .values = scalarize_classes },
   { "CUDAVK_NO_HOIST_INPUTS", CP_FLAG_BOOL_PRESENCE, F(no_hoist_inputs),
     "stop hoisting a vertex shader's input loads above the arithmetic that "
     "consumes them; that hoist is 6.14 against 8.63 ms on instancing" },
   { "CUDAVK_NO_REG_SSA", CP_FLAG_BOOL_PRESENCE, F(no_reg_ssa),
     "leave NIR registers alone before the backend, so each becomes an alloca "
     "and the NVPTX backend gives the kernel a __local_depot" },
   { "CUDAVK_KEEP_SMALL_DYNAMIC_REGCAP", CP_FLAG_BOOL_PRESENCE,
     F(keep_small_dynamic_regcap),
     "keep the tuned register cap on small fragment shaders that call the "
     "dynamic sampler helper; they cannot amortise its spills" },

   /* ---- rasterizer tuning (NVRTC -D options) ---- */
   { "CUDAVK_SMALL_THRESHOLD", CP_FLAG_OPT_INT, F(small_threshold),
     "triangle area below which the small-primitive path is used",
     .empty_is_unset = true },
   { "CUDAVK_MEDIUM_THRESHOLD", CP_FLAG_OPT_INT, F(medium_threshold),
     "triangle area below which the medium-primitive path is used",
     .empty_is_unset = true },
   { "CUDAVK_POINT_THRESHOLD", CP_FLAG_OPT_INT, F(point_threshold),
     "triangle area below which a primitive is rasterized as a point",
     .empty_is_unset = true },
   { "CUDAVK_TILE_BOUND", CP_FLAG_OPT_INT, F(tile_bound),
     "0 walks the whole tile; 1 walks only the primitive's bounding box",
     .empty_is_unset = true },
};

#undef F

/*
 * Names this driver no longer reads, and what to use instead.
 *
 * This exists because a variable that changes nothing looks exactly like a
 * variable that works. CPVK_BATCH and CPVK_BATCH_BLEND were left exported in
 * one shell, every process launched from it inherited them, and every A/B run
 * after that compared a feature against itself. The numbers reproduced, which
 * is what made them convincing, and they meant nothing. That cost a day.
 *
 * A driver whose behaviour an invisible variable can change must say so, and
 * the only place that can say it is the file that reads the environment. So
 * the retired names live beside the live ones: `now` is the replacement when
 * the switch was renamed, and NULL when nothing reads the name at all any
 * more. Silent when none is set, which is the common case.
 */
static const struct {
   const char *old;
   const char *now;
} retired[] = {
   /* Renamed into the registry: one prefix, one table. */
   /* Default flipped on, so the switch that changes behaviour is now the
    * one that turns it off. The old name is not silently ignored: a script
    * still exporting CUDAVK_OPAQUE_STREAMS=1 would otherwise appear to select
    * the arm it already gets, and =0 would appear to select the old one and
    * would not. */
   { "CUDAVK_OPAQUE_STREAMS",          "CUDAVK_NO_OPAQUE_STREAMS" },

   { "CPVK_DEBUG_ROWS",                "CUDAVK_DEBUG_ROWS" },
   { "CPVK_DEBUG_CLIP",                "CUDAVK_DEBUG_CLIP" },
   { "CPVK_DEBUG_PASS",                "CUDAVK_DEBUG_PASS" },
   { "CPVK_DEBUG_EPISODE",             "CUDAVK_DEBUG_EPISODE" },
   { "CPVK_DEBUG_RT",                  "CUDAVK_DEBUG_RT" },
   { "CPVK_DEBUG_FACES",               "CUDAVK_DEBUG_FACES" },
   { "CPVK_ABUF_MIN_TRIS",             "CUDAVK_ABUF_MIN_TRIS" },
   { "CPVK_NO_BATCH",                  "CUDAVK_NO_BATCH" },
   { "CPVK_NO_BATCH_BLEND",            "CUDAVK_NO_BATCH_BLEND" },
   { "CPVK_NO_MERGE_SCISSOR",          "CUDAVK_NO_MERGE_SCISSOR" },
   { "CPVK_KEEP_VOFF",                 "CUDAVK_KEEP_VOFF" },
   { "CPVK_KEEP_INSTKEY",              "CUDAVK_KEEP_INSTKEY" },
   { "CPVK_KEEP_PUSHKEY",              "CUDAVK_KEEP_PUSHKEY" },
   { "CPVK_SCALARIZE",                 "CUDAVK_SCALARIZE" },
   { "CPVK_NO_HOIST_INPUTS",           "CUDAVK_NO_HOIST_INPUTS" },
   { "CPVK_NO_REG_SSA",                "CUDAVK_NO_REG_SSA" },
   { "CPVK_KEEP_SMALL_DYNAMIC_REGCAP", "CUDAVK_KEEP_SMALL_DYNAMIC_REGCAP" },

   /* No replacement: nothing in the driver reads these at all. The first two
    * outlived the code that consulted them; CPVK_MERGE_PUSH named the opt-in
    * for a behaviour that is now the default, so its inverse
    * CUDAVK_KEEP_PUSHKEY is the switch to reach for and not a rename of it;
    * the last two are the pair that cost the day above. */
   { "CPVK_NO_DESC_KEY",  NULL },
   { "CPVK_DEBUG_PUSH",   NULL },
   { "CPVK_MERGE_PUSH",   NULL },
   { "CPVK_BATCH",        NULL },
   { "CPVK_BATCH_BLEND",  NULL },
};

static void
report_retired(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(retired); i++) {
      if (!getenv(retired[i].old))
         continue;
      if (retired[i].now)
         fprintf(stderr, "cudavk: %s is retired and ignored; use %s\n",
                 retired[i].old, retired[i].now);
      else
         fprintf(stderr, "cudavk: %s is retired and ignored\n",
                 retired[i].old);
   }
}

static struct cp_debug debug_state;
const struct cp_debug *cp_debug = &debug_state;

/* Whether each variable was present in the environment, which is not always
 * recoverable from the parsed value: a flag defaulting on and a flag set to 1
 * land in the same field. apply_couplings() needs the difference. */
static bool present_in_env[ARRAY_SIZE(flags)];

static void *
field(const struct cp_flag_def *f)
{
   return (char *)&debug_state + f->offset;
}

static int64_t
clamp_to_range(const struct cp_flag_def *f, int64_t v)
{
   if (!f->has_range)
      return v;
   if (v < f->lo)
      return f->lo;
   if (v > f->hi)
      return f->hi;
   return v;
}

static void
parse_one(const struct cp_flag_def *f, unsigned index)
{
   const char *v = getenv(f->name);
   bool present = v != NULL;

   if (present && f->empty_is_unset && !*v)
      present = false;

   present_in_env[index] = present;

   switch (f->type) {
   case CP_FLAG_BOOL_PRESENCE:
      *(bool *)field(f) = present;
      break;

   case CP_FLAG_BOOL_VALUE:
      *(bool *)field(f) = present ? atoi(v) != 0 : f->dflt != 0;
      break;

   case CP_FLAG_OPT_BOOL: {
      struct cp_opt_int *o = field(f);
      o->set = present;
      o->value = present ? (atoi(v) != 0) : 0;
      break;
   }

   case CP_FLAG_OPT_INT: {
      struct cp_opt_int *o = field(f);
      o->set = present;
      o->value = present ? atoi(v) : 0;
      break;
   }

   case CP_FLAG_UINT:
      *(unsigned *)field(f) =
         (unsigned)clamp_to_range(f, present ? (int64_t)strtoll(v, NULL, 0) : f->dflt);
      break;

   case CP_FLAG_U64:
      *(uint64_t *)field(f) =
         (uint64_t)clamp_to_range(f, present ? (int64_t)strtoull(v, NULL, 0) : f->dflt);
      break;

   case CP_FLAG_INT:
      *(int *)field(f) =
         (int)clamp_to_range(f, present ? (int64_t)strtoll(v, NULL, 0) : f->dflt);
      break;

   case CP_FLAG_FLOAT:
      *(double *)field(f) = present ? atof(v) : f->fdflt;
      break;

   case CP_FLAG_ENUM: {
      int64_t val = f->dflt;
      if (present) {
         const struct cp_flag_value *m;
         for (m = f->values; m->name; m++)
            if (!strcmp(v, m->name)) {
               val = m->value;
               break;
            }
         if (!m->name) {
            /* Preserved from the hand-written parse: an unrecognised mode is
             * not an error, it selects the first entry's fallback. Saying so
             * is new — CUDAVK_SMALL_ALLOC=advize silently disabled the arena
             * and would have been measured as "advise made no difference". */
            fprintf(stderr, "cudavk: %s='%s' is not a known value — "
                    "falling back to the disabled mode.\n", f->name, v);
            val = 0;
         }
      }
      *(int *)field(f) = (int)val;
      break;
   }
   }
}

static bool
flag_was_set(const char *name)
{
   for (unsigned i = 0; i < ARRAY_SIZE(flags); i++)
      if (!strcmp(flags[i].name, name))
         return present_in_env[i];
   return false;
}

/*
 * The couplings the table cannot express. Each is a real dependency in
 * the code being replaced, not a tidy-up.
 */
static void
apply_couplings(void)
{
   /*
    * Compositing and verifying are exclusive: verification compares the
    * A-buffer's output against the direct path, which needs the direct path's
    * result left alone. So COMPOSITE defaults to !VERIFY, and an explicit
    * COMPOSITE wins outright.
    */
   if (!flag_was_set("CUDAVK_ABUFFER_COMPOSITE"))
      debug_state.abuffer_composite = !debug_state.abuffer_verify;
   if (debug_state.abuffer_composite)
      debug_state.abuffer_verify = false;

   /*
    * The hardware texture path is the default. It was opt-in through
    * iteration 24, which is why the registry carries the revert rather than
    * the opt-in: the whole driver has been measured with it on since, and the
    * enabled path is the better-tested one. Deriving it here rather than
    * inverting sixty use sites keeps the change to one line of behaviour.
    */
   debug_state.texture_cache = !debug_state.no_texture_cache;

   /*
    * Programmatic dependent launch is the default, and the level is the knob
    * rather than the switch. Two names because they answer two questions and
    * a level cannot answer both: CUDAVK_NO_PDL is what the house convention
    * requires -- the one switch that restores the pre-PDL driver, and the one
    * the default table names -- while CUDAVK_PDL says how much of it to keep.
    * CUDAVK_PDL=0 means the same thing as the revert and is the natural end of
    * a bisect; the revert wins over an explicit level, because a reader who
    * writes NO_PDL=1 means it.
    */
   if (debug_state.no_pdl)
      debug_state.pdl = 0;
}

/*
 * CUDAVK_PDL is not a retired name -- it is still read, and every value it
 * ever took still means the same set of links. What changed is the default
 * when it is unset, and that is enough to catch someone out: a script written
 * while PDL was opt-in exports CUDAVK_PDL=1 to mean "on", and now silently
 * selects LESS than the default instead. That is the same trap the
 * CUDAVK_OPAQUE_STREAMS entry above exists to close, so it gets the same
 * treatment -- a line on stderr -- rather than an entry in a table that says
 * "retired and ignored" about a variable that is neither.
 *
 * Only fires when the variable is set, so a bisect says out loud which level
 * it is on and a default run says nothing.
 */
static void
report_pdl_level(void)
{
   if (!flag_was_set("CUDAVK_PDL") || debug_state.no_pdl)
      return;
   if (debug_state.pdl < CP_PDL_LEVEL_DEFAULT)
      fprintf(stderr, "cudavk: CUDAVK_PDL=%u is BELOW the default of %u: "
              "programmatic dependent launch is partly disabled. Unset it for "
              "the default, or CUDAVK_NO_PDL=1 to turn it off entirely.\n",
              debug_state.pdl, (unsigned)CP_PDL_LEVEL_DEFAULT);
}

static const char *
type_name(enum cp_flag_type t)
{
   switch (t) {
   case CP_FLAG_BOOL_PRESENCE: return "bool(presence)";
   case CP_FLAG_BOOL_VALUE:    return "bool(value)";
   case CP_FLAG_OPT_BOOL:      return "bool(optional)";
   case CP_FLAG_OPT_INT:       return "int(optional)";
   case CP_FLAG_UINT:          return "uint";
   case CP_FLAG_U64:           return "uint64";
   case CP_FLAG_INT:           return "int";
   case CP_FLAG_FLOAT:         return "float";
   case CP_FLAG_ENUM:          return "enum";
   }
   return "?";
}

static void
format_current(const struct cp_flag_def *f, char *buf, size_t len)
{
   switch (f->type) {
   case CP_FLAG_BOOL_PRESENCE:
   case CP_FLAG_BOOL_VALUE:
      snprintf(buf, len, "%s", *(const bool *)field(f) ? "on" : "off");
      break;
   case CP_FLAG_OPT_BOOL: {
      const struct cp_opt_int *o = field(f);
      snprintf(buf, len, "%s", !o->set ? "unset" : (o->value ? "on" : "off"));
      break;
   }
   case CP_FLAG_OPT_INT: {
      const struct cp_opt_int *o = field(f);
      if (o->set)
         snprintf(buf, len, "%d", o->value);
      else
         snprintf(buf, len, "unset");
      break;
   }
   case CP_FLAG_UINT:
      snprintf(buf, len, "%u", *(const unsigned *)field(f));
      break;
   case CP_FLAG_U64:
      snprintf(buf, len, "%" PRIu64, *(const uint64_t *)field(f));
      break;
   case CP_FLAG_INT:
      snprintf(buf, len, "%d", *(const int *)field(f));
      break;
   case CP_FLAG_FLOAT:
      snprintf(buf, len, "%g", *(const double *)field(f));
      break;
   case CP_FLAG_ENUM: {
      int cur = *(const int *)field(f);
      for (const struct cp_flag_value *m = f->values; m->name; m++)
         if (m->value == cur) {
            snprintf(buf, len, "%s", m->name);
            return;
         }
      snprintf(buf, len, "%d", cur);
      break;
   }
   }
}

static void
format_default(const struct cp_flag_def *f, char *buf, size_t len)
{
   switch (f->type) {
   case CP_FLAG_BOOL_PRESENCE:
      snprintf(buf, len, "off");
      break;
   case CP_FLAG_BOOL_VALUE:
      snprintf(buf, len, "%s", f->dflt ? "on" : "off");
      break;
   case CP_FLAG_OPT_BOOL:
   case CP_FLAG_OPT_INT:
      snprintf(buf, len, "unset");
      break;
   case CP_FLAG_FLOAT:
      snprintf(buf, len, "%g", f->fdflt);
      break;
   case CP_FLAG_ENUM:
      for (const struct cp_flag_value *m = f->values; m->name; m++)
         if (m->value == f->dflt) {
            snprintf(buf, len, "%s", m->name);
            return;
         }
      snprintf(buf, len, "%" PRId64, f->dflt);
      break;
   default:
      snprintf(buf, len, "%" PRId64, f->dflt);
      break;
   }
}

static void
cp_debug_help(void)
{
   fprintf(stderr,
      "cudavk: %zu environment switches. Value shown is what this process "
      "resolved.\n\n", ARRAY_SIZE(flags));
   fprintf(stderr, "  %-32s %-15s %-9s %-9s %s\n",
           "NAME", "TYPE", "DEFAULT", "NOW", "MEANING");

   for (unsigned i = 0; i < ARRAY_SIZE(flags); i++) {
      char now[64], dflt[64];
      format_current(&flags[i], now, sizeof(now));
      format_default(&flags[i], dflt, sizeof(dflt));
      fprintf(stderr, "  %-32s %-15s %-9s %-9s %s\n",
              flags[i].name, type_name(flags[i].type), dflt, now,
              flags[i].help);
   }

   fprintf(stderr,
      "\n  bool(presence) is set by the variable existing at all, so =0 turns "
      "it ON.\n"
      "  bool(value) reads the value, so =0 turns it off.\n");
}

void
cp_debug_init(void)
{
   static bool done;
   if (done)
      return;
   done = true;

   for (unsigned i = 0; i < ARRAY_SIZE(flags); i++)
      parse_one(&flags[i], i);

   apply_couplings();

   report_retired();
   report_pdl_level();

   if (getenv("CUDAVK_HELP"))
      cp_debug_help();
}
