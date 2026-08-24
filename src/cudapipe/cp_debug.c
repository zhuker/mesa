/*
 * The registry. See cp_debug.h for why this exists.
 *
 * Every entry states how the variable is parsed today, not how it would be
 * parsed if the naming and the parsing had been designed together. Both
 * boolean kinds are real and both have to survive: presence-only flags treat
 * `=0` as ON, and changing that would silently switch tracing off for anyone
 * who wrote CUDAPIPE_DEBUG_DRAW=0 in a script and has been getting tracing
 * ever since.
 */

#include "cp_debug.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/macros.h"

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
 * Grouped as the subsystems are, and in the order CUDAPIPE_HELP prints them.
 * The help text is one line and says what the flag is *for* — "trace every
 * draw" rather than "sets cp->debug_draw" — because the reader is someone who
 * has a bug and no idea which of these to reach for.
 */
static const struct cp_flag_def flags[] = {
   /* ---- tracing ---- */
   { "CUDAPIPE_DEBUG_DRAW", CP_FLAG_BOOL_PRESENCE, F(debug_draw),
     "trace every draw: geometry, attachments, and why a draw was skipped" },
   { "CUDAPIPE_DEBUG_TEX", CP_FLAG_BOOL_PRESENCE, F(debug_tex),
     "trace sampler and texture-handle setup" },
   { "CUDAPIPE_PLAN_STATS", CP_FLAG_BOOL_VALUE, F(plan_stats),
     "report at teardown what deciding one draw at a time costs: batch keys "
     "built, pairwise merge tests, flushes, pass episodes closed and every "
     "reactive reallocation, with per-scope averages" },
   { "CUDAPIPE_SPEC_STATS", CP_FLAG_BOOL_VALUE, F(spec_stats),
     "report at teardown how many fragment launches used a sampler-specialised "
     "kernel, and which shaders sample textures the specialiser could not "
     "match; a silent drop to zero is a performance regression with no error" },
   { "CUDAPIPE_TEXTURE_CACHE_STATS", CP_FLAG_BOOL_VALUE, F(texture_cache_stats),
     "report hardware-texture fragment execution hits and software fallbacks" },
   { "CUDAPIPE_DEBUG_VFETCH", CP_FLAG_BOOL_PRESENCE, F(debug_vfetch),
     "dump what the GPU vertex fetch gathered; syncs, so debug-only" },
   { "CUDAPIPE_DEBUG_WORK", CP_FLAG_BOOL_PRESENCE, F(debug_work),
     "report how much of the shading launch did work; syncs" },
   { "CUDAPIPE_DEBUG_DISCARD", CP_FLAG_BOOL_PRESENCE, F(debug_discard),
     "report covered and discarded fragment counts per pass; syncs" },
   { "CUDAPIPE_DEBUG_FS", CP_FLAG_BOOL_PRESENCE, F(debug_fs),
     "dump fragment-shader inputs and outputs per pixel; syncs" },
   { "CUDAPIPE_DEBUG_FS_VSTEP", CP_FLAG_UINT, F(debug_fs_vstep),
     "with DEBUG_FS, print every Nth pixel", .dflt = 1, .has_range = true, .lo = 1, .hi = UINT32_MAX },
   { "CUDAPIPE_DEBUG_FS_ROW", CP_FLAG_INT, F(debug_fs_row),
     "with DEBUG_FS, restrict the dump to one framebuffer row", .dflt = -1 },
   { "CUDAPIPE_DEBUG_LAUNCH", CP_FLAG_BOOL_PRESENCE, F(debug_launch),
     "trace compute dispatch: bound UBO and SSBO pointers" },
   { "CUDAPIPE_DEBUG_TIME", CP_FLAG_BOOL_PRESENCE, F(debug_time),
     "time the pipeline stages" },
   { "CUDAPIPE_DEBUG_BATCH", CP_FLAG_BOOL_PRESENCE, F(debug_batch),
     "report why each draw batch ended" },
   { "CUDAPIPE_DEBUG_BATCHDIFF", CP_FLAG_BOOL_PRESENCE, F(debug_batchdiff),
     "report which state field broke a batch, field by field" },
   { "CUDAPIPE_DEBUG_PASSSEQ", CP_FLAG_BOOL_VALUE, F(debug_passseq),
     "log one line per framebuffer bind and per draw: shaders, blendedness, "
     "eligibility — the raw material for pass-structure statistics" },
   { "CUDAPIPE_FRAG_CENSUS", CP_FLAG_BOOL_PRESENCE, F(frag_census),
     "count fragments per draw; also compiles the instrumented kernels in" },
   { "CUDAPIPE_NVTX", CP_FLAG_BOOL_PRESENCE, F(nvtx),
     "push an NVTX range around each draw and stage, for nsys" },

   /* ---- subsystem switches ---- */
   { "CUDAPIPE_NO_ABUFFER", CP_FLAG_BOOL_PRESENCE, F(no_abuffer),
     "disable the A-buffer; blended draws go back to the direct path" },
   { "CUDAPIPE_NO_ABUF_BATCH", CP_FLAG_BOOL_PRESENCE, F(no_abuf_batch),
     "disable batching of A-buffer draws" },
   { "CUDAPIPE_NO_PASS_EPISODE", CP_FLAG_BOOL_VALUE, F(no_pass_episode),
     "disable pass episodes: consecutive blended batches stop sharing one "
     "A-buffer build and drain" },
   { "CUDAPIPE_UPLOAD_STATS", CP_FLAG_BOOL_VALUE, F(upload_stats),
     "count small host-to-device copies, clears, context syncs and upload "
     "ring wraps per call site, and report them at teardown" },
   { "CUDAPIPE_NO_META_FOLD", CP_FLAG_BOOL_VALUE, F(no_meta_fold),
     "upload the vertex stage's vcount and stride as their own block again "
     "instead of carrying them in the argument block's scalar area" },
   { "CUDAPIPE_NO_COUNTER_BLOCK", CP_FLAG_BOOL_VALUE, F(no_counter_block),
     "clear the A-buffer's scalar counters one at a time again instead of "
     "clearing the whole counter block once per draw or episode" },
   { "CUDAPIPE_NO_UPLOAD_COALESCE", CP_FLAG_BOOL_VALUE, F(no_upload_coalesce),
     "send one host-to-device copy per upload block again instead of one per "
     "launch boundary" },
   { "CUDAPIPE_UPLOAD_FLUSH_FAIL_AT", CP_FLAG_UINT, F(upload_flush_fail_at),
     "fault injection: fail the Nth coalesced upload flush" },
   { "CUDAPIPE_NO_FETCH_FOLD", CP_FLAG_BOOL_VALUE, F(no_fetch_fold),
     "clear the clip and raster queue counters with their own device clears "
     "again instead of seeding them inside cp_vertex_fetch" },
   { "CUDAPIPE_FUSED_VFETCH", CP_FLAG_BOOL_VALUE, F(fused_vfetch),
     "gather vertex attributes inside the vertex shader, from bitcode inlined "
     "into it, instead of launching cp_vertex_fetch before it", .dflt = 1 },
   { "CUDAPIPE_NO_FUSED_VFETCH", CP_FLAG_BOOL_VALUE, F(no_fused_vfetch),
     "do not even build the fused vertex execution: the resource-isolated "
     "control, and the revert once the fusion is the default" },
   { "CUDAPIPE_FS_GRID_WAVES", CP_FLAG_UINT, F(fs_grid_waves),
     "size the fragment grid to fill the machine this many times over and let "
     "it grid-stride, instead of covering the framebuffer's worst case; 0 "
     "keeps the 4096-block cap, which is the only bound either way" },
   { "CUDAPIPE_VFETCH_DECLINE_NTH", CP_FLAG_UINT, F(vfetch_decline_nth),
     "fault injection: refuse the fused execution for the Nth vertex shader "
     "compiled, or for every one at 4294967295, after building it either way" },
   { "CUDAPIPE_VFETCH_SKIP_SEED", CP_FLAG_BOOL_VALUE, F(vfetch_skip_seed),
     "fault injection: a fused draw does not seed the clip and raster "
     "counters, and the host still skips their clears -- the negative control "
     "for the seeding moving into the fused vertex shader" },
   { "CUDAPIPE_NO_OPAQUE_EPISODE", CP_FLAG_BOOL_VALUE,
     F(no_opaque_episode),
     "disable consecutive opaque-run visibility deferral" },
   { "CUDAPIPE_NO_SAMPLER_VARIANT", CP_FLAG_BOOL_VALUE,
     F(no_sampler_variant),
     "disable literal-state fragment sampler variants" },
   { "CUDAPIPE_TEXTURE_CACHE", CP_FLAG_BOOL_VALUE, F(texture_cache),
     "opt in to same-LLVM direct CUDA hardware-texture fragment execution" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_TABLE_UPLOAD_AT", CP_FLAG_UINT,
     F(texture_cache_fail_table_upload_at),
     "fault injection: fail the Nth hardware texture table upload enqueue" },
   { "CUDAPIPE_TEXTURE_CACHE_PURGE_AT_PREFLIGHT", CP_FLAG_UINT,
     F(texture_cache_purge_at_preflight),
     "fault injection: purge derived textures before the Nth HW preflight" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_AUTHORITATIVE_ALLOC_AT_PREFLIGHT",
     CP_FLAG_UINT, F(texture_cache_fail_authoritative_alloc_at_preflight),
     "fault injection: OOM one renderer allocation at the Nth HW preflight" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_AFTER_BOUNDED_GROUP0", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_after_bounded_group0),
     "fault injection: latch fatal after bounded episode FS group zero" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_AFTER_MAIN_GROUP0", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_after_main_group0),
     "fault injection: latch fatal after main episode FS group zero" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_AFTER_FS_ENQUEUE", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_after_fs_enqueue),
     "fault injection: sync then latch fatal after one hardware FS enqueue" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_ARRAY_ALLOC_AT", CP_FLAG_UINT,
     F(texture_cache_fail_array_alloc_at),
     "fault injection: OOM the Nth derived CUDA array allocation" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_OBJECT_CREATE_AT", CP_FLAG_UINT,
     F(texture_cache_fail_object_create_at),
     "fault injection: OOM the Nth CUDA texture-object creation" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_CONVERSION_ENQUEUE_AT", CP_FLAG_UINT,
     F(texture_cache_fail_conversion_enqueue_at),
     "fault injection: fail the Nth converted-cache kernel enqueue fatally" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_FS_ARG_BEGIN", CP_FLAG_BOOL_VALUE,
     F(texture_cache_fail_fs_arg_begin),
     "fault injection: fail first hardware FS-argument reservation fatally" },
   { "CUDAPIPE_TEXTURE_CACHE_FAIL_CREATE_STAGE", CP_FLAG_UINT,
     F(texture_cache_fail_create_stage),
     "fault injection: fail cache create stage 1=event or 2=surface" },
   { "CUDAPIPE_NO_ABUF_SHORT_SORT", CP_FLAG_BOOL_VALUE,
     F(no_abuf_short_sort),
     "disable the single-thread short A-buffer run sorter" },
   { "CUDAPIPE_NO_ABUF_WARP_BUCKET", CP_FLAG_BOOL_VALUE,
     F(no_abuf_warp_bucket),
     "disable warp-aggregated A-buffer segment bucketing atomics" },
   { "CUDAPIPE_ABUF_SHORT_SORT_MAX", CP_FLAG_UINT,
     F(abuf_short_sort_max),
     "largest A-buffer run sorted by one thread", .dflt = 16,
     .has_range = true, .lo = 2, .hi = 64 },
   { "CUDAPIPE_FLUSH_DRAIN", CP_FLAG_BOOL_VALUE, F(flush_drain),
     "restore the draining flush: cp_flush waits for the whole device and "
     "rewinds the arenas in place instead of ping-ponging generations" },
   { "CUDAPIPE_NO_SEG_MERGE", CP_FLAG_BOOL_VALUE, F(no_seg_merge),
     "disable merged shading groups: every episode segment shades in its "
     "own launch group, as before" },
   { "CUDAPIPE_NO_ABUF_APPEND", CP_FLAG_BOOL_VALUE, F(no_abuf_append),
     "disable the single-pass A-buffer build: the count pass stops appending "
     "(pixel, prim) records and the fill rasterizes a second time" },
   { "CUDAPIPE_NO_FUSED_ABUF_INTERP", CP_FLAG_BOOL_VALUE,
     F(no_fused_abuf_interp),
     "launch A-buffer interpolation separately instead of calling it from "
     "the generated fragment shader; performance experiment only" },
   { "CUDAPIPE_NO_FUSED_INTERP", CP_FLAG_BOOL_VALUE,
     F(no_fused_interp),
     "restore the direct shade path's separate cp_fs_interpolate launch "
     "instead of the slim compaction plus in-shader interpolation" },
   { "CUDAPIPE_NO_INLINE_FS", CP_FLAG_BOOL_VALUE, F(no_inline_fs),
     "restore separate interpolation plus the resource-isolated classic "
     "fragment binary instead of same-LLVM inline interpolation" },
   { "CUDAPIPE_INLINE_FS", CP_FLAG_BOOL_VALUE, F(inline_fs),
     "opt in to same-LLVM fragment interpolation; default stays on the "
     "pre-inline fused path because dual module ownership is expensive" },
   { "CUDAPIPE_FORCE_FUSED_FS", CP_FLAG_BOOL_VALUE, F(force_fused_fs),
     "force pre-inline fused interpolation execution with its isolated tuner "
     "(overrides NO_FUSED flags; "
     "NO_INLINE_FS takes precedence)" },
   { "CUDAPIPE_NO_FUSED_RAST", CP_FLAG_BOOL_VALUE,
     F(no_fused_rast),
     "restore the separate clip and rasterize-stage1 launches instead of "
     "the fused clip+classify kernel; stages 2 and 3 are separate either "
     "way" },
   { "CUDAPIPE_NO_PRIM_REFS", CP_FLAG_BOOL_VALUE, F(no_prim_refs),
     "copy every accepted clipped primitive into contiguous scratch instead "
     "of publishing a reference to immutable vertex-shader output" },
   { "CUDAPIPE_NO_SETUP_CACHE", CP_FLAG_BOOL_VALUE, F(no_setup_cache),
     "recompute huge-primitive setup independently in every stage-3 tile" },
   { "CUDAPIPE_FORCE_PASS_FALLBACK", CP_FLAG_BOOL_VALUE,
     F(force_pass_fallback),
     "make every blended episode take the classic re-execution fallback; the "
     "output must be identical, which is what makes it a test" },
   { "CUDAPIPE_TILE_CENSUS", CP_FLAG_UINT, F(tile_census),
     "tile edge in pixels for the blended tile-bin shader census; 0 is off, "
     "and it changes no rendering", .dflt = 0, .has_range = true,
     .lo = 0, .hi = 256 },
   { "CUDAPIPE_TILE_CENSUS_EVERY", CP_FLAG_UINT, F(tile_census_every),
     "report the tile census this often, in episodes", .dflt = 2000,
     .has_range = true, .lo = 1, .hi = 1000000 },
   { "CUDAPIPE_TILED_OPAQUE", CP_FLAG_BOOL_VALUE, F(tiled_opaque),
     "use the experimental opaque episode sort-middle tile rasterizer" },
   { "CUDAPIPE_TILED_OPAQUE_CENSUS", CP_FLAG_BOOL_VALUE,
     F(tiled_opaque_census),
     "collect opaque tile population statistics without changing rendering" },
   { "CUDAPIPE_UNSAFE_NO_OVERFLOW", CP_FLAG_BOOL_VALUE,
     F(unsafe_no_overflow),
     "skip episode overflow readback and assume fragment/quad arrays fit; "
     "unsafe diagnostic only" },
   { "CUDAPIPE_NO_BATCH", CP_FLAG_BOOL_PRESENCE, F(no_batch),
     "disable draw batching entirely" },
   { "CUDAPIPE_NO_BINCACHE", CP_FLAG_BOOL_PRESENCE, F(no_bincache),
     "disable the compiled-kernel binary cache" },

   /* ---- A-buffer ---- */
   { "CUDAPIPE_ABUFFER_VERIFY", CP_FLAG_BOOL_VALUE, F(abuffer_verify),
     "check A-buffer output against the direct path; excludes compositing" },
   { "CUDAPIPE_ABUFFER_VERIFY_DRAWS", CP_FLAG_UINT, F(abuffer_verify_draws),
     "how many draws to verify before giving up", .dflt = 8 },
   { "CUDAPIPE_ABUFFER_COMPOSITE", CP_FLAG_BOOL_VALUE, F(abuffer_composite),
     "composite the A-buffer; defaults on unless VERIFY is set", .dflt = 1 },
   { "CUDAPIPE_ABUFFER_TIMING", CP_FLAG_BOOL_VALUE, F(abuffer_timing),
     "per-draw CUDA-event breakdown; costs a drain per draw" },
   { "CUDAPIPE_ABUFFER_DEBUG", CP_FLAG_BOOL_VALUE, F(abuffer_debug),
     "report which draws were eligible for the A-buffer, and why not" },
   { "CUDAPIPE_ABUFFER_LAYERS", CP_FLAG_UINT, F(abuffer_layers),
     "cap A-buffer layers per pixel; 0 uses the built-in limit",
     .empty_is_unset = true },
   { "CUDAPIPE_ABUF_COMPILE", CP_FLAG_OPT_BOOL, F(abuf_compile),
     "force the A-buffer branches in (1) or out (0) of the NVRTC build",
     .empty_is_unset = true },

   /* ---- draw batching ---- */
   { "CUDAPIPE_BATCH_MAX", CP_FLAG_UINT, F(batch_max),
     "cap draws per batch; 1 must stay bit-identical to NO_BATCH",
     .dflt = CP_MAX_BATCH_DRAWS, .has_range = true, .lo = 1,
     .hi = CP_MAX_BATCH_DRAWS, .empty_is_unset = true },

   /* ---- small-allocation arena ---- */
   { "CUDAPIPE_SMALL_ALLOC", CP_FLAG_ENUM, F(small_alloc),
     "arena residency: advise | pinned | managed | blocksonly | off",
     .dflt = CP_ARENA_ADVISE, .values = arena_modes },
   { "CUDAPIPE_SMALL_ALLOC_MAX", CP_FLAG_U64, F(small_alloc_max),
     "allocations up to this many bytes come from the arena",
     .dflt = CP_ARENA_DEFAULT_MAX, .has_range = true, .lo = 0,
     .hi = CP_ARENA_MAX_SIZE },
   { "CUDAPIPE_SMALL_ALLOC_WARMUP", CP_FLAG_U64, F(small_alloc_warmup),
     "allocations to let past before the arena opens",
     .dflt = CP_ARENA_DEFAULT_WARMUP },
   { "CUDAPIPE_SMALL_ALLOC_STATS", CP_FLAG_BOOL_PRESENCE, F(small_alloc_stats),
     "dump the allocation mix and arena hit rates at exit" },

   /* ---- shader compilation ---- */
   { "CUDAPIPE_NO_REGCAP", CP_FLAG_BOOL_PRESENCE, F(no_regcap),
     "disable the register cap and the occupancy trial that tunes it" },
   { "CUDAPIPE_REGCAP_STATIC", CP_FLAG_BOOL_PRESENCE, F(regcap_static),
     "cap registers from a static estimate instead of the trial" },
   { "CUDAPIPE_MAX_REGISTERS", CP_FLAG_UINT, F(max_registers),
     "force a register cap on every shader; 0 leaves it to the driver" },
   { "CUDAPIPE_LAUNCH_BOUNDS", CP_FLAG_UINT, F(launch_bounds),
     "emit maxntidx metadata with this block size; 0 emits none" },
   { "CUDAPIPE_TUNE_VETO", CP_FLAG_FLOAT, F(tune_veto),
     "how much worse the capped build may be before it is refused",
     .fdflt = CP_TUNE_VETO },
   { "CUDAPIPE_SHADER_STATS", CP_FLAG_BOOL_PRESENCE, F(shader_stats),
     "report register counts and occupancy-trial outcomes" },
   { "CUDAPIPE_DUMP_NIR", CP_FLAG_BOOL_PRESENCE, F(dump_nir),
     "print each shader's NIR" },
   { "CUDAPIPE_DUMP_IR", CP_FLAG_BOOL_PRESENCE, F(dump_ir),
     "print each shader's LLVM IR" },
   { "CUDAPIPE_DUMP_PTX", CP_FLAG_BOOL_PRESENCE, F(dump_ptx),
     "print each shader's generated PTX" },

   /* ---- rasterizer tuning (NVRTC -D options) ---- */
   { "CUDAPIPE_SMALL_THRESHOLD", CP_FLAG_OPT_INT, F(small_threshold),
     "triangle area below which the small-primitive path is used",
     .empty_is_unset = true },
   { "CUDAPIPE_MEDIUM_THRESHOLD", CP_FLAG_OPT_INT, F(medium_threshold),
     "triangle area below which the medium-primitive path is used",
     .empty_is_unset = true },
   { "CUDAPIPE_POINT_THRESHOLD", CP_FLAG_OPT_INT, F(point_threshold),
     "triangle area below which a primitive is rasterized as a point",
     .empty_is_unset = true },
   { "CUDAPIPE_TILE_BOUND", CP_FLAG_OPT_INT, F(tile_bound),
     "0 walks the whole tile; 1 walks only the primitive's bounding box",
     .empty_is_unset = true },
};

#undef F

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
             * is new — CUDAPIPE_SMALL_ALLOC=advize silently disabled the arena
             * and would have been measured as "advise made no difference". */
            fprintf(stderr, "cudapipe: %s='%s' is not a known value — "
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
   if (!flag_was_set("CUDAPIPE_ABUFFER_COMPOSITE"))
      debug_state.abuffer_composite = !debug_state.abuffer_verify;
   if (debug_state.abuffer_composite)
      debug_state.abuffer_verify = false;
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
      "cudapipe: %zu environment switches. Value shown is what this process "
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

   if (getenv("CUDAPIPE_HELP"))
      cp_debug_help();
}
