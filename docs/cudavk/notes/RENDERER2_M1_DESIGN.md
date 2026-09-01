# Renderer 2 M1: segmentation, scope state table, FS bucketing, walk budget

Branch `redesign/tilewalk`, worktree `mesa-renderer2`, HEAD at writing
`f045c753743`. Status: **design only**. No kernel is written in this milestone
and no driver source is changed by it. Every structural claim below carries a
`file:line` citation into `src/cudavk/` at this HEAD; every number carries a
document or trace citation. Where a number does not exist, the sentence says
**ASSUMPTION** and names the measurement that would settle it.

Note on the M0 window: both captures contain a heavy segment where frame time
roughly doubles (`RENDERER2_M0_RESULTS.md:15-19`), and the M0 distributions are
whole-window while the trace used in section 0 is heavy-band. Where the two are
combined below — the per-scope figures of sections 4.6 and 5.1 — the result is
an **upper** figure, and it is labelled as such.

Inputs: `RENDERER2_DESIGN.md` (charter), `RENDERER2_M0_RESULTS.md` (census),
`RENDERER2_STATE_SURVEY.md` (state inventory), `DEAD_ENDS.md` entries 2, 10,
15, 22, 24, 30, 33. Three read-only probes taken for this milestone are kept at
`/tmp/renderer2-m1/{abi,seg,perf}_report.md`.

---

## 0. Correction to the numbers this milestone was briefed with

The four per-stage exclusive figures in circulation — raster 0.82, FS 0.78,
clip 0.73, VS 0.67 ms/frame — **appear in no document in this tree.** The only
near matches are unrelated (`DEAD_ENDS.md:481` is a 0.82 µs empty-block cost;
`DEAD_ENDS.md:2019` 0.821026 is the vertex-stage *pool* union;
`RESIDUAL_AUDITS_2026-08-31.md:62` 0.824 is the VS+clip *pair pool*). They are
not used here.

The real partition was recomputed read-only from the post-pin heavy-band trace
`/tmp/postpin-rtx2/trace.sqlite` — the log the charter cites at
`RENDERER2_DESIGN.md:14-16` — whose kernel+memcpy+memset union over its 1,013
real frames is **5.210771 ms/frame**, reproducing the charter's 5.21
(`RENDERER2_DESIGN.md:20`) to four digits. "Union-exclusive" is used in the
sense fixed by `/tmp/residual-gpu-audit/AUDIT.md:15-19` and governed by the
0.500 ms/frame gate at `RESIDUAL_AUDITS_2026-08-31.md:6`.

| pool, RTX heavy band | summed ms/f | **union-exclusive ms/f** | launches/f | self-overlap |
|---|---:|---:|---:|---:|
| generated VS (`main` → clip/raster) | 1.4997 | 0.6892 | 110.3 | 1.45x |
| generated FS (`main` → writeback/composite) | 0.6776 | 0.6776 | 67.7 | 1.00x |
| `cp_clip_rast_fused` (clip + stage 1) | 1.9365 | 0.5919 | 61.5 | 2.34x |
| `cp_rasterize_stage3` | 0.7726 | 0.4242 | 66.4 | 1.29x |
| `cp_fs_compact` | 0.3724 | 0.3724 | 66.4 | — |
| `cp_rasterize_stage2` | 0.3734 | 0.2138 | 66.4 | 1.13x |
| A-buffer support (`cp_abuf_*`) | 0.1944 | 0.1944 | 67.0 | — |
| `cp_fs_writeback` | 0.1098 | 0.1098 | 66.4 | — |
| `cp_rasterize_stage3_abuf` | 0.3148 | 0.0926 | 39.5 | 1.41x |
| `cp_rasterize_stage1` | 0.0289 | 0.0289 | 5.0 | — |
| `cp_rasterize_stage2_abuf` | 0.1493 | 0.0262 | 39.5 | — |
| memcpy / memset / other / cross-family overlap | — | 0.490 / 0.326 / 0.143 / 0.369 | — | — |
| **whole window** | **7.9907** | **5.2108** | **727.9** | 1.53x |

Two facts from that table govern everything below. **`cp_fs_compact` and
`cp_fs_writeback` together (0.482 exclusive) are larger than either A-buffer
raster stage**, and **`cp_clip_rast_fused` is 2.34x self-overlapped**, so its
time is the least collectible in the frame per the merge rule
(`DEAD_ENDS.md:1160-1188`, entry 22).

---

## 1. Segmentation: cutting a scope into maximal expressible runs

### 1.1 Why this is the entry ticket

`RENDERER2_M0_RESULTS.md:30-37`: opaque-only drawing scopes are **26 on
favorite3 and 0 on favorite2** out of 4,199 / 4,190 (favorite2's count corrected
to its own frame window at `ca06e42d674`; no percentile moved). A path that admits whole
opaque scopes admits nothing. Segmentation is M2's first line of code, not M3's.

### 1.2 The run predicate, and the result that makes it cheap

**A run is exactly what the existing renderer already calls an episode.** That
is not a convenience; it is the whole correctness argument, because the driver
already proves those boundaries sound in production.

A *run* is a maximal sequence of consecutive draws inside one render scope for
which all of the following hold.

1. **Same scope serial** — `cpvk_cmd.c:2677`
   (`scope->serial != dev->prev_scope->serial` breaks the batch).
2. **No non-draw op in between.** Every `cpvk_op_kind`
   (`cpvk_private.h:555-575`) except `CPVK_OP_DRAW`, `BEGIN_RENDER` and
   `END_RENDER` cuts through `cpvk_execute_order_point()`
   (`cpvk_cmd.c:4252-4257`) or a direct `cp_batch_flush()`: clear
   `cpvk_cmd.c:2244`, copy `cpvk_cmd.c:4312`, fill
   `cpvk_device_memory.c:309`, dispatch `cpvk_cmd.c:1307`, query
   `cpvk_cmd.c:4834`, query-copy `cpvk_cmd.c:4791`, barrier/event
   `cpvk_cmd.c:4284`. An indirect draw is also an order point
   (`cpvk_cmd.c:2895-2899`) and carries a host wait.
3. **Run-constant state** — the existing predicate
   `cpvk_draws_episode_compatible()` (`cpvk_cmd.c:2630-2646`), used verbatim:
   both pipelines non-null, same scope serial, identical viewport, identical
   raster state, identical depth state, identical blend state, identical
   sample count.
4. **A class predicate**, one of the two below.

**Opaque run.** Every draw satisfies `cp_batch_order_free()`
(`cp_renderer.c:7285-7306`) — quoted in full because it is the walk's licence:

```c
   if (state->blend.enable || state->fs->uses_discard)
      return false;
   if (cp_debug->unsafe_force_opaque)
      return true;
   if (!state->depth.depth_enabled || !state->depth.depth_writemask)
      return false;
   switch (state->depth.depth_func) {
   case CP_FUNC_LESS: case CP_FUNC_LEQUAL:
   case CP_FUNC_GREATER: case CP_FUNC_GEQUAL:
      return true;
   default: return false;
   }
```

so: no blend, no discard, depth test **on**, depth write **on**, and the depth
function one of the four monotone ones. `ALWAYS` is rejected even though it is
order-free in colour, because it is not order-free in depth. Stencil and
alpha-to-coverage are **not tested here at all** — an ASSUMPTION inherited from
the existing opaque episode, which relies on them being absent from these
captures; the measurement that settles it is a host-side census of pipelines
with stencil enabled inside drawing scopes, and it costs no GPU time. Plus the
framebuffer conditions of `cp_opaque_appendable()` (`cp_renderer.c:7547-7581`):
`fs` non-null, `!fs->writes_memory`, `attachment_samples == 1`,
`fb->nr_cbufs == 1`, `fb->color` non-null, `fb->color_encoding >= 0`.

**Blended run.** Every draw satisfies the per-draw A-buffer admission the
existing path enforces inside `cp_draw_execute_batch`: `!fs->writes_memory`
(`cp_renderer.c:7418`), `fb_samples == 1` (`:6346`), **no depth write**
(`:6350-6364` — "A draw that writes depth changes the depth-test outcome
between peel passes"), `nr_cbufs == 1` (`:6365`), `color_encoding >= 0`
(`:6370`).

**Everything else falls through** to the existing renderer, draw by draw,
unchanged. This is the charter's rule at `RENDERER2_DESIGN.md:49-51`.

Two caps that bound *today's* episodes do **not** bound a run:
`CP_PASS_MAX_SEGS = 64` (`cp_rast_types.h:321`) and `CP_PRIM_ID_LIMIT = 1<<30`
(`cp_rast_types.h:1358`) are the segment-snapshot table and the A-buffer
primitive field. A run's own caps are its reference array and its setup array
(§4.4). Removing the 64-segment cap is a coverage gain renderer 2 gets for
free, because `struct cp_pass_seg` — a whole `cp_rasterize_args` plus a
complete `cp_draw_batch` copy per segment (`cp_renderer.h:255-268`) — is
exactly the per-batch state the run table replaces.

### 1.3 Blended draws inside a mixed scope

They get **their own blended run**, not a fallback. Two facts make this legal
with no new mechanism:

- A mixed scope can already contain several A-buffer episodes in sequence. The
  A-buffer is resolved into the colour attachment at every episode end by
  `abuf_composite` (`cp_renderer.c:9281-9283`, `.color_out = fb->color` at
  `:9262-9277`), and the per-pixel counts are cleared at the next episode's
  start (`cp_renderer.c:9400-9413`). Primitive ids restart per episode
  (`cp_renderer.c:8842`, `:7871`) and "cannot be compared across one"
  (`cp_renderer.c:8046-8047`).
- The A-buffer's ordering key is `abuf_prim_base + tri_id`
  (`cp_rasterize.cu:719-739`, documented `cp_rast_types.h:249-256`) — an
  episode-global primitive index, ascending equals submission order, with no
  depth in it. Merging blended batches is legal only because the clipper lays a
  batch's triangles out in submission order (`cp_renderer.c:7308-7321`, stable
  mode `cp_rast_types.h:366-384`).

So a blended run is a run of the same kind, with the same key, and the tile
walk's job in it is to build the per-pixel run **already sorted** instead of
building it unsorted and paying `cp_abuf_scan_*` + `cp_abuf_sort*` +
`cp_abuf_seg_scatter` to sort it (`cp_renderer.c:8945`, `:8958`). Shading and
compositing are unchanged.

A blended draw the A-buffer refuses (depth write, MSAA, >1 attachment, FS
memory writes) ends the run and falls through, exactly as
`cp->pass.append_failed` does today (`cp_renderer.c:6423-6432`).

### 1.4 The handoff at a cut, and why the result is bit-identical

**The state carried across every cut is two device buffers and nothing else:**
`cp->depthbuf` and `fb->color`.

*Opaque run exit.* The walk writes what `cp_opaque_finish()` writes: the
winning fragment's colour into `fb->color` and its depth into `cp->depthbuf`.
Depth is committed in the writeback, not in the rasterizer
(`cp_fs.cu:1144-1160`, `args.depth_write && args.depthbuf`), and that writeback
is reached from `cp_shade_fragments()` at `cp_renderer.c:8008-8021`.

*Opaque run entry.* `cp->depthbuf` as it stands at the cut. This is already the
existing behaviour: **every segment of an opaque episode depth-tests against
the depth buffer as it stood at episode entry**, because depth commits only at
the shade, which runs once at `cp_opaque_finish()`. That is the real reason
`cp_batch_order_free()` exists.

*The bit-identity argument, in one paragraph.* The opaque resolve is

```c
   uint32_t key = args->depth_key_invert ? ~depth_uint : depth_uint;
   atomicMin(&visbuf[at], PACK_VISBUF(key, args->abuf_prim_base + tri_id));
```

(`cp_rasterize.cu:762-764`) with
`PACK_VISBUF(d,t) = ((uint64)d << 32) | (uint64)~(uint32)t`
(`cp_rasterize.cu:68`). The winner of a pixel is therefore the argmin over the
reference *set* of `(key, ~prim)` — a pure function of the set, independent of
the order in which references are applied, with ties on equal depth resolved to
the **largest** primitive id. Three consequences, all load-bearing:

1. **The walk may reorder references freely**, including into depth slabs
   (§4.3), and still produce the identical winner.
2. **The walk may update its tile-local depth as it goes**, even though the
   existing episode tests against frozen entry depth. Doing so rejects some
   fragments the existing path admits into the visbuf — but every such fragment
   would have lost the `atomicMin` anyway, because it was rejected by a
   *nearer* fragment already in the tile. The two paths differ only in
   *losers*, and only the winner is shaded and written.
3. **The early-out is therefore free of correctness risk**, which is the
   difference between this design and entry 30's stale-depth Hi-Z
   (`DEAD_ENDS.md:1552`), whose depth was cross-episode and had to be
   conservative.

*Blended run exit/entry.* Exit: `fb->color` composited by the existing
`abuf_composite`; `cp->depthbuf` unchanged, because a blended run may not write
depth (`cp_renderer.c:6350-6364`). Entry: `cp->depthbuf` for the test only.

*Scope entry and exit are unchanged.* `cp_render_scope_begin()`
(`cp_renderer.c:9674-9694`) flushes, sets the framebuffer, and runs the depth
attachment LOAD (`cp_depth_attachment_xfer()`, `cp_renderer.c:2661-2695`, call
site `:9691`); `cp_render_scope_end()` (`:9696-9713`) flushes and runs the
STORE (`:9711`). Renderer 2 touches neither.

### 1.5 Expected run population

`RENDERER2_M0_RESULTS.md:41-46`, favorite3 per drawing scope: draws p50 28 /
p90 269 / p99 326 / max 349; batches p50 16 / p90 56 / p99 102 / max 132.
Runs are coarser than batches, because only four state changes break a batch
without ending the episode: a draw whose key broke the batch, and the VS, FS
and vertex-element binds (`cp_renderer.c:7391-7410`).

The heavy-band trace gives 4.3 `cp_abuf_composite` launches per frame — one per
blended episode — against 2.02 drawing scopes per frame
(`RENDERER2_M0_RESULTS.md:27`), i.e. about **2.1 blended runs per drawing
scope**, with opaque runs interleaved between them.

**ASSUMPTION: 4-8 runs per drawing scope, 8-16 runs per frame.** The
measurement that settles it is a host-side run census on this branch — walk
each scope's recorded draw list, apply the §1.2 predicate, and histogram runs
per scope and tile references per run. It needs no GPU work and no kernel, and
it must be run before M2 (§5.4).

**This corrects the charter.** `RENDERER2_M0_RESULTS.md:78-82` says "single-digit
launches per frame". At 8-16 runs per frame and 2 + *k* launches per run (bin,
walk, and one shade launch per FS bucket, §3), the honest figure is
**40-120 launches per frame against today's 727.9** — a 6-18x reduction, not a
100x one. Launch arithmetic is reported, not banked (`RENDERER2_DESIGN.md:41`,
`DEAD_ENDS.md:1160-1188`).

---

## 2. The scope state table

### 2.1 The finding that makes this small

Survey problem 2 asks how a scope-global draw id is threaded through the
generated-shader ABI at `cp_rast_types.h:395-423`. **It already is.** The rule
is documented there:

```
 *   row  = ((const uint32_t *)args[CP_ARG_SLOT_BATCH_ROWS])
 *             [thread_id & *(uint32_t *)args[CP_ARG_SLOT_BATCH_MASK]]
 *   base = ((void **)args[CP_ARG_SLOT_UBO_TABLE])[row * CP_ARG_UBO_STRIDE + i]
```

and the generated code implements it as a **runtime `i32` load**, never a
constant: `emit_batch_row()` at `cp_nir_to_llvm.c:454-494` builds
`rows[tid & mask]` with `LLVMBuildLoad2(..., i32, ...)` at `:489`;
`emit_const_buf_base()` at `:496-523` multiplies by the baked constant
`CP_ARG_UBO_STRIDE` (`:517-518`). The mask is an AND-mask
(`LLVMBuildAnd`, `:488`) that the host writes as only `0` or `0xFFFFFFFF`
(`cp_renderer.c:5860`, `:3762-3763`) — the row *count* never appears in it.

The row is `uint32_t` end to end, so the arithmetic ceiling is `2^32/16 =
268,435,456` rows against an M0 maximum of **349 draws per scope**. There is no
width to widen.

Better: **the scope-wide row already exists in production.**
`cp_seg_range::row_base` (`cp_rast_types.h:898-907`, field at `:905`) and the
staging buffer `cp->pass_group_ubos` (`cp_renderer.h:283-285`) concatenate FS
UBO rows across segments today for merged blended shading groups. The host
computes `row_base` as a running counter and copies each segment's block at
that offset (`cp_renderer.c:8698-8719`); the device resolves the range by
binary search (`cp_fs_interp.h:266-295`, `args->row_base = range->row_base` at
`:293`); the interpolator adds it (`cp_fs.cu:143`, `cp_fs.cu:253`,
`cp_fs_inline.c:179`); **the shader adds nothing.** The comment states the
intent at `cp_renderer.c:9148-9153`: "The interpolator writes group-global rows,
so the shader indexes the concatenated table exactly as it indexes a single
batch's."

`cp->pass_group_ubos` is sized `CP_PASS_MAX_SEGS * CP_MAX_BATCH_DRAWS *
CP_ARG_UBO_STRIDE` = **8,192 rows** (`cp_renderer.c:7936-7939`, `:8688-8691`,
`:9159-9162`), which covers the M0 maximum of 349 rows with 23x margin. No new
allocation is required.

### 2.2 The row layout

Because the run predicate is `cpvk_draws_episode_compatible()`
(`cpvk_cmd.c:2630-2646`), viewport, raster state, depth state, blend state and
sample count are **run-constant by construction** and belong in the run header,
not in a row. This is the single biggest simplification the choice of predicate
buys, and it is why the row is 32 bytes.

```c
/* One per draw in the run. Uploaded once per run; never per batch. */
struct cp_run_draw_row {           /* 32 B */
   uint32_t fs_row;                /* the u32 written into CP_ARG_SLOT_BATCH_ROWS;
                                    * index into the run's concatenated FS UBO
                                    * table (cp_renderer.h:283-285) */
   uint32_t shade_key;             /* index into the run's kernel table, sec 3 */
   uint32_t prim_base;             /* first run-global primitive id of this draw */
   uint32_t prim_count;            /* == rast_num_triangles for this draw */
   int32_t  clip_rect[4];          /* scissor; cp_rast_types.h:154-166 */
};

/* One per run. */
struct cp_run_header {
   /* run-constant, by the sec 1.2 predicate */
   float    vp_x, vp_y, vp_w, vp_h, depth_scale, depth_translate;
   uint32_t cull_mode, front_face;         /* cp_rast_types.h:167-169 */
   uint32_t depth_test, depth_func, depth_write, depth_key_invert;
                                            /* cp_rast_types.h:173-176 */
   uint32_t blended;                        /* run class */
   uint64_t color, depthbuf;                /* scope-constant, cpvk_cmd.c:2353-2362 */
   uint32_t width, height, ndraws;
   uint64_t rows;                           /* struct cp_run_draw_row[ndraws] */
   uint64_t fs_ubo_table;                   /* uint64[ndraws * CP_ARG_UBO_STRIDE] */
};
```

### 2.3 Which existing slots change meaning

| slot / field | file:line | change |
|---|---|---|
| `CP_ARG_SLOT_BATCH_ROWS` | `cp_rast_types.h:422` | the `uint32` per shaded slot now holds a **run-global** row. **No ABI change** — already `uint32`, and `row_base` already writes group-global values (`cp_fs.cu:143`) |
| `CP_ARG_SLOT_UBO_TABLE` | `cp_rast_types.h:421` | points at the run's concatenated table instead of one batch's. Already variable per launch (`cp_renderer.c:3567-3569`); staging buffer already 8,192 rows |
| `CP_ARG_SLOT_BATCH_MASK` | `cp_rast_types.h:423` | unchanged; stays `0xFFFFFFFF` |
| `args[7]` draw-parameter rows | `cp_rast_types.h:502-507` | fragment stage does not read them; unchanged |
| `cp_rasterize_args.clip_rects`, `draw_slices` | `cp_rast_types.h:154-166`, `:481-496` | not read by the walk; they stay geometry-pass structures |
| `struct cp_pass_seg` | `cp_renderer.h:255-268` | **replaced** for a run: today's "scope table" is 64 whole-batch snapshots; it becomes `ndraws` x 32 B |

### 2.4 Does the generated shader ABI change? No.

Evidence, in order of strength:

1. The row is a runtime `i32` load, never a constant
   (`cp_nir_to_llvm.c:487-490`), so its full 32-bit range is already legal.
2. The mask is a runtime load too (`cp_nir_to_llvm.c:473-475`) and never
   encodes a count.
3. `cp_rast_types.h:405-408` states the design intent: "All three slots are
   always filled, so the generated code has no branch and no batched/unbatched
   variant."
4. The `row_base` mechanism already changed the row's meaning from batch-local
   to group-global **with no compiler change at all** — `row_base` appears
   nowhere in `cp_nir_to_llvm.c`.

**Compatibility with the existing renderer is therefore automatic.** The same
compiled module serves both paths, because neither the module nor the shape of
its argument block changes; only the *values* in the row array and the *length*
of the UBO table differ, and both are already per-launch quantities.

Three constants stay baked and must not be touched: `CP_ARG_UBO_STRIDE` at
`cp_nir_to_llvm.c:518`, `CP_ARG_DRAW_PARAM_STRIDE` at `:849`, and the block
size 256 in the thread-id reconstruction at `:481`.

**The one place a recompile would be needed is deliberately avoided.** The
*vertex* row is computed inside the fused shader by the inlined
`cp_vf_lane_ids()` (`cp_nir_to_llvm.c:466-471`, `cp_vs_inline.c:94`,
`cp_vf_lane.h:150-161`) and has no `row_base`; making it scope-global would
change `cp_vf_lane.h`, force a bitcode rebuild and break the offset assertions
at `cp_vs_inline.c:37-40`. Renderer 2 does not do this: the **geometry pass
stays per batch and unchanged**, as the charter requires
(`RENDERER2_DESIGN.md:55-57`).

What that costs, stated plainly: the VS pool (0.6892 ms/frame exclusive, 110.3
launches/frame) and `cp_vertex_fetch` (0.1262) are **not** collected by
renderer 2. Neither is `cp_clip_rast_fused`'s clip half.

### 2.5 Row build cost against the M0 draw counts

Per scope, at favorite3 percentiles (`RENDERER2_M0_RESULTS.md:41-46`):

| | p50 (28 draws) | p99 (326) | max (349) |
|---|---:|---:|---:|
| `cp_run_draw_row[]` | 896 B | 10.4 KB | 11.2 KB |
| FS UBO table (16 x u64 per row) | 3.6 KB | 41.7 KB | 44.7 KB |
| **total upload per scope** | **4.5 KB** | **52.1 KB** | **55.9 KB** |

Against what it replaces: the existing path owes about **11 KiB of argument and
row upload per segment** (`cp_renderer.c:7618-7627`, survey item 16) and a scope
carries p99 102 batches — roughly **1.1 MB per scope today**. The run table is
about **20x smaller**, and the per-row memcpy is the same one already performed
at `cp_renderer.c:8715-8717`. Row build is host work of order
`O(draws in run)`, bounded at 349 rows, and is not a new cost — it is the
existing cost done once per run instead of once per batch.

---

## 3. FS bucketing

### 3.1 The problem, restated with its real size

The launched CUfunction is a function of four axes (survey problem 1):
the FS binary; the execution mode (`cp_nir_to_llvm.h:70-83`, chosen at
`cp_renderer.c:4120-4147`); the sampler variant, which the host resolves by
**CPU-dereferencing every per-draw FS UBO row** through `cp_host_ptr` and the
`host_maps[]` published at submit (`cp_renderer.c:3577-3644`, requiring every
row and descriptor to agree at `:3588-3617`, JIT-compiling a new variant at
`:3627-3635`); and a pending register-cap tune alternate applied at the launch
itself (`cp_renderer.c:3851-3866`).

M0 bounds the first axis: **at most 21 distinct FS binaries in the worst scope**
(`RENDERER2_M0_RESULTS.md:48-52`). ASSUMPTION: the four-axis product stays
under 64 per run; the measurement that settles it is a host-side census that
runs the existing `cp_fs_launch_shader()` resolution for every draw of every
scope and counts distinct `(fs, exec, variant, alt)` tuples. No GPU work.

### 3.2 Chosen mechanism: visibility first, shading second, bucketed by kernel

**The walk kernel never executes shader code.** It evaluates edge functions and
depth only, and resolves each tile's 256 pixels into a tile-local visibility
buffer holding, per pixel, `(depth_key, run-global draw row, primitive id)`.
Shading is a second phase: for each distinct kernel identity present in the run,
one launch over the compacted list of pixels whose winner belongs to that
identity.

Per-pixel draw order is preserved, and the argument is different for each class:

- **Opaque run.** Visibility has already chosen *one* winner per pixel, and that
  winner is order-independent (§1.4). There is no per-pixel order left to
  preserve at shading time. Bucketing by kernel is therefore free.
- **Blended run.** Order is the whole semantics. The walk emits each pixel's
  fragment list **already sorted ascending on `abuf_prim_base + tri_id`** — the
  same key the existing sort produces (`cp_renderer.c:8945`, `:8958`) — and
  hands it to the **existing** bucket-by-segment, shade-densely, composite
  structure (`cp_renderer.c:9155-9291`). Only the *build* is replaced; the order
  contract is untouched.

**This is not a new mechanism — it is what the driver already does.**
`cp_opaque_finish()` groups segments by shading identity (vs, fs,
`num_fs_ubos`, mode, ndraws, `fs_ubos` bytes) at `cp_renderer.c:7913-7934` and
issues one `cp_shade_fragments()` per group over the shared visbuf
(`:8008-8021`). Renderer 2 changes the group key from that six-field tuple to
the four-axis kernel identity, and changes the visbuf from per-episode-global to
per-tile-then-scattered. Everything downstream — argument block, row table,
sampler-variant resolution, hardware-texture pinning — runs unchanged.

Consequences worth stating:

- The sampler variant and tune alternate must be **frozen at table-build time**,
  before the walk launches, and the resolution is host CPU work that already
  happens per launch today. Freezing it earlier is strictly less work, not more.
- A draw whose rows do not resolve to one identical `cp_sampler_info`
  (`cp_renderer.c:3588-3617`) gets the generic kernel, as today, and becomes its
  own bucket. That is a per-row performance cliff the A/B must account for
  (survey problem 1) — it does not affect correctness.
- Module globals `cp_sampler_table` and `cp_quad_derivs`
  (`cp_renderer.c:3805-3839`) are written per module when stale; a run using *k*
  modules pays *k* of them once, instead of once per batch.

### 3.3 Rejected alternative 1: per-draw indirect dispatch inside the walk

Have the walk call the right shader body per reference, through an indirect call
or a switch over the run's binaries.

**Rejected, with a measured precedent.** `DEAD_ENDS.md` entry 4
(`:226`, "Out-of-line sampler dispatch and per-key fragment execution —
REFUTED") already closed per-key fragment execution. The rules distilled from
the campaign say why it would fail again: rule 10 (`DEAD_ENDS.md:2108-2112`) —
"Instruction and code footprint beat occupancy in the sampler ... iteration
22's noinline form reached 143-182 registers and a 225 KiB `main` SASS and
regressed every arm." A switch over up to 21 inlined shader bodies gives one
kernel whose register allocation is set by the worst body, in a kernel that also
has to hold the tile's visibility state. It also destroys the property that
makes the walk cheap: the walk's inner loop is 40-ish instructions of edge and
depth arithmetic (§4.2), and inlining shader bodies into it makes the *rejected*
references pay shader register pressure.

### 3.4 Rejected alternative 2: one specialised walk per FS identity over its tiles

The charter's third option (`RENDERER2_DESIGN.md:78-80`): compile one walk
kernel per pipeline and run each over the tiles its draws touch.

**Rejected on two counts.**

1. **It re-serialises what the single walk exists to unify.** A tile whose list
   spans *k* identities would be walked *k* times, and each pass would need the
   depth results of the other passes to be correct, because occlusion crosses
   draws. That is a cross-pass dependency inside one run — exactly the shape the
   single walk removes, and exactly the shape `DEAD_ENDS.md` entry 2
   (`:154`, persistent/dynamic-claim schedulers, **+140.15 ms**) and entry 25
   (device-side episode chaining) closed.
2. **It multiplies launches by up to 21** and, by entry 24's one-block-duration
   finding (`DEAD_ENDS.md:1260-1269`: `sm__cycles_active.max` 99.75% of elapsed,
   average SM 22%, minimum 0.10%), each of those launches is priced by its own
   longest block, not by its total work. Twenty-one tail blocks instead of one.

---

## 4. Walk kernel budget

### 4.1 The shape, and the one thing entry 24 got structurally wrong

Entry 24's prototype was **one block per tile, one reference at a time**, with
thread 0 running `setup_triangle()` and copying `cp_rasterize_args` into shared
memory behind a CTA barrier; 73.3% of stall cycles sat at that barrier
(`DEAD_ENDS.md:1260-1269`). Its measured price was **~3,409 cycles of block time
per reference** (`DEAD_ENDS.md:1278-1283`), and applied to the old capture's
10,033-reference longest lists it predicted +12.8 ms/frame against a recorded
+10.38 ms/frame regression.

Renderer 2 inverts the mapping: **one thread per reference, not one block per
reference.** This is the same shape as the shipping `cp_rasterize_stage1`,
which already runs one thread per triangle and rasterises sub-16 px triangles
inline (`cp_rasterize.cu:908-942`) — and M0 measures the mean triangle edge at
~1.7 px (`DEAD_ENDS.md:1271-1276`), so essentially every reference takes that
inline path. The change is the *destination*: 2 KB of shared memory instead of
the global visbuf, and setup produced by the bin pass instead of by thread 0
inside the barrier.

### 4.2 Block, grid, and per-block state

| | value | basis |
|---|---|---|
| tile | **16 x 16 px** | `RENDERER2_M0_RESULTS.md:129-137` — 32 px raises the p99 serial list 62% (f3) / 57% (f2) and cuts blocks from 3,600 to 900 |
| grid | **3,600 blocks, flat** (80 x 45) | `RENDERER2_M0_RESULTS.md:54`; empty tiles exit at once. Justified by `DEAD_ENDS.md:2090-2092` rule 6: "Idle blocks are free ... under 0.2 ns for an idle 256-thread block" — so no compaction pass |
| block | **256 threads (8 warps)** | one thread per reference in flight; matches the 256 baked into `cp_nir_to_llvm.c:481` for the shading phase |
| blocks/SM | 21.2 on 170 SMs, 24.3 on 148 | `DEAD_ENDS.md:497` (170-SM card); charter's 148-SM B200 |
| depth/coverage state | `uint64_t vis[256]` = **2,048 B shared**, packed `PACK_VISBUF` exactly as `cp_rasterize.cu:68` | keeps the resolve bit-identical by using the same packing and the same `atomicMin`, in shared memory |
| plus | `uint32_t tile_ceiling` (1 word), `uint32_t slab_cursor` | §4.3 |

Shared memory per block is 2 KB + a few words, so occupancy is bounded by
registers, not by shared memory.

**Per-pixel state is one `uint64` and there is no separate coverage mask.**
Saturation is expressed by the ceiling itself: while any pixel is still
`VISBUF_EMPTY` (`cp_rasterize.cu:72`) the ceiling is `0xFFFFFFFF` and no
early-out can fire; once every pixel has a winner the ceiling becomes finite and
falls monotonically. This is the correct representation because "full coverage"
and "shallow depth ceiling" are the same condition.

### 4.3 The early-out, and why it is legal

Two levels, both resting on §1.4's order-freedom.

1. **Per-reference rejection.** The bin pass stores, per reference, the
   conservative nearest depth key — `min` over the triangle's three
   `ndc_z` values (`cp_tri_setup.ndc_z0/1/2`, `cp_rast_types.h:1339`) put
   through the same `float_to_sortable_uint(CP_WINDOW_DEPTH(...))` the
   rasteriser uses (`cp_rasterize.cu:665-666`) and inverted when
   `depth_key_invert` (`cp_rasterize.cu:763`). A reference whose nearest key is
   greater than `tile_ceiling` cannot win any pixel of the tile and is skipped
   before any edge evaluation.
2. **Whole-walk termination.** The bin pass buckets each reference into one of
   **8 depth slabs** by the top 3 bits of that key. The walk processes slabs
   near to far and stops when the current slab's near bound exceeds
   `tile_ceiling`. **Reordering is legal because the resolve is order-free**
   (§1.4) — this is the property entry 24's prototype did not have and entry
   30's stale-depth Hi-Z could not use.

`tile_ceiling` is refreshed by a 256-lane shared-memory max reduction (8 steps)
every 256 references — about 8 cycles per 256 references, i.e. under 0.04
cycles per reference of overhead.

Honest bound on the prize: entry 30 measured that *episode-entry* depth rejects
6.739% of primitive/tile references and 16.562% of covered pixels
(`DEAD_ENDS.md:1586-1589`). The walk's depth is stronger — intra-run,
current-frame, and slab-ordered so the near geometry lands first — but entry 30
is the only measured anchor, and it is a **lower** bound on what a slab-ordered
walk rejects. **ASSUMPTION: 20-60% of references skipped in the hot tail tiles,
near 0% in the median tile (whose hottest list is 237 refs on f3, 24 on f2,
`RENDERER2_M0_RESULTS.md:62`).** The measurement that settles it is a host-side
replay of the recorded per-scope reference lists with the slab walk simulated in
software — no GPU, no kernel.

### 4.4 Memory traffic per reference

| item | bytes | note |
|---|---:|---|
| tile reference record | 8 | `{uint32 setup_index; uint16 draw_row; uint16 slab/flags}` — the shape `cp_opaque_tile_ref` already sketches (`cp_rast_types.h:1015-1019`) widened to carry the run-global row |
| triangle setup, amortised | 80 / 1.16 = **69** | `struct cp_tri_setup`, 80 B, `static_assert` at `cp_rast_types.h:1365`; 1.16 refs per triangle at 16 px (`RENDERER2_M0_RESULTS.md:60`) |
| **read per reference** | **77 B** | |
| written per reference | **0** | the resolve is shared-memory only |
| written per tile, once | 2,048 B | plus the compacted survivor list |

Per-frame totals at 412k references (`RENDERER2_M0_RESULTS.md:59`): **31.7 MB
read** by the walk; the bin pass reads 354k post-clip triangles x 48 B of
position (`cp_rast_types.h:354`) and writes 354k x 80 B of setup plus 412k x 8 B
of references, about **48 MB**. Roughly 80 MB/frame against an RTX 5090's
bandwidth — and `PERFORMANCE.md:670-694` records DRAM at **1%**. Bandwidth is
not the constraint here and is not claimed as one.

Capacity: 412k references/frame across ~8-16 runs. Size the reference array per
run from the run's post-clip triangle count with the same grow-only,
overflow-counter-and-fall-back discipline the A-buffer uses
(`cp_rasterize.cu:719-724`, drain check `cp_renderer.c:9023-9024`, fallback
`cp_renderer.c:9032-9040`). The existing sketch's `CP_MAX_OPAQUE_TILE_REFS = 2M`
(`cp_rast_types.h:1013`) is already 5x the whole-frame reference count.

### 4.5 The arithmetic

Serial ceiling per `RENDERER2_M0_RESULTS.md:63`: **refs_max p99 12,823 (f3) /
10,096 (f2), max 14,814 / 13,259.** With 256 threads that is
**12,823 / 256 = 50.1 references per thread** at p99.

Inner loop per reference per thread: load 77 B; three edge functions evaluated
at the pixel with the fill-rule bits already in `cp_tri_setup`
(`cp_rast_types.h:1341`); `ndc_z` interpolated; one `float_to_sortable_uint`;
one shared `atomicMin` per covered pixel. At a mean edge of ~1.7 px
(`DEAD_ENDS.md:1273`) a reference covers 1-3 pixels, so the bbox loop is 1-4
iterations. **Estimate ~40 instructions and 1-3 shared atomics per reference per
thread.**

Block time at p99, two ways:

- *Issue-bound.* 8 warps x 50.1 refs x 40 instructions = 16,032
  warp-instructions. At 2 issued per cycle (8 warps over 4 schedulers, thin
  latency hiding in the tail block when it runs alone) = **8,016 cycles ~ 3.3 µs
  at 2.4 GHz**.
- *Memory-bound.* 12,823 x 77 B = 0.99 MB into one SM. At ~96 B/cycle of L2
  read = **10,300 cycles ~ 4.3 µs**.

So the p99 tail block is **memory-bound at roughly 10k cycles, ~4 µs**, or about
**0.8 cycles of block time per reference**.

**The budget, stated as a falsifiable gate.** Allow the walk 0.15 ms per scope,
i.e. 0.30 ms/frame at 2.02 drawing scopes/frame. At 2.4 GHz that is 360,000
cycles for the longest block, against 12,823 references:

> **The walk must cost at most 28 cycles of block time per reference at the p99
> tile. The estimate is ~0.8. Entry 24's measured prototype was 3,409 — 122x
> over budget.**

The margin between 0.8 and 28 is a factor of 35, and it exists to absorb the
things this arithmetic does not model: shared-atomic contention on a hot pixel,
tail-block latency exposure, and the setup load's L2 miss rate. If a prototype
measures above 28, the design is refuted at that point and no more should be
built (§5.4).

### 4.6 Against what the existing chain spends

Per drawing scope, at 2.02 drawing scopes/frame
(`RENDERER2_M0_RESULTS.md:27`) — all derived, and the scope rate is a whole-run
average against a heavy-half trace, so the per-scope figures are **upper**:

| | ms/scope |
|---|---:|
| whole device union today | 2.580 |
| the pools the walk replaces (§5.1), exclusive | **0.724** |
| bin + walk budget | **0.15 (gate) / ~0.02 (estimate)** |

The walk is budgeted at **21% of the exclusive time it replaces** and estimated
at about 3%.

One caution that must travel with these numbers, from
`PERFORMANCE.md:670-694`: the machine is at 20% SMs Active and **3% SM Issue**.
The existing chain spends roughly 1,478 SM-cycles per reference of wall time
across 170 SMs — the same order as entry 24's 3,409 — which means **the existing
chain's time is not arithmetic**. It follows that the walk's own instruction
count is not what will decide its duration either, and that the 4.5.-style
arithmetic above is a *feasibility bound*, not a prediction. The prediction is
the 28-cycle gate, and it must be measured.

---

## 5. Go / no-go

### 5.1 The collectible, with the arithmetic

Displaceable union-exclusive pools in the RTX heavy band (§0 table), counting
only what a bin+walk actually removes:

| pool | exclusive ms/f | why it goes |
|---|---:|---|
| `cp_rasterize_stage3` | 0.4242 | replaced by the walk |
| `cp_rasterize_stage2` | 0.2138 | replaced by the walk |
| `cp_rasterize_stage1` | 0.0289 | replaced by the walk |
| `cp_rasterize_stage3_abuf` | 0.0926 | replaced by the blended walk |
| `cp_rasterize_stage2_abuf` | 0.0262 | replaced by the blended walk |
| A-buffer support `cp_abuf_*` | 0.1944 | the walk emits sorted lists; scan/sort/scatter go |
| `cp_fs_compact` | 0.3724 | the tile knows its survivors; compaction is free |
| `cp_fs_writeback` | 0.1098 | fused into the shade; legal here by the claimant argument (`RENDERER2_DESIGN.md:69` and the entry-15 row at `:44`, and entry 15's disproof at `DEAD_ENDS.md:814-821` does not apply because a tile belongs to one block) |
| **displaced total** | **1.4623** | |
| less bin + walk own cost | −0.15 to −0.45 | §4.5 gate; entry 24 priced binning at "about 0.13 ms" (`DEAD_ENDS.md:1285-1290`) |
| **net device collectible** | **1.01 to 1.31 ms/frame** | |

Explicitly **not** claimed: the VS pool (0.6892), `cp_vertex_fetch` (0.1262),
`cp_clip_rast_fused` (0.5919 — and it is 2.34x self-overlapped, the least
collectible pool in the frame under entry 22), the FS `main` pool (0.6776 — the
existing renderer already shades only visbuf winners, so the walk does not
reduce shaded pixels), memcpy, memset.

Reported and **not banked**, per `RENDERER2_DESIGN.md:41` and entry 22
(`DEAD_ENDS.md:1160-1188`): about **416 launches/frame removed**, worth
**0.325-0.337 ms/frame** of host issue at the measured removal price of
0.78-0.81 µs (`PERFORMANCE.md:242`, `:309`, `:315`).

Not modelled, and it is the largest single uncertainty: **device time is not
frame time** (`PERFORMANCE.md:438-441`). The heavy band has the host blocked
4.031 ms/frame and the device idle 3.195 ms/frame. The conversion slope is what
M4 measures, and nothing before M4 may quote a frame-time number.

### 5.2 Verdict

**GO.** The net device collectible of **1.01-1.31 ms/frame** on the RTX heavy
band is 2.0-2.6x the 0.500 ms/frame union-exclusive admission gate
(`RESIDUAL_AUDITS_2026-08-31.md:6`), and no single closed entry in
`DEAD_ENDS.md` covers it: entry 24 refused a 2.1-2.6 ms ceiling because it had
to surrender a +2.73 ms mechanism and had no occlusion; this design has the
occlusion (§4.3, legal by §1.4) and replaces the surrendered fan-out with 3,600
independent blocks rather than giving it up for nothing.

The go is conditional on the three pre-M2 measurements in §5.4, all of which are
host-side and cost no GPU work.

### 5.3 The smallest M2 slice

**The opaque prefix run, into `cp->depthbuf` and `fb->color`, with everything
else falling through.**

Concretely: for each drawing scope, take the maximal *leading* run of draws
satisfying §1.2's run predicate and the opaque class predicate. Bin it, walk it,
shade it per bucket, write colour and depth. Then hand control to the existing
renderer at the first non-conforming draw, with no extra state.

Why this one is first:

1. **The handoff is one buffer in and two buffers out** (§1.4), and both are
   already what `cp_opaque_finish()` produces. Nothing new is threaded anywhere.
2. **It needs no A-buffer work at all** — no scan, no sort, no composite, no
   overflow/fallback path (`cp_renderer.c:9032-9040`).
3. **It exercises segmentation from day one**, because every mixed scope ends
   the prefix at its first blended draw — which is finding 1 exactly
   (`RENDERER2_M0_RESULTS.md:33-37`).
4. **The flag can be turned on immediately.** `CUDAVK_RENDERER2` is a value bool
   in the `cp_debug.c` registry (never a `getenv`, per `CLAUDE.md`), the run
   predicate is a pure host-side function of state the front end already
   compares, and a scope with an empty opaque prefix simply falls through.

Gate is unchanged from the charter: bit-identical sentinels and stdout hashes on
both captures with the flag ON, 79/79 suite.

### 5.4 What would falsify this design early — in order, before any kernel

1. **The opaque prefix may be empty.** If drawing scopes typically begin with a
   blended draw, the M2 slice has no work and the milestone is void.
   **Test:** host-side run census on this branch — for each drawing scope, apply
   §1.2 and record runs, their draw counts and their share of the scope's tile
   references. **Falsified if the leading opaque run holds under 20% of a
   scope's references at p50.** No GPU, no kernel; the census plumbing already
   exists (`cp_renderer.h:287-318`, reducer `cp_renderer.c:8402`).
2. **The opaque-stream fan-out may still be worth more than the walk.** Entry 24
   died because the tiled arm surrendered a mechanism worth **+2.73 ms/frame on
   the old capture** (`PERFORMANCE.md:56-59`), and the asymmetry is driven by
   segments per episode — 12.42 on old against 1.71 on Crossroads
   (`PERFORMANCE.md:64-66`). **The value of that fan-out on favorite3 and
   favorite2 at this HEAD is unmeasured.** **Test:** an A/B of
   `CUDAVK_NO_OPAQUE_STREAMS=1` on both captures, alternating arms in one
   session per `CLAUDE.md`. **Falsified if it exceeds ~1.0 ms/frame**, because
   the net collectible would then be inside its own surrender.
3. **The walk may be too slow per reference.** **Test:** replay a recorded
   hot-tile reference list through a standalone microbenchmark — not the driver
   — and measure block-time cycles per reference at 12,823 references.
   **Falsified above 28 cycles/ref** (§4.5).
4. Secondary, cheap, and worth doing with (1): the four-axis kernel-identity
   census (§3.1) — falsified if a run commonly needs more than 64 buckets — and
   a stencil/alpha-to-coverage census (§1.2), which `cp_batch_order_free()` does
   not test.

### 5.5 One prerequisite this milestone found

`cp_tile_census_cut()` (`cp_renderer.c:8419-8431`) still has **no callers** on
this branch; the four cut counters printed at `cp_renderer.c:8391-8394` are
structurally zero, as `RENDERER2_M0_RESULTS.md:85-88` already warned for the
residency-cut numbers. The op table in §1.2 is exactly where they belong. Wire
them as part of falsification test 1 — the run census cannot report why a run
ended without them.

---

## Appendix: claims this design makes that are not measured

| claim | where | measurement that settles it |
|---|---|---|
| 4-8 runs per drawing scope | §1.5 | host-side run census |
| under 64 kernel identities per run | §3.1 | host-side four-axis census |
| 20-60% of references skipped in tail tiles | §4.3 | software replay of recorded reference lists with slab ordering |
| ~40 instructions per reference per thread | §4.5 | standalone walk microbenchmark |
| no stencil or alpha-to-coverage inside candidate runs | §1.2 | host-side pipeline census |
| the value of the opaque-stream fan-out on these captures | §5.4 | `CUDAVK_NO_OPAQUE_STREAMS=1` A/B, both captures |
| device-to-frame conversion slope | §5.1 | M4, and only M4 |
