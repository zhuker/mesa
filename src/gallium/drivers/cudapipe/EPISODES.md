# Pass episodes: 77 → 42.5 ms, and the road to 16

The tenth performance pass. Goal: 16 ms/frame median on the HeadlessStreamer
capture. This session took the median from **77.2 to 42.5 ms** (p95 95.1 →
56.1) in eight commits, `7683344facb..e5a45a5acf3`, every one gated. This
document is the record of what was built, the fresh profile of what remains,
and the ranked designs for the next session. Read
`HEADLESS_STREAMER_PERF.md` first for the capture, the correctness method and
the measurement discipline — none of that changed, and all of it still binds.

Working artefacts from this session live in `~/claude-scratchpad/perf16/`:
`BASELINE.md` (the running log with every measurement), `fps_plugin.so` +
`submits_*.txt` (per-frame timing), `cushim.so` (the API-count shim),
`trace_*.sqlite` (nsys traces at each stage), `dumps_*` (gated frame dumps),
`passseq2.log` + `passseq_stats.py` (the pass-structure statistics).

---

## Where the number stands, and how it was verified

| build | median ms | what changed |
|---|---|---|
| session start (`b3a10d11d16`) | 77.2 | — |
| per-draw fs UBO rows + draw params | 63.8 | `7683344facb` |
| per-draw VB bases + scissor rects | 62.5 | `a9b4e0b333f` |
| pass episodes | 56.7 | `4818654bec0` |
| grid-stride shaders + visbuf skip | 55.7 | `8155046b61f` |
| 8 side streams + stage-3 clamp | 42.8 → 42.3 | `238ce41f304` |
| drainless tiny draws | 42.5 | `e5a45a5acf3` |

Every commit passed: the 10-frame capture dump (`cp10.json` plan) holding
**exactly** 0.441% > 32/255 and 0.002% > 96/255 against release llvmpipe,
the four loading-screen frames bit-exact, and the 18-sample sweep
(`cp_iterate.sh`, labels `p1_fsdp2`, `p1_vbsc`, `p2_episode`, `p3_stride`,
`p3_visskip`, `p3_streams`, `p3_tune`, `p3_drainless` under
`~/git/Vulkan/build/iter/`) moving only within the documented
nondeterministic jitter. The sweep total also *improved*, 38.3 → 36.3 ms —
multisampling −44% and pbribl −40%, opaque batches now merging across
material rebinds.

Two verdict lines are standing failures older than this work and appear
identically in every stored iteration: `gltfscenerendering REGRESSED (budget
25547)` and `texture3d 2→3090 at frame 24` (gap 1 in the handoff). Do not
re-diagnose them as new.

---

## What the frame is made of now

Established with the same ladder as always — cushim for exact API counts
(**never wrap the replay in `/usr/bin/time` when preloading it: the wrapper
also loads the shim and its empty atexit report overwrites the real one**),
nsys for device time, CPU sampling for host attribution.

At 42.5 ms/frame:

- **Device union-busy is 32 ms** — the frame is device-throughput-bound.
  Host CPU is ~96% CUDA API machinery; lavapipe and gfxrecon are under 2%
  combined, so **removing the lavapipe layer is not a lever** and was ruled
  out by measurement, not taste.
- Device kernel time by owner (sums overlap across streams, union is less):
  compiled shaders `main` **15.0**, A-buffer raster count+fill
  (stage1/2/3_abuf) **10.9**, interpolators 5.2, vertex fetch+clip 3.4,
  `cp_abuf_sort` 1.9, opaque raster 3.4, memsets ~2.
- Waits: episode drains ~19/frame at ~0.97 ms each (the device executing,
  not the host asking), classic single-draw abuf drains ~20/frame (the bloom
  pyramid), `cp_flush` full-context syncs 31.7/frame totalling 5.3 ms.
- Launches 3,078/frame (was 7,686), stream syncs 47/frame (was 259),
  memsets 1,558/frame (was 3,727).

---

## What was built, and the invariants that keep it correct

### Per-draw tables (first two commits)

Every batch — opaque now included — carries per draw: fragment UBO rows
(through the existing `CP_ARG_SLOT_UBO_TABLE` indirection), draw parameters
(`args[7]` became one `CP_ARG_DRAW_PARAM_STRIDE` row per draw, indexed by
the batch row; `base_vertex` split from `first_vertex` and reads 0 for a
non-indexed draw), resolved vertex-buffer element bases
(`cp_vertex_fetch_args.elem_bases`), and scissor rectangles
(`cp_rasterize_args.clip_rects`, resolved per primitive in
`setup_triangle()` by the same slice search the fragment row uses). The
batch key keeps only counts and truly launch-wide state.

Traps preserved on purpose:

- **Stable clip is gated on consumption**: `batch_draws > 1 &&
  (blend_enabled || fs->reads_const_bufs)`. Forcing it on every batch cost
  multithreading 72% — stable mode rasterizes the whole 4× slot array,
  holes included, for ordering nothing consumes. Same for the fs table
  width: one row when the shader reads no constant buffer.
- **Deferral is the hazard, not merging** (BATCHING.md's law, met twice
  more): every gate that used to consult live binding state at flush time
  now reads the snapshot (`has_vs`, `vb_data2`).

### Pass episodes (`cp_pass_*`, the core of the session)

The capture's big passes bind 25-46 distinct shaders over 180-400 draws, so
consecutive-key merging saturates at ~3 draws. An episode strings
consecutive *blended* batches together: each flushed blended batch becomes a
**segment** whose vertex stage and A-buffer count rasterization run at
append time (inside `cp_draw_execute`, which returns after the count when
`cp->pass.appending` is set) into per-pixel lists shared by the whole
episode, fragments carrying an episode-global primitive base
(`abuf_prim_base`) so the one sort's ascending order **is** submission order
across segments. `cp_pass_finish()` then runs the scan, one fill relaunch
per segment from its saved `cp_rasterize_args`, the sort, the quad merge,
buckets quads by segment (`cp_abuf_seg_count/scatter`; the counts ride the
episode's one drain in the same `ab->counters` copy, extended by
`CP_PASS_MAX_SEGS` words), shades each segment densely over its own quads —
own shaders, strides, per-draw tables, via `cp_abuf_shade`'s `seg`
parameter — and composites once, resolving colours through
`quad_seg`/`quad_dense`/`cp_seg_desc`.

Load-bearing rules:

- **What may defer vs what must finish.** Exactly four sites call
  `cp_batch_flush_defer_why()` (key break in `cp_draw_vbo`, vs/fs/
  vertex-elements binds). Everything else goes through
  `cp_batch_flush_why()`, which finishes the episode — either because it
  observes rendering, or because it changes state the episode reads
  episode-wide (framebuffer, viewport, blend, depth, rasterizer). That is
  also why the episode needs no invariant key: it cannot outlive an
  invariant change. The CSO delete callbacks flush too — a deferred segment
  holds shader pointers.
- **An episode is one consecutive run of blended draws.** Reordering across
  an intervening opaque draw is unsound and there is a worked counterexample
  in the commit message of `4818654bec0` (B1, O, B2 with O between B1 and
  the prior opaque depth). Do not "optimize" the finish away at opaque
  boundaries.
- **Fallback = re-execute classically.** On overflow, allocation failure or
  a failed launch, `cp_pass_fallback()` re-runs every segment from its
  snapshot through the classic path, in order. This is sound because
  A-buffer count/fill rasterizations never write anything the renderer
  reads. The fallback (and the finish) restore per-segment live state
  around `cp_draw_execute` via `cp_pass_live_save/restore` — the segment
  snapshot is the only true copy of bindings that were rebound since.
- **Scratch epoch**: only a segment with `nsegs == 0` rewinds the arenas;
  every segment's clipped stream must survive until the episode's shading
  read it. The classic paths are untouched (`CUDAPIPE_NO_PASS_EPISODE=1`
  turns the whole thing off).

### Side streams (`8a29e8a7622`, `238ce41f304`)

Segment counts, fill relaunches and shades are mutually independent — the
per-pixel counts/cursors are order-free atomics whose order the sort erases,
and the dense shading slots are disjoint — so they fan out over
`CP_PASS_STREAMS` (8) side streams: episode-start clears + gate event on
main, join before the scan, broadcast before the fills, join before the
sort, broadcast after the scatter, join before the composite. **Each stream
owns its own rasterizer queue set** (`cp->seg_qsets`, routed through
`cp->cur_qset`) because the queues are rebuilt per launch group and two
streams may be in one concurrently. The upload arena's reuse discipline
already covered multiple streams — its wrap and the scratch reclaim drain
the whole context. Determinism holds because fill order within a pixel's
run was never meaningful (the sort keys on prim id) and dense shading order
only permutes slots the composite resolves through the per-quad map.

This was the single largest step: −20% alone. Going 4 → 8 streams was worth
another 3.7%, which says the per-stream serial chains still bind — see
"what's next".

### Two measured lessons (do not re-try these as wins)

- **Idle-block scheduling is not a cost.** Compiled shaders now grid-stride
  (a virtual `ctaid` phi in the `cp_nir_to_llvm.c` kernel skeleton;
  `emit_workgroup_id()` returns it inside the body; renders bit-identical,
  `oit` included) and the worst-case grids are capped — and the capture
  moved 56.7 → 56.3, noise. Kept for robustness. The device time is
  latency-bound *real* work in small serialized kernels; the cure is fewer,
  bigger, more concurrent kernels, not smaller grids.
- **Removing a wait relocates it unless the device gets emptier.** The
  drainless tiny-draw path (`e5a45a5acf3`: a ≤2-triangle draw provably
  cannot overflow anything, so the count drain has a known answer) was
  wall-neutral — the wait moved into the copies after each bloom pass.
  Kept because those serialization points return the moment the ones around
  them shrink. The same relocation happened when episodes removed the
  per-draw drains: `cuCtxSynchronize` in `cp_flush` went from 2.5 s to 11 s
  per run, catching a now-deep pipeline.

---

## What is left, ranked — with the designs

1. **Single-pass A-buffer build.** The count+fill double rasterization is
   10.9 ms of device sum and exists only to size the per-pixel CSR runs.
   Replace it: rasterize **once**, appending 64-bit `(pixel << 32 | prim)`
   records through one global atomic cursor; radix-sort the records
   (LSD, 8-bit digits, histogram+scatter, ~20 launches per episode — cheap
   at episode granularity); derive counts/offsets by boundary detection.
   The global sort also deletes `cp_abuf_sort`'s per-pixel bitonic
   (1.9 ms) — sorting on the full key is the per-pixel sort. Overflow is
   gated exactly as today (cursor vs capacity, the ok-word in the drain).
   Determinism improves if anything: the sorted array is a pure function of
   the fragment set. Frag memory doubles to 8 B/entry. Estimated −5-7 ms
   device sum; the quad merge and composite are unchanged consumers.
2. **Ping-pong arena reclaim in `cp_flush`.** 31.7 flushes/frame each pay a
   full `cuCtxSynchronize` to rewind the arenas. The managed scratch
   already has two slots — `cp->scratch.base[2]` with `current` never
   flipped, vestigial. Flip generations at flush, wait the generation's
   retire event from **two** flips ago (usually already complete), split the
   upload staging/arena in halves the same way, keep the periodic full sync
   as a backstop and the threshold reclaim unchanged. The device scratch's
   per-draw rewind needs nothing: it rides main-stream ordering. Note the
   retire event must be recorded after the episode joins — `cp_flush`
   already runs after `cp_batch_flush`, which guarantees it. Estimated
   −3-5 ms, plus pass-to-pass pipelining the drainless change was waiting
   for.
3. **Segment merge, then opaque episodes.** The capture alternates two
   vertex shaders draw by draw (143k of 193k vs-flushes), so episodes carry
   ~2× more segments than unique keys. Give the clipper per-draw output
   bases (a slice-table search by input triangle — the same search
   everything else uses) and a segment becomes a *group* owning
   non-contiguous global slot ranges; segments per episode halve, and the
   per-stream serial chains with them. The same machinery then gives
   **opaque** multi-shader episodes: raster per group into the shared
   visbuf (atomicMin is order-free — no slot ordering needed at all),
   shade per group with a prim-range filter. That attacks the ~79
   full-frame interpolate/fs/writeback trios (~7 ms device) the opaque path
   still pays per batch.
4. **Shading itself.** `main` is 15 ms/frame and becomes the floor once 1-3
   land. Unexplored: helper-lane fraction per quad on this capture (a
   census would say), fs_in/fs_out layout (strided float4 per varying —
   coalescing), whether the register-cap tuner's 5% threshold is right for
   the episode-era launch sizes, occupancy of the big-pass fragment
   shaders under `ncu` (the one place `NCU=1 cp_profile.sh` is now the
   right tool, since these kernels are finally large).
5. **Instanced draws** (91k/run ineligible, 60/frame solo executes) —
   extend `cp_draw_slice` with per-draw `verts_per_instance` (the pad word
   is free) and teach the fetch kernel's search to divide within a slice;
   removes the largest remaining eligibility hole. `depthfunc` (17k) and
   `topology` (12k) are the smaller two.
6. **The bloom pyramid's serial latency.** ~11 single-draw passes chained
   by sampling dependencies; episodes cannot merge them. What would help:
   cheaper per-pass fixed cost (1 above), the flush pipelining (2), and if
   still visible, fusing the blur chain's copies (`cp_resource_copy_region`
   syncs between passes — re-measure the async-copy negative result from
   `HEADLESS_STREAMER_PERF.md` under the new pipeline shape before
   believing it still holds).

A sober forecast from the current anatomy: items 1-3 land somewhere near
25-30 ms; 16 needs item 4 to take a real bite out of the 15 ms of `main`.
Every estimate above is against the *sum* view of an overlapped timeline —
re-measure union-busy after each step, and remember the two relocation
lessons before crediting a win.

## The measurement loop that worked

One cycle, in order, ~25 minutes end to end:

```sh
P=~/claude-scratchpad/perf16
# 1. build; smoke a few samples incl. oit (memory-writing shader)
# 2. capture correctness: ~17 s dump + compare (both from the same plan!)
bash ~/claude-scratchpad/abuf-resize/dump.sh $P/dumps_X <icd>
python3 ~/claude-scratchpad/abuf-resize/compare.py $P/dumps_X \
        ~/claude-scratchpad/abuf-resize/dumps_lvp     # MEAN must stay 0.441/0.002
# 3. capture timing: full replay + fps plugin, read the MEDIAN
# 4. sweep gate before any commit:
DESC="..." tests/cp_iterate.sh <label> <prev-label>
# 5. after the median moves: cushim (counts/waits), then nsys (device),
#    in that order — and nothing else on the GPU while any of it runs
```

Between cycles the histograms answer "why": `CUDAPIPE_DEBUG_BATCH` /
`CUDAPIPE_DEBUG_BATCHDIFF` piped through `sort | uniq -c` for batch breaks,
`CUDAPIPE_DEBUG_PASSSEQ` + `passseq_stats.py` for pass structure. Resolve
cushim call-site offsets with `addr2line -e <the .so> -f <offset>`.

---

# Session 2: 42.5 → 35.0 ms

Four commits, `53d221399d6..dcb29473a7f`, every one gated by the same
discipline (capture MEAN exactly 0.441%/0.002%, loading screens bit-exact,
18-sample sweep, only the two standing verdicts). Sweep labels
`p4_recfill`, `p4_ppflush`, `p4_segmerge`, `p5_inst` under
`~/git/Vulkan/build/iter/`. The session log with every measurement is
`~/claude-scratchpad/perf16/BASELINE.md` (Progress logs 7-10 and three
negative results).

| build | median ms | what changed |
|---|---|---|
| session start (`38b84429c80`) | 42.6 | — |
| single-pass A-buffer build | 40.9 | `53d221399d6` |
| drainless flush + honest fences + drainless barriers | 40.8 | `601df245e75` |
| merged shading groups | 40.6 | `7a495f72346` |
| instanced draws batch | **35.0** | `dcb29473a7f` |

p95 56.1 → 47.2. What each was:

1. **Single-pass A-buffer build** (item 1, staged): the count rasterization
   also appends `(pixel << 32 | prim)` u64 records through one
   warp-aggregated global cursor; the fill is `cp_abuf_fill_recs`, a linear
   replay with today's exact cursor/bounds/overflow semantics, so downstream
   is bit-identical and the episode's per-segment fill relaunches (and their
   broadcast/join) vanish. The radix half of the design was **evaluated and
   skipped**: at this capture's episode sizes (~12 episodes/frame), five
   8-bit passes × (histogram + scatter + scan ladder) is ~26 launches per
   episode against the ~150 µs the bitonic sort plus scan actually cost —
   net loss on launch latency alone. `CUDAPIPE_NO_ABUF_APPEND=1` restores
   the two-pass build.
2. **Drainless flush**: the arenas ping-pong through `CP_FLUSH_GENS` (8)
   generations; `cp_flush` records a retire event and waits the generation
   from seven flushes ago (measured: 47,797 waits = 10 ms per *run*). The
   fence is now real — a refcounted `cp_fence` holding an event recorded on
   `cp->stream` after the episode joins; a bare destroy double-freed under
   the first triangle because one submit's fence lands in several vk_syncs.
   `cp_fence_finish` waits that event, and `handle_pipeline_barrier` in
   lvp_execute flushes without finishing for cudapipe only — stream order
   already meets a barrier's device-device dependencies and every host
   reader drains on its own. Wall-neutral (the episode drain absorbed the
   freed 5.1 ms, as both relocation lessons predict) but 29.7 full pipeline
   drains per frame are gone; `CUDAPIPE_FLUSH_DRAIN=1` restores the drain.
3. **Merged shading groups** (item 3, first half): same-(vs,fs,ubos,mode)
   segments shade as one launch over a group-major slice of the grouped
   quad list; the interpolator resolves each quad's positions/slices/bases
   through a `cp_seg_range` table and the group's fs-UBO rows concatenate
   behind per-range row bases. Segments-to-groups is **5.4×** on this
   capture (9.3 segments per blended run, 1.7 unique keys — much more than
   the 2× estimated). Interpolate launches −3.9×, `main` device −27%/s —
   and the union didn't move, because the shades were already fanned out.
   Kept for the structure; `CUDAPIPE_NO_SEG_MERGE=1` restores per-segment.
4. **Instanced draws batch** (item 5, the sleeper): the slice table's pad
   word is now per-draw `verts_per_instance` and the fetch kernel divides
   within the slice; `instance_count` stays in the key so same-count clumps
   merge. Everything else — fs rows, clip rects, the VS's per-vertex
   zero-based instance id — already worked. **−5.6 ms, the largest single
   step of the pass**, and mostly not from the shading trios: the ~60 solo
   classic executes per frame each paid drains, memsets and ~12 launches of
   host machinery, and all of that went with them (stream syncs 47 →
   19/frame, launches 2,928 → 2,053/frame).

## Negative results (do not re-try; measurements in BASELINE.md)

- **Register caps.** NCU shows the big fragment shaders at 156-195
  regs/thread, 14-18% occupancy, 5-6% SM issue, and it still is not an
  occupancy problem: static 128 (`CUDAPIPE_REGCAP_STATIC=1`), forced 96 and
  forced 64 all regressed the capture (41.3/42.2/42.1 vs 40.75). Spills
  cost more than warps buy; the runtime tuner's veto is right.
- **fs_in SoA layout.** The varying loads really are 4.2-of-32
  bytes-per-sector uncoalesced — and a full plane-major implementation
  (interp + codegen + plane-stride upload, capture-gate clean) changed
  per-kernel device time by *zero* (main@4096 91.0 vs 91.1 µs). L1/L2
  absorb the waste; the latency chain is texture fetches. Reverted.
- **Async device copies, again.** Re-measured under the drainless pipeline
  as this document asked: still 0.0 (35.02 → 34.97). Reverted again.

## Where the frame is now (35.0 ms, from trace_inst.sqlite + cushim_inst)

- Union-busy ≈ 26.5 ms/frame (75.7%). The **main stream owns ~17.2 ms** of
  it: compiled fs ~5.5 (68 launches/frame of gx=4096 at 67 µs — texture
  latency at 14-18% occupancy), opaque raster stages 3.0, `cp_abuf_sort`
  1.9, `cp_fs_interpolate` 1.6, seg count/scatter 1.5, writeback 0.7,
  vertex fetch + clip 1.3, composite 0.4.
- Episode drains are the one big wait left: 12.5/frame, 19.5 ms wall — the
  host in `cp_pass_finish` while the device runs that critical chain. They
  are the big passes' blended kind-runs; the count is already minimal, and
  speculative (drain-free) episode completion is **unsound** for the same
  interleave reason as the `B1,O,B2` counterexample — a failed episode's
  repaint would land after later opaque color.
- The opaque runs have **merge ratio 1.00** on (vs,fs) — every batch in a
  run has a unique shader pair — so opaque episodes' win is not fewer
  trios; it is overdraw elimination in the fs plus per-shade-key grouping
  across the index_resource breaks (12.4/frame). The full deferral scout
  (depth is written only by writeback; the visbuf carries depth in its high
  word; color ordering pins the boundary to a consecutive opaque run;
  `cp_seg_range` needs a `prim_end` for filters) is in BASELINE.md's
  session-2 notes.

## What is left, re-ranked

1. **The texture sampler path.** The fs floor is latency in
   `cp_tex_sample` — a fully generic device function (every target, wrap
   and filter mode branched at runtime) whose linked allocation is what
   pins every textured shader at 195 registers. Specializing it at JIT time
   per bound sampler state (the sampler table is in the batch key already)
   attacks both the register ceiling and the dependent-load chain — the
   only lever left on the ~10 ms of `main`.
2. **Opaque episodes** for overdraw + shade-key grouping (~1-2 ms critical),
   using the range table and the scout above.
3. **Eligibility tail**: `depthfunc` (11/frame) and `topology` (8/frame)
   solo executes.
4. **Bloom chain latency**: the copies are measured-neutral twice now; what
   remains is the per-pass fixed cost itself.

A sober forecast: 2-4 are each ~1-2 ms; reaching 16 requires 1 to roughly
halve the fragment-shading floor, and then the episode drains (which shrink
with the device chain under them) come down with everything else.

## Session 3 checkpoint: sampler negatives, opaque groundwork

Fresh measurement reproduced the 35 ms handoff: 34.77 ms median, 61.15 s
cushim wall time, 2,054 launches/frame, 860 D32 memsets/frame, 12.49 episode
drains/frame at 19.42 ms/frame, and 75.5% device union in a 15-second nsys
window collected with `--kill=none`.

Two bounded sampler-specialization designs were correctness-clean and
performance-negative, then reverted:

- a runtime normalized-2D repeat/nearest/no-mip fast path measured 34.98 ms
  enabled versus 35.00 disabled;
- an indexed six-state NVRTC specialization with out-of-line generic fallback
  measured 35.95 ms specialized versus 35.88 generic, while its call boundary
  regressed the original baseline by about 1 ms.

The sampler table pointer/count in the batch key is not the selected sampler
state: shaders load sampler indices dynamically from descriptors. Do not retry
runtime index dispatch, common-state branches, or out-of-line sampler variants.
A viable attempt must resolve descriptor indices and clone the selected path
directly into shader IR/PTX without a device-call boundary.

Work moved to opaque episodes. Three independently gated prerequisites are in
the working tree: `cp_seg_range` has an exclusive `prim_end` and rejects gaps;
ordinary interpolation shares the bounded range resolver; ordinary visibility
packing honors the existing primitive base. The capture remains exactly
0.441%/0.002% against llvmpipe with loading screens bit-exact. The next diff is
a separate, single-stream consecutive-opaque-run episode with per-segment
shading first; shade-key grouping comes only after shared-visbuf/deferred-depth
semantics pass the gate.

## Session 3 result: consecutive opaque-run episodes

Implemented the separate opaque episode. Eligible consecutive batches rasterize
into one shared visibility buffer with episode-global primitive ids while depth
and color remain uncommitted. At the run boundary, ordinary interpolation
filters the winners through bounded segment ranges and shades each compatible
group once. `CUDAPIPE_NO_OPAQUE_EPISODE=1` restores the classic path.

The ungrouped implementation passed the exact capture gate first. Grouping
different material UBO tables was incorrect even with range-local row indices;
the safe retained key therefore requires identical `(vs, fs, num_fs_ubos,
mode, ndraws, fs_ubo_table)` state. The final grouped build again matches the
llvmpipe gate exactly at 0.441%/0.002%.

Paired full-capture medians were 34.57 ms grouped, 35.17 ms with
`CUDAPIPE_NO_SEG_MERGE=1`, and 35.03 ms with opaque episodes disabled. Thus
opaque deferral retains a modest 0.46 ms net win, while safe grouping saves
about 0.60 ms inside the opaque design. A fresh 15-second Nsight window remains
75.7% union-busy: compiled `main` owns 4.12 s, A-buffer stage2 1.70 s, A-buffer
interpolate 1.08 s, and ordinary interpolation 0.67 s. The next large lever is
still direct in-shader sampler specialization or another reduction in textured
fragment work; opaque episodes do not materially change the device floor.

## Session 3 result: sampler variants and short-run sorting

The CPU-profile pass used `perf -F 999 --call-graph dwarf` over the full replay
and collected 58k samples without loss. `cp_pass_finish ->
cuStreamSynchronize` dominates cudapipe's stacks because the CUDA driver
busy-spins; ordinary host computation is not the limiter. The remaining drain
time follows the device chain, so removing the wait without moving its
dependency would not improve frame completion.

Lowered-NIR provenance capture now records fixed `(ubo_slot, sampler_offset)`
references on each shader. The common capture shape is UBO 2 plus descriptor
offsets 64, 160, and so on. At launch, a single-static-descriptor shader whose
batch rows all resolve the same sampler state lazily links a replacement
sampler PTX compiled with that entire `cp_sampler_info` as literals. It retains
the existing one device-call boundary and falls back for dynamic, mixed-row,
or multi-descriptor shaders. `CUDAPIPE_NO_SAMPLER_VARIANT=1` disables it.
Image output remains exact; the narrow variant measured 34.47 versus 34.63 ms.
Generalizing one literal state across multiple descriptors regressed to
34.73 ms and was rejected.

The larger win is A-buffer sorting. Runs of 2 through 16 fragments now use a
one-thread insertion sorter while only longer runs enter the 256-thread
block-wide bitonic worklist. The same framebuffer scan builds the composite
worklist, deleting another launch and memset. Cutoff sweep medians were 32.97
(8), 32.84 (16), 32.87 (32), and 33.03 ms (64), so 16 is the default;
`CUDAPIPE_NO_ABUF_SHORT_SORT=1` restores the prior sorter. In the final Nsight
window, general `cp_abuf_sort` fell from about 0.81 s to 0.063 s and the new
short sorter cost 0.073 s.

Final paired median is **32.84 ms enabled versus 34.67 ms with both new
features disabled**, a retained 1.83 ms/frame improvement. The exact llvmpipe
gate remains 0.441%/0.002%. The final device window is still 76.5% union-busy;
`main`, A-buffer raster stage 2, and A-buffer interpolation are now the clear
remaining targets rather than sorting or host launch bookkeeping.

## Session 4 result: warp-aggregated segment bucketing

The A-buffer segment count and scatter kernels now aggregate equal segment IDs
within a warp. Count performs one global atomic add per peer group, and scatter
reserves one contiguous span per peer group before each lane computes its local
rank. `CUDAPIPE_NO_ABUF_WARP_BUCKET=1` restores the per-quad atomic path.

The framebuffer gate remains effectively identical to the preceding cudapipe
gold (0.000%/0.000% mean, with only the existing tiny nondeterministic values).
The paired full-capture medians are **30.52 ms enabled versus 33.10 ms
disabled**, a retained 2.58 ms/frame improvement. Launch count is unchanged at
about 2,053/frame, demonstrating that this win comes from reduced device-side
atomic contention rather than host launch removal.

Two follow-ups were rejected. Sharing the first lane's segment-search interval
across the warp was incorrect because grid-stride iterations can present
unrelated segment distributions. Per-texture-call sampler variants compiled
and ran correctly but increased JIT work and regressed the median to 33.20 ms;
the retained sampler variant remains the narrow single-static-state case.

The A-buffer allocation floor is now 5,242,880 fragments, eliminating the
repeatable startup overflow/replay path. NVRTC sampler float constants are
encoded through exact `__int_as_float` bit expressions, avoiding unsupported
C++14 hexadecimal-float tokens while preserving signed zero and special values.

## Session 5 checkpoint: launch-chain reduction

The short-run sorter now builds the long-run worklist during the same
framebuffer scan. This removes one launch and one counter clear from every
episode using short sorting. Prefix scans also use a three-launch path whenever
the first level has at most 1,024 block totals: scan the input, scan those
totals in one block directly into the established grand-total allocation, then
add the bases. The capture's pixel and quad scans both satisfy that bound,
instead of unconditionally using five launches.

Together these changes retain exact framebuffer output and improve the full
capture median from 30.52 to **30.33 ms**. Folding active-block discovery into
quad counting was also correct but regressed to 31.16 ms because it discarded
the sparse worklist's scheduling advantage; that experiment was reverted.

Fresh cushim measurement reports 3,026,331 launches for the replay, down from
3,100,691 before these fusions: about 74,360 launches, or 49.2/frame, removed.
The remaining A-buffer slot-map clear was also redundant: `quad_fill` assigns
every live fragment slot, and either capacity is proven sufficient or overflow
is drained and falls back before composite can read the map. Removing it from
both A-buffer paths passes the exact gate and removes another 25,155 launches
(16.7/frame). Its 30.39 ms median is wall-neutral against 30.33 within replay
noise, so it is retained as a pure structural launch reduction.

Overflow clamping is now part of the final prefix-add pass. That pass already
has each completed offset in a register and the grand total is device-visible;
when the total exceeds capacity it applies the same per-run room calculation
and increments the same overflow counter. This deletes the standalone clamp
launch while preserving fallback semantics. The exact gate passes and the
median is 30.38 ms, again wall-neutral. It removes another 25,154 launches
(16.7/frame), putting the estimated replay total near 2,976,022, or
1,971 launches/frame, before the next cushim confirmation.

Stage-2 threshold sweep confirmed that medium triangles were occupying one warp
too long. Full-capture medians were 29.62 ms at 1024, **29.60 ms at 1536**,
29.64 ms at 2048, 30.38 ms at the old 4096 default, and 30.76 ms at 8192.
The default is now 1536, moving larger bounding boxes to the tiled stage 3
path. The exact framebuffer gate remains unchanged. This is retained as a
0.78 ms/frame improvement over the immediately preceding build.

Fresh batch-break telemetry found 45,925 `instance_count` breaks and another
18,450 combined `instance_count,index_resource` breaks. The fetch kernel already
resolves `verts_per_instance` from each draw slice, but the batch host path had
discarded each draw's instance count and multiplied every slice by the first
draw's value. Batches and episode fallback snapshots now retain an
`instance_counts[]` row, total triangle sizing and slice spans use it per draw,
and `instance_count` is no longer a merge-key field. The exact capture gate
passes. Full replay improves from 29.60 to **27.09 ms**, a 2.51 ms/frame win.

Direct interpolation/shader fusion was mapped to a concrete design: retain
relocatable FS helper PTX, link a callable quad interpolator into each generated
fragment module, extend the shader argument block with interpolation state, and
have one lane per quad populate inputs before a warp barrier. It cannot be
introduced as a simple kernel concatenation because derivatives require every
quad lane to observe completed helper inputs. As the safe first step, the
A-buffer interpolator now has distinct range-free and range-aware entry points;
the common single-segment path compiles without range-table branches or state.
This passes the exact gate and improves 27.09 to **26.97 ms**.

Doubling the per-shader sampler cache from four to eight variants was exact but
measured 26.92 ms, only 0.05 ms from the four-entry result and below replay
noise while doubling retained modules and potential NVRTC work. It was
reverted. Per-call inlining still requires moving sampler implementation into
the LLVM-generated shader body (or a shared LTO representation); linking
separate relocatable PTX preserves the external device-call boundary.

Shader-stat replay showed the occupancy tuner accepting several capped builds
that were 1-5% slower, by design through `CP_TUNE_VETO=1.05`. The spill cost is
especially high here (typical 195-register shaders become 126-128 registers
with 160-224 bytes of spills), so accepting a slower individual trial has no
demonstrated downstream benefit. A strict 1.0 veto improves the full replay
from 26.97 to **26.68 ms** and is now the default: a cap is retained only when
its own median trial is faster.

The final 18-sample sweep remains correctness-stable: all processes exit zero
and only the standing `gltfscenerendering` and `texture3d` NVIDIA verdicts are
reported. Total sample time is 36.16 versus 36.13 ms (+0.1%). `particlesystem`
moves 3.71 to 3.96 ms (+6.7%); isolated controls at the old 4096 medium
threshold and old 1.05 tuner veto remain 3.98 and 3.96 ms, respectively, so
the tradeoff is the broader varying-instance batching. It is retained because
the target capture gains 2.51 ms from that batching and 3.85 ms across this
session's retained changes.

Final cushim confirmation reports 2,620,869 launches over 1,510 frames, or
about **1,736 launches/frame**, versus the session-start 3,100,691 / 2,053.
Replay wall falls from 56.15 to 50.39 seconds. Host launch API time falls from
4.81 to 3.71 seconds; episode-drain synchronization remains the dominant host
stack because it waits for the now-shorter device chain rather than creating
independent work.

Preparing triangle interpolation state once per quad, rather than repeating
the position loads, perspective divides, viewport transform, winding test, and
inverse-area calculation for all four lanes, passes the exact framebuffer gate
and improves the full-capture median from 26.68 to **26.38 ms**. Reusing the
already-known pixel x/y coordinates in the same path was also exact, but
regressed to 26.62 ms, likely from extra live arguments/register pressure; only
the triangle setup hoist is retained.

## Session 6: generated-stage fusion

The A-buffer interpolator is now linked as relocatable device code into every
generated fragment module. A non-null argument slot makes `main` interpolate
its own lane before entering the NIR-generated body; ordinary fragment passes
leave the slot null and retain the old path. Four lanes share one triangle
setup through warp shuffles, preserving derivative-safe quad execution while
removing the standalone `cp_abuf_interpolate` launch and its inter-kernel
`fs_in` handoff. The full capture improves from 26.49 to **26.00 ms**. Four
targeted frames compare exactly on two frames; the other two differ in only 41
and 24 pixels respectively, by at most two channel levels, which is the same
floating-point tie class accepted by the existing capture gate.

The final cushim census reports 2,588,655 launches, or **1,714/frame**. That is
another 32,214 launches removed from the prior 2,620,869 / 1,736 checkpoint;
replay wall is 49.51 seconds versus 50.39. The remaining dominant fixed chains
are vertex-fetch + VS + clip per geometry group and the three adaptive raster
stages, which is why graph reuse—not another local A-buffer fusion—is the next
credible route to a large launch-count reduction.

The matching vertex-fetch fusion was implemented and measured rather than
assumed. It linked a per-thread gather helper into each generated vertex module
and removed the standalone `cp_vertex_fetch` launch while retaining the packed
input ABI. It regressed the median from 26.00 to **26.72 ms** and increased
startup JIT cost substantially (the replay's largest lazy-compile frame grew
from roughly 0.74 to 3.20 seconds). Attribute conversion, slice lookup and the
VS body competing for registers in one kernel cost more than the launch and
intermediate buffer. The experiment was fully reverted; fetch remains a
separate kernel.

The sampler-inline experiment is likewise a completed negative result, not an
unimplemented guess: literal state variants and per-call provenance were built,
but linking the sampler as a callable external implementation retained the
device-call boundary and broader specialization regressed to 33.20 ms. True
inlining requires emitting the sampler implementation into LLVM-generated PTX;
the retained narrow variant is limited to the cases where it already wins.

Alternating vertex shaders are already admitted into opaque episodes and
merged into shading groups through global primitive ranges. This eliminates
the visibility and fragment work between those VS changes, but not each
segment's vertex/fetch launch: deferring those launches would also defer the
geometry buffers the shared raster pass consumes and requires a new episode
geometry arena rather than another key relaxation.

A resident CUDA kernel cannot dispatch the independently JIT-linked VS and FS
modules, so a literal persistent executor cannot preserve the current shader
ABI. The viable equivalent is a reusable CUDA graph whose generated-shader
nodes and fixed raster nodes are updated per episode. Capturing a fresh graph
per variable episode merely replaces launch calls with graph construction and
is expected to lose; reuse therefore requires canonical episode shapes and
stable parameter storage before it can reduce submission overhead safely.

## Session 7: Crossroads clipping correctness

The `headless_streamer_1818_20260817T173522.gfxr` capture exposed large black
regions in frames 633 and 756. Disabling opaque episodes, segment merging,
batching, sampler variants and the A-buffer did not change a pixel. Implementing
the previously unsupported `f2f32` and `unpack_32_2x16` NIR operations removed
real compiler diagnostics but also did not change these frames. Depth-mirror
clears, 3D slice filtering and implicit texture LOD were tested and rejected as
causes; performance-affecting diagnostics were reverted.

Disabling clipping restored the scene, which isolated the fault to the clip
volume. The clipper used Vulkan's `0 <= z <= w` convention while every later
stage uses Gallium's `-w <= z <= w` convention, including the explicit
`ndc_z * 0.5 + 0.5` depth mapping. Valid negative clip-space Z was therefore
cut away and the resulting interpolants drove the material shader to black.
Changing the lower plane distance from `z` to `z + w` fixes both target frames.
Their mean RGB difference from llvmpipe falls from 59.47 to 1.47 and from
17.96 to 1.62 respectively. The other two nearby probes fall to 1.30 and 4.62.

The full 1,496-frame timing replay shows no regression: median frame time moves
from 8.76 to **8.57 ms** (mean 10.86 to 9.89 ms; startup JIT remains in the
mean). The correction changes only one add in the clip kernel and does not add
launches or synchronization. The 18-sample one-frame Vulkan sweep also exits
zero for every sample.

Frame 907 exposed the matching side-plane omission. Its sky and UI rendered,
but large ground/building triangles crossing the right side of the frustum
disappeared because only Z/W clipping was performed; clamping their post-divide
bounding boxes was not numerically equivalent. The clipper now handles the
full homogeneous X/Y/Z volume and reserves eight stable output triangles per
input triangle. Frame 907's mean RGB error against llvmpipe falls from 26.55 to
**1.20**; frames 906 and 908 remain at 1.20, and the earlier four probes remain
between 1.29 and 1.62. A full replay measures 8.78 ms median versus the original
8.76 ms, with no measurable performance regression.

## Session 8: Crossroads performance profile

The corrected 1,496-frame Crossroads capture was profiled independently with
an eight-second Nsight Systems CUDA window, an eight-second 999 Hz DWARF
`perf` attachment, and a full-replay cushim census. The persistent artifacts
are under `/home/alexzhukov/crossroads/profile-cudapipe-nsys-fullclip`,
`/home/alexzhukov/crossroads/profile-cudapipe-cpu-fullclip`, and
`/home/alexzhukov/crossroads/profile-cudapipe-cushim-fullclip`.

This workload is submission- and synchronization-bound rather than saturated
device compute. In the Nsight window, kernels occupy a 3.32-second union and
kernels plus memory operations occupy 3.85 seconds, only **48.2%** of the
7.99-second window. Kernel durations sum to 3.82 seconds because some streams
overlap. The generated shader `main` functions are the largest device item at
1.240 seconds, followed by `cp_vertex_fetch` at 0.511 seconds,
`cp_fs_writeback` at 0.400 seconds, the three A-buffer raster stages at 0.612
seconds combined, `cp_clip_triangles` at 0.243 seconds, the three ordinary
raster stages at 0.400 seconds combined, and `cp_fs_interpolate` at 0.198
seconds. Full-volume clipping is therefore not the new bottleneck: it is 6.4%
of summed kernel time and its 3.5-microsecond median is mostly fixed launch
cost on this capture.

The full cushim run records 689,019 kernel launches, or **460.6 launches per
frame**. Each frame averages 41.4 vertex-fetch and clip launches, 17.4 ordinary
raster/interpolate/writeback chains, and 24.1 A-buffer raster chains. It also
records 239 `cuMemsetD32Async`, 41 `cuMemsetD8Async`, 222 small asynchronous
host uploads, 9.0 stream synchronizations, and 7.8 context synchronizations
per frame. This is much better than the old capture's 1,714 launches/frame,
but is still far above the amount of useful parallel work. The Nsight CUDA API
table attributes 0.623 seconds to 311,949 `cuLaunchKernel` calls and 2.487
seconds to 6,316 `cuStreamSynchronize` calls. On-CPU `perf` independently
places the dominant cudapipe stack in `cp_pass_finish` at the episode drain;
`LZ4_decompress_safe` consumes 7.7% of sampled process cycles but belongs to
gfxreconstruct rather than the driver.

The priority order for this recording is consequently:

1. **Remove synchronous episode drains.** Replace the counter read in
   `cp_pass_finish` with device-side indirect sizing and overflow/status flags,
   retaining a host check only at an externally visible boundary. The full
   run performs 10,775 drains at this site and spends 5.31 seconds waiting in
   them under cushim. This is the highest-confidence latency improvement.
2. **Reuse canonical CUDA graphs for draw chains.** Cache graph executables for
   fetch + generated VS + clip + three raster stages and for the ordinary
   interpolate + generated FS + writeback tail, updating stable argument
   blocks rather than issuing each node separately. The repeated chains alone
   account for hundreds of launches per frame; graph reuse is preferable to a
   resident executor because generated shaders remain independently linked
   CUDA functions.
3. **Move small clears/uploads into device command preparation.** The capture
   issues roughly 280 asynchronous clears and 222 tiny uploads per frame.
   Persistent argument/counter arenas with generation stamps can eliminate
   most zeroing and driver calls without changing raster semantics.
4. **Build an episode geometry arena.** Concatenate compatible vertex/index
   rows, run one fetch/VS/clip dispatch per shading identity, and carry global
   primitive ranges into the existing shared raster machinery. This attacks
   the 41.4 fetch and 41.4 clip launches per frame and their combined 19.8% of
   summed device time, but is more invasive than graph reuse.
5. **Optimize generated fragment shaders only after launch work.** Shader
   `main` is 32.5% of summed kernel time, but the GPU is idle over half the
   trace and prior broad sampler specialization regressed. Per-shader register,
   spill, and texture-instruction data should select narrow variants; another
   unconditional sampler rewrite is not justified by this profile.

An off-CPU BPF capture was not collected because sudo authentication was not
cached. That is not blocking this conclusion: both the on-CPU caller stacks
and CUDA API trace identify the same episode synchronization path, while the
device timeline directly measures the large idle gaps and short repeated
kernels.

## Session 9: Crossroads synchronization reduction

The first implementation pass removed the unconditional
`cuCtxSynchronize()` from `cp_resource_copy_region`. Cudapipe resources are
CUDA-managed allocations, so their block-correct, mip-aware rectangular copy
can remain in stream order through `cuMemcpy2DAsync`; the synchronized CPU
row-copy remains as the fallback when either resource is host-only. The
equal-format blit path now uses the asynchronous form of the same CUDA copy as
well, rather than the blocking `cuMemcpy2D` call.

This removes **3,335 full-context synchronizations per replay** from the new
capture. Cushim measured that call site at 402.6 ms before the change. After
the change, total `cuCtxSynchronize` calls fall from 11,716 to 8,381 and the
new 28,025 rectangular async copies cost 71.5 ms to issue. End-to-end frame
time is neutral within run variance: 8.81 ms median versus the 8.78 ms
full-clipping baseline, while cushim replay wall is 17.12 versus 17.18 seconds.
The change is retained because it removes an unnecessary global ordering
point, does not regress the median, and is required before later work can
overlap resource transfers with rendering.

Frames 633, 756, and 907 were dumped again after the change. Their mean RGB
errors against llvmpipe are respectively 1.64, 1.62, and 1.21, matching the
corrected rendering class. The probe artifacts are under
`/home/alexzhukov/crossroads/perf-async-resource-copy/probes`.

The proposed direct removal of the `cp_pass_finish` drain was audited but not
implemented as a speculative shortcut. The readback currently supplies the
only trustworthy fragment/quad overflow verdict before colour is committed,
as well as exact per-segment counts used for group bases, allocations, and
composite descriptors. Merely launching capacity-sized kernels and checking
the flag later makes overflow unreplayable after partial writeback. A correct
no-drain implementation must first move segment prefix sums, dense bases, and
descriptor addressing to the device, allocate bounded group-wide output
arenas before shading, and preserve a pre-commit overflow gate. That is one
coherent device-compaction change, not a safe deletion of a synchronization.

CUDA graph reuse has the same prerequisite. Current graph node arguments point
into bump-allocated upload and scratch arenas, and episode shape, generated VS
and FS functions, launch grids, and side streams vary between instances.
Capturing or instantiating a graph for each episode would add work rather than
remove it. Reuse requires stable per-shape argument slots and cyclic lifetime
tracking first. The trace makes the eventual target clear: 23,313 one-block
generated `main` launches in eight seconds perform only 102 ms of device work,
whereas the 6,663 4,096-block launches perform 527 ms. Graph/geometry batching
should target the former; shader code generation should target the latter.

The first no-drain episode specialization is now implemented for the case that
can prove correctness without a device conditional. If every segment has one
shading identity and the conservative bounds of one quad per primitive per
2x2 block and four fragments per such quad fit the existing allocations, the
episode cannot overflow. It shades the original global quad stream directly
through the segment range table, uses the device-resident exact quad count for
thread rejection, and composites through global shading slots. It therefore
needs neither the host counter read nor the segment count/scatter compaction.
Any mixed-shader or insufficiently bounded episode takes the unchanged checked
path.

On Crossroads this recognizes 2,442 episodes per replay. Stream
synchronizations fall from 13,494 to **11,052**, synchronous counter copies
from 13,311 to **10,869**, and kernel launches from 689,019 to **684,135**.
The paired full replay improves from 8.78 to **8.74 ms median** (8.81 ms for
the async-copy-only checkpoint), and the three correctness probes are
pixel-identical to the async-copy checkpoint. The remaining 8,333
`cp_pass_finish` drains are mixed-group or conservatively too large; removing
those requires the device prefix/descriptor compaction described above.

The bounded path now also supports multiple shading groups without a host
prefix sum. A GPU ownership pass maps each original quad to its segment. Every
group shades against the same global quad-slot space, with its range table
rejecting quads owned by other groups, and each segment descriptor selects its
group's output while retaining the original slot index. The composite can
therefore resolve segment, group output, coverage, and discard state without
host-visible counts, dense bases, or a scatter pass. The same conservative
fragment/quad bound remains the pre-commit overflow proof; episodes that do not
fit it retain the old drained path.

The first implementation ran the ownership pass even for one-group episodes.
It was exact but raised launches from 684,135 to 686,577 and moved the median
from 8.74 back to 8.78 ms, so that form was rejected. The retained split keeps
the direct zero-ownership-kernel route for one group and invokes device
ownership only for real multi-group episodes. Crossroads has no additional
mixed-group episode whose worst-case bound fits the current arrays, so its
counts remain those of the single-group checkpoint; the final paired median is
**8.73 ms**. The implementation nevertheless removes the CPU prefix/descriptor
dependency for bounded mixed groups instead of weakening overflow handling.

## Session 10: synchronization audit and allocation reuse

The two replay baselines for this pass were 8.77 ms median over 1,496 frames
for `headless_streamer_1818_20260817T173522.gfxr` and 26.95 ms over 1,510
frames for `headless_streamer_20260814T155742.gfxr`.

Resource mapping no longer performs a device-wide `cuCtxSynchronize`. Batch
flushes join deferred side streams into the context's main stream, so waiting
on that stream preserves the map contract without stalling unrelated CUDA
work. This changed the new/old medians to 8.81/26.88 ms: neutral on the new
capture and a small improvement on the old capture. Cushim confirms the
ordering point moved as intended: context synchronizations fell from 8,381 to
338, with 8,043 corresponding stream waits added.

The resource allocation audit found two proposed synchronization changes were
already present. Scratch generations retire through CUDA events and only the
rare threshold/overflow reclaim synchronizes the context; 38,572 generation
event waits cost only 6.8 ms in the old profile. Shader tuning likewise uses
`cuEventQuery` and harvests completed trials asynchronously. Neither path had
an unconditional wait left to remove.

The retained CPU/API win is a bounded cache for separate managed
`VkDeviceMemory` allocations from 256 bytes through 4 MiB. Freed allocations
are reused by exact power-of-two size class, with a 512 MiB and 4,096-entry
cap. Vulkan requires device memory to be idle before it is freed, so reuse does
not need a deferred-free event; keeping allocations separate also avoids the
managed-memory page-sharing regression seen with enlarged slabs. Allocation
calls fell from 24,059 to 7,689 and frees from 24,656 to 8,190 on the new
capture. With the map change, medians improved to 8.52 and 26.24 ms.

Two more conservative no-drain bounds were tested. Computing the episode's
worst case from each segment's raster clip rectangle rather than the whole
framebuffer measured 8.43/26.19 ms, but did not change launch, counter-copy, or
stream-wait counts; the apparent improvement is therefore run variance. The
same rectangle proof applied to standalone A-buffer draws measured
8.46/26.27 ms and again admitted no additional work. It is a slight negative
and was removed along with the neutral episode-bound experiment before the
final commit.

Removing initialization of freshly created resources was also tested and was
wall-neutral. The 60-frame Vulkan sweep then exceeded the existing NVIDIA
animation budgets in `gltfscenerendering` and `texture3d`, so the initialization
was restored immediately. Repeating the complete 18-sample sweep after the
restore produced the same two verdicts. These are standing differences already
recorded by earlier final sweeps rather than new failures from allocation
reuse; every sample exits successfully, and the tested total is 34.01 ms versus
36.56 ms for `crossroads_multigroup` (-7.0%). Representative Crossroads frames
633, 756, and 907 remain effectively identical to the corrected pre-pass
images (maximum channel errors 0, 1, and 0).

The exact pre-cleanup tested tree measures **8.47 ms** on the new replay and
**26.21 ms** on the old replay. Relative to the pass baselines those are wins
of 0.30 ms (3.4%) and 0.74 ms (2.7%). Most of the credible gain and the large
drop in allocation API traffic come from managed-allocation reuse. The map
change is structurally preferable and neutral-to-positive. Both bound changes
were removed because neither changed the launch or synchronization census; the
cleanup restores the admission rules used by the already validated 8.52/26.24
ms allocation-cache checkpoint.

The remaining mixed/unbounded A-buffer drain cannot safely be deleted: exact
counts and the overflow verdict are required before color is committed. CUDA
conditional graph nodes can express a device-side fast/fallback choice, but a
useful implementation first needs reusable stable graph bodies for both
pipelines. Per-episode graph construction would add more work than it removes.
This remains the next architectural synchronization project rather than a safe
local patch.

## Session 11: undefined allocation initialization and executor boundary

The largest blocking-memory cost was not resource initialization. It was
clearing every fresh or recycled `VkDeviceMemory` allocation even though
Vulkan defines ordinary allocation contents as undefined. Lavapipe already
performs the required host clear when
`VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` is present, after mapping the
allocation. Cudapipe therefore no longer calls `cp_zero_managed` from
`cp_allocate_memory`; directly created resources retain their initialization.

This deletes all 1,571 blocking `cuMemsetD8` calls in the new replay, previously
830.6 ms of CUDA API time. The new capture improves from 8.47 to **7.52 ms**
median and the old capture from 26.21 to **25.06 ms**. The complete 60-frame,
18-sample sweep is behavior-stable relative to the preceding build and moves
34.01 to 33.88 ms (-0.4%); the same standing `gltfscenerendering` and
`texture3d` NVIDIA-reference verdicts remain. Crossroads frames 633, 756, and
907 are bit-exact against the corrected pre-change images.

With allocation clearing gone, the current cushim replay wall is 15.06 seconds.
The 8,333 episode drains consume 5.45 seconds (**36.2% of wall**), standalone
A-buffer drains another 0.86 seconds, and kernel launch API work 0.93 seconds.
The blocking initialization line is absent. The CPU decision is now more
clearly the dominant remaining cost.

Two device-sized-work variants were measured and rejected. Making ordinary
interpolation use a capped resident grid measured 7.48 ms on the new capture
but 25.09 ms on the old one, effectively noise versus 7.52/25.06, and was
reverted. Generated `main` and writeback already grid-stride over GPU-resident
counts; reducing `main` from 4,096 to 2,048 blocks regressed 7.48 to 7.53 ms and
was reverted. Raising the A-buffer startup floor from 5 to 8 million fragments
did not avoid the observed two-million-fragment overflow, increased resident
memory by roughly 75 MB across fragment and quad arrays, and regressed 7.52 to
7.56 ms, so it too was reverted.

The overflow is live rather than theoretical: the correctness probe records a
new-capture draw exceeding even the enlarged array and safely falling back to
peeling. Consequently the episode counter read cannot be replaced by an
unchecked fast path. A correct device executor must include both branches
before any visible writeback: the normal grouped A-buffer shade/composite and
the classic per-segment fallback. The fallback itself contains device-dependent
peel termination plus dynamically linked shader launches. CUDA conditional
graphs therefore require reusable graph bodies for the complete normal and
fallback pipelines, stable cyclic argument storage, and kernel-node parameter
updates; a conditional node around only the composite is insufficient.

CUDA 12.8 conditional graphs were exercised directly on the target GPU. An
`IF/ELSE` node whose condition is set by a preceding device kernel executes
correctly. A 10,000-iteration microbenchmark measures 10.27 us per reusable
conditional-graph launch versus 4.10 us for two direct one-block kernel
launches. Conditional graphs are therefore not a profitable wrapper for one or
two existing stages. They are still appropriate for the episode decision: one
graph launch can replace a roughly 654 us average stream drain and the many
normal/fallback launches below it. This reinforces the required granularity —
cache the complete episode tail, not individual raster stages.

There is also a toolchain prerequisite. The device setter calls
`cudaGraphSetConditional`, supplied by the CUDA device runtime, while
cudapipe's common kernels are currently compiled by NVRTC and loaded directly
from PTX without `libcudadevrt`. The executor implementation must link a small
setter module against the device-runtime archive (or move that one module to a
build-time fatbin) before constructing conditional driver graphs. Adding the
opaque handle to an ordinary NVRTC module without that link leaves an
unresolved device function.

The current capture's 8,333 drained episodes have a very small shape set:
5,325 contain one segment, 2,730 contain nine, 195 contain five, 69 contain
three, and 14 contain four. None overflowed. A deliberately unsafe diagnostic
mode skipped the episode counter readback and assumed the bounded path; median
frame time moved only from 7.52 ms to 7.40 ms. The large synchronization API
duration therefore mostly represents GPU work on the critical path, not CPU
decision overhead. A conditional executor remains useful for launch reduction,
but is no longer the leading expected frame-time win.

Unlinking fused A-buffer interpolation from generated fragment shaders reduced
the three sampler-free shaders from the usual roughly 195 registers to 22, 12,
and 4 registers. Textured/helper shaders remained at roughly 195 registers due
to the sampler module. Restoring a separate interpolation launch globally made
the new replay slower, 7.59 ms versus 7.52 ms, so the experiment was reverted.
A dual module could retain a lean ordinary path for the few sampler-free
shaders, but its likely benefit is smaller than its module/tuning complexity.

Literal sampler variants had silently failed NVRTC compilation because the
generated exact float constants were emitted as C++14-incompatible hex-float
literals. They now use bit-preserving `__int_as_float((int)0x...U)` expressions,
including correct handling of signed zero and special values. The new capture
measures 7.47 ms with variants enabled and 7.51 ms with them disabled; the old
capture measures 24.97 ms enabled. This is a small win, but it also restores the
intended specialization path instead of permanently falling back after compile
failure.

Two scalar argument uploads were redundant. Fragment shaders uploaded their
input stride separately even though the main argument allocation already has
inline scalar storage, and compute dispatches uploaded their three grid sizes
separately from the pointer array. Both now use one contiguous DMA block. This
removes one `cuMemcpyHtoDAsync` per fragment shader launch and one per compute
dispatch. The new replay measures 7.44--7.46 ms versus 7.47 ms immediately
before the change; the effect is small but consistently reduces host/API work
without adding device work.

The new capture has 13 fragment shaders with two or four statically traceable
sampler descriptors, versus 29 with zero or one. Specializing the multi-sampler
subset only when every descriptor resolved to the same state moved the new
capture from 7.46 ms to 7.43 ms, but regressed the old capture from 24.83 ms to
25.15 ms. That global-state generalization was reverted. Future multi-sampler
work needs per-texture-instruction literal entry points; making every call use
one shared state does not remove enough code to justify the added variants.

## Session 12: unified-memory profile

Nsight Systems 2026.4.1 traced both CPU and GPU unified-memory page faults over
a 14.31-second steady-state window of the newer gfxrecon capture. It recorded
509,695 GPU faults across 673 pages and 71,898 CPU faults across 791 pages.
GPU fault intervals occupy a 1.552-second union, 10.84% of the window. UVM
migrated 14.25 GB, of which 1.71 GB was directly attributed to page faults and
12.53 GB to speculative prefetch around them. CPU faults overwhelmingly came
from host memcpy (66,192 events), confirming the expected host-write/device-read
ping-pong rather than cold startup alone.

At the replay's approximately 7.45 ms median this corresponds to an estimated
265 GPU faults, 37 CPU faults, and 7.42 MB of migration per frame. Nsight
instrumentation can perturb timing, so the 10.84% union is not itself a promised
speedup, but the counts and migration volume are large enough to promote split
host-visible and true device-local Vulkan memory ahead of persistent execution.
The trace is stored under
`~/claude-scratchpad/perf16/uvm-profile-1787062168/`.

## Session 13: split host-visible and device-local memory

Lavapipe now exposes three memory types when its screen supplies the optional
device allocator: device-local-only, host-visible/coherent/cached, and a managed
compatibility type carrying both sets of flags. Ordinary llvmpipe still exposes
its original single universal type. Cudapipe backs the first type with
`cuMemAlloc`; the other two retain the managed/pinned allocation path and its
small-allocation/cache policy. The compatibility type avoids rejecting software
that explicitly requires both flag sets without attracting the samples, whose
first-match searches select the two specialized types.

Device allocations are opaque Gallium allocation objects containing their CUDA
address. Resource binding unwraps that address, normal copies and rendering stay
device-to-device, and CPU maps or format-conversion blits explicitly stage the
allocation through host memory. `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT`
uses a device memset rather than dereferencing the opaque address. A bounded
device-allocation cache was also tested; retaining its VRAM worsened the old
capture from 25.39 to 25.80 ms, so it was removed.

The newer Crossroads capture improves from 7.44--7.46 ms to 7.18--7.21 ms
median, about 3.5--3.8%. Its 14-second UVM trace changes as follows:

| metric | managed-only | split | change |
|---|---:|---:|---:|
| migrated bytes | 14.25 GB | 1.59 GB | -88.9% |
| GPU faults | 509,695 | 403,209 | -20.9% |
| unique GPU-fault pages | 673 | 212 | -68.5% |
| CPU faults | 71,898 | 34,451 | -52.1% |
| GPU-fault interval union | 10.84% | 5.73% | -47.1% |

The older, overflow-heavy capture regresses from 24.83 to 25.39 ms. The full
18-sample, 60-frame sweep is pixel-stable relative to the preceding iteration;
the standing NVIDIA differences remain `gltfscenerendering` and `texture3d`.
Crossroads frames 633, 756, and 907 are byte-for-byte identical to the corrected
reference. The post-split UVM artifact is
`~/claude-scratchpad/perf16/uvm-memory-split-1787065175/`.
