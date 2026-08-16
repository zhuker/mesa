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
