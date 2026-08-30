# 2026-08-30: the merge-key and threshold campaign, on the compiled HeadlessStreamer harnesses

Goal set by the user: favorite3 (the occlusion capture,
`~/headless_streamer_101754227324635_20260830T013431.gfxr`) from 9.69 ms to
at most 6 ms, ideally 5. Result: **6.12-6.25 ms by session** (favorite2
5.87, favorite1 5.10), four commits, all correctness-gated. Everything was
measured on the tocpp-compiled replay harnesses (`~/favorite{,2,3}-cpp`,
READMEs inside), which carry no gfxrecon replayer floor; frame time is the
paired-submit median, skip 50.

## The chain of evidence

1. `CUDAVK_PLAN_STATS` batch-break tally on favorite3: 332,027 forced
   flushes, led by **vertex buffers 144,008** and index buffer 23,276 —
   draws differing only in buffer bindings, everything else equal (the
   tally is taken at the first failing check, and the buffer checks are
   last).
2. The per-draw mechanisms already existed: `elem_bases` rows in
   `cp_vf_lane.h` ("so draws bound to different vertex buffers merge", its
   own words) fed by `cp_batch_record_packet`; batching structurally
   requires a classic-exec VS, which keeps the batch-wide no-VS passthrough
   unreachable. The frontend key was simply never relaxed.
3. NCU on `cp_clip_rast_fused` (32 mid-replay launches): occupancy 3-6%,
   ~8.4 cycles/issued-instruction — healthy issue, no warps. Stage 1 lanes
   were serially walking up to 128-pixel boxes; the thresholds, not stalls,
   were the kernel-side lever.
4. The episode-drain conversion probe (ported to HEAD as
   `CUDAVK_WAIT_SPIN_[BEFORE_]US`): before-arm slope +0.02, after-arm
   +0.21 — both directions have slack, a deferral mechanism ceilings at
   ~0.2 x 1.27 ms/frame. That lead is closed at today's numbers.

## The commits

| commit | change | favorite3 effect |
|---|---|---|
| 32c73fe21c8 | vertex buffers out of the merge key (`CUDAVK_KEEP_VBKEY`) | 9.69 -> 6.78; segments -72%, launches -37% |
| c3f75a2ac70 | index buffer out of the merge key, per-draw slice base (`CUDAVK_KEEP_IBKEY`) | -0.19 at tuned thresholds |
| 188bb5646c2 | `CP_SMALL_THRESHOLD` 128->16, `CP_MEDIUM_THRESHOLD` 1536->256 | 6.78 -> 6.13 out of the box |
| 5f2bb8de327 | the drain conversion probe (instrument only) | evidence above |

Cross-capture: favorite2 9.46 -> 5.87, favorite1 9.23 -> 5.10, thresholds
swept and at their plateau on all three. `CUDAVK_CTX_SCHED=spin` is a
favorite3-only ~-0.11 that does not transfer; left as env.

## Correctness

- Suite 79/79 after every commit; FLAGS.md regenerated; no-getenv clean.
- Frame dumps: VBKEY and both threshold flips **byte-identical** on all 18
  dumped favorite3 frames. IBKEY moves at most 0.11% of pixels on 6 of 18
  frames with the distance to the NVIDIA reference unchanged to 1e-4 —
  the equal-depth tie-break class, from batch composition renumbering
  primitive IDs.
- 18-sample sweep (`campaign-vbib-thresh`): 16 ok, `renderheadless`
  missing as always, `gltfscenerendering` REGRESSED — and a four-arm
  revert bisect reproduces the identical deviation (worst 137,028+-1) with
  every campaign change disabled. The iterate history shows the same
  number since 2026-08-26 (`pdl-l3`, 67244f6dc62f): **pre-existing, not
  this campaign's**, and still unchased.

## Stale numbers this leaves behind

`CUDAVK.md` and `PERFORMANCE.md` §1 still quote the old capture pair
(12.78 / 5.69); this campaign measured only the three HeadlessStreamer
compiled harnesses, so those tables are not touched here — but any future
measurement of the old captures must expect the thresholds and merge keys
to have moved them.

## What is left on favorite3

The frame is device-saturated in the heavy phase (GPU busy 82%) and
mostly saturated mid (65%); the drain family is closed by the probe. The
honest remaining pools: the `main` fragment-shader time (2.9-5.0 ms/frame
of real shading), and the rearchitecture families in
`docs/cudavk/notes/REARCHITECTURE_IDEAS.md` (in the `cudavk/rearch-ideas`
branch/worktree).
