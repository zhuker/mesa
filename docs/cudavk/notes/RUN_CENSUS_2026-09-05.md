# The redesign probes (P0 and P1)

Status after re-running the host probes with the exact shipped episode
predicate: **P0 passes on both RTX captures. P1 passes on the RTX 5090; its
required B200 run is still pending.** The benchmark programs now return a
non-zero exit status when a correctness or timing gate fails.

| probe | gate | favorite3 | favorite2 | status |
|---|---|---|---|---|
| 1 run census | admitted >= 60% of references | **99.1%** | **99.2%** | pass (triangle proxy) |
| 2 identities/run | p99 <= 64 | **18** | **13** | pass |
| 3 stencil | none in candidate runs | **0** | **0** | pass for these captures |
| 4 split walk | p99 chunk <= 30 us, merge <= 10% | **10.10 us, 2.8%** | — | RTX pass; B200 pending |
| 5 bin pass | <= 0.15 ms/heavy frame | **0.0391 ms** | — | RTX pass; B200 pending |
| 6 `.local` baseline | recorded | `DEAD_ENDS` 42 | | done |

## Probe 1: the run census

Run at HEAD on both captures with `CUDAVK_RUN_CENSUS=1`. Host-side only: it
classifies every draw the way the run-level renderer would and counts the runs
it would form. It changes no rendering decision. Both runs completed with the
expected stdout hashes and 6,947 / 6,965 submit rows.

The first implementation compared only class and scope serial. The corrected
implementation asks `cpvk_draws_episode_compatible()` directly and ends a run
at every ordered non-draw operation. Using the front end's real predicate is
important: the shipped wide-episode change proved viewport, rasterizer and
depth state are per-segment and therefore must not be copied into a second,
stricter census predicate.

**The gate was: stop if admitted runs hold under 60% of a frame's references.
They hold 99.1% and 99.2%. The probe passes, and not marginally.**

Triangles remain a proxy for tile references. M0 measured 1.16 references per
triangle with no strong per-draw skew, so this is enough to keep the 60% gate
far from its boundary, but it is not an exact per-run reference census.

## favorite3 (2,083 real frames)

| class | runs | runs/frame | draws/run | longest | triangles |
|---|---:|---:|---:|---:|---:|
| opaque | 2,153 | 1.03 | 181.3 | 244 | **90.8%** |
| depth-only | 2,876 | 1.38 | 13.0 | 34 | 6.0% |
| blended | 45,802 | 22.0 | 2.9 | 121 | 2.2% |
| fallthrough | 4,449 | 2.1 | 1.5 | 4 | 0.9% |
| **admitted** | **50,831** | **24.4** | | | **99.1%** |

## favorite2 (2,078 real frames)

| class | runs | runs/frame | draws/run | longest | triangles |
|---|---:|---:|---:|---:|---:|
| opaque | 2,098 | 1.01 | 164.0 | 216 | **95.2%** |
| depth-only | 755 | 0.36 | 10.4 | 23 | 1.7% |
| blended | 45,205 | 21.8 | 2.2 | 15 | 2.3% |
| fallthrough | 3,591 | 1.7 | 1.6 | 3 | 0.8% |
| **admitted** | **48,058** | **23.1** | | | **99.2%** |

## What this says about the plan

- **One opaque run per frame still carries ~91-95% of the triangles**, at
  164-181 draws each. The exact shipped compatibility predicate leaves this
  central premise intact.
- **The depth-only relaxation is real but is a tail item.** favorite3 has 1.38
  depth-only runs per frame at 6.0% of triangles; favorite2 has 0.36 at 1.7%.
- **Blended runs are shorter than the original report said.** Ordered non-draw
  boundaries raise them from about 15 to about 22 runs/frame and reduce them
  to 2.2-2.9 draws/run. M3 cannot rely on batching for its win; it must remove
  the scan/fill/sort/list work.
- **There are 23-24 admitted runs/frame**, not the original 16-18 estimate.
  At 8-30 launches/run this implies 184-720 launches before fallbacks. The
  plan's 100-250 launch estimate is therefore not established.

## Corrections made while running it

The first version hooked `cp_batch_record_packet`, which sees only draws the
batcher accepted. It reported zero depth-only runs. The hook moved upstream of
the batching decision, where all draws are visible.

The next version joined draws using only class and scope serial, so it missed
ordered non-draw boundaries and did not call the renderer's compatibility
predicate. The fixed probe uses the predicate directly and closes after the
pending renderer work has resolved its shader identities. This keeps the
identity histogram attached to the run that generated it.

**A census of a decision must be taken upstream of that decision and must use
the same predicate as the decision.**

## Probe 2: fragment-shader identities per run

The identity is the `(binary, resolved exec)` pair `cp_fs_args_prepare()`
settles on; that pointer already folds in exec mode, sampler variant and tune
alternate, so it is the plan's four axes and it is exactly what one shade
launch would cover. **Gate: stop if p99 exceeds 64.**

| class | favorite3 mean / p50 / p90 / p99 / max | favorite2 |
|---|---|---|
| opaque | 12.75 / 13 / 17 / **18** / 18 | 11.00 / 11 / 13 / **13** / 13 |
| depth-only | 1.00 / 1 / 1 / 1 / 2 | 1.00 / 1 / 1 / 1 / 1 |
| blended | 0.98 / 1 / 1 / 2 / 3 | 0.98 / 1 / 1 / 2 / 3 |
| fallthrough | 1.00 / 1 / 1 / 1 / 1 | 1.00 / 1 / 1 / 1 / 1 |

No run exceeded the 128 tracked identities. The opaque gate passes with a
3.5x margin. A small number of blended runs record zero identities because
they produce no shade launch; that is a launch census result, not an identity
collision.

The number that matters for the design: **an opaque run of 181 draws needs
13-18 shade launches**. Add one geometry launch per VS identity, one clip+bin,
one walk and one compaction and the run costs roughly 20-25 launches where
today its ~110 batch chains cost several hundred.

## Probe 3: stencil and alpha-to-coverage

**No code needed; the driver already answers it.** `cpvk_pipeline.c` computes
`stencil_visible` -- a stencil configuration that would change pixels -- and
announces it once per process before ignoring it. **Neither capture emits that
line**, so no candidate run contains stencil the redesign would break. The
driver has no alpha-to-coverage state at all, and `cp_draw_state` carries no
stencil fields, so there is nothing for a run to disagree about.

This is a limitation of the *driver*, recorded in `TODO.md` ("full stencil
state"), not of the redesign: both renderers ignore stencil identically.

## Probe 4: hot-tile splitting

`src/cudavk/tests/cp_tilewalk_bench.cu`, extended with chunk splitting and an
`atomicMin` merge. RTX 5090, the M0 mixed grid (3,600 tiles, 1,744 non-empty,
hot tile pinned to M0's p99 of 12,823 references).

| arrangement | chunks | p50 | p99 | max | walk wall |
|---|---:|---:|---:|---:|---:|
| unsplit, one block per tile | 3,600 | 0.21 us | 23.11 us | 35.94 us | 0.0440 ms |
| **chunk 512** | 3,968 | 2.22 | **10.10** | 11.96 | **0.0221 ms** |
| chunk 1024 | 3,737 | 1.30 | 16.33 | 19.40 | 0.0281 ms |
| chunk 2048 | 3,644 | 0.24 | 26.40 | 32.01 | 0.0398 ms |

The `atomicMin` merge costs **2.8%** at chunk 512, inside its 10% gate. Splitting is order-independent, so it changes no result.

**Splitting does not merely fix the tail; it halves the walk.** 0.0435 ->
0.0221 ms at chunk 512, because 3,968 evenly sized blocks fill 170 SMs where
3,600 wildly uneven ones do not. Dead end 24 and Renderer 2 both died on the
one-block signature (94% of a grid's duration in one block); at chunk 512 the
worst block is 11.28 us against a 0.0221 ms kernel.

## Probe 5: the bin pass

`src/cudavk/tests/cp_binpass_bench.cu`, new. 354,000 post-clip triangles into
3,600 tiles of 16 px, count -> scan -> scatter, every list exactly sized from
a device count (sizing from a worst-case bound is dead end 11).

| phase | ms |
|---|---:|
| count | 0.0103 |
| scan | 0.0061 |
| scatter | 0.0228 |
| **total** | **0.0391** |

Gate 0.15 ms/heavy frame: **passes with 3.8x margin**. The generator was fixed
to include tile-boundary crossings when matching M0; it now emits **1.161
references per triangle**, not the old synthetic 1.685. The device's reference
count matches the host's expectation exactly (411,096), which checks the scan
and cursors.

For scale, this replaces `clip_all` 0.520 + `stage2` 0.187 + `stage3` 0.392 =
**1.099 ms/frame** of union-exclusive device time.

## Where that leaves the plan's arithmetic

Walk 0.0221 ms/scope at 2.02 drawing scopes = 0.045 ms/frame, plus bin 0.039 =
**0.084 ms/frame** for walk + bin, against the plan's ESTIMATED 0.25-0.5 for
walk + bin + compaction. The two new kernels are cheaper than the plan
budgeted, and they displace 1.099 ms/frame. Compaction is not yet measured.

This does **not** validate the frame-time estimate. Device time is not frame
time (`PERFORMANCE.md` §5.1), the M2 A/B is the only thing that settles it,
and the plan's own gate stands: **>= 1.0 ms/frame on the heavy band or stop.**

## Remaining pre-build gates

Run the split-walk and bin benchmarks on the B200 and require the plan's 1.6x
cross-device bound. An exact tile-reference-per-run census would replace the
triangle proxy, although the measured 99% share is too far above 60% for the
proxy error to change the current gate. Probe 6's `.local` baseline remains in
`DEAD_ENDS.md` 42.

These probe results do not overturn the later integrated results. The partial
M2 implementation on `redesign/runlevel` was bit-identical but slower than the
default renderer; `DEAD_ENDS.md` 45 records +1.57/+1.80 ms at 32 px and
+2.09/+2.51 ms at 16 px. Do not merge that prototype as a performance change.
