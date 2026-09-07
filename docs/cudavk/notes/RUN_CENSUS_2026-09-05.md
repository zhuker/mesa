# The redesign probes (P0 and P1)

All six pre-build gates of `REDESIGN_PLAN_2026-09-05.md` §7. **All pass.**

| probe | gate | favorite3 | favorite2 | |
|---|---|---|---|---|
| 1 run census | admitted >= 60% of references | **99.1%** | **99.2%** | pass |
| 2 identities/run | p99 <= 64 | **18** | **13** | pass |
| 3 stencil | none in candidate runs | **0** | **0** | pass |
| 4 split walk | p99 chunk <= 30 us, merge <= 10% | **10.0 us, 1.3%** | — | pass |
| 5 bin pass | <= 0.15 ms/heavy frame | **0.055 ms** | — | pass |
| 6 `.local` baseline | recorded | `DEAD_ENDS` 42 | | done |

## Probe 1: the run census

Run at HEAD on both captures with
`CUDAVK_RUN_CENSUS=1`. Host-side only: it classifies every draw the way the
run-level renderer would and counts the runs it would form. It changes no
rendering decision.

**The gate was: stop if admitted runs hold under 60% of a frame's references.
They hold 99.1% and 99.2%. The probe passes, and not marginally.**

Triangles stand in for tile references (M0 measured 1.16 references per
triangle); the gate tests a share, so the factor cancels.

## favorite3 (2,083 real frames)

| class | runs | runs/frame | draws/run | longest | triangles |
|---|---:|---:|---:|---:|---:|
| opaque | 2,153 | 1.03 | 181.3 | 244 | **90.8%** |
| depth-only | 2,876 | 1.38 | 13.0 | 34 | 6.0% |
| blended | 31,786 | 15.3 | 4.2 | 128 | 2.2% |
| fallthrough | 4,406 | 2.1 | 1.5 | 4 | 0.9% |
| **admitted** | **36,815** | **17.7** | | | **99.1%** |

## favorite2 (2,078 real frames)

| class | runs | runs/frame | draws/run | longest | triangles |
|---|---:|---:|---:|---:|---:|
| opaque | 2,098 | 1.01 | 164.0 | 216 | **95.2%** |
| depth-only | 755 | 0.36 | 10.4 | 23 | 1.7% |
| blended | 31,438 | 15.1 | 3.1 | 19 | 2.3% |
| fallthrough | 3,591 | 1.7 | 1.6 | 3 | 0.8% |
| **admitted** | **34,291** | **16.5** | | | **99.2%** |

## What this says about the plan

- **One opaque run per frame carries ~91-95% of the triangles**, at 164-181
  draws each, and 2,084 of favorite3's 2,153 opaque runs are in the 128+
  bucket. Today that run is rendered as ~110 separate per-batch chains. This
  is the redesign's central claim, and it is now measured rather than assumed.
- **The depth-only relaxation is real and it is not large.** favorite3 has
  1.38 depth-only runs per frame holding 18.0 draws -- the shadow scope the
  plan describes, at 6.0% of triangles. Collapsing 18 single-draw chains and
  their 18 visibility clears into one run is worth doing, but it is a tail
  item, and on favorite2 it is 1.7%.
- **Blended is where the runs are short**: 15 runs per frame of 3-4 draws, and
  23,000 of ~31,500 blended runs are a single draw. The plan's M3 stage
  should expect little from run-level batching here; its win must come from
  the walk emitting sorted lists directly, not from longer runs.
- **~18 runs per frame** against ~700 launches today. At the plan's 8-30
  launches per run that is 140-530, bracketing its 100-250 estimate but not
  confirming it.

## A correction made while running it

The first version hooked `cp_batch_record_packet`, which sees only draws the
batcher accepted. It reported 251 draws/frame against the plan's ~300 and
**zero depth-only runs** -- because a depth-only draw fails
`cpvk_batch_structural` and takes the direct path, so the census was blind to
exactly the draws the relaxation targets. The hook moved to
`cpvk_prepare_draw`'s caller, before the batching decision. Draws/frame went
251 -> 273 and the depth-only class appeared.

**A census of a decision must be taken upstream of that decision.**

## Probe 2: fragment-shader identities per run

The identity is the `(binary, resolved exec)` pair `cp_fs_args_prepare()`
settles on; that pointer already folds in exec mode, sampler variant and tune
alternate, so it is the plan's four axes and it is exactly what one shade
launch would cover. **Gate: stop if p99 exceeds 64.**

| class | favorite3 mean / p50 / p90 / p99 / max | favorite2 |
|---|---|---|
| opaque | 12.75 / 13 / 17 / **18** / 18 | 11.00 / 11 / 13 / **13** / 13 |
| depth-only | 1.00 / 1 / 1 / 1 / 2 | 1.00 / 1 / 1 / 1 / 1 |
| blended | 1.41 / 1 / 3 / 4 / 5 | 1.41 / 1 / 3 / 3 / 4 |
| fallthrough | 1.00 / 1 / 1 / 1 / 1 | 1.00 / 1 / 1 / 1 / 1 |

No run exceeded the 128 tracked identities. **Passes with a 3.5x margin.**

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
| unsplit, one block per tile | 3,600 | 0.22 us | 23.88 us | 34.61 us | 0.0435 ms |
| **chunk 512** | 3,968 | 2.18 | **10.03** | 11.28 | **0.0221 ms** |
| chunk 1024 | 3,737 | 1.27 | 16.32 | 19.15 | 0.0281 ms |
| chunk 2048 | 3,644 | 0.24 | 26.12 | 31.47 | 0.0395 ms |

The `atomicMin` merge costs **0.8-1.3%**, an order of magnitude inside its 10%
gate. Splitting is order-independent, so it changes no result.

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
| count | 0.0144 |
| scan | 0.0061 |
| scatter | 0.0346 |
| **total** | **0.0551** |

Gate 0.15 ms/heavy frame: **passes with 2.7x margin**, and the generator emits
**1.685 references per triangle** against M0's measured 1.16, so the real
input is lighter than the one measured. The device's reference count matches
the host's expectation exactly (596,415), which is the correctness check on
the scan and the cursors.

For scale, this replaces `clip_all` 0.520 + `stage2` 0.187 + `stage3` 0.392 =
**1.099 ms/frame** of union-exclusive device time.

## Where that leaves the plan's arithmetic

Walk 0.0221 ms/scope at 2.02 drawing scopes = 0.045 ms/frame, plus bin 0.055 =
**0.10 ms/frame** for walk + bin, against the plan's ESTIMATED 0.25-0.5 for
walk + bin + compaction. The two new kernels are cheaper than the plan
budgeted, and they displace 1.099 ms/frame. Compaction is not yet measured.

This does **not** validate the frame-time estimate. Device time is not frame
time (`PERFORMANCE.md` §5.1), the M2 A/B is the only thing that settles it,
and the plan's own gate stands: **>= 1.0 ms/frame on the heavy band or stop.**

## Next gates

Probe 2 (four-axis identity census per run, stop if p99 > 64), probe 3
(stencil and alpha-to-coverage inside candidate runs), probes 4-5 (the
split-walk and bin-pass microbenchmarks). Probe 6, the `.local` baseline of
every fused VS, is already recorded in `DEAD_ENDS.md` 42.
