# Probe 1: the run census

`REDESIGN_PLAN_2026-09-05.md` §7 probe 1, run at HEAD on both captures with
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

## Next gates

Probe 2 (four-axis identity census per run, stop if p99 > 64), probe 3
(stencil and alpha-to-coverage inside candidate runs), probes 4-5 (the
split-walk and bin-pass microbenchmarks). Probe 6, the `.local` baseline of
every fused VS, is already recorded in `DEAD_ENDS.md` 42.
