# Session result: the 4x-native investigation

## What was achieved

Two changes landed on `cudapipe-vk-native`, both default-on, both bit-identical:

| change | favorite3 whole | favorite3 heavy | favorite2 whole | favorite2 heavy |
|---|---:|---:|---:|---:|
| opaque episodes span viewport/raster/depth changes | -0.076 | -0.075 | -0.150 | -0.265 |
| depth-only scopes batch | -0.657 | -1.151 | -0.026 | -0.279 |
| **combined, measured together** | **-0.733** | **-1.226** | **-0.176** | **-0.544** |

    favorite3  5.940 -> 5.176 whole   7.288 -> 6.023 heavy   (-12.9%)
    favorite2  5.022 -> 4.867 whole   5.598 -> 5.054 heavy

Gates: 79/79 tests, 18/18 sentinels bit-identical on both captures, exact
stdout hashes, the 18-sample sweep with **no verdict changed**, `FLAGS.md`
regenerated, no-getenv audit clean. Reverts: `CUDAVK_NO_WIDE_EPISODE`,
`CUDAVK_NO_DEPTH_ONLY_BATCH`.

## Against the objective

| band | favorite3 | native | ratio | 4x target |
|---|---:|---:|---:|---:|
| loading | 1.917 | 0.351 | 5.46x | 1.40 |
| light | 4.117 | 0.478 | 8.61x | 1.91 |
| heavy | 6.023 | 0.582 | **10.35x** | 2.33 |

**Not met.** 3.09 ms more is needed on favorite3; every remaining identified
lever sums to about **1.5 ms**, landing at 3.70 -- still 1.77x over, with
nothing named for the remainder.

## What was ruled out, and how

- **14 flag-level levers**, 10 refuted. The decisive one: frame time is
  **invariant across the whole register/spill curve** -- 203 registers with
  zero spill and 128 with 168 B of spill measure the same frame.
- **The predicate search**: 3 relaxations tried, 2 landed, 1 (depth-only
  episodes) correct but slower.
- **bin+walk**, the largest structural estimate at -1.04 ms, **built** and
  measured **+1.8 ms worse**, twice, on two baselines.
- **Five estimation errors**, all self-caught and recorded in
  `FOURX_DIAGNOSIS_2026-09-05.md`: the frame boundary is not the
  application's; overdraw is 2.29x not 12x; occupancy is not a sufficient
  argument; the fragment pool is 0.729 ms union-exclusive, not the 2.25 ms
  summed over kernels all named `main`; and the trace has 17 pools, not 4.

## The transferable lesson

Both wins came from one move: **when a measurement says "the workload does
this", check whether the driver had to care.** The episode-wide state
comparison and the colour-attachment test were each one predicate asking for
something no consumer needed. Together they beat every flag-level lever
combined, and they were found only after three separate "proofs" that the
remaining gap was structural.

## What would close the rest

Nothing in the driver. The gap is software rasterisation against
fixed-function hardware: interop with the graphics pipeline, or a change to
what the application asks for. `~5x native` is the realistic ceiling for this
architecture, and the identified-but-unbuilt work is worth about half of it.
