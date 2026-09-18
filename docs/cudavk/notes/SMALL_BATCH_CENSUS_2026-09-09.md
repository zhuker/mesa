# Small-batch fused kernel: refuted at census (2026-09-09)

**Outcome: not built.** The idea was a single kernel per small batch — vertex
shader body, clip, setup, in-place raster into the visibility buffer and
compaction in one launch, followed by the fragment shader with fused writeback
— on the reading that ~68 small batches per frame carry most of the frame's
launches and therefore most of its serial launch gaps. The plan made a census
the first step, with a stop rule: **if the small batches hold under ~1.5 ms of
main-stream wall, stop.** They hold **0.67-0.69 ms traced** on the current
build, of which the part a fusion could collect is about 0.5 ms traced and
less untraced. The prototype was not started; nothing was measured on a
harness. `src/cudavk/tests/cp_chain_census.py` reproduces every number here.

## Method

Three RTX 5090 nsys exports of the default renderer, all CUPTI-traced (so the
gaps are inflated by the per-launch host cost `CLAUDE.md` warns about; the
figures are populations and shares, not frame times):

| trace | build | window | frames | traced ms/frame |
|---|---|---|---:|---:|
| `/tmp/kb.sqlite` | 2026-09-08, post-wins | whole favorite3 replay | 3,442 | 4.25 |
| `/tmp/nsys-runclip-control.sqlite` | 2026-09-08 | favorite3 heavy window | 124 | 7.26 |
| `/tmp/rtx-head.sqlite` | 2026-09-03 HEAD | favorite3 heavy band | 924 | 6.82 |

A frame is two submits, each ending in the 3,686,400-byte readback copy, so
frames are every other readback. A *chain* is one vertex shader (a `main`
whose next same-stream kernel is a clip-family kernel) through to the next on
its stream: VS, `cp_clip_rast_fused`, stage 2, stage 3, `cp_fs_compact`, FS,
`cp_fs_writeback`, plus any peel or A-buffer tail. Chains are cut at frame
ends. "Small" is a VS grid of at most 4,096 vertices. Wall is first kernel
start to last kernel end; busy is the sum of kernel durations.

## The population

Main-stream chains, per frame:

| trace | class | chains | launches | wall ms | busy ms | med launches | med verts / tris | med wall |
|---|---|---:|---:|---:|---:|---:|---|---:|
| kb (4.25) | shaded ≤4096 | 6.22 | 59.9 | **0.671** | 0.470 | 7 | 256 / 64 | 54 µs |
| | shaded >4096 | 2.47 | 45.5 | 0.720 | 0.482 | 13 | 7,168 / 2,368 | 164 µs |
| | blended ≤4096 | 2.30 | 48.1 | 1.551 | 0.400 | 18 | 256 / 64 | 302 µs |
| | blended >4096 | 0.81 | 58.5 | 0.842 | 0.317 | 68 | 7,424 / 2,432 | 970 µs |
| control (7.26) | shaded ≤4096 | 6.19 | 65.3 | **0.686** | 0.503 | 8 | 256 / 64 | 58 µs |
| | shaded >4096 | 5.69 | 94.3 | 1.183 | 0.908 | 12 | 14,080 / 4,736 | 138 µs |
| | blended ≤4096 | 2.63 | 68.3 | 2.809 | 0.717 | 20 | 256 / 64 | 1,154 µs |
| | blended >4096 | 1.62 | 132.6 | 2.134 | 0.880 | 74 | 7,424 / 2,432 | 1,242 µs |
| rtx-head (6.82) | shaded ≤4096 | 7.77 | 240.2 | 2.046 | 1.344 | 8 | 256 / 64 | 60 µs |
| | shaded >4096 | 5.34 | 90.2 | 1.585 | 1.024 | 8 | 9,728 / 3,264 | 135 µs |
| | blended ≤4096 | 2.00 | 36.3 | 1.627 | 0.324 | 20 | 256 / 64 | 857 µs |
| | blended >4096 | 1.33 | 82.8 | 1.185 | 0.381 | 62 | 7,424 / 2,432 | 787 µs |

Side-stream chains (the episode fan-out; these overlap each other and the main
stream, so their wall is not frame time):

| trace | class | chains | launches | busy ms | med launches | med verts | med wall |
|---|---|---:|---:|---:|---:|---:|---:|
| kb | raster-only ≤4096 | 3.12 | 12.8 | 0.433 | 4 | 1,024 | 179 µs |
| | raster-only >4096 | 6.83 | 28.9 | 1.297 | 4 | 28,416 | 241 µs |
| | blended ≤4096 | 18.30 | 78.0 | 0.750 | 4 | 256 | 34 µs |
| control | raster-only ≤4096 | 5.35 | 21.8 | 0.762 | 4 | 1,024 | 196 µs |
| | raster-only >4096 | 13.32 | 56.9 | 2.833 | 4 | 30,592 | 280 µs |
| | blended ≤4096 | 48.35 | 200.1 | 1.827 | 4 | 256 | 34 µs |

So the "~68 small batches" of the 2026-09-02 analysis are real, but most of
them are the **side-stream** blended appends (18-48 per frame, four launches
each, ~34 µs wall each, wall almost equal to busy) whose latency the fan-out
already hides. On the main stream the small *shaded* population is 6-8 chains
and 60-65 launches per frame on the current build.

The 2026-09-03 trace shows 240 launches per frame in that row against 60-65
now; the 2026-09-05..09 changes moved that work. The gate is applied to the
two current traces.

## The gate arithmetic

What a fused kernel can collect from a chain is its launch gaps (wall − busy)
plus the launch shells inside busy (each of the seven kernels in a 54 µs chain
is a ~5-10 µs latency shell; the real work is a fraction of it). For the
current build's small shaded main chains:

| | kb | control |
|---|---:|---:|
| wall | 0.671 | 0.686 |
| busy | 0.470 | 0.503 |
| gaps (wall − busy) | 0.201 | 0.183 |
| shells, generous (60% of busy) | 0.282 | 0.302 |
| **collectable, traced** | **~0.48** | **~0.49** |

Both are a third of the 1.5 ms stop rule, and traced. Untraced the gaps are
smaller: the measured price of one whole extra batch on the real frame is
22.5 µs whole / 27.0 µs heavy (`CUDAVK_BATCH_MAX=2`,
`FOURX_DIAGNOSIS_2026-09-05.md` §2), so a fused kernel that removed the entire
fixed cost of 6-8 small chains and cost nothing itself would be worth at most
**0.14-0.22 ms/frame** — and it would cost one launch and the work.

The small **blended** main chains look larger (1.55 / 2.81 ms wall) but their
wall is not launch gaps: busy is 0.40 / 0.72 and the rest is the main stream
waiting. The single largest edge in every trace is `memset<4K >
cp_abuf_scan_reduce` (0.39 / 0.74 / 0.51 ms per frame) — the blended episode's
count phase waiting on the side-stream appends — and the peel loop's host
waits. Fusing the four-launch side chains would shorten each by perhaps 10 µs
of its 34, spread across eight streams, and cannot move the join. A per-pixel
shared-memory sort replacing the eleven-kernel A-buffer tail would attack the
tail's ~0.3 ms of busy, which is the same population `DEAD_ENDS` 45 measured
as already overlapped.

## What the main-stream wall is instead

The same script decomposes the main stream per frame (traced):

| ms/frame | kb (4.25) | control (7.26) | rtx-head (6.82) |
|---|---:|---:|---:|
| kernel busy | 1.683 | 3.009 | 3.074 |
| gaps >200 µs (frame boundary, host waits) | 0.992 | 1.876 | 1.296 |
| gaps ≤200 µs, everything idle (launch gaps) | 0.685 | 1.121 | 1.120 |
| gaps ≤200 µs, side streams busy (joins) | 0.535 | 0.614 | 0.650 |
| memcpy busy | 0.258 | 0.422 | 0.378 |
| memset busy | 0.094 | 0.217 | 0.304 |
| main-stream ops (kernels / memcpy / memset) | 212 / 65 / 52 | 360 / 107 / 98 | 450 / 89 / 150 |
| launch gaps counted | 313 | 536 | 606 |

This is the picture `PERF_ANALYSIS_2026-09-02.md` §1 and `DEAD_ENDS` 44-45,
52 already record, from a different cut: kernel busy is 40-45% of the traced
frame; the launch-gap pool is ~1.1 ms traced across 500-600 gaps of ~2 µs,
spread over every operation on the stream rather than concentrated in the
small batches; the joins are the fan-out's exposed remainder, which entry 45
measured at its optimum; the >200 µs gaps are the application's frame
boundary and the peel-loop host waits. No population in this table is both
under the driver's control and large enough to clear the 0.5 ms admission
gate by itself, which is what entry 52 concluded from the summed-duration
side.

## Ledger

Recorded as `DEAD_ENDS.md` 53 on 2026-09-09.
