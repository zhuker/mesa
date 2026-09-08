# What stands between cudavk and 4x the native driver

Goal: every band within 4x of NVIDIA's own driver on the same GPU. Measured
2026-09-05 on the RTX 5090, favorite3 unless stated.

## Where we are, per band

| band | cudavk | native | ratio | 4x target |
|---|---:|---:|---:|---:|
| loading | 1.942 | 0.351 | 5.53x | 1.40 |
| light | 4.171 | 0.478 | 8.73x | 1.91 |
| heavy | 7.288 | 0.582 | **12.52x** | **2.33** |

favorite2 is the same shape: 5.50 / 9.04 / 10.81.

**The gap grows with scene complexity.** Across the bands the native driver
moves 0.35 -> 0.58 ms (1.66x); cudavk moves 1.94 -> 7.29 (3.75x). Per unit of
extra scene, cudavk pays 23x what the hardware pays: +5.35 ms against +0.23.

So 5.53x on loading is cudavk's fixed overhead -- identical on both captures --
and everything above it is marginal cost the hardware absorbs for free.

## What the limiter is, measured three independent ways

**1. Device counters (`nsys --gpu-metrics-devices`, no CUPTI, no replay).**

| | p50 | p90 |
|---|---:|---:|
| GR Active | 81% | 100% |
| SMs Active | **17%** | 83% |
| **SM Issue** | **2%** | 12% |
| compute warps in flight | 5% | 33% |
| DRAM read / write | 1% / 1% | 8% / 25% |

Busy 81% of the time, issuing on **2% of cycles**, a sixth of the SMs active,
DRAM at 1%. **Not compute bound, not memory bound.** This is `CLAUDE.md`'s
"busy is not working", now measured on the capture rather than a sample.

**2. Frame-time A/B on batch count.** `CUDAVK_BATCH_MAX=2` doubles batches per
frame (86.4 -> 173.6) without changing the work: +1.96 ms whole, +2.36 heavy,
arms disjoint. **A batch costs 22.5 us whole / 27.0 us heavy** whatever it
draws.

**3. The native driver is flat across bands** while cudavk is not, which is the
same fact from the outside.

## What is NOT the limiter -- refuted

- **Register pressure and spilling.** The fused fragment shaders compile to 203
  registers, are capped to 128, and all 69 then spill 168-400 B/thread
  (`DEAD_ENDS` 42). Disabling the cap (`CUDAVK_NO_REGCAP=1`) moved the frame
  **-0.04 ms, arms overlapping**. The spill costs nothing.
- **The tile walk.** Rewritten twice and now 3.6x the serial walk it replaced;
  the tiled path is still +1.95 ms because it covers 21% of batches
  (`M2_PROGRESS_2026-09-05.md`).
- **Host waits.** `DEAD_ENDS` 41: injecting 0.449 ms/frame before the largest
  wait does not move the frame.

## Why batches break, and what collapsing them is worth

`CUDAVK_PLAN_STATS` attributes every break:

| cause | count | share |
|---|---:|---:|
| **vertex shader** | 95,448 | **69%** |
| **fragment shader** | 38,263 | **28%** |
| draw not batchable | 2,894 | 2% |
| triangle cap / rasterizer | 412 | — |

Batches break on **shader identity**, not on viewport, blend or scissor. That
is what run-level identity grouping targets, and it bounds the prize:

| | batches/frame |
|---|---:|
| today | 86.4 |
| grouped by identity (probe 1 + 2 measured) | 57.3 |
| with per-batch raster stages replaced by one bin+walk per run | ~20 |

**0.65 to 1.50 ms/frame**, i.e. favorite3 heavy 7.29 -> 5.5-6.5. **Launch
collapse alone does not reach 2.33.** An earlier 1.85 ms estimate in this
campaign was too high: it counted runs, not identity groups per run.

## Where the rest would have to come from

Fragment shading (`main`) is **2.25 ms/frame over 197 launches of 11.42 us** --
the largest single item, larger than clip, stage2, stage3 and compaction
together. The native driver executes **the same compiled shaders on the same
GPU, plus all the vertex, raster and ROP work, in 0.582 ms**. So `main` alone
is **3.9x an entire native frame**, and that is not irreducible shader work; it
is execution inefficiency of the kind the counters describe -- 2% issue.

That is the only remaining pool big enough to close the gap, and it has never
been measured at the kernel level on this capture. `ncu` cannot reach it: it
replays every launch, so at the observed 69 launches/s, arriving at the heavy
band's ~1.54 M launches takes **6.2 hours** and exceeds any sane timeout. A
different instrument is required -- profiling a representative shader in
isolation, or a synthetic harness that reproduces the launch geometry.

## Verdict

**Not proven unreachable, and not yet reachable.** The arithmetic:

    heavy 7.288
      - launch collapse            0.79 to 1.78   (measured price, bounded count)
      - fragment efficiency        ??             (2.25 ms pool, 2% issue, unmeasured)
      = 2.33 target

Closing it needs the fragment pool to give up roughly 3 ms, and nothing yet
says whether it can. **That measurement is the next step**, and it is the last
large unpriced quantity in this driver.
