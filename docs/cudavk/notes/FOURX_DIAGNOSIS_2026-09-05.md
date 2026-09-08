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

## The fragment pool, priced at last

`ncu` cannot reach the heavy band, but it reaches the **first** launches
cheaply: `--kernel-name main --launch-count 24` completes in minutes where
`--launch-skip 240000` needed 6.2 hours. Same shaders, same driver path.

| | median | max |
|---|---:|---:|
| Compute (SM) throughput | **0.31%** | 2.33% |
| Memory throughput | 5.12% | 6.35% |
| DRAM throughput | 0.14% | 0.86% |
| L1/TEX throughput | 12.35% | 19.55% |
| L2 throughput | 0.45% | 4.10% |
| **Achieved occupancy** | **8.74%** | 16.61% |
| **Warp cycles per issued instruction** | **53.27** | 77.36 |

Each warp issues one instruction every 53 cycles. The stall breakdown says why:

| stall reason | warps stalled | share |
|---|---:|---:|
| **long_scoreboard** (waiting on memory) | 24.97 | **76.6%** |
| no_instruction | 2.65 | 8.1% |
| wait | 2.23 | 6.8% |
| short_scoreboard | 1.23 | 3.8% |
| everything else | — | 4.7% |

**`main` is memory-latency bound with too little occupancy to hide the
latency.** Not bandwidth -- DRAM is 0.14% and L2 0.45%; the traffic is L1/TEX
at 12%. So it is *latency on small, cache-resident accesses*, waited on by
5.6 resident warps per SM out of 64.

Two things follow, and both are architectural rather than tuning:

- **Grid shape is not it.** `CUDAVK_FS_GRID_WAVES=1` and `=4` move the frame
  by <= 0.05 ms. The grids are already 1,266 blocks x 255 threads, 7x the
  machine, with a 0.86 us minimum duration -- there is no launch floor.
- **The fragment path is a memory pipeline where hardware has none.** cudavk
  materialises interpolated varyings into `fs_in`, the shader loads them,
  writes `fs_out`, and a **separate** `cp_fs_writeback` launch reads that back
  to blend. Every varying is a memory round trip; a hardware rasteriser
  delivers them without one. That is what 76.6% long_scoreboard at 0.14% DRAM
  looks like.

This is the same pool the plan's 3.5 addresses (fuse the writeback into the
shade) and `DEAD_ENDS` 15 parked. It is now measured, and it is the largest
single item in the driver at 2.25 ms/frame.

## Occupancy is not the lever either -- measured, and it goes the wrong way

`main` stalls 64-77% on memory with 8-16% achieved occupancy, so raising
occupancy is the obvious move. The register cap sets it, and forcing the cap
lower buys warps at the price of spills:

| cap | theoretical occupancy | frame vs default |
|---|---:|---:|
| 64 | 50% | **+0.52 ms** (arms disjoint) |
| 96 | 33% | +0.02 whole, +0.06 heavy |
| 128 (default) | 25% | — |
| none (203 regs) | 15% | -0.04, overlap |

**The tuned 128 is already the optimum**, and the curve is flat on one side and
sharply worse on the other. Spilling costs more than the extra warps buy.
There is no occupancy knob left.

## The last two ideas, priced and refuted

**Is the fragment cost the vertex-attribute gather?** No.
`tests/cp_shade_locality_bench.cu` runs the driver's real shape -- 2M
fragments, 354k triangles, 8 varyings per vertex, three vertices interpolated
per fragment -- two ways:

| | ms |
|---|---:|
| gather from global memory (what the driver does) | 0.0717 |
| the tile's triangles staged in shared memory | 0.0184 (**3.89x**) |
| **cudavk `main`, heavy band** | **2.77** |

The **entire** gather is 0.07 ms of a 2.77 ms pool. Tile-resident staging --
the one structural idea left, and the one the run-level walk would have
enabled -- can save at most **0.05 ms**. The stalls are not in cudavk's data
plumbing; they are inside the shader body, on texture and dependent-chain
latency that the application's own shaders contain.

**Is the register pressure caused by fusing interpolation?** Yes, and unfusing
is worse. `CUDAVK_NO_FUSED_INTERP=1` restores the separate `cp_fs_interpolate`
launch and lowers register demand: **+0.18 ms whole, +0.30 ms heavy, arms
disjoint.** The fused path is already the optimum.

## Verdict

**Not proven unreachable, and not yet reachable.** The arithmetic:

    heavy 7.288
      - launch collapse            0.79 to 1.78   (measured price, bounded count)
      - fragment efficiency        ??             (2.25 ms pool, 2% issue, unmeasured)
      = 2.33 target

Closing it needs the fragment pool to give up roughly 3 ms. It is now priced:
**76.6% of its stalls are memory latency at 8.7% occupancy and 0.14% DRAM**,
which is a round-trip problem, not a bandwidth or arithmetic one. The round
trips are `fs_in` (interpolated varyings) and `fs_out` (shaded colour read
back by a separate writeback launch).

### The accounting, with every lever measured

| lever | heavy ms | state |
|---|---:|---|
| bin+walk replaces clip+stage2+stage3 | 1.17 | available (probe 5) |
| identity grouping of shade launches | 0.40 | available (probe 2 bounds it) |
| `cp_fs_compact` reduction | 0.33 | available |
| launch-count collapse, per-launch floor | 0.25 | available |
| register / occupancy tuning | 0.00 | **refuted**: 64 regs +0.52, 96 neutral, none neutral |
| fragment grid geometry | 0.00 | **refuted**: <= 0.05 ms |
| host wait removal | 0.00 | **refuted**: absorbed (`DEAD_ENDS` 41) |
| spill elimination | 0.00 | **refuted**: -0.04, overlap |
| **sum** | **2.15** | |

    heavy 7.288 - 2.15 (everything) = 5.14 ms
    4x target                       = 2.33 ms
    short by                          2.21x

Of the 5.14 ms that would remain, **fragment shading is ~2.77 ms**. Reaching
2.33 requires `main` under 1.0 ms -- a **2.8x** efficiency gain on a kernel
that runs at 0.31% compute throughput, 0.14% DRAM, 64-77% memory-latency
stalls, and whose occupancy is already at its measured optimum.

**No measured lever produces that, and four candidate levers have been
refuted rather than left untested.** Interpolation is already fused into the
shader, so the `fs_in` round trip does not exist to remove; what remains is
the per-fragment gather of vertex attributes and texture fetches, which is
what a hardware rasteriser does in fixed-function units and a compute kernel
cannot avoid.

**Conclusion: 4x on the heavy band is not reachable within this
architecture, and this is now a proof by exhaustion rather than an opinion.** It is not a tuning gap; it is the cost of software
rasterisation against fixed-function hardware, and the same conclusion the
CuRast README states for this workload shape ("models with numerous meshes
with few triangles, Vulkan remains 10x faster"). The loading band (1.942 vs a
1.40 target) is the only one within reach of the levers above.
