# Item 3 - stage3 stall attribution (JOB E) - RESULT

Shipping build `/home/alexzhukov/mesa/build-cudavk`, old capture, NCU
2025.1.1 unprivileged. Predictions and falsifiers were registered BEFORE any
profiler output existed: `/tmp/perf-audit/item3_predictions.md`.

## Exclusivity

Two windows, each with the 1 Hz sampler over the whole window and the start
gated on 8 consecutive idle samples.
* source pass: **100 samples, t+0..t+101 s, none foreign**
* wide pass: **216 samples, t+0..t+223 s, none foreign**
Neither overlaps the drain-tree rebuild window (t=…225..…251): job E ran
…099-…193, job E2 …290-…506. Both used the shipping build, not the drain tree.

## THE HEADLINE, and it corrects the briefing

**The 1.00 sectors/request figure is real for `cp_rasterize_stage3_abuf` and is
an ARTEFACT OF THE PROFILED WINDOW for `cp_rasterize_stage3`.**

900 consecutive launches (skip 200), global LOADS only:

| kernel | launches | degenerate | share of kernel time | **sectors/request, working launches** | overall |
|---|---:|---:|---:|---:|---:|
| `cp_rasterize_stage3` | 764 | **499 (65.3%)** | 7.3% | **3.087** | 2.987 |
| `cp_rasterize_stage3_abuf` | 136 | 31 (22.8%) | 1.7% | **1.000** | 1.000 |

"Degenerate" = 16 instructions per warp, i.e. the kernel read its tile counter,
found `num_tiles == 0` and exited. Those launches have exactly ONE global load
instruction, necessarily 1.00 sectors/request.

**Two-thirds of `cp_rasterize_stage3` launches do no work at all.** They are
7.3% of its time and 4.8% of its load requests, but they are 65% of its
LAUNCHES - so any profile that skips into a quiet stretch reports 1.00
sectors/request for a kernel whose working launches run at 3.09. My first
6-launch sample did exactly that, which is how I caught it. This is the same
hazard as the recipe's "two windows of one replay disagreed by 50x".

**So the briefed premise "wider requests, not coalescing" does not survive for
`cp_rasterize_stage3`.** Its working launches are already issuing ~3 sectors
per request.

## WHICH LOADS, for the kernel where 1.00 IS real (`_abuf`)

SASS attribution, 6 profiled launches aggregated (no `-lineinfo` in the NVRTC
build, so correlation is to SASS; the mapping to source below is by offset and
by the structure layouts in `cp_rast_types.h`).

**Every global LOAD in the kernel is at exactly 1.00 sectors/request - 24
distinct instructions, 9,144 requests, 9,144 sectors.** They divide into three
groups:

| load | requests | share | what it reads |
|---|---:|---:|---|
| `LDG.E R2, desc[UR18][R2.64]` | 6,144 | **67.2%** | the tile counter `*huge_counter`, one uniform 4-byte read per block, immediately followed by `R2UR UR5, R2` (move to uniform register) |
| `LDG.E R28.64+0x00 .. +0x50`, 22 instructions | 2,520 | **27.6%** | the 84-byte `struct cp_setup_cache_entry` read FIELD BY FIELD as 21x 32-bit + 1x `LDG.E.U8`, by `threadIdx.x == 0` only |
| `LDG.E R4` / `R32+0x4` | 480 | 5.2% | `huge_queue[tile_idx]`, the 8-byte `cp_tile_pair`, split into `.tri_id` and `.tile_x/.tile_y` |

**None of these is a per-lane strided walk.** All are single-lane or uniform.
A load issued by one active lane produces one sector per request BY
CONSTRUCTION, and that is the optimum for such a load, not a defect.

By contrast the per-lane addressed traffic in the same kernel is the OUTPUT
side, and it is already wide: `STG.E.64` at **2.56** sectors/request (1,719
requests) and `REDG.E.ADD` / `ATOMG.E.ADD` at 2.08 (2,039 requests) - 3
instructions, 12,934 requests, 26,520 sectors, **2.05 sectors/request**.

## WHERE THE STALL IS

`stall_long_sb` by the instruction that waits (NCU attributes to the consumer),
6 launches aggregated, 308 samples:

| consumer | samples | share | waiting on |
|---|---:|---:|---|
| `R2UR UR5, R2` | 159 | **51.6%** | the tile-counter load, 4 bytes, one per block |
| `MOV R4, R10` (before `STG.E.64`) | 115 | 37.3% | the shading/atomic result chain |
| `MOV R33, R4` | 20 | 6.5% | the `cp_tile_pair` queue read |
| top 3 | 294 | **95.5%** | |

Warp State for these launches: Long Scoreboard **47.75** of 76.87 warp cycles
per issued instruction = **62.1%** (briefed 56.3-70.7%; same phenomenon).

**More than half of the long-scoreboard stall in `_abuf` is every warp in the
block waiting on a single uniform 4-byte counter load.** That is a LATENCY
dependency at the top of the kernel, not a bandwidth or a coalescing problem.

## SCORING THE REGISTERED PREDICTIONS

* **P1 CONFIRMED.** The 1.00 sectors/request is the signature of single-lane
  and uniform loads, not of an AoS stride. Measured: 100% of the 1.00-sector
  load requests are uniform (67.2%), single-lane setup (27.6%) or uniform queue
  reads (5.2%).
* **P2 CONFIRMED.** Top 3 instructions carry **95.5%** of the long-scoreboard
  stall, against a predicted >50%.
* **P3 CONFIRMED.** `cp_setup_cache_entry` is 84 bytes with an 84-byte stride
  and is compiled to a run of 32-bit `LDG.E` plus one `LDG.E.U8` - no
  `LDG.128` anywhere. The layout forbids it.
* **X1 REFUTED.** There is no hot record type read per lane with a `sizeof`
  stride. "Split it and win" is not available.
* **X2 CONFIRMED - "inherent to the algorithm".** The loads are already minimal
  per request; the per-lane traffic is stores and atomics that already run at
  2.05-2.56 sectors/request. **A wider load is not the remedy because there is
  no narrow per-lane load to widen.**
* **X3 CONFIRMED, and it is the trap the briefing walked into.** A uniform
  4-byte broadcast at 1.00 sectors/request is ALREADY OPTIMAL. It is 67% of the
  load requests and 52% of the stall. Reporting it as a width defect would be
  an error.
* **X5 CONFIRMED - the two kernels do NOT share a mechanism.** `stage3` working
  launches are at 3.09 sectors/request; `_abuf` at 1.000. They must not be
  pooled, and the briefing's single "both read 1.00" sentence is what needs
  correcting.

## IS A WIDER LOAD LEGAL? Yes for one of them, and it is worth almost nothing

The only load where width is even available is the setup-cache entry: 22
instructions reading one 84-byte record from one lane. Padding
`cp_setup_cache_entry` to 96 bytes would allow `LDG.128` and cut 22
instructions to ~6. But it is 27.6% of the load requests and carries **0**
long-scoreboard samples in this sample; the fields ARE contiguous and are all
consumed (X4 does not bite), so the change is legal and nearly pointless.

The counter load cannot be widened - it is 4 bytes and there is nothing beside
it to fetch. Structure-of-arrays on the producing stage does not apply: the
consumer is one lane reading one record, so SoA would turn 1 request into 21
requests to different cache lines and make it worse.

## SIZING, THROUGH EXCLUSIVITY, NOT KERNEL TIME

The raster chain at 2x is **0.752 ms/frame** of union busy removed and at
infinite speed **2.044**. Nothing found here removes more than a small slice of
the chain's latency:
* the setup-cache widening touches 0% of the measured stall - **~0**;
* removing the counter-load dependency entirely (e.g. passing `num_tiles` in
  the launch arguments, or making the empty launches not happen) would remove
  at most the 52% of `_abuf`'s stall that is that one load. `_abuf` is 6.6 ms
  of kernel time over 136 launches in this window against the chain's total, so
  even at 100% conversion this is a fraction of the 0.752 at 2x, not of it.

**The real lead this run found is not a memory-width lead at all: 65.3% of
`cp_rasterize_stage3` launches do no work.** That is a launch-count question -
the same currency as item 4 and PDL, at ~0.782 us per launch removed - and it
is measurable with the counters that already exist.

## Raw

/tmp/perf-audit/jobE/ (source pass, 2 ncu-rep) and /tmp/perf-audit/jobE2/
(900-launch wide pass, wide.csv), plus watch.log and progress.log in each.
