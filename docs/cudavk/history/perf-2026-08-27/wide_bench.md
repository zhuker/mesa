# wide_bench: what widening a launch actually buys on this hardware

RTX 5090, 170 SMs, 48 warps/SM, 24 blocks/SM, 65536 registers/SM. Driver
580.173.02, CUDA 12.8 (nvcc V12.8.93), `-arch=native` (sm_120). Old capture
`headless_streamer_20260814T155742.gfxr`. Shipping driver at
`/home/alexzhukov/mesa` HEAD `e2fea470d04`; the working tree was NOT modified.
Instrumented copy built in `/tmp/perf-audit/wide/mesa`.

## 0. Exclusivity ledger

The user's rule is binding: a run with a foreign tenant on the card is INVALID,
not noisy. A point-in-time `nvidia-smi` does not satisfy it, because the libvpx
`vpxenc` batch on this machine issues a new ~1 GB process about once a second.

Every run below was (a) gated on N consecutive idle samples before it started,
(b) stamped with its own start/end epoch, and (c) verified AFTERWARDS against a
1 Hz sampler (`/tmp/perf-audit/gate/gpu_watch.sh` -> `wide/gpu_watch.log`),
PID-resolving `[No data]` entries and requiring EVERY process on EVERY sample in
the window to be mine. A run that failed was discarded and re-run, never
reported. Cost of that policy: `fine_sweep` needed 32 attempts and `shapes_ncu`
26 before a window came back clean.

| run | window (epoch) | samples | foreign |
|---|---|---:|---:|
| injection sweep, 18 replays | 1787788267..1787789446 | 621 | **0** |
| driver_ncu | 1787789481..1787789580 | 99 | **0** |
| timing_rerun | 1787790032..1787790042 | 13 | **0** |
| fine_sweep | 1787795031..1787795041 | 13 | **0** |
| shapes_ncu | 1787798118..1787798141 | 26 | **0** |
| phase-1 timing sweep (first) | 1787786085..1787786095 | 10 | **0** |

Two runs I originally reported are WITHDRAWN as contaminated: the first NCU
shape sweep and the first driver NCU pass. Both were re-run clean and only the
clean numbers appear below. My first verifier also had a bug - a sampler line
lists several processes, and it asked "does this line mention me" instead of
"is every entry mine". Fixed and re-applied to everything.

Reproducibility: the coarse shape sweep was run twice, in separate clean
windows. The 20 points agree to within **0.64%** (max), 13 of 20 to 0.00%.

## 1. The shape curve: total work held constant

`/tmp/perf-audit/wide/wide_bench.cu`. One **item** = one block = a fixed
4096-pixel loop: three incremental edge functions plus a depth plane (6 FMA per
pixel), one load from a 32 KiB table whose NEXT index is derived from the value
just loaded (a real dependent chain, so latency is not free), and a
depth-test-gated accumulate. **38 registers/thread, 0 shared memory**, so its
theoretical occupancy is 100% at every block size and nothing about the
benchmark caps width. The curve is the hardware's, not the kernel's.

Work per item is FIXED. Total is always 60000 items, issued P per launch in
60000/P launches on one stream, CUDA-event timed, best of 3.

### 1a. ns per item (lower is better; every cell is the same total work)

| threads/item | P=1 | P=12 | P=100 | P=1000 | P=10000 |
|---|---:|---:|---:|---:|---:|
| 64 | 10244.76 | 879.67 | 120.30 | 8.19 | 3.07 |
| 128 | 6146.94 | 512.24 | 61.53 | 8.19 | 3.11 |
| 256 | 4098.00 | 341.50 | 41.50 | 8.19 | 3.28 |
| 512 | 4094.01 | 341.18 | 40.96 | 8.19 | 3.48 |

### 1b. Speedup over the same work issued one item per launch

| threads/item | P=1 | P=12 | P=100 | P=1000 | P=10000 |
|---|---:|---:|---:|---:|---:|
| 64 | 1.0x | 11.6x | 85.2x | 1250.9x | 3337.1x |
| 128 | 1.0x | 12.0x | 99.9x | 750.5x | 1976.5x |
| 256 | 1.0x | 12.0x | 98.7x | 500.4x | 1249.4x |
| 512 | 1.0x | 12.0x | 100.0x | 499.9x | 1176.4x |

**Today's shape (64 threads/item, the driver's median queue of 12) is
879.67 ns/item. The same work in one wide launch is
3.07 ns/item. That is 287x.**

### 1c. NCU on ONE launch of each shape (clean window 1787798118..1787798141)

| thr | items/launch | waves/SM | theo occ | achieved occ | warps in flight | SM ISSUE %elapsed | SM ISSUE %active | SM-active MAX / AVG / MIN |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 1 | 0.00 | 100% | 4.01% | 1.92 | 0.01 | 1.71 | 51428 / 303 / 0 |
| 64 | 12 | 0.00 | 100% | 4.11% | 1.97 | 0.19 | 3.10 | 30052 / 2008 / 0 |
| 64 | 100 | 0.02 | 100% | 4.12% | 1.98 | 1.67 | 3.39 | 28136 / 15315 / 0 |
| 64 | 1000 | 0.25 | 100% | 23.76% | 11.40 | 24.24 | 29.23 | 18400 / 17741 / 17068 |
| 64 | 10000 | 2.45 | 100% | 89.15% | 42.79 | 58.31 | 62.63 | 86169 / 82799 / 79307 |
| 128 | 1 | 0.00 | 100% | 7.94% | 3.81 | 0.02 | 3.36 | 27264 / 160 / 0 |
| 128 | 12 | 0.01 | 100% | 8.12% | 3.90 | 0.35 | 6.04 | 15569 / 1069 / 0 |
| 128 | 100 | 0.05 | 100% | 8.13% | 3.90 | 2.81 | 6.17 | 16110 / 8716 / 0 |
| 128 | 1000 | 0.49 | 100% | 46.91% | 22.52 | 27.56 | 33.65 | 16439 / 15993 / 14916 |
| 128 | 10000 | 4.90 | 100% | 91.77% | 44.05 | 60.78 | 63.84 | 85744 / 84309 / 82799 |
| 256 | 1 | 0.00 | 100% | 15.78% | 7.57 | 0.03 | 6.41 | 15317 / 90 / 0 |
| 256 | 12 | 0.01 | 100% | 15.95% | 7.66 | 0.55 | 10.58 | 9547 / 655 / 0 |
| 256 | 100 | 0.10 | 100% | 16.06% | 7.71 | 4.26 | 10.03 | 10586 / 5757 / 0 |
| 256 | 1000 | 0.98 | 100% | 90.35% | 43.37 | 29.35 | 36.25 | 16637 / 15935 / 14651 |
| 256 | 10000 | 9.80 | 100% | 89.60% | 43.01 | 62.37 | 65.33 | 90076 / 88423 / 86288 |
| 512 | 1 | 0.00 | 100% | 31.45% | 15.10 | 0.05 | 12.09 | 9231 / 54 / 0 |
| 512 | 12 | 0.02 | 100% | 31.69% | 15.21 | 0.83 | 17.50 | 6490 / 450 / 0 |
| 512 | 100 | 0.20 | 100% | 32.05% | 15.38 | 5.88 | 14.12 | 8229 / 4649 / 0 |
| 512 | 1000 | 1.96 | 100% | 90.66% | 43.52 | 31.62 | 38.50 | 17801 / 17050 / 15743 |
| 512 | 10000 | 19.61 | 100% | 84.27% | 40.45 | 65.46 | 68.31 | 97657 / 96106 / 94174 |

**Read the MIN column.** At 1, 12 and 100 items per launch, SM-active MIN is
ZERO: some SMs never turn on for the whole launch. It is non-zero only from
P=1000.

**The driver's 3% is reproduced exactly.** The driver reads SM ISSUE 3% with 5
warps in flight. This benchmark at 64 threads x 12-100 items reads 0.19-1.67%
with 1.97 warps; at 256 threads x 100 items, 4.26% with 7.71 warps. The driver
sits inside that band. **3% issue is the signature of a grid smaller than the
machine, not of bad code.**

## 1d. THE CROSSOVER (the deliverable)

Fine sweep across the fill point, 15 points, clean window 1787795031..1787795041.
One full wave at 64 threads is 170 SM x 24 blocks = **4080 blocks**.

| items/launch | blocks per SM | us per LAUNCH (T=64) | ns per item (T=64) | speedup vs P=12 |
|---:|---:|---:|---:|---:|
| 1 | 0.01 | 10.24 | 10244.76 | 0.1x |
| 3 | 0.02 | 10.25 | 3415.51 | 0.3x |
| 12 | 0.07 | 10.55 | 879.50 | 1.0x |
| 30 | 0.18 | 11.76 | 392.09 | 2.2x |
| 60 | 0.35 | 11.89 | 198.16 | 4.4x |
| 150 | 0.88 | 12.27 | 81.83 | 10.7x |
| 300 | 1.76 | 10.52 | 35.05 | 25.1x |
| 600 | 3.53 | 8.19 | 13.65 | 64.4x |
| 1200 | 7.06 | 8.21 | 6.84 | 128.6x |
| 2400 | 14.12 | 12.21 | 5.09 | 172.8x |
| 3000 | 17.65 | 12.28 | 4.09 | 215.0x |
| 6000 | 35.29 | 20.59 | 3.43 | 256.4x |
| 12000 | 70.59 | 36.84 | 3.07 | 286.5x |
| 30000 | 176.47 | 84.93 | 2.83 | 310.8x |
| 60000 | 352.94 | 164.16 | 2.74 | 321.0x |

**1. Width is FREE up to about 7 blocks per SM.** Per-LAUNCH time is flat from
P=1 to P=1200 - 10.24, 10.25, 10.55, 11.76, 11.89, 12.27, 10.52, 8.19, 8.21 us -
while the work in the launch grows 1200-fold. A launch's duration is the LATENCY
OF ONE ITEM; the other 1199 ride along at no cost. Speedup therefore tracks P
almost exactly over that whole range (128.6x at P=1200).

**2. The knee is between P=1200 and P=2400**, that is 0.29 to 0.59 waves. Past
it, per-launch time rises roughly with P and the gain per extra block falls.

**3. Saturation is around 2.5-3 waves.** ns/item bottoms out near 2.7-3.1 and
full-machine SM issue on this hardware is 59-65%, not 100%.

**4. The launch floor is 2.047 us**, back-to-back empty kernels, identical for
1 / 12 / 100 / 1000 blocks (2.047 / 2.047 / 2.047 / 2.049 us).
A 1314-launch frame spends >= 2.7 ms in launch floor alone even if every kernel
were empty.

### 1e. Bigger BLOCKS are not the lever; more BLOCKS are

At P=1000 all four block sizes land on **8.19 ns/item to three digits**, and NCU
says why: SM-active cycles are 17741 / 15993 / 15935 / 17050 for 64 / 128 / 256 /
512 threads - the same per-SM work either way. At maximum width the 64-thread
block is the FASTEST (3.07 vs 3.48 ns/item; 82799 vs 96106 SM-active
cycles). Widening the BLOCK buys only what widening the GRID already buys, and
slightly less of it.

**Consequence for the architecture question:** even where register pressure caps
the real kernels' block size, that does not cost the win. A wide-launch design
means MORE BLOCKS PER LAUNCH, and that is the axis that pays anyway.

## 2. The real kernels' occupancy limiters

NCU over the old capture, 3000 launches from launch 4000, clean window
1787789481..1787789580 (99 samples, none foreign). Limit columns are BLOCKS PER
SM allowed by each resource; the binding one is the minimum.

| kernel | n | grid med/max | block | regs | smem static | reg | smem | blk | blksz | **LIMITER** | max warps/SM | theo occ | achieved occ | warps in flight | waves/SM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `cp_rasterize_stage3` | 254 | 2048 / 2048 | 64 | 43 | 88 | 20 | 28 | 24 | 24 | **REGISTERS** | 40 | 83.33% | 13.60% | 6.53 | 0.60 |
| `cp_rasterize_stage3_abuf` | 69 | 512 / 2048 | 64 | 46 | 88 | 20 | 28 | 24 | 24 | **REGISTERS** | 40 | 83.33% | 6.40% | 3.07 | 0.15 |
| `cp_rasterize_stage1` | 147 | 75 / 75 | 256 | 47 | 0 | 5 | 16 | 24 | 6 | **REGISTERS** | 40 | 83.33% | 13.33% | 6.40 | 0.09 |
| `cp_clip_rast_fused` | 107 | 1 / 38 | 64 | 56 | 0 | 18 | 32 | 24 | 24 | **REGISTERS** | 36 | 75.00% | 2.14% | 1.03 | 0.00 |
| `cp_clip_rast_fused_abuf` | 69 | 1 / 2 | 64 | 56 | 0 | 18 | 32 | 24 | 24 | **REGISTERS** | 36 | 75.00% | 2.33% | 1.12 | 0.00 |
| `main` | 593 | 8 / 4096 | 256 | 94 | 0 | 2 | 16 | 24 | 6 | **REGISTERS** | 16 | 33.33% | 11.14% | 5.35 | 0.01 |
| `cp_abuf_scan_finish` | 74 | 450 / 450 | 512 | 26 | 4104 | 4 | 6 | 24 | 3 | **block size** | 48 | 100.00% | 76.84% | 36.88 | 0.88 |
| `cp_abuf_scan_reduce` | 37 | 450 / 450 | 512 | 34 | 2048 | 3 | 10 | 24 | 3 | **REGISTERS** | 48 | 100.00% | 77.50% | 37.20 | 0.88 |
| `cp_fs_compact` | 254 | 900 / 900 | 256 | 38 | 0 | 6 | 16 | 24 | 6 | **REGISTERS** | 48 | 100.00% | 77.84% | 37.36 | 0.88 |
| `cp_fs_writeback` | 255 | 2048 / 2048 | 256 | 40 | 0 | 6 | 16 | 24 | 6 | **REGISTERS** | 48 | 100.00% | 52.67% | 25.28 | 2.01 |

Notes. `cp_rasterize_stage1_abuf` does not appear: the fused
`cp_clip_rast_fused_abuf` replaces it on this capture. The fragment shader is
NVRTC-compiled and NCU names it `main`; register count varies per shader (39 to
140), so its row is a median over 593 launches.

### 2a. Is a wider BLOCK legal today?

**Two different answers, and the difference is the design fork.**

* **The fragment shader: NO, registers already cap it.** At 126 registers the
  register file allows **2 blocks of 256 threads per SM = 16 warps = 33.3%
  theoretical occupancy**. 65536/126 = 520 threads/SM whatever the block shape.
  A wider block buys nothing; it just moves the same 16 warps around.
* **The rasterizer stages: YES, but it does not help.** `cp_rasterize_stage3`
  and `_abuf` run **64-thread blocks at 43/46 registers**, limited to 20
  blocks/SM = **40 warps = 83.3% theoretical**, and they ACHIEVE 13.6% and 6.4%.
  They are **grid-limited, not register-limited**: 0.60 and 0.15 waves. Note
  65536/46 = 1424 threads = ~44 warps, so the register file caps the SM at ~40
  warps no matter how the blocks are shaped - block widening cannot beat that,
  and section 1e shows it would not pay anyway.

**So "wider launches" must mean MORE BLOCKS PER LAUNCH, not bigger blocks.**
That is the different design the brief asked about, and it is the one that is
available.

### 2b. The launches that are barely launches

| kernel | n | median grid | SM-active fraction of elapsed |
|---|---:|---:|---:|
| `cp_clip_triangles` | 102 | 1 | 0.5% |
| `cp_clip_rast_fused` | 107 | 1 | 0.5% |
| `cp_clip_rast_fused_abuf` | 69 | 1 | 0.5% |
| `cp_rasterize_stage2` | 254 | 10 | 0.7% |
| `cp_rasterize_stage1` | 147 | 75 | 2.0% |
| `cp_rasterize_stage3_abuf` | 69 | 512 | 38.1% |
| `cp_rasterize_stage3` | 254 | 2048 | 20.3% |

`cp_clip_rast_fused` launches **one block** 107 times and the SM is on for 0.5%
of the launch's elapsed time. `cp_rasterize_stage1` is on for 2.0%. These are the
P=1..12 rows of section 1 in the driver's own code.

## 3. THE HEADROOM CHECK - and it is the most important result

The brief asked: if the driver ALREADY has wide launches that also issue at 3%,
width is not the lever and the problem is the work itself.

**It does, and they do.**

The fragment shader `main`, 593 launches, bucketed by grid width. Its median
grid is **8 blocks**, not 4096 - so the "fixed grid" of 5.1 is the CAP, and the
typical FS launch is tiny. But 228 launches DO run at grid 4096:

| grid | n | regs | waves/SM | warps in flight | achieved occ | SM ISSUE %elapsed | SM ISSUE %active | SM-active frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-4 | 295 | 39 | 0.00 | 2.28 | 4.76% | 0.000 | 0.84 | 0.5% |
| 5-16 | 10 | 72 | 0.01 | 2.00 | 4.17% | 0.315 | 4.29 | 7.5% |
| 17-64 | 8 | 140 | 0.18 | 7.70 | 16.04% | 0.640 | 8.12 | 7.9% |
| 65-256 | 7 | 140 | 0.67 | 7.73 | 16.10% | 2.450 | 8.17 | 30.0% |
| 257-1024 | 45 | 116 | 1.32 | 10.86 | 22.61% | 6.260 | 9.22 | 76.3% |
| 1025-4096 | 228 | 126 | 12.05 | 12.33 | 25.69% | 3.640 | 10.32 | 35.5% |

**TWELVE WAVES PER SM AND IT STILL ISSUES 3.6%.** That launch has 4096 blocks -
enough to fill the machine twelve times over - and reads 3.64% elapsed / 10.32%
active, with achieved occupancy 25.7% and the SM switched on for only 35% of the
launch's elapsed time.

Two more already-wide launches say the same thing:

| kernel | waves/SM | warps in flight | achieved occ | SM ISSUE %elapsed | SM ISSUE %active | SM-active frac |
|---|---:|---:|---:|---:|---:|---:|
| `cp_rasterize_stage3_abuf` | 0.15 | 3.07 | 6.40% | 1.80 | 5.12 | 38.1% |
| `cp_rasterize_stage3` | 0.60 | 6.53 | 13.60% | 1.25 | 6.82 | 20.3% |
| `cp_fs_compact` | 0.88 | 37.36 | 77.84% | 7.17 | 8.28 | 86.0% |
| `cp_abuf_composite` | 0.88 | 37.20 | 77.49% | 18.60 | 24.41 | 76.7% |
| `cp_fs_writeback` | 2.01 | 25.28 | 52.67% | 2.63 | 13.38 | 19.6% |
| `cp_peel_advance` | 3.53 | 36.62 | 76.30% | 5.13 | 6.88 | 74.6% |

Going from the narrow raster launches (0.15-0.60 waves) to the wide ones
(2.01-12.05 waves):

* warps in flight rise from **3.1-6.5 to 12.3-36.6** - width DOES fill the SM;
* achieved occupancy rises from **6.4-13.6% to 25.7-76.3%** - width DOES work;
* but SM ISSUE rises only from **1.3-1.8% to 2.6-5.1%**.

My synthetic kernel at 2.45 waves reached **58.3%**. The driver's kernels at 2.01
and 12.05 waves reach 2.63% and 3.64%. **The difference is the WORK, not the
width. The driver's warps are resident and STALLED, not absent.**

### 3a. The device-side factor a merged raster launch can claim

`cp_rasterize_stage3_abuf` has a median grid of 512 blocks and old-capture
episodes average 12.42 segments, so a merged launch is ~6360 blocks. In this
benchmark's terms that is P=600 -> P=6000:

* P=600 -> P=6000: **3.98x**

* P=12 -> P=150: **10.75x**
* P=12 -> P=600: **64.43x**

**The answer to the 4x-or-8x question is 4x** (3.98x measured), not 8x - and
that is the SYNTHETIC kernel's factor, whose warps issue. The driver's own
kernels stall instead, and section 3 shows its 12-wave launch gains almost
nothing in issue. **Treat 4x as an upper bound, not an estimate.**

## 4. The marginal price of a launch, measured IN SITU

Requested mid-session, because the microbenchmark floor and the union-gap
analysis disagreed. Injection and slope, not a microbenchmark.

**Instrument.** Tree copied to `/tmp/perf-audit/wide/mesa` and built there; the
working tree was not touched. New registry flag **`CUDAVK_INJECT_NULL_LAUNCHES`**
(`CP_FLAG_UINT`, default 0, range 0..4096) launches N copies of a new
`cp_null_launch` kernel - no arguments, no memory traffic, writes nothing - into
`dev->renderer.stream` at one fixed point per submit, placed immediately before
the `cuEventRecord` the submit reports completion on. The injected launches are
therefore INSIDE the measured frame and BEHIND the real queue depth the driver
has already built. At N=0 nothing is launched, so output is unchanged by
construction. The capture submits twice per frame, so injected/frame = 2N.

Old capture, fps plugin, paired submits, skip 50, median; 1460 frames per run;
3 reps; N ladder run forwards, backwards, forwards. 18 runs, all clean.

| injected/frame | frame ms (mean of 3) | sd | us/launch vs N=0 |
|---:|---:|---:|---:|
| 0 | 13.128 | 0.031 | - |
| 100 | 13.388 | 0.009 | 2.597 |
| 200 | 13.541 | 0.018 | 2.065 |
| 400 | 13.942 | 0.054 | 2.036 |
| 800 | 14.731 | 0.009 | 2.004 |

**Least squares over all 15 points: slope = 1.9745 us/launch, intercept
13.154 ms, R2 = 0.99680, bootstrap 95% CI [1.925, 2.032].**

Segment slopes 2.597 / 1.532 / 2.007 / 1.972 us: **no rise with N**, so the host
is not being pushed onto the critical path even at 800 extra launches per frame
(+61% on the driver's 1314). The top point is not suspect.

**Control:** shipping ICD at N=0 measured 13.1615 ms against the patched
ICD's 13.1281 ms. The instrument costs -33 us/frame - nothing
outside noise.

This is **96.5% of the 2.047 us microbenchmark floor**. The floor was real.
The launch price is not paid in issue bandwidth - the driver uses only 20% of
the launch rate limit - it is paid **in the frame**, and injection can see that
while gap attribution cannot.

**Caveat, stated plainly:** injected launches are EMPTY. This is the price of a
launch that does no work, so 1.97 us is the UPPER BOUND of what merging one away
can return.

## 5. What widening can and cannot buy on this hardware

**CAN BUY - and these are large and measured:**

1. **The launch itself - but at 0.78 us each, not 1.974** (sections 4 and 7).
   CORRECTED after section 7: 1.974 us is the price of ADDING an exposed launch;
   removing a real, 75%-hidden chain launch is worth **0.782 us** (spread arm)
   / **0.809 us** (union idle in the trace). Removing 554 chain launches is
   worth **0.43 ms/frame**, not 1.09.
2. **Occupancy and warps in flight.** Width moves the driver's raster launches
   from 3.1-6.5 warps to 12-37, and achieved occupancy from 6-14% to 26-76%.
3. **Free carriage below the knee.** Up to ~7 blocks/SM, per-launch time does
   not change as items are added. The driver's launches sit at 0.00-0.60 waves,
   i.e. deep inside the free region. Everything merged into an existing launch
   below the knee costs nothing at all.
4. **Turning SMs on.** Below ~1 block/SM, SM-active MIN is literally zero: SMs
   never start. Every kernel with a median grid of 1 to 75 blocks in section 2b
   leaves most of a 170-SM machine switched off.

**CANNOT BUY:**

1. **Issue rate on this driver's kernels.** The clearest fact in this report: the
   FS shader at grid 4096 and **12.05 waves/SM issues 3.64%**, statistically the
   same 3% the driver reads overall. `cp_fs_writeback` at 2.01 waves issues
   2.63%. Width has already been tried by the driver, at scale, and it did not
   produce issue. **A merged raster launch will not run fast because it is wide.**
2. **A bigger block.** The fragment shader is register-capped at 2 blocks/SM and
   33.3% theoretical occupancy; and where blocks COULD grow, section 1e shows
   they should not - all block sizes converge, and 64-thread blocks win at max
   width.
3. **8x on the merged raster chain.** The measured synthetic factor for the
   actual merge geometry (512 -> ~6360 blocks) is **3.98x**, and that is an
   upper bound because the synthetic kernel issues where the driver's stalls.

**The one-line verdict** (revised by sections 7 and 8). Widening is worth taking,
but for the LAUNCH, not for the ISSUE - and the launch is worth less than the
floor suggested. The 287x in the synthetic curve is real and is what a starved
machine looks like, but the driver's own 12-wave launch proves the driver's
kernels do not convert width into issue. Budget a wide-launch programme against
**0.78 us x launches removed** - about **0.43 ms/frame** on the 554 mergeable
chain launches - and NOT against a device-side speedup multiple. If the
programme is sold on "the merged kernel will run 8x faster because it fills the
machine", section 3 says that is not what happens on this hardware; and at
0.78 us the launch axis ranks BEHIND the episode drain rather than ahead of it.

**What the 3% actually is:** not one problem but two, and they need different
fixes. Grid-limited launches (clip, stage1, stage2, stage3_abuf, most FS
launches: 0.00-0.60 waves, SM-active MIN of zero, SM on for 0.5-38% of elapsed)
are starved and width fixes them. Wide launches (FS main at 12.05 waves,
fs_writeback at 2.01, peel_advance at 3.53) are stalled, and only changing the
work fixes them.

## 6. Artefacts

```
/tmp/perf-audit/wide/wide_bench.cu        the microbenchmark
/tmp/perf-audit/wide/orchestrate*.py      gate -> run -> verify -> retry driver
/tmp/perf-audit/wide/gpu_watch.log        1 Hz sampler, whole session
/tmp/perf-audit/wide/res/clean_windows.json  every clean window, per run
/tmp/perf-audit/wide/res/inject.json      injection sweep, 18 clean replays
/tmp/perf-audit/wide/res/fine.log         fine P sweep (clean)
/tmp/perf-audit/wide/res/timing2.log      coarse sweep, clean re-run
/tmp/perf-audit/wide/ncu/shape_T*_P*.csv  NCU per shape (clean)
/tmp/perf-audit/wide/ncu/driver_all.csv   NCU, 3000 driver launches (clean)
/tmp/perf-audit/wide/mesa/                instrumented copy; working tree untouched
```

## Appendix A. Is the benchmark really L1-friendly, as claimed?

Checked rather than asserted. Clean window 1787798412..1787798416.

| items/launch | L1 hit rate | DRAM bytes | warps in flight |
|---:|---:|---:|---:|
| 12 | 75.01% | 39,424 | 1.97 |
| 1000 | 95.47% | 102,656 | 11.40 |
| 10000 | 99.33% | 300,800 | 42.77 |

The driver's real kernels show 92.5% L1 hit and 1% DRAM. The benchmark reaches
**95.5% at P=1000 and 99.3% at P=10000** with DRAM traffic in the hundreds of
kilobytes, so at the widths that matter it is a fair match. **Honest
qualification:** at the narrow shape (P=12) it hits only 75.0%, below the
driver's 92.5%, because a single block cannot keep the 32 KiB table resident on
its own. That makes the narrow points slightly PESSIMISTIC relative to the
driver, so the 287x is, if anything, a mild over-statement of the span - the
crossover SHAPE (flat per-launch time below the knee) does not depend on it.

## 7. Is 1.974 us the price of ADDING an exposed launch, or of REMOVING a real one?

It is the price of adding an EXPOSED one. A real chain launch is 75% hidden.

### 7a. The free route: read the trace of a run already paid for

nsys traces, `-t cuda`, 40 s each, both CLEAN (`nsys_inject800` window
1787798705..1787798773, 70 samples; `nsys_base` 1787798839..1787798900, 63
samples; neither overlapped the later rebuild at t=1787799152). PER-STREAM gaps
are not the right statistic - this driver runs eight side streams plus the main
one - so the number below is the UNION idle: for each kernel, how long NO kernel
was running anywhere on the device immediately before it started.

| population | n | zero union-gap | median | mean union-idle |
|---|---:|---:|---:|---:|
| injected nulls, BUNCHED (N=800/frame) | 1,208,800 | **0.0%** | 1792 ns | 3099 ns |
| injected nulls, SPREAD (N=800/frame nominal) | 600,676 | 45.6% | 960 ns | 1085 ns |
| REAL chain kernels, N=0 control | 955,301 | **74.9%** | 0 ns | **809 ns** |

**The hiding model is confirmed, and by the route that costs nothing.** Not one
of the 1.21 million bunched injected launches had a kernel in front of it
(0.0% zero-gap); each one sat behind 1792 ns of device idle and then ran for
256 ns, so it consumed 2048 ns of device time - which is the 2.047 us floor and
the 1.974 us frame slope, to within 4%. **The frame-time slope is device idle,
not host CPU time**, which independently confirms the ceiling agent's arithmetic
rejection of the host interpretation.

And the control says the driver's own chain is NOT like that: **74.9% of real
chain kernels start with zero union idle**, and the mean union idle in front of
one is **809 ns**. That is close to the ceiling agent's 80.8% / 0.42 us from the
stored trace - same phenomenon, measured independently on a clean run.

So my own caveat was right but understated it. 1.974 us is an upper bound for
**two** reasons: the injected kernels do no work, AND they have no neighbour to
hide behind.

### 7b. What removing a REAL chain launch is worth

Two independent instruments:

* **Union idle in front of real chain kernels: 809 ns.** This is what a
  merge actually gives back - the device idle that disappears when the launch
  does. Measured directly on the shipping code path.
* **The spread injection arm** (below): null launches placed one behind each
  real launch, so each has a neighbour to hide behind.

### 7c. The spread arm (coded), and the count correction that mattered

`CUDAVK_INJECT_SPREAD` places one null launch behind each real launch inside
`cp_launch()` - the driver's single launch choke point - drawing on a quota
refilled per submit. Same ladder, same convention, 15 clean runs.

**The raw result was wrong and I nearly reported it.** The nominal x-axis
over-counts: nsys shows the spread arm lands only ~50% of the requested count,
because the capture submits TWICE per frame and only one of the two submits
carries the draw work, so the quota is consumed on one and untouched on the
other. Verified over the SAME PREFIX OF REAL KERNELS in both arms (the bunched
arm is exact by construction, so it calibrates the other):

| N per submit | nominal/frame | nulls injected | bunched equivalent | landed | ACTUAL/frame |
|---:|---:|---:|---:|---:|---:|
| 50 | 100 | 6,950.0 | 13,850.0 | 50.2% | 50.2 |
| 100 | 200 | 13,900.0 | 27,700.0 | 50.2% | 100.4 |
| 200 | 400 | 27,720.0 | 55,400.0 | 50.0% | 200.1 |
| 400 | 800 | 52,276.0 | 110,800.0 | 47.2% | 377.4 |

Against the CORRECTED x-axis:

| actual injected/frame | frame ms (mean of 3) | sd | us/launch |
|---:|---:|---:|---:|
| 0.0 | 13.125 | 0.012 | - |
| 50.2 | 13.176 | 0.031 | 1.024 |
| 100.4 | 13.210 | 0.054 | 0.845 |
| 200.1 | 13.285 | 0.019 | 0.799 |
| 377.4 | 13.653 | 0.015 | 1.399 |

All four points: slope **1.374 us/launch**, R2 0.927, CI [0.82, 1.52].

But the parent's own pre-registered guard applies: *if the slope rises with N,
the top point is suspect.* It does - 1.024, 0.845, 0.799, then **1.399**. At 377
nulls against 1155 real launches, one launch in three has a null behind it, so
the placement is drifting back toward bunching and the nulls begin to expose one
another. The linear region is the first three points:

**SPREAD, linear region: 0.782 us/launch, 95% CI [0.633, 0.949].**

### 7d. Verdict on measurement A

| instrument | us per launch |
|---|---:|
| union idle in front of a real chain launch (nsys, N=0 control) | **0.809** |
| spread injection, linear region (frame-time slope) | **0.782** |
| driver's own price table (prior work) | 0.6 - 1.0 |
| bunched injection - an EXPOSED launch, not a removed one | 1.975 |
| back-to-back empty-kernel floor (microbenchmark) | 2.047 |

**Three independent instruments agree on 0.78-0.81 us, and the ceiling agent's
model is confirmed.** 1.974 us is the price of ADDING an exposed launch. The
price of REMOVING a real, 75%-hidden one is about **0.8 us**. My earlier framing
of 1.974 as the marginal price was wrong and is withdrawn.

Against the parent's registered prediction (spread 0.8-1.2, falsifier at 1.6):
the linear region reads 0.782 with CI [0.633, 0.949] - **at the bottom edge of
the predicted band**, not above 1.6, so the hiding model stands.

**Conversion.** 554 mergeable chain launches x 0.782 us = **0.433 ms/frame**.
To tie an episode drain worth ~1.0 ms/frame you would need ~1278 launches
removed, and only 554 are legally mergeable. **The launch axis ranks BEHIND the
drain.** At the 1.974 figure it would have led; at 0.78 it does not.

## 8. The stall axis, with the falsifier registered before the data was read

**REGISTERED BEFORE LOOKING.** A warp is 32 lanes x 4 B = 128 B = 4 sectors of
32 B, so 4.0 sectors/request is perfect coalescing. *If sectors-per-request is
already at 4, the stall is NOT access-pattern; the fix is a data-layout change
worth a quarter's work, and the lead is recorded as large-but-expensive rather
than opened. If it is well above 4, uncoalesced access is confirmed.*

NCU, old capture, 12 launches each, clean window 1787798966..1787799103
(137 samples, none foreign).

### 8a. The falsifier fired

| kernel | sectors/request, global LD | global ST | L1 hit | warps in flight |
|---|---:|---:|---:|---:|
| `cp_rasterize_stage3` | **1.00** | 0.00 | 95.8% | 20.39 |
| `cp_rasterize_stage3_abuf` | **1.00** | 4.91 | 47.4% | 4.52 |
| `cp_clip_rast_fused` | **9.44** | 6.83 | 81.2% | 1.64 |

**Both rasterizer stages read 1.00 sectors per request.** That is not
uncoalesced - it is the far side of coalesced. Every global load request pulls a
SINGLE 32-byte sector, so the access pattern is not wasting bandwidth; the
kernel is issuing many narrow requests instead of few wide ones. The hypothesis
that a 41-68% long-scoreboard stall at 1% DRAM means uncoalesced access is
**falsified for the two kernels that carry the cost.**

`cp_clip_rast_fused` is the exception at 9.44 sectors/request - genuinely
scattered - but it launches a median of ONE block, so its problem is width, not
access pattern.

### 8b. The full stall breakdown

`smsp__warp_issue_stalled_*_per_warp_active`, median over 12 launches, as a
share of all warp-cycles:

| stall reason | cp_rasterize_stage3 | cp_rasterize_stage3_abuf | cp_clip_rast_fused |
|---|---:|---:|---:|
| `long_scoreboard` | 70.7% | 56.3% | 45.8% |
| `short_scoreboard` | 16.8% | 13.6% | 7.5% |
| `wait` | 5.2% | 9.7% | 23.9% |
| `no_instruction` | 4.2% | 15.0% | 3.0% |
| `selected` | 1.0% | 2.9% | 9.5% |
| `branch_resolving` | 1.0% | 1.5% | 9.5% |
| `drain` | 1.0% | 0.0% | 0.0% |
| `barrier` | 0.0% | 1.0% | 0.0% |
| `dispatch_stall` | 0.0% | 0.0% | 1.0% |

`long_scoreboard` (waiting on a global/local load to return) dominates all
three: **70.7% / 56.3% / 45.8%**. `selected` - actually issuing - is **1.0% /
2.9% / 9.5%**.

### 8c. What that means, and the honest size of the lead

The two stage-3 kernels differ in a way that settles the mechanism:

* `cp_rasterize_stage3` has **20.4 warps in flight** and STILL stalls 70.7% on
  long scoreboard. Twenty warps is not starvation. More width cannot hide this.
* `cp_rasterize_stage3_abuf` has only **4.5 warps** and stalls 56.3%. For this
  one, width WOULD help - it is both starved and stalled.

So the stall is memory LATENCY on narrow requests, not bandwidth and not
coalescing. The remedy is not "coalesce the access" - it is already coalesced -
but "make each request carry more data": wider loads, vectorised or
structure-of-arrays layout so one instruction fetches 128 B instead of 32 B.

**Per the registered falsifier, this lead is recorded as LARGE-BUT-EXPENSIVE and
is NOT opened as an iteration.** It is a data-layout change across the
rasterizer's setup and queue structures, not a local fix.

**Sizing, converted through exclusivity rather than kernel time**, as instructed:
the chain at 2x is 0.752 ms/frame of union busy and at infinity 2.044 ms/frame.
A change that removed the long-scoreboard stall entirely is bounded by the
infinity case, so the honest headline is **up to ~2.0 ms/frame, at data-layout
cost** - the largest single number in this session, and the least available.


## 9. Standing down: everything this session settled

| question | answer | instrument |
|---|---|---|
| Does a wide launch issue better than many narrow ones? | Yes, hugely, on a kernel whose warps issue: 287x for the same work | synthetic sweep, 2 clean runs agreeing to 0.64% |
| Where does width stop being free? | Flat per-launch time to ~7 blocks/SM; knee at 0.29-0.59 waves; saturation ~2.5-3 waves | fine P sweep |
| Is a wider BLOCK legal? | FS shader no (126 reg -> 2 blocks/SM, 33.3% theo). Raster stages yes but pointless - all block sizes converge and 64 wins at width | NCU limiter table |
| Does the driver already have wide launches? | Yes. FS main at grid 4096, **12.05 waves/SM, issues 3.64%** | NCU, 3000 launches |
| So is width the lever? | **No for issue, yes for launches and occupancy.** The driver's warps are resident and stalled, not absent | measurement 3 |
| What is a launch worth? | **0.78-0.81 us** removed (not 1.974, which is an exposed launch) | 3 instruments agreeing |
| Is the merged-raster device factor 4x or 8x? | **3.98x**, and that is an upper bound | fine sweep at the real merge geometry |
| Is the long-scoreboard stall uncoalesced access? | **No - 1.00 sectors/request.** Falsifier fired; lead is data-layout, large-but-expensive | registered before looking |

**Two things I got wrong and corrected in-session**, both recorded above rather
than quietly fixed: the first exclusivity checker asked "does this line mention
me" instead of "is every entry mine", and the spread arm's x-axis over-counted
by 2x until nsys was used to count what actually landed. Both would have
produced a confident wrong number.

**Working tree untouched.** `/home/alexzhukov/mesa` is at `e2fea470d04` with the
same four pre-existing doc modifications it had at the start; all instrumentation
lives in `/tmp/perf-audit/wide/mesa`.
