# Item 4, F4 - the device-side fork. **STRUCTURAL, not implementation cost per item**

nsys 2026.4.1 CUDA traces, four windows (`--delay 12 --duration 8`) alternating
arms, plus two NCU launch-stat passes. Both arms carry
`CUDAVK_NO_ABUF_APPEND=1`; candidate adds `CUDAVK_MERGE_FILL_RAST=1`.

## Exclusivity - a tenant DID appear, the gate caught it, no used run touched it

Two windows, 1 Hz sampler throughout, start gated on 8 consecutive idle samples.

* **window 1: 223 samples, t+0..t+225 s.**
* **window 2: 87 samples, t+0..t+88 s.**

The raw audit flagged entries, and both classes are resolved by PID, not by
assumption:

1. **`build_vp9_cuda/vpxenc`, the user's own batch, 16 samples at
   t=1787858614..631.** These fall in window 2's GATE LOOP. The gate refused to
   start until 8 consecutive idle samples and began the first run at
   t=1787858640. **No vpxenc sample falls inside any run interval.**
2. Entries reading `[No data], 12452 MiB` at t=1787858414 and t=1787858699.
   nvidia-smi failed to read a process name at that instant. Their PIDs are
   **3183165** and **3185954**, and each capture's own `PROCESSES` table names
   those PIDs as **my** `gfxrecon-replay` (cand-a and cand-c respectively).

Per-run audit after resolving PIDs - every entry, not presence of mine:

| run | samples | foreign |
|---|---:|---|
| ctrl-a | 31 | none |
| cand-a | 29 | none |
| ctrl-b | 28 | none |
| cand-b | 29 | none |
| ctrl-c | 30 | none |
| cand-c | 26 | none |

**Every run used below was exclusive.** This is the first window this session
in which the box's vpxenc batch appeared at all; it appeared before a start and
the gate held.

## Captures used

`ctrl-b` and `cand-c` produced reports with **zero CUDA kernel rows** and are
discarded, not repaired. That leaves two valid controls (ctrl-a, ctrl-c) and
two valid candidates (cand-a, cand-b), cross-paired into two independent
estimates.

## GATE BEFORE READING - the nsys windows reproduce the counter arms

Total kernel launches per `cp_fs_compact` (an untouched kernel, one per batch):

| | ctrl-a | ctrl-c | cand-a | cand-b |
|---|---:|---:|---:|---:|
| launches per fs_compact | 25.712 | 25.778 | 20.862 | 20.838 |

Candidate/control = **0.8107**, against the counter arms' 1,383.6 / 1,682.4 =
**0.8224**. Agreement to **1.4%**. The traced windows carry the same
launch-count reduction the F1 counter runs measured, so they are the right
windows to read.

## F4 - **PASSES, and not marginally: merged per item is a THIRD of control**

Per-item device time. The candidate's remaining un-merged `*_abuf` launches are
the untouched append half; scaling them to the control's work volume and
subtracting isolates the control's fill half, which is exactly the set the
merge replaced.

| stage | control ns/item | merged ns/item | **ratio** | items per merged launch |
|---|---:|---:|---:|---:|
| **pair A** (ctrl-a x cand-a) | | | | |
| stage1 | 8,113 | 3,428 | **0.423** | 4.95 |
| stage2 | 5,563 | 2,410 | **0.433** | 4.95 |
| **stage3** | **17,065** | **5,747** | **0.337** | 4.95 |
| **pair B** (ctrl-c x cand-b) | | | | |
| stage1 | 8,125 | 3,412 | **0.420** | 4.98 |
| stage2 | 5,537 | 2,374 | **0.429** | 5.02 |
| **stage3** | **17,035** | **5,670** | **0.333** | 5.02 |

The bar was **<= 1.25x**. Measured **0.333-0.337x** for stage3, on two
independent pairs agreeing to 1%. **The merged form is three times CHEAPER per
item than the loop it replaced.**

## The implementation cost the author feared IS present - in geometry, not in time

NCU launch statistics, same build, both arms:

| kernel | registers/thread | static shared/block | theoretical occupancy |
|---|---:|---:|---:|
| `cp_rasterize_stage1_abuf` | 48 | 0 B | 83.33% |
| `cp_rasterize_stage1_abuf_merged` | **64** | **400 B** | 66.67% |
| `cp_rasterize_stage2_abuf` | 56 | 0 B | 66.67% |
| `cp_rasterize_stage2_abuf_merged` | **109** | **400 B** | **33.33%** |
| `cp_rasterize_stage3_abuf` | 46 | 88 B | 83.33% |
| `cp_rasterize_stage3_abuf_merged` | **94** | **488 B** | **41.67%** |

Registers roughly **double** and theoretical occupancy **halves** on all three
merged kernels. So the register and shared pressure is real and measurable - it
simply does not dominate, because it is amortised over ~5 items while the
launches it replaced were tiny (control stage1/stage2 fill launches run grids
of 1 to 26 blocks).

## SO WHERE DID THE 0.410 ms GO? Not into device work. Into LOST CONCURRENCY.

Union of kernel intervals - the exclusivity instrument, not summed kernel time -
normalised to a common work volume:

| | ctrl-a | ctrl-c | cand-a | cand-b |
|---|---:|---:|---:|---:|
| union busy, ALL kernels | 4,342.2 ms | 4,357.3 ms | **4,303.1 ms** | **4,314.6 ms** |
| union busy, raster-abuf chain | 1,278.9 ms | 1,285.0 ms | **1,243.4 ms** | **1,238.2 ms** |
| **overlap factor of that chain** | **2.19x** | **2.19x** | **1.37x** | **1.38x** |

**The candidate is LESS busy on both measures** - 0.9% less device busy
overall, 3.2% less in the chain it changed - **and it is still 0.410 ms/frame
SLOWER.** The device is not paying for the merge; it is being paid.

The overlap factor names the mechanism. The control's fill chain runs at
**2.19x overlap** - its per-segment launches are concurrent, over the eight
side streams blended segments have always used. The merged form drops to
**1.37x**: a chunk is one grid on one stream. Per-launch wall time follows:

| stage | control launch avg | merged launch avg (covers ~5 items) |
|---|---:|---:|
| stage1 | 8,285 ns | 16,980 ns |
| stage2 | 5,132 ns | 11,932 ns |
| stage3 | 14,181 ns | 28,453 ns |

A merged launch takes about **2.0x as long as ONE control launch** while
replacing about **five that were running concurrently**. If those five were
perfectly concurrent, the critical path per chunk grows by
(28,453-14,181) + (16,980-8,285) + (11,932-5,132) = **29.8 us**, and at 82.3
chunks per frame that bounds the loss at **2.45 ms/frame**. The observed
0.410 ms/frame is **17% of that bound** - i.e. the bound comfortably contains
the measurement, and most of the lost concurrency is still being hidden by the
rest of the frame.

## THE FORK, ANSWERED

**STRUCTURAL.** Merged per item is at 0.33x, far below the 1.25x
implementation-cost bar, and union busy FELL. The device is not paying for the
merge. **The launch-count axis is refuted at this width for launches that were
already concurrent.**

But the refutation is sharper than "launch count does not pay", and the sharper
form is what should be carried forward:

> **A merge only collects the launch-removal credit where the launches it
> merges were SERIAL. Where they were concurrent, merging converts parallel
> work into a longer critical path, and the credit inverts.**

At this position the control's fill loop was running at 2.19x overlap, so the
298.8 launches/frame removed were never costing 0.782 us each on the critical
path - they were overlapping each other. That is why the credit did not
transfer, and why it arrived with the wrong sign.

**This does NOT close a cheaper merged form as a possibility, but it changes
what such a form would have to fix.** Halving the register count would not
help: the loss is not per-item efficiency, which is already 3x better. It would
have to restore concurrency - i.e. issue the merged chunks on several streams,
which is most of what the loop already did.

**What it DOES close:** any launch-count forecast for a merge at a position
whose launches are currently concurrent. The count-phase merge must be checked
for overlap factor at its own position BEFORE its launch arithmetic is
believed.

## Raw

/tmp/perf-audit/jobF4/ - 6 nsys attempts (4 usable), 2 NCU launch-stat passes,
watch.log, watch2.log, progress.log.
