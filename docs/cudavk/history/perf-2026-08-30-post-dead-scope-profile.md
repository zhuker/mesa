# Post-dead-scope favorite3 profile — 2026-08-30

Tree: `f3caa35e4ce` (dead render-scope elimination landed). Compiled
HeadlessStreamer replay: `~/favorite3-cpp`, real frames start at frame 1391 /
submit timestamp index 2782.

This record replaces pre-dead-scope kernel rankings. It does not replace the
normal alternating compiled-replay timing oracle.

## Current frame and no-profiler duty cycle

The compiled replay's current real-frame control median is **7.604 ms**.
A separate run sampled `nvidia-smi` at 20 ms without CUPTI and aligned samples
to submit-shim `CLOCK_MONOTONIC` timestamps. Over frames 1391+ it collected 863
samples in 17.305 seconds:

| statistic | GPU busy |
|---|---:|
| median | **78%** |
| mean | 67.6% |
| p10 / p90 | 12% / 80% |
| max | 81% |

The low mean is real but not representative of the median frame: one roughly
two-second interval inside the capture reports 0–12% busy. The steady modes
are 76–80%. For the median-frame budget, 78% leaves an honest no-profiler idle
ceiling of `7.604 * (1 - .78) = 1.673 ms/frame`. It is a ceiling, not a prize:
dependency waits and issue work cannot all be overlapped.

Raw data: `/tmp/postdead-busy/`.

## Current host waits and operation rate

One complete replay with `CUDAVK_PLAN_STATS=1 CUDAVK_UPLOAD_STATS=1` retained
6,939 submits and the standard stdout hash. Over 3,473.5 frames it issued
1,439,348 kernels, or **414.4 kernels/frame** (about 54.5k/s at 7.604 ms):

| main-thread site | whole-replay waits/frame | blocked ms/frame |
|---|---:|---:|
| episode drain | 3.74 | **0.983** |
| segment counters | 1.47 | **0.314** |
| peel checks | 1.13 | **0.133** |
| descriptor uploads | 1.00 | 0.008 |

These are blocked-host symptoms, not an additive 1.438 ms opportunity. Prior
site-specific conversion and mechanism probes in `PERFORMANCE.md` still govern:
the broad drain, peel-predication, worst-bound, run-ahead-capacity and
device-chain paths remain closed. Raw data: `/tmp/postdead-stats.err`.

## CUPTI slice: current device ownership

Nsight Systems traced 412.5 real frames, from the frame-1391 submit at relative
9.276 s through the last recorded kernel at 12.341 s. CUPTI captured 176,208
kernels in this slice. The app still emitted all 6,939 submit timestamps, but
kernel activity after 12.341 s is absent, so all figures below use only this
bounded common interval.

CUPTI inflates host launch time and reports only **4.510 ms/frame** of union GPU
activity (60.7% of its 7.429 ms traced frame). Do not use that percentage to
classify the untraced workload; the 78% sampler above is authoritative. The
trace remains useful for relative kernel ownership and union-exclusive caps:

| class | launches/frame | summed ms/frame | union-exclusive ms/frame |
|---|---:|---:|---:|
| shader kernels (`main`) | 90.56 | 1.547 | **1.098** |
| all raster/clip | 157.52 | 2.675 | **1.400** |
| direct raster/clip | 110.39 | 2.280 | **1.134** |
| A-buffer raster/clip | 47.13 | 0.395 | **0.266** |
| direct FS support | 67.77 | 0.251 | **0.250** |
| A-buffer support | 89.92 | 0.252 | **0.252** |
| vertex fetch | 7.51 | 0.195 | **0.168** |

The leading individual exclusive rows are shader `main` 1.098,
`cp_clip_rast_fused` 0.406, `cp_rasterize_stage3` 0.305,
`cp_fs_compact` 0.186, `cp_rasterize_stage2` 0.169, `cp_vertex_fetch` 0.168 and
`cp_rasterize_stage3_abuf` 0.132 ms/frame.

Raw report: `/tmp/postdead/trace.sqlite`; submit alignment:
`/tmp/postdead/submit-ts.txt`.

## Consequence for the 5 ms target

The remaining 2.604 ms cannot come from either current axis alone:

- perfect collection of the untraced idle pool is capped at 1.673 ms/frame;
- deleting every traced raster/clip interval exclusively is capped at 1.400;
- deleting every shader interval exclusively is capped at 1.098.

A successful architecture must combine duty-cycle improvement with less device
work, or change the work unit. This is why the next raster probe is a
count-only episode-entry stale-depth/Hi-Z census: it can shorten the recorded
depth-complexity tile lists without surrendering the eight-stream opaque
fan-out. A separate two-slot opaque raster-to-shade pipeline remains probe-only,
with a whole-profile arithmetic cap of `min(1.400, 1.098) = 1.098 ms/frame`.

## Correctness rejection found during this profile

Relaxing the hardware-inline `.local` veto appeared to save 0.077 ms on
favorite3 and 0.036 ms on favorite2, but changed 15/18 and 14/18 sentinel
frames respectively. `DEAD_ENDS.md` entry 28 records the refusal. No source
from that branch was merged.
