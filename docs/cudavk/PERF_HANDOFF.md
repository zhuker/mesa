# Performance handoff

State of the driver's performance work as of `cudavk-0.0.1` and the session
that followed it. Read `GETTING_STARTED.md` first if you have not built the
driver; read `WORKFLOW.md` before quoting any number; read `DEAD_ENDS.md`
before optimising anything.

## Where the numbers are

Compiled-replay paired-submit medians, each capture on its own real-work
window (WORKFLOW.md 4.0), all defaults, clean environment:

> **The floor is 0.522 / 0.502 ms.** The same GPU runs the same submits on
> NVIDIA's own driver 10-11x faster than cudavk does
> (`notes/NATIVE_DRIVER_REFERENCE.md`, measured 2026-09-03). Read that before
> planning optimisation work: the remaining distance to the 5.0 goal is ~1.7%
> of the gap to a real driver, and no lead in the ledger addresses the
> architecture that gap belongs to.

| capture | RTX 5090 (sm_120) | B200 (sm_100) |
|---|---:|---:|
| favorite3 | **5.940** | **9.883** |
| favorite2 | **5.022** | **8.283** |

**Against the previous release**, measured 2026-09-05 with both drivers
alternating run-by-run in one session (`CTRL_ICD`/`CAND_ICD`), same harness
binary, all twelve runs passing their gates and every arm disjoint:

| capture | band | cudavk-0.0.1 | cudavk-0.0.2 | delta | |
|---|---|---:|---:|---:|---:|
| favorite3 | whole | 6.6823 | **5.9213** | -0.7610 | **-11.4%** |
| favorite3 | heavy | 8.2369 | 7.2881 | -0.9487 | -11.5% |
| favorite2 | whole | 5.4947 | **5.0433** | -0.4514 | -8.2% |
| favorite2 | heavy | 6.0580 | 5.6064 | -0.4515 | -7.5% |

Six changes account for it: the append prefilter, the episode stream gate, the
clip-scratch hoist, the vertex side lane, the context-scope trim and the
layered copy merge. Both builds produce **0 sentinel mismatches against the
same control**, so the two releases render identically.

Both hosts now include everything landed on 2026-09-02/03. The six changes
were validated on the B200 as one set (control = all six revert switches on,
three alternating rounds, every gate passed): they are worth **-0.985 ms/frame
on favorite3 (heavy band -1.403) and -0.678 on favorite2 there**, against
-0.736 and -0.458 for the same six on the RTX -- **1.3x and 1.5x**, consistent
with host-side scheduling changes on the more latency-bound host. favorite2's
heavy band overlapped and is inconclusive. Individual attribution on the B200
was not measured, only the set. (`notes/LEAD_RESULTS_2026-09-02.md` has the
run.)

Reference points: llvmpipe from this tree, release build, is 46.40 / 39.72 on
the same two captures — cudavk is **6.9x / 7.2x** faster. Rendering is
bit-identical between sm_120 and sm_100 on both captures.

Both captures have a light and a heavy band, and the heavy band is where the
remaining time is:

| capture | light | heavy |
|---|---:|---:|
| favorite3 | ~3.8-4.4 (frames 1700-2100) | **7.6-9.5** (frames ~2200-3150) |
| favorite2 | **4.795** (1388-2734) | **6.046** (2735+) |

favorite2's light band already sits under 5 ms. Nothing else does.

## The shape of a frame, measured three ways

These three agree, which is why the conclusion below is firm.

**Device side** (nsys, RTX heavy band, 1,011 frames): device union 5.21
ms/frame of an 8.40 ms wall, so 3.19 ms idle. Largest exclusive pools:
generated VS 0.689, generated FS 0.678, `cp_clip_rast_fused` 0.592 (2.34x
self-overlapped), `cp_rasterize_stage3` 0.424, `cp_fs_compact` 0.372, memcpy
0.490. 727.9 launches/frame.

**Kernel counters** (ncu, both GPUs, matched build): these kernels run at
**0.5-12% of peak SM throughput and 3-25% occupancy**. `cp_clip_rast_fused`
sits at 0.5% SM and 0.02 waves per SM. They finish before the machine fills.
The B200 executes them in *equal or fewer cycles* than the RTX
(`cp_fs_compact` 0.59x cycles at 1.75x IPC), so its slower wall time is clock
and issue, not architecture.

**Host side** (perf, DWARF, no CUPTI, 2,078 real frames): the main thread is
on-CPU **97% of wall** but spends **4.382 ms/frame spinning in CUDA sync
calls** — `cp_smallop_ctxsync` 2.885, `cp_sync_timed` 1.481. Issue costs
1.356, driver planning 0.678 of which 0.510 is shader compilation confined to
13 frames of 2,078. **Steady-state driver CPU work is about 0.17 ms/frame.**

**Therefore**: the frame is a long dependent chain of kernels each too small to
fill the GPU. The CPU is not the bottleneck (0.17 ms of real work), the GPU is
not saturated (single-digit occupancy), and the wait time is device pacing
rather than a cost that can be deleted.

## What that rules out, with the measurements

Blocked host time is not collectible: threaded submit recovered **0.126**
ms/frame (34), scratch high-water reuse removed 1.65 ms of blocking and paid
**0.037** (29). Launch-count reduction is closed from four sides: the wide
merge removed 298.8 launches/frame and cost **+0.410** (20), its multi-stream
retry showed no signal (20 retry), the count-phase merge was refuted before
building (21), and the merge rule (22) explains all of them — *a launch-removal
credit is only collectable where the launches were serial*. Here they are
concurrent, and that concurrency is load-bearing: surrendering the opaque
fan-out costs **0.66-0.84 ms/frame** (measured 2026-09-01).

The renderer-level rewrite was taken to a design decision and refused on
arithmetic (36): the tile walk itself is fine — **26.5 cycles per reference**
against the previous prototype's 3,409 — but the pools it displaces do not
contain the goal. Removing the *entire* displaced pool at zero cost and
surrendering nothing lands favorite3 at 5.21 ms.

## What is left, honestly

No untried mechanism with a defensible >=0.500 ms/frame ceiling is known on
either GPU. The pools that hold the time — vertex and fragment arithmetic,
clip, and copies — are clock-bound work that scheduling does not shrink. The
sub-gate leads that exist and are documented: more pass side streams (38,
-0.084 on favorite2 for +153 MB), opaque FS UBO row concatenation (TODO,
0.04-0.06), `cp_tri_setup` padded 80->96 B to stop straddling 32 B sectors
(36, unmeasured in production).

If you want a materially faster frame, the evidence points outside the
driver's scheduling: less work submitted per frame, or hardware with a higher
clock. Both are outside what this campaign could change.

## Tooling and hosts

- **RTX 5090**, local. nsys 2026.4.1.191, ncu 2025.3.1.0 build 36398880 at
  `~/opt/nsight-compute-2025.3.1/ncu` (extracted, not installed system-wide).
  `RmProfilingAdminOnly: 0`, so counters work unprivileged.
- **B200**, via `ssh azhukov@192.168.1.154 ssh b200`. Same nsys and ncu builds.
  `RmProfilingAdminOnly: 1`, but the workspace now carries CAP_SYS_ADMIN, so
  **`sudo ncu` works**. Trees at `~/mesa` (CUDA 13 build), harnesses at
  `~/favorite3-cpp/out` and `~/favorite2-cpp/out`, sentinels at `~/sentinels`.
  Home persists across restarts; **system packages and `/tmp` do not**.

## Measurement rules this campaign paid to learn

1. **Each capture has its own real-work window.** favorite3 starts at frame
   1391 (submit 2782), favorite2 at frame **1388** (submit 2776). Applying
   one to the other is an error made here for a whole campaign (WORKFLOW 4.0).
2. **Align profiler builds across hosts, and run a positive control.** nsys
   2026.1.3 reported 1.9 MB/frame of unified-memory migration that 2026.4.1
   shows does not exist; a driver flag was written against that phantom before
   the versions were matched (37, WORKFLOW 4.05).
3. **The B200 needs >=3 alternating rounds of one pair.** Its session drift
   reaches 1.3 ms/frame against the RTX's 0.06. A two-run B200 probe read
   -0.65 ms; the controlled A/B reversed the sign (37).
4. **Check that a census reducer is reachable before trusting its zeroes.**
   `cp_tile_census_end_pass` and `cp_tile_census_cut` had no callers, so every
   number that machinery printed was structurally zero (24, 36).
5. **A win on one capture must be measured on the other, and on both GPUs if
   it touches memory policy** — the residency advice helped one arm in a bad
   probe and cost 1.98 ms/frame on the RTX in a good one.
