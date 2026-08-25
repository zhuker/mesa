# Performance

Where the driver's time goes today, what it costs to remove different kinds of
work, and what is left. Read this before choosing what to optimise next.

Every number here comes from a measurement that is named beside it. The long
form of each one is in `history/PERF16_ITERATIONS.md` (the iteration record) and
in the raw evidence under `/tmp/perf16/`. Approaches that are closed are in
`DEAD_ENDS.md`; open questions are in `TODO.md`; how to run a measurement is in
`WORKFLOW.md` and `TESTING.md`.

Two GFXR captures carry the work:

| name used here | file |
|---|---|
| **old capture** | `headless_streamer_20260814T155742.gfxr` |
| **Crossroads** | `headless_streamer_1818_20260817T173522.gfxr` |

---

## 1. Where the driver is today

The shipping default, measured alone — no arms, no flags beyond the base
environment, one session, six replays of the old capture and four of Crossroads
(`history/PERF16_ITERATIONS.md`, "The standing default, stated as a
distribution"):

| capture | run medians (ms/frame) | median | IQR | full range |
|---|---|---:|---|---|
| old | 15.7200, 15.7544, 15.7614, 15.7483, 15.7245, 15.7911 | **15.7514** | [15.7305, 15.7596] | [15.7200, 15.7911] |
| Crossroads | 5.9007, 5.8984, 5.8980, 5.8569 | **5.8982** | [5.8877, 5.8990] | [5.8569, 5.9007] |

Quote the distribution, not one median. When the 16 ms line was first crossed
the median read 15.97 while the middle half of the run medians straddled 16.0,
which is a claim that depends on which session is quoted. Today the whole range
is under 16.0: the slowest of six runs is 15.7911, so the margin is 0.21 ms at
the worst observed run rather than at the median. All six old runs and all four
Crossroads runs produced one stdout hash per capture.

600-frame offscreen sample sweep at the same default, per sample
(`/tmp/perf16/iter28-bench.log`; `renderheadless` drives its own frames and has
no benchmark row):

| sample | ms/frame | sample | ms/frame |
|---|---:|---|---:|
| gltfscenerendering | 3.90 | texturemipmapgen | 0.74 |
| multithreading | 2.84 | vulkanscene | 0.60 |
| instancing | 2.06 | pbribl | 0.52 |
| particlesystem | 1.76 | texturecubemap | 0.39 |
| bloom | 1.10 | computeshader | 0.31 |
| multisampling | 0.97 | dynamicuniformbuffer | 0.23 |
| texture3d | 0.14 | pushconstants | 0.13 |
| negativeviewportheight | 0.14 | texture | 0.11 |
| | | triangle | 0.09 |

The hot sum is **16.03 ms**, against 22.59 ms before the hardware texture path
became the default and 30.44 ms at the baseline of this work.

Untraced GPU busy, `nvidia-smi` at 10 Hz with no profiler attached: **74% on
old, 59% on Crossroads** (`/tmp/perf16/iter28-profile/report.md`). A busy
percentage only says a kernel was resident; §5.2 says what the rest of the frame
is doing.

### What each default costs to turn off

Each row was measured at its own tree, one binary, both arms, identical stdout
hashes. They do not sum to today's number.

| default | revert switch | old | Crossroads |
|---|---|---:|---:|
| hardware texture cache | `CUDAVK_NO_TEXTURE_CACHE` | 15.8194 / 15.7279 → 21.9152 / 21.9127 | 5.8737 / 5.8801 → 7.1203 |
| four small-operation removals | `CUDAVK_NO_META_FOLD`, `CUDAVK_NO_FETCH_FOLD`, `CUDAVK_NO_UPLOAD_COALESCE`, `CUDAVK_NO_COUNTER_BLOCK` | 16.5021 → 17.4331 (+0.9310) | 6.0467 → 6.2336 (+0.1868) |
| vertex fetch fused into the vertex shader | `CUDAVK_NO_FUSED_VFETCH` | 16.0194 → 16.5119 (+0.4925) | 5.9928 → 6.0715 (+0.0787) |
| A-buffer scan and quad fusions | `CUDAVK_NO_ABUF_FUSE_SCAN`, `CUDAVK_NO_ABUF_FUSE_QUAD` | 15.7488 → 15.9696 (+0.2208) | 5.8777 → 5.9685 (+0.0907) |

Every switch is in the registry (`../../src/cudavk/FLAGS.md`, 97 entries), and
each of these reverts restores its old path exactly.

The texture-cache row is the largest single lever in the driver, and it is also
the newest default. It was opt-in through iterations 24 to 28 and every number
in this document from iteration 24 on was taken with it enabled.

---

## 2. How a number in this document is measured

- **A frame is two `vkQueueSubmit` events.** The median is
  `median(diff(submit_ts[::2])[50:])` — pairs, hot tail, skip 50. A plain
  difference over all submits measures the intra-frame gap too and understates
  the frame by about 40%.
- **The skip matters.** The same replay restated from skip 100 to skip 50 moved
  an old-capture baseline from 17.41 to 17.5146 ms
  (`/tmp/perf16/iter28-profile/report.md`). Numbers are comparable only under
  one convention.
- **A control belongs in the same session as its candidate.** Two AB/BA runs per
  arm have a spread as wide as a whole iteration's win. Strictly alternating
  arms in one session (`cp_decisive_ab.sh`) is what settles an "is this over
  the line" question; `cp_two_replay_ab.sh` is enough for "did it help".
- **For a default-on change, put the reverts on the control arm.** The harness
  calls a positive delta a win, so reverts in the candidate arm give correct
  medians under an inverted verdict.
- **The old capture is not deterministic and that is not a regression.** Two
  runs of one binary with one set of flags differ on six of ten sentinel
  frames, by a mean absolute delta of 0.00003 and at most 13/255 on a few
  pixels — about a thousandth of the llvmpipe envelope that already passes.
  Crossroads is byte-identical run to run. Compare old against the llvmpipe
  envelope, not against its own previous frames.
- **Any probe whose stdout hash changes is invalid until the submit count and
  frame count are checked.** A run that dies early produces a fast, meaningless
  median: one iteration-29 probe first read as +8.07 ms and was a device loss
  after 26 frames (52 submits against 3,022).

Three profiler-free instruments produce most of the counts in this document,
and they are the right tools when a profiler would change the thing being
measured: `CUDAVK_UPLOAD_STATS=1` (every small copy, clear and context sync
attributed to its call site, plus a launch counter), `CUDAVK_PLAN_STATS=1`
(every main-thread wait, timed) and `CUDAVK_SHADER_STATS=1` (per-shader
registers, spill, blocks per SM and vertex-launch weight).

The two replay harnesses (`cp_two_replay_ab.sh`, `cp_decisive_ab.sh`,
`cp_two_replay_report.py`) live under `/tmp/perf16/`, not in the tree. The
sweep, comparison and profiling tools are in `src/cudavk/tests/`.

---

## 3. How it got here

Old-capture paired-submit median, one row per iteration that moved it.

| stage | old median (ms) | what it bought |
|---|---:|---|
| baseline `5ebc37aa36e` | 24.53 | 1,789 launches/frame at 3.5 µs median, GPU 85–96% GR-active but starved (SM issue 3–7%) |
| iteration 1 — per-variant register-cap trials | 24.51 | neutral here; kept because it is self-selecting and removes the "variants are 87% untuned" blind spot |
| iteration 2 — interpolate fused into the generated FS | 24.16 | three launches per direct shade become two, `fs_in` round trip goes intra-kernel |
| iteration 3 — clipping fused into raster stage 1 | 23.55 | one launch per batch removed; stage2 fusion rejected in the same iteration |
| iteration 10 — zero-copy primitive references | 23.255 | trivially accepted clipping stops copying primitives |
| iteration 11 — huge-primitive setup cached across tiles | 23.217 | setup recomputation removed from stage 3 |
| **standing before the texture cache** | **23.07** | measured in iteration 24 as its cache-off arm (23.0296 / 23.1026, average 23.0661) |
| iteration 24 — epoch-coherent hardware texture cache | 17.2279 | **−5.84 ms**: CUDA arrays and texture objects instead of the software sampler, 99.9% of old launches covered |
| iteration 26 — the sub-4 KB device operations | 16.5021 | **+0.93 ms**: small operations per frame 1,868 → 1,072 (−43%) |
| iteration 27 — vertex fetch fused into the vertex shader | 16.0194 | **+0.49 ms**: −194.3 launches, −194.3 clears (−84.9 MB), −192.0 copies per frame |
| iteration 28 — A-buffer support chain in two static fusions | **15.7514** | **+0.22 ms**: −91.1 launches and −27.4 clears per frame, −49.9 MB of clear traffic |

Iteration 24 was kept opt-in and became the default later; iteration 14's
same-LLVM fragment architecture is still opt-in (`CUDAVK_INLINE_FS`). Eighteen
iterations — 4 to 9, 12, 13, 15 to 23 and 25 — were rejected and reverted, and
they are the more useful half of the record: what was tried, what it measured
and why it was dropped are in `DEAD_ENDS.md`, with the full text in
`history/PERF16_ITERATIONS.md` and one row each in `iterations.json`.

---

## 4. What it costs to remove work

These prices are measured on this driver, on both captures, and they are the
only sound basis for forecasting a change here. A model that is not one of these
has never survived contact with a measurement.

| operation removed | measured price | where it was measured |
|---|---|---|
| small copy, removed outright | **1.30 µs** | iteration 26 S1: +0.2651 ms over 203.7 copies on old, +0.0654 ms over 50.3 on Crossroads — the same price on two workloads whose batch counts differ fourfold |
| copies merged at an unchanged boundary | **0.59 µs** | iteration 26 S3: +0.2979 ms over 502.5 copies on old |
| small clear | **0.66–1.01 µs** | iteration 26 S2 (0.66 old, 0.75 Crossroads) and S4 (0.16 old, 1.01 Crossroads) |
| bare same-stream launch | **under 1 µs** (take 0.6–1.0) | iteration 27: 0.512–0.529 ms measured, of which 0.241 ms is priced clears and merged copies and 0.065 ms is 84.9 MB never written, leaving 0.13–0.21 ms for 194.3 launches |
| device operation that also carries bandwidth or a whole pass | **about 1.86 µs** | iteration 28 item 4: 0.2208 ms over 91.1 launches and 27.4 clears, whose clears carried 49.9 MB/frame and whose compaction pass over 230,400 blocks stopped running |
| idle 256-thread block | **under 0.2 ns** | iteration 27 grid sweep: 306,686 fewer blocks scheduled per frame cost less than 0.06 ms |

Three rules come with the table:

1. **Removing an operation outright pays about twice what merging operations at
   an unchanged boundary pays.** 1.30 µs against 0.59 µs. Removing the boundary
   is the lever; flushing more cleverly is not — 76% of the driver's 1,785 flush
   points per frame already carry nothing.
2. **Do not forecast a fusion with the bare launch price when it also deletes a
   large clear or a whole pass.** That under-counts by about a factor of two.
   The same price over-counts a fusion that only deletes a launch.
3. **Per-fusion forecasts are not additive when the fusions overlap.** In
   iteration 28 item 4, S1 alone did not resolve (1.1 σ) and S2 alone paid
   +0.105 ms, but together they paid +0.221 ms while removing *fewer* launches
   than the halves sum to (91.1 against 59.6 + 47.3). What paid was making the
   episode's whole support chain a short static sequence, which neither half
   does alone. Iteration 26's removals *were* additive, because they were
   independent. An earlier apparent sub-additivity was control drift between
   sessions, not overlap.

A fourth rule is about the two captures, not about operations: **Crossroads is
overhead-bound where old is work-bound.** Both run a similar number of episodes
per frame (23.0 old, 16.5 Crossroads) but a Crossroads frame is 2.8× shorter, so
a fixed per-episode cost is nearly three times the share of its frame. Fixed-cost
removals keep favouring Crossroads; launch-count and fragment-side work keep
favouring old.

---

## 5. Where the frame goes today

From `/tmp/perf16/iter28-profile/report.md`, taken at `ba8891878df` — that is
before iteration 28 item 4 landed, so the frame is 15.99 ms there and 15.75 ms
now, and the launch counts below are 91.1/frame higher than today's. Everything
else in this section is unchanged by that iteration.

### 5.1 Kernel time by class, old capture

Sum of kernel durations. The sum (14.52 ms/frame) is above the union-busy figure
(11.97 ms/frame) because the driver uses nine streams and some work overlaps.
Read the column as a share, not as a budget.

| class | launches/frame | ms/frame | share |
|---|---:|---:|---:|
| raster stage3 abuf | 134.6 | 2.484 | 17.1% |
| raster stage1 (clip+s1 fused, direct) | 74.5 | 1.894 | 13.0% |
| VS main (fetch now fused in) | 221.5 | 1.843 | 12.7% |
| raster stage1 (clip+s1 fused, abuf) | 134.6 | 1.461 | 10.1% |
| FS main (A-buffer) | 22.3 | 1.064 | 7.3% |
| FS main (direct, fixed grid) | 74.5 | 0.998 | 6.9% |
| raster stage2 abuf | 134.6 | 0.796 | 5.5% |
| raster stage3 direct | 74.5 | 0.767 | 5.3% |
| raster stage2 direct | 74.5 | 0.547 | 3.8% |
| abuf quad | 32.6 | 0.441 | 3.0% |
| abuf composite | 14.2 | 0.425 | 2.9% |
| fs_compact (interpolation/compaction) | 74.5 | 0.368 | 2.5% |
| vertex_fetch (declining shaders only) | 10.4 | 0.350 | 2.4% |
| abuf sort | 28.7 | 0.347 | 2.4% |
| abuf scan | 128.2 | 0.211 | 1.5% |
| everything else (seg, writeback, fill_recs, standalone clip, worklist, clears, texture convert) | 154.4 | 0.526 | 3.6% |
| **kernels** | **1,389** | **14.52** | 100% |
| memcpy | 526 | 1.125 | — |
| memset | 240 | 0.190 | — |
| **all device operations** | **2,155** | **15.84** | — |

- **The raster chain is still the frame.** stage1+2+3 across both paths is
  7.95 ms/frame in 627 launches/frame, 54.7% of all kernel time.
  `cp_rasterize_stage3_abuf` alone is 2.48 ms/frame and is the largest single
  class.
- **Fragment shading is 2.06 ms/frame in 96.8 launches.**
- **A-buffer support** (scan, sort, worklist, quad, fill_recs, seg, composite)
  was 1.72 ms/frame in 266 launches/frame at this commit; iteration 28 item 4
  then removed 91.1 launches/frame from it.
- Device operations per frame fell from 4,033 to 2,155 (−47%) between the
  iteration-25 profile and this one: memcpy 1,392 → 526, memset 1,016 → 240,
  kernels 1,625 → 1,389.

Crossroads has the same shape at a smaller scale: 439 kernels/frame, 3.157 ms of
kernel time, raster chain 1.63 ms of that (51.6%), 815 device operations.

### 5.2 The finding that matters most: it is a ping-pong, not a pipeline

| old capture, per frame | ms | share |
|---|---:|---:|
| frame | 15.99 | 100% |
| host blocked in a device wait | **12.44** | 77.8% |
| host issuing (frame − blocked) | 3.55 | 22.2% |
| device resident | 11.83 | 74.0% |
| **device idle** | **4.16** | **26.0%** |

**The device idle is almost exactly the host's own command-issue time** — 4.16
against 3.55 on old, 2.45 against 3.39 on Crossroads. The driver blocks about
seventeen times a frame, so it cannot issue frame N+1's work while frame N is
running. The host issues a burst with the device empty, then blocks while the
device drains.

Where the host blocks, measured with `CUDAVK_PLAN_STATS=1` — host timers, no
profiler, so this instrument does not manufacture the cost it measures:

| wait | ms/frame | waits/frame | mean |
|---|---:|---:|---:|
| episode drain (`cp_renderer.c:8008`) | 8.663 | 9.88 | 0.877 ms |
| peel checks (`cp_renderer.c:6342`) | 2.765 | 1.71 | **1.620 ms** |
| segment counters (`cp_renderer.c:6073`) | 0.995 | 4.17 | 0.239 ms |
| descriptor uploads | 0.011 | 1.00 | 0.011 ms |
| **total** | **12.435** | **16.76** | |

Plus `vkDeviceWaitIdle` → `cuCtxSynchronize` 2.24 times a frame and one blocking
`cuMemcpyHtoD` of the descriptor arena per submit. Crossroads blocks 2.580
ms/frame over 8.26 waits. **Every one of these waits is a read-back-and-decide**:
the drain copies six counters plus the per-segment quad counts to the host and
branches on overflow and coverage.

The trace agrees about the shape. Gaps on the union of kernels, copies and
clears: 2.20 gaps per frame over 200 µs hold 59.5% of the traced idle, and both
sit at the frame boundary bounded by a memcpy — the inter-submit stall (median
2.453 ms) and the end-of-frame drain and readback (median 0.464 ms).

### 5.3 How much is left in removing operations

Using the driver's own prices from §4 rather than a model:

- The realistic fusion programme that existed at this profile was the A-buffer
  support chain plus the fragment writeback. **The A-buffer half is done** and
  paid 0.2208 ms. The writeback half is 43.0 reachable launches/frame, worth
  about 0.04 ms at the measured launch price.
- The **theoretical ceiling** — every remaining launch, copy and small clear
  gone, which would require a driver that issues nothing — is
  1.39 + 0.55 + 0.14 = **2.08 ms, or 50% of the 4.16 ms idle**.

**So the operation-removal programme that iterations 26, 27 and 28 ran is close
to exhausted.** What is left of the idle is reachable only by removing or
overlapping the seventeen host↔device round trips a frame, and §6 says what is
known about that.

---

## 6. What is left

Ranked by value. "Supported" means a measurement in this project points at the
number; "estimate" means it is a forecast from the price table and has not been
measured.

### 1. Peel checks — 0.20–0.50 ms, risk medium, estimate

2.765 ms/frame of blocked host over only 1.71 waits, at **1.620 ms each** — the
most expensive single wait in the driver, and the best ratio of value to blast
radius of the three synchronisation sites. The design
(`/tmp/perf16/iter29-sync/design.md` §5) predicates a whole peel pass on the
device flag the rasterizer already honours, moves the interval reset to a
one-thread kernel so no host store races it, and predicts the trip count instead
of asking. Its correctness argument is already written in the driver's own
comment: once a pass composites nothing, every later pass does too, and a pass
after convergence is already a no-op for output. The overshoot must be capped —
budget about 8 launches per wasted pass, and an unbounded overshoot on a
256-layer draw would add about 1.8 ms and lose. *Supported*: the wait cost and
its distribution. *Estimate*: the frame-time gain.

### 2. Widen the `bounded` fast path — 0.10–0.25 ms, risk medium, estimate

4.17 waits/frame, 0.995 ms/frame, mean 0.239 ms. The fast path already exists
and is already correct: when the driver can bound `nblocks × triangles` it skips
the drain and uses a computed bound. Bounding by the clip rectangle instead of
by the whole framebuffer is exact, host-computed and often one or two orders of
magnitude tighter than 230,400 blocks. Over-estimating costs slots and idle
blocks, never a wrong pixel, and an idle block is under 0.2 ns. *Supported*: the
wait cost, and the bound/actual ratio being the expensive quantity (see item 4).
*Estimate*: the gain. **Raising the `512u << 10` slot budget is withdrawn** —
that constant caps over-allocation cost.

### 3. Fuse `cp_fs_writeback` into the fragment shader — about 0.04 ms, risk low but currently broken

74.5 launches/frame and 0.127 ms/frame of device time, of which 43.0
launches/frame are reachable. Admission is **A = 1.0000 by launch** on the old
capture, 92 → 94 registers, no local memory, suite 65/65 with the flag on. The
patch exists (`/tmp/perf16/iter28-item3/`). It is parked because the old capture
renders visibly wrong frames with it — whole background quads missing — and
because the invariant the design rests on is **measured false**: 6,343,098
second claimants of a pixel inside one launch. Note that this measured 0.04 ms
replaces the 0.10–0.20 ms the earlier profile estimated for it. *Supported*: the
admission, the reachable share and the failure. **Any later work that assumes a
direct-path pixel is written once per launch is assuming something this capture
disproves six million times.**

### 4. Let the host run past the episode drain — value unknown, risk high

The largest wait by total time (8.663 ms/frame over 9.88 waits) and therefore
the obvious target, but the obvious route into it is **refuted by measurement**.
`CUDAVK_UNSAFE_NO_OVERFLOW=1`, which removes the wait by sizing for the worst
case, made Crossroads **0.38 ms slower (−6.37%)** with a byte-identical hash and
killed the old capture after about 26 frames: worst-case sizing asks for
8,605,856,768 bytes against an 8,589,934,592-byte scratch cap, the allocation is
refused, and the no-replay rule correctly latches device loss. The mechanism is
that a request the arena cannot serve reallocates the whole arena and frees the
old base at every flush, so worst-case sizing buys gigabyte-scale
allocate/free churn and bound-sized bulk clears.

**The episode drain pays for itself**: its 0.877 ms buys the exact quad count
that sizes the shade arrays, and the bounded path's worst-case bound is not a
cheaper way to get the same thing. Any variant that routes episodes through the
bounded path inherits this, because the disease is the ratio bound/actual and
not the wait. The earlier 0.5–1.5 ms estimate came from a serialisation model,
and the only direct test of that model produced a negative number; treat the
model as unsupported, not merely unproven.

Two instrumentation-only probes decide what is left here, and both are one
`fprintf` (`/tmp/perf16/iter29-sync/addendum.md` §5):

- **P1** — log `total/quads` per episode at the existing drain. If the median
  ratio is small (say ≲ 3), sizing the shade arrays from the scan's exact
  fragment total, which the driver already computes near the *start* of the
  chain, is worth building. If it is large, every bound-based sizing scheme is
  dead by the mechanism above.
- **P2** — time the host's inter-drain issue burst. A deferral mechanism can
  recover at most `min(burst, wait)` per episode, so this is an exact ceiling on
  it before anything is written.

Do these before designing anything at this site.

### 5. Small cleanups — claim 0.00 ms

Emptying the drain of the work that does not need the host removes about 9.9
copies/frame and some host-side loops, and adds about 9.9 launches. Land it on
tidiness or not at all.

---

## 7. What is closed

Short list; `DEAD_ENDS.md` has the evidence and the reasons.

- **Grid tuning is closed.** Sizing the direct fragment grid to the machine
  instead of to the framebuffer reaches the launches it aims at — 306,686 fewer
  blocks scheduled per frame at `CUDAVK_FS_GRID_WAVES=1` — and the frame does
  not move. That bounds an idle 256-thread block at under 0.2 ns, so a
  4,096-block grid carries at most 0.82 µs of empty-block cost and 74.5
  launches/frame is at most 0.061 ms. An earlier 0.15–0.25 ms estimate for this
  is retracted. The flag stays at its byte-identical default so the measurement
  can be repeated on other hardware in one command.
- **Kernel-internal raster redesign is closed.** `cp_rasterize_stage3_abuf` is
  still the largest kernel class at 2.48 ms/frame, and NCU measured
  `launch__waves_per_multiprocessor` at median 0.15 and maximum 0.60 over 270
  launches, with 0 of 270 reaching 1.0. It is not a kernel-internal problem.
  Two whole-scheduler rewrites (persistent machine-scaled stage2/3, and the
  episode-global tagged stage 3) were built, measured and reverted.
- **Further copy coalescing is exhausted.** All 423.3 copies/frame come from one
  site, already one per launch boundary. They fall only when boundaries fall,
  and merging at an unchanged boundary pays 0.59 µs against 1.30 µs for removal.
- **Busy is not working.** The GPU can read 94% GR-active while issuing
  instructions on 5% of cycles. Ask `src/cudavk/tests/cp_gpu_busy.sh` whether a
  frame is host-bound before reaching for a profiler, and remember that CUPTI
  adds host-side cost to every launch in a driver that issues over a thousand
  a frame.
