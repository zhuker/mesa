# Performance

Where the driver's time goes today, what it costs to remove different kinds of
work, and what is left. Read this before choosing what to optimise next.

Every number here comes from a measurement that is named beside it. The long
form of each one is in `history/PERF16_ITERATIONS.md` (the iteration record) and
in the raw evidence under `/tmp/perf16/`. The 2026-08-26/27 measurement session
is archived **in the tree**, 52 reports plus its reusable instruments, at
`history/perf-2026-08-27/`; entries below cite it by filename, and
`history/perf-2026-08-27/SESSION_HANDOFF.md` is its consolidated form. Approaches that are closed are in
`DEAD_ENDS.md`; open questions are in `TODO.md`; how to run a measurement is in
`WORKFLOW.md` and `TESTING.md`.

Two GFXR captures carry the work:

| name used here | file |
|---|---|
| **old capture** | `headless_streamer_20260814T155742.gfxr` |
| **Crossroads** | `headless_streamer_1818_20260817T173522.gfxr` |

---

## 1. Where the driver is today

**Today's shipping default is old 12.7826 ms and Crossroads 5.6936 ms**, with
programmatic dependent launch landed and on by default at level 3
(`6e6e00968ca`; level 4 is offered as a diagnostic in `b9766f720a4`). Those are
the candidate arm of the decisive PDL run on the final tip `b9766f720a4` — 6
runs per arm on old and 4 on Crossroads, all 20 full length at 3,022 / 2,994
submits, one stdout hash per capture across both arms, arms non-overlapping, IQR
[12.7632, 12.8075] and [5.6868, 5.7007]
(`history/perf-2026-08-27/pdl_landing.md`).

**The two measurements below are the pre-PDL baseline** at `e2fea470d04`, and
they are kept because §3, §5, §5.2b and every census in this document are
anchored to them. They are directly comparable to the figure above: the PDL
run's own control arm reads **13.1641** against the 13.1626 measured here in a
different session hours earlier — **1.5 µs apart.**

The pre-PDL driver, measured alone — no arms, no flags beyond the base
environment, one session, three replays of each capture at `e2fea470d04`
(`history/perf-2026-08-27/reprofile_baseline.md`):

| capture | run medians (ms/frame) | median | IQR | full range |
|---|---|---:|---|---|
| old | 13.1626, 13.1116, 13.2302 | **13.1626** | [13.1371, 13.1964] | [13.1116, 13.2302] |
| Crossroads | 5.8165, 5.8509, 5.8230 | **5.8230** | [5.8197, 5.8370] | [5.8165, 5.8509] |

Quote the distribution, not one median. All three old runs and all three
Crossroads runs completed the capture — 3,022 and 2,994 submits — and produced
one stdout hash per capture.

**These numbers are 2.59 ms and 0.08 ms below the ones this document carried
until `20611f5b131`, and the whole difference is that commit** — the
opaque-episode fan-out becoming the default. Measured in the same session on
the same binary, `CUDAVK_NO_OPAQUE_STREAMS=1` gives 15.8437 / 15.9427 on old
and 5.9001 / 5.8985 on Crossroads, i.e. **+2.7306 ms (−17.2%) on old and
+0.0763 ms (−1.3%) on Crossroads**, with the stdout hash unchanged between the
arms. The previous distribution — old 15.7514, IQR [15.7305, 15.7596];
Crossroads 5.8982, IQR [5.8877, 5.8990] — is now the *reverted* arm's
distribution and is kept here because §5.2 is validated against it.

The asymmetry is the mechanism: an opaque episode averages 12.42 segments on
old and 1.71 on Crossroads, so one capture has segments to overlap and the
other does not.

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
became the default and 30.44 ms at the baseline of this work. That sweep
predates `20611f5b131`, which reports the win concentrated where the segments
are — `multithreading` 2.83 → 2.37 ms, −16.3% — with no sample moving outside
noise the other way.

Untraced GPU busy, `nvidia-smi` at 10 Hz with no profiler attached: **72% on
old (IQR 71–73) and 59% on Crossroads** at `e2fea470d04`
(`/tmp/perf-audit/reprofile_stats.md`); the reverted arm reads 75% and the
earlier profile read 74% on old, 59% on Crossroads
(`/tmp/perf16/iter28-profile/report.md`). A busy percentage only says a kernel
was resident; §5.2 says what the rest of the frame is doing.

### What each default costs to turn off

Each row was measured at its own tree, one binary, both arms, identical stdout
hashes. They do not sum to today's number.

| default | revert switch | old | Crossroads |
|---|---|---:|---:|
| hardware texture cache | `CUDAVK_NO_TEXTURE_CACHE` | 15.8194 / 15.7279 → 21.9152 / 21.9127 | 5.8737 / 5.8801 → 7.1203 |
| four small-operation removals | `CUDAVK_NO_META_FOLD`, `CUDAVK_NO_FETCH_FOLD`, `CUDAVK_NO_UPLOAD_COALESCE`, `CUDAVK_NO_COUNTER_BLOCK` | 16.5021 → 17.4331 (+0.9310) | 6.0467 → 6.2336 (+0.1868) |
| vertex fetch fused into the vertex shader | `CUDAVK_NO_FUSED_VFETCH` | 16.0194 → 16.5119 (+0.4925) | 5.9928 → 6.0715 (+0.0787) |
| A-buffer scan and quad fusions | `CUDAVK_NO_ABUF_FUSE_SCAN`, `CUDAVK_NO_ABUF_FUSE_QUAD` | 15.7488 → 15.9696 (+0.2208) | 5.8777 → 5.9685 (+0.0907) |
| programmatic dependent launch | `CUDAVK_NO_PDL` | 12.7826 → 13.1641 (+0.3815) | 5.6936 → 5.8458 (+0.1522) |
| host-pinned readback pages | `CUDAVK_NO_HOST_PIN_READBACK` | favorite3 6.6757 → 7.6048 (+0.9291), favorite2 5.4800 → 6.5433 (+1.0633) | B200: favorite3 10.789 → 12.188 (+1.399), favorite2 8.967 → 10.566 (+1.599) |
| opaque episode gate recorded once | `CUDAVK_NO_EPISODE_GATE_ONCE` | favorite3 6.3535 2192 6.4580 (+0.1045), favorite2 5.1209 2192 5.2393 (+0.1184) | not yet measured (no B200 access) |
| refusing un-appendable blended batches | `CUDAVK_NO_APPEND_PREFILTER` | favorite3 6.4465 → 6.6881 (+0.2416), favorite2 5.2330 → 5.4685 (+0.2355) | not yet measured (no B200 access) |


### What it is worth against the software rasterizer

Same tree, same captures, same hosts, **both drivers built release**
(`-Dbuildtype=release -Db_ndebug=true`); paired-submit medians on each
capture's own window (4.0), full submit populations on every run.

| capture | window | cudavk | llvmpipe | speedup |
|---|---|---:|---:|---:|
| favorite3 | relevant (1391+) | 6.679 | 46.40 | **6.9x** |
| favorite3 | heavy (2200-3150) | 8.318 | 54.02 | 6.5x |
| favorite2 | relevant (1388+) | 5.487 | 39.72 | **7.2x** |
| favorite2 | heavy (2735+) | 6.054 | 39.90 | 6.6x |

RTX 5090 against a 32-thread Ryzen 9 9950X3D. On the B200 host (152-thread
Xeon Platinum 8559C) llvmpipe release is **66.19 / 56.41** ms on the two
captures against cudavk's 10.79 / 8.97 - **6.1x / 6.3x**. More cores did not
help it: that host's llvmpipe is *slower* than the desktop's despite 4.75x the
threads, so this workload is single-thread-bound in the software rasterizer
much as it is latency-bound on the GPU.

The ratio is flat across captures, bands and hosts, which says both
implementations are limited by the same shape of work rather than by different
bottlenecks - consistent with dead end 36, where the time is vertex/fragment
arithmetic that no scheduler removes.

**Build type matters for llvmpipe and not for cudavk**, which is worth knowing
before quoting either. Alternating two-round A/B, same session:

| | debugoptimized | release | delta |
|---|---:|---:|---:|
| cudavk favorite3 | 6.6799 | 6.6791 | -0.0008 (noise) |
| cudavk favorite2 | 5.4849 | 5.4871 | +0.0022 (noise) |
| llvmpipe favorite3 | 50.09 | 46.40 | **-7.4%** |
| llvmpipe favorite2 | 42.53 | 39.72 | **-6.6%** |

cudavk is indifferent because its frame is device time and CUDA API calls, not
host arithmetic - the shipping `debugoptimized` configuration in
`GETTING_STARTED.md` costs nothing. llvmpipe rasterizes on the CPU, so its
asserts are in the hot path. Rendering was not compared pixel-wise here; the
replays' stdout differs only in memory-type remapping warnings.

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
- **A census produces MEAN-based per-frame figures; this project's frame is a
  hot-tail MEDIAN.** An instrument that sums over a whole replay and divides by
  a frame count includes start-up, shader compilation and teardown. On the old
  capture the two differ by 54%: the mean paired-submit interval is **20.305
  ms** and the skip-50 median is **13.163 ms** — both of the census's own
  pre-PDL replay, which is the pair to compare — because the largest single
  interval in the replay is 1,781 ms of shader compilation. Checked, not
  assumed: §5.2b's `blocked + issued = 20.33 ms/frame` lands on the 20.305 ms
  wall span to 0.1%, which is one thread accounting for all of its own time —
  two threads would have summed *above* the span. So **ratios out of a census
  are comparable to anything** (they divide two sums over the same interval),
  and **per-frame figures out of a census are not comparable to a median frame
  time unless that is said out loud**.
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
| iteration 28 — A-buffer support chain in two static fusions | 15.7514 | **+0.22 ms**: −91.1 launches and −27.4 clears per frame, −49.9 MB of clear traffic |
| `20611f5b131` — opaque episode segment fan-out made the default | **13.1626** | **+2.73 ms**: no operation removed at all — the episode drain's mean wait falls 0.874 → 0.606 ms because an episode's 12.42 segments now overlap on the side streams. Crossroads gains 0.08 ms, because its episodes average 1.71 segments |
| `6e6e00968ca` — programmatic dependent launch on by default at level 3 | **12.7826** | **+0.38 ms**: again no operation removed — a dependent kernel may start before its predecessor has drained. Crossroads gains 0.15 ms. Decisive instrument, p = 0.0011 / 0.0143 |

**The last two rows are the only ones in this table that bought their
milliseconds by overlapping work rather than by removing it**, and together they
are 3.11 ms — the second largest block in the table after the texture cache.
They are also why the earlier rows must never be quoted as today's frame:
anything above 12.7826 is a driver at least one mechanism out of date, and the
15.7514 that stood in this document until 2026-08-26 was two.

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
| **a real chain launch removed** | **0.78–0.81 µs** | 2026-08-27, three instruments (`history/perf-2026-08-27/wide_bench.md` §7): spread injection, frame slope over its linear region, **0.782 µs, 95% CI [0.633, 0.949]**; union idle in front of a real chain kernel, nsys over 955,301 kernels, **0.809 µs**; and the row above at 0.6–1.0 |
| **an exposed launch added** | **1.974 µs** | same session, same probe run bunched instead of spread: least squares over 15 points, 1.9745 µs/launch, against a back-to-back empty-kernel floor of 2.047 µs. Not one of 1.21 million injected launches had a kernel in front of it |
| **a launch removed from a population that was running concurrently** | **the credit is divided by the overlap factor, and can invert** | 2026-08-27: the wide raster merge removed 298.8 launches/frame and cost **0.410 ms/frame**, an implied −1.372 µs per launch removed. See the overlap factor below and `DEAD_ENDS.md` §22 |
| device operation that also carries bandwidth or a whole pass | **about 1.86 µs** | iteration 28 item 4: 0.2208 ms over 91.1 launches and 27.4 clears, whose clears carried 49.9 MB/frame and whose compaction pass over 230,400 blocks stopped running |
| idle 256-thread block | **under 0.2 ns** | iteration 27 grid sweep: 306,686 fewer blocks scheduled per frame cost less than 0.06 ms |
| kernel-to-kernel dependency **overlapped**, secondary with no preamble | **0.44 µs** (old), **0.84 µs** (Crossroads) | PDL level 2 against level 1: +0.1859 ms over 421.50 converted links on old, +0.0686 ms over 81.95 on Crossroads. Every secondary here reads the queue its predecessor filled as its first instruction, so this is the inter-grid gap alone |
| the same, secondary with one independent global load in front of the wait | **0.78 µs** (old), **1.32 µs** (Crossroads) | PDL level 3 against level 2: +0.0637 ms over 82.10 links on old, +0.0220 ms over 16.66 on Crossroads |
| the same, secondary with a whole clear hoisted in front of the wait | **3.02 µs** (old), **2.14 µs** (Crossroads) | PDL level 1 against level 0: +0.1430 ms over 47.28 links on old, +0.0570 ms over 26.67 on Crossroads. One link per episode overlaps a 3.7 MB fill-cursor clear that was moved ahead of the wait on purpose |
| **cross-stream hand-off (one gate in, one join out)** | **about 32 µs per episode** | `b3f716bd5f5`: moving segment 0 back to the main stream removed exactly one hand-off per episode and recovered 0.81 ms of the 600-frame sweep's hot sum, with bit-identical pixels and unchanged launch counts |

The hand-off price is what decides whether a fan-out is worth having on a given
capture, and it explains the two captures' answers arithmetically
(`/tmp/perf-audit/reprofile_stats.md` §B.6, counters only, no extra run):

* **old** fans **2.28 episodes/frame** onto side streams (3,443 over 1,511
  frames) = **0.073 ms/frame** of hand-off, against the **2.650 ms/frame** of
  episode-drain time the fan-out removes. The price is **2.8% of what it buys**.
* **Crossroads** fans **0.98 episodes/frame** (1,468 over 1,497) = **0.031
  ms/frame**, against a drain gain of **0.024 ms/frame**. The price is larger
  than the gain, which is why the capture whose episodes average 1.71 segments
  gains nothing measurable from the fan-out.

That is arithmetic on the published 32 µs, not a re-measurement of it. It is
worth carrying because it predicts the sign of a fan-out from two counters
(`opaque episodes: … ran on the side streams`) before anything is built.

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

A fourth rule comes from the three PDL rows, which price a dependency that is
*overlapped* rather than an operation that is *removed*: **what a converted
dependency is worth is decided by what the secondary can execute before its
wait, and that is a property of the kernel, not of the link.** The bare gap is
0.44 µs — a third of the 1.30 µs for removing a small copy outright, and below
the bare-launch price, which is what it should be, because the launch still
happens. One independent load ahead of the wait takes it to 0.78 µs; a hoisted
clear takes it to 3.02 µs, seven times the bare figure. So the first question
about a candidate link is not how often it fires but where `ACQBULK` lands in
its secondary's SASS, and the second is whether the kernel writes an array it
never reads, which is the one thing that can always be moved in front of the
wait.

A fifth rule is about the two captures, not about operations: **Crossroads is
overhead-bound where old is work-bound.** Both run a similar number of episodes
per frame (23.0 old, 16.5 Crossroads) but a Crossroads frame is 2.8× shorter, so
a fixed per-episode cost is nearly three times the share of its frame. Fixed-cost
removals keep favouring Crossroads; launch-count and fragment-side work keep
favouring old.

### The launch has two prices, and they are the same cost measured either side of the hiding

A sixth rule, and it decides how every launch-count lead in this document is
quoted. **0.78–0.81 µs is the price of REMOVING a real launch; 1.974 µs is the
price of ADDING an exposed one.** The front end costs about 2 µs per launch and
it is hidden when the kernel in front of it runs long: **74.9% of real chain
kernels start with zero union idle**, while the bunched injection put 1.21
million launches on an empty queue where nothing could hide them.

**Quote a launch-removal lead as the range 0.78–0.81 µs, never as the 1.974
floor and never as the 0.6 bottom edge of the older row.** The difference is not
academic — at 1.974 µs the launch axis would have led the 2026-08-27 ranking; at
0.782 it ranks behind the episode drain (554 legally mergeable launches × 0.782
= 0.433 ms/frame). An earlier reading of that run quoted 1.974 as *the* marginal
price; that reading is withdrawn by its own author in the same document.

### The overlap factor: the quantity that decides whether a merge pays at all

A count of launches is not a forecast until it is divided by how concurrent
those launches already are. **Overlap factor = summed kernel time ÷ union of
kernel intervals** over the population under study — 1.00 means strictly serial,
and the launch price applies in full; anything above 1 means the launches are
already running together and the credit is scaled by the serial fraction
`1/overlap`.

Measured on the shipping default (PDL 3, fan-out on), two nsys windows agreeing
to 2.5%, gated first against §5.1's 1,314.2 launches/frame
(`history/perf-2026-08-27/item4_countphase_overlap.md`):

| population | overlap factor |
|---|---:|
| `cp_rasterize_stage3_abuf` alone — 53% of the count-phase triple's time | **1.89×** |
| `cp_clip_rast_fused_abuf` alone | 1.44× |
| `cp_rasterize_stage2_abuf` alone | 1.22× |
| the three A-buffer count-phase stages, time-weighted | **1.59×** |
| the abuf count-phase triple taken as a group | 2.50× |
| the opaque/direct triple taken as a group | **2.68×** |
| all kernels | 1.80× |

Two readings that are easy to get wrong:

1. **Use the per-stage SELF-overlap, not the group's.** A merge of "the same
   stage across the segments of one episode" concatenates that stage's own
   population; it does not remove the pipelining between *different* stages, so
   the group figure flatters it. Here that is 1.59× time-weighted rather than
   2.50×.
2. **The scaled credit is the ceiling, not the answer.** Scaling the count-phase
   merge's 0.433 ms/frame by 1/1.59 gives about 0.16 ms — *before* subtracting
   the device cost of the merged form, which was measured to be larger than
   that. See `DEAD_ENDS.md` §22 for the rule this produced.

---

## 5. Where the frame goes today

§5.1 comes from `/tmp/perf16/iter28-profile/report.md`, taken at `ba8891878df`
— that is before iteration 28 item 4 and before the fan-out default, so the
frame is 15.99 ms there, 13.16 ms at `e2fea470d04` and **12.78 ms with PDL
landed**. §5.2 has been re-derived at `e2fea470d04` and states both arms. Every
share and every census in §5 is anchored to the 13.16 ms pre-PDL frame, so a
share taken from here and applied to today's 12.78 ms frame is out by 3%.

**The launch counts in §5.1 are stale by a known amount and no more.** The
driver's own counter reads **1,314.2 launches/frame on old and 346.0 on
Crossroads** at `e2fea470d04` (`CUDAVK_UPLOAD_STATS=1`), against §5.1's 1,389
and 439 kernels/frame — −74.8 and −93.0. The text below predicts −91.1 on old
from iteration 28 item 4 alone, so the two instruments differ by 16.2
launches/frame, about 1.2%, and one is a device trace while the other is the
driver's `CP_LAUNCH` counter. Read that as "no unexplained drift". Small
operations are 438.4 copies and 235.4 clears per frame on old, 118.2 and 100.8
on Crossroads. **The fan-out changed none of these counts** — launches, copies,
clears and context syncs are equal to the digit in both arms, which is the
check that it is a scheduling change and not a work change.

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

**A class's share of this table is NOT its share of the frame, and the gap is
large.** Only **16.7%** of `cp_rasterize_stage3_abuf`'s time is *exclusive* —
83.3% of it already has another kernel running. Shrinking every launch of it and
recomputing the union of all kernel intervals over a window checked against this
table (`history/perf-2026-08-27/tiling_ncu.md`):

| change | device busy removed |
|---|---:|
| `stage3_abuf` 2× faster | **0.187 ms/frame** |
| `stage3_abuf` 4× faster | 0.288 ms/frame |
| `stage3_abuf` **infinitely** fast | **0.404 ms/frame** |
| whole raster chain 2× faster | 0.763 ms/frame |
| whole raster chain **deleted** | **2.106 ms/frame** |

The raster chain is 54.7% of kernel time in the table above, and **deleting all
of it returns 2.1 ms of device busy time** — and device time is not frame time,
because §5.2 has the host blocked 73.7% of the frame and the site's own slope
decides what a device saving is worth.

**So size a kernel-side lead as `class time × exclusive fraction`, then apply
the site's measured slope. Never from the share in this table.** This is the
generalisable result of the 2026-08-27 tiling audit, and it is why entry 24 of
`DEAD_ENDS.md` refuses a mechanism whose whole best case is 2.1–2.6 ms.

### 5.2 The finding that matters most: it is a ping-pong, not a pipeline

| old capture, per frame | ms | share | reverted arm | earlier profile |
|---|---:|---:|---:|---:|
| frame | 13.21 | 100% | 15.77 | 15.99 |
| host blocked in a device wait | **9.73** | 73.7% | 12.41 | 12.44 |
| host issuing (frame − blocked) | 3.48 | 26.3% | 3.36 | 3.55 |
| device resident (72% busy) | 9.51 | 72% | 11.84 | 11.83 |
| **device idle** | **3.70** | **28%** | 3.94 | 4.16 |

**The device idle is still almost exactly the host's own command-issue time** —
3.70 against 3.48 on old. The shape §5.2 has always described is unchanged; only
its size shrank. The driver still blocks **16.76 times a frame**, so it still
cannot issue frame N+1's work while frame N is running.

Where the host blocks, measured with `CUDAVK_PLAN_STATS=1` — host timers, no
profiler, so this instrument does not manufacture the cost it measures. Line
numbers are at `e2fea470d04`; the reverted column is
`CUDAVK_NO_OPAQUE_STREAMS=1` in the same session
(`/tmp/perf-audit/reprofile_stats.md`):

| wait | ms/frame | waits/frame | mean | reverted: ms/frame, mean |
|---|---:|---:|---:|---:|
| episode drain (`cp_renderer.c:8443`) | **5.989** | 9.88 | 0.606 ms | 8.639, 0.874 ms |
| peel checks (`cp_renderer.c:6689`) | 2.751 | 1.70 | **1.614 ms** | 2.759, 1.617 ms |
| segment counters (`cp_renderer.c:6420`) | 0.982 | 4.17 | 0.235 ms | 0.998, 0.239 ms |
| descriptor uploads | 0.013 | 1.00 | 0.013 ms | 0.012, 0.012 ms |
| **total** | **9.734** | **16.76** | | **12.407**, 16.76 |

**The reverted column reproduces the figures this document carried before the
fan-out — 12.407 against 12.435 ms over the same 16.76 waits, every site within
0.02 ms — which is the check that the instrument still means what it says.**

**Only one site moved.** The fan-out took 2.650 ms/frame out of the episode
drain at an *unchanged* 9.88 waits per frame: the mean wait fell from 0.874 to
0.606 ms because an episode's segments now overlap on the side streams and the
episode finishes sooner. Peel checks, segment counters and descriptor uploads
each moved by less than 0.02 ms, inside their own run-to-run noise. The frame
moved −2.55 ms and the drain moved −2.65 ms, so the attribution is complete.

Plus `vkDeviceWaitIdle` → `cuCtxSynchronize` 2.24 times a frame and one blocking
`cuMemcpyHtoD` of the descriptor arena per submit. **Crossroads blocks 2.563
ms/frame over 8.26 waits** — unchanged by the fan-out (2.591 reverted) and equal
to the earlier figure of 2.580 — of which the episode drain is 2.129 ms over
5.57 waits and the segment counters 0.428 ms over 1.69. **Crossroads runs no
peel checks at all**: that counter is zero on every run, so §6 item 1 was an
old-capture-only lead before it was closed.

**Every one of these waits is a read-back-and-decide**: the drain copies six
counters plus the per-segment quad counts to the host and branches on overflow
and coverage. One of those branches is redundant: **the tail drain's
`quad_over` term cannot fire when `fill_over == 0`**, because
`quad_capacity == capacity` and `quads ≤ total`, so a clear fragment-overflow
flag already proves the quad array did not overflow.

**A wait can be spent on nothing, and it cannot be skipped.**
`CUDAVK_DRAIN_PROBE=1` records the pair the drain reads: **6.9% of old-capture
drains and 18.3% of Crossroads drains return `quads == 0`** — the episode
covered nothing. On old that is 0.68 drains and **0.053 ms/frame**, 0.9% of the
site's blocked time. On Crossroads it is 1.02 drains and **0.322 ms/frame**,
**15.1%** of that capture's blocked time, and an empty drain there costs almost
as much as a productive one (0.315 against 0.397 ms mean).

**No mechanism collects this, and it is not a lead.** The drain is taken for
`fill_over` and `quad_over`; `quads` only rides along in the same copy, and
`cp_pass_can_retry` forbids running the fragment shader before the overflow
answer arrives, so an oracle for `quads == 0` would save no wait. The numbers
are recorded because the one design that would attack them is the addendum's
S1d — drain at the scan instead of at the tail — which is a different piece of
work.

The trace agrees about the shape. Gaps on the union of kernels, copies and
clears: 2.20 gaps per frame over 200 µs hold 59.5% of the traced idle, and both
sit at the frame boundary bounded by a memcpy — the inter-submit stall (median
2.453 ms) and the end-of-frame drain and readback (median 0.464 ms).

### 5.2b Every host block, censused — the eight sites and what deferring them could buy

`CUDAVK_WAIT_CENSUS=1` (worktree `/tmp/peel-tree` `bd6f2e7a81f`), counts only,
one run per capture, hashes unchanged. `ready` is a `cuStreamQuery` taken
before the sync: a site with `ready > 0` would be a wait with **no device work
behind it**. The ceiling is `Σ min(issue gap, blocked)` per site — the most a
perfect deferral could recover, before any conversion into frame time.

Validated against `CUDAVK_PLAN_STATS` in the same run: 9092.5 vs 9091.9 ms on
the drain, 4146.9 vs 4146.8 on peel, 1461.0 vs 1460.7 on the segment counters.

**old capture, per frame**

| site | waits/f | blocked ms/f | mean ms | ready | ceiling ms/f | ceiling % of blocked | gap-bound / wait-bound |
|---|---:|---:|---:|---:|---:|---:|---|
| episode drain | 9.88 | **6.017** | 0.609 | **0** | **2.066** | **34.3%** | 10,819 / 4,113 |
| peel checks | 1.71 | 2.744 | 1.609 | **0** | 0.301 | 11.0% | 2,504 / 73 |
| segment counters | 4.17 | 0.967 | 0.232 | **0** | 0.253 | 26.1% | 5,896 / 405 |
| descriptor upload | 1.00 | 0.012 | 0.012 | **0** | 0.012 | 100.0% | 0 / 1,510 |
| `vkDeviceWaitIdle` | 2.24 | 0.514 | 0.229 | **0** | 0.500 | 97.2% | 51 / 3,339 |
| upload-arena rewind | 0.05 | 0.073 | **1.377** | **0** | 0.005 | 6.8% | 75 / 5 |
| scratch reclaim | 0.04 | 0.091 | **2.255** | **0** | 0.004 | 4.1% | 59 / 2 |
| **total** | **19.10** | **10.42** | | **0 of 28,852** | **3.140** | **30.1%** | |

**Crossroads, per frame** — no peel site at all, and neither capture reaches
the quad counters:

| site | waits/f | blocked ms/f | mean ms | ready | ceiling ms/f | ceiling % |
|---|---:|---:|---:|---:|---:|---:|
| episode drain | 5.57 | 2.123 | 0.381 | **0** | 0.757 | 35.6% |
| segment counters | 1.69 | 0.431 | 0.255 | **0** | 0.152 | 35.2% |
| descriptor upload | 1.00 | 0.006 | 0.006 | **0** | 0.006 | 100.0% |
| `vkDeviceWaitIdle` | 1.59 | 0.385 | 0.243 | **0** | 0.385 | 99.8% |
| scratch reclaim | 0.00 | 0.000 | 0.019 | **0** | 0.000 | — |
| **total** | **9.84** | **2.947** | | **0 of 14,738** | **1.300** | **44.1%** |

**Every per-frame figure in these two tables is MEAN-based** — the census sums
over the whole replay and divides by the frame count, so it includes start-up
and shader compilation, while this document's frame time is a hot-tail median.
On the old capture that is 20.305 ms mean against 13.163 ms median — both from
the census's own pre-PDL replay, which is the right pair to compare. So
**2.066 ms/frame is 10.2% of the mean frame**, not 15.7% of the median one, and
it must be quoted with that attached. The **ratio** columns — ceiling as a share
of blocked, gap-bound versus wait-bound — are unaffected, being ratios of two
sums over the same interval. See §2's rule.

Four readings, and the last three are the ones a later reader is most likely to
get wrong:

1. **`ready = 0` at every site on both captures.** No wait in this driver is
   pure overhead. Every one is device-paced, so its blocked time is a *symptom*
   of device work, not a cost that removing the wait would recover. **Four sites close and three are live**, once
   each site's conversion is measured rather than assumed (slope table above):
   the episode drain at ≈2.07 ms/frame, the segment counters at ≈0.26 and
   `vkDeviceWaitIdle` at ≈0.22. The drain is the largest by far — 34.3% of its
   blocked time, three times peel's ratio, gap-bound on 72% of its waits.
2. **Three of these sites appear in no other budget in this document:**
   `vkDeviceWaitIdle`, the upload-arena rewind and the scratch reclaim. That
   matters arithmetically, not just for completeness: a census that does not
   know about a blocking call counts it as host *issue* time and inflates its
   neighbours. Measured here — the single-site peel instrument credited peel
   with 599.7 ms of issue and a 0.376 ms/frame ceiling; the eight-site census
   credits 480.2 ms and **0.301**, 24% lower.
3. **A mean without a rate is a trap, and these two are the trap.** The
   upload-arena rewind and the scratch reclaim have the **largest means in the
   driver** — 1.377 ms and 2.255 ms per call — and cost **0.005 and 0.004
   ms/frame**, because they happen 80 and 61 times in an entire replay.
4. **`vkDeviceWaitIdle` blocks only 0.514 ms/frame**, so the 2.453 ms median
   inter-submit stall §5.2 describes is **not** mostly that call. Where that
   stall actually lives is an **open question**, not an answered one.


#### The conversion at each site, measured

The census's ceiling is what a perfect deferral could recover; the **slope** is
what a millisecond of host time at that site is worth in frame time.
`CUDAVK_WAIT_SPIN_US` measures the second directly by injecting a busy-wait
after the wait returns (old capture; every run 3,022 submits and one stdout
hash; D scaled per site so the injection is comparable):

| site | ceiling ms/f | **slope** | linear? | ceiling × slope |
|---|---:|---:|---|---:|
| episode drain | 2.066 | **+1.02** | yes, to 1.98 ms/f injected | **≈2.07** |
| `vkDeviceWaitIdle` | 0.500 | **+0.44** | yes, to 1.01 ms/f (residuals < 0.012) | **≈0.22** |
| segment counters | 0.253 | **+1.03** near the origin (+0.78 fitted to 1.00 ms/f) | saturates above ~0.5 ms/f | **≈0.26** |
| peel checks | 0.301 | **−0.03** | flat to 2.05 ms/f | **≈0.00** |

**Three sites are live, not one.** An earlier reading closed the segment
counters and `vkDeviceWaitIdle` "by size" using an 11% conversion that §6 item 4
has since retired. At their measured slopes they are worth about **0.26 and
0.22 ms/frame** — each comparable to a whole accepted iteration — so they are
**sized, and open**, not closed. Only the peel site is closed on its conversion.

**Correction, 2026-08-27: `vkDeviceWaitIdle`'s 0.22 ms/frame is almost all the
application's own call, not the driver's.** A per-caller census
(`CUDAVK_DESTROY_CENSUS`, both captures, agreeing with this census to 0.00% —
1,510 + 814 + 1,067 = 3,391 exactly) splits the site: the application's own
`vkDeviceWaitIdle` blocks 0.4847 ms/frame at 485.0 µs per drain, `destroy_view`
0.0149 at 21.1 µs, and `destroy_image` 0.0001 at **0.28 µs**. The driver's share
is **3.00% on old and 0.24% on Crossroads**, which at this site's +0.44 slope is
**0.0066 ms/frame**. What is left of the 0.22 is a call the application makes and
the driver cannot defer. See `DEAD_ENDS.md` §17 and `WORKFLOW.md` §4.6:
**attribute a site by caller before ranking it.**

**The three are unlikely to be additive.** The segment-counter sweep saturates:
its point slopes run 1.16, 1.03, 0.80 as injection rises from 0.25 to 1.00
ms/frame, which is the host running out of slack elsewhere. Recovering time at
one site consumes the same slack another would have used.

All four slopes are **add-direction** measurements. Symmetry is demonstrated
only at peel (add ≈ 0, ceiling ≈ 0, device never idle); at the other three a
deferral mechanism still has to be built and measured.

### 5.3 How much is left in removing operations

Using the driver's own prices from §4 rather than a model:

- The realistic fusion programme that existed at this profile was the A-buffer
  support chain plus the fragment writeback. **The A-buffer half is done** and
  paid 0.2208 ms. The writeback half is 43.0 reachable launches/frame, worth
  about 0.04 ms at the measured launch price.
- The **theoretical ceiling** — every remaining launch, copy and small clear
  gone, which would require a driver that issues nothing — is
  1.39 + 0.55 + 0.14 = **2.08 ms**. That was 50% of the 4.16 ms idle at this
  profile; the idle is **3.70 ms** now, so the same ceiling is 56% of it, and
  the launch count it is computed from is 1,314/frame rather than 1,389.

**So the operation-removal programme that iterations 26, 27 and 28 ran is close
to exhausted.** What is left of the idle is reachable only by removing or
overlapping the seventeen host↔device round trips a frame, and §6 says what is
known about that.

---

## 6. What is left

Ranked by value. "Supported" means a measurement in this project points at the
number; "estimate" means it is a forecast from the price table and has not been
measured.

### 0. What bounds everything below: the device is not full, and never has been

Read this first, because it sizes the whole section. `nsys --gpu-metrics` with
no CUDA tracing, 60,163 samples over a window checked against §5.1
(`history/perf-2026-08-27/tiling_ncu.md`):

| counter | over the whole window | over busy samples |
|---|---:|---:|
| GR Active | **81%** | 87% |
| SMs Active | **20%** | 31% |
| SM Issue | **3%** | 4% |
| compute warps in flight | 5 per cycle | |
| DRAM read / write | **1% / 1%** | |

The engine is occupied four fifths of the time, a fifth of the SMs have work,
and those SMs issue on 3% of their cycles. **And this has not moved since
iteration 1.** The baseline at `5ebc37aa36e` recorded 85–96% GR-active, SM issue
3–7%, ~25% SMs active and 5 warps in flight, on the same capture with the same
instrument (`nsys --gpu-metrics`, what `cp_profile.sh METRICS=1` runs), so the
quantities are comparable. The frame went 24.53 → 13.16 ms (−46%, and −47.9% at
today's post-PDL 12.78; this audit ran at the pre-PDL `e2fea470d04`) across every
accepted change in §3 — the register-cap trials, the three fusions, the texture
cache, the small-operation work, the A-buffer fusions, the fan-out, PDL — and
**the machine's utilisation profile is where iteration 1 found it.**

**Every millisecond this driver has won was won by issuing less, or by
overlapping what it issues. None of it was won by filling the GPU.** A lead
whose argument is "the GPU is empty, give it more work" has to explain why
forty-six percent of the frame came out without any of these counters moving.
Size kernel-side leads with §5.1's exclusive-fraction rule and host-side leads
with the site's own slope.

**And width is not the missing lever.** The driver already launches 12.05
waves/SM in FS main (grids of 1,025–4,096) and issues on **3.64%** of cycles,
against 1.25–1.80% for the 0.15–0.60 wave raster launches, while a synthetic
kernel at 2.45 waves reached **59%**. **The warps are resident and stalled, not
absent.** Width moves occupancy from 6–14% to 26–78% and buys back the launch;
it does not move issue. Nor are bigger blocks the lever: at 1,000 items all four
block sizes land on the same time, and 64-thread blocks are fastest at maximum
width (`history/perf-2026-08-27/wide_bench.md`).

### The ranking after the 2026-08-27 session

| # | lead | status |
|---|---|---|
| 1 | let the host run past the episode drain (item 4 below) | **LIVE.** Site ceiling 2.066 ms/frame; the built run-ahead mechanism's own censused ceiling is **0.387 ms/frame**, 0.092 of it converted today at ≈1.0, and the gap is scheduling, not capacity |
| 2 | segment counters ≈0.26 ms/frame, `vkDeviceWaitIdle` ≈0.22 | sized and open, unlikely to be additive with 1 — **but see the correction under item 4**: the driver's own share of the `vkDeviceWaitIdle` site is 3.0%, worth 0.0066 ms/frame |
| 3 | fuse `cp_fs_writeback` into the fragment shader (item 3 below) | **PARKED** at about 0.04 ms on a correctness failure |
| — | peel checks (item 1 below) | **CLOSED**, 0.00 ms |
| — | widen the `bounded` fast path (item 2 below) | **CLOSED**, 0.00 ms |
| — | launch-count merges, kernel stalls, object-destruction drains, tiling v2 | **CLOSED**, see `DEAD_ENDS.md` §17–§24 |

### 1. Peel checks — **CLOSED, 0.00 ms, measured**

**The 0.20–0.50 ms estimate this document carried is retracted.** It is left
here in full because the design below is still the best written record of the
site, and because the retraction is itself evidence: the estimate was never
supported by anything but the size of the wait.

Three independent measurements closed it (§7 has them in full): a perfect
predictor still pays 1.093 checks/frame of today's 1.706; the census puts the
deferral ceiling at 0.301 ms/frame with the device already busy at **every one
of 2,577 checks**; and `CUDAVK_WAIT_SPIN_US` injects up to 2.05 ms/frame of host
time at that exact site for a frame slope of **−0.03**. Adding host time there
is free, which is the same statement as removing it being worthless.

What follows is the design as it stood, and the wait figures it was written
from.

2.751 ms/frame of blocked host over only 1.70 waits, at **1.614 ms each** — the
most expensive single wait in the driver, and the best ratio of value to blast
radius of the three synchronisation sites. **The fan-out did not touch it**, so
it is now 20.8% of the frame where it was 17.3%, and it is worth relatively
more than when this item was written. **It is an old-capture lead only**:
Crossroads runs zero peel checks. The sites at `e2fea470d04` are the host store
at `cp_renderer.c:6512`, the drain at `:6689`, the host read at `:6692` and the
interval doubling at `:6695–6696`. The design
(`/tmp/perf16/iter29-sync/design.md` §5) predicates a whole peel pass on the
device flag the rasterizer already honours, moves the interval reset to a
one-thread kernel so no host store races it, and predicts the trip count instead
of asking. Its correctness argument is already written in the driver's own
comment: once a pass composites nothing, every later pass does too, and a pass
after convergence is already a no-op for output. The overshoot must be capped —
budget about 8 launches per wasted pass, and an unbounded overshoot on a
256-layer draw would add about 1.8 ms and lose. *Supported*: the wait cost and
its distribution. *Estimate*: the frame-time gain.

### 2. Widen the `bounded` fast path — **CLOSED, 0.00 ms, measured**

**The estimate of 0.10–0.25 ms is withdrawn: it is unsupported, and the
mechanism is refuted by direct measurement** (`/tmp/perf-audit/bound_probe_p3.md`,
`CUDAVK_DRAIN_PROBE=1` at `e2fea470d04`, both captures, hashes unchanged).

The wait is real — 0.982 ms/frame over 4.17 waits on old, 0.428 over 1.69 on
Crossroads, unmoved by the fan-out, falling back to the drain at
`cp_renderer.c:6420` when the predicate at `:6402–6413` fails. What is refuted
is that a tighter rectangle can convert any of it.

Sampled at that drain — the exact population this item wanted to convert:

| | old | Crossroads |
|---|---:|---:|
| samples | 6,301 | 2,535 |
| clip rectangle tighter than the framebuffer | **0 (0.00%)** | **0 (0.00%)** |
| **median `nblocks × tris / actual quads`** | **34,560,000** | **29,491,200** |
| median `rast_num_triangles` | 400 | 4,800 |
| **median actual quads** | **32** | **64** |
| admitted today | **0** | **0** |
| admitted with a clip-rectangle bound | **0** | **0** |
| admitted with a clip-rectangle bound and `tris <= 2` dropped | **0** | **0** |
| the slot test alone (`bound × 4 <= 524,288`) | **0** | **0** |

Three things follow, and each is worth carrying separately:

1. **The clip rectangle is never tighter than the framebuffer here.** It is
   230,400 blocks in 8,836 of 8,836 samples. And it is the *right* rectangle:
   `bounded` and `per_draw_rects` are mutually exclusive at HEAD — `scissors`
   is non-NULL only under `compact_rows` (`:4626`), which puts `fs_ubo_table`
   behind the same guard (`:4621`), which sets `ndraws = batch_draws`
   (`:4651`), while `per_draw_rects` needs `batch_draws > 1` (`:4808`) and
   `bounded` needs `ndraws <= 1`. So whenever the fast path can fire, the
   scissor-intersect branch at `:4813` has already run.
2. **`bound/actual` is seven orders of magnitude, not one or two.** The host
   cannot do better: `blocks × triangles` cannot know post-transform triangle
   area until the vertex shader has run. **This is item 4's disease at draw
   scope**, established before anything was built rather than after.
3. **The `bounded` fast path is effectively dead code on both captures.** Zero
   admissions in 8,836 samples of the branch that falls back to it. A correct
   fast path that never fires is a maintenance cost with no benefit, and that
   is a fact about the driver worth knowing independently of this item.

Admission needs `blocks × tris <= 131,072` jointly, which at the measured
medians is ≤ 327 blocks on old (about 72 × 18 px) and ≤ 27 on Crossroads (about
54 × 2 px) — a scanline, not a scissor. The `quad_capacity` test fails
independently: 230,400 × 400 = 92.2 M against a 5–32 M capacity. **Raising the
`512u << 10` slot budget stays withdrawn**; it caps exactly the over-allocation
this ratio would produce. Reopening this needs no new probe: the whole test is
the joint distribution of (clip blocks, `rast_num_triangles`) against
`blocks × tris <= 131,072`, and it is already measured.

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

### 4. Let the host run past the episode drain — **LIVE, up to about 2.07 ms/frame, the largest item in the driver**

**Status after the 2026-08-26 session: the largest of the three blocking sites
still open, and the largest single item in the driver.** §5.2b censused every
host block and measured each live site's conversion; four sites close, and the
other two open ones — segment counters ≈0.26 ms/frame and `vkDeviceWaitIdle`
≈0.22, of which only **0.0066 is the driver's own** (§5.2b, correction of
2026-08-27) — are an order of magnitude smaller than this one. The drain's
ceiling is **2.066 ms/frame on old** (34.3% of its blocked time) and **0.757 on
Crossroads** (35.6%), and it is **gap-bound on 10,819 of 14,932 waits**, so on
72% of drains the host has issue work in hand and the wait is the shorter term.

**Its conversion factor is now measured directly, and it is 1.02.**
`CUDAVK_WAIT_SPIN_US` injects a busy-wait of D µs immediately after a wait
returns, outside that wait's own timing, and the slope of frame time against
injected time is the conversion (`/tmp/perf-audit/conversion_probe.md`, run
`/tmp/perf-audit/spin/`, old capture, 20 runs, every one 3,022 submits and one
stdout hash):

| injected ms/frame | 0.0000 | 0.2470 | 0.4941 | 0.9882 | 1.9764 |
|---|---:|---:|---:|---:|---:|
| **episode drain**, frame ms | 13.2038 | 13.4375 | 13.7393 | 14.2005 | 15.2223 |

**Slope +1.0207** over the whole sweep (residuals under 0.031 ms; +1.0183 with
the top point dropped, so the linear region is the entire range). Host time
added at this site lands **one for one** on the frame, which is what a wait
that empties the stream should look like.

The same probe at the peel site, injection matched at 0.26–2.05 ms/frame,
returns **slope −0.0262** — flat. That is the control, and it validates the
probe by returning ~0 where three other instruments say there is slack and ~1
where the mechanism says there cannot be any.

**The 11% "conversion" is retired.** It came from the peel patch adding
1.01 ms/frame of blocking for 0.110 ms of frame, and this probe shows that
injecting 2.05 ms/frame of pure host time at that site costs *nothing*. So the
patch's 0.110 ms was its **mechanism** — predication issuing further ahead, and
the overshoot passes that follow — not the blocking. Do not use 11% to size
anything.

**A useful side effect: the mean/median caveat does not apply to this number.**
The injected quantity is mean-based per frame (D × 9.882 waits/frame) and the
response is the skip-50 median frame, so a slope of 1.02 measures the transfer
between the two conventions at this site and finds it at par. §5.2b's 2.066
ms/frame therefore does **not** need discounting for the median convention.

**The symmetry caveat is retired at this site, 2026-08-27.** Every wait-site
number above is an *add*-direction measurement, and the add and remove
directions are different measurements. P0 measured the remove direction directly
by moving the existing injected spin to the **other side** of the drain's sync,
so the probe becomes the exact inverse of the mechanism, at the same site, in
the same units (`history/perf-2026-08-27/item2_p0_results.md`; 28 replays, arms
strictly alternating on one binary, 931 sampler observations with none foreign,
3,022 submits and one stdout sha on all 28 runs, injection line read on every
run — 14,932 injections, µs wanted against µs spun agreeing to 0.1%):

| D, µs injected per drain | 62 | 125 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|
| slope, spin **after** the sync (positive control) | 0.992 | 1.045 | 1.050 | 1.025 | 1.026 |
| slope, spin **before** the sync (the mechanism's direction) | **0.034** | **0.065** | 0.257 | 0.413 | 0.589 |

The after-arm is the positive control and it reproduces the published +1.02, so
the port is right. The registered bar was "slope at D=125 below 0.25"; measured
**0.065**, four times inside it. **Host work moved to just before this drain is
absorbed by the wait rather than added to the frame** — about 1.24 ms/frame of
relocatable work for under 0.08 ms/frame of cost.

A second, independent instrument agrees: the driver's own episode-drain counter
is unchanged at every D in the after arm, and in the before arm it *falls* by
what was injected — 98.9% absorbed at D=62, 6,419 of 14,932 ms at D=1000. And
the decoy control did not fire: 238,912 and 955,648 clears issued on a side
stream with 0 injections did not lengthen the mean wait (it fell 2% and 6.7%),
so side-stream work does not lengthen the drain.

**The drain's wait CDF, which nothing in this project had.**
`slope_before/slope_after` is `E[max(0, D−W)]/D`; differencing `D·G(D)` gives

| D, µs | 62 | 125 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|
| **P(wait < D)** | 0.034 | 0.091 | 0.427 | 0.561 | 0.745 |

Mean wait 0.579 ms, almost no mass below 125 µs, and **a knee between 125 and
250 µs — that knee is the relocation budget in one number.**

#### The mechanism was built, and what it found is a scheduling gap

A run-ahead that holds a drain's successor batches and issues their vertex phase
in front of the drain was built, measured and swept
(`history/perf-2026-08-27/item2_tier2.md`, `item2_min_verts_sweep.md`,
`item2_capacity_read.md`, `item2_resolve_why.md`). All correctness gates pass at
every setting: 0 stray launches and 0 drops including at `MIN_VERTS=0`, resolves
equal drains exactly (14,932/14,932), and one stdout hash across all 24 sweep
runs plus the earlier 30.

**Its ceiling is 0.387 ms/frame on old** — P1's census of the vertex work that
sits inside the drain's shadow, 585.4 of 585.7 ms issued, i.e. essentially *all*
the vertex work; 0.072 ms/frame on Crossroads. That figure is **mean-based**
(census total ÷ frame count) like everything in §5.2b.

| `MIN_VERTS` | relocated | share of the 0.387 ceiling | batches held | dominant decline |
|---:|---:|---:|---:|---|
| 32 (default) | 0.028 ms/f | 7% | 9,467 | `small` only |
| 8 | 0.045 | 12% | 16,822 | `small`, then `full` |
| 2 | 0.092 | 24% | 36,129 | **`full` only** |
| 0 | 0.092 | 24% | 36,129 | `full` only |

The frame followed monotonically — medians control-minus-candidate −0.011,
+0.009, +0.028, **+0.069** — and every point is inside the 0.119 ms session
spread, with at least one pair of opposite sign at every value. So neither
registered refutation fires: the moved work *is* on the critical path, and
holding does not cost more than it saves. **0.092 ms/frame simply cannot be
resolved at a 0.119 ms spread.**

**The conversion is 1.01, not the 0.75 the sweep first implied.** The held-call
timer brackets the mechanism's own 56.6 KB record copy at 3.84 µs per hold
against P1's 2.87 µs of censused vertex phase, so 0.98 µs is self-overhead: of
0.092 ms relocated, **0.068 is work removed from the burst** and 0.024 is new
work added in front of the drain, where P0 says it is free. 0.069 of frame for
0.068 of true relocation is **1.01 — the site's own +1.02 recovered end to end.**
That makes the forecast *worse*, not better: at 1.01 the frame gain can never
exceed the vertex work actually moved.

**Do not raise the hold capacity.** The binding decline is `full`, the hold
capacity of 8, which is `CP_PASS_STREAMS` — held segments keep their own queue
set so segments 0..7 map one-to-one onto `seg_qsets[0..7]`, which is what
preserves `fetch_fold`. At conversion 1.0 and the measured reach factor 0.62 the
ladder is:

| cap | true work moved | verdict |
|---:|---:|---|
| 8 (today) | 0.068 ms/f | measured 0.069, agrees |
| 16 | 0.104 | **still inside the 0.119 spread** |
| 32 | 0.163 | |
| 64 | 0.238 | needs 56 more queue sets at 19.2 MB each = **1.07 GB** |

The honest question was never "is 8 structural" but "is +153 MB worth +0.10 ms",
and the ladder answers no. Nor can it be had by sharing queue sets: `fetch_fold`
requires seed(A) → count(A) → seed(B) → count(B), and the mechanism issues every
seed before the drain and every count after it, so two batches on one set have
count(B) accumulating on count(A)'s residue. **No event fixes an ordering the
mechanism deliberately breaks.**

**The real limit is reach, and the reason is scheduling.** 56.4% of deferrals
hold nothing and never reach an admission decision at all. The resolve-reason
probe (counters only, self-check exact on all three runs, no median quoted)
found that **99.99% of zero-hold deferrals had a LATE SUCCESSOR** — 8,311 of
8,312 on old, 5,520 of 5,521 on Crossroads. By the pre-registered definition
that is the good branch: not "there was no successor" (which would have closed
the item at 0.068 ms/frame) but "the successor existed and a required resolve
came first".

| site | zero-hold | share | late |
|---|---:|---:|---:|
| `draw_execute` | 3,823 | **46.0%** | 100% |
| `scope_end` | 3,728 | 44.9% | 99.97% |
| `unknown` | 761 | 9.2% | 100% |
| `flush`, `admit`, `opaque_append` | 0 | — | — |

The `MAX_SEGS=1` cross-check is decisive about capacity: `draw_execute` zero
stays at exactly 3,823 and total zero at exactly 8,312 while the histogram
collapses and `admit` rises 4,096 → 4,414. **Capacity moves work between `admit`
and the held set without changing how many drains find nothing to hold.**

**Status: alive, small, and the next lever is cheaper than the last one.** The
ceiling is unchanged at 0.387 ms/frame, 0.092 of it is converted today at ≈1.0,
and the gap is a scheduling problem — a required resolve pre-empting an
available successor — recoverable with **no capacity change**. It is not "a
0.387 ms opportunity"; nothing has yet resolved above the run spread.

The obvious route into it is still **refuted by measurement**.
`CUDAVK_UNSAFE_NO_OVERFLOW=1`, which removes the wait by sizing for the worst
case, made Crossroads **0.38 ms slower (−6.37%)** with a byte-identical hash and
killed the old capture after about 26 frames: worst-case sizing asks for
8,605,856,768 bytes against an 8,589,934,592-byte scratch cap, the allocation is
refused, and the no-replay rule correctly latches device loss. A request the
arena cannot serve reallocates the whole arena and frees the old base at every
flush, so worst-case sizing buys gigabyte-scale allocate/free churn and
bound-sized bulk clears.

**S1d — draining at the scan — is closed separately, on its own ceiling** (§7).

The two probes that used to gate this item have both been run
(`/tmp/perf-audit/probes_p1_p2.md`):

**P1 — `total/quads` per episode at the drain. Bound-based sizing is _not_
dead.**

| | old | Crossroads |
|---|---:|---:|
| median `total/quads` | **3.753** | **3.725** |
| p10 / p90 | 2.761 / 3.866 | 2.319 / 3.817 |
| min / max over 20,704 episodes | 1.000 / 3.973 | 1.000 / 3.991 |
| aggregate Σtotal/Σquads | 3.7746 | 3.7999 |

The ratio is a small constant, and it is **capped at 4 by geometry**: a quad is
2×2 pixels, so `total ≤ 4 × quads` always, and the captures sit at 94% of that
ceiling. That kills the four-orders-of-magnitude death that
`UNSAFE_NO_OVERFLOW` demonstrated: an over-allocation factor of 3.75 is not an
over-allocation factor of 10⁷.

**What 3.75 costs, stated correctly — an earlier version of this section was
out by a factor of four.** The shade arrays are dense over **quads**, indexed
`slot = 4·q + lane`, and sized `want_slots = num_quads * 4` at
`cp_renderer.c:4182` (`pixel_list`, `fs_in`, `fs_out`, `coverage`,
`frag_coord`, `discard_mask`, `batch_rows`). A host substitute that knows only
the scan's fragment `total` must bound the quads by it — a quad needs at least
one covering fragment, so `quads ≤ total` — and therefore allocate
`4 × total ≈ 3.75 × (4 × quads)`: **3.75× today's allocation, not 0.94× of it.**
The 94% figure is how *full* today's arrays are, which is a different quantity
and not an allocation. Sizing from `total` is only cheap if the arrays are
repacked to be dense over fragments, which changes the indexing.

So S1d survives P1 but is **gated on memory, not on ratio**: 3.75× the shade
arrays is the S1b question again, against the same 8,589,934,592-byte scratch
cap the `UNSAFE_NO_OVERFLOW` probe hit. The measurement that decides it is the
`dscratch` high-water at the drain, not this ratio. The addendum's acceptance
threshold was "≲ 3" and the measurement is 3.75; the threshold was a proxy for
"small constant or four orders of magnitude", and the answer is the first one —
but "small constant" here means 3.75× of the driver's largest arrays.

**P1 is not the `bound/actual` ratio of the `bounded` fast path.** That is
`ab->nblocks × rast_num_triangles` over the actual quads, a different quantity
this probe does not touch. **Do not size item 2 from the 3.75 in this section**;
it needs its own probe at the predicate.

**P2 — the host's inter-drain issue burst. There is work to overlap.**

| µs | old gap | old issue | Crossroads gap | Crossroads issue |
|---|---:|---:|---:|---:|
| p25 | 82.5 | 80.6 | 73.3 | 73.3 |
| **median** | **335.4** | **306.2** | **138.8** | **138.8** |
| p75 | 446.8 | 429.1 | 976.6 | 498.6 |
| p90 | 3966.0 | 3847.6 | 2370.0 | 2362.8 |

"issue" is the gap minus the time the host spent blocked at the other three
timed waits inside it. Read the median: the samples span the whole process, so
the means carry start-up and shader compilation (the largest single gap on old
is 1.77 s). The medians check out against an independent instrument — 9.88
drains/frame × 306 µs = 3.02 ms/frame against §5.2's frame-minus-blocked figure
of 3.48 ms/frame — and the p90 of about 4 ms is the frame boundary, which is
the inter-submit stall §5.2 already describes.

So the addendum's kill branch — "if the burst is tens of microseconds, site 1 is
dead in every form" — **does not fire**: the median burst is **half the median
wait** (306 µs against 606 µs). **But `min(burst, wait) × 9.88 = 3.02 ms/frame`
double-counts and is not a budget**: the host cannot spend the same burst twice,
its total issue time is only **3.48 ms/frame**, and the device is 72% busy, so
the frame cannot fall below its resident **≈9.5 ms**. The honest statement is
that a perfect deferral has **2.5–3.5 ms/frame of device idle** to attack on
old, and under 1 ms on Crossroads, and that P2 has removed the objection that
there is nothing to overlap. It says nothing about S1b's 250 MB–1.6 GB memory
cost, which remains the reason S1b stays demoted.

### 5. Small cleanups — claim 0.00 ms

Emptying the drain of the work that does not need the host removes about 9.9
copies/frame and some host-side loops, and adds about 9.9 launches. Land it on
tidiness or not at all.

---

## 7. What is closed

Short list; `DEAD_ENDS.md` has the evidence and the reasons.

### The lesson of the 2026-08-26 session, stated once so it is not rediscovered

**The host-wait programme has now been tested three times and failed three
times, while both mechanisms that paid worked by overlapping device work rather
than by removing a host wait.**

**One correction inside this lesson, made after the slope probe.** The first
two rows are outcomes and stand. The third was written using an 11%
blocked-to-frame conversion that the probe has since shown does not exist, so
its *value* is understated; the memory objections against it are independent
and unaffected.

| attempt at a host wait | outcome |
|---|---|
| S0, `CUDAVK_UNSAFE_NO_OVERFLOW` — remove the drain by sizing for the worst case | **regressed**: Crossroads −0.38 ms, old dead after ~26 frames |
| peel predication (S2a/b/c, `CUDAVK_PEEL_PREDICATE`) | **regressed**: old −0.11 ms, checks −19% but blocked time +0.98 ms/frame. Injecting host time at that site is *free* (slope −0.03), so the 0.11 ms was the mechanism, not the blocking |
| S1d, drain at the scan instead of the tail | **not built**: ceiling 1.05 ms/frame on old, 0.16 on Crossroads, and the wider bound crosses the arena on Crossroads. Its *value* was first understated using the retired 11% figure — see §7 |

| mechanism that paid | how |
|---|---|
| opaque episode fan-out (`20611f5b131`) | **+2.73 ms on old**: overlaps an episode's segments across side streams. No operation removed; wait *count* unchanged at 9.88/frame |
| programmatic dependent launch (`CUDAVK_PDL`) | **+0.38 ms on old, +0.15 on Crossroads** (decisive, level 3 against level 0, re-measured on the landed binary): overlaps a kernel's preamble with its predecessor's tail. No operation removed |

The reason is measured, not inferred: **blocked host time at these sites does
not convert into frame time one for one, and the factor is per site.** The
`CUDAVK_WAIT_SPIN_US` probe (§6 item 4) injects host time at a site and reads
the slope of frame time against it: **−0.03 at the peel checks and +1.02 at the
episode drain**. The peel site is off the critical path — the census also found
the device busy at **0 of 2,577 checks** — while the drain, which empties the
stream by definition, is fully on it.

**Size a host-wait proposal against that site's measured slope and its census
ceiling, never against the blocked figure in §5.2.** The blocked column is an
upper bound that is not collectable, and the ratio between them differs by two
orders of magnitude between two sites in the same driver.

**Device-side removal is exhausted, and so is width.** Removal: §5.3's
operation-removal ceiling was already close to exhausted, and the last
raster-side candidate closed at 0.44% of stage-1 invocations (last entry
below). Width: blended segments **already** fan out over eight streams with no
flag, and at 0.15 waves/SM about **6.7 concurrent launches fill the machine**,
so nine streams is already more width than the hardware can use.

So the remaining programme is neither removal nor width. It is **duty cycle and
launch rate** — keeping the machine issuing, which is what the fan-out and PDL
both do.

### Closed in that session, with the number

- **Widen the `bounded` fast path with a clip rectangle — 0.00 ms.** §6 item 2.
  The clip rectangle is never tighter than the framebuffer (0 of 8,836
  samples), `bound/actual` has a median of 3.46 × 10⁷ on old and 2.95 × 10⁷ on
  Crossroads, and no variant admits a single draw. The `bounded` path is
  effectively dead code on both captures.
- **S1d, draining at the scan — closed on cost and mechanism, not on value.** P4 measured `Σ min(wait, scan→bucket)` at **17.6% of the old
  drain (1.05 ms/frame)** and **9.1% on Crossroads (0.16 ms/frame)**; the chain
  *after* the scan is 88.5% and 71.5% of the episode's device time, so the
  count phases are not what the host is waiting for. P3 adds that the wider
  bound **crosses the arena on Crossroads** (607.3 MB projected against 506.6
  MB held) and that 25.3% of old-capture episodes would be refused by the
  `capacity/4` gate.
  **Its value was understated by about a factor of nine.** It was first closed
  partly on "a 1.05 ms ceiling buys well under 1.05 ms", which used the 11%
  figure that §6 item 4 has now retired; at the drain's measured conversion of
  **1.02**, a 1.05 ms/frame ceiling is larger than PDL's entire measured win.
  Its author re-examined every objection at the measured conversion and the
  closure holds, but on a **narrower base than it was first written**. Two of
  the five weakened under the check: P4's 17.6% is a small *share* of a large
  wait and was wrongly used as evidence about the *quantity*; and the memory
  objection is capture-asymmetric the inconvenient way — the drop-in form
  **fits on old** (1069.3 MB against 7914.8 MB held, 0 regrows), which is
  exactly where the 1.05 ms is, and crosses only on Crossroads where 0.16 ms is
  at stake. Memory does not refuse the mechanism where the value is; the
  `capacity/4` gate does. What the closure now rests on is the **dead packed
  form** (shade slots are dense over quads because lanes 4q..4q+3 *are* the
  derivative quad — see the shade-slot entry below, and it is architectural
  rather than economic), the `capacity/4` gate, and the fail-closed checks that
  would have to be reproduced on the device or demoted.
  **What survives at this size is not S1d.** It is issuing the *next* episode's
  vertex stage across the drain: 2.19 ms/frame of VS kernel time sits behind a
  round trip it does not depend on, and it needs one more **VS output buffer**
  rather than a second set of A-buffer arrays, so it does not inherit what
  refuted S0 and S1b. Named, not designed; read `cp_scratch_reset`, which frees
  overflow arenas at every flush, before building anything.
- **The peel site is closed, by three independent measurements.** (i) A perfect
  predictor still pays **1.093 checks/frame** of today's 1.706 under
  `CP_PEEL_PREDICT_MAX` (0.037 without the cap), so at most 35.9% of the checks
  are removable, and key stability is 82.0% over 79 distinct keys against a
  64-entry ring — a one-constant fix, not worth making. (ii) The census puts
  its deferral ceiling at **0.301 ms/frame** with the device **already busy at
  every one of 2,577 checks**. (iii) `CUDAVK_WAIT_SPIN_US` injects up to
  **2.05 ms/frame** of host time at that exact site and the frame does not move
  — **slope −0.0262**. Adding host time there is free, which is the same
  statement as removing it being worthless. The predication patch that was
  built and rejected (−0.11 ms on old) cost what it cost through its mechanism,
  not through its blocking. **Crossroads runs no peel loop at all.**
- **`quads == 0` drains — closed on mechanism.** 6.9% of old drains and 18.3%
  of Crossroads drains return nothing (0.053 and 0.322 ms/frame). The drain is
  taken for `fill_over`/`quad_over`; `quads` rides along, and
  `cp_pass_can_retry` forbids the fragment shader before the overflow answer,
  so an oracle would save no wait.
- **Depth/stencil load-store elision — 0.005 ms/frame, overstated 20–40×.**
  The driver launches **one** kernel keyed on the depth aspect; there is no
  stencil and no colour-attachment kernel, so that traffic is already elided by
  construction. The kernel is 4.65 µs, 26th of 26, moving 7.4 MB at 1.6 TB/s —
  already at the memory roofline. *Reopen only for 8× MSAA depth*, which would
  make it ~37 µs/kernel and ~0.04 ms/frame; neither capture does MSAA depth.
- **The A-buffer shade slots cannot be packed.** Lanes `4q..4q+3` are the 2×2
  quad that `ddx`/`ddy` is taken across, via `shfl.sync.bfly` over `tid & 28`.
  CUDA texture instructions do not infer fragment derivatives, so an uncovered
  lane is shaded **deliberately**: the 6.2% of lanes carrying no fragment are
  helper lanes and are architecturally required. Any future proposal to
  renumber, compact or pack shade slots dies here.
- **Per-triangle setup reuse in the peel loop — 0.44%, below the measurable
  line.** A per-triangle queued bit would skip 1,842 setups a frame on old,
  **0.44% of direct stage-1 thread invocations** (0.0083 ms/frame against a
  0.119 ms run spread), and **0.00% on Crossroads**, which runs zero reusing
  passes. 98.6% of old-capture peel loops run exactly one pass. **This was the
  last raster-side removal candidate.**
- **PDL level 4 (stage 1 as a secondary) — closed, no mechanism.** Level 4 adds
  18.43 offers/frame on old and **0.004 takes/frame**; every new offer
  declines. The declines identify themselves: 27,848 new declines against the
  27,847 reusing passes counted independently. The offers *are* the reusing
  passes, and they decline because the visbuf re-clear sits in front of stage 1
  on exactly those passes — the epoch check working as designed. Hoisting that
  clear is not worth it either: the population behind it is the 0.44% above.
- **`CUDAVK_INLINE_FS` re-measured at HEAD and confirmed opt-in.** Iteration
  14's architecture had not been re-tested since the texture cache became the
  default or since the fan-out landed. AB/BA on 2026-08-26: **old −0.2112 ms
  (−1.60%)**, Crossroads +0.0090 ms (+0.15%, noise), hashes and submit counts
  clean. It is a regression at HEAD; do not spend another AB/BA on it.
- **The `cuLaunchKernelEx` premium is not a cost — +6.3 ns.** Empty-kernel
  microbenchmark, interleaved arms, minima: `cuLaunchKernel` 2020.1 ns,
  `cuLaunchKernelEx` with no attributes 2026.5 ns. Against PDL's measured
  0.44 µs/link that is a factor of seventy. *The same benchmark's PDL arm
  (1151.6 ns) is an artefact* — at 200,000 back-to-back launches the host is
  throttled by queue depth, so it measures throughput with overlap enabled, not
  entry cost. Do not quote it as "PDL makes launches cheaper".

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

### Closed in the second half of the same session (2026-08-27), with the number

The evidence is in `DEAD_ENDS.md` §17–§24 and the raw reports are archived under
`history/perf-2026-08-27/`. Each of these was a ranked lead when the session
started.

- **Object-destruction drains — 0.0066 ms/frame.** The driver owns 55.5% of the
  `vkDeviceWaitIdle` site *by count* and 3.0% of it *by blocked time*, because
  the driver's drains arrive at a device that is already empty. `DEAD_ENDS.md`
  §17.
- **Kernel stalls / "wider requests" — nothing to widen.** `cp_rasterize_stage3`
  reading 1.00 sectors/request was an artefact of the profiled window (its
  working launches read 3.087); `_abuf`'s 1.00 is the **optimum**, because 67%
  of its requests are one uniform 4-byte broadcast per block. `DEAD_ENDS.md`
  §18.
- **Removing degenerate stage-3 launches — capped at 0.164 ms/frame** without
  needing the launch mix, and the host cannot know the tile count without a
  readback that costs a hundred times the launch. `DEAD_ENDS.md` §19.
- **The wide raster merge — removing 298.8 launches/frame COST 0.410 ms/frame.**
  `DEAD_ENDS.md` §20.
- **The count-phase merge — refuted by the same rule before it was built.**
  `DEAD_ENDS.md` §21, and the rule itself in §22.
- **Tier 1, hoisting the shading-group tables above the drain — measured zero**
  with the hash gate passing on ten runs. `DEAD_ENDS.md` §23.
- **The opaque sort-middle tiling prototype, and any v2 of it — refused.** Best
  case 2.1–2.6 ms/frame of kernel time against the +2.73 ms/frame fan-out it
  has to surrender, and the note's recorded explanation is dead in both halves
  (compute throughput 2.53%, branch efficiency 94.32%, triangles 1.7 px across).
  `DEAD_ENDS.md` §24.
