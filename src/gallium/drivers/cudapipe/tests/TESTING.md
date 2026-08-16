# Methodology: how this driver is checked and how it is measured

Both halves are here, because they use the same runs and the same tools and
because neither is worth much alone — a change that is faster and wrong is not
a result, and the second-largest source of wasted effort on this driver has
been measuring the wrong thing confidently.

| | where |
|---|---|
| **correctness** — is it drawing the right pixels | the sweeps below, "calibrate against llvmpipe", "traps" |
| **cost** — what a frame costs, reliably enough to compare | "benchmark", "traps", "iterating on performance" |
| **diagnosis** — where the time actually goes | "finding where the time goes" |

The pass records — `../PERFORMANCE_PROGRESS.md`, `../PHASE_1A.md`,
`../INSTANCING.md` and `../BATCHING.md` — are what was done and what it was
worth, not how to do it. This is how to do it.

---

## The sample sweep: running it, checking it, reading it

cudapipe is checked differentially. The Sascha Willems samples at `~/git/Vulkan`
render offscreen and reproducibly, so the same binary is run against the NVIDIA
ICD, against cudapipe, and against lavapipe/llvmpipe, and the frames are
compared. That is a far stronger signal than a pass/fail suite, and it has found
every rendering bug fixed so far.

There are three sweeps, and they answer different questions.

| | frames | answers |
|---|---|---|
| **single frame** | 1 | does the driver draw the scene |
| **animated** | 60 | does it keep drawing it |
| **benchmark** | 600 | what that same work costs |

The single frame sweep is the cheap one and catches most things. The animated
one catches what a still image cannot: state that leaks between frames, a
particle system settling, anything that only goes wrong once the camera moves.
Two of the regressions in the set are invisible at frame 0.

The benchmark sweep renders the same work as the animated one — same loop, same
fixed frame time, same orbit — and stores none of it. That split is not a
convenience: storing a 1280x720 frame takes about 10 ms, which is longer than
several of these samples spend rendering one, so a pass that writes images
cannot be timed. Correctness and cost therefore come from two runs of one
workload, which is what lets a slow frame in the chart be looked at in the other
run's images.

It runs the orbit for more frames, because sixty is too few to time — the orbit
is periodic, so that is the same path covered again rather than a different one.
"The frame count is the one flag that may differ" below has the measurement that
justifies it.

---

## The pieces

| Tool | What it does |
|---|---|
| `cp_iterate.sh` | one optimisation iteration: build, time, render, compare, record |
| `cp_iter_report.py` | an iteration's json record, and the pages over them |
| `headless_streamer_samples.txt` | the sample set — one entry per capability the capture needs, with the mapping in comments |
| `cp_perf_run.sh` | one driver's pass: frames, timing, GPU load |
| `cp_compare.py` | two images, or the PPM/PNG reader the rest import |
| `cp_compare_frames.py` | every frame of every sample against a reference, with a pass/fail verdict |
| `cp_gallery.py` | HTML for the single frame sweep, two or three drivers |
| `cp_perf_report.py` | HTML for the animated sweep: cost, GPU, per-frame differences, frame inspector |
| `cp_gpu_busy.sh` | is the GPU actually busy — the measurement gate, untraced |
| `cp_profile.sh` | one sample under a profiler, in three modes: `METRICS=1` device counters, default CUDA trace, `NCU=1` per-kernel counters |
| `cp_prof_kernels.py` | a trace split by kernel *and grid size*, with the fixed-cost kernels flagged |
| `cp_prof_nvtx.py` | the same trace split by *pipeline stage*, from the driver's NVTX ranges |
| `cp_metrics_sweep.sh` | device counters for every sample, into the iteration's record |

**Run these with the repo venv's interpreter**, `$MESA/venv/bin/python3`
(`pip install numpy pillow` beyond what the build needs), which is what
`cp_iterate.sh` does. `cp_compare_frames.py` requires `numpy`: it is the one
tool that counts pixels over a whole sweep rather than over one image, which
is a thousand 720p images a run and belongs in C rather than in a loop. The
rest only get faster — `cp_compare.py` decodes through Pillow when it is
importable and falls back to its own PNG reader when it is not, so the tools
that read images still run under a bare `python3`, about a hundred times
slower per filtered PNG. `cp_perf_report.py` additionally needs `ffmpeg`,
which does all of its image work.

---

## Running a sweep

### Single frame

`run_offscreen.sh` in the Vulkan tree, once per driver, then the gallery:

```bash
cd ~/git/Vulkan
S="$(grep -v '^#' ~/mesa/src/gallium/drivers/cudapipe/tests/headless_streamer_samples.txt | tr '\n' ' ')"
M=~/mesa/build-cudapipe/src/gallium/targets

SAMPLES="$S" VALIDATION=0 OUT=build/compare/ref      ./run_offscreen.sh
SAMPLES="$S" VALIDATION=0 OUT=build/compare/cuda \
  VK_ICD_FILENAMES=$M/cudapipe/cudapipe_devenv_icd.x86_64.json ./run_offscreen.sh
SAMPLES="$S" VALIDATION=0 OUT=build/compare/llvmpipe \
  VK_ICD_FILENAMES=$M/lavapipe/lvp_devenv_icd.x86_64.json ./run_offscreen.sh

cd build/compare
python3 ~/mesa/src/gallium/drivers/cudapipe/tests/cp_gallery.py ref cuda llvmpipe \
    -o three.html --ref-label nvidia --test-label cudapipe --test2-label llvmpipe
```

The third directory is optional and worth giving. See "Calibrate against
llvmpipe" below.

### Animated

`cp_perf_run.sh` per driver, then the report:

```bash
cd ~/git/Vulkan
R=$PWD/build/frames60
M=~/mesa/build-cudapipe/src/gallium/targets
T=~/mesa/src/gallium/drivers/cudapipe/tests

$T/cp_perf_run.sh nvidia   ""                                          $R/nvidia   60
$T/cp_perf_run.sh cudapipe $M/cudapipe/cudapipe_devenv_icd.x86_64.json $R/cuda     60
$T/cp_perf_run.sh llvmpipe $M/lavapipe/lvp_devenv_icd.x86_64.json      $R/llvmpipe 60

python3 $T/cp_perf_report.py $R --ref nvidia --drivers nvidia cuda llvmpipe \
    --bench $PWD/build/bench60 -o build/compare/perf60.html
```

`--bench` points at the benchmark pass below and is where the page's cost
section and its GPU charts come from. Without it they come from this run, whose
times are mostly image writing — the page says which it used.

Roughly five minutes for the three passes and half a minute for the report.
It writes about 8 GB of frames and 2.7 GB of report images.

### Benchmark

The same command with `1` as a fifth argument times the frames instead of
storing them. The frame count is the one thing that may sensibly differ from
the storing run — see below — and the script refuses `0`:

```bash
B=$PWD/build/bench60

$T/cp_perf_run.sh nvidia   ""                                          $B/nvidia   600 1
$T/cp_perf_run.sh cudapipe $M/cudapipe/cudapipe_devenv_icd.x86_64.json $B/cuda     600 1
$T/cp_perf_run.sh llvmpipe $M/lavapipe/lvp_devenv_icd.x86_64.json      $B/llvmpipe 600 1
```

It writes `_bench.csv` — `sample,frames,fps,ms_avg,ms_best,ms_worst,wall_s,exit`
— and one csv of every frame's time per sample, and no images. Two and a half
minutes for cudapipe's pass at 600 frames.

**The fps figure times recording and submitting a frame, not finishing it.**
Nothing waits for the GPU until the pass ends, so a driver that submits
asynchronously is measured on its CPU side alone — NVIDIA reports around 48,000
fps on `triangle`, which is the rate it queues frames at, not the rate it draws
them. It is the honest number for cudapipe, which blocks the host on every draw,
and close to it for llvmpipe. **Across drivers compare `wall_s`**, which cannot
miss anything because the process does not exit until `vkDeviceWaitIdle`
returns, at the cost of also containing start up and shader compilation. Both
columns are there so the comparison is possible without re-running anything.

`renderheadless` and `computeheadless` never enter the offscreen loop — they
drive their own frames — so they have nothing to time and get no row.
`renderheadless` writes its image itself, in the working directory, benchmark
mode or not; the script deletes it so that a timed pass really does leave
nothing behind.

**Which means `renderheadless` is not in the correctness gate either, and its
row lies about it.** Its stored directory is empty, so is the nvidia
reference, `verdict.txt` says `missing`, and a bit-identity diff of one empty
directory against another reports IDENTICAL. That is not a pass; it is two
absences agreeing. A change that broke `renderheadless` outright would show up
as `missing` and `IDENTICAL`, exactly as it does today.

It came up when framebuffer-sized buffers were made grow-only, where
`renderheadless` was one of the four size-changing samples most likely to
break. Checking it meant running it by hand against both builds in separate
working directories and comparing the `headless.ppm` each left behind:

```sh
cd $(mktemp -d) && VK_ICD_FILENAMES=$OLD_ICD ~/git/Vulkan/build/bin/renderheadless
cd $(mktemp -d) && VK_ICD_FILENAMES=$NEW_ICD ~/git/Vulkan/build/bin/renderheadless
md5sum */headless.ppm
```

Worth doing whenever a change could plausibly touch it, and worth fixing in
the harness the day someone needs it more than once.

**The frame count is the one flag that may differ, and only because it was
measured.** Everything else has to mirror the storing run. Sixty frames is
right for storing — it is the animation the samples were set up to render, and
every frame is kept — and far too few for timing. A sample's process spends a
couple of seconds on start-up, shader compilation, the warm-up second and
teardown, all of it with the GPU idle, so at sixty frames the render loop is a
*minority* of the process:

| sample | ms/frame | 60 frames render for | of a process lasting |
|---|---|---|---|
| instancing | 27.0 | 1.62 s | 4.3 s |
| dynamicuniformbuffer | 17.0 | 1.02 s | 3.7 s |
| bloom | 12.1 | 0.73 s | 3.5 s |
| triangle | 0.7 | 0.04 s | 2.8 s |

A mean taken over one and a half seconds is dominated by the sweep's own
run-to-run spread, which is why two runs of one build differed by 0.6-1.2%
during the phase 1a pass — the same size as several of the changes being looked
for. `cp_iterate.sh` therefore times `BENCH_FRAMES` (600) and stores `FRAMES`
(60), which puts every sample above ten seconds of rendering.

That is allowed to differ only because the two still render the same work: the
orbit is periodic, so a longer run covers the same camera path again rather
than a different one. Checked rather than assumed, 60 against 600 over the
whole sweep:

| | total | multithreading | dynamicuniformbuffer | instancing | bloom |
|---|---|---|---|---|---|
| 60 frames | 168.80 | 29.70 | 16.95 | 27.02 | 12.13 |
| 600 frames | 170.14 | 29.70 | 16.99 | 27.30 | 12.09 |
| | +0.8% | 0.0% | +0.2% | +1.0% | −0.3% |

The two samples that move more than that, `gltfscenerendering` (−5.4%) and
`particlesystem` (+4.0%), are both on the nondeterministic list below. A
600-frame pass takes about two and a half minutes; 6000 takes twenty-five and
is for a final confirmation.

### 600 frames is not enough to call a regression on `gltfscenerendering`

It is enough for the other sixteen. That one has been seen to spread **6.9%
across three runs of an identical build** — 13.87, 14.83, 13.88 — which is
larger than the ±5% the gate flags at, so a single sweep can manufacture a
regression out of nothing and can equally hide a real one.

It has done both. One pass reported it +7.1% and the number vanished under a
controlled comparison, the baseline build reproducing the "regressed" figure on
its own. A later pass reported +7.1% again and that one was real, reproducing
at +0.99 ms across five interleaved pairs. The two are indistinguishable from
the sweep alone.

So a >5% move on a >3 ms sample is a **question, not a finding**. Answer it by
building both trees and alternating:

```sh
git worktree add /tmp/base <baseline-commit>     # build it there
for r in 1 2 3; do
  for v in old new; do
    SAMPLES=gltfscenerendering tests/cp_perf_run.sh $v $ICD_$v $OUT/$v$r 600 1
  done
done
```

Report the per-pair deltas and their median, not the two means. Within an arm
the spread is about 0.1 ms, so pairing resolves a 1% move that the sweep cannot
see at all.

The suspected mechanism is the driver's own: `cp_tune_before` times register-cap
candidates at runtime and keeps whichever build wins, so two runs of one binary
can settle into different states. That makes it a property of the driver rather
than of the machine, and it will not go away by asking the sweep more politely.

The same caution applies to the **capture replay** described in
`GFXRECONSTRUCT.md`, which is otherwise a far finer instrument than the sweep —
four consecutive pairs of it agreed to 0.07%, and then a later pair disagreed by
3.9%. Treat anything under about 10% there as needing a paired A/B too: build
two `.so` files, write one ICD json each pointing at its `library_path`, and
alternate. A change worth under 3% may simply not be resolvable in wall time,
and is better argued from a direct count — an `LD_PRELOAD` shim over the CUDA
driver API costs 0.1% and gives exact per-call-site totals, where CUPTI would
distort this driver badly.

**The rest of the flags have to mirror the storing run exactly**, which is easy
to get wrong, because offscreen benchmarking quietly ignores the options a
windowed benchmark uses:

| | storing | timing |
|---|---|---|
| count | `--offscreenframes 60` | `--offscreenframes 600`, **not** `--benchruntime` — see above |
| motion | `--offscreenorbit` per the ORBIT list | the same list, or the still samples benchmark a static scene |
| output | `--offscreenfilename` | none — nothing is stored |

`--benchruntime` is what a windowed benchmark uses to run for N seconds, and
offscreen it does nothing but print a note. The mode this script had before
passed it and no frame count and no orbit, so it timed an unstated number of
frames of a workload the correctness run never rendered. A benchmark that does
not render the same thing is not measuring the thing you are comparing.

**Warm up is a second per sample, and it lands in `wall_s`.** The samples
render frame 0 repeatedly for `WARMUP` seconds before the timed loop, and
deliberately do not advance their state while doing it, so how many warm up
frames a fast machine gets through cannot change which frames are measured.
Eighteen samples means eighteen seconds, which is most of llvmpipe's wall and
none of cudapipe's — do not read a `wall_s` total as rendering time.

---

## What each tool records

### `cp_perf_run.sh`

Per sample: wall clock, user and system CPU, peak RSS and exit status, into
`_timing.csv`, in both modes. For the whole pass: GPU utilisation, memory used
and total, and power at 2 Hz into `_gpu.csv`, and per-process GPU memory into
`_gpu_procs.csv`. In benchmark mode also `_bench.csv` and the samples' own
per-frame times, described above.

The GPU samplers cover the pass rather than each sample deliberately. Several
samples finish in well under a second, so a per-sample capture would get a
single poll; a sample's slice is cut out afterwards by the process names in
`_gpu_procs.csv`, which is what `perf.html` does. `_timing.csv` cannot do it —
it records how long each sample took but not when it ran.
**Utilisation figures are only meaningful in benchmark mode**, where each sample
runs for seconds rather than milliseconds.

**`ORBIT`.** Nine samples render a still scene — nothing moves unless the camera
does, and sixty frames of one is sixty copies of the same image. Those run with
`--offscreenorbit`, which walks the camera around the subject. The list is a
default in the script; `ORBIT=all` orbits everything, `ORBIT=` orbits nothing.
This is not cosmetic: `computeshader` matched NVIDIA exactly on a static frame
and diverged by 28,805 pixels within five frames of the camera moving, because
nothing clipped its geometry to the viewport and the sample draws two of them
side by side. Nothing but an orbit could have found that.

### `cp_compare_frames.py`

Compares every frame against a reference and prints a verdict per sample. The
number that matters is judged **against what frame 0 already differs by**, not
against zero — most samples carry a standing difference that says nothing about
animation. `--tolerate` sets how much worse than frame 0 a frame may be as a
multiple, `--floor` adds a constant for samples whose frame 0 is near zero.
Exit status is 1 if anything regressed, so it can gate a benchmark run.

### `cp_perf_report.py`

The page has three sections: cost, the GPU over the run, and differences over
the animation.

Cost and GPU come from the `--bench` pass when one is given, differences always
from the frame pass — the two render the same work, so a slow frame in the cost
section and a wrong frame in the difference charts are the same frame. The cost
table carries both `ms/frame` and `wall` because neither is enough on its own:
`ms/frame` times recording and submitting a frame rather than finishing it, so
it under-reports any driver that submits asynchronously, and `wall` catches
everything but also carries start up, shader compilation and the warm up
second. `pbribl` on llvmpipe is the clearest case — 2.6 ms a frame and 16.6 s
of wall, nearly all of it precomputing its IBL textures before the first frame.

Each sample gets a difference-over-time chart scaled to itself — a shared axis
would flatten every small one, and the small ones are where the signal is. Under
each chart is a frame inspector:

- **click the chart** to send the panes to that frame; a cursor marks where they are
- **hover a driver's frame** to flip it to the reference
- **click any pane** for the full resolution PNG, where the same hover works
- **difference panes** show `|driver − reference|` brightened (`--diff-gain`,
  default 12)

`--no-frames` skips the image export and leaves the charts.

**Re-rendering a driver means clearing its exported images.** The export skips
any sample that already has its frames, and the differences are cached in
`_diffs.json`, so a second run over new frames otherwise puts the old pictures
beside the new numbers. Delete `<images>/<driver>` and `<images>/_diff/<driver>`
and pass `--recompute`.

---

## How the image work is done

All of it is ffmpeg, in one invocation per sample rather than one per frame.

**Counting differing pixels.** The difference of the two streams, the max across
the three channels via two `lighten` blends (lighten is a per-pixel max), a
threshold at the tolerance, then `signalstats`' `YAVG` of the resulting 0/255
mask — which is the differing fraction. It agrees exactly with the pure Python
count in `cp_compare.py`, and a 60 frame sample takes about a third of a second.

**Difference images.** `blend=all_mode=difference` then `colorlevels` to rescale
the input range, which is a gain.

Three things about this were got wrong first and are worth not repeating:

- **`eq=contrast` is not a gain.** It pivots around mid grey, so on an image
  that is almost entirely near zero it drives everything *down*. A "gain" of 12
  produced a mean of 0.2 where the raw difference was 3.7. `colorlevels` is the
  right filter.
- **Pin the pixel format to RGB before blending.** The filter graph negotiates a
  format with the output encoder. With a JPEG encoder it picks YUV, and a
  difference taken there leaves U and V at zero rather than neutral, which
  converts back to *saturated green*. Every difference image was green.
- **Nothing here may be lossy.** The page exists to judge whether a pixel is
  wrong; JPEG artefacts sit directly on top of that. It also hid the bug above,
  because the PNG path negotiated RGB and looked correct. PNG only.

---

## Calibrate against llvmpipe, not against zero

llvmpipe is the backend cudapipe's sampler and clipper were ported from and
shares lavapipe as its frontend, so it is the honest target. It is mature and
still differs from NVIDIA by tens of thousands of pixels across the set, most of
it the same anisotropic filtering difference cudapipe has.

Run it as a third driver in both sweeps. A residual both software renderers
share is what software rasterization costs; a residual only cudapipe has is a
defect. That distinction has settled several questions that the NVIDIA delta
alone made look like cudapipe bugs — `gltfscenerendering` grows fourfold over an
orbit, but llvmpipe grows further on the same frames.

**llvmpipe runs through the same lavapipe.** A change under
`src/gallium/frontends/lavapipe` moves both renderers and can invalidate the
comparison set without touching cudapipe. Re-render llvmpipe and `cmp` it
against the stored set after any such change.

---

## Traps

**Read the exit codes, not just the images.** `_summary.txt` and `_timing.csv`
both carry them. Four samples rendered correctly and segfaulted on the way out
for weeks because only the images were being looked at, and one of those was
then misdiagnosed as a 60-frame regression when it reproduced at one frame in
under a second.

**Six samples are not deterministic.** `multithreading`, `gltfscenerendering`
and `instancing` differ run to run on the same build — thread scheduling changes
the order command buffers are recorded. `multisampling`, `vulkanscene` and
`particlesystem` do too, for a reason not yet established, though only barely:
over sixty frames two runs of one build move by 36, 1 and 0 pixels at tolerance
8, with `particlesystem` differing byte-wise on 13 frames without any pixel
crossing the tolerance. Before attributing a small delta to a change, run the
sample twice against itself. Their last few hundred pixels are not signal.

**Rebuild a sample after editing it.** Editing a sample's `.cpp` to bisect
something and restoring the source afterwards does not rebuild the binary. An
hour of runs once went into comparing a bilinear render against an anisotropic
reference, and produced a confident and completely wrong conclusion.

**Sample every frame.** A stride of 10 missed `texture3d` entirely, reporting
855 differing pixels where the full pass finds 3,090 — the excursion is narrower
than the sampling interval. Strided passes are for a quick look, not a verdict.

**A whole-process time is not a rendering time.** `pbribl` was recorded in the
handoff as the one sample cudapipe beat llvmpipe on, 14.2 s against 16.0. Those
were process wall clocks, and llvmpipe spends 16.6 s of `pbribl` before its
first frame, precomputing the IBL textures; per frame it renders at 2.6 ms
against cudapipe's 135, a factor of fifty the other way. Any measure that
contains start up will eventually be read as if it did not. This is what the
`ms/frame` column exists for, and why the page prints both.

**A per-frame minimum is not the cost of a frame.** The samples keep two frames
in flight, so the fence wait for one lands in the time of another:
`computeshader` on cudapipe reports a best of 1.6 ms and a worst of 191 against
a mean of 133. The mean over sixty is the number; `ms_best` and `ms_worst` are
for spotting a genuine outlier, and only when they are wide of the mean on both
sides.

**Run the timing passes one at a time, with nothing else on the GPU.** Three
drivers over eighteen samples is a few minutes, and it is tempting to overlap
them. Two passes sharing the card measure each other. The same goes for a sweep
running while a build does — `nvidia-smi` during the pass is the check.

**A number from `_bench.csv` is not comparable to a hand-run sample, because
nine samples orbit and a hand-run one usually does not.** `cp_perf_run.sh`
appends `--offscreenorbit` for `triangle`, `pushconstants`, `texture`,
`negativeviewportheight`, `texturecubemap`, `computeshader`, `vulkanscene`,
`pbribl` and `gltfscenerendering`. Run one of those by hand without the flag and
it renders a different camera path — a different workload, not a slower one:

| | by hand | `--offscreenorbit` | in the sweep |
|---|---|---|---|
| gltfscenerendering | 17.15 | **15.33** | 15.10 |
| pbribl | 2.00 | **1.64** | 1.67 |

An afternoon went into a 13.7% `gltfscenerendering` "regression" that was
entirely this flag, and it survived being "confirmed" three times because every
confirmation repeated the same mistake. It was then written up here as a 13%
*clock and sweep-position* effect, which was invented to explain a gap that had
a flag behind it — the wrong explanation is the part worth remembering, because
a plausible mechanism is exactly what stops the boring cause being checked.

The rule that does hold: **build the thing you are comparing against and
measure it beside the new number, with the same flags, in the same sitting.**
An A/B is only valid when both sides were run identically — which is what makes
`cp_iterate.sh` the right tool and a hand-run pair the fragile one. `ORBIT=` on
the command line disables orbiting for a whole pass if a hand comparison is
what is wanted.

**A probe that changes what the compiler can prove is not measuring the thing
it names.** Probes — cutting a kernel to an early `return`, pinning an input to
a constant — are how nearly every ceiling in this document was established, and
they have one failure mode. A probe that hardcoded the rasterizer's bounding
box wrote over the four values it had just loaded, so NVRTC saw the loads were
dead and removed them: the run measured *no bounding box read at all* while
claiming to measure *a small bounding box*, and reported a 17% win that did not
exist. It sent a whole investigation down the wrong path.

Write a probe so the work it is meant to keep is still observable — consume the
loaded value, or `if (never_true) use(value);` behind something the compiler
cannot fold. And when a probe result is surprisingly good, check the generated
code before believing it.

---

## Finding where the time goes

Four questions, in this order. Answering the first with the wrong tool is the
mistake this section exists to prevent.

### 1. Is the frame kernel-bound or host-bound?

```bash
cp_gpu_busy.sh instancing 20        # SAMPLE, then seconds of rendering
```

`>90%` busy means the host is already ahead of the device and there is nothing
to win by submitting faster. Below that, the device is waiting on the host for
the remainder, and that gap is what work on submission and synchronisation is
worth.

**A profiler cannot answer this.** CUPTI adds host-side cost to every
`cuLaunchKernel`, and cudapipe issues thousands of them a frame, so a traced
run manufactures exactly the host-side gap the question is about. A trace of
`multithreading` reported the GPU 66% busy and simultaneously reported more GPU
kernel time per frame than the untraced frame takes end to end, which is the
tell. `cp_gpu_busy.sh` asks `nvidia-smi` over a run with nothing attached.

**It takes seconds, not frames, and that matters.** A sample's process spends a
couple of seconds on start-up, shader compilation, the warm-up second and
teardown with the GPU idle. Sampling across all of it averages the render loop
together with that idle and reports a number far below the truth. Measured
across a sixty-frame run `multithreading` reads as 77% busy and
`dynamicuniformbuffer` as 66%; over seventeen seconds of steady state both read
as 84-85%. The script computes the frame count from the sample's recorded cost,
trims the ramp at each end, prints the window it actually measured, and refuses
a window with fewer than sixty samples rather than reporting one.

The measurement still carries a couple of points of run-to-run spread —
`instancing` gives 35-40% across runs — so read it as a band, not a figure.

### 2. Is the GPU full, or merely occupied?

```bash
METRICS=1 cp_profile.sh particlesystem mylabel 100
```

On-die counter sampling with the CUDA API left uninstrumented. Because nothing
intercepts `cuLaunchKernel`, this is the one profile of this driver that its own
launch count does not distort — which is why it can be trusted where question 1
says a profiler cannot be. Needs GPU counter permission; if the GPU is not
listed, check `/proc/driver/nvidia/params` before suspecting the tool.

**The pair that matters is `SMs Active` against `SM Issue`.** `particlesystem`,
100 frames:

```
                                                   med     p90     max
GR Active [Throughput %]                            94      99     100
SMs Active [Throughput %]                           36      53     100
SM Issue [Throughput %]                              5      11      19
Compute Warps in Flight [Avg Warps per Cycle]        8      13      94
DRAM Read Bandwidth [Throughput %]                   0       1       1
PCIe RX Throughput [Throughput %]                    1       1      21
```

`GR Active` 94% is the same figure `cp_gpu_busy.sh` gets by polling
`nvidia-smi` — two unrelated mechanisms agreeing, which is what licenses
reading the rest of the column. And the rest says the machine is nearly empty
while it is occupied: a third of the SMs active, eight warps a cycle in flight,
DRAM and PCIe flat. That is not an expensive kernel. It is a long sequence of
small ones that never fill the device, which is what 260 peel passes a frame
look like from outside.

**So "94% busy" never meant the GPU was working — only that a kernel was
resident.** Question 1 says the host is not holding the device up; this one says
whether the device is doing anything with the time it has. `PHASE_1A.md` reads
`particlesystem` as "kernel-bound at 94%" and concludes stages 2 and 3 must get
faster; these counters point instead at the pass structure, which is a phase 3
question. Neither reading is proven by this tool.

It names no kernel — that is question 3 — so a finding here is a hypothesis
until the traced run or `ncu` attributes it to something.

**The whole sweep, measured this way, and it is one shape.** Each sample sized
to about 12 seconds of rendering, median of the active window:

| sample | ms | GR | SMs | **issue** | warps | DRAMr | DRAMw |
|---|---|---|---|---|---|---|---|
| particlesystem | 51.98 | 94 | 40 | **6** | 9 | 0 | 1 |
| multithreading | 29.74 | 86 | 21 | **2** | 7 | 0 | 1 |
| gltfscenerendering | 15.10 | 97 | 27 | **3** | 5 | 1 | 1 |
| dynamicuniformbuffer | 12.35 | 86 | 12 | **2** | 5 | 0 | 1 |
| bloom | 12.06 | 83 | 11 | **2** | 5 | 1 | 1 |
| **instancing** | 7.01 | **100** | **100** | **2** | **71** | **13** | **17** |
| multisampling | 3.57 | 98 | 9 | **1** | 1 | 1 | 1 |
| vulkanscene | 2.85 | 87 | 18 | **2** | 6 | 1 | 1 |
| pushconstants | 2.30 | 84 | 22 | **1** | 5 | 0 | 1 |
| pbribl | 1.67 | 77 | 23 | **3** | 6 | 1 | 1 |
| texturemipmapgen | 1.43 | 67 | 19 | **2** | 6 | 1 | 3 |
| texturecubemap | 1.10 | 4 | 2 | 1 | 1 | 1 | 3 |
| computeshader | 0.87 | 1 | 0 | 0 | 0 | 0 | 2 |
| negativeviewportheight | 0.86 | 1 | 0 | 0 | 0 | 0 | 5 |
| texture | 0.74 | 1 | 0 | 0 | 0 | 0 | 2 |
| texture3d | 0.71 | 1 | 0 | 0 | 0 | 0 | 3 |
| triangle | 0.67 | 1 | 0 | 0 | 0 | 0 | 1 |

Three things fall out of it, and none is visible from `GR Active` alone:

- **The device issues on 1-6% of cycles on every sample that uses it at all.**
  `GR Active` reads 83-100% for the eleven samples above `texturecubemap`, and
  `cp_gpu_busy.sh` reports the same figure because it is the same quantity. So
  **every "kernel-bound" verdict in this repository means only that a kernel was
  resident**, and the pass records that read one as "the host is not the
  problem, make the kernels faster" were reading it wrong. `SMs Active` at 9-40%
  says most of the machine has no work at all: the grids are small because the
  work is per draw, and stage 1 of a small draw is one or two blocks on a card
  with 170 SMs.
- **`instancing` is the one sample that fills the machine and the only one with
  real DRAM traffic** — 100% SMs, 71 warps in flight, 13/17% bandwidth — and it
  still issues on 2% of cycles. Full and stalled is a different problem from
  empty and stalled, and it is the only sample in the set where `ncu` and
  question 3 are the right next step rather than the launch structure.
- **Six samples never occupy the GPU at all.** `triangle`, `texture`,
  `texture3d`, `computeshader`, `negativeviewportheight` and `texturecubemap`
  read 1-4% `GR Active` over a fifteen-second window that is almost entirely
  render loop. Their cost is host-side in its entirety, and `triangle` at
  0.67 ms a frame for one triangle is the driver's per-frame floor rather than
  anything about drawing.

Reading this table next to the sweep totals is what the two together are for:
the samples worth the most time are the ones with the emptiest device.

**Same seconds-not-frames trap as question 1, and it bites harder**, because
the default is ten frames. Ten frames of `instancing` gave a median `GR Active`
of 1%: its render loop is a sliver of a process that spends its time compiling
shaders and tearing down, so the window averaged mostly idle and every row read
low. The summary prints a warning below 50% busy — raise `FRAMES` until it
stops firing before reading anything under it.

`METRICS_HZ` sets the sampling rate (default 10000), `METRICS_SET` the metric
set (default is auto-selected; `gb20x` on this machine, `gb20x-top` is
lighter), `METRICS_DEV` the device. The report is kept beside the traced one as
`SAMPLE.metrics.nsys-rep`, so both can exist for the same label.

### 3. Which kernel owns the frame?

```bash
cp_profile.sh instancing mylabel 10
$MESA/venv/bin/python3 cp_prof_kernels.py \
    build/prof/mylabel/instancing.sqlite --frames 10
```

`cp_profile.sh` runs Nsight Systems and reduces the trace to which kernel owns
the frame, what the host is spending its time in, and what the memory traffic
is. It keeps the `.nsys-rep` so a later profile can be diffed against an
earlier one rather than re-argued.

**`FRAMES` is the number of frames in the trace, and it did not used to be.**
The profile rendered `--benchwarmup 1` as well, so that what it traced would be
steady state — but nsys traces the whole process, so the effect was to put about
seventy-seven frames of a *static* scene (warm-up renders frame 0 repeatedly
without advancing it) in front of the ones asked for. `FRAMES` then described
about a twentieth of the trace, and per-frame arithmetic off it was wrong by
that factor: a four-frame profile of `dynamicuniformbuffer` reported 2,531 draws
a frame where the sample makes 125.

Dropping the warm-up costs nothing it was supposed to buy. With and without it
kernel shares agree to a percent and the per-launch averages are identical to
the nanosecond, because the fixed-cost kernels this driver is full of do not
care which frame they are in.

**What it did buy, by accident, was frames to dilute the context.** Creating and
destroying the CUDA context is a one-off cost inside every trace, and a short
one does not spread it: ten frames of `dynamicuniformbuffer` puts `cuCtxCreate`
and `cuCtxDestroy` at 61% of host API time with the frame's own work a rounding
error underneath. So the summary now warns when they exceed 20% and says to
raise `FRAMES` before reading that section. **The kernel summary is unaffected**
— only the API section is — which is why it warns rather than refuses.
`WARMUP=1` restores the old behaviour for a trace that has to match an older
one.

`cp_prof_kernels.py` is what makes the result readable, and is not optional:
**every shader cudapipe compiles is a CUDA kernel named `main`**, so the vertex
and fragment stages land in one row of `nsys`'s own summary and the largest
entry in every profile is uninterpretable. Splitting by grid size separates
them — the vertex shader is launched over the vertex count and the fragment
shader over a fixed worst case.

It also flags **kernels whose duration does not vary with the draw**. That is
the shape of nearly every defect found in this driver so far: work sized to the
worst case the host can compute rather than to what the draw does. A kernel
with a coefficient of variation near zero across thousands of launches is doing
the same amount of work whatever it was asked to draw.

Remember that tracing inflates short kernels much more than long ones, so read
shares rather than absolute times, and never compare a traced total to an
untraced one.

#### The driver names its own timeline

`cp_profile.sh` traces `--trace=cuda,nvtx` and runs the sample with
`CUDAPIPE_NVTX=1`, so the driver pushes an NVTX range around each draw and each
stage — `vertex`, `raster`, `interp`, `fs`, `writeback`, one per `pass`, and a
`flush` mark per frame. `cp_prof_nvtx.py` aggregates them:

```
stage                    count   ms total   us/draw  % of draw
draw (all)               10125      421.5     41.63     100.0%
pass 0                   10125      236.8     23.39      56.2%
vertex                   10125      180.6     17.84      42.9%
raster                   10125       80.9      7.99      19.2%
```

That is `dynamicuniformbuffer`, and it says the host spends 43% of its per-draw
issue time on the vertex stage — a fact no kernel summary contains, because the
cost is in submitting the work rather than in running it. The vertex block makes
about seven of the sixteen CUDA calls a draw makes, and that ratio is what the
share is measuring.

**Per draw, not per frame.** The trace covers the whole process and
`cp_profile.sh` renders `--benchwarmup 1` before the frames it was asked for, so
a four-frame profile of a 12 ms sample holds about eighty-four frames. Dividing
by the number given to `cp_profile.sh` overstates every per-frame figure by
twenty times, which is exactly what happened the first time this table was
printed. Draws are counted in the trace itself, so they are the denominator.

**These are issue times, not device times.** A range closes when the launches
are queued, not when the GPU finishes them. That is the useful reading for a
launch-bound frame and the wrong one for anything else; `CUDAPIPE_DEBUG_TIME`
and its CUDA events are the device measure. Read the two together — a stage
costing far more to issue than to run is exactly what launch-bound looks like
from the host side.

The ranges cost about 4% on `multithreading` when enabled and nothing when not,
so they are off unless `CUDAPIPE_NVTX` is set. `NVTX=0 cp_profile.sh ...` turns
them off for a trace that has to be compared against an older one.

This is also the structural fix for the `main` problem: `cp_prof_kernels.py`
splits kernels by grid size because every compiled shader is named `main`, which
works from outside. A range says which stage issued a launch, so the trace names
itself.

`NCU=1` runs Nsight Compute instead, for the counters `nsys` cannot give:
occupancy, memory throughput and warp stall reasons. It used to fail here with
`ERR_NVGPUCTRPERM` — reading GPU performance counters needs a root-level
modprobe option — and that is now set:

```bash
cat /etc/modprobe.d/nvidia-profiling.conf   # NVreg_RestrictProfilingToAdminUsers=0
grep RmProfilingAdminOnly /proc/driver/nvidia/params   # must read 0
```

If a fresh kernel or driver install ever drops that file, `ncu` starts failing
with `ERR_NVGPUCTRPERM` again and `nsys --gpu-metrics-devices` stops listing
the GPU. Both are the same permission; check `/proc/driver/nvidia/params`
before concluding anything about the tools.

**Reach for it only once a sample is known to be kernel-bound**, which
`cp_gpu_busy.sh` is what establishes. Every win in this driver so far came from
finding work that should not have been done at all, and that is a question
about launch counts and grid sizes rather than about warp stalls — `ncu` is for
after the profile already names one kernel and the question has become why
*that kernel* is slow. It replays every launch several times to gather the
counter set, so it is perhaps fifty times slower, and its totals are
meaningless for comparing anything.

### 4. Is the cost in the kernels at all?

Not everything shows up as kernel time. The largest single win of the phase 1a
pass was 5.3 ms of a 27.6 ms frame sitting in `cuMemAlloc` and `cuMemFree` —
visible in the API summary, invisible in the kernel summary, and explained by
neither. When the kernel breakdown does not add up to the frame, read the
`--- CUDA API ---` section of the profile before theorising.

And when the kernel breakdown *does* add up but names something implausible,
read the `--- memory ops ---` section. **A kernel that touches a managed page
the host has just written stalls until the page arrives, and the stall is
charged to the kernel.** So the cost lands on whichever kernel touches the page
first, which need not be the one doing anything wrong. `instancing` reported
`cp_vertex_fetch` at 58.7% of GPU time with a median launch of 2.48 ms —
absurd for gathering three attributes, and entirely the four managed id arrays
the host had built for it. The line that named it was 15,713 unified
host-to-device migrations per ten frames, one table further down; the fix took
the median launch to 93.6 µs without touching the gather at all. See
`../INSTANCING.md`.

The general form: **a kernel whose duration is absurd for the work it
describes is reporting somebody else's page faults.** Ask what the host wrote
just before it ran.

---

### The Nsight Systems agent skill pack

Nsight Systems 2026.4 ships a skill pack for AI agents at
`/opt/nvidia/nsight-systems/2026.4.1/skills/nsight-systems/SKILL.md`. It is a
gated evidence CLI: `doctor`, `inspect-cli` (exact flag syntax for the
installed version), `report-context`, `report-fact`, `report-query` (bounded
SQL over a report), `report-doctor`, plus recipes. Run its bootstrap first —
`sh scripts/determine_local_unix_platform.sh` — and use the Python it reports,
not the one on `PATH`.

Worth using for CLI syntax and for report queries. Three things to know before
relying on it:

**`search-docs` and `lookup-recipes` are broken out of the box**, and this
install has been repaired so that they work. The shipped `manifest.json`
records content hashes computed against a different build — all 496 disagree
with the files beside them — so both commands fail with `content hash
mismatch: SKILL.md` until `cp_nsys_skill_fix.py` rewrites the digests. **An
Nsight Systems upgrade or reinstall puts the broken manifest back**, so run the
script again then; it checks by default and exits 0 when clean, which is also
how to find out whether a newer release fixed this upstream. Everything that
talks to the installed `nsys` or to a report works without the fix.

This matters more than two broken commands sounds: they are the only sanctioned
route to the reference corpus — `references/notes/llm-analysis-pitfalls.md`,
`references/notes/sql_query_tips.md`,
`references/curated/investigation_methodology.md` — and SKILL.md forbids
reading them by hand.

Quote multi-word `--query` arguments; unquoted they split into extra
positionals and the tool reports `unrecognized arguments`, which reads like a
wrong command rather than a shell mistake.

**Its per-kernel summary is wrong for this driver, and confidently so.**
`report-fact --intent kernel_summary` returns one row named `main` with the
vertex and fragment stages summed, for the reason described above. Use
`report-query` with `GROUP BY gridX`, or `cp_prof_kernels.py`.

**It cannot know that traced evidence misleads here.** It reasons from what
nsys recorded, and question 1 above is the standing exception. Where the skill
and `cp_gpu_busy.sh` disagree about whether a frame is host-bound, the untraced
measurement wins.

Much of the corpus is graphics, stutter and Windows scheduling material that a
CUDA-compute rasterizer never touches, so the useful fraction is smaller than
497 files suggests.

### Versions

Three versions of Nsight Systems are installed and `$CUDA/bin/nsys` is a
wrapper pinned to the oldest, so `cp_profile.sh` resolves the newest under
`/opt/nvidia/nsight-systems` itself and records which it used in the summary.
`NSYS=/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys` names an older
one, which is what a stored `.nsys-rep` has to be re-read against. **Do not
compare host-side time between traces taken with different versions**: CUPTI's
per-launch cost differs between them and this driver issues thousands of
launches a frame. The GPU-busy question still belongs to `cp_gpu_busy.sh`,
which attaches no profiler at all.

## Iterating on performance

`cp_iterate.sh LABEL [COMPARE_LABEL]` is one iteration end to end: build, time
the frames, render and store them, compare them to the stored NVIDIA reference,
print the cost delta, and write the whole thing down.

```bash
DESC="what this tried" cp_iterate.sh mylabel previouslabel
```

**Nothing below is done by hand.** One command produces the whole directory —
the frames, the timing, the verdict, the png conversion and the
`iteration.json` record. There is no step where files are moved into place or a
json is edited, and there should not be: the point of the record is that it
describes the run that actually happened, and a hand-maintained one describes
what someone remembered afterwards.

It times `BENCH_FRAMES` (600 by default) and stores `FRAMES` (60) — see "the
frame count is the one flag that may differ" above for why those are not the
same number.

Everything an iteration produced lands under `build/iter/LABEL`:

```
build/iter/
    nvidia/<sample>/frame0000.png     the reference, rendered once
    llvmpipe/<sample>/frame0000.png   the calibration renderer
    LABEL/
        <sample>/frame0000.png        the frames this build rendered
        bench/_bench.csv              the timed pass, and a csv per sample
        _render/                      the storing pass's own gpu and timing csvs
        verdict.txt  delta.txt  _commit.txt
        iteration.json                all of it as one record
    iterations.json                   every iteration.json, collected
    iterations.html                   the summary, a row per iteration
    perf.html                         one iteration in full, ?iter=LABEL
```

`build/iter` is therefore exactly the root `cp_compare_frames.py` wants — one
directory per thing being compared — and an iteration is one directory that can
be copied, kept or deleted whole.

### Recording what the device was doing, not just what it cost

```bash
METRICS=1 DESC="..." cp_iterate.sh mylabel previouslabel
```

`ms/frame` cannot tell a kernel that got faster from one that stopped being
launched, and for a change whose point is to fill the device — batching draws,
growing a grid, raising occupancy — `issue` and `sms` move before the frame time
does. `METRICS=1` runs `cp_metrics_sweep.sh` after the timed pass and puts a
row per sample into `metrics.csv` and into `iteration.json`:

```
sample         ms    gr  sms  issue  warps  dram_r  dram_w
multithreading 29.74 86  21   2      7      0       1
instancing      7.01 100 100  2      71     13      17
```

It costs about five minutes on top of a pass that already takes ten, which is
why it is opt-in. `METRICS_SECONDS` sets the per-sample window (12 by default);
the seconds-not-frames trap from question 1 applies here identically.

An iteration without it records **no** metrics rather than zeroes — a row of
zeroes reads as "the device was idle" instead of "nobody measured", which is
the same distinction `correctness.gate_ran` exists for.

Run by hand it refuses a label whose `_commit.txt` does not match `HEAD`, since
it measures whatever is built now and files it under that label; `FORCE=1`
overrides. `cp_iterate.sh` calls it straight after its own build, where the two
always agree.

**Set `DESC`.** A label and a number stop meaning anything within a day of the
run. What the iteration was *trying* is the part nobody can reconstruct
afterwards, and it is what the summary page shows.

### Reaching for the tool directly

`cp_iterate.sh` calls `cp_iter_report.py record` itself, so the only reasons to
run it by hand are these:

```bash
# refresh the pages after recording — the only one needed routinely
cp_iter_report.py page

# fix or add a description on an iteration already run, without re-running it
cp_iter_report.py record LABEL --against PREV --desc "..." --note "..."

# ppm -> png for an older tree, or one interrupted before it converted
cp_iter_report.py convert [LABEL ...]
```

`record` is idempotent and re-reads everything from the iteration's own files,
so re-running it never invents a number — it can only pick up a description, a
note, or a series (the llvmpipe calibration) that was not computed the first
time.

### The pages

```bash
cp_iter_report.py page     # rewrites iterations.json; the html never changes
```

**`cp_iterate.sh` runs this itself**, so an iteration appears in the history
without anyone remembering to. Run it by hand only after editing a record or
adding one from elsewhere. It used to be a separate step, and the consequence
was the one this script guards against everywhere else: an iteration that ran,
recorded itself and passed the gate was simply absent from the page, because
`iterations.json` is rebuilt from the per-iteration records rather than
appended to.

Both pages are static and fetch their data at load time, so they have to be
served over HTTP rather than opened from a `file://` URL. `iterations.html` is
the history — description, total, delta, and whether the correctness gate ran —
and each row links to `perf.html?iter=LABEL`, which is that iteration in full:
the cost and verdict table, GPU load and memory per sample, frame time over the
run, and differing pixels per frame against the reference.

**The GPU table is cut out of the pass-wide samplers by process name.** Both
samplers run for the whole timed pass rather than per sample, so the page reads
`bench/_gpu_procs.csv` — which names the process behind every poll, and the
process is the sample's binary — to find each sample's slice, and averages
`bench/_gpu.csv` over it. The load figure trims that slice the way
`cp_gpu_busy.sh` does, dropping the idle ends and then a tenth of what is left
at each end, so it is the render loop and not the several seconds of start-up
and shader compilation around it; `multithreading` reads 84% there, which is
what the untraced measurement above gives. Memory is not trimmed, because
memory held is held whether it is touched or not, and both the card's
`memory.used` and the sample's own share are shown — on an otherwise idle card
those differ by the ~170 MiB of context the driver keeps.

**Read the poll count before the percentage.** At 2 Hz a sample that runs for
three seconds contributes four polls, and a mean of four polls is not a
utilisation figure. Rows under twelve — the six seconds of steady state
`cp_gpu_busy.sh` refuses to report below — are greyed for that reason, and say
which end of the scale a sample is at and nothing finer. `cp_gpu_busy.sh SAMPLE
20` is how one of those gets an answer. This is a by-product of a pass that ran
anyway, not a substitute for the measurement gate.

The per-frame charts are the reason to look. `texture3d` reads as 2 differing
pixels at frame 0 and 3,090 at frame 24, and only one of those is visible in a
table.

Each difference chart carries a second, dashed line: **llvmpipe against the same
reference**. That is what makes a residual readable, and it is the calibration
this document argues for below. At frame 47 of `gltfscenerendering` cudapipe
differs by 59,939 pixels and llvmpipe by 91,091, so that residual is what
software rasterization costs; `texture3d` peaks at 3,090 against llvmpipe's 0,
so that one is a cudapipe defect.

Under the charts is the frame inspector: the reference, this iteration and
llvmpipe side by side, with each one's difference beneath it. Hovering a
renderer's frame flips it to the reference, and clicking opens it at full
resolution where the same hover works. A frame is linkable —
`perf.html?iter=LABEL&sample=NAME&frame=N`, plus `&full=test` to open it large —
so a finding can be pointed at rather than described.

**Frames are stored as PNG.** The samples write PPM, and `cp_iterate.sh`
converts in place and drops the PPM once the gate has read it: a browser cannot
display PPM, and a 1280x720 PPM is 2.7 MB whatever it holds against 150 KB to
2 MB as PNG. The conversion is bit-exact, and `cp_compare.read_image` reads
either. `cp_iter_report.py convert [LABEL ...]` does it for an older tree.

### Two things the record exists to catch

**A gate that did not run.** `cp_compare_frames.py` exits 1 both when a sample
regressed and when the comparison never happened, so the status cannot tell
them apart — and a cost delta prints underneath either way. `cp_iterate.sh`
refuses to report a cost number without a verdict table, and `iteration.json`
carries `correctness.gate_ran` so the summary page can say so on the row.

**A label reused.** Reusing one overwrites an iteration in place, and the label
that suggests itself for a change is the same one that suggested itself last
time something touched that code — so the record most likely to be destroyed is
the one most useful to compare against. `cp_iterate.sh` refuses an existing
label; `FORCE=1` when replacing it is the intent.
