# The sample sweep: running it, checking it, reading it

cudapipe is checked differentially. The Sascha Willems samples at `~/git/Vulkan`
render offscreen and reproducibly, so the same binary is run against the NVIDIA
ICD, against cudapipe, and against lavapipe/llvmpipe, and the frames are
compared. That is a far stronger signal than a pass/fail suite, and it has found
every rendering bug fixed so far.

There are two sweeps, and they answer different questions.

| | frames | answers |
|---|---|---|
| **single frame** | 1 | does the driver draw the scene |
| **animated** | 60 | does it keep drawing it |

The single frame sweep is the cheap one and catches most things. The animated
one catches what a still image cannot: state that leaks between frames, a
particle system settling, anything that only goes wrong once the camera moves.
Two of the regressions in the set are invisible at frame 0.

---

## The pieces

| Tool | What it does |
|---|---|
| `headless_streamer_samples.txt` | the sample set — one entry per capability the capture needs, with the mapping in comments |
| `cp_perf_run.sh` | one driver's pass: frames, timing, GPU load |
| `cp_compare.py` | two images, or the PPM/PNG reader the rest import |
| `cp_compare_frames.py` | every frame of every sample against a reference, with a pass/fail verdict |
| `cp_gallery.py` | HTML for the single frame sweep, two or three drivers |
| `cp_perf_report.py` | HTML for the animated sweep: cost, GPU, per-frame differences, frame inspector |

Everything is Python standard library or shell. `cp_perf_report.py` additionally
needs `ffmpeg`, which does all the image work.

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
    -o build/compare/perf60.html
```

Roughly five minutes for the three passes and half a minute for the report.
It writes about 8 GB of frames and 2.7 GB of report images.

Benchmark mode instead of frames — the samples' own timing loop, no files
written, fps reported:

```bash
$T/cp_perf_run.sh cudapipe "$ICD" $R/bench_cuda 0 1
```

---

## What each tool records

### `cp_perf_run.sh`

Per sample: wall clock, user and system CPU, peak RSS and exit status, into
`_timing.csv`. For the whole pass: GPU utilisation, memory used and total, and
power at 2 Hz into `_gpu.csv`, and per-process GPU memory into
`_gpu_procs.csv`.

The GPU samplers cover the pass rather than each sample deliberately. Several
samples finish in well under a second, so a per-sample capture would get a
single poll; a sample's slice is found by timestamp against `_timing.csv`.
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
