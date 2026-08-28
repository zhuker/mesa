# Testing cudavk: the correctness gates, and how to read them

This document answers one question: **is this change correct?** It merges the
two documents it replaces: the old `tests/TESTING.md`, which described how the
sweeps are run, and the old `CUDAVK_VALIDATION.md`, which described how a
change is accepted. Both were removed when this file was written, so
`git log --follow` is where their history lives.

Cost is measured with the same runs and the same tools, but the mechanics of a
measured iteration — build, capture replay, timing convention, the profiling
decision tree — live in `docs/cudavk/WORKFLOW.md`. This document references
that one rather than repeating it.

**Where a source disagreed with the tree, the tree won.** One case is live:
`cp_iterate.sh`, `cp_perf_run.sh` and `cp_gpu_busy.sh` still resolve llvmpipe
(and the retired Gallium driver) under `$MESA/build-cudavk/src/gallium/targets`,
which a `-Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers=` build does not
produce; the llvmpipe reference on this machine is the separate release build
at `~/mesa/build-lvp-release`, so pass its ICD path explicitly.

---

## 1. The rule everything else follows

**An implementation is never its own correctness oracle.** A cudavk baseline
compared against a cudavk candidate is change detection and a nondeterminism
probe. It is not acceptance. Acceptance needs an image from an **external**
renderer, produced from the same plan in the same session:

| reference | what it is good for |
|---|---|
| **release llvmpipe** (lavapipe, `~/mesa/build-lvp-release`) | the primary GFXR image reference; software rasterisation done maturely, so a residual it shares is the cost of software rasterisation rather than a cudavk defect. Use the **release** build: the debug one asserts in anisotropic sampling on the old capture |
| **NVIDIA** | the stored reference for the 18-sample sweep, and the independent check when llvmpipe cannot run the path or when a shared frontend makes llvmpipe circular |

A zero delta against a cudavk baseline only proves the change preserved the
baseline, **including any defect the two share**.

Two consequences that used to be written down as caveats and are now simply
the rule:

* **There is no second in-tree frontend left to cross-check against.** The
  Gallium-hosted driver was removed (`docs/cudavk/GALLIUM_RETIREMENT.md`), so
  the old advice "never accept native-versus-Gallium agreement" now has nothing
  to reach for. External or nothing.
* **llvmpipe is only external as long as nothing shared moved.** cudavk no
  longer links lavapipe, so an ordinary cudavk change cannot move both. A NIR
  or common-Vulkan change still can; after one, regenerate the llvmpipe
  reference and use NVIDIA as the independent check. Record the llvmpipe Mesa
  commit and build type with every reference set.

---

## 2. The gates

| # | gate | what it covers | typical cost |
|---|---|---|---|
| 1 | **native test suite**, `meson test -C build-cudavk --suite cudavk` | the API boundary, object lifetimes, the pixel behaviour of specific driver mechanisms, and the fault paths | minutes |
| 2 | **18-sample sweep**, 60 stored frames per sample against the stored NVIDIA reference | that the common paths still draw the scene, and keep drawing it over an animation | ~10 min through `cp_iterate.sh` |
| 3 | **GFXR sentinel frames** from both captures against the frozen llvmpipe envelope | real-application paths the samples cannot reach | ~10 min per capture |
| 4 | **a device-side equivalence gate**, when the change alters what the GPU computes | that the new arithmetic agrees with the old one over a whole replay, not on a spot check | one replay per arm |

Plus one standing rule: **a timing replay is not a correctness gate.** A replay
that completes every frame proves timing and completion. It captures no pixels.
The tiled measurements of 2026-08-18 completed both captures and extracted no
Crossroads sentinel at all; they are timing results only
(the removed `VALIDATION.md`, in git history).

Which gates a change needs is in `WORKFLOW.md` §9, "What a kept change has to
show". This document is the authority on how gates 1-4 are run and read.

---

## 3. Before any gate

```bash
export PATH="$HOME/mesa/venv/bin:$HOME/vulkan-sdk/1.4.357.1/x86_64/bin:$PATH"
./venv/bin/ninja -C build-cudavk
./venv/bin/python3 src/cudavk/tests/cp_debug_doc.py --check
./venv/bin/python3 src/cudavk/tests/cp_no_getenv.py --check
```

The second check says `FLAGS.md` still matches the registry. The third says the
registry is still the only way into the environment: it fails on a `getenv` in
driver code, allowing only `cp_debug.c`, `tests/` and `samples/`. Both are
cheap, need no GPU, and exist because the rule they enforce was written in
prose first and nineteen switches grew around it anyway.

**Use the repository interpreter**, `$MESA/venv/bin/python3`, for every image
tool. `cp_compare_frames.py` needs numpy and only the venv has it;
`cp_compare.py` decodes through Pillow when it is importable and falls back to
its own PNG reader when it is not, about a hundred times slower per frame.
This is not style. `cp_iterate.sh` once ran the comparison with an ambient
`python3`, which printed an import error where the verdict table goes and a
cost delta underneath as usual — and both "a sample regressed" and "the check
never ran" exit 1, so the status could not tell them apart
(`docs/cudavk/history/PHASE_1A.md`).

**A successful build says nothing about the CUDA kernels.** The `.cu` files are
stringified at build time and compiled by NVRTC at device creation, so a kernel
that does not compile builds clean and fails at first use as a driver error.
Smoke-run one test before starting a pass.

**A flag-gated change is not exempt from the gates.** Adding a kernel to a
`.cu` file changes the translation unit every other kernel in that module is
compiled in, so inlining and floating-point contraction can move in code that
was not edited. That is the failure behind the watertight-coverage work:
contraction broke the edge function's antisymmetry and cracked a shared edge
for its whole length (`docs/cudavk/history/HANDOFF.md`). "Default off" is an
argument about the new path, not about the old one.

### Flags: three runs, not two

The driver has **116 switches** (`src/cudavk/FLAGS.md`, generated from the
registry in `src/cudavk/cp_debug.c`). Two boolean kinds exist and the
difference bites: **presence** flags are set by the variable existing, so
`CUDAVK_DEBUG_DRAW=0` turns tracing **on**, and **value** flags read the value.
`CUDAVK_HELP=1 <any vulkan app>` prints which kind each one is, with what it
resolved to in that process. **Use `unset` for a control, never `FLAG=0`.**

| run | flag | must equal |
|---|---|---|
| baseline | unset | the reference |
| candidate, flag **off** | unset | the baseline, exactly |
| candidate, flag **on** | set | judged on its own merits |

The middle row is the one that gets skipped and the one that catches the most.
It separates "the new path is wrong" from "the new code perturbed the default
path" — which happens for reasons that have nothing to do with rendering:
instrumentation that allocates from the upload arena shifts every later offset
in it, and a new kernel changes the NVRTC translation unit as above.

When the new path is already the default, the revert flag goes on the
**control** arm instead — `CUDAVK_NO_TEXTURE_CACHE=1`,
`CUDAVK_NO_FUSED_VFETCH=1`, and so on.

---

## 4. Gate 1: the native test suite

```bash
./venv/bin/meson test -C build-cudavk --suite cudavk                    # all 72
./venv/bin/meson test -C build-cudavk --print-errorlogs cpvk_batchblend # one
```

**72 tests**, and **72/72** is the state the branch is kept in: 54 C tests,
four `cpvk_vfetch` modes, thirteen Python gates and one fault-mode rerun of a C
test, counted from the registry in `src/cudavk/meson.build`. Iteration 28
recorded 65/65 in the default state and 65/65 with every revert flag set
(`docs/cudavk/history/PERF16_ITERATIONS.md`). This paragraph said 67 while the
registry held 71 and the commit messages quoted 71/71, which is what a count
carried forward by hand does; recount it from the registry every time it is
touched.

**Every one of them passes, `cpvk_gather` included.** That test was written
against an open gap — `nir_texop_tg4` was not in the supported set in
`cp_nir_to_llvm.c`, so every `textureGather()` returned the constant
`0 0 0 1` — and it was registered as an ordinary failing test rather than
parked outside the suite, because a gap nothing runs is a gap nobody fixes.
The sampler serves gathers now and it passes, on the same pixels lavapipe and
NVIDIA produce. **Any failure is a regression.**

**Every test is registered `is_parallel : false`, and that is load-bearing.**
Two CUDA contexts at once produce false OOM and false device-lost, because each
process owns large arenas. Never run the suite concurrently with itself, with a
replay, or with a sweep. A "failure" from a parallel run is a report about the
machine, not about the driver
(`/tmp/perf16/iter27-vsfusion/handoff.md`).

A flag-conditioned run is just the environment:

```bash
CUDAVK_NO_TEXTURE_CACHE=1 ./venv/bin/meson test -C build-cudavk --suite cudavk
```

**Do not pass a shader path positionally to a test you have not read.** Several
tests take the *output* path as `argv[1]`; passing a shared `.spv` there
overwrites it with a PPM and every later test fails with something unrelated.

### `cp_launch_audit`

Wired into the suite. It fails if a raw `cuLaunchKernel` appears outside the
allowlisted launch wrapper, which is what keeps every launch countable by the
telemetry the performance work depends on.

### The gate style: a negative control that must fail

Fourteen of the 65 entries are Python gates rather than tests. They exist
because a test can pass for the wrong reason. The pattern is always the same:

1. The gate **selects its own state** rather than inheriting it. Each one pops
   the flags it cares about out of the environment and sets what it means, so
   a suite run in the reverted state cannot silently report that a mechanism
   which was never built agrees with itself
   (`src/cudavk/tests/cpvk_vfetch_gate.py`).
2. It requires the positive result — the pixels, the statistics line, the exit
   code.
3. It requires a **negative control to fail**. `cpvk_vfetch_seed_gate.py` says
   it plainly: the canary must pass with the fused path on and must fail with
   `CUDAVK_VFETCH_SKIP_SEED=1`, because "a gate that cannot fail is not
   covering anything".

The fault gates apply the same shape to error handling.
`cpvk_texture_cache_fault_gate.py` injects a failed cache-table upload with
`CUDAVK_TEXTURE_CACHE_FAIL_TABLE_UPLOAD_AT=1` and then requires **all** of: a
nonzero exit, `VK_ERROR_DEVICE_LOST` propagated, `fs_attempts=0` — the fragment
shader must not have been attempted after the fault — and the absence of the
normal pixel `PASS` line. A fault path that quietly recovers into wrong pixels
fails that gate.

Write new gates this way. If it cannot be made to fail on purpose, it is not
evidence.

---

## 5. Gate 2: the 18-sample sweep

The Sascha Willems samples at `~/git/Vulkan` render offscreen and reproducibly,
so the same binary runs against NVIDIA, against cudavk and against llvmpipe,
and the frames are compared. The set is one entry per capability the
HeadlessStreamer capture needs, with the mapping in the comments of
`src/cudavk/tests/headless_streamer_samples.txt` — **18 samples**, and dropping
one leaves a requirement with no test.

Three sweeps answer different questions:

| | frames | answers |
|---|---|---|
| **single frame** | 1 | does the driver draw the scene |
| **animated** | 60 | does it keep drawing it |
| **benchmark** | 600 | what that same work costs |

The single-frame sweep is the cheap one and catches most things. The animated
one catches what a still image cannot: state that leaks between frames, a
particle system settling, anything that only goes wrong once the camera moves.
Two of the regressions in the set are invisible at frame 0.

The benchmark sweep renders the same work as the animated one and stores none
of it. That split is not a convenience: storing a 1280x720 frame takes about
10 ms, which is longer than several of these samples spend rendering one, so a
pass that writes images cannot be timed.

### The normal way to run it

One command, through the iteration helper:

```bash
DESC="what this tried" src/cudavk/tests/cp_iterate.sh mylabel previouslabel
```

It builds, times `BENCH_FRAMES` (600), stores `FRAMES` (60), compares the
stored frames against the stored NVIDIA reference, prints the cost delta and
writes `iteration.json`. `WORKFLOW.md` §6 has the knobs and the output layout.
Nothing below is done by hand when this is available; the point of the record
is that it describes the run that actually happened.

Two things the record exists to catch:

* **A gate that did not run.** `cp_compare_frames.py` exits 1 both when a
  sample regressed and when the comparison never happened. `cp_iterate.sh`
  refuses to print a cost number without a verdict table, and `iteration.json`
  carries `correctness.gate_ran`.
* **A label reused.** Reusing a label overwrites an iteration in place, and the
  label that suggests itself is the one that suggested itself last time
  something touched that code. `cp_iterate.sh` refuses an existing label;
  `FORCE=1` is the intent.

### Running the sweeps by hand

Single frame, three drivers, then the gallery:

```bash
cd ~/git/Vulkan
S="$(grep -v '^#' ~/mesa/src/cudavk/tests/headless_streamer_samples.txt | tr '\n' ' ')"
CP=~/mesa/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json
LVP=~/mesa/build-lvp-release/src/gallium/targets/lavapipe/lvp_devenv_icd.x86_64.json

SAMPLES="$S" VALIDATION=0 OUT=build/compare/ref ./run_offscreen.sh
SAMPLES="$S" VALIDATION=0 OUT=build/compare/cuda \
  VK_ICD_FILENAMES=$CP ./run_offscreen.sh
SAMPLES="$S" VALIDATION=0 OUT=build/compare/llvmpipe \
  VK_ICD_FILENAMES=$LVP ./run_offscreen.sh

cd build/compare
~/mesa/venv/bin/python3 ~/mesa/src/cudavk/tests/cp_gallery.py ref cuda llvmpipe \
    -o three.html --ref-label nvidia --test-label cudavk --test2-label llvmpipe
```

Animated, then the report:

```bash
R=$PWD/build/frames60
T=~/mesa/src/cudavk/tests

$T/cp_perf_run.sh nvidia   ""     $R/nvidia   60
$T/cp_perf_run.sh cudavk   "$CP"  $R/cuda     60
$T/cp_perf_run.sh llvmpipe "$LVP" $R/llvmpipe 60

~/mesa/venv/bin/python3 $T/cp_perf_report.py $R --ref nvidia \
    --drivers nvidia cuda llvmpipe --bench $PWD/build/bench60 \
    -o build/compare/perf60.html
```

Roughly five minutes for the three passes and half a minute for the report. It
writes about 8 GB of frames and 2.7 GB of report images. Benchmark mode is the
same command with `1` as a fifth argument and a larger frame count; it stores
nothing and writes `_bench.csv`.

**Re-rendering a driver means clearing its exported images.** The export skips
any sample that already has frames and the differences are cached in
`_diffs.json`. Delete `<images>/<driver>` and `<images>/_diff/<driver>` and
pass `--recompute`, or the old pictures sit beside the new numbers.

### The reference has to be identified, not assumed

`cp_iterate.sh` reuses `build/iter/nvidia` whenever that directory exists and
does not validate its provenance. Before trusting it, record the Vulkan-Samples
commit and build, the NVIDIA driver and GPU, `FRAMES`, `ORBIT`, the sample list
and a directory checksum. Regenerate it deliberately after a sample, orbit,
driver or frame-count change. A stale 60-frame reference will validate a
different test in silence.

### Reading a verdict

`cp_compare_frames.py` compares every frame of every sample against a reference
and prints a verdict per sample. Its defaults are tolerance 8, `--tolerate 1.5`
and `--floor 2000`.

**The number that matters is judged against what frame 0 already differs by**,
not against zero. Most samples carry a standing difference that says nothing
about animation; `--tolerate` sets how much worse than frame 0 a frame may be,
as a multiple, and `--floor` adds a constant for samples whose frame 0 is near
zero. Exit status is 1 if anything regressed, so it can gate a benchmark run.

For a path that is supposed to be deterministic, run it strictly against the
previous iteration as a local diagnostic — this is change detection, and the
NVIDIA verdict remains the gate:

```bash
$MESA/venv/bin/python3 $T/cp_compare_frames.py ~/git/Vulkan/build/iter \
    --ref previouslabel --test candidatelabel --tol 0 --tolerate 0 --floor 0
```

Also verify both directories hold the **same 60 frame indices**: the comparator
uses the shorter list, and a `missing` row does not make it fail. Reject every
unexplained `missing` row rather than trusting the summary line.

### The two standing exceptions

A clean 60-frame comparison against the stored NVIDIA reference shows exactly
two rows that are not passes, and both are documented rather than accepted
silently (`docs/cudavk/history/PERF16_ITERATIONS.md`, iterations 24, 27 and 28):

| sample | row | why |
|---|---|---|
| `gltfscenerendering` | `REGRESSED` | its own run-to-run nondeterminism, not a driver defect. A standing exception is not permission for it to grow: record its frame number, differing-pixel count, maximum delta and coherent regions against both NVIDIA and the immediate baseline |
| `renderheadless` | `missing` | it never enters the offscreen loop, so its stored directory and the NVIDIA directory are both empty |

**`renderheadless` is a hole in the automated gate, and its row lies about it.**
A bit-identity diff of one empty directory against another reports IDENTICAL.
That is not a pass; it is two absences agreeing, and a change that broke the
sample outright would look exactly the same. When a change can plausibly touch
it, run it by hand in two empty directories and compare what it leaves behind:

```sh
cd $(mktemp -d) && VK_ICD_FILENAMES=$OLD_ICD ~/git/Vulkan/build/bin/renderheadless
cd $(mktemp -d) && VK_ICD_FILENAMES=$NEW_ICD ~/git/Vulkan/build/bin/renderheadless
md5sum */headless.ppm
```

Baseline-versus-candidate identity is a diagnostic; produce a manual NVIDIA (or
llvmpipe) PPM too before claiming external correctness. Iteration 24's manual
check gave zero pixels above tolerance 8 with a maximum channel delta of 1.

### Calibrate against llvmpipe, not against zero

llvmpipe is the backend cudavk's sampler and clipper were ported from, so it is
the honest target. It is mature and still differs from NVIDIA by tens of
thousands of pixels across the set, most of it the same anisotropic filtering
difference cudavk has.

Run it as a third driver in both sweeps. **A residual both software renderers
share is what software rasterisation costs; a residual only cudavk has is a
defect.** At frame 47 of `gltfscenerendering` cudavk differs from NVIDIA by
59,939 pixels and llvmpipe by 91,091, so that residual is the cost of software
rasterisation. `texture3d` peaks at 3,090 against llvmpipe's 0, so that one was
a cudavk defect. `cp_perf_report.py` and the iteration page draw llvmpipe as a
dashed second line on every difference chart for exactly this reason.

### Sweep traps that decide whether a result is real

**Read the exit codes, not just the images.** `_summary.txt` and `_timing.csv`
both carry them, and `cp_perf_run.sh` silently skips a sample whose binary is
absent or not executable. Both `bench/_timing.csv` and `_render/_timing.csv`
must hold all 18 rows at exit 0. Four samples rendered correctly and segfaulted
on the way out for weeks because only the images were being looked at, and one
of those was then misdiagnosed as a 60-frame regression when it reproduced at
one frame in under a second.

**Six samples have a nonzero same-build image floor.** `multithreading`,
`gltfscenerendering` and `instancing` differ run to run because thread
scheduling changes the order command buffers are recorded; `multisampling`,
`vulkanscene` and `particlesystem` do too, for a reason not established, though
only barely — over sixty frames two runs of one build move by 36, 1 and 0
pixels at tolerance 8, with `particlesystem` differing byte-wise on 13 frames
without any pixel crossing the tolerance. **Before attributing a small delta to
a change, run that sample twice against itself.** This calibrates the last few
pixels. It excuses nothing coherent and no missing object.

**`ORBIT` is not cosmetic.** Nine samples render a still scene, so sixty frames
of one is sixty copies of one image; those run with `--offscreenorbit`, which
walks the camera around the subject. `computeshader` matched NVIDIA exactly on
a static frame and diverged by 28,805 pixels within five frames of the camera
moving, because nothing clipped its geometry to the viewport and the sample
draws two of them side by side. Nothing but an orbit could have found that.
The corollary for cost: a hand-run sample without the flag renders a *different
workload*, and an afternoon once went into a 13.7% `gltfscenerendering`
"regression" that was entirely this flag.

**Sample every frame.** A stride of 10 missed `texture3d` entirely, reporting
855 differing pixels where the full pass finds 3,090 — the excursion is
narrower than the sampling interval. `texture3d` reads as 2 differing pixels at
frame 0 and 3,090 at frame 24, and only one of those is visible in a table.
Strided passes are for a quick look, never for a verdict.

**Rebuild a sample after editing it.** Editing a sample's `.cpp` to bisect
something and restoring the source afterwards does not rebuild the binary. An
hour of runs once went into comparing a bilinear render against an anisotropic
reference and produced a confident, wrong conclusion.

**Keep the whole comparison path lossless.** Frames are stored as PNG —
the samples write PPM and `cp_iterate.sh` converts in place once the gate has
read it, bit-exactly, because a browser cannot display PPM and a 1280x720 PPM
is 2.7 MB against 150 KB to 2 MB as PNG. Never route a reference still through
a video or JPEG path. Three mistakes in the difference images are worth not
repeating: `eq=contrast` is not a gain (it pivots around mid grey and drove a
near-zero image *down*, showing a mean of 0.2 where the raw difference was
3.7 — `colorlevels` is the right filter); the pixel format must be pinned to
RGB before blending, or a JPEG encoder negotiates YUV and every difference
image comes out saturated green; and nothing in this path may be lossy,
because the page exists to judge whether a pixel is wrong and JPEG artefacts
sit directly on top of that.

**A green sweep says only that the paths these small samples reach stayed
correct.** Record which sample or capture actually exercises the changed state.
Several real capture bugs were structurally unreachable from the sample set.

---

## 6. Gate 3: sentinel frames from the two captures

Two GFXReconstruct captures of the same offscreen application:

| name | file |
|---|---|
| **old capture** | `~/headless_streamer_20260814T155742.gfxr` |
| **Crossroads** | `~/headless_streamer_1818_20260817T173522.gfxr` |

`docs/cudavk/GFXRECONSTRUCT.md` is how a capture is taken, indexed and turned
into frames, and why it is worth more than the sweep. This section is how it is
used as a gate.

### Why ordinary screenshots do not work

Both captures are offscreen. They never call `vkQueuePresentKHR`, so GFXR
reports zero frames and **`gfxrecon-replay --screenshots` produces nothing** —
it hooks the present call. `GFXRECON_CAPTURE_FRAMES` counts presents too, so
the file cannot be trimmed by frame range either.

What the application does have is a readback: each rendered frame is copied
with `vkCmdCopyImageToBuffer` for its own encoder. `cp_gfxr_frames.py` locates
those copies, asks GFXR to dump the destination buffers with
`--dump-resources`, and converts the raw buffers to lossless PNG. That is the
mechanism, and there is no other one.

### Index each capture once

```bash
MESA=$HOME/mesa
T=$MESA/src/cudavk/tests
GFX=$HOME/gfxreconstruct/build
OLD=$HOME/headless_streamer_20260814T155742.gfxr
CROSSROADS=$HOME/headless_streamer_1818_20260817T173522.gfxr

mkdir -p /tmp/cp-validation/old /tmp/cp-validation/crossroads

$T/cp_gfxr_frames.py index "$OLD" -o /tmp/cp-validation/old/blocks.tsv \
  --gfxrecon-convert $GFX/tools/convert/gfxrecon-convert
$T/cp_gfxr_frames.py frames /tmp/cp-validation/old/blocks.tsv
```

The index depends only on the capture and is reusable across builds. The
dominant readback image is the frame output:

| capture | readbacks | dominant image |
|---|---|---|
| old | 1,510 | 1,509 frames at 1280x720 |
| Crossroads | 1,496 across three images | 1,494 at 1280x720, so `plan --frames all` selects 1,494 |

Treat total readbacks, dominant-image frames and timing-plugin frames as three
separate counts. If a count or a dimension changes unexpectedly, stop and
inspect the index before comparing anything.

### One dump plan, reused by every driver

`--dump-resources` identifies work by GFXR **block index**, not by a friendly
frame number. **Generate one plan and reuse that exact JSON** for llvmpipe, for
the baseline and for the candidate. An early mistake generated a plan per
driver and silently compared frame 500 from one with frame 501 from the other;
the images looked plausible, so it was not obvious.

Once a plan has been used for a baseline, copy `blocks.tsv` and the JSON into
the experiment directory and record a checksum. Never regenerate a missing plan
in the middle of an A/B. `/tmp` is a working directory, not an archive: it
disappears on reboot.

### The sentinel sets

Crossroads — frames 633, 756 and 907 are mandatory, each with its neighbours so
an indexing mistake or temporal instability shows up:

```bash
$T/cp_gfxr_frames.py plan /tmp/cp-validation/crossroads/blocks.tsv \
  --frames 632,633,634,755,756,757,906,907,908 \
  -o /tmp/cp-validation/crossroads/sentinels.json
```

| frame | what it caught |
|---|---|
| 633 | rendered almost entirely black when valid negative clip-space Z was clipped |
| 756 | the same lower-plane problem in a different scene |
| 907 | missing homogeneous X/Y side-plane clipping; a large part of the 3D scene was lost |

Old capture — ten frames sampling loading, early rendering, the middle and the
end:

```bash
$T/cp_gfxr_frames.py plan /tmp/cp-validation/old/blocks.tsv \
  --frames 0,1,2,10,50,200,500,754,1200,-1 \
  -o /tmp/cp-validation/old/sentinels.json
```

The early loading-screen frames are the valuable ones: they have been
deterministic and should stay so.

### References, baseline, candidate

Every `--out` directory must be **new and empty**. `replay` creates a missing
directory and deliberately does not clean an existing one, so stale raw buffers
or a second `_dr.json` manifest can make a failed replay look complete.

```bash
LVP=$MESA/build-lvp-release/src/gallium/targets/lavapipe/lvp_devenv_icd.x86_64.json
CP=$MESA/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json
R=$GFX/tools/replay/gfxrecon-replay

$T/cp_gfxr_frames.py replay "$CROSSROADS" \
  /tmp/cp-validation/crossroads/sentinels.json \
  --icd "$LVP" --out /tmp/cp-validation/crossroads/llvmpipe --gfxrecon-replay $R
$T/cp_gfxr_frames.py png /tmp/cp-validation/crossroads/llvmpipe
```

The helper adds `-m remap`, `--remove-unsupported`, a replay log and
`--dump-resources` itself. Warnings that the replay device differs from the
capture device are expected.

Keep reference directories immutable during an experiment. Beside the images,
save the plan and index with their SHA-256 sums, the capture path and size, the
GFXReconstruct version, the llvmpipe Mesa commit and build type, the ICD path
and the replay log. **A directory named `llvmpipe` without that identity is not
a reproducible reference.** The accepted frozen dumps from iteration 24 are at
`/tmp/perf16/iter24-acceptance/final-frozen/frames/{old,cross}`; copy them
somewhere durable before depending on them.

Render the cudavk baseline and the candidate with the same plan, in the same
sitting, one GPU job at a time, preferably from two worktrees so both binaries
keep existing. Environment variables are inherited by `replay`, so an opt-in
flag is prefixed on the candidate arm only. For a dirty tree, save
`git status --short`, `git diff --binary`, the library's SHA-256 and the
resolved flags from `CUDAVK_HELP=1`; a commit hash alone does not identify that
binary.

### The replay's exit status is not the correctness status

`cp_gfxr_frames.py replay` reports a nonzero replay exit but returns success
itself, because a crashed replay may still have produced useful partial dumps.
For every accepted run:

* read `OUT/replay.log` and the helper's `gfxrecon-replay exited ...` line;
* require normal completion and **exactly** the expected number of dumps;
* search for CUDA launch or compile errors, illegal accesses, unhandled NIR
  intrinsics, allocation or A-buffer overflow and fallback messages, and
  unexpected classic-path fallbacks;
* compare the warnings against the immediate baseline instead of dismissing one
  because an image exists.

Killing the replay once an early dump lands turns minutes into seconds and is
fine for an exploratory probe. **A killed replay is partial evidence, never an
accepted gate.**

### Map dumps back to frames through the manifest

GFXR dump filenames contain resource identifiers, not frame numbers. Their
order in the replay manifest matches the order the plan selected. Pair them
that way — never by sorting the hashed names:

```bash
$MESA/venv/bin/python3 - /tmp/cp-validation/crossroads/cp-candidate \
  632,633,634,755,756,757,906,907,908 <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "mesa/src/cudavk/tests"))
from cp_gfxr_frames import manifest_files

directory = Path(sys.argv[1])
frames = [int(v) for v in sys.argv[2].split(",")]
files = manifest_files(directory)
if len(files) != len(frames):
    raise SystemExit(f"expected {len(frames)} dumps, found {len(files)}")
for frame, name in zip(frames, files):
    print(frame, directory / Path(name).with_suffix(".png"))
PY
```

If the count assertion fires, stop. Inspect the plan, the `_dr.json` manifest
(the documentation calls it `_rd.json`; replay writes `_dr.json`), the replay
exit, the image dimensions and the missing raw buffers. Do not make the counts
agree by dropping a file or pairing the first N sorted names.

### The comparisons, and what "mean RGB" means

```bash
cmp REF.bin TEST.bin                                   # raw, catches alpha and padding
$MESA/venv/bin/python3 $T/cp_compare.py REF.png TEST.png 32
$MESA/venv/bin/python3 $T/cp_compare.py REF.png TEST.png 96
```

`cp_compare.py` compares **RGB only**, even when the PNG is RGBA, so tolerance
0 means exact RGB and not exact buffer contents. It prints differing pixels at
the tolerance, the maximum channel delta, non-background counts for both
images, and a 32x32 difference map. Compare the raw `.bin` as well when alpha,
row padding or readback layout could move; a raw mismatch with an RGB match is
a result to explain, and for an alpha-sensitive change it is a failure until
classified.

Two thresholds, and both are used:

* **above 32/255** in the worst channel measures meaningful shading or tone
  difference;
* **above 96/255** isolates structural error — black regions, missing objects,
  wrong textures, wrong blend order.

**"Mean RGB" in the acceptance records is the mean absolute error over all
three channels of the whole frame, not a per-pixel maximum.** It is a small
number by construction: the iteration-24 envelope for the old capture is a mean
of **0.0206 to 0.4783** across its ten sentinels, with at most 1,815 pixels
over 32 and 45 over 96; Crossroads is **0.8063 to 1.3899**, with maxima of
6,250 and 65 (`docs/cudavk/history/PERF16_ITERATIONS.md`). Those two ranges are
the **envelope**: a candidate whose sentinels sit inside them, on the same
plan, against the same immutable llvmpipe dumps, has passed this gate.

The scale of a real failure, for comparison: the Crossroads clipping fixes took
frames 633 and 756 from mean RGB 59.47 and 17.96 to about 1.5, and frame 907
from 26.55 to about 1.2. A candidate that returns to large black regions is an
obvious failure whatever a global average says. For historical context, the old
capture's post-correction ten-frame standard was a mean 0.441% of pixels above
32/255 and 0.002% above 96/255; treat that as context, not as a substitute for
a fresh comparison.

Record for each sentinel: differing pixels, maximum channel delta,
non-background counts, and whether the difference is isolated edge and
depth-tie pixels or a coherent region. **A large change in non-background
pixels usually means missing geometry, and a large coherent area in the
difference map is never dismissed as floating-point noise.**

### The old capture is not deterministic; Crossroads is

This is the single most useful fact in this document, because it decides what a
frame difference means.

Dumping the ten old-capture sentinels **twice from one binary with one set of
flags** produces six frames that differ. They differ by a **mean absolute delta
of 0.00003 and at most 13/255 on a handful of pixels** — about a thousandth of
the llvmpipe tolerance that already passes
(`docs/cudavk/history/PERF16_ITERATIONS.md`; the same measurement is in
`/tmp/perf16/iter27-vsfusion/handoff.md`). Later acceptance runs saw the same
class with different draws of it: 5 of 10 identical with a worst mean of
0.000067 and a maximum of 7/255 in one pass, and a single pixel at 51/255 on
two frames in another.

**So the correct test on the old capture is the external llvmpipe envelope, not
frame-to-frame equality.** A frame-to-frame difference there is evidence of
nothing until its magnitude is placed against the envelope.

**Crossroads is byte-identical run to run**, which makes it the sensitive
detector: its sentinels are compared directly against the frozen reference and
have been 9 of 9 byte-identical through iterations 24, 27 and 28. A single
changed byte on Crossroads is a real event.

Run the self-comparison at tolerance 0 when it is used as a floor. A rounded
`0.000%` at a nonzero threshold is not proof of byte identity.

### Visual inspection is still required

Numbers first, then look. At minimum: Crossroads 633 (the 3D scene must not be
predominantly black), 756 (the full 3D region must render), 907 (ground and
building geometry crossing the right frustum side must stay present), the old
capture's loading frames (must match exactly), and any alpha-heavy or particle
frame for blend-order changes.

`$T/cp_gfxr_timeline.sh "$CROSSROADS" /tmp/timeline "$CP"` builds a clickable
all-frame timeline — one timing replay, one image replay, lossless PNGs and an
`index.html`. It is for browsing; it compares nothing.

### Full-frame validation when the risk is high

For clipping, memory residency, resource copies, synchronisation, rasterisation
or pass-episode changes, sentinels are necessary and not sufficient. Dump every
frame with one shared `plan --frames all`, generate the external reference from
the same plan, compare corresponding manifest entries, summarise per-frame
differing pixels and maximum delta, and inspect the worst frames.

**Dumping perturbs replay timing. A dump-enabled replay is never the
performance measurement.**

---

## 7. Gate 4: device-side equivalence, with a negative control

When a change alters what the GPU computes — a fused kernel replacing a chain,
a counter moved into another launch — image comparison is a weak instrument:
it samples ten frames out of fifteen hundred and it hides small disagreements
under a tolerance. The stronger gate runs **both forms over a whole replay and
compares on the device**, reporting counts.

The pattern, from the A-buffer fusion (iteration 28 item 4):

* `CUDAVK_ABUF_FUSE_CHECK=1` runs the classic chain first into shadow buffers,
  with its clamp disabled so it cannot disturb what the fused chain then reads,
  runs the fused chain into the live buffers, and compares every offset, the
  grand total, every `blk_counts` entry, and the coverage invariant the fused
  skip rests on.
* The result is reported as counts over the whole replay, not as a spot check:
  **old capture 23,814 scans and 23,814 quad builds compared, 0 differing
  elements, 0 differing totals, 0 entries never written, 0 coverage violations;
  Crossroads 13,310 and 13,310, all four counters 0.**
* `CUDAVK_ABUF_FUSE_BREAK` is the **negative control** and it has teeth: `=1`
  drops the block base and the gate reports 60,208,448 differing elements;
  `=2` stops the fused count writing an uncovered zero — what a missing clear
  looks like — and reports 1,691,296 differing, 1,691,296 never written and
  1,691,296 coverage violations. With the check off, break 1 also destroys the
  output: four sentinel frames with a mean channel error of 60.5 and 440,741
  pixels over 32, against an envelope whose mean is under 0.5.

Both the check and the break flag belong in the registry
(`src/cudavk/cp_debug.c`) like everything else, so `CUDAVK_HELP=1` and
`FLAGS.md` describe them. Run `cp_debug_doc.py --check` after touching it, and
`cp_no_getenv.py` if you were tempted to read the variable directly instead.

### The other negative control: proving a patch is INERT

A revert flag is only a control if the flag-off build really is the old driver
*and* the flag-on build really contains the feature. The second half is the one
that gets skipped, and skipping it produced a **vacuous proof** on 2026-08-27:

> A stray `git checkout` deleted a probe's implementation. The commit carried
> **the flags and no code.** The inertness check did not catch it — it
> *confirmed* it, reporting `0 instruction lines differ, 520 of 520 sections
> byte-identical`. A perfect result, produced by the bug it was meant to detect.

**A folded-off build and a build with the feature MISSING are
indistinguishable unless something distinguishes them.** This is `WORKFLOW.md`
§4.4 ("a different hash is usually a run that died") applied to a *build* rather
than to a run.

**Four different claims, routinely confused.** Say which one you are making:

| claim | what it compares | what it proves |
|---|---|---|
| **collateral damage** | files you never edited, base against shipped | no struct offset moved, nothing unrelated shifted |
| **containment** | only the intended functions changed | the edit is where you say it is |
| **folding** | base against flags-off | the off arm costs nothing |
| **positive control** | base against the feature **live** | **the feature exists** |

The first three are worthless without the fourth. Report all four as one line:
*"0 collateral, +4,361 present, +1,076 removable by folding."*

**Match the control arm to the KIND of gate.** The general form is: *fold the
gate to the value that should DELETE the code, and require the delete to show
up.*

- **A compile-time macro** — the **live** arm must differ. Worked example: PTX
  656,339 → 756,883, cubin `.text` 357,888 → 412,544, 40 → 43 kernels, and
  folded-off is **0 bytes** from base.
- **A runtime bool** — the informative arm is the opposite one, with the gates
  **hardcoded false**. If the implementation had been deleted, flags-off would
  *equal* folded. Worked example: base 28,472 / off 32,833 / on 32,507 / folded
  31,757.
- **A flag registry row** — **`.text` is the wrong section.** A new row is a
  *table* entry, so the `.text` delta of `cp_debug.c.o` is 0 by construction,
  which has exactly the shape of the vacuous result. Its evidence is
  `.data.rel.ro.local.flags` (+88) and `.bss.present_in_env` (+1). **Control the
  flag in `.data` and the implementation in `.text`.**

**The strongest control is not a size — it is a relocation.** For a
runtime-gated patch, disassemble the flags-off entry point and read its
relocations and offsets. The shipped `cpvk_DestroyImage` can reach
`cpvk_device_drain`, `cpvk_drain_census_call` and
`cpvk_texture_cache_image_retire`; the base object has **zero** matching
symbols. In `cp_pass_finish`, `cmpb $0x0,0x11c` is the flag test and
`0x238/0x240/0x248` are the merged kernel handles being launched; the base has
zero references to `0x11c`. **A deleted implementation cannot produce a
relocation.**

**Two implementation rules, each measured twice in one session:**

1. **A field added for measurement goes at the END of its struct.** Mid-struct
   placement cost **1,601** changed displacement lines in one object and **915**
   in another — two agents, two structs, the same lesson. **Trap:** `struct
   cp_kernels` is *embedded* in `struct cp_device`, so appending to the
   innermost struct still shifts everything after it in the outer one.
   End-of-struct is necessary, not sufficient: what it buys is that the residue
   becomes **classifiable** — 93 displacements at exactly +0x18, 8 `__LINE__`
   immediates, 1 label renumber, 0 unexplained — rather than zero.
2. **Gate even the parts too cheap to gate.** One unguarded counter increment
   costs nothing to run and **363 bytes to prove**, because it survives folding
   and then has to be explained.

**Rebuilding in place invalidates runs.** Producing these control arms needed
three in-place rebuilds of a build directory that a measurer was using. **Any
run started inside such a window is discarded and repeated.** Build variants in
a separate directory, or announce the window before you start.

---

## 8. A real regression against the noise

Every gate has a floor. Anything under it is a question, not a finding.

| measurement | floor, measured |
|---|---|
| old-capture sentinel frames, same binary | 6 of 10 frames differ; mean absolute 0.00003, at most 13/255 |
| Crossroads sentinel frames, same binary | byte-identical |
| the six nondeterministic samples, 60 frames, same build | 36, 1 and 0 pixels at tolerance 8; `particlesystem` differs byte-wise on 13 frames with nothing over tolerance |
| `gltfscenerendering` cost, three runs of one build | 13.87, 14.83, 13.88 ms — a **6.9%** spread, wider than the ±5% the gate flags at |
| the 600-frame sweep, two runs of one build | 0.6-1.2% during the phase 1a pass |
| capture replay wall time, paired runs | four consecutive pairs agreed to 0.07%, then a later pair disagreed by 3.9% |
| the driver at iteration 28, old capture | median of run medians **15.7514 ms**, IQR [15.7305, 15.7596], range [15.7200, 15.7911] |
| the driver at iteration 28, Crossroads | median of run medians **5.8982 ms**, IQR [5.8877, 5.8990] |
| **today's shipping default**, old / Crossroads | **12.7826** IQR [12.7632, 12.8075] / **5.6936** IQR [5.6868, 5.7007] |

The first two are from `docs/cudavk/history/PERF16_ITERATIONS.md`, "The standing
default, stated as a distribution", six runs on old and four on Crossroads; the
third is from `history/perf-2026-08-27/pdl_landing.md` on the tip that landed
PDL, same run counts. **The level is not what this table is for — the SPREAD
is**, and the spread is what survives a re-baselining: an IQR of about 0.03 ms
on old at both 15.75 and 12.78, roughly 0.2% of the frame, which is why a 1%
move needs paired arms and not two means.

So:

* **A >5% move on a >3 ms sample is a question.** Answer it by building both
  trees and alternating arms in one session, then reporting the per-pair deltas
  and their median rather than the two means. Within an arm the spread is about
  0.1 ms, so pairing resolves a 1% move the sweep cannot see. One reported
  +7.1% on `gltfscenerendering` vanished under a controlled comparison; a later
  +7.1% was real and reproduced at +0.99 ms across five interleaved pairs. The
  sweep alone cannot tell those apart.
* **Anything under about 10% on a capture replay needs a paired A/B too.** A
  change worth under 3% may not be resolvable in wall time at all and is better
  argued from a direct count.
* **A structural difference is never noise, whatever its mean.** Missing
  objects, black regions, a coherent region in the difference map, or a large
  move in non-background pixel counts.

Two more rules that stop a probe from measuring the wrong thing:

* **A probe that changes what the compiler can prove is not measuring the thing
  it names.** A probe that hardcoded the rasteriser's bounding box wrote over
  the four values it had just loaded, so NVRTC removed the loads as dead: the
  run measured *no bounding box read at all* while claiming to measure a small
  one, and reported a 17% win that did not exist. Consume the value, or hide
  the use behind something the compiler cannot fold — and when a probe result
  is surprisingly good, read the generated code before believing it.
* **Any probe whose stdout hash changes is invalid until the submit and frame
  counts are checked.** A run that dies early produces a fast, meaningless
  median: an iteration-29 probe first read as +8.07 ms on the old capture and
  was a device loss after 26 frames — 52 submits against 3,022
  (`/tmp/perf16/iter29-sync/`, and `WORKFLOW.md` §4.4).

---

## 9. When a gate fails

Triage the first divergence. Do not relax a threshold.

1. Reproduce with the immediate baseline and the candidate, the same plan, and
   no concurrent GPU job. Confirm both replay exits and both logs.
2. For an opt-in change, run the candidate with the flag **unset**. If that
   differs from the baseline, debug default-path perturbation before entering
   the new path at all.
3. Find the earliest divergent frame, then narrow to the first draw or resource
   operation. GFXR's JSON `DumpResources` mode can dump around a command and
   honours `DumpBeforeCommand`; do not infer the first bad draw from the final
   frame.
4. Read the **first** CUDA or NVRTC error. Context faults are sticky, so later
   launch, copy and teardown failures are usually consequences.
5. Disable one subsystem at a time and save `CUDAVK_HELP=1` for each run. A
   bisect result without the resolved flags is not reproducible.
6. For a suspected out-of-bounds, use-after-free, uninitialised-memory or
   synchronisation fault, reduce to the smallest test or replay and run CUDA's
   own diagnostics:

   ```bash
   VK_DRIVER_FILES=$CP compute-sanitizer --tool memcheck --error-exitcode 99 \
     build-cudavk/src/cudavk/cpvk_smoke
   ```

   Follow with `initcheck` or `synccheck` when the first report points there.
   Sanitizer runs are diagnostics, never performance runs.

Preserve the failing images, the difference map, the plan, the replay log and
the exact binary before trying a fix.

---

## 10. Acceptance checklist

Before keeping a correctness-sensitive change:

* Build clean; `cp_debug_doc.py --check` and `cp_no_getenv.py --check` pass;
  one test smoke-run proves NVRTC still compiles the kernels.
* Baseline and candidate binaries identified: commit, dirty diff, build type,
  library SHA-256, resolved flags.
* **Native suite 65/65** in the default state and with every new revert flag
  set, run serially.
* The candidate with each new opt-in flag **unset** matches the immediate
  baseline. This is a diagnostic, not the external verdict — and it is worthless
  on its own: pair it with a **positive control** showing the feature is present
  in the shipped build, or a deleted implementation will pass it (§7, "The other
  negative control").
* Sample sweep: both `_timing.csv` files hold all 18 rows at exit 0; reference
  and candidate hold the same 60 frame indices; the verdict table actually ran
  (`correctness.gate_ran`); only the two standing exceptions appear; no
  standing deviation has grown; the same-build floor was measured for any small
  movement in a nondeterministic sample; `instancing` and `multithreading`
  visibly retain all objects; `renderheadless` was checked by hand if the
  change could touch it.
* The stored NVIDIA reference's provenance is known and still matches.
* GFXR: every run started in a new empty directory; both captures completed
  normally and produced exactly the planned dumps; logs carry no new CUDA,
  compiler, fallback or overflow message; one checksummed plan was shared by
  the llvmpipe reference, the baseline and the candidate; files were paired
  through the manifest.
* Crossroads 633, 756 and 907 extracted and compared; old-capture loading
  frames compared and still exact; sentinels inside the documented llvmpipe
  envelope; structural-threshold populations not grown; the worst frames
  inspected by eye.
* A same-build replay established the nondeterminism floor if the differences
  are small.
* High-risk clipping, memory, copy, synchronisation, rasterisation or episode
  changes got full-frame validation against an external reference, not
  sentinels alone.
* Where the change alters device arithmetic, an equivalence gate ran over a
  whole replay **and** its negative control was shown to fail.
* Timing was measured separately, without dumping or a profiler, and only after
  correctness passed.
* Commands, identities, checksummed plans, logs, lossless images and
  conclusions are in a durable directory and summarised in the relevant
  document.

---

## 11. Related documents

| document | what it holds |
|---|---|
| `docs/cudavk/WORKFLOW.md` | build, run, the two captures, timing conventions, `cp_iterate.sh`, and the profiling decision tree ("where the time goes") |
| `docs/cudavk/GFXRECONSTRUCT.md` | capturing, indexing and replaying a real application, and `cp_gfxr_frames.py` in full |
| `src/cudavk/FLAGS.md` | all 116 switches, generated from the registry |
| `docs/cudavk/history/PERF16_ITERATIONS.md` | the iteration record: every acceptance battery quoted here, with its numbers |
| `docs/cudavk/history/PHASE_1A.md`, `docs/cudavk/history/EPISODES.md` | the passes these rules were learned in |
| `docs/cudavk/GALLIUM_RETIREMENT.md` | what the removed second frontend was, and why it is no longer available as a cross-check |

Diagnosis of *why* a frame is slow is a different question from whether it is
correct, and it has its own order of tools. Ask it in `WORKFLOW.md` §8, and
note the two rules there that apply to any trace taken since the driver went
multi-stream: sum device time as a **union of intervals**, and credit a removed
wait only if the **sum over every synchronisation site** drops.
