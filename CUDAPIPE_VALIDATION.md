# Cudapipe Correctness Validation

## Purpose

This document describes the correctness workflow used while changing
Cudapipe. It is intended to be repeatable by a new session without relying on
old terminal history.

Correctness has three separate gates:

1. A complete 18-sample Vulkan render sweep catches common API and rendering
   regressions and is compared with the stored NVIDIA output.
2. Selected frames from both HeadlessStreamer GFXReconstruct captures catch
   real-application regressions that the samples do not exercise and are
   compared with release llvmpipe or NVIDIA output.
3. Performance replays are run separately. A replay completing every frame is
   not proof that those frames rendered correctly.

**An implementation is never its own correctness oracle.** Cudapipe baseline
and same-binary comparisons are valuable change-detection and nondeterminism
diagnostics, but acceptance requires llvmpipe or NVIDIA images generated from
the same plan. This applies equally to the Gallium-hosted and native ICDs.

Do not accept a performance change when any correctness gate shows new missing
geometry, black regions, broken blending, or other structural differences.

## Known Captures and Drivers

```bash
MESA=$HOME/mesa
TESTS=$MESA/src/gallium/drivers/cudapipe/tests
GFX=$HOME/gfxreconstruct/build

OLD=$HOME/headless_streamer_20260814T155742.gfxr
CROSSROADS=$HOME/headless_streamer_1818_20260817T173522.gfxr

CP_ICD=$MESA/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json
CP_NATIVE_ICD=$MESA/build-cudapipe/src/cudapipe/cudapipe_native_devenv_icd.x86_64.json
LVP_ICD=$MESA/build-lvp-release/src/gallium/targets/lavapipe/lvp_devenv_icd.x86_64.json
NVIDIA_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
```

Use the release llvmpipe build as the primary GFXR image reference. It shares
the lavapipe frontend with Cudapipe, while NVIDIA can legitimately differ in
floating-point details. The debug llvmpipe build is unsuitable for the old
capture because it asserts in anisotropic sampling.

That shared frontend is both the reason to use llvmpipe and a limitation. A
change outside `src/gallium/drivers/cudapipe` can affect lavapipe and Cudapipe
together and make their agreement circular. After a lavapipe, NIR, Gallium, or
common Vulkan change, regenerate the llvmpipe reference and use NVIDIA as the
independent check. Record the llvmpipe Mesa commit and build type with every
reference set.

NVIDIA remains useful as a second independent reference and is the stored
reference for the Vulkan sample sweep. Commands below use `CP_ICD`; substitute
`CP_NATIVE_ICD` when validating the native driver. Never compare native output
only with Gallium-hosted Cudapipe: both paths share the CUDA renderer and can
agree on the same defect.

## Use the Repository Interpreter for Python Image Tools

```bash
PY=$MESA/venv/bin/python3
```

Use `$PY` for Python image comparison commands, not whatever an ambient
`python3` resolves to. `cp_compare_frames.py` requires numpy and only the
repository venv has it; `cp_compare.py` decodes through Pillow when it is
importable and falls back to its own PNG reader when it is not, about a hundred
times slower per frame. The directly executed helpers use an env-based
`python3` shebang; keep the repository venv first in `PATH` as shown below.

This is not a style preference. `cp_iterate.sh` once invoked the comparison
with the ambient `python3`, which printed an import error where the verdict
table goes and a cost delta underneath as usual. Both "a sample regressed" and
"the check never ran" exit 1, so the status could not tell them apart. See
`src/gallium/drivers/cudapipe/PHASE_1A.md`.

## Build Before Validation

```bash
export PATH="$MESA/venv/bin:$HOME/vulkan-sdk/1.4.357.1/x86_64/bin:$PATH"
ninja -C $MESA/build-cudapipe
$PY $TESTS/cp_debug_doc.py --check
```

`build.ninja` records the absolute path of the meson that generated it, so a
missing venv fails before anything compiles; recover with
`$MESA/venv/bin/meson setup --reconfigure $MESA/build-cudapipe`.

**A successful build says nothing about the CUDA kernels.** The `.cu` files are
stringified at build time by `cu_to_inc.py` and compiled by NVRTC at screen
creation, so a kernel that does not compile builds clean and fails at first
use, as a driver error rather than a build error. Smoke-run one sample before
starting a validation pass.

**A flag-gated change is not exempt from the gates.** Adding a kernel to a
`.cu` file changes the translation unit that every other kernel in that module
is compiled in, so inlining and floating-point contraction can move in code
that was not edited. That is precisely the failure behind the watertight
coverage work in `CUDAPIPE_HANDOFF.md`: contraction broke the edge function's
antisymmetry and cracked a shared edge for its whole length. "Default off" is
an argument about the new path, not about the old one.

## Flags: Test Both Arms

For a performance flag or experimental path, pass the same environment flag
to both the timing and image replays. For example:

```bash
export CUDAPIPE_TILE_BOUND=1
```

Unset it before producing the flag-off control:

```bash
unset CUDAPIPE_TILE_BOUND
```

Do not use `FLAG=0` as a generic way to turn an experiment off. Cudapipe has
both presence booleans, where `=0` is still **on**, and value booleans. Check
`CUDAPIPE_HELP=1` or `FLAGS.md` and use `unset` for the control.

Three runs are needed, not two:

| run | flag | must equal |
|---|---|---|
| baseline | unset | the reference |
| candidate, flag **off** | unset | the baseline, exactly |
| candidate, flag **on** | set | judged on its own merits |

The middle row is the one that gets skipped and the one that catches the most.
It separates "the new path is wrong" from "the new code perturbed the default
path" — which happens for reasons that have nothing to do with rendering:
instrumentation that allocates from the upload arena shifts every later
offset in it, and a new kernel changes the NVRTC translation unit as above.

## Why Ordinary Screenshots Do Not Work

Both captures are offscreen. They never call `vkQueuePresentKHR`, so GFXR
reports zero frames and `gfxrecon-replay --screenshots` produces nothing.

Each rendered frame is eventually copied with `vkCmdCopyImageToBuffer` for the
application's own readback. `cp_gfxr_frames.py` locates those commands and asks
GFXR to dump the destination buffers. The raw buffers are then converted to
lossless PNG files.

## Critical Rule: Reuse One Dump Plan

`--dump-resources` identifies work by GFXR block index, not by a friendly frame
number. Generate one plan from one capture index and reuse that exact JSON for
llvmpipe, Cudapipe baseline, and Cudapipe test runs.

Never independently generate a plan for each driver. An earlier mistake did
that and silently compared frame 500 from one driver with frame 501 from the
other. The images looked plausible, so the error was not obvious.

## Index Each Capture Once

The index depends only on the capture and is reusable across driver builds.
The `/tmp` paths below are convenient working directories, not an archive:
they disappear on reboot. Once a plan has been used for a baseline, copy its
`blocks.tsv` and JSON plan into the experiment directory and record a checksum.
Do not silently regenerate a missing plan in the middle of an A/B comparison.

```bash
mkdir -p /tmp/cp-validation/old /tmp/cp-validation/crossroads

$TESTS/cp_gfxr_frames.py index "$OLD" \
  -o /tmp/cp-validation/old/blocks.tsv \
  --gfxrecon-convert $GFX/tools/convert/gfxrecon-convert

$TESTS/cp_gfxr_frames.py index "$CROSSROADS" \
  -o /tmp/cp-validation/crossroads/blocks.tsv \
  --gfxrecon-convert $GFX/tools/convert/gfxrecon-convert

$TESTS/cp_gfxr_frames.py frames /tmp/cp-validation/old/blocks.tsv
$TESTS/cp_gfxr_frames.py frames /tmp/cp-validation/crossroads/blocks.tsv
```

The dominant readback image should be the main frame output. The old capture
has 1,510 readbacks in total, with 1,509 from its dominant 1280x720 image.
Crossroads has 1,496 readbacks across three images, with 1,494 from the dominant
1280x720 image; `plan --frames all` therefore selects 1,494 dumps, not 1,496.
Treat total readbacks, dominant-image frames, and timing-plugin frames as
separate counts. If a count or dominant image dimension unexpectedly changes,
stop and inspect the index and generated plan before comparing.

## Sentinel Frame Plans

### Crossroads

Frames 633, 756, and 907 are mandatory:

- Frame 633 previously rendered almost entirely black when valid negative
  clip-space Z was incorrectly clipped.
- Frame 756 exposed the same lower-plane problem in a different scene.
- Frame 907 exposed missing homogeneous X/Y side-plane clipping and lost a
  large part of the 3D scene.

Include adjacent frames to detect an indexing mistake or temporal instability:

```bash
$TESTS/cp_gfxr_frames.py plan \
  /tmp/cp-validation/crossroads/blocks.tsv \
  --frames 632,633,634,755,756,757,906,907,908 \
  -o /tmp/cp-validation/crossroads/sentinels.json
```

### Old Capture

The established old-capture probe set samples loading, early rendering, the
middle, and the end:

```bash
$TESTS/cp_gfxr_frames.py plan /tmp/cp-validation/old/blocks.tsv \
  --frames 0,1,2,10,50,200,500,754,1200,-1 \
  -o /tmp/cp-validation/old/sentinels.json
```

The early loading-screen frames are especially valuable. They have previously
been deterministic and should stay bit-exact.

## Generate References

Every `--out` directory below must be new and empty. `cp_gfxr_frames.py replay`
creates a missing directory but deliberately does not clean an existing one;
stale raw buffers or a second `_dr.json` manifest can make a failed replay look
complete or make the mapping read the wrong manifest. Use a fresh experiment
root or move the old directory aside — never replay candidate output on top of
baseline or reference output.

Generate references once from release llvmpipe using the shared plan:

```bash
$TESTS/cp_gfxr_frames.py replay "$CROSSROADS" \
  /tmp/cp-validation/crossroads/sentinels.json \
  --icd "$LVP_ICD" --out /tmp/cp-validation/crossroads/llvmpipe \
  --gfxrecon-replay $GFX/tools/replay/gfxrecon-replay

$TESTS/cp_gfxr_frames.py png /tmp/cp-validation/crossroads/llvmpipe

$TESTS/cp_gfxr_frames.py replay "$OLD" \
  /tmp/cp-validation/old/sentinels.json \
  --icd "$LVP_ICD" --out /tmp/cp-validation/old/llvmpipe \
  --gfxrecon-replay $GFX/tools/replay/gfxrecon-replay

$TESTS/cp_gfxr_frames.py png /tmp/cp-validation/old/llvmpipe
```

The helper automatically uses `-m remap`, `--remove-unsupported`, a replay log,
and `--dump-resources`. Warnings that the replay device differs from the capture
device are expected.

Keep reference directories immutable during an experiment. If references must
be regenerated, regenerate all compared drivers from the same plan. Beside the
images, save the plan and index, their SHA-256 sums, the capture path and size,
the GFXReconstruct version, the llvmpipe Mesa commit/build type, the ICD path,
and the replay log. A directory named `llvmpipe` without that identity is not a
reproducible reference.

## Render the Cudapipe Baseline and Candidate

Create a baseline immediately before the change or from the known-good commit,
then render the candidate with the same plan. This pair detects what the change
moved; it does not decide whether either image is correct. The llvmpipe or
NVIDIA replay generated from that same plan remains the acceptance reference.
Prefer separate git worktrees and build directories so both binaries continue
to exist and can be alternated. For a dirty tree, save `git status --short`,
`git diff --binary`, the built library's SHA-256, and all resolved Cudapipe
flags (`CUDAPIPE_HELP=1`); a commit hash alone does not identify that binary.

Run the pair in the same sitting, one GPU job at a time. Do not reuse a baseline
from a differently versioned GFXReconstruct replay or a machine state that is
not recorded.

```bash
$TESTS/cp_gfxr_frames.py replay "$CROSSROADS" \
  /tmp/cp-validation/crossroads/sentinels.json \
  --icd "$CP_ICD" --out /tmp/cp-validation/crossroads/cp-baseline \
  --gfxrecon-replay $GFX/tools/replay/gfxrecon-replay
$TESTS/cp_gfxr_frames.py png /tmp/cp-validation/crossroads/cp-baseline

# Rebuild after applying the candidate change, then:
$TESTS/cp_gfxr_frames.py replay "$CROSSROADS" \
  /tmp/cp-validation/crossroads/sentinels.json \
  --icd "$CP_ICD" --out /tmp/cp-validation/crossroads/cp-candidate \
  --gfxrecon-replay $GFX/tools/replay/gfxrecon-replay
$TESTS/cp_gfxr_frames.py png /tmp/cp-validation/crossroads/cp-candidate
```

Repeat the same commands for the old capture and its sentinel plan.

Environment variables are inherited by `cp_gfxr_frames.py replay`. Therefore,
when testing an opt-in path, prefix or export the flag only for the candidate:

```bash
CUDAPIPE_TILE_BOUND=1 \
  $TESTS/cp_gfxr_frames.py replay "$CROSSROADS" \
  /tmp/cp-validation/crossroads/sentinels.json \
  --icd "$CP_ICD" --out /tmp/cp-validation/crossroads/cp-tile-bound \
  --gfxrecon-replay $GFX/tools/replay/gfxrecon-replay
$TESTS/cp_gfxr_frames.py png /tmp/cp-validation/crossroads/cp-tile-bound
```

Also render the candidate binary with the flag unset and compare it with
`cp-baseline`. Do not infer flag-off correctness from a flag-on result.

## Replay Completion and Logs

`cp_gfxr_frames.py replay` deliberately reports a nonzero replay exit but
returns success itself, because a crashed replay may still have produced useful
partial dumps. Therefore the helper's shell status is **not** the correctness
status. For every accepted run:

- inspect `OUT/replay.log` and the helper's `gfxrecon-replay exited ...` line;
- require normal replay completion and exactly the expected number of dumps;
- search for CUDA launch/compile errors, illegal accesses, unhandled NIR
  intrinsics, allocation or A-buffer overflow/fallback messages, and unexpected
  classic-path fallbacks;
- compare warnings with the immediate baseline instead of dismissing a warning
  merely because an image exists.

For an exploratory early-frame probe only, it is reasonable to terminate the
replay after the requested dump lands; this can turn minutes into seconds. A
killed replay is partial evidence, never an accepted gate. Re-run to normal
completion for the recorded result.

## Map Dump Files Back to Requested Frames

GFXR dump filenames contain resource identifiers, not frame numbers. Their
order in the replay manifest matches the order selected by the plan. Use this
snippet to print the mapping:

```bash
$MESA/venv/bin/python3 - \
  /tmp/cp-validation/crossroads/cp-candidate \
  632,633,634,755,756,757,906,907,908 <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "mesa/src/gallium/drivers/cudapipe/tests"))
from cp_gfxr_frames import manifest_files

directory = Path(sys.argv[1])
frames = [int(value) for value in sys.argv[2].split(",")]
files = manifest_files(directory)
if len(files) != len(frames):
    raise SystemExit(f"expected {len(frames)} dumps, found {len(files)}")
for frame, name in zip(frames, files):
    print(frame, directory / Path(name).with_suffix(".png"))
PY
```

Do not compare files merely by lexicographically sorting their hashed names.
If the count assertion fires, stop: inspect the plan, `_dr.json`/`_rd.json`
manifest, replay exit, image dimensions, and missing raw buffers. Do not make
the counts agree by dropping a file or pairing the first `N` sorted names.

## Pixel Comparisons

There are three useful comparisons.

### Candidate Versus Immediate Cudapipe Baseline

This is the strict change-detection diagnostic, not a correctness oracle. For
deterministic paths, require exact output; any movement must be explained and
then judged against llvmpipe or NVIDIA. A zero delta only proves that this
change preserved the baseline, including any defect they share.

`cp_compare.py` intentionally compares RGB only, even when the PNG is RGBA, so
tolerance 0 means exact **RGB**, not exact buffer contents. Compare the raw
dumps as well when alpha, row padding, or readback layout can be affected:

```bash
cmp REF.bin TEST.bin
$PY $TESTS/cp_compare.py REF.png TEST.png 0
```

A raw mismatch with an RGB match is still a result to explain; for an
alpha-sensitive change it is a correctness failure until classified.

For each sentinel, record:

- Differing pixel count.
- Maximum RGB-channel delta.
- Non-background pixel counts.
- Whether differences are isolated edge/depth ties or coherent regions.

A large change in non-background pixels usually means missing geometry. A
large coherent area in the difference map is never dismissed as floating-point
noise.

### Candidate Versus Release llvmpipe

This is the primary external GFXR correctness gate. NVIDIA is the independent
fallback/check when llvmpipe cannot run the path. It finds standing Cudapipe
errors and decides whether an unchanged native/self result is actually
acceptable:

```bash
$PY $TESTS/cp_compare.py LLVMPipe.png Cudapipe.png 32
$PY $TESTS/cp_compare.py LLVMPipe.png Cudapipe.png 96
```

Use both thresholds:

- More than 32/255 in the worst RGB channel measures meaningful shading or
  tone differences.
- More than 96/255 emphasizes structural errors such as black regions,
  missing objects, wrong textures, or incorrect blend order.

For the old capture's established ten-frame probe set, the documented
post-correction standard was a mean 0.441% of pixels above 32/255 and 0.002%
above 96/255. Treat those values as historical context, not a substitute for
an immediate baseline comparison.

Earlier Crossroads clipping fixes reduced mean RGB error for frames 633 and
756 from 59.47 and 17.96 to approximately 1.5, and frame 907 from 26.55 to
approximately 1.2. A candidate returning to large black regions is an obvious
failure regardless of a global average.

### Cudapipe Versus Itself

Replay the same binary twice with the same plan and compare outputs. This
measures the nondeterminism floor only; it cannot establish correctness. The old capture measured bit-exact after the
submission-order fixes, but that is a measured property of that build and
capture, not an assumption for the next one. Run the control at tolerance 0;
a rounded `0.000%` at a nonzero threshold is not proof of byte identity. If
self-comparison moves, record the differing-pixel count and maximum delta before
interpreting a candidate difference of the same scale.

## Visual Inspection

Numeric comparison is required, but inspect the PNGs as well. Keep the whole
comparison path lossless: raw dumps, PNG, or the samples' PPM output only. Do
not extract reference stills through a video/JPEG path.

Inspect at least these images:

- Crossroads 633: the 3D scene must not be predominantly black.
- Crossroads 756: the full 3D region must render.
- Crossroads 907: ground and building geometry crossing the right frustum side
  must remain present.
- Old loading frames: UI/loading imagery must match exactly.
- Check alpha-heavy and particle frames for blend-order changes.

The clickable all-frame timeline can be produced independently:

```bash
$TESTS/cp_gfxr_timeline.sh "$CROSSROADS" \
  /tmp/cp-validation/crossroads/timeline-cudapipe "$CP_ICD"
```

This runs one timing replay and one image replay, creates lossless PNGs, and
writes `index.html` referencing those PNGs. It is useful for browsing but does
not automatically compare against a reference.

## Full-Frame Validation When Risk Is High

For clipping, memory residency, resource copies, synchronization, rasterization,
or episode changes, sentinel frames are necessary but not sufficient. Dump all
frames using one shared plan:

```bash
$TESTS/cp_gfxr_frames.py plan /tmp/cp-validation/crossroads/blocks.tsv \
  --frames all -o /tmp/cp-validation/crossroads/all.json

$TESTS/cp_gfxr_frames.py replay "$CROSSROADS" \
  /tmp/cp-validation/crossroads/all.json \
  --icd "$CP_ICD" --out /tmp/cp-validation/crossroads/all-candidate \
  --gfxrecon-replay $GFX/tools/replay/gfxrecon-replay

$TESTS/cp_gfxr_frames.py png /tmp/cp-validation/crossroads/all-candidate
```

Generate the external reference with the same `all.json` using release
llvmpipe or NVIDIA. A complete native dump beside a Gallium-hosted Cudapipe dump
proves execution coverage and shared-backend stability only; it is not a
full-frame correctness pass. Compare corresponding manifest entries, not
filenames. Summarize per-frame differing pixels and maximum delta, then inspect
the worst frames.

Dump and PNG conversion are correctness jobs and perturb replay timing; never use a
dump-enabled replay as the performance measurement.

## Vulkan Sample Sweep

GFXR validation does not replace the 18-sample sweep. Use the iteration helper,
which times 600 frames by default, stores all 60 frames of the correctness
orbit, compares those frames with the stored NVIDIA reference, and writes an
iteration record. Sixty frames is enough for the image orbit but too short for
a trustworthy timing window.

```bash
cd $MESA
FRAMES=60 BENCH_FRAMES=600 \
DESC="describe the exact candidate and expected effect" \
  $TESTS/cp_iterate.sh candidate-label previous-label
```

For an opt-in path:

```bash
CUDAPIPE_TILE_BOUND=1 FRAMES=60 BENCH_FRAMES=600 \
DESC="tile-bound path correctness and performance" \
  $TESTS/cp_iterate.sh tile-bound-candidate previous-label
```

`cp_iterate.sh` reuses `build/iter/nvidia` whenever that directory exists; it
does not validate its contents or provenance. Before trusting it, record the
Vulkan-Samples commit/build, NVIDIA driver and GPU, `FRAMES`, `ORBIT`, sample
list, and directory checksum. Regenerate the reference intentionally after a
sample, orbit, driver, or frame-count change. Do not let a stale 60-frame
reference silently validate a different test.

`previous-label` makes `cp_iterate.sh` print a **timing** delta; it does not
image-compare the two iterations. Run that comparison explicitly as the local
regression diagnostic, while retaining the NVIDIA verdict as the correctness
gate. For paths that should be deterministic, remove the comparator's normal
frame-0 allowance:

```bash
$PY $TESTS/cp_compare_frames.py ~/git/Vulkan/build/iter \
  --ref previous-label --test candidate-label \
  --tol 0 --tolerate 0 --floor 0
```

If a known nondeterministic sample moves, calibrate it with two runs of the
same binary rather than weakening this command globally. Also verify the two
sample directories contain the same 60 frame indices: the comparator currently
uses the shorter count instead of reporting extra trailing files.

Review all of:

- `~/git/Vulkan/build/iter/candidate-label/verdict.txt`.
- `~/git/Vulkan/build/iter/candidate-label/iteration.json`.
- Per-sample logs and rendered frames.
- Both `bench/_timing.csv` and `_render/_timing.csv`: each must contain all 18
  expected sample rows and exit 0 for every row. `cp_perf_run.sh` silently skips
  a sample whose binary is absent or not executable.
- Exactly the same 60 frame indices in candidate and reference for every
  applicable offscreen sample. `cp_compare_frames.py` compares only the shorter
  list, and a `missing` row does not make it fail; reject every unexplained
  `missing` row rather than trusting the final summary.
- The explicit image comparison against `previous-label`, not only the stored
  NVIDIA verdict and the timing delta.

Do not mistake a comparison that failed to execute for a comparison failure.
The helper has historically printed a Python import error where the verdict
belongs and still printed the timing delta below it. Require a complete verdict
table and inspect its text, not merely the wrapper status or the existence of
`iteration.json`.

`renderheadless` is a known hole in the automated image gate: it writes
`headless.ppm` in its working directory instead of entering the offscreen loop,
so both its stored directory and the NVIDIA directory are empty. `missing` plus
an `IDENTICAL` comparison of two empty directories is not a pass. For a change
that can affect it, run the baseline and candidate manually in different empty
directories and compare their `headless.ppm` files as a regression diagnostic.
Also produce a manual NVIDIA (or llvmpipe, if valid for the sample) PPM before
claiming external correctness; baseline/candidate identity alone is not enough.

Six samples have a nonzero same-build image floor:
`multithreading`, `gltfscenerendering`, `instancing`, `multisampling`,
`vulkanscene`, and `particlesystem`. Before assigning a small movement in one
of them to the candidate, run that sample twice with the same binary. This does
not excuse missing objects or a coherent region; it only calibrates the last
few pixels. Also check every process exit: four samples once rendered correct
images and then segfaulted during teardown for weeks.

A green sweep says only that the paths reached by these small samples stayed
correct. Record which sample or capture actually exercises the changed state;
several real capture bugs were structurally unreachable from the sample set.

A standing NVIDIA difference remains in `gltfscenerendering`. A standing
failure is not permission for it to grow: record its frame number, differing-
pixel count, maximum delta, and coherent regions against both NVIDIA and the
immediate baseline. The old Gallium `texture3d` single-slice result is knowingly
wrong; the native 3D differential test and corrected filtered result match
NVIDIA, so Gallium must not be used as that path's oracle. Explicitly inspect
`instancing` and `multithreading`; an earlier scratch allocator regression
caused most objects to disappear while performance work continued.

## Performance Validation Is Separate

Use the FPS replay only after image correctness passes:

```bash
mkdir -p /tmp/cp-validation/timing
cd /tmp/cp-validation/timing

VK_DRIVER_FILES="$CP_ICD" \
  $GFX/tools/replay/gfxrecon-replay \
  -m remap --remove-unsupported \
  --replay-event-plugin-path $HOME/claude-scratchpad/perf16/fps_plugin.so \
  --replay-event-plugin-params submits.txt \
  "$CROSSROADS" > replay.log 2>&1

$TESTS/cp_gfxr_frames.py fps submits.txt --per-frame 2
```

Run baseline and candidate as separate replays, one GPU job at a time. Pair
runs because changes below roughly 10% can be obscured by variance.

The FPS plugin proves only timing and frame completion. It does not capture
pixels. Completing every frame is never a correctness result without extracting
and comparing pixels. In particular, the tiled measurements from 2026-08-18
completed both captures but extracted none of the Crossroads sentinel frames;
they are timing results only.

## When a Gate Fails

Triage the first divergence rather than relaxing a threshold:

1. Reproduce it with the immediate baseline and candidate, the same plan, and
   no concurrent GPU job. Confirm both replay exits and logs.
2. For an opt-in change, run the candidate with the flag unset. If that differs,
   debug default-path perturbation before entering the new path.
3. Find the earliest divergent frame, then narrow to the first draw or resource
   operation. GFXR's JSON `DumpResources` mode can dump around a command; do not
   infer the first bad draw from the final frame alone.
4. Inspect the **first** CUDA/NVRTC error. CUDA context faults are sticky, so
   later launch, copy, and teardown failures are usually consequences.
5. If needed, disable one registered subsystem at a time and save
   `CUDAPIPE_HELP=1` output for each run. A bisect result without the resolved
   flags is not reproducible.
6. For suspected out-of-bounds, use-after-free, uninitialized-device-memory, or
   synchronization faults, reduce to the smallest sample/replay and run CUDA's
   diagnostics, for example:

   ```bash
   APP=$HOME/git/Vulkan/build/bin/instancing
   VK_DRIVER_FILES="$CP_ICD" compute-sanitizer \
     --tool memcheck --error-exitcode 99 "$APP"
   ```

   Follow with `initcheck` or `synccheck` when the first report points there.
   Sanitizer runs are diagnostics, not performance runs.

Preserve the failing images, difference map, plan, replay log, and exact binary
before trying a fix. Do not replace an unexplained structural difference with a
larger allowed threshold.

## Acceptance Checklist

Before retaining or committing a correctness-sensitive performance change:

- The build and an NVRTC runtime smoke test succeed; `cp_debug_doc.py --check`
  passes.
- Baseline and candidate binaries are identified, including dirty diffs, build
  type, library SHA-256, and resolved flags.
- The candidate with every new opt-in flag unset matches the immediate baseline;
  this is a regression diagnostic, not the external correctness verdict.
- Both sample `_timing.csv` files contain all 18 expected rows at exit 0; all
  applicable reference/candidate directories contain the same 60 frame indices,
  the comparison table actually ran, and it has no unexplained `missing` row.
- The stored NVIDIA reference's sample-build, driver, orbit, count, and checksum
  are known and still match this sweep.
- `renderheadless` is tested manually when the change can affect it; two empty
  stored directories are not counted as a pass, and a Cudapipe/Cudapipe match
  is not substituted for a manual external reference.
- Every applicable sample is judged against the identified NVIDIA output;
  Gallium/native agreement is used only to localize frontend regressions.
- No new sample deviation appears relative to the immediate baseline or the
  external reference, standing
  deviations do not materially grow, and the same-build floor is measured for
  any small movement in a nondeterministic sample.
- `instancing` and `multithreading` visibly retain all objects.
- Every GFXR run starts in a new empty output directory. Both captures complete
  normally and produce exactly the planned dumps; their replay logs contain no
  new CUDA, compiler, fallback, or overflow issue.
- Each capture uses one shared, checksummed plan for release llvmpipe or NVIDIA
  reference, baseline, and candidate; files are paired through the replay
  manifest rather than names. Native or Gallium-hosted Cudapipe output is not
  accepted as the reference.
- Crossroads frames 633, 756, and 907 are extracted and compared.
- Old-capture loading and representative scene frames are compared; established
  deterministic loading images remain exact.
- Structural-threshold populations do not grow, and the worst numeric
  differences and coherent regions are visually inspected.
- A same-build replay establishes the capture's nondeterminism floor when small
  changes are present.
- High-risk clipping, memory, copy, synchronization, rasterization, and episode
  changes receive full-frame validation against llvmpipe or NVIDIA, not
  sentinels alone or a second Cudapipe frontend.
- Timing is measured separately, without dumping or a profiler, and only after
  correctness is accepted.
- Commands, capture/GFXR/build identities, flags, checksummed plans, logs,
  lossless images, metrics, and conclusions are saved in a durable experiment
  directory and summarized in the relevant Cudapipe Markdown log.

## Related Documentation

- `src/gallium/drivers/cudapipe/tests/GFXRECONSTRUCT.md`
- `src/gallium/drivers/cudapipe/HEADLESS_STREAMER_PERF.md`
- `src/gallium/drivers/cudapipe/tests/TESTING.md`
- `src/gallium/drivers/cudapipe/EPISODES.md`
