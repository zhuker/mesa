# Running a measured iteration

This is the operating manual for the driver: how to build it, how to run it on
the two things it is measured against, how to run one performance iteration end
to end, and which measurement rules must not be rediscovered.

Read it once before the first iteration. Everything in "Measurement
conventions" was learned by getting it wrong, and each rule has a wrong answer
behind it that survived being believed for a while.

`docs/cudavk/TESTING.md` is the companion: it is the methodology for
correctness and the long form of the profiling questions. This document is the
commands and the order.

---

## 1. Build

The driver is one meson option. It is off by default and is deliberately not in
`vulkan-drivers=auto/all`:

```bash
cd ~/mesa
./venv/bin/meson setup build-cudavk -Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers=
./venv/bin/ninja -C build-cudavk
```

The last recorded standalone setup also passed `-Dglx=disabled -Degl=disabled
-Dplatforms= -Dgbm=disabled` and built `debugoptimized`
(`/tmp/perf16/step1-setup.log`, "User defined options"). None of that is
required to get the ICD; it only keeps the configure step small.

The ICD lands at:

```
build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json
```

**Name the build directory `build-cudavk`.** `MESA` and `VULKAN` are
environment overrides in every script, but the build directory name is not: it
is written into `cp_iterate.sh`, `cp_gpu_busy.sh`, `cp_profile.sh` and
`cp_gfxr_timeline.sh` as `$MESA/build-cudavk`. `CUDAVK.md`'s quick start uses
`build` because it is a quick start; the tooling does not follow it.

### The venv, and the PATH trap that looks like a compiler error

meson and ninja come from a pip venv at `./venv` — the distro meson is too old.
It also needs `mako`, `packaging` and `pyyaml`. Two failures come out of this
and neither says what it is:

* **`ninja` fails before compiling anything.** `build.ninja` records the
  absolute path of the meson that generated it, so a missing, moved or
  unactivated venv breaks the regeneration step rather than a compile. Recover
  with `./venv/bin/meson setup --reconfigure build-cudavk`.
* **configure cannot find `glslangValidator`.** It comes from the Vulkan SDK at
  `~/vulkan-sdk/1.4.357.1/x86_64`, which has to be on `PATH` at setup time.

Both are avoided by exporting the same `PATH` the harness does before doing
anything by hand:

```bash
export PATH="$HOME/mesa/venv/bin:$HOME/vulkan-sdk/1.4.357.1/x86_64/bin:$PATH"
```

`cp_iterate.sh` does not rely on this: it calls `$MESA/venv/bin/ninja` by
absolute path and prepends the SDK `bin` itself.

### After editing a `meson.build`

Run `./venv/bin/meson setup --reconfigure build-cudavk`. Two build traps sit
behind this and both were paid for:

* **A `custom_target`'s `input` list is its dependency list.** The fragment
  bitcode target did not name a newly added shared header, so edits to that
  header did not rebuild the `.bc` and the driver linked a stale module. If a
  header edit appears to have no effect on an embedded-bitcode path, check the
  input list before anything else
  (`/tmp/perf16/iter27-vsfusion/handoff.md`).
* **A stale build silently measures the previous iteration.** That is the
  single most expensive mistake available here, which is why `cp_iterate.sh`
  builds first and refuses to continue if the build fails.

CUDA is 12.8 at `/usr/local/cuda`, so `cuCtxCreate` takes three arguments; code
written against CUDA 13's four-argument form does not compile.

---

## 2. Running the driver

```bash
export LD_LIBRARY_PATH="$HOME/vulkan-sdk/1.4.357.1/x86_64/lib"
export VK_DRIVER_FILES="$PWD/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json"
<any vulkan application>
```

`VK_DRIVER_FILES` selects the ICD without installing anything. Do not leak it,
`VK_ICD_FILENAMES` or any `CUDAVK_*`/`CPVK_*` variable into a reference or
timing run of another driver.

`CUDAVK_HELP=1 <any vulkan app>` prints all 97 switches with what each one
resolved to in that process. `src/cudavk/FLAGS.md` is the same table,
generated. Note the two boolean kinds: **presence** flags are set by the
variable existing, so `=0` turns them **on**; **value** flags read the value.

The native test suite is the fast gate:

```bash
./venv/bin/meson test -C build-cudavk --suite cudavk
```

65 tests, and 65/65 is the state the branch is kept in
(`docs/cudavk/history/PERF16_ITERATIONS.md`, iteration 28). It includes
`cp_launch_audit`, which fails if a raw `cuLaunchKernel` appears outside the two
allowlisted definitions. Run `src/cudavk/tests/cp_debug_doc.py --check` after
touching the flag registry.

---

## 3. The two captures

Performance is measured on two GFXReconstruct captures of the same offscreen
application, both replayed with no display server:

| name | file | shipping default, paired-submit median |
|---|---|---:|
| old capture | `~/headless_streamer_20260814T155742.gfxr` | **15.7514 ms** |
| Crossroads | `~/headless_streamer_1818_20260817T173522.gfxr` | **5.8982 ms** |

Both figures are the median of run medians of the shipping default measured
alone in one session — six runs on old, four on Crossroads — in
`docs/cudavk/history/PERF16_ITERATIONS.md`, "The standing default, stated as a
distribution". A complete old-capture replay is **3,022 submits**
(`iterations.json`, status note).

`docs/cudavk/GFXRECONSTRUCT.md` is how a capture is taken, read and turned
into frames. To replay one:

```bash
GFX=~/gfxreconstruct/build
VK_DRIVER_FILES=$PWD/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json \
  $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported CAPTURE.gfxr
```

* **`-m remap` is required.** Memory type indices are baked into the capture.
* **`--remove-unsupported` is required.** The application asks for extensions a
  software driver does not expose and `vkCreateDevice` fails outright without
  it.
* **Do not pass `--wsi`.** It requests surface extensions this driver does not
  expose and instance creation fails.

### Timing a replay: the fps plugin

The application never presents, so gfxrecon's own FPS measurement counts
nothing and `--measurement-file` writes an empty file. Instead a replay event
plugin records a timestamp per queue submit:

```bash
$GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
    --replay-event-plugin-path ~/claude-scratchpad/perf16/fps_plugin.so \
    --replay-event-plugin-params OUT/submits.txt \
    CAPTURE.gfxr > OUT/stdout 2> OUT/stderr
```

The plugin's source is `src/cudavk/tests/cp_gfxr_fps_plugin.cpp`; it is
driver-agnostic, so the same plugin measures cudavk, lavapipe and NVIDIA.
`submits.txt` is one line per submit: timestamp in nanoseconds, then the submit
index. The built `.so` lives outside the repo today
(`~/claude-scratchpad/perf16/fps_plugin.so`); building it from the tree as part
of the test target would be an improvement.

---

## 4. Measurement conventions

These are not style. Each one has produced a wrong published answer.

### 4.1 A frame is the interval between every *other* submit

The captures submit **twice per frame**, so submit boundaries are not frame
boundaries. The median is

```python
d = (ts[2::2] - ts[:-2:2]) / 1e6     # ns -> ms, every other submit
median(d[50:])                        # drop the first 50 intervals
```

which is what `/tmp/perf16/cp_two_replay_report.py` computes. Taking
`diff()` over all submits measures the intra-frame gap as well and understates
the frame by about a quarter (`CUDAVK.md`).

The skip is **50**, not 100. Iteration 25's report used 100; iteration 28
re-derived the convention at skip 50 and reproduced the iteration-27 decisive
table exactly, 16.0193 / 16.5119 / 5.9928 / 6.0716
(`/tmp/perf16/iter28-profile/report.md`, "Frame-time convention"). A restated
baseline moved from 17.41 to 17.5146 ms purely from the skip, so a number
compared against an older one has to be re-derived, not looked up.

### 4.2 Alternate the arms, in one session

Run candidate, control, candidate, control, … in one process sequence on one
binary. Do not run all of one arm and then all of the other, and do not compare
against a control measured in an earlier session.

**Control drift between sessions looks exactly like a real effect.** An earlier
pair of iteration-26 stages appeared sub-additive; re-measured with the control
in the same session, the three stages together came out at 102% and 138% of the
sum of the stages measured separately, and the sub-additivity was drift
(`PERF16_ITERATIONS.md`, iteration 26). *A control belongs in the same session
as its candidate.*

Where an effect is near the noise, alternate more: `/tmp/perf16/cp_decisive_ab.sh`
runs strictly alternating arms, six per arm on old and four on Crossroads, in
one session. Prefer it to a two-runs-per-arm AB/BA when the question is "is
this over the line" — AB/BA's spread can be as wide as the whole win.

### 4.3 The candidate arm must carry the new behaviour

The report prints `delta = control - candidate` and calls a positive delta a
win. If the new behaviour is already the default, the *control* arm is the one
that gets the revert flag (`CTRL="CUDAVK_NO_X=1"`), not the candidate. Put the
new behaviour on the control side and the printed verdict inverts silently.

### 4.4 Any probe whose stdout hash changes is invalid until the counts are checked

Every timed run should record `sha256sum` of its stdout. Identical hashes per
capture mean the two arms did the same work and the result is purely a cost
result.

**A different hash is not a small correctness question — it is usually a run
that died.** A run that dies early produces a fast and meaningless median. The
iteration-29 S0 probe first read as **+8.07 ms on the old capture** and was a
device loss after 26 frames: **52 submits against 3,022**
(`/tmp/perf16/iter27-vsfusion/handoff.md`). Check the submit count and the
frame count before believing anything, and check them again when a result is
surprisingly good.

### 4.5 Other standing rules

* **Nothing else on the GPU.** Two passes sharing the card measure each other.
  `nvidia-smi --query-compute-apps=pid` is the check; `cp_iterate.sh` warns.
* **Record the arms.** `arms.txt` naming candidate, control and base
  environment is written before the runs, so a later reader can tell what was
  actually set.
* **A gate flag in a timed run inverts the result.** `CUDAVK_ABUF_FUSE_CHECK`
  is an equivalence gate that synchronises; all sixteen iteration-28 runs
  recorded it unset for that reason.
* **The base environment is empty now.** Everything from iteration 24 onward
  was measured with `CUDAVK_TEXTURE_CACHE=1` in the base environment. That flag
  no longer exists: the hardware texture path is default-on and
  `CUDAVK_NO_TEXTURE_CACHE=1` is the revert (`FLAGS.md`). A harness that still
  sets the old name enables nothing, and a control arm that needs the software
  sampler has to set the `NO_` flag instead.
* **A probe that changes what the compiler can prove is not measuring what it
  names.** A probe that hardcoded the rasteriser's bounding box let NVRTC drop
  the loads it had just made dead and reported a 17% win that did not exist.
  Keep the work observable, and read the generated code when a probe looks too
  good (`TESTING.md`, "Traps").

---

## 5. The two-capture rule

**A win on one capture is not accepted until it is measured on the other.** The
two captures differ in shape — the old capture's frame is about 2.7 times
Crossroads' — and a mechanism can be neutral on one and negative on the other.
Iteration 29's S0 probe was invalid on old and *0.38 ms slower* on Crossroads.
Iteration 25's episode raster removed a net 112 stage-3 launches a frame on
old and a net 15 on Crossroads, because that capture's episodes average 1.36
segments — the same code, a different amount of work to remove.

The reference implementation is:

```
/tmp/perf16/cp_two_replay_ab.sh        runs both captures, AB then BA per capture
/tmp/perf16/cp_two_replay_report.py    medians, deltas, hashes, cross-replay verdict
```

```bash
LABEL=iter30 CAND="CUDAVK_NEW_THING=1" \
  /tmp/perf16/cp_two_replay_ab.sh
python3 /tmp/perf16/cp_two_replay_report.py /tmp/perf16/iter30-tworeplay
```

The report prints a per-capture delta and ends with
`CROSS-REPLAY VERDICT: acceptable` or `BLOCKED: meaningful regression on the
other capture` — it blocks below −0.10 ms on either capture.

**Both scripts live outside the repository, in `/tmp/perf16/`, which is not
durable. Moving them into `src/cudavk/tests/` — with the ICD path, the flag
names and the empty base environment brought up to date — would be an
improvement.** As they stand they still carry pre-rename paths
(`build-cudapipe`), pre-rename flag names (`CUDAPIPE_*` in
`cp_decisive_ab.sh`) and `BASE_ENV=CUDAVK_TEXTURE_CACHE=1`, which is now a
no-op name. Read them as a reference implementation and fix the header before
running one.

---

## 6. The iteration loop: `cp_iterate.sh`

The captures answer "did the frame get faster". The eighteen-sample sweep
answers "did it stay correct while it did", and it is the only loop that writes
its own record. One command is one iteration:

```bash
DESC="what this tried" src/cudavk/tests/cp_iterate.sh mylabel previouslabel
```

It builds, times `BENCH_FRAMES` frames, renders and stores `FRAMES` frames,
compares them to the stored NVIDIA reference, prints the cost delta against
`previouslabel`, and writes the whole thing down.

**Nothing in that list is done by hand.** There is no step where files are moved
into place or a json is edited. The point of the record is that it describes the
run that actually happened.

### The knobs

| variable | default | meaning |
|---|---|---|
| `DRIVER` | `native` | `native` is this driver; `llvmpipe` times lavapipe for calibration; `nvidia` renders the reference |
| `FRAMES` | 60 | the **stored** pass — the correctness gate and the report |
| `BENCH_FRAMES` | 600 | the **timed** pass |
| `BENCH_ONLY` | 0 | skip the frame pass when only cost is in question |
| `METRICS` | 0 | add the device-counter sweep, about five minutes |
| `METRICS_SECONDS` | 12 | per-sample counter window |
| `FORCE` | 0 | overwrite an existing label |
| `DESC` | — | what the iteration was trying; set it |
| `MESA`, `VULKAN` | `~/mesa`, `~/git/Vulkan` | tree locations |

`DRIVER=cudavk` still appears in the script and points at the removed
Gallium target; it no longer resolves.

**Two frame counts, because the two passes answer different questions.** Sixty
frames is the animation worth storing and looking at. Sixty frames is far too
few to measure cost: a sample's process spends seconds on start-up, shader
compilation, warm-up and teardown, so at sixty frames the render loop is a
minority of the run. Six hundred puts every sample above ten seconds of
rendering. They still render the same work — the orbit is periodic — and the
totals agree to 0.8%. Raise `BENCH_FRAMES` for a final confirmation; 6000 takes
about twenty-five minutes and puts the spread below a tenth of a percent.

### What lands where

Everything an iteration produced is one directory under
`~/git/Vulkan/build/iter/LABEL`:

```
build/iter/
    nvidia/<sample>/frame0000.png   the reference, rendered once and reused
    llvmpipe/<sample>/...           the calibration renderer
    LABEL/
        <sample>/frame0000.png      the frames this build rendered
        bench/_bench.csv            the timed pass, plus a csv per sample
        _render/                    the storing pass's own gpu and timing csvs
        verdict.txt                 cp_compare_frames.py against the reference
        delta.txt                   ms/frame against COMPARE_LABEL
        _commit.txt                 HEAD and a diffstat
        iteration.json              all of it as one record
    iterations.json                 every iteration.json, collected
    iterations.html  perf.html      the pages
```

That layout is exactly the root `cp_compare_frames.py` wants — one directory per
thing being compared — so an iteration is one directory that can be copied,
kept or deleted whole.

* **`verdict.txt`** is the correctness gate. `cp_compare_frames.py` exits 1 both
  when a sample regressed and when the comparison never ran, so the exit status
  cannot tell them apart while a cost number prints underneath either way.
  `cp_iterate.sh` therefore refuses to report a cost delta unless a verdict
  table was actually produced, and `iteration.json` carries
  `correctness.gate_ran` so the page can say so.
* **`iteration.json`** is the record: the commit, `DESC`, the cost, the delta
  per sample, which samples moved past 5% either way, the metrics rows if
  `METRICS=1` ran, and what the gate said. An iteration without metrics records
  **no** metrics rather than zeroes, because a row of zeroes reads as "the
  device was idle" instead of "nobody measured".
* **A reused label is refused.** Reusing one overwrites an iteration in place,
  and the label that suggests itself for a change is the one that suggested
  itself last time something touched that code. It has happened once and the
  earlier numbers a document still cites are simply gone. `FORCE=1` overrides,
  for the case where the old numbers are the ones being replaced.

### The pages

`cp_iterate.sh` calls `cp_iter_report.py record` and then `page` itself, so an
iteration appears in the history without anyone remembering to. Run the tool by
hand only for these:

```bash
# refresh the pages after editing a record or adding one from elsewhere
src/cudavk/tests/cp_iter_report.py page

# fix or add a description on an iteration already run, without re-running it
src/cudavk/tests/cp_iter_report.py record LABEL --against PREV --desc "..." --note "..."

# ppm -> png for an older tree, or one interrupted before it converted
src/cudavk/tests/cp_iter_report.py convert [LABEL ...]
```

`record` is idempotent and re-reads everything from the iteration's own files,
so re-running it can only pick up a description or a note — it never invents a
number. `page` rebuilds `iterations.json` from the per-iteration records rather
than appending, which is why an iteration that ran and recorded itself used to
stay invisible until someone remembered a second command.

**The pages must be served over HTTP.** Both are static and fetch their data at
load time, so a `file://` URL shows an empty page. For example:

```bash
python3 -m http.server -d ~/git/Vulkan/build/iter 8000
# then http://localhost:8000/iterations.html
```

`iterations.html` is the history, a row per iteration, and each row links to
`perf.html?iter=LABEL` — that iteration in full: cost and verdict tables, GPU
load and memory per sample, frame time over the run, and differing pixels per
frame against the reference, with llvmpipe drawn as a second dashed line so a
residual can be read as "what software rasterisation costs" or "a cudavk
defect". A single frame is linkable:
`perf.html?iter=LABEL&sample=NAME&frame=N`.

Read the poll count before any GPU percentage on those pages: rows under twelve
polls are greyed because a mean of four polls is not a utilisation figure.

`iterations.json` at the tree root is the separate, hand-curated record of the
28 performance iterations with their verdicts; `docs/cudavk/history/PERF16_ITERATIONS.md`
is its long form.

---

## 7. The rest of the toolchain

Everything below is in `src/cudavk/tests/`.

| tool | reach for it when |
|---|---|
| `cp_iterate.sh` | one whole iteration: build, time, render, gate, record |
| `cp_perf_run.sh` | one driver's pass over the sweep, if a pass is needed outside an iteration |
| `cp_compare_frames.py` | every frame of an animated sweep across drivers — this is what `verdict.txt` is |
| `cp_compare.py` | two images: percentage differing plus a coarse ASCII map |
| `cp_gallery.py` | an HTML page of reference / result / difference per sample, worst first |
| `cp_perf_report.py` | an HTML report over a multi-driver sweep that was not an iteration |
| `cp_iter_report.py` | record, page, convert (see above) |
| `cp_gpu_busy.sh` | is the frame host-bound or kernel-bound — no profiler attached |
| `cp_metrics_sweep.sh` | is the device full, for every sample, in one table |
| `cp_profile.sh` | which kernel owns the frame; also `METRICS=1` for counters |
| `cp_prof_kernels.py` | make a trace readable: split `main` by grid size |
| `cp_prof_nvtx.py` | what the **host** spent its time issuing, by pipeline stage |
| `cp_cpu_profile.sh` | on-CPU and off-CPU folded stacks for one command |
| `cp_gfxr_frames.py` | pull rendered frames out of a capture that never presents |
| `cp_gfxr_timeline.sh` | a clickable frame-time timeline with the frames beside it |
| `cp_capture_requirements.py` | what formats, samplers, stages and extensions a capture requires |
| `cp_debug_doc.py` | regenerate `FLAGS.md`; `--check` fails if it has drifted |
| `cp_launch_audit.py` | fail the suite on a raw `cuLaunchKernel` outside the allowlist |
| `cp_nsys_skill_fix.py` | after an Nsight Systems upgrade, before trusting its skill pack |
| `cp_offscreen_bench.c` | a self-contained scene benchmark with no sample tree |
| `cp_gfxr_fps_plugin.cpp` | the source of the submit-timestamp plugin |

---

## 8. Where the time goes: ask in this order

From `CLAUDE.md`. Answering an early question with a later tool is the mistake
the order exists to prevent.

| question | tool | what the answer means |
|---|---|---|
| Is the frame host-bound or kernel-bound? | `cp_gpu_busy.sh SAMPLE 20` | above ~90% busy the host is already ahead of the device and submitting faster wins nothing; below that, the gap is what work on submission and synchronisation is worth |
| Is the GPU full, or merely occupied? | `METRICS=1 cp_profile.sh` | `SMs Active` against `SM Issue`. High `GR Active` with low issue means small kernels that never fill the machine — a launch-structure problem, not a slow kernel |
| Which kernel owns the frame? | `cp_profile.sh` then `cp_prof_kernels.py` | a ranking, and a hypothesis. A kernel whose duration does not vary with the draw is sized to a worst case the host computed |
| Why is that kernel slow? | `NCU=1 cp_profile.sh` | per-kernel counters. Only worth reaching for once question 2 says the device is full |

Four things override the tools' own reading of themselves:

* **Busy is not working.** `particlesystem` runs at 94% GR Active, issues
  instructions on 5% of cycles and has about a third of its SMs active. Every
  "kernel-bound" verdict written from a busy percentage alone means only that a
  kernel was resident. The current captures sit in the same place: 74% busy on
  old and 59% on Crossroads, with 26% and 41% of the frame having no device
  operation at all (`/tmp/perf16/iter28-profile/report.md`).
* **A profiler cannot answer question 1.** CUPTI adds host-side cost to every
  `cuLaunchKernel` and this driver issues thousands a frame, so a traced run
  manufactures exactly the host-side gap being looked for. A trace of
  `multithreading` once reported more GPU kernel time per frame than the
  untraced frame took end to end. `cp_gpu_busy.sh` attaches nothing.
* **Both counter modes need enough frames.** A sample's process is mostly
  compilation and teardown; ten frames of `instancing` measures 1% GPU busy.
  `METRICS=1` warns when its window is mostly idle. Raise `FRAMES` until it
  stops warning.
* **Every compiled shader is a CUDA kernel named `main`.** Any per-kernel
  summary — including the Nsight skill pack's own `kernel_summary` — sums the
  vertex and fragment stages into one uninterpretable row. Split by grid size
  with `cp_prof_kernels.py`, or `GROUP BY gridX`.

Two further rules apply to any trace taken since the driver went multi-stream:
sum device time as a **union of intervals**, not per kernel, because kernels on
different streams overlap; and a removed wait is only a win if the **sum over
every synchronisation site** drops, because the next synchronisation downstream
inherits the backlog. `TESTING.md` has both, under "Finding where the time
goes", question 5, with the measurements behind them.

Nsight Systems ships an agent skill pack at
`/opt/nvidia/nsight-systems/2026.4.1/skills/nsight-systems/SKILL.md`. Run its
bootstrap first and use the Python it reports. Its `search-docs` and
`lookup-recipes` fail out of the box with `content hash mismatch`;
`cp_nsys_skill_fix.py` repaired this install, and an upgrade puts the broken
manifest back. Quote multi-word `--query` arguments or the shell splits them
and the error reads like a wrong command.

`cp_profile.sh` resolves the newest installed nsys itself and records the
version in its summary. **Traces taken with different versions are not
comparable on host-side time.**

---

## 9. What a kept change has to show

Not a checklist to be filled in — this is the shape the accepted iterations
have, from `PERF16_ITERATIONS.md`:

1. **A design written before the code**, saying what it expects to save and
   what would make it wrong.
2. **A revert flag in the registry** (`cp_debug.c`), so both arms are the same
   binary. Every mechanism kept so far has one; `CUDAVK_NO_*` when the new path
   is the default.
3. **The native suite at 65/65 in both states**, default and reverted.
4. **A device-side equivalence gate** where the change alters what the GPU
   computes, run over a whole replay and reported as counts of disagreements —
   not a spot check.
5. **The timed result on both captures**, alternating arms, one session,
   identical stdout hashes per capture, submit counts checked.
6. **The 18-sample sweep** through `cp_iterate.sh`, showing only the standing
   documented exceptions.
7. **An entry in `iterations.json`** with the verdict.

`TESTING.md` is the authority on 3 and 6. The equivalence gates in 4 are
per-mechanism and are described with the mechanism they gate, in
`docs/cudavk/history/PERF16_ITERATIONS.md`. Open questions and unfinished work
belong in `docs/cudavk/TODO.md`, not here.

---

## 10. Where the raw evidence is

The recent work left its evidence outside the repository, under `/tmp/perf16/`,
which is not durable:

| path | what it holds |
|---|---|
| `cp_two_replay_ab.sh`, `cp_two_replay_report.py` | the two-capture harness and its report |
| `cp_decisive_ab.sh` | the strictly alternating measurement |
| `iter25-profile/report.md`, `iter28-profile/report.md` | the frame decomposed, twice, three iterations apart |
| `iter26-smallops/design.md` | the small-operations removal design |
| `iter27-vsfusion/design.md`, `handoff.md` | the vertex-fetch fusion, and the price table for a removed device operation |
| `iter28-item3/README.md` | the parked writeback fusion and the invariant it disproved |
| `iter29-sync/design.md`, `addendum.md` | the synchronisation work that was designed but not landed |

Anything from there that a later decision will depend on should be copied into
`docs/cudavk/` before it is needed.
