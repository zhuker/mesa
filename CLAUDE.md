# cudavk

This tree is a Mesa fork used for one thing: `src/cudavk`, a CUDA implementation
of Vulkan exposed as an ICD. No other part of Mesa is being worked on here.

    meson setup build -Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers=
    ninja -C build
    export VK_DRIVER_FILES=$PWD/build/src/cudavk/cudavk_devenv_icd.x86_64.json

The Gallium-hosted driver this replaced has been removed. `docs/cudavk/GALLIUM_RETIREMENT.md`
records what it could do that this cannot, and how to restore it from the
`gallium-cudapipe-last` tag.

## Read first

- `CUDAVK.md` — what this is, and which document answers which question.
- `docs/cudavk/ARCHITECTURE.md` — how the driver works, for changing it.
- `docs/cudavk/WORKFLOW.md` — how to build, run and measure an iteration.
- `docs/cudavk/TESTING.md` — how correctness is decided.
- `docs/cudavk/PERFORMANCE.md` — where the time goes and what removing work costs.
- `docs/cudavk/DEAD_ENDS.md` — **read before optimising anything.** Fifteen
  entries: fourteen directions built or probed and closed with measurements,
  one parked with a known next step. Re-running one of them by accident is the
  most expensive mistake available here, and one entry exists precisely because
  a narrow path was rejected on coverage and later paid 5.84 ms once the
  coverage condition it recorded was met.
- `docs/cudavk/TODO.md` — what is unfinished, including known correctness gaps.

`docs/cudavk/history/` holds the long-form records: the decision log, the
iteration-by-iteration performance record, and the documents from the removed
Gallium driver. `docs/cudavk/notes/` holds research that has not been acted on.

## Environment switches

The driver has 97 of them and reads **none** of them with `getenv`. They are
declared in one array in `src/cudavk/cp_debug.c`, resolved once at device
creation into a read-only `struct cp_debug`, and read as `cp_debug->field`.

**A new switch goes in that array. Do not add a `getenv` to the driver.** The
array is the single source of truth for the name, the parse, the default and
the one-line meaning, which is what makes `CUDAVK_HELP=1` and the generated
`FLAGS.md` correct by construction. A `getenv` somewhere else is invisible to
both, and the flags are this driver's debugging surface — the point of the
registry is that the next person can find them without grep.

- `src/cudavk/FLAGS.md` — all of them, generated.
- `CUDAVK_HELP=1 <any vulkan app>` — the same table, with what each one
  resolved to in that process.
- `src/cudavk/tests/cp_debug_doc.py --check` — fails if `FLAGS.md` has drifted
  from the registry. Run it after touching the array.

Two boolean kinds exist and both are load-bearing: **presence** flags are set
by the variable existing at all, so `CUDAVK_DEBUG_DRAW=0` turns tracing **on**,
and **value** flags read the value, so `=0` turns them off. That is not a
design, it is what they grew into, and it is preserved on purpose.
`CUDAVK_HELP=1` says which kind each one is; check before assuming `=0` is off.

A flag whose name begins `CUDAVK_NO_` reverts something that is on by default.
Those are the ones to reach for when bisecting a regression: the four
small-operation stages, the fused vertex fetch, the two A-buffer chain fusions
and the hardware texture path all have one.

## Measuring

Read `docs/cudavk/WORKFLOW.md` before quoting a number. Three conventions
cause wrong answers if they are not known:

- **One frame is two `vkQueueSubmit` events** on both captures, so frame time
  is the interval between every *other* submit. Measuring every submit
  understates the frame by about 25%.
- **Alternate the arms and keep them in one session.** Control drift between
  sessions has already masqueraded as sub-additivity once.
- **Any run whose stdout hash changed is invalid until the submit and frame
  counts are checked.** A replay that dies early produces a fast, meaningless
  median — this happened and looked like an 8 ms win.

A win on one capture must be measured on the other before it is accepted.

## Profiling

Ask these in order; `docs/cudavk/WORKFLOW.md` has the long form, and answering
an early question with a later tool is the mistake it exists to prevent.

| question | tool |
|---|---|
| Is the frame host-bound or kernel-bound? | `tests/cp_gpu_busy.sh` — no profiler attached |
| Is the GPU full, or merely occupied? | `METRICS=1 cp_profile.sh` — device counters, no CUPTI |
| Which kernel owns the frame? | `cp_profile.sh` — CUDA API trace |
| Why is that kernel slow? | `NCU=1 cp_profile.sh` — per-kernel counters |

**Busy is not working.** `particlesystem` runs at 94% GR Active and issues
instructions on 5% of cycles, with a third of its SMs active. Any claim that a
sample is "kernel-bound" on a busy percentage alone is unsupported — that
number only says a kernel was resident.

**The driver is currently host-bound, and that is measured.** It blocks about
seventeen times a frame and spends about 12.44 ms of a 15.74 ms frame waiting;
device idle is 4.16 ms, almost exactly host issue time. Removing device
operations is close to exhausted — see `docs/cudavk/PERFORMANCE.md`.

**Both counter modes need enough frames to be meaningful.** A sample's process
is mostly shader compilation and teardown; ten frames of `instancing` measures
1% GPU busy. `METRICS=1` warns when its window is mostly idle. Raise `FRAMES`.

Nsight Systems ships an agent skill pack:

    /opt/nvidia/nsight-systems/2026.4.1/skills/nsight-systems/SKILL.md

Use it for nsys CLI syntax and report analysis — it is authoritative on the
tool and removes the need to guess flags or table schemas, and its
`report-query` runs bounded SQL against a report. Run its bootstrap first and
use the Python it reports.

**Every command works on this machine — but only because the pack was
repaired here.** A fresh install has `search-docs` and `lookup-recipes` failing
with `content hash mismatch`, because the shipped `manifest.json` records stale
digests. `tests/cp_nsys_skill_fix.py` fixed this install; re-run it after any
Nsight Systems upgrade or reinstall, since the package restores the broken
manifest. It checks by default and exits 0 when there is nothing to do, so it
is also the quickest way to tell whether a newer release fixed this upstream.
Everything that talks to `nsys` or to a report was never affected.

Quote multi-word `--query` arguments. Unquoted they split into extra
positionals and the error reads `unrecognized arguments`, which looks like the
command is wrong rather than the shell.

**It is not authoritative on this driver.** Two project facts override it, and
it has no way to know either:

- **Whether a frame is host-bound or kernel-bound comes from
  `tests/cp_gpu_busy.sh`, which attaches no profiler.** CUPTI adds host-side
  cost to every `cuLaunchKernel`, and cudavk issues over a thousand per frame,
  so a traced run manufactures exactly the host-side gap being looked for.
- **Every compiled shader is a CUDA kernel named `main`**, so any per-kernel
  summary — including the skill's own `report-fact --intent kernel_summary` —
  sums the stages into one uninterpretable row. Split by grid size
  (`tests/cp_prof_kernels.py`) or by the driver's own NVTX ranges: since the
  vertex fetch was fused into the vertex shader, the predecessor-kernel trick
  no longer separates vertex from fragment work, and `--nvtx-include "fs/"` is
  what does.

`tests/cp_profile.sh` resolves the newest installed nsys itself and records the
version in its summary. Traces taken with different versions are not comparable
on host-side time.

GPU performance counters are enabled on this machine
(`NVreg_RestrictProfilingToAdminUsers=0`), so `ncu` and `nsys
--gpu-metrics-devices` both work. If either starts failing with
`ERR_NVGPUCTRPERM`, check `/proc/driver/nvidia/params` before suspecting the
tools.
