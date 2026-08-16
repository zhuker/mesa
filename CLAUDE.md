# cudapipe

This tree is a Mesa fork used for one thing: `src/gallium/drivers/cudapipe`, a
CUDA software rasterizer exposed as a Vulkan ICD. No other part of Mesa is
being worked on here.

## Read first

- `CUDAPIPE_HANDOFF.md` — what the driver is, how to build and run it, its
  architecture, the known gaps, and the lessons that cost the most to learn.
- `src/gallium/drivers/cudapipe/tests/TESTING.md` — how correctness is checked,
  how cost is measured reliably enough to compare, and how to find where the
  time actually goes. Read it before trusting a number from either half.

`CUDAPIPE_PLAN.md` and `src/gallium/drivers/cudapipe/{PERFORMANCE_PLAN,
PERFORMANCE_PROGRESS,PHASE_1A,INSTANCING,BATCHING,ABUFFER}.md` are records of past
passes. Their forward-looking sections have been overtaken and say so where
they have.

## Environment switches

The driver has 43 of them and reads **none** of them with `getenv`. They are
declared in one array in `src/gallium/drivers/cudapipe/cp_debug.c`, resolved
once at screen creation into a read-only `struct cp_debug`, and read as
`cp_debug->field`.

**A new switch goes in that array. Do not add a `getenv` to the driver.** The
array is the single source of truth for the name, the parse, the default and
the one-line meaning, which is what makes `CUDAPIPE_HELP=1` and the generated
`FLAGS.md` correct by construction. A `getenv` somewhere else is invisible to
both, and the flags are this driver's debugging surface — the point of the
registry is that the next person can find them without grep.

- `src/gallium/drivers/cudapipe/FLAGS.md` — all of them, generated.
- `CUDAPIPE_HELP=1 <any vulkan app>` — the same table, with what each one
  resolved to in that process.
- `tests/cp_debug_doc.py --check` — fails if `FLAGS.md` has drifted from the
  registry. Run it after touching the array.

Two boolean kinds exist and both are load-bearing: **presence** flags are set
by the variable existing at all, so `CUDAPIPE_DEBUG_DRAW=0` turns tracing
**on**, and **value** flags read the value, so `=0` turns them off. That is
not a design, it is what they grew into, and it is preserved on purpose.
`CUDAPIPE_HELP=1` says which kind each one is; check before assuming `=0` is
off.

`CUDAPIPE_HANDOFF.md` "Debug" has the mechanics of adding one.

## Profiling

Ask these in order; `tests/TESTING.md` "Finding where the time goes" is the
long form, and answering an early question with a later tool is the mistake it
exists to prevent.

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
  cost to every `cuLaunchKernel`, and cudapipe issues thousands per frame, so a
  traced run manufactures exactly the host-side gap being looked for. A trace of
  `multithreading` once reported more GPU kernel time per frame than the
  untraced frame took end to end. See `PHASE_1A.md`.
- **Every compiled shader is a CUDA kernel named `main`**, so any per-kernel
  summary — including the skill's own `report-fact --intent kernel_summary` —
  sums the vertex and fragment stages into one uninterpretable row. Split by
  grid size: `tests/cp_prof_kernels.py`, or the skill's `report-query` grouped
  by `gridX`.

`tests/cp_profile.sh` resolves the newest installed nsys itself and records the
version in its summary. Traces taken with different versions are not comparable
on host-side time.

GPU performance counters are enabled on this machine
(`NVreg_RestrictProfilingToAdminUsers=0`), so `ncu` and `nsys
--gpu-metrics-devices` both work. If either starts failing with
`ERR_NVGPUCTRPERM`, check `/proc/driver/nvidia/params` before suspecting the
tools.
