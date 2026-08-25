# CPU-side profiling for cudapipe

How to see where *host* time goes, in formats meant to be read by an agent
(ranked text, one-line-per-stack, greppable) rather than a GUI. This augments
nsys; it does not replace the decision table in the top-level `CLAUDE.md` —
`tests/cp_gpu_busy.sh` still answers host-bound vs kernel-bound first, and
none of the tools below attach anything to the CUDA API, so they avoid the
CUPTI distortion that makes traced runs untrustworthy for host-side cost
(thousands of `cuLaunchKernel`s per frame, each inflated; see `PHASE_1A.md`).

perf is the engine for all of this. Its interactive TUI is the weak part;
skip it and use the text outputs.

For a repeatable paired capture, use the repository wrapper after caching
sudo credentials:

    sudo -v
    tests/cp_cpu_profile.sh /tmp/cpu-profile 30 -- ./demo --offscreen

It writes `oncpu.folded`, `oncpu-flat.txt`, `offcpu.folded`, the raw
`perf.data`, and the target log. `DELAY=seconds` controls how much startup is
skipped before both profilers attach.

## Folded stacks — the primary format

One line per unique call chain, with a sample count. Grep it, sort it, diff
two runs of it, sum subtrees with awk. This is the format flamegraph SVGs are
rendered from; skip the rendering.

    perf record -F 999 -g --call-graph dwarf -o /tmp/perf.data -- ./demo --offscreen
    perf report -i /tmp/perf.data --stdio -g folded,0.5,caller --no-children

Canonical alternative (needs Brendan Gregg's FlameGraph scripts):

    perf script -i /tmp/perf.data | stackcollapse-perf.pl | sort -t' ' -k2 -rn

yields lines like:

    main;draw_cb;cp_launch_grid;cuLaunchKernel 412

## Flat hot-list and per-line attribution

Ranked self-time table:

    perf report -i /tmp/perf.data --stdio --no-children -g none

Per-instruction / per-source-line counts inside one hot function:

    perf annotate -i /tmp/perf.data --stdio2 <symbol>

`pprof` can ingest `perf.data` and gives slightly cleaner `-top` / `-list
'regex'` output, but it is an extra dependency for marginal gain.

## Off-CPU time — usually the bigger half here

Host-bound time in cudapipe is often *blocked* time — `cuStreamSynchronize`,
futexes, ioctls — which on-CPU sampling cannot see at all. bcc's offcputime
emits folded stacks of blocked time directly, so on-CPU and off-CPU profiles
read identically:

    offcputime-bpfcc -f -p <pid> 30 | sort -t' ' -k2 -rn

(`-f` = folded output; the trailing number is the capture duration in
seconds. Needs root or CAP_BPF.)

The on-CPU folded profile plus the off-CPU folded profile together account
for the whole host frame.

CUDA synchronization may busy-spin rather than block. In that case
`cuStreamSynchronize` appears prominently in the on-CPU folded profile and is
largely absent from `offcputime`; that is still wait time, not useful host
computation. Confirm it by the caller stack and pair it with the corresponding
device interval before attempting to optimize instructions inside that caller.

## Caveats on this tree

- **Always `--call-graph dwarf`.** Mesa release builds do not keep frame
  pointers; default `-g` (fp unwinding) produces broken one-frame stacks that
  look like profiler incompetence but are just the wrong unwinder mode.
  DWARF recordings are large — bound the run to a fixed number of frames the
  way `tests/cp_profile.sh` does.
- **`kernel.perf_event_paranoid` must be ≤ 2** for unprivileged user-space
  sampling. Check `sysctl kernel.perf_event_paranoid` once before debugging
  a permissions error mid-profile.
- **Sampling frequency:** `-F 999` (not 1000) avoids lockstep with periodic
  timers. Raise it only for very short runs.
- **Comparing runs:** folded-stack files diff well with
  `difffolded.pl` (FlameGraph repo) or plain awk joins on the stack string.
  Compare counts as fractions of total samples, not absolutes — run lengths
  differ.

## If sampling ever stops being enough

Tracy (instrumentation profiler, zone macros in the code, frame-oriented
timeline) is the escalation path when the question becomes "what happens
inside frame N" rather than "which function is hot overall". It requires
adding markers and a GUI to read the capture, so it is a human tool, not an
agent tool — reach for it deliberately.

For a human-facing sampled view, `samply record <cmd>` opens the same
perf_event data in the Firefox Profiler UI (timeline, flame graph, off-CPU).
Same engine, different audience.
