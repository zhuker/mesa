# Toward 16 ms/frame on the old capture

Working log for the performance goal: old-capture GFXR replay
(`headless_streamer_20260814T155742.gfxr`) at a paired-submit **median ≤ 16 ms**
without sacrificing correctness, preferring designs that keep later work
possible over quick wins. Structured record: `iterations.json` /
`iterations.html`. Correctness gates for every kept iteration: 43-test Meson
suite, six batch reproducers byte-identical across five batching modes,
llvmpipe sentinels on both captures, NVIDIA 60-frame comparison.

## Baseline (commit `5ebc37aa36e`)

- old capture median **24.53 ms**, Crossroads 7.61 ms, 600-frame sweep hot sum
  30.44 ms.
- Frame decomposition (perf + `nvidia-smi` + METRICS + NCU, reports in
  `/tmp/cpvk-kernel-ladder.md`): GPU 85–96% GR-active but **starved** — SM
  issue 3–7%, ~25% SMs active, 5 warps in flight on 170 SMs. 1,789
  launches/frame at 3.5 µs median, near-serialized (kernel-sum 38.8 s vs
  busy-union 34.0 s). Compiled fragment class 9.16 ms/frame (35.6% of kernel
  time), 126–210 regs → 1–2 blocks/SM. Main thread blocks in episode drains
  (36.5 s/replay), peel checks (13.1 s), segment counters (5.8 s): legitimate
  completion waits on a starved GPU.
- Ranked levers from the ladder: (1) register-cap tuner skips sampler-variant
  fragment shaders — 87% of fragment time untuned; (2) launch granularity —
  fuse/overlap the sub-20 µs chains; (3) fragment input-load hoisting
  (`long_scoreboard`); (4) stage3+A-buffer fusion (4.5 ms/frame); (5)
  depth/stencil load/store elision (~1.1 dead kernels/frame here).

## Iteration 1 — per-variant register-cap trials

**What:** sampler-variant fragment shaders inherited the base shader's
register-cap verdict, which was timed on the *generic* sampler path; the
specialised build inlines its sampler and has different register pressure.
Now a variant whose inherited build is register-bound below the occupancy
target gets its own capped alt build and its own A/B trial — same state
machine, generalized to a `cp_tune_ctx` view over either owner.

**Result so far:** mechanism verified on the old capture. Six variants went on
trial; five concluded in one replay: `80.4 → 64.0 µs CAPPED`,
`31.6 → 31.5 µs CAPPED`, three `left as built` (11.2, 48.0, 27.6 µs —
differences under 1%). The base trials on this capture are also mixed, with
one large win (1865.7 → 1155.6 µs CAPPED).

**Honest read before timing:** the trial verdicts suggest the ladder's
occupancy lever is real but *selective* — most variant launches on this
capture are not register-bound in the way the 4096×256 tail is, which matches
NCU's finding that the slow launches were per-warp efficient. Expect a modest
median move, not a transformative one.

**Interference:** none expected with other levers — the trial only picks
between two builds of the same shader. It adds ~80 launches of trial overhead
per variant (skips + samples), amortized over a replay.

**Timing:** old median **24.51 ms** vs 24.53 baseline — neutral on this
capture. That is consistent with NCU rather than contradicting the ladder: the
long fragment launches (the class's 43% tail) were already per-warp efficient,
and the trial now *proves* per shader what a static cap would have guessed.
**Kept**: the mechanism is self-selecting (a variant that is not
register-bound is never trialed), costs ~80 amortized launches per variant,
and removes the "variants are 87% untuned" blind spot for workloads where the
cap does matter (the base trial's 1865→1155 µs win shows the shape). Gates:
43/43, five-mode hashes unchanged.

**Refactor note:** none needed — the `cp_tune_ctx` generalization is the
refactor, and it leaves the trial machinery ownable by any future build pair
(e.g. a fused-writeback build vs split, if iteration 2+ wants an A/B).

## Iteration 2 — (next) launch granularity: fuse the fragment three-launch chain

The frame's dominant cost is 1,789 near-serialized launches; the shade path is
three launches per batch (interpolate → fs → writeback) with a full
global-memory round-trip (`fs_in`/`fs_out`) between each. The A-buffer path
already fuses interpolation *into* the generated shader
(`CUDAPIPE_NO_FUSED_ABUF_INTERP` is the revert switch), so the mechanism has
an in-tree precedent. Plan: extend the fused form to the direct path,
starting with interpolate→fs; writeback second. Expected: −2 launches and −1
buffer round-trip per shade, ~204 batches/frame here.
