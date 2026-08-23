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

## Sizing the road to 16 ms (from the ladder's class table, ms/frame)

| class | ms/frame | launches/frame | median µs | note |
|---|---:|---:|---:|---|
| fs stage (generated) | 9.16 | 103 | 65 (grid-4096 class) | iteration 2 target |
| rasterize_stage3_abuf | 2.44 | 128 | 18.1 | rasterize-chain fusion |
| rasterize_stage3 | 2.04 | 82 | 7.9 | rasterize-chain fusion |
| clip_triangles | 1.73 | 204 | 3.9 | one per batch — vertex-chain fusion |
| rasterize_stage1 | 1.71 | 82 | 13.7 | rasterize-chain fusion |
| fs_interpolate | 1.68 | 82 | 18.9 | removed by iteration 2 if fused |
| vertex_fetch | 1.52 | 204 | 4.2 | one per batch — vertex-chain fusion |
| vertex main (generated) | 0.75 | 204 | 3.5 | vertex-chain fusion |

Needed: −8.5 ms. The candidates, in dependency-free order:

- **Iteration 2 (in flight):** interpolate→fs fusion on the direct path.
  Upper bound ≈ fs_interpolate's 1.68 ms plus the fs_in round-trip's share of
  the fs class and the per-launch gaps.
- **Iteration 3: vertex-chain fusion.** vertex_fetch + generated VS (+ clip's
  launch) are 204 launches/frame *each* at ~4 µs — 4.0 ms/frame across three
  classes that form a linear pipeline per batch with global-memory hand-offs.
  Fusing fetch into the generated VS mirrors the fused-interp mechanism
  exactly (a per-lane helper reading the vertex buffers). Clip is
  per-triangle, different grid — stays separate at first.
- **Iteration 4: rasterize-chain fusion** (stage1 → stage3 → stage3_abuf):
  6.19 ms/frame at 8–18 µs per launch. The ladder's lever 4.
- **Iteration 5: episode-drain overlap** — bounded by the ~15% GR-idle
  (≈3.7 ms), and it compounds with the fusions (fewer, larger kernels make
  the remaining gaps a bigger fraction).

Interference map: iterations 2–4 all edit the codegen/kernels/renderer trio,
so they are strictly sequential. Iteration 5 edits the episode machinery and
can only start after 2–4 settle. Load/store elision (~1.1 kernels/frame here)
is independent but small; it stays parked until the sequential chain is done.

## Iteration 2 — result: kept (commit `a9a952f1c09`)

Old median **24.42 → 24.14/24.18 ms** (−1.2%), sweep 30.41 → 29.81
(gltfscenerendering 9.39 → 9.00). All gates pass, including six-mode batch
hashes (five batching modes + `CUDAPIPE_NO_FUSED_INTERP=1`) and both llvmpipe
sentinel sets at their recorded envelopes to the digit.

**Why so small, in numbers:** the direct chain is 7.0 ms/frame of GPU busy,
but 5.5 of that is the generated fs kernels, which fusion keeps by design.
The interpolate launch was a full-framebuffer grid with 1-in-30 occupancy
(20.7 µs); its replacement compaction is 5.0 µs, and the fs absorbed the
interpolation at +8% kernel time. Structurally right, quantitatively bounded.

**Interference:** none with stage3/abuf work (abuf fs time measured
unchanged); writeback fusion is now *easier* (the fs knows its slot's
coverage and pixel, and per-(pixel,sample) writes are unique in a shade);
one fewer serialized kernel helps future overlap.

## Iteration 3 — (next) rasterize-chain fusion

Largest remaining non-fs class group: rasterize_stage1 (1.71) + stage3 (2.04)
+ stage3_abuf (2.44) = **6.19 ms/frame** at 8–18 µs/launch, communicating
through device queues written and reread across kernel boundaries. Measure
first: how much is launch gap + queue round-trip vs irreducible work.
Candidate fusions in ascending risk: clip→stage1 (adjacent per batch),
stage1→stage3 (producer/consumer over the nontrivial-triangle queue),
stage3→abuf-build. Kernel-only work (cp_rasterize.cu + renderer launch
sites), no codegen. Vertex-chain fusion (fetch+VS+clip, 4.0 ms class, 204
launches/frame each) is the fallback if the queue handoff proves
irreducible.

## Iteration 3 — result: kept (commit pending review)

**Kept design:** factor clipping and raster stage 1 into per-primitive device
bodies and execute them in one 64-thread-block kernel. Clipping still allocates
stable compact output IDs. Stage 1 still rasterizes small primitives in place
and appends nontrivial work to the unchanged global queue. Stages 2 and 3 stay
machine-wide queue consumers. `CUDAPIPE_NO_FUSED_RAST=1` restores the classic
chain. Instrumented and non-adjacent segment-replay paths refuse fusion.

**Measured result:** old median **23.421/23.672 ms** vs **24.070 ms** with the
revert flag (−0.40–0.65 ms, −1.7–2.7%). Crossroads **7.480 ms**. The 600-frame
sweep is **29.17 → 28.51 ms**; three interleaved glTF pairs showed the fused
form consistently 0.08–0.09 ms faster, proving the first sequential pair's
apparent regression was a machine-state shift. Final trace shows combined
clip+s1 device work also fell: direct 32.369→31.105 µs and A-buffer
13.511→12.342 µs, in addition to removing the 2.784/3.488-µs launch gap.

**Failed sub-iteration — clip+s1+s2:** correctness passed, performance did
not: **24.23 → 27.21/27.27 ms**, Crossroads 7.88. No spills. The fusion let
producer warps consume their own medium-triangle work, serializing up to 32
queued entries behind one warp. Classic stage 2 uses a machine-sized grid (up
to 4096 warps) over the global queue. This is an architectural constraint, not
a block-size tuning problem: later queue consumers can fuse only through a
persistent/cooperative design that preserves machine-scaled consumer
parallelism. The failed form was removed rather than left as dormant code.

**Gates:** clean build/docs/diff; root independently reran 43/43 at `-j8`, both
`FLAGS.md` checks, Gallium byte-integrity, and six reproducible hashes across
five batching modes plus the revert flag. Child validation found both
llvmpipe sentinel envelopes exact and flag-on/off stored frames limited to the
known one-pixel atomic-order nondeterminism.

**Interference:** stage2/3 queue formats are unchanged, so a future persistent
rasterizer starts at the same interface. Stage3+A-buffer fusion, vertex-chain
fusion, and multi-stream overlap remain unblocked. Vertex fusion must honor
the new deferred-clip block when consuming clip itself. The ~5 KiB clipping
stack and unrestricted multi-process CUDA OOM are separate robustness work;
test `-j8` is stable.
