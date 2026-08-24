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

## Iteration 4 — (in flight) fuse vertex fetch into generated VS

`cp_vertex_fetch` (1.52 ms/frame, ~204 launches/frame) feeds generated vertex
`main` (0.75 ms/frame, another ~204 launches) through a scratch input buffer.
The clean design is the fragment-fusion pattern: link a per-lane vertex-fetch
helper into the shader and pass an immutable fetch argument block, so formats,
strides, divisors, batch rows, indexed/non-indexed IDs, and the explicit shader
ABI stay owned in one implementation. Keep the old path behind a registry flag
and decline fusion for unsupported/instrumented cases.

## Raster stage2+3 architecture note (analysis after iteration 3)

The rejected producer-local fusion does not rule out stage2+3 fusion; it rules
out binding consumer parallelism to producer warps. A clean candidate is an
ordinary **persistent 2048×64 kernel** with two dynamic-claim stage-2 warps per
CTA, a device-scope release/acquire completion boundary, then one dynamic-claim
stage-3 CTA consumer. Reuse the current 256-byte counter allocation by adding
`nt_next`, `nt_done`, and `tile_next`; keep both existing global queues. This
preserves up to 4096 logical stage-2 warps, 2048 stage-3 CTAs, queue BUILD/REUSE,
and the multi-stream A-buffer architecture. Cooperative launch is a useful
reference but a poor default because full-residency grids can interfere with
the measured 8-stream episode win; dynamic parallelism is rejected.

The final trace contains 113,443 stage2/stage3 pairs per 15-second window. One
launch and ~2.2 µs median gap per chain offers only **~0.3–0.6 ms/frame** gross;
per-item scheduler atomics, register growth, or >2.2 µs extra work erase it.
The huge-tile queue cannot disappear safely: independent CTA consumers require
its 16 MB / 16-byte-per-item write+read handoff. Therefore this becomes a
measured opt-in A/B after iteration 4, not a speculative default. Full design:
`/tmp/perf16/rast-persistent-design.md`.

## Iteration 4 — result: rejected and fully reverted

Fresh trace measured 120,894 exact `cp_vertex_fetch → main` chains in 15 s:
fetch 1.358 ms/frame, generated VS 0.699 ms, and a CUPTI-inflated traced gap
of 1.597 ms (~193 pairs/frame). The opportunity was structurally real. The
capture is 82% direct indexed, 57% instanced and 39% batched, so a hard-coded
format fast path would not be an acceptable implementation.

Two source-sharing forms failed for different architectural reasons:

1. A lane-local result block preserved one generic fetch implementation and
   eliminated global scratch, but LLVM NVPTX local pointers crossed into the
   NVRTC helper as raw addresses rather than correctly converted generic
   pointers. Compute Sanitizer found invalid local reads. This ABI is invalid.
2. The established global-buffer ABI was correct and passed 43/43, but linking
   the generic fetch graph raised a trivial VS from **20→108 registers** and
   **6→2 blocks/SM**. The old capture's partial median became **48.05 ms**
   versus 23.42–23.67. `__noinline__` did not isolate allocation. Passing a
   null runtime helper still carried 108 registers, so it was not an honest
   revert.

**Why/retry condition:** fusion needs resource isolation first: separate
classic and fused VS binaries; an address-space-correct lane-result ABI built
in one NVPTX model; register/occupancy-tier admission per real shader. If CUDA
12.8 device calls still force union allocation, generate a fetch prologue from
a declarative format ABI or use a persistent two-phase vertex pipeline. Do not
copy static format cases into LLVM. The rejected patch is
`/tmp/perf16/iter4-rejected.patch`; the final source tree is clean.

**Interference:** none because everything was reverted. The lesson applies to
future helper fusion: a null call path is not resource isolation, and every
revert control must select a binary without the helper call graph.

## CUDA Graph architecture note (analysis after iteration 4)

The final iteration-3 trace (622 ordinary frame intervals) measures a 23.608 ms
median, 17.903 ms union of kernels/copies/memsets, and 5.679 ms with no device
operation. Global idle gaps under 20 µs total 3.287 ms/frame; runtime activity
is ~1329 kernels, 909 HtoD calls and 840 memsets/frame. CUPTI intercepts the
very enqueue calls being measured, so these are ceilings, not predicted
untraced wins. Perfect deletion of every traced idle interval still leaves
**17.903 ms**: graphs cannot reach 16 ms without ≥1.9 ms device-work savings.

Clean progression:

1. Explicit, update-free three-node raster-tail graph cached by exact parameter
   and allocation epoch (mechanism A/B; only 0.5–1 ms expected).
2. Extend the record-time command-buffer plan into immutable batch/episode
   recipes plus a command-owned device argument blob.
3. Lazy-resolve update-free whole-direct-batch and blended-segment graph execs
   per scratch/allocation generation, with retained shader modules and retired
   execution slots. Expected untraced opportunity: **1–3 ms**.
4. Keep the existing host episode drain initially; only then move bounded
   overflow/no-work/group/peel decisions into CUDA 12.8 IF/SWITCH/WHILE nodes.

No-go designs: whole-renderer stream capture, per-kernel graphs, per-submit
node parameter updates, stale rotating scratch pointers, or a full-scope graph
before decisions are device-resident. Graphs are enabling architecture after a
device-work reduction, not iteration 5 by themselves. Full report:
`/tmp/perf16/graph-design.md`.

## Fragment-stage architecture note and iteration 5

Current-tip trace splits generated fragment work at **8.157 ms/frame**:
direct 5.086 and A-buffer 3.071. Register-time ownership is 236 regs at 3.891
ms, 126 regs at 2.241, and 127 regs at 1.517. The direct grid-4096 class alone
is 4.838 ms/frame. Software input/output traffic is cache-hot (NCU DRAM only
1–9%), so deleting bytes without resource isolation is not a multi-ms design.

**Iteration 5:** retain a truly bare classic FS binary with no reference to
`cp_fs.cu` beside the current helper-linked fused binary. A null helper argument
is not a revert: iteration 4 proved linked call graphs retain union register
allocation. A/B the *whole chain*: classic interpolator + bare FS versus slim
compaction + fused FS, including sampler variants. The separate interpolator
costs ~1.1 ms/frame; direct 236+127-reg groups alone cost 4.11 ms, so recovering
an occupancy tier offers a plausible **1–3 ms net**. The classic/fused selection
must choose distinct modules, and module lifetime/cache keys/tuners must own the
mode explicitly. This is also the resource-admission foundation for hardware
texture binaries and an honest revert control.

**Later hardware texture path:** CUDA PITCH2D objects cover 50.4% of descriptor
image updates with per-mip objects (34.7% single-mip). Cube 19.7%, 3D 15.3%,
BC1/3 12.8%, and packed 1.7% need CUDA arrays or software fallback. A durable
design aligns row pitches and every mip base, caches objects per
(view,sampler,mip), defers destruction, and emits NVVM texture intrinsics
directly so a linked wrapper does not recreate the register-union problem.
PITCH2D ceiling: 0.8–2.5 ms; full array/mip/BC/cube/3D path: 2–5 ms. CUDA
bilinear uses 8 fractional bits, so validation is against Vulkan tolerances and
external reference sentinels, not native byte identity. Full analysis:
`/tmp/perf16/fs-design.md`.

## Iteration 5 — result: rejected and fully reverted

The dual-execution prototype proved honest resource isolation: `cpvk_tri` bare
FS dropped **190→18 registers** and 1→6 blocks/SM. It also proved the capture's
bottleneck is elsewhere. For **64 of 65** old-capture FS shaders, classic and
fused had identical uncapped facts: 203 or 236 registers, identical spills,
one block/SM. Their shared software-sampler graph fixes the allocation.

Interleaved full replays: fused **23.652/23.531 ms**, bare classic
**24.764/24.730 ms** — classic lost **1.156 ms (+4.9%)**, almost exactly the
restored interpolator cost. Specialisation stayed 28.8%, so this is not a
cache/admission miss. The prototype was reverted; full keep gates correctly
stopped. Patch: `/tmp/perf16/iter5-rejected.patch`.

**Architectural consequence:** capability-tagged execution objects remain the
right ownership model, but a profitable split must first remove `cp_sampler.cu`
from textured binaries while retaining fused interpolation. Module isolation
then becomes the admission mechanism for hardware-texture, software-texture,
and future writeback choices, each with independent resources/tuning.

## Iteration 6 — (next) admitted CUDA hardware-texture execution

For compatible 2D sampled-image/sampler pairs, cache CUDA PITCH2D texture
objects per (view, sampler, mip), store an owned handle-array pointer in the
descriptor ABI's existing tail space, and emit NVVM/PTX texture operations in
the generated shader so `cp_sampler.cu` is absent from that execution. Preserve
software sampling as the complete fallback for cube, 3D, BC, packed formats,
misalignment, unsupported addressing/LOD, separate-descriptor combinations,
and any precision mismatch. Measure actual *sample execution* coverage before
enabling; descriptor-update coverage alone is not enough. Hardware linear
weights use Vulkan-compatible 8 fractional bits, but external reference tests
and Vulkan tolerances—not native byte identity—are the correctness authority.

## Iteration 6 — result: rejected and fully reverted

The direct NVVM hardware-texture execution was correct and honestly isolated:
`tex.2d` in registers, no `cp_sampler.cu`, aligned PITCH2D storage, retained
(view,sampler) objects, descriptor-copy/lifetime handling, and focused nearest/
linear/repeat/clamp tests. Runtime coverage was only **21,991/156,159 FS
launches (14.1%)**. Uncapped eligible modules moved 203→190 registers but stayed
at one block/SM; full replay hardware center **23.724** vs software **23.504 ms**
(+0.220).

The required capped follow-up reached **126 registers / 2 blocks/SM** for all
28 eligible modules, with 104–136 bytes local/spill. Interleaved pairs canceled:
−0.083 then +0.089 ms; centers hardware **23.608**, software **23.605 ms**
(+0.003, +0.014%). The 14.1% subset is too small and spills consume its gain.
Both prototypes were reverted. Patches: `/tmp/perf16/iter6-rejected.patch` and
`iter6-capped-followup.patch`.

**Retry condition:** primary CUDA mipmapped-array representation or coherent
array backing for multi-mip, cube, 3D and BC resources, plus interpolation in
the same LLVM address-space model so hardware shaders do not retain the
190-register helper floor. Only then does capability-tagged hardware execution
cover enough of the 8.16-ms FS class. Graph nodes must key on execution and
texture-object generation.

## Iteration 7 — (next) occupancy-sized persistent FS grid

The direct grid-4096 class is **4.838 ms/frame, 95.1% of direct FS**. Every
launch starts 1,048,576 physical lanes, while the exact compact count remains
on device and the shader already strides virtual block IDs. Instrument counts
without a host sync, then A/B 1–8 resident waves based on SM count and the
selected execution's blocks/SM. Keep **4096 as the hard upper cap and fallback**.
The design changes scheduling only: ABI, order, shader function, and useful
invocations remain identical. Admission must account for sampler-variant and
register-cap execution changes; a fixed magic block count is only a probe.

## Iteration 7 — result: rejected and fully reverted

Stats recorded 156,154 FS launches with zero drops. Direct grid4096 exact count
was p50 **3,108**, p90 68,972, p95 231,868, p99 923,520, mean 46,196: only
**4.406%** of first-wave lanes useful (base 2.417%, sampler variants 12.190%).
A-buffer sizing was 99.96% useful. Despite the striking waste, unprofiled grids
340/680/1024/1360/2048 measured 23.645/23.558/23.610/23.632/23.724 ms against
4096 controls 23.629/23.596/23.544. Best delta −0.16% is below control spread.
Idle blocks retire cheaply; persistent striding does not reduce useful work and
can reduce tail parallelism. No default was justified; patch reverted at
`/tmp/perf16/iter7-rejected.patch`. The 4096 cap remains.

## Iteration 8 — compact interpolation before a bare hardware FS

Hardware texture PTX alone needs 42 registers; linking `cp_fs.cu` raises it to
190, and capping reaches 126 only by spilling 104–136 bytes. Avoid that union
without restoring iteration 5's full-frame interpolator: extend the existing
`cp_fs_compact` launch to perform the same per-quad interpolation and write the
established `fs_in` ABI for only its compacted slots, then launch a truly bare
hardware FS. The direct chain stays two launches. A-buffer uses its existing
exact-quad standalone interpolator when choosing a bare execution. Software
sampling remains current/default unless whole-chain A/B says otherwise. This
execution boundary is also the clean prerequisite for broad CUDA-array texture
coverage and graph nodes with settled functions.

## Iteration 8 — result: rejected and fully reverted

The mechanism reached its resource goal: 28 eligible hardware modules at 34–79
registers, 3–6 blocks/SM, no helper/sampler references; focused case 38 regs,
zero spill, six blocks. Direct admission was only **13.6%**. Full replay center
was hardware **23.885** vs software **23.818 ms** (+0.067, neutral/slower).

A bounded matched-chain trace proves the operation itself works:

- software FS 25.206 → bare hardware FS 2.688 µs;
- slim compact 6.410 → compact+interpolate 13.224 µs;
- whole admitted device chain **33.348 → 17.648 µs** (−15.700 mean, −5.344
  paired median).

Pre-interpolation consumes ~30% of the mean FS saving; low coverage, not texture
instructions, rejects the frame result. Break-even is 18.7–28.8% direct
coverage. Adding multi-mip pure-2D draws predicts ~22%, straddling break-even.
Even an unrealistic all-direct hardware upper bound is only 0.31–1.05 ms net,
so coherent CUDA mipmapped arrays are not justified now. Reports:
`/tmp/perf16/iter8-report.md`, `iter8-decomp.md`; patch reverted.

## Current device-work ranking and iteration 9

Final trace (625 frame markers): generated `main` 8.944 ms/frame; raster chain
**7.725 ms** (fused clip+s1 direct 1.762, abuf 1.541; stage2 total 1.172;
stage3 total 3.252); vertex fetch 1.374; remaining A-buffer resolve/sort/scan
under 1.5. The next multi-ms target is raster work, not enqueue geometry.

Every fused clip+s1 thread carries the polygon clipper's ~5,248-byte stack even
though wholly-inside triangles take the fast branch. Iteration 9 first measures
inside/outside/crossing rates. If crossings are sparse, a lightweight
accept+copy+stage1 kernel handles inside triangles and appends only crossing
input IDs to a clip worklist; a second heavy clip+stage1 kernel consumes that
list before unchanged machine-wide stage2/3. Both use the same `cp_clip_one`/
`cp_rast_small_or_defer` bodies and exact output/active/stable-ID rules. This
adds a launch but removes the 5-KiB stack from the common grid and is the
long-term work-distribution pattern needed by more complete clipping.

## Iteration 9 — result: rejected and fully reverted

No-sync census: 401,856,992 input triangles, **71.898% inside**, **26.886%
common-plane reject**, **1.217% crossing**; samples were 99.795% trivial. The
split achieved lightweight 47–48 regs/96-B local versus heavy 56–58 regs/
5,248-B, and clip hashes were exact. It nevertheless lost: accept→crossing
gaps 1.92/2.53 µs, device sums direct 30.992→41.021 and A-buffer
12.401→14.804 µs; old center **23.592→24.443 ms (+0.850, +3.6%)**. The current
fast branch already avoids touching polygon arrays; a worklist, second launch,
and duplicate classification only add work. Patch reverted:
`/tmp/perf16/iter9-rejected.patch`. A later persistent lightweight/heavy worker
scheduler must use epochs/device queues and integrate stage2 to pay for itself.

## Iteration 10 — zero-copy primitive references

The useful census suggests a different clip redesign: do not copy the 71.9%
wholly-inside outputs at all. Allocate an 8-byte primitive-reference table:
inside IDs point directly at their original VS triangle, true crossing outputs
point at clipped scratch, rejected stable slots are null. Raster setup and
fragment interpolation resolve one pointer per primitive; all later math and
ID/queue rules remain unchanged. Only 1.2% crossings copy 3×slots×16 bytes.
This keeps one fused launch, removes large accepted-output writes and stable
retirement writes, and makes the clipped representation explicit for future
layered/MRT work. A/B must include the pointer-read cost and retain classic
contiguous storage as fallback.

## Iteration 10 — result: kept (commit pending review)

Exact slot census found accepted clipping copied **109.96 GB/replay**,
72.824 MB/frame, averaging 7.929 vec4 slots per vertex. Primitive references
publish 1.570 MB/frame; true crossings retain 1.864 MB/frame of copies. Net
accepted writes removed: **~71.253 MB/frame**.

The optional 8-byte table owns one base pointer per fixed/compact primitive ID.
Inside outputs point at immutable VS storage; crossing outputs publish clipped
scratch only after all vertices are written; retired stable IDs are null. One
CUDA-visible resolver feeds raster setup and interpolation. Active IDs, counts,
stable ordering, queue BUILD/REUSE, episode ranges and tiled replay remain IDs,
not pointers. Scratch refusal or `CUDAPIPE_NO_PRIM_REFS=1` selects the exact
contiguous representation. Current `CP_CLIP_MAX_OUT=8`, so stable no-active
retirement nulls all 7/8 unused fixed entries rather than relying on an old
four-slot assumption.

**Measured:** old **23.682→23.255 ms** (−0.427, −1.80%), walls agree; direct
clip+s1 −14.4%, A-buffer clip+s1 −12.1%, downstream kernels neutral. Sweep hot
sum **28.42→25.99 ms** (−8.6%), wall 39.18→38.09 s; multithreading −29%,
instancing −17%; Crossroads neutral at 7.427 ms.

**Gates:** child and root build/docs/diff clean; root 43/43 `-j8`; six
reproducers byte-identical and unchanged across five batching modes plus refs
revert; Compute Sanitizer zero; both sentinel discriminator sets exact; NVIDIA
60 calibrated to the standing glTF orbit residual (llvmpipe worse); Gallium
exact.

**Interference:** adds 16.24 MiB/frame logical worst-capacity scratch request
and two clip-kernel registers without changing occupancy. Allocation refusal is
exact fallback. Existing clipped worst-case storage remains for crossings;
removing it requires a second crossing allocator without remapping stable IDs.
Stage2/3 queues, graphs and persistent-raster designs are unblocked. Layered/MRT
can reuse the explicit primitive-base representation.
