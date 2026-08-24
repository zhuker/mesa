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

## Iteration 11 — cache huge-primitive setup across stage3 tiles

Stage3 remains ~3.3 ms/frame and invokes `setup_triangle()` once per 64×64
tile, repeating position/reference loads, perspective divides, cull/bounds and
edge setup that stage2 already performed once before enumerating those tiles.
Measure huge primitives and tiles-per-setup first. If reuse is material, retain a
bounded setup record per huge primitive in the queue generation and encode its
index in the existing 32-bit tile `tri_id` word (the high bit is already outside
the driver's <2^30 primitive range). Stage3 loads the cached setup; capacity
overflow keeps the old tri ID and recomputes. Queue BUILD/REUSE and eight side
streams own separate generations. This preserves independent machine-wide
stage3 CTAs and adds no host decision/readback.

In parallel, design an exact subpixel/fixed-edge representation shared by stages
1–3. The goal is not a fast approximate rasterizer: snapping, top-left bias,
MSAA positions, cull/bbox and barycentrics must be one owned definition that
enables drift-free incremental edges and valid tile accept/reject. That is the
larger multi-ms raster architecture if setup caching is bounded.

## Exact fixed-edge raster roadmap (analysis during iteration 11)

A fixed rasterizer is viable only as one shared all-stage definition, not a
stage-local optimization. Use signed int32 window coordinates snapped to eight
fractional bits; signed int64 unbiased/biased edge planes; exact integer MSAA
offsets; one top-left bias definition; and a canonical affine depth/barycentric
helper consumed by rasterization and FS interpolation/front-facing. Build a
direct fixed oracle first, then stage1 stepping, stage2 stepping, and stage3
integer min/max accept/reject behind modes whose zero value compiles the
byte-for-byte float path. Never mix fixed stage1 with float later stages: that
changes shared-edge ownership and can create cracks.

The expected gain is **0.5–1.1 ms (7–15% of the ~7.4-ms raster chain)**, not a
16-ms solution alone, because stage3's winning pixels still issue atomicMin.
No-go forms: 32-bit edge values, accumulated float stepping, corner accept
without top-left bias and all sample offsets, or a partial production rollout.
The migration improves conformance and enables exact tile acceptance, so it
follows setup caching. Full design: `/tmp/perf16/fixed-edge-design.md`.

## Iteration 11 — result: kept (commit pending review)

Census shows strong setup reuse: old direct 1.86M setups built 15.74M tiles
plus 23.23M REUSE executions; A-buffer 3.26M setups built 28.62M tiles. Median
tiles/setup 4 direct, 6 A-buffer; max 240. A 1024-entry queue-local cache
covers the measured maximum 776. Publishing only for ≥4 tiles retains 91.9%
direct and 96.8% A-buffer tile records while avoiding known-poor reuse.

Stage2 publishes an 84-byte immutable setup and tags its index in the existing
8-byte tile record; overflow stays an untagged primitive ID and recomputes.
Stage3 restores the original ID and setup once per tile. BUILD/FILL reset the
third queue counter; REUSE retains cache and tagged tiles together. Main plus
eight side queues cost **756 KiB**. Allocation failure and
`CUDAPIPE_NO_SETUP_CACHE=1` are exact classic paths.

**Measured:** stage3 direct −2.45%, A-buffer −2.35%; stage2+3 chain −0.254/
−0.273 µs per launch. Old **23.295→23.217 ms** (−0.077, −0.33%) in two
winning interleaved pairs. Crossroads neutral. Warm sample hot sum excluding
nondeterministic glTF 17.16→17.12; walls 37.69→37.38 s. This is deliberately
recorded as a small redundant-work removal, not a multi-ms foundation.

**Gates:** worker/root build/docs/diff; root 43/43 and unchanged six-mode
hashes; huge triangle, huge point, forced capacity overflow, sanitizer, stored
frames, NVIDIA calibration and both sentinels pass; Gallium exact.

## Iteration 12 — explicit update-free raster-tail CUDA Graph

The graph trace ceiling is ~3.29 ms/frame of sub-20-µs device-idle gaps; the
raster tail's two handoffs contribute ~0.81 ms/frame traced. Start with a
mechanism whose ownership is exact: explicit graph nodes for fused clip+s1 →
stage2 → stage3 (plain and A-buffer specializations), cached only when concrete
CUfunctions, by-value argument bytes, queue set and allocation/scratch epoch all
match. Hits perform one `cuGraphLaunch` and no node updates. Misses build lazily
or run classic; instrumentation/revert/error paths stay classic. Modules are
context-lifetime kernels; graph execs retire with the renderer generation.

Measure exact-key reuse before code. Do not use whole-stream capture, stale
rotating pointers, host/event nodes, or per-submit `SetParams` calls that erase
the benefit. The tail graph is expected at most 0.5–1 ms and is kept only with
unprofiled evidence. Its real value is proving the cache/epoch/fallback model
needed to expand the record-time command plan into update-free whole-batch and
episode-segment launch DAGs (1–3 ms opportunity).

## Iteration 12 — result: measure-first rejection, source unchanged

Complete old census: 318,454 candidate three-kernel tails but 65,905 exact
argument/function keys. Reuse-distance p50/p90/p99/max 155/750/25,343/50,857.
Bounded LRU hit rates are only 69.2–76.5% at 256–4096 entries; even an
optimistic two-touch ghost policy needs 27,563–43,944 graph instantiations.
CUDA 12.8 microbench: 14.8–15.1 µs instantiate and 44–46 KiB retained per
graph+exec, giving 0.27–0.43 ms/frame build cost and 11–183 MiB caches before
138–149 graph launches/frame. Exact-hit gap ceiling is just 0.63–0.68 ms/frame.
Per-tail graphs cannot repay churn; no source survives. Report:
`/tmp/perf16/iter12-report.md`.

Path detail reinforces the architecture: plain fused tails recur well (90.3%
at 256), but REUSE is 0.87% and A-buffer 68.5%. Whole-batch/segment graphs need
record-time recipes plus stable command-owned execution storage, not a larger
context LRU over rotating scratch pointers. Tail rejection does not invalidate
the 1–3-ms whole-DAG opportunity.

## Iteration 13 — exact fixed-edge all-stage raster migration

Implement the correctness oracle from `fixed-edge-design.md`: signed int32
window coordinates snapped to eight fractional bits, signed int64 edge planes
and top-left bias, exact integer sample offsets, and one affine depth/
barycentric contract shared with FS interpolation. Production cannot mix fixed
and float stages. Staged modes validate direct fixed coverage first, then exact
stage1/2 stepping and stage3 min/max classify; mode zero compiles the legacy
path. The goal is drift-free coverage and conformance that also removes edge
recomputation and enables valid tile acceptance, estimated 0.5–1.1 ms.

## Iteration 13 — result: production rejected; oracle retained

The all-stage implementation was exact on its focused contract: Q8/int64 modes
0–4 passed the shared-edge CPU oracle, all forced stage/path variants, identical
`cpvk_tri`, and mode0/mode4 native 44/44. It nevertheless failed both paired
performance controls. Mode4 versus legacy regressed **+0.827/+0.653 ms**
(center **+0.740 ms/frame**); mode4 versus direct fixed mode1 regressed
**+0.231/+0.220 ms** (center **+0.225 ms**). Int64 issue/register cost and the
recurrence/classifier cost exceed the removed float evaluations. Production
fixed code and flag are fully restored; rejected patch:
`/tmp/perf16/iter13-fixed-rejected.patch`.

Retain `cpvk_fixed_edge`: two additive triangles form a 176×176 quad whose
shared diagonal crosses pixel centers and nine 64×64 tiles. Its independent
Q8 RN-even/int64/top-left CPU oracle detects cracks and double ownership, plus
a one-pixel on-edge scissor. Legacy passes exactly. The current oracle does not
yet cover MSAA, cull/front-face permutations, clipped fan seams or depth ties;
the requested shadow scene census was not completed and no claims depend on it.
Full report: `/tmp/perf16/iter13-report.md`.

## Iteration 14 — same-LLVM fragment interpolation and shader

The generated fragment class remains roughly 9 ms/frame. Its current fused
form links NVRTC interpolation PTX into the NIR-generated LLVM/PTX shader. A
hardware-texture-only shader was 34–79 registers, but linking that helper raised
it to 190; software-sampler builds remain 126–236 registers. Nulling the helper
does not isolate its call graph, while a separate interpolator pays a launch and
global `fs_in` round trip.

Build the interpolation helper as LLVM 18 NVPTX bitcode from one owned source,
link it into the generated NIR module before optimization, force inline it, and
target a kernel-local input array/SSA values rather than global `fs_in`. Preserve
quad/helper semantics, direct and A-buffer segment resolution, batch rows,
points, perspective/front-facing/FragCoord and shader-memory helper masks. The
classic separate interpolate → bare shader chain is the exact fallback and
resource-isolated control. Do not pass an LLVM-local pointer across an NVRTC
boundary (iteration 4's invalid ABI).

Admit only if an LLVM-toolchain proof shows stable bitcode parsing/linking and
the focused module removes the external interpolation symbol. Compare complete
compact/interpolate → FS → writeback chains, resource counts and old replay,
not FS registers alone. Even a modest direct win is architectural: the same
module permits dead varying elimination and is prerequisite to combining the
proven-fast bare hardware texture instruction with broad image coverage without
the 190-register helper floor.

## Iteration 14 — result: architecture kept opt-in, fused remains default

The mechanism is real: a matching build-time Clang emits deterministic NVPTX
LLVM bitcode from the shared interpolation source; the generated NIR module
links it, forces it inline and promotes its live input slots to SSA. Strict
admission rejects any surviving helper or PTX local-memory traffic. Admitted
PTX has no interpolation call, no local depot and no global `fs_in` route. It
admits 38/65 old-capture FS modules; the remaining 27 use proven fused PTX.
Untextured resource use falls fused 190→inline 72 registers. Software-sampled
modules stay at 203/236 because the sampler is now the sole resource floor.

Complete focused chains improve **25.3% direct** and **16.4% A-buffer**. An
opt-in 600-frame hot sum was 25.39 versus forced fused 26.13 ms, but old replay
23.240/23.231 is neutral against the standing fused 23.217 ms. More importantly,
opt-in dual ownership costs **+8.34 s cold/warm replay wall and +110.6 MiB host
RSS** across 65 shaders (+~20 MiB device). Therefore the clean adoption policy
is: fused-only remains default; `CUDAPIPE_INLINE_FS=1` builds fused+inline for
experiments and the forthcoming resource-isolated hardware texture path;
`FORCE_FUSED_FS` is the reproducible fused control; `NO_INLINE_FS` is the
resource-isolated classic control. No default frame-time claim is made.

Review hardened constant/dynamic input-array footprints, frozen bitcode ABI,
convergent initialized quad shuffles, transactional sampler variants, quiet
expected fallback, standalone-capable pipeline admission and pristine NIR on
fused JIT failure. Inline requires LLVM ≥18 plus matching optional Clang; without
it all parse/link/pass references compile out. Exact-version static LLVM module
deps link successfully (the distro's unrelated missing Polly archives were
removed only from the `/tmp` probe link), and that ICD runs `cpvk_tri`.

**Gates:** worker/root 44/44; inline focus and input array/matrix/indirect
probes; sanitizer; six reproducers over batching modes and default/inline/force/
classic precedence; stored60/NVIDIA/sentinels; docs/diff; Gallium exact. Root
mode hashes remain unchanged. Report: `/tmp/perf16/iter14-report.md`.

## Iteration 15 — same-LLVM hardware texture execution

Iterations 6/8 proved the hardware operation itself: admitted bare `tex.2d`
chains saved 15.7 µs mean / 5.34 µs median, but only 13–14% of launches hit and
the separately linked interpolation helper imposed 190 registers (or 126 plus
104-byte spills). Iteration 14 removes that second blocker. Rebuild the
resource-isolated 2D single-mip PITCH2D execution with the hardware intrinsic
and same-LLVM interpolation in one module. The software fused module remains
complete fallback when any shader instruction, descriptor row, image layout or
sampler state is unsupported.

This phase deliberately reuses the proven narrow image/object lifetime model.
It must show a two-block/no-spill resource class and a complete-chain/old replay
win before expanding storage. If admitted, its descriptor-handle ABI and
execution ownership become the base for coherent mipmapped/layered CUDA arrays,
cube/3D/BC views and A-buffer coverage. If still neutral, broad arrays remain
economically unjustified despite working hardware sampling. Default software
behavior and `CUDAPIPE_INLINE_FS` remain unchanged until full gates.

## Iteration 15 — result: compound mechanism works, coverage rejection

Same-LLVM hardware execution reaches **88 registers, zero spill, two blocks/SM**
versus software fused 203/one block. PTX contains `tex.2d` and no software
sampler, interpolation helper or local traffic. Focused direct chains improve
18 vs 26–32 µs and A-buffer shade+composite 20 vs 30 µs. But old coverage is
21,929/157,234 FS launches (**13.9%**): 13.4% direct, 16.0% A-buffer. Paired
old results HW 23.390/23.342 vs software 23.318/23.353 ms, center **+0.031 ms**
with opposite pair signs. Production and padded image layouts are fully restored.
Patch/report: `/tmp/perf16/iter15-rejected.patch`, `iter15-report.md`. Coherent
arrays still need launch-weighted direct coverage well above 29%; do not fund
them from this neutral result.

## Iteration 16 — same-LLVM specialized software sampler

The compiler boundary is now the dominant fragment blocker. Hardware proves a
shader without `cp_sampler.cu` can run at 88 registers, while software remains
203/236 even with same-LLVM interpolation. Sampler variants cover most hot
launches but still link an opaque PTX call graph, so literal state cannot remove
its unused wrap/filter/LOD/format paths from the generated shader allocation.

First build a focused proof: an owned freestanding software-sampler core linked
as LLVM bitcode into a retained pristine NIR clone, with literal sampler state
and instruction target embedded before optimization. Require the call and local
objects to disappear, exact software results, and a real occupancy/instruction
move on representative old shaders. Only then design retained-NIR lifetime and
lazy per-state variants. The prize is broad coverage of the ~9-ms FS class and
a clean path to specialize image encoding/dimension later; a generic inlined
megafunction that stays 203/236 is rejected before integration.

## Iteration 16 — result: whole-launch specialization rejected before source

The LLVM sampler proof is strong: same-LLVM interpolation plus literal RGBA8
sampling compiles to **54 registers** (BC3 56), zero spill/local/helper, versus
203/236 today. Implicit RGBA8 is bit-exact over 131,072 float words. Explicit
LOD still differs in 122 words and cannot ship. But a timed 5,437-launch census
shows current variants cover only 15.2% FS device time; a richer ordered
per-site sampler+encoding key reaches only **25.85%**, below the 60% gate. No
production source changed. Report: `/tmp/perf16/iter16-report.md`.

Per-row analysis exposes the architectural opportunity: broad literal keys can
cover **58.00% total FS time and 94.33% A-buffer FS time**, including 42.80% of
the expensive 236-register class. Whole-launch agreement hid that because one
batched shader launch contains several material rows. The narrow proven core
would cover only 32.02%; broad cube/3D/BC/packed semantics and exact explicit
LOD remain mandatory.

## Iteration 17 — bounded coherent row-key sampler dispatch

Measure exact row keys and live shaded slots first. If the dominant mixed-row
launches truly have at most two keys, compile one generated FS execution with
two literal per-site sampler paths and a quad-coherent row-key-index branch at
each texture operation. LLVM can allocate mutually exclusive branches at their
maximum live set rather than linking the opaque generic sampler. The module has
no generic sampler reference; every row in a launch must map to one admitted
key or the entire launch uses fused software. Geometry/A-buffer batching, slot
identity and launch count remain unchanged.

This is preferred over fragment rebucketing as the first mechanism because it
adds no scan/list/launch work. Cap at two until resource/code-size/compile-time
and exact-key census justify more. The retained-NIR/key cache remains bounded
and opt-in. Exact current sampler results, including explicit LOD and all target/
encoding classes admitted, are non-negotiable.

## Iteration 17 — result: exact coverage/resource pass, numerical rejection

The complete old census observes 157,229 FS launches, 21.475 billion live slots
and **zero row disagreement in 5.369 billion quads**. Two exact live row keys
cover 91.38% of slots (97.06% A-buffer). A fresh timed window gives 57.57% total
FS time and **98.25% A-buffer FS time**; four cached key sets/shader retain
93.38% A-buffer time. A mixed RGBA8-nearest/BC3-linear two-key module is **64
registers, zero spill/local, four blocks/SM**, with valid four-lane derivative
shuffle masks.

Production is nevertheless rejected: the literal core differs from current
software in 122/131,072 explicit-LOD float words. Every mismatch is positive
fractional/trilinear; integer levels are exact. Fixed 4.81 output proves Clang
uses the expected RN FMA of independently exact mip results, while NVRTC's
in-context optimizer produces another operation tree. No tolerance is accepted
and no production source changed. Report: `/tmp/perf16/iter17-report.md`.

## Iteration 18 — canonical mip blend, then A-buffer-first row sampler

First define the sampler's fractional mip blend explicitly in the one shared
semantic source using a round-to-nearest FMA, and force the existing NVRTC path
to the same operation. Require zero implicit/explicit/bias vector differences,
then run reference/stored gates for the tiny canonical output change relative to
the old optimizer. This is a numeric contract, not an approximation.

Before renderer work, census **all host rows admitted to each launch**, not only
historically live slots. A sampler-only module has no generic device arm, so a
third possible host key must reject the entire specialized launch unless a real
fallback/partition exists. If cap two still owns the A-buffer opportunity, build
only A-buffer SW-inline variants initially: bounded two-key modules and four
sets/shader, exact row-key map, direct path unchanged fused. This targets the
93–98% A-buffer FS class with no geometry split or extra launch.

## Iteration 18 — result: exact broad sampler rejected by hot execution

Canonical mip math, all-host cap-two admission, 25-class exact shared core,
retained NIR/cache ownership and full correctness all pass. Real variants admit
up to eight/nine texture sites at 92–104 registers, zero spill; cap-two/four-set
coverage owns 93.38% A-buffer FS time. Yet old replay regresses fused 23.434 to
inline 24.605 ms (**+5.00%**), after-first-100 +5.29%, standing policy +6.64%,
and wall +13.77%. The larger duplicated instruction/code footprint dominates
the occupancy gain. Production is fully restored. Report/patch:
`/tmp/perf16/iter18-report.md`, `iter18-rejected-production.patch`. Do not retry
same-LLVM software sampling without an instruction-count redesign.

## Iteration 19 — stable-recipe whole-batch CUDA Graphs

Tail graphs were rejected because rotating transient pointers produced 65,905
exact keys and 27–44k instantiations for only 0.6–0.7 ms ceiling. The trace still
contains ~5.7 ms/frame of device-idle gaps, and an immutable direct batch spans
fetch → VS → clip/raster → compact/interpolate → FS → writeback with many launch
handoffs. The graph unit must move up to that batch/decision-free segment.

Measure pointer-independent record-time recipe recurrence and command-buffer
reuse first. If admitted, give a recipe bounded command-owned execution storage
with stable device addresses and owned host argument bytes; build one explicit,
update-free graph per allocation/function epoch. Graph hits contain kernel,
memset and required memcpy nodes and launch once. No stream capture, per-submit
node updates, stale arena pointers or host-dependent A-buffer/peel branches. A
miss/failure and unsupported batch use the exact classic path. This is the clean
foundation for later conditional episode graphs and should target 1–3 ms, not
the already-rejected two-handoff tail ceiling.

## Iteration 19 — result: direct graph has zero old-capture coverage

Full replay: 4,532 command-buffer submits reuse seven mutable native command
objects; 307,811 batches and 496,497 episode attempts. There are 12,999
non-appending unblended calls, but **zero strict whole-direct candidates**: 9,250
are strips, 2,944 points, 664 triangle-list MSAA and 141 other exceptional
cases. Normal safe triangle-list opaque work is routed into opaque episodes.
Samples expose only two chains/frame with <0.5-ms absolute and ~0.03–0.06-ms
normal API ceiling, plus >=446 MiB copied scratch if command-owned. No graph
source survives. Report: `/tmp/perf16/iter19-report.md`.

The Driver mechanism is not the blocker: 12-node instantiate median is 5.8 µs;
args copy at node add; same-exec launches order safely on the one stream. The
unit was wrong. Do not retry direct batch caching.

## Iteration 20 — completed opaque-episode graph

Opaque episodes are the normal decision-free workload. Measure completed episode
frequency, segment/group/node shape, exact resource/function recipe recurrence,
small-gap/device time and scratch high-water. If admitted, defer the episode's
segment operations rather than enqueueing them during append; at finish resolve
one immutable operation list containing geometry, visibility and grouped shading,
then send it to either classic or explicit-graph sink. A graph hit launches the
whole episode DAG once.

The episode owns stable argument/upload bytes; large deterministic draw scratch
remains shared at fixed base+offset under allocation epochs and the one serialized
stream, not copied per graph. External resource contents remain dynamic at baked
addresses. Tuner/module/framebuffer/scratch epochs invalidate. No A-buffer, peel,
overflow, side-stream, query, debug or host decision enters this iteration.
Classic replay is complete fallback before launch; graph-launch failure is device
loss, never partial replay. Target the 1–3-ms whole-DAG opportunity.

## Iteration 20 — result: opaque graph rejected by reuse and ceiling

Full old: 5,902 opaque episodes, 73,280 segments, 166,694 draws and 1.314M
joined device operations. All internal opaque gaps sum to 1.373 ms/frame. Even
an optimistic semantic LRU gives 0.900-ms one-touch / 0.677-ms two-touch hit
ceiling before launch/cache cost. More decisively, 4,532 submits reuse seven
command objects but have 4,532 unique plan generations and **zero repeated
episode recipes within a generation**. Command-owned graphs retired on re-record
have no old hits. Shared scratch reaches 1.03 GiB old/4.65 GiB multithreading;
256 large retained graphs measure ~1.46 GiB RSS +632 MiB device. No graph code
survives. Report: `/tmp/perf16/iter20-report.md`. This closes CUDA Graphs until
command recording or workload reuse changes materially.

## Iteration 21 — machine-scaled persistent raster stage2/3

The refreshed raster chain remains ~7.4 ms/frame; stage2+stage3 own roughly
4.5 ms. Iteration 3's clip+s1 fusion wins, but producer-local stage2/3 fusion
regressed 3.6 ms because each producer warp consumed only its own tiles and
serialized queue drain. The retained design is different: preserve a bounded
machine-wide stage2 producer population and a persistent machine-scaled stage3
consumer population in one launch, using global queue publication and an exact
done protocol. Consumers steal all tiles, not producer-local work.

Measure queue production/concurrency and choose role counts from SM/resource
facts. Consumer polling must be bounded/backed off; publication uses release/
acquire-safe CUDA atomics or kernel-visible fences; overflow and queue reuse stay
classic. Stable primitive IDs, setup cache tags, direct/A-buffer semantics and
coverage/depth atomics remain exact. Keep separate stage2/3 as fallback. The
goal is real device work/latency improvement across the 4.5-ms chain, not merely
one fewer host call. Full design starts at `/tmp/perf16/rast-persistent-design.md`.

## Iteration 21 — result: persistent raster scheduler loses the boundary saving

The exact ordinary two-phase implementation used dynamic machine-wide claims,
device-scope release/acquire publication and the current shared queues. It passed
44/44 and focused classic/persistent hashes. The merged direct/A-buffer kernels
used 56 registers, 92 bytes shared, no spill and allowed 18 blocks/SM.

That resource result did not pay. In an 8-second trace classic stage2+stage3
used 1,158.61 ms versus 1,298.76 ms merged (+140.15 ms), while complete paired
gaps offered only ~123.7 ms. Per chain, direct adds 3.894 µs before removing a
1.642-µs mean gap; A-buffer adds 1.751 µs before removing a 2.853-µs mean gap.
Old reverse controls were 23.686→23.456 then 23.595→23.603 ms: only 0.111 ms
aggregate and non-reproducing. The scheduler atomics/work assignment cost more
than the boundary. Production is restored. Report `/tmp/perf16/iter21-report.md`;
rejected patch `/tmp/perf16/iter21-rejected.patch`. Do not retry ordinary
persistent stage2/3 without eliminating dynamic per-item scheduling.

## Iteration 22 — exact out-of-line sampler / per-key execution proof

Generated shader `main` remains 11.1 ms/frame and the 4096x256 class alone is
7.71 ms/frame. Iteration 18 proved exact canonical software sampling but its
always-inline two-key modules regressed 5–7%: lower registers duplicated too much
instruction/code footprint. Phase A therefore changes the mechanism, not the
coverage claim.

In a detached tree compare current generic callable sampling, the rejected exact
inline form, a canonical exact **noinline direct-call** helper pruned by operation,
target, encoding, sRGB and sampler state, and smaller noinline semantic layers.
Deduplicate identical site/class bodies. Measure real 203/236-register hot shader
variants by whole launch, executed instructions, code bytes, call-frame local
traffic and compilation cost. The existing 25-class matrix including fractional
LOD remains a zero-tolerance gate.

Single-key launches can use one exact module without a row branch (47.64% of the
fresh timed FS window). If two keys are needed, evaluate one device quad partition
followed by two reusable per-key modules, rather than another `(shader,key-set)`
mixed inline module. Counts stay device-resident; slots are disjoint; A-buffer
composites only after both kernels. All possible host rows must resolve or the
whole launch uses the generic path. No production edit until a real whole-launch
win supports at least 1.5 ms gross opportunity.

## Iteration 22 — result: noinline exact sampler still expands real shaders

The canonical exact noinline class helper passes all 25 classes and 3,276,800
words, including fractional LOD, with zero mismatches. A trivial one-site RGBA8
shader remains 54 registers. Real multi-site variants do not isolate: 143–182
registers, 112–136 bytes caller stack/local, and only 48/83 variants retain two
blocks/SM. A real two-key/nine-site module reaches 345,675-byte PTX, 298,944-byte
cubin and 225,536-byte `main` SASS.

Every unprofiled old control loses: cap-one direct+A-buffer +1.28/+1.45 ms;
cap-two direct+A-buffer +0.70/+0.95 ms; cap-two A-buffer-only +0.97/+1.03 ms.
Quad partition itself would cost only ~0.1 ms/frame, but cap-one modules with no
partition or row branch already regress. Exact inline and exact noinline software
sampler specialization are both closed. Report `/tmp/perf16/iter22-phaseA-report.md`;
prototype `/tmp/perf16/iter22-phaseA.patch` in a detached tree. Retry requires
fewer executed sampling instructions, not another helper boundary.

## Iteration 23 — canonical CUDA mipmapped-array ownership census

Hardware texture operations are the qualitatively different sampler path. Existing
exact-row evidence gives only an ownership-agnostic upper bound: complete 2D
RGBA8/R8/BC1/BC3-class A-buffer launches cover 545.292/681.332 ms (80.03%) of
A FS time and an estimated 2.48 ms/frame gross, while direct coverage is only
9.44% (~0.75 ms). RGBA8/R8 without BC is ~1.59 ms gross. Design:
`/tmp/perf16/cuda-array-design.md`.

Before ownership code, take a fresh census joining every possible descriptor row
to stable image/view/sampler/memory IDs and a delayed Nsys FS range. Separate:
(1) array ownership eligibility (optimal, sampled/transfer-only, nonalias, supported
format/target, dedicated/nonhost memory); (2) exact view+sampler texture-object
semantics; and (3) hardware NIR site/intrinsic support. Record tiling, usage,
mips/layers, bind intervals, view format/range/swizzle, transfer/attachment/storage
history and candidate bytes. Report launch/grid/live-slot/device-time funnels by
reject reason. Live rows are only an upper bound; all possible rows decide.

Admit the A-buffer alternative only if the full ownership∩object∩site intersection
still contributes >=2.0 ms/frame gross and memory/transfer/object cost is bounded.
Direct remains subject to the historical >29% launch screen and cannot pass from
current format keys alone. If admitted, use canonical deferred CUDA mipmapped arrays
backed by dedicated VMM allocations, never a stale linear shadow; current linear
software images remain the complete fallback.

## Iteration 23 — result: canonical array ownership has no real intersection

The full ownership census resolved 157,216 FS launches, 1.152M rows and 8.733M
site rows with zero missing IDs, disagreement, unresolved or overflow. A delayed
12-second trace joined 16,255 `fsarr` ranges one-to-one to generated `main`. It
exposed 2,855.455 ms FS device time over 306 submits.

All 1,799 allocations omit explicit image dedication, so the strict ownership
intersection is zero. A separate future lazy-allocation/effective-exclusive model
leaves only 2,048/125,047 direct launches (1.638%) and 7/32,166 A launches. In the
fresh trace they expose 54.298/2,855.455 ms (1.902%), 0.177 ms/submit raw or a
labeled 0.211-ms upper on the older 11.1-ms class. A-buffer contributes 0.00083 ms.
Both >29% direct and >=2-ms gross gates fail.

The alleged dynamic descriptor class was the production specialiser's four-reference
capacity limit: all 39,167 launches have 11–16 complete static refs, no dynamic index.
A follow-up enumerated them all; zero requested-tier launches survive companion
format/target facts. The class is costly (64.1% of fresh direct FS) but ownership
does not reach it. No array code was written; source/build restored. Report
`/tmp/perf16/iter23-report.md`; census patch `/tmp/perf16/iter23-census.patch`;
raw/tables/trace remain under `/tmp/perf16/iter23-*`.

## Iteration 24 — result: kept opt-in hardware texture-cache checkpoint

The production path keeps tightly packed linear Vulkan images authoritative and
lazily materializes disposable CUDA mipmapped arrays and immutable texture objects.
It keys derived data by image content epoch, resolves every descriptor row before a
launch, pins cache lifetime through table upload and FS enqueue, and permits software
fallback only before any FS attempt. Strict same-LLVM HW_INLINE is preferred;
resource-isolated HW_FUSED recovers helpers/local-memory shaders without weakening
texture PTX validation. Hardware modes never use software sampler variants or sampler
globals.

The final format/target matrix covers the observed R8/RG8/RGBA8, RG16, float16,
A2, R11 and BC1/BC3 2D/cube/3D intersections. Exact gates cover mutation epochs,
mutable and immutable descriptors, view lifetime, mip ranges, implicit LOD, explicit
LOD/gradients, runtime `txb` bias, converted R11 cube and BC selectors. The last
acceptance blocker was a generic `llvm.exp2.f32` emitted for `txb`; NVPTX selected
an unresolved `exp2f`. It now uses `llvm.nvvm.ex2.approx.f`. A nonconstant SSA bias
1.0/1.25 selects mip 3 exactly on native HW_FUSED and NVIDIA validation.

Lifetime and fault handling are fail-closed. Descriptor view cookies are monotonic
IDs; framebuffer mutation tokens remain trusted internal image pointers. Stream
wait memoization uses monotonic stream serial plus ready generation, never a raw
`CUstream`. OOM at optional arrays/objects can refuse a whole launch; CUDA context,
async, enqueue, event, surface, cleanup and purge failures latch device loss. The
application-SSBO atomic no-replay matrix proves 1024 executions after a post-enqueue fatal
and zero executions for pre-FS argument/create faults. Serialized final suites pass
58/58 both default-off and cache-on; focused cache/fault gates pass 14/14.

Final external correctness passes old 10/10 and Crossroads 9/9 readback sentinels.
Against immutable llvmpipe, old mean RGB is 0.02057..0.47833 with maxima 1815 pixels
above 32 and 45 above 96; Crossroads is 0.80632..1.38985 with maxima 6250 and 65.
All 18 stored-60 sample processes exit zero; all automated NVIDIA rows pass except
the standing `gltfscenerendering` nondeterminism exception, while manual
`renderheadless` differs by zero pixels above 8 (maximum channel delta 1). All 17
600-frame offscreen-loop benchmark rows exit zero.

The final old-capture AB/BA medians are 17.2334 and 17.2224 ms cache-on versus
23.0296 and 23.1026 ms cache-off. Average medians are 17.2279 versus 23.0661 ms:
a 5.8382-ms hardware mechanism gain. Coverage is 157095/157211 old launches (99.9%)
and 37680/39822 Crossroads launches (94.6%). The change is kept as the correctness-
clean opt-in checkpoint, but it does **not** meet the absolute 16 ms goal: 1.228 ms
remains. The next performance work is iteration25 episode-global tagged raster, whose
design is recorded separately; it was not implemented here.

Reports and evidence: `/tmp/perf16/iter24-production-review.md`,
`/tmp/perf16/iter24-report.md`, and
`/tmp/perf16/iter24-acceptance/final-frozen/`.


## Iteration 25 — episode-global tagged A-buffer stage 3 (COUNT only)

An opt-in `CUDAPIPE_EPISODE_RASTER=1` mechanism replaces the per-segment
A-buffer COUNT stage 3 of a pass episode with one global tagged launch. Stage 2
publishes each segment's exact `cp_rasterize_args` from the device into a
job table, reserves the whole tile run of a triangle with one 64-bit
`atomicAdd`, and appends `{tri_id, tile_x, tile_y, job_id, flags}` pairs to a
separate episode-lifetime queue. The eight per-stream classic queue sets are
never deferred, and the setup cache is disabled for admitted COUNT work so a
tagged pair always names a raw primitive. Admission requires the record array
and `abuf_fill_recs`, so FILL keeps its existing linear replay.

The host enqueues the single stage 3 straight behind the checked producer
join: no synchronisation, no tail read, no job upload. Exactly one decision
follows, at the latest point at which nothing has been shaded — before the
bounded-group path's first shade, or at the main path's existing drain.
Overflow is bounded and re-renders the whole episode classically before any
fragment shader; corruption (bad job id, flags, setup tag, misaligned tile,
invalid primitive, tile outside the producer's own bounding box) latches
device loss. Pre-producer OOM is a soft refusal; every failure after a
producer has published is fatal.

Gates prove two full 64-job episodes, job id 63 produced and consumed, exact
capacity admitted, capacity+1 falling back before any FS and then retiring the
overflow by growing in the same context, recordless and allocation-refusal
exclusions, and five deterministic fault seams. A negative control confirms the
state-leak gate: with the `pending` reset removed, a following classic episode
validates the abandoned counters and re-renders segments it had already drawn.
Output is bit-identical to llvmpipe and to the classic chain, and the native
suite passes 59/59 both default-off and mechanism-on.

The mechanism works and removes launches, but it does not pay. Corrected
two-replay AB/BA medians, paired submits, same binary on and off:

| capture | mechanism | control | delta |
|---|---|---|---|
| old | 17.4509 ms | 17.4501 ms | -0.0008 ms (-0.004%) |
| Crossroads | 6.2784 ms | 6.2501 ms | -0.0283 ms (-0.452%) |

Stdout hashes are identical on both captures, so this is purely a cost result.

The cause is measured, not inferred. On the old capture the mechanism removes
123.8 classic COUNT stage 3 launches per frame and adds 11.6, a net 112 fewer
launches per frame; on Crossroads it removes only 22.4 and adds 7.2, because
that capture's episodes average 1.36 segments. Splitting the decision wait by
call site shows where the saving goes:

| capture | bounded checks | bounded wait | main checks | main wait |
|---|---|---|---|---|
| old | 2581 | 630.3 ms (244 us each, 0.417 ms/frame) | 14932 | 2.0 ms |
| Crossroads | 2442 | 556.1 ms (228 us each, 0.371 ms/frame) | 8333 | 1.2 ms |

The main path's check is free because that path already drained. The bounded
group path did not: it ran entirely asynchronously, and the mechanism's one
required decision introduces a full synchronisation there, worth about the
same as the launches it removes. The per-segment COUNT stage 3 launches were
also already spread over eight side streams, so merging them onto the main
stream serialises work that had been concurrent.

Verdict: **rejected as neutral, and the mechanism code was reverted.** It is
not a correctness risk, but it has no route to a win while the bounded-group
path must synchronise once per episode, and a neutral result does not justify
carrying a device ABI, a second stage-2/stage-3 kernel pair and six switches.
A future attempt has to prove the overflow and corruption bounds on the device
so that path keeps its asynchrony; it should restart from the design and gates
recorded here rather than from scratch.

What survives in the tree is the fail-closed hardening this iteration exposed
on the default path, which is independent of the mechanism: `cp_pass_join()`
and `cp_pass_broadcast()` now report failure and every caller stops, so a
failed join can no longer be followed by a composite, a re-render or a rollback
re-execution; the episode's shared-list clears and its gate event in
`cp_pass_append()` are checked; the rollback rechecks `device_fatal` before
re-executing the batch; and the A-buffer count pass checks the queue-counter
clear it depends on.

The reverted mechanism diff is kept at `/tmp/perf16/iter25-full-mechanism.patch`
(and the pre-revert renderer at `/tmp/perf16/iter25-cp_renderer.c.mechanism`).

Evidence: `/tmp/perf16/iter25-mechanism-report.md`,
`/tmp/perf16/iter25-tworeplay/`, `/tmp/perf16/iter25-tworeplay-report.txt`,
`/tmp/perf16/iter25-stats2/`, `/tmp/perf16/iter25-gate5.out`, and the negative
control `/tmp/perf16/iter25-gate-negctl.err`.


## Iteration 26 — removing the sub-4 KB device operations

Iteration 25's profile said the old capture spends 3.83 ms a frame with the
device idle, 69% of it directly behind a small copy or a small clear, and that
it issues 1,988 operations under 4 KB per frame to move 0.59 MB. The lever is
the count, not the bytes.

### S0 — count them at the call site

`CUDAPIPE_UPLOAD_STATS` attributes every small copy, clear, context sync and
upload-ring wrap to its call site with no profiler attached, because CUPTI adds
host cost to exactly the calls being counted. The census confirms the profile
and comes in slightly under it: old measures 1,098.3 small copies and 769.8
small clears per frame, Crossroads 263.5 and 244.8. Six clear sites are 80% of
old's small clears; six upload sites are 70% of its small copies.

It also retired a suspicion. `cp->scratch.current` is never assigned in the
native driver, so the upload arena stays on generation 0 and a wrap takes a
whole-context `cuCtxSynchronize`. That was expected every frame or two. It
fires 79 times in 1,511 frames on old and never in 1,497 frames on Crossroads,
because `cp_scratch_reset()` rewinds both offsets at every flush. Generation
rotation needed no change.

The instrument costs nothing when off: two binaries from one tree, AB/BA on
both captures, old 17.4166 against 17.4044 and Crossroads 6.2170 against
6.2246.

### S1 — the vertex stage's two scalars (`CUDAPIPE_NO_META_FOLD` reverts)

`vcount` and `stride` travelled as their own eight-byte upload, the single most
frequent host-to-device operation in the driver. They now sit in two more words
of the argument block's own scalar area. Measured alone, identical hashes:

| capture | fold | control | delta | copies removed |
|---|---:|---:|---:|---:|
| old | 17.0910 | 17.3560 | +0.2651 ms (+1.53%) | −203.7/frame |
| Crossroads | 6.1539 | 6.2192 | +0.0654 ms (+1.05%) | −50.3/frame |

**0.2651 ms over 203.7 operations is 1.30 µs each, and 0.0654 ms over 50.3 is
also 1.30 µs** — the same price on two workloads whose batch counts differ
fourfold. That is the per-copy number the rest of the iteration is predicted
from.

### S2 — the counters the next launches fill (`CUDAPIPE_NO_FETCH_FOLD` reverts)

`cp_vertex_fetch` runs once per executed batch, on the same stream, ahead of
the clipper and the rasterizer, so it seeds the clipper's output counter and
the three raster queue counters itself.

| capture | fold | control | delta | clears removed |
|---|---:|---:|---:|---:|
| old | 17.1152 | 17.3781 | +0.2629 ms (+1.51%) | −396.0/frame |
| Crossroads | 6.1805 | 6.2485 | +0.0680 ms (+1.09%) | −91.2/frame |

**A removed clear is worth 0.66 µs on old and 0.75 µs on Crossroads, about half
what a removed copy is worth.** The price is not one constant: a copy costs a
host call, a device operation and a boundary; a clear costs less. The remaining
small clears on old are therefore worth at most 0.38 ms and the remaining small
copies at most 1.16 ms, which is what makes the copy-side mechanism the one to
build.

### S3 — one copy per launch boundary (`CUDAPIPE_NO_UPLOAD_COALESCE` reverts)

`cp_upload_end()` stops copying and records that the staging bytes below
`upload_offset` are owed; one `cuMemcpyHtoDAsync` per flush point sends the
whole span. The allocator, the alignment, the generations, the device
addresses, the block contents and the shader ABI are unchanged.

Flush points are every launch (`CP_LAUNCH` and all eighteen former raw
`cuLaunchKernel` sites now go through `cp_launch()`), every stream switch
(`cp_stream_set()`, which flushes onto the *old* stream), every wait, every
event another stream waits on, every arena rewind, and the end of a submit.
`tests/cp_launch_audit.py` fails the suite if a raw `cuLaunchKernel` appears
outside the two allowlisted definitions, because a launch that does not flush
is the one silent bug this mechanism can have.

| capture | coalesced | control | delta | copies removed |
|---|---:|---:|---:|---:|
| old | 17.0954 | 17.3933 | +0.2979 ms (+1.71%) | -502.5/frame |
| Crossroads | 6.2354 | 6.2178 | -0.0176 ms | -115.1/frame |

Crossroads is **neutral within spread**, not a win: both arms vary by about
0.03 ms across runs. The merge ratio explains it -- only 1.82 blocks per copy
on old and 1.75 on Crossroads, because every launch is a flush point and this
driver launches constantly.

**The rule this iteration produced, and the one to carry forward:**

> **Removing an operation outright pays about twice what merging operations at
> an unchanged boundary pays.** S1 removed a copy *and* its boundary and was
> worth 1.30 us per operation. S3 merges copies at a boundary that stays where
> it was, removing the host call and the device operation but not the gap, and
> is worth 0.59 us. Both numbers are measured, on both captures.

It also predicts its own future: when iteration 27's vertex-chain fusion
removes 222 launches a frame it removes 222 flush points, the merge ratio
rises, and S3 should get *better*. If it does not, the boundary model is wrong
in a way worth knowing.

### S1+S2+S3 together

| capture | all three | control | delta |
|---|---:|---:|---:|
| old | 16.5786 | 17.4239 | **+0.8453 ms (+4.85%)** |
| Crossroads | 6.0926 | 6.2522 | **+0.1597 ms (+2.55%)** |

That is 102% and 138% of the sum of the stages measured separately, so the
sub-additivity an earlier pair suggested was control drift between sessions,
not overlap. A control belongs in the same session as its candidate.

### What S2 must not fold, and why

The design also asked for the packed vertex input's pre-clear to move into the
gather kernel, predicting −600 to −650 operations a frame. Folding all three
hit that target exactly — clears fell from 968.94 to 369.20 per frame, −599.7 —
and the frame got **1.8412 ms slower on old and 0.2640 ms slower on
Crossroads**, with identical output.

That clear is small by call count and bulk by bytes: only 117.4 of its 203.7
calls a frame are under 4 KB, and it moves about 90 MB a frame, which
`cuMemsetD8Async` does in one kernel at DRAM speed. Row-by-row inside the
gather kernel it lost even with 16-byte stores. The losing path was deleted;
the site carries a comment so the next person does not re-derive it.

The rule the census suggests: fold an operation for its **count** only when its
**bytes** are negligible. Counters are; buffer clears are not.

**A coupling to carry forward:** S2's counter seeding lives inside
`cp_vertex_fetch`, so it exists only while that launch does. Iteration 27
fuses the fetch into the vertex shader; the seeds must move with it, or S2 is
silently reverted — the counters would go back to their own clears with
nothing failing. The seeding site says so in a comment.

### S4 -- one clear for the counter block (`CUDAPIPE_NO_COUNTER_BLOCK` reverts)

The A-buffer's scalar counters were already one allocation for `sum3`,
`bsum3`, `clist_count`, `seg_counts[64]` and `rec_cursor`; `list_count`,
`blk_list_count` and the interpolator's debug counters were three more. They
now share one block, and a draw or an episode clears all of it once instead of
issuing six to eight clears. The per-pixel arrays keep their own bulk clears: a
counter whose size depends on the framebuffer does not belong in the block.

| capture | block | control | delta | clears removed |
|---|---:|---:|---:|---:|
| old | 17.4079 | 17.4268 | +0.0190 ms | -115.8/frame |
| Crossroads | 6.1851 | 6.2542 | **+0.0692 ms (+1.11%)** | -68.2/frame |

Old is **neutral within spread** -- the delta is the size of that session's
control spread, 0.0195 ms. Crossroads is real: twelve times its control spread.
The prices are 0.16 us per removed clear on old and 1.01 us on Crossroads,
against S2's 0.66 and 0.75.

**Why they disagree, and it is worth re-testing:** these are per-draw and
per-episode *fixed* costs. Both captures run a similar number of episodes per
frame -- 23.0 on old, 16.5 on Crossroads -- but a Crossroads frame is 2.8x
shorter, so the same fixed cost is nearly three times the share of its frame.
Old's frame is dominated by work that scales with geometry and fragments.
**Crossroads is overhead-bound where old is work-bound**, which predicts that
further fixed-cost removals keep favouring Crossroads while launch-count and
fragment-side work keep favouring old.

### The shipping default

All four stages are on by default; each has a `CUDAPIPE_NO_*` switch that
restores its old path exactly. The reverts are verified by operation count, not
by inspection -- each flag puts back precisely the operations its stage removed,
and all four together reproduce the original census to the decimal:

| configuration | copies/frame | clears/frame |
|---|---:|---:|
| default | 615.32 | 457.08 |
| `NO_META_FOLD` | 615.32 (blocks +203.7) | 457.07 |
| `NO_FETCH_FOLD` | 615.32 | 853.11 |
| `NO_UPLOAD_COALESCE` | 914.06 | 457.07 |
| `NO_COUNTER_BLOCK` | 615.32 | 572.92 |
| all four | 1117.78 | 968.94 |

Small operations per frame fall from 1,868.1 to 1,072.4 on old, 43% fewer.

| capture | default | all four reverted | delta |
|---|---:|---:|---:|
| old | **16.5021** | 17.4331 | +0.9310 ms (+5.34%) |
| Crossroads | **6.0467** | 6.2336 | +0.1868 ms (+3.00%) |

Acceptance: the native suite passes 59/59 with the default, with each revert
alone and with all four; stdout hashes are identical across every replay arm;
the Crossroads sentinel frames are byte-identical to the ones iteration 24
accepted, and the old capture's differ by at most 13/255 on a handful of pixels
-- the same variation two runs of one binary show, and about a thousandth of
the tolerance the llvmpipe comparison already accepts. The sample sweep
reproduces iteration 24's verdict exactly, including its one standing
`gltfscenerendering` nondeterminism exception.

### Two things future acceptance runs need to know

**The old capture's sentinel frames are not deterministic, and that is not a
regression.** Dumping the ten readback sentinels of
`headless_streamer_20260814T155742.gfxr` twice, from one binary with one set of
flags, produces six frames that differ. They differ by a mean absolute delta of
0.00003 and at most 13/255 on a handful of pixels. The llvmpipe envelope
iteration 24 accepted for this capture is a mean of 0.0206 to 0.4783 with up to
1815 pixels over 32 and 45 over 96, so the run-to-run variation is about a
thousandth of the tolerance that already passes. Crossroads, by contrast, is
byte-identical run to run.

**The correct test on the old capture is therefore the external llvmpipe
envelope, not frame-to-frame equality**, and a frame-to-frame difference there
is evidence of nothing until its magnitude is compared against that envelope.
Four dump sets are kept as the demonstration:
`/tmp/perf16/iter26-acceptance/frames/old` (default),
`.../frames/old-again` (default, second run),
`.../frames/old-revert` (all four stages reverted) and the frozen
`/tmp/perf16/iter24-acceptance/final-frozen/frames/old`. The same six frames
differ in every pairing, including default against default.

**Measuring a default-on change needs the reverts on the control arm.**
`cp_two_replay_ab.sh` puts `CAND` on the candidate arm and calls a positive
delta a win, so putting the revert switches in `CAND` yields correct medians
under an inverted verdict. The harness now takes `CTRL` for control-arm
environment and writes `arms.txt`, so a default-on stage is measured as
`CAND="" CTRL="CUDAPIPE_NO_...=1 ..."` and the sign stays right.

## Iteration 27 — the vertex fetch inlined into the vertex shader

**Result: kept, default on, `CUDAPIPE_NO_FUSED_VFETCH=1` reverts.**
Old capture **16.5119 → 16.0194 ms** (+0.4925 ms, +2.98%), Crossroads
**6.0715 → 5.9928 ms** (+0.0787 ms, +1.30%), on the decisive alternating
measurement below. The old capture ends this iteration *at* 16 ms rather than
under it.

**The old capture now sits *at* 16 ms, not below it.** Three AB/BA sessions
put the default arm at 15.9448, 15.9726 and 16.0544, which is a spread wide
enough that the target could be claimed or missed depending on which session
was quoted. The decisive measurement settles it: six runs per arm, strictly
alternating default and reverted, one session, one binary, exclusive GPU
(`/tmp/perf16/iter27-decisive`, `cp_decisive_ab.sh`).

| arm | runs (ms) | median | IQR | range |
|---|---|---:|---|---|
| default | 15.9337 16.0557 16.0340 16.0344 15.9967 16.0047 | **16.0194** | [15.9987, 16.0343] | [15.9337, 16.0557] |
| reverted | 16.5541 16.4981 16.5912 16.5201 16.5038 16.5030 | 16.5119 | [16.5032, 16.5456] | [16.4981, 16.5912] |

**The default arm's interquartile range straddles 16.0 and four of its six
runs are at or above it.** The goal of this log is therefore *reached at the
line and not safely*: quote 16.02, not 15.97, and do not treat the difference
as headroom. What is robust is the delta — paired run against adjacent run it
is +0.4424 to +0.6203 with a median of **+0.5027 ms**, and the median-of-
medians difference is +0.4925 (+2.98%) — and the reverted arm, which lands
within 0.01 ms of the 16.5021 this branch stood at.

Crossroads, four runs per arm the same way: default median **5.9928**
(IQR [5.9764, 6.0037]), reverted 6.0715 (IQR [6.0621, 6.0822]), delta
+0.0787 ms (+1.30%). Every run of all twenty produced the same stdout hash as
every other run of its capture.

`cp_vertex_fetch` ran once per executed batch, immediately before the
generated vertex shader, over the same vertices, on the same stream, and
handed its result over through a global buffer nothing else read. That launch,
its 90 MB pre-clear and the upload flush it forced are gone for any vertex
shader whose fused build is admitted: the per-lane gather is linked into the
shader as NVPTX bitcode in the same `LLVMContext`, force-inlined before
optimisation, and its results live in a function-entry alloca that SROA
promotes to registers. `kernels/cp_vf_lane.h` is the only description of vertex
format semantics in the driver; the standalone kernel and the shader both
compile it, so the two forms cannot drift.

Iteration 4 tried this and was rejected at 48.05 ms against a 23.42 ms base,
for two reasons that no longer apply: it passed an LLVM `alloca` pointer to an
NVRTC-compiled helper (two address-space models, one pointer), and when it
retreated to a device call the link unioned the register allocation — a
trivial vertex shader went 20 → 108 registers and `__noinline__` did not help.
Here there is no second compiler and no call in the emitted PTX at all. The
admission gate is measured, not assumed, and a declining shader keeps its
classic binary and its separate fetch launch permanently.

### The finding that decides whether this works at all

**A loop over the elements cannot be left to LLVM's unroller.** The first
working form passed `N` and a live-slot mask as constants and gathered in a
loop bounded by `N`, with `#pragma clang loop unroll(full)` on it — the shape
the fragment interpolator uses. LLVM refused to unroll it for every real
capture shader (the body is a whole conversion tree; the pragma loses to the
size heuristic), the slot array was then indexed dynamically, and a
dynamically indexed alloca is local memory: every shader came back with a
`__local_depot` of exactly `N * 16` bytes. Launch-weighted admission in that
form was **A = 0.1081**.

Emitting the unrolled sequence from the backend instead — one
`cp_vs_fetch_element` call per live slot, with the element index an
`LLVMConstInt` — removed all of it. Two smaller rules fell out of the same
work and are in the header's comments:

* **never write a slot as bytes in one path and as words in another.** SROA
  will not split a partition with conflicting access types, and it takes the
  whole array to memory when it refuses. The fused build uses one word-based
  shape for every 32-bit-per-component format; the standalone kernel keeps its
  three-branch copy, which is why that path is a `#ifdef` and not a rewrite.
* **a fixed bound with a `continue` unrolls; a fixed bound with a `break` does
  not.** The inner component loops run to four and skip, rather than exiting
  early on the runtime channel count.

The zero fill has to happen in the helper: the classic caller's slots read zero
because the host memset the buffer, and registers have no such history. Missing
it is silent wrong geometry, not a crash — §3.7 of the design predicted exactly
that and `cpvk_vfetch formats` is the test for it.

### Admission, measured before any performance claim

`CUDAPIPE_SHADER_STATS=1` now prints a per-shader census weighted by **vertex
launches**, which nothing recorded before this iteration: a verdict counted per
shader says nothing about a frame when one shader takes two launches and
another two hundred.

| capture | shaders | launches | A by launch | A by shader | declines |
|---|---:|---:|---:|---:|---|
| old | 48 | 307,811 | **0.9538** | 0.8125 | 9 shaders / 14,228 launches, all `lost-a-block` |
| Crossroads | 19 | 75,243 | **0.8838** | 0.7895 | 4 shaders / 8,746 launches, all `lost-a-block` |

No shader on either capture declined for a surviving helper symbol, for new
local memory or for spill: the only decline reason that occurs is losing a
block per SM, which is a shader being heavy rather than the inline being
fragile. Launch-weighted registers on old: classic 53.7 → fused 55.7. The
`instancing` sample — the one the design worried about, because
`nir_opt_move_to_top` hoists vertex-input loads on purpose — admits all three
of its shaders and its 4.42-million-vertex rock shader goes **120 → 104
registers at the same occupancy**. The hoisting is harmless now: what it hoists
reads registers.

### What the frame stops doing

Old capture, per frame over 1,511 frames (`CUDAPIPE_UPLOAD_STATS=1`,
`/tmp/perf16/iter27-census/old-{0,1}.stderr`):

| operation | classic | fused | delta |
|---|---:|---:|---:|
| vertex-fetch launches | 194.3 | 0 | **−194.3** |
| `cuMemsetD8Async` | 457.07 | 262.78 | **−194.3**, and −84.9 MB/frame |
| `cuMemcpyHtoDAsync` | 615.31 | 423.31 | **−192.0** (the fetch launch was a flush point) |
| upload-ring wraps | 0.05 | 0.05 | unchanged |

### The launch price, corrected

Iteration 26 priced a removed small copy at 1.30 µs, a removed small clear at
0.66 µs and a copy merged at an unchanged boundary at 0.59 µs. This design
*assumed* a removed same-stream launch was worth 1.4–2.2 µs and predicted
0.65 ms × A = 0.62 ms on old. The measurement is **0.512 ms** in the opt-in
A/B and **0.529 ms** with the flip, i.e. 0.537–0.554 ms per unit admission —
at or just below the bottom of the design's own 0.55–0.85 band.

Holding iteration 26's prices fixed, the removed clears and merged copies
account for 0.128 + 0.113 = 0.241 ms and the 84.9 MB never written for about
0.065 ms, which leaves roughly **0.13–0.21 ms for 194.3 removed launches plus
the whole `vs_in` round trip — under 1 µs per launch**. That split is
arithmetic over the census, not a measured decomposition; the
`CUDAPIPE_VFETCH_KEEP_VSIN` attribution switch the design proposed was not
built, because it needs the fused kernel to store `vs_in` as well and that is
more codegen for a number this arithmetic already bounds.

**Take 0.6–1.0 µs, not 1.4–2.2 µs, as the price of a removed same-stream
launch on this capture.** Iteration 3's 1.96–3.19 µs included a device-work
reduction that this fusion does not have. The consequence is concrete and
should be applied before the next design is written: iteration 25's
opportunity 3, the A-buffer support-chain fusion at ~180 launches/frame, drops
from an expected 0.3–0.6 ms to roughly **0.11–0.18 ms**, which no longer
obviously pays for its risk.

### Sweep, and the one sample that regressed

600-frame offscreen sweep, hot sum **25.28 → 22.59 ms (−10.6%)**:
`instancing` 4.54 → 2.06 (−55%), `pushconstants` 0.20 → 0.13,
`vulkanscene` 0.72 → 0.65, `bloom` 1.15 → 1.12, everything else flat.

`pbribl` regresses **0.47–0.49 → 0.51–0.52 ms**, reproducibly, over three
paired repeats. Its host-side operation counts all *fall* with the fusion
(copies 5.47 → 4.10, clears 4.10 → 2.73 per frame, 39.9 → 27.8 GB of clear
traffic), all four of its vertex shaders admit, and three of the four lose
registers, so the 0.03 ms is not in the host work this iteration removes.

**Attributed as far as it is cheap to.** Three arms in one session, three
repeats each: (a) fused, (b) `NO_FUSED_VFETCH`, and (c)
`CUDAPIPE_VFETCH_DECLINE_NTH=4294967295` — the second binary built, its
registers measured, the per-draw decision taken, and every shader then forced
onto the classic path, so the arm pays the mechanism's infrastructure and
executes none of it.

| arm | ms/frame | wall s |
|---|---:|---:|
| (a) fused | 0.51–0.52 | 1.76–1.87 |
| (b) reverted | 0.48–0.49 | 1.70 |
| (c) built, measured, not executed | 0.47–0.48 | 1.73–1.74 |

**(c) matches (b), not (a)**, so the cost is in executing the fused kernel on
this workload (or in its larger per-draw argument block), not in the second
binary, the admission measurement or the decision — those cost 0.03–0.04 s of
process wall time and nothing per frame. That bounds it; it is still not
explained.

This is a **literal miss of the design's rejection criterion 6** ("any sample
in the 600-frame sweep regressing more than 2%"), and it is kept deliberately:
0.03 ms on a 0.5 ms sample against −2.5 ms on `instancing`, −0.53 ms on the old
capture and −0.09 ms on Crossroads, with every correctness gate intact. The
criterion was not forgotten; it was weighed.

### Gates

Native suite **65/65** in the default state and in the reverted state — the 59
existing tests plus four new `cpvk_vfetch` modes and two gates. Six batch
reproducers byte-identical across five `CUDAPIPE_BATCH_MAX` modes crossed with
default/reverted (30 runs, one hash). Crossroads sentinel frames
**byte-identical** to iteration 24's frozen reference; the old capture's
differ by at most 51/255 on one pixel of two frames, which is the run-to-run
class that reference already shows and a thousandth of the llvmpipe envelope.
The 60-frame NVIDIA comparison reproduces iteration 24's verdict exactly,
including the two standing exceptions (`gltfscenerendering` REGRESSED,
`renderheadless` missing). `cp_launch_audit`, both `FLAGS.md` checks,
`git diff --check` and the empty `src/gallium` diff all clean.

Four new tests exist because neither capture can reach these cases — iteration
4's census found no divisored attribute anywhere in them, and each conversion
class appears in one shape only:

* `cpvk_vfetch formats` — one attribute per conversion class at one to four
  components, a BGRA swizzle, three halves on a two-byte boundary (the
  byte-at-a-time path), a 32-bit pair off a sixteen-byte boundary, every
  `fill_w` case, drawn indexed with a non-zero `firstIndex` and `vertexOffset`
  over a buffer whose first vertex is poison. Passes on classic, on fused and
  on llvmpipe.
* `cpvk_vfetch sparse` — locations 0, 3 and 7 bound, only location 3 read, with
  the other two holding values that fail the test if the shader is handed them.
* `cpvk_vfetch divisor` — two per-instance attributes, four instances,
  `firstInstance = 2`. It also documented a driver gap: this driver maps
  `VK_VERTEX_INPUT_RATE_INSTANCE` to divisor 1 and does not implement
  `VK_EXT_vertex_attribute_divisor`, so a divisor above one is **unreachable
  through the API** — the kernel path exists and nothing can select it. Now in
  `CUDAPIPE_HANDOFF.md` with the other conformance gaps.
* `cpvk_vfetch decline` — the instanced draw into the left half and the sparse
  draw into the right, one render pass, one submit, with
  `CUDAPIPE_VFETCH_DECLINE_NTH` forcing one of the two shaders onto the classic
  path. This is the mixed-mode case per-shader admission creates in every real
  frame, and all five flag states are byte-identical.

**The seeding negative control matters most of the four gates.** Iteration 26
S2 seeds the clip and raster counters inside `cp_vertex_fetch`; if that job had
not moved into the fused kernel, nothing would fail — the counters would go
back to being cleared by the launches that consume them and the frames would
still be right, while this iteration reported a saving that included S2's.
`CUDAPIPE_VFETCH_SKIP_SEED=1` puts the driver in exactly that state, and three
of the existing tests then fail. `cpvk_vfetch_seed_gate` requires the canary to
pass seeded and to **fail** unseeded, so the gate cannot pass vacuously.

### Flags

| flag | meaning |
|---|---|
| `CUDAPIPE_FUSED_VFETCH` | default **1**; gather inside the vertex shader for admitted shaders |
| `CUDAPIPE_NO_FUSED_VFETCH` | the revert: the second binary is not even built, so the classic path is the only linked call graph |
| `CUDAPIPE_VFETCH_DECLINE_NTH` | fault injection: the Nth vertex shader compiled declines |
| `CUDAPIPE_VFETCH_SKIP_SEED` | fault injection: an admitted fused draw does not seed, and the host still skips the clears |

Artifacts: `/tmp/perf16/iter27-s1-tworeplay` (opt-in A/B),
`/tmp/perf16/iter27-flip-tworeplay` (default vs revert),
`/tmp/perf16/iter27-census/` (operation census),
`/tmp/perf16/iter27-census-old2.stderr` and `-cross.stderr` (admission),
`/tmp/perf16/iter27-acceptance/` (sentinels),
`/tmp/perf16/iter27-sweep/` (sweep).

### Two measurements taken after the fusion landed

**The upload merge ratio rose exactly as iteration 26 S3 predicted, and the
residual is not a merge problem.** S3's model was that every launch is a flush
point, so removing launches should merge the uploads that sat on either side of
one. Removing 194.3 launches a frame does that, measured from one binary with
the mechanism on and off (old capture, 1,511 frames):

| arm | blocks/frame | copies/frame | blocks per copy | empty flush points/frame | carried |
|---|---:|---:|---:|---:|---:|
| default | 914.07 | 423.31 | **2.159** | 1362.08 | 23.7% |
| reverted | 914.06 | 615.31 | 1.486 | 1364.35 | 31.1% |

The same upload blocks exist in both arms, as they must: the fusion removes
launches, not uploads. The ratio rose 45%.

Two things about that number. S3 recorded 1.82, from a census whose block count
was 1,688,964 against today's 1,381,144 for the same 929,7xx flushes — about
307,800 fewer blocks, suspiciously close to the 307,811 vertex-fetch launches,
so something that contributed a block per fetch launch is no longer counted.
**Quote the same-binary comparison above, not the 1.82.**

And the answer to "is there more to get from smarter flush placement" is **no**,
which the census settles rather than argues: there are 1,785 flush points a
frame and only 423 of them carry anything, so 76% are already an empty
predicate. The 914 blocks that exist are spread over the 423 carrying points at
2.16 each, and merging further would mean deferring an upload past a launch
boundary — where the launch on the other side is what reads it. The remaining
copies are one per boundary that carries data. **The lever is removing the
boundary, not flushing more cleverly.**

**Sizing the fragment grid to the machine instead of to the framebuffer is
neutral, and that refutes the candidate rather than deferring it.** The direct
fragment launch covers the framebuffer's worst case and caps at 4,096 blocks,
which on these captures it always hits; the shader grid-strides, and NCU
measured 12.05 waves per SM with 65% of launches running under 50 instructions
per thread. The obvious reading is that eleven of those twelve waves are blocks
scheduled to discover they have nothing to do.

`CUDAPIPE_FS_GRID_WAVES=W` caps the grid at `SMs × blocks_per_sm × W` instead —
occupancy taken from the execution being launched, since a 40-register shader
and a 126-register one do not fit the same number of blocks, with 4,096 kept as
the hard upper bound. It reaches the launches it is aimed at, by the census this
change also adds:

| setting | FS launches | blocks | blocks/launch |
|---|---:|---:|---:|
| 0 (today) | 157,215 | 502,548,728 | 3196.6 |
| 1 | 157,212 | 39,147,304 | **249.0** |
| 4 | 157,217 | 147,941,756 | 941.0 |

That is 104.0 fragment launches a frame and **306,686 fewer blocks scheduled per
frame** at `W=1`. The frame does not move. Cycling the settings and repeating
the cycle so drift is shared by all of them, three cycles on old and two on
Crossroads, every median lands inside the within-setting spread:

| waves | old median | vs today | Crossroads median | vs today |
|---|---:|---:|---:|---:|
| 0 | 15.9899 | — | 5.9786 | — |
| 1 | 15.9903 | −0.0005 | 5.9662 | +0.0124 |
| 2 | 15.9720 | +0.0179 | 5.9668 | +0.0118 |
| 4 | 15.9344 | +0.0555 | 5.9903 | −0.0117 |
| 8 | 16.0472 | −0.0573 | 5.9692 | +0.0095 |

`W=1` alone spans 15.9465–16.0075, so the whole column is one distribution.
**The result to keep is the bound it puts on the thing everyone assumes:
306,686 scheduled blocks a frame cost less than 0.06 ms, which is under 0.2 ns
for an idle 256-thread block on this GPU.** A block that reads a device-side
count and exits is free. The NCU instruction mix is real; the conclusion that
the grid was the cost is not.

Where the time is instead, measured with nothing attached (nvidia-smi at 5 Hz
across whole replays): **75.2% GPU utilisation at `W=0` and 72.1% at `W=1`** —
about a quarter of the replay has no kernel resident at all, and shrinking the
grids does not change it. That is consistent with this iteration's win coming
from removing host-side launches and boundaries, and inconsistent with idle
blocks being where the next 0.15–0.25 ms was going to come from.

The flag stays at `0`, which is today's behaviour byte for byte, for the same
reason `CUDAPIPE_LAUNCH_BOUNDS` stays: re-measuring this on different hardware
should cost one command, not a re-implementation. The fragment-grid census
stays because it is what proves a grid change reached the launches it aimed at.

**This is deliberately not what happened to iteration 26's S2b, and the
difference is the point.** S2b changed behaviour and lost 1.84 ms, so the path
was deleted and only a comment was left; a kept switch there would have been a
loaded gun. This one is byte-identical at its default — the grid is computed
exactly as before unless someone sets the variable — and its only purpose is to
be re-measured on a machine whose block scheduler is not free. Delete it the day
the fragment launch stops grid-striding, because then it would be a way to drop
work.

## Iteration 28 item 4 — the A-buffer support chain fused

`CUDAPIPE_PERF16.md`'s iteration-28 profile ranked this fourth: 251.3
launches/frame over `scan`, `sort`, the worklists, the quad count/fill and
`fill_recs` for 1.30 ms/frame of device time, estimated at 0.20–0.40 ms of
frame time at the measured price of under 1 µs per removed same-stream launch.

The chain was re-derived from the source before anything was changed.
`CP_ABUF_SCAN_BLOCK` is 512 and the old capture is 1280x720, so the pixel scan
(n = 921,600, nb1 = 1800) takes `cp_abuf_scan_n`'s five-launch three-level
branch and the quad-block scan (n = nblocks = 230,400, bnb1 = 450) takes its
three-launch branch. One pass episode issues fifteen launches and two large
clears. The predicted split — 3+2 per episode `scan_block` and 2+1 `scan_add`
against 16.3 episodes/frame, so 81.5 and 48.9 — reproduces the profile's
measured 80.4 and 47.8, which is what established that the code being changed
is the code that was profiled.

### The mechanism, in two static per-episode fusions

**S1, the scan in two launches instead of five (or three).** One tiling is
computed on the host and shared by every kernel that touches it:
`ept = max(1, ceil(n / (512*512)))` elements per thread and
`grid = ceil(n / (512*ept))`, which makes `grid <= 512` for every n by
construction. That bound is the whole mechanism: the array of per-block sums
is then small enough that *every* block can scan all of it itself, in shared
memory, and read out both its own exclusive base and the grand total without
waiting for or communicating with any other block.

* `cp_abuf_scan_reduce` writes one block sum per block.
* `cp_abuf_scan_finish` does the redundant top-level scan, then walks its own
  tile in `ept` rounds of the same 512-wide Hillis-Steele scan
  `cp_abuf_scan_block` already ran, and writes the final offsets directly.
  `cp_abuf_scan_add`'s clamp is carried over unchanged; the grand total it
  tests is now a register rather than a load.

The redundant top scan sums **uint32 counts**, so the reassociation is exact:
the fused total is bit-identical to the classic one by construction, not
within a tolerance. Global traffic falls from 2n reads + 2n writes to 2n reads
+ n writes. An optional `zero` argument folds the 3.7 MB fill-cursor clear into
the same pass — only on the episode path, because `cp_abuf_size_arrays()` can
replace `ab->cursor` between the standalone path's scan and its fill.

**S2, the quad build in three launches instead of six.**
`cp_abuf_quad_count_all` is laid out on exactly the S1 tiling of `nblocks`, one
thread per 2x2 block. It tests coverage itself, from the four counts
`cp_abuf_block_worklist` was reading anyway, so the compaction pass disappears;
it writes `blk_counts[b] = 0` for an uncovered block, so the 0.9 MB clear
disappears; and because its tile *is* the scan's tile it block-reduces its own
counts into `bsum1`, so that scan's reduce disappears too. What is left is the
count, `cp_abuf_scan_finish`, and `cp_abuf_quad_fill_all`, which skips a block
whose `blk_counts` is zero and otherwise runs the identical merge at the
identical offset. Parallelism goes up rather than down: the classic count is
1024 blocks of 32 threads grid-striding an ~8,100-entry list, the fused count
is 450 blocks of 512 threads with at most one merge per thread.

### Why this is none of the shapes already rejected

* **Not iteration 21.** Every kernel keeps a static grid, a static
  index-to-work mapping and one exit. No persistent residency, no per-item
  claiming, no stealing, no queue publication and no done protocol.
  `scan_finish` recomputes the top-level scan per block precisely so that no
  block ever waits on another.
* **Not iteration 25.** No host synchronisation is added on any path. The
  episode's one drain stays where it was, reads the same six counters plus the
  per-segment quad counts, and the bounded-group path's asynchrony is
  untouched. Nothing is merged across segments and no new pre-FS decision
  exists. Iteration 25 removed 112 launches/frame and gained nothing because
  one mandatory wait cancelled it; there is no such wait here.
* **Not bound-based sizing.** No buffer changes size and nothing is sized to a
  worst case. `s1` already holds `nb1 = ceil(n/512) >= grid` words. The episode
  drain that buys the exact quad count is untouched.
* Overflow and corruption are still detected before any fragment shader: the
  clamp, the fill's `overflow` and `quad_overflow` are computed by the same
  arithmetic in the same order and read by the same drain at the same point.

### Gates, all before any timing

`CUDAPIPE_ABUF_FUSE_CHECK=1` runs the classic chain first into shadow buffers,
with its clamp disabled so it cannot disturb the counts the fused chain then
reads, runs the fused chain into the live buffers, and compares on the device:
every offset, the grand total, every `blk_counts` entry, and the invariant
`cp_abuf_quad_fill_all`'s skip rests on — `blk_counts[b] != 0` if and only if
block b is covered. Full coverage of `blk_counts`, which is what retiring its
clear requires, is proved rather than argued: the live array is pre-filled with
a sentinel and survivors are counted.

| gate | result |
|---|---|
| old capture, whole replay | **23,814 scans and 23,814 quad builds compared; 0 differing elements, 0 differing totals, 0 entries never written, 0 coverage violations** |
| Crossroads, whole replay | **13,310 and 13,310; all four counters 0** |
| native suite, fusion on | 65/65 |
| native suite, both reverts | 65/65 |
| old sentinels vs llvmpipe | inside the accepted envelope on all ten frames; four byte-identical to the frozen reference and six differing by ≤0.00005 mean and ±1 pixel over 32, which is the old capture's documented run-to-run nondeterminism |
| Crossroads sentinels | **byte-identical to the frozen reference, 9 of 9** |
| stdout hashes, all sixteen timed runs | identical per capture |

The negative controls have teeth. `CUDAPIPE_ABUF_FUSE_BREAK=1` drops the block
base in `scan_finish`; `=2` stops the fused count writing an uncovered zero,
which is exactly what a missing clear looks like. On the old capture the first
reports 60,208,448 differing elements and the second 1,691,296 differing,
1,691,296 never written and 1,691,296 coverage violations. Both also destroy
the output: with the check off, break 1 produces four sentinel frames with a
mean channel error of 60.5 and 440,741 pixels over 32 — against an envelope of
0.02–0.48 and at most 1,815 — and then latches device loss, and break 2 latches
device loss before the first sentinel frame is written at all.

### What the frame stops doing, counted by the driver

A launch counter at `cp_launch()` — the one place a launch can happen — and
the iteration-26 call-site census, both profiler-free:

| capture | arm | launches/frame | clears/frame | launches removed | clears removed | MB of clear traffic removed |
|---|---|---:|---:|---:|---:|---:|
| old | classic | 1405.4 | 262.8 | — | — | — |
| old | S1 only | 1345.8 | 251.2 | 59.6 | 11.6 | 36.9 |
| old | S2 only | 1358.1 | 247.0 | 47.3 | 15.8 | 13.1 |
| old | **both** | **1314.2** | **235.4** | **91.1** | **27.4** | **49.9** |
| Crossroads | classic | 396.1 | 116.9 | — | — | — |
| Crossroads | **both** | **346.0** | **100.8** | **50.1** | **16.1** | **27.3** |

The two halves overlap on the block scan, so their launch removals are
sub-additive: 59.6 + 47.3 = 106.9 against 91.1 together.

### Timing

Two independent sessions of the four-arm alternating harness, both captures,
palindromic arm order within each capture so a monotone drift cancels. Session 1
ran the fusions opt-in; session 2 ran them as the shipping default with the
`NO_*` reverts on the other arms. `CUDAPIPE_ABUF_FUSE_CHECK` was unset in all
sixteen runs, which `arms.txt` records — the gate synchronises, so a timed run
with it on would silently invert the result.

| capture | arm | session 1 | session 2 | pooled mean of 4 | delta vs classic |
|---|---|---:|---:|---:|---:|
| old | classic | 15.9831 | 15.9561 | 15.9696 | — |
| old | S1 only | 15.8890 | 15.9762 | 15.9326 | +0.0370 (1.1 σ) |
| old | S2 only | 15.8687 | 15.8606 | 15.8647 | **+0.1049 (3.6 σ)** |
| old | **both** | 15.7606 | 15.7370 | **15.7488** | **+0.2208 ms, +1.38% (6.3 σ)** |
| Crossroads | classic | 5.9725 | 5.9644 | 5.9685 | — |
| Crossroads | S1 only | 5.9634 | 5.9476 | 5.9555 | +0.0130 (1.0 σ) |
| Crossroads | S2 only | 5.9410 | 5.9239 | 5.9325 | +0.0360 (2.7 σ) |
| Crossroads | **both** | 5.8619 | 5.8936 | **5.8777** | **+0.0907 ms, +1.52% (6.8 σ)** |

**The pair is worth 0.2208 ms on the old capture and 0.0907 ms on Crossroads,**
reproduced to 0.003 ms across two sessions on old. It lands at the bottom of
the item's 0.20–0.40 ms band and at the top of my own 0.15–0.25 ms forecast.

Which half paid, asked because the halves are independent: **S2 paid and S1
alone did not resolve.** S1 alone is +0.094 in one session and −0.020 in the
other on old — 1.1 σ over four runs — and +0.013 on Crossroads. S2 alone is
+0.105 and +0.036, consistent in both sessions. Yet together they are +0.221,
more than the +0.142 the two halves sum to, while removing *fewer* launches
than the two halves sum to. The pair is what makes the episode's whole
support chain a short static sequence; neither half does that alone. What is
not claimed is a per-launch price: 91.1 launches and 27.4 clears for 0.2208 ms
is 1.86 µs per device operation, well above the under-1-µs launch price,
because the clears that went also carried 49.9 MB/frame and the compaction
pass over 230,400 blocks stopped running.

### Flags

`CUDAPIPE_NO_ABUF_FUSE_SCAN` and `CUDAPIPE_NO_ABUF_FUSE_QUAD` revert S1 and S2
independently; both fusions are on by default. `CUDAPIPE_ABUF_FUSE_CHECK` is
the equivalence gate and `CUDAPIPE_ABUF_FUSE_BREAK` its negative control. The
registry is 97 flags.

Evidence under `/tmp/perf16/iter28-item4/`: `design.md`, `gate-old.log`,
`gate-cross.log`, `gate-old-default.log`, `gate-cross-default.log`,
`gate-old-break{1,2}.log`, `frames/` (sentinel dumps for the fused build and
both breaks), `census-{old,cross}-{off,scan,quad,both}.log`,
`timing1/`, `timing2/`, `ab4.sh`, `ab4_default.sh`.

## The direct path's one-fragment-per-pixel invariant is measured FALSE

Recorded here because it outlives the work that found it. The parked
iteration-28 item 3 (fusing `cp_fs_writeback` into the fragment shader) rested
on the invariant that on the direct path a pixel is claimed by at most one
fragment within a launch. An explicit checked invariant measured it **false on
the old capture: 6,343,098 second claimants of a pixel inside a single
launch.** The item was parked for that reason and for that reason only — the
check existed, so the assumption was measured instead of believed.

This matters beyond that fusion: any later work that assumes a direct-path
pixel is written once per launch is assuming something this capture disproves
six million times a frame-set. Full context is in
`/tmp/perf16/iter28-item3/README.md`, with `wip.patch` beside it; the agent
that produced them has been retired and those two files are the only record.
