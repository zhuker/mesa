# The performance pass: what was done, and what it was worth

A record of one pass over `PERFORMANCE_PLAN.md`. It exists because the headline
number and the plan's table of contents do not agree, and the disagreement is
the most useful thing in here: **the sweep got 18.4x faster and almost nothing
from the plan was implemented.**

Commits `1b944740fa1` through `6476255398e`, all on `cudapipe`.

---

## The number

Sixty frames of each sample, the still scenes orbited, timed rather than stored;
mean milliseconds per frame. The same seventeen samples, the same flags, the
same machine, nothing else on the GPU.

| | nvidia | cudapipe before | cudapipe now | llvmpipe |
|---|---|---|---|---|
| particlesystem | 0.1 | 1361.7 | **55.2** | 15.7 |
| bloom | 0.0 | 558.0 | **12.4** | 3.5 |
| gltfscenerendering | 0.1 | 343.9 | **19.9** | 14.8 |
| instancing | 0.1 | 241.7 | **27.2** | 74.3 |
| multithreading | 0.3 | 156.6 | **50.2** | 96.6 |
| pbribl | 0.0 | 135.4 | **3.8** | 2.6 |
| computeshader | 0.0 | 132.9 | **1.0** | 1.0 |
| dynamicuniformbuffer | 0.0 | 128.5 | **22.5** | 1.1 |
| negativeviewportheight | 0.0 | 124.2 | **0.8** | 0.4 |
| texturecubemap | 0.0 | 124.7 | **1.2** | 1.6 |
| texturemipmapgen | 0.1 | 116.3 | **1.4** | 1.8 |
| texture3d | 0.0 | 62.1 | **0.7** | 0.4 |
| texture | 0.0 | 45.5 | **0.7** | 0.5 |
| multisampling | 0.0 | 41.1 | **3.6** | 5.3 |
| triangle | 0.0 | 30.5 | **0.7** | 0.4 |
| pushconstants | 0.0 | 5.7 | **2.4** | 3.3 |
| vulkanscene | 0.0 | 183.1 | **3.0** | 9.7 |
| **total** | **0.9** | **3792.2** | **206.7** | **233.0** |

**cudapipe began 16x slower than llvmpipe over the set and now finishes ahead of
it**, and ahead on seven of the seventeen individually. The handoff's line that
"nothing in the set is faster than llvmpipe" was true when written and is not
any more.

Two runs of the final build gave 206.65 and 205.32 ms. Both are quoted because
they differ by 0.6%, which is about the sweep's run-to-run spread; a later change
of that size is not a result.

### Why the number is believable

`ms_avg` measures recording and submitting a frame, so on its own it would be a
weak claim. The cross-check is `wall_s`, which cannot miss anything because the
process does not exit until `vkDeviceWaitIdle` returns:

|  | before | now |
|---|---|---|
| total wall, all samples | 275.4 s | 58.6 s |
| less the fixed warm-up and start-up floor (~2.7 s x 17) | ~229 s | ~13 s |

And correctness held at every step: `cp_compare_frames.py` against stored NVIDIA
frames, sixty per sample, after every single change. **No verdict changed at any
point**, the two standing regressions included. Nothing was made faster by
drawing less — if anything the opposite, since two of the fixes restored
geometry that was being silently dropped.

---

## Where it actually came from

| step | total ms | saved | share | kind | what |
|---|---|---|---|---|---|
| baseline | 3792.2 | | | | |
| `stage23` | 1675.4 | 2116.7 | **59.0%** | defect | `CP_SMALL_THRESHOLD` 999999 → 128, plus two bugs in the stages it unblocked |
| `points` | 514.0 | 1161.4 | **32.4%** | defect | points took the same size test as triangles |
| `fsbound` | 353.2 | 160.8 | **4.5%** | defect | the fragment shader got the bounds check the vertex shader already had |
| `upload` | 291.0 | 62.2 | 1.7% | optimization | per-draw constants by DMA into device memory |
| `invariant` | 291.2 | −0.2 | −0.0% | optimization | load each argument slot once — **measured at zero, kept for the code** |
| `dscratch` | 223.4 | 67.8 | 1.9% | optimization | shading buffers into device-only memory |
| `pointtiles` | 212.5 | 10.9 | 0.3% | optimization | large sprites to stage 3; stage 2 block size |
| `discardmask` | 209.0 | 3.5 | 0.1% | defect | stop clearing the discard mask twice a draw |
| `s2final` | 206.7 | 2.4 | 0.1% | optimization | bound stage 2's grid by the draw |

**96% of the gain was defects. The plan is a list of optimizations.** An
optimization asks how to do the right work faster; it assumes the work being
done is right. Three times over, it was not — and that is why none of this
appears in the plan, and why the plan is not thereby wrong. The one part of it
executed as written was the measurement gate, and the measurement gate is what
found all three.

### The three defects

**1. The rasterizer had three stages and ran one.** `CP_SMALL_THRESHOLD` was
`999999`; a 1280x720 framebuffer is 921,600 pixels, so every triangle in the
driver fell under the threshold and took stage 1's one-thread-per-triangle path.
A full-screen quad was two threads walking 921,600 pixels each on a 5090. nsys
had stage 1 at 92% of GPU time on instancing, particlesystem and bloom alike,
with one launch of it reaching 299 ms against a median of 9.6 us.

Stages 2 and 3 were unreachable, and being unreachable had never been debugged.
Lowering the threshold exposed two bugs in them at once: neither strided its
queue, so stage 2 took `warp_id` as the queue index against a 512-warp grid and
dropped every primitive past the 512th (stage 3 the same past 2048 tiles); and
stage 2 bounded triangle ids by `args.num_triangles`, which is the count from
before the clipper ran, so clipped geometry was dropped as out of range.

**2. Points were exempt from all of it.** `rasterize_point` returned before the
size test, with a comment reasoning that `gl_PointSize` is clamped so one thread
can afford the loop. The clamp is `CP_MAX_POINT_SIZE`, which is 256 — a side.
The bound it guarantees is 65,536 pixels on one lane. particlesystem draws its
fire as sprites and ran ~300 peel passes a frame over them.

**3. The fragment shader had no bounds check.** The codegen emitted one only for
`MESA_SHADER_VERTEX`. Every draw therefore ran the entire shader body on
1,843,200 threads — sampling textures on uninitialised varyings — and the
writeback then kept as many results as the draw had actually covered. In one
frame of multithreading the median draw covers 588 pixels.

### Why three defects of this size survived

**All three produced pixel-correct output.** The arithmetic was right; only the
amount of work was absurd. No correctness suite can find that, and the frame
sweep was already passing. They are visible only in a profile.

Four of the six substantive changes share one shape, which is the thing to go
looking for next: **work sized to the worst case the host can compute, rather
than to what the draw does.**

---

## Negative results, kept on purpose

**`cuMemAdvise` (plan §1b.3) is a large regression.** Pinning pages on the device
and mapping them for the host: 84% slower over the sweep applied to every
allocation, still 26% slower applied only to resources, particlesystem 79 → 267
ms and dynamicuniformbuffer 42 → 123. The driver writes managed memory from the
host constantly — a per-draw argument block, strides and counts, descriptor and
uniform data lavapipe writes straight through the mapping — and the advice turns
every one of those into an uncached PCIe write, which costs far more than the
migrations it avoids. Reverted.

It is an argument *for* §1b.2 rather than against residency: where the
distinction was made explicitly instead of hinted at — the shading buffers, which
no host code touches, moved to a `cuMemAlloc` arena of their own — the same
underlying win was real and multithreading halved.

**Sizing stage 3's grid from the triangle count is a loss** — texturemipmapgen
17% slower, texturecubemap and computeshader 12%. Its queue entries are tiles,
not primitives, and a full-screen quad is two triangles and 510 tiles, so the
samples with the fewest triangles are exactly the ones whose triangles cover
whole tiles. The same bound on stage 2, whose entries *are* primitives, is a
small win and was kept.

**Loading each argument slot once bought nothing** — 291.02 against 291.23 ms,
inside the noise. Kept anyway: the reloads were redundant, the vertex shader went
from twelve dependent global loads to two, and the result ruled out the
dependent-load chain as the reason the vertex shader was the frame's largest
kernel. Committed as a code change with a number of zero attached, not as a win.

---

## Tooling added

Both are in `tests/`, both exist so that a claim can be re-run rather than
re-argued.

**`cp_iterate.sh LABEL [COMPARE_LABEL]`** — one optimisation iteration. Builds,
times sixty frames, renders the same sixty, compares them to the stored NVIDIA
reference, and prints the cost delta against any earlier iteration. Every number
in this document came out of it. The flags that have to match between the timed
and the stored pass are exactly the ones that are easy to get wrong by hand,
which is why this is a script.

**`cp_profile.sh SAMPLE [LABEL] [FRAMES]`** — Nsight Systems over one sample,
reduced to the three questions the measurement gate asks: which kernel owns the
frame, what the host is doing, and what the memory traffic is. Keeps the
`.nsys-rep` so a later profile can be diffed against an earlier one. The sqlite
export is worth querying directly for per-launch grid sizes and unified-memory
migration counts, which is how several of the findings above were pinned down.

`NCU=1` runs Nsight Compute instead, **but it does not work on this machine**:
`ERR_NVGPUCTRPERM`, because reading GPU performance counters needs a root-level
`NVreg_RestrictProfilingToAdminUsers=0` modprobe option and a reboot. No
occupancy or warp-stall counters until someone sets that. nsys needs no such
permission and answered every question here.

**Rasterizer thresholds are now NVRTC `-D` overrides** —
`CUDAPIPE_SMALL_THRESHOLD` and `CUDAPIPE_MEDIUM_THRESHOLD`. NVRTC compiles at run
time, so they sweep without a rebuild. That is what isolated the stage 2 bug to
stage 2 in a single command.

---

## What is left, biggest first

| sample | ms | vs llvmpipe | |
|---|---|---|---|
| particlesystem | 55.2 | 3.5x slower | 260 peel passes a frame |
| multithreading | 50.2 | **2x faster** | 343 draws each paying full-screen costs |
| instancing | 27.2 | **2.7x faster** | |
| dynamicuniformbuffer | 22.5 | **20x slower** | launch-bound |
| gltfscenerendering | 19.9 | comparable | |
| bloom | 12.4 | 3.5x slower | full-screen passes |

**`dynamicuniformbuffer`, profiled.** Launch-bound, not kernel-bound. 625 cubes
of twelve triangles a frame, nine kernel launches and five memsets each — 5,635
launches a frame and 10.4 ms of host time in `cuLaunchKernel` alone against a
22 ms frame. Its vertex shader launches one block for 36 vertices. Sizing grids
down was worth 2% and cannot be worth more, because the cost is the number of
launches rather than their size. Two ways out and both structural: merge stages
(cuRE's argument for a persistent megakernel is exactly this) or batch draws,
which is Phase 3. llvmpipe is 20x faster here because it has no per-draw launch
at all.

**Per-draw full-screen work.** `cp_fs_interpolate` launches one thread per quad
of the whole framebuffer on every draw and every peel pass, and the visibility
buffer is memset whole — 7.4 MB — just as often, both regardless of what the
draw covers. On multithreading that is 18 ms of 50. The visbuf clear has an
exact answer already available: `pixel_list` holds precisely the entries a draw
made non-empty, so clearing those instead of the buffer needs no host-side
count. The interpolator wants the draw's bounding box, which stage 1 could
accumulate on the device. Neither needs Phase 3.

**A correctness item the calibration turned up.** `texture3d` diverges from
NVIDIA by 3,090 pixels by frame 24 while **llvmpipe matches it exactly** — so
unlike `gltfscenerendering`, whose divergence llvmpipe exceeds, this one is a
cudapipe defect rather than the cost of software rasterization. It is the
clearest correctness lead in the set.

**Queue overflow is now reachable.** Until the threshold was lowered the stage 2
and 3 queues were never written, so `CP_MAX_NONTRIVIAL` and `CP_MAX_HUGE_TILES`
could not be hit. Now every non-trivial primitive goes through them. The
counters are clamped on the read side (`cp_queue_used()`), because they are
unclamped `atomicAdd`s that on overflow count past entries nobody wrote — but
that makes overflow *safe*, not *visible*. Plan §0.5, counting and reporting the
drops, is more urgent than it was.

---

## What was not done

Essentially the whole plan. Recorded precisely so nobody assumes otherwise:

| item | status |
|---|---|
| Measurement gate | **done** — the only plan item executed as written |
| 1a.1 streams | not done; still zero `cuStreamCreate`, every launch on the NULL stream |
| 1a.2 delete syncs | not done; `cuCtxSynchronize` went 12 → 16 (two debug read-backs, one upload fallback) |
| 1a.3 CPU paths to kernels | not started |
| 1a.4 CUDA-event timing | not started — and it is the prerequisite for trusting `cp_lap()` the moment 1a.1 lands |
| 0.1–0.4, 0.6 | none done |
| 0.5 silent geometry drop | partial: overflow made safe, **not** counted or reported |
| 1b.1 ABI leak, 1b.4 growable queues | not started |
| 1b.2 split memory types | partial and driver-internal only — a `cuMemAlloc` arena for scratch. No `pipe_screen` hook, no `memoryTypeBits`, no lavapipe change; Vulkan still advertises one memory type with all four flags |
| 1b.3 `cuMemAdvise` | tried, measured, reverted |
| Phase 2 | not started |
| Phase 3 | a sliver of §3.5: stages 2 and 3 grid-stride their queues. No binning, no per-tile lists, no shared-memory tile |

Phase 0's inner-loop items deserve a note. They were justified by a profile in
which stage 1 was 92% of the frame, and it no longer is — the rasterizer left
the top of the profile without any of them being done. **Re-measure before
taking any of them.**

---

## Reproducing this

```bash
cd ~/git/Vulkan
T=~/mesa/src/gallium/drivers/cudapipe/tests

$T/cp_iterate.sh mylabel s2final     # build, time 60, render 60, compare, diff
BENCH_ONLY=1 $T/cp_iterate.sh quick s2final   # cost only, no frames
$T/cp_profile.sh particlesystem mylabel 6     # where the frame goes
```

Iterations live under `build/iter/<label>/`, frames under `build/iter/_frames/`,
profiles under `build/prof/<label>/`. The three-driver HTML report for the
current state is `build/iter/_report/perf60.html`, built with
`cp_perf_report.py` over `_report/frames` and `_report/bench`.

**Run timing passes one at a time with nothing else on the GPU.** Two passes
sharing the card measure each other.
