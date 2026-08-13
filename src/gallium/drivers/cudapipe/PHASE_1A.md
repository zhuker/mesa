# Phase 1a: submission and synchronization

A record of one pass over `PERFORMANCE_PLAN.md` phase 1a, the phase the plan
had demoted. It exists for the same reason `PERFORMANCE_PROGRESS.md` does: the
headline number and the plan's reasoning disagree, and the disagreement is the
useful part.

**The sweep went 206.65 → 168.10 ms, and the largest single win was not in the
phase.**

---

## The number

Sixty frames of each sample, the still scenes orbited, timed rather than
stored; mean milliseconds per frame. Same machine, nothing else on the GPU.

| sample | before | after | llvmpipe | |
|---|---|---|---|---|
| particlesystem | 55.17 | **50.14** | 15.70 | |
| multithreading | 50.25 | **29.73** | 96.85 | −40.8% |
| instancing | 27.16 | **27.10** | 73.04 | |
| dynamicuniformbuffer | 22.46 | **17.37** | 1.04 | −22.7% |
| gltfscenerendering | 19.86 | **15.01** | 15.24 | −24.4% |
| bloom | 12.39 | **12.09** | 3.59 | |
| multisampling | 3.62 | **3.55** | 5.31 | |
| vulkanscene | 2.98 | **2.85** | 9.35 | |
| pushconstants | 2.43 | **2.25** | 3.35 | |
| pbribl | 3.81 | **1.66** | 2.66 | −56.4% |
| texturemipmapgen | 1.43 | **1.44** | 1.73 | |
| texturecubemap | 1.18 | **1.13** | 1.58 | |
| computeshader | 1.02 | **0.85** | 1.01 | −16.7% |
| negativeviewportheight | 0.78 | **0.82** | 0.44 | |
| texture | 0.73 | **0.72** | 0.47 | |
| texture3d | 0.72 | **0.72** | 0.38 | |
| triangle | 0.66 | **0.67** | 0.40 | |
| **total** | **206.65** | **168.10** | **232.14** | **−18.7%** |

cudapipe was 1.13x faster than llvmpipe over the set and is now **1.38x**.

Correctness held at every step. `cp_compare_frames.py` against the stored
NVIDIA frames, sixty per sample, after every change: **no verdict changed at
any point**, the two standing regressions (`gltfscenerendering`, `texture3d`)
included.

---

## The plan said 1a was not worth doing. It was wrong, and so was the measure

`PERFORMANCE_PLAN.md` demotes phase 1a on the strength of a profile: "the host
syncs at 0.5% of the frame and the GPU already ~100% busy, so none of this is
on the critical path today."

That was true when written. The 18.4x of the previous pass made the kernels
short enough that per-launch host cost became visible, and the conclusion went
stale without anything being wrong with the reasoning that produced it.

**nsys cannot detect this.** CUPTI adds host-side cost to every
`cuLaunchKernel`, and cudapipe issues thousands a frame, so a traced run
manufactures exactly the host-side gap the measurement gate is looking for. A
trace of multithreading reports the GPU 66% busy and also reports more GPU
kernel time per frame than the untraced frame takes end to end, which is the
tell.

`tests/cp_gpu_busy.sh` exists for this. It asks `nvidia-smi` for GPU
utilisation over a run with no profiler attached, which is the only version of
the number that means anything:

| sample | before 1a | after |
|---|---|---|
| dynamicuniformbuffer | 70% | 82% |
| multithreading | 77% | 84% |
| bloom | 80% | 81% |
| particlesystem | 91% | 94% |
| gltfscenerendering | — | 93% |
| **instancing** | — | **39%** |

The lesson is the plan's own, applied to the plan: an ordering argument is a
hypothesis about where time goes, and it decays. Re-measure before trusting a
phase that was ranked on a profile taken before the last three changes.

---

## Where it came from

| step | total ms | saved | kind | what |
|---|---|---|---|---|
| baseline | 206.65 | | | |
| `hostapi` | 195.83 | 10.8 | **1a.2** | module globals resolved once, not per draw |
| `syncs` | 192.63 | 3.2 | **1a.2** | peel interval, `cp_clear`, compute dispatch |
| `noflush` | 193.33 | −0.7 | 1a.2 | **reverted** — removing the flush drain is a loss |
| `streams` | 190.18 | 2.5 | **1a.1** | one non-blocking stream for the frame path |
| `events` | 190.97 | ~0 | **1a.4** | CUDA-event stage timing (measurement only) |
| `drawrewind` | 168.80 | 22.2 | — | device scratch rewound per draw |

**The largest win, 22 ms of the 38, is not a phase 1a item.** It was found by
the measurement gate that phase 1a's exit criteria call for.

### 1a.2 — the syncs

Three of the five sites the plan lists came out.

**The peel loop** is the one the plan calls the worst, and it is. Reading
`cp->peel_any` is a host read of memory a kernel just wrote, so it drains the
device; particlesystem's fire runs the full `CP_BLEND_LAYERS` passes because
it never converges early, and it paid a drain per layer — around 1100 a frame
— to be told so. The check only ever asks "has this stopped compositing", and
once a pass composites nothing every later pass does too, because `peel_next`
only moves forward. So it is safe to ask less often, provided the flag is
reset once per interval and read as "did any pass in this interval do
anything". The interval doubles from one, so a draw that does not overlap
itself still converges on pass 2 with nothing wasted, and
`CP_PEEL_CHECK_MAX` bounds the overshoot for one that runs to the cap.

**The sampler's module globals** were not on the plan's list and were the
biggest of them. `cp_sampler_table` and `cp_quad_derivs` were resolved with
`cuModuleGetGlobal` and written with `cuMemcpyHtoD` on every draw, to store
sixteen bytes that were almost always the same as last time — and
`cuMemcpyHtoD` is the blocking variant. pbribl halved and gltfscenerendering
fell 17% on that alone.

**`cp_clear`** synchronised at the top of every frame for ordering its own
clear kernels already have.

**`cp_launch_grid`** allocated its argument blocks with `cuMemAllocManaged`
and freed them after the dispatch, which is what its sync was for. They go
through the upload arena now. computeshader −21.8%.

### 1a.1 — the stream, and why the flag is the whole change

All 15 launches, 14 memsets and the upload copy were on the legacy NULL
stream. Moving them to a per-context stream created with `CU_STREAM_DEFAULT`
is a **1.1% regression** — a default-flagged stream still synchronises
implicitly with the legacy stream, and there is nothing left on NULL for that
to order against which host program order does not already order.

`CU_STREAM_NON_BLOCKING` instead: −0.9%/−1.2% over two runs, and
dynamicuniformbuffer −12.2%, which is the launch-bound sample and the expected
shape. Safe because everything still on NULL is a synchronous call —
`cuMemcpyHtoD` to the sampler table, the texture info block and the UBO shadow
copies do not return until the copy is made.

### 1a.4 — timing

`cp_lap()` bracketed each stage with `clock_gettime`, which measured how long
the host spent *issuing* the stage. After 1a.1 that is unrelated to how long
the device spent running it. Events recorded on the stream give device time,
cross-checked against nsys on dynamicuniformbuffer: interp 12 µs against
`cp_prof_kernels`' 10.46, writeback 6 µs against 4.06.

It also shows what it was built to show. `vertex` reads 40 µs where the kernels
in it total about 7 — that gap is the stream sitting idle inside the stage,
which is what being launch-bound looks like from the inside, and the old
measure reported it as time the vertex shader spent running.

---

## The largest win, which the gate found

The gate put **instancing at 38% GPU busy** — the most host-bound sample in the
set and the third largest. Profiling it put 5.3 ms of a 27.6 ms frame in
`cuMemAlloc` and `cuMemFree`: 584 calls over ten frames at 70–108 µs each. No
kernel explains that shape.

The device-only arena was reaching **5.1 GB inside one frame**. `max_pixels` is
twice the pixels in the framebuffer, so a draw's shading buffers are hundreds
of megabytes whatever the draw covers, and the arena carried every draw of the
frame at once. Past `CP_SCRATCH_RECLAIM_BYTES` `cp_scratch_alloc()` drains the
device and frees the overflow arenas, and instancing crossed that line several
times a frame.

Nothing needed the arena to carry more than one draw. Every buffer in it is
written by one kernel of a draw and read by another, and no host code touches
any of it. So the next draw may reuse the memory, for the reason the pass loop
already rewinds on: kernels on one stream are serialized. The managed arena is
deliberately left alone — the host writes into that one while the device is
still reading the draw before it.

multithreading −36.4%, dynamicuniformbuffer −17.1%, **−11.6% over the sweep.**

This is the fourth instance of the shape `PERFORMANCE_PROGRESS.md` names as the
one to look for — **work sized to the worst case the host can compute rather
than to what the draw does.** Here it was not the work but the memory, and the
cost came back as allocator time rather than as kernel time, which is why no
amount of staring at the kernel profile would have found it.

---

## Negative results, kept on purpose

**Removing the flush drain (plan 1a.2) is a regression.** +0.4% over the sweep,
pbribl +14.5%, negativeviewportheight +9.2%, texture +8.5%. Memory stayed
bounded at 3.4 GB of 32 and correctness was unaffected.

Resetting at flush costs a drain and nothing else — the arena rewinds to zero
and the next frame reuses the pages. Deferring it runs `cp_scratch_alloc()`'s
reclaim instead, which frees and reallocates the overflow buffers, and
`cuMemAlloc`/`cuMemFree` are tens of microseconds each against a few for
draining an almost-idle queue. **The drain is not what costs; the reallocation
is.** Doing this properly means making the arena reclaimable without
reallocating.

Note that the per-draw rewind above attacks the same cost from the other side
and won 22 ms, which is the argument for finishing the job here.

**`CU_STREAM_DEFAULT` for the frame path** — see 1a.1.

---

## What was not done

| item | status |
|---|---|
| 1a.1 streams | **done** |
| 1a.2 delete syncs | **done** for the peel loop, `cp_clear` and compute dispatch. The flush drain measured as a loss and stays; the map-for-READ sync is one the plan says to keep |
| 1a.3 CPU paths to kernels | **not done, and measured as not worth doing.** Only `texturemipmapgen` reaches the CPU blit path, 9 times per process at load time for its mip chain. No sample reaches `cp_resource_copy_region` or `cp_clear_texture` per frame. It stays on the plan as robustness and as a prerequisite for Phase 2.3, not as performance |
| 1a.4 CUDA-event timing | **done** |

The phase's exit criterion — "a frame issues with no host synchronization
between the first clear and the final readback" — is **not met**, and two of
the remaining syncs are there on measurement rather than by omission: the
flush drain is cheaper than the alternative, and the peel loop still checks on
an interval.

---

## What is left, biggest first

| sample | ms | vs llvmpipe | |
|---|---|---|---|
| particlesystem | 50.14 | 3.2x slower | stage 3 and stage 2; kernel-bound at 94% |
| multithreading | 29.73 | **3.3x faster** | |
| instancing | 27.10 | **2.7x faster** | **39% GPU busy — the clearest lead in the set** |
| dynamicuniformbuffer | 17.37 | 17x slower | launch-bound; 82% busy |
| gltfscenerendering | 15.01 | comparable | |
| bloom | 12.09 | 3.4x slower | full-screen passes |

**`instancing` is the next thing to look at.** It is the one sample the
per-draw rewind did not help (−2.1%) and it is still 39% GPU busy, so more than
half its frame is the host. Its profile is unlike anything else in the set: 15
draws a frame, `cp_vertex_fetch` at 57% of GPU time with a grid of 17,280
blocks, and a median launch of 2.4 ms. Whatever is costing it is not the
allocator and not the launch count.

**`cp_fs_interpolate` is still the one kernel near the top of every sample**,
and it is a fixed full-framebuffer launch per draw and per peel pass —
18.98 µs on multithreading with a coefficient of variation of 0.20, against a
median draw covering 588 pixels. The plan's "per-draw full-screen work" item,
which needs no phase 3: stage 1 can accumulate the draw's bounding box on the
device, and the interpolator can skip the quads outside it.

---

## Tooling added

**`tests/cp_gpu_busy.sh SAMPLE [FRAMES] [DRIVER]`** — the measurement gate,
asked without a profiler attached, for the reason above. Reports median GPU
busy and says what it implies for phase 1a.

**`tests/cp_prof_kernels.py SQLITE [--frames N]`** — splits an nsys trace by
kernel *and grid size*, which is what makes cudapipe's profile readable: every
compiled shader is a CUDA kernel named `main`, so the vertex and fragment
stages land in one row of `cuda_gpu_kern_sum` and the largest entry in every
profile is uninterpretable. It also flags kernels whose duration does not vary
with the draw, which is the shape of every defect found so far, and prints the
union of GPU busy intervals against wall time.

**`tests/cp_iterate.sh` refuses to overwrite an existing iteration.** This pass
named an iteration `dscratch` on top of the first pass's `dscratch`, and the
earlier bench numbers and sixty frames — cited as a step in
`PERFORMANCE_PROGRESS.md` — were overwritten in place with no warning. The
label that suggests itself for a change is the same one that suggested itself
last time something touched the same code, which is exactly when the record
being destroyed is the one worth keeping. `FORCE=1` to re-run a label on
purpose. This pass's iteration is now `drawrewind`.

**`tests/cp_iterate.sh` refuses to report a cost number if the correctness gate
did not run.** It invoked the comparison with whatever `python3` was on PATH,
while `cp_compare_frames.py` needs numpy and only the repo venv has it — so a
run printed a one-line import error where the verdict table goes and a cost
delta underneath as usual. Both "a sample regressed" and "the check never ran"
exit 1, so the status could not tell them apart. That is how a regression
ships.
