# cudapipe performance plan

Correctness is at 14 of 18 samples within a handful of pixels. This is the plan
for the performance pass that follows, ordered by dependency rather than by
size.

Every item names the reference implementation it comes from. That is deliberate:
each of these ideas exists in shipped code somewhere in this tree, in CuRast, or
in cuRE, and reading the original is faster than rediscovering the reasoning.

Where the sources are on this machine:

| reference | checkout | the part worth reading |
|---|---|---|
| Mesa (llvmpipe, panfrost) | this tree | paths of the form `src/...` |
| CuRast | `~/git/CuRast` | `src/kernels/`, paper at `docs/CuRast_arxiv.pdf` |
| cuRE | `~/git/cuRE` | `source/cure/pipeline/`, paper at `SIGGRAPH-2018_cuRE-authorversion.opt.pdf` |

CuRast and cuRE are different codebases whose names differ by one letter; see the
cuRE section for why the distinction matters.

## Ordering, and why — superseded by measurement

This section argued that the phases had to be done in dependency order, 1a
first, because **Phase 0's wins would be unmeasurable until Phase 1a landed**:
if the GPU spends its time idle waiting on the host, making a kernel faster
changes nothing observable, and a profile taken beforehand reports stall time
rather than kernel time.

That was wrong, and the way to find out was to profile first rather than to
argue about whether profiling would mean anything. `tests/cp_profile.sh` exists
for that now. On the three samples the measurement gate asks for, the answer was
the same and it was not close:

| sample | dominant kernel | share of GPU time | `cuCtxSynchronize` |
|---|---|---|---|
| instancing | `cp_rasterize_stage1` | 92.5% | 0.5% |
| particlesystem | `cp_rasterize_stage1` | 92.3% | — |
| bloom | `cp_rasterize_stage1` | 92.3% | 0.5% |

The frame was **kernel-bound from the start**, by a wide margin, and the comment
asserting the GPU is "typically already caught up (99% utilized)" was simply
correct. Phase 1a has not been done and has not been needed; nothing below was
blocked on it.

The general lesson is what the rest of this document now rests on: **an ordering
argument is a hypothesis about where time goes, and such hypotheses are cheap to
test and usually wrong.** Profile, then order.

```
measure ──► 0 (kernel-bound) ──► 1b (in part) ──► 2 ──► 3
```

---

# What has been done, and what it bought

Sixty frames per sample, mean ms per frame, against the same sixty frames stored
and compared to NVIDIA after every step; `tests/cp_iterate.sh` runs both passes
and the comparison. No verdict against NVIDIA changed at any point, the two
standing regressions (`gltfscenerendering`, `texture3d`) included.

| | baseline | after the first pass | after phase 1a | after instancing |  |
|---|---|---|---|---|---|
| **total over the sweep** | **3792.15 ms** | **205.3 / 206.7 ms** | **168.10 ms** | **149.83 ms** | **~25.3x** |
| llvmpipe, same sweep | 233.00 ms | 233.00 ms | 232.14 ms | 232.14 ms | — |

Two runs of the final build are quoted because they differ by 0.6%, which is
about the run-to-run spread of the sweep and worth carrying so that a later
change of that size is not read as a result.

cudapipe began 16x slower than llvmpipe over the set and is now slightly ahead
of it, and ahead on seven of the seventeen samples.

The changes, in order, each measured on its own:

1. **The rasterizer had three stages and used one.** `CP_SMALL_THRESHOLD` was
   999999, larger than a 1280x720 frame, so every triangle took stage 1's
   one-thread-per-triangle path and stages 2 and 3 never ran. A full-screen quad
   was two threads walking 921,600 pixels each. Lowering it exposed two bugs in
   the stages that had never executed: neither strided its queue, so stage 2
   dropped everything past its 512th primitive and stage 3 past its 2048th tile,
   and stage 2 bounded triangle ids by the pre-clip count. **2.3x.**
2. **Points were exempt from all of it**, on the grounds that the point-size
   clamp bounded the loop — at 256 pixels a side, to 65,536 pixels on one lane.
   **particlesystem 10.9x.**
3. **The fragment shader had no bounds check**, so every draw ran the whole
   shader body on 1,843,200 threads and the writeback then kept as many results
   as the draw had covered. The median draw in multithreading covers 588 pixels.
   **1.5x overall.**
4. **Per-draw constants were written by the host into managed memory** that the
   next kernel read, so the first warp of each launch stalled on a page coming
   back. They go by DMA into device-only memory now. **1.2x overall.**
5. **The shading buffers were managed too**, and the arena's bump pointer means
   each draw lands on fresh, non-resident pages. The vertex shader paid for it,
   being the first to touch its output. **multithreading 2x.**
6. **A sprite got one warp** however many pixels it covered, leaving each lane
   hundreds of atomicMins to issue back to back. Large points now decompose into
   tiles like any other large primitive. **particlesystem 1.2x.**

Four of those six are one mistake in different places: **work sized to the worst
case the host can compute rather than to what the draw does.** Worth looking for
more of them before starting anything structural.

## What is left, biggest first

**Superseded twice — see `PHASE_1A.md` and `INSTANCING.md`.** The table below is
the state after the first pass; the current one is at the end of
`INSTANCING.md`. The two structural items named underneath it are still open
and are still the right two, which is why the section stays.

| sample | ms | vs llvmpipe | why |
|---|---|---|---|
| particlesystem | 55.11 | 3.5x slower | 260 peel passes a frame; Phase 3 |
| multithreading | 48.39 | **2x faster** | 343 draws each paying full-screen costs |
| instancing | 27.62 | **2.7x faster** | since 7.08; the id arrays, `INSTANCING.md` |
| dynamicuniformbuffer | 22.42 | **20x slower** | launch-bound; see below |
| gltfscenerendering | 19.79 | 1.3x slower | |
| bloom | 12.44 | 3.5x slower | full-screen passes |

Two things stand out, and neither needs Phase 3:

- **`dynamicuniformbuffer` is 20x slower than llvmpipe**, and nothing else in
  the set is off by that margin. Profiled: it is **launch-bound**, not
  kernel-bound. It draws 625 cubes of twelve triangles a frame, and each draw
  costs nine kernel launches and five memsets — 5,635 launches a frame, 10.4 ms
  of host time in `cuLaunchKernel` alone against a 22 ms frame. Its vertex
  shader launches one block for 36 vertices. Sizing the grids down helped by 2%
  and could not help more, because the cost is the number of launches rather
  than their size.

  Two ways out were named here, both structural: merge stages so a draw costs
  fewer kernels, on cuRE's argument for a persistent megakernel; or batch draws,
  which is Phase 3.

  **The first has since been read and closed — see "cuRE" below.** cuRE's own
  measurements have the multi-kernel sort-middle design it competes with
  (CUDARaster, structurally what cudapipe is) faster than the megakernel on
  nearly every scene. The megakernel buys bounded memory and consistency, not
  speed. So batching is the live answer: if primitives from several draws can
  share a binning pass, per-draw cost stops scaling with draw count. Note that
  llvmpipe is 20x faster here for the structural reason that it has no per-draw
  launch at all.
- **Per-draw full-screen work — now Phase 4, and measured.** `cp_fs_interpolate`
  launches one thread per quad of the whole framebuffer on every draw and every
  peel pass, and the fragment shader and writeback launch over `max_pixels`
  regardless of what the draw covers. Worth about 19% on `multithreading`,
  `bloom` and `dynamicuniformbuffer` together.

  Both remedies this section used to propose have since been tried and neither
  is the answer: bounding the interpolator by a device-accumulated draw bounding
  box is **neutral across the sweep** (§4.1), and the visibility-buffer clear is
  already at peak memset bandwidth at 2.78 µs, so clearing less costs a launch
  to save nothing (§4.3). The cost that remains is the fixed grids of §4.2.

---

# Phase 1a — submission and synchronization

Pure cudapipe. No lavapipe changes. This is the phase that makes the GPU
pipeline instead of ping-ponging with the host.

**Done, and the demotion below was correct when written and then went stale.**

The paragraph that stood here said the host syncs were 0.5% of the frame and
the GPU ~100% busy, so none of this was on the critical path. That held for the
profile it came from. The previous pass's 18.4x then made the kernels short
enough that per-launch host cost became visible, and the conclusion decayed
without anything being wrong with the reasoning that produced it — the GPU was
66-80% busy on the launch-heavy samples by the time anyone asked again.

The phase is written up in `PHASE_1A.md`: 206.65 -> 168.10 ms over the sweep,
cudapipe going from 1.13x to 1.38x faster than llvmpipe. 1a.1 and 1a.4 are
done, 1a.2 is done bar two syncs that measured as worth keeping, and 1a.3 is
measured as not worth doing for this workload. The exit criterion is not met.

Two things there are worth carrying back into how this document is read:

- **nsys cannot answer the measurement gate for this driver.** CUPTI adds
  host-side cost to every `cuLaunchKernel` and cudapipe issues thousands a
  frame, so a traced run manufactures the host-side gap the gate looks for.
  `tests/cp_gpu_busy.sh` asks `nvidia-smi` over an untraced run instead.
- **An ordering argument decays.** This section is the second one in this
  document to have been ranked correctly and then overtaken by a change
  elsewhere. Re-measure a phase before trusting where it sits.
- **A kernel can be charged for somebody else's page faults.** The lead this
  phase left — `instancing` at 38% busy with `cp_vertex_fetch` at 57% of GPU
  time — was neither a slow kernel nor a slow host, but four managed buffers
  the host wrote and that kernel happened to touch first. Written up in
  `INSTANCING.md`: 27.02 → 7.08 ms. A kernel whose duration is absurd for the
  work it describes is reporting a migration, and the memory-operations table
  is where that shows.

Line numbers below predate the changes since and are stale.

## 1a.1 Put every launch on a real stream

All 14 `cuLaunchKernel` calls in `cp_context.c` pass `hStream = NULL` — the
legacy default stream. Correctness is fine (everything is ordered), but there is
no infrastructure for overlap, and no way to express "encode frame N while
rendering frame N+1".

Create a stream per context, thread it through every launch and every
`cuMemcpy`/`cuMemset`. The `cuMemcpyHtoD` calls in the UBO and sampler-table
paths (`cp_context.c:678, 1953, 2291, 2363`) should become the `Async` variants
on the same stream.

## 1a.2 Delete the per-frame host syncs

`cuCtxSynchronize()` appears 12 times. Four are debug-only or teardown and can
stay. These are in the frame path:

| site | fires | replacement |
|---|---|---|
| `cp_context.c:1498` (flush) | every submit | stream-ordered event; sync only where Gallium requires it |
| `cp_context.c:1477` (compute dispatch) | every dispatch | free the arg buffers via a stream callback or an arena |
| `cp_context.c:1531` (peel advance) | **every blend layer** | device-side early-out; see below |
| `cp_resource.c:658` (`cp_clear`) | every frame | remove; the clear kernels are already stream-ordered |
| `cp_resource.c:145` (map for READ) | every readback | keep, but it should stop firing once the frame stays on the GPU |
| `cp_resource.c:208, 294, 317` (copy/blit CPU paths) | every readback | removed by 1a.3 |

The peel loop is the worst of these. It launches `peel_advance`, syncs, reads
`cp->peel_any` on the host, and breaks if nothing was composited. For
particlesystem — "tens of additive sprites deep", per the comment at
`cp_context.c:1431` — that is tens of full device drains per draw.

Replace with a device-side counter and either a fixed pass count bounded by the
primitive count (already computed at `cp_context.c:1448`) or a persistent kernel
that loops internally. The host round-trip per layer is the thing to remove; the
peeling algorithm itself is addressed in Phase 3.

## 1a.3 Move the remaining CPU paths onto kernels

These run on the host today and force a device drain plus, under managed memory,
a page migration of everything they touch:

- `cp_resource_copy_region` (`cp_resource.c:183-234`) — always a CPU `memcpy`
  loop, even when it is a straight same-format copy that `cuMemcpy2DAsync`
  handles directly.
- `cp_blit` same-size format conversion (`cp_resource.c:323-338`) —
  `util_format_translate`. This is the readback path an application takes when
  it blits B8G8R8A8 into an R8G8B8A8 staging image.
- `cp_blit` scaling path (`cp_resource.c:340-441`) — per-row unpack, lerp, pack
  in float. This is runtime mip-chain generation; it runs once per level per
  texture.
- `cp_clear_texture` (`cp_resource.c:540-560`) and `cp_clear_depth_stencil`
  (`cp_resource.c:506-537`) — CPU loops, while `cp_clear_render_target` right
  next to them already uses a kernel.

None of these need lavapipe involvement. Format pack/unpack on the device can
follow what `cp_fs.cu` already does for framebuffer writeback.

## 1a.4 Check the timing hooks before measuring

`cp_lap()` and `timing->writeback_ms` (`cp_context.c:739` and nearby) need to be
CUDA-event based. Host-side wall-clock timing around asynchronous launches
measures launch latency, not kernel duration, and will report the same distorted
picture the syncs currently do.

## Exit criteria

A frame issues with no host synchronization between the first clear and the
final readback, and per-kernel timings come from CUDA events.

---

# Measurement gate

Before choosing between Phase 0 and Phase 1b, get per-stage timings for two or
three representative samples — something geometry-heavy, something
fragment-heavy, something blend-heavy (`particlesystem` is the obvious third).

The question to answer is: **is the frame kernel-bound or transfer-bound?**

- Kernel-bound → Phase 0 pays immediately.
- Transfer-bound, or dominated by page-fault stalls → Phase 1b first.

Also worth recording at this point: draws per frame. The capture requirements
list 77 graphics pipelines and 2 compute but no draw counts, and per-draw CPU
cost in `lvp_execute.c` is the one argument that would eventually favour
replacing lavapipe.

---

# Phase 0 — rasterizer inner loop

Pure cudapipe, local edits to `kernels/cp_rasterize.cu`. Every item here comes
from CuRast (`~/git/CuRast`), which is the direct ancestor of cudapipe's
three-stage design and has already paid for these lessons.

**None of the items below has been done, and the rasterizer is nonetheless off
the top of the profile.** What it needed was not a faster inner loop but for the
loop to run on more than one thread: `CP_SMALL_THRESHOLD` at 999999 meant every
triangle in the driver took the single-thread path and the other two stages were
dead code. That is worth remembering when reading the rest of this section,
which is a list of constant-factor improvements to a loop that was being run
with a factor of a thousand left on the table. Measure before taking any of
them; the profile that made 0.1 and 0.2 look attractive was taken when stage 1
was 92% of the frame, and it no longer is.

The threshold is now overridable at NVRTC time with `CUDAPIPE_SMALL_THRESHOLD`
and `CUDAPIPE_MEDIUM_THRESHOLD`, so a sweep of it needs no rebuild.

## 0.1 Specialize on sample count

`cp_rasterize.cu:530` runs

```c
for (int sm = 0; sm < (int)args.num_samples; sm++) {
   float ox, oy;
   cp_sample_pos(args.num_samples, sm, &ox, &oy);
   ...
}
```

in the innermost scope of the hottest loop in the driver: a runtime trip count
wrapping a function call that computes a compile-time constant. For the
overwhelmingly common single-sample case a template specialization collapses the
whole thing.

**Reference:** `CuRast/src/kernels/triangles_visbuffer.cu:33-35` —

> Some compile-time template specializations here because for perf reasons, we
> need each variation of getVertex to be a separately compiled function.
> Branching at runtime may increase rendering duration by a couple of percent.

They emit 14 separate `extern "C" __global__` entry points across index-fetch ×
compression × instancing to avoid a *branch*. cudapipe's case is a loop, so the
cost is higher.

## 0.2 Fold the half-pixel offset into setup

cudapipe computes `cx = (float)px + ox` per pixel, per sample. The alternative
is to subtract 0.5 from the screen-space vertices once during triangle setup and
sample at integer coordinates.

**Reference:** `CuRast/src/kernels/triangles_visbuffer.cu:163-176`, which is
worth quoting because the magnitude is surprising:

> For unclear reasons, specifying a SAMPLE_OFFSET of 0.5f can be up to 40%
> slower compared to offsetting the screen-space coordinates by 0.5f. E.g. in
> Zorah, it can make the difference between 74ms and 102ms per frame.

Safe for watertightness: it is the same affine translation applied to every
triangle, so two triangles sharing an edge still see exactly opposite edge
function values.

## 0.3 Fast division where it cannot affect coverage

cudapipe uses zero fast-math intrinsics. `__fdividef` is safe for the area
reciprocal and the perspective depth reciprocal, because those feed
*interpolation* only — the inside test uses the raw `e0`/`e1`/`e2` values with
the top-left rule (`cp_rasterize.cu:534-536`), so coverage is unaffected.

**Reference:** CuRast uses `__fdividef` in stage 1
(`triangles_visbuffer.cu:244, 247-249, 287`) and deliberately reverts to exact
division in stage 2 (`563-565`, with the fast versions commented out), because
precision matters more once triangles are large. Worth copying that split.

## 0.4 Replace the discard boolean with an early/late-ZS table

Today the entire decision is one bit:

```c
bool retry = cp->fs_shader && cp->fs_shader->uses_discard && ...   /* cp_context.c:1424 */
```

so every discarding shader pays the full multi-pass retry loop. Mali has
cudapipe's exact hazard — fixed-function depth/stencil potentially resolving
before the shader runs — and solves it by classifying the shader instead of
retrying.

**Reference:** `src/panfrost/lib/pan_earlyzs.c:30-90` and
`src/panfrost/lib/pan_earlyzs.h:15-28`. The state is a precomputed lookup table
(`states[2][2][2][MODE_COUNT]`) read inline in the draw hot path, driven by:

```c
bool shader_writes_zs = s->fs.writes_depth || s->fs.writes_stencil;
bool late_update      = shader_writes_zs || alpha_to_coverage;
bool late_kill        = shader_writes_zs;
bool force_early_kill = s->fs.early_fragment_tests;
bool late_coverage    = s->fs.writes_coverage || s->fs.can_discard || alpha_to_coverage;
```

plus `best_early_mode()`, which uses `zs_always_passes` to pick *weak early*
(test early, shader still always runs) over *force early*.

Three gaps this exposes in cudapipe:

1. **`early_fragment_tests` is not honoured anywhere** — no references in the
   driver. A shader carrying SPIR-V's `EarlyFragmentTests` execution mode has
   explicitly declared that depth/stencil runs before shading regardless of
   discard. Those should skip the retry loop entirely.
2. **The always-passes case is free and unused.** If depth test is disabled or
   `depth_func == PIPE_FUNC_ALWAYS`, nothing is being displaced by the early
   resolve, so there is no hazard and no retry is needed even with discard.
   `cp->depth_stencil.depth_enabled` and `depth_func` are already read a few
   lines below the retry decision (`cp_context.c:1028-1029`). This covers UI and
   overlay draws, and many blended draws.
3. **`writes_depth` is not handled at all** — no references to `frag_depth` in
   the driver. If a fragment shader writes `gl_FragDepth`, the visibility buffer
   resolved on the interpolated depth computed before the shader ran, which the
   shader was about to change. That is a correctness gap rather than a
   performance one.

**Measured priority: low for this workload.** All 1001 SPIR-V modules in
`~/git/Vulkan` were disassembled and scanned for `BuiltIn FragDepth` and
`ExecutionMode DepthReplacing`, and the GLSL/Slang sources for `gl_FragDepth`
and the `depth_*` layout qualifiers:

- **`gl_FragDepth`: zero hits**, repo-wide — gap 3 is not live.
- **`EarlyFragmentTests`: 3 modules, all the `oit` sample**, which is not in the
  in-scope list — gap 1 has nothing to fire on.
- **`discard`: one in-scope sample, `gltfscenerendering`.** The entire retry
  path is driven by that sample's alpha-tested foliage, and it needs ordinary
  depth testing, so gap 2 does not apply to it either.

So §0.4 buys nothing measurable on the current sample set. Keep it on the plan
as robustness — a shader writing `gl_FragDepth` would be silently wrong today,
and the classification is the right structure — but do not sequence it ahead of
anything. Re-check if the sample set grows: the scan is
`spirv-dis <f> | grep -E "BuiltIn FragDepth|ExecutionMode .* DepthReplacing"`
over `find shaders -name '*.spv'`.

Corollary worth carrying into Phase 3: with one discarding sample in scope, the
retry path's cost is not what makes Phase 3 worthwhile — the blend peeling is.

## 0.5 Fix the silent geometry drop (correctness, not performance)

`cp_rast_types.h:469-470` caps the queues, and on overflow the triangle is
discarded with no error, no counter, and no log:

```c
if (idx < CP_MAX_NONTRIVIAL) { queue[idx] = tri_id; }   /* else it vanishes */
```

Same for `CP_MAX_HUGE_TILES`. At minimum, count the drops and report them. The
real fix is growable allocation (Phase 1b.4).

**More urgent than it was.** Until the threshold was lowered these queues were
never written at all, so the cap could not be hit; now every non-trivial
primitive in every draw goes through them. The counters are also now clamped on
the read side (`cp_queue_used()`), because they are unclamped `atomicAdd`s that
on overflow count past entries nobody wrote — which turns a silent drop into a
read of uninitialised memory. That clamp makes overflow safe, not visible.

## 0.6 Cheap early-outs worth taking while in the file

- **Tiny-triangle sample-miss cull** — a triangle whose bounding box falls
  between sample positions can be dropped before any per-pixel work.
  Reference: `CuRast/src/kernels/triangles_visbuffer.cu:190-196`.
- **View-space backface cull** — `dot(a_view, cross(ab, ac))` needs no
  projection and no division, and handles negative-determinant (mirrored)
  instances explicitly via `det(worldView) < 0`.
  Reference: `CuRast/src/kernels/triangles_visbuffer.cu:206-226`.

---

# Phase 1b — memory model

This is the phase that touches lavapipe. It is the enabler for Phases 2 and 3,
not an alternative to them: optimal-tiled images are only possible once images
can be device-only, because **an image the CPU can map must have a layout the
CPU can compute.**

## 1b.1 Fix the `pipe_memory_allocation` ABI leak first

`struct pipe_memory_allocation` is never defined — `p_state.h:606` is a bare
forward declaration. It is an opaque handle, minted by the driver in
`allocate_memory` and handed back to `map_memory` / `free_memory` /
`resource_bind_backing`. llvmpipe puts a heap struct behind it; cudapipe makes
the handle *be* the `CUdeviceptr`. Both are legal.

Lavapipe breaks the contract in three places by fabricating a handle out of
llvmpipe's private struct layout:

```c
struct llvmpipe_memory_allocation alloc = { .cpu_addr = mem };
pscreen->resource_bind_backing(pscreen, pres, (void *)&alloc, 0, 0, 0);
```

| site | address source | lifetime |
|---|---|---|
| `lvp_execute.c:283` | `VkDeviceAddress` | **stack local** |
| `lvp_descriptor_set.c:299` | `VkDeviceAddress` | **stack local** |
| `lvp_device.c:2225` | `VK_EXT_external_memory_host` pointer | field in `lvp_device_memory` |

Under cudapipe, `&alloc` — the address of the struct — becomes the buffer base.
For the first two that is a stack address handed to a kernel.

Reached through `VK_KHR_buffer_device_address` (`lvp_execute.c:1296, 1324, 3003,
3032, 3649, 3680`) and `VK_EXT_descriptor_buffer` (`lvp_execute.c:4784, 4886`,
`lvp_descriptor_set.c:1143, 1294`). The capture's extension list requests
neither, so it is latent for the current workload — but lavapipe advertises both
unconditionally (`lvp_device.c:134, 244`).

**Fix — two `pipe_screen` entry points:**

```c
/* Bind a resource to an address the caller already holds. For buffer-device-
 * address and descriptor-buffer resources, where no allocation object exists. */
bool (*resource_bind_backing_address)(struct pipe_screen *screen,
                                      struct pipe_resource *pt,
                                      uint64_t address, uint64_t offset);

/* Import application-owned host memory (VK_EXT_external_memory_host).
 * Released through free_memory. */
struct pipe_memory_allocation *(*import_memory_host)(struct pipe_screen *screen,
                                                     void *ptr, uint64_t size);
```

The third site needs the second hook rather than the first because it is a real
`VkDeviceMemory` that must support bind, map and free — and note that
`lvp_FreeMemory` currently skips both `unmap_memory` and `free_memory` for the
`USER_PTR` case, so anything cudapipe registers there would leak. That needs
fixing in the same patch.

cudapipe implements `resource_bind_backing_address` by validating the pointer
with `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_MEMORY_TYPE)` and rejecting
addresses the GPU cannot reach, and `import_memory_host` with
`cuMemHostRegister` (the `minImportedHostPointerAlignment = 4096` at
`lvp_device.c:1238` already guarantees the required page alignment).

This must land before 1b.2, because once `map_memory` can legitimately return
NULL, the current `(char *)mem + offset` shortcut cannot distinguish "device-only"
from "foreign address" and the ambiguity becomes load-bearing.

With this done, `llvmpipe_memory_allocation` disappears from lavapipe entirely
(the only two references are `lvp_private.h:256` and `lvp_device.c:2225`, both
removed), and cudapipe's allocation struct becomes its own:

```c
struct cp_memory_allocation {
   CUdeviceptr device_ptr;
   void       *cpu_addr;         /* NULL when device-only */
   uint64_t    size;
   bool        device_only;
   bool        host_registered;
};
```

`cp_map_memory` returns `cpu_addr`, which is NULL exactly when the memory is not
mappable — the property falls out of the struct rather than needing a check.

## 1b.2 Split the memory types

`lvp_GetPhysicalDeviceMemoryProperties` (`lvp_device.c:1778`) advertises exactly
one type on one heap, with all four flags:

```c
DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
```

That is correct for llvmpipe, where device-local and host-visible describe the
same malloc'd pages. It is wrong across a PCIe bus, and it is why every cudapipe
allocation is `cuMemAllocManaged` today.

The assumption is baked in at three levels, and all three need touching:

**Memory properties** — make it driver-queryable via a new `pipe_screen` hook.
llvmpipe returns exactly what is hardcoded now; cudapipe returns two types
(device-only, and host-visible/coherent).

**`memoryTypeBits`** — hardcoded to `1` at seven sites: `lvp_device.c:1827, 2381,
2409, 2458, 2491, 2667` and `lvp_device_generated_commands.c:327`. These become
real bitmasks. Host-pointer import (1827) and fd properties (2667) stay
host-visible-only.

**Eager mapping** — `lvp_AllocateMemory` calls `map_memory` immediately in every
branch (`lvp_device.c:2229, 2259, 2271, 2283`), not lazily at `vkMapMemory`. That
must be skipped for device-only types, which in turn means `poison_mem` and
`VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` (`lvp_device.c:2284-2290`) become
driver-side clears rather than CPU `memset`. `lvp_MapMemory2KHR`
(`lvp_device.c:2337`) can assert for device-only memory, since Vulkan forbids
mapping non-`HOST_VISIBLE` memory.

**Lavapipe's own allocations** must keep requesting mappable memory:
`lvp_descriptor_set.c:398` (descriptor sets) and `lvp_pipeline.c:374` (embedded
samplers) both allocate, map, and then write through the CPU pointer while the
GPU reads the same bytes. `pipe_screen::allocate_memory` has no flags parameter
(`p_screen.h:654`), so it needs one.

**cudapipe side:** two allocators (`cuMemAlloc` for device-only,
`cuMemAllocManaged` or `cuMemHostAlloc` for mappable); `map_memory` returns NULL
for device-only; `resource_copy_region` and `blit` dispatch on which side is
which and become `cuMemcpyHtoDAsync` / `DtoHAsync` / `DtoDAsync`.
`cp_resource_create` (`cp_resource.c:77`) also allocates directly and needs the
same decision, taken from `tmpl->bind` and `tmpl->usage` — render targets and
depth attachments device-only, staging mappable.

## 1b.3 Make residency deterministic — tried, and it does not work

The idea was that `cuMemAdvise` gives device residency without changing the
memory model:

```c
cuMemAdvise(ptr, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, dev);
cuMemAdvise(ptr, size, CU_MEM_ADVISE_SET_ACCESSED_BY, CU_DEVICE_CPU);
```

Preferred-location pins pages on the device; accessed-by establishes a direct
CPU mapping so host access maps over PCIe rather than migrating.

**Measured, and it is a large regression whichever allocations it is applied
to.** Over the whole sweep: 84% slower applied to all of them, still 26% slower
applied only to resources, with particlesystem going from 79 to 267 ms a frame
and `dynamicuniformbuffer` from 42 to 123.

The reason is the second line. The driver writes managed memory from the host
constantly — a per-draw argument block, strides and counts, descriptor and
uniform data lavapipe writes straight through the mapping — and pinning the
pages on the device turns every one of those into an uncached PCIe write. That
costs far more than the migrations it avoids.

This is the argument for doing 1b.2 properly rather than taking the shortcut:
the advice cannot tell host-written memory from device-only memory, because the
driver does not currently distinguish them. Where that distinction *has* been
made explicitly — the shading buffers, which no host code touches, moved to a
`cuMemAlloc` arena of their own — the same underlying win was available and
real, and multithreading halved. The residency is worth having; the hint is not
the way to get it.

The reason to do the full split anyway rather than stop here: managed residency
is *emergent*. One unplanned CPU read — a debug path, a query result, an
application mapping something unexpected — silently costs thousands of page
migrations with no error and no log line. Device-only memory turns that into a
hard failure at map time. For something with a frame budget, a property you can
rely on beats one that currently happens to hold. Kernels touching a
non-resident managed page also stall the warp on a host round-trip, which
`cuMemAlloc` memory structurally cannot do.

## 1b.4 Growable queues instead of fixed caps

`CP_MAX_NONTRIVIAL` (1,000,000) and `CP_MAX_HUGE_TILES` (2,000,000) are fixed
reservations that silently drop geometry on overflow.

**Reference:** `src/panfrost/lib/pan_tiler.c:57-69` describes Mali's approach —
the tiler heap is a huge *reservation* with lazy commit, growing on page fault
via the kernel's growable-memory path. CUDA has the same primitive, and CuRast
already wraps it: `CuRast/src/CudaVirtualMemory.h:84-137`
(`cuMemAddressReserve` → `cuMemCreate` → `cuMemMap`).

Note that the huge-tile queue is the one most likely to overflow, and Phase 3's
hierarchical bin sizes is what actually fixes the underlying cause.

---

# Phase 2 — image layout

Requires 1b. Enables the full value of Phase 3.

## What it is worth — the first measurement was wrong, and `ncu` says so

Everything is still pitch-linear: `cp_resource_layout` computes
`row_stride = nblocksx * block_size` with no alignment, and every internal
buffer indexes `y * width + x`.

**The 32% figure below is not a measurement of layout and should not be used.**
It came from pinning every texel fetch in `cp_fetch_texel` to (0, 0), which was
meant to keep the filtering, the decode and the fetch *count* while giving
perfect cache behaviour. It does not: with the coordinates constant, the four
bilinear taps and the anisotropic loop all resolve to the same address, so the
compiler folds them into one fetch. The probe measured **doing fewer fetches**,
not **the same fetches with better locality** — the identical failure recorded
in `tests/TESTING.md` under probes, committed twice in one session.

**What the hardware counters say instead.** `ncu` on `gltfscenerendering`, the
one textured scene, over its fragment shader:

| | |
|---|---|
| **L1/TEX hit rate** | **94.7%** |
| L2 hit rate | 56.4% |
| Memory throughput | 19.4% |
| DRAM throughput | 1.8% |
| Compute (SM) throughput | 11.8% |
| **Achieved occupancy** | **8.8%** |

A 94.7% L1 hit rate is the cache already absorbing whatever 2D locality the
linear layout costs, and 1.8% DRAM throughput is a kernel nowhere near
bandwidth. **There is very little for a swizzle to recover here.** What the
same column does say is that the shader has too few warps resident to hide any
latency at all, and that is a register-pressure problem rather than a layout
one — see "Fragment shader occupancy" below.

This does not close Phase 2 — aligned strides still matter for coalescing, and
the argument that the sample set is not representative of the capture still
stands. It closes the claim that layout was measured to be worth 32%.

The superseded measurement, kept so the mistake is legible:

| sample | baseline | perfect texture locality | |
|---|---|---|---|
| **gltfscenerendering** | 17.18 | **11.65** | **−32.2%** |
| texturemipmapgen | 1.43 | 1.19 | −16.8% |
| texturecubemap | 1.11 | 1.06 | −4.5% |
| particlesystem | 51.90 | 50.76 | −2.2% |
| bloom | 12.01 | 11.91 | −0.8% |
| multithreading | 29.70 | 29.68 | 0.0% |

Read now as what it is: a probe that removed fetches, on the one sample with
enough fetches for that to show. It says nothing about layout.

**The sample set is still not representative, and that argument survives.** The sample set is sixteen
teaching demos and one Sponza; the workload this driver exists for is the
capture, which lists 77 graphics pipelines. `gltfscenerendering` is the only
sample that looks like it. The sweep understates this phase more than it
understates anything else on the plan — but that is an argument for measuring it
against a representative scene, not a number.

## Fragment shader occupancy — a real item, and not a layout one

`ncu` puts the Sponza fragment shader at **195 registers per thread**, which
fits one 256-thread block on an SM: theoretical occupancy 16.7%, achieved 8.8%,
with the SM issuing on 11.8% of cycles and DRAM at 1.8%. Neither compute nor
bandwidth bound — simply too few warps to hide anything behind.

`link_shader_module()` passes no JIT options at all, so the JIT spends
registers freely on instruction-level parallelism. `CUDAPIPE_MAX_REGISTERS`
caps them, and it is a knob rather than a constant because the trade is
per-shader — fewer registers buys warps and costs spills:

| sample | off | 96 | 128 |
|---|---|---|---|
| **gltfscenerendering** | 17.15 | **15.19** | 15.39 |
| instancing | 7.12 | **6.83** | 6.87 |
| bloom | 12.05 | 12.43 | 12.22 |
| particlesystem | 52.07 | 52.19 | 51.94 |
| multithreading | 29.75 | 29.75 | 29.70 |
| dynamicuniformbuffer | 12.31 | 12.28 | 12.32 |

−11.4% on the sample that most resembles the capture, −4.1% on `instancing`,
**+3.2% on `bloom`**, and flat elsewhere. So a fixed cap is wrong. The shape of
the real fix is to compile, read `CU_FUNC_ATTRIBUTE_NUM_REGS` back, and cap only
the shaders whose count is costing occupancy — or emit `__launch_bounds__` from
the backend, which tells the JIT the block size it must fit rather than guessing
a register count.

Worth noting against Phase 4: `main` is the one kernel in the profile with a
grid large enough to fill the device (42 waves per SM against 0.36-0.88 for the
rasterizer stages), so it is the one place occupancy is the binding constraint
rather than grid size.

## Two corrections to the framing above

- **The internal buffers are not where this helps, and swizzling them would
  hurt.** `cp_fs_interpolate` gives quad *q* to thread *q*, so a warp covers 64
  pixels across two rows — two contiguous 512-byte runs of the visibility
  buffer, which is already perfectly coalesced. Stage 1 walks a row per thread,
  which linear also suits. This is analysis, not measurement, but it is the
  reason the probe above moved the texture-heavy samples and nothing else.
- **§2.1 may need much less of 1b than this header claims.** cudapipe's sampler
  reads its own `cp_texture_info`, not `lp_jit_texture`, and Vulkan already
  forbids `vkGetImageSubresourceLayout` on OPTIMAL images — so the layout of an
  OPTIMAL texture is nobody's business but the driver's. The real coupling is
  narrower: cudapipe advertises every memory type HOST_VISIBLE until 1b.2, so
  an application may legally map an OPTIMAL image and write it expecting
  linear. The samples all upload through staging and `vkCmdCopyBufferToImage`,
  which cudapipe controls. That makes 2.1 attemptable before 1b with a known
  risk, rather than blocked on it.

## 2.1 Swizzled layout for `VK_IMAGE_TILING_OPTIMAL`

`cp_resource_layout` (`cp_resource.c:30-56`) computes
`row_stride = nblocksx * block_size` — pitch-linear, tightly packed, zero
alignment. Vertically adjacent texels are `row_stride` bytes apart, so every
bilinear tap touches at least two cache lines, and neighbouring quads get no
vertical reuse. Real GPUs use swizzled or Morton-order layouts precisely so a 2D
neighbourhood lands in one cache line. The same applies to quad writes: a 2×2
pixel block spans two rows.

This is more achievable than it first appears. **Vulkan forbids
`vkGetImageSubresourceLayout` on `OPTIMAL`-tiling images** — it is legal only for
`LINEAR` and DRM-modifier images. Lavapipe's implementation
(`lvp_image.c:476-496`) just forwards to
`resource_get_param(PIPE_RESOURCE_PARAM_STRIDE)`, so the app-visible stride
contract binds only images where linear is required anyway.

So: swizzle `OPTIMAL`, keep `LINEAR` pitch-linear, exactly as real drivers do.

What genuinely stays linear:

- The bindless / descriptor-buffer path. `struct lp_jit_texture`
  (`src/gallium/auxiliary/gallivm/lp_bld_jit_types.h:59-77`) describes an image
  as `base` + `row_stride[]` + `img_stride[]` + `mip_offsets[]`. There is no
  tiling mode or swizzle parameter — pitch-linear is the only thing it can
  express, and lavapipe fills it in at 44 sites in `lvp_descriptor_set.c`.
- `llvmpipe_get_texel_offset` in the sparse-binding path
  (`lvp_image.c:1053`).

The ordinary sampler path is unaffected: cudapipe's shaders go through its own
`cp_texture_info` (`cp_context.c:2317-2363`), not `lp_jit_texture`, so cudapipe
already describes textures in its own terms.

## 2.2 Aligned strides

`cp_resource_layout` currently applies no alignment at all. Aligned rows matter
for coalesced writes independently of swizzling, and would matter for a
fixed-function encoder if one were ever used (the current plan is a CUDA-based
encoder, which removes that constraint but not the coalescing one).

## 2.3 Convert on upload

Layout conversion belongs in the blit and copy kernels from 1a.3, which by this
point already run on the device.

---

# Phase 3 — tiled rasterization

The largest change, and the one every reference design agrees on.

## 3.1 Why this is the convergent answer

| design | binning |
|---|---|
| Mali | hardware tiler, hierarchical bin sizes, tile-local framebuffer |
| llvmpipe | bins per tile, replays each tile's list in submission order |
| CuRast | per-tile sort + one block per tile (translucent path) |
| Nanite | visibility buffer, clustered/binned upstream |
| **cudapipe** | **none** — tiles exist only to split huge triangles |

cudapipe's two worst known behaviours are direct consequences, and both have the
same shape — *the whole screen pays for the worst pixel*:

- **Blend peeling**: N full-screen passes, one per layer
  (`cp_context.c:1428-1444`), with a host sync between each.
- **Discard retry**: N full-screen passes, same structure
  (`cp_context.c:1418-1423`).

The comment at `cp_context.c:1435` already identifies the alternative:

> llvmpipe has no such problem because it never defers: it bins primitives per
> tile and replays each tile's list in submission order, shading and blending
> inline, so ordering falls out of the data structure.

Phase 3 is building that.

## 3.2 The structure

- Bin primitives into per-tile lists.
- One block per tile. The tile's colour and depth live in **shared memory** for
  the duration. *(cuRE tried this and reversed it — see "Where cuRE contradicts
  this document". Its reason may not apply here, but settle it by measurement.)*
- Walk the tile's list in submission order, shading and blending inline.
- Write the tile out once, coalesced.

What this subsumes:

- Blend peeling collapses to a single pass. Depth complexity is paid per tile
  rather than per screen — a tile with 40 overlapping sprites is one block
  iterating 40 primitives, not 40 screen-wide passes.
- Discard resolves inside the tile loop; the retry machinery disappears.
- Framebuffer traffic becomes one store per tile instead of scattered atomics.
- Swizzled layout (Phase 2) becomes natural, since the tile is the unit of both
  storage and work.
- Quad derivatives survive — a tile block has the 2×2 neighbourhood, which is
  the part that would be expected to block this and does not.

## 3.3 Sort key: submission order, not depth

**Reference:** CuRast's translucent path
(`CuRast/src/kernels/triangles_translucent.cu`) is the closest existing
implementation of this data flow:

- 64-bit sort key `tile_y(8) | tile_x(8) | depth(24) | queueIndex(24)`
  (lines 143-159)
- CUB radix sort over the whole queue (hence `CuRast/src/CubSort.cu`)
- `tileRanges[tileID]` gives each tile's `[start, end)` after sorting
- One block per tile, prefetching 256 primitives at a time into shared memory
  (`kernel_stage4_blend`, lines 460-680)

But its **semantics are wrong for a Vulkan driver** and must not be copied:

- It sorts by **depth**, not submission order. Vulkan defines blending by
  primitive order. Additive sprites give the same answer either way, which is
  why it works for their content; `VK_BLEND_OP_ADD` with
  src_alpha/one_minus_src_alpha over coplanar geometry does not.
- One hardcoded blend equation (`rgba += c.rgb*c.a*T; T *= (1-c.a)`), where
  Vulkan has ~19 factors × 5 ops, separate RGB/alpha, per attachment.
- The 4-slot fragment stash exists because the key holds one depth per triangle
  *piece*, not per fragment, so ordering within a tile is approximate. More than
  four overlapping fragments and it blends wrong.
- 24-bit truncated depth keys (`__float_as_uint(depth) & 0xffffff00`).

**The fix is the sort key.** Replace depth with submission index:

```
tile_y(8) | tile_x(8) | submissionIndex(32)
```

The sort becomes exact — submission index has no per-pixel variation — so the
stash disappears entirely, and the ordering is Vulkan-correct by construction.
Take the machinery, not the semantics.

The index field does not need 32 bits; see "Sort-key bit reduction" under cuRE.
Note also that the harder half of primitive ordering — knowing that no earlier
primitive can still arrive — does not exist here, because binning and
rasterization are separated by a barrier. cuRE needs a whole mechanism for it.

## 3.4 Hierarchical bin sizes

cudapipe (following CuRast) splits huge triangles at a **fixed** tile size, so
one screen-covering triangle at 1080p with 64×64 tiles produces 30×17 = 510
queue entries. That is also why `CP_MAX_HUGE_TILES` is the cap most likely to
blow.

**Reference:** `src/panfrost/lib/pan_tiler.c:18-28`. Mali keeps bins at multiple
tile sizes simultaneously (16×16 through 4096×4096) and files each triangle at
the level matching its bounding box:

> The idea behind hierarchical tiling is to use low tiling levels for small
> triangles and high levels for large triangles, to minimize memory bandwidth
> and repeated fragment shader invocations.

Choosing the split tile size from the bounding box turns 510 entries into a
handful, and keeps entries-per-triangle roughly constant regardless of size.

**There is a second answer to this, from cuRE** — one queue entry per (triangle,
rasterizer) with the hit bins re-derived on the consuming side, which caps the
same 510 entries at the rasterizer count without any hierarchy. It is written up
under "Static bin ownership with per-rasterizer queue entries"; it costs a design
decision Phase 3 has not taken, which is why both are recorded rather than one.

Note the tiler itself is fixed-function hardware — there is no binning algorithm
to port from panfrost or panvk, only the concept. `pan_tiler.c` is nonetheless
worth reading in full; it is effectively a design document for this problem.

**Also from that file (`pan_tiler.c:93-126`):** a heuristic for choosing the
minimum tile size, `triangle area ≈ screen area / triangle count` — few triangles
on a large screen means UI compositing, many means a 3D mesh. cudapipe's
`CP_SMALL_THRESHOLD` is a fixed constant today. cudapipe has an advantage Mali
does not: vertex positions are already on the device when stage 1 runs, so the
real distribution can be measured rather than guessed.

## 3.5 Persistent blocks with atomic work claiming

Stage 1 launches `(n + 255) / 256` blocks, one thread per triangle
(`cp_context.c:1464`). A batch containing one huge triangle and 255 tiny ones
runs at the speed of the huge one.

**Reference:** `CuRast/src/kernels/triangles_visbuffer.cu:315-426` (stage 1) and
`442-531` (stage 2). Launch exactly enough blocks to fill the device, then loop
claiming work from a global counter: one `atomicAdd` per block with the mesh
cursor in shared memory for stage 1, one per warp with `warp.shfl` broadcast for
stage 2.

Stage 2 also has a trick worth taking on its own: thread 0 does all the transform
and setup, then ~15 `warp.shfl()` calls broadcast the result, instead of 32 lanes
redundantly computing the same values (lines 460-531).

## 3.6 Fixed-point subpixel edge stepping

The biggest single win in the rasterizer, and the most care required.

cudapipe deliberately re-evaluates edge functions from the vertices at every
pixel rather than stepping by gradients. The reasoning is recorded at
`cp_rasterize.cu:511-521`: stepping in float accumulates rounding, so the two
triangles either side of a shared edge stop seeing exactly opposite values, and
coverage stops being watertight.

That is a real difference in requirements, not an oversight. CuRast steps in
float (`triangles_visbuffer.cu:251-309`) because it renders opaque photogrammetry
meshes where a one-pixel crack is invisible — its README states blending and
transparency are unsupported. A Vulkan driver cannot make that trade.

The resolution is the one cudapipe's own comment names: snap vertices to a
subpixel grid and iterate in integers, the way `src/gallium/drivers/llvmpipe/
lp_setup_tri.c` does. That gives incremental stepping *and* exactness, which is
what hardware does. It removes the per-pixel edge-function evaluation entirely.

**Half of this may be superseded.** With cuRE's coverage masks there is no
per-pixel edge evaluation to step: a tile's coverage comes from three edges by
eight row intersections, and the inner loop iterates set bits. The *snapping* is
still wanted, because the row intersections have to be exact for the mask to be
the arbiter — but the *stepping* has nothing left to step. The conservative
variant recorded under cuRE needs neither, and is the reason that item is not
sequenced behind this one. Note that cuRE's own `snapVertex`
(`PerWarpPatchGeometryStage.cuh:56`) is commented out at its only call site: they
did not ship the snapping either.

## 3.7 Stage 3: consider ray-tracing

**Reference:** `CuRast/src/kernels/triangles_visbuffer.cu:688-800`. Their stage 3
ray-traces huge triangles rather than rasterizing them (`#define RAYTRACE`).
Notably their rasterize path carries a "TODO: Proper perspective-correct
interpolation" while the ray-trace path computes exact depth from
`dot(t·rayDir, viewDir)`. Möller–Trumbore returns barycentrics directly, so
attribute interpolation still works.

Worth evaluating for cudapipe's stage 3, where the same precision problem exists.

**Likely to evaporate.** Under coverage masks a large triangle stops being a
special case — it is one that hits many bins, and each bin costs the same three
edges by eight rows as any other. Do not sequence this ahead of that.

---

# Phase 4 — the fragment stage's fixed costs

**Numbered last and available first.** Nothing depends on this phase and it
depends on nothing; the number is an identity, not a position. Phase 3 subsumes
all of it, which is the argument for taking only the cheap parts until Phase 3
is decided.

This is the "per-draw full-screen work" item from "What is left" above, promoted
to a phase because it is the largest measured item in the driver and had no
number while §0.4 — measured as buying nothing on this sample set — had a full
section. The document was organised by where ideas came from rather than by what
they are worth.

## What it actually is

Three kernels, not one. Every one of them is launched at a fixed size per draw
and per peel pass, and the coefficient of variation across thousands of launches
says how little any of them depends on the draw. On `multithreading`:

| kernel | share of GPU time | avg | CV | launched over |
|---|---|---|---|---|
| `cp_fs_interpolate` | 33.0% | 18,305 ns | 0.21 | every quad of the framebuffer |
| `main` (fragment) | ~10% | 5,211 ns | — | `max_pixels` = 2·w·h |
| `cp_fs_writeback` | 7.5% | 4,076 ns | **0.03** | `max_pixels` = 2·w·h |

A CV of 0.03 is a kernel doing the same amount of work whatever it was asked to
draw, which is the shape `cp_prof_kernels.py` exists to flag.

**The prize is real and it is worth about 19%.** A probe that made
`cp_fs_interpolate` return immediately took `multithreading` from 29.70 to
24.12 ms, `bloom` from 12.01 to 10.06 and `dynamicuniformbuffer` from 12.31 to
11.05. Note what that probe actually measured: with the interpolator inert the
pixel counter stays zero, so the fragment shader and the writeback early-out
too. It is the upper bound for **all three together**, not for the interpolator
alone — which is the point. Bounding one of the three is not worth much, and
that is what §4.1 found out the expensive way.

## 4.1 Bound the interpolator by the draw's bounding box — tried, and it is neutral

The interim answer this document proposed. **Implemented, measured across the
sweep, and reverted.**

Stage 1 accumulates the union of the primitive boxes `setup_triangle` already
clamps to the clip rectangle — complete by construction, since every primitive
passes through stage 1 whether it rasterizes there or is queued onward, and a
point's box is its full sprite square. Four `uint32` as
`[min_x, min_y, ~max_x, ~max_y]`, so all four are `atomicMin` and all four
initialise from one `cuMemsetD32`; a shared-memory reduction makes it four
atomics per block rather than four per primitive. The interpolator then drops
any quad the box does not meet, which is **exactly output-neutral**: outside the
box every visibility entry is still EMPTY, so the quad search would return
`ntris == 0` after reading four cache lines to find out.

It works, and it does not pay:

| | baseline | with the box |
|---|---|---|
| multithreading | 29.70 | 29.67 |
| dynamicuniformbuffer | 12.31 | 12.32 |
| bloom | 12.01 | 12.00 |
| particlesystem | 51.90 | 51.58 |
| gltfscenerendering | 17.13 | 17.12 |
| instancing | 7.01 | 7.09 |

The kernel does get faster where the box is tight —
`dynamicuniformbuffer`'s `cp_fs_interpolate` fell 12,295 → 8,645 ns, a 30% cut —
and the frame did not move, because that sample is launch-bound and its GPU time
is not what binds it. Where a sample *is* GPU-bound the box is not tight enough:
`multithreading`'s interpolator fell only 18,305 → 16,940 ns with a median
launch unchanged at 17,440.

**The reason is the one this document should have caught before writing the
item.** It justified a bounding-box fix with "a median draw covering 588
pixels" — but coverage and bounding box are different quantities, and a draw
that covers 588 pixels spread along an object's silhouette has a bounding box
far larger than 588 pixels. Nothing in the sample set is both GPU-bound and
spatially compact.

Two probe failures on the way, both worth not repeating:

- **A probe that hardcoded a tiny box measured nothing.** Overwriting the four
  loaded values immediately after loading them made the loads dead, NVRTC
  removed them, and the "tiny box" run was really the "no box at all" run. It
  showed a 17% win that did not exist. A probe that changes what the compiler
  can prove is not measuring the thing it names.
- **A 13.7% regression on `gltfscenerendering` was an artefact of comparing to a
  number from an earlier session.** Rebuilding the previous commit and measuring
  it in the same session gave the same 17.1 ms. `TESTING.md` already says to run
  a sample twice against itself before attributing a delta; it applies to
  baselines carried between sessions just as much.

Keep the idea recorded rather than repeated. If Phase 3 does not happen, the
box becomes worth revisiting only together with §4.2, since neither is worth
much alone.

## 4.2 Size the fragment and writeback grids by the pixel count

The floor §4.1 could not reach. Both kernels launch `max_pixels` = 2·w·h threads
— 1.8 million at 1280x720 — and each thread reads the device-side counter and
returns if it is past it. `cp_fs_writeback` costs 4 µs a launch at a CV of 0.03
doing essentially nothing.

The count is on the device, which is why it is launched this way: sizing the
grid from it needs the host to know it, and reading it costs a sync. The ways
out are a device-side launch, a CUDA graph whose launch dimensions are updated
on the device, or the structural answer in §4.4. This is the item that actually
holds the 19%.

## 4.3 Bound the visibility-buffer clear — measured as not worth it

The other half of the original item, and the cheaper-looking one. **It is not
where the time is.** The clear is 7.4 MB once per draw and costs **2.78 µs**,
which is peak memset bandwidth on this card; a kernel launch to clear less costs
about the same as the memset it replaces. Measured across the set the visbuf
clears are 0.85–4.0 ms a frame traced, and every alternative to them is a launch.

For contrast, and this is where the memsets did pay: the largest memset cost in
the driver was a **64-byte** blocking `cuMemsetD8` at 31 µs, eleven times the
7.4 MB one. See the "zero small managed allocations on the host" commit — the
count told the wrong story and the duration told the right one.

## 4.4 Quad coverage masks — the end state

Moved here from "Definite, but not next" below, because it is the answer to
§4.1, §4.2 and §4.3 at once rather than a separate idea. cuRE paper §5.3: derive
a quad mask from the pixel mask with a few bit operations, start four
consecutive lanes per set bit, suppress helper-thread writes with the original
pixel mask, take derivatives by `__shfl` between the four lanes. There is then
no full-screen pass, no separate interpolate kernel, and no fixed grid to size —
the work list is the coverage.

It is still not next, for the reason given there: wiring it into a standalone
interpolate kernel is plumbing that Phase 3 deletes. What has changed is that
the interim answer this document offered instead has now been measured and does
not work, so there is no cheap version to do first.

---

# cuRE — what to take, and what it settles

Kenzel et al. 2018, read in full for this section. The checkout is `~/git/cuRE`:

- **`source/cure/pipeline/`** — the pipeline itself, and everything cited below.
  Bare `.cuh` filenames in this section are relative to that directory.
- **`SIGGRAPH-2018_cuRE-authorversion.opt.pdf`** at the repository root — the
  paper. Section and figure numbers below refer to it.
- `source/cure/shaders/`, `source/CUDARaster/`, `source/FreePipe/` — the shader
  set, and plugin implementations of the two designs the paper compares against.
- `source/cure/pipeline/config.h.template` and `gpu_configs.h` — the knobs, and
  worth a look on their own: the pipeline is compiled per GPU launch
  configuration, which is the part of the design least applicable here.

**It is not CuRast.** The names differ by one letter and the two are independent
codebases that reach different conclusions: CuRast (`~/git/CuRast`) is cudapipe's
direct ancestor and the source of the three-stage design, cuRE is a persistent
megakernel that argues against exactly that decomposition. A reference to one is
not a reference to the other.

## What it settles without any work

The megakernel is not the answer to the launch-bound sample. Table 3 of the
paper has CUDARaster — a multi-kernel sort-middle design, structurally what
cudapipe already is — faster than cuRE by 1.5–4x on nearly every scene. cuRE's
stated wins are **bounded memory** (Fig. 10: CUDARaster and Piko exhaust device
memory on large scenes, cuRE does not), **consistency** (the only approach that
completed every scene at every resolution), and primitive-order correctness. §3.1
is explicit that the cost is provisioning every SM for the most expensive stage.

That closes one of the two options offered for `dynamicuniformbuffer` above, and
it is the second time in this document that an appealing structural argument did
not survive contact with the numbers behind it.

## Definite

**Conservative coverage masks in stage 3.** `cp_rasterize.cu:834-869` walks every
pixel of every tile evaluating three edge functions per pixel per sample,
whether the tile is fully covered, barely covered, or clipped to a corner. cuRE
builds the whole tile's coverage as a bitmask instead
(`TileRasterizerMask.cuh:53-78`, paper §5.2 and Fig. 8): for
each row of the tile, intersect each edge with that row and construct the row's
bits by shifting an all-ones mask. Three edges by eight rows is 24 intersections
for the tile, against 192 edge evaluations today — and in the partial case the
uncovered pixels are never touched at all. The bin rasterizer
(`BinRasterizer.cuh:66-95`) is the same construction one level up.

**Take the rejection, not the arbitration.** cuRE's mask *is* the coverage
decision, which is why it carries a `0.008f` fill-convention fudge
(`TileRasterizerMask.cuh:66`) and is not watertight — the property
`cp_rasterize.cu:511-521` exists to protect. Adopting that version gates the item
on §3.6's fixed-point snapping. It does not have to: round the row intersections
**outward** and use the mask only to reject, leaving the existing exact edge test
as the arbiter for pixels that survive. Coverage is then bit-identical to today
by construction, most of the saving remains, and the item lands against the
current rasterizer with no prerequisite.

**Vertex reuse in the geometry stage.** `cp_vertex_fetch.cu:13` runs one thread —
and one vertex shader invocation — per *index*. For an indexed closed mesh
indices are about 3T and unique vertices about T/2, so the shader runs up to six
times per vertex. cuRE deduplicates inside a warp before shading
(`PerWarpPatchGeometryStage.cuh:190-226`, paper §4): 32 lanes take a 96-index
batch, a 32-iteration ballot loop reduces it to at most 32 distinct vertices, the
shader runs once each, and triangles are reassembled with `__shfl`. Triangle
order is preserved by construction (`:255`), so nothing downstream changes.

Two scope limits to know before measuring: it does nothing for non-indexed draws,
and nothing for the pre-expanded topology path (`args.vertex_ids` set — strips,
fans, points). `gltfscenerendering` is where it should show. The redundancy is
certain; whether removing it moves the frame depends on the vertex shader's
current share of GPU time, which this document's own rule says to measure rather
than argue about. One `tests/cp_profile.sh` run answers it.

The one coupling is with the output layout, not with anything in Phase 3: the VS
output array becomes indexed per distinct vertex rather than per assembled
vertex, so the rasterizer's `num_varyings + 1` stride needs an indirection.

**Sort-key bit reduction.** Small and certain, for whenever §3.3 happens. The
proposed `tile_y(8) | tile_x(8) | submissionIndex(32)` key does not need 32 bits
of index. cuRE remaps ids into the live window so the radix sort runs on about
ten bits (`rasterization_stage.cuh:267` and `:301`). Fewer passes, same order.

## Definite, but not next

**Quad coverage masks and helper threads** (paper §5.3). The right end state for
the per-draw full-screen work named above, and needed inside Phase 3's tile loop
regardless: derive a quad mask from the pixel mask with a few bit operations,
start four consecutive lanes per set bit, and suppress helper-thread writes using
the original pixel mask. Derivatives then come from `__shfl` between the four
lanes, with no full-screen pass and no separate interpolate kernel.

It is not next because wiring quad masks into a standalone interpolate kernel is
plumbing that Phase 3 then deletes. Right mechanism, wrong moment.

**The "cheaper interim answer" this entry used to cite is gone.** It was the
device-accumulated draw bounding box and clearing visbuf from `pixel_list`; both
have since been measured, and §4.1 and §4.3 record why neither pays. So the
choice is now between this and §4.2 rather than between this and something
cheap — see Phase 4.

**`BlockWorkAssignment`** (`work_assignment.cuh:20-82`). Every thread reports a
work count, a block-wide prefix sum hands out exactly NUM_THREADS work items
across all of them, pull, repeat until drained. It replaces the
`CP_SMALL_THRESHOLD` / `CP_MEDIUM_THRESHOLD` dispatch with one uniform loop, and
it is a better answer than §3.5's atomic work claiming for two reasons: no
atomics, and **a prefix sum is order-preserving**, so primitive order falls out
of the mechanism instead of being reconstructed afterwards. Its companion is the
*encode only, expand in next stage* rule (paper §5.1): a thread writes a
fixed-size record saying how many threads the next stage needs, never the
expansion itself — which is the structural fix for §0.5's silent drops and
§1b.4's growable queues, since the expansion is never stored.

Not next because its value is as the inside of Phase 3's tile loop. Retrofitting
it onto the current three-kernel dispatch would be restructuring a stage that is
no longer at the top of the profile.

## Optional exploration

**Static bin ownership with per-rasterizer queue entries.** The highest upside
here and the least certain. cuRE assigns bins to blocks by a fixed pattern
(`BinTileSpace.cuh:500-666`) and files **one queue entry per (triangle,
rasterizer)** rather than per tile; the consuming block re-derives which of its
own bins the triangle hits from the bounding box (`numHitBinsForMyRasterizer`,
`getHitBinForMyRasterizer`), so `MAX_TRIANGLE_REFERENCES` is just the rasterizer
count — 40 on their GTX 1080 configuration.

That is a second answer to §3.4. The worked example there — a screen-covering
triangle at 1080p producing 30x17 = 510 entries, `CP_MAX_HUGE_TILES` being the
cap most likely to blow — becomes at most 40 entries, with no hierarchical bin
sizes and no Mali-style multi-level tiler. Note cudapipe would not need a
megakernel for this: a persistent rasterizer kernel with fixed bin ownership is
an ordinary launch.

It stays optional because it forks Phase 3 away from "one block per tile". The
entry can name a rasterizer only if that rasterizer is guaranteed to process it,
which dynamic per-tile assignment cannot promise. And the pattern arithmetic is
~170 lines of dense index math that has to be ported and proven before it teaches
anything. Prototype only once Phase 3's shape is otherwise settled.

**In-block blend ordering without locks** (`StampShading.cuh:66-141`). Each
fragment searches backwards for the nearest earlier fragment hitting the same
tile with an overlapping mask, records that predecessor, and blends when the
predecessor signals done; §5.4 notes that with no conflicts this completes
without any serialization. Conditional on decisions not yet taken — it only pays
if the load balancer above is adopted *and* blending happens inline.

Worth recording that these two are coupled: `BlockWorkAssignment` fills the block
from several primitives at once, which is the point of it, so two threads in one
round can reach the same pixel from different primitives and in-block order has
to be resolved explicitly. Taking the load balancer into a blending path means
taking this as well. It does not bite depth-tested opaque draws, which still
resolve by `atomicMin` — but the blend path is what Phase 3 exists to fix.

## Read, do not port

**The progress queue** (`progress_queue.cuh`, paper Fig. 6). A bitmask over
primitives that have finished geometry processing, giving a watermark below which
the input queue is known complete, so a rasterizer can tell "sorted" from "safe
to consume". Genuinely clever and genuinely unnecessary here: it answers "can an
earlier primitive still arrive", and cudapipe's barrier between binning and
rasterization makes that unaskable. cudapipe gets the easy half of §3.3 for free.

## Where cuRE contradicts this document

§3.2 says the tile's colour and depth "live in **shared memory** for the
duration". cuRE tried small tiles held in shared memory and backed out: §5.4
reports that in a streaming pipeline work arrives gradually, so small tiles leave
most rasterizers idle waiting for input, and they were forced to give each
rasterizer many bins and blend **in global memory** — with blending then
dominating their runtime.

cudapipe is not fully streaming. The barrier between binning and rasterization
means a tile's whole list is known before its block starts, which is exactly the
condition cuRE lacked, so §3.2 may well be right. But this is the one place where
a reference design tried what is planned here and reversed it, and that belongs
next to the plan rather than being rediscovered later. It is a measurement, not a
borrow.

---

# Explicitly not on the plan

**Replacing lavapipe with a native driver.** Nothing above requires it. The three
arguments that would favour it are each defused for this workload:

- **Barriers** — a CUDA-based encoder on the same stream gets render→encode
  ordering implicitly. No barrier information is needed to sequence them.
- **Memory types** — Phase 1b fixes this inside the current architecture.
- **WSI forcing CPU images** — `lvp_init_wsi` sets `sw_device = true`
  (`lvp_wsi.c:42-46`), which makes every swapchain image a `WSI_IMAGE_TYPE_CPU`
  image (`src/vulkan/wsi/wsi_common_headless.c:485`). Irrelevant with no
  swapchain.

What remains is per-draw CPU cost in `lvp_execute.c`, which interprets
`vk_cmd_queue_entry` records at submit time through 6127 lines of handler
dispatch. That cost does not shrink as the kernels get faster.

**Revisit if** the measurement gate shows per-draw CPU time is binding. Note
that the kernels and the NIR→PTX backend port to a native driver unchanged —
lavapipe is the part that would be discarded, and it is the part that was not
written here.

One coupling worth recording even though it is not actionable: lavapipe writes
descriptor sets directly in llvmpipe's JIT ABI (44 `lp_jit_*` call sites in
`lvp_descriptor_set.c`), so any driver behind lavapipe inherits llvmpipe's
descriptor layout. Unlike the memory model, that has no clean incremental fix.

---

# References

**CuRast** — Schütz, Lipp, Kristmann, Wimmer; *CuRast: Cuda-Based Software
Rasterization for Billions of Triangles*, Computer Graphics Forum 2026.
`~/git/CuRast`, paper at `docs/CuRast_arxiv.pdf`. The direct ancestor of
cudapipe's three-stage design.

**Mali / panfrost** — `src/panfrost/lib/pan_tiler.c` (hierarchical tiling,
growable heap, tile-size heuristic) and `src/panfrost/lib/pan_earlyzs.c`
(early/late depth-stencil classification). Both are in the shared lib used by
panfrost and panvk. The tiler is fixed-function hardware; these files are the
driver's reasoning about it.

**llvmpipe** — `src/gallium/drivers/llvmpipe/lp_setup_tri.c` (fixed-point
subpixel rasterization), `lp_rast.c` / `lp_setup.c` (binned tile replay in
submission order), `src/gallium/auxiliary/gallivm/lp_bld_sample.c` (anisotropic
filtering, already borrowed once).

**Nanite** — Karis, *Nanite: A Deep Dive*, SIGGRAPH 2021. Source of the
visibility-buffer-with-packed-atomic approach cudapipe already uses, and of the
finding that software rasterization beats fixed-function below roughly a pixel
or two per triangle.

**FreePipe** — Liu et al. 2010. First use of `atomicMin` for direct
rasterization without sorting.

**CUDARaster** — Laine & Karras, HPG 2011. Hierarchical software rasterization
with persistent threads and queue-based work distribution.

**cuRE** — Kenzel, Kerbl, Schmalstieg, Steinberger; *A High-Performance Software
Graphics Pipeline Architecture for the GPU*, SIGGRAPH 2018. `~/git/cuRE`, paper
at `SIGGRAPH-2018_cuRE-authorversion.opt.pdf`. A persistent megakernel that keeps
data in registers across pipeline stages, arguing against the multi-kernel
structure cudapipe has.

**Read in full; see the "cuRE" section above.** The architectural argument does
not survive the paper's own Table 3, but four mechanisms inside it do, and the
coverage-mask rasterizer and the vertex-reuse scheme are the two most useful
things found in any reference so far that cudapipe is not already doing. Do not
confuse this with CuRast, which is a different codebase by an overlapping group.
