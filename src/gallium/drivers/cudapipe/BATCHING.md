# Draw batching: merging draws that cannot tell the difference

`PERFORMANCE_PLAN.md` has named draw batching as Phase 3's payoff since it was
written, on the reasoning that per-draw cost stops scaling with draw count if
primitives from several draws share a pass. This is the record of doing a narrow
version of it — narrow enough to need none of Phase 3's binning, sorting or tile
loop.

**The sweep went 145.53 → 108.98 ms, and three samples carry all of it.**

---

## The number

| sample | before | after | |
|---|---|---|---|
| dynamicuniformbuffer | 12.34 | **1.29** | **−89.5%** |
| multithreading | 29.78 | **6.04** | **−79.7%** |
| pushconstants | 2.29 | **0.78** | **−65.9%** |
| particlesystem | 52.25 | 52.02 | −0.4% |
| gltfscenerendering | 15.33 | 15.31 | −0.1% |
| instancing | 7.12 | 7.11 | −0.1% |
| bloom | 12.07 | 12.09 | +0.2% |
| **total over the sweep** | **145.53** | **108.98** | **−25.1%** |

Nothing outside those three moves beyond ±1.5%. cudapipe was 1.55x faster than
llvmpipe over the set and is now **2.13x**.

Correctness held. Every verdict against the stored NVIDIA reference is
unchanged, both standing regressions at the identical frame and pixel count —
`gltfscenerendering` 15,697 / 59,939 at frame 47, `texture3d` 2 / 3,090 at frame
24. The four nondeterministic samples moved by 0–5 pixels.

Three independent checks, because a 25% sweep result deserves them:

- **`wall_s` agrees.** It cannot miss anything, since the process does not exit
  until `vkDeviceWaitIdle` returns: 10.0 → 3.4 s and 20.5 → 6.3 s, which net of
  the ~2.7 s fixed start-up is 10.4x and 4.9x — the per-frame figures.
- **`CUDAPIPE_NO_BATCH=1` restores 12.37 and 29.67 ms**, so the win is
  attributable to batching and not to anything else that changed.
- **Six runs of `dynamicuniformbuffer`, including one of 6,000 frames**, give
  1.26–1.28 ms with no spread.

---

## Why it needs none of Phase 3

Phase 3 batches by binning primitives per tile and replaying each tile's list in
submission order, and §3.3 spends its length on the sort key that makes the
replay correct. None of that is needed here, because of a property the driver
already had:

**With blending off, the visibility buffer is order-independent.** It resolves
the nearest fragment per pixel with `atomicMin`, and a minimum does not care in
what order its inputs arrived. So draws whose result cannot depend on their
order can be concatenated and rasterized as one, and the answer is the same.

That is the whole idea. What it costs is a list of conditions under which the
order *could* show, and refusing those draws. `cp_batch_eligible()`:

- blending enabled, or a shader that can discard
- depth test off, or depth write off, or a depth function outside
  {`LESS`, `LEQUAL`, `GREATER`, `GEQUAL`} — a minimum has to mean something
- anything the host expands into a refs table: not `TRIANGLES`, more than one
  draw, instanced, indirect, user indices
- no vertex buffer, no colour attachment
- a uniform binding copied out of a user pointer, since those reuse one device
  buffer and two draws can hold the identical address while meaning different
  bytes — nothing in the key can see that

and `struct cp_batch_key`, compared with `memcmp`, holds everything else that
has to match. It is built from a zeroed struct by one function, so state added
to the driver and forgotten here is simply not a merge condition rather than a
silent bug.

---

## The three pieces

**Geometry.** `cp_vertex_fetch` takes `verts_per_draw` and derives
`local = v % verts_per_draw`, replaying one index range once per merged draw.
This is the same shape as the vertex and instance ids it already derives — see
`INSTANCING.md` — and for the same reason: the answer is a function of the
thread index, so the host should not be building a table of it.

**Per-draw uniforms.** What a batch is allowed to vary is exactly the vertex
stage's uniform bindings, which is what `dynamicuniformbuffer` varies: one
buffer at 125 dynamic offsets, which is what dynamic uniform buffers are for.
The VS argument block now carries a table of those pointers behind it, one row
per draw, and `cp_upload` was split into `cp_upload_begin`/`cp_upload_end` so
the block can hold the device address of a table appended to itself.

**Codegen.** `emit_const_buf_base()` becomes the single place generated code
reads `args[18..]`, and applies the table for the vertex stage only. A batch of
one points the table at `&args[18]` with a divisor of `0xFFFFFFFF`, so
`row = tid / divisor` is 0 and the address computed is exactly the one the
shader used before. **There is no branch and no second shader variant**, and
`CUDAPIPE_NO_BATCH=1` reproduces the unbatched frame byte for byte — which is
the property that makes the change reviewable at all.

---

## Where the win comes from, and it is not what was predicted

`PERFORMANCE_PLAN.md` argued batching would pay by growing the grids, and the
device counters supported it: `SM Issue` 2%, `SMs Active` 12%, stage 2 on five
blocks. The grids did grow — stage 2 from **5 blocks to 512** — but that is the
smaller half.

| | before | after |
|---|---|---|
| kernel launches per frame | 1,127 | **11** |
| memsets per frame | 626 | **5** |
| GPU kernel time per frame | 7.8 ms | **0.23 ms** |

Most of 7.8 ms was **multiplicity, not size**: the visibility clear, the
interpolator, the fragment grid and the writeback are sized to the framebuffer,
and they ran 125 times for one frame's output. Batching deleted 124 of them.
That is the same finding as `CUDAPIPE_DEBUG_WORK`'s 0.09% shading utilisation,
arriving from the other side.

`dynamicuniformbuffer` is now **host-bound at 25% GPU busy** where it was 86%,
and too fast for `METRICS=1` to profile — even 8,000 frames leaves a window
that is mostly idle. What remains is roughly 125 `cuMemAllocManaged` and 130
`cuMemFree` a frame, one pair per draw, from lavapipe's descriptor path through
`pipe_screen::allocate_memory`: about a quarter of host API time, and not
kernels at all.

---

## The hazard is deferral, not merging

Holding a draw back means `cp_draw_execute()` runs **after the next draw's state
has been bound**. That was enough to render `vulkanscene` 12.9% wrong with every
batch of size *one* — the merging was not even involved.

So every state setter flushes the pending batch when the incoming value differs,
and every path that observes rendering flushes before it looks: `cp_flush`,
`launch_grid`, `destroy_context`, `set_framebuffer_state`, buffer and texture
map, `resource_copy_region`, `blit`, all five clear entry points. Twenty-seven
call sites.

**That invariant is what the correctness rests on, and it is not
self-enforcing.** A state setter added later that does not flush will be wrong
silently, and the sample sweep may well not catch it — this one did not catch
`vulkanscene` until a batch cap of one was tried deliberately. The same applies
to `reads_const_bufs`, which is sound only while `emit_const_buf_base()` remains
the only reader of `args[18..]`.

`CUDAPIPE_DEBUG_BATCH=1` prints why each batch ended, `CUDAPIPE_BATCH_MAX=N`
caps the size, and `CUDAPIPE_NO_BATCH=1` disables it. The bit-identical check is
`CUDAPIPE_BATCH_MAX=1`, and it is worth re-running after any change here: it
separates "deferral is sound" from "merging is sound", which are different
questions and fail differently.

---

## One known deviation

With `LEQUAL` and coplanar geometry spanning merged draws, a batch keeps the
lowest triangle index — the earliest draw — where sequential drawing keeps the
later one. No sample in the set shows it.

It is mitigated rather than absent: the clipper compacts its output with
`atomicAdd`, so triangle order is already nondeterministic *within* a draw, and
the tie was never exact. But this is a real difference in what the driver
computes, not a rounding one, and it is the first place to look if a
depth-fighting artefact appears.

---

## What is left

The condition that excludes the most is **identical geometry**: the draw range,
count and bias are in the key, so only draws replaying one index range merge.
That is why `gltfscenerendering` — 22 primitives out of one shared buffer — and
`bloom` never batch at all.

Per-draw index ranges are the next increment and the larger one: a per-draw
table of offsets plus a prefix search in `cp_vertex_fetch`, and per-triangle
draw indices threaded through the clipper for the fragment stage, which this
pass avoided by making identical fragment bindings a merge condition.

**Done for the vertex half — `bloom` 12.07 → 2.16 ms, `vulkanscene` 2.81 → 1.53,
`particlesystem` 5.07 → 4.54, the sweep 61.82 → 49.99.** `cp_vertex_fetch`
binary-searches a per-draw slice table and the range leaves the key. The
per-triangle draw index was **not** needed: instrumenting the key's `memcmp` to
name the differing field says `bloom`'s 298 batch breaks are all `draw_start`
and `draw_count` and nothing else — no state, no fragment binding, no shader.
`bloom` goes from 151 batches a frame to 2, and stage 1's grid from 1–9 blocks
to 193.

**The paragraph above is wrong about `gltfscenerendering`, and the shape of the
error is worth keeping.** It does not fail on geometry; it never reaches the key
at all. All 42 of its batch breaks are the vertex shader, because the sample
creates one pipeline per material and binds 25 distinct VS binaries a frame.
Per-draw index ranges cannot help it and neither can per-draw fragment bindings
alone — it needs shader dedup by compiled content, per-draw cull mode, per-draw
bindings, and its `MASK` materials discard, which the batcher refuses outright.
It is untouched at 15.32 ms and is now the largest sample in the sweep.

`multithreading` batches 44 of its 342 draws before hitting `CP_MAX_BATCH_TRIS`,
which is bounded by the clipper's output buffer being sized at 3x the input
triangle count. Raising the cap from 64K to 256K was worth 6.47 → 6.04 ms;
sizing the clipper from need rather than from the worst case would raise it
further and is independent of everything else here.

Blending, discard, instancing and depth-only passes are refused by design, so
`particlesystem`, `bloom` and `instancing` are untouched. Blended draws are what
Phase 3's tile loop is for, and this pass does not bring it closer.

**Two of those three moved once the ranges came out of the key**, which this
paragraph did not anticipate: each has a handful of *opaque* draws with
different ranges that merge fine. `vulkanscene` −45.6%, `particlesystem` −10.5%.
Only `instancing` is genuinely refused by design.

**And the known deviation below is now reachable.** `bloom` was bit-identical
against itself over 60 frames and differs on 4 of them at 1–2 pixels once its
draws merge; `vulkanscene` goes 1/60 to 5/60. It needs `CUDAPIPE_BATCH_MAX` at 8
to appear and not 2 — enough merged draws for two to collide at equal depth. The
underlying defect is the clipper's output order rather than batching: `atomicAdd`
compaction means primitive order is not stable, which breaks Vulkan's guarantee
independently of any of this. A stable compaction is the fix, and it is worth
its own pass rather than being paid for by merging less.

---

## Reproducing this

```bash
cd ~/git/Vulkan
T=~/mesa/src/gallium/drivers/cudapipe/tests

DESC="..." $T/cp_iterate.sh mylabel batchtest      # build, time, render, compare
CUDAPIPE_DEBUG_BATCH=1 build/bin/dynamicuniformbuffer --offscreen --offscreenframes 2
CUDAPIPE_BATCH_MAX=1 ...                           # must be bit-identical to NO_BATCH
```

**HEAD is what `batchtest` measured.**
