# `main`, the 4.99 ms in front of it, and whether the vertex half can merge

Code and existing-data analysis only. Tree `/home/alexzhukov/mesa` at
`e2fea470d04`; **the working tree was not touched, nothing was built, nothing was
run on the GPU, nothing was committed.** The only external tool used was
`gfxrecon-info` reading a capture file (no device, no replay).

Companion to `/tmp/perf-audit/wide_merge_census.md`. Item 3 is answered first
because it outranks the other two, and because its answer changes what items 1
and 2 are worth.

Line numbers are at `e2fea470d04`. Claims about dependence carry a `file:line`.
Anything I could not read out of the source is marked **INFERENCE** and carries
the one-line check that would settle it.

---

## 0. Verdict

**Item 3 — name it.** The idle in front of `main` is not a dependency and not a
launch-width problem. It is the **inter-submit stall**, and it is structural for
one reason that is visible in twenty lines of code:

> **This driver issues device work only from inside `vkQueueSubmit`, on the
> application's thread.** Recording builds an op list and touches CUDA not at all
> (`cpvk_cmd.c:1402-1422`, `:1923-1963`); every launch, copy and clear is emitted
> during the submit call, op by op, in record order
> (`cpvk_device_memory.c:214-348`). So while the replayer is decoding and
> re-recording the next frame, **the driver has nothing queued, by construction** —
> and the previous frame has already been drained.

The frame's first `main` is the first batch's vertex launch (`cp_renderer.c:5571`,
the first launch `cp_draw_execute_batch()` reaches). Nothing on the device has to
finish before it. What has to finish is the **host**: gfxrecon's decode, the
driver's record path, and the submit's own preamble. That is why the gap sits
"in front of `main`" — `main` is the head of every chain, so every host gap lands
there by topology.

**And the driver's own documents already sized it, untraced:** device idle
**3.70 ms/frame** against host non-blocked time **3.48 ms/frame**
(`PERFORMANCE.md` §5.2), with the largest single gap named there —
*"the inter-submit stall (median 2.453 ms)"*. The ceiling agent's **6.33 ms of
idle and 4.20 ms frame-boundary gap exceed the untraced total device idle of
3.70 ms**, so those figures carry tracer inflation of the host side and must not
be used as absolute sizes (§8 instrument caveats, §11.5 R8). The **shape** they
report is right and matches §5.2; the **size** must come from the untraced
instrument.

**So: yes — the wide-launch programme is aimed at ~0.15 ms of chain idle while
3.5 ms sits in the host's own frame-assembly time.** But that 3.5 ms is mostly
*not the driver's to remove* (§1.6): it is the replayer plus the record path, and
the only driver-owned pieces are the ones already on the lead list — the episode
drain (2.07 ms, slope 1.02) and `vkDeviceWaitIdle` (0.514 ms).

**Item 1 — the `CUfunction` blocker is real but small, and it is already
measured.** Every generated shader is a `CUfunction` named `main`
(`cp_nir_to_llvm.c:4494`, looked up at `:4128`), one module per shader per
execution mode, so a merged launch cannot span two shaders. But the merge unit is
the **episode**, not the pass: a pass binds 25–46 distinct shaders over 180–400
draws (`docs/cudavk/history/EPISODES.md:100`), while an episode's ~13 segments
collapse to about **2 shading identities** — that is exactly what §5.1's
**134.6 blended segments → 22.3 FS launches** is measuring (6.0x). **Not fatal.**

**Item 2 — the FS merge is a port, not an invention, and the VS path is
*freer*, not equally free.** What the FS gave up is listed in §3. The one thing
it needed that the VS does not is the **device-side item boundary**: the FS merge
had to bucket and scatter quads by segment and read the per-segment counts back
on the host, which is what the episode drain carries
(`cp_renderer.c:8411-8451`, `:8511-8524`). A vertex launch's item boundary is a
**host-known vertex count**, so the VS merge needs no bucketing kernels, no
scatter and no drain. What it must pay instead is a change in the generated-kernel
skeleton — and §3.4 shows that skeleton already has the exact hook.

---

## 1. Item 3: why there is 4.99 ms of idle in front of `main`

### 1.1 First, a naming hazard in the attribution itself

`cp_nir_to_llvm.c:4494` creates every generated kernel as `LLVMAddFunction(...,
"main", ...)`, and `:4128` / `:3888` / `:4030` look it up by that name. **Vertex
shaders, fragment shaders and all six execution modes are all called `main`**
(`ARCHITECTURE.md` §2.5 lists the modes; they are separate modules, same symbol).
A gap attribution keyed on the kernel *name* therefore cannot tell a vertex
launch from a fragment launch.

§5.1 splits them: **VS `main` 221.5/frame, FS `main` 22.3 + 74.5 = 96.8/frame,
total 318.3**. The reported "~225 launches/frame" matches the **vertex** figure
alone, not the sum. **INFERENCE:** the ceiling agent's `main` is the vertex
population (or a window in which the fragment launches are counted elsewhere).
*Check:* group the trace's `main` launches by CUfunction handle or by grid shape
(VS grids are `MIN((verts+255)/256, 4096)`, `cp_renderer.c:5571-5573`; FS grids
are capped at 4096 too but follow the interpolator) and re-read the per-class
counts against §5.1 before quoting the split.

This matters because the two have different predecessors (§1.2), and only one of
them is the head of a frame.

### 1.2 What must complete before a `main` can start

**Before a vertex `main`** (`cp_renderer.c:5571`), on the same stream:

| predecessor | site | cost |
|---|---|---|
| the visibility-buffer clear, when the draw is not a segment | `:4760-4762`, 7.4 MB per clear at 720p×2 | ~18.4 memsets/frame (`reprofile_stats.md` B.4 lists `:6506` at 18.43) |
| the depth clear, once per pass | `:4764-4765` | once |
| the vertex-input clear, when the fetch is not fused into the shader | `:5350-5351`, *"about 90 MB a frame"* | 9.42/frame (B.4 lists `:5351`) |
| `cp_vertex_fetch`, when the shader declined the fold | `:5353-5355` | 10.4/frame (§5.1) |
| the coalesced upload flush that `cp_launch()` performs | `:496` | 423.31 htod-async/frame at `:464` (B.4) |

That is **device work, not idle** — a few hundred microseconds a frame of fills.
**Nothing else in the driver has to complete before a vertex launch**: it reads
vertex buffers the application owns and an argument block the host just uploaded.
There is no device-side producer.

**Before a fragment `main`** (`:3648`, `:4340`): the interpolator or compactor
(`:3914`, `:4334`), a counter clear (`:3767`, 82.76/frame — the largest single
memset site in B.4), and for the A-buffer path the whole build. Those *are* real
dependencies, and they are back-to-back — which is exactly the "80.7% of chain
kernels start with zero gap" the ceiling agent measured.

**Conclusion: the idle in front of `main` is not waiting for anything on the
device.** It is a period in which the driver has issued nothing.

### 1.3 The structural reason: nothing can be issued outside `vkQueueSubmit`

* Recording allocates an op and copies state; it never calls CUDA.
  `cpvk_op_alloc()` `cpvk_cmd.c:1402-1422`; `cpvk_record_draw_cmd()` `:1923-1963`;
  descriptor snapshots are host memcpys into the command buffer's arena
  (`cpvk_snapshot_set()` `:978-1005`, `cpvk_arena_append()` `:924-977`).
* Submission walks the op list **in record order** and translates each op into
  CUDA calls on the application's thread: `cpvk_device_memory.c:266-347`, with
  draws going to `cpvk_execute_draw_cmd()` (`cpvk_cmd.c:2580-2673`) which batches
  and finally calls `cp_draw_execute_batch()`.
* The submit does **not** drain at the end — completion is published by a worker
  thread that waits on one event (`cpvk_device_memory.c:45-85`, `:350-409`). That
  was already optimised; it is not the problem.

So the device's feed rate is exactly "how often the application is inside
`vkQueueSubmit`". These captures submit **twice per frame**
(`cpvk_device_memory.c:156-158`, and the fps plugin counts 3,022 submits over
1,511 frames).

### 1.4 What the frame boundary actually contains

**There is no swapchain and no present.** `gfxrecon-info` on
`headless_streamer_20260814T155742.gfxr` reports `Application exe name:
./HeadlessStreamer`, **`Total frames: 0`** and an empty `Used resolutions` —
i.e. the capture contains no WSI. So every present-path hypothesis is dead,
including the sw-WSI fence wait at `src/vulkan/wsi/wsi_common.c:2870-2872` and the
X11 CPU blit; none of it runs. (The driver *would* take that path with a window:
`cpvk_device.c:450-462` initialises WSI with `sw_device = true`.)

What is left at the boundary, in order:

1. **the tail of submit N** — `cp_batch_flush()` and any `cp_pass_finish()` still
   owed, then the completion event (`cpvk_device_memory.c:353-378`). Host work;
   the device may still be running.
2. **the application's synchronisation** — `vkDeviceWaitIdle` →
   `cuCtxSynchronize()` plus a wait for the worker to retire every pending submit
   (`cpvk_device_memory.c:640-650`), measured at **2.24 waits/frame, 0.514
   ms/frame, ceiling 0.500 (97.2%), wait-bound 3,339 of 3,390**
   (`PERFORMANCE.md` §5.2b). The smallop census counts **2.34
   `cuCtxSynchronize`/frame** in total (`reprofile_stats.md` B.4), so essentially
   every context sync in the frame is one of these.
3. **the host assembling frame N+1** — gfxrecon decode plus the driver's record
   path, with **an empty CUDA stream**. This is the piece that has no device work
   in it at all and no way to acquire any.
4. **the preamble of submit N+1** — wait semaphores (`:177-188`), a possible
   `cp_scratch_reset()` with `cuMemFree`s (`:198-199`, `cp_renderer.c:603-638`),
   and a **blocking, synchronous** `cuMemcpyHtoD` of every dirty descriptor arena
   (`cpvk_device_memory.c:242-259`) — 1.01/frame, 0.012 ms/frame (§5.2). Only then
   does op 0 run.

**A driver-owned item worth flagging, found while tracing this.** With the
hardware texture cache on (the default), destroying an image, an image view, or
freeing bound memory **drains the whole device**:

| site | what it does |
|---|---|
| `cpvk_image.c:376-379` (`vkDestroyImage`) | `cpvk_DeviceWaitIdle()` **then** `cpvk_texture_cache_image_destroy()`, which does its **own** `cuCtxSynchronize()` (`cpvk_texture_cache.c:933-935`) |
| `cpvk_image.c:580-584` (`vkDestroyImageView`) | same shape; `cpvk_texture_cache_view_destroy()` syncs again at `cpvk_texture_cache.c:797-799` |
| `cpvk_device_memory.c:927-928` (`vkFreeMemory` with bindings) | `cpvk_DeviceWaitIdle()` |

That is **two full device drains per destroyed view**, on the application thread,
at whatever point in the frame the application destroys transient objects — which
in a streaming application is the frame boundary. The census cannot separate these
from the application's own `vkDeviceWaitIdle` calls because both land in the same
function. **INFERENCE:** part of the 2.24 waits/frame is driver-owned cache
invalidation rather than an application request. *Check:* one counter per call
site inside `cpvk_DeviceWaitIdle()`, or simply count
`vkDestroyImage`/`vkDestroyImageView`/`vkFreeMemory`-with-bindings per frame; if
they are non-zero per frame, a deferred-free list keyed on the submit completion
event removes those drains without changing any launch.

### 1.5 The arithmetic that bounds all of it

From `PERFORMANCE.md` §5.2, untraced, host timers only:

| old capture, per frame | ms |
|---|---:|
| frame | 13.21 |
| host blocked in a device wait (16.76 waits) | 9.73 |
| **host issuing (frame − blocked)** | **3.48** |
| device resident (72% busy) | 9.51 |
| **device idle** | **3.70** |

**There is no fourth bucket.** Every microsecond the host is not blocked at one of
the named waits is counted as "issuing", including the record path, gfxrecon's
decode and the submit-time translation. And device idle (3.70) ≈ host non-blocked
time (3.48). The driver's own text says it: *"The device idle is still almost
exactly the host's own command-issue time."*

Three consequences:

1. **The frame-boundary gap cannot exceed 3.48 ms/frame in the untraced driver.**
   A traced measurement reporting 4.20 ms in that gap is measuring a host that the
   tracer slowed down. Re-read it with `CUDAVK_PLAN_STATS` (host timers, no
   profiler — §5.2's own instrument) before sizing anything from it.
2. **79% "in front of `main`" and "80.7% of chain kernels start with zero gap" are
   the same statement.** The chain is back-to-back because it is one dependency
   chain issued in one burst; the gap is between bursts; the first kernel of a
   burst is a `main`. It is a topology fact, not a property of the shader.
3. **The whole device-idle budget is 3.70 ms and the raster chain's share of it is
   the 0.15 ms the ceiling agent measured.** That is the reframe, and it is
   correct.

### 1.6 So what can touch the inter-submit stall, ranked

| lever | mechanism | size available |
|---|---|---|
| **let the host run ahead of the device** — the episode drain | the host blocks 16.76×/frame *inside* the submit, so frame N+1 is never being issued while frame N runs (`PERFORMANCE.md` §5.2, last paragraph) | **2.07 ms/frame at slope 1.02** (`SESSION_HANDOFF.md` §6.1.1) — already the #1 lead |
| stop draining on object destruction | `cpvk_image.c:376-379`, `:580-584`, `cpvk_device_memory.c:927-928` + the two cache syncs | ≤ **0.514 ms/frame**, wait-bound, ceiling 97.2% (§5.2b) — unknown split app vs driver, one counter settles it |
| fewer forced order points | every `vkCmdPipelineBarrier`, event and query becomes `cp_batch_flush_why()` + `cp_pass_finish()` at `cpvk_cmd.c:3425-3429`, and a blended `cp_pass_finish()` contains the drain (`cp_renderer.c:8443`) | this is *why* there are 15.50 episodes closed/frame (`reprofile_stats.md` B.4) — reducing them is the drain lead by another route |
| cheaper record path | `cpvk_snapshot_set()`/`cpvk_arena_append()` per draw, 180–400 draws/frame | unmeasured; **probe:** `tests/cp_cpu_profile.sh` split by symbol set (§5) |
| **launch width** | — | **nothing.** A merged launch changes neither when the host enters the submit nor how long it takes to get there |

**Nothing on that list is a width change, and the largest item on it is already
the session's #1 open lead.** That is the honest answer to item 3.

---

## 2. Item 1: how many distinct shaders are actually live per episode

### 2.1 What the blocker is, exactly

One `CUmodule`/`CUfunction` pair per shader **per execution mode**, selected by
the host before the launch (`ARCHITECTURE.md` §2.5; `cp_nir_to_llvm.c:4128`,
`:4190`; the mode is chosen at `cp_renderer.c:3862-3888` for the direct path and
`:4299-4327` for the A-buffer path). All of them export the same symbol, `main`
(`cp_nir_to_llvm.c:4494`). A merged launch must therefore be **one shader in one
mode**; a group whose members disagree cannot merge and needs a fallback to
per-member launches.

### 2.2 How many identities there are, measured

| quantity | value | source |
|---|---:|---|
| distinct shaders bound in one of the capture's big **passes** | **25–46** over 180–400 draws | `docs/cudavk/history/EPISODES.md:100` |
| distinct shader **pairs** in a sample | 34 | `docs/cudavk/history/VK_NATIVE_DECISIONS.md:1000` |
| blended **segments** per frame | 134.6 raster triples | `PERFORMANCE.md` §5.1 |
| blended **shading groups** per frame (= FS `main`, A-buffer) | **22.3** | §5.1 |
| **segments per shading identity** | **6.0** | 134.6 / 22.3 |
| blended episodes per frame | 9.88 drains | §5.2 |
| **shading identities per episode** | **≈2.3**, and ≈1.8 after removing the ~4 standalone A-buffer draws/frame | 22.3 / 9.88 — **INFERENCE**, see §5 |

The grouping predicate is `vs == vs && fs == fs && num_fs_ubos == num_fs_ubos &&
mode == mode` (`cp_renderer.c:8494-8508`). So: **a pass has 25–46 shaders, an
episode has about two.** The blocker is not theoretical and it is not fatal; it is
a grouping predicate that already exists and already compresses 6:1.

### 2.3 The counter-example is the opaque path, and it is the most useful finding here

§5.1 gives **FS `main` (direct) 74.5/frame** against **74.5 direct raster
triples/frame** — one shading launch per segment, i.e. **the opaque group merge is
firing at about 1:1**. Its key is stricter than the blended one: it additionally
requires equal `ndraws` **and a byte-identical `memcmp` of the fs-UBO rows**
(`cp_renderer.c:7516-7519`), which different materials never satisfy.

The blended path solved exactly that problem and the opaque path did not adopt the
solution: concatenate the members' fs-UBO rows into one group table and give each
range its `row_base` so the shader indexes the concatenated table exactly as it
indexes a single batch's (`cp_renderer.c:8578-8631`, `cp_rast_types.h:807`).

**Porting the blended path's row concatenation to `cp_opaque_finish()` is a
same-file, same-mechanism change that could take FS `main` (direct) from 74.5
launches/frame toward the blended path's 6:1** — ~60 launches/frame of `main`, and
it needs no new kernel, no ABI change and no scheduling change. **INFERENCE** that
the UBO rows are what refuses; *check:* `CUDAVK_NO_SEG_MERGE=1` and compare FS
direct launches/frame — if the count does not move, the opaque grouping is already
merging nothing, and one `fprintf` of `ngroups` vs `nsegs` at `:7526` says why.

### 2.4 The counters that answer item 1 today, without new code

* **`CUDAVK_SHADER_STATS=1`** prints one line per distinct vertex shader with its
  **launch count** — `cp_vs_census_report()`, `cp_nir_to_llvm.c:102-140`, fed by
  `p_atomic_inc(&state->vs->vs_census->launches)` at `cp_renderer.c:5564-5567`.
  That is the distinct-VS population and its launch weighting, directly.
* **`CUDAVK_DEBUG_BATCHDIFF`** names the state field that broke a batch, field by
  field (`cpvk_cmd.c:2188-2212`, `FLAGS.md`). A VS merge across segments is the
  same question as "why did these two draws not become one batch", so this flag is
  the shortest route to the merge's own eligibility census.
* **Missing, one line each:** `cp->plan.groups += ngroups` in `cp_pass_finish()`
  (`:8491`) and `cp_opaque_finish()` (`:7526`), and `cp->plan.blended_segs +=
  nsegs` at `:8264`. That turns every ratio in §2.2 from an inference into a
  reading.

---

## 3. Item 2: what the FS half gave up, and whether the VS half has the same freedom

### 3.1 The five things the FS merge paid

Reading `cp_seg_range` (`cp_rast_types.h:789-809`) and the two finishers:

1. **A narrower merge key.** Members must share both shader binaries, the
   constant-buffer count and the primitive mode (`cp_renderer.c:8494-8508`);
   anything else starts a new group. Non-merging members are not an error, they
   are just more groups.
2. **A per-item indirection in the hot loop.** The interpolator resolves *each
   quad* to its segment by a **binary search** over the range table, keyed on the
   global primitive id (`cp_fs_interp.h:271-281`). That is a per-item runtime cost
   the unmerged launch did not have.
3. **Host-side concatenation of per-draw rows.** The group's members' fs-UBO rows
   are memcpy'd into one staging table and uploaded per group
   (`cp_renderer.c:8589-8631`; buffer `cp->pass_group_ubos`, sized
   `CP_PASS_MAX_SEGS × CP_MAX_BATCH_DRAWS × CP_ARG_UBO_STRIDE`, `:8589-8592`), so
   the shader can keep using the single-batch row indirection unchanged.
4. **Dense addressing downstream.** The composite still resolves per *segment*,
   through `cp_seg_desc` entries synthesised as offsets into the group's dense
   arrays (`:8654-8669`) — the merge was confined to shading and every consumer
   below it kept per-segment addressing.
5. **The expensive one — a device→host readback to find the item boundaries.**
   The quads are bucketed by segment (`abuf_seg_count`, `:8436-8438`), the host
   reads the per-segment counts **in the episode drain**
   (`:8442-8451` — the copy is `CP_ABUF_COUNTERS + nsegs` words), computes
   group-major bases (`:8511-8524`) and only then scatters into the grouped list
   (`abuf_seg_scatter`, `:8547-8548`). **The FS merge is paid for with the
   driver's largest wait.**

### 3.2 The VS path does not need (5), and that is the whole difference

A vertex launch's item boundary is `total_verts` for that segment — a **host**
value, known when the segment is issued (`cp_renderer.c:5571-5573` sizes the grid
from it). So a merged vertex launch needs:

* no bucketing kernel, no scatter kernel, and **no drain** — the host can compute
  the per-item prefix at issue time and upload it with the item table;
* the per-item resolve **once per block** (a search over ≤64 prefix entries) rather
  than once per item, because the mapping from a global block index to an item is
  monotone in the host-built prefix. Compare (2) above, which is per quad.

**So the VS merge is strictly cheaper at run time than the FS merge already
shipping.** Its per-item record is also tiny: the shader takes **one** kernel
argument (§3.3), so the record is `{arg_block_ptr, first_thread}` — 16 bytes,
against `cp_seg_range`'s 48.

### 3.3 The ABI is one pointer, and that is the good news and the bad news

`cp_arg_slot()` (`cp_nir_to_llvm.c:264-295`) is how a generated shader reads
**every** argument: `ctx->kernel_args[0]` is bitcast to `ptr*`, the slot is
`args_pp[index]`, the load is emitted **into the entry block** and marked
`invariant.load`. The host passes exactly that one pointer
(`cp_renderer.c:3588-3589` for fragment, `:5548-5549` for vertex).

Good news: **one pointer per launch is the smallest possible thing to make
per-item.** Bad news: the slot loads are hoisted to the entry block *precisely
because* they are launch-invariant, so making them per-item moves them inside a
loop. The clean form is an outer per-item loop with the slot loads at its top —
still one load per slot per item, CSE'd across the item's whole extent.

### 3.4 The hook already exists, and it was already proven bit-identical

`cp_nir_to_llvm.c:3195-3250` wraps every vertex and fragment shader body in a
**grid-stride loop over a virtual block id**:

* the invocation count is read from **argument slot 0** as a *device-side* value,
  `invariant.load`, with the comment *"Written by the launch before this one, so
  constant throughout this"* (`:3199-3205`);
* `vid = vbid*256 + tid`, and the body exits when `vid >= count` (`:3227-3232`);
* `emit_workgroup_id()` returns the virtual id inside the body (`:396-408`,
  `:3246`), *"so every slot-derived address follows the loop"*.

A merged launch is the same transformation one level up: **`vid` becomes
`(item, local_vid)` and `kernel_args[0]` becomes `items[item]`.** Two facts make
this a port rather than an invention:

* the shader already takes its extent from memory rather than from the grid, so a
  merged launch does not have to know any item's size at launch time — exactly the
  property that made grid-striding safe;
* the same skeleton change was already made once and *"renders bit-identical,
  `oit` included"* (`docs/cudavk/history/EPISODES.md`, "Two measured lessons").

**The cost to respect.** Changing the skeleton changes codegen for **every**
shader — register allocation, occupancy, scheduling. That is the objection that
retired the `interp → fragment shader` PDL link (`/tmp/perf-audit/pdl_landing.md`
§8.2), and it applies here in the same form. The tree's own answer is the
execution-mode mechanism: a merged form is **a seventh `cp_shader_exec_mode`**,
separately linked, selected by the host, so choosing the old one is a real revert
(`ARCHITECTURE.md` §2.5, `nir_to_ptx/cp_nir_to_llvm.h`). That is the landing path,
and it also gives the fallback item 1 needs for groups that do not share a shader.

### 3.5 What a merged VS would be worth, honestly

221.5 VS launches/frame (§5.1) at the blended path's measured 6.0:1 → **≈37/frame,
about 185 launches/frame removed, 14% of the driver's 1,314**. VS `main` is
1.843 ms/frame of kernel time and would not shrink; what shrinks is launch count
and the number of places a host gap can appear. By §1, **that is not where the
idle is**, so this must be justified as a launch-rate experiment
(`SESSION_HANDOFF.md` §6.3), not as an idle fix — and the cheaper opaque-FS row
port in §2.3 should be tried first, because it removes ~60 launches/frame with no
codegen change at all.

---

## 4. What this does to the wide-merge programme

| claim | status after this census |
|---|---|
| the raster stages cannot merge with each other | **unchanged and confirmed** (`wide_merge_census.md` §2.1) |
| the same stage across segments can merge, ~200 items/frame | **unchanged**, but its value must now be argued as launch *rate*, since the chain it lives in is 80.7% gap-free |
| merging attacks the idle | **wrong, and this document withdraws it.** The idle is in front of the first launch of a burst, and no width change moves it |
| the first prototype should be the fill-relaunch merge | **still the cheapest experiment**, but it is now clearly a *launch-rate* experiment with a ≤0.15 ms chain-idle prize, not an idle fix |
| the largest open item is the episode drain | **reinforced by an independent route:** it is the mechanism that stops the host running ahead, which is what keeps the device empty during frame assembly |

**The ranking I would now give the parent:** (1) the episode drain — unchanged,
2.07 ms, slope 1.02; (2) the object-destruction drains in §1.4, ≤0.514 ms, one
counter away from being sized, and a deferred-free list away from being removed;
(3) the opaque-path fs-UBO row port in §2.3, ~60 `main` launches/frame with a
mechanism already in the tree; (4) everything about launch width.

---

## 5. Inferences and their one-line checks

| # | inference | check |
|---|---|---|
| I1 | the traced `main` population is the vertex launches, not VS+FS | group the trace by CUfunction handle, or compare per-class counts against §5.1 (`SESSION_HANDOFF.md` §11.5 R8) |
| I2 | the 6.33 ms idle / 4.20 ms boundary gap carry tracer inflation | re-read with `CUDAVK_PLAN_STATS=1` (host timers, no profiler); §5.2's untraced ceiling on all device idle is 3.70 ms/frame |
| I3 | ≈1.8–2.3 shading identities per blended episode | `cp->plan.groups += ngroups` at `cp_renderer.c:8491` and `:7526`, read from `CUDAVK_PLAN_STATS` |
| I4 | the opaque group merge fires ≈1:1 because of the fs-UBO `memcmp` at `:7516-7519` | `CUDAVK_NO_SEG_MERGE=1` and compare FS-direct launches/frame; unchanged ⇒ it merges nothing today |
| I5 | part of the 2.24 `vkDeviceWaitIdle`/frame is driver-owned cache invalidation | one counter per call site in `cpvk_DeviceWaitIdle()`, or count `vkDestroyImage`/`vkDestroyImageView`/`vkFreeMemory`-with-bindings per frame |
| I6 | the inter-submit stall is host frame-assembly, not a driver wait | `tests/cp_cpu_profile.sh` (perf, DWARF) and split the on-CPU profile into record-path symbols (`cpvk_Cmd*`, `cpvk_snapshot_set`, `cpvk_arena_append`) vs submit-path symbols (`cpvk_queue_submit`, `cp_draw_execute_batch`) vs gfxrecon's own decode |
| I7 | events and queries insert host callbacks into the stream and stall it (`cpvk_cmd.c:3467`, `:3920`; `cpvk_event_callback_run` even blocks on a condvar at `:3440-3444`) | count `CPVK_OP_EVENT_*` and `CPVK_OP_QUERY` per frame in `cpvk_queue_submit`'s op loop; if non-zero, each one is a stream stall of at least a thread wake-up |

---

## 6. What this document did not do

No build, no replay, no profiler, no GPU. The working tree's four modified files
are untouched and nothing was committed. The one external command run was
`gfxrecon-info` reading `headless_streamer_20260814T155742.gfxr`, which opens the
capture file and prints its header — no device, no replay.
