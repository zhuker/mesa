# cudapipe as a native Vulkan driver

The same CUDA rasterizer backend with Mesa's common Vulkan runtime in front of
it, instead of lavapipe and Gallium.

**Branch `cudapipe-vk-native`. Status: milestone 1 of 6 — a physical device is
enumerated. Nothing renders yet.**

## Why, and — first — why not

**Not for the reasons that sound best.** Two capabilities were named as the
motivation and both were measured on the two captures before any code was
written. Both price at zero:

- **Pass-wide draw reordering.** With the whole render pass visible, opaque
  draws could be regrouped by shader, since opaque visibility resolves by
  `atomicMin` and is order-free. Measured with `CUDAPIPE_DEBUG_PASSSEQ` over
  both replays: opaque geometry groups today (consecutive equal `(vs, fs)`)
  versus if freely reordered (distinct `(vs, fs)`) are **6,027 vs 6,027** on
  Crossroads and **32,387 vs 32,387** on the old capture. Exactly equal, in
  every pass, including the worst one in the tree (136 opaque draws → 34 runs →
  34 distinct shader pairs). The applications already submit opaque geometry
  sorted by material. Reordering buys nothing.
- **Immutable samplers known at pipeline-compile time.** Instrumented
  `lvp_CreateDescriptorSetLayout`: of 382 sampler bindings on Crossroads and
  1,088 on the old capture, **zero** are immutable and none is an array. The
  sampler state lives in descriptor writes at run time, which is exactly where
  the existing provenance capture already finds it — and that specialization
  measured 34.47 vs 34.63 ms, about 0.5%, with every broader variant
  regressing.

**And not for host CPU time.** EPISODES measured lavapipe and gfxrecon together
at under 2% of host time. That number wants re-taking now the frame is three to
five times smaller, but it will not become a headline.

**The reason is that a large part of this driver exists only to reconstruct
what Vulkan states and Gallium discards**, and that reconstruction is where the
subtlest correctness constraints live:

| machinery | what it rebuilds | lines |
|---|---|---|
| draw batching + its flush discipline | that a draw is a complete, immutable record | ~550 in `cp_context.c`, 25 `cp_batch_flush` call sites |
| pass episodes + segment snapshots | that a run of draws can be examined together | ~1,350, plus ~50 KB × `CP_PASS_MAX_SEGS` (64) ≈ 3 MB of context state |
| `cp_pass_live_save/restore` | that a draw's bindings are its own | inside the above |
| `cp_pass_fallback` | a second execution path over that snapshot | inside the above |
| small-allocation arena + managed allocation cache | that descriptor sets need not each own a device allocation | ~270 here, more in `cp_resource.c` |

Roughly 2,000 lines of `cp_context.c`'s 9,034, and three of this driver's most
expensive bugs came from that seam: `vulkanscene` 12.9% wrong with batches of
size *one*; a held-back draw launching with the next draw's fragment bindings,
57% of the image wrong; and the clip-convention mismatch (Vulkan `0 ≤ z ≤ w`
against Gallium `−w ≤ z ≤ w`) that blacked out Crossroads frames 633 and 756.

Two further things the layer costs that no hook recovers cheaply: render-pass
load/store ops, which a tile renderer wants, and per-command-buffer parallel
translation — the driver is single threaded on the host and lavapipe's
record-then-replay-on-submit model is why.

So the case is **simplification and host parallelism**, not unlocked
optimisation. Anyone extending this document should keep that distinction; it
is the one the tiling prototype lost.

## Shape

`src/cudapipe/` — a Vulkan driver on `src/vulkan/runtime`, the nvk/radv/turnip
shape. Kept:

- `spirv_to_nir` and the whole NIR→PTX backend, unchanged;
- the kernels, rasterizer, A-buffer, episodes-as-an-algorithm (the *grouping*
  survives; the *deferral* does not);
- `vk_meta` for blit, resolve, clear, copy and fill on top of our own draws;
- `vk_render_pass`, `vk_graphics_state`, `vk_descriptors`, `vk_sync`.

Dropped: `pipe_context` and everything written to satisfy it.

Not needed at all: WSI. This driver is headless, which removes the single
largest chunk of ordinary Vulkan bring-up.

## Staging, and the gate at each step

The rule this project already proved twice — verify a refactor is bit-identical
before layering behaviour on it — is the whole plan:

1. **Factor, no behaviour change.** Split `cp_context.c` into a backend core
   with a Vulkan-shaped internal API and a thin Gallium adapter over it.
   *Gate: 18-sample sweep and capture unmoved.* **Started.** The first slice is
   done and was chosen because its boundary was already almost there:
   `cp_kernels_init()` took a `struct cp_screen` and used exactly two fields of
   it, so it takes `sm_major, sm_minor` now and the file is free of Gallium
   entirely. Both drivers build the same NVRTC kernels from the same sources.
   Same for `cp_compile_sampler_variant`.

   That slice also found a real bug in the native driver: `cp_debug_init()` was
   never called, so all 59 environment switches read as their zero value rather
   than what the environment asked for. `CUDAPIPE_HELP=1` now prints the same
   table from either ICD.

   **Second slice: the draw description.** Measured before touching anything —
   `cp_draw_execute()` is 1,973 lines and touches Gallium on **sixteen** of
   them, reading exactly eight fields of `pipe_draw_info`; across the whole
   file 54 of 148 functions (2,099 lines, 23%) are already Gallium-free. So
   `cp_draw_call` and `cp_draw_range` now carry those eight fields, with the
   field names deliberately those of `pipe_draw_info` so that introducing them
   changed no line of the pipeline that reads them, and `cp_draw_range` is
   layout-compatible with `pipe_draw_start_count_bias` so the array passes
   straight through. **`cp_draw_vbo` is the only function left in the driver
   that sees a Gallium draw type.**

   **Third slice: pipeline state.** The pipeline reads eighteen fields across
   seven Gallium state structs; three of those are pure scalars and are now
   `cp_viewport_state`, `cp_raster_state` and `cp_depth_state`, field names
   kept.

   What deliberately did **not** move: the comparison that decides whether a
   held-back batch must be flushed, and the batch key. Both still carry the
   whole Gallium structs, because narrowing them to the fields the pipeline
   reads would let batches merge that do not merge today — and batch size is
   exactly what makes the clipper's unstable primitive order visible (gaps 15
   and 16: `bloom` went from bit-identical against itself to differing on 4
   frames of 60 once its draws merged). **A factoring must not change what
   merges.**

   **Fourth slice: blend.** `cp_blend_desc` was already the driver's own type —
   it lives in the kernel ABI header so the peel writeback and the A-buffer
   composite cannot describe one draw differently — and it was being rebuilt
   from `pipe_rt_blend_state` per draw. It is built once at bind time now, and
   the only other reader of the Gallium type was a debug print.

   **Fifth slice: the attachments.** The draw path asked a
   `pipe_framebuffer_state` the same four questions on every draw — is there a
   colour buffer, where is its memory, what encoding does the writeback use,
   what is its sample stride — and answered them by unwrapping a
   `pipe_resource` and calling `cp_color_encoding_from_format` each time, in
   eight places. `cp_fb_desc` answers them once at bind time.

   **Sixth slice: vertex input.** Six properties per element — conversion,
   channel count, channel size, swizzle, attribute size, fill-w — are all
   functions of the element's `pipe_format` and none varies with the draw, yet
   all six were derived per draw, with a `pipe_resource` unwrapped for each
   buffer base alongside. `cp_vertex_elem` and `cp->vb_base` hold them from
   bind time. The Gallium arrays stay untouched beside them, because the batch
   key compares the element array and the pass-segment snapshot saves a copy
   for the fallback replay.

   **Seventh slice: the scissor.** `pipe_scissor_state` is four plain
   unsigneds and contains nothing the pipeline does not use, so this one
   converts whole — live state, the batch's per-draw array, the segment
   snapshot *and* the batch key's copy — and because the key still compares
   the same four values, **nothing about what merges changes**. That is why
   this one could go all the way and the others could not.

   **`cp_draw_execute` now has no Gallium type in its signature at all**: it
   takes a `cp_draw_call`, a `cp_draw_range`, a `cp_rect` and its own context,
   and every piece of state it reads is the driver's own.

   Gallium references in `cp_context.c`: **334 → 264**.

   **Eighth slice: the containment.** `pipe_context` was the first member of
   `cp_context` so that every entry point could cast one to the other. It now
   sits in a `cp_gallium` beside the renderer, and `cp_ctx()` is the single
   place that converts.

   **That conversion is a function rather than a cast because of what happened
   while making the change.** Ten hand-written casts in `cp_resource.c` and
   three more inline in `cp_context.c` kept compiling and silently pointed at
   the wrong offset. Every sample still exited 0. `triangle` rendered
   differently and the capture went **28% wrong**, and the only thing that
   said so was a byte comparison against a stored frame. That is this driver's
   documented worst failure mode — an exit code of 0 and wrong pixels —
   reproduced by a refactor the compiler had no way to catch. Nobody should be
   able to write that cast by hand again.

   **Ninth slice: the batch key.** The key held five Gallium structs whole.
   Narrowing it to what the pipeline reads would have let draws merge that do
   not merge today, so instead it holds an opaque blob the front end fills and
   the batcher only `memcmp`s — the adapter puts CSO structs in, the native
   driver will put a pipeline handle and dynamic state in, and neither changes
   what merges for the other. `CUDAPIPE_DEBUG_BATCHDIFF` still names the piece
   of state that broke a batch, through a field table the front end supplies.

   **Tenth slice: a bug the gate could not catch.** The segment snapshot was
   still saving the Gallium element array rather than the resolved one the
   draw path had started reading, so a segment re-executed by
   `cp_pass_fallback` would have gathered with whatever layout was live. No
   sample reaches that path and neither capture's overflow enters it, so
   `CUDAPIPE_FORCE_PASS_FALLBACK=1` now makes every episode take it: the check
   is that the forced result equals the *classic* path, and `particlesystem`
   is bit-identical to `CUDAPIPE_NO_ABUFFER=1`.

   **Eleventh slice: `struct cp_context` contains no Gallium type at all.**
   The framebuffer state, the CSO copies and the element array moved into
   `cp_gallium`; two fields turned out to be dead once the resolved forms
   existed (`vertex_buffers`, an uncounted resource reference nothing read,
   and `tex_resources[].format`, written once and never read) and were deleted
   rather than moved.

   **The draw pipeline, its state, its batching and its episode machinery are
   now a renderer a Vulkan front end can drive without Gallium existing.**

   Gallium references in `cp_context.c`: **334 → 243**, all of them now in the
   entry points, the resource and sampler-view paths, and the two
   `PIPE_FORMAT` tables the native driver already replaces.
2. **Milestone 1 — enumerate.** ✅ done. `src/cudapipe` builds a second ICD;
   `tests/cpvk_smoke.c` reports the RTX 5090 as a Vulkan physical device with
   the three memory types session 13 had to negotiate with lavapipe.
3. **Milestone 2 — device, queue, memory, buffers.** ✅ done for buffers;
   images outstanding. `vkCreateDevice` builds the CUDA context and a
   non-blocking stream; one queue, whose submit drains the stream because
   nothing records work yet; the three memory types are three branches over
   `cuMemAlloc`, `cuMemHostAlloc` and `cuMemAllocManaged`; a device-local
   allocation refuses `vkMapMemory` instead of staging behind the caller;
   allocations are not zeroed unless `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT`
   asks. `vulkaninfo --summary` completes against the native ICD.
   **Next: images, then `vkCreateImageView` and the format table.**
4. **Milestone 3 — pipelines and descriptors**, feeding the existing NIR→PTX
   compiler. ✅ done for compute pipelines; descriptors outstanding.
   `vkCreateComputePipelines` takes SPIR-V through the runtime's
   `vk_pipeline_shader_stage_to_nir` and straight into `cp_compile_nir_to_ptx`.

   **The compiler is not forked, ported or copied.** `cp_nir_to_llvm.c` (3,277
   lines) and `cp_debug.c` compile into this target from where they already
   live: their only `gallium` include is `util/u_memory.h`, which is `src/util`.
   Measured coupling per file, counting `pipe`/`gallium` references:

   | file | references |
   |---|---|
   | `cp_debug.c`, `cp_debug.h`, `cp_kernels.h`, `cp_nir_to_llvm.h` | 0 |
   | `cp_kernels.c`, `cp_nir_to_llvm.c` | 1 (`util/u_memory.h`) |
   | `cp_screen.c` | 32 |
   | `cp_resource.c` | 65 |
   | `cp_context.c` | 162 |

   So the backend was portable all along and the Gallium coupling is
   concentrated in exactly the three files that *are* the adapter. The NIR
   options moved to `cp_nir_options.h`, shared by both drivers rather than
   duplicated.

   The backend's own unconditional unimplemented-intrinsic warning named the
   missing lowering on the first run — `load_deref`, `store_deref`,
   `vulkan_resource_index`, `load_vulkan_descriptor`. Generic lowering is now
   done here, including telling `nir_lower_compute_system_values` that the
   dispatch base is always zero, without which `load_base_global_invocation_id`
   reached a backend with no case for it. What is left is exactly the three
   descriptor intrinsics, which need this driver's own descriptor model — the
   next milestone, and the one where the descriptor-set simplification pays.
5. **Milestone 4 — command buffers**: record, then translate a whole render
   pass at submit. This is where batching and episodes stop existing.
   ✅ done for compute. Descriptor set layouts, pipeline layouts, pools, sets,
   command buffers and a queue submit that launches; `cpvk_smoke` dispatches
   256 threads of `data[i] = i * 2` and checks the values, 256 of 256.

   The descriptor model is the simplification arriving in code. The generated
   kernel's argument block already *is* an array of pointers whose slot
   `18 + i` the backend dereferences for a 32-bit-index `load_ubo` or
   `load_ssbo`, so **a descriptor set is an array of device addresses**,
   binding one is filling in argument slots, and `vulkan_resource_index`
   lowers to a compile-time flat index. No descriptor memory exists — no
   `VkDeviceMemory` per set, therefore none of the page-granular managed
   allocation that cost 22.4 ms of a 122.6 ms frame under lavapipe and two
   optimisation passes to claw back.

   Two placeholders, marked as such in the code: the sync type is
   always-signalled, which is correct *only* because every submit drains the
   stream (`cp_fence` shows the shape it becomes — refcounted, because one
   submit's fence lands in several `vk_sync`s and a bare destroy double-freed
   under the first triangle); and the per-dispatch argument block is a managed
   allocation freed after a synchronise, which is what the upload arena
   replaces once there is a frame to amortise over.

6. **Milestone 5 — images, views, format table.** ✅ done. Linear layout with
   every level at its own offset (checked in the smoke test by offsets and
   pitches, not by a return code: 64×64 with four mips at 0/16384/20480/21504,
   pitches 256/128/64/64, no overlap). Format support is derived from the same
   `CP_TEXEL_*` and `CP_COLOR_*` encodings the kernels decode and encode, so
   what the driver advertises and what its kernels can read cannot drift; a
   format outside the table is refused rather than clamped.

   **Next: the graphics pipeline and render pass — the point at which a sample
   can be pointed at the native ICD and the first frame can be diffed against
   the Gallium-hosted build.** That means vertex input state, the rasterizer
   and A-buffer entry points from `cp_context.c` behind a Vulkan-shaped call,
   and `vk_render_pass`'s attachment info driving the framebuffer. It is the
   largest remaining piece and the one that milestone 0's factoring exists to
   make possible.
6. **Milestone 5 — parity.** triangle → the 18 samples → both captures,
   **diffed against the Gallium-hosted build, not against NVIDIA.** Both
   targets are built from one tree precisely so this comparison exists.
7. **Milestone 6 — spend the new information**: pass-scoped binning for the
   tiler, load/store elision, parallel command translation. One measured change
   at a time.

llvmpipe is untouched throughout and stays the calibration reference. The two
ICDs coexist, so any frame can be rendered both ways.

## Milestone 1, as built

```
src/cudapipe/
    meson.build          entrypoints (prefix cpvk), static lib, ICD, devenv json
    cpvk_private.h       instance / physical device / device objects
    cpvk_device.c        instance, enumeration, properties, memory, queues
    cpvk_target.c        vk_icdGetInstanceProcAddr
    tests/cpvk_smoke.c   the milestone test
```

```
$ VK_DRIVER_FILES=.../cudapipe_native_devenv_icd.x86_64.json cpvk_smoke
vkEnumeratePhysicalDevices -> 0, count 1
  [0] cudapipe (NVIDIA GeForce RTX 5090)  api 1.0.354  type 2
      3 memory types, 2 heaps, heap0 31.4 GiB
      1 queue family
ok
```

Known gaps at this milestone, recorded rather than discovered later:

- `vulkaninfo --summary` reaches **`vkCreateDevice`** and stops there, which is
  milestone 2. Getting it that far took two fixes worth recording, because both
  present as a segfault with no driver frame in the backtrace — a jump to
  address zero from inside the loader:
  - `GetPhysicalDeviceImageFormatProperties2` and the external
    buffer/fence/semaphore queries were unimplemented. Any application walks
    these for every format, and an unimplemented physical-device entrypoint is
    a null pointer rather than a clean refusal. They now refuse explicitly.
  - `KHR_get_physical_device_properties2` was advertised and had to be taken
    straight back out: an application that enables it calls the KHR *aliases*,
    and at `apiVersion` 1.0 those are not wired to the runtime's core
    implementations. It comes back with 1.1, or with the aliases implemented.
    This is the driver's own "a capability you advertise but clamp is worse
    than one you refuse" rule, in its sharpest form.
- `cpvk_GetPhysicalDeviceFormatProperties2` deliberately reports nothing. A
  capability advertised and then clamped is worse than one refused — the
  multisampling sample already demonstrated that here.
- Limits are the capture's requirements, not the backend's measured maxima.
  They should be raised as features land, not ahead of them.

## The end state: shared Mesa back to upstream

The native driver is finished when nothing outside `src/cudapipe/` differs from
upstream. Measured against `upstream/main` (merge base `0d82e6c4072`), the
current divergence in shared code is **12 files, 258 insertions** — this is the
retirement checklist, and every line of it is a thing the native driver owns
instead:

| shared file | what it is | what replaces it |
|---|---|---|
| `src/gallium/include/pipe/p_screen.h` (+10) | `allocate_memory_device` and `clear_memory` hooks added for session 13's memory split | `vkAllocateMemory` in the driver; no hook |
| `src/gallium/frontends/lavapipe/lvp_device.c` (+167) | three memory types when the screen supplies a device allocator; sample-count limits derived from `is_format_supported` | `cpvk_GetPhysicalDeviceMemoryProperties` (already written at milestone 1) and a real format table |
| `lvp_execute.c` (+65) | routes byte-identical copies through Gallium's `image_copy_buffer` (session 16); **and a `strstr(screen->get_name(), "cudapipe")` test** that flushes a pipeline barrier without finishing (session 10) | `vkCmdCopyBufferToImage` implemented directly; barriers handled from their real stage/access masks |
| `lvp_private.h` (+13), `lvp_device_generated_commands.c` (+4) | supporting fields | gone |
| `src/gallium/auxiliary/driver_trace/tr_screen.c` (+36) | trace wrappers for the two new screen hooks | gone with the hooks |
| `src/gallium/auxiliary/target-helpers/sw_helper.h` (+9) | selects the cudapipe gallium driver | gone |
| `src/gallium/meson.build` (+9), `meson.build`, `meson.options`, `.gitignore` | the gallium driver and target | a `vulkan-drivers` entry for `cudapipe` |
| `src/gallium/drivers/cudapipe`, `src/gallium/targets/cudapipe` | the driver itself, ~15,500 lines | `src/cudapipe/` |

The `strstr(get_name(), "cudapipe")` in `lvp_execute.c` is worth singling out:
a driver-name sniff inside a shared frontend, added because Gallium has no way
to say "this driver's barriers are already ordered". It is the clearest single
argument in the tree for the move.

A side effect worth having: once lavapipe is untouched, `TESTING.md`'s standing
warning — that a change under `frontends/lavapipe` moves llvmpipe too and can
invalidate the comparison set without cudapipe changing at all — stops
applying. llvmpipe becomes a fixed reference again.

## Standing gate

Every milestone is measured against the recorded numbers, not against intent.
The gate is the objective of this work: **both gfxr captures replay to the last
frame, and the 18-sample sweep shows no verdict worse than the standing two.**

Recorded state at the time this branch was cut (Gallium-hosted driver, census
flags off, tiling prototype present but disabled):

| | recorded | measured on this branch |
|---|---|---|
| old capture, 10-frame dump vs release llvmpipe | 0.441% > 32/255, 0.002% > 96/255 | **0.429% / 0.001%** |
| old capture median | ~25.20 ms | 25.18 / 25.29 / 25.30 / 25.32 ms |
| Crossroads median | ~7.10-7.21 ms | 7.16 / 7.21 / 7.22 / 7.24 ms |
| 18-sample sweep total | 35.85 ms (`buffer-image-device`) | **35.91 ms**, +0.17% |
| sweep verdicts | `gltfscenerendering`, `texture3d` standing; `renderheadless` vacuous | unchanged, all 18 exit 0 |

Iteration records: `~/git/Vulkan/build/iter/census-vk-native`, and
`nir-options-shared` after the shared-header move — 35.90 ms total, −0.0%, all
18 exiting 0, the same two standing verdicts, and the capture unchanged at
0.429% / 0.001% with replays of 25.27 ms and 7.22 ms median.

One thing that measurement established and is worth keeping: **byte-identity
against itself is not currently a property of this tree.** The same build
dumped twice differs on 3 of 10 capture frames while the gated percentages are
identical, so a byte diff between two builds is not evidence of a change. Use
the >32/255 and >96/255 means, which is what the gate has always been.

When the native driver renders, it takes the same table, and additionally has
to match the Gallium-hosted build frame for frame — which is why both targets
are built from one tree.
