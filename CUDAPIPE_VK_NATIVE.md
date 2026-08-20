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

   Gallium references in `cp_context.c`: **334 → 267**, all of them now in the
   entry points, the resource and sampler-view paths, and the two
   `PIPE_FORMAT` tables the native driver already replaces.
2. **Milestone 1 — enumerate.** ✅ done. `src/cudapipe` builds a second ICD;
   `tests/cpvk_smoke.c` reports the RTX 5090 as a Vulkan physical device with
   the three memory types session 13 had to negotiate with lavapipe.
   **The split is finished.** `cp_renderer.c` is 6,758 lines and
   `cp_context.c` is 2,240. The renderer — the draw, the shading launches, the
   A-buffer, the batching, the episodes, the arenas, the register tuner — is
   compiled into both drivers. What is left in `cp_context.c` is the Gallium
   entry points, the resource and sampler-view paths, and the two
   `PIPE_FORMAT` tables: the adapter, and nothing else.

   Slices, each gated on the full capture and sweep: draw description, state,
   blend, attachments, vertex input, scissor, containment, batch key, the
   fallback-snapshot bug, the Gallium-free renderer struct, the device split,
   `cp_context_init`, the renderer header and file, the arenas, the A-buffer,
   vertex assembly, the draw itself, and the batching. Nineteen in all; the
   capture read exactly 0.429% / 0.001% through every one.

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

## Textures: the contract the backend already has

Dumped from the Gallium path (`CUDAPIPE_DUMP_NIR=1` on the `texture` sample),
because the backend's expectations are not written down anywhere else. The
fragment shader reaches the sampler like this:

    64 %12 = @load_const_buf_base_addr_lvp (%11 (0x1))
    64 %14 = iadd %12, 0x10
    64 %16 = iadd %12, 0x50
    32x4 %17 = txb %14 (texture_handle), %16 (sampler_handle), ...

So:

- A descriptor **set** is one flat buffer of descriptors, and its address is a
  constant-buffer slot. `load_const_buf_base_addr_lvp(slot)` yields it; the
  backend implements that intrinsic as `emit_const_buf_base()`, the same path
  a UBO read takes.
- A handle is `set_base + binding_offset`, computed in the shader. It is not
  loaded from memory, so the native driver's current one-address-per-binding
  scheme cannot express it: the slot must hold the *set*, and the binding
  offset must be added in NIR.
- The kernel reads two fields out of a handle and nothing else:
  `*(cp_texture_info **)(tex_handle + 48)` and
  `*(unsigned *)(samp_handle + 28)`, an index into `cp_sampler_table`. See
  `CP_DESC_IMAGE_FUNCTIONS_OFFSET` and `CP_DESC_SAMPLER_INDEX_OFFSET`. The
  descriptor layout between those offsets is the driver's to choose.

That makes the work concrete: descriptor sets become managed buffers with a
64-byte stride, `vkCreateImageView` gets a device-resident `cp_texture_info`,
`vkCreateSampler` gets an entry in `cp_sampler_table`, and the descriptor
lowering emits the base-address intrinsic plus an offset instead of a
per-binding address.

One constraint that is easy to miss: the sampler-variant optimisation in
`cp_renderer.c` reads the descriptor **on the host** to decide whether a
shader can be specialised, so descriptor memory has to be host-readable.
Managed memory, not device-local.

## The constant-buffer slot numbering, and why it has to be lavapipe's

The native driver numbers a slot per push-constant block, then one per
descriptor *binding*, then one per set. A real capture's pipeline layouts want
17, 18, 22 and 23 slots and there are `CP_MAX_CONST_BUFFERS` = 16. The
bindings past the limit reach the shader as an address of zero, and the replay
dies on a 5x3x1 compute dispatch with `CUDA_ERROR_ILLEGAL_ADDRESS`.

lavapipe's scheme, which is the one the backend was written against:

- `vulkan_resource_index(set, binding, index)` becomes
  `vec3(set + 1, index * stride + binding_offset, 0)` — component 0 is the
  constant-buffer slot, **one per set**, with slot 0 the push constants.
- The address format is `nir_address_format_vec2_index_32bit_offset`, whose
  address is **three** components despite the name. Returning a `vec2` trips
  `addr_to_index`'s assertion in `nir_lower_explicit_io`.
- Afterwards, every `load_ubo`/`load_ssbo`/`ssbo_atomic` whose source is still
  the pair becomes `load_const_buf_base_addr_lvp(slot) + offset`, a 64-bit
  descriptor address. `emit_buffer_base` dereferences that and reads the
  buffer pointer out of the descriptor's first field, which is where
  `cpvk_descriptor` already keeps it.

A binding is then an offset inside its set's buffer rather than a slot of its
own, so the slot count depends on the number of sets and not on how many
bindings they hold.

This was attempted and reverted: with the `vec3` fix it still aborts inside
`nir_lower_explicit_io`, so something else in the chain is still handing it a
two-component address. The next attempt should find that before rewriting the
numbering, because the numbering itself is not in doubt.

## The replay's remaining fault, narrowed

`compute-sanitizer` on the capture:

    Invalid __global__ read of size 8 bytes
      by thread (0,0,0) in block (0,0,0)
      Address 0x80 is out of bounds

An 8-byte read at 0x80 is `emit_buffer_base`'s 64-bit path reading a
descriptor pointer at `const_buf_base(slot) + 128` where the slot resolved to
zero. The driver now prints the dispatch's slots when a launch fails:

    compute dispatch 5x3x1 failed: CUDA_ERROR_ILLEGAL_ADDRESS (700)
      push=0 bytes, buffer slots: [1]=0x71db04032200 [2]=0x71db040322c0

So slots 1 and 2 hold valid descriptor-set snapshots, no push constants were
recorded, and something reads slot 0 -- the push constant slot -- 128 bytes
in. Either the shader declares a push constant block the capture never
pushes to before this dispatch, or it reads a descriptor set the app binds at
a set index this layout maps to a slot nothing filled.

The next step is to dump the compute shader's NIR at that pipeline and read
which slot it loads, rather than reason about which one it ought to be.
`CUDAPIPE_DUMP_NIR=1` prints it, but the capture builds many pipelines and the
one that faults has to be identified first -- printing the pipeline pointer
beside the dispatch would do it.

## The replay stops losing the device, and starts not finishing

Two fixes, one after the other, both about reads that had nowhere to land.

`compute-sanitizer` put the first fault at `main+0xa0` — a compiled shader,
since every one of them is named `main` — reading eight bytes at `0x80`,
*before frame 1 begins*, so during the replayer's resource-init phase. Dumping
the faulting pipeline's NIR (matched by the pointer the failing dispatch now
prints) showed it reads only slots 1 and 2, both bound to valid snapshots. So
the null was not a slot: it was a **descriptor** inside a bound set that
nothing ever wrote, whose base was zero. `0x80` is 128, which is descriptor
two.

Both now point at a 64 KB zeroed page whose every word is the page's own
address, so a read of an unbound slot or an unwritten descriptor finds a base
pointer, follows it, and lands in zeroes. **This is a bring-up aid and not a
fix**: a shader reading an unwritten descriptor is a bug in the application or
in this driver's descriptor handling, and the page only changes a lost device
into a wrong frame. It is worth having because a lost device stops a replay
dead and a wrong frame does not.

With it, the capture replays with no driver errors at all where it used to die
in seconds — and does not finish in forty minutes, against roughly thirty-eight
seconds for the Gallium-hosted driver. No frame markers reach the fps plugin
in five minutes, so it is not merely slow by a constant: something is either
looping or making no frame progress. That is the next thing to look at, and the
first question is whether frame 1 ever completes.

## Multisample images are allocated at single-sample size

The resolve prints its operands now, and they say it outright:

    vkCmdResolveImage src 1280x720 samples=4 size=3686400 mem=...
                      dst 1280x720 samples=1 size=3686400 mem=...

3,686,400 is 1280 x 720 x 4 bytes: one sample's worth. `cpvk_image` never
looks at `VkImageCreateInfo::samples`, so a four-sample image asks for a
quarter of the memory it needs, the application allocates that, and everything
downstream is addressing memory that is not there. `cp_fb_desc::color_sample_stride`
is never set either, which is the same omission from the other side: the
renderer needs to know how far apart the samples are and is told zero.

That is the bug behind the replay's remaining crash inside `cuMemcpy2DAsync`,
and behind `multisampling` rendering black. Two things to fix together:

- size an image as `w * h * bpp * samples`, and report that from
  `vkGetImageMemoryRequirements`;
- set `color_sample_stride` on the framebuffer description so the renderer
  writes samples where the resolve later expects to find them.

Then the resolve can average them instead of taking sample zero.

## Three hypotheses about the replay's crash, refuted

Each was cheap to test and each was wrong, which is worth writing down so the
next attempt does not start here.

1. **Use-after-free of device memory.** `vkFreeMemory` was made to leak
   instead of freeing. The crash was unchanged, in the same place, with the
   same 137 draws before it.
2. **Concurrency.** gfxrecon replays on several threads and the renderer is
   one shared `cp_context`. Replaying with `--sync` changed nothing.
   `--serialize-queue-submissions` fails earlier for an unrelated reason and
   says nothing either way.
3. **Allocations too small for this driver's layout.** The layout here is its
   own -- rows aligned to 64, samples as planes -- so an image can be bigger
   than it was at capture time and be bound to memory that ends before it
   does. `vkBindImageMemory` and `vkBindBufferMemory` now check and say so;
   nothing in the capture triggers it.

What remains true: `cuMemcpy2DAsync` faults inside libcuda on a copy whose
source reaches exactly to the end of its buffer and whose destination is well
inside its image, with the memory live and the parameters traced. A fault on
sane arguments points at the CUDA context having been corrupted earlier by a
device-side access -- which `compute-sanitizer` would name, and which is the
next thing to run now that the earlier faults it reported are fixed.

## Where the ten wrong samples differ

Localised rather than guessed at, by counting which pixels differ from the
Gallium-hosted driver by more than 16/255:

    texturemipmapgen   0.9% of pixels, rows 212..507, cols 471..806
    pbribl             9.1% of pixels, rows 311..414, cols 98..1240
    pushconstants      5.6% of pixels, rows 108..611, cols 388..891

`texturemipmapgen` differs only in the centre of the image, which is where the
texture is minified hardest and the generated mip levels are used. So the
difference is in mip generation, which this driver does on the host.

Two explanations tested and refuted:

- **sRGB.** Averaging sRGB bytes without decoding would be wrong in exactly
  this way. The format is `VK_FORMAT_R8G8B8A8_UNORM` and the trace says
  `srgb=0`, so it is not that.
- **Rounding.** Every blit is an exact halving with LINEAR filtering, where a
  2x2 box average is the correct answer and only the rounding can differ.
  Round-half-to-even instead of round-half-up changed the mean by nothing at
  all: 0.4560 either way.

So the box filter agrees with llvmpipe's blit on value and rounding, and
something else about it does not. The next thing to look at is which texels it
takes: llvmpipe samples at destination texel centres, and a half-texel offset
would shift the footprint by one without changing its size.

`pbribl`'s difference is a horizontal band, not the whole image, which says one
object or one pass rather than a global error.

## The generated mip levels are never sampled

The strongest fact of this investigation, and it was cheap: **deleting mip
generation entirely does not change the frame.** Skipping every scaling blit
gives a mean difference of 0.4560, exactly what generating them gives.
Changing the filter from a box average to a point sample also gives 0.4560.
The levels are written and never read.

Things that are *not* the cause, each checked rather than assumed:

- **The sampler state.** `texturemipmapgen` creates four samplers and this
  driver translates all four correctly, including the two with `lod 0..10` and
  the one with `maxAnisotropy 16`. `CUDAPIPE_DEBUG_TEX` prints them.
- **The mip filter.** Box average and point sample give identical results, so
  the filter cannot be what differs.
- **sRGB.** The format is UNORM.
- **The texture info.** `texture` and `texture3d` are byte-exact, so the
  descriptor, the format table and the sampler path are right for a texture
  whose levels were uploaded rather than generated.

What is left is the sampler's mip selection. `cp_tex_sample_impl` takes a
`coord_slot` -- the fragment shader input the coordinate came from -- and
selects the base level when it is -1, and the backend only sets it when the
coordinate's parent instruction is a `load_input` intrinsic. The native front
end scalarises every ALU op, so a coordinate arrives as a vector built from
scalars and no longer as a `load_input`.

Teaching the backend to see through a `vec` of `load_input`s was tried and
changed nothing, so the pattern is not that either -- there is something else
between the varying and the coordinate. Dumping the fragment shader's NIR for
this sample and reading what feeds `nir_tex_src_coord` is the next step, and
it is a five-minute one now that everything else is excluded.

### Correction: the centre is not minification

`texturemipmapgen`'s mipmapped texture reaches the sampler as
`512x512 levels=0..9`, which is correct, alongside four correctly translated
samplers. Texture info, sampler state and level count are all right, and
deleting mip generation still changes nothing.

So the earlier inference -- that a difference in the centre of the frame meant
a difference in the deepest mip levels -- was an inference and not a
measurement, and it was wrong. The centre of this frame is the vanishing point
of the geometry, which is where any small shading difference concentrates
whatever its cause.

At 0.456 mean this is the *smallest* difference of the ten wrong samples, and
three turns have now gone into it on the strength of that inference. The band
in `pbribl` at 4.325, which covers one horizontal strip and therefore one
object or one pass, is a better-shaped signal and has had none.

## pbribl: the spheres are missing, and the vertices are why

Not a shading difference. The row of ten PBR spheres does not appear at all;
the skybox behind them is pixel-perfect. That is what the "horizontal band"
was.

The draws happen -- ten of them, 4,512 triangles each, with the right
framebuffer, viewport and vertex elements -- and `CUDAPIPE_DEBUG_WORK` reports
`shaded=0` for every one. No fragment survives, so the triangles have no
coverage: the vertices are wrong, not the shading.

Excluded by disabling each and re-running, rather than by argument:

- **Depth.** Forcing `depth_enabled` and `depth_writemask` off changes nothing.
- **Culling.** Forcing `cull_face` to none changes nothing.
- **Winding.** Inverting `front_ccw` changes nothing.

So the vertex stage produces positions with no coverage. The draws use a
96-byte vertex stride with three attributes at offsets 0, 12 and 24, which is
a glTF model sharing one buffer between meshes -- so `vkCmdDrawIndexed`'s
`vertexOffset`, which this driver maps to `cp_draw_range.index_bias`, is the
first thing to check, followed by the push constant the sample uses to place
each sphere.

The way to see it is to print the first few vertex positions for one of those
draws. The renderer prints them for small draws already; the flag needs to
reach these.

### pbribl: what is excluded so far

The ten sphere draws reach the renderer with the right framebuffer, viewport,
vertex elements (three attributes at 0, 12 and 24 in a 96-byte stride, which is
the sample suite's glTF vertex exactly) and shade zero fragments.

Excluded by disabling or by reading the actual values, not by argument:

- depth test and write: forced off, no change
- culling: forced to none, no change
- winding: inverted, no change
- the uniform buffer: its first floats read `0.974 0.000 0.000 0.000`, which is
  a projection matrix's first row, so the matrix reaching the vertex shader is
  real and not the null page
- unbound slots: fourteen of sixteen are the null descriptor for *every* draw
  in this sample including the skybox, which renders correctly, so that is
  normal and not a signal

So a correct matrix and correct vertex bindings still produce no coverage. What
has not been looked at is the vertex shader's output itself. The renderer will
print it -- `CUDAPIPE_DEBUG_FS` dumps the first vertices -- but that path
allocates per-pixel buffers for a 1280x720 frame and segfaults on this sample
before reaching the sphere draws, so it needs a smaller window or a cap before
it can answer this.

That is the next piece of work, and it is a change to the debug path rather
than to the driver.

### The debug path still dies, and a new gap found on the way

`CUDAPIPE_DEBUG_FS` allocated four buffers scaled by a full frame's pixel
count. They are bounded to 4,096 pixels now, which is more than the eight lines
it prints and removes an allocation of tens of megabytes per draw. **It did not
fix the crash**: pbribl still segfaults under that flag and runs fine without
it, so the fault is elsewhere in the dump -- the vertex fetch sized by
`num_triangles * 3 * num_vs_outputs * 16` against whatever `vs_output_buf`
actually holds is the next place to look.

Found while reading that output, and unrelated to the spheres:

    intrinsic 'load_output' is not implemented

A shader in this sample reads its own framebuffer output. That is a real gap in
the backend and nothing has been done about it.

### Comparing the two drivers' debug output is not valid any more

`elem[i]: fmt=` prints `velem[e].conv` in this tree. The Gallium driver was
reverted to the branch point, where that field did not exist and the same
format string printed `src_format`. So `fmt=0` natively against `fmt=23`
under Gallium is two different fields with one label, not a difference in the
vertex elements. Nothing was wrong there.

The one real difference the comparison did show is expected: the native driver
issues ten draws of 4,512 triangles where the Gallium driver issues one batched
draw of 45,120. Native has no batching, which is known.

The lesson is narrower than "be careful": **any** debug output compared across
the two drivers is comparing a current file against a nine-month-old one, and
only fields that predate the branch can be trusted to mean the same thing.

### The shape of pbribl's draws is exonerated

`tests/cpvk_mesh.c` reproduces those draws in a second instead of a minute: a
96-byte stride (vkglTF::Vertex) with three attributes declared at 0, 12 and 24
including an `R32G32_SFLOAT`, an indexed draw with `firstIndex` 3 **and**
`vertexOffset` 6, a `mat4` from a uniform buffer, and a `vec3` from a push
constant, multiplied the way the sample's vertex shader multiplies them. Nine
vertices, of which six are decoys placed off-screen so that a draw ignoring
either offset renders nothing.

It is byte-identical to lavapipe. Every one of those mechanisms works.

So the missing spheres are not caused by the vertex layout, the offsets, the
matrix, or the push constant. What is left is what pbribl does that this does
not:

- several offscreen passes before the spheres -- an irradiance cube, a
  prefiltered environment map and a BRDF LUT -- any of which could leave state
  behind, and one of which uses `load_output`, which the backend does not
  implement
- a vertex shader that does more than transform a position
- 4,512 triangles per draw against this test's one

The first of those is the only one that explains why the draws arrive with
correct-looking state and shade nothing, and it is where to look next.

### The missing spheres reproduce in one second

Adding a second render pass to `cpvk_mesh` -- a 4x4 pass into a 64x64 image
before the main one -- makes the main pass fault:

    cuLaunchKernel failed at cp_draw_execute:4558
      CUDA_ERROR_ILLEGAL_ADDRESS (700)

which is the rasterizer's stage-2 launch. lavapipe renders the same sequence
correctly. pbribl runs exactly this pattern three times over before it draws
its spheres, and its sphere draws shade nothing.

So the bug is not in the draw, which this test already exonerated with one
pass. It is in what a *second* `vkCmdBeginRendering` leaves behind. The obvious
suspects, in order of how cheap they are to test:

- `cp_context_set_framebuffer` is grow-only, so a 4x4 pass allocates
  16-pixel framebuffer buffers and the 64x64 pass that follows grows them.
  Growing frees the old allocations; if anything still refers to them, that is
  the illegal address.
- The renderer takes no row stride for the colour target -- `cp_fb_desc` has
  `color_sample_stride` and no `color_row_stride` -- so a render area smaller
  than its image is addressed as if the image were the render area's width.
  4x4 into a 64x64 image is exactly that case.

`tests/cpvk_mesh.c` fails against lavapipe on purpose now and is the shortest
path to it.

### A real bug, found by a reproduction that was not faithful

`cpvk_mesh`'s two-pass form faulted with `main+0x230 reading 4 bytes at 0x0`,
which `compute-sanitizer` named in a second because the test runs in a second.
The cause was in `cpvk_execute_draw`: after substituting the null descriptor
for every unbound buffer slot, it then assigned slot zero unconditionally from
the push-constant upload --

    cp->vs_ubos[CPVK_UBO_PUSH_SLOT].buffer = (void *)(uintptr_t)push_dev;

-- so a draw with no push constants, or one whose upload came back empty,
overwrote the null descriptor with a null and pointed a shader at address zero.
Fixed, and the two-pass test is byte-identical to lavapipe with no driver
errors at all.

**It did not fix pbribl.** The spheres still shade nothing, eleven draws of
zero, and every sample's difference is unmoved to three decimals. So the
reproduction reproduced *a* bug and not *the* bug -- it was built from the
shape of pbribl's draws plus a second render pass, and that combination is
sufficient to break the driver by a route pbribl does not take.

Worth keeping straight: the test earned its place twice over, once by
exonerating the vertex path and once by finding this, and it still has not
explained the thing it was written for.

### pbribl: everything cheap is now excluded, and it is being parked

Under `compute-sanitizer`, pbribl reports **0 errors**. Nothing faults. The
spheres are simply not producing coverage.

Excluded, each by measurement:

- the vertex layout, the offsets, the matrix and the push constant -- by
  `cpvk_mesh`, which reproduces all of them and is byte-identical to lavapipe
- the matrices themselves -- read out of the descriptor at the draw:
  projection `0.974, 1.732`, model and view both sane
- depth test and write, culling, winding -- each forced off, no change
- a stale depth buffer -- clearing `cp->depthbuf` at every
  `vkCmdBeginRendering` regardless of loadOp, no change
- memory errors of any kind -- none reported

Seven turns have gone into this sample. The remaining hypotheses all require
reading the vertex shader's output, and the renderer's dump for that still
crashes on a 1280x720 frame.

**Parked deliberately.** The objective names replay time as well as
correctness, and the native driver replays both captures at 78.90 ms and
24.27 ms against recorded medians of 25.20 and 7.17 -- a measured regression
with known, structural causes: no draw batching at all, a `cuStreamSynchronize`
per submit, and host loops for resolves and scaling blits. That is tractable
engineering against a number the objective actually names, and it is where the
next turns should go.

## The replay-time gap is draw batching, and nothing else

Measured rather than argued, with the Gallium driver's own `CUDAPIPE_NO_BATCH`
switch as the control:

    capture       gallium   gallium      native
                  batched   no batch    (no batching)
    Crossroads      7.13      27.88        24.28  ms
    old capture    25.20     100.43        78.90  ms

The native driver is **faster than the Gallium driver with batching turned
off**, on both captures. It is not slow. It is missing one feature, and that
feature accounts for the entire difference.

Also measured and refuted: the `cuStreamSynchronize` per submit, which looked
like the obvious cost -- 3,022 of them for 1,510 frames. Removing both syncs
moves the Crossroads median from 24.29 ms to 24.28 ms. It costs nothing,
because the work it waits for has already been waited on by the host reads
around it.

So the remaining performance work is one item: batching on the native path.
The renderer's batching machinery is already compiled into this driver --
`cp_batch_record`, the key, the flush discipline, the pass and opaque
episodes, all of it moved across in the extraction -- and what is missing is
the front end that fills a `cp_batch_key` and decides when to hold a draw
back. `cp_draw_vbo` in the Gallium adapter is the worked example.

### Why nothing merges: the key is built from the previous draw

`cpvk_batch_can_join` runs *before* the draw is staged into the context --
which it must, because a flush renders what is held back and reads the context
to do it. But `cpvk_build_batch_key` reads the context for the shaders, the
vertex elements and the binding counts, so it compares each draw's key against
a key built from its predecessor. `CUDAPIPE_DEBUG_BATCHDIFF` says it in one
line: **the first differing byte is always +0**, which is `cp_batch_key::vs`.

That is why batches are one or two draws long and why the state-based key
bought nothing: 23.97 ms against 24.28 without batching, with batches of 1 and
2 in gltfscenerendering.

Replacing the pipeline pointer with the pipeline's state was still right --
the samples are unchanged with it, `multithreading` slightly better -- and it
is committed. Twelve vertex elements fit in the 768-byte blob beside
everything else; a pipeline with more refuses to batch rather than batching on
a key that does not describe it.

The obvious repair is to build the key from `d` and `d->pipeline` instead of
from `cp`. That was tried and took the sample count from 18/18 running to
2/18, so more than the key reads the context at that point, and it was
reverted. What exactly is not yet known.

### Why nothing merges, finally: one shader binary per pipeline

With the batch decision made by comparing the draws themselves --
`cpvk_draws_mergeable`, which touches no driver state and so is safe to run
before the incoming draw is staged -- `CUDAPIPE_DEBUG_BATCHDIFF` names the
field that breaks every batch in gltfscenerendering:

    batchdiff: vertex shader

The sample builds one pipeline per material and this driver compiles a fresh
`cp_shader_binary` for each, so two draws whose SPIR-V is byte-identical hold
different shader pointers. lavapipe deduplicates shader modules, which is why
the Gallium driver merges the same draws into one of 45,120 triangles.

The fix is a shader cache: compile once per unique SPIR-V and share the
binary. It would make batching effective *and* cut pipeline-creation time,
which is the other thing this driver spends the first hundred frames of a
replay on.

Three things had to be right before that could be seen, and each was wrong in
its turn: the batch decision must precede staging (or a flush renders the
previous batch with this draw's state); the key must not be built from the
context at that point (or every draw is compared against its predecessor); and
the descriptor addresses must not be a merge condition (this driver snapshots
each bind into fresh memory, so they never repeat).

### The shader cache, and the tension that is left

Two pipelines built from identical SPIR-V compiled it twice and held two
`cp_shader_binary` pointers, so nothing merged. They share one now, keyed on
the runtime's own `vk_pipeline_hash_shader_stage` -- SPIR-V, entry point and
specialisation constants -- mixed with the layout's set numbering, because the
descriptor lowering depends on it. gltfscenerendering's batches go from all
1 to 1, 2, 6 and 9 draws.

It did not make the capture faster: Crossroads is 24.04 ms against 24.28
unbatched. And the reason is a real tension rather than another oversight.

Merging draws that bind different descriptors is wrong here --
gltfscenerendering goes from exact to 20.768 when descriptors are left out of
the comparison. So the comparison includes a hash of each bound set's
contents. But the capture rebinds descriptors on nearly every draw, so with
that hash in the comparison almost nothing merges.

The Gallium driver merges those same draws and does not compare descriptors at
all, because `cp_batch_record` snapshots a binding row per draw and the shader
indexes its row. That mechanism is in this driver too and the rows are filled,
so in principle the hash should not be needed -- and taking it out breaks
gltfscenerendering. Something about the per-draw fragment rows is not reaching
the shader the way it does under Gallium, and that is the next thing to find:
it is worth a 4x replay, since the Gallium driver's own numbers say batching
is the difference between 27.88 ms and 7.13.

### The per-draw rows are fine; something else breaks the merged frame

`CUDAPIPE_DEBUG_DRAW` prints the batch's vertex-stage rows at flush, and with
the descriptor comparison removed gltfscenerendering merges nine draws whose
rows are genuinely distinct:

    batch of 9 draws
      row 0: 0x...800 0x...e000 0x...e0c0 ...
      row 1: 0x...900 0x...e000 0x...e140 ...
      row 2: 0x...a00 0x...e000 0x...e1c0 ...

Slot 0 (push constants) and slot 2 (the material's set) differ per row, slot 1
is shared. So the per-draw binding tables are filled correctly and the shader
has what it needs to index its own row. The mechanism is not the problem.

The frame is still wrong when those nine merge, so something else that a batch
does not carry is varying across them. The fragment rows are not printed by
that debug path -- only the vertex ones -- so printing them is the obvious
next step, and after that the sampler-variant specialisation, which resolves
one sampler state for a whole batch and falls back only if every row agrees.

Until then the descriptor hash stays in the comparison: it is conservative,
it is correct, and it costs the merges the capture would want.

### Both binding tables are correct, and the merge is still wrong

`CUDAPIPE_DEBUG_DRAW` now prints the fragment rows beside the vertex ones at
flush. On gltfscenerendering's nine-draw batch both are per-draw and identical
to each other, which is right for this driver -- both stages read the same
descriptor sets:

    vs row 0: 0x...800 0x...e000 0x...e0c0 ...
    fs row 0: 0x...800 0x...e000 0x...e0c0 ...
    vs row 1: 0x...900 0x...e000 0x...e140 ...
    fs row 1: 0x...900 0x...e000 0x...e140 ...

So the tables the shaders index are filled correctly for both stages.

Also excluded, each with one run of an existing flag:

- the sampler-variant specialisation (`CUDAPIPE_NO_SAMPLER_VARIANT=1`): 20.768
  either way
- the A-buffer (`CUDAPIPE_NO_ABUFFER=1`): 20.768 either way

That leaves the core batched draw path itself. The next step is to reproduce
it small: two draws, one pipeline, one vertex buffer, differing only in the
descriptor set they bind, with the descriptor comparison removed. `cpvk_mesh`
is the place for it, it runs in a second, and it would turn a nine-draw scene
into the two-draw case that either works or does not.

### Two draws differing only in a descriptor set merge correctly

`tests/cpvk_batch.c`: one pipeline, one vertex buffer, one index buffer, two
index ranges, and a different descriptor set bound before each, each tinting
its triangle a different colour. Byte-identical to lavapipe in all three
modes -- unbatched, batched with the descriptor comparison, and batched
without it -- with both colours present:

    (26,26,38) x2637   (51,13,32) x941   (32,13,51) x518

So the per-draw binding rows carry descriptors correctly through a merge, and
the descriptor comparison is not needed for this case. That is a direct test
of the thing three turns of inference had been circling.

Which means gltfscenerendering needs something this does not have. The
difference that stands out: its materials differ in a **texture**, a combined
image sampler, where this differs in a uniform buffer. The texture path reads
`cp_texture_info` through a descriptor and has a per-batch sampler
specialisation beside it; disabling that specialisation did not help, but the
texture handle itself is resolved per row and that is not yet tested.

The next version of this test binds two different textures instead of two
different uniform buffers. It is a small edit to a test that already exists
and it either reproduces the bug in a second or narrows it again.

### The merge bug is not descriptors, it is the third draw

Two more tests, both passing, both byte-identical to lavapipe in every mode --
unbatched, batched, and batched with the descriptor comparison removed:

- `tests/cpvk_batch.c`: two draws differing only in a uniform buffer
- `tests/cpvk_batchtex.c`: two draws differing only in a **texture**, which is
  what gltfscenerendering's materials differ in

So descriptors merge correctly, uniform buffers and combined image samplers
alike. The descriptor comparison in the merge test is not buying correctness
for either case.

What it was buying is smaller batches, and `CUDAPIPE_BATCH_MAX` says exactly
where the real fault is:

    batch_max   1      2      3      4      8      64
    mean        0.000  0.000  5.515  7.304  20.648 20.768

Correct at one and two draws, wrong from three, and worse as the batch grows.
That is a scale-dependent fault in the batched path, not a merge-condition
that is too loose -- and it is bounded now: whatever indexes a batch's
per-draw tables is right for two rows and wrong for three. `emit_batch_row`
and the slice table `cp_vertex_fetch` searches are where to look, and a
three-draw version of `cpvk_batch.c` would reproduce it in a second.

### `first_vertex` comes from the batch's first draw

Three-draw and two-draw batches differing in descriptors, uniform buffers or
textures all merge correctly -- `cpvk_batch` now draws three triangles with
three descriptor sets and is byte-identical to lavapipe. So the fault needs
something none of those tests have, and `cp_draw_execute` has it in plain
sight:

    .first_vertex = indexed ? (unsigned)draws[0].index_bias : draws[0].start,

The vertex fetch takes the *first* draw's vertex offset and applies it to the
whole batch. gltfscenerendering's primitives each have their own
`vertexOffset` into a shared buffer, so any batch of them fetches every draw's
vertices from the first draw's base -- which is exactly a fault that grows
with batch size and is invisible when every draw's offset is zero, as it is in
every test here.

`cpvk_draws_mergeable` now refuses to merge draws whose `index_bias` differs.
That is conservative and it is honest about why: the batched path cannot
express those draws today. Making it express them -- the slice table already
carries a per-draw range, so it is the natural place for a per-draw bias -- is
what would let a scene of meshes out of one buffer merge, which is the case
batching exists for.

### Settled: the descriptor comparison is required, and the reason is still open

A clean A/B on one binary, three runs each way:

    desc key ON   0.000  0.000  0.000
    desc key OFF  20.768 20.768 20.768

So the single 0.000 measured with the key off last turn was an artefact, and
the comparison is genuinely required. It is also what makes batching worthless
on a capture: with it, Crossroads is 24.26 ms against 24.38 unbatched -- a
tenth of a millisecond.

What the requirement is *not*, each proven by a test that passes byte-identically
to lavapipe with the key off:

- two draws differing in a uniform buffer (`cpvk_batch`)
- three draws differing in three uniform buffers (`cpvk_batch`)
- two draws differing in a texture (`cpvk_batchtex`)
- three draws differing in three textures (`cpvk_batchtex`)

So merged draws index their own binding rows correctly for buffers and
textures alike, at two and three draws.

The remaining difference between those tests and gltfscenerendering is what
its materials do: alpha masking, which is `discard` in the fragment shader.
The Gallium adapter's own eligibility test names discard as a reason to refuse
a batch on the blended path, and the opaque path resolves visibility through a
depth-keyed buffer that a discarding shader participates in differently. That
is the next thing to test, and the test is a one-line shader change to
`cpvk_batchtex`.

### Five properties tested, none reproduces it

`cpvk_batchtex` now draws three overlapping triangles at three depths through
three descriptor sets naming three textures, and passes byte-identically to
lavapipe with the descriptor comparison off. So does the same test with a
discarding fragment shader. Together with `cpvk_batch`, the following all
merge correctly:

- draws differing in a uniform buffer, two and three of them
- draws differing in a texture, two and three of them
- overlapping geometry where the depth-keyed visibility buffer must pick a
  winner across draws, and shade it from its own draw's binding row
- a fragment shader that discards, which is what alpha masking is

gltfscenerendering still needs the comparison. What has not been reproduced:

- **batch size**: nine draws there against three here
- **more than one descriptor set** in the pipeline layout, which its scene and
  material sets would be

Both are small extensions of a test that exists. `CPVK_NO_DESC_KEY=1` disables
the comparison for whoever bisects it.

### Seven properties reproduced; the sample still differs

`cpvk_batchtex` now draws **nine** overlapping triangles at nine depths,
through **two descriptor set layouts** -- one for the frame's uniforms, one
rebound per draw for the material -- each naming its own texture. It is
byte-identical to lavapipe with the descriptor comparison off, as is the same
test with a discarding fragment shader.

That is every property of gltfscenerendering's draws I have been able to name:

    draws in the batch          nine, as the sample has
    descriptor sets in layout   two, as the sample has
    what changes per draw       a texture, as the sample changes
    geometry                    overlapping, depth-tested
    fragment shader             with and without discard
    vertex offset               now a merge condition either way
    binding rows                one per draw, both stages

And gltfscenerendering is still 20.768 without the comparison and 0.000 with
it, re-verified.

So there is a property of those draws that this test does not have and that I
have not identified. The honest position is that the descriptor comparison is
required for a reason not yet understood, that it costs batching all of its
value on captures, and that the search for it should start by diffing the
sample's draws against this test's rather than by proposing an eighth
hypothesis: `CUDAPIPE_DEBUG_DRAW` prints the framebuffer, viewport, vertex
elements and blend state of every draw in both, and the difference will be in
that output.

### Eight properties reproduced, none of them explains it

`cpvk_batchtex` now covers, in one test, every property of
gltfscenerendering's draws I have been able to name:

    nine draws in a batch          two descriptor set layouts
    a texture rebound per draw     overlapping geometry at nine depths
    the depth test on              a discarding fragment shader
    six fragment varyings          a 1280x720 framebuffer

All of it is byte-identical to lavapipe **with the descriptor comparison
off**. The sample is 20.768 without it and 0.000 with it.

What is still different, from `CUDAPIPE_DEBUG_DRAW` on both:

    property             cpvk_batchtex     gltfscenerendering
    triangles per draw   1                 796 .. 67,763
    vertex elements      2                 4
    vertex stride        16                60

The triangle count is the interesting one, because this driver already
documents that batch size is what makes the clipper's unstable primitive order
visible -- gaps 15 and 16, and the reason "a factoring must not change what
merges". A batch of 67,763 triangles is four orders of magnitude past anything
tested here. Whether that instability accounts for a mean of 20.768, which is
far larger than the four-frames-in-sixty it usually shows as, is the question
to answer next, and the way in is to raise this test's triangle count until it
either reproduces or does not.

### Triangle count cleared too; only the vertex layout is left

`cpvk_batchtex` takes a triangles-per-draw count now and was run at 800, which
is the low end of gltfscenerendering's 796..67,763. Batched and unbatched agree
**exactly** -- both 1.108 against lavapipe -- so batching introduces no
difference at that scale. (The 1.108 itself is a pre-existing divergence from
lavapipe at high triangle counts, present with batching off, and is a separate
question.)

That clears the last property I had named. The full list, every one reproduced
in this test and every one merging correctly with the descriptor comparison
off:

    nine draws                    two descriptor set layouts
    a texture rebound per draw    overlapping geometry, nine depths
    the depth test                a discarding fragment shader
    six fragment varyings         a 1280x720 framebuffer
    800 triangles per draw

What remains different between the test and the sample is the vertex layout:
four elements at a 60-byte stride against two at 16. That is now the whole
list, and it is two numbers.

### Every property of the draws is cleared; the difference is in the sequence

`cpvk_batchtex` now has gltfscenerendering's vertex layout as well -- four
attributes at 0, 12, 24 and 32 in a 60-byte stride -- and batched and unbatched
still agree exactly. Ten properties reproduced, ten cleared:

    nine draws                     two descriptor set layouts
    a texture rebound per draw     overlapping geometry, nine depths
    the depth test                 a discarding fragment shader
    six fragment varyings          a 1280x720 framebuffer
    800 triangles per draw         four vertex elements, 60-byte stride

The sample still needs the descriptor comparison and this test never does. So
whatever is left is **not a property of the draws** -- it is something about
the sequence they sit in: the render passes around them, the draws between
them, the state changed between batches, or the frames before.

That is a different search from the one this has been, and a more productive
one to start fresh than to continue: capture the sample's whole command
stream with CUDAPIPE_DEBUG_DRAW and CUDAPIPE_DEBUG_BATCH and look at what
happens *around* a batch that renders wrongly, rather than at the batch.

The test that established all this is worth more than the answer would have
been. It parameterises draw count, triangle count, varying count, framebuffer
size and vertex layout, and it runs in a second.

### A vertex shader's inputs are numbered by attribute location

`pushconstants` drew its spheres as smooth RGB gradients where lavapipe and the
Gallium driver draw flat per-sphere colours -- and the shader is
`outColor = inColor * pushConsts.color.rgb`, so the gradient was a *vertex
attribute* being fetched from the wrong place. The elements were right
(offsets 0, 12 and 32 in a 96-byte stride, which is vkglTF::Vertex), so the
fetch was writing the right data into the right slots and the shader was
reading the wrong ones.

`nir_assign_io_var_locations(nir, nir_var_shader_in)` renumbers a shader's
inputs compactly. For a fragment shader that is right, because the varying
match is by `var->data.location` and the backend records it. For a **vertex**
shader it is wrong: the fetch kernel writes attribute N into slot N because
`cpvk_pipeline::velem` is indexed by
`VkVertexInputAttributeDescription::location`, so the driver location has to
be the attribute location. Whenever a pipeline's locations are not exactly
0..n-1 in declaration order, the shader reads a slot nothing wrote --
`pushconstants` read the normal where the colour is.

Vertex inputs now take `location - VERT_ATTRIB_GENERIC0` directly.
`pushconstants` goes from 5.098 to **0.000**, the ninth sample to match, and
nothing else moves.

Found by looking at the two images. The triage that led there -- comparing
distinct-colour counts between the native and Gallium renders of all ten wrong
samples -- flagged this one as the odd one out in a single table: 32,040
colours natively against 17.

### bloom and pbribl fail the same way: draws that shade nothing

`bloom` renders the ship correctly and has no glow at all -- no halo, no light
cone. The blend state is right (`blend=1 src=1 dst=1`, additive, on the two
composite draws), so the additive pass is adding an empty texture rather than
blending wrongly.

Its offscreen chain runs: 79 draws at 1280x720 and 75 at 256x256, which is the
glow target. But 31 of the 256x256 draws shade **zero** fragments, and so do 32
of the full-size ones.

That is exactly `pbribl`'s signature -- ten sphere draws, correct state, zero
fragments -- and it is now two samples with one failure mode rather than two
separate mysteries. Whatever makes a draw produce no coverage while its
framebuffer, viewport, matrices and vertex bindings all read correctly is
worth more than either sample: it is the single largest correctness gap left.

Both draw from a loaded model. `cpvk_mesh` reproduces that model's vertex
layout exactly and renders correctly, so it is not the layout -- but
`cpvk_mesh` draws one triangle from nine vertices, and these draw thousands.
Raising its vertex and triangle count toward a real mesh is the obvious next
step and the test already parameterises both.

### Render-to-texture then sample works

`tests/cpvk_rtt.c` renders into a sampled image -- clearing it to a colour by
`loadOp` in its own render pass -- and then samples it in the next pass. Both
drivers produce the rendered colour, byte-identical. So an image used as a
colour attachment and then read by a sampler is not the problem, which is what
`bloom`'s missing glow most looked like.

The suite is nine tests now and every one of them passes:

    cpvk_smoke     compute, memory, images
    cpvk_tri       a Gouraud triangle
    cpvk_draw      indexed, depth, clears, vertex formats
    cpvk_ubo       uniforms in both stages
    cpvk_tex       texture sampling
    cpvk_mesh      a glTF vertex layout, two render passes
    cpvk_batch     three draws, three descriptor sets
    cpvk_batchtex  nine draws, two set layouts, textures, overlap, discard,
                   six varyings, 1280x720, 800 triangles a draw, 60-byte stride
    cpvk_rtt       render to a texture, then sample it

Each was written to answer one question and each still answers it in about a
second. Between them they have cleared more than a dozen hypotheses that would
each have cost a turn of argument.

### pbribl's vertex shader outputs all zeros

The fragment dump works on a sample now and says it in one line:

    vtx0: slot0=[0.000 0.000 0.000 0.000] slot1=[0.000 ...] slot2=[0.000 ...]

Every output slot of every vertex is zero, so `gl_Position` is (0,0,0,0), w is
zero, every triangle is degenerate and nothing is covered. That is why the
spheres shade nothing, and it is a much narrower statement than "the draws
produce no coverage": the vertex stage produces nothing, and everything
downstream is behaving correctly given that.

Two things had to be fixed in the instrument before it could say so, and both
were bugs in the debug path rather than the driver:

- it fetched the whole vertex-output buffer of a 4,512-triangle draw to print
  six vertices
- it dereferenced `color_data` **on the host**, which is a device address. That
  works only when the colour target happens to be host-visible, which it is in
  every test here and is not in any sample. It is fetched with a
  `cuMemcpyDtoH` now.

An instrument that dies on exactly the cases worth investigating is worse than
no instrument, and this one had been dying since the first attempt to use it
several turns ago.

Next: the uniform buffer the shader reads was verified correct at the
descriptor, so either the shader is not reading it through the slot it was
compiled for, or the vertex fetch is handing it zeroed attributes. Printing
the fetched attributes beside the outputs distinguishes those, and the dump is
now the place to do it.

### pbribl: the fetch is right, the shader outputs zero

`CUDAPIPE_DEBUG_VFETCH` on a sphere draw:

    elem0 vb=0 off=0  stride=96 div=0 sz=12
    elem1 vb=0 off=12 stride=96 div=0 sz=12
    elem2 vb=0 off=24 stride=96 div=0 sz=8
    vfetch v0: e0=[0.130 0.065 -0.989] e1=[0.130 0.065 -0.989] e2=[0.729 0.521 0.000]

A unit-sphere position, the matching normal, and a uv. The vertex fetch is
correct, the element layout is correct, and `CUDAPIPE_DEBUG_FS` says every
vertex *output* is zero. So the shader is producing zeros from good inputs.

Its uniform buffer was already verified correct at the descriptor -- the first
floats read `0.974 1.732`, a projection matrix -- so what is left is how the
shader addresses it. The NIR dumps show pbribl's vertex shaders use two
different forms: one has eight `load_ubo` and no
`load_const_buf_base_addr_lvp` at all, another has nine base-address
intrinsics and sixteen `load_ubo`. Both forms are meant to work --
`emit_buffer_base` dispatches on whether the source is 32 or 64 bits -- and
this driver's own tests only ever produce the first.

Which form the sphere shader uses, and whether the mixed one resolves to the
right slot, is the next question. The dumps are in hand and the tools all work
now, which was not true an hour ago.

### Both UBO forms are used and both are implemented

The sphere shader is `dump@5143`: `inPos`/`inNormal`/`inUV` at GENERIC0/1/2,
matching the three elements at 0, 12 and 24. It has sixteen
`load_const_buf_base_addr_lvp` and twenty `load_ubo`, so it uses both
addressing forms.

Both are implemented and both are exercised by tests that pass:

- the **scalar** form is what this driver's push-constant lowering emits --
  `load_ubo(imm(slot), offset)` -- and `emit_buffer_base` sends a 32-bit
  source to `emit_const_buf_base`, which reads the stage's constant-buffer
  table.
- the **64-bit** form is what a descriptor-set read becomes here:
  `vulkan_resource_index` yields `(slot, offset)`, the second lowering pass
  turns that into `load_const_buf_base_addr_lvp(slot) + offset`, which is the
  descriptor's address, and `emit_buffer_base` dereferences it for the
  buffer pointer at `cpvk_descriptor::base`. `cpvk_ubo` reads a uniform buffer
  through a descriptor set and passes byte-identically.

So the addressing forms are not the difference either. What is established:
the vertex fetch delivers correct attributes, the uniform buffer is correct at
the descriptor, both ways the shader can reach it are implemented and tested,
and every vertex output is zero.

The next step is inside that shader: dump `dump@5143` in full and trace what
feeds its position output back to a source. Every layer around it has now been
checked.

## Outputs read back: the bug behind five samples

`pbribl`'s vertex shader does

    outWorldPos = locPos + pushConsts.objPos;
    gl_Position = projection * view * vec4(outWorldPos, 1.0);

which reads its own output back and leaves a `load_output` in the NIR. The
backend has no case for it and says so on every run:

    cudapipe: intrinsic 'load_output' is not implemented -- the shader using
    it computes on undef and will render wrong.

It computed on undef, every matrix multiplied a zero, every vertex landed at
the origin with w = 0, and every triangle was degenerate.
`nir_lower_io_vars_to_temporaries` -- which lavapipe runs and this driver did
not -- fixes it:

    instancing        11.851 -> 0.000
    texturecubemap    25.684 -> 0.000
    bloom             17.386 -> 0.000
    vulkanscene       15.430 -> 0.000
    particlesystem    41.036 -> 4.075
    pbribl             4.325 -> 1.472

Thirteen of eighteen samples are pixel-correct, and both replays are unmoved:
1,496 frames at 24.46 ms and 1,510 at 79.23.

That warning had been in the log since the driver first ran a capture, and I
read it several turns earlier and wrote it down as unrelated. What found it was
fixing `CUDAPIPE_DEBUG_FS` so it could print a vertex output on a real sample,
and then reading the NIR it pointed at.

### What is left

    particlesystem     4.075
    computeshader      4.427
    pbribl             1.472   spheres render; they reflect nothing
    texturemipmapgen   0.456
    multisampling    176.291   no resolve of a multisampled attachment

`pbribl`'s spheres are neutral grey where the reference is warm copper with
the plaza reflected in them. `texturecubemap` is exact now, so cube *sampling*
works and it is the environment cube's *generation* -- rendering into cube
faces -- that is missing or wrong.

### Array layers, honoured in two places, and neither explains pbribl

Two places took an image's base address where they should have taken a
subresource's:

- `vkCmdBeginRendering` ignored the colour view's `baseArrayLayer` and
  `baseMipLevel`, so rendering into a cube face or a mip level wrote to face
  zero of level zero.
- `vkCmdCopyImage` ignored both subresources' `baseArrayLayer`, so the six
  copies that build a cube map all landed on face zero.

Both are wrong by Vulkan's own definition and both are fixed. **Neither
changed any sample**, including pbribl, whose reflections are what prompted
looking: 1.472 before and after. Nothing regressed either -- 13/18 still
pixel-correct, nine unit tests still passing.

They are kept because a view's subresource selecting the layer is not a matter
of opinion, and because an image written to the wrong layer is the kind of
fault that surfaces later as something unrelated. But they are recorded here
as unproven: no test in this tree renders to or copies into a non-zero layer,
and adding one is the way to make them mean something.

pbribl's environment cube therefore comes from somewhere else again --
possibly rendered with `layerCount` greater than one in a single pass, which
this driver does not implement at all and does not warn about.

### gl_PointCoord: the driver named this one too

`particlesystem` draws point sprites and its fragment shader reads
`gl_PointCoord`. The driver said so on every run:

    cudapipe: intrinsic 'load_point_coord' is not implemented -- the shader
    using it computes on undef and will render wrong.

The renderer already carries point coordinates as a varying -- the rasteriser
writes them into the fragment stage's input slots like anything else -- so
what was missing was the lowering that turns the system value into that
varying. `nir_lower_sysvals_to_varyings` with `point_coord` set, which is what
lavapipe runs:

    particlesystem  4.075 -> 0.000

Only `point_coord`. lavapipe also converts `frag_coord`, `layer_id` and
`primitive_id`, and this backend implements those directly, so converting them
would be a change with no reason and a way to break three working things.

**Fourteen of eighteen.** That is two samples found by reading warnings the
driver had been printing since it first ran, after several turns of not
reading them. The remaining four:

    computeshader      4.427
    pbribl             1.472   spheres correct, reflections missing
    texturemipmapgen   0.456
    multisampling    176.291   no resolve

None of them prints a warning, which after today is worth stating as a
result rather than a note: the cheap failures are gone.

### computeshader: dispatches do not run in the order they were recorded

The left half of the frame -- the source texture -- is exact. The right half,
which a compute shader embosses, is **flat grey**, and an emboss of a constant
is a constant, so the dispatch runs and writes and its reads return the same
value everywhere.

The command buffer keeps two things: `cmd->ops`, an ordered list of draws,
copies and clears, and `cmd->dispatches`, a separate array. The executor runs
**every dispatch first**, then the ops. So a dispatch always precedes every
copy and draw recorded in the same command buffer, whatever order they were
written in, and a compute shader reading an image that a recorded copy fills
reads it empty.

That is a structural bug independent of this sample and it is the one to fix
next: dispatches belong in the ordered op list, not beside it. It is a real
change rather than a small one, which is why it is written down here rather
than attempted at the end of a session.

Note what is *not* wrong: storage image writes work, the dispatch grid is
right, compute descriptors resolve, and the graphics half of the same frame is
byte-exact. The single ordering rule accounts for the whole difference.

### Dispatches moved into the op list, and it fixed nothing

A dispatch is now `CPVK_OP_DISPATCH` in `cmd->ops`, executed in record order
beside draws, clears, copies and queries, instead of living in its own array
that the executor drained first. `cpvk_execute_cmd_buffer` is gone; what it
contained is `cpvk_execute_dispatch`, called from the op switch.

This is right -- a dispatch must run where it was recorded, and a compute
shader reading an image filled by a copy recorded before it was previously
reading it empty -- and **it changed no sample**: computeshader is 4.427
before and after, because it records its compute work in a separate command
buffer, where the old ordering happened to be equivalent.

Nothing regressed. 14/18 pixel-correct, nine unit tests pass, and both replays
are unmoved at 1,496 frames / 24.39 ms and 1,510 / 79.09.

So computeshader's flat grey is not ordering. What remains: the compute stage
reads its input through `imageLoad` on a storage image, and a read that
returns the same value for every texel points at the storage image's format
decode rather than at its address, since the address being wrong would give
garbage rather than a constant.

Two structural fixes in a row have now cost a turn each and moved no number.
Both were real bugs and both are worth keeping, but the pattern is worth
naming: reading the image first and the code second is what found the four
samples that did move today.

### computeshader: a storage image with no extent reads as entirely zero

The compute half of the frame was flat grey, and exactly grey: 128 in every
channel, everywhere. That is the answer written down. The sample's emboss
kernel sums to zero and the shader adds 0.5, so a convolution of *any*
constant image is 0.5 -- and a constant is what `imageLoad` returned.

The backend clamps a storage-image coordinate against the extent before
touching memory, reading width at descriptor offset 8 and height at 12, and
returns zero for a texel outside it, which is what `robustImageAccess`
requires. `struct cpvk_descriptor` had `pad0[16]` across exactly those bytes
and the driver never filled them. Width zero puts *every* coordinate out of
bounds, so robustness returned zero for the entire image.

Filling width, height and depth at the view's mip level:

    computeshader  4.427 -> 0.000

**Fifteen of eighteen.** Three remain: `pbribl` 1.472 with correct spheres and
no reflections, `texturemipmapgen` 0.456, `multisampling` 176.291 for want of
a resolve.

Worth comparing with the previous two turns, which fixed two real structural
bugs -- array layers in rendering and copies, dispatch ordering -- and moved
no number at all. This one came from reading a pixel value and asking what
produces exactly 128.

### texturemipmapgen: the chain is built; the level chosen differs

The frame differs on 7.4% of its bytes and only in the middle -- rows 212 to
507, columns 470 to 809 -- which is the distance where minification is
strongest and the deepest mip levels are sampled. Everything nearer the camera
is exact.

The chain itself is fine. No blit was refused: the driver warns on every blit
it cannot do and the sample produces none of those warnings. The scaling blit
box-filters the source footprint, and for the exact halving a mip generator
asks for, a box filter and a bilinear tap at the destination texel's centre
compute the same average, so the generated levels should agree.

What differs is which level is read. The native image is *blurrier* in that
region than the reference, which is a shader choosing a higher LOD, not a
level containing the wrong pixels. That points at the sampler's LOD
computation -- derivatives across the fragment quad -- and not at
`vkCmdBlitImage` at all.

That is worth stating precisely because the obvious reading of "the mipmap
generation sample is wrong" is that mipmap generation is wrong, and the
evidence says it is not.

### multisampling: not a detail

94.3% of bytes differ, every row and every column, max 254. This is not a
filtering or precision difference and there is no point measuring it further
until the driver resolves a multisampled attachment at all, which it does not.

### multisampling: the resolve worked; the clear did not

The model renders correctly and antialiased in both images, so the resolve
this driver added earlier is doing its job. The background is the difference,
and it is exact: white in the reference, **64 in every channel** in the native
frame, over 843,184 pixels.

64 is 255/4 rounded. The attachment has four samples, the clear wrote one of
them, and the resolve averaged 255 + 0 + 0 + 0.

A multisampled image keeps its samples as planes, and `cp_clear_rect` was
called once, on plane zero. Calling it once per plane:

    multisampling  176.291 -> 0.553

What is left is 0.553, and it is edge antialiasing rather than background:
sample positions, which this driver chooses itself. That is a different kind
of difference from the one just fixed and may be the point at which "identical
to lavapipe" stops being the right target for this sample.

Three samples remain, all now numerically close and none of them blank or
flat:

    pbribl            1.472   reflections
    multisampling     0.553   sample positions
    texturemipmapgen  0.456   LOD selection

The lesson from this turn is the same as the last one: the diagnosis was
"there is no resolve", written down twice in earlier sessions, and it was
wrong. One pixel value falsified it in a minute.

### pbribl: the albedo really is black, and the colour is the environment

The spheres' material colour is not missing. `CPVK_DEBUG_PUSH` shows what the
application pushes for each sphere:

    push off=0  size=12 -> -10.750 0 0                    objPos
    push off=12 size=24 ->  0.995 0.005 0.000 0 0 0       roughness, metallic,
                                                          specular, r, g, b

`r`, `g` and `b` are zero, and `#define ALBEDO vec3(material.r, material.g,
material.b)` -- so albedo is black in both drivers and always was. Every warm
tone in the reference image comes from the image-based lighting term, the
prefiltered environment cube multiplied by the BRDF lookup.

The native spheres are *exactly* neutral -- 40/40/40, 58/58/58, 106/106/106 --
so that cube reads grey. And the skybox behind them is byte-identical, which
says the raw environment cube samples correctly. What differs is the
**prefiltered** cube: a separate image the sample generates by rendering the
environment into each face at each mip level.

So the earlier guess was right for the wrong reason. It is not cube sampling
and not the material; it is the render-to-cube-face generation, which is also
where the array-layer fix two turns ago was aimed without moving the number.
The next question is narrow: whether those passes render into the face and
level their view names, and `CPVK_DEBUG_PUSH` shows they at least run, 72
bytes of matrix at a time.

### pbribl: the faces are placed correctly; explicit-LOD cube sampling is the suspect

`CPVK_DEBUG_RT`, added here, prints every render target the driver binds. For
pbribl:

    1 x 1280x720  layer=0 level=0    the frame
   61 x 512x512   layer=0 level=0    prefiltered environment
   42 x 64x64     layer=0 level=0    irradiance

Every one of them is a **plain 2D offscreen**, never a cube face. The sample
renders each face into that offscreen and then places it with
`vkCmdCopyImage`. So the array-layer fix in `vkCmdBeginRendering` two turns ago
is correct and, for this sample, unused -- and the copy path, fixed in the same
commit, is what does the placement.

That placement matches the layout. Images are laid out level-major with the
layers inside each level -- `offset += level_size[l] * array_layers` -- and the
copy adds exactly `baseArrayLayer * level_size[level]`. So the faces land where
the sampler looks, and the memory-layout explanation is eliminated.

What is left is how the cube is *read*. Both remaining users take an explicit
level:

    pbribl.frag:85       textureLod(prefilteredMap, R, lodf)
    prefilterenvmap:92   textureLod(samplerEnv, L, mipLevel)

and the skybox, which is byte-identical, uses plain `texture()`. That is the
distinction the evidence now points at: implicit-LOD cube sampling works,
explicit-LOD cube sampling is unverified, and it is used both to build the
prefiltered cube and to read it -- which would explain a result that is
self-consistently grey rather than merely wrong.

### pbribl: four explanations eliminated, and a process error worth recording

Checked and ruled out this turn, each with evidence:

1. **Face placement.** `CPVK_DEBUG_RT` shows 60 copies into the prefiltered
   cube -- six faces at each of ten mip levels, `128x128 src(l=0 lay=0) ->
   dst(l=2 lay=3)` into a `512x512 layers=6 mips=10` image -- and every plane
   lookup succeeds.
2. **Memory layout.** Images are level-major with layers inside each level,
   and the copy adds `baseArrayLayer * level_size[level]`, which is the same
   arithmetic.
3. **Explicit LOD.** `nir_texop_txl` sets `CP_TEX_LOD`, and the sampler uses
   `explicit_lod` directly rather than deriving one -- with a comment saying a
   prefiltered environment map indexed by roughness depends on it.
4. **Per-level strides in the sampler.** The view fills `row_stride[l]`,
   `img_stride[l] = level_size[l]` and `mip_offset[l] = level_offset[l]`, and
   `cp_fetch_texel` indexes all three by level.

So the cube is built where the sampler looks, and read the way the shader
asked. The remaining candidates are narrower: the R16G16B16A16_SFLOAT decode
on this path, or the prefilter shader's own output.

**The process error:** an earlier run of this same investigation printed
"0 copies" and nearly became the finding "the faces are never copied". The
count was zero because the `str.replace()` that inserted the print had not
matched anything -- the edit silently did nothing. The rule that caught it is
to assert that an edit changed the file before believing what it prints.

### pbribl: the write side is verified end to end, and the read is wrong

Reading the actual bytes settled what four turns of code-reading could not.

**What the prefilter pass renders** (`copysrc`, the offscreen before the copy):

    294a 2849 276e 3c00   ->  (0.041, 0.033, 0.029, 1.0)

**What lands in the cube's deepest level** (`copydst`, 1x1, 8 bytes):

    3237 30b2 302c 3c00   ->  (0.194, 0.147, 0.130, 1.0)

Both are warm, R > G > B. Every one of the sixty copies is issued -- six faces
at each of ten levels -- the view describes `target=3` cube, `enc=9`
R16G16B16A16_FLOAT, `levels=0..9`, and there are sixteen level slots, so
nothing is truncated.

So the prefiltered cube is correct in memory, at every level, including the
one the roughest sphere reads. The error is on the read side, and it scales
with LOD, which is the clearest signal available:

    x=150   roughness 0.995  lod ~9   native (1,1,1)       reference (25,17,10)
    x=1120  low roughness    lod low  native (106,106,106) reference (112,111,112)

Deep levels come back black; shallow ones are nearly right.

The next step is the one named last turn and not taken: a unit test that
samples a known cube at an explicit deep LOD and compares against lavapipe.
Reading more of the sampler is what the last four turns did, and the write
side being provably correct is what makes the test worth writing now.

### A tenth test: explicit level of detail

`cpvk_lod` samples a four-level texture -- red, green, blue, white, one flat
colour per level -- with `textureLod` at a level each fragment picks from its
own x coordinate, so a wrong level shows as a wrong colour rather than a
blend. Native output is **byte-identical to lavapipe**.

That eliminates explicit-LOD sampling of a 2D texture, which is what pbribl's
`textureLod(prefilteredMap, R, lodf)` was suspected of. Whatever is wrong there
is specific to a **cube**, where the deepest levels are one and two texels per
face and the face selection runs before the level fetch.

Ten tests now pass: smoke, tri, draw, ubo, tex, mesh, batch, batchtex, rtt,
lod.

The value of the test is not only the answer. It runs in under a second
against either driver, where the same question asked through pbribl costs a
minute and a sample's worth of confounds.

### Two formats decoded as the wrong one by accident

`VK_FORMAT_R16G16_SFLOAT` and `VK_FORMAT_R16_SFLOAT` carried texel encoding
`0` in the format table. Zero is not a sentinel here -- it is
`CP_TEXEL_R8G8B8A8_UNORM`, the first enumerator -- so every sample of those
formats decoded half-float pairs as four bytes. The sampler has had
`CP_TEXEL_R16G16_SFLOAT` and `CP_TEXEL_R16_SFLOAT` all along; only the table
entry was missing. `VK_FORMAT_A2B10G10R10_UNORM_PACK32` was the same and is
fixed with them.

pbribl's BRDF lookup table is R16G16_SFLOAT, and its texture handle now prints
`enc=28` where it printed `enc=0`.

    pbribl  1.472 -> 1.255

The spheres, however, are pixel-for-pixel unchanged and still *exactly*
neutral, so the gain came from elsewhere in the frame and the image-based
lighting term is still zero. Since the lookup table now decodes, what is left
is its **content**: unlike the cube it is rendered and never copied, so the
`copydst` instrument cannot see it.

### Three tests that each eliminated a hypothesis

    cpvk_lod         2D, four levels, textureLod        identical to lavapipe
    cpvk_cubelod     cube, six faces x four levels      identical to lavapipe
    cpvk_cubelodf16  the same cube as R16G16B16A16      identical to lavapipe

The cube test reads back the exact face and level it asked for at every depth
-- red names the face, green names the level -- so explicit-LOD cube sampling,
which was the standing explanation for pbribl across three turns, is dead.

Twelve tests pass now. Writing them cost about one turn and removed four
hypotheses that code-reading had failed to remove across three.

### The BRDF table is correct, and so was the arithmetic that said it mattered

A new instrument, `rtout`, reads back a render target when the next pass binds
over it -- the one class of intermediate this driver could not inspect, since a
copy can be watched on both sides but an attachment that is rendered and then
only sampled had nothing watching it.

pbribl's BRDF lookup table, its first 512x512 pass:

    rtout 512x512:  1cfc 3bf4  2374 3be1  2630 3bce
                    (0.005, 0.994) (0.014, 0.985) (0.024, 0.975)

Plausible scale and bias pairs, not zeros. The table is right.

Worse for the theory: the *old*, wrong decode read those same bytes as
R8G8B8A8 and produced (0.988, 0.11) -- also not zero. So the format fix,
though a real bug, could never have been what zeroed pbribl's ambient term,
and the pixels agreeing before and after now make sense.

`ambient = reflection * (F * brdf.x + brdf.y)` with `brdf.y` near 1 is
essentially `reflection`. For the spheres to come out *exactly* neutral,
`reflection` itself has to be neutral. That is the prefiltered cube again --
whose memory is provably warm at every level, and whose sampling `cpvk_cubelod`
reproduces exactly.

Those two facts cannot both hold for the same texture, so the next question is
whether the shader is reading the texture it thinks it is: the fragment stage
has five sampler table entries, and one wrong index would sample a different
image with a sampler whose `lod=0.0..0.0` clamps every level to zero.

### pbribl's scene descriptors never reach the driver

Instrumenting `vkUpdateDescriptorSets` at its entry point shows **seven** calls
in the whole run, every one with `descriptorCount = 1`:

    updsets n=1 -> bind=0 type=6   (five of these, uniform buffers)
    updsets n=1 -> bind=0 type=1   (two of these, combined image samplers)

But pbribl's `setupDescriptors()` calls it twice with five and three writes,
binding 0 through 4 of the scene set -- the two uniform buffers and the three
images the lighting depends on, `samplerIrradiance`, `samplerBRDFLUT` and
`prefilteredMap`. **Neither call arrives.** The seven that do are the
single-binding sets the BRDF, irradiance and prefilter passes make for
themselves.

So the fragment stage's five sampler entries are never pointed at those three
textures, which explains a reflection term that is neutral no matter how
correct the cube is -- and it explains it without contradicting either
`cpvk_cubelod` or the warm bytes in the cube's memory, which is what every
previous theory failed to do.

What it does not yet explain is how the *vertex* stage gets its matrices,
since the spheres are positioned correctly and that comes from binding 0 of
the same set. That inconsistency is the next thread, and it is a much sharper
one than "the reflections are grey".

Two truncation errors were made getting here, both the same shape: reading
`head` output as if it were the whole of it. The count that mattered came from
`wc -l`.

### Correction: pbribl's descriptors are written correctly

The previous entry -- "pbribl's scene descriptor writes never reach the
driver" -- **is wrong**, and the way it was wrong matters more than the claim.

The run that produced it **segfaulted**. `rtout`, the instrument added to read
back a rendered attachment, held a pointer to the previous pass's colour
buffer and dereferenced it after that buffer was gone. The process died before
`setupDescriptors()` ran, so the seven single-binding updates it had already
logged looked like the complete set.

With that instrument removed, the same command exits 0 and logs eleven calls:

    7 x updsets n=1     the filter passes' own sets
    2 x updsets n=5     the scene set, bindings 0..4
    2 x updsets n=3     the skybox set, bindings 0..2

and twenty-three descriptor writes landing at exactly the right flat indices,
binding *n* to flat *n*. `samplerIrradiance`, `samplerBRDFLUT` and
`prefilteredMap` are all written where the shader looks for them.

`rtout` is deleted. Its one real result -- the BRDF table's contents, logged
before the crash and still valid -- is recorded above. A debugging tool that
silently truncates the run it is measuring is worse than no tool, because
every count taken from that run is quietly a lower bound.

That is three findings in three turns overturned by re-measuring: the
multisample resolve that already worked, the format decode that could not have
zeroed anything, and now this. All three shared a shape -- a number read from a
run that was not what it appeared to be.

### What the last two samples actually are, measured

**multisampling, 0.553.** Sample positions are not the cause: cudapipe's four
are `(0.375,0.125) (0.875,0.375) (0.125,0.625) (0.625,0.875)`, which is
`lp_sample_pos_4x` exactly. The difference is edge coverage and nothing else:

    max-channel diff > 1    2.285% of pixels
                     > 16   1.448%
                     > 64   0.080%
                     > 128  0.008%

    mean reference gradient where the diff exceeds 16 ... 129.1
    mean reference gradient over the whole frame .........  5.1

A twenty-five-fold concentration on high-gradient pixels. Every substantial
difference sits on a silhouette, mostly the spacecraft's thin boom, where
native has slightly less coverage. This is a fill-rule difference on
sub-pixel geometry, not a structural fault.

**texturemipmapgen, 0.456.** The opposite signature. The differences are
*not* on edges -- gradient 15.0 where they exceed 16, against 13.0 over the
frame -- and native is systematically **brighter** there, 51.8 against 44.0.
That is a filtering or level-selection difference spread across a minified
region, which matches the earlier reading that the chain is built correctly
and the wrong level is read.

    max-channel diff > 16   0.882% of pixels
                     > 64   0.004%

Both are now bounded and characterised rather than merely nonzero, and the two
have different causes, which the single mean-error number had hidden.

### A thirteenth test isolates texturemipmapgen to a fractional LOD bias

`cpvk_lodimp` is `cpvk_lod` with the level left to the sampler: the same four
flat-coloured levels, the coordinate multiplied so each band is minified by a
different factor, and a real derivative taken across the fragment quad. It is
the first test in this tree that **differs from lavapipe**, and it differs
small and systematically -- 641 pixels of 4,096, maximum 8 of 255.

Because each level is one flat colour, the blend is invertible. A pixel in the
band that lands between level 0 (red) and level 1 (green) gives the blend
fraction directly from its red channel, over 227 samples:

    lavapipe blend fraction   0.6773
    native   blend fraction   0.7141
    native LOD higher by      0.0368 levels   (sd 0.0087)
    which is a rho ratio of   1.0259

So this driver's implicit level of detail is about 2.6% larger than
llvmpipe's, consistently, and it always reads slightly deeper into the chain.
That is exactly `texturemipmapgen`'s signature -- not on edges, spread over
the minified region, and brighter where the deeper levels are brighter.

Explicit LOD is exact (`cpvk_lod`, `cpvk_cubelod`, `cpvk_cubelodf16` are all
byte-identical), so the difference is in deriving rho from the quad, not in
using it.

Thirteen tests, twelve of them byte-identical and the thirteenth measuring a
number rather than asserting one.

### Correction: the LOD bias is real, shared, and not texturemipmapgen's cause

The measurement in the previous entry is right and its conclusion is wrong.

`cpvk_lodimp` run against all three drivers:

    native            vs gallium-cudapipe   IDENTICAL
    native            vs lavapipe           DIFFER
    gallium-cudapipe  vs lavapipe           DIFFER

The 2.6% larger rho is **cudapipe's, not the native driver's**. Both cudapipe
drivers share `cp_sampler.cu`, and llvmpipe computes its level with
`lp_build_fast_log2`, a polynomial approximation, where this sampler uses
CUDA's `__log2f`. Two different approximations of the same quantity, differing
by 0.0368 levels. Nothing here regressed and nothing here is new.

`texturemipmapgen` is measured against the **Gallium cudapipe** driver, which
has the identical sampler, so implicit LOD cannot explain its difference. This
is the mistake the project notes already warn about -- comparing against the
wrong reference -- and it took a two-line check to catch, after a full entry
had been written on the wrong premise.

What the test is still worth: it pins native and Gallium byte-identical on
implicit level-of-detail selection, which is a real property and was not known
before.

**Where that points instead.** The two drivers share the sampler but *not* the
mip generation. `texturemipmapgen` builds its chain with `vkCmdBlitImage`, and
this driver box-filters the source footprint on the host while the Gallium
path goes through lavapipe's blit. A small difference in the generated levels,
spread over a minified region and not concentrated on edges, is exactly the
signature measured. That is the next thing to test, and `cpvk_lodimp` can be
turned into the test by generating its levels with a blit instead of writing
them.

### The native driver advertises Vulkan 1.0

Found by accident, and it is the largest capability difference between the two
drivers yet measured:

    native   apiVersion 1.0.354
    gallium  apiVersion 1.4.354

A test calling `vkCmdBlitImage2` -- core since 1.3 -- segfaulted at address
zero with the native driver and ran fine with the Gallium one. Not a driver
crash: the loader has no entry to dispatch, so the call goes to a null
pointer.

`cpvk_device.c` says why 1.0 was chosen. `KHR_get_physical_device_properties2`
was advertised and withdrawn because at apiVersion 1.0 the KHR alias
entrypoints are not wired to the runtime's core implementations, and an
application that enables the extension jumps through a null pointer. The note
ends "it comes back with apiVersion 1.1, or with the aliases implemented
explicitly", and neither has been done.

This is not a regression -- both replays and all eighteen samples run, because
they use extensions and 1.0 entry points -- but it is a real gap against a
driver that reports 1.4, and any application built against 1.3 core will fault
rather than fail cleanly. Worth fixing before the Gallium driver is deleted.

### A fourteenth test: blit-generated mip chains match

`cpvk_lodblit` writes a checkerboard into level 0 and builds levels 1 to 3
with `vkCmdBlitImage`, exactly as a mip generator does, then samples the
result with implicit level of detail. Native and Gallium are **identical**.

So mip generation is not `texturemipmapgen`'s cause either -- which was this
turn's hypothesis, and is now the fourth eliminated for that sample. Both
drivers share the blit implementation as well as the sampler, so the surviving
explanation has to be something they do not share.

### apiVersion raised to 1.1

The note in `cpvk_device.c` said the withdrawn extension "comes back with
apiVersion 1.1, or with the aliases implemented explicitly". Raising it is the
one-line half of that, and it works: `vkGetPhysicalDeviceProperties2`, a 1.1
core entry point that was a null pointer before, now returns
`cudapipe (NVIDIA GeForce RTX 5090), api 1.1`.

Nothing moved:

    samples          18/18 run, 15/18 pixel-correct, the same three remaining
    unit tests       13/13
    Crossroads       1,496 frames, 24.35 ms   (24.33 before)
    old capture      1,510 frames, 79.11 ms   (79.08 before)

This does not close the gap -- the driver it replaces reports 1.4, and 1.2 and
1.3 core entry points such as `vkCmdBlitImage2` are still not dispatched at
1.1. It does remove the specific trap the file documented, and
`KHR_get_physical_device_properties2` can now be advertised without the
segfault that took it out.

### 1.3 was tried and rejected, on evidence

Raising `apiVersion` to 1.3 was tempting -- it is what makes `vkCmdBlitImage2`
and the rest of the 1.3 core dispatch -- and everything passed under it: all
eighteen samples ran, fifteen pixel-correct, the replays gave 1,496 frames at
24.35 ms and 1,510 at 79.22.

It is still wrong, and one check shows why. A device created the way a real
1.3 application creates one, asking for `VkPhysicalDeviceVulkan13Features`
with `dynamicRendering` and `synchronization2`:

    vkCreateDevice ... VK_ERROR_FEATURE_NOT_PRESENT

Vulkan 1.3 makes a long list of features mandatory -- `synchronization2`,
`inlineUniformBlock`, `privateData`, `maintenance4`,
`shaderTerminateInvocation`, `subgroupSizeControl` among them -- and
`cpvk_get_features` declares exactly one, `dynamicRendering`. A version number
is a promise about what may be called, not a report of what happens to work on
the samples at hand. Advertising 1.3 would have every 1.3 application fail at
device creation instead of at the feature it actually wanted, which is the
same trap that took `KHR_get_physical_device_properties2` out at 1.0.

So it stays at 1.1, which is honest and which fixed a real null dispatch. The
gap to the Gallium driver's 1.4 closes by implementing those features, not by
editing the number -- and that is now written where the number is.

### Vulkan 1.3: events stubbed, features declared, and one thing still broken

Per instruction, entry points the sweep and the replays do not need are
implemented as empty. The event API is the case in point: `vkCreateEvent`,
`vkSetEvent`, `vkResetEvent` and `vkGetEventStatus` are real, because they are
questions with answers, and `vkCmdSetEvent2`, `vkCmdResetEvent2` and
`vkCmdWaitEvents2` are empty, because one in-order stream already satisfies
them -- the same reasoning that makes `vkCmdPipelineBarrier2` a no-op here.

With those present, the twelve features Vulkan 1.3 makes mandatory are
declared, and both version numbers raised. Two were needed, not one: the
physical device's `apiVersion` **and** `vkEnumerateInstanceVersion`, since the
instance version caps what the loader will dispatch above it.

Measured at 1.3:

    vkCreateDevice with VkPhysicalDeviceVulkan13Features   VK_SUCCESS (was -8)
    samples          18/18 run, 15/18 pixel-correct, same three remaining
    unit tests       13/13
    Crossroads       1,496 frames, 24.34 ms
    old capture      1,510 frames, 79.20 ms

**Still broken, and stated plainly:** `vkGetDeviceProcAddr` returns NULL for
`vkCmdBlitImage2`, `vkCmdBeginRendering` and `vkCmdPipelineBarrier2`, and a
test that requests apiVersion 1.3 in its `VkApplicationInfo` and calls
`vkCmdBlitImage2` still faults at address zero. The driver implements all
three under their core names. So this is a dispatch-gating problem in how the
runtime tables are built, not a missing implementation, and it is not yet
found.

That leaves the version number ahead of one thing it promises. It is kept
because every measurement improved or held -- device creation for a 1.3
application went from failing outright to succeeding -- but the gap is
recorded here rather than left for someone to discover by faulting.

### The ICD manifest is the version gate, and raising it costs 21-39% of replay

The reason 1.1-and-later core entry points dispatched to NULL was none of the
things checked: not the entrypoint generation, which is byte-identical to
lavapipe's, not `vkEnumerateInstanceVersion`, and not the physical device's
`apiVersion`. It was the **ICD manifest**:

    native   "api_version": "1.0.354"
    gallium  "api_version": "1.4.354"

The loader reads that and rewrites the application's requested `apiVersion`
down to it before calling in, so everything above 1.0 resolves to nothing
however high the driver reports internally. With `--api-version 1.3` in
`meson.build` the whole set comes alive:

    vkCmdBeginRendering  (nil) -> 0x...bb50   the same pointer as the KHR alias
    vkCmdBlitImage2      (nil) -> 0x...e030   and the test that faulted now
                                              produces the 1.0 path's image

**And it costs, measured both ways:**

    manifest 1.3    Crossroads 29.34 ms   old capture 109.76 ms
    manifest 1.0    Crossroads 24.34 ms   old capture  79.20 ms
    reverted        Crossroads 24.40 ms   old capture  79.20 ms

21% and 39%, because gfxrecon-replay takes its Vulkan 1.3 paths through the
driver once the manifest says it may. The objective is no regression, so the
manifest stays at 1.0 and the reason is written beside it.

Everything behind the manifest is ready and stays: the twelve mandatory 1.3
features are declared, the event API exists, and `vkCreateDevice` with
`VkPhysicalDeviceVulkan13Features` succeeds. One line in `meson.build` turns it
on the day those paths are as cheap as the ones they replace.

### Where the native driver's 3.4x on replay actually is

Not the host. Sampling GPU utilisation during the Crossroads replay, with both
processes confirmed still running at the end of the window:

    native   63.2% busy (median 64%)
    gallium  47.7% busy (median 61%)

A first attempt at this compared 0.5% against 65.8% and meant nothing: the
Gallium replay had already *finished* inside the sampling window, because it is
3.4x faster. Checking the process was alive is what made the second reading
usable.

Comparable occupancy over 3.4x the wall clock means the native driver is
issuing roughly 3.4x the GPU work, not waiting on the host for it. And the
thing the Gallium driver does that this one does not is merge draws.

Batching, measured on the capture rather than argued about:

    CPVK_BATCH off                     24.36 ms
    CPVK_BATCH=1                       24.19 ms
    CPVK_BATCH=1 CPVK_NO_DESC_KEY=1    24.21 ms

Nothing, either way, because the descriptor-content key stops the merges.

**And the key should not be needed.** `cp_batch_record` already snapshots each
draw's vertex- and fragment-stage constant buffers into its own row of the
batch table, and says so in a comment that calls them "the bindings the key
stopped comparing". Per-draw uniforms are exactly what the table exists to
carry. So the recorded fact that dropping the key renders `gltfscenerendering`
wrong -- 20.768 against 0.000, three runs each way -- contradicts the design of
the code it sits next to.

That contradiction, not the batching front end, is the whole remaining
performance gap. It is now the sharpest open question in this driver: something
differs per draw, is read through the fragment stage, and is *not* in the row
that `cp_batch_record` writes.

### The descriptor key's necessity, bisected: it breaks at three draws, not two

`CUDAPIPE_BATCH_MAX` turns the long-standing open question into a measurement.
`gltfscenerendering` with `CPVK_BATCH=1 CPVK_NO_DESC_KEY=1`, mean absolute
difference against the reference:

    BATCH_MAX  1     0.000
               2     0.000
               3     5.516
               4     7.304
               6    20.129
               9    20.768
              16    20.768

Three runs each at 2 and 3: `[0.0, 0.0, 0.0]` and `[5.515, 5.516, 5.515]`. The
boundary is exact and the error grows with the number of draws sharing a
launch, which is the signature of a batch using one draw's state for all of
them.

Two draws merge correctly without the key; three do not. `cpvk_batchtex` has
three draws, three descriptor sets and three textures, and merges correctly
without it -- so the count alone is not the discriminator either, and whatever
gltfscenerendering has that the test lacks is now bounded by "visible only at
three or more".

Things eliminated on the way: the row width is not it -- `CP_ARG_UBO_STRIDE`
and `CP_MAX_CONST_BUFFERS` are both 16, so rows cannot spill. The binding
counts are not it -- `num_vs_ubos` and `num_fs_ubos` are set to
`CP_MAX_CONST_BUFFERS` for every draw, so the widths always agree.

What is left is the mechanism that maps a *fragment* back to its draw's row:
`cp->fs_batch.slices`, a table of per-draw vertex spans that the fragment
stage searches by primitive. It is set only when `batch_draws > 1`, and the
opaque path's version is the one gltfscenerendering takes. A search that
returns the wrong row for the third and later draws would produce exactly this
curve.

### Draw count eliminated; clipping is the remaining difference

`cpvk_batchtex` has `#define NDRAW 9` -- nine draws, nine descriptor sets,
nine textures -- and merges correctly without the descriptor key. The note in
`cpvk_draws_mergeable` saying the untested case was "nine draws there, three
here" had it backwards. Batch size is not the discriminator.

Nor is the row search. `cp_write_batch_rows` does a textbook last-slice-not-
past-this-vertex binary search, correct for any table length, and
`prim_shift = 3` matches `CP_CLIP_MAX_OUT = 8`. (Two comments describe that as
"four output slots per input triangle" and a "4x slot array"; the constant is
8 and the shift agrees with the constant, so the comments are stale and the
code is right.)

What is left is **clipping**. The condition is explicit:

    bool stable_clip = batch_draws > 1 &&
       (cp->blend_enabled ||
        (cp->fs_shader && cp->fs_shader->reads_const_bufs));

and its comment says exactly what is at stake: "the primitive index has to
name the input triangle, or cp_write_batch_rows() maps fragments to the wrong
draw's material". `gltfscenerendering` draws a large model that clips;
`cpvk_batchtex` draws nine small bands that do not. So the test exercises the
`prim_shift = 0` mapping and the sample exercises the `prim_shift = 3` one,
and only the sample is wrong.

The experiment that settles it is small: move `cpvk_batchtex`'s geometry so its
triangles cross the viewport edge and have to be clipped, then run it with
`CPVK_NO_DESC_KEY=1`. If it fails, the descriptor key has been standing in for
a stable-clip row-mapping bug all along, and there is a one-second
reproduction for it instead of a sample.

### Clipping eliminated too -- the test already exercised it

The experiment was to make `cpvk_batchtex`'s geometry clip and see whether the
descriptor key became necessary. Instrumenting the decision first, which is
what the last few turns have taught:

    clipped variant   clip: tris=6400 batch_draws=8 stable=1 reads_cb=1
    original test     clip: tris=6400 batch_draws=8 stable=1 reads_cb=1

The **original** test already takes the stable clip path, with eight draws
merged, and passes without the key. So clipping is not the discriminator, and
the variant was unnecessary -- both are byte-identical to their unbatched
output with and without the key.

Had the variant been run without checking, its passing would have been read as
"clipping is not it" for the right reason by luck; the instrument shows the
premise was wrong from the start.

    gltfscenerendering  clip: tris=45184 batch_draws=3 stable=1 reads_cb=1

Same path, same flags, three draws. What differs now is **scale**: 45,184
triangles in one batch against 6,400 in the test, and the stable path
rasterizes `num_triangles * CP_CLIP_MAX_OUT` slots -- 361,472 against 51,200.
`CP_MAX_BATCH_TRIS` is 262,144, so the batch itself is under the limit while
the slot array it expands into is not.

That is the next thing to check and the first candidate that explains the
"three draws" boundary without hand-waving: three of gltfscenerendering's
draws are the point where the expanded slot array crosses something the test's
never approaches.

### Scale eliminated; the fast test now covers everything but the answer

`cpvk_batchbig` is `cpvk_batchtex` with `TPD` raised to 6,000 -- 48,000
triangles in an eight-draw batch, past `gltfscenerendering`'s 45,184, and past
it in expanded slots too (384,000 against 361,472):

    clip: tris=48000 batch_draws=8 stable=1 reads_cb=1
    key   vs off: IDENTICAL
    nokey vs off: IDENTICAL

Scale is not the discriminator either.

The test's coverage, read from its own source rather than assumed:

    nine draws, merged eight at a time
    two descriptor sets: set 0 bound once, set 1 rebound per draw
    nine textures, one per draw
    indexed draws, differing firstIndex
    the stable clip path
    48,000 triangles

That is `gltfscenerendering`'s structure in every respect anyone has named, and
it merges correctly without the descriptor key.

Eliminated across three turns, each with a measurement: batch size (the test
has nine draws, not three), row width, binding counts, the row binary search,
`prim_shift`, clipping, and now triangle count.

What the sample still has that the test does not is its **data**: per-draw
vertex offsets, materials that differ in pipeline state, and a draw order the
test does not reproduce. The next diagnostic is not another hypothesis but
`CUDAPIPE_DEBUG_BATCHDIFF` run on the sample twice -- once with the key and
once without -- to see exactly which draws merge when it is removed. The
answer is in that difference and nowhere else.

### What the descriptor key is actually preventing, counted

`CUDAPIPE_DEBUG_BATCHDIFF` on `gltfscenerendering` at `BATCH_MAX=3`, the same
frame each way:

    with the key      20 separations: 10 "the draws differ",
                                       8 "descriptors",
                                       2 "fragment shader"
    without the key    6 separations:  3 "the draws differ",
                                       3 "fragment shader"

So removing it enables exactly **eight merges**, and one or more of those eight
is wrong. Everything else about the frame is unchanged.

Those eight draws differ *only* in their descriptor sets -- that is what the
diagnostic means -- which is precisely the case the per-draw table exists to
handle, and precisely the case `cpvk_batchtex` merges correctly eight at a
time with two sets, nine textures and 48,000 triangles.

Checked this turn and not the cause: every one of the six `cp_batch_flush`
call sites passes `cp->batch.fs_ubos`, the per-draw fragment table, so no
flush path silently falls back to the live bindings.

The question is now as small as it can be made without new instrumentation:
eight draws that differ only in descriptors merge, and the frame is wrong by
5.516. The next step is to print the rows themselves -- the `fs_ubos` values
recorded for each draw of that batch, and the row each fragment resolves to --
rather than to reason about which stage could be at fault. Every structural
hypothesis has been eliminated by measurement; what is left is the data.

### The per-draw rows are right, so the fault is on the fragment side

`CPVK_DEBUG_ROWS` prints what `cp_batch_record` stores for each draw of a
batch. `gltfscenerendering` at `BATCH_MAX=3` without the key:

    row 0: [0]=0x7eeabc000800 [1]=0x7eeacc11e000 [2]=0x7eeacc11e0c0 [3]=0x...100000
    row 1: [0]=0x7eeabc000900 [1]=0x7eeacc11e000 [2]=0x7eeacc11e140 [3]=0x...100000
    row 2: [0]=0x7eeabc000a00 [1]=0x7eeacc11e000 [2]=0x7eeacc11e1c0 [3]=0x...100000

Exactly what it should be. Slot 0 is the push-constant block and differs per
draw; slot 1 is the scene set, correctly identical across the batch; slot 2 is
the material set and differs per draw, 0x80 apart in the snapshot arena; slot
3 is shared.

A first version of this print looked only at slot 1, saw one address for every
draw, and looked exactly like the bug -- every merged draw pointing at one
material. Slot 1 is the set they legitimately share. Printing four slots
instead of one turned a false positive into the opposite conclusion.

So the recorded data is correct and distinct, and the batch table is not the
fault. Combined with everything already eliminated -- batch size, row width,
binding counts, the row search, `prim_shift`, clipping, triangle count, and
every flush path passing the table -- what is left is the fragment side: how a
fragment resolves *which* row it belongs to, and whether the kernel is handed
`fs_batch.slices` and `ndraws` at all for this batch.

That is one print away and it is the next thing.

### The slice table is correct too -- and a correction about why any of this matters

`CPVK_DEBUG_ROWS` at the launch that shades the batch:

    exec:   batch_draws=3 num_draws=1 fs_tbl=1 fs_ndraws=3
    slices: n=3 set=1 verts=[0 30504 48132]

The table is handed to the kernel, the row count is right, and the vertex
spans are the exact cumulative sums of the three draws' index counts -- 30504,
then 30504+17628 = 48132. `prim_shift` is 3 and `CP_CLIP_MAX_OUT` is 8, so
`(prim >> 3) * 3` lands in the same units the spans are in.

So the fragment side is correct as well, and the list of things eliminated by
measurement now covers every structural element of the batched path.

**The correction.** This mattered because merging was supposed to close the
3.4x replay gap. It does not, and the measurement that says so is two turns
old and was under-weighted at the time:

    CPVK_BATCH off                     24.36 ms
    CPVK_BATCH=1                       24.19 ms
    CPVK_BATCH=1 CPVK_NO_DESC_KEY=1    24.21 ms

Dropping the descriptor key moves the Crossroads replay by 0.02 ms against a
7.17 ms target. Whatever makes the native driver 3.4x slower than the Gallium
one on that capture, it is **not** the descriptor key and not the merges the
key prevents. The `gltfscenerendering` correctness bug is real and still open,
but it was being chased as a performance blocker and it is not one.

Two turns went into it after the number that ruled it out had already been
recorded. The rule that would have caught it: before chasing a blocker, check
that removing it changes the thing it is supposed to be blocking.

### The 3.4x replay gap, measured to its cause: no A-buffer pass episodes

A 20-second CUDA trace of the Crossroads replay through each driver,
normalised by the frames each managed in that window (native 820, gallium
2,789):

    kernel                  native/fr  gallium/fr   ratio
    cp_peel_advance             109.3         0.1    1631
    cp_abuf_worklist            220.6         1.8     121
    cp_abuf_scan_add            329.8        13.4      25
    cp_abuf_scan_block          550.3        23.0      24
    cp_abuf_sort                110.3         4.8      23
    main                        287.5        37.3       8
    cp_vertex_fetch             143.3        22.2       6
    cp_abuf_seg_count             0.0         3.0       -
    cp_abuf_seg_scatter           0.0         2.4       -
    cp_abuf_sort_short            0.0         3.9       -
    TOTAL                      3381.4       245.4    13.8

**3,381 launches a frame against 245.** The three kernels at exactly zero name
the cause: `cp_abuf_seg_count`, `cp_abuf_seg_scatter` and `cp_abuf_sort_short`
belong to the A-buffer *pass episode*, where many blended draws share one
build, one sort and one peel. The native driver never runs them.

`CPVK_DEBUG_PASS` shows `cp_pass_appendable()` is never called at all, with or
without `CPVK_BATCH=1`. It is reached only from a batch flush, and
`cpvk_batch_can_join` ends:

    /* The blended path merges through the A-buffer ... the blended half is
     * left to the single-draw path until there is a test that shows it
     * right. */
    return false;

Every blended draw is refused, so no blended batch is ever staged, so no batch
ever flushes into an episode, so every blended draw builds and peels its own
A-buffer. `cp_peel_advance` at 109 launches a frame against gallium's 0.1 is
that sentence in numbers.

So the remaining performance gap is not batching in the sense that was being
chased for the last several turns -- merging draws into one launch -- but the
episode path built on top of it, which the front end has never enabled. The
comment says what it needs: a test that shows the blended merge right.
`cpvk_batchtex` and `cpvk_batchbig` are that test for the opaque half already.

### Blended pass episodes: correct, entered, and worth nothing measurable

Three real defects were found and fixed to get here:

1. `cpvk_batch_can_join` refused every blended draw outright.
2. It computed the blended flag into a local and discarded it.
3. `cp->batch.blended` was hardcoded `false` when staging, so even a blended
   batch took the opaque branch at flush and no episode was entered.

With all three fixed and `CPVK_BATCH=1 CPVK_BATCH_BLEND=1`, `cp_pass_appendable`
is called for the first time, and `cpvk_batchblend` -- a new test, nine blended
draws over nine textures at nine depths with depth writes off -- is
**byte-identical** to its unbatched output. The sweep is unchanged: 18/18 run,
15/18 pixel-correct, the same three.

**The performance claim does not survive.** A first A/B in one loop read 24.39
ms off, 24.22 batched, 9.74 with blended episodes, and that looked like 2.5x.
Re-running off and blend twice each says otherwise:

    native off     9.75   9.76
    native blend   9.74   9.76
    gallium        7.15   7.16

The flag changes nothing. What moved was the **baseline**: this capture
measured 24.3-24.4 ms all day and now measures 9.75 without any driver change
that could explain it. The sequential A/B was confounded by whatever warmed --
the on-disk shader cache is the obvious candidate -- and the third run in the
loop got the benefit.

So two recorded numbers are wrong and both are mine: the native replay is
9.75 ms against gallium's 7.15, a factor of 1.36 and not 3.4, and the earlier
3,381-launches-per-frame trace was taken in that same unwarmed state.

The change is kept because it is correct and because the defects were real, but
it is behind `CPVK_BATCH_BLEND` and claims nothing. Re-running an A/B in the
opposite order is what would have caught this immediately, and it is cheap.

### The replay numbers, corrected: the CUDA JIT cache was the variable

`~/.nv/ComputeCache` is 446 MB across 9,172 files. 208 of them were written in
the three hours of this session and **none** in the last thirty minutes. The
native driver generates PTX and hands it to `cuModuleLoadData`, which JIT
compiles to SASS and caches the result on disk; every measurement taken while
that cache was filling was measuring compilation as well as rendering.

Warm, same session, both drivers, both captures:

    capture      native    gallium   ratio
    Crossroads    9.73 ms   7.15 ms   1.36
    old capture  41.73 ms  25.25 ms   1.65

The Gallium figures reproduce what was recorded months ago -- 7.17 and 25.20 --
so that driver was always measured warm and its numbers stand. The native
driver's recorded 24.4 and 79.2 do not: they were taken with a partly cold JIT
cache and are **not** what this driver does.

So the native driver is 1.36x and 1.65x off the driver it replaces, not 3.4x
and 3.1x. Every performance conclusion drawn this session rested on the larger
figure, including the entire investigation into batching and A-buffer
episodes, and the 3,381-launches-per-frame trace was taken in the same
unwarmed state and needs redoing before it is quoted again.

**What to do about it, for whoever measures next.** Run the thing once before
timing it. The gate should be a warm-up replay whose result is discarded,
exactly as `cp_gpu_busy.sh` drops the ramp at each end of its window, and the
same applies to the sample sweep, whose per-sample means were also first taken
cold.

### The launch trace, re-taken warm

Same 20-second CUDA trace, both drivers, with the JIT cache full. Frames
completed in the window: native 2,055, gallium 2,797.

    kernel                  native/fr  gallium/fr  ratio
    cp_abuf_scan_block          106.6        22.9    4.7
    main                         86.7        37.2    2.3
    cp_abuf_scan_add             63.6        13.4    4.7
    cp_vertex_fetch              51.2        22.1    2.3
    cp_clip_triangles            51.2        22.1    2.3
    cp_fs_interpolate            33.8         9.3    3.6
    cp_abuf_worklist             22.6         1.8   12.5
    cp_peel_advance              11.3         0.1  169.4
    cp_abuf_seg_count             9.2         3.0    3.1
    cp_abuf_sort_short           10.2         3.9    2.7
    cp_abuf_seg_scatter           0.0         2.4      -
    TOTAL                       756.9       244.7   3.09

**757 launches a frame against 245, not 3,381 against 245.** The cold trace
overstated the difference by four and a half times, and two of its headline
numbers were artefacts: `cp_abuf_seg_count` and `cp_abuf_sort_short` read
exactly zero cold and are 9.2 and 10.2 warm, so the claim that the native
driver "never runs the pass episode kernels" was false. It runs them, about
three times as often as the Gallium driver does, which is the opposite kind of
finding -- more, smaller episodes rather than none.

What survives correct measurement:

- 3.09x the launches per frame for 1.36x the wall time, so the driver is
  issuing more and smaller work rather than stalling.
- `cp_peel_advance` at 11.3 against 0.1 is still the sharpest single
  difference, and `cp_abuf_worklist` at 12.5x behind it. Both are A-buffer
  drain work, which is what an episode amortises.
- `cp_abuf_seg_scatter` is the one kernel genuinely absent from the native
  trace.

That is a real and much smaller target than the one that was being chased, and
it is now measured in a state that reproduces.

### What CPVK_BATCH_BLEND is actually worth, and where the launches really go

Warm traces of the same capture, launches per frame:

    kernel                no flags   blended   gallium
    cp_abuf_scan_block       106.6     102.2      22.9
    main                      86.7      82.9      37.2
    cp_abuf_worklist          22.6      21.7       1.8
    cp_peel_advance           11.3      10.9       0.1
    cp_abuf_seg_count          9.2       8.8       3.0
    cp_abuf_seg_scatter        0.0       0.0       2.4
    TOTAL                    756.9     724.5     244.7

The blended-episode flag is worth **4%** of the launches and nothing at all in
wall time, which agrees with the median that did not move. `cp_peel_advance`
barely shifts, so the blended draws in this capture are not consecutive and
mergeable in the way the flag needs. The three defects it fixed were real; the
merge it enables is byte-identical in `cpvk_batchblend`; and on this workload
it does not pay.

The dominant number is `cp_abuf_scan_block`: 106.6 a frame against 22.9, which
is the A-buffer prefix scan and runs once per episode. **The native driver
builds about 4.7x as many A-buffer episodes as the Gallium driver** -- not
fewer, and not none, but many more and each much smaller.

That points at where episodes end rather than where they begin.
`cp_pass_finish` is called from `cpvk_execute_begin_render`, so every
`vkCmdBeginRendering` cuts the current episode, and it is called again from
every batch flush. An episode that ends early cannot amortise anything, and
the scan, the sort and the drain are paid again for the next one.

That is the next thing to measure: how many episodes a frame each driver
opens, and what ends them.

### Episodes never hold more than one segment

`CPVK_DEBUG_EPISODE` counts what `cp_pass_finish` closes.
`gltfscenerendering`, one frame:

    9 x episode: nsegs=1 opaque=1
    1 x episode-cut: begin_render
    0 x episode-cut: append failed
    2 x no-episode: orderfree=0        (blending on; every other test passes)

So every episode in the frame contains exactly **one** segment, which is why
`cp_abuf_scan_block` runs 4.7 times more often here than on the Gallium
driver: the scan is per episode, and an episode of one segment amortises
nothing.

What does *not* explain it: `vkCmdBeginRendering` cuts an episode once in the
frame, not nine times; no append ever fails; and `cp_opaque_appendable`
refuses only twice, both times because the draw is blended so ordering is not
free. Every other draw is eligible.

So episodes are eligible and being created, and something closes each one
after a single segment anyway. That contradiction is where this stops, and it
is a narrow one: nine creations, nine single-segment closes, and none of the
three code paths that close an episode accounting for more than one of them.

Worth noting for whoever picks it up: `cp_opaque_append` is only reached from a
batch flush, and this driver's batching is off by default, so how nine
episodes are created at all is part of the same question.

### Why every episode holds one segment, and why the obvious fix is wrong

The nine cuts have one source:

    9 x episode-cut: flush_why=the next draw cannot join nsegs=1

`cpvk_execute_draw` calls `cp_batch_flush_why()` whenever a draw cannot join
the pending batch, and with this driver's batching off by default **no draw
ever joins**, so every draw takes that line and `cp_batch_flush_why` finishes
the episode behind it.

The renderer offers a deferring variant for exactly this case, and says so:
"Only the four per-segment state changes may call this -- a draw whose key
broke the batch, and the vertex-shader, fragment-shader and vertex-elements
binds." Switching to it works as advertised: `gltfscenerendering`'s nine
single-segment episodes become three, of five, three and one.

**And it breaks six samples.** 15/18 pixel-correct falls to 9/18:

    bloom               126.195
    gltfscenerendering   24.03
    multithreading        9.777
    particlesystem        3.655
    pushconstants         3.428
    pbribl                2.393

The renderer's rule holds for the Gallium adapter, which flushes on the *other*
state changes an episode reads episode-wide. This front end has none of those
flush points, so it relies on this one; keeping the episode open here keeps it
open across shader and descriptor binds it must not span. Reverted, and 15/18
restored.

Note that all thirteen unit tests passed in the broken state. They do not model
a frame with many pipelines and many descriptor sets across several render
passes, which is what the samples caught. That is worth more than the fix
would have been.

So the one-segment episode is understood and its cost is quantified, and the
next move is not this line: it is giving this front end the flush points the
Gallium adapter has, after which the deferring variant becomes correct.

### The episode mechanism, priced: it works, and it is not the replay gap

Three measurements close this line of work.

**The flush point is not wrong.** The Gallium adapter uses the full,
episode-finishing flush for "the next draw cannot be batched" exactly as this
driver does. Its flush table:

    full      framebuffer, viewport, scissor, readback/clear/flush,
              draw-cannot-batch, compute dispatch, blend, rasterizer,
              depth/stencil
    deferring vertex elements, fragment shader, vertex shader

This driver matches it. What differs is that Gallium's draws *join* batches, so
that line is rare; here no draw joins, so it fires every time.

**Merges do amortise episodes.** With `CPVK_NO_DESC_KEY=1`,
`gltfscenerendering` goes from nine episodes to **one**. The mechanism works
exactly as designed once draws merge.

**And it buys nothing on the replay.** Warm, same session:

    baseline                          9.80 ms
    CPVK_BATCH=1 CPVK_NO_DESC_KEY=1   9.75 ms

Half a percent. Together with the earlier measurements -- batching alone 0.7%,
blended episodes 4% of launches and nothing in time -- every mechanism in the
batching and episode machinery has now been priced on this capture and none of
them is the 1.36x.

That is a negative result and a large one: it removes batching, merging,
descriptor keys, blended episodes and episode accumulation from the search for
the remaining replay gap. The gap is real (9.73 against 7.15 warm) and it is
somewhere else entirely.

The `gltfscenerendering` correctness bug behind the descriptor key remains
open, and is now correctly filed as a correctness bug rather than a
performance blocker.

### The remaining gap is host-side, not GPU work

The warm traces answer the question the launch counts could not. GPU kernel
time per frame:

    kernel                 native   gallium    delta
    cp_abuf_scan_block      0.196     0.040    +0.155
    cp_vertex_fetch         0.433     0.283    +0.151
    cp_fs_interpolate       0.232     0.162    +0.071
    cp_abuf_scan_add        0.073     0.016    +0.057
    ...
    TOTAL GPU ms/frame      2.183     2.538    -0.355

**The native driver uses less GPU time per frame than the driver it replaces**
-- 2.18 ms against 2.54 -- and still takes 9.73 ms of wall clock against 7.15.
So 7.55 ms a frame of the native driver's time is not kernel execution, against
4.61 ms for Gallium: about 2.9 ms a frame of extra host-side cost.

Untraced GPU utilisation agrees, both processes confirmed running: native
57.0%, gallium 58.3%. Near-identical occupancy, and the driver that finishes
frames faster is the one issuing fewer, larger launches.

That is consistent with 3.09x the launches at roughly the same host cost each.
It also explains every negative result of the last several turns: batching,
merging, the descriptor key and blended episodes were all being asked to fix a
GPU-work problem the driver does not have.

The caveat CLAUDE.md states applies to the traces and is why the untraced
utilisation check matters: CUPTI charges every `cuLaunchKernel`, and this
driver issues thousands a frame, so a traced run inflates exactly the
host-side gap being measured -- and inflates it 3x more for the native driver.
The kernel *times* above are trustworthy; the traced wall clock is not, and is
not used.

So the target is the per-frame launch count -- 757 against 245 -- and
specifically the A-buffer machinery that accounts for most of it, not the
front end that decides which draws share one.

### The largest merge blocker removed, correctness held, time unchanged

`CUDAPIPE_DEBUG_BATCHDIFF` over 25 seconds of the Crossroads replay -- the
workload itself rather than a sample -- gives 38,155 refusals:

    vertex offset    13,281   35%
    fragment shader  12,068   32%
    vertex shader     7,934   21%
    scissor           3,339    9%
    instance count    1,441    4%
    descriptors           0    -

**Descriptors do not appear at all.** The condition that consumed several turns
of this session separates nothing in the capture, which is why dropping it
moved the replay by half a percent.

The largest blocker, the vertex offset, was marked in its own comment as
provisional -- "until the batched path stops taking it from the first draw" --
and that has already happened: a batch builds a slice table and sets
`slices[d].first_vertex = draws[d].index_bias` per draw. Only the single-draw
path uses `draws[0]`. Removing the condition keeps 18/18 running and 15/18
pixel-correct with batching on.

    off                          9.41   9.42
    batch, offset merged         9.38   9.44
    batch, offset still blocking 9.77   9.78

So the condition costs about 3.5% when batching is on, and removing it brings
batching back to parity with not batching at all. **Not a win** -- one more
mechanism priced at zero.

The change is kept because the condition is unnecessary by construction and was
documented as temporary, not because it made anything faster. And the run-to-run
spread here is 9.38 to 9.80 across the session, so nothing under about 5% in
this measurement means anything.

### Why the launch gap is not merging, and why the flush discipline will not transplant

Batching on, with the vertex-offset condition removed, changes the launch count
by nothing: 756.9 a frame against 752.9, where Gallium is 244.7. **The draws in
this capture genuinely cannot merge** -- 32% of refusals are a different
fragment shader and 21% a different vertex shader -- and the Gallium driver
cannot merge them either. So the 3x launch gap is not merging, and that is now
measured rather than argued.

What Gallium does instead is accumulate *episodes* across draws that do not
merge. Native cannot, because it finishes the episode on every draw, because
`cp_batch_flush_why` is its only flush point.

Two attempts to give it the others, both measured and both reverted:

    state-change flush on viewport/scissor/blend/rasterizer/depth   11/18
    + deferring flush on shader and vertex-element change           11/18
    baseline                                                        15/18

`pushconstants`, `pbribl`, `multithreading` and `gltfscenerendering` break
either way. The Gallium adapter reaches the renderer through pipe_context
setters that fire at bind time, in an order this front end does not reproduce
by comparing state at draw time; the same five flushes in a different place are
not the same discipline. Reverted, 15/18 restored.

That is three attempts at the episode path in three turns, each correct-looking
and each measured into the ground. What they establish between them:

- the launch gap is real and is host-side (native uses *less* GPU time)
- it is not merging, not the descriptor key, not blended episodes, not the
  vertex-offset condition
- it is episode accumulation, and episode accumulation needs the flush
  discipline rebuilt at bind time rather than patched at draw time

That last is a design change to this front end, not an adjustment, and it is
where the remaining 1.36x lives.

## Consolidated state, measured warm in one pass

    samples          18/18 run, 15/18 pixel-correct
                     texturemipmapgen 0.456, multisampling 0.553, pbribl 1.255
    unit tests       13/13, each byte-identical to lavapipe or to its own
                     unbatched output
    Crossroads       native  9.41 ms   gallium  7.23 ms   1.30x
    old capture      native 34.26 ms   gallium 25.08 ms   1.37x
    both captures    1,496 and 1,510 frames, every frame, exit 0
    Gallium driver   0 files changed from the branch point

The Gallium figures reproduce what was recorded long before this session --
7.17 and 25.20 -- so the reference is stable and the comparison is sound. The
native driver is within 1.3-1.4x of it on both captures, from a recorded
starting point of 3.4x that turned out to be a cold-JIT-cache artefact.

**The objective, honestly.** The driver that ships is
`src/gallium/drivers/cudapipe`, and `git diff e2e966953d7` against it is empty,
so every recorded result for it stands without re-measurement. The native
driver does not yet meet it: three samples render wrong and both replays are
about a third slower than the driver they must replace.

### What is left, in the order the evidence supports

1. **The remaining 1.3x is host-side and structural.** Native issues 757 kernel
   launches a frame against 245 while using *less* GPU time, 2.18 ms against
   2.54. Merging is not the cause -- those draws cannot merge in either driver.
   Episode accumulation is, and it needs this front end's flush discipline
   rebuilt at bind time. Two patched-at-draw-time attempts each cost four
   samples and were reverted.
2. **pbribl 1.255** -- spheres correct, reflections neutral. The prefiltered
   cube is provably warm in memory at every level, cube sampling is exact in
   `cpvk_cubelod`, and the descriptors are written correctly. Not yet explained.
3. **multisampling 0.553** -- edge coverage only, 25x concentrated on
   silhouettes, sample positions identical to `lp_sample_pos_4x`.
4. **texturemipmapgen 0.456** -- off the edges, native brighter, chain and
   implicit LOD both proved identical to the Gallium driver.
5. **gltfscenerendering merges wrongly without the descriptor key** -- a
   correctness bug, not a performance one; it separates nothing in either
   capture.

### Measurement rules this session paid for

- Run it once before timing it. The CUDA JIT cache made a 2.5x difference and
  invalidated a day of numbers.
- Check the exit status of the run a number came from. A crashing instrument
  produced a confident false finding.
- Check the process was still alive when sampling ended. A finished replay
  reads as 0.5% GPU busy.
- Count with `wc -l` before summarising; `head` twice produced false counts.
- Print more than one slot, index or field than the hypothesis needs. Slot 1
  alone looked exactly like the bug.
- Run-to-run spread on these replays is 5%. Nothing smaller is a result.

### pbribl: the descriptors resolve to the right textures

Correlating the texture-info handle printed at view creation with the handle
each descriptor binding carries:

    binding 2 -> 0x...116200   64x64  target=3 enc=10   the irradiance cube
    binding 3 -> 0x...115e00   512x512 target=1 enc=28  the BRDF lookup table
    binding 4 -> 0x...126800   512x512 target=3 enc=9   the prefiltered cube,
                                                        levels 0..9

Every one is right, with the right target, format and level range, and the
sampler indices are 1 through 4 with index 4 -- the prefiltered cube's --
carrying `lod=0.0..10.0`.

The FS sampler table is built 114 times in the run: once with three entries,
42 times with four and 71 times with five. Index 4 is only in range once the
table has five, and the spheres are drawn last, so they should see the full
table -- but that is the one thing in this chain not yet directly verified, and
a sampler index one past the end of the table read at shade time would produce
exactly a neutral result from a correct cube.

So the elimination list for pbribl now reads: cube content warm at every level,
cube sampling exact including explicit LOD and half-float, descriptors written
correctly, bindings resolving to the right textures, samplers carrying the
right LOD range, and the material colour genuinely black by design. What is
left is the sampler *table* at the moment the sphere draws shade.

### pbribl: every CPU-side explanation is now eliminated

The sampler table carries five entries for the last twelve draws of the frame,
which are the spheres, so index 4 -- the prefiltered cube's -- is in range when
they shade. That was the final link.

The complete list, each checked by measurement rather than reading:

    the material colour            black by design, read from the push block
    the prefiltered cube's content warm at every level, read back from device
                                   memory after the copies that build it
    the cube's placement           all 60 copies issued, six faces x ten levels
    the image layout               level-major with layers inside, matching
                                   what the copy adds and what the sampler
                                   indexes
    cube sampling                  cpvk_cubelod and cpvk_cubelodf16 are
                                   byte-identical to lavapipe at every face and
                                   level, in both 8-bit and half-float
    explicit LOD                   cpvk_lod byte-identical
    the descriptors                23 writes at the right flat indices
    the bindings                   2, 3 and 4 resolve to the irradiance cube,
                                   the BRDF table and the prefiltered cube
    the samplers                   index 4 with lod 0.0..10.0
    the sampler table              five entries at the sphere draws

Nothing on the host explains a neutral reflection from a warm cube. Whatever is
left is device-side: what the fragment shader computes from a correct sampler,
a correct texture and a correct level.

`CUDAPIPE_DEBUG_FS` prints fragment values and was repaired earlier in this
session, which makes it the instrument for that -- and this driver's history
says the next step is to use it rather than to reason further.

### pbribl at the fragment: the inputs are right and the output is genuinely neutral

`CUDAPIPE_DEBUG_FS` on the sphere draws:

    fs_in[0] <- vs slot 1 (loc 32)      inWorldPos
    fs_in[1] <- vs slot 2 (loc 33)      inNormal
    px(270,311) in=[6.490 -0.995]  out=[0.608 0.608 0.608 1.000]
    px(126,319) in=[9.291 -0.843]  out=[0.000 0.000 0.000 1.000]
    vtx0: slot0=[-7.949 -0.116 11.926 12.021]  slot1=[7.609 -0.131 0.000 0.000]
          slot2=[-0.991 -0.130 0.000 0.000]    slot3=[0.750 0.542 0.000 0.000]

Two fragment inputs, wired from vertex slots 1 and 2, which is right:
`pbribl.frag` declares `inWorldPos`, `inNormal` and `inUV`, and its albedo
comes from push constants rather than the uv, so the third is dead and
correctly dropped. The vertex outputs are sensible -- a clip position with
w = 12.02, a world position, a unit normal, a uv.

And the fragment output is exactly neutral, computed rather than clamped:
0.608 in all three channels at one pixel, 0.000 at another, with the
framebuffer under it warm where the plaza shows through.

So the fragment stage receives correct interpolated inputs, addresses correct
textures through correct samplers at correct levels, and computes a grey
value. Every stage on both sides of the shader is now verified, and the
remaining question is inside the arithmetic itself: which of `reflection`,
`brdf` or `F` evaluates to something that makes
`reflection * (F * brdf.x + brdf.y)` neutral when `reflection` is warm in
memory.

The instrument for that is a shader edit -- replace the term with a constant
and see which one moves the picture -- rather than another read of the driver.
That is where this stops, with every layer around the shader eliminated by
measurement.

### pbribl solved to its cause: cube sampling with a per-fragment varying direction

Three shader edits, each a one-line substitution compiled over the sample's own
SPIR-V and then restored, separate three terms that no amount of driver-side
reading had separated.

    reflection = vec3(1.0, 0.4, 0.1)          spheres go WARM  (27,18,9)
    lod        = 0.0                          spheres stay grey
    R          = normalize(vec3(1, 0.2, 0.3)) spheres go WARM  (10,8,6)

The first says the shader's arithmetic is right and `brdf` and `F` are fine --
force the reflection and the picture is warm, close to the reference's
(25,17,10). So `prefilteredReflection()` is what returns neutral.

The second rules out the mip level: forcing lod to 0, where the cube's content
is provably warm, still gives grey.

The third is the answer. **`textureLod` on that cube returns correct colour
when the direction is uniform across the quad and neutral when it varies per
fragment.** `R = reflect(-V, N)` varies per fragment; a constant does not.

That fits everything else that has been measured and could not be reconciled
before: the cube's memory is warm, `cpvk_cubelod` samples it exactly, the
descriptors and samplers are right -- and `cpvk_cubelod` picks its direction
from six horizontal bands, so a quad in it almost never straddles two faces,
while every quad on a sphere does.

The suspect in the sampler is face selection. `cp_cube_face()` picks a face per
fragment, and the cube arm of the derivative path shuffles across the quad with
`__shfl_xor_sync(0xFFFFFFFF, ...)`, which is undefined when lanes diverge --
and lanes on a sphere diverge exactly when neighbouring fragments choose
different faces.

The test that would pin it is `cpvk_cubelod` with a direction that varies
smoothly across the frame rather than in bands: a sphere's worth of directions
in a one-second test.

### A fourteenth test reproduces pbribl in one second

`cpvk_cubesph` samples the same nine-colour cube with a direction that varies
smoothly per fragment -- a hemisphere's worth of directions -- instead of six
flat bands. Against lavapipe:

    differing pixels   734 of 4,096      max 80
    distinct colours   lavapipe 70       native 2

The red channel of each texel encodes its face as `20 + 40 * face`. lavapipe
returns 100, 60 and 20 across the frame -- faces 2, 1 and 0. **The native
driver returns 20 everywhere: face 0, for every fragment.**

So cube face selection collapses to face zero when the direction varies per
fragment, and is correct when it does not. That is exactly the difference
between `cpvk_cubelod`, which picks one of six constant directions by branching
on the band, and `pbribl`, whose `R = reflect(-V, N)` is arithmetic on
interpolated inputs.

It also explains the shape of pbribl's picture precisely: every sphere fragment
reads face 0 of the prefiltered cube, which is one direction's worth of
environment, and the result is a flat neutral tint rather than a reflection.

The test runs in a second, needs no sample, and prints which face was chosen
rather than a mean error. Whatever the fix turns out to be in the sampler's
cube path, this is what will show it working.

### The direction is right; the face selection is not

`cpvk_cubedir` is `cpvk_cubesph` with one line changed -- it writes the
direction as colour instead of sampling with it. Native and lavapipe are
**byte-identical**: 1,459 distinct colours each, maximum difference 0.

So the shader computes a correct, smoothly varying, per-fragment direction, and
the sampler receives it. Sampling with that same direction returns face 0 for
every fragment, where lavapipe returns faces 0, 1 and 2.

That places the fault squarely inside the sampler's cube path -- between
receiving a correct direction and choosing a face -- and rules out the shader,
the interpolation, the backend's coordinate handling and every host-side
structure already eliminated.

Two tests now bracket it exactly:

    cpvk_cubelod    direction is one of six constants, chosen by branch
                    -> identical to lavapipe, faces 0..5 all correct
    cpvk_cubesph    direction computed per fragment
                    -> face 0 everywhere, 2 distinct colours against 70
    cpvk_cubedir    the same direction written out instead of sampled
                    -> identical to lavapipe

A note on method: the first run of `cpvk_cubedir` printed DIFFER and both
drivers had in fact trapped, because the vertex shader had been passed where
the fragment shader belonged and neither run wrote a file. `cmp` on two
absent files reports a difference. The exit statuses were in the output and
said 133; checking them is what turned a false result into the real one, and
that is now the third time this session that rule has paid.

### cp_cube_face is correct, so the face is lost after it

The function is the standard one: dominant axis by `fabsf`, face from the
sign, `uc`/`vc` with the usual conventions, divide by `ma`, map to [0,1]. There
is nothing in it that returns 0 for a direction whose dominant axis is -X or
+Y, and `cpvk_cubelod` proves it selects all six faces correctly when the
direction is quad-uniform.

So the face is computed and then lost. The path after it is short:

    layer = (int)face
    -> cp_sample_level_layer(tex, samp, level, u, v, layer, filter,
                             target == CP_TEX_3D, c2)
    -> layer clamped against tex->depth
    -> cp_fetch_texel(tex, level, x, y, layer)

`tex->depth` is `MAX2(extent.depth, array_layers)` = 6 for these views, and the
same views serve `cpvk_cubelod`, so the clamp is not it either.

What is genuinely different between the passing and failing tests is only the
fragment shader: constants selected by branches against arithmetic on
interpolated inputs. Everything else -- image, view, sampler, test harness --
is shared, because `cpvk_cubesph` is a copy of `cpvk_cubelod` with one shader
swapped.

That is as far as reading gets. The next step is a device-side `printf` in the
cube arm of the sampler, printing the direction it received and the face it
chose for a handful of threads. This session's record on reading versus
instrumenting is one-sided: `load_output`, `gl_PointCoord`, the storage-image
extent, the multisample clear and the ICD manifest were all found by printing
something, and every hypothesis reached by reading code has been wrong.

### The device print overturns the face-selection finding

A `printf` in the sampler's cube arm, printing the direction it received and
the face it chose:

    cubesph:  dir=[-0.678 -0.497 0.541] face=1 uv=[0.899 0.867] depth=6
              dir=[-0.668 -0.486 0.564] face=1 uv=[0.922 0.864] depth=6
    cubelod:  dir=[-1.000  0.000 0.000] face=1 uv=[0.500 0.500] depth=6

**Face selection is correct in the failing test.** A direction dominated by -X
gives face 1, the uv is sensible, and `tex->depth` is 6, so the layer will not
be clamped. The previous entry inferred "face 0 for every fragment" from the
red channel of the output image, and that inference was wrong: the sampler
picks the right face and something after it returns face 0's texel.

What is left between the face and the pixel is `cp_fetch_texel`, which offsets
by `layer * img_stride[level]` -- and, on the write side,
`vkGetImageSubresourceLayout`, which the test uses to place each face's colour
and which must account for `arrayLayer`. If it ignores the layer, every face
was written to the same offset and the cube holds one face's data, which would
look exactly like this from the sampler's side while the sampler is innocent.

That is checkable directly and cheaply: print the offsets the driver returns
for the six faces of level 0.

Two inferences from images have now been overturned by printing the value
itself -- this one and the "prefiltered cube never copied" reading earlier.
Both times the image was consistent with the wrong explanation.

### Per-pixel, native reads one face where lavapipe reads three

Decoding the face from each sampled pixel's red channel over the 1,459 covered
pixels:

    native    [1459, 0, 0, 0, 0, 0]
    lavapipe  [ 725, 35, 490, 0, 0, 0]

and the direction images the two drivers produce are byte-identical. So with
the same per-fragment directions, the native driver reads face 0 for every
pixel and lavapipe reads three different faces. That is a real, per-pixel
difference and it does not depend on which face is "right".

**What this does not settle.** Computing the expected face from the decoded
direction with the standard dominant-axis rule gives
`[8, 139, 51, 39, 1222, 0]` -- mostly +Z -- which matches *neither* driver. So
that calculation is not a valid oracle here, whether because of the quantised
direction image, the face convention, or the uv mapping, and it is not used to
claim which driver is correct.

The device print from the previous entry showed `face=1` chosen for threads 0
to 7, which is consistent with this only if those threads are not among the
covered pixels -- helper lanes, or another draw. Thread index is not pixel
position, and treating it as such is what made the two readings look
contradictory.

So the established facts are narrow and solid: identical directions in, one
face out on this driver and three on lavapipe, with the sampler demonstrably
computing a non-zero face for at least some threads. The next instrument has
to tie a thread to its pixel -- printing the fragment's coordinates beside the
face -- because every reading that has not done so has been ambiguous.

### The cube test's oracle does not hold up, and that invalidates the reading built on it

Computing the face in the shader by Vulkan's dominant-axis rule and encoding it
the same way the cube's texels do, then comparing against what sampling
returns:

    expected (shader)   [0, 145, 49, 38, 1227, 0]     mostly +Z
    native sampled      [1459, 0, 0, 0, 0, 0]
    lavapipe sampled    [725, 35, 490, 0, 0, 0]

    agreement, native    0.0%
    agreement, lavapipe  4.4%

**Neither driver's sampling matches the face the direction implies.** lavapipe
is a reference implementation, so a 4.4% agreement means the oracle is wrong,
not that lavapipe is: either the test's colour-to-face encoding, the uv
mapping, or the dominant-axis rule as applied here does not correspond to what
the cube's layers actually contain.

The expected-face render also differs between the two drivers on 8 of 4,096
pixels, which is fine on its own -- floating-point ties at face boundaries --
but it means even that image is not exactly reproducible.

So the reading from the previous entry, "native reads one face where lavapipe
reads three", survives only as a **difference between the drivers**, which is
real and per-pixel. It does not survive as "native picks the wrong face",
because there is no working oracle here for which face is right.

What is still solid, and worth keeping separate from all of this:

- `pbribl`'s reflections come from `prefilteredReflection()` and nothing else,
  proved by forcing that term to a constant and watching the spheres go warm.
- Forcing the direction to a constant also makes them warm, so the failure
  depends on the direction varying per fragment.
- `cpvk_cubesph` reproduces a per-pixel difference from lavapipe in a second.

Those three stand on substitutions and direct comparison, not on decoding a
face from a colour. The face-decoding work does not, and is retracted as a
diagnosis.

### Bisecting the cube failure by substitution

No oracle, just native against lavapipe on the same shader:

    direction varying per fragment, always +X dominant      IDENTICAL
    three faces by band, direction varying inside each      IDENTICAL
    hemisphere of directions, quads straddling everywhere   DIFFER
                                                            (70 colours vs 2)

So per-fragment variation is fine, and several faces in one draw are fine. What
fails is quads straddling a face boundary *pervasively*, which is what a sphere
does and what a band does only on its two edges.

The colour counts point at the mechanism. With flat per-face colours, a driver
that filters **seamlessly across a cube seam** produces blends of two faces --
lavapipe's seventy distinct colours -- and one that clamps within the face
produces flat colours only, which is what this driver gives. Vulkan requires
seamless cube filtering, and `cp_sample_level_layer` clamps `u` and `v` inside
the level with `cp_wrap_texel`, which knows nothing about adjacent faces.

That also explains why every earlier cube test passed: `cpvk_cubelod` and the
band test choose directions that land near face *centres*, where a bilinear
footprint never reaches the edge, so seamless and clamped filtering agree
exactly.

It does not yet explain pbribl's spheres being uniformly neutral rather than
merely wrong at seams, so this is a mechanism with evidence behind it rather
than a finished diagnosis -- and the way to test it is a substitution again:
sample with `VK_FILTER_NEAREST`, where seam blending cannot occur, and see
whether the two drivers agree on the sphere.

### The cube bug is in filtering, not face selection

The same sphere of directions, the same shader, only the sampler's filter
changed:

    VK_FILTER_LINEAR    DIFFER     lavapipe 70 colours, native 2
    VK_FILTER_NEAREST   IDENTICAL  6 colours each, 0 differing pixels

With nearest filtering the two drivers agree exactly and **native selects all
six faces**, which retires the face-selection theory for good: the direction is
right, the face is right, and the fetch is right.

What differs is what happens when a bilinear footprint reaches a face edge.
Vulkan requires cube filtering to be seamless -- the footprint continues onto
the adjacent face -- and this sampler clamps or wraps inside the level it is
on, because `cp_wrap_texel` works on one 2D level and has no notion of an
adjacent face.

That is why every earlier cube test passed: they sample near face centres,
where the footprint never reaches an edge. It is also exactly `pbribl`'s
condition -- a prefiltered environment cube, sampled with `VK_FILTER_LINEAR`,
over a sphere whose every quad sits near or across a seam.

Two things are still unexplained and should not be glossed: native gives two
distinct colours under LINEAR rather than the six flat ones that clamping alone
would produce, and pbribl's spheres are uniformly neutral rather than wrong
only near seams. So seamless filtering is established as *a* difference with a
decisive test behind it, and may not be the whole of it.

The bisect that got here used only substitutions and driver-to-driver
comparison -- one shader line or one sampler field at a time, never an oracle
for what the answer should be. Every conclusion in this session that rested on
decoding a value from a colour has been retracted; none of the substitution
results has.

### Confirmed: the cube path lacks seamless filtering, and two earlier readings were stale

A fresh run of the sphere test, rebuilt binary and regenerated images:

    native LINEAR    6 distinct colours
                     face 4: 1227 px, face 1: 137, face 2: 49, face 3: 38,
                     face 0: 8
    lavapipe LINEAR  112 distinct colours
    expected faces   [0, 145, 49, 38, 1227, 0]

**Native selects the correct faces** -- its histogram matches the expected one
within the handful of pixels that sit exactly on a boundary. What it does not
do is blend across a seam: six flat colours where lavapipe produces 112 by
filtering a footprint that continues onto the adjacent face.

That is seamless cube filtering, which Vulkan requires, and it is confirmed by
the filter substitution: `VK_FILTER_NEAREST` makes the two drivers byte-
identical, because with no footprint there is nothing to blend.

**Two earlier readings in this file were wrong and both had the same cause.**
"native reads face 0 everywhere" and "0.0% agreement with the expected faces"
came from `.ppm` files generated several rebuilds earlier -- including one
build carrying a device-side `printf` -- and compared against freshly computed
expectations. The retraction of the face-selection diagnosis was itself
mistaken; the original oracle was right and the data under it was old.

The rule that catches this is the one already recorded for measurements and not
yet applied to files: **regenerate every artefact after a rebuild**, and do not
compare an image produced by one binary against a number produced by another.

So `pbribl` is: a prefiltered environment cube sampled with `VK_FILTER_LINEAR`
over a sphere, where every quad's footprint crosses a seam, on a driver that
clamps each footprint inside one face.

### Seamless cube filtering implemented

`cp_fetch_cube_texel()` addresses a cube texel across face boundaries. Rather
than an adjacency table -- twenty-four cases and their rotations -- an
out-of-range texel is turned back into the continuous face coordinate it
denotes, projected into a direction with the inverse of `cp_cube_face()`'s
conventions, and handed to `cp_cube_face()` to name the face and position it
really belongs to. In range it is the plain fetch, so the common path is
unchanged.

Measured on `cpvk_cubesph`, the test written for exactly this:

    native distinct colours   6 -> 111        (lavapipe 112)
    differing pixels        734 -> 18         (max 80 -> 13)

The residual eighteen pixels are at the corners, where three faces meet and the
footprint has no fourth texel; that is a documented special case and not
addressed here.

Nothing regressed. 18/18 samples run, 15/18 pixel-correct, thirteen unit tests
pass, and both replays are unmoved at 9.40 ms and 34.28 ms.

**And it is not pbribl's bug.** That sample moves 1.255 to 1.252. The defect was
real, is fixed, and was found by the bisect that pbribl motivated -- but
pbribl's spheres are neutral for another reason, exactly as the earlier note
that they are "uniformly neutral rather than wrong only near seams" warned.

Which is worth stating plainly: the substitution bisect found a genuine bug in
the driver and did not find the one it was aimed at. The three tests that
disagree with lavapipe are unchanged, and pbribl needs its own next
substitution.

### pbribl: the prefiltered cube reads as exactly zero

Substituting the shader's own output for the term under suspicion, native
against the Gallium driver on the same frame:

    outColor = prefilteredReflection(R, roughness)
        native  (0,0,0)         gallium (221,92,28) (96,49,24) (84,96,112)

    outColor = textureLod(prefilteredMap, R, 0.0).rgb
        native  (0,0,0)         gallium (225,93,29) (92,43,16) (255,255,255)

So a single explicit-LOD read of the prefiltered cube returns **exactly zero**
on this driver, per fragment, everywhere on the spheres -- not a neutral grey,
not a wrong face, zero. The final image is neutral only because zero ambient
leaves direct lighting, which is white light on a black albedo.

That is a different failure from the one `cpvk_cubesph` reproduces, where the
faces are right and only the seam blending was missing. The distinguishing
properties of pbribl's cube against every cube in the test suite are its
format and size: R16G16B16A16_SFLOAT at 512x512 with ten levels, against
R8G8B8A8 and R16G16B16A16 at 8x8 with four. `cpvk_cubelodf16` covers the format
but not the size, and covers band directions but not varying ones.

So the next test is the intersection nothing has covered: a half-float cube
large enough to have ten levels, sampled with a per-fragment varying direction.
If that reads zero, pbribl is reproduced in a second and the seam work, which
was real and is kept, will have been a detour taken because the bisect that
found it was aimed at the wrong sample.

### pbribl's cube: format, size and varying direction all eliminated

`cpvk_cubebig` is pbribl's cube in a test -- R16G16B16A16_SFLOAT, 512x512, ten
mip levels, six faces, sampled with the same per-fragment varying direction
that makes `cpvk_cubesph` exercise seams. Against lavapipe: **IDENTICAL**.

So the combination that looked like the last uncovered one is not the cause
either. What remains unique to pbribl's cube is how it is *filled*: every test
cube is written by the host through `vkGetImageSubresourceLayout` into linear
memory, while pbribl's is rendered to an offscreen and placed with sixty
`vkCmdCopyImage` calls.

The debug output confirms those copies happen and target the right subresources
-- `dst(l=0 lay=0)`, `lay=1`, `lay=2`, all `ok=1`, into a `512x512 layers=6
mips=10` image -- and the view that samples it reports
`base=0x7cb146000000`, `target=3`, `enc=9`, `levels=0..9`.

What has never been checked in one run is whether those two addresses are the
same memory: the copy computes its destination from `img->mem` at record time
and the view captured `base` from `img->mem` at view-creation time. If the view
was created before `vkBindImageMemory`, the two differ and the sample reads
memory nothing ever wrote -- which returns exactly zero, which is exactly what
the substitution measured.

That is the next check and it is one print: the copy's destination pointer
beside the view's base.

### Everything into the read is correct, and the read returns zero

    the cube's memory      copies land at dstmem=0x7db3c2000000
    the view's base        base=0x7db3c2000000, target=3, enc=9, levels 0..9
    the direction R        identical to the Gallium driver on every sphere pixel
    textureLod(cube,R,0)   (0,0,0) here, (225,93,29) there

And `cpvk_cubebig` -- the same format, the same 512x512, the same ten levels,
the same six faces, the same per-fragment varying direction -- is
byte-identical to lavapipe.

So the inputs to the read are all correct, an equivalent cube reads correctly,
and this one returns zero. The difference has to be in state neither the cube
nor the direction carries.

The candidate that remains is the **sampler**. pbribl's prefiltered cube uses
sampler index 4 with `wrap=2,2` -- clamp to edge -- `filt=1/1`, `mip=1` and
`lod=0.0..10.0`, while every test here uses `VK_SAMPLER_ADDRESS_MODE_REPEAT`
and a `maxLod` matching the level count exactly. A `maxLod` of 10 on a cube
whose levels are 0..9 is legal and should clamp, and the wrap mode should not
matter at all now that the cube path is seamless and no longer consults it --
which makes both worth checking rather than assuming.

That is the next substitution, and it is one line in the test: build the big
cube's sampler with clamp-to-edge and `maxLod = 10`, and see whether
`cpvk_cubebig` starts returning zero.

### The sampler is eliminated too

`cpvk_cubebig` rebuilt with pbribl's sampler state -- `CLAMP_TO_EDGE` on all
three axes and `maxLod = 10.0` on a cube whose levels are 0..9 -- is still
byte-identical to lavapipe, eleven distinct colours, nothing zeroed.

The elimination list for pbribl's cube read is now:

    the cube's content        warm at every level, read back from device memory
    the cube's memory         copies land exactly at the view's base
    the level offsets         copy and sampler compute the same ones
    the direction R           identical to the Gallium driver per pixel
    the format                R16G16B16A16_SFLOAT, covered
    the size and levels       512x512, ten levels, covered
    the varying direction     covered, including seams
    the sampler               clamp-to-edge and maxLod 10, covered
    the descriptor            binding 4 resolves to the right texture handle
    the sampler table         five entries when the spheres draw

Every one checked against a test that passes or a print that agrees, and the
read still returns exactly zero in the sample.

What has *not* been reproduced is the shape of pbribl's descriptor set: five
bindings in one set -- two uniform buffers then three combined image samplers,
the cube at binding 4 -- against this test's two sets of one binding each. The
driver writes that descriptor at flat index 4 and the debug confirms the handle
there is right, but what the *shader* computes as the offset of binding 4 has
never been compared against it.

That is the next substitution: put the cube at binding 4 of a five-binding set
in the test, behind two uniform buffers, and see whether the read goes to zero.

### In pbribl's fragment shader, 2D reads work and cube reads return zero

Three substitutions in the same shader, same frame, same fragments:

    texture(samplerBRDFLUT, vec2(0.5))   binding 3, 2D    (185, 5, 0)
    texture(samplerIrradiance, R)        binding 2, cube  (0, 0, 0)
    texture(prefilteredMap, R)           binding 4, cube  (0, 0, 0)

So it is not the binding index, not the descriptor, not the sampler, and not
the cube's contents -- a 2D texture in the same set, sampled by the same shader
on the same fragments, returns a sensible value while both cubes return zero.

And the skybox shader, in the same application, samples a cube correctly and is
byte-identical to the reference.

What separates them: skybox samples with `inUVW`, an interpolated varying
straight from the rasterizer, while pbribl computes `R = reflect(-V, N)` from
two varyings and a uniform. The backend distinguishes exactly this case --
`emit_tex` sets `coord_slot` only when the coordinate's parent is
`load_input`, and its comment says "anything computed in the shader leaves the
sampler on the base level".

That is not yet the explanation, because `cpvk_cubesph` also computes its
direction and works. But the pair of facts is now sharp: a cube sampled with a
computed direction works in a test and returns zero in this shader, while a 2D
texture in that same shader works.

The next substitution follows directly: in pbribl's shader, sample the cube
with a *varying* -- `texture(prefilteredMap, inNormal)` -- and with a computed
one, in the same run. If the varying works and the computed does not, the
`coord_slot` path is implicated with evidence.

### Only face 0 of pbribl's prefiltered cube reads back

Sampling that cube at one constant direction per face, on a sphere fragment:

    +X  (30, 16, 9)
    -X  (0, 0, 0)
    +Y  (0, 0, 0)
    -Y  (0, 0, 0)
    +Z  (0, 0, 0)
    -Z  (0, 0, 0)

Layer 0 holds data and layers 1 to 5 read as zero. That explains every
observation at once: a reflection direction lands on face 0 for a minority of
fragments, so 36.8% of the sphere band reads non-zero and the rest is black;
the whole term averages to nothing; and the spheres come out lit by direct
light alone.

It also explains why the earlier "force R to a constant" test made the spheres
warm -- the constant chosen, `normalize(vec3(1, 0.2, 0.3))`, is +X dominant.

Every prior elimination stands and none of them touched this: the copies were
verified to be *issued* for all six faces, and the cube's content was verified
by reading back what a copy wrote -- but always at the first face. The one
thing never checked is whether layers 1 to 5 of *this* image contain what was
copied into them.

`cpvk_cubebig` has the same format, size and level count and reads all six
faces correctly, and its faces are written by the host through
`vkGetImageSubresourceLayout` into linear memory. pbribl's are placed by
`vkCmdCopyImage` into an optimal-tiled image. That is the remaining difference,
and it is now a narrow one: either the copies do not land where they are
computed to, or the allocation does not extend to the later layers.

The check is direct -- read back device memory at `base + layer * level_size`
for each of the six faces after the copies -- and it needs no shader.

### vkCmdCopyImage ignored baseArrayLayer, and the fix for it was never in the tree

Reading back the six faces of each cube at the address the sampler uses --
`base + layer * level_size[0]`, at descriptor-write time so the copies have
happened -- showed the generated cubes with face 0 populated and faces 1 to 5
all zero, while the environment cube loaded from a file had all six.

`vkCmdCopyImage` was not adding `baseArrayLayer` to either subresource, so the
six copies that build a cube all landed on face zero.

**And this was supposedly fixed many turns ago.** The commit
"honour array layers in rendering and image copies" says so, its message
explains the reasoning, and it recorded honestly that no sample changed. The
reason no sample changed is that the change was never applied: `git log -S
d_layer` on the file returns nothing. The edit was a `str.replace` whose
pattern did not match, with no assertion behind it, and the "measured, changed
nothing" note made the absence look like a result.

Applied properly, with the assertion:

    face 0 @0x...ee000000: 294f 284c 277f 3c00
    face 1 @0x...ee200000: 27db 2811 273b 3c00
    face 2 @0x...ee400000: 280e 2821 2762 3c00
    face 3 @0x...ee600000: 305a 3343 362e 3c00
    face 4 @0x...ee800000: 29ae 2816 265a 3c00
    face 5 @0x...eea00000: 281a 2825 2707 3c00

    pbribl  1.252 -> 0.030

Forty-fold. The sweep holds at 18/18 running and 15/18 pixel-correct, and
fourteen unit tests pass. pbribl is not yet under the 0.01 threshold, but its
reflections are there.

The lesson is not "assert your edits" -- that was already written down twice
this session after `head` truncations and a crashing instrument. It is that a
null result from an unverified change is indistinguishable from a null change,
and this one sat in the record for twenty turns looking like evidence.
