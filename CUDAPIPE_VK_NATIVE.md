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
