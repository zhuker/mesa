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
   *Gate: 18-sample sweep byte-identical, capture at exactly 0.441% / 0.002%.*
2. **Milestone 1 — enumerate.** ✅ done. `src/cudapipe` builds a second ICD;
   `tests/cpvk_smoke.c` reports the RTX 5090 as a Vulkan physical device with
   the three memory types session 13 had to negotiate with lavapipe.
3. **Milestone 2 — device, queue, memory, buffers, images.**
4. **Milestone 3 — pipelines and descriptors**, feeding the existing NIR→PTX
   compiler.
5. **Milestone 4 — command buffers**: record, then translate a whole render
   pass at submit. This is where batching and episodes stop existing.
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

- `vulkaninfo --summary` still segfaults on an entrypoint the driver returns
  NULL for; the smoke test is the milestone, not vulkaninfo. Finding that
  entrypoint is the first task of milestone 2.
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

Iteration record: `~/git/Vulkan/build/iter/census-vk-native`.

When the native driver renders, it takes the same table, and additionally has
to match the Gallium-hosted build frame for frame — which is why both targets
are built from one tree.
