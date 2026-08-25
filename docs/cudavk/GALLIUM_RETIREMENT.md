# Retiring the Gallium-hosted cudapipe driver

`src/gallium/drivers/cudapipe` and `src/gallium/targets/cudapipe` were removed
after the CUDA Vulkan driver in `src/cudapipe` replaced them. This file records
what was lost, measured rather than assumed, so the decision can be revisited
with numbers instead of memory.

## What it was

A Gallium pipe driver with **lavapipe's Vulkan frontend** linked in front of it
(`link_whole : [liblavapipe_st]` in its target). That is the whole story of its
capability profile: the API surface was lavapipe's, and the CUDA backend sat
underneath it. The native driver instead uses Mesa's common Vulkan runtime and
implements the API itself.

## What was lost: API surface

Measured with `vulkaninfo` at commit `ab7b431611a`, both ICDs on the same
machine:

| | Gallium-hosted | native |
|---|---|---|
| `apiVersion` | **1.4.354** | 1.1.354 |
| extensions enumerated | **198** | 37 |
| `driverID` | `MESA_LLVMPIPE` | — |
| `deviceType` | `OTHER` | `DISCRETE_GPU` |

161 extensions were advertised by the Gallium path and are not advertised by
the native one. The ones a real application is most likely to demand:

    VK_KHR_buffer_device_address        VK_KHR_multiview
    VK_EXT_descriptor_indexing          VK_KHR_sampler_ycbcr_conversion
    VK_KHR_dynamic_rendering_local_read VK_KHR_separate_depth_stencil_layouts
    VK_KHR_8bit_storage                 VK_KHR_16bit_storage
    VK_KHR_maintenance2 .. maintenance11

Those overlap almost exactly with the native driver's known gaps: multiple
render targets, layered rendering, input attachments, timeline semaphores and
multiple queues.

**The important caveat: advertised is not verified.** lavapipe advertises this
surface; whether each extension behaves correctly with the CUDA backend
underneath it was never tested. The delta is an upper bound on what was lost,
not a list of working features.

## What was not lost: the workloads we actually run

Both GFXR captures and all 18 samples run on the native driver. The Gallium
path was slower on everything measured:

| | Gallium-hosted | native (default) |
|---|---:|---:|
| old capture, paired-submit median | 25.1548 ms | **15.73 - 15.82 ms** |
| 600-frame sample sweep, sum | 35.77 ms | **16.02 ms** |

It did still work when it was removed: the old capture replayed to completion,
rc 0, all 3,022 submits.

## Why it was removed anyway

1. Nothing we run needs the extra surface, and the surface is untested.
2. It cost 22,632 lines of driver plus a target, and every change to shared
   kernels or docs had to consider two drivers.
3. It was the only reason `src/cudapipe` could not be built on its own; that
   coupling is now cut (`-Dcudavk=true`).
4. Keeping a slower, less-tested second path "just in case" is how a fork
   accumulates two half-maintained drivers instead of one good one.

## How to get it back

The commit immediately before the removal is tagged:

    git show gallium-cudapipe-last
    git checkout gallium-cudapipe-last -- src/gallium/drivers/cudapipe \
                                          src/gallium/targets/cudapipe

The build wiring it needs is in that commit too: the `cudapipe` entry in
`gallium-drivers` (`meson.options`), `with_gallium_cudapipe` (`meson.build`)
and the `subdir` calls in `src/gallium/meson.build`. Build it with
`-Dgallium-drivers=llvmpipe,cudapipe -Dvulkan-drivers=swrast`; its ICD lands at
`build/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json`.

If the reason for bringing it back is an application that needs an extension
the native driver lacks, measure that the extension actually works on the CUDA
backend before relying on it.

## Where its documentation went

The mechanism documents were kept, not deleted: `docs/cudavk/history/`. They
describe A-buffer construction, batching, pass episodes, instancing and the
adaptive rasteriser, all of which still exist in the native driver in evolved
form. The forward-looking plans among them were overtaken and say so.
