# cudapipe Handoff — Continuing the CUDA Rasterizer

## What This Is

A Mesa Vulkan ICD (`libvulkan_cudapipe.so`) that rasterizes triangles on NVIDIA
GPUs using CUDA compute kernels instead of fixed-function hardware. It reuses
lavapipe as the Vulkan frontend and replaces the Gallium driver underneath.

## Build

Needs meson >= 1.4, LLVM 18 with the NVPTX backend, and a CUDA toolkit whose
NVRTC knows the target GPU (`/usr/local/cuda`, *not* the older nvrtc that may
sit in `/usr/lib/x86_64-linux-gnu`).

```bash
meson setup build-cudapipe -Dvulkan-drivers=swrast \
  -Dgallium-drivers=llvmpipe,cudapipe -Dllvm=enabled -Dglx=disabled \
  -Degl=disabled -Dplatforms= -Dgbm=disabled -Dgles1=disabled -Dgles2=disabled \
  -Dopengl=false -Dglvnd=disabled
ninja -C build-cudapipe
```

The build emits `cudapipe_devenv_icd.<arch>.json` pointing at the build tree, so
the driver runs without installing:

```bash
VK_DRIVER_FILES=$PWD/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json \
  <vulkan app>
```

## Architecture

```
Vulkan app (SPIR-V)
    ↓
lavapipe frontend (reused as-is)
    ↓ pipe_context calls
cudapipe Gallium driver
    ├── create_{vs,fs,compute}_state → cp_compile_nir_to_ptx() → PTX → CUmodule
    ├── launch_grid → cuLaunchKernel (compute)
    └── draw_vbo:
        1. Assemble vertices from vertex buffers (host side, multi-VB/indexed)
        2. Vertex shader kernel      → clip positions + varyings
        3. cp_rasterize_triangles    → visibility buffer (atomicMin depth|triID)
        4. cp_fs_interpolate         → compact covered pixels, interpolate
                                       varyings and their derivatives
        5. <compiled fragment shader> → one thread per covered pixel
        6. cp_fs_writeback           → blend into the colour attachment
```

Shaders run as ordinary CUDA kernels named `main`, taking one argument: a
pointer to an array of pointers. The slots are shared across stages:

| Slot | Meaning |
|---|---|
| 0 | thread/vertex/pixel count |
| 2 | input buffer (vertex attributes, or interpolated varyings) |
| 3 | input stride |
| 4 | output buffer (varyings, or fragment colour) |
| 5 | vertex-id array (vertex stage only) |
| 6 | fragment coordinates |
| 18.. | uniform/descriptor buffers |

Because `load_input`/`store_output` index by `blockIdx.x * 256 + threadIdx.x`,
the same emitter code serves the vertex stage (indexed by vertex) and the
fragment stage (indexed by covered pixel).

## Texture sampling

The sampler lives in `kernels/cp_sampler.cu` as a `__device__` function. NVRTC
compiles it to relocatable PTX once at screen init; `cuLink*` links it into each
shader's PTX that contains a `nir_tex`. `emit_tex()` therefore only has to emit
a call to `cp_tex_sample_2d()`.

**Where texture state comes from.** lavapipe drives everything through its
descriptor buffers — it never calls `set_sampler_views`/`bind_sampler_states`
with real state (verified: it only ever unbinds). But it *does* call this
driver's `create_texture_handle()` once per image view and once per sampler, and
copies two fields out of whatever we return:

* `->functions`, which we point at our own `struct cp_texture_info`
  (dimensions, format, strides, mip offsets)
* `->sampler_index`, which we make an index into our own `cp_sampler_info` table

So the sampler never parses llvmpipe's internal descriptor layout. It reads only
two offsets from lavapipe's descriptors, and `cp_context.c` `static_assert`s both
against `offsetof()` so an upstream change breaks the build rather than the
rendering.

Mip level selection uses derivatives computed analytically in
`cp_fs_interpolate` (the barycentrics are re-evaluated one pixel right and one
pixel down). `emit_tex()` traces the coordinate back to its `load_input` slot at
compile time and passes that slot to the sampler; a coordinate computed inside
the shader gets no derivatives and samples the base level.

## Status

Verified with dEQP (`vulkan_headless` target):

| Area | Result |
|---|---|
| `texture.filtering.2d.formats.*` | 72/75 supported pass |
| `compute.pipeline.basic.*` | 70/71 supported pass |
| `draw...simple_draw.*` | 2/4 (both non-instanced pass) |

Known gaps, roughly in the order they matter for a real workload:

1. **Instanced draws are not implemented.** `cp_draw_vbo` ignores
   `info->instance_count` entirely — it never loops over instances. This is why
   the two `simple_draw_instanced_*` tests fail.
2. **No depth buffer.** Depth testing is implicit in the visibility buffer
   (closest triangle wins), which is enough for a single pass but not for
   multi-pass rendering, shadow maps, or explicit depth reads.
3. **Vertex assembly is done on the host**, per draw, with a `memcpy` per
   attribute per vertex. This dominates the cost of large draws — the full
   `draw.dynamic_rendering` group does not finish in 25 minutes.
4. **Compressed formats are written but unverified.** DXT1/3/5 decode exists in
   `cp_fetch_texel` but no test in the suites run so far exercises it. BC4-7 are
   missing. These matter for real game content.
5. **Depth/stencil aspect sampling** returns floats only, so the three
   `*_stencil`/`s8_uint` filtering tests fail (they need integer texture returns).
6. `copy_ssbo_bounds` fails — SSBO robustness/bounds behaviour.
7. Only 2D textures are sampled. Cube maps, arrays, 3D textures, texel fetches
   (`nir_texop_txf`) and shadow compares fall through to a zero result in
   `emit_tex()`.

## Debug environment variables

| Variable | Effect |
|---|---|
| `CUDAPIPE_DEBUG_DRAW` | draw call summary, vertex elements, shaded pixel count |
| `CUDAPIPE_DEBUG_TEX` | sampler/texture descriptor resolution |
| `CUDAPIPE_DEBUG_FS` | per-pixel fragment inputs/outputs and varying mapping |
| `CUDAPIPE_DEBUG_FS_ROW` | restrict `CUDAPIPE_DEBUG_FS` to one framebuffer row |
| `CUDAPIPE_DEBUG_LAUNCH` | compute UBO/SSBO bindings |
| `CUDAPIPE_DUMP_NIR` / `DUMP_PTX` / `DUMP_IR` | dump shader IR at each stage |

## Notes for whoever picks this up

* `.cu` kernels are stringified into the binary at build time by
  `kernels/cu_to_inc.py`; edit the `.cu` and rebuild, there is nothing to
  regenerate by hand.
* LLVM's NVPTX backend only knows architectures that existed when it was
  released. On anything newer it warns and silently emits PTX the driver
  rejects, so `CP_MAX_PTX_SM` in `cp_nir_to_llvm.c` caps the architecture we ask
  for and lets the driver JIT forward. Raise it together with the PTX ISA
  version in the same function.
* `struct cp_resource` embeds `struct llvmpipe_resource` as its first member
  because lavapipe's descriptor code reads llvmpipe fields at fixed offsets.
  Don't reorder it.
* Every mip level needs its own `row_stride`/`img_stride`/`mip_offsets` entry.
  Leaving them zero doesn't just break the small levels — uploads of level 1 land
  at offset 0 and silently overwrite the first row of level 0.
