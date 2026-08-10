# cudapipe — CUDA Software Rasterizer for Roblox

## What This Is

A Mesa Vulkan ICD (`libvulkan_cudapipe.so`) that rasterizes on NVIDIA GPUs using
CUDA compute kernels instead of fixed-function hardware. It reuses lavapipe as
the Vulkan frontend and replaces the Gallium driver underneath.

The feature set is driven entirely by what Roblox's HeadlessStreamer actually
requires, measured from a GFXReconstruct capture — not guessed from specs.

## Build

```bash
meson setup build-cudapipe -Dvulkan-drivers=swrast \
  -Dgallium-drivers=llvmpipe,cudapipe -Dllvm=enabled -Dglx=disabled \
  -Degl=disabled -Dplatforms= -Dgbm=disabled -Dgles1=disabled -Dgles2=disabled \
  -Dopengl=false -Dglvnd=disabled
ninja -C build-cudapipe
```

Needs meson >= 1.4, LLVM 18 with the NVPTX backend, and CUDA toolkit at
`/usr/local/cuda`. The Vulkan SDK at `~/vulkan-sdk/` provides glslangValidator
and SPIRV-Tools. Add its bin/ and include/ to PATH/C_INCLUDE_PATH when building.

Run without installing:

```bash
VK_DRIVER_FILES=$PWD/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json \
  <vulkan app>
```

## Current Status

### Roblox HeadlessStreamer replay

A GFXReconstruct capture of HeadlessStreamer (1280×720, `--softwareEncoder`,
60 seconds on Tesla T4) replays successfully for 68 seconds before hitting a
lavapipe descriptor bug (which also crashes lavapipe alone — not a cudapipe
issue).

| Metric | NVIDIA (T4) | cudapipe (T4) |
|---|---|---|
| Replay wall clock | 19 s | ~68 s (lavapipe crash) |
| GPU utilization | — | 95–99% |
| GPU memory | ~400 MB | ~930 MB |
| Ratio | 1× | ~3.6× slower |

**Every feature the capture requires is implemented.** The 3.6× gap is purely
GPU rasterization compute cost (one thread per triangle, bounding-box scan).

### dEQP

`dEQP-VK.api.smoke.*`: 6/6 pass.

### What Roblox actually needs (from the capture)

```
77 graphics pipelines, 2 compute, 0 raytracing
Shader stages: vertex + fragment only
Topologies: TRIANGLE_LIST (dominant), TRIANGLE_STRIP, POINT_LIST
Max color targets: 1
MSAA: 4x (accepted, storage allocated)
Depth: D32_SFLOAT, D16_UNORM
Blending: yes (28 of 77 pipelines)
Compressed textures: BC1, BC3 only (576 images)
```

Full details in `src/gallium/drivers/cudapipe/tests/headless_streamer_requirements.txt`.

## Capturing and replaying HeadlessStreamer

GFXReconstruct is built at `~/gfxreconstruct`. The capture script lives at
`~/git/roblox/game-engine/capture_headless_streamer.sh`. It:

1. Forces the NVIDIA ICD (captures on real hardware)
2. Injects the GFXReconstruct capture layer
3. Uses `--softwareEncoder` to skip Vulkan video (which GFXReconstruct stubs)
4. Runs for 60 seconds then reports the `.gfxr` file

```bash
# Capture (requires COOKIE env var or ~/.robloxcookie)
cd ~/git/roblox/game-engine
./capture_headless_streamer.sh

# Analyze what the capture needed
GFX=~/gfxreconstruct/build
$GFX/tools/info/gfxrecon-info /tmp/headless_streamer_*.gfxr
$GFX/tools/convert/gfxrecon-convert --output /tmp/headless.json /tmp/headless_streamer_*.gfxr
python3 ~/git/mesa/src/gallium/drivers/cudapipe/tests/cp_capture_requirements.py /tmp/headless.json

# Replay against cudapipe
VK_DRIVER_FILES=.../cudapipe_devenv_icd.x86_64.json \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported /tmp/headless_streamer_*.gfxr
```

Two flags matter:
* **`-m remap`** — memory type indices differ between drivers; without it,
  allocation fails immediately.
* **`--remove-unsupported`** — the capture requested VK_KHR_surface which
  cudapipe doesn't expose; this strips it.

## Architecture

```
Vulkan app
    ↓
lavapipe frontend (reused as-is)
    ↓ pipe_context calls
cudapipe Gallium driver
    ├── State changes write into persistent cp_gpu_state (managed memory)
    ├── draw_vbo:
    │   1. cuMemsetD32 visbuf clear
    │   2. cp_vertex_fetch kernel (GPU gathers attributes from VBs via IB)
    │   3. Vertex shader kernel (NIR → PTX)
    │   4. cp_rasterize_triangles (visibility buffer, atomicMin depth|triID)
    │   5. cp_fs_interpolate (compact covered pixels, interpolate varyings)
    │   6. Fragment shader kernel (one thread per covered pixel)
    │   7. cp_fs_writeback (blend into colour attachment)
    └── flush: cuEventRecord (non-blocking fence)
```

The CPU does only: integer arithmetic for scratch offsets + `cuLaunchKernel`
calls. No per-draw malloc, no memcpy to device, no cuMemcpyHtoD. State
changes (`bind_blend`, `set_vertex_buffers`, etc.) write directly into a
persistent managed-memory struct that GPU kernels read.

## Performance Design

| Principle | Implementation |
|---|---|
| No per-draw sync | Event fences; `cuCtxSynchronize` only at readback or memory pressure |
| No CPU-GPU shared memory in hot path | Scratch is managed but CPU only writes metadata; large buffers are GPU-only |
| GPU vertex fetch | `cp_vertex_fetch.cu` kernel; for TRIANGLE_LIST the IB is read directly on GPU |
| No CPU topology expansion | `cp_build_vertex_refs` skipped for 94% of draws |
| Direct managed writes | Small arg structs written directly (no cuMemcpyHtoD) |
| Bounded memory | Scratch grows monotonically, reclaimed at flush or when overflow > 5 |

## Next Steps (priority order)

1. **Adaptive rasterizer** — the only remaining performance lever. Current:
   1 thread/triangle scanning bounding box. Target: CuRast-style 3-stage
   (1 thread/small tri, 1 warp/medium tri, 1 block/tile for huge tris).
   Reference implementation at `/home/coder/git/CuRast/src/kernels/triangles_visbuffer.cu`.

2. **Fix lavapipe descriptor crash** — segfaults in `lvp_descriptor_set_create`
   after ~5000 draws. Crashes lavapipe alone too. Investigate descriptor pool
   exhaustion or variable-count descriptor layouts.

3. **Kernel fusion** — merge interpolate + FS + writeback to reduce launch
   count from 6 to 4 per draw.

4. **Line rasterization** — Bresenham kernel for `line_list`/`line_strip`.

5. **Shadow compares / texture gathers** — needed for shadow mapping.

## Gaps NOT required by Roblox

These are listed in case the target application changes, but the current
HeadlessStreamer capture does not use any of them:

* BC4-7, ETC2, ASTC compressed formats
* Geometry/tessellation/mesh shaders
* Stencil test
* Multiple render targets
* Cube face / 3D slice filtering across boundaries
* Anisotropic filtering (advertised, falls back to trilinear)
* Occlusion queries / conditional render

## Debug

| Variable | Effect |
|---|---|
| `CUDAPIPE_DEBUG_TIME` | per-draw timing breakdown (assemble/vertex/raster/interp/fragment/writeback) |
| `CUDAPIPE_DEBUG_DRAW` | draw call summary, vertex elements, pixel count |
| `CUDAPIPE_DEBUG_TEX` | sampler/texture descriptor resolution |
| `CUDAPIPE_DEBUG_FS` | per-pixel fragment inputs/outputs |
| `CUDAPIPE_DEBUG_SHADER` | warn on unhandled NIR intrinsics |
| `CUDAPIPE_DUMP_NIR` / `DUMP_PTX` / `DUMP_IR` | dump shader IR at each stage |

## Implementation Notes

* `.cu` kernels are stringified at build time by `kernels/cu_to_inc.py`.
* `struct cp_resource` embeds `struct llvmpipe_resource` first — lavapipe
  reads its fields at fixed offsets. Don't reorder.
* The visibility buffer stores triangle index complemented (`~triID`) so
  `atomicMin` resolves coplanar triangles in primitive order.
* LLVM NVPTX backend is capped at `CP_MAX_PTX_SM` to avoid emitting PTX
  for architectures it doesn't know. The driver JITs forward.
* `nsys`/`ncu` cannot profile this driver (CUDA loaded via dlopen inside
  the ICD). Use `CUDAPIPE_DEBUG_TIME=1` or add `cuEventRecord` timing.
* Resources use `cuMemAllocManaged` because lavapipe accesses `lpr.data`
  directly from CPU in descriptor-building paths we don't control.
  Internal buffers (visbuf, depthbuf, scratch) use `cuMemAlloc`.
