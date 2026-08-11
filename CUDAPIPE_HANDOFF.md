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

### Adaptive Rasterizer (NEW)

The single-thread-per-triangle rasterizer has been replaced with a 3-stage
adaptive dispatch:

| Stage | Scope | Method |
|---|---|---|
| Stage 1 | bbox ≤ 128 px | 1 thread/tri, incremental edge stepping |
| Stage 2 | bbox ≤ 4096 px | 1 warp (32 threads) per tri, `__shfl_sync` broadcast |
| Stage 3 | bbox > 4096 px | 1 block (64 threads) per 64×64 tile, trivial accept/reject |

**Performance (GFXReconstruct replay):**

| Metric | Before | After (adaptive) |
|---|---|---|
| ms/draw average | 0.35 | 0.18 |
| Draws before crash | 522 (15s) | 3042 |
| GPU utilization | 95–99% | 57–64% |

The ~2× raster speedup shifts the bottleneck from GPU compute to CPU draw
submission (single-threaded `cuLaunchKernel` calls).

### Roblox HeadlessStreamer (live)

HeadlessStreamer now runs live against cudapipe with software encoding.
Device creation succeeds, draws execute on GPU, video frames are produced.
Frames are currently **black** due to a memory-model mismatch (see "Known
Issues" below).

**Command:**
```bash
ICD=build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json
VK_DRIVER_FILES=$ICD ./HeadlessStreamer \
    --authCookie "$COOKIE" --placeId "$PLACEID" \
    --softwareEncoder \
    --setFlags=FFlagHeadlessStreamerEnableGIR=false,FFlagVulkanVideoEncodingEnabled=false,FFlagSupportHeadlessDeviceVulkan=true
```

Required app-side change: in `DeviceVulkan.cpp:2271`, remove the
`FFlag::VulkanVideoEncodingEnabled` guard from the headless framebuffer
creation so it fires unconditionally in headless mode:
```cpp
// Before:  if (FFlag::SupportHeadlessDeviceVulkan && FFlag::VulkanVideoEncodingEnabled)
// After:   if (FFlag::SupportHeadlessDeviceVulkan)
```

**Fixes applied for live mode:**
- `VK_KHR_swapchain` advertised unconditionally (app requests it at device
  creation even in headless mode)
- `VK_PHYSICAL_DEVICE_TYPE_OTHER` instead of `CPU` (app rejects CPU devices
  when video encoding flag is set on non-Linux-client builds)
- `cp_fence_finish` uses `cuCtxSynchronize()` (the fence handle from
  lavapipe's threaded submit isn't a real CUevent)
- `cp_launch_grid` guards `cuMemAllocManaged` failure (returns early instead
  of crashing on NULL)
- `cp_blit` CPU fallback path syncs GPU before reading managed memory
- `cp_resource_copy_region` syncs GPU then uses memcpy (destination may be
  host-only memory)

### dEQP

`dEQP-VK.api.smoke.*`: 6/6 pass.
`dEQP-VK.texture.filtering.2d.formats.r8g8b8a8_unorm*`: 6/6 pass (12 unsupported).

### GFXReconstruct replay

```bash
ICD=build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json
GFX=~/gfxreconstruct/build
CUDAPIPE_DEBUG_TIME=1 VK_DRIVER_FILES=$ICD \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
    /tmp/headless_streamer_20260809T231703.gfxr
```

Runs 3042 draws at 0.18 ms/draw, ~14 seconds total, then crashes (lavapipe
descriptor bug — see below).

## Known Issues

### 1. Black frames in live HeadlessStreamer (BLOCKING)

**Root cause:** `pipe_screen::allocate_memory` (backing for `VkDeviceMemory`)
uses `calloc` because `cuMemAllocManaged` fails from lavapipe's submit thread
(CUDA context not current). The app maps `VkDeviceMemory` directly to read
framebuffer pixels, but the GPU rasterizer writes to a separate CUDA-managed
buffer (from `cp_resource_create`). These are different physical memory.

**What works:** The GFX replay uses `--remove-unsupported` and doesn't do
`vkMapMemory`-based readback — it just exercises the draw path. dEQP passes
because its readback goes through `pipe_transfer_map` which returns the
CUDA-managed `tex_data` pointer (after sync).

**Fix direction:** Make `allocate_memory` return CUDA-accessible memory from
any thread. Options:
- Use CUDA's primary context API (`cuDevicePrimaryCtxRetain`) which is
  thread-safe, instead of `cuCtxCreate`. Failed previously because something
  in the process releases the primary context (error 716). Needs investigation.
- Use `cuMemAllocHost` (pinned host memory, GPU-accessible, no context needed
  per-thread). Needs context current at allocation time but memory is
  accessible from any thread after.
- Preallocate a pool of CUDA-managed memory on the main thread and hand out
  chunks from it (avoids per-allocation CUDA calls from the submit thread).

### 2. Lavapipe descriptor crash (GFX replay only, ~3042 draws)

Segfault in `lvp_descriptor_set_create` → `memset(NULL)` when
`allocate_memory` returns NULL. With the `calloc` fallback this doesn't
crash (calloc doesn't fail for small sizes). With `cuMemAllocManaged` it
fails from the submit thread. This is the same underlying issue as #1.

### 3. Stage 2/3 rasterizer needs pixel bounds check

Stage 2's warp-parallel loop can produce pixel coordinates outside the
framebuffer when edge cases in floating-point screen-space positions occur.
A bounds check (`px < 0 || px >= width || py < 0 || py >= height`) is in
place and prevents CUDA_ERROR_ILLEGAL_ADDRESS.

## Architecture

```
Vulkan app
    ↓
lavapipe frontend (reused as-is)
    ↓ pipe_context calls (on submit thread)
cudapipe Gallium driver
    ├── State changes write into persistent cp_gpu_state (managed memory)
    ├── draw_vbo:
    │   1. cuMemsetD32 queue counters
    │   2. cp_vertex_fetch kernel (GPU gathers attributes)
    │   3. Vertex shader kernel (NIR → PTX)
    │   4. cp_rasterize_stage1 (small tris in-place, large → queue)
    │   5. cp_rasterize_stage2 (warp-cooperative, huge → tile queue)
    │   6. cp_rasterize_stage3 (block per tile, trivial accept/reject)
    │   7. cp_fs_interpolate (compact pixels, interpolate varyings)
    │   8. Fragment shader kernel (one thread per covered pixel)
    │   9. cp_fs_writeback (blend into colour attachment)
    └── flush: cuCtxSynchronize + scratch reclaim
```

## Files Modified (this session)

| File | Change |
|---|---|
| `kernels/cp_rast_types.h` | Queue structs, thresholds, `cp_tile_pair` |
| `kernels/cp_rasterize.cu` | 3-stage adaptive rasterizer |
| `cp_context.h` | Queue device pointers |
| `cp_context.c` | Queue alloc, 3-stage launch, `launch_grid` error guard |
| `cp_kernels.h` | Stage 1/2/3 function handles |
| `cp_kernels.c` | Load 3 kernel functions |
| `cp_screen.c` | `cuCtxCreate` (private context), `fence_finish` sync, device name |
| `cp_resource.c` | `allocate_memory`→calloc, `free_memory`→free, blit/copy sync |
| `lvp_device.c` | `VK_KHR_swapchain` unconditional, `DEVICE_TYPE_OTHER` |

## Performance Design

| Principle | Implementation |
|---|---|
| No per-draw sync | `cuCtxSynchronize` only at flush/readback |
| GPU vertex fetch | `cp_vertex_fetch.cu`; TRIANGLE_LIST reads IB on GPU |
| Adaptive rasterize | 3-stage dispatch matches triangle size to parallelism |
| No CPU topology expansion | `cp_build_vertex_refs` skipped for 94% of draws |
| Direct managed writes | Small arg structs written directly (no cuMemcpyHtoD) |
| Bounded memory | Scratch grows monotonically, reclaimed at flush |

## Next Steps (priority order)

1. **Fix black frames** — solve the `allocate_memory` threading problem so
   `VkDeviceMemory` and the GPU render target share the same physical memory.
   This is the only blocker for live HeadlessStreamer producing video.

2. **Restore adaptive rasterizer threshold** — currently `CP_SMALL_THRESHOLD`
   is set to 999999 (all stage 1) for debugging. Restore to 128 once the
   stage 2 bounds issue is fully resolved.

3. **Kernel fusion** — merge interpolate + FS + writeback to reduce launch
   count from 9 to 6 per draw.

4. **Line rasterization** — Bresenham kernel for `line_list`/`line_strip`.

## Debug

| Variable | Effect |
|---|---|
| `CUDAPIPE_DEBUG_TIME` | per-draw timing breakdown |
| `CUDAPIPE_DEBUG_DRAW` | draw call summary, blit info, pixel count |
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
* Resources use `cuMemAllocManaged` (in `cp_resource_create`) because
  lavapipe accesses `lpr.data` directly from CPU. Internal buffers (visbuf,
  depthbuf, scratch, rasterizer queues) use `cuMemAlloc`.
* `pipe_screen::allocate_memory` uses `calloc` as a workaround for CUDA
  context threading. This is why `VkDeviceMemory`-backed resources can't be
  GPU-rendered (the black frames issue).
* `cuCtxCreate` gives cudapipe a private CUDA context. `cuCtxSetCurrent` is
  called before every CUDA operation. The submit thread and main thread
  contend for the context — only one can use it at a time.
* `VK_KHR_swapchain` is advertised but not implemented (HeadlessStreamer
  requests it at device creation but never calls swapchain functions).
