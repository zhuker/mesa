# cudapipe: CUDA-Based Vulkan Rasterizer for Mesa

## Context

A GPU without fixed-function rasterization hardware needs to render game content via Vulkan. We create a new Mesa Vulkan ICD (`libvulkan_cudapipe.so`) that performs rasterization in CUDA compute kernels, inspired by CuRast's visibility-buffer approach. Only Vulkan is targeted (no GL). The target engine requires Vulkan 1.0, vertex+fragment+compute stages, 32 textures, 4 MRTs, dynamic UBOs, storage buffers/images, and compute shaders for light culling.

## Architecture

```
The target application (SPIR-V shaders, Vulkan 1.0)
    │
    ▼
Lavapipe Frontend (reused as-is)
    ├── vk_cmd_queue recording
    ├── lvp_execute.c replay → pipe_context calls
    ├── SPIR-V → NIR compilation + lowering
    └── Descriptor/resource management
    │
    ▼ (pipe_context interface)
cudapipe Gallium Driver (NEW)
    ├── cp_screen.c    — CUDA device init, caps, format queries
    ├── cp_context.c   — pipe_context dispatch
    ├── cp_resource.c  — cuMemAllocManaged for buffers/images
    ├── cp_draw.c      — launches rasterization kernel pipeline
    ├── cp_compute.c   — cuLaunchKernel for compute shaders
    ├── nir_to_ptx/    — NIR → LLVM IR (NVPTX) → PTX compiler
    └── kernels/       — fixed-function rasterization .cu files
    │
    ▼
CUDA Driver API (cuLaunchKernel, cuMemAlloc, cuModuleLoad)
    │
    ▼
NVIDIA GPU (compute-only)
```

## Key Architectural Decisions

### 1. Shader execution: visibility buffer with deferred shading
- Vertex shaders → device-callable functions, invoked from vertex kernel (1 thread/vertex)
- Fragment shaders → device-callable functions, invoked from resolve kernel (1 thread/pixel)
- Compute shaders → `__global__` kernel entry points directly
- Rasterization kernels are fixed-function (CuRast-inspired), never recompiled per pipeline

### 2. Shader linking: nvJitLink at pipeline creation time
- `create_vs_state` / `create_fs_state`: NIR → LLVM IR → PTX (relocatable device code)
- First draw with new VS+FS combo: nvJitLink VS + FS + rasterizer kernel → cubin → cache
- CUmodule cached per (vs_hash, fs_hash) pair; amortizes link cost

### 3. Memory: cuMemAllocManaged
- Buffers/images use managed memory (host+device accessible, zero-copy maps)
- Framebuffers get `cuMemAdvise(CU_MEM_ADVISE_SET_PREFERRED_LOCATION, device)` hint
- Simplifies initial implementation; can migrate to explicit copies later for perf

### 4. Synchronization: one CUstream, CUevent for fences
- Single stream per Vulkan queue (the application uses one queue family)
- `pipe_fence_handle` wraps CUevent; `fence_finish` calls `cuEventSynchronize`
- Pipeline barriers within a stream are no-ops (CUDA stream is ordered)

## Rasterization Pipeline (per draw call)

Adapted from CuRast's 3-stage adaptive approach:

1. **Vertex kernel** — 1 thread/vertex. Calls compiled VS. Writes clip-space positions + varyings to transform buffer.
2. **Triangle setup kernel** — 1 thread/primitive. Perspective divide, NDC, screen coords. Backface/frustum cull. Classifies triangles by fragment count into work queues.
3. **Rasterize small** — 1 thread/triangle (< 128 fragments). Edge-function scan, `atomicMin(depth|triID)` to visibility buffer.
4. **Rasterize large** — 1 block per 64×64 tile. Tile-based with shared-memory edge functions.
5. **Fragment/resolve kernel** — 1 thread/pixel. Reads winning triangle ID, recomputes barycentrics, interpolates varyings, calls compiled FS, outputs color.
6. **Blend kernel** — 1 thread/pixel (where blend enabled). Applies blend equation.

## Shader Compiler: NIR → PTX

New module modeled after AMD's `ac_nir_to_llvm.c` but targeting NVPTX:

- Triple: `nvptx64-nvidia-cuda`, target SM version from device query
- Walk NIR SSA graph, emit LLVM IR with NVPTX intrinsics
- Key mappings:
  - `load_workgroup_id` → `blockIdx`
  - `load_local_invocation_id` → `threadIdx`
  - `load_ubo` / `load_ssbo` → pointer deref from kernel args
  - `barrier` → `@llvm.nvvm.barrier0`
  - `nir_tex_instr` → `@llvm.nvvm.tex.unified.*`
  - Shared memory → addrspace(3)
- For VS/FS: emit as device functions (not `__global__`), linked into rasterizer kernels

Reference files:
- `src/amd/llvm/ac_nir_to_llvm.c` — NIR→LLVM architecture template
- `src/gallium/auxiliary/gallivm/lp_bld_nir.h` — gallivm's NIR→LLVM (different target, same concept)
- `CuRast/src/CudaModularProgram.h` — NVRTC + nvJitLink workflow

## Implementation Phases

### Phase 0: Build skeleton + device enumeration
- Create `src/gallium/drivers/cudapipe/` with `cp_screen.c`, `cp_context.c` (stubs)
- Create `src/gallium/targets/cudapipe/` ICD target linking lavapipe frontend + cudapipe
- Modify `meson.options`, top-level `meson.build`, `src/gallium/meson.build`
- No WSI/swapchain needed — headless only (skip pipe_loader winsys, link `libws_null` only)
- `cuInit` + `cuDeviceGet` + `cuCtxCreate` at screen creation, targeting SM 7.5+
- **Milestone**: `VK_ICD_FILENAMES=cudapipe_icd.json vulkaninfo` shows cudapipe device
- **Build**: `meson setup build -Dvulkan-drivers=swrast -Dgallium-drivers=llvmpipe,cudapipe -Dllvm=enabled -Dglx=disabled -Degl=disabled -Dplatforms= -Dgbm=disabled -Dgles1=disabled -Dgles2=disabled -Dopengl=false -Dglvnd=disabled`
- **Test**: `VK_ICD_FILENAMES=build/src/gallium/targets/cudapipe/cudapipe_icd.x86_64.json vulkaninfo --summary`

### Phase 1: Resource management
- Implement `resource_create` / `destroy` / `map` / `unmap` with cuMemAllocManaged
- Implement `allocate_memory` / `resource_bind_backing` for lavapipe's unbacked resource pattern
- **Milestone**: dEQP-VK.api.buffer.* and dEQP-VK.memory.* pass

### Phase 2: Compute shader compiler + dispatch
- Build `nir_to_ptx/cp_nir_to_llvm.c` — compute shader subset of NIR → NVPTX LLVM IR → PTX
- Implement `create_compute_state`, `launch_grid` with cuLaunchKernel
- **Milestone**: dEQP-VK.compute.basic.* pass

### Phase 3: Rasterization kernels
- Write .cu kernel files (vertex processing, triangle setup, small/large rasterizers, resolve, blend, clear)
- Compile via NVRTC at screen init, cache as LTOIR for linking
- **Milestone**: Hardcoded triangle renders to readback buffer

### Phase 4: Draw pipeline integration
- Implement `draw_vbo`: assembles state, launches kernel pipeline
- Implement VS/FS compilation as device functions
- nvJitLink VS+FS+rasterizer → cubin at first draw; cache per pipeline
- **Milestone**: dEQP-VK.draw.simple_draw.* pass

### Phase 5: Texturing + descriptors
- CUDA Texture Objects for sampled images (bilinear filtering, wrapping)
- Descriptor argument buffer passed to kernels
- **Milestone**: dEQP-VK.texture.filtering.* pass

### Phase 6: Render pass, blit, synchronization
- Clear/load/store ops, CUevent fences, image blit kernel (mipmap gen)
- **Milestone**: dEQP-VK.renderpass.* pass

### Phase 7: application validation
- Indexed draws (uint16/uint32), 4 MRTs, depth formats, dynamic viewport/scissor
- Compare output against lavapipe reference
- **Milestone**: the client renders a game scene

## Target Configuration

- **GPU**: SM 7.5+ (Turing) — enables independent thread scheduling, warp-level primitives (`__shfl_sync`), cooperative groups
- **Display**: Headless only — no VK_KHR_swapchain needed, render to offscreen framebuffer with host readback
- **CUDA**: 12+ — nvJitLink available for cross-module LTO between shaders and rasterizer kernels
- **PTX target**: `sm_75` minimum, compile with `-arch=sm_75` for broadest Turing+ compatibility

Headless simplifies Phase 0 significantly: no WSI, no surface extensions, no pipe_loader winsys. The lavapipe frontend's headless path (used by the headless services) already supports `VK_NULL_HANDLE` surface.

## External Dependencies

| Dependency | Minimum | Required For |
|---|---|---|
| CUDA Driver API (`libcuda.so`) | 12.0 | All GPU ops |
| NVRTC (`libnvrtc.so`) | 12.0 | Rasterizer kernel compilation |
| nvJitLink (`libnvJitLink.so`) | 12.0 | Shader+kernel linking (LTO) |
| LLVM (NVPTX backend enabled) | 15 | NIR → PTX |

## Verification Strategy

Use Mesa's existing test infrastructure for sub-second iteration:

```bash
# Build
ninja -C build

# Run single dEQP-VK test (< 1 second)
VK_ICD_FILENAMES=build/src/gallium/targets/cudapipe/cudapipe_icd.json \
  deqp-vk --deqp-case=dEQP-VK.draw.simple_draw.draw_triangle_list

# Run a category
deqp-vk --deqp-case=dEQP-VK.compute.basic.*

# Compare framebuffer output against lavapipe reference
VK_ICD_FILENAMES=build/src/gallium/targets/lavapipe/lvp_icd.json deqp-vk ... > ref.qpa
VK_ICD_FILENAMES=build/src/gallium/targets/cudapipe/cudapipe_icd.json deqp-vk ... > test.qpa
```

Progressive test categories per phase:
- Phase 0: `dEQP-VK.api.device_init.*`, `dEQP-VK.api.info.*`
- Phase 1: `dEQP-VK.api.buffer.*`, `dEQP-VK.memory.*`
- Phase 2: `dEQP-VK.compute.basic.*`, `dEQP-VK.ssbo.*`
- Phase 4: `dEQP-VK.draw.simple_draw.*`, `dEQP-VK.rasterization.*`
- Phase 5: `dEQP-VK.texture.filtering.*`, `dEQP-VK.binding_model.*`
- Phase 6: `dEQP-VK.renderpass.*`

Additional tools:
- **Piglit** — simpler/faster Mesa tests
- **Trace replay** — `traces-lavapipe.toml` pattern for frame comparison
- **Validation layers** — always run with `VK_LAYER_KHRONOS_validation`
- the target application only needed for final Phase 7 integration

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| NIR→NVPTX compiler bugs | High | Start with compute (simplest), validate against dEQP-VK.glsl.*, compare with lavapipe |
| Rasterization precision (edge rules, subpixel) | Medium | CuRast proves feasibility; use top-left rule; validate with dEQP-VK.rasterization.* |
| nvJitLink latency at first draw | Medium | Cache aggressively; warm cache on pipeline create (not bind) |
| cuMemAllocManaged performance | Low | Acceptable for correctness-first; migrate hot paths later |
