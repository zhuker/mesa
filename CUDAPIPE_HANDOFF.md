# cudapipe Handoff — Continuing the CUDA Rasterizer

## What This Is

A new Mesa Vulkan ICD (`libvulkan_cudapipe.so`) that rasterizes triangles on NVIDIA GPUs using CUDA compute kernels instead of fixed-function hardware. Targets GPUs without rasterization HW (like compute-only accelerators). Built on a Tesla T4 (SM 7.5).

## Branch & Build

```bash
cd /home/coder/git/mesa
git checkout cudapipe   # 39 commits ahead of main

# Build
meson setup build-cudapipe -Dvulkan-drivers=swrast -Dgallium-drivers=llvmpipe,cudapipe \
  -Dllvm=enabled -Dglx=disabled -Degl=disabled -Dplatforms= -Dgbm=disabled \
  -Dgles1=disabled -Dgles2=disabled -Dopengl=false -Dglvnd=disabled
ninja -C build-cudapipe

# Test
VK_ICD_FILENAMES=/home/coder/git/mesa/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_icd.x86_64.json \
  /home/coder/git/VK-GL-CTS/build/external/vulkancts/modules/vulkan/deqp-vk \
  --deqp-archive-dir=/home/coder/git/VK-GL-CTS/external/vulkancts/data \
  --deqp-case=dEQP-VK.draw.dynamic_rendering.primary_cmd_buff.simple_draw.simple_draw_triangle_list
```

## What Works

| Feature | Status | dEQP Verification |
|---|---|---|
| Compute shaders | ✅ 50/50 basic tests pass | All Roblox compute ops (SSBO, UBO, shared mem, atomics, images) |
| VS execution on CUDA | ✅ Pixel-perfect | simple_draw, draw_indexed tests pass |
| Triangle rasterization | ✅ Visibility buffer + atomicMin | Correct shape, coverage, depth |
| Color interpolation | ✅ Barycentric from VS varyings | Gradient matches reference exactly |
| Indexed draws | ✅ uint16/uint32 | draw_indexed tests pass |
| Multi-draw | ⚠️ Partial | Single draws perfect, 45-draw has issues |
| Blending | ✅ SRC_ALPHA/ONE_MINUS_SRC_ALPHA | Implemented in CPU resolve |
| Depth test | ✅ Implicit via visbuf | Closest triangle wins |
| Texture sampling | ❌ Returns white | Architecture complete, parameter bug |

## Immediate Task: Fix Texture Sampling

The texture test (`dEQP-VK.texture.filtering.2d.formats.r8g8b8a8_unorm.nearest`) renders white instead of the expected colorful pattern. The reference image is a 64x64 grid of colored cells.

### Root Cause Chain

The texture data IS in managed memory (CPU-accessible). The path to sample it:

1. **VS reads UVs from VB 1** → `load_input(base=1)` in the VS kernel
2. **VS outputs UVs as varying** → `store_output(base=1)` 
3. **CPU resolve interpolates UVs** → barycentric interpolation of varying slot
4. **CPU resolve samples texture** → reads from FS UBO[1] descriptor → tex base ptr → texel

### Where It's Broken

The output is pure white, meaning either:
- The VS doesn't produce visbuf hits (quad not rasterized) — unlikely since draw is reported
- The VS outputs wrong varying data (UVs are all zero or the extraction fails)
- The texture descriptor isn't found (FS UBO[1] not bound or wrong format)

### Debugging Steps

1. **Check if VS produces visbuf hits for texture test:**
   - Add `cuCtxSynchronize()` after rasterize, count non-empty visbuf entries
   - If 0 hits: VS positions are wrong (quad renders off-screen)

2. **Check VS varying output (UV values):**
   - After VS runs, print `packed_colors[0..3]` — should be UV values (0-1 range)
   - If all zeros: multi-VB assembly failed (elem[1] from VB 1 not copied correctly)

3. **Check FS UBO[1] binding:**
   - Print `cp->fs_ubos[1].buffer` — should be non-NULL
   - Print first 8 bytes as pointer — should be valid tex data address

4. **Check texture data at descriptor address:**
   - Dereference the base ptr from FS UBO descriptor
   - First pixels should be non-zero (colorful test texture)

### The Multi-VB Assembly Bug

The texture test has:
```
elem[0]: offset=0, fmt=24 (RGBA32F), vb=0, stride=16  (position)
elem[1]: offset=64, fmt=22 (RG32F), vb=1, stride=???  (UV)
```

Our assembly code (`cp_context.c` ~line 290) copies each element from its VB:
```c
char *src = evb_start + vert_idx * elem_stride + src_off;
memcpy(vs_in + out_off + e * 16, src, copy_size);
```

`elem_stride` = `cp->vertex_elements[e].src_stride` — this might be wrong for elem[1] if it's 0 or if `src_off=64` causes reading past buffer.

### Key Files

| File | Role |
|---|---|
| `src/gallium/drivers/cudapipe/cp_context.c` | Draw pipeline, VS launch, CPU resolve |
| `src/gallium/drivers/cudapipe/nir_to_ptx/cp_nir_to_llvm.c` | NIR → LLVM IR → PTX shader compiler |
| `src/gallium/drivers/cudapipe/cp_resource.c` | Resource management, clear, copy |
| `src/gallium/drivers/cudapipe/cp_screen.c` | CUDA init, capabilities |
| `src/gallium/drivers/cudapipe/cp_kernels.c` | NVRTC kernel compilation |
| `src/gallium/drivers/cudapipe/kernels/cp_rasterize.cu` | Triangle rasterizer + resolve kernel |
| `src/gallium/drivers/cudapipe/kernels/cp_rast_types.h` | Shared host/device structs |

### Architecture

```
Vulkan App (SPIR-V)
    ↓
Lavapipe Frontend (reused as-is, liblavapipe_st.a)
    ↓ pipe_context calls
cudapipe Gallium Driver
    ├── create_compute_state → cp_compile_nir_to_ptx() → CUmodule
    ├── create_vs_state → nir_lower_io + cp_compile_nir_to_ptx() → CUmodule  
    ├── create_fs_state → nir_lower_io + cp_compile_nir_to_ptx() → CUmodule
    ├── launch_grid → cuLaunchKernel (compute)
    └── draw_vbo:
        1. Assemble vertices from VBs (multi-VB, indexed, strip/fan)
        2. Launch VS kernel (transforms positions, outputs varyings)
        3. Launch rasterize_triangles kernel (atomicMin visbuf)
        4. CPU resolve (barycentric interpolation, texture sampling, blending)
        5. Write to color buffer
```

### Descriptor System (Critical for Textures)

Lavapipe uses a "descriptor heap" where SSBO/UBO/texture descriptors are packed into constant buffers. The shader accesses them via:
- `load_const_buf_base_addr_lvp(slot)` → returns base address of UBO[slot]
- The UBO contains `lp_jit_buffer` structs (for SSBOs: {ptr base, u32 num_elements})
- Or `lp_image_descriptor` structs (for textures: {ptr base, u32 width, u16 height, ...})

For compute shaders, this works perfectly (50/50 tests pass). The same mechanism is used for FS texture access — the texture data pointer is in the descriptor buffer.

### Key Struct: llvmpipe_resource

We embed `struct llvmpipe_resource` as the first field of `struct cp_resource` because lavapipe's descriptor code calls `llvmpipe_resource_data()` which reads from fixed offsets (tex_data at 440, data at 456). This is critical — DON'T change the struct layout without verifying these offsets.

### Debug Environment Variables

- `CUDAPIPE_DEBUG_DRAW=1` — prints draw call info (tri count, viewport, stride)
- `CUDAPIPE_DUMP_NIR=1` — dumps VS/FS NIR before compilation
- `CUDAPIPE_DUMP_PTX=1` — dumps generated PTX
- `CUDAPIPE_DUMP_IR=1` — dumps LLVM IR before PTX emission

### After Texture Fix: Remaining Work for Full Roblox

1. **Multi-draw precision** — the .45 tests (45 sequential draws) fail because vertex_id assembly doesn't account for per-draw offsets correctly
2. **Depth buffer write** — visbuf gives implicit depth test but Roblox needs explicit depth buffer for shadow maps / multi-pass
3. **More blend modes** — currently only SRC_ALPHA/ONE_MINUS_SRC_ALPHA; Roblox may use additive blending
4. **GPU-side FS execution** — current CPU resolve is slow; for production, run FS as CUDA kernel per-pixel
5. **Performance** — switch from cuMemAllocManaged to explicit cuMemAlloc + copies for hot resources

### Test Commands

```bash
# All draw tests
VK_ICD_FILENAMES=...cudapipe_icd.x86_64.json deqp-vk --deqp-archive-dir=...data \
  --deqp-case=dEQP-VK.draw.dynamic_rendering.primary_cmd_buff.simple_draw.*

# Compute tests  
deqp-vk --deqp-case=dEQP-VK.compute.pipeline.basic.*

# Texture test (currently fails)
deqp-vk --deqp-case=dEQP-VK.texture.filtering.2d.formats.r8g8b8a8_unorm.nearest

# Quick sanity check
deqp-vk --deqp-case=dEQP-VK.draw.dynamic_rendering.primary_cmd_buff.simple_draw.simple_draw_triangle_list
```

### Dependencies on This Machine

- CUDA 12+ (nvcc at /usr/bin/nvcc, libcuda.so, libnvrtc.so, libnvJitLink.so)
- LLVM 18 with NVPTX backend (`llc-18 --version` shows nvptx64)
- Tesla T4 GPU (SM 7.5)
- dEQP at /home/coder/git/VK-GL-CTS/build/external/vulkancts/modules/vulkan/deqp-vk
- CuRast reference at /home/coder/git/CuRast/ (for rasterization algorithm inspiration)
