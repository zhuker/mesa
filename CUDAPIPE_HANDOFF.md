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

## Scope

**Offscreen only.** Applications draw into `VkImage`s and read the result back.
The driver exposes no `VK_KHR_swapchain` and the build deliberately leaves
`-Dplatforms=` empty; dEQP runs against the `vulkan_headless` target for the
same reason. Nothing here depends on a display server.

**Feature target: what a simple Roblox-style game needs, and nothing else.**
Everything outside that is declared unsupported rather than half-implemented —
a clean refusal lets an application choose another path, while a false claim
crashes it.

The bar is deliberately low: think of the feature set a mid-range mobile GPU
exposes as a rough gauge of *how little* is required, not as a specification to
implement against. Features get added because something real fails without
them, not because some class of hardware happens to have them.

Wanted, and in place unless noted:

| | |
|---|---|
| Vertex and fragment shaders | yes |
| Indexed and instanced draws | yes |
| Depth test and depth write | yes |
| 2D, 2D array and cube textures, mipmapped | yes |
| Bilinear and trilinear filtering, wrap modes | yes |
| Alpha blending | yes |
| Uniform buffers, storage buffers, compute | yes |
| Compressed textures | DXT1/3/5 only, untested |
| Line and point primitives | **missing** |

Deliberately unsupported, and advertised as such:

* geometry and tessellation stages
* ray tracing, mesh shaders, transform feedback
* sparse resources, multisampling, protected memory
* stencil test and stencil attachments
* multiple colour attachments (the writeback resolves one)
* 64-bit integers in shaders
* anisotropic filtering, occlusion queries, conditional render, primitive restart

Compressed texture support is the open question. The sampler decodes DXT1/3/5
and nothing else — no ETC2, no ASTC, no BC4-7 — and none of that decode has
ever been exercised by a test. Which family actually matters depends on what
the content ships, which is worth measuring rather than guessing: run something
representative and see which formats it asks for. Each family is bounded,
well-specified work once it's known to be needed.

One consequence is easy to miss and matters: mobile GPUs use **ETC2 and ASTC**,
not BC/DXT. The sampler currently decodes DXT1/3/5 — a desktop format family —
and no ETC2 or ASTC at all. Committing to the mobile target makes ETC2 and
ASTC LDR decode required work, and BC optional.

A useful smell test for over-claiming: compare the advertised extension list
against a real GPU on the same machine. cudapipe should never advertise more
than hardware does.

```bash
for icd in /usr/share/vulkan/icd.d/nvidia_icd.json <cudapipe icd>.json; do
   VK_DRIVER_FILES=$icd vulkaninfo 2>/dev/null |
      awk '/^Device Extensions/,/^$/' | grep -oE "VK_[A-Za-z0-9_]+" | sort -u
done
```

It is a heuristic, not a law — lavapipe implements some extensions in the
frontend that NVIDIA has never shipped — but anything cudapipe claims and
hardware doesn't deserves justification.

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

## Testing

Two complementary harnesses, and both are needed:

**dEQP** (`vulkan_headless` target) for breadth of individual features.

**`tests/cp_offscreen_bench.c`** for whole-frame correctness and timing. It
renders a textured, depth-tested, instanced scene offscreen and writes a PNG,
and it is ICD-agnostic — so the same binary can be run against the machine's
real NVIDIA driver to produce ground truth, and `tests/cp_compare.py` diffs the
two. Differential testing against real hardware is a far stronger oracle than
dEQP's per-feature pass/fail.

```bash
cd src/gallium/drivers/cudapipe/tests
glslangValidator -V cp_bench.vert -o cp_bench.vert.spv
glslangValidator -V cp_bench.frag -o cp_bench.frag.spv
cc -O2 cp_offscreen_bench.c -o cp_offscreen_bench -lvulkan -lz -lm

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./cp_offscreen_bench 3 ref.png
VK_DRIVER_FILES=<cudapipe icd>.json                     ./cp_offscreen_bench 3 out.png
python3 cp_compare.py ref.png out.png
```

**GFXReconstruct** for real applications: capture a frame on a driver that
works, read back what it required, and replay it against cudapipe — which turns
a frame of real content into a regression test for this driver. Already built at
`~/gfxreconstruct`. See `tests/GFXRECONSTRUCT.md` for the full workflow,
including the two replay flags that are easy to get wrong.

Current result: 15 pixels of 262144 differ by more than 8/255 from the RTX
5090's output, all on triangle edges. Timing on that scene (768 triangles,
512x512) is 2.3 ms against 0.02 ms for the hardware — roughly 100x slower.

Note the benchmark texture is a smooth gradient on purpose. A checkerboard
minifies into heavy aliasing, where two *correct* implementations sampling
slightly different points disagree enormously; that noise masks real bugs.

## Status

### Real workload: Roblox HeadlessStreamer (GFXReconstruct replay)

A 60-second capture of Roblox's HeadlessStreamer (1280×720 offscreen, 77
graphics pipelines, 2 compute, vertex+fragment only) replays successfully
against cudapipe for 68 seconds before hitting a lavapipe descriptor bug
(which also crashes lavapipe alone — not a cudapipe issue).

| Metric | NVIDIA (T4) | cudapipe (T4) |
|---|---|---|
| Full replay wall clock | 19 s | ~68 s (crash) |
| GPU utilization | N/A | 95–99 % |
| GPU memory | ~400 MB | ~930 MB |
| Draws/sec (estimated) | 263 | ~73 |
| Ratio | 1× | ~3.6× slower |

The 3.6× gap is pure GPU rasterization/shading compute cost. The GPU is
fully saturated — no CPU stalls, no memory-migration overhead, no sync
gaps between kernel launches.

### dEQP

| Area | Result |
|---|---|
| `api.smoke.*` | 6/6 |
| `texture.filtering.2d.formats.*` | 72/75 |
| `texture.filtering.2d_array.formats.*` | 72/75 |
| `texture.filtering.cube.formats.*` | 36/150 |
| `texture.filtering.3d.formats.*` | 12/75 |
| `compute.pipeline.basic.*` | 70/71 |
| `draw...simple_draw.*` | 4/4 |
| `draw...basic_draw.draw.*` | 8/12 |

### Performance work done

1. Eliminated all mid-draw `cuCtxSynchronize` calls (was 7 per draw, now 0)
2. Implemented proper CUDA event fences (`flush` is non-blocking)
3. GPU vertex fetch kernel — no CPU-side attribute gathering
4. Replaced `cuMemcpyHtoD` with direct managed-memory writes
5. Persistent `cp_gpu_state` struct updated on state changes, not per draw
6. Scratch arena with monotonic growth + bounded reclaim
7. Skip `cp_build_vertex_refs` for TRIANGLE_LIST (94% of draws)
8. `cuMemsetD32` for visbuf/depthbuf clears, `cuMemcpy2D` for blits/copies

### What the Roblox capture required (and was added)

* Formats: R16_SFLOAT, R16G16_SFLOAT, R16G16_UNORM, A2B10G10R10_UNORM,
  R32_SINT, R16_SINT (sampler); R11G11B10_FLOAT, A2B10G10R10_UNORM,
  R16_SFLOAT, R16G16_SFLOAT, R8_UNORM (render target)
* Fixed R16G16B16A16_FLOAT render target store (was truncating to UNORM8)
* 4x MSAA format acceptance
* CUBE_ARRAY texture target
* POINT_LIST topology
* Anisotropic filtering advertised (trilinear fallback)
* Query/timestamp stubs
* Geometry/tessellation stage bind stubs
* Depth-only draw support
* Blit (same-format cuMemcpy2D + nearest-neighbor scaling)
* Fixed `cp_resource_bind_backing` memory offset (was ignoring it)
* Fixed NIR bcsel type mismatch in the LLVM backend

### Known gaps

1. **Rasterization speed.** One thread per triangle, bounding-box scan. A
   CuRast-style adaptive 3-stage rasterizer (1 thread/small tri, 1 warp/medium
   tri, 1 block/tile for large tris) would close most of the remaining 3.6×
   gap. See `/home/coder/git/CuRast/src/kernels/triangles_visbuffer.cu`.
2. **Lavapipe descriptor crash at replay second 68.** Happens with lavapipe
   alone too. Likely a descriptor pool exhaustion or layout the replay uses
   that lavapipe can't handle.
3. **Lines are not rasterized** — only triangles and points. `line_list` and
   `line_strip` draw calls silently produce nothing.
4. **Filtering between cube faces and 3D slices** is missing.
5. **Shadow compares and texture gathers** return zero.
6. **BC4-7, ETC2, ASTC** compressed format decode is missing (BC1/3/5 work).
7. **Profiling** — `nsys`/`ncu` cannot trace CUDA Driver API calls made from
   inside a dlopen'd Vulkan ICD. Workaround: lower
   `/proc/sys/kernel/perf_event_paranoid` to 2, or add in-driver
   `cuEventRecord` timing (env var `CUDAPIPE_DEBUG_TIME` shows per-stage
   wall-clock timing already).

### Next steps (priority order)

1. **Adaptive rasterizer** — the single biggest remaining performance win.
   Small triangles (< 128 fragments) stay at 1 thread/tri; medium triangles
   get 1 warp (32 threads) cooperatively scanning the bounding box; huge
   triangles get 1 block per 64×64 tile. This matches CuRast's proven
   architecture and would bring GPU utilization from "saturated but slow" to
   "saturated and efficient."
2. **Fix the descriptor crash** — investigate why `lvp_descriptor_set_create`
   segfaults after ~5000 draws. May be a descriptor pool limit or a layout
   with variable-count descriptors that lavapipe doesn't handle.
3. **Line rasterization** — Bresenham in a CUDA kernel, one thread per line.
4. **Kernel fusion** — merge interpolate + FS + writeback into fewer launches
   to reduce per-draw overhead (currently 5–6 kernel launches per draw).
5. **Shadow compare and texture gather** — needed for shadow mapping which
   most real games use.

## Debug environment variables

| Variable | Effect |
|---|---|
| `CUDAPIPE_DEBUG_DRAW` | draw call summary, vertex elements, shaded pixel count |
| `CUDAPIPE_DEBUG_TEX` | sampler/texture descriptor resolution |
| `CUDAPIPE_DEBUG_FS` | per-pixel fragment inputs/outputs and varying mapping |
| `CUDAPIPE_DEBUG_FS_ROW` | restrict `CUDAPIPE_DEBUG_FS` to one framebuffer row |
| `CUDAPIPE_DEBUG_LAUNCH` | compute UBO/SSBO bindings |
| `CUDAPIPE_DEBUG_SHADER` | warn on NIR intrinsics the backend doesn't implement |
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
* Unhandled NIR intrinsics return `undef`, and LLVM propagates that through
  everything downstream, so one missing intrinsic can collapse a whole shader
  into a constant with no error anywhere. `CUDAPIPE_DEBUG_SHADER=1` lists them.
  This is how `gl_InstanceIndex` silently did nothing for a while: it lowers to
  `load_instance_id + load_base_instance`, and only the first was implemented.
* The visibility buffer stores the triangle index *complemented*, so `atomicMin`
  resolves equal depths in favour of the last primitive. Vulkan requires
  primitive order for coplanar geometry; the obvious encoding gets it backwards.
