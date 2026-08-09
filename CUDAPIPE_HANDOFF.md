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

**GFXReconstruct** for real applications. Capture a frame on a driver that
works, then read what it required and replay it against cudapipe. Built at
`~/gfxreconstruct` with `-DGFXRECON_ENABLE_OPENXR=OFF` (the bundled OpenXR
loader wants `xcb/glx.h`, and nothing here needs it).

```bash
GFX=~/gfxreconstruct/build

# capture on hardware
VK_LAYER_PATH=$GFX/layer VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_gfxreconstruct \
GFXRECON_CAPTURE_FILE=/tmp/app.gfxr \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json <application>

# what did it require?
$GFX/tools/convert/gfxrecon-convert --output /tmp/app.json /tmp/app_*.gfxr
python3 tests/cp_capture_requirements.py /tmp/app.json

# replay it on cudapipe
VK_DRIVER_FILES=<cudapipe icd>.json \
   $GFX/tools/replay/gfxrecon-replay -m remap /tmp/app_*.gfxr
```

`-m remap` is required: memory type indices are baked into the capture and
differ between drivers. Do **not** pass `--wsi`; it adds surface extensions
this driver deliberately doesn't have, and instance creation then fails.

Current result: 15 pixels of 262144 differ by more than 8/255 from the RTX
5090's output, all on triangle edges. Timing on that scene (768 triangles,
512x512) is 2.3 ms against 0.02 ms for the hardware — roughly 100x slower.

Note the benchmark texture is a smooth gradient on purpose. A checkerboard
minifies into heavy aliasing, where two *correct* implementations sampling
slightly different points disagree enormously; that noise masks real bugs.

## Status

Verified with dEQP (`vulkan_headless` target). Counts are of *supported* cases;
each group also reports many `NotSupported` that the driver never sees.

| Area | Result |
|---|---|
| `texture.filtering.2d.formats.*` | 72/75 |
| `texture.filtering.2d_array.formats.*` | 72/75 |
| `texture.filtering.cube.formats.*` | 36/150 |
| `texture.filtering.3d.formats.*` | 12/75 |
| `compute.pipeline.basic.*` | 70/71 |
| `draw...simple_draw.*` | 4/4 |
| `draw...basic_draw.draw.*` | 8/12 |

A capability sample of 406 cases spread across the whole suite, run one process
per case so crashes don't abort the sweep, went from 111 crashes to 42 once the
driver stopped advertising what it can't do. The same sample on this machine's
NVIDIA driver passes 243 and crashes none, which is the yardstick.

```bash
# every 8000th case, then one process per case
awk 'NR % 8000 == 0' all_cases.txt > sample.txt
```

Known gaps, roughly in the order they matter for a real workload:

1. **Performance.** Vertex assembly is done on the host, per draw, with a
   `memcpy` per attribute per vertex, and the rasterizer is one thread per
   triangle walking its bounding box — so a single large triangle serializes
   onto one CUDA thread. Together these dominate everything: the full
   `draw.dynamic_rendering` group does not finish in 25 minutes. Fixing it
   means GPU-side vertex fetch and a binned/tiled rasterizer.
3. **Lines and points are not rasterized at all** — only triangles. The four
   remaining `basic_draw.draw` failures are `line_list`, `line_strip` and
   `point_list`.
4. **Filtering between cube faces and between 3D slices** is not implemented,
   which is most of the remaining cube and 3D failures.
5. **Compressed formats are written but unverified.** DXT1/3/5 decode exists in
   `cp_fetch_texel` but nothing in the suites run so far exercises it. BC4-7 are
   missing. These matter for real game content.
6. **Depth/stencil aspect sampling** returns floats only, so the `*_stencil`
   and `s8_uint` filtering cases fail; they need integer texture returns.
7. **Shadow compares and texture gathers** fall through to a zero result in
   `emit_tex()`. Shadow compares in particular are needed for shadow mapping.
8. The rasterizer's depth buffer is internal and is never written back to the
   application's depth attachment, so a shader cannot sample depth from a
   previous pass.
9. `copy_ssbo_bounds` fails — SSBO robustness/bounds behaviour.

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
