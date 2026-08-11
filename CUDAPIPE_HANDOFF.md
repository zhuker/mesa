# cudapipe — CUDA Software Rasterizer for Roblox

## What This Is

A Mesa Vulkan ICD (`libvulkan_cudapipe.so`) that rasterizes on NVIDIA GPUs using
CUDA compute kernels instead of fixed-function hardware. It reuses lavapipe as
the Vulkan frontend and replaces the Gallium driver underneath.

The feature set is driven by what Roblox's HeadlessStreamer actually requires,
measured from a GFXReconstruct capture — see
`src/gallium/drivers/cudapipe/tests/headless_streamer_requirements.txt`.

## Build

```bash
meson setup build-cudapipe -Dvulkan-drivers=swrast \
  -Dgallium-drivers=llvmpipe,cudapipe -Dllvm=enabled -Dglx=disabled \
  -Degl=disabled -Dplatforms= -Dgbm=disabled -Dgles1=disabled -Dgles2=disabled \
  -Dopengl=false -Dglvnd=disabled
ninja -C build-cudapipe
```

meson and ninja come from a pip venv at `./venv` (the distro's meson is too
old); it also needs `mako`, `packaging` and `pyyaml`. If the venv is missing,
`ninja` fails before compiling anything, because `build.ninja` records the
absolute path of the meson that generated it — recover with
`./venv/bin/meson setup --reconfigure build-cudapipe`.

CUDA is 12.8 at `/usr/local/cuda`, so `cuCtxCreate` takes three arguments;
code written against CUDA 13's four-argument form does not compile. The Vulkan
SDK is at `~/vulkan-sdk/1.4.357.1/x86_64` — put its `bin` on `PATH`.

```bash
export PATH="$PWD/venv/bin:$HOME/vulkan-sdk/1.4.357.1/x86_64/bin:$PATH"
ninja -C build-cudapipe src/gallium/targets/cudapipe/libvulkan_cudapipe.so
```

Run without installing:

```bash
VK_DRIVER_FILES=$PWD/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json \
  <vulkan app>
```

## Testing: the sample sweep

The Sascha Willems samples at `~/git/Vulkan` all render offscreen and
reproducibly, which makes them a differential oracle: run the same binary
against the NVIDIA ICD and against cudapipe, and diff the frames. This is a far
stronger signal than dEQP's per-feature pass/fail, and it found every bug fixed
so far.

`tests/headless_streamer_samples.txt` is the minimal set — each entry is the
simplest sample covering a capability the capture needs, with the mapping in the
file.

```bash
cd ~/git/Vulkan
S="$(grep -v '^#' ~/mesa/src/gallium/drivers/cudapipe/tests/headless_streamer_samples.txt | tr '\n' ' ')"
ICD=~/mesa/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json

SAMPLES="$S" VALIDATION=0 OUT=build/compare/ref \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./run_offscreen.sh
SAMPLES="$S" VALIDATION=0 OUT=build/compare/cuda \
  VK_DRIVER_FILES=$ICD ./run_offscreen.sh

cd build/compare
python3 ~/mesa/src/gallium/drivers/cudapipe/tests/cp_gallery.py ref cuda \
    -o cudapipe_vs_nvidia.html
```

`cp_compare.py` diffs two images (PNG or the PPM the samples write) and prints a
percentage plus an ASCII map. `cp_gallery.py` builds an HTML page: summary table
sorted worst-first, then reference / result / difference per sample, every panel
full resolution and clickable, with the result hover-flipping to the reference.

## Status

Differing pixels versus the NVIDIA driver at tolerance 8/255, on the committed
tree (`f713031fc91`, i.e. without the in-progress anisotropy work):

| Sample | Differing | Note |
|---|---|---|
| triangle | 0 | |
| pushconstants | 0 | |
| negativeviewportheight | 0 | |
| renderheadless | 0 | host readback path |
| texture | 4 | |
| texture3d | 4 | |
| bloom | 10 | |
| dynamicuniformbuffer | 15 | |
| vulkanscene | 34 | |
| multithreading | 0.23% | speckle, cause unknown |
| computeshader | 0.31% | |
| texturecubemap | 1.32% | reflections too sharp, see below |
| particlesystem | 2.65% | no POINT_LIST rasterization |
| pbribl | 3.02% | reflections too sharp |
| multisampling | 3.20% | no MSAA |
| gltfscenerendering | 6.48% | some alpha-tested leaves missing |
| instancing | 8.47% | cause unknown |
| texturemipmapgen | 18.37% | no anisotropic filtering |

Ten of eighteen are within a handful of pixels, from one before this work.

## In progress: anisotropic filtering

**Uncommitted, in the working tree.** `texturemipmapgen` renders a plane at a
grazing angle with `maxAnisotropy` at the device limit, and without aniso the
tunnel blurs out; the Roblox capture also lists anisotropic samplers, so this is
required work rather than polish.

`max_anisotropy` is plumbed from `pipe_sampler_state` through `cp_sampler_info`
into `cp_tex_sample`, which computes a tap count and steps along the footprint's
long axis, averaging. Where it stands:

| Sample | committed | current WIP |
|---|---|---|
| texturemipmapgen | 18.37% | 5.50% |
| instancing | 8.47% | 8.96% |
| gltfscenerendering | 6.48% | 9.24% |
| texture | 0.00% | 0.51% |

**What is left, and it is mechanical.** The current code derives the major axis
direction as `theta = 0.5 * atan2(B, A - C)` and then `cos`/`sin`. On a
near-isotropic footprint both `B` and `A - C` are near zero, so the angle is
numerical noise and the taps scatter — which is why `texture`, a flat quad
facing the camera, regressed.

llvmpipe solves this without trig. `lp_apply_ellipse_transform`
(`src/gallium/auxiliary/gallivm/lp_bld_sample.c:278`) rewrites the two
derivative vectors into an equivalent pair aligned to the ellipse's own axes, so
the cheap "longer of x or y" choice becomes correct:

```
A = dx.t² + dy.t²      C = dx.s² + dy.s²
B = -2(dx.s·dx.t + dy.s·dy.t)
F = det²,  det = dx.s·dy.t − dy.s·dx.t
p = A − C,  q = A + C,  t = sqrt(p² + B²)

newDx.s² = F(t+p) / (t(q+t))    newDx.t² = F(t−p) / (t(q+t))
newDy.s² = F(t−p) / (t(q−t))    newDy.t² = F(t+p) / (t(q−t))
```

guarded by three degeneracies that must be checked *before* applying it, and
which are the part hardest to rediscover: zero-length derivative, zero
determinant (parallel vectors), and zero dot product (already perpendicular, so
the transform is unnecessary). Then, from `lp_build_rho_aniso`:

```
eta² = clamp(rho_max² / rho_min², 1, aniso²)
rate = ceil(sqrt(eta²))          // tap count
rho_min² = rho_max² / eta²       // LOD basis — the UNROUNDED ratio
```

Dividing the level of detail by the rounded tap count instead of `eta²` drops it
below the minor axis and renders sharper than the hardware; that was the
over-sharpening in the first attempt, and correcting it recovered most of the
`instancing` and `gltfscenerendering` regression.

Replace the `atan2`/`cos`/`sin` block in `cp_sampler.cu` with the closed forms
and the three guards, keep the existing tap loop, and re-run the sweep.

## Architecture

```
Vulkan app
    ↓
lavapipe frontend (reused as-is)
    ↓ pipe_context calls (on submit thread)
cudapipe Gallium driver
    ├── State changes write into persistent cp_gpu_state (managed memory)
    ├── draw_vbo:
    │   1. cp_vertex_fetch      (GPU gathers attributes)
    │   2. Vertex shader kernel (NIR → PTX)
    │   3. cp_clip_triangles    (near plane and w > 0)
    │   4. cp_rasterize_stage1/2/3 (adaptive: thread, warp, block per tile)
    │   5. cp_fs_interpolate    (compact covered pixels, interpolate varyings)
    │   6. Fragment shader kernel (one thread per covered pixel)
    │   7. cp_fs_writeback      (discard mask, blend, into the attachment)
    └── flush: cuCtxSynchronize + scratch reclaim
```

Shaders run as CUDA kernels named `main` taking one argument, a pointer to an
array of pointers:

| Slot | Meaning |
|---|---|
| 0 | thread/vertex/pixel count |
| 2 | input buffer (vertex attributes, or interpolated varyings) |
| 3 | input stride |
| 4 | output buffer |
| 5 | vertex-id array |
| 6 | fragment coordinates |
| 7 | draw parameters |
| 8 | discard mask (fragment stage) |
| 18.. | uniform/descriptor buffers |

## Known gaps, roughly by how much they matter

1. **Derivatives only exist for varyings.** `emit_tex` traces a texture
   coordinate back to its `load_input` slot at compile time, because
   `cp_fs_interpolate` computes derivatives analytically per varying. A
   coordinate computed inside the shader — `reflect(-V, N)` for an environment
   map — gets none and samples the base level, which is why the spheres in
   `pbribl` and `texturecubemap` reflect too sharply. Fixing it properly means
   shading in 2x2 quads with cross-lane derivatives, which is also what
   `textureGrad` and the remaining `emit_tex` gaps need.
2. **Explicit LOD on cube samples** may not be honoured — `pbribl` uses
   `textureLod(prefilteredMap, R, roughness * mips)` and still reflects too
   sharply. Narrow and worth checking before the item above.
3. **Anisotropic filtering** — in progress, see above.
4. **No MSAA.** The capture needs 4x on D32_SFLOAT, A2B10G10R10 and R8_UNORM.
5. **No line or point rasterization.** The capture uses POINT_LIST.
6. **Alpha-tested geometry that overlaps itself within one draw** is not exact:
   visibility resolves before the shader runs, so a discarded fragment can
   already have occluded another of the same draw. Separate draws are fine.
7. **BC1/BC3 decode is written but never exercised** — no upstream sample uses
   compressed textures, and the capture has 576 BC images.
8. `multithreading` (0.23%) and `instancing` (8.47%) have no diagnosis yet.

## Debug

| Variable | Effect |
|---|---|
| `CUDAPIPE_DEBUG_DRAW` | draws, blend state, why a draw was skipped, clears, blits |
| `CUDAPIPE_DEBUG_TIME` | per-draw timing breakdown |
| `CUDAPIPE_DEBUG_TEX` | sampler/texture descriptor resolution |
| `CUDAPIPE_DEBUG_FS` | per-pixel fragment inputs/outputs, VS output positions |
| `CUDAPIPE_DEBUG_LAUNCH` | compute UBO/SSBO bindings |
| `CUDAPIPE_DEBUG_SHADER` | warn on unhandled NIR intrinsics |
| `CUDAPIPE_DUMP_NIR` / `DUMP_PTX` / `DUMP_IR` | dump shader IR at each stage |

**`compute-sanitizer` is the fastest way to diagnose a CUDA fault.** Errors 700
(illegal address) and 716 (misaligned) are sticky and surface at whatever launch
comes next, so the reported site is rarely the cause:

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --print-limit 2 \
    env VK_DRIVER_FILES=$ICD ./sample --offscreen -ofn out.ppm
```

That is how the vertex-fetch alignment fault and the out-of-bounds `imageLoad`
were found — the latter killed the whole frame, because once the context faults
even `cuMemAlloc` fails, and draws then bailed out for want of a visibility
buffer several stages away from the cause.

## Implementation notes

* `.cu` kernels are stringified at build time by `kernels/cu_to_inc.py`.
* `struct cp_resource` embeds `struct llvmpipe_resource` first — lavapipe reads
  its fields at fixed offsets. Don't reorder.
* The visibility buffer stores the triangle index complemented (`~triID`) so
  `atomicMin` resolves coplanar triangles in primitive order.
* `pipe_screen::allocate_memory` returns `cuMemAllocManaged` memory with
  `cuCtxSetCurrent` first, because lavapipe allocates from its submit thread. A
  context may be current on several threads at once.
* **An unimplemented operation that returns zero destroys everything
  downstream.** `textureSize` read as zero made a blur kernel compute `1/0` for
  its tap offsets and sample at infinity, blackening the whole frame;
  `terminate_if` returning undef collapsed shaders; `gl_InstanceIndex` silently
  did nothing for a while. `CUDAPIPE_DEBUG_SHADER=1` lists them.
* **Read the other Mesa backends before deriving an algorithm.** llvmpipe is in
  the same tree, and `lp_bld_sample.c` had a better answer for anisotropic
  filtering than two hand-derived attempts, including the degenerate-case guards.
* NIR keeps a storage image's format on the intrinsic and hands the shader
  whatever type it asked for, so the driver owns both the texel stride and the
  packing. Taking the stride from the destination type walks the image at the
  wrong rate.
* Vertex attributes with fewer than four components read as `(0, 0, 0, 1)`. A
  shader taking a vec3 position as vec4 otherwise loses its transform's
  translation column — the geometry still draws and still writes depth, so it
  looks like a missing object rather than a broken one.
* LLVM's NVPTX backend only knows architectures that existed when it was
  released, so `CP_MAX_PTX_SM` in `cp_nir_to_llvm.c` caps the architecture and
  lets the driver JIT forward. Raise it together with the PTX ISA version.
