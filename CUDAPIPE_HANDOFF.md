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

Differing pixels versus the NVIDIA driver at tolerance 8/255:

| Sample | Differing | Note |
|---|---|---|
| triangle | 0 | |
| pushconstants | 0 | |
| negativeviewportheight | 0 | |
| renderheadless | 0 | host readback path |
| texture3d | 4 | |
| bloom | 10 | |
| dynamicuniformbuffer | 15 | |
| vulkanscene | 27 | |
| texture | 593 | anisotropic taps on a slightly tilted quad |
| multithreading | 0.22% | speckle, cause unknown |
| computeshader | 0.31% | |
| texturecubemap | 1.32% | reflections too sharp, see gaps |
| particlesystem | 2.51% | no POINT_LIST rasterization |
| multisampling | 2.58% | no MSAA |
| pbribl | 3.02% | reflections too sharp |
| texturemipmapgen | 3.50% | anisotropic filter differences |
| gltfscenerendering | 1.70% | |
| instancing | 0.38% | |

Ten of eighteen are within a handful of pixels, from one before this work.

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

1. **`pbribl`'s prefiltered environment map is wrong above level 0.** The
   sample builds it by rendering at 512x512 and copying into each mip of a
   cube, and the driver only ever sees 512x512 and 64x64 render targets — so
   the mips come from `cp_resource_copy_region`, into a level and an array
   layer. This was invisible until `textureLod` started being honoured, since
   everything used to sample level 0; now the spheres lose their reflections
   and go dark, and the sample reads 5.58% where it used to read 3.02%. The
   earlier number was luck, not correctness.
2. **No MSAA.** The capture needs 4x on D32_SFLOAT, A2B10G10R10 and R8_UNORM.
4. **No line or point rasterization.** The capture uses POINT_LIST.
5. **Alpha-tested geometry** costs CP_DISCARD_LAYERS passes over the draw.
   Visibility resolves before shading, so a fragment that discards has already
   displaced the one behind it; each pass records what discarded where and
   repeats so the next fragment can win. Four layers took Sponza's foliage from
   543 discards to 1 across the passes. Anything still discarding after the
   last layer is lost.
6. **BC1/BC3 decode is written but never exercised** — no upstream sample uses
   compressed textures, and the capture has 576 BC images.
7. `multithreading` (0.23%) has no diagnosis yet.

`instancing` was 8.56% and is now 0.38%: NIR was lowering sin/cos to a cheap
polynomial (`.lower_sincos`), whose error is harmless when a shader rotates a
direction but not when it rotates a *position*. The sample's asteroids orbit at
radius 7 while their own vertices span 0.06 — an 80x lever that turned the
polynomial's error into a visible displacement of every rock, while the very
same sin in their local rotation was fine. The planet, which shares every
matrix but is not instanced, never moved, which is what localised it.

Worth remembering how it was found, because static reasoning got it wrong twice:
the sample's shader was edited directly (identity rotations, then each rotation
restored one at a time, then the hardware sin/cos swapped for a Taylor
polynomial) and re-run against both drivers. That bisect took minutes and was
conclusive where estimating error magnitudes was not — an approximation good to
1e-6 was dismissed as far too small to matter, and the real error was nearer
1e-3.

About a quarter of the sample's differing pixels are a separate and inherent
effect: `starfield.frag` builds stars from a hash that multiplies by ~440 and
takes `fract`, so any last-bit difference in an interpolated varying relocates
stars. Two correct implementations disagree there. Treat a procedural hash like
the checkerboard note above — a poor oracle, not a bug report.

Related: `nir_op_fsin`, `fcos`, `fexp2` and `flog2` emit the NVVM `.approx`
intrinsics unconditionally (`cp_nir_to_llvm.c:1316`), because the generic LLVM
ones lower to device-less libcalls. Roughly 2 ULP, which is looser than Vulkan
wants, though far too small to move geometry visibly. If tighter precision is
ever needed, llvmpipe's `lp_build_sin`/`lp_build_cos` carry the polynomial.

## Debug

| Variable | Effect |
|---|---|
| `CUDAPIPE_DEBUG_DRAW` | draws, blend state, why a draw was skipped, clears, blits |
| `CUDAPIPE_DEBUG_TIME` | per-draw timing breakdown |
| `CUDAPIPE_DEBUG_TEX` | sampler/texture descriptor resolution |
| `CUDAPIPE_DEBUG_FS` | per-pixel fragment inputs/outputs, VS output positions |
| `CUDAPIPE_DEBUG_LAUNCH` | compute UBO/SSBO bindings |
| `CUDAPIPE_DEBUG_VFETCH` | dump what the GPU vertex fetch gathered (syncs) |
| `CUDAPIPE_DEBUG_DISCARD` | covered and discarded pixels per alpha-test pass (syncs) |
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
  filtering than two hand-derived attempts, including the degenerate-case
  guards. The filter now follows `lp_build_rho_aniso()` and
  `lp_apply_ellipse_transform()`: the derivative pair is rewritten into the
  ellipse's own axes in closed form, and the level of detail divides by the
  unrounded clamped ratio rather than the rounded tap count.
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
