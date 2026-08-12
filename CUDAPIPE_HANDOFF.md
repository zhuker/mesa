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
against the NVIDIA ICD and against cudapipe, and diff the frames. That is a far
stronger signal than dEQP's pass/fail, and it found every bug fixed so far.

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

`cp_compare.py` diffs two images (PNG, or the PPM the samples write) and prints a
percentage plus an ASCII map. `cp_gallery.py` builds an HTML page: summary table
sorted worst-first, then reference / result / difference per sample, every panel
full resolution and clickable, the result hover-flipping to the reference.

**Editing a sample's shader is the fastest way to bisect a difference.** The
samples load `.spv` at runtime, so `glslangValidator -V shader.vert -o
shader.vert.spv` and re-running both drivers takes seconds and needs no C++
rebuild. Two bugs below were found that way after static reasoning got them
wrong; back the `.spv` up first and restore it afterwards.

## Status

Differing pixels versus the NVIDIA driver at tolerance 8/255, worst first.
Total across the set: 105,212, from 110,015 before the watertight-coverage and
cube-derivative work, 143,856 before the quad-derivative work, and far more
before that.

| Sample | Differing | Note |
|---|---|---|
| texturemipmapgen | 3.50% | anisotropic filter fidelity |
| multisampling | 2.58% | no MSAA |
| particlesystem | 2.51% | no POINT_LIST rasterization |
| gltfscenerendering | 1.70% | no diagnosis |
| texturecubemap | 0.45% | reflection sharper than the reference at grazing angles |
| instancing | 0.38% | mostly the procedural starfield, see below |
| computeshader | 0 | |
| multithreading | 0.22% | speckle, no diagnosis |
| pbribl | 28 px | |
| texture | 591 px | anisotropic taps on a slightly tilted quad |
| vulkanscene | 27 px | |
| dynamicuniformbuffer | 15 px | |
| bloom | 10 px | |
| texture3d | 4 px | |
| renderheadless | 0 | the host readback path |
| negativeviewportheight | 0 | |
| pushconstants | 0 | |
| triangle | 0 | |

Fourteen of eighteen are within a handful of pixels, from one before this work.
Of the four that are not, two are unimplemented features rather than defects.

### Calibrate against llvmpipe, not against zero

The same sweep run through lavapipe/llvmpipe, which is the backend cudapipe's
sampler and clipper were ported from, and which shares lavapipe as its frontend:

| Sample | llvmpipe vs NVIDIA | cudapipe vs NVIDIA | cudapipe vs llvmpipe |
|---|---|---|---|
| gltfscenerendering | 26263 | **15697** | 5975 |
| multisampling | **1454** | 23773 | 23145 |
| particlesystem | **327** | 23106 | 22416 |
| texturemipmapgen | **21070** | 32237 | 17921 |
| instancing | 5829 | **3506** | 1869 |
| texturecubemap | 5165 | **4144** | 3792 |
| multithreading | **1954** | 2087 | 1831 |
| texture | **426** | 589 | 320 |
| pbribl | 73 | **28** | 96 |
| vulkanscene | **11** | 25 | 20 |
| everything else | ~0 | ~0 | ~0 |
| **total** | **62574** | 105212 | 77405 |

Two things follow, and both change how the status table above should be read.

**A software rasterizer does not converge on NVIDIA.** llvmpipe is mature and
still differs by 62574 pixels over the set. Most of that is the same
anisotropic filtering difference cudapipe has: on texturemipmapgen llvmpipe's
mean error runs to +12.7/255 in the top luminance band against cudapipe's
+20.5, the same sign and shape, about 60% of the size. So there is real
headroom there, but the floor is llvmpipe's number, not zero.

**Excluding the two unimplemented features, cudapipe is already at parity.**
Take out multisampling and particlesystem, which are MSAA and POINT_LIST and
nothing to do with fidelity, and it is 58333 for cudapipe against 60793 for
llvmpipe. On gltfscenerendering cudapipe is closer to NVIDIA than llvmpipe is,
by a factor of 1.7, and that holds region by region across the surfaces where
the difference is most visible — the curtains, the pillars, the arches.

Worth re-running whenever a sampler or rasterizer change looks like it is not
paying off; llvmpipe is the honest target.

Note also that all 18 samples run clean under llvmpipe, while renderheadless,
gltfscenerendering and pbribl segfault during teardown under cudapipe after
writing their images. Those are cudapipe bugs, not sample bugs.

    VK_ICD_FILENAMES=build-cudapipe/src/gallium/targets/lavapipe/lvp_devenv_icd.x86_64.json \
      VALIDATION=0 OUT=build/compare/llvmpipe ./run_offscreen.sh

    cp_gallery.py ref cuda llvmpipe -o three.html \
      --ref-label nvidia --test-label cudapipe --test2-label llvmpipe

cp_gallery.py takes the third directory and lays the two renderers beside the
same reference, with a panel of the two against each other that goes black
wherever they agree.

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
    │   5. cp_fs_interpolate    (compact into 2x2 quads, interpolate varyings)
    │   6. Fragment shader kernel (four threads per quad)
    │   7. cp_fs_writeback      (drop helpers, discard mask, blend, attachment)
    │   steps 4-7 repeat up to CP_DISCARD_LAYERS times for an alpha-tested draw
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

**Fragments are shaded four to a 2x2 quad**, so the sampler can take
screen-space derivatives by shuffling between lanes rather than reading a stored
per-varying derivative. That is what lets a coordinate the shader computed for
itself — a reflection vector — pick a mip level at all. A quad belongs to one
triangle: `cp_fs_interpolate` emits a separate quad per distinct triangle in a
2x2 block, because differencing across two triangles produces a meaningless
derivative at every seam. Corners no triangle covers are still shaded as helper
lanes and dropped by the writeback.

## Known gaps, roughly by how much they matter

1. **No MSAA.** The capture needs 4x on D32_SFLOAT, A2B10G10R10 and R8_UNORM.
2. **No line or point rasterization.** The capture uses POINT_LIST.
3. **Anisotropic filtering under-blurs relative to NVIDIA's.** This is nearly
   all of what is left in `texturemipmapgen`, and it also drives most of
   `gltfscenerendering`. Forcing each of that sample's three sampler modes and
   rendering both drivers separates it cleanly: no mipmaps differs by 19 pixels,
   mipmaps with bilinear by 1095, mipmaps with anisotropy by 32237. So the
   texture upload, the runtime-generated mip chain and LOD selection are all
   effectively exact, and the specular `pow` is too — sampler 0 runs the same
   `pow(dot(R, V), 16)` and still lands within 19 pixels.

   The signature is contrast, not brightness: cudapipe is darker than the
   reference where the reference is dark and brighter where it is bright, by up
   to 20/255 in the top luminance band. Turning anisotropy off entirely
   overshoots the other way.

   Two tap-placement schemes have been measured. The one in the tree spreads N
   taps over `rho_max * (N-1)/N`, which tiles the footprint exactly when
   `N == eta`. Switching to llvmpipe's — step along the raw ddx or ddy rather
   than a unit vector rescaled to the transformed axis, and space taps by the
   Vulkan specification's `(t - N/2 + 1/2)/(N + 1)` — improves `texture`
   (589 -> 164) and regresses everything else: `texturemipmapgen` 32237 -> 38864,
   `gltfscenerendering` 15697 -> 25036, `instancing` 3506 -> 5346,
   `texturecubemap` 4144 -> 4774. Do not re-apply it wholesale; the two halves of
   it have not been measured separately.
4. **Alpha-tested geometry costs CP_DISCARD_LAYERS passes over the draw.**
   Visibility resolves before shading, so a fragment that discards has already
   displaced the one behind it; each pass records what discarded where and
   repeats so the next fragment can win. Sponza's foliage falls from 543
   discards to 1 within four passes; halving the layers from 8 to 4 costs it
   0.14% of the frame, and anything still discarding after the last layer is
   lost.
5. **BC1/BC3 decode is written but never exercised** — no upstream sample uses
   compressed textures, and the capture has 576 BC images.
6. **`gltfscenerendering` (1.70%) is mostly gap 3 above, amplified.** Its
   fragment shader samples a normal map and then raises the result to the 32nd
   power, so an under-blurred normal map turns into scattered specular glints on
   exactly the grazing-angle surfaces where anisotropy applies — the side walls,
   the pillars and arches, the curtain folds. Disabling anisotropy cuts the
   differences on the stone pillars and arch from 2128 to 1233 and on the red
   curtain from 1723 to 757, while making the frame as a whole worse (15697 to
   29726), which is what tells you the filter is under-blurring rather than
   simply wrong. The remaining large per-pixel differences are paired: one pixel
   much brighter in cudapipe and its neighbour much darker, on thin
   high-contrast features. There is no global subpixel shift — a +-1 pixel
   search puts the minimum at (0, 0) by a factor of fifteen.

   **`multithreading` (0.22%) still has no diagnosis.**
7. `nir_op_fexp2`, `flog2` and lowered `fpow` still use the NVVM `.approx`
   intrinsics. Routing pow to the CUDA library version was tried and changed the
   image without moving it closer to the reference, so it was reverted. `fsin`
   and `fcos` do *not* — see below.

## Lessons that cost the most to learn

**An unimplemented operation that returns zero destroys everything downstream.**
`textureSize` reading as zero made a blur kernel compute `1/0` for its tap
offsets and sample at infinity, blackening a whole frame; `terminate_if`
returning undef collapsed shaders; `gl_InstanceIndex` silently did nothing for a
while. `CUDAPIPE_DEBUG_SHADER=1` lists them. Treat a zero from a missing feature
as a fault, not a default.

**A CUDA fault is sticky and surfaces at the wrong place.** An `imageLoad` one
texel outside its image killed a whole frame: the context faulted, the next
`cuMemAlloc` failed, and draws then bailed out for want of a visibility buffer
several stages away from the cause. `compute-sanitizer` turns this back into a
kernel name and a line.

**Floating point contraction breaks exact symmetry, and the rasterizer depends
on it.** Coverage is watertight only if the two triangles sharing an edge
compute exactly opposite values for it, so that the fill rule can hand a pixel
sitting on the edge to one of them. Writing the edge function as a cross
product of the vectors from the pixel makes it antisymmetric on paper — swap
the endpoints and the same two products change places around the subtraction.
nvcc then contracts `a * b - c * d` into `fma(a, b, -(c * d))`, which keeps only
one product exact, and *which* one depends on the order they were written in.
The contracted form is not antisymmetric: on the quad in computeshader both
triangles computed the same small negative, `-3.87569889e-06`, rather than
opposite values, and both rejected — a one pixel crack down the shared edge for
its whole length. `__fmul_rn` and `__fsub_rn` prevent the contraction. For the
same reason the edge tests are evaluated from the vertices at every pixel
rather than stepped by their gradients, and stage 3 has no trivial accept.

**Confirm which arithmetic actually ran before theorising about it.** The
diagnosis above came from a `printf` in the rasterizer for one pixel, printing
the three edge values and the fill rule flags for every triangle that touched
it. It took one build and disproved a hypothesis that had already survived two
plausible arguments. Kernel `printf` guarded by a hardcoded pixel is cheap;
reason about float rounding only with the numbers in front of you.

**A derivative taken after a projection is wrong wherever the projection is
discontinuous.** Differencing the final texture coordinate across the quad
covers every target with one piece of code, which is why it was written that
way, but a cube's u and v jump between parameterisations at a face boundary. A
quad straddling a seam gets a huge derivative, the level of detail collapses to
the coarsest mip, and the seam draws itself — a wireframe of the cube over
every reflective surface. Differentiate the continuous quantity, the direction
vector, and push it through the projection analytically. The same caution
applies to anything else computed from a coordinate after a branch.

**Read the other Mesa backends before deriving an algorithm.** llvmpipe is in
the same tree. `lp_bld_sample.c` had a better answer for anisotropic filtering
than two hand-derived attempts, including the degenerate-case guards that are
the hard part: `lp_apply_ellipse_transform` rewrites the derivative pair into
the ellipse's own axes in closed form, and `lp_build_rho_aniso` divides the
level of detail by the unrounded clamped ratio rather than the rounded tap
count.

**Bisect in the application's shader rather than estimating error magnitudes.**
`instancing` was 8.56% because NIR lowered sin/cos to a cheap polynomial
(`.lower_sincos`). The error is harmless when a shader rotates a *direction* and
not when it rotates a *position*: those asteroids orbit at radius 7 with
vertices spanning 0.06, an 80x lever. Static reasoning dismissed it twice — an
approximation good to 1e-6 was "far too small to matter" and the real error was
nearer 1e-3. Editing the shader to identity rotations, then restoring each in
turn, then swapping the hardware sin/cos for a Taylor polynomial, settled it in
minutes. `fsin`/`fcos` now call the CUDA library versions, carried by the
sampler module which is already NVRTC-compiled and linked on demand.

**Symmetric statistics hide per-object errors.** The same sample's rocks were
declared un-displaced because the best *global* shift over a crop was (0.00,
0.00) — a measure dominated by the unmoved majority. Per-object centroids showed
a median of 0.76 px and a tail to 9. Measure the thing that is claimed to be
wrong, not an aggregate over it.

**A guard has to sit at every site, not the shared helper.** The alpha-test
reject check was added to `rasterize_pixel` and changed nothing, because all
three rasterizer stages have their own inlined pixel loops; there are four
`atomicMin` sites and only one went through the helper. Identical covered *and*
discarded counts on every pass was the symptom.

**Procedural hashes are a poor oracle.** `starfield.frag` builds stars from a
hash that multiplies by ~440 and takes `fract`, so any last-bit difference in an
interpolated varying relocates a star. Two correct implementations disagree
there; about a quarter of `instancing`'s remaining pixels are this. The same
applies to a checkerboard test texture, where aliasing makes correct renderers
disagree enormously — the offscreen bench uses a smooth gradient on purpose.

**Record negative results.** Precise `pow` and forcing base-level LOD were both
tried, measured, and reverted; without the note in the commit they would be
retried.

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

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --print-limit 2 \
    env VK_DRIVER_FILES=$ICD ./sample --offscreen -ofn out.ppm
```

## Implementation notes

* `.cu` kernels are stringified at build time by `kernels/cu_to_inc.py`.
* `struct cp_resource` embeds `struct llvmpipe_resource` first — lavapipe reads
  its fields at fixed offsets. Don't reorder.
* The visibility buffer stores the triangle index complemented (`~triID`) so
  `atomicMin` resolves coplanar triangles in primitive order.
* `pipe_screen::allocate_memory` returns `cuMemAllocManaged` memory with
  `cuCtxSetCurrent` first, because lavapipe allocates from its submit thread. A
  context may be current on several threads at once.
* **Every level of a resource sits at its own `mip_offsets[level]`.** Leaving it
  out of an address does not merely lose the small levels: `cp_resource_copy_region`
  omitted it, so a mip chain built by copying into successive levels had level 0
  written over and over and the rest untouched. That stayed invisible until
  `textureLod` started being honoured. The same class of bug is why uploads of
  level 1 once landed on level 0.
* NIR keeps a storage image's format on the intrinsic and hands the shader
  whatever type it asked for, so the driver owns both the texel stride and the
  packing. Taking the stride from the destination type walks the image at the
  wrong rate.
* Vertex attributes with fewer than four components read as `(0, 0, 0, 1)`. A
  shader taking a vec3 position as vec4 otherwise loses its transform's
  translation column — the geometry still draws and still writes depth, so it
  looks like a missing object rather than a broken one.
* A draw with no vertex buffer is legitimate: a vertex shader may build its
  positions from `gl_VertexIndex` alone, which is how a fullscreen pass is
  drawn. `load_vertex_id` reads the id array rather than recomputing it, so
  shaders carry `reads_vertex_id` and the array is materialised only for those.
* `bind_rasterizer_state` carries the cull mode. Drawing what should have been
  culled is not just wasted work — a back face can win the depth test and hide
  the surface in front of it.
* LLVM's NVPTX backend only knows architectures that existed when it was
  released, so `CP_MAX_PTX_SM` in `cp_nir_to_llvm.c` caps the architecture and
  lets the driver JIT forward. Raise it together with the PTX ISA version.
