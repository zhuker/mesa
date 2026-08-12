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

There is a second, animated sweep — sixty frames per sample, with the still
scenes orbited — which catches what a single frame cannot: two of the
regressions in the set are invisible at frame 0. `tests/TESTING.md` describes
both sweeps, every tool, and the traps.

**Editing a sample's shader is the fastest way to bisect a difference.** The
samples load `.spv` at runtime, so `glslangValidator -V shader.vert -o
shader.vert.spv` and re-running both drivers takes seconds and needs no C++
rebuild. Two bugs below were found that way after static reasoning got them
wrong; back the `.spv` up first and restore it afterwards.

## Status

Differing pixels versus the NVIDIA driver at tolerance 8/255, worst first.
Total across the set: 61,622 — under llvmpipe's 62,574, with no unimplemented
feature left in the set. From 83,638 before multisampling, 105,212 before
ordered blending, 110,015
before the watertight-coverage and cube-derivative work, 143,856 before the
quad-derivative work, and far more before that.

| Sample | Differing | Note |
|---|---|---|
| texturemipmapgen | 3.50% | anisotropic filter, accepted — see gap 4 |
| gltfscenerendering | 1.70% | anisotropic filter, accepted — see gap 4 |
| texturecubemap | 0.45% | reflection sharper than the reference at grazing angles |
| instancing | 0.38% | mostly the procedural starfield, see below |
| multithreading | 0.23% | speckle, no diagnosis, and the sample is nondeterministic |
| multisampling | 0.19% | |
| particlesystem | 0.17% | |
| texture | 589 px | anisotropic taps on a slightly tilted quad |
| pbribl | 28 px | |
| vulkanscene | 25 px | |
| dynamicuniformbuffer | 14 px | |
| bloom | 4 px | |
| texture3d | 2 px | |
| computeshader | 0 | |
| negativeviewportheight | 0 | |
| pushconstants | 0 | |
| renderheadless | 0 | the host readback path |
| triangle | 0 | |

Eleven of eighteen are within a handful of pixels and five match exactly, from
one at the start of this work. Everything still above 0.2% is the anisotropic
filter difference that gap 4 closes out.

Summed over the sixty frame animated sweep instead of frame 0, the set totals
4,259,670 against llvmpipe's 4,482,580 — the same conclusion the single frame
gives, and the sweep that has to be run to draw it, since a frame 0 of 0 says
nothing about the other fifty-nine. `computeshader` is 470 across the animation
where it was 608,619 before the viewport clipping went in, against llvmpipe's
9,505 over the same motion.

### Calibrate against llvmpipe, not against zero

The same sweep run through lavapipe/llvmpipe, which is the backend cudapipe's
sampler and clipper were ported from, and which shares lavapipe as its frontend:

| Sample | llvmpipe vs NVIDIA | cudapipe vs NVIDIA | cudapipe vs llvmpipe |
|---|---|---|---|
| texturemipmapgen | **21070** | 32237 | 17921 |
| gltfscenerendering | 26263 | **15697** | 5975 |
| texturecubemap | 5165 | **4144** | 3792 |
| instancing | 5829 | **3506** | 1869 |
| multithreading | **1954** | 2086 | 1834 |
| multisampling | **1454** | 1757 | 1023 |
| particlesystem | **327** | 1533 | 818 |
| texture | **426** | 589 | 320 |
| pbribl | 73 | **28** | 96 |
| vulkanscene | **11** | 25 | 20 |
| everything else | ~0 | ~0 | ~0 |
| **total** | 62574 | **61622** | 55008 |

Two things follow, and both change how the status table above should be read.

**A software rasterizer does not converge on NVIDIA.** llvmpipe is mature and
still differs by 62574 pixels over the set. Most of that is the same
anisotropic filtering difference cudapipe has: on texturemipmapgen llvmpipe's
mean error runs to +12.7/255 in the top luminance band against cudapipe's
+20.5, the same sign and shape, about 60% of the size. So there is real
headroom there, but the floor is llvmpipe's number, not zero.

**cudapipe is now ahead of llvmpipe over the whole set**, 61622 against 62574,
with nothing left unimplemented in it. On gltfscenerendering it is closer to
NVIDIA by a factor of 1.7, and that holds region by region across the surfaces
where the difference is most visible — the curtains, the pillars, the arches.
The numbers above are worth re-measuring rather than trusted to the last
hundred: six samples are nondeterministic (gap 9).

Worth re-running whenever a sampler or rasterizer change looks like it is not
paying off; llvmpipe is the honest target.

All 18 samples now exit cleanly on both. Four of them used to render correctly
and then segfault on the way out, which went unnoticed because only the images
were being compared — see the lessons.

    VK_ICD_FILENAMES=build-cudapipe/src/gallium/targets/lavapipe/lvp_devenv_icd.x86_64.json \
      VALIDATION=0 OUT=build/compare/llvmpipe ./run_offscreen.sh

    cp_gallery.py ref cuda llvmpipe -o three.html \
      --ref-label nvidia --test-label cudapipe --test2-label llvmpipe

cp_gallery.py takes the third directory and lays the two renderers beside the
same reference, with a panel of the two against each other that goes black
wherever they agree.

llvmpipe runs through the same lavapipe, so a change there moves both renderers
and can invalidate the comparison set without touching cudapipe at all. After
editing anything under `src/gallium/frontends/lavapipe`, re-render llvmpipe and
`cmp` it against the stored set before trusting a comparison — the sample count
limits below were exactly such a change, and llvmpipe came out bit-identical.

### What it costs

Sixty frames of each sample, the still scenes orbited, timed rather than stored
— milliseconds per frame, the mean over the sixty (`cp_perf_run.sh ... 60 1`):

| | nvidia | cudapipe | llvmpipe | before phase 1a | before any perf work |
|---|---|---|---|---|---|
| particlesystem | 0.1 | **50.1** | 15.7 | 55.2 | 1361.7 |
| multithreading | 0.3 | **29.7** | 96.9 | 50.2 | 156.6 |
| instancing | 0.1 | **27.1** | 73.0 | 27.2 | 241.7 |
| dynamicuniformbuffer | 0.0 | **17.4** | 1.0 | 22.5 | 128.5 |
| gltfscenerendering | 0.0 | **15.0** | 15.2 | 19.9 | 343.9 |
| bloom | 0.0 | **12.1** | 3.6 | 12.4 | 558.0 |
| pbribl | 0.0 | **1.7** | 2.7 | 3.8 | 135.4 |
| **total, one frame of each** | **0.9** | **168.1** | **232.1** | **206.7** | **3792.2** |

**cudapipe finishes the sweep 1.38x ahead of llvmpipe**, and ahead of it on ten
of the seventeen samples individually. It was 16x behind. Two passes got it
there and each has its own write-up:

- `PERFORMANCE_PROGRESS.md` — the first pass, 3792 -> 207 ms. 96% of it was
  three defects rather than any optimization, the largest being that
  `CP_SMALL_THRESHOLD` was set above the size of the framebuffer so two of the
  rasterizer's three stages had never executed.
- `PHASE_1A.md` — submission and synchronization, 207 -> 168 ms. The phase the
  plan had demoted, on a profile that had gone stale. The largest single win in
  it was not a phase 1a item but something the phase's measurement gate found:
  the device scratch arena reached 5.1 GB inside one frame, because it carried
  every draw at once and each draw's shading buffers are sized to twice the
  framebuffer rather than to what the draw covers.

NVIDIA's column is not a rendering time. Offscreen benchmarking measures
recording and submitting a frame, and nothing waits for the GPU until the pass
ends, so a driver that submits asynchronously is timed on its CPU side alone —
`tests/TESTING.md` says which column to compare across drivers and why. It is
the honest number for cudapipe, which blocks the host on every draw.

Things to take from it:

**cudapipe is single threaded on the host**, and this has not been worked on.
Over a storing run wall clock tracks CPU on every sample, so it is one core
blocking on the GPU. llvmpipe spreading across cores is why it stays close
despite shading on the CPU — and why the samples cudapipe now beats it on are
the ones with the most geometry.

**The peel loop is still the worst thing in the driver.** particlesystem runs
about 260 passes a frame, each re-rasterizing and re-shading the whole draw. The
host sync between them is now taken on a doubling interval rather than every
layer, which is worth 7%, but the passes themselves remain and it is 3.2x
slower than llvmpipe when nothing else in the set is. Phase 3 addresses it.

**`instancing` is the clearest lead in the set.** 39% GPU busy — more than half
its frame is the host — and it is the one sample the per-draw scratch rewind
did not help. Its profile looks like nothing else: 15 draws a frame,
`cp_vertex_fetch` at 57% of GPU time with a grid of 17,280 blocks and a median
launch of 2.4 ms.

**`dynamicuniformbuffer` is still an outlier**, 17x slower than llvmpipe and
launch-bound rather than kernel-bound: 625 twelve-triangle cubes a frame, nine
kernels each. Streams and the scratch rewind took it from 22.5 to 17.4 ms; the
rest is structural.

**cudapipe does not win pbribl, and never did** by the measure originally used.
It was once recorded as the one sample it beat llvmpipe on, 14.2 s against 16.0
— but those were whole process times, and llvmpipe spends 16.6 s of pbribl
before its first frame precomputing IBL textures. Per frame it is 2.6 against
3.8 ms. The lesson stands even though the totals have moved: a measure that
contains start-up will eventually be read as if it did not.

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
    │   4. cp_rasterize_stage1/2/3 (adaptive: thread, warp, block per tile),
    │      bounded by the clip rectangle: framebuffer ∩ viewport ∩ scissor
    │   5. cp_fs_interpolate    (compact into 2x2 quads, interpolate varyings)
    │   6. Fragment shader kernel (four threads per quad)
    │   7. cp_fs_writeback      (drop helpers, discard mask, blend, attachment)
    │   steps 4-7 repeat: up to CP_DISCARD_LAYERS times for an alpha-tested
    │   draw, up to CP_BLEND_LAYERS times for a blended one
    ├── blit: cp_resolve_samples when a multisample source meets a
    │   single-sample destination, a plain copy or format translate otherwise
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

**A point is one vertex wearing a square.** POINT_LIST arrives as one
degenerate triangle per point; `setup_triangle` turns that into a
screen-aligned square from `gl_PointSize` and stage 1 walks it with a half-open
box test, so stages 2 and 3 never see one. Every input takes the vertex's own
value, since there is nothing to interpolate between. `gl_PointCoord` is the
one thing that varies and no vertex shader output drives it, so the
interpolator writes it from the pixel's position inside the square; helper
lanes land outside [0, 1], which is what makes its derivative come out as
1/size. Both reach the kernels by varying location, `VARYING_SLOT_PSIZ` and
`VARYING_SLOT_PNTC`, so the shader compiler has no special case for either.

**A blended draw composites every layer, in submission order.** The visibility
buffer resolving one winner per pixel is what makes opaque overdraw cost a
single shade, and it is exactly wrong for transparency. With `blend_peel` set
the buffer keys on the primitive index instead of depth, so the same atomicMin
selects the lowest numbered primitive a pixel has not composited yet; the pass
blends it, `cp_peel_advance` steps that pixel past it, and the draw repeats.

llvmpipe does not need this because it never defers — it bins primitives per
tile and replays each tile's list in submission order, shading and blending
inline, so ordering falls out of the data structure (`tri_rasterize_bin` in
lp_rast.c walks the bin's command blocks in the order lp_setup_tri.c appended
them, and the blend is generated into the fragment shader itself by
`generate_unswizzled_blend`). Porting that shape directly would mean calling
the fragment shader from inside a tile kernel, and it is a separately compiled
module. Peeling reaches the same semantics: both do one shade per fragment per
pixel, llvmpipe serializing them within a tile and this serializing them across
passes while keeping every pixel parallel within one.

Passes are bounded by the primitive count and by `CP_BLEND_LAYERS`, and the
loop stops as soon as a pass selects nothing — the second pass, for the blended
draws that do not overlap themselves. particlesystem's fire is 512 additive
sprites piled tens deep and converges at 256 layers; 1024 gives a bit-identical
image.

**Multisampling resolves coverage and depth per sample and shades per pixel.**
The rasterizer tests each sample position in turn — the standard Vulkan
locations, with one sample being the pixel centre so 1x stays exactly what it
was — and resolves a winner per sample. The fragment shader still runs once per
pixel, and its colour goes to whichever samples that primitive won, which is
what per-fragment shading means. Visibility, depth and colour all hold the
samples plane after plane, so a sample is one multiply away and the
single-sample path indexes plane zero.

`cp_fs_interpolate` gathers distinct triangles across all of a block's samples
rather than its four pixels, since more of them can meet inside a block once it
holds sixteen or more sample points, and the coverage byte becomes a mask of
which samples each fragment won. `cp_resolve_samples` averages the planes, on
decoded values rather than packed bytes — an sRGB attachment has to average in
linear light or the resolve darkens exactly the edges multisampling exists to
smooth.

## Known gaps, roughly by how much they matter

1. **`texture3d` drifts over an animation.** 2 differing pixels at frame 0,
   3,090 at frame 24; llvmpipe is 0 across all sixty. Undiagnosed.

   It is not visible in the single frame sweep, which is what the animated one
   exists for — see `tests/TESTING.md`. It opens on its worst frame in the
   report.
2. **No line rasterization.** POINT_LIST and multisampling both work — see
   Architecture. Nothing else in the sample set is unimplemented.
3. **Sample shading is per fragment only.** `minSampleShading` and
   `sampleShadingEnable` are ignored, so a pipeline asking for per-sample
   shading gets per-pixel shading written to the covered samples. The
   multisampling sample builds such a pipeline but only binds it from the UI,
   which the offscreen runs do not touch.
4. **Anisotropic filtering under-blurs relative to NVIDIA's — accepted, closed.**
   Vulkan leaves the anisotropic filter implementation-defined, llvmpipe differs
   from NVIDIA in the same direction, and cudapipe is closer to NVIDIA than
   llvmpipe on the samples where it is most visible. Recorded here as a
   difference in NVIDIA's filter rather than a cudapipe defect, and not pursued
   further. Do not reopen it without a reason better than the pixel count; what
   follows is what is known, so it does not have to be rediscovered.

   This is nearly all of what is left in `texturemipmapgen`, and it also drives
   most of `gltfscenerendering`. Forcing each of that sample's three sampler
   modes and rendering both drivers separates it cleanly: no mipmaps differs by
   19 pixels, mipmaps with bilinear by 1095, mipmaps with anisotropy by 32237.
   So the texture upload, the runtime-generated mip chain and LOD selection are
   all effectively exact, and the specular `pow` is too — sampler 0 runs the
   same `pow(dot(R, V), 16)` and still lands within 19 pixels.

   The signature is contrast, not brightness: cudapipe is darker than the
   reference where the reference is dark and brighter where it is bright, by up
   to 20/255 in the top luminance band. Turning anisotropy off entirely
   overshoots the other way.

   `gltfscenerendering` is the same thing amplified. Its fragment shader samples
   a normal map and raises the result to the 32nd power, so an under-blurred
   normal map becomes scattered specular glints on exactly the grazing-angle
   surfaces where anisotropy applies — the side walls, the pillars and arches,
   the curtain folds. Disabling anisotropy cuts the differences on the stone
   pillars and arch from 2128 to 1233 and on the red curtain from 1723 to 757,
   while making the frame as a whole worse (15697 to 29726), which is what tells
   you the filter is under-blurring rather than simply wrong. The remaining
   large per-pixel differences are paired: one pixel much brighter in cudapipe
   and its neighbour much darker, on thin high-contrast features. There is no
   global subpixel shift — a +-1 pixel search puts the minimum at (0, 0) by a
   factor of fifteen.

   Two tap-placement schemes have been measured. The one in the tree spreads N
   taps over `rho_max * (N-1)/N`, which tiles the footprint exactly when
   `N == eta`. Switching to llvmpipe's — step along the raw ddx or ddy rather
   than a unit vector rescaled to the transformed axis, and space taps by the
   Vulkan specification's `(t - N/2 + 1/2)/(N + 1)` — improves `texture`
   (589 -> 164) and regresses everything else: `texturemipmapgen` 32237 -> 38864,
   `gltfscenerendering` 15697 -> 25036, `instancing` 3506 -> 5346,
   `texturecubemap` 4144 -> 4774. Do not re-apply it wholesale; the two halves of
   it have not been measured separately.
5. **A blended draw costs one `cuStreamSynchronize` per layer.** The host reads
   a managed flag between passes to decide whether another is worth launching.
   That is the first thing to attack if blended draws ever dominate a frame; the
   flag could instead drive a device-side loop or a launch graph.
6. **Peeling does not combine with the alpha-test retry loop.** Both want the
   same multi-pass machinery for different reasons, so a shader that discards
   keeps the retry path and gets the old single-layer blending. No sample in the
   set needs both at once.
7. **Alpha-tested geometry costs CP_DISCARD_LAYERS passes over the draw.**
   Visibility resolves before shading, so a fragment that discards has already
   displaced the one behind it; each pass records what discarded where and
   repeats so the next fragment can win. Sponza's foliage falls from 543
   discards to 1 within four passes; halving the layers from 8 to 4 costs it
   0.14% of the frame, and anything still discarding after the last layer is
   lost.
8. **BC1/BC3 decode is written but never exercised** — no upstream sample uses
   compressed textures, and the capture has 576 BC images.
9. **`multithreading` (0.23%) has no diagnosis**, and the sample is
   nondeterministic: two runs of the same build differ, because thread
   scheduling changes the order its command buffers are recorded. Do not read
   its last few hundred pixels as signal.

   Five others vary run to run as well, measured by rendering the same build
   twice: `gltfscenerendering` and `instancing` for the same reason as
   `multithreading`, and `multisampling`, `vulkanscene` and `particlesystem`
   for one not yet established — over sixty frames they move by 36, 1 and 0
   pixels at tolerance 8, `particlesystem` differing byte-wise on 13 frames
   without any pixel crossing the tolerance. Small enough not to matter for a
   verdict, large enough to be mistaken for the effect of a change: the
   viewport-clip fix below appeared to move three of them by a handful of
   pixels until the same build was run against itself.
10. **`bindless_image_store` has no bounds check.** A latent memory-safety hole
   rather than a visible bug — the one sample that stores dispatches
   `width / 16`, so it never addresses out of range. The load path clamps the
   address and selects the value back to zero; the store should drop instead.
11. `nir_op_fexp2`, `flog2` and lowered `fpow` still use the NVVM `.approx`
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

**The near plane is not the only plane.** `cp_clip_triangles` cuts w <= 0 and
nothing else, on the usual reasoning that x and y need no geometric clipping
because the rasterizer walks a bounding box anyway — true only if that box is
bounded by the viewport, and it was bounded by the framebuffer. Nothing showed
it for as long as every sample drew one viewport covering the whole
framebuffer. `computeshader` draws two side by side, so the moment its geometry
ran past NDC +-1 the right pane's quad appeared 40 columns into the left pane:
0 differing pixels at frame 0 and 28,805 by frame 5 of an orbit, which had been
sitting at the top of the gap list undiagnosed. Clipping in x and y is now the
clip rectangle the stages are bounded by — framebuffer, viewport and scissor
intersected on the host, in llvmpipe's half-pixel convention
(`lp_setup_set_viewports`). Stage 3 needs the test explicitly: it walks whole
tiles, so its edges run past the box it was enumerated from.

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

**Rebuild the sample after editing it, and check the baseline still
reproduces.** Forcing `texturemipmapgen` through each of its three sampler
modes meant editing its `.cpp`; the source was restored afterwards but the
binary was not rebuilt, so every run for the next hour was a bilinear render
being compared against an anisotropic reference. It produced a confident and
completely wrong conclusion — that llvmpipe's tap scheme "overshoots into
blurrier" — which only came apart because two unrelated experiments returned
byte-identical images. Re-render the baseline and `cmp` it against the last
known-good frame before trusting any number that follows a change outside the
driver.

**Verify a refactor is bit-identical before layering behaviour on it.**
Ordered blending needed its selection test in one place, so all three
rasterizer stages were first routed through a single `emit_fragment()` with no
intended change in behaviour, built, and swept: identical on every
deterministic sample. Only then did the peel logic go in. When something broke
afterwards there was no question about which half to look at. The alternative
had already been demonstrated twice this project — a combined change whose two
halves each looked plausible and whose failure implicated neither.

**Not every sample is a deterministic oracle.** `multithreading` renders
differently run to run on the same build, because thread scheduling changes the
order its command buffers are recorded. That was quietly polluting a few
hundred pixels of every comparison until two identical runs were diffed against
each other. Before attributing a small delta to a change, check the sample
against itself.

**A bump allocator that is only reset between draws will eat the machine.**
Scratch is handed out per draw and reclaimed when the arena has expanded a few
times, which works until something allocates inside a loop. The ordered
blending pass loop called the shading stage once per pass without rewinding, so
a 256 layer draw asked for a couple of hundred megabytes of buffers 256 times
over. It is `cuMemAllocManaged`, so it is backed by system RAM and does not
fail — it invoked the OOM killer and took the desktop down. Counting expansions
did not catch it either: once the arena is big enough that nothing has to grow,
that counter stops moving and the pointer climbs for the rest of the frame, and
a frame of bloom reached eleven gigabytes. Reclaim now triggers on bytes handed
out as well, the pass loop rewinds, and a hard cap turns any future variant into
a black frame and a message. Watch `nvidia-smi` and `free` when running
something new for the first time; the trace of that run showed the GPU pinned at
32,072 MiB of 32,607 for five minutes before the kill.

**Reusing a buffer exposes what it relied on being zero.** Rewinding the arena
between passes broke four samples, because the pixel counter and the discard
mask were read where they had not been written and had been getting a fresh
`cuMemAllocManaged`, which happens to be zeroed. Anything reused has to be
initialised explicitly.

**Read the exit codes, not just the images.** Four samples rendered correctly
and then took SIGSEGV during teardown, and had been doing so for weeks:
`pipe_resource_release()` dispatches through the *context*, and cudapipe set the
screen's `resource_destroy` but never `pipe->resource_release`, so the call went
to address zero. One line, and llvmpipe had the same line. It survived that long
because the sweep's images were being compared while `exit=139` sat unread in
`_summary.txt` — and then it was misdiagnosed as a 60-frame regression, when it
reproduced at one frame in under a second. `gdb` gave the answer in a single
run after a long time spent speculating.

**A capability you advertise but clamp is worse than one you refuse.** lavapipe
named a fixed set of sample counts regardless of the driver under it, so the
multisampling sample asked for the largest, got eight, and cudapipe quietly
rasterized four samples into an eight sample attachment. Nothing errored: the
resolve simply averaged four samples it had written with four it never touched,
and the result was a plausible, slightly wrong image. The limits now come from
the screen's `is_format_supported`. When adding a feature that has a range,
make the advertised range the implemented one, and check what the application
actually ends up asking for rather than what you had in mind for it.

**A second allocation path is a second place to get the size right.**
`resource_create_unbacked` computed its layout but never multiplied by
`nr_samples`, so a multisample attachment created that way reported one
sample's worth of memory. It was invisible for as long as nothing wrote to the
later planes, and would have become memory corruption the moment multisampling
started working. When a resource property scales the allocation, grep for every
site that returns a size.

**A guard has to sit at every site, not the shared helper.** The alpha-test
reject check was added to `rasterize_pixel` and changed nothing, because all
three rasterizer stages have their own inlined pixel loops; there are four
`atomicMin` sites and only one went through the helper. Identical covered *and*
discarded counts on every pass was the symptom. The stages now all route
through `emit_fragment()`, so there is one site again — keep it that way.

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
* A multisample resource holds its samples plane after plane, `lpr.sample_stride`
  apart, and the same layout is used for the driver's own visibility and depth
  buffers as `sample * width * height + pixel`. Both resource creation paths
  have to multiply the allocation by `nr_samples`.
* LLVM's NVPTX backend only knows architectures that existed when it was
  released, so `CP_MAX_PTX_SM` in `cp_nir_to_llvm.c` caps the architecture and
  lets the driver JIT forward. Raise it together with the PTX ISA version.
