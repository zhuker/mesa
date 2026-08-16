# cudapipe — CUDA Software Rasterizer for Roblox

## What This Is

A Mesa Vulkan ICD (`libvulkan_cudapipe.so`) that rasterizes on NVIDIA GPUs using
CUDA compute kernels instead of fixed-function hardware. It reuses lavapipe as
the Vulkan frontend and replaces the Gallium driver underneath.

The feature set is driven by what Roblox's HeadlessStreamer actually requires,
measured from a GFXReconstruct capture — see
`src/gallium/drivers/cudapipe/tests/headless_streamer_requirements.txt`.

That capture now *replays* against the driver, all 1,509 frames of it, which is
a much stronger thing than reading its requirements. Three documents carry that
work and are the place to start on it:

- `src/gallium/drivers/cudapipe/HEADLESS_STREAMER_PERF.md` — the capture, the
  six bugs that stopped it replaying, what each fix was worth, how frame
  correctness is verified against a software reference, and where the frame goes
  now. **Read the correctness method before trusting any frame comparison.**
- `src/gallium/drivers/cudapipe/TODO_CONFORMANCE.md` — what dEQP says, fixed and
  outstanding. Second priority.
- `src/gallium/drivers/cudapipe/TODO_DEBUG_FLAGS.md` — the driver reads 44
  environment variables and there is no list of them; this is the inventory and
  a plan for a registry. Third priority, but it contains two silent-`undef`
  paths in the shader backend that are worth fixing on their own.

`tests/GFXRECONSTRUCT.md` is how to capture and replay in the first place,
including why an application that never presents needs a different extraction
path entirely.

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
stronger signal than dEQP's pass/fail.

`tests/headless_streamer_samples.txt` is the minimal set — each entry is the
simplest sample covering a capability the capture needs, with the mapping in the
file.

**It is a regression guard, not evidence, and this sentence used to claim
otherwise.** It said the sweep "found every bug fixed so far", which stopped
being true the moment a real application was replayed against the driver. Six
bugs came out of that capture in a week and **four of them are structurally
unreachable by this set** — not unlucky, unreachable:

| bug | why no sample can find it |
|---|---|
| packed vertex attributes never expanded | exactly one sample binds an 8-bit vertex attribute (`imgui`, UNORM colour), and `gltfskinning` converts glTF's packed joints to float on load |
| clipper missing the `z <= w` plane | `VK_COMPARE_OP_GREATER` appears nowhere in the tree — nothing uses reversed-Z |
| A-buffer disabling itself on resize | a sample that renders one size never triggers it |
| fragment shaders skipped with no colour attachment | only `oit/geometry.frag` writes memory from a fragment shader |

Five separate times this session a green sweep meant "did not touch it" rather
than "works". So for each change, find the thing that actually exercises it —
`tests/TODO_CONFORMANCE.md` ends with the lookup table, and
`HEADLESS_STREAMER_PERF.md` has the same one. Note also that `renderheadless`'s
row in the correctness gate is two absences agreeing: its stored directory is
empty, so is the reference, and a diff of one against the other reports
IDENTICAL.

What the set *is* good at is bit-identity. Ten of the eighteen are fully
deterministic, so "byte-identical over sixty frames" is a far stronger statement
than any tolerance check, and it is what caught both real regressions this
branch produced.

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
regressions in the set are invisible at frame 0. `tests/TESTING.md` is the methodology for both
halves of this — how correctness is checked, how cost is measured reliably
enough to compare, and how to find where the time actually goes. Read it before
trusting a number from either.

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

Six hundred frames of each sample, the still scenes orbited, timed rather than
stored — milliseconds per frame, the mean over the run (`cp_perf_run.sh ... 600
1`, which is what `cp_iterate.sh` runs). Sixty frames is what gets *stored* and
compared, and is far too few to time: at sixty the render loop is a minority of
the process. `tests/TESTING.md` has the measurement that settles it.

| | nvidia | cudapipe | llvmpipe | before the capture work | before the register cap | before any perf work |
|---|---|---|---|---|---|---|
| gltfscenerendering | 0.1 | **14.3** | 15.2 | 13.9 | 15.3 | 343.9 |
| instancing | 0.1 | **6.1** | 73.0 | 6.7 | 7.0 | 241.7 |
| multithreading | 0.3 | **5.5** | 96.9 | 6.0 | 6.0 | 156.6 |
| particlesystem | 0.1 | **3.5** | 15.7 | 4.1 | 4.5 | 1361.7 |
| multisampling | 0.0 | **2.6** | 5.3 | 3.3 | 3.4 | 41.1 |
| bloom | 0.0 | **1.4** | 3.6 | 2.2 | 2.2 | 558.0 |
| pbribl | 0.0 | **1.1** | 2.7 | 1.6 | 1.6 | 135.4 |
| vulkanscene | 0.0 | **0.9** | 9.4 | 1.5 | 1.5 | 183.1 |
| dynamicuniformbuffer | 0.0 | **0.3** | 1.0 | 1.3 | 1.3 | 128.5 |
| **total, one frame of each** | **0.9** | **38.3** | **232.1** | **47.6** | **50.0** | **3792.2** |

The ninth pass is the one that produced the fourth column, and it did not set
out to touch the sweep at all — see `HEADLESS_STREAMER_PERF.md`. Making a real
application's capture replay found allocation churn that these samples are
host-bound on: `triangle` −84%, `texturemipmapgen` −45%, `bloom` −34%,
`dynamicuniformbuffer` −78% from a small-allocation arena and grow-only
framebuffer buffers. The capture itself went 276 → 82 ms a frame over the same
work. `gltfscenerendering` reads +3% here and measured at parity under paired
alternating arms; it spreads 6.9% across three runs of one build, so read it
that way rather than as a regression.

**cudapipe finishes the sweep 4.88x ahead of llvmpipe**, ahead of it on twelve
of the seventeen samples, and **every sample above 1.5 ms is ahead of it** —
`gltfscenerendering` included, as of the eighth pass. What remains behind are
five sub-1.3 ms samples like `triangle` and `texture3d`, which measure the
driver's per-frame floor rather than anything about drawing. It was 16x behind.
Eight passes got it there; five have their own write-up:

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
- `INSTANCING.md` — 168 -> 150 ms, almost all of it one sample. `instancing`
  was 38% GPU busy because one draw of 8,192 instances took a host path that
  built a 4.4 million entry vertex/instance id table — twice, into managed
  memory — for values that are arithmetic on the thread index. The fetch kernel
  derives them now. 27.1 -> 7.1 ms, and the profile had been naming
  `cp_vertex_fetch`, which was not the problem: a kernel touching a managed page
  the host just wrote is charged for the migration.
- `BATCHING.md` — 145 -> 109 ms. Consecutive draws that cannot depend on their
  order are merged and rasterized as one, which needs none of Phase 3 because
  with blending off the visibility buffer resolves by `atomicMin` and a minimum
  is order-independent. `dynamicuniformbuffer` 12.3 -> 1.3, `multithreading`
  29.8 -> 6.0, `pushconstants` 2.3 -> 0.8.

Plus one step that belongs to no pass, 150 -> 145 ms: `cuMemsetD8` is the
blocking variant, and `dynamicuniformbuffer` creates 764 transient 64-byte
resources a frame, each cleared with one at 31.5 µs of device time. That put it
ahead of `cuLaunchKernel` in host API time to move 64 bytes. Small managed
allocations are zeroed by the host now; past 64 KB the clear becomes real
bandwidth and the device keeps it.

The fifth pass has no write-up of its own; it is one commit, "stop the peel loop
paying for work it already did", 109 -> 92 ms and particlesystem 52.0 -> 35.1.
Three changes to the peel loop: the binning queues are built once per draw
rather than rebuilt on all 256 passes; points take their own medium threshold,
so a sprite between 128 and 4096 pixels stops getting one warp however much it
covers; and stage 3 walks only the part of a tile the primitive's bounding box
reaches. The second is nearly all of it, at -27.6%. Two of the three predictions
behind them were wrong in instructive ways.

- `ABUFFER.md` — the sixth pass, 92 -> 62 ms, and particlesystem 35.1 -> 5.1.
  The peel loop is gone for eligible blended draws: one rasterization into
  per-pixel fragment lists sorted by submission index, shaded once, composited
  once. `emit_fragment()` was being called about 960 million times a frame to
  composite 2.28M fragments — 99.76% of the calls rejected for being below their
  pixel's `peel_next` — and that ratio is the whole case. Verified stepwise
  against the peel loop, and the checks are still runnable.
- `BATCHING.md`, second half — the seventh pass, 62 -> 50 ms. Draws that replay
  *different* index ranges now merge: `cp_vertex_fetch` binary-searches a
  per-draw slice table and the range leaves the batch key. `bloom` 12.1 -> 2.2,
  151 batches a frame to 2; `vulkanscene` -45.6% and `particlesystem` -10.5%,
  neither predicted. The per-triangle draw index that pass expected to need
  turned out not to be needed at all — `bloom`'s draws differ in nothing but the
  range.

The eighth pass has no write-up either; it is one commit, "pick each shader's
register cap by trying both", 50 -> 47.6 ms. `ncu` had Sponza's fragment shader
at 195 registers and one 256-thread block per SM, latency-bound with no
spilling. The plan proposed capping only the shaders whose register count costs
occupancy — but **every fragment shader that links the sampler is exactly 195
registers, in all 17 samples**, because that is the *sampler's* allocation and
not the shader's, so the count cannot discriminate. `__launch_bounds__` is worse:
`.minnctapersm 2` makes every sampler-linked shader fail to load. What
discriminates is whether capping buys an occupancy step, which is not
predictable from the count — so the driver compiles both builds and times them
on the application's own draws, keeping the capped one unless it is more than 5%
worse. texturemipmapgen -13.3%, particlesystem -10.0%, gltfscenerendering -8.1%.

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

**The peel loop is no longer the default, and `gltfscenerendering` is now the
largest sample in the set.** particlesystem was 3.3x slower than llvmpipe and is
3.5x *faster* now; `bloom` was the largest remaining gap and the seventh pass
took it 12.1 → 2.2. `gltfscenerendering` at 15.3 is untouched by all seven, and
is the one sample that cannot batch for a structural reason — it creates one
pipeline per material and binds 25 distinct vertex shaders a frame, so it never
reaches the batch key at all.

The peel loop is still there, still correct, and still what runs for any blended
draw the A-buffer refuses — multisample, depth-writing, more than one colour
attachment. `CUDAPIPE_NO_ABUFFER=1` puts everything back on it. Its structure is
worth understanding before touching the blend path: about 260 passes a frame,
each re-rasterizing and re-shading the whole draw, with `emit_fragment()` called
960 million times to composite 2.28M fragments.

Two findings from the passes that got it there, both still live:

- **Stage 3's cost is the `atomicMin` per covered sample, not the coverage test
  on rejected ones.** Bounding the tile walk cut its trip count by the predicted
  factor and its duration by 11%. Stage 3 takes one block per (primitive, tile)
  pair, so a tile in the fire's core had hundreds of blocks contending on the
  same few thousand addresses, 260 times over with identical values. The
  A-buffer removes the repetition rather than the contention; whether Phase
  3's one-block-per-tile form would beat it is open, and now has a baseline.
- **Half the passes cannot be seen.** Rendering at 128 layers instead of 256
  leaves red and green bit-identical across the whole frame and moves 641 pixels
  of blue. The fire's core saturates long before the cap. That is not a licence
  to lower `CP_BLEND_LAYERS` — the change is visible in the brightest pixels, a
  tolerance-8 count over the frame reads it as 0.06% and hides it, and the value
  would be tuned to this scene. It says the work is there to be removed by
  something that knows when a pixel can no longer change.

  An exact early-out is harder than it looks here: the blend is
  `src + dst * (1 - a)`, so a saturated pixel can go back *down* and saturation
  is not an absorbing state. Only pure additive blending gives that, and this
  pipeline is premultiplied `over` — it merely behaves additively because the
  texture's alpha is near zero, which is data the driver cannot see.

**"94% GPU busy" never meant the GPU was working.** Device counters have
particlesystem at 94% GR Active while issuing instructions on 5% of cycles,
with a third of its SMs active and DRAM flat. That is what a long sequence of
small passes looks like from outside, not an expensive kernel. Every
"kernel-bound" verdict in these documents means only that a kernel was
resident; `tests/TESTING.md` question 2 is how to ask the other half.

**`dynamicuniformbuffer` is no longer an outlier, and the reason it looked like
one was wrong.** It was recorded here as launch-bound at 625 twelve-triangle
cubes a frame and nine kernels each. `OBJECT_INSTANCES` is 125, and the profile
behind those figures divided by the frame count handed to `cp_profile.sh` while
the trace also held the warm-up frames nsys records and nobody asked for — so
every per-frame number in it was 5x too large, and the sample was not
launch-bound. Batching then took it 12.3 -> 1.3 ms, mostly by deleting
*multiplicity* rather than by growing grids: the framebuffer-sized costs were
being paid 125 times for one frame's output. It is now host-bound at 25% GPU
busy, in lavapipe's per-draw descriptor allocation — about 125
`cuMemAllocManaged`/`cuMemFree` pairs a frame through
`pipe_screen::allocate_memory`.

**`pbribl` is a lesson about the measure rather than a win.** It was once
recorded as the one sample cudapipe beat llvmpipe on, 14.2 s against 16.0 — but
those were whole process times, and llvmpipe spends 16.6 s of pbribl before its
first frame precomputing IBL textures. Per frame at the time it was 3.8 against
2.7, the other way round. It is 1.6 against 2.7 now, so cudapipe does win it —
which is why the original claim is worth keeping rather than deleting: it was
right by accident, for a reason that had nothing to do with rendering. A measure
that contains start-up will eventually be read as if it did not.

## Architecture

```
Vulkan app
    ↓
lavapipe frontend (reused as-is)
    ↓ pipe_context calls (on submit thread)
cudapipe Gallium driver
    ├── State changes write into persistent cp_gpu_state (managed memory)
    ├── draw_vbo: batches consecutive draws that cannot depend on their
    │   order (blending off, so the visibility buffer's atomicMin resolve is
    │   order-independent) and runs the pipeline below once for the batch.
    │   See BATCHING.md. A batch of one is the unbatched path, bit-identical.
    ├── the pipeline, per batch:
    │   1. cp_vertex_fetch      (GPU gathers attributes)
    │   2. Vertex shader kernel (NIR → PTX)
    │   3. cp_clip_triangles    (z >= 0, w > 0, and z <= w — the last is
    │      the near plane under a reversed-Z projection, which the capture uses)
    │   4. cp_rasterize_stage1/2/3 (adaptive: thread, warp, block per tile),
    │      bounded by the clip rectangle: framebuffer ∩ viewport ∩ scissor
    │   5. cp_fs_interpolate    (compact into 2x2 quads, interpolate varyings)
    │   6. Fragment shader kernel (four threads per quad)
    │   7. cp_fs_writeback      (drop helpers, discard mask, blend, attachment)
    │   steps 4-7 repeat up to CP_DISCARD_LAYERS times for an alpha-tested draw.
    │   A blended draw takes the A-buffer instead — rasterized once into
    │   per-pixel fragment lists, sorted, shaded and composited once; see
    │   ABUFFER.md. Only the draws it refuses (multisample, depth-writing, more
    │   than one attachment) fall back to CP_BLEND_LAYERS passes of 4-7.
    │   Consecutive blended draws now merge into one A-buffer episode, which
    │   needed per-draw fragment bindings and a stable clipper; see BATCHING.md.
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
screen-aligned square from `gl_PointSize`, and every stage resolves it with the
same half-open box test rather than with edge functions. A point under
`CP_SMALL_THRESHOLD` is walked by one thread in stage 1; above it, it goes
straight to stage 3 and gets a block per tile, because `CP_POINT_THRESHOLD`
defaults to the small threshold — a sprite is all bounding box, so the one-warp
middle path stage 2 offers a triangle is the worst of both for it. Raising
`CUDAPIPE_POINT_THRESHOLD` puts the middle path back. Every input takes the vertex's own
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
draws that do not overlap themselves.

**This is no longer the default path — see `ABUFFER.md`.** An eligible blended
draw is rasterized once into per-pixel fragment lists sorted by submission
index, shaded once and composited once; `CUDAPIPE_NO_ABUFFER=1` puts it back on
the loop above, and draws the A-buffer refuses still take it.

Two things this paragraph used to say are wrong, both measured while building
that. **particlesystem's fire is not additive**: the pipeline is
`src = ONE, dst = ONE_MINUS_SRC_ALPHA, op = ADD`, premultiplied *over*, which is
order-dependent. It behaves additively only where the texture's alpha is near
zero, and that is data the driver cannot see. **And it does not converge at 256
layers — it hits the cap.** A census counting at `emit_fragment()` puts the true
maximum depth at 411, with 4,040 pixels deeper than `CP_BLEND_LAYERS` and
**13.6% of all coverage never composited**. That 1024 gives a bit-identical
image is still true and is now explained: the core saturates long before either
bound, so the dropped layers are invisible rather than absent.

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
5. **Largely closed twice over.** The peel loop's host sync per layer became a
   doubling interval in Phase 1a, and the loop itself is no longer the default —
   an eligible blended draw takes the A-buffer and syncs once, to read the
   fragment count. What is left applies only to draws the A-buffer refuses.
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
8. **BC1/BC3 decode is exercised by the capture now**, and the first thing it
   found was not in the decoder: `cp_resource_copy_region` measured copies in
   pixels while multiplying by the *block* size, so a BC1 `vkCmdCopyImage`
   walked 16x its extent and ran off the allocation. Fixed, along with the
   identical bug in `cp_blit`. No upstream sample uses compressed textures, so
   the capture remains the only coverage.
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
12. **Stage 2 and stage 3 do not interpolate depth to the same bits.** Both
   evaluate the same edge functions with the same fill rule, but the
   barycentric-to-depth arithmetic is written out separately in each, and NVRTC
   is free to contract the products into FMAs differently. The visibility
   buffer resolves by `atomicMin`, so a one-ulp difference flips a tie and a
   different triangle wins the pixel. Moving triangles across the stage 2/3
   boundary — by lowering `CUDAPIPE_MEDIUM_THRESHOLD` — changes two pixels of
   `gltfscenerendering` for this reason. It is why `CP_POINT_THRESHOLD` is a
   separate constant rather than the medium threshold simply being lowered, and
   any change to the size thresholds or to Phase 3's tiling will keep meeting
   it. The fix is one interpolation routine both stages call, the way
   `emit_fragment()` already unified the four `atomicMin` sites. Points are
   unaffected: they carry a single depth and both stages pass it through.
13. **`VK_BLEND_FACTOR_CONSTANT_COLOR` and `CONSTANT_ALPHA` are enumerated and
   unimplemented.** `cp_blend_factor` falls through and returns 1.0 for both,
   silently, on the A-buffer path and the peel path alike. Nothing in the sample
   set uses them. It predates the A-buffer and is more exposed now that path is
   the default.
14. **The peel loop is nondeterministic under blend equations that are nonlinear
   in the destination.** Two runs of one build differ by thousands of pixels
   under `SUBTRACT`, `MAX`, `ONE`/`ONE_MINUS_DST_COLOR` and
   `DST_ALPHA`/`ONE_MINUS_SRC_COLOR` — measured on the driver before the
   A-buffer existed, so it is not that path's doing. The fragment population is
   bit-identical run to run, so the variable is per-fragment shaded colour:
   particlesystem's documented one-LSB instability, amplified by the equation's
   conditioning. Under `MAX`, which is order-independent and idempotent, the
   A-buffer is bit-exact across twenty frames while peeling wanders by 1,496
   pixels. No sample in the set uses these equations, which is why it went
   unnoticed; `ABUFFER.md` has the matrix.
15. **`cp_clip_triangles` compacts its output with `atomicAdd`, so post-clip
   primitive order is not stable run to run — half fixed.** A stable mode now
   exists and is used for blended batches: input triangle *t* owns output slots
   *4t..4t+3*, and the slots it does not fill are retired as zero-area triangles
   that `setup_triangle` already rejects. It costs nothing (stage 1 was always
   sized for four times the input) and it is what makes blending exactly
   submission-ordered — the driver is now bit-identical against itself on the
   capture, where it used to differ by ~0.3% of pixels. The compacting path
   still runs for everything else, so the paragraph below stands for unblended
   draws. Vulkan defines rasterization
   order by primitive order, so this breaks the guarantee independently of
   anything else — it was simply invisible while batches were small and ties
   were rare. Merging draws that replay different index ranges made it
   reachable: `bloom` was bit-identical against itself over 60 frames and now
   differs on 4 of them at 1–2 pixels, `vulkanscene` 1/60 → 5/60, appearing at
   `CUDAPIPE_BATCH_MAX=8` and not at 2. A stable compaction is the fix and it is
   worth its own pass; narrowing what merges would only hide it again. Gap 16
   is the same defect seen from the other side.
16. **Batched draws break a depth tie the other way.** With `LEQUAL` and
   coplanar geometry spanning two merged draws, the batch keeps the lowest
   triangle index — the earliest draw — where drawing them in sequence keeps the
   later one. No sample in the set shows it, and it is mitigated rather than
   absent: the clipper compacts its output with `atomicAdd`, so triangle order
   is already nondeterministic *within* a draw and the tie was never exact. It
   is still a real difference in what the driver computes, and the first place
   to look if a depth-fighting artefact appears. `CUDAPIPE_NO_BATCH=1`
   distinguishes it from anything else in one run.

## Lessons that cost the most to learn

**An unimplemented operation that returns zero destroys everything downstream.**
`textureSize` reading as zero made a blur kernel compute `1/0` for its tap
offsets and sample at infinity, blackening a whole frame; `terminate_if`
returning undef collapsed shaders; `gl_InstanceIndex` silently did nothing for a
while. Treat a zero from a missing feature as a fault, not a default.

`gl_FrontFacing` is the most expensive instance so far, and it is the one that
changed how this is reported. `nir_intrinsic_load_front_face` had no case in the
backend, so it became `undef`, so a surface shader's tangent frame was flipped
by an undefined sign, so every lit surface in a real application lost its direct
lighting. The frames looked plausible — dim, faintly green — and finding it took
bisecting one frame to a single draw and dumping every descriptor of it from two
drivers. `CUDAPIPE_DEBUG_SHADER=1` had been printing
`unhandled intrinsic 'load_front_face'` the whole time, behind a variable nobody
sets until they already suspect the shader.

**So that warning is unconditional now**, once per intrinsic name. It cost
nothing and it is compile time, not draw time. The very next silent-render bug
after it took four commands to diagnose, mostly because the *absence* of that
warning eliminated an entire class of cause immediately. Two siblings are still
gated on the env var and should not be — `TODO_DEBUG_FLAGS.md` names them.

**A CUDA fault is sticky and surfaces at the wrong place.** An `imageLoad` one
texel outside its image killed a whole frame: the context faulted, the next
`cuMemAlloc` failed, and draws then bailed out for want of a visibility buffer
several stages away from the cause. `compute-sanitizer` turns this back into a
kernel name and a line.

It reappeared in a worse form and is worth knowing in that form too. A vertex
shader indexing an array with an unexpanded packed attribute read about two
gigabytes out of bounds; the context died with `CUDA_ERROR_ILLEGAL_ADDRESS`;
`cp_allocate_memory` then returned NULL, which nothing checked; and lavapipe
took that NULL straight to `memset()`. The visible failure was a segfault inside
libc **with no cudapipe frame in the backtrace at all**. Under
`CUDA_LAUNCH_BLOCKING=1` the first two errors the driver reported were a vertex
shader launch and `fs_interpolate` — both merely the first launches *after* the
fault, neither the cause.

Two things follow. The driver reports failures now (`CP_CU_WARN`, `CP_LAUNCH`),
and for the sticky errors it says so explicitly and names the tools, because
"the first report is not the cause" is the part that costs a day. And
`compute-sanitizer` remains the thing that actually answers it: about sixty
seconds on a capture that faults in five.

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
| `CUDAPIPE_DEBUG_BATCH` | why each draw batch ended |
| `CUDAPIPE_DEBUG_BATCHDIFF` | which field of the batch key differed — "state or geometry" is one `memcmp` and this is what names it |
| `CUDAPIPE_NO_BATCH` | disable draw batching; reproduces the unbatched frame byte for byte |
| `CUDAPIPE_BATCH_MAX` | cap the batch size; `1` is the bit-identical check |
| `CUDAPIPE_DEBUG_WORK` | shaded pixels against threads launched, per shading pass (syncs) |
| `CUDAPIPE_NVTX` | NVTX timeline ranges per draw and stage; read with `tests/cp_prof_nvtx.py` |
| `CUDAPIPE_NO_ABUFFER` | back to the peel loop for blended draws — see `ABUFFER.md`. The A-buffer is the default |
| `CUDAPIPE_ABUFFER_COMPOSITE=0` / `_VERIFY=1` / `_LAYERS=N` / `_TIMING=1` | build the lists beside the peel loop; the stepwise checks; cap the composite; **switch the stage timing on** — it defaults off, and this row said `_TIMING=0` for a long time, which reads as though it defaults on |
| `CUDAPIPE_NO_ABUF_BATCH` | stop merging consecutive blended draws into one A-buffer episode |
| `CUDAPIPE_SMALL_ALLOC` / `_MAX` / `_WARMUP` / `_STATS` | the small-allocation arena: `advise\|pinned\|managed\|off\|blocksonly`, the size ceiling, how many allocations before it opens, and an exit-time dump |
| `CUDAPIPE_NO_BINCACHE` | rebuild the binning queues on every peel pass, as before |
| `CUDAPIPE_SMALL_THRESHOLD` / `MEDIUM_THRESHOLD` / `POINT_THRESHOLD` | rasterizer stage boundaries, `-D` at NVRTC time — sweep without rebuilding |
| `CUDAPIPE_TILE_BOUND` | `0` puts stage 3 back to walking the whole tile |
| `CUDAPIPE_MAX_REGISTERS` | cap shader registers via `CU_JIT_MAX_REGISTERS`; forces every shader, bypassing the policy below |
| `CUDAPIPE_NO_REGCAP` | disable the per-shader register-cap policy — the driver compiles both builds and times them on real draws |
| `CUDAPIPE_SHADER_STATS=1` | every shader's registers, spill, blocks/SM and capping decision |
| `CUDAPIPE_REGCAP_STATIC=1` / `CUDAPIPE_TUNE_VETO` | the compile-time-only shape, kept measurable; the veto threshold (not load-bearing between 1.03 and 1.05) |
| `CUDAPIPE_DUMP_NIR` / `DUMP_PTX` / `DUMP_IR` | dump shader IR at each stage |

**This table is incomplete and always has been** — the driver reads 44
environment variables across 68 call sites, and the ones above are the subset
somebody remembered to write down. `TODO_DEBUG_FLAGS.md` has the full inventory
with each one's parsing and default, and a plan to generate tables like this
one from a registry so they stop drifting. Until then, `grep -rn 'getenv(' 
src/gallium/drivers/cudapipe/` is the authority, not this table.

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
  lets the driver JIT forward. **Do not raise it expecting a win — `SM120.md`
  asked and answered this.** LLVM 18 tops out at `sm_90a`/`ptx83` and sm_120
  needs PTX ISA 8.7, so it cannot emit loadable sm_120 PTX at any setting; the
  register allocation and SASS already target sm_120 because ptxas compiles for
  the context's device whatever the PTX names; and the sm_86 → sm_120 delta for
  a shader is tensor cores and TMA, which the NVPTX backend only emits from
  intrinsics. The features that would help this driver — `griddepcontrol`,
  clusters — are already open to the `.cu` kernels, which go through NVRTC at
  `compute_120` and never touch LLVM.
* **Batching's correctness rests on deferral, not on merging, and the invariant
  is not self-enforcing.** Holding a draw back means `cp_draw_execute()` runs
  after the next draw's state has been bound, which rendered `vulkanscene` 12.9%
  wrong with every batch of size *one*. Every state setter therefore flushes the
  pending batch when the incoming value differs, and every path that observes
  rendering flushes before it looks — 27 call sites. A state setter added later
  that does not flush is silently wrong, and the sample sweep may not catch it:
  this one did not, until a batch cap of one was tried deliberately. The same
  holds for `reads_const_bufs`, which is sound only while
  `emit_const_buf_base()` remains the only reader of `args[18..]`.
  `CUDAPIPE_BATCH_MAX=1` must stay bit-identical to `CUDAPIPE_NO_BATCH=1`; that
  check separates "deferral is sound" from "merging is sound", which fail
  differently.
