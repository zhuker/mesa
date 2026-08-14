# The A-buffer: one pass instead of two hundred and fifty-six

`PERFORMANCE_PLAN.md` Phase 3 asks for per-tile primitive lists replayed in
submission order, and names blended draws as the part draw batching could not
reach. This is the record of building that for blended draws in a different
shape than the plan proposes, and what it was worth.

**`particlesystem` 34.96 → 5.13 ms, and the sweep 91.81 → 62.00.**

It is **opt-in**, behind `CUDAPIPE_ABUFFER=1`. With the variable unset the
driver peels exactly as before and benchmarks within 0.05% of the build without
it. Making it the default is a separate decision and the reasons are at the end.

---

## The number

| | peel | A-buffer | |
|---|---|---|---|
| particlesystem | 34.96 | **5.13** | **−85.3%** |
| **sweep total** | **91.81** | **62.00** | **−32.5%** |

Both sides `--benchwarmup 0`, same binary, same sitting; the peel baseline
agrees with the stored `tilebound` iteration to 0.05%. Records are at
`build/iter/peelw0` and `build/iter/abufw0`.

particlesystem was 2.2x slower than llvmpipe and is now **3.1x faster**; over
the set cudapipe goes from 2.53x to **3.74x** ahead. It is no longer the worst
sample in the set by any measure — `bloom`, untouched by this, now is.

An in-binary A/B over 600 frames, one `.so`:

| | ms/frame |
|---|---|
| `CUDAPIPE_ABUFFER` unset | 34.94 |
| `CUDAPIPE_ABUFFER=1 COMPOSITE=0` (lists built, still peeling) | 38.65 |
| `CUDAPIPE_ABUFFER=1` | **5.09** |

---

## What the peel loop was actually doing

Three measurements, in the order they were taken, and the third is the one that
makes the case.

**The population.** A census counting at the top of `emit_fragment()` — the one
site all three rasterizer stages route through — puts the fire's draw at
**3.61–3.75M raw coverage events over ~42,000 covered pixels**, 4.6% of the
framebuffer, with a median depth of 26, p90 of 300 and a **maximum of 411**.
The depth-passing subset is 2.64–2.73M over ~32,000 pixels.

**The cap truncates, invisibly.** 4,040 pixels are deeper than
`CP_BLEND_LAYERS`, so **360,096 fragments — 13.6% of the population — are never
composited**. Rendering at a 512 cap is nonetheless byte-identical to 256: the
fire's core saturates long before the cap, and at 128 layers red and green are
bit-identical frame-wide with 641 pixels of blue moving. So the truncation is
real, invisible, and means the A-buffer renders *more* than what it replaces.

**The redundancy.** `emit_fragment()` is called roughly **960 million times a
frame** — 3.75M coverage events × 256 passes — and **2.28M of those calls
composite anything.** The other 99.76% are rejected because the primitive index
is below that pixel's `peel_next`. The loop re-derives the entire coverage of
the draw, 256 times over, to find one more layer per pixel each time.

That ratio is the argument for the rewrite, more directly than any share table.

---

## The shape, and why not the plan's

§3.2 says one block per tile, walking its list and **shading and blending
inline**. That requires calling the fragment shader from inside the tile kernel,
and cudapipe's shaders are separately compiled `__global__` kernels — so it
drags a compiler change in before a single pixel comes out. `CUDAPIPE_HANDOFF.md`
already names this as why llvmpipe's shape was not ported.

This takes the other arrangement, which keeps the existing shader machinery:

1. **Rasterize twice, cheaply**, to count fragments per pixel and then fill
   per-pixel runs. Both passes return from `emit_fragment` before the visibility
   buffer, so neither can change what is rendered.
2. **Sort** each pixel's run by primitive index — bitonic in shared memory, one
   block per covered pixel, over a worklist.
3. **Merge** each 2x2 block's four sorted runs into a quad stream of
   `(block, primitive, 4-bit coverage mask)`.
4. **Interpolate** the varyings for those quads and **shade** them in one flat
   launch.
5. **Composite**: one thread per covered pixel reads the attachment once, walks
   its run in ascending primitive order blending each fragment, writes back once.

Ordering is exact and Vulkan-correct by construction, because the sort key is
the submission index — which is §3.3's correction to CuRast, arrived at from
the other end. It has no per-pixel variation, so the sort is exact and CuRast's
four-slot fragment stash has nothing to do.

---

## Where the win comes from, and one prediction that was wrong in a useful direction

| step | ms/frame |
|---|---|
| count | 0.051 |
| scan | 0.017 |
| fill | 0.096 |
| sort | 0.144 |
| block worklist | 0.003 |
| merge | 0.61 |
| interpolate | 0.92 |
| shade | 0.60 |
| composite | 0.60 |
| **whole draw** | **3.04** |

Three ms for a draw that cost about thirty-two.

**Shading was expected to get more expensive and got 7.6x cheaper.** The worry
was that a 2x2 block sees primitives covering only one of its four pixels, so
helper lanes would multiply. The opposite happened:

| | quads | lanes shaded | carrying a fragment |
|---|---|---|---|
| peel, 256 passes | 1,463,364 | 5.85M | 2.28M — **39% useful** |
| merged quad stream | 680,392 | 2.72M | 2.64M — **97% useful** |

These sprites are large enough that a primitive touching a block usually covers
all four of its pixels. The peel path's 61% waste is the cost of shading one
layer at a time: the same `(block, primitive)` is re-emitted on several passes,
each time with partial coverage. About 2.1x of the shading win is simply less
work and the rest is one launch instead of 256.

**Interpolation, not shading, is the largest piece**, and the composite is the
same size as either — 0.60 ms, not the rounding error it was projected to be.

---

## How it was checked

Each step was verified against the peel loop before the next was built, and the
apparatus is still switchable (`CUDAPIPE_ABUFFER_VERIFY=1`) so the claims can be
re-run rather than re-argued.

| step | claim | result |
|---|---|---|
| lists | a pixel's sorted run is the sequence peeling composites there | all covered pixels 32 deep, plus the deepest 1000 at full depth: **0 mismatches** |
| quads | the `(block, primitive, mask)` multiset is what `cp_fs_interpolate` emits over all passes | 592,366 triples: **0 mismatches, 0 not-found, 0 extra** |
| shading | every fragment the peel path shades comes out bit-identical | ~2.37M fragments a frame, compared as raw bits: **0 mismatches** |
| image | the final frame is the peel frame | **19 of 30 frames stable on both paths, all 19 byte-identical; 0 frames where both were stable and differed** |

The shading check has teeth: perturbing the interpolation to sample one pixel
across produced immediate mismatches with coordinates and both colours.

**One claim as originally stated was false and had to be corrected.** The peel
path emits a quad per `(block, primitive)` *per pass*, and one primitive
routinely wins different pixels of the same block on different passes — so it
emits 1.46M quads where the merge emits 680k. They agree only after OR-ing the
peel side's masks per `(block, primitive)`, which is what "summed over the
passes" has to mean.

The sweep verdict is **identical on 15 of 18 samples**, both standing
regressions unchanged to the pixel. The three that move are on the documented
nondeterministic list, two of them do not take the new path at all, and
particlesystem's worst frame goes 1638 → 1636 — two pixels closer to NVIDIA.

---

## Eligibility, and the hazard that is not merging

A draw takes this path only if it is blended and peeled, single-sample, has
**no depth write mask at all**, has exactly one colour attachment with a known
encoding, and — decided on the device after the merge — produced quads without
overflowing either array. Anything else runs the peel loop.

The depth-write test is tighter than it looks necessary: it keys on the mask
alone rather than `enabled && writemask`, because `cp_fs_writeback` does, and a
draw with the test off and the mask on does write depth there.

**Only `particlesystem` takes the path.** `bloom`'s blended draw reports
`test=1 mask=1` and is refused, as it was under the old gate too.

**The composite must be quantised at every layer.** The peel loop stores each
layer into the attachment and reloads it, so on an 8-bit target every layer is
rounded before the next blends against it. The composite therefore round-trips
through `cp_store_dst`/`cp_load_dst` per layer rather than carrying float across
the run. Carrying float would be *more* accurate and would not be the peel
loop's image.

The blend arithmetic is factored into one `cp_blend_resolve()` over a shared
`struct cp_blend_desc`, called by both `cp_fs_writeback` and the composite.
Two copies that can drift is the failure this driver already has one of —
`CUDAPIPE_HANDOFF.md` gap 12, where stage 2 and stage 3 interpolate depth in
separately written code and disagree by an ulp.

---

## What is not done, and why it is not the default

- **One blend equation has ever run through the composite.** particlesystem is
  the only sample that reaches it. Vulkan has ~19 factors x 5 operations with
  separate RGB and alpha. The shared function means the arithmetic cannot
  drift; it is not evidence that the path generalises, and nothing here tests a
  second factor combination, a colour write mask, or an sRGB attachment.
- **Draws that never use it still pay.** `CUDAPIPE_ABUFFER=1` compiles the count
  and fill branches into `emit_fragment` for every draw, and `multisampling`
  costs **+3.5%** for it, reproducibly — it runs them four times per pixel.
  Those branches are load-bearing now, not instrumentation.
- **The allocation headroom is 13 points, not 27.** The fragment array is sized
  from the first draw seen; over 600 frames the population peaks at **87.1% of
  capacity**. Overrun falls back to the peel loop rather than corrupting
  anything, which is the right failure, but a scene that grew more over its
  animation would silently drop to 35 ms.
- **The path amplifies the sample's own nondeterminism and nobody knows why.**
  particlesystem's documented one-pixel/one-LSB instability fires on 11 of 30
  frames here against 2 of 30 while peeling. The variant images are the same
  ones the peel path produces, and runs of both paths land in the same groups,
  so it reads as an existing race being reached more often rather than a new
  one. It is unexplained.
- **+324 MiB** of device memory, flat across 600 frames.

## Where the next win is

The composite is a **gather** — 2.6M `float4` reads through the slot map at
~71 GB/s. Sizing its grid from the covered-pixel count changed nothing, which
says memory-bound rather than launch-bound. Scattering colours into slot order,
or reordering the quad stream so a pixel's run is contiguous, is the obvious
target. After that, the merge's counting pass is redundant — the four per-pixel
counts bound a block's quad count without merging at all — which is worth about
0.28 ms.

---

## Reproducing

```bash
cd ~/git/Vulkan
T=~/mesa/src/gallium/drivers/cudapipe/tests
ICD=~/mesa/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json

CUDAPIPE_ABUFFER=1 VK_DRIVER_FILES=$ICD build/bin/particlesystem --offscreen --benchmark --offscreenframes 600
CUDAPIPE_ABUFFER=1 CUDAPIPE_ABUFFER_VERIFY=1 VK_DRIVER_FILES=$ICD build/bin/particlesystem --offscreen --offscreenframes 8
DESC="..." $T/cp_iterate.sh mylabel abufw0
```

| variable | effect |
|---|---|
| `CUDAPIPE_ABUFFER` | build the A-buffer and composite from it |
| `CUDAPIPE_ABUFFER_COMPOSITE=0` | build the lists beside the peel loop, still peel |
| `CUDAPIPE_ABUFFER_VERIFY=1` | the step-by-step checks above; ~240 MB of comparison buffers |
| `CUDAPIPE_ABUFFER_TIMING=0` | drop the per-draw drain the CUDA-event timings need |
| `CUDAPIPE_ABUFFER_LAYERS=N` | cap the composite, for bisecting against the peel cap |
| `CUDAPIPE_ABUF_COMPILE` | force the kernels in or out of the NVRTC build |
