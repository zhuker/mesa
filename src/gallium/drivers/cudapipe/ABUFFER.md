# The A-buffer: one pass instead of two hundred and fifty-six

`PERFORMANCE_PLAN.md` Phase 3 asks for per-tile primitive lists replayed in
submission order, and names blended draws as the part draw batching could not
reach. This is the record of building that for blended draws in a different
shape than the plan proposes, and what it was worth.

**`particlesystem` 34.96 → 5.13 ms, and the sweep 91.81 → 62.00.**

**It is now the default.** `CUDAPIPE_NO_ABUFFER=1` goes back to the peel loop.
It landed opt-in first and was defaulted in a second pass that removed the two
things making that a bad idea — see "Making it the default" below, which also
records the twenty blend configurations it was tested against and a pre-existing
defect they turned up in the peel loop.

With the default flip the sweep is **61.82 ms** and particlesystem **5.07**.

> **Later, on a real application.** Two things below were true of every sample
> and false of the first captured application this was run against, and both
> cost more than everything this document measures. They are recorded here
> because the shape of the mistake is the same in each: a decision taken once,
> correct for a workload that renders one size and rebinds nothing.
>
> **The A-buffer switched itself off and nobody noticed.** `cp_abuf_setup` had
> no realloc path, so the first framebuffer size change set `ab->disabled` for
> the life of the context. A sample renders one size, so it never fired. The
> capture runs fifteen render passes a frame at 1280x720 and 160x90, so it
> fired at about frame five and the remaining fifteen hundred frames all peeled.
> Fixing it — grow-only capacity, per-size values recomputed — was worth 19% of
> the whole replay, and the peel loop's convergence drain, which was 61.5% of
> the frame, fell to 1.3% as a side effect. See commit "let the A-buffer survive
> a framebuffer resize".
>
> **Batching blended draws is worth 44%, and the batch key had to change first.**
> The batching section below is right that this is the next increment. What it
> could not know is that under the key as it stood, consecutive blended draws
> share *nothing*: mean run length 1.00 over 280,000 eligible draws, with
> `fs_ubos` breaking 97% of them, confirmed by hashing the contents rather than
> comparing pointers. The fragment shader rebinds a 64 and a 96 byte uniform
> block every draw and the constants genuinely differ. Per-draw fragment
> bindings, plus a stable clipper so the primitive index stays monotone across
> merged draws, took the capture from 333 to 185 seconds. Both are in commit
> "batch blended draws, once their bindings allow it".
>
> The numbers in this document are otherwise unchanged, and `particlesystem`
> still does not batch — the sweep is flat across both commits. But **5.07 ms
> is stale as a baseline**: the current tree runs it at 4.06, and a spot check
> against that figure was briefly mistaken for a 19% win that a controlled
> comparison then put at −0.5%.

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

## Making it the default

Three of the five reasons it landed opt-in were removed; two were accepted.

**Draws that never use it were paying for it.** The opt-in version compiled the
count and fill branches into `emit_fragment()` for every draw, and
`multisampling` cost 3.35–3.45% for branches it never takes, running them four
times per pixel. `emit_fragment()`, `rasterize_point()` and the three stage
bodies are now `template <bool ABUF>` and one compile emits six entry points:
the `<false>` set every draw launches, with no A-buffer code in them at all, and
the `<true>` set the count and fill passes launch.

That is §0.1's CuRast reference arriving from the other direction — *"branching
at runtime may increase rendering duration by a couple of percent"* — and it is
exactly what this was. A second module was measured rather than argued against:
over `triangle` at one frame, nearly all start-up, templating costs +0.21 s of
process wall where a second NVRTC compile and `cuModuleLoadData` cost +0.75 s
and leave a second module resident. Neither lands in `ms/frame`.

`multisampling`, 600 frames, interleaved: **3.406** compiled out, **3.520** as
the opt-in version would have defaulted, **3.375** now — below the build with
the feature compiled out, and the same whether the A-buffer runs or not.

**The allocation is now 2x headroom with growth at 75% occupancy**, once per
draw between the count and the fill, bounded by a cap and a growth count,
announced on stderr, falling back to the peel loop if refused. Peak over 600
frames is **54.9% of capacity with no growths**, against 87.1%. The growth path
was exercised deliberately with a throwaway build at 1x headroom: eight growths,
the cap, a clean fallback.

**The nondeterminism has an explanation, and it is not this path.** See below.

Accepted rather than fixed: **+358 MiB** of device memory, and
`VK_BLEND_FACTOR_CONSTANT_COLOR`/`CONSTANT_ALPHA`, which are in the driver's
enum and unimplemented — `cp_blend_factor` silently returns 1.0 — on **both**
paths. Inherited rather than introduced, and more visible now this is the
default.

## Blend coverage, and what it found in the peel loop

Twenty configurations, driven from the environment through one rebuild of the
sample so there is no edit-without-rebuild window; both edited files restored,
the binary verified md5-identical to the pre-edit one, and the baseline
reproduced afterwards. Floor established by running each path against itself:
**≤2 differing pixels at 1 LSB per frame.**

**Twelve are at or inside the floor and eight are byte-identical on all twenty
frames** — including `SRC_ALPHA`/`ONE_MINUS_SRC_ALPHA`, `SRC_ALPHA`/`ONE`,
`SRC_ALPHA_SATURATE`, `REVERSE_SUBTRACT`, `MIN`, separate RGB and alpha
equations, and an sRGB attachment.

**Seven are not, and the peel loop is the one that cannot reproduce itself.**
Under equations nonlinear in the destination — `SUBTRACT`, `MAX`,
`ONE`/`INV_DST_COLOR`, `DST_ALPHA`/`INV_SRC_COLOR` — two runs of the *peel
path* differ by thousands of pixels, on the pre-change driver with the A-buffer
unset. Under `MAX`, which is order-independent and idempotent, the A-buffer is
**bit-deterministic across twenty frames while peeling wanders by 1,496
pixels**. Against NVIDIA the A-buffer sits inside the peel path's own
run-to-run spread on every one of the seven.

The fragment population is bit-identical run to run and the pass count is a
constant 256, so the variable is per-fragment shaded colour: particlesystem's
documented one-LSB instability, amplified by the equation's conditioning rather
than by the path. **So "the A-buffer amplifies the sample's nondeterminism" was
the wrong reading — the amplifier is the arithmetic, and the peel loop has it
worse.** A driver that is nondeterministic under `SUBTRACT` and `MAX` is worth
its own pass.

Two caveats on the matrix itself: `particle.frag` writes alpha 0 for flame
particles, so the `SRC_ALPHA`-source rows under-test the composite — the
dst-dependent rows are the ones exercising ordering and per-layer quantisation.
And one anomaly recorded without explanation: `CUDAPIPE_ABUFFER_LAYERS=256`
makes the A-buffer nondeterministic under `MAX` where uncapped it is bit-exact.
Capping is a bisecting tool, not the default path.

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

The switches for this subsystem are the A-buffer section of
[`FLAGS.md`](FLAGS.md), which is generated from the registry in `cp_debug.c`
and cannot drift. `CUDAPIPE_HELP=1` prints the same thing from a running
driver, with what each one resolved to.

The table that used to be here was wrong, which is why it is gone: it gave
`CUDAPIPE_ABUFFER_TIMING=0` as the way to drop the per-draw drain, implying
the timing defaulted on. It has defaulted off since the A-buffer stopped being
opt-in, so anyone following that line set a variable that changed nothing.
`CUDAPIPE_ABUFFER_TIMING=1` is what turns it on.
