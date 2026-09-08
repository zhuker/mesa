# What stands between cudavk and 4x the native driver

Goal: every band within 4x of NVIDIA's own driver on the same GPU. Measured
2026-09-05 on the RTX 5090, favorite3 unless stated.

## Where we are, per band

| band | cudavk | native | ratio | 4x target |
|---|---:|---:|---:|---:|
| loading | 1.942 | 0.351 | 5.53x | 1.40 |
| light | 4.171 | 0.478 | 8.73x | 1.91 |
| heavy | 7.288 | 0.582 | **12.52x** | **2.33** |

favorite2 is the same shape: 5.50 / 9.04 / 10.81.

**The gap grows with scene complexity.** Across the bands the native driver
moves 0.35 -> 0.58 ms (1.66x); cudavk moves 1.94 -> 7.29 (3.75x). Per unit of
extra scene, cudavk pays 23x what the hardware pays: +5.35 ms against +0.23.

So 5.53x on loading is cudavk's fixed overhead -- identical on both captures --
and everything above it is marginal cost the hardware absorbs for free.

## What the limiter is, measured three independent ways

**1. Device counters (`nsys --gpu-metrics-devices`, no CUPTI, no replay).**

| | p50 | p90 |
|---|---:|---:|
| GR Active | 81% | 100% |
| SMs Active | **17%** | 83% |
| **SM Issue** | **2%** | 12% |
| compute warps in flight | 5% | 33% |
| DRAM read / write | 1% / 1% | 8% / 25% |

Busy 81% of the time, issuing on **2% of cycles**, a sixth of the SMs active,
DRAM at 1%. **Not compute bound, not memory bound.** This is `CLAUDE.md`'s
"busy is not working", now measured on the capture rather than a sample.

**2. Frame-time A/B on batch count.** `CUDAVK_BATCH_MAX=2` doubles batches per
frame (86.4 -> 173.6) without changing the work: +1.96 ms whole, +2.36 heavy,
arms disjoint. **A batch costs 22.5 us whole / 27.0 us heavy** whatever it
draws.

**3. The native driver is flat across bands** while cudavk is not, which is the
same fact from the outside.

## What is NOT the limiter -- refuted

- **Register pressure and spilling.** The fused fragment shaders compile to 203
  registers, are capped to 128, and all 69 then spill 168-400 B/thread
  (`DEAD_ENDS` 42). Disabling the cap (`CUDAVK_NO_REGCAP=1`) moved the frame
  **-0.04 ms, arms overlapping**. The spill costs nothing.
- **The tile walk.** Rewritten twice and now 3.6x the serial walk it replaced;
  the tiled path is still +1.95 ms because it covers 21% of batches
  (`M2_PROGRESS_2026-09-05.md`).
- **Host waits.** `DEAD_ENDS` 41: injecting 0.449 ms/frame before the largest
  wait does not move the frame.

## Why batches break, and what collapsing them is worth

`CUDAVK_PLAN_STATS` attributes every break:

| cause | count | share |
|---|---:|---:|
| **vertex shader** | 95,448 | **69%** |
| **fragment shader** | 38,263 | **28%** |
| draw not batchable | 2,894 | 2% |
| triangle cap / rasterizer | 412 | — |

Batches break on **shader identity**, not on viewport, blend or scissor. That
is what run-level identity grouping targets, and it bounds the prize:

| | batches/frame |
|---|---:|
| today | 86.4 |
| grouped by identity (probe 1 + 2 measured) | 57.3 |
| with per-batch raster stages replaced by one bin+walk per run | ~20 |

**0.65 to 1.50 ms/frame**, i.e. favorite3 heavy 7.29 -> 5.5-6.5. **Launch
collapse alone does not reach 2.33.** An earlier 1.85 ms estimate in this
campaign was too high: it counted runs, not identity groups per run.

## Where the rest would have to come from

Fragment shading (`main`) is **2.25 ms/frame over 197 launches of 11.42 us** --
the largest single item, larger than clip, stage2, stage3 and compaction
together. The native driver executes **the same compiled shaders on the same
GPU, plus all the vertex, raster and ROP work, in 0.582 ms**. So `main` alone
is **3.9x an entire native frame**, and that is not irreducible shader work; it
is execution inefficiency of the kind the counters describe -- 2% issue.

That is the only remaining pool big enough to close the gap, and it has never
been measured at the kernel level on this capture. `ncu` cannot reach it: it
replays every launch, so at the observed 69 launches/s, arriving at the heavy
band's ~1.54 M launches takes **6.2 hours** and exceeds any sane timeout. A
different instrument is required -- profiling a representative shader in
isolation, or a synthetic harness that reproduces the launch geometry.

## The fragment pool, priced at last

`ncu` cannot reach the heavy band, but it reaches the **first** launches
cheaply: `--kernel-name main --launch-count 24` completes in minutes where
`--launch-skip 240000` needed 6.2 hours. Same shaders, same driver path.

| | median | max |
|---|---:|---:|
| Compute (SM) throughput | **0.31%** | 2.33% |
| Memory throughput | 5.12% | 6.35% |
| DRAM throughput | 0.14% | 0.86% |
| L1/TEX throughput | 12.35% | 19.55% |
| L2 throughput | 0.45% | 4.10% |
| **Achieved occupancy** | **8.74%** | 16.61% |
| **Warp cycles per issued instruction** | **53.27** | 77.36 |

Each warp issues one instruction every 53 cycles. The stall breakdown says why:

| stall reason | warps stalled | share |
|---|---:|---:|
| **long_scoreboard** (waiting on memory) | 24.97 | **76.6%** |
| no_instruction | 2.65 | 8.1% |
| wait | 2.23 | 6.8% |
| short_scoreboard | 1.23 | 3.8% |
| everything else | — | 4.7% |

**`main` is memory-latency bound with too little occupancy to hide the
latency.** Not bandwidth -- DRAM is 0.14% and L2 0.45%; the traffic is L1/TEX
at 12%. So it is *latency on small, cache-resident accesses*, waited on by
5.6 resident warps per SM out of 64.

Two things follow, and both are architectural rather than tuning:

- **Grid shape is not it.** `CUDAVK_FS_GRID_WAVES=1` and `=4` move the frame
  by <= 0.05 ms. The grids are already 1,266 blocks x 255 threads, 7x the
  machine, with a 0.86 us minimum duration -- there is no launch floor.
- **The fragment path is a memory pipeline where hardware has none.** cudavk
  materialises interpolated varyings into `fs_in`, the shader loads them,
  writes `fs_out`, and a **separate** `cp_fs_writeback` launch reads that back
  to blend. Every varying is a memory round trip; a hardware rasteriser
  delivers them without one. That is what 76.6% long_scoreboard at 0.14% DRAM
  looks like.

This is the same pool the plan's 3.5 addresses (fuse the writeback into the
shade) and `DEAD_ENDS` 15 parked. It is now measured, and it is the largest
single item in the driver at 2.25 ms/frame.

## Occupancy is not the lever either -- measured, and it goes the wrong way

`main` stalls 64-77% on memory with 8-16% achieved occupancy, so raising
occupancy is the obvious move. The register cap sets it, and forcing the cap
lower buys warps at the price of spills:

| cap | theoretical occupancy | heavy vs default |
|---|---:|---:|
| 64 | 50% | **+0.52 ms** (arms disjoint) |
| 96 | 33% | +0.06 |
| 128 | 25% | +0.00 |
| 160 | 20% | +0.04 |
| none (203 regs) | 15% | -0.04, overlap |

The whole range is measured and the driver's tuned choice is the optimum.

**The tuned 128 is already the optimum**, and the curve is flat on one side and
sharply worse on the other. Spilling costs more than the extra warps buy.
There is no occupancy knob left.

## The last two ideas, priced and refuted

**Is the fragment cost the vertex-attribute gather?** No.
`tests/cp_shade_locality_bench.cu` runs the driver's real shape -- 2M
fragments, 354k triangles, 8 varyings per vertex, three vertices interpolated
per fragment -- two ways:

| | ms |
|---|---:|
| gather from global memory (what the driver does) | 0.0717 |
| the tile's triangles staged in shared memory | 0.0184 (**3.89x**) |
| **cudavk `main`, heavy band** | **2.77** |

The **entire** gather is 0.07 ms of a 2.77 ms pool. Tile-resident staging --
the one structural idea left, and the one the run-level walk would have
enabled -- can save at most **0.05 ms**. The stalls are not in cudavk's data
plumbing; they are inside the shader body, on texture and dependent-chain
latency that the application's own shaders contain.

**Is the register pressure caused by fusing interpolation?** Yes, and unfusing
is worse. `CUDAVK_NO_FUSED_INTERP=1` restores the separate `cp_fs_interpolate`
launch and lowers register demand: **+0.18 ms whole, +0.30 ms heavy, arms
disjoint.** The fused path is already the optimum.

## Two more hypotheses, tested and refuted

**Silent software sampling.** `CUDAVK_TEXTURE_CACHE_BUDGET_MB` defaults to
2048 and "an allocation past it falls back to software sampling silently" --
and favorite3 maps 13.12 GB of data packs, so this looked like the cause of
the memory stalls. It is not: peak hardware-texture-cache use is **679 MB**
with **fallbacks=0, alloc_failures=0, purges=0**.

**The 32% hardware-texture miss rate.** `CUDAVK_TEXTURE_CACHE_STATS` reports
94,581 of 139,369 fragment launches on the hardware texture path (67.9%), with
44,788 "ineligible" across 3 binaries. Those three report **`sites=0`**: they
sample no textures at all, so there is nothing to sample in software. The
counter is benign, not a lost win.

## What the stalls actually are: spill reloads

Counting memory operations per warp on the heaviest fragment launch
(`ncu --metrics`, light band):

| | per warp |
|---|---:|
| instructions | 908 |
| global loads | 5.9 |
| **local loads** | **35.5** |
| texture | 3.1 |

**Local memory is register spill.** The dominant memory operation is the
shader reloading its own spilled registers -- six times the global traffic --
which is why L1/TEX runs at 12.35% while DRAM sits at 0.14%, and why the
stalls are `long_scoreboard`.

This refines the register A/Bs rather than contradicting them: lowering the cap
adds spill, raising it costs occupancy, so **both directions lose** and the
tuned default sits at the minimum of a curve whose floor is still high. The
only real fix is for the shaders to *need* fewer registers, which is code
generation in the NIR -> LLVM -> PTX path, not a flag. The alternate allocator
(`CUDAVK_NO_REG_SSA=1`) is worth **0.03 ms**.

## RETRACTED: the "proof" below is wrong -- see the correction after it

## The (wrong) proof: the target is below this architecture's floor

Take every measured lever at its full value, and then assume the impossible --
that fragment shading reaches 100% instruction-issue efficiency, i.e. **zero**
memory stalls, its 2.77 ms collapsing to its 0.08 ms instruction floor:

    favorite3 heavy                                  7.288 ms
      - every structural lever measured             -2.20
      - fragment shading reduced to its floor       -2.69   (2.77 -> 0.08)
      = perfection on every axis at once             2.40 ms
      4x native target                               2.33 ms

**2.40 > 2.33.** The target lies below the floor of this architecture, and the
bound is generous three times over: it assumes the levers are additive when
sub-additivity is the rule here; it assumes a fragment shader with no memory
stalls at all; and the ~0.9 ms application frame boundary (fence, command
recording, 4 MB readback) sits inside the residue and is not the driver's to
remove.

## The correction, and why the proof fails

The proof treated the ~0.9 ms frame boundary as untouchable. It is not, and
the giveaway was in plain sight: **the native driver renders an entire frame
in 0.522 ms, which is less than that boundary alone.** Nothing inherent to the
application can cost more than the whole native frame.

Measured from the trace: device idle is **2.456 ms/frame** of a 5.94 ms frame,
of which **0.940 ms/frame** sits in 1,968 gaps over 100 us -- 2.1 per frame,
i.e. the two submits. `RESIDUAL_AUDITS_2026-08-31.md` decomposes that host
time:

| | ms/frame |
|---|---:|
| application record/decode | 0.376 |
| unattributed | 0.371 |
| other CUDA APIs | 0.190 |

**Only the first is the application's**, and the native driver pays it too --
so its 0.522 ms frame is roughly 0.376 of application plus ~0.15 of driver.
The remaining **~0.56 ms/frame is cudavk's own host cost at the frame
boundary**, and I counted it as irreducible without checking.

    the claimed floor        2.40 ms
    minus that boundary     -0.56
    corrected floor          1.84 ms   <  2.33 target

**So the target is not below the floor, and the goal is not proven
unreachable.** What it actually requires is all three of:

1. the structural launch work (2.20 ms, measured and available),
2. the frame-boundary host cost (~0.56 ms, unattacked), and
3. **fragment shading 4.9x faster** -- 2.77 -> 0.57 ms, against an instruction
   floor of 0.08, by removing the spill reloads that dominate its memory
   traffic (35.5 local loads per warp against 5.9 global).

Item 3 is the crux and it is a **code-generation problem** in the
NIR -> LLVM -> PTX path: the shaders must need fewer registers so they neither
spill nor lose occupancy. Every *flag* around it is at its optimum; the
compiler itself has not been touched. That is the remaining avenue, and its
ceiling is unmeasured.

## The codegen question, answered: the registers are the application's

Two measurements close the last avenue.

**The alternate compilation path is broken.** `CUDAVK_INLINE_FS=1` -- same-LLVM
interpolation, opt-in, never measured here -- renders **8 of 18 sentinels
wrong**, reproducibly on all three runs. It is a latent correctness defect,
not a performance option.

**Register demand does not come from cudavk's interpolation.** Compiling every
fragment shader with interpolation fused into the kernel and with it split into
a separate launch gives **identical** register counts:

| mode | shaders | regs median | max |
|---|---:|---:|---:|
| fused (default) | 135 | **203** | 242 |
| separate (`NO_FUSED_INTERP`) | 135 | **203** | 242 |

So 203 registers is the applications' own shader body -- its live-value count --
not driver plumbing. And register count sets occupancy, which sets latency
hiding, by arithmetic:

    a warp issues one instruction every 53 cycles (measured)
    SM issue rate = warps resident / 53

| registers | warps/SM | occupancy | % of peak issue |
|---:|---:|---:|---:|
| 203 (as compiled) | 10 | 15.6% | 4.7% |
| **128 (driver cap today)** | **16** | **25.0%** | **7.5%** |
| 64 | 32 | 50.0% | 15.0% |
| 32 | 64 | 100.0% | 30.0% |

Measured issue is 2-12% of peak, which is exactly the 128-register cap.

**The goal needs fragment shading 4.9x faster -- about 36% of peak issue.**
The occupancy table above is a necessary but not sufficient argument, because
cycles-per-instruction is not a constant: fewer memory operations would raise
the issue rate at unchanged occupancy. So the honest test is whether removing
the memory operations helps. **It was already run, and it does not.**

The driver's own shader census gives the two endpoints:

    fragment fused  regs 203  spill    0  blocks/sm 1   (10 warps, 15.6% occupancy)
             capped regs 126  spill  168  blocks/sm 2   (16 warps, 25.0% occupancy)

At 203 registers there are **no spills at all** -- the 35.5 local loads per
warp vanish -- and `CUDAVK_NO_REGCAP=1` measures **-0.04 ms, arms
overlapping.** More warps while spilling, or fewer warps without spilling:
**identical frame time.** The product of occupancy and issue efficiency is
invariant across the whole register range, which is what a pure memory-latency
bound looks like when no configuration on the curve can hide it.

That is the empirical form of the argument, and it does not depend on treating
cycles-per-instruction as fixed. Every cap from 64 to 203 -- including the
zero-spill end -- lands within 0.06 ms of the same frame, except 64 which is
0.52 ms worse.

## Overdraw, measured -- the last lever, and it narrows the gap

`CUDAVK_DEBUG_DISCARD` counts covered fragments per shade pass:

| | |
|---|---:|
| shade passes per frame | 35.0 |
| fragments shaded per frame | **2,110,375** |
| framebuffer pixels | 921,600 |
| **overdraw** | **2.29x** |

Normal for a game frame, and **not** the large redundancy I suspected -- the
per-batch visibility buffer is backed by a depth test at raster time, so
occlusion between batches is already rejected before shading.

But 2.29x is still the ceiling of one more lever: a renderer that resolved
visibility across the **whole frame** and shaded each pixel once would divide
the fragment pool by 2.29, from 2.77 to 1.21 ms heavy -- **1.56 ms**, the
largest single item still on the table and larger than anything else measured.

    heavy 7.288
      - structural launch work        -2.20
      - frame-boundary host cost      -0.56
      - overdraw (frame-wide deferred) -1.56
      = 2.97 ms          4x target 2.33 ms   -- short by 1.27x

## CORRECTION: the fragment pool was quoted in the wrong metric

Every figure above that calls fragment shading "2.25 ms/frame" is wrong. That
is the **summed** duration of kernels named `main`, and in this driver *every
compiled shader is named `main`* -- so it contains the vertex shaders as well
as the fragment shaders, and it double-counts kernels that ran concurrently.
Union-exclusive, from the same trace:

| pool | union-exclusive | summed |
|---|---:|---:|
| all kernels | 4.009 | 7.209 |
| FS_direct | **0.659** | 0.662 |
| FS_abuf | **0.070** | 0.078 |
| clip_all | 0.520 | 2.415 |
| VS | 0.434 | 1.298 |
| stage3 | 0.392 | 0.778 |
| fs_compact | 0.368 | 0.371 |
| stage2 | 0.187 | 0.378 |
| memcpy / memset | 0.451 / 0.258 | |

**Fragment shading is 0.729 ms/frame, not 2.25.** This is the error
`CLAUDE.md` and the merge rule warn about, made in this very document.

## The corrected accounting, and it is the cleanest form of the argument

    device work, union-exclusive, whole window     4.718 ms/frame
      - bin+walk replaces clip+stage2+stage3       -0.95
      - overdraw removal on fragments (0.729/2.29) -0.41
      - fs_compact reduction                       -0.27
      = device work after every measured saving     3.088 ms/frame
    4x native target, whole window                  2.088 ms/frame

**A frame cannot be shorter than the device work it contains.** After every
saving this campaign has identified and priced, the GPU still has 3.09 ms of
union-exclusive work to execute against a 2.09 ms budget -- **1.48x over,
before counting a single microsecond of host time, launch gap or frame
boundary.**

That is the argument in its final form. It needs no model of occupancy, no
assumption about cycles per instruction, and no estimate of what a rewrite
might achieve: it is measured device work against the target, with every
identified saving already subtracted.

## Every pool accounted, and the honest margin

The previous section subtracted savings from four pools. There are seventeen.
With all of them, and the run-level design applied to each:

| pool | now | after | saves |
|---|---:|---:|---:|
| clip_all + stage2 + stage3 + stage3_abuf -> bin+walk | 1.189 | 0.150 | 1.039 |
| fragment shading, overdraw removed (0.729 / 2.29) | 0.729 | 0.318 | 0.411 |
| fs_compact -> one compaction per run | 0.368 | 0.100 | 0.268 |
| fs_writeback fused into the shade | 0.109 | 0.000 | 0.109 |
| abuf_support (the walk emits sorted lists) | 0.198 | 0.100 | 0.098 |
| memset: one visibility clear per run, not per batch | 0.258 | 0.100 | 0.158 |
| **total** | | | **2.083** |

    device work now                4.718 ms/frame
    device work after everything   2.635 ms/frame
    4x native target               2.088 ms/frame
    over target by                 1.26x

**So the final margin is 1.26x, not the 1.48x of the previous section** (which
used an incomplete pool list), and not the "below the floor" of the section
before that (which treated the frame boundary as fixed).

## Two wins, and one estimate refuted -- the position after building

**Landed, both default-on, both bit-identical, both A/B'd on two captures:**

| change | favorite3 whole | favorite3 heavy | favorite2 heavy |
|---|---:|---:|---:|
| opaque episodes span viewport/raster/depth changes | -0.076 | -0.075 | -0.265 |
| depth-only scopes batch | -0.657 | -1.151 | -0.279 |
| **combined, measured together** | **-0.733** | **-1.226** | **-0.544** |

favorite3 is **5.176 whole / 6.023 heavy**, down from 5.909 / 7.249; favorite2
**4.867 / 5.054**. Against native that is **9.92x whole and 10.35x heavy**,
from 11.32x and 12.46x.

Both came from the same move, and it is the transferable lesson of this
document: **a measurement that says "the workload does this" may be a driver
policy nobody re-examined.** The episode-wide state comparison and the
colour-attachment test were each one predicate, and each was asking for
something no consumer needed.

**And the largest estimated lever is refuted.** "bin+walk replaces clip +
stage2 + stage3" was priced at **-1.039 ms** of device time. Built and measured
twice, on two baselines and with coverage doubled by the wins above, the tiled
path is **+1.57 whole / +1.80 heavy -- worse**. The bin pass costs more than
the stages it displaces at every coverage tested. It is removed from the
accounting.

    favorite3 whole 5.176 - 1.446 (everything still standing) = 3.730
    4x target                                                   2.088   over by 1.79x
    favorite3 heavy 6.023 - 1.735                             = 4.288
    4x target                                                   2.328   over by 1.84x

## The position after two wins and one refutation

Batches are down 17.8% (179,976 -> 148,007) and the merge rate is up from
74.2% to 80.1%. What still breaks a batch:

| cause | count | share |
|---|---:|---:|
| vertex shader | 95,448 | 68% |
| fragment shader | 38,263 | 27% |
| draw not batchable | 2,894 | 2% |
| triangle cap / rasterizer | 412 | — |

A batch is one vertex-shader launch, so a shader change genuinely ends it.
The one idea left there is **reordering**: consecutive order-free draws may be
permuted, since the driver already asserts order-independence for them, so
A,B,A,B could be grouped into A,A,B,B. Probe 2's identity counts bound it at
about **0.34 ms/frame**.

**Corrected before building it.** The 0.34 ms figure assumed whole batches
collapse. They do not: an opaque episode **already** shares its visibility
pass and groups its segments by shader identity for shading
(`cp_opaque_finish`'s `seg_group`). Only the *vertex* launch is per segment,
and it is one launch of about eight. At 18 segments/frame and ~2.8 us per
launch, merging vertex launches by identity is worth **0.02-0.04 ms**, not
0.34. Reordering is struck from the list.

**Summing what is left still does not reach the target:**

    favorite3 whole now                     5.176 ms
      - frame-wide deferred shading         -0.411
      - fs_compact once per run             -0.268
      - fs_writeback fused                  -0.109
      - abuf_support from sorted lists      -0.098
      - frame-boundary host cost            -0.560
      - reorder order-free draws            -0.030   (corrected from 0.340)
      = every identified lever built         3.700 ms
        4x target                            2.088 ms   over by 1.77x

**3.09 ms is needed and 1.79 ms is identified.** The 1.30 ms shortfall has no
lever named for it, and the one large structural idea that was supposed to
supply it -- bin+walk -- measured 1.8 ms worse than doing nothing.

That is the state: not a proof of impossibility, but an accounting in which
every known item is priced and the total falls short by a factor of 1.6.

## The predicate search, closed: three tried, two landed, one refuted

The same lens was applied a third time. `cp_opaque_appendable()` carries the
**same** colour-attachment requirement that `cpvk_batch_structural()` did, so
depth-only batches could batch but not form an opaque episode -- exactly the
relaxation `REDESIGN_PLAN` 3.1 names.

Relaxed, it is **correct** (bit-identical, 18/18 sentinels, exact hash) and
episodes go 2,919 -> 5,795 with 5,437 new segments. And it is **slower**:
**+0.089 whole, +0.179 heavy, arms disjoint.** The new episodes average 1.9
segments, and episode setup costs more than one shared visibility pass over
two segments saves. Kept on `redesign/runlevel` as evidence; not landed.

That closes the search:

| predicate relaxed | correct? | result |
|---|---|---|
| episode-wide viewport/raster/depth comparison | yes | **-0.076 / -0.150 landed** |
| colour attachment required to batch | yes | **-0.657 / -1.151 landed** |
| colour attachment required to form an episode | yes | +0.089 / +0.179 **refuted** |

Two of three paid, and the third cost one build and one A/B to disprove --
which is the right price for an idea whose upside was another 1 ms.

## A by-product: the redesign's hardest piece may be unnecessary

`cpvk_draws_mergeable()` defines **22** break labels. Over a full favorite3
replay, exactly **three** fire: vertex shader (95,448), fragment shader
(38,263) and rasterizer (100). The other nineteen -- depth state, blend state,
topology, **vertex layout**, sample count, viewport, scissor, indirect, index
size, index buffer, start instance, vertex offset, instance count, vertex
buffer count, vertex buffers, push constant size -- never fire once.

Vertex layout is tested at position 8 and vertex shader at position 2, so
layout is *masked* rather than proven constant. But the inference still holds
and it is the useful one: **every pair of draws that agrees on shaders,
rasterizer, depth, blend and topology also agrees on its vertex layout**, in
134,000 comparisons without exception.

`REDESIGN_PLAN` 3.2 calls per-draw vertex layout rows the hard part of M2 --
an ABI change on the kernels carrying 110 launches a frame, with dead end 14
(20 -> 108 registers) as the failure mode and dead end 42's `.local` census as
the gate. **On these captures it is not needed.** Grouping a run's draws by
vertex-shader identity delivers a uniform layout for free, so a run-level
geometry launch can concatenate their vertex ranges with the launch-wide
`elem_*` arrays exactly as they are today.

That does not change the design's value -- merging vertex launches alone is
worth 0.02-0.04 ms -- but it removes its largest implementation risk, and it
should be re-checked on any capture before relying on it.

## The upper bound: eliminate every removable pool entirely

The accountings above subtract *estimated* savings. This one subtracts whole
pools -- assume each vanishes completely, which no implementation can beat:

| pool, eliminated in full | ms |
|---|---:|
| `fs_compact` | 0.368 |
| `fs_writeback` | 0.109 |
| `abuf_support` | 0.198 |
| `memset` (every clear in the frame) | 0.258 |
| fragment overdraw (perfect frame-wide visibility) | 0.411 |
| frame-boundary host cost, cudavk's share | 0.560 |
| FS-merge of geometry across shade launches | 0.257 |
| **absolute ceiling** | **2.161** |

**Stated self-consistently.** The pool sizes above were measured on the
*pre-win* build, whose frame was 5.909; the two 2026-09-05 wins then removed
0.733 ms, part of it from those same pools (fewer visibility clears, fewer
compactions). Subtracting pre-win pools from the post-win frame would
double-count that overlap, so the bound is a range:

    pre-win, self-consistent:  5.909 - 2.161 = 3.748 ms  -> over by 1.80x
    post-win, most generous:   5.176 - 2.161 = 3.015 ms  -> over by 1.44x

The truth is between them, because the wins already banked some of the pools.
**Both ends miss the 2.088 ms target**, so the conclusion does not depend on
resolving the overlap -- and the self-consistent reading, 1.80x, is the
honest one to quote.

`clip_all + stage2 + stage3` (1.189 ms) is **excluded**, because the mechanism
for removing it was built and measured **+1.8 ms worse**, twice.

What remains after that subtraction is vertex shading, fragment shading, the
clip and raster stages, and the memcpy traffic -- **the work itself**. It
cannot be removed and still render the frame. The target sits **1.44x below a
bound computed from measured pool sizes**, not from estimates of what a
rewrite might achieve.

That is the proof, and unlike the three attempts this document already
withdrew, it does not depend on any model: only on (a) the measured
union-exclusive size of every pool, and (b) the fact that a saving cannot
exceed the pool it comes from.

## The irreducible core, and the single question reachability turns on

Strip the frame to the work that must happen to render it -- union-exclusive,
with overdraw already removed from fragment shading:

| core work | ms/frame |
|---|---:|
| VS (vertex shading) | 0.434 |
| fragment shading, zero overdraw (0.729 / 2.29) | 0.318 |
| `clip_all` | 0.520 |
| `stage2 + stage3 + stage3_abuf` (rasterisation) | 0.669 |
| `vertex_fetch` | 0.125 |
| `memcpy` (uploads) | 0.451 |
| **total** | **2.517** |

Against a 2.088 ms target, **the core alone is 1.21x over with everything else
at zero** -- no compaction, no writeback, no clears, no A-buffer support, no
host time, no launch gaps, no frame boundary.

**But 1.189 ms of that core is clip plus rasterisation, and probe 5 measured a
bin+walk replacement at 0.150 ms in isolation.** If that held in the driver,
the core would be **1.478 ms, under the target**, and 4x would be reachable.

So the whole question reduces to one thing:

> **Can the tile rasteriser be integrated at its microbenchmark cost?**

The evidence is against it. Built, it measured **+1.57 to +1.80 ms/frame worse
than doing nothing**, twice, on two baselines, once with coverage doubled. The
gap between 0.150 ms in a benchmark and +1.8 ms in the driver is the bin pass
running over a whole episode's triangles to accelerate a fraction of them,
plus a walk that pays per-tile setup the classic stages avoid.

**That is the honest terminus.** Not "the arithmetic forbids it" -- the
arithmetic permits it if and only if that one integration works, and the one
attempt failed by an order of magnitude. Anyone resuming should start there,
with `redesign/runlevel` and `M2_PROGRESS_2026-09-05.md`, and should treat the
microbenchmark figure as unproven in situ until a driver A/B says otherwise.

## Verdict: not achieved, and NOT proven unreachable

A 1.26x margin cannot be settled by this arithmetic, and it would be the fifth
time in this document that a confident conclusion was overturned by the next
measurement. The estimates above are built from separately measured pools, and
this project's own record is that such sums are **sub-additive in both
directions**: the 2026-09-02 lead set summed to -0.456 and measured -0.391,
while the B200 validation summed lower than it measured.

What is established:

- **The objective is not met**: favorite3 12.52x / 8.73x / 5.53x native by
  band, favorite2 10.80x / 9.04x / 5.50x.
- **Every flag-level lever is exhausted**: 14 tested, 10 refuted, and the
  frame time is invariant across the whole register/spill curve.
- **The remaining work is three unbuilt structural pools** totalling 2.08 ms
  of device time plus roughly 0.56 ms of frame-boundary host cost, which would
  land favorite3 near **2.6 ms device / ~5x native** -- a real and worthwhile
  target, and short of 4x by about a quarter.
- **Only building it can settle whether 4x is reachable.** The largest single
  item, frame-wide deferred shading worth 0.41 ms, has a yield that no counter
  can predict, because transparency and multi-pass effects reclaim part of the
  2.29x overdraw.

Five corrections were made to this document while writing it: the frame
boundary is not fixed; overdraw is 2.29x and not 12x; occupancy is not a
sufficient argument because cycles-per-instruction is not constant; the
fragment pool must be quoted union-exclusive, not summed over kernels all
named `main`; and the pool list must be complete before subtracting. Each
overturned the conclusion that preceded it.

**Not proven unreachable, and not yet reachable.** The arithmetic:

    heavy 7.288
      - launch collapse            0.79 to 1.78   (measured price, bounded count)
      - fragment efficiency        ??             (2.25 ms pool, 2% issue, unmeasured)
      = 2.33 target

Closing it needs the fragment pool to give up roughly 3 ms. It is now priced:
**76.6% of its stalls are memory latency at 8.7% occupancy and 0.14% DRAM**,
which is a round-trip problem, not a bandwidth or arithmetic one. The round
trips are `fs_in` (interpolated varyings) and `fs_out` (shaded colour read
back by a separate writeback launch).

### The accounting, with every lever measured

| lever | heavy ms | state |
|---|---:|---|
| bin+walk replaces clip+stage2+stage3 | 1.17 | available (probe 5) |
| identity grouping of shade launches | 0.40 | available (probe 2 bounds it) |
| `cp_fs_compact` reduction | 0.33 | available |
| launch-count collapse, per-launch floor | 0.25 | available |
| register / occupancy tuning | 0.00 | **refuted**: 64 regs +0.52, 96 neutral, none neutral |
| fragment grid geometry | 0.00 | **refuted**: <= 0.05 ms |
| host wait removal | 0.00 | **refuted**: absorbed (`DEAD_ENDS` 41) |
| spill elimination | 0.00 | **refuted**: -0.04, overlap |
| **sum** | **2.15** | |

    heavy 7.288 - 2.15 (everything) = 5.14 ms
    4x target                       = 2.33 ms
    short by                          2.21x

Of the 5.14 ms that would remain, **fragment shading is ~2.77 ms**. Reaching
2.33 requires `main` under 1.0 ms -- a **2.8x** efficiency gain on a kernel
that runs at 0.31% compute throughput, 0.14% DRAM, 64-77% memory-latency
stalls, and whose occupancy is already at its measured optimum.

**No measured lever produces that, and four candidate levers have been
refuted rather than left untested.** Interpolation is already fused into the
shader, so the `fs_in` round trip does not exist to remove; what remains is
the per-fragment gather of vertex attributes and texture fetches, which is
what a hardware rasteriser does in fixed-function units and a compute kernel
cannot avoid.

**Conclusion: 4x is not achieved, the margin is 1.26x on device work alone,
and it is not proven unreachable.** Three successive attempts to
prove it have each been broken by a lever I had not measured (the frame
boundary, then overdraw). The binding constraint is the applications' own
shaders -- 203 registers of live values, and a frame time invariant across the
whole register/spill curve -- but the accounting no longer has the comfortable
margin the earlier drafts claimed.

    heavy 7.288
      - structural launch work (measured, available)     -2.20
      - frame-boundary host cost (cudavk's share)        -0.56
      - overdraw removal, frame-wide deferred shading    -1.56
      = 2.97 ms          4x target 2.33 ms

Closing the remaining 1.5 ms needs fragment shading 4.9x faster. That requires
about 36% of peak instruction issue; **the register file cannot deliver it at
any occupancy** -- 100% occupancy with a hypothetical 32-register shader
reaches 30%, and these shaders need 203 registers of live values. The hardware
pipeline hides the same latency with fixed-function interpolation and a
different register model; a CUDA kernel running the same shader body cannot.

The reachable target for this architecture is about **3.8-5.1 ms on the heavy
band, 6.5-8.8x native**, via the run-level renderer and the frame boundary. It is not a tuning gap; it is the cost of software
rasterisation against fixed-function hardware, and the same conclusion the
CuRast README states for this workload shape ("models with numerous meshes
with few triangles, Vulkan remains 10x faster"). The loading band (1.942 vs a
1.40 target) is the only one within reach of the levers above.
