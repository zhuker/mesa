# Renderer2 risk 3 settled: what the M1 tile walk costs per reference

`RENDERER2_M1_FANOUT.md` promoted this from third falsifier to **the decisive
gate**: *"the walk must clear the M1 budget with enough margin that
`displaced pool - walk cost - surrender` exceeds 0.500 ms/frame on the
favorite3 heavy band; if it lands within a small factor of entry 24's 3,409
cycles per reference, the design is refuted and M2 must not be written."*

**The walk clears its own gate by a factor of about 129, and the M2
arithmetic still does not robustly clear 0.500 ms/frame.** Both halves of that
sentence are measured below and neither is a judgement call.

Benchmark: `src/cudavk/tests/cp_tilewalk_bench.cu`, standalone, not in
`meson.build`, links nothing from the driver and changes no kernel. Built
`nvcc -O3 -arch=sm_120 -lineinfo`, CUDA 12.8, RTX 5090 (170 SMs, 96 MB L2),
GPU lock held, no other compute process. Three consecutive runs of the final
binary agree to better than 2% on every number quoted; the run reproduced in
the appendix is the first of them.

---

## 1. The answer

**Gate (`RENDERER2_M1_DESIGN.md` section 4.5): at most 28 cycles of block time
per reference at the p99 tile of 12,823 references.**

| arm | cycles of block time per reference | vs 28 | vs entry 24's 3,409 |
|---|---:|---|---:|
| **early-out OFF**, isolated tail block | **26.4 - 26.5** | **passes** | 129x cheaper |
| **early-out OFF**, hottest tile of a realistic scope grid | **26.6 - 28.2** | **passes, at the line** | 124x |
| **early-out ON**, isolated tail block | **7.8 - 7.9** | **passes, 3.5x margin** | 434x |
| early-out OFF, 3,600 blocks all at 12,823 refs | 92 - 95 | fails | 36x |
| early-out ON, 3,600 blocks all at 12,823 refs | 24 - 27 | passes | 130x |

The two rows that matter are the first two. The last two rows are a grid of
46.2 million references -- **145x the reference count of a whole drawing
scope** -- and that regime is memory-bandwidth saturated, not walk-limited:
the load-only calibration kernel, which reads the identical 88 B per reference
and does no edge test, no depth and no atomic, costs **108 cycles per
reference** on the same grid, *more* than the walk itself. No real scope ever
enters it. It is reported because it is the honest ceiling if binning ever
produced that much work.

**Answer to the only question asked: 26.5 cycles of block time per reference
with the early-out disabled, 7.8 with it enabled, against a budget of 28 and
against entry 24's measured 3,409.**

## 2. What the per-scope cost is

Whole 3,600-block grid, one drawing scope's worth of references (318,842 refs
over 1,744 non-empty tiles; section 4 says how that list was built):

| | ms/scope | ms/frame at 2.02 scopes | vs M1's 0.15 ms/scope budget | vs the 0.724 ms/scope it replaces |
|---|---:|---:|---|---:|
| walk, early-out OFF | **0.1262** | 0.255 | passes (84% of budget) | **17.4%** |
| walk, early-out ON | **0.0457** | 0.092 | passes (30% of budget) | **6.3%** |

M1 budgeted the walk at 21% of the exclusive time it replaces and estimated
about 3%. Measured: **17.4% conservative, 6.3% with the early-out** -- inside
the budget in both arms, and between the budget and the estimate.

## 3. The three things this measurement found that the design did not have

### 3.1 The early-out is bit-identical, and that is now measured, not argued

Over the whole 3,600-block grid, **0 of 921,600 resolved pixels differ**
between the arm that looks at every reference and the arm that skips on the
depth ceiling. Section 1.4's order-freedom argument is what makes reordering
legal; this is the empirical half of it. The harness also compares the GPU
visbuf against a host reference walk over 64 tiles: **0 wrong winners**, 147
pixels whose depth key differs by 1-2 ULP with the same primitive winning --
host and device contract the three-term barycentric depth interpolation
differently, and the edge functions, which are pinned by the round-to-nearest
intrinsics, never disagree at all.

### 3.2 The 3-bit slab as section 4.3 specifies it does not separate anything

Section 4.3 buckets each reference by *"the top 3 bits of that key"*, the key
being `float_to_sortable_uint(CP_WINDOW_DEPTH(...))`. For a window depth in
[0, 1] the sortable key of every value in [0.25, 1] begins `101` and every
value in [0, 0.25) begins `100`: **half a uniform depth range lands in one
slab and 8 slabs behave as 2.** The exponent field is what the top bits of a
float are, and depth is not exponentially distributed.

What is used here instead costs no extra bytes: the reference record's 16-bit
field holds the **top 16 bits of the key, truncated**, so
`(near_key_hi << 16) <= true key` and rejecting on it is conservative. The bin
pass orders the list by that field, which is a 65,536-bin counting sort rather
than an 8-bin scatter and is no harder for a device bin pass. The whole-walk
termination is then exact: the head of the current 256-reference chunk bounds
every reference left in the list.

**If a 3-bit slab is kept, bucket on a monotone remap of the depth, never on
the raw exponent bits.**

### 3.3 Ordering is worth about 20%, and the walk's shape is entry 24's shape

At the p99 tile, near-to-far order against submission order:

| order | references processed | longest block |
|---|---:|---:|
| near to far (what the bin pass produces) | **16.2%** | 24.9 - 27.2 cyc/ref |
| submission order (rejection only, no sort) | 20.5% | 29.9 - 31.0 cyc/ref |

Both beat the 28-cycle gate; the sort is worth about a fifth of the walk.
Depth rejection removes **83.8%** of the p99 tile's references under the
uniform-depth assumption of section 4, comfortably above M1's ASSUMPTION of
20-60% -- and that assumption is exactly what the uniform-depth synthesis
decides, so it is reported and **not banked**. The gate is decided by the
early-out-OFF arm, which does not depend on it.

**And the load balance is the thing entry 24 died of.** Nsight Compute on the
realistic mixed grid, early-out OFF: `sm__cycles_active.avg` **58,406**
against `sm__cycles_elapsed.max` **421,388** -- the average SM is busy for
**13.9%** of the kernel, `sm__throughput` is **4.6%** of peak and DRAM is
**3.9%**. The hottest tile's block is 346,128 cycles of a ~362,000-cycle
kernel (0.1262 ms at the measured 2.87 GHz): **one block is 94% of the grid's duration**, exactly the one-block
signature `DEAD_ENDS.md:1247-1255` records for the entry-24 prototype. The
walk is 129x cheaper per reference and has the identical shape.

This matters for one specific claim. `RENDERER2_M1_FANOUT.md` argues the
0.66-0.84 ms/frame fan-out surrender is *"pessimistic by construction"*
because the walk *"replaces the concurrency with 3,600 independent tile blocks
in one grid."* **The measurement does not support that.** The walk's grid is
3,600 blocks wide on paper and one block wide in practice; it fills 13.9% of
the machine. Whatever the walk gives back for the surrendered fan-out, it is
not occupancy.

## 4. Method, and every assumption it rests on

### 4.1 What the kernel is

The inner loop is `cp_rast_small_or_defer()`'s small-triangle path
(`cp_rasterize.cu:880-905`) with the visibility buffer moved to shared memory,
which is what `RENDERER2_M1_DESIGN.md` section 4.1 specifies. Copied verbatim
by hand from `src/cudavk/kernels/` at `229a7a3e0b8`: `PACK_VISBUF` /
`VISBUF_EMPTY` (`cp_rasterize.cu:68-72`), `edge_function` with its
round-to-nearest intrinsics (`:99-106`), `edge_is_top_left` / `edge_inside`
(`:115-129`), `cp_float_to_sortable_uint` and `CP_WINDOW_DEPTH`
(`cp_rast_types.h`), and `struct cp_tri_setup` with its 80-byte
`static_assert`.

Per M1 section 4.2: 16x16 px tile, 3,600-block flat grid, 256 threads, one
thread per reference, `uint64_t vis[256]` = 2,048 B of shared memory resolved
by shared `atomicMin`, `tile_ceiling` refreshed by a 256-lane shared max
reduction every 256 references, no separate coverage mask.

Measured footprint: **40 registers, 2,096 B static shared, 6 blocks/SM, 100%
theoretical occupancy** (2,056 B and the same 6 blocks/SM with the early-out
compiled out). Achieved: `sm__warps_active` **51.3-53.8%** of peak on the
mixed grid, 16.7% for a single block alone on one SM (8 warps of 48).

### 4.2 How the geometry is synthesised, and what that assumes

1. **Coordinates are tile-local (0..16), not screen coordinates.** The walk
   never touches a pixel outside its own tile, so this changes no arithmetic,
   no memory traffic and no atomic contention -- but it lets all 3,600 tiles
   draw from one shared triangle pool, which is what makes the working set
   match a real scope's instead of being 3,600 private copies. What it does
   not model is float magnitude in the edge functions, which costs the same
   either way.
2. **Triangle size targets M0's ~1.7 px mean edge** (`DEAD_ENDS.md:1271-1276`).
   Centres are uniform over [-1.5, 17.5]^2 so a share of triangles straddle
   the tile edge and are bbox-clipped, which is where M0's 1.16 references per
   triangle comes from. The generator **prints what it achieved** rather than
   asserting it: mean edge **1.591 px**, mean area **1.086 px^2**, and
   **9.43 pixels tested per reference** -- that last number is larger than M1
   section 4.5's "1-4 iterations" because the driver's own `setup_triangle()`
   takes `floorf(min)` to `ceilf(max)`, which is a 3x3 box for a 1.6 px
   triangle. Measured coverage is **0.83 covered pixels per reference**, i.e.
   0.83 shared atomics per reference.
3. **Depth is uniform in [0, 1] per triangle**, +-0.002 across the vertices,
   `depth_scale` 1 and `depth_translate` 0. Uniform is the neutral choice: it
   is neither front-to-back submission, which flatters the early-out, nor
   back-to-front, which ruins it. **The gate is decided by the early-out-OFF
   arm, which does not depend on this at all.**
4. **The pool is 176,000 triangles = 13.4 MB**, which is one drawing scope of
   post-clip triangles (M0: ~354k/frame at ~2.02 drawing scopes/frame). Every
   tile indexes into it uniformly at random. This reproduces the real cache
   situation -- a scope's setup array is L2-resident and re-read by many tiles
   -- rather than a synthetic streaming one.
5. **The mixed grid is one drawing scope**: lengths are lognormal(3.466,
   1.879), fitted so the median non-empty tile lands in M0's peak log2 bucket
   (16-63) and the maximum lands on M0's `refs_max` p99 of 12,823, over 1,744
   non-empty tiles of 3,600. Achieved: p50 32, p90 376, p99 2,712, max 12,823,
   **318,842 references per scope**. The remaining 1,856 tiles exit at once,
   which M1 section 4.2 prices at under 0.2 ns per idle block; measured here,
   an empty 3,600 x 256 grid is **0.0041 ms and 10.6 block cycles**.

### 4.3 Memory traffic, measured rather than budgeted

M1 section 4.4 budgets **77 B read per reference** (8 B record + 80 B setup
amortised over 1.16 references per triangle). This benchmark gives each
reference its own setup load, so it requests **88 B**; the SASS confirms five
`LDG.E.128` per reference in both the walk and the calibration kernel.

Nsight Compute measures **1.70 MB for 12,823 references = 132 B per
reference** at L1TEX. The gap is sector granularity: an 80 B struct at an 80 B
stride straddles 32 B sectors, so 80 B wanted fetches 96-128 B.
**Padding `cp_tri_setup` to 96 B and aligning it would remove about a third of
the walk's read traffic**, and 176k x 96 B = 16.9 MB still fits L2. This is
free and is not in the design.

### 4.4 Instruction count

Isolated tail block, early-out OFF, 12,823 references in 401 full warps:
**387,648 warp-instructions = 30.2 per reference**, against M1 section 4.5's
estimate of "~40 instructions". The estimate was right; the *conclusion* drawn
from it was not. M1 predicted the p99 block at 8,016 cycles issue-bound or
10,300 memory-bound, i.e. **~0.8 cycles of block time per reference**.
Measured: **339,297 cycles, 26.5 per reference -- 33x the estimate.** The
35-fold margin between 0.8 and 28 that section 4.5 said existed "to absorb the
things this arithmetic does not model" was almost entirely consumed by them.
The gate is passed; the estimate is not vindicated.

### 4.5 Harness sanity, so a wrong number is detectable

- **Empty kernel**, 3,600 x 256: 0.0041 ms per launch, 10.6 block cycles mean.
  Nothing in the walk numbers is launch overhead.
- **Load-only floor**, same block shape, same list, same 88 B per reference,
  no edges, no depth, no atomic: 108 cycles per reference at the saturated
  homogeneous p99 grid, against the walk's 92-95. The walk *is* the memory
  floor in that regime, which is what says it is bandwidth-bound there and
  that its arithmetic is free.
  **This calibration was wrong on the first attempt and the check caught it**:
  with only five of the twenty words consumed, nvcc dropped the vector loads
  and issued 92 scalar `LDG`s, reading a third of the bytes over four times
  the transactions -- making the "floor" *slower* than the walk it bounds. The
  fix consumes all twenty words; `cuobjdump -sass | grep -c LDG.E.128` is the
  check, and it must show the same five per reference as the walk.
- **SM clock measured under load**, not read from `nvidia-smi` at idle:
  **2.87-2.88 GHz**, against the 2.4 GHz M1 section 4.5 assumed. The gate
  stated in cycles is therefore about 20% conservative -- 0.15 ms/scope at
  2.88 GHz is 33.7 cycles per reference, not 28. **The 28 is used anyway.**
- **Correctness**, section 3.1 above.

### 4.6 What this benchmark does not measure

- **The bin pass.** Entry 24 priced binning at "about 0.13 ms"
  (`DEAD_ENDS.md:1285-1290`) and that number is carried below unmeasured. The
  bin pass here must also produce the ordered reference lists (section 3.2),
  which entry 24's did not.
- **Shading, FS bucketing, the survivor-list write, the blended walk, the
  handoff.** The walk writes 2,048 B per tile and nothing else, as designed.
- **Any driver integration cost**, and **the device-time to frame-time slope**,
  which `RENDERER2_M1_DESIGN.md` section 5.1 reserves for M4.
- **B200.** These are sm_120 facts on one RTX 5090, and entry 25's closing
  warning applies verbatim: re-run before believing them on the target.

---

## 5. The M2 arithmetic, done with the measured number

Inputs: displaced union-exclusive pool **1.4623 ms/frame**
(`RENDERER2_M1_DESIGN.md` section 5.1), fan-out surrender **0.66-0.84
ms/frame** (`RENDERER2_M1_FANOUT.md`), bin pass **~0.13 ms/frame** (entry 24,
unmeasured here), walk **0.092-0.255 ms/frame** (section 2 above, the range
being early-out ON to early-out OFF).

| | best corner | worst corner | midpoint |
|---|---:|---:|---:|
| displaced | 1.4623 | 1.4623 | 1.4623 |
| less walk | -0.092 | -0.255 | -0.174 |
| less bin pass | -0.13 | -0.13 | -0.13 |
| less fan-out surrender | -0.66 | -0.84 | -0.75 |
| **net ms/frame** | **0.580** | **0.237** | **0.408** |

Without the bin pass counted at all -- the most favourable reading available:
**0.710 / 0.367, midpoint 0.538.**

**Against the 0.500 ms/frame admission gate
(`RESIDUAL_AUDITS_2026-08-31.md:6`) this does not clear.** It clears in the
best corner and fails in the worst, and the midpoint is **0.408** -- below the
gate. The walk is not what puts it there: even at zero walk cost the midpoint
is 0.58, and even so the worst corner is 0.49.

Two further pressures are on the same side, and both are already measured
elsewhere in this milestone:

1. **`RENDERER2_M1_RUNS.md` redefined the M2 slice to opaque runs, which carry
   95% of references but are not the whole displaced pool.** Of the 1.4623,
   the three A-buffer entries (`stage3_abuf` 0.0926, `stage2_abuf` 0.0262,
   `cp_abuf_*` 0.1944 = **0.3132**) serve blended runs, which an opaque-only
   M2 leaves on the existing chain. An M2-slice-only displaced pool is
   therefore closer to **1.15**, and the midpoint net falls to about **0.10**.
   The full 1.4623 is only collectible once the blended walk exists too.
2. **The fan-out surrender's "pessimistic by construction" defence is not
   supported by this measurement** (section 3.3): the walk's grid runs at
   13.9% average SM activity, so it does not restore the concurrency it takes
   away.

## 6. Verdict

**Risk 3 is settled in the design's favour and it is not enough.**

- The walk costs **26.5 cycles of block time per reference** with every
  reference processed and **7.8** with the early-out on, against a budget of
  28 and against entry 24's 3,409. **The design is not refuted on walk cost.**
  Entry 24's failure was the one-block-per-reference shape and the CTA barrier
  in front of thread 0's setup, and inverting the mapping removes it -- that
  much of `RENDERER2_M1_DESIGN.md` section 4.1 is now measured, not argued.
- **The M2 arithmetic does not robustly clear 0.500 ms/frame.** Midpoint
  0.408 with the bin pass, 0.538 without it, and about 0.10 if M2 is the
  opaque-only slice `RENDERER2_M1_RUNS.md` settled on. It clears only in the
  favourable corner of three independent uncertainties at once.

**What would change the answer, in the order it should be attempted:**

1. **Measure the bin pass, and measure the fan-out surrender against a real
   walk rather than against `CUDAVK_NO_OPAQUE_STREAMS`.** Those two terms are
   0.79-0.97 of the 1.46 and only one of them was measured for this design at
   all. They, not the walk, decide the milestone.
2. **Fix the load balance before believing any per-scope number scales.** One
   block is 94% of the walk grid's duration. Splitting the hot tile's list
   across several blocks with a second-pass merge is the obvious move and the
   walk is cheap enough to afford it -- but it is new design, and entry 24's
   retry-if is explicit that the ceiling must exceed the surrender before more
   is built.
3. Only then: `cp_tri_setup` padded to 96 B (section 4.3), and the slab fix
   (section 3.2).

---

## Appendix: the run reproduced

```
device: NVIDIA GeForce RTX 5090  SMs 170  L2 96.0 MB  clockRate(nominal) 2.407 GHz  sm_120
measured SM clock during a busy kernel: 2.873 GHz (300000019 cycles in 104.4107 ms)
cp_tile_walk<early=on> : 40 regs, 2096 B static shared, 6 blocks/SM, 100.0% theoretical occupancy
cp_tile_walk<early=off>: 40 regs, 2056 B static shared, 6 blocks/SM, 100.0% theoretical occupancy
pool: 176000 triangles, 13.43 MB; measured mean edge 1.591 px, mean area 1.086 px^2, mean bbox 9.43 px tested per reference

CALIBRATION
  empty 3600x256 kernel: wall 0.0041 ms/launch, block cycles mean 10.5 max 34
  88 B/ref load-only floor, 3600 x 12,823: wall 1.2764 ms/launch, longest block 1377884 cyc = 107.45 cyc/ref, mean 73.08 cyc/ref

HOMOGENEOUS GRID (all 3,600 blocks at the same list length)
 p50 hottest tile   237
  early-out OFF (near-to-far list)   refs    853200  longest block    34184 cyc over    237 refs =  144.24 cyc/ref | weighted mean  76.59 cyc/ref | wall   0.0310 ms | processed 100.0% | covered 0.831 px/ref
  early-out ON  (near-to-far list)   refs    853200  longest block    35929 cyc over    237 refs =  151.60 cyc/ref | weighted mean  80.55 cyc/ref | wall   0.0328 ms | processed 100.0% | covered 0.831 px/ref
  early-out ON  (submission order)   refs    853200  longest block    36787 cyc over    237 refs =  155.22 cyc/ref | weighted mean  80.66 cyc/ref | wall   0.0328 ms | processed 100.0% | covered 0.831 px/ref
 p90              9,208
  early-out OFF (near-to-far list)   refs  33148800  longest block   861056 cyc over   9208 refs =   93.51 cyc/ref | weighted mean  71.49 cyc/ref | wall   0.9172 ms | processed 100.0% | covered 0.832 px/ref
  early-out ON  (near-to-far list)   refs  33148800  longest block   297530 cyc over   9208 refs =   32.31 cyc/ref | weighted mean  16.73 cyc/ref | wall   0.2502 ms | processed  22.4% | covered 0.187 px/ref
  early-out ON  (submission order)   refs  33148800  longest block   355320 cyc over   9208 refs =   38.59 cyc/ref | weighted mean  20.56 cyc/ref | wall   0.3221 ms | processed  28.4% | covered 0.236 px/ref
 p99             12,823
  early-out OFF (near-to-far list)   refs  46162800  longest block  1209589 cyc over  12823 refs =   94.33 cyc/ref | weighted mean  72.23 cyc/ref | wall   1.2688 ms | processed 100.0% | covered 0.832 px/ref
  early-out ON  (near-to-far list)   refs  46162800  longest block   333648 cyc over  12823 refs =   26.02 cyc/ref | weighted mean  12.41 cyc/ref | wall   0.2462 ms | processed  16.2% | covered 0.135 px/ref
  early-out ON  (submission order)   refs  46162800  longest block   382687 cyc over  12823 refs =   29.84 cyc/ref | weighted mean  14.87 cyc/ref | wall   0.3202 ms | processed  20.5% | covered 0.170 px/ref
 max             14,814
  early-out OFF (near-to-far list)   refs  53330400  longest block  1370743 cyc over  14814 refs =   92.53 cyc/ref | weighted mean  71.87 cyc/ref | wall   1.4626 ms | processed 100.0% | covered 0.832 px/ref
  early-out ON  (near-to-far list)   refs  53330400  longest block   319519 cyc over  14814 refs =   21.57 cyc/ref | weighted mean  10.61 cyc/ref | wall   0.2498 ms | processed  14.2% | covered 0.118 px/ref
  early-out ON  (submission order)   refs  53330400  longest block   354821 cyc over  14814 refs =   23.95 cyc/ref | weighted mean  12.82 cyc/ref | wall   0.3199 ms | processed  17.7% | covered 0.147 px/ref

ISOLATED TAIL BLOCK (1 block of 12,823 refs, machine otherwise idle)
  early-out OFF                      refs     12823  longest block   339678 cyc over  12823 refs =   26.49 cyc/ref | weighted mean  26.49 cyc/ref | wall   0.1209 ms | processed 100.0% | covered 0.837 px/ref
  early-out ON                       refs     12823  longest block   100563 cyc over  12823 refs =    7.84 cyc/ref | weighted mean   7.84 cyc/ref | wall   0.0369 ms | processed  20.0% | covered 0.165 px/ref

MIXED GRID (one drawing scope: lognormal tile lists, 1,744 of 3,600 non-empty)
  non-empty tiles 1744, total refs 318842, mean 182.8, p50 32, p90 376, p99 2712, max 12823
  HARNESS CHECK
  visbuf: early-out ON vs OFF -> 0 of 921600 pixels differ
  visbuf: GPU vs host reference over 64 tiles -> 0 wrong winners, 147 same-winner keys differing (max 2 ULP), 1476 host coverage events  OK
  early-out OFF                      refs    318842  longest block   346128 cyc over  12823 refs =   26.99 cyc/ref | weighted mean  63.36 cyc/ref | wall   0.1262 ms | processed 100.0% | covered 0.831 px/ref
  early-out ON                       refs    318842  longest block   119406 cyc over   3684 refs =   32.41 cyc/ref | weighted mean  79.01 cyc/ref | wall   0.0455 ms | processed  79.7% | covered 0.663 px/ref

PER-SCOPE ARITHMETIC (M1 section 4.6: budget 0.15 ms/scope, 21% of the 0.724 ms/scope exclusive it replaces)
  walk kernel, one scope, early-out OFF : 0.1262 ms/scope  (17.4% of 0.724)
  walk kernel, one scope, early-out ON  : 0.0455 ms/scope  (6.3% of 0.724)
  at 2.02 drawing scopes/frame          : 0.2549 / 0.0920 ms/frame

GATE (M1 section 4.5): <= 28 cycles of block time per reference at the p99 tile (12,823 refs).
Entry 24's prototype: 3,409 cycles/reference.
```
