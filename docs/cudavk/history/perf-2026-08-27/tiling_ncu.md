# Why the opaque sort-middle tiling prototype lost — Nsight Compute, the census, and the verdict

Measured 2026-08-26 on the reference machine (RTX 5090, 170 SMs, driver 580.173.02,
SM clock 2.01 GHz during the runs, CUDA 12.8, `ncu` from `/usr/local/cuda/bin`).
Tree `/home/alexzhukov/mesa` at `e2fea470d04`, build `build-cudavk`, working tree
untouched and nothing committed. Raw output: `/tmp/perf-audit/tiling/`.

This is the measurement `docs/cudavk/notes/OPAQUE_TILING_PROTOTYPE.md` names as the
required next step ("use Nsight Compute on `cp_opaque_tile_raster` to measure executed
instructions, branch efficiency, barrier stalls, achieved occupancy, register count,
and memory throughput"). It had never been run.

---

## 0. Verdict first

1. **The document's recorded explanation is wrong in both halves.** It is *not* the
   1024-pixel coverage loop that sets the kernel's time, and it is *not* divergence.
   The kernel's duration is set by **one thread block**: `sm__cycles_active.max` is
   **99.7% of elapsed** while the average SM is active **22%** and the least busy SM
   **0.1%**. Branch efficiency is **94.3%**.
2. **The failure is kernel structure and work distribution, not binning precision.**
   Of the six successor directions, the evidence **requires 4 and 5**, **kills 3**,
   makes 1, 2 and 6 secondary (they reduce work but not duration), and **adds a
   seventh the list does not contain**: per-tile occlusion. The long lists are
   *depth complexity*, not big triangles.
3. **A v2 is not worth designing yet.** Its device-side ceiling on the old capture is
   about **2.1–2.6 ms/frame of kernel time**, and the tiled arm must give up the opaque
   stream fan-out, which is already banked at **2.73 ms/frame** (`7f38d2a9b65`,
   PERFORMANCE §7). The ceiling is not larger than the mechanism it must surrender.
4. **The most valuable finding is not about tiling.** `cp_rasterize_stage3_abuf`, the
   largest kernel class in the *shipping* driver at 2.484 ms/frame, has **the same
   one-block tail**: median `sm__cycles_active.max` = **93.9% of elapsed** while the
   median average SM is active **16.4%**. The disease is in the code that ships today.
5. **`CUDAVK_TILE_CENSUS` has never been able to report anything.** See §6.

---

## 1. Nsight Compute on `cp_opaque_tile_raster`

Sample `multithreading` (the one the document says amplifies the failure), offscreen
benchmark, `CUDAVK_TILED_OPAQUE=1`. Framebuffer 1280x736, so 40x23 = **920 tiles**,
one 256-thread block each, one launch per frame.

No permission problem: `RmProfilingAdminOnly: 0`, and `ncu` returned counters on the
first attempt. `src/cudavk/tests/cp_profile.sh` has an `NCU=1` mode and drives a
sample binary fine (the note about not wrapping `gfxrecon-replay` did not apply); the
numbers below were taken by calling `ncu` directly so the kernel filter and launch
count could be set per question.

### 1.1 Full counter set (`--set full`, one launch)

| metric | value |
|---|---:|
| Duration | 54.92 ms (44.65 ms without the profiler, see §3) |
| Grid / block | 920 x 256 |
| Registers per thread | **54** |
| Static shared memory per block | **436 B** |
| Waves per SM | 1.35 |
| Theoretical occupancy | 66.7% (limited by registers, 4 blocks/SM) |
| **Achieved occupancy** | **18.0%** (8.64 active warps per SM of 48) |
| Executed instructions | **1,151,423,905** warp-instructions |
| Compute (SM) throughput | **2.53%** |
| Memory throughput / DRAM | 2.53% / **0.23%** (4.03 GB/s) |
| L1 hit rate / L2 hit rate | 92.5% / 5.1% |
| **Branch efficiency** | **94.32%** |
| Avg active threads per warp | 21.76 of 32 |
| Warp cycles per issued instruction | 31.24 |
| **of which stalled at the CTA barrier** | **22.9 cycles = 73.3%** |
| Issue slots busy | 6.91% (one instruction every 14.5 cycles per scheduler) |

Nothing here is a roofline story. The kernel is 2.5% of compute peak and 0.2% of DRAM
peak. It is latency and serialisation.

### 1.2 The decisive measurement: per-SM activity, single pass, three launches

`--metrics sm__cycles_active.{max,min,avg},gpc__cycles_elapsed.max,smsp__inst_executed.sum`,
one pass per launch (no kernel replay), three consecutive launches:

| launch | elapsed (gpc) | sm active MAX | sm active AVG | sm active MIN | inst executed |
|---|---:|---:|---:|---:|---:|
| 1 | 110,798,012 | 110,526,769 (**99.75%**) | 24,523,914 (22.1%) | 110,797 (**0.10%**) | 1,151,423,905 |
| 2 | 113,110,607 | 112,871,175 (99.79%) | 24,506,410 (21.7%) | 108,825 (0.10%) | 1,151,599,125 |
| 3 | 114,308,576 | 114,030,511 (99.76%) | 24,529,785 (21.5%) | 153,794 (0.13%) | 1,153,145,013 |

**One SM is busy for the whole kernel. The least busy SM is busy for one thousandth of
it.** 920 blocks on 170 SMs is 1.35 waves; the machine drains in the first fifth and
then waits for a single tile.

### 1.3 Comparison with the default driver's largest class

`cp_rasterize_stage3_abuf`, old capture replay, default driver, same counter set,
eight consecutive launches after `--launch-skip 4000`:

| grid | elapsed | sm max | sm avg | max/elapsed | avg/elapsed | inst | µs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 | 37,580 | 34,646 | 3,745 | 92.2% | 10.0% | 37,968 | 18.7 |
| 512 | 54,872 | 51,944 | 3,289 | 94.7% | 6.0% | 86,684 | 27.3 |
| 1408 | 187,073 | 183,889 | 170,873 | 98.3% | 91.3% | 7,390,821 | 93.1 |
| 1024 | 16,256 | 12,187 | 5,280 | 75.0% | 32.5% | 136,886 | 8.1 |
| 2048 | 151,946 | 147,974 | 115,638 | 97.4% | 76.1% | 5,809,328 | 75.6 |
| 512 | 70,501 | 66,709 | 3,616 | 94.6% | 5.1% | 98,199 | 35.1 |
| 640 | 5,793 | 1,416 | 1,326 | 24.4% | 22.9% | 15,360 | 2.9 |
| 512 | 57,137 | 53,246 | 2,084 | 93.2% | 3.6% | 32,654 | 28.4 |

**Median max/elapsed 93.9%, median avg/elapsed 16.4%.** The shipping driver's largest
kernel class has the same defect. Read the second row: 512 blocks, 86,684 warp
instructions in total — about 170 instructions per block — and it still takes 27 µs,
because one block runs for 94.7% of that. `cp_rasterize_stage3` on `multithreading`
behaves the same way (elapsed 80,678 cycles, SM active avg 18,598 = 23%, achieved
occupancy 6.6%).

So the answer to "is the tiled kernel's problem the same one the default already has"
is **yes, the same disease, about 100x larger**: 55 ms of one-block tail instead of
30 µs.

---

## 2. The census: the reference distribution, and what it kills

`CUDAVK_TILED_OPAQUE_CENSUS` prints only tile dimensions, segment count and the
overflow word, which does not answer the question. `CUDAVK_TILE_CENSUS=<edge>` is the
instrument that does — and it cannot report at this commit (§6). With the reducer
called (a one-line fix in a **/tmp copy of the tree**, never in the working tree) and
the raw per-tile arrays dumped, the distribution is exact. Note that the binner adds
**at most one reference per (triangle, tile) pair**, so *a tile's list length is the
number of distinct triangles overlapping that tile*.

### 2.1 `multithreading`, one opaque episode = one `cp_opaque_tile_raster` launch

| tile edge | grid | refs total | non-empty tiles | median list | p90 | p99 | **max list** | visible frags | (refs x tile area)/visible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 80x45 = 3600 | 1,190,127 | 1,373 | 197 | 2,742 | 6,114 | **14,859** | 214,066 | 1,423 |
| **32** | 40x23 = 920 | **1,074,095** | 411 | **557** | 9,504 | 16,708 | **26,297** | 214,066 | **5,138** |
| 64 | 20x12 = 240 | 1,027,161 | 126 | 1,796 | 23,650 | 56,940 | **67,781** | 214,066 | 19,654 |

At the shipped 32x32: **max/median = 47x, max/mean = 10x**, and only 411 of 920 tiles
carry anything. The hottest tile is at pixel (704, 352) — screen centre — and holds
**26,297 distinct triangles**, of which **563 pixels** of its 1024 end up visible.

**The triangles are tiny, not large.** Total references barely move with tile size
(1.19 M at 16 px, 1.03 M at 64 px, a ratio of 1.16). If triangles spanned many tiles
the 16-px count would be several times the 64-px count. Solving
`((s/16+1)/(s/64+1))^2 = 1.16` gives a mean triangle edge of **about 1.7 pixels**.
So the binner puts a 1.7-pixel triangle into one 32x32 tile and the consumer then
tests **1024 pixel candidates** for it.

**This is the answer to the parent's critical distinction: many triangles landing in
one tile, not one triangle referenced into many tiles.** References per triangle are
~1.07 at 32 px. No binning policy can shorten a list that is the count of distinct
triangles over a tile.

### 2.2 Old capture (`headless_streamer_20260814T155742.gfxr`), 32x32, all 10,033 opaque episodes

| quantity | median | mean | p99 | max |
|---|---:|---:|---:|---:|
| references per episode | 3,559 | 27,921 | 152,653 | 162,798 |
| **longest tile list per episode** | **665** | **1,138** | 4,594 | 5,994 |
| median tile list per episode | 57 | — | — | — |
| non-empty tiles (of 920) | — | 282 | — | — |
| max/median list ratio | 6.6x | 14.3x | — | — |
| (refs x 1024)/visible fragments | 91 | 2,097 | — | — |

6.64 opaque episodes per frame, 185,394 references per frame.

---

## 3. The cost model, and it predicts the recorded regression

From §1.2 and §2.1: elapsed 110.8 M cycles for a longest list of 26,297 references
gives **4,213 cycles per reference** under the profiler; the unprofiled kernel is
44.59 ms/frame (nsys, 10 frames), i.e. **3,409 cycles per reference**.

Two independent checks:

* **Cross-check against the average SM.** Total block work = 1,074,095 refs x 4,213
  cycles = 4.53e9 SM-cycles; spread over 170 SMs that is 26.6 M cycles, against the
  measured `sm__cycles_active.avg` of 24.5 M. **9% agreement.** The model is right and
  the duration really is `longest list x per-reference cost`.
* **Cross-check on a workload it was not fitted to.** Applying 3,409 cycles per
  reference to the old capture's measured per-episode longest lists gives
  **12.8 ms/frame** of tile raster. The recorded 2026-08-18 regression on that capture
  is **+10.38 ms/frame** (25.23 → 35.61). The model over-predicts by 23%, which is
  within the overlap the side streams provide. **The regression is explained
  quantitatively by one number per episode: the longest tile list.**

Where the 3,409–4,213 cycles per reference go:

| term | value |
|---|---:|
| warp-instructions per reference | 1,072 |
| issue-limited floor for one block on one SM (4 schedulers) | **268 cycles** |
| measured | 4,213 cycles |
| **issue efficiency** | **6.4%** |
| stall cycles at the CTA barrier | 73.3% |

Per reference the kernel makes **thread 0 alone** copy `cp_rasterize_args` into shared
memory (the 436 B static shared block) and run `setup_triangle()`, with 255 threads
parked at `__syncthreads()`, twice. That serial section, not the pixel loop, is where
94% of the per-reference time goes.

### The three factors, separated

For `multithreading`, per frame:

| configuration | tile-raster cost | factor removed |
|---|---:|---|
| as built | **44.59 ms** | — |
| perfect distribution only (same work, same per-ref cost, spread over 170 SMs) | **13.2 ms** | the one-block tail, ~3.4x |
| + per-reference cost at the issue floor (268 cycles) | **0.84 ms** | the serial setup and barriers, ~16x |
| what it replaces (classic raster displaced, measured by nsys) | **1.60 ms** | — |

For the old capture, per frame: as built ~12.8 ms (modelled) / +10.38 ms (measured);
perfect distribution only **1.85 ms**; distribution and per-reference cost fixed
**0.145 ms**; what it could replace, the direct raster chain minus the clip share,
**about 2.87 ms**.

---

## 4. Which of the six directions the evidence supports, and which it kills

The document's list, judged against the measurements:

| # | direction | verdict |
|---|---|---|
| 1 | coarse regions then compact 8x8 fine tiles | **Secondary.** Cuts candidates per reference 16x, but the tail survives: at 16x16 the longest list is still 14,859 against a median of 197 (75x). Reduces work, not duration. |
| 2 | exact triangle/tile rejection or stored coverage masks at fill | **Secondary, and cheaper than it looks.** With a 1.7-pixel mean triangle, an exact mask makes almost every reference a handful of pixels instead of 1024. It attacks the 5,138:1 candidate ratio — the instruction count — and not the tail. |
| 3 | separate small triangles, keep the classic path for them | **KILLED.** References per triangle are 1.07 at 32 px and the mean triangle is ~1.7 px across. There is no large-triangle population to split off; "keep classic for the small ones" keeps classic for essentially all geometry, which is the flag-off driver. |
| 4 | compact active tiles, group references into warp-sized work units | **REQUIRED.** The only direction on the list that touches the one number that is the kernel's duration. 411 of 920 blocks are empty and one block holds 26,297 references. |
| 5 | precompute compact edge/depth coefficients per reference | **REQUIRED.** 4,213 cycles per reference against a 268-cycle issue floor, 73.3% of stall cycles at the barrier, caused by thread 0 re-running `setup_triangle()` and copying `cp_rasterize_args` into shared memory once per reference. |
| 6 | reconsider tile size from measured distributions | **Now measurable, mildly supported.** 16x16 costs 11% more references (1.19 M against 1.07 M) and cuts candidates per reference 4x; 64x64 is clearly worse (longest list 67,781). But the tail ratio is worse at 16 px (75x) than at 32 px (47x), so tile size alone is not the fix. |

**And one the list does not contain — call it 7: per-tile occlusion.** The hot tile
holds 26,297 distinct triangles and resolves 563 visible pixels. That list length is
depth complexity, and it is invariant to every binning choice above. The classic
sort-middle answers are a per-tile z-max maintained during the walk with early
rejection, or ordering references so the near ones resolve first. Nobody has costed
this, and it is the only mechanism that shortens the list itself.

**Classification asked for in the brief:** this is a **kernel-structure failure
(directions 4 and 5)** with a large secondary work-amount term (directions 1, 2, 6).
It is **not** a binning-policy failure in the sense the document means — the binner
puts each triangle in about one tile, exactly as intended.

---

## 5. Is a v2 worth designing? A number, and the answer is no — not yet

The optimistic case, old capture, per frame:

* cost of a v2 that fixes both required directions: **0.15–0.6 ms** of tile raster
  (0.145 ms at the issue floor, times a realistic 1–4x for real issue efficiency) plus
  about **0.13 ms** of count/fill/scan (measured 0.77 ms/frame on `multithreading`,
  scaled by references per frame: 185 k against 1.07 M).
* what it can displace: **about 2.87 ms** — the whole direct raster chain
  (`stage1 direct` 1.894 + `stage2 direct` 0.547 + `stage3 direct` 0.767, less the
  clip share of the fused stage 1, measured at 17.6% on `multithreading`).
* **net device-side ceiling: about 2.1–2.6 ms/frame of kernel time**, 14–18% of the
  14.52 ms/frame kernel total.

Three reasons that ceiling does not justify the work:

1. **It is smaller than what the tiled arm has to give up.** `7f38d2a9b65` refuses the
   opaque stream fan-out under `CUDAVK_TILED_OPAQUE`, and the fan-out is worth
   **+2.73 ms/frame on old** (PERFORMANCE §7). A v2 starts 2.73 ms behind and its whole
   best case is 2.1–2.6 ms. It would have to re-earn the fan-out's overlap from its own
   width before the first millisecond of tiling gain counts.
2. **Kernel time does not convert to frame time here.** PERFORMANCE §5.2/§7: the host
   is blocked 73.7% of the frame, the device is idle 28%, "device-side removal is
   exhausted", and both mechanisms that paid (fan-out, PDL) worked by *overlap*, not by
   removing work. A 2.4 ms kernel-time saving plausibly converts to well under half of
   that in frame time.
3. **The same defect is available for far less money elsewhere.** `stage3_abuf` is
   2.484 ms/frame today, at a median one-SM-active fraction of 93.9% against an average
   of 16.4%. Attacking the tail *inside the kernel that already ships* needs no new
   architecture, no new binning, no new memory, and no fan-out sacrifice. **That is the
   iteration to run next**, and this session's counters are the evidence for it.

**Recommendation: do not build tiling v2. Keep the flag off. Re-open the question only
if the tail fix lands in the classic raster chain and the mechanism is proven there.**

---

## 6. Reportable defect: the tile census cannot report

`cp_tile_census_end_pass()` is defined at `src/cudavk/cp_renderer.c:7998` and declared
at `src/cudavk/cp_renderer.h:653`, and **it is called nowhere in the tree**. The only
other call to `cp_tile_census_reduce_pass()` is at `cp_renderer.c:7676`, inside
`cp_tile_census_begin()`, and fires only when the framebuffer changes size inside one
bind. Consequently `CUDAVK_TILE_CENSUS=<edge>` accumulates per-tile counts and then
never reduces or prints them on any fixed-size workload — which is every sample and
both captures. Verified: `CUDAVK_TILE_CENSUS=16/32/64` with
`CUDAVK_TILE_CENSUS_EVERY=60` on `multithreading` produced **no output at all**.

The prototype's own instrumentation was therefore unavailable to the people who
abandoned it; the census section of `OPAQUE_TILING_PROTOTYPE.md` ("Implementation, 1.
Census") has never been executed. The one-line fix used here (call the reducer at the
end of each opaque episode) lives only in `/tmp/perf-audit/tiling/mesa` and is **not**
proposed for the tree in this session.

---

## 7. GPU exclusivity, stated positively

`/tmp/perf-audit/gate/gpu_watch.sh` sampled compute processes at 1 Hz for the whole
session: **1,006 samples over 1,024 s**. 226 samples named our own processes
(`multithreading`, `gfxrecon-replay`), 748 were empty, and **32 samples caught a
foreign tenant** — `build_vp9_cuda/vpxenc`, `build_cuda/vpxenc` and
`/tmp/bins/vpxenc_s9`, ~1,000 MiB each — in three bursts at t+5..7 s, t+129..135 s and
t+252..273 s. The libvpx batch is real and a point check would have missed it.

**Every headline number in this report was taken after t+273 s, and in that window
every non-empty sample named only our own processes — no foreign process appears in
the log again.** Specifically: the per-SM tiled measurement at t+380 s, the
`stage3_abuf` per-SM measurement at t+537 s, the `multithreading` census at t+615 s,
the nsys pairs at t+757/760 s, and the full old-capture census at t+949 s.

Independently of the sampler: a time-slicing tenant *cannot* produce a sustained
`sm__cycles_active.max` equal to 99.7% of elapsed cycles on our own kernel, because a
foreign context's slices would idle every SM. The headline result is robust to the
window question by construction.

Anyone timing on this machine after us needs the sampler, not a point check.

---

## 8. Artifacts

```
/tmp/perf-audit/tiling/ncu/A_tile_raster.txt     ncu --set full, cp_opaque_tile_raster
/tmp/perf-audit/tiling/ncu/A2_persm.txt          per-SM activity, 3 launches, 1 pass each
/tmp/perf-audit/tiling/ncu/B_stage3_direct.txt   ncu --set full, cp_rasterize_stage3 (multithreading)
/tmp/perf-audit/tiling/ncu/C_stage3_abuf.txt     ncu --set full, cp_rasterize_stage3_abuf (old capture)
/tmp/perf-audit/tiling/ncu/C2_abuf_persm.txt     per-SM activity, 8 launches, 1 pass each
/tmp/perf-audit/tiling/refs_{16,32,64}.bin       raw per-tile refs/visible arrays, multithreading
/tmp/perf-audit/tiling/refs_old32_full.bin       the same for all 10,033 old-capture opaque episodes
/tmp/perf-audit/tiling/logs/census2_*.txt        tilecensus reports (patched build)
/tmp/perf-audit/tiling/logs/nsys_mt_*.txt        nsys kernel summaries, default and tiled arms
/tmp/perf-audit/tiling/logs/planstats_old.txt    CUDAVK_PLAN_STATS on the old capture
/tmp/perf-audit/tiling/gpu_watch.log             1 Hz compute-process sampler, whole session
/tmp/perf-audit/tiling/mesa/                     throw-away copy of the tree with the census fix
```
