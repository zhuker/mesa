# Redesign plan: the run-level renderer

Written 2026-09-05, after the 2026-09-02/03 leads landed (favorite3 **5.940**
ms, favorite2 **5.022** on the RTX 5090; **9.883 / 8.283** on the B200) and
after the first measurement of the floor: NVIDIA's own driver executes the
same submits in **0.522 / 0.502 ms** (`NATIVE_DRIVER_REFERENCE.md`). cudavk is
11x off a real driver on the same GPU, the arithmetic in a frame is 0.3-0.7
ms of device time (`PERF_ANALYSIS_2026-09-02.md` §1.2), and every incremental
lead inside the current architecture has now been built or measured out
(`LEAD_RESULTS_2026-09-02.md`, dead ends 39-43). What is left is the shape of
the renderer, and this document is the plan for changing it.

It is a re-charter of Renderer 2 (`redesign/tilewalk`, notes
`RENDERER2_{DESIGN,STATE_SURVEY,M0_RESULTS,M1_DESIGN,M1_FANOUT,M1_RUNS,
M1_WALKBENCH,VERDICT}.md`), which was refused at M1 on arithmetic. §6 goes
through all 43 dead ends one by one and says where this plan collides with
one and why it is worth trying anyway. The short version: Renderer 2 kept one
vertex, clip and fragment launch per *batch* and charged the fan-out surrender
in full; this plan removes the per-batch launch structure, which is where the
verdict's own "undisplaced pools" were, and needs no fan-out to surrender.

Labels as elsewhere: **MEASURED**, **MODEL**, **ESTIMATED**, **HYPOTHESIS**.
Conventions: `WORKFLOW.md` §3.1 and §4.

---

## 1. What the frame is, and why shaving it stopped paying

Heavy band, RTX, post-leads (MEASURED unless marked):

- **~700 launches, ~250 copies, ~115 memsets, 17 blocking calls per frame** for
  ~300 draws in 11 render passes, of which about 2 are drawing scopes with
  p50 28 / p90 269 draws and p50 3 / p90 18 / max 21 distinct fragment
  binaries (`RENDERER2_M0_RESULTS.md`), the rest single-draw post and shadow
  passes.
- **One device operation runs at a time for 83% of device-busy time.** The
  fan-out overlaps segments with each other only; the main stream and the
  side streams never overlap (`kernel-chain` §2.2, `closure-audit` §2.2).
- **Every kernel class runs at 0.5-15% of peak SM throughput.** Per-launch
  fixed cost is 71% of the VS pool, 79% of the clip pool, 100% of stage 3
  (`closure-audit` §2.5). A 64-thread clip launch takes 9.8 µs with a 0.1 µs
  spread. Rooflines put real work at 0.3-0.7 ms/frame (`arith-vs-latency` §4).
- **The B200 pays 1.49x on kernels and 2.3-2.6x on every API call**
  (`B200_MEASUREMENTS_2026-09-03.md` §1), so the same structure costs it 9.9 ms.
- **The host issues ~1.7 ms/frame of CUDA API calls**, all inside libcuda
  (`host-launch-audit` §4); an infinitely fast GPU would still leave 3.9 ms
  (MODEL, `closure-audit` §2.3).

The structure is CuRast's pipeline, run once per draw batch. CuRast was built
for one enormous mesh per frame, and its own README names the case it does
not handle: "models with numerous meshes with few triangles, Vulkan remains
10x faster." A game frame with 300 draws and 50 pipelines is that case. The
per-batch chain is not a bug in the implementation; it is the design applied
to a workload it was not designed for.

## 2. What the redesign is, in one paragraph

Make the unit of work the **run** — a maximal sequence of consecutive draws in
one render scope that share the episode-compatibility predicate the driver
already uses (`cpvk_draws_episode_compatible`, `cp_batch_order_free`,
`cp_pass_appendable`) — and render each run with a fixed, short, wide pipeline:
one geometry launch per vertex-shader identity over *all* the run's draws;
one clip-and-bin launch over all its triangles into 16 px tile lists; one
tile-walk launch that resolves visibility per tile with early-out and
hot-tile splitting; one compaction by fragment-shader identity; one shade
launch per identity over its winners, with the writeback fused; and for
blended runs, the tile walk emits the per-pixel sorted fragment lists the
existing shade-and-composite path consumes. Everything the run predicate
refuses falls through, draw by draw, to the renderer that exists today.

Roughly **8-30 launches per run and 100-250 per frame instead of ~700**, no
host round trip inside an opaque run, one per blended run, and the
side-stream fan-out becomes unnecessary for covered runs because the kernels
are wide.

## 3. Architecture

### 3.1 Runs, not scopes, not batches

`RENDERER2_M0_RESULTS.md` finding 1: pure-opaque scopes are 26 on favorite3
and 0 on favorite2 out of ~4,200 drawing scopes. Nearly every drawing scope
mixes opaque and blended draws, so the unit has to be the run inside a scope.
`RENDERER2_M1_DESIGN.md` §1 defines it and its correctness argument stands:
a run is exactly what the existing renderer already proves sound as an
episode, plus one relaxation this plan adds.

**Opaque run**: consecutive draws, same scope, no non-draw op between them,
episode-compatible state, and every draw `cp_batch_order_free()` (no blend, no
discard, depth test and write on, depth func in LESS/LEQUAL/GREATER/GEQUAL),
with the framebuffer conditions of `cp_opaque_appendable()` **except**
`nr_cbufs == 1 && fb->color`: this plan admits **depth-only scopes** as opaque
runs with no colour output. That is the relaxation. Today a depth-only scope's
draws are refused by `cpvk_batch_structural` because `colorAttachmentCount` is
0 (`SHADOW_VISBUF_CLEARS.md`), so the heavy band's shadow scope runs **19
single-draw chains and 19 clears of a 34.6 MB visibility buffer per frame**
(0.20 ms of clears alone, `verify2-clears`). A run-level renderer is not bound
by the batch key; the shadow scope becomes one run.

**Blended run**: consecutive episode-compatible draws satisfying the A-buffer's
own admission (`!fs->writes_memory`, single-sampled, no depth write, one colour
attachment with an encoding).

**Everything else** — discard draws (alpha test), side-effect shaders,
multisampled, stencil, indirect draws, anything the identity census (§3.5)
refuses — falls through to the existing path. Coverage is reported per run,
per frame, from the first build on.

### 3.2 Geometry: one launch per vertex-shader identity per run

Today: one fused-fetch VS launch per batch, 110 per heavy frame, median grid
2 blocks, 48 of them one block (`closure-audit` §2.5). Batches break far more
often than shader identity changes — p90 56 batches against p90 18 fragment
binaries per scope — on vertex layout, index size, `start_instance`, scissor
and topology (`RENDERER2_STATE_SURVEY.md` §1.2).

The plan: for each run, group draws by VS identity (binary × execution ×
fetch admission) and launch once per group over the concatenation of the
draws' vertex ranges. The per-draw indirection already exists for uniforms
(`batch_rows → ubo_table`), draw parameters, slices, vertex-buffer bases and
index buffers (`ARCHITECTURE.md` §3.2, §4.1). What does **not** exist per draw
is the vertex *layout*: `cp_vertex_fetch_args.elem_*` are launch-wide arrays,
and the batch key splits on `num_vertex_elements`, `vertex_stride` and
`index_size`. Two ways to close that, and the first is the plan:

1. **Per-draw layout rows in a device table**, indexed by draw row like the
   UBO table, read by the inlined fetch through global loads. This keeps the
   bitcode-inline mechanism of iteration 27 untouched (no device call; dead
   end 14) and adds no local memory, because the dynamic index is into
   *global* memory, not into a per-thread array. The `.local` census of
   `CUDAVK_SHADER_STATS` (dead end 42) is the gate: zero new bytes of local
   on every fused VS.
2. If (1) costs occupancy on some shader, split the geometry launch by layout
   as a fallback; the launch count rises toward today's only where the layout
   changes.

Vertex outputs land in one run-wide buffer with a per-draw base; primitive
ids are run-global (the A-buffer already offsets them per segment,
`abuf_prim_base`).

### 3.3 Clip and bin: one launch per run

One flat kernel over the run's post-transform triangles: clip (the existing
7-plane homogeneous clip and stable slot mode), setup (`cp_tri_setup`, padded
80 → 96 B so an entry no longer straddles 32 B sectors — the verdict's kept
by-product), and binning into 16 px tiles: for each covered tile, append
`(tile, triangle ref, run-global draw row)` to that tile's list. Lists are
built with per-tile counters and a device prefix sum, so every list is dense
and every length is known before the walk launches. The M0 census sizes it:
~412k references per frame on favorite3, 1.16 per triangle, mean 117 per
non-empty scope-tile, p99 hottest tile 12,823, max 14,814; 32 px tiles halve
binning work but raise the p99 list by 62%, so 16 px stands
(`RENDERER2_M0_RESULTS.md` appendix).

This replaces `cp_clip_rast_fused`, stage 2 and stage 3 for covered runs. The
clip arithmetic is unchanged; the in-place rasterisation of small triangles
that stage 1 does today moves into the walk, where the tile owns its pixels.

### 3.4 Tile walk: one launch per run, hot tiles split at bin time

One block per **tile chunk**. Ordinary tiles are one chunk. Any tile whose
list exceeds a chunk length (≈1-2k references, to be tuned; p99 is 12.8k) is
split into consecutive chunks by the bin pass, each chunk its own block. This
is decided from the counts the bin pass already produced — **no dynamic
claiming, no persistent kernel** — and it is the direct answer to the finding
that closed both tiling attempts: one block holding a 26k-reference tile set
94% of the grid's duration (`RENDERER2_VERDICT.md`, dead end 24).

Within a block: the inverted mapping M1 benchmarked — one thread per
reference, edge functions and depth in registers, a shared-memory visibility
buffer for the tile's 256 pixels resolved by `atomicMin` on `depth << 32 |
~tri_id` — at **26.5 cycles per reference conservative, 7.8 with the
front-to-back early-out**, against dead end 24's 3,409, with the early-out
bit-identical over 921,600 resolved pixels (`RENDERER2_M1_WALKBENCH.md`).
Chunks of one tile merge into the global visibility buffer with the same
`atomicMin`, which is order-independent, so splitting changes no result. The
early-out uses the tile's depth ceiling; a split tile's later chunks take
the ceiling the earlier chunks publish, or run conservative.

Opaque output: the run's visibility buffer, depth committed per pixel.
Depth-only runs stop here.

Blended output: per pixel, the fragment list sorted by `abuf_prim_base +
tri_id`, which is submission order by the stable-clip argument
(`ARCHITECTURE.md` §2.3, §4.2). The walk writes it directly, so scan, fill,
sort and worklist go; the run's fragment total comes from the bin pass, so the
arrays are sized exactly and once (one host read per blended run — dead end 41
says that wait is GPU-paced and costs ~15 µs of bubble; it stays).

### 3.5 Shading: visibility first, one launch per fragment-shader identity

The walk executes no shader code. That is the decision `RENDERER2_M1_DESIGN.md`
§3 made and its two rejected alternatives still stand: per-draw indirect
dispatch inside the walk re-creates dead end 4's register and footprint
blow-up, and one specialised walk per identity re-serialises what the single
walk unifies and multiplies tail blocks.

After the walk, one compaction pass emits, per fragment-shader identity
present in the run, the dense list of pixels whose winner belongs to it (the
identity is the four-axis tuple: binary × execution mode × sampler variant ×
tune alternate, frozen at table-build time as M1 §3.2 specifies). Then one
existing shade kernel launch per identity over its list — the same
`cp_shade_fragments` path, argument block, row table and hardware-texture
pinning as today, with the grid sized to the list. For blended runs the
existing bucket-shade-composite structure runs unchanged over the walk's
sorted lists.

**Writeback is fused into the shade for opaque runs.** Dead end 15 measured
the "one claimant per pixel per launch" invariant false and parked the
fusion; it was false because per-batch direct launches raced over a shared
visibility buffer. Here shading runs over the compacted winner list of a
*fully resolved* buffer, one entry per pixel by construction. The invariant is
still checked in code (dead ends rule 11), and the standalone writeback stays
as the fallback (`ARCHITECTURE.md` §8 rule 21).

Identity count per run is bounded by M0 at ≤21 binaries per scope; the
four-axis product is an ASSUMPTION under 64 that the identity census (§7,
probe 2) measures before any kernel is written.

### 3.6 Clears, post passes, fallbacks

- A run clears its visibility buffer once. The heavy band's 36 visibility
  clears per frame (0.39 ms union, `arith-vs-latency` §3.5) become one per run.
- Fullscreen post-process draws (1-2 triangles, 8.5 chains per heavy frame
  today) are ordinary single-draw runs: every tile gets one reference and the
  walk trivially resolves it; no special case is needed for correctness,
  though a covering-triangle shortcut in the bin pass is a later option.
- Any run the plan refuses executes on the existing renderer exactly as
  today, including its fan-out and PDL. The flag is `CUDAVK_RENDERER2` (value
  bool, registry, default off until M5), with `CUDAVK_NO_RENDERER2` as the
  revert once it is the default.

### 3.7 Memory

Per run: a reference array (~8 B × references, ~3.3 MB/frame), a setup array
(96 B × post-clip triangles, ~34 MB/frame at 354k), per-tile counters and
offsets (~4k tiles), the run's vertex output buffer, and for blended runs the
fragment lists the A-buffer already allocates. All bump-allocated per run from
`dscratch` with the existing generation rules (`ARCHITECTURE.md` §8 rules
4-5), sized from device counts (no worst-case bound — dead end 11), with
overflow of the reference or setup arrays a fail-closed flag that re-executes
the run on the existing path.

## 4. What it should be worth, and the honest uncertainty

Per frame on the RTX heavy band (ESTIMATED; the derivations are
`arith-vs-latency` §6.2 and the kernel-chain census, and every term carries
±2x on instruction counts because no favorite3 SASS or ncu `main` sample
exists):

| term | today (MEASURED) | plan (ESTIMATED) |
|---|---:|---:|
| launches | ~700 | 100-250 |
| host issue (1.36-1.70 µs per call plus copies and memsets) | ~1.7 ms | 0.3-0.5 |
| device: real vertex, raster, fragment work | 0.3-0.7 inside 5.2 of union | 0.3-0.7 |
| device: launch shells and dependent gaps | ~4 ms of the union is this | 0.4-0.8 (150 dependent steps × 2.5-5 µs) |
| device: walk + bin + compaction | — | 0.25-0.5 |
| device: clears and compaction scans (once per run) | 0.39 + 0.37 | 0.1-0.2 |
| host round trips inside the frame | 17 | ~9 (one per blended run) |
| frame boundary (application: fence, record, 4 MB memcpy) | ~0.9 | ~0.9 unchanged |
| **frame** | **8.2** heavy / **5.9** whole | **2.5-3.5** heavy / **2-3** whole |

B200: kernels 1.5x and calls 2.3-2.6x (MEASURED ratios) on a frame with far
fewer of both gives **3.5-4.5 ms** (ESTIMATED), which puts <= 5.0 on the
deployment target inside the range and not at its edge. The native driver's
0.52 says the hardware could go further; this design does not claim it.

Three uncertainties bound this and none is resolvable before building:
1. **Coverage.** How much of the frame lands in admitted runs. M0 says nearly
   every drawing scope mixes classes and blended draws are common (8.9
   blended episodes per frame); if runs are short the launch count does not
   fall as far. The run census (§7, probe 1) gives the number.
2. **Hot-tile balance.** Splitting fixes the one-block signature only if
   chunks are small enough and the merge is cheap; the walk benchmark ran
   3,600 blocks with one hot tile, not a split one.
3. **Device-to-frame conversion.** Device time is not frame time
   (`PERFORMANCE.md` §5.1); the fence-bound frame boundary stays at ~0.9 ms
   whatever the device does. Only the A/B measures this.

## 5. Staged plan with gates

| stage | deliverable | gate to continue | scale |
|---|---|---:|---|
| **P0 probes** (no GPU, no kernel) | run census at HEAD on both captures (runs per scope, draws and references per run, share of references in admitted runs, depth-only scopes admitted); four-axis identity census per run; stencil / alpha-to-coverage census; `.local` census baseline of every fused VS | admitted runs hold >= 60% of references; identities per run <= 64 at p99; no stencil in candidate runs | days |
| **P1 microbenchmarks** (standalone, both GPUs) | split-tile walk: chunks of 512/1k/2k references over the M0 p99 list with `atomicMin` merge; bin pass over a recorded run (references per triangle 1.16, 354k triangles) | walk p99 chunk <= 30 µs; bin pass <= 0.15 ms per heavy frame; on the B200 the same within 1.6x | days |
| **M2 opaque runs** | geometry per VS identity with per-draw layout rows; clip+bin; split walk; compaction; per-identity shade with fused writeback; depth-only scopes admitted; everything else falls through; `CUDAVK_RENDERER2` | 79/79 suite with the flag on; both captures render within the native-driver tolerance (§7); then a three-round alternating A/B on favorite3: **>= 1.0 ms/frame on the heavy band or stop** | 4-6 weeks |
| **M3 blended runs** | walk emits sorted per-pixel lists; existing shade/composite; one sizing read per run | same correctness gates; A/B both captures; the UI episode and the ~9 blended episodes covered | 2-4 weeks |
| **M4 B200** | the same set validated on the B200, three rounds, two sessions | disjoint arms both sessions | days |
| **M5 default** | `CUDAVK_RENDERER2` on by default, `CUDAVK_NO_RENDERER2` revert, `FLAGS.md` regenerated, `PERFORMANCE.md` defaults table row | wins on both captures on both GPUs, or reverts cleanly | — |

A stage that misses its gate stops the project at that stage, as the
Renderer 2 charter said; the branch stays as evidence.

**Correctness oracle.** Bit-identity against the current renderer is the
wrong gate for M2: run-global primitive ids change which triangle wins an
exact depth tie, which the current renderer already does not promise across
runs (`ARCHITECTURE.md` §9, "determinism"). The gate is the one that did not
exist until this week — the native driver's full-frame render
(`NATIVE_DRIVER_REFERENCE.md`): the same per-frame tolerance compare, every
25th frame, plus the 18 sentinels and the 20-run `multithreading` compare.
Frame 3225's known shading defect is excluded until `TODO.md` item 14 is fixed.

## 6. Every dead end, and whether this plan collides with it

Entries not listed touch nothing this plan changes (5, 12, 13, 16, 17, 23,
26, 28, 29, 32, 33, 34, 37, 38, 40, 42). For each of those the plan's answer
is "unchanged and orthogonal", and 34, 37 and 40 remain live items on their
own (the application fence and the 30.9 MB managed allocation).

| # | what died | collision? | why this plan differs, or why it is worth trying anyway |
|---|---|---|---|
| 1 | CUDA graphs at three units: no recipe reuse | no | no graphs. Note for later: at 100-250 launches with a fixed per-run topology, graphs become a plausible second step; not part of this plan |
| 2 | persistent / dynamic-claim scheduler, +140 ms | **adjacent** | no persistent kernel, no per-item claiming. Tiles and chunks are a static grid decided by counts the bin pass already produced. The lesson kept: scheduling cost must not scale with items |
| 3 | trivial/heavy clip split, +0.85 | no | one flat clip pass over all of a run's triangles; no classification into two kernels |
| 4 | out-of-line sampler dispatch, 143-182 registers | no | the walk executes no shader; shading is one existing kernel per identity. M1 §3.3 rejected in-walk dispatch on exactly this entry |
| 6 | episode-global tagged stage 3: the one required decision introduced a full sync | no | opaque runs have no host decision; blended runs keep the one sizing read they have today |
| 7 | folding a 90 MB/frame clear into a kernel, +1.84 | no | clears are not folded; there are fewer of them because the visibility buffer is per run (36 → ~15 per heavy frame). Bytes still move at memset speed |
| 8 | smarter flush placement | no | the upload ring is unchanged; flush points fall with launches |
| 9 | fragment grid sizing: idle blocks are free | no | shade grids are sized to compacted lists, but no credit is claimed for it |
| 10 | kernel-internal redesign of `stage3_abuf`: 0.15 waves/SM, "the work belongs in fewer, larger launches" | no — it is the retry clause | this plan is that clause: work per launch grows to a run |
| 11 | worst-case A-buffer sizing: bound/actual 3×10⁷ | no | every array is sized from a device count after the bin pass; the one blended sizing read stays |
| 14 | vertex fetch as a device link, 20 → 108 registers | **adjacent** | the fused fetch stays bitcode-inlined. The new per-draw layout table is a global-memory table indexed by draw row, the same shape as the UBO table, not a device call and not a dynamically indexed local array. Gate: zero new `.local` bytes (`CUDAVK_SHADER_STATS`) |
| 15 | writeback fusion: 6.3 M second claimants per pixel per launch | **collision** | the invariant was false because per-batch direct launches raced over a shared visibility buffer. Here the shade runs over the compacted winner list of a fully resolved buffer, one entry per pixel by construction. The invariant is re-checked in code before it is believed, and the standalone writeback stays as the fallback |
| 18 | wider raster requests: SoA would make it worse | no | the setup entry is padded to 96 B (the verdict's kept by-product), which is the one widening entry 18 allowed |
| 19 | not issuing degenerate stage-3 launches | no | there is no stage 3 |
| 20 | the wide merge: removed 299 launches, cost +0.41 | **collision on the surface** | the wide merge concatenated concurrent per-segment launches into one grid *of the same per-item latency chain* on one stream, and lost cross-stage pipelining and 199 PDL links. This plan's launches are wide grids that finish in ~max(item) at real occupancy: 3,600 tile blocks against 8 side streams. And the claim is not the launch credit — see 22 |
| 21 | count-phase merge, refuted on overlap | as 20 | same answer; the A-buffer count phase disappears rather than being merged |
| 22 | **the merge rule**: a launch-removal credit is only collectable where launches were serial | **collision, and adopted** | the plan banks **no** launch-removal credit (0.78 µs × N). Its claim is the device-time change of the run pipeline against the pool it displaces, priced by entry 22's own rule 3 (subtract the merged form's cost) and tested by its rule 4 (union busy must fall *and* the frame must fall; if union falls and the frame rises, stop). The launch and host-issue reductions are reported, not banked, exactly as the Renderer 2 charter did |
| 24 | opaque sort-middle tiling: one-block signature, no occlusion, surrendered +2.73 | **collision** | its two recorded retry conditions are met — occlusion exists (early-out, bit-identical) and the fan-out is not surrendered piecemeal — and its one-block signature is answered at bin time by chunk splitting, which entry 24's prototype and Renderer 2 both lacked. The fan-out's value (1.15 ms on the B200, 0.66-0.84 RTX) is the price of serialising *today's small launches*; a run's wide launches do not need it |
| 25 | device-side chaining: 7.3 µs per device launch | no | no device launches; the host issues 8-30 per run |
| 27 | scopes form a chain, depth 8-9 of 10 | no | scopes stay serial and so do runs within a scope; the plan shortens each link, it does not overlap them |
| 30 | episode-entry Hi-Z, 0.103 | no | the walk's occlusion is intra-run, current-frame, computed as it goes — the charter's distinction stands |
| 31 | depth-only chain deferral: 0.170 for compact+FS+writeback of safe calls | **numbers collide, mechanism differs** | 31 priced deferring three kernels of the shadow scope's 19 chains. This plan collapses the 19 chains, the 19 clears (0.20 ms) and the 19 single-draw structures into one run; the shadow scope's whole pool, not 31's subset, is at stake, and it must be re-measured as such, not quoted from 31 |
| 35 | within-batch vertex reuse, 0.438 | no | vertex work is unchanged per invocation; a per-run geometry launch may later make reuse cheaper, not claimed |
| 36 | **Renderer 2 refused at design**: displaced 1.46 − walk − bin − surrender 0.66-0.84 = 0.41; upper bound 5.21 above the goal | **collision: this is that design re-chartered** | the verdict's arithmetic was right for the design as chartered — geometry "existing, unchanged" per batch, shading per batch, surrender at full. Its own list of undisplaced pools (VS 0.69, FS 0.68, clip 0.59) is what §3.2, §3.3 and §3.5 collapse, because those pools are 71-100% per-launch fixed cost (`closure-audit` §2.5), not arithmetic; recomputed under that premise the same design's bound is 4.2-5.3 (`closure-audit` §2.5 accounting C). The surrender does not apply to wide launches (24). And four of the verdict's five "kept regardless" findings are inputs here: the walk cost, the exact early-out, the census infrastructure, the 96 B setup entry. What still stands from the verdict and is designed for, not argued away: the hot-tile imbalance (§3.4) and the possibility that blended runs show little win (§5 M3 gate) |
| 39 | argument-block hoist: relocating a copy onto a critical link | no | per-run argument blocks are uploaded once per run before its first launch, on a host-paced link; the rule is respected by construction and re-measured per 22's rule 4 |
| 41 | device-decided episodes: the wait is the GPU's | no, and it simplifies the plan | the one sizing read per blended run is kept; 41 says its cost is ~15 µs of bubble, so no device-side sizing machinery is built |
| 43 | retry convergence check: population a tenth of the gate | no | discard draws are not admitted to runs; they fall through to the existing retry loop unchanged |

Also from `PERFORMANCE.md` §7: S1d (drain at the scan) is moot — there is no
scan; the `bounded` fast path stays dead code; PDL stays on the fallback path
and is applied to the run pipeline's dependent links where the secondary can
hoist a clear (rule 4 of §4).

## 7. Probes to run before any kernel, in order

1. **Run census on both captures at HEAD** (host-side, the census plumbing on
   `redesign/tilewalk` plus wiring `cp_tile_census_cut`, which still has no
   callers): runs per drawing scope, draws and tile references per run, share
   of a frame's references inside admitted opaque and blended runs, and how
   many depth-only scopes the relaxed predicate admits. **Stop if admitted
   runs hold under 60% of references.**
2. **Four-axis identity census**: run `cp_fs_launch_shader()`'s resolution for
   every draw of every scope and count distinct `(fs, exec, variant, alt)`
   per run. **Stop if p99 exceeds 64.**
3. **Stencil and alpha-to-coverage census** of pipelines inside candidate
   runs, which `cp_batch_order_free()` does not test.
4. **Split-walk microbenchmark** on the M0 p99 list (12,823 references) at
   chunk sizes 512, 1k, 2k with an `atomicMin` merge, on both GPUs; extend
   `tests/cp_tilewalk_bench.cu`. **Stop if the p99 chunk exceeds 30 µs or the
   merge exceeds 10% of walk time.**
5. **Bin-pass microbenchmark** over a recorded run's post-clip triangles.
   **Stop above 0.15 ms per heavy frame.**
6. **`.local` baseline** of every fused VS under `CUDAVK_SHADER_STATS`, to
   compare against after the per-draw layout table lands.

Each is hours to a day; none touches the driver's default path.

## 8. Risks, plainly

- **Coverage is the whole bet.** If the run predicate admits a small share of
  the frame, the design pays its fixed cost and collects little. Probe 1
  decides it before anything is built.
- **Per-draw vertex layout in the fused fetch** is an ABI change on the
  kernels that carry 110 launches a frame; dead end 14 is the failure mode
  and dead end 42's census is the gate.
- **Identity explosion**: the sampler-variant axis is resolved by
  dereferencing every per-draw UBO row on the host; a run whose rows disagree
  becomes its own bucket per draw (M1 §3.2). Probe 2 bounds it.
- **Tie-break determinism** changes with run-global ids; the oracle is the
  native render, not the old renderer.
- **The frame boundary is untouched.** ~0.9 ms of every frame is the
  application's fence, recording and 4 MB memcpy; the redesign floor on the
  RTX includes it. The application change (`PERF_ANALYSIS_2026-09-02.md`
  §3.4) is independent and still worth 0.5-1.1.
- **Effort**: M2 alone is a new geometry ABI, two new kernels, a compaction
  pass and a fused writeback, plus the fall-through plumbing — one person,
  two to three months to M4, and the first decisive number (the M2 A/B)
  arrives at the end of the first six weeks, not before.

## 9. Sources

`RENDERER2_DESIGN.md`, `RENDERER2_M0_RESULTS.md`, `RENDERER2_M1_DESIGN.md`,
`RENDERER2_M1_WALKBENCH.md`, `RENDERER2_VERDICT.md` (branch
`redesign/tilewalk`); `PERF_ANALYSIS_2026-09-02.md` and
`history/perf-2026-09-02/` (kernel-chain, closure-audit, arith-vs-latency,
verify2-*); `LEAD_RESULTS_2026-09-02.md`; `B200_MEASUREMENTS_2026-09-03.md`;
`NATIVE_DRIVER_REFERENCE.md`; `SHADOW_VISBUF_CLEARS.md`; `DEAD_ENDS.md` 1-43;
`~/git/CuRast/README.md` (the many-small-meshes limitation);
`~/git/cuRE` (`source/cure/pipeline/`, `source/CUDARaster/cuda/` — bin,
coarse and fine stages at 8 px tiles in 16×16 bins) and its paper, whose
result that the multi-kernel sort-middle CUDARaster beats the streaming
megakernel is why this plan has no megakernel; Mesa `llvmpipe`
(`lp_scene.h`, `lp_rast.h`: 64 px tiles, per-tile command bins with in-order
state changes), the proven shape for arbitrary API state.
