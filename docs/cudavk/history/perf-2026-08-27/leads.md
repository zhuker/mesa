# cudavk performance leads: designed or planned, never measured

Audit date: this session. Tree `/home/alexzhukov/mesa`, branch `cudapipe-vk-native`,
HEAD `e2fea470d04`. **No GPU was used to produce this file.** Everything below comes
from reading documents, `git log` and `src/cudavk`. Where a number would settle a
question, it is written down as a proposed experiment, not run. A second agent is
re-measuring at HEAD; this file is deliberately honest about which side of that
measurement each piece of evidence sits on.

Status vocabulary (`DEAD_ENDS.md`'s, plus one):

- **NEVER TRIED** — described in a document, no code in `src/cudavk`, no flag in
  `src/cudavk/FLAGS.md`. Verified by grep, not by the document's own claim.
- **PARKED, KNOWN NEXT STEP** — work exists, is incomplete, the entry names the step.
- **PARTIALLY TRIED** — code or a probe exists and produced a number, but the idea
  was not taken to an accepted result.
- **CLOSED** — measured and lost or neutral. Listed only in §5.

Evidence-side vocabulary, used in its own column:

- **PRE** — the evidence was collected before `20611f5b131` (opaque episode-segment
  stream fan-out defaulted ON, 2026-08-26 07:38). The frame was 15.7-16.0 ms then.
- **POST** — collected at or after that commit.
- **N/A** — the lead is a proposal with no measurement behind it either way.

---

## 0. The baseline moved and the documents did not

`20611f5b131` measures:

    old capture   15.77 -> 13.23 ms   (-16%)
    Crossroads     5.90 ->  5.82 ms   (-1.4%)

`PERFORMANCE.md` was written 2026-08-24 and `TODO.md`'s header still says "old
capture 15.74 ms/frame". Both predate that commit. A fresh reader will quote
15.75 and be 16% wrong.

**There are three different "current" old-capture numbers in the tree.** Ranked
by recency of measurement:

| number | where | when | what it was |
|---:|---|---|---|
| **13.0878 / 13.0799 ms** | commit `67f9a158dab` body | 08-26 07:37 | base and new arms of the timeline-semaphore A/B, 18 replays, three arms alternated in verified-unshared GPU windows. Crossroads 5.7807 / 5.7817 |
| **13.23 ms** | commit `20611f5b131` body | 08-26 07:38 | the fan-out's on-arm. Crossroads 5.82 |
| 15.7514 ms | `PERFORMANCE.md` §1, `DEAD_ENDS.md` conventions, `TODO.md` :7 | 08-24 | the pre-fan-out shipping default. Crossroads 5.8982 |

The first two are one minute apart and are both post-fan-out in substance, so
**HEAD is somewhere in 13.08-13.23 ms on old and 5.78-5.82 on Crossroads**. The
two do not have to be reconciled — they are different sessions — but neither is
in any document.

**Why this matters for every estimate in `PERFORMANCE.md` §6.** Those estimates
were sized against a frame with 12.44 ms of host-blocked time and **4.16 ms of
device idle**, and the fan-out attacks exactly that overlap. `0458a51e9a4`
measured the mechanism directly: forcing every draw opaque made the union of
kernel intervals *equal their sum* — 10.79 ms of kernel work in 10.79 ms of wall
clock — while the default arm compressed 12.34 ms into 9.18 ms. All of that
compression came from the blended fan-out and none of it reached an opaque
episode. `20611f5b131` gave the opaque path the same treatment.

So the ranking below separates two classes:

- **Leads justified by DEVICE IDLE** — L5, L15, L18, L20, and the CUDA-graph item
  in §2F. The idle is the quantity the fan-out consumed. Assume their ceiling has
  **fallen** until L0 says otherwise. This is the group most at risk.
- **Leads justified by a specific BLOCKED WAIT on the blended path** — L3 (segment
  counters) and L4 (peel checks). These sit in the A-buffer / peel path, and the
  blended path **already had** a segment fan-out before this commit — the fan-out
  gave the *opaque* path what the blended path already had (`0458a51e9a4`). So
  their absolute figures probably survive, while their *share of the frame* rises
  from ~17.5% to ~21%. **Reasoning, not measurement.** L0 settles it in one run.
- The episode drain (8.663 ms over 9.88 waits) is in between: it is a blended-path
  wait, but it waits on `cp->stream`, and opaque work no longer queues there. Its
  figure is more exposed than L3's or L4's.

---

## 1. Ranked leads

Ranked by expected value per unit of risk. Probes rank first: they cost nothing,
cannot change output, and four other leads are gated on them.

| # | lead | mechanism, one line | written down in | status | evidence side | evidence that exists | expected win | risk | effort |
|---|---|---|---|---|---|---|---|---|---|
| L0 | **Re-baseline the wait table at HEAD** | run `CUDAVK_PLAN_STATS=1` on both captures at HEAD and restate the four waits, the blocked total and the device idle | nowhere; implied by `PERFORMANCE.md` §5.2 against `20611f5b131` | NEVER TRIED | N/A | the instrument exists (`cp_sync_timed`, `cp_renderer.c:585`; sites at 6420, 6689, 6715, 8443). The table it would replace is PRE | 0 ms; it is what makes L3, L4, L11, L15 and L18 forecastable at all | none — host timers, no profiler, no output change | ~1 h + one session |
| L1 | **P1 — log `total/quads` per episode** | one `fprintf` at the existing tail drain, which already holds `ctr[0]`=total and `ctr[3]`=quads; histogram at teardown | `/tmp/perf16/iter29-sync/addendum.md` §5 P1; `DEAD_ENDS.md` 11 "Retry if" | NEVER TRIED | PRE (design) | `DEAD_ENDS.md` rule 7: bound-based sizing is priced by `bound/actual`, not by the wait it removes. The S0 probe measured what a loose bound costs: +0.38 ms on Crossroads, 8.6 GB refused on old | 0 ms; **decides L11 outright, calibrates L3** | none | ~2 h + one session |
| L2 | **P2 — time the inter-drain host issue burst** | add the complement of `cp_sync_timed`: host time between one drain returning and the next starting | `addendum.md` §5 P2 | NEVER TRIED | PRE (design) | `design.md` §1.2 *assumed* the host has work to overlap; nothing measured it. S1b recovers at most `min(burst, wait)` per episode | 0 ms; **decides L18** | none | ~2 h + one session |
| L3 | **Site 3 — bound the `bounded` fast path by the clip rectangle** | replace `ab->nblocks` (230,400 blocks, whole framebuffer) with the 2x2-block count of `clip_x0..clip_y1`; drop `num_triangles <= 2` | `design.md` §6; `PERFORMANCE.md` §6.2; `addendum.md` §4 | **NEVER TRIED** — `cp_renderer.c:6402-6413` still tests `num_triangles <= 2` and `ab->nblocks`; no flag | **PRE** | wait cost supported: 4.17 waits/frame, 0.995 ms, mean 0.239 ms. The bound is a proof, not a prediction; over-estimating costs idle blocks at <0.2 ns each | **0.10-0.25 ms** (design) / **0.10-0.20 ms** (addendum §6, the later number) — estimate. Blended-path wait, so **probably not collected by the fan-out** | medium-low: no output can change; a wrong bound is detectable, `abuf_quad_fill` already carries `quad_overflow` | small-medium |
| L4 | **Site 2 — predicate the peel pass and predict its trip count** | S2a carry the existing device `peel_any` predicate through a whole pass; S2b move the interval reset to a 1-thread kernel; S2c skip the check when `peel_passes <= 4`, else run `predicted+slack` and confirm once | `design.md` §5; `PERFORMANCE.md` §6.1; `addendum.md` §4 ("survives intact") | **NEVER TRIED** — `cp_renderer.c:6512` still stores `peel_any` from the host, `:6692` reads it, `:6696` uses `CP_PEEL_CHECK_MAX` doubling; no flag | **PRE** | the most expensive single wait in the driver: 2.765 ms/frame over **1.71 waits at 1.620 ms each**. The correctness argument is already in the driver's comment (`:5876-5878`) | **0.20-0.50 ms**, 1.4-1.9 ms of blocked host — estimate. Peel-loop wait on the blended path, so **probably not collected by the fan-out**; its share of the frame rises | medium: side-effect shaders must shade exactly zero slots; **the overshoot cap is mechanism, not a knob** (unbounded overshoot on a 256-layer draw adds ~1.8 ms and loses). Addendum: audit *every* `cuMemsetD*Async` reachable from a peel pass. Add: see §3 hazard H2 | medium, ~1 iteration |
| L5 | **`griddepcontrol` (programmatic dependent launch)** | let the next chain start before the previous drains, on the chained `.cu` kernels | `notes/CUDA13_UPGRADE.md` §5 | **NEVER TRIED** — grep: zero occurrences in `src/cudavk` | **PRE** | measured on this machine on chained dependent kernels: stream 4.10 us/kernel -> graph 2.02 -> **PDL 1.89 (-53.9%)**. Assembles today at `compute_120`, no toolkit upgrade | no frame estimate. Bounded by 3.287 ms/frame of traced sub-20 us idle gaps — **a PRE number, and idle is what the fan-out consumed. Assume this ceiling has fallen.** The note also warns: graphs and PDL remove the same ~2.2 us gap, do not add them up | medium: relaxes a dependency the driver gets from stream order; needs a per-kernel audit of what may legally start early | small-medium |
| L6 | **Depth/stencil load/store elision** | retention (skip the depth LOAD when `depthbuf` already holds that attachment) plus lazy store (defer the conversion until a consumer) | `history/VK_NATIVE_DECISIONS.md` :526-541, next-step 2 at :910; raw evidence `/tmp/cpvk-loadstore.md` | **NEVER TRIED** — grep for `elid`, `lazy_store`, `pending_store`, `depth_retain` returns nothing | **PRE**, and older than most — this is native-driver-era evidence | full-stream analysis of both captures: **not one `loadOp=LOAD` in either capture is a real load** — 100% are preceded by a full-area store or clear of the same image. Elidable/frame: Crossroads 1.0 depth STORE, 1.43 stencil LOAD, 2.15 stencil STORE; old 1.02 depth STORE | **order 0.1-0.2 ms/frame** (~4.6 full-screen kernels/frame on Crossroads, ~1.1 on old). These are device-time passes, not bare launches, so §4's launch price does not apply (§4 rule 2). Independent of the fan-out | medium: cross-submission pending-store state, and a consumer hook in every copy and sample path. Its own doc says "Worth doing; not blindly" | medium |
| L7 | **`CU_JIT_SPLIT_COMPILE` = `(CUjit_option)34`** | pass JIT option 34 (value 0 or 16) in the shader link path | `notes/CUDA13_UPGRADE.md` §1 | **NEVER TRIED** — `nir_to_ptx/cp_nir_to_llvm.c:3788-3833` sets only MAX_REGISTERS and the log buffers | PRE, but frame-independent | verified on a real 531 KB cudapipe PTX module with the JIT cache disabled: **536 ms -> 168 ms (-65%)** at 16 threads | **0 ms of frame time.** Cold first-compile latency only; the second load is ~1.1 ms either way | low | ~2 lines |
| L8 | **Raise `CUDA_CACHE_MAXSIZE`** | raise the driver's on-disk cubin cache above the 1 GiB default | `notes/CUDA13_UPGRADE.md` §2 | **NEVER TRIED** — not in the registry; an NVIDIA env var, so a deployment/doc change | PRE, frame-independent | `~/.nv/ComputeCache` measured at **797 MB, 12,286 files** against a 1 GiB default. The cache is the existing cold/warm 2.5x replay factor (24.4 -> 9.73 ms/frame); eviction re-pays the cold JIT bill | 0 ms steady-state; it protects a 2.5x factor that already exists | none | minutes |
| L9 | **`.pragma "enable_smem_spilling"` on the whole-program `.cu` kernels** | spill to shared memory instead of L2, with explicit launch bounds | `notes/CUDA13_UPGRADE.md` §3 | **NEVER TRIED** — grep: zero occurrences. `CUDAVK_LAUNCH_BOUNDS` already exists | PRE | measured on **this tree's own PTX**, text-patched to `.version 9.0`, JIT-loaded on r580: `cp_fs_interpolate` 150 regs / 1 CTA per SM -> 128 regs / **2 CTA per SM**; `cp_abuf_interpolate` 8 B local -> **0 B**, 5 -> 6 CTA/SM | no frame estimate | medium-high, **contested — §2J**. Two named blockers: without explicit launch bounds shared memory hits 43 KB/CTA and occupancy *drops* 6 -> 2; `cuLinkAddData` rejects the pragma for per-function compilation, so the NIR path needs the sampler inlined first | medium |
| L10 | **Re-A/B `CUDAVK_INLINE_FS` at HEAD** | flip the existing opt-in same-LLVM fragment interpolation on and measure | derived: `PERFORMANCE.md` §3 closing note + `FLAGS.md`; original result iteration 14 | **PARTIALLY TRIED, stale** — the flag exists and the architecture is complete; the opt-in decision is from iteration 14 | **PRE, and by a long way** — iteration 14 predates the hardware texture cache becoming the default | iteration 14: focused chains improve **25.3% direct / 16.4% A-buffer**, old replay neutral (23.240/23.231 vs 23.217), dual ownership costs **+8.34 s cold/warm replay wall and +110.6 MiB host RSS** over 65 shaders | **no estimate.** The reason it was neutral was that the software sampler was the resource floor — and the hardware texture path is now the default at 99.9% coverage, so that floor has moved | low — the flag reverts, both arms in one binary | one A/B session |
| L11 | **S1d — take the episode drain at `cp_abuf_scan`, not at the tail** | the scan already writes the exact fragment total (`sum3[0]`) and the clamp flag near the *start* of the chain; size the shade arrays from those and delete the tail drain | `addendum.md` §2; `DEAD_ENDS.md` 11 "Retry if" | **NEVER TRIED, gated on L1** | PRE | the sizing proof already exists in the driver (`cp_renderer.c:1565-1570`): "a quad needs at least one covering fragment, so there can never be more quads than fragments". Over-allocation factor becomes `total/quads`, not `capacity/quads` | **no estimate — "its sign is unknown and it must not be built on that description"** | medium; dead outright if L1's median ratio is large | medium |
| L12 | **Fuse `cp_fs_writeback` into the fragment shader** | emit `cp_fs_write_lane()` at the end of the generated shader body; the lane reads its own colour from `fs_out` in L1 instead of a kernel round trip | `DEAD_ENDS.md` 15 (PARKED); `PERFORMANCE.md` §6.3; `/tmp/perf16/iter28-item3/` — README, `wip.patch`, `cp_fs_write.h`, **confirmed still on disk** | **PARKED, KNOWN NEXT STEP** | PRE | admission A = 1.0000 by launch, 92 -> 94 registers, no local memory, suite 65/65 with the flag on; 43.0 reachable launches/frame; the fused lanes provably do store (~2.5 M colour stores/frame) | **~0.04 ms** — this number *replaces* an earlier 0.10-0.20 ms estimate (§2A) | **high as it stands: the old capture renders visibly wrong frames** — whole background quads missing, 1.4-2.0 M pixels over tolerance. **Rests on the known-false invariant — §4** | next step is one instrumented run: record the slot->pixel mapping the fused lane used for one draw, compare against what the standalone kernel computes for the same launch |
| L13 | **Re-sweep the adaptive rasterizer thresholds** | sweep `CUDAVK_SMALL_THRESHOLD` (128), `CUDAVK_MEDIUM_THRESHOLD` (1536), `CUDAVK_POINT_THRESHOLD`, `CUDAVK_TILE_BOUND` at the current default | `FLAGS.md` "Rasterizer tuning"; reasoning in `kernels/cp_rast_types.h:1069-1140` | **PARTIALLY TRIED, stale** — flags exist and were swept historically; no document records a sweep since the texture cache, the vertex-fetch fusion, the A-buffer fusions **or the fan-out** | **PRE** | "the rasterizer's main tuning knob". The raster chain is still **54.7% of all kernel time** (7.95 ms/frame, 627 launches). The point threshold was previously worth **28% of `particlesystem`** | no estimate | low for points (`cp_rast_types.h`: a point resolves identically in both stages, so which stage takes it cannot change a pixel). **Not output-neutral for triangles** — moving them across the boundary flips two pixels of `gltfscenerendering` on an `atomicMin` tie | one sweep session |
| L14 | **`cuMemDiscardAndPrefetchBatchAsync` on the managed arena** | `LOAD_OP_DONT_CARE` for unified memory: discard then prefetch instead of faulting | `notes/CUDA13_UPGRADE.md` §6 | **NEVER TRIED** — grep: zero occurrences; reachable today on r580 via `cuGetProcAddress` | PRE | measured on a 128 MiB managed arena the host writes and the GPU overwrites: faulting **8.3 ms -> prefetch 2.67 ms -> discard+prefetch 0.55 ms (14x)**. Maps onto 1,916 recorded UM-fault events on one 490.7 us launch | no frame estimate | medium. Named traps: ~14 us per range, so few large arenas; **discard without a following prefetch is a pessimisation (34 -> 120 ms)**. Relevance depends on `CUDAVK_SMALL_ALLOC` (default `advise`) | medium |
| L15 | **S1a — empty the drain of what does not need the host** | move D5 to the existing `abuf_seg_prefix` launch, defer D1, delete D3, size D6 from the host bound; the drain becomes one sync plus one 4-byte copy | `design.md` §4.1 S1a; `PERFORMANCE.md` §6.5; `addendum.md` §4 | NEVER TRIED | PRE | removes ~9.9 copies/frame and host-side loops, adds ~9.9 launches | **claimed at 0.00 ms by its own author.** "Land it on tidiness or not at all" | low | medium — a real refactor for a zero claim |
| L16 | **Diagnose `pbribl`'s +0.03 ms with fused vertex fetch** | find why one sample regresses with the fusion on | `TODO.md` open leads item 5 | PARTIALLY TRIED — bounded and deliberately accepted | PRE | the cost is in fused execution on that workload, not in the machinery around it | **<= 0.03 ms on one 0.52 ms sample.** Note `pbribl` was *also* the worst victim of the segment-0 fan-out bug (2.3x, `b3f716bd5f5`) — it is a sample that is unusually sensitive to per-episode fixed cost | low | small-medium |
| L17 | **Conservative stage-3 tile-interior accept** | certify tile interiors in stage 2 so stage 3 skips `e2` and all three coverage comparisons on interior samples | `/tmp/perf16/tile-interior-design.md` — analysis only, "no source was edited and no GPU run was made", **and not referenced from any tracked document** | **NEVER TRIED** | PRE | its own audit of `cp_rasterize.cu:1087-1288`. The old census has 67.59 M stage-3 tile executions but **never classified interiors**, and the primitive p50 is 4 tiles direct / 6 A-buffer, so most queued work is boundary-shaped | no estimate; the design sets its own gate at **>= 0.3 ms projected gross opportunity** and demands a predicate census first | high, **double-contested — §2G and §2J** | census first (~1 iteration), implementation after |
| L18 | **S1b — defer the shade by exactly one episode, two A-buffer arenas** | `cp_pass_finish` splits into build and shade; the host takes episode M's wait after issuing episode N's build | `design.md` §4.1 S1b; `addendum.md` §3 | **PARKED, EXPLICITLY DEMOTED** — "not refuted, but not to be written" until P2 (L2) | PRE | eight numbered correctness invariants (`design.md` §4.2) and the flush points (§4.3) | **0.5-1.5 ms, and the estimate is retracted by the same tree — §2C.** It was sized against device idle, which the fan-out consumed: **the most exposed estimate in this file** | **high.** Two arenas cost 250 MB-1.6 GB against an 8,589,934,592-byte cap the S0 probe already hit at 8,605,856,768 bytes; the scratch grow path turns pressure into per-flush `cuMemAlloc`/`cuMemFree`. The only new failure mode is a missed flush point, whose symptom is wrong pixels | large |
| L19 | **Opaque sort-middle tiling, second version** | coarse bin, then compact fine tiles; keep adaptive raster behaviour instead of one uniform tile loop | `notes/OPAQUE_TILING_PROTOTYPE.md`, six named directions | **PARTIALLY TRIED and it LOST** — `CUDAVK_TILED_OPAQUE` / `_CENSUS` exist and ship off | PRE, **and now handicapped** | Nsight Systems over 60 `multithreading` frames: `cp_opaque_tile_raster` **2,562 ms total, 88.7% of tiled kernel time, ~42.7 ms/frame alone**. Old 25.23 -> 35.61 ms, Crossroads 7.27 -> 10.77 ms. Count and fill were only ~0.77 ms/frame, so the surrounding architecture is not the problem | no estimate. **`7f38d2a9b65` refuses the opaque fan-out for `CUDAVK_TILED_OPAQUE`**, so the tiled arm now competes against a default that overlaps opaque segments while the tiled arm does not. The gap is now wider than the numbers above | very high | very large. Named next step is NCU on `cp_opaque_tile_raster`; the doc is explicit that its failure explanation is "a well-supported hypothesis until checked with Nsight Compute" |
| L20 | **The six paradigms in `notes/DEVICE_AUTONOMOUS_SYNC.md`** | conditional graph nodes; device graph launch; warp-cooperative slab lists; VMM growth with an async pager thread; chunked processing; persistent megakernel | `notes/DEVICE_AUTONOMOUS_SYNC.md` | **NEVER TRIED, and `TODO.md` calls it "unacted"** | N/A — no measurement in it at all | none in this tree. It is a literature survey with external citations | ceiling quoted as ~4.16 ms/frame of device idle — **a PRE number the fan-out has partly consumed** | **very high, and largely mis-aimed — §2G.** `TODO.md` states the ~17 blocks are **read-back-and-decide, not synchronisation**, so no sync primitive removes them; and "the most obvious approach is already refuted three ways in `DEAD_ENDS.md` entry 1" | very large |
| L21 | **The doorbell across two processes** | `vkCmdFillBuffer` writes a sequence number into the exported buffer; the consumer's `cuStreamWaitValue64(GEQ)` releases | `TODO.md` Tier 2 item 12 | **PARTIALLY TRIED** — measured in one process; `vkCmdFillBuffer` landed (`d2573840f2e`) | **POST** (08-26, the newest evidence in this file) | `/tmp/interop/doorbell.c`, still on disk: 64-bit stream mem-ops supported, blocks correctly, releases correctly, **1795 ns per write+wait pair** | not a cudavk frame-time lead. The host handshake it replaces is ~0.17 ms against a 13.2 ms frame | medium and **structural**: the consumer must enqueue its wait several frames ahead, and `samples/interop/PROTOCOL.md` is written the other way. `TODO.md` names the fan-out itself as one of three things that could break it | medium + a protocol change |
| L22 | **Async descriptor arena upload** | `cuMemHostAlloc` + `cuMemcpyHtoDAsync` on `cp->stream` instead of a pageable synchronous `cuMemcpyHtoD` | `TODO.md` correctness item 2, Tier 4 item 17 | NEVER TRIED | PRE | a documented race by CUDA's own pageable-copy contract, with **no failure ever observed** | **0.011 ms/frame over 1.00 wait — 0.09% of the blocked time.** Listed only so nobody re-sells it as performance | low | small |

---

## 2. Contradicted or retracted estimates

The tree retracts estimates on purpose. These are the ones a reader will trip on.

**A. Fragment writeback fusion: 0.10-0.20 ms -> ~0.04 ms.** Iteration 28's ranking
listed the larger figure before the reach was measured. `DEAD_ENDS.md` 15 and
`PERFORMANCE.md` §6.3 both state the built item's own number and entry 15 adds
"It must not carry a goal margin."

**B. Fragment-grid sizing: 0.15-0.25 ms -> retracted, closed.** An idle 256-thread
block costs under 0.2 ns, so 74.5 launches/frame carry at most 0.061 ms.

**C. S1b episode deferral: 0.5-1.5 ms -> unsupported.** `addendum.md` §3 and
`PERFORMANCE.md` §6.4: "The earlier 0.5-1.5 ms estimate came from a serialisation
model, and the only direct test of that model produced a negative number; treat the
model as unsupported, not merely unproven." **Now doubly exposed**, because it was
sized against device idle and the fan-out consumed part of that idle.

**D. Raising the `512u << 10` slot budget: withdrawn.** Proposed in `design.md`
§6.1 item 2, withdrawn in `addendum.md` §4 and again in `PERFORMANCE.md` §6.2. The
constant caps over-allocation cost and is load-bearing. **Keep it, or lower it.**
If you implement L3, implement only the clip-rectangle bound and the
`num_triangles <= 2` drop.

**E. S1c, widening the bounded-groups admission at *episode* scope: dead.**
`addendum.md` §4: "real geometry episodes fit no host bound". L3 is the same idea
at *draw* scope and survives. Do not confuse them.

**F. CUDA graphs: a 0.63-0.68 ms ceiling against a 1-3 ms estimate.**
`DEAD_ENDS.md` 1 measures an exact-hit gap ceiling of **0.63-0.68 ms/frame** and
rejects graphs at three units. `notes/CUDA13_UPGRADE.md` §4 reopens them on a
different mechanism (recapture + `cuGraphExecUpdate`, never re-instantiate: 5.3 us
per frame against 43 us of raw launches) and quotes "the tree's own estimate for
whole-batch graphs is 1-3 ms" against 3.287 ms of traced sub-20 us gaps. **The two
documents do not agree and the newer one does not cite the older one's ceiling.**
Entry 1's "Retry if" asks for *stable command-owned execution storage with fixed
device addresses*; the recapture route sidesteps instantiation cost but does not
supply fixed addresses, so it meets the condition only in part. Both figures are
PRE, and both rest on idle-gap totals the fan-out has reduced. Reconcile all three
before opening this.

**G. `DEVICE_AUTONOMOUS_SYNC.md` recommends three things this tree has already
measured as losses.** Its top recommendation is CUDA conditional graph nodes — the
class `DEAD_ENDS.md` 1 refutes three ways. Its long-term recommendation is a tiled
sort-middle architecture — built here and measured at 25.23 -> 35.61 ms
(`OPAQUE_TILING_PROTOTYPE.md`). Its Paradigm VI is a persistent megakernel with
dynamic work queues — the shape `DEAD_ENDS.md` 2 rejects at +140.15 ms of scheduler
cost, and rule 3. It also frames the waits as *synchronisation* where `TODO.md`
states they are *read-back-and-decide*. There is no measurement in the note.
**Do not act on it as written.**

**H. The whole §5/§6 baseline is pre-fan-out.** See §0. This is the most
consequential inconsistency in the tree right now.

**I. `TODO.md:7` contradicts `TODO.md:313`** — "old capture 15.74 ms/frame" against
"a 13.2 ms cudavk frame", in one file. And `PERFORMANCE.md` §1 says the registry
has "97 entries" while `FLAGS.md` says **118 switches** (`d974c6bed6a` moved 16
stray `CPVK_*` `getenv` reads into the registry and took it from 100 to 116; two
more have landed since). `TESTING.md`'s suite count was corrected 65 -> 66 in
`4dc0718a5fa`, and `67f9a158dab` reports 67/67.

**J. `enable_smem_spilling`'s occupancy argument sits against two other findings.**
`notes/CUDA13_UPGRADE.md` §3 argues it survives `SM120.md` because it changes spill
latency and occupancy, not instruction count. But `DEAD_ENDS.md` 10 measured
`launch__waves_per_multiprocessor` at median **0.15**, max 0.60, with 0 of 270
launches reaching 1.0 — if the launches never fill the machine once, more CTAs per
SM buys nothing on *those* kernels. The two kernels the pragma was measured on
(`cp_fs_interpolate`, `cp_abuf_interpolate`) are not the ones entry 10 profiled, so
this is a scoping question rather than a flat contradiction — but answer it before
the work, not after. The same tension applies to L17, which reduces instructions
per sample, where `SM120.md` measured **-49.8% SASS moving the sweep -0.1%**.

---

## 3. Measured prices and hazards that landed in commits and never reached the docs

`git log` carries numbers `PERFORMANCE.md` §4 does not have. Anyone forecasting a
change that adds streams, events or per-episode fixed cost needs these.

**P1 — a cross-stream hand-off costs about 32 us per episode.** `b3f716bd5f5`:
putting an opaque episode's *first* segment on a side stream bought nothing (it has
nothing to overlap) and cost a gate in and a join out. The 600-frame sweep regressed
**0.81 ms of hot sum, `pbribl` by 2.3x and `pushconstants` by 9x, with
bit-identical pixels and unchanged launch counts** — the whole cost was the stream
hand-off. **This belongs in the §4 price table and is not there.** It prices L18's
arena hand-offs, L21's cross-process gating, and any future per-episode stream work.

**P2 — an all-opaque frame has zero overlap.** `0458a51e9a4`: forcing every draw
opaque made the union of kernel intervals equal their sum, 10.79 ms of kernel work
in 10.79 ms of wall clock, while the default arm compressed 12.34 ms into 9.18 ms.

**P3 — the two captures differ in how much there is to fan out.** `02267162955`: an
opaque episode averages **12.42 segments and reaches 49** on old, and **1.71,
maximum 3** on Crossroads. Measured gains 2.61 and 0.08 ms/frame. Read the
Crossroads number as a capture with little to overlap, not as a mechanism that
failed there. This sharpens `PERFORMANCE.md` §4's fourth rule.

**H1 — coalesced uploads are *owed*, not issued, when a draw is appended.**
`2eda3cf74c2`: `cpvk_prepare_draw` reserves push constants and uniform rows in the
upload ring; `cp_stream_set` flushes the owed span **on the stream it is leaving**,
which is exactly wrong when every reader is about to be issued on the stream being
switched to. The symptom was intermittent, ~9,940 of 921,600 pixels wrong on
`multithreading`. The fix is a gate **per segment**, not per episode. **Any lead
that moves work onto another stream inherits this hazard**: L18, L21, and L4 if its
predicated kernels are ever moved off `cp->stream`.

**H2 — two instruments are now invalid at the shipping default.** `7f38d2a9b65`:
`CUDAVK_DEBUG_TIME` records CUDA events into one flat array and differences
consecutive pairs, which is meaningless across eight streams — it can produce
negative numbers. A blended episode already refuses itself for the same reason
(`ab->timing`). So **`CUDAVK_DEBUG_TIME` and `CUDAVK_ABUFFER_TIMING` cannot be used
to measure anything at HEAD's default**, and the fan-out refuses itself when they
are set. Use `CUDAVK_PLAN_STATS` and `CUDAVK_UPLOAD_STATS`, which are host timers.

**H3 — `CUDAVK_TILED_OPAQUE` is refused the fan-out** (`7f38d2a9b65`), so L19's arm
is now handicapped relative to the default by construction, not by its rasterizer.

**H4 — the `getenv` rule is now machine-enforced.** `d974c6bed6a` added
`tests/cp_no_getenv.py`, which fails on a `getenv` in driver code and also catches
`debug_get_bool_option`. Any lead that adds a gate adds a registry entry, and
`cp_debug_doc.py --check` must regenerate `FLAGS.md`. Neither needs a GPU.

---

## 4. Leads that rest on the known-false invariant

The invariant is *"on the direct path a pixel is claimed by at most one fragment
within a launch"*. It was checked rather than believed, and it is **false:
6,343,098 second claimants of a pixel inside a single launch on the old capture**
(`history/PERF16_ITERATIONS.md`, "The direct path's one-fragment-per-pixel
invariant is measured FALSE"; `TODO.md` correctness item 1).

- **L12, fragment writeback fusion — rests on it directly and explicitly.** This is
  the item the check was written for. Note what the evidence does *not* say: a few
  million collisions cannot by themselves blank a background, so the false
  invariant is not the whole explanation of the wrong frames. It is why the safety
  argument as written does not hold.
- **Any future fusion that writes a colour, depth or blend result from the
  generated fragment shader** inherits it. `PERFORMANCE.md` §6.3 states the general
  rule: "Any later work that assumes a direct-path pixel is written once per launch
  is assuming something this capture disproves six million times."
- **Test before designing:** if a design's correctness depends on slot -> pixel
  being injective within one direct-path launch, it is already wrong.
- **Checked and clear:** L17 does *not* assume the invariant — it changes coverage
  *evaluation*, not fragment *ownership*, and `emit_fragment` keeps its `atomicMin`.
  Recorded so the check is not repeated.

---

## 5. Closed — do not retry

Evidence and the exact "retry if" condition are in `DEAD_ENDS.md`.

| what | verdict | the number that closed it |
|---|---|---|
| CUDA graphs at three units (raster tail, per-batch, per-episode) | REFUTED | 0.63-0.68 ms/frame gap ceiling against 27k-44k instantiations; 4,532 submits have 4,532 unique plan generations and **zero** repeated episode recipes. *But see §2F* |
| Persistent / dynamic-claim raster scheduler | REFUTED | +140.15 ms scheduler cost against a ~123.7 ms gap ceiling. Static per-episode chaining is a *different* shape and it paid (+0.2208 ms) |
| Splitting trivial clip acceptance from heavy clipping | REFUTED | +0.850 ms on old; 98.8% of triangles are already trivial |
| Out-of-line (`noinline`) sampler dispatch, per-key fragment execution | REFUTED | every old arm +0.70 to +1.45 ms; 143-182 registers |
| Canonical CUDA mipmapped-array ownership | REFUTED | 1.638% of direct launches; 0.211 ms upper bound |
| Episode-global tagged A-buffer stage 3 | REFUTED (neutral) | -0.0008 ms; one decision costs 244 us x 2,581 |
| Folding the vertex-input bulk clear into the gather kernel | REFUTED | -599.7 operations/frame and **1.8412 ms slower**. The right answer was to delete the buffer (iteration 27), not to re-time the memset |
| Smarter upload flush placement | REFUTED | 76% of the driver's 1,785 flush points/frame already carry nothing |
| Sizing the fragment grid to the machine (`CUDAVK_FS_GRID_WAVES`) | REFUTED | 306,686 fewer blocks/frame moved nothing; an idle 256-thread block is **under 0.2 ns** |
| Kernel-internal redesign of `cp_rasterize_stage3_abuf` | REFUTED | 0.15 waves/SM median, 0 of 270 launches reach 1.0. Largest kernel class (2.48 ms/frame) and **not** a kernel-internal problem |
| Worst-case A-buffer sizing; routing every episode through the bounded path | REFUTED | 8.6 GB refused on old (device loss after ~26 frames); **+0.38 ms slower on Crossroads with a byte-identical hash**. The episode drain pays for itself |
| Resource-isolated classic fragment binary | REFUTED | 64 of 65 capture shaders unchanged; classic lost 1.156 ms |
| Narrow hardware texture paths | REFUTED, then **superseded** | 13.6-14.1% coverage; iteration 24 reached 99.9% and is the default (-5.84 ms) |
| Vertex fetch fused as a **device link** | REFUTED, then **superseded** | 48.05 ms; iteration 27's bitcode inline paid +0.49 ms and is the default |
| Blocking-sync CUDA context (`CUDAVK_CTX_SCHED=blocking`) | REFUTED | +5.4% at idle, **+97.8% under CPU contention**, and it saves no total CPU there either. `yield` is the arm that returns CPU under load |
| Further copy coalescing | EXHAUSTED | all 423.3 copies/frame come from one site, already one per launch boundary; merging at an unchanged boundary pays 0.59 us against 1.30 us for removal |
| Pass-wide opaque draw reordering | REFUTED before any code | opaque groups today vs freely reordered: **6,027 vs 6,027** on Crossroads, **32,387 vs 32,387** on old. Exactly equal — applications already submit sorted by material |
| Immutable-sampler specialisation at pipeline-compile time | REFUTED before any code | **zero** of 382 + 1,088 sampler bindings are immutable; the specialisation measured 34.47 vs 34.63 ms (~0.5%), every broader variant regressing |
| Raising the LLVM target to sm_120 / instruction-count reduction | REFUTED | ptxas already targets the context's device; **-49.8% SASS moved the sweep -0.1%** |
| Upgrading the CUDA driver for performance | REFUTED | driver 580.173.02 already reports `cuDriverGetVersion() = 13000`; no 13.x release reduces per-launch CPU cost, sync APIs are byte-identical, and an upgrade discards 797 MB of cached cubins |
| Making NVRTC 13.3 the default | REFUTED | 5.87 s vs 2.73 s uncached over the five kernels (`cp_rasterize` 0.82 -> 3.83 s) |
| `CPVK_ASYNC_SUBMIT` (asynchronous submission alone) | REFUTED | made `instancing` worse, 1.48x -> 1.59x, because the app fences immediately |
| Putting an opaque episode's **first** segment on a side stream | REFUTED, **POST** | sweep +0.81 ms hot sum, `pbribl` 2.3x, `pushconstants` 9x, bit-identical pixels — ~32 us of pure hand-off per episode (`b3f716bd5f5`) |
| dma-buf / zero-copy WSI | BLOCKED BY THE DEVICE | `DMA_BUF_SUPPORTED=0` on this GeForce; every `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returns `NOT_SUPPORTED` |
| `VK_KHR_external_semaphore_fd`, **export** half | NOT IMPLEMENTABLE | no call in 606 driver API entry points creates or exports a semaphore; 32 of 32 non-NVIDIA fds refused with 999, including a raw `open("/dev/nvidiactl")` |
| Extending the register-cap tuner to sampler-variant fragment shaders | **DONE, not open** | landed: `cp_renderer.c:3598-3610`, "A sampler variant carries its own trial" |
| `CU_CTX_SCHED_BLOCKING_SYNC` as an interop courtesy | DONE, answer was no | `CUDAVK_CTX_SCHED` is in the registry; default stays `auto` |

---

## 6. What is stale and needs re-measuring

For a fresh reader who would otherwise quote 15.75 ms.

| number | where it is written | why it is stale | what replaces it |
|---|---|---|---|
| old 15.7514 ms, Crossroads 5.8982 ms | `PERFORMANCE.md` §1 and §3, `DEAD_ENDS.md` conventions, `TODO.md` :7, `history/PERF16_ITERATIONS.md` | pre-`20611f5b131` | 13.08-13.23 ms old, 5.78-5.82 Crossroads — from two commit bodies, not from any document |
| host blocked 12.435 ms/frame, 16.76 waits, and the four-row wait table | `PERFORMANCE.md` §5.2 | taken at `ba8891878df` (15.99 ms) | **L0**: one `CUDAVK_PLAN_STATS=1` run per capture |
| device idle 4.16 ms/frame (old), 2.45 (Crossroads) | `PERFORMANCE.md` §5.2 | the quantity the fan-out consumed | L0, plus a fresh trace if the gap distribution matters |
| the 2.08 ms "theoretical ceiling" for removing every remaining operation | `PERFORMANCE.md` §5.3 | derived from the 4.16 ms idle | recompute after L0 |
| 3.287 ms/frame of traced sub-20 us device-idle gaps | `notes/CUDA13_UPGRADE.md` §4 and §5 | pre-fan-out; it bounds L5 and the graphs item | a fresh trace, before either is opened |
| kernel time by class, 1,389 kernels and 14.52 ms/frame | `PERFORMANCE.md` §5.1 | taken at `ba8891878df`, before iteration 28 item 4 *and* the fan-out. Launch counts are 91.1/frame high | a fresh profile; the *shape* (raster chain 54.7%) is likely intact, the totals are not |
| GPU busy 74% old / 59% Crossroads | `PERFORMANCE.md` §1 | pre-fan-out; overlap is exactly what changed | `src/cudavk/tests/cp_gpu_busy.sh` |
| "the registry, 97 entries" | `PERFORMANCE.md` §1 | `d974c6bed6a` took it to 116 | `FLAGS.md` says **118**; regenerate with `cp_debug_doc.py` |
| "suite 65/65" | `DEAD_ENDS.md` 15, `PERFORMANCE.md` §6.3, several places | corrected 65 -> 66 in `4dc0718a5fa` | `67f9a158dab` reports 67/67 |
| `CUDAVK_OPAQUE_STREAMS` as an opt-in flag | any script or note written before 08-26 | renamed; the arm is now the default | `CUDAVK_NO_OPAQUE_STREAMS` reverts. A script exporting the old name selects the arm it gets anyway |

**Not stale, and worth saying so:** the §4 price table (1.30 us to remove a small
copy, 0.59 us to merge at an unchanged boundary, 0.66-1.01 us per small clear,
under 1 us per bare same-stream launch, ~1.86 us for an operation that also carries
bandwidth or a whole pass, under 0.2 ns per idle 256-thread block). Those are prices
per operation, not shares of a frame, so the fan-out does not move them. **Add the
32 us cross-stream hand-off (§3 P1) to that table.**

---

## 7. Proposed experiments, in the order they should be run

Not run here. Each is one GPU session; the first three change no output.

1. **E0 (= L0).** `CUDAVK_PLAN_STATS=1` on both captures at HEAD. Restate the four
   waits, the blocked total and the device idle. This is the prerequisite for
   ranking L3, L4, L5, L11, L15 and L18 honestly.
2. **E1 (= L1, P1).** One `fprintf` of `total`/`quads` per episode at the tail
   drain; histogram at teardown. Decision rule (`addendum.md` §5): median ratio
   <= ~3 means L11 is worth building; large means every bound-based sizing scheme
   is dead and site 1 is dead permanently.
3. **E2 (= L2, P2).** Time the inter-drain host issue burst. Decision rule: tens of
   microseconds kills L18 outright; hundreds gives it a case. Run E1 and E2 in one
   session, on both captures, with no mechanism in the tree.
4. **E3.** Re-A/B `CUDAVK_INLINE_FS=1` at HEAD with `cp_decisive_ab.sh` (= L10).
   One binary, both arms, one session.
5. **E4.** Threshold sweep (= L13). Points first, because they are provably
   output-neutral; triangles only with the `gltfscenerendering` `atomicMin` tie in
   mind.
6. **E5.** Only then build L3, then L4 — each measured on its own, each with a
   revert flag in the registry, each proved by a byte-identical hash where the
   change is meant to be behaviour-preserving.

Standing rules that apply to all of the above, from `CLAUDE.md` and `TESTING.md`:
check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` is
empty before any timing run and discard runs whose h2d exceeds ~0.16 ms; a frame is
two `vkQueueSubmit` events and the median is the paired-submit median over the hot
tail; both arms of a comparison belong in one session; build through the codec
build script and never with `make`; and prove a behaviour-preserving change with
the bitstream, not with a benchmark. Do **not** reach for `CUDAVK_DEBUG_TIME` or
`CUDAVK_ABUFFER_TIMING` — see §3 H2.
