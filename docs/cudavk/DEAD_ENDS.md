# Dead ends

Work that was built or measured, did not pay, and is not in the driver. It is
written down so the next person can either **reassess** an entry against new
evidence or **avoid re-implementing** it.

Every figure here comes from a file in the tree or from the raw evidence under
`/tmp/perf16/`, and each entry says which. Nothing is estimated unless it is
labelled as an estimate.

## How to read an entry

- **REFUTED** — the thing was measured and it lost or was neutral. Do not retry
  it without new hardware or a new mechanism. Each entry names what would have
  to change.
- **PARKED** — the work is incomplete and its next step is known. The entry
  gives that step.

Conventions used throughout, from `docs/cudavk/TESTING.md` and the
iteration record:

- **old capture** is `headless_streamer_20260814T155742.gfxr`; **Crossroads** is
  `headless_streamer_1818_20260817T173522.gfxr`.
- **A frame is two `vkQueueSubmit` events.** Medians are paired-submit medians
  over the hot tail. A median from a run that died early is meaningless: check
  the submit count first (an iteration-29 probe first read as +8.07 ms and was a
  device loss after 26 frames, 52 submits against 3,022 —
  `/tmp/perf16/iter27-vsfusion/handoff.md`).
- Both arms of a comparison belong in **one session**. Control drift between
  sessions once faked sub-additivity (iteration 26).
- The shipping default today is **12.7826 ms** on old and **5.6936 ms** on
  Crossroads, with PDL landed and on at level 3 (`6e6e00968ca`;
  `docs/cudavk/PERFORMANCE.md` §1, raw run in
  `docs/cudavk/history/perf-2026-08-27/pdl_landing.md`). Two older figures
  appear in entries below and **neither is today's driver**:
  **13.1626 / 5.8230** is the pre-PDL baseline at `e2fea470d04`, and
  **15.7514 / 5.8982** is that same driver again with the opaque episode fan-out
  reverted (`20611f5b131`). Each step is one commit and a scheduling change, not
  a re-measurement, so an entry's *ratio* to its own baseline survives while its
  share of the frame does not.
- Entries 17 to 23 come from the 2026-08-27 session, whose 52 raw reports are
  archived under `docs/cudavk/history/perf-2026-08-27/`. Each entry names the
  file its numbers came from.
- Evidence files and history documents were written when the driver was called
  `cudapipe`, so they name flags `CUDAPIPE_*`. The current names are `CUDAVK_*`
  (`src/cudavk/FLAGS.md`). This document uses the current names.
- `/tmp/perf16/` is scratch, not backed up. Where an entry says a patch exists
  only there, that is literally true.

Open questions and live leads are in `docs/cudavk/TODO.md`, not here. Full
iteration-by-iteration context is in
`docs/cudavk/history/PERF16_ITERATIONS.md`; the measured shape of the frame is
in `docs/cudavk/PERFORMANCE.md`.

## Index

| # | dead end | iteration | verdict | headline |
|---|---|---|---|---|
| 1 | CUDA graphs, at three different units | 12, 19, 20 | REFUTED | 0.63–0.68 ms ceiling against 27k–44k instantiations; zero recipe reuse |
| 2 | Persistent / dynamic-claim raster scheduler | 21 (and 3) | REFUTED | +140.15 ms scheduler cost against a ~123.7 ms gap ceiling |
| 3 | Split trivial clip from heavy clip | 9 | REFUTED | +0.850 ms on old, 98.8% of triangles already trivial |
| 4 | Out-of-line (noinline) sampler dispatch | 22 (16–18) | REFUTED | every old arm +0.70..+1.45 ms; 143–182 registers |
| 5 | Canonical CUDA array ownership | 23 | REFUTED | 1.638% of direct launches, 0.211 ms upper bound |
| 6 | Episode-global tagged A-buffer stage 3 | 25 | REFUTED (neutral) | −0.0008 ms; the one decision costs 244 µs × 2581 |
| 7 | Vertex-input bulk clear folded into the gather | 26 S2b | REFUTED | −599.7 operations/frame and **1.8412 ms slower** |
| 8 | Smarter upload flush placement | after 27 | REFUTED | 76% of flush points are already empty |
| 9 | Sizing the fragment grid to the machine | 7, after 27 | REFUTED | an idle 256-thread block costs under 0.2 ns |
| 10 | Kernel-internal redesign of `cp_rasterize_stage3_abuf` | 25 (NCU) | REFUTED | 0.15 waves/SM median, 0 of 270 launches reach 1.0 |
| 11 | Worst-case A-buffer sizing / the bounded path for every episode | 29 S0 | REFUTED | 8.6 GB refused on old; +0.38 ms slower on Crossroads |
| 12 | Resource-isolated classic fragment binary | 5 | REFUTED | 64 of 65 capture shaders unchanged; classic lost 1.156 ms |
| 13 | Narrow hardware texture paths | 6, 8, 15 | REFUTED, then **superseded** | 13.6–14.1% coverage; iteration 24 reached 99.9% |
| 14 | Vertex fetch fused as a device link | 4 | REFUTED, then **superseded** | 48.05 ms; iteration 27's bitcode inline paid +0.49 ms |
| 15 | Fragment writeback fused into the shader | 28 item 3 | **PARKED** | wrong frames; the central invariant is measured FALSE |
| 16 | Blocking-sync context to coexist with an ML pipeline | interop | REFUTED | +97.8% frame under load, and it saves no CPU there either |
| 17 | Deferring the driver's object-destruction drains | 2026-08-27 item 1 | REFUTED | the driver owns 55.5% of the site by count and **3.0% by blocked time** — 0.0066 ms/frame |
| 18 | Wider memory requests in the rasterizer stages | 2026-08-27 item 3 | REFUTED | one 1.00 sectors/request was an artefact, the other is the **optimum**; SoA would make it worse |
| 19 | Not issuing the degenerate stage-3 launches | 2026-08-27 item 3 | REFUTED | capped at **0.164 ms/frame** without the mix; the host cannot know the count without a readback worth 100× the launch |
| 20 | Merging the per-segment raster launches (the wide merge) | 2026-08-27 item 4 | REFUTED | removing **298.8 launches/frame cost 0.410 ms/frame** — the credit inverted |
| 21 | Merging the A-buffer count-phase launches | 2026-08-27 item 4 | REFUTED | refuted before it was built: its dominant stage is already **1.89× self-overlapped** |
| 22 | **The merge rule** | 2026-08-27 | **RULE** | a launch-removal credit is only collectable where the launches were **serial** |
| 23 | Hoisting the shading-group tables above the drain (Tier 1) | 2026-08-27 item 2 | REFUTED | measured zero on both captures; one hash across both arms on ten runs |
| 24 | The opaque sort-middle tiling prototype, and a v2 of it | 2026-08-27 audit | REFUTED | best case **2.1–2.6 ms/frame** against the **+2.73 ms** fan-out it has to surrender |
| 25 | Device-side episode chaining: CDP2 tails, predicated pre-issue, conditional graphs | 2026-08-30, HeadlessStreamer | REFUTED | a device tail launch costs **7.3–8.7 µs against 1.5 host** on sm_120; the admissible unit is 0.037 ms/frame |
| 26 | Emptying the stage3→fs_compact link: interp-in-argblock and compact PDL | 2026-08-30, HeadlessStreamer | REFUTED | INTERP_INLINE **+0.13 ms slower**; COMPACT_PDL inert without it; both in-tree, default off |
| 27 | Scope-level concurrency: overlapping independent render scopes | 2026-08-30 census | REFUTED | the scope DAG is a chain — 10.0 scopes/frame at depth 8.0–9.0, max width 2 |
| 28 | Allow hardware-inline fragment shaders whose PTX still uses local memory | 2026-08-30, HeadlessStreamer | REFUTED (wrong frames) | +0.077/+0.036 ms timing, but **15/18 favorite3 and 14/18 favorite2 sentinels differ** |
| 29 | Reuse device-only scratch high-water instead of the 1 GiB context drain | 2026-08-30, favorite3/favorite2 | REFUTED (below noise) | removes 1.65/1.19 ms blocked, but only +0.037/+0.012 ms post-dead-scope with pair signs disagreeing |
| 30 | Episode-entry stale-depth Hi-Z at 8-pixel tiles | 2026-08-30, favorite3 | REFUTED | complete admitted pool is at most **0.103 ms/frame** |
| 31 | One-visbuf depth-only chain deferral | 2026-08-30, favorite3 | REFUTED | complete compact + shader + writeback safe pool is **0.170 ms/frame** |
| 32 | Immutable two-slot segment-0 shading overlap | 2026-08-30, favorite3 | REFUTED | useful compact + shader + writeback pool is **0.036 ms/frame** |
| 33 | Cross-frame retained raster-output recurrence | 2026-08-31, favorite3 | REFUTED | isolated appending census leaves only **0.206 ms/frame** collectible union-exclusive |
| 34 | Fully threaded Mesa runtime submit | 2026-08-31, favorite3 | REFUTED | completion-aware alternating census improves only **0.126 ms/frame** |
| 35 | Exact within-batch post-transform vertex reuse | 2026-08-31, favorite3 | REFUTED | generous weighted union-exclusive upper is only **0.438 ms/frame** |
| 36 | Renderer 2: scope-level tile-binned shading | 2026-08-31, both captures | REFUTED at design | walk is **129x** cheaper than entry 24, yet the ceiling is **0.409 ms/frame** and the impossible upper bound still lands at **5.21 ms** against a 5.0 goal |
| 37 | Host-visible memory residency advice (the B200 "page ping-pong") | 2026-09-01, both GPUs | REFUTED, **and its premise was a profiler artefact** | the migration it targeted exists only under nsys 2026.1.3; the advice costs +1.98 ms/frame on RTX and +0.36 on B200 |
| 38 | More pass side streams than 8 (16 and 32 lanes) | 2026-09-01, both captures | MEASURED, **declined on cost** | real but small: favorite2 -0.084 ms/frame at 16 lanes, favorite3 neutral, for **+153 MB** of device memory |
| 39 | Hoisting the shade chain's argument blocks above the rasterizer | 2026-09-02, both captures | REFUTED | two copies became one, and the frame got **slower**: +0.085/+0.045 ms/frame, arms disjoint |

---

## 1. CUDA graphs, at three different units — REFUTED

Iterations 12, 19 and 20. (Iterations 15–18 are sampler work, not graphs; the
graph line is 12, 19, 20, with the architecture note after iteration 4.)

**Tried.** Three units, smallest first. Iteration 12: an explicit update-free
graph for the raster tail (fused clip+s1 → stage2 → stage3), cached on exact
CUfunction, argument bytes, queue set and allocation epoch. Iteration 19: one
graph per immutable direct batch, from a record-time recipe with command-owned
device storage. Iteration 20: one graph per completed opaque episode, the
driver's normal decision-free workload.

**Promising because.** The traced frame carried ~3.29 ms/frame of sub-20-µs
device-idle gaps (iteration 12) and later ~5.7 ms/frame of gaps (iteration 19),
and the raster tail's two handoffs alone were ~0.81 ms/frame traced.

**Measured.**

- Iteration 12: 318,454 candidate three-kernel tails but only **65,905 exact
  keys**; reuse distance p50/p90/p99/max 155/750/25,343/50,857. A bounded LRU
  hits only **69.2–76.5%** at 256–4096 entries, and even an optimistic two-touch
  ghost policy needs **27,563–43,944 instantiations**. CUDA 12.8 microbenchmark:
  **14.8–15.1 µs** to instantiate, **44–46 KiB** retained per graph+exec, so
  0.27–0.43 ms/frame of build cost and an 11–183 MiB cache against an exact-hit
  gap ceiling of **0.63–0.68 ms/frame**. Plain fused tails recur well (90.3% at
  256 entries), REUSE tails 0.87%, A-buffer 68.5%.
- Iteration 19: 4,532 command-buffer submits reuse seven mutable native command
  objects; 307,811 batches, 496,497 episode attempts, 12,999 non-appending
  unblended calls and **zero strict whole-direct candidates** (9,250 strips,
  2,944 points, 664 triangle-list MSAA, 141 other). Samples expose two chains a
  frame under 0.5 ms absolute. Command-owned storage would copy ≥446 MiB.
- Iteration 20: 5,902 opaque episodes, 73,280 segments, 166,694 draws, 1.314 M
  joined device operations; all internal opaque gaps sum to **1.373 ms/frame**
  and an optimistic semantic LRU ceiling is 0.900 ms one-touch / 0.677 ms
  two-touch. Decisively: 4,532 submits have **4,532 unique plan generations and
  zero repeated episode recipes within a generation**. Retained graphs cost
  ~1.46 GiB RSS + 632 MiB device for 256 large graphs; shared scratch reaches
  1.03 GiB on old and 4.65 GiB on `multithreading`.

**Mechanism.** The Driver API is not the blocker — a 12-node graph instantiates
in 5.8 µs median (iteration 19). The *unit* is wrong at every size tried. Keys
rotate because scratch pointers rotate, and the command plan is re-recorded
every submit, so a cached graph is never asked for twice inside the generation
that could use it.

**Retry if.** Command recording changes so that plans are reused within a
generation, or a workload appears with real recipe recurrence. That means stable
command-owned execution storage with fixed device addresses, not a bigger LRU
over rotating scratch. Iteration 20 closes graphs "until command recording or
workload reuse changes materially".

**Cost.** Three iterations, **no production source in any of them** — all three
were measure-first rejections paid for with census instrumentation and a
microbenchmark (`/tmp/perf16/iter12-report.md`, `iter19-report.md`,
`iter20-report.md`, `iter19-graph-microbench.c`). This is the cheapest way three
iterations have ever been spent here.

---

## 2. Persistent and dynamic-claim raster schedulers — REFUTED

Iteration 21, with iteration 3's producer-local variant as the first refutation.

**Tried.** One launch holding a bounded machine-wide stage-2 producer population
and a persistent machine-scaled stage-3 consumer population, with dynamic
machine-wide claims, device-scope release/acquire publication and the existing
shared queues. Consumers steal all tiles rather than draining their own
producer's work — which is exactly the flaw that sank iteration 3's
`clip+s1+s2` fusion (27.21/27.27 ms against 23.42–24.07).

**Promising because.** stage2+stage3 owned roughly 4.5 ms of a ~7.4 ms/frame
raster chain, and every chain paid a launch boundary and a queue handoff.

**Measured.** The implementation was correct and cheap in resources: 44/44
tests, matching classic/persistent hashes, 56 registers, 92 bytes shared, no
spill, 18 blocks/SM. In an 8-second trace classic stage2+stage3 took
**1,158.61 ms against 1,298.76 ms merged, +140.15 ms**, while the complete
paired gaps on offer were only **~123.7 ms**. Per chain, direct adds 3.894 µs to
remove a 1.642 µs mean gap; A-buffer adds 1.751 µs to remove 2.853 µs. Old
reverse controls moved 23.686→23.456 then 23.595→23.603: +0.111 ms aggregate and
non-reproducing.

**Mechanism.** Dynamic per-item scheduling atomics and work assignment cost more
than the launch boundary they delete. The boundary is worth under 1 µs
(iteration 27); the scheduler is not.

**Retry if.** Per-item dynamic scheduling is eliminated — a static assignment
whose cost does not scale with items. Note that **static per-episode chaining is
a different shape and it paid**: iteration 28 item 4 fused the A-buffer support
chain into two static per-episode passes for +0.2208 ms with no persistent
kernel. Do not read this entry as closing that.

**Cost.** One iteration, 1,073-line prototype fully reverted
(`/tmp/perf16/iter21-rejected.patch`, report `iter21-report.md`), plus an
8-second trace pair.

---

## 3. Splitting trivial clip acceptance from heavy clipping — REFUTED

Iteration 9.

**Tried.** A lightweight accept+copy+stage1 kernel for wholly-inside triangles,
appending only crossing primitive IDs to a worklist that a second heavy
clip+stage1 kernel consumes.

**Promising because.** Every fused clip+s1 thread carries the polygon clipper's
~5,248-byte stack even though most triangles take the fast branch. A no-sync
census confirmed the premise: 401,856,992 input triangles, **71.898% inside,
26.886% common-plane reject, 1.217% crossing** — 98.783% trivial, and 99.795%
trivial on the samples.

**Measured.** The split did what it promised to the resources: lightweight 47–48
registers with 96 bytes local, heavy 56–58 with 5,248 bytes, clip hashes exact.
It still lost. Accept→crossing gaps 1.92/2.53 µs; device sums direct
**30.992 → 41.021 µs** and A-buffer **12.401 → 14.804 µs**; the old capture went
**23.592 → 24.443 ms (+0.850, +3.6%)**.

**Mechanism.** The existing fast branch already avoids touching the polygon
arrays, so the big stack was never being paid on the common path. The split adds
a launch, a worklist and a second classification, and removes nothing real.

**Retry if.** The lightweight/heavy split lives inside one persistent
heterogeneous worker that also absorbs stage 2 — which is the design iteration
21 then measured as a loss. Both would have to change together.

**Cost.** One iteration, 697-line patch reverted
(`/tmp/perf16/iter9-rejected.patch`).

---

## 4. Out-of-line sampler dispatch and per-key fragment execution — REFUTED

Iteration 22, closing the line that iterations 16, 17 and 18 opened.

**Tried.** A canonical exact software sampler as a `noinline` direct-call helper
pruned by operation, target, encoding, sRGB and sampler state, with identical
site bodies deduplicated, plus per-key module dispatch (one exact module per
launch when one key covers it, otherwise a device quad partition and two
per-key modules).

**Promising because.** Generated `main` was 11.1 ms/frame and the 4096×256 class
alone 7.71 ms/frame. Iteration 18 had proved the *arithmetic* exact with 93.38%
A-buffer coverage; only the always-inline packaging had failed there
(hot old **23.434 → 24.605 ms, +5.0%**, standing +6.64%, wall +13.77%). A
noinline boundary looked like the packaging fix.

**Measured.** Correctness was perfect: all 25 classes and 3,276,800 words with
zero mismatches, including fractional LOD. Resources were not. A trivial
one-site RGBA8 shader stays at 54 registers, but real multi-site variants reach
**143–182 registers with 112–136 bytes of caller stack/local**, and only 48 of
83 variants keep two blocks/SM. A real two-key nine-site module is 345,675 bytes
of PTX, 298,944 bytes of cubin and 225,536 bytes of `main` SASS. Every
unprofiled old control regressed: cap-one direct+A-buffer **+1.28/+1.45 ms**,
cap-two **+0.70/+0.95 ms**, cap-two A-buffer-only **+0.97/+1.03 ms**. The quad
partition itself would have cost only ~0.1 ms/frame — the modules lose before it
is reached.

**Mechanism.** The cost is executed instructions and code footprint, not
occupancy. Moving the same sampling work behind another call boundary changes
where the registers are allocated and not how much work runs. Iteration 17 also
found the numerical wall: explicit fractional LOD mismatched 122 of 131,072
words while integer mips were exact.

**Retry if.** A design executes **fewer sampling instructions**, not another
helper boundary. In practice that is the hardware texture path, which is now the
default (entry 13).

**Cost.** Iteration 22 was a 4,009-line prototype in a detached tree with **no
production edit** (`/tmp/perf16/iter22-phaseA.patch`,
`iter22-phaseA-report.md`), on top of three earlier iterations (16, 17, 18) that
proved the mathematics and one of which (18) was implemented and reverted.

---

## 5. Canonical CUDA mipmapped-array ownership — REFUTED

Iteration 23.

**Tried.** Nothing, deliberately: a census first. Join every possible descriptor
row to stable image/view/sampler/memory IDs and a delayed Nsight Systems
fragment-shader range, then admit array ownership only if the full
ownership ∩ texture-object ∩ shader-site intersection was worth ≥2.0 ms/frame
gross.

**Promising because.** Ownership-agnostic row evidence suggested complete
2D RGBA8/R8/BC1/BC3-class A-buffer launches covered 80.03% of A-buffer fragment
time and ~2.48 ms/frame gross.

**Measured.** The census resolved 157,216 fragment launches, 1.152 M rows and
8.733 M site rows with zero missing IDs and zero disagreement; a 12-second trace
joined 16,255 `fsarr` ranges one-to-one to generated `main` over 2,855.455 ms of
fragment device time in 306 submits. **All 1,799 allocations omit explicit image
dedication**, so the strict ownership intersection is zero. Even a future
lazy-allocation effective-exclusive model leaves **2,048 of 125,047 direct
launches (1.638%)** and 7 of 32,166 A-buffer launches — 54.298 of 2,855.455 ms
(**1.902%**), a labelled **0.211 ms** upper bound, with A-buffer at 0.00083 ms.
Both admission gates (≥2 ms gross, >29% direct launches) fail.

The census also corrected a standing belief: the supposed dynamic-descriptor
class was the specialiser's four-reference capacity limit. All 39,167 launches
have 11–16 complete static references and no dynamic index. The class is
expensive — 64.1% of fresh direct fragment time — but array ownership does not
reach it.

**Mechanism.** Applications do not dedicate their images, and the driver's own
allocations are shared, so there is nothing for a canonical array to own.

**Retry if.** The driver owns image allocation itself (lazy, effectively
exclusive) *and* a re-run census shows the intersection above the 2 ms gate. The
census patch is on disk and re-runnable.

**Cost.** One iteration, 1,750-line census patch, **no production code**
(`/tmp/perf16/iter23-census.patch`, `iter23-report.md`, tables under
`/tmp/perf16/iter23-*`).

---

## 6. Episode-global tagged A-buffer stage 3 — REFUTED (measured neutral)

Iteration 25.

**Tried.** Replace the per-segment A-buffer COUNT stage 3 of a pass episode with
one global tagged launch: stage 2 publishes each segment's exact arguments from
the device into a job table, one 64-bit `atomicAdd` reserves a triangle's whole
tile run, and `{tri_id, tile_x, tile_y, job_id, flags}` pairs go to a separate
episode-lifetime queue. The eight per-stream classic queue sets are untouched.

**Promising because.** It removes launches in the largest kernel class, and the
host enqueues the single stage 3 straight behind the checked producer join with
no synchronisation, no tail read and no job upload.

**Measured.** Corrected two-replay AB/BA paired-submit medians, same binary both
arms, identical stdout hashes on both captures:

| capture | mechanism | control | delta |
|---|---:|---:|---:|
| old | 17.4509 ms | 17.4501 ms | −0.0008 ms (−0.004%) |
| Crossroads | 6.2784 ms | 6.2501 ms | −0.0283 ms (−0.452%) |

It removed **123.8 classic COUNT stage-3 launches per frame on old and added
11.6** (net −112); on Crossroads it removed only 22.4 and added 7.2, because
those episodes average 1.36 segments.

**Mechanism**, measured rather than inferred. The mechanism needs exactly one
pre-fragment decision, and splitting the wait by call site shows where the
saving went:

| capture | bounded checks | bounded wait | main checks | main wait |
|---|---:|---:|---:|---:|
| old | 2581 | 630.3 ms — **244 µs each, 0.417 ms/frame** | 14932 | 2.0 ms total |
| Crossroads | 2442 | 556.1 ms — 228 µs each, 0.371 ms/frame | 8333 | 1.2 ms total |

The main path's check is free because that path already drained. The
**bounded-group path did not**: it ran entirely asynchronously, and the one
required decision introduces a full synchronisation there worth about as much as
the launches removed. Merging the per-segment COUNT launches also serialises onto
the main stream work that had been spread over eight side streams.

**Retry if.** The overflow and corruption bounds are proven **on the device**, so
the bounded-group path keeps its asynchrony. Restart from the recorded design and
gates, not from scratch — the gate set already proves two full 64-job episodes,
job 63 produced and consumed, exact capacity, capacity+1 falling back before any
fragment shader and then growing in the same context, five fault seams and a
state-leak negative control.

**Cost.** One iteration; a 1,684-line mechanism with a device ABI, a second
stage-2/stage-3 kernel pair and six switches, all reverted
(`/tmp/perf16/iter25-full-mechanism.patch`,
`iter25-cp_renderer.c.mechanism`, `iter25-mechanism-report.md`). Not wasted:
the fail-closed hardening it exposed on the default path stayed —
`cp_pass_join()`/`cp_pass_broadcast()` report failure and every caller stops,
episode clears and the gate event are checked, the rollback rechecks
`device_fatal`, and the count pass checks the queue-counter clear.

---

## 7. Folding the vertex-input bulk clear into the gather kernel — REFUTED

Iteration 26, stage S2b.

**Tried.** Remove the packed vertex input buffer's pre-clear by having the gather
kernel write the zeroes itself, row by row.

**Promising because.** The S0 census counts every small device operation at its
call site, and this clear was 203.7 calls a frame. Folding it hit its predicted
target exactly: clears fell from **968.94 to 369.20 per frame, −599.7**.

**Measured.** Output identical, and the frame **1.8412 ms slower on old** and
0.2640 ms slower on Crossroads.

**Mechanism.** The clear is small by call count and **bulk by bytes**: only
117.4 of its 203.7 calls a frame are under 4 KB, and it moves about **90 MB a
frame**, which `cuMemsetD8Async` does in one kernel at DRAM speed. Row-by-row
inside the gather kernel lost even with 16-byte stores.

The rule: **fold an operation for its count only when its bytes are negligible.**
Counters are; buffer clears are not.

**Retry if.** Never in this form. The correct answer arrived one iteration later
and is in the driver: iteration 27's fused vertex fetch removes the buffer
itself, and with it **84.9 MB/frame of clear traffic** and 194.3 clears a frame.
Delete the data, do not re-time the memset.

**Cost.** One stage of one iteration. The losing path was deleted and the site
carries a comment so it is not re-derived.

---

## 8. Smarter upload flush placement — REFUTED

Measured after iteration 27, recorded in the iteration record and in
`/tmp/perf16/iter28-grid-doc.md`.

**Tried.** The question, not a mechanism: after S3 made it one host-to-device
copy per launch boundary, is there more in choosing *better* flush points?

**Promising because.** S3's merge ratio was only 1.82 blocks per copy, which
looks like a merging failure.

**Measured**, one binary, both arms, old capture, 1,511 frames
(`/tmp/perf16/iter27-census/`):

| arm | blocks/frame | copies/frame | blocks per copy | empty flush points/frame |
|---|---:|---:|---:|---:|
| default | 914.07 | 423.31 | **2.159** | 1362.08 |
| reverted | 914.06 | 615.31 | 1.486 | 1364.35 |

There are **1,785 flush points a frame and only 423 carry anything: 76% are
already an empty predicate.** Removing 194.3 launches a frame raised the merge
ratio by 45%, exactly as S3's model predicted.

**Mechanism.** Every launch is a flush point and the launch on the far side is
what reads the bytes, so an upload cannot be deferred past it. The remaining
copies are one per boundary that carries data. **The lever is removing the
boundary, not flushing more cleverly.**

**Retry if.** Uploads stop being owned by the launch boundary — a different
ownership model, not a better heuristic. Otherwise this number only improves as
a side effect of removing launches.

**Cost.** A census that already existed (`CUDAVK_UPLOAD_STATS=1`) and one A/B
run. Also note S3's stale figure: quote the same-binary table above, not the
1.82 recorded during iteration 26, whose block count included a per-fetch-launch
contribution that no longer exists.

---

## 9. Sizing the fragment grid to the machine — REFUTED

Iteration 7, and again after iteration 27 (commit `5a3315bca60`,
`/tmp/perf16/iter28-grid-doc.md`).

**Tried.** Cap the direct fragment grid at `SMs × blocks_per_sm × W` instead of
the framebuffer's worst case of 4,096 blocks, with occupancy taken from the
execution actually being launched. `CUDAVK_FS_GRID_WAVES=W` still does this.

**Promising because.** The grid always hits its 4,096 cap; iteration 7 measured
the exact compacted count at p50 **3,108** against 1,048,576 physical lanes —
only **4.406%** of first-wave lanes useful. NCU measures 12.05 waves per SM with
65% of launches running under 50 instructions per thread. It looks like eleven
of twelve waves exist to discover they have nothing to do.

**Measured.** The change reaches the launches it aims at: at `W=1`, 157,212
launches schedule **39,147,304 blocks against 502,548,728** (3,196.6 → 249.0
blocks per launch), **306,686 fewer blocks scheduled per frame**. The frame does
not move. Cycling settings so drift is shared, three cycles on old and two on
Crossroads:

| waves | old median | vs today | Crossroads median | vs today |
|---|---:|---:|---:|---:|
| 0 (today) | 15.9899 | — | 5.9786 | — |
| 1 | 15.9903 | −0.0005 | 5.9662 | +0.0124 |
| 2 | 15.9720 | +0.0179 | 5.9668 | +0.0118 |
| 4 | 15.9344 | +0.0555 | 5.9903 | −0.0117 |
| 8 | 16.0472 | −0.0573 | 5.9692 | +0.0095 |

`W=1` alone spans 15.9465–16.0075, so the whole column is one distribution.
Untraced utilisation is 75.2% at `W=0` and 72.1% at `W=1`. Iteration 7 had
already measured fixed grids 340/680/1024/1360/2048 within 0.16% of the 4,096
controls.

**Mechanism.** **306,686 scheduled blocks a frame cost under 0.06 ms, which is
under 0.2 ns for an idle 256-thread block on this GPU.** A block that reads a
device-side count and exits is free. A 4,096-block grid therefore carries at most
0.82 µs of empty-block cost, ≤0.061 ms/frame over 74.5 launches, while the
direct fragment launch's observed floor is 2.72 µs — most of that floor is the
launch and the compaction read. The NCU instruction mix is real; the conclusion
that the grid was the cost is not. An earlier 0.15–0.25 ms/frame estimate for
this candidate was **retracted** on this evidence.

**Retry if.** Different hardware. The flag stays at `0` — byte for byte today's
behaviour — precisely so that re-measuring costs one command rather than a
re-implementation, and the fragment-grid census stays because it is what proves
a grid change reached the launches it aimed at.

**Cost.** Iteration 7 was a 375-line patch, reverted
(`/tmp/perf16/iter7-rejected.patch`). The re-measurement was one flag, one
census and a cycled sweep, both retained in the tree.

---

## 10. Kernel-internal redesign of `cp_rasterize_stage3_abuf` — REFUTED

Iteration 25's NCU pass, restated in `/tmp/perf16/iter28-profile/report.md`.

**Tried.** Nothing yet, by design: the pass/fail criteria were registered before
any counter was read, and the first one fired.

**Promising because.** It is the largest single kernel class in the frame —
**2.484 ms/frame in 134.6 launches/frame**, 17.1% of kernel time (iteration 28
budget) — and it was the obvious candidate for a body-level rewrite.

**Measured**, Nsight Compute 2025.1.1.0 on `gb202`, two independent windows of
135 launches each:

| | window 1 (skip 40,000) | window 2 (skip 120,000) |
|---|---:|---:|
| `launch__waves_per_multiprocessor` median | **0.15** | **0.15** |
| p90 / max | 0.60 / 0.60 | 0.60 / 0.60 |
| launches reaching ≥ 1.0 wave | **0 of 135** | **0 of 135** |
| grid | 512–2048 blocks × 64 threads | same |
| registers | 46 | 46 |
| achieved / theoretical occupancy | 9.05% / 83.33% | 8.93% / 83.33% |
| `sm__throughput` | 0.38% | 0.38% |
| `gpu__dram_throughput` | 0.10% | 0.10% |
| `smsp__issue_active` | 5.32% | 5.07% |

**Mechanism.** The median launch is 512 blocks of 64 threads — about 32,768
threads, roughly 3 CTAs and 6 warps per SM on a 170-SM card. Achieved 9% against
theoretical 83% with 46 registers is not a register or shared-memory limit:
**there is not enough work in the grid.** The kernel is geometry/launch bound.
No kernel-internal redesign is justified; the work belongs in fewer, larger
launches.

**Retry if.** The work per launch grows — that is, an episode-scale batching
change that pays for itself first — and then re-run the A1 criterion. Do not
start from the kernel body.

**Cost.** No code. One smoke run (46 s) plus two profiled windows (112 s and
~110 s) and a fragment-shader window (~150 s) on an exclusive GPU
(`/tmp/perf16/iter25-ncu/`, method and criteria in
`/tmp/perf16/iter25-profile/report.md` §8–§9).

---

## 11. Worst-case A-buffer sizing, and routing every episode through the bounded path — REFUTED

Iteration 29 probe S0. Design `/tmp/perf16/iter29-sync/design.md`, verdict
`/tmp/perf16/iter29-sync/addendum.md`.

**Tried.** `CUDAVK_UNSAFE_NO_OVERFLOW=1`, which already routes every episode
through the asynchronous bounded path, as an upper-bound probe on removing the
episode drain. The drain is the driver's single most expensive host wait:
**8.663 ms/frame over 9.88 waits, 0.877 ms each** on old.

**Promising because.** The host is blocked 12.44 ms of a 15.99 ms frame (77.8%)
and the device is idle 4.16 ms (26.0%). If the sizes were bounded instead of
counted, the host would not block.

**Measured.**

- **Old: invalid, and informative.** Device lost after ~26 frames. The bounded
  path sizes the A-buffer for the worst case and asks for **8,605,856,768 bytes
  against the 8,589,934,592-byte `CP_SCRATCH_MAX_BYTES` cap**; the allocation is
  refused, the draw cannot shade, and the no-replay rule correctly latches
  device loss.
- **Crossroads: valid and negative.** 6.0036 → **6.3860 ms, 0.38 ms slower
  (−6.37%)**, byte-identical stdout, with 8.26 waits a frame removed.

**Mechanism.** `cp_scratch_alloc_device` is not a passive reservation: a request
the arena cannot serve **reallocates the whole arena** to `MAX2(end, size*2)`,
pushes the old base onto `dscratch.overflow[]`, and `cp_scratch_reset` frees
those at **every flush**. So worst-case sizing costs three things per frame, not
one — gigabyte-scale `cuMemAlloc`/`cuMemFree` churn on the flush path, bulk
clears that scale with the bound (the `discard_mask` memset over
`ALIGN_POT(4*num_quads, 256)`), and the footprint itself. **The expensive
quantity is the ratio bound/actual, not the wait.**

The finding to keep: **the episode drain pays for itself.** Its 0.877 ms buys the
exact quad count that sizes the shade arrays, and the bounded path's worst-case
bound is not a cheaper way to get the same number. The `512u << 10` slot-budget
constant is therefore load-bearing — it is a cap on over-allocation cost, and the
proposal to raise it is withdrawn.

**Retry if.** A scheme keeps the over-allocation factor a small constant instead
of four orders of magnitude. One candidate survives and is a different shape:
take the drain at `cp_abuf_scan`, which already writes the exact fragment total
and the clamp flag near the *start* of the episode chain, so the factor becomes
`total/quads` rather than `capacity/quads`. Decide it with one `fprintf` — log
the `total`/`quads` pair the host already holds at the tail drain and report the
distribution; a median ratio ≲3 makes it worth building. That lead lives in
`docs/cudavk/TODO.md`.

Everything that routes episodes through the bounded path is closed, including
wider bounds, clip-rectangle bounds and a per-context arena sized to the array
capacity: they all inherit the ratio.

**Cost.** One probe with an existing flag and no code — and it killed a designed
0.5–1.5 ms iteration item before it was written.

---

## 12. A resource-isolated classic fragment binary — REFUTED

Iteration 5.

**Tried.** Keep a truly bare classic fragment binary with no reference to
`cp_fs.cu` beside the helper-linked fused binary, then A/B the whole chain:
classic interpolator + bare fragment shader against slim compaction + fused
shader.

**Promising because.** Iteration 4 had proved that a linked call graph retains
union register allocation, and the direct 236-register and 127-register groups
alone cost 4.11 ms/frame. Recovering an occupancy tier looked like 1–3 ms.

**Measured.** Resource isolation worked exactly as intended on a focused test:
`cpvk_tri`'s bare fragment shader dropped **190 → 18 registers** and 1 → 6
blocks/SM. On the capture it changed nothing: for **64 of 65 old-capture
fragment shaders**, classic and fused had identical uncapped facts — 203 or 236
registers, identical spills, one block/SM. Interleaved full replays: fused
**23.652/23.531 ms**, bare classic **24.764/24.730 ms**, so classic **lost
1.156 ms (+4.9%)**, almost exactly the cost of the restored separate
interpolator. Specialisation stayed at 28.8%, so this was not an admission miss.

**Mechanism.** The shared software sampler call graph fixes the allocation.
Isolating the fragment binary does not remove `cp_sampler.cu`, so the registers
do not move where it matters.

**Retry if.** Already answered, and kept: the way to remove the sampler graph is
to stop executing software sampling (entry 13) and to keep interpolation in the
same LLVM module (iteration 14, `CUDAVK_INLINE_FS`). Resource isolation itself
survives in the driver as the `HW_FUSED` execution mode, which recovers shaders
that keep helpers or local memory.

**Cost.** One iteration, 1,295-line prototype reverted
(`/tmp/perf16/iter5-rejected.patch`, report `iter5-report.md`).

---

## 13. Narrow hardware texture paths — REFUTED, then superseded

Iterations 6, 8 and 15. **This is the entry to read before declaring anything
dead on a coverage number.**

**Tried.** Three narrowings of the same idea. Iteration 6: CUDA PITCH2D texture
objects for compatible 2D sampled-image/sampler pairs, with NVVM texture
operations in the generated shader. Iteration 8: the same, but with
`cp_fs_compact` pre-interpolating so a *bare* hardware fragment shader could run
with no `cp_fs.cu` link. Iteration 15: the same execution built in one LLVM
module.

**Promising because.** The operation itself is dramatically faster, and it was
measured on a bounded matched chain (iteration 8): software fragment shader
**25.206 → 2.688 µs**, slim compact 6.410 → 13.224 µs, and the **whole admitted
device chain 33.348 → 17.648 µs** (−15.700 mean, −5.344 paired median).

**Measured.** Every version was neutral or slower on the frame:

| iteration | resources reached | coverage | frame |
|---|---|---:|---|
| 6, uncapped | 203 → 190 regs, still 1 block/SM | **14.1%** (21,991/156,159 launches) | 23.724 HW vs 23.504 SW (+0.220 ms) |
| 6, capped | 126 regs / 2 blocks, 104–136 B spill | 14.1% | 23.608 vs 23.605 (+0.003 ms) |
| 8, bare | 34–79 regs, 3–6 blocks/SM, no spill | **13.6%** | 23.885 vs 23.818 (+0.067 ms) |
| 15, same-LLVM | 88 regs, 2 blocks, no spill | **13.9%** | 23.366 vs 23.336 (+0.031 ms) |

Break-even was computed, not guessed: **18.7–28.8% direct coverage**. Adding
multi-mip pure-2D draws predicted ~22%, straddling it, and an unrealistic
all-direct upper bound was only 0.31–1.05 ms net.

**Mechanism.** Coverage, not texture instructions. A path that is only admitted
for a seventh of the launches cannot move a frame however fast it is, and the
capped variants spent their gain on spill.

**What changed, and why this entry is a warning.** The recorded retry condition
was explicit — broad mipmapped arrays for multi-mip, cube, 3D and BC resources,
plus interpolation in the same LLVM address-space model. Iteration 24 built
exactly that: authoritative linear images plus epoch-keyed disposable CUDA
mipmapped arrays and immutable texture objects, strict same-LLVM `HW_INLINE`
preferred with resource-isolated `HW_FUSED` as the recovery. Coverage went to
**157,095/157,211 old launches (99.9%)** and 94.6% on Crossroads, worth
**5.8382 ms** (17.2279 vs 23.0661 ms AB/BA). It is the default today; the revert
is `CUDAVK_NO_TEXTURE_CACHE` (old **15.8194/15.7279 ms** against
**21.9152/21.9127 ms** with the revert).

**The lesson: a narrow path measured neutral is a statement about coverage, not
about the mechanism.** Three iterations were spent proving the mechanism worked;
the fourth was spent making it apply.

**Cost.** Three prototype-and-revert iterations
(`/tmp/perf16/iter6-rejected.patch` 1,282 lines, `iter6-capped-followup.patch`
1,287, `iter8-rejected.patch` 1,750, plus iteration 15's restored tree), then
one large iteration that paid.

---

## 14. Vertex fetch fused as a device link — REFUTED, then superseded

Iteration 4, against iteration 27's bitcode inline.

**Tried.** Link the generic `cp_vertex_fetch` helper into the generated vertex
shader as an NVRTC device function and pass an immutable fetch argument block.

**Promising because.** A fresh trace measured 120,894 exact
`cp_vertex_fetch → main` chains in 15 s: fetch 1.358 ms/frame, generated vertex
shader 0.699 ms and ~193 pairs/frame. The opportunity was structurally real —
iteration 27 later collected it.

**Measured.** Two forms failed for two different reasons.

1. A lane-local result block removed the global scratch but was **invalid**:
   LLVM NVPTX local pointers crossed into the NVRTC helper as raw addresses
   rather than converted generic pointers, and Compute Sanitizer found invalid
   local reads.
2. The global-buffer ABI was correct and passed 43/43, but linking the generic
   fetch graph took a trivial vertex shader from **20 to 108 registers** and
   **6 to 2 blocks/SM**. The old capture's partial median became **48.05 ms**
   against 23.42–23.67. `__noinline__` did not isolate the allocation, and
   passing a **null** runtime helper still carried 108 registers — so a null
   call path is not an honest revert control.

**Mechanism.** A device call graph forces union register allocation across the
caller, and two compilers with different address-space models cannot share a
pointer ABI.

**Why iteration 27 succeeded where this failed**, since the two look alike:

- The helper is compiled to **NVPTX bitcode in the same `LLVMContext` as the
  shader and force-inlined before optimisation** — the iteration-14 fragment
  interpolation machinery generalised — so there is no device call and no union
  allocation. Any surviving `cp_vs_fetch_lane` symbol in the emitted PTX is a
  hard stop, checked twice, because that was iteration 4's failure mode.
- The results land in a function-entry alloca that SROA promotes to registers,
  so the packed input buffer, its clear, its round trip, the ID arrays and the
  batch-row array all cease to exist.
- **The shape of the emitted call is the whole mechanism.** A loop bounded by a
  constant `N` and marked `unroll(full)` was **refused by LLVM's unroller for
  every real capture shader** (the body is a whole conversion tree and the
  pragma loses to the size heuristic); the slot array was then indexed
  dynamically, and a dynamically indexed alloca is local memory. Launch-weighted
  admission in that form was **A = 0.1081**. Emitting the unrolled sequence from
  the backend — one call per live slot with the element index an `LLVMConstInt`
  — gives **A = 0.9538 on old and 0.8838 on Crossroads** with no `.local`
  anywhere, and the only decline reason that occurs is losing a block per SM
  (9 shaders / 14,228 launches on old).
- Iteration 4's register blow-up did not reproduce: launch-weighted registers
  went 53.7 → 55.7, and `instancing`'s rock shader went **120 → 104 registers**
  at equal occupancy.

Result: **+0.4925 ms median of medians** on old (16.0194 against 16.5119) and
+0.0787 ms on Crossroads. Two smaller rules came out of the same work and are in
the header comments: never write a slot as bytes in one path and as words in
another (SROA refuses a partition with conflicting access types and takes the
whole array to memory), and bound inner loops with `continue`, not `break`.

**Retry if.** Not applicable — the fusion exists and is default-on
(`CUDAVK_NO_FUSED_VFETCH` reverts). What stays dead is the **device-link form**.

**Cost.** Iteration 4: 622-line patch, fully reverted
(`/tmp/perf16/iter4-rejected.patch`). Iteration 27 then cost a full iteration
plus the iteration-14 machinery it reused, and needed a new per-shader **launch**
census before any performance claim, because a verdict counted per shader says
nothing about a frame when one shader takes two launches and another two hundred.

---

## 15. Fragment writeback fused into the fragment shader — PARKED

Iteration 28 item 3. The only record is
`/tmp/perf16/iter28-item3/README.md` with `wip.patch` (1,590 lines) and
`cp_fs_write.h` beside it; the tree does not carry the work and the agent that
produced it has been retired. **Read the README before touching it.**

**Tried.** Move the per-slot writeback body into a shared header, export
`cp_fs_write_lane()` from the same bitcode module the hardware-inline execution
links, carry a pointer to the existing `cp_fs_writeback_args` in argument slot
17 (inside the fragment argument block's own upload, so no new device
operation), and have the backend emit the call at the end of the shader body
after every `store_output`. The lane reads this thread's own colour from
`fs_out` — same thread, same address, L1, not a kernel-boundary round trip. The
standalone `cp_fs_writeback` kernel stays for declined draws.

**Promising because.** This is the unfinished half of iteration 2, which fused
interpolate→FS and left writeback for later, and the mechanism, the revert
pattern and the gates all existed.

**Measured and TRUE.**

- Admission on the old capture: 36 fusable fragment shaders, **A = 1.0000 by
  launch**, 92 → 94 registers, **no local memory**, no block lost.
- Reach: 65,029 of 125,046 writeback launches over the replay = **43.0 removed
  per frame** (52%, matching the hardware-inline share).
- Expectation at under 1 µs per removed same-stream launch: **~0.04 ms/frame**.
  (Iteration 28's ranking listed 0.10–0.20 ms for this item before the reach was
  measured; the built item's own number is the smaller one. It must not carry a
  goal margin.)
- Suite 65/65 with the flag on.
- The `__local_depot` fix worth keeping: the first build carried 48 bytes of
  local because the colour arrays were accessed at two widths and SROA refuses a
  partition it sees at two sizes. Carrying the colour as four scalars / a
  by-value `float4` removed the depot completely — the same class of finding as
  iteration 27's.

**Why it is parked — the failure.** On the old capture the fused writeback
renders **visibly wrong frames**: whole background quads missing, mean delta
24–30 on 4 of 10 sentinels, ~1.4–2.0 million pixels over tolerance. Ruled out,
with evidence:

- Running fused **and** standalone writeback together restores frames to within
  run-to-run noise (mean ≤ 5e-05). The fused write corrupts nothing; the pixels
  it should have written are simply absent at the end.
- The fused lanes **do** store: ~3.75 billion colour stores over the replay,
  ~2.5 million a frame. The lane runs, has a non-null block, passes coverage and
  the resolved tests, and executes `cp_store_dst4`.
- Not the A-buffer: `CUDAVK_NO_ABUFFER=1` leaves the same two sentinels wrong.
- Not the terminator case: emitting the call before an existing terminator
  changed nothing.

**The invariant is measured FALSE, and this outlives the fusion.** The design
rests on "one fragment per pixel per launch on the direct path". An explicit
checked invariant found **6,343,098 second claimants of a pixel inside a single
launch** on the old capture. A few million collisions cannot by themselves blank
a background, so this is not the whole explanation — but the central safety
argument does not hold as written, and **any later work that assumes a
direct-path pixel is written once per launch is assuming something this capture
disproves six million times.** The check existed, so the assumption was measured
instead of believed.

**Next step, and it is one instrumented run rather than a redesign.** For one
draw, record the slot→pixel mapping the fused lane used and compare it against
what the standalone kernel computes for the same launch: prove or disprove that
the fused lane's slot equals the writeback's `i`. The leading hypothesis is that
the fused stores land for slots whose pixels are not the ones the standalone
launch would have written, or are overwritten later.

**Cost.** One item of one iteration: 1,590 lines plus a new header, built end to
end, suite-clean, then reverted. It also found two build traps that cost real
time and are now written down — `cp_shader_exec_is_hardware()` did not recognise
the new execution mode, so it got a null `CUtexObject` table and painted black
(**adding an execution mode means auditing every predicate over the mode enum**),
and the meson bitcode `custom_target` did not list the new shared header in its
`input`, so header edits silently did not rebuild the bitcode.

---

## 16. Blocking-sync CUDA context to coexist with a co-located ML pipeline — REFUTED

Prompted by the CUDA interop work: the driver is to feed frames to a PyTorch
pipeline on the same host, so the spinning wait threads looked like stolen
cores.

**Tried.** `CUDAVK_CTX_SCHED`, a registry flag selecting the CUDA context
scheduling mode: `auto` (the shipping default, and with one context and 32
logical processors it resolves to `CU_CTX_SCHED_SPIN`), `spin`, `yield`,
`blocking` (`CU_CTX_SCHED_BLOCKING_SYNC`). Measured on both captures, arms
alternated in one session, three pairs per arm per capture, 54 runs across
three load conditions. Competitor: N single-threaded CPU-side PyTorch trainers
with `CUDA_VISIBLE_DEVICES=""`, so it never touches the GPU. Pixels identical
throughout — one stdout sha256 per capture across every arm and every load.

**Promising because.** The driver is blocked ~12.4 ms of a 15.74 ms frame
(`PERFORMANCE.md`), and on an idle machine `blocking` really does give back
51% of host CPU, about half a core. The inference was that a contended machine
would make that trade obviously correct.

**Measured.** Frame ms, delta against `auto`:

| load | capture | auto | blocking | yield |
|---|---|---:|---:|---:|
| idle | old | 15.74 | 16.60 (+5.4%) | 15.64 (−0.6%) |
| idle | Crossroads | 5.86 | 6.03 (+2.8%) | 5.86 (0.0%) |
| moderate (16) | old | 33.46 | 33.78 (+1.0%) | 33.75 (+0.9%) |
| moderate (16) | Crossroads | 13.59 | 13.77 (+1.3%) | 11.64 (−14.4%) |
| saturated (32) | old | 42.94 | **84.95 (+97.8%)** | 42.59 (−0.8%) |
| saturated (32) | Crossroads | 24.26 | **46.55 (+91.9%)** | 18.08 (−25.5%) |

**Mechanism.** The prediction was backwards on both halves.

- **`blocking` gets worse under contention, not better.** At saturation it
  roughly doubles the frame, ranges fully disjoint (old: auto 41.2–46.2,
  blocking 82.0–87.9). Each of the ~17 waits a frame ends in a kernel wake-up,
  and a wake-up behind a full runqueue is not a fixed cost. Spinning is what
  makes seventeen waits a frame affordable.
- **It saves no CPU there either.** The CPU *rate* drops (86% → 55%) but the
  replay runs 62% longer, so total CPU-seconds are equal on old (91.7 vs 90.8)
  and worse on Crossroads (44.6 vs 37.2). "Half a core given back" is an
  idle-machine artefact.
- **The competitor never benefits.** Best case +2.6% throughput, negative on
  the other capture. One spinning wait thread is one core of thirty-two, and a
  16- or 32-way trainer cannot see it. The premise needs a 4–8 core host to be
  testable at all.
- **`yield` inverts too, in the useful direction.** At idle it is a spin with
  syscall overhead — frame-neutral, saves nothing, moves 10 s of user time into
  system time. Under saturation it is the arm that returns CPU: frame-neutral
  on old, 25% faster on Crossroads, CPU per frame down 17–20%.

**Kept anyway.** The flag stays, with corrected guidance: `blocking` for an
otherwise idle machine, `yield` under contention, `auto` everywhere else. Each
arm wins only in the condition the other loses, so none of them can be the
default.

**Caveats.** This is a 9950X3D: 16 physical cores, SMT2, so "moderate" at 16
workers already fills every physical core and frame time doubles against idle
before any arm is varied — at moderate load no arm separates on the old
capture. n=3 per cell. One moderate-load `blocking` run was anomalously fast
(19.5 / 36.5 / 33.8 ms), so no claim rests on that median.

**Retry if.** The host is small enough that one core matters — 4 to 8 logical
processors, not 32 — or the block count per frame falls far enough that
wake-up latency stops being multiplied by seventeen. The second is the real
condition, and it is the same one that makes most of `PERFORMANCE.md`
interesting.

## 17. Deferring the driver's object-destruction drains — REFUTED

2026-08-27, item 1. Evidence:
`history/perf-2026-08-27/item1_census_results.md` and `item1_destroy_defer.md`.

**Tried.** With the texture cache on, destroying an image or a view drains the
whole device (`cpvk_DeviceWaitIdle` plus the cache's own `cuCtxSynchronize`;
`cpvk_image.c:376-379`, `:580-584`, `cpvk_texture_cache.c:933-935`, `:797-799`).
The plan was a retirement queue keyed on the pending submit's event, so a free
waits for that event instead of for the whole device.

**Promising because.** The wait census measured the `vkDeviceWaitIdle` site at
0.514 ms/frame blocked with a 0.500 ms ceiling and a measured +0.44 slope, the
driver reaches that site from three of its own call paths, and it was the only
**wait-bound** site in the driver (3,339 of 3,390), so it would be additive with
everything else. It was ranked first at 0.22 ms/frame.

**Measured.** `CUDAVK_DESTROY_CENSUS`, both captures, GPU exclusivity positive,
and the instrument agreeing with the wait census to 0.00% (1,510 + 814 + 1,067 =
3,391 exactly):

| caller | ms/frame blocked, old | µs per drain |
|---|---:|---:|
| the application's own `vkDeviceWaitIdle` | 0.4847 | 485.0 |
| `destroy_view` | 0.0149 | 21.1 |
| `destroy_image` | 0.0001 | **0.28** |
| **driver share** | **3.00%** | |

Crossroads' driver share is **0.24%**. At the site's measured +0.44 slope the
entire driver share is worth **0.0066 ms/frame**.

**Mechanism.** **Count and cost point opposite ways.** By count the driver owns
**55.5%** of the site (1,881 of 3,391) and the capture-file arithmetic that
predicted that was exactly right. By blocked time it owns **3.0%**, because the
driver's drains arrive at a device that is already empty. `destroy_image` is
*already* a null drain at 0.28 µs. **Deferring frees cannot recover time that is
not being spent.**

A second registered prediction was confirmed on the way past: the texture
cache's second sync is a null sync — 0.19 µs per drain, constant on both
captures, 1.63% and 12.49% of the destroy sites' blocked time, under the 20%
bar. "Destroying drains twice" was a call count, never a cost.

**Retry if.** The workload changes so that destruction happens while the device
is busy — a capture that destroys images inside a frame rather than at pass
boundaries. The mechanism exists and is inert with its flag off
(`CUDAVK_DESTROY_DEFER`, branch `destroy-defer`); it was never measured and
never landed.

**Cost.** One census and one built-but-unmeasured mechanism. The rule it paid
for is in the rules list below and in `WORKFLOW.md` §4.6.

---

## 18. Wider memory requests in the rasterizer stages — REFUTED

2026-08-27, item 3. Evidence: `history/perf-2026-08-27/item3_stall_attrib.md`.

**Tried.** Nothing was built. The lead said `cp_rasterize_stage3` and
`cp_rasterize_stage3_abuf` both read **1.00 sectors/request** — the far side of
coalesced — and inferred that the fix is wider requests: vector loads, or
structure-of-arrays on the producer. It was sized at up to ~2.0 ms/frame through
exclusivity and costed at a quarter of an iteration.

**Measured.** Source-level attribution over 900 consecutive launches (skip 200)
refutes half of it and recategorises the other half.

- **stage3's 1.00 was an artefact of the profiled window.** 65.3% of its
  launches are **degenerate** — 16 instructions per warp: read the tile counter,
  find `num_tiles == 0`, exit. Those are 65% of launches but 7.3% of the time
  and 4.8% of the load requests. Its **working** launches issue **3.087
  sectors/request**. There is nothing to widen. (A 6-launch sample reproduced
  the 1.00 artefact exactly — the same hazard as `WORKFLOW.md` §4.9, caught by
  the agent on itself.)
- **`_abuf`'s 1.00 is real and is the OPTIMUM.** All 24 global loads sit at
  exactly 1.00, in three groups, none of them a per-lane strided walk:

| share of requests | what |
|---:|---|
| **67.2%** | `LDG.E` of `*huge_counter` — **one uniform 4-byte read per block**, followed by `R2UR` |
| 27.6% | the 84-byte `cp_setup_cache_entry`, read field by field by `threadIdx.x == 0` only |
| 5.2% | `huge_queue[tile_idx]`, an 8-byte pair split into its fields |

**A load issued by one active lane gives one sector per request by
construction, and that is its optimum.** The per-lane traffic in the same kernel
is the *output* side and is already wide — `STG.E.64` at 2.56 sectors/request,
atomics at 2.08.

**Mechanism.** By consumer instruction, `R2UR` waiting on the tile counter is
**51.6%** of long-scoreboard samples, the store address chain 37.3%, the third
6.5% — top three 95.5%. More than half the stall is every warp in the block
waiting on a single uniform 4-byte load at the top of the kernel. That is a
**latency dependency, not bandwidth and not coalescing.** Widening is legal in
exactly one place — padding the setup entry 84 → 96 bytes to permit `LDG.128`,
collapsing 22 instructions to about 6 — and it touches **0% of the measured
stall**. **SoA on the producer would make it worse**: one lane reading one
record would go from 1 request to 21, on different lines.

**Retry if.** The uniform counter read at the top of `_abuf` can be removed or
hoisted rather than widened — that is the 51.6%, and it is a dependency
question, not a layout one. Nothing about sectors per request reopens this.

**Cost.** One profiling run. It also produced the only lead of that item, which
is entry 19.

---

## 19. Not issuing the degenerate stage-3 launches — REFUTED

2026-08-27, item 3, follow-up. Evidence:
`history/perf-2026-08-27/SESSION_HANDOFF.md` §16.1 with
`item3_stall_attrib.md`.

**Tried.** Nothing was built. Entry 18 found that **65.3% of
`cp_rasterize_stage3` launches do no work** — they start 512–2,048 blocks, read
a zero tile counter and exit. That is a launch-count question in the same
currency as PDL and the merges, priced at 0.78–0.81 µs per launch removed.

**Measured — as a cap, deliberately without the mix.** The follow-up window's
stage3:`_abuf` launch mix is 5.62:1 against `PERFORMANCE.md` §5.1's 0.55:1, a
**10.1× disagreement**, because launches 201–1,100 are the first four or five
frames and not steady state. Multiplying a start-up fraction by a steady-state
count would have produced a number with two incompatible parents, so the item
was **capped instead**: the degenerate fraction cannot exceed 1, so even if
*every* stage-3 launch on both variants were degenerate and removable, the item
is 209.1 launches/frame × 0.782 µs = **0.164 ms/frame, CI [0.132, 0.198]** —
1.3% of the frame at the absolute maximum, 0.5% at the conditional estimate of
0.062.

**Mechanism, and the trap in the framing.** **A device-side early exit is
already the current behaviour** — the 16-instructions-per-warp launches *are*
that exit. The only saving left is not *issuing* the launch, which is a host
decision that requires a device number. The host cannot know the tile count
without a readback, and a readback at that point **is** the wait the census
prices at 0.514–6.018 ms/frame: trading a 0.782 µs launch for a sync of that
order loses by a factor of hundreds.

**Retry if.** Stage 3 is fused into stage 2 for another reason and this comes
along free, or a device-side conditional launch becomes available that does not
cost a round trip. Neither is worth building for 0.164 ms.

---

## 20. Merging the per-segment raster launches — the wide merge — REFUTED

2026-08-27, item 4. Evidence: `history/perf-2026-08-27/item4_merge_results.md`,
`item4_f4_device_fork.md`, `item4_merge.md`.

**Tried.** Replace the per-segment loop that launches stage 1, 2 and 3 for each
blended segment with one merged grid per chunk that does the same items.

**Promising because.** It removes launches, and a launch was priced at 0.78 µs.

**Measured.** **Removing 298.8 launches/frame COST 0.410 ms/frame** — an implied
**−1.372 µs per launch removed** against a settled credit of **+0.782 µs**. It
is the first measurement in this project where removing launches made the frame
slower. All supporting gates passed (one stdout hash across 8 runs, PDL links
down 199 as predicted), so the result is interpretable rather than suspect.

**Mechanism — structural, not implementation.** F4 measured the merged form per
*item*, against the loop it replaces, on two independent pairs agreeing to 1%:

| kernel | control ns/item | merged ns/item | ratio |
|---|---:|---:|---:|
| stage1 | 8,113 | 3,428 | 0.42 |
| stage2 | 5,563 | 2,410 | 0.43 |
| stage3 | 17,065 | 5,747 | **0.34** |

The bar was ≤1.25×. **The merged form is three times cheaper per item.**
Register pressure is real (48→64, 56→109, 46→94, occupancy roughly halved) and
does not dominate, because it is amortised over about 5 items. **Union busy
FELL** on both measures while the frame got slower — all kernels −0.9%, the
changed chain −3.2%. The device is not paying for the merge; it is being paid.

**The overlap factor names it: control 2.19×, candidate 1.37×.** The control's
per-segment launches run concurrently across the eight side streams that blended
segments have always used; a merged chunk is one grid on one stream. A merged
launch takes about 2.0× as long as *one* control launch while replacing about
**five that were running concurrently**. Critical path per chunk grows 29.8 µs.

**A self-correction that tightened it.** F4's critical-path bound first used an
assumed batches/frame instead of the counter arm's own launches/frame: frames
per window 137.6 → 415.3, merged chunks per frame 82.3 → 27.26, **bound 2.45 →
0.81 ms/frame**, and the observed 0.410 moves from 17% to **51% of the bound**.
Everything else in F4 is a ratio or a per-unit-work quantity and is unchanged.
The corrected bound is a tighter fit, so the causal account is strengthened.

**Retry if.** A merged form issues its chunks across several streams and so
keeps the concurrency the loop already had — which is most of what the loop was
doing. **Halving the register count cannot help**; per-item efficiency is
already 3× better. What this closes for good is any launch-count forecast for a
merge whose launches are currently concurrent (entry 22).

**Retried on B200 (2026-08-31), still no signal.** The retry clause was built
exactly as written: `diag/wide-merge2` commit `61c8586b9a1` distributes
width-2 merged chunks across the existing side streams (width bounded by the
queue-set aliasing, so 4 concurrent chains against the loop's measured 2.19×),
reuses the original merged device bodies, and passes all static gates with the
flag off byte-identical. On B200 — where the launch price is higher (and the
SM array is in fact *narrower*: 148 against the RTX 5090's 170) — the screened medians were: default band
10.81–11.14 ms, `CUDAVK_NO_ABUF_APPEND=1` alone 10.86, plus
`CUDAVK_WIDE_MERGE2=1` 11.01. No arm left the session noise band, matching
entry 22's arithmetic (launch credit scaled by the kept overlap bounds the
upside near 0.07 ms). The branch is kept for reference; do not re-measure
without a mechanism that changes the bound itself.

---

## 21. Merging the A-buffer count-phase launches — REFUTED

2026-08-27, item 4. Evidence:
`history/perf-2026-08-27/item4_countphase_overlap.md`.

**Tried.** Nothing was built, and that is the point. This was the next merge on
the list, forecast at 554 mergeable launches × 0.782 µs = **0.433 ms/frame**.

**Measured, on the shipping default** (PDL 3, fan-out on), two windows agreeing
to 2.5% and gated first — the counter arm reproduces 1,314.2 launches/frame,
which is `PERFORMANCE.md` §5.1's own figure to the decimal:

| population | overlap factor |
|---|---:|
| `cp_rasterize_stage3_abuf` alone, 53% of the triple's time | **1.89×** |
| `cp_clip_rast_fused_abuf` alone | 1.44× |
| `cp_rasterize_stage2_abuf` alone | 1.22× |
| the three stages, time-weighted | **1.59×** |
| the abuf count-phase triple as a group | 2.50× |
| the opaque/direct triple as a group | **2.68×** |
| all kernels | 1.80× |

The faithful figure is the per-stage **self**-overlap, not the group's, because
a merge of "the same stage across the segments of one episode" concatenates that
population and does not remove the pipelining between different stages. The
dominant stage is at 1.89×, past the 1.8× bar registered in advance.

**Mechanism.** Scaling the credit by the serial fraction 1/1.59 makes the
0.433 ms/frame arithmetic worth about **0.16 ms** — before subtracting the
device-side cost of the merged form, which entry 20 measured *at this width* to
be larger than the credit. The direct/opaque path is worse for a merge, not
better, at 2.68×.

**Retry if.** Nothing here reopens on implementation quality. **This driver's
mergeable launches are the ones the fan-out already made concurrent**, and the
fan-out is worth +2.73 ms/frame. A merge would have to buy back that concurrency
by other means first.

**Cost.** One profiling window, against a build that entry 20 shows would have
cost an iteration and lost.

---

## 22. THE MERGE RULE — the credit is only there where the launches were serial

This is a rule, not a lead, and it is here because it closed two leads in one
session (entries 20 and 21) and would have closed a third before it was built.

> **A merge only collects the launch-removal credit where the launches it merges
> were SERIAL. Where they were concurrent, merging converts parallel work into a
> longer critical path and the credit inverts.**

How to apply it, in order:

1. **Measure the overlap factor of the population you intend to merge, at its
   own position in the shipping driver** — summed kernel time ÷ union of kernel
   intervals. Not of a related population, and not of the group it sits in: use
   the per-stage self-overlap, because a merge concatenates one stage's own
   launches and leaves the pipelining between stages alone.
2. **Scale the launch-count credit by `1/overlap`.** At 1.59× the credit is 63%
   of the arithmetic; at 2.68× it is 37%.
3. **Then subtract the merged form's own device cost**, which is a separate
   measurement and can exceed what is left. In entry 20 the merged kernels were
   **three times cheaper per item** and the frame still got **0.410 ms slower**,
   because per-item efficiency is not the currency — critical path is.
4. **Watch the sign of union busy.** In entry 20 union busy *fell* while the
   frame *rose*. That combination is the signature of lost concurrency and of
   nothing else; if it appears, stop looking for an implementation defect.

The general form: a launch count is a currency only where the launches are on
one critical path. Everywhere else it is a proxy, and the proxy is wrong by the
overlap factor.

---

## 23. Hoisting the shading-group tables above the drain (Tier 1) — REFUTED

2026-08-27, item 2 Tier 1. Evidence:
`history/perf-2026-08-27/item2_tier1_remeasure.md`, with the first measurement
in `tier1_tier2_results.md`.

**Tried.** Reorder the work around the episode drain so that the shading-group
tables are built before the drain rather than after it — pure host-side
reordering, no work removed, no device behaviour changed.

**Promising because.** P0 had just shown that host work moved to just before
this drain is absorbed by the wait (slope 0.065 at D = 125 µs), so a reorder
should have been free real estate.

**Measured, re-run on the default path after its revert arm was fixed.** All
four gates pass, including the one that decides it:

| gate | result |
|---|---|
| both arms complete | 10 of 10 runs rc=0 |
| submit counts | 3,022 old / 2,994 Crossroads, control included |
| candidate hash | session standard, unchanged from before the fix |
| **control hash == candidate hash** | **one hash per capture across both arms** |

So the reorder really is a reorder — no byte of output moves when the hoist is
reverted, which was Tier 1's only correctness argument and now has ten runs
behind it.

**The median is consistent with zero on both captures.** Old: the candidate is
nominally slower by 0.045 ms, but the three pairwise differences disagree in
sign, span 0.168 ms, and the control arm's own spread across repeats is 0.185 ms
— larger than the session spread and four times the claimed effect. Crossroads:
−0.003 ms, a tenth of that capture's spread. The claim was "0.00 to +0.05"; the
measurement cannot distinguish +0.05 from 0.00 from −0.05.

**Mechanism.** There is no mechanism to find: the tables are not what the host
is waiting for. Recorded as what it was declared to be — **a tidiness change
worth about zero, with an unchanged bitstream.** The forced-fallback pair
(+0.0184 ms at 28.85 ms/frame) stays on the record as a different-regime
datapoint, not as the result.

**It cost a bug and paid for its own fix.** The refactor made
`cp_pass_group_table_one()` both the single producer of the group rows and the
allocator of `pass_group_ubos`; the revert path did not re-read that pointer
after the producer ran, so the first multi-member group whose FS reads const
bufs launched a shade with a null table, `rows` collapsed to 1 while `ndraws`
still carried the real row count, and the shader walked off the argument block.
Deterministic device loss at the same capture index on both captures, and
**invisible under `FORCE_PASS_FALLBACK`**, because that path never reaches the
grouped shade loop.

**Retry if.** Never on performance grounds at this size. If it is landed, land
it as tidiness with the hash gate quoted.

---

## 24. The opaque sort-middle tiling prototype — REFUTED, and its recorded reason was wrong

2026-08-27 audit. Evidence: `history/perf-2026-08-27/tiling_ncu.md` and
`stage3_imbalance.md`. The prototype itself is
`notes/OPAQUE_TILING_PROTOTYPE.md`, behind `CUDAVK_TILED_OPAQUE`.

**Tried.** A sort-middle tiled path for the opaque stream: bin references into
screen tiles, then raster each tile once. The note named Nsight Compute on
`cp_opaque_tile_raster` as the required next step; **that step had never been
run.** It was run here, on `multithreading`, the sample the note says amplifies
the failure.

**The note's own explanation is dead in both halves.** It blames a
`references × tile area` coverage loop and register/divergence pressure.
Measured: compute throughput **2.53%**, DRAM **0.23%**, branch efficiency
**94.32%**. What is actually happening is a one-block signature — per-SM
counters, single pass, three consecutive launches agreeing to 0.2%:
`gpc__cycles_elapsed.max` 110.8 M against `sm__cycles_active.max` **110.5 M =
99.75% of elapsed**, average SM 22%, **minimum 0.10%**. One block sets the
kernel's duration, and 73.3% of stall cycles are at the CTA barrier waiting for
thread 0's per-reference `setup_triangle()` and its copy of `cp_rasterize_args`
into shared memory.

**The census kills the successor directions.** Total references barely move with
tile size — 1,190,127 at 16 px, 1,074,095 at 32, 1,027,161 at 64 — which solves
to a mean triangle edge of about **1.7 px** and 1.07 references per triangle.
The hot tile at pixel (704, 352) holds **26,297 distinct triangles** and resolves
563 visible pixels. So there is no large-triangle population to split off, and
the tile lists are depth complexity rather than coverage.

**The model, built on one workload and tested on another.** Duration = longest
tile list × ~3,409 cycles per reference. Cross-check A: 1.07 M refs × 4,213
profiled cycles over 170 SMs = 26.6 M against a measured `sm__cycles_active.avg`
of 24.5 M, 9% out. Cross-check B: applied to the old capture's 10,033 measured
per-episode longest lists it predicts **+12.8 ms/frame** against the recorded
2026-08-18 regression of **+10.38 ms/frame** — a workload it was not fitted to.

**The refusal, with the number.** A v2 that fixed both required directions would
cost 0.15–0.6 ms/frame of tile raster plus about 0.13 ms of binning and could
displace about 2.87 ms/frame of the direct raster chain: **a ceiling of 2.1–2.6
ms/frame of kernel time.** But the tiled arm must give up the opaque stream
fan-out, which `PERFORMANCE.md` §1 banks at **+2.73 ms/frame on old**. **The
lead's whole best case is smaller than the mechanism it has to surrender.**

**Retry if.** The fan-out stops being worth +2.73 ms — a workload with few
segments per episode, where Crossroads-like behaviour dominates — *and* a
per-tile occlusion mechanism exists to shorten a list that is depth complexity.
Keep `CUDAVK_TILED_OPAQUE` off; do not design a v2 without both.

**A process finding of equal value.** `CUDAVK_TILE_CENSUS` has never reported
anything: `cp_tile_census_end_pass()` (`cp_renderer.c:7998`) is **called
nowhere**, and the only other call to `cp_tile_census_reduce_pass()` fires only
when the framebuffer changes size inside one bind — which no sample and neither
capture does. The prototype's own "Implementation, 1. Census" step therefore
never ran, and the reference distribution that kills its stated explanation was
one line of code away from the people who abandoned it. **Before trusting a
census flag, check that its reducer is reachable on the workload you are
running.** The one-line fix used to take these numbers lives only in a throw-away
copy and is not proposed for the tree.

**Related, and it does not reopen anything.** In the shipping driver
`cp_rasterize_stage3_abuf` shows the same `sm__cycles_active.max` signature
(median 93.9% of elapsed against a median average SM of 16.4%) for the opposite
reason. A probe on every 29th launch across a full old-capture replay (6,669
launches) read back the device-built tile queue: median 12 entries, p90 404,
max 4,434; **17.6% of launches have an empty queue**; and the grid is
`CLAMP(rast_num_triangles * 8, 512, 2048)`, sized from the triangle count rather
than from the queue stage 2 builds on the device, so **98.3% of launches have
queue ≤ grid** and the median launch leaves 500 of 512 blocks idle. The longest
block holds *one* item. Compaction and rebalancing buy nothing here; only more
warps per item or more items per launch can — and by the exclusive-fraction rule
the whole question is worth under 0.4 ms/frame of device time. See entry 10.

---

## 25. Device-side episode chaining — CDP2 tails, predicated pre-issue, conditional graphs — REFUTED

2026-08-30, on the HeadlessStreamer occlusion capture's compiled harness
(`~/favorite3-cpp`), tree at 0f6e7436db7. Full record with the census and the
microbenchmark source: `docs/cudavk/notes/DEVICE_EPISODE_CHAINS.md` on the
`cudavk/device-episode-chains` branch (nothing was built, so nothing merged).

**Tried.** Measure-first evaluation of moving the per-episode launch chain off
the host: CDP2 fire-and-forget/tail-launch chaining of the serial blended
tail, pre-issued predicated chains, and conditional graph nodes.

**Promising because.** The frame is ~900 launches of 10–30 µs latency shells;
the host issues every one. `REARCHITECTURE_IDEAS.md` ideas B/F.

**Measured.**

- The unit (nsys census, real-frame window): the merge rule admits only the
  serial blended tail — **50.36 launches/frame in 4.95 chains**; 65.6% of the
  links are already PDL-hidden at a 1.7 µs median gap. Chaining removes 45.4
  host launches/frame = **0.035–0.037 ms/frame** at the settled 0.78–0.81 µs
  price.
- The mechanism (standalone `cdp2-microbench.cu`, sm_120, CUDA 12.8, two runs
  agreeing to 1.5%): host-issued back-to-back links **1.48–1.51 µs**; CDP2
  tail launch **7.32–7.44 µs**; fire-and-forget **8.65–8.73 µs**. A
  device-side launch costs **~5× the host price on this machine**, so
  converting the 45.4 links would *add* ≈0.26 ms/frame of device critical
  path to save ≤0.037 of host issue.

**Mechanism.** The launch-latency floor is the device front end, not the
host's API call: `cuLaunchKernel` costs 1.43 µs of CPU while the device link
is 1.5 µs end to end — the host is not the bottleneck it looks like from an
API trace. Pre-issued predicated chains fail the same arithmetic from the
other side: they issue the same launches earlier plus a repair arm (a
launch-count *increase*), and the wait they would hide is measured flat
(spin-probe slopes +0.02 before / +0.05–0.21 after across two campaigns).
Conditional graphs stay closed under entry 1's retry-if: the prize on this
axis (≤0.037 ms) cannot fund stable-address surgery.

**Retry if.** The serial-tail population grows by an order of magnitude (a
workload with tens of blended episodes per frame), or a CUDA release brings
device-side launch latency to parity with host issue — re-run
`cdp2-microbench.cu` before believing either. **And re-run it on the target
hardware regardless**: these prices are sm_120 (RTX 5090 dev box) facts, and
the driver's primary target is B200 (sm_100) — no number in this entry has
been measured there.

**Cost.** Zero production code. One census, one microbenchmark, three
locked replays.

---

## 26. Emptying the stage3→fs_compact link — interp-in-argblock and compact PDL — REFUTED

2026-08-30, same capture and method, on the `cudavk/raster-chain-merge`
branch (merged at 0bd790b550c). Both mechanisms are **in-tree, default off**,
with their losses recorded in `FLAGS.md`: `CUDAVK_INTERP_INLINE` and
`CUDAVK_COMPACT_PDL`.

**Tried.** The direct shade chain's one remaining exposed stretch was
stage3 → (counter memset) → (interp upload) → fs_compact, all serial on the
main stream. The counter memset became the counter pool and **paid +0.08–0.10
ms** (the accepted half of this work). The other two ops: fold the fused-interp
block into the FS argument-block upload so no stream op is left between
stage3 and compact, and give `cp_fs_compact` a `griddepcontrol.wait` PDL
prologue.

**Promising because.** The measured gap before fs_compact was 5.70 µs median
and every piece of it is device front-end latency (the host is 0.8–3 ms
ahead); pricing put ~0.2–0.35 ms/frame on the pair.

**Measured.** INTERP_INLINE: **+0.13 ms/frame slower**, all four palindromic
pairs agreeing, microcause unattributed. COMPACT_PDL: buys back 0.10 of
INTERP_INLINE's 0.13 but is inert without it; the pair nets **−0.05**. The
counter pool alone took the gap to 0.83 µs — there was less left than the
5.70 µs suggested.

**Mechanism.** Not fully attributed, and recorded as such. The interp block
riding in the argument upload grows the per-shade upload and moves it onto
the timing-sensitive edge the counter pool had just cleaned; whatever the
microcause, the direction is measured in four independent pairs.

**Retry if.** Someone attributes INTERP_INLINE's +0.13 to a removable cause
(the flag makes the experiment one environment variable), or the shade-chain
shape changes so the compact link is exposed again — check with the entry's
own numbers: gap-before-compact median was 0.83 µs *with* the counter pool.

**Cost.** Two flag-gated mechanisms kept as controls, one palindromic
session each.

---

s * 8, 512, 2048)`, sized from the triangle count rather
than from the queue stage 2 builds on the device, so **98.3% of launches have
queue ≤ grid** and the median launch leaves 500 of 512 blocks idle. The longest
block holds *one* item. Compaction and rebalancing buy nothing here; only more
warps per item or more items per launch can — and by the exclusive-fraction rule
the whole question is worth under 0.4 ms/frame of device time. See entry 10.

---

## 27. Scope-level concurrency — overlapping independent render scopes — REFUTED

2026-08-30, HeadlessStreamer occlusion capture, nothing built. The idea: the
segment fan-out one level up — run data-independent render scopes' whole
episode chains concurrently on their own streams, since the frame is
latency shells at 3–6% SM issue and concurrent chains would overlap for
free.

**Measured.** A three-print execute-time census (`exec-scope:`/`sampled:`/
`written:` behind `CUDAVK_DEBUG_RT`+`CUDAVK_DEBUG_TEX`) over the whole
replay, 1,189 steady-state frames: **10.0 scopes/frame, dependency-DAG
critical path 9.0, maximum level width 2.0** — parallelism ratio 1.11×.
With only true read-after-write edges (anti- and output-dependencies
assumed breakable by copies): depth 8.0, ratio 1.25×. About one adjacent
scope pair per frame shares a write target (a single scope-fusion
candidate, ≤0.3 ms class, unpursued).

**Mechanism.** The content is a pipeline, not a fan: each scope samples its
predecessor's output. There is no independent-cascade population — the
shadow work feeds the main pass, the main pass feeds the post chain.

**Retry if.** A workload appears whose scope census (the prints are one
flag away) shows width ≥3 over a meaningful share of frames. The
scheduling machinery should not be built ahead of that number.

**Cost.** Three debug prints, kept in-tree, and one instrumented replay.

---

## 28. Hardware-inline fragment shaders with surviving local memory — REFUTED (wrong frames)

2026-08-30, after dead-scope elimination. Branch
`cudavk/hw-inline-local-probe`, not merged.

**Tried.** Relax only `cp_compile_nir_one()`'s `.local` veto for the
same-LLVM hardware-texture interpolation build. Surviving
`cp_fs_inline_lane` helpers and malformed hardware-texture PTX remained hard
failures. The flag moved the shaders whose only rejection was local memory
from `HW_FUSED` to `HW_INLINE`; about 8–9 fragment launches per real frame
were in the target population.

**Promising because.** The veto was broader than the fused-vertex-fetch
admission: it rejected any `.local`, without comparing against the shader's
classic build. The admitted test shader used 92 registers, 8 bytes local and
2 blocks/SM, against a 203-register fused form tuned down to 126 registers,
176 bytes local and 2 blocks/SM.

**Measured.** Locked alternating compiled replay, two rounds, complete counts
and one stdout hash per capture:

| capture | control real median | candidate | apparent gain |
|---|---:|---:|---:|
| favorite3 | 7.6041 ms | 7.5270 ms | **0.0771 ms** |
| favorite2 | 6.5358 ms | 6.5002 ms | **0.0355 ms** |

Those timing numbers are not an acceptable win: the pixel gate fails.
Candidate/control sentinel dumps differ in **15 of 18 favorite3 frames** and
**14 of 18 favorite2 frames**, beginning around frame 400. The simple
texture-cache tests still produced exact pixels; their three suite failures
were only assertions that the selected mode was `HW_FUSED`. Real shaders are
the negative control the unit tests lacked.

**Mechanism.** The surviving local storage is semantically load-bearing for
real generated shaders. Resource counts cannot distinguish a legitimate local
array/depot from interpolation state that failed scalar replacement, and
admitting it changes results even though no helper symbol survives. The
existing all-or-nothing veto is therefore a correctness boundary, not merely
an occupancy heuristic.

**Retry if.** The local objects are identified individually in LLVM IR and a
specific one is proven equivalent after SROA, with both captures' sentinel
frames as the first gate. Never retry by accepting PTX containing `.local` as
a class.

**Cost.** One flag, two alternating sessions and two 18-frame sentinel pairs;
all implementation changes stay unmerged. Raw data:
`/tmp/hwinline-ab/`, `/tmp/hwinline-dumps/`.

---

## 29. Reusing the device-only scratch high-water — REFUTED (below noise)

2026-08-30. Worktrees `~/mesa-scratchprobe` and `~/mesa-scratch2`, not
merged. Raw alternating sessions: `/tmp/scratchreuse-ab.log` and
`/tmp/scratch2-ab/`.

**Tried.** Stop treating `dscratch.used > 1 GiB` as a reason for
`cp_scratch_begin()` to drain the whole CUDA context. Keep the managed-arena
high-water and both five-overflow safety triggers. Device-only scratch is
rewound at its episode boundary under stream order, so the byte high-water is
stale rather than evidence that live allocations still need a context drain.

**Promising because.** Instrumentation attributed **1.65 ms/frame on
favorite3** and **1.19 ms/frame on favorite2** of main-thread blocking to this
one trigger. The candidate retained complete submit counts and one stdout hash
per capture.

**Measured twice.** Before dead-scope elimination, the locked two-round pooled
result was only +0.038 ms favorite3 and +0.058 ms favorite2. Rebased onto the
landed dead-scope tree (`~/mesa-scratch2`), strict control/candidate order was
control, candidate, candidate, control:

| capture | pooled control | pooled candidate | nominal gain |
|---|---:|---:|---:|
| favorite3 | 7.6059 ms | 7.5689 ms | **0.0370 ms** |
| favorite2 | 6.5518 ms | 6.5398 ms | **0.0120 ms** |

The pooled favorite3 number is not decisive: pair 1 makes the candidate 0.0440
ms slower and pair 2 makes it 0.0920 ms faster. Favorite2 also disagrees in
sign (+0.0493, then -0.0193 ms), and its pooled 0.0120 ms is below the 0.0143
ms control spread. Every favorite3 arm has 6,939 valid timestamps and the
standard hash; every favorite2 arm has 6,965 and its standard hash.

**Mechanism.** The context drain usually waits on useful GPU work. Removing it
lets the host advance only to the next algorithmic episode/segment dependency,
which then absorbs almost all of the wait. Blocked host time is not frame time;
this is the clearest current example, with a 1.2–1.7 ms blocked quantity
converting into at most a few hundredths.

**Retry if.** Only on B200, or after an architectural change removes the next
counter dependency, and only with at least four tightly alternating pairs.
The change is simple and plausibly correct, but it has not earned the sentinel
and 18-sample sweep cost on this machine. Do not quote the removed blocked time
as its performance ceiling.

**Cost.** Two count/timing probes and two rebased alternating sessions. No
source landed.

---

## 30. Episode-entry stale-depth Hi-Z at 8-pixel tiles — REFUTED (≤0.103 ms/frame ceiling)

2026-08-30/31, favorite3 real frames (timestamp index 2782 onward).
Diagnostic branch `cudavk/hiz-census`, commits `fedb1ff1b4a` through
`615274e9b5e`; no production source merged. Raw control/candidate output:
`/tmp/hiz-f3-8/` and `/tmp/hiz-f3-8-cand2.log`.

**Tried.** At each normal opaque episode entry, build conservative tile depth
bounds from the unchanged depth attachment, then replay the shipping
route-specific raster arithmetic without changing visbuf, depth, color or
render state. For every small, medium and huge primitive/tile reference, count
whether the episode-entry bound can reject it before the ordinary raster work.
The final probe used 8x8 tiles. It audited raw, unclamped queue counts from a
private per-segment snapshot: the earlier version saved pointers into eight
reused queue sets, so segment `s + 8` could overwrite segment `s` and make a
false `shipping_match=yes`. The corrected probe copied both counters on the
segment stream before reuse and required every snapshot, reducer equality,
zero queue overflow and successful CUDA completion.

**Valid target-window result.** The compiled replay produced all 6,939 timestamp
events and the standard stdout hash; 18/18 candidate sentinel dumps equal the
control. The driver saw eight additional command-bearing virtual-swapchain
helper submits which bypass the timestamp shim. One occurs before the target,
so external timestamp 2783 is internal command-bearing submit 2784; the first
selected episode is exactly that submit. The other helper command buffers carry
no opaque draws. Thus the 5,957 admitted episodes are the requested real-frame
population even though the two counter spaces end at 6,939 and 6,947.
`failed=0`, `reducer_check=ok`, `shipping_match=yes`, `result_valid=yes`,
36,963/36,963 queue snapshots were present, and both queue-overflow counts were
zero.

| quantity | before | after | reduction |
|---|---:|---:|---:|
| whole valid primitives | 414,649,151 | 406,198,925 | **2.038%** |
| primitive/tile references | 1,458,883,449 | 1,360,565,811 | **6.739%** |
| bounding-box pixels | 36,474,987,587 | 33,261,052,256 | **8.811%** |
| covered pixels | 6,763,694,506 | 5,643,471,136 | **16.562%** |
| sum of each episode's longest tile list | 9,744,153 | 9,646,707 | **1.000%** |
| maximum tile list | 7,818 | 7,818 | **0%** |

The class split does not hide a larger prize: reference reductions are 1.656%
small, 10.282% medium and 6.215% huge; covered-work reductions are 10.250%,
26.286% and 13.788% respectively.

**Ceiling.** In the clean post-dead-scope trace, the only directly avoidable
shipping kernels — ordinary `cp_rasterize_stage2` plus stage 3 — occupy 0.623
ms/frame of union time (0.779 ms/frame summed across streams) over the 412.5
real-frame slice. Charging the largest measured reduction, covered pixels,
linearly against the whole union gives only **0.103 ms/frame**; charging the
reference reduction gives 0.042 ms/frame. Both are optimistic because fixed
launch/setup work remains and the longest-list sum barely moves. The diagnostic
itself built 85,384,768 tile bounds by reading 5,464,625,152 depth words — about
14.3k tiles and 917k words per admitted episode — so a production full-screen
builder already costs more work than this ceiling can repay. Diagnostic timing
is intentionally not used.

**Mechanism.** Episode-entry depth is useful on real frames, unlike the loading
smokes where it rejected nothing, but the useful unit is only a fraction of two
already-small raster stages. It cannot reduce final fragment shading: those
losers already do not survive the visibility buffer. A hierarchy that must be
built or refreshed for each episode has no remaining budget.

**Retry if.** A future renderer already maintains exact conservative tile depth
metadata as a free by-product, or a target-device trace raises the affected
stage-2/3 union by several times. Reuse the exact raster arithmetic and private
queue snapshots; never infer bounds from transformed vertex extrema, and never
accept a capped/reused queue audit as shipping equality. Do not build a
full-screen Hi-Z maintenance mechanism for the present captures.

**Cost.** One hardened count-only census, two loading smokes and one corrected
full target-window sentinel replay. Favorite2 was not run: favorite3's
optimistic affected-work ceiling is already below the cost class, and a change
must win on both captures.

---

## 31. One-visbuf depth-only chain deferral — REFUTED (0.170 ms/frame whole safe pool)

2026-08-31, favorite3 real frames. Diagnostic branch
`diag/depthonly-census`, commits `25a4c4f3190` through `7eebcbf44ff`; no
production source merged. Count/correctness output is under
`/tmp/depthonly-f3/`; the complete bounded traces are the three overlapping
registered-NVTX reports under `/tmp/depthonly-chunks/`.

**Tried.** Classify direct `cp_shade_fragments()` calls which produce depth but
no observable color, multisample coverage, discard/demote, shader memory
side effect, special fragment output, query observation, peel/retry result or
unsupported depth result. Consecutive safe calls are the population a future
one-visbuf-per-scope design could defer and resolve once. Query lifetime is
tracked over the final flattened primary/secondary stream, not just at the draw
where a query begins or ends. Each admitted call has a dedicated
`depthonly-safe submit=... scope_exec=... serial=... run=... chain=...` NVTX
range around the exact direct shade call and no census CUDA work.

**Correct population.** The unprofiled arm has all 6,939 timestamp events, the
standard stdout hash and 18/18 exact sentinels. The flag-off suite is 79/79.
The selected target contains 71,123 direct shade calls, of which 40,184 are
structurally depth-only and 37,406 are exactly safe. They form 2,876 runs:
1,120 singleton and 1,756 multi-chain runs containing 36,286 calls, maximum
length 34. Classification, run sum, run kind/shape and absolute chain/run
sequence checks all pass; final internal command-bearing submit is 6,947.

**Exact clean-trace price.** A whole-process CUDA trace silently stopped before
this late population, so the final trace used registered process ranges in
three overlapping internal-submit windows. Plain NVTX strings are recorded by
Nsight Systems 2026.4.1 but do not trigger `--capture-range=nvtx`; the domain
and message must be registered. Overlaps were deduplicated by the complete
absolute range label, preferring the later report. All 37,406 unique calls are
present and every one contains exactly one correlated kernel of each expected
class, joined host launch -> GPU activity by correlation ID:

| selected kernel | launches | device time/frame |
|---|---:|---:|
| `cp_fs_compact` | 37,406 | 0.092105 ms |
| generated FS `main` | 37,406 | 0.053595 ms |
| `cp_fs_writeback` | 37,406 | 0.024089 ms |
| **whole direct chain** | **112,218** | **0.169790 ms** |

The selected kernels do not overlap, so summed and union time are identical.
Nsight emits a generic “might not have collected” diagnostic when each bounded
capture stops. Completeness here does not rely on that warning being absent: it
comes from the full 37,406-label identity set, exact overlap equality, three
kernels per label and complete correlation joins.

**Mechanism.** The earlier broad direct-shade union was 0.693 ms/frame, but only
0.170 ms/frame belongs to calls the required visibility rewrite may legally
collect. Even deleting every compact, fragment and writeback kernel in every
safe run is below the 0.5 ms build gate, before the one-visbuf resolve, state
retention and final commit cost. Long run counts do not make the affected
kernel time larger.

**Retry if.** A later capture makes the exact safe-chain union exceed 0.5
ms/frame, or another already-required mechanism provides the shared visbuf and
resolve for free. Preserve the lifetime query refusal and exact output/side-
effect gates. Do not price this proposal from all direct shade calls.

**Cost.** One host/NVTX census, a flag-off suite, two full sentinel replays and
three bounded CUDA/NVTX reports. Favorite2 was not run because favorite3's
entire legal pool is only one third of the admission threshold.

---

## 32. Immutable two-slot segment-0 shading overlap — REFUTED (0.036 ms/frame useful chain)

2026-08-31, favorite3 real frames. Diagnostic branch
`diag/two-slot-census`, commits `f2dcee0a156` through `bb04f1dce67`; no
production source merged. Count output is under `/tmp/slot0-weight-f3/`; the
complete clean labels-only trace is the three-report set under
`/tmp/slot0-chunks/`.

**Tried.** Measure the only correct speculative form left after the mutable-
visbuf design was rejected: snapshot immutable segment-0 candidates before the
segment-1 gate, shade them into private provisional storage while later
segments rasterize, then retain only candidates which are still final winners.
The census snapshots raw queue evidence on the owning stream, reduces exact
2x2 quads after the ordinary join, and attributes candidate/useful/killed/final
quads to full stable 256-bit fragment-shader content identities. It does not
use pointers, process-local bins or selected ordinals as shader identity.

**Population proof.** Both census and clean labels arms count every nonempty
physical episode from process start and hash an arm-independent v2 population
record: command-bearing submit, absolute episode ID, dimensions, primitive end,
segment count and each ordered segment's primitive base/slots and full shader
key. Both arms produce 3,759 selected episodes and the identical digest
`4f8b99ac67fdf356553ebca7f74ee920340ebc54eed812801c4985062c66cc82`.
The labels arm records 34,765 expected, attempted and successful keyed generated
FS launches, with zero setup, ID, overflow, active-scope or per-scope mismatch.
The full replay has 6,939 timestamp events, the standard hash and 18/18 exact
sentinels; the final branch passes 79/79 tests.

The count-only result is:

| quad quantity | count |
|---|---:|
| segment-0 candidates | 43,826,342 |
| still-useful candidates | 25,563,331 (58.33% of candidates) |
| overwritten/killed candidates | 18,263,011 (41.67%) |
| ordinary final quads | 584,640,942 |
| useful share of ordinary final work | **4.37%** |

**Full-key timing, without scaling a partial trace.** Whole-process CUPTI
collection reached its event/buffer boundary before the selected population.
Three overlapping registered-NVTX submit windows cover it instead. Inner
`fs/<64-hex-key>` ranges were joined to GPU `main` by correlation ID, nested in
outer `slot0-selected/episode=<absolute>/ordinal=<selected>` ranges, and
deduplicated by `(absolute episode ID, inner launch ordinal)`, preferring the
later overlap. The result has exactly 3,759 unique outer episodes and 34,765
unique keyed launches; every inner range has exactly one generated `main`, and
overlap keys, grids and kernel sequences agree exactly.

The current serial selected chains cost 0.108640 ms/frame compact, 0.415560
ms/frame generated FS and 0.032081 ms/frame writeback: **0.556281 ms/frame**
total. Weighting each full-key chain by that shader's exact useful/final quad
ratio gives only **0.036406 ms/frame** of useful work which correct speculation
could hide. Candidate speculation would execute 0.280788 ms/frame at the same
optimistic linear price, of which **0.244381 ms/frame is killed**. Generated FS
alone contributes 0.021904 ms/frame useful. These projections already favor
the proposal: they ignore per-draw fixed cost, private provisional storage,
extra launches and the one final attachment commit.

**Mechanism.** Aggregate candidate survival looked substantial, but it is
concentrated in a small share of ordinary shaded work. Full shader weighting
raises the earlier average-cost estimate, but only to 0.036 ms/frame, while
wrong speculation costs about seven times as much. The immutable architecture
is correct in principle; its real population is not economically useful.

**Retry if.** A future capture changes the full-key useful-chain price by more
than an order of magnitude, or provisional shading becomes a free by-product
of another renderer. Preserve absolute episode IDs, structural population
digests and actual attempted/successful launch accounting. Never scale a
bounded early trace against full census counts, and never compare NVTX host
timestamps directly with GPU timestamps.

**Cost.** One exact count census, two independent reviews, a 79-test suite, a
full sentinel replay, a clean full-key labels arm and three bounded CUDA/NVTX
reports. Favorite2 was not run because favorite3 is two orders of magnitude
below the build gate after mandatory wasted speculation.

---


## 33. Cross-frame retained raster-output recurrence — REFUTED (0.206 ms/frame collectible upper)

2026-08-31, favorite3 real frames. Diagnostic branch
`diag/geometry-reuse-census`, commits `ffb03c80361` through `ade83f5b5ef`; no
production source merged. Full-run output is under `/tmp/geometry-b1s-f3/`, and
the clean cost join is `/tmp/geometry-b1s-f3/cost.json` against
`/tmp/postdead/trace.sqlite`.

**Tried.** Establish the largest sound population for a resource-versioned
cross-frame cache of transformed/rasterized geometry before designing the
cache. B0 hashed each safe post-stage2 visibility output with an exact,
domain-separated SHA-256 Merkle tree. B1R admitted retry units and hashed two
fixed domains back-to-back on the same snapshot, separating collision/domain
behavior from replay scheduling variation. B1S directly covered ordinary
opaque appending segments: after their original stage 3 consumed the live queue
set, it rerasterized that one segment on the same stream into an initialized
private visibility buffer and hashed the isolated small/medium/huge atomicMin
contribution. Nine disjoint visibility/SHA banks cover the main and eight side
streams; reuse beyond eight is same-stream ordered. A whole-episode final hash
was deliberately rejected because one changed segment cannot prove every other
segment changed, and partial segment reuse was in scope.

The digest is a generous retained-output necessary condition, not a realizable
input key. Full reusable raster payload equality implies equality of the hashed
projection; digest equality may remain a false match. Raw setup structs were
never hashed: `cp_setup_cache_entry` contains padding and inactive variant bytes,
so doing so would read indeterminate data. Clip remains 100% recurrent because
B1S rerasterizes completed clipped geometry rather than refetching or clipping.

**Correct population and correctness.** Both mode-4 runs completed 6,939
external and 6,947 internal command-bearing submits, kept the standard stdout
hash, and matched all 18 sentinel frames byte for byte. The flag-off suite is
79/79. Each run published all **108,086** candidates: 108,086 admitted, zero
appending/retry/other exclusions, `classification_ok=yes`, and
`result_valid=yes`. Both SHA domains produced identical classifications within
each snapshot. Across 2,079 frame transitions, 107,995 records were compared;
the runs found 42,391 and 42,448 membership matches. Absolute structural
population was identical and 205 classifications flipped between executions,
so the decision arm conservatively treats a unit recurrent if either run saw a
match. Enabled timing is invalid: the diagnostic reraster/hash storage alone
reached 2.325 GB.

**Exact clean-trace price.** The repeated-run OR arm maps all 13,937 selected
clean-trace units: 4,438 recurrent matches, 9,464 observed nonmatches, 35 missing
values charged recurrent, and zero excluded/unadmitted units. All 3,553 clip
intervals stay recurrent. The generous retained-output figures are:

| accounting | ms/frame |
|---|---:|
| summed target duration | **0.356183** |
| target union | **0.318789** |
| project union-exclusive collectible duration | **0.206136** |

The individual union-exclusive diagnostics are 0.206113 and 0.206075 ms/frame.
The decision value is less than half the 0.500 ms/frame admission threshold.
Even the broader summed target-duration bound is below the threshold.

**Mechanism.** A production design would still need canonical semantic setup
serialization, resource versions, stable lookup, retained device storage,
invalidation and a post-key collection point. The complete generous population
cannot fund that machinery. No geometry cache or serialization implementation
is justified for these captures.

**Retry if.** A later capture raises the exact repeated-run retained-output
collectible pool above 0.5 ms/frame on a clean trace. Preserve absolute
submit/unit identities, simultaneous domains, repeated-run OR classification,
private per-segment evidence for appending work, and conservative clip/unknown
charging. Never infer per-segment nonrecurrence from a whole-episode mismatch or
hash uninitialized setup tails.

**Cost.** One GPU SHA oracle, B0 and retry-inclusive B1R runs, two full B1S
replays, a 79-test flag-off suite, exact hashes and 18/18 sentinels per B1S run.
Favorite2 was not run because favorite3's complete generous pool is already far
below the build gate; a production change must win on both captures, but no
production change exists here.

---

## 34. Fully threaded Mesa runtime submit — REFUTED (0.126 ms/frame completion gain)

2026-08-31, favorite3 real frames. Reproducible diagnostic implementation and
analyzer: side branch `diag/threaded-submit-census`, commit `d16ec485935e`.
Measurement data: `/tmp/threaded-submit-f3-census`.

**Proposed.** Force Mesa's assisted-timeline runtime into fully threaded submit
with `MESA_VK_ENABLE_SUBMIT_THREAD=1`. The application thread would enqueue
recorded work while the existing single renderer owner translated the previous
submit and blocked in its ordinary device waits. This did not claim to remove
CUDA work. It was the only remaining host-side candidate with an honest
0.500-ms/frame admission possibility: the post-dead-scope run had 1.673
ms/frame of GPU idle and 1.437 ms/frame in timed renderer waits.

**Why pre-submit timing is invalid.** In fully threaded mode `vkQueueSubmit`
returns after enqueue. `submit_shim` therefore timestamps queue admission, not
CUDA completion, and can make work that is merely running ahead look free. The
diagnostic recorded `CLOCK_MONOTONIC_RAW` at driver-callback entry, after the
pending CUDA completion record was linked, and after `cuEventSynchronize`
returned. The decision metric is the interval from the first to last target
**completion**, divided by 2,078 frame transitions. Enqueue timing is retained
only as a cross-check.

**Correctness work needed even for the experiment.** Current Mesa/cudavk
lifecycle was not safe to force fully threaded across a complete replay. The
diagnostic branch drains runtime work in QueueWaitIdle and DeviceWaitIdle,
finishes Mesa's queue before stopping cudavk's completion worker, broadcasts
worker failures, bounds WAIT_PENDING polling so cross-queue loss cannot strand
teardown, retains live marker/resource storage on terminal loss, propagates
CUDA completion failures to Mesa device loss, and moves image metadata mutation
after its conditional wait. A delayed wait-before-signal DeviceWaitIdle case
covers the lost-marker hang. These changes are diagnostic branch work only;
the measured result below does not justify merging them.

**Population and target.** Each arm had 6,939 external shim submits, 6,947
command-bearing records and 1,663 signal-only runtime markers, for 8,610
complete census records with no overflow. External target submit 2,782 maps to
absolute command-bearing position 2,783 because one virtual-swapchain helper
precedes it. Runtime markers are interspersed, so that position mapped to
absolute record ordinal 3,114 in all four measured arms; it is not valid to use
2,783 as a record ordinal or infer frame parity from it. The selected range has
4,164 command-bearing records and 2,079 frames.

**Correctness gates.** Every arm ended with favorite3's benign post-output
`rc=139`, exactly 6,939 complete shim timestamps, the standard stdout SHA-256
`e3f24a1dcdc8568be217d249e480623958e2621b3d4f056ae4c44ad20d08da49`,
and `result_valid=1`. Before measurement the existing suite passed 79/79 with
`MESA_VK_ENABLE_SUBMIT_THREAD` absent, `=0` and `=1`; the focused timeline test
passed all three direct arms; static audits and all 11 analyzer tests passed.
Fault-injection expectations distinguish the same terminal error reported
synchronously by `vkQueueSubmit` from its fully threaded report by
`vkDeviceWaitIdle`.

**Alternating completion measurement.** All figures are diagnostic because the
census itself records clocks and fixed records. Both arms carry the same census.

| arm | Mesa thread | completion ms/frame | issue-begin ms/frame | enqueue ms/frame | target depth p95/max |
|---|---:|---:|---:|---:|---:|
| c1 | 0 | 8.549611 | 8.552320 | 8.539533 | 0 / 0 |
| t1 | 1 | 8.264747 | 8.267166 | 8.256351 | 1 / 2 |
| c2 | 0 | 8.301126 | 8.303556 | 8.290263 | 0 / 0 |
| t2 | 1 | 8.333623 | 8.336444 | 8.326683 | 1 / 2 |
| median | — | **8.425368 → 8.299185** | **8.427938 → 8.301805** | **8.414898 → 8.291517** | — |

The completion-aware gain is only **0.126184 ms/frame**. Issue-begin and enqueue
deltas agree at 0.126132 and 0.123381 ms/frame. Threaded target depth was p95 1
and maximum 2, so this is not an unbounded-run-ahead artefact. It misses the
0.500-ms/frame admission threshold by 0.373816 ms/frame.

**Decision.** Stop. Favorite2, bounded-depth sweeps, full-frame shared dump
comparison and a production merge are not justified because the primary
capture already fails the build gate. Do not time this mode with pre-submit
shim medians and do not repeat the lifecycle implementation without first
showing at least 0.500 ms/frame in completion-aware favorite3 data.

**Retry if.** A later workload has materially more application recording or
decode work available to overlap, or a renderer wait population whose measured
completion-throughput gain exceeds 0.500 ms/frame. Preserve explicit
external-to-command calibration, filter signal-only markers without treating
the filtered ordinal as an absolute record ID, alternate arms in one session,
and require bounded depth plus both-capture correctness before shipping.

**Cost.** One lifecycle/census implementation, independent concurrency review,
three 79-test suite arms, three focused timeline arms, 11 analyzer tests, one
failed analysis invocation that exposed marker-vs-command ordinal ambiguity,
and four valid full favorite3 census arms. No favorite2 replay was spent after
favorite3 failed admission.

---

## 35. Exact within-batch post-transform vertex reuse — REFUTED (0.438 ms/frame generous ceiling)

2026-08-31, favorite3 real frames. Diagnostic implementation and analyzers:
side branch `diag/vertex-reuse-census`, commits `5445711f243` and
`213d7cdc655`. Data: `/tmp/vertex-reuse-f3` and
`/tmp/vertex-reuse-cost.json`.

**Proposed.** The renderer executes standalone vertex fetch and the generated
vertex shader, or their fused form, once per assembled triangle corner.
Indexed meshes repeat vertices. Reuse the post-transform result for an exact
within-batch `(draw row, effective vertex_id, instance_id)` key, then rewrite
primitive references to the unique output. This is not entry 33's cross-frame
retained geometry: it removes repeated vertex work inside one current batch.
The initial clean-trace pool appeared large enough to census — generated VS
plus standalone fetch was about 0.775 ms/frame union-exclusive in the broader
412.5-frame report.

**Exact diagnostic.** `CUDAVK_VERTEX_REUSE_CENSUS` loads a separate CUDA
module only when its value boolean is true. After the real VS succeeds, on the
same stream, its kernel repeats the production `cp_vf_lane_ids()` identity
calculation but never reads or changes shader output. Exact open addressing
stores all three original integers; the hash chooses only the first probe.
The main and eight segment streams have disjoint generation-tagged banks. Old
grown allocations survive until cleanup, BUSY waits and probes are bounded,
and generation zero is reserved. Results stay on device until the existing
context-wide cleanup join; there is no per-batch host read or synchronization.
Allocation, launch, lane, probe or table failures charge the whole batch
unique/nonduplicate. Record overflow invalidates the run.

Identity is absolute command-bearing submit plus absolute executed-batch
ordinal. Slots, banks and pointers are never identity. Records also retain the
fused/standalone, fetch-ran, indexed and shader-memory-write classes. The
disabled path does not compile or load the module, allocate, assign identities,
select banks, scan, launch, report or add a join.

**Correct population and correctness.** Two full compiled favorite3 replays
both ended with the capture's benign post-output `rc=139`, exactly 6,939
complete external shim rows and final internal submit 6,947. Both kept stdout
SHA-256
`e3f24a1dcdc8568be217d249e480623958e2621b3d4f056ae4c44ad20d08da49`
and matched all 18 sentinel frames byte for byte. The flag-off suite is 79/79.
A controlled repeated-index GPU oracle classified 21,573 duplicates out of
21,600 inputs with the expected 27 row-local unique keys.

Every one of **210,944** executed batches was admitted in both full runs, with
zero record/table/launch/lane/probe exclusions. All structural fields and
unique counts repeated exactly; only the non-semantic atomic probe count
varied. The whole process had 3,545,618,907 inputs and 2,160,968,089 duplicates
(60.948%). From real internal submit 2,784 onward:

| population | total | duplicates | fraction |
|---|---:|---:|---:|
| all | 3,479,473,815 | 2,116,760,651 | **60.836%** |
| fused VS/fetch | 3,426,230,853 | 2,076,471,075 | **60.605%** |
| standalone fetch + VS | 53,242,962 | 40,289,576 | **75.671%** |
| indexed | 3,117,299,295 | 2,106,766,521 | **67.583%** |
| nonindexed / expanded topology | 362,174,520 | 9,994,130 | **2.759%** |

Count ratios are not the decision metric because shader and fetch costs vary by
batch.

**Exact clean-trace price.** The calibrated join uses the established 411-frame
window in `/tmp/postdead/trace.sqlite`, internal submits `[2786,3608)`. That
trace has no NVTX table. Its fail-closed structural classifier accepts a
`main` only when its immediate same-stream kernel successor is
`cp_clip_triangles`, `cp_clip_rast_fused` or
`cp_clip_rast_fused_abuf`; immediate same-stream `cp_vertex_fetch` identifies
the standalone form. Runtime launch APIs join to GPU intervals only through
`correlationId`. It maps all **19,929/19,929** VS records and all
**3,087/3,087** fetches. Every absolute identity, grid, bank-to-stream mapping
and fetch shape agrees. The two census runs agree over the selected population.
Exact integer and fractional calibration fingerprint:
`6106732e85413b2110fd269fcd5f1d691bcf443361da2d7c1b039cfb0476c187`.

| accounting | ms/frame |
|---|---:|
| unweighted VS+fetch summed duration | 1.181784 |
| unweighted pool union | 0.821026 |
| unweighted pool union-exclusive | 0.725432 |
| duplicate-fraction weighted summed duration | 0.582517 |
| weighted exclusive, maximum active fraction | 0.405892 |
| weighted exclusive, active-fraction average | 0.384042 |
| **generous weighted union-exclusive upper** | **0.437503** |

The decision arm is deliberately more generous than max or average weighting:
for each pool-exclusive half-open span it credits
`min(1, sum(active duplicate fractions))`. Even that gives only
**0.437502653 ms/frame**, below the 0.500-ms/frame admission threshold. The
broader 0.582517 summed duration double-counts concurrent work and is not
collectible frame time.

**Mechanism and decision.** Stop. The generous ceiling is already short by
0.062497 ms/frame before building a topology hash/remap, rewriting primitive
references, scattering unique output, retaining storage, tracking exact buffer
identity/content versions, or excluding invocation-sensitive semantics.
`writes_memory` is zero in the selected population, but the driver does not yet
retain complete subgroup/clock/query/transform-feedback eligibility metadata;
those unknowns can only lower a realizable pool. No favorite2 replay or
production cache is justified after favorite3 fails the build gate.

**Retry if.** A later capture or a B200 clean trace raises the exact
per-batch-weighted union-exclusive pool above 0.5 ms/frame *after* a measured
remap/build charge. Preserve full row/vertex/instance comparison, independent
stream banks, absolute submit/batch identity, repeated exact runs, strict shim
parsing, same-stream structural correlation, conservative unknown charging and
the union-exclusive decision metric. Do not promote the global duplicate ratio
or the summed-duration bound to a performance claim.

**Cost.** One diagnostic module/census, CPU hash oracle, no-device actual-source
NVRTC compile, 15 analyzer tests, eight cost-join tests, one 79-test flag-off
suite, three focused GPU smokes including the repeated-index oracle, and two
full favorite3 replays. Favorite2 was not spent after the exact primary-capture
ceiling failed admission.

---

## The rules these produced

Each is tied to the evidence that produced it. They are ordered by how often they
would have saved an iteration.

1. **Measure the unit before building the mechanism.** Iterations 12, 19, 20 and
   23 were rejected by censuses and microbenchmarks with no production code at
   all — four iterations for the price of instrumentation. Two of them (19, 20)
   found the mechanism was cheap and the *unit* was wrong, which no amount of
   implementation quality would have fixed.
2. **A neutral narrow path is a coverage result, not a mechanism result.**
   Iterations 6, 8 and 15 each measured the hardware texture operation working
   (whole chain 33.348 → 17.648 µs) and each lost the frame at 13.6–14.1%
   coverage against an 18.7–28.8% break-even. Iteration 24 kept the mechanism,
   fixed the coverage to 99.9%, and it is worth 5.8 ms today. Write the
   break-even down before rejecting.
3. **Dynamic per-item scheduling costs more than the boundary it removes.**
   Iteration 21 paid +140.15 ms of scheduler against a ~123.7 ms gap ceiling;
   iteration 9 paid +0.850 ms to remove a stack the fast branch never touched.
   Static per-episode chaining is a different shape and paid (+0.2208 ms,
   iteration 28 item 4).
4. **Know the price of what you are removing.** Removing a small copy outright is
   1.30 µs; merging copies at a boundary that stays is 0.59 µs; removing a small
   clear is 0.66 µs on old and 0.75–1.01 µs on Crossroads; a bare same-stream
   launch is **0.6–1.0 µs**, not the 1.4–2.2 µs earlier designs assumed. A fusion
   that also deletes a pass or a large clear is worth more — 1.86 µs per device
   operation in iteration 28 item 4, because the clears carried 49.9 MB/frame.
   Forecasting with the launch price alone is wrong in both directions.
5. **Fold an operation for its count only when its bytes are negligible.**
   Iteration 26 S2b hit its −599.7 operations/frame target exactly and cost
   1.8412 ms, because that clear moves ~90 MB a frame and `cuMemsetD8Async` does
   it at DRAM speed.
6. **Idle blocks are free; do not pay to remove them.** 306,686 fewer scheduled
   blocks a frame moved nothing — under 0.2 ns for an idle 256-thread block.
   Occupancy and wave counts describe a kernel; they do not price it.
7. **Bound-based sizing is priced by the ratio bound/actual, not by the wait it
   replaces.** The iteration-29 S0 probe asked for 8.6 GB against an 8.59 GB cap
   on old and lost 0.38 ms on Crossroads while removing 8.26 host waits a frame,
   because the scratch arena doubles and is freed at every flush. The episode
   drain pays for itself: 0.877 ms buys the exact count.
8. **A focused resource win is not a capture resource win.** Iteration 5 took a
   test shader 190 → 18 registers and left 64 of 65 capture shaders at 203/236,
   because the shared sampler graph fixed the allocation. Measure on the shaders
   the capture actually launches, weighted by launches — iteration 27 had to add
   a per-shader launch census before any performance claim, and admission by
   shader (0.8125) and by launch (0.9538) are different numbers.
9. **A null helper call is not a revert control.** Iteration 4 passed a null
   runtime helper and still carried 108 registers. A revert must select a
   differently built binary, which is how iterations 14, 24, 27 and the parked
   item 3 all do it.
10. **Instruction and code footprint beat occupancy in the sampler.** Iteration
    18's exact inline sampler regressed 5.0% with correct coverage and lower
    registers; iteration 22's noinline form reached 143–182 registers and a
    225 KiB `main` SASS and regressed every arm. Retry requires executing fewer
    sampling instructions, not another helper boundary.
11. **Check the invariant in code.** The parked writeback fusion's central
    assumption was disproved 6,343,098 times in one capture by a check that took
    one run to write. Assumptions about "once per pixel", "at most one claimant"
    and "the count fits" are cheap to test and expensive to believe.
12. **Guard the measurement itself.** A frame is two submits; both arms belong in
    one session; any probe whose stdout hash changes is invalid until the submit
    and frame counts are checked, because a run that dies early produces a fast
    and meaningless median.
13. **A merge only collects the launch-removal credit where the launches it
    merges were serial.** Entry 22, and entries 20 and 21 are what it cost to
    learn. Removing 298.8 launches a frame made the frame 0.410 ms slower while
    the merged kernels were three times cheaper per item.
14. **Attribute a wait site by caller before ranking it.** Entry 17: the driver
    owns 55.5% of the destruction-drain site by call count and 3.0% of it by
    blocked time — an 18× disagreement in the direction that matters, because
    the driver's own drains arrive at an already-empty device. A census that
    counts calls can agree with the API trace to the unit and still mislead by
    two orders of magnitude on cost.
15. **A launch has two prices.** 0.78–0.81 µs to remove a real one, 1.974 µs to
    add an exposed one, because the front end is hidden behind the previous
    kernel on 74.9% of real chain launches (`PERFORMANCE.md` §4). Quote a
    removal lead as a range. The wrong one of these two numbers would have put
    the launch axis at the top of the 2026-08-27 ranking.
16. **A report file is not a trace-completeness proof.** The favorite3
    whole-process CUPTI reports stopped CUDA activity before the late selected
    population while the replay still completed and the report looked valid.
    Nsight Systems 2026.4.1 records plain NVTX strings but requires a registered
    domain/message to trigger `--capture-range=nvtx`. Bound large captures,
    overlap them, deduplicate by absolute work identity and require every
    expected host launch to have its correlation-ID GPU activity. A generic
    collection warning is neither proof of loss in that selected set nor proof
    that the set is complete.

---

## 36. Renderer 2 — scope-level tile-binned shading — REFUTED at design

2026-08-31. Branch `redesign/tilewalk` (kept), documents
`notes/RENDERER2_{DESIGN,STATE_SURVEY,M0_RESULTS,M1_DESIGN,M1_FANOUT,M1_RUNS,M1_WALKBENCH,VERDICT}.md`.
This is the renderer-level rewrite the iteration campaign was exhausted
against: bin a render scope's post-clip triangles into 16 px tiles once, then
walk each tile front-to-back with per-tile occlusion, replacing the per-batch
raster/compact/writeback chain.

**Reached M1 and stopped there, on arithmetic, before writing a kernel.**

**The walk itself passes, decisively.** A standalone benchmark
(`tests/cp_tilewalk_bench.cu`, one thread per reference, shared `uint64[256]`
visbuf, depth-ceiling early-out) measures **26.5 cycles per reference**
conservative and **7.8** with the early-out, against entry 24's **3,409** —
129x and 434x. Inverting entry 24's one-block-per-reference barrier design is
the fix, and the early-out is bit-identical over 921,600 resolved pixels.
**Keep this result**; it is the only published cost for this shape on sm_120.

**Three terms kill it anyway.** Displaced pool 1.4623 ms/frame (RTX heavy
band, per-kernel union-exclusive), less walk 0.092–0.255, less bin pass ~0.13,
less the **measured** fan-out surrender 0.66–0.84
(`CUDAVK_NO_OPAQUE_STREAMS=1`, three-round alternating A/B, both captures):

| scoping | best | worst | midpoint |
|---|---:|---:|---:|
| full design | 0.580 | 0.237 | **0.409** |
| opaque-only first slice | 0.267 | −0.076 | **0.096** |
| walk entirely free | 0.672 | 0.492 | 0.582 |

The gate is 0.500. **A free walk still only reaches 0.582** — the walk was
never the binding term; the surrender and the undisplaced pools are.

**And the upper bound is above the goal.** favorite3 is 6.676 with a 1.676
gap to 5.0. Best conceivable net (0.710) lands at **5.966**; removing the
*entire* displaced pool at zero cost and surrendering nothing — impossible —
lands at **5.214**. Tile binning does not contain the goal, because the pools
it displaces are not where the time is. The time is generated VS 0.689,
generated FS 0.678, `cp_clip_rast_fused` 0.592 and memcpy 0.490: vertex and
fragment arithmetic plus copies, which a different scheduler does not shrink.

**It also reproduces entry 24's signature.** On a realistic scope grid the
hottest tile's block is 346,128 of ~362,000 kernel cycles — **one block is 94%
of the grid's duration**, average SM busy 13.9%. So a tiled walk does *not*
hand back the concurrency it takes from the opaque fan-out; splitting hot
tiles across blocks with a merge is the known fix and was deliberately not
built, because entry 24's retry clause requires the ceiling to exceed the
surrender first, and it does not.

**Retry if.** The undisplaced pools shrink (a workload whose cost is raster
rather than shading), *or* the fan-out stops being worth 0.66–0.84, *and* hot
tiles are split across blocks. Not before all three.

**Two by-products worth keeping.** The per-scope tile census, run
segmentation and cut counters are now real — all three were structurally dead
(`cp_tile_census_end_pass` and `cp_tile_census_cut` had no callers, so every
number the old census printed was zero). And `cp_tri_setup` at 80 B straddles
32 B sectors; padding it to 96 B removes about a third of the walk's read
traffic, unrelated to this verdict.

---

## 37. Host-visible memory residency advice — REFUTED, and its premise was a profiler artefact

2026-09-01. Probe flag `CUDAVK_HOSTMEM_ADVICE` (reverted, not in the tree).

**The lead.** A post-pin B200 trace showed about 1.9 MB/frame of unified-memory
migration through five host-visible regions — roughly 940 KB each way, 4 KB
pages, `migrationCause` split PREFETCH 2.75 GB / COHERENCE 463 MB — costing
0.242 ms/frame, where the same replay on RTX 5090 showed **none**. The obvious
reading was that the B200's UVM heuristics were speculatively bouncing pages
the RTX left alone, and that advising residency would recover it.

**The measurement.** A flag applied `SET_ACCESSED_BY` (mode 1) and
additionally `SET_PREFERRED_LOCATION` host (mode 2) to every host-visible
managed allocation. Three-round alternating A/B, full submit populations,
standard hashes, sentinels clean:

| arm | favorite3 | delta |
|---|---:|---:|
| RTX control / advice=2 | 6.6413 / 8.6166 | **+1.98** |
| B200 control / advice=2 | 10.9151 / 11.2724 | **+0.36** |

It is slower on both, so the mechanism is refuted on its own terms.

**CORRECTION, 2026-09-02: the premise-collapse below is itself wrong.** The
"zero UVM rows under 2026.4.1" was a **dropped CUPTI buffer**, not an absence.
Every harness run exits by SIGSEGV during teardown, and the final activity
buffer is never flushed. Re-run with `--duration` ending *before* the crash,
same 2026.4.1 build, same RTX: **99,999 UVM rows, 1,338 MB host-to-device and
1,180 MB device-to-host.** The figure 99,999 is a record cap and appears again
in the B200 2026.1.3 trace, whose four cause buckets sum to exactly 99,999 —
so both traces were truncated and every migration volume quoted here is a
**floor**, not a measurement.

**Unified-memory migration is therefore real on both GPUs**, and the paragraph
below is retained only to show how the wrong conclusion was reached. What does
*not* change: the mechanism stays refuted, because it was refuted by frame-time
A/B (+1.98 ms/frame RTX, +0.36 B200), which no profiler touches.

The flaw was in the positive control. WORKFLOW 4.05 required one, and one was
run — a deliberate ping-pong probe that showed migration under 2026.4.1. But
**that probe exits cleanly and the harness does not**, so it never exercised
the path that loses the data. A positive control must reproduce the measured
workload's *exit behaviour*, not just its activity class.

**Root cause found and fixed, 2026-09-02.** The harness's SIGSEGV was a
use-after-free in `favorite3.cpp`: it called `vkDeviceWaitIdle` on a device
`src/frame_0000_2908.cpp` had already destroyed, under a comment asserting the
capture never destroys it. With that call removed the harness exits 0, and a
full-length trace with no `--duration` now records **169,040 UVM rows and
3.9 GB of migration** — so the earlier 99,999 figures were partial flushes,
not a record cap, and the true migration volume is larger than either. The
harness fix also restored 23 lines of stdout and 8 shim timestamps that every
previous run had discarded (WORKFLOW 3.1).

**Then the premise collapsed (WRONG — see the correction above).** The two hosts had been traced with different
Nsight Systems builds — B200 2026.1.3, RTX 2026.4.1. Installing the *identical*
2026.4.1 package on the B200 (md5-verified) and re-tracing the same replay with
the same flags reports **zero UVM rows**, exactly like the RTX. A deliberate
managed-memory ping-pong compiled and traced on the B200 under 2026.4.1 does
report migration (640 MB / 624 MB), so the newer tool is not blind to it there.
**The migration this lead was built on is reported only by nsys 2026.1.3.**

**Two rules come out of it.**

1. **Align profiler versions before comparing hosts, and not only for
   host-side time.** The existing warning covers CUDA API timing; this one is
   worse, because a version difference *manufactured a whole activity class*
   that does not exist. Any cross-host claim needs one profiler build, and a
   positive control that the build can see the thing being claimed.
2. **The B200 needs at least three alternating rounds of one pair.** Its
   session-to-session drift is 0.3-1.3 ms/frame — the identical mode-2
   configuration measured 9.80-10.02 in one session and 11.14-11.28 in the
   next. An early two-run probe inside that band read as -0.65 and was
   reported as promising; the controlled A/B reversed the sign. RTX drift is
   about 0.06, so B200 evidence needs the stricter design.

**Retry if.** Never in this form. If a future trace shows UVM migration during
a replay, reproduce it under two profiler builds with a positive control
before designing anything against it.

---

## 38. More pass side streams than 8 — MEASURED, and declined on cost

2026-09-01. Both captures, RTX 5090, three alternating rounds per arm, every
run with its standard stdout hash, full timestamp population and 18/18
sentinels. Experiment branch `exp/streams16` (deleted; the change is one line).

**Tried.** `CP_PASS_STREAMS` has been a hard-coded 8 since the fan-out was
built, with no flag and no record of it ever being tuned. It was raised to 16
and 32 and measured.

**Promising because.** The fan-out is load-bearing — surrendering it costs
0.66-0.84 ms/frame (entry 36) — and the ncu counters taken the same day show
these kernels occupy **3-25% occupancy at 0.5-12% of peak SM throughput** on
both sm_120 and sm_100. The device plainly has room for more concurrent lanes.

**Measured**, favorite2 (the capture that showed signal), whole window and
heavy band:

| lanes | whole | heavy | vs 8 |
|---|---:|---:|---|
| 8 | 5.5021 | 6.0552 | — |
| 16 | 5.4178 | 5.9319 | **-0.084 / -0.123**, arms disjoint |
| 32 | 5.4104 | 5.9132 | -0.092 / -0.142, arms disjoint |

Reproduced in a second independent session (-0.062 whole). On favorite3 the
same change measured -0.029 with **overlapping arms**, so neutral rather than
positive. The knee is at 16: doubling again buys 0.007 ms more.

**The cost, and why it was declined.** Every lane owns its own rasterizer
queue set — `nontrivial` 4 MB + `huge_tiles` 16 MB + `counts` + an 86 KB setup
cache = **19.2 MB per lane**. 8 lanes cost 153 MB; 16 cost 306; 32 cost 613.
So the price of -0.084 ms/frame on one capture is **+153 MB of device memory**,
and -0.092 costs +460 MB.

That gain is below this project's 0.500 ms/frame admission gate by 6x, and
below every default already in `PERFORMANCE.md` §1 — the smallest of those is
+0.0907 ms on its weaker capture, and it costs no memory at all. A 1.5%
median on one capture is not worth a fifth of a gigabyte on a GPU where the
A-buffer already grows into the same budget. **Kept at 8.**

**Retry if.** Device memory stops being scarce *and* the workload carries more
concurrent segments per episode than these captures do (entry 24 records the
fan-out's value swinging 2.73 ms to 0.076 ms across captures on exactly that
property), *or* the per-lane queue sets are made to share their large
allocations so extra lanes are close to free — that last one changes the trade
rather than the measurement, and is the only version worth building.

**Method note.** The 8-lane constant had never been tuned, and nothing in the
tree said so. When a hard-coded constant sits in a load-bearing mechanism,
measure it once and write the number down even when the answer is "leave it" —
this entry exists so the next person spends ten minutes reading instead of an
afternoon rebuilding.

---

## 39. Hoisting the shade chain's argument blocks above the rasterizer — REFUTED

2026-09-02. Branch `exp/argblock-hoist` commit `738dd40114f` (kept, not merged).
Flag `CUDAVK_NO_ARGBLOCK_HOIST`. Proposed as "lead B" of
`notes/PERF_ANALYSIS_2026-09-02.md` at 0.18-0.30 ms/frame.

**Tried.** Every direct shade chain runs `cp_rasterize_stage3` -> copy ->
`cp_fs_compact` -> copy -> FS on one stream. Those copies are not explicit
`cuMemcpy` calls: `cp_upload_end()` leaves a block *owed* and the next
`cp_launch()` flushes it, so each copy is the following launch paying for a
block reserved after the previous one. Measured on a post-lead-A trace, 966
frames: a copy-free dependent link is **0.26 us**, the two copied links are
**3.55 and 3.65 us**, at 44.0 and 66.4 per frame — about **0.37 ms/frame** of
link excess. The change reserves and closes both blocks earlier so one flush
carries them.

**A correction found while building it, worth keeping.** The obvious target —
reserve immediately above the stage-3 launch so *its* flush carries the bytes —
is wrong: **stage 3's flush is empty**, so a reservation there creates a copy
instead of riding one. Two independent facts say so: the copy census has no
copy class in front of any rasterizer stage, and the measured 0.26 us
stage2->stage3 link could not be that short if a flush sat in it. The
implementation therefore hoisted above the *whole rasterizer group*, turning
two copies into one rather than zero, with a predicted -0.10 to -0.22.

**Measured**, three-round alternating A/B in one session per capture, RTX 5090,
control = `CUDAVK_NO_ARGBLOCK_HOIST=1`, all runs exiting 0 with their standard
hashes, full timestamp populations and zero sentinel mismatches:

| arm | favorite3 whole | favorite3 heavy | favorite2 whole | favorite2 heavy |
|---|---:|---:|---:|---:|
| hoist | **+0.0848** | +0.1300 | **+0.0446** | +0.0584 |
| hoist + `CUDAVK_COMPACT_PDL=1` | +0.0315 | +0.0801 | +0.0373 | +0.0336 |

Arms disjoint in the wrong direction on both captures. The PDL the hoist
unlocks recovers about half the loss on favorite3 — consistent with its own
~0.12 estimate, and the first time `COMPACT_PDL` was measured without
`INTERP_INLINE` (dead end 26 only measured it on top of that) — but the hoist's
cost exceeds what the PDL returns.

**Why, and this is inference rather than a traced mechanism** (the confirming
trace failed to record kernels and was not retaken): the merged copy was moved
onto the link in front of the whole rasterizer group, which is the batch's
critical path, and away from two tail links that were already device-paced.
Fewer copies is not the same as cheaper copies.

**The rule this sharpens.** Entry 22 says a launch-removal credit is only
collectable where the launches were serial. The same caution applies to
*relocation*: **moving a device operation pays only if the link it lands on is
cheaper than the links it leaves, and that has to be measured on the arm that
moves it.** A link's nominal cost when idle does not predict what it costs
when a copy is placed in it.

**Retry if.** The remaining half of the original idea is untested and is a
different change: move the clip's scratch allocations above the *vertex shader*
launch, where a copy already exists (107/frame, on a link that is host- or
gate-paced 79% of the time), and prepare the shade blocks there too. Then both
copies vanish into an existing copy instead of forming a new one. The branch's
report sizes it at -3.16 us per chain, about -0.21 ms/frame, and notes it moves
the clip-refusal fallbacks and takes the preparation out of the peel loop — so
it needs its own flag and its own A/B. **The arithmetic in this entry is not
evidence for that one**; it is evidence that placement is what decides.

## 40. The memory residuals (D, E, K): refuted as a class on a measured ceiling

**What it was.** Three items from the 2026-09-02 analysis, each proposing to
move a small driver allocation out of managed memory: **D**, the fused vertex
fetch's `vfetch_vid`/`vfetch_iid` tables out of managed scratch into the pinned
upload arena (estimated 0.07-0.15 on the heavy band); **E**, `cp->peel_any` and
the TRANSFER_SRC-only staging buffers pinned host-resident (0.07-0.14); **K**,
the UBO ring as device memory with a pinned shadow (0.07-0.25).

**Why they are closed together.** They are one population -- unified-memory
page-fault stall -- and that population was never measured, only estimated per
item. It is now, and it is smaller than the sum of the estimates.

A page-fault trace of favorite3 (`nsys profile --cuda-um-cpu-page-faults=true
--cuda-um-gpu-page-faults=true`, 911 real frames):

| | |
|---|---:|
| GPU page-fault events | 8,626 |
| pages migrated | 125,966 (138/frame) |
| CPU faults on managed pages | 26,307 (29/frame) |
| **total GPU fault stall** | **0.3735 ms/frame** |

**0.3735 ms/frame is the ceiling for D, E and K put together**, and it is
generous twice over: it counts every fault's full service time as if none of it
overlapped other work, and it assumes a perfect fix that eliminates all of it.
The admission gate for a new mechanism here is 0.500 ms/frame. The whole class
does not clear it even when all three are built and all three work perfectly.

The distribution makes it worse for these three specifically. The faults land
in exactly **two** managed regions, and they are not evenly weighted:

| region | span | pages/frame | share |
|---|---:|---:|---:|
| 0x70dbdc189000 | 21.46 MB | 124.6 | **90.1%** |
| 0x70db21800000 | 4.20 MB | 13.7 | 9.9% |

The second region faults **exactly once per frame** (911 events over 911
frames) across a 4.2 MB span -- one whole-buffer touch per frame, which is the
shape D and K describe. It is worth **0.037 ms/frame**. The small driver
allocations these three items target cannot be the 21.5 MB region, so the
realistic budget for all three is a tenth of a millisecond, not the 0.21-0.54
the estimates summed to.

**What this does not say.** It does not identify the 21.5 MB region, which owns
0.336 ms/frame on its own. That is an app-side `vkAllocateMemory` mapped
managed (`cpvk_device_memory.c:807`), not one of the driver's own scratch
buffers, and it is the only part of this class that was ever big enough to
matter. Entry 37 already measured the obvious lever on it -- forcing residency
-- and it made the frame *slower* (+1.98 ms RTX). A narrower attack on that one
region is the only version of this idea still worth anyone's time, and it
belongs to the same conversation as the readback fence, because both are about
what the application asks for rather than how the driver serves it.

**The rule.** Three items estimated separately summed to more than their shared
population contains. Estimates that share a mechanism have to be sized against
that mechanism once, before any of them is built -- which is the same rule
entry 24 and the renderer2 redesign (36) were closed by, applied earlier and
for a tenth of the cost. The trace that settled all three took nine minutes and
was only possible because the harness stopped crashing, which is its own
argument for fixing the tools first.

## 41. Device-decided episodes (G): the wait it removes is the GPU's, not the host's

**What it was.** The last open driver-side lead, and the one carried longest:
let the device decide episode sizing so the renderer stops reading back
counters and blocking on them. The 2026-09-02 analysis modelled it at 0.50 ms
oracle, 0.2-0.4 realistic, and specified a one-to-two day "oracle replay" to
price it -- record every read-back value, replay it without waiting.

**What was measured instead.** The driver already reports what it waits for
(`CUDAVK_PLAN_STATS`), and already carries the iteration-29 conversion probe at
the drain (`CUDAVK_WAIT_SPIN_BEFORE_US`). Neither needed a line of code.

The population, favorite3, 3,473 frames:

| wait site | total | per frame | each |
|---|---:|---:|---:|
| episode drain | 3,196.1 ms | **0.920** | 246 us |
| segment counters | 1,019.2 ms | 0.294 | 200 us |
| peel checks | 365.8 ms | 0.105 | 93 us |
| desc uploads | 20.3 ms | 0.006 | 6 us |
| **total** | **4,601.4 ms** | **1.325** | |

Larger than the model. And entirely uncollectible, which the probe shows by
injecting **pure host busy-work before the drain's sync** and measuring the
frame (3 alternating rounds each, every gate passed):

| injected per frame | whole delta | heavy delta |
|---:|---:|---:|
| 0.224 ms | +0.0111 | -0.0133 |
| 0.449 ms | -0.0032 | +0.0341 |

**0.449 ms/frame of host time was added and the frame did not move.** Both arms
overlap at both levels, against a session drift of 0.06. The host arrives at
that sync at least 120 us early on every one of its 3.74 calls per frame: the
wait is the GPU finishing, not the host being slow.

**Therefore G collects nothing**, and no implementation of it can do better --
device-decided episodes remove a wait whose time is not the host's to reclaim.
The oracle replay would have measured the same zero after one to two days of
building it.

**The rule, and it is the same one as entry 40.** A wait is not a cost until
something is shown to be waiting *for the host*. Blocked host time was already
excluded from frame-time credit by this project's rules; what this adds is the
cheap positive test -- **inject host time into the wait and see whether the
frame notices**. Absorption is a direct measurement of slack, it takes six runs,
and it is available at any site that can be made to spin. Ask it before
designing a mechanism to remove a sync.

**What it changes beyond G.** The driver is no longer host-bound in the way
`CLAUDE.md` described (seventeen blocks, 12.44 ms of a 15.74 ms frame waiting).
It blocks **7.25 times a frame for 1.325 ms of a 5.94 ms frame**, and that
1.325 is GPU-paced. The remaining distance to the 5.0 goal is not on the host
side of the driver.
