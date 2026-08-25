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
- The shipping default today is **15.7514 ms** on old and **5.8982 ms** on
  Crossroads (iteration 28, distribution in
  `docs/cudavk/history/PERF16_ITERATIONS.md`).
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
