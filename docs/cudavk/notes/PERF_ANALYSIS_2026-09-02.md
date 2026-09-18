# What can still move the frame — analysis of 2026-09-02

Read-only analysis of cudavk's performance on the compiled `favorite3` and
`favorite2` replays, written for the owner. Nothing was built and nothing ran
on a GPU: every number comes from the documents in this tree, from the two
Nsight Systems exports that survived in `/tmp` (`/tmp/postdead/trace.sqlite`,
light band, commit `f3caa35e4ce`, pre host-pin; `/tmp/postpin-rtx2/trace.sqlite`,
heavy band, post host-pin), from the no-CUPTI perf profile in
`/tmp/favorite3-perf-current/`, from the paired ncu CSVs (`/tmp/ncu-rtx/v2.csv`,
`/tmp/ncu-b200-v2.csv`), from the harness source under `~/favorite3-cpp/out/`,
and from the driver at `1f2b3c99e00`. The eight evidence reports, one design and
eleven adversarial reviews it rests on are archived in
`history/perf-2026-09-02/`; each carries its SQL and derivations. **Every lead
below carries the number its adversarial review left it with, not the number
the evidence report first proposed**; §5 lists both.

Labels: **MEASURED** (document, code or trace), **MODEL** (the
dependency-respecting replay of a trace described in §1.4, validated against
the unchanged trace to 15 µs per frame), **ESTIMATED** (arithmetic on measured
inputs, shown), **HYPOTHESIS** (not measured). Conventions are `WORKFLOW.md`
§4: a frame is two submits; favorite3's real work starts at frame 1391; census
per-frame figures are means, the frame time is a hot-tail median; the heavy
band is frames ~2150-3150.

The state of play: favorite3 **6.68 ms** whole-window median on the RTX 5090
(light band ~4.0, heavy band **8.2-8.4**), **10.79** on the B200; favorite2
5.48 / 8.97. The goal is <= 5.0.

**Outcome (2026-09-03):** `LEAD_RESULTS_2026-09-02.md` records what each lead
measured when built. Six landed — A, C, the VS-launch form of B, F, the
context-scope trim and the layered copy — for **−0.736 on favorite3 (6.676 →
5.940) and −0.458 on favorite2 (5.480 → 5.022)** on the RTX; on the B200 the
set measured −0.985 / −0.678 (10.868 → 9.883, 8.961 → 8.283). B's first form
regressed (dead end 39); I and L were refuted without building; G, D/E/K, E′
and H were then closed by measuring their populations with instruments the
driver already had (dead ends 40-43). Three errors that ledger found in this
note are corrected in place below and marked *(corrected)*; §5 carries every
lead's measured outcome.

The B200 has since been traced (`B200_MEASUREMENTS_2026-09-03.md`): kernels
1.49x, API calls 2.3-2.6x, unified-memory fault stall 1.83 ms/frame in one
application allocation, the fan-out worth 1.15 ms there; the NUMA-binding
result did not replicate and 16 lanes showed nothing on favorite2. And the
floor is now known: NVIDIA's own driver runs the same submits at **0.52 ms**
(`NATIVE_DRIVER_REFERENCE.md`), so the 2-3 ms estimate for a
scope-granularity design in §2.2 is not the limit of the hardware, only of
that design.

---

## 1. Diagnosis

### 1.1 The frame is a serial chain of small device operations, not a pipeline

Heavy band, RTX, post-pin trace (MEASURED): over the 1,024-frame window
(frames ~2448-3462) **729 kernels, 298 copies, 114 memsets and ~1,560 CUDA API
calls per frame**, with **17.2 blocking calls** per frame; over the heaviest
700 frames alone (2450-3150) 846 kernels, 354 copies and 135 memsets. *(corrected:
the first version of this note quoted the 700-frame counts against the
1,024-frame window; the ledger's independent count, 727.6, matches the wider
window.)* Device union 5.15 ms of an 8.14 ms traced frame (8.24 untraced). **Exactly one device operation is running for 83% of the
device-busy time**; two or more are in flight for only 0.85 ms/frame, and that
0.85 ms carries 3.6 ms of summed work — the entire benefit of the nine-stream
fan-out is concentrated there. The light-band trace gives the same shape
(3.74 of 4.51 ms single-op).

**The side streams never overlap the main stream.** Over 412 light-band frames
and 700 heavy-band frames the intersection of main-stream busy time and
side-stream busy time is exactly zero (MEASURED, two independent analyses).
Whenever a side stream runs, the main stream is waiting at a gate or a join.
The dependency-respecting critical path of the median frame is **the main
stream, node for node**: 595 of 598 path nodes are main-stream operations and
the path's device time equals the main stream's busy union.

Where the median light-band frame (7.17 ms traced) goes, from the main
stream's point of view (MEASURED, `kernel-chain` §3):

| component | ms/frame | note |
|---|---:|---|
| main-stream operations, busy | 3.40 | 343 kernels, ~100 uploads, ~92 memsets, ~47 copies on the path |
| main-stream gaps, all | 3.70 | 600 gaps/frame; the two rows below and the four after them are each a partition of this 3.70, not additions to it *(corrected: the first version added them and summed to 8.3)* |
| … of which side streams busy | 1.06 | fan-out bursts, failed-append excursions (§1.5) |
| … of which nothing running anywhere | 2.65 | |
| … by size: < 3 µs | 0.64 | 526 gaps/frame, median 1.2 µs — the device front end between dependent launches |
| … 3-20 µs | 0.32 | 44/frame |
| … 20-200 µs | 1.01 | fan-out joins (1.7/frame × 337 µs), sync bubbles, the submit head |
| … >= 200 µs | 1.86 | the frame boundary (1.14 pre-pin) and the two `cuCtxSynchronize` reclaims |
| **frame** | **7.17** | busy 3.40 + gaps 3.70 = 7.10; the rest is UVM rows and ordering slack |

Post host-pin the boundary is **0.55 ms** between the readback submit and the
next render submit (MEASURED, untraced, three runs; light band 0.50, heavy 0.58),
plus ~0.18 ms from the render submit to its first kernel and ~0.13 ms from the
last kernel to the readback submit — about **0.9 ms/frame of device idle at the
frame boundary**, nearly constant across bands. The heavy band's extra time is
inside the render phase: more launches, more fan-outs, more clears.

### 1.2 The kernels are latency shells, not arithmetic

The handoff describes the remaining pools as "clock-bound work that scheduling
does not shrink". The evidence does not support the first half of that sentence.

- ncu on both GPUs: **0.5-15% of peak SM throughput, 0.03-0.42 instructions
  per SMSP-cycle** for every kernel class sampled (MEASURED, `PERF_HANDOFF`,
  re-read from the CSVs).
- Duration versus grid size (MEASURED, heavy band): a 64-thread
  `cp_clip_rast_fused` launch takes **9.76 µs with a p10-p90 spread of 0.1 µs**
  — 153 ns per triangle — while a 262k-thread launch runs at 0.47 ns per
  triangle. One-block VS launches: **55.5 per heavy frame at 6.85 µs each =
  0.37 ms/frame summed**. Least-squares fixed cost per launch is 71% of the VS
  pool, 79% of the clip pool, 100% of the stage-3 pool (`closure-audit` §2.5).
- Rooflines from ncu-measured instruction counts and the geometry census put
  the frame's **real work at 0.3-0.7 ms/frame of device time** (ESTIMATED,
  `arith-vs-latency` §4) against a device union of 5.2-5.9. Only two pools are
  throughput-bound — `cp_fs_compact` (77 full-target visibility scans per
  frame) and memset (1.09 GB/frame of clears, **36 visibility buffer clears per
  frame**, 0.39 ms/frame union; the verifier found the 2080² shadow clears and
  scans are L2-resident at 3.4-5.1 TB/s effective) — and both are volume set
  by the launch structure. The memset pool is absent from the handoff's pool
  list.

The campaign's own `wide_bench.md` §1d had already measured the mechanism: a
launch's duration is the latency of one item, flat from 1 to ~1,200 blocks.
Re-scheduling today's launch units cannot shrink them — that half of the
handoff's sentence is measured true (DEAD_ENDS 20-22, 36) — but changing the
work unit would.

### 1.3 The host: 1.7 ms/frame of pure API issue, none of it the driver's own

perf without CUPTI, same frames as the light-band trace (MEASURED,
`host-launch-audit` §4): `cuLaunchKernel` **1.70 µs**, `cuLaunchKernelEx`
**1.36 µs**, `cuMemcpyHtoDAsync` **1.30 µs** per call, 100% inside libcuda;
the driver's own code on the launch path does not register a single sample.
Heavy band: ~1,560 issue calls ≈ **1.68 ms/frame** of host issue (ESTIMATED
from counts × no-CUPTI prices). Blocking spin is 3.9-4.4 ms/frame and is
device-paced at every site (`ready = 0`, PERFORMANCE §5.2b). Steady-state
driver planning is ~0.33 ms; the app's own recording ~0.4.

### 1.4 What a dependency-respecting replay of the trace says (MODEL)

`closure-audit/trace_audit.py` rebuilds each traced frame as a discrete-event
schedule: every host call in issue order, every device op at
max(host call end, stream predecessor, event dependency) + its recorded
front-end gap, every blocking call returning when its target completes. With
nothing changed it reproduces the traced medians to 15 µs. What-ifs, heavy band
(8.14 traced):

| what-if | frame delta | leaves |
|---|---:|---:|
| device infinitely fast | −4.21 | **3.94** — the host issue floor |
| host infinitely fast | −1.35 | 6.81 — the device chain with its launch gaps |
| launch front-end gaps zero / half | −1.61 / −0.82 | |
| host never blocks, **application readback fence kept** (`verify2-G`) | **−0.51** | 7.63 |
| host never blocks, fence also removed (the closure audit's original figure) | −1.37 | 6.78 — the extra 0.87-1.15 is the app-owned boundary item, §3.4 |
| VS of batch k+1 starts at issue, main stream (`verify2-F`) | −0.45 | |
| delete the 131 serial shade-chain argument copies (`verify2-E`) | −0.41 | |
| delete all memsets | −0.51 | |

Two readings. First, **neither side alone reaches the goal in the heavy
band**: an infinitely fast GPU leaves 3.9 ms of host, an infinitely fast host
leaves 6.8 ms of device chain. Second, the campaign's per-site closure of the
sync structure was not wrong by as much as the closure audit first claimed:
with the application's fence modelled as the dependency it is, all in-frame
syncs together are worth **0.5 ms**, and single sites sum to 67-84% of that.
It is not all-or-nothing; each site is simply ~0.1-0.2 ms, which is inside the
0.04-0.12 ms A/B spread that per-site probes (DEAD_ENDS 29) were run at.

### 1.5 Three things no document records

All three are verified in the code at `1f2b3c99e00`; the first is confirmed
SOUND by its adversarial review.

1. **Draws that are neither order-free nor A-buffer-eligible are routed
   through the pass-episode append and back out.** `cpvk_pipeline_order_free`
   (`cpvk_cmd.c:2707-2737`) returns false for any draw with blending off whose
   depth test or depth write is off, whose depth function is not
   LESS/LEQUAL/GREATER/GEQUAL, or whose fragment shader uses `discard`;
   `cpvk_batch_eligible` (`:2790-2811`) then marks every such draw `blended`;
   `cp_pass_append` (`cp_renderer.c:9298-9333`) gates a side stream, runs the
   vertex stage there, the A-buffer arm at `:6416-6425` backs out because
   `peel` requires `color_data && blend.enable` (`:6172`), the driver **joins all
   eight side streams, finishes the open episode and re-executes the batch on
   the main stream**. The code comment calls the case rare. It fires **8.65
   times per light-band frame and 10.1 per heavy-band frame** (MEASURED:
   every `cp_clip_triangles` launch in both traces — 3,563 and 7,056 of them —
   is on a side stream and matched to a same-grid main-stream re-execution),
   leaving the main stream idle **0.31-0.39 ms/frame**. 88% of the back-outs
   also open a would-be episode and pay its two clears first. Post-process
   fullscreen draws with depth off are the typical victim.
2. **Opaque segment 0 is serialized ahead of its own fan-out.**
   `cp_opaque_append` (`cp_renderer.c:9339-9435`) runs segment 0 on the main
   stream and, for each later segment, `cp_pass_gate_stream` (`:7629-7642`)
   records the gate on the main stream *after* segment 0's launches, so the
   side segments wait for segment 0's stage 3 (MEASURED: every one of 822 light
   and 1,385 heavy bursts has main stage 3 immediately in front; only 1.4-2.9%
   of side segments were host-issued late). The gain is min(segment 0, burst):
   **0.19-0.27 ms/frame light band, 0.13-0.19 heavy** (the segment-0 chain is
   shorter there, 60 vs 111 µs per burst), ~0.2 whole window. The comment at
   `:7605-7627` explains why the gate is per segment (the segment's uniform
   rows are still owed in the upload ring); the fix is a switch-then-flush of
   the owed span on the side stream and one gate after the episode's clears.
3. **The serial argument-block copies in every shade chain — measured
   real, and the proposed fix refuted (dead end 39).** Each direct shade
   issues stage 3 → [interp-block copy] → `cp_fs_compact` → [FS-block copy] →
   FS on one stream, device-paced; a copied link costs 3.55-3.65 µs against
   0.26 for a copy-free one, 44 and 66 per frame, ~0.37 ms/frame of link
   excess. *(corrected)* This note and its review claimed the blocks could
   ride "the copy that already exists" at the stage-3 launch. **There is no
   such copy**: stage 3's flush is empty, so a reservation there creates a
   copy instead of joining one. The implemented form hoisted above the whole
   rasterizer group, turned two copies into one, and the frame got slower by
   0.03-0.08 ms on both captures with disjoint arms — the merged copy landed
   on the batch's critical-path link. The untested remainder is to hoist above
   the vertex-shader launch, where a copy does already exist on a host-paced
   link (see dead end 39's retry clause).

Two more, smaller: the alpha-test retry loop runs all `CP_DISCARD_LAYERS` = 8
passes with no convergence check (`:6111`, `:6179`; the check at `:7072-7090`
exists only for `peel`): writeback duration bimodality says 78-88% of retry
draws have converged after pass 1 and 92-99% after pass 2, but retry draws
occur in only a third of frames, so the median-metric value is 0.03-0.08
heavy and ~0 whole-window. And the render submit's first 178 µs are 32
per-layer 2 KB `cuMemcpy2DAsync` calls plus 32 `cuEventRecord` for one
32×32×16 image copy issued onto an empty device.

**Withdrawn after review:** the "timestamp callbacks stall the stream 0.26
ms/frame" finding double-counted the submit head; the callbacks' real stall
is ~30 µs/frame (0.03-0.06 ms), inside the A/B noise.

### 1.6 The B200: the case for "not the kernels" is weaker than first written

**Superseded on 2026-09-03 by `B200_MEASUREMENTS_2026-09-03.md`**, the first
trace of that host: kernel wall is **1.49x** the RTX's (per-class medians,
same commit, same script), API calls cost **2.3-2.6x**, launch gaps
1.25-2.4x, and unified-memory fault stall is **1.83 ms/frame** there against
0.38 on the RTX. So the "0.742x cycles" reading below was the wrong end of its
own range: at replay clocks the kernels take about the same number of cycles
on both parts and the B200's lower clock is paid in full, *and* every host
event costs more. The rest of this section is kept as the record of what was
inferred before the measurement.

The paired ncu run on disk (45 identical launches, three kernel families,
MEASURED): the B200 executes the same instruction count in **0.742× the RTX's
active SM cycles** (`cp_fs_compact` 0.65×, `cp_clip_rast_fused` 0.61×, stage 3
0.92×), on **148 SMs** (derived from `launch__waves_per_multiprocessor`;
DEAD_ENDS 20's "SM array twice as wide" is wrong). The evidence report turned
that into a kernel-wall ratio of 0.91-1.17× and concluded the +4.1 ms/frame is
host pacing. The review (`verify2-M`) downgrades that to MODEL: **ncu ran at
locked base clocks that nobody recorded**, the CSVs carry no time metric, and
for the latency-bound share of these kernels cycles scale with clock, so the
kernel-wall ratio could be anywhere from ~0.9 to ~1.5. "Multiplicative, not
additive" does not discriminate either (a uniform device slowdown is also
multiplicative). What stands: nothing about the B200 has been measured at all
beyond two frame-time A/Bs — no trace, no perf profile, no launch price, and
its environment (SM clock under a bursty load, PCIe link, NUMA placement of the
process, cgroup quota) has never been inspected. The first B200 step is
therefore **one nsys trace under the checksum-matched 2026.4.1 build, fed
through `closure-audit/trace_audit.py`**, plus ten minutes of environment
checks — not a design.

### 1.7 Where the previous campaign was right, and where its inference failed

**The previous sessions' diagnosis of the frame's shape is correct, and this
analysis reaches the same one by a different route.** PERFORMANCE §5.2 ("a
ping-pong, not a pipeline") and the handoff's "Therefore" paragraph say the
host spends most of the frame waiting on the device, that the device is
running a long dependent chain of kernels each too small to fill it, and that
the wait is device pacing rather than a deletable cost. §1.1, §1.3 and §1.4
here confirm every part of that: the host spins 3.9-4.4 ms/frame, every wait
is device-paced, the device runs one operation at a time for 83% of its busy
time, and the review pass cut the "sync structure" prize from 1.4 ms to 0.5 —
i.e. the campaign's per-site closures were right in size as well as in kind.

Also right, and reproduced here: every measured closure (the wide merge's
+0.410, threaded submit's 0.126, the fan-out surrender, the CDP2 price, the
per-site bubbles, union 5.15-5.21). The merge rule (DEAD_ENDS 22) holds for
what it measured. And its verdict that nothing >= 0.5 ms remains **inside the
per-batch architecture** survives this pass too: after review, no single lead
here clears 0.5 on the whole-window median; what changed is that several
0.2-0.35 leads nobody had found are real, and the largest single item is the
application's.

The disagreement is narrower than "they were wrong". It is about what follows
from the shared diagnosis. The handoff concludes that the time inside those
small kernels is "clock-bound work that scheduling does not shrink", so the
only way out is "less work submitted per frame, or hardware with a higher
clock", and DEAD_ENDS 36 refuses the change of work unit on arithmetic that
keeps one launch per batch per stage. §1.2 says the time inside those kernels
is per-launch fixed latency, not work — which is a different statement about
the same chain, and it changes only one thing: a design that issues fewer,
wider launches would shrink the kernels themselves, not just their gaps.

Not right:

- **"Clock-bound arithmetic"** — refuted by the same documents' counters
  (§1.2). The pools are per-launch fixed cost × launch count.
- **The union-exclusive admission gate** — a dependency replay of the same
  trace gives frame impacts 1.1-1.5× the union-exclusive figure for main-stream
  kernels and **2-4× for small copies and memsets**, whose front-end latency a
  union cannot see. It is neither an upper nor a lower bound.
- **The fan-out surrender charged to every alternative** — it is the price of
  serialising today's small launches. A design with one wide kernel per stage
  finishes in ~max(item), which the fan-out group already takes; the charge is
  0-0.84 depending on the design, not a constant.
- **DEAD_ENDS 37's premise-collapse** ("the migration exists only under nsys
  2026.1.3") is itself a profiler artefact: the 2026.4.1 UVM row counts on
  disk are 699,993 = 7 × 99,999 and 99,999, a clean-exiting probe has 25,420,
  and every harness run exits by SIGSEGV, so the final partial 99,999-row
  buffer is dropped (confirmed by `verify2-K`). A post-pin RTX 2026.4.1 trace
  shows **0.7-0.8 MB/frame each way of residual migration**. The refuted
  mechanism (global host-pin advice, +1.98 ms) stays refuted.
- **The B200 diagnosis** — in both directions, §1.6.

---

## 2. The floor

### 2.1 Current per-batch architecture, everything collectable (ESTIMATED, post-review)

| step | heavy band (8.24) | whole window (6.68) |
|---|---:|---:|
| landed 2026-09-02/03: A, C, B2, F, context trim, layered copy | measured −1.0 heavy (A −0.27, C −0.15, the four −0.66) | **−0.736 measured → 5.940** |
| remaining small items: managed scratch tables, `peel_any`/staging pins, UBO ring (D/E/K) | −0.1 … −0.3 | −0.1 … −0.25 |
| sync structure as a set, fence kept (§3.2 G) — ceiling reduced now that F hides much of the host | −0.15 … −0.3 | −0.1 … −0.3 |
| **projected, driver only** | **~6.6-7.0** | **~5.4-5.7** |
| application pipelines its readback fence (§3.4) | −0.5 … −1.1 | −0.5 … −1.1 |
| **projected, driver + app** | **~5.5-6.5** | **~4.3-5.2** |

*(corrected after the ledger, twice: the hoisting row is gone and every landed
change is entered at its measured value.)* **The whole-window RTX median is
at 5.94; the remaining driver items inside the current architecture reach
about 5.4-5.7, and 5.0 needs the application change.** favorite2 sits at
5.022 already. The heavy band does not reach 5.0 either way; the B200 is at
9.88 after the same set and cannot be projected further until it has a trace.

### 2.2 Scope-granularity architecture (ESTIMATED, `arith-vs-latency` §6.2)

One launch per stage per render scope (or per episode) instead of per batch,
with per-draw state in tables the kernels already read: device time
**0.8-2.3 ms/frame** (VS 0.11-0.22, clip and raster 0.06-0.25, compaction once
per scope 0.04-0.08, FS 0.05-0.20, clears once per scope 0.04-0.13, A-buffer
0.10-0.30, mandatory copies 0.19-0.25, ~80 dependent launch shells 0.2-0.8),
so a frame of roughly **2-3 ms on the RTX** and, because the memory-bound rows
run faster on HBM3e, no slower on the B200. Uncertainty ±2× on every
instruction count (no favorite3 SASS or ncu `main` sample exists) and the
hot-tile imbalance DEAD_ENDS 36 measured (one block = 94% of a grid) is not
solved for free. This is the only route to <= 5.0 on the B200 and on the heavy
band with margin. It is not DEAD_ENDS 36's design as chartered: that verdict
kept the per-batch VS/FS/clip launches and charged the fan-out surrender, the
two premises §1.2 and §1.7 reject; recomputed under corrected premises its own
upper bound is 4.2-5.3 (`closure-audit` §2.5, HYPOTHESIS grade).

---

## 3. Ranked leads (post-review)

Ceilings are RTX ms/frame; "heavy" is the heavy band, "whole" the whole-window
median. B200 figures are HYPOTHESIS until the B200 has a trace. Each entry
names its closest `DEAD_ENDS.md` precedent and its review.

### 3.1 Build now — SOUND or OVERSTATED-but-live, each under a week

| # | lead | mechanism | ceiling | gate | review |
|---|---|---:|---:|---|---|
| A | stop routing non-A-buffer draws through the pass append | a static refusal in `cp_pass_appendable` (`cp_renderer.c:7409`), **not** in `cpvk_batch_eligible` (the blended flag is part of the batch key), reproducing the back-out condition exactly: blend off; `retry` (discard with reject/resolved/colour); depth write mask; samples != 1; `nr_cbufs != 1`; `color_encoding < 0`; no colour. ~30 lines with a `CUDAVK_NO_` revert | **0.25-0.33 heavy, 0.20-0.28 light** (SOUND; ~40% of it is front-end latency, so not additive with launch-latency leads) | side-stream `cp_clip_triangles` count → 0, main-stream `cuStreamWaitEvent` down ~70/frame, hash + 18 sentinels, both captures; note `CUDAVK_DEBUG_EPISODE` does not print this arm — count launches instead | `verify2-A` |
| B | ~~hoist the interp-block and FS-block reservations above the stage-3 launch~~ **REFUTED, dead end 39**: stage 3 carries no copy to ride; hoisting above the rasterizer group measured +0.085 / +0.045 (favorite3 / favorite2), +0.03 / +0.04 with `COMPACT_PDL` | the only version left is above the VS launch, where a copy exists on a host-paced link (107/frame) — untested, needs its own flag and A/B | −0.1 … −0.2 (branch estimate −0.21, unmeasured) | `exp/argblock-hoist` report | `verify2-E`, dead end 39 |
| C | record the opaque gate once after the episode clears; switch-then-flush each segment's owed span on its side stream | ~15 lines + a stream-switch helper; keep the per-segment gate under `CUDAVK_NO_UPLOAD_COALESCE`; add the same fix to `cp_pass_append`, which has the same unordered pair for k >= 1 | **0.13-0.19 heavy, 0.19-0.27 light, ~0.2 whole** | `multithreading` ×20 image compare (the `2eda3cf74c2` protocol), full-frame dump, 3-round A/B both captures | `verify2-C` |
| D | move the `vfetch_vid`/`vfetch_iid` tables out of the managed scratch arena into the pinned upload arena | `cp_scratch_alloc` is not dead code: `cp_renderer.c:5311, 5413-5414` write these per draw on the host and the GPU faults them back mid-submit (7.8 faults/frame — the "unidentified region"); 20-40 lines | **0.07-0.15 heavy** | nsys `--cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true` with `--duration` ending before the SIGSEGV | `verify2-K` |
| E | pin `cp->peel_any` and TRANSFER_SRC-only staging buffers host-resident | `peel_any` is the 1-page ping-pong (CPU fault 7.6 µs after every peel sync, GPU fault in stage 2/3); staging copies fault the copy engine (27 µs vs 0.6 µs, 2/frame at the head) | 0.02-0.04 + 0.05-0.10 | same trace | `verify2-K` |
| F | the submit head: one `cuMemcpy3DAsync` per layered image copy; batch the app's staging copies | 32 × (2 KB copy + event) → 1 | 0.06-0.10 | trace head length | `host-sync` §5 |
| G | drop `CPVK_CTX_SCOPE` from entry points that only record | ~900 push/pop pairs per heavy frame protect nothing where no CUDA call is made | 0.07 (MEASURED host time on the host-bound boundary) | `CUDAVK_CTX_CHECK` over the suite | `host-launch-audit` §7 |
| H | convergence check for the alpha-test retry loop | a reject counter in `cp_fs_writeback_args` (atomicAdd after the coverage and resolved early-outs), zeroed by widening the existing 12-byte memset; after pass >= 1 a `cp_sync_timed` + 4-byte read, break on zero — provably bit-exact | 0.07-0.11 heavy mean, **0.03-0.08 heavy median, ~0 whole** | `CUDAVK_DEBUG_DISCARD=1` in frames 1400-1700 first, then a timing-only build with `CP_DISCARD_LAYERS = 2` as the ceiling A/B; A/B together with A (same draw path) | `verify2-D` |

Sum **0.9-1.3 ms/frame heavy, 0.8-1.0 whole**, with A, B and C carrying
three quarters of it.

### 3.2 Probe first — medium effort

| # | lead | corrected ceiling | decisive probe | review |
|---|---|---:|---|---|
| G | the sync structure as a set, fence kept: device-decided episodes (`cp_abuf_plan` sizes arrays and patches the uploaded blocks, fixed-grid consumers stride over device counts, fail-closed flag, lazy repair) | **0.50 oracle both bands (MODEL); 0.41-0.44 after the 7-10 µs device decision op; realistic 0.2-0.4** after grid-bound, snapshot and arena costs — below the 0.5 gate on its own, and it does not compose with the VS run-ahead (with syncs free the host is already hidden) | the **oracle replay**: record every read-back value at `:8884-8890`, `:6801-6808`, `:7081-7084` in run 1, replay without waiting in run 2 (plus DEAD_ENDS 29's worktree for the context drain); 1-2 days; measured untraced on both captures | `verify2-G`, `design-device-autonomy` |
| F | VS run-ahead: batch k+1's VS on a side lane while batch k's shade chain runs | **0.3-0.5 heavy, ~0.35 whole** (MODEL 0.45); only the VS may run ahead — stage 1 reads the depth `cp_fs_writeback` commits, writes the visibility buffer `cp_fs_compact` is reading, and appends to the queue set stage 2/3 drain; the fused VS also zeroes the queue counters under `fetch_fold` (seed fallback 0.05-0.07) | `CUDAVK_NO_FETCH_FOLD=1` A/B prices the seed fallback with no code; then a `CUDAVK_VS_LANE` flag | `verify2-F` |
| E' | the full kernel-parameter ABI (after B) | **0.1-0.3 beyond B**: parameters are fixed-size per kernel, so either every launch carries the 128-draw maximum (~20 KB, more host memcpy than the call it removes) or the in-parameter table is capped at ~8-10 rows with a device table for larger batches; a `.local` census of every generated kernel is required before any timing claim | standalone microbenchmark of launch cost vs parameter size; `.local` census | `verify2-E` |
| I | fewer visibility-buffer clears in the shadow scope | **0.16-0.19 heavy device, 0.06-0.09 whole** (the light band has one shadow batch per frame); the full per-scope form is DEAD_ENDS 31 plus the memsets, 0.15-0.24 whole | one heavy-band replay with `CUDAVK_DEBUG_EPISODE=1 CUDAVK_DEBUG_BATCHDIFF=1`: which key field splits the 19 shadow batches | `verify2-clears` |
| K | UBO ring as device memory + pinned shadow; prefetch at submit | **0.05-0.15 and 0.02-0.10, overlapping**; the shadow needs GPU-write download before fence return and range uploads for vertex/index/copy sources — cite dEQP `memory.mapping.*` | the page-fault trace above | `verify2-K` |
| L | the long `cp_clip_rast_fused` launches: 5.8/frame above 100 µs (median 79 µs at <= 4 blocks, max 392) summing 0.87 ms/frame, 91% on side streams where they set the burst length | unverified, 0.2-0.4 if per-thread serial work that can be split | one ncu pass on those launch ordinals | `kernel-chain` §2.1 (not reviewed) |

### 3.3 Do not build

Everything in `DEAD_ENDS.md` stays closed except where §1.7 names the premise.
Added by this pass: **segment-level CUDA graphs** (`verify2-H`, REFUTED: at the
inter-sync unit build-then-launch serialises a host burst that is already
pipelined 111 µs ahead of the device, +2.3 ms/frame in the model; the
batch-chain unit is 0-0.15 and needs an in-graph edge latency below today's
1.3 µs stream+PDL gap, unmeasured); **device timestamps** (`verify2-B`, 0.03);
a device-side stop for the retry loop (memsets cannot be predicated, 0.02-0.04);
`cuMemAdvise` on the UBO ring as a probe (it re-measures DEAD_ENDS 37).

### 3.4 Application side — the largest single item, and the owner controls the application

| lead | ceiling | note |
|---|---:|---|
| wait on frame N−1's readback fence after recording N+1 (two readback buffers), with Mesa's threaded submit (`MESA_VK_ENABLE_SUBMIT_THREAD`, DEAD_ENDS 34's branch) | **0.5-1.1** (the boundary measured 0.55 untraced; the fence-dependency difference in the replay 0.87-1.15 MODEL) | DEAD_ENDS 34 measured 0.126 precisely because the app's fence capped it; its retry clause is satisfied by the app change |
| delete the 3,686,400-byte per-frame write *into the readback buffer* | 0.15-0.22 RTX, more on B200 | a GFXReconstruct page-guard artefact: the capture recorded the GPU-written readback as host fills and the replay memcpy's them back before the GPU overwrites the buffer; the real application reads instead, so this makes the replay more faithful |
| fewer render scopes, material-sorted draws (`APP_IMPROVEMENTS.md`) | 0.2-0.4 per removed scope (that document's estimate) | the scope chain is depth 8-9 of 10 and only the app can shorten it |

---

## 4. Recommended programme

1. **Probes, this week, no driver code.** `CUDAVK_DEBUG_DISCARD=1` over 20
   frames in 1400-1700 (H). One page-fault nsys run with `--duration` ending
   before the harness's SIGSEGV (D, E, K, and it settles DEAD_ENDS 37).
   `CUDAVK_NO_FETCH_FOLD=1` A/B (prices F's fallback). On the B200: ten minutes
   of environment checks (SM clock and throttle reasons during an untraced
   replay, `nvidia-smi topo -m` against the process's CPU set, cgroup quota,
   PCIe link), then **one nsys trace under the checksum-verified 2026.4.1
   build run through `trace_audit.py`** — the B200's first real measurement.
2. **Tier 1 (A-H), one to two weeks.** Each behind a `CUDAVK_NO_*` revert,
   three-round alternating A/B on both captures, hash and sentinels; A and H
   in one session (same draw path). Expected heavy band −0.9 to −1.3, whole
   window −0.8 to −1.0: favorite3 to **~5.8** on the RTX, favorite2 to **~4.7**.
3. **The oracle replay (G), one week.** It prices the sync-structure prize
   untraced without building anything; continue only at >= 0.4 favorite3.
4. **One of F or G, two to four weeks**, whichever the probes favour:
   −0.3 to −0.5 → whole window **~5.3-5.5**.
5. **Application changes in parallel** (§3.4): the readback fence is worth
   0.5-1.1 and is what takes the whole-window median **under 5.0**; then the
   readback fill and scopes.
6. **The B200 decision waits for its trace.** If the trace shows the RTX's
   shape with larger per-event gaps, Tier 1 plus G are worth 2-3× their RTX
   value there and the B200 lands around 7.5-8.5 (HYPOTHESIS); <= 5.0 on the
   B200 needs §2.2, and its first slice is the tile-walk benchmark DEAD_ENDS 36
   kept, re-chartered to shade once per scope.

---

## 5. What the adversarial review changed

| lead | proposed | verdict | corrected (RTX) |
|---|---:|---|---:|
| A failed-append back-out | 0.25-0.31 | **SOUND** | 0.25-0.33 heavy, 0.20-0.28 light; fix site moved to `cp_pass_appendable`. **Built: −0.242 / −0.236 (f3 / f2), 9,400 side-stream `cp_clip_triangles` → 0** |
| B device timestamps | 0.13-0.26 | OVERSTATED | 0.03-0.06 (head interval double-counted) |
| C segment-0 gate | 0.20-0.24 | OVERSTATED | 0.13-0.19 heavy, 0.19-0.27 light; switch-then-flush required. **Built: −0.105 / −0.118, about a quarter of the ledger's 0.434 upper bound** |
| D retry-loop check | 0.13-0.19 | OVERSTATED | 0.03-0.08 heavy median, ~0 whole |
| E kernel parameters | 0.25-0.6 (MODEL 0.80) | OVERSTATED | 0.30-0.41 shade chain; the review's "host-only hoisting collects it without an ABI change" was **wrong — built and measured +0.085 / +0.045, dead end 39** |
| F geometry run-ahead | 0.5-0.9 (MODEL 1.80) | OVERSTATED | 0.3-0.5 heavy; VS only, clip and stage 1 illegal. **Built: −0.319 / −0.048**; the review's "no rotating slots needed" was also wrong — `dscratch` rewinds per batch, so the lane uses two dedicated slots |
| B2 hoist above the VS launch (dead end 39's retry clause) | ~0.21 | — | **Built: −0.134 / −0.033** |
| G-ctx context scope trim, F-head layered copy | 0.07, 0.06-0.10 | — | **Built as a pair: −0.003 / −0.035** |
| the four above together | sum −0.456 / −0.115 | — | **measured −0.391 / −0.070**: sub-additive, three of them shorten the same main-stream chain |
| G free syncs | 0.4-1.4 (MODEL 1.37) | OVERSTATED | 0.50 oracle with the fence kept; realistic 0.2-0.4, lower again now that F hides much of the host |
| H segment graphs | 0.3-0.7 | **REFUTED** | <= 0 at the segment unit |
| I clear volume | 0.15-0.25 | OVERSTATED | 0.06-0.09 whole; **then refuted from the code**: depth-only scopes have no colour attachment, so `cpvk_batch_structural` refuses every draw and the 19 shadow "batches" are single draws no key change merges (`SHADOW_VISBUF_CLEARS.md`) |
| L long clip launches | 0.2-0.4 | not reviewed | **refuted by ncu**: a one-block launch is active 48 of 10,869 cycles; its wall time is machine-sharing with the side streams, and summing it is the summed-concurrent error the union-exclusive rule exists to refuse |
| K memory residuals | 0.1-0.3 per item | OVERSTATED | 0.30-0.45 topic total; the scratch tables are the largest item, `peel_any` is the 1-page region; open, wants one page-fault trace |
| M B200 environment | "up to the whole excess" | OVERSTATED | 0-1.5 with a real chance of 0; kernel-wall ratio is MODEL |

Disagreements resolved along the way: host-sync's bubble sum (0.385) and the
closure audit's free-syncs (1.37) were reconciled by the fence-kept replay
(0.50); frame-boundary's untraced 0.55 ms boundary beats host-sync's traced
mean; the kernel-chain report's "device-paced streaming phases" and the closure
audit's "host worth 0.8 at the margin" are both true along an alternating
chain.

---

## 6. Open questions, each with its cheap probe

1. Do alpha-test retry passes 2-8 shade anything? `CUDAVK_DEBUG_DISCARD=1`.
2. Which key field splits the 19 shadow-map batches per heavy frame?
   `CUDAVK_DEBUG_BATCHDIFF=1`.
3. Why do 4-160-block `cp_clip_rast_fused` launches take 200-400 µs? One ncu
   pass on their ordinals.
4. Does a fixed-size kernel-parameter block of 4-20 KB cost more than the
   copy it removes, and does the generated code stay free of `.local`?
   Standalone microbenchmark and a census.
5. Every B200 fact: its trace shape, launch gap, sync round trip, clock under
   load, PCIe link, NUMA placement, CPU quota.
6. Is DEAD_ENDS 37's "zero UVM rows" a dropped buffer? One nsys run that ends
   before the SIGSEGV.

---

## 7. Sources

`history/perf-2026-09-02/`: `frame-boundary/`, `host-visible-memory/`,
`host-launch-audit/`, `kernel-chain/`, `host-sync/`, `arith-vs-latency/`,
`b200/`, `closure-audit/` (each `REPORT.md` with its scripts),
`design-device-autonomy/DESIGN.md`, `design-incremental-sum/census.out`
(heavy-band counts), `verify2-*/REVIEW.md` (the eleven adversarial reviews,
with their replay variants), and `README.md`. Raw traces: `/tmp/postdead/`,
`/tmp/postpin-rtx2/`, `/tmp/postpin-rtx/`, `/tmp/favorite3-perf-current/`,
`/tmp/ncu-rtx/`, `/tmp/ncu-b200-v2.csv` — not persistent; copy them if they
matter.
