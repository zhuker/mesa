# Verifier review — Lead A: stop routing non-A-buffer draws through the pass-episode append

Read-only. Nothing built, nothing run on the GPU, nothing edited in the repo. Scratch:
`/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/verify2-A/`
(`backouts.py`, `followups.py`, `fetch.py`, `whatif_L1.py`, `whatif_L1b.py`, logs `*_L1*.log`).

Source lead: kernel-chain REPORT §4 L1. Claim: 8.65 doomed appends/frame, 313 µs/frame main-stream idle in
the excursion window, ceiling 0.25-0.31 ms/frame RTX light band, ~20-line fix.

**Verdict: SOUND, with corrections.** The mechanism is real and exactly as described; the count and the
idle window reproduce on both traces; a dependency-respecting replay (the closure-audit's validated model,
extended here) gives -0.256 (light) / -0.288 (heavy) ms/frame median; the in-tree hand-off price
(PERFORMANCE §4, 32 µs per gate-in/join-out) independently predicts 0.277 / 0.323. No DEAD_ENDS entry refutes
it; entry 22 (merge rule) supports collectability because the excursion is strictly serial with the main
stream (main ∩ side = 0). The corrections: the report's cheapest probe does not exist for this path, the heavy
band has *more* back-outs (10.08/frame), the ceiling omits two main-stream clears and one vertex fetch per
frame that the same fix also removes, and the fix belongs in the renderer's admission predicate, not the
front end's classifier, and needs a revert switch.

---

## 1. PRECEDENT

### 1.1 DEAD_ENDS.md
Index read in full (entries 1-38); grepped for `append`, `backs out`, `re-execut`, `episode`, `discard`,
`order-free`. **No entry covers a failed blended append or the double vertex work.** Closest:

| entry | relation | refutes? |
|---|---|---|
| 22 merge rule | "credit is only collectable where the launches were serial" | supports: kernel-chain §2 measures main ∩ side = 0.000 in every frame; the excursion sits between main ops via gate → side → 8 joins, so its whole window is critical path |
| 20 wide merge | removing launches cost +0.410 when they were concurrent | not applicable: nothing is merged, launches are removed from a serial section |
| 8 flush placement | 76% of flush points empty | not applicable (the excursion's `cp_upload_flush` calls are incidental) |
| 6 tagged A-buffer stage 3 | decision cost 244 µs × 2581 | different mechanism |
| 38 more side streams | side-stream lane count | orthogonal; the doomed append always lands on stream 14 (`k = nsegs % 8 = 0`) |
| 11 bounded path for every episode | episode routing | orthogonal |

### 1.2 In-tree precedent for exactly this fix class (supportive, MEASURED)
`history/VK_NATIVE_DECISIONS.md:5780-5806` and `cp_renderer.c:9239-9250`: the `CUDAVK_ABUF_MIN_TRIS`
size test used to set `abuf = false` *after* fetch+clip inside `cp_pass_append`'s `cp_draw_execute`, "so
every batch it rejected paid for its vertex work twice" (cp_vertex_fetch 32.6/frame vs 24.3
rasterizations). Moving the decision to the caller measured Crossroads 7.85 → 7.59 ms, old 25.32 → 23.45,
17/18 pixel-correct. Lead A is the *remaining* members of that same family (blend off, discard/retry,
depth write, samples, cbufs, encoding) still decided after the vertex work.

### 1.3 Price table (PERFORMANCE §4)
`PERFORMANCE.md:265`: **cross-stream hand-off (one gate in, one join out) ≈ 32 µs per episode**, measured by
`b3f716bd5f5` (moving segment 0 to main removed one hand-off/episode, bit-identical pixels). A doomed append
is precisely one gate in (`cuStreamWaitEvent(seg_streams[k], pass_gate)`, 9300-9305) plus one join out
(8 × `cuEventRecord` + 8 × `cuStreamWaitEvent`, `cp_pass_join` 7588-7606) wrapped around ~3 device ops nobody
reads. 32 µs × 8.65 = **0.277 ms/frame** (light), × 10.08 = **0.323** (heavy) — matches the measured window
(§2) without any new price.

Launch prices: removed real chain launch 0.78-0.81 µs, small copy removed 1.30 µs (`PERFORMANCE.md:253-258`).
Per occurrence the fix removes 2 kernels + 1 HtoD + ~0.5 memset on the side stream and (88% of the time, §2.4)
2 memsets on main — by the launch price alone ~6 µs/occurrence = 0.05 ms/frame; the hand-off price is the
one that matters here and it is 5x larger. This is consistent with `PERFORMANCE.md:286` rule 1: the lever is
removing the *boundary*, not the launches.

### 1.4 "Why different" argument
Holds. The report says the draws take exactly the path they take today after the back-out. Verified in code:
`cp_pass_append` 9319-9333 on `append_failed` does `cp_pass_join`, `cp_pass_finish`, `cp_draw_execute_batch`;
the fallthrough in `cp_batch_flush_defer_why` 9512-9514 does `cp_pass_finish`, `cp_draw_execute_batch`. Same
batch object, same state, same stream (main), same `cp_pass_finish` cut of any open episode. Pixels are
unchanged by construction as long as the up-front predicate is a **subset-or-equal** of today's back-out
condition (§3.1).

## 2. ARITHMETIC

### 2.1 Is every `cp_clip_triangles` a back-out? (MEASURED, both traces)
`backouts.py`: light band (postdead, submits 2782-3606, 412 frames): 3,563 launches, **3,563 on stream 14,
0 on main**, 3,563/3,563 matched to a main-stream `main` kernel with identical `gridX` launched after the
clip. Heavy band (postpin, submits 4900-6300, 700 frames = 2450-3149): 7,056 launches, **7,056 on stream 14,
7,056/7,056 matched**. The other `clip_triangles` launch sites (`cp_renderer.c:6044` unfused path, 6521/6905/
6951 fused-refused fallbacks, 6905 tiled_opaque) never fire in this capture. **Yes — every one is a back-out.**

### 2.2 Heavy band count (the report's UNKNOWN)
**10.08/frame mean, median 11** (light: 8.65 / 9). Not "similar" — 17% more, and the median-frame count is
what a hot-tail median sees.

### 2.3 The 313 µs window reproduced and decomposed (MEASURED)
Per frame, `backouts.py` (chains.py method re-implemented independently, same numbers to 0.1 µs):

| quantity | light band median (mean) | heavy band median (mean) |
|---|---:|---:|
| back-outs | 9 (8.65) | 11 (10.08) |
| side excursion device wall | 222.9 (221.2) µs | 288.8 (295.0) |
| main idle in [excursion start, re-exec VS start) | **313.1 (310.1)** | **385.5 (392.4)** |
| main busy inside that window (episode tail of a cut episode) | 31.0 | 70.6 |
| re-executed VS duration | 38.9 | 42.5 |
| host window, side-VS launch → re-exec launch (CUPTI) | 151 | 157 |
| host API time in it (CUPTI) | 130 | 135 |
| host blocking calls in it | **0** | **0** |
| per occurrence: side wall / main idle / host win | 19.7 / 29.9 / 14.7 µs | 19.7 / 29.2 / 13.9 |
| re-exec lag (kernel start − launch return), per occurrence | 83 µs median, p10 15.6 | 133 median, p10 27, mean 627 |

API calls between the side-VS launch and the re-exec launch, per occurrence: cuEventRecord 8.0,
cuStreamWaitEvent 8.0, cuLaunchKernel 2.1 (VS + clip), cuMemcpyHtoDAsync 1.1, cuMemsetD32Async 1.1,
cuMemsetD8Async 0.2 — exactly the code path (gate, VS, uploads, lazy depth clear, clip, 8-way join).

**Is the idle on the critical path?** The device is *behind* the host at every re-exec (lag median 83/133 µs,
p10 16/27 µs, never ≈0), so the ~30 µs/occurrence of main-stream dead time is device-critical-path time,
not host-issue time. Where the lag is small (light-band p10 ≈ 16 µs) part of the saving becomes host-bound
and collapses to the host cost (~14 µs CUPTI ≈ 7-11 µs real per occurrence: host-launch-audit prices
cuEventRecord 0.09 µs, cuStreamWaitEvent 0.03 µs without CUPTI, so the 16 event calls are ~2 µs; the two
launches, memcpy and memset carry the rest). With 17-23 blocking syncs/frame and the host blocked 2.9-3.9
ms/frame (closure-audit §2.1), device time removed from the main chain converts to frame time.

### 2.4 What the report's window misses (MEASURED, adds to the ceiling)
* **Episode-start clears on main.** `followups.py`: the host sequence immediately before the side-VS launch is
  `cuMemsetD32Async ×2, cuEventRecord(pass_gate), cuStreamWaitEvent(side)` in ~88% of back-outs (312-B counter
  memset 3,145/3,563 light, 6,356/7,056 heavy; `ab->counts` clear 3,686,400 B 1,435/3,563 or 57,600 B for the
  small targets). That is `cp_pass_append` 9262-9285 opening a would-be episode (`nsegs == 0`) that never gets
  a segment: 2 main-stream memsets (~1.5 µs device + ~2.5 µs launch gap each) per doomed append, **outside**
  the window the report measured. ESTIMATED +0.03-0.05 ms/frame.
* **Unfused vertex fetch on the side stream.** `fetch.py`: 1.00/frame of the back-outs is preceded by a
  `cp_vertex_fetch` on stream 14 (417/3,563 light, 700/7,056 heavy) that the chains.py window excludes.
  Extending the window: main idle 310 → **328 µs** mean (light), 392 → **409** (heavy). +0.02.
* The doomed `cp_draw_execute_batch` host pass (scratch allocs, argument packing) between its API calls is
  host work the fix also removes; the replay below keeps it (conservative).

### 2.5 Dependency-respecting replay (MODEL, validated by the closure-audit: E0 error p50 15-18 µs/frame)
`whatif_L1.py` extends `closure-audit/trace_audit.py` with the excursion op set (side VS, its uploads/clears,
`cp_clip_triangles`; 30.6 / 32.2 ops/frame, 0.178 / 0.213 ms/frame of device time) deleted together with
their host API time, keeping the join calls (whose device semantics become trivial once the excursion is gone):

| what-if | light band (postdead 7.153 median) | heavy band (postpin 8.143) |
|---|---:|---:|
| L1a delete excursion device ops only | -0.230 | -0.274 |
| **L1b + their host API time** | **-0.256 (mean -0.248)** | **-0.288 (mean -0.295)** |
| L1b on top of host ×0.78 (CUPTI deflate) | -0.271 incremental | -0.310 incremental |
| L1b on top of launch-gaps-zero | -0.160 incremental | -0.195 incremental |
| control: delete all clip kernels | -0.763 | -1.016 |

(A variant that also skipped the 16 event calls, `L1c`, reads -0.02; that is a model artefact — skipping a
`cuStreamWaitEvent` corrupts the model's launch-gap derivation, which then charges the whole join wait as
front-end latency — and is disregarded.) The saving is stable against host speed (it is device-chain time)
and roughly 60% of it survives even if launch latency were free, i.e. ~40% of it *is* front-end latency
(gate → side start, side end → 8 events → main): **do not add L1 at face value to a lead that removes launch
latency** (closure-audit "launch gaps zero", geometry run-ahead).

### 2.6 Corrected ceilings
* **RTX light band** (frames 1391-1802): report 0.25-0.31 ESTIMATED. Reproduced: window 0.31 MEASURED, replay
  0.25 MODEL, hand-off price 0.28; plus 0.03-0.07 for the clears and fetch not in the window. **Corrected
  0.25-0.33 ms/frame, realistic 0.20-0.28** (the re-executed chain still pays its own first launch gap, and
  part of the window is host-bound in the lightest frames).
* **RTX heavy band** (frames 2450-3149): window 0.39 MEASURED, replay 0.29 MODEL, hand-off price 0.32; plus
  0.04-0.08. **Corrected 0.29-0.41, realistic 0.25-0.33.** Against the 8.2-8.4 ms band this is 3-4%, against
  the 3.2-3.4 ms gap to 5.0 it is ~9%.
* **B200**: the report's 0.4-0.5 by the 1.6x frame ratio is unsupported — this is a latency cost (gate/event
  round trips, small-launch front-end), not clock-bound arithmetic, and the b200 report says those prices are
  unmeasured there (its §3.1: exposed-launch latency L_B 4-8 µs hypothesised; completion→host 2-3x). With 10
  occurrences × ~12 device-side boundaries each, **HYPOTHESIS 0.3-0.6 ms/frame**, sign certain, size unmeasured.

## 3. FEASIBILITY

### 3.1 Where the fix goes, and what the predicate must be
The report proposes classifying in `cpvk_batch_eligible` (cpvk_cmd.c:2790-2811). Better: **`cp_pass_appendable`
(cp_renderer.c:7409)**, which already receives the batch and refuses `fs->writes_memory`, and has
`batch->state` and `batch->scope` — the same fields `cp_opaque_appendable` (7542-7550) reads. Reasons:
(a) the front end's `blended` flag is part of the batch key (`cpvk_build_batch_key` cpvk_cmd.c:2370
`key->blend_enabled = blended`) and of `cp_batch_begin_packet`; changing the classification there changes
what may merge with what and touches `CUDAVK_NO_BATCH_BLEND` semantics; a renderer-side refusal changes
nothing about batching. (b) `cp_batch_flush_defer_why` 9505-9514 then falls through to exactly the back-out's
`cp_pass_finish` + `cp_draw_execute_batch`.

The predicate must reproduce the back-out condition at 6414-6425 from static state, mirroring
`cp_draw_execute`:
- `!state->blend.enable` → refuse (peel, 6172, needs `color_data && blend.enable`; the `!color_data &&
  fs_side_effects` arm is already refused by `writes_memory`).
- `retry` (6111): `fs->uses_discard && cp->reject && cp->resolved && fb->color` → refuse (peel requires
  `!retry`). Mirror the allocation test too, or accept that a discard draw whose reject buffers are absent
  goes classic-peel instead of A-buffer — pixel-different in principle, gate-detectable.
- `state->depth.depth_writemask` (6339), `MAX2(attachment_samples,1) != 1` (6331, `fb_samples` at 5076),
  `fb->nr_cbufs != 1` (6354), `fb->color_encoding < 0` (6359), `!fb->color` → refuse.
- Keep the existing `cp_abuf_enabled/ab->disabled/verify/composite/timing/census`, kernel-handle,
  `frags/grow_to/shade_slot/clist`, `nsegs/next_prim` checks (7433-7449) and the `abuf_min_tris` test in
  `cp_pass_append` 9244-9250 as they are.
The `!abuf` back-out also fires on `cp_abuf_setup` failure and `grow_to` — already handled between episodes
by 7446; nothing dynamic remains. The 1.5-1.7/frame re-executions that go on to an unfused `cp_rasterize_stage1`
re-pass chain (`followups.py`: 684/3,563 light, 1,031/7,056 heavy) are the blend-on+discard (retry) and
blend-on+depth-write (peel) populations — the predicate must include them or they keep backing out.

### 3.2 Invariants (ARCHITECTURE §8)
- 7 (only the four per-segment changes defer) — untouched; the deferring flush still runs, the refusal just
  happens earlier inside it.
- 8 (blended merging requires stable clip) — untouched; batch composition and `stable_clip_early` (5647) are
  unchanged because the front end's classification is unchanged.
- 9 (opaque merging requires `cp_batch_order_free`) — untouched; these draws never reach `cp_opaque_append`.
- Upload-span ordering under `cp_pass_gate_stream` (7605-7642) — *less* exposure, one fewer gate per frame×9.
- 20 (default-on optimisation ships a revert switch) — **must add a `CUDAVK_NO_…` entry to the `cp_debug.c`
  array**, regenerate `FLAGS.md`, run `cp_debug_doc.py --check` and `cp_no_getenv.py`. That makes the fix
  ~30 lines, not 20.
- 21 (old path stays runnable) — the back-out arm at 6414 stays as the safety net; under the revert flag it is
  exercised again.
- Arena generations / ABI — untouched (no kernel change, so TESTING gate 4 is not required).

### 3.3 Hidden work / blast radius
Small. One predicate, one flag, one doc regen. `CUDAVK_DEBUG_EPISODE` output changes only by the absence of
the cut. `cp->plan` has no failed-append counter (`pass_finish_calls`, `pass_finishes` only) — worth adding one
line so PLAN_STATS can show the count going to zero.

### 3.4 Tests it must pass (TESTING.md §2)
Gate 1 `meson test --suite cudavk` (includes `cpvk_batchblend`), gate 2 the 18-sample sweep, gate 3 the GFXR
sentinel frames on **both** captures (18 sentinels each) plus the stdout hash unchanged; timing per WORKFLOW:
alternate arms in one session, both captures, check submit/frame counts. Gate 4 not needed (no device
arithmetic changes).

### 3.5 The report's "cheapest probe" is wrong
`CUDAVK_DEBUG_EPISODE=1` prints `episode-cut: append failed=…` only in **`cp_opaque_append`**
(cp_renderer.c:9434-9437). `cp_pass_append`'s failure arm (9319-9333) prints nothing, and `debug_pass` prints
its `pass?:` line three times per process. Counting `cp_clip_triangles` on side streams in an nsys export is
the probe — and it is already done here for both bands (§2.1), so no further counting probe is needed.

### 3.6 Decisive probe
Build the predicate behind a revert flag; A/B favorite3 and favorite2 alternating in one session, stdout hash +
18 sentinels each arm, submit counts checked; expect −0.20…−0.30 ms/frame on favorite3 RTX (both bands) with
`cp_clip_triangles` side-stream count going to 0 and `cuStreamWaitEvent` on main dropping by ~70/frame. If the
measured win is < 0.1 ms with counts at zero, the model's assumption that the excursion window is device-critical
is wrong and the residual is the host-only 0.06-0.10. Then measure once on B200 to replace the HYPOTHESIS.

## 4. Summary of corrections to the lead as written
1. Heavy band: 10.08 back-outs/frame (median 11), window 385 µs median — higher, not "similar/unknown".
2. Ceiling omits the two episode-start clears on main (88% of occurrences) and one unfused fetch/frame: +0.05-0.07.
3. ~40% of the saving is front-end latency; not additive with launch-latency leads.
4. Fix site: `cp_pass_appendable`, not `cpvk_batch_eligible`; ~30 lines with the mandatory revert flag.
5. Probe as stated does not exist for the blended path; the nsys count is the probe and it is done.
6. B200 0.4-0.5 is a HYPOTHESIS by the wrong scaling law; keep the sign, drop the number until measured.
