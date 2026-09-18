# Verifier review — Lead G: remove the read-back-and-decide sync points as a set

Read-only review, 2026-09-02. Nothing built, no GPU run, no file in the repo or home edited.
Scratch: this directory (`replay_fence.py`, `postpin_fence.log`, `postdead_fence.log`, `*_fence.json`).

**Verdict: OVERSTATED.** The claimed ceiling (-1.37 ms/frame heavy, -1.65 light, MODEL) is a model artefact: the
closure-audit replay let the host run through the application's readback fence. With that one dependency restored the
same replay, same trace, same code path gives **-0.51 heavy / -0.50 light** as an *oracle* ceiling (zero mechanism
cost), and **-0.41..-0.44 heavy / -0.35..-0.39 light** once the one device-side decision op per freed site is priced at
the two measured prices (7.3 µs CDP2, 10.27 µs conditional-graph launch). The "all-or-nothing" thesis is also mostly the
same artefact: with the fence kept, the sum of single-site removals is 67-83% of the all-sites figure. What survives is
a ~0.5 ms/frame oracle prize sitting exactly at the 0.5 ms gate before any of the mechanism's several priced costs, on
top of the largest build this programme has ever refused (EPISODES.md session 11 reached the same design conclusion).

Labels: MEASURED = read from a trace/doc or computed directly; MODEL = output of the dependency-respecting replay;
ESTIMATED = arithmetic on measured inputs; HYPOTHESIS = inference.

---

## 1. What the claim actually models — and the missing dependency

`closure-audit/trace_audit.py` `simulate(free_sync=True)` (lines 394-476):

- The host timeline `H` is **main-thread CUDA API rows only** (`select ... where globalTid=:main`, line ~172).
- Host inter-call gaps are preserved and slide earlier: `t_new = hclock_new + (s - last_e)` (line 405).
- A freed blocking call returns after `min(residual, 3 µs)` (lines 439-441, 466-468).
- Frame boundaries are the **completion worker's** `cuEventSynchronize` returns (lines 40-56) — a different thread.

The application waits on the readback submit's fence every frame (`vkWaitForFences(pFence_3)` immediately after submit B,
frame-boundary REPORT §0 item 1 and §1 steps 14-15, MEASURED from `~/favorite3-cpp/out/src/frame_*.cpp`). That wait
is `cnd_wait` on the worker's `cuEventSynchronize` (`cpvk_sync.c:354-397`) — it is **not a main-thread CUDA call**, so
it appears in `H` only as an ordinary host gap. Under `free_sync` the host clock finishes the render submit ~2.9 ms early
(all in-frame blocking removed), the preserved gap (fence wait + app read + record) is added to that early clock, and the
model issues frame N+1 while frame N's device chain is still running. The real app cannot: it reads the 3.69 MB image
the fence guards before it records the next frame. In other words the closure-audit's "free syncs" arm also deleted the
app's one-frame-in-flight serialisation — the frame-boundary item the frame-boundary report (L1) assigns to the **app
owner**, 0.55 ms/frame untraced post-pin, ~0.8-1.1 ms traced.

Check (MEASURED, `replay_fence.py`): in both traces the first main-thread CUDA call after each readback-fence return is
**never** earlier than the fence return (0 negatives out of 1,013 / 409 fences); the gap is 790 µs median heavy band
(p10 581, p90 1158) and 1,120 µs light (pre-pin, includes the page-fault storm the pin removed).

## 2. Arithmetic — the replay re-run with the readback fence kept

`replay_fence.py` is the closure-audit simulator (its header and `H` construction spliced verbatim, lines 1-183 and
358-393) plus: (a) a site classifier for every main-thread blocking call, (b) a readback-fence dependency — the first
main-thread call after a worker `cuEventSynchronize` whose waited op is the 3,686,400-byte copy cannot start before that
copy's *new* end plus the original (fence-return → first call) gap, (c) per-site freeing, (d) an optional device-side
decision price inserted on the freed stream. Validation: E0 reproduces the closure-audit's E0 exactly (8.1549 / 7.1668
medians), E0+fence is identical to E0 (the dependency is inactive when nothing changes), and `free_sync` without the
fence reproduces the closure-audit's -1.3723 / -1.6493 to four decimals. 40 s per trace, `mode=ro`.

Site census (MEASURED, per frame, CUPTI-inflated blocked time; classification: `cuStreamSynchronize` followed within 5
rows by a sync DtoH of `4*(6+nsegs)` B → S1 drain, of 24 B → S2 segment counters, by none → S3 peel; `cuCtxSynchronize`
→ S4 scratch reclaim/upload wrap/destroy; sync `cuMemcpyHtoD_v2` → S6 descriptors):

| site | heavy (postpin) n/f, blocked ms/f | light (postdead) n/f, blocked ms/f |
|---|---|---|
| S1 drain | 4.60, 0.594 (+0.032 copy) | 5.44, 0.559 (+0.038) |
| S2 seg counters | 1.96, 0.286 (+0.019) | 3.48, 0.461 (+0.027) |
| S3 peel | 1.67, 0.154 | 2.72, 0.253 |
| S4 ctx sync | 1.41, **2.841** | 2.07, 1.596 |
| S6 desc upload | 0.99, 0.012 | 1.00, 0.010 |

(The light-band counts match host-sync §1 to the unit: 5.44 / 3.48 / 2.48→2.72 incl. the wait_quads-less S3 variant / 1.65→2.07 incl. S9.)

What-if deltas, frame **median** ms (mean in brackets), MODEL:

| arm | heavy band (8.155) | light band (7.167) |
|---|---:|---:|
| free all syncs, **no fence** (closure-audit) | -1.372 (-1.826) | -1.649 (-2.067) |
| **free all syncs, fence kept** | **-0.507 (-0.520)** | **-0.496 (-0.604)** |
| host infinitely fast, fence kept | -0.487 (-0.506) | -0.491 (-0.614) |
| free S1 alone, fence kept / no fence | -0.192 / -0.482 | -0.175 / -0.504 |
| free S2 alone | -0.101 | -0.187 |
| free S3 alone | -0.026 | -0.042 |
| free S4 alone | -0.017 | -0.009 |
| free S6 alone | -0.002 | -0.002 |
| sum of singles | -0.338 | -0.415 |
| free S1+S2 (staged form), fence kept | -0.339 | -0.347 |
| free S1+S2+S3 | -0.474 | -0.402 |
| free S1+S2+S3+S4 | -0.501 | -0.482 |
| S1..S4 freed + 2.0 µs device decision per S1/S2/S3 | -0.482 | -0.457 |
| S1..S4 freed + **7.3 µs** (CDP2, DEAD_ENDS 25) | **-0.439** | **-0.391** |
| S1..S4 freed + **10.3 µs** (conditional graph launch, EPISODES 11) | **-0.412** | **-0.349** |
| free syncs, fence kept, launch gaps zero | -2.244 | -1.588 |

Readings:

1. **The corrected oracle ceiling is 0.50 ms/frame on both bands**, 2.7× (heavy) / 3.3× (light) below the claim. The
   0.87 / 1.15 ms difference is exactly the app's fence-to-first-call gap being overlapped — the frame-boundary lead
   (app-owned, L1), counted here a second time.
2. **"Free syncs" ≡ "host infinitely fast" once the fence is kept** (-0.507 vs -0.487). After freeing the in-frame
   syncs the frame is the device chain (5.15 ms union + ~1.6 ms launch front-end gaps, closure-audit §2.3) plus the
   app's serial fence-read-record-submit. The closure-audit's "device-chain-bound at 6.78" becomes 7.65 heavy / 6.67
   light; the 6.78 figure requires the app to stop waiting on its own readback.
3. **Not all-or-nothing.** Sum of singles is 67% (heavy) / 84% (light) of the joint figure — mildly super-additive.
   The dramatic single-vs-all gap in the closure-audit was S1's fence artefact (-0.19 with fence vs -0.48 without;
   S2/S3/S4 alone are identical either way because freeing them never carries the host to the frame end).
4. **S4 (the 1 GiB context drain) blocks 2.84 ms/frame in the heavy band and is worth 0.017.** DEAD_ENDS 29's measured
   +0.037 (light band, pre-dead-scope) is the same mechanism; the model agrees with the one site that was measured.
5. The staged form the task asked to price — **drain + segment counters only** — is worth **0.34-0.35 ms/frame oracle**,
   before the decision op, grid bound, arena and repair costs.
6. Everything here is a *median* frame delta on CUPTI-inflated host time; with the fence kept the freed arm is
   host-insensitive (reading 2), so the inflation does not flatter the result, but the model is still a model — it
   under-predicted the host-pin win (0.48 vs 0.93 measured) because it has no UVM/cache effects, and it has no
   mechanism costs at all.

## 3. Precedent

- **DEAD_ENDS 29** (S4 alone: 1.65 ms blocked → +0.037 measured, sign disagreement between pairs). Reproduced by the
  fence-kept model (-0.009 light, -0.017 heavy). Its retry clause — "only after an architectural change removes the
  next counter dependency" — is what this lead proposes; the model says the combination adds 0.03-0.08 for S4.
- **EPISODES.md session 11** (the strongest precedent, and the closure-audit did not cite it): on favorite3, "a
  deliberately unsafe diagnostic mode skipped the episode counter readback and assumed the bounded path; median frame
  time moved only from 7.52 ms to 7.40 ms" — a MEASURED near-oracle for S1, **-0.12 ms**, with the bounded path's
  sizing overhead included. The fence-kept model puts S1 alone at -0.18/-0.19; the no-fence model at -0.48/-0.50. The
  measurement sits with the corrected model. The same session measured conditional graphs (**10.27 µs per reusable
  conditional-graph launch vs 4.10 µs for two one-block launches**) and concluded a correct device executor needs the
  *complete* normal and fallback tails as graph bodies, because "the fallback itself contains device-dependent peel
  termination plus dynamically linked shader launches" — plus a toolchain prerequisite (link `libcudadevrt` for
  `cudaGraphSetConditional`; the common kernels are NVRTC-from-PTX today).
- **DEAD_ENDS 25**: CDP2 tail 7.32-7.44 µs, host link 1.48-1.51 µs (sm_120, MEASURED). The closure-audit is right that
  entry 25 priced conditional nodes against the launch credit (0.037) rather than the sync-structure prize; but the
  sync-structure prize is 0.50, not 1.37, and the priced decision op takes 0.06-0.12 of it back (§2 table).
- **DEAD_ENDS 11 / PERF §7 S1d / §6 item 4 P1**: worst-case sizing churn (8.6 GB; +0.38 Crossroads) and the
  dense-over-quads slot layout (helper lanes, architectural — ARCHITECTURE §8 #10). Device-side *exact* sizing
  escapes the bound/actual ratio objection (the device knows `quads`), so entry 11 does not refute the mechanism; it
  transfers to the **grid** instead — see §4 item 3. The `capacity/4` gate (:8417-8431) and fail-closed checks
  (`running != quads` :8964, `fill_over‖quad_over` :8903) must be reproduced on the device; host-sync §4.1(i) confirms
  all three hold at HEAD.
- **PERF §6 item 4 run-ahead reach limit** (99.99% of zero-hold deferrals had a late successor): not a refutation, but
  it is the reason "host never blocks" is *not* today's host running ahead — the successor work is a required resolve.
  The model sidesteps that by issuing the resolve's dependents (seg_scatter grid, `cp_abuf_shade`'s 6-9 allocations and
  interp/FS grids, the composite grid, `passes = 0`) at their *recorded* sizes: an oracle. Buildable forms are (a) the
  oracle probe itself, (b) device-side sizing + argblock patching + grid-stride kernels + lazy repair, (c) EPISODES 11's
  conditional-graph episode tail including the fallback.
- **PERF §7 peel closure** (slope -0.03, 2.05 ms/frame injected for free): the model gives S3 0.026-0.042 — consistent,
  and not worth a mechanism.
- **The "why different" argument** ("per-site slopes cannot see an all-or-nothing prize") does not hold once the fence
  is modelled: the prize is 67-84% additive, and the per-site instruments (entry 29, EPISODES 11's -0.12) land where the
  fence-kept model says they should.

## 4. Feasibility — hidden work, invariants, blast radius

Code sites read: `cp_renderer.c` 8860-8995 (drain and its seven host decisions), 6780-6890 (segment counters,
`abuf_prod`, `passes = 0`), 7060-7100 (peel check), 655-700 (`cp_scratch_begin`), 318-360 (`cp_scratch_alloc_device`),
5255-5266 (per-draw `dscratch` rewind), 7655-7702 (`cp_pass_broadcast`, `cp_pass_can_retry`, `cp_pass_fallback`),
8715-8735 (`force_pass_fallback`), 4490-4640 (`cp_abuf_shade` sizing and `interp_dev = cp_upload(&interp)`),
3880-3935 (`fs_attempts++` at FS launch), 9172-9175 (`hw_attempt_start`).

1. **What the host must still size.** At S1 the readback feeds: `grouped`/`quad_dense` (quads×4 B), `seg_cursor`,
   `seg_base` upload, the seg_scatter grid, per group `cp_abuf_shade` → `num_slots = align(4·gq, 256)` sizing 6-9
   arrays via `cp_scratch_alloc_device` (a **host** bump pointer, :318-360) and the interp/FS grids, and the composite
   grid from `covered`. At S2: `abuf_quads`, `abuf_covered`, and the `passes = 0` branch. None of this is in the model's
   "free syncs" arm; the model assumes it for free.
2. **Device-side decision op on the critical path.** One plan kernel per freed S1/S2 (and per peel check if S3 is
   included) at 7.3-10.3 µs measured, or ~2 µs if it can be an ordinary host-issued 1-block kernel whose *consumers*
   read device-side sizes — but see item 3. Priced in §2: 0.06-0.12 ms/frame heavy.
3. **Grid sizing without the readback is the hidden cost that can eat the whole prize.** The consumers (seg_scatter,
   interp, FS, composite) are launched with host-known grids. Without the count the host must launch at a bound: the
   A-buffer grew to 5,242,880 fragments on favorite3 (host-sync §4.1), shade slots = 4·quads ≤ 4·capacity → up to ~82k
   blocks of 256. DEAD_ENDS 9 bounds an idle block at **≤0.2 ns**, i.e. ≤16 µs per bounded launch; at ~8 launches per
   episode × 4.6-5.4 episodes that is up to **0.6-0.7 ms/frame**, larger than the corrected prize. The stale
   "previous episode × 1.5" hint (host-sync §4.1 ii-2) fails whenever a 1-segment episode precedes a 9-segment one
   (EPISODES 11: shapes 5,325×1 seg and 2,730×9 segs). The only cost-free form is machine-sized grid-stride kernels
   (the shape DEAD_ENDS 9 measured as free for the FS grid) — a kernel change to seg_scatter, interp, composite and
   the generated FS launch path, before anything else is built.
4. **Repair semantics under lazy fail-closed.** If the device flag is read at the next sync or at frame end, later
   scopes have already blended over and possibly *sampled* the failed scope's attachments; correct repair is a rollback
   of every scope/copy/dispatch since the failure. That needs a snapshot of the attachments at every scope start
   (~7 MB × ~10 scopes/frame ≈ 70 MB/frame of copies, ~50 µs DRAM time plus 10 copy launches, on a side stream) and
   re-execution of everything after it — a transactional submit in `cpvk_queue_submit`. The fallback itself
   (`cp_pass_fallback` → `cp_draw_execute_batch` per segment) runs the composite/peel paths, i.e. **contains S2/S3
   syncs**, which is tolerable only because overflow never fires on favorite3 (0 "overflowed"/"falls back" lines,
   host-sync §4.1). It must nonetheless be *exercised*: TESTING §4/§7 require a negative control that fails, and §10
   classes synchronisation/episode changes as high-risk needing full-frame external validation, not sentinels.
5. **`cp_pass_can_retry` (:7668-7681).** A retry after any hardware-texture FS attempt in the episode latches device
   loss. Under speculation *every* episode has FS attempts before the overflow answer, so today's fallback path becomes
   device loss on every overflow. The gate must be re-derived with the texture-cache accounting
   (`cpvk_texture_cache.c:214-374`, `fs_attempts`); whether it can be relaxed when the FS early-exits on the device flag
   is a texture-cache question this review cannot settle.
6. **Argblock patching and ordering.** `interp_dev = cp_upload(&interp)` (:4599) and the FS argblock are pinned-staging
   → `cuMemcpyHtoDAsync` flushes (host-launch-audit §1: owed spans flushed at launch). A device patch of the pointer
   fields must run after `cp_upload_flush` of that span and before `cuEventRecord(pass_gate)` gates the side streams
   (`cp_pass_broadcast` :7655ff — the upload-span ordering invariant). Field offsets become kernel-visible ABI →
   `cp_shader_abi.h`/`cp_rast_types.h` static asserts (ARCHITECTURE §8 #13).
7. **Device bump arena.** `dscratch.used` accounting moves to the device; exhaustion → flag; reset by
   `cuMemsetD32Async` at the :5266 rewind points; interacts with invariant #4/#5 (`batch_uploads_live`,
   `cp_scratch_reset`) and with S4's removal (DEAD_ENDS 29's worktree form is the right one).
8. **Blast radius**: `cp_renderer.c` episode tail, `cp_abuf_shade`, four kernels in `kernels/*.cu`, the generated FS
   launch path, `cpvk_queue_submit` (transactional repair), `cpvk_texture_cache.c` (retry gate), plus a `libcudadevrt`
   link if conditional graphs are used. Tests: gate 1 native suite incl. `CUDAVK_FORCE_PASS_FALLBACK` equivalence, gate 2
   18-sample sweep, gate 3 favorite3/favorite2 sentinels (18 each), gate 4 device-side equivalence (the sizing kernel
   changes what the device computes), a forced-overflow negative control, full-frame external validation.

## 5. Double counting with other leads

- **Frame-boundary L1 (app double-buffers its readback wait):** the 0.87-1.15 ms the closure-audit attributed to
  "free syncs" *is* this item. If the app owner ever adopts L1, the free-sync prize grows back toward the closure-audit
  figure — but that is L1's credit, not Lead G's, and it is app-owned.
- **Launch front-end gaps (4.4 / "gaps zero"):** free syncs + gaps zero = -2.24 heavy with the fence kept (vs the
  closure-audit's 4.98 → -3.17 without it). These compose, but the gaps pool is a separate lead with its own costs.
- **Geometry run-ahead (4.1)** and **PERF §6 item 4**: both overlap the host's issue work with the drain; with syncs
  free the host is already fully hidden (reading 2 of §2), so their credit is not additive with Lead G.

## 6. Corrected ceilings

- **RTX 5090**: **0.50 ms/frame oracle** (heavy median -0.507, light -0.496; MODEL, zero mechanism cost, fence kept);
  **0.41-0.44 heavy / 0.35-0.39 light** with the measured 7.3-10.3 µs device decision op per freed S1/S2/S3; staged
  drain+counters only **0.34-0.35** before costs. Grid-bound (item 3), snapshot and arena costs come off these figures
  and are not yet priced; the honest realistic number is **0.2-0.4 ms/frame**, below the 0.5 gate.
- **B200**: no trace exists (b200 REPORT §5; must be taken under nsys 2026.4.1 with a positive control, DEAD_ENDS 37).
  ESTIMATED/HYPOTHESIS: the RTX 0.5 plus the extra per-turn latency of the ~10-14 freed round trips/frame ×
  (s_B − s_R ≈ 10-30 µs, b200 REPORT §3.3 "20-40 µs vs 5-10") ≈ **0.6-0.9 ms/frame oracle**, inside the B200's
  0.3-1.3 ms session drift and resolvable only with ≥4 alternating pairs. `replay_fence.py` runs unchanged on a B200
  trace and would turn this into a MODEL number in 40 s.

## 7. Decisive probe, and a cheaper one

- **Zero code, already done:** EPISODES.md session 11's unsafe S1 skip measured **-0.12 ms** on favorite3; the
  fence-kept model predicts -0.18 for S1 alone, the claim's model -0.48. That is the first check of the corrected
  arithmetic against a measurement, and it agrees.
- **The oracle probe, staged and bounded.** One registry flag (`cp_debug.c` array, not `getenv`) with record/replay
  modes: at :8884-8890, :6801-6808 and :7081-7084 record `ctr[]` / `peel_any` per call index in run 1; in run 2 skip
  `cp_sync_timed` and the DtoH and read the recorded values. Add DEAD_ENDS 29's `~/mesa-scratch2` form for S4. Run ≥4
  strictly alternating pairs in one session, hash-gated (an oracle mismatch produces a fallback or garbage → hash
  change → invalid run), report both bands. **Bar:** the model says -0.50; anything **≤ 0.6 ms/frame closes the lead**
  because the cheapest priced mechanism cost (decision op 0.06-0.12 + grid-stride conversion or grid bound + snapshot
  copies + arena) is ≥ 0.1-0.2. Only a measured oracle ≥ 0.8 would justify designing the mechanism. Effort: 1-2 days.
- Do **not** re-run `CUDAVK_UNSAFE_NO_OVERFLOW` as the probe: it carries the bounded path's sizing churn (DEAD_ENDS 11)
  and measures the mechanism cost, not the oracle.

## 8. Must-fix before this lead is quoted again

1. Model the app's readback fence; quote 0.50 (RTX, both bands), not 1.37/1.65; attribute the 0.87-1.15 ms to the
   frame-boundary app item and count it once.
2. Withdraw "all-or-nothing": singles sum to 67-84% of the joint figure with the fence kept.
3. Price the device decision op (7.3-10.3 µs measured) on the critical path: 0.41-0.44 heavy.
4. Price the bounded grids or commit to grid-stride kernels first (≤0.2 ns/block × 82k blocks × ~8 launches/episode
   can exceed the whole prize).
5. Specify the repair: snapshot-per-scope + rollback of later scopes, the fallback's own S2/S3 syncs, and the
   `cp_pass_can_retry` gate — with a forced-overflow negative control in the test plan.
6. Cite EPISODES.md session 11 (-0.12 measured for the S1 skip; 10.27 µs conditional launch; "cache the complete episode
   tail, not individual raster stages"; `libcudadevrt` prerequisite).
7. B200: take a 2026.4.1 trace and rerun the fence-aware replay before any B200 number is written down.

## 9. Reproduction

```
cd /tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/verify2-G
python3 replay_fence.py /tmp/postpin-rtx2/trace.sqlite postpin 0 0 postpin_fence.json > postpin_fence.log
python3 replay_fence.py /tmp/postdead/trace.sqlite postdead 9286588499 12338994337 postdead_fence.json > postdead_fence.log
```
Fence dependency (SQL inputs, `mode=ro`): worker `cuEventSynchronize` rows (`CUPTI_ACTIVITY_KIND_RUNTIME`, worker tid);
the waited op is the latest `CUPTI_ACTIVITY_KIND_MEMCPY` row with `bytes = 3686400` ending before the return; the first
main-thread runtime row after the return is constrained to `new_end(copy) + (orig_first_start − orig_copy_end)`.
