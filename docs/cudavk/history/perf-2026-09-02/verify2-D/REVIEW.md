# Verify Lead D — convergence check for the alpha-test retry loop

Verdict: **OVERSTATED.** Mechanism real, correctness argument holds, the trace shows the
passes are dead from pass 2-3 on, and the device time is collectable 1:1 where it sits. But
the 0.13-0.19 ms/frame ceiling is a *light-band* number (frames 1391-1802, 77% of frames
carry one retry draw) multiplied by the *median* draw count; in the heavy band the draw is in
36-51% of frames and the whole-window mean is 0.35 draws/frame. Corrected: **RTX heavy band
0.07-0.11 ms/frame mean (0.03-0.08 on the band's hot-tail median), whole window ~0.07 mean
and 0.00-0.03 on the whole-window median; B200 0.07-0.13 heavy mean (HYPOTHESIS).**

Scripts and outputs: `retry.py` (`heavy.out`, `light.out`), `collect.py` (`collect-heavy.out`),
`conv.py` (`conv.out`), `median.py`/`wholemed.py` (`median.out`, `wholemed.out`),
`align.py` (`align.out`), `bubble.py` (`bubble.out`) in this directory. All read-only SQL on
`/tmp/postdead/trace.sqlite` (light band, pre host-pin) and `/tmp/postpin-rtx2/trace.sqlite`
(heavy band, post host-pin). Frame = 2 external submits; `ts.txt[i] - systemClockNs`.

## 1. Mechanism (code, MEASURED in trace)

- `retry = fs->uses_discard && cp->reject && cp->resolved && color_data` (cp_renderer.c:6111);
  `passes = retry ? CP_DISCARD_LAYERS` (6179; `CP_DISCARD_LAYERS 8`, kernels/cp_rast_types.h:345).
  Before pass 0: `cuMemsetD8Async(resolved, w*h)` + `cuMemsetD32Async(reject, 0xFF.., w*h*8*4)`
  (6193-6196) = 0.92 MB + **29.5 MB** per retry draw (8.8 us device, MEASURED).
- Pass loop 6860-7090: pass k>0 clears the visbuf (7.37 MB), `rast_args.reject_passes = pass`;
  the only convergence check is `if (peel)` (7072-7090: `peel_advance` + `cp_sync_timed` +
  `peel_any` with `check_interval` doubling from 1, i.e. checks after pass 0, 2, 6, ...).
  The retry arm has **no check**: 8 passes unconditionally.
- Per pass the rasterizer re-rasterizes **every** triangle, skipping only (tri,pixel) pairs in
  `reject` (`cp_tri_rejected`, cp_rasterize.cu:615-626, loop over `reject_passes` entries —
  which is why stage1 grows 6.8 -> 7.8 us from pass 1 to 7). `cp_fs_compact` does not read
  `resolved`, so the FS re-shades every covered pixel every pass; only `cp_fs_writeback_one`
  (cp_fs.cu:1104-1122) returns early for `resolved[pixel]`, records `reject` for a discarded
  fragment on an unresolved pixel, and sets `resolved` otherwise.
- Retry draws are identified in the trace by the 29,491,200-byte memset on stream 13 (exactly
  one per draw; 921,600-byte `resolved` memset count matches: 402/402 heavy, 353 light; no
  other reject-sized memset exists for any target, so no retry draw on another target).
  Every one runs exactly 8 passes (`passes per draw: {8: 402}` and `{8: 318}`).

Pass op sequence (heavy band, MEASURED): pass 0 = `clip_rast_fused s2 s3 memset1843200 memcpy
compact memcpy main writeback` (9 ops, 31.9 us wall); pass k>0 = `memset7372800 memset12
stage1 s2 s3 memset1843200 memcpy compact memcpy main writeback` (11 ops, **34.6 us median
wall, 26 us summed device, 8.9 us intra-pass gaps**). Passes 1-7 wall 262 us median per draw
(268.8 mean); whole draw incl. reject memset 307 us. Light band: identical (35 us/pass, 266 us).
Per-pass kernel medians (us): s1 6.8-7.8, s2 2.5, s3 1.4, compact 5.4->4.8, FS 2.78 (4096
blocks; its floor), wb 1.95/1.86/1.41...

## 2. Do passes 2-8 shade anything? (the report's UNKNOWN) — INFERRED from durations

`conv.out`: writeback duration is bimodal — 1.8-2.0 us when it writes colour/depth/reject,
**1.41 us when every fragment returns at the `resolved` early-out**.
- pass 1 wb at floor: 1.7% (light) / 7.9% (heavy)  -> converged after pass 0
- pass 2 wb at floor: **87.5% / 77.9%**            -> converged after pass 1
- pass 3 wb at floor: **98.9% / 91.8%**            -> converged after pass 2
- compact is flat within 0.03-0.06 us from pass 2 to 3 and 0.16-0.32 us pass 3 -> 7 (coverage
  essentially constant; the s1 growth is the `reject_passes` loop, not new work).
So passes 3-7 are no-ops for >=92% of draws and passes 2-7 for ~80%. This is INFERRED (a
duration signature), not a count; `CUDAVK_DEBUG_DISCARD=1` gives the direct count and would
settle it. The FS duration says nothing either way (grid = max_pixels, 2.78 us shell).

Stop rule that is provably exact: **"pass k wrote zero new `reject` entries"**. Then pass k+1
resolves the same visbuf (same reject set; `atomicMin` on a key that includes the triangle id
is order-independent), every covered unresolved pixel of pass k became `resolved` (none was
rejected), so pass k+1's writeback returns early everywhere — bit-identical output, hash
unchanged. `CUDAVK_DEBUG_DISCARD`'s per-pass `discarded` count equals that quantity here (a
resolved pixel's nearest non-rejected fragment is the one that resolved it, so it does not
discard again), so the existing probe measures the right thing.

## 3. Is the saved device time collectable? (MEASURED, heavy band `collect-heavy.out`)

- Side streams 14-21 are **idle for 100%** of passes 1-7 (busy fraction median 0.000, p90
  0.000): main is the device critical path; the merge rule (DEAD_ENDS 22) applies at overlap
  factor 1.00 — full credit.
- Host lead at the retry draw's first op: **1.93 ms median (p10 1.70, p90 2.14)** heavy;
  0.18-0.28 ms light. The device, not the host, paces this stretch.
- The host's next blocking call after issuing the draw is `cuCtxSynchronize` in **402/402**
  cases (the S4 scratch-reclaim/S5 upload-wrap drain of the host-sync report), blocked
  **3.25 ms median**, returning 1.4 ms of device time after the draw ends. A whole-context
  drain is the +1.02-slope kind of site (PERFORMANCE §7): device time removed before it
  lands on the frame one for one.
- Conversion caveat: main is only 70.7% busy between the draw's end and that drain's return
  (406 us idle median) — dependency structure inside the queued work, all device-internal
  (host blocked), so it shifts with the saving; no absorption mechanism identified.

## 4. Price of the check itself

Two forms:
- **Host stop** (do not issue passes k+1..7): one `cp_sync_timed` + 4-byte DtoH per check.
  The host blocking is free where the device is 1.9 ms ahead (PERFORMANCE §6.1/§7: peel-site
  slope -0.03). The real price is the pipeline bubble on main after the sync returns:
  MEASURED on the 6,835 existing `cuStreamSynchronize` sites in the heavy window
  (`bubble.out`): **p50 11-14 us, p90 22-27 us** (sync-return lag 0.8 us + host re-issue
  10-13 us, CUPTI-inflated, so ~8-11 us uninflated). Budget **12-25 us per check**.
- **Device stop** (flag written by writeback, later kernels early-exit): DEAD_ENDS 19 —
  "a device-side early exit is already the current behaviour"; the launches are still
  issued and the two `cuMemset*Async` (7.37 MB + 1.84 MB, ~4.5 us device) cannot be
  predicated at all. Savings limited to the s1 body (6.8->~2) and compact body (5->~2):
  ~8-10 us/pass x 6 = **~50-60 us/draw = 0.02-0.04 ms/frame**. DEAD_ENDS 25 closes device-
  side launch/conditional graphs (7.3 us/link, <=0.037 prize). Not worth building; the host
  stop is the only real form, exactly as the report says.
- **Prediction** (PEEL_PREDICATE-style, run last frame's pass count): rejected by
  PERFORMANCE §7 (+0.11 ms from overshoot, "budget about 8 launches per wasted pass") and
  unsafe here (an undershoot leaves unshaded pixels — wrong frames, fails Gate 3). A real
  stop is required.

## 5. Arithmetic

Per retry draw, host stop, check after every pass from pass 1 (bubble 12-25 us):
- converged after pass 1 (78% heavy / 88% light): skip 6 x 37.4 us (262/7 incl. gaps) = 224
  minus one bubble -> **200-212 us**
- converged after pass 2 (14% / 11%): skip 5 = 187 minus two bubbles -> 140-165
- later (8% / 1%): ~80
Weighted: **~187 us/draw heavy, ~197 us/draw light; hard upper bound 212** (all draws at
pass 1). The reject/resolved memsets (9.7 us/draw) and pass 0-1 are unaffected.

Draws per frame (MEASURED, `median.out`/`wholemed.out`; frame numbering is by submit index,
deterministic replay, so the two traces' frames align):
| window | draws/frame mean | frames with a draw | frames in the window |
|---|---:|---:|---:|
| light 1391-1802 (postdead) | 0.77 | 77% | retry draws in frames ~1350-1710 only |
| heavy 2150-3149 (task's band) | **0.403** | 36% | draws in 2450-2920; none 2920-3150 |
| heavy 2450-3149 (census band) | 0.574 | 51% | runs of 185 and 171 retry frames |
| whole window 1391-3449 | **0.351** | 33% | frames 1803-2448 uncovered by any trace; assumed none (frame times 3.7-5.7 ms) |

Ceilings (RTX, ESTIMATED from the above):
- light band: 0.77 x 0.197 = **0.15 ms/frame mean**; on the band median 0.15-0.20 (retry
  frames are the upper 77% of the band, so the median frame carries the draw). The report's
  0.13-0.19 is this number and is fair *for that band*.
- heavy band 2150-3149: 0.403 x 0.187 = **0.075 mean**; census band 0.574 x 0.187 = 0.107.
  On the band's hot-tail median: subtracting 0.15/0.20/0.26 ms per draw moves the 1000-frame
  median by **-0.033 / -0.043 / -0.075** (retry frames sit at 9.7 ms against a band median of
  8.6, mostly above it).
- whole window: 0.351 x 0.19 = **0.066 mean**; the whole-window median moves **-0.013 to
  -0.025** for -0.15..-0.26/draw (retry frames are either below the median (light band ~6 ms)
  or far above it (heavy 8.4-10.3)). This is the owner's metric (6.68 -> 5.0).
- B200: per-pass cost is latency shells + memsets, not arithmetic; kernel-wall ratio 0.91-1.17
  (b200 REPORT §2.4), host lead unknown there. **0.07-0.13 ms/frame heavy mean, HYPOTHESIS.**

CUPTI: all device figures are kernel/memset timestamps (not inflated); the bubble and host
lead are host-side (inflated ~1.28x), which only makes the bubble smaller uninflated.

## 6. Precedent

- PERFORMANCE §6.1 / §7 (peel checks CLOSED 0.00; `CUDAVK_WAIT_SPIN_US` slope -0.03 at the
  peel site): says host blocking at a device-paced site converts at ~0, i.e. the check's
  *wait* is not the cost. It does not price a *new* sync's bubble; measured here at 11-14 us.
  It does not refute a retry stop — it removes the "sync = 0.5-6 ms" objection.
- `CUDAVK_PEEL_PREDICATE` (+0.11 from overshoot): refutes prediction, not a real stop.
- DEAD_ENDS 19: refutes the device-only variant (early-exit launches are still issued; memsets
  are not kernels). DEAD_ENDS 25: no device-side conditional issue. DEAD_ENDS 22 (merge rule):
  satisfied — the removed ops are strictly serial (side streams idle, main critical).
- DEAD_ENDS 9 (idle blocks <0.2 ns): consistent with the FS/compact shells being launch- and
  latency-priced; shrinking grids would not help; skipping the passes is the only lever.
- Not covered by any entry; TODO 'Performance leads' does not list it. **Why different**
  from the peel closure holds: peel already stops at the first empty interval; the retry arm
  never stops, and the passes it runs are 26 us of device work + 9 us of gaps each, not host
  blocking.
- Double counting: the retry draw is one of L1's 8.65 failed appends (discard => "blended"
  in `cpvk_batch_eligible`, `peel = !retry && ...` at 6172 => `abuf` false => back-out =>
  re-executed on main where the 8 passes run). L1 accounts the excursion/idle *before* the
  re-execution, L3 the passes *inside* it: additive, not overlapping. closure-audit did not
  model L3.

## 7. Feasibility

- Change: (a) `cp_fs_writeback_args` gains a `reject_count` pointer (ABI: kernels/
  cp_rast_types.h static asserts, invariant 13); writeback does one `atomicAdd` per new
  reject entry (after the `cov` and `resolved` early-outs, so helper lanes never count —
  invariant 10 intact); (b) zero it per pass by widening the existing 12-byte
  `cuMemsetD32Async(cp->cur_qset.counts, 0, 3)` (6927) to 4 words, or use the pooled counter
  block; (c) in the pass loop, `if (retry && pass >= 1) { cp_sync_timed(...wait_retry...);
  cuMemcpyDtoH(&n, ..., 4); if (!n) break; }` (or a peel-style interval); (d) registry flag
  `CUDAVK_NO_RETRY_CHECK` (invariants 19-21), `FLAGS.md` regen via `cp_debug_doc.py --check`,
  `cp_no_getenv.py`, PLAN_STATS wait site.
- `passes_run` is only consumed by the A-buffer log comparison (2321-2520, TEMPORARY abuf
  paths); nothing after the loop assumes 8 passes.
- Invariants: no order change (same stream, same draw), no merging change, uploads flushed by
  `cp_sync_timed` itself (713-733), arena rewind per pass unchanged. Blast radius small:
  one kernel, one arg struct, ~25 lines of host code.
- Gates: stdout hash must be **identical** (the stop is provably exact); 18-sample sweep
  (samples with `discard` FSes exercise both stop-early and run-long); Gate 3 sentinels on
  favorite3 — check the sentinel set includes a frame in 1350-1710 or 2450-2920, otherwise
  add one; favorite2 must be measured too (CLAUDE.md rule) — its retry-draw count is unknown.
- Risk: none identified beyond the `atomicAdd` contention on a hot discard (bounded by the
  number of discarded fragments, small here: wb difference 0.45 us).

## 8. Decisive probe (cheaper than building)

1. Zero-cost: **`CUDAVK_DEBUG_DISCARD=1` on ~20 frames inside 1400-1700** (light band, one
   draw/frame; the flag syncs per pass so timing is void). Expect `discarded 0` at pass 2 for
   ~80-88% of draws and at pass 3 for >=92%. If the counts stay >0 through pass 7, the
   duration inference is wrong and the lead dies.
2. Ceiling in one A/B without writing the check: a timing-only build with `CP_DISCARD_LAYERS`
   set to 2 (or 3), three alternating rounds against control, medians on the whole window
   and on 2150-3149. It removes exactly the passes the check would remove and *omits* the
   bubble, so it is a strict upper bound on the gain (predicted: <=0.03 whole-window median,
   <=0.08 heavy-band median, ~0.11 heavy mean). The hash/sentinel diff of that build also
   reports directly how many frames need more than 2-3 passes. If the upper bound does not
   clear 0.05 ms on the heavy-band median, do not build the check.

## 9. Must fix in the lead as written

1. Ceiling quoted for the light band with the median draw count; state the heavy-band and
   whole-window means (0.40/0.35 draws/frame) and the median-metric effect (0.03-0.08 /
   0.00-0.03).
2. "One added host round trip" -> price it as a main-stream bubble (12-25 us, measured), and
   expect ~1.2 checks per draw with a per-pass check (or use the peel doubling interval and
   accept 1 wasted pass for the 78-88% that converge at pass 1).
3. The FS-at-floor observation is not evidence of convergence (grid = max_pixels); use the
   writeback floor (1.41 us) and the compact plateau, or the direct count.
4. Note the draw is an L1 back-out re-execution; the two leads are additive but touch the same
   draw path and should be measured in one session if both are built.
