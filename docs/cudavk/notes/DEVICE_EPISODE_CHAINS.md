# Device-side episode chaining (ideas B/F family) — measured park, 2026-08-30

Branch `cudavk/device-episode-chains`. Verdict: **all three candidates parked or
refuted by probes; no production code built.** Per DEAD_ENDS rule 1, the unit
was measured before the mechanism, and the unit is too small — and for CDP2 the
mechanism itself is priced and loses outright on this machine.

Workload: favorite3 compiled harness (`~/favorite3-cpp/out`), this tree at
0f6e7436db7 (post merge-key campaign HEAD). Baseline sanity runs, unmodified
build: real-frame median **7.935 / 7.942 ms** (6,939 submits each, valid), vs
the campaign's ~7.99.

## The unit: what the host actually issues per episode tail

nsys 2026.4.1 census, one full run (traced arm 8.634 ms — CUPTI inflation as
expected; used for counting only). Real-frame window (frames 1391+, n=2078),
monotonic clocks aligned via `TARGET_INFO_SESSION_START_TIME.systemClockNs`:

- 329.5 kernel launches/frame total (199.6 `cuLaunchKernel` + 129.8
  `cuLaunchKernelEx`); 209.1 of them on the pass main stream.
- The **serial blended-episode tail** — the only population the merge rule
  (DEAD_ENDS 22) lets a chaining idea collect on — is `scan_reduce`,
  `scan_finish` (×2/episode), `fill_recs`, `sort_short`, `worklist`, `sort`,
  `quad_count_all`, `quad_fill_all`, `seg_count`, `seg_scatter`, `composite`:
  **50.36 launches/frame in 4.95 chains/frame** (≈10.2 per chain;
  `seg_prefix` never ran — the compact path was cold).
- Device-side exposure of those launches: 65.6% follow their predecessor at a
  gap ≤2 µs (median gap 1.7 µs) — the PDL-fed hidden case.

## Candidate 1 — CDP2 fire-and-forget tail chaining (idea F): REFUTED

**Credit side (park by the task's own bar).** Chaining converts 50.36 host
launches into 4.95, removing **45.4 host launches/frame**. At the settled
remove price of 0.78–0.81 µs/launch (rule 15) the ceiling is
**0.035–0.037 ms/frame** — a quarter of the 0.15 ms park bar, and 0.6% of
the 6.0 ms goal gap. Even pricing all 50.36 at the exposed 2 µs figure gives
0.10 ms. Parked on arithmetic alone.

**Mechanism side (refuted, not just parked).** Standalone microbenchmark
(`cdp2-microbench.cu` beside this note; sm_120, CUDA 12.8, chain of 512
1-block kernels, 20 reps, two runs agreeing to 1.5%):

| arm | per-link µs (best/mean) |
|---|---|
| host-issued back-to-back, one stream | **1.48–1.51 / 1.55–1.62** |
| CDP2 `cudaStreamTailLaunch` chain | **7.32–7.43 / 7.33–7.44** |
| CDP2 `cudaStreamFireAndForget` | 8.65 / 8.73 |
| host `cudaLaunchKernel` API CPU cost, stream busy | 1.43 |

A device-side tail launch costs **~5× a host-issued link** end to end on this
GPU. Converting the 45.4 links would *add* roughly 45.4 × (7.3 − 1.5) ≈
**+0.26 ms/frame of device critical path** to buy back ≤0.037 ms of host
issue. The idea-F note estimated its own ceiling at 0.1–0.3 ms; the measured
unit is at the bottom of that range and the mechanism price is negative.

## Candidate 2 — pre-issued predicated chains: REFUTED on its only admissible justification

The task rule: justify by launch-count reduction only, because the drain wait
is measured flat (campaign record: `CUDAVK_WAIT_SPIN_[BEFORE_]US` slopes
+0.02/+0.21, deferral ceiling ≈0.2 × 1.27 ms — closed). Pre-issuing the tail
with device-side early-exit predication issues the **same 50.36 launches**
(earlier, not fewer) and adds a repair arm — at the note's own ≤10-launch
budget that is up to **+49.5 launches/frame** at 0.78–2.0 µs each
(+0.04–0.10 ms), for a launch-count *increase*. Idle blocks are free (rule 6)
but launches are not. There is no launch reduction to collect; refuted.

## Candidate 3 — conditional graph nodes (CUDA 12.4+): stays behind DEAD_ENDS 1

Only reachable through idea A (stable device addresses — medium surgery, its
own probe unrun). The prize on *this* axis is candidate 1's launch credit
(≤0.037 ms) plus wait removal that is already closed at ≈0. That does not pay
for A's surgery, so entry 1's retry-if remains unmet and graphs stay closed.
If idea A is ever built for other reasons (temporal sizing, idea C), re-derive
this arithmetic before touching graphs.

## What this closes and what it does not

- Closes: host-launch-issue reduction on the serial episode tail as a route to
  6.0 ms on favorite3. The tail is 50 launches ≈ 0.04 ms of host issue; the
  frame is 7.94 ms. The latency floor of the tail is the launch pipeline and
  the kernels themselves, and CDP2's pipeline is slower than the host's.
- Does not close: idea C (temporal sizing, wait-shape change), idea A's own
  probe, or anything about the 42 fragment-shader (`main`) launches/frame on
  the main stream — those are real work, not issue overhead.
- Durable machine fact for any future device-launch idea: **on sm_120 /
  CUDA 12.8, CDP2 tail-launch ≈ 7.3–7.4 µs/link and fire-and-forget ≈
  8.7 µs/link vs ≈1.5 µs host-issued** — device-side launching starts 5×
  behind before any credit is counted.

Evidence: `/tmp/optB/` (base1/base2/trace_ts submit series, census1.nsys-rep +
sqlite, cdp2_bench.out, session1.log).
