# STEP B — where the frame goes at HEAD (e2fea470d04)

Instruments: `CUDAVK_PLAN_STATS=1` (host waits, host timers, no profiler) and
`CUDAVK_UPLOAD_STATS=1` (every small copy, clear and context sync by call
site, plus the launch counter). One run per instrument per capture, plus the
same two instruments with `CUDAVK_NO_OPAQUE_STREAMS=1` so the 2.7 ms of STEP A
can be attributed to a site. GPU idle before each run; all runs completed the
capture (3,022 / 2,994 submits). Per-frame figures divide by the driver's own
frame count: 1,511 on old, 1,497 on Crossroads.

Raw stderr: `/tmp/perf-audit/stepB/*/stderr`, GPU-busy series in
`/tmp/perf-audit/stepB-busy/*/busy.txt`.

**PERFORMANCE.md §5 was taken at `ba8891878df`, before iteration 28 item 4 and
before the fan-out default. Both drifts are quantified below.**

## B.1 The wait table, old capture

| wait | PERFORMANCE.md §5.2 | **HEAD, default** | HEAD, fan-out reverted |
|---|---|---|---|
| episode drain (`cp_renderer.c:8443`) | 8.663 ms / 9.88 waits / 0.877 ms | **5.989 ms / 9.88 / 0.6060 ms** | 8.639 ms / 9.88 / 0.8742 ms |
| peel checks (`cp_renderer.c:6689`) | 2.765 ms / 1.71 / 1.620 ms | **2.751 ms / 1.70 / 1.6135 ms** | 2.759 ms / 1.71 / 1.6171 ms |
| segment counters (`cp_renderer.c:6420`) | 0.995 ms / 4.17 / 0.239 ms | **0.982 ms / 4.17 / 0.2354 ms** | 0.998 ms / 4.17 / 0.2393 ms |
| descriptor uploads | 0.011 ms / 1.00 / 0.011 ms | **0.0125 ms / 1.00 / 0.0125 ms** | 0.0116 ms / 1.00 / 0.0116 ms |
| quad counters | — | 0 / 0 | 0 / 0 |
| **total blocked** | **12.435 ms / 16.76** | **9.734 ms / 16.76** | **12.407 ms / 16.76** |
| frame (this run) | 15.99 | **13.2124** | 15.7672 |

**Where the 2.55 ms went: all of it, and only it, came out of the episode
drain.**

- episode drain **8.639 → 5.989 ms/frame (-2.650)**, at an
  **unchanged 9.88 waits/frame**. The mean wait fell
  0.8742 → 0.6060 ms, −30.7%.
- peel checks moved -0.0081 ms/frame, segment counters -0.0162, descriptor
  uploads +0.0009. All three are inside their own run-to-run noise.
- The frame moved -2.5548 ms and the drain moved -2.650 ms. The
  attribution is complete to within 0.1 ms.

The reverted arm also **reproduces PERFORMANCE.md §5.2 almost exactly**
(12.408 against 12.435 ms, 16.76 waits against 16.76, every site within 0.02
ms), which is the strongest available check that the instrument still means
what the document says it means.

The mechanism is visible in the same output: an opaque episode on old averages
**12.42 segments and reaches 49**, 3,443 episodes have more than one segment,
and with the default on **3,443 episodes ran on the side streams** against 0 in
the reverted arm. The drain waits for the same episode; the episode now
finishes sooner because its segments overlap.

## B.2 The wait table, Crossroads

| wait | PERFORMANCE.md §5.2 | HEAD, default | HEAD, fan-out reverted |
|---|---|---|---|
| episode drain | — | **2.129 ms / 5.57 / 0.3824 ms** | 2.153 ms / 5.57 / 0.3868 ms |
| peel checks | — | **0 / 0** | 0 / 0 |
| segment counters | — | **0.428 ms / 1.69 / 0.2525 ms** | 0.432 ms / 1.69 / 0.2552 ms |
| descriptor uploads | — | **0.0065 ms / 1.00** | 0.0063 ms / 1.00 |
| **total blocked** | **2.580 ms / 8.26** | **2.563 ms / 8.26** | **2.591 ms / 8.26** |
| frame (this run) | — | **5.8633** | 5.8843 |

Crossroads still matches its documented 2.580 ms over 8.26 waits, and the
fan-out moves it by -0.0285 ms/frame — the same near-zero result STEP A
measured on the frame. **Crossroads runs no peel checks at all**, which is why
lead #1 has to be justified on the old capture alone.

## B.3 Ping-pong shape, re-derived

Untraced GPU busy, `nvidia-smi utilization.gpu` at 10 Hz, no profiler, steady
window, median with the interquartile range (nvidia-smi repeats a value for
several polls, so the median is the honest statistic here):

| capture / arm | busy median | IQR | frame ms | blocked ms | issuing = frame − blocked | device idle = frame × (1 − busy) |
|---|---:|---|---:|---:|---:|---:|
| old, default | **72%** | 71–73 | 13.1410 | 9.734 | 3.478 | **3.679** |
| old, reverted | 75% | 73–77 | 15.7871 | 12.407 | 3.360 | 3.947 |
| PERFORMANCE.md §5.2 | 74% | — | 15.99 | 12.44 | 3.55 | 4.16 |
| Crossroads, default | **59%** | see note | 5.8334 | 2.563 | 3.300 | 2.392 |

Note: Crossroads' replay is only 11 s, so its busy series has ~90 usable polls
and its ramp is a larger share; 59% is its median and equals the documented
figure exactly.

**The shape §5.2 describes is unchanged; only its size shrank.**

- Host blocked is **73.7%** of the frame on old (was 77.8%).
- Host issuing is **3.478 ms/frame** (was 3.55) — the host's own issue
  time barely moved, as expected: the fan-out changed what the device does
  with the work, not how much host work there is.
- Device idle is **3.679 ms/frame** (was 4.16) and is still almost exactly
  the host's issue time (3.478 ms). The driver still cannot issue frame
  N+1 while frame N runs, and it still blocks 16.76 times a frame.

## B.4 Launch and operation counts, and the drift PERFORMANCE.md predicts

§5.1's profile counted **1,389 kernels/frame** on old at `ba8891878df` and the
text says today's counts are **91.1/frame lower**, i.e. an implied ~1,298.

| counter | old | Crossroads |
|---|---:|---:|
| kernel launches / frame (driver's own counter) | **1314.2** | **346.0** |
| episodes closed / frame | 15.50 | 9.56 |
| episode drains / frame | 9.88 | 5.57 |
| launches / episode | 84.8 | 36.2 |
| memcpy / frame (htod async + htod sync + dtoh) | 438.4 | 118.2 |
| memset-async / frame | 235.4 | 100.8 |
| `cuCtxSynchronize` / frame | 2.34 | 1.59 |
| blocking `cuMemcpyHtoD` (descriptor arena) / frame | 1.01 | 1.00 |

**Drift against §5.1: −74.8 launches/frame on old, against the −91.1 the
document predicts — so the count is +16.2/frame away from the document's own
implied figure**, about 1.2%. On Crossroads it is 346.0 against §5.1's 439
kernels/frame, −93.0/frame. The two counters are not identical
instruments — §5.1 comes from a device trace and this one is the driver's
`CP_LAUNCH` counter — so read the 1.2% as "no unexplained drift", not as a
discrepancy.

Operation counts are **bit-identical between the fan-out arms** on both
captures (memcpy, memset, ctxsync, launches, episodes all equal to the digit),
which is the check that the fan-out is a scheduling change and not a work
change.

Top small-operation sites, old capture, per frame:

| site | kind | /frame |
|---|---|---:|
| `cp_renderer.c:464` | htod-async | 423.31 |
| `cp_renderer.c:3767` | memset-async | 82.76 |
| `cp_renderer.c:8831` | memset-async | 22.99 |
| `cp_renderer.c:8836` | memset-async | 22.99 |
| `cp_renderer.c:4226` | memset-async | 21.29 |
| `cp_renderer.c:4761` | memset-async | 20.00 |
| `cp_renderer.c:6506` | memset-async | 18.43 |
| `cp_renderer.c:8446` | dtoh | 9.88 |
| `cp_renderer.c:5351` | memset-async | 9.42 |
| `cp_renderer.c:8539` | memset-async | 9.20 |
| `cp_renderer.c:2481` | memset-async | 7.51 |
| `cp_renderer.c:5825` | memset-async | 4.44 |

The drain's own read-back is visible as `cp_renderer.c:8446` dtoh at
9.88/frame — exactly the 9.88 episode drains/frame — and the
segment-counter read at `cp_renderer.c:6427` at 4.17/frame, exactly the
4.17 segment waits. The instruments agree with each other.

## B.5 What this does to the §6 leads

| lead | §6 estimate | status at HEAD |
|---|---|---|
| 1. peel checks | 0.20–0.50 ms | **intact and relatively more valuable.** 2.751 ms/frame is unchanged in absolute terms, but the frame is 2.55 ms shorter, so it is now **20.8% of the frame** instead of 17.3%. It is still the most expensive single wait (1.6135 ms each). Old capture only; Crossroads runs none. |
| 2. widen the `bounded` fast path | 0.10–0.25 ms | **intact.** 0.982 ms/frame over 4.17 waits, mean 0.2354 ms — within 0.02 ms of the documented figures. |
| 3. fuse `cp_fs_writeback` | ~0.04 ms | untouched by this measurement; still parked on the correctness failure. |
| 4. let the host run past the episode drain | value unknown | **the fan-out has already collected 31% of this site** (8.639 → 5.989 ms/frame) without deferring anything, by making the wait shorter rather than removing it. 5.989 ms/frame over 9.88 waits remains, and it is still the largest single site. P1/P2 (STEP C) still gate it. |

The 2.55 ms did **not** come out of the two leads. Both keep their measured
basis and their estimates.

## Blockers

One run was contaminated and was discarded and redone: the first GPU-busy
attempt started while the last two STEP B stats runs were still replaying, so
two processes shared the card. It was killed, the card was confirmed empty,
and all three busy runs in this file are from the clean re-run. No other
blocker.

## B.6 Instrument validity, exact code sites, and one price

**Instrument validity.** Everything in this file comes from `CUDAVK_PLAN_STATS`
(host `clock_gettime` timers around `cuStreamSynchronize`) and
`CUDAVK_UPLOAD_STATS` (counters), plus `nvidia-smi` with nothing attached.
**`CUDAVK_DEBUG_TIME` and `CUDAVK_ABUFFER_TIMING` were not used anywhere in
STEP A, B or C.** Those two take CUDA events, and since 7f38d2a9b65 spread work
over eight streams an event pair on one stream no longer bounds the episode, so
their numbers would be invalid at the current default. The two instruments used
here are unaffected by stream count: a host timer measures the calling thread
and a counter counts calls.

**The two open leads, at their HEAD line numbers.** Both are still
unimplemented in the tree I measured (`git show HEAD:src/cudavk/cp_renderer.c`):

| lead | code at HEAD | the wait it removes | measured here |
|---|---|---|---|
| clip-rectangle bound (§6 item 2) | the `bounded` predicate, **6402–6413**, still bounds by `(size_t)ab->nblocks * rast_num_triangles` — whole framebuffer, no clip rectangle; when it fails, the `else if (ab->composite)` branch drains at **6420** | `wait_seg` | **0.982 ms/frame over 4.17 waits, mean 0.2354 ms** (old); 0.428 ms over 1.69 (Crossroads) |
| peel predication (§6 item 1) | host store `*(volatile uint32_t *)cp->peel_any = 0` at **6512**, drain at **6689**, host read at **6692**, interval doubling at **6695–6696** | `wait_peel` | **2.751 ms/frame over 1.70 waits, mean 1.6135 ms** (old); **zero on Crossroads** |

These are separate counters in separate branches, so the two leads are sized
independently and neither number includes the other.

**The cross-stream hand-off price, from counters already taken** (no extra
run). b3f716bd5f5 measured a hand-off — the gate in and the join out — at about
**32 µs per episode**, and PERFORMANCE.md's price table does not carry it. What
this session's counters can say about it:

- old: 3443 episodes ran on the side streams over 1511 frames = **2.28 fanned
  episodes/frame**. At 32 µs each that is **≈0.073 ms/frame of hand-off cost**,
  against the **2.650 ms/frame** of drain time the fan-out removes — the
  hand-off costs about 2.8% of what it buys, so the gross saving is
  ≈2.723 ms/frame.
- Crossroads: 1468 fanned episodes over 1497 frames = **0.98/frame**, ≈0.031 ms/frame
  of hand-off against a measured net win of 0.024 ms/frame of drain time — i.e. on
  this capture the price is roughly a third of the gain, which is exactly why
  the capture with 1.71 segments per episode gains almost nothing.

This is arithmetic on the 32 µs figure, not an independent measurement of it.
It is offered as the consistency check that the published price predicts the
right asymmetry between the two captures.

**The blended path already had this.** 0458a51e9a4 fanned blended segments out
earlier; 20611f5b131 gave the opaque path what the blended path already had.
The attribution in B.1 agrees: the delta is confined to the episode drain, and
the opaque-episode counters are the ones that change arm to arm
(3443 episodes on side streams against 0).
