# Verify LEAD B — replace `cuLaunchHostFunc` timestamp queries with a device timestamp

Verdict: **OVERSTATED**. The mechanism is real (the two head-of-submit host funcs
stall the stream for 36–54 µs median, MEASURED) but the claimed ceiling of
0.26 ms/frame is a measurement artefact: it counts one idle interval twice and
the interval is mostly host issue time. The critical-path effect is
**~0.03 ms/frame median (≤0.06 mean) on RTX**, and the sub-claim that the
frame-end host func "delays the readback copy and is 1:1 by construction" is
**refuted** by the trace (the copy starts 2.1 µs after its own API returns).

Scripts: `hf.py` (decomposition per host-func cluster, reproduces the report's
0.263), `hf2.py` (backlog propagation to the first kernel and beyond). Both are
read-only SQL over `/tmp/postdead/trace.sqlite` (light band, pre-pin, 413 frames)
and `/tmp/postpin-rtx2/trace.sqlite` (heavy band frames 2448–3462, post-pin,
1014 frames). Labels: MEASURED / ESTIMATED / MODEL / HYPOTHESIS.

## 1. Precedent

- **No DEAD_ENDS entry covers `cuLaunchHostFunc`.** The word appears nowhere in
  DEAD_ENDS.md, PERFORMANCE.md or TODO.md (grep). The eight-site block census
  (PERFORMANCE §5.2b) never saw these nodes because they block the *stream*, not
  the host. The host-sync report is right that this is "found, not in the docs".
- **DEAD_ENDS 17 (destruction drains)** is the entry the report names as
  closest; the "why different" argument ("97% have queued work behind them")
  holds for the head pair (the 16 layer copies and the first kernel queue
  behind it) — but that is precisely the population whose gap is host-issue-
  bound, see §2.
- **DEAD_ENDS 29 is the price precedent that applies**: a 1.65 ms/frame blocked
  quantity converted into +0.037 ms/frame, undecidable against a 0.044/0.092 ms
  pair spread. The corrected ceiling here (0.03–0.06 ms/frame) is the *same
  size as DEAD_ENDS 29's unresolvable result*, i.e. below the session spread
  the programme has measured (0.119 ms on old, PERF §6 item 4; pin A/B controls
  7.521 / 7.594 / 8.014, frame-boundary report §5).
- **PERFORMANCE §4 prices** that the replacement must pay: the head replacement
  device ops (memset for reset + 1-block kernel for TOP) are *exposed* launches
  on an idle device at the head of the submit → 1.974 µs each, not 0.78. The
  frame-end kernel is behind the last render kernel and hidden. Net replacement
  cost ≈ 4 µs/frame (MODEL).
- The merge rule (DE22) is not engaged: nothing is merged.

## 2. Arithmetic

### 2.1 What the 0.263 ms/frame actually is (MEASURED, reproduced exactly)

`hf.py` reproduces the report's number bit-for-bit on postdead: n 1238, 38
crossing, gap med 77.5 / p90 155.7 / mean 90.4 µs, sum **0.263 ms/frame**.

The 3.00 host funcs/frame are two clusters (MEASURED, 413 + 412 of each):

| cluster | position (prev API → host funcs → next API) | n/frame |
|---|---|---:|
| **head pair** (ResetQueryPool + WriteTimestamp TOP) | `cuMemcpyHtoD_v2` (desc arena) → HF, HF → `cuMemcpy2DAsync_v2` (32x32x16 layer copies) | 1.00 |
| **frame-end single** (WriteTimestamp BOTTOM) | `cuLaunchKernel` (last render kernel) → HF → `cuEventCreate` (done event); next device op is the readback copy in submit B | 1.00 |

The report's gap is `first kernel start ≥ HF API end` − `last device end before HF
API end`. For the head pair this is measured **per host func**, so the same
interval is counted twice (826 entries at ~99 µs median = 0.130 × 2 = 0.260
ms/frame); the frame-end singles contribute 0.003 (median gap 3.1 µs — the
"next kernel" it finds is a kernel launched *before* the host func that was
still queued). **0.260 + 0.003 = 0.263.** So the entire figure is the head-of-
submit interval, twice.

### 2.2 What is in the head interval (MEASURED, postdead / postpin)

| component | postdead med / mean | postpin med / mean |
|---|---:|---:|
| host issue from HF API end to first kernel launch API start (16 × `cuMemcpy2DAsync`+`cuEventRecord`, first launch) | ≈ 60 µs (timeline: 112.5 → 173.3 in frame-boundary §4) | — |
| stream stall seen by the first device op after the pair: `dev_start − max(api_end, prev_dev_end)` | **53.6 / 81.3 µs** | **36.1 / 47.4 µs** |
| reference: same quantity for `cuMemcpy2DAsync` not after a host func, device idle | 14.5 med | 25.7 med |
| **backlog at the first kernel: `dev_start − api_end`** | **34.8 / 59.2 µs** (p90 109.6) | **29.2 / 37.2 µs** (p90 70.4) |

So the two callbacks hold the stream ~36–54 µs (18–27 µs each — ordinary
host-func dispatch latency); by the time the host has issued the 16 copies and
the first launch, the device is **~30–35 µs behind** it. That backlog is the
only part of the "gap" that a device-side timestamp could recover.

### 2.3 Does the head backlog reach the frame? (MEASURED structure, HYPOTHESIS conversion)

`hf2.py`: after the first kernel the backlog *grows* — postdead medians
1: 34.8, 2: 103, 4: 163, 40: 262 µs, and at the last kernel before the first
host sync (538 µs after the pair, 106 launches) it is 1,156 µs (postpin:
1,405 µs later, 399 launches, 2,907 µs). The device is the bottleneck from the
first kernel to the first sync, so there is no host slack to absorb a 35 µs
head shift; it propagates to the sync return (the +1.02 slope measured at the
drain site, TODO item 1, is the same mechanism). Conversion ≈ 1 is therefore a
fair HYPOTHESIS for the *head* backlog — which makes the ceiling the backlog
itself, not the gap.

### 2.4 The frame-end host func (MEASURED — refutes the "1:1 by construction" claim)

Next device op after the BOTTOM host func is the 3,686,400 B readback copy
(submit B). `dev_start − api_end` of that copy: **2.1 µs median** on both
traces (p90 7.7 / 11.6, mean 3.4 / 10.4 µs → 0.003 / 0.010 ms/frame). The host
takes 78 µs (postdead) / 58 µs (postpin) to get from the host func to issuing
the copy (submit A tail, app work, submit B translation), and the callback
finishes inside that. The report's own §5 table already contradicted itself:
a "77 µs" stall inside a 65.6 µs `last kernel → submit B` interval, with the
copy starting 9.1 µs after ts(B).

### 2.5 Corrected ceiling

RTX (both bands): head backlog 29–35 µs median, 37–59 µs mean, minus ~4 µs of
exposed replacement launches (MODEL) → **≈0.03 ms/frame median, ≤0.06 mean**;
frame-end adds ≤0.01. Whole-window hot-tail median will sit nearer the 0.03.
Untraced it may differ either way (CUPTI inflates host issue ~1.28×, and the
callback runs on a runtime thread CUPTI also instruments), but the untraced
budget is bounded by the untraced head-to-first-kernel interval, which the
frame-boundary report estimates at "perhaps 0.1 ms" *including* 32 copies and
the desc upload.

B200: **no B200 trace exists** (b200 report has no host-func or head-of-submit
row). The mechanism is host-func dispatch latency on the host, the same per
frame (3 nodes), so ESTIMATED ≈ 0.03–0.06 ms/frame by transfer; it cannot
exceed the B200's own head-to-first-kernel interval, which is unmeasured.

Against the 5.0 ms goal (gap 1.68 ms RTX, 5.8 ms B200) this is 2–4% of the gap
and below the 0.5 ms admission bar by an order of magnitude; it is also inside
the 0.04–0.12 ms A/B spread, so it would be *unresolvable* the way DEAD_ENDS 29
was.

## 3. Feasibility

Code sites read: `cpvk_execute_query` cpvk_cmd.c:4830-4851,
`cpvk_query_complete` :4743-4771, `cpvk_GetQueryPoolResults` :4669-4735,
`cpvk_execute_query_copy` :4786-4828, `cpvk_execute_order_point` :4253-4257,
`cp_render_scope_end` cp_renderer.c:9560-9568, submit done-event
cpvk_device_memory.c:371-413 and worker :58-83, `timestampPeriod = 1.0f`
cpvk_device.c:367, `timestampValidBits = 0` :675.

- **The order point is not a cost here.** At the head nothing is open
  (`cp_pass_finish` returns on `nsegs == 0`, :8703); at the frame end the last
  `END_RENDER` op already ran `cp_batch_flush_why` + `cp_pass_finish`
  (cp_renderer.c:9561-9565). The query's order point forces no episode on
  favorite3; the 8/frame `vkCmdPipelineBarrier` order points are a different
  lead.
- **Timestamp semantics are already void.** The driver reports
  `timestampValidBits = 0` and `timestampComputeAndGraphics = false`, so by the
  spec the app's `vkCmdWriteTimestamp` on this queue is invalid usage and the
  values are undefined; `timestampPeriod = 1.0` with `%globaltimer` (ns) would
  be consistent, but nothing depends on it. The replay discards the results.
- **Availability must stay ordered, not immediate.** `tests/cpvk_query_async.c`
  asserts a query recorded after a `vkCmdWaitEvents` is `VK_NOT_READY` until
  the event is set — so the naive "mark available at issue" is *not* allowed;
  availability-by-submit-done-event (the `pending->done` event the worker
  already synchronises, cpvk_device_memory.c:61) satisfies it because the
  event-wait host func precedes `done` in stream order. `cpvk_query_lifetime.c`
  (pool retained through submit) must also pass.
- **`VK_QUERY_RESULT_WAIT_BIT`**: the app calls with flags = 1 (WAIT) after
  `vkWaitForFences` on the previous render fence, so the `usleep(5000)` loop is
  dead code today and stays dead under done-event availability (the fence and
  the done event are the same completion). Exactness is required only in the
  sense the test above encodes.
- **`vkCmdCopyQueryPoolResults`** (`cpvk_execute_query_copy`) drains the stream
  and reads `pool->results` on the host; under a device buffer it becomes a
  DtoD copy plus availability word — small, but it is a second path to change.
- Hidden work: the device query buffer, its reset memset, reading it back at
  `vkGetQueryPoolResults`, and keeping the worker publishing availability with
  the right memory ordering. No ARCHITECTURE §8 invariant is touched (no
  helper-lane, ordering, arena or ABI change). Blast radius: query semantics
  only. Effort: probe half a day; full design 1–2 days plus the 74-test suite —
  not justified at ≤0.06.
- **Cheaper decisive probe than building the kernel**: in a worktree, make
  `cpvk_execute_query` for TIMESTAMP/reset ops write `clock_gettime` (or 0) into
  `pool->results` at issue and publish availability from the completion worker
  on the submit's `done` event — zero device ops, zero host funcs — then A/B by
  the submit-shim medians (WORKFLOW §4.2, ≥3 alternating rounds, both bands).
  Expected: ≤0.03–0.06 ms/frame, inside the control spread. If it does not
  resolve, record it beside DEAD_ENDS 29 and stop; the `%globaltimer` design
  cannot beat a zero-op probe.

## 4. Must fix in the lead before it is quoted again

1. Count the head pair once: the 0.263 double-counts one interval.
2. Use `first-kernel dev_start − api_end` (29–35 µs) as the ceiling, not the
   `last device end → next kernel start` gap (99 µs), which is ~60% host issue.
3. Drop "the frame-end one is 1:1 by construction": the readback copy starts
   2.1 µs after its API on both traces.
4. State that the order point is a no-op at all three positions on favorite3.
5. State the noise floor: the corrected ceiling equals DEAD_ENDS 29's result.

## 5. Verdict

OVERSTATED. Real ~36–54 µs stream stall at the head pair (MEASURED), reaching
the critical path as a ~30–35 µs backlog at the first kernel (MEASURED,
conversion ≈1 HYPOTHESIS); frame-end node ≈0 (MEASURED). Corrected ceiling
≈0.03 ms/frame RTX (≤0.06 mean), B200 ESTIMATED ≈0.03–0.06 with no trace to
back it — 4–9× below the claim and below the measurement floor.
