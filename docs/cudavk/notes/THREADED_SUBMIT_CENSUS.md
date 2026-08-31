# Threaded-submit completion census

This is a diagnostic for the fully threaded submit experiment. It is not a
shipping timer and it does not make a performance claim.

## What is timed

With asynchronous submit, an external `submit_shim` timestamp taken before a
`vkQueueSubmit` call measures enqueue pace. The call can return while Mesa's
submit worker still owns the queued record. That timestamp therefore cannot
show when CUDA finishes the work. Treating it as frame time would credit work
that is only queued.

The census timestamps three internal points with `CLOCK_MONOTONIC_RAW`:

- `issue_begin_ns`: the Mesa submit worker enters cudavk;
- `issue_end_ns`: cudavk has appended the CUDA event to its completion queue;
- `complete_ns`: the completion worker's `cuEventSynchronize` returned.

Only `complete_ns` can support the experiment's completion-throughput
question. It is still diagnostic timing. The census adds clock reads, record
writes, a runtime queue-depth scan, and fixed storage when enabled.

## Record spaces and target selection

Do not derive internal frame parity from record ordinals. The favorite3 replay
has 6,939 external shim submit events but 6,947 internal command-bearing
completions. Virtual-swapchain helper submissions account for the difference,
and one helper occurs before the target window. Marker records with zero
command buffers are a separate class and the analyzer filters them.

The target must therefore be supplied explicitly:

```sh
python3 src/cudavk/tests/cp_submit_census_analyze.py RUN.log \
  --internal-target-start ABSOLUTE_RECORD_ORDINAL \
  --target-frame-count FRAMES
```

`--internal-target-start` is the absolute `internal_ordinal` printed in the
record, not an ordinal after filtering. The selected record must itself be
command-bearing. Signal-only runtime markers can be interspersed and can vary
with runtime mode, so a command-bearing position must first be calibrated to
an absolute record ordinal from an exact structural population. The selected
internal range begins at that record and ends at the final command-bearing
completion. The reported diagnostic frame interval is exactly:

```
(last_complete_ns - first_complete_ns) / (FRAMES - 1)
```

It is not a median and it does not guess an odd/even submit phase. The JSON
also reports how many internal command-bearing records the range contains, so
that count is not confused with the external submit or frame count.

The optional shim input accepts either `timestamp_ns submit_index` or:

```
CUDAVK_SUBMIT_SHIM_TIMESTAMP timestamp_ns=N submit_index=N
```

Pass `--external-target-start` with it. The analyzer requires the exact
external range of `2 * (FRAMES - 1) + 1` complete submits. A process killed
while writing the shim can leave one numeric timestamp token at EOF with no
newline; that narrow final fragment is ignored. Malformed complete lines and
any malformed earlier line are rejected. Shim results are labeled enqueue
diagnostics, not completion timing.

For the recorded favorite3 target, external submit 2,782 maps to absolute
command-bearing position 2,783 because one of the eight virtual-swapchain
helpers precedes it. That is observed data, not a parity rule. The measured
runs each had exactly 6,947 command-bearing records and 1,663 signal-only
markers. Position 2,783 therefore mapped to absolute record ordinal 3,114 in
all four arms. The target contains 2,079 frames and 4,164 command-bearing
records through the end of the run.

## Measured favorite3 outcome

The alternating `control, threaded, control, threaded` census is in
`/tmp/threaded-submit-f3-census`. Every arm ended with the capture's expected
post-output `rc=139`, 6,939 complete shim timestamps, stdout SHA-256
`e3f24a1dcdc8568be217d249e480623958e2621b3d4f056ae4c44ad20d08da49`,
8,610 complete census records, no overflow, and the exact structural counts
above.

| arm | Mesa thread | completion ms/frame | issue-begin ms/frame | enqueue ms/frame | target depth p95/max |
|---|---:|---:|---:|---:|---:|
| c1 | 0 | 8.549611 | 8.552320 | 8.539533 | 0 / 0 |
| t1 | 1 | 8.264747 | 8.267166 | 8.256351 | 1 / 2 |
| c2 | 0 | 8.301126 | 8.303556 | 8.290263 | 0 / 0 |
| t2 | 1 | 8.333623 | 8.336444 | 8.326683 | 1 / 2 |

The completion medians are **8.425368 ms control** and **8.299185 ms
threaded**, a diagnostic improvement of only **0.126184 ms/frame**. The issue
and enqueue deltas agree (0.126132 and 0.123381 ms/frame), while the completion
metric remains the decision metric. Target queue depth stayed at p95 1 and
maximum 2, so the result is not hiding unbounded run-ahead.

Before the replay, the existing suite passed **79/79** with
`MESA_VK_ENABLE_SUBMIT_THREAD` absent, `=0`, and `=1`. The delayed
wait-before-signal timeline regression also passed as a direct test in all
three modes. Two fault-injection gates were made mode-aware: immediate submit
reports `VK_ERROR_DEVICE_LOST` from `vkQueueSubmit`, while fully threaded submit
has already enqueued and reports the same terminal failure from
`vkDeviceWaitIdle`. Static build/audit gates and all 11 analyzer tests passed.

The 0.126184 ms result is far below the 0.500 ms/frame admission gate. Fully
threaded Mesa submit is therefore refuted for production work on favorite3;
favorite2 measurement, bounded-depth sweeps, full-frame dump comparison, and a
shipping merge are not justified.

## Validity and admission

The analyzer rejects a run unless records are initialized, complete,
contiguous, successful, and monotonic, and the summary reports a valid,
non-overflowed, fully complete census. Queue depth is capped at 64. If
`depth_capped=1`, its reported values are lower bounds rather than a complete
distribution.

A threaded-submit candidate does not enter production work unless
completion-aware timing shows at least **0.5 ms/frame on both captures**. That
is only the admission threshold for the next correctness and measurement
round. It is not evidence by itself and must not be reported as a performance
result.
