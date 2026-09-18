# Performance analysis session, 2026-09-02 — raw reports

Read-only session: nothing was built, nothing ran on a GPU, no driver source
changed. The consolidated result is `docs/cudavk/notes/PERF_ANALYSIS_2026-09-02.md`;
this directory holds the reports it cites, with the scripts that produced every
number so they can be re-run against the traces.

Traces used (all on the RTX 5090 host, not persistent — copy them if they
matter): `/tmp/postdead/trace.sqlite` (light band, frames 1391-1802, commit
`f3caa35e4ce`, pre host-pin), `/tmp/postpin-rtx2/trace.sqlite` (heavy band,
frames ~2448-3462, post host-pin), `/tmp/postpin-rtx/trace.sqlite` (post-pin,
UVM rows to frame 3268), `/tmp/favorite3-perf-current/` (perf, no CUPTI),
`/tmp/ncu-rtx/v2.csv` and `/tmp/ncu-b200-v2.csv` (paired ncu, 45 launches),
`/tmp/pin-rtx-ab/` (untraced submit timestamps of the host-pin A/B).

| directory | question it answers |
|---|---|
| `frame-boundary/` | what the application and the driver do between the two submits and across the frame boundary; the post-pin boundary interval (0.55 ms) and who owns it |
| `host-visible-memory/` | where host-visible memory lives (all managed), the readback-fill capture artefact, residual UVM migration post-pin, the 99,999-row UVM buffer-loss reading of DEAD_ENDS 37 |
| `host-launch-audit/` | per-launch host cost without CUPTI (1.36-1.70 µs, all in libcuda), the per-frame API census, the argument-block copies and the kernel-parameter lead |
| `kernel-chain/` | the kernel sequence per batch/episode type, per-kernel census, the main-stream critical path (main ∩ side = 0), the failed-append back-out, the segment-0 gate, the discard retry loop, fullscreen chains |
| `host-sync/` | every host blocking site, post-sync bubbles (0.385 ms/frame in total), the `cuLaunchHostFunc` timestamp stalls (0.26 ms/frame), the frame-boundary decomposition |
| `arith-vs-latency/` | rooflines per pool against measured durations (real work 0.3-0.7 ms/frame), duration-vs-threads tables, memset volume, the B200 cycle interpretation, a device floor for a scope-granularity design |
| `b200/` | every B200 number in the tree, the paired ncu re-analysis (0.742× cycles, 148 SMs), the host-pacing explanation of the 1.6× slowdown, the price-table microbenchmarks to run |
| `closure-audit/` | adversarial audit of the closure logic; `trace_audit.py` is the dependency-respecting what-if replay (validated to 15 µs) behind the MODEL figures |
| `design-device-autonomy/` | the one design that completed: device-decided episodes, device timestamps, chain graphs; the oracle-replay first slice |
| `design-incremental-sum/` | heavy-band counts for the cheap leads (failed appends 10.1/frame, discard re-passes 4.0, fan-outs 1.96, fullscreen chains 8.5, submit head 178 µs) |
| `design-b200-app/` | per-pass draw/bind/run census of the harness by frame range |
| `verify2-*/` | the adversarial verification pass over the ranked leads, one review per lead |

Three designs (granularity, incremental sum, B200-first) and the synthesis
agent did not complete; the analysis note was written from the evidence
directly and the verification pass was run separately.
