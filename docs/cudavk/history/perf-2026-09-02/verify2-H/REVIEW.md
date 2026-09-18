# Lead H — segment-level CUDA graphs with cuGraphExecUpdate: adversarial review

Read-only, 2026-09-02. Scratch: this directory (`segments.py`, `postpin_segments.log`,
`postpin_segments_submitsplit.log`, `ta_api.py`, `ta_api_postpin.log`). Nothing built, nothing run on the GPU,
nothing edited in the repository.

Labels: MEASURED (read from a trace table or a document), ESTIMATED (my arithmetic on measured inputs),
MODEL (discrete-event arithmetic on trace timestamps), HYPOTHESIS (no measurement).

## Verdict: REFUTED as stated (segment unit); a sub-segment unit survives only as an unmeasured 0-0.3 ms item

The lead's topology census is right and reproduces. Its ceiling does not, for a structural reason the trace
measures directly: inside a submit-bounded segment the host is **already pipelined against the device** — the
device starts the segment's first op ~2 µs after the first launch and the host keeps issuing for a **median
111 µs** afterwards (4.27 ms/frame summed, MEASURED, heavy band), while the device waits on the host for only
**0.143 ms/frame** (MEASURED). A graph per segment must be fully parameterised before `cuGraphLaunch`, so it
moves that whole host burst in front of the device chain. The deferral costs more than the entire host issue-API
budget it is meant to shrink (1.70 ms/frame de-inflated): the model lands at **+2.3 ms/frame slower** at the
lead's own per-node price. Separately, the "host 0.5-0.6" half of the ceiling is a misattribution of the
closure audit's `host x0.5` sensitivity: re-running that replay with only the issue/event API durations
scaled (the quantity graphs change) gives **-0.23 ms at x0.5 and -0.37 ms at x0** (MODEL, CUPTI-inflated
durations, so smaller in truth) — and that is with pipelined issue, which the segment graph forfeits.

## 1. PRECEDENT

**DEAD_ENDS 1** (`docs/cudavk/DEAD_ENDS.md:100-157`) closed graphs as an exact-argument cache at three units
(raster tail, direct batch, opaque episode). Its "why different" argument here is fair as far as it goes:
`cuGraphExecUpdate` re-parameterises a template and needs topology recurrence, not recipe recurrence, and the
retry-if ("stable command-owned execution storage with fixed device addresses") is a cache condition that a
template does not need. `cuda_leads.md` §4 (`docs/cudavk/history/perf-2026-08-27/cuda_leads.md:288-383`) says
the same and asked for exactly the census the closure audit took. So entry 1 does not refute the mechanism.

What entry 1 *does* price and the lead inherits: instantiate 14.8-15.1 µs and 44-46 KiB per graph+exec
(MEASURED, CUDA 12.8, this machine), and "the Driver API is not the blocker — the unit is wrong". The unit is
again the problem, from the other side: iteration 19's per-batch unit was too small to recur by key; the
segment unit recurs by topology but is too large to pipeline (§2.3 below).

**DEAD_ENDS 25** (`:1327-1376`): `cuLaunchKernel` costs 1.43 µs of CPU while the device link is 1.5 µs end to
end, "the host is not the bottleneck it looks like from an API trace". That is the same finding as §2.2 here.
Entry 25 also records conditional graphs at 10.27 µs per launch (EPISODES session 11) — a graph launch is not
free on the device side either.

**PERFORMANCE §4** (`docs/cudavk/PERFORMANCE.md:245-372`): the two launch prices. Removing a real chain launch
is worth 0.78-0.81 µs because 74.9% of chain launches have their front end hidden behind the previous kernel.
A graph does not remove launches; it changes who issues them. The relevant row is the PDL one: 0.44-3.02 µs
per converted link, which a graph must preserve through programmatic edges or lose (199 lost PDL links cost
0.09-0.17 ms in item4_merge_results, per the closure audit §2.6).

**Rules** (`DEAD_ENDS.md:2062-2151`): rule 1 (measure the unit before the mechanism) is the one this lead is
about to break — the census counted kernel names, not the unit's cost structure. Rule 15 (a launch has two
prices) applies: the lead quotes 1.36-1.70 µs host per launch as the saving; that is the CPU cost, not the
frame cost, and the frame cost of a hidden launch is what §2.2 measures.

**TODO 'Performance leads'** item 4 keeps "reopening CUDA graphs on 12.8 specifically" open with the caveat
"needs a re-traced idle-gap budget at HEAD before it is worth an iteration". This review is that budget.

## 2. ARITHMETIC

### 2.1 The census reproduces, and a fuller signature still recurs (MEASURED, `/tmp/postpin-rtx2/trace.sqlite`, 1024 frames, 8.379 ms/frame wall)

`segments.py`: segments = main-thread API rows between consecutive blocking returns (`cuStreamSynchronize`,
`cuCtxSynchronize`, `cuMemcpyDtoH_v2`) **and vkQueueSubmit boundaries** (submit shim `ts.txt`, converted with
`TARGET_INFO_SESSION_START_TIME.systemClockNs`). Device ops joined by `correlationId`.

| signature | segments (non-empty) | distinct | seen in prev frame | same position | ops in once-only signatures |
|---|---:|---:|---:|---:|---:|
| kernel names only (closure-audit definition) | 11,663 (11.4/frame) | 412 | 92.8% | 80.3% | 82.7 ops/frame |
| **full node signature**: ordered (kind, name, stream) over kernels + memcpy + memset + host-func, plus event record/wait streams | 11,663 | **525** | 91.6% | 79.1% | **124.4 ops/frame** (11% of ops) |

Per frame: 719.6 kernels, 261.0 HtoD arg copies, 114.1 memsets, 29.3 2D/3D copies, 3.0 host-func nodes,
411.1 event record/wait calls; 1,124 device ops in 11.4 non-empty segments; ops/segment p50 25, p90 221;
kernels/segment p50 16. So the recurrence premise holds with a topology that includes copies, memsets and
stream placement — the census was not wrong. 2% of segments carry a topology seen once in 1,024 frames; that is
~0.24 instantiations/frame (~15 µs each plus node adds), negligible in time, ~10 MB/1,000 frames in memory.

### 2.2 What the host issue API is worth to the frame, with pipelined issue (MODEL, closure-audit replay)

`ta_api.py` is the closure audit's validated `trace_audit.py` (E0 error p50 15 µs) with the host scale split:
`host_scale` on API durations only, `gap_scale` on inter-call gaps (planning, app recording).

| what-if (postpin heavy band, median frame 8.155) | Δ ms/frame |
|---|---:|
| host gaps x0.5 AND API x0.5 (the closure audit's "host twice as fast", quoted by the lead as 0.81) | -0.813 |
| **inter-call gaps x0.5, API unchanged** | **-0.563** |
| **issue + event API durations x0.5, gaps unchanged** | **-0.234** |
| issue + event API x0.25 | -0.329 |
| issue + event API x0 (every launch/copy/memset/event call free) | -0.367 |

The API durations in the trace are CUPTI-inflated (host-launch-audit §4: +28-49% per call), so the true API
sensitivities are ~1.3x smaller: **~0.18 at x0.5, ~0.28 at x0**. The lead's "host 0.5-0.6" took the whole
`host x0.5` figure, 70% of which is inter-call gaps (planning after syncs, app recording between submits) that a
graph does not touch. The most any change to per-call issue cost can buy in this frame, with issue still
pipelined against the device, is ~0.3 ms/frame; at the lead's 0.4 µs/node against 1.36-1.70 (API x0.25-0.3)
it is ~0.25 CUPTI-inflated, ~0.2 real.

### 2.3 The segment unit serialises host and device (MEASURED + MODEL, `postpin_segments_submitsplit.log`)

Per non-empty segment, with `d0` = first device op start, `d1` = last device op end, `s_k` = start of the
blocking call that ends the segment:

- host issue continuing after the device has started (`s_k - d0`): **median 111.5 µs, sum 4.268 ms/frame**
  (MEASURED). The device is running while the host issues the rest of the segment.
- device idle inside the segment's device span where the next op's API call had not yet returned
  ("host-late"): **0.143 ms/frame** (MEASURED; CUPTI inflation of API end times makes even this an
  over-estimate). Before splitting at vkQueueSubmit this read 1.584 — the difference is the application's
  inter-submit recording time, which no driver-side change collects (frame-boundary report).
- device idle inside segments attributable to launch latency alone: **1.422 ms/frame** (MEASURED).
- 83.3% of segments are device-bound: the host reaches the blocking call a median **78.7 µs before** the device
  finishes (MEASURED).

A graph per segment launches after its last node is parameterised, i.e. at ~`s_k - savings`. Frame delta per
segment = `(s_k - d0) - host_late - savings + first_node_latency` (MODEL; `savings` = today's de-inflated API
time minus `n_ops x u + launch`):

| assumptions | Δ ms/frame (median) |
|---|---:|
| u = 0.4 µs/node, graph launch 3 µs, first node 5 µs, inter-node gaps unchanged (the lead's numbers) | **+2.37** |
| same, inter-node gaps halved (the lead's "device 0-0.82") | +1.60 |
| same, inter-node gaps zero (impossible) | +0.86 |
| u = 0.4, no launch cost, no first-node latency (pure serialisation) | +2.28 |
| u = 1.0, launch 5, first 8 (pessimistic host) | +3.11 |

The sign is not model-sensitive: 4.27 ms/frame of host burst goes onto the critical path and the entire host
issue-API budget it could remove is 1.70 ms/frame. **No per-node price makes a segment-sized graph pay.**

### 2.4 The sub-segment unit (what is left)

If the unit is a chunk small enough to keep issue pipelined (a batch chain: VS, clip, s2, s3, compact, FS,
writeback plus 3 copies and a memset — DEAD_ENDS 1 iteration 19's unit, without its cache), the deferral
shrinks to one chunk per segment (~11 x ~20-30 µs = 0.2-0.3 ms/frame exposed, ESTIMATED) and the host saving is
bounded by §2.2 (~0.2 ms real at u = 0.4; ~0 at u = 1.0, since `cuGraphLaunch` itself submits per node). Net
host side: **-0.1 to +0.1 ms/frame**. The only pool that could make this positive is the device side:
1.42 ms/frame of launch-latency-only idle inside submits (MEASURED), if graph edges resolve dependent nodes
faster than stream launches do today (1.3 µs median gaps with PDL hiding 74.9% of front ends). That is a
HYPOTHESIS with no number on either GPU; the PDL links must survive as `CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC`
edges (`cuda.h:1880-1884`, kernel-to-kernel only) or the graph loses 0.44-3.02 µs per converted link.

### 2.5 Corrected ceilings

- **RTX, segment unit as proposed: ≤ 0 (MODEL: +2.3 ms/frame slower).** Realistic: negative.
- **RTX, chunk unit: 0.0-0.3 ms/frame**, the positive end only if a microbenchmark shows graph inter-node
  latency below today's 1.3 µs stream gap with programmatic edges preserved. Realistic 0.0-0.15.
- **B200: no trace exists; the segment-unit argument is GPU-independent** (it is host burst vs device chain,
  and a larger launch latency L_B makes the exposed first-node latency worse, not better). Chunk unit: bounded
  by the B200 launch-latency idle pool (b200 report: 342 gaps/frame x L_B, 1.2-2.9 ms if L_B = 4-8 µs) times
  whatever fraction graph edges shave — 0 to ~0.5 ms/frame HYPOTHESIS, contingent on the same microbench run on
  sm_100 (b200 report §probe table rows 1-2).

## 3. FEASIBILITY (`cuda.h` 12.8, code sites)

**What ExecUpdate accepts** (`/usr/local/cuda-12.8/include/cuda.h:21580-21679`, `:21011-21064`,
`:21066-21162`):
- Kernel nodes: all `nodeParams` fields may change including `func`, grid and block dims;
  `CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED` is documented as "CUDA driver < 11.2" (`:4262`); on 12.8 the
  restriction is `UNSUPPORTED_FUNCTION_CHANGE` for CDP / device-side-update / device-graph-launch / cooperative
  changes and context change. The driver's kernels are none of these. Priority must not change if
  `USE_NODE_PRIORITY` was set. **Accepted — the lead is right here.**
- Memcpy nodes: same contexts, same memory type, and for `cuGraphExecMemcpyNodeSetParams` **"both the
  instantiation-time memory operands and the memory operands in copyParams must be 1-dimensional"**
  (`:21073-21076`); returns `INVALID_VALUE` if multidimensional. The arg-block flush copies are 1D pinned-host
  to device (`cp_renderer.c:135` `cuMemAllocHost`, `:464-520`) with changing source, destination and size —
  updatable. But **29.3 `cuMemcpy2DAsync`/`cuMemcpy3DAsync` per frame sit inside 3.23 segments/frame that carry
  779.8 of the 1,124 ops/frame** (MEASURED). Those segments must be split at each pitched copy (more graph
  launches, more exposed first-node latency) or the copies issued outside the graph with event edges.
- Memset nodes: value and address may change; 1D size changes are "opportunistically allowed if the resulting
  work maps onto the work resources already allocated" (`:21122-21125`). The driver's memsets range from
  3-word counter clears to multi-MB fill-cursor clears whose sizes track `quads` and `nsegs`; a rejected update
  means a re-instantiation (14.8 µs, 46 KiB) on that segment. Unmeasured rejection rate; must be censused.
- Stream capture across the 8 side streams: fork/join via `cuEventRecord`/`cuStreamWaitEvent` is supported;
  `cp_pass_join` (`cp_renderer.c:7588-7607`) joins every used side stream back into `cp->stream` before the
  drain, `cp_pass_gate_stream`/`cp_pass_broadcast` (`:7633-7666`) fork them; `cp_pass_append` (`:9275-9310`)
  gates one stream per segment. A capture ends with `CUDA_ERROR_STREAM_CAPTURE_UNJOINED` if any side stream
  is left unjoined at the blocking call; the S2 segment-counter and S3 peel paths are single-stream, S1 joins.
  Plausible, but `cuLaunchKernelEx` with `PROGRAMMATIC_STREAM_SERIALIZATION` under capture becomes a
  programmatic edge only on 12.3+ drivers and the PDL take/decline accounting in `cp_launch_after`
  (`:541-600`) has no graph equivalent — the epoch rule (3) that refuses PDL across a flush copy becomes an
  edge-type decision at build time.
- Host-func nodes (`cuLaunchHostFunc`, 3/frame, cpvk_cmd.c:4831-4846) capture as host nodes and stall the
  graph exactly as they stall the stream today (host-sync §3: 77.5 µs median each); 1.98 segments/frame carry one.
- The upload ring: with issue deferred, `cp_upload_flush` (`:464-505`) copies execute later than today but the
  host watermark logic and the wrap `cuCtxSynchronize` (`:414-427`) still protect the staging bytes; arena
  generations (ARCHITECTURE §8 invariant 4) are unaffected because scratch addresses are chosen at issue time
  and only the launch moves. Invariants 8-10 (stable primitive order, order-free opaque merge, helper lanes)
  are untouched — the graph changes when work is submitted, not what work.
- Blast radius: every one of the ~1,124 device ops/frame goes through a new issue path; `cp_launch_audit`
  (the test forbidding raw `cuLaunchKernel`) needs a graph-aware form; `CUDAVK_NO_*` revert flag required
  (invariant 20); both captures' hashes and the dEQP/native suites (TESTING.md) must pass; PDL take-rate
  telemetry must be re-derived from edge data.

**Cheaper decisive probe than the proposed microbenchmark.** The proposed "SetParams+Launch for 13- and 86-node
templates vs the same number of cuLaunchKernel" measures host CPU, which §2.2 shows is worth ≤0.3 ms/frame
even if free. The decisive questions are two, and both are standalone `.cu` microbenchmarks in the style of
`notes/cdp2-microbench.cu` (no driver change):

1. **Dependent-node device gap**: a chain of 64 tiny kernels, (a) stream-launched back-to-back, (b) stream-
   launched with PDL, (c) one instantiated graph with default edges, (d) one graph with programmatic edges;
   `%globaltimer` at each kernel's first block minus the predecessor's last block end. If (c)/(d) are not
   below today's 1.3 µs median (and 0.44 µs PDL link value), the device half of the lead is zero and the
   whole lead is dead at u ≥ 0.4.
2. **Host cost per node of update + `cuGraphLaunch`** for 8-, 25- and 220-node execs (the p50/p90 segment
   sizes), versus 1.36-1.70 µs per `cuLaunchKernel(Ex)`, on RTX and on B200 (sm_100 numbers are absent; b200
   report probe rows 1, 2, 8).

Run 1 first; it is the only measurement that can make the corrected ceiling positive.

## 4. What must be fixed in the lead before it is quoted again

1. Drop the segment unit. The unit that keeps issue pipelined is a chunk (batch chain / support tail), which is
   iteration 19's unit with ExecUpdate instead of a cache — say so and inherit its costs.
2. Replace the "host 0.5-0.6 from host x0.5" ceiling with the API-only sensitivity: ≤0.37 CUPTI-inflated,
   ~0.28 real, at u = 0; ~0.2 at u = 0.4; minus first-chunk deferral 0.2-0.3.
3. Source the 0.4 µs/node update price or mark it HYPOTHESIS; `cuGraphLaunch` cost scales with node count and
   was not counted.
4. Count the 2D/3D copies (29.3/frame) and variable-size memsets as segment splits / re-instantiation risk.
5. State that the "device 0-0.82" half rests on graph inter-node latency being below the stream+PDL gap, which
   is unmeasured on both GPUs, and that PDL links must be re-expressed as programmatic edges to keep 0.44-3.02
   µs per link.

## 5. Reproduction

```
cd /tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/verify2-H
python3 segments.py /tmp/postpin-rtx2/trace.sqlite postpin.json /tmp/postpin-rtx2/ts.txt   # postpin_segments_submitsplit.log
python3 ta_api.py /tmp/postpin-rtx2/trace.sqlite postpin 0 0                                  # ta_api_postpin.log (38 s)
```
Key SQL (mode=ro): the closure audit's §5 set, plus
`select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=:main order by start`
joined to KERNEL/MEMCPY/MEMSET by `correlationId`, and `select systemClockNs from TARGET_INFO_SESSION_START_TIME`
to place the submit shim's CLOCK_MONOTONIC stamps on the trace clock.
