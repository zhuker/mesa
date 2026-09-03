# Outcome ledger: the 2026-09-02 analysis leads

`PERF_ANALYSIS_2026-09-02.md` proposed a ranked set of leads against a driver
that the previous campaign had declared exhausted. This is what each one turned
out to be worth when built and measured. Every figure below is an RTX 5090
paired-submit median, each capture on its own window (`WORKFLOW.md` 3.1), from a
three-round alternating A/B in one session with the change's own revert flag as
the control arm.

**Standing at the start**: favorite3 6.676, favorite2 5.480.
**Standing now**: favorite3 **5.940**, favorite2 **5.022** — a 11.0% and 8.4%
reduction, with favorite2 within 0.022 ms of the 5.0 goal.

**On the B200** the same set is worth more: favorite3 10.868 -> **9.883**
(-0.985) and favorite2 8.961 -> **8.283** (-0.678), measured as a set with
every switch reverted in the control arm, all twelve runs passing their gates.
That is 2.5x and 9.7x the RTX gain, which is what these changes should do --
they remove host-side serialisation and the B200 is the more latency-bound
host. Its favorite2 heavy band overlapped (that host drifts 0.3-1.3 ms between
sessions); every other band was disjoint. Individual attribution on the B200
was not measured, only the set.

## Built and measured

| lead | predicted | measured (f3 / f2) | verdict |
|---|---|---|---|
| **A** refuse un-appendable blended batches before the vertex work | 0.25-0.33 heavy, 0.20-0.28 light | **-0.2416 / -0.2355** | **LANDED** `f2fadd0c3b4` |
| **C** record the opaque episode gate once | 0.13-0.19 heavy, 0.19-0.27 light | **-0.1045 / -0.1184** | **LANDED** `21de64eed0a` |
| **B** hoist the shade chain's argument blocks above the rasterizer | 0.18-0.30 | **+0.0848 / +0.0446** | **REFUTED**, dead end 39 |
| **B2** the same hoist, but above the *vertex* launch | ~0.21 | **-0.1337 / -0.0326** | **LANDED** `b6f9e6577fc` |
| **F** VS run-ahead on a side lane | 0.3-0.5 heavy | **-0.3194 / -0.0475** | **LANDED** `a76d2926825` |
| **G-ctx** record-only context scope | 0.07 | pair below | **LANDED** `3756a6d576f` |
| **F-head** one `cuMemcpy3DAsync` per layered copy | 0.06-0.10 | pair: -0.0032 / -0.0351 | **LANDED** `1f97dffa4f7` |

Total landed: **-0.736 on favorite3, -0.458 on favorite2** (11.0% and 8.4%).
The four later changes measured **-0.3908 / -0.0699 together**, sub-additive
against a -0.456 / -0.115 sum because three of them shorten the same chain.

### A — the failed-append back-out
Blended batches that the A-buffer arm was certain to refuse were still admitted
into a pass episode, ran vertex fetch, VS and clip on a side stream, backed
out, forced a join of all eight side streams, and re-executed whole on the main
stream. The code comment called it rare; it was **9.3 times per frame**. The
refusal moved to `cp_pass_appendable()`, where every input to that decision
except the triangle count is already known. Mechanism falsifier: side-stream
`cp_clip_triangles` launches **9,400 -> 0**.

### C — the episode gate
Every side segment's gate was recorded on the main stream *after* segment 0's
launch chain, so none could start until segment 0 finished. Recording it once
after the episode's clears lets them overlap. Sized at a 0.434 ms/frame upper
bound (13.94 episodes/frame, segment-0 chain 61.1 us, side burst 284.8 us);
collected about a quarter of that. Required a switch-then-flush so each
segment's uniform rows land on its own stream, bounded by a fail-safe mark.

### B — the argument-block hoist
The copies between `stage3 -> fs_compact -> FS` are the coalesced upload flush
paying for blocks reserved after the previous launch: a copy-free dependent
link is 0.26 us, the copied ones 3.55 and 3.65, at 44 and 66 per frame. Hoisting
turned two copies into one and the frame got **slower**, on both captures, with
disjoint arms — even with the `COMPACT_PDL` it unlocks. See dead end 39 for the
rule it sharpened.

## Refuted without building

| lead | predicted | why |
|---|---|---|
| **I** shadow visibility clears | 0.06-0.09 whole | premise wrong: a depth-only scope has `colorAttachmentCount 0`, so `cpvk_batch_structural` refuses every draw in it. The 19 shadow "batches" per heavy frame are unbatchable single draws and no key relaxation merges them (`notes/SHADOW_VISBUF_CLEARS.md`) |
| **L** long `cp_clip_rast_fused` launches | 0.2-0.4 | ncu: a `grid=1` launch does **48 active cycles in 10,869 elapsed** (1/170 — one SM). The 200-400 us wall time is sharing the machine with seven side streams, and totalling it is summed concurrent duration, which the union-exclusive rule refuses |

## Still open

| lead | predicted | state |
|---|---|---|
| **G** device-decided episodes (stop waiting on readbacks) | 0.2-0.4 predicted, **unsized** | the last open driver-side item, and the only one not yet priced. It does **not** compose with F: the VS lane already hides much of the host time G was meant to recover, so its ceiling is below 0.4 and possibly below the 0.500 gate on its own. **Size it with the oracle-replay probe before building anything** — that is the rule entry 40 was just written to enforce |
| **E'** full kernel-parameter ABI | 0.1-0.3 beyond B | needs a `.local` census of every generated kernel first; B2 already collected the host-side part without an ABI change |
| **H** alpha-test retry convergence check | 0.03-0.08 heavy, ~0 whole | never attempted. Retry draws occur in about a third of frames, so it cannot move a whole-window median; it is a mean-only item |
| ~~**D, E, K**~~ scratch tables out of managed memory, pin `peel_any` and staging, UBO ring | 0.02-0.15 each | **CLOSED** — `DEAD_ENDS` 40. The page-fault trace was taken and sized all three at once: total UM fault stall is **0.3735 ms/frame** for every managed buffer in the process, below the 0.500 gate before any of them is built, and 90% of it belongs to one 21.5 MB app allocation none of these three touch. Their real budget is **0.037** |
| **§3.4** application readback fence, readback fill, scope count | 0.5-1.1 | outside the driver — the largest single remaining item anywhere, and it needs the application owner |

## What the analysis got right, and wrong

**Right, and this is the substantive point**: it found real structure in traces
this campaign had already taken and read the other way. Leads A and C were both
mechanisms nobody had seen, both confirmed exactly as described, and A's
prediction (0.25-0.33 heavy) bracketed the measurement (0.27). Its central
disagreement with the previous handoff — that the pools are per-launch fixed
latency rather than "clock-bound arithmetic" — is supported by the ncu counters
(0.5-15% of peak SM throughput, 3-25% occupancy).

**Wrong in three places**, each verified here:
1. Its section 1.1 per-frame counts divide by ~870 frames while the text claims
   1,024, inflating every count there by ~16% (846 kernels against a measured
   727.6). Its ms/frame figures do not carry the error.
2. Its frame decomposition sums to 8.29 ms against the 7.17 ms frame it claims
   to decompose.
3. Lead B was predicted at 0.18-0.30 and measured a regression.

**And one thing it caught that this project had gotten wrong**: dead end 37's
"the UVM migration exists only under nsys 2026.1.3" was itself an artefact. The
harness crashed in teardown on every run, dropping the final CUPTI buffer, so
an entire activity class read as zero. Root-caused and fixed (see below); the
refuted *mechanism* stays refuted, since that was decided by frame-time A/B.

## Side effects of the work, which outlast the leads

- **The favorite3 harness no longer crashes.** `favorite3.cpp` called
  `vkDeviceWaitIdle` on a device `frame_0000_2908.cpp` had already destroyed,
  under a comment asserting the capture never destroys it. That fault had been
  discarding **23 lines of stdout, 8 shim timestamps and every trace's final
  activity buffer** on every run since the harness was made. New gates: exit 0,
  6,947 rows, hash `e4a60a71...`, no torn tail (`WORKFLOW.md` 3.1).
- **99,999 was never a record cap** — it is one partial buffer flush. A clean
  run records 169,040 UVM rows.
- **`multithreading` is not bit-deterministic**: 20 runs give 20 hashes. Its
  historical corruption is invisible to a hash gate and needs the 20-run
  tolerance compare established here (baseline band: worst diff 2084-2089).
- **`WORKFLOW.md` now documents how a measurement is actually run** — the
  harnesses, their gates, the A/B recipe and the per-capture frame windows —
  which existed only in session memory before.
