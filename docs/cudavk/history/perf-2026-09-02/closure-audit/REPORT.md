# Adversarial audit of the cudavk performance-closure logic

Read-only audit, 2026-09-02. Nothing was built, run on the GPU, or modified in the repository.
Evidence: the documents named below, two existing nsys exports (`/tmp/postpin-rtx2/trace.sqlite`, the
post-host-pin RTX heavy-band trace the handoff's device numbers come from, and `/tmp/postdead/trace.sqlite`,
the 2026-08-30 post-dead-scope trace), the compiled harness `~/favorite3-cpp/out/src/frame_*.cpp`, and one
pure-python read-only analysis script in this directory, `trace_audit.py` (logs `postpin_audit.log`,
`postdead_audit.log`; JSON `postpin.json`, `postdead.json`).

Labels: MEASURED = read from a document or computed directly from trace tables; ESTIMATED = my arithmetic on
measured inputs (shown); MODEL = output of the dependency-respecting what-if replay (2.3), validated against the
unchanged trace but still a model; HYPOTHESIS = inference without a measurement.

## 0. Verdicts

| # | closure premise | verdict | reason |
|---|---|---|---|
| 1 | union-exclusive device time (0.5 ms gate) measures frame impact | questionable | a dependency-respecting replay of the same trace gives frame impacts 1.1-1.5x the union-exclusive figure for main-stream kernels and 2-4x for small copies/memsets (front-end latency is invisible to a union); no bound in either direction |
| 2 | VS/FS/clip pools are "arithmetic a scheduler does not shrink" | wrong as stated | trace duration model: VS 109 launches/frame, median grid 2 blocks, median 6.9 us, fitted fixed cost 9.6 us = 71% of the pool; clip 79%; stage3 100%. The campaign's own wide_bench.md: a launch's duration is the latency of one item up to ~1200 blocks. These are launch-count pools |
| 3 | the merge rule (entry 22) | sound for the shape measured, over-generalised | +0.410 stands; but "merged launch takes 2x one control launch" is reproduced by max/mean = 2.5 inside the fan-out groups without lost concurrency, so the mechanism attribution is not established, and the rule prices no host-side value |
| 4 | fan-out surrender 0.66-0.84 charged to every alternative | questionable | it is the price of serialising the same small launches (CUDAVK_NO_OPAQUE_STREAMS); a wide grid per stage finishes in ~max(item), which the fan-out group already takes; the charge is 0..0.84 depending on the design's intra-grid balance |
| 5 | device-paced waits are not collectable | sound per site, unsound in aggregate | removing one wait moves the block to the next (entry 29, measured); removing all of them is worth 1.37 ms/frame (heavy band) / 1.65 (post-dead) in the replay. Per-site slopes cannot see an all-or-nothing prize. The post-sync refill bubble is 0.17-0.26 ms/frame, so that hypothesis is refuted |
| 6 | entry 25 closes device-side decisions on the CDP2 price | questionable | CDP2 7.3 us/link measured and stands; conditional graph nodes were priced against the launch credit (0.037) instead of the sync-structure prize; stream memops measured elsewhere (1.8 us per write+wait pair) and never evaluated for this |
| 7 | graphs closed on "zero repeated episode recipes" | sound for exact-argument caching, not for topology reuse | frame-level Vulkan structure never repeats (1,426 distinct skeletons in 1,426 heavy-band frames); inter-sync segment kernel topology repeats: 411 distinct signatures over 17,391 segments, 95.6% seen in the previous frame, 90.4% same-position match - the census cuda_leads.md §4 said was never taken |
| 8 | per-item 0.5 gate proves the goal unreachable | does not follow; the sub-gate list does not reach it either | measured sub-gate items sum to ~0.35 ms; optimistic ceilings ~1.3 ms against a 1.68 ms whole-window gap (~3.2 heavy band). What reaches the goal in the replay is structural: geometry run-ahead 1.80, free syncs 1.37, launch latency 1.61, small-op elimination 0.4-1.3 |

What is solid: every measurement I could check reproduces (union 5.15-5.21, per-pool union-exclusive within 0.01,
17-23 blocking syncs/frame, blocked host 3.9 ms/frame, entry 20's +0.410, entry 34's 0.126, the fan-out surrender,
the CDP2 price). What does not hold is the inference layer: the admission metric, the "clock-bound arithmetic"
characterisation, and the per-site closure of aggregate mechanisms.

## 1. Documents and premises

Read: PERF_HANDOFF.md; PERFORMANCE.md §4-§7; DEAD_ENDS.md index, entries 1, 20-27, 29-38, "The rules these produced";
notes/RESIDUAL_AUDITS_2026-08-31.md; history/perf-2026-08-30-post-dead-scope-profile.md; perf-2026-08-30-mergekeys.md;
perf-2026-08-31-host-pin-readback.md; history/perf-2026-08-27/{wide_bench,wide_ceiling,item4_merge,item4_f4_device_fork,
item4_merge_results,cuda_leads}.md; notes/DEVICE_EPISODE_CHAINS.md, cdp2-microbench.cu, SHADER_CONTENT_CENSUS_RESULT.md;
RENDERER2_{DESIGN,M1_DESIGN,M1_FANOUT,VERDICT}.md (branch redesign/tilewalk); TODO.md "Performance leads that are open";
WORKFLOW.md §4.0/4.7/4.8; /tmp/residual-gpu-audit/AUDIT.md; /tmp/residual-host-audit/REPORT.md;
/tmp/residual-sequence-audit/result.json; ~/favorite3-cpp/APP_IMPROVEMENTS.md; history/EPISODES.md session 11.

Closure chain as written:
1. RESIDUAL_AUDITS line 6: "The 0.500-ms/frame union-exclusive admission gate governs all verdicts."
2. PERF_HANDOFF "Therefore": long dependent chain; CPU not the bottleneck (0.17 ms real work); waits are device pacing.
3. PERF_HANDOFF "What is left" / DEAD_ENDS §36 / RENDERER2_VERDICT: VS, FS, clip, copies are "clock-bound work that scheduling does not shrink".
4. DEAD_ENDS §22 merge rule, §20 (+0.410), §21.
5. §36: displaced 1.4623 - walk - bin - surrender 0.66-0.84 = 0.237-0.580; impossible upper bound -> 5.214.
6. §25: CDP2 7.3 us/link; conditional graphs closed under §1 retry-if, prize <= 0.037.
7. §1: 4,532 unique plan generations, zero repeated recipes.
8. post-dead-scope profile "Consequence": 2.604 cannot come from either axis alone.

## 2. Trace evidence

### 2.1 Windows and reproduction (MEASURED)

Frame boundaries = completion worker's cuEventSynchronize returns (one per submit; a frame is two).
```sql
select globalTid, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by 2 desc;
select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME
 where globalTid=:worker and nameId in (select id from StringIds where value='cuEventSynchronize') order by start;
```
| trace | window | frames | traced median | mean | device union | summed | idle | overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| postpin (RTX heavy band, post-pin driver) | all complete pairs after the first two | 1,024 | 8.1432 | 8.379 | 5.1517 | 7.8995 | 3.227 | 1.533x |
| postdead (2026-08-30 driver) | ns [9286588499,12338994337) = internal submits [2786,3608) | 410 | 7.1529 | 7.427 | 4.5073 | 6.4287 | 2.920 | 1.426x |

Documented figures reproduce: 5.2108 (their 1,013 frames) vs 5.1517 (my 1,024 with eleven lighter edge frames); 4.5075 vs 4.5073; postdead mean 7.429 vs 7.427.

Union-exclusive per pool, postpin, ms/frame (documented values from RENDERER2_M1_DESIGN.md §0 in brackets):

| pool | union-exclusive | summed | launches/frame | self-overlap |
|---|---:|---:|---:|---:|
| all kernels | 3.9968 | 7.0500 | 728.9 | 1.76x |
| clip (fused + fused_abuf + clip_triangles) | 0.7254 | 2.3181 | 108.99 | 3.20x |
| generated VS | 0.6815 [0.6892] | 1.4825 | 108.99 | 2.18x |
| generated FS direct | 0.6646 [0.6776] | 0.6646 | 65.71 | 1.00x |
| cp_clip_rast_fused | 0.5854 [0.5919] | 1.9142 | 60.78 | 3.27x |
| memcpy (all) | 0.4848 [0.490] | 0.5191 | 298.20 | 1.07x |
| cp_rasterize_stage3 | 0.4196 [0.4242] | 0.7642 | 65.71 | 1.82x |
| cp_fs_compact | 0.3683 [0.3724] | 0.3683 | 65.71 | 1.00x |
| memset | 0.3220 [0.326] | 0.3268 | 114.09 | 1.01x |
| cp_rasterize_stage2 | 0.2114 [0.2138] | 0.3692 | 65.71 | 1.75x |
| memcpy HtoD (kind 1) | 0.1954 | 0.2296 | 261.99 | 1.18x |
| A-buffer support | 0.1921 [0.1944] | 0.1921 | 66.23 | 1.00x |
| cp_vertex_fetch | 0.1244 | 0.1491 | 5.83 | 1.20x |
| cp_fs_writeback | 0.1086 [0.1098] | 0.1086 | 65.71 | 1.00x |
| cp_rasterize_stage3_abuf | 0.0915 [0.0926] | 0.3110 | 39.02 | 3.40x |

`main` split by same-stream grammar: VS if next same-stream kernel is a clip kernel or cp_vertex_fetch; FS direct if prev is cp_fs_compact or next is cp_fs_writeback; FS abuf if a neighbour is cp_abuf_*.

Host (MEASURED, CUPTI-perturbed), postpin: blocking API residency 3.937 ms/frame over 17.19 calls/frame (cuStreamSynchronize 8,419, cuMemcpyDtoH_v2 6,722, cuCtxSynchronize 1,444, cuMemcpyHtoD_v2 1,013 in window); all main-thread CUDA API residency 6.242 of 8.143. Postdead: 2.944 blocked over 23.37 calls/frame; 4.695 API of 7.153.

### 2.2 Concurrency profile (MEASURED)

Time with k device ops active, postpin, ms/frame: k=0 3.159; k=1 4.300; k=2 0.223; 3 0.184; 4 0.118; 5 0.087; 6 0.077; 7 0.074; 8 0.089; 9 0.001.
Exactly one op runs for 4.30 of the 5.15 ms union (83%); two or more for 0.85 ms/frame, carrying 3.60 ms of summed work. Postdead: 3.738 of 4.507 single-op. The frame is a serial chain of ~1,146 device ops/frame (728.9 kernels + 298 copies + 114 memsets) with ~17 host round trips, plus a fan-out covering 17% of busy time.

Stream placement, postpin, ops/frame main/side: memcpy HtoD 191.3/70.7; memset 102.6/11.5; VS 46.2/62.8; clip_rast_fused 44.2/16.6; stage2 and stage3 49.1/16.6 each; fs_compact, FS direct, fs_writeback 65.7/0 each; A-buffer chain entirely on side streams (37.05/frame each). The direct per-batch chain is mostly serial on stream 13.

### 2.3 Dependency-respecting what-if replay (MODEL, validated)

trace_audit.py rebuilds the frame as a discrete-event schedule from the trace: every main-thread CUDA call in issue order (launches, memcpys, memsets, cuEventRecord, cuStreamWaitEvent, blocking syncs); each device op starts at max(host call end, previous op on its stream, event dependency) + its recorded front-end gap (negative gaps kept so PDL secondaries still start early) and runs its recorded duration; a blocking call returns at the completion of what it waits on (cuStreamSynchronize resolved to its stream via SYNCHRONIZATION.syncType=3; blocking copies at their own copy's end) plus the recorded return residual; host inter-call gaps are preserved and slide earlier when a wait shortens. Dependencies: CUDA_EVENT (record) joined with syncType=2 (wait) by eventId.

Validation E0 (nothing changed): postpin median 8.1549 vs 8.1432; per-frame duration error p50 15.2 us, p90 27.5 us. Postdead 7.1668 vs 7.1529, p50 18.1 us.

Caveats: deleting an op removes its duration and its own front-end gap; host timing is CUPTI-inflated (traced heavy-band 8.14 vs 8.24 untraced, small); UVM faults and cache effects are not modelled - the pre-pin postdead trace under-predicts the measured host-pin win (model "delete all memcpy" -0.48 vs measured -0.93), so the model is not an upper bound; nothing is re-planned.

What-if results, frame median delta ms (postpin 8.143 / postdead 7.153):

| what-if | postpin | postdead |
|---|---:|---:|
| delete generated VS | -0.837 | -0.601 |
| delete FS direct | -0.776 | -0.508 |
| delete all clip kernels | -1.016 | -0.763 |
| delete cp_clip_rast_fused | -0.846 | -0.557 |
| delete stage3 | -0.468 | -0.342 |
| delete stage2 | -0.232 | -0.229 |
| delete fs_compact | -0.469 | -0.256 |
| delete fs_writeback | -0.136 | -0.080 |
| delete memcpy HtoD (262/frame arg uploads) | -0.799 | -0.427 |
| delete all memcpy | -1.011 | -0.476 |
| delete memset | -0.508 | -0.323 |
| delete A-buffer support | -0.269 | -0.369 |
| delete stage3_abuf | -0.088 | -0.133 |
| delete FS abuf | -0.063 | -0.060 |
| delete vertex_fetch | -0.119 | -0.156 |
| launch gaps zero (front-end latency free) | -1.607 | -0.987 |
| launch gaps half | -0.822 | -0.508 |
| all kernel durations x0.5 | -1.999 | -1.643 |
| all device durations x0.5 | -2.380 | -1.770 |
| device infinitely fast (durations and gaps 0) | -4.211 -> 3.944 | -2.997 -> 4.170 |
| host twice as fast | -0.813 | -1.129 |
| host infinitely fast | -1.348 -> 6.807 | -1.613 -> 5.554 |
| free syncs (host never blocks; device deps kept) | -1.372 -> 6.783 | -1.649 -> 5.518 |
| free syncs + host x0.5 | -1.372 (same) | -1.649 (same) |
| free syncs + launch gaps zero | -3.171 -> 4.984 | -2.801 -> 4.366 |
| geometry run-ahead (VS/fetch/clip start at issue, not behind previous batch's chain) | -1.801 -> 6.354 | -1.038 -> 6.129 |
| geometry run-ahead + gaps zero | -2.820 | -1.667 |
| VS only on a private lane | -0.780 | -0.521 |
| delete VS + clip | -1.959 | -1.405 |
| delete VS + clip + stage2 + stage3 | -2.655 | -1.841 |
| delete memcpy + memset | -1.569 | -0.789 |
| delete all kernels | -4.139 -> 4.016 | -2.973 -> 4.194 |

Readings: (a) neither side alone is under the goal: host infinitely fast leaves 6.81 (device chain with launch latencies and sync residuals); device infinitely fast leaves 3.94 (host issue plus sync residuals); they sum to 10.75 vs an 8.15 frame, overlapping 2.6 ms - a latency chain where host and device alternately hide each other; "the CPU is not the bottleneck" is true only in that the device chain alone exceeds the goal, the host path is worth 0.8-1.3 ms at the margin. (b) front-end latency is a first-class pool: 1.61 ms/frame heavy band, on no axis in the closure documents. (c) the sync structure is worth 1.37-1.65 ms as a whole; with free syncs the frame is purely device-chain-bound.

### 2.4 Union-exclusive vs frame impact (suspect 1)

| pool | union-exclusive (postpin) | replay impact | ratio | postdead ratio |
|---|---:|---:|---:|---:|
| VS | 0.6815 | 0.837 | 1.23 | 1.09 |
| FS direct | 0.6646 | 0.776 | 1.17 | 1.15 |
| clip (all) | 0.7254 | 1.016 | 1.40 | 1.34 |
| cp_clip_rast_fused | 0.5854 | 0.846 | 1.45 | 1.37 |
| stage3 | 0.4196 | 0.468 | 1.12 | 1.12 |
| stage2 | 0.2114 | 0.232 | 1.10 | 1.36 |
| fs_compact | 0.3683 | 0.469 | 1.27 | 1.37 |
| fs_writeback | 0.1086 | 0.136 | 1.25 | 1.25 |
| memcpy HtoD | 0.1954 | 0.799 | 4.09 | 3.68 |
| memset | 0.3220 | 0.508 | 1.58 | 2.29 |
| A-buffer support | 0.1921 | 0.269 | 1.40 | 1.46 |
| all memcpy | 0.4848 | 1.011 | 2.09 | 0.46 (UVM copies off the critical chain) |
| stage3_abuf | 0.0915 | 0.088 | 0.96 | 1.00 |
| FS abuf | 0.0748 | 0.063 | 0.84 | 0.84 |

Union-exclusive is neither an upper nor a lower bound: it under-states main-stream serial pools by 10-45% (no front-end latency; time-overlap read as non-criticality), under-states small copies and clears by 2-4x, and over-states side-stream pools off the critical chain (0.46x). The segment-level proxy "max per-stream summed busy" (also computed: VS 0.369, clip 0.505) errs the other way, which is why a dependency replay was needed.

Consequences: entries 31 (0.170), 32 (0.036), 33 (0.206) remain below 0.5 at any ratio (x1.25-1.45 -> 0.21-0.30); entry 35 (0.438 generous) crosses to 0.48-0.54 at the VS ratio and its closure then rests on the unpriced mechanism cost. The RESIDUAL_AUDITS VS->clip fusion refusal claimed the 0.717 ms of VS-end-to-clip-start gaps are "host issue-floor idle"; measured clip front-end gap after host issue AND stream availability is 2.3 us median, 0.666 ms/frame summed (postpin) - premise wrong; corrected ceiling is the main-stream share (44 of 61 clips), 0.15-0.5 ms, so the conclusion probably survives.

### 2.5 Latency shells (suspect 2)

Duration model, postpin, least squares duration = a + b*blocks (percentiles are the primary evidence; fit quality varies):

| class | n/frame | min | p10 | p50 | p90 (us) | a (us) | b (ns/block) | fixed share n*a/sum | blocks p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generated VS | 109.0 | 1.4 | 3.1 | 6.9 | 31.1 | 9.6 | 64.6 | 71% | 2 |
| FS direct | 65.7 | 0.8 | 2.8 | 3.2 | 18.0 | ~0 | 3.2 | grid always 4096; p50 3.2 us = almost-empty grid | 4096 |
| cp_clip_rast_fused | 60.8 | 1.5 | 2.3 | 9.8 | 113.6 | 24.9 | 46.8 | 79% | 10 |
| cp_clip_rast_fused_abuf | 39.0 | 2.1 | 3.2 | 5.9 | 23.3 | 7.7 | 316 | 87% | 1 |
| cp_rasterize_stage2 | 65.7 | 0.5 | 0.7 | 4.1 | 11.2 | 3.3 | 6.8 | 58% | 512 |
| cp_rasterize_stage3 | 65.7 | 1.3 | 1.4 | 4.8 | 30.4 | 11.6 | 0.0 | 100% (grid-independent) | 2048 |
| cp_fs_compact | 65.7 | 1.1 | 2.0 | 5.3 | 7.8 | 3.9 | 1.1 | 69% | 900 |
| cp_fs_writeback | 65.7 | 0.6 | 1.2 | 1.4 | 2.0 | 0.9 | 0.4 | 55% | 2048 |
| cp_vertex_fetch | 5.8 | 2.7 | 4.3 | 9.9 | 65.1 | 17.7 | 454 | 69% | 18 |

VS by grid: 48.0 of 109 launches/frame have one block (median 6.6 us, 0.36 ms/frame summed); 89% have <= 64 blocks; 3.2/frame exceed 1,024. FS direct: 46/frame at grid 4096 median 6.4 us; 16.6/frame at <= 2048 median 3.0 us.

wide_bench.md §1d (MEASURED, the campaign's own): per-launch time flat from P=1 to P=1200 items - "A launch's duration is the LATENCY OF ONE ITEM; the other 1199 ride along at no cost"; §2b: cp_clip_rast_fused launches one block 107 times with the SM on for 0.5% of elapsed. PERF_HANDOFF ncu: 0.5-12% SM throughput, 3-25% occupancy, 0.02 waves/SM. The B200 "equal or fewer cycles at lower clock" does not discriminate (pipeline latency is also clock-independent in cycles).

Therefore "vertex and fragment arithmetic plus copies, all clock-bound work that a different scheduler does not make smaller" is not supported by the documents' own counters: the pools are per-launch fixed cost times launch count; the work-scaling remainder is ~29% (VS) and ~21% (clip).

Renderer 2 recomputed (ESTIMATED; both accountings):

| accounting | displaced | undisplaced collapse | walk | bin | surrender | net | favorite3 lands at |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. RENDERER2_VERDICT (union-exclusive, per-batch VS/FS/clip kept, serial-launch surrender) | 1.462 | 0 | 0.092-0.255 | 0.13 | 0.66-0.84 | 0.237-0.580 | 6.10-6.44 |
| A'. impossible upper bound | 1.462 | 0 | 0 | 0 | 0 | 1.462 | 5.214 |
| B. frame-impact scale (replay deltas: stage3 0.468 + compact 0.469 + stage2 0.232 + abuf 0.269 + writeback 0.136 + stage3_abuf 0.088 + stage1/2_abuf ~0.06), same design | ~1.72 | 0 | 0.09-0.26 | 0.13 | 0-0.84 (2.7) | 0.49-1.50 | 5.18-6.19 |
| C. B plus per-run instead of per-batch geometry/shade launches (batches p50 16/scope -> runs 4-8/scope per M1 §1.5): VS 0.71 x 0.837 x (1-30/109) = 0.43; clip 0.79 x 0.85 x 0.5 = 0.34; FS ~0.5 x 0.78 x 0.5 = 0.2 | ~1.72 | ~0.9-1.0 | 0.09-0.26 | 0.13 | 0-0.84 | 1.4-2.5 | 4.2-5.3 |

C is HYPOTHESIS grade and does not remove the FS-bucketing problem or the hot-tile imbalance (one block = 94% of grid cycles, M1_WALKBENCH) that would bring the surrender back. The narrower point: "the upper bound of the whole redesign is above the goal" depends on keeping the per-batch launch structure, a choice of that charter, not a property of the workload.

### 2.6 Merge rule (suspect 3)

Entry 20 stands (+0.410, three pairs; B200 width-2 retry no signal). Contested: mechanism attribution and generalisation. F4 argues lost concurrency from "merged launch takes 2.0x one control launch while replacing five concurrent". In the shipping fan-out groups (side-stream cp_rasterize_stage3 inside one inter-sync segment, 1,644 groups, mean 10.3 launches): median max/mean duration = 2.52 (clip 1.87). A merged grid finishes at max(item) = 2.5x mean(item) - and so does the fan-out group. The 2x observation is explained without lost concurrency; "union busy fell" is what fewer, longer intervals look like. What cost 0.41 is not established by F4: candidates are lost cross-stage pipelining (a stage-barriered merge forbids segment j's stage3 under segment k's stage1) and the 199 PDL links lost per frame (item4_merge_results F5; 199 x 0.44-0.84 us = 0.09-0.17 ms). Consequences: the rule is sound for concatenating one stage across concurrent segments with a grid barrier; it does not test a merge that preserves per-item pipelining; it prices no host-side value (host path worth 0.8 ms at x0.5 in the heavy band).

### 2.7 Fan-out surrender (suspect 4)

CUDAVK_NO_OPAQUE_STREAMS serialises the same small launches: +0.71/+0.84 favorite3, +0.66/+0.72 favorite2 (MEASURED). A fan-out group of ~10 launches completes in ~max(item) instead of sum; with max/mean 2.5 that saves ~75% of the group's summed time. A wide grid per stage over the same items also finishes in ~max(item) (wide_bench: flat to ~1,200 blocks). For a one-wide-launch-per-stage design the surrender is approximately zero unless the wide kernel has the one-block-sets-duration imbalance the tile walk showed. M1_FANOUT itself called the proxy "pessimistic by construction ... an upper bound"; the VERDICT withdrew that only for the tile walk. Charging the full 0.66-0.84 to every alternative is not legitimate; it is 0-0.84, design-specific.

### 2.8 Sync structure (suspect 5)

Post-sync refill bubble (sync return -> first device op start): median 4.4 us, sum 0.171 ms/frame over 17.19 syncs (postpin); 4.2 us / 0.261 over 23.37 (postdead) - refuted as a place where time goes. Join tail 2.82 ms/frame is the single-stream part of each segment, not a lost-overlap cost. PERFORMANCE §5.2b's ceiling sum min(issue gap, blocked) measures host issue hideable behind waits, not sensitivity to the wait structure. Replay: free syncs = -1.372 (heavy band), -1.649 (postdead), all-or-nothing (entry 29: one site removed, 1.65 ms blocking -> 0.037). Every per-site slope (+1.02, +1.03, +0.44, -0.03; favorite3 drain +0.02/+0.21) is a one-site marginal with all other sites still blocking, which cannot see this prize. With free syncs the frame is device-chain-bound (host x0.5 changes nothing), so this composes with device-chain leads (free syncs + gaps zero = 4.98).

### 2.9 Entry 25 mechanisms (suspect 6)

MEASURED and standing: CDP2 tail 7.32-7.44 us/link, fire-and-forget 8.65-8.73, host 1.48-1.51. Not measured for this purpose: stream memops (cuStreamWriteValue64 + cuStreamWaitValue64 = 1,795 ns per pair, TODO.md Tier 2 item 12; ~900 ns each, CUDA13_UPGRADE.md:170); conditional graph nodes measured once (EPISODES.md session 11: 10.27 us per reusable conditional-graph launch vs 4.10 us for two direct one-block launches; "still appropriate for the episode decision: one graph launch can replace a roughly 654 us average stream drain"), then entry 25 set the prize to the launch credit (<= 0.037) - the wrong quantity; the prize is the sync-structure cost of 2.8 (1.37-1.65 ms/frame, ~80 us per sync), against which 10.27 us per decision and per-episode instantiation (14.8-15.1 us, ~5/frame = 0.07 ms) are small. Candidate 2 (predicated pre-issue) was refuted on +0.04-0.10 ms of repair launches and per-site spin slopes - a per-site instrument applied to an all-sites mechanism.

### 2.10 Graph units (suspect 7)

Frame level (harness): heavy-band files (external submits 4300-6300 -> files 1123-2548, 1,426 files): Vulkan call skeleton distinct in every file (1,426/1,426), adjacent-frame similarity 0.17-0.88, draws/frame 105-281 (median 229), render passes 0-11 (median 8), coarse pass structure 1,379 distinct. One-graph-per-frame ExecUpdate is dead at the Vulkan level.

Segment level (trace; signature = ordered kernel names between consecutive main-thread blocking calls):

| trace | segments | distinct signatures | seen in previous frame | same-position match | top-50 coverage | kernels/segment p50/p90/max |
|---|---:|---:|---:|---:|---:|---|
| postpin | 17,391 (17.0/frame) | 411 | 95.6% | 90.4% | 93.4% | 13 / 86 / 601 |
| postdead | 9,450 (23.1/frame) | 75 | 98.3% | 85.3% | 99.7% | 13 / 72 / 178 |

(6,742 postpin segments are empty: back-to-back blocking calls.) Topology recurs almost perfectly while arguments rotate - the distinction cuda_leads.md §4 drew and said was never counted. cuda.h 12.8: CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED applies only to "CUDA driver < 11.2", so kernel node functions may change under update; memcpy node params may change within kind/context.

### 2.11 The 5 ms arithmetic (post-dead-scope profile)

7.604 - 2.604 = 5.000 and 7.604 x 0.22 = 1.673 check; each axis (1.673 idle, 1.400 raster/clip, 1.098 shader) is < 2.604, so the sentence is correct arithmetic. As a closure: (i) axes are union-exclusive and a sampler busy% - neither is frame impact (2.4); (ii) two pools were omitted - memcpy/memset/UVM (union-exclusive 1.03 + 0.14 in that trace) and launch front-end latency (replay 0.99 in that trace) - and the omitted memcpy pool paid 0.93 ms/frame the next day (host-pin); (iii) 78% busy is a whole-window sampler median mixed with trace-window figures over a different 412-frame slice (WORKFLOW 4.8/4.9).

### 2.12 Additivity (suspect 8)

| item | value ms/frame | status | additive? |
|---|---:|---|---|
| 38 side streams 16 | favorite2 -0.084 (-0.123 heavy), favorite3 -0.029 | measured, declined (+153 MB) | yes |
| TODO 7 opaque FS UBO row concatenation | 0.04-0.06 | unmeasured | yes (66 FS launches/frame all on stream 13) |
| 34 threaded submit | 0.126 | measured | partially |
| 29 scratch high-water | 0.037/0.012 | below noise | no |
| CUDAVK_CTX_SCHED=spin | -0.11 favorite3 only (mergekeys.md) | measured, left as env | yes |
| 15 / §6 item 3 writeback fusion | ~0.04 | parked | yes |
| 31 depth-only deferral | 0.170 ceiling (x1.25 -> 0.21) | ceiling | partially |
| 32 two-slot | 0.036 | ceiling | - |
| 33 cross-frame geometry | 0.206 ceiling (x1.3 -> 0.27) | ceiling | overlaps 35 |
| 35 vertex reuse | 0.438 ceiling (x1.23 -> 0.54) | ceiling | overlaps 33 |
| §6 item 4 run-ahead across the drain | 0.092 converted / 0.387 ceiling (old capture) | measured on old | overlaps 4.1 |
| 30 Hi-Z | <= 0.103 | closed | - |
| 25 tail chaining | 0.035-0.037 host, -0.26 device | refuted | - |
| 26 INTERP_INLINE/COMPACT_PDL | -0.05 | refuted | - |

Measured, plausibly additive: 0.126 + 0.11 + 0.029 + 0.05 + 0.04 = ~0.35. With optimistic ceilings: ~1.3. Whole-window gap 1.676; heavy-band gap ~3.2. The gate did not prove unreachability; the sub-gate list does not reach the goal either; the structural set of 2.3 does.

## 3. Internal consistency notes

- PERF_HANDOFF "steady-state driver CPU work about 0.17 ms/frame" vs RESIDUAL_AUDITS "steady-state planning about 0.33 ms/frame", same profile.
- PERF_HANDOFF "wait time is device pacing rather than a cost that can be deleted" vs its own record that the fan-out and PDL - scheduling changes deleting no work - were the two mechanisms that paid.
- M1_FANOUT calls the surrender "an upper bound ... pessimistic by construction"; VERDICT and PERF_HANDOFF quote it as the cost.
- DEAD_ENDS §25 sizes conditional graphs by the launch credit; EPISODES.md session 11 sized them by the ~654 us drain they replace.
- §5.2b `ready = 0 at every site` shows each wait is for real device work; it says nothing about whether the structure is free.

## 4. Leads reopened by corrected premises

Ceilings are RTX heavy-band replay deltas (MODEL) unless marked; B200 unknown (no B200 trace available; per entry 37 it must be taken under nsys 2026.4.1 with a positive control).

4.1 Per-batch geometry run-ahead inside a host burst. Let batch k+1's VS (fetch, clip) start when issued instead of behind batch k's shade chain on the same stream, via N rotating VS output slots and clip queue sets. Ceiling 1.80 (mean 1.58) heavy band; 1.04 postdead; VS-only 0.78/0.52. Realistic ~half: queue-set reuse hazards (clip(k+8) vs stage3(k) on set s%8), VS output lifetime, bookkeeping. Closest: 27 (chain of scopes; this is batches inside a scope, not censused), the fan-out (raster of an opaque episode's segments; this covers the 46 VS + 44 clip per frame on stream 13), §6 item 4 (holds batches across the drain), RESIDUAL_AUDITS VS->clip fusion. Probe: done (replay); next a flag issuing VS on a rotating side stream with N slots, hash-gated A/B. Effort medium.

4.2 Remove the read-back-and-decide points as a set (speculative issue with end-of-submit repair, or conditional graph nodes). Ceiling 1.37 heavy / 1.65 postdead minus repair on overflow and one decision op per episode (10.27 us conditional node or 0.9 us memop gate). All-or-nothing. Closest: 25 candidates 2/3, 29, §7 S0 UNSAFE_NO_OVERFLOW (worst-case sizing churn; flag text: "skip episode overflow readback and assume fragment/quad arrays fit"). Probe: oracle replay - record each decision's outcome in run 1, replay them without waiting in run 2 (output correct by construction); the frame delta is the prize. Effort low (probe) / high (mechanism).

4.3 Segment-level CUDA graphs with cuGraphExecUpdate. 411 topologies (46 KiB each = 19 MB; instantiate 14.8 us once each), per-segment node param updates (~0.4 us/node vs 1.1-1.4 us launch API), one launch per segment; needs capture across 8 streams (cp_pass_join/broadcast shape). Ceiling 0.5-1.4 (host 0.5-0.6 from the x0.5 sensitivity 0.81; device 0-0.82 from "launch gaps half"), realistic 0.3-0.7. Closest: 1 (exact-argument cache). Different: template + re-parameterisation, no key; cuda_leads §4/§7 "never tried". Probe: microbench SetParams+Launch for 13- and 86-node templates vs the same number of cuLaunchKernel; census of ExecUpdate-refused parameter changes. Effort high.

4.4 Eliminate per-batch argument copies and counter clears. 262 HtoD copies (mean 2.2 KB) and 114 memsets/frame, one per launch boundary (§8). Replace by one argument table upload per host burst and prologue clears. Ceiling: replay 0.80 + 0.51 heavy (0.43 + 0.32 postdead); documented marginal prices give 262 x 1.30 + 114 x 0.8 = 0.43 - 3x apart, regime-dependent; quote 0.4-1.3, realistic 0.3-0.6. Closest: 8, 7, §7 "copies fall only when boundaries fall" (agreed: this removes the boundaries' copies). Probe: bytes and destinations per burst from CUPTI_ACTIVITY_KIND_MEMCPY; a flag batching one burst's uploads. Effort medium.

4.5 Renderer 2 re-chartered to collapse launches per run. Accounting C: 1.4-2.5 (HYPOTHESIS). Closest 36, 24. Probe: host census of runs per scope (M1 §1.5, no GPU). Effort very high.

Leads 4.1-4.4 attack different resources and compose in the replay (free syncs + gaps zero = 4.98; run-ahead + gaps zero = 5.33) but are not additive at face value; honest combined statement: ~3.2 ms of structural cost across these axes in the heavy band vs a 3.2 ms heavy-band gap to 5.0.

## 5. Reproduction

```
cd /tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/closure-audit
python3 trace_audit.py /tmp/postpin-rtx2/trace.sqlite postpin 0 0 postpin.json > postpin_audit.log
python3 trace_audit.py /tmp/postdead/trace.sqlite postdead 9286588499 12338994337 postdead.json > postdead_audit.log
```
Key SQL (mode=ro):
```sql
select start,end,streamId,correlationId,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?;
select start,end,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?;
select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>? and start<?;
select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=:main and end>? and start<? order by start;
select timestamp, streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT;
select start, end, streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2;
select correlationId, streamId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=3;
```
Harness recurrence: files mapped to external submit indices via their vkQueueSubmit( counts (0/1/2/4/8/10 per file; 6,947 total); files holding submits 4300-6300; sequences of \bvk[A-Z][A-Za-z0-9]* hashed. Fan-out statistic: side-stream cp_rasterize_stage3 (and cp_clip_rast_fused) launches within one inter-sync segment, groups >= 3, max/mean duration per group, median over groups.
