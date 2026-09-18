# Session result: the 4x-native investigation


**B200 current-tree standing (2026-09-08):** favorite3 is **8.5293 ms whole /
9.9813 ms heavy**, combined across a three-round six-arm identical-control
session. All six arms preserved the exact hash, 6,947 rows and 18/18 sentinels.
This improves the old pre-wide-episode 9.883-ms whole result, but does not meet
the 2-ms objective.

## What was achieved

Two changes landed on `cudapipe-vk-native`, both default-on, both bit-identical:

| change | favorite3 whole | favorite3 heavy | favorite2 whole | favorite2 heavy |
|---|---:|---:|---:|---:|
| opaque episodes span viewport/raster/depth changes | -0.076 | -0.075 | -0.150 | -0.265 |
| depth-only scopes batch | -0.657 | -1.151 | -0.026 | -0.279 |
| **combined, measured together** | **-0.733** | **-1.226** | **-0.176** | **-0.544** |

    favorite3  5.940 -> 5.176 whole   7.288 -> 6.023 heavy   (-12.9%)
    favorite2  5.022 -> 4.867 whole   5.598 -> 5.054 heavy

Gates: 79/79 tests, 18/18 sentinels bit-identical on both captures, exact
stdout hashes, the 18-sample sweep with **no verdict changed**, `FLAGS.md`
regenerated, no-getenv audit clean. Reverts: `CUDAVK_NO_WIDE_EPISODE`,
`CUDAVK_NO_DEPTH_ONLY_BATCH`.

## Against the objective

| band | favorite3 | native | ratio | 4x target |
|---|---:|---:|---:|---:|
| loading | 1.917 | 0.351 | 5.46x | 1.40 |
| light | 4.117 | 0.478 | 8.61x | 1.91 |
| heavy | 6.023 | 0.582 | **10.35x** | 2.33 |

**Not met.** 3.09 ms more is needed on favorite3; every remaining identified
lever sums to about **1.5 ms**, landing at 3.70 -- still 1.77x over, with
nothing named for the remainder.

## What was ruled out, and how

- **14 flag-level levers**, 10 refuted. The decisive one: frame time is
  **invariant across the whole register/spill curve** -- 203 registers with
  zero spill and 128 with 168 B of spill measure the same frame.
- **The predicate search**: 3 relaxations tried, 2 landed, 1 (depth-only
  episodes) correct but slower.
- **bin+walk**, the largest structural estimate at -1.04 ms, **built** and
  measured **+1.8 ms worse**, twice, on two baselines.
- **Five estimation errors**, all self-caught and recorded in
  `FOURX_DIAGNOSIS_2026-09-05.md`: the frame boundary is not the
  application's; overdraw is 2.29x not 12x; occupancy is not a sufficient
  argument; the fragment pool is 0.729 ms union-exclusive, not the 2.25 ms
  summed over kernels all named `main`; and the trace has 17 pools, not 4.

## The transferable lesson

Both wins came from one move: **when a measurement says "the workload does
this", check whether the driver had to care.** The episode-wide state
comparison and the colour-attachment test were each one predicate asking for
something no consumer needed. Together they beat every flag-level lever
combined, and they were found only after three separate "proofs" that the
remaining gap was structural.

## What would close the rest

Nothing in the driver. The gap is software rasterisation against
fixed-function hardware: interop with the graphics pipeline, or a change to
what the application asks for. `~5x native` is the realistic ceiling for this
architecture, and the identified-but-unbuilt work is worth about half of it.

## Run-level implementation follow-up (2026-09-08)

The remaining structural proposal was implemented far enough to settle its M2
gate. Default-off arms covered authoritative run collection, exact per-batch
prepare, stable global primitive IDs, transactional clipped storage, dense
bin/tile visibility, compact direct visibility, and a global-visbuf classic
continuation. All correctness gates passed, including the expanded 83-test
suite and exact capture sentinels.

None produced the required gain. Dense and direct visibility were slower. The
best exact classic transaction was neutral in strict alternating measurement:
5.1847 -> 5.2192 ms whole and 6.0361 -> 6.0009 ms heavy. A deferred VS-to-clip
PDL follow-up was disjointly worse by 0.1343/0.1225 ms and was removed. This is
now measured implementation evidence for the ceiling stated above, rather than
an estimate based only on partial predecessors.


## Final parked-lead closure: fused writeback

The one parked `DEAD_ENDS` item was recovered and its correctness defect fixed:
fused colour writes had omitted hardware-texture cache notification. The
current default-off port, plus a stricter SSA colour handoff, is exact and
resource-gated. It is nevertheless neutral in strict alternating timing
(favorite3 −0.0147/−0.0017 ms whole/heavy; favorite2 −0.0278/−0.0128, all arms
overlap). Both default and enabled 83-test suites pass. This closes the last
parked implementation without changing the architecture-floor conclusion.


## Final unmeasured-lead closure: opaque UBO concatenation

The last TODO item without a measurement was implemented behind default-off
`CUDAVK_OPAQUE_UBO_CONCAT`. It reuses the existing concatenated row table and
per-range `row_base`, and both capture hashes and row counts are exact. The old
estimate no longer describes the current renderer: only 1,772 favorite3 and
1,749 favorite2 FS launches disappear over the complete replays, about **0.5
launch/frame** or **0.0004 ms/frame** at the measured serial-boundary price.
The zero-code gate rejects timing or a default-on change.

The final validation also corrected the sanitizer cleanup itself. Gating
optional sampler-module globals on `num_tex_instrs` was too narrow because that
counter covers only statically specialisable samples. A generated-binary
`uses_tex` bit now covers dynamic, gradient, gather, size and other software
sampler operations while still avoiding `cuModuleGetGlobal` on texture-free
modules. The serialized suite is 83/83 and compute-sanitizer reports zero
errors on both textured and texture-free probes.


## Joint architecture-floor audit

`tests/cp_architecture_floor.py` makes the latest 273-frame Nsight interval
accounting reproducible. The trace spans 3.635500 ms/frame, of which the union
of all GPU kernels is 2.110307 and shader kernels named `main` alone occupy a
0.909511-ms union. An impossible deletion of **every non-shader kernel** has
only 1.200796 ms/frame of overlap-correct credit and leaves 2.434705 ms/frame
with retained timestamps. Even subtracting the independently measured
completion-aware threaded-submit gain of 0.126184 ms leaves 2.308521 ms/frame,
above the 2-ms target.

This is a deletion bound on a CUPTI trace, not an unprofiled timing prediction:
it does not prove that a joint redesign cannot compact the retained schedule.
It does prove that another independent fixed-kernel rewrite, host-submit tweak,
or their simple additive combination is insufficient. A viable proposal now
has to change shader execution and scheduling together, or change the external
constraints.

Reproducer:

```
src/cudavk/tests/cp_architecture_floor.py --frames 273 \
  --host-gain-ms 0.126184 --target-ms 2 \
  /tmp/nsys-runclip-classic-nvtx.sqlite
```


## Persistent shader-interpreter gate

A shader-bytecode VM was the one remaining mechanism capable in principle of
combining launch removal, fixed-function scheduling and shader dispatch without
the linked megakernel's register union. `tests/cp_shader_interp_bench.cu`
measures its minimum mechanism cost. On RTX 5090, a uniform one-opcode local VM is
5.53x slower than compiled code at only eight operations and 21.65x at 128;
the realistic four-opcode loop ranges from 5.75x to 28.75x. Moving the virtual
register file to shared memory costs 2.99x at eight operations, rising to 8.33x
at 64. Scalar-register switches are 3.00x to 11.02x. A quad-per-thread shared
VM amortises decode and reaches 1.50x/2.01x at 8/16 ops, but is 3.49x by 32.
The capture census is decisive: only 15.96% of favorite3 and 10.21% of
favorite2 FS launches are at or below 16 ops, while 47.20%/48.01% exceed 128;
80.2%/86.4% of scheduled blocks are above 32 ops. Both hashes and row counts
remain exact. The complete traced renderer allowed at most 2.20x shader growth
even if all other work vanished. The VM is therefore closed on this GPU before implementation. The same CUDA
13/sm_100 benchmark now runs on B200: quad/shared costs 1.66x at 8 ops, 2.32x
at 16, 3.58x at 32 and 4.48x at 64. Only the <=8 bucket fits there, covering
7.66%/4.27% of the two captures. The VM is therefore closed on both required
architectures.


## Residency correction after the architecture audit (2026-09-09)

Matching Nsight Systems 2026.4.1 traces on B200 identified migration of the
renderer-owned host-written scratch arena. Device-local peel convergence plus
mapped pinned scratch is exact and now default-on. It saves 0.170-0.289 ms
whole-frame on RTX and 1.034-1.055 ms on B200 across both captures; heavy-band
savings are 0.212-0.269 ms and 1.095-1.137 ms respectively. Default and exact
revert suites pass 83/83. `PERFORMANCE.md` records all alternating arms and
`DEAD_ENDS.md` 50-51 records why broad application-memory treatments were
removed.

The current shipped-default favorite3 median is 4.9813 ms on RTX; the exact
B200 candidate median is 7.7948
ms on B200 in their respective paired sessions. These are substantial gains,
but neither is the requested approximately 2 ms. The architectural conclusion
therefore remains: the 4x objective is not complete and a new joint execution
mechanism, not another independent migration or fixed-kernel tweak, is needed.


## Current default stage census (2026-09-09)

A complete 6,947-submit CUDA+NVTX trace of the shipped default used Nsight
Systems 2026.4.1 with a 1,000-ms periodic CUDA-buffer flush. The flush is why
this report is attribution evidence, not frame-time evidence. It retained
1,110,019 CUDA kernel events; unlike the unflushed attempt, no incomplete CUDA
events were dropped. The application stdout prefix remained exact.

Launch APIs were associated with the innermost exact cudavk NVTX stage range.
The report warns that some NVTX events might be missing, so the named split is
a census rather than an overlap or utilization metric. It classifies 98.6% of
generated `main` launches; the complete kernel table provides the totals:

| submitted stage | launches | summed ms/frame |
|---|---:|---:|
| raster | 257,409 | 1.769 |
| vertex | 157,911 | 0.808 |
| fragment shader | 101,795 | 0.437 |
| interpolation | 85,803 | 0.148 |
| writeback | 85,803 | 0.057 |
| composite | 12,010 | 0.027 |

Generated `main` shader kernels sum to 1.275 ms/frame and every kernel sums to
3.966 ms/frame. Sums count concurrent kernels twice and therefore are not
critical-path time. They do give a strict necessary deletion gate: moving the
unprofiled shipped frame from 4.9813 to 2.0 ms requires 2.9813 ms of actual
frame credit, at least **75.2% of all summed kernel duration** even before
overlap erodes deletion credit. Raster alone (1.769) or all generated shaders
alone (1.275) cannot qualify. Raster plus every generated shader is only 3.045
ms/frame summed, just 0.064 above the required saving before overlap and
mechanism cost. A credible next design must therefore replace or overlap
almost the whole raster+shader pipeline together; another independent stage
optimization cannot meet the objective.


## CUDA feature audit conclusion

Independent renderer and CUDA-capability audits applied the corrected
whole-pipeline gate. Thread-block clusters and Blackwell Cluster Launch Control
only coordinate/rebalance one CUfunction; green contexts only partition SMs;
TMA only accelerates copies; graphs, PDL and CDP retain child work. None
provides resource-isolated indirect execution of arbitrary generated shader
CUfunctions inside a persistent raster kernel. The available representations
remain linked selectors (measured occupancy collapse), interpretation
(measured 3.49x+ / 3.58x+ on dominant RTX/B200 shaders), or child launches.
Exact winner-before-shade and blended sort/composition dependencies prevent
free speculative overlap.

Consequently there is no evidence-admissible CUDA 12.8/13 mechanism left for
the 2-ms goal under the stated constraints. `DEAD_ENDS.md` 52 records the
0.265-ms prerequisite for any future claim. The goal is not achieved; progress
requires allowing graphics hardware/ICD use, relaxing exact/general Vulkan
semantics for a workload-specific renderer, raising the target, or acquiring a
new CUDA execution capability.


## OptiX RT-core candidate audit

The last hardware-assisted software-visibility idea was reopened because the
old rejection used an unmeasured 2.5-ms LBVH estimate. Pure-build probes
corrected that estimate to 0.520-0.527 ms for one 354,000-custom-AABB GAS and
about 0.040 ms for a fast-build refit.

A boundary re-audit then corrected the workload shape: approximately one
opaque run carries 90.8-95.2% of current-capture triangles. A one-GAS
fast-trace probe is therefore the optimistic relevant case. Three locked runs
measured 0.179-0.181 ms for fixed-count refit and 0.057-0.058 ms for one
1280x720 trace, or **0.237-0.238 ms together**. That nearly consumes the entire
0.265-ms joint-renderer allowance while producing only closest-hit visibility,
and can displace at most the 1.769-ms raster pool rather than the required
2.9813-ms frame credit. A 16-GAS split probe remains only a pessimistic bound:
1.475-1.613 ms for refit plus trace.

OptiX closest-hit is not Vulkan rasterization. Exact use needs conservative
custom AABBs and cudavk's edge test, must append and stable-ID-sort every hit
because traversal is not submission order, and must retain the CUDA 2x2 helper
quad shader path because OptiX does not preserve neighbor lanes. Discard,
shader-written depth and ordered blending also stay outside traversal. It
therefore cannot replace the joint six-stage pool. The optional reproducible
probe is in `src/cudavk/tests/cp_optix_candidate_bench/`; the corrected decision
is in `DEAD_ENDS.md`. OptiX would also change the CUDA-Driver-only constraint
and still needs an OptiX 9.x B200 smoke test.


## Current native denominator audit

Three explicit-NVIDIA-ICD runs per capture confirm native medians of
0.5193/0.5682 ms (favorite3 whole/heavy) and 0.5062/0.5244 ms (favorite2).
The exact 4x limits are therefore 2.0773/2.2727 and 2.0248/2.0977 ms. Current
cudavk is 9.59x/10.15x and 9.30x/9.28x native respectively. The target failure
is not an obsolete 0.5-ms assumption.
