# Ordered lazy clipping prototype — experimental, default off

**Not admitted. Validation-only replay is authorized; no timing or acceptance.** The offline compiler audit
found changed SASS FFMA multiplicand order in varying replay versus baseline.
This is not a measured output mismatch, but exact exceptional-value bit identity
has not been established. No tolerance or altered sentinel contract is accepted.

Base: `69a58a486d4`. Switch: `CUDAVK_CLIP_LAZY=1` (value boolean).
Independent of common-plane rejection and identity-plane copy bypass.
No queue, launch, kernel ABI, or shader change. No performance claim.

## Representation and side effects

The existing all-inside path is unchanged. Only the slow path is replaced.
Positions use two arrays of ten float4s and parallel node IDs. Planes, directed
edges, classification, capacity guards, denominator tests and float expressions
keep their original order. A copied vertex retains its node ID. Each actual
interpolation creates a backward edge pair and the exact computed `t`.

There are 73 node slots: three leaves plus at most ten emitted intersections
per plane for seven planes. This bound does not assume finite convex geometry.
Depth is at most seven. The original ten-vertex truncation is unchanged.
Rejected polygons return through the existing retirement rule without touching
varying payload. Surviving polygons evaluate all created nodes, slot by slot,
in creation order with the unchanged `clip_lerp` expression. There are no
barycentric weights or reassociation. Slot zero uses the stored position result.
No reachability pruning is included.

Each output fan ID is allocated once in the original per-triangle fan order,
with the same failure/count semantics. Slot-wise scatter writes only allocated
outputs. References are published only after all payload is complete, before
the caller can rasterize. Optional IDs and retirement keep their original
conditions. Publication moves later within the same thread; no concurrent
consumer exists before this helper returns or the kernel ends. This does NOT
promise identical inter-thread compact atomic ID scheduling. Both arms already
use scheduler-dependent atomic allocation. The CPU gate compares per-input
sequential effects; future compact device comparisons must normalize identity.
There is no second side-effecting clip invocation or duplicate allocation.

## Admission

The arithmetic graph is preserved in source. This alone is not a CUDA bit
identity proof. CPU source differential tests and offline compiled instruction
inspection are required before this experiment can advance. In particular,
contraction and special-value operand order may change after scalarization.
Any unexplained raw-bit mismatch blocks admission; no tolerance is allowed.

The existing `CLIP_EQ_CHECK.md` in the separate clip-check worktree was reviewed.
That debug checker is not integrated here. Future GPU admission needs independent
original/candidate canonical clip outputs and untouched baseline sentinels,
negative controls, full input/mode coverage, both captures and all-frame external
image validation. Side effects must still occur once. Do not time this until
those gates pass. A smaller declared local array is not a measured saving.

## Offline CUDA result: proof blocker

CUDA 12.8 NVRTC and NVCC compiled both arms for compute_120 with CP_PDL=3;
PTXAS assembled both for sm_120. Standalone PTX retains `sub(b,a)` followed
by `fma(t,delta,a)`, with no FTZ. SASS changes varying-replay multiplicand order:
baseline uses `FFMA(delta,t,a)`; some candidate sites use `FFMA(t,delta,a)`.
This leaves finite rounded results unchanged, but does not establish identical
NaN payload/sign behavior. PTX specifies single-precision NaN results as
unspecified. No differing output bit was observed; this is a proof blocker,
not a demonstrated numerical regression. Fused arithmetic audit stopped at
this earliest blocker. Offline PTXAS is not a claim about a future driver JIT.

| Entry | Baseline stack / registers | Candidate stack / registers |
|---|---:|---:|
| cp_clip_triangles | 5120 / 44 | 2480 / 56 |
| cp_clip_rast_fused (also abuf) | 5248 / 56 | 2512 / 56 |
| cp_run_clip_setup_direct | 5152 / 72 | 2512 / 72 |
| cp_run_clip_setup_tile_count | 5152 / 64 | 2512 / 72 |

All listed entries report zero spill loads/stores. These are resource counts,
not timing, dynamic local-memory traffic, or evidence of progress toward 3 ms.
Full compiler options, source snapshots, hashes, PTX/SASS and the caveats are
in evidence `offline/REPORT.md`. No workaround or approximation was attempted.
The prototype is committed on its isolated experimental branch for further
validation, not admitted to production.

## CPU checks completed

`meson setup build-cudavk -Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers=
-Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled`, then
`ninja -C build-cudavk -j12` completed successfully. Registry documentation,
no-getenv, launch audits and `git diff --check` passed.

`python3 src/cudavk/tests/cp_clip_lazy_test.py` passed three configurations:
contraction off, native contraction fast, and ASan/UBSan. Each made 10,326,528
paired calls over 498 directed/random geometries. Slots 1..18 include the
16-slot clamp; all 16 optional-output modes, eight capacities, three counter
starts, and three sequential input triangles are checked. Every output bit,
untouched region, input, count, ID and normalized reference is compared against
independently extracted original source and ABI from the baseline commit.
The corpus includes raw uint32 positions, geometry-biased random positions,
all seven planes, repeated vertices, finite overflow/cancellation, boundary
values, and explicit varying NaN/sNaN/Inf/subnormal/signed-zero patterns.
This is finite corpus coverage, not an exhaustive node/topology proof or a GPU
arithmetic/concurrency test. No CPU mismatch was found.

Evidence: `/home/alexzhukov/cudavk-3ms-evidence/lazyclip/`.

## Bounded actual-target GPU differential

The subsequent RTX 5090 / CUDA 12.8 differential passed 14,224,896 paired
calls and 75,050,551,296 exact uint32 comparisons, including surviving NaNs
and the disputed FFMA operand ordering in the loaded SASS. The injected
one-bit error was detected. This is finite-corpus, target-specific evidence,
not universal proof, production JIT validation, or replay admission. See
`lazyclip/gpu-differential/REPORT.md` in the evidence directory for coverage
and limitations. Production-device equivalence, unchanged replay sentinels,
external full-frame validation and performance remain separate gates.
