# M2 run clip+bin structural slice

This default-off slice is selected by `CUDAVK_RUN_CLIP_BIN=1` and requires
`CUDAVK_RUN_GEOMETRY=1`. It does not claim a performance result.

## Implemented boundary

Before any grouped vertex launch, `cp_run_clip_bin_preflight()` validates the
entire collected opaque run and builds `cp_run_clip_bin_segment` entries in
Vulkan batch order. Admission is deliberately narrow:

- `MESA_PRIM_TRIANGLES` batches only; each descriptor records its draw count,
  so a later geometry-preparation step can retain the proven local VS for
  multi-draw batches while the clip/bin continuation becomes run-wide;
- at most 64 segments;
- vertex and fragment shaders have no memory side effects;
- vertex output slots do not exceed `CP_MAX_CLIP_SLOTS`;
- `triangles * CP_CLIP_MAX_OUT` and the run-global stable primitive-slot prefix
  fit below `CP_PRIM_ID_LIMIT`, the visibility ID/tag boundary.

Each descriptor records the geometry group/member, triangle count, output-slot
count, and a stable `(prim_base, prim_slots)` range. `positions` stays zero
until grouped geometry has completed. Any refusal is side-effect free.


Capture validation reaches the preflight on every collected run: 3,069/3,069
for favorite3 and 4,284/4,284 for favorite2. After the exact metadata and CUDA
fault-classification seams, both exact-fallback replays still pass the expected
stdout hash, 6,947/6,965 submit rows, and 18/18 sentinels (one alternating gate
round each; timing is not claimed from one round).

`cp_run_geometry_prepare_batch()` now exposes the exact current fused-VS
preparation for a collected batch and returns `{output, num_verts, out_stride}`
from grow-only run storage. The handoff now also retains the exact multi-draw
slice table, optional per-draw clip rectangles, primitive shift, and pre-clip
raster template in span-owned device storage. This closes the metadata seam;
the renderer still does not consume it.
Its prepare-only mode suppresses visibility/depth clears, FS state mutation,
counter seeding, clip/shade allocation, the VS lane and every downstream
stage. It is not called until the replacement continuation is transactional.
`RUN_CLIP_BIN_GEOMETRY_SEAM.md` records its lifetime and fallback contract.

`cp_run_clip_setup_tile_count` and its shared ABI now exist in the raster
module. One grid accepts at most 64 device segment descriptors. It invokes the
existing `cp_clip_one()` and `setup_triangle()` helpers, assigns stable IDs as
`segment.prim_base + triangle * CP_CLIP_MAX_OUT + output`, atomically counts
32-pixel bounding-box tile references, and compactly appends only
`{stable_prim, segment}` for live setup results. Append order is immaterial;
the stable ID is not an atomic allocation. The ABI has an explicit capacity
and overflow word. It does not allocate or clear an eight-slot setup array.

`cp_run_clip_bin_gate.cu` compares this run-wide kernel against the existing
per-segment `cp_clip_triangles` entry point followed by the shared setup helper.
It checks compact live records after sorting by stable ID, every tile count,
the implied reference total, and explicit overflow signalling. Cases cover
fully inside, homogeneous clipping, fully clipped, face culled, multiple
segments, and distinct viewport/scissor rectangles. The gate also passes
`compute-sanitizer --tool memcheck` with zero errors.

`cp_run_clip_bin_fill` consumes the compact live records after an exclusive
tile-count scan. It derives each segment-local stable ID, recomputes setup with
the shared helper, and scatters the existing `cp_opaque_tile_ref` format through
per-tile atomic cursors. A separate device kernel validates every cursor.
Distinct device error bits cover live, segment, primitive, reference-capacity,
and cursor failures. Both kernels return immediately when an earlier device
predicate is nonzero, and neither writes visibility. A one-block
`cp_run_clip_bin_check_counts` reduction checks the uint32 count array in
uint64 before the existing prefix scan, including the wraparound case. The gate
compares exact
per-tile reference multisets and exercises each refusal class.

An `sm_120` ptxas inspection reports 64 registers, a 5,152-byte stack frame
and no spills for the count kernel. The stack is the existing clipper's local
polygon storage (the standalone clip kernel is 5,120 bytes), so representative
run timing remains a required gate before integration is judged. The fill is
40 registers with no stack or spills; cursor validation is 14 registers, and
the checked total is 44 registers plus 2,048 bytes of shared memory.

The renderer now has a default-off transactional continuation. A refused run
exact-replays the legacy member path. An admitted run suppresses every legacy
per-member clip/raster chain. Proven one-draw groups use grouped VS and then
harvest exact ordinary-path metadata without a second VS; other batches use
the prepare-only fused VS. The count/check/scan/fill/validate chain feeds
`cp_opaque_tile_raster`, while the same error word predicates one run-wide direct fallback visibility
grid over the already clipped geometry. Clipping never repeats, exactly one
visibility arm does work, and the success path pays only one empty inverse-grid
launch instead of three launches per segment.

Admission is intentionally narrower than the CUDA ABI. It requires one sample,
one color attachment, a real common depth attachment, depth test+write, no
blend/discard/side effects, identical framebuffer identity, and a common
inclusive `LEQUAL` or `GEQUAL` function. Strict and mixed ordering refuse before
CUDA work. Grouped shading uses the complete saved batch key plus exact FS UBO
rows. Run clipped geometry, stable refs, dense lists, scan scratch and predicates
live in one grow-only allocation. `CUDAVK_RUN_CLIP_BIN_CAPACITY` can force the
device-predicated classic arm for testing; it is not a shipping capacity knob.

## Actual-kernel RTX probe

`cp_run_clip_bin_bench.cu` runs the real count, checked-total, scan, fill and
cursor kernels at 1280x720 with 32-pixel tiles. On the RTX 5090, 354,000 input
triangles split over 64 descriptors produced 354,000 live primitives and
382,320 references (1.080/input) in **0.0899-0.0900 ms** end to end after
warmup, including one contiguous fallback-ready clear of the 21.6 MiB stable
`prim_refs` table. Representative phases were 0.0119 ms clear, 0.0354 ms
clip/setup/count, 0.0042 ms checked total, 0.0043 ms scan, 0.0387 ms fill and
0.0023 ms validation (phase event boundaries add overhead, so do not sum them
as the separately timed total).

A synthetic clipping-heavy case emitted two primitives/input and took
**0.2888-0.2891 ms** (about 0.1614 count plus 0.1104 fill). It is not representative-capture
evidence and exceeds the 0.15 ms P1 budget, so capture clipping density must be
measured. The reduced benchmark passes Compute Sanitizer with zero errors.
The required B200 result is still unavailable.

## First committed-arm capture result

The first default-off end-to-end arm committed every collected run on both RTX
captures and kept both stdout hashes and submit-row counts exact: favorite3
3,069/3,069 commits and 6,947 rows; favorite2 4,284/4,284 and 6,965 rows.
This proves the transaction boundary, not the renderer: favorite3 regressed
from a 5.168 ms whole-run median and 6.004 ms heavy-band median to **24.343 ms**
and **31.992 ms** in a solo run. Kernel launches rose by exactly seven per run
(1,151,282 to 1,172,765), so launch submission is not the large regression.
The extra time appears in later waits for the GPU. Replacing the serial tile
walk with a 256-lane reference-stride walk over each triangle's true tile
intersection reduced that arm to **6.527 ms** whole and **7.284 ms** heavy,
with the same hash, 6,947 rows and 18/18 sentinels. Favorite2 also kept its exact
hash, 6,965 rows and 18/18 sentinels, but measured 6.400/6.230 ms whole/heavy against 4.848/4.586 ms control.
The new kernel uses 48 registers, no stack, spills, shared memory or barriers.
It removes the catastrophic algorithmic mistake, but favorite3 still loses to
the 5.168/6.004 ms control by 1.359/1.280 ms, and favorite2 by 1.552/1.644 ms.
This
matches the previously measured penalty for surrendering per-segment stream
overlap. Exact per-batch VS preparation was therefore moved onto the existing
eight side streams after whole-run storage and upload-ring admission, with one
join before count. On an immediate same-binary favorite3 serial/side pair it
improved whole time from 6.530 to 6.335 ms and heavy time from 7.296 to 7.258
ms. All 33,230 prepare launches fanned out and 3,069 joins completed. This is
useful plumbing but recovers only 0.195/0.038 ms and leaves the arm rejected.

A bounded one-second Nsight Systems 2026.4.1 CUDA trace confirms what remains
inside the parallel arm. Across 130 complete tile launches (131 count/fill
launches at the trace boundary), mean device duration was 0.222 ms for
clip/setup/count, 0.135 ms for fill and 0.684 ms for tile visibility; the
longest tile launch was 1.251 ms. These are per-kernel device durations under
CUPTI, not frame-time sums. Compiled shaders are all named `main`, so their
summary is not stage-attributable. The three named stages nevertheless show
that binning plus visibility still contributes about 1.04 ms of serial device
work per admitted run before shading.

A temporary synchronized density probe identified the immediate defect. Across
its first 401 favorite3 runs the median was 409,688 stable input slots, 24,173
live clipped primitives and 41,706 tile references; p90 was 171,475 references
and the maximum was 308,276. No capacity predicate fired. References/live was
only 1.34 median, but `cp_opaque_tile_raster` serially broadcasts each reference
and tests all 1,024 pixels in its 32-pixel tile. Those 401 runs therefore asked
for about 30 billion pixel/reference tests even though most triangles cover
very few pixels. The next gate is a parallel reference walk clipped to each
triangle's true tile intersection. The current serial tile kernel must not be
measured again as a candidate.

## Tile visibility compatibility boundary

The existing tile raster is a global depth reduction, while run collection is
broader and permits raster/depth state changes at deferred batch boundaries.
The first renderer transaction must therefore fail closed unless every member
has a real common depth attachment, depth test and writes enabled, and one
common depth function/direction. With the current packed tie, only `LEQUAL` and
`GEQUAL` are safe: equal depth selects the latest primitive. `LESS`/`GREATER`
require an explicit first-primitive tie rule before admission, and directions
must never mix. The unsafe opaque relaxation is not valid admission.

Per-segment viewport, depth transform, scissor, cull and rectangle tables stay
in each raster descriptor. Fragment shade groups must additionally split on
exact viewport/depth transform bits and `front_ccw`; using the first segment's
state for a mismatched group changes `gl_FragCoord`, interpolation or
`gl_FrontFacing`. The tile raster writes only disposable visibility state, so
ordered exact fallback remains legal until the first fragment/writeback launch.

## Transaction sequence and remaining gates

1. Whole-run host preflight completes before CUDA. It admits only the compatible
   depth/attachment subset above. A refusal selects unchanged batch-order replay
   and suppresses speculative grouped geometry.
2. One grow-only block reserves all clipped outputs, the contiguous stable-ref
   table, compact live records, dense refs, scan arrays and device predicates.
   Proven one-draw groups launch together. Every remaining batch uses exact
   prepare-only fused VS. Any OOM before shared work may replay; other CUDA
   faults latch device loss.
3. One uploaded segment table drives count. A checked uint64 reduction runs
   before the existing fused uint32 scan, so prefix wrap cannot expose invalid
   offsets. Fill and cursor validation observe the same error word.
4. Tile visibility runs only when that word is zero. One predicated run-wide
   direct grid consumes the complete clipped stable buffers only when it is
   nonzero. Both arms are enqueued without host readback. Only disposable
   visbuf changes before grouped fragment shading.
5. The standalone and sample gates are complete. Both real captures must now
   pass exact hashes, row counts and 18/18 sentinels on the committed arm before
   timing. Then measure strict alternating RTX A/B and union-exclusive device
   time. DEAD_ENDS 45 requires a substantial gain despite lost stream fanout.
   B200 correctness and timing remain mandatory.

## 2026-09-08 successor results

The compact direct-list successors did not recover the transaction. Favorite3
results, all with the exact 6,947-row stdout hash, were:

| arm | whole ms | heavy ms | verdict |
|---|---:|---:|---|
| global one-warp direct | 7.922 | 8.656 | rejected |
| global 16-pixel hybrid | 8.016 | 8.777 | rejected |
| global 1,024-pixel split | 8.698 | 9.662 | rejected |
| per-segment side-stream direct | 8.035 | 8.825 | rejected |

The side-stream arm also matched all 18 favorite3 image sentinels. Nsight made
the failure structural: 40 tiny-grid (`gridX=3`) direct raster launches averaged
3.687 ms because their few primitives covered large screen regions. Compact
record count was not a useful scheduling proxy.

Reusing the existing classic fused clip + stage2 + stage3 machinery inside the
same run transaction fixed that pathology and was exact, but it was neutral.
A strict three-round alternating favorite3 result was 5.1847 -> 5.2192 ms whole
(+0.0345) and 6.0361 -> 6.0009 ms heavy (-0.0352), with overlapping arms and
all stdout, row, and 18/18 sentinel gates passing. This misses M2's predeclared
1.0 ms heavy-band continuation gate. The dense bin, compact direct, and classic
continuation shapes are therefore evidence, not a shipping renderer.

A bounded NVTX trace of the classic arm split the otherwise ambiguous shader
kernel name `main`: vertex shader kernels were **0.773 ms/frame across 40.4
launches/frame**, while fragment shader kernels were **0.324 ms/frame across
25.9 launches/frame**. Normalized by the trace's 274 depth clears, all named
kernel time was about 3.8 ms/frame. This is profiling evidence, not an
end-to-end win; CUPTI changes host launch cost and the strict unprofiled A/B
above remains authoritative.

A follow-up audit closed run-wide VS launch merging as an independent lever.
The exact design would need a second LLVM-generated fused-fetch entry that
selects an unchanged per-batch 64-slot argument block and virtual block ID.
Reusing the loaded PTX `main` through a device call is not valid (`.entry`, not
`.func`) and repeats the rejected 20 -> 108-register device-link shape. At 40.4
VS launches/frame, even deleting every launch boundary is only 0.031-0.033
ms/frame before overlap; the complete VS pool (impossible to delete) is 0.773
ms/frame. Keep this only as a geometry seam if a larger accepted design needs
it, not as a performance item.

Deferring each exact VS until its segment continuation made all nine VS-to-clip
PDL offers take, but was rejected by strict A/B: +0.1343 ms whole and +0.1225
ms heavy, disjoint. The arm and its revert switch were removed; classic keeps
side-stream preparation followed by the shared broadcast.

## Post-M2 mechanism bounds

Two independent follow-ups checked whether a qualitatively different kernel
shape could reopen the missed gate.

- A static cooperative persistent clip/stage2/stage3 kernel is distinct from
  the old per-item claimant, but deleting those three kernels entirely removes
  only **0.5598 ms/frame** from the overlap-correct GPU union. Their actual
  adjacent inter-stage gaps are only **0.0422 ms/frame**. A scheduler which
  preserves their work cannot meet the 1.0 ms gate.
- Explicit fragment-shader continuations are technically formable by cutting
  NIR SSA and materializing live-across values, but the whole FS pool is only
  **0.729 ms/frame** union-exclusive; a 2x FS improvement is about 0.36 ms.
  One 32-word live record would also move roughly 540 MB/frame at the measured
  2.11M fragments. A general continuation runtime therefore cannot deliver a
  2x frame result.

The only defensible fragment follow-up is a diagnostics-only liveness census:
consider a hand-split single shader only if one straight-line hardware-fused
binary alone covers at least 0.5 ms and has a cut with no more than 16 scalar
words live across it. Current launch/resource census makes that admission very
unlikely. Do not build a general scheduler without first passing it.

The ceiling is reproducible from an exported report rather than copied from a
notebook:

```
./venv/bin/python src/cudavk/tests/cp_runlevel_ceiling.py \
  /tmp/nsys-runclip-classic-nvtx.sqlite --frames 274 \
  --require-union-ms 1.0
```

It prints 1.510225 ms/frame summed target duration, 0.559753 ms/frame
union-exclusive impossible deletion, and about 0.044 ms/frame of adjacent
inter-stage gaps, then rejects the 1.0 ms gate.

## Why separate persistent shader workers do not evade the call-image result

Keeping each shader in its own long-lived kernel preserves its register
isolation, but consumes the same finite resident-block budget. A typical
favorite FS block has 256 threads at about 96--154 registers, so two blocks
already consume most or all of an SM's 64K register file. Giving one shader the
current two-block/SM grid consumes roughly 340 resident blocks on the 170-SM
RTX and leaves no useful residency for the fixed-function workers. Splitting
those blocks across 13--18 shader identities can preserve aggregate shader
throughput only by consuming the same whole-machine budget; it cannot also
preserve the eight-way fixed-function overlap. Giving each identity only one
block uses just 13--18 blocks total and cuts available shader parallelism to
about 4--5% of the current grid.

Resident CTAs do not surrender their registers while polling. Therefore a
process-long pool either starves the other stages or underprovisions shading.
An on-demand pool must exit at the phase/run boundary to release resources and
is then ordinary per-shader launches with a queue protocol added. It can remove
at most a handful of the already sub-microsecond boundaries, far below the
recorded gates. Separate workers do not reopen the rejected combined-image
mechanism.
