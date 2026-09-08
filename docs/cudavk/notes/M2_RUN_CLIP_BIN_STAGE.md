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
for favorite3 and 4,284/4,284 for favorite2. Both exact-fallback replays pass
the expected stdout hash, submit-row count and 18/18 sentinels.

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
exact-replays the legacy member path. An admitted run intentionally suppresses
all legacy per-member clip/raster chains. Proven one-draw groups use the grouped
VS and then harvest exact ordinary-path metadata without a second VS; other
batches use prepare-only fused VS. The run count/check/scan/fill/validate chain
feeds `cp_opaque_tile_raster`, while the same error word predicates one
run-wide queue-free ordinary visibility grid over the exact per-segment raster
arguments and already clipped stable geometry. It is exactly one 256-thread
block, so a successful transaction schedules one no-op CTA rather than up to
64K predicate-only CTAs or three launches per segment. Thus clipping
never repeats and exactly one visibility arm writes.

Admission is intentionally narrower than the kernel ABI. It requires one
sample, one color attachment, a real depth attachment, depth test+write, no
blend/discard/side effects, identical framebuffer identity, and a common
inclusive LEQUAL or GEQUAL function. Strict LESS/GREATER and mixed depth
ordering refuse before CUDA work because the current tile winner tie rule is
not equivalent. Grouped shading uses the complete batch key plus exact FS UBO
rows. Run clipped geometry, stable primitive refs, dense lists, scan scratch,
and predicates live in grow-only storage. Every segment reserves its complete
`input triangles * CP_CLIP_MAX_OUT` stable range before clipping starts, and
live capacity is the sum of those ranges. Therefore an admitted run cannot set
the predicate with partial clipped output: reachable overflow/cursor refusals
happen only after the exact stable geometry is complete. Primitive refs are
cleared before count so holes are safe for the predicated inverse arm.

## Exact continuation

1. After a successful whole-run preflight, launch only the proven groups of
   one-draw members. Prepare existing multi-draw batches with an extracted
   form of their exact current fused-VS path; do not route their grouped output
   through the known-bad per-member replay. Patch every descriptor's
   `positions` only after all geometry preparation succeeds. On any failure,
   discard the transaction before clip/bin and replay every member unchanged.
2. Patch and upload one `cp_run_clip_bin_device_segment` per host descriptor.
   Its `clip` half points at the prepared position span plus per-segment clipped
   output/ref storage. Its `rast` half copies the legacy viewport, scissor,
   cull, and slot state. Clear only tile counters, the compact-live counter and
   overflow word. Launch the count kernel with `grid.y = num_segments`.
3. Check overflow without allowing visibility writes. Prefix-scan all 32-pixel
   tile counts with checked arithmetic and capacity-check the final sum. The
   current uint32 scan is insufficient by itself because its prefix can wrap;
   run `cp_run_clip_bin_check_counts` first so its checked uint64 total sets the
   shared predicate. Then launch fill/scatter over the compact live records. For each record derive its local
   stable ID, call `setup_triangle()` again, enumerate its tile bounds and
   scatter dense `(stable primitive ID, segment)` references. Validate every
   cursor equals its count. Any failure discards the whole transaction before
   fragment effects and replays the unchanged run.
4. Feed those dense lists to the existing 32-pixel tile raster and grouped shade
   continuation. Only then remove each admitted member's old
   `cp_clip_rast_fused`/stage2/stage3 work. There must be no arm that performs
   both run-wide and old clip/raster work.
5. Gate with the full suite and native-driver image oracle before measurement.
   DEAD_ENDS 45 requires checking union-exclusive device time and the loss of
   stream fan-out; launch count alone is not a performance result.
