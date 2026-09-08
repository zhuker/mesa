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
  fit in `uint32_t`.

Each descriptor records the geometry group/member, triangle count, output-slot
count, and a stable `(prim_base, prim_slots)` range. `positions` stays zero
until grouped geometry has completed. Any refusal is side-effect free.


Capture validation reaches the preflight on every collected run: 3,069/3,069
for favorite3 and 4,284/4,284 for favorite2. Both exact-fallback replays pass
the expected stdout hash, submit-row count and 18/18 sentinels.

`cp_run_geometry_prepare_batch()` now exposes the exact current fused-VS
preparation for a collected batch, including multi-draw slices and binding
rows, and returns `{output, num_verts, out_stride}` from grow-only run storage.
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

`cp_run_clip_bin_fill` now consumes the compact live records after an
exclusive tile-count scan. It derives the local ID from the segment's
`prim_base`, calls `setup_triangle()` again, and scatters the existing
`cp_opaque_tile_ref` format through per-tile atomic cursors. A separate device
validation entry point checks every cursor against its count. Distinct error
bits report live-list bounds, segment and primitive bounds, reference capacity,
and cursor mismatch. Neither kernel writes visibility, and neither has a
renderer launch site. The eventual prefix scan and fallback decision remain
on-device; this slice adds no host readback synchronization.

`cp_run_clip_bin_gate.cu` compares the count kernel against the existing
per-segment `cp_clip_triangles` entry point followed by the shared setup helper.
It performs a trusted CPU exclusive scan, then compares exact per-tile sorted
reference multisets from the run fill against an existing fixed-stable-slot
reference fill. It checks compact live records, counts, cursors, stable IDs,
reference-capacity overflow, malformed segment bounds, and forced cursor
mismatch. Cases cover fully inside, homogeneous clipping, fully clipped, face
culled, multiple segments, and distinct viewport/scissor rectangles.

The renderer still takes the exact legacy path. It intentionally suppresses
the grouped VS launches as well: issuing grouped geometry and then the old
per-member clip/raster chains would be additive work and would not validate a
replacement. The CUDA entry point is registered but has no renderer launch
site until the transaction below is complete.

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
   tile counts with checked arithmetic and capacity-check the final sum. Launch
   fill/scatter over the compact live records. For each record derive its local
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
