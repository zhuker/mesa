# M2 run clip+bin structural slice

This default-off slice is selected by `CUDAVK_RUN_CLIP_BIN=1` and requires
`CUDAVK_RUN_GEOMETRY=1`. It does not claim a performance result.

## Implemented boundary

Before any grouped vertex launch, `cp_run_clip_bin_preflight()` validates the
entire collected opaque run and builds `cp_run_clip_bin_segment` entries in
Vulkan batch order. Admission is deliberately narrow:

- one-draw `MESA_PRIM_TRIANGLES` batches only;
- at most 64 segments;
- vertex and fragment shaders have no memory side effects;
- vertex output slots do not exceed `CP_MAX_CLIP_SLOTS`;
- `triangles * CP_CLIP_MAX_OUT` and the run-global stable primitive-slot prefix
  fit in `uint32_t`.

Each descriptor records the geometry group/member, triangle count, output-slot
count, and a stable `(prim_base, prim_slots)` range. `positions` stays zero
until grouped geometry has completed. Any refusal is side-effect free.

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
It checks compact live records after sorting by stable ID, every tile count and
the implied reference total. Cases cover fully inside, homogeneous clipping,
fully clipped, face culled, multiple segments, and distinct viewport/scissor
rectangles.

The renderer still takes the exact legacy path. It intentionally suppresses
the grouped VS launches as well: issuing grouped geometry and then the old
per-member clip/raster chains would be additive work and would not validate a
replacement. The CUDA entry point is registered but has no renderer launch
site until the transaction below is complete.

## Exact continuation

1. After a successful whole-run preflight, launch every admitted geometry
   group and patch each descriptor's `positions` from its member span. If any
   group fails, discard the transaction and replay every member through the
   unchanged path.
2. Patch and upload one `cp_run_clip_bin_device_segment` per host descriptor.
   Its `clip` half points at the grouped position span plus per-segment clipped
   output/ref storage. Its `rast` half copies the legacy viewport, scissor,
   cull, and slot state. Clear only the tile counters, live counter and overflow
   word, then launch `cp_run_clip_setup_tile_count` as `grid.y = num_segments`.
3. Check overflow without allowing visibility writes. Prefix-scan all 32-pixel
   tile counts with checked arithmetic and capacity-check the final sum. Launch
   fill/scatter over the compact live array. For each record derive the local
   stable ID from `stable_prim - segment.prim_base`, call the existing
   `setup_triangle()` again, enumerate its recorded-equivalent tile bounds,
   and scatter dense `(stable primitive ID, segment/draw row)` references using
   the scanned offsets and per-tile cursors. Validate every cursor equals its
   count. Any allocation, launch, overflow, sum or cursor failure discards the
   whole transaction and replays the unchanged run before fragment effects.
4. Feed those dense lists to the existing 32-pixel tile raster and grouped shade
   continuation. Only then remove each admitted member's old
   `cp_clip_rast_fused`/stage2/stage3 work. There must be no arm that performs
   both run-wide and old clip/raster work.
5. Gate with the full suite and native-driver image oracle before measurement.
   DEAD_ENDS 45 requires checking union-exclusive device time and the loss of
   stream fan-out; launch count alone is not a performance result.
