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

This slice then takes the exact legacy path. It intentionally suppresses the
grouped VS launches as well: issuing grouped geometry and then the old
per-member clip/raster chains would be additive work and would not validate a
replacement. The focused gate checks identical pixels, the dependency, whole
run preflight, stable nonzero slot ranges, and absence of an additive grouped
launch.

## Exact continuation

1. After a successful whole-run preflight, launch every admitted geometry
   group and patch each descriptor's `positions` from its member span. If any
   group fails, discard the transaction and replay every member through the
   unchanged path.
2. Add a run-wide CUDA clip/setup/tile-count kernel. Reuse the existing
   seven-plane clip math and `CP_CLIP_PRIM_SHIFT` stable slots. It must consume
   the descriptors in original batch order and produce 16-pixel tile counts.
3. Prefix-scan the tile counts, capacity-check every sum, then run a fill/scatter
   pass which emits dense `(stable primitive ID, segment/draw row)` references.
   A failure before visibility writes must discard the transaction and replay
   the whole run exactly.
4. Feed those dense lists to the existing tile raster and grouped shade
   continuation. Only then remove each admitted member's old
   `cp_clip_rast_fused`/stage2/stage3 work. There must be no arm that performs
   both run-wide and old clip/raster work.
5. Gate with the full suite and native-driver image oracle before measurement.
   DEAD_ENDS 45 requires checking union-exclusive device time and the loss of
   stream fan-out; launch count alone is not a performance result.
