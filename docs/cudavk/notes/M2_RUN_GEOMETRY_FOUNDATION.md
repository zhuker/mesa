# M2 run-geometry first slice

This commit is the compiling, correctness-gated foundation for the CUDA-only
M2 geometry change. It is behind the default-off value flag
`CUDAVK_RUN_GEOMETRY`.

## What is implemented

- O run collection happens in `cp_batch_flush_defer_why()`. A deferrable batch
  break appends an immutable `cp_draw_batch` snapshot. It does not end a run.
- `cp_batch_flush_why()` is the authority for an ordering boundary. It drains
  the collected run before `cp_pass_finish()`. Refused draws drain the older
  run first and then use the unchanged renderer path.
- Only opaque batches admitted by `cp_opaque_appendable()` and an existing
  fused-fetch VS are collected. A VS with memory side effects is refused.
- At the boundary, batches are grouped by the actual fused `CUfunction`, index
  width, `firstInstance`, vertex-element count and bytewise vertex layout.
  These are the values which remain launch-wide in `cp_vertex_fetch_args`.
  VB/IB addresses, ranges, instance counts, UBOs and draw parameters are not
  in the key because existing slices and row tables already carry them.
- The member table and row bases are dynamically allocated. A run is not
  silently split at `CP_MAX_BATCH_DRAWS`; that limit applies only to each
  existing immutable batch snapshot.
- The upload lifetime is held until all collected snapshots have executed.
  Allocation failure and every admission refusal fail closed to the existing
  path. Existing batch order is preserved.

The focused gate runs the same three-triangle workload with
`CUDAVK_BATCH_MAX=1`, observes a real 3-to-1-or-better group, and compares the
full output hash against the flag-off renderer.

## Exact continuation

This slice deliberately does **not** claim a launch reduction yet. The grouped
plan is prepared, but draining still calls `cp_opaque_append()` once per saved
batch. The next patch must extract the fused-fetch VS preparation/launch block
from `cp_draw_execute_batch_inner()` into a helper which accepts the dynamic
flattened member table and output per member. It must then feed those
saved output spans into the current clip/raster/shade continuation.

That extraction also must:

1. disable the per-batch VS lane for grouped launches;
2. give geometry its own counters instead of using fused-fetch seeding for a
   single raster queue set;
3. allocate all group outputs monotonically until every member downstream has
   consumed them;
4. launch each group once, then submit downstream members in original opaque
   order; and
5. retain the current whole-run classic fallback before any side-effecting FS
   attempt.

Until that continuation lands, the flag is a correctness and grouping gate,
not a performance option.
