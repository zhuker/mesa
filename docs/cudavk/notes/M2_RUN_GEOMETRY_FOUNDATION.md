# M2 run-geometry first slice

This commit is the compiling, correctness-gated foundation for the CUDA-only
M2 geometry change. It is behind the default-off value flag
`CUDAVK_RUN_GEOMETRY`.

## What is implemented

- Opaque run collection happens in `cp_batch_flush_defer_why()`. A deferrable batch
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

## CUDA execution in this slice

For each launchable triangle-list group, one existing
`CP_SHADER_EXEC_VS_FETCH` CUDA kernel is launched over dynamic flattened slices
and rows. A launchable group has at least two members and every member is an
existing one-draw batch. Its output is a single device allocation with one
saved span per source batch. The old per-batch VS launch is suppressed when
that batch is replayed. The span then continues through the current clip,
raster and shade code without changing its ordering.

The grouped launch does not seed a per-batch raster queue. Each downstream
member takes the existing explicit-clear arm instead. The VS lane is also
refused for these members because its two-slot lifetime is per batch. Group
storage remains live until `cp_pass_finish()` has issued all consumers.

Unsupported topology, missing fused execution, VS memory side effects, an
allocation error, or a CUDA launch error leaves that member/group on the
existing renderer. A grouped VS is side-effect-free, so a partially prepared
run can safely fall back before downstream fragment execution.

## Known limits

- Only `MESA_PRIM_TRIANGLES` one-draw members use the grouped launch.
  Topologies needing host-expanded vertex/instance ID arrays use the old path.
  Existing multi-draw batches also stay on their proven fused launch. A broad
  version passed stdout hashes but reproducibly corrupted 38,376 pixels in
  favorite3 sentinel frame 1400; refusing multi-draw members restored 18/18
  exact sentinels. The broad path must not be re-enabled without finding that
  downstream replay defect.
- Group argument blocks use the existing ordered upload ring. Output storage
  is context-owned and grow-only. Output slots are reused after the hard
  boundary has issued all old consumers; main-stream order plus the opaque
  side-stream join prevents overwrite races. `cuMemAlloc`/`cuMemFree` occurs
  only when an output slot grows and at cleanup, not once per group or run.
- The current clip/raster/shade path still runs once per saved batch. This
  slice removes only fused VS launches; run-global clip/bin/walk/compaction is
  later M2 work.
