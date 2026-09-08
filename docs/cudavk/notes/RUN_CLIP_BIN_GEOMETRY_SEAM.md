# RUN_CLIP_BIN geometry-preparation seam

`cp_run_geometry_prepare_batch()` is the correctness seam between collected
legacy batches and a future run-wide clip/bin continuation. It calls
`cp_draw_execute_batch_inner()` in a narrowly gated prepare-only mode. It does
not reconstruct the fused-fetch ABI. This matters for existing multi-draw
batches: their slice rows, index bases, vertex-buffer rows, UBO rows, draw
parameters, instance counts, and output layout come from the same code as the
legacy draw.

The returned tuple is `{output, num_verts, out_stride}`. `output` is one
collected batch's complete vertex stream, including all merged draws and
instances in their legacy order. The helper only admits the existing
side-effect-free fused-fetch lane. It does not broaden admission.

## Why this is a mode, not a second launcher

A naive early return from the old function is unsafe. Before its VS launch the
function normally clears visibility and depth, may seed downstream counters,
uploads raster rectangles, mutates fragment batch state, and may reserve clip
and shade work. The mode is viable only because every one of those effects is
explicitly gated and the return is directly after the ordinary VS launch.
Extracting that large preparation block would duplicate or parameterize its
many early exits and is more likely to drift. If more downstream preparation
moves above the VS in future, it must be added to the prepare-only exclusion;
otherwise the mode must be replaced by an extracted VS helper.

## Ownership and ordering

- Each collected batch slot owns a grow-only CUDA allocation. It survives all
  clip, bin, raster, and shade consumers of the run. It is reused only after
  the renderer's hard drain boundary has submitted those consumers. Context
  cleanup frees it. Growth can synchronize through `cuMemFree`; steady-state
  reuse does not allocate or free.
- Temporary fetch/id buffers remain device scratch. Sequential preparation on
  the main renderer stream may reuse them because launch order makes each VS
  consume its inputs before the next launch overwrites them.
- Collection-time push uploads remain protected by `batch_uploads_live` while
  `run_geometry.draining` is true. New slice and argument blocks use the normal
  upload ring. `cp_upload_end()` closes them before `cp_launch()` flushes the
  ordered copy and launch.
- Prepare-only mode never takes the VS side stream. A run-wide continuation
  receives outputs produced in main-stream batch order.

## Work deliberately omitted

Prepare-only mode does not clear the visibility or depth buffers. It does not
upload per-draw raster rectangles, mutate fragment batch tables, allocate clip
or shade scratch, or launch clip, raster, interpolation, fragment, or
writeback work. The existing one-draw grouped launcher remains separate and
may still share a launch under its current narrow launch key. The rejected
broad `active_output` replay is unchanged and is not extended to multi-draw
batches.

Counter folding is disabled while preparing. In the legacy path the fused VS
may seed raster queue and clip counters as an optimization. Those counters
belong to the downstream consumer and cannot be seeded for several prepared
batches before a run-wide continuation. Their values do not affect VS output;
the continuation must initialize its own counters once, in its own order.

## Admission and fallback

Run-wide descriptor and resource preflight must finish before the first call.
Every prepared VS is side-effect-free, writes only private run storage, and
seeds no downstream counter. Therefore a host-capacity refusal or
`CUDA_ERROR_OUT_OF_MEMORY`, even after earlier batches were prepared, may replay
the whole run through the exact legacy path. That duplicates exceptional VS
work but has no rendering or ordering effect.

That rule is deliberately narrow. A non-OOM CUDA allocation result, an upload
ring wrap whose context synchronization fails, an upload failure, a failed
`cuMemFree`, or a kernel launch failure latches `cp->device_fatal` through the
renderer's existing fatal callback. The drain then stops before exact replay,
and the native ICD reports its existing `VK_ERROR_DEVICE_LOST`; this layer does
not invent a Vulkan status. `CUDAVK_RUN_GEOMETRY_FAIL_ALLOC=1` gates the soft
OOM arm, `=2` gates a fatal allocation result, and
`CUDAVK_RUN_GEOMETRY_FAIL_LAUNCH=1` gates the fatal group-launch arm. Once the
future clip/bin path writes shared visibility or runs a fragment shader, even
OOM is no longer a valid legacy fallback.

This commit adds the seam but does not call it from the current
`RUN_CLIP_BIN` preflight. That path still has no run-wide continuation and
therefore stays on its exact fallback without speculative VS launches.
