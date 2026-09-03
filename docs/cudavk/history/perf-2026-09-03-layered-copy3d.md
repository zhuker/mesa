# One 3D copy for a layered image copy

A layered image copy is recorded one operation per plane, and the submit path
turns each of those into a `cuMemcpy2DAsync` plus a texture-cache
`cuEventRecord`. The 2026-09-02 analysis measured the render submit's first
**178 us** as exactly that: **32 two-kilobyte copies and 32 event records for
one 32x32x16 image**, issued onto an idle device. 64 KB of traffic behind 64
host API calls.

`cpvk_copy` now carries `slices`, `src_slice` and `dst_slice`, and the three
recording sites that loop over planes -- `vkCmdCopyImage2`,
`vkCmdCopyBufferToImage2` and `vkCmdCopyImageToBuffer2` -- emit one operation
for the whole run when the layout allows. `cpvk_execute_copy` issues it as a
single `cuMemcpy3DAsync`, and because it is one operation it also records one
texture-cache event instead of one per plane.

## When the layout allows it

`CUDA_MEMCPY3D` has no arbitrary plane stride: it derives one as
`pitch * Height`. So a merge is exact only where each side's plane stride is a
whole number of that side's rows, and that number covers the rows being copied.
`cpvk_copy_slices_fit()` is that test, and the per-plane loop stays as the
fallback for everything it refuses -- a rounded stride would silently copy the
wrong bytes.

What this driver lays out passes it: `level_size[l]` is
`row_stride[l] * h * d`, a 3D level's z stride is `row_stride * h`, and a
staged buffer image is `src_row * bufferImageHeight`. An application chooses
the buffer geometry, though, so the refusal is real code and not a comment.

The other condition is uniformity, and it is the caller's: a buffer-image copy
of several array layers *of several depth slices* has two different strides
interleaved on the image side, so that shape is never merged. A run of layers
at one depth, or of depth slices in one layer, is.

The bounds check in `cpvk_execute_copy` -- the one that stands between a wrong
region and a fault inside libcuda -- was extended rather than bypassed: the
reach of each side now includes `(slices - 1)` plane strides, and a merged copy
whose strides are not a whole number of rows is refused there too, so the
record-time test is not the only thing holding it.

## What was verified

- Both record loops were transcribed and compared against their previous form
  over layer/depth/3D combinations: the same (dst, src, width) byte triples in
  the same order, 32 operations becoming 1, and with
  `CUDAVK_NO_LAYERED_COPY3D=1` the operation list is identical field for field
  to the old one.
- A region whose plane stride is not a whole number of rows
  (`bufferImageHeight` below the copied height) falls back to 16 operations,
  unchanged.
- Builds clean; `cp_debug_doc.py --check`, `cp_no_getenv.py` and
  `cp_launch_audit.py` pass.

**Not measured.** Predicted 0.06-0.10 ms/frame (lead F-head). The A/B against
`CUDAVK_NO_LAYERED_COPY3D=1` and the sentinel compare -- which is what proves
the copied image, and matters more here than the median -- belong to whoever
owns the measurement session.
