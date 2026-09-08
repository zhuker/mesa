# RUN_CLIP_DIRECT prototype

This default-off successor to `CUDAVK_RUN_CLIP_BIN` is selected with
`CUDAVK_RUN_GEOMETRY=1 CUDAVK_RUN_CLIP_DIRECT=1`. The old dense tile arm is
still independently selectable with `CUDAVK_RUN_CLIP_BIN=1` for A/B.

`cp_run_clip_setup_direct` clips each run segment, applies the existing exact
`setup_triangle()` rules once, and compacts `{cp_tri_setup, stable primitive,
segment}`. `cp_run_clip_direct_raster` is one fixed 512-block grid. Each warp
claims compact records and its lanes stride the full bounding box. Thus a
fullscreen triangle is slow rather than unsafe: there is no one-thread cutoff,
tile queue, dense-reference capacity, or cooperative-launch requirement.

The path preserves the shared clip, viewport/scissor, cull and top-left helpers.
It tests the old depth attachment but writes only the disposable visibility
buffer. `PACK_VISBUF` complements the stable primitive ID, so equal depth under
LEQUAL or GEQUAL selects the latest stable primitive. The normal shade path
performs the first real depth write.

All clip output and the worst-case compact array are preallocated before the
transaction. A setup capacity or descriptor error sets the existing device
predicate. The direct raster then refuses, or may leave partial `atomicMin`
writes if a bad compact record is found concurrently; the predicated classic
arm rasterizes the complete clipped superset on top. No host readback is used.
CUDA API failures use the existing fatal-device classification.

## Side-stream transaction prototype

The successor keeps the eight-stream overlap instead of collecting clip and
visibility on main. After whole-run preflight, grow-only prepared-output
reservation, and an upload-ring no-wrap proof, each segment uses its existing
round-robin pass stream. Its ordered chain is exact prepare-only fused VS,
direct clip/setup compaction into a private compact range, then direct
visibility into the shared cleared visbuf. Small bounding boxes use one thread;
larger boxes use one warp, within the same visibility launch. Stable run-global
primitive IDs make all `atomicMin` winners independent of stream order.

The main stream owns the visbuf/depth clears and uploaded segment tables. A
post-clear broadcast orders those shared inputs before every side continuation.
Per-segment live counters avoid append/consume races. One join orders all
partial visbuf writes and the shared error word before the complete
run-predicated classic fallback and grouped shade. Therefore a device refusal
can leave disposable partial visibility, but cannot reach depth or color.
CUDA launch/event failures after fan-out remain fatal; there is no host
readback or partial host replay.

## Validation

`cp_run_clip_bin_gate` now compares the direct fused clip/setup records,
including all 80 setup bytes, with legacy clip followed by the shared setup
helper, then compares the full visbuf with the dense tile raster. It covers homogeneous clipping, cull, offset viewport/scissor, and the
capacity predicate. The full 81-test cudavk suite passes on RTX 5090.

A smoke execution of the 3,473-frame `favorite2` compiled capture completed on
RTX 5090 for both A/B arms. First-process wall times (and therefore not accepted
frame benchmarks) were 62.92 s for the dense bin arm and 26.43 s for direct.
They include shader compilation and teardown. A proper alternating steady-state
submit benchmark remains required before accepting the prototype.
