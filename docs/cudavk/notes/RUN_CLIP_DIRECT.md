# RUN_CLIP_DIRECT prototype

This default-off successor to `CUDAVK_RUN_CLIP_BIN` is selected with
`CUDAVK_RUN_GEOMETRY=1 CUDAVK_RUN_CLIP_DIRECT=1`. The old dense tile arm is
still independently selectable with `CUDAVK_RUN_CLIP_BIN=1` for A/B.

`cp_run_clip_setup_direct` clips each run segment, applies the existing exact
`setup_triangle()` rules once, and compacts `{cp_tri_setup, stable primitive,
segment}`. Two fixed 512-block direct grids consume it. The first assigns one thread per
record for bboxes up to the existing tuned `CP_SMALL_THRESHOLD` (16 pixels).
The second assigns one warp per record above that split and its lanes stride
the full bounding box. Thus a
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

## Validation

`cp_run_clip_bin_gate` now compares the direct fused clip/setup records,
including all 80 setup bytes, with legacy clip followed by the shared setup
helper, then compares the full visbuf with the dense tile raster. It covers homogeneous clipping, cull, offset viewport/scissor, and the
capacity predicate. The full 81-test cudavk suite passes on RTX 5090.

The first one-warp-per-record direct prototype was rejected on the real
`favorite3` measurement (7.922/8.656 ms whole/heavy versus 6.335/7.258 ms for
tile+sideprep). The hybrid revision replaces that scheduling and has only a
completion smoke run so far. A proper alternating steady-state submit benchmark
is required; do not quote the old direct result for this revision.

The first proper hybrid favorite3 run was also rejected: **8.016 ms whole and
8.777 ms heavy**, with the exact stdout hash and 6,947 rows. Teardown reported
4,096.0 MiB of run storage plus 1,826.1 MiB of retained prepared outputs. The
classic 16-pixel split is a poor fit here because the compact-record warp path
pays much more state-load overhead than the existing queued raster. A 1,024
pixel split, matching the maximum work of one 32-pixel tile reference, is the
only remaining direct-list probe worth running before parking this arm.

The final 1,024-pixel thread/warp split was worse again at **8.698 ms whole and
9.662 ms heavy** on favorite3, with the exact hash and 6,947 rows. The direct
compact-list arm is therefore parked. Keep the tuned 16-pixel implementation
only as a reproducible negative control; the next viable shape is per-segment
side-stream pipelining, not another global threshold.

## Side-stream successor

The next prototype keeps the eight-stream overlap. After whole-run admission,
each segment uses its existing round-robin stream for exact prepare-only VS,
direct clip/setup into a private compact range, and hybrid visibility into the
shared cleared visbuf. Per-segment counters prevent append/consume races. A
main-stream broadcast orders descriptors and clears before continuation; one
join orders all partial visibility and the shared error word before the one
complete predicated classic fallback and grouped shade. Stable run-global IDs
make `atomicMin` winners independent of stream order. This path still needs
capture timing before it can be accepted.

Dense RUN_CLIP_BIN preparation is now forced serial. Its side-stream synthetic
gate produced an all-background image (1,382/4,096 pixels differed), while the
serial preparation and the new same-stream direct consumer were exact. The
older favorite captures did not expose that race.

The side-stream successor was exact on favorite3 but also rejected at **8.035
ms whole / 8.825 ms heavy**. A bounded Nsight Systems 2026.4.1 trace made the
failure specific: 556 `cp_run_clip_direct_raster` launches consumed 236.056 ms
(57.4% of all traced CUDA kernel time), while direct setup consumed 53.819 ms.
The 40 raster launches with `gridX=3` averaged **3.687 ms** and reached 4.569
ms. These are tiny primitive-count segments with large coverage, so compact
record count is the wrong scheduling proxy. Existing classic stage2/stage3
queue rasterization handles that shape far better. The next arm must reuse that
large-primitive machinery rather than revise direct-list thresholds again.

A deferred VS-to-clip PDL variant was built and then removed. All nine focused
gate offers took the tier-2 attribute, but a strict three-round favorite3 A/B
was disjointly worse: **+0.1343 ms whole / +0.1225 ms heavy**. It serialized
exact metadata/upload preparation on main before fan-out and did not recover
that cost through the dependency link. Do not retry it as a launch-only change.
