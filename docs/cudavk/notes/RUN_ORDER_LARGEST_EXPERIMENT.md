# Largest-first classic run issue experiment (CPU-only)

## Scope

Default-off `CUDAVK_RUN_ORDER_LARGEST=1` applies only to the existing
`CUDAVK_RUN_GEOMETRY=1 CUDAVK_RUN_CLIP_CLASSIC=1` side-stream transaction.
It does not enable that transaction itself. `CUDAVK_NO_RUN_PREP_STREAMS=1`
leaves the original serial order. Dense-bin and direct continuations are
unchanged. No GPU execution or performance claim was made for this change.

The static score is original input triangle count, descending, stable on ties.
The admitted fused triangle-list VS executes three vertices per triangle, so
this also predicts vertex invocation count. It is deliberately not a model of
shader complexity, clipping expansion, or pixel coverage. There are no capture
IDs, device changes, stream-count changes, or assignment changes.

## Safe seam and ordering audit

The permutation is constructed after `cp_run_clip_bin_reserve_prepared()` has
reserved every batch's output and descriptor storage, after the transaction
has fixed every clipped-output offset and primitive-reference range, and after
older episode work has finished. It contains original segment IDs, not moved
segment records.

Both VS prepare issue and classic raster issue use that same permutation.
All VS preparation is still issued first. Main then uploads metadata and
clears visibility, and the existing shared broadcast gates classic raster.
This is **not** the rejected deferred-VS-to-clip PDL arm from M2. Each classic
clear + fused clip/raster + stage2 + stage3 chain is issued intact. PDL policy
inside that unchanged chain is unchanged.

- Segment `s` still uses stream and queue set `s % CP_PASS_STREAMS` (eight).
  Sorted rank never chooses storage or stream identity. Reordering can change
  which segment on a shared stream runs first, but no queue reset can overtake
  that stream's previous stage3 consumer. This also holds above eight segments.
- Prepared outputs, draw-slice tables, clip rectangles, clipped outputs,
  primitive refs, primitive bases, and metadata remain indexed by original ID.
  `pass_segs`, its shade grouping, and final shade order remain untouched.
- Prepare-only disables the two-slot VS lane, fetch-fold queue seeding and
  downstream allocation/launches. Fused triangle lists need no temporary
  expanded input/vertex-ID/row arrays. Retained descriptors are span-owned.
- Collection-time uploads remain live while draining. Upload admission walks
  the actual permutation, including alignment, before any reordered launch.
  Thus reordering variable-sized upload blocks cannot silently invalidate the
  existing no-wrap proof. Stream switches flush pending copies before their
  consumer launches. Metadata upload/clear broadcast and the final side-stream
  join are retained.
- Admission is unchanged: native episode compatibility and order-point drains,
  `cp_opaque_appendable`, fused-fetch VS without memory writes, then classic
  preflight excluding FS writes, blending/discard, unsupported topology,
  framebuffer mismatch, strict/mixed depth functions, and multisampling.
  No application resource write moves across an application resource read:
  shader stores are refused and non-draw Vulkan order points drain the run.
  The remaining shared raster writes are the existing packed visibility/depth
  reduction with original primitive IDs. Color and attachment writeback remain
  in final shading, after the join. No new admission for attachment feedback
  or aliasing is introduced; this relies on the existing valid Vulkan episode
  contract, not a new allocation-range alias detector.

`CUDAVK_DEBUG_EPISODE=1` prints `run-order-largest:` followed by
`original-segment:triangles@stream-index`. Do not enable this for timing.

## CPU validation

- Full standalone Meson/Ninja ICD build, no GPU execution.
- `python src/cudavk/tests/cp_debug_doc.py --check`: pass.
- `python src/cudavk/tests/cp_no_getenv.py`: pass.
- `cc -std=c11 -Wall -Wextra -Werror -Isrc/cudavk
  src/cudavk/tests/cp_run_order_test.c -o /tmp/cp_run_order_test`
  then `/tmp/cp_run_order_test`: pass. Covers 0..64 segments, 200 score sets
  each, ties, uint32 maximum, permutation completeness, identity when disabled,
  and a host model of original queue ownership and wrap. The model is not a
  GPU memory-safety proof.

## Required next gate (parent owns GPU)

Control: RUN_GEOMETRY=1, RUN_CLIP_CLASSIC=1, RUN_ORDER_LARGEST=0.
Treatment: same binary and flags, RUN_ORDER_LARGEST=1.
Compare against **classic**, not against default renderer. A separate direct
measurement is required before claiming any improvement over default.

First check both captures' exact stdout hashes, 6,947/6,965 submit rows and
18/18 image sentinels; include a >8-segment episode to exercise queue reuse.
Then alternate arms in one session. Use paired-submit real frames and correct
loading/relevant/heavy bands from the critical REPORT, not the old M2 trace's
unvalidated frame denominator. Record first issue, last producer, join consumer
and terminal completion for each episode. Keep eight streams in both arms.
Reject if aggregate critical-tail movement is below roughly 0.2 ms or if only
API issue moves. A useful partial lead requires 0.3--0.5 ms real paired-submit
benefit on both captures, not an isolated producer speedup.

This experiment does not rebalance cumulative per-stream load. Fixed modulo
ownership may dominate its tail, and issuing all VS before the broadcast limits
what changing issue order can recover. The distinct append-path balancing
experiment is not a substitute for this probe.

References: DEAD_ENDS 45; M2_RUN_GEOMETRY_FOUNDATION.md;
M2_RUN_CLIP_BIN_STAGE.md (including rejected deferred PDL); offline
`/home/alexzhukov/cudavk-3ms-evidence/critical/REPORT.md`.
