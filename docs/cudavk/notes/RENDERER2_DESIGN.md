# Renderer 2: scope-level tile-binned shading — design charter

Branch `redesign/tilewalk`, worktree `mesa-renderer2`, base `cudavk-0.0.1`
(`150978fd192`). Status: **design/prototype**. Nothing here is a performance
claim; every milestone below carries its own measured gate.

## Why a rewrite, and why now

The iteration campaign is exhausted on both validated GPUs: every mechanism
with a defensible >=0.500 ms/frame ceiling has been measured and either landed
or refuted (`DEAD_ENDS.md`, `RESIDUAL_AUDITS_2026-08-31.md`). What remains is
the *structure* of the current renderer:

- 414-740 kernel launches per frame, most of them small and latency-bound.
  Heavy frames are heavy because they carry ~2.7x more launches, not bigger
  kernels (post-pin heavy-band trace, 1,011 frames).
- The per-batch chain (VS -> clip -> raster stages -> FS -> writeback) pays
  its structure per batch; batches exist because host-side state changes force
  flushes, and the batch population is capture-shaped (median tiny).
- The device union alone in the RTX heavy band is 5.21 ms/frame; the 5.0-ms
  goal cannot be met there without *removing device work*, and per-kernel work
  removal is measured-exhausted. The workload is micro-triangles: mean edge
  ~1.7 px, ~1.07 tile references per triangle, and the hot tile holds 26,297
  triangles resolving 563 visible pixels (entry 24 census). **Depth
  complexity, not coverage, is the load.**

A scope-level renderer attacks exactly the two things iterations cannot: the
launch structure (one binning + one shade pass per render scope instead of
per-batch chains) and the depth complexity (per-tile front-to-back walk with
early termination).

## The dead ends this must not repeat, and how it differs

This charter is written against `DEAD_ENDS.md`; the entries below are binding
constraints, not history.

| entry | what died | what renderer 2 does differently |
|---|---|---|
| 2 | persistent/dynamic-claim scheduler: +140.15 ms scheduler cost | no persistent kernel, no dynamic claiming. Tiles are a static grid; a tile's work list is built by a binning pass and walked by one block. Scheduling is the grid launch, once per scope |
| 10 | kernel-internal redesign of stage3_abuf: 0.15 waves/SM | renderer 2 does not touch the existing chain; it replaces the chain's *existence* for scopes it covers |
| 22 | the merge rule: launch-removal credit only where launches were serial | the credit claimed here is not launch removal; it is depth-complexity removal (occluded shading skipped) plus the structural end of per-batch state. Launch arithmetic is reported but not banked |
| 24 | opaque sort-middle tiling: one-block barrier signature; surrendered the +2.73 ms fan-out; no occlusion | the prototype's two named retry conditions are the design's two pillars: (1) per-tile occlusion exists (front-to-back walk over a per-tile depth ceiling with early-out once coverage is full), (2) fan-out is not surrendered piecemeal - the tiled path takes whole scopes, and tiles supply the parallelism the segment fan-out supplied (a 1280x720 frame at 16px tiles is 3,600 independent blocks vs 8 side streams). The one-block-sets-duration failure is answered by binning *before* raster (no thread-0 setup_triangle inside the walk; setup is a separate flat pass) |
| 30 | episode-entry stale-depth Hi-Z: 0.103 ceiling | that measured *cross-episode* reuse of stale depth on the existing chain. Renderer 2's occlusion is *intra-scope and current-frame*: the walk consumes depth it computes, in order |
| 33 | retained cross-frame geometry: 0.206 collectible | nothing is retained across frames; binning is per scope, per frame |
| 15/28-item-3 | writeback fusion: >1 claimant per pixel per launch | the per-tile walk owns its pixels exclusively (a tile belongs to one block), so the claimant invariant holds *by construction* - this is the shape under which fusing shade+write is legal |

## Architecture

One flag, `CUDAVK_RENDERER2` (value bool, default off), selecting per render
scope. Scopes the new path cannot express fall through to the existing
renderer unchanged - correctness never depends on coverage.

Per render scope (the existing episode boundary):

1. **Geometry pass** (existing, unchanged): the generated VS already lands
   positions/attributes in device buffers per batch. Renderer 2 keeps the
   fused VS exactly as is - shader content and vertex math are untouched.
2. **Bin pass** (new, one kernel per scope): walk the scope's post-clip
   triangles (all batches), emit (tile, triangle-ref, draw-order-key) into
   per-tile lists. Grid is flat over triangles; no barriers; the entry 24
   prototype's shared-memory staging bottleneck does not exist here.
3. **Tile walk** (new, one kernel per scope): one block per non-empty tile.
   Opaque: front-to-back over the tile list with a running coverage/depth
   mask, shade only survivors, stop when the tile is saturated. Blended: the
   list is walked in draw order into the existing A-buffer format (or shaded
   directly when a tile's list is short). Per-draw state (pipeline,
   descriptors, push constants) is indexed from a scope-level state table
   built at record time - state changes become table rows, not flushes.
4. **Writeback** fused into the walk (legal here; see table above).

Determinism: within a tile, opaque resolve order is depth then draw order,
and blending walks draw order exactly - output must be bit-identical to the
existing renderer on the sentinel set. This is gate M2, not an aspiration.

## What is honestly at risk

- The FS is `main` with per-batch specialization; a tile walk crossing draws
  needs either per-draw indirect calls (slow), sorted shading buckets per
  tile, or one specialized walk per pipeline over its tiles. This is the
  hardest open design problem; M1 exists to size it before any kernel is
  written.
- Entry 24's census was one sample plus one capture window; M0 re-runs the
  census on both captures at current HEAD before believing its shape.
- Blended scopes may show little depth-complexity win (transparency shades
  everything it keeps); the win there is structural only, and may be small.
- If the scope state table cannot express some state kind, coverage shrinks;
  the fallback keeps correctness but the win shrinks with it. Coverage is
  reported per run from M2 on.

## Milestones and gates

| # | deliverable | gate (measured, no vibes) |
|---|---|---|
| M0 | tile/depth-complexity census on favorite2+favorite3 at HEAD (fix the unreachable `cp_tile_census_end_pass` reducer noted in entry 24 on this branch only) | census reports non-empty on both captures; distributions recorded in this directory |
| M1 | state-table + shading-bucket design doc sized against M0 data (pipelines per scope, draws per scope, refs per tile) | design review against every table row above; explicit go/no-go |
| M2 | opaque tile walk behind the flag for the simplest expressible scope class, everything else falls through | bit-identical sentinels + hashes on both captures with the flag ON; 79/79 suite |
| M3 | coverage growth (multi-pipeline scopes, blended-into-A-buffer) | same correctness gates at each step; coverage % reported |
| M4 | first honest A/B on both captures, both GPUs | alternating sessions, counts/hashes/sentinels; accept only if the whole-run median improves |
| M5 | default decision | same bar as every landed default: wins on both captures or reverts cleanly |

A milestone that misses its gate stops the project at that milestone; the
branch stays as evidence either way. Estimated scale is multiple sessions;
nothing lands on `cudapipe-vk-native` before M4 passes.
