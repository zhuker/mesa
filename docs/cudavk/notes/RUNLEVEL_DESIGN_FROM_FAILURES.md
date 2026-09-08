# The run-level pipeline, designed backwards from five failures

`FOURX_DIAGNOSIS_2026-09-05.md` establishes the target is **above** the work
floor: 1.638 ms of real work against a 2.088 ms budget. The frame is dominated
by machine underutilisation -- clip and rasterisation run at roughly 10% work
share -- so 4x is an engineering problem, not a physical limit.

Five partial attempts at collecting that idle time all measured slower. Each
failed for a specific, diagnosed reason, and together they constrain the design
tightly enough to be worth writing down.

## What the failures rule out

| attempt | result | what it proves |
|---|---|---|
| bin+walk, 32 px tiles | +1.57 / +1.80 | replacing the raster **algorithm** loses to per-block cost |
| bin+walk, 16 px tiles | +2.09 / +2.51 | the walk is per-tile/per-block bound; smaller tiles are worse |
| depth-only episodes | +0.089 / +0.179 | more, shorter episodes cost more setup than they save |
| merged stage 3 (v1) | device lost | **8 queue sets vs 12.84 segments**: an episode wraps its queues |
| merged stage 3 (v2, capped) | +0.147 / +0.208 | draining per-segment queues **in turn** serialises what the fan-out overlapped |

Two constraints fall out, and both must hold simultaneously:

1. **Keep the existing kernels.** Every attempt to replace the rasterisation
   algorithm lost. The kernels are efficient per unit of work; they are simply
   starved.
2. **Never serialise what the fan-out overlaps.** Any merged form must be one
   *parallel* grid, not one launch looping over segments.

## The design those constraints force

The failures share a root cause: **per-segment queue sets**. They bound how
late work can run (constraint from attempt 4) and they force sequential drains
(attempt 5). Remove them and both problems go.

**One episode-wide queue set with segment-tagged entries.**

- The clip is data-parallel over triangles, so **one grid over all segments'
  post-transform triangles** fills the machine. Each entry it appends carries
  its segment index, exactly as `cp_opaque_tile_ref` already does.
- Stages 2 and 3 then run **once per episode** over that single queue,
  indexing `rast_args[entry.segment]`. This is parallel, not a loop: the
  serialisation that killed attempt 5 came from *separate* queues, not from
  merging itself.
- Memory falls rather than rises: one queue set per episode instead of eight
  per stream (`DEAD_ENDS` 45 priced per-segment sets at +307 MB; this needs
  none of it).

**Ordering.** Segments' vertex stages still fan out and are joined before the
merged clip, which the episode already does at `cp_opaque_finish`. Visibility
writes stay `atomicMin` on globally numbered primitives, so merging changes no
result -- the property that made every correctness check pass in attempts 4
and 5.

## What it should be worth

Collecting the idle share of clip (0.47) and raster (0.60) takes the core from
2.698 to ~1.63 ms. With the frame boundary unchanged that lands the whole
window near **2.0-2.6 ms**, i.e. **3.8-5x native** -- the first configuration
in this investigation whose ceiling reaches the objective rather than falling
short of it.

## Probe 1, answered: no tagging is needed at all

The queue entries do not need a segment index, because **they already carry
one implicitly**. A `cp_tile_pair` holds `tri_id`, and inside an episode that
id is **episode-global** (`abuf_prim_base` offsets each segment's primitives
into a shared numbering, `CP_PRIM_ID_LIMIT` = 2^30). The driver already
resolves a segment from a global id:

    cp_resolve_seg_range(args, gprim)          cp_fs_interp.h:267
      binary-searches struct cp_seg_range[]    sorted by prim_base
      and rewrites positions, prim_refs, draw_slices, prim_shift,
      abuf_prim_base and row_base for that segment

`struct cp_seg_range` (`cp_rast_types.h:898`) carries exactly the per-segment
state stages 2 and 3 would need, and `cp_fs_compact` already calls this per
primitive on the blended path. So an episode-wide queue needs **no format
change, no widening, and no extra bandwidth** -- only that stages 2 and 3
resolve their segment the same way the fragment path already does.

That removes the largest risk in the design and the only one that would have
cost bandwidth on the hottest array in the frame.

## What else to measure before building
1. **One episode-wide queue's capacity**: today's sets are sized per segment;
   an episode of 12.84 segments needs the sum, and `CP_MAX_OPAQUE_TILE_REFS`
   already bounds a comparable array.
2. **The join cost**: merging requires all segments' vertex and clip work
   complete before stage 2. The episode joins already, but earlier in the
   pipeline than today.

Each is a census or a microbenchmark, and each can refute the design before a
kernel is written -- which is the discipline `DEAD_ENDS` 40-43 established and
which this investigation abandoned when it started building.
