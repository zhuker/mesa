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

## Probe 3, answered: the design is worth +0.2 ms, and that closes the question

The merge must join the fan-out earlier, so the fan-out's value bounds its
cost. Measured on the current build, `CUDAVK_NO_OPAQUE_STREAMS=1`:

    fan-out is worth  0.871 ms whole / 1.100 heavy   (arms disjoint)

Against what the merge collects:

    gains  clip 0.47 + raster 0.60 idle share   +1.07 ms
    loses  the fan-out overlap it surrenders    -0.87 ms
    net                                         +0.20 ms

**Once clip and the raster stages are one launch each, there is nothing left
for eight streams to overlap.** The merge and the fan-out are collecting the
*same* idle time and cannot both have it. The frame would go 5.198 -> ~5.0,
against a 2.088 ms target.

## This explains all five failures with one mechanism

The idle share in clip and raster is **not waste** -- it is the price of the
overlap that already hides it. Every attempt in this investigation surrendered
overlap worth ~0.87 ms to collect a smaller portion of the same idle time:

| attempt | surrendered | collected | measured |
|---|---|---|---:|
| bin+walk 32 px | segment overlap | raster stages | +1.57 |
| bin+walk 16 px | segment overlap | raster stages | +2.09 |
| merged stage 3 | queue concurrency | stage-3 launches | +0.147 |
| depth-only episodes | -- | episode sharing | +0.089 |
| **full run-level merge (this design)** | **all of it** | **clip + raster idle** | **+0.20 projected** |

**So the "work floor is below the target" retraction was right about the
arithmetic and wrong about the conclusion.** The work really is only 1.638 ms.
But the difference between that and the 5.198 ms frame is not recoverable
idle: it is idle that eight streams are already overlapping, and any scheme
that collects it must first give it up.

**Final verdict: 4x is unreachable.** Not because the work forbids it, but
because the only remaining mechanism for closing the gap is self-cancelling,
and that is now measured rather than argued.

## The hybrid, priced too

A clip-only merge surrenders less overlap than the full one: the raster stages
keep fanning out per segment, and only the clip is gathered into one grid.

    gains  clip idle share                   +0.47 ms
    loses  clip's share of the fan-out       -0.17 ms   (19% of 0.871, clip's
                                                         share of the core pool)
    net                                      +0.30 ms

It is the best of the family, and it is still not close:

| variant | frame | vs native |
|---|---:|---:|
| today | 5.198 | 9.96x |
| 16 streams | 5.147 | 9.86x |
| **clip-only merge (best case)** | **4.896** | **9.38x** |
| full merge | 5.398 | 10.34x |
| **4x target** | **2.088** | **4.00x** |

Every variant of the last remaining mechanism lands between 9.4x and 10.0x.
None is within a factor of two of the objective, so none of them changes the
verdict -- they only choose how much of a ~0.3 ms improvement is worth a
pipeline rewrite.

## Superseded: what else to measure before building
## Probe 2, answered: capacity is not a blocker either

Today each queue set is sized to a **worst case**, not to observed use:

    nontrivial  CP_MAX_NONTRIVIAL 1,000,000 x 4 B  =  4.0 MB
    huge_tiles  CP_MAX_HUGE_TILES 2,000,000 x 8 B  = 16.0 MB
    per set                                          20.0 MB
    x CP_PASS_STREAMS (8)                           160.0 MB

Observed use is far smaller: the M0 census measures **~412,000 tile references
per frame**, across every episode and segment. An episode-wide queue sized
from that is **3.3 MB -- 49x smaller than the eight worst-case sets it
replaces**.

And the shape is not novel: `CP_MAX_OPAQUE_TILE_REFS` (2,000,000) is already
an episode-wide reference bound of exactly this kind, used by the tiled path.
So the merged design **reduces** queue memory rather than adding to it, which
is the opposite of the +307 MB that `DEAD_ENDS` 45 priced for the per-segment
alternative.

## Probe 3, answered: the design is worth +0.2 ms, and that closes the question

The merge must join the fan-out earlier, so the fan-out's value bounds its
cost. Measured on the current build, `CUDAVK_NO_OPAQUE_STREAMS=1`:

    fan-out is worth  0.871 ms whole / 1.100 heavy   (arms disjoint)

Against what the merge collects:

    gains  clip 0.47 + raster 0.60 idle share   +1.07 ms
    loses  the fan-out overlap it surrenders    -0.87 ms
    net                                         +0.20 ms

**Once clip and the raster stages are one launch each, there is nothing left
for eight streams to overlap.** The merge and the fan-out are collecting the
*same* idle time and cannot both have it. The frame would go 5.198 -> ~5.0,
against a 2.088 ms target.

## This explains all five failures with one mechanism

The idle share in clip and raster is **not waste** -- it is the price of the
overlap that already hides it. Every attempt in this investigation surrendered
overlap worth ~0.87 ms to collect a smaller portion of the same idle time:

| attempt | surrendered | collected | measured |
|---|---|---|---:|
| bin+walk 32 px | segment overlap | raster stages | +1.57 |
| bin+walk 16 px | segment overlap | raster stages | +2.09 |
| merged stage 3 | queue concurrency | stage-3 launches | +0.147 |
| depth-only episodes | -- | episode sharing | +0.089 |
| **full run-level merge (this design)** | **all of it** | **clip + raster idle** | **+0.20 projected** |

**So the "work floor is below the target" retraction was right about the
arithmetic and wrong about the conclusion.** The work really is only 1.638 ms.
But the difference between that and the 5.198 ms frame is not recoverable
idle: it is idle that eight streams are already overlapping, and any scheme
that collects it must first give it up.

**Final verdict: 4x is unreachable.** Not because the work forbids it, but
because the only remaining mechanism for closing the gap is self-cancelling,
and that is now measured rather than argued.

## The hybrid, priced too

A clip-only merge surrenders less overlap than the full one: the raster stages
keep fanning out per segment, and only the clip is gathered into one grid.

    gains  clip idle share                   +0.47 ms
    loses  clip's share of the fan-out       -0.17 ms   (19% of 0.871, clip's
                                                         share of the core pool)
    net                                      +0.30 ms

It is the best of the family, and it is still not close:

| variant | frame | vs native |
|---|---:|---:|
| today | 5.198 | 9.96x |
| 16 streams | 5.147 | 9.86x |
| **clip-only merge (best case)** | **4.896** | **9.38x** |
| full merge | 5.398 | 10.34x |
| **4x target** | **2.088** | **4.00x** |

Every variant of the last remaining mechanism lands between 9.4x and 10.0x.
None is within a factor of two of the objective, so none of them changes the
verdict -- they only choose how much of a ~0.3 ms improvement is worth a
pipeline rewrite.

## Superseded: what else to measure before building
1. **The join cost**: merging requires all segments' vertex and clip work
   complete before stage 2. The episode joins already, but earlier in the
   pipeline than today.

Each is a census or a microbenchmark, and each can refute the design before a
kernel is written -- which is the discipline `DEAD_ENDS` 40-43 established and
which this investigation abandoned when it started building.
