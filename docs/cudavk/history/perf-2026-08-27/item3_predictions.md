# Item 3 - stage3 stall attribution: PREDICTIONS REGISTERED BEFORE THE NCU RUN

Written before any NCU source-level output was collected. Source of
`cp_rasterize_stage3_body` and `cp_rast_types.h` was read first; the profiler
was not.

## The facts I am starting from (briefed, not re-measured yet)

`cp_rasterize_stage3` and `cp_rasterize_stage3_abuf` both read **1.00
sectors per request** on global loads while stalling **70.7%** and **56.3%** on
`long_scoreboard`, with **20.4** and **4.5** warps in flight. Twenty warps is
not starvation, so width cannot hide the latency; the briefed remedy is wider
requests, not coalescing.

## What the source says the candidate loads are

1. `huge_queue[tile_idx].tri_id / .tile_x / .tile_y` - `struct cp_tile_pair`,
   8 bytes (u32 + u16 + u16), read by **EVERY thread of the block with the
   same index**. A uniform broadcast.
2. `cache[setup_idx]` - `struct cp_setup_cache_entry`, **84 bytes**
   (u32 tri_id + 80-byte `cp_tri_setup`), read by **`threadIdx.x == 0` only**.
3. `setup_triangle(&args, tri_id, &s)` - the miss path, also
   **`threadIdx.x == 0` only**, gathering vertex attributes for one primitive.
4. `visbuf[]` per-pixel traffic in the fragment loop, which is the only
   genuinely per-lane addressed stream in the kernel.

## PREDICTION P1 (primary)

The 1.00 sectors/request is **not** an AoS-stride pathology. It is the
signature of loads issued by a **single lane or as a uniform broadcast**: one
active thread per request means one sector per request by construction, and
that is already the optimum for such a request. I predict the source-level
attribution puts the majority of the 1.00-sector requests on the
`threadIdx.x == 0` setup path (candidates 2 and 3) and on the uniform
`huge_queue` read (candidate 1), NOT on a strided per-lane walk of an
array-of-structures.

## PREDICTION P2

Because a block's whole setup is fetched by one lane and the other lanes wait
at `__syncthreads()`, the `long_scoreboard` stall will concentrate on **few
instructions with very high per-instruction stall** rather than spread over
the pixel loop. I predict the top 3 source lines carry >50% of stage3's
`long_scoreboard`.

## PREDICTION P3

`cp_setup_cache_entry` is 84 bytes and **not 16-byte aligned as an array
element** (84 = 4 + 80, stride 84). So `cached = cache[setup_idx]` cannot be
compiled to `LDG.128` for every field. I predict the SASS shows a run of
`LDG.E` / `LDG.E.64` and at most opportunistic 128-bit loads, and that padding
the entry to 96 or 112 bytes is the cheap legality fix if width is the answer.

## FALSIFIERS, registered

* **X1 - "one hot record type, split it and win".** If >60% of the 1.00-sector
  requests come from ONE structure read with a per-lane index and a stride
  equal to `sizeof(struct ...)`, then it is a true AoS problem, SoA on the
  producing stage is legal, and the item is a contained change. I predict this
  is FALSE.
* **X2 - "inherent to the algorithm, this is a rewrite".** If the requests are
  dominated by the single-lane setup fetch and the per-pixel `visbuf` access,
  no load-width change helps: the first is already minimal per request and the
  second is addressed by pixel. Then the remedy is cooperative loading of the
  80-byte setup across lanes, or hoisting setup out of stage 3, both of which
  are structural. I predict this is TRUE.
* **X3 - the broadcast trap.** If the `huge_queue` read dominates, note that a
  uniform 8-byte broadcast at 1.00 sectors/request is ALREADY optimal and the
  metric is being misread. Reporting it as a defect would be an error.
* **X4 - width is illegal anyway.** If the fields a thread needs are not
  contiguous (e.g. `sx0..sy2` used together but `ix_min..iy_max` used in a
  different branch), a wider load fetches bytes the thread discards and the
  sector count falls while the byte count rises. Must be checked before any
  claim.
* **X5 - the two kernels disagree.** `stage3` has 20.4 warps in flight and
  `stage3_abuf` 4.5. If the attribution is the same in both, the mechanism is
  the setup path (occupancy-independent). If it differs, the abuf variant's
  stall is a different phenomenon and must not be pooled with stage3's.

## SIZING RULE, registered before the number exists

Any prospective gain is to be sized through **EXCLUSIVITY, never kernel time**.
The raster chain at 2x is **0.752 ms/frame** of union busy removed and at
infinite speed **2.044 ms/frame**. A change that halves stage3's stall cycles
is worth at most its share of the 0.752, not its share of its own duration.
