# Renderer2 verdict: NO-GO, and the ceiling is below the goal

All three M1 risks are now measured. The walk works; the milestone does not.
This document closes the project at M1 rather than writing M2.

## The three risks, settled

| risk | result | document |
|---|---|---|
| 1. fan-out surrender | **0.66-0.84 ms/frame** | `RENDERER2_M1_FANOUT.md` |
| 2. opaque prefix empty | confirmed empty (0.0% of refs at p50); slice redefined to any opaque run, restoring 95-96% of refs | `RENDERER2_M1_RUNS.md` |
| 3. walk cost | **26.5 cyc/ref** conservative, **7.8** with early-out, against a 28 budget and entry 24's 3,409 | `RENDERER2_M1_WALKBENCH.md` |

Risk 3 - the one that killed the previous attempt - **passes decisively**.
Inverting entry 24's mapping (one thread per reference instead of one block
per reference behind a CTA barrier) is worth 129x conservative and 434x with
the early-out, and the early-out is bit-identical over 921,600 resolved pixels.
That result stands on its own and is worth keeping.

## The arithmetic that closes it

Displaced pool 1.4623 ms/frame (RTX heavy band, per-kernel union-exclusive),
walk 0.092-0.255, bin pass ~0.13 (entry 24's figure, still unmeasured),
surrender 0.66-0.84:

| scoping | best | worst | midpoint |
|---|---:|---:|---:|
| full design | 0.580 | 0.237 | **0.409** |
| full design, bin pass free | 0.710 | 0.367 | 0.539 |
| **opaque-only M2 slice** (A-buffer pools not displaced) | 0.267 | -0.076 | **0.096** |
| walk entirely free | 0.672 | 0.492 | 0.582 |

The gate is 0.500 ms/frame. The full design misses it at the midpoint; the
slice that would actually be written first misses it by 5x. **A free walk
still only reaches 0.582** - the walk was never the binding term.

## Why it cannot reach the goal even if every uncertainty lands well

favorite3 is 6.676 ms/frame and the goal is 5.0: a **1.676 ms/frame** gap.

* best conceivable renderer2 net (full design, free bin pass, best corner):
  0.710 -> favorite3 lands at **5.966 ms**.
* remove the *entire* displaced pool at zero cost, surrendering nothing -
  a physically impossible upper bound: 1.4623 -> **5.214 ms**.

**The upper bound of the whole redesign is above the goal.** Tile-binned
shading cannot reach 5.0 ms on this capture, because the pools it displaces
(raster stages, A-buffer support, FS compact/writeback) do not contain
1.676 ms/frame. The pools that do are the ones it explicitly does not touch:
generated VS 0.689, generated FS 0.678, `cp_clip_rast_fused` 0.592, memcpy
0.490 - vertex and fragment arithmetic plus copies, all clock-bound work that
a different scheduler does not make smaller.

## The finding that would have to be fixed first, if this is ever revived

The walk reproduces entry 24's one-block signature: on a realistic scope grid
the hottest tile's block is 346,128 of ~362,000 kernel cycles - **one block is
94% of the grid's duration**, average SM busy 13.9%. So the claim in
`RENDERER2_M1_FANOUT.md` that the surrender is "pessimistic by construction
because the walk replaces the concurrency with 3,600 blocks" **is not
supported**: the walk does not give the concurrency back. Splitting hot tiles
across blocks with a merge step is the known fix and is not attempted here,
because entry 24's retry clause requires the ceiling to exceed the surrender
*before* more is built, and the table above shows it does not.

## Kept, because they are true regardless

1. The inverted walk mapping and its measured cost (risk 3) - the only
   published number for this shape on sm_120.
2. The early-out is bit-identical, verified pixel-exact against a host walk.
3. The census infrastructure: per-scope tile census, run segmentation, and the
   cut counters, all of which were structurally dead before this project.
4. `cp_tri_setup` at 80 B straddles 32 B sectors; padding to 96 B removes about
   a third of the walk's read traffic. Cheap, unrelated to this verdict.
5. M1 section 4.3's 3-bit depth slab is degenerate as specified - the top bits
   of a float are its exponent, so every window depth in [0.25,1] lands in one
   slab. Use a monotone remap if a slab id is ever wanted.

## Status

Branch `redesign/tilewalk` is kept as evidence. Nothing merges to
`cudapipe-vk-native`. The driver remains exactly as tagged `cudavk-0.0.1`.
