# Renderer2 M0: tile/depth-complexity census results

Census build `57ff264ddb8`, `CUDAVK_TILE_CENSUS=16`, full replays of both
captures on RTX 5090. Both runs kept their standard stdout hashes and full
timestamp populations (favorite3 rc=139/6,939 rows, favorite2 rc=0/6,965) and
pass the strict analyzer. Logs: `/tmp/renderer2-m0/{favorite3,favorite2}/stderr`.
All numbers below are real frames only (absolute submit > 2782). Census
timing is invalid by construction and none is quoted.

## Scope population

| | favorite3 | favorite2 |
|---|---:|---:|
| real scopes | 18,865 | 17,014 |
| class=none (zero draws: clear/copy-only scopes) | 14,666 (78%) | 12,830 (75%) |
| drawing scopes | 4,199 (~2.02/frame) | 4,184 (~2.4/frame) |
| mixed (opaque+blended) | 2,083 | 2,090 |
| blended-only | 2,090 | 2,094 |
| **opaque-only** | **26** | **0** |
| solo (unbinned classic-path) draws, whole run | 140 | 0 |

**Finding 1, and it changes M2:** pure-opaque scopes are statistically absent.
Nearly every drawing scope contains blended draws. A tile walk that only takes
opaque-only scopes exercises nothing; the scope-segmentation problem (survey
problem 3: maximal expressible runs inside mixed scopes, with depth/A-buffer
handoff at cuts) is the *entry ticket*, not a later milestone.

## Per drawing scope

| percentile | draws f3 | draws f2 | FS binaries f3 | FS binaries f2 | batches f3 | batches f2 |
|---|---:|---:|---:|---:|---:|---:|
| p50 | 28 | 14 | 3 | 2 | 16 | 10 |
| p90 | 269 | 193 | 18 | 15 | 56 | 29 |
| p99 | 326 | 215 | 20 | 15 | 102 | 29 |
| max | 349 | 224 | **21** | **15** | 132 | 29 |

**Finding 2:** the FS-bucket dimension is bounded and small — at most 21
distinct fragment binaries in the worst scope. Per-tile shading buckets (sort
tile list by FS identity within draw-order constraints) are feasible; the
4-axis identity (binary x exec x sampler variant x tune alternate) multiplies
this but stays two digits.

## Depth complexity / tile lists (16 px tiles, 80x45 grid)

| | favorite3 | favorite2 |
|---|---:|---:|
| post-clip triangles, whole run | 735.1M (~354k/frame) | 717.2M (~419k/frame) |
| tile references, whole run | 855.2M (~412k/frame) | 798.2M (~466k/frame) |
| refs per triangle | 1.16 | 1.11 |
| mean refs per non-empty scope-tile | 117 | 116 |
| hottest tile of median scope (refs_max p50) | 237 | 24 |
| refs_max p90 / p99 / max | 9,208 / 12,823 / 14,814 | 7,263 / 10,096 / 13,259 |

Aggregate per-tile reference histogram (log2 buckets, scope-tiles):

- favorite3: peak at 16-63 refs (3.34M tiles), long tail through 4k-16k
  (39k tiles above 2k refs).
- favorite2: same shape, peak 16-63 (3.30M), 54k tiles above 2k refs.

Entry 24's shape is confirmed at current HEAD on both captures: micro
triangles (~1.1 refs each), lists that are depth complexity rather than
coverage. The early-out prize is real (the hot-tail tiles hold thousands of
refs for hundreds of pixels), but it is concentrated: the median scope's
hottest tile is only a few hundred refs.

## Structural prize restated

Drawing scopes are ~2.0-2.4 per frame. A per-scope bin+walk structure is
single-digit launches per frame against today's 414-740, and the per-scope
serial tile-list ceiling (refs_max p99 ~13k at 16 px) bounds the longest
block's work. Tile-size sensitivity (32/64 px) is unmeasured at this HEAD;
rerun the census at 32 before fixing the tile size.

## Census caveats

- `cp_tile_census_cut()` remains unwired (pre-existing); residency-cut
  numbers in the cumulative report are structurally zero and were not used.
- class=none scopes are real scopes; the tile walk never sees them, and the
  dead-scope elimination already prices them.
- One capture, one census tile size, one GPU; the histogram is for sizing,
  not for banking wins.

## M1 go/no-go inputs now in hand

1. Segmentation is mandatory (finding 1) - design the run-splitter first.
2. FS buckets bounded by ~21 (finding 2) - per-tile bucket walk is viable.
3. Scope-global draw id must be threaded through the ABI (survey problem 2).
4. Hot-tile serial ceiling ~13-15k refs at 16 px - the walk kernel budget
   must be sized against this, with early-out doing the heavy lifting on the
   opaque runs.
