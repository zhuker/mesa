# Cudapipe Opaque Sort-Middle Tiling Prototype

## Summary

Build an opt-in opaque-only tiled visibility path. Eligible opaque episodes use
per-tile primitive lists and one block per 32x32 tile. Blended draws, including
the expensive part of `particlesystem`, continue using the current A-buffer
unchanged.

Reuse existing opaque episodes, shaders, interpolation, writeback, global
primitive IDs, segment ranges, and fallback machinery. Change only binning and
visibility resolution.

Add `CUDAPIPE_TILED_OPAQUE=1`, disabled by default. Retain the architecture
behind the flag even if initially slower; default it only after correctness and
performance gates pass.

## Executive Report

This experiment implemented a complete opt-in opaque sort-middle visibility
path far enough to answer the architectural question: can Cudapipe reduce
opaque episode raster work by binning primitives once, resolving a shared
visibility buffer per tile, and then reusing the existing grouped fragment
shading path?

The answer from this first implementation is **not with the present tile
raster kernel**. The surrounding architecture works:

- Eligible opaque batches are collected into existing opaque episodes.
- Vertex processing and clipping remain unchanged and produce saved geometry.
- Primitive-to-tile count, prefix, and fill stay entirely on the GPU.
- Overflow is decided on the GPU without a host readback.
- Exactly one of tiled visibility and classic visibility executes.
- Existing interpolation, generated fragment shaders, writeback, global
  primitive IDs, and segment-range addressing are reused.
- Blended rendering and the A-buffer are not routed through tiling.
- A forced overflow correctly selects classic rasterization and reproduces the
  baseline output.

However, the tile visibility kernel is much slower than the adaptive classic
rasterizer. Nsight Systems measured `cp_opaque_tile_raster` at 2.562 seconds
over 60 `multithreading` frames, or approximately 42.7 ms per frame by itself.
It accounts for 88.7% of GPU kernel time with tiling enabled. Count and fill
together cost only about 0.77 ms per frame, and CPU launch overhead is not
material. Both real captures also regress: the old capture median rises from
25.23 to 35.61 ms and Crossroads rises from 7.27 to 10.77 ms.

The implementation therefore remains disabled by default. It is useful as a
correct architectural scaffold and as evidence that coarse 32x32 bounding-box
binning followed by exhaustive tile-wide coverage testing is not competitive
with Cudapipe's current adaptive rasterizer.

## What Was Actually Implemented

The sections later in this document describe the intended design. This section
records the exact prototype that was built, including deviations from that
plan.

### Public Controls

Two value-style debug flags were added and documented in `FLAGS.md`:

- `CUDAPIPE_TILED_OPAQUE=1` enables the prototype.
- `CUDAPIPE_TILED_OPAQUE_CENSUS=1` copies the overflow word to the host,
  synchronizes for diagnostics, and prints tile dimensions, episode segment
  count, and overflow count. The normal tiled path performs no such readback.

Both default to off. Consequently, ordinary Cudapipe behavior and performance
are unchanged unless the experiment is explicitly enabled.

### Shared Host/Device ABI

`kernels/cp_rast_types.h` gained:

- `CP_OPAQUE_TILE_SIZE = 32`.
- A fixed capacity of 2,000,000 tile references.
- `cp_opaque_tile_ref`, containing an episode-global primitive ID and segment
  index.
- Count/fill arguments carrying the saved raster state and device arrays.
- Tile-raster arguments carrying all saved segment raster states, tile arrays,
  overflow state, dimensions, and segment count.
- An optional `path_flag` and `path_value` in `cp_rasterize_args` for
  device-side guarding of classic raster kernels.

Kernel handles for `cp_opaque_tile_count`, `cp_opaque_tile_fill`, and
`cp_opaque_tile_raster` were added to `cp_kernels` and resolved from the common
CUDA module during screen initialization.

### Episode Recording

The existing `cp_opaque_appendable()` restrictions define eligibility. This
was deliberately conservative: single-sample, one supported color attachment,
order-free depth testing and writing, no blend/discard, and no fragment memory
side effects.

With tiling enabled, `cp_draw_execute()` still performs fetch, vertex shading,
clipping, setup of draw slices, and construction of `cp_rasterize_args`. At the
point where it would normally launch classic raster stages 1, 2, and 3, it
instead records the segment and returns. Saved state includes positions,
varyings, depth/raster state, primitive ranges, queue pointers, shaders, UBO
rows, draw IDs, vertex-buffer bases, scissors, and classic replay information.

This matters because the tiled experiment did not introduce a second geometry
pipeline. It changes visibility only; everything before and after visibility
is the existing production machinery.

### GPU Binning

At opaque-episode finish, `cp_opaque_tile_visibility()` allocates tile arrays
from device scratch:

- One count, offset, and cursor per framebuffer tile.
- Up to 2,000,000 eight-byte references, approximately 15.3 MiB.
- One overflow word.
- Reused multi-level scan scratch.

For every saved segment, `cp_opaque_tile_count` calls the existing
`setup_triangle()` helper for each primitive. It converts the conservative
pixel bounding rectangle into a 32x32 tile rectangle and atomically increments
every tile in that rectangle. The generalized A-buffer scan produces exclusive
offsets and sets overflow if capacity is exceeded. `cp_opaque_tile_fill`
repeats setup and bounding-box traversal, atomically reserves a slot in each
tile, and stores `(global_prim, segment)`.

The prototype does **not** currently compact active tile IDs, perform exact
triangle-versus-tile rejection during binning, sort references, or maintain
persistent per-context tile storage. It launches across all framebuffer tiles,
empty tiles return immediately, and arrays come from the existing device
scratch allocator. Those are deviations from the more ambitious original
plan, but the profile shows they are not the dominant observed cost: count and
fill together are below one millisecond per `multithreading` frame.

### Tile Visibility

`cp_opaque_tile_raster` launches one 256-thread block for every 32x32
framebuffer tile. Each thread owns four fixed pixels and retains four packed
visibility winners in registers for the whole list walk.

For each reference in the tile:

1. Thread zero loads the segment raster state and reference.
2. Thread zero converts the global primitive ID to a segment-local ID.
3. Thread zero calls the existing `setup_triangle()` and stores setup in shared
   memory.
4. The block synchronizes.
5. All 256 threads test their four pixels against bounding bounds, rejection
   state, point or triangle coverage, persistent depth, and packed visibility
   ordering.
6. The block synchronizes before reusing shared setup for the next reference.

After all references, each thread writes its four winners once to the existing
global visibility buffer. This removes per-fragment global visibility atomics,
but replaces the adaptive division of raster work with a uniform
`tile references × 1024 pixels` coverage loop plus two block barriers per
reference.

### Device-Selected Overflow Fallback

The first implementation rasterized classically during episode append, then
ran tiled visibility over the already-correct result. If binning overflowed,
the tiled kernel returned and the classic result remained. This was safe but
duplicated normal-case visibility work.

The final implementation removes that duplication:

- The overflow word is initialized to zero and populated by scan/fill.
- The tiled kernel returns when overflow is nonzero.
- Classic stage1/2/3 kernels are enqueued after it for every saved segment,
  carrying the same overflow pointer and requesting the nonzero path.
- Each classic kernel checks the flag before touching queues or visibility.
- Shading is naturally ordered after both sets of launches.

Thus the CPU submits both possibilities, but the GPU executes only one. No
host decision or CPU/GPU round trip is needed. A forced capacity of one
reference produced an overflow value of 542. That test exposed an initial bug:
the guard compared the word to literal `1`, although overflow records a count.
Changing the guard to compare zero versus nonzero made the fallback render
three triangle frames exactly like the baseline.

### Reused Shading Path

No tile-specific shader ABI was introduced. After visibility resolution,
`cp_opaque_finish()` retains its existing shader-group formation and
`cp_seg_range` tables. `cp_shade_fragments()` performs interpolation, invokes
the generated fragment `main`, and writes color/depth. Because the visibility
buffer stores episode-global primitive IDs, grouped shading can recover the
owning segment, local primitive, vertex positions, draw slice, and UBO row.

This kept the experiment scoped to raster organization and allowed profiles to
show that shader, interpolation, and writeback times remain essentially
unchanged.

## Development Iterations

### Iteration 1: Correctness-First Duplicate Path

The initial kernel had two independent performance problems:

- Classic visibility still executed before tiled visibility.
- Every pixel thread independently called `setup_triangle()` for every tile
  reference, redundantly repeating setup up to 256 times per reference.

The 18-sample aggregate rose from 35.85 to 163.70 ms (+356.6%). This version
was useful only to validate data flow and visibility semantics.

### Iteration 2: Shared Triangle Setup

Triangle setup moved to thread zero and shared memory. Each thread retained
four pixel winners, so one block still covered all 1024 pixels while setup was
performed once per reference. The aggregate improved to 88.09 ms. This large
improvement confirmed that redundant setup was a real implementation bug, but
the path remained 145.7% slower than baseline and still duplicated classic
visibility.

### Iteration 3: Exclusive Device-Gated Paths

Classic raster launches were removed from episode append and reintroduced at
finish behind the device overflow predicate. The aggregate improved only from
88.09 to 86.02 ms. `multithreading` remained 46.42 ms versus 5.54 ms baseline.

This small 2.07 ms aggregate improvement was important evidence: duplicate
classic work was not the explanation for the remaining regression. The tiled
kernel itself had to be profiled.

## Why Tiling Was Expected to Help

The experiment targeted several real weaknesses in the current pipeline:

- Opaque episodes already defer shading and resolve a shared winner, so they
  provide a natural unit for visibility consolidation.
- A tile block can keep winners in registers and write visibility once,
  avoiding global atomic updates for every covered fragment.
- Grouping consecutive opaque draws can eliminate internal overdraw before
  fragment shading.
- Device-side overflow selection avoids the latency of reading a capacity
  decision back to the CPU.
- One tile launch per episode appeared capable of replacing many per-segment
  raster launches.

Those expectations were directionally reasonable, but they omitted a crucial
comparison: the classic rasterizer is already adaptive. Small triangles are
handled directly, larger/nontrivial triangles are processed cooperatively, and
huge coverage is queued into tiles. Replacing it with tiling is beneficial only
if binning plus per-tile work is cheaper than that existing classification and
coverage strategy.

## Why the Prototype Did Not Deliver

### What the Profile Proves

Nsight Systems proves the following for 60 `multithreading` frames:

- `cp_opaque_tile_raster` is the regression: 2,562.1 ms total, 88.7% of all
  tiled GPU kernel time, 39.4 ms per launch, and about 42.7 ms per frame.
- Count and fill are comparatively small: 23.3 and 23.1 ms total.
- Fragment `main`, clipping, vertex fetch, interpolation, and writeback remain
  near their baseline totals.
- Classic raster work is genuinely skipped for eligible episodes: its stages
  fall from 39.0 to 20.2 ms total.
- CUDA launch API time changes by only 2.8 ms total despite the launch count
  increasing from 5,007 to 6,233.

Therefore the failure is not caused primarily by duplicate classic raster,
CPU launch cost, prefix scan, tile-reference fill, shader execution, or an
overflow synchronization.

### What the Kernel Structure Strongly Suggests

Nsight Systems identifies the dominant kernel but does not provide instruction,
occupancy, divergence, or cache counters. The following explanation is derived
from the kernel's work structure and should be treated as a well-supported
hypothesis until checked with Nsight Compute:

- Binning uses conservative screen-space bounding boxes. A triangle is placed
  in every overlapped 32x32 tile even when it covers only a small portion of
  that tile.
- Every reference causes all 256 threads to inspect four pixels: 1024 pixel
  candidates per reference, before bounding and edge tests discard most of
  them.
- Large or elongated triangles duplicate that 1024-pixel loop across many
  tiles. Dense overlapping geometry produces long lists whose work grows as
  `references × tile area`, not approximately as covered fragments.
- Every reference requires two `__syncthreads()` barriers, serial shared setup
  by thread zero, and a broadcast-sized copy of `cp_rasterize_args` into shared
  memory.
- Threads carry four 64-bit winners through the entire list, while coverage
  code has multiple divergent rejection, point/triangle, edge, and depth
  branches. Register pressure and divergence may reduce occupancy, but this
  has not yet been measured with Nsight Compute.
- All nonempty tiles launch even if a more selective hierarchy could reject
  them earlier. There is no coarse/fine bin hierarchy, exact tile coverage
  mask, active-tile compaction, or reference subdivision.
- The classic rasterizer chooses different execution strategies by primitive
  size. The prototype discards that specialization and applies one expensive
  strategy to every binned primitive.

The decisive lesson is not that tiled rasterization cannot work. It is that
**coarse bounding-box binning is insufficient if the consumer exhaustively
tests a full 32x32 tile for every reference**. Removing global atomics and
launches did not compensate for the extra coverage instructions and barriers.

## Status and Recommended Direction

The flag must remain off by default. The prototype misses every performance
gate and has small depth-tie output differences in some workloads, although
the overflow fallback itself is exact in the forced test.

Do not spend time optimizing count, scan, fill, CPU launch submission, or the
overflow selector first; measured costs show they cannot recover the loss.
Before further architectural work, use Nsight Compute on
`cp_opaque_tile_raster` to measure executed instructions, branch efficiency,
barrier stalls, achieved occupancy, register count, and memory throughput.

If development continues, the next version should preserve adaptive raster
behavior rather than use one uniform tile loop. Plausible directions are:

1. Bin into coarse regions, then create compact 8x8 fine-tile work only where
   conservative coverage warrants it.
2. Reject triangle/tile pairs more precisely during fill, or store coverage
   masks/edge setup so the consumer does not repeat broad tests.
3. Separate small triangles from large triangles and retain the classic direct
   or warp-cooperative path for the former.
4. Compact active tiles and group references into warp-sized work units rather
   than making every block process every listed primitive with all 256 lanes.
5. Precompute compact edge/depth coefficients per reference if the memory
   traffic is cheaper than rerunning `setup_triangle()` and copying full raster
   state.
6. Reconsider tile size using measured reference distributions rather than a
   fixed 32x32 choice.

Any successor should be tested first against `multithreading`, because it
amplifies the proven tile-raster failure, then against both GFXRs and the full
sample sweep. The current implementation should be retained only as an
experimental scaffold or reverted if its maintenance cost interferes with the
production path.

## Architecture

### Eligibility

Use the existing `cp_opaque_appendable()` envelope unchanged:

- Single sample.
- One color attachment.
- No blending or discard.
- No fragment memory writes.
- Depth test and write enabled.
- LESS, LEQUAL, GREATER, or GEQUAL.
- Supported color format.

Blended, multisampled, attachment-incompatible, side-effecting, and unsupported
draws retain their current paths.

### Tile Geometry

- Default tile size: 32x32 pixels.
- 256 threads per tile block.
- Each thread owns four fixed pixels for the entire primitive-list walk.
- Keep the existing adaptive-rasterizer `CP_TILE_SIZE=64` independent; use
  `CP_OPAQUE_TILE_*` names.
- Preserve current viewport, scissor, clipping, point, triangle, depth, and tie
  behavior.

### Device Storage

Add reusable per-context storage:

```c
struct cp_opaque_tile_ref {
   uint32_t global_prim;
   uint16_t segment;
   uint16_t flags;
};
```

Allocate:

- Tile counts.
- Exclusive tile offsets.
- Tile cursors.
- Tile references.
- Active tile IDs and count.
- Scan scratch.
- Overflow flags.
- A device mode word selecting tiled or classic visibility.

Start with `CP_MAX_HUGE_TILES` references. Record required capacity before
changing it.

## Implementation

### 1. Census

Add `CUDAPIPE_TILED_OPAQUE_CENSUS=1`, which changes no rendering.

For 16x16, 32x32, and 64x64 candidates, record:

- Tile references per primitive and episode.
- Active and empty tiles.
- References per tile: median, p90, p99, maximum.
- Shader groups and segments per tile.
- Estimated memory.
- Capacity overflows.
- Existing stage1/2/3 work attributable to accepted episodes.

Run both GFXRs and all 18 Vulkan samples.

Keep 32x32 unless another size reduces total references by at least 20% while
keeping p99 references per tile at or below 256.

### 2. Count and Prefix

At opaque-episode finish:

1. Clear counts, cursors, active count, overflow, and mode asynchronously.
2. Launch a tile-count kernel for each saved segment on existing side streams.
3. Compute the same conservative clipped primitive bounds as the current
   rasterizer.
4. Increment every touched tile's count.
5. Join side streams.
6. Reuse generalized `cp_abuf_scan_block` and `cp_abuf_scan_add` kernels to
   produce exclusive offsets.
7. Emit active tile IDs.
8. Set overflow if the final reference count exceeds capacity.

No CPU reads any count or overflow state.

### 3. Fill and Order Tile Lists

Relaunch each segment's primitive walk:

- Reserve positions with tile-local atomic cursors.
- Store global primitive and segment IDs.
- Never write beyond capacity.
- Set overflow on any rejected write.
- Sort each valid tile list by `global_prim`.
- Use the existing short-list single-thread strategy and block/shared-memory
  sorting strategy.

Opaque resolution does not require ordering, but ordered lists make the
retained infrastructure reusable for later blended work.

### 4. Device-Side Path Selection

Add a one-thread selector kernel after fill:

```text
overflow clear -> mode = TILED
overflow set   -> mode = CLASSIC
```

Do not synchronize or copy this result to the CPU.

Every later visibility kernel receives the device mode pointer:

- New tile raster kernel immediately returns unless mode is `TILED`.
- Existing classic raster stage launches immediately return unless mode is
  `CLASSIC`.
- Both paths target the same cleared episode visbuf.
- Stream ordering guarantees shading starts after both possible paths complete.
- Exactly one path writes visibility.

The host enqueues the classic fallback raster launches unconditionally but
guarded. This adds launch submission cost, not duplicate raster work or a
CPU/GPU round trip.

### 5. Tile Visibility Kernel

Launch one block per framebuffer tile; empty tiles return immediately.

Each block:

1. Assigns four tile pixels to every thread.
2. Loads persistent depth and initializes four packed winners in registers.
3. Walks the tile's ordered primitive references.
4. Resolves segment and local primitive from the reference.
5. Builds triangle or point setup once per reference and broadcasts it.
6. Applies exact existing coverage tests.
7. Applies exact existing depth transformation, comparison, and equal-depth
   primitive tie rules.
8. Updates register winners without global visibility atomics.
9. Writes one final packed winner per framebuffer pixel to the existing visbuf.

Factor common packing, depth, edge, point, facing, and tie logic into shared
device helpers used by old and tiled paths.

### 6. Classic Guarded Fallback

Extend existing raster arguments with an optional device mode pointer and
required mode value.

- Normal draws pass a null pointer and behave unchanged.
- Tiled opaque fallback launches require `CLASSIC`.
- Each raster kernel checks the mode before reading or writing queues.
- If tile-list overflow occurs, the classic stage1/2/3 pipeline rebuilds the
  episode visibility from saved segment geometry.
- No color or persistent depth has been committed before selection.
- CUDA failures remain diagnostic; correctness fallback handles data-capacity
  overflow, not arbitrary partial kernel failure.

### 7. Existing Shading

After both guarded visibility paths:

- Reuse current opaque shading groups.
- Reuse `cp_seg_range`.
- Reuse fused interpolation, generated fragment `main`, UBO rows, and
  writeback.
- Do not change the shader ABI.
- Do not add tile-local color or depth.
- Do not alter blended A-buffer rendering.

## Performance Isolation

Record separately:

- Count, scan, fill, sort, and tiled raster time.
- Guarded-classic launch count and actual executed count.
- Host launch API overhead from guarded fallback.
- Old stage1/2/3 work avoided.
- Global visibility atomics avoided.
- Active tiles and references.
- Scratch and persistent memory.
- Per-sample number of eligible opaque episodes.

For `particlesystem`, explicitly report:

- Blended A-buffer time, which must remain structurally unchanged.
- Number and cost of opaque tiled episodes.
- Net frame delta.
- Guarded-launch overhead.

If mixed workloads regress, distinguish actual tiled GPU work from empty
guarded launch overhead.

## Validation

### Opaque prototype results (2026-08-18)

- The first correctness-first implementation retained classic visibility as
  an already-rendered fallback and then overwrote it with tiled visibility.
  Its 18-sample total was 163.70 ms versus the 35.85 ms baseline.
- Moving triangle setup out of the per-pixel loop reduced the total to
  88.09 ms, but still executed both visibility paths.
- Device-side exclusive gating now records geometry without classic raster at
  append time. Binning writes an overflow word; tiled raster runs when it is
  zero and the three classic raster stages run when it is nonzero. No CPU
  readback selects the path.
- The exclusive version measured 86.02 ms total versus 35.85 ms baseline
  (+139.9%). `multithreading` remains the dominant failure at 46.42 ms versus
  5.54 ms. Eliminating duplicate classic raster therefore helped only 2.07 ms
  in aggregate and did not make this tile traversal competitive.
- A forced one-reference-capacity test produced overflow=542 and initially
  exposed an equality-predicate bug. Comparing zero/nonzero fixed it; three
  triangle frames then matched the classic baseline exactly through the
  device-selected overflow fallback.
- Focused no-overflow checks matched triangle exactly. Instancing, particles,
  and glTF differed by at most isolated pixels; multithreading retained tiny
  submission/depth-tie differences. The full sweep still reports the standing
  glTF and texture3d deviations against NVIDIA, so this remains opt-in only.

#### Nsight profile: multithreading

An Nsight Systems CUDA/NVTX comparison over 60 offscreen frames confirmed the
location of the regression rather than inferring it from the algorithm:

- With tiling disabled, all GPU kernels total about 300 ms. The largest are
  `cp_clip_triangles` at 144.8 ms, fragment `main` at 62.1 ms, vertex fetch at
  35.8 ms, and all three classic raster stages together at 39.0 ms.
- With tiling enabled, `cp_opaque_tile_raster` alone takes 2,562.1 ms across
  65 launches: 88.7% of GPU kernel time, 39.4 ms per launch, or approximately
  42.7 ms per rendered frame.
- Tile count and fill are not the main problem: they take 23.3 ms and 23.1 ms
  total respectively, about 0.77 ms per frame combined.
- The remaining shader, clip, vertex-fetch, interpolation, and writeback times
  are effectively unchanged. Classic raster stages fall from 39.0 ms to 20.2
  ms total because eligible episode work is correctly skipped.
- Launch count rises from 5,007 to 6,233, but launch API time rises by only
  2.8 ms total. The regression is therefore device execution inside
  `cp_opaque_tile_raster`, not CPU launch overhead, bin construction, or the
  overflow decision.

Artifacts:

- `~/claude-scratchpad/perf16/tile_multithreading_off.nsys-rep`
- `~/claude-scratchpad/perf16/tile_multithreading_on.nsys-rep`
- corresponding `_stats.csv` and `.sqlite` files in the same directory

#### GFXR replay results

FPS-plugin replays completed every recorded frame on both captures, comparing
the ordinary path and `CUDAPIPE_TILED_OPAQUE=1` from the same build:

| Capture | Mode | Frames | Median | Mean | p95 |
|---|---:|---:|---:|---:|---:|
| `headless_streamer_20260814T155742.gfxr` | baseline | 1510 | 25.23 ms | 30.54 ms | 35.64 ms |
| `headless_streamer_20260814T155742.gfxr` | tiled | 1510 | 35.61 ms | 38.99 ms | 49.68 ms |
| `headless_streamer_1818_20260817T173522.gfxr` | baseline | 1496 | 7.27 ms | 8.52 ms | 8.68 ms |
| `headless_streamer_1818_20260817T173522.gfxr` | tiled | 1496 | 10.77 ms | 12.02 ms | 14.35 ms |

Tiling regresses the old capture median by 41.1% and Crossroads by 48.1%.
The p95 also regresses by 39.4% and 65.3% respectively. Timing artifacts and
replay logs are under `~/claude-scratchpad/perf16/tiled_gfxr/`.

### Kernel Verification

Under a debug flag, render old and tiled visibility into separate buffers and
compare:

- Packed depth.
- Winning global primitive.
- Coverage count.
- Equal-depth ties.
- Front-facing behavior.

Test:

- LESS, LEQUAL, GREATER, GEQUAL.
- Reversed depth.
- Equal-depth primitives across segments.
- Points and triangles.
- Clipped primitives.
- Viewport/scissor boundaries.
- Odd framebuffer sizes.
- Partial edge tiles.
- Multiple shader groups.
- Forced tile-reference overflow.
- Gaps in primitive ranges.

### Application Validation

Run flag-on:

- Complete 18-sample, 60-frame sweep.
- Compare against the current Cudapipe iteration.
- Explicitly inspect instancing and multithreading for missing geometry.
- Confirm `particlesystem` blended output is unchanged.
- Preserve existing `gltfscenerendering` and `texture3d` NVIDIA deviations
  without growth.
- Render every frame of both GFXRs.
- Check Crossroads frames 633, 756, and 907 plus old-capture sentinels.
- Force overflow with a tiny capacity and require guarded classic output to
  match flag-off output.

No correctness regression is accepted.

## Retention and Default Gates

Retain a correct implementation behind `CUDAPIPE_TILED_OPAQUE=1` regardless of
speed.

Enable it by default only if:

- One GFXR improves by at least 2%.
- The other GFXR regresses by no more than 1%.
- The 18-sample aggregate regresses by no more than 1%.
- No important sample regresses by more than 5%.
- `particlesystem` does not regress by more than 1%.
- Persistent memory stays below 256 MiB per context.
- No new CPU/GPU synchronization is introduced.
- Empty guarded fallback launches do not erase the tiled raster gain.

## Documentation

Record with the implementation:

- Census results and selected tile size.
- Tile-reference capacity and observed maximum.
- Correctness artifacts.
- Performance tables.
- Overflow frequency.
- Empty guarded-launch overhead.
- Per-workload eligible opaque share.
- What helped and what failed.
- Whether the path stays experimental or becomes default.
- Requirements for a later blended tiled implementation.

Update `EPISODES.md`, `FLAGS.md`, and iteration records as results are produced.

## Assumptions

- Opaque-first prototype only.
- Opaque-only means eligible opaque episodes, not the whole application.
- Blended `particlesystem` draws remain on the A-buffer.
- Device-gated classic fallback replaces a host overflow readback.
- Existing shading and fragment ABI remain unchanged.
- Blended tiling, tile-local color, multisampling, multiple attachments, shader
  depth replacement, and fragment side effects remain out of scope.
