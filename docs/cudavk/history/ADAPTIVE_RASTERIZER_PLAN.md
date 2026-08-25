# Adaptive Rasterizer Implementation Plan

## Goal

Replace the current single-thread-per-triangle bounding-box rasterizer with a
3-stage adaptive rasterizer that combines CuRast's CUDA execution model with
llvmpipe's tile-level trivial accept/reject. Target: close the 3.6× gap vs
NVIDIA hardware (currently at 99% GPU utilization but doing redundant work per
pixel).

## Current State

File: `src/gallium/drivers/cudapipe/kernels/cp_rasterize.cu`

The current rasterizer (`cp_rasterize_triangles`) assigns 1 CUDA thread per
triangle. Each thread:
1. Loads 3 vertices from VS output (strided by `num_varyings + 1`)
2. Perspective divide → screen space
3. Computes bounding box, clips to viewport
4. Iterates every pixel in the bounding box
5. Evaluates 3 edge functions per pixel (3 multiplications + comparisons)
6. If inside: interpolates depth, tests against depthbuf, atomicMin into visbuf

**Problem:** A triangle covering 10,000 pixels runs 10,000 iterations on ONE
thread. The GPU's 2560 cores can only work on other triangles in parallel.
Large triangles serialize the entire GPU.

## Design: 3-Stage Adaptive Dispatch

### Stage 1: Small triangles (1 thread per triangle)

Threshold: bounding box ≤ `SMALL_THRESHOLD` pixels (e.g., 64 or 128).

Keeps the current approach — it's already optimal for small triangles (which
are the majority in the Roblox workload). Additions:

- **Precompute edge function gradients** once per triangle:
  ```
  // Edge 0: from v1 to v2
  float e0_dx = sy2 - sy1;  // dE0/dx
  float e0_dy = sx1 - sx2;  // dE0/dy
  // (similarly for edges 1, 2)
  ```
- **Incremental stepping** in the inner loop:
  ```
  float e0 = e0_init;  // value at (ix_min+0.5, iy_min+0.5)
  for (py...) {
      float e0_row = e0;
      for (px...) {
          if (e0_row >= 0 && e1_row >= 0 && e2_row >= 0) { ... }
          e0_row += e0_dx;
          e1_row += e1_dx;
          e2_row += e2_dx;
      }
      e0 += e0_dy;
      e1 += e1_dy;
      e2 += e2_dy;
  }
  ```
  This replaces 6 multiplications per pixel with 3 additions.

- If `bb_area > SMALL_THRESHOLD`: push `tri_id` into a **nontrivial queue**
  (global device array + atomicAdd counter). Don't rasterize it here.

### Stage 2: Medium triangles (1 warp per triangle)

Consumes from the nontrivial queue. 32 threads cooperate on one triangle.

- Thread 0 of the warp loads vertices, computes screen positions, edge
  gradients, bounding box. Broadcasts via `__shfl_sync(0xFFFFFFFF, val, 0)`.
- All 32 threads iterate the bounding box with stride 32:
  ```
  for (int i = threadIdx.x % 32; i < bb_area; i += 32) {
      int px = ix_min + (i % bb_width);
      int py = iy_min + (i / bb_width);
      // edge test + depth + atomicMin
  }
  ```
- If `bb_area > MEDIUM_THRESHOLD` (e.g., 4096 = 64×64): split into tiles and
  push tile-triangle pairs into a **huge queue** for stage 3.

### Stage 3: Huge triangles (1 block per tile, with trivial accept/reject)

Consumes tile-triangle pairs from the huge queue. Each block handles one
64×64 tile of one triangle.

**Key insight from llvmpipe:** Before testing any pixel, evaluate edge functions
at the 4 tile corners. For each edge:
- If all 4 corners are inside: the entire tile is inside this edge (**trivial accept** for this edge)
- If all 4 corners are outside: the entire tile is outside (**trivial reject** — skip tile entirely)
- Otherwise: partial — test per pixel

```
// Evaluate edge at tile corners
float e_tl = edge_at(tile_x, tile_y);
float e_tr = e_tl + e_dx * TILE_SIZE;
float e_bl = e_tl + e_dy * TILE_SIZE;
float e_br = e_tl + e_dx * TILE_SIZE + e_dy * TILE_SIZE;

bool all_outside = (e_tl < 0) && (e_tr < 0) && (e_bl < 0) && (e_br < 0);
bool all_inside  = (e_tl >= 0) && (e_tr >= 0) && (e_bl >= 0) && (e_br >= 0);
```

- **Trivial reject** (any edge all-outside): entire block returns immediately
- **Trivial accept** (all 3 edges all-inside): shade all 4096 pixels without
  edge tests — just interpolate depth and atomicMin
- **Partial**: use incremental edge stepping with per-pixel tests (same as
  stage 1's inner loop, but parallelized across the 64-thread block)

Block size: 64 threads. Each thread handles one column of 64 pixels (loop over
rows). This gives coalesced memory access to the visbuf (consecutive threads
write consecutive pixels in the same row).

Alternatively: 256 threads, each handles a 4×4 sub-tile of the 64×64 tile.
The trivial-accept optimization eliminates most sub-tiles entirely.

## Constants

```c
#define SMALL_THRESHOLD   128   // max pixels for stage 1 (1 thread)
#define MEDIUM_THRESHOLD  4096  // max pixels for stage 2 (1 warp)
#define TILE_SIZE         64    // pixels per tile edge for stage 3
#define MAX_NONTRIVIAL    1000000
#define MAX_HUGE_TILES    2000000
```

## Data Structures

```c
struct cp_rast_queues {
    uint64_t nontrivial;     // Device pointer to uint32_t[MAX_NONTRIVIAL]
    uint64_t nontrivial_count; // Device pointer to atomic uint32_t
    uint64_t huge_tiles;     // Device pointer to TilePair[MAX_HUGE_TILES]
    uint64_t huge_count;     // Device pointer to atomic uint32_t
};

struct TilePair {
    uint32_t tri_id;
    uint16_t tile_x;  // tile origin in pixels
    uint16_t tile_y;
};
```

These live in device memory, allocated once at context creation (alongside
visbuf/depthbuf). Counters are zeroed per draw (cuMemsetD32).

## Kernel Signatures

```c
// Stage 1: same launch config as current (1 thread per triangle)
extern "C" __global__ void
cp_rasterize_stage1(struct cp_rasterize_args args, struct cp_rast_queues queues);

// Stage 2: launched with enough warps to cover nontrivial_count
// Grid: (nontrivial_count + 31) / 32 blocks of 32 threads
extern "C" __global__ void
cp_rasterize_stage2(struct cp_rasterize_args args, struct cp_rast_queues queues);

// Stage 3: launched with huge_count blocks of 64 threads
extern "C" __global__ void
cp_rasterize_stage3(struct cp_rasterize_args args, struct cp_rast_queues queues);
```

## Host-Side Changes (cp_context.c)

The draw path currently launches one rasterize kernel. Change to:

```c
// Zero queue counters
cuMemsetD32(nontrivial_count, 0, 1);
cuMemsetD32(huge_count, 0, 1);

// Stage 1: all triangles (small ones rasterize in place, others queue)
cuLaunchKernel(stage1, (num_triangles+255)/256, 1, 1, 256, 1, 1, ...);

// Read nontrivial_count back (need sync here — or use a fixed grid and
// have stage 2 self-bound from the counter)
// Better: launch stage 2 with a fixed large grid and self-bound
cuLaunchKernel(stage2, MAX_STAGE2_BLOCKS, 1, 1, 32, 1, 1, ...);

// Same for stage 3
cuLaunchKernel(stage3, MAX_STAGE3_BLOCKS, 1, 1, 64, 1, 1, ...);
```

To avoid a sync between stages, launch stages 2 and 3 with fixed grid sizes.
Each thread checks if its index exceeds the counter and exits early. The
counter is in device memory — the GPU reads it directly. This keeps the entire
3-stage pipeline in one stream with zero CPU syncs.

## Optimization Details

### 32-bit edge math for small triangles

If the bounding box fits in 2048×2048 pixels (which it always does for
SMALL_THRESHOLD ≤ 128), the edge function intermediate values fit in 32-bit
float without precision issues. Only stage 3's large triangles need careful
precision handling.

### Reduced plane count per tile (from llvmpipe)

In stage 3, after trivial-accept classification, if an edge is fully inside
the tile, skip its per-pixel test. Track a `plane_mask` (3 bits) and only
evaluate edges that are partial:

```c
uint32_t plane_mask = 0;
if (!all_inside_e0) plane_mask |= 1;
if (!all_inside_e1) plane_mask |= 2;
if (!all_inside_e2) plane_mask |= 4;

// Per pixel: only test edges in plane_mask
bool inside = true;
if (plane_mask & 1) inside &= (e0 >= 0);
if (plane_mask & 2) inside &= (e1 >= 0);
if (plane_mask & 4) inside &= (e2 >= 0);
```

For a large triangle covering many tiles, interior tiles have `plane_mask = 0`
(trivial accept) and skip ALL edge tests — just depth interpolation + atomicMin.

### Warp-level broadcast in stage 2

```c
// Lane 0 computes setup
float sx0, sy0, ...; // screen positions
float e0_dx, e0_dy, ...; // gradients
if (lane_id == 0) { /* load vertices, compute setup */ }

// Broadcast to all 32 lanes (zero cost, no shared memory)
sx0 = __shfl_sync(0xFFFFFFFF, sx0, 0);
sy0 = __shfl_sync(0xFFFFFFFF, sy0, 0);
// ... all setup values
```

## Testing

### Build

```bash
export PATH="$HOME/vulkan-sdk/1.4.335.0/x86_64/bin:$PATH"
export C_INCLUDE_PATH="$HOME/vulkan-sdk/1.4.335.0/x86_64/include"
ninja -C build-cudapipe src/gallium/targets/cudapipe/libvulkan_cudapipe.so
```

### dEQP (correctness)

```bash
ICD=build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json

# Smoke test (must pass 6/6)
VK_DRIVER_FILES=$ICD deqp-vk --deqp-case='dEQP-VK.api.smoke.*'

# Triangle draw tests
VK_DRIVER_FILES=$ICD deqp-vk --deqp-case='dEQP-VK.draw.*triangle*'

# Texture filtering (exercises full pipeline: draw + texture sample)
VK_DRIVER_FILES=$ICD deqp-vk --deqp-case='dEQP-VK.texture.filtering.2d.formats.r8g8b8a8_unorm*'
```

dEQP binary is at `/home/coder/git/VK-GL-CTS/build/external/vulkancts/modules/vulkan/deqp-vk`.

### GFXReconstruct replay (performance + real-world correctness)

```bash
ICD=build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json
GFX=~/gfxreconstruct/build

# 60-second timeout (should not crash)
time VK_DRIVER_FILES=$ICD timeout 60 \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
    /tmp/headless_streamer_20260809T231703.gfxr

# Check GPU utilization in another terminal during replay
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# Compare timing against NVIDIA baseline (19 seconds)
time VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    $GFX/tools/replay/gfxrecon-replay -m remap \
    /tmp/headless_streamer_20260809T231703.gfxr
```

**Target:** replay completes in under 30 seconds (currently 68s before crash).
The crash at 68s is a lavapipe descriptor bug (not rasterizer-related) — it
will still happen regardless of rasterizer performance.

### Per-draw timing

```bash
CUDAPIPE_DEBUG_TIME=1 VK_DRIVER_FILES=$ICD timeout 15 \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
    /tmp/headless_streamer_20260809T231703.gfxr 2>/tmp/timing.txt

# Average ms/draw
awk '{match($0,/total *([0-9.]+)/,t); sum+=t[1]} END{printf "%.2f ms/draw\n", sum/NR}' /tmp/timing.txt
```

## Implementation Order

1. **Add queue data structures** — allocate nontrivial/huge arrays in context
   creation, zero counters in draw path. Keep the current kernel as `stage1`
   but add the threshold check + queue push.

2. **Implement incremental stepping in stage 1** — replace the per-pixel
   `edge_function()` calls with precomputed gradients + additions. This alone
   is ~2× speedup for stage 1 without any structural change.

3. **Add stage 2 (warp-cooperative)** — new kernel that consumes the nontrivial
   queue. Verify with dEQP that rendering is still correct.

4. **Add stage 3 (block per tile, trivial accept/reject)** — new kernel for
   huge triangles. Add the tile corner classification.

5. **Tune thresholds** — run the replay with different SMALL_THRESHOLD and
   MEDIUM_THRESHOLD values, measure total time.

## Files to Modify

- `kernels/cp_rasterize.cu` — rewrite: 3 kernels + queue logic
- `kernels/cp_rast_types.h` — add queue structs
- `cp_context.c` — allocate queues, launch 3 kernels, zero counters
- `cp_context.h` — add queue device pointers to context
- `cp_kernels.c` — load new kernel functions
- `cp_kernels.h` — add function handles for stage2/stage3

## Reference Code

- CuRast 3-stage dispatch: `/home/coder/git/CuRast/src/kernels/triangles_visbuffer.cu`
- CuRast queue management: same file, `atomicAdd` into `args.numNontrivial`
- CuRast warp broadcast: same file, `warp.shfl(val, 0)` pattern
- llvmpipe trivial accept/reject: `src/gallium/drivers/llvmpipe/lp_rast_tri_tmp.h`
  (lines 94-120 for `outmask`/`partmask` classification)
- llvmpipe edge offset computation: `lp_setup_tri.c` (search for `eo`)
