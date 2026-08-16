# The driver's environment switches

**Generated from the registry in `cp_debug.c` by
`tests/cp_debug_doc.py`. Do not edit by hand — edit the registry.**

`CUDAPIPE_HELP=1` prints the same table from a running driver, with the
value each variable actually resolved to in that process, which is the
form to use when the question is what a run was configured to do.

Two boolean kinds appear here and the difference bites:

- **bool (presence)** is set by the variable existing at all, so
  `CUDAPIPE_DEBUG_DRAW=0` turns tracing **on**.
- **bool (value)** reads the value, so `=0` turns it off.

That is not a design, it is what the flags grew into, and it is preserved
deliberately: someone's script sets one of these to 0 today.

47 switches.

## Tracing

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_DEBUG_DRAW` | bool (presence) | `off` | — | trace every draw: geometry, attachments, and why a draw was skipped |
| `CUDAPIPE_DEBUG_TEX` | bool (presence) | `off` | — | trace sampler and texture-handle setup |
| `CUDAPIPE_DEBUG_VFETCH` | bool (presence) | `off` | — | dump what the GPU vertex fetch gathered; syncs, so debug-only |
| `CUDAPIPE_DEBUG_WORK` | bool (presence) | `off` | — | report how much of the shading launch did work; syncs |
| `CUDAPIPE_DEBUG_DISCARD` | bool (presence) | `off` | — | report covered and discarded fragment counts per pass; syncs |
| `CUDAPIPE_DEBUG_FS` | bool (presence) | `off` | — | dump fragment-shader inputs and outputs per pixel; syncs |
| `CUDAPIPE_DEBUG_FS_VSTEP` | uint | `1` | &ge; 1 | with DEBUG_FS, print every Nth pixel |
| `CUDAPIPE_DEBUG_FS_ROW` | int | `-1` | — | with DEBUG_FS, restrict the dump to one framebuffer row |
| `CUDAPIPE_DEBUG_LAUNCH` | bool (presence) | `off` | — | trace compute dispatch: bound UBO and SSBO pointers |
| `CUDAPIPE_DEBUG_TIME` | bool (presence) | `off` | — | time the pipeline stages |
| `CUDAPIPE_DEBUG_BATCH` | bool (presence) | `off` | — | report why each draw batch ended |
| `CUDAPIPE_DEBUG_BATCHDIFF` | bool (presence) | `off` | — | report which state field broke a batch, field by field |
| `CUDAPIPE_DEBUG_PASSSEQ` | bool (value) | `off` | — | log one line per framebuffer bind and per draw: shaders, blendedness, eligibility — the raw material for pass-structure statistics |
| `CUDAPIPE_FRAG_CENSUS` | bool (presence) | `off` | — | count fragments per draw; also compiles the instrumented kernels in |
| `CUDAPIPE_NVTX` | bool (presence) | `off` | — | push an NVTX range around each draw and stage, for nsys |

## Subsystem switches

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_NO_ABUFFER` | bool (presence) | `off` | — | disable the A-buffer; blended draws go back to the direct path |
| `CUDAPIPE_NO_ABUF_BATCH` | bool (presence) | `off` | — | disable batching of A-buffer draws |
| `CUDAPIPE_NO_PASS_EPISODE` | bool (value) | `off` | — | disable pass episodes: consecutive blended batches stop sharing one A-buffer build and drain |
| `CUDAPIPE_FLUSH_DRAIN` | bool (value) | `off` | — | restore the draining flush: cp_flush waits for the whole device and rewinds the arenas in place instead of ping-ponging generations |
| `CUDAPIPE_NO_ABUF_APPEND` | bool (value) | `off` | — | disable the single-pass A-buffer build: the count pass stops appending (pixel, prim) records and the fill rasterizes a second time |
| `CUDAPIPE_NO_BATCH` | bool (presence) | `off` | — | disable draw batching entirely |
| `CUDAPIPE_NO_BINCACHE` | bool (presence) | `off` | — | disable the compiled-kernel binary cache |

## A-buffer

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_ABUFFER_VERIFY` | bool (value) | `off` | — | check A-buffer output against the direct path; excludes compositing |
| `CUDAPIPE_ABUFFER_VERIFY_DRAWS` | uint | `8` | — | how many draws to verify before giving up |
| `CUDAPIPE_ABUFFER_COMPOSITE` | bool (value) | `on` | — | composite the A-buffer; defaults on unless VERIFY is set |
| `CUDAPIPE_ABUFFER_TIMING` | bool (value) | `off` | — | per-draw CUDA-event breakdown; costs a drain per draw |
| `CUDAPIPE_ABUFFER_DEBUG` | bool (value) | `off` | — | report which draws were eligible for the A-buffer, and why not |
| `CUDAPIPE_ABUFFER_LAYERS` | uint | `0` | — | cap A-buffer layers per pixel; 0 uses the built-in limit |
| `CUDAPIPE_ABUF_COMPILE` | bool (optional) | `unset` | — | force the A-buffer branches in (1) or out (0) of the NVRTC build |

## Draw batching

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_BATCH_MAX` | uint | `128` | 1&ndash;128 | cap draws per batch; 1 must stay bit-identical to NO_BATCH |

## Small-allocation arena

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_SMALL_ALLOC` | enum | `advise` | — | arena residency: advise \| pinned \| managed \| blocksonly \| off |
| `CUDAPIPE_SMALL_ALLOC_MAX` | uint64 | `128` | 0&ndash;1024 | allocations up to this many bytes come from the arena |
| `CUDAPIPE_SMALL_ALLOC_WARMUP` | uint64 | `64` | — | allocations to let past before the arena opens |
| `CUDAPIPE_SMALL_ALLOC_STATS` | bool (presence) | `off` | — | dump the allocation mix and arena hit rates at exit |

## Shader compilation

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_NO_REGCAP` | bool (presence) | `off` | — | disable the register cap and the occupancy trial that tunes it |
| `CUDAPIPE_REGCAP_STATIC` | bool (presence) | `off` | — | cap registers from a static estimate instead of the trial |
| `CUDAPIPE_MAX_REGISTERS` | uint | `0` | — | force a register cap on every shader; 0 leaves it to the driver |
| `CUDAPIPE_LAUNCH_BOUNDS` | uint | `0` | — | emit maxntidx metadata with this block size; 0 emits none |
| `CUDAPIPE_TUNE_VETO` | float | `1.05` | — | how much worse the capped build may be before it is refused |
| `CUDAPIPE_SHADER_STATS` | bool (presence) | `off` | — | report register counts and occupancy-trial outcomes |
| `CUDAPIPE_DUMP_NIR` | bool (presence) | `off` | — | print each shader's NIR |
| `CUDAPIPE_DUMP_IR` | bool (presence) | `off` | — | print each shader's LLVM IR |
| `CUDAPIPE_DUMP_PTX` | bool (presence) | `off` | — | print each shader's generated PTX |

## Rasterizer tuning (NVRTC -D options)

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_SMALL_THRESHOLD` | int (optional) | `unset` | — | triangle area below which the small-primitive path is used |
| `CUDAPIPE_MEDIUM_THRESHOLD` | int (optional) | `unset` | — | triangle area below which the medium-primitive path is used |
| `CUDAPIPE_POINT_THRESHOLD` | int (optional) | `unset` | — | triangle area below which a primitive is rasterized as a point |
| `CUDAPIPE_TILE_BOUND` | int (optional) | `unset` | — | 0 walks the whole tile; 1 walks only the primitive's bounding box |
