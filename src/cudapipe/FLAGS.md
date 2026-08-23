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

64 switches.

## Tracing

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAPIPE_DEBUG_DRAW` | bool (presence) | `off` | — | trace every draw: geometry, attachments, and why a draw was skipped |
| `CUDAPIPE_DEBUG_TEX` | bool (presence) | `off` | — | trace sampler and texture-handle setup |
| `CUDAPIPE_PLAN_STATS` | bool (value) | `off` | — | report at teardown what deciding one draw at a time costs: batch keys built, pairwise merge tests, flushes, pass episodes closed and every reactive reallocation, with per-scope averages |
| `CUDAPIPE_SPEC_STATS` | bool (value) | `off` | — | report at teardown how many fragment launches used a sampler-specialised kernel, and which shaders sample textures the specialiser could not match; a silent drop to zero is a performance regression with no error |
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
| `CUDAPIPE_NO_OPAQUE_EPISODE` | bool (value) | `off` | — | disable consecutive opaque-run visibility deferral |
| `CUDAPIPE_NO_SAMPLER_VARIANT` | bool (value) | `off` | — | disable literal-state fragment sampler variants |
| `CUDAPIPE_NO_ABUF_SHORT_SORT` | bool (value) | `off` | — | disable the single-thread short A-buffer run sorter |
| `CUDAPIPE_NO_ABUF_WARP_BUCKET` | bool (value) | `off` | — | disable warp-aggregated A-buffer segment bucketing atomics |
| `CUDAPIPE_ABUF_SHORT_SORT_MAX` | uint | `16` | 2&ndash;64 | largest A-buffer run sorted by one thread |
| `CUDAPIPE_FLUSH_DRAIN` | bool (value) | `off` | — | restore the draining flush: cp_flush waits for the whole device and rewinds the arenas in place instead of ping-ponging generations |
| `CUDAPIPE_NO_SEG_MERGE` | bool (value) | `off` | — | disable merged shading groups: every episode segment shades in its own launch group, as before |
| `CUDAPIPE_NO_ABUF_APPEND` | bool (value) | `off` | — | disable the single-pass A-buffer build: the count pass stops appending (pixel, prim) records and the fill rasterizes a second time |
| `CUDAPIPE_NO_FUSED_ABUF_INTERP` | bool (value) | `off` | — | launch A-buffer interpolation separately instead of calling it from the generated fragment shader; performance experiment only |
| `CUDAPIPE_NO_FUSED_INTERP` | bool (value) | `off` | — | restore the direct shade path's separate cp_fs_interpolate launch instead of the slim compaction plus in-shader interpolation |
| `CUDAPIPE_NO_FUSED_RAST` | bool (value) | `off` | — | restore the separate clip and rasterize-stage1 launches instead of the fused clip+classify kernel; stages 2 and 3 are separate either way |
| `CUDAPIPE_FORCE_PASS_FALLBACK` | bool (value) | `off` | — | make every blended episode take the classic re-execution fallback; the output must be identical, which is what makes it a test |
| `CUDAPIPE_TILE_CENSUS` | uint | `0` | 0&ndash;256 | tile edge in pixels for the blended tile-bin shader census; 0 is off, and it changes no rendering |
| `CUDAPIPE_TILE_CENSUS_EVERY` | uint | `2000` | 1&ndash;1000000 | report the tile census this often, in episodes |
| `CUDAPIPE_TILED_OPAQUE` | bool (value) | `off` | — | use the experimental opaque episode sort-middle tile rasterizer |
| `CUDAPIPE_TILED_OPAQUE_CENSUS` | bool (value) | `off` | — | collect opaque tile population statistics without changing rendering |
| `CUDAPIPE_UNSAFE_NO_OVERFLOW` | bool (value) | `off` | — | skip episode overflow readback and assume fragment/quad arrays fit; unsafe diagnostic only |
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
| `CUDAPIPE_TUNE_VETO` | float | `1.0` | — | how much worse the capped build may be before it is refused |
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
