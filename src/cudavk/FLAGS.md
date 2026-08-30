# The driver's environment switches

**Generated from the registry in `cp_debug.c` by
`tests/cp_debug_doc.py`. Do not edit by hand — edit the registry.**

`CUDAVK_HELP=1` prints the same table from a running driver, with the
value each variable actually resolved to in that process, which is the
form to use when the question is what a run was configured to do.

Two boolean kinds appear here and the difference bites:

- **bool (presence)** is set by the variable existing at all, so
  `CUDAVK_DEBUG_DRAW=0` turns tracing **on**.
- **bool (value)** reads the value, so `=0` turns it off.

That is not a design, it is what the flags grew into, and it is preserved
deliberately: someone's script sets one of these to 0 today.

126 switches.

## Tracing

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_DEBUG_DRAW` | bool (presence) | `off` | — | trace every draw: geometry, attachments, and why a draw was skipped |
| `CUDAVK_DEBUG_TEX` | bool (presence) | `off` | — | trace sampler and texture-handle setup |
| `CUDAVK_PLAN_STATS` | bool (value) | `off` | — | report at teardown what deciding one draw at a time costs: batch keys built, pairwise merge tests, flushes, pass episodes closed and every reactive reallocation, with per-scope averages |
| `CUDAVK_SPEC_STATS` | bool (value) | `off` | — | report at teardown how many fragment launches used a sampler-specialised kernel, and which shaders sample textures the specialiser could not match; a silent drop to zero is a performance regression with no error |
| `CUDAVK_TEXTURE_CACHE_BUDGET_MB` | uint | `2048` | — | hardware texture cache byte budget in MiB; an allocation past it falls back to software sampling silently, which cost 1.76 ms/frame on the occlusion capture at the old default of 384 |
| `CUDAVK_TEXTURE_CACHE_STATS` | bool (value) | `off` | — | report hardware-texture fragment execution hits and software fallbacks |
| `CUDAVK_DEBUG_VFETCH` | bool (presence) | `off` | — | dump what the GPU vertex fetch gathered; syncs, so debug-only |
| `CUDAVK_DEBUG_WORK` | bool (presence) | `off` | — | report how much of the shading launch did work; syncs |
| `CUDAVK_DEBUG_DISCARD` | bool (presence) | `off` | — | report covered and discarded fragment counts per pass; syncs |
| `CUDAVK_DEBUG_FS` | bool (presence) | `off` | — | dump fragment-shader inputs and outputs per pixel; syncs |
| `CUDAVK_DEBUG_FS_VSTEP` | uint | `1` | &ge; 1 | with DEBUG_FS, print every Nth pixel |
| `CUDAVK_DEBUG_FS_ROW` | int | `-1` | — | with DEBUG_FS, restrict the dump to one framebuffer row |
| `CUDAVK_DEBUG_LAUNCH` | bool (presence) | `off` | — | trace compute dispatch: bound UBO and SSBO pointers |
| `CUDAVK_DEBUG_TIME` | bool (presence) | `off` | — | time the pipeline stages |
| `CUDAVK_DEBUG_BATCH` | bool (presence) | `off` | — | report why each draw batch ended |
| `CUDAVK_DEBUG_BATCHDIFF` | bool (presence) | `off` | — | report which state field broke a batch, field by field |
| `CUDAVK_DEBUG_PASSSEQ` | bool (value) | `off` | — | log one line per framebuffer bind and per draw: shaders, blendedness, eligibility — the raw material for pass-structure statistics |
| `CUDAVK_FRAG_CENSUS` | bool (presence) | `off` | — | count fragments per draw; also compiles the instrumented kernels in |
| `CUDAVK_NVTX` | bool (presence) | `off` | — | push an NVTX range around each draw and stage, for nsys |
| `CUDAVK_DEBUG_ROWS` | bool (presence) | `off` | — | trace the per-draw tables a batch builds: fragment UBO slots and the vertex slice table the fragment stage searches |
| `CUDAVK_DEBUG_CLIP` | bool (presence) | `off` | — | report what each batch hands the clip stage: triangles, draws, whether the stable slot mode is on |
| `CUDAVK_DEBUG_PASS` | bool (presence) | `off` | — | report, for the first three batches only, every condition that decides whether a blended batch may join the open pass episode |
| `CUDAVK_DEBUG_EPISODE` | bool (presence) | `off` | — | trace episode boundaries: what appended to an episode and what cut it |
| `CUDAVK_DEBUG_RT` | bool (presence) | `off` | — | trace descriptor writes, the attachments a render pass resolved to, and the first texels either side of an image copy; the copies sync |
| `CUDAVK_DEBUG_FACES` | bool (presence) | `off` | — | when a cube descriptor is written, read back the first texel of each of its six faces from where the sampler will look; syncs |

## Subsystem switches

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_NO_ABUFFER` | bool (presence) | `off` | — | disable the A-buffer; blended draws go back to the direct path |
| `CUDAVK_NO_ABUF_BATCH` | bool (presence) | `off` | — | disable batching of A-buffer draws |
| `CUDAVK_NO_PASS_EPISODE` | bool (value) | `off` | — | disable pass episodes: consecutive blended batches stop sharing one A-buffer build and drain |
| `CUDAVK_UPLOAD_STATS` | bool (value) | `off` | — | count small host-to-device copies, clears, context syncs and upload ring wraps per call site, and report them at teardown |
| `CUDAVK_NO_META_FOLD` | bool (value) | `off` | — | upload the vertex stage's vcount and stride as their own block again instead of carrying them in the argument block's scalar area |
| `CUDAVK_NO_COUNTER_BLOCK` | bool (value) | `off` | — | clear the A-buffer's scalar counters one at a time again instead of clearing the whole counter block once per draw or episode |
| `CUDAVK_NO_UPLOAD_COALESCE` | bool (value) | `off` | — | send one host-to-device copy per upload block again instead of one per launch boundary |
| `CUDAVK_UPLOAD_FLUSH_FAIL_AT` | uint | `0` | — | fault injection: fail the Nth coalesced upload flush |
| `CUDAVK_NO_FETCH_FOLD` | bool (value) | `off` | — | clear the clip and raster queue counters with their own device clears again instead of seeding them inside cp_vertex_fetch |
| `CUDAVK_FUSED_VFETCH` | bool (value) | `on` | — | gather vertex attributes inside the vertex shader, from bitcode inlined into it, instead of launching cp_vertex_fetch before it |
| `CUDAVK_NO_FUSED_VFETCH` | bool (value) | `off` | — | do not even build the fused vertex execution: the resource-isolated control, and the revert once the fusion is the default |
| `CUDAVK_FS_GRID_WAVES` | uint | `0` | — | size the fragment grid to fill the machine this many times over and let it grid-stride, instead of covering the framebuffer's worst case; 0 keeps the 4096-block cap, which is the only bound either way |
| `CUDAVK_VFETCH_DECLINE_NTH` | uint | `0` | — | fault injection: refuse the fused execution for the Nth vertex shader compiled, or for every one at 4294967295, after building it either way |
| `CUDAVK_VFETCH_SKIP_SEED` | bool (value) | `off` | — | fault injection: a fused draw does not seed the clip and raster counters, and the host still skips their clears -- the negative control for the seeding moving into the fused vertex shader |
| `CUDAVK_NO_OPAQUE_EPISODE` | bool (value) | `off` | — | disable consecutive opaque-run visibility deferral |
| `CUDAVK_NO_OPAQUE_STREAMS` | bool (value) | `off` | — | issue an opaque episode's segments back to back on the main stream instead of fanning them over the pass side streams |
| `CUDAVK_NO_PDL` | bool (value) | `off` | — | issue every kernel with an ordinary launch again, so that no dependent kernel starts before its predecessor in the same stream has drained; the revert for the programmatic dependent launch default |
| `CUDAVK_PDL` | uint | `3` | — | how much of the driver runs its dependent kernels with the programmatic stream serialization attribute, so that a kernel may start before its predecessor on the same stream has drained: 0 none, 1 the A-buffer scan chain, 2 also the rasterizer stage links, 3 also the fragment writeback and the segment scatter, 4 also offers stage 1, which is a diagnostic rather than a default: most of those links are expected to refuse themselves and the counters are the measurement. The default is 3; lower it to bisect a regression, raise it to 4 to measure stage 1, and see CUDAVK_NO_PDL for the plain revert. Needs compute capability 9.0 and CUDA 11.8, and any link whose predecessor turns out not to be the named kernel falls back to an ordinary launch |
| `CUDAVK_NO_SAMPLER_VARIANT` | bool (value) | `off` | — | disable literal-state fragment sampler variants |
| `CUDAVK_NO_TEXTURE_GATHER` | bool (value) | `off` | — | refuse every textureGather() again, so that it returns the constant 0 0 0 1 the driver returned before the sampler could serve one |
| `CUDAVK_NO_GPU_SEM_WAIT` | bool (value) | `off` | — | block the submitting thread on a queue-submit wait semaphore instead of making the renderer stream wait on its completion event |
| `CUDAVK_NO_TEXTURE_CACHE` | bool (value) | `off` | — | disable direct CUDA hardware-texture fragment execution; fall back to software sampling |
| `CUDAVK_TEXTURE_CACHE_FAIL_TABLE_UPLOAD_AT` | uint | `0` | — | fault injection: fail the Nth hardware texture table upload enqueue |
| `CUDAVK_TEXTURE_CACHE_PURGE_AT_PREFLIGHT` | uint | `0` | — | fault injection: purge derived textures before the Nth HW preflight |
| `CUDAVK_TEXTURE_CACHE_FAIL_AUTHORITATIVE_ALLOC_AT_PREFLIGHT` | uint | `0` | — | fault injection: OOM one renderer allocation at the Nth HW preflight |
| `CUDAVK_TEXTURE_CACHE_FAIL_AFTER_BOUNDED_GROUP0` | bool (value) | `off` | — | fault injection: latch fatal after bounded episode FS group zero |
| `CUDAVK_TEXTURE_CACHE_FAIL_AFTER_MAIN_GROUP0` | bool (value) | `off` | — | fault injection: latch fatal after main episode FS group zero |
| `CUDAVK_TEXTURE_CACHE_FAIL_AFTER_FS_ENQUEUE` | bool (value) | `off` | — | fault injection: sync then latch fatal after one hardware FS enqueue |
| `CUDAVK_TEXTURE_CACHE_FAIL_ARRAY_ALLOC_AT` | uint | `0` | — | fault injection: OOM the Nth derived CUDA array allocation |
| `CUDAVK_TEXTURE_CACHE_FAIL_OBJECT_CREATE_AT` | uint | `0` | — | fault injection: OOM the Nth CUDA texture-object creation |
| `CUDAVK_TEXTURE_CACHE_FAIL_CONVERSION_ENQUEUE_AT` | uint | `0` | — | fault injection: fail the Nth converted-cache kernel enqueue fatally |
| `CUDAVK_TEXTURE_CACHE_FAIL_FS_ARG_BEGIN` | bool (value) | `off` | — | fault injection: fail first hardware FS-argument reservation fatally |
| `CUDAVK_TEXTURE_CACHE_FAIL_CREATE_STAGE` | uint | `0` | — | fault injection: fail cache create stage 1=event or 2=surface |
| `CUDAVK_NO_ABUF_SHORT_SORT` | bool (value) | `off` | — | disable the single-thread short A-buffer run sorter |
| `CUDAVK_NO_ABUF_WARP_BUCKET` | bool (value) | `off` | — | disable warp-aggregated A-buffer segment bucketing atomics |
| `CUDAVK_ABUF_SHORT_SORT_MAX` | uint | `16` | 2&ndash;64 | largest A-buffer run sorted by one thread |
| `CUDAVK_NO_ABUF_FUSE_SCAN` | bool (value) | `off` | — | restore the three- or five-launch A-buffer prefix sum instead of the fused reduce plus finish, and its separate fill-cursor clear |
| `CUDAVK_NO_ABUF_FUSE_QUAD` | bool (value) | `off` | — | restore the A-buffer quad build's separate block worklist, counting pass, three-launch scan and blk_counts clear |
| `CUDAVK_ABUF_FUSE_CHECK` | bool (value) | `off` | — | run the classic A-buffer scan and quad count beside the fused ones and compare every element on the device; gate only, never timed |
| `CUDAVK_ABUF_FUSE_BREAK` | uint | `0` | 0&ndash;2 | negative control for the fusion check: 1 drops the block base in the fused scan, 2 stops the fused quad count writing an uncovered zero |
| `CUDAVK_FLUSH_DRAIN` | bool (value) | `off` | — | restore the draining flush: cp_flush waits for the whole device and rewinds the arenas in place instead of ping-ponging generations |
| `CUDAVK_NO_SEG_MERGE` | bool (value) | `off` | — | disable merged shading groups: every episode segment shades in its own launch group, as before |
| `CUDAVK_NO_ABUF_APPEND` | bool (value) | `off` | — | disable the single-pass A-buffer build: the count pass stops appending (pixel, prim) records and the fill rasterizes a second time |
| `CUDAVK_NO_FUSED_ABUF_INTERP` | bool (value) | `off` | — | launch A-buffer interpolation separately instead of calling it from the generated fragment shader; performance experiment only |
| `CUDAVK_NO_FUSED_INTERP` | bool (value) | `off` | — | restore the direct shade path's separate cp_fs_interpolate launch instead of the slim compaction plus in-shader interpolation |
| `CUDAVK_NO_INLINE_FS` | bool (value) | `off` | — | restore separate interpolation plus the resource-isolated classic fragment binary instead of same-LLVM inline interpolation |
| `CUDAVK_INLINE_FS` | bool (value) | `off` | — | opt in to same-LLVM fragment interpolation; default stays on the pre-inline fused path because dual module ownership is expensive |
| `CUDAVK_FORCE_FUSED_FS` | bool (value) | `off` | — | force pre-inline fused interpolation execution with its isolated tuner (overrides NO_FUSED flags; NO_INLINE_FS takes precedence) |
| `CUDAVK_NO_FUSED_RAST` | bool (value) | `off` | — | restore the separate clip and rasterize-stage1 launches instead of the fused clip+classify kernel; stages 2 and 3 are separate either way |
| `CUDAVK_NO_PRIM_REFS` | bool (value) | `off` | — | copy every accepted clipped primitive into contiguous scratch instead of publishing a reference to immutable vertex-shader output |
| `CUDAVK_NO_SETUP_CACHE` | bool (value) | `off` | — | recompute huge-primitive setup independently in every stage-3 tile |
| `CUDAVK_FORCE_PASS_FALLBACK` | bool (value) | `off` | — | make every blended episode take the classic re-execution fallback; the output must be identical, which is what makes it a test |
| `CUDAVK_TILE_CENSUS` | uint | `0` | 0&ndash;256 | tile edge in pixels for the blended tile-bin shader census; 0 is off, and it changes no rendering |
| `CUDAVK_TILE_CENSUS_EVERY` | uint | `2000` | 1&ndash;1000000 | report the tile census this often, in episodes |
| `CUDAVK_TILED_OPAQUE` | bool (value) | `off` | — | use the experimental opaque episode sort-middle tile rasterizer |
| `CUDAVK_TILED_OPAQUE_CENSUS` | bool (value) | `off` | — | collect opaque tile population statistics without changing rendering |
| `CUDAVK_UNSAFE_NO_OVERFLOW` | bool (value) | `off` | — | skip episode overflow readback and assume fragment/quad arrays fit; unsafe diagnostic only |
| `CUDAVK_UNSAFE_FORCE_OPAQUE` | bool (value) | `off` | — | force every draw's blend state to disabled and let the result reach an opaque episode; renders wrong output, diagnostic upper bound for the cost of the blended path |
| `CUDAVK_NO_BATCH` | bool (presence) | `off` | — | disable draw batching entirely |
| `CUDAVK_NO_BATCH_BLEND` | bool (presence) | `off` | — | batch opaque draws only: a blended draw never joins a batch, so it never reaches a pass episode and builds, sorts and peels its own A-buffer (23.49 ms against 8.79 on Crossroads) |
| `CUDAVK_NO_MERGE_SCISSOR` | bool (presence) | `off` | — | require an equal scissor before two blended draws merge, as the front end did before it trusted the renderer's per-draw rectangles |
| `CUDAVK_KEEP_VOFF` | bool (presence) | `off` | — | put the vertex offset back in the merge key; it is the largest merge blocker there is, 13,281 separations of 38,155 on Crossroads |
| `CUDAVK_KEEP_INSTKEY` | bool (presence) | `off` | — | put the instance count back in the merge key, which the renderer's per-draw instance_counts[] row makes unnecessary |
| `CUDAVK_KEEP_PUSHKEY` | bool (presence) | `off` | — | put the push-constant block back in the merge key, so draws that push different constants stop merging |
| `CUDAVK_WAIT_SPIN_US` | int (optional) | `unset` | — | conversion probe: busy-wait this many microseconds after each episode drain returns; the slope of frame time against injected time is the site's conversion factor |
| `CUDAVK_WAIT_SPIN_BEFORE_US` | int (optional) | `unset` | — | conversion probe: busy-wait this many microseconds before each episode drain's sync; time absorbed by the wait rather than added to the frame is what a deferral mechanism could relocate for free |
| `CUDAVK_KEEP_IBKEY` | bool (presence) | `off` | — | put the index buffer back in the merge key, which the slice table's per-draw base makes unnecessary; 23,276 separations on the occlusion capture |
| `CUDAVK_KEEP_VBKEY` | bool (presence) | `off` | — | put the vertex-buffer bindings back in the merge key; the per-draw elem_bases rows carry them, and this was the largest merge blocker on the occlusion capture, 144,008 separations of 332,027 |
| `CUDAVK_NO_BINCACHE` | bool (presence) | `off` | — | disable the compiled-kernel binary cache |

## A-buffer

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_ABUFFER_VERIFY` | bool (value) | `off` | — | check A-buffer output against the direct path; excludes compositing |
| `CUDAVK_ABUFFER_VERIFY_DRAWS` | uint | `8` | — | how many draws to verify before giving up |
| `CUDAVK_ABUFFER_COMPOSITE` | bool (value) | `on` | — | composite the A-buffer; defaults on unless VERIFY is set |
| `CUDAVK_ABUFFER_TIMING` | bool (value) | `off` | — | per-draw CUDA-event breakdown; costs a drain per draw |
| `CUDAVK_ABUFFER_DEBUG` | bool (value) | `off` | — | report which draws were eligible for the A-buffer, and why not |
| `CUDAVK_ABUFFER_LAYERS` | uint | `0` | — | cap A-buffer layers per pixel; 0 uses the built-in limit |
| `CUDAVK_ABUF_MIN_TRIS` | uint | `0` | &ge; 0 | triangles a blended batch must have before the A-buffer's fixed cost is worth paying; 0, the default, means always, and is deliberate — with the bootstrap scan fixed Crossroads measured 10.9 ms at 0 against 33.0 at 256 |
| `CUDAVK_ABUF_COMPILE` | bool (optional) | `unset` | — | force the A-buffer branches in (1) or out (0) of the NVRTC build |

## Draw batching

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_BATCH_MAX` | uint | `128` | 1&ndash;128 | cap draws per batch; 1 must stay bit-identical to NO_BATCH |

## CUDA context

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_CTX_SCHED` | enum | `auto` | — | GPU-wait behaviour: auto \| spin \| yield \| blocking; blocking only helps on an idle machine, yield only under CPU contention |
| `CUDAVK_CTX_CHECK` | bool (presence) | `off` | — | verify every CUDA call runs in this device's context; names the caller that escaped an entry-point scope |

## Small-allocation arena

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_SMALL_ALLOC` | enum | `advise` | — | arena residency: advise \| pinned \| managed \| blocksonly \| off |
| `CUDAVK_SMALL_ALLOC_MAX` | uint64 | `128` | 0&ndash;1024 | allocations up to this many bytes come from the arena |
| `CUDAVK_SMALL_ALLOC_WARMUP` | uint64 | `64` | — | allocations to let past before the arena opens |
| `CUDAVK_SMALL_ALLOC_STATS` | bool (presence) | `off` | — | dump the allocation mix and arena hit rates at exit |

## Shader compilation

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_NO_REGCAP` | bool (presence) | `off` | — | disable the register cap and the occupancy trial that tunes it |
| `CUDAVK_REGCAP_STATIC` | bool (presence) | `off` | — | cap registers from a static estimate instead of the trial |
| `CUDAVK_MAX_REGISTERS` | uint | `0` | — | force a register cap on every shader; 0 leaves it to the driver |
| `CUDAVK_LAUNCH_BOUNDS` | uint | `0` | — | emit maxntidx metadata with this block size; 0 emits none |
| `CUDAVK_TUNE_VETO` | float | `1.0` | — | how much worse the capped build may be before it is refused |
| `CUDAVK_SHADER_STATS` | bool (presence) | `off` | — | report register counts and occupancy-trial outcomes |
| `CUDAVK_DUMP_NIR` | bool (presence) | `off` | — | print each shader's NIR |
| `CUDAVK_DUMP_IR` | bool (presence) | `off` | — | print each shader's LLVM IR |
| `CUDAVK_DUMP_PTX` | bool (presence) | `off` | — | print each shader's generated PTX |
| `CUDAVK_SCALARIZE` | enum | `all` | — | which ALU operations to scalarise: all \| none \| basic \| sel \| alu \| move \| intr \| rest; all is the NULL-filter pass every sample is rendered with, and the classes are there to bisect a backend bug |
| `CUDAVK_NO_HOIST_INPUTS` | bool (presence) | `off` | — | stop hoisting a vertex shader's input loads above the arithmetic that consumes them; that hoist is 6.14 against 8.63 ms on instancing |
| `CUDAVK_NO_REG_SSA` | bool (presence) | `off` | — | leave NIR registers alone before the backend, so each becomes an alloca and the NVPTX backend gives the kernel a __local_depot |
| `CUDAVK_KEEP_SMALL_DYNAMIC_REGCAP` | bool (presence) | `off` | — | keep the tuned register cap on small fragment shaders that call the dynamic sampler helper; they cannot amortise its spills |

## Rasterizer tuning (NVRTC -D options)

| variable | type | default | range | meaning |
|---|---|---|---|---|
| `CUDAVK_SMALL_THRESHOLD` | int (optional) | `unset` | — | triangle area below which the small-primitive path is used |
| `CUDAVK_MEDIUM_THRESHOLD` | int (optional) | `unset` | — | triangle area below which the medium-primitive path is used |
| `CUDAVK_POINT_THRESHOLD` | int (optional) | `unset` | — | triangle area below which a primitive is rasterized as a point |
| `CUDAVK_TILE_BOUND` | int (optional) | `unset` | — | 0 walks the whole tile; 1 walks only the primitive's bounding box |
