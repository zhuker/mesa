# Current-polygon clip-copy bypass (default off)

Independent experiment from snapshot `69a58a486d4`. Enable with
`CUDAVK_CLIP_COPY_BYPASS=1`; unset or `=0` restores the original source path.
No performance claim. No GPU was used during implementation.

## Contract

Before each plane, scan the CURRENT polygon with the exact existing
`clip_dist(...) >= 0.0f` predicate. If every vertex passes, the old clipper
copies exactly n vertices in order and performs no interpolation. Retaining
src therefore preserves every slot's bits. n starts at 3, is capped at 10 by
clip_poly, and a count below 3 returns immediately after each executed plane.
Thus a bypass cannot skip capacity truncation or retirement.

Use `!(distance >= 0)` for failure: `distance < 0` would classify NaN
incorrectly. Infinities and overflow follow the old comparisons; no finite-only
shortcut or epsilon is added. Signed zeros and varying NaN payloads are copied
unchanged. Distances are recalculated on crossing planes, but the crossing
clipper's arithmetic, order and publication are untouched. Floating-point
exception flags are not a driver-visible output.

After a bypass, plane parity no longer describes the source buffer. Select
poly_b only when src is poly_a, otherwise select poly_a (including original
VS input). This keeps source and destination distinct. The all-inside triangle
fast path, fan triangulation, output publication, counters, primitive IDs and
slot retirement are unchanged. No launch, allocation, worklist or scheduling
change. Only raster NVRTC compilation gets the new macro; compile options are
already included in the cache key. The default arm compiles out the change.

## Prior art

Read DEAD_ENDS 3 and `/tmp/perf16/iter9-{rejected.patch,report.md}`. Iteration 9
split trivial work into separate kernels and also classified common-plane
rejects in its fused fallback. This experiment does neither. It only avoids
copies on individually unchanged current-polygon planes. The separate
clipreject investigation and its CPU-source test methodology were reviewed;
no reject code was imported. Prior timing does not measure this mechanism.

## CPU-only checks

`tests/cp_clip_copy_bypass.py` compiles the actual helper source in both arms
and compares the complete byte-level output and normalized reference pointers.
Includes all 16 stable/compact, active-ID, reference/materialized and fused-ID
modes; 7 capacities (0, 1, 2, 8, 9, 10, 16); slots 1/4/16 with unique raw-bit
varyings; 2000 numerical random and 1000 raw-bit random triangles; seven planes,
near/far crossings with an intervening bypass (alias regression), NaN/Inf,
overflow, signed zero and subnormal coordinates. Registered as CPU-only Meson
test. Passed (Meson 1/1). Full build-cudavk ICD build, registry documentation
and no-getenv audits passed.

Offline CUDA 12.8 `nvcc -cubin -arch=sm_120 -std=c++14 -Xptxas=-v`
compiled both macro arms. This is not runtime NVRTC resource evidence:

| kernel | registers off/on | stack bytes off/on | spills off/on |
|---|---:|---:|---:|
| cp_clip_rast_fused | 56/55 | 5248/5248 | 0/0 |
| cp_clip_rast_fused_abuf | 56/53 | 5248/5248 | 0/0 |
| cp_clip_triangles | 44/44 | 5120/5120 | 0/0 |
| cp_run_clip_setup_direct | 72/72 | 5152/5152 | 0/0 |
| cp_run_clip_setup_tile_count | 64/64 | 5152/5152 | 0/0 |

Artifacts: `/tmp/clipcopy-{off,on}.cubin`,
`/tmp/clipcopy-{off,on}-codegen.log`, `/tmp/clipcopy-test.log`,
`/tmp/clipcopy-build.log`.

## Remaining risks / gates

The added distance scan costs work on crossing planes. Branch divergence and
compiler scheduling may outweigh removed copies. CPU floating-point checks
are not CUDA device equivalence or external-renderer image acceptance. Parent
owns GPU validation: full pipeline A/B on both captures, exact completion and
hash checks, external image gates, and device-side equivalence remain required.
Do not accept or enable by default from offline resource counts.
