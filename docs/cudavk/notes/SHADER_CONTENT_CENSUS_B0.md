# Generated fused-VS full-stage identity census (B0)

Status: diagnostic population instrument only. It is not a production cache,
tuning rule, or optimizer. The screen asks whether generated fused vertex cost
is concentrated in a small population of immutable compiled-stage identities.

`CUDAVK_SHADER_CENSUS=1` is a generated value boolean, default false. When it
is false, the driver does not attach cache keys to binaries, allocate records,
assign submit/batch/launch identities, scan records, or report. The launch site
has one cached flag branch. It adds no GPU work, readback, synchronization,
NVTX, module/JIT change, layout change, or batching change.

When enabled, the winning canonical cache binary receives the complete 32-byte
**final resource-aware compiled-stage cache key** inside the insertion lock,
immediately before the cache entry is published. This is not merely the raw
SPIR-V stage digest. Cache hits compare all 32 bytes while holding the same
lock. A racing loser remains untagged and is destroyed. The key is immutable;
no pointer, module handle, prefix, cache ordinal, or selected-set ordinal is an
identity.

A fixed 262,144-row CPU array is allocated once per renderer context. A row is
appended only after a generated vertex `cp_launch` returns `CUDA_SUCCESS`.
It records the full key, explicit `stage=VS`, absolute command-bearing submit,
executed-batch ordinal, host launch ordinal, CLASSIC/VS_FETCH execution,
fused/fetch facts, total vertices, grid/block, draw count, stable main/seg0..7
stream lane, active execution resources, and SM target. The append does CPU
stores only. Allocation failure, capacity drops, a missing/mismatched key, an unknown
stream, or failure of cleanup's existing context synchronize invalidates the
report but never changes rendering.

Cleanup prints machine-readable records after the existing context synchronize.
The strict CPU analyzer requires a complete nonempty population, 64 lowercase
hex digits for every key, monotonic unique absolute identities, exact summary
markers, and identical reruns. Every report states
`diagnostic_timing_invalid=1`, `cost_join_required=yes`, and
`production_admitted=no`.

`cp_shader_census_cost.py` performs the later CPU-only join to a clean nsys
SQLite trace. It maps generated VS `main` by immediate same-stream predecessor
and successor, absolute `(submit,batch)`, grid, and stable lane. It accepts only
fused rows and reports launch count, summed duration, union, and union-exclusive cost both
for the whole mapped fused pool and grouped by the complete cache key. An RTX 5090 result is evidence for that GPU
only. B200/sm_100 is primary, and no architecture-dependent conclusion is
admitted before the B200 rerun.
