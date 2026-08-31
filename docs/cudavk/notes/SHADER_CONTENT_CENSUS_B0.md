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


## Favorite3 result

The implementation is side-branch commit `8c32c74e0d0`. Static validation
passed the no-device build, 32 focused CPU tests, all three registered Meson
suites, generated flag documentation, the no-`getenv` and launch audits,
Python byte compilation, and `git diff --check`. The flag-off full suite passed
**82/82**: the pre-existing 79 tests plus the three new CPU-only census tests.
Enabled fused and classic smoke runs each produced one complete record with the
same full stage key, correct execution metadata, and no invalidation.
`CUDAVK_SHADER_CENSUS=0` produced no census output.

Two complete favorite3 captures are in `/tmp/shader-census-f3/run1` and
`run2`. Each ended with the capture's expected post-output `rc=139`, had 6,939
complete shim rows plus the allowed torn final numeric fragment, reached
internal command-bearing submit 6,947, retained stdout SHA-256
`e3f24a1dcdc8568be217d249e480623958e2621b3d4f056ae4c44ad20d08da49`,
and matched all 18 sentinels. Both recorded 210,944/210,944 successful
vertex launches with zero drops, missing keys, key mismatches, or unknown
streams. Context synchronization succeeded. Their complete structural records
were exactly equal: 197,416 fused launches, 13,528 classic launches, and 48
full keys process-wide.

The clean-trace join uses internal submits `[2786,3608)`, 411 frames, and
`/tmp/postdead/trace.sqlite`. Both exact census reruns produce byte-identical
`/tmp/shader-census-cost.json` output, SHA-256
`67db0f66cb7904989966cccd24f97de76989f8c2239fef6be6b920c7976493f8`.
It maps every one of 16,842 fused VS launches in the window, grouped into 22
full keys:

| fused pool accounting | ms/frame |
|---|---:|
| summed duration | 0.924079 |
| union | 0.626962 |
| **union-exclusive** | **0.520663** |

The aggregate barely clears the 0.500 admission gate, but content is not
concentrated. The largest single full key owns only **0.209405 ms/frame**
union-exclusive. The best possible set of five keys owns **0.376335**.

An exact set analysis labels every elementary collectible span with all full
keys active on that span. A selected identity set receives the span only when
that required label is a subset, which preserves cross-key overlap rather than
summing per-key exclusives. A subset-zeta transform exhaustively evaluated all
`2^22` sets. No set of 14 or fewer keys reaches 0.500 ms/frame. Exactly two
15-key sets do; their maximum is only **0.502497 ms/frame**. The complete
machine result is `/tmp/shader-census-set-gate.json`, SHA-256
`297c0b7016e5322a0c00781a89b7970f2e75c5a9beb12b0badb33b083f10d6a9`.

This evidence is from RTX 5090/GB202/sm_120, not the primary B200/sm_100.
The lead is therefore **parked, not admitted**. There is no small actionable
content population, and even the broad 15-of-22 set barely passes before any
mechanism cost or attainable-speedup fraction. Do not build a per-content
optimizer, register-cap rule, block-size rule, fused/classic selector, PTX
change, or NIR change from this result. Remeasure on B200/sm_100 first; only a
B200 set that independently clears the gate can reopen design work. Favorite2
was not spent because favorite3 did not pass the architecture and concentration
gates.

Reproduce the clean join with:

```
python3 src/cudavk/tests/cp_shader_census_cost.py \
  --trace /tmp/postdead/trace.sqlite \
  --timestamps /tmp/postdead/submit-ts.txt \
  --census /tmp/shader-census-f3/run1/stderr \
  --submit-lo 2786 --submit-hi 3608 --frames 411 \
  > /tmp/shader-census-cost.json
```
