# Vertex post-transform reuse B0 census

This branch contains a diagnostic-only population census. It does not contain a
post-transform cache, remap, or production reuse mechanism.

Enable it with the value boolean:

```
CUDAVK_VERTEX_REUSE_CENSUS=1 <application>
```

Enabled timing is invalid. With the flag false, the driver does not compile or
load the census module. It does not allocate census storage, assign census
identities, select banks, scan keys, launch the census kernel, or add a stream
join. The ordinary context cleanup already performs a context-wide join.

## Population and exact key

The hook is after the successful generated vertex shader launch. This one site
covers both paths:

* standalone `cp_vertex_fetch` followed by generated VS `main`;
* vertex fetch inlined into generated VS `main`.

The separate diagnostic kernel receives the same by-value
`cp_vertex_fetch_args` and calls `cp_vf_lane_ids()` again. It does not read or
modify shader output. For every assembled triangle corner, the exact key is:

```
(batch-local draw row, effective vertex_id, instance_id)
```

The row forbids cross-draw reuse. Indexed strip/fan and other pre-expanded refs
are included because the diagnostic uses the same identity routine, not the
index-buffer pointer as a classifier. The `indexed` record field comes from the
original draw state. `fetch_runs` distinguishes classic VS work that had an
actual standalone fetch launch from classic VS work that did not.

Each record identity is the absolute internal command-bearing submit ordinal
plus the absolute ordinal of the `cp_draw_execute_batch()` execution within
that submit. Early-return executions can make batch ordinals sparse. Retries
and fallbacks get new ordinals. Record slots and stream banks are metadata, not
identity.

## Exact table and failures

The main stream and eight segment streams own disjoint banks. Each bank grows
to a power-of-two capacity at no more than 50% nominal load. Old allocations
are retained until cleanup, so growth does not synchronize or race an older
kernel. A generation tag avoids a table clear per batch. Generation zero is
reserved.

An entry has BUSY/READY publication plus the stored row, vertex, and instance.
The hash only selects the first probe. Equality compares all three original
integers, so hash collisions cannot classify two different keys as equal. Key
fields are read only after READY. BUSY waits and probes are bounded.

Allocation, launch, lane, probe, or table failure excludes the whole batch.
The report then charges every vertex in it unique (nonduplicate). This is
conservative against the reuse candidate. Record-array overflow is separately
counted and makes `result_valid=no`; capacity is 262,144 records, above the
about 210,944 executed batches in the full favorite3 replay.

Device result slots are zeroed once. Each batch writes only its own slot. There
is no per-batch CPU read or synchronization. Cleanup first joins the whole CUDA
context, then copies all device results once and prints every record followed
by one summary.

## Interpretation

The analyzer is:

```
python3 src/cudavk/tests/cp_vertex_reuse_analyze.py stderr.txt \
  --expect-final-submit 6947
```

It validates full population, identities, record and summary reconciliation,
conservative exclusions, and the overall/indexed/nonindexed/fused/standalone
fractions. The `64.5%` value is only a generous **population screen** against
the 0.775 ms/frame pool (0.602 generated VS union-exclusive plus 0.173
standalone fetch). Batch costs are not uniform. Therefore a global count ratio,
above or below 64.5%, is not the final decision. A surviving population still
requires an exact clean-trace union-exclusive cost join by `(submit,batch)`.
No production cache follows from B0 alone.

`writes_memory` is recorded. The driver does not currently retain complete
subgroup/clock/query/transform-feedback eligibility metadata, so
`semantic_metadata_complete=0`. B0 is intentionally an all-work upper bound.
Any production reuse design would also need those semantic gates, input
resource identity and content versions, and versioned descriptor/uniform state.

## No-device checks

```
python3 src/cudavk/tests/cp_vertex_reuse_oracle_test.py
python3 src/cudavk/tests/cp_vertex_reuse_analyze_test.py
python3 src/cudavk/tests/cp_vertex_reuse_nvrtc_test.py
python3 src/cudavk/tests/cp_debug_doc.py --check
python3 src/cudavk/tests/cp_no_getenv.py --check
python3 src/cudavk/tests/cp_launch_audit.py
ninja -C build-cudavk
```
