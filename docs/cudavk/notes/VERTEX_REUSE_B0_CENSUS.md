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

## Favorite3 result

The implementation is side-branch commit `5445711f243`. Two full compiled
replays are in `/tmp/vertex-reuse-f3/run1` and `run2`. Both ended with the
capture's benign post-output `rc=139`, exactly 6,939 complete external shim
rows and final internal submit 6,947. Both kept stdout SHA-256
`e3f24a1dcdc8568be217d249e480623958e2621b3d4f056ae4c44ad20d08da49`
and matched all 18 sentinel frames byte for byte.

Every one of the 210,944 executed batches was admitted. There were no record,
table, launch, lane or probe exclusions. The two runs agreed on every
structural field and every exact unique count; only the diagnostic probe count
varied with atomic scheduling. The whole-process population was 3,545,618,907
assembled inputs, of which 2,160,968,089 were duplicates (**60.948%**).

For real work, beginning at internal submit 2,784, the count-only population
was:

| population | total inputs | duplicates | duplicate fraction |
|---|---:|---:|---:|
| all real batches | 3,479,473,815 | 2,116,760,651 | **60.836%** |
| fused VS/fetch | 3,426,230,853 | 2,076,471,075 | **60.605%** |
| standalone fetch + classic VS | 53,242,962 | 40,289,576 | **75.671%** |
| indexed | 3,117,299,295 | 2,106,766,521 | **67.583%** |
| nonindexed, including expanded topology | 362,174,520 | 9,994,130 | **2.759%** |

These ratios are population facts, not the decision metric.

## Clean-trace cost join and decision

`cp_vertex_reuse_cost.py` joins both exact runs to the calibrated 411-frame
window in `/tmp/postdead/trace.sqlite`. The trace has no NVTX table. The
fail-closed structural fallback classifies a generated `main` only when its
immediate same-stream kernel successor is `cp_clip_triangles`,
`cp_clip_rast_fused` or `cp_clip_rast_fused_abuf`; immediate same-stream
`cp_vertex_fetch` is the standalone predecessor. Runtime launch APIs join to
GPU kernels only through `correlationId`.

The join maps all **19,929/19,929** VS records in internal submits
`[2786,3608)`, plus all **3,087/3,087** standalone fetches. Every absolute
`(submit,batch)`, grid, bank-to-stream mapping and fetch shape agrees. Exact
integer calibration and the full fractional result have SHA-256
`6106732e85413b2110fd269fcd5f1d691bcf443361da2d7c1b039cfb0476c187`.

| clean-trace accounting | ms/frame |
|---|---:|
| unweighted VS+fetch summed duration | 1.181784 |
| unweighted pool union | 0.821026 |
| unweighted pool union-exclusive | 0.725432 |
| duplicate-fraction weighted summed duration | 0.582517 |
| weighted union-exclusive, maximum active fraction | 0.405892 |
| weighted union-exclusive, active-fraction average | 0.384042 |
| **generous weighted union-exclusive upper** | **0.437503** |

The decision arm is deliberately more generous than max or average weighting:
on every pool-exclusive half-open span it credits
`min(1, sum(active duplicate fractions))`. It still reaches only
**0.437502653 ms/frame**, below the 0.500-ms/frame admission threshold. This is
before semantic exclusions, remap/hash construction, reference rewriting,
extra storage, or cache/version machinery. Vertex reuse therefore does not
justify a production implementation on favorite3. Favorite2 was not spent
after the primary capture failed the build gate.

Reproduce the join with:

```
python3 src/cudavk/tests/cp_vertex_reuse_cost.py \
  /tmp/vertex-reuse-f3/run1/stderr /tmp/vertex-reuse-f3/run2/stderr \
  > /tmp/vertex-reuse-cost.json
python3 src/cudavk/tests/cp_vertex_reuse_cost_test.py -v
```

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
