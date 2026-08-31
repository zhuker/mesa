# Geometry recurrence B0 and B1R diagnostics

`CUDAVK_GEOMETRY_CENSUS=1` or `=2` enables the B0 post-stage2
visibility-buffer census. The values select separate, fixed SHA-256 Merkle
domains. Mode `3` enables the B1R increment: it uses the same census hook,
admits otherwise-safe retry units, and hashes each admitted output in both fixed
domains back-to-back. Mode 3 is one simultaneous observation, not two replay
runs. All three modes interpose hash kernels between raster stages 2 and 3, so
all timing from an enabled process is invalid. Mode `0` is the normal path and
does not compile or load the diagnostic NVRTC modules.

## Admission and safety

The census hook runs after successful stage 2 and before stage 3. B0 rejects a
retry at that hook. B1R records it and marks the record `retry=1`; non-retry
records are marked `retry=0`. This is a measurement of the narrow retry
retained-output hypothesis. It does not implement reuse.

Each admitted hash covers exactly the initialized
`width * height * samples * 8` byte visibility-buffer range. Pass 0 is preceded
by a full D32 all-ones clear of that range. Every later pass, including every
retry pass, repeats the full clear. Stage 1 and successful stage 2 then write
the output. The clear, stage 1, stage 2 and hash are ordered on the same CUDA
stream. The retry path's prior `reject` and `resolved` state is also ordered on
that stream, and no other writer can race the measured range. In mode 3 the
domain-1 hash completes before the domain-2 hash starts, and no source writer
is inserted between them. Therefore equality in the proposed output cache
necessarily implies visibility-buffer byte equality (under the stated SHA-256
collision-resistance assumption). The pair gives two
domain-separated roots for the same bytes; it is not a prior-buffer versus
current-buffer comparison.

Shared side-stream/appending units, fallback and path-predicate units, A-buffer
units, memory-writing shaders, indirect submits, non-timestamp semantic
query-bearing submits, unsupported shapes and failed launches remain excluded.
B0 also excludes retries. The summary charges every exclusion at 100%
recurrence. The clean-trace cost join also leaves optional clip work fully
charged; a post-stage2 equality is not evidence for a realizable pre-clip key.

The B0 analyzer's generous match is a non-consuming membership upper bound
keyed by `(w,h,samples,SHA)`. It asks whether an output appears anywhere in the
immediately previous frame. It also reports the stricter chronological-position
match and a digest of the admitted structural population. Both keys exist only
after the geometry work has produced the visibility buffer. The membership
number is therefore a theoretical retained-output upper bound, not a pre-work
key an implementation can realize. B1R narrows the decision to retry work; it
still does not manufacture such a key.

## Output and analysis

The fixed B0 record allocation is 131072 records of 128 bytes (exactly 16 MiB).
Mode 3 additionally reserves one 32-byte second root per record (exactly 4 MiB).
It computes the two domains consecutively against the same source range. Cleanup
synchronizes the CUDA context, prefetches managed output to the CPU,
synchronizes again, and emits the existing machine-readable prefixes:

- `CUDAVK_GEOM_B0_RECORD` has the structural fields, `retry=0|1`, the first
  root as `sha256=`, and, in mode 3, the second root as `sha256_2=`;
- `CUDAVK_GEOM_B0` is the summary. It reports `mode`, `salt`, `salt2`,
  `dual_domain`, `second_roots_bytes`, admissions, exclusions and validity.
  Modes 1 and 2 report `dual_domain=no` and `second_roots_bytes=0`; mode 3
  reports `dual_domain=yes` and the second-domain allocation.

The B0 names and prefixes are retained so existing collection tooling can
recognize the increment. Any allocation, capacity, copy, launch, queue-bound or
asynchronous stream error makes `result_valid=no`.

For one mode-3 log, the analyzer computes membership and chronological
classification independently from each root and requires exact equality at
every absolute unit. It reports the paired domains as checked and equal. For
repeated mode-3 logs it instead requires the complete structural population to
match exactly. It does not require replay classifications to match. It reports
the absolute flip set and count and exposes the conservative union-recurrent
classification for the cost join: a unit recurrent in either replay remains
recurrent. Thus simultaneous domains test one output for domain dependence;
repeated runs measure the separate raw-scheduling variation seen above.

Use `CUDAVK_GEOMETRY_CENSUS_SUBMIT_START=N` to hash only command-bearing
internal submits after absolute ordinal N. The ordinal is not claimed to be an
external submit ID. A traditional two-run B0 analysis is:

```
CUDAVK_GEOMETRY_CENSUS_SUBMIT_START=2783 <replay>
python3 src/cudavk/tests/cp_geometry_b0_analyze.py run-salt1.log run-salt2.log \
  --submits-per-frame 2 --frame-origin 2784
```

A count rate is not a cost rate. The analyzer does not emit a project verdict;
an external clean-trace interval join must price the exact matched unit set.

## Current B0 result and B1R decision

The clean 411-frame calibration is:

- full direct optional-clip/predecessor/stage2 exclusive pool:
  0.7368784501 ms/f;
- admitted single-stream pool: 0.5522067056 ms/f;
- excluded remainder, charged 100% recurrent: 0.1846717445 ms/f;
- B0 kill threshold: 0.3153282555 ms/f, or 57.10330068% of the admitted pool.

Both standard-valid B0 runs admitted 63835 of 108086 candidates. Each excluded
36963 appending units and 7288 retry units. Their membership rates were
60.2621% and 60.2699%. The project's clean-trace union-exclusive collectible
figures were 0.56498849 and 0.56502384 ms/f, so B0 survived its kill threshold.
The corresponding summed target durations were 1.61321338 and 1.61324873
ms/f. Those deliberately looser sums remain visible in the cost report: unlike
the project's established union-exclusive metric, they do not remove overlap
and therefore also cover arbitrary resource-contention effects.

The classification was not replay-stable: the two fixed-salt runs had 201
absolute equality flips, while a same-salt repeat still had 186 flips. The
same-salt result shows that replay variation, not merely the selected hash
domain, changes the observed classification. Mode 3 makes the two domain
observations back-to-back on one output and reports both roots; it does not
erase replay variation between outputs.

B1R kills the retained-output direction under the project's established
union-exclusive accounting if retry nonrecurrence removes more than
approximately 0.065 ms/f from the current collectible figure. Appending work
remains excluded and conservatively charged. The result is not an unconstrained
upper bound on secondary resource-contention effects, and B1R must not claim its
diagnostic timing as a performance result.

The two completed B1R mode-3 runs each recorded 71123 of 108086 candidates;
the only remaining exclusion was 36963 appending units. Both kept 6939 replay
timestamps, the standard stdout hash and all 18 sentinels, and every paired
same-snapshot domain classification agreed. The repeated structural population
was exact; 194 absolute classifications flipped between executions. The
conservative OR of the two runs maps 13937 clean-trace units as 3880 recurrent
admitted, 3981 nonrecurrent admitted, 6053 excluded appending and 23 unknown.
With every clip and every appending/unknown unit still charged recurrent, the
decision-bearing union-exclusive collectible figure is **0.5023787810 ms/f**
(the individual diagnostics are 0.5023435888 and 0.5023316764). B1R therefore
survives by only 0.0023788 ms/f; it does not close the direction. A follow-up
must directly census the appending population rather than infer it from stream
or count rates.

For one SHA-256 domain and `N` compared values, the union/birthday collision
bound is at most `N(N-1)/2^257`. Mode 3 requires equality in both fixed domains
for its paired classification. This is collision evidence under the SHA-256
assumption, not a guarantee that separate replay runs produce the same output.


## B1S mode 4: isolated ordinary opaque segments

Mode `4` keeps mode 3's simultaneous dual-domain behavior for every existing
nonappending record, including retries, and adds direct evidence for ordinary
opaque appending segments. Each record now prints `isolated=0|1`. Nonappending
records retain the post-stage2 shared-unit projection and use `isolated=0`.
Appending records use the isolated post-reraster-stage3 projection below and use
`isolated=1`. The analyzer includes this projection kind in its ordinary
membership key, `(isolated,w,h,samples,SHA)`. Thus an identical byte digest can
never make a shared post-stage2 unit match an isolated post-stage3 contribution.
It also includes `isolated` in mode-4 repeated structural population identity.

A final visibility-buffer mismatch for a whole opaque episode cannot mark every
segment nonrecurrent. Segments append atomic minima into one shared buffer, and
partial segment reuse is in scope: one changed segment can change the episode's
final buffer while other segments made the same contribution as in the prior
frame. B1S therefore obtains evidence one segment at a time rather than
propagating one episode verdict.

The population proof is structural. The B0 hook is in the plain raster loop.
Blended/A-buffer appends record and return before it, as does the tiled opaque
path. Consequently any `cp->pass.appending` candidate reaching the hook is a
non-tiled ordinary `cp_opaque_append` segment. Mode 4 still checks opaque,
pass 0, no A-buffer/retry/peel/path predicate, successful predecessor/stage2 and
clear chain, and all existing indirect/query/shader-memory/shape safety gates.
A fused clip+stage1 predecessor is allowed after successful completion: its
finished clipped positions, primitive references and active-ID map are exactly
the geometry saved for the ordinary reraster.

After the original stage 3 successfully consumes that segment's queue set, and
before `cp_pass_record_segment`, mode 4 reissues only ordinary raster stages 1,
2 and 3 on the same current CUDA stream. It does not refetch vertices or reclip.
The copied raster arguments retain the completed positions, `prim_refs`, active
IDs, clip rectangles, scalar raster/depth state, stable episode depth input and
episode-global `abuf_prim_base`. Census, peel, reject, A-buffer and path fields
are neutralized. The copied queues use `CP_QUEUE_FILL`; their three counters are
cleared only after original stage 3 consumed them. Ordinary
`cp_opaque_finish` does not consume the saved queue contents. The tiled opaque
path can use the saved queue addresses, but it clears and rebuilds them first;
that path returns before the B1S hook in any case. The later content reuse is
the excluded blended A-buffer fill path. Thus this overwrite cannot alter live
ordinary opaque state.

The reraster targets an all-ones private visibility buffer and is followed by
both Merkle domains only after reraster stage 3. Its digest is the initialized
per-segment atomicMin contribution from small, medium and huge primitives
against the stable episode depth input. The depth buffer is read-only during
rasterization and opaque episode shading does not run until the segment streams
are joined at finish. Completed clip allocations likewise remain owned by the
recorded segment until finish. These lifetime and ordering facts make the
reraster reproducible without snapshotting the shared visibility buffer.

There are nine device-private banks: bank 0 for the main stream/context queue
set and banks 1 through 8 for the exact eight side streams/queue sets. Each bank
has a disjoint visibility range, Merkle scratch range and saved-root range.
Segments beyond eight reuse the corresponding bank and queue set on the same
stream, so stream order makes the reuse safe. Managed record/root output slots
are reserved sequentially on the host and are disjoint. High-water replacement
occurs only at framebuffer change after the prior pass is finished and joined;
all allocation/free/clear/launch results are checked. The summary reports
`private_banks`, `private_visbuf_bytes`, `private_scratch_bytes` and
`private_saved_roots_bytes`. Any pending, allocation, reraster or hash failure
adds one exclusion and makes `result_valid=no`; there is no per-unit host sync.

This remains diagnostic-only. The digest exists only after reraster and is not
an input cache key. Its enabled timing is invalid. Since refetch/reclip is not
reissued, the cost join still charges every clip interval recurrent. A payload
match is only a conservative retained-output upper: full raster-payload
equality implies equality of this initialized digest projection, while a digest
match can remain a false match for work outside the projection.

The clean B1R conservative upper, **0.5023787810 ms/f**, remains the baseline,
not a mode-4 result. Mode 4 is sized for all 108086 candidates and is expected
to admit the former 36963 safe appending candidates in addition to B1R's 71123
records. The existing absolute `(submit,unit)` cost join therefore directly
classifies the mapped 6053 appending units. It does not propagate an episode
classification; uncovered units and all clip work remain recurrent. Repeated
mode-4 runs require the exact salt pair, paired-domain agreement and identical
structural population, report absolute flips, and use the conservative OR union
as the decision-bearing arm.
