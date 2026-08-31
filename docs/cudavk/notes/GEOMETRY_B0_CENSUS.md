# Geometry recurrence B0 diagnostic

`CUDAVK_GEOMETRY_CENSUS=1` or `=2` enables the B0 visibility-buffer census.
The two values select fixed, independent SHA-256 Merkle domain salts and region
IDs. This mode interposes hash kernels between raster stages 2 and 3. Therefore
all timing from an enabled process is invalid. `CUDAVK_GEOMETRY_CENSUS=0` is the
normal path and does not compile or load the diagnostic NVRTC modules.

B0 hashes only the initialized `width * height * samples * 8` byte visibility
region. Pass 0 is preceded by the full D32 all-ones clear. Each later ordinary
peel pass repeats that full clear. Stage 1 and successful stage 2 precede the
hash on the same stream. Stage 3 follows it. Shared side-stream/appending units,
retry/fallback and path-predicate units, A-buffer units, memory-writing shaders,
indirect submits, non-timestamp semantic query-bearing submits, unsupported shapes and failed launches are
excluded. The summary charges every exclusion at 100% recurrence.

The fixed record allocation is 131072 records of 128 bytes (exactly 16 MiB).
Cleanup synchronizes the CUDA context, prefetches the managed allocation to the
CPU, synchronizes again, and emits `CUDAVK_GEOM_B0_RECORD` lines followed by a
machine-readable `CUDAVK_GEOM_B0` summary. Any allocation, capacity, launch,
queue-bound or asynchronous stream error makes `result_valid=no`.

Use `CUDAVK_GEOMETRY_CENSUS_SUBMIT_START=N` to hash only command-bearing
internal submits after absolute ordinal N. The ordinal is not claimed to be an
external submit ID.

Analyze one run, or validate two independent-salt reruns, with:

```
CUDAVK_GEOMETRY_CENSUS_SUBMIT_START=2783 <replay>
python3 src/cudavk/tests/cp_geometry_b0_analyze.py run-salt1.log run-salt2.log \
  --submits-per-frame 2 --frame-origin 2784
```

The clean 411-frame calibration is:

- full direct optional-clip/predecessor/stage2 exclusive pool: 0.7368784501 ms/f;
- admitted single-stream pool: 0.5522067056 ms/f;
- excluded remainder, charged 100% recurrent: 0.1846717445 ms/f;
- B0 kill threshold: 0.3153282555 ms/f, or 57.10330068% of the admitted pool.

The analyzer compares only the immediately previous frame. It reports a
generous non-consuming membership upper bound keyed by `(w,h,samples,SHA)`
and a stricter chronological-position match. It also emits a digest of the
absolute admitted structural population for joining to the calibrated trace.
A count rate is not a cost rate, so the analyzer never emits a project verdict.
The verdict remains unavailable until an external clean-trace interval join
prices the exact matched unit set.

Under the standard SHA-256 collision-resistance assumption, one false equality
is bounded by `2^-256`. For `N` compared values, the union/birthday bound is
at most `N(N-1)/2^257`. A mode-2 rerun must reproduce the absolute equality
classification under its distinct domain before the census is accepted.
