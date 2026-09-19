# Validation-only vfetch specialization gate

This branch implements a device-input equivalence gate for static vfetch.
It is not performance instrumentation and does not establish correctness until
full positive and live-negative-control replays complete on both captures.

Enable `CUDAVK_VFETCH_EQ_CHECK=1` with `CUDAVK_SPECIALIZE_VFETCH=1`.
Set `CUDAVK_VFETCH_EQ_BREAK=1` for the detector control. It changes a bit in
an actual specialized gathered slot, compares it, and restores it before VS
consumption. It never executes shader side effects a second time.

The independent oracle's original source revision is recorded in
`src/cudavk/kernels/cp_vfetch_eq_oracle.h`. It precedes both real VS dispatch
paths on the selected stream. Private two-slot row buffers have completion
events; no draw scratch allocation is reused. CHECK-off generated shaders
have no comparison calls. Ordinary uninstrumented compilation determines
production fusion admission before instrumented compilation, so debug spills
cannot silently make specialized coverage disappear.

The device's embedded renderer emits one `VFETCH_EQ` JSON record after its
all-stream drain and cleanup. Host/device expected/checked invocation, slot
and word counts must agree exactly. Generic, standalone and layout-mismatch
launches are separately counted, never credited as specialized comparisons.
Require actual lane and grouped launches where the replay exercises them.

`src/cudavk/tests/cpvk_vfetch_eq_gate.py` validates complete replay logs.
Run its `self-test`, then use `validate --help` or `suite --help`.
Plugin mode still requires anchored `--submit-regex`, `--frame-regex` and
`--eof-regex`. Native tocpp mode instead requires **both** `--submit-ts` and
`--expected-stdout-sha256`; these are mutually exclusive with all regex options.
It checks the raw stdout hash, exit zero, and strict `ns index` timestamp rows:
monotonic nondecreasing integer nanoseconds and contiguous zero-based indices.
Equal timestamps are allowed because `CLOCK_MONOTONIC` can tie.
Submit count is the row count; complete real frames are `(rows - 1) // 2`.
No textual EOF is required in native mode. Missing/truncated timestamps fail.
Device reports, teardown checks, identity, coverage and BREAK checks are unchanged.
Do not use image counts as submits.

## Native replay commands (not run during this CPU-only change)

Authority: `/home/alexzhukov/cudavk-3ms-evidence/replay-manifest.json` and
`/home/alexzhukov/favorite-cpp/submit_shim.c`. The executable takes **no arguments**
and uses **no replay plugin**. For each capture and arm choose a fresh OUT and
set ICD to the candidate ICD's absolute path. Positive arm example:

```sh
OUT=$(mktemp -d /tmp/vfetch-eq-favorite3-XXXXXX)
cd /home/alexzhukov/favorite3-cpp/out
LD_PRELOAD=/home/alexzhukov/favorite-cpp/submit_shim.so \
SUBMIT_TS_FILE="$OUT/ts.txt" DUMP_DIR="$OUT" DUMP_EVERY=200 DUMP_MAX=20 \
VK_DRIVER_FILES="$ICD" CUDAVK_VFETCH_EQ_CHECK=1 CUDAVK_SPECIALIZE_VFETCH=1 \
./build/vulkan_app >"$OUT/stdout" 2>"$OUT/stderr"
RC=$?
python3 /home/alexzhukov/mesa-three-ms-vfetch-check/src/cudavk/tests/cpvk_vfetch_eq_gate.py validate \
  --stdout "$OUT/stdout" --stderr "$OUT/stderr" --exit-code "$RC" \
  --submit-ts "$OUT/ts.txt" --expected-submits 6947 --expected-frames 3473 \
  --expected-stdout-sha256 e4a60a7156141649a4c16660f674025af7b9fc3fe2702f4f57845527075d058d \
  --arm positive --command-json '["./build/vulkan_app"]' --output "$OUT/gate.json"
```

For favorite2 use `cd /home/alexzhukov/favorite2-cpp/out`, a fresh OUT,
`--expected-submits 6965 --expected-frames 3482`, and
`--expected-stdout-sha256 0240ff4ec576c62b49461a384d90e0c25463f69ecdaf1b2286e92d14264d947e`.
The native command remains exactly `./build/vulkan_app`.
For negative arms add `CUDAVK_VFETCH_EQ_BREAK=1` and validate `--arm negative`.
For disabled arms keep CHECK on, set `CUDAVK_SPECIALIZE_VFETCH=0`, unset BREAK,
and validate `--arm disabled`. Positive arms must also have BREAK unset.
Keep the same hash/count expectations across arms. Supply the required
`--devices`, `--min-lane-launches` and `--min-group-launches` coverage expectations.
Suite JSON uses the same underscore field names (`submit_ts`,
`expected_stdout_sha256`), with separate timestamp paths for each arm.
The manifest's relevant ranges are favorite3 `[1391,3473)` (2082 frames)
and favorite2 `[1388,3482)` (2094 frames); never substitute these for full counts.
Dump and CHECK durations are not performance evidence.
Two captures require six arms: full positive, full BREAK, and feature-disabled
control each. The runner returns nonzero on failure.

CPU checks: `meson test -C build-cudavk --suite cudavk-cpu`, registry/doc and
launch audits, and the runner self-test. No GPU tests ran during implementation.
The detailed handoff, CPU logs, offline NVRTC compiler script and exact
flag-off folding/live-feature PTX seam evidence are under
`/home/alexzhukov/cudavk-3ms-evidence/vfetch-device-equivalence-implementation.md`
and its adjacent `vfetch-check-*` artifacts. This finite seam evidence is not
full generated-shader SASS or replay coverage.
