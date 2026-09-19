# Clipping equivalence gate (validation-only)

Base common-reject `ded250960ee62c`, plus copy bypass `cad04fff89d`
cherry-picked as `ab896b3da77`. This tree is not a timing build.

## Contract

`CUDAVK_CLIP_EQ_CHECK=1` compiles the independent original plane loop into
all `cp_clip_one` paths. `CUDAVK_CLIP_EQ_BREAK=1` corrupts a real nonempty
copy-bypass output word; `=2` changes a real shortcut rejection to canonical
three vertices. Both restore the normal result after the ordinary comparator.
These are detector controls, not broken-image demonstrations. Invalid break
configurations fail device creation. All switches default off and use registry.

The oracle is the original all-inside check and parity-selected plane loop
from `69a58a486d4bf74ba8d77089894d17bed1ce598c`. It shares only unchanged
clip_dist/clip_poly/clip_lerp/copy helpers, never either optimization predicate.
Local oracle arrays live throughout the comparison. Every terminal compares
canonical n (n<3 becomes zero), then every live ordered float4 component as
uint32 bits if n agrees. A count mismatch fails without reading mismatched
array bounds. No reference allocation, publication, counter increment, ID
assignment, or retirement occurs. Actual fan/publication/retirement execute
once, unchanged. Thus allocation ordering is outside the comparison boundary.

One device-owned report survives all streams and contexts. Initialization uses
pinned synchronous HtoD, not a new sync. Final check-mode context drain after
renderer cleanup precedes one report readback. No per-draw readback/sync.
Host independently counts admitted input triangles. Device records exactly one
terminal/path per invocation, every branch, all storage modes, actual shortcut
hits, bypass planes, interpolation after bypass, injections, mismatches and
first (dispatch,triangle) evidence. Allocation overflow is a hard gate error.
Grouping fallback raster consumes clipped slots without reclipping; fused
path predicates currently have no nonzero host caller. Future caller changes
must revisit expected-count accounting.

## CPU validation (no GPU operations)

```
meson setup build-cudavk -Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled
ninja -C build-cudavk
python3 src/cudavk/tests/cp_clip_eq.py
python3 src/cudavk/tests/cp_clip_copy_bypass.py
python3 src/cudavk/tests/cp_clip_common_reject.py
python3 src/cudavk/tests/cp_debug_doc.py --check
python3 src/cudavk/tests/cp_no_getenv.py
```

The instrumented source CPU test compiles eight configurations: off, copy,
reject, both, plus each enabled fault. It checks full output/storage/ref/ID
side effects against the uninstrumented original and exercises 1/4/16 slots,
all 16 storage modes, seven capacities, exceptional floats and 3000 random
triangles. Fault counters and real mismatches must be nonzero only in fault
arms; outputs remain original after restoration. CPU arithmetic is not CUDA
arithmetic and this is not the replay gate.

NVCC 12.8 compile-only PTX for compute_120 passes off/live/copy-fault/reject-
fault arms. Evidence is under `/home/alexzhukov/cudavk-3ms-evidence/clip-equivalence`.
PTX sizes are 764113 / 1290102 / 1292286 / 1305339 bytes respectively.
Copy fault PTX has conditional load, xor 1, store on the live candidate pointer,
then injection atomic and ordinary compare. Check-off has no report atomics.
An uninstrumented source with the *same extended ABI* differs from check-off
only in four independent mov instruction reorderings after normalizing compiler
file-name symbols. This is folding evidence, NOT zero ABI collateral: appended
cp_clip_args changes kernel parameter/segment sizes, and appended cp_device
changes offsets after the embedded owner in cpvk_device. No timing claim.

## Full replay gate still required

For BOTH authoritative captures run CHECK=1 with off/copy/reject/both feature
states. For each retained enabled feature run a separate whole-replay negative
control, also in the retained combination. Preserve argv/environment, process
exit status, stdout/stderr, exact library/capture hashes and authoritative
submit/frame counts. Plugin replays also require normal EOF. Native tocpp
replays instead require the exact stdout hash and shim census below.
Never infer submits from image/readback counts.
Use the accompanying `cpvk_clip_eq_gate.py` offline validator. Require explicit
path/mode minima for paths expected by the replay coverage plan; zero feature
hits cannot support a live-feature verdict. No missing report or incomplete
replay can pass. NVRTC smoke, PTX/SASS review on the actual target, and all-frame
external image comparisons from TESTING.md remain pending. This gate proves
only canonical clip equivalence on exercised inputs, not all Vulkan behavior
or full renderer equivalence. CPU side-effect tests do not replace GPU storage
mode/path coverage. No GPU command was run for this implementation.

### Recorded-log verdict command

Run the replay with CHECK=1 and save raw stdout/stderr and its real exit status.
For reject-only positive, set COPY_BYPASS=0, COMMON_REJECT=1, BREAK=0;
repeat BREAK=2 for negative-reject. Copy-only uses 1/0/0 then 1/0/1.
Both uses 1/1/0 then separate 1/1/1 and 1/1/2. Disabled uses 0/0/0.
The full flag names are CUDAVK_CLIP_COPY_BYPASS and CUDAVK_CLIP_COMMON_REJECT.

```
python3 src/cudavk/tests/cpvk_clip_eq_gate.py self-test
python3 src/cudavk/tests/cpvk_clip_eq_gate.py validate \
  --arm positive --expected-copy off --expected-reject on \
  --stdout "$RUN.stdout" --stderr "$RUN.stderr" --exit-code "$REPLAY_EXIT" \
  --expected-submits "$AUTHORITATIVE_SUBMITS" \
  --expected-frames "$AUTHORITATIVE_TIMING_FRAMES" \
  --submit-regex "$SUBMIT_LINE_REGEX" --frame-regex "$TIMING_LINE_REGEX" \
  --eof-regex "$NORMAL_EOF_REGEX" --command-json "$REPLAY_ARGV_JSON" \
  --min-stable 1 --min-refs 1 --min-fused 1 \
  --output "$RUN.verdict.json"
```

Use `--arm negative-copy`, `negative-reject`, or `disabled` as appropriate.
Both features must have real hits if enabled. Regexes must be anchored complete
line formats; named `(?P<count>...)` accepts one summary count. Do not invent
counts or regexes: the current capture census and actual timing-plugin output
are authoritative. Add `--min-standalone`, `--min-group-bin`,
`--min-group-direct`, `--min-active-ids`, `--min-ids`, and
`--min-interpolate-after-bypass` according to the intended coverage matrix.
Default minima alone do not prove every optional mode ran. Positive/negative
logs must have matching capture identity and authoritative full census.
The validator checks one final device report by default; trailing stderr is
rejected rather than silently hiding a late failure. Offline validation cannot
authenticate caller-supplied argv, exit status, regex, or census: preserve raw
execution artifacts. No image-only fallback exists.


### Native tocpp captures (no replay arguments or plugin)

The authoritative census is
`/home/alexzhukov/cudavk-3ms-evidence/replay-manifest.json`.
`submit_shim.c` writes `CLOCK_MONOTONIC` nanoseconds followed by a zero-based
submit index, separated by one space, with one newline per row. The validator
requires nondecreasing nanoseconds, contiguous indices starting at zero and
exact row counts. Complete paired-submit frames are `(rows - 1) // 2`.
Both `--submit-ts` and `--expected-stdout-sha256` select tocpp mode; all three
plugin regex options are forbidden in this mode. Plugin mode remains unchanged.
Tocpp has no EOF line. Exact raw stdout SHA256, recorded exit zero and the full
census establish replay completion together, not any one alone. These checks
do not replace the device report, coverage minima or negative controls.

Example for the both-enabled positive arm (use a fresh absolute `$RUN` directory,
set `$ICD` to the existing validation ICD; do not rebuild during a replay):

```sh
mkdir "$RUN"
cd /home/alexzhukov/favorite3-cpp/out
LD_PRELOAD=/home/alexzhukov/favorite-cpp/submit_shim.so \
SUBMIT_TS_FILE="$RUN/ts.txt" DUMP_DIR="$RUN" DUMP_EVERY=200 DUMP_MAX=20 \
VK_DRIVER_FILES="$ICD" CUDAVK_CLIP_EQ_CHECK=1 CUDAVK_CLIP_EQ_BREAK=0 \
CUDAVK_CLIP_COPY_BYPASS=1 CUDAVK_CLIP_COMMON_REJECT=1 \
./build/vulkan_app >"$RUN/stdout" 2>"$RUN/stderr"
REPLAY_EXIT=$?
printf '%s\n' "$REPLAY_EXIT" >"$RUN/exit-code"
python3 /home/alexzhukov/mesa-three-ms-clip-check/src/cudavk/tests/cpvk_clip_eq_gate.py validate \
  --arm positive --expected-copy on --expected-reject on \
  --stdout "$RUN/stdout" --stderr "$RUN/stderr" --exit-code "$REPLAY_EXIT" \
  --submit-ts "$RUN/ts.txt" --expected-submits 6947 --expected-frames 3473 \
  --expected-stdout-sha256 e4a60a7156141649a4c16660f674025af7b9fc3fe2702f4f57845527075d058d \
  --command-json '["./build/vulkan_app"]' --output "$RUN/verdict.json"
```

For favorite2, use a new `$RUN`, change the `cd` to
`/home/alexzhukov/favorite2-cpp/out`, and use this exact verdict command after
the same native launch and exit capture:

```sh
python3 /home/alexzhukov/mesa-three-ms-clip-check/src/cudavk/tests/cpvk_clip_eq_gate.py validate \
  --arm positive --expected-copy on --expected-reject on \
  --stdout "$RUN/stdout" --stderr "$RUN/stderr" --exit-code "$REPLAY_EXIT" \
  --submit-ts "$RUN/ts.txt" --expected-submits 6965 --expected-frames 3482 \
  --expected-stdout-sha256 0240ff4ec576c62b49461a384d90e0c25463f69ecdaf1b2286e92d14264d947e \
  --command-json '["./build/vulkan_app"]' --output "$RUN/verdict.json"
```

Add the explicit mode/path coverage minima required by the coverage plan to
both commands. Repeat the off/copy/reject/both and negative-control matrix above
with fresh artifacts per arm. A passing command without optional mode minima
does not establish those modes' coverage. Preserve environment and library and
capture identities alongside every verdict. No timestamps, stdout lines or EOF
markers may be synthesized for a real verdict. Self-test census fixtures are
synthetic CPU tests only, not replay evidence.
