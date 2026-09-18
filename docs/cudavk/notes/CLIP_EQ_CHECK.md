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
exit status, stdout/stderr, exact library/capture hashes, normal EOF, authoritative
submit and timing-plugin counts. Never infer submits from image/readback counts.
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
