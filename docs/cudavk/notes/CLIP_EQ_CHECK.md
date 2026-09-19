# Strict lazy clipping equivalence check

This is a diagnostic device comparator, not a performance feature. It checks
actual candidate results against an independent original polygon walk.
It does not replace external-renderer image gates.

## Contract

- `CUDAVK_CLIP_EQ_CHECK=1` enables collection; check-off is the default.
- `CUDAVK_CLIP_LAZY=1` selects the production lazy evaluator.
- `CUDAVK_CLIP_EQ_BREAK=1` flips a bit in an actual live varying value,
  snapshots it, then restores it before output scatter. The negative arm
  must show canonical **and** scatter mismatches. Output and VS input must
  remain unchanged. The checker must not evaluate varyings a second time.
- Every actual clip invocation contributes to the terminal and dispatch-path
  partitions. Full success requires `expected == invocations == comparisons`
  with a nonzero expected count. `checked` aliases `comparisons`.
- A surviving polygon in the lazy clipping path (`n >= 3`) with `k == 0`
  has no actual varying replay.
  It increments `unsupported_k0`, not `comparisons`. This is **INCOMPLETE**,
  even for position-only input. It must not be classified as an empty polygon.
- Missing actual materialization increments `unsupported_missing`.
  Both unsupported counters must be zero for replay acceptance. Zero actual
  work, missing report storage, readback failure, and an absent teardown footer
  are not successful comparisons.
- All-inside acceptance exposes the actual canonical input directly, including
  pointer-alias publication. It does not need varying interpolation. Aliased
  input is never faulted. Rejected polygons compare canonical count zero.
- Partial fans compare the whole actual canonical polygon when replay exists,
  then check only allocated scatter output. No extra output allocation occurs.

The footer is `CUDAVK_CLIP_EQ_JSON:` followed by one JSON object with type
`clip_equivalence`. It contains 34 unsigned 64-bit counters. Counters 0–29
retain the reference layout, except 7 is reserved (`reject_shortcut`, zero),
10 is `replay_slots`, 11 is `replay_invocations`, and 12 is `partial_fan`.
Counters 30–33 are `unsupported_k0`, `unsupported_missing`, `scatter_words`,
and `scatter_mismatch_words`. The boolean feature field is `lazy`.
`readback_errors`, `zero_checked`, and `checked` are mandatory.

## CPU-only checks

```
python3 src/cudavk/tests/cp_clip_eq.py
python3 src/cudavk/tests/cp_clip_eq.py --sanitize
python3 src/cudavk/tests/cpvk_clip_eq_gate.py self-test
```

The source test extracts the actual kernel helper and ABI. Four arms check:
plain eager clipping with instrumentation; lazy clipping; live-value bit
injection; and a synthetic missing-materialization coverage fault. All arms
compare complete output storage, published reference identity, output counts,
active IDs, returned IDs, and unchanged input bits against check-off eager
clipping. The corpus includes 3,000 random triangles, exceptional positions
and varying bits, slot counts 1/4/16, all 16 optional-output modes, seven
capacities, and all four dispatch-path IDs.

The corpus deliberately includes insufficient capacity. A successful CPU
regression therefore does **not** mean its aggregate counters would pass the
strict runtime validator: the lazy arms intentionally produce unsupported k0
invocations. The test proves that these are explicitly counted and excluded
from comparisons, not hidden in a successful total. The missing-materialization
arm deliberately clears actual seed-evaluation evidence in the extracted
candidate. Dependency evidence then propagates to the gather coverage check.

The parser tests reject unsupported work, missing readback, old/short counter
arrays, invalid types, inconsistent aliases, missing invocation coverage,
missing negative evidence, duplicate/malformed/non-final footers, incomplete
replay census, and stdout hash changes.

## Existing replay validation

`cpvk_clip_eq_gate.py validate --help` lists the complete interface. Supply
saved stdout/stderr, recorded command JSON, exit status, authoritative expected
submit/frame counts, and either:

- `--submit-ts` plus `--expected-stdout-sha256` for tocpp, or
- anchored submit, frame, and EOF regexes for the plugin protocol.

Use `--arm positive`, `--arm negative`, or `--arm disabled`. Feature and path
coverage can be required with `--expected-lazy` and `--min-*` options. The
validator requires a zero exit status and a final footer per expected device.
One frame is two submit intervals. A changed stdout hash is not accepted just
because reported timing improved. No GPU work is performed by the validator.

Evidence for this implementation belongs only under
`/home/alexzhukov/cudavk-3ms-evidence/lazyclip/native-validation/strict-check/`.
CPU/parser results are not proof of completed native replay or GPU correctness.

## Generated-code evidence (CPU only)

`strict-check/CPU-READY.md` records source hashes, compile commands and results.
The exact production source was compiled with NVRTC for compute_120 and ptxas
for sm_120 in checker-off, plain-check, lazy-check, and actual-fault arms.
PTX and SASS show live node load, one-bit XOR, store, snapshot loads and restore
before scatter. Checker-off standalone has no 64-bit report atomics.
The extended ABI is not free; no zero-overhead or performance claim is made.
No GPU command or native admission run was performed for this implementation.
