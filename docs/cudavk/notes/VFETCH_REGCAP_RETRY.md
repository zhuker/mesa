# Fused vertex-fetch register-cap retry (default off)

Baseline: `69a58a486d4bf74ba8d77089894d17bed1ce598c`.
Worktree: `/home/alexzhukov/mesa-three-ms-vfetch-retry`.
Flag: `CUDAVK_VFETCH_REGCAP_RETRY=1` (value boolean, default 0).

## Scope and mechanism

The B200 diagnosis in `cudavk-3ms-evidence/b200-diagnosis/report.md`
records classic VS 32 registers/8 blocks and fused VS 35 registers/6 blocks,
both with zero local/spill. The existing guard correctly declines. Its global
cap-32 probe is not a fix because it also changes fragment shaders.

This candidate makes one extra JIT attempt only after that occupancy refusal.
The cap is the smaller of the classic register count and
`registers_per_SM / 256 / classic_blocks_per_SM`. Checked CUDA resource queries
must reproduce the classic resource budget first. This conservative cap is
not an occupancy prediction: final CUDA occupancy must meet the original
classic comparison. The candidate must meet the cap and all original local,
spill, occupancy and forced-decline guards. It reuses the exact PTX whose
helper-survival checks already passed; there is no new lowering/codegen path.

`CUDAVK_MAX_REGISTERS`, `CUDAVK_NO_REGCAP`, or an already capped execution
suppress this retry. The generic fragment autotuner and global cap are unchanged.
This is not another general-cap sweep (DEAD_ENDS 44), and it gives no fragment
spill/occupancy trade permission. Static fragment tuning remains unchanged too.

## Lifetime and cache

Only the private fused VS execution is loaded again during pipeline creation.
The refused module remains owned until the replacement passes every check.
A failed load, lookup, query, cap, or admission check unloads only the temporary
replacement. Success unloads the private refused module and replaces its
module/kernel/resources, then invalidates its cached module-global addresses.
The normal attachment transfers it into the original shader binary once.
Classic and fragment modules, linked PTX inputs, PTX ownership, stage-cache
keys, global JIT policy, and fallback selection are unchanged. The registry
value is immutable during the device lifetime; no cache can mix changing retry
settings. No driver getenv was added. No lifetime blocker was found.

## CPU validation performed

- Fresh isolated Meson setup and full `ninja -C build-cudavk`: pass.
- `meson test -C build-cudavk cp_vfetch_regcap_retry_host`: 1/1 pass.
- `cp_debug_doc.py --check`: pass.
- `cp_no_getenv.py`: pass.
- `git diff --check`: pass.

The host test compiles the actual production helpers with mock CUDA calls.
It covers the B200-shaped success, disabled/override cases, original local,
spill/occupancy/forced refusal, JIT/query failures, ignored cap, and ownership
rollback. It checks the sole call site's fused-only wiring. Mocks prove host
control flow, not GPU compilation or pixels. An initial combined Meson test
command named unregistered `cp_debug_doc`/`cp_no_getenv` tests and failed test
selection; both audits were then run directly as above.

## Pending live validation

No GPU commands, remote edits, or capture runs were performed.
The standalone live presence test is deliberately not in the unconditional
suite: `tests/cpvk_vfetch_regcap_retry_gate.py <cpvk_batchbig>` requires a
real device/JIT/fixture with the uncapped occupancy refusal. It fails, not
skips, when no retry is needed. Under the normal exclusive GPU lock, with the
candidate ICD selected, it checks default and explicit-off refusal, explicit-on
admission, actual fused and grouped launches, identical fixture images, and
unchanged initial VS/FS resource diagnostic lines. It clears inherited
CUDAVK/CPVK switches so a global cap cannot fake the result. It does not enable
DEBUG_VFETCH, which would prevent grouping.

After that, run the unchanged vfetch and run-geometry gates, native suite,
sample gates, and both real capture correctness gates with explicit arm
provenance. The baseline B200 required-path failure remains a baseline failure;
an opt-in candidate result does not retroactively accept it. A synthetic shader
needing this retry does not establish useful coverage on either real capture.
No performance claim or default-on recommendation is made.
