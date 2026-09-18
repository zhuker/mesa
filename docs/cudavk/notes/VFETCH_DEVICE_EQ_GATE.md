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
Supply authoritative submit/frame counts, anchored log count/EOF expressions,
recorded argv, exits, stdout and stderr. Do not use image counts as submits.
Two captures require six arms: full positive, full BREAK, and feature-disabled
control each. The runner returns nonzero on failure.

CPU checks: `meson test -C build-cudavk --suite cudavk-cpu`, registry/doc and
launch audits, and the runner self-test. No GPU tests ran during implementation.
The detailed handoff, CPU logs, offline NVRTC compiler script and exact
flag-off folding/live-feature PTX seam evidence are under
`/home/alexzhukov/cudavk-3ms-evidence/vfetch-device-equivalence-implementation.md`
and its adjacent `vfetch-check-*` artifacts. This finite seam evidence is not
full generated-shader SASS or replay coverage.
