# B200 register retry presence gate repair

The original gate required a `vsfetch shader` census row. This is teardown
telemetry (`cp_vs_census_report`, guarded by SHADER_STATS), not a launch row.
`cpvk_batchbig` returns without destroying its device, so no census is printed.
DEBUG_VFETCH is neither needed nor enabled: it disables grouping.

The repaired gate requires default and explicit off to refuse `lost-a-block`.
On must report cap 32, actual 32 registers, zero spill, eight blocks/SM, and
installed fused resources admitted. It requires exactly one successful
`run-geometry-launch: members=9 draws=9 verts=162000` with a non-null kernel.
`cp_run_geometry_fused` requires the VS_FETCH kernel. The launch row is emitted
only after `cp_launch(group->kernel)` succeeds, so this proves actual fused
execution, not merely planning. Pixels and original shader resource rows must
remain exact across all three arms. No global register cap or guard override.

CPU parser tests use original B200 raw logs. They reject original refusal,
plan-only logs, a census without a launch, wrong resources/geometry, and missing
initial resource rows. Run `python3 src/cudavk/tests/cpvk_vfetch_regcap_retry_gate_test.py`.

Validation is recorded under `evidence/b200-retry-validation` and remotely under
`/home/builder/b200-retry-validation`. Only the gate script is transmitted;
no driver rebuild. Original binary SHA256:
`b48f4cc9dc03411bfba3a51697b7fcac11435ac68734327ce7b23359e3ec9a6b`.
Baseline off is known to fail the required grouping gate on original occupancy
refusal; this is not a baseline all-suite-pass claim.
