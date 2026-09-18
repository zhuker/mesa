# Static pass-stream assignment experiment

Default off: `CUDAVK_BALANCE_PASS_STREAMS=1`. `=0` keeps modulo assignment.
`CUDAVK_BALANCE_TRIANGLE_WEIGHT` defaults to 1, range 0..16.

## Scope and estimate

Only whole-segment append geometry changes placement. Score is submitted
vertices/indices plus weighted pre-clip triangles, multiplied by instances,
summed over batch draws. It does not predict clipping, coverage, shader cost,
GPU occupancy or completion. Estimates and accumulated loads saturate at
UINT64_MAX; every segment costs at least one unit. No GPU query, readback,
new event, stream, queue set, kernel or synchronization is added.

The first eight **side** assignments remain 0..7. Later assignments select the
least accumulated estimated load; ties start from the original modulo slot.
Opaque segment 0 stays on main with the context queue set and contributes no
side load. Blended segment 0 starts at side slot 0. Primitive allocation,
segment recording, host issue order and fallback replay order do not change.

## Ownership and lifecycle audit

- Stream k still owns seg_qsets[k]. Selection changes both together.
- Filling the prefix first preserves cp_pass_join/cp_pass_broadcast semantics.
- Opaque gate-once uses its existing selected-slot bit. The upload split,
  entry wait, carry switch and per-segment-gate revert are unchanged.
- Blended entry gate and upload stream switch are unchanged.
- Each blended segment saves its actual geometry stream with its saved queue
  arguments. The no-record-buffer fill relaunch uses this stream, not modulo.
  Its count join, scan and broadcast remain intact. Reassigning that replay
  independently would race queue scratch and is deliberately forbidden.
- Record-buffer fill remains one main-stream replay. Shader-group fan-out and
  experimental run-geometry placement remain independent and unchanged.
- Loads reset at both episode starts and at both geometry-finish boundaries,
  before any fill/shade/fallback phase. A failed first append that leaves no
  episode is harmless: the next start resets again. Fatal errors prohibit
  further work. Failed appends retain the existing all-side-stream join.
- Fallback still joins before classic replay in Vulkan order. It does not
  infer a segment's owner from modulo and needs no new replay machinery.

## Admission and validation

The offline report `/home/alexzhukov/cudavk-3ms-evidence/balance/REPORT.md`
prices a small hypothesis, not a path to the whole-frame 3 ms objective.
Its 0.089/0.171 ms equal-arrival imbalance estimates and 0.192/0.325 ms
release-aware optimistic tail estimates describe different models, are not
measured savings, and are not current paired-capture coverage. No speedup is
claimed here.

The host-only `cp_pass_schedule_test` checks initial-prefix use, wrapped
least-load choices, deterministic ties, reset and saturation. Build and
registry audits do not prove device correctness. Before any acceptance, run
both captures and the correctness gates with balancing on/off, including
forced episode fallback, record-buffer-disabled fill and opaque gate reverts.
GPU validation and unprofiled alternating timing require separate approval.

Implementation validation: full standalone ICD build passed; the named host
scheduler Meson test passed; cp_debug_doc.py --check, cp_no_getenv.py and
`git diff --check` passed. No GPU program was run.
