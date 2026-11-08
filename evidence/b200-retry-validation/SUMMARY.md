# B200 retry validation: completed

Gate repair commit: d070f12695f. Only Python gate transmitted; no rebuild.
Binary before/after SHA256:
b48f4cc9dc03411bfba3a51697b7fcac11435ac68734327ce7b23359e3ec9a6b

- CPU parser regressions: 6/6.
- Live gate: original default/off lost-block refusal; retry cap32, actual32
  registers, spill0, eight blocks/SM, admitted installed fused resources.
- Actual successful grouped launch: 9 members, 9 draws, 162000 vertices, one
  VS_FETCH launch. Default/off/on PPM SHA256 all
  cab20de2d74b1bb33e7a060c3c598bc75fb0a64b3fe76e7dae929b36a10fd557.
- Unchanged native suite, retry ON, --no-rebuild: 84/84 pass.
- Both captures: three alternating OFF/ON rounds. All 12 runs rc0, exact
  expected stdout hashes, 6947/6965 rows, exactly18 sentinels with exact names
  and hashes matching the established control directories. exact-audit.json
  independently verifies the full sets, not only their intersections.
- Exclusive GPU lock and continuous process-tree monitor; no unrelated process
  detected. Runner completed and released lock.

Whole-window medians (ms/frame):

| capture | off runs | on runs | median delta |
|---|---|---|---|
| favorite3 | 7.8948, 7.9038, 7.8065 | 7.7937, 7.8820, 7.5309 | -0.1011 |
| favorite2 | 7.4625, 7.5013, 7.4794 | 7.5696, 7.3790, 7.3816 | -0.0978 |

Both arms overlap. No performance win is accepted. Heavy-band median deltas
are -0.1421 ms (favorite3) and +0.1291 ms (favorite2), also overlapping.
Baseline OFF still has the known original occupancy-refusal grouping gate
failure. This is not a claim that the baseline full suite passes.

Raw evidence and images remain on B200 at /home/builder/b200-retry-validation.
Local evidence includes logs, timestamps, runner, binary hashes, and exact
sentinel hash audit. Sentinel equality is change detection, not a new external
renderer acceptance result.
