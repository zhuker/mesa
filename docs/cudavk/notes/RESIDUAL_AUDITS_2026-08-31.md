# Favorite3 residual audits after the shader content census

Status: three read-only audits plus one new no-CUPTI host CPU profile.
Baseline is unchanged: favorite3 7.604 ms/frame, device union 4.510 ms/frame,
device idle ceiling 1.673 ms/frame. No production change is proposed here.
The 0.500-ms/frame union-exclusive admission gate governs all verdicts.

## GPU residual partition (`/tmp/residual-gpu-audit/AUDIT.md`)

The 4.510-ms device union partitions exactly by family. Only two fine
families reach the gate: direct raster/clip (1.134433 ms/frame) and fused
generated VS (0.520663 ms/frame). Every evidenced architecture-independent
mechanism against raster/clip is already closed in `DEAD_ENDS.md`, and the
fused-VS content pool is parked for B200 with no concentration (largest key
0.209405; minimum qualifying set 15 of 22 keys at 0.502497). Broader pools
above the gate bundle mandatory semantics or already-censused mechanisms.

## Host residual proof (`/tmp/residual-host-audit/REPORT.md`)

Within the complete CUPTI window, all housekeeping/resource APIs total
0.046266 ms/frame, ruling out event/pending/resource pooling. The
submit-entry to completion-wait envelope leaves only 0.578025 ms/frame of
non-CUDA callback work under a perturbed trace. Fully threaded submit, the
one honest host candidate, was already measured at 0.126184 ms/frame.

## New measurement: no-CUPTI DWARF perf profile

`/tmp/favorite3-perf-current` holds a full favorite3 replay under
`perf record -F 999 --clockid mono --call-graph dwarf` with the standard
stdout hash, 6,939 shim rows, and the analysis window restricted to real
frames (external submit 2782 on; 2,078 frames; 16,064 main-thread samples).
Main-thread on-CPU time partitions as:

| class | ms/frame |
|---|---:|
| CUDA wait spin (device-paced) | 4.436 |
| CUDA issue APIs | 1.498 |
| driver CPU planning (no CUDA API in stack) | 0.868 |
| application record/decode | 0.376 |
| unattributed | 0.371 |
| other CUDA APIs | 0.190 |

The wait spin is two known closed sites: `cp_smallop_ctxsync` 2.917 and
`cp_sync_timed` 1.508 ms/frame. Both are blocked-time symptoms; scratch
high-water reuse already proved 1.65 ms of that blocking converts to only
0.037 ms of frame time. Inside driver planning, the only subtree above the
gate is `cp_compile_nir_one` at 0.539 ms/frame **on the window average**, but
it is entirely cold-start: all compile samples land in 13 of 2,078 frames
(one frame absorbs about 0.67 s). It cannot move the hot-tail median.
Steady-state planning is about 0.33 ms/frame across dozens of functions.
CUDA issue-API residency (1.498) is not collectible beyond the measured
0.78-0.81-us marginal launch price (about 0.33 ms/frame for impossible full
deletion); the remainder is driver-internal locking that only launch-count
reduction (closed) or graphs (closed) would address.

**No exclusive architecture-independent host subtree reaches 0.500 ms/frame
in steady state.**

## Sequence survivor refuted: VS -> clip/raster fusion

`/tmp/residual-sequence-audit/result.json` surfaced one unbuilt candidate:
fusing generated VS into `cp_clip_rast_fused` (pair pool 0.824 ms/frame
union-exclusive). The pool is not collectible by fusion:

- The arithmetic of both kernels remains; fusion removes only the boundary.
- VS outputs must stay in global memory for fragment interpolation, so the
  position round trip does not disappear.
- The same-stream VS-end to clip-start gaps sum to 0.717 ms/frame, but the
  frame is host-bound: those gaps are host issue-floor idle on one stream,
  and the host work between the two launches survives fusion, so the gap
  moves rather than vanishes. The host-side saving is about 48.5 removed
  launches/frame at the marginal price, roughly 0.04 ms/frame.
- Cross-domain fusion (per-vertex to per-triangle) needs a grid-wide sync,
  the same family as closed device-chaining/persistent entries.

Realistic collectible is an order of magnitude below the gate. Refuted
without a build.

## Conclusion

After the geometry, vertex, threading, and shader-content censuses and these
three audits, no untried mechanism on this machine has a defensible
>=0.500-ms/frame collectible ceiling. The remaining gap to the 5.0-ms goal
(2.604 ms) exceeds the sum of every surviving sub-gate lead. Progress now
requires B200/sm_100 measurement (the primary target, where the
architecture-dependent fused-VS pool and device pacing must be re-derived),
or a renderer-level redesign outside the incremental-mechanism space this
campaign covers.
