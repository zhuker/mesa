# Recording the opaque episode's stream gate once

`cp_pass_gate_stream()` recorded a fresh event on the main stream for every
side segment, at the moment that segment was issued. Segment 0 runs on the
main stream, so every later segment's gate sat *behind segment 0's whole
launch chain* and no side segment could start until it finished. Measured with
the visbuf clear as an episode marker: 13.94 opaque episodes/frame, 7.12 with
side segments, segment-0 chain median 61.1 us against a side burst median of
284.8 us — an upper bound of 0.434 ms/frame if the two could overlap.

The gate is now recorded **once**, after the episode's clears. Segment 0 is
untouched: it stays on the main stream, which is itself a measured decision
(moving it cost ~32 us/episode and made one-segment episodes pay for a fan-out
they never use).

**The hazard this ordering originally fixed.** Every recorded draw leaves its
uniform rows owed in the upload ring, and `cp_stream_set()` flushes the owed
span on the stream it is *leaving* — the main one. A gate recorded once at
episode start would sit behind those bytes, letting a segment's kernels read an
unfilled argument arena. The symptom was the `multithreading` sample losing
whole models, intermittently. The fix is to switch to the side stream first and
flush the segment's own span *there*, so the copy is ordered ahead of its
kernels on its own stream.

That inverts the risk, so the split is bounded by a mark: `pass.upload_mark` is
taken at the end of every append, everything owed below it is flushed on main
exactly as before, and only bytes reserved since reach the side stream. The
mark is fail-safe in one direction — stale or wrong-high sends *more* on main
and degrades to the old sequence; it cannot push an older byte onto a side
stream. Between two appends of one episode the only reservations are
`cpvk_prepare_draw()`'s push blocks, because `cp_batch_flush_defer_why()` is
the only entry that leaves an episode open and it has a single caller. Under
`CUDAVK_NO_UPLOAD_COALESCE` the coalescing that makes this possible is gone, so
the per-segment gate is kept.

**Measured**, three-round alternating A/B in one session per capture, RTX 5090,
control = `CUDAVK_NO_EPISODE_GATE_ONCE=1`:

| capture | window | control | gate once | delta |
|---|---|---:|---:|---:|
| favorite3 | relevant (1391+) | 6.4580 | **6.3535** | **-0.1045** |
| favorite3 | heavy (2200-3150) | 8.0278 | 7.8750 | -0.1528 |
| favorite2 | relevant (1388+) | 5.2393 | **5.1209** | **-0.1184** |
| favorite2 | heavy (2735+) | 5.8092 | 5.6893 | -0.1199 |

Arms disjoint in every band, 0 sentinel mismatches across 12 runs, standard
hashes and full timestamp populations.

**The correctness gate that matters here is not the median.** `multithreading`
is not bit-deterministic — 20 runs of the shipping driver produce 20 different
hashes — so the historical corruption is invisible to a hash. The gate is a
tolerance compare against the NVIDIA reference over 20 runs:

| | worst pixel diff | verdicts |
|---|---|---|
| shipping baseline, 20 runs | 2084-2089 | all ok |
| gate once, 20 runs | 2082-2090 | all ok |

A lost model would move that by orders of magnitude. Also 79/79 suite and an
18-sample sweep identical to the accepted baseline.

**Not measured on the B200** (host unreachable). This is host-side stream
ordering, not architecture-dependent.
