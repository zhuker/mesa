# Renderer2 risk 1 settled: what surrendering the opaque fan-out costs

`RENDERER2_M1_DESIGN.md` gave a GO at an estimated **1.01-1.31 ms/frame** net
collectible, with risk 1 stated as: the tiled walk must give up the opaque
stream fan-out, whose value on these captures was unknown, and *"if it exceeds
~1.0 ms/frame the net collectible is inside its own surrender."* Entry 24 is
the precedent: the same surrender was +2.73 ms/frame on the old capture and
+0.076 on Crossroads, so it had to be measured here rather than assumed.

## Measurement

`CUDAVK_NO_OPAQUE_STREAMS=1` issues an opaque episode's segments back to back
on the main stream instead of fanning them over the pass side streams - the
concurrency the walk gives up. Strictly alternating three-round A/B in one
session per capture, main-tree build at `1acb7eabc1b` (all landed defaults),
every run with its standard stdout hash and full timestamp population, each
capture on its own window (WORKFLOW.md 4.0).

| capture | band | fan-out on | fan-out off | **surrender cost** |
|---|---|---:|---:|---:|
| favorite3 | relevant (1391+) | 6.6686 | 7.3802 | **+0.7116** |
| favorite3 | heavy (2200-3150) | 8.2397 | 9.0793 | **+0.8396** |
| favorite2 | relevant (1388+) | 5.4656 | 6.1296 | **+0.6640** |
| favorite2 | heavy (2735+) | 6.0198 | 6.7349 | **+0.7151** |

Arms are disjoint in every band; the three medians within each arm agree to
better than 0.06 ms.

## What this does to the M1 verdict

The surrender is **below the 1.0 ms/frame line risk 1 named, but not small**.
Taken at face value against the M1 estimate:

    net = 1.01 .. 1.31  -  0.66 .. 0.84   =   0.17 .. 0.65 ms/frame

which straddles the 0.500 gate and sits below it at the midpoint. **The GO is
downgraded to conditional.**

Two things keep it from being a NO-GO, and both are testable rather than
hopeful:

1. **This proxy is pessimistic by construction.** `CUDAVK_NO_OPAQUE_STREAMS`
   removes the concurrency and gives nothing back: segments run back to back
   on one stream, each still its own small launch. The walk removes the
   concurrency *and replaces it with 3,600 independent tile blocks in one
   grid*. The measured 0.66-0.84 is therefore an upper bound on what the walk
   surrenders, not an estimate of it.
2. **The walk's own cost is still unmeasured.** M1 budgets it at <=28 cycles
   of block time per reference and estimates ~0.8; entry 24's prototype
   measured 3,409. The whole question is where the real walk lands between
   those two numbers.

## Consequence for the plan

Risk 3 - the standalone walk microbenchmark over a recorded hot-tile list -
is promoted from third falsifier to **the decisive gate**, and it must run
before any driver integration. Pass/fail is arithmetic, not judgement:

* the walk must clear the M1 budget with enough margin that
  `displaced pool - walk cost - surrender` exceeds 0.500 ms/frame on the
  favorite3 heavy band, using the numbers above;
* if it lands within a small factor of entry 24's 3,409 cycles per reference,
  the design is refuted and M2 must not be written.

No kernel is integrated into the driver until that microbenchmark reports.
