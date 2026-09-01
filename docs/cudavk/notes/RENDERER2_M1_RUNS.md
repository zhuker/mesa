# Renderer2 risk 2 settled: where the opaque work actually sits

`RENDERER2_M1_DESIGN.md` proposed the **leading opaque prefix run** as the
smallest M2 slice, and stated the falsifier: *"falsified if the leading opaque
run holds under 20% of a scope's references at p50."*

Measured with the run census (`df67e222c0c`), `CUDAVK_TILE_CENSUS=16`, full
replays of both captures, standard hashes, full timestamp populations,
analyzer clean, each capture on its own frame window.

## The literal falsifier tripped, hard

| | favorite3 | favorite2 |
|---|---:|---:|
| drawing scopes that **open** with an opaque run | 1,250 (3.9%) | 1,563 (5.2%) |
| leading-opaque-prefix share of scope references, p50 | **0.0%** | **0.0%** |
| same, p90 | 0.6% | 0.7% |

Against a 20% bar this is not close. **The leading opaque prefix is not a
viable M2 slice** - implementing it would turn the flag on over almost no
work.

## The premise underneath it got stronger

Classifying every run instead of only the first one:

| run class | favorite3 runs | share of refs | favorite2 runs | share of refs |
|---|---:|---:|---:|---:|
| **opaque** | 5,221 | **95.0%** | 5,165 | **96.4%** |
| blended | 28,920 | 5.0% | 28,397 | 3.6% |
| other | 5,225 | 0.0% | 3,593 | 0.0% |

Opaque runs carry essentially all the binned geometry - about 1.24 opaque runs
per drawing scope - but they **do not start the scope**. Scopes typically open
with a blended or other run and reach their opaque bulk later.

Run shape is also simpler than the design assumed: runs per scope p50 **1**,
p90 2, p99 13 (max 19), against M1's assumed 4-8. Runs end by scope end or by
state incompatibility; only 485 run ends in the whole favorite3 replay are
class changes.

## Consequence: change the slice, not the design

The M2 slice becomes **the maximal opaque run wherever it sits in the scope**,
not the leading prefix. The cost of moving it is one extra handoff edge: a
prefix run needs correct state only on the way *out*, a mid-scope run needs it
on the way *in* as well. M1's handoff is already symmetric - two buffers,
`cp->depthbuf` and `fb->color` (`RENDERER2_M1_DESIGN.md` section 1) - and an
interior run's inputs are simply whatever the earlier runs already wrote
through the existing chain, which is unchanged. No new mechanism is required.

This also improves the addressable pool: the slice now covers 95-96% of tile
references instead of the ~0-0.6% the prefix would have reached.

## Status of the three M1 risks

1. fan-out surrender - **measured**, 0.66-0.84 ms/frame (`RENDERER2_M1_FANOUT.md`);
   the estimate net of it straddles the 0.500 gate.
2. opaque prefix empty - **confirmed empty; slice redefined** (this document).
   Addressable references restored to 95-96%.
3. cycles per reference - **still unmeasured, and now the decisive gate.** The
   standalone walk microbenchmark over a recorded hot-tile list decides
   whether M2 is written at all.
