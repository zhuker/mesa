# Item 2 - the `CUDAVK_AHEAD_MIN_VERTS` sweep (final run)

Same binary as the Tier 1/Tier 2 session, **md5
f64caa0d10e6299f96a53000c73b31b4 verified before the first run and after the
last** - unchanged. Old capture, `CUDAVK_AHEAD=2`, control empty, arms
alternating, 3 reps per value, 24 runs.

## Exclusivity

1 Hz sampler over the whole window, gated on 8 consecutive idle samples.
**732 samples, window t+0..t+742 s, none foreign.** Every entry audited.

## 1. Stray launches and drops - **0 AT EVERY VALUE, INCLUDING 0**

| MIN_VERTS | drains deferred | batches held | counts replayed | resolves | dropped | **stray** |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 14,932 | 9,467 | 9,467 | 14,932 | 0 | **0** |
| 8 | 14,932 | 16,822 | 16,822 | 14,932 | 0 | **0** |
| 2 | 14,932 | 36,129 | 36,129 | 14,932 | 0 | **0** |
| 0 | 14,932 | 36,129 | 36,129 | 14,932 | 0 | **0** |

Resolves equal drains exactly and counts replayed equal batches held at every
value. **Lowering the threshold to zero did not make the held state escape its
extent.** The fail-open never fired, so this is a tuning question throughout
and not a correctness one.

## 2. Stdout hash - **ONE VALUE ACROSS ALL 24 RUNS: `320e993599cc`**

Every arm, every value, every rep, plus 3,022 submits on all 24. **No ordering
was being hidden by today's threshold.**

## 3. Relocated work - RISES 3.3x, THEN SATURATES

| MIN_VERTS | us relocated per deferral | **ms/frame of host work moved** | vs the 0.387 ms/frame vertex ceiling |
|---:|---:|---:|---:|
| 32 (default) | 2.8 | **0.028** | 7% |
| 8 | 4.6 | **0.045** | 12% |
| 2 | 9.3 | **0.092** | **24%** |
| 0 | 9.3 | **0.092** | 24% |

The threshold was worth a **3.3x** increase in relocated work, and **V=0 is
byte-identical to V=2** - the sweep saturates before the threshold is even
removed. At best it reaches **24% of the ceiling P1 measured**, still a factor
of **4.2** short.

## 4. The held histogram - approaches P1's shape, then hits a wall at 8

| | drains holding nothing | median (holding) | max |
|---|---:|---:|---:|
| P1's appendable population | - | 3 (mean 5.92) | **16** |
| MIN_VERTS 32 | 69.1% | 1 | **7** |
| MIN_VERTS 8 | 62.4% | 2 | **8**, with 963 at the cap |
| MIN_VERTS 2 / 0 | 56.4% | 1 | **8, with 4,114 at the cap** |

At V=2 the distribution is bimodal and **63% of the drains that hold anything
hold exactly 8** - the cap. P1 said a median of 3 and up to 16 were available.
**The mechanism can no longer take them.**

## 5. WHICH DECLINE DOMINATES - **the constraint MOVED, and it is not `budget`**

| MIN_VERTS | declines |
|---:|---|
| 32 | `small:6174` (only reason) |
| 8 | `small:4024`, **`full:949`** |
| 2 | **`full:4096`** (only reason) |
| 0 | **`full:4096`** (only reason) |

**`small` recedes exactly as predicted and disappears entirely by V=2. It is
replaced by `full`, not by `budget`.** `budget` does not fire once at any
value on any run - the 125 us budget is never reached, and P0's measured
absorption of ~125 us per drain is never tested.

This is the third of the three outcomes you named: `small` recedes and
**something else appears**, and it names the next real limit. **The limit is
the hold capacity of 8**, a fixed extent in the implementation, not a
measurement-derived threshold. At V=2 the mechanism is refusing 4,096 batches
purely because the buffer is full while its time budget is 13x unspent.

## 6. The medians - the frame does NOT follow the relocated work

Control-minus-candidate, positive = candidate faster. Session spread 0.119.

| MIN_VERTS | pairwise deltas | **median** | relocated ms/frame |
|---:|---|---:|---:|
| 32 | -0.1171, -0.0086, -0.0106 | **-0.0106** | 0.028 |
| 8 | +0.1748, -0.0512, +0.0086 | **+0.0086** | 0.045 |
| 2 | +0.0284, +0.0302, -0.0935 | **+0.0284** | 0.092 |
| 0 | +0.0694, +0.1569, -0.0455 | **+0.0694** | 0.092 |

There is a **monotone trend in the medians** (-0.011, +0.009, +0.028, +0.069)
that tracks the relocated work, and its sign is the one the design wants. But
**every one of those four medians is inside the 0.119 session spread**, and at
every value at least one of the three pairs has the opposite sign. **None of
these is a measurement of a real gain.** The largest, +0.069 at V=0, is 58% of
the session spread and rests on pairs of +0.069, +0.157 and **-0.046**.

## The answer, stated once

**The threshold was NOT the whole constraint, and the relocation still does not
convert.**

* The threshold was real and worth 3.3x: `small` was rejecting two-thirds of
  the population and removing it more than tripled the work moved.
* It saturates at V=2 against a **hold capacity of 8** that is not
  measurement-derived. `budget` - the one limit P0 actually measured, and
  measured as generous - never binds at all.
* Even fully saturated, the mechanism moves **0.092 ms/frame** of the
  **0.387 ms/frame** P1 said was there, and the frame's response to that is
  **+0.069 ms at best, inside the session spread**.

Against your registered expectations: relocated work **did** rise
substantially, and the frame **did** follow it in sign and roughly in
magnitude - +0.069 ms of frame for 0.092 ms of relocated work is a plausible
conversion near 0.75, and is exactly what a real relocation into a free shadow
should look like. **What is missing is not the conversion but the
magnitude**: 0.092 ms/frame simply cannot produce an effect this workload can
resolve at a 0.119 spread.

So this is not "the moved work was never on the critical path" (the frame did
move, in the right direction, monotonically), and it is not "holding costs more
than it saves" (no value made the frame worse). It is: **the mechanism is
correct, its admission rule has now been fixed, and it is still capacity-bound
at a factor of four below the population it was designed to capture.**

The next lever is the **hold capacity of 8**, not `MIN_VERTS` and not the site.
Whether raising it is worth doing should be decided on the arithmetic first:
reaching P1's full 0.387 ms/frame at the ~0.75 conversion this sweep suggests
would be worth about 0.29 ms/frame - which is the original forecast, and it is
now attached to a capacity change rather than a threshold change.

## Raw

/tmp/perf-audit/jobI/ - 24 run directories, watch.log, progress.log with md5
before and after.
