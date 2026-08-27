# Item 2 P0 - the inverse spin (JOB C) - RESULT

ICD `/tmp/drain-tree/build-cudavk-drain/src/cudavk/cudavk_devenv_icd.x86_64.json`,
old capture, `CUDAVK_WAIT_SPIN_SITE=episode`, `CUDAVK_PLAN_STATS=1` on every
run. 28 replays, arms strictly ALTERNATING on one binary in one session, two
full reps of the sweep, then the decoy pair twice.

## Exclusivity

1 Hz sampler over the whole window, start gated on 8 consecutive idle samples.
**931 samples, window t+0..t+944 s, none foreign.** Every entry of every
sample audited.

## Gates, checked BEFORE any median was read

* **submit count: 3,022 on all 28 runs**, one value.
* **stdout sha256: `320e993599cc` on all 28 runs**, one value - and the same
  hash job A's census produced on a different build.
* **injection line read on every run.** Mechanism count is **14,932
  injections** in every swept arm, `us wanted` and `us spun` agree to 0.1%
  (e.g. 925.8 wanted / 926.5 spun at D=62). No point injected nothing.

## THE POSITIVE CONTROL FIRST - the after-arm reproduces +1.02

`CUDAVK_WAIT_SPIN_US`, spin AFTER the sync. Mean of two reps, slope is
delta / (D x 9.88 / 1000):

| D us | median ms | delta | **slope** | rep1 | rep2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 12.7971 | - | - | - | - |
| 62 | 13.4046 | +0.6075 | **0.992** | 1.089 | 0.894 |
| 125 | 14.0881 | +1.2910 | **1.045** | 1.087 | 1.004 |
| 250 | 15.3917 | +2.5946 | **1.050** | 1.073 | 1.028 |
| 500 | 17.8596 | +5.0624 | **1.025** | 1.028 | 1.021 |
| 1000 | 22.9338 | +10.1367 | **1.026** | 1.034 | 1.018 |

**Every point is 1.02 to within its own noise; the mean over the five points is
1.028.** The port is right, the site is right, and the before-arm may be read.

## THE RESULT - the before-arm is FLAT near the origin

`CUDAVK_WAIT_SPIN_BEFORE=1`, spin BEFORE the sync, same D, same binary.

| D us | median ms | delta | **slope** | rep1 | rep2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 12.8136 | - | - | - | - |
| 62 | 12.8341 | +0.0206 | **0.034** | 0.089 | -0.022 |
| **125** | 12.8941 | +0.0806 | **0.065** | 0.110 | 0.020 |
| 250 | 13.4483 | +0.6347 | **0.257** | 0.298 | 0.216 |
| 500 | 14.8523 | +2.0387 | **0.413** | 0.412 | 0.413 |
| 1000 | 18.6310 | +5.8174 | **0.589** | 0.598 | 0.580 |

**The registered bar was "slope at D=125 below 0.25". Measured 0.065**, four
times inside the bar, and 0.034 at D=62. Neither rep exceeds 0.11 at either
point. **The premise holds: work moved to just BEFORE this drain is absorbed by
the wait instead of being added to the frame.** The "add-direction only,
symmetry unproven" caveat is now a measurement: at this site, up to about
125 us per drain (1.24 ms/frame of work) is free to within 0.08 ms/frame.

## The mechanism engaging, independently of the frame time

The driver's own `episode drain ... ms/N` line, same runs:

| D us | after-arm drain ms | before-arm drain ms | absorbed |
|---:|---:|---:|---:|
| 0 | 8,636 | 8,661 | - |
| 62 | 8,694 | 7,745 | 916 of 926 injected |
| 125 | 8,659 | 6,995 | 1,666 of 1,866 |
| 250 | 8,675 | 5,915 | 2,746 of 3,733 |
| 500 | 8,688 | 4,277 | 4,384 of 7,466 |
| 1000 | 8,714 | 2,242 | 6,419 of 14,932 |

The after-arm's drain is UNCHANGED at every D (8,636-8,714 ms), which is what
"outside that wait's own timing" is supposed to mean. The before-arm's drain
falls by almost exactly what was injected at small D. **At D=62 the spin
absorbed 98.9% of itself.** This is a second, independent instrument agreeing
with the frame median.

## THE WAIT CDF, which is what the six points buy

Slope_before / slope_after removes the conversion and leaves
`E[max(0, D-W)] / D`, i.e. the MEAN of the CDF over [0, D]. Differencing
`D x G(D)` gives the local CDF:

| D us | G(D) = mean CDF on [0,D] | **F(D) = P(W < D), local estimate** |
|---:|---:|---:|
| 62 | 0.034 | **0.034** |
| 125 | 0.062 | **0.091** |
| 250 | 0.245 | **0.427** |
| 500 | 0.403 | **0.561** |
| 1000 | 0.574 | **0.745** |

Read plainly: **only ~3% of the 14,932 episode drains per replay are shorter
than 62 us, ~9% shorter than 125 us, but ~43% are shorter than 250 us and
~75% shorter than 1 ms.** The mean wait is 0.579 ms. The distribution has
almost no mass below 125 us and a knee between 125 and 250 us. That knee is
the budget: it says how much work may be relocated to just before the drain
before the relocation starts costing.

## THE DECOY CONTROL - X3 did not fire

`CUDAVK_WAIT_SPIN_DECOY`, no injection, side-stream work during the drain.
Mechanism count printed: **238,912 clears at 16** (= 14,932 x 16) and
**955,648 at 64** (= 14,932 x 64), 0 injections, as intended.

| arm | mean wait ms | vs D=0 | frame median |
|---|---:|---:|---:|
| D=0 (4 runs) | 0.5792 | - | 12.798 / 12.814 |
| decoy 16 | 0.5678, 0.5683 | **-2.0%** | 12.880, 12.870 |
| decoy 64 | 0.5409, 0.5400 | **-6.7%** | 12.992, 12.919 |

Predicted unchanged. Measured unchanged in the falsifying direction: the mean
wait did NOT rise, it fell slightly. **X3 has not fired.** Side-stream work
does not lengthen the drain; if anything the decoy clears overlap it. The
frame median moves +0.1 to +0.2 ms, which is the host cost of issuing 64 extra
clears per drain, not a device effect.

## What this does and does not license

It licenses relocation INTO the shadow of this drain up to about 125 us per
drain. It does not license adding work: the after-arm is +1.02, unchanged and
re-confirmed here at five D values. The two arms differ by a factor of 30 at
D=125, which is the whole point of the probe.

## Raw

/tmp/perf-audit/jobC/ - 28 run directories, watch.log, progress.log.
