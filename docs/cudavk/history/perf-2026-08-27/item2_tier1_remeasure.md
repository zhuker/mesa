# Tier 1 re-measure on the default path, with the fixed revert arm (b6232f5ff3f)

Same build directory `/tmp/drain-tree/build-cudavk-drain`, rebuilt in place.
My runs started at epoch 1787857658, after the disclosed rebuild window
(1787857225-1787857251); no run of mine overlaps it.

## Exclusivity

1 Hz sampler over the whole window, start gated on 8 consecutive idle samples.
**229 samples, window t+0..t+232 s, none foreign.** Every entry audited.

## THE GATES - all four pass

| check | result |
|---|---|
| both arms complete | **YES**, 10 of 10 runs rc=0. The control no longer loses the device. |
| submit counts | **3,022 on all six old runs, 2,994 on all four Crossroads runs**, control included |
| candidate hash | **320e993599cc** old, **e727020fc796** Crossroads - the session standard |
| **CONTROL HASH == CANDIDATE HASH** | **YES. 320e993599cc on all six old runs, e727020fc796 on all four Crossroads runs. One hash per capture across both arms.** |

The correctness gate is the one that mattered and it passes: **the reorder is a
reorder.** No byte of output moves when the hoist is reverted.

## THE MEDIANS - the claim of about zero is confirmed as about zero

Arms alternating, one binary, one session.

**Old capture** (session spread 0.119):

| rep | candidate | control | control - candidate |
|---:|---:|---:|---:|
| 1 | 12.8110 | 12.7409 | -0.0702 |
| 2 | 12.8280 | 12.9259 | +0.0979 |
| 3 | 12.8612 | 12.7833 | -0.0779 |
| **median** | **12.8280** | **12.7833** | **-0.0447** |

**Crossroads** (session spread 0.034):

| rep | candidate | control | control - candidate |
|---:|---:|---:|---:|
| 1 | 5.6760 | 5.6741 | -0.0018 |
| 2 | 5.6940 | 5.6895 | -0.0045 |
| **median** | **5.6850** | **5.6818** | **-0.0032** |

Reading, stated the way the claim was made:

* **Old: -0.045 ms/frame, i.e. the candidate is nominally SLOWER, and the
  measurement cannot resolve it.** The three pairwise differences are -0.070,
  +0.098, -0.078 - they do not agree in sign, they span 0.168 ms, and the
  control arm's own spread across its three repeats is 0.185 ms, larger than
  the session spread of 0.119 and four times the whole claimed effect.
* **Crossroads: -0.003 ms/frame**, both pairs agreeing in sign but at a tenth
  of that capture's 0.034 session spread. Also unresolvable, and an order of
  magnitude smaller than the claim's upper end.
* The claim was **0.00 to +0.05 ms/frame**. The measurement is **consistent
  with zero on both captures and cannot distinguish +0.05 from 0.00 or from
  -0.05.** It does not confirm a gain and it does not refute one.

**Recorded as what it was declared to be: a tidiness change whose value is
about zero, with an unchanged bitstream.** Its gate was the hash, and the hash
passed on ten runs.

## Standing on the record beside it

The forced-fallback pair from the earlier session (candidate 28.8512, control
28.8696, +0.0184 ms, one hash) stays as a different-regime datapoint, not as
the result. That regime runs 2.26x slower than the default path.

## Raw

/tmp/perf-audit/jobD2/ - 10 run directories, watch.log, progress.log.
