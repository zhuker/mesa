# Tier 1 (repeat) and Tier 2 - final binary, one session

Binary `/tmp/drain-tree/build-cudavk-drain/src/cudavk/libvulkan_cudavk.so`,
**md5 f64caa0d10e6299f96a53000c73b31b4 verified BEFORE the first run and AFTER
the last** - unchanged, so nothing rebuilt underneath these runs. 30 runs,
arms strictly alternating, one binary, one session, entirely after the
disclosed rebuild window (started epoch 1787859197, window ended 1787858621).

The earlier Tier 1 re-measure is **DISCARDED**, not argued about: it began
433 s inside the rebuild window.

## Exclusivity

1 Hz sampler over the whole window, start gated on 8 consecutive idle samples.
**673 samples, window t+0..t+683 s, none foreign.** Every entry audited.

## Universal gates - all pass on all 30 runs

* **30 of 30 runs rc=0.**
* **submits: 3,022 on every old run, 2,994 on every Crossroads run**, controls
  included.
* **stdout sha256: exactly two values, `320e993599cc` (old) and
  `e727020fc796` (Crossroads)** - one per capture across BOTH tiers and BOTH
  arms, and the same values every run in this session has produced.

---

# TIER 2 (`CUDAVK_AHEAD=2`)

## (a) The `run-ahead:` line - **0 stray launches, 0 dropped. PASSES.**

Old capture, identical across all six candidate runs:

```
run-ahead: 14932 drains deferred, 9467 batches held, 9467 counts replayed,
           14932 resolves, 0 dropped after a fallback, 0 stray launches
run-ahead: 2.7 us relocated per deferral (mean), 1.61 MB scratch worst drain,
           cap 24 MB, budget 125 us
run-ahead: batches held per drain: 1:3340 2:93 3:29 4:12 5:1073 6:28 7:39
run-ahead: declines: small:6174
```

Crossroads: 8,333 deferred, 1,879 held, **0 dropped, 0 stray**, 1.0 us
relocated, histogram `1:1879`, declines `small:1853 seam:15`.

**The fail-open never fired on either capture.** Resolves equal drains exactly
(14,932 / 14,932 and 8,333 / 8,333), and counts replayed equal batches held.

## (b) One stdout hash across both arms - **PASSES**

`320e993599cc` on all twelve old runs and `e727020fc796` on all eight
Crossroads runs. Since the author could not verify rendering, this and (a) were
the gate, and both hold: **no wrong ordering is visible in the bytes.**

## (c) Submit counts - **PASSES** (above).

## (d) The held histogram - **THIS IS THE RESULT, and it is a weak engagement**

| | P1's appendable batches | Tier 2's batches actually held |
|---|---|---|
| old, median | **3** | **1** (among drains that held anything) |
| old, mean | **5.92** | **2.05** holding / **0.63 over all drains** |
| shape | 36% at one, 21% at sixteen, max 16 | 72% at one, max **7** |
| drains that held nothing | - | **10,318 of 14,932 = 69.1%** |

**Tier 2 held something at only 30.9% of drains, and a median of one batch when
it did.** P1 said a median of 3 and up to 16 were appendable. The mechanism is
live and correct, but it is capturing roughly **a tenth** of the population P1
measured as available: 9,467 batches held against 14,932 drains.

Consistent with that, the driver reports **2.7 us relocated per deferral** -
about **0.027 ms/frame** of host work actually moved, against a forecast that
needed ten times more. This is not a run that held nothing, so it is a genuine
candidate rather than a second control - but it is a candidate whose mechanism
engaged at a tenth of its designed strength, and **that, not the median, is the
finding.**

## Decline reasons - **Y5 PASSES**

**`small` is the ONLY decline reason on old (6,174), and dominates Crossroads
(1,853 of 1,868, with `seam` 15).** `budget` never fires - not once, on either
capture. So the admission rule is not budget-limited, exactly as Y5 required,
and the 125 us budget against a 39 us/drain vertex ceiling is not the
constraint. **The constraint is that the candidate batches are too SMALL to be
worth holding**, which is a different and more interesting problem: the
population P1 counted as appendable is dominated by batches the admission rule
then rejects as not worth relocating.

## (e) The medians - only now, and they are consistent with zero

**Old capture** (session spread 0.119), control-minus-candidate so positive =
candidate faster:

| pair | candidate | control | delta |
|---:|---:|---:|---:|
| 1 | 12.7728 | 12.7651 | -0.0077 |
| 2 | 12.8582 | 12.8074 | -0.0509 |
| 3 | 12.8491 | 12.7845 | -0.0646 |
| 4 | 12.7921 | 12.8261 | +0.0340 |
| 5 | 12.8762 | 12.8504 | -0.0258 |
| 6 | 12.8491 | 12.7731 | -0.0760 |
| **median of pairwise** | | | **-0.0384** |
| delta of medians | 12.8491 | 12.7959 | -0.0531 |

**Crossroads** (session spread 0.034):

| pair | candidate | control | delta |
|---:|---:|---:|---:|
| 1 | 5.6687 | 5.7194 | +0.0507 |
| 2 | 5.7195 | 5.6932 | -0.0263 |
| 3 | 5.7045 | 5.6998 | -0.0047 |
| 4 | 5.6634 | 5.7176 | +0.0542 |
| **median of pairwise** | | | **+0.0230** |

Registered forecast: **old +0.21 to +0.31**, Crossroads +0.00 to +0.06 (a null
prediction).

* **Old: -0.038 ms/frame. The forecast MISSES by its entire width** - the
  measurement is not merely below the band, it is on the other side of zero.
  Four of six pairs are negative, but the spread of the six differences is
  0.110, inside the 0.119 session spread, so the honest statement is
  **consistent with zero and firmly excluding +0.21**.
* **Crossroads: +0.023 ms/frame, inside the predicted +0.00 to +0.06 band and
  inside that capture's own 0.034 spread.** The null prediction is met, which
  is the predicted outcome and not a failure.

## What Tier 2 measures, stated once

The mechanism is **correct** (0 stray, 0 dropped, resolves matching drains, one
hash) and **weak** (30.9% of drains engaged, median 1 batch, 2.7 us relocated
per deferral). The forecast assumed it would capture P1's appendable
population; it captures about a tenth of it, and the driver names why: those
batches are declined as **`small`**, never as over `budget`.

So the item is not refuted by the frame time - the frame time simply has
nothing to show, because only 0.027 ms/frame of work was moved. **The next
question is the admission rule's `small` threshold, not the mechanism and not
the site.** Job C already proved the site absorbs up to ~125 us per drain for
free; Tier 2 is currently offering it 2.7.

---

# TIER 1 (repeat, `CUDAVK_NO_DRAIN_HOIST=1` = control)

Same binary, same session, alternating.

| capture | candidate | control | pairwise deltas | median pairwise |
|---|---|---|---|---:|
| old | 12.7769 / 12.8678 / 12.7846 | 12.7834 / 12.7971 / 12.7570 | +0.0065, -0.0706, -0.0276 | **-0.0276** |
| Crossroads | 5.6931 / 5.6915 | 5.7032 / 5.7289 | +0.0101, +0.0374 | **+0.0238** |

* **Control completes on both captures** - the fix holds, and 10 of 10 Tier 1
  runs produced output.
* **Control hash == candidate hash** on every run.
* Old **-0.028**, Crossroads **+0.024**, both far inside their session spreads
  (0.119 / 0.034) and disagreeing in sign.

**Confirms the discarded result at the same value: consistent with zero on both
captures, claim of 0.00 to +0.05 neither confirmed nor refuted, correctness
gate passed.** Recorded as the tidiness change it was declared to be.

## Raw

/tmp/perf-audit/jobH/ - 30 run directories, watch.log, progress.log with the
md5 before and after.
