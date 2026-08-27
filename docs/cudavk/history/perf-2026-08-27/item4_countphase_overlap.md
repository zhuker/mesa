# Item 4 follow-up - the COUNT-PHASE overlap factor, on the shipping default

The measurement my own F4 rule demanded. Shipping build
`/home/alexzhukov/mesa/build-cudavk`, PDL level 3, fan-out on, **no**
`NO_ABUF_APPEND`, nothing set. One counter arm plus two nsys windows.

## Exclusivity

1 Hz sampler over the whole window, start gated on 8 consecutive idle samples.
**101 samples, window t+0..t+102 s, none foreign.** Every entry audited.

## GATE BEFORE READING

Counter arm on the shipping default: **1,314.2 launches/frame**, which is
`PERFORMANCE.md` §5.1's own figure to the decimal. Submits 3,022, stdout
`320e993599cc`.

Using it to size the traced windows (597,226 and 585,856 kernels -> 454.4 and
445.8 frames):

| class | measured/frame | §5.1 | agreement |
|---|---:|---:|---|
| `cp_rasterize_stage3_abuf` | 145.0 | 134.6 | +7.7% |
| `cp_rasterize_stage3` (direct) | 67.0 | 74.5 | -10.0% |

Both within a tenth of the documented counts at a different commit. **Gate
passes**; the windows are representative and may be read.

## THE RESULT - the count-phase launches are ALREADY CONCURRENT

Overlap factor = summed kernel time / union of kernel intervals. Read plainly:
**when one of these kernels is running, this is how many are running on
average.**

| class | window A | window B | launches |
|---|---:|---:|---:|
| **ABUF count-phase triple**, all three stages together | **2.50x** | **2.44x** | 197,649 |
| -- `cp_rasterize_stage3_abuf` alone | **1.89x** | **1.85x** | 65,883 |
| -- `cp_clip_rast_fused_abuf` alone (fused clip+stage1) | 1.44x | 1.43x | 65,883 |
| -- `cp_rasterize_stage2_abuf` alone | 1.22x | 1.22x | 65,883 |
| **OPAQUE/direct triple**, all three together | **2.68x** | **2.67x** | 91,386 |
| -- `cp_rasterize_stage3` alone | 1.24x | 1.23x | 30,462 |
| ALL kernels | 1.80x | 1.78x | 597,226 |

Two windows agree to 2.5% on every row.

## Reading it against the registered decision rule

The rule was: **>= ~1.8x refutes the count-phase merge; ~1.0x means serial and
the credit is available.**

Which number to apply it to matters, and I am giving both rather than picking
the convenient one:

* **The group figure is 2.50x** and is well past the bar. But a triple's group
  overlap includes PIPELINING between different stages, which a merge of "the
  same stage across the segments of one episode" would not remove.
* **The faithful figure is the per-stage self-overlap**, because that is
  exactly the population the merge would concatenate. Those are **1.89x
  (stage3), 1.44x (clip+stage1), 1.22x (stage2)**, and time-weighted across the
  three, **1.59x**.

So:

* **`cp_rasterize_stage3_abuf` - the dominant stage, 53% of the triple's
  summed time - is at 1.89x, ABOVE the 1.8x bar.** For that stage the rule
  fires and the merge is refuted by the same mechanism F4 measured.
* The two cheaper stages are at 1.22x and 1.44x - not serial, but not past the
  bar either.
* Time-weighted, **1.59x**: in the "in between" region the rule reserved for
  reporting rather than forcing.

**My reading, stated as a judgement and labelled as one:** the count-phase
merge cannot collect the full `554 x 0.782 us` credit, because none of its
mergeable populations is serial and its most expensive one is already running
1.89-way concurrent. A first-order correction is to scale the credit by the
serial fraction: at 1.59x time-weighted, roughly **1/1.59 = 63% of the launches
are not on the critical path**, so the arithmetic's 0.433 ms/frame is worth
about **0.16 ms/frame** before any device-side cost of the merged form is
subtracted - and F4 measured that cost to be, at this width, larger than the
credit.

The direct/opaque path is worse for a merge, not better: 2.68x as a triple.

## What is now safe to say about item 4

1. The `554 x 0.782 us` arithmetic is **void as written**. It prices launches
   as if each sat on the critical path; the measurement says 1.59x of them
   overlap.
2. F4 already showed the merged form costs more critical path than it saves at
   this width, on launches that were 2.19x concurrent. The count phase is
   **1.59-2.50x concurrent**. The mechanism that inverted the credit at the
   fill position is present at the count position too.
3. I do not claim it is refuted outright, because 1.59x time-weighted sits
   inside the band the rule left open. The dominant stage alone does clear the
   bar.

## Correction to my F4 report - the bound, not the result

While gating this window I found that the frames-per-window figure I used in
F4's critical-path bound was wrong. I had divided by an assumed 203.7
batches/frame; the correct normaliser is the counter arm's own launches/frame.

| | as reported | corrected |
|---|---:|---:|
| frames in cand-a window | 137.6 | **415.3** |
| merged chunks per frame | 82.3 | **27.26** |
| critical-path bound | 2.45 ms/frame | **0.81 ms/frame** |
| observed 0.410 as share of bound | 17% | **51%** |

**Nothing else in F4 changes**: the per-item ratios (0.333-0.337x), the
registers and occupancy, the overlap factors 2.19x -> 1.37x, and the union-busy
comparison are all ratios or per-unit-work quantities and are unaffected. The
corrected bound is a **tighter** fit - the observed loss is half of the
perfectly-concurrent bound rather than a sixth of it - so the causal account is
strengthened, not weakened.

## Raw

/tmp/perf-audit/jobG/ - counter run, two nsys windows, watch.log, progress.log.
