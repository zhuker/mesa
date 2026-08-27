# Item 4 - the merge diagnostic (JOB F) - RESULT

ICD `/tmp/merge-tree/build-cudavk-merge/...`, old capture, PDL level 3.
`CUDAVK_NO_ABUF_APPEND=1` on BOTH arms. Candidate adds
`CUDAVK_MERGE_FILL_RAST=1`. 8 runs: two counter runs first, then three timed
pairs alternating on one binary.

## Exclusivity

1 Hz sampler over the whole window, start gated on 8 consecutive idle samples.
**267 samples, window t+0..t+270 s, none foreign.** Every entry audited.

## F1 - THE GATE. **PASSES.**

From `cudavk: kernel launches:` in the counter runs (PLAN_STATS never in a
timed arm), over 1,511 frames:

| arm | launches | per frame |
|---|---:|---:|
| control | 2,542,125 | 1,682.4 |
| candidate | 2,090,572 | 1,383.6 |
| **fall** | **451,553** | **298.8** |

Bar was "at least 250". **298.8 removed.** The mechanism fired; the timing may
be read. (The forecast band was 313-353, so the fall is 4.5% below the bottom
of the forecast - worth noting but not a gate.)

## F5 - the PDL interaction. **PASSES.**

`cudavk: programmatic dependent launches:` 1,219,140 control / 918,098
candidate = **806.8 -> 607.6 per frame, a fall of 199.2**, against a predicted
150-260. Take-up stays at 99.9% with the same 994 declined in both arms.

## F3 - correctness. **PASSES.**

**One stdout sha256 across both arms and all repeats: `320e993599cc`**, on all
8 runs, and it is the same hash as every other run in this session.
Submit count 3,022 on all 8 runs.

## F2 - THE PRICE. **FAILS, AND NOT LOW. IT FAILS INVERTED.**

| rep | candidate | control | control - candidate |
|---:|---:|---:|---:|
| 1 | 13.9412 | 13.4602 | **-0.4810** |
| 2 | 13.9291 | 13.5559 | **-0.3732** |
| 3 | 14.0028 | 13.5315 | **-0.4713** |
| **median** | **13.9412** | **13.5315** | **-0.4097** |

Forecast was **+0.20 to +0.34 ms**, "below +0.10 the price does not transfer".
Measured **-0.410 ms/frame: the candidate is SLOWER.** All three pairs agree in
sign and the spread of the three differences is 0.108 ms, inside the session
spread of 0.119, so this is not noise.

**Implied price per launch: -0.410 ms / 298.8 launches = -1.372 us per launch
REMOVED.** The settled removal credit is +0.782 us [0.633, 0.949].

So the outcome is not the one the author pre-wrote as embarrassing (F1 passes,
F2 fails low, F4 passes). It is **stronger and worse than that: at this
position the launch-width credit does not merely fail to transfer, it arrives
with the WRONG SIGN.** Removing 298.8 launches per frame COST 0.410 ms per
frame. Something in the merged form costs about 2.15 us per launch removed
more than the launches were worth.

## F4 - **NOT MEASURED, and it is now the decisive follow-up**

F4 needs per-kernel device time (merged stage3 per item vs control per item,
from nsys). I did not run it: it was specified as the confound check for an
F2 that failed LOW, and I have no nsys pass in this session.

It is now the most informative run left on this item, because F2's inversion
is exactly the failure F4 was built to explain: the merged entry point resolves
a block to its item through a prefix and stages 400 B of per-item state through
shared memory, which raises register and shared pressure and loses the
per-launch specialisation of the three stage bodies. A device-side cost of
~1.4 us per removed launch is the natural candidate, and F4 would confirm or
refute it in one nnsys capture.

## What this says about the merge programme

This position was chosen because it changes launch width and NOTHING about when
work is issued, so a number here is a statement about launch count alone. The
statement it makes is: **launch width is not free to buy at this width.** A
merge that removes 300 launches per frame here does not collect 0.23 ms; it
pays 0.41 ms.

The count-phase merge was to be justified on the same currency, with its
deferral cost weighed against a "known width credit". **There is no width
credit at this position to weigh.** Before anything lands on the episode drain
- the site with the measured +1.02 conversion, where any delay is paid at par -
the device-side price of the merged form has to be measured (F4), because on
this evidence it is larger than the launch saving.

Context for scale: the same binary at the shipping default (no
`NO_ABUF_APPEND`) runs 12.83 ms/frame. `NO_ABUF_APPEND=1` alone costs +0.70,
and the merge on top of it costs a further +0.41.

## Raw

/tmp/perf-audit/jobF/ - 8 run directories, watch.log, progress.log.
