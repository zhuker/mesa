# Item 2 - the resolve-reason probe (`CUDAVK_AHEAD_WHY`) - FINAL RUN

Binary rebuilt in place; **md5 836f7c612b9e97599b0e30161d7d5564 verified before
the first run and after the last**, unchanged. No run of mine falls in the
disclosed rebuild window (epoch 1787861710-1787861712): my sweep ended at
1787860782, ~15 minutes earlier, and this probe started at 1787861857.
Counters only - **no median from these runs is quoted.**

## Exclusivity

1 Hz sampler over the whole window, gated on 8 consecutive idle samples.
**78 samples, window t+0..t+78 s, none foreign.** Every entry audited.
Three runs, all rc=0, submits 3,022 / 2,994 / 3,022, hashes `320e993599cc` and
`e727020fc796` - the session standard, unchanged by the new binary.

## 1. THE SELF-CHECK - **PASSES EXACTLY ON ALL THREE RUNS**

`resolve` column sums:

| run | sum | required |
|---|---:|---:|
| old | **14,932** | 14,932 |
| Crossroads | **8,333** | 8,333 |
| old, MAX_SEGS=1 | **14,932** | 14,932 |

No site hint is lost. Everything below may be read.

## 2. `zero` - and it agrees with the sweep's own histogram

Old capture, `MIN_VERTS=2`:

| site | resolve | **zero-hold** | share of zero | late successor |
|---|---:|---:|---:|---:|
| `unknown` | 761 | 761 | 9.2% | **761 (100%)** |
| `scope_end` | 4,488 | **3,728** | **44.9%** | **3,727 (99.97%)** |
| `flush` | 1,764 | 0 | 0% | 0 |
| **`draw_execute`** | **3,823** | **3,823** | **46.0%** | **3,823 (100%)** |
| `admit` | 4,096 | 0 | 0% | 0 |
| **total** | **14,932** | **8,312** | | **8,311 (99.99%)** |

**The zero column sums to 8,312**, which is exactly `14,932 - 6,620` from my
own sweep histogram at this setting. (Your brief said 8,422; the histogram and
the instrument both say 8,312 - a 110-deferral difference, and the two
instruments agree with each other.)

**The answer to "whichever site carries it": it is SPLIT, 46.0%
`draw_execute` and 44.9% `scope_end`, with 9.2% `unknown`.** No single site
owns it.

## 3. `late` - **THE COLUMN THAT REOPENS THE ITEM**

**8,311 of the 8,312 zero-hold deferrals had a LATE SUCCESSOR. 99.99%.**
Per site: `unknown` 100%, `scope_end` 99.97%, `draw_execute` 100%.
Crossroads is the same: **5,520 of 5,521 = 99.98%**.

Read against your own definition, this is unambiguous and it is the good
branch:

> *"the successor existed and a required resolve came first, which is
> recoverable with NO capacity change and is the best outcome available here."*

**It is not "there was no successor at all". Item 2 does NOT close at
0.068 ms/frame.** In essentially every case where the run-ahead held nothing,
there WAS something to hold, and a required resolve arrived before it could be
taken.

## 4. Scoring the four registered checks

### C4 - **FAILS on the strict reading, passes only if `flush`+`unknown` count**

The hypothesis was `scope_begin + scope_end + opaque_append` > half of the
zero-hold population.

* `scope_begin` does not appear as a site at all.
* `opaque_append` is **0** on both captures - it carries none of it.
* `scope_end` alone is **3,728 = 44.9%**, i.e. **below half**.

So **C4 fails**, by 5 percentage points. As you instructed, that is a failure
of the EXPLANATION, not of the item: 55.7% still hold nothing and capacity
still cannot reach them.

The driver's own second grouping is what rescues the shape of the idea without
rescuing the claim: `pass_finish 7,013 (4,489 zero)` = 54.0% of the zero
population, because `pass_finish` bundles `scope_end` + `flush` + `unknown`.
**If C4 is re-read as "the pass-finish family", it clears half - but that is a
different claim from the one registered, and `opaque_append` contributes
nothing to either.**

### C5 - **FIRES. FLAGGING IT LOUDLY: `draw_execute` IS NOT 0. IT IS 3,823.**

`draw_execute` carries **3,823 resolves, all 3,823 zero-hold, all 3,823 late**,
on both the MIN_VERTS=2 run and the MAX_SEGS=1 run (identically 3,823 in both -
it is insensitive to capacity). Crossroads: 1,468.

**It is the single largest contributor to the zero population, at 46.0%.**

On whether it contradicts an existing measurement: I report the conflict rather
than adjudicate it. My `stray launches` counter reads **0** here and at all
four sweep settings, and it reads 0 in these very same runs. The two counters
are measuring different events - `stray` counts held state that escaped its
extent, `draw_execute` counts the SITE at which the resolve happened - so they
are not necessarily inconsistent: a resolve taken AT a draw, before any launch,
would be counted by one and not the other. **But the author registered
`draw_execute == 0` as his expectation, and it is not 0, so his model of where
resolves happen is wrong in a way that matters: the safety net is carrying
nearly half the population, not none of it.**

### C6 - **PASSES, and it is a clean cross-instrument check**

| | sweep's decline counter | probe's `admit` column |
|---|---:|---:|
| old, MIN_VERTS=2 | `full:4096` | **4,096** |
| old, +MAX_SEGS=1 | `full:4414` | **4,414** |
| Crossroads | `full:920 seam:15` | **admit 920, seam 15** |

**Exact agreement on every value, including the `seam:15` that only Crossroads
produces.** The new key and the old counters count one population.

### C7 - **FAILS as registered, and its failure is the finding**

Registered: *"`late` is small on scope_begin/scope_end, and is where a
recoverable population would show up if one exists."*

Measured: `late` on `scope_end` is **3,727 of 3,728 = 99.97%**, which is not
small - it is total. **The recoverable population is not a corner of the data;
it is the data.**

## 5. The MAX_SEGS=1 cross-check - it isolates `draw_execute` cleanly

With `MAX_SEGS=1` the held population is capacity-limited nowhere in the same
way (histogram collapses to `1:6620`, relocated falls to 2.0 us):

| site | MIN_VERTS=2 | +MAX_SEGS=1 | change |
|---|---:|---:|---|
| `draw_execute` zero | 3,823 | **3,823** | **identical** |
| `scope_end` zero | 3,728 | 3,748 | +20 |
| `unknown` zero | 761 | 741 | -20 |
| `admit` | 4,096 | 4,414 | +318 |
| **total zero** | **8,312** | **8,312** | **identical** |

**`draw_execute` is exactly invariant to capacity**, which confirms it is a
structural resolve site and not an artefact of the buffer filling. The total
zero population is also invariant: capacity moves work between `admit` and the
held set, but it does not change how many drains find nothing to hold.

## What this says, and what should be done next

1. **The item does not close.** 99.99% of the empty deferrals had a successor
   available; they were pre-empted by a required resolve, not starved of work.
2. **The dominant pre-emptor is `draw_execute` (46.0%), which the author
   expected to be zero.** That is the first thing to look at, and it is
   capacity-invariant, so no capacity change will touch it.
3. **`scope_end` (44.9%) is second**, and `opaque_append` - a named part of the
   registered hypothesis - contributes nothing at all.
4. The recoverable path needs **no capacity change**, which makes it cheaper
   than the hold-capacity lever my sweep pointed at. If those resolves can be
   deferred past the successor, the population my sweep could not reach
   (55.7% of drains holding nothing) becomes reachable without enlarging
   anything.

That said, the honest ceiling is unchanged: P1 measured 0.387 ms/frame of
vertex work in the shadow, my sweep converted 0.092 of it at roughly 0.75, and
nothing here raises the ceiling - it only says the gap is a scheduling problem
rather than a capacity one.

## Raw

/tmp/perf-audit/jobJ/ - 3 run directories, watch.log, progress.log with md5
before and after.
