# The peel-check lead is closed

**Old capture only.** Crossroads produced zero peel loops, so the mechanism
does not exist there at all.

---

## The mechanism, in one paragraph

A blended draw is rendered by peeling: the whole draw is re-rasterized once per
layer, and between layers the host drained the device to ask "did that pass
composite anything". That drain was the most expensive single wait in the
driver — 2.751 ms/frame over 1.70 waits, 1.6135 ms each, 20.8% of a 13.16 ms
frame. `CUDAVK_PEEL_PREDICATE` predicated a whole pass on the device flag the
rasterizer already honours, moved the interval reset to a one-thread kernel so
no host store could race it, and predicted the trip count instead of asking.
It worked, and it lost: **−0.1096 ms on the old capture**, neutral on
Crossroads.

## The three numbers that closed it

| | |
|---|---|
| **Oracle** | An *exact* predictor still pays **1.093 checks/frame** against today's 1.706, so **at most 35.9% of the checks are removable**. Uncapped it pays 0.037, so the entire margin is `CP_PEEL_PREDICT_MAX`'s own forced re-checks: **64% of today's checks are structurally unavoidable given the cap, and the cap is what keeps the patch from losing catastrophically.** |
| **Ceiling** | The most any deferral of this site could recover — `Σ min(host issue burst after the check, the blocked time)` — is **0.376 ms/frame** against 2.756 ms blocked. |
| **Idle** | **The device was never idle at a peel check: 0 of 2,577.** |

## The conversion factor

The rejected patch blocked **+1.01 ms/frame** and cost **+0.110 ms of frame**.
So **89% of added blocking cost nothing**, and blocked time converts to frame
time here at about **11%**.

Applied to the ceiling: `0.376 × 0.11 ≈ **0.04 ms/frame**` — a tenth of this
capture's run-to-run spread. That is the whole remaining value of the lead.

**The ring's 79-distinct-keys-against-64-slots overflow is NOT worth fixing.**
It is a one-constant fix and it would buy a fraction of 0.04 ms. "One-constant
fix" is a description of the defect, not an invitation.

## The five predictions, scored

| | prediction | result | verdict |
|---|---|---|---|
| P1 | exact predictor still pays ≥ 1.0 check/frame | 1.093 capped (0.037 uncapped) | **PASS** on the operative capped oracle |
| P2 | stability > 80%, < 64 distinct keys | 82.0% stable; 79 distinct, 47 slots used, 75 evictions | **SPLIT** — stability passes, the bound fails, and it fails the cheap way (a constant), not the fatal way (stability < 50%) |
| P3 | free arm < half the loops and most of the removal | 26.4% of loops, 145 of 509 removed checks (28.5%) | **SPLIT** — "< half" passes, "most" fails; hit-exact removed the larger share |
| P4 | ceiling ≤ 0.35 ms/frame | 0.376 | **MARGINAL FAIL**, well inside the stated 0.5 falsifier |
| P5 | device idle at < 5% of checks | 0 of 2,577 | **PASS**, and it is the reason the lead is closed |

## The general rule this site now supports

> **At a synchronisation site where the device is never idle when the host
> arrives, removing or deferring the host wait cannot pay.** The blocked time
> is a symptom of device work, not a cost that can be recovered. Measure the
> conversion from blocked time to frame time before valuing any wait: here it
> was about **11%**, so the site's headline 2.75 ms was worth about 0.3 ms, and
> the reachable part of that was 0.04 ms.

Two cheap instruments answer it before anything is built, both in
`cp_sync_timed` and both host-side:

* **`cuStreamQuery` before the sync.** One non-blocking call. If it never
  returns `CUDA_SUCCESS`, the site is device-paced and the wait is not the
  cost.
* **`Σ min(issue burst after the wait, the wait)`.** A deferral recovers at
  most the host work that would have been issued during it. Generous twice
  over — not every blocking point goes through one function, and the device
  work still has to happen — which is what makes a *small* answer decisive.

**CUDA events cannot answer this.** An event is a stream marker, so a pair
measures a *span* of the stream timeline and device idle inside that span is
invisible. Bracketing per pass, per stage or per kernel only subdivides the
span. This is why the instrument is host arithmetic and not another bracket.

## The methodological rule, which nearly cost us the answer

**Never read a predictor's accuracy from a run in which the predictor drives
the schedule.** With `CUDAVK_PEEL_PREDICATE=1` the same census reads **96.6%
stable and 32.8% misses**; with it off, **82.0% stable**. The flag-on figure is
an artefact: the ring records the prediction it was handed, so "predicted
exactly" is self-fulfilling and "predicted high" is invisible because the loop
simply runs to the prediction. Had Q1 been read from those runs the conclusion
would have been "the ring is excellent, the design is sound" — the opposite of
the truth.

## What is left standing

* The cap worked. `CP_PEEL_FREE_MAX` and `CP_PEEL_PREDICT_MAX` held the loss to
  0.11 ms where an unbounded overshoot on a 256-layer draw was worth 1.8 ms.
* Correctness was never the problem: 67/67 with the flag off and 67/67 with it
  on, byte-identical stdout between the arms on both captures.
* The failure mode is worth naming: **removing a check and letting the loop
  issue further ahead are the same act.** The next check then absorbs
  everything issued since — checks fell 19% and the survivors grew 68%, so the
  wait was displaced rather than removed.

Flags `CUDAVK_PEEL_PREDICATE` and `CUDAVK_PEEL_CENSUS` stay in the tree, off,
on branch `peel-predicate` (`d133edc3ce1`, `c9c4cb5229c`). The census is the
reusable part.
