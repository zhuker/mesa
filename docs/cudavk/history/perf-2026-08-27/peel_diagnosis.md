# Peel site — diagnosis probe (Q1 predictor, Q2 deferral ceiling)

Worktree `/tmp/peel-tree`, branch `peel-predicate`. Instrumentation only: no
kernel is added, no schedule changes, nothing is written to stderr except once
at teardown. Flag: **`CUDAVK_PEEL_CENSUS=1`**, bool value, default off.

This is a probe, not a patch. The rejected patch (`d133edc3ce1`) is untouched
and still off by default.

---

## 0. What the measurement already proves, before any new code

| | control | candidate | delta |
|---|---:|---:|---:|
| checks/frame | 1.706 | 1.381 | −0.325 (−19.0%) |
| ms per check | 1.611 | 2.701 | +1.090 (+67.7%) |
| blocked ms/frame | 2.748 | 3.731 | **+0.982** |
| frame ms | 13.1842 | 13.2938 | **+0.110** |

The host blocked **0.982 ms/frame more** and the frame got only **0.110 ms
worse**. So **0.872 ms of added blocking — 89% of it — cost nothing at all.**
Blocking at this site is very nearly free, which is the same statement as: the
device already has work outstanding whenever the host arrives at a peel check.
The converse is what kills the design — if blocking longer is nearly free, then
blocking less is nearly worthless.

That is an argument, not a measurement, and the two probes below turn each half
of it into a number.

---

## 1. Q1 — why only 19%? What the census counts

Every finished peel loop is classified against the same ring the predicated
schedule uses, and the loop's own trip count is recorded.

* **arms** — `free` (`peel_passes <= CP_PEEL_FREE_MAX`), `hit exact`,
  `hit LOW` (predicted under: one extra check, falls back to doubling),
  `hit HIGH` (predicted over: passes wasted predicated), `ring MISS`. Loops,
  checks and passes for each.
* **the oracle line** — how many checks an *exact* predictor would still pay:
  one confirming check per loop that converges before its own bound, zero for
  a loop the bound stops, plus the extra checks `CP_PEEL_PREDICT_MAX` forces on
  a loop longer than the cap. **This is the ceiling for Q1**: no ring, however
  good, removes more checks than `today − oracle`.
* **trip-count and loop-bound histograms** — 1, 2, 3, 4, 5-6, 7-8, 9-16, 17-32,
  33-64, 65-128, 129+.
* **the key, as a population** — distinct `(vs, fs, num_triangles)` triples in a
  4096-entry census table (deliberately far larger than the 64-entry ring, so
  "the key is unstable" and "the ring is too small" are two different
  findings), how often a key repeats, and **how often a repeat carries the same
  trip count as last time**. Plus the ring's used slots and evictions.

**Run it with the predicated schedule OFF.** With the schedule on, the ring
records the prediction it was given, so "predicted exactly" is self-fulfilling
and a high prediction is invisible — the loop simply runs to it. Only the
classic run measures whether the trip count is genuinely a function of the key.
The census runs in both configurations and prints which one was active.

---

## 2. Q2 — can a removed wait pay at all? The cheapest honest instrument

**What does not work.** CUDA events are stream markers, so an event pair
measures a *span* of the stream timeline and any device idle inside that span
is invisible. Bracketing per pass, per stage or per kernel does not separate
busy from idle; it only subdivides the span. `ab->ev[]` and `cp_stage_end`
have this shape and cannot answer the question. An external profiler could,
but it perturbs, needs the GPU, and is not cheap.

**What does work, host-side and for two API calls per check:**

1. **The deferral ceiling.** A removed wait can recover at most the host work
   that would have been issued during it. `cp_sync_timed` is the single choke
   point for all five wait sites, so it now records, for each peel check, the
   host issue burst that followed it up to the *next blocking wait of any
   kind*, and accumulates

   > `ceiling = Σ min(issue burst after the check, the check's blocked time)`

   This is the arithmetic the iteration-29 addendum's P2 proposed for the
   episode drain, and it is an **upper** bound twice over: not every blocking
   point in the driver goes through `cp_sync_timed`, so the burst measured is
   if anything too long; and the device work the wait was waiting for still has
   to happen, which the bound does not charge for.

2. **`cuStreamQuery` before the sync.** Was there *any* outstanding device work
   when the host arrived? A check that finds the stream already idle is a wait
   that had nothing to wait for. One non-blocking API call, taken outside the
   timed region.

Also reported: peel-loop host wall time split into blocked and issuing, which
is the magnitude the ceiling is bounded by.

---

## 3. PREDICTIONS, stated before the run

Written now so that the result is a test rather than a story. Each has a
number and a falsifier.

**P1 — the oracle line closes Q1, not the ring.** An exact predictor still pays
**≥ 1.0 check/frame** on the old capture against today's 1.706, so **at most
~40% of the checks are removable in principle**, and 19% is not the small
fraction of a large opportunity it looks like. Mechanism: a loop with
`peel_passes > 4` that converges before its bound must pay one confirming
check, and I predict those loops are the majority.
*Falsified if* the oracle is below 0.5 checks/frame — then two thirds of the
checks really are removable and the ring is worth fixing.

**P2 — the key is right, the ring is not the bottleneck.** Key stability
**> 80%** on repeats (the trip count *is* a function of `(vs, fs,
num_triangles)`), and **fewer than 64 distinct keys** on the old capture, so
the 64-entry ring is not thrashing and `ring_evict` is small.
*Falsified if* distinct keys ≫ 64 with high stability — then the ring is simply
too small, the fix is one constant, and the arithmetic changes. *Also falsified
if* stability is below ~50% — then the trip count is not a function of that
triple, the design is wrong at the root, and no ring tuning saves it.

**P3 — most peel loops are longer than the free arm.** `peel_passes <= 4`
covers **less than half** of the loops and its checks alone account for most of
the 0.325 removed. Corollary: the `hit exact` arm removed close to nothing in
the candidate run.
*Falsified if* the free arm covers most loops — then the 19% is a bug in the
patch, not a property of the workload.

**P4 — the deferral ceiling is small: ≤ 0.35 ms/frame, and I expect
0.05–0.20 ms.** After a check inside a loop the host has only the next pass's
~11 launches (~10–20 µs) to issue before it blocks again; only the *last* check
of a loop is followed by a long burst.
*Falsified if* the ceiling exceeds ~0.5 ms/frame — then there is real host work
to overlap here and the site is alive for a scheme that overlaps it without
letting the loop run further ahead.

**P5 — the device is never idle at a check: `ready` < 5% of checks.**
*Falsified if* a large fraction of checks find the stream already idle — then
the 1.611 ms is not device work and something else entirely is being measured.

**What I expect the combination to say:** P1 + P4 + P5 together close the lead.
The honest form of the closure is: *at this site "removing the wait" and
"letting the loop issue further ahead" are the same act, the second is what
makes the next check absorb everything issued since, and the host has only tens
of microseconds of work to put in the gap. The device is the pacer through the
peel loop, so the 2.75 ms of blocking is a symptom of device work, not a cost
that can be recovered.* If the numbers say that, close the lead.

---

## 4. Exact commands for the measurer

Not a timed run — the census adds two clock calls and one `cuStreamQuery` per
check. Do **not** read frame times from these runs; read the stderr tail.

```bash
MESA=/tmp/peel-tree
ICD=$MESA/build-cudavk-peel/src/cudavk/cudavk_devenv_icd.x86_64.json
GFX=~/gfxreconstruct/build
OLD=~/headless_streamer_20260814T155742.gfxr
CROSS=~/headless_streamer_1818_20260817T173522.gfxr
D=/tmp/perf-audit/peel-census
mkdir -p $D

# 1. THE HONEST Q1 RUN: classic schedule, shadow ring.
for cap in old:$OLD cross:$CROSS; do
  n=${cap%%:*}; f=${cap#*:}
  VK_DRIVER_FILES=$ICD CUDAVK_PEEL_CENSUS=1 CUDAVK_PLAN_STATS=1 \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported "$f" \
    > $D/$n-classic.out 2> $D/$n-classic.err
done

# 2. WHAT THE CANDIDATE ACTUALLY DID: predicated schedule, live ring.
for cap in old:$OLD cross:$CROSS; do
  n=${cap%%:*}; f=${cap#*:}
  VK_DRIVER_FILES=$ICD CUDAVK_PEEL_CENSUS=1 CUDAVK_PEEL_PREDICATE=1 \
    CUDAVK_PLAN_STATS=1 \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported "$f" \
    > $D/$n-predicated.out 2> $D/$n-predicated.err
done

grep -E "PEEL CENSUS|cudavk:   (arm|ORACLE|trip|loop|keys|Q2|peel)" $D/*.err
grep -E "main-thread waits|render scopes" $D/*.err
```

`CUDAVK_ABUF_FUSE_CHECK`, `CUDAVK_ABUFFER_VERIFY`, `CUDAVK_ABUFFER_TIMING` and
`CUDAVK_DEBUG_WORK` must stay unset as always; each of them synchronises.
`CUDAVK_PLAN_STATS=1` is included only for the render-scope count, which is
what the per-scope figures divide by; convert to per-frame with the frame count
the harness already computes.

Four runs, one per capture per schedule. Crossroads runs zero peel checks, so
its census should print no loops or only a handful — that is itself a check on
the instrument.

---

## 5. How to read the output

```
cudavk: PEEL CENSUS (schedule: classic), N loops over S render scopes, ...
cudavk:   N checks, N passes (x per loop), N loops ran to their own bound, ...
cudavk:   arm free (<=CP_PEEL_FREE_MAX)  ... loops (x%), ... checks (y% of all)
cudavk:   arm hit, exact                 ...
cudavk:   arm hit, predicted LOW         ...
cudavk:   arm hit, predicted HIGH        ...
cudavk:   arm ring MISS                  ...
cudavk:   ORACLE: an exact predictor still pays K checks (k/scope), ...
cudavk:   trip counts: 1=.. 2=.. 3=.. ...
cudavk:   loop bounds: ...
cudavk:   keys: D distinct ...; R repeats, S with the SAME trip count (x% stable)
cudavk:   Q2 deferral ceiling: blocked B ms ..., device ALREADY idle at Q of C,
          ... min(issue, block) = Z ms total, z ms/scope
cudavk:   peel loops cost W ms of host wall time, B blocked and I issuing
```

The three numbers that decide the lead:

1. **ORACLE checks/frame** vs 1.706 — the ceiling on Q1.
2. **`min(issue, block)` per frame** — the ceiling on Q2, in milliseconds.
3. **`% stable`** — whether the key is the right key at all.

If (1) is ≥ 1.0 and (2) is ≤ 0.2 ms, the lead is closed and should be recorded
as closed with those two numbers. If (3) is below 50%, the lead is closed for a
different and more interesting reason: the trip count is not a function of the
draw, and every prediction-based scheme at this site is dead.

---

## 6. What the probe costs, and what it cannot say

* Two `os_time_get_nano` calls and one `cuStreamQuery` per peel check; one
  `os_time_get_nano` pair and about forty counter updates per peel loop. At
  1.7 loops/frame that is well under 10 µs/frame, but it is not zero, which is
  why these runs are not timed runs.
* 128 KB of context for the census key table.
* With the flag off, every one of these paths is behind
  `cp_debug->peel_census` and nothing above is reached. `cp_debug_doc.py
  --check` and `cp_no_getenv.py` pass; `cp_launch_audit` passes; the build is
  clean.
* **It cannot separate device-busy from device-idle inside a stream span** —
  see §2. It answers "can the host recover anything here" and "was there
  outstanding work at the check", which is the decision the lead needs, and it
  does not answer "what fraction of the peel loop is the device saturated". If
  the ceiling comes back large and the lead survives, *that* question needs an
  external profiler over the NVTX `pass %u` ranges the loop already emits, and
  it should be asked then and not before.
