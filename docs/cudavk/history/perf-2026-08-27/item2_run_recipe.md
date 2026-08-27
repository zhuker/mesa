# Run recipe: P0 and P1 for the episode drain

Branch `drain-work`, worktree `/tmp/drain-tree`, on `43df52c383e`.
Build: `build-cudavk-drain`, ICD at
`/tmp/drain-tree/build-cudavk-drain/src/cudavk/cudavk_devenv_icd.x86_64.json`.

Three commits:

| | |
|---|---|
| `3f62f2174c3` | **Tier 1** — the shading-group tables built above the drain. Default ON, `CUDAVK_NO_DRAIN_HOIST=1` is the revert. |
| `5666b049347`+ | **P0 and P1** — `CUDAVK_WAIT_SPIN_BEFORE`, `CUDAVK_WAIT_SPIN_DECOY`, `CUDAVK_AHEAD_CENSUS`. All default off and machine-level inert. |

Standing rules apply: GPU exclusive under a 1 Hz sampler, one `sha256sum` of
stdout per capture across every arm, submit counts checked (3,022 old / 2,994
Crossroads) **before** any median is read, `median(diff(submit_ts[::2])[50:])`
with skip 50, arms strictly alternating in one session on one binary.

---

## P0 — the direction sweep. THE ONE THAT DECIDES THE ITEM.

Two arms per sweep point, same binary, alternating. `D` is microseconds per
drain; the drain runs 9.88 times a frame on old and 5.57 on Crossroads, so the
injected ms/frame is `D * waits_per_frame / 1000`.

```
base: CUDAVK_WAIT_SPIN_SITE=episode
after  arm:  CUDAVK_WAIT_SPIN_US=$D
before arm:  CUDAVK_WAIT_SPIN_US=$D CUDAVK_WAIT_SPIN_BEFORE=1
D in { 0, 62, 125, 250, 500, 1000 }
```

**Read `cudavk: wait injection:` on every run before reading any median.** It
prints the direction, the injection count, the microseconds asked for and the
microseconds actually spun. A point that injected nothing looks exactly like the
control.

What the two arms answer:

* the **after** arm must reproduce **+1.02**. It is the positive control for the
  whole probe; if it does not, the port is wrong and the before arm means
  nothing.
* the **before** arm is the result. Predicted **flat near the origin**: slope at
  D = 125 µs below 0.25. Its slope at D is `1.02 * P(W < D)`, so the six points
  give the drain's wait CDF at 62/125/250/500/1000 µs.
* **if the before arm is not flat, stop.** The mechanism's premise is false and
  nothing at this site should be built. That is falsifier X6 firing at the
  cheapest possible cost.

### P0b — the decoy control, one extra pair

```
CUDAVK_WAIT_SPIN_SITE=episode CUDAVK_WAIT_SPIN_DECOY=16
CUDAVK_WAIT_SPIN_SITE=episode CUDAVK_WAIT_SPIN_DECOY=64
```

Read `episode drain ... ms/N` from `CUDAVK_PLAN_STATS=1` and compare the MEAN
wait against the D = 0 arm. Predicted unchanged. If the mean wait rises, work
issued on a side stream during the drain lengthens the drain, the run-ahead is
paying for itself on the device, and falsifier X3 has fired.

### P0c — the peel control, old capture only, if there is budget

`CUDAVK_WAIT_SPIN_SITE=peel` with `D` scaled by 9.88/1.71 = 5.8x. The after arm
should reproduce -0.03. It is the same validation that made the original number
believable and it costs two runs.

---

## P1 — the vertex-phase census. One run per capture.

```
CUDAVK_AHEAD_CENSUS=1 CUDAVK_PLAN_STATS=1
```

Counters only; it adds two clock reads per appended batch, so **this is a probe
run and its frame median must not be quoted**.

Read the six `cudavk: ahead census:` lines **in order**:

1. drains, closed gaps, blocked ms, mean wait — cross-check the mean against
   `CUDAVK_PLAN_STATS`' own drain line, they must agree.
2. **`ceiling all`. THE SELF-CHECK.** Divide by the replay's frame count and it
   must reproduce the wait census: **2.066 ms/frame on old, 0.757 on
   Crossroads**, and 34.3% / 35.6% of blocked. **If it misses by more than 0.30
   ms/frame, falsifier X2 has fired and nothing may be concluded from line 3 in
   either direction.**
3. **`ceiling VERTEX`. THE RESULT.** The mechanism's own ceiling. Predicted
   **0.35–0.90 ms/frame on old, 0.10–0.35 on Crossroads**. Below **0.25 on old**
   is falsifier X1 and Tier 2 is dead.
4. appendable batches after a drain, as a histogram. Predicted median **4–10 on
   old, 2–6 on Crossroads**. A median below **3 on old** is falsifier X5: the
   value is real and unreachable.
5. arena bytes per segment and per episode. Predicted peak per episode **8–40 MB
   on old**. This sets the run-ahead arena cap; the design guessed 64 MiB per
   half from a placeholder and says so.
6. `dscratch` and `scratch` at the drain — the measurement `PERFORMANCE.md` §6.4
   names as the one that decides any allocation at this site.

**X1 and X5 are built to be able to disagree.** X1 says "there is nothing to
move"; X5 says "there is something to move and nowhere to move it from". Only
their disagreement distinguishes a dead lead from a mis-designed one.

---

## Tier 1 — a normal A/B, and its claim is near zero

```
candidate: (nothing set)
control:   CUDAVK_NO_DRAIN_HOIST=1
```

The new behaviour is the default, so **the control arm carries the flag**
(WORKFLOW §4.3); put it the other way round and the printed verdict inverts.

**Revised forecast: +0.01 to +0.05 ms on old, +0.00 to +0.02 on Crossroads.**
The design's original N7 of +0.03 to +0.12 was written before the hoistable set
was read in the source, and only the fs-UBO row concatenation in it is large;
the group mapping is ~185 pointer compares. Both figures are at or under the
session spread (0.119 / 0.034), so **this is a tidiness change with a claim of
about zero and should be recorded as one whatever it measures.** Its gate is the
stdout hash, not the median.

Also worth one run: `CUDAVK_FORCE_PASS_FALLBACK=1` on both arms. It makes every
episode take the classic re-execution path, which is the only way the captures
exercise the fallback, and the hash must match between arms there too.

---

## What P1 cannot answer, and the one-line ask that can

**The exclusivity fraction of the VS kernel class.** Without it the 2.19 ms/frame
of VS kernel time cannot be turned into any frame claim, and the design
deliberately keeps it out of the forecast for that reason. It needs a device
timeline, not host counters:

> On an `nsys` capture of the old capture taken in a window whose per-frame
> launch counts check out against `PERFORMANCE.md` §5.1 (see §11.5 — two windows
> of one replay disagreed by 50x), take the union of all kernel intervals and
> report, for the VS main class: total kernel time, the fraction of it during
> which no other kernel is running, and the device-busy time removed by making
> that class infinitely fast. This is the same computation that gave
> `cp_rasterize_stage3_abuf` 16.7% exclusive and a 0.404 ms/frame infinite-speed
> ceiling.

If the exclusive fraction is small, the run-ahead's device-side story is worth
nothing and the whole case rests on host relocation — which is where the design
already puts it.
