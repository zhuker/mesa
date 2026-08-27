# JOB 3 — the peel predication patch (`CUDAVK_PEEL_PREDICATE=1`)

Worktree `/tmp/peel-tree`, branch `peel-predicate`, commit `d133edc3ce1` on
`e2fea470d04`; build `/tmp/peel-tree/build-cudavk-peel`. Candidate arm carries
the flag, control is empty, base environment empty. `CUDAVK_ABUF_FUSE_CHECK`,
`CUDAVK_ABUFFER_VERIFY`, `CUDAVK_ABUFFER_TIMING` and `CUDAVK_DEBUG_WORK` were
unset in every arm. GPU idle before the job. Raw data `/tmp/perf-audit/peel/`.

## Verdict

**The mechanism engaged and the frame got worse.** Not accepted.

| capture | candidate | control | delta | expected |
|---|---:|---:|---:|---|
| old | **13.2938** | 13.1842 | **−0.1096 ms (−0.83%)** | +0.20 to +0.50 ms |
| Crossroads | 5.8131 | 5.8196 | +0.0066 ms (+0.11%) | neutral by construction |

Crossroads is neutral, which is the designed outcome and its only job — it runs
zero peel checks, so it can only show absence of harm, and it does. **The
cross-replay verdict must not be read as a wash: old is a regression and
Crossroads is a non-test.** The regression on old is past the harness's own
−0.10 ms block threshold.

## The runs, gated before the medians were believed

| run | median ms | submits | rc | stdout sha256 |
|---|---:|---:|---:|---|
| old cand1 | 13.3331 | 3022 | 0 | `320e993599cc` |
| old cand2 | 13.2545 | 3022 | 0 | `320e993599cc` |
| old ctrl1 | 13.1657 | 3022 | 0 | `320e993599cc` |
| old ctrl2 | 13.2027 | 3022 | 0 | `320e993599cc` |
| Crossroads cand1 | 5.8108 | 2994 | 0 | `e727020fc796` |
| Crossroads cand2 | 5.8154 | 2994 | 0 | `e727020fc796` |
| Crossroads ctrl1 | 5.7999 | 2994 | 0 | `e727020fc796` |
| Crossroads ctrl2 | 5.8394 | 2994 | 0 | `e727020fc796` |

All eight runs are full replays — 3,022 on old and 2,994 on Crossroads — with
one stdout hash per capture, identical between the arms. Nothing died early,
which matters here because the patch changes trip counts and an early exit is
exactly the failure that would have looked like a large win.

**The arms do not overlap on old**: the fastest candidate run (13.2545) is
slower than the slowest control run (13.2027). The sign is consistent, not a
coin flip inside the spread.

## The suite, run on the GPU the author did not have

| arm | result |
|---|---|
| `meson test -C build-cudavk-peel --suite cudavk`, flag off | **67/67 Ok, 0 Fail** |
| same, with `CUDAVK_PEEL_PREDICATE=1` set | **67/67 Ok, 0 Fail** |

Logs `/tmp/perf-audit/peel/suite_off.log` and `suite_on.log`. Correctness is
not the problem with this patch.

## The mechanism count — this is the diagnosis

`CUDAVK_PLAN_STATS=1`, one run per arm per capture (instrument on, so read as
diagnosis rather than as the timing result):

| old capture | candidate | control | change |
|---|---:|---:|---|
| **peel checks per frame** | **1.381** | 1.706 | **−19.0%** |
| **peel blocked ms/frame** | **3.731** | 2.748 | **+0.983 ms/frame (+35.8%)** |
| mean per check | **2.701 ms** | 1.611 ms | **+67.7%** |
| episode drain ms/frame | 6.053 | 6.024 | +0.029 |
| segment counters ms/frame | 0.987 | 0.985 | +0.002 |
| total blocked ms/frame | 10.784 | 9.768 | **+1.014** |
| frame median in these runs | 13.2155 | 13.2448 | −0.029 |

**Predication engaged: the check count fell by 19%** (2,578 → 2,087 checks over
the replay). The gate you set is met — this is not a case of the frame moving
while the mechanism sat idle.

**But it removed one check in five and made the survivors 68% more expensive**,
so the site's blocked time rose by 0.98 ms/frame. The wait was displaced, not
removed: with the check predicated away the peel loop issues further ahead, and
the next check waits for everything issued since. `CP_PEEL_FREE_MAX=4` bounds
how far, which is why the cost is bounded and not catastrophic.

Crossroads confirms the read from the other side: 0 peel checks in both arms,
every other counter within noise, frame within noise.

## The line the next person at this site must read first

**The candidate blocks 1.01 ms/frame MORE while being 0.029 ms FASTER in the
instrumented pair, and 0.110 ms slower in the clean AB/BA.** Both frame deltas
are small; the blocking delta is not. The only way to hold all three is that
**the added host blocking at the peel site is largely overlapped with device
work the host was going to wait for anyway.**

That is a statement about the *site*, not about this patch.

**Later measurement, and it changes the reading of this file: the ratio
0.110 / 0.982 = 11% is NOT a conversion factor.** `CUDAVK_WAIT_SPIN_US`
(JOB 15) injected up to 2.05 ms/frame of pure host time at this exact site and
the frame did not move — **slope −0.0262**. Host time at the peel check is
free. So this patch's 0.110 ms was its **mechanism** — predication issuing
further ahead, and the overshoot passes that follow — and not its blocking.
**Do not use 11% to size anything.** The same probe reads **+1.02** at the
episode drain, so the conversion is per site and differs by two orders of
magnitude between two sites in one driver.

§5.2's 2.751 ms/frame of peel blocking is an upper bound that is not
collectable, and the peel site is now closed by three independent
measurements — census ceiling 0.301 ms/frame, device busy at 0 of 2,577 checks,
and injected host time free.

The clean AB/BA is the timing result; the instrumented pair is the mechanism.

## What this does not settle

The 19% take rate is the number to attack, not the 68%. Three checks in four
still run, so this is not a test of "predication removes the wait" — it is a
test of "predication removes a fifth of the waits and lengthens the rest". If
the prediction ring can be made to fire on the other 81% the arithmetic changes
completely; as measured, the trade is negative on the only capture that runs
the mechanism at all.

Escalating to `cp_decisive_ab.sh` would sharpen −0.11 ms into a number with a
distribution behind it, but it would not change the sign, and the sign is what
decides the patch. I did not spend it. Say the word if you want it.
