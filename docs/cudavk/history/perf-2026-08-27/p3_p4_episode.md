# JOB 5 — P3 (episode sizing) and P4 (drain-wait split)

Worktree `/tmp/bnd-tree`, branch `bounded-clip`, commit `5c8eb0428e3`, build
`build-cudavk-bnd`. One pass per capture with both probe flags and
`CUDAVK_PLAN_STATS=1`. `CUDAVK_ABUFFER_TIMING` and `CUDAVK_ABUF_FUSE_CHECK`
were not set. **These are not timed runs** — P4 records about 30 events a
frame — and no frame median from them appears in this file or in any document.

Raw stderr: `/tmp/perf-audit/p3p4/probe-{old,cross}/stderr`.

## 0. Gates

| check | old | Crossroads |
|---|---|---|
| submits | 3,022 | 2,994 |
| stdout sha256 against a flagless run on the same ICD | `320e993599cc` = `320e993599cc` | `e727020fc796` = `e727020fc796` |

**Cross-check of the probe's own attribution — passes, and the residual is
explained by a third measurement.**

| | old | Crossroads |
|---|---:|---:|
| `plan_stats` episode drain | 9,087.4 ms / 14,932 waits | 3,176.5 ms / 8,333 waits |
| probe "measured drain wait" | 9,007.5 ms / 13,899 episodes | 2,698.1 ms / 6,805 episodes |
| agreement | **99.1%** | **84.9%** |
| episodes the probe drops (left before the shade) | 1,033 | 1,528 |

The dropped episodes are exactly the `quads == 0` drains counted independently
by `CUDAVK_DRAIN_PROBE` (1,033 and 1,528), and the time the probe is missing —
0.9% on old, 15.1% on Crossroads — is exactly the empty-drain share that probe
measured (0.9% and 15.1%). Two probes written by different authors agree to a
tenth of a percent on a quantity neither was aiming at.

## 1. P3 — episode sizing

### old capture

```
cudavk: episode sizing: 13899 episodes (1033 dropped before the shade)
  total/quads: median 3.754 p90 3.866 max 3.973 (4.000 is the geometric cap)
  slots today   (4*quads): median 571392 p90 3183384 max 3299204
  slots at bound (4*total): median 2037256 p90 12307712 max 12729836  (cap 16777216)
  episode device scratch: median 64.34 MB p90 227.71 MB max 279.20 MB
  device arena: peak today 286.8 MB, projected at the bound 1069.3 MB (x3.73), arena 7914.8 MB, cap 8192.0 MB
  VERDICT (size): the wider bound FITS inside the arena (1069.3 MB projected against 7914.8 MB held, 0 regrows observed)
  capacity/4 gate: 3519 of 13899 episodes (25.3%) have total > capacity/4 and would be refused at the wider bound
cudavk: episode wait split: 13899 episodes
```

### Crossroads

```
cudavk: episode sizing: 6805 episodes (1528 dropped before the shade)
  total/quads: median 3.726 p90 3.818 max 3.992 (4.000 is the geometric cap)
  slots today   (4*quads): median 90748 p90 235432 max 2125472
  slots at bound (4*total): median 345860 p90 869944 max 8412068  (cap 16777216)
  episode device scratch: median 7.27 MB p90 16.85 MB max 152.03 MB
  device arena: peak today 157.6 MB, projected at the bound 607.3 MB (x3.85), arena 506.6 MB, cap 8192.0 MB
  VERDICT (size): the wider bound CROSSES the arena (607.3 MB projected against 506.6 MB held, 0 regrows observed)
  capacity/4 gate: 32 of 6805 episodes (0.5%) have total > capacity/4 and would be refused at the wider bound
cudavk: episode wait split: 6805 episodes
```

The `x3.73` and `x3.85` projections confirm, from a third direction, the
factor-of-four correction: a host substitute that knows only `total` needs
`4 × total`, which is 3.75× today's `4 × quads`, not 0.94×.

## 2. P4 — the drain-wait split

### old capture

```
cudavk: episode wait split: 13899 episodes
  measured drain wait: median 393 us p90 1877 us max 9659 us, total 9007.529 ms
  device counts->scan:  median 14 us p90 16 us, total 206.285 ms
  device scan->bucket:  median 57 us p90 242 us, total 1586.314 ms
  VERDICT (wait): moving the drain to the scan can remove at most 1586.314 ms of the 9007.529 ms measured here (17.6%); the chain after the scan is 88.5% of the episode's device time

```

### Crossroads

```
cudavk: episode wait split: 6805 episodes
  measured drain wait: median 245 us p90 812 us max 3039 us, total 2698.135 ms
  device counts->scan:  median 14 us p90 15 us, total 97.999 ms
  device scan->bucket:  median 37 us p90 41 us, total 246.315 ms
  VERDICT (wait): moving the drain to the scan can remove at most 246.315 ms of the 2698.135 ms measured here (9.1%); the chain after the scan is 71.5% of the episode's device time

```

Per frame, that ceiling is **1.0498 ms/frame on old** and **0.1645 ms/frame on
Crossroads**.

## 3. What P4 decides

**The count phases do not dominate — and that is not why S1d is small.**
`counts->scan` is 206.3 ms of 9,007.5 on old (2.3%) and 98.0 of 2,698.1 on
Crossroads (3.6%), with a median of 14 µs on *both* captures. What dominates is
the chain **after** the scan: **88.5%** of the episode's device time on old and
**71.5%** on Crossroads.

So S1d moves the wait a short distance not because counting is slow, but
because almost everything the host is waiting for is issued after the scan and
still has to run. **The 5.989 ms/frame drain is not the prize.** The prize is a
**ceiling** of 1.05 ms/frame on old and 0.16 ms/frame on Crossroads.

Three things argue against spending the ceiling:

1. **The ceiling assumes perfect overlap.** It is `Σ min(wait, scan→bucket)`,
   a true upper bound rather than a model, but an upper bound.
2. **Blocked host time does not convert into frame time one-for-one at these
   sites.** The peel patch (`/tmp/perf-audit/peel_measurement.md`) blocked
   **1.01 ms/frame more** and the frame did not move. Against that conversion
   rate a 1.05 ms/frame ceiling is not 1.05 ms/frame of win.
3. **On Crossroads the wider bound crosses the arena** (607.3 MB projected
   against 506.6 MB held) and 25.3% of old-capture episodes would be refused by
   the `capacity/4` gate.

Recommendation: **close S1d, or keep it only as a documented option carrying
P4's number.** The two verdict lines above are quoted verbatim and are not
editorialised; this section is the reading, and it is separable from them.
