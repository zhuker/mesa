# Item 2 P1 census and Tier 1 A/B (JOB D) - RESULT

ICD `/tmp/drain-tree/build-cudavk-drain/...`. 16 runs.

## Exclusivity

1 Hz sampler over the whole window, start gated on 8 consecutive idle samples.
**268 samples, window t+0..t+271 s, none foreign.** Every entry audited.

## PART 1 - the ahead census, read IN ORDER

### Line 1 - drains and mean wait

* old: **14,932 drains**, 14,931 closed gaps, 8,629.7 ms blocked, mean
  **0.578 ms**. The count is the wait census's own 14,932 exactly, and the
  mean agrees with `CUDAVK_PLAN_STATS`' drain line in the same run
  (8,629.7 ms / 14,932).
* Crossroads: **8,333 drains**, 8,332 closed gaps, 3,002.8 ms, mean 0.360 ms.
* drains/frame 9.88 (old) and 5.57 (Crossroads) - the recipe's own figures.

### Line 2 - `ceiling all`, THE SELF-CHECK. **PASSES on both captures.**

| capture | census ceiling all | per frame | wait census | miss | bar |
|---|---:|---:|---:|---:|---|
| old | 3,376.5 ms of 15,891.1 issued (39.1% of blocked) | **2.235** | 2.066 | 0.169 | 0.30 |
| Crossroads | 1,418.9 ms of 6,410.3 (47.3%) | **0.948** | 0.757 | 0.191 | 0.30 |

Both inside the 0.30 ms/frame tolerance. **X2 did not fire**, so line 3 may be
read in either direction. Note both miss HIGH and by a similar amount, which is
the expected sign: this census counts issued work the wait census could not
see.

### Line 3 - `ceiling VERTEX`, THE RESULT

* **old: 585.4 ms of 585.7 ms issued = 0.387 ms/frame** (6.8% of blocked,
  17.3% of the all-work ceiling).
* Crossroads: 107.6 of 107.6 ms = **0.072 ms/frame** (3.6% of blocked, 7.6%).

Predicted 0.35-0.90 on old: **inside the band, at its bottom edge.**
X1's bar was "below 0.25 on old" - **not fired, Tier 2 is alive**, but with
0.387 ms/frame as its ceiling, not its value.
Crossroads at 0.072 is **below** its 0.10-0.35 band.

The `of ... issued` figures are worth naming: on both captures the VERTEX
ceiling is essentially 100% of the vertex work issued (585.4 of 585.7;
107.6 of 107.6), whereas `all` is only 21% of all work issued on old. **Vertex
work is almost entirely inside the drain's shadow; most other work is not.**

### Line 4 - appendable batches after a drain, histogram

old: `1:5315 2:2006 3:1100 4:891 5:722 6:234 7:259 8:181 9:37 10:209 11:7
12:219 13:14 14:555 15:3 16:3179` over 14,931 drains.
**Median 3, mean 5.92.** Predicted median 4-10. X5's bar was "below 3 on old" -
**not fired, but only just**: the median sits exactly on the bar.
The shape matters more than the median: it is **bimodal**, 36% of drains have
exactly one appendable batch and 21% have the full 16, with a thin middle.
Crossroads: `1:4865 3:529 4:13 5:195 9:1969 11:761`, **median 1**, i.e. below
its own 2-6 prediction and more strongly bimodal.

X1 and X5 were built to be able to disagree. **Neither fired on old**, so the
lead is not dead and not mis-designed; the honest reading is that it is
smaller than forecast on both axes at once.

### Line 5 - arena demand. **Far below forecast.**

old: **0.01 MB per segment mean, 0.59 MB worst segment, 2.14 MB peak per
episode** (device), 0.00 MB managed. Crossroads: 0.00 / 0.08 / **0.35 MB**.
Predicted peak per episode 8-40 MB on old. Measured 2.14 MB - **4x below the
bottom of the band, and 30x below the design's 64 MiB placeholder.**
The allocation this measurement was meant to size is therefore cheap.
Supporting counters: 204,309 appended segments, 50 triangles and 151 vertices
each, 98.5% fused the fetch, 91.7% stable clip (old).

### Line 6 - occupancy at the drain

old: **dscratch 8.6 MB peak / 6.2 MB mean**, scratch 0.1 / 0.1.
Crossroads: dscratch 6.2 / 5.2, scratch 0.1 / 0.0.

## PART 2 - Tier 1 A/B. **THE CONTROL ARM LOSES THE DEVICE. NO MEDIAN QUOTED.**

`CUDAVK_NO_DRAIN_HOIST=1` (the control, i.e. the REVERT) fails on both
captures, every time:

| run | rc | submits | note |
|---|---:|---:|---|
| t1-cand-old-1,2,3 | 0 | 3,022 | sha 320e993599cc |
| **t1-ctrl-old-1,2,3** | **255** | **3** | `VK_ERROR_DEVICE_LOST` at capture index **4295** |
| t1-cand-cross-1,2 | 0 | 2,994 | sha e727020fc796 |
| **t1-ctrl-cross-1,2** | **255** | **3** | same failure |

**5 of 5 control runs, two captures, the same capture index, inside one
second.** It is deterministic, not a tenant and not a flake: the candidate runs
interleaved with them all completed with the session's standard hashes, and the
1 Hz sampler shows no foreign process anywhere in the window.

The flag is accepted (no unknown-flag warning) and it demonstrably takes
effect - see the fallback pair below, where the same flag runs to completion.
So this is not a wiring error in my harness; the reverted path itself does not
survive these captures.

Reading the source at `cp_renderer.c:9237-9367`: the hoisted path indexes
group-major bases (`ranges_all[group_range_base[g]]`,
`pass_group_ubos + group_row_base[g] * CP_ARG_UBO_STRIDE`), while the reverted
path calls `cp_pass_group_table_one(..., g, 0, ...)` and leaves `group_ubos` at
base 0 for every group. That asymmetry is the first thing to look at; I did not
change anything and did not commit.

**Consequence: the Tier 1 claim of "zero to +0.05 ms" cannot be tested on the
default path in this build.** Its gate was the stdout hash, and the control
produces no stdout to hash.

## PART 3 - the fallback pair, which IS a complete A/B

`CUDAVK_FORCE_PASS_FALLBACK=1` on both arms - the only way these captures
exercise the episode fallback. **Both arms completed**, including the control:

| arm | submits | sha | median ms/frame |
|---|---:|---|---:|
| candidate (hoist on, default) | 3,022 | 320e993599cc | 28.8512 |
| control (`NO_DRAIN_HOIST=1`) | 3,022 | 320e993599cc | 28.8696 |

* **F3-style gate passes: one stdout hash across both arms**, and it is the
  same hash as every normal-path run in this session.
* **candidate is faster by +0.0184 ms/frame**, inside the claimed
  0.00 to +0.05 ms band, at its low end. Treat it as the tidiness change it
  was declared to be.
* Note the regime: the forced fallback runs at 28.85 ms/frame against 12.78
  on the default path, 2.26x slower. The +0.018 ms is measured there, not on
  the default path.
* This run also proves the revert flag is functional in itself - it is the
  episode path, not the flag, that loses the device.

## Raw

/tmp/perf-audit/jobD/ - 16 run directories, watch.log, progress.log.
