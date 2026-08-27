# The eight-site wait census, and the session's measured leads

Everything here was measured on 2026-08-26 on the reference machine, at HEAD
`e2fea470d04` unless a worktree is named. Raw data under `/tmp/perf-audit/`.

## 1. The census (JOB 12) — `/tmp/peel-tree` `bd6f2e7a81f`, `CUDAVK_WAIT_CENSUS=1`

Counts only, not a timed run. `ready` is `cuStreamQuery` before the sync: a
site with `ready > 0` would be a wait with no device work behind it. **Every
site reads `ready = 0`, 0 of 28,852 on old and 0 of 14,738 on Crossroads.**

Cross-check against `CUDAVK_PLAN_STATS` in the same run: episode drain 9092.5
vs 9091.9 ms, peel 4146.9 vs 4146.8, segment 1461.0 vs 1460.7, descriptor 17.7
vs 17.8 — agreement to 0.007%.

### old capture, per frame (1,511 frames)

| site | waits/f | blocked ms/f | mean ms | ready | ceiling ms/f | ceiling % of blocked | gap-bound / wait-bound |
|---|---:|---:|---:|---:|---:|---:|---|
| episode drain | 9.88 | **6.017** | 0.609 | 0 | **2.066** | **34.3%** | 10,819 / 4,113 |
| peel checks | 1.71 | 2.744 | 1.609 | 0 | 0.301 | 11.0% | 2,504 / 73 |
| segment counters | 4.17 | 0.967 | 0.232 | 0 | 0.253 | 26.1% | 5,896 / 405 |
| descriptor upload | 1.00 | 0.012 | 0.012 | 0 | 0.012 | 100.0% | 0 / 1,510 |
| **`vkDeviceWaitIdle`** | 2.24 | 0.514 | 0.229 | 0 | 0.500 | 97.2% | 51 / 3,339 |
| **upload-arena rewind** | 0.05 | 0.073 | **1.377** | 0 | 0.005 | 6.8% | 75 / 5 |
| **scratch reclaim** | 0.04 | 0.091 | **2.255** | 0 | 0.004 | 4.1% | 59 / 2 |
| **total** | **19.10** | **10.42** | | **0** | **3.140** | **30.1%** | |

### Crossroads, per frame (1,497 frames)

| site | waits/f | blocked ms/f | mean ms | ready | ceiling ms/f | ceiling % |
|---|---:|---:|---:|---:|---:|---:|
| episode drain | 5.57 | 2.123 | 0.381 | 0 | 0.757 | 35.6% |
| segment counters | 1.69 | 0.431 | 0.255 | 0 | 0.152 | 35.2% |
| descriptor upload | 1.00 | 0.006 | 0.006 | 0 | 0.006 | 100.0% |
| `vkDeviceWaitIdle` | 1.59 | 0.385 | 0.243 | 0 | 0.385 | 99.8% |
| scratch reclaim | 0.00 | 0.000 | 0.019 | 0 | 0.000 | — |
| **total** | **9.84** | **2.947** | | **0** | **1.300** | **44.1%** |

**Crossroads has no peel site at all**, and neither capture reaches the quad
counters.

### Three things this table says that no earlier document does

1. **Three of the eight sites appear in no budget in the docs**:
   `vkDeviceWaitIdle` (2.24/frame, 0.514 ms/frame blocked), the upload-arena
   rewind and the scratch reclaim. A census that does not know about a blocking
   call counts it as host issue time and inflates its neighbours' ceilings —
   which is measurable here: the single-site peel instrument (JOB 9) credited
   peel with 599.7 ms of issue and a 0.376 ms/frame ceiling; the eight-site
   census credits 480.2 ms and **0.301 ms/frame**, 24% lower, because it no
   longer counts those three calls as issue.
2. **The arena rewind and the scratch reclaim have the largest means in the
   driver — 1.377 and 2.255 ms per call — and cost essentially nothing**,
   0.005 and 0.004 ms/frame, because they happen 80 and 61 times in an entire
   replay. Do not read a mean without its count at these two sites.
3. **`ready = 0` everywhere.** No wait in this driver is pure overhead. Every
   one is device-paced, so its blocked time is a symptom of device work rather
   than a cost that removing the wait would recover.

### What survives, and what closes

**Six of the seven live sites close.** Peel 0.301, segment counters 0.253, and
everything else under 0.02 ms/frame of ceiling — before any conversion.

**The episode drain survives.** 2.066 ms/frame of ceiling at **34.3%** of its
blocked time, three times peel's 11.0% ratio, and **gap-bound on 10,819 of
14,932 waits (72%)** — on nearly three drains in four the host really does have
issue work in hand and the wait is the shorter term. That is qualitatively
different from peel, where the host was gap-bound too but held only 0.48 s of
gap against 4.15 s of block.

**What is not measured is the drain's blocked-to-frame conversion.** 11% is a
peel-site number (the peel patch added 1.01 ms/frame of blocking for 0.110 ms
of frame). At 11% the drain is worth 0.23 ms/frame; at perfect conversion,
2.07. Nothing in this session establishes that a site with three times the
ceiling ratio converts at the same rate. **The next measurement at this site is
the conversion, not another ceiling.**

## 2. The launch-cost microbenchmark (JOB 13)

Empty kernel, host time only, interleaved arms, minimum reported, idle GPU:

| arm | min ns/launch |
|---|---:|
| `cuLaunchKernel` | 2020.1 |
| `cuLaunchKernelEx`, no attributes | 2026.5 |
| `cuLaunchKernelEx` + PDL attribute | 1151.6 |

**The extended entry point costs +6.3 ns.** Against the 0.44 µs/link that PDL
tier 2 measured, that is a factor of seventy, so the entry point is free and
the "inter-grid gap alone" reading of tier 2 stands.

**The −868.6 ns figure is an artefact and must not be quoted as "PDL makes
launches cheaper".** At 200,000 back-to-back empty launches the host is
throttled by queue depth, so that arm measures launch *throughput with overlap
enabled*, not launch entry cost.

## 3. `CUDAVK_INLINE_FS`, re-measured at HEAD (JOB 8)

Iteration 14's same-LLVM fragment architecture has been opt-in ever since, and
had not been re-tested since the hardware texture cache became the default or
since the fan-out landed — both of which changed the fragment path underneath
it. AB/BA, both captures, hashes and submit counts clean:

| capture | candidate (`CUDAVK_INLINE_FS=1`) | control | delta |
|---|---:|---:|---:|
| old | 13.5289, 13.2455 | 13.1732, 13.1788 | **−0.2112 ms (−1.60%)** |
| Crossroads | 5.7963, 5.8282 | 5.8078, 5.8347 | +0.0090 ms (+0.15%), noise |

**Confirmed opt-in. Re-measured 2026-08-26 at `e2fea470d04`.** Nobody needs to
spend another AB/BA on this.

## 4. PDL level 4 (JOB 14) — the mechanism does not engage

| | offers/frame | takes/frame | declines/frame |
|---|---:|---:|---:|
| old, level 3 | 551.53 | 550.87 | 0.66 |
| old, level 4 | 569.96 | **550.88** | **19.09** |
| Crossroads, level 3 | 125.47 | 125.28 | 0.19 |
| Crossroads, level 4 | 125.47 | **125.28** | 0.19 |

Added takes: **+0.004/frame on old, 0.000 on Crossroads**, against a forecast
of ~155 and ~33. Every new offer declines.

**The declines identify themselves.** 28,842 − 994 = **27,848** new declines on
old; JOB 11 counted **27,847 reusing passes** on the same capture. The level-4
offers are the reusing peel passes, and they decline because the visbuf
re-clear sits in front of stage 1 on exactly those passes. The epoch check is
working as designed, demonstrated by an independent route.

The `CUDAVK_NO_FETCH_FOLD=1` negative control is uninformative here rather than
alarming: it cannot collapse a take count that is already zero, and the 550.88
takes it would have to move are tier 1–3 links the fetch fold does not gate.
No level-4 AB/BA was run, because there is no mechanism to time.


## 5. The conversion slope (JOB 15) — `CUDAVK_WAIT_SPIN_US`, old capture

A busy-wait of D µs is injected immediately after a wait returns, outside that
wait's own timing. The slope of frame time against injected time is the site's
conversion factor. 20 runs, every one 3,022 submits and stdout hash
`320e993599cc`, zero point alternated through each sweep, D scaled per site so
the injection matches (the drain fires 9.882 times a frame, peel 1.705).

| injected ms/frame | 0.0000 | 0.2470 | 0.4941 | 0.9882 | 1.9764 | slope |
|---|---:|---:|---:|---:|---:|---:|
| **episode drain** frame ms | 13.2038 | 13.4375 | 13.7393 | 14.2005 | 15.2223 | **+1.0207** |

| injected ms/frame | 0.0000 | 0.2557 | 0.5115 | 1.0230 | 2.0460 | slope |
|---|---:|---:|---:|---:|---:|---:|
| **peel checks** frame ms | 13.1656 | 13.1880 | 13.1722 | 13.1473 | 13.1259 | **−0.0262** |

Linear region is the whole sweep on both: drain residuals under 0.031 ms
(+1.0183 with the top point dropped), peel residuals under 0.015 ms. The drain
is monotonic in D at every step.

**Three consequences.**

1. **The peel site is off the critical path, positively.** Injecting up to
   2.05 ms/frame there costs nothing. With the census ceiling of 0.301 ms/frame
   and the device busy at 0 of 2,577 checks, that is three independent
   measurements giving one answer.
2. **The 11% conversion is retired.** The rejected predication patch added
   1.01 ms/frame of blocking for 0.110 ms of frame, and that ratio was read as
   a conversion factor. It is not one: pure host time at that site converts at
   ~0, so the 0.110 ms was the patch's *mechanism* — predication issuing
   further ahead and its overshoot passes.
3. **The drain is on the critical path at par**, which is what a wait that
   empties the stream should be. It also settles the mean/median question for
   that number: the injected quantity is mean-based per frame and the response
   is a skip-50 median, so slope 1.02 measures the transfer between the two
   conventions and finds it at par. §5.2b's **2.066 ms/frame needs no
   discount**.

**Still unproven: symmetry.** This measures the *add* direction. At peel both
directions agree at ~0; at the drain only the add direction is measured, so a
deferral mechanism must still be built and measured. What changed is that its
ceiling is no longer discounted by a factor that does not exist.


## 6. The other two sites (JOB 16) — segment counters and `vkDeviceWaitIdle`

Same probe, old capture, three injection points each plus zero, two passes per
point; all 16 runs 3,022 submits and hash `320e993599cc`; monotonic in D.

| injected ms/frame | 0.0000 | 0.2502 | 0.5004 | 1.0008 |
|---|---:|---:|---:|---:|
| **segment counters** frame ms | 13.0639 | 13.3539 | 13.5776 | 13.8605 |

Point slopes 1.16, 1.03, 0.80. **Slope +1.03 fitted to 0.50 ms/frame**, +0.78
fitted to 1.00 — the sweep **saturates**, which is the host running out of slack
elsewhere. The site's own ceiling is 0.253 ms/frame, inside the linear region.

| injected ms/frame | 0.0000 | 0.2468 | 0.4937 | 1.0098 |
|---|---:|---:|---:|---:|
| **`vkDeviceWaitIdle`** frame ms | 13.1740 | 13.2669 | 13.3947 | 13.6152 |

Point slopes 0.376, 0.447, 0.437. **Slope +0.4425 over the whole range**,
residuals under 0.012 ms — linear, and about *half* on the critical path.

**Both were closed "by size" using the retired 11% and both were wrong to be.**
At 11% they read 0.028 and 0.055 ms/frame; at their measured slopes they are
**≈0.26 and ≈0.22 ms/frame**, each comparable to a whole accepted iteration.
They are sized and open.

**The three live sites are unlikely to be additive.** The segment saturation is
direct evidence: slack recovered at one site is slack another cannot use twice.
