# The ceiling for a "fewer, wider launches" architecture in cudavk

Analysis only. **No GPU work, no benchmark, no code change, no build.** Every
number is either quoted from a document of the 2026-08-26 session, supplied by
the shape-curve and merge-census agents, or recomputed here from an nsys trace
that session already stored (`/tmp/perf-audit/tiling/stage3/old_d12.sqlite`,
426,759 kernels, 5.84 s, 447 frames). Recomputing a stored trace is not a
measurement of the card; the card was not touched.

Base commit `e2fea470d04`. Old capture unless said otherwise.

**Three things in this document disagree with the briefs I was given.** They are
in §1.4, §2.3 and §3.2 and each is marked **DISAGREEMENT**. I would rather be
wrong in the open.

---

## 0. Instrument validation, and what is new here

The device half rests on the union-busy simulation of `stage3_imbalance.md`
§3.3. I re-implemented it from the stored trace and reproduced every published
value exactly, so the new rows below are on the same instrument, not a new one.

| quantity | published | recomputed |
|---|---:|---:|
| union device busy in the window | 51.5% | **51.54%**, 6.734 ms/frame |
| `stage3_abuf` kernel time | 1.897 | **1.897** |
| raster chain kernel time | 6.56 | **6.559** |
| `stage3_abuf` 2x / 4x / infinite | 0.187 / 0.288 / 0.404 | **0.1866 / 0.2873 / 0.4039** |
| raster chain 2x / deleted | 0.763 / 2.106 | **0.7627 / 2.1063** |

**New rows (device union busy removed, ms/frame):**

| classes shrunk by f | 1.33 | 2 | 4 | 8 | inf |
|---|---:|---:|---:|---:|---:|
| raster chain (6 classes) | 0.359 | 0.763 | **1.298** | **1.648** | 2.106 |
| chain + `main` | 0.842 | 1.775 | 2.931 | 3.688 | 4.695 |
| every kernel | 1.349 | 2.791 | 4.457 | 5.470 | 6.734 |

**New measurement — where the device idle actually is.** For every device
operation I took the union gap immediately in front of it (`start` minus the
running maximum `end`) and attributed it to that operation's class.

**CORRECTED IN REVISION 2 (see §7.1): my first version of this table unioned
kernels only and therefore mis-attributed 4.8 ms/frame of idle that is in fact
filled by memcpy and memset.** The corrected table, over kernels + memcpy +
memset (union busy 7.659 ms/frame, 58.6%):

| class | ops/frame | idle in front, ms/frame | us per op |
|---|---:|---:|---:|
| **memcpy** | 514.0 | **4.255** (2.430 of it in gaps > 200 us) | 8.28 |
| memset | 167.3 | 0.381 | 2.28 |
| `main` (every generated shader, VS and FS) | 225.4 | 0.162 | 0.72 |
| `cp_fs_writeback` | 48.5 | 0.082 | 1.69 |
| `cp_fs_compact` | 48.5 | 0.069 | 1.43 |
| **whole raster chain** | **447.3** | **0.132** | **0.30** |
| **total traced device idle** | | **5.412** | |

* **80.8% of raster-chain kernels start with exactly zero union gap.** The chain
  runs back to back on the device. This is a ratio, so it survives every
  instrument caveat, and it is the load-bearing finding.
* **1.89 gaps per frame above 200 us hold 2.430 ms/frame, and every one of them
  is in front of a memcpy** — the frame boundary. That agrees independently with
  `PERFORMANCE.md` §5.2 (2.20 gaps/frame over 200 us; inter-submit stall median
  2.453 ms), which the kernel-only version did not.
* Absolute idle sizes here are traced (5.412 ms/frame) against an untraced
  figure of 3.70. Neither is authoritative — see §7.1. **Use the ratios.**

---

## 1. The three axes, and which are the same milliseconds

### 1.1 The frame identity

| old capture, per frame | ms |
|---|---:|
| frame | 13.1626 (13.21 in the `PLAN_STATS` run) |
| host issuing | 3.478 |
| host blocked | 9.734 |
| device idle | 3.679 (6.33 in the trace window, assumption A6) |
| device union busy | 6.734 (window) / 9.51 (§5.2, assumption A6) |

The driver alternates. Host issue ~ device idle; host blocked ~ device busy. All
43,590 censused waits found the stream busy (`ready = 0`).

### 1.2 (a) device busy and (c) wait duration are the SAME milliseconds

A wait shortens because the device work behind it shortened. This is not an
argument, it is in the record: the opaque fan-out removed no kernel and no
launch, only overlapped an episode's segments. It showed up as **−2.650 ms at
the episode drain and nowhere else** (every other site moved < 0.02 ms) and the
frame moved **−2.555 ms**. Counting (a) and (c) separately would have priced that
change at 5.2 ms. It paid 2.55.

> **k_dev = 2.555 / 2.650 = 0.964 ms of frame per ms of device union busy
> removed.** Central 0.96; pessimistic arm 0.6. It is n = 1 (A1).

### 1.3 (b) launch count is a different millisecond from (a) — and it splits in two

* **(b1) host CPU per launch.** Spent while the device is idle. Disjoint from
  (a), which is spent while the host is blocked. **Additive with (a).**
* **(b2) device front-end serialisation per launch.** Would appear as device
  idle *between* kernels, inside the host's blocked time. Disjoint from (a)
  (idle is not busy), so additive with (a) — but **not** additive with
  (c)-as-deferral, because it shortens the very wait a deferral would run past.

The parent asked whether (b) is additive with (a) + (c). **Answer: (b1) is
additive with everything. (b2) is additive with (a) and subtractive against
deferral. (c) is not a third quantity at all** — it is the host-side view of (a)
and (b2). The only separable part of (c) is deferral, i.e. the episode-drain
lead, and that is anti-correlated with both (a) and (b), see §3.3.

### 1.4 DISAGREEMENT 1 — the 2.047 us launch floor is not 2.69 ms of removable host issue

The brief says: *"at 1,314.2 launches/frame that is >= 2.69 ms/frame spent in
launch floor alone ... ~77% of host issue time. Use 2.047 us, not the 0.6-2.0 us
range from the price table."* I cannot use it as a price, for four reasons, and
the fourth is new evidence.

1. **It is the same number the session already refused to quote.**
   `pdl_landing.md` §8.3 measured `cuLaunchKernel` at **2020.1 ns** in a
   back-to-back loop and its own author wrote: *"at 200,000 back-to-back empty
   launches the host is throttled by queue depth, so what is being timed is
   launch throughput, not entry cost."* 2.047 us and 2.020 us agree to 1.3%.
   The new curve is a better *shape* instrument, but on the absolute it is the
   same measurement with the same caveat, taken the same way.
2. **A rate limit is not a price.** 2.047 us/launch is a ceiling of about 489
   launches per millisecond. The driver issues 1,314.2 launches in a 13.16 ms
   frame — **one launch per 10.0 us, 20% of the front end's rate.** Over the
   3.478 ms issue window alone it is one per 2.65 us, i.e. 77% of the limit, so
   the driver is near-saturated *during a burst* and far from it over the frame.
   Both are true and they support opposite conclusions; the microbenchmark
   cannot distinguish them.
3. **The arithmetic leaves no room for the driver.** 1,314 x 2.047 = 2.690 ms.
   Add 438.4 copies and 235.4 clears at even 1 us and the host has **0.11
   ms/frame** left for gfxrecon dispatch, Vulkan entry points, command
   recording, descriptor-arena building, plan bookkeeping and 16.76 wait sites.
   At 0.5 us per copy/clear it has 0.45 ms. The high reading is possible only in
   the second case, and only just.
4. **NEW — the front-end cost does not appear as device idle in front of the
   work a merge would delete.** §0: the raster chain's union gap is **0.30 us per
   launch, 0.132 ms/frame, zero for 80.8% of its launches.** So **(b2) is
   already close to zero for the chain**: merging 554 chain launches can recover
   at most ~0.13 ms of device idle, not 1.13 ms. Whatever the 2.047 us is, the
   device is not paying it between chain kernels.
5. **NEW, from the merge census — the driver issues device work only from inside
   `vkQueueSubmit`, on the application thread.** Recording builds an op list and
   touches CUDA not at all. So the frame-boundary gap is the replayer decoding
   and re-recording, with nothing queued by construction. **No change to launch
   width or launch count can touch it**, and it is the largest single block of
   device idle in the driver (2.430 ms/frame in 1.89 gaps). This and item 4 are
   one statement: the chain is one dependency chain issued in one burst, and the
   idle is *between* bursts, not inside them.

**What I use instead.** The driver's own in-situ price, **0.6–1.0 us of frame
time per launch removed** (iteration 27, 194.3 launches, 0.13–0.21 ms), as the
measured central value, and **2.047 us** as the optimistic arm for the case
where the floor is genuinely host CPU *and* every removed launch sits on the
critical path. The gap between them is not noise, it has a mechanism: **72% of
episode drains are gap-bound** — the host runs out of work before the wait ends
— so host issue inside those windows is already hidden behind device work and
removing it pays nothing. Host issue only costs frame time where the device is
idle waiting for it, which §0 locates precisely: the 4.20 ms frame-boundary gap
in front of `main`.

This is R3/F5 exactly: a constant measured at one site (a saturated launch loop)
carried to another (a driver issuing at a fifth of that rate). Three leads were
mis-sized that way today.

---

## 2. The joint bound

    Δframe  =  k_dev × ΔUB  +  p × ΔL          (there is no third term)

`ΔUB` device union busy removed (§0 table); `k_dev` = 0.96 (0.6 pessimistic);
`p` = 0.6–1.0 us measured / 2.047 us optimistic; `ΔL` launches removed.

### 2.1 The realistic point, from the merge census

The census says only one merge is provably legal: **the same stage across the
segments of one episode.** Inside a draw the chain cannot merge — stage 2 reads
stage 1's device-built queue and the launch boundary is the publication barrier;
the weaker version was built and lost 24.2 -> 27.2 ms. So:

| | |
|---|---:|
| segments/frame | ~200 |
| chain launches inside them | ~600 of 1,314.2 |
| after concatenation (15.50 episodes/frame x 3 stages) | ~46 |
| **launches removed** | **~554 (−42%)** |
| median stage-3 grid | 12 items -> ~150 items |
| per-item cost on the shape curve | ~880 ns -> ~120 ns (**7.3x**) |
| per-item cost from the driver's own deep-queue launches | 32–37k -> 7.8–9.5k SM-cycles (**~4x**) |

Device factor: **f = 4 (driver-corroborated) to 8 (shape-curve)**. I do not use
the 287x figure: it is a synthetic item, while the driver's stage-3 item is a
64x64 tile walk with 41–68% of stall cycles on the L1TEX scoreboard. 4–8x is
what two independent instruments agree on for real items.

| component | pessimistic | central | optimistic |
|---|---:|---:|---:|
| (a) device, chain f=4..8, x k_dev | 0.78 (f=4, k=0.6) | 1.25–1.58 | 1.58 |
| (b) 554 launches x p | 0.33 (p=0.6) | 0.33–0.55 | 1.13 (p=2.047) |
| (b2) chain device gap recovered | ~0.00 | ~0.00 | ~0.15 (inside p) |
| **joint** | **1.11** | **1.58–2.13** | **2.71** |

> **THE JOINT BOUND FOR THE PROVABLY-LEGAL MERGE: 1.1–2.7 ms/frame on the old
> capture, central 1.6–2.1 ms, i.e. 8–21% of the frame, most likely 12–16%.**
>
> On Crossroads the same mechanism is worth much less: 346 launches/frame total,
> 0.98 episodes/frame fanned out, and no exclusivity simulation exists. The
> launch half is bounded at 346 x 0.6–2.047 us = **0.21–0.71 ms** and the device
> half is **unbounded from existing data**.

### 2.2 Sensitivity, including the hypothetical rows

Launch axis alone (host half), k_dev-free:

| launches fall to | removed | p=0.6 | p=1.0 | p=2.047 |
|---|---:|---:|---:|---:|
| **760 (−42%), the legal merge** | **554** | **0.33** | **0.55** | **1.13** |
| 50% | 657 | 0.39 | 0.66 | 1.35 |
| 25% | 986 | 0.59 | 0.99 | 2.02 |
| 10% | 1,183 | 0.71 | 1.18 | 2.42 |
| 0%, impossible | 1,314 | 0.79 | 1.31 | 2.69 |

Device axis alone (k_dev = 0.96), two readings of "device busy falls by X%":

| | kernel time −25% | −50% | −75% |
|---|---:|---:|---:|
| chain only | 0.34 | 0.73 | 1.25 |
| chain + FS | 0.81 | 1.70 | 2.81 |
| every kernel | 1.29 | 2.68 | 4.28 |

| union busy −X% | ΔUB | frame at k=0.96 | what it needs |
|---|---:|---:|---|
| −25% | 1.68 | 1.62 | every kernel 1.45x, or chain+FS 1.9x, or **the chain 9x** |
| −50% | 3.37 | 3.23 | every kernel 2.45x. **Unreachable from the chain at any factor** |
| −75% | 5.05 | 4.85 | every kernel 5.7x. Unreachable from chain+FS at any factor |

**The distance between the two device readings is the whole of F6.** Cutting the
chain's kernel time by 75% — a large programme — moves union busy by 1.30 ms,
19% of union busy, because only the exclusive fraction converts.

### 2.3 DISAGREEMENT 2 — 2.106 is not the device ceiling of the architecture

The brief says the device side is *"bounded by the union-busy numbers ... which
say the whole raster chain is worth 2.106 ms even if it became free."* That is
the ceiling of a **raster-chain** programme, and the legal merge is one, so it
binds §2.1. It is **not** the ceiling of a wide-launch *architecture*: with the
fragment shaders included the same simulation gives 4.695 ms, and with every
kernel 6.734 ms. The bound that matters for §2.1 is 2.106; the larger numbers
matter only if the census is later extended to `main` (225 launches/frame, 63%
exclusive — the highest-value class in the driver and the only one the census
did not clear).

### 2.4 The frame-time floor

Ping-pong model, calibrated so it reproduces today exactly:

    frame = H' + B'
    H'  = 3.478 − ΔL × p
    B'  = (6.734 − ΔUB) + (2.14 − ΔGAP) + 0.86

2.14 ms = measured non-frame-boundary device idle inside waits; 0.86 = the
residual that closes `B = 9.734` today; ΔGAP = the part of that idle the merge
removes, which §0 measures at **0.17 ms** and not, as I first modelled it, in
proportion to the launch count. My own first version of this model scaled the
in-wait idle with φ and over-predicted the merge by 0.7 ms. The chain's gap is
0.186 ms/frame in total, so that is the cap. Check of the identity today:
3.478 + 6.734 + 2.14 + 0.86 = 13.21 ms.

| scenario | H' | B' | frame | Δ |
|---|---:|---:|---:|---:|
| today | 3.48 | 9.73 | 13.21 | — |
| legal merge, f=4, p=0.6 | 3.15 | 8.27 | 11.41 | 1.80 |
| legal merge, f=8, p=1.0 | 2.92 | 7.92 | 10.84 | 2.37 |
| legal merge, f=8, p=2.047 | 2.34 | 7.92 | 10.26 | 2.95 |
| φ=0.25, chain 4x, p=1.0 (hypothetical) | 2.49 | 8.25 | 10.74 | 2.47 |
| φ=0.10, every kernel infinite, p=2.047 | 1.06 | 0.86 | 1.92 | (nonsense limit) |

The model runs about 0.2–0.3 ms hotter than §2.1 because it implicitly applies
k_dev = 1.0 to the device term where §2.1 applies 0.96 and a 0.6 arm. **§2.1 is
the headline; this table is the structural check**, and the two agree within
that difference.

**Hard floors, in order of how hard they are.**

1. **frame ≥ device union busy.** 6.734 ms today; 5.44 ms with the chain 4x;
   5.09 ms with the chain 8x. If §5.2's 9.51 ms base is the right one instead,
   every floor here rises by 2.8 ms (A6).
2. **frame ≥ host issue.** 3.478 ms today; 2.35 ms even at the optimistic
   launch price with 42% of launches gone.
3. In the ping-pong the two **add**; in a perfect pipeline they would be
   `max()`. The driver is at the additive end and no part of this programme
   moves it towards `max()`. **That is the drain lead's job, not this one's.**

So the legal merge lands the frame at about **10.3–11.4 ms** (11.0–11.6 by §2.1's
more conservative arithmetic), and the floor it is approaching is the union-busy
floor of ~5.1–5.4 ms, which no launch-side change can reach.

---

## 3. Break-even

### 3.1 Against the episode drain (up to 2.07 ms/frame)

The target is `ceiling 2.066 x slope 1.02`. It is an upper bound
(add-direction; symmetry unproven; a mechanism still has to be built), so it is
being compared against upper bounds. That is fair.

* **Launch count alone cannot beat it at the measured price.** Beating 2.07 ms
  needs 2,070 launches removed at 1.0 us or 3,450 at 0.6 us; the driver issues
  1,314. **The launch axis alone beats the drain if and only if the marginal
  frame price of a launch exceeds 1.575 us AND essentially every launch can be
  merged away.** The in-situ measurement says 0.6–1.0. The rate limit says
  2.047. This is the single number that decides the axis, and §4 item 1 says how
  to settle it in one run.
* **The legal merge's launch half is 0.33–1.13 ms.** It cannot beat the drain on
  its own under any of the three prices.
* **The chain's device half is capped at 2.106 ms** (chain deleted) and at
  1.25–1.58 ms at f = 4–8, so it cannot beat the drain on its own either.
* **Together, at the central estimate, the merge ties the drain: 1.6–2.1 against
  up to 2.07.** At the optimistic launch price it wins by ~30%; at k_dev = 0.6
  and p = 0.6 it loses by half.

**Verdict: a draw, decided by two numbers nobody has measured** (the true
marginal launch price, and whether f is 4 or 8). This is not a refusal. It is a
statement that the wide-launch programme is in the same class as the largest
open item in the driver, at roughly ten times the blast radius.

### 3.2 DISAGREEMENT 3 — they are not additive, and the merge makes the drain worth less

The brief hoped that (b) attacking host issue makes the programme *"worth
materially more than the ~2 ms I told you to beat"*. The total does go up. But
the same mechanism reduces the drain's value:

* the drain's ceiling is `Σ min(issue gap, blocked)` and **72% of its waits are
  already gap-bound** — the host runs out of work first. The merge removes 554
  launches of host work, so the gap shrinks and more waits become gap-bound.
* the merge shortens the episode's device work, so `blocked` shrinks too. Both
  terms of the `min` fall.
* in the other direction, a landed deferral hides host issue behind device work,
  which drives the marginal launch price towards its measured low end or below.

**Expect the pair to deliver 2.1–3.0 ms together, not 3.7–4.2.** Assume the
union is nearer the maximum than the sum (A7).

The order still matters and the recommendation is unchanged from the pre-census
draft: **drain first.** One mechanism, one site, a slope measured directly at
1.02 by a probe that validated itself against a control reading −0.03. The merge
is an architecture change whose device half rests on a factor measured two ways
that disagree by 2x. And after the drain lands, (a) is the only lever left in
the driver — the merge is the right second programme.

### 3.3 Against PDL, which it partly replaces

PDL converts 550.87 links/frame at level 3 and about **400 of them are launches
the merge deletes** — 73% overlap. PDL is measured at **+0.3815 ms** (the
handoff records +0.4342 on old from the decisive alternating run; either way it
is ~0.4 ms).

* **The merge subsumes about 0.28 ms of PDL's 0.38 ms.** Whichever lands first
  claims that part.
* **Measured on top of PDL, the merge will read about 0.28 ms smaller** than
  against the shipping default. Anyone measuring the merge must state which
  control it used, or R6 applies.
* PDL's residual value after a merge is about **0.10 ms** — inside the 0.119 ms
  session spread on old. **A landed merge effectively retires PDL as a separate
  item**; the links it accelerates would no longer exist.
* **Recommendation: land PDL now** (it is measured, built and small) and treat
  its 0.38 ms as an advance against the merge's 1.6–2.1, not as an addition to
  it. If the merge is later built, its correct headline is
  **merge-minus-PDL ≈ 1.3–1.8 ms**, measured against a PDL-enabled control.

---

## 4. What still cannot be bounded, and the exact measurement that would

1. **The marginal frame price of a launch. This one number decides the whole
   (b) axis** and it spans 0.6 to 2.047 us — a factor of 3.4 and the difference
   between 0.33 and 1.13 ms on the realistic row. The microbenchmark cannot
   settle it; it is queue-throttled by construction. **Two cheap settlements:**
   (i) `perf record -F 1999 --call-graph dwarf` on a replay and read self time
   in `libcuda`'s launch path against total user time — if launch entry is 2.69
   of 3.478 ms/frame it will be 77% of host self time and unmissable;
   (ii) `CUDAVK_WAIT_SPIN_US`-style injection in reverse: add a fixed number of
   *null launches* per frame and read the slope, which prices a launch in situ
   at the driver's own issue rate. (ii) is the same instrument that settled the
   drain and it takes one sweep.
2. **Is the device factor 4 or 8?** The driver's own deep-vs-shallow queue
   comparison says ~4x but compares *different launches with different work*;
   the shape curve says 7.3x at 150 items but on a synthetic item. The driver's
   stage-3 item is L1TEX-latency-bound at 41–68% of stall cycles, so its
   scaling with block count is not the synthetic kernel's. **Needed: the same
   items run as 13 launches of 12 and as 1 launch of 150,** `sm__cycles_active`
   and `smsp__inst_executed` on both. This moves the central estimate by ±0.33 ms.
3. **The device-busy base is uncertain by 40%.** The window says union busy
   6.734 ms/frame (51.5%); `PERFORMANCE.md` §5.2 says device resident 9.51 ms
   (72%), a figure carried from the iteration-28 profile at a 15.99 ms frame.
   The window is also lighter than §5.1 (100.6 vs 134.6 `stage3_abuf`
   launches/frame; 955 traced kernels/frame against 1,314 `CP_LAUNCH`es).
   **Needed: union busy over a window whose per-frame launch counts match §5.1,
   at HEAD.** If 9.5 is right, every ΔUB here is understated by up to a third.
4. **k_dev has n = 1.** 0.964, from the fan-out. A second device-side change
   with both its device effect and its frame effect measured would turn an
   anecdote into a conversion. This is the device-side twin of R2', and R2'
   exists because exactly this was borrowed once already.
5. **Can `main` merge across segments?** The census cleared the chain and stopped
   there. `main` is 225 launches/frame, 3.096 ms/frame, **63.2% exclusive** — by
   far the best exclusivity in the driver — and **4.99 ms/frame of device idle
   sits in front of it**. If the same order-free argument covers fragment
   shading, the device half of this programme roughly doubles (chain+FS f=4 =
   2.93 ms of ΔUB). If it does not, §2.1 is the whole story. **This is the
   highest-value open question in the programme.**
6. **The host price of a merged launch is not the price of a removed launch.** A
   concatenated launch still needs its segment list described — a bigger
   argument block, or a device-built descriptor. If that costs 0.3 us per merged
   segment, `p` halves and §2.2 halves with it. Unpriced.
7. **What the 2.14 ms of in-wait device idle actually is.** The chain owns only
   0.186 ms of it; `main` owns 0.886 and `fs_compact` 0.357. Whether those gaps
   are front-end latency (attackable by merging) or the drain round trip
   (attackable only by deferral) is not derivable from the trace, and it decides
   whether §3.2's substitution is partial or total.
8. **Crossroads has no device-side data at all** — no exclusivity simulation, no
   union busy, no per-class exclusive share, no measured slope at any site.

---

## 5. What is measured, what is inferred, what is assumed

**Measured, in this driver, and load-bearing:**
frame 13.1626; host issue 3.478; host blocked 9.734 over 16.76 waits; device
idle 3.679; 1,314.2 launches/frame; union busy 6.734 ms/frame and every ΔUB in
§0 (recomputed and reproduced); the per-class idle attribution in §0 (new);
exclusivity 16.7% for `stage3_abuf`, 63.2% for `main`; k_dev 0.964 from the
fan-out; the drain slope +1.02 and the peel slope −0.03; the price table
(launch 0.6–1.0 us, removal 1.30 us, hand-off 32 us); PDL +0.38–0.43 ms and
550.87 links/frame; the merge census population (~200 segments, ~600 launches,
~46 after concatenation).

**Inferred from the shape curve** (measured, but on a synthetic kernel, so
transfer is an inference): per-launch duration constant to ~170 blocks; width
free below one block per SM; the 3% issue rate is grid shape, not kernel code;
block size is not the lever; per-item 880 -> 120 ns at 150 items, which is where
`f = 8` comes from.

**Assumptions, numbered so they can be attacked:**
* **A1.** k_dev = 0.96 from one event; pessimistic arm 0.6.
* **A2.** Union busy is the right currency. It is translation-invariant, so
  "shrink in place" is correct for it even for a dependent chain. The fan-out
  says the frame responds to it at 0.96.
* **A3.** Merging N serial launches equals a uniform shrink by f = N, capped by
  machine width. Supported by the shape curve; f is applied uniformly to all six
  chain classes, which no real merge would do.
* **A4.** `main` cannot merge (conservative). §4 item 5.
* **A5.** The trace window is representative. It is 20–25% lighter than §5.1 and
  I did not correct for it in either direction; a denser phase has more kernel
  time but also more overlap, and the sign of the net effect on ΔUB is not
  derivable.
* **A6.** Host figures at a 13.21 ms frame are used with a 13.1626 ms headline;
  the 0.05 ms mismatch is inside the 0.119 ms session spread. Device busy is
  taken as 6.734 (window), not 9.51 (§5.2).
* **A7.** The merge and the drain are substitutes; their union is nearer the
  maximum than the sum.
* **A8.** ~554 removed launches all sit on the critical path when priced at the
  optimistic 2.047 us. The measured 0.6–1.0 us arm makes no such assumption.
* **A9.** The chain at f = 4 (1.298 ms) and f = 8 (1.648 ms) are computed from
  the trace, not interpolated.
* **A10.** The 2020.1 ns / 2.047 us launch loop is used **only** as an
  optimistic arm and as a rate limit, never as a price. Using it as a price
  would inflate the host half by 2–3x, which is the same class of error the
  session made three times today.

---

## 6. The answer in one paragraph

(§7 is a later revision that corrects one of my own measurements, folds in the
structural stall, sizes two new leads and ranks the whole programme. The
paragraph below survived it unchanged.)

A "fewer, wider launches" architecture changes **two** independent quantities,
not three: device union busy, and launch count. Wait duration is those two seen
from the host side; counting it separately would double the programme. The one
merge the census proves legal — the same stage across an episode's segments —
removes **554 of 1,314 launches** and widens the median stage-3 grid from 12 to
~150 items, and it is worth **1.1–2.7 ms/frame, central 1.6–2.1 ms (12–16% of
the frame)**, of which **1.25–1.58 ms is device-side and 0.33–1.13 ms is
launch-side**. That **ties** the episode drain's ~2.07 ms rather than beating
it, and it is **not additive with it**: the merge removes both the host work a
deferral would run ahead with and the wait it would run ahead of. It also
**subsumes about 0.28 ms of PDL's 0.38 ms**, so the three must be counted as one
programme worth roughly **2.1–3.0 ms**, not as three worth 4.5. The launch axis
alone can never beat the drain unless the marginal frame price of a launch is
above **1.575 us**, and the only in-situ measurement of it says **0.6–1.0 us**
while the only microbenchmark says 2.047 us in a loop the driver does not run.
**Settle that one number and the factor-of-two on the device side, land PDL and
the drain deferral first, and re-size the merge against a PDL-enabled control on
a driver that has stopped ping-ponging** — because after deferral the device
half is the only half left, and it is the half this architecture is actually
about.

---

## 7. Revision 2 — three corrections, two new leads, and the ranking

Written after the merge census named the stall and after a caution on my own
gap attribution. **The headline of §2.1 is unchanged by all three corrections:
1.1–2.7 ms/frame, central 1.6–2.1.** That it survived is worth as much as the
number.

### 7.1 A correction to my own §0 — and it was not tracer inflation

The caution I was given is that my traced idle (6.33 ms/frame) exceeds the
untraced 3.70, so the absolutes carry tracer inflation. The caution is right to
distrust the absolutes, but **the actual defect was mine and larger: I unioned
kernels only.** Adding memcpy and memset:

| | kernels only | + memcpy + memset |
|---|---:|---:|
| union device busy | 6.734 ms/frame (51.5%) | **7.659 ms/frame (58.6%)** |
| traced device idle | 6.332 | **5.412** |
| gaps > 200 us | 1.06/frame, 4.195 ms | **1.89/frame, 2.430 ms** |
| idle in front of `main` | 4.989 ms | **0.162 ms** |

**The 4.99 ms in front of `main` was memcpy time that I had left out of the
union. I withdraw it, and every sentence that rested on it.** The corrected
picture agrees with `PERFORMANCE.md` §5.2 by an independent route (2.20
gaps/frame over 200 us; inter-submit stall median 2.453 ms — I get 1.89 and
2.430), which the wrong version did not. That agreement is the check I should
have run before publishing the first version.

Two further notes on the absolutes, so they are not over-trusted in either
direction:

* the untraced 3.70 ms/frame of device idle is not a measurement at HEAD either
  — it is `frame x 28%`, and the 28% is inherited from the iteration-28 profile
  at a 15.99 ms frame (`PERFORMANCE.md` §5.2 states 9.51 and 3.70 in the same
  row as "72%"). So the comparison is 5.41 traced against 3.70 derived. **Both
  are soft. Use the ratios**, which is what every conclusion here does.
* `main` is **every NIR-generated shader, vertex and fragment**, not the
  fragment shaders. §5.1 splits it into VS main 221.5, FS abuf 22.3, FS direct
  74.5 launches/frame. I called it "fragment shaders" in §0 of revision 1; that
  is corrected above and it matters for §7.3.

**What survives untouched:** the chain is **80.8% gap-free**, its gap is
**0.30 us per launch**, and the mergeable population is not where the idle is.
Also unchanged, because they were computed on the same union either way: every
ΔUB. On the full-operation union the chain gives **1.266 (f=4), 1.603 (f=8),
2.044 (infinite)** against 1.298 / 1.648 / 2.106 on kernels alone — a 2–3%
difference that moves no conclusion.

### 7.2 The stall is structural, and it caps the (b) axis a third time

The census's finding — **device work is issued only from inside
`vkQueueSubmit`, on the application thread; recording touches CUDA not at all** —
explains the whole shape of §0 and closes the last route by which launch width
could have paid big:

* the 2.430 ms/frame in 1.89 gaps is the replayer decoding and re-recording,
  with the previous frame drained and nothing queued. **Launch width changes
  neither when the host enters the submit nor how long it takes to get there.**
* the chain's 80.8% zero-gap figure is the same fact seen inside a burst.
* therefore (b) can only shorten the *submit*, never the *record*. The 2.047 us
  rate limit, even if it is entirely host CPU, applies to a window that is at
  most the 3.478 ms of issue time and pays only where the device is idle
  *within* it.

**Third independent cap on (b), after the in-situ price and the zero-gap
finding. I now regard the 2.047 us arm of §2.2 as an upper bound that the
driver almost certainly does not reach**, and the central launch half of the
legal merge as **0.33–0.55 ms**.

### 7.3 A cross-check nobody has run: where is width actually free?

The shape curve says width is free below about one block per SM. I sorted every
launch in the trace by grid size:

| grid (blocks) | launches/frame | kernel ms/frame |
|---|---:|---:|
| 1 | **241.0** | **2.028** |
| 2–8 | 125.2 | 1.624 |
| 9–64 | 79.6 | 1.172 |
| 65–170 | 27.4 | 0.360 |
| 171–512 | 180.2 | 2.366 |
| 513–2048 | 217.4 | 2.393 |
| > 2048 | 84.0 | 1.752 |

**473 of 955 launches per frame (50%) run 170 blocks or fewer, and hold 5.18 of
12.0 ms/frame of kernel time. 241 launches per frame run a single block.** That
is the population the shape curve says is free to widen.

And here is the discipline of F6 applied to it: **making every one of those
launches infinitely fast removes 1.718 ms/frame of union busy.** At f=4 it is
1.194, at f=8 it is 1.437. So the *entire* width opportunity in the driver,
classified by grid rather than by class, is **1.2–1.7 ms/frame** — which lands
on top of the class-based estimate for the chain (1.27–1.60) and is the second
independent derivation of the same bound. (The grid classification undercounts
`stage3_abuf`, whose 512-block grid carries 12 items; the class-based figure
catches that, and the two agreeing is the point.)

**This also prices the opaque row port.** Its target, `FS main (direct)`, is the
**gridX = 4096** population: 45 launches/frame at a median 14.24 us, 1.121
ms/frame. **4096 blocks is 24 waves on 170 SMs, so width is not free there and
merging those launches buys no device time.** Sizing it with the in-situ launch
price, as instructed, is not a conservative choice — it is the only correct one.

### 7.4 The two new leads, sized

**Object-destruction drains — 0.22 ms/frame, and the best value-to-risk in the
driver.** All four numbers are already measured, at the site, in the census and
the injection sweep:

| | |
|---|---:|
| `vkDeviceWaitIdle` site, blocked | 0.514 ms/frame over 2.24 waits |
| ceiling (`Σ min(issue gap, blocked)`) | **0.500 ms/frame, 97.2% of blocked** |
| measured slope at that site | **+0.44** |
| **ceiling x slope** | **≈ 0.22 ms/frame** |
| corroboration | 2.34 `cuCtxSynchronize`/frame in the smallop census vs 2.24 waits — essentially every context sync in the frame is one of these |

This is the **only** site in the driver that is nearly all ceiling (97.2%) and
**wait-bound in 3,339 of 3,390** rather than gap-bound. It is therefore the one
place where recovering the wait does not depend on the host having something
else to issue — which is precisely the constraint that limits the drain (72%
gap-bound) and that a wide-launch programme makes worse. A deferred-free list
keyed on the submit completion event touches no launch, no kernel and no
schedule. **Bound: 0.22 ms/frame on old** (add-direction slope; symmetry
unproven, as everywhere). Unknown on Crossroads — that capture's census total
ceiling is 1.300 ms/frame and the per-site split was not published.

**Opaque fs-UBO row port — 0.04–0.06 ms/frame.** 74.5 `FS main (direct)`
launches against 74.5 direct raster triples, 1:1, because the opaque group key
requires a byte-identical `memcmp` of the fs-UBO rows; the blended path already
solved this with row concatenation and `row_base`. ~60 launches/frame removed at
the in-situ price of 0.6–1.0 us = **0.036–0.060 ms/frame**. §7.3 shows there is
no device half to add: the grid is 4096 blocks. It is worth doing because it is
a port of an existing mechanism in the same file with no ABI or scheduling
change — not because of its size. **Do not let a per-launch floor of 2.047 us
turn 60 launches into 0.12 ms; that is the exact error this document exists to
prevent.**

### 7.5 The ranking, with the numbers

| # | item | value, ms/frame (old) | how known | blast radius |
|---|---|---:|---|---|
| 1 | **episode drain deferral** | **up to 2.07** | ceiling measured, slope measured (+1.02), mechanism unbuilt | one site |
| 2 | **legal segment merge** (chain, 554 launches) | **1.6–2.1** (range 1.1–2.7) | device half simulated twice, launch half at the in-situ price | architecture |
| 3 | **object-destruction drains** | **0.22** | ceiling measured, slope measured (+0.44) | one list, no launches |
| 4 | PDL levels 1–3 | 0.38–0.43 | measured, built, not merged | moderate; ~73% subsumed by #2 |
| 5 | opaque fs-UBO row port | 0.04–0.06 | launch price only | one file |
| 6 | any further launch-width work | ≤ 1.7 total, and overlapping #2 | simulated (§7.3) | large |

**I agree with the census's ranking on #1 and on width being last, and I differ
on the middle.** The census puts destruction drains second and the merge fourth.
By *value* the merge is second: 1.6–2.1 against 0.22, a factor of seven. By
*value per unit of risk* the destruction drains win outright and should be built
first — they are 0.22 ms for a deferred-free list, they are the only wait-bound
site in the driver, and they are the one item on this list that is additive with
everything else, because they remove a whole-device drain rather than compete
for the same idle.

**Recommended order, which is neither ranking:**

1. **Destruction drains** — 0.22 ms, days not weeks, additive with everything.
2. **PDL** — 0.38 ms, already built and measured; land it before #4 subsumes it.
3. **Episode drain deferral** — up to 2.07 ms, the largest item, one site.
4. **Opaque row port** — 0.06 ms, free ride while in that file.
5. **Re-size the merge** against a PDL-enabled control, on a driver that has
   stopped ping-ponging. Only then is its device half the whole of its value.

Sum of 1–4, allowing for the drain's 72% gap-bound share and no double counting:
**about 2.3–2.6 ms/frame, 17–20% of the frame**, of which every millisecond has
a measured slope behind it. The merge adds **1.3–1.8 ms on top** (its 1.6–2.1
less PDL's subsumed 0.28) *if* its device factor is 4 or better and *if* its
launch price is at the top of the measured range — and less than that if the
deferral has already hidden the host.

### 7.6 The launch-price sweep, both branches

The injection sweep will report a marginal frame price per launch. Both branches
are already decided here:

* **p ≤ 1.0 us** (the in-situ price is confirmed): the launch half of the merge
  is 0.33–0.55 ms, the merge is 1.55–2.09, it **ties** the drain and ranks
  second, and the width axis stays last. Nothing in §7.5 moves.
* **p ≥ 1.6 us** (the floor is real host CPU on the critical path): the launch
  half is 0.89–1.13 ms, the merge is **2.1–2.7 ms and overtakes the drain**, the
  opaque row port doubles to 0.10–0.12, and #6 in the ranking becomes worth
  re-opening — but only for launches issued *inside* the submit, because §7.2
  says the 2.430 ms record gap is not reachable either way.
* **1.0 < p < 1.6 us**: the two are a tie inside the range, and the tie should be
  broken on blast radius, which favours the drain.

---

## 8. Revision 3 — the price is settled, the device half is not, and the programme closes

### 8.1 The launch price: I accept the instrument, and the reconciliation is mechanical

**1.974 us/launch, CI [1.925, 2.032], R² 0.9968, in situ, gated, ladder run
forwards-backwards-forwards.** That is a far better instrument than either the
back-to-back loop or my union-gap attribution, and it is the first marginal
launch price this project has ever measured directly. I accept it as measured.

**Reconciling it with my 0.30 us of measured device gap.** The proposed
reconciliation — gap attribution measures *device* idle, injection measures
*frame* cost including host time that is in front of no kernel — is half right.
The arithmetic says it cannot be the whole answer:

* host issue is **3.478 ms/frame**, and the census says the driver **touches CUDA
  not at all while recording**. The record phase is the inter-submit stall,
  which my trace measures at **2.43 ms/frame** (§7.1) and `PERFORMANCE.md` §5.2
  at 2.453 ms median.
* that leaves **1.05–1.51 ms/frame** of host issue for every CUDA call the
  driver makes: 1,314 launches, 438 copies, 235 clears. **0.53–0.76 us per
  device operation.**
* **1,314 x 1.974 us = 2.594 ms does not fit in 1.05–1.51 ms.** The price
  therefore cannot be predominantly host CPU.

**So what is it?** The mechanism that fits every measurement at once:

> The GPU front end costs about 2 us to process a launch. **When the previous
> kernel runs for 18 us, that cost is entirely hidden** — which is exactly what
> my gap attribution measures: 0.30 us of exposed gap per chain launch, **zero
> for 80.8% of them**. **When the launches are empty and adjacent, nothing hides
> it** — which is the injection arm (800 null launches into one stream) and the
> back-to-back loop. That is why `1.974 ≈ 2.020 ≈ 2.047`: all three measure the
> same *exposed* front-end cost, and two of them are in the regime where it
> cannot hide.

This also explains why the injection's frame response is real and linear: an
exposed 2 us per injected launch lengthens the driver's own stream, the drain
waits longer, and the drain converts at the measured +1.02. Nothing about the
injection is wrong. It measures the price of **adding** a launch to a queue of
empty ones.

**Therefore 1.974 us is an upper bound on removal for TWO reasons, not one.**
The author's caveat — an injected launch carries no work — and mine: **an
injected launch has no neighbour to hide behind, while 80.8% of the chain's real
launches demonstrably do.** The session's own standing caveat applies with
force here: every slope in this driver is an **add-direction** measurement and
symmetry is not proven.

**What I now believe, and it is a narrower claim than either side started with:**

| | |
|---|---:|
| host CPU per device operation (budget arithmetic, independent) | 0.53–0.76 us |
| price table, measured by removal in situ | 0.6–1.0 us |
| exposed device front end per launch (injection, loop) | ~1.97–2.05 us |
| measured exposure of a real chain launch | **0.30 us** |
| **marginal return on removing one chain launch** | **0.8–1.1 us**, upper bound 1.974 |

**The price table is not refuted; it is re-interpreted.** 0.6–1.0 us is the
host-CPU half of a two-part cost whose second half the driver already hides for
most of its launches. Two independent routes — the removal price and the host
budget — agree on 0.5–1.0 us, and neither can be reconciled with 1.974 us of
host time.

**The one arm that decides it, and it is cheap.** Re-run the injection with the
null launches **spread** — one after each real launch, or into a side stream —
instead of bunched. If the price stays near 1.97 us I am wrong and the exposure
model is dead. If it collapses towards 0.6–1.0 us the model is confirmed and
every launch-removal lead in the project should be quoted at the low end.
Cheaper still: `nsys` the existing N=800 arm and look for ~2 us union gaps in
front of the injected kernels. If they are there, the cost is exposed device
time by direct observation.

**Registered before that run:** I predict the spread arm reads **0.8–1.2
us/launch** and the bunched arm reproduces 1.97. Falsifiers: a spread arm above
1.6 us kills the exposure model; a bunched arm that shows no union gaps in front
of the injected launches kills it too, by the other route.

### 8.2 The device half: I disagree with "close to 1", and I withdraw my own 4–8

The headroom result is decisive about one thing and silent about another.

**Decisive:** width does **not** fix the issue rate. The driver's own 12-wave FS
launch issues 3.64% and its 2-wave writeback issues 2.63%, against 1.25–1.80%
for the narrow raster launches — so widening buys a factor of about 2–3 in issue
rate, not the 59% the synthetic kernel reached. **The warps are resident and
stalled, not absent.** That kills the shape curve's 287x and it kills any
argument that width alone makes this driver's work fast.

**Silent:** every one of those comparisons is between **different kernels**.
That is precisely the confound I used to discount the deep-vs-shallow 4x, and
applied symmetrically it discounts this too. FS `main` at 4096 blocks is not
`cp_rasterize_stage3_abuf` at 12 items; their stall profiles differ (73.3%
barrier in one measured kernel, 41–68% L1TEX scoreboard in the other). **No
controlled measurement of the same work at two widths exists in this project.**

So I **withdraw f = 4 and f = 8**. I decline to replace them with 1, because a
merge that puts 150 items on 150 SMs instead of 12 items on 12 SMs still
finishes 13 items' work in one item's latency, and the shape curve measured that
directly (4.098 / 4.098 / 4.150 us at 1, 12, 100 items). What the headroom result
proves is that the *per-item* latency will not improve — occupancy stays at 2
warps per SM either way — so the gain is concurrency only, and concurrency is
capped by exclusivity.

**The device half, as an undetermined range** (full-operation union, k_dev 0.96):

| chain factor | 1 | 1.5 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| ΔUB ms/frame | 0.000 | 0.481 | 0.752 | 1.074 | 1.266 |
| frame ms/frame | 0.00 | 0.46 | 0.72 | 1.03 | 1.22 |

**Defensible statement: 0 to 1.03 ms, with no central value, until someone runs
the same work at two widths.**

### 8.3 The merge, re-stated

| half | value ms/frame | status |
|---|---:|---|
| launch count, 554 x 0.8–1.1 us | **0.44–0.61** | **measured**, two independent routes; upper bound 1.09 if the exposure model is wrong |
| device, chain f = 1–3 | **0.00–1.03** | **unproven**; the factor has never been measured on the same work |
| **total** | **0.44–1.64** | |

> **I agree with the framing offered: the merge is now a launch-count play. I
> reach it by a different route and land slightly lower — about 0.8–1.3 ms if
> the device factor is 1.5–2, and 0.44–0.61 ms if width buys nothing at all.**
> It does not beat the episode drain (up to 2.07), and at the settled price the
> launch axis needs **1,049 launches removed** to tie the drain against the 554
> that are legally mergeable.

**Two historical cross-checks that the settled price passes**, and they matter
because they are the only in-situ removals on record:

* iteration 28 item 4: 91.1 launches x 1.974 us = **0.180 ms** against a measured
  **0.2208 ms** that also removed 27.4 clears, 49.9 MB of clear traffic and a
  compaction pass over 230,400 blocks. At 1.974 the launches alone are 82% of it
  and the rest is over-credited; at 0.6–1.0 us they are 25–41% of it. **Both
  readings survive**, which is why this cannot settle the price either.
* `PERFORMANCE.md` §4 rule 2 — *"do not forecast a fusion with the bare launch
  price when it also deletes a large clear or a whole pass; that under-counts by
  about a factor of two"* — **is suspect at the settled price**, because the
  launch term alone would then explain most of the observation.

### 8.4 The stall, as a lead: the evidence supports opening it

**Yes, open it, and it is the only axis this session has not touched.** The case
is already assembled from measurements that exist: `sm__cycles_active.avg` at
16–22% while `.max` is 94–99%; issue 3% at iteration 1 and 3–4% now, across a
46% frame reduction and every accepted change in `PERFORMANCE.md` §3; DRAM 1%/1%
and L2 hit 5.1% against L1 92.5%; 73.3% of stall cycles at the CTA barrier in
one kernel and 41–68% on the L1TEX long scoreboard in the other; and now the
headroom result showing 12-wave launches that still issue at 3.64%. **A driver
whose warps are resident and stalled has a latency problem inside its kernels,
and no amount of width, launch merging or scheduling will touch it.** What would
size it, in order: (1) `ncu` on `cp_rasterize_stage3_abuf` and
`cp_clip_rast_fused` — 3.7 ms/frame between them — with the full
`smsp__average_warps_issue_stalled_*` breakdown **and**
`l1tex__average_t_sectors_per_request`, because an uncoalesced access is the
single most likely cause of a long-scoreboard stall at 1% DRAM; (2) convert any
candidate fix through **exclusivity, not kernel time** — the chain at f=2 is
0.752 ms of union busy and at f=infinity 2.044, so even a perfect stall fix is
capped by the same ceiling every other device lead has hit; (3) register the
falsifier first: if sectors-per-request is already at 4 and the stall is pure
dependent-load latency, the fix is a data-layout change and the lead is a
quarter's work, not a week's.

### 8.5 Final ranking, at the settled price

| # | item | ms/frame (old) | evidence |
|---|---|---:|---|
| 1 | **episode drain deferral** | **up to 2.07** | ceiling and slope both measured; mechanism unbuilt |
| 2 | **kernel stall / memory latency** | **0 to 2.04** (chain union-busy cap) | diagnosis strong, size unknown — the only untouched axis |
| 3 | **legal segment merge** | **0.44–1.64**, central ~1.0 | launch half measured, device half unproven |
| 4 | **object-destruction drains** | **0.22** | ceiling 0.500 x slope 0.44, both measured; only wait-bound site |
| 5 | PDL levels 1–3 | 0.38–0.43 | measured and built; ~73% subsumed by #3 |
| 6 | opaque fs-UBO row port | **0.12** at the settled price, 0.04–0.06 at the price table | launch count only; grid 4096, no device half |
| 7 | further launch-width work | ≤ 1.72 total and overlapping #3 | simulated |

**Recommended order is still not the value order:** destruction drains (0.22 ms,
days, additive with everything), PDL (built and measured, land it before #3
subsumes it), drain deferral (largest item, one site), row port (0.12 ms, free
ride), then the stall diagnosis — and the merge last, re-sized against a
PDL-enabled control, because two thirds of its remaining value now rests on a
device factor nobody has measured.

### 8.6 One sentence for the top of the handoff

> **The wide-launch architecture does not survive — the driver's own 12-wave
> launches issue at 3.64% just like its 0.3-wave ones, so its warps are resident
> and stalled rather than absent, and width cannot be the lever — but the work
> that killed it produced the driver's first in-situ marginal launch price,
> 1.974 us/launch, which is an upper bound that re-prices every launch-removal
> lead in the project's history and leaves the episode drain, at up to 2.07
> ms/frame, still the largest open item.**

One qualification I will not drop from that sentence if it is quoted:
**1.974 us is the price of adding an exposed launch, not of removing a hidden
one**, and 80.8% of the chain's launches are hidden. Until the spread-injection
arm runs, every launch-removal lead should be quoted as a **range, 0.8 to 1.97
us**, and never as a point.
