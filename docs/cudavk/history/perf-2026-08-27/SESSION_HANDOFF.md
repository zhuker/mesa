# cudavk performance session, 2026-08-26 — consolidated handoff

Written for someone who was not here. Nothing in it needs a message thread to
reconstruct. Every number carries its instrument and the session it came from,
because this session produced at least four figures that were right in one
instrument and wrong when compared across sessions.

Base commit for everything below: **`e2fea470d04`** on `cudapipe-vk-native`,
which is still the branch tip. Reference machine is `CLAUDE.md` §6 (RTX 5090,
driver 580.173.02, CUDA 12.8).

Frame convention throughout: **one frame is two `vkQueueSubmit` events**,
`median(diff(submit_ts[::2])[50:])`. A number taken any other way is not
comparable and is marked where it appears.

---

## 1. The baseline moved, and the docs did not

| | `PERFORMANCE.md` §1 | measured at HEAD | delta |
|---|---:|---:|---:|
| old capture | 15.7514 | **13.1626** | **−2.5888 (−16.4%)** |
| Crossroads | 5.8982 | **5.8230** | −0.0752 (−1.3%) |

Instrument: three full replays per capture, one session, GPU verified idle,
one stdout hash per capture across all six runs, full submit counts (3,022 and
2,994). `reprofile_baseline.md` A.1.

**Cause, attributed completely.** `20611f5b131` ("fan opaque episode segments
out by default") landed after the documentation was written. It took
**2.65 ms/frame out of the episode drain and nothing else**: the drain went
8.639 → 5.989 ms/frame at an *unchanged* 9.88 waits, while peel checks moved
−0.008, segment counters −0.016 and descriptor uploads +0.001 — all three
inside their own noise. Frame moved −2.555; drain moved −2.650. A revert arm
(`CUDAVK_NO_OPAQUE_STREAMS=1`) reproduces the documented 15.7672.
`reprofile_stats.md` B.1.

**Anyone quoting 15.75 is quoting a pre-2026-08-26 driver**, and every §5 and
§6 figure derived from it inherits the error.

**Session spreads, which are the noise floor any lead must clear:** three run
medians span **0.119 ms on old** and **0.034 ms on Crossroads**. Use these
before designing anything.

---

## 2. What is measured and ready to land: programmatic dependent launch

**PDL is not merged.** It is branch `pdl-prototype`, `b9766f720a4`, worktree
`/tmp/pdl-tree`, four commits on `e2fea470d04`. `git log` on the main branch
shows nothing. `pdl_landing.md` is the complete record.

| capture | control (level 0) | candidate (level 3) | delta | p |
|---|---:|---:|---:|---:|
| old | 13.2032 | **12.7690** | **+0.4342 ms (+3.29%)** | 0.0011 |
| Crossroads | 5.8246 | **5.6948** | **+0.1297 ms (+2.23%)** | 0.0143 |

Instrument: strictly alternating, 6 runs per arm on old and 4 on Crossroads,
**one session, one binary**, all 20 full length, one stdout hash per capture
across both arms. 600-frame sample sweep 15.67 → 15.44 ms with **not one
sample regressed**; correctness gate unchanged.

**Three caveats that must travel with the number.**

1. **The control is 13.2032, not 13.1626.** The decisive runs predate the
   reorder gate, so that binary's level-0 arm carried one moved
   `cuMemsetD32Async`. The 0.04 ms is inside the spread and the comparison is
   control-to-candidate *within one session*, which is the valid one. Do not
   subtract 12.7690 from 13.1626.
2. **An earlier "−0.5173 ms" is wrong and has been removed everywhere.** It
   subtracted a control from one session from a candidate in another — exactly
   what `WORKFLOW` §4.2 warns about — and was inflated by 0.09 ms. If you find
   it in any document, that document is stale.
3. **Read the converted-link count before believing any PDL timing** — a run
   that converts nothing looks exactly like a run without the flag.
   `CUDAVK_PLAN_STATS`: 550.87 links/frame on old, 125.28 on Crossroads at
   level 3.

**The per-link prices this produced**, which belong in the driver's price table:
no preamble (the gap alone) **0.44 µs** old / 0.84 µs Crossroads; one
independent global load ahead of the wait **0.78 / 1.32 µs**; a whole clear
hoisted ahead of the wait **3.02 / 2.14 µs**. These are **net** of the
`cuLaunchKernelEx` premium — the level-2 arm made 421.50 more extended launches
per frame and still won.

---

## 3. What closed, with the number that closed it

Nine items. Each row: what was believed, what was measured, what would reopen it.

### 3.1 Clip-rectangle bound on the `bounded` fast path — `bounded_clip.md`

*Believed:* 0.10–0.25 ms/frame from bounding the A-buffer fast path by the clip
rectangle rather than the framebuffer. *Measured:* the clip rectangle **equals
the whole framebuffer in 6,301/6,301 samples on old and 2,535/2,535 on
Crossroads**; admission is **0** under every variant; bound/actual ≈ 3.5e7.
Admission needs `blocks × tris ≤ 131,072`, so at the median triangle counts the
rectangle would have to be 72×18 px on old and 54×2 px on Crossroads. *Also
settled:* `bounded` and `per_draw_rects` are **mutually exclusive** — they share
the `compact_rows` guard — so the per-draw rectangles are unreachable and the
batch-wide one already carries the scissor whenever the path can fire.
*Reopen:* only if (clip blocks × rast triangles) ever falls under 131,072.

### 3.2 `quads == 0` empty drains — `s1d_design.md` §2.3, `probes_p1_p2.md`

*Believed:* 6.9% of old drains and 18.3% of Crossroads drains return `quads==0`,
so the host blocks only to learn the episode was empty. *Measured/argued:* the
drain is taken for `fill_over || quad_over`, not for `quads`; `quads` rides the
same copy for free. The zero is only knowable at that drain, and
`cp_pass_can_retry` forbids running the FS before the overflow answer. An
oracle would save no wait. *Reopen:* only as a consequence of moving the drain,
which §3.3 closes. **There is no four-line version of this.**

### 3.3 S1d — take the episode drain at the scan — `s1d_design.md`

**CLOSURE REVISED. S1d stays closed, on cost and mechanism rather than on value,
and its value was understated by about a factor of nine.** The full revision is
the first section of `s1d_design.md`.

*Believed:* the largest wait in the driver can be moved to near the start of the
chain. *Measured*, by a probe written for the purpose:

```
old    at most 1586.314 ms of 9007.529 ms (17.6%); chain after the scan is 88.5% of device time
cross  at most  246.315 ms of 2698.135 ms  (9.1%); chain after the scan is 71.5%
```

= a **1.05 ms/frame ceiling on old, 0.16 on Crossroads**. `counts→scan` is only
2.3% / 3.6% of the drain, so the drain moves a short distance.

**One sentence of the original closure is retracted:** *"a 1.05 ms ceiling buys
well under 1.05 ms."* That borrowed the 11% from the peel site, and §5 R2′ shows
the 11% does not exist. At the drain's measured conversion of **1.02**, the 1.05
ms ceiling is worth up to **~1.07 ms of frame — 2.4× PDL's entire win.** The
share is small; the quantity is not.

**What keeps it closed, none of which uses a conversion:** the packed form is
dead on the helper-lane architecture (§4 F1) and that one is architectural, not
economic; the `capacity/4` gate refuses **25.3%** of old episodes and is
load-bearing because it caps the over-allocation that killed S0; and two
fail-closed safety checks would have to be demoted. The memory objection is
weaker than first written — the drop-in form **FITS** on old (1,069.3 MB against
7,914.8 held, 0 regrows) where the value is, and **CROSSES** only on Crossroads
where 0.16 ms is at stake. *Reopen:* not on size. **This closes one mechanism at
the drain, not the site — see §6.1, where the drain is now sized at up to 2.07
ms/frame and is the largest open item in the driver.**

### 3.4 Peel predication, and the peel site itself — `peel_measurement.md`, `peel_diagnosis.md`, `peel_closed.md`

*Believed:* 0.20–0.50 ms; the site is 2.751 ms/frame over 1.70 waits at 1.6135
ms each — 20.8% of the frame. *Measured:* the patch worked and **lost 0.1096 ms
on old**, neutral on Crossroads, which runs **zero** peel loops and is therefore
a non-test, not a wash. Three numbers closed the whole site: an **exact**
predictor still pays 1.093 checks/frame against 1.706 (at most 35.9% removable);
the ceiling is **0.376 ms/frame** against 2.756 ms blocked; and **the device was
never idle at a peel check, 0 of 2,577**. *Reopen:* nothing; it is device-paced.

### 3.5 Depth/stencil load-store elision (L6) — `l6_depth_elision.md`

*Believed:* 0.1–0.2 ms/frame, ~4.6 full-screen kernels/frame on Crossroads.
*Measured:* **0.005 ms/frame**, 1.00–1.10 kernels/frame at 4.65 µs, 0.02% of
kernel time, 26th of 26 kernels. Three independent errors: the census counted
API aspect-ops rather than driver kernels (§4 F2); the colour rows are zero
kernels, including the census's own headline item; and the per-kernel price was
never looked up and was already in the tree. Cross-validated by an interception
log giving **1,646 store + 136 load launches**, exactly equal to the census's
depth rows. *Reopen:* **MSAA depth only** — the grid is `…, samples`, so 8× MSAA
is ~37 µs/kernel and ~0.04 ms/frame. Neither capture does it.

### 3.6 C1 — stage-1 setup recomputed on reusing peel passes — `raster_removable_survey.md` §6

*Believed (by me):* stage 1 calls `setup_triangle()` unconditionally and
discards it for every non-small triangle on every reusing pass, and stage 2
already avoids the mirror of this one stage later. *Measured*, by a counter
written with the prediction registered first:

| | old | Crossroads |
|---|---:|---:|
| peel loops | 23,921 | 19,200 |
| ran more than one pass | **344 (1.4%)** | **0 (0.0%)** |
| longest loop | 256 passes | 1 |
| reusing passes | 27,847 | 0 |
| stage-1 invocations on reusing passes | 132,652,576 (19.1% of all) | 0 |
| **discarded setups** | **3,032,607** | **0** |
| **as a share of all direct stage-1 invocations** | **0.44%** | **0.00%** |

Against a **5% line drawn before the run** (5% of the 1.894 ms/frame direct
stage-1 class is 0.095 ms, just under the 0.119 ms session spread), this is
**BELOW**, by more than a factor of ten. *Reopen:* a workload with many-pass
peel loops on large triangles. The mechanism is real; the population is not.

### 3.7 The width axis — `overlap_survey.md`

*Believed:* the A-buffer raster chain could be fanned out the way the opaque one
was. *Found in the code:* **blended segments have always fanned out** —
`cp_pass_append()` sends segment `nsegs % CP_PASS_STREAMS` to `seg_streams[k]`
with no flag, and the eight streams were built for the blended admission test in
the first place. *And the width already suffices:* NCU puts
`cp_rasterize_stage3_abuf` at median **0.15 waves/SM, 0 of 270 launches reaching
1.0**, so **≈6.7 concurrent launches fill the machine against 8 existing side
streams**. Raising `CP_PASS_STREAMS` cannot pay. *Reopen:* nothing; what is left
there is duty cycle, not width.

### 3.8 CUDA graphs, and `interp → fragment shader` as a PDL link

**Graphs stay refuted** (`cuda_leads.md` §4), and the fan-out made it worse:
whole-episode graphs now span eight streams, and the recorded refutation —
4,532 submits with 4,532 unique plan generations and zero repeated recipes — is
unaddressed. Still zero graph code in `src/cudavk`. *Reopen:* only if command
recording changes so plans are reused within a generation.

**`interp → FS` is recommended against** (`pdl_landing.md` §8.2) on PDL's own
evidence: a link is worth what the secondary can execute before its wait, and a
fragment shader's first act is to read its interpolated inputs.

---

## 4. Five architecture facts that kill classes of lead — `driver_facts.md`

These are properties of the driver, not of a capture. Each has already caused an
over-estimate in this session.

**F1. Lanes `4q..4q+3` are the derivative quad.** `cp_quad_derivative()`
(`cp_nir_to_llvm.c:2345-2374`) computes `ddx`/`ddy` with `shfl.sync.bfly` over
`tid & 28`, implicit LOD is expressed through those gradients (`:2379`), and the
driver states the rule at `:3393-3397`: *"A quad is shaded whole so that
derivatives can be taken across it, so a lane the primitive does not cover is
shaded too."* So `total/quads = 3.753` means **6.2% of shade slots are helper
lanes, not waste**, and **any proposal to renumber or pack shade slots is dead
on arrival.** This retracted a claim in `s1d_design.md` §5 that was mine.

**F2. There are exactly two attachment-transfer kernels and both key on the
depth aspect.** `cpvk_cmd.c:1586-1590` derives `.load` and `.store` from
`pDepthAttachment` alone; the stencil attachment is read only for
`stencil_clear`/`stencil_value`; `fb.color` is the colour image's own device
pointer. So **stencil and colour load/store traffic is already elided, by
construction**, and a capture-level `loadOp`/`storeOp` census over-counts this
driver by about four times. (Also recorded there: the driver honours a stencil
`storeOp` only when the depth one is STORE — 1,717 dropped per Crossroads
replay, all independently found dead, but it is an assumption.)

**F3. The host cannot bound the A-buffer's quad count.** `nblocks × triangles`
over-estimates by ~3.5e7 and the clip rectangle is not tighter. **`bound/actual`,
not the wait, is the expensive quantity** — three separate leads have died on it.

**F4. Blended episode segments already fan out** across the eight side streams,
with no flag — see §3.7.

**F5. There is no universal blocked-to-frame conversion.** Measured add-slopes:
episode drain **+1.0207**, peel checks **−0.0262**. The 11% never existed. A
conversion is a property of the site, must be measured there, and is nearly free
to measure — host time can be *added* at a wait for nothing, and the slope is
the same derivative in the add direction. See §5 R2′.

**The rule they share.** Before costing a lead, **map its unit onto the driver's
unit** — API op onto kernel, slot onto lane, bound onto count, and a constant
onto the site it was measured at — then check whether the driver already does
the thing, or cannot. F1–F4 are numbers that looked like waste and were
architecture; **F5 is a number that was real and borrowed from the wrong
site.**

---

## 5. The rules this session produced

**R1. At a synchronisation site where the device is never idle when the host
arrives, removing or deferring the host wait cannot pay.** The blocked time is a
symptom of device work. Two host-side instruments answer it before anything is
built: one `cuStreamQuery` before the sync, and `Σ min(issue burst after the
wait, the wait)`. **CUDA events cannot answer it** — an event pair measures a
span of the stream timeline and idle inside the span is invisible.

**R2 IS RETIRED. There is no universal conversion factor, and the 11% never
existed.** It was inferred from the rejected peel patch, which blocked
+1.01 ms/frame and cost +0.110 ms of frame. Direct injection at that same site
now shows that **adding 2.05 ms/frame of pure host time there costs nothing at
all** (slope −0.0262). So the patch's +0.110 ms was its **mechanism** —
predication issuing further ahead, plus overshoot passes — and never its
blocking. The two conversions actually measured in this driver are **0.00 at the
peel site and 1.02 at the episode drain**, and that range spans everything.

**R2′, the rule that replaces it: a wait's conversion is a property of the site,
must be measured at that site, and cannot be carried between sites.** The cheap
way to measure it is to lengthen the host at that point — a wait cannot be
shortened without a mechanism, but host time can be added for free, and the
slope is the same derivative in the add direction. Scale the injection by
`waits/frame` so each site gets comparable signal.

**What this does and does not overturn.** It does **not** touch any lead closed
on device time or on a count — §3.1, §3.5, §3.6 and §3.7 never used a
conversion. It **strengthens** the peel closure (§3.4): three independent
instruments now say that site is off the critical path. It **invalidates every
sizing done with the 11%**, which is §3.3's headline sentence (revised there)
and the "closed by size" claim in §6.1 (corrected there).

**R3. Map the lead's unit onto the driver's unit before costing it.** F4 above.
Two leads were over-estimated 4× and 20–40× by this alone, and one of our own
instruments divided by the wrong unit (§8).

**R4. Anchor a forecast on the primary kernel's own launches per frame, from the
census — never on a structural model.** PDL's link-count forecast missed by 3.3×
because it was written in *episodes* when the code loops over *draw batches*,
and because it added a second triple for a fallback branch that does not run.
The simple anchor — one triple per batch, two links per triple — fits to 3.5%.

**R5. Never read a predictor's accuracy from a run in which the predictor drives
the schedule.** The peel ring reads 96.6% stable with the predicate on and
**82.0% with it off**; the flag-on figure is self-fulfilling. Reading Q1 from
those runs would have concluded "the ring is excellent, the design is sound" —
the opposite of the truth.

**R6. A control belongs in the same session as its candidate.** The worked
example is ours: an early PDL figure of **−0.5173 ms** was a cross-session
subtraction, inflated by 0.09 ms, and has been removed from every document. The
correct figure is +0.4342 ms, control and candidate alternating in one session
on one binary.

**R7. Register the prediction, with falsifiers, before the run.** It is the
strongest evidence in the record, invisible unless stated, and it caught two of
our own errors: PDL's prediction 4, a 3× scale miss whose falsifier proved the
*model* wrong rather than the constant, and C1's falsifier 2.

**R8. Register several falsifiers, not one.** *A single falsifier is either
falsely reassuring or falsely alarming depending which one you happen to pick,
and you cannot tell which until they disagree.* C1 is the worked example (§7):
one of three falsifiers fired, the conclusion held anyway, and the disagreement
is what revealed that the fired one was measuring population where the question
was work. With only that falsifier a dead lead would have been designed; with
only the surviving one the right answer would have rested on a reason that could
not be defended. This is the most transferable rule in the document and it has
nothing to do with this driver.

---

## 6. What is still open, in order

### 6.1 The wait census — RAN. Seven of eight sites closed; the drain survives

`CUDAVK_WAIT_CENSUS=1`, branch `peel-predicate`, both captures,
one run each. Per frame (÷1,511 old, ÷1,497 Crossroads):

| site | waits/frame | blocked ms/frame | **ceiling ms/frame** | % of blocked | ready |
|---|---:|---:|---:|---:|---:|
| **episode drain** | 9.88 | 6.018 | **2.066** | **34.3%** | **0/14,932** |
| peel checks (old only) | 1.71 | 2.744 | 0.301 | 11.0% | 0/2,577 |
| segment counters | 4.17 | 0.967 | 0.253 | 26.1% | 0/6,301 |
| `vkDeviceWaitIdle` | 2.24 | 0.514 | 0.500 | 97.2% | 0/3,391 |
| descriptor upload | 1.00 | 0.012 | 0.012 | 100% | 0/1,510 |
| upload-arena rewind | 0.05 | 0.073 | 0.005 | 6.8% | 0/80 |
| scratch reclaim | 0.04 | 0.091 | 0.004 | 4.1% | 0/61 |
| **TOTAL, old** | **19.09** | **10.419** | **3.140** | **30.1%** | **0 / 28,852** |
| **TOTAL, Crossroads** | 9.85 | 2.947 | **1.300** | 44.1% | **0 / 14,738** |

**`ready = 0` at every site on both captures — 43,590 waits, not one of them
arriving at an idle stream.** R1 generalises: **no wait in this driver is pure
overhead.** That is the strongest single result of the session and it was a
registered prediction (`wait_census_c1.md` §2: *"any site with a non-zero ready
count would overturn the peel conclusion"*).

**One site is closed; five are UNSIZED, and this corrects an earlier claim in
this document.** An earlier draft read "six sites are closed by size, at the
peel-measured 11% conversion". **That reasoning is void** — R2′ in §5 — because
the 11% does not exist and the two measured conversions are 0.00 and 1.02.

* **Peel is closed**, and by three independent instruments rather than by a
  conversion: ceiling 0.301 ms/frame, `ready` 0 of 2,577, and a measured
  add-slope of **−0.0262**. Adding 2 ms/frame of host time there is free.
* **The other five are unsized.** Their ceilings — segment counters 0.253,
  `vkDeviceWaitIdle` 0.500, descriptor upload 0.012, arena rewind 0.005,
  scratch reclaim 0.004 ms/frame — are worth those numbers times a conversion
  nobody has measured. The three under 0.02 are negligible at any conversion
  and can be treated as closed. **`vkDeviceWaitIdle` at 0.500 and the segment
  counters at 0.253 cannot**, and if either converts near 1.0 it is PDL-sized.
* A structural guess, flagged as a guess: the segment counters sit *inside* the
  episode chain with more chain behind them, like a peel check, and should
  behave like one; `vkDeviceWaitIdle` sits at the frame boundary next to the
  2.453 ms inter-submit stall and is the one to measure next. **Two more runs of
  `CUDAVK_WAIT_SPIN_SITE` would settle both**, and until they do, neither
  should be described as closed.

**The episode drain survives, and is the only thing left.** Ceiling **2.066
ms/frame on old, 0.757 on Crossroads**, which is 66% of the whole driver's
ceiling. Two things bound it before anyone gets excited:

* it is **72% gap-bound on old and 75% on Crossroads** (10,819 of 14,932 waits)
  — the host runs out of work before the wait ends, so most of the ceiling is
  limited by how little there is to issue, not by the wait;
* the whole value is `ceiling × conversion`, and **the conversion is the one
  number nobody has for this site.** At the peel-measured 11% it is 0.227
  ms/frame; at 30% it is 0.62.

**So the drain moves from "unknown" to "open, sized, and awaiting one number",
and that number is being measured now** by a delay-injection probe: it adds host
block time at a site and reads the frame's response, giving a **slope** =
Δframe / Δblocked. It runs the **peel site as its control**, because that site's
conversion is already known by an independent route (≈0.11). **Nothing should be
designed at this site until it reports.**

Two prediction misses worth carrying: the segment counters came in at 0.253
ms/frame against a predicted <0.1 and **gap-bound where wait-bound was
predicted**, and `vkDeviceWaitIdle` blocked 0.514 ms/frame against a predicted
>1.

### 6.1.1 The ending that happened: the drain is SIZED at up to 2.07 ms/frame

The delay-injection probe reported. **None of the three endings pre-written for
this subsection fits**; they are scored in §7 and the reason they missed is the
finding. `CUDAVK_WAIT_SPIN_US` busy-waits a fixed interval immediately after a
named wait returns, outside that wait's own timing, so it lengthens host time at
a site without changing any device work. Slope = Δframe / Δ(injected ms/frame).

| injected ms/frame | 0 | 0.247 | 0.494 | 0.988 | 1.976 |
|---|---:|---:|---:|---:|---:|
| **drain**, frame ms | 13.2038 | 13.4375 | 13.7393 | 14.2005 | **15.2223** |

| injected ms/frame | 0 | 0.256 | 0.512 | 1.023 | 2.046 |
|---|---:|---:|---:|---:|---:|
| **peel**, frame ms | 13.1656 | 13.1880 | 13.1722 | 13.1473 | **13.1259** |

**Drain slope +1.0207**, linear across the whole sweep, residuals under 0.031.
**Peel slope −0.0262**, flat. All 20 runs 3,022 submits, one stdout hash across
all 20.

**The probe validated itself, which is why the drain number can be believed.**
It returned ≈0 at the site where three independent instruments already say there
is slack (ceiling 0.301 ms/frame, `ready` 0 of 2,577, and a patch that removed
19% of the checks and did not help), and ≈1 at the site where the mechanism says
there cannot be any — a drain empties the stream by definition, so the device is
idle immediately after one and any host delay lands straight on the critical
path. A probe that gave the same answer at both would have measured itself.

**The mean/median caveat is resolved rather than carried.** The injected
quantity is a mean (total spun ÷ frames) and the response is the skip-50 median.
Had the injection been concentrated, the median would have moved less than the
mean and the slope would have come in below 1. It came in at 1.02, so **the
transfer is at par and the census's mean-based 2.066 ms/frame ceiling needs no
discount to a median-equivalent** — the earlier worry that it should be read as
~1.50 is retired.

**So the episode drain is sized, not bounded: up to about 2.07 ms/frame of
median frame time on old (15.7% of the frame) and 0.77 on Crossroads.** For
scale, PDL's entire measured win is 0.434 ms. This is the largest open item in
the driver by a factor of nearly five, and it is the only one left.

**The one caveat the measurer refuses to drop, and it should not be dropped.**
This measures the **add** direction. Symmetry is *demonstrated* at peel, where
both directions read ≈0, but it is not *proven* at the drain: adding host time
after a drain costs a frame because the device is empty, whereas recovering wait
time does not remove the device work the wait was waiting for. **A deferral
mechanism still has to be built and measured before 2.07 becomes a win.** What
has changed is that it is now worth building one — and every previously refused
mechanism at this site was refused against a value nobody had measured.

### 6.2 PDL level 4 — built, queued, unmeasured

`pdl_landing.md` §8.1.
Registered predictions: take ≈155/frame on old, ≈33 on Crossroads; +0.53% and
+0.49%. **The negative control matters more than the result:**
`CUDAVK_NO_FETCH_FOLD=1` at level 4 should collapse the take count towards zero,
because that revert puts the queue-counter clear back in front of every stage 1.
If it does not, the predecessor check is not doing what it claims.

### 6.3 The launch-rate framing

PDL bought 0.43 ms by starting the same kernels
sooner; the driver issues ~1,314 launches/frame on old and 346 on Crossroads.
Whether launch rate rather than launch count is the right axis is the one
structural question this session opened and did not answer.

### 6.4 The CUDA-side items nobody ran — `cuda_leads.md`

* **`CUDA_CACHE_MAXSIZE` at 4 GiB. Free, and still unset.** Verified today:
  the cache is **1,074,252,980 bytes over a 1,073,741,824-byte default** with
  the file count *down* from 12,286 to 9,957 — eviction in progress, and every
  evicted entry re-pays 0.3–0.5 s of `ptxas`. Zero frame time; it protects the
  existing 2.5× cold/warm factor. One environment variable.
* **`CU_JIT_SPLIT_COMPILE` (option 34).** 536 → 168 ms cold JIT out of tree.
  `cp_kernels.c:307` loads the four `.cu` modules with plain `cuModuleLoadData`
  and **zero options**. Cold-start only; 0 ms of frame.
* **`.pragma "enable_smem_spilling"`.** The note claimed PTX 9.0; that is
  **wrong** — `ptxas` 12.8 already implements it and its error is the pragma
  being recognised at the wrong scope. Needs explicit launch bounds, and
  `cuLinkAddData` rejects it per-function, so the NIR path needs the sampler
  inlined first.

### 6.5 Genuinely untouched in `leads.md`

 L10 (re-A/B `CUDAVK_INLINE_FS` at
HEAD, stale since before the hardware texture cache became the default), L13
(re-sweep the rasterizer thresholds — the points axis is output-neutral by
construction, the triangle axis is not), L14 (`cuMemDiscardAndPrefetchBatchAsync`),
L16 (`pbribl`'s +0.03 ms), L17 (stage-3 tile interior — gated by its own author
on a census never run).

**Closed for the programme, and worth saying once:** removal is exhausted in the
raster chain. The top six kernel classes are irreducible work; per-tile setup is
already cached above four tiles, the A-buffer is not rasterized twice, the
per-pass counter clear is already skipped on reuse, and stage-3 grid sizing was
measured and lost.

**The session in one sentence:** *removal is exhausted in the raster chain, what
remains is overlap — plus exactly one wait, the episode drain, now measured at a
conversion of 1.02 and sized at up to 2.07 ms/frame, which makes it the largest
open item in the driver and the only one worth building a mechanism for.*

---

## 7. Predictions that were registered before their measurement

Marked because R7 says they should be, and because the misses are the
informative half.

| prediction | registered in | outcome |
|---|---|---|
| tier 1 under 2% on both | `pdl_prototype.md` | **PASS** 1.08% / 0.97% |
| tier-1 links track A-buffer episodes, not opaque segments | `pdl_prototype.md` | **PASS decisively**, ratio 1.77 while opaque segments differ 48.5 vs 4.0 |
| added level-2 links ≈128 old / 27 Crossroads | `pdl_tier2.md` | **FAIL, 3.3× low** — wrong unit (episodes, not batches) and a branch that does not run |
| L3−L2 between +0.1% and +0.6% | `pdl_tier2.md` | **PASS** +0.50% / +0.39% |
| exact peel predictor still pays ≥1.0 check/frame | `peel_predicate.md` | **PASS** 1.093 |
| device idle at <5% of peel checks | `peel_predicate.md` | **PASS** — 0 of 2,577, and it is why the lead is closed |
| peel deferral ceiling ≤0.35 ms/frame | `peel_predicate.md` | **MARGINAL FAIL** 0.376, inside the stated 0.5 falsifier |
| wait census: `ready = 0` at every stream site | `wait_census_c1.md` §2 | **PASS, and stronger** — 0 at all eight, both captures, 43,590 waits |
| wait census: drain ceiling 0.8–2.0 ms/frame (15–35%) | same | **PASS on %, marginal over on ms** — 2.066 ms/frame, 34.3% |
| wait census: peel reproduces ≈0.376 ms/frame (its own control) | same | **PASS** 0.301, inside the stated 0.30–0.45 band |
| wait census: segment counters <0.1 ms/frame, wait-bound | same | **FAIL twice** — 0.253 ms/frame and gap-bound |
| C1 discarded setups <5%, likely <1% | `raster_removable_survey.md` §6.4 | **PASS** 0.44% / 0.00% |
| C1: >half of peel loops run exactly one pass | same | **PASS** 98.6% / 100% |
| C1: reusing passes <10/frame | same | **FAIL** 16.9/frame on old |
| conversion probe: peel add-slope 0.5–1.0, i.e. disagrees with 11% | `conversion_probe.md` §3 | **FAIL, and decisively** — −0.0262. It disagreed with the 11% in the other direction: peel converts at *zero* |
| conversion probe: drain add-slope 0.6–1.0 | same | **PASS, at the top** — +1.0207 |
| conversion probe: slope constant across the sweep | same | **PASS** — residuals under 0.031 over 0→1.976 ms/frame |
| handoff §6.1.1's three pre-written endings | this document, previous revision | **ALL THREE MISSED.** (b) needed the peel control in 0.05–0.20 and (c) needed 0.5–1.0; it came back at −0.026, below both. The conclusion (b) reaches — the drain is sized, not bounded — is right, but by a validation argument neither ending contained: the probe returned ≈0 where three instruments say there is slack and ≈1 where a drain empties the stream by construction. **Pre-writing endings bounded the judgement; it did not replace it.** |

**The C1 falsifier that fired was the wrong falsifier, and that is the lesson.**
Reusing passes are 16.9/frame and stage-1 invocations on them are 19.1% of all
of them — yet discarded setups are 0.44%, because the 344 many-pass loops are
**small-triangle** draws whose setups stage 1 uses. I had conflated "many
reusing passes" with "much discarded work". The operative falsifier — the 5%
line on discarded setups — was the right one and it held.

**R8, which follows from it: register SEVERAL falsifiers, not one.** With only
the population falsifier I would have concluded C1 was real and designed it;
with only the work falsifier I would have been right for a reason I could not
have defended. A single falsifier is either falsely reassuring or falsely
alarming depending which one you happen to pick, and you cannot tell which
until they disagree. Three of the ten predictions in this table only became
informative because a neighbouring one contradicted them.

---

## 8. Instrument caveats you must carry

* **`CUDAVK_DEBUG_TIME` and `CUDAVK_ABUFFER_TIMING` are invalid at the current
  default.** Since work spread over eight streams, a CUDA event pair on one
  stream no longer bounds an episode. Host timers (`CUDAVK_PLAN_STATS`) and
  counters (`CUDAVK_UPLOAD_STATS`) are unaffected.
* **A gate flag in a timed run inverts the result.** `CUDAVK_ABUF_FUSE_CHECK`,
  `CUDAVK_ABUFFER_VERIFY` and `CUDAVK_ABUFFER_TIMING` synchronise; the last two
  also disable the `bounded` arm outright.
* **`CUDAVK_NO_BINCACHE` also turns off `cache_queues`** (`cp_renderer.c:6171`),
  which removes the reusing passes `CUDAVK_PEEL_SETUP_CENSUS` counts.
* **The C1 census divides by `cp_depth_attachment_store` calls, and that is
  1.09/frame on old, not 1.00.** Its `/frame` columns are therefore about 9% low
  on old (34.3 passes/frame, not 31.45; 2,008 discarded setups/frame, not
  1,842). Crossroads is exact, 1,496 against 1,497. **The 0.44% verdict is a
  ratio of two totals and is unaffected.**

  **This is F4 catching an error inside this session, and it is worth reading
  as one.** The "once per frame" convention is `iter25-profile`'s, and it is
  true only where every frame's depth `storeOp` is STORE — which is **F2**,
  three sections above, written by the same author two tasks earlier. The
  instrument trusted the convention without applying the fact that qualifies
  it, and old has 1,646 depth stores against ~1,510 frames because some frames
  store twice. So the finding came back to bite the probe that relied on it,
  and it was caught by re-reading our own document rather than by a
  measurement. **The unit was "frames"; the driver's unit was "depth stores".**
* **Probe runs are not timed runs.** P4 records ~30 events a frame; the wait
  census adds a `cuStreamQuery` per wait. No frame median from any of them
  appears in any document.

---

## 9. Documents to change

**Already done by the measurer** — do not redo: `PERFORMANCE.md` §1 headline and
§4 price table (PDL's three per-link prices), and the §5 staleness note.

**Still to do:**

1. `PERFORMANCE.md` §5 — the whole section is anchored to 15.99 ms at
   `ba8891878df`. Shares and launch counts survive; absolute idle figures and
   the frame attribution do not.
2. `PERFORMANCE.md` §6 — items 1, 2 and 4 are all closed by §3 above. Item 3
   (`cp_fs_writeback` fusion) is unchanged and still parked on its correctness
   failure.
3. `PERFORMANCE.md` §7 — add the width paragraph from `overlap_survey.md` §C3,
   and add "removal in the raster chain is exhausted" with §3.6's number.
4. `DEAD_ENDS.md` — new entries for the clip-rectangle bound, S1d, the peel
   site, L6 and C1, each with its killing number and its reopen condition.
5. `FLAGS.md` is generated; run `src/cudavk/tests/cp_debug_doc.py` after any
   registry change and `cp_no_getenv.py` after any flag work.
6. `CLAUDE.md` §6 needs no change, but every number here is from that machine
   and the pre-2026-08-25 figures elsewhere are not.

---

## 10. Where the code is

| branch | worktree | what | state |
|---|---|---|---|
| `pdl-prototype` | `/tmp/pdl-tree` | PDL levels 1–4, `CUDAVK_NO_PDL` / `CUDAVK_PDL` | **measured, decisive, unmerged** |
| `peel-predicate` | `/tmp/peel-tree` | rejected predication patch + `CUDAVK_PEEL_CENSUS` + `CUDAVK_WAIT_CENSUS` | patch off by default; census is the reusable part |
| `bounded-clip` | `/tmp/bnd-tree` | `CUDAVK_ABUF_CLIP_BOUND`, `CUDAVK_ABUF_BOUND_ANY_TRIS`, `CUDAVK_EPISODE_SIZE_STATS`, `CUDAVK_EPISODE_WAIT_SPLIT`, `CUDAVK_PEEL_SETUP_CENSUS` | all default off; the two bound flags are inert by measurement and should never be set |

Every branch is inert with its flags off. On `bounded-clip` that was checked at
machine-code level: with all five flags folded to `false`, `cp_renderer.c.o`
differs from `e2fea470d04` by **one scheduled `mov` and nop padding**, over 172
symbols with none added and none removed.


---

## 11. The tiling audit and the device-level ceiling — `tiling_ncu.md`, `stage3_imbalance.md`

Added after §10 by a later task in the same session, on the same base commit
`e2fea470d04` and the same machine. Nothing above was changed. Both documents
are in `/tmp/perf-audit/`; raw output is under `/tmp/perf-audit/tiling/`. The
working tree was not touched: one instrumentation fix and one probe were built
in a throw-away copy at `/tmp/perf-audit/tiling/mesa`.

### 11.1 The opaque sort-middle tiling prototype is refused, and its recorded reason was wrong

`docs/cudavk/notes/OPAQUE_TILING_PROTOTYPE.md` names Nsight Compute on
`cp_opaque_tile_raster` as the required next step. It had never been run. It was
run, on `multithreading`, the sample the document says amplifies the failure.

**The document's explanation is dead in both halves.** It blames a
`references × tile area` coverage loop and register/divergence pressure. Measured:
**compute throughput 2.53%, DRAM 0.23%, branch efficiency 94.32%**. Instead, per-SM
counters, single pass, three consecutive launches agreeing to 0.2%:
`gpc__cycles_elapsed.max` 110.8 M, **`sm__cycles_active.max` 110.5 M = 99.75% of
elapsed**, avg 22%, **min 0.10%**. One block sets the kernel's duration; 73.3% of
stall cycles are at the CTA barrier waiting for thread 0's per-reference
`setup_triangle()` and its copy of `cp_rasterize_args` into shared memory.

**The census says the triangles are 1.7 pixels across.** With the reference count
per tile dumped (see §11.3), total references barely move with tile size —
1,190,127 at 16 px, 1,074,095 at 32, 1,027,161 at 64 — which solves to a mean
triangle edge of about 1.7 px and 1.07 references per triangle. The hot tile at
pixel (704, 352) holds **26,297 distinct triangles** and resolves 563 visible
pixels. **That kills successor direction 3** (keep the classic path for small
triangles: there is no large-triangle population to split off) and demotes 1, 2
and 6 to work-amount fixes that do not touch the duration. Directions 4 and 5 are
required, and a seventh the list does not contain — per-tile occlusion — is the
only thing that shortens a list which is depth complexity.

**The model, built on one workload and tested on another.** Duration = longest
tile list × ~3,409 cycles per reference. Cross-check A: 1.07 M refs × 4,213
(profiled) cycles over 170 SMs = 26.6 M against a measured `sm__cycles_active.avg`
of 24.5 M, 9%. Cross-check B: applied to the old capture's 10,033 measured
per-episode longest lists it predicts **+12.8 ms/frame** against the recorded
2026-08-18 regression of **+10.38 ms/frame**, a workload it was not fitted to.

**The refusal, with the number.** A v2 that fixed both required directions would
cost 0.15–0.6 ms/frame of tile raster plus ~0.13 ms of binning and could displace
about 2.87 ms/frame of the direct raster chain: **a ceiling of 2.1–2.6 ms/frame of
kernel time**. The tiled arm must give up the opaque stream fan-out, which
`7f38d2a9b65` refuses under the flag and which §1 of this document banks at
**+2.73 ms/frame**. **The lead's whole best case is smaller than the mechanism it
has to surrender.** Keep `CUDAVK_TILED_OPAQUE` off; do not design a v2.

### 11.2 F6. A kernel class's share of kernel time is not its share of the frame — the conversion is bounded by exclusivity

This belongs with §4's architecture facts and is the generalisable result of the
session.

`cp_rasterize_stage3_abuf` is the largest class in `PERFORMANCE.md` §5.1 at
2.484 ms/frame. On a representative window (below), **only 16.7% of its time is
exclusive** — 83.3% already has another kernel running. Shrinking every launch and
recomputing the union of all kernel intervals:

| change | device busy removed |
|---|---:|
| `stage3_abuf` 2× faster | **0.187 ms/frame** |
| `stage3_abuf` 4× faster | 0.288 ms/frame |
| `stage3_abuf` **infinitely** fast | **0.404 ms/frame** |
| whole raster chain 2× faster | 0.763 ms/frame |
| whole raster chain **deleted** | **2.106 ms/frame** |

The raster chain is 6.56 ms/frame of kernel time in that window and 54.7% of
kernel time in §5.1. **Deleting all of it returns 2.1 ms/frame of device busy
time**, and device time is not frame time — §5.2's host is blocked 73.7% and R2′
applies. **Size a kernel-side lead as `class time × exclusive fraction`, then
apply the site's measured slope. Never from the §5.1 share.**

**Why, at device level, without reference to any kernel.** `nsys --gpu-metrics`,
no CUDA tracing, 60,163 samples over the same window: **GR Active 81% (87% over
busy samples), SMs Active 20% (31%), SM Issue 3% (4%), compute warps in flight 5
per cycle, DRAM read and write 1% each.** The engine is occupied four fifths of
the time, a fifth of the SMs have work, and those SMs issue on 3% of their cycles.

**And it has not moved since iteration 1.** The baseline at `5ebc37aa36e`
(`PERF16_ITERATIONS.md` "Baseline", `VK_NATIVE_DECISIONS.md`:481) recorded
**85–96% GR-active, SM issue 3–7%, ~25% SMs active, 5 warps in flight on 170
SMs**, on the old capture, with the same instrument — `nsys --gpu-metrics`, which
is what `cp_profile.sh METRICS=1` runs. The quantities are therefore comparable:
same counters, same tool, same capture, and the figures above are quoted over the
busy window as the older ones were. **GR-active 85–96 → 81–87, SM issue 3–7 → 3–4,
SMs active ~25 → 20–31, warps in flight 5 → 5.** The frame moved 24.53 → 13.16 ms
(−46%) across every accepted change in `PERFORMANCE.md` §3 — the register-cap
trials, the three fusions, the texture cache, the small-operation work, the
A-buffer fusions, the fan-out — and **the machine's utilisation profile is where
iteration 1 found it.** Every millisecond this driver has won was won by issuing
less, or by overlapping what it issues; none of it was won by filling the GPU.

### 11.3 `cp_rasterize_stage3_abuf`: the imbalance that has no rebalance

The one-block signature above is not unique to the prototype. In the shipping
driver, `stage3_abuf` shows median `sm__cycles_active.max` = **93.9% of elapsed**
against a median average SM of **16.4%**. But the cause is the opposite one, and
the tiled kernel's fixes do not apply.

A probe on every 29th launch across a full old-capture replay (**6,669 launches**)
read back the device-built tile queue:

* queue entries: p25 6, **median 12**, p75 32, p90 404, p99 2,413, max 4,434.
* **17.6% of launches have an empty queue** — 512–2,048 blocks start, read a zero
  counter and exit, at a floor of 2.34 µs.
* the grid is `CLAMP(rast_num_triangles * 8, 512, 2048)` — sized from the
  **triangle count**, which is unrelated to the queue stage 2 builds on the
  device. **98.3% of launches have queue ≤ grid**, so each active block gets
  exactly one item and the median launch leaves **500 of 512 blocks idle**.

So "one block sets the duration" is trivially true: the longest block holds **one
item**, and one item is up to a 64×64 tile walked by **64 threads = 2 warps**.
Stall breakdown: **long-scoreboard L1TEX 41–68%, barrier 1.1–6.4%** — the mirror
image of the tiled kernel's 73.3% barrier. The same item costs **32–37 k SM-cycles
when ~10 are in flight and 7.8–9.5 k when 2,500 are**: the unit is not slow, the
machine is empty around it. **Compaction and rebalancing buy nothing here; only
more warps per item, or more items per launch, can.** And by §11.2 the whole
question is worth under 0.4 ms/frame of device time.

### 11.4 `CUDAVK_TILE_CENSUS` has never reported anything

`cp_tile_census_end_pass()` is defined at `cp_renderer.c:7998` and declared at
`cp_renderer.h:653`, and **is called nowhere**. The only other call to
`cp_tile_census_reduce_pass()` is at `cp_renderer.c:7676`, inside
`cp_tile_census_begin()`, and fires only when the framebuffer changes size inside
one bind. So the flag accumulates per-tile counts and never reduces or prints them
**on any fixed-size workload — which is every sample and both captures**. Verified:
16/32/64 on `multithreading` produced no output at all.

**This is a process finding as much as a bug.** The tiling prototype's own
"Implementation, 1. Census" step has therefore never been executed, and the
reference distribution that kills its stated explanation (§11.1) was one line of
code away from the people who abandoned it. **Before trusting a census flag,
check that its reducer is reachable on the workload you are running.** The
one-line fix used to take these numbers lives only in the `/tmp` copy and is not
proposed for the tree.

### 11.5 R8. An nsys window must be checked against §5.1 before it is read

A capture is not homogeneous, and `nsys --duration` samples a *phase*, not the
capture.

An 8-second window taken from process start on the old capture reported
`cp_rasterize_stage3_abuf` at **2.0 launches/frame** and `cp_rasterize_stage3`
(direct) at **100% exclusive**, with device busy 43%. A second window at
`--delay=12` reported **100.6 launches/frame at a median 18.56 µs** against
§5.1's **134.6 at 18.5 µs**, with device busy 51.5%. The first window is a
peel-heavy phase of the same replay; every number in §11.2 and §11.3 comes from
the second, and the first was discarded rather than reported.

**Check a window's per-frame launch counts against `PERFORMANCE.md` §5.1 before
reading anything out of it.** Two windows of one replay disagreed by 50× on the
launch count of the class under study.

### 11.6 The tenant, and the sampler

A foreign libvpx `vpxenc` batch shared the card in **four bursts** during this
work (t+5..7, t+129..135, t+252..273, t+1083..1113 s of a 2,156-sample 1 Hz log,
`/tmp/perf-audit/tiling/gpu_watch.log`). It runs as a rapid series of short
processes, and a single `nvidia-smi` point check landed in a gap and missed it
earlier in the day. **Use a 1 Hz sampler for the duration of a measurement and
report a positive claim — "N samples, nothing but our own processes" — not "the
card looked idle."** Every number in §11 was taken in a window with zero foreign
samples; the headline tiling result is additionally robust by construction,
because a time-slicing tenant cannot produce a sustained `sm__cycles_active.max`
equal to 99.7% of elapsed on our own kernel.


---

## 12. The wide-launch discovery (appended by the root agent)

Asked: can the driver go wider, given SM ISSUE 3%? Answer: **no as an
architecture, yes as a small launch-count play.** Three agents, all measurements
GPU-exclusive under a 1 Hz sampler with per-run verification.

### 12.1 The marginal launch price, measured three ways

| instrument | price |
|---|---:|
| union idle in front of a real chain launch (nsys, 955,301 kernels) | 0.809 us |
| spread injection, linear region (frame slope, n=12) | **0.782 us**, CI [0.633, 0.949] |
| the driver's own price table (prior work) | 0.6-1.0 us |
| bunched injection - an EXPOSED launch | 1.974 us |
| back-to-back empty-kernel floor | 2.047 us |

The two groups differ because the front end costs ~2 us per launch and is
**hidden when the previous kernel runs long**: 74.9% of real chain kernels start
with ZERO union idle. Not one of 1.21 million bunched injected launches had a
kernel in front of it. So **0.78 us is the price of removing a real launch;
1.97 us is the price of adding an exposed one.** Quote launch-removal leads as a
range and never as the floor.

Consequence: 554 legally mergeable launches x 0.782 us = **0.433 ms/frame**. The
launch axis ranks BEHIND the episode drain. At 1.974 it would have led.

### 12.2 Width does not buy issue rate on this driver

The driver already launches 12.05 waves/SM (FS main, grid 1025-4096) and it
issues **3.64%** - against 1.25-1.80% for the 0.15-0.60 wave raster launches. A
synthetic kernel at 2.45 waves reached 59%. **The warps are resident and
stalled, not absent.** Width moves occupancy 6-14% -> 26-78% and buys back the
launch; it does not move issue. Bigger blocks are not the lever either: at 1000
items all four block sizes land on the same time and 64-thread blocks are
fastest at maximum width.

### 12.3 The stall axis: falsifier fired, recorded large-but-expensive

Registered before looking: if sectors/request is ~4, the stall is not
access-pattern and the fix is a data-layout change worth a quarter, not an
iteration.

| kernel | sectors/req | L1 hit | warps in flight | long_scoreboard |
|---|---:|---:|---:|---:|
| cp_rasterize_stage3 | **1.00** | 95.9% | 20.39 | 70.7% |
| cp_rasterize_stage3_abuf | **1.00** | 47.4% | 4.52 | 56.3% |
| cp_clip_rast_fused | 9.44 | 81.2% | 1.64 | 45.8% |

Both rasterizer stages are at 1.00 - the far side of coalesced, one 32-byte
sector per request. stage3 holds 20.4 warps in flight and still stalls 70.7%, so
this is not starvation and more width cannot hide it. The remedy is **wider
requests** - vector loads, structure-of-arrays - not coalescing. Ceiling ~2.0
ms/frame through exclusivity; cost a quarter. NOT opened.

### 12.4 Two leads found by reading, both cheap

- **Object-destruction drains.** With the texture cache on, destroying an image
  or view drains the whole device TWICE (cpvk_DeviceWaitIdle plus the cache's own
  cuCtxSynchronize; cpvk_image.c:376-379, :580-584, cpvk_texture_cache.c:933-935,
  :797-799). 0.22 ms/frame, and **the only wait-bound site in the driver**
  (3,339 of 3,390), so it is additive with everything and does not depend on the
  host having other work. Best value per unit of risk.
- **The opaque fs-UBO row port.** FS main (direct) is 74.5 launches/frame against
  74.5 raster triples - 1:1, no merging - because the opaque key demands a
  byte-identical memcmp of the UBO rows. The blended path already solved this
  with row concatenation plus row_base. ~0.05-0.12 ms at the settled price.

### 12.5 The inter-submit stall, named

The driver issues device work **only from inside vkQueueSubmit**. Recording
touches CUDA not at all, so while the replayer decodes the next frame the driver
has nothing queued by construction. That is the 2.43-2.45 ms/frame stall, and it
is why idle appears "in front of main": main is the head of every chain, so every
host gap lands there. Width cannot touch it; only letting the host run past the
drain can.

### 12.6 Ranking at the settled price

1. episode drain, up to 2.07 ms (add-direction only, symmetry unproven)
2. kernel stall, up to ~2.0 ms, quarter-scale, not opened
3. merge / launch count, 0.43-1.6 ms
4. destruction drains, 0.22 ms - cheapest, additive, only wait-bound site
5. PDL, 0.38 ms - measured, gated, ready; land before a merge subsumes 73% of it
6. opaque row port, 0.05-0.12 ms


---

## 13. Proving a patch is inert: the taxonomy, and the vacuous result

Found three ways in one session, twice by agents auditing their own work after a
third agent's proof turned out to be vacuous.

**THE FAILURE.** A stray `git checkout` deleted a probe's implementation. The
commit carried THE FLAGS AND NO CODE. The inertness check did not catch it - it
CONFIRMED it, reporting `0 instruction lines differ, 520 of 520 sections
byte-identical`. A perfect result produced by the bug it was meant to detect.

> A folded-off build and a build with the feature MISSING are indistinguishable
> unless something distinguishes them.

This is WORKFLOW 4.4 ("a different hash is usually a run that died") applied to a
BUILD rather than a RUN.

### 13.1 Four different claims, routinely confused

| claim | what it compares | what it proves |
|---|---|---|
| **collateral damage** | files you never edited, base vs shipped | no struct offset moved, nothing unrelated shifted |
| **containment** | only the intended functions changed | the edit is where you say it is |
| **folding** | base vs flags-off | the off arm costs nothing |
| **positive control** | base vs feature LIVE | THE FEATURE EXISTS |

The first three are worthless without the fourth. Report them as one line:
**"0 collateral, +4,361 present, +1,076 removable by folding."**

### 13.2 Match the control arm to the KIND of gate

- **compile-time macro** -> the LIVE arm must differ. Merge patch: PTX 656,339 ->
  756,883, cubin .text 357,888 -> 412,544, 40 -> 43 kernels, and folded-off is
  0 bytes from base.
- **runtime bool** -> the informative arm is the opposite, gates HARDCODED FALSE.
  If the implementation had been deleted, flags-off would EQUAL folded.
  Destroy-defer: base 28,472 / off 32,833 / on 32,507 / folded 31,757.
**The general form, in one line:** *fold the gate to the value that should
DELETE the code, and require the delete to show up.* For a compile-time macro
that is the off arm; for a runtime bool it is a hardcoded false. In both cases
the folded build must be SMALLER than the shipped one by a stated amount.

- **a flag registry row** -> `.text` IS THE WRONG SECTION. A new row is a TABLE
  entry: `cp_debug.c.o` .text delta is 0 by construction, which has exactly the
  shape of the vacuous result. Its evidence is in `.data.rel.ro.local.flags`
  (+88) and `.bss.present_in_env` (+1). **Control the flag in .data and the
  implementation in .text.**

### 13.3 The strongest control is not a size

For a runtime-gated patch, disassemble the flags-off entry point and read its
RELOCATIONS or offsets. Shipped `cpvk_DestroyImage` can reach
`cpvk_device_drain`, `cpvk_drain_census_call`, `cpvk_texture_cache_image_retire`;
the base object has zero matching symbols. In `cp_pass_finish`, `cmpb $0x0,0x11c`
is the flag test and `0x238/0x240/0x248` are the merged kernel handles being
launched; the base has zero references to `0x11c`. **A deleted implementation
cannot produce a relocation.**

### 13.4 Two implementation rules, each measured twice

1. **A field added for measurement goes at the END of its struct.** Mid-struct
   placement cost 1,601 changed displacement lines in one object and 915 in
   another - two agents, two structs, same lesson. **Trap:** `struct cp_kernels`
   is EMBEDDED in `struct cp_device`, so appending to the innermost struct still
   shifts everything after it in the outer one. Necessary, not sufficient: the
   residue becomes *classifiable* (93 displacements at exactly +0x18, 8 `__LINE__`
   immediates, 1 label renumber, 0 unexplained), not zero.
2. **Gate even the parts too cheap to gate.** One unguarded counter increment
   costs nothing to run and **363 bytes to prove**, because it survives folding
   and then has to be explained.

### 13.5 Rebuilding in place invalidates runs

Producing the control arms needed three in-place rebuilds of a build directory a
measurer was using. The agent disclosed the window unprompted and restored the
artifact byte-for-byte. **Any run started inside such a window is discarded and
repeated.** Build variants in a separate directory, or announce the window.


## 14. Item 1 (object-destruction drains) — CLOSED at 3.0%, not 0.22 ms

The wait census measured the vkDeviceWaitIdle SITE at 0.514 ms/frame blocked and
a 0.500 ms ceiling, and the driver was known to reach that site from three of its
own call paths. I ranked it first at 0.22 ms. **That was a mis-attribution: a
site's ceiling is not the ceiling of any one caller of that site.**

Per-call-site census (`CUDAVK_DESTROY_CENSUS`), both captures, exclusivity
positive, and the instrument agreeing with the wait census to 0.00% (1510 + 814
+ 1067 = 3,391 exactly):

| caller | old, ms/frame blocked | us per drain |
|---|---:|---:|
| application's own `vkDeviceWaitIdle` | 0.4847 | 485.0 |
| `destroy_view` | 0.0149 | 21.1 |
| `destroy_image` | 0.0001 | 0.28 |
| **driver share** | **3.00%** | |

Crossroads: driver share **0.24%**. At the site's measured +0.44 slope the entire
driver share is worth **0.0066 ms/frame**.

**COUNT AND COST POINT OPPOSITE WAYS.** By count the driver owns 55.5% of the
site (1,881 of 3,391) and the capture-file arithmetic that predicted it was
exactly right. By blocked time it owns 3.0%, because *the driver's drains arrive
at a device that is already empty*. `destroy_image` is ALREADY a null drain at
0.28 us. Deferring frees cannot recover time that is not being spent.

The mechanism was built (`CUDAVK_DESTROY_DEFER`, retirement queue keyed on the
pending submit's event, inert with the flag off) and **is not measured and not
landed**. It stays on branch `destroy-defer` if the workload ever changes.

Second registered prediction confirmed on the way past: the texture cache's
second sync IS a null sync - 0.19 us per drain, constant on both captures, 1.63%
and 12.49% of the destroy sites' blocked time, under the 20% bar. "Drains twice"
was a call count, never a cost.

**THE RULE:** before ranking a wait site, attribute it BY CALLER. A census that
counts calls will agree with the API trace and still mislead by two orders of
magnitude on cost.


## 15. P0 — the symmetry caveat is retired at the episode drain

Every wait-site number in this session carried "add-direction only, symmetry
unproven". P0 measures the remove direction directly by moving the existing spin
to the OTHER SIDE of the drain's sync: the probe becomes the exact inverse of the
mechanism, at the same site, in the same units.

28 replays, arms strictly alternating on one binary, 931 sampler observations
with none foreign, submits 3,022 and stdout sha 320e993599cc on ALL 28 runs, and
the injection line read on every run (14,932 injections, us wanted vs spun agree
to 0.1%).

**Positive control (spin AFTER the sync) reproduces +1.02**: 0.992 / 1.045 /
1.050 / 1.025 / 1.026 across D = 62..1000 us. The port is right.

**Result (spin BEFORE the sync):**

| D (us/drain) | 62 | 125 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|
| slope | **0.034** | **0.065** | 0.257 | 0.413 | 0.589 |

Registered bar was "slope at D=125 below 0.25". Measured **0.065**, four times
inside it. **Host work moved to just before this drain is absorbed by the wait
rather than added to the frame** — about 1.24 ms/frame of relocatable work for
under 0.08 ms/frame of cost.

**A second, independent instrument agrees.** The driver's own episode-drain
counter is unchanged at every D in the AFTER arm, and in the BEFORE arm it FALLS
by what was injected: 98.9% absorbed at D=62, 6,419 of 14,932 ms at D=1000.
Frame timer and drain counter tell the same story.

### 15.1 The drain's wait CDF, which nothing in this project had

`slope_before/slope_after` is `E[max(0,D-W)]/D`; differencing `D*G(D)` gives:

| D (us) | 62 | 125 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|
| **P(W < D)** | 0.034 | 0.091 | 0.427 | 0.561 | 0.745 |

Mean wait 0.579 ms. Almost no mass below 125 us and **a knee between 125 and
250 us — that knee is the relocation budget in one number.**

### 15.2 The decoy control did not fire

238,912 clears at DECOY=16 and 955,648 at 64, on their own stream, with 0
injections. Mean wait did not rise — it FELL 2% and 6.7%. **Side-stream work does
not lengthen the drain.** The +0.1..0.2 ms frame movement is the host price of
issuing 64 extra clears per drain, not a device effect.


## 16. Item 3 (kernel stalls / "wider requests") — CLOSED, and the briefing was wrong

The handoff said cp_rasterize_stage3 and cp_rasterize_stage3_abuf BOTH read 1.00
sectors/request, and inferred "the fix is wider requests". Source-level
attribution over 900 consecutive launches (skip 200) refutes half of it and
recategorises the other half.

**stage3's 1.00 was an ARTEFACT of the profiled window.** 65.3% of its launches
are DEGENERATE - 16 instructions per warp: read the tile counter, find
num_tiles==0, exit. Those are 65% of launches but 7.3% of time and 4.8% of load
requests. Its WORKING launches issue **3.087 sectors/request**. There is nothing
to widen. (A 6-launch sample reproduced the 1.00 artefact exactly, which is the
recipe's own "two windows of one replay disagreed 50x" hazard, caught by the
agent on itself.)

**_abuf's 1.00 is real and is the OPTIMUM, not a defect.** All 24 global loads
sit at exactly 1.00, in three groups, none a per-lane strided walk:

| share of requests | what |
|---:|---|
| 67.2% | `LDG.E` of `*huge_counter` - ONE UNIFORM 4-BYTE READ PER BLOCK, followed by `R2UR` |
| 27.6% | the 84-byte `cp_setup_cache_entry`, read field by field by `threadIdx.x==0` ONLY |
| 5.2% | `huge_queue[tile_idx]`, an 8-byte pair split into its fields |

**A load issued by one active lane gives one sector per request by construction,
and that is its optimum.** The per-lane traffic in the same kernel is the OUTPUT
side and is already wide - `STG.E.64` at 2.56 sectors/request, atomics at 2.08.

**Where the stall is:** by consumer instruction, `R2UR` waiting on the tile
counter is 51.6% of long-scoreboard samples, the store address chain 37.3%, third
6.5% - top three 95.5%. So more than half the stall is **every warp in the block
waiting on a single uniform 4-byte load at the top of the kernel**: a latency
dependency, not bandwidth and not coalescing.

Widening is legal in exactly one place - padding the setup entry 84->96 bytes to
permit `LDG.128`, collapsing 22 instructions to ~6 - and it touches **0% of the
measured stall**. SoA on the producer would make it WORSE: one lane reading one
record would go from 1 request to 21 on different lines.

**THE LEAD THIS RUN ACTUALLY FOUND is not about memory width: 65.3% of
cp_rasterize_stage3 launches do no work.** That is a launch-count question in the
same currency as item 4 and PDL, priced at 0.782 us per removed launch.


### 16.1 The degenerate-launch lead is a footnote, capped without the mix

The follow-up refused to convert its fractions: its 900-launch window has a
stage3:abuf mix of 5.62:1 against 5.1's 0.55:1, a **10.1x disagreement**, because
launches 201-1100 are the first 4-5 frames and not steady state. Multiplying a
startup fraction by a steady-state count would have produced a number with two
incompatible parents.

**It capped the item instead, which needs no mix:** the degenerate fraction
cannot exceed 1, so even if EVERY stage-3 launch on both variants were degenerate
AND removable, the item is 209.1 launches/frame x 0.782 us = **0.164 ms/frame,
CI [0.132, 0.198]** - 1.3% of the frame at the absolute maximum, 0.5% at the
conditional estimate of 0.062. Footnote.

**And the obvious follow-up is closed before anyone spends a week on it.** The
host cannot know the tile count without a readback, and a readback at that point
IS the wait the census prices at 0.514-6.018 ms/frame: trading a 0.782 us launch
for a sync of that order loses by a factor of hundreds. The trap in the framing:
**a device-side early exit is ALREADY the current behaviour** - the
16-instructions-per-warp launches ARE that exit. The only saving left is not
ISSUING the launch, which is a host decision requiring a device number. Routes
are fusing stage 3 into stage 2, or a device-side conditional. Neither is worth
0.164 ms.


### 15.3 Tier 1 (hoisting the shading-group tables above the drain) — measured zero

Re-measured on the default path after its revert arm was fixed. All four gates
pass, including the one that decides it:

| gate | result |
|---|---|
| both arms complete | 10 of 10 runs rc=0 |
| submit counts | 3,022 old / 2,994 Crossroads, control included |
| candidate hash | session standard, unchanged from before the fix |
| **control hash == candidate hash** | **one hash per capture across both arms** |

So **the reorder really is a reorder** - no byte of output moves when the hoist
is reverted. That was Tier 1's only correctness argument and it now has ten runs
behind it.

**The median is consistent with zero on both captures.** Old: candidate
nominally SLOWER by 0.045 ms, but the three pairwise differences disagree in
sign, span 0.168 ms, and the control arm's own spread across repeats is 0.185 ms
- larger than the session spread and four times the claimed effect. Crossroads:
-0.003 ms, a tenth of that capture's spread. The claim was 0.00 to +0.05; the
measurement cannot distinguish +0.05 from 0.00 from -0.05.

Recorded as what it was declared to be: **a tidiness change worth about zero,
with an unchanged bitstream.** Not landed on performance grounds. The
forced-fallback pair (+0.0184 ms at 28.85 ms/frame) stays on the record as a
different-regime datapoint, not as the result.

**It also cost a bug and paid for its own fix.** The Tier 1 refactor made
`cp_pass_group_table_one()` the single producer of the group rows AND the
allocator of `pass_group_ubos`; the revert path did not re-read that pointer
after the producer ran, so the first multi-member group whose FS reads const
bufs launched a shade with a null table, `rows` collapsed to 1 while `ndraws`
still carried the real row count, and the shader walked off the argument block.
Deterministic device loss at the same capture index on both captures, invisible
under FORCE_PASS_FALLBACK because that path never reaches the grouped shade loop.


## 17. Item 4 — the merge INVERTS, and the rule that explains it

Removing 298.8 launches/frame COST 0.410 ms/frame: an implied **-1.372 us per
launch removed** against a settled credit of **+0.782 us**. First measurement in
this project where removing launches made the frame slower. All supporting gates
passed (one stdout hash across 8 runs, PDL links down 199 as predicted), so the
result is interpretable rather than suspect.

**F4 answers the fork as STRUCTURAL, not implementation.** Per item, merged
against the loop it replaces, two independent pairs agreeing to 1%:

| kernel | control ns/item | merged ns/item | ratio |
|---|---:|---:|---:|
| stage1 | 8,113 | 3,428 | 0.42 |
| stage2 | 5,563 | 2,410 | 0.43 |
| stage3 | 17,065 | 5,747 | **0.34** |

The bar was <=1.25x. **The merged form is three times CHEAPER per item.** Register
pressure is real - 48->64, 56->109, 46->94, occupancy roughly halved - and does
not dominate, because it is amortised over ~5 items.

**Union busy FELL on both measures while the frame got slower**: all kernels
-0.9%, the changed chain -3.2%. The device is not paying for the merge; it is
being paid.

**The overlap factor names it: control 2.19x, candidate 1.37x.** The control's
per-segment launches run concurrently across the eight side streams blended
segments have always used; a merged chunk is one grid on one stream. A merged
launch takes ~2.0x as long as ONE control launch while replacing ~FIVE THAT WERE
RUNNING CONCURRENTLY. Critical path per chunk grows 29.8 us; at 82.3 chunks/frame
that bounds the loss at 2.45 ms/frame, and the observed 0.410 is 17% of the bound.

> **A merge only collects the launch-removal credit where the launches it merges
> were SERIAL. Where they were concurrent, merging converts parallel work into a
> longer critical path and the credit inverts.**

What this does NOT close: a merged form that restores concurrency by issuing
chunks across several streams - which is most of what the loop already did.
Halving the register count cannot help; per-item efficiency is already 3x better.
**What it DOES close: any launch-count forecast for a merge whose launches are
currently concurrent.** The count-phase merge's 554 x 0.782 us = 0.433 ms
forecast is now UNSAFE until its overlap factor is measured at its own position -
and its population is exactly the one the fan-out made concurrent for +2.73 ms.


### 17.1 The count-phase merge is refuted too — its launches are already concurrent

Measured on the SHIPPING default (PDL 3, fan-out on), two windows agreeing to
2.5%, gated first: the counter arm reproduces 1,314.2 launches/frame, which is
5.1's own figure to the decimal.

Overlap factor = summed kernel time / union of intervals:

| population | overlap |
|---|---:|
| abuf count-phase triple, all three stages | **2.50x** |
| `cp_rasterize_stage3_abuf` alone | **1.89x** |
| `cp_clip_rast_fused_abuf` alone | 1.44x |
| `cp_rasterize_stage2_abuf` alone | 1.22x |
| opaque/direct triple, all three | 2.68x |
| all kernels | 1.80x |

**The faithful figure is the per-stage SELF-overlap, not the group's**, because a
merge of "the same stage across the segments of one episode" concatenates that
population and does not remove the pipelining between different stages. Time
weighted, 1.59x - and the dominant stage, 53% of the triple's time, is at 1.89x,
past the 1.8x bar registered in advance.

First-order correction: scale the credit by the serial fraction 1/1.59 = 63%, so
the arithmetic's **0.433 ms/frame is worth about 0.16 ms** before subtracting the
device-side cost of the merged form - and F4 measured that cost, at this width,
to be LARGER than the credit. The direct/opaque path is worse for a merge, not
better, at 2.68x.

**Item 4 is closed.** Not because launch count does not matter, but because this
driver's mergeable launches are the ones the fan-out already made concurrent.

### 17.2 Self-correction to F4's bound

Gating this window exposed that F4's critical-path bound used an assumed
batches/frame instead of the counter arm's own launches/frame: frames per window
137.6 -> 415.3, merged chunks per frame 82.3 -> 27.26, **bound 2.45 -> 0.81
ms/frame**, and the observed 0.410 moves from 17% to **51% of the bound**.
Everything else in F4 is a ratio or a per-unit-work quantity and is unchanged.
The corrected bound is a TIGHTER fit, so the causal account is strengthened, not
weakened.


## 18. Item 2 Tier 2 — the mechanism works; the binding knob was not the one anyone named

Built, measured, and swept. **All correctness gates pass at every setting**:
stray launches 0 and drops 0 including at MIN_VERTS=0, resolves equal drains
exactly (14,932/14,932), counts replayed equal batches held, and ONE stdout hash
across all 24 sweep runs plus the earlier 30. The fail-open never fired, so this
stayed a tuning question throughout.

### 18.1 The sweep

| `MIN_VERTS` | relocated | share of the 0.387 ceiling | batches held | dominant decline |
|---:|---:|---:|---:|---|
| 32 (default) | 0.028 ms/f | 7% | 9,467 | `small` (only) |
| 8 | 0.045 | 12% | 16,822 | `small`, then `full` |
| 2 | 0.092 | 24% | 36,129 | **`full` (only)** |
| 0 | 0.092 | 24% | 36,129 | `full` (only) |

`budget` NEVER FIRES, at any value, on any run. **P0's measured ~125 us/drain of
free absorption is never even tested** - the mechanism's time budget is 13x
unspent while it refuses work for another reason entirely.

### 18.2 The constraint moved, and not where predicted

`small` recedes as forecast and disappears by V=2. It is replaced by **`full` -
the hold capacity of 8**, a fixed extent in the implementation rather than a
measured threshold. At V=2, 63% of the drains that hold anything hold EXACTLY 8,
and 4,096 batches are refused purely because the buffer is full. P1 measured a
population reaching SIXTEEN.

**Note the capacity is not arbitrary**: 8 is `CP_PASS_STREAMS`, and held segments
keep their own queue set so that segments 0..7 map one-to-one onto
`seg_qsets[0..7]` - which is what preserves `fetch_fold`. Raising it is a
queue-set aliasing problem, not a constant.

### 18.3 The frame followed, and could not be resolved

Medians, control-minus-candidate, session spread 0.119: **-0.011, +0.009, +0.028,
+0.069** across V = 32, 8, 2, 0. Monotone, correct sign, tracking the relocated
work - and **every one inside the spread**, with at least one pair of opposite
sign at every value.

So neither registered refutation applies. It is NOT "the moved work was never on
the critical path" (the frame moved monotonically with it) and NOT "holding costs
more than it saves" (no value made the frame worse). **0.092 ms/frame simply
cannot be resolved at a 0.119 spread.** The implied conversion, +0.069 of frame
for 0.092 relocated, is about 0.75 - what a real relocation into a free shadow
should look like.

### 18.4 What that makes the forecast

Reaching P1's full 0.387 ms/frame at a 0.75 conversion is **about 0.29 ms/frame -
the original forecast**, now hanging on a CAPACITY change rather than a threshold
change. The forecast was not wrong about the site, the mechanism, or the
conversion. **It was wrong about which knob was binding, and the driver's own
decline counter says so in one word.**


### 18.5 Three corrections to 18.4, and the finding that replaces it

**The conversion is 1.01, not 0.75.** The held-call timer brackets the
mechanism's own 56.6 KB record copy: 3.84 us per hold measured against P1's
2.87 us of censused vertex phase = 0.98 us of self-overhead. So of 0.092
relocated, **0.068 is work removed from the burst** and 0.024 is new work added
in front of the drain (free, absorbed). 0.069 frame / 0.068 true = **1.01, the
site's own +1.02 recovered end to end.** That makes the forecast WORSE: at 0.75
there was slack; at 1.01 the frame gain can never exceed the vertex work moved.

**P1's histogram is truncated.** It caps at 16 and sums to 88,455 batches while
the same run counted 204,309 appended segments - the 3,179 gaps in the top
bucket average **52.4 each**. The population reaches ~50, not 16.

**The capacity ladder, at conversion 1.0 and the measured reach factor 0.62:**

| cap | true work moved | vs the 0.119 spread |
|---:|---:|---|
| 8 (today) | 0.068 ms/f | measured 0.069, agrees |
| 16 | 0.104 | **still inside** |
| 32 | 0.163 | |
| 64 | 0.238 | needs 56 more queue sets = **1.07 GB** |

A queue set is 19.2 MB, so the honest question was never "is 8 structural" but
**"is +153 MB worth +0.10 ms"** - and the ladder answers no. Nor can it be had
by sharing sets: `fetch_fold` requires seed(A) -> count(A) -> seed(B) -> count(B),
and the mechanism issues EVERY seed before the drain and EVERY count after it, so
two batches on one set have count(B) accumulating on count(A)'s residue. **No
event fixes an ordering the mechanism deliberately breaks.**

**AND THE REAL LIMIT IS REACH, NOT CAPACITY.** 56.4% of deferrals hold nothing
**and are not counted as a decline** - they never reached an admission decision.
The mechanism reaches the BIG gaps essentially perfectly (P1's 4,223 gaps with
>=9 appendable against 4,096 held at the cap, 97%) and almost NONE of the small
ones (P1's 7,321 gaps of one or two against 8,422 drains holding nothing).
Hypothesis under test: **run-ahead cannot cross a render-scope boundary**, because
the fb admission rule must refuse a different framebuffer while the deferred tail
is still reading arrays `cp_abuf_setup` would resize. If so, the bimodal
histogram is simply "the pass ended" and "the pass continued".

**What item 2 lacks is QUANTITY, not conversion and not correctness.**


### 18.6 The resolve-reason probe — item 2 does NOT close, and the gap is scheduling

Self-check passed exactly on all three runs (resolve sums 14,932 / 8,333 /
14,932). Counters only; no median quoted.

**99.99% of the zero-hold deferrals had a LATE SUCCESSOR** - 8,311 of 8,312 on
old, 5,520 of 5,521 on Crossroads. By the pre-registered definition that is
unambiguously the good branch: not "there was no successor at all" (which would
have closed the item at 0.068 ms/frame) but **"the successor existed and a
required resolve came first" - recoverable with NO capacity change.**

| site | zero-hold | share | late |
|---|---:|---:|---:|
| `draw_execute` | 3,823 | **46.0%** | 100% |
| `scope_end` | 3,728 | 44.9% | 99.97% |
| `unknown` | 761 | 9.2% | 100% |
| `flush`, `admit`, `opaque_append` | 0 | - | - |

**Scoring, and two registered predictions failed:**

- **C4 FAILS.** The render-scope explanation does not carry it: `scope_begin`
  never appears, `opaque_append` is 0 on both captures, and `scope_end` alone is
  44.9% - below half by five points. Reported as a failure of the EXPLANATION,
  not of the item: 55.7% still hold nothing and capacity still cannot reach them.
- **C5 FIRES.** `draw_execute` was predicted 0 (by me) and is **3,823 - the
  single largest contributor**, capacity-invariant. The measurer reported it
  rather than adjudicating it against the stray counter, correctly: the two count
  different events (state escaping its extent vs the SITE of a resolve), so they
  need not be inconsistent - but the expectation of zero was wrong by 3,823.
- **C6 PASSES exactly** - `admit` 4,096 against the sweep's `full:4096`, 4,414
  against 4,414 at MAX_SEGS=1, and Crossroads' `seam:15` reproduced to the unit.
  The new key and the old counters count one population.
- **C7 FAILS, and its failure is the finding:** `late` on `scope_end` is 99.97%,
  not small. The recoverable population is not a corner of the data, it IS the
  data.

**The MAX_SEGS=1 cross-check is decisive about capacity:** `draw_execute` zero
stays EXACTLY 3,823 and total zero stays EXACTLY 8,312 while the histogram
collapses and `admit` rises 4,096 -> 4,414. Capacity moves work between `admit`
and the held set **without changing how many drains find nothing to hold.**

**Status: item 2 is ALIVE, the ceiling is unchanged at 0.387 ms/frame, 0.092 of
it is converted today at ~1.0, and the gap is a SCHEDULING problem - a required
resolve pre-empting an available successor - not a capacity one.** The next lever
is cheaper than the one the sweep pointed at.

(Correction: the parent briefed 8,422 zero-hold deferrals. The histogram and the
instrument independently both say 8,312.)
