# What else can overlap? — survey, no code, no GPU

Read against `docs/cudavk/PERFORMANCE.md` §5 and §7 at `e2fea470d04`. Section
5's kernel table was taken at `ba8891878df`, i.e. **before** the opaque fan-out
landed, so its absolute idle figures are stale; the shares and the launch
counts are not.

## 0. The correction that matters first

**Blended (A-buffer) episode segments already fan out.** `cp_pass_append()`
sends segment `k = nsegs % CP_PASS_STREAMS` to `cp->seg_streams[k]` whenever the
streams exist, with no flag, and `cp_pass_streams_init()` says so in its own
comment: the eight streams were built for *the blended admission test* and the
2026-08-26 commit only extended the same machinery to opaque episodes. So
"fan out the A-buffer raster chain the way the opaque one was" is **not an
available idea** — `cp_rasterize_stage3_abuf`'s 134.6 launches/frame are
already spread over eight streams. Anyone proposing it is proposing the status
quo.

## 1. The width is already done, and this is the number that says so

`cp_rasterize_stage3_abuf`: median **0.15 waves per SM**, max 0.60, **0 of 270
launches reach 1.0**, median grid 512 blocks, 18.5 µs per launch, 2.484 ms/frame
— the largest single kernel class in the driver.

> One launch occupies 15% of a wave, so **about 6.7 concurrent launches fill the
> machine. There are already 8 side streams plus the main one.**

So the fan-out width (8) already exceeds the width the machine needs (≈7).
**Raising `CP_PASS_STREAMS` cannot pay**, and neither can anything
kernel-internal: PERFORMANCE §7 has already closed that with two whole-scheduler
rewrites built, measured and reverted. The `k = nsegs % 8` stream reuse does
create a false dependency for episodes longer than 8 segments (old averages
12.42 and reaches 49), but by the same arithmetic the machine is full at 7, so
the reuse costs queueing rather than throughput.

**The remaining problem is duty cycle, not width.** Pre-fan-out the device was
resident 11.83 ms of a 15.99 ms frame and idle 4.16 ms, and *the idle was almost
exactly the host's own 3.55 ms of issue time*, because the driver blocks 16.76
times a frame and cannot issue while blocked. Sum of kernel time 14.52 ms
against union-busy 11.97 ms says only 2.55 ms of concurrency existed then; the
fan-out added its 2.55 ms of frame on top. What is left is not "more streams"
but "issue across a host round trip".

## 2. Three candidates

### C1 — Price the episode drain with the instrument that just closed the peel site. **Recommended.**

*What:* generalise the two counters already in `cp_sync_timed` — `cuStreamQuery`
before the sync, and `Σ min(host issue burst after the wait, the wait)` — from
the peel site to all five wait sites. About ten lines; the arithmetic, the
plumbing and the report all exist on branch `peel-predicate`.

*Why it is first:* the episode drain is **8.663 ms/frame over 9.88 waits**, the
largest wait in the driver by a factor of three, and PERFORMANCE §6 item 4
already names this exact probe (the addendum's **P2**) as the thing to do
*before designing anything there*. It is now nearly free.

*Evidence:* the same instrument closed the peel lead in one run per capture. It
produced the conversion factor — **blocked time converted to frame time at 11%**
— which is the number that turns a headline wait into a value. Nobody has that
number for the drain, and every estimate at that site has been made without it.

*Prediction worth recording before it runs:* `ready` will again be 0 (the device
is busy at the drain, as it was at every peel check), but the drain's
`min(gap, block)` will be **much larger** than the peel site's 0.376 ms, because
the drain sits between episodes where the host has a whole episode's issue in
front of it, whereas a peel check sits mid-loop with one pass (~11 launches,
tens of µs) to issue. If the drain's ceiling is ~1 ms at a ~30% conversion, that
is ~0.3 ms and worth a design. **If its conversion is 11% too, item 4 collapses
the same way this lead did**, and the driver's two largest waits are both
closed for the same reason. Either answer is worth having, and it costs one run.

### C2 — Issue the *next* episode's vertex stage across the drain.

*What:* the vertex stage is the one phase that touches none of the A-buffer's
shared arrays: it reads immutable vertex buffers and writes its own output
buffer. VS main is **1.843 ms/frame in 221.5 launches**, plus 0.350 ms of
`vertex_fetch` — 2.19 ms of kernel time sitting behind a host round trip it does
not depend on.

*Why it is not the refuted route:* every scheme iteration 29 refuted (S0, S1b)
failed by needing a **second set of A-buffer arrays** — 250 MB–1.6 GB against an
8.59 GB scratch cap, on a grow path that turns pressure into per-flush
`cuMemAlloc`/`cuMemFree` churn. Running one phase ahead needs **one more VS
output buffer**, not a second A-buffer, so it does not inherit the disease. The
disease was the ratio bound/actual on the *fragment* arrays.

*What actually serialises it today, and must be checked first:* the scratch
arena is a single bump allocator and `cp_scratch_reset()` frees overflow arenas
at **every flush**, so the next episode's VS output would have to survive the
current episode's flush. That, not the concurrency, is the blocker, and it is
readable from the code before anything is built.

*Bounded experiment, instrumentation only:* time the VS phase that immediately
follows each episode drain, and the drain it follows. If the VS phase is under
~0.1 ms per episode there is nothing to hide and the candidate dies for a
tenth of the cost of building it. This shares the C1 timer.

### C3 — The honest negative: say plainly that width is collected.

**Ready to paste into `PERFORMANCE.md` §7, in that section's terms:**

> - **Making the raster chain wider is closed.** `cp_rasterize_stage3_abuf` is
>   the largest single kernel class at 2.484 ms/frame, and it is *already*
>   concurrent work: `cp_pass_append()` has sent blended segment `nsegs % 8` to
>   `seg_streams[k]` since before the opaque fan-out existed, with no flag, and
>   `cp_pass_streams_init()` records that the eight streams were built for the
>   blended admission test in the first place. NCU measures
>   `launch__waves_per_multiprocessor` at median 0.15 over 270 launches, so one
>   launch occupies 15% of a wave and **about 6.7 concurrent launches fill the
>   machine against the 8 side streams that already exist.** Raising
>   `CP_PASS_STREAMS` therefore cannot pay, `k = nsegs % 8` reuse costs
>   queueing rather than throughput, and the kernel-internal route is closed
>   above with two reverted rewrites. **What is left at this site is duty
>   cycle, not width**: the device idle is almost exactly the host's own issue
>   time, because the driver blocks about seventeen times a frame and cannot
>   issue while blocked.


Record in PERFORMANCE §7 that **the fan-out collected the available width**, with
the arithmetic: 0.15 waves/SM × 8 streams ≈ 1.2 waves, ≈6.7 concurrent launches
fill the machine, so more streams, wider grids and kernel-internal rewrites are
all closed. `cp_rasterize_stage3_abuf` being the largest kernel class at
2.484 ms/frame is **not** an invitation to make it faster; it is 134.6 launches
of already-concurrent work at 18.5 µs each.

## 3. The honest summary

Three things have paid in this driver and all three are overlap: the opaque
segment fan-out (+2.73 ms), PDL levels 1–3 (≈ −0.52 ms), and nothing else. The
fan-out took the width. What is left is the sixteen-odd host round trips a
frame, and the peel result has just shown that **a round trip is only worth
removing where the device is idle behind it** — which at the peel site it never
was, 0 times in 2,577.

So the next question is not "what else can be made concurrent" but **"which of
the remaining round trips has an idle device behind it"**, and C1 answers that
for all eight sites in one run. If none of them does, the overlap programme is
finished and the driver's remaining 3–4 ms of device idle is the host's own
issue time, which is a launch-rate problem and not an overlap problem.

**And that is consistent with what is still paying.** PDL is a launch-rate
change — stream 4.10 µs/kernel → graph 2.02 → PDL 1.89 — on a driver that
issues over a thousand launches a frame, and it is the only lead still moving
the number. If C1 comes back with `ready = 0` everywhere, the next iteration
should be about launch rate rather than about overlap, and PDL is the evidence
that the axis is real.
