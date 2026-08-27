# Item 2 Tier 2 — the run-ahead, built

Branch `drain-work`, worktree `/tmp/drain-tree`, build `build-cudavk-drain`.
Base `43df52c383e`; `3f62f2174c3` Tier 1, `c7f2f8e9f08` P0+P1 probes,
`b6232f5ff3f` the Tier 1 revert-path fix (JOB 1), this commit Tier 2.

**Nothing here was run on the GPU.** A measurer owns the card. Every number
below is either a measurement someone else made (P0, P1, the wait census) or a
machine-level property of the object file, and the two are labelled apart.

---

## 1. What it does

`cp_pass_finish()` is cut in two at the sync.

* The **front half** is everything it always did up to the drain: the join, the
  scan, the fill, the sort, the quad build, the segment bucketing, and Tier 1's
  shading-group tables. It ends by building `struct cp_pass_tail` — every local
  the rest of the function needs — on its own stack.
* The **tail** is `cp_pass_tail_run()`: the drain, the counter read-back, the
  dense bases, the scatter, the shades and the composite. With the flag off it
  is called immediately, with a pointer to that stack struct, so the split costs
  a call and copies nothing.
* With `CUDAVK_AHEAD` set, `cp_ahead_defer()` takes the tail instead and
  `cp_pass_finish()` returns. The host goes back to replaying the command
  stream.

The appendable blended batches that follow are then **held**: their vertex phase
is issued on a dedicated ahead stream and stops at the seam in
`cp_draw_execute_batch()` — the line the P1 census already marks — recording a
`struct cp_ahead_seg` instead of counting. The count phase is not run, because
it writes the episode's shared per-pixel lists, which the deferred tail is still
reading.

`cp_ahead_resolve()` runs the tail (the drain is taken at the same point in the
stream order, for the same length), then issues the episode's shared clears and
the gate, then replays each held batch's count phase on the segment stream it
would have used. The host reaches the sync having already issued the vertex work
that used to sit behind it.

**One producer, not two.** The count phase is `cp_abuf_count_phase()`, factored
out of `cp_draw_execute_batch()` and called from both places — the same
discipline `cp_pass_group_table_one()` got in Tier 1, and for the same reason: a
second copy of a hundred lines of launch arguments is a second thing to keep in
agreement.

### 1.1 Four hazards the design did not name, and what closes each

Found while writing it, all four are in the source as comments.

1. **The episode's clears cannot run at the first held batch.** They zero
   `ab->counts`, which the deferred tail's composite reads, and they would be
   issued *ahead* of it on the same stream. They are factored into
   `cp_pass_episode_begin()` and issued at replay time, after the tail — which
   is the same order the unmodified driver has, since there the clears also
   follow the previous episode's composite on that stream.
2. **`cp->pass_segs` would be written under the deferred tail.** The tail shades
   from that array. Fixed by NOT recording held batches as segments until
   replay: `cp_pass_record_segment()` runs inside the replayed count phase, in
   index order, so `prim_base` is still assigned in submission order and the
   sort's ascending order is still submission order. No second segment array,
   and the design's "two halves" generation trick is not needed.
3. **The held vertex stage races the previous episode's count stages.** A held
   batch keeps the rasterizer queue set of the segment it will become, so that
   `fetch_fold` can still seed that set's counters — and those counters are the
   ones the previous episode's segment on the same set is still reading, because
   the drain that would have ordered them has not been taken. Closed by one
   event recorded on the main stream at the first hold of each drain (the main
   stream is already behind every segment stream, via the front half's
   `cp_pass_join`), waited on by each ahead stream once. **One record and at
   most two waits per drain, issued before the sync and so absorbed by it.** The
   alternative — turning `fetch_fold` off for held batches — pays two clears per
   held segment at *replay* time, i.e. after the sync at the full 1.02 slope:
   0.08–0.19 ms/frame against this form's ~0.005. That is the design's own
   failure mode 2, and it is why the ordered form is the mechanism.
4. **An opaque run may start while a blended tail is deferred.**
   `cp_opaque_append()`'s two guards are both on `nsegs`, which is zero while a
   tail is deferred, so neither fires and its segments would land in
   `cp->pass_segs`. It now resolves first.

### 1.2 What the design worried about that turned out not to exist

* **Stale A-buffer addresses (its failure mode 1).** The held record captures
  `rast_args` *before* the A-buffer fields are set; `cp_abuf_count_phase()` fills
  them from the live `ab` at replay. `aa.abuf_frags` therefore cannot be stale,
  and the fail-open item the design called its second-largest is gone. The
  admission rule still refuses to hold when `ab->grow_to` is set, because a
  growth served between hold and replay would free arrays the RECORDED segments
  reference later.
* **The two ahead arenas.** See §4.

---

## 2. The admission rule, and why it is a time budget

**`CUDAVK_AHEAD_BUDGET_US`, default 125.** Stop admitting once that much host
time has been relocated in front of one drain. It is the knee of the wait CDF
P0 reconstructed:

| D (us) | 62 | 125 | 250 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|
| P(W < D) | 0.034 | **0.091** | **0.427** | 0.561 | 0.745 |
| measured slope, work moved BEFORE the sync | 0.034 | **0.065** | 0.257 | 0.413 | 0.589 |

Almost no mass below 125 us and a knee between 125 and 250. Under 125 us per
drain is nearly free — 0.065 of it lands on the frame — and past 250 it is paid
for at a rising rate. The rule spends in the units the CDF is measured in, which
a batch count cannot do: the same three batches are 8 us on one drain and 300 on
another.

The budget is measured, not estimated: each hold is bracketed with
`os_time_get_nano()` and the blocked-time delta is subtracted, exactly as the
P1 census separates issue time from gap. The batch that crosses the budget is
admitted (the cost is not knowable before the work) and is the last.

**The population is bimodal and the rule serves both modes.** P1: 36% of drains
have exactly ONE appendable batch, 21% have the full 16, median 3, mean 5.92,
thin middle; Crossroads median 1.

* The **one-batch mode is where the cost lands**, so it is protected by
  `CUDAVK_AHEAD_MIN_VERTS` (default 32 assembled vertices). Below about a dozen
  triangles a vertex phase is its fixed part — a malloc, the ref loop, the
  slice-table upload, one or two launches — and that is a few microseconds
  against roughly 1.3 us of added stream switch and ordering edge. The floor
  exists so that a degenerate batch does not pay a switch for nothing. It is a
  sweep knob, not a discovery; 0, 32 and 256 are the arms to try.
* The **sixteen-batch mode is where the win is**, and there the budget and
  `CUDAVK_AHEAD_MAX_SEGS` bound it. Note the arithmetic that says the budget
  will rarely bind: the censused vertex ceiling is 0.387 ms/frame over 9.88
  drains = **39 us per drain**, which is 31% of the 125 us that is free. The
  whole prize fits inside the region P0 measured as absorbed.

`CUDAVK_AHEAD_MAX_SEGS` defaults to 8 and is capped there by correctness, not
taste: held batch k takes queue set k and there are eight sets.
`CUDAVK_AHEAD=1` holds at most two, as the cheap arm of the cost/gain trade.

Four more admission rules are correctness: no pending A-buffer growth, the same
framebuffer as the A-buffer is sized for (so `cp_abuf_setup()` is the no-op it
is for an equal size and cannot resize arrays the tail is reading), the streams,
and a device that is not already lost. Every refusal is counted by reason.

---

## 3. What stays fail-closed, and what becomes fail-open

Untouched, and this is the line S1d could not hold: **the overflow branch,
`cp_pass_can_retry()`, the `running != quads` check and the `capacity/4` gate**
are all inside the tail and are byte-for-byte what they were. Nothing that is
fail-closed for the correctness of the drawn image becomes fail-open.

What does become fail-open is the **extent of the held state** — the design's
own item 1. It is not asserted about, it is counted, and the counters print
without any other flag:

```
cudavk: run-ahead: N drains deferred, N batches held, N counts replayed,
        N resolves, N dropped after a fallback, N stray launches
cudavk: run-ahead: X us relocated per deferral (mean), Y MB scratch worst
        drain, cap 24 MB, budget 125 us
cudavk: run-ahead: batches held per drain: 1:.. 2:.. ...
cudavk: run-ahead: declines: full:.. budget:.. bytes:.. small:.. abuf:..
        fb:.. state:.. seam:..
```

* **stray launches** is the design's `cp_launch()` assertion, as a count: no
  kernel may be issued while a tail is deferred except a held batch's own vertex
  phase. It is one null test per launch with the flag off. **It must be zero; a
  non-zero value is a missed resolution point, which is this design's one silent
  failure mode.**
* **dropped after a fallback** is the back-out the design warned would be
  invisible (its failure mode 4). A tail that fell back re-executed its segments
  classically, and a classic execution rewinds the device scratch over the held
  vertex output — so the holds are dropped and those batches run normally
  instead. Correct, and on these captures it should never happen.
* **batches held per drain** is the histogram to compare against P1's
  appendable-batch histogram. If the mechanism is not seeing the same
  distribution, the admission rule was tuned on the wrong population.

The resolution points are three, not nine, because the driver only has three
ways out of the append loop: `cp_pass_finish()` (which every readback, clear,
copy, query, dispatch, barrier, scope change and submit reaches through
`cp_batch_flush*`), `cp_opaque_append()`, and — as a net rather than a rule —
the top of `cp_draw_execute_batch()`. The resolve in `cp_pass_finish()` runs
BEFORE `cp->pass.nsegs` is read, because while a tail is deferred that is zero
and the early return would step over the whole deferred episode.

---

## 4. The memory objection, answered by deleting the allocation

P1 measured the demand the design had guessed: **0.01 MB per segment mean, 0.59
MB worst segment, 2.14 MB peak per episode** on old (0.35 on Crossroads),
against the design's own 8–40 MB band and its 64 MiB-per-half placeholder.

At that size the two dedicated arenas are not worth their own failure modes. The
held vertex phase allocates from the device scratch it already uses, with the
rewind suppressed while holding — so its buffers land ABOVE the deferred
episode's high-water, which is exactly what every second and later segment of an
episode already does. That removes the bump allocator, the two-half generation
trick, the eight pointer patches the design's §2.3 needed to move `clipped` and
`prim_refs` to replay time, and the mid-batch decline that would have gone with
them.

**The fixed number that replaces the 64 MiB placeholder is a CAP, not an
allocation: `CUDAVK_AHEAD_HOLD_MB`, default 24.** It is the measured 2.14 MB
peak per episode, times eight for the clip outputs the census deliberately
excluded (`clipped` is `384*T*V` against the vertex output's `48*T*V`), plus
margin. The design's no-grow property is kept exactly: there is no grow path,
the run-ahead **declines** and the batch runs normally, and the decline is
counted. The bytes are measured per drain from the scratch high-water, not
projected from a formula.

---

## 5. Inertness: base / feature LIVE / gates folded

Handoff §13. The vacuous result is a folded-off build that is identical to base
*because the implementation is missing*, so the arm that matters is the one that
shows the feature EXISTS.

`cp_renderer.c.o`, same command line as the build, `.text*` summed over
function sections (`-ffunction-sections`, so `size`'s single `.text` is zero and
would be the wrong number to quote):

| arm | .text bytes | function sections | vs base |
|---|---:|---:|---|
| **A** base (`b6232f5ff3f`, no Tier 2) | 155,332 | 175 | — |
| **B** shipped, flag defaults off | 160,753 | 179 | **+5,421, +4 sections** |
| **C** gates folded to the value that deletes the code | 157,034 | 173 | +1,702, −2 |

**B − A = +5,421 bytes: THE FEATURE IS THERE.** A deleted implementation cannot
produce that.
**B − C = −3,719 bytes and 6 function sections: THE GATE DELETES IT.**

The fold is stated so it can be repeated: `cp_debug->ahead` → `0`,
`cp_ahead_deferred()` → `return false`, `cp_ahead_holding()` → `return false`,
and the two counter hooks `cp_ahead_note_launch()` / `cp_ahead_note_fallback()`
→ empty. The last two are there because of §13.4's second rule — the parts too
cheap to gate are the ones that survive folding and then have to be explained.

Sections in B and not in C: `cp_ahead_resolve`, `cp_ahead_drop`,
`cp_ahead_streams_ready`, `cp_pass_episode_begin`, `cp_pass_tail_run`,
`cp_batch_total_triangles` (its only remaining caller is the admission rule, so
folding re-inlines it), plus one inlining flip (`cp_opaque_tile_visibility` in
B, `cp_opaque_finish` in C) which is a decision change, not code.

**C is NOT equal to A, and it is not claimed to be.** C − A = +1,702 bytes is
the refactor: `cp_abuf_count_phase` exists as a real function (2,956 bytes) in
both C and B, and `cp_pass_finish` shrinks by 1,607 in exchange. Like Tier 1,
this part is a reorder whose generated code differs by construction; the
evidence that it cannot change a value is the single-producer argument in §1,
and the gate for it is the stdout hash from the A/B — which JOB 1 has just made
obtainable again.

Registry control, where §13.2 says it must be — `.data`, not `.text`:

| section | base | shipped | delta |
|---|---:|---:|---|
| `cp_debug.c.o` `.text*` | 3,011 | 3,011 | **0, by construction** |
| `.data.rel.ro.local.flags` | 11,088 | 11,528 | **+440 = 5 rows x 88** |
| `.bss.present_in_env` | 126 | 131 | **+5** |
| `.bss.debug_state` | 312 | 336 | +24 (five fields, appended) |

Collateral, files not edited, compiled from a base worktree and from this tree
with the same command: `cpvk_device_memory.c`, `cpvk_cmd.c`, `cpvk_device.c`,
`cp_kernels.c`, `cpvk_texture_cache.c` — **`.text` byte-identical in every
one** (10,321 / 42,126 / 2,928 / 4,853 / 11,064). The only section deltas are
`__FILE__` strings in assert helpers, which differ because the two compiles see
different source paths. The new field went at the END of `cp_context` and is one
pointer, so no struct offset moved.

The house checks pass: `cp_debug_doc.py --check`, `cp_no_getenv.py`,
`cp_launch_audit.py`.

---

## 6. The A/B

Standing rules apply and are not optional: GPU exclusive under a 1 Hz sampler,
arms strictly alternating in one session on one binary, **one `sha256sum` of
stdout across every arm**, submit counts (3,022 old / 2,994 Crossroads) checked
BEFORE any median is read, `median(diff(submit_ts[::2])[50:])`.

```
control:    (nothing set)
candidate:  CUDAVK_AHEAD=2
cheap arm:  CUDAVK_AHEAD=1                  # holds at most two batches
```

Read `cudavk: run-ahead:` on EVERY candidate run before reading any median. A
run that deferred nothing or held nothing is not a candidate arm, it is a second
control — R4's lesson and PDL's caveat 3. Then check, in this order:

1. `stray launches` is **0** and `dropped after a fallback` is **0**;
2. the stdout sha is the same on both arms (`320e993599cc` old,
   `e727020fc796` Crossroads);
3. submits are 3,022 / 2,994 on both arms;
4. `batches held per drain` against P1's appendable histogram;
5. `us relocated per deferral` against the 39 us/drain the ceiling implies;
6. only then the median.

Worth two more runs: `CUDAVK_FORCE_PASS_FALLBACK=1` on both arms, which is the
only way these captures exercise the drop path, and `CUDAVK_AHEAD_MIN_VERTS=0`
as the single-batch-mode arm.

---

## 7. Forecast, registered before any measurement

Session spread: 0.119 ms on old, 0.034 on Crossroads.

| | old | Crossroads |
|---|---|---|
| ceiling (P1, measured) | 0.387 ms/frame | 0.072 ms/frame |
| design's tier-2 conversion | 55–80% | 55–80% |
| **forecast delta, candidate − control** | **+0.21 to +0.31 ms/frame** | **+0.00 to +0.06, i.e. not distinguishable from zero** |

The old figure is 2–3x the session spread and should be visible in three runs.
The Crossroads figure is inside its own spread and is a **null prediction**: it
is quoted so that a Crossroads gain would be as surprising as a regression.

Two forecasts about the mechanism rather than the frame:

* **held per drain**: bimodal, ~36% at 1 and a fifth at the cap of 8 on old,
  mostly 1 on Crossroads;
* **declines**: dominated by `full` and `small` on old, NOT by `budget` — the
  ceiling implies 39 us per drain against a 125 us budget.

### Falsifiers

* **Y1 (did not fire).** `drains deferred` or `batches held` is 0, or `resolves`
  is 0. Nothing may be concluded in either direction; the run is a control.
* **Y2 (correctness, and it voids the timing).** The stdout sha differs between
  arms, or submits are not 3,022 / 2,994, or `stray launches` is not 0, or
  `dropped after a fallback` is not 0 on captures that never overflow. Any one
  of these means a resolution point is missing and no median may be quoted.
* **Y3 (value).** Old comes in below **+0.10 ms/frame** with the histogram
  showing a median hold of 3 or more. Then the conversion is under 26% of the
  ceiling, against a design that predicted 55–80%, and the cost is somewhere the
  §6-mode-2 table does not have a row for. Look at the replay-side edges first.
* **Y4 (device).** The mean episode drain wait rises by more than 10% against
  the control. Then held work on a side stream lengthens the drain after all,
  X3 fires, and P0's decoy control was measuring the wrong thing.
* **Y5 (admission).** `budget` is the dominant decline reason on old. Then the
  hold cost is roughly three times what the ceiling implies, or the timer is
  charging blocked time to the hold, and the 125 us default is not the right
  number to spend.
* **Y6 (population).** The held histogram is not bimodal. Then the mechanism is
  not seeing the population P1 censused and the rule was tuned on the wrong
  distribution — check `small` and `fb` declines.
* **Y7 (capture split).** Crossroads regresses by more than 0.05 ms while old
  gains. The cost is per held batch and Crossroads holds one at a time; the
  default stays OFF permanently.

**Y3 and Y6 are built to be able to disagree**, in the same way X1 and X5 were:
Y3 says the moved work is not worth what it cost, Y6 says the work was never
moved in the quantity the census promised. Their disagreement is what separates
a mechanism that is too expensive from one that is not firing.

---

## 8. What is code-verified here and what is not

**Verified by building it:** the object-level numbers in §5, the three house
checks, and that the tree builds clean.

**NOT verified, because there is no GPU in this session:** that it renders
correctly. Every ordering argument in §1.1 is a reading of the source, and the
first four things the A/B checks (§6) are exactly the ones that would catch a
wrong one. The mechanism is default-off; nothing about a shipping run changes
until a measurement says it should.
