# Item 2 — the hold capacity of 8: what it is bounded by, and what raising it buys

Answer to three questions on `/tmp/perf-audit/item2_min_verts_sweep.md`. **No code
was written and nothing was run on the GPU.** Everything below is either the
sweep's own numbers, P1's census, or arithmetic on the two — labelled which.

---

## 0. The short answers

1. **Neither, cleanly.** Three constants are aliased to 8 today: the held record
   array, the flag's range, and the queue set index. The first two are a
   one-line change. The third is the real coupling — but it binds to
   **the number of rasterizer QUEUE SETS**, which happens to equal
   `CP_PASS_STREAMS` only because a queue set is allocated per stream. Queue
   sets are a resource, not a law. **8 is not a structural constant; it is a
   19.2 MB-per-unit resource constant.**
2. **No, and no event can fix it.** Reuse within one deferral is not a race that
   an edge orders — it is an ordering that cannot exist. The seed must sit
   *between* two count phases; the mechanism puts every seed *before* every
   count by construction. My main-stream gate orders held vertex work against
   the PREVIOUS episode's counts, which is a different pair.
3. **No. I do not believe 0.29.** Capacity 16 is worth about **+0.10 ms/frame**,
   which is *below* the 0.119 session spread — it would buy another
   unmeasurable result. The 0.29 arithmetic assumes capacity 16 captures the
   whole ceiling; the measured shape says it captures 43% of the population and
   the mechanism reaches 62% of that. **And the conversion is not 0.75, it is
   1.01** — which makes the forecast *worse*, not better, because it removes the
   slack that could have hidden a bigger number.

There is a fourth thing you did not ask about and it is larger than all three:
**the mechanism holds nothing on 56.4% of drains, and those are the gaps with
one or two appendable batches — not the capped ones.** §4.

---

## 1. Q1 — what actually bounds 8

Four uses of the constant, and they are not the same kind of thing:

| site | what it costs to raise | kind |
|---|---|---|
| `struct cp_ahead_seg held[CP_AHEAD_MAX_SEGS]` | `sizeof(cp_ahead_seg)` ≈ **56.6 KB** each (it is dominated by `cp_draw_batch`, 56,016 B, measured). 8 → 452 KB, 16 → 0.88 MB, 64 → 3.6 MB of host memory in one calloc | **aliased**, one line |
| `held_hist[CP_AHEAD_MAX_SEGS + 1]` | 8 bytes each | **aliased**, one line |
| `CUDAVK_AHEAD_MAX_SEGS` range 1–8 | nothing | **aliased**, one line |
| `cp->seg_qsets[slot % CP_PASS_STREAMS]` | a queue set per concurrently-held batch | **the real bound** |

**The replay STREAM is not a bound at all.** Held batch *k* replays on
`seg_streams[k % 8]`, and segments sharing a stream is already normal: an
episode may have up to `CP_PASS_MAX_SEGS` = 64 segments and segment 8 already
runs on stream 0 behind segment 0 today. Streams serialise, which is exactly
why sharing them is safe.

**The queue SET is the bound, and it is a resource.** One set is
`CP_MAX_NONTRIVIAL`×4 (4.0 MB) + `CP_MAX_HUGE_TILES`×`sizeof(cp_tile_pair)`
(16.0 MB) + 256 B counts + `CP_SETUP_CACHE_CAPACITY`×84 (84 KiB) =
**19.2 MB per set**, and the driver allocates eight of them (153 MB) today
because it has eight side streams. Nothing stops allocating dedicated
run-ahead sets: **+8 sets = +153 MB, +24 = +460 MB, +56 = +1.07 GB**, permanent,
on a box where the handoff already records a 12 GB tenant.

So the honest form of your closure question is not "is 8 structural" but
**"is another 153 MB of device memory worth +0.10 ms/frame?"** — and §3 says it
is not.

---

## 2. Q2 — does `fetch_fold` survive a reused queue set inside one deferral

**No, and this is stronger than a race.** Write the required order out.

`fetch_fold` means the fetch/VS kernel writes the three queue counters to zero,
and the count phase then SKIPS its `cuMemsetD32Async(cur_qset.counts, 0, 3)`.
Stage 1 fills those counters by atomic add; stages 2 and 3 read them. For two
batches A and B sharing a set, correctness needs

```
seed(A) -> count(A) -> seed(B) -> count(B)
```

The mechanism issues **every seed before the drain and every count after it**:

```
seed(A) seed(B) | drain | count(A) count(B)
```

`count(B)` therefore accumulates on top of `count(A)`'s residue: the queues are
appended after A's entries, the counts read A+B, and stage 3 walks A's entries
as if they were B's. Silent wrong geometry, or `CP_MAX_NONTRIVIAL` overflow.

**No event can reorder this.** Putting `seed(B)` after `count(A)` means putting
B's vertex phase after the drain, which is the one thing the mechanism exists
to avoid. The gate I added is a different pair — it orders the ahead streams
behind the main stream at deferral time, which covers the *previous* episode's
count stages (the hazard the design missed). It says nothing about A and B
inside one deferral, and a second event would have nothing to say either.

Today's driver is safe for the same reason it is safe with 64 segments and 8
sets: segments *k* and *k+8* are on the SAME stream, so stream order interleaves
their seeds and counts correctly. Run-ahead breaks that interleaving on purpose.

**So capacity beyond 8 has exactly two forms:**

* **(a) more queue sets** — no ordering change at all. The cross-deferral reuse
  of an ahead set is already covered by the existing gate, because the previous
  deferral's count ran on a segment stream before the current deferral's gate
  event was recorded on the main stream. Cost is the 19.2 MB per set above.
* **(b) `fetch_fold` off for slots ≥ 8** — two clears per such batch, issued at
  REPLAY, i.e. **after the sync at the site's full 1.02 slope**. Priced from
  the sweep's own population in §3, it costs roughly what it buys.

---

## 3. Q3 — the honest ceiling at capacity 16

### 3.1 First, the conversion is 1.01, not 0.75

The sweep's 9.3 us per deferral is what the mechanism's timer measured, and the
timer brackets the whole held call — including the **56.6 KB record copy and the
stream switch that the mechanism itself adds**. Per held batch:

| | us |
|---|---:|
| measured per held batch (138.9 ms / 36,129) | **3.84** |
| P1's censused vertex phase per segment (585.4 ms / 204,309) | **2.87** |
| **the mechanism's own overhead per hold** | **0.98** |

So of the 0.092 ms/frame "relocated", **0.068 is vertex work removed from the
post-drain burst** and 0.024 is new work added in front of the drain — which is
free, because that region is absorbed at slope 0.065. Against the measured
+0.069 ms/frame that is a conversion of **1.01**: the site's own +1.02, recovered
exactly. **The mechanism converts perfectly. It simply moves too little.**

This matters for the forecast in the direction nobody wants: at 0.75 there was
slack to hope for; at 1.01 the frame gain can never exceed the true vertex work
moved, and that quantity is now the entire question.

### 3.2 The population, corrected — P1's histogram is truncated

P1's `appends_hist` caps at 16 (`MIN2(gap_appends, 16)`), and the sweep exposed
what that hides. Summing the histogram gives 88,455 batches over 14,931 gaps,
but the same run counted **204,309 appended segments**. So:

* gaps with fewer than 16 appendable batches hold **37,591** of them;
* the **3,179 gaps in the "16" bucket hold 166,718 — a mean of 52.4 each.**

**21% of drains are followed by fifty-odd appendable batches, not sixteen.**
"P1's population reaches sixteen" is the histogram's ceiling, not the
workload's. Capacity 16 is not the top of this distribution; it is a fifth of
the way up its tail.

### 3.3 What each capacity captures

`ideal` = Σ min(k, C) over the corrected population. `reach` = the measured
fraction of ideal the mechanism actually holds, **0.62**, from
36,129 held / 58,250 ideal at capacity 8. True vertex work = held × 2.87 us.

| capacity | ideal batches | % of population | true ms/frame moved | **frame gain at conversion 1.0** | vs 0.119 spread |
|---:|---:|---:|---:|---:|---|
| 8 (today) | 58,250 | 28.5% | 0.068 | **+0.068 measured (+0.069)** | inside |
| 16 | 88,455 | 43.3% | 0.104 | **+0.104** | **inside** |
| 32 | 139,300 | 68% | 0.163 | +0.163 | 1.4x |
| 64 | ~204,300 | ~100% | 0.238 | +0.238 | 2.0x |

(32 and 64 model the ≥16 gaps as uniform at their 52.4 mean; the shape inside
that bucket was never measured, so treat those two rows as an upper bound with
the shape unknown.)

**Capacity 16 buys +0.10 ms/frame and stays inside the spread.** The 0.29
arithmetic needs 75% of the ceiling; capacity 16 reaches 43% of the population
before the 0.62 reach factor, i.e. 27%.

### 3.4 And form (b) cancels itself

If the extra slots pay two clears at replay instead of taking new queue sets:

| capacity | extra clears/frame | cost at 0.79 us | cost at 1.97 us (exposed) | incremental gain | **net** |
|---:|---:|---:|---:|---:|---|
| 16 | 40.0 | 0.032 | 0.079 | +0.036 | **+0.004 to −0.043** |
| 32 | 66.5 | 0.053 | 0.131 | +0.095 | +0.042 to −0.036 |
| 64 | 119.8 | 0.095 | 0.236 | +0.170 | +0.075 to −0.066 |

Those clears are issued immediately after a drain, which empties the stream by
definition, so **1.97 us is the expected column, not the optimistic one**. Form
(b) at capacity 16 is a coin flip around zero. This is the design's failure mode
2 arriving exactly where it said it would.

### 3.5 What binds at 16, and what would have to bind for this to be worth it

Not the budget. At capacity 64 the relocation would still be 24 us per deferral
against a 125 us budget — **the CDF admission rule was derived correctly and
this workload simply cannot spend it.** It would bind on a workload with five
times the vertex work per gap, and it should stay in the driver for that reason,
but on these two captures `budget` will never fire at any capacity.

What binds at 16 is what binds at 8: **the population the mechanism can reach**,
which is 62% of the batches that exist, and §4 is why.

---

## 4. The number that is larger than the capacity, and it is in your own table

**56.4% of deferrals hold NOTHING, and no decline is counted for them.** The
only decline at V=2 is `full:4096`, so those 8,422 deferrals never saw an
admission decision at all: the tail was resolved before an appendable blended
batch arrived.

Two of the sweep's numbers say exactly which drains those are:

* P1 has **4,223 gaps with ≥9 appendable batches**; the sweep measures **4,096
  deferrals at the cap of 8**. That is 97% agreement. **The mechanism reaches
  the big gaps essentially perfectly.**
* P1 has **7,321 gaps with one or two** appendable batches; the sweep measures
  **8,422 deferrals holding nothing**. **The mechanism reaches almost none of
  the small gaps.**

So the missing 38% of reach is not spread evenly — it is the entire
one-and-two-batch population, which is 49% of all gaps. Capacity cannot touch
it. My reading of the call graph says why, and it is a structural boundary
rather than a bug:

* `cp_render_scope_begin()` and `cp_render_scope_end()` each call
  `cp_batch_flush_why()` — which calls `cp_pass_finish()`, deferring the tail —
  and then call `cp_pass_finish()` **again**, which resolves it with nothing
  held. A gap whose next episode begins in a NEW RENDER SCOPE therefore always
  holds zero;
* and it must, because the admission rule refuses a different framebuffer:
  `cp_abuf_setup()` would resize the very arrays the deferred tail is still
  reading. **Run-ahead cannot cross a render-scope boundary. That is not a
  tuning constant, it is the mechanism's outer edge.**
* a gap of one or two appendable batches is what the tail of a render pass looks
  like; a gap of fifty is a mid-pass run. The two populations in the sweep's
  bimodal histogram are, on this reading, "the pass ended" and "the pass
  continued".

**This is a hypothesis about a call graph, not a measurement**, and it is worth
one cheap probe before anyone spends 153 MB on capacity: split the resolve
counter by call site — `pass_finish` / `opaque_append` / `draw_execute` /
admission-refusal — and count deferrals resolved with `nheld == 0` by the same
key. Six counters, no behaviour change, one run per capture. It converts
"56.4% hold nothing" into a named cause, and it distinguishes the scope-boundary
reading above from the other live possibility: **that the next thing after
those drains is an OPAQUE episode**, which is the design's Tier 3 — named,
never designed, and a strictly easier mechanism, because an opaque run uses
`visbuf` and has no A-buffer count to hold at all.

---

## 5. What I would do with this

**Item 2 is capacity-bound, and the capacity is bounded by a resource rather
than by a law — but raising it does not reach a measurable result.** Ranked:

1. **Do not raise the capacity to 16.** +0.10 ms/frame is another run inside the
   session spread, and it costs either 153 MB of permanent device memory (form
   a) or a coin flip around zero (form b).
2. **Run the six-counter resolve-reason probe.** It is the cheapest thing left
   in this item and it decides between three different futures: a scope
   boundary (item 2 closes at 0.068 ms/frame, honestly measured and inside the
   spread), an opaque successor (Tier 3 becomes the lead, and it is easier than
   what I built), and something else.
3. **If and only if the probe says the reach is fixable**, revisit capacity —
   with form (a), 32 sets, and the arithmetic in §3.3 re-derived on whatever the
   probe says the population is.
4. **Record what is already true, whatever happens next.** The mechanism
   converts relocated host work into frame time at **1.01**, measured on this
   binary at four admission settings, with stray launches 0 and drops 0 at every
   one including the setting it was not designed for. That is the first
   end-to-end confirmation that P0's absorbed region is usable by a real
   mechanism and not only by a spin loop. The reason item 2 does not pay is
   **quantity, not conversion, and not correctness** — and that is a different
   sentence from the one this item started with.

---

## 6. Registered, so §3's arithmetic can be wrong out loud

* **C1.** Capacity 16 with dedicated queue sets moves **0.09 to 0.12 ms/frame**
  of true vertex work and produces a frame delta **inside 0.119**. If it
  produces more than +0.15, §3.2's uniform model of the ≥16 bucket is wrong and
  the tail is far heavier than 52.4 — which would make capacity 32 worth
  measuring after all.
* **C2.** `budget` still never fires at capacity 16, 32 or 64.
* **C3.** The resolve-reason probe attributes **more than half** of the
  zero-hold deferrals to a render-scope boundary or an opaque successor. If it
  attributes them to something inside `cp_pass_append`, I have misread the call
  graph and the reach is recoverable without any capacity change at all — which
  would be the best outcome available in this item.
