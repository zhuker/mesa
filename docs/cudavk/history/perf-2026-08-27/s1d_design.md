# CLOSURE REVISED, 2026-08-26 (later the same day)

**S1d stays closed — but on cost and mechanism, not on value, and its value was
understated by about a factor of nine. One sentence in the closure below is
wrong and is retracted here.**

The delay-injection probe measured the episode drain's conversion directly:
**slope +1.0207**, linear over an injected 0 → 1.976 ms/frame, residuals under
0.031, with the peel site as a control returning **−0.0262**. Host time at the
drain is on the critical path, essentially one for one.

### The retracted sentence

> *"Blocked time does not convert to frame time one-for-one at these sites, so a
> 1.05 ms ceiling buys well under 1.05 ms."*

**That is false at this site.** It generalised the 11% inferred at the peel site,
and that figure does not exist — direct injection shows adding 2.05 ms/frame of
host time at peel costs nothing, so the rejected peel patch's +0.110 ms was its
mechanism and never its blocking. At a conversion of 1.02, **S1d's 1.05 ms/frame
ceiling on old is worth up to about 1.07 ms of frame**, not the ~0.12 ms my
sentence implied. That is **2.4× PDL's entire measured win of 0.434 ms**, and
about 8% of the frame. I sized it with a borrowed constant and the constant was
not real.

### What survives, item by item

Each of these was stated in the closure below and none of them used the
conversion. They are what keeps S1d closed.

| # | objection | depends on the conversion? | verdict |
|---|---|---|---|
| 1 | **P4's structural ceiling.** Moving the drain to the scan reaches 1,586 ms of 9,007 ms of drain wait = **17.6%**, because 88.5% of an episode's device time sits *after* the scan | **no** — it is where the work sits | **STANDS**, but see below: 17.6% of the largest wait is now a large number, not a small one |
| 2 | **Memory.** `4 × total` **CROSSES** the arena on Crossroads, 607.3 MB projected against 506.6 MB held | no | **STANDS**, and is weaker than I implied — see below |
| 3 | **The packed form is dead.** Slots are `4q + lane` because that *is* the derivative quad (`driver_facts.md` F1) | no | **STANDS, and it is the one that hurts** |
| 4 | **The `capacity/4` gate** refuses **25.3%** of old episodes at the wider bound, and they are the large ones | no | **STANDS** |
| 5 | **The lost fail-closed checks** — `running != quads` and the fill/count disagreement | no | **STANDS** |

### Two of those are weaker than I wrote them, and honesty requires saying so

**Objection 1 inverts in tone.** I used 17.6% as evidence that the mechanism
"moves the drain a short distance". It does — but 17.6% of the driver's largest
wait, at a conversion of 1.02, is **1.05 ms/frame on old**. The share is small;
the quantity is not. Anyone re-reading the closure below should read §1's
"the mechanism was sound; the arithmetic was not" as *the arithmetic about where
the work sits was right, and the arithmetic about what it was worth was wrong*.

**Objection 2 is capture-asymmetric in the inconvenient direction.** The drop-in
form **FITS** on old — 1,069.3 MB projected against 7,914.8 MB held, 0 regrows
observed — and old is where the 1.05 ms is. It **CROSSES** on Crossroads, where
only 0.16 ms is at stake. So memory does not refuse the mechanism where the
value is; it refuses it where the value is not. What refuses it on old is
objection 4, the `capacity/4` gate — and that gate is load-bearing, because it
is what caps the over-allocation cost that killed S0.

**Objections 3, 4 and 5 are what actually hold the closure**, and 3 is
architectural rather than economic: the only form of S1d that avoids the
over-allocation entirely requires renumbering shade slots dense over fragments,
and that would break `ddx`/`ddy` and implicit-LOD sampling for every shader that
uses either.

### The conclusion, stated precisely

**S1d stays closed. It is closed on mechanism and cost — a dead packed form, a
load-bearing admission gate refusing a quarter of the episodes that matter, and
two fail-closed safety checks that would have to be demoted — and it is no
longer closed on value. Its value was understated by about a factor of nine.**

### What survives at 1.05 ms/frame that is worth a fresh design — and it is not S1d

Naming it, not designing it, and it should not be reopened on size alone.

**The deferral family at the drain, and specifically `overlap_survey.md`'s C2:
issue the *next* episode's vertex stage across the drain.** It is the only
member of that family that does not inherit the disease which refuted S0 and
S1b, because it needs **one more VS output buffer, not a second A-buffer** —
S0 and S1b were refused at 250 MB–1.6 GB against an 8.59 GB cap on a grow path
that turns pressure into per-flush `cuMemAlloc`/`cuMemFree`. VS main is 1.843
ms/frame plus 0.350 of `vertex_fetch`, i.e. **2.19 ms of kernel time sitting
behind a host round trip it does not depend on**, against a measured site
ceiling of 2.066 ms/frame at a conversion of 1.02.

Two things must be checked before anything is built, both readable from the
source: the scratch arena is a single bump allocator and `cp_scratch_reset()`
frees overflow arenas at every flush, so the next episode's VS output has to
survive the current episode's flush — **that, not the concurrency, is the
blocker**; and the add-direction caveat means a mechanism still has to be built
and measured before 2.07 becomes a win.

**And the general point this leaves for the record:** S0 and S1b were both
refused on memory *against a value nobody had measured*. That value is now
measured at up to 2.07 ms/frame. The refusals stand — the memory cost is real —
but "too expensive for an unknown return" and "too expensive for 2 ms" are
different judgements, and only the second one has been made.

---

# S1d — CLOSED BY MEASUREMENT

**Status: designed, measured, closed. Do not build it.** The probes in §7 were
written, run on both captures, and returned this:

```
old    moving the drain to the scan can remove at most 1586.314 ms of the
       9007.529 ms measured here (17.6%); the chain after the scan is 88.5%
       of the episode's device time
cross  moving the drain to the scan can remove at most  246.315 ms of the
       2698.135 ms measured here  (9.1%); the chain after the scan is 71.5%
       of the episode's device time
```

Per frame that is a **1.05 ms ceiling on old and 0.16 ms on Crossroads**, and
it is a ceiling that assumes perfect overlap. It is not worth the change.

**The mechanism was sound; the arithmetic was not.** `counts->scan` is only
2.3% (old) and 3.6% (Crossroads) of the drain, so almost everything the host
waits for is issued *after* the scan — the drain moves a short distance, not
because counting is slow but because the chain in front of it is short. §1's
table was right about what moves and wrong about how much of the wait it
carries. That is exactly what P4 was written to find out.

**P3 confirmed the sizing correction from a third direction.** Projected arena
**x3.73** (old) and **x3.85** (Crossroads) against the 3.75x predicted in §3.1
from `total/quads = 3.753`. Old **FITS** (1069.3 MB against 7914.8 MB held) but
the `capacity/4` gate of §4 refuses **25.3%** of episodes; Crossroads
**CROSSES** (607.3 MB projected against 506.6 MB held) at a 0.5% refusal rate.
So the drop-in form is dead on Crossroads by the S0 mechanism and crippled on
old by the driver's own cap — both as §4 predicted, now measured.

**And the conversion factor has since been priced.** The peel-predication patch
blocked **1.01 ms/frame more** and the frame did not move. Blocked host time
does not convert to frame time one-for-one at these sites, so a 1.05 ms ceiling
buys well under 1.05 ms. §7 F3's asymmetry stands and is now the general rule
for this driver, not a caveat about one site.

**Cross-validation.** The episodes this probe drops (1,033 old, 1,528
Crossroads) are exactly the measurer's independent P1 `quads == 0` empty
drains, and the time unaccounted for is 0.9% / 15.1% — matching his figures to
a tenth of a percent. Two probes, two authors, one number.

### What survives, and is worth keeping

* **§2.2 stands as a result independent of S1d:** `quad_over` is redundant at
  the tail drain whenever `fill_over == 0`, because `quad_capacity == capacity`
  and `quads <= total`. That is the driver's own comment, proved.
* **§2.3 stands, and stays closed:** `quads == 0` is exactly `total == 0`, and
  it is only free if the drain moves. It is **not** separately harvestable and
  there is no four-line version of it.
* **§5's packed numbering is WITHDRAWN, and not merely unmotivated.** I claimed
  a second, independent payoff of ~6% of shaded lanes. **That claim is wrong,
  and checking the next lead is what disproved it.** The 4-lanes-per-quad
  layout is not slack: it is the helper-lane quad that derivatives are taken
  across. `cp_nir_to_llvm.c:2345-2374` (`cp_quad_derivative`) computes `ddx`
  and `ddy` with `shfl.sync.bfly` over `tid & 28`, i.e. it assumes lanes
  `4q..4q+3` are the four pixels of one 2x2 block — exactly the numbering §5
  proposed to destroy. Implicit-LOD texture sampling is expressed through those
  gradients (*"CUDA texture instructions do not infer graphics fragment
  derivatives"*, `:2379`), and the driver states the rule itself at `:3393-3397`:
  *"A quad is shaded whole so that derivatives can be taken across it, so a
  lane the primitive does not cover is shaded too."* The uncovered 6.2% are
  **helper lanes and are architecturally required.** Packing would break
  derivatives and implicit mip selection for every shader that uses either.
  It could only ever apply to shaders provably using neither, which is a much
  narrower and more fragile claim than the one I made. Do not build it.
* **§7 F4's LOST items** never have to be traded away now. Good.

The rest of this document is the design as it was written before the
measurement, kept because the reasoning is what produced the probe.

---

# S1d — take the episode drain at the scan, and what it costs to size from it

**Design only. No code written. No GPU used.** Line numbers are
`e2fea470d04` (`git show e2fea470d04:src/cudavk/cp_renderer.c`), which is HEAD
for this worktree. Kernel citations are `src/cudavk/kernels/cp_rasterize.cu`
and `cp_fs.cu` at the same commit.

**Verdict up front, because it changes what should be built:**

1. The drain **can** move from the tail (`:8497`) to just after the scan
   (`:8369`). Every host decision it feeds is available there, and two of them
   become *safer* rather than merely earlier. §2 proves each one.
2. The 94% figure in the brief is **utilisation, not an allocatable size.**
   `total / (4 x quads) = 3.753/4 = 0.938` is the fraction of today's shade
   slots that carry a real fragment. Substituting `total` for `quads` as the
   host's *bound* costs `4 x total = 3.75x` today's allocation, not 0.94x,
   because the shade slot index is `4*q + lane` and is dense over quads, not
   over fragments (`cp_fs.cu:310`, `cp_rasterize.cu:1970-1972`). §3.
3. That 3.75x is **refused by the driver's own over-allocation cap**, not by
   my judgement: `cp_pass_finish_bounded_groups` gates on
   `quad_bound > ab->capacity / 4` (`:8080`), and with `CP_ABUF_HEADROOM = 2`
   (`:1121`) that gate is calibrated for the true quad count. §4.
4. So **S1d is worth building only in its packed form**, where the shade slots
   are numbered dense over fragments. That form allocates `total` slots — 94%
   of today, the brief's number, now correctly earned — passes the existing cap
   with room to spare, and removes about 6% of shaded lanes as a side effect.
   It is a device-ABI change across four kernels. §5.
5. One thing falls out for free either way and should be taken first: the
   `total == 0` early-out is **exactly** the `quads == 0` early-out
   (§2.3), which is the lead that died last round. At the scan it is sound.

---

## 1. Where the wait is, and what moving it buys

`cp_pass_finish()` (`:8313`) issues, in order:

| # | step | line |
|---|---|---|
| 1 | `cp_pass_join` — the segments' count phases | `:8359` |
| 2 | **`cp_abuf_scan`** — counts to offsets, clamp, grand total | `:8369` |
| 3 | fill (`abuf_fill_recs`, or one relaunch per segment) | `:8373-8412` |
| 4 | sort, both worklists | `:8416-8449` |
| 5 | `cp_abuf_quad_build` — quad count, scan, quad fill | `:8456` |
| 6 | `cp_pass_finish_bounded_groups` — returns false today | `:8460` |
| 7 | `abuf_seg_count` — bucket quads by segment | `:8490` |
| 8 | **the drain** `cp_sync_timed(... wait_episode_ns ...)` | `:8497` |
| 9 | copy 6 counters + nsegs quad counts | `:8500` |
| 10 | overflow verdict, group merge, dense bases, allocations, shade, composite | `:8507-8800` |

The drain at step 8 waits for **steps 1-7**. A drain at step 2 waits for
**step 1 only**. Steps 3-7 are the fill, the sort and the quad build — the
expensive part of the episode. That difference is the whole proposal.

At HEAD the episode drain is **5.989 ms/frame over 9.88 waits** (0.606 ms
each), the largest remaining wait in the driver.

Note what is *not* being claimed: this does not remove device work, and
§1.2 of the iteration-29 design still applies — a removed wait is worth its
blocked time only to the extent the host has other work to issue during it.
That was P2, and it has now been **measured**: 306 us of inter-drain issue
burst on old against a 606 us wait, and 139 us on Crossroads. §7 F3 has the
table and what it means. **The ceiling here is the reachable device idle, not
the 5.989 ms**, and on Crossroads that idle is under 1 ms in total.

---

## 2. Every host decision the tail drain feeds, and where it can be taken

The drain copies `ctr[0..5]` plus `ctr[6 + s]` per segment. The counter block
is one allocation (`cp_renderer.h:844-852`, pointers at `:1288-1295`):

```
ctr[0] = sum3[0]  total fragments        ctr[3] = bsum3[0] quads
ctr[1] = sum3[1]  fill overflow          ctr[4] = bsum3[1] quad overflow
ctr[2] = sum3[2]  long runs              ctr[5] = clist_count covered pixels
ctr[6+s]          per-segment quad count
```

`ab->overflow == ab->sum3 + 4 == &ctr[1]` (`:1292`). `ab->quad_overflow ==
ab->bsum3 + 4 == &ctr[4]` (`:1294`).

### 2.1 D1 — growth bookkeeping (`:8510-8514`). Available, same word.

`ab->peak` and `ab->grow_to` are read from `total = ctr[0]`, which
`cp_abuf_scan` itself writes into `ab->sum3[0]`. It affects only *future*
draws. No change beyond reading it earlier.

### 2.2 D2 — the overflow verdict (`:8516`). Available, and provably complete.

This is the one that must be right, because `cp_pass_can_retry` (`:7326-7340`)
latches device loss when a retry follows an FS attempt, so **the fallback
decision must precede any FS launch.** Today it does, by one drain. Under S1d
it precedes it by more.

`ab->overflow` (`fill_over`) has exactly three writers:

* **A — the count pass.** `cp_rasterize.cu:669`: a record whose slot is past
  `abuf_capacity` is dropped and counted. The count phases are joined at
  `:8360`, **before** the scan. This contribution is final at the scan.
* **B — the scan's own clamp.** `cp_abuf_scan_add` (`cp_rasterize.cu:1444-1452`)
  and `cp_abuf_scan_finish` (`:1576-1585`): when `total > capacity`, a run that
  cannot fit is cut and the difference is added. This contribution is written
  *by* the scan and is final when it returns.
* **C — the fill.** `cp_abuf_fill_recs` (`cp_rasterize.cu:1678-1681`) and
  `emit_fragment` in fill mode (`:683`): one per record that cannot be placed,
  i.e. `slot >= counts[p]` or `base + slot >= capacity`.

**Claim: C can only fire if A or B fired.** For the records path: the count
pass does `atomicAdd(counts + p, 1)` for every emitting lane and appends a
record for the same lane, dropping it only when A fires. `nrec` is clamped to
`rec_capacity`, which again only drops records A already counted. So the number
of records replayed for pixel `p` is **at most** `counts[p]`, and if B did not
clamp then `offsets` is the exact exclusive prefix, so slots run
`0 .. records_for_p - 1 < counts[p]` and `base + slot < total <= capacity`.
Neither branch of C can be taken.

Two consequences and one restriction:

* **`quad_over` is impossible when `fill_over == 0`.** `ab->quad_capacity ==
  ab->capacity` — both are set to the same `want` at `:1579` and `:1601` — and
  the driver's own comment at `:1580-1585` supplies the proof: *"A quad needs
  at least one covering fragment, so there can never be more quads than
  fragments."* So `quads <= total <= capacity == quad_capacity` and
  `abuf_quad_fill` cannot overflow. The tail drain's `quad_over` term is
  therefore **redundant given `fill_over == 0`**, and reading it later than the
  verdict costs nothing.
* **The relaunch fill is different and must be excluded.** On the
  `CUDAVK_NO_ABUF_APPEND` path the fill re-rasterizes, and C is a genuine
  fill-versus-count *consistency check* — the comment at
  `cp_rasterize.cu:676-678` says so: *"so that a fill that disagrees with the
  count is reported rather than allowed to write over the next pixel's
  fragments."* That check has no earlier equivalent. **S1d must refuse the
  episode when `ab->recs` is null.** One condition, and the flag that produces
  that state is already a revert flag.
* The consistency check does not disappear on the records path either; it
  simply cannot fire without A or B. Keep the counter, and report a nonzero
  late arrival loudly at the next natural synchronisation. It is a bug
  detector, not a capacity event, and it must not silently become unchecked.

### 2.3 D3 — `!quads` return (`:8525-8526`). Available, and exactly equivalent.

`quads == 0 <=> total == 0`. Forward: a quad is emitted per distinct primitive
in a block's four pixel lists (`cp_rasterize.cu:1899-1996`), so no fragments
means no quads. Backward: any fragment gives its block at least one distinct
primitive, hence at least one quad. So `if (!total) return;` at the scan is the
same statement as `if (!quads) return;` at the tail, taken 4-7 steps earlier.

**This is the `quads == 0` lead, resurrected.** Last round it died because the
zero was only knowable at a drain that had to happen anyway for the overflow
verdict. Under S1d the drain moves, the overflow verdict moves with it, and the
zero arrives at the same moment for free. It is worth **6.9% of old-capture
drains and 18.3% of Crossroads drains** skipping steps 3-10 entirely — and
those drains are *also* the cheapest ones to skip, so do not expect their share
of the 5.989 ms.

**It is not separately harvestable, and nobody should reopen it as one.** The
early-out exists only because the drain has moved. At the tail it is what it
was last round: the host is already past the fill, the sort and the quad build
by the time it learns the episode was empty, and it had to drain there anyway
for the overflow verdict. There is no four-line version of this. It is a
consequence of §2.2, not a change of its own.

### 2.4 D4 — the shading-group merge (`:8543-8563`). Already host-only.

It reads `batch.state.vs`, `.fs`, `.num_fs_ubos` and `.info.mode`. Never a
device word. `cp_pass_finish_bounded_groups` runs the identical loop at
`:8086-8102`.

### 2.5 D5 — dense bases (`:8566-8581`). Already implemented on the device.

`seg_base_host[]`, `group_base[]`, `group_quads[]` are a prefix over the
per-segment quad counts. `abuf_seg_prefix` computes exactly this on the device
(`cp_rasterize.cu:2560-2594`), and `cp_pass_finish_bounded_groups` already
consumes it through `group_base_dev` / `group_counts_dev` (`:8134-8166`,
`:8173-8174`). **Reused unchanged.**

What *is* lost is the `running != quads` consistency test at `:8579-8582`.
That is a check on the device prefix against the device total, and the device
can do it itself — one comparison in `abuf_seg_prefix`, reported into a sticky
word. Say so in the patch; do not drop it silently.

### 2.6 D6 — the allocation sizes (`:8583-8600`, and `cp_abuf_shade` `:4182`).

This is the only decision that genuinely needs a device-produced number, and it is the subject
of §3 and §4.

---

## 3. What sizing from `total` actually costs

### 3.1 The correction

The shade arrays are indexed by **slot**, and the slot of quad `q`'s lane `i`
is `4*q + i`:

```
cp_fs.cu:310-311     uint32_t base = iq * 4u;
cp_fs.cu:322-336     ... coverage[base + i] = (covered && ok) ? 1u : 0u;
cp_rasterize.cu:1970 * cp_abuf_interpolate gives quad q the four shading slots
                     *   4q..4q+3, lane i being the block's pixel i.
cp_renderer.c:4182   size_t want_slots = (size_t)num_quads * 4;
cp_rasterize.cu:2601 *args.slots = 4u * *args.count;   (the device-read extent)
```

So the array must have `4 x quads` slots. The maximum slot index is
`4*quads - 1`; nothing is packed.

`total` is a sound upper bound on `quads` — that is the driver's own proof at
`:1580-1585` — so the host substitute for `quads` is `total`, and the slot
count becomes `4 x total`.

```
today                      4 x quads
size from total            4 x total  =  4 x 3.753 x quads  =  3.75x today
                           bounded at 4.00x by geometry (a quad is 2x2)
brief's 0.94x              total / (4 x quads)  —  utilisation of today's
                           arrays, achievable only by renumbering (§5)
```

`total/quads` median 3.753 (old) and 3.725 (Crossroads), max 3.99 over 20,704
episodes. The distribution is **tight against its geometric cap**, so this is
not a median-versus-tail problem: it is ~3.75x essentially always.

### 3.2 What that costs in bytes

Per slot, from `cp_abuf_shade` (`:4190-4203`): `pixel_list` 4 B, `fs_in`
`fs_in_stride`, `fs_out` `fs_out_stride`, `coverage` 1 B, `frag_coord` 16 B,
plus `discard_mask` and `front_face` at 1 B when the shader uses them. The
header's own estimate is *"about 90 bytes a slot"* at five varyings
(`cp_renderer.h:725-727`).

So the extra footprint is about `2.75 x 4 x quads x 90 B` = **990 B per quad of
the episode**, on the arena that `cp_scratch_alloc_device` grows by
reallocating the whole thing and freeing the old base at every flush
(iteration-29 addendum §1). That mechanism is what made the S0 probe a
regression. 3.75x is four orders of magnitude better than S0's bound, but it is
not free, and **nothing in this session knows the absolute per-episode `quads`,
so nothing here knows whether 2.75x extra fits.** That is P3 in §7.

---

## 4. The driver already refuses 3.75x, and it is right to

`cp_pass_finish_bounded_groups` gates its bound at `:8079-8082`:

```c
if (!quad_bound || quad_bound > ab->quad_capacity ||
    quad_bound > ab->capacity / 4u ||
    quad_bound > CP_ABUF_MAX_SHADE_SLOTS / 4u)
   return false;
```

The second and third terms are both `4 * quad_bound <= budget`: `capacity` for
the arena, `CP_ABUF_MAX_SHADE_SLOTS` (16 M, `cp_renderer.h:728`) for the
absolute cap. Now put the numbers in:

* `capacity` is `MAX2(2 x total + 65536, 5 M)` — `CP_ABUF_HEADROOM = 2`,
  `CP_ABUF_SLACK = 65536`, `CP_ABUF_MIN_FRAGS = 5 M` (`:1121-1122`,
  `:1087-1088`), sized from the **peak** total, regrown when a count exceeds
  `0.75 x capacity` (`CP_ABUF_GROW_AT`).
* With the **true** quad count: `4 x quads = 4 x total/3.753 = 1.066 x total`,
  against `capacity >= 2 x total`. Passes with room. **The gate is calibrated
  for the true count.**
* With `quad_bound = total`: `4 x total` against `capacity >= 2 x total`.
  **Fails whenever `total > capacity/4`**, which for a steady-state episode
  (`total <= 0.75 x capacity`, `capacity >= 2 x total`) is most of them, and is
  *certainly* the large ones — which are exactly the episodes carrying the
  0.606 ms wait.

So the drop-in form of S1d is refused by the driver's own over-allocation cap,
for the episodes it was meant to help. **Do not raise that cap.** It is the
same class of constant as the `512u << 10` slot budget the addendum withdrew,
and the S0 probe already measured what exceeding an arena budget does.

---

## 5. The form that works: pack the shade slots over fragments

Number the shade slots densely over **fragments** instead of over
quads-times-four. Then:

* the episode's slot count is exactly `total` — the number `cp_abuf_scan`
  already leaves in `ab->sum3[0]`, at the moment S1d drains;
* it is **0.94x today's allocation**, not 3.75x: the brief's number, now
  earned;
* the `capacity/4` gate becomes `total <= capacity`, which holds whenever
  `fill_over == 0` — i.e. the gate is subsumed by the overflow verdict and
  stops being a separate refusal;
* about **6% of shaded lanes disappear**, because today every quad shades four
  lanes and masks the uncovered ones with `coverage[]` (`cp_fs.cu:322-336`);
* the over-allocation ratio, which the iteration-29 addendum identified as
  *the* expensive quantity, becomes **1.00 exactly**. Not a bound — a count.

### 5.1 What has to change

Four kernels and one host arithmetic block. Every piece has an existing
same-shape precedent in the tree, which is the reason to think it is tractable:

| # | change | precedent |
|---|---|---|
| P1 | the quad build also accumulates **fragments per block** — the sum of `counts[p]` over the block's four pixels, which `cp_abuf_quad_count_all` already loads — and scans them exactly as it scans `blk_counts` into `blk_offsets` | `cp_abuf_scan_finish_only` over `nblocks`, `cp_renderer.c:2055-2059` |
| P2 | `cp_abuf_merge_block` gives quad `at` the base `blk_frag_offset[b] + (running popcount within the block)`, which it can accumulate locally because it already emits a block's quads in order; `out_shade_slot[...] = base + popcount(mask & ((1<<i)-1))` | it already computes `at*4 + i` at `cp_rasterize.cu:1971-1978` and already has `mask` |
| P3 | `cp_abuf_interpolate_body`: `base = iq*4` becomes the quad's packed base, and the lane loop visits only covered lanes | `cp_fs.cu:310-336` |
| P4 | `cp_abuf_prepare_shade_count` writes the group's **fragment** count, not `4 x` its quad count | `cp_rasterize.cu:2597-2603` |
| P5 | the per-segment `descs[s].fs_out = ss.fs_out + off * stride` with `off = (seg_base_host[s] - group_base[g]) * 4` becomes a **fragment** offset | `:8700-8712`, and `abuf_seg_prefix` already prefixes per group |

### 5.2 The hard part, stated as the hard part

**P5 is the risk.** The shade is launched per *group* over a slice of the
grouped quad list, and the packed base has to be dense **within the group**,
not episode-global. `abuf_seg_prefix` already produces per-group quad bases; it
would have to produce per-group *fragment* bases too, which means
`abuf_seg_count` must accumulate fragments per segment as well as quads — it
has the quad's `mask` in hand, so this is a popcount, not a second pass.

If that turns out not to be expressible in the existing prefix kernel, packing
is a much larger change than it looks and **S1d should be dropped rather than
forced.** The tail drain pays for itself (addendum §1), and 0.606 ms of blocked
host is not worth a device-ABI rewrite whose ceiling is unmeasured.

---

## 6. What still has to be waited for

After S1d, per episode:

| | before | after |
|---|---|---|
| drains | 1, at step 8 | 1, at step 2 |
| chain waited on | count + scan + fill + sort + quad build + bucket | count + scan |
| words copied | 6 + nsegs | 2 (`total`, `fill_over`) |
| overflow verdict | after step 7 | after step 2 — **earlier**, so `cp_pass_can_retry`'s FS-ordering rule is respected with a wider margin, not a narrower one |
| what is issued after the drain | steps 10 only | steps 3-10, one unbroken burst |

Nothing is deferred past a flush, no second arena is introduced, no allocation
is doubled. This is deliberately **not** S1b, which the addendum demoted below
its own probe.

---

## 7. Failure modes, and the probes that decide them before code

**F1 — the fill overflows without the scan having clamped.** §2.2 proves it
cannot on the records path. If it ever did, the host would have committed a
size and possibly launched an FS before finding out, and `cp_pass_can_retry`
would latch device loss. Mitigation: refuse the episode when `ab->recs` is
null; keep the counter and report a late nonzero loudly; add a negative-control
flag that forces a too-small capacity and requires the flag to fire and the
output still to match, in the iteration-24 style.

**F2 — over-allocation costs more than the wait.** The exact disease of S0 and
of the clip-rectangle lead. In the drop-in form the factor is 3.75x; in the
packed form it is 1.00x. This is the reason §5 is the recommendation.

**F3 — does the host have work to issue during the removed wait? MEASURED,
and the answer is an asymmetry, not a number.** This was the addendum's P2 and
it has now been run:

| | old | Crossroads |
|---|---:|---:|
| median inter-drain host issue burst | **306 us** | **139 us** |
| median episode-drain wait | 606 us | — |
| total host issue | 3.478 ms/frame | — |
| device busy | 72% | — |
| frame floor from the device | ~9.5 ms | — |
| reachable idle | — | **under 1 ms** |

So on **old** the host does hold queued work — roughly half a wait's worth per
drain — and the addendum's "tens of microseconds, therefore dead in every form"
branch does **not** fire. On **Crossroads** there is under 1 ms of reachable
idle in total, which is the same fact that made S0 a regression there with 8.26
waits a frame removed.

**Restate the risk as the asymmetry, not as the wait size.** The wait is
similar on both captures; the host's ability to spend it is not. S1d pays on
old and is expected to pay little or nothing on Crossroads, and a measurement
that reports one number across both captures will hide that. Two hard caps
survive and must be carried in any estimate: total host issue is 3.478 ms/frame,
and at 72% device busy the frame cannot go below about 9.5 ms.

**F4 — the lost consistency tests.** `running != quads` (§2.5) and the fill's
disagreement counter (§2.2) both currently fail *closed* into
`cp_pass_fallback`. Under S1d they have to be reproduced on the device or
demoted to loud reports. Demoting a fail-closed test to a report is a real
reduction in safety and must be written down in the patch, not buried.

### The two probes — WRITTEN, and ready to run

Committed on branch `bounded-clip` in `/tmp/bnd-tree`, commit `5c8eb0428e3`.
Instrumentation only: no mechanism, no allocation change, no device ABI change,
no new synchronisation. Both default off. Both buffer their samples and report
at teardown — an `fprintf` between the drain and the shade would be timed as
part of the thing being measured.

| flag | probe | what it decides |
|---|---|---|
| `CUDAVK_EPISODE_SIZE_STATS` | P3 | drop-in versus packed |
| `CUDAVK_EPISODE_WAIT_SPLIT` | P4 | whether either form is worth anything |

They are independent and can be set together; the report prints one block per
flag.

#### What P3 records and prints

Per episode, at the existing tail drain where `total` and `quads` are both in
hand, and closed after the last group's shade arrays are allocated:
`total`, `quads`, `ab->capacity`, `nsegs`, `ngroups`, `cp->dscratch.used`
before and after the shade, `cp->dscratch.size`, and
`cp->dscratch.num_overflow` (which is the device arena's regrow count — the S0
mechanism's own counter).

At teardown:

```
cudavk: episode sizing: N episodes (M dropped before the shade)
  total/quads: median X p90 X max X (4.000 is the geometric cap)
  slots today   (4*quads): median .. p90 .. max ..
  slots at bound (4*total): median .. p90 .. max ..  (cap 16777216)
  episode device scratch: median .. MB p90 .. MB max .. MB
  device arena: peak today .. MB, projected at the bound .. MB (x..),
                arena .. MB, cap 8192.0 MB
  VERDICT (size): the wider bound FITS inside / CROSSES the arena
                  (.. MB projected against .. MB held, N regrows observed)
  capacity/4 gate: K of N episodes (P%) have total > capacity/4 and would be
                   refused at the wider bound
```

The projection scales the whole post-drain scratch delta by `total/quads`. That
over-states it slightly — `quad_seg` is sized from `quad_capacity` and does not
scale — so the error is on the safe side. **CROSSES means the drop-in form is
dead by the S0 mechanism and §5 is the only route.**

#### What P4 records and prints

Three events on `cp->stream`: after `cp_pass_join` (the counts are joined),
after `cp_abuf_scan` (everything the proposed drain would wait for is on the
stream), and after `abuf_seg_count` (the last thing the present drain waits
for). They are read **after** the existing drain, which has already
synchronised the stream — so nothing waits and no `cuEventSynchronize` is
issued. An episode that leaves early records fewer than three and is dropped
rather than read against a previous episode's events.

At teardown:

```
cudavk: episode wait split: N episodes
  measured drain wait: median .. us p90 .. us max .. us, total .. ms
  device counts->scan:  median .. us p90 .. us, total .. ms
  device scan->bucket:  median .. us p90 .. us, total .. ms
  VERDICT (wait): moving the drain to the scan can remove at most X ms of the
                  Y ms measured here (Z%); the chain after the scan is W% of
                  the episode's device time
```

`X` is `sum over episodes of min(wait, scan->bucket)`. A drain at the scan
still waits for everything up to the scan, so what a move can recover is the
part of the wait the fill, sort, quad build and bucketing account for, and
never more than the wait itself. **That minimum is the ceiling, before any
question of whether the host can spend it (F3).**

#### The exact command

```sh
export VK_ICD_FILENAMES=/tmp/bnd-tree/build-cudavk-bnd/src/cudavk/cudavk_devenv_icd.x86_64.json

# both probes, one run per capture, stderr captured
CUDAVK_EPISODE_SIZE_STATS=1 CUDAVK_EPISODE_WAIT_SPLIT=1 \
   <replay old>        2> /tmp/perf-audit/probe_old.txt   | sha256sum
CUDAVK_EPISODE_SIZE_STATS=1 CUDAVK_EPISODE_WAIT_SPLIT=1 \
   <replay crossroads> 2> /tmp/perf-audit/probe_cross.txt | sha256sum

grep -A9 "episode sizing"     /tmp/perf-audit/probe_*.txt
grep -A5 "episode wait split" /tmp/perf-audit/probe_*.txt
```

Rules that apply to these runs specifically:

* **The stdout hash must match a flagless run of the same binary.** Neither
  probe touches rendering; a different hash is a run that died (WORKFLOW.md
  §4.4). Check the submit count too.
* `CUDAVK_PLAN_STATS=1` may be added and is useful — the probe's "measured
  drain wait" total should agree with `plan_stats`'s "episode drain X ms/N".
  If it does not, the probe is attributing the wait wrongly and its verdict is
  void.
* Do **not** set `CUDAVK_ABUFFER_TIMING` or `CUDAVK_ABUF_FUSE_CHECK`. Both
  synchronise, which changes the very quantity P4 measures.
* These are **not timed runs**. P4's three event records per episode are ~30 a
  frame and perturb the frame slightly; the probe reports its own quantities,
  not a frame time. Do not read a median frame time out of a probe run.
* The GPU must be idle, as always.

**P3 decides drop-in versus packed. P4 decides whether either is worth it.**
Run both on both captures in one session and read the two VERDICT lines.

## 8. Recommendation

* Build **nothing** until P3 and P4 report.
* Take the `total == 0` early-out (§2.3) first whatever they say: it is four
  lines, it needs the scan drain and nothing else, and it recovers the
  `quads == 0` lead that died last round.
* If P4 says the fill/sort/quad-build share of the wait is small, **close S1d**
  and record that the episode drain pays for itself, as the addendum already
  concluded for site 1.
* Whatever P3 and P4 say, judge S1d **per capture**. P2 measured 306 us of
  issue burst on old and 139 us on Crossroads, with under 1 ms of reachable
  idle on Crossroads at all. A single number averaged over the two hides the
  only thing that predicts where this pays.
* If P4 says it is large and P3 says `4 x total` fits, the drop-in form is a
  small patch and should be measured first, with the `capacity/4` gate left
  where it is and its refusals counted.
* If P4 says it is large and P3 says `4 x total` does not fit, the packed form
  (§5) is the only route, and it should be judged as what it is: a device-ABI
  change across four kernels, whose second, independent payoff is removing 6%
  of the fragment-shader lanes.
