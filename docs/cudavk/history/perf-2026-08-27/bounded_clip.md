# The clip-rectangle bound for the `bounded` fast path

Worktree: `/tmp/bnd-tree`, branch `bounded-clip`, commit `35425dcbcee`, based on
`e2fea470d04`. Build: `/tmp/bnd-tree/build-cudavk-bnd`, links
`src/cudavk/libvulkan_cudavk.so`. **Nothing here was measured. No GPU work was
done.** Another agent owns the card and owns the A/B.

**Every line number below is `e2fea470d04`'s, i.e. before this patch.** The
patch shifts `cp_renderer.c` by about +60 lines after 6400.

Flags added, both `CP_FLAG_BOOL_VALUE`, both **default off**:

| flag | what it does |
|---|---|
| `CUDAVK_ABUF_CLIP_BOUND` | bound the per-draw A-buffer fast path by the clip rectangle instead of by the whole framebuffer |
| `CUDAVK_ABUF_BOUND_ANY_TRIS` | drop the `num_triangles <= 2` condition from that path |

`src/cudavk/tests/cp_debug_doc.py --check` passes (`FLAGS.md` regenerated, 120
switches) and `src/cudavk/tests/cp_no_getenv.py` passes (36 sources, every
switch through the registry). No `getenv` was added anywhere.

---

## 1. Mechanism

### 1.1 What the path is

`cp_draw_execute_batch()` in `src/cudavk/cp_renderer.c` builds the A-buffer for
a blended draw, then has to decide whether the A-buffer *renders* the draw or
merely *describes* it. Deciding needs three device words — did the fill place
every fragment it counted, did the merge place every quad, are there any quads
— so the code drains the stream and copies them back. At HEAD that drain is
`cp_sync_timed(cp, cp->stream, &cp->plan.wait_seg_ns, ...)`, and on the fresh
numbers it costs **0.982 ms/frame over 4.17 waits, mean 0.2354 ms**. The
opaque fan-out that landed in `20611f5b131` did not touch it.

Directly above the drain there is already a wait-free arm, `bool bounded`. When
the host can *prove* neither array can overflow, the answer to all three
questions is known and the drain does not happen. The proof is two products:

```
fragments <= num_triangles      * pixels   <= ab->capacity
quads     <= rast_num_triangles * blocks   <= ab->quad_capacity
slots     =  4 * quads                     <= 512 KiB / 4
```

### 1.2 What was wrong with it

`pixels` was `n = w*h` and `blocks` was `ab->nblocks`, i.e. **the whole
framebuffer**: 921,600 and 230,400 at 1280x720. That is a true statement about
where a fragment can land, but it is the loosest one available. Against the
`512u << 10` byte slot budget, `230400 * tris * 4 <= 524288` admits **zero**
triangles at full-screen size. The arm fires only for the small render targets
of the bloom pyramid. Every scissored UI draw and every viewport-confined post
pass takes the drain, no matter how little of the screen it can touch.

### 1.3 What replaces it

The exact statement is already computed on the host, one screenful of code
higher up, and already handed to the rasterizer: `clip_x0..clip_y1` of
`struct cp_rasterize_args` — "the rectangle of pixels a fragment may land in",
framebuffer intersected with the viewport and, when `state->raster.scissor` is
set, the scissor. It is a function-scope local at the `bounded` site, computed
at `cp_renderer.c:4794-4816` and not reassigned in between.

With `CUDAVK_ABUF_CLIP_BOUND=1`:

```c
size_t bound_blocks = ab->nblocks;
size_t bound_pixels = n;
if (cp_debug->abuf_clip_bound) {
   if (clip_x1 >= clip_x0 && clip_y1 >= clip_y0) {
      size_t cb   = (clip_x1/2 - clip_x0/2 + 1) * (clip_y1/2 - clip_y0/2 + 1);
      size_t cpix = (clip_x1   - clip_x0   + 1) * (clip_y1   - clip_y0   + 1);
      bound_blocks = MIN2(cb,   (size_t)ab->nblocks);
      bound_pixels = MIN2(cpix, n);
   } else {
      bound_blocks = 0;      /* an empty rectangle draws nothing */
      bound_pixels = 0;
   }
}
```

Blocks are counted **by block index**, not by dividing the rectangle's width: a
two-pixel-wide rectangle straddling an even boundary touches two 2x2 blocks,
not one. `MIN2` against the framebuffer is belt-and-braces; the rectangle is
already clamped to `w-1`/`h-1` at construction.

`bound_blocks` and `bound_pixels` then replace `ab->nblocks` and `n` in all
three tests and in `abuf_quads`.

### 1.4 The composite grid, which is part of the same mechanism

The `bounded` arm used to set `abuf_covered = 0`, meaning "size the composite's
grid to the framebuffer". That was free when the arm only ever fired for small
render targets. It is **not** free once a scissored draw is admitted: the draw
would skip a 0.235 ms drain and then pay a 3,600-block composite grid for a
corner of the screen. So when the clip bound is in use the arm now passes
`bound_pixels` as the grid hint. This is safe for exactly the reason the
existing comment at `cp_renderer.c:4396-4399` gives: the kernel reads the
worklist's true length from the device and its tail threads return, so
`num_covered` "is a statement about how much of the machine to use rather than
a bound anything depends on."

### 1.5 The separately flagged sub-change

`num_triangles <= 2` predates the product tests and is subsumed by them. It is
its own switch, `CUDAVK_ABUF_BOUND_ANY_TRIS`, so it can be measured apart from
the clip bound — the two are expected to interact strongly (see §5), and a
single flag would hide which one paid.

Kept, deliberately, exactly as the design says:

* `cp->fs_batch.ndraws <= 1` — a garbage batch row would dereference a garbage
  table entry.
* `!state->fs->writes_memory` — the side-effect gate reads coverage the
  interpolator never wrote for slots past the total.
* `512u << 10` — **not raised.** The iteration-29 addendum measured what
  exceeding it does (gigabyte-scale `cuMemAlloc`/`cuMemFree` churn per flush,
  and an old-capture request of 8,605,856,768 bytes against an 8,589,934,592
  cap). That constant caps over-allocation cost. The proposal to raise it was
  withdrawn and is not in this patch.

---

## 2. Why it cannot change a pixel

Three independent arguments, in the order a reviewer should check them.

### 2.1 The bound is a bound

*Claim:* no fragment of this draw exists outside `clip_x0..clip_y1`.

Only two places in `cp_rasterize.cu` establish the pixel walk of a primitive,
and both clamp it to the clip rectangle:

```
491-494  point sprites:  ix_min = max(..., clip_x0);  ix_max = min(..., clip_x1);
546-549  triangles:      ix_min = max(..., clip_x0);  ix_max = min(..., clip_x1);
```

The batched case (`cp_rasterize.cu:452`) *replaces* `clip_*` with the
primitive's own rectangle from `clip_rects`. Those rectangles are built at
`cp_renderer.c:5197-5200` as `MAX2(clip_x0, scissor.minx)` etc. — intersections
of the batch-wide rectangle, so **subsets, never supersets**. The batch-wide
rectangle is therefore a valid bound in both cases.

*Claim:* quads are bounded by `rast_num_triangles * blocks_in_clip_rect`.

A quad is emitted by `cp_abuf_merge_block()` once per distinct primitive id
present in a block's four pixel lists (`cp_rasterize.cu:1899-1996`). So at most
one quad per (block, primitive), and a block only holds fragments, so only
blocks that intersect the clip rectangle can hold any. The block index is
`(y/2)*quad_width + (x/2)`, so the blocks that intersect the rectangle are
exactly those with `bx` in `[x0/2, x1/2]` and `by` in `[y0/2, y1/2]` — which is
what the code counts.

*Claim:* fragments are bounded by `num_triangles * pixels_in_clip_rect`.
Identical argument, with the unchanged existing note that clipping triangulates
each input polygon without multiplying covered pixels, so the input triangle
count is the right multiplier.

### 2.2 Even a wrong bound could only waste work, not corrupt output

The bound sizes launches. It is not read as a count by anything:

* the interpolator and the FS take their extent from `num_quads_dev = ab->bsum3`,
  the **device's own** quad total (`cp_renderer.c:4258`);
* the composite kernel takes its list length from `ab->clist_count`, on the
  device; `num_covered` is a grid size only (`cp_renderer.c:4401`);
* the merge itself is bounded on the device by `ab->quad_capacity` and reports
  a stream that outran its array in `ab->quad_overflow`, and the product test
  against `quad_capacity` is unchanged.

Over-estimating therefore costs slots and idle 256-thread blocks — under 0.2 ns
each, iteration 28 — and never a wrong pixel. **Under-estimating** would be a
bug, and it is the one thing §2.1 has to be right about; §6 says how to try to
break it.

### 2.3 Flag off is the same program

With both flags off, `bound_blocks == ab->nblocks` and `bound_pixels == n` and
`bound_tris_ok == (num_triangles <= 2)`, so every test and every assignment is
textually the old one.

This was checked at the machine-code level, not only by reading. Compiling
`cp_renderer.c` with `cp_debug->abuf_clip_bound` and
`cp_debug->abuf_bound_any_tris` textually replaced by `false`, and comparing
`objdump -d` against the object built from `HEAD`:

```
total instructions   29,983 (HEAD)   29,981 (flags forced off)
mnemonic deltas      nopl 364->361, xchg 37->36, cs 21->23
```

i.e. the two objects differ **only in nop padding**; every non-padding
instruction count is identical, and the 835 differing text lines are register
renaming and scheduling. Artefacts in `/tmp/bnd-proof/`.

That is a static argument. The runtime one is in §4: flag off must be
byte-identical by hash, and this is a driver where "a different hash is usually
a run that died" (WORKFLOW.md §4.4).

---

## 3. Expected admission-rate change, and how to confirm it from existing counters

**The counter already exists.** `CUDAVK_PLAN_STATS=1` prints

```
cudavk: main-thread waits: episode drain X ms/N, quad counters ..., peel checks ...,
        segment counters S ms/M, desc uploads ...
```

`M` is `cp->plan.wait_seg_n`, incremented **only** at the drain this patch
avoids (`cp_renderer.c:6420`). A draw that reaches the A-buffer either takes the
`bounded` arm or increments `M`. So

```
newly admitted draws = M(control) - M(candidate)
admission rate change = 1 - M(candidate)/M(control)
```

is exact, needs no new instrumentation, and needs no GPU timing to read. `S`
should fall in the same proportion if the admitted draws are typical, and by
less if the ones admitted first are the cheap ones.

Divide `M` by the frame count (or read `cp->plan.scopes`) to compare against the
baseline of **4.17 waits/frame**.

**What I expect, and what I explicitly do not claim.**

* Direction: `M` falls, `S` falls, monotonically, with
  `M(clip+any_tris) <= M(clip) <= M(control)` and
  `M(any_tris alone) ~= M(control)`.
* `CUDAVK_ABUF_BOUND_ANY_TRIS=1` **alone should do almost nothing.** With the
  framebuffer bound, `230400 * tris * 4 <= 524288` fails for every `tris >= 1`
  at full-screen size, so dropping the triangle condition admits nothing new
  there. Its value is conditional on the clip bound: together they admit a draw
  covering `<= 128*128` pixels (4,096 blocks) with up to 32 triangles, where
  the clip bound alone caps that draw at 2. If `ANY_TRIS=1` alone moves `M`
  measurably, my model of the budget is wrong and I want to know.
* Magnitude: **I am not sizing this and I will not guess.** The number of draws
  whose clip rectangle is materially smaller than the framebuffer is a property
  of the capture that nothing in this session has measured. The `total/quads =
  3.753` probe is a different ratio (fragments per quad, capped at 4 by
  geometry) and must not be used here. The measuring agent's dedicated probe —
  `quad_bound` against actual quads at the `bounded` site — is the input that
  sizes this lead. **Stated as a dependency, not filled in with a guess.**
* Frame time: the design's estimate is 0.10-0.25 ms and the ceiling for the
  whole site is 0.982 ms of blocked host, of which only the part the host can
  spend issuing turns into frame time. A removed wait is worth its blocked time
  only if the host has work to issue during it, and that has never been
  measured in this driver.

---

## 4. The exact A/B command

Both flags default off, so **the candidate arm carries the new behaviour** and
the control arm is the empty environment — WORKFLOW.md §4.3, which warns that
putting the new behaviour on the control side inverts the printed verdict
silently.

Preconditions: `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv,noheader` empty, and the driver taken from
`/tmp/bnd-tree/build-cudavk-bnd`.

Three A/Bs, in this order. Each is a strictly alternating single session
(WORKFLOW.md §4.2), six arms per side on old and four on Crossroads.

```sh
export VK_ICD_FILENAMES=/tmp/bnd-tree/build-cudavk-bnd/src/cudavk/cudavk_devenv_icd.x86_64.json

# A1 - the clip bound alone.  This is the lead.
CAND="CUDAVK_ABUF_CLIP_BOUND=1" CTRL="" /tmp/perf16/cp_decisive_ab.sh

# A2 - the clip bound plus dropping the triangle condition.
CAND="CUDAVK_ABUF_CLIP_BOUND=1 CUDAVK_ABUF_BOUND_ANY_TRIS=1" CTRL="" \
   /tmp/perf16/cp_decisive_ab.sh

# A3 - the triangle condition alone, as the separability check.
CAND="CUDAVK_ABUF_BOUND_ANY_TRIS=1" CTRL="" /tmp/perf16/cp_decisive_ab.sh
```

(The exact harness variable names are the measuring agent's to confirm; the
*arms* are what matter and they are above. Write `arms.txt` first,
WORKFLOW.md §4.5.)

Alongside, one untimed run per arm for the admission numbers and the hash:

```sh
# control
CUDAVK_PLAN_STATS=1 <replay old> 2>plan_ctrl.txt | sha256sum
# candidate
CUDAVK_PLAN_STATS=1 CUDAVK_ABUF_CLIP_BOUND=1 <replay old> 2>plan_cand.txt | sha256sum
grep "main-thread waits" plan_ctrl.txt plan_cand.txt
```

Rules that apply to this patch specifically:

* Do **not** set `CUDAVK_ABUFFER_TIMING`, `CUDAVK_ABUF_FUSE_CHECK` or
  `CUDAVK_ABUFFER_VERIFY` in a timed run. The first two synchronise; all three
  also disable the `bounded` arm outright (`!ab->verify && !ab->timing`), so the
  candidate would silently become the control.
* Check the submit count and the frame count before believing any median
  (WORKFLOW.md §4.4). A fast median from a run that died at frame 26 is exactly
  what the iteration-29 S0 probe produced.
* Frame convention: `median(diff(submit_ts[::2])[50:])`, skip 50.

Reference point for the session, taken today at HEAD: **frame 13.1626 ms old,
5.8230 ms Crossroads.** Do not compare against anything older; the opaque
fan-out in `20611f5b131` moved old from 15.77.

---

## 5. Expected result per capture

| arm | old capture | Crossroads |
|---|---|---|
| control | 13.1626 ms, `M` ~ 4.17 waits/frame | 5.8230 ms |
| A1 clip bound | `M` **falls**; frame 0 to -0.25 ms | `M` falls; frame 0 to -0.15 ms |
| A2 clip + any-tris | `M` <= A1's `M`; frame <= A1's | same shape |
| A3 any-tris alone | `M` unchanged, frame unchanged (within spread) | same |

The frame-time ranges are the design's estimate carried forward, not a
prediction of mine — see §3 on why I will not size this before the probe. The
**admission numbers are the result I am claiming**; the frame time is what the
A/B is for. Crossroads is expected to gain less in absolute ms because its
frame is 5.82 ms, but it has the higher share of scissored UI work, so its
`M` may fall further.

Byte-identical output on every arm, both captures. A hash change is a failure,
not a finding.

---

## 6. What would falsify it

Ranked by how badly each would end the lead.

1. **A hash change with a flag on.** The bound is then not a bound, or the clip
   rectangle is not what §2.1 says it is. Reproduce with
   `CUDAVK_ABUF_CLIP_BOUND=1 CUDAVK_ABUFFER_VERIFY=1` (which forces the drain
   arm and compares against the direct path) and with
   `CUDAVK_ABUF_MIN_TRIS=0`. **Revert, do not patch around it.** The specific
   thing to suspect first is a rasterizer path that writes a fragment outside
   `clip_x0..clip_y1` — §2.1 rests on there being exactly two setup sites.
2. **`M` does not fall.** The captures' clip rectangles are not materially
   smaller than the framebuffer, or the binding test is a different one. Print
   which of the three product tests refuses; if it is
   `num_triangles * bound_pixels <= ab->capacity`, the fragment bound and not
   the quad bound is what limits admission, and the lead is smaller than the
   design thought.
3. **`M` falls and the frame does not move.** Then the wait was not on the
   critical path: the host had nothing to issue during it. This is the failure
   mode iteration 29's addendum measured for site 1 and it would apply here
   too. It does not make the patch wrong, it makes it worthless; land it or
   drop it on tidiness, and record that the P2 ceiling question now governs
   this site as well.
4. **`M` falls and the frame gets *worse*.** The newly admitted draws are
   paying more in over-sized launches than the drain cost. That is the
   iteration-29 disease — the ratio bound/actual — arriving at draw scope. The
   diagnostic is the measuring agent's `quad_bound` vs actual probe: if the
   ratio on the newly admitted draws is large, the clip rectangle is not tight
   for this capture and the lead is dead in the same way site 1 is. Check the
   composite grid hint too (§1.4); if removing it recovers the loss, the
   over-sizing is in the shade, not the composite.
5. **A2 is worse than A1.** Dropping `num_triangles <= 2` admits draws whose
   bound is loose. Ship A1 only; that is why they are two flags.
6. **A3 alone moves anything.** My model of the slot budget in §3 is wrong.
   Not fatal, but the A2 result would then need re-attribution.

---

## 7. The second item: drains that return `quads == 0`

**Not clean. Not done. Here is why, so nobody re-derives it.**

The claim was that 6.9% of old-capture drains and 18.3% of Crossroads drains
come back with `quads == 0`, so the host blocked only to be told the episode
covered nothing.

The premise does not survive reading the site. The episode drain at
`cp_renderer.c:8497` is **not** taken in order to learn `quads`. It is taken to
learn `fill_over` and `quad_over`, and it fetches six counters plus the
per-segment quad counts in one copy. `quads` is a free passenger. Three facts
close it:

1. **The zero is only knowable at the drain.** `quads == 0` is a device-computed
   fact — it depends on culling, on near-plane clipping and on the transformed
   geometry, none of which the host has. Skipping the drain *because* the
   episode is empty requires already knowing it is empty. The only host-side
   proofs of emptiness are "no segment has any primitives" and "the clip
   rectangle is empty", and neither is what makes an episode empty in these
   captures — the geometry is there and the device discards it.
2. **Removing the zero case saves no wait.** Even given a free oracle for
   `quads == 0`, the drain still has to happen for `fill_over || quad_over`
   (`cp_renderer.c:8516`), which decides `cp_pass_fallback`. The saving would be
   the work *after* the drain, which for an empty episode is already a bare
   `return` at `:8526` — a few host instructions.
3. **The overflow answer cannot be deferred.** `cp_pass_can_retry`
   (`cp_renderer.c:7333`, "refusing episode retry after FS attempt") latches
   device loss when a retry follows an FS attempt. So the FS may not be enqueued
   before the overflow answer is known, and no scheme that runs the tail first
   and checks later is admissible. This is iteration 24's no-replay rule, not a
   local judgement call.

The one shape that is *not* refuted is the addendum's **S1d** — take the drain
at `cp_abuf_scan` instead of at the tail, where `ab->sum3[0]` (the exact
fragment total) and `ab->overflow` (the clamp flag) are already written near
the start of the chain. That drains a strictly shorter chain and would make an
empty episode cheap as a side effect. It is a different item with a different
correctness argument, it is already designed and already owned, and its sign is
unknown. It is **not** folded into this patch.

Routing empty episodes through `cp_pass_finish_bounded_groups` is not an
option: that is S1c, and it is dead by measurement.

---

## 8. Files

```
/tmp/bnd-tree                       worktree, branch bounded-clip on e2fea470d04
/tmp/bnd-tree/build-cudavk-bnd      build, BUILD OK, no new warnings
/tmp/bnd-proof/orig.asm             objdump of cp_renderer.c.o at HEAD
/tmp/bnd-proof/forced_off.asm       objdump with both flags constant-folded off
/tmp/bnd-proof/off_vs_orig.diff     the 835 lines of register/scheduling noise
```

Changed: `src/cudavk/cp_renderer.c` (+62/-8), `src/cudavk/cp_debug.c` (+8),
`src/cudavk/cp_debug.h` (+2), `src/cudavk/FLAGS.md` (regenerated).

`docs/cudavk/PERFORMANCE.md` was deliberately **not** edited — the measuring
agent owns section 6 and a speculative edit there would conflict. Item 2 of that
section should be updated with the A/B result, not with this patch.


---

## 9. CLOSURE (added after the measurement came back)

**The lead is closed. Both flags stay default off and should never be set.**

The measuring agent's `quad_bound` probe at the `bounded` site, full replays,
both captures:

* clip blocks == 230,400 == `ab->nblocks` in **6,301/6,301** samples on old and
  **2,535/2,535** on Crossroads. The clip rectangle computes the same number as
  today's bound at every sample.
* admission is **0** under every variant: today 0/6,301 and 0/2,535;
  clip-rect bound 0 and 0; clip-rect with `tris <= 2` dropped 0 and 0.
* bound/actual is ~3.46e7 on old and 2.95e7 on Crossroads.

That is falsifier **2** of §6, and it fired.

### 9.1 The per-draw-rectangle question, settled

The probe sampled the batch-wide rectangle, and §1.3 of this document says the
patch reads the same one. The follow-up question was whether the per-draw
rectangles at `:5197` are tighter. **They are unreachable from this path**, and
the proof is two lines of the same guard:

```
:4626  scissors     = batch->compact_rows ? batch->scissors : NULL
:4621  fs_ubo_table = batch->compact_rows ? batch->fs_ubos  : NULL
:4651  cp->fs_batch.ndraws = fs_ubo_table ? batch_draws : 0
:4808  per_draw_rects = scissors && batch_draws > 1 && rows_stable && scissor
:6448  bounded requires cp->fs_batch.ndraws <= 1
```

`batch->fs_ubos` is an inline array (`cp_renderer.h:161`), never null. So
`per_draw_rects` implies `scissors != NULL` implies `compact_rows` implies
`fs_ubo_table != NULL` implies `ndraws == batch_draws > 1` implies `bounded` is
false. **`bounded` and `per_draw_rects` are mutually exclusive.**

The useful half of that: whenever `bounded` *can* fire, `per_draw_rects` is
false, so the `:4813` scissor intersection did run and the batch-wide rectangle
**already carries the scissor**. It is the tightest rectangle in existence at
this site.

### 9.2 Why no rectangle could have worked

The slot test is `blocks x tris x 4 <= 524,288`, i.e. `blocks x tris <=
131,072`. Against the measured median triangle counts:

| capture | median rast tris | largest admissible rect |
|---|---:|---|
| old | 400 | 327 blocks = 1,308 px, e.g. 72x18 |
| Crossroads | 4,800 | 27 blocks = 108 px, e.g. 54x2 |

That is a scanline, not a scissor. The `quad_capacity` test fails too:
230,400 x 400 = 92.2 M against a 5-32 M `quad_capacity`.

**Admission needs `blocks x tris <= 131,072` jointly.** That single inequality
is the whole reopening test; it needs no new probe, only the joint distribution
of (clip blocks, rast triangles) already in hand.

### 9.3 What the branch is now for

The mechanism does not work, and the reason is worth keeping: **the host cannot
bound this.** `nblocks x triangles` over-estimates by seven orders of
magnitude, and the host does not know post-transform triangle area before the
vertex shader has run. That is site 1's disease arriving at draw scope, and it
is the third independent confirmation that *bound/actual*, not the wait, is the
expensive quantity in this driver.

The follow-on work is `/tmp/perf-audit/s1d_design.md`.
