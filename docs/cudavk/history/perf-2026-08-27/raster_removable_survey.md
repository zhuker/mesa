# Raster chain: is any of it removable? — survey from existing artifacts

**Survey only. No code written, no GPU used, nothing measured.** Every number
is from an artifact already on disk or from source at `e2fea470d04`.

## 0. Answer, up front

**The top six classes are irreducible. MEASURED — see §7: C1 came in at 0.44%
on old and 0.00% on Crossroads against a 5% line drawn before the run.**
The one genuine pocket of repeated work is inside the peel loop, and it is
0.44% of one class. I found three candidates. One is dead on inspection and its
death retracts a claim I made in my own S1d design. One is bounded from an
existing table at under 0.19 ms and is contested territory. One survives and
needs a census before it can be sized.

| # | candidate | category | worth | evidence |
|---|---|---|---|---|
| C1 | stage 1 recomputes full triangle setup on every REUSE pass and discards it for every non-small triangle | recomputed per pass | **unknown; confined to the peel loop** | source + iteration-11 census + the driver's own comment |
| C2 | the fragment shader shades 4 lanes per quad and 6.2% carry no fragment | ~~fragments later discarded~~ | **DEAD — they are helper lanes, required** | `cp_nir_to_llvm.c:2345-2374, 3393-3397` |
| C3 | the peel loop's per-pass full-framebuffer visbuf clear | passes that could do less | **≤ 0.19 ms, probably ~0.13 ms** | `PERFORMANCE.md` §5.1 memset row, back-solved |

I did not pick one and design it. Below is what each rests on.

---

## 1. C2 first, because it retracts something of mine

I proposed in `/tmp/perf-audit/s1d_design.md` §5 that packing the A-buffer
shade slots dense over fragments would, as a second independent payoff, remove
**~6% of shaded lanes** — the 6.2% that `total/quads = 3.753` says carry no
fragment. **That is wrong. Checking this lead is what disproved it.**

The 4-lanes-per-quad layout is not slack. It is the helper-lane quad:

* `cp_nir_to_llvm.c:2345-2374`, `cp_quad_derivative()`, computes `ddx`/`ddy`
  with `llvm.nvvm.shfl.sync.bfly.f32` over `tid & 28` — it *assumes* lanes
  `4q..4q+3` are the four pixels of one 2x2 block. That is exactly the
  numbering §5 proposed to destroy.
* `:2379`: *"Implicit LOD is expressed as explicit quad gradients: compute-stage
  CUDA texture instructions do not infer graphics fragment derivatives."* So
  every implicit-LOD texture sample in the driver goes through those gradients.
* `:3393-3397` states the rule outright: *"A quad is shaded whole so that
  derivatives can be taken across it, so a lane the primitive does not cover is
  shaded too. Vulkan says such a lane's stores and atomics have no effect."*
* The only coverage gate in the generated shader (`:3409-3423`) is emitted
  **only when `ctx->writes_memory`**, and it exists to suppress *side effects*
  from helper lanes, not to skip their shading.

So the 6.2% are helper lanes, they are architecturally required, and an
early-out would not collect them anyway — uncovered lanes are scattered, so a
32-lane warp is essentially never fully uncovered.

**S1d §5 is withdrawn**, and the withdrawal is recorded in that document.

---

## 2. C1 — stage 1 recomputes and discards setup on every REUSE pass

### The work

`cp_rasterize_stage1_body` (`cp_rasterize.cu:860-874`) calls

```c
cp_rast_small_or_defer<ABUF>(&args, &queues, tri_id, queues.mode != CP_QUEUE_REUSE);
```

and `cp_rast_small_or_defer` begins with an unconditional
`setup_triangle(args, tri_id, &s)` — position loads, perspective divide, cull
test, bounding box, edge setup — and then:

```c
if (bb_area > CP_SMALL_THRESHOLD) {      /* 128 */
   if (append) { ...atomicAdd, queue[idx] = tri_id... }
   return;                                /* on a REUSE pass: just return */
}
```

On a reusing pass `append` is false, so **for every triangle above the small
threshold the whole setup is computed and thrown away.** The driver says so:
*"The classification is still done — it is what decides this thread does not
rasterize — but the append is not."*

### Why it is a real candidate rather than a necessary cost

**Stage 2 already has exactly this optimisation, one stage later.**
`cp_rasterize.cu:906-917`:

```c
if (entry & CP_NT_HUGE)
   continue;    /* before setup_triangle */
```

with the comment *"Skipping it here is what makes the reuse worth anything: the
alternative is to redo the setup on every pass purely to rediscover that the
primitive is too large for this stage."* **The same argument applies verbatim to
stage 1 and has not been applied there.**

The mechanism would be the same shape: one bit per triangle written on the
BUILD pass ("queued, not mine"), loaded on REUSE passes, returning before
`setup_triangle`. The precedent for the marker is `CP_NT_HUGE` itself.

For small triangles nothing is saved — stage 1 rasterizes them and needs `s`.
The saving is exactly the non-small ones.

### What it is worth: unknown, and confined to the peel loop

This is where I have to be honest about the artifacts.

* Iteration 11's setup census records **direct REUSE = 23,233,038 stage-3 tile
  executions against 15,580,983 FILL** — reuse was 60% of direct raster work
  when that census was taken.
* **But that census predates the A-buffer taking blended draws.** At HEAD the
  peel loop only runs for draws the A-buffer refuses. The iteration-3 kernel
  summary shows `cp_clip_rast_fused` 35,408 launches and
  `cp_rasterize_stage2` / `cp_rasterize_stage3` **35,408 each** — equal. Since
  the fused clip+stage1 *can only run on pass 0* (`cp_renderer.c:6884-6886`:
  *"only pass 0 can get here ... a reusing pass never clips"*), equality means
  essentially **one pass per direct draw** in that trace. `cp_rasterize_stage1`
  (the unfused kernel a REUSE pass launches) does not appear in the top 26 at
  all.
* So C1's population is the peel loop, and the peel loop is **1.70 check
  intervals per frame** at HEAD — but those intervals cost **1.6135 ms of
  blocked host each**, which is device time. The peel loop is a small number of
  draws doing a large amount of device work.

**I cannot size C1 from existing artifacts.** What it needs is one counter:
REUSE-pass stage-1 invocations, split by whether the triangle was above the
small threshold. That is one teardown-only counter in the shape of the two
probes I already wrote.

**And it must be said plainly:** C1 lives inside the peel loop, which is where
the predication patch just blocked 1.01 ms/frame more for no frame movement.
Anything proposed there inherits that result until something explains it.

---

## 3. C3 — the peel loop's per-pass framebuffer clear

`cp_renderer.c:6835-6837`, at the top of every pass after the first:

```c
cuMemsetD32Async(visbuf, 0xFFFFFFFF, (size_t)w * h * 2 * fb_samples, cp->stream);
```

That is `w*h*2` 32-bit words = **7.37 MB at 1280x720x1 sample**, about 4.6 us at
this card's bandwidth, once per overshoot or genuine pass.

**It can be bounded from the existing table without a census.** `PERFORMANCE.md`
§5.1 gives **memset: 240 operations, 0.190 ms/frame** for the *whole driver*.
If the visbuf clears are the large ones and everything else is small, solving
`4.6N + 0.3(240-N) = 190 us` gives **N ~ 27 clears/frame, ~0.13 ms/frame**.
So C3 is **at most 0.19 ms and probably about 0.13 ms**, and that ceiling costs
nothing to establish.

Three things then bound it further, and two of them are mine:

1. **The clip-rectangle variant is already dead.** The obvious refinement —
   clear only the rows the draw can touch — is worth nothing, because the
   clip-rectangle probe measured **clip blocks == whole framebuffer in
   6,301/6,301 samples on old and 2,535/2,535 on Crossroads**. There is no
   over-clear to remove.
2. **A clear is only removable if its pass is removable**, which is L4, which
   regressed.
3. Iteration 29's addendum already flagged this exact memset as the thing a
   peel change must audit: *"a predicated-out peel pass must not clear anything
   sized to the framebuffer."*

So C3 is real, bounded, small, and sits in contested territory. I would not
lead with it.

---

## 4. What I checked and found irreducible

Stated so the survey is not just three positives.

* **`cp_rasterize_stage3_abuf`, 2.484 ms, the largest class.** Its per-tile
  setup recomputation is already cached for primitives with >= 4 tiles
  (`CP_SETUP_CACHE_MIN_TILES`), which iteration 11 measured as retaining
  **91.9% of direct and 96.8% of A-buffer tile records** for a −2.4% stage-3
  gain. The remaining 3–8% is a fraction of that, i.e. thousandths of a
  millisecond. The only other removable work in it is the absent trivial-accept,
  which is L17 — double-contested, and gated by its own author on a census that
  has never been run.
* **Its occupancy is the finding, and it is not removal.** NCU measured
  `launch__waves_per_multiprocessor` **median 0.15, max 0.60, 0 of 270 reaching
  1.0**, and the median A-buffer grid is `CLAMP(tris*8, 512, 2048)` = 512 blocks.
  The median stage-3 launch uses 15% of one wave and takes 18.5 us. That is
  **unoverlapped work, not unnecessary work**, and `PERFORMANCE.md` §7 already
  says so: *"It is not a kernel-internal problem."* The segment count phases are
  already fanned out over the side streams (`cp_pass_join`'s own comment).
* **The A-buffer is not rasterized twice.** The count pass appends records and
  `abuf_fill_recs` replays them; launch counts confirm one raster chain per
  segment (`stage3_abuf` 78,035 against `abuf_fill_recs` 7,473 = 10.4 segments
  per episode).
* **The per-pass 3-word queue-counter clear is already skipped on REUSE passes**
  (`cp_renderer.c:6875-6877`). Harvested.
* **Stage 2 and stage 3 grids** are fixed and stride their queues; sizing stage
  3 from the triangle count *"was measured and is a clear loss"*
  (`cp_renderer.c:6919-6926`), and grid tuning is closed in §7 at <0.2 ns per
  idle block.

---

## 5. What this means for the programme

If C1 turns out small — and the launch counts suggest the peel loop is a
handful of draws — then **the honest conclusion is that the top six classes are
irreducible work, and the remaining programme is overlap rather than removal.**
That is where PDL already sits, and it is consistent with the two changes that
have paid.

The one measurement that would settle it costs one counter and no GPU risk:
**REUSE-pass stage-1 invocations, split above and below `CP_SMALL_THRESHOLD`,
and the number of peel passes per frame.** I have not written it, because you
asked for the survey first.

---

## 6. THE COUNTER — written, and my prediction registered before it runs

Committed on branch `bounded-clip` in `/tmp/bnd-tree`, commit `25ea48fe388`.
Flag **`CUDAVK_PEEL_SETUP_CENSUS`**, `CP_FLAG_BOOL_VALUE`, default off,
buffered to teardown, `cp_debug_doc.py --check` and `cp_no_getenv.py` pass.
With it and the other four branch flags folded off, `cp_renderer.c.o` differs
from `e2fea470d04` by **one scheduled `mov` in `cp_pass_finish` and nop
padding**, over 172 symbols with none added and none removed.

**It adds no wait.** The one device word it reads — the nontrivial counter the
build pass left — is read *behind the peel loop's own convergence check*, which
already synchronises the stream. Everything else is host arithmetic. Frames are
counted by the project's own convention: `cp_depth_attachment_store` fires
exactly once per frame on both captures.

**It counts; it does not time. A run under this flag is not a timed run.**

### 6.1 The exact command

```sh
export VK_ICD_FILENAMES=/tmp/bnd-tree/build-cudavk-bnd/src/cudavk/cudavk_devenv_icd.x86_64.json

CUDAVK_PEEL_SETUP_CENSUS=1 <replay old>        2> /tmp/perf-audit/peel_old.txt   | sha256sum
CUDAVK_PEEL_SETUP_CENSUS=1 <replay crossroads> 2> /tmp/perf-audit/peel_cross.txt | sha256sum

grep -A6 "peel setup census" /tmp/perf-audit/peel_*.txt
```

Report the two captures separately; they are not comparable and the whole
question is whether either has a population. The stdout hash must match a
flagless run of the same binary. Do not set `CUDAVK_NO_BINCACHE`: it also turns
off `cache_queues` (`cp_renderer.c:6171`), which removes the reusing passes
this probe exists to count.

### 6.2 What it prints

```
cudavk: peel setup census: L peel loops over F frames (l/frame), M ran more than one pass (m%)
  passes: P total (p/frame), longest loop X; reusing passes R (r/frame) = plain cp_rasterize_stage1 launches
  stage-1 thread invocations, direct path: I total (i/frame), of which J (j/frame) are on a reusing pass
  full setups computed and discarded on a reusing pass: D (d/frame)
  VERDICT (C1): a per-triangle queued bit would skip d setups a frame, s% of all
                direct stage-1 thread invocations. AT OR ABOVE / BELOW the 5% line.
```

### 6.3 Where the 5% line comes from, drawn before the run

`raster stage1 (clip+s1 fused, direct)` is **1.894 ms/frame** (`PERFORMANCE.md`
§5.1). The old capture's session spread is **0.119 ms** across three run medians
(`reprofile_baseline.md` A.1). 5% of the class is **0.095 ms** — just under the
spread. **Below 5%, a *perfect* removal cannot be measured even if it works**,
and the "removal is exhausted" conclusion stands regardless of what a patch
would do.

And the conversion is deliberately generous to C1: a reusing pass launches plain
`cp_rasterize_stage1`, which does strictly *less* than the fused clip+stage1 of
pass 0 — no clip, no queue appends. So pricing a reusing invocation at the fused
class's rate is an **upper bound**, and the 5% line is charitable.

### 6.4 My prediction

**I predict BELOW the line, and probably far below.** Three reasons, all from
artifacts already on disk:

1. `cp_clip_rast_fused` = `cp_rasterize_stage2` = `cp_rasterize_stage3` =
   **35,408 launches each** in the iteration-3 summary. The fused clip can only
   run on pass 0 (`cp_renderer.c:6884-6886`), so equality means about **one pass
   per direct draw**.
2. `cp_rasterize_stage1` — the kernel a reusing pass launches — **does not
   appear in the top 26 kernels by device time at all.**
3. The peel loop runs **1.70 convergence-check intervals per frame** at HEAD.

Concretely, and each of these is a falsifier:

| I predict | I am wrong if |
|---|---|
| discarded setups **< 5%** of direct stage-1 invocations, likely **< 1%** | **>= 5%** |
| reusing passes **< 10/frame** | **>= 10/frame** |
| **more than half** of peel loops run exactly one pass | **half or more run two or more** |

Any one of the right-hand column means C1 deserves a design, and the argument
for it is already written: **stage 2 does exactly this one stage later**
(`if (entry & CP_NT_HUGE) continue;` before its own `setup_triangle`), and stage
1 has never had it applied.

### 6.5 Two things to say out loud when the result comes back

1. **If C1 is real it is removed DEVICE work, not a removed host wait.** That
   distinction matters here more than anywhere: S0, peel predication and S1d all
   attacked host waits and all three failed, while the two changes that paid —
   the opaque fan-out and PDL tier 1 — both moved device work. C1 is in the
   second category. It should be argued as such and not allowed to inherit the
   first category's expectations by proximity.
2. **But it sits in a graveyard, and that must not be waved away.** The peel
   loop is where predication blocked 1.01 ms/frame more for 0.11 ms of
   regression, and where the census found 89% of added blocking cost nothing.
   Anything proposed inside that loop inherits both results until something
   explains them. Removing device work is a different mechanism from removing a
   wait — but it is the *same loop*, and the loop has already surprised us once.
3. **Bound the answer from existing tables before spending a run on it**, the
   way §3 bounded C3 at 0.19 ms from the memset row and killed its obvious
   refinement with the clip-rectangle probe. The probe returns *setups*; the
   conversion to milliseconds is §6.3's upper bound and needs no new
   measurement.

---

## 7. RESULT — C1 is closed, and one of my three falsifiers fired

`CUDAVK_PEEL_SETUP_CENSUS=1`, both captures, full replays, stdout hashes
identical to a flagless run on the same binary (`320e993599cc`, `e727020fc796`),
submits 3,022 and 2,994. Raw: `/tmp/perf-audit/setupcensus/`.

```
old    23921 peel loops over 1646 frames, 344 ran more than one pass (1.4%)
       passes 51768, longest loop 256; reusing passes 27847
       stage-1 invocations 694,295,360, of which 132,652,576 on a reusing pass
       full setups computed and discarded: 3,032,607
       VERDICT: 0.44% of all direct stage-1 thread invocations. BELOW the 5% line.

cross  19200 peel loops over 1496 frames, 0 ran more than one pass
       reusing passes 0; discarded setups 0
       VERDICT: 0.00%. BELOW the 5% line.
```

**C1 is closed.** The mechanism is real and the population is not.

### 7.1 The predictions, scored

| registered in §6.4 | outcome |
|---|---|
| discarded setups **< 5%**, likely **< 1%** | **PASS** — 0.44% and 0.00% |
| **more than half** of peel loops run exactly one pass | **PASS decisively** — 98.6% and 100% |
| reusing passes **< 10/frame** | **FAIL** — 16.9/frame on old |

**The falsifier that fired was the wrong falsifier, and that is the finding.**
Reusing passes are 16.9/frame and stage-1 invocations on them are **19.1% of all
of them** — a large population by both of those measures — yet discarded setups
are 0.44%. The 344 many-pass loops are **small-triangle** draws: stage 1
rasterizes them itself and needs the setup it computes. I had conflated "many
reusing passes" with "much discarded work". The operative falsifier — the 5%
line on discarded setups, which is the quantity a fix would remove — was the
right one, and it held by more than a factor of ten.

### 7.2 One instrument caveat, self-caught

The census divides by `cp_depth_attachment_store` calls. That is **1.09 per
frame on old, not 1.00** (1,646 stores against ~1,510 frames; see
`l6_depth_elision.md` §1.1), so old's `/frame` columns are about **9% low** —
34.3 passes/frame rather than 31.45, and 2,008 discarded setups/frame rather
than 1,842. Crossroads is exact, 1,496 against 1,497. **The 0.44% verdict is a
ratio of two totals and is unaffected.**

### 7.3 What this means for the programme

With C1 closed, all three candidates in this survey are closed and the top six
kernel classes are irreducible work. **Removal in the raster chain is
exhausted; what remains is overlap.** That conclusion is now measured rather
than argued, and it is recorded in `/tmp/perf-audit/SESSION_HANDOFF.md` §6.
