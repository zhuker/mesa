# Programmatic dependent launch in cudavk: the complete record

This document is self-contained. It is the whole case — what was built, what
was measured with which instrument, what was predicted and how each prediction
scored, what is still unrun, and what the reader should not believe. No message
thread or other file is needed to reconstruct it.

## 0. Status: what exists, where, and what has not been decided

**None of this is in the driver. Landing it is a decision nobody has taken.**

* `cudapipe-vk-native`, the branch the driver is actually built from, is at
  **`e2fea470d04` and contains no PDL at all.** It has not been touched.
* PDL exists only as the branch **`pdl-prototype`**, four commits on top of that
  same base, checked out in the worktree **`/tmp/pdl-tree`**. Nothing has been
  pushed anywhere.
* The commits themselves are **not** in `/tmp`. `/tmp/pdl-tree/.git` is a
  worktree pointer into `/home/alexzhukov/mesa/.git`, and the ref lives at
  `.git/refs/heads/pdl-prototype` in the main repository, so the work survives
  the worktree directory being removed. What is `/tmp`-only is the build
  directory `/tmp/pdl-tree/build-cudavk-pdl` and the SASS artefacts under
  `/tmp/perf-audit/`.
* Working tree clean apart from two untracked build artefacts.

| commit | what |
|---|---|
| `a79567a1310` | tier 1 — the A-buffer scan chain |
| `67244f6dc62` | tiers 2 and 3 — raster stage links, fragment writeback, segment scatter |
| `6e6e00968ca` | on by default at level 3, `CUDAVK_NO_PDL`, docs, the reorder gate |
| `b9766f720a4` | level 4 — stage 1 offered as a diagnostic, default unchanged |

Total against the base: **12 files, +772 / −42 lines** — eleven under
`src/cudavk` (including `FLAGS.md`, regenerated) and
`docs/cudavk/PERFORMANCE.md`.

To see it: `git -C /home/alexzhukov/mesa log --oneline e2fea470d04..pdl-prototype`.
To land it: cherry-pick or merge that range onto `cudapipe-vk-native`, then run
the gate list in §9, which has never been run.

Machine for every number here: RTX 5090 (sm_120), driver 580.173.02, CUDA 12.8 —
the reference machine in `CLAUDE.md` §6.

Two earlier reports, `pdl_prototype.md` and `pdl_tier2.md`, are the per-iteration
records and are kept for their working. Everything load-bearing from them is
restated here.

---

## 1. The mechanism

`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (value 6) on a
*secondary* launch, through `cuLaunchKernelEx`, lets the driver start that grid
before the previous kernel in the same stream has completed and flushed. The
secondary must then execute `griddepcontrol.wait` before it touches anything the
primary wrote. The primary needs no change: with no explicit
`griddepcontrol.launch_dependents`, the trigger happens implicitly once all its
CTAs exit. So what is bought is the inter-grid gap plus whatever the secondary
can execute before its wait.

Nothing is removed. The same kernels run, with the same grids, the same
arguments and the same stream order. Only *when* the GPU may begin the next one
changes.

NVRTC 12.8 does not declare `cudaGridDependencySynchronize`, so the PTX
instruction is written with inline asm and a `"memory"` clobber. `griddepcontrol`
requires PTX ISA 7.8 and sm_90; ptxas assembles the wait to `ACQBULK`.

### 1.1 The predecessor check, which is the whole safety argument

The attribute is defined against "the previous kernel in the stream", so it may
only be set when that is literally true. Two things can make it false and
neither is visible at the call site:

* **(a)** `cp_launch()` flushes an owed coalesced upload on the same stream
  immediately before every launch. If that flush issues a copy, the thing in
  front of the launch is a memcpy.
* **(b)** `cuMemsetD32Async` on the queue counters sits directly in front of
  stage 1 at several sites.

Both are **checked, not assumed**. `cp_smallop_tele.h` already `#undef`s and
renames every CUDA stream call for its small-operation census, so each of them
bumps an epoch: async H2D (including the raw wrapper the upload flush uses),
`cuMemsetD32Async`, `cuMemsetD8Async`, and three interceptions added for this
work with no census role — `cuEventRecord`, `cuStreamWaitEvent`,
`cuMemcpyDtoHAsync`. Those seven are every stream-ordered call in
`cp_renderer.c`.

`cp_launch_after()` then sets the attribute only when all of these hold:

1. the loaded modules were built at this link's tier or higher
   (`cp_kernels.pdl >= pdl_tier`) — so a kernel can never be launched with the
   attribute and be found to contain no wait;
2. the previous launch through `cp_launch()` was the named kernel, on this same
   stream;
3. the epoch has not moved since — no clear, no copy, no event, and in
   particular not the upload flush two lines above.

Condition (3) is deliberately global rather than per stream. It over-refuses
when two threads submit at once and never under-refuses, which is the right way
round for a scheduling hint. A declined link is an ordinary launch, so a site
whose assumption later becomes wrong loses its overlap, never its ordering.

`CP_PDL_ANY` (used only by stage 1) drops condition (2)'s identity comparison
and nothing else. That is sound only for a secondary whose wait is its first
instruction: the identity test exists to justify hoisting loads *above* the
wait, and a kernel that hoists nothing needs only "the previous stream item is a
kernel", which a non-null predecessor plus an unmoved epoch already prove.

---

## 2. What is converted

Levels are ordered by mechanism, not by size, so that each increment answers
one question.

### Level 1 — the A-buffer scan chain (3 links)

| site | link |
|---|---|
| `cp_abuf_scan_n` | `abuf_scan_reduce` → `abuf_scan_finish` |
| `cp_abuf_quad_build` | `abuf_quad_count_all` → `abuf_scan_finish` |
| `cp_abuf_quad_build` | `abuf_scan_finish` → `abuf_quad_fill_all` |

`cp_abuf_scan_finish` had **no independent preamble at all** — its first act was
`sums[tid]`, the predecessor's output, and `ACQBULK` would have landed at
offset `0x40` buying nothing. It was given one: the fill cursor is an array this
kernel *writes and never reads*, so its clear was hoisted out of the main loop
and placed before the wait. That is the pattern to reach for whenever a
secondary looks barren.

### Level 2 — the rasterizer stage links (10 links, five triples × two)

`stage1 → stage2 → stage3`, in both the plain and `_abuf` specialisations, at
all five triples: the A-buffer count pass, the A-buffer fill pass, the plain
draw path, the tiled-opaque fallback, and the per-segment pass loop. Where
stage 1 may be replaced by the fused clip+stage1 kernel, the site names whichever
actually ran.

**These secondaries have nothing to overlap** (§4). That is what makes level 2
a measurement of the inter-grid gap on its own.

### Level 3 — fragment writeback and segment scatter (2 links)

* compiled fragment shader → `cp_fs_writeback`. The shader is a generated
  kernel from `nir_to_ptx` and needs no change; its `CUfunction` is carried out
  of `cp_fs_launch_shader()` through an out-parameter rather than guessed.
* `abuf_seg_prefix` → `abuf_seg_scatter`, enabled by one host reorder (§6.2).

### Level 4 — stage 1, offered as a diagnostic (default off, and CLOSED)

Stage 1 is offered at all five triples with `CP_PDL_ANY`. Not a default: the
point was to let the epoch check refuse the blocked sites and report, through
the take/decline counters, exactly how much of the surface is already clear.

**It has since been measured and the answer is negative: every new offer
declines, and the population behind the blocker is independently priced at
0.0083 ms/frame, fourteen times below the measurable line.** The code stays as a
diagnostic; §8.1 is the closure. Nothing further should be built on it.

### Deliberately not converted

* **stage 1 as a default.** See level 4 — it is a measurement first.
* **`abuf_seg_count` → `abuf_seg_prefix`.** Uploads and scratch allocations sit
  between them, and the prefix is a one-thread kernel with a serial loop and no
  preamble, so the link is worth nothing even if they were moved.
* **interp → fragment shader.** Needs `nir_to_ptx` to emit the wait; §8.2 says
  why that is a bad trade.
* **`cp_abuf_scan_classic`.** Convertible, but only runs under
  `CUDAVK_NO_ABUF_FUSE_SCAN=1`.

---

## 3. Every measurement, with its instrument

### 3.1 The levels, separately (`cp_two_replay_ab.sh`, AB/BA, both captures)

Each row is one AB/BA set against the level below it, so the increments are
interpretable. One stdout hash per capture in every set; arms non-overlapping on
all four comparisons.

| step | old | Crossroads |
|---|---:|---:|
| L1 vs L0 | +0.1430 ms (+1.08%) | +0.0570 ms (+0.97%) |
| L2 vs L1 | +0.1859 ms (+1.42%) | +0.0686 ms (+1.19%) |
| L3 vs L2 | +0.0637 ms (+0.50%) | +0.0220 ms (+0.39%) |

### 3.2 The headline (decisive, strictly alternating, one session, one binary)

**These are the numbers to quote. They were measured on the final tip
`b9766f720a4` on 2026-08-26 (`/tmp/perf-audit/pdl_gate_results.md` §6), 6 runs
per arm on old and 4 on Crossroads, all 20 full length (3,022 / 2,994 submits),
one stdout hash per capture across both arms, arms non-overlapping on both
captures.**

| capture | control (L0) | candidate (L3) | delta | p, one-sided |
|---|---:|---:|---:|---:|
| old | 13.1641 (IQR 13.1575–13.1896) | 12.7826 (IQR 12.7632–12.8075) | **+0.3815 ms (+2.90%)** | 0.0011 |
| Crossroads | 5.8458 (IQR 5.8381–5.8540) | 5.6936 (IQR 5.6868–5.7007) | **+0.1522 ms (+2.60%)** | 0.0143 |

A second statistic, computed differently, agrees: taking the per-frame-index
median across an arm's runs and then the median of the paired difference —
which no burst in a single run can move — gives **+0.386 ms** on old and
**+0.163 ms** on Crossroads, with the candidate faster on **97.0%** of old
frames. A median-of-medians and a per-frame-paired method agreeing is worth more
than either alone.

The control arm also lands on the independently measured HEAD baseline,
13.1641 against 13.1626 ms (`reprofile_baseline.md`, a different session hours
earlier) — 1.5 µs apart. That confirms §6.2's gate from the timing side rather
than the hash side.

#### SUPERSEDED: the pre-gate figures

| capture | superseded figure | measured at |
|---|---:|---|
| old | +0.4342 ms (+3.29%), control 13.2032 / candidate 12.7690 | `67244f6dc62` |
| Crossroads | +0.1297 ms (+2.23%), control 5.8246 / candidate 5.6948 | `67244f6dc62` |

They are kept visible, not deleted, so that nobody finding +0.4342 in an older
artefact concludes the number was quietly improved. **Why they no longer hold:**
they were measured before the landing commit `6e6e00968ca` gated the
`seg_cursor` clear reorder behind level >= 1, so their level-0 control arm
carried a host-side reorder that the shipping revert no longer has. The binary
under the control arm changed, so the control had to be re-measured.

**Both directions, stated honestly.** §6.2 predicted that the gate can only make
the control faster, so the delta should be the same or up to ~0.04 ms smaller.
Old moved 0.053 ms **smaller** — the predicted direction, slightly past the
predicted window. Crossroads moved 0.023 ms **larger**, which the gate cannot
explain at all and is session drift. One side of the prediction held and one did
not; both differences are drift-sized, and neither is a claim about the reorder.

**Do not quote any other cumulative figure.** An earlier "−0.5173 ms" was a
control from one session subtracted from a candidate in another — the exact
cross-session comparison `WORKFLOW` §4.2 warns about — and was inflated by
0.09 ms. It has been removed from every document.

### 3.3 The additivity check

The three separate steps sum to **+0.392** (old) and **+0.1255** (Crossroads);
the direct decisive measurement is **+0.4342** and **+0.1297**. They agree to
11% and 3%, with the direct figure marginally the *larger* on both.

This was worth checking rather than assuming: all three levels shorten the same
episode drain, so if they had double-counted one bottom, the direct figure
would have come in **smaller** than the sum. It did not.

### 3.4 The 600-frame sample sweep (level 3 against level 0, same binary)

Total **15.67 → 15.44 ms (−1.5%)**. gltfscenerendering −3.6%, triangle −11.1%,
texture −9.1%, negativeviewportheight −7.1%, dynamicuniformbuffer −4.3%,
computeshader −3.3%, pbribl −1.9%, bloom −0.9%, multithreading −0.8%, seven
samples flat.

**Re-run at the final tip `b9766f720a4` on 2026-08-26: 15.70 → 15.44 ms
(−1.7%), sample for sample the same picture, and the correctness gate's only
two non-ok entries still the two standing exceptions**
(`pdl_gate_results.md` §7).

**Not one sample regressed**, and pbribl — which regressed 0.03 ms under the
vertex-fetch fusion and had to be accepted as a bounded cost — improves here.
The correctness gate found nothing that differs from the control arm; its only
two non-ok entries are the standing exceptions (gltfscenerendering 137,026 at
frame 47, renderheadless missing), identical to the last four iterations.

### 3.5 Converted links per frame (`CUDAVK_PLAN_STATS`)

| level | old | Crossroads |
|---|---:|---:|
| 1 | 47.28 | 26.67 |
| 2 | 468.78 (added 421.50) | 108.62 (added 81.95) |
| 3 | 550.87 (added 82.10) | 125.28 (added 16.66) |

100% take at levels 1 and 2. Level 3 declines 994 and 288 in absolute terms,
which is the predecessor check refusing links whose stream had an event or a
clear on it — the mechanism working, not failing. **A PDL run that converted
nothing looks exactly like a run without the flag, so this line is the first
thing to read before believing any timing.**

### 3.6 The per-link prices — the new evidence

| link class | old | Crossroads | from |
|---|---:|---:|---|
| no preamble (the gap alone) | **0.44 µs** | **0.84 µs** | L2−L1 over 421.50 / 81.95 links |
| one independent global load ahead of the wait | **0.78 µs** | **1.32 µs** | L3−L2 over 82.10 / 16.66 links |
| a whole clear hoisted ahead of the wait | **3.02 µs** | **2.14 µs** | L1−L0 over 47.28 / 26.67 links |

These are in `PERFORMANCE.md` §4 with the rest of the driver's prices. The rule
that comes with them: **what a converted dependency is worth is decided by what
the secondary can execute before its wait, and that is a property of the kernel,
not of the link.** 0.44 µs is a third of the 1.30 µs for removing a small copy
outright and below the bare-launch price, as it should be — the launch still
happens. So the first question about a candidate link is where `ACQBULK` lands
in its secondary's SASS, and the second is whether the kernel writes an array it
never reads.

**These prices are NET of the `cuLaunchKernelEx` premium.** The level-2
candidate arm made 421.50 *more* extended launches per frame than its control
and still won 0.1859 ms. Subtracting an estimated premium from them again
double-counts a cost already inside the measurement.

---

## 4. The SASS, per converted kernel

NVRTC 12.8 at `compute_120`, `ptxas -arch=sm_120 -O3`, `cuobjdump -sass`, level
4 build. "before" counts SASS instructions ahead of `ACQBULK`.

| kernel | level | ACQBULK | insts before | global before | total | verdict |
|---|---:|---:|---:|---:|---:|---|
| `cp_abuf_scan_finish` | 1 | `0x770` | 119 | 15 stores | 336 | the hoisted cursor clear, on the pixel scan only |
| `cp_abuf_quad_fill_all` | 1 | `0x100` | 16 | 1 | 768 | `blk_counts[b]` load and its branch |
| `cp_rasterize_stage2` | 2 | `0x0e0` | 14 | 1 | 1184 | only the `path_flag` test, branched over when null |
| `cp_rasterize_stage3` | 2 | `0x0e0` | 14 | 1 | 904 | as stage 2 |
| `cp_rasterize_stage2_abuf` | 2 | `0x030` | **3** | **0** | 1304 | **nothing** |
| `cp_rasterize_stage3_abuf` | 2 | `0x030` | **3** | **0** | 1032 | **nothing** |
| `cp_fs_writeback` | 3 | `0x090` | 9 | 1 | 1608 | slot-count load and grid-stride setup |
| `cp_abuf_seg_scatter` | 3 | `0x150` | 21 | 2 | 112 | two independent loads, one-thread predecessor |
| `cp_rasterize_stage1` | 4 | `0x0d0` | 13 | 1 | 816 | `path_flag` only |
| `cp_rasterize_stage1_abuf` | 4 | `0x020` | **2** | **0** | 944 | **nothing** |

Three readings that matter:

* The `_abuf` stage kernels have **three constant-bank loads and no memory
  traffic** before the wait. Every value they use comes out of the queue their
  predecessor filled; there is no write-only array to clear and no load whose
  address is known before the queue is read. No preamble could be invented for
  them. They were shipped anyway, and said so, because zero preamble is what
  makes level 2 a clean measurement of the gap.
* The plain variants look better than they are: their one global load is the
  volatile `path_flag` test, which the SASS branches over whenever the flag is
  null — every path except the tiled-opaque fallback.
* In stage 1, the instruction immediately *after* `ACQBULK` is
  `LDG.E.STRONG.SYS`, the volatile `tri_count` load. **ptxas left it behind the
  wait**, so stage 1 is the 0.44 µs class, checked rather than hoped.

No `ERRBAR` anywhere: there is no explicit `griddepcontrol.launch_dependents`,
by design.

---

## 5. Flag off, and every level, is the binary it was measured with

NVRTC PTX md5 per level. Level 0 of the rasterizer module is the same md5 the
first report recorded against the pre-PDL base revision.

| module | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| `cp_rasterize.cu` | `1c6cf66f…` | `a01ca6db…` | `b809ae98…` | `3055705d…` | `15cdf06c…` |
| `cp_fs.cu` | `f1c9c41f…` | `f1c9c41f…` | `f1c9c41f…` | `85b2c3c4…` | `85b2c3c4…` |

`griddepcontrol` count per level: rasterize 0 / 2 / 6 / 7 / 9, fs 0 / 0 / 0 / 1 / 1.

This is stronger and cheaper than a bitstream diff, and it is what lets a level
be described as a *binary* rather than a branch: dropping the level does not
merely take a different path through the same kernel, it loads the kernel that
level was measured with. It was re-verified on the final tip after every commit,
including the default flip and the reorder gate, both of which are host-side
only.

`tests/cp_debug_doc.py --check`, `tests/cp_launch_audit.py` and
`tests/cp_no_getenv.py` pass. `FLAGS.md` regenerated, 120 switches.

---

## 6. The flag, and the one host-side reorder

### 6.1 Two names, because there are two questions

| name | type | default | question |
|---|---|---|---|
| `CUDAVK_NO_PDL` | bool (value) | off | "what restores the pre-PDL driver?" |
| `CUDAVK_PDL` | uint | 3 | "how much of it do I want?" |

`CUDAVK_NO_PDL=1` means level 0, is what `PERFORMANCE.md`'s default table names,
and **wins over an explicit level** in `apply_couplings()` — someone who writes
it means it. `CUDAVK_PDL` is the tuning knob: 1 and 2 keep their exact meanings
for bisection and `=0` is a synonym for the revert. A regression is bisected to
a group of links by lowering one character.

This follows an existing instance of the house rule rather than bending it:
`texture_cache` is already a derived field set from `no_texture_cache` for the
same reason. PDL adds only that the revert is one of four values the knob can
take, so the coupling is `no_pdl → pdl = 0` rather than a boolean inversion.

**The retired-name table gets no entry, and one would be wrong.** That table
reports names that are *ignored* — `report_retired()` prints "retired and
ignored" — and `CUDAVK_PDL` is neither ignored nor renamed. But the hazard the
`CUDAVK_OPAQUE_STREAMS` entry exists to close is real here in a new shape: a
script written while PDL was opt-in exports `CUDAVK_PDL=1` to mean "on", and now
silently selects *less* than the default. So it gets the same treatment from the
right mechanism — `report_pdl_level()` prints one line on stderr when the level
is explicitly set below the default. It fires only when the variable is set. If
the knob is ever removed, *that* is when the name earns a retired entry.

### 6.2 The reorder, and why it is gated

`cuMemsetD32Async(seg_cursor, …)` moves from *between* the prefix and the
scatter to *before* the prefix, so the two kernels are adjacent and the scatter
can be a secondary. The prefix never touches `seg_cursor`, so this is the same
zeros to the same words at a different point in an order that already had to
hold.

It is **gated on the level being at least 1**. It rode at every level including
0 for one commit, which made `CUDAVK_NO_PDL` not quite a revert — and a revert
that still reorders one stream operation is not one; level 0 would be a fourth
behaviour rather than the original. Gating at `>= 1` rather than at the level
that uses it (`>= 3`) is deliberate: levels 1 and 2 were *measured* with the
reorder present, and gating higher would have changed binaries that already have
numbers.

**Consequence, stated rather than buried:** the decisive figures in §3.2 were
measured before the gate existed, on a binary whose level-0 control carried the
reorder. That control read 13.2032 against a separately observed HEAD baseline
of 13.1626. The gate can only make the control faster, so the true delta may be
up to about 0.04 ms smaller on old. That is inside session drift and is not a
performance claim about the reorder — but the headline should be re-measured on
the final binary before it is quoted to four figures.

---

## 7. The predictions, scored — including the misses

Predictions were registered before each run. The misses are the load-bearing
part: a model that only ever confirms itself is fitted, not structural.

| # | prediction | outcome |
|---|---|---|
| 1 | tier 1 under 2% on both | **PASS** — 1.08% and 0.97% |
| 2 | tier-1 links track **A-buffer episodes**, not opaque segments; "equal take counts per frame would falsify me" | **PASS, decisively** — 47.28 and 26.67, ratio 1.77 while opaque segments differ 48.5 vs 4.0. 3 × episodes predicts 46.5 and 28.7, within 7% |
| 3 | added level-2 links track **draws and segments**, ratio ≥ 4× against tier 1's 1.77 | **SHAPE PASS** — 5.14 |
| 4 | added level-2 links ≈ 128 old, 27 Crossroads | **MAGNITUDE FAIL, 3.3× and 3.0× low** — see §7.1 |
| 5 | L2−L1 at most +0.5%, under 1.0 µs per link | **TIME FAIL in my favour** (+1.42%, +1.19%); **PER-LINK PASS** (0.44 and 0.84 µs) |
| 6 | L3−L2 between +0.1% and +0.6% | **PASS** — +0.50% and +0.39% |
| 7 | level-3 link counts "much closer between the captures" | **FAIL** — ratio 4.93, essentially level 2's 5.14 |
| 8 | `cuLaunchKernelEx` host cost could cancel level 2 | **did not occur**, and the framing was wrong: the premium is already netted out of the measured price (§3.6) |
| 9 | levels might share one drain and be sub-additive | **did not fire** — §3.3, if anything marginally super-additive |
| 10 | level 4 takes ≈ 155/frame on old (140–175), ≈ 33 on Crossroads | **FAIL, by four orders of magnitude** — +0.004 and 0.000. See §7.3 |
| 11 | `CUDAVK_NO_FETCH_FOLD=1` collapses the level-4 take count towards zero | **UNTESTABLE** — it was already zero. The control is uninformative and must not be read either way |
| 12 | the `cuLaunchKernelEx` premium is well under 0.44 µs, likely 0.05–0.25 | **PASS, and by more than predicted** — +6.3 ns entry-point premium |

### 7.1 The 3× scale miss, and what its falsifier proved

The forecast was `triples = episodes + segments`. Measured `batches executed`
settles it: **203.7/frame on old, 50.3 on Crossroads**, against episodes of
15.50 and 9.56.

```
triples/frame = links / 2 = 210.75 (old)   40.98 (Crossroads)
triples/batch             =   1.035          0.815
```

The simple model is right — **one triple per batch, two links per triple** —
fitting old to +3.5% and agreeing with the independent anchor of ~209 stage-1
launches per frame. Two errors, both structural constants, which is why the same
factor appeared on both captures:

1. **Wrong unit.** The code loops over *draw batches*; the forecast was in
   *episodes*.
2. **Wrong multiplicity — and this one the falsifier killed.** I claimed the
   A-buffer path rasterizes each batch twice, a count triple and a fill triple.
   It does not. `cp_renderer.c:6483`: the single-pass build appends records
   during the count and the fill is **one** `abuf_fill_recs` replay; the second
   triple is the fallback branch, which normally does not run. I had read a
   branch that is not taken.

A third effect, the peel loop, is a tail.

**Crossroads in one line:** 0.815 triples per batch is *below 1*, which no
rasterizing path can produce, so about **18% of Crossroads batches never launch
stage 1 at all** — batches that flush with no rasterizable draw, including the
`pass.append_failed` early return at `cp_renderer.c:6220`. The test is one
division already in the census: `stage1 launches / batches executed`, 1.03 on
old against 0.82 on Crossroads.

Prediction 7 has the same root cause: a writeback runs once per fragment-shading
launch, which is per draw, so it *must* share level 2's driver.

### 7.2 The calibration rule this produced

**Anchor a tier's link count on the primary kernel's own launches per frame,
from the census, and never on a structural model of episodes and segments.** The
structural model lost to the simple anchor twice in one exchange — out by 3× on
scale, and its extra term a branch that does not run. As a secondary check:
per-episode tiers show an old:Crossroads link ratio near 1.8, per-draw tiers
near 4.4–5.1, and there is no third case so far.

### 7.3 The level-4 miss, and the third time the same rule was needed

The largest miss of the session: **155 predicted, 0.004 measured**. It is worth
more than the tier it killed, because the reason is the same one twice before.

I forecast level-4 links from **batches** — 203.7/frame — and from reading which
*sites* carry an unconditional clear in the source. Two facts I did not have,
and both already existed in someone's census:

1. **Which of those batches carry a clear at run time.** JOB 11 had already
   counted 27,847 reusing passes on that capture. Those are the only passes that
   issue a standalone stage-1 launch, and a visbuf re-clear sits in front of
   every one of them. The decline count and that census agree to one launch in
   28,000.
2. **How much of "209 stage-1 launches a frame" is even a
   `rasterize_stage1*` launch.** Most of it is the fused clip+stage1 kernel, so
   the population I was forecasting over was an order of magnitude smaller than
   the kernel-time figure suggested.

So the structural model was not wrong about *shape* — the offers really are the
population the source analysis pointed at — it was wrong because it tried to
derive a run-time count from static reading. **That is the third time this
session the batch-unit model needed a driver fact rather than a structural
correction** (the first was the 3× scale, the second the level-3 link ratio),
which is §7.2's rule earning its place a final time and being restated with a
harder edge:

> Before forecasting a launch-site tier, find the counter that already measures
> its population. If no counter measures it, the forecast is a guess with
> arithmetic attached, and the honest move is to build the diagnostic and read
> it — which is what level 4 turned out to be worth.

One thing the miss did *not* cast doubt on: the epoch check. It selected exactly
the right 27,848 launches out of 569.96 per frame, and was validated by a probe
written for a different question. A prediction can fail while the mechanism it
was made about is confirmed, and this is that case.

---

## 8. What was measured after the default landed, and what should not be built

### 8.1 Level 4 — stage 1: measured, and CLOSED NEGATIVE

Level 4 was built as a diagnostic rather than a default, and it did its job: it
closed the question without a timed run being spent.

| | offered/frame | took | took % | declined/frame |
|---|---:|---:|---:|---:|
| old | 569.96 | 550.88 | 96.7% | 19.09 |
| Crossroads | 125.47 | 125.28 | 99.8% | 0.19 |

Against level 3 that is **+18.43 offers/frame on old and +0.001 on Crossroads,
for +0.004 and 0.000 added takes.** Every single new offer is declined. There is
no mechanism to time, so no level-4 AB/BA will be run.

**The number that explains it.** New declines on the old capture:
28,842 − 994 = **27,848**. An independent census (JOB 11) counted **27,847
reusing passes** on the same capture. One launch in twenty-eight thousand apart.
The level-4 stage-1 offers *are* the reusing passes, and they decline because a
visbuf re-clear sits in front of stage 1 on exactly those passes. Crossroads
confirms it from the other side: zero reusing passes, zero new offers.

So the epoch check did not merely work — **it selected exactly the right
population**, and a probe written for an unrelated question priced that same
population independently. That is a stronger validation of the check than the
negative control was ever going to give.

**And the prize behind the blocker is already known to be too small.** JOB 11
measured that same population at 0.44% of direct stage-1 thread invocations —
**0.0083 ms/frame, about fourteen times below the measurable line.** So even if
the visbuf re-clear were hoisted to open those links, there is nothing to win.
**Tier 4 is closed on evidence, not deferred**, and the clear-folding design in
§8.1 of the earlier draft should not be built.

**Why only 18.43 offers a frame and not ~209.** Stage 1 accounts for ~209
launches/frame of kernel time, but a standalone `rasterize_stage1*` launch is
not how most of that work is issued: the fused clip+stage1 kernel covers it, and
a reusing pass is precisely the case the fusion cannot take —
`if (pending_clip.pending && rast_queues.mode != CP_QUEUE_REUSE)`. That is
consistent with the offers being exactly the reusing passes. This last paragraph
is an inference from the source that matches the counters, not a measured fact;
the check is one census line, `clip_rast_fused*` launches per frame, which
should read about 190 on old.

**The negative control is uninformative, and must not be read either way.**
`CUDAVK_NO_FETCH_FOLD=1` at level 4 changes takes by three launches in 832,000
— but it cannot collapse a take count that is already zero. It would only have
been diagnostic had the level-4 links taken. The epoch check is *not* under
suspicion; the 27,847 match vindicates it by an independent route.

### 8.2 `interp → fragment shader` — recommended against

`src/cudavk/nir_to_ptx/cp_nir_to_llvm.c` caps generated shaders at
`CP_MAX_PTX_SM 86` and builds the target machine with `"+ptx75"` (two sites).
`griddepcontrol` needs sm_90 and PTX ISA 7.8, so the edit is two constants —
but the consequence is that **every generated vertex and fragment shader
changes target architecture**, moving instruction selection, scheduling,
register allocation and occupancy across the whole shader path, and it is not
separable from the wait. The payoff is bounded by 60–80 fragment-shading
launches per frame in the no-preamble class: 0.026–0.035 ms, 0.2–0.3% on old.
Worst risk-to-reward in this work. If the sm_90 bump is ever wanted for other
reasons, measure the bump alone first; the wait then becomes a free second arm.

### 8.3 The `cuLaunchKernelEx` premium — measured, and a defect in my own tool

Measured on an idle GPU with `/tmp/perf-audit/pdl_launch_cost.cpp`:

| arm | ns/launch |
|---|---:|
| `cuLaunchKernel` | 2020.1 |
| `cuLaunchKernelEx`, `numAttrs = 0` | 2026.5 |
| `cuLaunchKernelEx` + PDL attribute | −868.6 **— do not quote** |

**The entry-point premium is +6.3 ns**, seventy times smaller than the 0.44 µs
per link tier 2 measured. So the double-counting correction in §3.6 stands, and
with it the "gap alone" reading of tier 2: the premium is nowhere near large
enough to be hiding inside those prices.

**The third arm is a defect in the benchmark I wrote, not a result.** At 200,000
back-to-back empty launches the host is throttled by queue depth, so what is
being timed is launch *throughput*, not entry cost — and the PDL attribute lets
the empty grids overlap on the device, which drains the queue faster and
un-throttles the host. That is why it reads negative. Arms 1 and 2 remain
comparable to each other because both are throttled the same way, which is why
their difference is meaningful while the 2020 ns absolute is not.

The fix, if the attribute's own host cost is ever wanted: keep the queue shallow
— synchronise every few launches, or use a launch count small enough that the
host never blocks — so that all three arms measure entry cost rather than drain
rate. Nobody should quote −868.6 ns as a premium.

### 8.3a Why the premium is accounting, not a gate

`/tmp/perf-audit/pdl_launch_cost.cpp`, written, compiled and linked, **not
run**. Three arms — `cuLaunchKernel`, `cuLaunchKernelEx` with no attributes, and
`Ex` with the PDL attribute — an empty NVRTC kernel, host time only, arms
interleaved and repeated, minimum reported. Build and run lines are in its
header; it needs no capture and takes seconds on an idle GPU.

Its role is accounting because the premium is already inside the measured
prices (§3.6). **Prediction, derived from the level-2 result rather than
guessed: the premium is well under 0.44 µs, most likely 0.05–0.25 µs.** If it
comes back above 0.44 µs, then level 2's GPU-side gain would have to exceed
0.88 µs per link, and the "gap alone" reading of that tier is wrong — so the
benchmark is also a consistency check on §3.6. If the premium is large, the
answer is to cache one `CUlaunchConfig` per context rather than to abandon the
mechanism.

---

## 9. What has never been run

`meson test` has not been run against any of this. Before it ships, the gate
list is: `CUDAVK_ABUF_FUSE_CHECK=1` at `CUDAVK_PDL=3`, which compares the fused
chain against the classic one on the device element by element; the `cpvk_*`
draw tests, which cover the raster stage links the fuse check does not; and one
run at `CUDAVK_NO_PDL=1` to confirm the revert produces identical output.

## 10. Two process rules this work proved, worth more than the patch

1. **Never subtract medians from two sessions.** A "cumulative −0.5173 ms" was a
   control from one session against a candidate from another and was inflated by
   0.09 ms. A cumulative claim must be one decisive measurement, or an explicit
   sum of same-session steps labelled as such.
2. **A revert that reorders anything is not a revert.** An enabling host-side
   reorder must be gated on the feature being on, or the "off" state is a new
   behaviour rather than the original one — and gate it at the *lowest* enabled
   level, so that already-measured intermediate levels keep their binaries.
3. **A refusal counter is a measuring instrument.** Level 4 answered a design
   question — "can PDL get past the clear in front of stage 1, and is it worth
   moving?" — with **no timed run at all**, because a link that refuses itself
   still reports which launches it refused. Offering a link that is expected to
   decline costs nothing (a declined link is an ordinary launch) and buys a
   census of the blocked population. When the next tier's value is unclear,
   build the diagnostic before the design and read it.

## 11. Where this ends

On this branch — and only on this branch, see §0 — PDL is the default at level
3: **+0.3815 ms (+2.90%) on old and +0.1522 ms (+2.60%) on Crossroads,
decisive (p = 0.0011 / 0.0143), with no sample in the 600-frame sweep
regressing** — measured on the final tip `b9766f720a4`, §3.2. The pre-gate
figures +0.4342 / +0.1297 are **superseded**, for the reason given in §3.2:
their level-0 control carried the `seg_cursor` reorder that the shipping revert
no longer has. Level 4 is measured and closed negative. `interp → FS` is
recommended against on risk-to-reward. There is no tier 5 in this design that is
worth building: the links that remain either have nothing to overlap, are
blocked by a clear whose population is already priced below the measurable line,
or need a codegen-architecture change for 0.2–0.3%.

**§9's gate list has now been run, on the final tip, and nothing failed.** The
record is `/tmp/perf-audit/pdl_gate_results.md`: clean build with no new
warning; 67/67 at the default and 67/67 with `CUDAVK_NO_PDL=1`; the fuse check
finding 0 differing elements on both captures; byte-identical stdout for the
revert against a clean HEAD build; the headline above; and the sweep at
15.70 → 15.44 ms with the two standing exceptions unchanged. That file's §8 also
records a measurement-hygiene defect found while re-measuring: a foreign GPU
tenant made of short, repeated processes is invisible to a single point-in-time
`nvidia-smi --query-compute-apps` check.
