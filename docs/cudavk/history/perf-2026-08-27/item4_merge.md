# Item 4, THE RASTER MERGE — a launch-count experiment, built and ready

Branch `raster-merge`, commit `c6681a43fc8`, worktree `/tmp/merge-tree`, base
`43df52c383e` on `cudapipe-vk-native` (PDL landed, default level 3). Build
directory `/tmp/merge-tree/build-cudavk-merge`; ICD at
`/tmp/merge-tree/build-cudavk-merge/src/cudavk/cudavk_devenv_icd.x86_64.json`.

**Flag: `CUDAVK_MERGE_FILL_RAST`, default off.** One flag, one change.

**No GPU was used.** Nothing was replayed, benchmarked or `meson test`ed. The
only device compilation is NVRTC and `ptxas` on the rasterizer module, off
line, for the inertness proof in §4.

---

## 1. What it does

`cp_pass_finish()` ends an A-buffer episode. When the single-pass build is off
(`CUDAVK_NO_ABUF_APPEND=1`) the fill is a loop of `nsegs` raster triples issued
back to back at one host point behind one join
(`cp_renderer.c:8556-8595` at base). The merge replaces that loop with, per
chunk of at most eight segments, **three launches instead of twenty-four**.

Mechanism, in one paragraph. The three stage bodies keep their code and take
the block index and the grid width as arguments instead of reading `blockIdx.x`
and `gridDim.x`. The merged entry point resolves a block to its item through a
prefix over the per-item block counts, then calls the identical body with
`blockIdx.x - block_base[item]` out of `block_base[item+1] - block_base[item]`
— which is **bit for bit the `(blockIdx.x, gridDim.x)` pair that block would
have been given by its own segment's launch**. Per-item state is the two
structs a single-segment launch passed by value (`cp_rasterize_args` 344 B +
`cp_rast_queues` 56 B = 400 B), uploaded as a table and staged through shared
memory. The prefix itself travels by value in the launch arguments, so the
block→item map costs constant-bank reads and no global traffic.

The block→item map was checked exhaustively on the host for every segment count
1..64 and all three stages: every item sees exactly the block indices its own
launch would have produced, with the grid width its own launch would have had.

**Why this position and no other.** That loop is the only place in the driver
where several segments' raster chains are already pending at one host point
behind one join. Merging it changes launch width and changes **nothing** about
when work is issued. Every other candidate — the count phase above all — also
has to defer work to the end of the episode, and deferral lands on the episode
drain, the one site with a measured conversion of **+1.02**, where any delay is
paid at par. A number taken here is a statement about launch count alone.

**Why the chunk is eight.** `CP_RAST_MERGE_MAX` is `CP_PASS_STREAMS`, and it is
a correctness bound, not a knob: segment `s` rasterizes into queue set
`s % CP_PASS_STREAMS` (`cp_renderer.c:9095`), so segments `s` and `s + 8` share
a nontrivial queue, a huge queue and a setup cache. The per-segment loop keeps
them apart by putting them on the same stream; the merge keeps them apart by
putting them in different launches on one stream. Chunks start at multiples of
the width, so queue sets inside a chunk are always distinct.

**Per-launch semantics that had to become per-item** (the prompt's watch list):
`abuf_prim_base` — per item, it is already per segment in `sg->rast`;
`args.path_flag` — per item for free, and unreachable here anyway because
`CUDAVK_TILED_OPAQUE` is excluded from episodes; `rast_queues.mode` — per item,
set to `CP_QUEUE_FILL` for every item exactly as the loop sets it. Nothing had
to be refused.

**What did NOT change:** the per-segment counter clears
(`cuMemsetD32Async(..., 3, ...)`) are still one per segment, in the loop's
order, now all on the main stream. Merging those is a second change and is not
in this patch.

---

## 2. What is byte-identical, and why

### 2.1 Flag off

Proven at machine level, §4. Nothing else needs to be argued.

### 2.2 Flag on — the output must be identical, and here is the chain

1. **Each block does what it did.** It runs the same body, with the same
   `(block, grid)` pair, over the same `cp_rasterize_args`, into the same queue
   set. Verified exhaustively for 1..64 segments (§1). No work item is added,
   dropped or moved between blocks.
2. **The only thing that changes is which blocks are co-resident.** Inside a
   chunk, item 3's stage-1 blocks now share a grid with item 0's instead of
   sharing the machine through eight streams.
3. **That is already the default behaviour.** Blended segments have always
   fanned out over eight side streams with no flag
   (`SESSION_HANDOFF.md` §3.7, §4 F4), so segments 0..7 already race.
4. **The result is order-free.** Per-pixel accumulation is `atomicAdd` plus one
   global record cursor (`cp_rast_types.h:203-227`) and the sort restores
   submission order by primitive id (`ARCHITECTURE.md` §2.4, §4.2).
5. **And it has been observed.** `reprofile_stats.md` B.4 reports **one stdout
   hash across the fan-out arms** — i.e. append order provably does not change
   the bytes on this workload.

The residual assumption, stated as one: that no tie-break in the sort depends
on append order. It is inference I6 of `wide_merge_census.md`, and point 5 is
the evidence that it already holds. **The stdout hash is falsifier F3 below.**

### 2.3 One behavioural difference that is not byte-visible

If every segment of a chunk has zero triangles, the merged stage-1 grid is zero
blocks and the launch is skipped, where the loop would have issued a zero-block
launch per segment (which `cuLaunchKernel` rejects). This cannot change output;
it can only change a warning that today's code would emit.

---

## 3. The size, and the forecast, at the settled price

**The price is 0.782 us per launch removed from a real queue** (95% CI
[0.633, 0.949]), confirmed by spread injection, by union idle in front of real
chain kernels (0.809 us) and by the driver's own price table (0.6-1.0). The
1.974 us figure is the price of ADDING an exposed launch and is not used here.

### 3.1 The launch arithmetic

Per episode with `n` segments:

| | launches issued by the fill |
|---|---|
| control | `3n` |
| merged | `3 * ceil(n / 8)` |

Old capture anchors, from the census and from the driver's own counters:
**134.6 `_abuf` raster triples/frame** (`PERFORMANCE.md` §5.1) and **15.50
episodes/frame closed** (`reprofile_stats.md` B.4). With
`CUDAVK_NO_ABUF_APPEND=1` the fill adds one triple per segment, so:

* control fill launches/frame = `3 x 134.6` = **403.8**
* merged fill launches/frame = `3 x SUM ceil(n_e / 8)`, and
  `134.6/8 = 16.8 <= SUM <= (134.6 + 7 x 15.50)/8 = 30.4`, so **50.4 to 91.2**
* **removed: 313 to 353 launches/frame**

At 0.782 us: **+0.244 to +0.276 ms/frame**. Carrying the CI: **+0.198 to
+0.335 ms/frame**.

**Registered forecast, old capture, NO_ABUF_APPEND arm: +0.26 ms/frame,
band +0.20 to +0.34.**

### 3.2 Crossroads

I have no measured blended-segment count for Crossroads in any existing
document, so this anchor is a scaling and is marked as one — R4 says anchor on
the class's own launches per frame, and this one is not. Scaling by the two
figures that are measured on both captures (PDL links 125.28 / 550.87 = 0.227;
total launches 346 / 1314 = 0.263) gives ~32 blended segments/frame, ~4
episodes/frame, so removed ~73 to 84 launches/frame:

**Registered forecast, Crossroads: +0.06 ms/frame, band +0.04 to +0.09 —
and this band is the weak one. Recompute it from the run's own PLAN_STATS
before scoring it.**

### 3.3 What this position is NOT worth

**This is a diagnostic, not a shippable win, and the number above is not the
0.433 ms/frame of the merge census.** The 554 legally mergeable launches live
in the **count** phase, at append time, which is the shipping default path. The
fill relaunch loop is reachable only with `CUDAVK_NO_ABUF_APPEND=1`; the
shipping default replaces the whole loop with **one** `cp_abuf_fill_recs`
launch per episode, so it is already merged and this patch cannot touch it.
What this experiment buys is the **price of launch width, measured with
deferral held fixed** — the one thing every other candidate confounds. If it
pays here, the count-phase merge is worth building and its deferral cost can
be weighed against a known width credit. If it does not pay here, the
count-phase merge cannot be justified on launch count at all, and that is worth
knowing before anything lands on a site with a 1.02 conversion.

---

## 4. Inertness, proven at machine level

### 4.0 The positive control, first — because without it §4.1 and §4.2 are vacuous

**A folded-off build and a build with the feature MISSING are indistinguishable
unless something distinguishes them.** An inertness check that only compares
base against flags-off will report a perfect result for a patch whose
implementation was lost. So: three objects, not two.

**Device side, where the gate really is a compile-time fold** — this is the
true A/B/C:

| | PTX | cubin `.text` | kernels |
|---|---:|---:|---:|
| **A** base | 656,339 | 357,888 | 40 |
| **B** patched, `-DCP_RAST_MERGE=1` | 756,883 | **412,544** | **43** |
| **C** patched, folded off | 656,563 | 357,888 | 40 |

`B − A = +54,656 bytes of SASS and +3 kernels` — the feature is demonstrably
present. `C − A = 0 bytes` — folding removes exactly it. Identical at
`CP_PDL=0` (`A` 356,992, `B` 411,648, `C` 356,992). The three added kernels are
`cp_rasterize_stage{1,2,3}_abuf_merged`.

**Host side there is no C, and saying so is part of the answer.** The host gate
is a *runtime* flag, so the merged branch is compiled in unconditionally and
selected by an `if`. There is only A and B, and B must differ:

| object | `sum(.text*)` A | B | delta |
|---|---:|---:|---:|
| `cp_renderer.c.o` | 149,581 | 151,233 | **+1,652** |
| `cp_kernels.c.o` | 4,853 | 5,008 | **+155** |
| `cp_debug.c.o` | 3,011 | 3,011 | 0 — *the registry is a table, not code* |

`cp_pass_finish` **3,587 → 3,952 instructions (+365), with +18 `call`
instructions**. `cp_kernels_init` +16, `compile_cuda_source` +15. The flag row
lives in data: `.data.rel.ro.local.flags` +88 bytes, `.bss.present_in_env` +1,
and the description string is in the shipped `.so`.

**So the host claim in §4.2 must be read for what it is: a *containment* proof,
not a folding proof.** It says no function other than the four the patch edits
changed behaviour. The +365 instructions in `cp_pass_finish` are the positive
control that there is something to contain.

**And the branch is reachable, not dead code.** Offsets from a probe compiled
with the object's own command line: `offsetof(cp_debug, merge_fill_rast)` =
284 = `0x11c`; the three merged `CUfunction` handles are at device-relative
568/576/584 = `0x238`/`0x240`/`0x248`. In the disassembly of `cp_pass_finish`:

```
b4a:  cmpb   $0x0,0x11c(%rax)      <- cp_debug->merge_fill_rast
b5c:  cmpq   $0x0,0x238(%rbp)      <- kernels.rasterize_stage1_abuf_merged
e58:  mov    0x238(%rbp),%rsi      <- merged stage 1 launched
ea3:  mov    0x240(%rbp),%rsi      <- merged stage 2 launched
ecc:  push   0x238(%rbp)           <- ...after stage 1 (the PDL predecessor)
f1a:  mov    0x248(%rbp),%rsi      <- merged stage 3 launched
f43:  push   0x240(%rbp)           <- ...after stage 2
```

The base object contains **zero** references to `0x11c` in that function.

**On "gate even the parts too cheap to gate": there is nothing to gate.** This
patch adds no counter, no probe and no unconditional statement anywhere. The
gate itself sits *inside the arm it gates* — it is the `else if` of
`if (ab->recs && kernels.abuf_fill_recs)`, so on the shipping default the flag
byte is never even loaded.

### 4.1 Device side — byte-identical SASS

The rasterizer module compiled with NVRTC 12.8 (a five-line driver that mirrors
`compile_cuda_source()`), assembled with `ptxas -arch=sm_90 -O3`, disassembled
with `cuobjdump -sass`, addresses stripped:

| comparison | result |
|---|---|
| base `cp_rasterize.cu` vs patched, `CP_RAST_MERGE` undefined, `CP_PDL=0` | **byte-identical SASS** |
| base vs patched, `CP_RAST_MERGE` undefined, `CP_PDL=3` | **byte-identical SASS** |
| base vs patched with `-DCP_RAST_MERGE=1`, `CP_PDL=0` | **all 40 existing kernels byte-identical**, 3 added |
| base vs patched with `-DCP_RAST_MERGE=1`, `CP_PDL=3` | **all 40 existing kernels byte-identical**, 3 added |

So passing the block index and the grid width as arguments instead of reading
them from `blockIdx`/`gridDim` costs literally nothing, and the merged kernels
do not perturb the ones beside them. The three added entry points are
`cp_rasterize_stage{1,2,3}_abuf_merged`.

The flag also controls whether they are compiled at all: `-DCP_RAST_MERGE=1` is
added to the NVRTC option string only when the flag is set, and the option
string is part of the NVRTC disk-cache key — so the off arm re-uses the exact
module every existing measurement was taken with, on the same rule the PDL
levels already follow.

### 4.2 Host side

`objdump -d` of the three changed objects against the same objects built from
the pristine sources with the same command line (`ninja -t commands`):

```
cp_renderer.c.o : 173 functions, 23 differ, 1 of them edited by the patch
cp_kernels.c.o  :   9 functions,  6 differ, 2 of them edited by the patch
cp_debug.c.o    :   2 functions,  1 differ, 1 of them edited by the patch
TOTAL           : 184 functions, 30 differ; none added, none removed
```

The four functions the patch edits are `cp_pass_finish` (3588 -> 3953
instructions), `cp_kernels_init` (418 -> 434), `compile_cuda_source`
(364 -> 379) and `cp_debug_init` (659 -> 659). **Outside those four, no
function gains or loses a single instruction**, and every one of the 102
changed instructions falls into one of three classes with **zero left over**
(this is a containment claim; §4.0 is what makes it non-vacuous):

```
  93  struct displacement shifted by exactly +0x18
   8  __LINE__ immediate (the upload-stats site attribution)
   1  local-label renumber in a branch target
```

The `+0x18` is the 24 bytes of three `CUfunction` handles appended to
`struct cp_kernels`, which is embedded in `struct cp_device`. The flag and the
handles are appended at the **end** of their structs deliberately: inserted
beside their neighbours they would have shifted every later member and rewritten
the displacement in every function that reads one, which is how an experiment's
real footprint gets buried in noise. The classifier is
`/tmp/merge-ptx/objdump_evidence.txt`.

House checks: `cp_debug_doc.py --check` PASS (FLAGS.md regenerated),
`cp_no_getenv.py` PASS, `cp_launch_audit.py` PASS. No `getenv` outside the
registry; every launch still goes through `cp_launch()`.

---

## 5. The confounds, with their sizes — read these before scoring the result

These are the reasons the experiment can give the wrong answer about launch
count, and each has a number.

**C1. Register pressure. Measured, and it is the biggest one.** An argument in
the constant bank can be re-read for free by `ptxas` at any point; an argument
reached through a runtime item index cannot, so values stay live. Measured with
`ptxas -O3` at sm_90:

| kernel | base | merged, global | merged, `__constant__` | **merged, shared (shipped)** |
|---|---:|---:|---:|---:|
| stage 1 | 40 reg / 96 B stack | 70 / 160 | 71 / 160 | **64 / 160** |
| stage 2 | 48 / 112 | 121 / 144 | 121 / 144 | **113 / 144** |
| stage 3 | 59 / 96 | 117 / 144 | 123 / 144 | **104 / 144** |

The shared form is the cheapest of the three and is what shipped; the
`__constant__` form was tried and is worse, which says the cost is the dynamic
index and not the address space. At 64 threads a block, stage 3 goes from
16 resident blocks per SM to 8. **That is not binding for the median launch** —
`stage3_imbalance.md` puts the median queue at 12 entries against 512-2048
blocks, so eight items is still ~96 items against ~1,360 resident blocks — but
it is binding at the p99 (2,413 entries) and the max (4,434).

**C2. The merge takes the fan-out away from this phase.** The loop it replaces
runs `nsegs` chains concurrently on eight streams, so item 7's stage 2 can run
while item 0's stage 1 is still going. The merged form puts a launch boundary
between the stages of every item in the chunk. Width inside a stage goes up;
pipelining across stages goes away. For scale, the fan-out was worth
**+2.73 ms/frame** in the count phase (`SESSION_HANDOFF.md` §1) — a different
phase and a different scale, but the same mechanism, and it is the reason a
regression here would not be surprising.

**C3. It competes with PDL, which is landed and default-on at level 3.** In the
control arm the fill loop offers two links per segment (stage1->2, 2->3), about
**269 links/frame** on old; merged it offers two per chunk, about **34 to 61**.
So the merged arm converts ~210 fewer links, and PDL's own credit for them
moves into the merge's column. **Measure both arms at the PDL default and read
`pdl_taken` in both.**

**C4. The arm is not the default.** Both arms need
`CUDAVK_NO_ABUF_APPEND=1`. Its control is not the 13.16 ms baseline and must
not be compared to it (R6).

---

## 6. The A/B command

Strictly alternating, one session, one binary, GPU exclusive under a 1 Hz
sampler, per `WORKFLOW.md` §4 and R6.

```bash
D=/tmp/perf-audit/merge \
I=/tmp/merge-tree/build-cudavk-merge/src/cudavk/cudavk_devenv_icd.x86_64.json \
BASE_ENV="CUDAVK_NO_ABUF_APPEND=1" \
CAND="CUDAVK_MERGE_FILL_RAST=1" \
CTRL="" \
bash /tmp/perf-audit/cp_decisive_ab_fixed.sh
```

One counter run per arm, separately, because `PLAN_STATS` prints to stderr and
is not a timed run:

```bash
env VK_DRIVER_FILES=$I CUDAVK_NO_ABUF_APPEND=1 CUDAVK_PLAN_STATS=1 \
    <replay old>            # control counters
env VK_DRIVER_FILES=$I CUDAVK_NO_ABUF_APPEND=1 CUDAVK_MERGE_FILL_RAST=1 \
    CUDAVK_PLAN_STATS=1 <replay old>   # candidate counters
```

Do **not** put `CUDAVK_PLAN_STATS` in a timed arm.

---

## 7. The evidence a reviewer can check, from counters that already exist

No new counter was added. Three existing lines settle whether the mechanism
fired, before any timing is believed:

1. **`cudavk: kernel launches: N over M episodes and K render scopes (X per
   episode)`** (`cp_renderer.c:825-828`). This is the whole experiment in one
   number. The candidate's launches/frame must fall by **313 to 353** on old.
2. **`cudavk: programmatic dependent launches: T of O offered took ...`**. `T`
   must fall by roughly **210/frame** on old — the links whose launches the
   merge deleted (C3).
3. **The stdout sha256 the A/B script already writes per run.** It must be one
   value across both arms and every repeat.

---

## 8. Falsifiers, registered before any measurement

Five, not one — R8, and because the informative case is two of them
disagreeing.

**F1 (mechanism).** Launches/frame in the candidate arm falls by **at least
250** on old. *If it does not, the merged path did not fire* (wrong flag, wrong
arm, `nsegs` far smaller than the census says, or the fill loop unreachable) and
**no timing from that run may be read at all.** This is the "PDL that converted
nothing looks exactly like PDL" check.

**F2 (price transfer).** The frame improves by **+0.20 to +0.34 ms** on old.
*Below +0.10: the 0.782 us price does not transfer to this site.* *Above
+0.45: something other than launch count moved and the attribution is wrong.*

**F3 (correctness).** One stdout sha256 across both arms and all repeats.
*A second hash falsifies §2.2 — and specifically inference I6, that the sort has
no append-order tie-break — and the patch is wrong, not the theory.*

**F4 (the device-side confound, C1+C2).** Kernel time of the merged stage 3 per
item is **no worse than 1.25x** the control's `cp_rasterize_stage3_abuf` per
item (nsys, total class time divided by items processed). *If F1 passes and F2
fails low while F4 fails, the loss is register pressure or the lost stage
pipelining and NOT the launch price* — the launch axis would then still be
open, measured through a mechanism that happened to be expensive.

**F5 (the PDL interaction, C3).** `pdl_taken`/frame falls by **150 to 260** in
the candidate arm. *If it does not fall, the control's fill links were already
declining* — in which case C3 is void, the merge is not taking credit from PDL,
and F2's band should be read as the whole effect rather than a net one.

### What I expect, and where I expect to be wrong

I expect **F1 and F3 to pass and F2 to come in at or below the bottom of its
band**, because this is the position where a launch is *most* hidden: the loop
is back-to-back on eight streams behind one join, and 74.9% of real chain
kernels already start with zero union idle. The 0.782 us was measured by
injection into a real queue, which is this situation, so if the price is real it
should transfer — but if it transfers at half strength here, the honest reading
is that the fan-out was already hiding it, not that the price is wrong.

I expect **F4 to be the interesting one.** The register table in C1 is the only
part of this patch I could not make free, and if the merge loses, that is where
I would look first.

The pre-written ending that would embarrass me: **F1 passes, F2 fails low, F4
passes.** That combination says the launches were removed, the kernels did not
get worse, and the frame did not move — which would mean a launch removed from
*this* queue is worth far less than 0.782 us, and the 0.433 ms/frame sizing of
the whole merge programme would need re-deriving at whatever price this site
reports.

---

## 9. Files

| what | where |
|---|---|
| patch | branch `raster-merge`, `c6681a43fc8`, worktree `/tmp/merge-tree` |
| build | `/tmp/merge-tree/build-cudavk-merge` |
| SASS/PTX inertness working set | `/tmp/merge-ptx/` (`rtc.c`, `*.ptx`, `*.sass`) |
| host objdump classification | `/tmp/merge-ptx/objdump_evidence.txt` |

Nothing was committed to `/home/alexzhukov/mesa`, and the user's two modified
files there were not touched.
