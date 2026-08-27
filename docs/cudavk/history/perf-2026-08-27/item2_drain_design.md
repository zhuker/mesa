# Item 2 — let the host run past the episode drain: design

Author: design agent, 2026-08-26. Base `43df52c383e` on `cudapipe-vk-native`.
Worktree `/tmp/drain-tree`, build `/tmp/drain-tree/build-cudavk-drain` (BUILD OK,
508/508, unmodified base — the tree carries no patch yet).

**DESIGN ONLY. No mechanism code is written. Nothing was run on the GPU.**

Read with `/tmp/perf-audit/SESSION_HANDOFF.md` (§5, §6.1.1, §12), `PERFORMANCE.md`
§5.2/§5.2b/§6.4, `WORKFLOW.md` §4.

---

## 0. The answer in six lines

* The surviving candidate — run the next episode's **vertex** stage across the
  drain — is **designable without a second A-buffer and without a grow path**,
  and the memory it needs is a **fixed 144 MiB outside the 8.59 GB scratch cap**.
* It attacks the **2.066 ms/frame deferral ceiling only**. It does **not** touch
  the 2.43–2.45 ms inter-submit stall, and §8.2 says why no variant of it can.
* Its value is **not** the 2.066 ceiling: it is the *vertex-phase share* of the
  post-drain host burst, which is a strict subset. The ceiling is a valid upper
  bound and nothing more.
* The symmetry objection **largely does not apply**, for a reason nobody has
  written down yet: the mechanism is the *exact inverse of the probe's own
  perturbation*, at the same point, in the same units. §7.
* There is a probe that settles the whole item for **one flag, two runs, and no
  mechanism at all** — move the existing `CUDAVK_WAIT_SPIN_US` injection from
  *after* the sync to *before* it. §9. It is strictly better than the
  "time the VS phase" pre-test in the brief, and it should be run first.
* **Recommendation: do not build the mechanism yet. Run P0 and P1 (§9). Build
  Tier 1 (§4.1) regardless — it is cheap, contained and independently useful.**

---

## 1. What the drain is, at machine level, at this commit

`cp_pass_finish()` (`cp_renderer.c:8486`) is the episode tail. Line numbers are
`43df52c383e`.

```
8532  cp_pass_join(nsegs)              join the segments' side streams
8542  cp_abuf_scan                     prefix sum over per-pixel counts
8545  abuf_fill_recs (or per-seg fill) place the fragments
8598  abuf_sort_short / abuf_sort      order each pixel's run
8638  cp_abuf_quad_build               the quad stream
8672  abuf_seg_count                   per-segment quad counts
----- the seam ------------------------------------------------------------
8679  cp_sync_timed(cp->stream, ...)   THE DRAIN  — 9.88/frame, 6.017 ms/frame
8682  cuMemcpyDtoH(ctr, ab->counters)  6 counters + nsegs quad counts
8698  if (fill_over || quad_over) fallback
8707  if (!quads) return
8725  segment -> shading-group mapping (host only, counter-INDEPENDENT)
8748  dense bases                      (needs ctr)
8765  grouped / quad_dense / seg_cursor allocations
8783  abuf_seg_scatter
8788  cp_pass_broadcast + per-group interpolate/FS launches
....  abuf_composite
```

The drain is a **read-back-and-decide**, as the brief says. Nothing about the
device chain in front of it changes in any design below; the wait keeps its
length. What changes is what the host does with the time.

**Where a drain comes from, and why "the next episode is in hand" is false.**
Every call site of `cp_pass_finish()`:

| site | cut cause | is the next episode's first batch in hand? |
|---|---|---|
| `9321` `cp_batch_flush_why` | framebuffer / viewport / blend / raster state change, scope | **no** |
| `9341` `cp_render_scope_begin` | begin render | **no** |
| `9358` `cp_render_scope_end` | end render | **no** |
| `9309` `cp_batch_flush_defer_why` fall-through | batch not appendable | in hand, but it goes to the **classic** path, not to a new episode |
| `9030` `cp_pass_append` | pending episode was opaque | opaque tail, no drain |
| `9045` | batch under `abuf_min_tris` | classic |
| `9061` / `9123` | setup failure / append rollback | classic |

A blended episode only *starts* at `cp_pass_append()` with `nsegs == 0`, reached
from `cp_batch_flush_defer_why` when the batch is blended and appendable. So at
the instant of the drain the driver has **no** future draw in hand: they are
still ahead of the cursor in `cmd->ops[]` (`cpvk_device_memory.c:266`).

**Consequence, and it is the design's first constraint.** "Issue the next
episode's VERTEX stage across the drain" cannot be done by reordering work the
host already holds. It requires either (a) a look-ahead planner over `cmd->ops`
— rejected in §8.4 — or (b) **deferring the drain itself past the first appends
of the next episode**. This design takes (b), and everything below follows from
it.

---

## 2. The mechanism

### 2.1 Statement

Split `cp_pass_finish()` at the seam above into a **front half** (join → scan →
fill → sort → quad → `abuf_seg_count`, all launches, no sync) and a **tail**
(sync → counter copy → group → shade → composite). The front half returns with
the episode recorded as **deferred**.

The host then keeps replaying `cmd->ops`. When the next blended batch arrives,
`cp_pass_append()` runs it in **held mode**: `cp_draw_execute_batch()` issues the
vertex phase on a dedicated *ahead stream* and **stops at `cp_renderer.c:6251`**,
before the A-buffer count block, recording a **held segment**. Up to
`CP_AHEAD_MAX_SEGS` batches may be held.

The deferred tail is then **resolved** — sync, counter copy, group, shade,
composite — and immediately afterwards the held segments' count phases are
**replayed** in submission order onto their normal side streams.

So the drain is still taken, at the same point in the device timeline, for the
same length. The host reaches it having already issued the next episode's vertex
work.

### 2.2 Why the seam is at 6251 and not anywhere else

`cp_draw_execute_batch()` is ~1,500 lines with a hundred live locals; splitting
it at an arbitrary point is a refactor nobody should sign off. **Line 6251 is
different.** By that line the two things the count block needs are already
complete, self-contained, plain-old-data value structs:

```c
struct cp_rasterize_args aa;     /* 344 bytes, fully populated at 6253-6301 */
struct cp_rast_queues  rast_queues;  /* 56 bytes */
struct cp_pending_clip pending_clip; /* the held clip, args complete at 5858 */
```

plus five scalars (`rast_num_triangles`, `num_triangles`, `fetch_fold`,
`abuf_recs_filled`, `cp->fs_batch.prim_shift`) and the batch, which
`cp_pass_record_segment()` **already copies by value** (`sg->batch = *batch`,
`8979`; `sizeof(struct cp_pass_seg)` is 56,448 bytes, measured). The held record
is therefore a struct of about **57 KB** and the replay is a verbatim copy of
lines 6295–6365. Nothing else crosses the seam.

`CP_AHEAD_MAX_SEGS = 8` → **452 KB of host memory**, one `calloc` at first use,
beside the 3.6 MB `cp->pass_segs` already allocated.

### 2.3 The clip stays with the count, on purpose

The clip launch is already deferred inside `cp_draw_execute_batch`
(`pending_clip`, `5858`) so that `clip_rast_fused_abuf` can run clip and stage 1
as one kernel. **The held record carries `pending_clip` unissued**, and the
replay issues the fused kernel exactly as today.

This is not a detail. Splitting the clip out would:

* add ~134.6 launches/frame on old (one per abuf segment) at the settled
  removal price, **0.105 ms/frame**, and
* change stage 1's grid from the clip kernel's `(num_triangles+63)/64` back to
  `(max_clipped+255)/256` with `max_clipped = 8 × num_triangles` — a real work
  increase, not just a launch.

**Correction to my own first draft, because it matters for the arena size.**
Holding the clip *launch* does not by itself hold the clip's *output
allocations*: `clipped`, `clip_count`, `prim_refs` and `active_ids` are taken
from `dscratch` at `5759–5821`, inside the vertex phase, and only the launch is
deferred. `clipped` alone is `max_clipped × 3 × out_stride = 384·T·V` bytes
against the VS output's `48·T·V` — **eight times larger**. Left where they are,
the brief's "one more VS output buffer" would be nine.

So held mode must additionally **move `clipped` and `prim_refs` to replay
time**, allocating them from the freshly rewound `dscratch` and patching four
fields of the held `aa` (`positions`, `prim_refs`, `tri_count`, `active_ids`)
and four of the held `pending_clip.args` (`out`, `out_count`, `active_ids`,
`prim_refs`). That is eight mechanical assignments, and it is the one place
where the replay is not a verbatim copy of `6295–6365`. It must be written with
the `cp_restore_preclip`/`cp_rast_args_unclip` pair (`6329–6333`) in view, since
those already exist to undo exactly this substitution on the fused-launch
failure path.

`clip_count_early` (4 bytes) and `active_ids_early` (`32·T`) stay in the ahead
arena: the vertex shader **seeds** `clip_count_early` under `fetch_fold`, and
`stable_clip_early`'s seed value depends on whether `active_ids_early` exists
(`5503`). Forcing `active_ids_early = 0` instead would push held stable-clip
segments onto the hole-filled path that rasterizes all eight slots — correct,
and a real work increase. 160 KB per segment is cheaper than that.

### 2.4 What runs ahead, exactly

Held mode issues, on the ahead stream, for each held batch:

* `cp_build_vertex_refs()` and the host id loop (`5249–5259`) — a malloc,
  an O(3T) loop, two memcpys, a free. **Pure host work, and the largest host
  item in the vertex phase.**
* the slice table and UBO row uploads (`cp_upload`, flushed by
  `cp_stream_set()` on the stream switch, `618`)
* `cp_vertex_fetch` when the shader declines fusion (10.4 launches/frame)
* the vertex shader (`5740`, 221.5 launches/frame class-wide, 1.843 ms/frame of
  kernel time)

Held mode does **not** issue: the clip, stage 1/2/3 abuf, or anything that
touches `ab->*`.

### 2.5 `fetch_fold` is kept, by pinning held segments to their own queue sets

`fetch_fold` (`5460`) lets the fetch/VS launch seed `cp->cur_qset.counts` (3
words) and `clip_count_early` (1 word), replacing two small clears — that is
iteration 26 S2, and `CUDAVK_VFETCH_SKIP_SEED` exists as its negative control.
Under held mode the seeding happens on the ahead stream while the reader —
stage 1 — runs on a side stream. Two things could go wrong and only one of them
does.

**Ordering: solved by an edge that is needed anyway.** §5.2 already requires one
event on the ahead stream after the last held VS, waited on by each side stream
before its first replayed launch. That edge orders the seed before its reader.

**Aliasing: solved by pinning, and it happens to be free.** Two held segments
sharing a queue set would both seed the same `counts` before either stage 1 read
it. But `cp_pass_append` assigns segment *k* to `seg_streams[k % 8]` and
`seg_qsets[k % 8]`, and held segments are the *next* episode's segments 0..7 —
so with **`CP_AHEAD_MAX_SEGS = 8` they already get one queue set each**, and no
aliasing is possible. The cap is load-bearing, not arbitrary.

**Cost of the ordered form: one `cuEventRecord` plus up to 8
`cuStreamWaitEvent` per episode.** Those are host-side dependency edges, not
device operations: ≈0.5 µs each, ≈9 per episode, ≈89/frame, **≈0.04 ms/frame**.

**The simple fallback, if the pinning invariant proves fragile:** set
`fetch_fold = false` for held segments and pay two clears each at replay time.
Price it through the driver's own table, not through a guess — §5.1 puts 240
memsets/frame at 0.190 ms, i.e. **0.79 µs each** — so 16 clears/episode ×
9.88 = 158/frame = **0.125 ms/frame**. **But if those clears behave as *exposed*
launches rather than as memsets — and they are issued immediately after a drain,
which empties the stream by definition, so exposure is the expected case — the
price is 1.97 µs and the cost is 0.31 ms/frame**, half the forecast win. That is
the difference between the two prices in §12.1 of the handoff and it is exactly
why the ordered form is the design and the simple one is the fallback.

### 2.6 Streams

Held vertex work must **not** go on `seg_streams[k]`. Today `cp_pass_append`
puts segment *k* on `seg_streams[k % 8]` and the tail's shade groups go on the
same eight streams. If a held VS were issued on `seg_streams[0]` before the
tail, the tail's shade would queue **behind** it and the composite — the last
link in the frame's chain — would start later. That is the peel patch's failure
mode reproduced exactly.

**Design: `CP_AHEAD_STREAMS = 2` new `CU_STREAM_NON_BLOCKING` streams**, created
once beside the existing eight, held segment *i* on stream *i % 2*. They carry no
rasterizer queue set (the count is what uses one, and the count is held), so the
per-stream cost is one `cuStreamCreate` and nothing else. Two rather than eight
because §3.7 already measured that eight side streams over-serve a machine that
`stage3_abuf` fills at 0.15 waves/SM; adding eight more buys nothing and costs
eight `cuStreamWaitEvent` edges at every join.

---

## 3. Exactly which allocation lives where, and how big

### 3.1 The two ahead arenas

The vertex phase's outputs must survive (a) `cp->dscratch.used = 0` at the next
episode's first segment (`5098`), (b) `cp_scratch_begin()`'s reclaim (`675`,
which is a `cuCtxSynchronize` plus `cp_scratch_reset`), and (c)
`cp_pass_fallback()` re-executing the failed episode's segments with
`appending == false`, which does both of those.

They therefore may not come from `cp->scratch` or `cp->dscratch`. Two dedicated
arenas, allocated once, never grown, never freed until context teardown:

| arena | allocator | default size | holds |
|---|---|---|---|
| `cp->ahead.dev[2]` | `cuMemAlloc` | **2 × 64 MiB** | `vs_output_buf` (`5190`), `vs_input_buf` (`5215`), `out_vid`/`out_iid` (`5284`,`5290`), `batch_rows` (`5341`), `active_ids_early`, `clip_count_early` |
| `cp->ahead.host[2]` | `cuMemAllocManaged` | **2 × 8 MiB** | `vfetch_vid`, `vfetch_iid` (`5246`,`5247`) — host-written |

Both are bump allocators with **one rule that S0 and S1b do not have: there is no
grow path.** `cp_ahead_alloc()` returns 0 when the request does not fit, and a
zero **declines run-ahead for that batch**: the batch backs out of held mode and
executes normally, on the normal stream, from the normal arenas. A decline costs
nothing but the bytes already handed out from `ahead.dev` for that batch, which
are reclaimed when that half is next reused.

**Why two halves, and it is not optional.** The obvious rule — reset the bump
pointer after the held counts are replayed — is racy, and the race is subtle
enough to be worth writing out. The order is: E's front half issued → E's tail
deferred → **E+1's segments held** → E's tail resolved (the sync) → E+1's counts
replayed → … → E+1's front half → E+1's tail deferred → **E+2's segments held**.
E+2's holds therefore happen *before* any host sync has proved that E+1's
replayed count kernels have finished reading the ahead arena — `cp_pass_join`
(`8532`) is a device-side `cuStreamWaitEvent`, not a host wait. A single buffer
would hand E+2's vertex shader the bytes E+1's stage 1 is still reading.

Alternating two halves per episode fixes it with no sync at all: E+1 reads half
A, E+2 writes half B, and by the time E+3 writes half A again, E+1's counts are
provably complete because E+1's own drain has been taken and it waits on the
chain behind them. This is the same generation trick `cp->scratch.base[]`
already uses over `CP_FLUSH_GENS`, at a depth of 2 instead of 8, and the
argument for depth 2 is the drain in the middle.

### 3.2 Sizing, per held segment

With `T` triangles, `V = num_vs_outputs`, `E = num_vertex_elements`,
`total_verts = 3T`, `out_stride = 16V`:

| buffer | bytes | at T=5,000, V=4, E=3 |
|---|---|---:|
| `vs_output_buf` | `48·T·V` | 960 KB |
| `vs_input_buf` (declining shaders only) | `48·T·E` | 720 KB |
| `active_ids_early` (stable clip) | `32·T` | 160 KB |
| `batch_rows` (batched draws) | `12·T` | 60 KB |
| `clip_count_early` | 4 | 4 B |
| **device total** | | **≈ 1.9 MB** |
| `vfetch_vid` + `vfetch_iid` (managed) | `24·T` | 120 KB |

At 8 held segments that is **≈15 MB device / 1 MB managed per half**, against
per-half caps of 64 and 8 MiB. The caps are ~4× the expected demand, which is
the margin that keeps the decline rare. **`T = 5,000` and `V = 4` are
placeholders, not measurements** — P1 (§9) reports the real distribution and the
caps are set from it, not from this table.

**The worst case is what sets the cap, not the mean.** `CP_MAX_BATCH_TRIS` is
262,144 and `CP_MAX_CLIP_SLOTS` is 16, so one segment can legally want
`48 × 786,432 × 16 = 201 MB` of VS output. **That segment declines**, which is
correct and cheap. It also means run-ahead is structurally biased toward small
batches — and small batches are the ones with the *least* host cost to move.
§9's P1 must report the joint distribution of (host vertex-phase time, ahead
bytes) per segment, or the admission threshold cannot be set honestly.

### 3.3 Admission

A batch enters held mode only if all of:

1. a deferred tail exists,
2. `held_count < CP_AHEAD_MAX_SEGS`,
3. the batch is blended and `cp_pass_appendable()` is true,
4. every `cp_ahead_alloc()` for it succeeds,
5. `total_verts >= CUDAVK_AHEAD_MIN_VERTS` (default from P1; **0 means "no
   threshold"** and must not be the shipped default),
6. `ab->grow_to == 0` — see §6, failure mode 1.

Otherwise the tail is resolved first and the batch runs normally. That is always
legal and is the base case of the correctness argument.

---

## 4. Tiers, so that the cheap part is not hostage to the expensive part

### 4.1 Tier 1 — hoist the counter-independent host work above the sync

`cp_pass_finish`'s tail does real host work that does **not** read `ctr[]`:

* the segment → shading-group mapping, `8725–8745`, `O(nsegs²)`;
* the group's fs-UBO row concatenation, `8821+`, which memcpys
  `members × ndraws × CP_ARG_UBO_STRIDE × 8` bytes per group and allocates
  `cp->pass_group_ubos` on first use;
* `descs[]` zeroing, `8792`, `64 × sizeof(struct cp_seg_desc)`.

All of it can move above line 8679. This is a **pure reorder inside one
function**, no new arena, no new stream, no held state, no change to any launch
or its arguments, and the output is byte-identical by construction. It is worth
whatever that host work costs — which P1 measures — and it is the honest floor
of this item.

**Build Tier 1 whether or not Tier 2 is approved.** One flag,
`CUDAVK_NO_DRAIN_HOIST`, registered off; off means the hoist is live and the
flag is the revert, so **the control arm carries the flag** (WORKFLOW §4.3). The
byte-identity gate for Tier 1 is therefore run with the flag *set*, and the
inertness `objdump` is against the flag folded to `true`, not `false`.

### 4.2 Tier 2 — the held-segment run-ahead of §2 and §3

### 4.3 Tier 3, named and not designed — opaque episodes ahead of a blended drain

An opaque episode uses `visbuf` and its own arrays, not the A-buffer, so the
conflict §5 spends most of its length on does not exist for it. If P1 shows the
op stream often puts an opaque episode after a blended one, Tier 3 is a strictly
easier version of Tier 2. It is not designed here and it is not in the forecast.

---

## 5. Correctness

### 5.1 What the invariant actually is

`cp_pass_can_retry()` (`7468`) plus the drain enforce: **no fragment shader for
episode E may run before E's overflow answer is on the host.** The run-ahead work
is `cp_vertex_fetch` and the vertex shader for episode E+1. It is not E's
fragment shader, it does not read or write any `ab->*` array, and it does not
touch the framebuffer. **The invariant is untouched.**

The brief's phrasing — "`cp_pass_can_retry` forbids running the FS before the
overflow answer" — is why S1b needed a second A-buffer and this does not. S1b ran
the next episode's *count* ahead, which writes `ab->counts` and `ab->recs`. This
runs only the stage in front of the count.

### 5.2 Ordering, stated as edges

* **E's tail vs. held VS.** Independent. Different buffers, different streams,
  no edge needed, and that is the point.
* **Held VS vs. its own replayed count.** `cp_pass_append` already records
  `cuStreamWaitEvent(seg_streams[k], cp->pass_gate)`. The replay must
  additionally record an event on the ahead stream after the last held VS and
  make each side stream wait on it. **One event record + up to 8 waits per
  episode.** This edge is required for correctness *and* it is what lets §2.5
  keep `fetch_fold`; it is the mechanism's entire ordering cost.
* **Held VS vs. E's front-half kernels.** Independent — E's front half reads
  `ab->*` and E's segments' clipped streams; the held VS writes only the ahead
  arena.
* **Submission order.** `cp_pass_record_segment()` is called at **hold** time,
  not at replay time, so `prim_base` is assigned in submission order and
  `pass.next_prim` advances as today. The replay issues in index order. The
  sort's ascending primitive order is submission order, unchanged.
* **Uploads.** `cp_stream_set()` flushes owed uploads on the stream being left
  (`618`), so switching to and from the ahead stream orders the held batch's
  slice-table and UBO uploads correctly with no new code.

### 5.3 Forced resolution points

The deferred tail **must** be resolved (and the held counts replayed) before any
of:

1. `cp_pass_finish()` is called again — including E+1's own close;
2. `cp_abuf_size_arrays()` would run, i.e. `ab->grow_to != 0`;
3. `cp_abuf_setup()` would run for a different framebuffer size;
4. `cp_scratch_begin()` wants to reclaim (`675`);
5. the upload arena wants to rewind;
6. `cp_batch_flush()` at the end of `cpvk_queue_submit` (`cpvk_device_memory.c:353`);
7. `cpvk_DeviceWaitIdle`, any query, copy, fill, dispatch or barrier op;
8. `cp_pass_fallback()` for E — resolution *is* the fallback's trigger, so this
   is ordering, not a new case;
9. `cp->device_fatal` latches.

**1, 4, 6 and 7 are the ones that will be forgotten.** The safe implementation is
a single `cp_ahead_resolve(cp)` called at the top of every function that reaches
CUDA outside the draw path, and an assertion in `cp_launch()` that no launch
outside the held set is issued while a tail is deferred. That assertion is the
design's main defence and it costs one branch on a path that already does more.

### 5.4 Byte identity

Flag off, every predicate folds to `false`, the held path is dead, and both
arenas are never allocated. Prove it the way `bounded-clip` was proved
(`SESSION_HANDOFF.md` §10): `objdump -d` the base `cp_renderer.c.o` against the
patched one with the flag folded, and report symbol count, added/removed
symbols, and the residual diff. Anything beyond scheduled `mov`s and nop padding
is a bug in the gating, not in the mechanism.

---

## 6. Failure modes, ranked

**1. An A-buffer growth between hold and replay. Silent corruption.**
`aa.abuf_frags`, `aa.abuf_recs`, `aa.abuf_capacity` are captured into the held
record at hold time. `cp_abuf_size_arrays()` (`6244`) frees and reallocates
exactly those. Today this cannot happen because the args are built and launched
in the same breath. Under the mechanism, a growth requested by E's drain
(`8694–8696`) and served at the next batch would leave the held launches
pointing at freed device memory — no fault, wrong pixels, or a fault at a random
later kernel. **Mitigation: admission rule 6 (§3.3) refuses to hold when
`grow_to` is already set, and resolution point 2 forces the replay before any
growth is served.** This must be an assertion, not a comment.

**2. The mechanism cost exceeds the deferral gain — the peel outcome.**
Everything the mechanism adds is issued *after* the drain returns, i.e. at the
site's own measured slope of 1.02, so it is charged at full price. Sized through
the driver's price table rather than through kernel time:

| added, per episode | count/frame | unit price | ms/frame |
|---|---:|---|---:|
| `cuEventRecord` on the ahead stream | 9.9 | ~0.5 µs host | 0.005 |
| `cuStreamWaitEvent` per side stream | 79 | ~0.5 µs host | 0.040 |
| `cp_stream_set` upload flushes on the extra stream switches | ≤20 | 0.79 µs | ≤0.016 |
| **ordered form (§2.5), total** | | | **≈0.06** |
| *fallback form: 2 clears per held segment, as memsets* | 158 | 0.79 µs | 0.125 |
| *fallback form, if those clears are exposed launches* | 158 | **1.97 µs** | **0.311** |

Against a forecast gain of 0.35–0.90 ms, the ordered form spends 7–17% and the
fallback's bad case spends 35–89%. **This is the failure mode that killed the
peel patch, and the arithmetic above is the reason the ordered form is the
design.** `CUDAVK_AHEAD_MAX_SEGS` exists so the cost can be traded against the
gain empirically; X4 is its falsifier.

**3. The held VS delays the chain the drain waits on.** The ahead stream's VS
kernels compete for SMs with E's fill/sort/quad chain. §12.2 and §11.3 say the
machine is empty around those kernels (`stage3_abuf` median leaves 500 of 512
blocks idle; warps resident and stalled, not absent), so the expectation is
"free". It is an expectation. **Falsifier X3.**

**4. Ahead-arena decline in the middle of a batch.** Rule 4 of §3.3 is checked
per allocation, and the later allocations happen after the fetch launch has been
issued. Backing out then means the batch re-executes normally and its vertex work
runs twice. Correct, but invisible. **It must be counted** (`ahead_declines`,
`ahead_backouts` in `CUDAVK_PLAN_STATS`) or a run where run-ahead never fires
looks exactly like a run where it fires and does not pay — R4/PDL's own lesson.

**5. Held state leaks across a submit.** Resolution point 6. A tail deferred past
`cuEventRecord(done, stream)` would let `vkQueueSubmit` signal completion for
work not yet issued. Fatal and easy to write by accident.

**6. `cp_pass_fallback` for E while held segments exist.** The held VS output is
in the ahead arena, which the fallback's `dscratch` rewind and possible
`cp_scratch_reset` cannot reach — that is why the arenas are separate. The
fallback re-executes E's segments classically; the held counts replay after it.
Correct by construction, and **untestable on the captures**, which never
overflow. `CUDAVK_FORCE_PASS_FALLBACK` (`8523`) exists precisely for this and
must be part of the gate.

**7. Reclaim pressure.** Suppressing `cp_scratch_begin`'s reclaim (resolution
point 4 resolves instead of suppressing) keeps the 11 GB-bloom backstop intact.
If it were suppressed instead of resolved, the backstop would be gone. Named so
it is not "simplified" later.

**8. The upload-arena rewind rate rises.** That site is 0.05 waits/frame at a
1.377 ms mean (0.073 ms/frame). Held uploads add to the arena before a rewind
can happen. Even a doubling is 0.07 ms/frame. Noted, not defended against.

---

## 7. What becomes fail-open that is fail-closed today

Three things, and only three:

1. **The extent of the held state is no longer implied by the call stack.**
   Today, "an episode is open" is `cp->pass.nsegs != 0` and a single function
   owns the tail. After the change there is a second, longer-lived piece of
   state — a deferred tail plus up to 8 held segments — whose invariants are
   maintained by the nine resolution points in §5.3 rather than by control flow.
   A missed resolution point is a silent wrong-pixels bug, not a crash. **This
   is the real cost of the design and it should be weighed as such.**
2. **The A-buffer array addresses become a captured value rather than a live
   read.** §6 failure mode 1. Today `aa.abuf_frags` cannot be stale; afterwards
   it can, and only an admission rule and an assertion stop it.
3. **A back-out is silent.** §6 failure mode 4. Today `cp_draw_execute_batch`
   either runs a batch or the append fails loudly through `append_failed`; a
   held-mode back-out re-executes correctly and says nothing.

Nothing that is fail-closed for *correctness of the drawn image* becomes
fail-open: the overflow branch, `cp_pass_can_retry`, the `running != quads`
check (`8761`) and the `capacity/4` gate are all untouched. That distinguishes
this from S1d, which had to demote two fail-closed checks.

---

## 8. What it attacks, and what it does not

### 8.1 It attacks the 2.066 ms/frame ceiling, and only that

The ceiling is `Σ min(host issue burst after the wait, the wait)` — the most any
mechanism can relocate into the drain. The held vertex phase is a **strict
subset** of that burst: the burst spans the drain-to-drain gap and contains E's
tail, E+1's full appends (vertex *and* count) and any classic draws; we move the
vertex part of E+1's appends only. Therefore

```
value  =  Σ min(V_vertex, W)  ≤  Σ min(burst, W)  =  2.066 ms/frame (old)
                                                     0.757 ms/frame (Crossroads)
```

That inequality is the only rigorous thing that can be said about the size
before P1 runs, and it is worth saying because it is the correct way to use the
ceiling: **as a bound, not as a forecast.**

Two discounts apply before P1 even reports:

* **The last drain of each submit has no successor.** `cp_batch_flush()` at
  `cpvk_device_memory.c:353` closes the episode at the submit tail, so ~2 of
  9.88 drains/frame on old (**20%**) and ~2 of 5.57 on Crossroads (**36%**) get
  no run-ahead at all.
* The median post-drain burst is 306 µs against a 606 µs median wait
  (`PERFORMANCE.md` §6.4, P2), so the site is gap-bound on 72% of waits and the
  vertex share of a 306 µs burst is the operative quantity.

### 8.2 It does NOT attack the inter-submit stall

The 2.43–2.45 ms/frame stall exists because the driver issues device work only
from inside `vkQueueSubmit` and recording touches CUDA not at all (§12.5). This
mechanism moves host work *within* a submit. It queues nothing across the submit
boundary. The last episode of a submit is resolved before `cuEventRecord(done)`
by resolution point 6 — it must be, or the queue signals completion for work not
issued.

The only thing that would attack the stall is issuing device work during command
recording, which is a different item with a different blast radius and is not
this one.

**What is easy to confuse, and should not be.** Shortening the host's blocked
time inside the submit *does* shorten the frame, at slope 1.02 — that is the
measured conversion. But it shortens the frame by returning from `vkQueueSubmit`
earlier and starting the next frame's recording earlier. It does not fill the
stall. Both are real; only the first is this item's.

### 8.3 Kernel-time overlap is upside, not forecast

2.19 ms/frame of VS kernel time issued earlier could overlap E's tail on the
ahead streams. Per the handoff's architecture fact F6, a class's share of kernel time is
not its share of the frame, the host is blocked 73.7% of the frame, and no exclusivity measurement
exists for the VS class. **It is not in the forecast and must not be used to
size the item.** If it appears, it appears as a bonus in the measured frame
delta.

### 8.4 Why look-ahead over `cmd->ops` is rejected

The alternative to deferring the drain is to scan forward in `cmd->ops[]` and
pre-plan the next episode. That needs a second, stateful pass over descriptor
binds, push constants, pipeline binds and scissors to reconstruct
`cp->batch` — i.e. a second planner that must agree with the first exactly, or
the two disagree and the episode boundaries move. R4's lesson (PDL's link
forecast missed 3.3× because it was written in the wrong unit) applies with
force. Rejected on blast radius, not on value.

---

## 9. The probes, in the order they should run

### P0 — **the probe that settles the item without building it.** One flag, two runs.

`CUDAVK_WAIT_SPIN_US` already busy-waits D µs immediately **after** a named wait
returns and produced the +1.0207 slope. Add one bit: **spin before the sync
instead of after it**, at the same site, same D, same scaling by waits/frame.
This is a two-line change to an instrument that already exists on branch
`peel-predicate` and it requires no mechanism at all.

It is the *exact inverse* of the perturbation the mechanism performs. Host time
placed before the sync is absorbed by the wait up to the wait's own length; host
time placed after it lands on the frame at 1.02.

**What it reports, and it reports more than a yes/no.** Frame response is
`≈ 1.02 × Σ max(0, D − W_i) / frames`, so:

* the curve is **flat** while D is below the wait distribution and bends upward
  as D passes it;
* the **slope at D is `1.02 × P(W < D)`** — the probe therefore reconstructs the
  drain's wait CDF, which nothing in the record has;
* the value of relocating `V` ms/drain of host work is read directly off the
  curve as `1.02 × V × 9.88 − Δframe(D=V)`.

Run it on both captures at D ∈ {0, 62, 125, 250, 500, 1000} µs/drain, strictly
alternating with the existing after-sync arm as its control, in one session on
one binary, hashes checked. **Nothing else in this item should be built before
P0 reports.**

P0 also directly tests the mechanism's load-bearing assumption in §6 failure
mode 3, if a second arm issues K empty launches on a dedicated stream before the
sync instead of spinning: if the drain's mean wait is unchanged, the device is
indifferent to work issued on a side stream during the drain.

**Why this is better than the brief's pre-test.** "Time the VS phase right after
each drain; under ~0.1 ms/episode and the idea dies" is the right instinct, but
0.1 ms/episode is 0.99 ms/frame at 9.88 drains — a threshold so high that a
genuinely PDL-sized win (0.43 ms) fails it. P0 measures the response rather than
the input, so it cannot be mis-thresholded, and it is cheaper.

### P1 — the vertex-phase census. One flag, two runs.

`CUDAVK_AHEAD_CENSUS=1`, counters only, no behaviour change. Per drain record
`W`; per inter-drain gap accumulate

* `A` — total host issue time in the gap (**self-check: `Σ min(A,W)/frames` must
  reproduce 2.066 on old and 0.757 on Crossroads**);
* `V` — host wall time inside the vertex phase of appending segments only, i.e.
  `cp_draw_execute_batch` entry to line 6251;
* per segment: `total_verts`, `V`, `E`, ahead-arena bytes by §3.2's formula,
  whether `refs` was built, whether `fused_vfetch` fired;
* how many appendable blended batches follow each drain before the next
  `cp_pass_finish`, capped at 16 — this is the **achievable** hold count and it
  is what falsifier X5 tests;
* `dscratch` and `scratch` high-water at each drain.

Report `Σ min(V, W)/frame` — **the mechanism's exact ceiling** — and the
distribution of ahead bytes per episode, which sets the arena caps and
`CUDAVK_AHEAD_MIN_VERTS`.

### P2 — Tier 1 alone, measured

Tier 1 (§4.1) is cheap enough to build before Tier 2 is approved and it is a
clean A/B on its own flag. Its value is bounded by P1's measurement of the
counter-independent share of the tail's host work.

---

## 10. Flags

House rules: one flag per change, registered in `src/cudavk/cp_debug.c`, default
OFF, no `getenv` anywhere else, `cp_debug_doc.py --check` and `cp_no_getenv.py`
after every registry touch.

| flag | type | default | meaning |
|---|---|---|---|
| `CUDAVK_WAIT_SPIN_BEFORE` | `CP_FLAG_BOOL_VALUE` | 0 | P0: spin before the named wait's sync instead of after it. Rides on the existing `CUDAVK_WAIT_SPIN_US`/`_SITE` pair; adds no site of its own. |
| `CUDAVK_AHEAD_CENSUS` | `CP_FLAG_BOOL_VALUE` | 0 | P1 counters. Probe only, never timed. |
| `CUDAVK_NO_DRAIN_HOIST` | `CP_FLAG_BOOL_VALUE` | 0 | Tier 1 revert. New behaviour is the default, so **the control arm carries the flag** (WORKFLOW §4.3). |
| `CUDAVK_AHEAD` | `CP_FLAG_UINT`, range 0–2 | **0** | Tier 2 level: 0 off, 1 hold at most 2 segments, 2 hold up to `CP_AHEAD_MAX_SEGS`. Opt-in until measured, then the pair flips to `CUDAVK_NO_AHEAD` per house convention, exactly as PDL did. |
| `CUDAVK_AHEAD_MAX_SEGS` | `CP_FLAG_UINT`, range 1–8 | 8 | trades §6 failure mode 2 against the gain. |
| `CUDAVK_AHEAD_MIN_VERTS` | `CP_FLAG_UINT` | from P1 | admission threshold, rule 5 of §3.3. |

Counters added to `CUDAVK_PLAN_STATS`: `ahead_held`, `ahead_declines`,
`ahead_backouts`, `ahead_bytes_peak`, `ahead_resolve_forced[9]` by reason. **Read
`ahead_held` before believing any timing** — a run that holds nothing looks
exactly like a run without the flag (R4, and PDL's caveat 3).

---

## 11. Forecast, registered before any measurement

Session spreads, which anything below must clear: **0.119 ms on old, 0.034 on
Crossroads** (three run medians, one session).

**Predictions**

| # | claim | old | Crossroads |
|---|---|---|---|
| **N1** | P0's before-sync arm is flat near the origin: slope at D = 125 µs/drain is **below 0.25**, against +1.02 after the sync | | |
| **N2** | P0's implied wait CDF puts **P(W < 250 µs) between 0.20 and 0.45** | | |
| **N3** | P1's self-check `Σ min(A,W)/frame` reproduces the census | 2.066 ± 0.15 | 0.757 ± 0.08 |
| **N4** | P1's `Σ min(V,W)/frame` — the mechanism's ceiling | **0.35–0.90 ms** | **0.10–0.35 ms** |
| **N5** | achievable hold count, median appendable blended batches after a drain | **4–10** | **2–6** |
| **N6** | peak ahead-arena bytes per episode | **8–40 MB** | **2–15 MB** |
| **N7** | Tier 1 alone — **REVISED DOWN after reading the hoistable set in the source; see §16** | **+0.01 to +0.05 ms** | +0.00 to +0.02 ms |
| **N8** | Tier 2 delivers 55–80% of N4 after mechanism cost | **+0.25 to +0.65 ms** | **+0.03 to +0.20 ms** |
| **N9** | *device operations* per frame rise by **less than +20** in the ordered form (the added edges are host-side, not launches); host-side CUDA calls rise by ~89/frame | **< +20 ops** | **< +12 ops** |
| **N10** | drain mean wait moves by less than 5% | | |
| **N11** | stdout hash unchanged on both captures, submit counts 3,022 / 2,994 | | |

**Falsifiers (X-numbered so they do not collide with the handoff's
architecture facts F1–F6) — several, per R8, and they are chosen to be able to disagree**

* **X1 (value).** N4 comes in below **0.25 ms/frame on old**. Then Tier 2 is
  dead: 0.25 is twice the session spread and the mechanism cost in §6 mode 2
  eats a win that size. Build Tier 1, close the item, and the record gains the
  vertex-phase share of the burst, which nothing else measures.
* **X2 (instrument).** N3 misses by more than 0.30 ms on old. Then P1 is not
  measuring what the census measured and **nothing may be concluded from N4 in
  either direction** — this is the falsifier that stops X1 from being falsely
  reassuring.
* **X3 (mechanism, device).** The drain's mean wait rises by more than 10%, or
  P0's decoy-launch arm moves the wait at all. Then held work is not free on the
  device and §6 mode 3 is the binding constraint, whatever N4 said.
* **X4 (mechanism, host).** Device operations per frame rise by more than +60,
  or `CUDAVK_PLAN_STATS` shows the ordered form fell back to the clears. Then
  §6 mode 2's bad row applies and Tier 2's net is not attributable to deferral.
  Retry at `CUDAVK_AHEAD_MAX_SEGS=2`, which halves the added edges.
* **X5 (population).** N5 comes in below **3 on old**. Then the value is real
  and unreachable: episodes are not followed by enough appendable batches to
  fill the wait, and N8 is void even if N4 held. **X1 and X5 can disagree, and
  that is the point** — X1 says "there is nothing to move", X5 says "there is
  something to move and nowhere to move it from"; only their disagreement
  distinguishes a dead lead from a mis-designed one.
* **X6 (symmetry, the one the session refuses to drop). It fires at two
  different costs and the cheap one comes first.**
  * *At P0, for the price of one flag:* N1 fails — the before-sync arm is **not**
    flat near the origin, i.e. its slope at D = 125 µs/drain is 0.25 or more
    while the after-sync arm on the same binary reads +1.02. Then host time
    placed before the sync is not absorbed by the wait, the mechanism's premise
    is false, and **nothing at this site should be built at all.**
  * *At Tier 2, if P0 passed:* N4 ≥ 0.35 and the measured frame delta is
    **negative** on old anyway. Then the mechanism, not the premise, is the
    problem — and §6 modes 2 and 3 say where to look.
  Either firing **closes the whole item and is worth more than the patch**,
  because every remaining mechanism at this site rests on the same assumption.
* **X7 (capture split).** Crossroads regresses by more than 0.05 ms while old
  gains. The mechanism cost is per segment and Crossroads has fewer and smaller
  ones (5.57 drains/frame, 36% of them last-in-submit). The driver has no way to
  gate by capture, so this makes the default OFF permanent.

**What I expect to be wrong.** N4 is the number with the least evidence behind
it: it is inferred from a 306 µs median burst and a launch-count decomposition,
not measured. If one prediction in this table fails it will be that one, and
P1 exists to fail it cheaply.

---

## 12. On symmetry, precisely

The session's caveat is: *"adding host time after a drain costs a frame because
the device is empty, whereas recovering wait time does not remove the device
work the wait was waiting for."*

**That caveat does not apply to this mechanism, and the reason is worth stating
carefully because it is the difference between this design and every previous one
at this site.**

This design **does not recover wait time**. The wait keeps its length; the
device work behind it is untouched; the drain is taken at the same point in the
stream order. What moves is host work, from *after* the sync to *before* it.

* Probe: `[ wait W ] [ spin D ] [ issue burst B ]` → frame `+ 1.02·D`.
* Mechanism: `[ issue V ] [ wait max(0, W − V) ] [ issue B − V ]` → frame
  `− 1.02·min(V, W)`.

Same site, same units, opposite sign of the same perturbation. The residual
assumptions are exactly two, and both are testable by P0 before anything is
built:

1. the sync's length does not depend on host launches issued on other streams
   during it (X3, and P0's decoy arm), and
2. the response is linear through the origin in the negative direction as it was
   measured linear in the positive one to 1.976 ms/frame (N1, and P0's curve).

If P0's before-sync arm is flat near the origin while the after-sync arm is
+1.02 on the same binary in the same session, symmetry at this site is
**demonstrated, not assumed** — which is what the peel site has and this one
does not yet.

---

## 13. Recommendation

1. **Run P0.** One flag, two captures, alternating arms, one session. It settles
   symmetry, produces the wait CDF, and either kills or arms the whole item.
2. **Run P1.** One flag, two captures, counters only. It sizes N4 and sets the
   arena caps and the admission threshold.
3. **Build Tier 1** regardless. Small, contained, byte-identical, independently
   useful.
4. **Build Tier 2 only if all four hold:** N1 passes (P0's before-sync arm is
   flat near the origin), N3 passes (P1 reproduces the census), N4 ≥ 0.35 ms on
   old, and N5 ≥ 3. All four are new information whatever the decision is, and
   N1 is the one that can stop the work for the price of a flag.

The item is worth designing and it is not worth building blind. The blast radius
is §7's item 1 — a piece of driver state whose invariants live in nine
resolution points instead of in the call stack — and that is a price worth
paying for 0.25–0.65 ms and not for 0.10.

---

## 14. What is code-verified here, and what is assumed

Separated because §11 is a forecast and the reader must be able to tell which
parts of the design rest on reading the tree and which rest on inference.

**Verified by reading `43df52c383e` in `/tmp/drain-tree`:**

* every line number quoted, and the seam's argument structs
  (`sizeof(cp_rasterize_args)` 344, `cp_rast_queues` 56, `cp_pass_seg` 56,448,
  `cp_draw_batch` 56,016 — compiled and printed, not estimated);
* the seven `cp_pass_finish()` call sites and the fact that none of them holds
  the next episode's first batch (§1);
* that `cp_pass_append` → `cp_draw_execute_batch` returns at `6365` after the
  count, so a segment's vertex work is already done by the drain;
* `dscratch` is rewound only at `!appending || nsegs == 0` (`5098`) and
  `cp_scratch_begin`'s reclaim is a `cuCtxSynchronize` + `cp_scratch_reset`
  (`675–685`);
* `cp_stream_set` flushes owed uploads on a stream switch (`618`);
* the clip's output buffers are allocated in the vertex phase and only the
  launch is deferred (`5759–5871`) — the correction in §2.3;
* `CP_PASS_MAX_SEGS` 64, `CP_PASS_STREAMS` 8, `CP_CLIP_MAX_OUT` 8,
  `CP_MAX_CLIP_SLOTS` 16, `CP_MAX_BATCH_TRIS` 262,144,
  `CP_SCRATCH_MAX_BYTES` 8 GiB, `CP_SCRATCH_RECLAIM_BYTES` 1 GiB;
* the tree builds clean at base: 508/508, `libvulkan_cudavk.so` linked.

**Assumed, and each has a probe or a falsifier attached:**

* `T ≈ 5,000` and `V ≈ 4` per segment (§3.2) — **P1**;
* the vertex-phase share of the post-drain burst (N4) — **P1, X1, X2**;
* that held work on a side stream does not lengthen the drain (§6 mode 3) —
  **P0's decoy arm, X3**;
* that the +1.02 slope runs in reverse (§12) — **P0's before-sync arm, X6**;
* that enough appendable blended batches follow a drain (N5) — **P1, X5**;
* that a memset costs 0.79 µs rather than an exposed launch's 1.97 (§6 mode 2)
  — this one has **no probe**, and it only matters if the ordered form of §2.5
  fails and the fallback is used.

**Not measured by anyone, and named so it is not silently assumed:** the
exclusivity fraction of the VS kernel class, which is what §8.3 would need to
turn 2.19 ms/frame of VS kernel time into any frame claim at all. It is
deliberately absent from the forecast.

---

## 15. Documents to change, if this lands

* `PERFORMANCE.md` §6 item 4 — replace "the obvious route is refuted" with the
  tier structure and P0's result.
* `DEAD_ENDS.md` — a new entry only if X1 or X6 fires; X6's entry would be the
  most valuable in the file, because it would close every remaining mechanism at
  this site at once.
* `FLAGS.md` is generated: `cp_debug_doc.py --check` and `cp_no_getenv.py` after
  every registry touch, per the house rules.


---

## 16. What was built, and two corrections to this document

Built on branch `drain-work`, worktree `/tmp/drain-tree`, on `43df52c383e`.
Run recipe: `/tmp/perf-audit/item2_run_recipe.md`. Inertness:
`/tmp/perf-audit/inert/P0_P1_INERTNESS.md`.

| commit | what |
|---|---|
| `3f62f2174c3` | Tier 1 — the shading-group tables built above the drain. Default on, `CUDAVK_NO_DRAIN_HOIST=1` reverts. |
| `c7f2f8e9f08` | P0 and P1 — `CUDAVK_WAIT_SPIN_US/_SITE/_BEFORE/_DECOY`, `CUDAVK_AHEAD_CENSUS`. All off by default. |

Tier 2 is **not** built, per the approval.

### 16.1 N7 is revised down, and Tier 1's claim is about zero

§4.1 listed three hoistable items. Reading them in the source at implementation
time: the group mapping is about 185 pointer compares, the `descs` memset is
2.5 KB, and only the fs-UBO row concatenation is large. The range-table *upload*
was deliberately left inside the loop — hoisting it would move the copy onto a
different stream, which is an ordering change and not a reorder, and this tier's
whole value is that it cannot change a byte.

**Revised: +0.01 to +0.05 ms on old, +0.00 to +0.02 on Crossroads** — at or
under the session spread. Tier 1 is a tidiness change with a claim of about
zero, and should be recorded as one whatever it measures.

### 16.2 The inertness proof was VACUOUS on the first attempt, and the fix generalises

The first P0 commit contained the flags and none of the renderer code: a
`git checkout` run to undo an unrelated edit had deleted it. **The inertness
check did not catch that — it confirmed it**, reporting a perfect
"0 instruction lines differ" because a folded-off build and a build with the
feature missing are identical unless something distinguishes them.

The missing line is a **positive control**: build a third object with the flags
LIVE and require it to DIFFER from the base. The final measurement is A (base)
vs B (flags live, **+5,872 bytes of text** — the probes are demonstrably there)
vs C (flags folded off, **174 functions in both, none added or removed, 20 of 21
differing bodies differ only in `__LINE__` immediates**, verified by resolving
seven of them back to identical source statements; the one structural difference
is 45 instructions of null-pointer test in `cp_context_cleanup`).

This is `WORKFLOW` §4.4 and the `PLAN_STATS` converted-link caveat applied to a
build instead of a run, and it belongs beside them.

Two implementation rules came out of it and are worth keeping:

* **a field added for measurement goes at the END of its struct.** In the middle
  it shifted every later field by 48 and 16 bytes and put 915 displacement
  changes into the object that had nothing to do with the probe;
* **gate even the parts too cheap to gate.** One unguarded `+=` in
  `cp_sync_timed` costs nothing to run and 363 bytes to prove.

### 16.3 One correction to §9's P1

The design said P1 should record the vertex phase's byte demand from §3.2's
formula. The implemented census deliberately **excludes the clip's outputs**,
for §2.3's reason: the design moves `clipped` and `prim_refs` to replay time, so
counting them would size an arena the mechanism does not need. What is counted
is the vertex output, the vertex input, the id arrays, the batch rows,
`active_ids_early` and `clip_count_early`.
