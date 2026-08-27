# What could actually become one launch in cudavk, and what stops it

Code and existing-data analysis only. Tree `/home/alexzhukov/mesa` at
`e2fea470d04` (branch `cudapipe-vk-native`); **the working tree was not touched,
nothing was built, nothing was run on the GPU, nothing was committed**. Every
per-frame number is quoted from an existing document — `PERFORMANCE.md` §5.1/§5.2,
`/tmp/perf-audit/reprofile_stats.md` B.1/B.4, `/tmp/perf-audit/stage3_imbalance.md`,
`SESSION_HANDOFF.md` §4/§6/§11 — and is attributed where it is used. Line numbers
are at `e2fea470d04`.

Claims about **dependence** carry a `file:line`. Claims I could not read out of the
source are marked **INFERENCE** and carry the one-line check that would settle them.

---

## 0. Verdict first

1. **Inside one draw, the raster chain cannot merge.** Stage 2 reads the queue
   stage 1 writes (`cp_rasterize.cu:890-892`), stage 3 reads the queue stage 2
   writes (`cp_rasterize.cu:982-998`, `:1096-1100`). The launch boundary *is* the
   global barrier that publishes each queue — the kernel says so at
   `cp_rasterize.cu:966-967`. Fusing stage 2 into stage 1 was built and measured
   at **24.2 → 27.2 ms** on the old capture and the reason is recorded in the
   source: *"consumer parallelism has to scale with the work, not with the
   producers"* (`cp_rasterize.cu:1050-1057`). This half of the question is closed.

2. **The same stage across the segments of one episode can merge, and the driver
   already says so in its own words.** `cp_renderer.h:293-301`: *"Segment counts,
   fill relaunches and shades are mutually independent — the per-pixel counts and
   cursors are order-free atomics whose order the sort erases, and the dense
   shading slots are disjoint."* That independence is not a hope: those launches
   already run **concurrently on eight side streams** (`cp_renderer.c:8858-8868`
   blended, `:8966-8981` opaque), which bought **+2.73 ms/frame**
   (`SESSION_HANDOFF.md` §3.7, `reprofile_stats.md` B.1).

3. **How many items that is: about 200 segments per frame, carrying about 600 of
   the driver's 1,314 launches/frame** — 134.6 A-buffer raster triples and 74.5
   direct raster triples (`PERFORMANCE.md` §5.1). Concatenated per episode
   (15.50 episodes/frame closed, `reprofile_stats.md` B.4) that is **≈46 launches
   instead of ≈600: a 42% cut in the frame's launch count**, and a median grid that
   goes from 12 items to ~150.

4. **The merge is already implemented once in this driver — for shading, not for
   rasterization.** An episode's segments are grouped by shading identity and each
   group shades in **one launch across several segments**, with the per-segment
   state moved into a device-side `struct cp_seg_range` table that the
   interpolator indexes per quad (`cp_renderer.c:8474-8509`, `:8557-8634`,
   `cp_rast_types.h:789-809`, `cp_fs_interp.h:271-281`). The measured compression
   is visible in §5.1: **134.6 blended segments per frame shade in 22.3 launches**.
   Section 3 below is therefore not a design sketch; it is "do to
   `cp_rasterize_args` what `cp_seg_range` already did to the interpolator's
   arguments".

5. **The blocker is not the arguments and not correctness. It is *when* the
   launches are issued.** A segment's count raster is issued **at append time**,
   inside `cp_draw_execute_batch()` while the host is still recording the frame
   (`cp_renderer.c:6169-6177`, reached from `cp_pass_append()`
   `:8872`). Merging them means deferring all of them to `cp_pass_finish()`. That
   is a scheduling change, not a kernel change, and it lands on the one site in
   this driver with a **measured conversion of 1.02** (`SESSION_HANDOFF.md`
   §6.1.1): anything that makes the episode finish later costs frame time at par.

6. **So the honest prototype order is: build the merged kernel where the loop
   already exists and no scheduling changes at all** — the per-segment *fill*
   relaunch loop at `cp_renderer.c:8334-8356`, `nsegs` triples issued back to back
   at one host point, reachable with `CUDAVK_NO_ABUF_APPEND=1`
   (`cp_renderer.c:1575-1578`) — **and only then ask, separately, whether the count
   phase may be deferred.** Details in §5.

---

## 1. Inventory: the per-frame launch population

Counts are `PERFORMANCE.md` §5.1 (old capture, `ba8891878df`); the driver's own
counter at HEAD reads **1,314.2 launches/frame** (`reprofile_stats.md` B.4), and
B.4 explains the 1.2% drift against §5.1's 1,389. Every launch in the driver goes
through `cp_launch()` (`cp_renderer.c:490-502`), whose macro is
`CP_LAUNCH` (`cp_renderer.h:723`).

| class | launches/frame | one work item is | threads per item | warps per item | grid, and the site | what the launch's inputs depend on |
|---|---:|---|---|---|---|---|
| `clip_rast_fused_abuf` (stage 1, A-buffer) | 134.6 | one **input** triangle: clip it, then classify/rasterize its outputs | 1 | 1/32 | `(clip_tris+63)/64` blocks x 64 thr, `cp_renderer.c:6148-6151` | host triangle count; VS output buffer of *this* segment |
| `rasterize_stage2_abuf` | 134.6 | one queued nontrivial triangle | 32 (warp-cooperative) | 1 | `CLAMP((tris+7)/8, 1, 512)` x 256 thr, `:6172-6174` | **device** queue written by stage 1 |
| `rasterize_stage3_abuf` | 134.6 | one **(primitive, 64x64 tile)** pair | 64 | **2** | `CLAMP(tris*8, 512, 2048)` x 64 thr, `:6175-6177` | **device** queue written by stage 2 |
| `clip_rast_fused` (stage 1, direct) | 74.5 | as above, visibility-buffer variant | 1 | 1/32 | `(clip_tris+63)/64` x 64, `:6560-6562` | host triangle count |
| `rasterize_stage2` | 74.5 | one queued nontrivial triangle | 32 | 1 | `CLAMP((tris+7)/8,1,512)` x 256, `:6598`, `:6603-6605` | device queue |
| `rasterize_stage3` | 74.5 | one (primitive, tile) pair | 64 | **2** | **fixed 2048** x 64, `:6599`, `:6609-6611` | device queue |
| VS `main` (fetch fused in) | 221.5 | one assembled vertex | 1 | 1/32 | `MIN((verts+255)/256, 4096)` x 256, `:5571-5573` | per-batch arg block (one device pointer) |
| `cp_vertex_fetch` | 10.4 | one vertex's attribute gather | 1 | 1/32 | `(verts+255)/256` x 256, `:5353-5355` | per-batch arg struct |
| FS `main`, A-buffer | 22.3 | one shaded slot (4 lanes = one 2x2 quad) | 1 (4/quad) | 1/8 per quad | `MIN((slots+255)/256, 4096)` x 256, `:3619`, `:3648-3650` | one device pointer to the arg block |
| FS `main`, direct | 74.5 | one covered pixel | 1 | — | same, `:3648-3650` | same |
| `fs_compact` / `fs_interpolate` | 74.5 | one 2x2 quad of the **framebuffer** | 1 | — | `(w/2 * h/2 + 255)/256` x 256, `:3911-3916` | visibility buffer of this draw |
| `abuf_interpolate[_ranges]` | (in "else") | one quad of the episode's quad list | 1 | — | `(num_quads+255)/256` x 256, `:4334` | quad stream; **already takes a per-segment range table** |
| `fs_writeback` | 74.5 | one covered pixel | 1 | — | `MIN((pixels+255)/256, 2048)` x 256, `:3993-3995` | FS output of this draw |
| `abuf_scan_*` | 128.2 | one pixel (block-scan of 512) | 1 | — | `cp_abuf_scan_n` `:1923-1927`, classic chain `:1800-1876` | previous level of the same prefix sum |
| `abuf_quad_*` | 32.6 | one 2x2 block | 1 | — | `:2051-2052`, `:2066-2068` | the episode's sorted runs |
| `abuf_sort[_short]` | 28.7 | one pixel's run | 1 or a block | — | fixed 4096 x 256 `:8388`; `MIN((n+255)/256,4096)` `:8375` | the filled fragment array |
| `abuf_composite` | 14.2 | one covered pixel | 1 | — | `(covered+255)/256` x 256, `:8710-8712`, `:4404-4406` | every shade group's output |
| `abuf_fill_recs` | in "else" 154.4 | one appended record | 1 | — | fixed 1024/2048 x 256, `:6318`, `:8327` | the count pass's record array |
| `abuf_seg_count` / `_scatter` | in "else" | one quad | 1 | — | `:8436-8438`, `:8547-8548` | the quad stream + the host's per-segment bases |
| clears, `peel_advance`, `clip_triangles`, blit/resolve/dispatch | in "else" | pixel / triangle | 1 | — | `:9407-9410`, `:6684-6686`, `:4552-4554` | — |

Two things to read off the table before §2.

* **The whole raster chain is 627 launches/frame in 3 shapes** (§5.1), and every
  one of them is sized from a **host-side triangle count** while its actual work
  is a **device-built queue**. That mismatch is what `stage3_imbalance.md` measured:
  median 12 items into a 512-block grid, 98.3% of launches with `queue <= grid`,
  17.6% with an empty queue.
* **The two stages that starve are exactly the two whose item is big and whose
  count is device-side**: stage 2 (1 warp per item) and stage 3 (2 warps per item).
  Stage 1 and the shader launches are one thread per item and are already wide.

---

## 2. Dependence — the core question

### 2.1 Within one draw: the three raster stages are strictly ordered, and it is a data dependence

* Stage 2's first act is to read stage 1's counter and queue:
  `cp_rasterize.cu:890-892` (`queues.nontrivial_count`, `queues.nontrivial`).
* Stage 3's first act is to read stage 2's counter and queue:
  `cp_rasterize.cu:1096-1100` (`queues.huge_count`, `queues.huge_tiles`).
* Stage 2 publishes the huge-tile setup for stage 3 and names the launch boundary
  as the publication barrier: *"Publish one immutable setup before any tile refers
  to it. The stage-2/stage-3 kernel boundary is the publication barrier."*
  (`cp_rasterize.cu:966-967`).

So merging stage k with stage k+1 needs a device-wide barrier inside one launch
(a cooperative grid, or a persistent-CTA megakernel with an occupancy-limited
grid). It was tried in the weaker form — stage 2 folded warp-locally into the
fused clip+stage-1 kernel — and lost 3 ms/frame; the recorded reason is a
*parallelism* argument, not an engineering accident (`cp_rasterize.cu:1050-1057`).
**Closed.** Anything that merges the stages must first answer that paragraph.

### 2.2 Across segments of one episode: independent, and the driver already exploits it

The driver's own statement of the invariant is `cp_renderer.h:293-301`.
The mechanisms that rest on it:

| mechanism | site | what it proves |
|---|---|---|
| each blended segment gets its own side stream **and its own rasterizer queue set** | `cp_renderer.c:8858-8868`; sets allocated at `:7082-7106` | up to 8 segments' full raster chains run **concurrently**, so no ordering between segments is required for correctness |
| each opaque segment `s>0` likewise; segment 0 stays on the main stream deliberately | `:8966-8981` and the comment `:8944-8965` | same, for the visibility path |
| the join is only at the phases that read **across** segments | `cp_pass_join()` `:7193-7212`, called at `:8305`, `:8358`, `:8675` | the count phase itself needs no cross-segment order |
| per-pixel accumulation is order-free | `abuf_counts` atomicAdd and the one global record cursor, `cp_rast_types.h:203-227`; the sort restores submission order by primitive id, `ARCHITECTURE.md` §2.4 and §4.2 | the *result* does not depend on the order segments run in |
| opaque segments share one visibility buffer resolved by `atomicMin` | `ARCHITECTURE.md` §2.3 (*"A minimum is order-independent, and that single fact is what allows opaque draws to be batched and opaque episodes to share one buffer"*), clear at `cp_renderer.c:8917-8919` | same, for the direct path |

**How far the independence extends.** It extends to every phase before the first
join: the vertex stage, the clip, and all three raster stages of every segment of
one episode. It does **not** extend past `cp_pass_finish()`'s first join
(`:8305`), because the scan reads the counts of all segments.

**Where it stops being exploited today, and why that is not a correctness limit:**

* only **8** streams and **8** queue sets exist (`CP_PASS_STREAMS` 8,
  `cp_renderer.c:7082-7106`) while an episode may hold **64** segments
  (`CP_PASS_MAX_SEGS` 64, `cp_rast_types.h:282`) and opaque episodes on the old
  capture **average 12.42 and reach 49** (`reprofile_stats.md` B.1). Segment 8
  shares a queue set with segment 0 and therefore serialises behind it. The
  8-stream fan-out is a coarse, capped approximation of the merge this document
  is about.
* a segment's queue counters are cleared immediately before its own stage 1
  (`cp_renderer.c:6127` blended count, `:6321` blended fill, `:6546-6548` direct,
  `:8345` fill relaunch), or are seeded inside `cp_vertex_fetch` when the fold is
  on (`:5320-5336`). **This clear is per queue set, not global**: it is the only
  "arena reset between launches" in the raster path, and it is the reason two
  segments may not share a queue set concurrently. A merged launch removes the
  clear rather than working around it — one clear of one shared queue, before the
  merged stage 1.

### 2.3 Across the passes of one draw: dependent, by construction

* **Peel passes.** Each pass re-clears the visibility buffer and advances
  `peel_next` with a kernel of its own (`:6504-6508`, `:6683-6686`), and the host
  reads `peel_any` to decide whether to stop (`:6688-6697`). Pass *n+1*'s coverage
  depends on pass *n*'s output. **Cannot merge.** The queues are the exception,
  and the driver already exploits it: `CP_QUEUE_REUSE` rebuilds nothing
  (`:6513-6514`, `cp_rast_types.h:1244-1261`).
* **Alpha-test retry passes** (`CP_DISCARD_LAYERS` 8, `cp_rast_types.h:306`):
  same shape — each pass reads the `reject` layer the previous one wrote
  (`cp_rast_types.h:151-159`). **Cannot merge.**

### 2.4 Within the A-buffer support chain: dependent

The scan is a prefix sum: reduce then finish (`cp_abuf_scan_n` `:1923-1927`), or
the classic three-level chain (`:1800-1876`). Level *k+1* consumes level *k*.
The quad build is count → scan → fill (`:2046-2069`). `abuf_seg_count` →
host drain → `abuf_seg_scatter` has a **host readback in between**
(`:8436-8438`, drain `:8443-8451`, scatter `:8547-8548`) — this is the one class
in the driver where a host counter read genuinely separates two launches, and
the bounded path already removes it for the draws it can prove
(`cp_pass_finish_bounded_groups` `:8005-8118`, `:8093-8116`). **Cannot merge**;
already attacked from the other side.

### 2.5 Across batches that are not in one episode: separated by host work, not by a dependence

A non-episode draw's chain is issued from `cp_draw_execute_batch()` while the host
records the frame. Two consecutive such draws are usually independent, but there
is **no host point at which both are pending**, so "merging" them means holding
the first back. This is the same deferral question as §2.2, with a worse ratio:
episodes at least have an explicit end (`cp_pass_finish`).

### 2.6 What is already merged, and what that measures

| already merged | site | compression measured |
|---|---|---|
| draws → one batch (up to 128 draws, one VS + one raster chain) | `cp_batch_key` `ARCHITECTURE.md` §4.1, `cp_renderer.h:74` | the batch is why 221.5 VS launches serve far more draws |
| segments → shading groups, blended | `:8474-8509`, `:8557-8670` | **134.6 segments → 22.3 FS launches/frame** (§5.1) |
| segments → shading groups, opaque | `:7504-7528`, `:7540-7611` | same mechanism, `cp_seg_range` table |
| clip + stage 1 | `:6148-6151`, `:6560-6562` | one launch instead of two, everywhere |
| count + record append (single-pass build) | `cp_rast_types.h:218-227`, fill at `:8318-8328` | removed the whole per-segment fill relaunch loop from the default path |

**The pattern of every accepted merge in this driver is the same: a per-launch
uniform became a per-item record.** That is exactly §3.

---

## 3. The argument problem

Today a raster launch carries **400 bytes of per-launch state as kernel
arguments**: `struct cp_rasterize_args` = **344 B** and `struct cp_rast_queues` =
**56 B** (measured by compiling a `sizeof` probe against
`src/cudavk/kernels/cp_rast_types.h` in /tmp; the header's own `_Static_assert`
already pins `cp_rast_queues` at 56, `cp_rast_types.h:1282-1285`). The kernels take
them **by value** (`cp_rasterize.cu:1300`, `:1310`, `:1320`, `:1330-1348`).

A merged launch cannot. It needs `items[]`, indexed by the item.

### 3.1 What has to move, per class

| merge class | per-item record | size | what stays launch-wide |
|---|---|---:|---|
| blended count triple across segments | `cp_rasterize_args` + `cp_rast_queues` minus the episode-wide fields | **≈400 B raw; ≈250–300 B after hoisting** the episode-wide A-buffer pointers (`abuf_counts/offsets/cursor/frags/overflow/capacity/recs/rec_cursor`, `cp_rast_types.h:203-227`), `width`, `height`, `depthbuf`, `num_samples` | the A-buffer arrays, the framebuffer geometry, `abuf_mode` (all segments are `CP_ABUF_COUNT`, set at `:6132`), the queue **arrays** if one shared queue is used |
| opaque visibility triple across segments | same struct; more of it is uniform because the episode shares one visibility buffer and one depth buffer (`:8917-8919`, `:8936-8937`) | ≈200–250 B | `framebuffer`, `depthbuf`, `width/height`, `num_samples` |
| VS across segments of one shading group | one device pointer to the existing arg block + that item's vertex count | **16 B** | nothing — but see §4.6, the kernel identity blocks this class |

At 400 B and `CP_PASS_MAX_SEGS` = 64, the **whole item table is 25.6 KB**: one
`cp_upload()` into the arena that `cp_launch()` flushes for free on the same
stream (`:496`, `:444-497`). There is no allocation problem and no ABI-size
problem here.

### 3.2 The driver already builds this table — on the host

`struct cp_pass_seg` (`cp_renderer.h:228-241`) stores, **per segment, for the
whole episode**: `struct cp_rasterize_args rast`, `struct cp_rast_queues queues`,
`rast_num_triangles`, `prim_base`, `prim_slots`, `prim_shift`, the batch snapshot
and the slice table. It is filled at `cp_pass_record_segment()` (`:8726-8752`) and
it is **already used to relaunch the identical stage 1/2/3 sequence per segment**
at `cp_pass_finish()` (`:8334-8356`).

So the per-item record for the top merge candidate is not a new object. It is
`sg->rast` and `sg->queues`, uploaded instead of passed by value.

### 3.3 The two device-side precedents the prompt asked me to check

**(a) `cp_seg_range` — confirmed, and it is the exact template.**
`cp_rast_types.h:789-809`, 48 B per segment. Its comment states the problem in the
same words this task uses: *"The fields are exactly what the launch-wide arguments
can no longer be when one launch spans segments."* The interpolator resolves a
quad's segment by binary search on the global primitive id
(`cp_fs_interp.h:271-281`). Built at `:7540-7566` (opaque) and `:8585-8626`
(blended), uploaded with one `cp_upload()`.

**(b) The huge-primitive queue — confirmed, and it has spare bits.**
`struct cp_tile_pair` is 8 B: `uint32 tri_id`, `uint16 tile_x`, `uint16 tile_y`
(`cp_rast_types.h:1218-1222`). A merged stage 3 needs the item index in the entry.
It fits **without growing the record**:

* a segment's triangle ids are indices into its own clipped list, bounded by
  `max_clipped = num_triangles * CP_CLIP_MAX_OUT` (`:5589`) with
  `num_triangles <= CP_MAX_BATCH_TRIS` = 256 K (`cp_renderer.h:58`) and
  `CP_CLIP_MAX_OUT` = 8 (`cp_rast_types.h:344`), i.e. **< 2^21**;
* bit 31 is `CP_TILE_SETUP_TAG` and the tagged form uses only 10 bits of index
  (`CP_SETUP_CACHE_CAPACITY` 1024, `cp_rast_types.h:1225-1227`);
* so **bits 21..30 are free in both forms** — 10 bits, against the 6 needed for
  `CP_PASS_MAX_SEGS` = 64.

The same argument applies to the nontrivial queue, whose entry is a bare `uint32`
tri_id with `CP_NT_HUGE` = bit 31 (`cp_rast_types.h:1263-1268`).

**This is the single most useful structural finding in this document: the merged
raster queues do not need a wider entry, and a merged stage 2/stage 3 is
literally today's kernel with `args` replaced by `items[entry_item(e)]`.**

Two consequences to design against, both readable rather than inferred:

* one shared queue must hold the **sum** over segments, not the max. Capacities are
  `CP_MAX_NONTRIVIAL` 1,000,000 and `CP_MAX_HUGE_TILES` 2,000,000
  (`cp_rast_types.h:1195-1196`), i.e. 4 MB + 16 MB per set, and nine sets exist
  today. Overflow is *silent truncation*: `cp_queue_used()` clamps an unclamped
  atomic counter (`cp_rasterize.cu:411-420`), so the
  merged form must either keep the 2 M bound with the sum, or grow it. Measured
  head-room: the stage-3 queue's **max over 6,669 sampled launches was 4,434
  entries** (`stage3_imbalance.md` §1.2) — three orders of magnitude below the
  cap, and 64 segments of that is still 284 K.
* the setup cache is per queue set with 1024 entries (`:7100-7106`); merged, it
  becomes one shared cache and the index space must cover the sum.

---

## 4. The blockers that are not about merging

**4.1 The launch audit.** `src/cudavk/tests/cp_launch_audit.py` forbids
`cuLaunchKernel(` anywhere but two files and pins the count at exactly 1 in
`cp_renderer.c` and 1 in `cpvk_texture_cache.c` (script lines 15-49). A merged
launch goes through `CP_LAUNCH`/`cp_launch()` like everything else, so **this is
not a blocker** — but it *is* a constraint on how the prototype is written: no
"temporary" direct launch, because `cp_launch()` is also the only flush point for
the coalesced upload span (`:486-502`), and a merged launch's item table lives in
exactly that span.

**4.2 The upload-span rule bites harder after a merge, not less.** `cp_launch()`
flushes on the **current** stream, and `cp_stream_set()` closes the span when the
stream changes (`:510-518`). A merged launch on the main stream is *simpler*: it
removes `cp_pass_gate_stream()`/`cp_pass_broadcast()`/`cp_pass_join()` traffic for
the merged phase (`:7233-7268`, `:7193-7212`) and with it the ~32 µs per-episode
hand-off priced in `reprofile_stats.md` B.6.

**4.3 PDL's epoch check — and the fact that merging *competes* with PDL.**
PDL is not in this tree (grep for `griddepcontrol`/`cuLaunchKernelEx` in
`src/cudavk/` returns nothing); it is branch `pdl-prototype`
(`SESSION_HANDOFF.md` §2). Its predecessor test requires that the previous
`cp_launch()` on this stream was the named kernel and that a **global epoch** has
not moved — no copy, clear, event or upload flush in between
(`/tmp/perf-audit/pdl_landing.md` §1). Two interactions:

* the item-table upload rides in the span that `cp_launch()` flushes immediately
  before the **merged stage 1**, so that launch declines PDL; the
  stage1→stage2 and stage2→stage3 links are unaffected, because nothing is
  uploaded between them. **INFERENCE** (from `pdl_landing.md` §1's own case (a)
  plus `cp_renderer.c:496`). *Check:* run the PDL branch with the merged path and
  read the take/decline counters per link.
* PDL converts **550.87 links/frame** at level 3. Merging the raster triples
  deletes the launches those links join: up to 2 links per segment x ~200 segments
  = **~400 links/frame, ~70% of PDL's converted population**. The two mechanisms
  are not additive; a merge that wins takes most of PDL's +0.434 ms with it, and a
  merge measured *on top of* PDL will read smaller than the same merge measured
  against the shipping default.

**4.4 Flags whose semantics are per launch.** Each must become per item, or the
merged path must refuse the mode:

| flag / field | site | disposition |
|---|---|---|
| `rast_queues.mode` (`CP_QUEUE_FILL/BUILD/REUSE`) | `cp_rast_types.h:1259-1261`, set `:6133`, `:6323`, `:6513-6514`, `:8343` | per item in principle; in an episode every segment is `FILL` (`:6133`), so the merged count/fill path can keep it launch-wide |
| `args.path_flag` / `path_value` (device-side launch predicate) | `cp_rast_types.h:81-86`, honoured at `cp_rasterize.cu:1302-1305` | used only by `CUDAVK_TILED_OPAQUE`, which is already excluded from the fan-out (`:7141`); the merged path should exclude it the same way |
| `args.abuf_mode`, `abuf_prim_base` | `cp_rast_types.h:198-217` | **must** be per item: `abuf_prim_base` is the running episode offset set at `:6113` from `cp->pass.next_prim` (`:8740`, `:8751`) |
| `args.census`, `census_depth` | `:6536-6539` | census mode already refuses episodes (`:7041-7043`); keep the refusal |
| `CUDAVK_NO_SETUP_CACHE`, `CUDAVK_ABUF_SHORT_SORT_MAX`, `CUDAVK_FS_GRID_WAVES` | `FLAGS.md` | launch-wide and stay so |
| `CUDAVK_NO_OPAQUE_STREAMS`, `CUDAVK_NO_PASS_EPISODE`, `CUDAVK_NO_SEG_MERGE` | `FLAGS.md`, `:7124`, `:7037`, `:8494` | a merged path needs its own kill switch of the same shape, and the reverted arm must issue what today's arm issues |

**4.5 Telemetry that attributes per launch.**

* `cp->launches` (`:499`) and the per-episode ratio printed at `:716-719` — a merge
  changes the denominator; every launches/frame number in the documents must be
  re-baselined, not compared.
* `CUDAVK_UPLOAD_STATS` attributes small operations by `__FILE__:__LINE__`
  (`:463-464`, table in `reprofile_stats.md` B.4); the per-site rows move.
* `CUDAVK_DEBUG_TIME` / `CUDAVK_ABUFFER_TIMING` record CUDA events **between**
  launches (`cp_abuf_mark()` `:1136-1141`, marks at `:6134`, `:6178`, `:8346`) and
  difference consecutive pairs. They are **already invalid** at the current default
  because of the fan-out (`:7136-7139`, `reprofile_stats.md` B.6), and a merge
  makes the per-stage breakdown unrecoverable in principle. Episodes already refuse
  `ab->timing` (`:7042`).
* `CUDAVK_TILE_CENSUS` never reduces on a fixed-size workload
  (`SESSION_HANDOFF.md` §11.4) — do not use it to validate anything here.

**4.6 What blocks the VS/FS classes specifically.** A shader launch takes **one
device pointer** to its arg block (`:3588-3589`, `:5548-5549`), and the kernel is a
**separately linked `CUfunction` per shader and per execution mode**
(`ARCHITECTURE.md` §2.5). Merging two segments' VS launches therefore requires
(a) identical `CUfunction` — the episode's segments deliberately alternate two
vertex shaders (`:8477-8479`) — and (b) a generated-shader preamble that maps
`blockIdx` to (item, local index), i.e. a change in `nir_to_ptx`. The FS side of
this has **already been paid** — that is what shading groups are — and the VS side
has not. This is the reason the VS class ranks below the raster classes despite
having the most launches.

**4.7 Correctness gate.** The acceptance criterion is byte-identical output
(`CLAUDE.md` §2, `ARCHITECTURE.md` §8). The merge is defensible on that criterion
precisely *because* the segments already race: the output is already invariant to
the order in which segments' fragments are appended, since the per-pixel run is
sorted by primitive id afterwards (`ARCHITECTURE.md` §2.4, §4.2). **INFERENCE:**
that no tie-break in the sort depends on append order. *Check:* one replay with
`CUDAVK_NO_OPAQUE_STREAMS=1` vs default and compare stdout hashes — if fan-out on
and off already produce identical bytes, append order provably does not matter.
(`reprofile_stats.md` B.4 reports one stdout hash across both arms, which is
strong evidence this check already passed.)

---

## 5. Ranking, and the one to prototype

Score = (items/frame that would be concatenated) x (how independent they already are).

| # | candidate | items/frame | independence today | what stops it | rank |
|---|---|---:|---|---|---|
| 1 | **stage 1/2/3 `_abuf` across the segments of one blended episode** (count phase) | **≈130** (INFERENCE, §6) | **proven**: 8-way concurrent, driver-stated (`cp_renderer.h:293-301`) | issue point is append time (`:6169-6177` reached from `:8872`), per-segment queue sets, per-launch args | **1** |
| 2 | **stage 1/2/3 across the segments of one opaque episode** | **≈70** (INFERENCE, §6) | **proven**: same fan-out (`:8966-8981`), order-free `atomicMin` visibility | same issue-point problem; segment 0 deliberately on the main stream (`:8944-8965`) | 2 |
| 3 | the per-segment **fill relaunch loop** at `:8334-8356` | `nsegs` (≈13/episode) | **proven**, and already a back-to-back loop at one host point | **default-off**: the single-pass build replaces it (`:8318-8328`); reachable with `CUDAVK_NO_ABUF_APPEND=1` (`:1575-1578`) | 3 (as a vehicle, see below) |
| 4 | VS across segments of one shading group | ≈130 | proven (disjoint output buffers, already on side streams) | kernel identity + generated-shader ABI (§4.6) | 4 |
| 5 | `cp_pass_fallback` re-execution loop `:7302-7305` | rare | proven | it is the error path; nothing to win | — |
| 6 | stage k with stage k+1 inside one draw | — | **none**: device queue dependence | `cp_rasterize.cu:890-892`, `:966-967`, `:1096-1100`; measured loss `:1050-1057` | dead |
| 7 | peel / discard-retry passes | — | **none**: each reads the previous pass's output | `:6504-6514`, `:6683-6697`; `cp_rast_types.h:151-159` | dead |
| 8 | A-buffer scan / quad-build levels | — | **none**: prefix-sum dependence | `:1923-1927`, `:2046-2069` | dead |
| 9 | `abuf_seg_count` → `abuf_seg_scatter` | — | **none**: host drain in between | `:8436-8451`, `:8547-8548` | dead (already attacked by the bounded path `:8005-8118`) |

**The honest possibility the task named is half true.** The raster *stages* cannot
merge with each other — that half is confirmed with citations and with a measured
regression. The other half is better than "only the same stage across segments":
that residual is **≈200 segments/frame carrying ≈600 launches/frame, 46% of the
driver's 1,314**, and the mechanism to do it already exists one layer away in the
shading path.

### The one I would prototype first

**Merge stage 1/2/3 `_abuf` across the segments of one episode — and build it in
the fill-relaunch position at `cp_renderer.c:8334-8356` first.**

Why that position, in order of weight:

1. **It isolates the kernel/ABI change from the scheduling change.** That loop is
   already `for (s = 0..nsegs) { 3 launches }` at one host point behind one join.
   Merging it changes *nothing* about when work is issued, so a win or a loss
   there is a statement about launch width alone. Every other candidate confounds
   width with deferral, and the deferral side lands on a site with a measured
   conversion of **1.02** (`SESSION_HANDOFF.md` §6.1.1) — i.e. any delay to episode
   completion is paid at par in frame time.
2. **The item table already exists** (`cp_renderer.h:228-241`, filled `:8726-8752`)
   and the loop already reads it. The prototype is: upload `segs[].rast/.queues` as
   `items[]`, allocate one shared queue set, put a 6-bit item index in the spare
   bits of the two queue entries (§3.3b), and replace `args` with
   `items[item]` in the three bodies.
3. **It is byte-identity testable immediately**: `CUDAVK_NO_ABUF_APPEND=1` on both
   arms makes the path live (`:1575-1578`), and the arm-to-arm output must be
   identical.
4. **It answers the measurement question the session opened and did not answer**
   (`SESSION_HANDOFF.md` §6.3: *"whether launch rate rather than launch count is
   the right axis"*), on a class where `stage3_imbalance.md` has already
   characterised the item cost from both sides: **32–37 k SM-cycles per item when
   ~10 are in flight, 7.8–9.5 k when 2,500 are.** A merged launch of 13 segments'
   queues is the direct test of that 4x.

Then, and only if the merged kernel wins there, the second step is the scheduling
question on its own: may the **count** phase be deferred from append time to
`cp_pass_finish()`? That is a separate patch, a separate flag and a separate
measurement, and it should be predicted before it is run — my own expectation,
registered here: **it loses**, because the frame is host-bound (73.7% blocked,
`PERFORMANCE.md` §5.2), the device idle equals the host's issue time
(3.70 vs 3.48 ms), and deferring the raster of ~13 segments moves ~6.5 ms/frame of
kernel time behind the last draw of the episode.

**Size the prize honestly before building.** By F6 (`SESSION_HANDOFF.md` §11.2),
the whole raster chain **deleted** returns 2.106 ms/frame of device busy time, and
`stage3_abuf` made infinitely fast returns 0.404. A merge does not delete work; it
re-shapes it. Its plausible device-time value is inside that 0.4 ms, and its frame
value is a fraction of that unless the win arrives as launch *rate* — which is
precisely the untested axis and the reason to run the experiment rather than
forecast it.

---

## 6. Inferences, and the one-line check for each

| # | inference | check |
|---|---|---|
| I1 | **≈130 blended segments/frame** — from §5.1's 134.6 `_abuf` raster triples minus ≈4 standalone A-buffer draws (the 4.17 segment-counter waits at `:6420` and 14.2 composites/frame vs 9.88 episode drains) | add `cp->plan.blended_segs += nsegs` in `cp_pass_finish()` (`:8264`) and read it from `CUDAVK_PLAN_STATS` |
| I2 | **≈70 opaque segments/frame** — from `plan.opaque_segs`, which is already printed | one `CUDAVK_PLAN_STATS=1` replay; the line at `:705-715` reports segments and episodes directly. No code change needed |
| I3 | the 74.5 direct raster triples/frame split as ≈70 opaque-episode segments + ≈5 peel/standalone | same run as I2; subtract |
| I4 | 6 bits of item index fit in `cp_tile_pair.tri_id` and in the nontrivial entry without growing either | static: `CP_MAX_BATCH_TRIS * CP_CLIP_MAX_OUT` = 2^21 < 2^30; assert it in `cp_rast_types.h` next to the existing `_Static_assert`s |
| I5 | a merged stage 1's PDL link declines because the item-table upload flushes in front of it; the 1→2 and 2→3 links survive | PDL branch + merged path, read the per-link take/decline counters (`pdl_landing.md` §1) |
| I6 | byte-identity survives the merge because the output is already invariant to segment order | compare stdout hashes of `CUDAVK_NO_OPAQUE_STREAMS=1` against the default on one capture — `reprofile_stats.md` B.4 already reports one hash across both arms |
| I7 | shared-queue capacity is ample (max sampled stage-3 queue 4,434 entries vs a 2,000,000 cap) | `stage3_imbalance.md` §1.2 already measured it; re-read the probe if the workload changes |

---

## 7. What this document did not do

No build, no replay, no profiler, no GPU. The working tree's four modified files
(`docs/cudavk/PERFORMANCE.md`, `docs/cudavk/TODO.md`,
`docs/cudavk/history/CPU_PROFILER.md`, `src/cudavk/tests/cp_cpu_profile.sh`) are
untouched and nothing was committed. The only compilation performed anywhere was a
five-line `sizeof` probe in `/tmp` against `src/cudavk/kernels/cp_rast_types.h`,
used for the three struct sizes in §3.
