# Host sync sites and device-side decisions — cudavk, favorite3 (RTX 5090)

Analyst report, read-only. Repo: /home/alexzhukov/mesa @ cudapipe-vk-native (HEAD 1f2b3c99e00).
Trace: /tmp/postdead/trace.sqlite (nsys 2026.4.1, commit f3caa35e4ce, 2026-08-30, **pre host-pin-readback**;
the frame was 7.60 ms then, 6.68 ms now). Scripts and JSON with every number below:
`/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/host-sync/{bubbles.py,bubbles2.py,timeline2.py,bubbles.json,bubbles2.json}`.

Labels: **MEASURED** = read from a document or computed from the trace by the code shown;
**ESTIMATED** = arithmetic on measured numbers, shown; **HYPOTHESIS** = not measured.

## 0. Window and conventions

- Real work starts at external submit 2782 (WORKFLOW §4.0); `submit-ts.txt[2782] - systemClockNs = 9.275862 s`
  relative; kernel activity in the trace ends at 12.340525 s (last `CUPTI_ACTIVITY_KIND_KERNEL.end`).
  826 submits = 413 frames fall inside; the docs quote 412.5 and I divide by **412.5** throughout for
  comparability (history/perf-2026-08-30-post-dead-scope-profile.md).
- Everything per-frame here is a **window MEAN**, not the project's hot-tail median (WORKFLOW §4.8).
- Host-side durations are **CUPTI-inflated** (CLAUDE.md "Profiling"); device timestamps are not.
- Device "busy" below = union of KERNEL ∪ MEMCPY(copyKind ∉ {11,12}) ∪ MEMSET rows. UVM migration rows (kinds 11/12)
  exist in this trace (70,662 HtoD + 28,564 DtoH in the window, 5.0 + 5.0 MB/frame) and are the readback-buffer
  ping-pong that history/perf-2026-08-31-host-pin-readback.md removed; they are excluded from "busy" but discussed in §5.
- Main thread globalTid 297865344641070; submit-worker thread 297865344641096 (only `cuEventSynchronize`).
- Stream ids: 13 = `cp->stream` (280,808 kernels), 14..21 = the 8 `seg_streams` (CP_PASS_STREAMS = 8,
  cp_renderer.h:329), 7 = null stream (all synchronous `cuMemcpyDtoH_v2` land there), 12/6 = eventless.

Window totals (MEASURED, bubbles.py): wall **7.429 ms/frame**, device union busy **4.390**, device idle **3.040**.

## 1. Catalogue of host blocking sites (a)

Every place the ICD blocks the calling thread on the device. Line numbers are HEAD.

| # | site | call | file:line | value read back | host decision (branch targets) | count/frame (window) | count/frame (whole replay, PLAN_STATS) | blocked ms/frame (window) | device-paced? |
|---|---|---|---|---|---|---:|---:|---:|---|
| S1 | **episode drain** | `cp_sync_timed(cp->stream, wait_episode)` + `cuMemcpyDtoH(ctr, ab->counters, 4*(6+nsegs))` | cp_renderer.c:8884-8890 | `total, fill_over, quads, quad_over, covered` (ctr[0,1,3,4,5]) + per-segment quad counts ctr[6..6+nsegs) | (1) `fill_over‖quad_over` → `cp_pass_fallback` (re-render every segment classically, :8905); (2) `quads==0` → return (:8909); (3) `grow_to` bookkeeping (:8898); (4) `seg_base/group_base` prefix sums from per-seg counts (:8952-8963); (5) `running != quads` fail-closed → fallback (:8964); (6) sizes `grouped`, `quad_dense` (= quads*4 B), `seg_scatter` grid (:8967-8988); (7) per group `cp_abuf_shade(gq)` → `num_slots = align(gq*4,256)` sizes 6–9 arrays and the interp/FS grids (:4504-4527); (8) `covered` sizes the composite grid | **5.44** | 3.74 | **0.558** | yes: `ready=0` at 100% (PERFORMANCE §5.2b); spin p10/50/90 = 42/64/222 µs |
| S2 | **segment counters** (single blended draw, composite path) | `cp_sync_timed(wait_seg)` + `cuMemcpyDtoH(ctr, ab->counters, 24)` | cp_renderer.c:6801-6808 | `total, fill_over, quads, quad_over, covered` | `abuf_prod = q_over==0 && f_over==0 && (quads‖total==0)` → skip the peel loop (`passes=0`, :6878) or run it; `abuf_quads` sizes `cp_abuf_shade`; `covered` sizes the composite; `grow_to` | **3.48** | 1.47 | **0.460** | yes; spin p50 75 µs, p90 183 |
| S3 | **peel checks** | `cp_sync_timed(wait_peel)`, then a host read of managed `cp->peel_any` (no DtoH) | cp_renderer.c:7080-7089 | one managed uint32 written by `peel_advance` | `peel_any==0` → break the peel loop; else next pass, `check_interval*=2` (cap CP_PEEL_CHECK_MAX=16) | **2.48** | 1.13 | **0.253** | yes; spin p50 80 µs |
| S4 | **scratch reclaim** (>1 GiB or 5 overflow arenas) | `cuCtxSynchronize()` then `cp_scratch_reset` | cp_renderer.c:666-690 (`cp_scratch_begin`), called :5211; `CP_SCRATCH_RECLAIM_BYTES = 1<<30` cp_renderer.h:1093 | nothing | none — it frees overflow arenas and rewinds bump pointers | **1.65** (window; the trace cannot separate this from S5) | not in PLAN_STATS; DEAD_ENDS 29 attributes 1.65 ms/frame to this trigger | **1.593** (median 1.07 ms per call) | yes: whole-context drain, waits mostly on the main stream's own tail (side streams quiet a median 479 µs earlier) |
| S5 | upload-arena wrap | `cuCtxSynchronize()` in `cp_upload_begin_checked` | cp_renderer.c:414-427 | nothing | rewinds the managed upload ring | folded into S4's 1.65 (both are followed by `cuMemcpyHtoDAsync`) | "0.05/frame" on old (§5.2b upload-arena rewind) | — | yes |
| S6 | descriptor upload | synchronous `cuMemcpyHtoD` of the recorded descriptor arenas | cpvk_device_memory.c:243-258 | nothing (HtoD) | none; ordering only | **1.006** | 1.00 | **0.0096** | device is already empty (frame start) |
| S7 | frame fence (application `vkWaitForFences`) | `cpvk_sync_wait` → `cnd_wait` (cpvk_sync.c:354-397) on the worker's `cuEventSynchronize(pending->done)` (cpvk_device_memory.c:61) | app: `~/favorite3-cpp/out/src/frame_*.cpp` (e.g. frame_0000_0009.cpp:286-288, 2413-2415) | fence state | app reads back the frame image and records the next frame | 2.0 fences/frame; worker `cuEventSynchronize` 2.0/frame, **0.659 ms/frame** on the worker thread | — | app thread blocked ≈ the readback copy time (see §5) | yes |
| S8 | `cuLaunchHostFunc` (device waits on host, not host on device) | `cpvk_execute_query` (timestamps) cpvk_cmd.c:4842; `cpvk_execute_order_op` (vkCmdSet/Reset/WaitEvents) :4295 | — | the callback fills `pool->results` from `clock_gettime` | none on the host; the **stream** stalls until the callback has run | **3.00** | — | not host time; device gap **0.263 ms/frame** (§3) | n/a |
| S9 | texture-cache view/image destroy drains | `cuCtxSynchronize` in `cpvk_texture_cache_view_destroy` (:854), `_image_destroy` (:990), `_purge` (:953) | cpvk_texture_cache.c | nothing | none | 0.39 (all at frame boundaries, followed by `cuTexObjectDestroy`/`cuSurfObjectDestroy`) | DEAD_ENDS 17: destroy_view 21 µs, destroy_image 0.28 µs per drain | **0.003** (median 0.4 µs — null drains) | device already empty (DEAD_ENDS 17) |
| S10 | indirect draw / dispatch / query copy / `vkDeviceWaitIdle` | `cuStreamSynchronize` + DtoH 16/20 B; `cuCtxSynchronize` | cpvk_cmd.c:1249-1253, 2900-2918, 4792; cpvk_device_memory.c:659 | draw/dispatch parameters | grid/draw counts | **0** in favorite3 (no 16/20 B DtoH; one `vkDeviceWaitIdle` at favorite3.cpp:2919, end of run) | — | 0 | — |
| S11 | debug-only syncs | `cp_abuf_fuse_tally` :1957, `cp_stage_resolve` :2715, `ab->timing` :4748, :6608 first-draw total (4 B, once per run), 7016/7166/7200 verify, 7834 census, 8282 census | cp_renderer.c | counters | printing | 0 unless flags set (the 4 B copy fired once in the whole run) | — | 0 | — |

Per-site consistency checks: DtoH sizes in the whole trace are 28 B ×3359, 36 B ×1392, 56 B ×388, 48/64/44/40 B ×24 (S1 with nsegs = 1,3,8,6,10,5,4; `4*(CP_ABUF_COUNTERS=6 + nsegs)`, cp_rast_types.h:318), 24 B ×1667 (S2), 4 B ×1 (:6609), and no 8/16/20 B copies (no `wait_quads`, no indirect draws) — matching PLAN_STATS "quad counters 0.0 ms/0". The window has 13.44 main-thread syncs/frame (`cuStreamSynchronize`+`cuCtxSynchronize`) against the docs' 16.76/frame on the old capture and the whole-replay 7.34/frame (3.74+1.47+1.13+1.00) — the traced window is the heavy band.

SQL/Python used (bubbles.py): syncs are `CUPTI_ACTIVITY_KIND_RUNTIME` rows on the main thread with `nameId ∈ {cuStreamSynchronize(783), cuCtxSynchronize(699)}`; a sync is "DtoH-bearing" when the next ≤5 runtime rows contain `cuMemcpyDtoH_v2(784)` before another sync; the copy row is joined by `correlationId` to `CUPTI_ACTIVITY_KIND_MEMCPY(copyKind=2)` and classified by `bytes`.

## 2. Structural cost of a sync beyond the spin (c)

Definition per sync: `last_end` = max `end` over device ops whose `start < sync.end` (the op the sync was waiting for);
`next_k` = first kernel `start ≥ copy.end` (or `≥ sync.end` when no copy). **Bubble = next_k − last_end**, split into
return latency (sync.end − last_end), host→copy (copy.start − sync.end), copy, and post (copy.end → next_k, itself
≈ host decision (next launch API start − copy.end) + launch latency (next_k − launch API end)).
Join: `side_quiet` = sync.end − last kernel end on streams 14..21; `main_alone` = last main-stream kernel end − last side-stream kernel end (time the main stream ran alone before the sync).

### 2.1 Per site (MEASURED, window, CUPTI-inflated host components)

| site | n/frame | spin ms/f (median µs) | return lat µs | host→copy µs | copy µs | post µs med/mean | **bubble µs p50/p90/p99/mean** | **bubble ms/frame** | side quiet before sync (med µs) | main alone (med µs → ms/frame) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 episode drain | 5.44 | 0.558 (64) | 0.81 | 4.11 | 0.35 | 9.1 / 10.7 | **14.6 / 30.8 / 43.2 / 17.1** | **0.093** | 38.8 | 37.9 → **0.207** |
| S2 segment counters | 3.48 | 0.460 (75) | 0.9 | 4.3 | 0.4 | 10.6 / — | 16.9 / 118 / 515 / 52.1 | **0.181** (0.128 of it in 258 bubbles >100 µs) | 453 | 452 → 2.05 (not a join: single-stream path) |
| S3 peel checks | 2.48 | 0.253 (80) | 0.8 | — | — | 14.3 / — | 15.1 / 33.5 / 654 / 37.9 | **0.094** (0.047 in 30 bubbles >100 µs) | 193 | 192 → 0.66 (single-stream) |
| S4/S5 ctxsync inside a submit | 1.65 | 1.593 (1072) | 1.3 | — | — | 8.4 / — | 8.4 / — / — / 10.5 | **0.017** | 485 | 479 |
| S9 ctxsync crossing a submit | 0.39 | 0.003 (0.4) | — | — | — | — | 2404 (the frame boundary, not this sync) | (counted in §5) | — | — |
| S8 host-func node (device→host→device) | 3.00 | n/a | — | — | — | — | **77.5 / 155.7 / — / 90.4** | **0.263** | — | — |
| **in-frame total, S1–S5** | 13.05 | 2.864 | | | | | | **0.385** | | |

Drain post-gap distribution (copy end → next kernel start), µs: p10 6.2, p25 8.5, **p50 9.2**, p75 11.8, p90 17.4, p99 27.6, max 38.6, mean 10.7 → **0.058 ms/frame**; of the 9.1 µs median, host decision (grouping, 3 `cp_scratch_alloc_device`, `cp_upload(seg_base)`, memset, launch) is 5.0 µs and launch latency 2.2 µs.

Reading:
1. **The collectible structural cost of every in-frame host sync together is 0.385 ms/frame** (window mean, CUPTI-inflated; the true figure is smaller because ~7 µs of each bubble is CUPTI-inflated API time). That is 12.7% of the 3.040 ms device idle. Against the 5.0 ms goal (gap 1.68 ms on the current 6.68) it is not the answer on its own.
2. The **spin is not a cost**: S4 blocks 1.593 ms/frame and leaves an 8 µs bubble; DEAD_ENDS 29 measured its removal at +0.037 ms/frame. The ratio bubble/spin is the site's true conversion: S1 0.17, S2 0.39, S3 0.37, S4 0.01.
3. **Where the rest of the idle is** (bubbles2.py, exact idle intervals of the busy union): 262,303 idle intervals/window. Intervals containing a submit boundary (±200 µs) **1.489 ms/frame**; intervals containing no sync, submit or host-func (**dependent-launch gaps**) **1.146 ms/frame** over 561 gaps/frame, median 1.3 µs, p90 2.8, 94% under 5 µs — the launch-latency floor the merge rule (DEAD_ENDS 22) already closes; sync-containing 0.148 + 0.024; host-func-containing 0.058 + 0.175 (shared with submit).
4. **Join cost at the drain is a serial single-stream tail, not a wait for stragglers.** Side streams go quiet a median 38.8 µs before the drain returns and the main stream runs alone for 37.9 µs — the scan → fill_recs → sort_short → sort → quad_count_all → scan_finish → quad_fill_all → seg_count chain (cp_renderer.c:8757-8877) at 0.02 waves/SM. 5.44 × 38 µs = **0.207 ms/frame** during which the machine holds one stream of tiny grids and nothing else can be queued because the host is blocked. This, not the 15 µs bubble, is the drain's real structural price, and it is only collectible if independent work (the next episode's raster) can be issued across the drain — which is exactly the run-ahead of PERFORMANCE §6 item 4 (ceiling 0.387 on old, 0.092 collected, unresolved in noise, reach-limited).
5. **S2's long tail is host planning, not device.** 258 of 1436 segment-counter bubbles exceed 100 µs (0.128 ms/frame) and contain only ~5 CUDA calls totalling ~26 µs (1.3 launches, 1 memset32, 1 memset8, 1 HtoDAsync, the DtoH). The host is doing non-CUDA work for 100–500 µs after reading the counters on 18% of these draws. Unattributed here (needs a perf anchor or a `CUDAVK_PLAN_STATS` timer between :6808 and the shade launch); at this site's measured slope (+1.03, §5.2b) it is 1:1 frame time.

### 2.2 Why "spin" numbers mislead, in one mechanism

Every in-frame sync is a gate in a chain. Removing one gate lets the host advance only to the next (S4 → S1 ~100–300 µs later on the same draw), and the device's chain is unchanged; the only saved time is that gate's bubble. DEAD_ENDS 29 is this mechanism measured (1.65 ms blocked → 0.037 gained). It follows that the sites are **not additive individually** and that the value of removing all of them is the sum of the bubbles (0.385) plus whatever *overlap* the freed host creates — and the last gate (the frame fence, §5) is the application's.

## 3. Host-func nodes: a device→host round trip the driver chose (found, not in the docs)

`vkCmdWriteTimestamp` (9/frame in the app) and `vkCmdResetQueryPool` (5/frame) become `CPVK_OP_QUERY` ops that execute as
`cuLaunchHostFunc(dev->renderer.stream, cpvk_query_complete, cb)` (cpvk_cmd.c:4831-4846); the callback reads
`clock_gettime` and flips `pool->available`. A host-func node stalls the stream until a CUDA runtime thread has run the
callback. MEASURED: 3.00 host funcs/frame in the window (1238; the driver coalesces the 14 query commands into 3 order
points), 1200 inside a submit with a kernel following; device gap around each **77.5 µs median, 155.7 p90, 90.4 mean →
0.263 ms/frame**. Two follow `cuMemcpy2DAsync` (buffer copies), one follows a `cuLaunchKernel` at the end of the render.
Because the frame is one dependent chain on stream 13, each stall is on the critical path (HYPOTHESIS for conversion; the
frame-end one delays the readback copy and thus the fence the app waits on, so it is 1:1 by construction).

Device-side alternative: a 1-block kernel writing `%globaltimer` (or `clock64`) into a device query buffer plus an
availability word; `vkGetQueryPoolResults` (called by the app after the fence, device idle) copies the slots back with
one `cuMemcpyDtoH`; `VK_QUERY_RESULT_WAIT_BIT` waits on the submit's done event instead of `pool->changed`.
`vkCmdResetQueryPool` becomes a `cuMemsetD32Async`. Ceiling **0.26 ms/frame** (MEASURED bubble; CUPTI may inflate the
callback dispatch latency — the cheapest probe is `CUDAVK_...` counting nothing: replace the callback with a memset in a
worktree and A/B, or first re-measure the gap with the untraced `tests/cp_gpu_busy.sh` idle delta). Event ops
(`vkCmdSetEvent/WaitEvents`) use the same node but the app records none. Closest DEAD_ENDS: 17 (drains at an already-empty
device) — not the same: here 97% of the nodes have queued work behind them (the "kernel follows" check above).
Blockers: timestamp semantics change from CPU clock to GPU clock (`timestampPeriod` must be reported for the GPU clock
and the two are not comparable across the boundary the app draws between them); the app's `while (vkGetQueryPoolResults
!= VK_SUCCESS) usleep(5000)` loop means a NOT_READY return costs 5 ms — availability must be exact.

## 4. Device-side alternatives per decision (b)

### 4.1 Episode drain (S1)

What the host needs from the device: `fill_over`, `quad_over` (fallback), `quads` (arrays and grids), per-segment quad
counts (group bases), `covered` (composite grid), `total` (growth bookkeeping only).

**(i) Size from the scan total (S1d).** PERFORMANCE §7 closes it on three grounds; status of each against HEAD:
- *Dead packed form* — shade slots are dense over quads because lanes 4q..4q+3 are the ddx/ddy quad
  (`shfl.sync.bfly` over `tid & 28`); still true (cp_abuf_shade: "Four slots a quad", :4501-4511). Sizing from `total`
  therefore allocates `4·total ≈ 3.75× today` (P1 median total/quads 3.753). **Holds.**
- *`capacity/4` gate* — `quad_bound = MIN3(quad_capacity, capacity/4, CP_ABUF_MAX_SHADE_SLOTS/4)` at :8417-8431 refuses
  25.3% of old-capture episodes. **Holds** (code unchanged).
- *Fail-closed checks* — `running != quads` (:8964), `fill_over‖quad_over` (:8903), and `cp_pass_can_retry` (:7668: a
  retry after any hardware-texture FS attempt latches device loss, because the FS may already have been launched against
  incomplete lists). **Hold**, and they are the real blocker for every "issue before you know" scheme: today the episode
  never shades before the answer, so the fallback is always legal.
- Value on favorite3: S1d moves the wait to the scan; the host still waits. PERF §7 P4 measured the movable share at 17.6%
  of the drain on old; 17.6% × 0.558 = **0.098 ms/frame** here (ESTIMATED), below the gate.
- Overflow never fires on favorite3: 0 "overflowed"/"falls back" lines in /tmp/postdead/stderr,
  /tmp/favorite3-perf-current/stderr, /tmp/postdead-stats.err (MEASURED); the A-buffer grew to 5,242,880 fragments
  (60 MB, 65 MB of quads) once and stayed.

**(ii) Device-evaluated conditional launch.** What is actually needed is not a *conditional* — the success path always
runs — but device-side *sizing*: base pointers and extents for `grouped`, `quad_dense`, `seg_cursor`, and the 6–9
per-group shade arrays, plus the per-group grid extents. Concrete design:
1. A `cp_abuf_plan` kernel (1 block) after `abuf_seg_count` reads `bsum3` (quads) and `seg_counts`, computes group
   bases (the grouping itself is host-known — it depends only on shader state, :8925-8945), carves the arrays from a
   **device-owned bump arena** (a second `dscratch` whose cursor lives in device memory, reset by `cuMemsetD32Async` at
   the same episode boundaries the host resets `dscratch.used`), and **patches the already-uploaded argument blocks**
   (`cp_fs_interp_args` is read from device memory via `interp_dev = cp_upload(&interp)` :4599 — the pointer fields
   `pixel_list/fs_in/fs_out/coverage/frag_coord/discard_mask/counter` can be overwritten in place; the FS argblock
   likewise). Stream order guarantees the patch lands after the upload and before the consumers.
2. Grids: `seg_scatter`, interp and FS launched at a bound with device-count early exit (the kernels already read
   `num_quads_dev` :4592, :4610). Idle blocks cost <0.2 ns (DEAD_ENDS 9) but a bound of `quad_capacity` (~5M quads →
   ~80k blocks) is ~16 µs per launch; use the previous episode's `quads × 1.5` published by the device into a
   host-mapped word (read with no sync, stale by one episode) and a device "grid too small" flag.
3. Fail-closed on the device: arena exhaustion, grid too small, `fill_over`, `quad_over`, `running != quads` all set one
   device flag; consumers early-exit; the host reads the flag at the *next* existing sync or at frame end and re-executes
   the scope with `force_pass_fallback` semantics (:8724). Blocker: the framebuffer must be restorable to scope start
   (LOAD_OP_LOAD scopes need a snapshot copy — ~7 MB per scope, ~10 scopes/frame on a side stream) and
   `cp_pass_can_retry`'s FS-attempt gate must be re-derived for the hardware-texture path.
4. Conditional graph nodes (CUDA 12.4+, present in the installed 12.8 headers: `cuGraphConditionalHandleCreate`,
   `CU_GRAPH_NODE_TYPE_CONDITIONAL`, 7 hits in /usr/local/cuda/include/cuda.h) and `cuStreamWaitValue32` (v2 present) are
   the two device-evaluated gates never measured here. DEAD_ENDS 25 priced CDP2 tail launch at 7.3 µs and did *not*
   measure either. What to measure, with `docs/cudavk/notes/cdp2-microbench.cu` as the template: (a) a stream of N
   kernels each preceded by `cuStreamWaitValue32(flag, EQ)` where a kernel sets the flag — per-link latency vs the
   1.48–1.51 µs host-issued floor; (b) a graph with an IF conditional node whose body is one kernel, launched N times with
   the device flipping the predicate — per-launch cost and the graph's instantiation cost (DEAD_ENDS 1: 14.8 µs
   instantiate, 44 KiB retained; zero recipe reuse across 4,532 submits is the standing objection); (c) `cuStreamBatchMemOp`
   write+wait pairs as a device-side "join" replacing `cuEventRecord`+`cuStreamWaitEvent` (8 pairs per episode today,
   ~0.2–0.3 µs API each). Success bar: any of these must land under the ~2 µs exposed-launch price to be worth a design.
5. Ceiling for the whole S1 programme: bubble **0.093** + single-stream tail overlap **≤ 0.207** (only if the next
   episode's raster can run across the drain; DEAD_ENDS 27 says scopes form a chain, but episodes inside a scope are
   independent in their raster phase, HYPOTHESIS) = **≤ 0.30 ms/frame** (ESTIMATED), against a large build.

**(iii) Both paths with predication.** The fallback path is `cp_pass_fallback` → `cp_draw_execute_batch` per segment
(:7686-7702), ~70 launches per episode (PLAN_STATS: 70.7 launches/episode). Pre-issuing it predicated for 5.44
episodes/frame adds ~380 exposed launches × 1.97 µs ≈ **+0.75 ms/frame** (ESTIMATED) to save 0.093. Refuted by
arithmetic, the same way DEAD_ENDS 25 candidate 2 was. The only viable form is (ii)-3: fail-closed on the device and
repair lazily on the host, because the fallback is never taken on this capture.

### 4.2 Segment counters (S2)

Read for `abuf_prod` (skip the peel loop) and `quads/covered` (array and grid sizes). Same device-side design as 4.1 (ii)
with one extra host branch: `passes = 0` when `abuf_prod` (:6878). A predicated form would issue the composite path and
the first peel pass both, each early-exiting on the device predicate — the peel pass is ~6 launches (visbuf clear,
stage1-3, shade, peel_advance) ≈ 12 µs exposed per draw, and `abuf_prod` is true on essentially every draw here (no
fallback lines in the logs), so the speculative pass is wasted every time: net ≈ −12 µs + 17 µs bubble ≈ +5 µs/draw,
0.02 ms/frame. Not worth building. The `bounded` fast path that would avoid this sync at zero cost admits no draw (PERF §6
item 2, P3: 0 of 8,836 samples). The collectible part of S2 is its **0.128 ms/frame of >100 µs host stalls** (§2.1
reading 5), which is a host-planning question, not a device one.

### 4.3 Peel checks (S3)

The device already tells the host to stop through a managed word; the sync exists only so the read is ordered. The
device-side alternative is speculative pass N+1 predicated on `peel_any` — 98.6% of loops run exactly one pass (PERF §7),
so the speculative pass is wasted 98.6% of the time at ~12 µs exposed against a 15 µs bubble. Net ≈ 0; the built form
(S2a/b/c predication) lost 0.11 ms (PERF §7). **Refuted.** Note the tension with the record: the spin probe measured slope
−0.03 at this site on the old capture ("adding host time is free"), while this trace shows the device empty for a median
15.1 µs after each of 2.48 checks/frame (0.094 ms/frame, 0.047 of it in 30 stalls >100 µs). Both can be true if the
frame's critical path at those points runs through the host (the host is then late for the next draw anyway); it is
worth one re-measurement with `CUDAVK_WAIT_SPIN_US` on favorite3 before anyone quotes either.

### 4.4 The >1 GiB scratch drain (S4)

What it protects: `dscratch` is a bump allocator whose cursor rewinds to 0 at every non-appending draw (:5266) but not
inside an episode; when `used > 1 GiB` or 5 overflow arenas exist, `cp_scratch_begin` drains the whole context and frees
the overflow arenas (:676-688, :736-748). The drain is needed only because `cuMemFree` of an overflow arena must not race
with kernels still reading it, and because the managed upload ring is rewound by the same reset (:751-767).
Stream-ordered replacement: record an event on every stream after the last consumer of an arena and free it from the
submit worker when the event completes (or `cuMemFreeAsync` on the stream); rewind the bump pointer under stream order
as the per-draw rewind already does. DEAD_ENDS 29 built the equivalent ("reuse the high-water instead of draining") and
measured **+0.037/+0.012 ms** with sign disagreement between pairs. Why below noise, from this trace: the sync's bubble is
8.4 µs (×1.65 = 0.014 ms/frame ≈ the measured 0.037 within noise), and its 1.07 ms median spin covers the main stream's
own tail (side streams quiet 479 µs earlier), so the freed host immediately hits the next gate (S1/S2 of the same or next
draw). Does that change when other syncs go? Yes, but only in combination: with S1–S3 also gone the host would run to the
frame fence, and S4's contribution becomes its bubble plus the concurrency the freed host can create — still bounded by
§2.1 (0.017 + share of 0.207). Retry it only as part of removing all in-frame gates, never alone (DEAD_ENDS 29 retry-if).

### 4.5 Descriptor upload (S6) and the destroy drains (S9)

S6 is a 9.6 µs synchronous HtoD at submit entry where the device is idle by construction; an async copy on the stream
would save ≤ 0.01 ms/frame. S9 drains are 0.3–0.5 µs null drains (DEAD_ENDS 17). Neither is a lead.

## 5. The frame boundary: the sync the driver does not own, and the one that dominates

MEASURED per frame in the window (bubbles2.py, exact: 412 frames with the 3,686,400-byte readback copy identified):

| segment | median µs | mean µs | ms/frame (sum/412) | what happens |
|---|---:|---:|---:|---|
| last kernel end → even submit ts | 65.6 | 129.7 | 0.130 | frame-end host func (S8) 77 µs, `cuEventRecord(done)`, `vkQueueSubmit` returns |
| even submit → readback copy start | 9.1 | 10.7 | 0.011 | app's `vkCmdCopyImageToBuffer` submit A (`cuMemcpy2DAsync`) |
| **readback copy** (device→**managed** memory, `srcKind=2,dstKind=4`, kind 8) | **582** | 617 | **0.617** | 3.69 MB at 6 GB/s because destination pages migrate (UVM11 rows follow); worker `cuEventSynchronize` waits on it |
| fence wake (worker sync end − copy end) | 2.3 | — | — | negligible |
| **copy end → odd submit ts** | **1120** | 1454 | **1.454** | pure host: app reads the buffer (UVM12 page faults, 69/frame), records ~2,100 vkCmd* calls, `vkQueueSubmit` B |
| **odd submit → first kernel** | **200.7** | 284 | **0.284** | driver translation: 38 `cuMemcpy2DAsync` (133 µs API), 7.6 `cuMemcpyDtoDAsync` (40), 47 `cuEventRecord` (14), 1 sync `cuMemcpyHtoD` (9.6), 4.4 HtoDAsync, 4 memsets, then the first launch; 277 µs of API time per submit |
| **total, last kernel → first kernel** | **2029** | 2496 | **2.496** | 33.6% of the traced 7.43 ms frame |

The frame boundary is the largest idle in the frame by a factor of ~7 over all in-frame sync bubbles together. It is a
host sync site (S7): the app submits the readback copy, **waits for it**, reads the image, records the next frame,
submits. Since this trace, history/perf-2026-08-31-host-pin-readback.md pinned the readback destination host-resident
(`cuMemAdvise` at `vkBindBufferMemory`, cpvk_device_memory.c:1116-1133) and measured **−0.929 ms/frame on favorite3 RTX**
(7.6048 → 6.6757). ESTIMATED residual boundary today: 2.496 − 0.929 ≈ **1.57 ms/frame mean** (≈1.1 median), composed of
~0.15 ms copy over PCIe (3.69 MB; B200 measured 0.323 ms/frame post-pin), ~0.9–1.0 ms of app CPU (read + record; the
perf profile's "application record/decode 0.376" plus the driver's record-side share of "planning 0.33" plus 0.37
unattributed — RESIDUAL_AUDITS), and **0.28 ms of driver submit→first-kernel latency**. Cheapest probe: one fresh
nsys trace at HEAD and rerun `bubbles2.py` (the boundary decomposition is 40 lines of SQL/Python; the window logic is
already validated against the docs' 412.5 frames and 4.51 ms union).

What the driver owns here:
1. **Submit → first kernel, 0.284 ms/frame mean** (MEASURED, CUPTI-inflated). It is the app's per-frame staging copies
   (`vkCmdCopyBuffer` → 38 `cuMemcpy2DAsync` + 7.6 `cuMemcpyDtoDAsync` of 2–116 KB, each ~3 µs of API for ~0.6 µs of
   device time) issued one at a time onto an empty device, plus 47 `cuEventRecord` (texture-cache "image written"
   events). Alternative: gather the copies recorded before the first draw of a submit into one copy kernel over a
   descriptor table (or `cuMemcpyBatchAsync`, new in CUDA 12.8) → 45 API calls become 1–2. ESTIMATED saving 45 × ~1.5–2 µs
   (uninflated) ≈ **0.07–0.09 ms/frame**, ceiling 0.28. Closest DEAD_ENDS: 8 (upload flush placement — the ring, not
   these copies) and 22 (merge rule — inapplicable: these ops are strictly serial on an empty device, so the removal
   credit is the full exposed price, which is the merge rule's own condition for collectability).
2. **The frame-end host func** (S8, one of the three) sits between the last kernel and the readback copy: 77 µs × 1 =
   0.077 ms/frame of the 0.130 ms "last kernel → even submit" segment; covered by §3.
3. The record-side CPU cost of `vkCmd*` in cpvk_cmd.c is inside the 1.454 ms host segment and is 1:1 frame time
   (device empty by construction). Its size is not separable in the existing perf artefact without re-running perf with
   the record path as the anchor; the docs' own split gives app 0.376 + driver planning 0.33 steady + 0.37 unattributed.
   HYPOTHESIS: a third to a half of the host segment is driver record/translate work.

What only the application owns: the copy → wait → record → submit serialization. Waiting for frame N−1's readback
*after* recording frame N (double-buffered readback) would overlap ~1.1 ms of host time with device work every frame.
APP_IMPROVEMENTS.md item 4 ("keep the current sync shape") should be amended with this measurement; the harness is a
fixed replay, so this is a recommendation to the app owner, not a driver lead.

## 6. Deliverable table (d)

| site | count/frame (window) | blocked ms/f | post-sync bubble ms/f | join cost | device-side alternative | expected saving, derivation | blockers | closest DEAD_ENDS |
|---|---:|---:|---:|---|---|---|---|---|
| S1 episode drain | 5.44 | 0.558 | **0.093** (14.6 µs med) | 0.207 ms/f single-stream tail (38 µs/drain) | device-side sizing kernel + argblock patching + device-owned bump arena + fail-closed flag with lazy host repair (§4.1 ii) | bubble 0.093 + tail overlap ≤ 0.207 = **≤ 0.30** (ESTIMATED; overlap needs the next episode's raster issued across the drain) | memory bound (S1b 250 MB–1.6 GB) unless a device bump arena; `capacity/4` gate; `running != quads`; `cp_pass_can_retry` FS gate; scope re-execution needs framebuffer snapshots for LOAD_OP_LOAD; unmeasured `cuStreamWaitValue32`/conditional-node latency | 11 (worst-case sizing), 25 (CDP2/predicated/conditional), 29; PERF §6 item 4 run-ahead (0.387 ceiling old, 0.092 collected) |
| S2 segment counters | 3.48 | 0.460 | **0.181** (0.128 in >100 µs host stalls) | none (single-stream path) | same as S1; predicated first peel pass refuted by arithmetic (+5 µs/draw) | the 0.128 host-stall share is 1:1 at this site's +1.03 slope → **≤ 0.13** once attributed | unknown host work after the copy (needs a perf anchor) | PERF §6 item 2 (bounded path admits nothing), §5.2b |
| S3 peel checks | 2.48 | 0.253 | 0.094 (0.047 in 30 stalls) | none | speculative predicated pass N+1 | ≈ 0: 98.6% of loops run one pass; 12 µs exposed vs 15 µs bubble | slope −0.03 measured on old contradicts a 15 µs bubble here — re-measure before quoting | PERF §7 peel closure; S2a/b/c predication −0.11 |
| S4/S5 scratch reclaim + upload wrap | 1.65 | **1.593** | 0.017 (8.4 µs) | none (waits on the main tail) | event-ordered arena free / `cuMemFreeAsync`, stream-ordered rewind | **≤ 0.02 alone** (matches DEAD_ENDS 29's +0.037); only as part of removing all in-frame gates | none technical; value | 29 |
| S6 descriptor upload | 1.006 | 0.0096 | n/a (device idle) | — | async HtoD on the stream | ≤ 0.01 | — | 8 |
| S8 host-func (timestamps) | 3.00 | 0 (device waits) | **0.263** (77.5 µs med) | — | device timestamp kernel (`%globaltimer`) + availability by done-event; results via one DtoH at `vkGetQueryPoolResults` | **≤ 0.26** (MEASURED bubble; conversion HYPOTHESIS 1:1 on a single-stream chain; verify with the untraced busy sampler) | GPU-clock semantics / `timestampPeriod`; exact availability (app sleeps 5 ms on NOT_READY) | 17 (verify the device has queued work behind it — it does) |
| S7 frame fence (app) | 2.0 | ≈ copy time | **2.50 ms/f pre-pin → ≈1.57 post-pin (ESTIMATED)** | — | driver: batch the frame-start copies (0.28 → ~0.2); remove the frame-end host func; app: double-buffer the readback wait | driver share **0.07–0.09** (copies) + 0.08 (host func); app share ~1.1 | app structure is fixed in the replay | 34 (threaded submit 0.126), 8; the pin record (−0.93) |
| S9 destroy drains | 0.39 | 0.003 | 0 | — | — | 0 | — | 17 |
| S10 indirect / DeviceWaitIdle | 0 | 0 | 0 | — | — | 0 | — | 19 |

## 7. Findings, ranked for the design agents

1. In-frame host syncs cost **0.385 ms/frame** of device idle in total (MEASURED, traced window); their 2.86 ms/frame of
   spin is device-paced and not collectible (DEAD_ENDS 29 confirms the mechanism). No single site reaches the 0.5 ms gate.
2. The frame boundary is **2.50 ms/frame** of device idle in this trace, 33.6% of the frame; the readback pin took
   0.93 of it; ≈1.57 remains (ESTIMATED), of which the driver owns 0.28 (submit→first kernel) + 0.08 (frame-end host
   func) + an unmeasured record-side share. This is where TODO item 6's "2.453 ms inter-submit stall" lives.
3. `cuLaunchHostFunc` timestamp queries stall the stream 3× per frame for 77 µs median = **0.263 ms/frame**, a
   device-waits-on-host round trip absent from every budget in the docs.
4. Removing the drain's round trip is worth at most **~0.30 ms/frame** and only with the next episode's raster issued
   across it; the conditional-node and `cuStreamWaitValue32` prices are the two unmeasured CUDA facts standing between
   this and a design, and the microbenchmark to take them is a 60-line variant of `cdp2-microbench.cu`.
5. The dependent-launch floor (1.146 ms/frame over 561 gaps, median 1.3 µs) is the second-largest idle class and is the
   population the merge rule already closed.

## 8. Unknowns

- Post-pin frame-boundary decomposition at HEAD (needs one fresh trace; estimated above by subtraction).
- What the host does in the 258 long S2 bubbles and 30 long S3 bubbles (0.175 ms/frame).
- The uninflated host-func dispatch latency (CUPTI adds callback-thread instrumentation).
- Per-link latency of `cuStreamWaitValue32` gating and of conditional graph nodes on sm_120/sm_100 (never measured).
- Whether the next episode's raster phase is independent of the current episode's tail on favorite3 (decides the 0.207).
- The record-side (cpvk_cmd.c) share of the boundary's host segment.
- B200: none of the trace numbers here are B200; the pin record shows the copy at 0.323 ms/frame there post-pin.
