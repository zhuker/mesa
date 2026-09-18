# Design: Autonomous Segments — device-decided episodes under exec-updated chain graphs

Angle: device autonomy and recurrence. Designer report, read-only, 2026-09-02. Nothing built, nothing run on a GPU,
nothing in the repo modified. Scratch scripts and their output are in this directory (`seg_census.py`,
`topo_census.py`). Labels: **MEASURED** (document or trace, source named), **MODEL** (the closure-audit agent's
dependency-respecting replay of a trace, validated E0 p50 15 us but still a model), **ESTIMATED** (my arithmetic on
measured inputs, shown), **HYPOTHESIS** (no measurement).

The one-paragraph verdict: the driver blocks the host 9.6 times per heavy-band frame to read five counters and
decide array sizes, grid sizes, one overflow branch and one loop exit. Every one of those decisions can be made on
the device from the same counters, and the only branch that cannot (A-buffer overflow) has fired 0 times in 17
logged replays of both captures. Moving the decisions onto the device is the mechanism; CUDA graphs are a second,
conditional layer whose value depends on two unmeasured prices. The plan reaches 5.0 ms on the whole-window RTX
median only at the top of its bracket, and not on the heavy band; it must be combined with the geometry run-ahead
lead to get there with margin. The first slice is an oracle replay that measures the whole prize in a week without
building the mechanism.

---

## 0. What the evidence says the frame is, in the terms this design needs

### 0.1 Decision points per frame (MEASURED, my census on `/tmp/postpin-rtx2/trace.sqlite`, heavy band, 1,025 frames)

Script `seg_census.py`; frame = two worker `cuEventSynchronize` returns; main thread = the globalTid with the most
RUNTIME rows (308125468474637), worker 308125468474653.

| main-thread call | per frame | CUPTI host ms/frame | role |
|---|---:|---:|---|
| `cuStreamSynchronize` | 8.23 | 1.034 | episode drain, segment counters, peel checks (`cp_sync_timed`, cp_renderer.c:713) |
| `cuCtxSynchronize` | 1.41 | 2.841 | scratch reclaim `cp_scratch_begin` (:666-690), upload wrap (:414-427) |
| `cuMemcpyDtoH` (sync) | 6.57 | 0.051 | the counters read after the stream syncs (:8887, :6808) |
| `cuMemcpyHtoD` (sync) | 0.99 | 0.011 | descriptor arena at submit entry |
| **blocking total** | **17.2** | **3.94** | matches closure-audit 3.937 |
| `cuLaunchKernel` / `cuLaunchKernelEx` | 425.2 / 295.2 | 0.831 / 0.557 | 720 launches/frame |
| `cuMemcpyHtoDAsync` | 261.2 | 0.450 | argument-block flushes |
| `cuEventRecord` / `cuStreamWaitEvent` | 216.5 / 195.0 | 0.051 / 0.058 | fan-out gates and joins |
| `cuMemsetD32Async` / `D8Async` | 102.9 / 11.3 | 0.202 / 0.025 | counter and visbuf clears |
| `cuMemcpy2DAsync` / `3DAsync` | 23.0 / 6.3 | 0.069 / 0.031 | app image copies (frame start) |
| `cuLaunchHostFunc` | 2.97 | 0.006 | timestamp queries (cpvk_cmd.c:4842) |
| **issue-class total** | ~1,560 | **2.282** | CUPTI-inflated 28-91% (launch-path audit) |

Uninflated host issue, ESTIMATED from the launch-path audit's no-CUPTI prices × these counts:
425.2×1.70 + 295.2×1.36 + 261.2×1.30 + 114.2×1.07 + 216.5×0.09 + 195×~0.10 + 29.3×~2.0
= 0.723 + 0.401 + 0.339 + 0.122 + 0.019 + 0.020 + 0.059 = **~1.68 ms/frame**.

Whole-replay means from `CUDAVK_PLAN_STATS` (post-dead-scope profile, PERF_HANDOFF): drain 3.74/frame,
segment counters 1.47, peel 1.13, descriptor 1.00 — the heavy band has ~2× the drains.

### 0.2 Recurrence at the "frame chunk between sync points" unit (MEASURED, `topo_census.py`, 1,012 frames)

Signature = ordered op types between consecutive main-thread blocking calls (`K:<kernel name>`, `S` memset,
`M1` 1-D HtoD, `M2` 2-D/3-D copy, `MD` DtoD, `E` event record, `W` stream wait, `H` host func). Frame index by op end
time against the worker's completion returns.

| quantity | value |
|---|---:|
| segments / non-empty segments per frame | 17.20 / 10.54 |
| ops per non-empty segment p10 / p50 / p90 / max | 21 / 58 / 361 / 3,184 (the max is the recording segment: 1,554 `cuCtxPush/Pop` pairs, 4 launches) |
| distinct signatures over 10,664 non-empty segments | **523** |
| segment signature seen in the previous frame | **90.7%** |
| same-position match against the previous frame | **87.6%** |
| top-20 / top-50 signatures cover | 75.4% / 86.9% of segments |
| **device ops+events lying in signatures seen ≥2 times** | **62.6%** of 1,654 ops/frame |
| segments carrying 2-D/3-D copies (not `cuGraphExecUpdate`-able, cuda.h:21111 "must be 1-dimensional") | 16.0% (29.3 copies/frame, mostly the frame-start segment) |
| segments carrying `cuLaunchHostFunc` | 18.8% (2.97/frame) |

This reproduces the closure-audit census (411 name-only signatures, 90.4% same-position) with a stricter signature.
Two readings: topology recurs almost perfectly at the segment unit, **but** the long unique segments (p90 361 ops)
hold 37% of the ops, and — decisive for the ordering of this plan — **the unit disappears when the syncs do**: with
no in-frame blocking calls a submit is one ~1,650-op segment, and frame-level Vulkan structure never repeats
(closure audit: 1,426/1,426 distinct heavy-band frames). So graphs at the inter-sync unit are the right unit only if
decision removal fails; if it succeeds the graph unit must be the episode/batch chain (§3.3).

In the one long recurring signature I checked (72 ops × 634 instances: seg_scatter → main → main → composite +
17 2-D copies + 27 event records), grid sizes, memset sizes and HtoD sizes did not vary across instances; that is the
frame-start segment and is not representative of the shading segments, whose grids follow quad counts.

### 0.3 The prizes, with the disagreement between two evidence agents stated

| quantity | value | kind | source |
|---|---:|---|---|
| post-sync device bubbles, all in-frame syncs, light band | **0.385 ms/frame** (drain 0.093, seg 0.181, peel 0.094, ctxsync 0.017) | MEASURED (CUPTI) | host-sync report §2.1, `/tmp/postdead` |
| drain single-stream tail (main alone after side streams quiet) | 0.207 ms/frame | MEASURED | host-sync §2.1 |
| S2 post-sync host stalls >100 us | 0.128 ms/frame | MEASURED | host-sync §2.1 |
| host-func (timestamp) stream stalls | **0.263 ms/frame** (77.5 us median × 3.0) | MEASURED (CUPTI callback) | host-sync §3 |
| "free syncs" (host never blocks, device deps kept) | **-1.372 heavy → 6.783; -1.649 light → 5.518** | MODEL | closure audit §2.3 |
| "host infinitely fast" | -1.348 heavy → 6.807 | MODEL | same |
| "free syncs + host ×0.5" | -1.372 (unchanged) | MODEL | same — **the sync pool and the host-issue pool are the same 1.35 ms** |
| "launch gaps zero / half" | -1.607 / -0.822 heavy | MODEL | same |
| "free syncs + launch gaps zero" | -3.171 → 4.984 heavy | MODEL | same |
| geometry run-ahead (VS/fetch/clip start at issue) | -1.801 heavy | MODEL | same, lead 4.1 (not this angle; composes) |
| one site removed alone (scratch ctxsync, 1.65 ms blocked) | **+0.037** | MEASURED | DEAD_ENDS 29 |

The host-sync analyst concludes "no host-sync mechanism reaches 0.5 ms alone" from the bubble sum; the closure
auditor concludes the sync structure is worth 1.37-1.65 **as a set** and that per-site probes cannot see it (entry 29
is the measured one-site case). Both are consistent with each other and with the record if the prize is all-or-nothing:
the bubble sum (0.385) is the floor, the model (1.37-1.65, with CUPTI-inflated host timing, so the untraced pool is
smaller) is the ceiling, and the extra comes from what a never-blocked host can overlap — its own planning between
calls (~0.33 ms/frame steady, RESIDUAL_AUDITS), the S2 host stalls (0.128), the drain's single-stream tail (0.207),
and the issue ramp after every sync. **The bracket is 0.39-1.65 ms/frame, and the first slice (§6) measures where in
it the truth sits, for about a week of work and no production code.**

---

## 1. Thesis

Make the device decide and the host merely narrate. Every in-frame host block in cudavk is a read-back-and-decide
whose inputs are already in device memory and whose outputs are pointers, extents, grid sizes and one overflow branch
that never fires on these captures. Compute them on the device in one tiny kernel per episode, patch them into the
argument blocks the host already uploaded, launch consumers at fixed grids that stride over device counts, fail closed
into a device flag, and repair lazily from the host once per submit. Then the host issues a whole submit without
blocking, the device queue is never empty inside a submit, and the remaining host↔device traffic is one flag read per
submit plus the application's fence. Graphs come second and only where a microbenchmark shows the device scheduler
dispatches a dependent chain faster than the stream front end does.

---

## 2. Mechanism, at the level of kernels, launches and syncs

Three layers, ordered by the certainty of their prize. B first, C second, A last and conditional.

### 2.1 Layer B — device-decided episodes (removes S1, S2, S3, S4, S6)

**B1. Episode drain → `cp_abuf_plan`.** Today `cp_pass_finish` (cp_renderer.c:8691-8990) after `abuf_seg_count`
does `cp_sync_timed(cp->stream)` + `cuMemcpyDtoH(ctr, ab->counters, 4×(6+nsegs))` (:8884-8890), then on the host:
`fill_over‖quad_over → cp_pass_fallback` (:8903); `quads==0 → return` (:8910); shader-state grouping (:8925-8945,
depends on nothing read back); `seg_base`/`group_base` prefix sums over the per-segment counts (:8952-8963);
`running != quads → fallback` (:8964); `cp_scratch_alloc_device` for `grouped`, `quad_dense` (quads×4 B each) and
`seg_cursor`; `cp_upload(seg_base)`; `cuMemsetD32Async(seg_cursor)`; `abuf_seg_scatter` at `(quads+255)/256`; then
per group `cp_abuf_shade` (:4480-4640): `num_slots = ALIGN(gq×4, 256)` sizes 6-9 `dscratch` arrays, a counter memset
or `abuf_prepare_shade_count`, `cuMemsetD8Async(discard_mask)`, `interp_dev = cp_upload(&interp)`, interp launch,
FS launch; then the composite sized by `covered`.

New sequence, no host wait:
1. Host computes the grouping table (unchanged, host-known) and the per-episode layout *recipe*: for each group the
   strides (`fs_in_stride`, `fs_out_stride`) and which optional arrays exist (`discard_mask`, `front_face`,
   `batch_rows`). It uploads the interp/FS argument blocks exactly as today, with the pointer and extent fields left
   as placeholders, and records their device addresses.
2. Host launches **`cp_abuf_plan`** (1 block, ≤64 threads = `CP_PASS_MAX_SEGS`) on `cp->stream` after
   `abuf_seg_count`. It reads `ab->counters[0..5]` and `seg_counts[0..nsegs)`, computes `seg_base`/`group_base`/
   `group_quads` in the same group-major order, checks `fill_over ‖ quad_over ‖ running≠quads`, carves `grouped`,
   `quad_dense`, `seg_cursor` and every group's shade arrays from a **device-owned bump arena** (a `cuMemAlloc`
   region whose cursor is a device word; the plan kernel bumps it; exhaustion is a failure), zeroes `seg_cursor` and
   the counter words itself, and **writes the pointers and extents into the already-uploaded argument blocks** at
   fixed offsets (`cp_fs_interp_args.pixel_list/fs_in/fs_out/coverage/frag_coord/discard_mask/front_face/
   batch_rows/counter/max_pixels`, the FS block's corresponding slots, `cp_abuf_seg_args.grouped/quad_dense/
   seg_cursor/seg_base/num_quads`). Stream order guarantees the upload has landed: `cp_launch_after` flushes the owed
   span before every launch (:531-547), on the same stream. On failure it writes `episode_fail = 1` and publishes zero
   extents.
3. `abuf_seg_scatter`, the per-group interp and FS, and the composite are launched at **fixed grids with grid-stride
   loops over the device counts** (`num_quads_dev` is already read by interp/FS, :4592/:4610; seg_scatter and
   composite gain a stride loop) and exit on `episode_fail` at block start. Idle blocks cost <0.2 ns (PERFORMANCE §4,
   rule 6); a 4,096-block grid shell is ≤0.82 us (PERFORMANCE §7 grid tuning).
4. The `discard_mask` clear (today a host-sized `cuMemsetD8Async` over `num_slots`) is written by the interp kernel
   for every slot it produces (it already writes `coverage` per slot), so no bound-sized clear exists.
5. `total → grow_to` bookkeeping (:8897-8900) moves to the per-submit flag read (B5).

Blended episodes shade on the side streams (kernel-chain: "side-stream FS"), so the plan kernel must run on `cp->stream`
**before** `cp_pass_broadcast` gates the side streams (:7629-7660) — then the side-stream consumers see the patched
blocks by the broadcast's event.

Host calls removed per drain: 1 `cuStreamSynchronize`, 1 `cuMemcpyDtoH`, ~3 `cp_scratch_alloc_device` (host-side
bookkeeping), 1 `cp_upload(seg_base)`, 1 `cuMemsetD32Async`. Added: 1 launch (`cp_abuf_plan`). Per frame heavy band:
−5.4 syncs, −5.4 DtoH, +5.4 launches (ESTIMATED from the whole-replay ratio of drains to blocking calls).

**B2. Segment counters (single blended draw, composite path, :6796-6826)** read `total, fill_over, quads, quad_over,
covered` to set `abuf_prod` (skip the peel loop) and sizes. Same plan kernel with `nsegs = 1`: sizes the shade arrays
and composite; `abuf_prod == false` (a broken merge) becomes `episode_fail`. `abuf_prod` is true on every draw of
both captures: 0 "falls back to the peel loop" lines in 17 logs (`/tmp/postdead/stderr`, `/tmp/pin-rtx-ab/*/stderr`,
`/tmp/postpin-rtx*/stderr`; MEASURED). −3.5 syncs/frame heavy band.

**B3. Peel checks (:7076-7090) and the discard-retry loop (:6111, :6179).** The peel loop already stops on a
device word (`cp->peel_any`, written by `peel_advance`); the host sync exists only to read it. The retry loop runs
`CP_DISCARD_LAYERS = 8` passes with no check at all (kernel-chain L3: 7.0 re-passes/frame, 221 us/frame, FS at its
floor in the shown passes). Two forms, chosen by microbenchmark M3:
- **Conditional WHILE graph node** (cuda.h:1802, CUDA 12.8 present): body = one pass (visbuf re-clear, stage 1-3,
  compact/shade, `peel_advance` or writeback), condition set on the device by `cudaGraphSetConditional(handle,
  peel_any)` from `peel_advance` (for discard: a per-pass "any discarded" counter written by `cp_fs_writeback`, which
  already handles `discard_mask`). Toolchain prerequisite (EPISODES.md session 11): the setter needs `libcudadevrt`
  (present: `/usr/local/cuda-12.8/lib64/libcudadevrt.a`); the `.cu` kernels are NVRTC → PTX → `cuModuleLoadData`
  (cp_kernels.c:368) and would go through `cuLinkCreate` + `cuLinkAddData(PTX)` + `cuLinkAddFile(libcudadevrt.a)`,
  which the shader path already does for the sampler (cp_nir_to_llvm.c:4223-4244). The body's `pass` argument
  (`cp_shade_fragments(..., pass, ...)`) must come from a device pass counter instead of a host constant.
  Measured cost: 10.27 us per conditional-graph launch vs 4.10 us for two direct launches (EPISODES.md session 11,
  MEASURED once, this GPU).
- **Predicated pre-issue** (fallback if M3 fails): issue passes 2..N with every kernel exiting at block start on
  `peel_any == 0` / `discarded == 0`. Costs the launch shells of the unneeded passes (~6-11 ops × ~2 us exposed).
  This is the shape that lost 0.11 ms as `CUDAVK_PEEL_PREDICATE` (PERFORMANCE §7) — acceptable only for the discard
  loop, where today's 7 wasted full passes cost more than 7 wasted shells.
Prize: S3 bubble 0.094 + kernel-chain L3 0.13-0.19 (if `CUDAVK_DEBUG_DISCARD` shows passes ≥3 discard nothing) =
0.22-0.28 ms/frame light band (MEASURED components, HYPOTHESIS that the loop converges early).

**B4. Scratch reclaim `cuCtxSynchronize` (S4, :666-690; 1.41/frame heavy, 2.84 ms/frame blocked CUPTI).** Take the
DEAD_ENDS 29 worktree mechanism (`~/mesa-scratch2`: reuse the high-water instead of draining) plus stream-ordered
freeing of overflow arenas (event recorded on every stream after the last consumer, freed from the completion worker).
Entry 29 measured this alone at +0.037 and its retry clause reads "after an architectural change removes the next
counter dependency" — B1-B3 are that change. Never ship it alone.

**B5. One flag read per submit, and lazy repair.** Before any op that observes rendering across a submit boundary —
the app's `vkCmdCopyImageToBuffer` in submit B, `vkGetQueryPoolResults`, `vkQueueSubmit` return path for the fence —
the driver reads `episode_fail` (one 4-byte `cuMemcpyDtoH` after the submit's last kernel; the app's fence drains the
device immediately after, so this round trip costs ~15 us once per frame, MEASURED drain bubble p50 14.6 us). If set:
grow whatever overflowed (the existing `grow_to` path) and **re-execute the command buffer's ops from the first failed
scope to the submit end** through the existing classic path (`force_pass_fallback` semantics, :8724). Re-execution
needs the failed scope's attachment contents at scope start, so every speculative scope whose attachments are
`LOAD_OP_LOAD` snapshots them at scope begin with a `cuMemcpyDtoDAsync` on a side stream (7.4-34.6 MB per target at
~1.5 TB/s = 5-23 us device each; ≤0.1-0.2 ms/frame if all ~10 scopes need it — HYPOTHESIS, the LOAD_OP census is
probe P2). Failure has never occurred on either capture after warm-up (0 "overflowed"/"falls back"/"refusing episode
retry" lines in 17 logs; the A-buffer grows once per run: `fragment array 5242880 entries` and three per-pixel sizes,
all before the real window), so the repair path is insurance whose cost is the snapshot copies, not the re-execution.

**B6. `cp_pass_can_retry` (:7668) is re-derived, not deleted.** Today a retry after any hardware-texture FS attempt
latches device loss because the FS may have run against incomplete lists. Under speculation the FS always runs before
the host knows the answer, so the invariant becomes: *the plan kernel sets `episode_fail` before any consumer runs
(stream order), and every consumer tests it at block start, so a failed episode's FS never touched the lists.* The
gate then admits a retry iff the flag was set by the plan kernel (a device-written "consumers exited" word), and the
deliberate-failure test flag (§5) exercises exactly this path every Nth episode with output identity as the check —
the same way `CUDAVK_FORCE_PASS_FALLBACK` makes today's fallback reachable.

### 2.2 Layer C — device timestamps (removes S8)

`cpvk_execute_query` (cpvk_cmd.c:4831-4846) issues `cuLaunchHostFunc` 3.0×/frame; each stalls the stream until a
runtime thread runs the callback: 77.5 us median device gap, **0.263 ms/frame** (MEASURED, host-sync §3; the callback
dispatch is CUPTI-instrumented so the untraced gap is smaller — bracket 0.13-0.26). Replace with a 1-block
`cp_timestamp` kernel writing `%globaltimer` and an availability word into a device query buffer;
`vkCmdResetQueryPool` → `cuMemsetD32Async`; `vkGetQueryPoolResults` → one DtoH after the app's own fence (device
idle by construction); `timestampPeriod` reported for the GPU clock (1 ns for `%globaltimer`). `WAIT_BIT` waits on the
submit's done event instead of `pool->changed`. Availability must be exact: the app sleeps 5 ms on `NOT_READY`
(host-sync §3). The order point (`cpvk_execute_order_point` → `cp_pass_finish`, cpvk_cmd.c:4253) stays for
semantics; after Layer B it no longer costs a host wait.

### 2.3 Layer A — chain graphs with `cuGraphExecUpdate` (conditional on M1/M2)

Unit: **the episode chain**, not the inter-sync segment (§0.2 explains why the segment unit dies with the syncs).
Templates, built once with the explicit node API and keyed by discrete shape only:
- blended episode tail: `scan_reduce → scan_finish → fill_recs → sort_short → sort → quad_count_all → scan_finish →
  quad_fill_all → seg_count → cp_abuf_plan → seg_scatter → {per group: interp → FS} → composite` (11-13 kernel nodes,
  2-4 memset nodes, 2-3 memcpy nodes) — one template per (ngroups, exec mode);
- direct batch chain: `[vertex_fetch] → VS → clip_rast_fused → stage2 → stage3 → fs_compact → FS → fs_writeback`
  (7-8 kernel nodes + 2-4 memsets + 2-3 HtoD copies) — one template per (fetch fused?, retry/peel loop?, PDL tier);
- the WHILE-node peel/discard body of B3 lives inside these templates.
Per instance: `cuGraphExecKernelNodeSetParams` (function may change: cuda.h:4262 ties `FUNCTION_CHANGED` to driver
< 11.2; this box runs 580.x), `cuGraphExecMemcpyNodeSetParams` (1-D only, cuda.h:21111), memset node update (1-D
dimension change "may fail if the resulting operation doesn't map onto the work resources already allocated",
cuda.h:21620-21626 — so the 7.4 MB and 34.6 MB visbuf clears need separate templates or classic issue), one
`cuGraphLaunch`. 2-D/3-D image copies (29/frame) stay classic, issued on the same stream ahead of the graph. The
fan-out becomes graph branches (fork/join edges replace 216 `cuEventRecord` + 195 `cuStreamWaitEvent` API calls per
frame). PDL links become `CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC` edges (cuda.h:1878-1885) — whether an exec with such
edges accepts updates is probe M5.

~75 graph launches per heavy frame (66 direct chains + 9 blended episodes; kernel-chain census) replace ~1,560 API
calls. Value has two axes, priced in §3.3.

---

## 3. Ceiling and projection

### 3.1 Prices used (MEASURED unless marked)

| price | value | source |
|---|---:|---|
| launch removed from a real chain | 0.78-0.81 us | PERFORMANCE §4 |
| exposed launch added | 1.974 us | PERFORMANCE §4 |
| host `cuLaunchKernel` / `Ex` / `cuMemcpyHtoDAsync` / memset / `cuEventRecord`, no CUPTI | 1.70 / 1.36 / 1.30 / 1.07 / 0.09 us | launch-path audit (perf, same frames as the trace) |
| device-paced link gap: PDL / plain / through a copy | 0.26 / 1.86 / 3.71 us p50 | launch-path audit |
| post-sync bubble: drain / seg / peel / ctxsync | 14.6 / 16.9 / 15.1 / 8.4 us median | host-sync §2.1 |
| graph launch (4-512 nodes) | ~1.0 us flat | CUDA13_UPGRADE.md §4 (this box) |
| stream recapture | 0.155 us/node | same |
| `cuGraphExecUpdate` | 0.39 us / 32 nodes | same — almost certainly measured with unchanged parameters; treat as a lower bound (HYPOTHESIS) |
| `cuGraphInstantiate` | 5.8 us @12 nodes, 12.6 @32, 48-62 @128, 251 @512 (0.38-0.5 us/node) | `/tmp/perf16/iter19-graph-microbench.txt`, `iter20-…txt` |
| conditional-graph launch (IF body 1 kernel) vs two direct launches | 10.27 vs 4.10 us | EPISODES.md session 11 |
| CDP2 tail launch | 7.32-7.44 us/link | DEAD_ENDS 25 (not used here) |
| `cuStreamWriteValue64 + WaitValue64` pair | 1,795 ns | TODO.md Tier 2 item 12 (not used here) |
| idle 256-thread block | <0.2 ns; 4,096-block empty grid ≤0.82 us | PERFORMANCE §4 rule 6, §7 |

### 3.2 Layer B ceiling (RTX)

Upper bound: the MODEL's "free syncs" −1.372 heavy / −1.649 light-window. Lower bound: the MEASURED bubble sum 0.385
(light band, 13 syncs/frame) scaled to the heavy band's 17.2 blocking calls ≈ 0.51, plus the 0.207 single-stream tail
that a never-blocked host can overlap, plus the 0.128 S2 host stalls = **~0.85 ms/frame heavy band if only the
directly measured components convert** (ESTIMATED). Costs to subtract (ESTIMATED): +5.4-9.6 `cp_abuf_plan` launches
at ≤1.974 us exposed = 0.01-0.02; fixed-grid shells on ~30 stride launches × ≤0.82 us = 0.02; snapshot insurance
0-0.2 (probe P2); one flag round trip per frame 0.015. Net **Layer B bracket 0.6-1.4 ms/frame heavy, central 0.9**;
whole-window median similar or larger per frame (the light window's model prize was 1.65 with 23 syncs/frame).
The CUPTI caveat cuts the top of the bracket: the traced light window is 7.43 ms against 5.78 untraced for the same
frames (b200 report, MEASURED, 1.28×), and CUPTI's per-call inflation lands exactly in the host-exposure pool the model
prices. The first slice measures the untraced value directly.

### 3.3 Layer A ceiling (RTX) — two axes, the host axis shared with Layer B

Host axis: today's uninflated issue ≈1.68 ms/frame (§0.1). If every op is a node updated at `u` us/node plus 75
launches at 1.0 us: host issue → 1,146×u + 0.075 ms. At u = 0.3 the saving is 1.26 ms of host time; at u = 0.8,
0.68; at u ≥ 1.4 nothing. Host time converts only through the 1.35 ms exposure pool (MODEL "host infinitely fast"
−1.348; "host ×0.5" −0.813), **and that pool is the same one Layer B empties** (MODEL: free syncs + host ×0.5 =
free syncs). So Layer A's host axis is worth 0.5-0.8 **instead of** Layer B, and ~0 **after** it.

Device axis: in-graph dependent dispatch versus the stream front end. Plain (non-PDL) links: 425/frame heavy, of which
~300 on the main stream (closure audit stream placement); p50 gap 1.86-2.1 us. If the device scheduler resolves a
graph edge in `g` us, the saving is ~300 × (1.9 − g) × 0.73 (serial fraction, closure audit) — **0.2-0.4 ms at
g = 0.5-1.0, 0 at g ≥ 1.9** (HYPOTHESIS; CUDA13_UPGRADE's "stream 4.10 → graph 2.02 → PDL 1.89 us/kernel" is the only
local datum and its methodology is undocumented). The MODEL's "launch gaps half" (−0.82) is the outer bound. Probe M1
decides; nothing in Layer A is built before it.

### 3.4 Layer C ceiling

0.13-0.26 ms/frame (MEASURED bubble 0.263 with CUPTI callback instrumentation; conversion 1:1 HYPOTHESIS on the
single main-stream chain; the frame-end node delays the readback copy the app's fence waits on, so that one is 1:1
by construction).

### 3.5 Projection

| | heavy band (8.24 untraced) | whole-window median (6.68) |
|---|---:|---:|
| Layer B | −0.6 … −1.4 (central −0.9) | −0.7 … −1.6 (central −1.0) |
| Layer C | −0.13 … −0.26 (central −0.2) | same |
| Layer A device axis (conditional) | 0 … −0.4 (central −0.1) | 0 … −0.3 |
| **total** | **−0.7 … −2.1 (central −1.2)** | **−0.8 … −2.2 (central −1.3)** |
| **projected frame** | **7.0 (6.1-7.5)** | **5.4 (4.5-5.9)** |

Honest statement: this angle alone reaches ≤5.0 ms on the whole-window RTX median only at the top of its bracket and
does not reach it on the heavy band. It composes with the two structural leads that attack different resources —
geometry run-ahead (closure audit 4.1: −1.80 MODEL heavy, realistic ~half; free syncs make the frame device-chain-bound,
which is the regime where run-ahead pays) and argument-block/counter-clear elimination (0.3-0.6 realistic) — and the
model's "free syncs + launch gaps zero = 4.98 heavy" is the combined shape the goal needs.

### 3.6 B200 (HYPOTHESIS throughout; no B200 trace exists and no B200 price has been measured)

The b200 report's decomposition puts 5.1-6.3 ms of the 10.79 in device idle and attributes it to per-event latency:
24.4 round trips × s_B + host decision × r_cpu + ~350 exposed launches × L_B. Layer B removes the round-trip term
(0.5-1.0 at s_B = 20-40 us) and hides the host decision term (0.5-1.3 at r_cpu 1.4-1.8); Layer C removes 3 host-func
round trips (0.1-0.3); Layer A's device axis attacks the L_B term only if in-graph dispatch is faster on sm_100.
**Ceiling 1.0-2.5 ms/frame; projected 8.3-9.8**, conditional on the B200 price table (b200 report §5) and on the
environment audit (SM clock, PCIe, NUMA, quota) that report ranks first. The 0.3-1.3 ms B200 session drift means only
≥0.5 ms effects are adjudicable there, with ≥3 alternating rounds.

---

## 4. DEAD_ENDS entries touched, and why this is not them

| entry | what it measured | how this plan differs or pays it |
|---|---|---|
| **1** CUDA graphs at three units — 65,905 exact keys, 27-44k instantiations at 14.8 us against a 0.63-0.68 gap ceiling; 4,532 generations, zero repeated recipes; retry-if "stable command-owned storage with fixed device addresses" | Every unit was an **exact-argument cache**. This plan caches **topology templates** and re-parameterises them with `cuGraphExecUpdate`/`SetParams`, which needs no key and no fixed addresses (rotating scratch pointers are exactly what a parameter update carries). Recurrence at the topology level is measured here (§0.2: 523 signatures, 87.6% same-position) and is by construction at the chain unit. Instantiation happens once per template (≤523 × 0.4 us/node × ~60 nodes ≈ 13 ms once per run), not per instance. What entry 1 did not measure and this plan must: the per-node cost of an update **with changed parameters** (M2) — the local 0.39 us/32 nodes figure is treated as a no-change lower bound. Footprint: iteration 20 measured ~5.7 MB RSS + 2.5 MB device per large graph; a top-64 template set is ~370 MB RSS / 160 MB device, stated, not hidden. |
| **11** worst-case sizing — bound/actual 3×10⁷, arena realloc churn, bound-sized clears; retry-if "over-allocation factor a small constant" | Sizing is **exact, on the device, after `quad_count`** (factor 1.0), from a device arena allocated once at high-water × margin; clears are device-sized (no bound-sized memset); the `512u<<10` budget is untouched. Growth stays a rare host event at the per-submit flag. |
| **25** CDP2 7.3 us/link; predicated pre-issue adds ~49.5 launches; conditional graphs closed under entry 1 on a 0.037 prize | **No device-side launches**: the device computes pointers/extents/flags; launches stay host- or graph-issued. Predication is applied only to the success path's consumers (which run anyway) plus one plan kernel per episode (+9.6 launches/frame ≈ +0.02 ms), with **lazy host repair instead of a pre-issued repair arm**. Conditional nodes are priced against the loop iterations and bubbles they remove (B3: 0.22-0.28) at the measured 10.27 us, not against the launch credit, and they are optional (M3). Entry 25's own retry clause — re-run the microbenchmark on sm_100 — is M1-M3 on both GPUs. |
| **29** scratch high-water reuse — 1.65 ms blocked → +0.037; retry-if "after an architectural change removes the next counter dependency, ≥4 alternating pairs" | Included as B4 **only inside the all-sites removal**; its measured one-site zero is the reason the first slice oracles every site at once. |
| **8** smarter flush placement — 76% of flush points empty | The plan kernel relies on the existing flush-before-launch ordering (:531-547) and moves no flush points; nothing about coalescing changes. |
| **17** driver destruction drains arrive at an empty device (3% of the site by time) | The per-submit flag read (B5) is deliberately placed where the device is about to be drained by the app's fence, for the same reason: ~15 us/frame. |
| **19** not issuing degenerate stage-3 launches needs a readback | This plan issues every launch; fixed grids stride over device counts and idle blocks are free (rule 6). Skipping the ~20 empty shade shells on the 30% of drains that find `quads == 0` via an IF node is an optional 0.05 ms (1.6 empty drains × ~34 us). |
| **20/21/22** merge rule — credit only where launches were serial | No launches are merged; a chain graph keeps every node and every branch. |
| **23** hoisting shading-group tables above the drain — zero | Grouping stays on the host (it was already cheap); the plan kernel replaces the prefix sums and allocations, not the grouping. |
| **27** scope-level concurrency refuted (scopes form a chain) | Untouched; the design overlaps the host with the device, not scopes with scopes. Entry 27's chain is why a failure repairs "from the first failed scope to the submit end". |
| **34** threaded submit — 0.126 (capped by the app's immediate fence) | Orthogonal: 34 hides translation behind app recording across the frame boundary; this removes round trips inside a submit. They compose. |
| **36** Renderer 2 refused at design | Not touched; the displaced pools are unchanged. |
| PERFORMANCE **§7 S1d** — drain at the scan: host still waits; 3.75× allocation (slots dense over quads sized from `total`); `capacity/4` gate; fail-closed checks | **No host wait**; sizing after `quad_count` at exact quads, so slots stay dense over quads at 1.0× (invariant "shade slots cannot be packed" is kept); the `capacity/4` gate belongs to `cp_pass_finish_bounded_groups` (:8409-8431), which this path does not use; the fail-closed checks are reproduced on the device with lazy repair. |
| PERFORMANCE **§7 S0** `CUDAVK_UNSAFE_NO_OVERFLOW` −0.38 Crossroads, old dead | That flag routes through the bounded path and sizes to the bound (entry 11's mechanism). This sizes exactly. |
| PERFORMANCE **§7 peel predication** `CUDAVK_PEEL_PREDICATE` −0.11 (over-issued passes) | B3's primary form issues exactly the needed passes (WHILE node); predication is the fallback and only for the discard loop where today's 7 full re-passes already exceed the shells' cost. |
| PERFORMANCE **§6 item 4** run-ahead across the drain — 0.092 converted, reach-limited by "a required resolve came first" | The required resolves *are* the drains; removing them removes the reach limit. Composes. |

---

## 5. Risks, including ARCHITECTURE §8 invariants

1. **The prize may be near the floor of the bracket (0.4-0.5 ms untraced).** The model's host timing is CUPTI-inflated
   and the light window inflates 1.28×. This is exactly what the first slice measures; the plan stops if the oracle
   shows <0.4.
2. **All-or-nothing.** Entry 29 shows one site removed returns ~0. Partial builds (e.g. B1 without B2/B4) will read as
   noise and must not be judged individually; the oracle and the production build both remove every in-frame site.
3. **§8 #4/#5 (arena generations, `batch_uploads_live`).** The device bump arena is reset by a stream op on
   `cp->stream` after the episode's join, so no consumer can still be reading it; the upload ring is untouched. The
   arg-block patch writes into the upload arena's *device* copy after its flush copy — a rewind of the ring under a
   staged batch (the invariant #5 case) would now also invalidate a block a plan kernel is about to patch; the
   `batch_uploads_live` guard must cover the plan kernel's blocks.
4. **§8 #6 (observers flush first).** Every observer must be preceded by the flag read; a scope that samples a failed
   scope's attachment is itself wrong, which is why repair re-executes from the first failed scope to the submit end,
   and why snapshots are needed for `LOAD_OP_LOAD` scopes. Snapshot cost is unpriced until P2.
5. **§8 #7, #8, #9, #10, #11** unchanged: deferring-flush set, stable clipper mode, order-free opaque merging, helper
   lanes (slots stay dense over quads), edge function.
6. **§8 #13/#15.** The patch offsets into `cp_fs_interp_args`, the FS argument block and `cp_abuf_seg_args` become
   ABI numbers and must live in `cp_shader_abi.h`/`cp_rast_types.h` with static assertions; the FS block's pointer
   slots are the fixed slots below `CP_ARG_UBO_BASE`.
7. **§8 #20/#21.** Every layer ships with a revert (`CUDAVK_NO_DEVICE_PLAN`, `CUDAVK_NO_DEVICE_TIMESTAMPS`,
   `CUDAVK_NO_CHAIN_GRAPHS`, registry entries, `cp_debug_doc.py --check`, `cp_no_getenv.py`), and the host-drain
   path stays runnable as both the revert and the repair path. A `CUDAVK_FORCE_EPISODE_FAIL=N` flag forces the device
   flag every Nth episode so the repair path is exercised with output identity as the check (§8 rule 11: check the
   invariant in code).
8. **`cp_pass_can_retry` re-derivation (B6)** changes a fail-closed invariant; the deliberate-failure test is the
   evidence it needs before landing.
9. **Conditional nodes' constraints** (cuda.h:1866-1872): a graph containing one cannot be a child graph, only one
   instantiation may exist, it cannot be cloned; conditional node parameters cannot be updated (body nodes can);
   host nodes are not allowed in bodies; the setter needs `libcudadevrt`. Managed memory cannot back the condition
   (`cp->peel_any` is managed today; it becomes a device word).
10. **`cuGraphExecUpdate` legality** for the driver's actual parameter changes — function changes between generated
    shaders with different register/shared-memory footprints, 1-D memset dimension changes (12 B … 34.6 MB), 1-D
    memcpy size changes (912 B … 31 KB), programmatic edges — is M5, and any refusal means classic issue for that
    template, not a wrong result.
11. **Fan-out capture/branching.** Side streams must join before a segment/chain graph ends; today a `cuStreamSynchronize`
    on the main stream leaves side-stream tails running. Chain graphs express the fan-out as branches, which removes
    this issue but changes where the join dependency sits (the blended composite already joins).
12. **Deterministic-replay oracle validity.** The first slice is correct by construction only on the same capture;
    it is a measurement instrument, never a mechanism.
13. **Measurement rules.** Two submits per frame; arms alternated in one session; stdout hash and 6,939 valid
    timestamps on favorite3 (6,965 favorite2) checked on every run; 18 sentinels before landing; both captures; B200
    with ≥3 alternating rounds and environment pinned (b200 report §6).

---

## 6. First slice — the oracle replay (decisive, no production code)

**Build** (worktree, registry flags `CUDAVK_ORACLE_RECORD=<path>` / `CUDAVK_ORACLE_REPLAY=<path>`, both
diagnostic-only):
- Record mode: at every in-frame decision site, append the values the host reads to a flat log in decision order —
  drain `ctr[0..6+nsegs)` (:8887), segment counters `ctr[0..6)` (:6808), peel `peel_any` per check (:7084) and the
  pass count at which each peel loop exited, plus a per-site ordinal.
- Replay mode: at each site skip `cp_sync_timed` and the `cuMemcpyDtoH`, take the recorded values, and size/launch
  exactly as today (kernels still read device counts, so output is identical if the values match — the stdout hash
  is the check). The peel loop issues the recorded number of passes without checks. The scratch reclaim uses the
  `~/mesa-scratch2` high-water mechanism (DEAD_ENDS 29) so no `cuCtxSynchronize` remains; the descriptor `cuMemcpyHtoD`
  becomes async. A decision-count mismatch aborts.
- Not included: host-func timestamps (Layer C is probed separately by swapping the host func for a
  `cuMemsetD32Async` of the availability word — results wrong, timing right, hash ignored for that arm only).

**Measure**: favorite3 and favorite2, three strictly alternating control/oracle pairs in one session, paired-submit
medians over each capture's real window, hashes and submit counts checked, `CUDAVK_PLAN_STATS=1` on both arms to
confirm zero in-frame waits in the oracle arm.

**Decision**: the oracle's frame delta is an upper bound on Layer B (it omits the plan kernel, the fixed-grid shells and
the snapshot insurance, ≤0.05-0.25 together) and is the whole sync-structure prize measured untraced.
- **≥0.8 ms/frame on favorite3 (and ≥0.5 on favorite2): continue** — Layer B nets ≥0.5 after its costs.
- 0.4-0.8: build Layer B only together with Layer C, and re-derive the projection.
- **<0.4: the sync-structure route is closed at production prices**; the closure documents stand; only Layer A's device
  axis (M1) and the other agents' leads remain.
Effort: 3-5 days.

---

## 7. Probes and microbenchmarks before building anything (with go/no-go thresholds)

Read-only or standalone; the ones marked (GPU) run a standalone binary, never the driver.

| id | what | threshold |
|---|---|---|
| **P0** (done here) | decision-site and topology census on `/tmp/postpin-rtx2` (§0.1, §0.2) | — |
| **P1** | `CUDAVK_DRAIN_PROBE`-style log at HEAD over both captures: `fill_over`, `quad_over`, `running≠quads`, `quads==0`, arena high-water per episode | expected 0 failures; if any failure occurs in the real window the repair path must be priced as a cost, not insurance |
| **P2** | `LOAD_OP` census per render scope on favorite3/favorite2 (a `CUDAVK_DEBUG_RT`-class listing, or parsing the numeric `loadOp` values in the harness's `VkAttachmentDescription`s — the text enum names are absent from the generated code) | snapshot insurance ≤0.1 ms/frame if ≤5 scopes/frame load ≤7.4 MB targets; else subtract the measured copy time from Layer B |
| **P3** | `CUDAVK_DEBUG_DISCARD=1` over ~20 heavy frames: per-pass discarded counts | B3's discard half is live only if passes ≥3 discard nothing |
| **M1** (GPU) | dependent chain of 512 one-block kernels with a ~3 us spin body and a 0 us body: stream plain / stream PDL / graph default edges / graph programmatic edges / graph launched repeatedly; sm_120 and sm_100 (variant of `notes/cdp2-microbench.cu`) | graph edge ≤1.0 us p50 while stream plain ≥1.8 → Layer A device axis ≥0.2 ms (go); graph edge ≥1.6 → Layer A device axis dead |
| **M2** (GPU) | templates of 21 / 58 / 247 nodes (§0.2 p10/p50/long) with pointers, grid dims, 1-D memset and memcpy sizes changed every iteration: (a) `cuGraphExecKernelNodeSetParams` loop, (b) recapture + `cuGraphExecUpdate`, (c) `cuGraphLaunch`; compare to N direct launches at 1.4-1.7 us | per-node ≤0.5 us and launch ≤3 us → host issue cut ≥60% (go for Layer A host axis, relevant only if the oracle fails); per-node ≥1.2 us → Layer A dead on the host axis |
| **M3** (GPU) | WHILE/IF conditional node, body 1 and 6 kernels, condition set by a device kernel linked against `libcudadevrt` via `cuLink`; per-iteration and per-launch overhead; can a body kernel node be `ExecUpdate`d (function change)? | per-iteration overhead ≤10 us → B3 WHILE form (go); >20 us → predication form for the discard loop only |
| **M4** (GPU) | `cuStreamWaitValue32` gate (kernel writes flag → wait → kernel) vs `cuEventRecord`+`cuStreamWaitEvent` as a join | <1.0 us per gate to be considered; otherwise events stay (expected: not used) |
| **M5** (GPU) | `cuGraphExecUpdate` legality census for the driver's real changes: function change between two `main` shaders with different regs/shmem; 1-D memset 12 B ↔ 34.6 MB; 1-D memcpy 912 B ↔ 31 KB; a captured PDL link → programmatic edge → update accepted? | each refusal maps to "that template issues classically"; if function changes are refused, Layer A collapses to fixed-kernel chains only |
| **P4** (B200) | environment audit and the six-price table of the b200 report §5, plus M1-M3 on sm_100 | decides the B200 projection; s_B ≥ 20 us over RTX makes Layer B's B200 value ≥0.5 on round trips alone |

---

## 8. Staged plan

| stage | content | gate to continue | effort |
|---|---|---|---|
| 0 | P1-P3, M1-M5 on RTX; P4 on B200 | thresholds in §7 | 1 week |
| 1 | **First slice**: oracle replay (§6), both captures | favorite3 ≥0.8 ms/frame, favorite2 ≥0.5 | 3-5 days |
| 2 | Layer B production: `cp_abuf_plan` kernel + device arena + arg-block patching + stride consumers + `episode_fail` + per-submit flag read + snapshot/repair + `cp_pass_can_retry` re-derivation + B4 stream-ordered reclaim; revert flag; `CUDAVK_FORCE_EPISODE_FAIL` test; hash + 18 sentinels; both captures; then B200 | measured ≥0.5 ms/frame favorite3 RTX, sign-consistent on favorite2; sentinels clean | 2-3 weeks |
| 3 | Layer C device timestamps | ≥0.1 ms/frame or free; app's `NOT_READY` loop never trips | 1 week |
| 4 | B3 loop exits: WHILE node (M3 pass) or discard-loop predication (P3 live) | ≥0.1 ms/frame | 1 week |
| 5 | Layer A chain graphs, blended tail first, then the direct chain (M1 pass; M2/M5 legal) | ≥0.2 ms/frame beyond stages 2-4 | 2-4 weeks |
| — | combine with geometry run-ahead (closure audit 4.1) and burst argument uploads for the remaining gap | — | other designers |

Total 6-10 weeks if every gate passes; the first decisive number arrives in the second week.

---

## 9. Reproduction

```
cd /tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/design-device-autonomy
python3 seg_census.py  /tmp/postpin-rtx2/trace.sqlite     # §0.1 API census and segment sizes
python3 topo_census.py /tmp/postpin-rtx2/trace.sqlite     # §0.2 topology recurrence
```
Key SQL (mode=ro): `select globalTid,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by 1 order by 2 desc` (main/worker);
`select end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=:worker and nameId in (select id from StringIds where
value='cuEventSynchronize')` (frame boundaries); `select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME
where globalTid=:main order by start` joined to `CUPTI_ACTIVITY_KIND_KERNEL/MEMSET/MEMCPY` on `correlationId` for
names, grids and sizes. Documents: DEAD_ENDS 1, 8, 11, 17, 19-23, 25, 27, 29, 34, 36; PERFORMANCE §4, §5.2b, §6 item 4,
§7; ARCHITECTURE §4, §6.2-6.3, §8; TODO "Performance leads that are open"; notes/DEVICE_EPISODE_CHAINS.md,
CUDA13_UPGRADE.md §4-5; history/EPISODES.md session 11; history/perf-2026-08-27/cuda_leads.md §4; cuda.h 12.8
(:1795-1872 conditional nodes, :1878-1917 programmatic edges, :4259-4266 and :21600-21679 exec update,
:17245 `cuStreamWaitValue32`, :21111 memcpy node update, :22089 conditional handle). Evidence reports: closure-audit,
host-sync, kernel-chain, host-launch-audit, frame-boundary, b200 (paths in the task brief).
