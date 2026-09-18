# Frame-boundary anatomy — favorite3 on cudavk (RTX 5090)

Read-only analysis, 2026-09-02. Nothing was built or run on the GPU. Evidence:
the compiled replay source (`~/favorite3-cpp/out/`), the driver source at the
current branch head, the pre-host-pin nsys trace `/tmp/postdead/trace.sqlite`
(commit `f3caa35e4ce`, nsys 2026.4.1 — confirmed from `META_DATA_CAPTURE`
`_=/opt/nvidia/nsight-systems/2026.4.1/target-linux-x64/nsys`), the pre-pin
no-CUPTI perf profile `/tmp/favorite3-perf-current/`, and the untraced
submit-timestamp files of the host-pin A/B (`/tmp/pin-rtx-ab/favorite3-{c1,c2,c3,p1,p2,p3}/ts.txt`,
3 control + 3 pinned runs, each 6,939 rows, standard hash — see its `driver.log`).

Labels: **MEASURED** = read from a document, source file or trace; **ESTIMATED** =
my arithmetic on measured parts (shown); **HYPOTHESIS** = not measured here.

Scripts and raw per-frame table are in this directory:
`load.py` (window extraction), `anatomy.py`, `uvm.py`, `partition.py`, `host.py`,
`m12.py`, `head.py`, `misc.py`, `kinds.py` (submit classification from source),
`tsab2.py` (untraced interval medians), `per_frame_trace.csv` (411 frames).

---

## 0. Headline

1. **The app runs one frame in flight.** Each frame submits twice — a render
   submit A (two command buffers) and a readback submit B (one
   `vkCmdCopyImageToBuffer`, 1280x720x4 = 3,686,400 B) — and immediately
   `vkWaitForFences` on B's fence. The alternating render fences
   (`pFence_0`/`pFence_1`) are waited too, but always after the previous
   frame's readback fence, so they are already signalled (MEASURED from source,
   §1). The device is therefore idle from the end of the readback copy until the
   next render submit reaches its first launch.

2. **Post-host-pin the boundary interval (readback submit → next render submit)
   is 0.553 ms/frame median** (MEASURED, untraced, pooled three pinned runs,
   2,072 regular frames; light band 0.495, heavy band 0.577). Pre-pin it was
   1.385 (1.272 / 1.470). The host-pin commit's entire frame win (0.916 ms
   pooled) is this interval shrinking (0.832); the render interval A→B moved
   0.075. (§5)

3. **Inside that 0.55 ms the device does one DMA (the readback copy) and
   nothing else.** The pre-pin trace decomposes the same interval (1.73 ms
   traced) as: B issue 9 µs; readback copy 582 µs (DtoD into un-pinned managed
   memory, with 144 HtoD page migrations riding along); worker wake 2.2 µs
   after the copy; **267 µs of application + driver recording** (from the fence
   wake to the first host page fault of `LoadBinaryData`); **757 µs of
   page-fault storm** while the replay memcpy'd 3.6 MB *into* the managed
   readback buffer; 99 µs tail; then 17 µs from the next `vkQueueSubmit` to the
   first device op and ~165-200 µs to the first kernel. Post-pin the storm and
   the slow copy are gone; what remains is record (~0.27) + a fault-free 4.1 MB
   host memcpy (~0.1-0.25, ESTIMATED) + a straight D→H DMA (~0.06-0.15,
   ESTIMATED) + hops (~0.02). (§3, §4, §6)

4. **The boundary is app-owned.** Driver-owned pieces: fence wake path 2.2 µs +
   one condvar hop; submit-entry to first CUDA call 1.9 µs; descriptor arena
   upload 9.6 µs synchronous (64.7 KB); frame head to first kernel 165-200 µs
   traced (32 per-layer 2 KB copies + 32 event records for a 32x32x16 array copy,
   3 host-func launches, one desc upload) — perhaps 0.1 ms untraced; record-time
   driver CPU 0.037 ms/frame (perf). **No driver-only mechanism at the boundary
   reaches 0.2 ms/frame**, and the one ≥0.5 ms mechanism found needs the app to
   stop waiting on the fence of the frame it just submitted (§7, L1).

5. **Answer to TODO item 6 for favorite3:** the "inter-submit stall" is the
   readback submit → next render submit interval. Pre-pin: 1.39 ms untraced,
   of which ~0.58 was the readback copy into migrating managed pages, ~0.76 the
   replay's 3.6 MB write faulting those pages back, ~0.27 recording. Post-pin:
   0.55 ms = recording + memcpy + DMA. It lives in the application's serial
   fence–record–submit chain, not in a driver wait; the driver has nothing
   queued during it *by construction of the app's sync*, and issuing at record
   time is unsafe because the app writes 4.1 MB of mapped memory after recording
   and before submitting (§7, L5).

---

## 1. Per-frame Vulkan call sequence (part a)

Source: `~/favorite3-cpp/out/favorite3.cpp` (only calls `frame_0000_NNNN()` in
order, then a patched final `vkDeviceWaitIdle` + `vkDestroyDevice`, lines
2917-2925) and `~/favorite3-cpp/out/src/frame_0000_*.cpp`. The generated
"frame" files are GFXReconstruct blocks, not app frames: files 0..389 hold the
first 2,783 submits (setup), and steady-state files hold exactly two submits
(1,068 of the 1,518 files from 1391 on; 411 hold none, 20 hold one, 7 hold
three, 12 hold four — texture-upload frames). Submit kinds were classified from
source (`kinds.py`: a submit is **B** if a submitted command buffer recorded
`vkCmdCopyImageToBuffer(... pImage_1 ... pBuffer_327)` since its last
`vkBeginCommandBuffer`, **A** if one recorded a draw, else **U**): 6,947 sites =
3,472 A + 3,470 B + 5 U; after external index 2782: 2,078 A, 2,076 B, 3 U, only
6 non-alternating positions (2989, 2991, 3923, 3925, 4276, 6945). The shim
records 6,939 (the truncated capture faults before the last eight). The trace
confirms the classification: every B is followed within 6-29 µs by the one
3,686,400-byte copy of the frame; every A by 322-492 kernels (§3).

Steady-state frame N, as recorded in `frame_0000_2500.cpp` (heavy band),
in call order. `cbR` is this frame's render command buffer
(`pCommandBuffers_13` on even frames, `pCommandBuffers_1` on odd), `cbU` its
small upload buffer (`pCommandBuffers_3934` / `pCommandBuffers_3`), and the
pools/fences alternate the same way:

| # | call (line in frame_0000_2500.cpp) | count/frame | driver-side effect (file:line) |
|---|---|---:|---|
| 1 | `vkAllocateDescriptorSets` / `vkUpdateDescriptorSetWithTemplate` / `vkCmdBindDescriptorSets` interleaved with draws into `cbR` | 114 / 160 / 215 | host-only; each bind snapshots the set into the command buffer's descriptor arena (`cpvk_cmd.c:1071-1094`, `cpvk_arena_append` memcpy) — 64,704 B/frame median (MEASURED, trace `cuMemcpyHtoD_v2` bytes) |
| 2 | `vkCmdBeginRenderPass`/`EndRenderPass` 11, `vkCmdDrawIndexed` 206 + `vkCmdDraw` 9, `vkCmdBindPipeline` 74, `vkCmdBindVertexBuffers` 96, `vkCmdBindIndexBuffer` 37, `vkCmdSetViewport` 12, `vkCmdSetScissor` 21, `vkCmdPipelineBarrier` 8, `vkCmdDispatch` 1, `vkCmdCopyImage` 1 (32x32x16), `vkCmdWriteTimestamp` 2 | — | host-only recording; `cpvk_EndCommandBuffer` runs `cpvk_plan_batches` (`cpvk_cmd.c:849-972`). **No CUDA call at record time** (trace: zero main-thread CUDA APIs between the fence wake and the next submit except one `cuEventDestroy_v2`, 0.5 µs) |
| 3 | `vkCmdWriteTimestamp(cbR, BOTTOM_OF_PIPE, q3)` (4587) | 1 | at submit: order point + `cuLaunchHostFunc` (`cpvk_cmd.c:4830-4851`) |
| 4 | `vkBeginCommandBuffer(cbU)`; barrier; `vkCmdCopyBufferToImage(pBuffer_2159x → pImage_48)`; barrier; `vkEndCommandBuffer(cbU)` (4513-4590) | 1 | 39,000 B DtoD from a managed staging buffer at submit (trace head, §3.4) |
| 5 | `vkEndCommandBuffer(cbR)` (4593) | 1 | `cpvk_plan_batches` |
| 6 | `LoadBinaryData` ×6 (4595-4599): 163,840 + 221,184 + 4,096 + 36,864 + **3,686,400** + 4,096 B | 6 | plain `memcpy` from eagerly mmapped packs into persistently mapped memory (`global_var.cpp:244-251`). The 3,686,400-B write goes to `ppData_6992` = `pMemory_6991`, **the readback buffer's own memory** (`frame_0000_0003.cpp:2805-2820`): a page-guard capture artefact standing in for the original app's host read |
| 7 | **`vkQueueSubmit(pQueue_0, {cbU, cbR}, pFence_1)`** (4614) — submit **A** | 1 | `cpvk_queue_submit` translates and issues everything synchronously (`cpvk_device_memory.c:151-417`); returns after `cuEventRecord(done)` |
| 8 | `vkWaitForFences(pFence_0)` (4616) — the **previous** frame's render fence | 1 | already signalled (worker retired it before the previous readback fence, same stream order) → returns without blocking |
| 9 | `vkResetFences(pFence_0)`; `vkDestroyBuffer(pBuffer_21596)` (previous frame's 32 KB staging buffer); `vkResetCommandPool(pCommandPool_0)` (4618-4624) | 1 each | reset drops the event ref → the one `cuEventDestroy_v2` per frame; `cpvk_cmd_buffer_reset` (`cpvk_cmd.c:704-738`) frees retired arenas only when they exist (0.1 `cuMemFree`/frame) |
| 10 | `vkGetQueryPoolResults(q0..1, WAIT)` (4627) — the previous frame's timestamps | 1 | condvar on `pool->changed` set by the `cuLaunchHostFunc` callback (`cpvk_cmd.c:4669-4700`, `4743`); already available |
| 11 | `vkBeginCommandBuffer(cbR')` for frame N+1; `vkCmdResetQueryPool`; `vkCmdWriteTimestamp(TOP, q0)` (4635-4641) | 1 | at submit: 2 host funcs at the head of the stream |
| 12 | `vkBeginCommandBuffer(cb3)`; barrier; **`vkCmdCopyImageToBuffer(pImage_1 → pBuffer_327, 1280x720)`**; barrier; `vkEndCommandBuffer` (4650-4724) | 1 | at submit: `cpvk_execute_copy` → one `cuMemcpy2DAsync_v2` on the main stream (`cpvk_cmd.c:4307`; trace §3) |
| 13 | `LoadBinaryData` ×2, 4,096 B each into `ppData_3286` (4726-4727) | 2 | host memcpy into a managed UBO allocation |
| 14 | **`vkQueueSubmit(pQueue_0, {cb3}, pFence_3)`** (4741) — submit **B** | 1 | copy issued 9 µs after the call (trace) |
| 15 | **`vkWaitForFences(pFence_3)`** (4743) — **this frame's readback** | 1 | the real wait: `cpvk_sync_wait` → `cnd_wait` (`cpvk_sync.c:354-397`) |
| 16 | `vkResetFences(pFence_3)`; `vkCreateBuffer(32768, TRANSFER_SRC)`; `vkGetBufferMemoryRequirements`; `vkBindBufferMemory(pMemory_3285, 49152)` (4745-4763) | 1 each | host-only (`cpvk_device_memory.c:1048-1152`); post-pin `cpvk_BindBufferMemory2` also runs `cuMemAdvise` for transfer-dst-only buffers — not this one (usage TRANSFER_SRC) |
| 17 | recording of frame N+1 continues: barriers, `vkCmdCopyImage(pImage_48 → pImage_46, 32x32x16)`, compute `vkCmdBindPipeline`, `vkResetDescriptorPool(pDescriptorPool_0)`, `vkAllocateDescriptorSets` ..., `vkCmdDispatch(5,3,1)`, render passes ... (4765-10799 and into the next file) | — | host-only |

Confirmations (MEASURED from source):

- **Frames in flight: effectively one.** Step 15 waits the fence of the submit
  issued at step 14, every frame. Step 8's wait is on the previous frame's
  render fence, which the worker signalled before that frame's readback fence.
- **`vkQueueWaitIdle`/`vkDeviceWaitIdle`: none per frame.** Over all 2,909
  frame files exactly one contains a WaitIdle (`frame_0000_2903.cpp`); the
  others are the patched end of `main()` and setup code in
  `swapchain_common.cpp:570`. `APP_IMPROVEMENTS.md` ("removing the per-frame
  vkDeviceWaitIdle + fence-based sync: -2.3 ms/frame") is confirmed.
- **No `vkMapMemory`/`vkUnmapMemory`/`vkFlushMappedMemoryRanges` per frame:**
  `vkMapMemory` appears in 28 setup files only; `vkFlushMappedMemoryRanges`
  never (the driver's mappable types are coherent, `cpvk_device_memory.c:1034`).
  All mapped pointers are persistent.
- **No acquire/present:** zero frame files call `vkAcquireNextImageKHR` or
  `vkQueuePresentKHR`; the readback is the frame's only output.
- **Host-written bytes/frame:** 4,124,672 (frames 2499-2502), 4,370,432
  (frame 1501), 3,981,312 (frame 2900) — `LoadBinaryData` sums; ≥3,686,400 of
  each is the write into the readback buffer. **Host-read bytes/frame: 0** in
  the replay (GFXReconstruct does not replay host reads; the shim's dump mode
  is off in timing runs, `submit_shim.c`).
- **Per-frame allocations:** `vkCreateBuffer`/`vkDestroyBuffer` of a 32 KB
  staging buffer bound into existing memory; no `vkAllocateMemory` in frames
  1391+ (0 in files 1391-2908).

Timestamp semantics: `~/favorite-cpp/submit_shim.c` records `CLOCK_MONOTONIC`
**at `vkQueueSubmit` entry**, before the driver runs. The trace's
`TARGET_INFO_SESSION_START_TIME.systemClockNs` (1503511535334936) is the same
clock: subtracting it maps shim timestamps onto the trace; the readback copy
then starts 6.4-29 µs (median ~15) after every B timestamp, with no drift over
the 3-second window (`align.py`).

## 2. Driver-side paths (part b)

Read at the current branch head (`cudapipe-vk-native`).

**Submit path** (`cpvk_device_memory.c:151-417`, `cpvk_queue_submit`), in
order, before the first kernel of a render submit:

1. `cp_smallop_hit` telemetry; device-lost check; `CPVK_CTX_SCOPE`
   (`cuCtxPushCurrent`, `cpvk_private.h:215`).
2. Wait semaphores: none in this app.
3. `submit_lock`: if the previous submit is not yet retired **and**
   `renderer.scratch.used >= 256 MiB`, `cpvk_submit_wait_pending` blocks on the
   worker (line 187-194). Trace shows the B copy issued 9 µs after the B
   call (median), so this does not fire measurably. If retired,
   `cp_scratch_reset` (frees overflow arenas; 0.1 `cuMemFree`/frame).
4. Per command buffer: publish host maps; if the descriptor arena is dirty,
   **synchronous `cuMemcpyHtoD`** of the arena (line 240-256): 1.0/frame,
   64,704 B median, 8.1 µs median API (9.6 µs mean) — MEASURED; matches the
   documented 0.008 ms/frame.
5. Ops in record order. The first ops of the render submit are `cbU`'s barrier
   + `CopyBufferToImage` (one 39 KB DtoD) + barrier, then `cbR`'s reset-query
   and TOP timestamp (2 `cuLaunchHostFunc`), then the 32x32x16 `CopyImage`
   (16 layer copies, each `cuMemcpy2DAsync_v2` + `cuEventRecord`), the compute
   dispatch, then the render scopes. Barriers and queries call
   `cpvk_execute_order_point` = `cp_batch_flush_why` + `cp_pass_finish`
   (`cpvk_cmd.c:4253-4257`), i.e. they force the episode drain.
6. Inside `cpvk_execute_draw_cmd`: `cp_scratch_begin` (`cp_renderer.c:666-690`)
   does **`cuCtxSynchronize`** when `dscratch.used > CP_SCRATCH_RECLAIM_BYTES`
   (the >1 GiB drain, DEAD_ENDS 29): 2.0 per frame in the trace, at 653 µs and
   2,124 µs after the submit, blocking 1,159 and 615 µs (medians), 1.59 ms/frame
   mean. `cp_sync_timed` (`cp_renderer.c:713`) wraps `cuStreamSynchronize`:
   11.4/frame, 1.27 ms/frame; `cuMemcpyDtoH` 8.9/frame, 0.065 ms.
7. End: `cp_batch_flush`, `cp_upload_flush`, `cuEventCreate` + `cuEventRecord`
   on the main stream, `cpvk_sync_arm` each signalled fence (`cpvk_sync.c:139`),
   append the pending record, `cnd_signal(submit_changed)`. No drain.

**Completion worker** (`cpvk_submit_worker`, `cpvk_device_memory.c:46-86`):
one thread; `cuEventSynchronize(pending->done)` in a loop, then
`cpvk_sync_signal_completion` (`cpvk_sync.c:167`) → `cnd_broadcast(sync->changed)`.
Context flags default `CU_CTX_SCHED_AUTO` (`cpvk_device_memory.c:482-491`); with
32 CPUs and one GPU that is spin — the perf profile shows the worker's samples
in libcuda + vDSO `clock_gettime` (folded stack
`cpvk_submit_worker;cuEventSynchronize;...;__GI___clock_gettime`), and the trace
shows the worker's `cuEventSynchronize` returning **2.2 µs median (p90 2.9)
after the readback copy's CUPTI end** (§3).

**Fence wait path** (`vkWaitForFences` → Mesa `vk_common_WaitForFences` →
`vk_sync_wait_many` → `cpvk_sync_wait`, `cpvk_sync.c:354-397`): the app thread
does **not** touch CUDA; with `timeout = UINT64_MAX` it `cnd_wait`s on
`sync->changed`. **Thread hops per fence wait: one** (worker → app via
condvar/futex). The app thread's wake latency after the broadcast is not
observable in the trace (recording makes no CUDA calls); typical futex wake
5-30 µs (HYPOTHESIS). Mesa's queue is in `VK_QUEUE_SUBMIT_MODE_IMMEDIATE`
(`vk_queue.c:69-71`; threaded only under `MESA_VK_ENABLE_SUBMIT_THREAD`,
DEAD_ENDS 34), so `vkQueueSubmit` runs the driver synchronously on the app
thread — no hop on the submit path.

**Record-time cost:** `cpvk_BeginCommandBuffer` (`cpvk_cmd.c:796-831`) resets
state and frees retired arenas (none in steady state); descriptor binds memcpy
snapshots; `cpvk_EndCommandBuffer` runs `cpvk_plan_batches`. perf (pre-pin,
`/tmp/favorite3-perf-current/flat.txt`, 2,078 real frames, 8.457 ms/frame,
main thread 97% on-CPU): all `cpvk_Cmd*`/descriptor/`vk_cmd_*`/plan symbols sum
to **0.44% of main-thread samples ≈ 0.037 ms/frame** (MEASURED). The replay's
own code (`vulkan_app` DSO self) is 0.05%; libc `__memmove_avx512_unaligned_erms`
is 4.99% ≈ **0.42 ms/frame** — the app's 4.1 MB `LoadBinaryData` plus the
driver's small staging/snapshot copies (user-mode cycles only; page-fault time
is kernel and excluded by `cycles:Pu`). The 7.0% in vDSO `clock_gettime` is
inside libcuda's spin loops (`cuCtxSynchronize`, `cuStreamSynchronize`,
`cuEventSynchronize`), not driver code.

## 3. Trace measurements (part c)

### 3.1 Window and method

`/tmp/postdead/trace.sqlite`, commit `f3caa35e4ce` (2026-08-30, **before** the
host-pin commit `1acb7eabc1b` of 2026-08-31 17:41). Device activity exists from
1.04 s to 12.34 s relative; the analysis uses external submit indices A =
2783..3605 (odd = render submit in this stretch, verified against the copy
pattern), i.e. **411 regular frames** from 9.278 s to 12.34 s
(`frames.pkl`, `per_frame_trace.csv`). One frame (A = 2989) is dropped because
two render submits occur without a readback between them (`kinds.py`).

Frame k := [ts(A_k), ts(A_k+1)). Device ops = union of
`CUPTI_ACTIVITY_KIND_KERNEL`, `_MEMCPY`, `_MEMSET` over all 16 streams.
UVM migration rows (`copyKind` 11/12) are reported separately and excluded
from "real" busy time. Equivalent SQL for the extraction (`load.py` runs these
with `LO=9.0e9, HI=12.5e9`):

```sql
select systemClockNs from TARGET_INFO_SESSION_START_TIME;            -- 1503511535334936
select start,end,streamId,shortName,gridX*gridY*gridZ,correlationId
  from CUPTI_ACTIVITY_KIND_KERNEL where start>=:LO and start<:HI order by start;
select start,end,streamId,copyKind,bytes,correlationId
  from CUPTI_ACTIVITY_KIND_MEMCPY where start>=:LO and start<:HI order by start;
select start,end,streamId,bytes,correlationId
  from CUPTI_ACTIVITY_KIND_MEMSET where start>=:LO and start<:HI order by start;
select start,end,globalTid,nameId,correlationId
  from CUPTI_ACTIVITY_KIND_RUNTIME where start>=:LO-5e8 and start<:HI order by start;
select start,end,streamId,syncType,correlationId,eventId
  from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=:LO-5e8 and start<:HI;
-- readback copies: 3,606 rows of 3,686,400 B in the whole trace, 883 in window
select start,end,copyKind,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY
  where bytes=3686400 order by start;
-- copy kinds: 7 = DtoA (texture-cache mirror of the rendered image, 7.4 us),
--             8 = DtoD (the readback into managed memory, 485-814 us),
--            11/12 = UVM HtoD / DtoH migrations (migrationCause 2 = page fault, 3 = prefetch)
select migrationCause, copyKind, count(*), sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY
  where copyKind in (11,12) group by 1,2;
-- (2,11,285330,3.17 GB) (2,12,82860,0.34 GB) (3,11,253926,10.56 GB) (3,12,77877,13.24 GB)
```

Threads: main `globalTid` 297865344641070 (1,210,485 API rows), worker
297865344641096 (6,950 rows: 6,947 `cuEventSynchronize` = one per submit).

### 3.2 Per-frame partition of the frame (411 frames, pre-pin, traced)

`partition.py`. `head` = ts(A) → first device op; `internal` = gaps in the
union of the render submit's device ops (starts before ts(B)); `inter` = last
render device op end → readback copy start; `rb` = the readback copy;
`bnd` = readback copy end → ts(A_k+1). `check` = frame − busy − head −
internal − inter − bnd = 0 on every frame.

| part | median | p10 | p90 | mean | unit |
|---|---:|---:|---:|---:|---|
| frame (ts A → next ts A) | 7.166 | 5.494 | 8.714 | 7.419 | ms |
| device busy, real (kernels+copies+memsets, no UVM) | 4.439 | 3.415 | 5.252 | 4.382 | ms |
| device busy incl. UVM migration rows | 4.562 | 3.519 | 5.391 | 4.502 | ms |
| **device idle, real** | **2.655** | 1.994 | 3.544 | 3.037 | ms |
| head: ts(A) → first device op (a DtoD copy) | 17.1 | 14.6 | 22.6 | 18.2 | µs |
| ts(A) → first kernel | 199.6 | 156.8 | 608.2 | 284.0 | µs |
| render span (ts(A) → last render device op) | 5.246 | 3.766 | 6.499 | 5.191 | ms |
| render-phase device busy | 3.823 | 2.867 | 4.627 | 3.765 | ms |
| **internal gaps inside the render phase** | **1.375** | 0.878 | 2.069 | 1.426 | ms |
| … of which gaps ≤ 20 µs | 1.170 | 0.805 | 1.334 | 1.113 | ms |
| … 20-200 µs | 0.125 | 0.041 | 0.193 | 0.117 | ms |
| … > 200 µs | 0.000 | 0.000 | 0.653 | 0.195 | ms |
| busy segments in the render phase (≈ gaps + 1) | 689 | 440 | 764 | 633 | — |
| **inter-submit gap** (last render op → readback copy) | **74.7** | 54.7 | 310.5 | 140.0 | µs |
| **readback copy** (DtoD, managed dst, stream 13) | **581.5** | 484.6 | 814.4 | 617.1 | µs |
| **boundary** (readback end → next ts(A)) | **1,119** | 993 | 1,410 | 1,453 | µs |

Mean-based split of the device idle: 3.037 ms = boundary-side (bnd + inter +
head) **1.611** + internal **1.426**. By frame-length tertile (medians):

| tertile | frame | busy | idle | head µs | internal | inter µs | rb µs | bnd µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light third | 5.642 | 3.529 | 2.126 | 16 | 0.953 | 66 | 564 | 1,024 |
| mid third | 7.166 | 4.474 | 2.645 | 18 | 1.388 | 72 | 576 | 1,136 |
| heavy third | 8.291 | 4.991 | 3.324 | 18 | 1.820 | 104 | 608 | 1,209 |

The boundary is a near-constant per-frame cost (1.0-1.2 ms pre-pin); the
internal idle scales with frame weight and is almost entirely (85%) in gaps
under 20 µs — the launch-issue floor, ~690 gaps a frame — not in the ~17 host
blocking waits (gaps over 200 µs are 0 in the median frame, 0.195 ms mean).

### 3.3 What bounds each gap (host markers, `anatomy.py`, `misc.py`)

| marker | median | p10 | p90 | mean | unit | reading |
|---|---:|---:|---:|---:|---|---|
| ts(A) → first main-thread CUDA API | 1.89 | 1.40 | 3.45 | 2.48 | µs | driver entry: `cuMemcpy2DAsync`/`cuMemcpyDtoDAsync` of `cbU`'s copy |
| descriptor upload `cuMemcpyHtoD_v2` | 8.1 | 6.2 | 12.4 | 8.8 | µs | 64,704 B median, 1.0/frame |
| first-kernel `cuLaunchKernel` API offset | 165.4 | — | — | — | µs | preceded by 32× (`cuMemcpy2DAsync_v2` 2.7 µs + `cuEventRecord` 0.2 µs) for the 32x32x16 array copy, 2× `cuLaunchHostFunc`, the desc upload |
| kernel start after its launch API returned | 33.8 | — | 109.6 | — | µs | the stream is still behind the copies/host funcs |
| last main-thread API of submit A → last render device op | 59.8 | 38.9 | 172.1 | 101.1 | µs | the device finishes ~60 µs after `vkQueueSubmit` A returns |
| A returns → ts(B) (app work between submits, steps 8-14) | 68.5 | 47.4 | 186.3 | 115.4 | µs | pure app+driver CPU, no CUDA call |
| ts(B) → readback copy start | 9.1 | 7.5 | 15.3 | 10.7 | µs | B's translation: `cuMemcpy2DAsync_v2` (2 µs in, 4.6-7.9 µs API) + `cuEventCreate` + `cuEventRecord` |
| worker `cuEventSynchronize` end − readback copy end | 2.24 | 1.85 | 2.87 | 2.43 | µs | fence completion detected by the spinning worker |
| readback end → first UVM DtoH copy (record proper) | 267 | 226 | 354 | — | µs | `vkResetFences`, `vkCreateBuffer`…, ~1,000 recording calls |
| UVM DtoH fault-storm span | 757 | 658 | 987 | — | µs | 46 copies (p10 43, p90 57), 4.35 MB, 4 KB fault + 60 KB..1 MB prefetch pairs every 12-112 µs: the replay's 3.6 MB memcpy into the device-resident managed readback buffer |
| last UVM copy → next ts(A) | 99 | 7 | 111 | — | µs | tail of the memcpy + last `LoadBinaryData` + call overhead |
| CUDA calls between readback end and next ts(A), whole window | — | — | — | — | — | 411 `cuEventDestroy_v2` (0.5 µs each), 6 `cuMemAlloc_v2` (84 µs each, arena growth), 4 `cuModuleLoadDataEx` (325 µs each) |

Inter-submit outliers: only 2 of 411 frames exceed 200 µs (A = 3023: 1,729 µs
with 1,097 µs of app work between the submits; A = 3025: 225 µs). The gap is
bounded by the **app's CPU work between the two submits** (68.5 µs) plus the
9 µs B translation, minus the ~60 µs the device is still finishing A.

Blocking on the main thread per frame (trace, CUPTI-inflated): `cuCtxSynchronize`
2.0/frame 1.593 ms; `cuStreamSynchronize` 11.4/frame 1.270 ms; `cuMemcpyDtoH_v2`
8.9/frame 0.065 ms; total blocked 3.01 ms median of an API-busy 4.84 ms; API
issue (busy − blocked) 1.75 ms; host time outside any CUDA API 2.29 ms median
(= record 1.12 + between-submits 0.07 + ~1.1 of driver planning between calls
inside the submit). Worker: exactly 2 `cuEventSynchronize` per frame; the one
for A starts 5,266 µs and ends 5,317 µs after ts(A) (render span 5,359).

### 3.4 UVM migration traffic per frame (pre-pin, `uvm.py`)

| kind | phase | count median | MB/frame mean | summed DMA µs/frame |
|---|---|---:|---:|---:|
| UVM HtoD (11) | render (A) | 15 | 1.358 | 51.5 |
| UVM HtoD (11) | boundary (B) | **144** | **3.686** | 217.6 |
| UVM DtoH (12) | render (A) | 15 | 0.396 | 17.8 |
| UVM DtoH (12) | boundary (B) | **46** | **4.619** | 103.7 |
| HtoD pinned (1) | render | 148 | 0.305 | 133.1 |
| DtoD (8) | render | 35 | 0.384 | 153.0 |
| DtoD (8) readback | boundary | 1 | 3.686 | 617.1 |
| DtoA (7) | render | 5 | 10.372 | 22.9 |

The 144 HtoD migrations in phase B are the readback buffer being pulled back to
the device under the copy (the host had just written it); the 46 DtoH
migrations are the same pages faulting back under the app's next write. This
is exactly the cycle `history/perf-2026-08-31-host-pin-readback.md` describes
and removed. Note this trace is nsys **2026.4.1 on the RTX** and does show UVM
rows — DEAD_ENDS 37's "zero UVM rows on RTX" is a **post-pin** observation. The
residual per-frame ping-pong of the UBO/vertex allocations seen here in phase A
(15 + 15 copies, 1.75 MB, ~70 µs of DMA) is not explained by 37; it is small
and the residency-advice mechanism against it is refuted regardless (+1.98 ms).

## 4. Timeline of the median frame (A = 3069, 7.166 ms; `onefame.py`)

Offsets in µs from ts(A). Device rows are CUPTI intervals; host rows are
main-thread API calls.

```
     0.0   TS    vkQueueSubmit A (shim)
     2.1   MAIN  cuMemcpyDtoDAsync (cbU: CopyBufferToImage, 39,000 B)        -> DEV 14.2 +47.5 us
    48.7.. MAIN  16x { cuMemcpy2DAsync 2.7us ; cuEventRecord 0.2us }  (CopyImage 32x32x16, 2,048 B/layer)
   102.8   MAIN  cuMemcpyHtoD_v2  9.3 us   (descriptor arena, sync)
   112.5   MAIN  cuLaunchHostFunc x2       (ResetQueryPool, WriteTimestamp TOP)
   116.5.. MAIN  16x { cuMemcpy2DAsync ; cuEventRecord }
   173.3   MAIN  cuLaunchKernel            -> first kernel DEV 178.6 (main, grid 15, 67 us)
   680.9   MAIN  cuCtxSynchronize  1,181 us  (cp_scratch_begin, >1 GiB drain)
  2142.5   MAIN  cuCtxSynchronize    680 us
  2856..5237  MAIN  14x cuStreamSynchronize (41-192 us each, 1,266 us total)  (drains, segment counters)
  5210     MAIN  cuEventRecord + cuStreamWaitEvent joins; cuEventCreate/Record(done)  = submit A returns
  ~5300    DEV   cp_abuf_composite (last render kernel) ends; worker cuEventSynchronize(A) returns
  5530.2   TS    vkQueueSubmit B (shim)             [app work between submits: ~70 us]
  5539.9   DEV   readback DtoD 3,686,400 B  +466 us (this frame; window median 582)
  6006.4   DEV   readback ends; worker wakes +2.2 us; cnd_broadcast -> app thread
  6287.4   DEV   first UVM DtoH (4 KB + 60 KB)      [record proper: 281 us]
  6287..7054 DEV 47 UVM DtoH copies, 4.39 MB        [LoadBinaryData 3.6 MB faulting: 767 us]
  7165.8   TS    vkQueueSubmit A' (shim)            [tail 111 us]
  7167.6   MAIN  cuMemcpy2DAsync (next frame's first CUDA call, +1.8 us)
  7182.6   DEV   next frame's first device op (+16.8 us)
```

## 5. Untraced, current driver: the boundary interval from the host-pin A/B (`tsab2.py`)

Submit timestamps only (shim, `CLOCK_MONOTONIC`, entry of `vkQueueSubmit`), so
these are **untraced** and **MEASURED**. Regular frames = A, B, A in sequence
per `kinds.py`. A→B = render-submit call + app work between submits;
B→A' = readback submit call + copy + fence wait + record + memcpy. Medians.

| band | arm | n | frame | A→B | B→A' | B→A' p10 / p90 |
|---|---|---:|---:|---:|---:|---|
| whole real window (idx ≥ 2782) | control c1/c2/c3 | 2,072 | 8.014 / 7.521 / 7.594 | 6.107 / 6.114 / 6.138 | **1.563 / 1.385 / 1.385** | 1.15-1.24 / 1.75-3.58 |
| | pinned p1/p2/p3 | 2,072 | 6.638 / 6.691 / 6.679 | 6.034 / 6.039 / 6.074 | **0.553 / 0.552 / 0.555** | 0.43 / 0.85-0.89 |
| | pooled delta | | **0.916** | 0.075 | **0.832** | |
| light band (frames 1700-2100) | control | 398 | 4.850 | 3.507 | **1.272** | 1.10-1.18 / 1.57-3.45 |
| | pinned | 398 | 3.971 | 3.423 | **0.495** | 0.42 / 0.77-0.80 |
| heavy band (frames 2150-3150) | control | 1,000 | 9.067 | 7.519 | **1.470** | 1.22-1.32 / 1.78-3.67 |
| | pinned | 1,000 | 8.157 | 7.463 | **0.577** | 0.46-0.48 / 0.87-0.95 |
| traced pre-pin (postdead), trace window | — | 411 | 7.166 | 5.367 | 1.730 | 1.56 / 2.10 |
| traced pre-pin, heavy band | — | 1,000 | 9.637 | 7.816 | 1.798 | 1.61 / 2.08 |
| perf run (pre-pin, perf attached), whole | — | 2,072 | 7.681 | 6.161 | 1.542 | 1.28 / 1.82 |

Readings:

- The host-pin win is the boundary interval, to within noise: frame −0.916,
  B→A' −0.832, A→B −0.075 (whole window).
- CUPTI adds ~0.19 ms to B→A' (1.730 traced vs 1.542 perf-attached vs 1.385
  untraced control) — mostly the copy API and the 144 HtoD migrations' records
  — and ~0.2 ms to A→B.
- **Post-pin the boundary interval is 0.49-0.58 ms and nearly flat across
  bands** (light 0.495, heavy 0.577): it is a fixed per-frame cost.

ESTIMATED decomposition of the post-pin 0.553 ms, from the traced pre-pin parts
that the pin does not change (record 267 µs, B issue ~5 µs untraced, wake 2 µs,
tail ~0.1 ms including the last `LoadBinaryData` and the call), constrained to
the measured total:

| piece | ms | label |
|---|---:|---|
| B translation + copy issue | 0.005-0.01 | MEASURED 9 µs traced |
| readback DMA 3.6 MB device → host-resident pages | 0.06-0.15 | HYPOTHESIS (PCIe 5.0 x16 per `nvidia-smi -q`; B200 post-pin measured 0.323 ms/frame total DtoD in the pin record) |
| worker wake + condvar hop to app thread | 0.01-0.03 | wake 2.2 µs MEASURED; hop HYPOTHESIS |
| app + driver recording (~1,000 calls; driver share 0.037) | 0.27 | MEASURED (wake → first fault) |
| `LoadBinaryData` 4.1 MB memcpy, fault-free | 0.10-0.25 | ESTIMATED (perf user-mode memmove 0.42 ms/frame pre-pin was fault-interleaved) |
| tail + `vkQueueSubmit` entry to first device op | 0.02-0.1 | MEASURED 17 µs entry; tail bounded above by the pre-pin 99 µs |
| **sum** | **0.47-0.81** | brackets the measured 0.553 |

## 6. Where the device idle is (part d)

Pre-pin, traced (411-frame window, mean-based): idle 3.037 = boundary-side
1.611 (bnd 1.453 + inter 0.140 + head 0.018) + internal 1.426 → **53% of the
device idle was at the frame boundary**, 47% inside the render phase.

Post-pin, ESTIMATED for the current driver: boundary-side device idle ≈
(B→A' 0.553 − DMA 0.06-0.15) + inter 0.075 + head-to-first-kernel ~0.1
(untraced guess for the traced 0.17-0.2) ≈ **0.55-0.7 ms/frame**; internal
idle ≈ 1.2-1.4 traced, of which CUPTI accounts for roughly 0.2 (A→B traced vs
untraced), so **≈1.0-1.2 ms untraced** — i.e. now roughly one third boundary,
two thirds launch-floor gaps inside the render phase. Consistency: 6.68 −
(0.55..0.7) − (1.0..1.2) ≈ 4.8-5.1 ms busy against 4.44 traced real busy in
this window (heavier frames in the whole window) — plausible.

**Caution on the documented 1.673 ms untraced idle ceiling** (78% busy from a
20 ms `nvidia-smi` sampler, pre-pin): the pre-pin boundary alone was 1.39 ms
untraced with no kernel running, which leaves only ~0.3 ms for the ~690
launch gaps a frame that the trace prices at ≥1.0 ms. The two cannot both be
right; the sampler's granularity makes the submit-timestamp measurement the one
to trust for the boundary.

**Who owns the boundary (post-pin, per frame, ESTIMATED from measured parts):**

| actor | ms/frame | what |
|---|---:|---|
| application CPU | ~0.40-0.55 | recording (~0.23 excluding the driver's 0.037), `LoadBinaryData` memcpy 4.1 MB (0.1-0.25), work between the two submits (0.07) |
| GPU DMA the app's fence forces | 0.06-0.15 | the readback copy; the app waits on it before recording |
| driver CPU | ~0.04-0.15 | record-time 0.037; frame head issue to first kernel (~0.1 untraced) |
| driver wake-up latency | 0.01-0.03 | 2.2 µs event detection + one condvar hop |

## 7. Leads, each checked against DEAD_ENDS.md

**L1. App + driver: pipeline the readback fence by one frame *and* run Mesa's
threaded submit.** Mechanism: the app waits `pFence_3` of frame N−1 (not N)
before recording N+1 (two readback buffers), so its recording + memcpy
(~0.4-0.5 ms) overlaps frame N; but because `vkQueueSubmit` A is synchronous
and blocks ~17 times on the device, the device is nearly drained when A returns
(+60 µs), so the app-only change overlaps just the copy (~0.1 ms). With the
Mesa submit thread (`MESA_VK_ENABLE_SUBMIT_THREAD=1`, DEAD_ENDS 34's diagnostic
branch) the app thread records N+1 while the submit thread translates N.
Ceiling ESTIMATED = B→A' 0.553 + between-submits 0.07 ≈ **0.5-0.6 ms/frame
RTX**; B200 similar or larger (its post-pin copy alone is 0.32 ms).
Closest dead end: **34** (fully threaded submit, 0.126 ms/frame) — that
measurement kept the app's fence-on-the-frame-just-submitted, which is exactly
what capped it; 34's own retry clause ("a later workload has materially more
application recording … available to overlap") is satisfied by the app change.
Also **25** (device-side chaining) is not this. Cost: the 34 branch's lifecycle
fixes plus an app change; the app-only half is measurable first with a
two-line edit to the replay (wait the previous frame's `pFence_3`) — expected
≈0.1 ms, which would confirm the mechanism before the driver half is touched.
This is the only ≥0.5 ms mechanism found, and it is not a driver-only one.

**L2. Driver: shorten the frame head.** 165-200 µs traced from submit to first
kernel: 32× (2 KB `cuMemcpy2DAsync` + `cuEventRecord`) for one 32x32x16 layered
`vkCmdCopyImage`/`vkCmdCopyBufferToImage`, 2 host funcs, one desc upload. One
`cuMemcpy3DAsync` per copy and one event would remove ~30 copies + ~30 records.
Ceiling ESTIMATED at PERFORMANCE §4's removal prices (1.30 µs/small copy,
0.6-1.0 µs/launch-like op): ~30×1.3 + ~30×0.8 ≈ **0.06 ms/frame**; the traced
0.1-0.15 is CUPTI-inflated. Closest dead ends: **8** (flush placement) and the
§4 "removing an operation outright" row; different mechanism (a 3D copy instead
of per-layer copies), same price class. Sub-gate.

**L3. Driver: wake path.** Worker detects completion 2.2 µs after the copy; one
condvar hop to the app thread. Letting `cpvk_sync_wait` call
`cuEventSynchronize` on the app thread when the sync carries an event would
remove the hop. Ceiling **≤0.03 ms/frame** (HYPOTHESIS on futex latency). Not
worth its correctness surface.

**L4. Driver: record-time work.** 0.037 ms/frame measured (perf). Nothing to
collect.

**L5. "Issue device work from record time" (TODO item 6's suggestion) —
REFUTED by construction.** The app writes 4.1 MB into persistently mapped
memory (UBO/vertex/instance data and the readback buffer) *after* recording and
*before* `vkQueueSubmit` (steps 6 and 13, §1), which Vulkan permits; work
issued at record time would read stale data. The ceiling, were it legal, would
be the record-phase idle (~0.3-0.5 ms). Closest dead end: 34 (translation off
the app thread is the legal form of the same idea).

**L6. Driver: readback copy speed.** DMA-bound; ceiling = the copy's own
0.06-0.15 ms and the app waits on it regardless. No mechanism.

**L7. App-only (for APP_IMPROVEMENTS.md, not the driver):** the replay's
3.6 MB `LoadBinaryData` into the readback buffer is a capture artefact (page
guard re-fill of pages the original app read); the original app's host read is
the same size. Nothing driver-side.

## 8. Unknowns

- The post-pin readback DMA time on RTX (0.06-0.15 ms is a range, not a
  measurement); a post-pin nsys trace would settle it and the post-pin
  boundary partition in one run.
- The app thread's condvar wake latency (invisible in the trace).
- Why DEAD_ENDS 37 saw zero UVM rows on RTX post-pin while this pre-pin
  2026.4.1 RTX trace shows ~30 UBO/vertex migrations a frame in the render
  phase (1.75 MB); irrelevant to the boundary, relevant to the "device pacing"
  story.
- The 78%-busy sampler-derived 1.673 ms idle ceiling is inconsistent with the
  boundary measured here; one of them is wrong (§6).
- The per-frame `cpvk_submit_wait_pending` path at ≥256 MiB scratch never
  fired measurably in the window; whether it fires in the heavy band is
  untested (B issue latency there is unknown without a trace).
