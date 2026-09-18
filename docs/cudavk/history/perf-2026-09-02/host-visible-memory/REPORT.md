# Host-visible memory path in cudavk — favorite3, read-only analysis (2026-09-02)

Analyst topic: trace the physical life of application-visible memory in cudavk and price it.
Everything below is labelled **MEASURED** (from a document, the code, or a trace query shown here),
**ESTIMATED** (arithmetic on measured inputs, shown) or **HYPOTHESIS**.

Scratch files (scripts that produced the harness numbers) live next to this report:
`parse_harness.py`, `parse_consumers.py`, `harness_census.json`.

## 0. Verdicts in one screen

1. **Both host-visible Vulkan memory types are `cuMemAllocManaged`** (`cpvk_device_memory.c:804-815`); type 2 is
   advertised DEVICE_LOCAL|HOST_VISIBLE but is the same managed allocation. `vkMapMemory` hands out the managed
   pointer itself (`:1023`); Unmap/Flush/Invalidate are no-ops (`:1027-1045`). Shaders read UBO/vertex/index data
   **directly from managed memory** through raw device addresses (`cpvk_cmd.c:329-337`, `:1995-1996`, `:2048-2050`).
2. **The harness writes 4.09 MB/frame into mapped host-visible memory** (MEASURED: 14,189,218,336 bytes over
   3,473 frames), of which **3,686,400 B/frame (90%) is a capture artefact**: the GFXReconstruct trace recorded
   the GPU-written readback buffer contents as host fills, and the replay memcpy's them back every frame before
   the GPU overwrites the whole buffer again. The real app never writes this buffer.
3. **Pre-pin (trace at f3caa35e4ce, RTX, nsys 2026.4.1)**: 10.0 MB/frame of unified-memory migration (5.02 MB
   HtoD, 5.00 MB DtoH), 0.39 ms/frame of copy-engine time, 240 migration rows/frame; 7.2 MB/frame of it is the
   readback buffer ping-pong, 2.6 MB/frame is UBO/vertex/staging ping-pong.
4. **The host-pin commit removed exactly the two exposed halves of the readback ping-pong**: a 0.58 ms
   copy that ran alone on the GPU at the frame boundary (median 581 us -> 75 us MEASURED) and a ~0.55 ms
   main-thread page-fault stall inside the harness's 3.6 MB memcpy (MEASURED span of the DtoH burst in frames
   200-202). 0.50 + ~0.4 = ~0.9 ms, matching the measured 0.93 ms. "Serialized dependent stream work" in the
   record is this: nothing else can run during either half.
5. **DEAD_ENDS 37's premise-collapse ("the migration exists only under nsys 2026.1.3") is not supported.**
   nsys 2026.4.1 UVM row counts on disk are 699,993 (= 7 x 99,999), 99,999 (= 1 x 99,999) and 0, while a
   clean-exiting probe has 25,420. HYPOTHESIS: UVM counter records are delivered in 99,999-record buffers and
   the final partial buffer is dropped because every harness run exits by SIGSEGV (RC=139 in every driver.log).
   `/tmp/postpin-rtx/trace.sqlite` — **post-pin, RTX, 2026.4.1** — shows **0.70-0.78 MB/frame each way of
   residual migration in five host-visible regions**, 68 rows/frame, 0.085-0.09 ms/frame of copy time, present
   in ~100% of frames. The *mechanism* refuted in 37 (global host-pin advice, +1.98 ms) stays refuted.
6. **Remaining host-visible cost post-pin (RTX heavy band)**: MEASURED components 0.085 ms copy engine +
   DtoH fault-burst spans 0.28 ms/frame mean (0.21 median) + HtoD burst spans 0.06 ms; readback copy 0.075 ms
   exposed; host memcpy ESTIMATED ~0.2 ms. The frame-start device gap that contains the host side is
   **0.995 ms median** (MEASURED, post-pin heavy band; 1.323 pre-pin). The whole topic's ceiling is therefore
   about **0.5-0.9 ms/frame on RTX**, and no single lead inside it clears 0.5 ms on its own. The two that
   could be worth ~0.2-0.4 each are (L1) deleting the readback fill in the harness (app-side, ~0.2) and
   (L5, adjacent) replacing the driver's 305 pinned HtoD arg-block copies per frame with kernel parameters
   (~0.18-0.40 at the PERFORMANCE §4 small-copy price).

## 1. Sources, windows, conventions

| item | value | source |
|---|---|---|
| Pre-pin trace | `/tmp/postdead/trace.sqlite`, nsys 2026.4.1.191, RTX 5090, driver 580.173.02, CUDA driver 13000, commit f3caa35e4ce, taken 2026-08-30 20:19 | META_DATA_EXPORT / TARGET_INFO_SYSTEM_ENV |
| Pre-pin window | internal submits [2786,3608) = 411 frames, rel 9.292-12.342 s; frame i = [submit 2786+2i, +2) | WORKFLOW §4 convention; `/tmp/postdead/submit-ts.txt` minus session start 1503511535334936 |
| Post-pin traces | `/tmp/postpin-rtx/trace.sqlite` (2026-08-31 18:59, `--delay=12`, kernels truncated at 3.5 s but UVM rows to 13.2 s = frames 1788-3268) and `/tmp/postpin-rtx2/trace.sqlite` (19:03, `--delay=14`, kernels frames 2448-3462, **zero UVM rows**) | run.sh in each dir; DIAGNOSTIC_EVENT |
| Post-pin heavy-band window used here | postpin-rtx2 submits [4900,6300) = 700 frames (frames 2450-3150), 846 kernels/frame | query in §4.4 |
| Host profile | `/tmp/favorite3-perf-current/` perf `cycles:Pu` (user only, kernel excluded), 2026-08-31 12:09 — **pre-pin** | header.txt |
| Harness | `~/favorite3-cpp/out/src/frame_0000_0000..0997.cpp` (998 files, 6,947 vkQueueSubmit, 3,473 frames), `global_var.cpp` | parse_harness.py |
| System | Linux 7.0.11, NVIDIA open kernel module 580.173.02, `uvm_disable_hmm=N`, `uvm_perf_prefetch_enable=1` threshold 51 min_faults 1, thrashing enable=1 threshold 3 lapse 500 us pin 300 ms epoch 2000 ms nap 1, `uvm_perf_map_remote_on_eviction=1`, `uvm_cpu_chunk_allocation_sizes=2166784`, CoherentGPUMemoryMode "" (PCIe, no C2C) | /proc/driver/nvidia/{version,params}, /sys/module/nvidia_uvm/parameters |

A census divided by a frame count is a MEAN; the project frame time is a hot-tail MEDIAN. Every per-frame
figure below from a trace is a mean over the stated window unless it says median.

## 2. (a) The driver's memory path — code facts

### 2.1 Memory types and allocators

| Vulkan type (cudavk) | propertyFlags (`cpvk_device.c:626-637`) | allocator (`cpvk_device_memory.c`) | host pointer |
|---|---|---|---|
| 0 `CPVK_MEM_DEVICE` | DEVICE_LOCAL | `cuMemAlloc` (`:801`) | none (Map fails, `:1020-1021`) |
| 1 `CPVK_MEM_HOST` | HOST_VISIBLE\|HOST_COHERENT\|HOST_CACHED | **`cuMemAllocManaged(..., CU_MEM_ATTACH_GLOBAL)`** (`:807`) — the enum comment at `cpvk_private.h:226` says "cuMemHostAlloc: pinned" and is **wrong** | the managed pointer (`:808-809`) |
| 2 `CPVK_MEM_MANAGED` | DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT\|HOST_CACHED | `cuMemAllocManaged` (`:812`) | the managed pointer |
| exportable (OPAQUE_FD) | any | `cuMemCreate`/`cuMemMap` VMM, device-only (`:685-741`) | none |

No `cuMemHostAlloc`, `cuMemPrefetchAsync`, or `cuMemAdvise` exists in the driver except the readback pin
(below). `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` memsets host-side (`:846-851`). The replayer remaps the
capture's memory types by property superset (`global_var.cpp:55-87`): stdout shows `1->0`, `3->1`, `4->1`,
`5->2`, i.e. the original app's VRAM, uncached sysmem, cached sysmem and BAR (device-local host-visible) types.

### 2.2 Map / Unmap / Flush / Invalidate

`cpvk_MapMemory2` returns `host_ptr + offset` (`:1008-1025`); `cpvk_UnmapMemory2`, `cpvk_FlushMappedMemoryRanges`
("every mappable type here is coherent") and `cpvk_InvalidateMappedMemoryRanges` return VK_SUCCESS and do
nothing (`:1027-1045`). Consequence: the driver has **no hook that tells it which bytes the host wrote** — the
harness makes 0 Flush calls and 0 Unmap calls (MEASURED: grep over all frame files).

### 2.3 The readback pin (commit 1acb7eabc1b)

`cpvk_BindBufferMemory2` (`:1116-1136`): for a buffer whose usage is **exactly** TRANSFER_DST, bound into a
managed allocation, `cuMemAdvise(SET_PREFERRED_LOCATION, CPU)` + `cuMemAdvise(SET_ACCESSED_BY, dev)` on the
bound range; `CUDAVK_NO_HOST_PIN_READBACK` reverts. In favorite3 this matches exactly one live buffer,
`pBuffer_327` (3,686,400 B, usage 2, in `pMemory_6991`; MEASURED from the harness) plus `pBuffer_89` (1 B) and
`pBuffer_13931` (921,600 B) in `pMemory_3913`. Record: `docs/cudavk/history/perf-2026-08-31-host-pin-readback.md`
— RTX favorite3 7.6048 -> 6.6757 (-0.929), favorite2 -1.066; B200 -1.399 / -1.590 (MEASURED there).

### 2.4 How copy commands execute (`cpvk_cmd.c`)

| Vulkan command | record | execute (`cpvk_execute_copy`, `cpvk_cmd.c:4306-4489`) | CUPTI signature |
|---|---|---|---|
| `vkCmdCopyBuffer2` | `cpvk_copy{src,dst}` raw device addresses (`:3158-3182`) | `cuMemcpyDtoDAsync` on the main stream (`:4351`) | copyKind 8, srcKind 4 (Managed) -> dstKind 2 (Device) for staging->device |
| `vkCmdUpdateBuffer` | **`cuMemAlloc` + blocking `cuMemcpyHtoD` at record time** into a staging block, then a copy op (`:3110-3155`) | `cuMemcpyDtoDAsync` | harness never calls it (0) |
| `vkCmdCopyBufferToImage2` | per layer/slice pitched copy (`:3692-3768`) | `cuMemcpy2DAsync` with both ends `CU_MEMORYTYPE_DEVICE` (`:4457-4469`) | 8/4->2; a 32x32x16 slice upload = 16 x 2,048 B rows (MEASURED 16.1/frame) |
| `vkCmdCopyImageToBuffer2` | pitched copy (`:3771-`) | `cuMemcpy2DAsync` | **8/2->4 (Device -> Managed)**: the 3,686,400 B readback, 1.0/frame |
| `vkCmdCopyImage` | pitched copy | `cuMemcpy2DAsync` | 8/2->2 |
| `vkCmdFillBuffer` | op | `cuMemsetD32Async` (`cpvk_device_memory.c:306`) | memset |
| MSAA resolve / scaled blit | kernels `resolve_samples`, `blit_linear` (`:4355-4440`) | kernel | — |

Nothing on the copy path is a kernel except resolve/blit. All copies are stream-ordered on the main stream after
`cp_batch_flush`.

### 2.5 How the GPU consumes application buffers

- UBO/SSBO descriptors store `buffer->mem->dev_ptr + buffer->offset + bi->offset` (`cpvk_cmd.c:329-337`);
  snapshots copy the set into the descriptor arena (`:1135-1168`) which is `cuMemAlloc` device memory with a
  `malloc` host mirror uploaded once per submit by blocking `cuMemcpyHtoD` (`cpvk_device_memory.c:242-259`,
  `cpvk_cmd.c:1030-1037`). **The descriptor rows point straight into the application's managed memory.**
- Vertex buffers: `cmd->vb_base[b] = mem->dev_ptr + offset` (`:1995-1996`); index buffer: `index_ptr` likewise
  (`:2048-2050`) and is only ever passed to kernels (`cp_renderer.c:5234`), never dereferenced on the host.
- So every UBO load, vertex fetch and index fetch of host-visible data is a **GPU read of a managed page**; if
  the page is CPU-resident (the CPU wrote it since the GPU last touched it) the read **faults and migrates**.

### 2.6 Driver-internal memory that is host-visible in the CUDA sense

| storage | allocator | host writes | GPU reads | per-frame behaviour |
|---|---|---|---|---|
| upload arena staging `cp->upload_host` 8 MB | `cuMemAllocHost` pinned (`cp_renderer.c:135`) -> `cuMemcpyHtoDAsync` to `arena_base` device (`:464-520`) | every arg block | device copy | 140.6 copies/frame (window) / **304.7/frame heavy band**, 0.24-0.47 MB/frame; **no UVM involvement** |
| descriptor arena | `cuMemAlloc` + `malloc` mirror (`cpvk_cmd.c:1030-1037`) | snapshots | shaders | 1 blocking `cuMemcpyHtoD` per submit, 63-117 KB, 9.6-12 us host |
| `dev->null_data` 1 MB, `dev->null_desc` 64 KB | `cuMemAllocManaged` (`cpvk_device_memory.c:521-534`) | once at init, **and `cp_host_ptr` dereferences any unmapped address on the host (`cp_renderer.c:3188-3200`)** | shaders (null bindings) | HYPOTHESIS: the 1-page ping-pong seen every frame (§4.2 region `0x6ffeb0000000` / `0x7bb9fc000000`) |
| `view->tex_info` (per image view) | `cuMemAllocManaged` (`cpvk_image.c:595`) | `cpvk_image_view_refresh` at create/bind (`:490-519`) | sampler path | 30 views per 285 heavy frames -> negligible |
| `cp->sampler_table` | `cuMemAllocManaged` (`cpvk_image.c:793`) | on new sampler only | shaders | 0 samplers created in the heavy band |
| `cp_scratch_alloc` managed arena (`cp_renderer.c:247-307`) | `cuMemAllocManaged` | — | — | **no callers** in the tree (grep) — dead code; all scratch goes through `cp_scratch_alloc_device` |
| query pools | `calloc` (`cpvk_cmd.c:4511-4512`), timestamps via `cuLaunchHostFunc` (3/frame) | host | — | no UVM |

## 3. (b) The application's host-visible memory (harness census, MEASURED unless noted)

### 3.1 Allocations (194 total)

| original type -> cudavk type | count | bytes | meaning on the original driver |
|---|---|---|---|
| 1 -> 0 (device) | 122 | 844,886,016 | VRAM |
| 3 -> 1 (managed) | 67 | 181,813,248 | host-visible coherent sysmem |
| 4 -> 1 (managed) | 4 | 13,156,352 | host-visible cached sysmem (incl. the readback buffer) |
| 5 -> 2 (managed) | 1 | 2,097,152 | **DEVICE_LOCAL\|HOST_VISIBLE (BAR)** — the UBO ring `pMemory_7086` |

All 72 host-visible allocations are mapped once at creation and never unmapped (72 `vkMapMemory`, 0
`vkUnmapMemory`, 0 `vkFlushMappedMemoryRanges`). Two more 3,686,400-B type-4 readback allocations appear
later (`pMemory_1402735`, `pMemory_1613662`), each bound to one TRANSFER_DST buffer (pinned by the commit).

Key host-visible allocations and what is bound in them (parse_harness.py; usage bits: 1 TRANSFER_SRC,
2 TRANSFER_DST, 16 UNIFORM, 64 INDEX, 128 VERTEX):

| allocation | size | cudavk type | buffers bound over the run | usages | role |
|---|---|---|---|---|---|
| `pMemory_6991` | 3,686,400 | 1 | 1 (`pBuffer_327`) | 2 | readback target of `vkCmdCopyImageToBuffer`, 1/frame — **pinned by 1acb7eabc1b** |
| `pMemory_7086` | 2,097,152 | 2 | 2 (`pBuffer_339`, `pBuffer_421`, 524,288 B each) | 16 | double-buffered UBO ring; **210.5 descriptor writes/frame** reference it (range 16,384), alternating buffers per frame |
| `pMemory_493011` / `_493016` | 4,096,000 each | 1 | 1 each (`pBuffer_982`, `_983`) | 128 | dynamic vertex buffers |
| `pMemory_3285` | 2,097,152 | 1 | 6,673 (created/destroyed per frame, ~2/frame) | 1,16,64,128 | texture-upload staging (source of 0.99 `vkCmdCopyBufferToImage`/frame and 0.31 `vkCmdCopyBuffer`/frame), plus **2 index-buffer binds/frame and 5.1 UBO descriptor writes/frame** |
| `pMemory_6508` | 2,097,152 | 1 | 1,594 | 1,16,64,128 | 2.0 descriptor writes/frame (ranges 336 / 9,792) |
| `pMemory_10214`, `_11385`, `_13264`, `_489305`, `_4991xx..._5025xx`, `_11858xx..._11966xx` | 2 MB each | 1 | 7-829 each | 1 | texture streaming staging (allocated in bursts, e.g. 24 x 2 MB around frames 2820-2850) |
| `pMemory_13254`, `_559706`, `_758319`, `_556704` | 14.7 / 16.8 / 8.4 / 4.2 MB | 1 | 1 each | 1 | static staging, written once |

### 3.2 Per-frame host writes (`LoadBinaryData` = plain `memcpy` from an mmapped pack, `global_var.cpp:235-252`)

Heavy band frames 2150-3150 (1,000 frames): **mean 4,212,179 B/frame, median 4,202,496 B/frame, 8.7 calls/frame.**

| destination (ppData -> allocation) | bytes/frame | cudavk type | consumer |
|---|---|---|---|
| `ppData_6992` -> `pMemory_6991` (readback) | **3,686,400** | 1 (host-pinned) | **nobody** — the GPU copy overwrites all 3,686,400 B before anything reads it; capture artefact |
| `ppData_7087` -> `pMemory_7086` (UBO ring) | 307,188 | 2 | shaders, via 210.5 descriptor rows/frame |
| `ppData_493017` / `_493012` -> vertex buffers | 76,018 + 76,001 | 1 | fused vertex fetch |
| `ppData_3286` -> `pMemory_3285` (staging/index/UBO) | 57,508 | 1 | copy engine (b2i), 2 index binds, 5 UBO descriptors |
| `ppData_6509`, `_10215`, `_11386`, `_5001` | 5,169 + 2,097 + 1,049 + 750 | 1 | UBO descriptors (6508), staging |

Light band 1391-2150: 4,286,100 B/frame mean (readback 3,671,829; UBO 204,120; staging 135,146; ...).
Whole run: 14,189,218,336 B in 25,808 calls; the readback fill is 12,788,121,600 B (3,469 calls).

Per-frame copy commands in the heavy band: `vkCmdCopyImageToBuffer` 1.00 (dst `pBuffer_327`), `vkCmdCopyBufferToImage`
0.99 (src type 1), `vkCmdCopyBuffer` 0.31 (type 1 -> type 0), `vkCmdCopyImage` ~1.3; `vkCreateBuffer`/`vkBindBufferMemory`
~2/frame (staging), `vkDestroyBuffer` ~2/frame, `vkCreateImage`/`vkCreateImageView` 0.1/frame, `vkAllocateDescriptorSets`
95/frame, `vkUpdateDescriptorSetWithTemplate` 122/frame, draws 325/frame, 0 indirect draws, 0 `vkCmdUpdateBuffer`,
0 `vkCmdPushConstants`.

### 3.3 Frame structure (MEASURED from the frame files; fully serial)

```
LoadBinaryData x4-6  (UBO ring half, readback fill 3.6 MB, staging)   <- main thread, GPU idle
vkQueueSubmit(render cmd buffers, fence A)                            <- driver issues ~700-850 launches, blocks ~17x
vkWaitForFences(fence of frame N-1)                                   <- already complete
vkDestroyBuffer x2 ; vkGetQueryPoolResults (N-1) ; record copy cmd buffer
LoadBinaryData x2  (8-24 KB staging)
vkQueueSubmit(copy-image-to-buffer, fence C) ; vkWaitForFences(fence C)   <- waits for ALL of frame N + the copy
vkCreateBuffer/vkBindBufferMemory x2 ; record frame N+1 (95 set allocs, 122 template updates, 325 draws)
```
The readback copy is therefore the last GPU operation of a frame and the frame-start gap after it is pure host
time: fence wake + destroy/create + record (0.38 ms app record/decode per PERF_HANDOFF) + 4.2 MB of memcpy with
whatever page faults the destination pages cost + submit translation to the first launch.

## 4. (c) What happens to a managed page written by the CPU and read by the GPU

### 4.1 Mechanism on this system (MEASURED settings, standard UVM behaviour)

- A managed page is resident on exactly one side. A **CPU write** to a GPU-resident page takes a CPU page fault
  serviced by `nvidia_uvm`: the page migrates DtoH (`migrationCause` 2 = COHERENCE), and with
  `uvm_perf_prefetch_enable=1` (threshold 51%, min_faults 1) neighbouring pages of the same 2 MB VA block are
  migrated too (cause 3 = PREFETCH, rows of 8 KB-1 MB). The CPU thread is stalled for the whole service.
- A **GPU read** of a CPU-resident page is a replayable GPU fault: the SM's warps stall, the fault buffer is
  drained by the UVM driver, the page (+ prefetch) migrates HtoD, the access is replayed. The kernel's duration
  grows by the service time.
- Thrashing mitigation (`uvm_perf_thrashing_lapse_usec=500`, threshold 3) never engages here: the two sides
  alternate once per frame (>= 6 ms apart), so the pages ping-pong **every frame forever** (MEASURED: UBO
  region present in 410/411 pre-pin frames and 450/455 post-pin frames).
- `SET_PREFERRED_LOCATION CPU + SET_ACCESSED_BY GPU` (the pin) makes the GPU map the page remotely instead of
  faulting: GPU accesses become PCIe reads/writes with no migration; CPU accesses are plain memory accesses.
  There is **no CPU-side equivalent** on PCIe: a CPU cannot map GPU-resident managed pages, so a GPU-preferred
  page written by the CPU always migrates. HMM is enabled (`uvm_disable_hmm=N`) but only affects system-allocated
  memory, not `cuMemAllocManaged`.
- B200 host (Xeon 8559C, PCIe Gen5, no C2C): the same mechanism. Documented differences (record, MEASURED
  there, nsys 2026.1.3): pre-pin readback copy 1.6-1.8 ms at 2.2 GB/s vs RTX 0.58 ms at 6.3 GB/s; post-pin DtoD
  0.323 ms/frame (11.4 GB/s) vs RTX 0.075 ms (49 GB/s); UVM copy time 0.98 -> 0.24 ms/frame vs RTX 0.39 -> 0.085.
  The B200's host-visible path costs ~3x per byte; the same leads are worth ~2-3x more there (HYPOTHESIS).

### 4.2 Pre-pin migration census (window [2786,3608), 411 frames)

```sql
-- per kind in the window
select copyKind,srcKind,dstKind,count(*),sum(bytes),sum(end-start)/1e6
from CUPTI_ACTIVITY_KIND_MEMCPY where start>=:t0 and start<:t1 group by 1,2,3;
-- per 2 MB VA region and cause
select (virtualAddress>>21)<<21, copyKind, migrationCause, count(*), sum(bytes)
from CUPTI_ACTIVITY_KIND_MEMCPY where start>=:t0 and start<:t1 and copyKind in (11,12) group by 1,2,3;
```
(t0 = submit-ts[2786] - 1503511535334936 = 9.292120249 s, t1 = 12.342282577 s)

| copy class | /frame | MB/frame | copy ms/frame | median us |
|---|---:|---:|---:|---:|
| readback `vkCmdCopyImageToBuffer` (8, Device->Managed) | 1.00 | 3.686 | **0.6155** | 580.7 (min 173, max 1165) |
| UVM HtoD (11) | 170.9 | 5.024 | 0.2684 | 0.99 |
| UVM DtoH (12) | 69.1 | 5.005 | 0.1213 | 0.67 |
| staging->device (8, Managed->Device) | 27.1 | 0.322 | 0.1391 | 0.61 |
| driver upload pinned HtoD (1, Pinned->Device) | 140.6 | 0.243 | 0.1304 | 0.77 |
| texture cache device->array (7) | 5.1 | 10.357 | 0.0228 | 4.8 |
| device->device (8,2,2) | 18.3 | 0.055 | 0.0115 | 0.61 |
| small readbacks (2, Device->Pageable) | 8.9 | 0.000 | 0.0041 | 0.35 |
| descriptor arena HtoD (1, Pageable->Device) | 1.0 | 0.063 | 0.0029 | 2.9 |
| memsets | 104.0 | 315 | 0.1431 | — |

UVM split (readback regions `0x6ffeb0c00000`-`0x6ffeb1000000` vs everything else):

| | HtoD KB/f | HtoD ms/f | HtoD fault rows/f | DtoH KB/f | DtoH ms/f | DtoH fault rows/f |
|---|---:|---:|---:|---:|---:|---:|
| readback buffer | 3,591 | 0.2172 | 74.6 | 3,591 | 0.0761 | 15.0 |
| other host-visible regions | 1,315 | 0.0512 | 9.7 | 1,297 | 0.0452 | 22.4 |

Causes: HtoD COHERENCE 34,633 rows / 428 MB, PREFETCH 35,597 rows / 1.64 GB; DtoH COHERENCE 15,352 / 63 MB,
PREFETCH 13,034 / 1.99 GB. Row sizes: 4 KB dominates by count (31,073 HtoD, 16,000 DtoH), prefetch rows up to 1 MB.

Regions (per frame, MEASURED; identities HYPOTHESIS from size, write pattern and timing):

| region | KB/frame each way | frames present | pattern | identity |
|---|---:|---:|---|---|
| `0x6ffeb0c00000` + `0x6ffeb0e00000` | 2,043 + 1,548 = 3,591 | 410/411 | GPU copy faults 74.6x then prefetch; CPU fill faults 15x | readback `pMemory_6991` (3,686,400 B spans two regions) — **certain** |
| `0x6ffeb0200000` | 499 | 411 | 7 CPU 4 KB faults before the render submit; copy-engine fault + 100 KB prefetch at first b2i; 2 more CPU faults after submit | `pMemory_3285` staging/index/UBO (98 KB written) |
| `0x6ffeb1000000` | 392 | 411 | **1 GPU fault + one 258-388 KB prefetch per frame; 3.5 CPU faults + ~108 KB prefetch each** | UBO ring `pMemory_7086` (307 KB written) |
| `0x6ffe03800000`, `0x6ffe03c00000` | 51 each | 206/411 | paired 4 MB | vertex buffers `pMemory_493011/493016` |
| `0x6ffeb1200000` | 61 | 411 | **6 CPU faults + 1 GPU fault mid-submit** (4.35-4.77 ms in frame 200, between the drain readback and the next launch) | unidentified — driver-side host access during submit (see §7) |
| `0x6ffeb0800000`, `0x6ffeb0600000`, `0x6ffeb1400000`, `0x6ffeb0a00000`, `0x6ffeb1600000` | 87 / 57 / 57 / 22 / 12 | 25-109 | episodic | 2 MB staging allocations |
| `0x6ffeb0000000` | 4 | 249/411 | one page each way | HYPOTHESIS `dev->null_desc` via `cp_host_ptr` fallback |
| `0x6ffe02000000..02600000` | 2 MB once | 1 | one-time | 14.7 MB static staging first read |

Intra-frame timeline, frame 200 of the window (MEASURED, ms from frame start; frame 6.237 ms):
readback copy 0.014-0.520 with 160 HtoD rows (GPU-side faults of the copy engine) 0.064-0.518; **CPU fill of the
readback buffer 1.020-1.574 (30 DtoH rows, 3,686,400 B) — a 0.55 ms main-thread stall**; other CPU faults
0.818-0.995 (UBO, staging, VB); render submit at 1.684; kernels 1.841-6.168; GPU faults on staging/UBO/VB
1.726-1.956 (first kernels); the unidentified CPU faults 4.354-4.688 and a GPU fault at 4.769; staging writes
6.208-6.229 after the kernels. Frames 201/202: fill stall 0.541 / 0.553 ms. Device gap copy-end -> first
kernel: median **1.323 ms**, mean 1.736 (410 frames).

### 4.3 The nsys 2026.4.1 "zero UVM rows" finding

| trace | exit | UVM rows | = n x 99,999? |
|---|---|---:|---|
| `/tmp/postdead` (pre-pin) | SIGSEGV at teardown | 699,993 | **7 x 99,999** |
| `/tmp/postpin-rtx` (post-pin, 12 s delay) | SIGSEGV | 99,999 | **1 x 99,999** (rows stop at 13.18 s exactly when the buffer filled; kernel rows stopped at 3.5 s) |
| `/tmp/postpin-rtx2` (post-pin, 14 s delay, 1,014 frames) | SIGSEGV | **0** | 0 x 99,999 (estimated 68 rows/frame x 1,014 = 69K < 99,999) |
| `/tmp/uvm-probe/pp.sqlite` (positive control, clean `return 0`) | clean | 25,420 | not a multiple — teardown flush happened |

DIAGNOSTIC_EVENT in every replay trace: "Buffers holding CUDA trace data will be flushed on CudaProfilerStop()
call" and "Not all CUDA events might have been collected". All replay runs exit with RC=139 (`/tmp/pin-rtx-ab/driver.log`,
`/tmp/postpin-rtx*/driver.log`, `/tmp/favorite3-perf-current/driver.log`).
**Conclusion (MEASURED counts, HYPOTHESIS mechanism):** the absence of UVM rows in postpin-rtx2 — the evidence
DEAD_ENDS 37 and PERF_HANDOFF rule 2 rest on — is a lost partial buffer, not absence of migration. The 2026.1.3
B200 figure (1.9 MB/frame in five regions, ~940 KB each way) is the same phenomenon this RTX post-pin trace shows.

### 4.4 Post-pin residual migration (RTX, `/tmp/postpin-rtx`, frames covered by UVM rows: submits 3576-6536 = 1,480 frames)

| region | HtoD KB/f (fault rows/f) | DtoH KB/f (fault rows/f) | frames present | identity (HYPOTHESIS) |
|---|---|---|---|---|
| `0x7bb9fd000000` | 260 (1.0 fault + 1.05 prefetch of 256 KB) | 260 (11.1 faults + 3.3 prefetch) | 450/455 | UBO ring `pMemory_7086` |
| `0x7bb9fc200000` | 172 (4.9 faults) | 172 (3.3 faults) | 451/455 | `pMemory_3285` staging/index/UBO |
| `0x7bb94fc00000`, `0x7bb94f800000` | 92 each (3.7 faults each) | 92 each (1.4 faults each) | 226/455 | vertex buffers |
| `0x7bb9fd200000` | 61 (1.55 faults) | 61 (7.65 faults) | 450/455 | unidentified (same as pre-pin `..1200000`) |
| `0x7bb9fc000000` | 3.5 (0.86 fault) | 3.5 (0.86 fault) | 238/455 | null-desc page |
| **total** | **729 KB, 29.6 rows, 16.1 fault rows, 0.050 ms copy** | **705 KB, 38.0 rows, 25.8 fault rows, 0.035 ms copy** | | |
| heavy-band subset (frames 2150-3150) | 779 KB, 34.0 rows, 18.1 faults, 0.052 ms | 779 KB, 40.5 rows, 27.6 faults, 0.038 ms | | |

Burst analysis (rows within 20 us of each other merged; span = first start to last end; a lower bound on the
stall the faulting side sees, since it excludes fault entry/exit): DtoH (CPU side) **8.2 bursts/frame, span
0.278 ms/frame mean, 0.210 median, p90 0.574**; HtoD (GPU side) 7.7 bursts, span 0.061 ms mean. Pre-pin
non-readback: 10.7 CPU bursts 0.219 ms; 7.3 GPU bursts 0.052 ms.

Post-pin heavy band (`/tmp/postpin-rtx2`, submits [4900,6300), 700 frames, 846 kernels/frame) memcpy census:

| class | /frame | MB/frame | ms/frame | median us |
|---|---:|---:|---:|---:|
| driver upload pinned HtoD | **304.7** | 0.473 | 0.261 | 0.7 |
| texture cache device->array | 7.0 | 98.0 | 0.184 | 7.3 |
| readback copy (device -> host-pinned managed) | 1.0 | 3.686 | **0.0764** (median 75.2 us = **49 GB/s**, min 72, max 104) | |
| staging -> device | 16.1 | 0.050 | 0.066 | 0.6 |
| device -> device | 16.2 | 0.035 | 0.010 | |
| descriptor arena HtoD (pageable) | 1.0 | 0.117 | 0.004 | 3.8 |
| small readbacks | 7.65 | 0 | 0.0035 | |
| host API: `cuMemcpyHtoDAsync` 304.7/f **0.521 ms/f**, `cuMemcpy2DAsync` 32.9/f 0.098, `cuMemcpyDtoH` 7.65/f 0.059, `cuMemcpyHtoD` 1/f 0.012 (CUPTI-inflated) | | | | |

Frame-boundary gaps post-pin (MEASURED): last kernel end -> readback copy start median 0.057 ms; **readback copy
end -> next kernel start median 0.995 ms, mean 1.327, p10 0.824**. Pre-pin equivalent: 1.323 median.

## 5. (d) Deliverables

### 5.1 Per-frame host-visible traffic, post-pin, RTX heavy band

| traffic | bytes/frame | direction | mechanism | who pays | ms/frame |
|---|---:|---|---|---|---|
| readback fill (capture artefact) | 3,686,400 | host -> host-resident pinned pages | `memcpy` in `LoadBinaryData` | main thread, inside the frame-start gap (critical path) | ESTIMATED 0.15-0.22: perf `__memmove_avx512_unaligned_erms` = 4.99% of user cycles pre-pin = 0.26 ms/frame of all memmove, harness bytes are 90% of memcpy'd bytes; 3.686 MB at 17-25 GB/s |
| readback copy | 3,686,400 | device -> host | copy engine over PCIe (`cuMemcpy2DAsync`) | GPU, alone, exposed at frame end | MEASURED 0.075 (was 0.58) |
| UBO ring writes | 307,188 | host -> managed, then GPU faults it back | CPU faults 11/f + prefetch; GPU fault 1/f + 256 KB prefetch | main thread + first UBO-reading kernel | copy MEASURED 0.021; stall HYPOTHESIS 0.1-0.3 |
| vertex buffer writes | 152,019 | same | CPU faults 2.8/f; GPU faults 7.4/f | main thread + vertex kernels | copy 0.012; stall HYPOTHESIS 0.05-0.3 |
| staging + index + small UBOs (`pMemory_3285`, `_6508`, ...) | ~66,600 | same | CPU faults 3.3/f; GPU/copy-engine faults 4.9/f | main thread + copy engine + shaders | copy 0.019; stall HYPOTHESIS 0.05-0.2 |
| unidentified region | ~61,000 each way | host access **during submit** | 7.65 CPU faults + 1.55 GPU faults | main thread inside `vkQueueSubmit` (critical path) | copy 0.009; stall HYPOTHESIS 0.05-0.2 |
| null-desc page | 4,096 each way | `cp_host_ptr` host read of a GPU-resident managed page | 1 CPU + 1 GPU fault | main thread + shader | HYPOTHESIS 0.02-0.06 |
| **all residual UVM** | **0.78 MB each way** | | 34 + 40.5 rows, 18 + 28 fault rows | | **copy MEASURED 0.090; burst spans MEASURED 0.28 (CPU) + 0.06 (GPU); true stall HYPOTHESIS 0.3-0.7** |
| texture staging -> images | ~50,000 | managed -> device | 16 pitched copies | copy engine | MEASURED 0.066 |
| driver arg blocks (not app-visible) | 473,000 | pinned host -> device | 304.7 `cuMemcpyHtoDAsync` | main thread 0.52 ms API (CUPTI-inflated; ~0.3 without) + copy engine 0.26 | MEASURED |
| descriptor arena | 117,000 | pageable -> device | 1 blocking `cuMemcpyHtoD` | main thread 12 us | MEASURED |

Upper bound on the whole topic: the post-pin frame-start gap is 0.995 ms median and contains the app's 0.38 ms
record/decode plus fence wake and submit translation, so the host-side memory costs inside it are at most
~0.5 ms; adding the mid-submit faults (~0.1-0.2) and GPU fault stalls (~0.1-0.3, partly overlapped) gives a
**topic ceiling of about 0.5-0.9 ms/frame on RTX** (ESTIMATED), spread over several independent items.

### 5.2 What the host-pin commit taught

- Pre-pin the readback ping-pong had **two halves, both fully exposed**: the copy (0.58 ms, the only thing on the
  GPU) and the CPU fill's fault stall (~0.55 ms of a single-threaded critical path). Post-pin: 0.075 + ~0.2. The
  0.93 ms RTX win is that arithmetic; "serialized dependent stream work" is the record's name for the same fact.
- The rule it encodes — **pin what the GPU only writes and the CPU only reads** (or, here, writes) — is the one
  case where remote GPU access is free of latency exposure (a DMA write stream). It says nothing about
  GPU-*read* data, and DEAD_ENDS 37 measured that pinning GPU-read data globally costs +1.98 ms on RTX.
- The B200 gained more (1.40) because its migration path is ~3x slower per byte, not because its GPU is slower.

### 5.3 Placement policy for per-frame CPU-written data: what decides it

The decision variable is the **GPU read pattern of the page after the CPU writes it**:

| data | CPU writes/frame | GPU reads | host-pinned + remote reads | device + explicit upload | managed (today) |
|---|---|---|---|---|---|
| UBO ring (type 2, 307 KB, 210 descriptors/frame read by every shader invocation) | dense | latency-bound, every invocation | **bad** (each UBO load a PCIe round trip; this is most of 37's +1.98 HYPOTHESIS) | **good**: 512 KB-2 MB DMA per submit (0.02-0.08 ms async), zero faults; needs a pinned shadow because there is no dirty tracking (§2.2) | 12 faults/frame |
| dynamic vertex buffers (type 1, 152 KB, read once per vertex by the fused fetch) | sparse (76 KB of 4 MB each) | streaming, once | acceptable (the original app placed them in sysmem) but a 4 MB allocation cannot be re-uploaded whole | full re-upload 8 MB/frame = 0.3 ms — worse than the faults | 10 faults/frame |
| texture staging (type 1) | ~60 KB | copy engine, once | **good** (that is what staging is) | pointless | 8 faults/frame |
| readback (type 1, TRANSFER_DST-only) | (artefact) | copy engine writes | **done** (1acb7eabc1b) | — | — |

So no single policy fits; the pin commit's usage-bit dispatch is the right shape and needs two more branches:
usage == UNIFORM in type 2 -> shadow+upload; TRANSFER_SRC-only in type 1 -> host-pinned. Vertex buffers are the
hard case and the smallest.

### 5.4 Leads (each checked against DEAD_ENDS 1-38)

**L1 — Delete the readback fill in the harness (app-side).** Mechanism: skip the 3,686,400-B `LoadBinaryData`
into `ppData_6992` (and `_1402736`, `_1613663`); the GPU copy overwrites the full buffer every frame so dumps
and sentinels cannot change. Ceiling ESTIMATED 0.15-0.22 ms/frame RTX (§5.1), 0.25-0.35 on B200 (Xeon
single-thread memcpy ~10-15 GB/s, HYPOTHESIS); 1:1 on the frame because it sits in the serial frame-start gap.
Closest dead end: 37 / the pin commit — different: this removes a host memcpy that the real application never
performs, not a migration. Probe: `sed` the three LoadBinaryData lines out of `~/favorite3-cpp/out/src/*.cpp`
(998 files), rebuild the harness, one alternating A/B with hash + sentinel check. Effort: hours. Caveat: it
changes the application side of the benchmark and must be declared as such; it also makes the replay *more*
faithful.

**L2 — Type-2 allocations as device memory + pinned shadow uploaded at submit.** Mechanism: `cuMemAlloc` +
`cuMemAllocHost` shadow returned by `vkMapMemory`; at `vkQueueSubmit`, `cuMemcpyHtoDAsync` of the ranges the
submitted command buffers' descriptors reference (the driver sees them in `cpvk_write_descriptor`), falling
back to the whole allocation (2 MB, 0.08 ms of async DMA). Removes 11 CPU faults + 1 GPU fault + 0.02 ms copy
per frame; shaders keep reading device memory. Ceiling HYPOTHESIS 0.1-0.3 ms RTX, 0.2-0.6 B200. Closest dead
end: 37 (global advice) — different because it never makes a GPU read remote and it removes the CPU faults by
construction. Cheapest probe (1 flag, 1 A/B): apply the reverted `CUDAVK_HOSTMEM_ADVICE=2` logic to
**type-2 allocations only**; if that alone costs frame time it measures the remote-read penalty of the UBO ring
and confirms the shadow design is the only shape. Effort: medium (~150 lines across `cpvk_device_memory.c`,
`cpvk_cmd.c` descriptor path, submit).

**L3 — `cuMemPrefetchAsync` of bound host-visible allocations at submit.** Mechanism: before translating a
submit, prefetch to the device every type-1/2 allocation whose buffers are bound by descriptors / vertex /
index binds in that submit (~5 calls/frame). Turns 16-18 GPU fault rows/frame into one stream-ordered bulk
migration per allocation; leaves the CPU-side faults untouched. Ceiling HYPOTHESIS 0.1-0.3 ms RTX (GPU fault
service is 20-50 us each and partly overlapped). Closest dead end: 37 — different mechanism (a one-shot
migration, not a residency policy; GPU reads stay local). Probe: one flag, one A/B. Effort: small.

**L4 — Stop `cp_host_ptr` from touching managed memory.** `cp_renderer.c:3188-3200` dereferences any address
not in a host map as a host pointer; when that is `dev->null_desc` (managed, GPU-resident) it costs a CPU fault
and then a GPU fault every frame (HYPOTHESIS for the 1-page region; MEASURED that a 1-page ping-pong exists
0.86x/frame). Keep a `malloc` mirror of the null page and return it. Ceiling HYPOTHESIS 0.02-0.06 ms. Probe:
`perf record -e page-faults -g` on a replay and filter faulting addresses against the managed VA range, or
nsys `--cuda-um-cpu-page-faults=true` (also identifies the unidentified region). Effort: an hour.

**L5 (adjacent, driver-internal HtoD) — Arg blocks as kernel parameters instead of pinned HtoD copies.**
MEASURED: 304.7 `cuMemcpyHtoDAsync`/frame in the heavy band (140.6 in the lighter window), 0.52 ms/frame of
CUPTI-inflated host API time, 0.26 ms copy-engine time; block sizes max 31,056 B, p99 11,408, p50 912,
6.2% > 4,096 B, **none > 32,764 B** (the CUDA >= 12.1 kernel-parameter limit; the build is CUDA 12.8 on driver 580).
Mechanism: pass each arg block as the launch's parameter buffer; the CUDA driver copies it into the pushbuffer
with the launch, so the separate DMA and its API call disappear, and `cp_launch_after`'s PDL condition (3) —
"nothing else issued since, in particular not the coalesced upload flush" — is met more often. Ceiling
ESTIMATED: 304.7 x 1.30 us (PERFORMANCE §4 "small copy removed outright") = **0.40 ms/frame heavy band**
(0.18 in the lighter window); net of the larger launch payload (~0.1-0.5 us per launch) 0.25-0.40. Closest
dead end: 8 (smarter flush placement, REFUTED) — different: 8's retry condition is literally "uploads stop being
owned by the launch boundary — a different ownership model", which this is; also DEAD_ENDS 1 (graphs) is
unrelated. Effort: high — the shader ABI (`cp_shader_abi.h`, `nir_to_ptx`) reads the block through a pointer and
the block contains device addresses of tables placed behind it in the same block (`cp_renderer.c:379-390`);
those become param-space offsets. Probe before building: a micro-benchmark of `cuLaunchKernel` cost vs
parameter size (1-31 KB) on sm_120, and a count of launches whose block exceeds 4 KB per frame.

Not proposed: global residency advice (37, refuted), read-mostly duplication (a CPU write invalidates the copy
every frame — same ping-pong), managed-to-HMM (irrelevant to `cuMemAllocManaged`), dirty tracking via
userfaultfd/mprotect (75 write-protect traps per frame at 2-5 us each is the same order as the faults it
replaces).

## 6. Queries and derivations

All SQL ran against `sqlite3.connect('file:...?mode=ro', uri=True)`. Frame boundaries: `subs[i] = int(line.split()[0]) -
TARGET_INFO_SESSION_START_TIME[3]` from the trace's ts/submit-ts file; frame f of a window starting at submit s0 is
`[subs[s0+2f], subs[s0+2f+2])`.

- Per-kind census: §4.2 SQL. Median: `select (end-start) from CUPTI_ACTIVITY_KIND_MEMCPY where ... order by 1 limit 1 offset n/2`.
- Region/cause census: §4.2 second SQL; region = `(virtualAddress>>21)<<21`.
- Readback vs other split: `virtualAddress>=0x6ffeb0c00000 and virtualAddress<0x6ffeb1000000`.
- Bursts: rows of one kind in one frame sorted by start, merged when `start - prev_end <= 20000 ns`; span = sum of (last end - first start).
- Frame-start gap: for each readback copy (`copyKind=8 and srcKind=2 and dstKind=4`), the first `CUPTI_ACTIVITY_KIND_KERNEL.start > copy.end`.
- Host API census: `join StringIds s on s.id=r.nameId` over `CUPTI_ACTIVITY_KIND_RUNTIME` in the window, grouped by name.
- 99,999 quantum: `select copyKind,count(*) ... where copyKind in (11,12)` per trace; 539,256 + 160,737 = 699,993 = 7 x 99,999; 43,825 + 56,174 = 99,999.
- Harness: `parse_harness.py` (allocations, binds, maps, LoadBinaryData per frame, copies) and `parse_consumers.py` (descriptor buffer infos, index/vertex binds per frame, multiline-aware) — outputs in `harness_census.json`.
- memcpy share: `flat.txt` 4.99% for `__memmove_avx512_unaligned_erms` of 8.6875e10 user cycles over 3,473 frames at ~4.8 GHz -> ~0.26 ms/frame; harness writes 14.19 GB of the ~15.8 GB memcpy'd (arg blocks 0.24-0.47 MB/frame, descriptor mirrors ~0.1 MB/frame) -> ~90%.
- Pin arithmetic: copy 0.581 -> 0.075 (-0.506) + fill stall ~0.55 -> ~0.15-0.2 (-0.35..-0.40) = -0.86..-0.91 vs measured -0.929.

## 7. Unknowns

1. Identity of the mid-submit region (`0x6ffeb1200000` / `0x7bb9fd200000`): 6-8 CPU faults per frame between
   the episode-drain readback (`cuStreamSynchronize` + `cuMemcpyDtoH`) and the next launch, 61 KB each way. No
   driver code path found that writes managed memory there (`cp_scratch_alloc` is dead, A-buffer mirrors are
   `malloc`, index buffers are never dereferenced on the host). The nsys UVM CPU-fault trace or `perf -e page-faults -g`
   resolves it in one run.
2. True stall per fault on this system (GPU replayable fault service; CPU fault service). Only the migration
   copy spans are measured (0.28 + 0.06 ms/frame); the fault entry/exit and driver locking are not.
3. The B200 per-frame residual migration under a trace that actually flushes; the 2026.1.3 figure (0.242 ms
   copy, 1.9 MB) is the only one and its collection differs.
4. Whether the frame-start gap's non-memory content (record 0.38 ms, fence wake, translation) leaves the
   memory items on the critical path in every frame; the gap is 0.995 ms median, so the sum of items fits, but
   the split is estimated.
5. Kernel-parameter launch cost vs size on sm_120 (needed to price L5 net).
