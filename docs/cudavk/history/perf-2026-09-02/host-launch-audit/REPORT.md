# Host launch/issue path audit — cudavk on favorite3 (RTX 5090)

Read-only audit. Nothing was built or run on the GPU. Every number is labelled
MEASURED (from a document, the nsys sqlite export, or the perf profile),
ESTIMATED (my arithmetic, shown) or HYPOTHESIS.

Scripts that produced every trace/perf table below are beside this file:
`census.py`, `devops.py`, `pred.py`, `pred2.py`, `perfparse.py`, `perfwin.py`.

## 0. Sources, windows, and three caveats that govern every number

| source | what | window used |
|---|---|---|
| `/tmp/postdead/trace.sqlite` | nsys export, **2026.4.1.191** (`META_DATA_EXPORT.EXPORT_PRODUCT_VERSION`), NV driver 580.173.02, CUDA driver 13000, RTX 5090 (`TARGET_INFO_SYSTEM_ENV`), commit f3caa35e4ce (task statement) | relative `[9,286,588,499 ns, 12,340,525,407 ns)` = submit-ts line 2786 (index 2785) to trace end; **823 submits = 411.5 frames** |
| `/tmp/postdead/submit-ts.txt` | one CLOCK_MONOTONIC ns per vkQueueSubmit; converted with `TARGET_INFO_SESSION_START_TIME` column 4 = 1503511535334936 | — |
| `/tmp/favorite3-perf-current/` | `perf record -F 999 --clockid mono --call-graph dwarf`, **no CUPTI**, 2026-08-31 (newer commit than the trace: host-pin readback landed) | whole real-work window `time-range` = 4157 submits = **2078.5 frames**; plus a sub-window on the *same frames as the trace* (ts.txt index 2785..3608 = 411.5 frames) |

Caveat 1 — **the trace covers the LIGHT band only.** Frames 1391..1802 lie
before the heavy band (~2150–3150, PERF_HANDOFF.md "Where the numbers are").
Wall per frame inside the trace window is (12.3405−9.2866) s / 411.5 =
**7.42 ms** (MEASURED, under CUPTI and with the pre-host-pin managed
readback). The RTX heavy band has 727.9 launches/frame (PERF_HANDOFF) against
427.1 here, so every per-frame count below must be scaled — and the ratio of
copies to launches in the heavy band is **not** measured (see §9).

Caveat 2 — **every per-frame figure is a MEAN** over the window
(WORKFLOW.md §4 convention). The project's frame time is a hot-tail median.

Caveat 3 — **this trace contains 240 unified-memory migration rows per frame**
(`CUPTI_ACTIVITY_KIND_MEMCPY.copyKind` 11/12, stream 0): 170.7 UVM HtoD
(5.02 MB/frame, 268 µs/frame device) and 69.0 UVM DtoH (5.00 MB/frame,
121 µs/frame). They are **not** the DEAD_ENDS 37 profiler artefact (this is
2026.4.1, the version DEAD_ENDS 37 trusts); they are the managed readback
buffer the host-pin change of 2026-08-31 removed
(`history/perf-2026-08-31-host-pin-readback.md`: "those pages migrated
device-ward under the copy and host-ward under the application's read, every
frame"). They are excluded from every table below and belong to a closed lead.

---

## 1. The launch path, from code (task a)

### 1.1 What one launch does on the host

`cp_launch()` → `cp_launch_after()` (`src/cudavk/cp_renderer.c:531–612`).
The `cp_launch_audit` test forbids any other `cuLaunchKernel` in the driver
(comment at 505–508; grep confirms only `cp_renderer.c:585,602` plus one
`cuLaunchKernel` in `cpvk_texture_cache.c:361` for the cache convert kernel).

| step | code | CUDA API issued | cost class |
|---|---|---|---|
| 1 | `cp_ctx_check("cp_launch", …)` (537) → `cp_debug.h:361–369` | `cuCtxGetCurrent` **only if `CUDAVK_CTX_CHECK`** (default off) | one predictable branch |
| 2 | `cp_upload_flush(cp)` (538 → 464–505) | **`cuMemcpyHtoDAsync`** of the owed staging span, *only if `hi > upload_flushed`*; otherwise `cp->upload.empty++` and return | 0 or 1 API call |
| 3 | `cp->launches++`, epoch load (`__atomic_load_n` if `cp_pdl_watch`), PDL admissibility (541–556) | none | ~10 ns |
| 4 | `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (575–585) if admitted, else `cuLaunchKernel` (602) | exactly one launch API | 1.36–1.70 µs, §4 |

Not on the path: no mutex (grep `mtx_lock|pthread_mutex|simple_mtx` in
`cp_renderer.c`: **zero** hits), no NVTX unless `CUDAVK_NVTX`
(`cp_nvtx.h:36–50`, off by default), no `cuFuncSetAttribute` /
`cuFuncSetCacheConfig` anywhere in `src/cudavk` (grep: none), no
`cuStreamQuery` (only inside the `CUDAVK_WAIT_CENSUS` instrument), no per-launch
event. Kernel params are passed via `kernelParams` (`extra == NULL` at every
site checked). The flush precedes the launch, and the epoch bumped by
`cp_smallop_htod_async_raw` (`cp_smallop_tele.h:96–100`) is read *after* the
flush, so **a launch whose flush issued a copy can never take PDL** (condition
(3) in the comment at 522–529). This is by design and is what makes the copy
question in §5 a PDL question too.

### 1.2 What is issued per unit of work (code structure)

Direct opaque batch (the common case; sites in `cp_renderer.c`):

| op | site | API |
|---|---|---|
| VS argument block reserved + written | `cp_upload_begin` 5809, `cp_upload_end` 5891 | none yet (owed) |
| VS launch | `cp_launch` 5917 | flush **copy #1** (VS block, ~1.6 KB for one draw) + `cuLaunchKernel` |
| clip+raster fused / stage1 | `CP_LAUNCH_AFTER` 6988 / 6961 | `cuLaunchKernel` (CP_PDL_ANY at tier 4 > default level 3, so plain) |
| stage2, stage3 | 6996, next | `cuLaunchKernelEx` (PDL taken) |
| counter/discard clears | 4029–4032 | 0–1 `cuMemsetD32Async` (pooled by default) + `cuMemsetD8Async` when a discard mask exists |
| interp block for the FS | `cp_upload` 4171 | owed |
| compact launch | `cp_launch` 4211 (PDL only with `CUDAVK_COMPACT_PDL`) | flush **copy #2** (interp block, 416 B) + `cuLaunchKernel` |
| FS argument block | `cp_upload_begin_checked` 3709 … `cp_upload_end` 3777 | owed |
| FS launch | `cp_launch` 3920 | flush **copy #3** (FS block, 656 B + 128 B/row) + `cuLaunchKernel` |
| writeback | `cp_launch_after` 4315 | `cuLaunchKernelEx` |

So a one-shade direct batch is **7 launches, 3 HtoD copies, 0–2 memsets**.
The trace agrees: kernels launched immediately after an HtoD on their stream
are `main` 90.55/frame (= 48.47 VS + 33.88 FS + 8.2 blended, §3.3),
`cp_fs_compact` 33.88, `cp_abuf_seg_count` 5.44, `cp_vertex_fetch` 4.03,
`cp_abuf_seg_scatter` 3.43, `cp_abuf_composite` 3.43 (MEASURED, `pred.py`).

Per pass episode (blended, side-stream fan-out; `cp_renderer.c:9265–9310`,
7590–7663): 2 large `cuMemsetD32Async` (w·h·4 = 3,686,400 B counts clear;
w·h·8 = 7,372,800 B visbuf clear) + 1 small (312 B counter block) +
`cp_upload_flush` + `cuEventRecord(pass_gate)`; per segment: 1
`cuStreamWaitEvent` on the side stream, `cp_stream_set` (flushes the owed
span on the stream being left); at join: `cuEventRecord(seg_ev[k])` +
`cuStreamWaitEvent(main)` per side stream used.

Per submit (`cpvk_device_memory.c:225–400`): `cuMemcpyHtoD` (sync) of the
descriptor arena when dirty (1.01/frame, 62.5 KB), `cp_batch_flush`,
`cp_upload_flush`, `cuEventCreate` + `cuEventRecord(done)`; the worker thread
then `cuEventSynchronize`s it (2/frame on tid …096). Queries add
`cuLaunchHostFunc` (3.00/frame).

Per Vulkan entry point during *recording*: `CPVK_CTX_SCOPE` =
`cuCtxPushCurrent` + `cuCtxPopCurrent` (`cpvk_private.h:216`; 51 sites in
`cpvk_cmd.c` including `CmdBindDescriptorSets2`, `CmdBindVertexBuffers2`,
`CmdDraw*`). A heavy-band frame file (`~/favorite3-cpp/out/src/frame_0000_2499.cpp`)
records **1,000 Vulkan calls**, of which 260 BindDescriptorSets, 258 draws,
145 BindVertexBuffers, 105 UpdateDescriptorSetWithTemplate, 83
AllocateDescriptorSets — so **~900 push/pop pairs per heavy frame**
(ESTIMATED from the call list). CUPTI does not trace these; perf does (§6).

---

## 2. Per-frame host API census (task a, trace)

SQL (window bounds substituted; `census.py`):

```sql
SELECT s.value, r.globalTid, COUNT(*), SUM(r.end-r.start), GROUP_CONCAT(r.end-r.start)
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId
WHERE r.start >= 9286588499 AND r.start < 12340525407
GROUP BY s.value, r.globalTid ORDER BY COUNT(*) DESC;
```

All rows are on the main thread (globalTid …070) except `cuEventSynchronize`
(submit worker …096). Host durations are **CUPTI-inflated** (§4 gives the
no-CUPTI prices). MEASURED, per frame, 411.5 frames:

| API | n/frame | host µs/frame | mean µs | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| cuLaunchKernel | 267.91 | 584.1 | 2.18 | 2.02 | 2.45 | 4.09 |
| cuLaunchKernelEx (PDL) | 159.18 | 321.2 | 2.02 | 1.96 | 2.17 | 2.94 |
| cuEventRecord | 220.38 | 59.7 | 0.27 | 0.22 | 0.34 | 1.02 |
| cuStreamWaitEvent | 153.46 | 48.1 | 0.31 | 0.29 | 0.42 | 0.70 |
| cuMemcpyHtoDAsync_v2 | 140.77 | 261.8 | 1.86 | 1.73 | 2.15 | 3.50 |
| cuMemsetD32Async | 88.94 | 181.7 | 2.04 | 2.07 | 2.54 | 4.77 |
| cuMemcpy2DAsync_v2 | 39.07 | 138.7 | 3.55 | 2.84 | 4.23 | 20.62 |
| cuMemsetD8Async | 15.21 | 35.0 | 2.30 | 2.21 | 2.85 | 4.58 |
| cuStreamSynchronize | 11.40 | **1271.7** | 111.6 | 71.6 | 209.9 | 572.1 |
| cuMemcpyDtoH_v2 (sync) | 8.92 | 65.0 | 7.29 | 5.89 | 14.67 | 18.66 |
| cuMemcpyDtoDAsync_v2 | 7.41 | 39.3 | 5.30 | 3.38 | 12.43 | 26.70 |
| cuMemcpy3DAsync_v2 | 5.11 | 25.9 | 5.06 | 4.46 | 7.01 | 12.15 |
| cuEventDestroy_v2 | 3.12 | 1.3 | 0.43 | | | |
| cuLaunchHostFunc | 3.00 | 7.8 | 2.61 | | | |
| cuEventCreate | 2.58 | 1.2 | 0.48 | | | |
| cuCtxSynchronize | 2.04 | **1594.4** | 782.0 | 972.0 | 1272.0 | 1532.1 |
| cuEventSynchronize (worker) | 2.00 | 657.0 | 328.9 | 86.4 | 723.4 | 978.5 |
| cuSurfObjectCreate/Destroy | 2.03 | 2.1 | | | | |
| cuMemcpyHtoD_v2 (sync, descriptors) | 1.01 | 9.6 | 9.58 | | | |
| everything else (alloc/free/tex/module) | 0.7 | 41 | | | | |
| **total** | **1134.4** | **5347.4** | | | | |

Reading: **1,134 CUDA API calls per light-band frame**, of which 427 are
launches, 141 argument-block copies, 104 memsets, 374 event records/waits, 60
other copies. Blocking calls take 3,523 µs/frame under CUPTI; the rest
(**1,824 µs/frame CUPTI-inflated**) is issue.

---

## 3. Device-side operation census (task a, trace)

### 3.1 Memcpy (excluding UVM rows, see caveat 3)

```sql
SELECT s.value, m.copyKind, m.srcKind, m.dstKind, m.bytes, m.end-m.start, r.end-r.start, m.streamId
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId
LEFT JOIN StringIds s ON s.id = r.nameId
WHERE m.start >= 9286588499 AND m.start < 12340525407;
```

| API / kind | n/frame | bytes p50 / p90 / max | bytes/frame | device µs mean / p50 / p90 | µs/frame device | streams |
|---|---:|---|---:|---|---:|---|
| **cuMemcpyHtoDAsync**, pinned→device (argument-block flushes) | **140.76** | 912 / 3,104 / 31,056 | 243 KB | 0.93 / 0.77 / 1.22 | 130.6 | main 100.3/f, side streams 40.5/f |
| cuMemcpy2DAsync DtoD, managed→device | 19.75 | 2,048 / 4,096 / 8.4 MB | 163 KB | 3.63 / 0.61 / 0.80 | 71.8 | main |
| cuMemcpy2DAsync DtoD, device→device | 18.32 | 2,048 / 2,048 / 2 MB | 55 KB | 0.63 | 11.5 | main |
| cuMemcpyDtoH (sync) | 8.92 | 28 / 56 / 64 | 263 B | 0.46 | 4.1 | stream 7 (legacy) |
| cuMemcpyDtoDAsync | 7.41 | 6,828 / 63,720 / 119,160 | 159 KB | 9.09 | 67.3 | main |
| cuMemcpy3DAsync device→array (texture cache) | 5.11 | 2 MB / 4 MB / 4 MB | 10.4 MB | 4.47 | 22.9 | main, 14 |
| cuMemcpyHtoD (sync, descriptor arena) | 1.01 | 64,640 / 72,320 / 80,128 | 63 KB | 2.86 | 2.9 | stream 7 |
| cuMemcpy2DAsync device→managed (readback, pre host-pin) | 1.00 | 3,686,400 | 3.68 MB | **615.5** | 614.8 | main |

HtoD flush size histogram (power-of-two buckets, n/frame): ≤256 B 12.75;
≤512 B 2.15; ≤1 KB **57.59**; ≤2 KB **41.09**; ≤4 KB 18.42; ≤8 KB 5.27;
≤16 KB 2.97; ≤32 KB 0.52. The ≤1 KB bucket is the FS block (656 B) and interp
block (416 B) copies; the ≤2 KB bucket is the one-draw VS block (1,592 B); the
larger ones are multi-draw batches (128–144 B per extra row) and hardware
texture tables. Sizes derived in §5.1.

### 3.2 Memset

```sql
SELECT s.value, m.bytes, m.value, m.end-m.start, r.end-r.start, m.streamId
FROM CUPTI_ACTIVITY_KIND_MEMSET m
LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId
LEFT JOIN StringIds s ON s.id = r.nameId
WHERE m.start >= 9286588499 AND m.start < 12340525407;
```

| API | n/frame | bytes/frame | device µs/frame | dominant (bytes × value → n/frame) |
|---|---:|---:|---:|---|
| cuMemsetD32Async | 88.94 | **300 MB** | 129.9 | 3,686,400×0 → 21.28 (A-buffer counts per episode, `cp_renderer.c:9273`); 7,372,800×0xFFFFFFFF → 21.09 (visbuf clear, 6887/9359); 312×0 → 17.55 (counter block, 9278); 12×0 → 5.44; 57,600 / 115,200 → 3.00 each; 4×0 → 2.43; 29,491,200×0xFFFFFFFF → 1.54; five 1.00/frame attachment clears |
| cuMemsetD8Async | 15.21 | 15.8 MB | 13.5 | 1,843,200×0 → 6.16 (discard/resolved masks); 576×0 → 2.48; 921,600×0 → 1.54; others ≤1.33 |

Small (<4 KB) memsets: 17.55 + 5.44 + 2.43 + 2.48 + ~1 ≈ **29/frame**
(MEASURED). At the §4 project price of 0.66–1.01 µs per small clear removed
that is 0.019–0.029 ms/frame (ESTIMATED) — most of the small-clear folding
(counter pool, `seed_counts` in the fetch kernel) is already done.

### 3.3 Kernels, syncs, events

```sql
SELECT k.streamId, k.launchType, COUNT(*), SUM(k.end-k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE k.start >= 9286588499 AND k.start < 12340525407 GROUP BY 1,2;
SELECT syncType, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start >= … GROUP BY 1;
```

- Kernels: **427.09/frame**; main stream (13) 304.74, stream 14 53.82, seven
  further side streams 7.5–13.4 each. Device time summed 4,970 µs/frame (not a
  union). Generated `main` kernels: 90.55/frame — split by same-stream
  neighbours (`pred.py` follow-up): **VS 48.47/frame (987 µs/frame), FS
  33.88/frame (443 µs/frame)**, 8.2 blended-path shaders. Every one of the
  90.55 is launched immediately after an HtoD copy.
- Sync API device rows: context sync 2.04/frame (1,594 µs), stream sync
  11.40/frame (1,271 µs), event sync 2.04/frame (655 µs), stream-wait 153.46/frame.
- Event records: 220.38/frame → main stream 116.67, stream 14 20.51, seven
  others 11.5–12.6. Matched against later `cuStreamWaitEvent`/`cuEventSynchronize`
  on the same `eventId` (`pred2.py`): **main-stream records never waited on:
  24.03/frame**; stream 14: 1.01; all side-stream records are waited. The
  never-waited ones are the texture-cache `writer_event` records after each
  image write (`cpvk_texture_cache.c:290`, called from
  `cpvk_device_memory.c:284,317` for clears/copies — 39 image copies per frame in
  §3.1) that no later cache rebuild consumed. 1,343 distinct event objects
  cycle through the window (2 created/destroyed per submit for `done`).

---

## 4. Per-launch host cost without CUPTI (task d deliverable)

`perfwin.py`: perf samples on the main thread restricted to the *same frames
as the trace* (ts.txt index 2785..3608, 411.5 frames; 1.001 ms per sample at
999 Hz), divided by the trace's per-frame call counts. On these frames the
main thread is on-CPU 6.242 ms of a 6.993 ms wall frame (MEASURED).

| API | perf ms/frame | samples | trace n/frame | **µs per call, no CUPTI** (±1σ Poisson) | CUPTI mean µs (§2) | inflation |
|---|---:|---:|---:|---:|---:|---:|
| cuLaunchKernel | 0.455 | 187 | 267.91 | **1.70 ± 0.12** | 2.18 | +28% |
| cuLaunchKernelEx | 0.216 | 89 | 159.18 | **1.36 ± 0.14** | 2.02 | +49% |
| cuMemcpyHtoDAsync | 0.182 | 75 | 140.77 | **1.30 ± 0.15** | 1.86 | +43% |
| cuMemsetD32Async | 0.095 | 39 | 88.94 | 1.07 ± 0.17 | 2.04 | +91% |
| cuMemsetD8Async | 0.019 | 8 | 15.21 | 1.28 ± 0.45 | 2.30 | |
| cuMemcpy2DAsync | 0.080 | 33 | 39.07 | 2.05 ± 0.36 | 3.55 | |
| cuMemcpyDtoDAsync | 0.019 | 8 | 7.41 | 2.63 ± 0.93 | 5.30 | |
| cuMemcpy3DAsync | 0.019 | 8 | 5.11 | 3.81 ± 1.35 | 5.06 | |
| cuMemcpyDtoH (sync) | 0.061 | 25 | 8.92 | 6.82 ± 1.36 | 7.29 | |
| cuEventRecord | 0.019 | 8 | 220.38 | 0.09 ± 0.03 | 0.27 | |
| cuStreamWaitEvent | 0.005 | 2 | 153.46 | 0.03 ± 0.02 | 0.31 | |
| cuCtxPushCurrent + PopCurrent | 0.061 | 25 | (not traced by CUPTI; ~900 pairs/heavy frame, §1.2) | ~0.07–0.1 per pair (ESTIMATED) | — | |
| cuCtxSynchronize | 2.167 | 891 | 2.04 | 1,062 | 782 | blocking |
| cuStreamSynchronize | 1.900 | 781 | 11.40 | 167 | 112 | blocking |

**Per-launch host cost: 1.36–1.70 µs, and 100% of it is inside libcuda.**
Leaf-DSO attribution over the whole 2,078-frame window (`perfparse.py`):
`cuLaunchKernel` 0.542 ms/frame with leaf `libcuda` 0.542; `cuLaunchKernelEx`
0.346 / 0.346; `cuMemcpyHtoDAsync` 0.303 / 0.303. Driver-side self time on the
launch path (`cp_launch_after`, `cp_upload_flush`, `cp_launch`) does not
register a single sample (< 0.0005 ms/frame, i.e. < 1 ns per launch at perf's
resolution). Driver code *around* the path that does register: building the
argument blocks in pinned staging (`memset` 0.011–0.014, `memcpy` 0.010
ms/frame), `cp_pdl_stream_op` 0.013 ms/frame (the relaxed atomic epoch
increment on ~620 stream ops/frame ≈ 21 ns each), `cp_draw_execute_batch`
self 0.028. None is a lead. DEAD_ENDS 25 independently measured
`cuLaunchKernel` at 1.43 µs of CPU on this machine — consistent.

The CUPTI inflation (28–91% per call) is the reason CLAUDE.md forbids deciding
host-vs-kernel-bound from a traced run; the §2 host-µs column must not be
quoted as cost.

---

## 5. The argument-passing design (task b)

### 5.1 Block sizes and what travels how

Two families of kernels, two mechanisms:

1. **Fixed CUDA kernels take their arguments by value** as kernel parameters:
   `cp_rasterize_stage*(struct cp_rasterize_args, struct cp_rast_queues)`,
   `cp_clip_rast_fused(cp_clip_args, cp_rasterize_args, cp_rast_queues)`,
   `cp_fs_compact(struct cp_fs_interp_args)`, `cp_fs_writeback(struct cp_fs_writeback_args)`,
   `cp_vertex_fetch(struct cp_vertex_fetch_args)` etc. (`kernels/cp_rasterize.cu:1387–1455`,
   `cp_fs.cu:602,1216`, `cp_vertex_fetch.cu:19`). These are several hundred bytes
   each (`cp_rast_queues` is static-asserted at 56 B, `cp_rast_types.h:1412`;
   `cp_fs_interp_args` ≈ 416 B by field walk of `cp_rast_types.h:593–756`;
   `cp_vertex_fetch_args` ≈ 904 B by field walk of 1442–1531). **No device copy
   is traced for any of these launches** — the MEMCPY table's correlated APIs are
   exclusively memcpy calls (§3.1). MEASURED: by-value params of this size cost no
   separate device op.
2. **Generated shaders (every kernel named `main`) take exactly one pointer**
   (`nir_to_ptx/cp_nir_to_llvm.c:4959–4961`: `arg0 = LLVMGetParam(fn, 0)`,
   `num_kernel_args = 1`) to a device block of `CP_MAX_ARG_SLOTS = 64` pointer
   slots (`cp_nir_to_llvm.c:167`; slot layout `cp_rast_types.h:393–500`), loaded
   once per slot with invariant loads (`cp_arg_slot`, 264–299). The block is
   written into pinned staging by `cp_upload_begin/_end` and lands by the
   deferred flush copy.

Block sizes (ESTIMATED from the layout code):

| block | layout | one draw | 16-draw batch | 128-draw batch (`CP_MAX_BATCH_DRAWS`, `cp_debug.h:150`) |
|---|---|---:|---:|---:|
| VS (`cp_renderer.c:5785–5807`) | 64×8 pointers + 32 B scalars + rows×16×8 UBO table + rows×4×4 draw params, 16-aligned, + 904 B `cp_vertex_fetch_args` | **1,592 B** | 3,752 B | 19,880 B |
| FS (`cp_renderer.c:3688–3697`) | 64×8 + 16 B scalars + rows×16×8 (+ 416 B interp if `CUDAVK_INTERP_INLINE`) | **656 B** | 2,576 B | 16,912 B |
| interp for the FS (`cp_upload` 4171) | `struct cp_fs_interp_args` | **416 B** | same | same |
| hardware texture table (3451) | `count × 8` CUtexObject | 8·count | | |

Cross-check against the trace: flush copies p50 912 B, mean 1,727 B, p99
11,408 B, max 31,056 B (§3.1). All observed copies are under 32 KB; 99% under
12 KB; the p90 (3,104 B) is under 4 KB.

The FS block also carries tables that the generated code reads through
**device pointers** written into the block: `args[9]` → per-draw UBO table
(`fs_args_dev + fs_tbl_off`), `args[10]` → batch rows, `args[11]` → mask,
`args[14]` → the interp block, `args[15]` → the texture table
(`cp_renderer.c:3730–3752`); likewise VS `args[7]` draw-parameter rows,
`args[9]` UBO table, `args[16]` fetch args (5822–5847). This self-reference is
why the block is reserved before it is written ("a block can refer to itself",
comment 379–385). Tables that change per batch (not per launch): UBO rows,
draw-param rows, slices, fetch args; per launch: the scalar area and the
pointer slots.

### 5.2 Does the copy sit between two dependent kernels on the same stream? Yes, three times per shade

Order on the issuing thread (host order, `pred.py` last table): 140.76/frame
`cuLaunchKernel` calls are immediately preceded by `cuMemcpyHtoDAsync` —
**52.5% of all plain launches**. Device order per stream (`pred.py`):

| kernel launched right after an HtoD on its stream | n/frame | what depends on what |
|---|---:|---|
| `main` (VS) | 48.47 | stream order only (previous batch's tail → VS) |
| `main` (FS) | 33.88 | **true dependency**: compact → [FS block copy] → FS |
| `cp_fs_compact` | 33.88 | **true dependency**: stage3 → [interp block copy] → compact |
| `cp_abuf_seg_count` / `_scatter` / `cp_abuf_composite` / `cp_vertex_fetch` | 5.44 / 3.43 / 3.43 / 4.03 | blended path |

Where PDL is set: after the flush, and refused if the flush issued (§1.1). The
trace confirms no requested PDL link is ever spanned by a copy: `Ex` launches
159.18/frame all follow a kernel; the only PDL-capable kernel seen plain is
`cp_fs_writeback` 0.71/frame against 33.17 `Ex` (take rate 97.9% at that site,
≈99.6% overall). `cp_fs_compact` is PDL-capable (`CP_PDL_ANY`, tier FS) **but
the site only asks when `CUDAVK_COMPACT_PDL` is set** (`cp_renderer.c:4195–4211`),
because the interp copy would decline it anyway — DEAD_ENDS 26.

### 5.3 What the copy costs on the stream's critical path (MEASURED, trace)

`pred2.py` splits each link by whether the launch API had *returned before the
predecessor kernel ended* (`RUNTIME.end < predecessor.end` — the device, not
the host, sets the gap) and measures predecessor-end → kernel-start:

| link type | n/frame | device-paced gap p10 / **p50** / p90 µs | mean |
|---|---:|---|---:|
| kernel → kernel, PDL (`Ex`) | 159.18 | 0.13 / **0.26** / 0.64 | 0.46 |
| kernel → kernel, plain | 98.08 | 1.02 / **1.86** / 3.58 | 7.75 |
| kernel → **HtoD copy** → kernel (plain), total | **102.96** | 2.34 / **3.71** / 12.00 | 15.41 |
| … of which pp_end → copy start | | 0.93 / 1.92 / 7.65 | 11.72 |
| … copy duration | | 0.38 / 0.70 / 0.99 | 0.87 |
| … copy end → kernel start | | 0.77 / 0.96 / 3.39 | 2.82 |
| kernel → memset → kernel (plain), total | 21.78 | 2.69 / 4.35 / 12.80 | 11.30 |
| host-paced HtoD chains (host arrived late) | 37.80 | p50 89 µs (host wait, not a device cost) | |

**A flush copy in a device-paced same-stream chain adds ≈ 3.71 − 1.86 = 1.85 µs
at p50 (ESTIMATED from two MEASURED medians) over a plain kernel→kernel link,
and ≈ 3.45 µs over a PDL link.** The copy engine's pickup latency (1.9 µs) is
the largest part, not the 0.7 µs DMA itself. DEAD_ENDS 26 reports "gap before
fs_compact 5.70 µs median … the counter pool alone took it to 0.83 µs"; in the
default configuration traced here the interp copy is in the compact link and
the device-paced link median through a copy is 3.7 µs.

### 5.4 Could the block be a kernel parameter instead?

Limits, verified as far as this machine allows:

- `/usr/local/cuda-12.8/include/cuda.h` (`CUDA_VERSION 12080`) contains **no
  constant for the parameter-space size** — grep for `MAX_PARAM`, `32764`,
  `paramSize` finds only `cuFuncGetParamInfo`/`cuKernelGetParamInfo` (17685,
  8752) and the `CU_LAUNCH_PARAM_BUFFER_*` `extra` mechanism (4622–4661). The
  4 KB classic limit and the 32,764-byte limit for Volta+ with CUDA ≥ 12.1 are
  documented in the CUDA Programming Guide / 12.1 release notes, not in the
  headers; the local install has no `doc/` tree to cite. **Treat 32,764 as
  documented-but-unverified-here (HYPOTHESIS until probed).** The driver here is
  580.173.02 / CUDA 13.0 runtime-compatible, toolkit 12.8, so the large-parameter
  path exists if the documentation is right.
- The `cuLaunchKernel`/`cuLaunchKernelEx` doc block (17730–17830) states the
  driver copies `kernelParams[i]` contents at launch; by-value structs of
  56–904 B are launched today with **no traced device op** (§5.1). Whether
  parameter blocks **above 4 KB** are staged through a hidden device buffer
  (which would reintroduce exactly the DMA being removed) is **not verifiable
  here** and is the first probe (§8).
- Every observed flush (max 31,056 B) fits 32,764 B; 99% fit 12 KB; the p90
  fits the classic 4 KB. A one-draw VS block (1,592 B) and FS block (656 B) fit
  4 KB with room; a 128-draw VS batch (19,880 B) needs the large-parameter path.

What would have to change:

1. `nir_to_ptx`: declare the kernel with a by-value `[N x i8]` parameter and
   turn `cp_arg_slot()` (264–299) into `ld.param` at compile-time offsets
   (faster than today's invariant global load). Tables indexed at run time by
   batch row (`args[9][row*16+i]`, `args[7]`, `args[10]`) would need either
   `cvta.param` generic addressing into parameter space (PTX ISA ≥ 7.7,
   sm_70+; unverified on this toolchain) or to stay in device memory — **and
   anything that stays in device memory keeps its upload**, so the copy is
   removed only if the *whole* block, tables included, goes by value.
2. `cp_shader_abi.h` static asserts (`CP_ARG_UBO_STRIDE == CP_MAX_CONST_BUFFERS`,
   slot ordering) survive unchanged; `CPVK_DESCRIPTOR_SIZE` rows are *pointed to*
   by the tables, not embedded, so descriptor arenas are unaffected.
3. The interp block (416 B) is passed **twice** today — by value to
   `cp_fs_compact` and uploaded for the FS's `args[14]` — and would be embedded
   in the FS parameter block, which is exactly `CUDAVK_INTERP_INLINE`'s shape
   in the upload world. DEAD_ENDS 26 measured that shape **+0.13 ms/frame slower,
   microcause unattributed**. That result is the strongest evidence that moving
   these bytes is not free even when a copy disappears.
4. Host-side: `cuLaunchKernel` must memcpy 0.6–20 KB of params per launch
   instead of 8 B. At ~10 GB/s single-thread memcpy that is 0.06–2 µs per
   launch (ESTIMATED), i.e. the same order as the 1.30 µs `cuMemcpyHtoDAsync`
   call it replaces for large batches; a wash for big batches, a clear saving
   for one-draw ones.

### 5.5 Pricing (task b), light band, MEASURED prices from PERFORMANCE.md §4

| component | count/frame (MEASURED) | price | ms/frame (ESTIMATED) |
|---|---:|---|---:|
| flush copies removed outright | 140.76 | 1.30 µs (§4 "small copy removed"; and exactly the no-CUPTI host cost measured in §4 here) | **0.183** |
| alternative device-side view: device-paced chains through a copy | 102.96 | 1.85 µs p50 chain excess (§5.3) | 0.190 |
| + host-paced copies, host API time | 37.80 | 1.30 µs | 0.049 |
| compact PDL becomes admissible (`CUDAVK_COMPACT_PDL`) | 33.88 | 0.44–0.78 µs (§4 PDL rows) — or 1.60 µs trace delta | 0.015–0.054 |
| VS as PDL secondary after the previous tail | 48.47 | would require a `griddepcontrol.wait` prologue in generated VS PTX (none exists: grep `griddepcontrol` in `nir_to_ptx/` = 0) | 0.02–0.08 if built |
| **ceiling, light band** | | | **≈0.2–0.3** |
| **ceiling, heavy band** (scale by 727.9/427.1 = 1.70 launches — assumes the copy:launch ratio holds, HYPOTHESIS) | | | **≈0.35–0.5** |

Against that ceiling stand two measured negatives: DEAD_ENDS 26's +0.13
ms/frame for enlarging/moving the FS upload, and the unmeasured cost of >4 KB
parameter launches. **Net expectation: under 0.5 ms/frame on the RTX heavy
band, with a realistic risk of zero or negative.** The B200 ceiling is not
derivable from this evidence: B200 per-link latencies, copy-engine pickup
latency and its host CPU are all unmeasured here (§9).

How this differs from the DEAD_ENDS entries it resembles:

- **DEAD_ENDS 8 (flush placement)** refused choosing better flush points and
  names the retry condition as "uploads stop being owned by the launch
  boundary — a different ownership model". Passing the block by value *is*
  that different model — there is no upload to place. It is not the same lead,
  and entry 8's own numbers (76% of 1,785 flush points empty) say nothing about
  it. Entry 8 also prices merging copies at an unchanged boundary at 0.59 µs,
  which caps the cheaper alternative (reserve the FS and interp blocks before
  the VS launch so all three ride one copy): (33.88+33.88) × 0.59 = **0.040
  ms/frame** — not worth a design change.
- **DEAD_ENDS 26 (INTERP_INLINE / COMPACT_PDL)** emptied the stage3→compact
  link in the upload world and lost 0.13. The kernel-parameter design removes
  the copy from *all three* links per shade and from the VS, so its scope is
  ~3× entry 26's; but entry 26 is direct evidence that the mechanism
  "fewer/larger argument transfers" can lose, and its microcause was never
  attributed. Any build must carry entry 26's palindromic method and its
  "gap-before-compact" instrument.
- **DEAD_ENDS 1 (graphs)** and **25 (device-side chaining)** attack the launch,
  not the copy; unrelated.

---

## 6. Where the ~1.5 ms/frame of CUDA issue APIs goes (task c)

`perfparse.py`, main thread, whole real-work window (2,078.5 frames,
16,064 samples = 7.736 ms/frame on-CPU of 8.454 wall; the residual audit's
16,064 samples / 7.739 ms reproduce exactly):

| class | ms/frame | note |
|---|---:|---|
| CUDA blocking APIs | 4.489 | `cuCtxSynchronize` ← `cp_smallop_ctxsync` ← `cp_scratch_begin` 2.916; `cuStreamSynchronize` ← `cp_sync_timed` 1.508; `cuMemcpyDtoH` 0.049; leaf `vdso` (clock_gettime in the spin) 0.62 of it |
| CUDA issue APIs (non-blocking) | **1.737** | breakdown below |
| no CUDA API in stack | 1.510 | app-side `memcpy` 0.359 (harness), shader compilation (`compile_module_to_ptx` 0.468 + `link_and_inline_bitcode` 0.053, cold-start only), driver planning ~0.33 |

Issue APIs by entry point and driver call site (ms/frame, whole window):

| libcuda entry | ms/frame | driver call site | leaf DSO |
|---|---:|---|---|
| cuLaunchKernel | 0.542 | `cp_launch` 0.531, `cp_draw_execute_batch` 0.006, `cpvk_cache_rebuild` 0.005 | libcuda 100% |
| cuLaunchKernelEx | 0.346 | `cp_launch_after` | libcuda 100% |
| cuMemcpyHtoDAsync_v2 | 0.303 | `cp_smallop_htod_async_raw` (the flush) | libcuda 100% |
| cuMemsetD32Async | 0.123 | `cp_smallop_memset32_async` 0.119 | libcuda |
| `?` (stripped libcuda entry) | 0.104 | unknown; 0.007 of it lands in libnvidia-ptxjitcompiler | libcuda |
| cuMemcpy2DAsync_v2 | 0.065 | `cpvk_execute_copy` (app image copies) | libcuda |
| cuLinkAddData_v2 + cuLinkComplete + cuModule* | 0.085 | `load_shader_module*` — cold-start compilation, 13 frames | |
| **cuCtxPushCurrent_v2 + cuCtxPopCurrent_v2** | **0.068** | `cpvk_CmdBindDescriptorSets2` 0.013, `vk_common_CmdBindVertexBuffers` 0.007, `vk_common_CmdBindDescriptorSets` 0.007, `cpvk_ctx_leave` 0.007, … (recording-time entry points, §1.2) | libcuda |
| cuMemcpy3DAsync_v2 | 0.027 | `cpvk_cache_rebuild` | |
| cuEventRecord | 0.022 | `cp_pdl_event_record` 0.012 (gates/joins), `cpvk_texture_cache_image_written` 0.010 | |
| cuMemsetD8Async | 0.017 | `cp_smallop_memset8_async` 0.009, `cp_draw_execute_batch` 0.008 | |
| cuMemcpyHtoD_v2 (descriptors, sync) | 0.012 | `cp_smallop_htod` | |
| cuStreamWaitEvent | 0.011 | `cp_pdl_stream_wait_event` | |
| cuMemcpyDtoDAsync_v2 | 0.008 | `cpvk_execute_copy` | |
| cuLaunchHostFunc, cuEventCreate, alloc/free/tex objects | ~0.015 | | |

Steady-state issue (excluding the 0.085 of compilation) ≈ **1.65 ms/frame**,
of which launches 0.888 (54%), argument-block copies 0.303 (18%), memsets
0.140 (8%), app-driven image copies 0.100 (6%), context push/pop 0.068 (4%),
event/wait traffic 0.033 (2%), unknown 0.104 (6%).

**Inside libcuda versus around it:** every issue API's samples have their leaf
in libcuda (the only exceptions are ptxjit under the link calls). The driver's
own work between calls — argument-block assembly (`memset`/`memcpy` into pinned
staging ≈ 0.025), the PDL epoch increments (0.013), batch bookkeeping
(`cp_draw_execute_batch` self 0.028), `cpvk_texture_cache_resolve_batch`
(0.005) — totals under 0.1 ms/frame and is part of the "no CUDA API" 1.51,
not of the 1.74. Locks: `___pthread_mutex_lock` appears in one sample in the
whole profile. There is no driver-side overhead on the issue path to remove;
the 1.65 ms is the CUDA driver's per-call cost multiplied by the call count.

---

## 7. Other host-issue redundancies found (task d)

| item | evidence | per-frame cost (no CUPTI) | verdict |
|---|---|---:|---|
| Context push/pop on every recording entry point | §1.2, §6; ~900 pairs/heavy frame | 0.068 ms (MEASURED, perf whole window) | real, small; recording touches no CUDA (TODO "the driver issues device work only from inside vkQueueSubmit"), so the push/pop protects nothing on `vkCmd*` calls that only record; removing it from pure-recording entry points is a ≤0.07 ms/frame item, below the 0.5 gate, and needs the `CUDAVK_CTX_CHECK` test run afterwards |
| Event records never waited on | §3.3: 24.03/frame on main + 1.01 on stream 14 | 25 × 0.09 µs = 0.002 ms (ESTIMATED) | negligible |
| `flush_retire[]` events created at context init and never recorded | `cp_renderer.c:106–116, 1044–1045` are the only references | 0 | dead code, no cost |
| Small memsets (<4 KB) | §3.2: ~29/frame | 29 × 1.07 µs host + 0.66–1.01 µs device price ≈ 0.05 ms (ESTIMATED) | already mostly folded (counter pool, `seed_counts`); remainder is 17.55 counter-block + 5.44 sum3 clears per episode |
| Interp block passed twice (by value to compact + uploaded for FS) | §5.4 item 3 | one 416 B copy of the 3 per shade | same lead as §5; DEAD_ENDS 26 measured the fold |
| Per-launch `cuFuncSetAttribute`, `cuStreamQuery`, per-launch events, mutexes, NVTX | §1.1 grep | none exist | nothing to remove |
| `cuEventCreate`/`cuEventDestroy` per submit for `done` | `cpvk_device_memory.c:378–380`, 2.6–3.1/frame | ~0.002 ms | negligible |
| Descriptor arena `cuMemcpyHtoD` synchronous, 1/frame, 62.5 KB | §2, §3.1; PERFORMANCE §5.2b "descriptor upload 1.00 waits/f, 0.012 ms" | 0.012 ms | known, closed by size |

---

## 8. Leads (structured; ceilings on RTX unless stated)

1. **Argument block by value as kernel parameters** (generated shaders, and the
   FS's interp block). Mechanism: remove the 140.76/frame flush copies and the
   1.85 µs p50 they add to each device-paced same-stream link; allow
   `CUDAVK_COMPACT_PDL` to take. Ceiling **0.2–0.3 ms/frame light band
   (ESTIMATED), 0.35–0.5 heavy band (HYPOTHESIS by launch-ratio scaling)**,
   minus unmeasured costs (large-parameter launch, DEAD_ENDS 26's unattributed
   +0.13). Closest dead ends: 8 (names this as its retry condition), 26 (same
   bytes moved in the upload world, lost). Cheapest probe, no driver change: a
   standalone microbenchmark in the style of `notes/cdp2-microbench.cu` on
   sm_120 and sm_100 measuring (i) back-to-back launch latency and host cost
   with 8 B / 1.6 KB / 4 KB / 20 KB by-value parameters, (ii) whether a >4 KB
   parameter launch produces a traced MEMCPY row under nsys 2026.4.1, (iii)
   `cvta.param` acceptance by this toolchain. If (i) shows >0.5 µs of added
   exposed latency at 1.6 KB, the lead is dead at the probe. Effort: probe 1 day;
   driver change (backend param model, ABI, tables) 1–2 weeks with correctness
   risk in `cp_shader_abi.h` invariants and every `args[]` reader.
2. **Drop `CPVK_CTX_SCOPE` from entry points that only record.** Ceiling 0.068
   ms/frame (MEASURED). Below gate; listed because it is free and the frame is
   host-bound in places. No DEAD_ENDS entry; the `cpvk_private.h:186–208`
   comment explains why the scope exists (interop with a co-located CUDA
   consumer) — pure-recording paths do not need it, but the change must be
   proven with `CUDAVK_CTX_CHECK` over the suite.
3. **Reserve the FS and interp blocks before the VS launch so all three blocks
   ride one flush** (upload-world alternative to lead 1). Ceiling
   (33.88+33.88) × 0.59 µs = **0.040 ms/frame** at DEAD_ENDS 8's merge price;
   the emptied compact link then admits `COMPACT_PDL` (+0.015–0.054). Not worth
   building on its own; it is DEAD_ENDS 8 and 26 territory.

Nothing in the host issue path reaches the 0.5 ms/frame gate on its own; the
largest single issue-side pool — 0.888 ms/frame of launch API — is the CUDA
driver's own per-call cost (1.36–1.70 µs × 427–728 launches) with no
driver-side overhead around it, and the routes to fewer launches are DEAD_ENDS
1, 20–22, 25.

---

## 9. Unknowns

- Copies per frame and the copy:launch ratio in the **heavy band** (this trace
  is light-band only). The heavy band's 727.9 launches/frame may carry
  proportionally more blended-episode kernels, which have fewer argument-block
  copies per launch.
- Whether `cuLaunchKernel` with >4 KB parameters stages through a device
  buffer (would re-add a DMA), and its host cost versus parameter size — not
  measurable read-only.
- `cvta.param` / dynamic indexing into kernel parameter space under this
  LLVM→NVVM→PTX→ptxas 12.8/580 chain.
- B200 copy-engine pickup latency, link gaps and host per-call cost: no B200
  trace or perf profile is on this machine.
- The identity of the 0.104 ms/frame stripped-symbol libcuda entry in perf.
- The microcause of DEAD_ENDS 26's +0.13 ms/frame, which any argument-block
  redesign inherits as a risk.

---

## Appendix: reproducing the numbers

All scripts open the database read-only
(`sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro', uri=True)`) and
use `T0 = submit_ts[2785] − 1503511535334936 = 9286588499`,
`T1 = 12340525407` (max `CUPTI_ACTIVITY_KIND_KERNEL.end`), frames = 823/2.

- `census.py` — §2 table (RUNTIME by name and thread, host duration percentiles).
- `devops.py` — §3.1–3.3 (MEMCPY/MEMSET/KERNEL/SYNCHRONIZATION/CUDA_EVENT by kind, stream, size).
- `pred.py` — same-stream predecessor of every kernel, kernel names after HtoD, host-order predecessor API of every launch.
- `pred2.py` — device-paced vs host-paced split of link gaps; event records matched to later waits by `eventId`.
- `perfparse.py` — perf `samples-flat.txt` main-thread attribution over the whole `time-range` window (API = outermost named libcuda frame; site = first driver/app frame beyond it; leaf DSO).
- `perfwin.py` — the same restricted to ts.txt index 2785..3608, divided by trace counts → µs per call without CUPTI.
