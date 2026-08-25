# Open questions and known gaps

Everything here is unfinished on purpose or unfinished by accident. It is kept
in one place so a fresh look does not have to reconstruct it from commit
messages. Read `DEAD_ENDS.md` first if you are about to optimise something --
this file is what is still open, that file is what has already been closed.

Numbers are as at HEAD, old capture 15.74 ms/frame, Crossroads 5.87 ms/frame.

## Correctness, ordered by how much they matter

1. **The direct path's one-fragment-per-pixel-per-launch invariant is false.**
   Measured: 6,343,098 second claimants of a pixel inside a single launch on the
   old capture. Nothing is known to be broken by it, but it was assumed true
   while designing a fragment-writeback fusion, and any future work that
   assumes it will be wrong the same way. See `DEAD_ENDS.md`, parked item.
2. **The descriptor arena upload is not ordered against the streams that read
   it.** `cpvk_queue_submit` uploads recorded descriptor arenas with the
   synchronous `cuMemcpyHtoD` (`cpvk_device_memory.c:217-229`), and the comment
   at `:209-213` claims "the synchronous copy also orders it before both CUDA
   streams used below". It does not, and both halves of that argument fail:
   - `desc_arena_host` is `malloc` (`cpvk_cmd.c:933`), so the host memory is
     **pageable**. CUDA 12.8 "API synchronization behavior" is explicit for that
     case: the call "will return once the pageable buffer has been copied to the
     staging memory for DMA transfer to device memory, **but the DMA to final
     destination may not have completed**". Only a copy from *pinned* memory is
     synchronous with respect to the host.
   - The synchronous forms "issue these copies through the default stream", and
     `cp->stream` and `cp->seg_streams[]` are all `CU_STREAM_NON_BLOCKING`
     (`cp_renderer.c:81`, `:7069`), which by definition does not order against
     the legacy default stream. The stale `CU_STREAM_DEFAULT` comment at
     `cp_renderer.c:77-80` is where that belief came from; see Housekeeping.

   So descriptor data can still be in flight when the kernels that read it are
   launched. **No failure has been observed** -- small arenas, a dirty-flag
   guard, and kernel launch latency all hide it -- but it is a race by the
   documented contract, not a safe shortcut. The fix is also the faster one:
   allocate the mirror with `cuMemHostAlloc` and issue `cuMemcpyHtoDAsync` on
   `cp->stream`, which is correct by construction. **Do not expect a speedup:**
   `PERFORMANCE.md` measures this wait at 0.011 ms/frame over 1.00 waits, which
   is 0.09% of the 12.435 ms the driver spends blocked. It removes one of the
   ~17 blocking sites and none of the time. This is a correctness fix only.
   The comment at `cpvk_cmd.c:934-935` that rejects page-locked memory because
   it "buys no transfer overlap" was reasoning about throughput, and on
   throughput it was right.
3. **`bindless_image_store` has no bounds check.** A latent memory-safety hole
   rather than a rendering bug.
4. **The A-buffer verifier keeps two historical internal mismatch cases** that
   have never been explained, even though external images match.
5. **Stage 2 and stage 3 do not interpolate depth to the same bits.**
6. **`texture3d` drifts over an animation** -- two differing pixels at frame 0.
7. **`multithreading` has no diagnosis.** It differs from the reference by a
   small amount and nobody has found out why.
8. **`gltfscenerendering` is a standing exception** in the 60-frame comparison
   against NVIDIA. It predates all recent work: the fused and reverted builds
   produce the same numbers for it, so it is not caused by anything recent.
9. **`renderheadless` is missing from the sweep** because it drives its own
   frames, so it is never compared. It is compared by hand instead, which means
   in practice it is compared rarely.

## Vulkan surface not implemented

The driver advertises Vulkan 1.1. The Gallium driver that was removed advertised
1.4 with 198 extensions, because it borrowed lavapipe's frontend; that surface
was never verified against the CUDA backend. See `GALLIUM_RETIREMENT.md` for the
full delta. The gaps that come up most:

- multiple render targets, layered rendering, input attachments
- timeline semaphores, multiple queues
- real occlusion and pipeline-statistics queries
- full stencil state
- vertex attribute divisors above one -- the kernel handles an arbitrary
  divisor, but `VK_EXT_vertex_attribute_divisor` is not implemented, so nothing
  can select one through the API
- no line rasterization
- sample shading is per fragment only
- non-device-0 CUDA device selection

`docs/cudavk/history/HANDOFF.md` has the longer list with the reasoning.

## CUDA interop work plan, in priority order

Making the driver usable as a frame source for a CUDA consumer (a PyTorch or
TensorRT pipeline), and close enough to the proprietary driver that one
application binary serves both. **`CUDA_INTEROP.md` holds the evidence** — what
CUDA permits, what the driver does today, the design and the rejected
alternatives. This section is only the order of work.

Two framings this list depends on:

- **Full parity with the proprietary driver is not the target.** That is 1.4 with
  ~200 extensions against 1.1 with six. The target is the surface the consuming
  application actually touches, proven by running one binary against both ICDs
  and diffing the output. Item 5 is that proof.
- **Two processes, not one.** The consumer gets the frame over an exported fd and
  a socket. That removes the context clobber, the allocator competition and the
  blast radius of correctness item 3 in one move, and it is measured working
  (`CUDA_INTEROP.md` §1.1). It also keeps external semaphores off the critical
  path, because the host handshake is cheap next to a 15.74 ms frame.

### Tier 0 — cheap, and everything rests on them

1. **~~Report a real `deviceUUID`~~ — DONE** (`8f9bba600b5`). All three UUIDs are
   real, and `deviceUUID` is byte-identical to the proprietary driver's on the
   same GPU, so a consumer matching by UUID cannot tell them apart.
   `pipelineCacheUUID` was fixed with it: left at zero it never invalidated, so
   a cache written by one build was accepted by the next. Original text:
   `cpvk_device.c:263-264` memsets `deviceUUID`
   and `driverUUID` to zero, and `cuDeviceGetUuid` is never called anywhere.
   Matching `VkPhysicalDeviceIDProperties.deviceUUID` against `cuDeviceGetUuid`
   is how a CUDA consumer identifies which GPU a Vulkan device is, and it was
   verified exact on the proprietary driver. Until this is done, a correct
   consumer cannot identify this driver's device at all. ~5 lines, and the best
   value per line in this section.
2. **~~Fix the CUDA context clobber~~ — DONE** (`b2781c4a7b1`). `CPVK_CTX_SCOPE`
   at 60 entry points, 23 inner `cuCtxSetCurrent` calls removed, only
   `cpvk_submit_worker`'s kept. `CUDAVK_CTX_CHECK` makes the coverage testable;
   removing one scope on purpose both fires it and fails 48 of 49 tests. Frame
   time unchanged on both captures, hashes identical. Original text:
   `cuCtxSetCurrent` at ~25 sites and no
   `cuCtxPushCurrent`/`PopCurrent` anywhere (`CUDA_INTEROP.md` §2.5). The CUDA
   Runtime adopts a driver context that is already current, so a co-located
   consumer can allocate inside a context that dies at `vkDestroyDevice`.
   Half a day. Matters less across processes, but it is a bug either way.
3. **~~Measure `CU_CTX_SCHED_BLOCKING_SYNC`~~ — DONE, and the answer was no.**
   `CUDAVK_CTX_SCHED` is in the registry; the default stays `auto`. Blocking
   costs 5.4% of the frame at idle and roughly **doubles** it under CPU
   contention, saving no total CPU there. `yield` is the arm that returns CPU
   under load. Full record: `DEAD_ENDS.md` entry 16. The original reasoning
   below is kept because it is what motivated the measurement, and it was
   wrong: `cuCtxCreate(&dev->cu_ctx, 0, ...)`
   at `cpvk_device_memory.c:409` selects `CU_CTX_SCHED_AUTO`, which with one
   context and many logical processors resolves to **spin** — the documented
   heuristic is C > P yields, otherwise it spins. The driver is blocked 12.435
   ms of a 15.74 ms frame, so it holds roughly 79% of a core doing nothing, plus
   the submit worker spinning alongside. Blocking-sync yields instead and pays
   wakeup latency on ~17 waits a frame. Add the flag to the `cp_debug.c`
   registry and measure both arms on both captures. Needed before co-locating
   with anything that wants CPU.

### Tier 1 — the feature, and most of the value

4. **`VK_KHR_external_memory_fd`, export path.** Honour
   `VkExportMemoryAllocateInfo` in `cpvk_AllocateMemory`
   (`cpvk_device_memory.c:565`) by allocating through the VMM — `cuMemCreate`
   with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`, then `cuMemAddressReserve` /
   `cuMemMap` / `cuMemSetAccess` — instead of `cuMemAlloc`; implement
   `vkGetMemoryFdKHR`; report `OPAQUE_FD` from
   `cpvk_GetPhysicalDeviceExternalBufferProperties` (`cpvk_device.c:585-593`)
   instead of zeros. **Add only `_fd` to `cpvk_device_extensions`** — the base
   and capabilities extensions are `promotedto="VK_VERSION_1_1"` and this driver
   already advertises 1.1, which is why that query entrypoint already exists.
   Self-contained, because one `VkDeviceMemory` is exactly one CUDA allocation.
   Watch the 2 MiB VMM granularity, and keep `vkMapMemory` working for
   exportable host-visible types.
5. **~~The acceptance test~~ — DONE, and ahead of item 4** (`4b763c7768c`).
   `src/cudavk/samples/interop/` is a two-process sample: Vulkan renders, a
   PyTorch process consumes over an exported fd with no copy, and the result
   comes back into a second shared buffer that Vulkan checks byte for byte. The
   binary cannot tell which driver it is on; `run_interop.py --icd` chooses.
   Against the proprietary driver: 5/5 including three negative controls, 1.272
   ms/frame in timeline mode, 0 of 1048568000 bytes wrong. Against cudavk it
   SKIPs with exit 3 and says why:

       this driver does not report OPAQUE_FD as exportable for this buffer usage
         VK_KHR_external_memory_fd : absent
         externalMemoryFeatures for OPAQUE_FD : 0x0

   **That is the whole of what is left in Tier 1: make item 4 turn those five
   SKIPs into five PASSes.** Original text: One binary that renders a
   known frame, exports it, imports it in CUDA and prints a hash, selected by
   `VK_DRIVER_FILES`. Run against both ICDs and diff. This is what turns
   "drop-in replacement" into a pass or a fail, and it is how Tier 1 is declared
   finished. Fits how `TESTING.md` already decides correctness.
6. **Images: honour `VkExternalMemoryImageCreateInfo`** instead of using it only
   to disqualify the hardware texture cache (`cpvk_image.c:317-328`), and
   implement `VkPhysicalDeviceExternalImageFormatInfo` in
   `cpvk_GetPhysicalDeviceImageFormatProperties2`, which is unhandled. Lower
   value than item 4 for a frame pipeline, which should export a buffer, but a
   drop-in claim needs it. Until it is done, passing that struct costs
   performance and buys nothing.
7. **Import path: `VkImportMemoryFdInfoKHR`.** Small once the VMM path exists.

*Tier 1 is done when the experiment programs recorded in `CUDA_INTEROP.md` Part 1
run unmodified against cudavk and produce the same zero-mismatch results they
produce against the proprietary driver.*

### Tier 2 — synchronisation, in dependency order

8. **Store the `CUevent` in the sync object** instead of a boolean, and stop
   destroying it in the submit worker (`cpvk_device_memory.c:64`). The header of
   `cpvk_sync.c` already anticipates exactly this, and that header's claim that
   submission drains the stream is itself stale. Nothing else in this tier is
   possible first.
9. **Make wait semaphores a `cuStreamWaitEvent`** rather than the host
   `vk_sync_wait_many` at `cpvk_device_memory.c:154-156`. Today `vkQueueSubmit`
   blocks the application thread until the producer finishes, which is a
   host wait standing in for a GPU dependency. Recovers submit-boundary
   pipelining; small once item 8 exists.
10. **Timeline semaphores.** `cpvk_sync.c:116-118` has no
    `VK_SYNC_FEATURE_TIMELINE`. The largest single item here, and the right
    primitive for a frame ring.
11. **`VK_KHR_external_semaphore_fd`**, which with timelines exports to
    `CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD` and removes the
    host round trip.

Items 8 and 9 are worth doing on their own merits. **Do not start 10 and 11 for
interop reasons before measuring that the host handshake costs something** — the
fence wait is needed regardless and a socket round trip is tens of microseconds
against a 15.74 ms frame.

There is a cheaper cudavk-only shortcut if the handshake ever does measure: the
submit event at `cpvk_device_memory.c:329` is already created with
`CU_EVENT_DISABLE_TIMING`, which is a prerequisite for `CU_EVENT_INTERPROCESS`,
so `cuIpcGetEventHandle` is close at hand. It needs one event per ring slot and a
sequence number, because a `CUevent` is binary and re-recording races a peer that
has not yet waited — and it does not exist on the proprietary driver, so it
splits the single code path. Measure first.

### Tier 3 — what a drop-in claim forces

12. **Stop hardcoding CUDA device 0** (`cpvk_device.c:314-316`). An inference
    host has four or eight GPUs. Also listed under "Vulkan surface not
    implemented"; interop is what makes it urgent.
13. **More than one queue.** `queueCount = 1` in one family
    (`cpvk_device.c:566-569`). Applications commonly use a dedicated transfer
    queue for exactly the readback a frame handover performs.
14. **Advertise Vulkan 1.2 or 1.3.** Not for interop directly — because
    applications written against the proprietary driver routinely *require* it
    at instance creation and refuse to start. Scope it to what the target
    application requests and let item 5 report what is missing.

### Tier 4 — correctness, with no speedup attached

15. **Bounds-check `bindless_image_store` and `bindless_image_atomic`**
    (correctness item 3). Low priority across two processes, high the moment
    anything is co-located, because the address space then holds the consumer's
    model weights.
16. **Fix the descriptor upload race** (correctness item 2). Correctness only:
    `PERFORMANCE.md` measures that wait at 0.011 ms/frame.

### Not part of this work

**The ~17 blocks a frame are out of scope and must not be folded in.** They are
read-back-and-decide, not synchronisation: the host copies counters back and
branches on overflow and coverage, so no sync primitive removes them. The ceiling
for removing them entirely is about 4.16 ms/frame of device idle, the research is
unacted in `notes/DEVICE_AUTONOMOUS_SYNC.md`, and the most obvious approach is
already refuted three ways in `DEAD_ENDS.md` entry 1.

**If only four things are done: 1, 2, 4 and 5.** That is an identifiable driver,
a safe context, working fd export, and a test that proves it against the real
driver.

## Performance leads that are open, not closed

Ranked by the evidence behind them, not by size. `PERFORMANCE.md` has the
measurements these come from.

1. **The host waits about 17 times a frame and is blocked about 12.44 ms doing
   it.** Device idle is 4.16 ms, almost exactly host issue time, so the driver
   is a ping-pong rather than a pipeline. This is the largest remaining item by
   a wide margin and the hardest. Two of its three sites survived contact with
   evidence; the third is closed. The design and the addendum that cut it down
   are in `/tmp/perf16/iter29-sync/`, and the two cheap probes that should be
   run before anything is built are described there: the ratio of the A-buffer
   bound to its actual count, and how much host work exists to overlap with a
   wait. Neither has been run.
2. **Fuse the fragment writeback into the fragment shader.** Parked with a
   working mechanism and a wrong result; about 0.04 ms and a known next step.
   See `DEAD_ENDS.md`.
3. **Five CUDA-side items from a research note**, none of which need a toolkit
   upgrade: `CU_JIT_SPLIT_COMPILE` (cold JIT 536 ms to 186 ms, measured),
   raising `CUDA_CACHE_MAXSIZE`, the `enable_smem_spilling` pragma, reopening
   CUDA graphs on 12.8 specifically, and `griddepcontrol`, which is used zero
   times today. `notes/CUDA13_UPGRADE.md`.
4. **The opaque sort-middle tiling prototype** in `notes/OPAQUE_TILING_PROTOTYPE.md`
   exists behind flags and was never taken to an accepted result.
5. **`pbribl` regresses by about 0.03 ms** with vertex-fetch fusion enabled and
   nobody knows why. Bounded and deliberately accepted; the cost is in fused
   execution on that workload, not in the machinery around it.

## Housekeeping

- The two-capture timing harness lives in `/tmp/perf16/`, outside the
  repository. It should be moved into `src/cudavk/tests/` so it survives a
  reboot and can be reviewed like everything else.
- The build directory is still called `build-cudapipe` in every document and
  script. Harmless, but wrong now.
- `nir_intrinsic_load_const_buf_base_addr_cudapipe` keeps the old name on
  purpose: it lives in Mesa-common files, where a fork-local rename only widens
  the diff against upstream. Rename it only if the fork stops tracking upstream.
- `notes/VIDEO_ENCODE_SURVEY.md` is a survey of what the tree contains, not a
  plan. Nothing in cudavk implements video encode.
