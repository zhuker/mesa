# Open questions and known gaps

Everything here is unfinished on purpose or unfinished by accident. It is kept
in one place so a fresh look does not have to reconstruct it from commit
messages. Read `DEAD_ENDS.md` first if you are about to optimise something --
this file is what is still open, that file is what has already been closed.

Numbers are as at HEAD (`e2fea470d04`), old capture **13.16 ms/frame**,
Crossroads **5.82 ms/frame** (`/tmp/perf-audit/reprofile_baseline.md`). The old
capture's 15.74 stood until `20611f5b131` made the opaque-episode fan-out the
default; `CUDAVK_NO_OPAQUE_STREAMS=1` still measures 15.89.

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
   against NVIDIA, and its differing-pixel count **more than doubled once**,
   which a standing exception is not permission to do. Bisected through
   `~/git/Vulkan/build/iter/iterations.json`:

   | iteration | commit | frame 0 | worst | at |
   |---|---|---:|---:|---:|
   | `step2-native-verify` | `5deebecefd5` | 15695 | 59926 | 47 |
   | `tc-default-on` | `ab7b431611a` | 16991 | **137025** | 47 |

   Every gated run since -- `fixstreams-off`, `fixstreams-on`, `interop-tier1`
   -- reports 137025/137026 at frame 47, so it stepped once and has been flat
   since. It is **not** caused by the interop work or by the opaque-stream
   fan-out; both post-date it.

   **Exactly one commit separates those two iterations**: `ab7b431611a`,
   "cudapipe: move the measurement tooling to the driver it measures". It is a
   file move plus one behavioural line -- `DRIVER=native` became the default in
   `cp_iterate.sh`, where it had been the Gallium-hosted driver. So the likely
   explanation is that **the two rows measure different drivers** and nothing
   regressed: the number stepped because what was compared changed. That is a
   hypothesis, not a finding. Nobody has re-rendered `5deebecefd5` under the
   native driver to confirm it, and until somebody does, 59926 must not be
   quoted as a native-driver baseline. `TESTING.md` asks for frame number,
   differing-pixel count, maximum delta and coherent regions; the first two are
   above and the last two are still unrecorded.
9. **`renderheadless` is missing from the sweep** because it drives its own
   frames, so it is never compared. It is compared by hand instead, which means
   in practice it is compared rarely.

## Vulkan surface not implemented

The driver advertises Vulkan 1.1. The Gallium driver that was removed advertised
1.4 with 198 extensions, because it borrowed lavapipe's frontend; that surface
was never verified against the CUDA backend. See `GALLIUM_RETIREMENT.md` for the
full delta. The gaps that come up most:

- multiple render targets, layered rendering, input attachments
- multiple queues
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

8. ~~**Store the `CUevent` in the sync object**~~ — **done.** A sync carries a
   refcounted `struct cpvk_cuevent` (`cpvk_sync.c`), armed by the submit that
   will signal it and released by whichever of the pending-submit record, a
   reset or the destroy gets there last. The worker no longer destroys it.
9. ~~**Make wait semaphores a `cuStreamWaitEvent`**~~ — **done.**
   `cpvk_queue_submit` waits on the renderer stream when the sync carries an
   event and falls back to the host wait when it does not, which is the case
   where nothing has been recorded into it yet. `CUDAVK_NO_GPU_SEM_WAIT`
   reverts it. Neutral on both captures, which is what it should be: neither
   capture blocks on a semaphore for long enough to matter.
10. ~~**Timeline semaphores.**~~ — **done.** One `vk_sync` type serves both
    kinds: a 64-bit counter, a monotone signal, `VK_SYNC_FEATURE_WAIT_PENDING`
    and `move`. `VK_KHR_timeline_semaphore` and `timelineSemaphore` are
    advertised; apiVersion stays 1.1 and the extension form is deliberate.
    Wait-before-signal works through the runtime's ASSISTED timeline mode
    rather than by claiming `WAIT_BEFORE_SIGNAL`, which this backend cannot
    honestly do — there is no way to enqueue a wait for a value nothing has
    promised. `src/cudavk/tests/cpvk_timeline_semaphore.c` is the test.
11. **`VK_KHR_external_semaphore_fd` — the export half is impossible, the
    import half is possible and pointless here.** The two halves are separate
    entry points, `vkGetSemaphoreFdKHR` and `vkImportSemaphoreFdKHR`, and
    Vulkan gives them separate capability bits —
    `VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT` and
    `..._IMPORTABLE_BIT` — so importable-only is a legal, advertisable state
    rather than a fudge. Take them one at a time.

    **Import: implementable, unimplemented for want of a producer.** CUDA is
    fully capable on the imported side — `cuImportExternalSemaphore` then
    `cuSignalExternalSemaphoresAsync` / `cuWaitExternalSemaphoresAsync`, proven
    honoured by the probe below. So `vkImportSemaphoreFdKHR` could hand an fd
    to CUDA and the driver could then wait and signal it on its own streams.
    What is missing is anyone to create the fd: only a real Vulkan driver, D3D
    or NvSciSync can mint one, and on the machine cudavk exists to serve it
    **is** the Vulkan driver, while the CUDA consumer has no export either. It
    would pay only in a mixed topology where another Vulkan device owns the
    sync objects. Left undone deliberately; revisit if that case appears.

    **Export: NOT IMPLEMENTABLE, and now MEASURED. Do not attempt it.**
    A driver built on CUDA has nothing to export.

    The API listing says so: of 606 driver API entry points in `cuda.h` 12.8
    there is **no call that creates or exports a semaphore**.
    `cuImportExternalSemaphore` is import-only, and every one of the ten
    `CUexternalSemaphoreHandleType` values names an object some *other* API
    made. `cuIpcGetEventHandle` is not a way round it: a 64-byte opaque struct,
    not a file descriptor, and CUDA-to-CUDA only.

    The docs alone would not settle it — the text for `OPAQUE_FD` says only
    "a valid file descriptor referencing a synchronization object" and never
    says who made it, and `TIMELINE_SEMAPHORE_FD` has no per-type paragraph at
    all. So every plausible fd was offered to it (`/tmp/interop/sem/`):

    | fd offered | as OPAQUE_FD | as TIMELINE_SEMAPHORE_FD |
    |---|---|---|
    | NVIDIA Vulkan timeline semaphore (`/dev/nvidiactl`) | **SUCCESS** | **SUCCESS** |
    | DRM binary syncobj (`anon_inode:syncobj_file`) | 999 | 999 |
    | DRM timeline syncobj | 999 | 999 |
    | DRM syncobj, `HANDLE_TO_FD_FLAGS_TIMELINE` | 999 | 999 |
    | Linux `sync_file` (`EXPORT_SYNC_FILE`) | 999 | 999 |
    | eventfd, memfd, plain file, `/dev/null`, pipe | 999 | 999 |

    Every non-NVIDIA descriptor fails identically to a plain file, on all four
    DRM nodes including NVIDIA's own (`nvidia-drm`), 32 of 32 attempts. The
    export side works fine unprivileged; CUDA simply will not take it.

    **The test that pins down the rule**: a raw `open("/dev/nvidiactl")` fd is
    also refused with 999 — and that is the very character device the working
    Vulkan semaphore fd points at, confirmed by `readlink` on both. So the
    requirement is not the device node, not GPU access and not the DRM driver
    identity. **The fd must carry a resource-manager object minted inside the
    NVIDIA kernel driver by the NVIDIA user-mode stack.** `/dev/nvidia0` and
    `/dev/nvidia-uvm` are refused too.

    And that is the whole asymmetry with memory, in one line: CUDA 12.8 has
    `cuMemExportToShareableHandle`, so a VMM fd exists for
    `cuImportExternalMemory` to accept. There is no `cuExportExternalSemaphore`.
    Import only ever takes what some NVIDIA component exported, and for
    semaphores CUDA cannot be that component. **Buffers can be shared;
    synchronisation cannot.**

    Type 9 having no per-type paragraph in the docs hides no looser rule: it
    behaved exactly like type 1 on every input offered. `sw_sync` could not be
    tested — debugfs is root-only here — but the gap is covered, because
    `EXPORT_SYNC_FILE` yields a genuine `anon_inode:sync_file`, the same kernel
    object type, and CUDA refuses that too.

    Note this also settles the sample: cudavk having timeline semaphores does
    not make the interop sample's timeline mode pass, because that protocol has
    the producer *export* two semaphores. 3/5 is the ceiling there.

    A lavapipe-style `SYNC_FD` shim remains possible for Vulkan-to-Vulkan use
    (`lvp_pipe_sync.c:266` drains the device and returns an already-signalled
    fd, or `-1`), but CUDA has no `SYNC_FD` handle type, so it buys nothing
    here.

    So cudavk cannot mint the `TIMELINE_SEMAPHORE_FD` a CUDA consumer would
    import, and the interop sample's timeline mode cannot pass against this
    driver. That is a capability gap in the same class as dma-buf export, not
    a matter of effort. **The host handshake is the supported synchronisation
    for cudavk interop**, and it is measured cheap: the sample reports 1.440
    ms/frame host against 1.272 timeline on a driver where both exist, about
    0.17 ms against a 13.2 ms cudavk frame.

    Items 8 to 10 were worth doing on their own merits — they are
    Vulkan-internal and do not depend on export, and they are done. Say that
    precisely, because the pair is easy to confuse: **cudavk has timeline
    semaphores, and cudavk still cannot pass the interop sample's timeline
    mode.** The first is a Vulkan feature and part of the drop-in surface; the
    second needs an exportable fd CUDA will accept, and the table above is what
    says there is not one.

**Do not start 11 for interop reasons at all** — it is closed. And do not
reach for external synchronisation before measuring that the host handshake
costs something: the fence wait is needed regardless and a socket round trip is
tens of microseconds against a 15.74 ms frame.

There is a cheaper cudavk-only shortcut if the handshake ever does measure: the
submit event at `cpvk_device_memory.c:329` is already created with
`CU_EVENT_DISABLE_TIMING`, which is a prerequisite for `CU_EVENT_INTERPROCESS`,
so `cuIpcGetEventHandle` is close at hand. It needs one event per ring slot and a
sequence number, because a `CUevent` is binary and re-recording races a peer that
has not yet waited — and it does not exist on the proprietary driver, so it
splits the single code path. Measure first.

#### 12. The doorbell: GPU-side synchronisation without an exportable semaphore

Item 11 is closed because CUDA will not take a synchronisation object from us.
It will take **memory**, which is the one thing sharing already works for — and
CUDA can wait on memory in hardware. So the missing semaphore can be routed
around rather than mourned.

    /* consumer, once per slot, enqueued ahead of the frame */
    cuStreamWaitValue64(op_stream, doorbell, seq, CU_STREAM_WAIT_VALUE_GEQ);

    /* producer, as the last command of the frame */
    vkCmdFillBuffer(cb, slot_buf, doorbell_off, 4, seq);

`GEQ` is a monotonic comparison, which is exactly a timeline semaphore's, and
the doorbell is a word inside the buffer already exported over the fd. Nothing
new has to cross the process boundary. **The producer side needs no new driver
feature**: `vkCmdFillBuffer` is a stream-ordered device write and lands after
the render work on the same CUDA stream.

Measured on this machine (`/tmp/interop/doorbell.c`), one context, two streams:

    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS : 1
    before the write, the waiting stream    : CUDA_ERROR_NOT_READY, correctly blocked
    after the write, the waiting stream     : released
    cuStreamWriteValue64 + cuStreamWaitValue64 : 1795 ns per pair

So it blocks and releases as a semaphore does, at roughly 900 ns an operation.
Compare the host handshake it would replace: GPU completes, the submit worker
wakes from `cuEventSynchronize`, signals a condvar, the application thread wakes
from `vkWaitForFences`, a socket or futex wakes the consumer, and only then is
CUDA work enqueued. Three thread wake-ups and a launch. **A futex only replaces
the socket** — one hop of four, perhaps 20 µs — whereas the doorbell removes the
CPU from the loop entirely, which is the property a real semaphore has and the
reason to want one.

**The catch, and it is structural rather than incidental.** The consumer must
enqueue its wait *before* the frame is ready, several frames ahead. A consumer
that waits for a message and then enqueues has put the CPU back in the loop and
gained nothing. That is how timeline semaphores are used everywhere, so it is
not exotic, but it changes the consumer from "receive, then submit" to "keep N
frames queued, each gated on doorbell >= seq" — and `samples/interop/PROTOCOL.md`
is written the first way, so the sample cannot demonstrate this without a
protocol change.

**What is proven and what is not.** Proven: the mechanism, the blocking, the
release, the cost — in one process, one context, two streams. Not proven, and it
is the configuration that matters: cross-process, across two CUDA contexts, on
*imported* memory, with the write coming from `vkCmdFillBuffer` on cudavk's
stream. Three things could bite — whether the write is visible to another
context's wait in the right order, whether one device's L2 makes that automatic,
and whether stream ordering survives the opaque fan-out onto the side streams
(`CUDAVK_NO_OPAQUE_STREAMS` is the control). Until that experiment is run this
is a promising mechanism and not a design.

### Tier 3 — what a drop-in claim forces

13. **Stop hardcoding CUDA device 0** (`cpvk_device.c:314-316`). An inference
    host has four or eight GPUs. Also listed under "Vulkan surface not
    implemented"; interop is what makes it urgent.
14. **More than one queue.** `queueCount = 1` in one family
    (`cpvk_device.c:566-569`). Applications commonly use a dedicated transfer
    queue for exactly the readback a frame handover performs.
15. **Advertise Vulkan 1.2 or 1.3.** Not for interop directly — because
    applications written against the proprietary driver routinely *require* it
    at instance creation and refuse to start. Scope it to what the target
    application requests and let item 5 report what is missing.

### Tier 4 — correctness, with no speedup attached

16. **Bounds-check `bindless_image_store` and `bindless_image_atomic`**
    (correctness item 3). Low priority across two processes, high the moment
    anything is co-located, because the address space then holds the consumer's
    model weights.
17. **Fix the descriptor upload race** (correctness item 2). Correctness only:
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

1. **The host waits 16.76 times a frame and is blocked 9.73 ms doing it.**
   Device idle is 3.70 ms, almost exactly host issue time (3.48 ms), so the
   driver is still a ping-pong rather than a pipeline. This is the largest
   remaining item by a wide margin and the hardest. The fan-out
   (`20611f5b131`) took 2.65 ms out of the episode drain at an **unchanged**
   9.88 waits/frame — it shortened each wait, it did not remove one — leaving
   the drain at 5.989 ms/frame, peel checks at 2.751 over 1.70 waits and
   segment counters at 0.982 over 4.17. Crossroads blocks 2.563 ms over 8.26
   waits and runs **no peel checks at all**. Two of the site's three routes
   survived contact with evidence; the third is closed
   (`/tmp/perf16/iter29-sync/`).

   **Both gating probes have now been run** — both captures, one session, at
   `e2fea470d04`, instrumentation only behind `CUDAVK_DRAIN_PROBE`, output
   hashes and submit counts unchanged
   (`/tmp/perf-audit/probes_p1_p2.md`, diff at
   `/tmp/perf-audit/drain_probe.diff`):

   - **P1 — bound-based sizing is not dead.** Median `total/quads` is **3.753**
     on old and **3.725** on Crossroads, max **3.99** over 20,704 episodes. The
     ratio is capped at 4 by geometry (a quad is 2×2 pixels), and the driver
     allocates `4 × quads` (`want_slots = num_quads * 4`,
     `cp_renderer.c:4182`) over arrays that are dense over **quads**
     (`slot = 4·q + lane`). A host substitute that knows only `total` must
     bound quads by it and allocate `4 × total`, i.e. **3.75× today's
     allocation** — the 94% figure is how full today's arrays are, not an
     allocation. S1d survives P1, but it is **gated on memory, not on ratio**,
     against the same 8.59 GB scratch cap `UNSAFE_NO_OVERFLOW` hit; the
     deciding measurement is the `dscratch` high-water at the drain. Related
     and free: the tail drain's `quad_over` test is redundant when
     `fill_over == 0`, since `quad_capacity == capacity` and `quads <= total`.
     **Caveat: P1 is _not_ the `bound/actual` ratio of the `bounded` fast
     path** — that is `ab->nblocks × rast_num_triangles` over actual quads, a
     different quantity — so the clip-rectangle lead must not be sized from
     3.75 and still needs its own probe at `cp_renderer.c:6402–6413`.
   - **P2 — there is host work to overlap.** Median inter-drain issue burst
     **306 µs** on old and **139 µs** on Crossroads, against a mean drain wait
     of 606 µs, so the "tens of microseconds → dead" branch does not fire.
     **Caveat: the naive `min(burst, wait) × 9.88 = 3.02 ms/frame` double-counts.**
     The host's total issue time is 3.48 ms/frame and the device is 72% busy,
     so the frame cannot fall below ≈9.5 ms; the real headroom is the 3.70 ms
     of device idle, and 2.5–3.5 ms/frame is the honest ceiling on old.
     P2 says nothing about S1b's 250 MB–1.6 GB memory cost, which is why S1b
     stays demoted.

   A third finding fell out of P1: **6.9% of old-capture drains and 18.3% of
   Crossroads drains return `quads == 0`** — the host blocks to be told the
   episode covered nothing. That is 0.053 ms/frame on old (0.9% of the site's
   blocked time) and **0.322 ms/frame on Crossroads (15.1%)**, where an empty
   drain costs almost as much as a productive one (0.315 against 0.397 ms
   mean). **It is not a lead**: the drain is taken for `fill_over`/`quad_over`,
   `quads` rides along in the same copy, and `cp_pass_can_retry` forbids the
   fragment shader before the overflow answer, so an oracle for `quads == 0`
   would save no wait. Only S1d — draining at the scan — would attack it.

   **P3, the probe P1 said was still needed, has also been run, and it closes
   the clip-rectangle lead** (`/tmp/perf-audit/bound_probe_p3.md`). At the drain
   the `bounded` predicate falls back to, the clip rectangle is **never tighter
   than the framebuffer** (0 of 8,836 samples), `nblocks × tris / actual quads`
   has a median of **3.46e7** on old and **2.95e7** on Crossroads, and **no
   variant admits a single draw** — today's, clip-rect, or with `tris <= 2`
   dropped. `PERFORMANCE.md` §6 item 2 is closed at 0.00 ms, and the `bounded`
   fast path is effectively dead code on both captures.
   **STATUS AFTER 2026-08-26**: six of the eight blocking sites are closed by
   the census (`PERFORMANCE.md` §5.2b), and the **episode drain is the one that
   is open — and it is now the largest single lead in the driver, worth up to
   about 2.07 ms/frame.** Its ceiling is 2.066 ms/frame at 34.3% of its blocked
   time, gap-bound on 72% of its waits, and its conversion factor is **measured
   at +1.02** by injecting host time at the site (`CUDAVK_WAIT_SPIN_US`, slope
   over a 0–1.98 ms/frame sweep, residuals under 0.031 ms). Host time added
   there lands one for one on the frame.
   The peel site is closed by the same probe returning **−0.03**, which also
   **retires the 11% "conversion"** taken from the rejected predication patch:
   injecting 2.05 ms/frame at the peel site costs nothing, so that patch's
   0.110 ms was its mechanism, not its blocking. Do not size anything with 11%.
   `ready = 0` at all 28,852 waits, so no wait in this driver is pure overhead.
   **What is unproven at the drain is symmetry** — the probe measures the *add*
   direction, and a deferral mechanism still has to be built and measured.
   **Three sites are live, not one.** The same probe at the other two gives
   **segment counters +1.03** near the origin (ceiling 0.253 → **≈0.26
   ms/frame**) and **`vkDeviceWaitIdle` +0.44**, linear to 1.01 ms/frame
   injected (ceiling 0.500 → **≈0.22 ms/frame**). Each is comparable to a whole
   accepted iteration. Both had been closed "by size" at the retired 11%, which
   was wrong: they are **sized and open**. The segment sweep saturates above
   ~0.5 ms/frame injected (point slopes 1.16, 1.03, 0.80), so the three are
   unlikely to be additive — recovering at one site spends slack another would
   have used.
   Also open: where the 2.453 ms median inter-submit stall actually lives, since
   `vkDeviceWaitIdle` blocks only 0.514 ms/frame and so is not mostly that.

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
6. **Where the 2.453 ms median inter-submit stall lives.** `PERFORMANCE.md`
   §5.2 attributes the frame-boundary gap to it, and the census now shows
   `vkDeviceWaitIdle` blocks only 0.514 ms/frame, so most of that stall is
   somewhere the driver does not currently instrument. Open.
7. **`CUDAVK_PDL` is measured and unlanded.** +0.4342 ms (+3.29%) on old and
   +0.1297 (+2.23%) on Crossroads at level 3 against level 0, decisive
   instrument, arms non-overlapping, p = 0.0011 and 0.0143; −1.5% on the
   eighteen-sample sweep with no sample regressed and the standing exceptions
   unchanged. It works by overlapping a kernel's preamble with its
   predecessor's tail — the same family as the fan-out. Levels are additive to
   within 0.042 ms.

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
