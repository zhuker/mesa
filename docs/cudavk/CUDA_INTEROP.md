# cudavk and CUDA interop

How a regular CUDA program — a PyTorch or TensorRT pipeline, say — gets at a
frame this driver rendered, and how to do that with one application code path
that also works on the real NVIDIA Vulkan driver.

The motivating case is one application that renders with cudavk on GPUs where
the NVIDIA driver's rasterizer is not available, and with the proprietary driver
where it is, and in both cases hands finished frames to an ML pipeline.

Everything in Part 1 was measured on this machine on 2026-08-24 (RTX 5090,
driver 580.173.02, CUDA 12.8.93). Everything in Part 2 is what the code at
commit `21d4021f958` does. Nothing here is implemented yet: Part 3 is a design,
not a description.

---

## Part 1 — What CUDA actually permits

These four were unknown when the question was asked, and each one decides part
of the design, so each was tested rather than reasoned about. Sources and raw
output: `/tmp/interop/exp/` (not in the tree).

### 1.1 A VMM-exported fd can be imported as external memory. This is the load-bearing result.

A POSIX file descriptor produced by

    cuMemCreate(...)                 /* prop.requestedHandleTypes =
                                        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR */
    cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)

is accepted by

    cuImportExternalMemory(&em, {.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
                                 .handle.fd = fd, .size = n, .flags = 0})
    cuExternalMemoryGetMappedBuffer(&p, em, ...)

Both calls return `CUDA_SUCCESS`. The result is the same memory, not a copy: a
pattern written through the original `cuMemMap` mapping reads back through the
imported pointer with 0 of 524288 words wrong, and a write through the imported
pointer is visible through the original.

- `CUDA_EXTERNAL_MEMORY_DEDICATED` is **not** required. `flags = 0` works.
- Allocation granularity is 2 MiB, so exports are 2 MiB-quantised.
- The fd survives `SCM_RIGHTS` to another process with its own `cuInit` and its
  own context: import succeeds, 0/524288 mismatch, and the consumer's writes are
  visible to the producer. This is the ML-consumer shape and it works.
- Control: `cuMemImportFromShareableHandle` + `cuMemMap` on the same fd also
  works and also aliases. Sanity: an ordinary file fd gives
  `CUDA_ERROR_UNKNOWN` (999), so the API does inspect what it is given.

**Consequence.** cudavk can implement `VK_KHR_external_memory_fd` honestly, over
the CUDA VMM, and a consumer written against the proprietary driver's export
path needs no change at all. That is the single code path this document
recommends.

### 1.2 The proprietary driver's export works, and needs less than expected

`/usr/share/vulkan/icd.d/nvidia_icd.json` is installed on this machine.
`VkPhysicalDeviceIDProperties.deviceUUID` matches `cuDeviceGetUuid` exactly, so
that is how a `VkPhysicalDevice` is mapped to a CUDA device.

`vkGetMemoryFdKHR` → `cuImportExternalMemory` → `cuExternalMemoryGetMappedBuffer`
returns `CUDA_SUCCESS` in all four arms of {dedicated, not dedicated} ×
{`DEVICE_LOCAL`, `HOST_VISIBLE|HOST_COHERENT`}. **A dedicated allocation is not
required for a `VkBuffer`.** `VkMemoryDedicatedAllocateInfo` and
`CUDA_EXTERNAL_MEMORY_DEDICATED` are both optional, but they must agree with
each other. On a host-visible type both directions were verified: CUDA reads
Vulkan's pattern (0/4096) and Vulkan's mapped pointer reads a CUDA kernel's
write (0/4096).

A `VkImage` in `VK_IMAGE_TILING_OPTIMAL` is a different matter — it imports as an
opaque `CUmipmappedArray`, because the tiling is the driver's business. Exporting
a **buffer** avoids that entirely and is what Part 3 does.

### 1.3 Cross-context pointers work on the same device, and this is not a licence to use them

With `ctxA = cuCtxCreate` and `ctxB = cuDevicePrimaryCtxRetain`, `ctxA` not
current, a pointer allocated in `ctxA`:

| | `cuMemAllocManaged(GLOBAL)` | `cuMemAlloc` |
|---|---|---|
| host read | works, correct | **SIGSEGV** |
| `cuMemcpyDtoH` from `ctxB` | `CUDA_SUCCESS`, correct | `CUDA_SUCCESS`, correct |
| kernel in `ctxB` reads/writes | `CUDA_SUCCESS`, correct | `CUDA_SUCCESS`, correct |
| writes from `ctxB` visible in `ctxA` | yes, coherent both ways | yes |
| `CU_POINTER_ATTRIBUTE_CONTEXT` | the *allocating* context | the *allocating* context |
| `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` | 2 (DEVICE), not 4 (UNIFIED) | 2 (DEVICE) |
| `cuMemPrefetchAsync` | `CUDA_SUCCESS` | `CUDA_ERROR_INVALID_VALUE` (1) |

Coherence was proven rather than assumed: the final word is `0xbeef1000`, where
`0xBEEF0000` came from a *host* write in `ctxA` and `+0x1000` from a *kernel*
write in `ctxB`.

Retaining the primary context before or after the allocation makes no
difference; all four cases were step-for-step identical.

So for device-side use, plain `cuMemAlloc` is already as good as managed across
contexts here. Managed buys host addressability and prefetch, and nothing else
that was measured.

**Do not build on this.** No CUDA documentation promises cross-context pointer
validity. It is a useful debugging fact and a legitimate prototype shortcut; it
is not a foundation.

### 1.4 Two failure modes worth knowing

- **IPC cannot bridge two contexts in one process.** `cuIpcGetMemHandle`
  succeeds, then `cuIpcOpenMemHandle` returns `CUDA_ERROR_INVALID_CONTEXT` (201).
  IPC is a cross-*process* mechanism.
- **Context lifetime is the real hazard, and it is silent.** After
  `cuCtxDestroy(ctxA)`, using a pointer from `ctxA` **segfaults the process with
  no CUresult at all**. Reproduced twice. Being current does not matter; being
  alive does. An fd-imported buffer has explicit ownership and a borrowed raw
  pointer does not, which is most of the argument for Part 3.

---

## Part 2 — What the driver does today

### 2.1 Nothing is exported, and nothing can be

- Device extensions (`cpvk_device.c:31-66`) are six: `KHR_dynamic_rendering`,
  `KHR_swapchain`, `KHR_maintenance1`, `KHR_descriptor_update_template`,
  `KHR_create_renderpass2`, `KHR_depth_stencil_resolve`. Instance
  (`cpvk_device.c:24-29`): `KHR_surface`, `EXT_headless_surface`. No
  `VK_KHR_external_memory_fd`, `_external_semaphore_fd` or `_external_fence_fd`.
- The **platform-agnostic half is nevertheless already present**, and this is
  easy to misread. `VK_KHR_external_memory` and
  `VK_KHR_external_memory_capabilities` are `promotedto="VK_VERSION_1_1"` in
  `vk.xml`, and this driver advertises 1.1 (`cpvk_device.c:131`), so
  `VkExportMemoryAllocateInfo`, `VkExternalMemoryBufferCreateInfo` and
  `VkExternalMemoryImageCreateInfo` are all already valid `pNext` structs here,
  and `vkGetPhysicalDeviceExternalBufferProperties` is already a required
  entrypoint. What is missing is only the *transport* — the fd extension. Those
  two extension names should never appear in `cpvk_device_extensions`.
- `cpvk_GetPhysicalDeviceExternalBufferProperties` (`cpvk_device.c:585-593`)
  exists for that reason, and returns `(VkExternalMemoryProperties){0}` for
  every query. Fences (`:595-606`) and semaphores (`:608-619`) are all-zero too.
  So a well-behaved application can already detect that cudavk cannot export,
  which is the backend-selection test Part 3 uses.
- `vkGetMemoryFdKHR`, `VkExportMemoryAllocateInfo` and `VkImportMemoryFdInfoKHR`
  do not occur anywhere. `cpvk_AllocateMemory` never scans `pNext` for them.
- `VkExternalMemoryImageCreateInfo` is read in exactly one place,
  `cpvk_CreateImage` (`cpvk_image.c:317-328`), and is **not** honoured. It only
  disqualifies the image from the hardware texture cache (`:326`). An
  application that passes it gets an ordinary internal image that has quietly
  lost a fast path.
- The VMM API and the IPC API are not used at all: zero hits for `cuMemCreate`,
  `cuMemMap`, `cuMemAddressReserve`, `cuMemSetAccess`,
  `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`,
  `cuIpcGetMemHandle`, `cuIpcOpenMemHandle`, `cuImportExternalMemory`,
  `cuGraphicsResource*`, `cuMemHostRegister`.
- Nothing escapes the shared object either: `gnu_symbol_visibility: 'hidden'`
  (`meson.build:169,181`), no installed headers, and `vkGetBufferDeviceAddress`
  is not an entrypoint — the device advertises Vulkan 1.1 (`cpvk_device.c:131`,
  `meson.build:199`), and buffer device address is a 1.2 feature. The internal
  field `buffer->vk.device_address` *is* set to a real `CUdeviceptr`
  (`cpvk_device_memory.c:853-855`), but no caller can reach it.

**The only handle that escapes the driver today is the pointer from
`vkMapMemory`.**

### 2.2 Memory

One `CUcontext` per `VkDevice`, created with `cuCtxCreate` — *not* the primary
context (`cpvk_device_memory.c:409`), on CUDA device 0, hardcoded
(`cpvk_device.c:314-316`). The same context is handed to the renderer
(`:418-419`). Two `vkCreateDevice` calls make two contexts.

Three memory types, one heap (`cpvk_device.c:510-553`):

| index | flags | enum | allocator |
|---|---|---|---|
| 0 | `DEVICE_LOCAL` | `CPVK_MEM_DEVICE` | `cuMemAlloc` (`cpvk_device_memory.c:589`) |
| 1 | `HOST_VISIBLE｜HOST_COHERENT｜HOST_CACHED` | `CPVK_MEM_HOST` | `cuMemAllocManaged(GLOBAL)` (`:595`) |
| 2 | the two above together | `CPVK_MEM_MANAGED` | `cuMemAllocManaged(GLOBAL)` (`:600`) |

One `VkDeviceMemory` is exactly one CUDA allocation. There is no suballocation
of application memory, which is what makes an fd export tractable. The 256 MiB
arena in `cp_renderer.c:108-111` is private renderer scratch and never backs a
`VkBuffer` or `VkImage`.

A resource's device address is `mem->dev_ptr + bind offset + level/layer offset`
(`cpvk_device_memory.c:853-855`; `cpvk_image.c:442`, `cpvk_cmd.c:1506`).

`vkMapMemory` (`cpvk_device_memory.c:747-763`) fails for type 0, which is
deliberately unmappable (`:754-759`). For types 1 and 2 it returns
`mem->host_ptr + offset`, where `host_ptr = (void *)(uintptr_t)mem->dev_ptr`
(`:596-597`, `:601-602`) — **the mapped host pointer is numerically the
`CUdeviceptr`**. Unmap, flush and invalidate are no-ops (`:765-783`).

*Stale comment.* `cpvk_private.h:164` says `CPVK_MEM_HOST` uses
`cuMemHostAlloc`. It does not; `:595` calls `cuMemAllocManaged`. No pinned host
memory backs any `VkDeviceMemory`. `cuMemAllocHost` appears once, for the
renderer's private 8 MiB staging (`cp_renderer.c:116`).

### 2.3 Images are linear, and that is guaranteed on purpose

Every `VkImage` is tightly packed row-major behind one `CUdeviceptr`. No tiling,
no swizzle, no pitch padding, no `CUarray` in the authoritative path.

`cpvk_image_layout()` (`cpvk_image.c:224-273`): `row_stride[l] = nblocksx *
blocksize` with no padding (`:265`), running `level_offset[l]` (`:266`), array
layers contiguous within a level (`:268`), MSAA samples as whole planes
(`:271-272`). The no-padding rule is load-bearing, not incidental — the shared
renderer addresses colour attachments as `pixel_index * blocksize` (`:259-264`).
`vkGetImageSubresourceLayout` reports these numbers verbatim (`:394-413`).
`linearTilingFeatures == optimalTilingFeatures` (`:160-167`).

Colour is written in place, in the image's own memory and encoding
(`cpvk_cmd.c:1502-1519`; kernel address `color_out + sm*sample_stride +
(y*width+x)*bpp`, `cp_fs.cu:1053-1065`). **There is no resolve or writeback step
producing a plain image, because it is plain throughout.** The only real resolve
is the MSAA average (`cpvk_cmd.c:1691-1706`).

**Depth is the exception and it is a trap.** During a render pass, depth lives in
the renderer-private `cp->depthbuf` as `uint32` per sample, indexed
`[(sample*height+y)*width+x]` (`cp_renderer.c:9010-9011`, `cp_clear.cu:78-80`),
in sign-flipped sortable-uint form, not float (`cp_renderer.c:2437-2444`). The
image is loaded at scope begin (`cp_renderer.c:8922-8926`) and stored back at
scope end (`:8929-8938`), re-encoded to `D32_SFLOAT`, `stencil|d24` or
`unorm16`. **The
depth `VkImage` is only valid after `vkCmdEndRendering` with `storeOp = STORE`.**

The hardware texture cache is a *derived, read-only* `CUmipmappedArray`
(`cpvk_texture_cache.c:144-187`, 384 MiB budget). Linear memory is the source of
truth; nothing is ever copied back. Coherence is `content_epoch` plus a writer
event (`:249-284`, `:304-390`). It is irrelevant to readback, and it is the
thing `VkExternalMemoryImageCreateInfo` currently disables.

WSI is the software path: `sw_device = true`, `wants_linear = true`
(`cpvk_device.c:354-365`), which forces `VK_IMAGE_TILING_LINEAR` and a
host-visible allocation that is mapped and copied. Headless present is a no-op.
**Do not tap the frame at the swapchain** — it is the wrong place here, and it is
not exportable at all on the proprietary driver.

### 2.4 Synchronisation: CUDA events inside, no exportable semaphore outside

Two stream families, both `CU_STREAM_NON_BLOCKING`: `cp->stream`, one per device
(`cp_renderer.c:81`), carrying every frame-path launch; and `cp->seg_streams[8]`,
created lazily for pass episodes (`:7069-7070`), joined with
`cuEventRecord`/`cuStreamWaitEvent` (`:7157-7180`).

`cpvk_queue_submit` (`cpvk_device_memory.c:151-410`) makes the renderer stream
wait for its wait semaphores (`:170-190`), does a blocking descriptor
`cuMemcpyHtoD` (`:243-259`), translates recorded ops into CUDA work *at submit
time* (`:262-332`), records a `CUevent` (`:372-373`), arms the syncs it will
signal with it (`:388-400`) and returns **without draining**. A worker thread
(`:95`) does `cuEventSynchronize` (`:57`) then publishes the signals (`:65-70`).

`VkFence` and `VkSemaphore` are one `vk_sync` type (`cpvk_sync.c:47-59`): a
64-bit value, a pending value, an epoch, and a **refcounted `CUevent`** shared
with the submit that recorded it. The event is what makes a queue-submit wait a
`cuStreamWaitEvent` on the renderer stream instead of a host block; the host
`cnd_timedwait` remains for a sync with nothing recorded into it, which is a
real case and not a fallback of convenience. **Timeline semaphores are
implemented** on the same type (`:385-410`): `VK_SYNC_FEATURE_TIMELINE`,
`VK_KHR_timeline_semaphore` and `timelineSemaphore` are advertised at
apiVersion 1.1. `VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL` is deliberately **not**
claimed — a CUDA stream cannot be told to wait for a value nothing has promised
— so the runtime runs this device in ASSISTED timeline mode and holds a
wait-before-signal submit on its own thread until the promise exists. That
choice is what the driver's `VK_SYNC_FEATURE_WAIT_PENDING` is for.

None of this is exportable, and that is the ceiling, not an omission: see
Part 5 and `TODO.md` item 11. `VkEvent` is still bool+condvar
(`cpvk_private.h:314-327`), with `cuLaunchHostFunc` as its device half
(`cpvk_cmd.c:3408`).

The driver blocks about 17 times a frame, all read-back-and-decide drains:
episode drain (`cp_renderer.c:8333-8337`, 9.88/frame, 8.663 ms), peel checks
(`:6696-6697`, 1.71, 2.765 ms), segment counters (`:6427-6434`, 4.17, 0.995 ms),
descriptor upload (`cpvk_device_memory.c:217-229`, 1.00), plus `vkDeviceWaitIdle`
→ **`cuCtxSynchronize` 2.24 times a frame** (`cpvk_device_memory.c:551-562`).
That last one is why Part 5 rejects the primary context.

*Stale comment, and a race behind it.* `cp_renderer.c:77-80` and
`cp_renderer.h:338-348` claim the main stream is `CU_STREAM_DEFAULT` "on
purpose", so that it orders implicitly against the legacy NULL stream. It has
always been `CU_STREAM_NON_BLOCKING` (commit `c488adb7676`). **There is no
implicit ordering with a caller's default stream** — which is good news for
interop, since it means an outside CUDA program cannot accidentally serialise
against the driver, and bad news inside the driver, because
`cpvk_device_memory.c:209-213` reasons from the false claim. The descriptor
arena mirror is `malloc`ed (`cpvk_cmd.c:933`), so the synchronous
`cuMemcpyHtoD` at `:217-229` is the one documented case that may return before
its DMA lands, and it lands on the legacy default stream, which the consuming
non-blocking streams do not wait for. See `TODO.md` correctness item 2. It is
unobserved, but it is a race by contract.

*Dead mechanism.* `cp->flush_retire[]` (`cp_renderer.h:576`) is created and
destroyed but never recorded or waited, and `scratch.current` is never assigned.
`ARCHITECTURE.md:631-636` describes this ring as live. It is not.

### 2.5 The context clobber

Nearly every entry point calls `cuCtxSetCurrent`: `cpvk_device_memory.c:36,131,
160,518,556,584,727`; `cp_renderer.c:65,827,2472,2492,4692,8190,9007,9220`;
`cpvk_cmd.c:3427`; `cpvk_image.c:483,561,670`; `cpvk_pipeline.c:875,1165`;
`cpvk_texture_cache.c:793,893,933`.

There is **no `cuCtxPushCurrent`, `cuCtxPopCurrent` or `cuCtxGetCurrent`
anywhere in the tree.** The driver never restores the caller's context. A thread
that mixes CUDA and Vulkan calls silently loses its own context to cudavk's.
This is a bug independent of interop and it should be fixed first.

---

## Part 3 — The recommended design

**Target `VK_KHR_external_memory_fd` on both drivers.** Given 1.1 and 1.2, the
application gets one Vulkan path and the ML consumer gets one CUDA path:

    render to your own offscreen VkImage
      → vkCmdCopyImageToBuffer into an exportable VkBuffer
      → vkGetMemoryFdKHR
      → cuImportExternalMemory(OPAQUE_FD)
      → cuExternalMemoryGetMappedBuffer
      → wrap as a tensor

Four decisions in that, each with a reason:

**Render offscreen, not to the swapchain.** cudavk's WSI copies through the host
(§2.3) and the proprietary driver will not export a swapchain image at all.

**Export a buffer, not an image.** A buffer sidesteps tiling and format
negotiation completely. On the proprietary driver an optimal-tiled image arrives
as an opaque `CUmipmappedArray`; a buffer arrives as a flat pointer that wraps
into a tensor directly.

**Keep the copy on both backends**, even though cudavk images are already linear.
It earns its place for a different reason on each: on the proprietary driver it
converts tiled to linear, and on cudavk it moves the frame out of unmappable
device-local memory into the exportable buffer. Same code, same reason to exist.
The copy is negligible next to any model.

**Synchronise with a host fence wait, at first.** `vkQueueSubmit` does not drain
(§2.4), so submit, wait the fence, then launch. With two or three frames in
flight this pipelines fine. External semaphores are more work than they are
worth until measured.

Select the backend at runtime by querying
`vkGetPhysicalDeviceExternalBufferProperties` for `OPAQUE_FD`, or by `driverID`.
Today cudavk answers zero (§2.1), so the test is unambiguous, and it stays
correct after the feature lands.

Match the CUDA device to the `VkPhysicalDevice` by
`VkPhysicalDeviceIDProperties.deviceUUID` against `cuDeviceGetUuid` (§1.2). Never
by index.

### Prototype available today, with no driver change

Allocate the readback buffer from memory type 1 or 2 and read the pointer
`vkMapMemory` returns; it is the `CUdeviceptr` (§2.2), and §1.3 confirms it is
valid and coherent from another context. Sync with `vkWaitForFences`. Reference
code exists: `src/cudavk/tests/cp_offscreen_bench.c:754-771` does
copy-to-buffer then map, and `cpvk_batch.c:437-450` maps an image and writes a
PPM.

This is a prototype and not the destination. Managed memory migrates on access,
which is wrong for a large per-frame buffer, and a borrowed pointer has the
silent lifetime hazard of §1.4.

---

### Consuming the frame in PyTorch

Measured on 2026-08-25 with torch 2.13.0+cu130 and cupy-cuda12x 14.2.0 in
`/home/alexzhukov/mesa/venv`, same box. Sources and raw output:
`/tmp/interop/torch/`.

**PyTorch has no external-memory API and does not need one.** There is no
wrapper for `cuImportExternalMemory`; `torch.cuda.CUDAPluggableAllocator`
replaces the allocator and cannot adopt a pointer, and the CUDA IPC used by
`torch.multiprocessing` only shares memory torch itself allocated. You import
with the driver API yourself and then wrap the pointer.

**The shortest thing that works, and it is zero copy.** An object exposing
`__cuda_array_interface__` passed to `torch.as_tensor`:

```python
class Wrap:
    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),      # False = writable; True is rejected
            "shape": shape,
            "typestr": typestr,        # "|u1" -> torch.uint8, so RGBA8 needs no conversion
            "strides": None,
            "version": 3,
        }

t = torch.as_tensor(Wrap(ptr, (h, w, 4), "|u1"), device="cuda")
assert t.data_ptr() == ptr             # verified exactly; memory_allocated() delta is 0
```

Verified: `t.data_ptr()` equals the imported `CUdeviceptr` exactly, and
`torch.cuda.memory_allocated()` does not move. Non-contiguous strides work. The
`version` field is not checked at all. `"data": (ptr, True)` is a hard
`TypeError` — torch does not accept read-only. **A shape larger than the buffer
is accepted silently**, so validate it yourself.

`torch.from_dlpack` on that same bare object fails with a misleading
`RuntimeError` about an "invalid capsule", which only means the object has no
`__dlpack__`. A real DLPack producer works and is also zero copy, but torch 2.13
calls `__dlpack__(max_version=(1,0), stream=1)` and reads
`DLManagedTensorVersioned`, so a legacy `dltensor` capsule **segfaults**, and the
deleter must not be a ctypes Python callback because torch calls it at
interpreter shutdown — `NULL` is legal. CuPy's `UnownedMemory` route also works,
zero copy. No method silently copied.

**Data flows both ways.** 0/524288 mismatches in every direction tested:
producer pattern read by torch, `t.add_(1.0)` seen through the producer's own
mapping, producer re-write seen by torch, `t.mul_(2.0)` seen by the producer.
Same across processes with the fd passed over `SCM_RIGHTS`.

**Bring up torch before you import.** This is the one rule that matters.
`cuExternalMemoryGetMappedBuffer` maps into whatever context is *current*, and
`CU_POINTER_ATTRIBUTE_CONTEXT` afterwards reports exactly that context. Both
orders work and both run torch kernels correctly, but they are not equally safe:

- Import with the **primary** context current — call `torch.zeros(1,
  device="cuda")` first — and the mapping is owned by the context torch holds
  for the life of the process. Safe.
- Import with a **separate** `cuCtxCreate` context current and it still works,
  by UVA, and torch displaces that context when it starts. But with the tensor
  alive, `cuCtxDestroy` on that context returns `CUDA_SUCCESS` and the next use
  of the tensor is **killed by SIGSEGV**. Undocumented, working, and unsafe.

One line of setup removes a whole class of crash. Do it.

**Import once, at startup. Never per frame.** Medians, essentially
size-independent from 2 MiB to 128 MiB:

| operation | median |
|---|---|
| `cuImportExternalMemory` | 491–547 µs |
| `cuExternalMemoryGetMappedBuffer` | 40–48 µs |
| import + map | ~530 µs |
| full export + import + map + free + destroy | 1.16–1.31 ms |
| re-wrapping an already-imported pointer | 1.09 µs |

530 µs is 3% of a 60 fps frame and 13% of a 240 fps one. Import each buffer in
the ring once and keep the tensor; re-wrapping costs a microsecond.

**Only one handle must outlive the tensors: the mapped `CUdeviceptr`.** Tested
in subprocesses: `cuDestroyExternalMemory` alone is **safe** and the tensor keeps
working; the producer's `cuMemUnmap` + `cuMemAddressFree` + `cuMemRelease` is
safe; `cuCtxDestroy` of the producer's context is safe when the import was done
in the primary context. Only `cuMemFree` on the mapped pointer kills it —
`CUDA_SUCCESS`, then SIGSEGV on next use.

**Torch does not need its caching allocator.** 21 of 21 operations worked on
imported memory, including `out=` writing back into the imported buffer, cuBLAS
matmul, cuDNN `conv2d` on a permuted non-contiguous view, `cat`, `empty_cache`,
side streams and autograd. **`record_stream()` does not error but is a no-op**
on this memory, so cross-stream ordering must use CUDA events or imported
semaphores, not `record_stream`.

*Incidental, and it costs an hour if you hit it.* libcuda 580 exports
`cuCtxSynchronize_v2`, which is **not** the public API. A ctypes binding that
prefers a `_v2` suffix gets `CUDA_ERROR_CONTEXT_IS_DESTROYED` (709) on a
perfectly healthy context.

---

### Two processes, and how they hand a frame over

Sharing one address space with an ML pipeline is workable but has sharp edges:
the driver clobbers the caller's CUDA context (§2.5), the two allocators compete
for the same GPU, and `bindless_image_store` / `bindless_image_atomic` compute a
raw address with **no bounds check** (`cp_nir_to_llvm.c:1443-1458`, `:1461-1495`;
`TODO.md` correctness item 3) — so a buggy shader that the Vulkan spec says
should have its store discarded will instead write wherever the arithmetic
points. In one process, that address space now contains the model weights.

Two processes remove all three at once, and the fd export already works across
them (§1.1). What remains is synchronisation, and it is the whole problem.

**The invariant is two-way, and the second half is the one people forget.**

1. PyTorch must not read slot `i` until the GPU has *finished writing* it.
2. Vulkan must not start rewriting slot `i` until PyTorch has *finished reading*
   it — including work still queued asynchronously on a torch stream.

**`vkQueueSubmit` returning is not a completion signal.** It returns without
draining (`cpvk_device_memory.c:306-345`); a private thread waits the `CUevent`
and only then signals the `vk_sync` (`:48-56`). `vkWaitForFences` is the
completion signal. On the proprietary driver the same is true for a different
reason.

#### The protocol to build first: a ring plus a host handshake

Three slots, each an exported buffer, each imported once by the consumer at
startup (§ PyTorch section: ~530 µs each, so never per frame).

    renderer                                consumer
    --------                                --------
    slot = wait_for_free()                  on "ready(slot, seq)":
    record cmds writing slot                    out = model(tensors[slot])
    vkQueueSubmit(..., fence)                   ev = torch.cuda.Event()
    vkWaitForFences(fence)   <-- completion     ev.record(); ev.synchronize()
    send "ready(slot, seq)"                     send "free(slot)"

`ev.synchronize()` before returning the slot is not optional. Torch work is
asynchronous, so a consumer that replies as soon as `model()` *returns* has
promised nothing, and the renderer will overwrite memory the GPU is still
reading.

This is portable — it works identically on cudavk and on the proprietary driver
— and it is fast enough that it should not be optimised on suspicion. The fence
wait is needed regardless. The added cost is one socket round trip, tens of
microseconds against a 15.74 ms frame, and with three slots it overlaps the next
frame's rendering, so in steady state it costs throughput nothing.

**On the proprietary driver, also release the buffer properly**: a
`VkBufferMemoryBarrier` with `dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL`
before the submit that finishes the frame, and acquire it back on reuse. cudavk's
barriers are no-ops beyond a batch flush (`cpvk_cmd.c:3365-3370`), so it changes
nothing there, but the code should be written for both.

#### If the host round trip ever measures as a cost

**CUDA IPC events, cudavk only.** The submit event at `cpvk_device_memory.c:329`
is already created with `CU_EVENT_DISABLE_TIMING`, which is a *requirement* for
an interprocess event. Adding `CU_EVENT_INTERPROCESS`, then `cuIpcGetEventHandle`
and sending the 64-byte handle, lets the consumer `cuIpcOpenEventHandle` once and
`cuStreamWaitEvent` on a torch stream. The GPU waits; the host does not. The
message can then be sent right after submit instead of after the fence.

Its limit: a `CUevent` is binary, not a timeline. Re-recording it for the next
frame races with a consumer that has not yet waited on the previous recording, so
you still need one event per ring slot and a sequence number in the message. And
it does not exist on the proprietary driver, so it splits the code path the rest
of this document works to keep single.

**~~Vulkan external semaphores are the portable endpoint.~~ They are not
available here, and that is now measured rather than reasoned.**
`VK_KHR_external_semaphore_fd` on a **timeline** semaphore exports to
`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD`, and a monotonic
counter is exactly the primitive a frame ring wants. On the proprietary driver
that path works: `vkGetSemaphoreFdKHR` yields an fd on `/dev/nvidiactl` and
`cuImportExternalSemaphore` accepts it both as `TIMELINE_SEMAPHORE_FD` and as
`OPAQUE_FD`. **No fd this driver could produce is accepted.** Every non-NVIDIA
descriptor was tried against `cuImportExternalSemaphore` and every one failed
identically to a plain file, with `CUDA_ERROR_UNKNOWN` (999): a DRM binary
syncobj, a DRM timeline syncobj, a syncobj created with the TIMELINE flag, a
Linux `sync_file`, and — as negative controls — an eventfd, a memfd, an
ordinary file, `/dev/null` and a pipe. The accepted handle is an NVIDIA RM
object, and CUDA has no call that creates one (§ `TODO.md` item 11). Evidence:
`/tmp/interop/sem/RESULTS.md`.

The timeline semaphores this driver now implements are Vulkan-internal, and for
the drop-in surface. They do **not** make the interop sample's timeline mode
pass, and no amount of driver work will: the missing piece is an export CUDA
cannot consume. The host handshake above is the supported mechanism.

---

## Part 4 — Work items, in order

*The authoritative priority order, with effort and done-when criteria, is
`TODO.md` under "CUDA interop work plan". This part is the technical detail
behind those items; if the two disagree, `TODO.md` is the plan and this is the
reasoning.*

1. **Fix the context clobber** (§2.5). Push/pop around entry points. Correctness,
   independent of everything else here.
2. **Implement `VK_KHR_external_memory_fd`.** That extension alone: the
   platform-agnostic half is already present, because
   `VK_KHR_external_memory` and `VK_KHR_external_memory_capabilities` are both
   `promotedto="VK_VERSION_1_1"` in `vk.xml` and this driver advertises 1.1
   (`cpvk_device.c:131`). `VkExportMemoryAllocateInfo` and
   `VkExternalMemoryBufferCreateInfo` are therefore already available to
   callers, and `cpvk_GetPhysicalDeviceExternalBufferProperties` already exists
   at `cpvk_device.c:585` for the same reason — it is a core 1.1 entrypoint that
   currently answers "no capability". Do not add the promoted extensions to the
   list; add `VK_KHR_external_memory_fd` to `cpvk_device_extensions`
   (`cpvk_device.c:31-66`); report `OPAQUE_FD` from
   `cpvk_GetPhysicalDeviceExternalBufferProperties`
   (`:585-593`) instead of zeros; honour `VkExportMemoryAllocateInfo` in
   `cpvk_AllocateMemory` (`cpvk_device_memory.c:565`) by allocating that memory
   through the VMM (`cuMemCreate` / `cuMemAddressReserve` / `cuMemMap` /
   `cuMemSetAccess`) instead of `cuMemAlloc`; implement `vkGetMemoryFdKHR`.
   Self-contained, because one `VkDeviceMemory` is one CUDA allocation (§2.2).
   Watch the 2 MiB granularity (§1.1) and keep `vkMapMemory` working for
   host-visible exportable types.
3. **Stop hardcoding CUDA device 0** (`cpvk_device.c:314-316`) and select by
   UUID. Required the moment the renderer and the consumer are not on the same
   GPU.
4. **Honour `VkExternalMemoryImageCreateInfo` properly** (`cpvk_image.c:317-328`)
   rather than using it only to disable the texture cache.
5. **`VK_KHR_external_semaphore_fd`**, only after measuring that the host fence
   wait costs something. Same shape as item 2: `VK_KHR_external_semaphore` and
   `_capabilities` are also promoted to 1.1, so only the `_fd` transport is
   missing. But `VkSemaphore` is a condvar today with no timeline support
   (§2.4), so this is real work, not a matching pair of edits.

---

## Part 5 — Rejected, and why

**Move cudavk to the primary context.** Tempting: `cuDevicePrimaryCtxRetain`
instead of `cuCtxCreate` at `cpvk_device_memory.c:409` would put driver memory
and PyTorch memory in one context and delete the sharing problem in-process, in
about ten lines. Rejected because cudavk calls `cuCtxSynchronize` 2.24 times a
frame from `vkDeviceWaitIdle` (`cpvk_device_memory.c:551-562`), which in a shared
context **drains the consumer's streams every frame**. It also shares cache
config, limits and error state with the rest of the process. It could be revisited
if the `cuCtxSynchronize` calls were replaced with stream-scoped waits, but the
fd path does not need that work.

**Rely on cross-context raw pointers.** Measured working (§1.3) and undocumented.
Fine for a prototype, unsound as an interface, and it carries the silent
`cuCtxDestroy` segfault of §1.4.

**dma-buf export.** Closed on this hardware, not by CUDA:
`docs/cudavk/notes/CUDA13_UPGRADE.md:171-173` records `DMA_BUF_SUPPORTED=0` on
this GeForce and every `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returning
`NOT_SUPPORTED`. `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` is a different
mechanism and it does work (§1.1).

**CUDA IPC.** Cannot bridge two contexts inside one process
(`CUDA_ERROR_INVALID_CONTEXT`, §1.4). It would only ever have addressed the
separate-process case, which the fd already covers.

**`VK_NVX_image_view_handle` + `VK_NVX_binary_import`.** This is how the
proprietary driver hands an image view straight to a CUDA kernel, and it is what
DLSS and the OptiX denoiser use. Not applicable: it is vendor-locked, thinly
documented, and cudavk would have to implement a cubin-launch extension to
match. `VK_KHR_external_memory_fd` achieves the single code path without it.

---

## Part 6 — Traps

- **Depth is not in the depth image during rendering** (§2.3). Read it only after
  `vkCmdEndRendering` with `storeOp = STORE`.
- **cudavk is Vulkan 1.1 with six device extensions.** One binary serving both
  drivers must stay inside that. No timeline semaphores, no buffer device
  address, no `synchronization2`.
- **No implicit stream ordering with a caller** (§2.4), whatever the comment at
  `cp_renderer.c:77-80` says.
- **Context lifetime kills silently** (§1.4).
- **`VkExternalMemoryImageCreateInfo` currently costs performance and buys
  nothing** (§2.1). Until item 4 of Part 4 is done, passing it only disables the
  hardware texture path.
- **The renderer, not the handover, will set the frame rate.** cudavk is
  host-bound at 15.74 ms/frame and 5.87 ms/frame on the two captures. See
  `docs/cudavk/PERFORMANCE.md` before optimising a copy.
- **Bring up the consumer's CUDA context before importing** (Part 3, PyTorch
  section). Importing under a context that later dies gives `CUDA_SUCCESS` and
  then SIGSEGV.
- **`__cuda_array_interface__` does not bounds-check.** A shape larger than the
  buffer is accepted silently.
- **`record_stream()` is a no-op on imported memory.** Use events.

---

## Reproducing Part 1

The programs are small and self-contained; they were built with `gcc ... -lcuda`
plus `-lvulkan` for the baseline, and left in `/tmp/interop/exp/` with their raw
output alongside `RESULTS.md`. The PyTorch experiments are in
`/tmp/interop/torch/` with `TORCH_RESULTS.md`; they are ctypes against
`libcuda.so.1` and need only `torch` in `venv/`. They are not in the tree because they test CUDA
and the proprietary driver, not cudavk. If any of Part 1 is doubted after a
driver or toolkit upgrade, rewrite them — each one is under 200 lines, and the
questions are stated precisely enough in §1.1 to §1.4 to redo from this document
alone.
