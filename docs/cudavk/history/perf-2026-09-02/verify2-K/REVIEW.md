# Verifier review — Lead K: host-visible memory residuals post host-pin

Verdict: **OVERSTATED** (mechanism real and measured; ceilings for L2/L3 do not reproduce at the
quoted upper ends; L4's identity is wrong; the topic's largest and cheapest item was missed because the
report called the code that produces it dead). The report's most consequential finding — that
DEAD_ENDS 37's "profiler artefact" premise is itself an artefact — is **SOUND** and reproduced here.

Scratch scripts: `uvm_ident.py`, `uvm_bands2.py`, `gpu_inflate.py` in this directory. All SQL is
read-only against `/tmp/postpin-rtx/trace.sqlite` (post-pin RTX, nsys 2026.4.1, the only post-pin
trace with UVM rows), `/tmp/postdead/trace.sqlite` (pre-pin) and `/tmp/postpin-rtx2/trace.sqlite`.
Frame boundaries from `ts.txt` (rel = ts − TARGET_INFO_SESSION_START_TIME[3]); the render/copy
parity of each submit pair was detected per frame from `cuLaunchKernel` API rows because it flips
between the light and heavy bands (1,100 frames even=render, 343 odd=render) — a naive fixed parity
mislabels the light band.

## 1. Precedent

**DEAD_ENDS 37.** Two separable claims. (a) *Mechanism*: global `SET_ACCESSED_BY` +
`SET_PREFERRED_LOCATION host` costs +1.98 ms RTX / +0.36 B200 — MEASURED, stands, and binds the
report's own "cheapest probe" for L2 (advice=2 on type-2 only measures the remote-UBO-read penalty
37 already paid for, not the value of a shadow+upload design). (b) *Premise*: "the migration exists
only under nsys 2026.1.3" because the 2026.4.1 re-trace showed zero UVM rows. **(b) is refuted by
the traces on disk** (MEASURED):

| trace | UVM rows (kinds 11+12) | = n × 99,999 | last UVM row | last RUNTIME row | tail lost | rows/s in last 3 s | projected rows in tail |
|---|---:|---|---:|---:|---:|---:|---:|
| postdead (pre-pin) | 539,256 + 160,737 = **699,993** | **7 ×** | 24.803 s | 27.892 s | 3.09 s | 24,380 | **75,300 (< 99,999)** |
| postpin-rtx | 43,825 + 56,174 = **99,999** | **1 ×** | 13.176 s | 14.225 s | 1.05 s | 8,330 | **8,700 (< 99,999)** |
| postpin-rtx2 | **0** | 0 × | — | 8.599 s | whole run | (≈68/frame × ~1,280 frames ≈ 87K < 99,999) | |

Every replay exits by SIGSEGV after complete output (`WORKFLOW.md:199` documents exit 139 as
expected), every trace carries "Buffers holding CUDA trace data will be flushed on
CudaProfilerStop()" + "Not all CUDA events might have been collected", and in both traces that have
rows the UVM stream stops at an exact 99,999 multiple while the runtime stream continues for a
tail whose projected row count is a partial buffer. The clean-exit positive control has 25,420
rows (not a multiple). 37's B200 re-trace "zero rows" is therefore expected *whether or not*
migration occurred, and its 2026.1.3 figure (1.9 MB/frame in five regions) matches what
postpin-rtx shows on RTX (0.7-0.78 MB/frame each way, five regions). **37 needs an addendum**: the
mechanism refutation stands; the "artefact" conclusion and the rule built on it ("reproduce under
two profiler builds") should be replaced by: *trace replays with `--duration` ending before the
SIGSEGV (or verify the UVM row count is not a multiple of 99,999) before concluding absence.*
The same loss affects the kernel stream in postdead (kernel rows stop at 12.34 s, runtime at 27.9 s).

**PERFORMANCE §4 prices.** The vid/iid fix below adds bytes to the coalesced flush, not copies
(no 1.30 µs price); the L2 shadow adds one `cuMemcpyHtoDAsync` per submit (~2 µs API + DMA hidden
behind the 165-200 µs frame-head issue); L3 adds ~5 `cuMemPrefetchAsync` calls (~0.01-0.015 ms).
**DEAD_ENDS 8** is irrelevant (flush placement). **TODO 'Performance leads'** lists nothing on
host-visible memory. The report's "why different from 37" holds for L2/L3 (GPU reads stay local),
but it is not evidence of value.

## 2. Arithmetic (RTX, MEASURED unless labelled; spans are lower bounds on stall — they exclude fault entry/exit)

**Where the CPU-side (DtoH) faults land**, heavy band frames 2150-3150 (994 frames after dropping
frames > 3× median), position = render submit interval [A,B) vs boundary [B, next A):

| region (identity) | CPU fault rows/f | prefetch rows/f | position | span mean | span median |
|---|---:|---:|---|---:|---:|
| UBO ring `pMemory_7086` | 12.20 | 3.68 | boundary (app `LoadBinaryData`; preceded by main-thread `cuEventDestroy` +432 µs median) | 0.144 | **0.048** |
| **driver managed scratch: `vfetch_vid/iid`** (`cp_renderer.c:5413-5424`) | 7.79 | 2.26 | **render submit, mid-issue** (preceded by `cuMemsetD32Async` +74 µs median, 3,323/3,323 rows) | 0.081 | **0.056** |
| staging `pMemory_3285` | 3.51 | 3.01 | boundary | 0.040 | 0.010 |
| VB a / VB b | 1.36 each | 1.28 | boundary | 0.006 each | 0.001 |
| `peel_any` 1 page (`0x7bb9fc189000`) | 0.86 | 0 | render (preceded by `cuStreamSynchronize` **+7.6 µs**, 237/237) | <0.002 | — |
| **all CPU-side** | | | | **0.296** (boundary 0.212, render 0.085) | **0.231** (0.129 / 0.060) |

Light band (1794-2237): all CPU-side 0.239 mean / 0.190 median; vfetch 7.38 faults/f, 0.078 / 0.060.
The report's 0.28 mean / 0.21 median is reproduced, but it is **not all in the frame-start gap**:
~30% of it is the driver's own mid-submit faults (1:1 on the frame, the issuing thread stalls), and
the boundary part has a heavy tail (p90 0.518) — the hot-tail *median* frame carries ~0.13 ms.

**Container check.** The report bounds the boundary items by the traced device gap (0.995 ms
median, postpin-rtx2). The untraced boundary interval is **0.553 ms median** (frame-boundary
report §5, pooled three runs; heavy 0.577), containing record ~0.27 + 4.1 MB memcpy ~0.1-0.15 +
DMA 0.075 + hops. That leaves ≤0.15-0.2 ms for CPU fault stall in the boundary, consistent with the
0.129 median span. The report's "~0.5 ms of host-side memory cost inside the gap" is the traced,
mean-based upper edge.

**GPU side.** Kernel-duration inflation is attributable only where a single kernel class is the
sole reader:

| population | with fault | without | per frame |
|---|---|---|---|
| `cp_vertex_fetch` overlapping a scratch-region GPU fault (614/614 such faults land in this kernel) | n=606, **43.6 µs median**, 70.8 mean | n=2,039, 5.7 median, 8.6 mean | 1.75 faults/f heavy → **0.025 (class-median) – 0.065 (raw) ms** |
| first `main` overlapping a UBO-ring GPU fault (1/frame, + 256 KB prefetch) | n=450, 61.1 median | n=35,097, 10.0 median | 0.003 (class) – 0.045 (raw) ms |
| staging→device `cuMemcpy2DAsync` 2,048 B with a copy-engine fault | n=925, **27.4 µs** | n=5,564, 0.6 µs | 2.08/f → **0.056-0.074 ms**, serial on the main stream at the frame head |
| `peel_any` page: GPU faults inside `cp_rasterize_stage2/3` (the store at `cp_rasterize.cu:757`) while the host waits in `cp_sync_timed` | 0.53-0.99/f | | ~0.016 ms class-median inflation, 1:1 (host is blocked on it) |
| VB regions (fused fetch inside `main`) | 4.6 faults/f each | | ≤0.05 ms, **contaminated** by coincidental overlap; unattributable from this trace |

Copy-time floor: 0.085-0.090 ms/frame (reproduces the report). Whole GPU-side burst span 0.064 ms
median (reproduces) but the kernel inflation above shows the span under-counts stall by ~5-10×
where a single reader can be isolated — the report's "true stall HYPOTHESIS 0.3-0.7" is too wide on
both ends; the attributable GPU side is ~0.1-0.2 ms.

**Corrected ceilings (heavy band, driver-collectable, gross of replacement cost):**

| item | mechanism | RTX ms/frame | label |
|---|---|---|---|
| **K6 (new) vid/iid tables → pinned upload arena** (`cp_upload_begin` instead of `cp_scratch_alloc` at `:5413-5424`; a hybrid fallback when the slice is full) | removes 7.8 CPU faults (0.056-0.081, 1:1) + 1.75 GPU faults in `cp_vertex_fetch` (0.025-0.065, side-stream 13/14/17/18 so conversion 0.5-1) | **0.07-0.15** | ESTIMATED from MEASURED |
| L2 UBO ring device + shadow | removes 12.2 CPU faults in the boundary (0.048 med / 0.144 mean) + 1 GPU fault (0.003-0.045); costs 1 copy/submit (~0.005) | **0.05-0.15** (report 0.1-0.3) | ESTIMATED |
| K7 (new) TRANSFER_SRC-only buffers in type 1 host-pinned (the report's §5.3 branch, not in its leads) | copy-engine faults 0.056-0.074 + boundary CPU faults 0.010-0.040 | 0.05-0.10 | ESTIMATED |
| L3 prefetch at submit | GPU side of UBO+VB+staging only, minus ~0.015 API; **overlaps L2** on the UBO ring; does nothing for CPU faults | 0.02-0.10 (report 0.1-0.3) | ESTIMATED |
| L4 as stated (null-desc via `cp_host_ptr`) | **no rows on any null_desc page**; the 1-page region is `peel_any` | ~0 as stated; `peel_any` → `cuMemAllocHost` 0.02-0.04 | MEASURED identity |
| L1 harness readback fill | app-side memcpy, not a UVM residual | 0.15-0.22, **out of the driver topic** | (report) |
| **topic total, driver-collectable** | sum of non-overlapping items | **0.35-0.50 gross, ~0.30-0.45 net** (report 0.5-0.9 incl. L1) | ESTIMATED |

Nothing here clears 0.5 ms; the largest single item is 0.07-0.15. **B200**: no post-pin trace
with UVM rows exists (37's re-trace lost its buffer). Scaling by the 2026.1.3 post-pin copy-time
ratio (0.242 vs 0.085-0.090 → ~2.7×) and PCIe-latency-bound fault service: **0.5-1.2 ms/frame
topic total, K6 0.15-0.4, L2 0.1-0.4** — HYPOTHESIS, and 37's B200 drift rule (three alternating
rounds) applies.

## 3. Feasibility

**K6 (vid/iid).** The code comment at `cp_renderer.c:5395-5410` describes the very failure mode
("written into memory the device then had to fault back page by page") for the *triangle-list*
case and fixed it by deriving ids in the kernel; the strip/fan/point/multi-draw case still
memcpy's two `total_verts × 4` tables into the managed scratch arena, which `cp_scratch_reset`
rewinds to base each flush so the same pages ping-pong every frame (GPU-fault rows are 60 KB
chunks; pages fd202/fd204/fd208 CPU-fault once per frame and never GPU-fault — they ride the
chunk). Replacement: reserve `total_verts × 8` via `cp_upload_begin` and write the ids straight
into the pinned staging (drop the two `malloc` + `memcpy` round trips too). Invariants: the
upload ring is generation-managed (§8 items 4-5) and every `cp_launch` flushes before the
kernel on the same stream (§6.3), which is how every arg block already reaches the side-stream
launches — no new ordering. Blast radius: strips/fans/points/multi-draw only; ids identical by
construction. Gates: native suite, 18-sample sweep (`instancing` retains objects), sentinels;
"memory/copy" → full-frame validation per TESTING §10. Fallback to today's path when the slice
is full (large strip draws). Revert switch required (§8 item 20). ~20-40 lines.

**`peel_any` → pinned.** `cuMemAllocHost` + `cuMemHostGetDevicePointer`; the GPU store at
`cp_rasterize.cu:757` becomes a posted PCIe write, the host read at `:7084` a local read. Racing
writers already store the same value. ~5 lines.

**L2 shadow+upload, correctness.** Host→device: uploading the ranges the submit's descriptors /
binds reference at `vkQueueSubmit` satisfies the implicit host-write availability for
HOST_COHERENT memory (writes after submit are not guaranteed visible to that submit anyway). But
type 2 is advertised DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|HOST_CACHED and `vkMapMemory` hands
out the pointer before any bind, so: (i) every type-2 allocation needs a shadow up front;
(ii) any GPU write into it (STORAGE, TRANSFER_DST, INDIRECT, linear images, query copies) must be
downloaded to the shadow before a fence/`vkWaitForFences`/`vkDeviceWaitIdle` return — Invalidate is
a no-op and HOST_COHERENT forbids requiring one; (iii) UBO ranges of `VK_WHOLE_SIZE`, vertex/index
binds and copy sources in type 2 all need the same upload; (iv) dEQP-VK.memory.mapping.* and
synchronization host-read tests exercise exactly this. Not "~150 lines"; a usage-gated design
(shadow only allocations whose bound buffers are all GPU-read-only, decided lazily at first bind,
with the shadow being the mapped pointer either way) is the minimum. Also the sampler-resolve host
read (`cp_host_ptr`, `:3590`) reads descriptor arenas, not app memory — unaffected.

**L3.** Safe (advisory), but it only re-times the GPU-side migrations and adds API calls; the
CPU-side faults, the larger half, are untouched.

**Double counting with other leads.** The vfetch and peel_any items are inside the render
submit's host-issue path and inside `cp_sync_timed` waits — the same time the closure-audit's
"free syncs" and the episode-drain lead count; the boundary items overlap frame-boundary L1 only
in that both shrink the app's serial gap.

## 4. Cheapest decisive probe

1. One nsys run, `-t cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true` with
   `--duration` ending **before** the replay's SIGSEGV so CUPTI flushes (or check the row count
   is not a 99,999 multiple). Confirms the faulting addresses/callchains for `vfetch_vid/iid`,
   `peel_any` and the UBO ring in one run, and fixes the artefact that produced DEAD_ENDS 37's
   premise. Hours.
2. K6 + `peel_any` behind one `CUDAVK_NO_` revert flag, one alternating A/B session on both
   captures, hashes + submit counts checked. Expected −0.1 to −0.2 ms/frame RTX heavy band, above
   the 0.06 RTX drift; it is the largest item, 1:1 on the host path, and it needs no correctness
   surface. Half a day plus gates. If it lands as predicted, the burst-span → stall conversion is
   calibrated and L2/K7 can be priced from it before either is built.
