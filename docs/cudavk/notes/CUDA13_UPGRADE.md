# Would upgrading to CUDA 13.x help cudapipe?

Short answer: **no, not the upgrade itself.** The research found four things worth doing.
Three of them need **no upgrade of anything**, and the fourth is a header-only toolkit
install that does not touch the driver.

Scope: `src/cudapipe` (the pure Vulkan ICD). The Gallium driver is being deleted.
Every number below was measured on this machine: RTX 5090 (sm_120), driver 580.173.02,
CUDA toolkit 12.8.1, LLVM 18, 2026-08-24. The five supporting reports this summarises are
kept out of tree in `~/cuda13-research/`:

| file | topic |
|---|---|
| `01_release_overview.md` | CUDA 13.0-13.3 release delta, driver ladder, removals, port cost |
| `02_launch_overhead.md` | launch/graph/PDL/green-context host costs, measured |
| `03_compile_codegen.md` | NVRTC, ptxas, PTX ISA 9.0, the compute cache |
| `04_memory_interop.md` | managed memory, dma_buf, stream mem ops, host pools |
| `05_codebase_profile.md` | where this driver's cost actually is, with file:line evidence |

---

## The fact that reframes the question

**The installed driver 580.173.02 already IS a CUDA 13.0 driver.**

    cuDriverGetVersion() = 13000        (verified independently, /tmp/verify/v.c)

It JITs PTX ISA **9.0** and rejects 9.1+. cudapipe hands *text PTX* to `libcuda`
(`cuLinkAddData(CU_JIT_INPUT_PTX)` / `cuModuleLoadData`), and JIT options are plain
integers. Therefore the entire CUDA 13.0 runtime and PTX 9.0 feature set is reachable
**today**, with 12.8 headers. "Upgrading to CUDA 13.0" would buy names for constants that
already work.

---

## Do these (measured, no upgrade required)

### 1. `CU_JIT_SPLIT_COMPILE` — cold JIT 536 ms -> 186 ms (-65%)

Option 34, new in the CUDA 13.0 headers, accepted by the installed r580 driver.
Independently verified on a real 531 KB cudapipe PTX module, JIT cache disabled:

| threads | 1 | 2 | 4 | 8 | 16 | 32 | 0 (auto) | off |
|---|---|---|---|---|---|---|---|---|
| load | 514 ms | 306 | 210 | 180 | **168** | 172 | 178 | 536 ms |

Use `(CUjit_option)34` with value 0 or 16 in `link_shader_module()`.

**Limit, which I measured and the reports did not stress:** with the JIT cache at its
default the *second* load is ~1.1 ms either way. Split-compile only pays on a genuine first
compile — a new shader, a cleared cache, or after a driver upgrade. Each distinct JIT
option set is its own cache entry.

### 2. Raise `CUDA_CACHE_MAXSIZE` — free, and currently at risk

The driver's on-disk cubin cache is the existing cold/warm 2.5x replay factor
(24.4 -> 9.73 ms/frame). It caches cross-process for both `cuModuleLoadData` and `cuLink`:
334 ms cold -> 1.5 ms in a fresh process.

    ~/.nv/ComputeCache = 797 MB, 12,286 files, against the 1 GiB default.

It is about to start evicting, and eviction re-pays the cold JIT bill. Raising the limit
costs nothing. A hand-written cubin cache would largely duplicate this; the only part it
does not cover is the first compile, which is what item 1 attacks.

### 3. `.pragma "enable_smem_spilling"` — occupancy doubled

PTX 9.0. Spills go to shared memory instead of L2, ~10x lower latency. Tested on this
tree's own `cp_fs.cu` PTX, text-patched to `.version 9.0`, JIT-loaded on r580:

| kernel | before | after |
|---|---|---|
| `cp_fs_interpolate` (maxntid 256, MAX_REGISTERS=32) | 150 regs, **1 CTA/SM** | 128 regs, **2 CTA/SM** |
| `cp_abuf_interpolate` | 8 B local mem, 5 CTA/SM | **0 B**, 6 CTA/SM |

This is the one codegen item that survives `SM120.md`'s finding, because it changes *spill
latency and occupancy*, not instruction count — the same class as the `nir_opt_move_to_top`
change that cut a shader 3.64 -> 1.36 ms with identical instruction counts.

Two blockers: it needs explicit launch bounds (without them shared memory hits 43 KB/CTA
and occupancy *drops* 6 -> 2), and `cuLinkAddData` rejects it ("not allowed for
per-function compilation modes"), so the NIR shader path must inline the sampler first.
The whole-program `.cu` kernels can use it now.

### 4. Reopen CUDA graphs — on 12.8, not on 13

Iteration 12 rejected per-tail graphs on one number: 14.8-15.1 us to instantiate. That
number is reproduced (0.38 us/node, 12.1 us at 32 nodes) but it was the wrong number to
measure, because graphs do not have to be re-instantiated:

| | cost |
|---|---|
| graph launch, 4..512 nodes | flat ~1.0 us |
| `cuGraphExecUpdate` | 0.39 us / 32 nodes |
| recapture | 0.155 us / node |
| **per frame: recapture + update + launch, 32 nodes** | **5.3 us** |
| the raw launches it replaces | **43 us** |

An 8x host cut with no cache and no LRU hit rate. Everything needed to kill the
per-variant instantiate — conditional IF/SWITCH nodes, device-updatable kernel nodes,
`cuGraphExecUpdate`, device graph launch — **is already in 12.8**.

Bounded by the frame's real headroom: traced sub-20 us device-idle gaps total 3.287 ms/frame,
and the tree's own estimate for whole-batch graphs is 1-3 ms.

### 5. `griddepcontrol` (PDL) — free, and used zero times

Assembles today from the `.cu` side at `compute_120`. Measured on chained dependent kernels:
stream 4.10 us/kernel -> graph 2.02 (-50.6%) -> PDL 1.89 (-53.9%). **Graphs and PDL remove the
same ~2.2 us dispatch gap; do not add them up.**

This is also the only shape of fix that can work here. The tree already proved that merely
making submission asynchronous pays nothing: `CPVK_ASYNC_SUBMIT=1` removed the submit drain
and made `instancing` *worse* (1.48x -> 1.59x), because the app fences immediately. Only
letting the next chain start before the previous drains can pay.

### 6. `cuMemDiscardAndPrefetchBatchAsync` — 14x on the managed-arena fault storm

New in 13.0, and it already works on r580 via `cuGetProcAddress`. On a 128 MiB managed arena
the host writes and the GPU overwrites: faulting 8.3 ms -> prefetch only 2.67 ms ->
**discard+prefetch 0.55 ms**. It is `LOAD_OP_DONT_CARE` for unified memory, and it maps
directly onto the recorded 1,916 UM-fault events on one 490.7 us launch.
Traps: ~14 us per range, so few large arenas; discard without a following prefetch is a
pessimisation (34 -> 120 ms).

---

## What a toolkit install would add (headers only, driver untouched)

* Names for the constants above instead of raw integers.
* `--Ofast-compile=max` for the NVRTC `.cu` path: `cp_rasterize` 822 -> 77 ms (10.7x), at
  unmeasured kernel-quality cost. 12.9+.
* `#pragma enable_smem_spilling` in `.cu` source rather than a PTX text patch.

Port cost is one line: `cuCtxCreate` becomes the 4-arg `_v4` form, and there is exactly one
call site in `src/cudapipe` (`cpvk_device_memory.c:379`), which already carries a comment
about it. Plus renames (`cuMemPrefetchAsync_v2`, `cuMemAdvise_v2`, `cuEventElapsedTime_v2`).
No driver-API function was removed. Ubuntu 24.04 / glibc 2.39 / gcc 13.3 are all supported.

**But NVRTC 13.3 is a regression uncached: 5.87 s vs 2.73 s over the five kernels**
(`cp_rasterize` 0.82 -> 3.83 s). Install it side by side; do not make it the default blindly.

---

## What upgrading the driver would cost

* **The compute cache is invalidated on every driver upgrade.** With the cold/warm factor at
  2.5x and 797 MB of cached cubins, a driver bump re-pays the entire cold-JIT bill once.
* 13.1 has a known HMM/KASLR init-failure issue, and HMM reads 1 on this box.
* CUDA 13.0 removed nvprof, the Visual Profiler, and several CUPTI APIs (Event, Metric,
  PC-sampling, SASS metrics). The profiling flow here leans on nsys/ncu, so check
  `tests/cp_profile.sh` and the nsys skill pack before moving.

Only three things actually need a newer driver, and none is worth it yet:
13.2 host-task spin-wait `cuLaunchHostFunc_v2` (>=595.45.04; today's blocking callback fires
66-175 us after kernel end, and there are two call sites);
13.3 `cuStreamBeginRecaptureToGraph` (>=610.43.02; optimises a path already costing
0.176 us/node);
13.3 dma_buf mmap (>=610.43.02) — useless here, see below.

---

## Firm negatives — where upgrading buys exactly zero

* **No 13.x release reduces per-launch CPU cost.** No new launch entry point, no driver-side
  batching, no new sync primitive. `cuLaunchKernel` is 1.35 us and stays 1.35 us.
  (`cuLaunchKernelEx` with attributes is 1.18 us — the extended form is not more expensive.)
* **Synchronisation APIs are byte-identical 12.8 -> 13.3.** Eleven functions, zero added.
  `cuStreamWaitValue64(GEQ)` plus a pinned-host doorbell already is a timeline semaphore.
  Note `cuStreamWriteValue32`/`WaitValue32` cost ~900 ns each — not cheap at this driver's rates.
* **External memory/semaphore interop is byte-identical.** Zero-copy WSI is blocked by the
  *device*, not by CUDA: `DMA_BUF_SUPPORTED=0` on this GeForce, and every
  `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returns `NOT_SUPPORTED`. Drop that thread.
* Already in 12.8, so no reason to upgrade for them: all four conditional graph node types,
  device graph launch, device-updatable nodes, every launch attribute, PDL, clusters,
  green contexts + SM partitioning, `cuMemAllocAsync` pools, VMM, `cuStreamBatchMemOp`,
  `cuMemcpyBatchAsync`, NVRTC precompiled headers, `--split-compile`, LTO via `nvJitLink`.
* **ptxas 13.3 vs 12.8 on identical PTX: no win.** +0.5% / +2.3% SASS, registers equal or
  1-4 lower, zero spills both, same wall time.
* Refuted earlier by `SM120.md` and unchanged by any of this: raising the LLVM target to
  sm_120 (LLVM 20+). ptxas already generates SASS for the context's device, so the target
  string buys only PTX expressiveness. And instruction-count wins do not move this driver:
  -49.8% SASS moved the sweep -0.1%.
* `cuCtxSetCurrent` is 10 ns. Multi-queue contexts are not a problem.

---

## Recommendation

1. **Do not upgrade the driver.** It is already CUDA 13.0, and upgrading throws away 797 MB
   of cached cubins.
2. Land the four zero-cost items in this order, each measured with `cp_gpu_busy.sh` first:
   raise `CUDA_CACHE_MAXSIZE`; add JIT option 34; add `griddepcontrol` to the chained `.cu`
   kernels; then graphs with `cuGraphExecUpdate` reuse for whole batches.
3. Try `enable_smem_spilling` on the `.cu` kernels with explicit launch bounds. Watch for the
   43 KB/CTA trap.
4. Install a 13.x toolkit side by side only for headers and `--Ofast-compile`. Keep NVRTC 12.8
   as the default until the 2.1x uncached regression is explained.
5. The honest expected payoff is **1-3 ms/frame of host time**, bounded by the 3.287 ms of
   traced idle gaps, plus a large cut in first-run shader compile latency. None of it comes
   from CUDA 13 as such.
