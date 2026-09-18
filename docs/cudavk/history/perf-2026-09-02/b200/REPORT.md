# The B200 (sm_100) target: what is known, why it is 1.6x slower, and what a B200-first plan must do differently

Analyst topic: B200. Read-only. 2026-09-02. Tree `/home/alexzhukov/mesa` at
`cudapipe-vk-native` (HEAD `1f2b3c99e00`). Nothing built, nothing run on a GPU,
no file in the repo or home modified. Scratch: this directory
(`ncu_pairs.txt`, `trace_window.txt`, `trace_union.txt`, `rtx_bands.txt`).

Labels: **MEASURED** = taken from a document or a raw file on disk;
**ESTIMATED** = my arithmetic on measured inputs (shown); **HYPOTHESIS** = a
mechanism proposed without a measurement behind it.

---

## 0. Headline

1. Every B200 frame-time number in the tree comes from **two sessions on
   2026-08-31 and 2026-09-01**, on a shared Coder workspace behind two ssh hops,
   with a CUDA 13 build. Total distinct B200 measurements: two frame-time A/Bs
   (host-pin, residency advice), one three-arm screen (wide-merge2 retry), one
   paired ncu run (45 launches, three kernel families), and one llvmpipe
   comparison. Nothing else — no perf profile, no nsys census, no PLAN_STATS
   census, no PDL/fan-out/lane re-measurement, no microbenchmark — exists for
   the B200 (§1, §4).
2. **The paired ncu data on disk contradicts the tree's explanation of the
   slowdown.** `GETTING_STARTED.md:118` says "the B200 gap is clock-bound
   arithmetic on small kernels" and `PERF_HANDOFF.md:45-47` says "clock and
   issue, not architecture". But over the 45 paired launches the B200 executes
   **0.742x the RTX's active SM cycles** (`cp_clip_rast_fused` 0.61x,
   `cp_fs_compact` 0.65x, `cp_rasterize_stage3` 0.92x) while executing the same
   instruction count (proxy ratio 1.08-1.12 ≈ 170/148 SMs). Kernel wall time
   therefore scales as 0.742 × f_RTX/f_B200, which is **0.91x to 1.17x**
   depending on which RTX clock actually held during the replay (2.407 GHz
   CUDA-reported vs 3.09 GHz nvidia-smi max) — nowhere near the **1.615x**
   frame ratio (§2, §3).
3. **The excess is therefore in device idle — host pacing — not in kernels.**
   ESTIMATED: of the +4.11 ms/frame on favorite3, kernel wall contributes
   −0.5 to +0.9 ms and the pacing terms (exposed-launch latency, blocking
   round trips, host API issue on a slower core) contribute +3.2 to +4.6 ms;
   the B200's idle is 2.8-3.5x the RTX's. The slowdown is **multiplicative,
   not a per-frame constant** (5.48 × 1.615 = 8.85 vs measured 8.97 on
   favorite2; a fixed +4.11 would predict 9.59), so it is a per-event price
   paid ~1,100 API calls and ~350 exposed launches per frame (§3).
4. Consequence for the plan: **the price table (`PERFORMANCE.md` §4) is
   sm_120/Ryzen-only and must be re-derived on the B200 before any design.**
   Every closed launch-count lead reopens at a B200 price this report states
   (§4). The six microbenchmarks plus three environment checks in §5 are what
   produces those prices; they cost about a day and no driver code.
5. The B200's 0.3-1.3 ms/frame session drift is 3-10x the size of every
   surviving sub-gate lead, so **the B200 can adjudicate only gate-level
   (≥0.5 ms) effects**; sub-gate decisions stay on the RTX (§6).

---

## 1. Every B200 number in the tree (a)

### 1.1 Frame times

| quantity | value (ms/frame) | source | label |
|---|---:|---|---|
| favorite3, defaults (host-pin on), whole real window | **10.789** (quoted 10.79) | `history/perf-2026-08-31-host-pin-readback.md` table; `PERF_HANDOFF.md:15`; `GETTING_STARTED.md:114`; `PERFORMANCE.md:109` | MEASURED |
| favorite2, defaults | **8.9763** (8.97) | same | MEASURED |
| favorite3, `CUDAVK_NO_HOST_PIN_READBACK` (control arm) | 12.188 | same; `PERFORMANCE.md:109` | MEASURED |
| favorite2, control arm | 10.5662 | same | MEASURED |
| host-pin delta favorite3 / favorite2 | **−1.399 / −1.5899** (RTX: −0.9291 / −1.0657) | same | MEASURED |
| favorite3, DEAD_ENDS 37 control arm (2026-09-01 session) | **10.9151** | `DEAD_ENDS.md:2233` | MEASURED |
| favorite3, residency advice mode 2 | 11.2724 (**+0.36**; RTX +1.98: 6.6413 → 8.6166) | `DEAD_ENDS.md:2232-2233` | MEASURED |
| favorite3, identical mode-2 config, two sessions | **9.80-10.02** in one, **11.14-11.28** in the next | `DEAD_ENDS.md:2253-2255` | MEASURED |
| two-run probe that reversed sign | read −0.65, controlled A/B reversed it | `DEAD_ENDS.md:2255-2256`; `PERF_HANDOFF.md:113-114` | MEASURED |
| favorite3, wide-merge2 retry screen: default band | **10.81-11.14** (within one session) | `DEAD_ENDS.md:1112-1113` | MEASURED |
| favorite3, `CUDAVK_NO_ABUF_APPEND=1` | 10.86 | same | MEASURED |
| favorite3, `NO_ABUF_APPEND` + `CUDAVK_WIDE_MERGE2=1` | 11.01 ("no arm left the session noise band") | same | MEASURED |
| llvmpipe release on the B200 host, favorite3 / favorite2 | **66.19 / 56.41** (cudavk 6.1x / 6.3x faster; RTX host 46.40 / 39.72, 6.9x / 7.2x) | `PERFORMANCE.md:125-130` | MEASURED |
| session drift, B200 vs RTX | **0.3-1.3 vs ~0.06** | `WORKFLOW.md:266`, `:321`; `DEAD_ENDS.md:2253`; `PERF_HANDOFF.md:112` | MEASURED |
| RTX→B200 ratio favorite3 / favorite2 | 10.789/6.6757 = **1.616**; 8.9763/5.4804 = **1.638** | arithmetic on the rows above | ESTIMATED |

No B200 band medians (light/heavy) are recorded anywhere. No B200 mean is
recorded. The 10.79 headline is one session's median; the same driver has
been measured between 9.80 and 11.28 across sessions (see §6).

### 1.2 B200 device-side evidence

| quantity | value | source | label / caveat |
|---|---|---|---|
| pre-pin B200 CUPTI trace: 2D readback copy | 2.2 GB/s, 1.6-1.8 ms for 3.7 MB | `history/perf-2026-08-31-host-pin-readback.md:7-10` | MEASURED |
| pre-pin B200: UVM page traffic | 0.98 ms/frame; ~2.4 ms/frame memcpy-exclusive total | same | MEASURED, **but** taken 2026-08-31, before the 2026-09-01 profiler alignment (DEAD_ENDS 37) — i.e. under nsys 2026.1.3, the build later shown to report migration that 2026.4.1 does not. The −1.40 ms frame delta is a timing fact; its UVM attribution is uncertain |
| post-pin B200: DtoD copy time | 1.454 → 0.323 ms/frame | same, `:40-42` | MEASURED (same caveat) |
| post-pin B200: UVM traffic | 0.98 → 0.24 ms/frame | same | MEASURED (same caveat) |
| DEAD_ENDS 37 "ping-pong": UVM migration through 5 host-visible regions | ~1.9 MB/frame, 4 KB pages, PREFETCH 2.75 GB / COHERENCE 463 MB, 0.242 ms/frame | `DEAD_ENDS.md:2218-2221` | MEASURED under nsys 2026.1.3; **zero rows** under 2026.4.1; positive control (deliberate ping-pong) visible under 2026.4.1 at 640/624 MB — a profiler artefact |
| ncu, both GPUs, matched build | kernels at 0.5-12% of peak SM throughput, 3-25% occupancy; `cp_clip_rast_fused` 0.5% SM, 0.02 waves/SM; `cp_fs_compact` **0.59x cycles at 1.75x IPC** on B200 | `PERF_HANDOFF.md:42-47`; `DEAD_ENDS.md:2275-2277` | MEASURED; raw files `/tmp/ncu-b200-v2.csv` and `/tmp/ncu-rtx/v2.csv` re-analysed in §2 |
| rendering | bit-identical sm_120 vs sm_100, both captures; stdout hash identical | `PERF_HANDOFF.md:19-20`; `WORKFLOW.md:205-206`; `GETTING_STARTED.md:117` | MEASURED |
| false B200 "rendering defect" | favorite2 stored controls disagreed on 9/18 frames — stale controls, failed identically on RTX | `WORKFLOW.md:250-255` | MEASURED |

### 1.3 B200 host and tooling facts

| fact | source |
|---|---|
| Reach: `ssh azhukov@192.168.1.154 ssh b200`; Coder workspace `standard.stratus23.azhukov.coder`; home persists, **system packages and `/tmp` do not** | `PERF_HANDOFF.md:97-101`; `/tmp/b200-xfer/run.sh` |
| CPU: 152-thread Xeon Platinum 8559C; llvmpipe there is *slower* than on the 32-thread Ryzen despite 4.75x threads | `PERFORMANCE.md:125-129` |
| Build: CUDA 13.0 toolkit (`cuCtxCreate` v4 guard), same nsys 2026.4.1 and ncu 2025.3.1 builds after alignment; `RmProfilingAdminOnly: 1` but CAP_SYS_ADMIN → `sudo ncu` works | `PERF_HANDOFF.md:97-99`; `GETTING_STARTED.md:17-18` |
| B200 evidence directories `/tmp/f3-b200/`, `/tmp/f2-b200/` live on the B200 host's `/tmp`, which does not persist | `history/perf-2026-08-31-host-pin-readback.md:50-52` |
| The two B200 commits in this tree: `3cc7741adb8` (wide-merge retry screen), `9051520863c` (DEAD_ENDS 25 prices are dev-box facts) | `git log` |

### 1.4 What was and was not measured on the B200 (confirmed from docs)

| mechanism | RTX delta (ms/frame) | B200 | source |
|---|---:|---|---|
| host-pinned readback | −0.929 / −1.066 | **−1.399 / −1.590** (×1.5) | host-pin history doc |
| residency advice mode 2 | +1.98 | **+0.36** | DEAD_ENDS 37 |
| wide merge, multi-stream retry | RTX original +0.410 | **no signal** inside 10.81-11.14 band | DEAD_ENDS 20 retry |
| opaque fan-out (`CUDAVK_NO_OPAQUE_STREAMS`) | surrender costs 0.66-0.84 (both captures, RTX) | **never measured** | DEAD_ENDS 36; `PERFORMANCE.md:66-70` |
| PDL level 3 (`CUDAVK_NO_PDL`) | +0.3815 old / +0.1522 Crossroads (RTX) | **never measured** | `PERFORMANCE.md:108` |
| 16/32 pass lanes | −0.084 / −0.092 favorite2 (RTX) | **never measured** | DEAD_ENDS 38 ("RTX 5090") |
| CDP2 / device chaining prices | host link 1.48-1.51 µs, tail 7.3-8.7 µs | **never measured** ("no number in this entry has been measured there") | DEAD_ENDS 25 |
| scratch high-water reuse | +0.037 / +0.012 | **never measured** ("Retry if. Only on B200") | DEAD_ENDS 29 |
| exact vertex reuse census | 0.438 ceiling | **never measured** ("or a B200 clean trace") | DEAD_ENDS 35 |
| fused-VS content census | no concentration (largest key 0.209) | **parked pending B200** | `notes/SHADER_CONTENT_CENSUS_RESULT.md:3,37-43`; `notes/RESIDUAL_AUDITS_2026-08-31.md:14-15,85` |
| small-op removals, fused vfetch, A-buffer fusions, texture cache | old/Crossroads RTX only | **never measured** | `PERFORMANCE.md:97-109` |
| host perf profile, PLAN_STATS wait census, nsys launch census | RTX only | **none exists** | `notes/RESIDUAL_AUDITS_2026-08-31.md`; `history/perf-2026-08-30-post-dead-scope-profile.md` |

A naming trap for whoever looks at `/tmp` on the RTX box: `/tmp/opt16_b200_*.txt`
and `/tmp/opt43_b200_*.txt` are **not B200 runs**. `/tmp/opt16.sh` and
`/tmp/opt43.sh` show `b200` = `CUDAVK_WAIT_SPIN_BEFORE_US=200` (arms `b0 b100
b200 b400 a100`), run on the RTX. No B200 submit-timestamp file exists on this
machine.

---

## 2. The paired ncu data, re-analysed (new)

### 2.1 Provenance

`/tmp/ncu-rtx/run2.sh` (RTX, 2026-09-01 11:42) and the B200 run whose output is
`/tmp/ncu-b200-v2.csv` (process `/home/builder/favorite3-cpp/out/build/vulkan_app`,
CC 10.0) used the same filter and window:

```
ncu -k 'regex:cp_rasterize_stage3|cp_fs_compact|cp_clip_rast_fused' \
    --launch-skip 60000 --launch-count 45 \
    --metrics sm__cycles_active.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,
              gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,
              sm__warps_active.avg.pct_of_peak_sustained_active,
              smsp__inst_executed.avg.per_cycle_active,
              launch__waves_per_multiprocessor,launch__registers_per_thread --csv
```

The replay is deterministic, so launch IDs 0-44 are the same 45 kernels with the
same grids on both GPUs (asserted in the script that produced `ncu_pairs.txt`).
ncu locks clocks to base and flushes caches by default, so the cycle counts are
clock-independent and comparable; they say nothing about the clock that holds
during a real replay.

### 2.2 Aggregate cycle ratios (MEASURED, from the two CSVs)

| kernel | n | Σ `sm__cycles_active.avg` RTX | Σ B200 | **B200/RTX cycles** | Σ (cycles × IPC) ratio ("instruction proxy") |
|---|---:|---:|---:|---:|---:|
| `cp_rasterize_stage3` (grid 2048×64) | 11 | 249,961 | 229,034 | **0.916** | 1.125 |
| `cp_clip_rast_fused` (grids 5-1825) | 11 | 268,446 | 164,625 | **0.613** | 1.077 |
| `cp_fs_compact` (grid 900×256) | 11 | 211,234 | 136,597 | **0.647** | 1.114 |
| `cp_clip_rast_fused_abuf` | 6 | 9,722 | 9,129 | 0.939 | 1.044 |
| `cp_rasterize_stage3_abuf` | 6 | 37,097 | 36,718 | 0.990 | 1.091 |
| **all 45** | 45 | 776,459 | 576,102 | **0.742** | — |

Reading: `sm__cycles_active.avg` averages over all SMs (170 on RTX, 148 on
B200 — §2.3), so with the same total work the B200 must do 170/148 = 1.149x the
instructions per SM. The instruction-proxy ratio (1.04-1.13) confirms the same
code did the same work; the cycle ratio below 1 says each B200 SM retired it in
fewer cycles, at 1.2-2.7x the RTX's IPC (per-launch table in `ncu_pairs.txt`).
That is what a latency-bound kernel looks like when the clock drops: the memory
latency in nanoseconds is roughly constant, so it is fewer cycles at a lower
clock.

Per-launch extremes worth knowing: `cp_clip_rast_fused` grid 1825 — 138,062 →
71,397 cycles (0.52x) at IPC 0.08 → 0.17; `cp_fs_compact` ID 43 — 0.45x at IPC
0.13 → 0.35. The 5-92-block `cp_clip_rast_fused` launches (IDs 3, 5, 7, 11, 13,
15, 41, 44) are 0.88-1.10x — those grids occupy under 3% of either SM array,
so the SM-count normalisation does not apply and per-block cycles are ~equal.

### 2.3 Derived hardware facts (ESTIMATED from `launch__waves_per_multiprocessor`)

`waves = grid_blocks / (SMs × max_resident_blocks_per_SM)`.

- `cp_rasterize_stage3`, 2048 blocks × 64 threads, RTX 43 regs, waves 0.60 →
  2048/(170 × 0.60) = **20.1 blocks/SM** on RTX. B200, 44 regs, waves 0.69 →
  SMs = 2048/(20 × 0.69) = **148.4**.
- `cp_fs_compact`, 900 × 256, RTX 38 regs, waves 0.88 → 900/(170 × 0.88) = 6.0
  blocks/SM (warp-limited: 48 warps / 8). B200, **48 regs**, waves 1.22 → at the
  register limit 65536/(48 × 256) = 5.33 → 5 blocks/SM, SMs = 900/(5 × 1.22) =
  **147.5**.

So the B200 here has **148 SMs**, not more than the RTX's 170. The DEAD_ENDS 20
retry text ("the SM array twice as wide", `DEAD_ENDS.md:1112`) is **wrong**; the
B200's per-SM resources are larger (64 warps and 228 KB shared per SM vs 48
warps / 99 KB on sm_120, per `/tmp/perf16/b200-arch-audit.md` assumption 22),
and its FP32 peak at 148 × 128 × 2 × ~1.97 GHz ≈ 75 TFLOPS is **below** the
5090's 170 × 128 × 2 × 2.407 ≈ 105 TFLOPS (ESTIMATED). None of that matters to
kernels running at 0.5-12% of peak.

Two more differences visible in the CSVs:

- **Register allocation differs** (different JIT: CUDA 13 NVRTC / sm_100 ptxas):
  `cp_clip_rast_fused` 56 → 48, `cp_fs_compact` 38 → **48** (which is why its
  occupancy became register-limited on B200), `cp_rasterize_stage3` 43 → 44,
  `*_abuf` 46/56 → 48. The generated `main` shaders are JITted from `.target
  sm_86` PTX on both, with per-arch register results the tree has not recorded
  for sm_100 (`SHADER_CONTENT_CENSUS_RESULT.md:37-43`).
- **DRAM % of peak is 4-6x lower on B200** (`cp_fs_compact` 27-45% → 5-8%),
  consistent with HBM3e's ~8 TB/s against GDDR7's 1.79 TB/s
  (`TARGET_INFO_GPU.memoryBandwidth` = 1,792,128,000,000 in the trace). No
  kernel here is bandwidth-bound on either part.

### 2.4 Implied kernel wall-time ratio (ESTIMATED)

`t = cycles / f_SM`, so `t_B/t_R = 0.742 × f_R/f_B`.

| RTX clock assumed | B200 clock assumed | kernel wall ratio |
|---|---|---:|
| 2.407 GHz (`TARGET_INFO_GPU.clockRate` in `/tmp/postdead/trace.sqlite`) | 1.965 GHz (commonly reported B200 max; **not verified on that host**) | **0.91** |
| 2.8 GHz | 1.965 | 1.06 |
| 3.09 GHz (`nvidia-smi` `clocks.max.sm` on this RTX) | 1.965 | **1.17** |

Neither GPU's SM clock during a replay has ever been sampled. Applied to the
RTX heavy-band exclusive pools these three families own (`cp_clip_rast_fused`
0.592 + `cp_rasterize_stage3` 0.424 + `cp_fs_compact` 0.372 = 1.388 ms/frame,
`PERF_HANDOFF.md:34-38`), the B200 delta on them is **−0.12 to +0.24 ms/frame**.
The generated VS + FS (0.689 + 0.678 = 1.367) are unmeasured on B200: if they
scale like these kernels the delta is ~0; if they scaled with the raw clock
ratio (1.57x, the "clock-bound arithmetic" model) it would be +0.78. Total
kernel-side delta: **−0.5 to +0.9 ms/frame** against an observed **+4.11**.

---

## 3. Why the frame is 1.6x slower: the mechanism (b)

### 3.1 The RTX frame's event structure — the multipliers

From `/tmp/postdead/trace.sqlite` (RTX, commit `f3caa35e4ce`, CUPTI on), window
9.276-12.341 s = 412.5 frames (frames ~1393-1804), all MEASURED. SQL in §8.

| per frame | count | CUPTI-traced time | note |
|---|---:|---:|---|
| kernels | **427.2** | summed 4.968 ms, union 3.335 (overlap 1.49) | 9 streams (main + 8 lanes) |
| kernels that start with the **whole device idle** | **348.4 (81.6%)** | idle gap before them: p50 1.89 µs, p90 11.8 µs | 341.8/frame have gaps <20 µs summing **1.203 ms/frame**; 6.5/frame have gaps ≥20 µs summing **2.89 ms/frame** |
| kernel duration | — | p10 1.41, p50 3.78, p90 23.1, mean 11.6 µs | latency shells |
| union of kernel+memcpy+memset | — | **4.509 ms** of a 7.430 traced frame (idle 2.921) | reproduces the doc's 4.510 |
| `cuLaunchKernel` + `cuLaunchKernelEx` | 268.0 + 159.2 | 0.584 + 0.321 ms; 2.18 / 2.02 µs mean | CUPTI-inflated |
| `cuMemcpyHtoDAsync_v2` | **140.8** | 0.262 ms; 1.86 µs mean (device side 0.94 µs, 0.306 MB/frame) | descriptor/upload traffic |
| `cuMemsetD32Async` + `D8Async` | 89.0 + 15.2 | 0.217 ms | |
| `cuEventRecord` / `cuStreamWaitEvent` | 220.5 / 153.4 | 0.108 ms | fan-out gates and joins |
| **blocking calls**: `cuStreamSynchronize` / `cuCtxSynchronize` / `cuEventSynchronize` / sync `cuMemcpyDtoH_v2` | **11.40 / 2.04 / 2.00 / 8.92 = 24.4** | 1.272 / 1.596 / 0.658 / 0.065 = **3.59 ms** | mean waits 112 / 783 / 329 / 7.3 µs |
| all CUDA API rows | ≈1,100 | 5.351 ms (CUPTI-inflated) | |
| DtoD copies | 46.7 | 0.768 ms device, 4.06 MB, mean 16.4 µs | |

The untraced median for the same frames is **5.78 ms** (`/tmp/pin-rtx-ab/favorite3-p1..3/ts.txt`,
frames 1393-1804: 5.782 / 5.764 / 5.746), so CUPTI inflates this window by
**1.28x**; the counts are what to trust. Consistency check: if the inflation is
host-side, untraced union ≈ 4.5 and idle ≈ 1.3 ms → 78% busy, which is exactly
what `nvidia-smi` sampled without a profiler (`history/perf-2026-08-30-post-dead-scope-profile.md`).

Host side without CUPTI (`notes/RESIDUAL_AUDITS_2026-08-31.md`, means over
2,078 frames): CUDA wait spin 4.436, CUDA issue APIs 1.498, driver planning
0.868 (0.33 steady), app record/decode 0.376, unattributed 0.371, other CUDA
0.190 ms/frame. Main thread on-CPU 97% of wall. Mean RTX frame in the pinned
arm is 7.37-7.41 (`rtx_bands.txt`), median 6.64-6.69.

RTX favorite3 shape for scaling (MEASURED, `rtx_bands.txt`, pinned arm p1-p3):
whole median 6.637-6.693, light (1700-2100) 3.94-3.98, heavy (2200-3150)
8.21-8.26 with p10 7.0-7.1 and p90 9.7-9.8; 77-83 frames ≥10 ms per run.

### 3.2 First-principles scaling factors, RTX → B200

| term | RTX (desktop: Ryzen 9 9950X3D 5.7 GHz, PCIe 5 x16 direct to CPU) | B200 host (Xeon Platinum 8559C in a shared container, PCIe 5 through a datacenter topology) | factor | label |
|---|---|---|---:|---|
| SM clock | 2.407 GHz reported, up to 3.09 | ~1.9-2.0 GHz (1.965 commonly reported), **unverified**; may sit lower under a bursty 10%-utilisation load | 1.22-1.57 | ESTIMATED |
| kernel cycles for this code | 1.00 | **0.742** (measured, §2) | — | MEASURED |
| kernel wall | 1.00 | 0.91-1.17 | — | ESTIMATED |
| CPU single-thread speed on the driver's issue path (`cuLaunchKernel` 1.35-1.43 µs CPU, `notes/CUDA13_UPGRADE.md`, DEAD_ENDS 25) | 1.00 | Zen 5 at 5.7 GHz vs a server core at ~2.1 base / ~3.5-3.8 turbo, possibly quota-throttled | **1.4-1.8x slower** | ESTIMATED (typical) |
| host launch → device start, empty queue | ~2.0 µs floor (`PERFORMANCE.md:258`: 1.974 µs exposed launch, 2.047 back-to-back) | datacenter parts through PCIe switches typically 4-8 µs; virtualised/containerised hosts can add more | **2-4x** | ESTIMATED (typical, unmeasured) |
| completion → host visible (sync round trip, spin poll) | ~5-10 µs | ~10-25 µs (PCIe read latency 1.5-3 µs through a switch and possibly across a UPI hop vs ~0.7-1 µs direct) | 2-3x | ESTIMATED (typical, unmeasured) |
| small HtoD async copy issue | 1.86 µs API / 0.94 µs device (trace) | unknown; managed-memory-backed sources add page-state checks on a 2-socket host | 1-3x | HYPOTHESIS |
| DRAM latency in ns | GDDR7 ~500-600 ns | HBM3e ~550-700 ns; **fewer cycles** at the lower clock, matching the 1.2-2.7x IPC rise | ~1.0-1.2x ns | ESTIMATED |
| copy bandwidth | 1.79 TB/s | ~8 TB/s | 0.22x (helps DtoD copies) | MEASURED spec |
| PCIe readback of 3.7 MB/frame | ~75-100 µs at ~40-50 GB/s | same if x16 Gen5 and pinned; **pre-pin it ran at 2.2 GB/s (1.6-1.8 ms)** | 1x if the link is x16 Gen5 | MEASURED pre-pin, post-pin unknown |

"Equal or fewer cycles at a lower clock" means: these kernels are bound by
dependent-latency chains measured in nanoseconds (memory, atomics, launch
ramp), not by issue throughput. A lower clock converts the same nanoseconds
into fewer cycles and leaves wall time about constant. It does **not** mean the
kernels shrink if fed more work, and it does not make the frame clock-bound.

### 3.3 ESTIMATED B200 decomposition of favorite3 (whole window medians)

Inputs: RTX 6.68 = device union U_R + device idle I_R. From the traced window
(union 4.51 of 7.43, inflation 1.28) and the heavy band (union 5.21 of 8.40,
idle 3.19, `PERF_HANDOFF.md:34-36`), the whole-window untraced split is
U_R ≈ 4.9 ± 0.4, I_R ≈ 1.8 ± 0.4 (ESTIMATED).

| term | RTX (ms/frame) | B200 (ms/frame) | derivation | dominates? |
|---|---:|---:|---|---|
| kernel + copy union | 4.9 ± 0.4 | **4.5-5.7** | U_R × (0.91..1.17) from §2.4 | no: −0.4 to +0.8 |
| device idle = host pacing | 1.8 ± 0.4 | **5.1-6.3** | 10.79 − union | **yes: +3.3 to +4.5 of the +4.11** |
| of which, exposed-launch gaps | ~0.7 (≈350 × ~2 µs) | 350 × L_B | L_B is unmeasured; 4-8 µs gives 1.4-2.8 | |
| of which, blocking round trips (24.4/frame) + host decision time | ~1.1 | 24.4 × s_B + host time × r_cpu | s_B unmeasured; 20-40 µs gives 0.5-1.0; host 0.7 × 1.4-1.8 = 1.0-1.3 | |
| **total** | 6.68 | 10.79 | | |

The idle term is 2.8-3.5x the RTX's. Splitting it further requires the B200
prices (§5); the table shows that plausible typical values (L_B 4-8 µs, s_B
20-40 µs, CPU 1.4-1.8x) do close the gap without any kernel slowdown, and that
if the B200 clock turns out to hold near its maximum, the kernel term may even
be slightly negative.

Cross-check against favorite2 (MEASURED inputs): a pure multiplicative penalty
predicts 5.4804 × 1.616 = 8.86 vs measured 8.976 (Δ 0.12, inside B200 drift); a
fixed per-frame penalty predicts 5.4804 + 4.11 = 9.59 (Δ 0.62, outside). The
slowdown is proportional to the number of events in the frame, not a per-frame
constant such as the readback or one frame-boundary sync.

Where CUPTI-era conclusions transfer and where they do not: the RTX host
already spins 4.4 ms/frame and its device is empty 22-38% of the time, so the
"ping-pong, not a pipeline" shape (`PERFORMANCE.md` §5.2) is the same on both
hosts — the B200 simply pays more per turn. Everything closed on RTX by the
*merge rule* (concurrency, DEAD_ENDS 22) stays closed on B200 because overlap is
a property of the schedule, not the host. Everything closed on RTX by *price*
must be re-priced.

### 3.4 What the two host-scaled measurements say

- **Host-pin readback**: −0.93 RTX vs −1.40 B200 (×1.5). The pre-pin B200 copy
  ran at 2.2 GB/s — a UVM-migrating 2D copy over a datacenter PCIe path. The
  B200 gained more because its bus-side penalty per byte was larger. This is
  the one B200-first win so far and it is a **bus/latency** effect, not a
  kernel effect — consistent with §3.3.
- **Residency advice**: +1.98 RTX vs +0.36 B200. The same `cuMemAdvise` policy
  costs the desktop 5.5x more. UVM/HMM placement behaviour differs between the
  two hosts (a 2-socket NUMA Xeon with HMM against a single-socket desktop), so
  **no memory-policy conclusion transfers in either direction**
  (`PERF_HANDOFF.md:118-120` already says this rule).

---

## 4. Which closed leads reopen on the B200, and at what price (c)

Notation: `p_B` = B200 price of a removed exposed launch (RTX 0.78-0.81 µs
removal, 1.974 µs added-exposed); `s_B` = B200 price of one blocking round trip
(device completion → host sees it → host issues the next launch → device
starts); `r_cpu` = B200/RTX host single-thread slowdown. The admission gate is
0.500 ms/frame (`notes/RESIDUAL_AUDITS_2026-08-31.md`). Overlap factors are
schedule properties and carry over unchanged.

| closed lead | RTX verdict | B200 reopen condition (ESTIMATED) | comment |
|---|---|---|---|
| DEAD_ENDS 21 — A-buffer count-phase merge | 554 launches × 0.782 / 1.59 overlap = 0.27 ms before a device cost measured larger | reaches the gate when 348 × p_B − devcost_B ≥ 0.5 → **p_B ≥ 2.0-2.3 µs** (devcost 0.2-0.3) | refuted before building on RTX; on B200 it needs the microbenchmark first |
| DEAD_ENDS 20 — wide merge | removed 298.8 launches, cost +0.410; overlap 2.19 | B200 retry already run: **no signal** inside a 0.33 ms session band; bound 0.07 | the only existing B200 constraint on `p_B`: 136 effective launches × p_B minus the merged form's device cost is < ~0.3 ms — weak, because the device cost was not re-measured |
| DEAD_ENDS 25 — CDP2 tail chaining | 45.4 host launches → 0.035-0.037; device tail 7.3-8.7 µs vs host link 1.5 | pays only if `tail_B < link_B`; reaches the gate only if `link_B − tail_B ≥ 11 µs` — implausible | the entry itself demands the microbenchmark on sm_100; run it, expect a sub-gate answer |
| DEAD_ENDS 25 — conditional graphs / predicated pre-issue | unit 0.037 | unit scales with `s_B`: 24.4 round trips × (s_B − s_R); **gate at s_B − s_R ≥ 20 µs** | the read-back-and-decide sites (drain 3.74, seg counters 1.47, peel 1.13, desc 1.00 waits/frame, `history/perf-2026-08-30-post-dead-scope-profile.md`) are where B200 latency is paid per turn |
| DEAD_ENDS 1 — CUDA graphs | 0.63-0.68 ceiling (sub-20 µs gap pool) vs 27k-44k instantiations, 65,905 keys | ceiling scales with the exposed-gap pool: 342 gaps/frame × L_B; **1.4-2.8 ms if L_B = 4-8 µs**; instantiation cost also scales ×r_cpu | the `cuGraphExecUpdate`/recapture path (5.3 µs vs 43 µs per 32 nodes, `notes/CUDA13_UPGRADE.md` §4) was never built; TODO item 4 keeps it open. Largest *potential* B200 lead if L_B is confirmed large; largest build |
| TODO 7 — opaque FS UBO row concatenation | 60 launches → 0.04-0.06 | 60 × p_B: **gate needs p_B ≥ 8.3 µs** — no; at p_B 3 µs it is 0.18 | measure its overlap factor first, as TODO says |
| DEAD_ENDS 34 — threaded submit | 0.126 | ≈ 0.126 × r_cpu = **0.18-0.23** | sub-gate on both |
| DEAD_ENDS 29 — scratch high-water reuse | removes 1.65 ms blocked, pays 0.037 | ~2.04 `cuCtxSynchronize`/frame × s_B → **0.05-0.10** | its own retry clause says "only on B200, ≥4 alternating pairs"; cheap, sub-gate |
| DEAD_ENDS 38 — 16 lanes | −0.084 favorite2, declined for +153 MB | **the cost side vanishes**: 153 MB on a ~180 GB part; value unknown on B200 (fewer SMs, more warps/SM) | free re-measure; sub-gate unless B200 overlap behaves differently |
| DEAD_ENDS 36 — fan-out surrender 0.66-0.84 | load-bearing default | never measured on B200; the hand-off price (32 µs/episode, `PERFORMANCE.md:265`) is a latency price and will differ | confirm sign and size before any redesign that touches streams |
| PDL level 3 | +0.38 / +0.15 | never measured on B200; per-link 0.44-3.02 µs are latency prices | confirm; try level 4 (`b9766f720a4` diagnostic) |
| DEAD_ENDS 37 — residency advice | +1.98 / +0.36 | **refuted on both**; do not retry in that form | but the asymmetry motivates a plain `CUDAVK_SMALL_ALLOC` mode A/B on B200 (HYPOTHESIS, unknown sign) |
| DEAD_ENDS 16 — `CUDAVK_CTX_SCHED` | blocking +5.4%, yield neutral on RTX idle; `auto` → spin | `auto` resolves to spin on 152 LPs too; a quota-throttled container could change `yield`'s value | cheap flag A/B, low expectation |
| kernel-side leads (24, 30-33, 35, 36, fused-VS census) | pools below gate | §2 says B200 kernels take ≤ RTX cycles; nothing reopens here | the fused-VS census was parked "pending B200" — the ncu evidence says do not expect a kernel-side prize |

**Ranking of what would move the B200 most** (HYPOTHESIS-ranked by §3.3; each
item's number is what the §5 measurements will replace):

1. **Environment**: the B200's SM clock during replay, PCIe link width/gen,
   NUMA placement of the process against the GPU's socket, container CPU
   quota. Zero code; any one of these being wrong is worth up to the whole
   excess. Never checked.
2. **The price table** on B200 (§5). Zero code; decides everything below.
3. **Launch batching via graph recapture/update** — reopens DEAD_ENDS 1 on a
   ceiling of 342 exposed gaps/frame × L_B; only if L_B measures ≥ ~3 µs.
4. **Round-trip removal at the read-back-and-decide sites** (conditional graph
   nodes or device-side decisions) — reopens DEAD_ENDS 25 if `s_B` ≥ 20 µs over
   RTX.
5. **Re-measure the load-bearing defaults** (fan-out, PDL level, lanes) — the
   hand-off and PDL prices are latency prices; DEAD_ENDS 38's memory objection
   does not exist on a 180 GB part.
6. Memory-policy flag A/Bs (`CUDAVK_SMALL_ALLOC` modes) — unknown sign; cheap.
7. Everything kernel-side: nothing.

---

## 5. What the owner should microbenchmark on the B200 before deciding (c)

`docs/cudavk/notes/cdp2-microbench.cu` is the template (512-link chains,
20 reps, best/mean, `cudaEvent` timing, plus a CPU-clock API-cost loop). Extend
it with the arms below, build once with the CUDA 13 toolkit on that host, run
pinned to the GPU's NUMA node, and run **each arm three times interleaved**
(the drift applies to microbenchmarks too if its cause is environmental).

| # | measurement | RTX reference (sm_120, this box) | decision it feeds |
|---|---|---|---|
| 0a | **SM clock during a replay**: `nvidia-smi dmon -s pc -d 1` (or `--query-gpu=clocks.sm,clocks_throttle_reasons.active` at 10 Hz) alongside a full favorite3 run; also `nvidia-smi -q -d CLOCK,PERFORMANCE` for application clocks and throttle reasons | never sampled on either GPU; RTX max 3090 MHz, CUDA-reported 2407 | fixes f_B and f_R in §2.4; if the B200 idles low under bursty load, `nvidia-smi -lgc`/application clocks is the first lead |
| 0b | **PCIe link**: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`; `bandwidthTest`-style pinned DtoH/HtoD for 3.7 MB | RTX Gen5 x16 | the per-frame readback and 141 HtoD copies |
| 0c | **NUMA/CPU**: `nvidia-smi topo -m` (CPU affinity of the GPU), `numactl -H`, `cat /sys/fs/cgroup/cpu.max`, `/proc/cpuinfo` MHz, `/proc/stat` steal during a run; then A/B a replay with `numactl --cpunodebind=<gpu node> --membind=<gpu node>` | single socket, no quota | drift cause (§6) and a possible zero-code win |
| 1 | **host launch-to-start latency**: empty kernel on an idle stream, `%globaltimer` at kernel entry minus host `clock_gettime` before `cuLaunchKernel` (clocks aligned once via a spin-sync kernel), and the exposed-launch slope method of `PERFORMANCE.md:258` | 1.974 µs exposed-launch slope; 2.047 µs back-to-back floor | L_B for §3.3 and DEAD_ENDS 1/21/TODO 7 reopen prices |
| 2 | **back-to-back dependent kernel gap, with and without PDL** (`griddepcontrol.launch_dependents` / `wait`), one stream, 512 links | host-issued link 1.48-1.51 µs (DEAD_ENDS 25); stream 4.10 → PDL 1.89 µs chained (`CUDA13_UPGRADE.md` §5); PDL link value 0.44 / 0.78 / 3.02 µs by secondary shape (`PERFORMANCE.md` §4) | whether PDL level 3/4 still pays; `p_B` |
| 3 | **DtoH sync round trip**: kernel writes 4-64 B, host `cuMemcpyDtoH` (sync) and separately `cuMemcpyDtoHAsync` + `cuStreamSynchronize`, and a pinned-host flag polled by the CPU; report device-end → host-visible latency | trace: `cuMemcpyDtoH_v2` API mean 7.29 µs, `cuStreamSynchronize` on an idle stream not isolated | `s_B` for the 24.4 blocking calls/frame; DEAD_ENDS 25/29 reopen |
| 4 | **small `cuMemcpyHtoDAsync` latency**, 16-256 B, from pageable, pinned and managed sources, issue cost and device-visible completion | API 1.86 µs mean, device 0.94 µs mean (trace) | the 141/frame descriptor uploads; whether the managed-memory arena (`CUDAVK_SMALL_ALLOC`) is the wrong default there |
| 5 | **conditional-graph-node evaluation latency**: IF node keyed on a device flag set by the previous kernel, chain of N; time per node vs a host-decided chain | not measured on RTX (DEAD_ENDS 25 closed on unit size, not on price) | DEAD_ENDS 25 reopen for the read-back-and-decide sites |
| 6 | **`cuStreamWaitValue32` release latency**: host writes a pinned doorbell, stream waiting on it releases a kernel; device-side `%globaltimer` at release minus host write time; also the API cost | ~900 ns API cost (`CUDA13_UPGRADE.md`, "not cheap at this driver's rates"); release latency unmeasured | host-run-ahead designs (PERFORMANCE §6 item 4) |
| 7 | **CDP2 tail / fire-and-forget link** (already in the template) | 7.32-7.44 / 8.65-8.73 µs | DEAD_ENDS 25 on sm_100 |
| 8 | **`cuLaunchKernel` CPU cost with the stream busy** (in the template) | 1.43 µs (DEAD_ENDS 25), 1.35 (`CUDA13_UPGRADE.md`) | `r_cpu` for host issue (1.5 ms/frame on RTX) |
| 9 | (optional) **DRAM pointer-chase latency** in ns on both GPUs | not measured | validates the latency-bound reading of §2 |

Deliverable from this day: a B200 column for `PERFORMANCE.md` §4. Until it
exists, no B200 forecast is admissible — the tree's own rule
(`PERFORMANCE.md:245-248`: "a model that is not one of these has never survived
contact with a measurement") applied to a host where none of these were measured.

---

## 6. The B200's session drift and how it must shape any A/B there (c)

MEASURED facts: drift 0.3-1.3 ms/frame between sessions (`WORKFLOW.md:266`);
the identical configuration measured 9.80-10.02 in one session and 11.14-11.28
in the next (`DEAD_ENDS.md:2253-2255`); within one session the default arm
spanned 10.81-11.14 (`DEAD_ENDS.md:1112-1113`); a two-run probe read −0.65 and
the controlled A/B reversed its sign; RTX drift ≈0.06 (three pinned runs here:
6.637 / 6.693 / 6.676, `rtx_bands.txt`).

Consequences:

1. **Quote every B200 number with its session band.** "10.79" is one session;
   the honest statement is "10.79, with the same driver measured 9.8-11.3
   across sessions". The RTX→B200 ratio is therefore 1.47-1.69, not 1.615.
2. **Minimum detectable effect.** With a within-session band of ~0.33 ms and
   three alternating rounds, only effects ≥ ~0.3-0.5 ms/frame can be resolved
   with disjoint arms. Every surviving sub-gate lead (0.08-0.13 ms) is
   **unmeasurable on the B200**. Decide sub-gate items on the RTX; use the
   B200 only for gate-level questions and for the prices in §5.
3. **Design**: strictly alternating pairs, ≥3 rounds (4-5 preferred), arms
   disjoint, per-pair sign consistency required, control runs at the start and
   end of the session as drift monitors, all in one process sequence
   (`WORKFLOW.md` §3.3 and §4.2). Record per run: `nvidia-smi` SM clock and
   throttle reasons, CPU node (`numactl --show`), `/proc/stat` steal, other
   compute apps on the GPU (`WORKFLOW.md` §4.11 — the GPU must be exclusively
   ours, and on a shared node a point check does not prove it).
4. **Pin the environment before measuring anything**: `numactl --cpunodebind`
   to the GPU's socket, fixed application clocks if permitted
   (`nvidia-smi -lgc`), and check the cgroup CPU quota. If drift drops to RTX
   levels after pinning, the cause is found and a zero-code lead may fall out.
5. **Profiler alignment** (`WORKFLOW.md` §4.05): identical nsys/ncu builds by
   checksum, plus a positive control before believing any activity class is
   present or absent — the one B200 lead built without this was a phantom.

HYPOTHESES for the drift's cause, cheapest first: process landing on the
socket not attached to the GPU (every PCIe doorbell, poll and managed-memory
host page becomes remote); GPU clock/power state varying with a bursty
10%-utilisation load; CPU turbo budget or CFS quota shared with other tenants;
HMM/managed page placement varying by run; other tenants' traffic on the PCIe
switch. Each is testable with the §5 row-0 checks.

---

## 7. What a B200-first plan must do differently (synthesis)

1. **Change the diagnosis.** Retire "clock-bound arithmetic on small kernels"
   (`GETTING_STARTED.md:118`, `PERF_HANDOFF.md:45-47`, `:82`): the paired ncu
   on disk shows B200 kernels take ≤ RTX cycles and ~equal wall time. The B200
   frame is **host-paced idle**, 2.8-3.5x the RTX's, paid per launch, per
   blocking read-back and per API call on a slower core through a longer bus.
2. **Re-derive the price table on the B200 first** (§5). `PERFORMANCE.md` §4
   and DEAD_ENDS 25's prices are sm_120/Ryzen facts; the reopen thresholds in
   §4 are written in `p_B`, `s_B`, `r_cpu` so the results plug straight in.
3. **Re-measure the load-bearing defaults on the B200** (fan-out, PDL level,
   lane count): all are latency-price decisions taken on the RTX.
4. **Event-count reduction becomes the lever** where it was not on the RTX,
   but only for leads closed on *price* (graphs, count-phase merge, CDP2,
   UBO concatenation), never for leads closed by the *merge rule* — overlap is
   a schedule property and DEAD_ENDS 22 holds on both hosts. The one B200
   launch-count datum (wide-merge2 retry, no signal) weakly bounds `p_B`.
5. **Treat memory policy as host-specific.** Pin −0.93 vs −1.40 and advice
   +1.98 vs +0.36 prove the two UVM/HMM environments differ; re-test the arena
   residency modes and the readback path on the B200 rather than inheriting.
6. **Measurement discipline** (§6): session bands, ≥3 alternating rounds,
   environment pinning and logging, profiler alignment with positive controls,
   and the rule that the B200 adjudicates only ≥0.5 ms effects.
7. **Fix the record.** DEAD_ENDS 20's "SM array twice as wide" is false (148
   vs 170 SMs); the host-pin note's B200 UVM figures predate the profiler
   alignment and should carry that caveat.

---

## 8. Unknowns (things this report could not settle read-only)

- The B200's actual SM clock during a replay, its `clocks.max.sm`, SM count
  from `nvidia-smi`/`cuDeviceGetAttribute` (148 is derived here, §2.3), PCIe
  link gen/width, NUMA topology, CPU quota, and whether other tenants share
  the GPU or node.
- The RTX 5090's actual SM clock during a replay (2.407 reported vs 3.09 max);
  it moves the kernel-wall ratio between 0.91 and 1.17.
- Every B200 latency price: L_B, s_B, small HtoD latency, PDL gap, CDP2 tail,
  conditional-node evaluation, `cuStreamWaitValue32` release, `cuLaunchKernel`
  CPU cost. None measured; §3.3's split of the idle term is conditional on them.
- The generated `main` shaders' cycles/IPC/registers on sm_100 — the ncu pairs
  cover only the three raster/compact families (1.39 of the RTX heavy-band
  exclusive 5.21 ms).
- B200 band medians, mean frame, PLAN_STATS wait census, launch census and host
  perf profile — none exists; every B200 event count in §3 is the RTX's, which
  is legitimate for counts (deterministic replay) but not for durations.
- Whether the B200 pinned-arm UVM/DtoD figures in the host-pin note were taken
  under nsys 2026.1.3 (probable from dates) and thus partly artefactual.
- The B200 evidence directories (`/tmp/f3-b200`, `/tmp/f2-b200`) are on a
  non-persistent `/tmp` on the B200 host and are likely gone.

---

## 9. Queries and scripts used

All against `file:/tmp/postdead/trace.sqlite?mode=ro` (RTX, `f3caa35e4ce`, nsys
2026.4.1), window `start >= 9.276e9 AND start < 12.341e9`, 412.5 frames.

```sql
-- kernels, summed time
SELECT count(*), sum(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<?;
-- intervals for union / gap analysis
SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<? ORDER BY start;
SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND start<?;
SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>=? AND start<?;
-- blocking syncs by type (ENUM_CUPTI_SYNC_TYPE: 1 event, 2 stream-wait-event, 3 stream, 4 context)
SELECT syncType,count(*),sum(end-start),avg(end-start) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
 WHERE start>=? AND start<? GROUP BY syncType;
-- copies by kind
SELECT copyKind,count(*),sum(bytes),sum(end-start),avg(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY
 WHERE start>=? AND start<? GROUP BY copyKind;
-- API rows by name
SELECT s.value,count(*),sum(r.end-r.start),avg(r.end-r.start)
 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
 WHERE r.start>=? AND r.start<? GROUP BY s.value ORDER BY count(*) DESC LIMIT 40;
-- GPU facts
SELECT name,clockRate,smCount,memoryBandwidth,maxWarpsPerSm,maxBlocksPerSm,maxRegistersPerSm FROM TARGET_INFO_GPU;
```

Union and exposed-launch computation (python, in `trace_union.txt`): sort all
intervals, sweep, `union += ce-cs` on each disjoint run; a kernel is "exposed"
when its `start` exceeds the running max `end` over all streams; the first gap
(pre-window, 9.276 s) was excluded from the ≥20 µs sum (25.378 − 22.487 =
2.89 ms/frame).

ncu pairing (python, in `ncu_pairs.txt`): parse CSV rows starting with `"`,
key by `(ID, Kernel Name, Grid Size, Block Size)`, assert equal kernel and grid
per ID across the two files, sum `sm__cycles_active.avg` and
`sm__cycles_active.avg × smsp__inst_executed.avg.per_cycle_active` per kernel.

RTX frame medians (python, in `rtx_bands.txt`): `ts.txt` rows → every-other
submit interval from index 2782 (frame 1391); light band frames 1700-2100,
heavy 2200-3150; window 1393-1804 for the CUPTI-inflation check.

Local hardware read-only: `nvidia-smi --query-gpu=name,clocks.max.sm,...` →
RTX 5090, 3090 MHz max SM, 14001 MHz mem, PCIe Gen5 x16, 32607 MiB, driver
580.173.02 (CUDA 13.0); `lscpu` → Ryzen 9 9950X3D, 16C/32T, max 5756 MHz.
