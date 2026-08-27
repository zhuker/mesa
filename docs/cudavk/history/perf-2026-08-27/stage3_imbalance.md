# `cp_rasterize_stage3_abuf`: why one block owns the launch, and what fixing it is worth

Measured 2026-08-26 on the reference machine (RTX 5090, 170 SMs, driver 580.173.02,
SM clock 2.01 GHz, CUDA 12.8). Tree `/home/alexzhukov/mesa` at `e2fea470d04`; the
working tree was not touched and nothing was committed. Diagnosis and sizing only —
no redesign and no patch is proposed. Raw output under `/tmp/perf-audit/tiling/stage3/`.

Follow-on to `/tmp/perf-audit/tiling_ncu.md` §1.3, which found that the shipping
driver's largest kernel class has the tiled prototype's one-block signature.

---

## Verdict first

1. **There is nothing to rebalance.** The median launch hands **12 work items to a
   512-block grid**: 98.3% of launches have fewer items than blocks, so each active
   block gets **exactly one** item and **98.4% of the grid does nothing**. The launch
   duration is one item's cost, not a tail over many items.
2. **It is a launch-shape problem and a latency problem, not a barrier problem and not
   an uneven-work problem.** The grid is sized from the *triangle* count
   (`CLAMP(tris*8, 512, 2048)`), which has no relation to the queue stage 2 builds on
   the device. One item is up to a 64x64 tile walked by **64 threads = 2 warps**, so a
   block alone on an SM has 2 of 48 warps to hide global latency: 41–68% of stall
   cycles are on the L1TEX scoreboard, against **1.1–6.4% at the barrier** (the tiled
   prototype was the mirror image: 73.3% barrier).
3. **The same item costs 4x more when the queue is short.** 32,000–37,000 SM-cycles per
   item when ~10 items are in flight, **7,800–9,500** when 2,500–3,300 are. The unit is
   not slow; the machine is empty around it.
4. **A fix is worth less than 0.4 ms/frame of device time, and less than that of frame
   time.** Only **16.7%** of this class's time is exclusive. Making it *infinitely
   fast* removes **0.404 ms/frame** of device busy time; a realistic 2x removes
   **0.187 ms/frame**. The fan-out and the nine streams are already spending the idle,
   exactly as suspected.
5. **The real number is not this kernel.** Over the same window the device shows
   **GR Active 81%, SMs Active 20%, SM Issue 3%**. The machine is busy four fifths of
   the time, using a fifth of its SMs, issuing on 3% of their cycles. That is the
   session's finding; one kernel class is 0.2 ms of it.

---

## 1. What the dominant block is doing

### 1.1 The kernel's unit of work

`cp_rasterize_stage3_body` (`src/cudavk/kernels/cp_rasterize.cu`): stage 1 files
non-trivial primitives, stage 2 rasterises the medium ones with one warp each and
**decomposes anything with a bounding box above `CP_MEDIUM_THRESHOLD` (1,536 px) into
`CP_TILE_SIZE` = 64x64 tiles**, pushing one `cp_tile_pair` per tile. Stage 3 is
**one 64-thread block per (primitive, tile) pair**, block-strided:

```c
for (uint32_t tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x)
```

The host cannot know `num_tiles` — stage 2 writes it on the device — so it sizes the
grid from the triangle count instead:

```c
CP_LAUNCH(kernels.rasterize_stage3_abuf,
          CLAMP(rast_num_triangles * 8, 512u, 2048u), 1, 1, 64, 1, 1, ...)
```
(`cp_renderer.c:6175`, `:6331`, `:8353`.)

### 1.2 The queue, measured

A throw-away probe in a **/tmp copy** of the tree (gated on `CUDAVK_TILED_OPAQUE_CENSUS`,
which does nothing in the default driver) read `huge_count` back after every 29th
stage-3 launch across a full old-capture replay. **6,669 launches sampled**:

| quantity | p0 | p25 | **p50** | p75 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| queue entries (`huge_count`) | 0 | 6 | **12** | 32 | 404 | 2,413 | 4,434 |
| grid (blocks) | 512 | 512 | **512** | 1,152 | 2,048 | 2,048 | 2,048 |
| triangles in the segment | 8 | 16 | 64 | 144 | — | — | 19,200 |
| non-trivial queue | 0 | 2 | 6 | 8 | — | — | 1,187 |

* **17.6%** of launches have an **empty** queue: 512–2,048 blocks start, read a zero
  counter and exit. At the measured floor of 2.34 µs that is ~0.04 ms/frame of pure
  launch.
* **98.3%** of launches have `queue <= grid`. With block-striding that means **at most
  one item per block**, and in the median launch **500 of 512 blocks do nothing**.
* **88.6%** of launches have fewer items than the machine has SMs (170) — but those
  launches hold only **11.2%** of all items. The work is bimodal: many tiny launches
  and a few large ones.

### 1.3 So "duration = longest unit x cost" is true, trivially

Nsight Compute, 12 consecutive launches on the old capture, single pass per metric
(`sm__cycles_active.{max,avg}`, `gpc__cycles_elapsed.max`, `smsp__inst_executed.sum`,
stall breakdown). Items estimated from the instruction count, calibrated against the
empty-queue launch (12 warp-instructions per warp to read the counter and exit) and a
full tile (~2,260 warp-instructions):

| grid | µs | sm max/elapsed | sm avg/elapsed | busy SMs (time-avg) | items | items/block | SM-cycles per item | long-scoreboard stall | barrier stall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 17.2 | 0.91 | 0.06 | 10.6 | 10 | 0.02 | **36,618** | 52.1% | 1.3% |
| 512 | 16.9 | 0.91 | 0.05 | 9.3 | 10 | 0.02 | 31,669 | 58.0% | 1.3% |
| 640 | 16.9 | 0.91 | 0.05 | 9.3 | 10 | 0.02 | 31,773 | 67.5% | 1.2% |
| 512 | 27.2 | 0.95 | 0.06 | 10.3 | 33 | 0.06 | 17,097 | 50.0% | 2.0% |
| 512 | 34.5 | 0.96 | 0.05 | 8.5 | 38 | 0.07 | 15,478 | 49.0% | 1.1% |
| 512 | 30.3 | 0.95 | 0.09 | 16.0 | 69 | 0.13 | 14,178 | 41.0% | 2.8% |
| 1024 | 7.0 | 0.78 | 0.36 | 60.6 | 50 | 0.05 | 17,275 | 53.7% | 6.4% |
| 2048 | 75.6 | 0.98 | 0.77 | 130.3 | 2,549 | 1.24 | **7,769** | 48.4% | 2.5% |
| 1408 | 98.6 | 0.98 | 0.92 | 155.8 | 3,255 | 2.31 | 9,480 | 58.6% | 1.8% |
| 640 | 2.3 | 0.26 | 0.24 | 40.6 | 0 (empty) | 0 | — | 69.9% | 0.0% |

Read the table in two halves:

* **Short queue (10–69 items).** One item per block. `sm max/elapsed` is 0.91–0.96 —
  the block that has the item runs essentially the whole launch — and the time-averaged
  number of busy SMs is **8.5 to 16 of 170**, which is the queue length. The launch
  duration *is* one item: 17–34 µs.
* **Long queue (2,549–3,255 items).** 1.2–2.3 items per block, `sm avg/elapsed` 0.77–0.92,
  the machine is properly loaded — and **the per-item cost drops to a quarter**.

The item is not expensive. It is expensive **when it runs alone**.

---

## 2. Why one block gets it

Three fix classes were on the table. The counters and the code separate them.

### Not a barrier problem
Barrier stalls are **1.1–6.4%** of warp issue-stall cycles here. The tiled prototype's
were 73.3%. Stage 3's thread-0 setup is paid once per item, and in the median launch a
block has one item, so the serial section is amortised over a 4,000-pixel walk.

### Not an uneven-work problem
Every item is one 64x64 tile of one primitive, and 98.3% of launches give each active
block exactly one. There is no long list to split and no unequal assignment to fix.
The `blockIdx`-strided loop is a fair mapping over an input that is simply too short.

### It is a launch-shape problem
`CLAMP(tris*8, 512, 2048)` is a guess made from the wrong quantity. In the median case
the guess is **512 blocks for 12 items — 40x too large** — and the guess cannot be made
right without a device-to-host readback of `huge_count`, which is precisely the
round-trip the design avoids. The cost of the wrong guess is not the empty blocks
(they cost 2.3 µs); it is that the *right* number of blocks is 12, which is 7% of the
machine, so the launch cannot fill the GPU no matter how it is sized.

### And it is a latency-hiding problem
One item is up to 4,096 pixels walked by **64 threads = 2 warps**, and with one block
resident per SM that is **2 of 48 warps**. From `--set full` on the same class:
**46.7 warp-cycles per issued instruction, 25.4 of them (54.3%) waiting on an L1TEX
scoreboard dependency**; achieved occupancy 5.5–15% of peak; DRAM throughput 0.1–9%.
The per-item cost falling 4x when the queue is long (§1.3) is the same statement from
the other side: the work is latency, and latency is hidden by concurrency the short
launches do not have.

**Fix classes this admits** (not designed here, and not recommended below):
give an item more warps (256 threads instead of 64, or split a 64x64 tile into
sub-tiles), or give the machine more items at once by merging the segments' queues into
one launch. Rebalancing and compaction — the answers for the tiled prototype — do
nothing here.

---

## 3. What a fix is worth, in frame milliseconds

### 3.1 The window

`nsys --trace=cuda --delay=12 --duration=6` on the old capture, shipping default.
**5.84 s, 426,759 kernels, ~447 frames (13.07 ms/frame)**, `cp_rasterize_stage3_abuf`
**100.6 launches/frame at a median of 18.56 µs** — against `PERFORMANCE.md` §5.1's
134.6 launches/frame at a median 18.5 µs, so the window is representative of the
documented profile. (A first attempt at `--duration=8` from process start landed in a
peel-heavy phase with a different mix — 2.0 stage3_abuf launches/frame — and is not
used for any number here. Phase matters; a window has to be checked against §5.1
before it is read.)

### 3.2 Kernel time is not opportunity

| class | ms/frame (kernel time) | **exclusive ms/frame** | exclusive share |
|---|---:|---:|---:|
| `cp_rasterize_stage3_abuf` | 1.897 | **0.317** | **16.7%** |
| `cp_clip_rast_fused` | 1.790 | 0.225 | 12.5% |
| `cp_rasterize_stage2_abuf` | 0.620 | 0.118 | 19.1% |
| `cp_clip_rast_fused_abuf` | 1.158 | 0.238 | 20.6% |
| `cp_rasterize_stage3` (direct) | 0.649 | 0.282 | 43.4% |
| `main` (fragment shaders) | 3.096 | 1.955 | 63.2% |
| `cp_abuf_composite` | 0.340 | 0.340 | 100% |

"Exclusive" is time during which **no other kernel is running anywhere on the device**.
83.3% of `stage3_abuf`'s time already has company. Device busy in the window is 51.5%
of wall, and **78.0% of that busy time has exactly one kernel running** — nine streams
notwithstanding.

### 3.3 The bound, by simulation

Shrink every `cp_rasterize_stage3_abuf` interval by a factor, leaving its start where it
is, and recompute the union of all kernel intervals:

| change | device busy removed |
|---|---:|
| stage3_abuf 2x faster | **0.187 ms/frame** |
| stage3_abuf 4x faster | 0.288 ms/frame |
| stage3_abuf **infinitely** fast | **0.404 ms/frame** |
| the *whole* raster chain 2x faster | 0.763 ms/frame |
| the *whole* raster chain infinitely fast | 2.106 ms/frame |

For scale, the raster chain in this window is **6.56 ms/frame of kernel time** and
`stage3_abuf` alone is 1.90 (2.484 in §5.1). **Deleting the entire raster chain would
remove only 2.1 ms/frame of device busy time**, because most of its kernel time is
concurrent with something else.

### 3.4 And device time is not frame time

`PERFORMANCE.md` §5.2/§7 measured the conversion, and it is per site: the host is
blocked **73.7%** of the frame, the device is idle 28% in the documented profile and
48% in this window, and the two mechanisms that paid (fan-out +2.73 ms, PDL +0.43 ms)
both worked by *overlap*, not by removing work. So the 0.187–0.404 ms/frame above is an
**upper bound on device time**, and the frame gain is a fraction of it.

**Honest answer: a distribution or latency fix to `cp_rasterize_stage3_abuf` is worth
well under 0.4 ms/frame, most likely 0.1–0.2 ms — about 1% of the frame. The fan-out is
already spending the idle this kernel leaves.**

### 3.5 The number that is worth more than this kernel

`nsys --gpu-metrics` over the same window, no CUDA tracing, 60,163 samples:

| metric | median | median over busy samples | p90 |
|---|---:|---:|---:|
| GR Active | 81% | 87% | 100% |
| **SMs Active** | **20%** | 31% | 77% |
| **SM Issue** | **3%** | 4% | 12% |
| Compute Warps in Flight | 5% | 9% | 32% |
| DRAM Read / Write | 1% / 1% | 1% / 3% | 3% / 19% |

The engine is occupied 81% of the time. A fifth of the SMs have work. Those SMs issue
an instruction on **3% of their cycles**. This is the same defect as §1–§2 measured at
device level rather than per kernel, and it is why per-class kernel-time shares in §5.1
overstate what removing a class can return: the classes overlap each other into an
engine that is busy and empty at the same time.

---

## 4. Answers, in the order asked

1. **What is the dominant block doing?** One `(primitive, 64x64 tile)` item, walked by
   64 threads. Median queue **12 items** against a **512-block** grid, so one item per
   active block and 98.4% of the grid idle; `sm__cycles_active.max` is 91–96% of
   elapsed because that one block *is* the launch. Duration = one item x its cost, and
   the cost is 32–37k SM-cycles when the machine is empty, 7.8–9.5k when it is full.
2. **Why does one block get it?** Not a barrier (1.1–6.4% of stall cycles) and not
   uneven work (every item is one tile, one per block). It is (a) a launch shape sized
   from the triangle count rather than the device-built queue —
   `CLAMP(tris*8, 512, 2048)`, 512 blocks for 12 items — and (b) a work unit of 4,096
   pixels given 2 warps, so 41–68% of stall cycles sit on the L1TEX scoreboard with
   nothing to switch to.
3. **What could a fix buy, in frame milliseconds?** At most **0.404 ms/frame** of
   device busy time and **0.187 ms/frame** at a realistic 2x, because only 16.7% of the
   class's time is exclusive — and less than that in frame time, because the frame is
   host-bound. **Do not size this from the 2.484 ms/frame in §5.1.** The whole raster
   chain, deleted, is worth 2.1 ms/frame of device time.

---

## 5. GPU exclusivity, stated positively

The 1 Hz sampler (`/tmp/perf-audit/gate/gpu_watch.sh`) ran for this session as well.
The foreign libvpx tenant (`build_vp9_cuda/vpxenc`, ~1,000 MiB) reappeared in a fourth
burst at **t+1083..1113 s**. **Every measurement in this document was taken between
t+1487 s and t+1907 s** — the queue probe (t+1487..1519), the Nsight Compute counters
(t+1519..1569), the three nsys timelines (t+1569..1907) and the GPU-metrics window —
and **no foreign process appears in the sampler after t+1113 s**. In the measurement
span every non-empty sample named only `gfxrecon-replay`, `ncu` or `nsys`.

Anyone timing after us needs the sampler, not a point check: this tenant runs as short
processes and reappeared twice during the session.

---

## 6. Artifacts

```
/tmp/perf-audit/tiling/stage3/probe_old.txt        6,669 sampled stage-3 launches: grid, triangles, queue length
/tmp/perf-audit/tiling/stage3/ncu_abuf_detail.txt  ncu, 12 launches, per-SM activity + stall breakdown
/tmp/perf-audit/tiling/stage3/old_d12.sqlite       representative nsys window (5.84 s, 426,759 kernels)
/tmp/perf-audit/tiling/stage3/old_timeline.sqlite  first window (peel-heavy, not used for the numbers)
/tmp/perf-audit/tiling/stage3/old_metrics.sqlite   nsys GPU metrics over the representative window
/tmp/perf-audit/tiling/mesa/                       throw-away copy of the tree carrying the probe
/tmp/perf-audit/tiling/gpu_watch.log               1 Hz compute-process sampler, whole session
```
