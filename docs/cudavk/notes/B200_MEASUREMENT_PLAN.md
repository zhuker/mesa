# What to measure on the B200 before deciding anything about it

Written 2026-09-03. The B200 (sm_100, 148 SMs, Xeon Platinum 8559C host,
Coder workspace behind two ssh hops) is the deployment target, and after the
2026-09-02/03 work it stands at favorite3 **9.883** and favorite2 **8.283**
(`PERF_HANDOFF.md`). Every B200 number in this tree is a frame-time A/B. There
is **no** B200 trace, no host profile, no launch price, and its environment
has never been inspected. The RTX price table (`PERFORMANCE.md` §4) and every
default decided on it (fan-out, PDL level, lane count) are sm_120/Ryzen facts.

The paired ncu data (`history/perf-2026-09-02/b200/REPORT.md` §2) says the
B200 executes the same kernels in 0.742x the RTX's active SM cycles, but ncu
ran at locked base clocks nobody recorded, so kernel wall time on the B200 is
anywhere from 0.9x to 1.5x the RTX's (`verify2-M/REVIEW.md`). Which end of
that range is true decides whether the remaining gap there is per-event host
pacing (launch-count leads reopen at B200 prices) or the clock (only a
redesign helps). The steps below settle it in about half a day of machine time
and no driver code.

Rules that apply to every step (`WORKFLOW.md` §3.3, §4.05, §4.11):
- The B200 drifts **0.3-1.3 ms/frame between sessions** against the RTX's
  0.06. Every A/B is three strictly alternating rounds in one session, arms
  disjoint, per-pair sign consistent; only effects >= ~0.3-0.5 ms are
  adjudicable there. Quote every number with its session band.
- `/tmp` and system packages do not persist on that workspace. Re-verify the
  nsys 2026.4.1 build **by checksum** against the RTX box before any trace, and
  run a positive control before believing an activity class is absent.
- Sample GPU exclusivity at 1 Hz across the whole window
  (`history/perf-2026-08-27/tools/gpu_watch.sh`), not at one instant; inside a
  PID namespace `--query-compute-apps` cannot prove it.
- The harness now exits 0 (favorite3 6,947 rows, hash `e4a60a71...`), so
  CUPTI's final buffer flushes; a trace whose UVM row count is a multiple of
  99,999 lost a buffer and is invalid.

---

## 0. Environment, during an untraced favorite3 replay — ten minutes, zero code

Run a normal replay and, alongside it:

```bash
nvidia-smi dmon -s pucm -d 1 > dmon.log &          # SM clock held, power, throttle reasons, PCIe util
nvidia-smi -q -d CLOCK,PERFORMANCE                  # application/max clocks, active throttle reasons
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi topo -m                                  # which CPU socket the GPU hangs off
cat /sys/bus/pci/devices/<gpu-bdf>/numa_node
grep Cpus_allowed_list /proc/<vulkan_app pid>/status   # numactl may be absent; this is the same fact
cat /sys/fs/cgroup/cpu.max                          # container CPU quota
grep '^cpu ' /proc/stat                             # steal column, before and after the run
```

| finding | what it means | next step |
|---|---|---|
| SM clock well below max, throttle reasons set | the bursty 10%-utilisation load is not holding boost; kernel wall scales with it | lock application clocks (`nvidia-smi -lgc`, needs the sudo the workspace has), re-measure both captures |
| process on the socket the GPU is not attached to | every doorbell, poll and managed-page access crosses UPI | bind (`taskset`/`numactl --cpunodebind --membind` to the GPU's node), re-measure; if the session drift collapses toward 0.06 the drift cause is found |
| PCIe below Gen5 x16 | the 3.7 MB readback and ~260 small uploads a frame are the lead | fix the slot/BIOS before anything else |
| CPU quota set, or steal time non-zero | the thread spinning in CUDA syncs is being descheduled | A/B `CUDAVK_CTX_SCHED=yield` (registry flag; on the RTX `auto` resolves to spin) |
| all nominal | the environment is not the story | proceed to §1 |

Typical misconfigurations here cost 0 to ~1.5 ms/frame (ESTIMATED); nothing
shows the environment is wrong, only that it was never looked at.

## 1. One CUDA trace — the B200's first real measurement

```bash
# checksum-matched nsys 2026.4.1; harness exits cleanly so buffers flush
nsys profile -t cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
     -o b200-f3 ./build/vulkan_app
nsys export --type sqlite b200-f3.nsys-rep
```

Then run the RTX scripts on it unchanged, so the comparison is like for like:

| script (in `docs/cudavk/history/perf-2026-09-02/`) | what it produces |
|---|---|
| `closure-audit/trace_audit.py <sqlite> b200 <t0> <t1> b200.json` | device union vs idle per frame, time with k ops in flight, per-class union-exclusive, the dependency-respecting what-if replay (host infinitely fast / device infinitely fast / launch gaps zero / free syncs) — the same table `PERF_ANALYSIS_2026-09-02.md` §1.4 gives for the RTX |
| `kernel-chain/census.py` | per-kernel launches/frame, duration percentiles, same-stream gaps, main-stream vs side-stream busy, the critical path |
| `host-sync/bubbles.py` | per sync site: spin, post-sync bubble, join cost — the DtoH round trip on that host |
| `host-launch-audit/pred2.py` | device-paced link gaps: PDL vs plain vs through-a-copy — the exposed-launch price `L_B` |
| `frame-boundary/partition.py` | head, internal, inter-submit and boundary idle per frame |
| `host-visible-memory` §4.2 SQL | residual UVM migration per region; which allocation the 21.5 MB region of dead end 40 is on this host |

Window: favorite3 real work starts at external submit 2782 (frame 1391); use
the shim's `ts.txt` to delimit frames exactly as on the RTX.

What the trace decides, read against `/tmp/postpin-rtx2/trace.sqlite` (RTX,
same driver era) frame for frame:

| quantity | RTX value | if the B200 shows | conclusion |
|---|---:|---|---|
| per-class kernel duration (VS, FS, clip, stage 3, compact), median | see `arith-vs-latency` §3.2 | ~1.0x | kernels are not the gap; it is per-event pacing |
| | | ~1.5x | the clock is the story; only the redesign (`PERF_ANALYSIS` §2.2) moves it |
| dependent-launch gap, plain / PDL, p50 | 1.86 / 0.26 µs | 2-4x | every launch-count lead closed on the RTX price reopens: `TODO` 7 at `p_B` >= 8 µs, count-phase merge (DE 21) at `p_B` >= 2.0-2.3 µs, graphs via recapture at 342 gaps x (`L_B` - 2 µs) |
| post-sync bubble per drain, p50 | 14.6 µs | >= 35 µs | device-decided episodes (DE 41) reopen on the B200 only: 7.25 waits x (`s_B` - `s_R`) |
| device idle share of the frame | ~22% whole, 38% heavy | > 50% | host-paced frame; confirm with §2 |
| frame boundary (readback end -> first kernel) | ~0.9 ms | larger | the application's readback fence is worth more there than the RTX's 0.5-1.1 |
| UVM rows / frame and their regions | 68 rows, 5 regions | | sizes dead end 40's class on this host; the `SET_ACCESSED_BY` result differed 5.5x between hosts (DE 37), so nothing transfers |

## 2. A host profile without CUPTI — the per-call price on the Xeon

If `perf` is available in the workspace:

```bash
perf record -F 999 --clockid mono --call-graph dwarf -o perf.data ./build/vulkan_app
perf script -i perf.data | ./stackcollapse-perf.pl > folded.txt      # or perf report --stdio
```

then `host-launch-audit/perfwin.py` (needs the shim `ts.txt` for the frame
window). It yields the no-CUPTI cost of `cuLaunchKernel` (RTX **1.70 µs**),
`cuLaunchKernelEx` (**1.36**) and `cuMemcpyHtoDAsync` (**1.30**), i.e. the
factor `r_cpu` by which ~1,560 issue calls a frame cost more on this core.
If `perf` is not available, the CUPTI-traced API durations from §1 give the
same ratio to within CUPTI's inflation (28-91% per call, but similar on both
hosts).

## 3. Three A/Bs of defaults decided on the RTX

Each three alternating rounds per capture, in one session, with
`src/cudavk/tests/cp_harness_ab.sh` exactly as `/tmp/b200-ab2.sh` ran the
2026-09-03 set:

```bash
ICD=$HOME/mesa/build/src/cudavk/cudavk_devenv_icd.x86_64.json \
CTRLFRAMES=$HOME/sentinels/ctrl OUT=/tmp/ab-pdl \
  $HOME/mesa/src/cudavk/tests/cp_harness_ab.sh favorite3 pdl "CUDAVK_NO_PDL=1" "" 3
```

| A/B | RTX result | why it may differ on the B200 |
|---|---|---|
| `CUDAVK_NO_PDL=1` | +0.38 old / +0.15 Crossroads | PDL's 0.44-3.02 µs per link is a latency price; if `L_B` is larger the level-3/4 choice may change |
| `CUDAVK_NO_OPAQUE_STREAMS=1` | +0.66-0.84 (favorite3/favorite2) | the fan-out's value swings with per-segment hand-off latency (32 µs/episode on the RTX) |
| `CP_PASS_STREAMS` 16 (a build; DE 38) | favorite2 -0.084, favorite3 neutral, declined for +153 MB | 153 MB is nothing on a 180 GB part; the B200's wider per-SM warp budget may change the sign |
| `CUDAVK_CTX_SCHED=yield` | neutral at idle, better under CPU contention (DE 16) | only if §0 finds a quota or steal time |

## 4. Only if §1 shows large gaps: the price microbenchmarks

`docs/cudavk/notes/cdp2-microbench.cu` is the template (512-link chains,
best/mean, CPU-clock API cost loop). Build with the CUDA 13 toolkit on that
host, pin to the GPU's NUMA node, run each arm three times interleaved:

| # | measurement | RTX reference |
|---|---|---|
| 1 | host launch-to-start latency, empty stream (exposed-launch slope of `PERFORMANCE.md` §4) | 1.974 µs slope, 2.047 back-to-back floor |
| 2 | back-to-back dependent kernel gap, with and without `griddepcontrol` (PDL) | 1.48-1.51 host-issued, 0.26 PDL link |
| 3 | DtoH sync round trip: kernel writes 4-64 B, host `cuMemcpyDtoH` and `cuStreamSynchronize` | `cuMemcpyDtoH_v2` API 7.3 µs mean (CUPTI) |
| 4 | small `cuMemcpyHtoDAsync` (16-256 B) from pinned, issue cost and device-visible completion | 1.86 µs API / 0.94 device |
| 5 | CDP2 tail launch (in the template) | 7.32-7.44 µs |
| 6 | `cuLaunchKernel` CPU cost with the stream busy (in the template) | 1.43 µs |

The deliverable is a **B200 column for `PERFORMANCE.md` §4**. Until it exists
no B200 forecast is admissible — that document's own rule ("a model that is
not one of these has never survived contact with a measurement"), applied to
a host where none of these were measured.

## 5. What to write down afterwards

- Correct the record where the trace contradicts it: `GETTING_STARTED.md:118`
  and `PERF_HANDOFF.md` ("clock-bound arithmetic on small kernels") if kernel
  wall is ~1.0x; `DEAD_ENDS.md` entry 20's "SM array twice as wide" is already
  wrong (148 vs 170 SMs) either way.
- Add the B200 band medians (light/heavy) — none are recorded anywhere.
- Note the session band beside every number.

Sources: `history/perf-2026-09-02/b200/REPORT.md` (every B200 number in the
tree and its provenance), `verify2-M/REVIEW.md` (why the kernel-wall ratio is
a model, and the environment checks), `verify2-F/REVIEW.md` §B200,
`verify2-G/REVIEW.md` §B200, `DEAD_ENDS.md` 16, 20, 25, 37, 38, 40, 41.
