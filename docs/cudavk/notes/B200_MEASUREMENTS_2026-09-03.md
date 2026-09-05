# The B200, measured

Answers to `B200_MEASUREMENT_PLAN.md`, run 2026-09-03. Before this the B200
had no trace, no host profile, no launch price and an uninspected environment;
every number in the tree for it was a frame-time A/B. That is no longer true.

**The headline: the plan asked whether the B200's gap is the clock or per-event
host pacing. It is both, and the host side is the larger ratio.** Kernels take
**1.49x** the RTX's wall time; CUDA API calls cost **2.3-2.6x**. A third thing
nobody was looking for is bigger than either: **unified-memory fault stall is
1.83 ms/frame there against 0.38 here, 4.8x**.

Standing when this was written: favorite3 **9.883**, favorite2 **8.283**
(3 alternating rounds, gates passed, `PERF_HANDOFF.md`).

---

## 0. Environment — one finding, and it is worth 0.5 ms

Sampled during an untraced favorite3 replay.

| check | result | verdict |
|---|---|---|
| SM clock under load | **1965 MHz**, = Max Customer Boost | not throttled |
| every throttle reason | Idle, SW Power Cap, HW Slowdown, Thermal, Sync Boost: **all Not Active** | not throttled |
| PCIe | **Gen5 x16**, = max | not the lead |
| cgroup CPU quota | `max 100000` (none) | not descheduled |
| steal during the run | 23 ticks | negligible |
| **CPU binding** | app `Cpus_allowed_list: 0-191`; **GPU0 affinity `0-47,96-143`** | **unbound across both sockets** |

The host is a dual-socket Xeon Platinum 8559C, 192 CPUs, two NUMA nodes; the
GPU hangs off node 0. The replay was free to run on node 1, where every
doorbell, poll and managed-page touch crosses UPI.

**Binding to the GPU's socket is worth 0.50 ms/frame on favorite3 and nothing
on favorite2** (`taskset -c 0-47,96-143`, 3 alternating rounds, all gates
passed):

| capture | unbound | bound | delta | arms |
|---|---:|---:|---:|---|
| favorite3 whole | 9.7774 | **9.2763** | **-0.5012** | **disjoint** |
| favorite3 heavy | 11.6108 | 11.4695 | -0.1413 | overlap |
| favorite2 whole | 8.2623 | 8.3878 | +0.1254 | overlap |
| favorite2 heavy | 9.3385 | 9.3835 | +0.0450 | overlap |

favorite3's whole-window arms are disjoint at a delta of 0.50, which is at the
edge of what that host can adjudicate (drift 0.3-1.3).

**It did not replicate, and the claim is withdrawn.** A second session ran the
experiment that separates the two mechanisms `taskset` conflates -- CPU
placement and page placement -- on favorite3, three alternating rounds each,
every gate passed:

| arm | whole | heavy | arms |
|---|---:|---:|---|
| session 1: `taskset` near vs unbound | **-0.5012** | -0.1413 | disjoint |
| session 2: `numactl --membind=0` (memory near, CPU free) | +0.0327 | +0.0857 | overlap |
| session 2: `numactl --cpunodebind=0` (CPU near, memory follows) | -0.2587 | -0.0206 | overlap |

Neither arm reproduces it. The CPU-only arm points the same way but at half
the size, its arms overlap, and it is **not sign-consistent** -- the candidate
wins two rounds of three (9.759/9.500, 9.437/9.661, 9.938/9.385). Against a
host that drifts 0.3-1.3 ms between sessions, that is drift, not an effect.

**So there is no NUMA win to adopt, and the question of "flag or source
change" does not arise.** What survives is a real negative and a useful one:

- **Page placement is not the mechanism.** Binding memory to the GPU's node
  alone changes nothing (+0.03). The 1.83 ms/frame of UM fault stall in 1.5 is
  therefore **not** far-node placement, which removes the most obvious
  explanation for it and rules out a `numactl`-shaped fix for item 2.
- **A single disjoint A/B on this host is not a result.** Yesterday's arms were
  disjoint and its rounds tight, and it was still drift. Three alternating
  rounds bound the noise *within* a session; they say nothing about the drift
  *between* sessions, which on the B200 is larger. Anything under ~0.5 ms here
  needs two sessions before it is written down as a number.

To measure it the A/B tool gained `CTRL_WRAP`/`CAND_WRAP`: an arm may now
differ by how the process is launched, not only by its environment.

## 1. The first CUDA trace ever taken on this host

`nsys 2026.4.1.191`, installed from the deb shipped from this workstation and
**verified by md5 (`e64519f9611c5bf192ac4346cc75cbe6`) on both machines**, as
the plan requires. favorite3, `--delay=22`, UM CPU and GPU page faults on,
exit 0 with 6,947 shim rows, 2,124 frames in the window. For a like-for-like
reading, a fresh RTX trace was taken **at the same commit with identical
flags** (928 frames) -- the archived `postpin-rtx2` predates leads A/C/B2/F and
its launch counts no longer match. Both were then read by the same script,
`closure-audit/trace_audit.py`, unchanged.

### 1.1 Kernel wall time: 1.49x, so the clock is real

Per-class median duration, same driver, same script:

| class | RTX p50 us | B200 p50 us | ratio |
|---|---:|---:|---:|
| cp_clip_rast_fused | 10.0 | 13.1 | 1.31x |
| VS | 8.9 | 11.7 | 1.31x |
| cp_clip_rast_fused_abuf | 5.8 | 7.8 | 1.34x |
| FS_abuf | 4.8 | 7.1 | 1.48x |
| cp_rasterize_stage3_abuf | 5.8 | 8.7 | 1.50x |
| cp_fs_compact | 5.3 | 8.3 | 1.57x |
| cp_rasterize_stage3 | 4.8 | 7.6 | 1.58x |
| cp_rasterize_stage2 | 4.1 | 6.7 | 1.63x |
| cp_fs_writeback | 1.4 | 2.5 | 1.79x |
| FS_direct | 3.3 | 6.6 | **2.00x** |
| cp_abuf_scan_finish | 3.7 | 4.5 | 1.22x |
| cp_vertex_fetch | 12.1 | 4.5 | **0.37x** |
| **median** | | | **1.49x** |

This lands on the plan's "~1.5x" branch: **the clock is the story**, and
`verify2-M`'s 0.9x-1.5x model is resolved at its pessimistic end. Two entries
are exceptions worth their own look -- `FS_direct` is twice as slow rather than
1.5x, and `cp_vertex_fetch` is nearly **three times faster** on the B200, the
only class that prefers that machine.

Device union follows: **7.005 ms/frame** against the RTX's 4.850 (1.44x), with
less overlap (1.44x vs 1.665x concurrent).

### 1.2 Host per-call cost: 2.3-2.6x, so pacing is real too

CUPTI-traced API durations (inflated on both hosts, and the plan accepts the
ratio for exactly that reason):

| CUDA API | RTX us | B200 us | ratio | calls in window |
|---|---:|---:|---:|---:|
| `cuLaunchKernel` | 1.94 | 4.46 | **2.29x** | 374k / 726k |
| `cuLaunchKernelEx` | 1.84 | 4.36 | **2.38x** | 271k / 502k |
| `cuMemcpyDtoH` | 7.69 | 19.93 | **2.59x** | 6.2k / 15.4k |
| `cuMemcpyHtoDAsync` | 1.72 | 2.64 | 1.54x | 157k / 318k |

`perf` could not be used -- the workspace kernel (6.12.58, Amazon Linux) has no
matching `linux-tools` package and `perf_event_paranoid` is 2 -- so this is the
plan's documented fallback rather than its first choice.

**The plan framed §1 as a fork: ~1.0x means pacing and the launch-count leads
reopen, ~1.5x means the clock and only a redesign helps. The measurement
refuses the fork.** Kernels are 1.49x and calls are 2.3x. Both are true, and
the host ratio is the bigger one.

### 1.3 What that does to the launch-count leads

Dependent-launch gaps, p50, measured on both:

| class | RTX p50 | B200 p50 | ratio |
|---|---:|---:|---:|
| memcpy1 | 2.8 | 4.2 | 1.50x |
| VS | 2.3 | 3.6 | 1.57x |
| cp_clip_rast_fused | 1.2 | 2.6 | **2.17x** |
| memset | 1.5 | 2.7 | 1.80x |
| main_other | 2.6 | 6.2 | **2.38x** |
| total gap time | 17.31 ms/f | 21.70 ms/f | 1.25x |

The what-if replay agrees: setting every launch gap to zero is worth
**-1.726 ms/frame** on the B200 against **-1.257** on the RTX.

The plan records the reopening threshold for the count-phase merge
(`DEAD_ENDS` 21) as `p_B >= 2.0-2.3 us`. **Measured B200 gaps are 2.2-4.2 us
across the dominant classes, so that threshold is met** and the lead reopens
*as a candidate on this host only*. It is not a result -- it is an A/B that has
not been run. `TODO` 7's threshold (`p_B >= 8 us`) is **not** met.

### 1.4 Device-decided episodes stay closed here

`DEAD_ENDS` 41 refuted G on the RTX by absorption. The plan gives the condition
for it to reopen on the B200: post-sync bubble >= 35 us.

| | RTX | B200 |
|---|---:|---:|
| post-sync refill bubble, p50 | 4.2 us | **7.8 us** |
| syncs per frame | 17.06 | 18.84 |
| bubble total | 0.119 ms/f | 0.657 ms/f |

**7.8 us is far below 35, so G does not reopen.** The single-stream join tail
is the opposite shape -- p50 7.3 us on the B200 against 63.1 us on the RTX --
because that host runs fewer streams concurrently at once.

### 1.5 The one that reopens: unified memory, 4.8x

| | RTX HEAD | B200 HEAD |
|---|---:|---:|
| GPU fault events | 8,722 | 18,927 |
| pages migrated | 133,495 | **308,581** |
| CPU faults on managed pages | 26,096 | 37,434 |
| distinct faulting regions | 2 | **4** |
| **fault stall per frame** | **0.3808 ms** | **1.8274 ms** |

`DEAD_ENDS` 40 closed the memory-residual class on the RTX because its whole
population -- 0.37 ms/frame -- sits under the 0.500 admission gate. **On the
B200 that same population is 1.83 ms/frame, comfortably over it**, on a frame
that is only 9.88 ms. The largest region is 30.9 MB and owns 90.1% of the
pages, the same shape as the RTX's 21.5 MB region: an application
`vkAllocateMemory` mapped managed, not driver scratch.

This does **not** revive D, E or K, which target small driver buffers and would
still be chasing the 10% tail. It makes the *app allocation* the largest
measured device-side item on the deployment target. `DEAD_ENDS` 37 already
measured the obvious lever on the B200 -- forcing residency, +0.36 there -- so
the mechanism must be something other than a blunt `SET_ACCESSED_BY`.

### 1.6 A caution about idle share

The B200 trace shows 21.69 ms/frame with zero device ops against a **mean**
frame of 28.69 ms. That reads as "75% host-bound" and it should not be quoted:
CUPTI inflates the host side, and this host pays 2.3x per call for it, so a
traced run manufactures precisely the gap being measured. Medians are the safe
comparison -- traced 12.29 vs untraced 9.88 (1.24x) on the B200, 7.11 vs 5.94
(1.20x) on the RTX -- and those are similar, which is why the per-class and
per-call ratios above are trustworthy while the idle share is not.

## 3. Three defaults decided on the RTX, re-decided here

favorite3, 3 alternating rounds, all gates passed on all 18 runs.

| A/B | RTX result | B200 whole | B200 heavy | arms |
|---|---|---:|---:|---|
| `CUDAVK_NO_PDL=1` (PDL off) | +0.38 | control 9.9100 -> **9.5404**, PDL worth **-0.370** | -0.401 | overlap |
| `CUDAVK_NO_OPAQUE_STREAMS=1` (fan-out off) | +0.66-0.84 | control 10.8776 -> **9.7307**, fan-out worth **-1.147** | **-1.169** | **disjoint** |
| `CUDAVK_CTX_SCHED=yield` | neutral | +0.121 | +0.100 | overlap |

**The opaque-stream fan-out is worth 1.15 ms/frame on the B200 against
0.66-0.84 on the RTX** -- 1.4-1.7x more, disjoint at both levels, and the
largest confirmed effect of any default on that host. It is the same story as
§1.2: more concurrency matters more where each event costs more.

PDL points the same way as on the RTX (worth 0.37) but its arms overlap, so on
this host it is directionally confirmed and numerically unadjudicated. Yield
scheduling does nothing, which §0 predicted -- there is no CPU quota and no
steal time for it to help with.

`CP_PASS_STREAMS=16` (`DEAD_ENDS` 38) was **not** run: it needs a rebuild
rather than a flag, and 153 MB against 180 GB of HBM makes the RTX's reason for
declining it inapplicable here. It is the cheapest untried item on this host.

## 4. Microbenchmarks: warranted, not run

The plan gates §4 on §1 showing large gaps. It does -- launch gaps are
1.25-2.38x and API calls 2.3-2.6x -- so the `PERFORMANCE.md` §4 price table
does need a B200 column, and `cdp2-microbench.cu` is the template for it.
Until that column exists, **no B200 forecast is admissible**, which is that
document's own rule applied to a host where none of its six prices were
measured. This is the next piece of work, not a result.

## 5. What this changes in the record

- **`GETTING_STARTED.md` and `PERF_HANDOFF.md`'s "clock-bound arithmetic on
  small kernels"**: supported. Kernel wall is 1.49x. But it is only half the
  story, and the smaller half -- API calls are 2.3x.
- **`DEAD_ENDS` 20's "SM array twice as wide"**: wrong either way, 148 vs 170.
- **`DEAD_ENDS` 40** (memory residuals, closed on the RTX at 0.37 ms/frame):
  its population is **1.83 ms/frame on the B200**. The entry stands for the
  RTX and for the three small-buffer items; the app allocation behind it is now
  the largest measured device-side item on the deployment target.
- **`DEAD_ENDS` 41** (device-decided episodes): stays closed. The reopening
  condition, a post-sync bubble >= 35 us, measures 7.8.
- **`DEAD_ENDS` 21** (count-phase merge): its recorded reopening threshold
  (2.0-2.3 us) **is met** on this host. A candidate again, on the B200 only.
- **Band medians**, previously recorded nowhere: favorite3 whole **9.88**,
  heavy **11.6**; favorite2 whole **8.28**, heavy **9.34**. Every number in
  this document carries its session band, because that host drifts 0.3-1.3 ms
  between sessions and only effects >= ~0.3-0.5 are adjudicable there.

## 6. Ranked, for whoever picks this up

1. ~~**CPU-bind the process to the GPU's socket**~~ -- **WITHDRAWN**, see 0.
   It did not replicate in a second session, and memory-only binding shows the
   page-placement mechanism is absent. No action.
2. **The 30.9 MB managed app allocation** -- 1.83 ms/frame of fault stall, 4.8x
   the RTX. Needs a mechanism that is not `SET_ACCESSED_BY` (`DEAD_ENDS` 37
   measured that at +0.36 here).
3. **`CP_PASS_STREAMS=16`** -- **session 1 measured, no effect on favorite2**:
   8 lanes 9.6692 vs 16 lanes 9.6773 whole (+0.008), heavy -0.035, arms
   overlapping. On the RTX this capture was the one that showed signal
   (-0.084, disjoint). favorite3 did not run -- the host went down mid-A/B.
   The "more lanes pay more on a latency-bound host" hypothesis is
   **unsupported so far**; one more session would close it. The two builds
   alternated inside one session via the new per-arm ICD, so this is not a
   between-session artefact.
4. **The count-phase merge** -- reopened by measurement, needs its own A/B.
5. **The §4 price table** -- required before any forecast for this host.

Method notes: `nsys` verified by md5 across both hosts; the RTX comparison
trace was retaken at the same commit rather than reusing the archived one,
because launch counts had moved under leads A/C/B2/F; `perf` is unavailable in
that workspace and the API ratio is the plan's documented fallback; traced idle
share is not quoted as a host-bound claim, for the reason in 1.6.
