# Verify-2, Lead M — B200 environment and price table before any code

Verifier: adversarial, read-only. 2026-09-02. Nothing built or run on a GPU;
the only hardware commands were `nvidia-smi` queries on the local RTX. Scratch:
this directory (`ncu.py` re-derives every ncu number below from the two CSVs).

Lead under review: `scratchpad/b200/REPORT.md` (the "B200 report"). Its claims:
(1) paired ncu shows B200 kernels take 0.742x the RTX's active SM cycles at equal
instruction counts, so kernel wall is 0.91-1.17x RTX, not 1.6x; (2) the +4.11
ms/frame excess is host pacing paid per event; (3) "multiplicative not additive"
across the two captures proves the per-event reading; (4) no B200 measurement of
PDL/fan-out/lanes exists; (5) run zero-code environment checks, then six
microbenchmarks, before any code.

**Verdict: OVERSTATED.** The measured parts reproduce exactly and the qualitative
direction (most of the excess is device idle, not kernel time) survives even an
adversarial reading. But the ceiling attached to the lead ("any one of these
being wrong is worth up to the whole excess", §4 ranking item 1) has no
measurement behind it, the kernel-wall ratio is derived from cycle counts taken
at locked base clocks that were never recorded, and the "multiplicative"
argument does not discriminate between the report's mechanism and the one it
says it retires. A cheaper and more decisive first step exists than the six
microbenchmarks.

---

## 1. What reproduces (MEASURED)

Re-parsed `/tmp/ncu-rtx/v2.csv` and `/tmp/ncu-b200-v2.csv` (`ncu.py`): 45
launches each, identical `(ID, kernel, grid, block)` sequence, 7 metrics.

| family | n | Σ cycles RTX | Σ cycles B200 | ratio | inst proxy |
|---|---:|---:|---:|---:|---:|
| `cp_rasterize_stage3` (2048×64) | 11 | 249,961 | 229,034 | 0.916 | 1.125 |
| `cp_clip_rast_fused` (5-1825 blk) | 11 | 268,446 | 164,625 | 0.613 | 1.077 |
| `cp_fs_compact` (900×256) | 11 | 211,234 | 136,597 | 0.647 | 1.114 |
| `cp_clip_rast_fused_abuf` | 6 | 9,722 | 9,129 | 0.939 | 1.044 |
| `cp_rasterize_stage3_abuf` | 6 | 37,097 | 36,718 | 0.990 | 1.091 |
| all | 45 | 776,459 | 576,102 | **0.742** | — |

Registers RTX→B200: 56→48, 38→48, 43→44, 46→48 — the report's reading is right.
Mean DRAM % of peak 9.61 → 2.01. Mean `sm__warps_active` 25.5% → 17.2%.

**148 SMs: confirmed.** `waves = grid/(SMs × blocks_per_SM)`; stage3 20
blocks/SM (RTX 2048/(170×0.60) = 20.1) → B200 2048/(20×0.69) = 148.4;
`cp_fs_compact` at 48 regs is register-limited to 5 blocks/SM → 900/(5×1.22) =
147.5. Both land on 148, which is also the published B200 SM count (ESTIMATED,
public spec). DEAD_ENDS.md:1112 "the SM array twice as wide" is wrong; the
same sentence's "where the launch price is higher" is an unmeasured assertion
(no B200 launch price exists anywhere in the tree, §3 below).

**Claim (4) — no B200 PDL/fan-out/lane measurement — confirmed.** `grep -i
b200` over `docs/cudavk/**` hits only: PERF_HANDOFF.md:13-20,45,97,112;
GETTING_STARTED.md:17,112,118; PERFORMANCE.md:109,125-126; DEAD_ENDS.md:95,
1106-1113 (wide-merge2 retry), 1370 (CDP2 "re-run on target"), 1546 (scratch
reuse "only on B200"), 2046, 2218-2257 (entry 37), 2278 (ncu occupancy "both
sm_120 and sm_100"); WORKFLOW.md:206,252,266,320; two notes. DEAD_ENDS 38
(lanes) is "RTX 5090" at line 2267; the PDL rows in PERFORMANCE.md §4 are
old/Crossroads on the RTX; the fan-out surrender 0.66-0.84 (entry 36,
PERF_HANDOFF.md:67) is "RTX heavy". So the tree holds exactly five B200
frame-time measurements (host-pin ×2, entry 37 ×1, wide-merge2 screen ×1,
llvmpipe ×1) plus the 45-launch ncu pair. Correct.

**The census numbers used for scaling are the light band's.** "~1,100 API
calls, ~350 exposed launches, 24 blocking round trips" come from
`/tmp/postdead/trace.sqlite` (frames ~1393-1804, 427 kernels/frame, 23.4
blocking/frame per `closure-audit/postdead_audit.log:5,48`). The heavy band
(`postpin_audit.log:5,48`) has 719.6 kernels/frame and **17.19** blocking
calls/frame. The 10.79 headline is a whole-window median. Mixing bands is
noted, not fatal: the report says so itself in §8.

## 2. ARITHMETIC — where the derivation is weaker than stated

### 2.1 The kernel-wall ratio 0.91-1.17 rests on a clock nobody recorded

`ncu` defaults to `--clock-control base`: both runs executed at each part's
**base** clock, and the CSVs carry no `sm__cycles_elapsed.avg.per_second`,
`gpu__time_duration` or clock column (metric list in `ncu.py` output: seven
metrics, none of them time or frequency). The report's formula
`t_B/t_R = 0.742 × f_R/f_B` treats the cycle count as clock-invariant and then
inserts **replay** clocks (2.407-3.09 GHz RTX, 1.965 GHz B200). Two problems:

- For a latency-bound kernel cycles scale with clock (the report's own
  explanation of the 1.2-2.7x IPC rise, §2.2), so the count taken at base
  clock is not the count at replay clock. For an issue/dependency-chain-bound
  kernel (stage3 at 0.92, the A-buffer variants at 0.94-0.99, the small clip
  launches at 0.88-1.10 — together ~40% of the sampled cycles) the count is
  clock-invariant and the wall ratio is `0.92-0.99 × f_R/f_B` ≈ 1.15-1.5x at
  plausible replay clocks. `arith-vs-latency/REPORT.md:363-376` reads the
  **same CSVs** the opposite way ("stretches every cycle-bound chain by
  ~1.5x") and explicitly declines to convert cycles to wall because of the
  clock lock. Neither report measured a B200 wall time. Both are MODEL.
- RTX 5090 base is 2017 MHz (spec); the B200's base clock is not in the tree
  and I do not know it reliably. If the B200 base is materially below the RTX
  base, part of the 0.742 is simply "fewer, longer cycles" and says nothing
  about wall time at all — which is what the report argues; if the two base
  clocks are similar, 0.742 really is fewer nanoseconds. **One unknown number
  decides which**, and `nvidia-smi -q -d CLOCK` on the B200 host reports it.
  Label: the 0.91-1.17 range is MODEL, not ESTIMATED-from-measured.

Adversarial re-weighting (MODEL): take the clip/compact families as
latency-bound (wall ≈ 1.0x) and the stage3/abuf/small-clip population as
cycle-bound at f_R/f_B = 1.45 (2.8/1.93 GHz): weighted wall ratio ≈ 1.2x on
the sampled 1.39 ms/frame of heavy-band exclusive time → +0.3 ms. The unsampled
generated VS+FS (1.367 ms exclusive, PERF_HANDOFF.md:34-36) at the same 1.2x
adds +0.3; at the raw clock ratio +0.6. Copies (0.77 ms DtoD on RTX,
`b200/REPORT.md:251`) shrink on HBM3e (B200 pinned-arm DtoD 0.323 vs RTX 0.768,
host-pin note :40-42): −0.3 to −0.4. **Kernel+copy delta under the adversarial
reading: +0.3 to +0.9 ms/frame**, against the report's −0.5 to +0.9. So even
read against it, the kernel side does not reach +4.11 unless the B200 replay
clock sits far below its maximum — which is exactly the report's row-0a check.
The qualitative conclusion holds; the quantitative split (idle 5.1-6.3 ms on
B200) inherits the ±0.7 ms of clock uncertainty plus the unsampled classes.

### 2.2 "Multiplicative, not additive" does not establish a per-event price

Ratios (MEASURED inputs, host-pin note table): pinned 10.789/6.6757 = 1.616,
8.9763/5.4804 = 1.638; controls 12.188/7.6048 = 1.603, 10.5662/6.5461 = 1.614.
Additive residuals 0.62 (pinned) and 0.56 (controls) against a within-session
band of ~0.33 (DEAD_ENDS.md:1112). Multiplicative is the better one-parameter
fit; that much is fair. But:

- **A uniform device slowdown is also multiplicative.** The model the report
  retires ("clock-bound arithmetic", GETTING_STARTED.md:118) predicts the same
  constant ratio on both captures. The test cannot separate "every kernel is
  1.6x slower" from "every event costs 1.6x more idle". Only the ncu evidence
  (§2.1) distinguishes them, and it is clock-ambiguous.
- **Events are not proportional to frame time even inside favorite3.** Heavy
  vs light band on the RTX: kernels/frame 719.6 vs 427.3 (**1.68x**), blocking
  calls/frame 17.19 vs 23.37 (**0.74x**), while frame time is 1.42x untraced
  (8.2 vs 5.78) and 1.14x traced (8.14 vs 7.15). A per-launch penalty would
  therefore *not* be multiplicative in frame time across bands, and whether it
  is across captures depends on favorite2's event/time ratio, which nobody has
  censused (no favorite2 launch census in the tree). The 1.616 vs 1.638
  agreement is consistent with the per-event model only by coincidence of two
  unknown ratios — or, more simply, with everything being uniformly slower.
- Two captures fit any two-parameter model exactly. The B200 report's own §6
  says the ratio is 1.47-1.69 across sessions; a 0.022 agreement between two
  ratios drawn from a ±0.1 band is not evidence.

Downgrade: (3) is HYPOTHESIS, not established. It should be dropped from the
argument; the ncu data plus a B200 trace (§4) is what can decide it.

### 2.3 The RTX evidence that the *pool* is large enough (supporting, MODEL)

`closure-audit/postpin_audit.log:175,181`: on the RTX heavy band, zeroing every
front-end launch gap is worth −1.61 ms median; an infinitely fast host −1.35 ms.
So the pacing pools on the RTX are ~1.5-3 ms; a B200 paying 2-3x per turn on
those pools does reach +3-4 ms. This is the strongest support for the
report's mechanism and it is the report's, not new — but it is a
dependency-respecting replay of RTX intervals, not a B200 measurement.

### 2.4 Corrected ceilings for the lead as written

- **RTX: 0.00 ms/frame.** The lead has no RTX content. The RTX host is
  characterised (single socket, PCIe 5 x16, 0.06 drift, `nvidia-smi` here:
  max SM 3090 MHz, Gen5 x16); nothing in "environment checks" moves it.
- **B200, environment-only (zero code): 0 to ~1.5 ms/frame, ESTIMATED, with a
  realistic chance of 0.** Components, all typical values, none measured on
  that host: NUMA misplacement of a latency-bound ping-pong process (remote
  PCIe doorbells, polls and managed pages) typically costs 5-15% on such
  workloads → 0.5-1.5 ms of 10.8; a GPU clock stuck below boost under a bursty
  ~10%-busy load would scale the kernel union (4.5-6 ms) by the clock deficit,
  0.5-1.5 ms if the deficit is 10-25%, 0 if the part already boosts; CFS quota
  or a co-tenant explains drift (0.3-1.3) more than median. The report's "worth
  up to the whole excess" (§4 ranking item 1) is unsupported: nothing in the
  evidence says the environment is misconfigured, only that it was never
  checked.
- **B200, price table: 0.00 ms directly.** It is an enabling measurement. The
  largest lead it could reopen (DEAD_ENDS 1, graphs) is gated by structure
  (rotating scratch addresses, zero recipe recurrence, DEAD_ENDS.md:139-146),
  not by price; the recapture path (`notes/CUDA13_UPGRADE.md:94-96`, 5.3 µs
  per 32 nodes) changes that structure and was never built. Its B200 ceiling is
  `342 gaps × (L_B − L_R)` and L_B is unmeasured; at L_B = 4 µs that is ~0.7
  ms, at 8 µs ~2 ms (MODEL, the report's own §4 row).

## 3. PRECEDENT

- **No entry refutes the mechanism**; none measured the B200 environment. The
  competing diagnosis (PERF_HANDOFF.md:45-47, GETTING_STARTED.md:118 "clock and
  issue") is the same CSVs read without a wall-time measurement — equally MODEL.
- **Entry 37** (DEAD_ENDS.md:2214-2264) prices what this lead must pay: one
  profiler build across hosts verified by checksum, a positive control, ≥3
  alternating rounds, and the note that system packages and `/tmp` on the B200
  do **not** persist (PERF_HANDOFF.md:99) — the 2026.4.1 nsys installed for
  entry 37 may be gone and must be re-verified before any trace. The report
  states these rules (§6); it does not budget the reinstall.
- **Entry 20 retry** (DEAD_ENDS.md:1106-1117) is the only B200 launch-count
  datum: no signal inside a 0.33 ms band. The report reads it as a weak upper
  bound on p_B; that is fair, and it also means the one B200 experiment that
  touched launch count found nothing.
- **Entry 16** (DEAD_ENDS.md:845-912) matters to the cgroup check and the
  report under-uses it: `CUDAVK_CTX_SCHED=auto` resolves to `SPIN`, and the
  main thread spins 4.4 ms/frame (`notes/RESIDUAL_AUDITS_2026-08-31.md`,
  table). Under a CFS quota shared with other tenants, spinning burns quota
  and can be throttled mid-frame; `yield` was neutral on the RTX and
  `blocking` +2.8-5.4%. A one-flag A/B on the B200 (`CUDAVK_CTX_SCHED=yield`)
  is the cheapest environment probe there is, and it is a registry flag
  (`src/cudavk/cp_debug.c:461-464`), so no code.
- **Entries 25 and 29** explicitly demand B200 re-measurement
  (DEAD_ENDS.md:1370, 1546); the report is consistent with them.
- **PERFORMANCE.md:245-248** ("a model that is not one of these has never
  survived contact with a measurement") applies to the report's §3.3 split
  (L_B 4-8 µs, s_B 20-40 µs, r_cpu 1.4-1.8): typical datacenter values, labelled
  ESTIMATED in the report, but they are the numbers the ranking is built on.
  They should be read as HYPOTHESIS until row 1-3 of §5 are run.
- The 'why different' argument — "overlap is a schedule property, so
  merge-rule closures carry over; price closures do not" — holds. Entry 22 is
  arithmetic on interval overlap and is host-independent. Fair.

## 4. FEASIBILITY, and the cheaper decisive probe

The environment checks are all read-only shell commands and are the right
first step; two caveats the report does not carry:

- `numactl` is a system package and does not persist on that workspace; the
  process/GPU placement can be read without it (`cat /proc/self/status | grep
  Cpus_allowed_list`, `/sys/bus/pci/devices/<gpu>/numa_node`, `nvidia-smi topo
  -m`). `nvidia-smi -lgc` needs root (they have `sudo`). Inside a PID
  namespace `nvidia-smi --query-compute-apps` cannot prove exclusivity
  (WORKFLOW.md §4.11); the report says so.
- The results that change the plan: SM clock during replay far below max →
  lock clocks and re-measure before anything else, and the kernel side
  reopens; process on the wrong socket → bind and re-measure, and if drift
  collapses toward the RTX's 0.06 the drift cause is found; PCIe below Gen5
  x16 → the 3.7 MB readback and 141 HtoD/frame are the lead; none of the
  above wrong → the environment is not the story and the price table is.

**Cheaper decisive probe than six microbenchmarks.** A B200 CUPTI trace
existed twice (host-pin note :7-10, :40-42; entry 37) and nobody recorded the
device union from it. Kernel timestamps are not CUPTI-inflated, so **one nsys
trace of favorite3 on the B200 under the checksum-matched 2026.4.1 build,
run through `closure-audit/trace_audit.py`**, yields in one run: device union
vs idle per frame (the report's whole §3.3 split, measured instead of
modelled), per-class kernel durations for *every* class including the
generated VS/FS the ncu pair missed (the kernel-wall ratio, settled), the
in-situ exposed-launch gap distribution (RTX p50 1.89 µs, `b200/REPORT.md:242`
— this *is* L_B in situ), the post-sync refill bubble (RTX 4.4 µs median,
`postpin_audit.log:262` — this is s_B in situ), and copy rates. That replaces
microbenchmarks 1, 3, 4 and 8 and answers the diagnosis directly; only PDL gap
(2), conditional node (5), `cuStreamWaitValue32` (6) and CDP2 tail (7) still
need `cdp2-microbench.cu`. Pair it with `nvidia-smi dmon -s pucm -d 1` on an
**untraced** run for the clock (row 0a). One session, ~2 hours including the
profiler-alignment check, no driver code.

Second-cheapest and also zero code: re-run the same 45-launch ncu pair with
`--clock-control none` and `--metrics gpu__time_duration.sum,
sm__cycles_elapsed.avg.per_second` added. It costs what the existing run cost
and turns the 0.742 into a wall-time ratio at replay clocks.

Invariants: none touched (ARCHITECTURE.md §8 is irrelevant to a zero-code
lead). Tests: none until code exists; the B200 A/B discipline of WORKFLOW.md
§3.3/§4.05 applies to every number produced.

## 5. Corrected statement of what a B200-first plan should do first

1. Read the environment (clock under replay, PCIe gen/width, socket vs GPU
   NUMA node, `cpu.max`, steal) — ten minutes, no code, decides whether step 2
   is even on a stable host. Do not quote a ceiling for it; it is a check.
2. Verify the nsys build by checksum (it may not have survived the workspace
   restart), then take **one** favorite3 trace and one untraced run with
   `dmon`, and compute union/idle/per-class/gap/bubble with the existing
   audit script. This is the measurement that decides between "host-paced
   idle" and "clock-stretched kernels"; today both are models of the same
   CSVs.
3. Only then the price table (§5 of the B200 report), reduced to the four
   items the trace cannot give (PDL gap, conditional node, WaitValue release,
   CDP2 tail), and the free flag A/Bs (`CUDAVK_CTX_SCHED=yield`,
   `CUDAVK_NO_PDL`, `CUDAVK_NO_OPAQUE_STREAMS`, lanes) at ≥3 rounds.
4. Fix the record now, independent of measurement: DEAD_ENDS.md:1111-1112
   ("launch price is higher", "SM array twice as wide") to "148 SMs; launch
   price unmeasured"; GETTING_STARTED.md:118 and PERF_HANDOFF.md:45-47 to
   "the ncu cycle counts are consistent with either explanation; wall time
   on the B200 has not been measured per kernel".

## 6. Labels summary

MEASURED: the ncu tables, 148 SMs, the five B200 frame-time datapoints, the
RTX band event counts, the RTX what-if pools. ESTIMATED: environment effect
sizes (typical, not this host), the adversarial re-weighting. MODEL: the
0.91-1.17 wall ratio, the 5.1-6.3 ms B200 idle, every reopen threshold in
p_B/s_B/r_cpu. HYPOTHESIS: "multiplicative ⇒ per-event", the drift causes.
