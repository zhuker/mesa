# Lead E — generated-shader argument blocks as kernel parameters instead of upload-ring HtoD copies

Adversarial verification, read-only. Verdict: **OVERSTATED**. Nothing was built or run on
the GPU; the only executables run were `python3`/`sqlite3` against the two traces and
`llc-18` on a 40-line scratch `.ll` file (no repo build, no driver code).

Scratch artefacts (all under this directory): `copies.py`/`copies.out` (copy census by
following kernel, both traces), `gapbyname.py` (plain-link gap by secondary kernel),
`cause.py`/`cause.out` (long pre-copy gap attribution), `replayE.py` +
`replay_postpin.log`/`replay_postdead.log` (closure-audit's dependency-respecting replay
with three lead-E pools added), `byval.ll` + `out_ptx*.ptx` (LLVM 18 byval lowering probe).

Labels: MEASURED = from a trace/document/tool output; MODEL = replay output; ESTIMATED = my
arithmetic on measured inputs; HYPOTHESIS = not measurable here.

---

## 0. Verdict in one screen

| claim | status | corrected |
|---|---|---|
| "what-if replay deletes all HtoD copies: −0.80 heavy / −0.43 light" | reproduced (MODEL: −0.799 / −0.427), but it is an **upper bound with three known over-credits** and it includes 14–16 copies/frame that sit before rasterizer/A-buffer kernels, not generated shaders | restricted to copies before generated shaders or `cp_fs_compact`: **−0.77 heavy / −0.39 light** (MODEL); restricted to the shade chain (FS + compact links, the only links where the copy is on a device-paced serial critical path): **−0.41 heavy / −0.21 light** (MODEL) |
| "documented small-copy price 305 × 1.30 = 0.40" | wrong count and wrong price for two of the three copy classes | the 1.30 µs is a host-API price measured on a host-paced 8-byte VS-link copy (iteration 26 S1). On the shade chain the collectable cost is device-side chain excess, MEASURED 1.5–1.9 µs p50 per copy plus a tail; on the VS boundary the copy is host- or gate-paced 79% of the time and its device excess is ≈0.006 ms/frame |
| "copy adds 1.85 µs over a plain link, 3.45 over PDL" | MEASURED and reproduced (postpin: 3.49–3.62 through-copy p50 vs 2.02 plain, 0.26 PDL) | but it applies to 131 of the 262 copies, not all; and only compact has a PDL prologue — the generated FS has no `griddepcontrol.wait`, so "forbids PDL on the following launch" is true for 66 of 262 copies |
| "max observed 31 KB fits the 32,764 B limit" | irrelevant as stated: **PTX kernel parameters are fixed-size per kernel** (`.param .b8 name[N]` in the entry signature; `cuLaunchKernel` copies the declared size) | either every launch carries the maximum block (~17–20 KB at `CP_MAX_BATCH_DRAWS = 128`), or the in-param block is capped at a few rows and larger batches keep a device table **and its upload**. The reports assume a variable-size parameter, which does not exist |
| ">4 KB parameter path unverified" | still unverified; my LLVM-side probe covers codegen only | the decisive question is device front-end cost per KB of inline parameter data — needs the microbench |
| "nir_to_ptx: ld.param at constant offsets; dynamic indexing via cvta.param" | **half wrong, in the lead's favour**: LLVM 18 lowers dynamically indexed byval kernel params to `mov.b64 %rd, param; ld.param [%rd+off]` with no `cvta.param` needed (MEASURED, `llc-18 -mattr=+ptx75`). **But any escaping use (pointer reaching a non-inlined call) copies the whole block to a per-thread `.local` depot of the full declared size** — 16,912 B in my probe | feasible only under the driver's existing "no surviving helper symbol" invariant (ARCHITECTURE §8 #16); the failure mode is DEAD_ENDS 14 item 1 / DEAD_ENDS 28 in a new dress |
| corrected ceiling RTX heavy | | **0.30–0.45 ms/frame defensible (shade chain, MEASURED excess 0.30 capped / MODEL 0.41); ≤0.7 full scope at model face value; realistic net 0.25–0.5** — below the 0.5 gate except at face value of the full scope |
| corrected ceiling RTX light | | **0.18–0.21 shade chain; ≤0.39 full scope (MODEL)** |
| B200 | no trace, no price table | HYPOTHESIS 0.3–0.6; the host-side share (262 × 1.3 µs × 1.4–1.8 CPU factor ≈ 0.5–0.6 ms of host time) would convert better on a host-paced frame, the copy-engine pickup latency is unknown |

---

## 1. PRECEDENT

### 1.1 DEAD_ENDS 8 (flush placement) — "why different" holds

Entry 8's retry condition is literally "uploads stop being owned by the launch boundary —
a different ownership model, not a better heuristic" (`DEAD_ENDS.md:434-437`). Passing the
block by value is that model. Entry 8's own numbers (76% of flush points empty, merge at an
unchanged boundary 0.59 µs) say nothing about it. **Passes.** One caveat carried forward:
entry 8's mechanism statement — "the launch on the far side is what reads the bytes, so an
upload cannot be deferred past it" — is also the reason the copy cannot be removed unless
*every* block in the span moves (see §3.2).

### 1.2 DEAD_ENDS 26 (INTERP_INLINE / COMPACT_PDL) — a risk, and an under-measurement

- **+0.13 ms/frame, four palindromic pairs agreeing, microcause unattributed**
  (`DEAD_ENDS.md:1396-1399`). INTERP_INLINE carries the 416 B interp block *inside the FS
  argument upload* — it moves the copy from the stage3→compact link to the compact→FS link
  (the copy count is unchanged: `cp_renderer.c:3688-3697, 4163-4171`). It therefore does
  **not** price removal, but it is direct evidence that changing which link carries the
  bytes can lose 0.13 with no attribution. Lead E's FS parameter block would embed the same
  416 B (the report's §5.4 item 3 concedes this). The risk is not refuted by anything in
  the evidence; it is inherited whole.
- Entry 26's instrument ("gap before fs_compact 5.70 µs median … the counter pool alone
  took it to 0.83 µs", `DEAD_ENDS.md:1401-1403`) measured **copy end → compact start**. In
  the current default the trace has that hop at 0.96 µs p50 (host-launch-audit §5.3, and
  reproduced), but the copy's own pickup latency in front of it is 1.92 µs p50, so the
  whole link is 3.49 µs p50 (MEASURED, postpin, 57,065 links). Entry 26's "there was less
  left than the 5.70 suggested" undercounts the link by the copy-engine pickup. This does
  not rescue lead E's ceiling (§2 below prices the full link), but it means COMPACT_PDL's
  measured −0.10 on top of INTERP_INLINE is roughly what the link is worth (0.116 ms/frame
  model upper at 65.7 × (2.02 − 0.26)), and there is no larger hidden prize on that link.

### 1.3 Other entries that price a cost this lead must pay

- **DEAD_ENDS 14** (device-link vertex fetch): "two compilers with different address-space
  models cannot share a pointer ABI"; a surviving device call took a 20-register shader to
  108. A parameter-space (`addrspace(101)`) pointer is exactly a pointer that cannot cross
  into anything not inlined. My `llc-18` probe (§3.3) shows the concrete penalty: a byval
  block whose address reaches a call becomes a per-thread `.local` copy of the *declared*
  size (16,912 B in the probe; ~17–20 KB for the real maximum). ARCHITECTURE §8 #16 is the
  guard; DEAD_ENDS 28's ".local anywhere is a hard stop" gate must be applied to every
  generated kernel under the new ABI.
- **DEAD_ENDS 7**: "fold an operation for its count only when its bytes are negligible."
  1–3 KB per launch is negligible; a fixed 17–20 KB parameter per launch is not (§3.1):
  181 generated launches/frame × 20 KB = 3.6 MB/frame through the pushbuffer, and a
  17–20 KB host memcpy inside every `cuLaunchKernel` (1–2 µs at 10–20 GB/s), which exceeds
  the 1.30 µs API call being removed.
- **DEAD_ENDS 22 (merge rule)**: the credit is collectable only where the removed operations
  were serial. The FS/compact copies are serial (100% main stream, 99–100% device-paced,
  MEASURED §2.1). The VS copies are 58.7% on the fan-out side streams (heavy band) and 79%
  host- or gate-paced; a per-copy price applied to them is the error entry 22 exists to
  prevent.
- **PERFORMANCE §4, rule 1** ("removing outright pays 1.30, merging 0.59"): the 1.30 was
  measured on iteration 26 S1 — the 8-byte `cp_vs_meta` copy in the *VS* link, which is
  host-paced. It is a host-API price. Applying it to the FS/compact copies underprices
  them (their collectable cost is the device-side 1.5–1.9 µs excess plus a tail) and
  applying it to VS copies on a device-paced side lane over-credits them. The two errors
  happen to roughly cancel in the total, which is why 262 × 1.30 = 0.34 lands near the
  measured 0.30 + model 0.1–0.3; but the decomposition, which decides *what to build*, is
  wrong.
- **TODO 'Performance leads'**: no entry on argument copies. Item 7 (fs-UBO row
  concatenation on the opaque path) would *increase* FS block row counts and interacts
  with the fixed-size parameter cap in §3.1.

---

## 2. ARITHMETIC

### 2.1 Copy census by the kernel that follows the copy on its stream (MEASURED)

`copies.py`, both traces; the class of a `main` kernel is decided by its same-stream
neighbours (prev `cp_fs_compact` → FS; next `cp_clip*` → VS). "device-paced" = the
following launch's API had returned before the predecessor kernel ended; "gap" =
predecessor kernel end → following kernel start (the whole link, through the copy).

Heavy band (`/tmp/postpin-rtx2`, 1,024 frames, window = closure-audit's; 261.99 HtoD/frame):

| copy precedes | n/frame | bytes p50/p90/max | >4 KB /frame | main-stream | device-paced | link p50 / mean µs |
|---|---:|---|---:|---:|---:|---|
| `main` VS | 107.03 | 2,048 / 3,104 / 37,376 | 7.90 | 41.3% | **21.4%** | 22.9 / 80.1 |
| `cp_fs_compact` | 65.71 | 720 / 1,264 / 8,720 | 0.99 | 100% | 99.3% | **3.49** / 14.6 |
| `main` FS | 65.71 | 752 / 1,648 / 15,088 | 1.57 | 100% | 100% | **3.62** / 4.2 |
| `main` FS (A-buffer) | 5.62 | 1,676 | 0.10 | 30% | — | — |
| `cp_abuf_*`, `cp_vertex_fetch`, other | 17.4 | 204–944 | 0 | mostly main | — | — |
| descriptor arena (`cuMemcpyHtoD` sync) | 0.99 | 64–169 KB | 0.99 | — | — | not a flush copy |

Light band (`/tmp/postdead`, 411.5 frames, 141.77/frame): VS 45.00 (26.6% main stream,
28.3% device-paced, link p50 65.6 µs), compact 33.88 (3.49 µs), FS 33.88 (3.42 µs), FS-abuf
7.18, other ~17.

Baseline plain device-paced kernel→kernel link on the main stream: **p50 2.02 µs** heavy,
1.86 light; PDL link 0.26 both (MEASURED, `copies.py`; matches host-launch-audit §5.3).

Every `main` launch in both traces is immediately preceded by an HtoD copy (0.00/frame
exceptions) — the report's structural claim holds.

### 2.2 Device-side excess that a copy adds on a serial critical link (MEASURED)

Sum over device-paced main-stream copies of max(0, link − 2.02 µs), gaps capped at 20 µs to
exclude join/gate waits that the following kernel would pay anyway (uncapped figure in
parentheses):

| pool | heavy ms/frame | light ms/frame |
|---|---:|---:|
| FS link | 0.151 (0.151) | 0.096 (0.096) |
| compact link | 0.149 (0.703) | 0.086 (0.533) |
| VS link | 0.006 | 0.009 |
| **shade chain total** | **0.300** | **0.182** |

The compact link's uncapped tail (0.55 ms/frame heavy in 1.7 copies/frame of ~400 µs) is
the fan-out **join**: the main stream waits on the side-stream events before the compact of
a fanned-out episode, and the interp copy is the first op after the wait. `cause.py`
confirms the long-gap shade copies coincide with other-stream kernel ends or are
unexplained by any copy-engine neighbour; none are copy-engine queueing. Removing the copy
does not remove the join, so the capped figure is the honest device-side ceiling for the
shade chain: **0.30 heavy / 0.18 light**.

The VS copies' long gaps (15.06 ms/frame summed, heavy) are gate releases: 17.5 copies/frame
start within 3 µs of another side stream's copy — eight lanes released by one
`cp_pass_gate_stream` event, not copy-engine serialisation of 1 µs copies. The VS kernel
would wait on the same gate. Their device-side excess is 0.006 ms/frame.

### 2.3 By-value parameter size does not show in the link today (MEASURED, ≤ ~900 B)

`gapbyname.py`: plain device-paced main-stream links by secondary kernel, heavy band —
`cp_clip_rast_fused` (~600 B by value) p50 2.05 µs, `cp_abuf_sort` 1.82, `cp_abuf_quad_count_all`
1.76, `cp_peel_advance` 2.27, `cp_fs_writeback` (~100 B) 2.59. No size signal in the
0.1–0.9 KB range. Nothing in either trace carries a >4 KB parameter, so the cost per KB of
inline parameter data above ~1 KB is **unmeasured here** (HYPOTHESIS: pushbuffer inline data
is fetched by the front end before the launch; several µs per 16–20 KB is plausible, zero is
not).

### 2.4 The replay, restricted to what lead E can remove (MODEL)

`replayE.py` = closure-audit `trace_audit.py` with three pools added, classified by the
kernel that follows the copy on its stream. E0 validation unchanged (median 8.1549 vs 8.1432
heavy; 7.1668 vs 7.1529 light).

| what-if | heavy Δ median | light Δ median |
|---|---:|---:|
| delete all HtoD (report's number) | −0.799 | −0.427 |
| delete all HtoD + their host API | −0.867 | −0.464 |
| **delete shade-chain copies (FS + compact links), 131.4/f** | **−0.410** | **−0.212** |
| … + host API | −0.411 | −0.212 |
| delete VS-boundary copies, 109.0/f | −0.325 | −0.170 |
| … + host API | −0.380 | −0.176 |
| delete all copies before generated shaders or compact, 246.6/f | −0.770 | −0.393 |
| … + host API | −0.834 | −0.413 |
| launch gaps zero (reference pool this is a subset of) | −1.607 | −0.987 |

Three over-credits in the model, all in the lead's favour:

1. **Post-copy kernel gap kept.** The replay deletes the copy's duration (0.107 ms/frame
   heavy, shade pool) and its own front-end gap (0.267 ms/frame for the 129.7 short-gap
   shade copies), but the following kernel keeps its *recorded* gap, which is measured
   from the copy's end: FS 1.1 µs, compact 1.0 µs p50 (replay log, launch-gap table). A
   real plain link is 2.02 µs p50. Over-credit ≈ 131 × 0.95 µs ≈ **0.12 ms/frame heavy**.
   0.41 − 0.12 ≈ 0.29, which is the MEASURED capped excess in §2.2.
2. **Replacement priced at zero.** The by-value launch is assumed to cost exactly what
   today's 8-byte-pointer launch costs on both the host and the front end.
3. **Host API time is CUPTI-inflated** (1.76 µs mean vs ~1.30 without CUPTI); the
   "+ host API" rows over-credit by ~25%. Immaterial for the shade pool (host is ahead:
   +0.001), material for the VS pool (+0.055 heavy).

The VS-pool credit (−0.32) is the copy's ~2 µs front-end gap plus ~1 µs duration on the
gate-released side lanes and on the main-stream VS after a join, collected once per
fan-out step. It is plausible as a model number and I could not measure it directly (the
predecessor-end metric is dominated by the gate). It is the least certain third of the
lead, and the most expensive to build (§3.2).

### 2.5 Corrected ceilings

- **RTX heavy band**: shade chain **0.30 (MEASURED, capped) – 0.41 (MODEL)**; adding the VS
  boundary at model face value **≤ 0.77**; the report's 0.80 includes ~16 copies/frame in
  front of rasterizer/A-buffer kernels that this lead does not touch. Net of the
  replacement's host memcpy (≈ +0.1 µs per 1–2 KB launch, −1.2 µs API removed: a small
  host gain) and of an unknown front-end cost for the >1 KB blocks: **realistic 0.25–0.5**,
  reaching the 0.5 gate only if the VS-boundary copies are also removed (which requires the
  slice, vertex-buffer and texture tables to move too — §3.2) and DEAD_ENDS 26's 0.13 does
  not recur.
- **RTX light band**: shade chain **0.18–0.21**; full scope **≤ 0.39** (MODEL).
- **B200**: no trace, no prices (b200/REPORT.md §5 lists small-HtoD latency and
  per-launch CPU cost as unmeasured). HYPOTHESIS: the copy-engine pickup component
  (0.3 ms of the RTX shade figure) is unknown on sm_100; the host-API component
  (262 × 1.3 µs × 1.4–1.8 slower core = 0.48–0.61 ms of host time) converts better on a
  frame the same report calls "host-paced idle, 2.8–3.5× the RTX's". **0.3–0.6 ms/frame**,
  quoted as a hypothesis, decidable only with a B200 nsys 2026.4.1 trace and this replay.

### 2.6 Double counting

The shade-chain excess is a subset of closure-audit's "launch gaps zero" pool (1.61 ms) and
of kernel-chain's critical-path front-end latency; it is the same 131 links that a
segment-graph lead (closure-audit 4.3) or a COMPACT_PDL revival would collect. The VS
boundary credit overlaps closure-audit 4.1 (geometry run-ahead), which re-schedules the
same links. Nothing in lead E is additive with those at face value.

---

## 3. FEASIBILITY

### 3.1 Kernel parameters are fixed-size; the reports assume they are not

A PTX entry declares `.param .align 8 .b8 name[N]` with a compile-time `N`
(`out_ptx75.ptx`, every case), and `cuLaunchKernel` copies the declared parameter size
(`cuda.h:17740-17750`). One compiled `main` cannot take 656 B on one launch and 15 KB on the
next. So one of:

- **(a) declare the maximum** — VS: 64×8 + 32 + 128×16×8 + 128×4×4 + 904 ≈ 20 KB; FS:
  64×8 + 16 + 128×16×8 + 416 ≈ 17.4 KB at `CP_MAX_BATCH_DRAWS = 128` (`cp_debug.h:150`,
  block layouts `cp_renderer.c:5785-5807, 3688-3697`). Every one of the ~181 generated
  launches/frame then carries 17–20 KB: a 17–20 KB host memcpy inside `cuLaunchKernel`
  (1–2 µs, ESTIMATED at 10–20 GB/s, which is *more* than the 1.30 µs call it removes) and
  3.6 MB/frame of pushbuffer inline data whose front-end cost is unmeasured
  (HYPOTHESIS, not zero). DEAD_ENDS 7's rule applies directly. This form should not be
  built.
- **(b) cap the in-parameter block** at a few rows (p90 VS copy is 3.1 KB ≈ 10 rows; p90
  FS 1.6 KB ≈ 8 rows) and keep a device-resident table with a pointer for larger batches.
  Then ~90% of VS and ~97% of FS launches fit ≤4 KB, and the remaining ~10 copies/frame
  keep their upload. This is the feasible form; it also stays inside the classic 4 KB
  limit, sidestepping the >4 KB path entirely — and it gives up nothing measurable, since
  the >4 KB copies are 4.4% of copies (11.6/frame) and mostly VS-boundary.
- **(c) compile per batch-size class** — multiplies module count and JIT time
  (TODO item 3: the cubin cache is already at its cap and evicting).

### 3.2 What actually rides in each flush copy, and what must move for the copy to vanish

The flush copy is the whole owed staging span (`cp_upload_flush`, `cp_renderer.c:464-503`),
not the argument block alone. Every `cp_upload*` reservation between two launches on a
stream rides in it. Per link:

- **compact link** (65.7/frame heavy): the interp block (416 B) uploaded at
  `cp_renderer.c:4171` *for the FS's `args[14]`* — compact already takes it by value
  (`interp_params = {&interp}`, 4173). The copy disappears only if the FS reads the interp
  block from its own parameters — INTERP_INLINE's shape, by value instead of by upload.
  Nothing else is reserved in this span in the default configuration (p50 720 B = 416 B +
  256-B staging alignment padding). **Removable in full.**
- **FS link** (65.7/frame): the FS block (656 B + rows × 128) and, when the texture set
  changes, the hardware texture table (`count × 8`, `cp_renderer.c:3451-3468`, pointed to
  by slot 15). Both can be embedded. **Removable in full**, at the cost of also embedding
  the texture table (its pointer becomes a param offset).
- **VS boundary** (107/frame): the VS block (1,592 B + rows × 144 + 904 B fetch args), plus
  for a batch of >1 draw the **slice table** (`cp_upload` 5506, read by the fused fetch, by
  the rasterizer via `rast_args.rect_draw_slices` 5550, and by the FS interpolator via
  `cp->fs_batch.slices` 5515/4618), the **per-draw clip rects** (5547, read by the
  rasterizer), and the **vertex-buffer base table** when `batch->compact_rows`
  (`elem_bases_dev`, 5562, read by the fused fetch through `vf_args.elem_bases`,
  `cp_rast_types.h:1513`). The VS block and vb table can be embedded; **the slice table is
  consumed by three kernel families** and would have to travel by value to `cp_clip_rast_fused`,
  `cp_rasterize_stage*` and `cp_fs_compact` as well (up to 128 × 32 B = 4 KB, pushing those
  fixed kernels' by-value structs past 4 KB) — or stay a device upload, in which case the
  VS-boundary copy **remains for every batched draw**. favorite3 uses no push constants
  (host-visible-memory §3.2: "0 `vkCmdPushConstants`"), so the per-draw push blocks
  (`cpvk_cmd.c:2846-2863`) do not appear in this capture's spans; in a capture that pushes,
  they are UBO-pointer targets and would need a param-space UBO path.
- **Compute** (`cpvk_cmd.c:1318-1339`): same ABI, block + grid + push constants; push
  constants are reached through a UBO slot pointer into the block, which has no
  param-space equivalent without codegen changes. Rare in favorite3; must not break.

Tables that must stay in device memory regardless: batch rows (written by `cp_vertex_fetch`
or the interpolator on the device), the A-buffer/segment tables, everything the fixed
kernels write. Those do not force copies — they are scratch allocations, not uploads.

### 3.3 The nir_to_ptx change — LLVM 18 probe (MEASURED, `llc-18`, `byval.ll`)

The tree compiles with LLVM 18 (`/usr/lib/llvm-18`, `meson.build:63-72`) at `+ptx75`
(`cp_nir_to_llvm.c:3905, 4036`); `libLLVM-18.so` contains no `cvta.param` string. Four
kernels with a 16,912 B `byval` block:

| use of the block | emitted PTX |
|---|---|
| constant-offset loads (`cp_arg_slot`, 22 sites) | `ld.param.u64 %rd, [k_param_0+16]` — direct, no local memory |
| dynamic index by `tid` (UBO table `row*16+i`, draw params `row*4+f`) | `mov.b64 %rd1, k_param_0; … ld.param.u64 [%rd6+528]` — **works at ptx75, no `cvta.param` needed** |
| `select(param ptr, global ptr)` then load (the `BATCH_ROWS` "zero word in block or device array" shape, `cp_renderer.c:5841-5849`) | folded into two loads; 8-byte depot only |
| pointer passed to a non-inlined function | **`.local .b8 __local_depot3[16912]`** — a full per-thread copy of the declared block |

So: the codegen is feasible without `cvta.param`, and cheaper than today (ld.param vs
invariant global load — a small device-side positive). The hazards are (i) any pointer into
the block that escapes the inliner's view — the inline helpers `cp_fs_inline_lane(const struct
cp_fs_interp_args *…)` (`kernels/cp_fs_inline.c:101`) and `cp_vs_fetch_*(const struct
cp_vertex_fetch_args *…)` (`cp_vs_inline.c:87-115`, `cp_vf_lane.h:111,215`) take pointers
that would now be param-space; they are force-inlined in the same `LLVMContext`
(ARCHITECTURE §3.3-3.4) so `NVPTXLowerArgs` sees loads, but the guarantee must be checked
per shader with a `.local` grep, as DEAD_ENDS 28 does; (ii) the `BATCH_ROWS`/`BATCH_MASK`
single-draw convention (point at a zero word *inside the block*) has no param-space
address and must become a compile-time `mask ? rows[tid & mask] : 0`; (iii) the block's
self-references (`args[3]`, `args[7]`, `args[9]`, `args[10]`, `args[11]`, `args[14]`,
`args[16]`) become offsets — a new ABI, so `cp_shader_abi.h` static asserts and
`cp_rast_types.h:393-507` need a second layout, and ARCHITECTURE §8 #21 ("a new default
path keeps the old one runnable") means both ABIs stay compiled and flag-selectable.

### 3.4 Invariants, blast radius, gates

- Untouched: helper-lane suppression (#10), stable primitive order (#8), order-free opaque
  merging (#9) — no rasterization or shading-order change. Arena generations (#4, #5) — less
  pressure, but the slice/rect/texture uploads still use the ring; `batch_uploads_live`
  still guards them. Upload-span ordering under `cp_pass_gate_stream` — unchanged semantics
  for the copies that remain.
- Touched: shader ABI (#13, #15), inline-helper pointer plumbing (#16), two ABIs (#21), a
  registry switch (#19, #20), `cp_launch_after`'s PDL condition (3) becomes satisfiable on
  the compact link — but only `CUDAVK_COMPACT_PDL` has a prologue to use it; the FS has
  none (grep `griddepcontrol` in `nir_to_ptx/` = 0).
- Gates (TESTING.md): Gate 1 native suite incl. `cp_launch_audit`; Gate 2 the 18-sample
  sweep with the three flag runs; Gate 3 sentinel frames on **both** captures (the ABI
  touches every draw); Gate 4 device-side equivalence with negative control;
  `cp_debug_doc.py --check`, `cp_no_getenv.py`. Plus the DEAD_ENDS 28 `.local` census over
  every generated kernel under the new ABI, and DEAD_ENDS 26's palindromic method with a
  *full-link* instrument (predecessor end → kernel start, not copy end → kernel start).

### 3.5 Effort

High. New ABI in `nir_to_ptx` (arg-slot loads, batch row, UBO table, draw params, interp and
fetch embedding, compute), dual-ABI module build, three host block builders
(`cp_renderer.c:3688-3777, 5785-5891`, `cpvk_cmd.c:1310-1340`), the slice-table question for
three fixed kernels if the VS boundary is in scope, registry entry, four gates on two
captures. Order of one to two weeks for the shade-chain scope alone; the VS-boundary scope
is a further redesign of how the rasterizer and interpolator receive per-draw tables.

---

## 4. A cheaper decisive probe

The reports propose a `cuLaunchKernel`-vs-parameter-size microbenchmark first. That is
necessary for the >1 KB front-end question (§2.3) but it does not decide the lead. Two
probes together do, and neither touches the shader ABI:

1. **Hoist the two shade-chain reservations above the batch's earlier launch** — reserve the
   interp block and the FS block (and the texture table) before the stage-3 launch (or the
   VS launch) so they ride the copy that already exists at that boundary, leaving *no*
   stream op in either the stage3→compact or compact→FS link. Host-only change, ~50–100
   lines behind a `CUDAVK_NO_`-style switch, no ABI change, no `.local` risk. It removes
   exactly the 131 serial device-paced copies whose excess is the MEASURED 0.30 ms/frame
   (MODEL 0.41), and it makes `CUDAVK_COMPACT_PDL` admissible for the additional
   0.03–0.12. host-launch-audit priced this alternative at 0.04 using the 0.59 µs merge
   price — the same host-side mis-pricing as §1.3: what it removes from the device chain is
   the same excess lead E removes there. If it measures under ~0.2 ms on the heavy band
   with the palindromic method, lead E cannot reach 0.5 either, because the parameter
   design adds only the VS-boundary copies (model ≤0.32, host/gate-paced, slice-table
   blocked) on top. If it measures ≥0.3, the parameter design is not needed for the shade
   chain at all. Precondition to check in code: the FS block's inputs (`fs_in`, `fs_out`,
   counter, discard mask, texture table, `fs_in_stride`, rows) and the interp block's
   (`out_prim_list`, slices, `batch_rows`) are all knowable before stage 3 on the direct
   path — the trace shows no readback between stage 3 and compact (99.3% device-paced).
2. **Microbenchmark on sm_120**: device-paced kernel→kernel gap with a secondary carrying
   8 B / 1 KB / 4 KB / 8 KB / 20 KB / 32 KB parameters (`%globaltimer` at entry), host
   `cuLaunchKernel` cost per size, and an nsys check that a >4 KB launch produces no
   `CUPTI_ACTIVITY_KIND_MEMCPY` row. Decides §3.1(a) vs (b) and the "hidden staging"
   question. One to two hours, no driver change.

Zero-GPU checks already done here: LLVM 18 lowers dynamic param indexing without
`cvta.param` at ptx75 and copies the block to `.local` on any escape (`byval.ll`).

---

## 5. Must-fix list for the lead as written

1. Replace the variable-size-parameter premise with the fixed-size design (§3.1 b): cap the
   in-param table rows, keep a device table for larger batches, and subtract the copies
   that therefore remain (~10/frame heavy, all VS-boundary).
2. Price the three copy classes separately: shade chain by device-side excess
   (MEASURED 0.30 capped / MODEL 0.41 heavy), VS boundary by the model only (≤0.32, with
   the slice-table dependency stated), other 16/frame excluded.
3. Remove the replay's post-copy-gap over-credit (~0.12 heavy) or state it.
4. State that the FS has no PDL prologue; the "PDL becomes admissible" bonus is the compact
   link only (0.03–0.12).
5. Carry DEAD_ENDS 26's +0.13 as an unretired risk and instrument the *whole* link.
6. Run the `.local` census on every generated kernel under the new ABI before any timing
   claim (LLVM 18 escapes cost a per-thread copy of the declared block).
7. Run probe 1 (hoisted reservations) before building the ABI; it captures the measured
   prize at a fraction of the cost and is the negative control for the larger design.
