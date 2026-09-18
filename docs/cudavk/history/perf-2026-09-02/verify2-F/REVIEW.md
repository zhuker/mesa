# Verifier review — Lead F: per-batch geometry run-ahead

Read-only, 2026-09-02. Scratch: this directory (`variants.py`, `postpin_var.log`,
`postdead_var.log`, `*_var.json`). Nothing built, run on the GPU, or edited in the repo.

## Verdict: OVERSTATED

The published ceiling (-1.80 ms/frame heavy band, -1.04 light, "realistic about half"
~0.9) rests on letting `cp_clip_rast_fused` (clip + raster stage 1) start ahead of the
previous batch's shade chain. That is not legal in this driver: stage 1 reads the depth
buffer that the previous batch's `cp_fs_writeback` commits, writes the one shared
visibility buffer the previous batch's `cp_fs_compact` is still reading, and appends into
the queue set the previous batch's stage 2/3 are still draining. Only the VS (and the
classic `cp_vertex_fetch`) can legally run ahead, and even that carries a hidden
write-after-read hazard on the queue-set counters that the published model does not see
(the fused VS zeroes them under `fetch_fold`). Re-running the same dependency-respecting
replay with the illegal part removed gives **-0.45 ms/frame heavy-band median (-0.42
mean), -0.28 light**, rising to -0.69/-0.35 only if the VS's coalesced argument copy moves
to the lane as well — a credit that lead 4.4 (argument copies) already claims. After the
counter-seed hazard and per-batch event overhead the realistic figure is **0.3-0.5 ms
heavy band, 0.2-0.3 light**, i.e. 2-3x below the published "about half".

The lead is not refuted: the VS-only mechanism has a legal mechanism, a measured pool
(0.437 ms/frame of main-stream VS time sitting behind a predecessor), and a cheap probe.
It simply does not carry the frame anywhere near 5.0 on its own.

## 1. PRECEDENT

| entry | applies? | how |
|---|---|---|
| DEAD_ENDS 27 (scope chain) | no refutation | scopes, not batches; the lead's "why different" holds on that point |
| PERF §6 item 4 (run-ahead across the drain) | **prices the queue-set constraint** | "Sharing queue sets is refused by `fetch_fold`'s required seed(A) → count(A) → seed(B) → count(B) ordering" — the identical hazard appears here inside a burst (see §3.2), and the ladder there says +19.2 MB per extra set. Its 0.092 collected of 0.387 ceiling is the closest measured analogue: a VS relocation converting at ~1.0 but small, because vertex work is a small share of the chain. Overlaps 4.1 (the closure audit says so itself) |
| PERF §4 prices | cross-stream hand-off "~32 µs/episode" | does **not** transfer per batch: measured join latency on main in the heavy trace is p50 2.2 µs, p90 28.9 µs (n 4.33/frame, dependency-binding cases only; `variants.py`), and adding +2/+5/+10 µs per VS to the model moves the credit by < 0.01 ms. The 32 µs figure was a frame-time price of a gate-in/join-out pair on the old capture, not a device latency |
| PERF §4 small clear 0.66-1.01 µs | prices the fallback for the seed hazard | 46-66 batches x 1 memset of 3 words if the seeding goes back to a main-stream clear: 0.05-0.07 ms/frame heavy |
| DEAD_ENDS 22 merge rule | not applicable | this adds concurrency, removes no launches; nothing to divide by an overlap factor |
| DEAD_ENDS 31/32/33/35 | not applicable | different pools; 32 is about speculative *shading*, this is VS only |
| RESIDUAL_AUDITS VS→clip fusion | the closure audit corrected its "host issue-floor idle" premise; independent of this lead | — |
| ARCHITECTURE §8 inv. 4/5 (arena generations, `batch_uploads_live`) | must hold on the lane | the flush-generation retire event is recorded on main; a lane VS reading a generation's rows must be covered (record after the lane join) — same problem the pass side streams already solve |
| ARCHITECTURE §6.3 upload-span ordering (`cp_pass_gate_stream`, 7605-7642) | must hold | if the VS's flush copy goes on the lane, every later reader of that span must be behind the lane event; main is (clip waits on it), side segments are transitively (gate recorded on main after the wait) |

## 2. ARITHMETIC (MODEL, same replay as the closure audit; MEASURED where marked)

Model check: variant A reproduces the published -1.8008 heavy / -1.0380 light exactly.

| variant (indep set) | postpin heavy median | mean | postdead light median |
|---|---:|---:|---:|
| A published: VS + fetch + clip, all streams | -1.801 | -1.580 | -1.038 |
| B published VS-only, all streams | -0.780 | -0.734 | -0.521 |
| **C VS only, main stream only (the legal, non-double-counted set)** | **-0.449** | **-0.416** | **-0.282** |
| D VS only, side streams only | -0.327 | -0.324 | -0.232 |
| E VS + fetch, main only | -0.471 | -0.441 | -0.312 |
| F VS + its arg copy, main only | -0.691 | -0.625 | -0.354 |
| G delete VS main only (upper bound on C) | -0.464 | -0.439 | -0.333 |
| H VS + clip, main only (illegal) | -1.035 | -0.921 | -0.612 |
| I clip only, main (the illegal share) | -0.539 | -0.509 | -0.374 |
| J = C + 2/5/10 µs join latency per VS | -0.449/-0.449/-0.447 | | -0.279/-0.275/-0.256 |

Reading the decomposition:

- **B → C: the side-stream half (-0.33) is not this lead's.** Side-stream VS launches
  (62.8/frame heavy) belong to opaque fan-out segments and blended segments, which wait on
  `pass_gate`. The model's `indep` ignores every `cuStreamWaitEvent`, so D lets them start
  before the gate — i.e. before the upload copy carrying their rows (illegal, the
  `multithreading` loss documented at `cp_renderer.c:7605`) and before segment 0's stage 3
  (kernel-chain L2, 0.20-0.24 ms, a separate lead). Double counting with L2.
- **H → C: the clip half (-0.54 to -0.59) is illegal** (§3.1). It is ~55% of the
  published VS+clip main-stream credit.
- **F − C = 0.24 ms is the argument copy leaving the main chain**, which lead 4.4 counts
  in its 0.80 (memcpy HtoD) figure. Quote it once.
- **C ≈ G**: run-ahead collects essentially the entire main-stream VS pool, so the model
  is internally consistent and 0.45 is a ceiling, not a mid-estimate. MEASURED cross-check
  from the trace: 46.18 main-stream VS launches/frame heavy band; 0.18/frame are issued to
  an idle main stream (not hideable); the rest add 0.4367 ms/frame of device time after
  their predecessor (gap + duration) — the pool C removes.
- Mean vs median: heavy-band mean is 0.03 lower than the median; whole-window (owner's
  6.68 convention) sits between light 0.28 and heavy 0.45, so **~0.35 ms whole-window**.
- CUPTI: host inflation 1.28x makes the untraced host run *further* ahead, which cannot
  shrink a credit bounded by device time on the main chain (G). Not a concern here.

### B200 (HYPOTHESIS — no trace)

Kernel time scales 0.91-1.17x (b200 §2.4) so the VS pool is ~0.4-0.5 ms; the frame there is
paced by host/launch latency (idle 5.1-6.3 ms), so hiding device work behind device work
buys less per µs but also removes 46 main-stream launch gaps at the unknown L_B (4-8 µs
→ 0.2-0.4 ms). Quote **0.3-0.6 ms heavy band, unmeasured**; take a B200 trace under nsys
2026.4.1 before sizing.

## 3. FEASIBILITY

### 3.1 Which stages may legally run ahead of batch k's chain

| stage of batch k+1 | reads/writes shared with batch k | legal ahead of k's shade chain? |
|---|---|---|
| generated VS (`cp_renderer.c:5916`) | reads app vertex/index/UBO rows (immutable once submitted); writes `vs_output_buf`, a per-batch dscratch bump allocation (`cp_renderer.c:5357`) — **no rotating slots are needed, they exist by construction** | **yes**, subject to §3.2 |
| `cp_vertex_fetch` (5.83/frame) | as VS, plus `vs_input` memset | yes, subject to §3.2 |
| `cp_clip_rast_fused` = clip + stage 1 (6941) | stage 1 `emit_fragment` **reads `args->depthbuf`** when `depth_test` (`cp_rasterize.cu:668-683`), wired from `cp->depthbuf`/`state->depth.depth_enabled` (`cp_renderer.c:5187-5188`); `cp_fs_writeback` **writes `depthbuf`** for the samples the fragment won (`cp_fs.cu:1144-1162`) → RAW on k's writeback. Writes the single `cp->visbuf` (`cp_renderer.c:5082`, one allocation at 9636) that k's `cp_fs_compact` reads → WAR. Reads `reject` written by k's writeback for discard-retry passes (`cp_tri_rejected`, `cp_rasterize.cu:614-627`). Appends to `cur_qset` that k's stage 2/3 drain → WAR | **no** |
| stage 2/3 | same queue set and visbuf | no |

The closure audit's own phrasing — "queue-set reuse hazards (clip(k+8) vs stage3(k) on set
s%8)" — treats the clip as movable with more sets. Sets do not fix the depth RAW: a
correct stage 1 for batch k+1 needs k's committed depth unless the stage-1 test is
dropped and re-done at writeback, which changes what the visbuf holds for every batch
whose depth function is not the order-free four and interacts with the reject buffer. That
is a renderer change, not a scheduling change, and it is not what the lead proposes.

### 3.2 Hidden work in the legal part (VS only)

1. **Queue-counter seeding (`fetch_fold`, default on).** The fused VS zeroes the three
   raster queue counters and the clip counter itself (`cp_vs_inline.c:106-126`,
   `cp_vertex_fetch.cu:27-37`, host side `cp_renderer.c:5636-5677`,
   `vf_args.seed_counts = cp->cur_qset.counts`). A lane VS(k+1) would zero the counters
   stage 2/3 of batch k are still reading. Options: (a) a second direct-path queue set,
   alternated per batch, with VS(k+2) waiting on an event after stage3(k) (one wait per
   batch; +19.2 MB); (b) seed the queue counters with a 3-word `cuMemsetD32Async` on main
   before the clip and let the lane VS seed only the clip counter (a per-batch dscratch
   word, no hazard): 46-66 x 0.66-1.01 µs = 0.05-0.07 ms/frame heavy (PERF §4 price).
   `CUDAVK_NO_FETCH_FOLD` already exists and prices the fallback with no code.
2. **Per-batch event pair.** `cuEventRecord(lane)` + `cuStreamWaitEvent(main)` per direct
   batch: 2 x ~1-1.5 µs host, 46-66/frame → 0.1-0.2 ms of host issue; the closure audit's
   host x0.5 sensitivity (-0.81 heavy) says the host is worth ~0.5 at the margin →
   ~0.05-0.1 ms/frame cost.
3. **Flush copy placement.** The VS launch is a flush point; the coalesced span (VS block +
   per-draw rows) must be sent on the lane, so `cp_stream_set`/`cp_upload_flush` must
   learn a target stream (they already do for side segments). Later main readers are
   behind the join; side segments' gates are transitively ordered.
4. **Arena generations (inv. 4/5).** The retire event must be recorded after the lane is
   joined, else `cp_flush()` can rewind under a lane VS still reading rows.
5. **Failed-append back-outs** (kernel-chain L1, 8.65/frame light) re-run the VS: the lane
   must handle a VS re-issue for the same batch without breaking the join order.
6. Not a cost: VS-output memory (bump allocation), 8-set aliasing (VS-only needs no
   queue set beyond item 1), drain-crossing machinery (the model and this lead operate
   inside a host burst; nothing is held across a blocking sync).

Net realistic: C 0.45 − 0.05..0.07 (seed) − 0.05..0.1 (events) ≈ **0.3-0.35 ms heavy
band**, plus the 0.24 arg-copy credit only if lead 4.4 is not also counted.

### 3.3 Tests

TESTING.md gates: native suite (79/79 at last count) with `cp_launch_audit`; the 18-sample
sweep; 18 sentinel frames on both captures; device-side equivalence with the negative
control (`CUDAVK_VFETCH_SKIP_SEED` is the existing negative control for the seeding and
must still fail). Flag in the `cp_debug.c` registry, `cp_debug_doc.py --check`,
`cp_no_getenv.py`. A/B alternating in one session, hash and submit count on every run,
both captures.

### 3.4 Cheaper decisive probe

The replay already bounds the legal mechanism at 0.45 heavy / 0.28 light (variant C).
The one thing not yet priced without code is the seed fallback: an A/B of
`CUDAVK_NO_FETCH_FOLD=1` on both captures (existing flag) gives the cost of option (b) in
one session. Then the build itself is the probe: `CUDAVK_VS_LANE=1` issuing the
direct-path VS (+ its flush) on one existing pass side stream, event-joined before the
clip, seeding via main-stream memset; 3 rounds alternating, hash + sentinels. Expected
resolvable signal: 0.3-0.45 heavy band against a ~0.12 session spread — resolvable, unlike
§6 item 4's 0.092.

## 4. Must-fix in the lead's statement

1. Drop clip/stage 1 from the run-ahead set: depth RAW (`cp_rasterize.cu:668` vs
   `cp_fs.cu:1144`), visbuf WAR, reject RAW. Legal set is VS (+ classic fetch).
2. Split the side-stream VS credit (-0.33) out: it is the pass gate / L2 lead and partly
   illegal as modelled (rows before their copy).
3. State the `fetch_fold` counter-seed hazard and its price (memset fallback 0.05-0.07 or
   a second queue set + one event per batch).
4. Quote the arg-copy share (0.24) once — here or in lead 4.4, not both.
5. Replace "N rotating VS-output slots" with the actual constraint: `vs_output_buf` is a
   per-batch dscratch allocation already; the shared state is the queue set.
6. Ceiling: 0.45 heavy / 0.28 light (MODEL), realistic 0.3-0.5 heavy, ~0.35 whole-window;
   B200 unmeasured.

## 5. Reproduction

```
cd /tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/verify2-F
python3 variants.py /tmp/postpin-rtx2/trace.sqlite postpin 0 0 postpin_var.json
python3 variants.py /tmp/postdead/trace.sqlite postdead 9286588499 12338994337 postdead_var.json
```
`variants.py` is the closure audit's `trace_audit.py` lines 1-520 (unchanged model) plus the
variant runs and two trace statistics (join latency on main; main-stream VS exposure).
