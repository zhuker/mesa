# Peel-pass predication (S2c) — CUDAVK_PEEL_PREDICATE

Worktree: `/tmp/peel-tree`, branch `peel-predicate`, commit `d133edc3ce1`,
based on `e2fea470d04`. Build: `/tmp/peel-tree/build-cudavk-peel`, links, ICD at
`build-cudavk-peel/src/cudavk/cudavk_devenv_icd.x86_64.json`.
Flag: **`CUDAVK_PEEL_PREDICATE=1`**, value boolean, default **off**.

Design: `/tmp/perf16/iter29-sync/design.md` §5 (S2a/S2b/S2c), which the
addendum §4 keeps intact. Not measured here — no replay, no benchmark, no CUDA
program was run. Generated PTX was read (see §7).

---

## 1. What the wait costs today, and where it is

Fresh numbers at HEAD, supplied by the measuring agent and superseding the
docs:

| | old capture | Crossroads |
|---|---:|---:|
| frame | 13.1626 ms | 5.8230 ms |
| peel checks | 2.751 ms/frame over 1.70 waits, 1.6135 ms each, **20.8% of the frame** | **zero peel checks** |

The opaque fan-out (`20611f5b131`) took its 2.55 ms out of the episode drain
and did not touch the peel site, so this lead is intact and relatively more
valuable than `docs/cudavk/PERFORMANCE.md` §6 item 1 says.

The site, verified unchanged at HEAD before editing: host store
`cp_renderer.c:6512`, drain `:6689`, host read `:6692`, doubling `:6695-6696`.

**Crossroads runs zero peel checks, so this lead is old-capture-only by
construction.** A neutral Crossroads result is the expected outcome, not a
failure. The two-capture rule is satisfied by "no regression on Crossroads",
not by a win there; stated here up front so the harness verdict is not read as
a wash.

---

## 2. Mechanism

Three parts. All three are needed; none of them is useful alone.

### S2a — a whole peel pass predicated on a device flag

Stages 1, 2 and 3 already carry `path_flag`/`path_value`
(`kernels/cp_rast_types.h`, `cp_rasterize.cu`): each returns unless
`*path_flag == path_value`. The opaque tiled episodes added it. The peel loop
now sets `rast_args.path_flag = cp->peel_prev`, `path_value = 1` for every pass
after the first.

Four things needed the same guard:

| what | before | after |
|---|---|---|
| per-pass visibility clear | `cuMemsetD32Async(visbuf, 0xFFFFFFFF, w*h*2*samples)`, 7.4 MB at 720p, **cannot be predicated** | `cp_peel_clear_visbuf`, one grid-strided launch, blocks return on the flag |
| `cp_peel_advance` | 3,600 blocks walking the whole visibility buffer | same kernel, one extra `const uint32_t *path_flag` parameter, null for every other caller |
| interpolation / compaction | `cp_fs_interpolate` or `cp_fs_compact`, one thread per screen quad | same kernels, guarded on `args.path_flag` |
| fragment shader extent | slot counter, `cuMemsetD32Async(counter, 0, 1)` every pass | **unchanged** |

The fragment shader is deliberately **not** touched. Its extent is the device
slot counter; the counter is memset to zero at the head of every
`cp_shade_fragments`; a compaction that returns leaves it at zero; the shader
and the writeback both already read that extent from the device and their tail
threads return. So a predicated pass shades zero slots and a side-effect
shader performs zero stores — the exactness a colourless peeled draw needs.
This is the same guarantee `cp_abuf_shade` already relies on.

### S2b — the interval reset moves to the device

`*(volatile uint32_t *)cp->peel_any = 0` becomes `cp_peel_pass_flag`, a
one-thread kernel on `cp->stream`, launched at the head of every pass. Once the
host stops draining every interval, a host store to `peel_any` races the
kernels of a pass still in flight; a device reset is ordered with them by the
stream.

Two words, not one, because a pass cannot both read the flag as its predicate
and write it as its result:

```c
*prev = init ? 1u : *any;   /* what the previous pass did */
*any  = 0u;                 /* this pass starts clean */
```

`init` is pass 0, which has no predecessor and always runs. The new word is
`cp->peel_prev`, one managed uint32 allocated beside `cp->peel_any`.

Monotone by construction: once `*prev` is zero the pass writes nothing, so
`*any` stays zero and every later pass is predicated off too.

### S2c — predict the trip count, confirm once

| arm | condition | schedule | first check |
|---|---|---|---|
| free | `peel_passes <= CP_PEEL_FREE_MAX` (4) | run all passes predicated | **never** |
| predicted | ring hit | first check at `MIN2(previous trip count, CP_PEEL_PREDICT_MAX)`, then today's doubling | once |
| cold | ring miss or eviction | today's schedule, unchanged | as today |

`peel_passes = MIN2(CP_BLEND_LAYERS, MAX2(num_triangles, 1))`, so a blended
full-screen quad is two passes and *cannot* need a check: the loop's own bound
is tighter than anything the device could report. That arm alone is the
plausible bulk of the win — 1.70 checks/frame is the signature of one or two
short peeled draws, and a two-pass draw pays exactly one 1.61 ms check today
to be told what its triangle count already said.

The predictor is a 64-entry direct-mapped host ring keyed on
`(vs, fs, num_triangles)`, holding the number of passes the loop ran. It stores
the *pass count the loop ran*, not the true convergence pass — a delayed check
cannot tell them apart — which makes it a **fixed point** instead of a value
that drifts upward by the slack every frame. A miss or a collision is the cold
case and costs nothing but the prediction.

---

## 3. The cap, and what happens at it

**The cap is the whole risk.** A predicated pass is not free:

| per wasted pass | count | cost |
|---|---:|---:|
| launches (flag, clear, stage 1/2/3, counter memset, compaction, shader, writeback, advance, argument upload) | ~11 | < 1 µs each |
| idle-block grids (shader 4,096 capped, writeback 2,048 capped) that load the counter and return | 2 | well under 1 µs |
| `discard_mask` clear, 1.8 MB, only for a shader with `discard` | 0–1 | ~1 µs |

Budget **~10 µs per wasted pass** against **1.6135 ms per removed check** — a
check is worth about 160 wasted passes. But unbounded overshoot loses: a
256-layer draw that converged at pass 30 would run 226 wasted passes ≈ 1.8 ms
and lose outright. So the bound is part of the mechanism, not a tuning knob.
Two named constants, both in `kernels/cp_rast_types.h` beside
`CP_PEEL_CHECK_MAX`:

* **`CP_PEEL_FREE_MAX = 4`** — the largest loop that runs with no check at all.
  Worst case **3 wasted passes ≈ 30 µs**, against one removed 1.61 ms check.
* **`CP_PEEL_PREDICT_MAX = 16`** — the most passes launched without a check
  when relying on a prediction. A *wholly* wrong prediction therefore wastes at
  most **16 passes ≈ 160 µs**, a tenth of the single check it is buying.

**At the cap the loop falls back to today's schedule.** `interval_start` and
`check_interval` continue exactly as they do now (`check_interval =
MIN2(check_interval * 2, CP_PEEL_CHECK_MAX)`), so a draw whose prediction was
too low, or which never converges, degenerates to today's behaviour plus the
bounded overshoot above. There is no state in which the loop runs unchecked for
more than `CP_PEEL_PREDICT_MAX` passes, except the free arm, where the loop
bound itself is `CP_PEEL_FREE_MAX`.

Worst case for a whole frame: a frame with 4 peeled draws all mispredicting is
4 × 16 = 64 wasted passes ≈ 0.64 ms — still under the 2.751 ms of checks it
would be removing, but this is the number to watch if the measured result is
negative.

---

## 4. What is byte-identical, and why

### Flag off

`cp_debug->peel_predicate` is false, so `peel_predicate` is false, so:

* the host store at the head of the interval is still a host store;
* the visibility clear is still `cuMemsetD32Async`;
* `check_interval` starts at 1 and doubles, `interval_start` moves as it does
  now;
* `path_flag` is 0 at every call site, and every guard is
  `if (args.path_flag && ...)`, so no guard is taken;
* `cp_peel_advance` gets a null flag pointer and returns to its old body after
  one compare;
* nothing calls `cp_peel_pass_flag` or `cp_peel_clear_visbuf`, and no
  prediction is read or written.

The residue with the flag off is **16 bytes of `cp_fs_interp_args`** (408 →
424, the ABI asserts in `cp_fs_inline.c` are updated and the new field is
asserted at offset 408) and **two predicate branches that are provably not
taken**. Same launches, same order, same pixels. This is not a claim of
instruction-level identity in the kernels — the guard exists in the binary in
both arms, which is what makes the A/B a clean one-binary comparison.

### Flag on

Scheduling and trip count change; output does not. The argument is the one
already written in the driver above the check (`cp_renderer.c:5867-5871`):

1. **A pass after convergence is already a no-op for output today.** Its
   rasterizer selects nothing (`emit_fragment` returns before the `atomicMin`
   when `tri_id < peel_next[pixel]`), its interpolator compacts zero fragments,
   its shader shades zero slots, its writeback writes nothing. The predicate
   changes what such a pass *costs*, not what it *does*.
2. **`peel_any` is monotone.** `peel_next` only moves forward, so once a pass
   composites nothing every later pass does too. This is what already makes
   `CP_PEEL_CHECK_MAX` overshoot legal.
3. **No host store participates.** `peel_prev` is written by a device kernel on
   the stream and read by the pass's kernels behind it, in stream order.
4. **The loop bound is unchanged.** `passes = peel_passes` either way. Running
   to it instead of breaking early is bit-identical by (1).
5. **The visibility buffer ends in the same state.** The first
   non-compositing pass runs in full and leaves it all-`VISBUF_EMPTY`; the
   predicated passes after it skip the clear and write nothing, so they leave
   it exactly as today's `break` leaves it.
6. **Nothing here is a device-side decision.** The device evaluates a predicate
   the host has already committed to. It never chooses between two rendering
   paths.

### Where predication declines

`peel_predicate` is false — and today's path runs — for an appending pass
(which never wires `peel_any` into the rasterizer, so its flag would read zero
and predicate the draw away), for `CUDAVK_FRAG_CENSUS`, for the A-buffer peel log,
and whenever the kernels are built instrumented. Those record per pass and
would gain entries for passes that ran as no-ops.

---

## 5. The exact A/B command

New behaviour is **not** the default, so the **candidate arm carries the flag**
and the control arm is empty (WORKFLOW §4.3: `delta = control - candidate`, and
putting the new behaviour on the control side inverts the verdict silently).

```bash
# base environment is EMPTY; CUDAVK_TEXTURE_CACHE no longer exists.
# CUDAVK_ABUF_FUSE_CHECK must be UNSET in every timed run: it is an
# equivalence gate and it SYNCHRONISES. So must CUDAVK_ABUFFER_VERIFY,
# CUDAVK_ABUFFER_TIMING and CUDAVK_DEBUG_WORK, for the same reason.

MESA=/tmp/peel-tree
ICD=$MESA/build-cudavk-peel/src/cudavk/cudavk_devenv_icd.x86_64.json
GFX=~/gfxreconstruct/build
PLUG=~/claude-scratchpad/perf16/fps_plugin.so
OLD=~/headless_streamer_20260814T155742.gfxr
CROSS=~/headless_streamer_1818_20260817T173522.gfxr
D=/tmp/perf-audit/peel-ab

run() { # run <name> <capture> [env...]
  n=$1; cap=$2; shift 2
  mkdir -p "$D/$n"
  env VK_DRIVER_FILES=$ICD "$@" \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
      --replay-event-plugin-path $PLUG \
      --replay-event-plugin-params "$D/$n/submits.txt" \
      "$cap" > "$D/$n/stdout" 2> "$D/$n/stderr"
  echo $? > "$D/$n/rc"
  sha256sum "$D/$n/stdout" | cut -d' ' -f1 > "$D/$n/stdout.sha256"
  wc -l < "$D/$n/submits.txt" > "$D/$n/submits.count"
}

# strictly alternating, one session, one binary: 6 per arm on old, 4 on Crossroads
for i in 1 2 3 4 5 6; do
  run "old-cand-$i"   "$OLD"   CUDAVK_PEEL_PREDICATE=1
  run "old-ctrl-$i"   "$OLD"
done
for i in 1 2 3 4; do
  run "cross-cand-$i" "$CROSS" CUDAVK_PEEL_PREDICATE=1
  run "cross-ctrl-$i" "$CROSS"
done
```

Before starting: `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv,noheader` must be empty.

Frame time is `median(diff(submit_ts[::2])[50:])` — every **other** submit,
skip **50** (`/tmp/perf16/cp_two_replay_report.py`).

Two acceptance gates that come before the number:

* **stdout sha256 must match between the arms, per capture.** A different hash
  is usually a run that died, not a small correctness question.
* **submit count must be 3,022 on a complete old-capture replay.** Iteration
  29's S0 probe first read as +8.07 ms and was a device loss after 26 frames:
  52 submits against 3,022.

Useful but optional, on one run per arm:
`CUDAVK_PLAN_STATS=1` prints the wait table — `peel checks %.1f ms/%PRIu64` is
the line whose *count* should drop from 1.70/frame toward 0. That line is the
mechanism check; the frame median is the verdict.

---

## 6. Expected result, per capture

| capture | expectation |
|---|---|
| **old** | **0.20–0.50 ms** faster (13.1626 → 12.66–12.96 ms). Blocked time removed 1.4–1.9 ms of the 2.751 ms; only the part of it that is not overlapped by the device turns into frame time. `peel checks` waits/frame should fall from 1.70 to at most a few tenths. |
| **Crossroads** | **neutral.** It runs **zero** peel checks, so there is nothing here for it to remove. `wait_peel_n = 0` in both arms. The expected delta is 0.00 ms inside session spread (about 0.03 ms), and this is the *designed* outcome, not a failure. Its job is to show no regression. |

---

## 7. What was checked without a GPU

* Builds clean: `ninja -C /tmp/peel-tree/build-cudavk-peel`, no warnings, ICD
  produced.
* `src/cudavk/tests/cp_debug_doc.py --check` — **FLAGS.md is up to date** (119
  switches; regenerated and committed).
* `src/cudavk/tests/cp_no_getenv.py --check` — **PASS**, 36 driver sources,
  every switch through the registry. No `getenv` was added anywhere.
* `meson test -C build-cudavk-peel cp_launch_audit` — **OK**. Every new launch
  goes through `cp_launch`/`CP_LAUNCH`.
* **`meson test --suite cudavk` was NOT run.** 66 of its 67 tests are Vulkan
  programs (`cpvk_*`) that run against the built ICD and therefore create a
  CUDA context and launch kernels on the GPU, which this task forbids. Only
  `cp_launch_audit` is a static check, and it was run. The suite must be run by
  whoever owns the GPU before this lands.
* **Generated PTX read** (compiled through NVRTC 12.8 at `compute_120`, the
  same path the driver uses, sources in `/tmp/peel-ptx/`):
  - `cp_peel_pass_flag` — 30 instructions, one `ld.volatile.global.u32` and
    two `st.volatile.global.u32`, all other threads branch to `ret`.
  - `cp_peel_clear_visbuf` — the flag load and `setp.eq.s32 → bra ret` are the
    first thing in the kernel, ahead of any address arithmetic; the store is
    `st.global.u64 -1`, i.e. exactly `VISBUF_EMPTY`, matching the
    `0xFFFFFFFF` memset it replaces.
  - `cp_peel_advance` — `setp.eq.s64 %p1, flag, 0` then the volatile load and
    early `bra`, ahead of the coordinate computation.
  - `cp_fs_interpolate` and `cp_fs_compact` — parameter block is 424 bytes,
    `ld.param.u64 [param+408]` is the flag, and the guard is the first thing
    after the local-depot setup, before any global load. Offset 408 is the one
    the `cp_fs_inline.c` `_Static_assert` now fixes, so an ABI drift is a
    compile error, not a runtime pointer-decode surprise.

---

## 8. What would falsify the design

Any one of these means the design is wrong, not that the run was noisy:

1. **The old capture gets slower.** The mechanism claims a wasted pass costs
   ~10 µs against a 1.6135 ms check. A negative delta on old says the
   per-wasted-pass price is far higher than the launch price the project has
   measured (`< 1 µs`, iteration 27) — or that far more passes are being wasted
   than the caps allow, which is a bug in the cap and should show as
   `passes_run` growth.
2. **`CUDAVK_PLAN_STATS=1` shows the peel-check count unchanged.** Then
   predication is not being reached at all: either every peeled draw has
   `peel_passes > 4` with a cold ring, or `peel_predicate` is declining for a
   reason listed in §4 (appending pass, instrumented kernels). The lead would
   then be real but unclaimed by this patch.
3. **The stdout hash differs between the arms on either capture.** The design
   claims output cannot change. A different hash means either the run died
   (check the submit count first — this is the usual cause) or claim (1) of
   §4 "flag on" is false, which would mean a pass after convergence is *not*
   a no-op for output. That would refute the driver's own comment, not just
   this patch, and would be worth more than the 0.3 ms.
4. **Crossroads regresses beyond session spread.** It runs zero peel checks, so
   nothing in this patch should execute on it at all. A regression there means
   the flag-off cost — the 16-byte argument block and the two untaken branches
   — is not free, or that `peel` is reached far more often on Crossroads than
   the wait counters say.
5. **The frame time improves by much more than 0.50 ms.** Too good is a trap
   here: check that the submit count is 3,022 and that the capture ran to
   completion. Iteration 29's S0 probe read +8.07 ms because it died after 26
   frames.
