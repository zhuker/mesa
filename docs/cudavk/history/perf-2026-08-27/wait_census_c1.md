# C1 — price every host wait in the driver, in one run

`CUDAVK_WAIT_CENSUS=1`, bool value, default off. Instrumentation only: no
kernel, no schedule change, no allocation, one report at teardown.
Worktree `/tmp/peel-tree`, branch `peel-predicate`.

This is the probe `PERFORMANCE.md` §6 item 4 has asked for since before this
session — the iteration-29 addendum's **P2** — generalised from the peel site,
where the same two instruments closed a lead.

---

## 1. What it measures, and at how many sites

Two instruments, at **eight** sites, not five. The three extra ones are not in
the plan-stats table, and they have to be here: the ceiling is measured as the
gap from one block to the next, so **a blocking call the census does not know
about is silently counted as host issue time and inflates every neighbouring
site's answer**.

| # | site | where |
|---|---|---|
| 1 | episode drain | `cp_sync_timed`, `cp_renderer.c:8909` |
| 2 | quad counters | `cp_sync_timed`, `:7181` |
| 3 | peel checks | `cp_sync_timed`, `:7129` |
| 4 | segment counters | `cp_sync_timed`, `:6769` |
| 5 | descriptor upload | blocking `cuMemcpyHtoD` per submit, `cpvk_device_memory.c` |
| 6 | `vkDeviceWaitIdle` | `cuCtxSynchronize`, `cpvk_DeviceWaitIdle` |
| 7 | upload-arena rewind | `cuCtxSynchronize`, `cp_renderer.c:403` |
| 8 | scratch reclaim | `cuCtxSynchronize`, `cp_renderer.c:571` |

What is left unhooked is debug-only or teardown. Sites 1–4 recover their
identity from which `plan` counter the caller passed, so the four call sites are
unchanged.

**Instrument A — `cuStreamQuery` before the wait.** Was there *any* outstanding
device work when the host arrived? Reported as `ready`. A site that is never
ready is device-paced. Only meaningful for a stream wait; sites 5–8 are
whole-context synchronises and report no ready count.

**Instrument B — `Σ min(issue burst after the wait, the wait)`.** A deferral
recovers at most the host work that would have been issued during the wait.
Reported as `CEILING`, with `gap-bound` / `wait-bound` counts: how often the
host ran out of work before the wait ended, versus the wait being the binding
term. **A site whose gaps are shorter than its waits is limited by how little
the host has to do, and no deferral scheme changes that.**

Output, one line per site plus a total:

```
cudavk: WAIT CENSUS over N render scopes. ...
cudavk:   episode drain     n=...  ... ms  mean ...  ready 0/...  gap ... ms (max ...)
                            CEILING ... ms = ...% of blocked, ... ms/scope
                            gap-bound ... wait-bound ...
```

---

## 2. PREDICTIONS, per site, before the run

Measured anchors: at HEAD the episode drain is **5.989 ms/frame** and the peel
site produced **ceiling 0.376 ms/frame, ready 0 of 2,577, conversion ≈ 11%**.

| # | site | prediction | falsifier |
|---|---|---|---|
| 1 | **episode drain** | `ready = 0`. Ceiling **0.8–2.0 ms/frame**, i.e. **15–35% of blocked** — several times the peel site's 0.376 ms, because a drain sits *between* episodes with a whole episode's issue in front of it while a peel check sits mid-loop with one pass (~11 launches, tens of µs). **Mostly `gap-bound`**: the host runs out of work before the 0.6 ms wait ends. | Ceiling below 0.3 ms/frame ⇒ item 4 collapses exactly as the peel lead did, and the driver's two largest waits are closed for one reason. Ceiling above 2.5 ms ⇒ there is a real deferral lead and C2 gets designed. |
| 2 | **quad counters** | `ready = 0`. Small in absolute terms (it was 0.995 ms/frame at 4.17 waits pre-fan-out) and **mostly `wait-bound`** — it sits inside the episode chain with more chain behind it. Ceiling < 0.15 ms/frame. | Ceiling above 0.4 ms. |
| 3 | **peel checks** | Reproduces the peel census: `ready = 0`, ceiling ≈ **0.376 ms/frame**. This site is the census's own control — **if it does not reproduce, the generalisation is wrong and nothing else in the table should be believed.** | Any ceiling outside roughly 0.30–0.45 ms/frame. |
| 4 | **segment counters** | `ready = 0`, ceiling < 0.1 ms/frame, `wait-bound`. | Ceiling above 0.3 ms. |
| 5 | **descriptor upload** | Tiny: 0.011 ms/frame at 1.00/frame pre-fan-out. Ceiling ≈ 0. Present to keep the gaps honest, not as a lead. | Above 0.1 ms/frame — then the blocking `cuMemcpyHtoD` on the submit path is worth its own look. |
| 6 | **`vkDeviceWaitIdle`** | 2.24/frame. **This is the one I am least sure of.** It sits at the frame boundary next to the 2.453 ms median inter-submit stall, so it could be large and it could be mostly `gap-bound`. No prediction beyond `ready` being unavailable and the blocked time being **> 1 ms/frame**. | Under 0.2 ms/frame — then the frame-boundary stall is not here and the trace's inter-submit gap is somewhere else entirely. |
| 7–8 | **arena rewind, scratch reclaim** | Near zero, possibly zero waits. Both are threshold paths. | Either above 0.3 ms/frame — that would be a finding on its own, since neither appears in any budget in the docs. |

**Overall prediction.** Total ceiling **1.5–3.5 ms/frame** against a total
blocked time around 10 ms, and **`ready = 0` at every stream site**. At the
peel-measured 11% conversion that is worth **0.15–0.4 ms of frame**, spread
over eight sites, which would close the deferral programme as a whole rather
than site by site.

**What would overturn the peel conclusion:** any site with a non-zero `ready`
count. That would be a wait with no device work behind it — pure overhead, and
worth removing outright regardless of every ceiling in the table.

---

## 3. Exact command

Two runs. Not timed: the census adds two clock reads and one `cuStreamQuery`
per wait, roughly 17 waits a frame.

```bash
ICD=/tmp/peel-tree/build-cudavk-peel/src/cudavk/cudavk_devenv_icd.x86_64.json
GFX=~/gfxreconstruct/build
D=/tmp/perf-audit/wait-census; mkdir -p $D

for cap in old:~/headless_streamer_20260814T155742.gfxr \
           cross:~/headless_streamer_1818_20260817T173522.gfxr; do
  n=${cap%%:*}; f=$(eval echo ${cap#*:})
  VK_DRIVER_FILES=$ICD CUDAVK_WAIT_CENSUS=1 CUDAVK_PLAN_STATS=1 \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported "$f" \
    > $D/$n.out 2> $D/$n.err
done

grep -E "WAIT CENSUS|cudavk:   (episode|quad|peel|segment|descriptor|vkDevice|upload-arena|scratch|TOTAL|READ IT)" $D/*.err
grep -E "main-thread waits|render scopes" $D/*.err
```

`CUDAVK_PLAN_STATS=1` is there only for the render-scope count the per-scope
column divides by. `CUDAVK_ABUF_FUSE_CHECK`, `CUDAVK_ABUFFER_VERIFY`,
`CUDAVK_ABUFFER_TIMING` and `CUDAVK_DEBUG_WORK` stay unset as always.

Convert `ms/scope` to per frame with the frame count the harness already
computes; the plan-stats line prints total ms per site for a cross-check
against the census's own blocked column, and **those two should agree to within
the census's own overhead — if they do not, read the census as suspect.**

---

## 4. How to decide, in one table

| finding | conclusion |
|---|---|
| every stream site `ready = 0` | the driver is device-paced everywhere it blocks; blocked time is a symptom, not a budget. The general rule from the peel closure applies driver-wide. |
| drain ceiling < 0.3 ms/frame | `PERFORMANCE.md` §6 item 4 closes. The overlap programme is finished, the residual device idle is the host's own issue time, and the remaining problem is **launch rate**, not overlap. |
| drain ceiling > 2.5 ms/frame, mostly `gap-bound` | C2 is worth designing: there is real host work to move across the drain, and the vertex stage is the phase that can move without a second A-buffer. |
| any site with `ready > 0` | that wait has no device work behind it. Remove it outright; it is cheaper than anything else in the table. |

**The launch-rate corollary is worth stating now, because the outcome above is
the likely one.** Three things have paid in this driver and all three are
overlap; the fan-out took the available width (F4 in `driver_facts.md`:
0.15 waves/SM × 8 streams, ≈6.7 concurrent launches fill the machine). **PDL is
already a launch-rate change, and it is the only thing still paying** —
stream 4.10 µs/kernel → graph 2.02 → PDL 1.89, on a driver that issues over a
thousand launches a frame. If C1 finds no idle device behind any wait, the next
iteration should be about launch rate and not about overlap.
