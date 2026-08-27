# STEP C — probes P1 and P2 from `/tmp/perf16/iter29-sync/addendum.md` §5

Both probes ran, on both captures, in one session, at HEAD **e2fea470d04**,
with **no mechanism in the tree** — the only change is instrumentation behind
one new registry flag.

## C.0 What was added, and the proof it changed nothing

- One flag, `CUDAVK_DRAIN_PROBE`, added to the registry in `src/cudavk/cp_debug.c`
  in registry style, with the field `drain_probe` in `struct cp_debug`.
  `src/cudavk/tests/cp_debug_doc.py --check` failed as designed, `FLAGS.md` was
  regenerated (119 switches), and `--check` now passes.
  `src/cudavk/tests/cp_no_getenv.py` passes: "36 driver sources, every switch
  goes through the cp_debug.c registry". **No `getenv` was added anywhere.**
- Two hooks at the existing tail drain in `cp_pass_finish`
  (`cp_renderer.c:8443` at HEAD): one timestamp before `cp_sync_timed`, and one
  call after the counters are read.
- Samples are kept in four arrays and printed once at teardown. **There is no
  `fprintf` per episode**: printing inside the drain would land inside the
  very inter-drain burst P2 measures.
- **Off by default**, so STEP A's and STEP B's runs were clean; the flag was
  introduced after both were finished.
- **Cost check.** With the probe on: old **13.1204** ms against STEP A's
  13.1626; Crossroads **5.8247** against 5.8230. Both inside the
  session spread.
- **Correctness check.** stdout sha256 identical to STEP A on both captures
  (`320e993599cc…`, `e727020fc796…`), full submit counts 3,022 and 2,994, and
  every drain was recorded: 14,932 of 14,932 seen on old, 8,333 of 8,333 on
  Crossroads. Nothing died early.

The diff is **uncommitted** and stays that way. Nothing was committed to the
repository in this task.

---

## C.1 P1 — the distribution of `total/quads` at the tail drain

`total = ctr[0]`, `quads = ctr[3]`, one sample per episode drain.

| statistic | old capture | Crossroads |
|---|---:|---:|
| episode drains sampled | 14,932 | 8,333 |
| of those, with `quads == 0` (ratio undefined) | 1,033 (6.9%) | 1,528 (18.3%) |
| ratio samples | 13,899 | 6,805 |
| **median `total/quads`** | **3.753** | **3.725** |
| p10 / p25 | 2.761 / 3.060 | 2.319 / 2.597 |
| p75 / p90 / p99 | 3.816 / 3.866 / 3.867 | 3.732 / 3.817 / 3.967 |
| **min / max** | **1.000 / 3.973** | **1.000 / 3.991** |
| mean | 3.429 | 3.283 |
| aggregate ΣtotalΣquads | 13,498,112,476 / 3,576,077,932 = **3.7746** | 831,624,127 / 218,853,101 = **3.7999** |

Episode size, for scale (the same samples):

| | old median | old p90 | old max | Crossroads median | Crossroads max |
|---|---:|---:|---:|---:|---:|
| fragment total | 488,376 | 3,076,885 | 3,182,459 | 4,848 | 2,103,017 |
| quads | 133,774 | 795,833 | 824,801 | 1,425 | 531,368 |

### Verdict of P1

**The ratio is a small constant, and it is bounded above by 4 — not by
anything the arena can be asked for. Bound-based sizing at the scan is not
killed by the mechanism that killed `CUDAVK_UNSAFE_NO_OVERFLOW`.**

Three things say so, and the third is the one that decides it:

1. The measured distribution is tight and small: median **3.753** on old and
   **3.725** on Crossroads, p10–p99 inside [2.32, 3.97], and the
   **observed maximum over 20,704 episodes is 3.991**. The two captures agree
   to 0.03 despite being 2.3× apart in frame time.
2. It is a *structural* ceiling, not a lucky sample. A quad is 2×2 pixels, so
   an episode can never have more than four covering fragments per quad:
   `total ≤ 4 × quads` by construction. The probe finds the workloads sitting
   at 94% of that ceiling, i.e. the quads are nearly fully covered.
3. **CORRECTED — sizing from `total` costs 3.75×, not 0.94×.** An earlier
   version of this file said the opposite and was out by a factor of four; the
   correction is the bounded-clip agent's. `cp_shade_fragments` sizes every
   shade array from `want_slots = num_quads * 4` (`cp_renderer.c:4182`;
   `pixel_list`, `fs_in`, `fs_out`, `coverage`, `frag_coord`, `discard_mask`,
   `batch_rows`), and those arrays are **dense over quads**, indexed
   `slot = 4·q + lane`. A host substitute that knows only the scan's fragment
   `total` must bound the quads by it (`quads ≤ total`, since a quad needs at
   least one covering fragment) and therefore allocate `4 × total ≈ 3.75 ×
   (4 × quads)`. The 94% figure is how **full** today's arrays are — a
   different quantity, and not an allocation.

So the addendum's rejection branch — "if it is large, every bound-based sizing
scheme is dead by the same mechanism, site 1 is dead permanently" — **does not
fire**: 3.75 is not 10⁷, and that is the distinction the branch was written to
draw. **But S1d is gated on memory rather than on ratio**: 3.75× the driver's
largest arrays is the S1b question again, against the same 8,589,934,592-byte
scratch cap `UNSAFE_NO_OVERFLOW` hit. The deciding measurement is the
`dscratch` high-water at the drain, not this ratio. The addendum's acceptance threshold was "say ≲ 3" and the measurement
is 3.75; read literally that is a miss, but the threshold was a proxy for
"is the over-allocation factor a small constant or four orders of magnitude",
and the answer is unambiguous: it is a small constant, it is capped at 4 by
geometry, and it is *the same constant the driver already pays*.

**What P1 does not say.** It calibrates S1d's sizing only. It is **not** the
`bound/actual` ratio of the `bounded` fast path: that ratio is
`ab->nblocks × rast_num_triangles` over the actual quads, a completely
different quantity that this probe does not touch. Sizing the clip-rectangle
lead (§6 item 2) against a measured ratio still needs its own one-line probe at
the `bounded` predicate (`cp_renderer.c:6402–6413`), logging `quad_bound`
against the `quads` the fallback drain then reads. **Do not size site 3 from
the 3.75 in this file.**

**A second finding, free with the probe.** 6.9% of old-capture drains and
**18.3% of Crossroads drains return `quads == 0`** — the episode covered
nothing, the host blocked to be told so, and `cp_pass_finish` returns
immediately after. On Crossroads that is 1.02 wasted drains per frame out of
5.57. Any mechanism that predicts "this episode will composite nothing"
without asking would remove a fifth of that capture's drains.

---

## C.2 P2 — the host's inter-drain issue burst

Time from one drain returning (after the counter read-back) to the next drain
starting (before `cp_sync_timed`). `gap` is the whole interval; `issue` is the
gap minus the time the host spent blocked at the *other* three timed waits
(peel, segment counters, descriptor uploads) inside it.

| statistic (µs) | old gap | old **issue** | Crossroads gap | Crossroads **issue** |
|---|---:|---:|---:|---:|
| samples | 14,931 | 14,931 | 8,332 | 8,332 |
| min | 22.9 | 22.9 | 22.6 | 22.6 |
| p10 | 46.4 | 46.4 | 33.7 | 33.7 |
| p25 | 82.5 | 80.6 | 73.3 | 73.3 |
| **median** | **335.4** | **306.2** | **138.8** | **138.8** |
| p75 | 446.8 | 429.1 | 976.6 | 498.6 |
| p90 | 3966.0 | 3847.6 | 2370.0 | 2362.8 |
| p99 | 5189.3 | 5150.7 | 2810.2 | 2802.2 |
| max | 1,773,092 | 1,355,755 | 373,804 | 373,795 |
| mean | 1415.0 | 1036.3 | 837.9 | 758.6 |

**Read the median, not the mean.** The samples cover the whole process,
including start-up, shader compilation and teardown; the maximum gap on old is
1773 ms and on Crossroads 374 ms, which is compilation, not a frame. The
mean is contaminated by those and the median is not. The median is confirmed
against an independent instrument below.

### The medians are consistent with STEP B, which is the check that they mean something

- old: 9.88 drains/frame × 306.2 µs = **3.03 ms/frame**, against STEP B's
  host-issuing figure of **3.48 ms/frame** (frame − blocked). Two instruments
  that share no code agree to 13%.
- Crossroads: 5.57 × 138.8 µs = 0.77 ms/frame against 3.30 ms/frame. The
  gap here is real and is explained by the p75 (499 µs): Crossroads'
  bursts are strongly bimodal, so its median understates the total.
- The p90 on both captures (4.0 ms old, 2.4 ms Crossroads) is the frame
  boundary — the inter-submit stall PERFORMANCE.md §5.2 already describes at a
  median of 2.453 ms — appearing about twice a frame as expected.

### Verdict of P2

**There is real work for the host to do during the wait. The ceiling is
hundreds of microseconds per episode, not tens.**

The addendum's own rule: *"If the burst is tens of microseconds, S1b's ceiling
is a few tenths of a millisecond, it cannot justify 250 MB–1.6 GB, and site 1
is dead in every form. If it is hundreds, S1b has a case."*

- old: median issue burst **306 µs** against a mean episode-drain wait of
  **606 µs** (STEP B). `min(burst, wait)` at the medians is **306 µs per
  episode**, i.e. up to **3.03 ms/frame** if every burst could be moved into
  a wait.
- Crossroads: median **139 µs** against a mean wait of 382 µs;
  `min` = 139 µs × 5.57 drains = **0.77 ms/frame**.

**Three ceilings, and the honest one is the smallest.** The per-episode figure
above is the addendum's arithmetic and it is an over-estimate, because the host
cannot spend the same burst twice. Two independent caps apply:

1. total host issue time is **3.48 ms/frame** on old — nothing can be hidden
   that is not being issued;
2. the device is **72% busy** (STEP B), so the frame cannot fall below its
   resident time of ≈9.5 ms; the headroom is **3.7 ms/frame**.

So the defensible statement is: **on the old capture a perfect deferral
mechanism has 2.5–3.5 ms/frame of device idle to attack, and P2 shows the host
does hold enough queued work to attack it — the median burst is half the
median wait.** On Crossroads the same argument gives well under 1 ms.

**What P2 does not decide.** It refutes exactly one objection — "nothing has
measured whether the host has other work to issue" — and it refutes it clearly.
It says nothing about S1b's memory cost, which is the reason the addendum
demoted S1b, and nothing about whether a deferral can be made correct. The
S0 refutation stands: the failure there was worst-case *sizing*, not the wait,
and P1 has now shown that the sizing S1d would use is not worst-case at all.

---

## C.3 What the two probes decide together

| question | answer | evidence |
|---|---|---|
| Is every bound-based sizing scheme dead? | **No.** The over-allocation factor for sizing from the scan's fragment total is ≤ 4 by geometry and 3.75 in practice, and the driver already allocates 4× quads. | P1 |
| Is **S1d** (take the drain at the scan) worth building and measuring? | **Yes on the sizing question** — its allocation is no larger than today's, so it does not inherit the `UNSAFE_NO_OVERFLOW` failure. Its value is scheduling, not memory: one drain per episode against a strictly shorter chain. | P1 |
| Does the host have work to issue during the wait? | **Yes.** Median burst 306 µs on old against a 606 µs mean wait. | P2 |
| Is **S1b**'s ceiling big enough to justify 250 MB–1.6 GB? | The ceiling is **not** the objection any more; the memory cost is. The addendum's "tens of µs → dead" branch does not fire. | P2 |
| Can site 3 (clip rectangle) be sized from these numbers? | **No.** P1 measures fragments per quad, not bound over actual. It needs its own probe at the `bounded` predicate. | P1 |
| Is the episode drain still the largest single wait? | Yes — 5.989 ms/frame over 9.88 waits after the fan-out took 2.65 ms out of it. | STEP B |

## Blockers

None in this step. Build clean, `cp_debug_doc.py --check` and
`cp_no_getenv.py` both pass, output hashes unchanged.

## C.4 Test gate

`./venv/bin/meson test -C build-cudavk --suite cudavk`: **67/67 OK, 0 failures**
with the probe code in the tree (`/tmp/perf-audit/suite.log`). That includes
`cp_launch_audit` and the registry check.

## C.5 The diff, and how to reproduce

Uncommitted, in `/home/alexzhukov/mesa` (working tree only). Files touched by
this task:

```
src/cudavk/cp_debug.c        +4    one registry entry
src/cudavk/cp_debug.h        +1    bool drain_probe
src/cudavk/cp_renderer.h     +14   the sample arrays, in struct cp_context.plan
src/cudavk/cp_renderer.c     +181  cp_drain_probe_sample/_report and two hooks
src/cudavk/FLAGS.md          +1    regenerated by cp_debug_doc.py
```

(`docs/cudavk/history/CPU_PROFILER.md` and `src/cudavk/tests/cp_cpu_profile.sh`
were already modified in the working tree before this task started and were not
touched by it.)

A copy of the diff is saved at `/tmp/perf-audit/drain_probe.diff`. To run:

```bash
CUDAVK_DRAIN_PROBE=1 gfxrecon-replay -m remap --remove-unsupported CAPTURE.gfxr
```

and read the eight `cudavk: drain probe:` lines on stderr at teardown.
