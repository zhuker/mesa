# Full-rearchitecture ideas — 2026-08-28

Status: **research, not acted on.** Nothing here is measured. Every claim that
carries a number cites where the number was measured; everything else is an
estimate and is labelled as one. This note exists to think past the incremental
frontier (`PERFORMANCE.md` §6), not to replace it — the live incremental item
(run-ahead past the episode drain, ceiling ≈2.07 ms/frame on old) is the
baseline any idea here must beat *after* subtracting what it costs to build.

## The disease, restated in one paragraph

The driver is host-bound and the host is not slow — it is *waiting to be told
what to decide*. 9.73 ms of a 13.21 ms frame is host blocked across 16.76
waits, device idle (3.70 ms) tracks host issue time (3.48 ms), and every single
wait is device-paced (`ready = 0` at all 28,852 census waits, §5.2b). Every
wait is a **read-back-and-decide**: the episode drain copies counters to the
host and the host branches on overflow before it may launch the fragment
shader (`cp_pass_can_retry`). Meanwhile the device is nearly empty — GR Active
81%, SM issue 3%, DRAM 1% — and has been since iteration 1. Every millisecond
won so far was won by issuing less or overlapping more, never by filling the
machine (§6 item 0).

So the axis of a real rearchitecture is not "make kernels faster" and not
"give the GPU more work". It is: **remove the host from the decision loop**.
Three ways to do that, in increasing order of surgery:

1. make the decision unnecessary (the answer is always "it fits"),
2. make the decision on the device (the host never hears the question),
3. make the decision late and asynchronous (the host hears it a frame later,
   and only acts on the rare miss).

Each idea below is one of those three, plus one enabling idea (address
stability) that several of them share, and one host-threading idea that only
pays once the decisions are gone.

---

## Idea A — Stable device address space: the keystone

**Class:** enabler. **Surgery:** medium. **Direct win:** ~0; it unlocks B, C, E.

`DEAD_ENDS.md` §1 refuted CUDA graphs at three units, and its mechanism
statement is the most important sentence for any rearchitecture: *"Keys rotate
because scratch pointers rotate, and the command plan is re-recorded every
submit, so a cached graph is never asked for twice inside the generation that
could use it."* Its retry-if is explicit: **stable command-owned execution
storage with fixed device addresses**.

Today `cp_scratch_alloc_device` reallocates the arena on growth and
`cp_scratch_reset` frees overflow blocks at every flush (§11), so no episode's
device addresses survive to the next frame. Replace the per-flush
rotate-and-free with a **slot residency model**: a ring of N per-episode-slot
arenas with fixed base addresses, sized by high-water mark, resized only on
growth events (rare after warm-up), never freed on the flush path. Episode k
of frame f runs in slot (k mod N); its pointers are the same every frame once
warm.

What this buys, none of it speculative in mechanism:

- **Recipe recurrence across frames becomes real.** Iteration 20 found 4,532
  submits with 4,532 unique plan generations and *zero* repeated recipes — but
  these captures replay the *same frame stream*; the uniqueness is address
  rotation, not workload novelty. With fixed addresses, episode k's kernel
  argument bytes are identical frame over frame, which is exactly the
  admission that iterations 12/19/20 measured as absent.
- Graphs (with per-frame parameter update, not re-instantiation), conditional
  nodes (Idea B), and cross-frame sizing (Idea C) all become addressable.

**Risks / costs.** Footprint: N slots at high-water instead of one rotating
arena — bounded by measured high-water, *not* by worst-case bounds, so §11's
8.6 GB failure mode does not apply, but the high-water census must be run
first. Fragmentation across differently-shaped episodes. The upload arena and
descriptor arenas need the same treatment or the recipes still rotate.

**Cheapest falsifying probe (no production code).** Extend the existing plan
instrumentation: hash each episode's (kernel identity, argument bytes with
scratch pointers *masked to slot-relative offsets*) and measure cross-frame
recurrence at fixed slot assignment. If episode recipes do not recur ≥90%
frame-over-frame with addresses factored out, everything downstream of A dies
here for the price of one census. Also record per-slot high-water to size the
footprint.

---

## Idea B — Move the branch onto the device: conditional-node / predicated episode tails

**Class:** decision-on-device. **Surgery:** large. **Ceiling:** the drain +
segment-counter sites, ≈2.3 ms/frame on old (site ceilings §5.2b) — more than
the run-ahead's 0.387, because it removes waits instead of shading them.

The drain exists because `cp_pass_can_retry` forbids launching the fragment
shader before the host reads the overflow flags. Invert it: **pre-issue the
whole decision tree and let the device take the branch.** Two mechanisms, in
order of preference:

- **CUDA 12.4+ conditional graph nodes** (`cuGraphConditionalHandle`): the
  episode tail becomes a graph whose IF-node reads the overflow word the scan
  already writes; the fits-arm (shade chain) and the repair-arm are both in
  the graph; the host launches once and never reads the counters
  synchronously. This is only reachable *through Idea A* — without stable
  addresses it is iteration 12/19/20 again, and `DEAD_ENDS.md` §1 already
  paid for that lesson three times.
- **Predication without graphs** (fallback, testable sooner): issue the shade
  chain unconditionally with every kernel early-exiting on the device-side
  overflow flag, and issue a repair chain predicated the other way. Rule 6
  says idle blocks are free (<0.2 ns per idle block); the cost is the extra
  *launches*, at 0.6–1.0 µs hidden / ~2.0 µs exposed each (rule 15). Budget:
  the repair arm must stay under ~10 launches per episode or the predication
  tax eats the win.

**Dead-end collisions, answered.** §1 (graphs): differs by satisfying its
retry-if via Idea A, and by using conditional nodes, which none of iterations
12/19/20 had. §2 (persistent schedulers): no dynamic per-item claims anywhere
in this — the branch is per-episode, static shape, and rule 3 explicitly
leaves static per-episode chaining open (iteration 28 item 4 paid +0.2208 ms
with that shape). §22 (merge rule): the episode tail chain is *serial*, so the
launch-boundary credit is collectible where this touches it.

**What it does not remove.** The memory-growth path: a true overflow still
needs a host-side reallocation eventually. That is what makes B want C — with
historical sizing the overflow arm almost never runs, and the growth path can
stay host-side and slow.

**Cheapest falsifying probe.** Two numbers, both one-flag censuses:
(1) the per-frame overflow/retry *rate* at the drain — how often does the
decision actually go the expensive way? If retries are >a few percent of
episodes, predication overshoot gets expensive and C must come first.
(2) a microbenchmark of conditional-node graph launch + parameter update per
frame at the measured episode count (~10/frame): must land under the 0.606 ms
mean drain wait it replaces, times 9.88 waits.

---

## Idea C — Temporal sizing: frame N−1's actuals size frame N

**Class:** decision-unnecessary (almost). **Surgery:** medium. **Ceiling:**
shares B's; standalone it converts drain waits into rare repair events.

§11's verdict was precise: worst-case bounds lose because the expensive
quantity is the **ratio bound/actual** — 3.75× for `4×total`, seven orders of
magnitude for `blocks×tris` — and its retry-if asks for *"a scheme that keeps
the over-allocation factor a small constant."* There is a predictor nobody has
tried whose ratio is ≈1.0×: **the same episode, last frame.** These are frame
streams with temporal coherence; the capture replays make that exact.

Mechanism: keyed by Idea A's slot identity, keep last frame's `quads`/`total`
per slot; allocate `last × (1+margin)`; the scan still writes exact counters,
the tail still writes overflow flags, but the host reads them **asynchronously
one frame late** (the completion worker already exists and already retires
things in queue order, §2.2). A miss (overflow flag set) is repaired by
re-running the episode with exact sizing — the current drain path, demoted to
an exception handler.

**Dead-end collisions, answered.** §11 refuted *worst-case* sizing at ratio
10^4 and arena churn; this is ratio ≈1.0–1.3 with no per-flush free (Idea A
removes the churn §11 measured as the mechanism of loss). The P1 probe
(`TODO.md`) already measured `total/quads` at 3.75 capped by geometry — the
*temporal* ratio `quads_f / quads_{f-1}` is the one number this idea rests on
and it has never been measured.

**Correctness rule it must not break:** the no-replay rule — a wrong size must
degrade to the existing drain path, never to device loss.

**Cheapest falsifying probe.** One `fprintf` at the drain (the P1 pattern,
already proven cheap): log (slot, quads, total) per episode, then compute the
distribution of `quads_f / quads_{f-1}` per slot offline. If p99.9 exceeds a
small margin (say 1.5×) on either capture, the repair arm runs too often and
the idea dies for the price of one run. Also measure the miss cost: one full
episode re-run at measured episode kernel time.

---

## Idea D — Guaranteed-fit A-buffer: paged fragment pool

**Class:** decision-unnecessary (fully). **Surgery:** very large — it changes
the data structure five kernels agree on. **Ceiling:** the drain site, plus it
deletes the count-pass questions rather than answering them.

The overflow question exists because the fragment array is a dense, pre-sized
allocation. Replace it with a **paged pool**: fixed-size pages (e.g. 64–256
fragments) from one large pool, per-pixel page chains built by a device-side
bump allocator during fill. Overflow becomes *pool exhaustion* — one flag, one
pool, sized by high-water across the whole frame, over-allocation factor
bounded by page slack (a small constant, satisfying §11's retry-if). No
per-episode sizing, no count-to-host, no drain in the common path at all. The
scan/sort/quad-stream stages re-read pages instead of a dense run; DRAM sits
at 1% (§6 item 0), so the bandwidth for pointer-chasing exists — the risk is
latency, and the A-buffer stages are already latency-bound like everything
else, which is the honest counter-argument.

**Dead-end collisions, answered.** §10 (stage-3 kernel-internal redesign) and
§6 (episode-global tagged stage 3) refuted *rasterizer-internal* reshuffles at
neutral; this changes what the rasterizer *writes to*, not how it walks
coverage. §11: ratio is page-slack-bounded, not bound-based. §21 (count-phase
merge): untouched — count can disappear entirely on the single-pass build,
which is a bigger statement than merging it.

**Cheapest falsifying probe.** Offline, from data the driver already has:
replay with `CUDAVK_DRAIN_PROBE=1`-style logging of per-episode fragment
totals; compute pool high-water per frame and page-slack at several page sizes
against measured per-pixel run-length distributions (the sort stage already
knows them). Then a standalone microbenchmark: sort+composite over paged runs
vs dense runs at the measured run-length distribution. If paged
sort/composite costs >~0.5 ms/frame extra kernel time, it cannot be bought
back by a 2.07 ms host ceiling that B/C can reach with less surgery — kill it.

---

## Idea E — The submit thread: pipeline frames once the decisions are gone

**Class:** host restructuring. **Surgery:** medium. **Ceiling:** the 2.453 ms
median inter-submit stall (§5.2) — explicitly an *open question*, since
`vkDeviceWaitIdle` explains only 0.514 ms of it.

Today `vkQueueSubmit` issues on the application thread and blocks mid-issue at
the drains, so the application cannot even *record* frame N+1 while the driver
waits inside frame N. Move issue to a dedicated driver thread: submit enqueues
the recorded plan and returns; the worker issues and takes whatever waits
remain. The existing completion worker (§2.2) already proves the
retire-in-queue-order half of this design.

Two honesty notes. First, this is the one idea here that does not need B/C/D
to exist — but without them the worker still ping-pongs internally; it only
overlaps the *application's* record time with the driver's blocked time, and
on a GFXR replay the application does almost nothing, so **the captures will
understate what a real app gains**. Second, the census's conversion discipline
applies: the stall must be attributed before it is sized (rule 14 — attribute
a wait site by caller before ranking it). Where the 2.453 ms inter-submit
stall actually lives has never been answered; answer it first.

**Cheapest falsifying probe.** Instrument the inter-submit gap: timestamp
submit-return to next-submit-entry, split into GFXR replayer time vs driver
time (present path, completion waits). If the stall is the replayer's own
frame pacing, this idea is worth ~0 on these captures and must be argued on an
interop/real-app workload instead.

**PROBE ANSWERED, 2026-08-28 — the stall is the replayer, and E is worth ~0
on these captures.** The old capture replayed on the native NVIDIA driver
(`/tmp/rearch-probe/native-old-cpu/`, fps plugin, full 3,022 submits): frame
median **2.879 ms** with the two half-intervals nearly symmetric (1.535 /
1.340), wall 5.08 s against **4.88 s of replayer CPU (96% of one core, sys
2.57 s of it — the 2.4 GB capture read and mapped-memory rewrites)**, GPU
utilisation sampled at **0–1%** throughout. The native driver renders these
frames in ~no GPU time; ~2.9 ms/frame is the replayer decoding the next frame
on the same thread that calls `vkQueueSubmit`. That work exists identically
under cudavk, and since the replayer is single-threaded it can only run
between submits — which is where cudavk's 2.453 ms "unexplained" inter-submit
stall sits. The stall is a harness floor, not driver time. Three consequences:

- **cudavk's device-idle budget shrinks.** Of the 3.70 ms device idle, ~2.45
  is replayer time no driver change can touch; the honest scheduling headroom
  is ~1.25 ms/frame plus whatever device work can be made to *ride* the
  replayer window (today the host has drained everything before submit
  returns, so the window is wasted overlap space — one more argument for B/C:
  decisions removed means device work spills into that window for free).
- **A dedicated submit thread cannot win on these captures** — the app thread
  *is* the bottleneck during the stall. E must be argued on a real-app or
  interop workload, exactly as the caveat above predicted.
- **Cross-arm comparisons stay valid but percentages are understated.** The
  ~2.9 ms floor is additive and driver-independent (same replayer, same
  capture), so A/B deltas are unaffected; but "frame time" overstates the
  driver's own cost by ~2.5–2.9 ms on every capture number in this project,
  and a driver improvement's percentage should be judged against the
  replayer-floor-subtracted frame.

---

## Idea F — Device-side chaining of serial episode tails (CDP2 fire-and-forget)

**Class:** decision-on-device, minor. **Surgery:** small-medium. **Ceiling:**
launch-issue share of host time only — small, order 0.1–0.3 ms (estimate).

The episode tail is a serial chain (scan → fill → sort → quad → shade →
composite). Let each kernel's last block launch its successor with CDP2
fire-and-forget, so the host issues one launch per chain instead of ~6, and
the *host-side* issue burst shrinks. Rule 3 permits the shape (static
per-episode chaining paid, iteration 28 item 4); §2 does not apply (no dynamic
claims). But rule 15 warns the removed launches are hidden 74.9% of the time
(worth 0.6–1.0 µs, not 2 µs), and §6 item 0 warns host issue time is 3.48 ms
*total* across >1,000 launches — the launch-API share is bounded well under
that. This is a supporting idea, not a headline: it matters mainly as B's
non-graph fallback plumbing.

**Probe:** count host-exposed (non-hidden) launches in the tail chains from
the existing trace; multiply by 0.78–0.81 µs (rule 15's remove price). If the
product is <0.15 ms/frame, park it.

---

## Anti-ideas — attractive, and already closed

Listed so this note cannot be read as reopening them.

- **"The GPU is 97% issue-idle, feed it more work."** Closed by §6 item 0:
  utilisation has not moved across a −47.9% frame; width moves occupancy, not
  issue. Any idea argued from emptiness alone is unsupported.
- **Megakernel / persistent scheduler with dynamic claims.** §2: +140.15 ms
  against a 123.7 ms ceiling. Only static-assignment shapes are open.
- **Worst-case sizing to skip the drain.** §11: device loss on old, −6.37% on
  Crossroads; the ratio is the cost. C is the version with ratio ≈1.
- **Caching graphs over today's rotating addresses.** §1, three iterations.
  Only reachable through Idea A.
- **Sort-middle tiling v2.** §24: best case smaller than the fan-out it must
  surrender, and the hot tile is depth complexity, not coverage.
- **Merging parallel launches for the launch credit.** §§20–22: the credit
  only exists where launches were serial.

## Suggested order of attack

Probes only, in one session, before any production code — the same
measure-first discipline that made §1's three graph rejections "the cheapest
way three iterations have ever been spent here":

1. **C's probe** (temporal ratio per slot) — one fprintf, one run per capture.
2. **A's probe** (address-masked recipe recurrence + slot high-water) — one
   census.
3. **B's probe #1** (retry rate at the drain) — rides in the same run as 1.
4. **E's probe** (attribute the inter-submit stall) — **DONE 2026-08-28**:
   the stall is the GFXR replayer's own CPU time (see Idea E), which closes
   §5.2b's open question and parks E on these captures.
5. Only then decide between B-via-A (graphs+conditional nodes) and B-fallback
   (predicated chains + F), with C in either case; D stays parked unless B/C
   both die.

The composite bet, stated once: **A + C + B turn 9.88 synchronous drains per
frame into zero in the common case, with the drain demoted to a repair
handler.** The evidence-backed ceiling for that family is the §5.2b drain +
segment-counter ceilings (≈2.3 ms/frame on old, ≈0.9 on Crossroads) plus
whatever share of the unexplained inter-submit stall E recovers. Every number
above that is an estimate, and rule 1 applies: measure the unit before
building the mechanism.
