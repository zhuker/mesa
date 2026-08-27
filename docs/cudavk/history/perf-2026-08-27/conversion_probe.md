# The conversion probe — measure dFrame/dHostTime at a wait site

`CUDAVK_WAIT_SPIN_US=<D>` with `CUDAVK_WAIT_SPIN_SITE=episode|peel|...`.
Default `D = 0`, site `episode`. At `D = 0` nothing is called and the driver is
what it was. Worktree `/tmp/peel-tree`, branch `peel-predicate`.

---

## 1. The idea, and the one thing it cannot do

A wait cannot be shortened without a mechanism, but the host's time at that
point can be **lengthened for free**, and the slope is the same derivative.
The probe busy-waits `D` microseconds on the monotonic clock immediately after
the named wait returns — a busy wait, not a sleep, so it does not yield the
core and change scheduling. The spin sits outside the wait's own timing, so it
contaminates neither the blocked column nor the next site's issue burst.

> **slope = dFrame / d(injected host time), both in ms/frame.**
> Near 1: host time there is on the critical path.
> Near 0: the host has slack there, and the site closes the way peel did.

**The direction caveat, stated up front because it decides how to read the
result.** This measures the cost of **adding** host time. The lead needs the
value of **recovering** wait time. They are the same derivative only if the
system is locally linear, and there is a concrete reason to doubt that here: a
drain *empties the stream by definition*, so immediately after one the device
has nothing queued and any host delay lands straight on the critical path —
whereas *removing* wait time does not remove the device work the wait was
waiting for. So:

* **a low slope is decisive** — if adding host time there is free, recovering
  it certainly is;
* **a high slope proves only the necessary condition**, not that the drain is
  worth its 2.066 ms.

The peel site is the control that says which world we are in, because peel's
**recovery**-direction conversion is already measured at about 11% (the
rejected patch added 1.01 ms/frame of blocking for 0.110 ms of frame). If the
peel *add*-slope also comes back near 0.11, the two directions agree and the
drain's slope can be read as its conversion factor. If the peel add-slope comes
back high, the probe has proved the asymmetry instead, and the drain's number
becomes an upper bound rather than an answer. **Either way the session learns
something it cannot currently state.**

## 2. Scale D per site, or the control is unmeasurable

The lever is `D × waits/frame`, not `D`. Old capture: the drain runs 9.88
times a frame, the peel check 1.70. Running both at the same `D` would inject
5.8× less at peel and put its answer in the noise.

| injected ms/frame | `D` at the drain (9.88/frame) | `D` at peel (1.70/frame) |
|---:|---:|---:|
| 0 | 0 | 0 |
| ~0.25 | 25 µs | 150 µs |
| ~0.49 | 50 µs | 300 µs |
| ~0.99 | 100 µs | 600 µs |
| ~1.98 | 200 µs | 1200 µs |

Report the slope against **injected ms/frame**, which is dimensionless and
directly comparable to the 11%. Signal-to-noise is not a problem: at the top
point a slope of 1.0 moves the old capture 13.16 → 15.14 ms and a slope of 0.11
moves it to 13.38, against a session spread of about 0.03 ms.

## 3. PREDICTIONS, before the run

**Drain add-slope: HIGH, 0.6–1.0.** A drain empties the stream, so right after
it the device is idle and waiting for the host to refill it; a delay there is
device-visible immediately. The census supports it: the drain is **gap-bound in
10,819 of 14,932 waits**, so on 72% of drains the host really does have work in
hand.
*Falsified below 0.2* — and that is the outcome that closes the lead outright,
because the generous direction would then be saying nothing is there.

**Peel add-slope: ALSO HIGH, 0.5–1.0 — that is, NOT 0.11.** A peel check is
also a full stream drain, so the same argument applies to it. I therefore
predict the control **disagrees** with peel's measured 11% recovery conversion,
and that the disagreement is the finding: *adding host time and recovering wait
time are not the same number at these sites.*
*Falsified if peel comes back in 0.05–0.20* — in which case the directions
agree, the probe does exactly what it was designed to do, and the drain's slope
is its conversion factor.

**Linearity: the slope is constant across the four points.**
*Falsified if the slope rises with D* — then the probe is pushing the host onto
the critical path by itself and only the smallest D is meaningful. Report the
linear region, and treat the largest point as suspect by default: at
~2 ms/frame injected the probe is a sixth of the frame.

**Hashes identical at every D**, because the probe changes no device work. A
changed hash means a run that died, not a small correctness question.

**Frame time rises monotonically with D.** If it does not, the spin is not
where it is believed to be.

**Combined prediction for what the session concludes.** Drain slope 0.6–1.0 and
peel slope 0.5–1.0 ⇒ the asymmetry is real, the drain's value is bounded above
by `slope × 2.066 ms` and below by `0.11 × 2.066 ≈ 0.23 ms`, and **the band
cannot be narrowed without building a mechanism.** That is a legitimate place
to stop: it is exactly the statement "the drain is the only site left with a
number big enough to be worth a mechanism, and nothing cheaper will price it."

## 4. Exact command

```bash
ICD=/tmp/peel-tree/build-cudavk-peel/src/cudavk/cudavk_devenv_icd.x86_64.json
GFX=~/gfxreconstruct/build
PLUG=~/claude-scratchpad/perf16/fps_plugin.so
OLD=~/headless_streamer_20260814T155742.gfxr
D=/tmp/perf-audit/spin; mkdir -p $D

run() {  # run <name> <site> <us>
  mkdir -p "$D/$1"
  env VK_DRIVER_FILES=$ICD CUDAVK_WAIT_SPIN_SITE=$2 CUDAVK_WAIT_SPIN_US=$3 \
    $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
      --replay-event-plugin-path $PLUG \
      --replay-event-plugin-params "$D/$1/submits.txt" \
      "$OLD" > "$D/$1/stdout" 2> "$D/$1/stderr"
  sha256sum "$D/$1/stdout" | cut -d' ' -f1 > "$D/$1/sha"
  wc -l < "$D/$1/submits.txt" > "$D/$1/submits.count"
}

# Alternate the zero point through the sweep so drift is shared, as always.
for i in 1 2; do
  run "drain-000-$i" episode 0
  run "drain-025-$i" episode 25
  run "drain-050-$i" episode 50
  run "drain-100-$i" episode 100
  run "drain-200-$i" episode 200
done
for i in 1 2; do
  run "peel-0000-$i" peel 0
  run "peel-0150-$i" peel 150
  run "peel-0300-$i" peel 300
  run "peel-0600-$i" peel 600
  run "peel-1200-$i" peel 1200
done
```

Frame time is the usual `median(diff(submit_ts[::2])[50:])`. Gate on the
sha256 and on 3,022 submits before believing any point. Old capture only:
Crossroads has no peel site, and its drain can be added later if the drain
answer is interesting.

Do **not** set `CUDAVK_WAIT_CENSUS=1` for these runs. It is legal — the spin is
excluded from both census columns by construction — but there is no reason to
carry its clock reads into a timed measurement.

## 5. A caveat on the 2.066 ms the slope will be multiplied by

`TOTAL blocked 10.42 + issued 9.91 = 20.33 ms/frame` against a **median** frame
of 13.16 ms. Those cannot both be per-frame wall time on one thread, and the
explanation is that the census columns are **sums divided by a frame count —
means — while the frame time is a median**, and this replay's mean frame is
substantially longer than its median. The *ratios* the census reports (ceiling
as a share of blocked, gap-bound versus wait-bound) are ratios of two sums and
are unaffected. But **"2.066 ms/frame" is a mean-based figure and slightly
overstates its share of a median frame**, so the drain's headline should be
carried with that caveat until someone re-derives it against the mean frame.

The slope probe is immune to all of this: it measures frame time directly,
by the same median convention as every other result in the project.
