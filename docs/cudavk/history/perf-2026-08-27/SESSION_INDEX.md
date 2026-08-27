# Session index — 2026-08-26 GPU measurement session

Every number here was measured on the reference machine with exclusive GPU
access, gated on stdout hash and submit count (3,022 old / 2,994 Crossroads).
Repo changes: **`docs/cudavk/PERFORMANCE.md` and `docs/cudavk/TODO.md` only.**
All probe instrumentation was reverted; the diffs are kept here.

| file | what it holds |
|---|---|
| `reprofile_baseline.md` | STEP A: the shipping default re-measured at HEAD, 13.1626 / 5.8230, and the fan-out attribution |
| `reprofile_stats.md` | STEP B: the wait table at HEAD, the 2.65 ms attributed to one site, launch/op counts, busy % |
| `probes_p1_p2.md` | P1 `total/quads` and P2 inter-drain burst, with the factor-of-four correction |
| `bound_probe_p3.md` | P3 clip-rectangle bound: 0 admissions in 8,836 samples; empty drains |
| `p3_p4_episode.md` | episode sizing and the drain-wait split, with the cross-probe agreement |
| `peel_measurement.md` | the rejected peel predication patch, and the retirement of its 11% |
| `wait_census_and_late_jobs.md` | the eight-site census, launch microbenchmark, INLINE_FS, PDL level 4, and the conversion slope |
| `drain_probe_full.diff` | the 569-line P1/P2/P3 instrumentation, reverted from the tree |

## The measured results, shortest form

| thing | result |
|---|---|
| shipping default at HEAD | old **13.1626 ms**, Crossroads **5.8230 ms** (PERFORMANCE.md said 15.7514 / 5.8982) |
| the difference | entirely `20611f5b131`, the opaque fan-out: revert arm gives 15.8932 / 5.8993 |
| where the 2.65 ms came from | **the episode drain only**, mean wait 0.874 → 0.606 ms at an unchanged 9.88 waits/frame |
| PDL, level 3 vs level 0 | **+0.4342 ms (+3.29%) old, +0.1297 (+2.23%) Crossroads**, decisive, p = 0.0011 / 0.0143, sweep −1.5% with no sample regressed |
| peel predication patch | **rejected**, −0.11 ms on old |
| `CUDAVK_INLINE_FS` at HEAD | **rejected**, −1.60% on old |
| clip-rectangle bound | **closed**, 0 admissions in 8,836 samples |
| peel site | **closed** by three measurements: ceiling 0.301 ms/frame, device busy at 0 of 2,577 checks, injected host time free (slope −0.026) |
| every host block | **`ready = 0` at all 28,852 waits** — no wait in this driver is pure overhead |
| the episode drain | **open, 2.066 ms/frame ceiling, conversion measured at +1.02** — the largest item left |
| the 11% conversion | **retired** — it was the peel patch's mechanism, not its blocking |
