# PDL landing gates — results

Worktree `/tmp/pdl-tree`, branch `pdl-prototype`, tip
**`b9766f720a4cccc320498be77e2db89d185f90bf`** ("cudavk: offer stage 1 as a PDL
secondary, at level 4, as a diagnostic"). Build dir
`/tmp/pdl-tree/build-cudavk-pdl`. Reference machine, exclusive GPU.

## 0. Verdict: **LAND**

Every gate in `pdl_landing.md` §9, plus the two stale measurements, has now been
run on the final tip. No gate failed.

| # | gate | result |
|---|---|---|
| 1 | clean build of the tip | **PASS** — 846/846, no new warning |
| 2 | `meson test --suite cudavk`, default level 3 | **PASS** — 67/67 |
| 3 | `meson test --suite cudavk`, `CUDAVK_NO_PDL=1` | **PASS** — 67/67 |
| 4 | `CUDAVK_ABUF_FUSE_CHECK=1` at level 3, both captures | **PASS** — 0 differing elements of 37,124 scans and 37,124 quad builds |
| 5 | byte identity of the revert against a clean HEAD build | **PASS** — same stdout sha256 per capture, 3,022 / 2,994 submits |
| 6 | decisive headline re-measured on the final binary | **PASS** — +0.3815 ms (+2.90%) old, +0.1522 ms (+2.60%) Crossroads, p = 0.0011 / 0.0143 |
| 7 | 600-frame sample sweep at the final tip | **PASS** — 15.70 → 15.44 ms (−1.7%), no sample regressed, no new correctness exception |

Three timing sessions were **discarded before use** because a foreign GPU tenant
shared the card; §8 is the whole account. The discarded sessions are S1–S3 of
step 6 (directories `decisive`, `decisive2`, `decisive3`) and one aborted sweep
whose half-written iteration directories were deleted. Nothing from them is
quoted anywhere in this file.

Two figures in `pdl_landing.md` are superseded by §6 and must be restated there
before it is read as the record: §3.2's +0.4342 / +0.1297 and §11's repetition
of them.

## 1. Clean build of the tip — PASS

`ninja -C build-cudavk-pdl clean && ninja -C build-cudavk-pdl`, 846/846 targets,
log `/tmp/perf-audit/gate/build.log`. Exactly two compiler warnings in the whole
tree and **neither is in `src/cudavk`**:

* `src/vulkan/runtime/vk_acceleration_structure.c:467` — ignored `vasprintf` return
* `src/gallium/auxiliary/gallivm/lp_bld_debug.cpp:382` — ignored `asprintf` return

Both are pre-existing upstream warnings on paths this branch does not touch. No
new warning. ICD at `build-cudavk-pdl/src/cudavk/cudavk_devenv_icd.x86_64.json`.

## 2. `meson test --suite cudavk` at the default (level 3) — PASS

**67 Ok, 0 Fail.** Log `/tmp/perf-audit/gate/test_default.log`, full meson log
`/tmp/perf-audit/gate/testlog_default.txt`. Includes `cp_launch_audit`, the
`cpvk_*` draw tests and the texture-cache gates.

## 3. `meson test --suite cudavk` with `CUDAVK_NO_PDL=1` — PASS

**67 Ok, 0 Fail.** Log `/tmp/perf-audit/gate/test_nopdl.log`, full meson log
`/tmp/perf-audit/gate/testlog_nopdl.txt`. The flag really reached the test
processes: meson records `Inherited environment: CUDAVK_NO_PDL=1` at the head of
the log and no test overrides it in its own env block. The revert path is not
broken.

## 4. Equivalence gate `CUDAVK_ABUF_FUSE_CHECK=1` at `CUDAVK_PDL=3` — PASS

Both captures replayed to completion (rc=0). Device-side element-by-element
comparison of the fused chain against the classic one:

| capture | scans | quad builds | differing elements | differing totals | never written | coverage violations |
|---|---:|---:|---:|---:|---:|---:|
| old | 23,814 | 23,814 | **0** | **0** | **0** | **0** |
| Crossroads | 13,310 | 13,310 | **0** | **0** | **0** | **0** |

No `abuf fusion check FAILED` line in either run; no other error on stderr.
Artefacts in `/tmp/perf-audit/gate/fusecheck/`. This run synchronises, so no
frame time is quoted from it.

## 5. Byte identity of the revert — PASS

`CUDAVK_NO_PDL=1` on the new build (tip `b9766f720a4`) against a plain replay on
the clean HEAD build (`e2fea470d04`, `/home/alexzhukov/mesa/build-cudavk`). Same
replay command line in both arms (fps plugin attached in both, so the submit
count is a completeness gate too).

| arm | rc | submits | stdout sha256 |
|---|---:|---:|---|
| new + `CUDAVK_NO_PDL=1`, old | 0 | 3,022 | `320e993599cc7211…` |
| HEAD build, old | 0 | 3,022 | `320e993599cc7211…` |
| new + `CUDAVK_NO_PDL=1`, Crossroads | 0 | 2,994 | `e727020fc7967501…` |
| HEAD build, Crossroads | 0 | 2,994 | `e727020fc7967501…` |

Identical hash per capture, and `diff` of the two stdouts is empty on both.
Submit counts are the expected 3,022 / 2,994, so neither run died early. With
the `seg_cursor` reorder now gated at level >= 1, **the revert really reverts**.

## 6. Headline re-measured on the final binary — WIN CONFIRMED

**Read §8 first if you care where the numbers come from: a foreign GPU tenant
was on the card between 11:45 and 12:14 and three earlier sessions were thrown
away because of it.** The table below is the clean session, S4, run from 12:16:51
to 12:24:37 with a 1 Hz process-name sampler that recorded **no process on the
card except this measurement's own `gfxrecon-replay`** for all 467 samples.

Decisive alternating runs, `CUDAVK_PDL=3` candidate against `CUDAVK_NO_PDL=1`
control, one binary (`b9766f720a4`), empty base environment, 6 runs per arm on
old and 4 on Crossroads, strictly alternating, one session.

| capture | control (L0) | candidate (L3) | delta | p, one-sided |
|---|---:|---:|---:|---:|
| old | 13.1641 (IQR 13.1575–13.1896) | 12.7826 (IQR 12.7632–12.8075) | **+0.3815 ms (+2.90%)** | 0.0011 |
| Crossroads | 5.8458 (IQR 5.8381–5.8540) | 5.6936 (IQR 5.6868–5.7007) | **+0.1522 ms (+2.60%)** | 0.0143 |

Gates on the runs themselves, checked before the medians were read:

* every run full length — **3,022 submits on old, 2,994 on Crossroads**, rc=0;
* **one stdout hash per capture across BOTH arms** — `320e9935…` old,
  `e727020f…` Crossroads, the same hashes as §5, so this is a cost result;
* **arms non-overlapping** on both captures: the slowest candidate run is faster
  than the fastest control run;
* robust cross-check (per-frame-index median across an arm's runs, then the
  median paired difference) agrees: **+0.386 ms** old, **+0.163 ms** Crossroads,
  with the candidate faster on **97.0%** of old frames and 77.0% of Crossroads
  frames.

**The control arm lands on the HEAD baseline.** 13.1641 ms against the
separately measured HEAD figure of 13.1626 ms in `reprofile_baseline.md` — a
1.5 µs difference. That is independent confirmation of §5 from the timing side:
with the `seg_cursor` reorder gated at level >= 1, `CUDAVK_NO_PDL=1` is the
pre-PDL driver, not a fourth behaviour.

**Against the pre-gate figures.** The prediction in `pdl_landing.md` §6.2 was
that the gate can only make the control faster, so the old-capture delta should
be the same or up to ~0.04 ms smaller than +0.4342.

| capture | pre-gate (67244f6dc62) | final tip, clean session | change |
|---|---:|---:|---:|
| old | +0.4342 ms (+3.29%) | **+0.3815 ms (+2.90%)** | 0.053 ms smaller |
| Crossroads | +0.1297 ms (+2.23%) | **+0.1522 ms (+2.60%)** | 0.023 ms larger |

Old moved the predicted way and slightly further than predicted; Crossroads
moved the other way by half as much. Both differences are the size of ordinary
session drift and neither is a claim about the reorder. **`pdl_landing.md` §3.2
and §11 should be restated with the two figures above.** The win survives on
both captures and is decisive on both.

## 7. 600-frame sample sweep at the final tip — PASS, no new exception

`cp_iterate.sh` with `MESA=/tmp/pdl-tree`, one build (`b9766f720a4`), level 3
against level 0: `pdl-final-l0` (`CUDAVK_NO_PDL=1`, `BENCH_ONLY=1`) then
`pdl-final-l3` (`CUDAVK_PDL=3`, full pass with the correctness gate) compared
against it. 1 Hz sampler attached for the whole sweep: 122 samples, nothing on
the card but the sweep's own sample binaries.

**TOTAL 15.70 → 15.44 ms (−1.7%). Not one sample regressed.**

| sample | L0 ms | L3 ms | change |
|---|---:|---:|---:|
| triangle | 0.09 | 0.08 | −11.1% |
| texture | 0.11 | 0.10 | −9.1% |
| negativeviewportheight | 0.14 | 0.13 | −7.1% |
| dynamicuniformbuffer | 0.24 | 0.23 | −4.2% |
| computeshader | 0.30 | 0.29 | −3.3% |
| gltfscenerendering | 3.89 | 3.76 | −3.3% |
| texturecubemap | 0.40 | 0.39 | −2.5% |
| bloom | 1.07 | 1.05 | −1.9% |
| **pbribl** | 0.52 | 0.51 | **−1.9%** |
| multithreading | 2.37 | 2.34 | −1.3% |
| multisampling | 0.98 | 0.97 | −1.0% |
| pushconstants, instancing, particlesystem, texture3d, texturemipmapgen, vulkanscene | — | — | 0.0% |
| **TOTAL** | **15.70** | **15.44** | **−1.7%** |

This reproduces the pre-gate sweep (15.67 → 15.44, −1.5%) sample for sample,
including pbribl improving 1.9% — the sample that had to be accepted as a
bounded cost under the vertex-fetch fusion.

**Correctness gate: the two standing exceptions, and nothing else.**

| sample | worst | at frame | verdict |
|---|---:|---:|---|
| gltfscenerendering | 137,025 | 47 | REGRESSED (budget 27,485) — **standing exception** |
| renderheadless | — | — | missing — **standing exception** |

Every other one of the 16 samples is `ok`. gltfscenerendering reads 137,025
against the 137,026 recorded for the last five iterations — one unit in 137,000,
the same sample at the same frame 47 — and `renderheadless` is missing as it has
been throughout. **No new exception**, which was the blocker to look for.

## 8. The foreign GPU tenant, and why the standing idle check missed it

**This section is a finding, not an excuse.** It changes what a "GPU is idle"
check has to be.

### What happened

Step 6 was first measured in three sessions between 11:45 and 12:12. All three
said the candidate was faster on both captures, but the magnitude would not
settle: the old-capture delta read +0.3772, +0.6235 and +0.4010 ms across the
three, and one Crossroads control session read 6.4196 ms against a standing
5.86. Individual runs carried bursts — a fifth of a run's frames at 1.5–2.3x the
rest of the same run:

* session 1, `cross-ctrl-4`: fifths 5.60 / 6.10 / **10.01 / 10.44 / 10.32** ms;
* session 2, `old-ctrl-3` first fifth **30.78** ms against a 12.6 ms run median;
* session 3, one candidate run at **23.28** ms and one control run at
  **24.31** ms, against ~12.8 and ~13.2 ms for their arm-mates.

`nvidia-smi --query-compute-apps=pid,used_memory` was run and was **empty**
before every one of those sessions, which is exactly what WORKFLOW §4.5 asks
for.

### What was actually on the card

A 1 Hz sampler that also recorded the *process name* caught it:

```
2582025, /tmp/bc/vpxenc_bc4, 996 MiB
  /tmp/bc/vpxenc_bc4 --codec=vp9 --end-usage=cbr --target-bitrate=4000 --rt \
     --passes=1 --lag-in-frames=0 --threads=4 --limit=1000 --cpu-used=7 ...
```

A libvpx VP9 CUDA-encoder A/B batch, owned by the same user and unrelated to
this work, writing to `/tmp/bc`. Its whole window is **11:45 to 12:14**
(`ab2.done` at 12:14) — the exact window of the three discarded sessions.

### Why the standing check could not see it

The batch is not one long process. It is a **rapid series of short ~1 GB CUDA
processes**, one per encode. A single point-in-time `nvidia-smi
--query-compute-apps` before a run lands in a gap between two of them with high
probability and returns empty, and it is never repeated during the ten minutes
that follow. The check is not wrong, it is **the wrong sampling rate**: it
samples once and the tenant is intermittent.

### The replacement check, proposed

Run a sampler for the *duration* of the measurement, not a check before it:

```bash
# /tmp/perf-audit/gate/gpu_watch.sh
while true; do
  echo "$(date +%s) $(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                        --format=csv,noheader | tr '\n' ';')"
  sleep 1
done
```

and have the AB harness record an epoch window per run
(`/tmp/perf-audit/gate/decisive_ab_ts.sh` writes `window` in each run
directory). Three things follow that the old check cannot give:

1. **`process_name`, not just `pid`.** A pid alone does not say whether the
   entry is the measurement's own replay or somebody else's encoder, and the
   process is gone by the time anyone looks it up.
2. **Attribution per run.** With a timestamp window per run and a timestamped
   sampler log, contaminated runs are *named* and discarded individually. A
   whole session does not have to be thrown away — and, more importantly, a
   session is not silently kept.
3. **A positive statement in the record.** The clean session's line is "467
   samples, no process on the card except this measurement's own
   `gfxrecon-replay`". That is a claim a reader can check; "nvidia-smi was empty
   before the run" is not.

This belongs in `docs/cudavk/WORKFLOW.md` §4.5 alongside the existing rule. It
is not edited into the repo here.

### What this does and does not touch

* **It does not touch gates 1–5.** A foreign tenant changes timings. It cannot
  change a test result, a device-side element comparison, or a stdout hash.
  Those gates ran and passed on their own evidence.
* **It does not touch the rest of the 2026-08-26 session.** Every measurement in
  `/tmp/perf-audit` was written by 11:29 — the last one, the seg/idle slope
  sweep, at 11:29 — and the libvpx batch started at 11:45. A sixteen-minute
  margin, verified from the file timestamps. The baselines, the earlier decisive
  PDL runs, the earlier sweep, the wait census and all four slope measurements
  are clean.
* **The three contaminated sessions are discarded, not averaged in.** Their
  directories are kept at `/tmp/perf-audit/gate/decisive`, `decisive2` and
  `decisive3` with this note; the clean session is `decisive4`.
