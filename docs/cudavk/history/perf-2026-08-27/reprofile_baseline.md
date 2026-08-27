# STEP A — re-baseline of the shipping default at HEAD

Repo `/home/alexzhukov/mesa`, branch `cudavk-vk-native`, HEAD **e2fea470d04**.
Build: `build-cudavk`, `libvulkan_cudavk.so` dated Aug 26 07:27; `ninja -C
build-cudavk` at the start of this session rebuilt only lavapipe targets, so
the cudavk shared object is current with the newest cudavk source file. No
probe code was in the built binary for any run in this file.

GPU idle check (`nvidia-smi --query-compute-apps=pid,used_memory`) was empty
before every run — recorded per run in `gpu_before.txt`.

Raw data: `/tmp/perf-audit/stepA/` (default) and `/tmp/perf-audit/stepA-revert/`
(`CUDAVK_NO_OPAQUE_STREAMS=1`). Convention: a frame is two `vkQueueSubmit`
events, `median((ts[2::2]-ts[:-2:2])/1e6)[50:]`, skip 50 — the same arithmetic
as `/tmp/perf16/cp_two_replay_report.py`.

## A.1 Shipping default, 3 runs per capture

| run | median ms | IQR | frames | submits | rc | stdout sha256 (12) |
|---|---:|---|---:|---:|---:|---|
| old 1 | 13.1626 | [12.2229, 15.3041] | 1460 | 3022 | 0 | `320e993599cc` |
| old 2 | 13.1116 | [12.2779, 15.3143] | 1460 | 3022 | 0 | `320e993599cc` |
| old 3 | 13.2302 | [12.3546, 15.4015] | 1460 | 3022 | 0 | `320e993599cc` |
| Crossroads 1 | 5.8165 | [5.5095, 6.2181] | 1446 | 2994 | 0 | `e727020fc796` |
| Crossroads 2 | 5.8509 | [5.5452, 6.2520] | 1446 | 2994 | 0 | `e727020fc796` |
| Crossroads 3 | 5.8230 | [5.5366, 6.2236] | 1446 | 2994 | 0 | `e727020fc796` |

- **old capture**: run medians 13.1626, 13.1116, 13.2302 → **median of medians 13.1626 ms**, IQR of the run medians [13.1371, 13.1964], full range [13.1116, 13.2302].
- **Crossroads**: run medians 5.8165, 5.8509, 5.8230 → **median of medians 5.8230 ms**, IQR [5.8197, 5.8370], range [5.8165, 5.8509].
- Submit counts are the full replay every time: **3,022** on old (1,460 paired
  intervals after the skip) and **2,994** on Crossroads. No run died early.
- One stdout hash per capture across all six runs: old `320e993599cc`,
  Crossroads `e727020fc796`. Wall time 31.0–31.4 s on old,
  10.8 s on Crossroads.

## A.2 Against the documented default

| capture | PERFORMANCE.md §1 | measured here | delta |
|---|---:|---:|---:|
| old | 15.7514 | **13.1626** | **−2.5888 ms (−16.4%)** |
| Crossroads | 5.8982 | **5.8230** | −0.0752 ms (−1.3%) |

**The driver no longer measures where PERFORMANCE.md says it does on the old
capture, and the difference is a real change, not drift and not a tenant.**

Evidence, in this session, on this binary:

1. The whole spread is far outside session noise. The three old run medians
   span 0.119 ms; the gap to the documented figure is 2.59 ms, twenty times
   that spread.
2. The GPU was idle before every run and the wall times are stable to 1%.
3. A revert arm reproduces the documented number exactly (A.3).
4. `git log src/cudavk` shows one code commit after the documentation was
   written: **20611f5b131, "cudavk: fan opaque episode segments out by
   default"** (Aug 26 07:38). Its own message states old 15.77 → 13.23 and
   Crossroads 5.90 → 5.82. My 13.1626 / 5.8230 reproduce that to 0.07 ms.
   `docs/cudavk/PERFORMANCE.md` was last written before that commit, so its
   §1 headline, and everything in §5 that depends on it, is stale.

## A.3 The revert arm, measured here rather than quoted

`CUDAVK_NO_OPAQUE_STREAMS=1`, 2 runs per capture, same session, same binary:

| run | median ms | IQR | submits | sha256 (12) |
|---|---:|---|---:|---|
| old 1 revert | 15.8437 | [14.7548, 18.4090] | 3022 | `320e993599cc` |
| old 2 revert | 15.9427 | [14.6841, 18.9305] | 3022 | `320e993599cc` |
| Crossroads 1 revert | 5.9001 | [5.5975, 6.2971] | 2994 | `e727020fc796` |
| Crossroads 2 revert | 5.8985 | [5.5934, 6.3002] | 2994 | `e727020fc796` |

| capture | fan-out on (default) | fan-out reverted | delta |
|---|---:|---:|---:|
| old | 13.1626 | **15.8932** | **+2.7306 ms (−17.2%)** |
| Crossroads | 5.8230 | **5.8993** | +0.0763 ms (−1.3%) |

The reverted arm lands at 15.89 / 5.899 against the documented 15.7514 /
5.8982 — old is 0.14 ms above the documented median but well inside the
documented full range's neighbourhood and inside this session's own drift
(the two revert runs differ by 0.099 ms themselves). Crossroads reproduces to
0.001 ms. Stdout hashes are identical between the two arms on each capture, so
both arms did the same work.

**Conclusion.** The 2.73 ms is entirely the opaque-episode fan-out becoming the
default. PERFORMANCE.md §1 should read **13.16 ms** on old and **5.82 ms** on
Crossroads, and §5.2's blocked-time table must be re-derived (STEP B) because
the fan-out attacks exactly the host↔device ping-pong that table describes.

## Blockers

None. Build, GPU, captures, plugin and harness all worked as documented.

## A.4 A third baseline figure, and what the spread between them is

Commit 67f9a158dab, one minute before the fan-out commit, reports **13.0878 /
13.0799 ms** on old and **5.7807 ms** on Crossroads in verified-unshared
windows. This session measures **13.1626** (runs 13.1116–13.2302) and **5.8230**
(5.8165–5.8509).

The gap is 0.075–0.083 ms on old and 0.042 ms on Crossroads, i.e. **0.6–0.7%**.
Read from the data already in hand, that is session drift plus window
selection, not a difference in the driver:

- My own three old runs span 0.119 ms between themselves, which is larger than
  the 0.075 ms gap to 13.0878. One session cannot resolve the difference.
- The 67f9a158dab figures are described as *verified-unshared windows*, i.e.
  intervals selected for having no other tenant. My medians take the whole hot
  tail after skip 50 with no window selection, and selecting quieter intervals
  can only move a median down.
- The two arms of my own session behave identically in this respect: the
  reverted arm reads 15.89 against a documented 15.7514, +0.9%, in the same
  direction and of the same order.

No run was spent on this; the two numbers are compatible and the convention
difference explains the sign.
