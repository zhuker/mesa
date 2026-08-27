# Item 1 - drain census (JOB A) - RESULT

Run: /tmp/perf-audit/run_jobA.sh, ICD
/tmp/destroydefer-tree/build-cudavk-destroydefer/src/cudavk/cudavk_devenv_icd.x86_64.json,
`CUDAVK_DESTROY_CENSUS=1`, one replay per capture.

## Exclusivity (the user's binding rule)

1 Hz sampler (`/tmp/perf-audit/gate/gpu_watch.sh`) over the WHOLE window.
Start gated on 8 consecutive idle samples (streak reached, `gate_idle_streak=8`).
Every entry of every sample audited, not just presence of my own process.

* whole window: **48 samples, t+0..t+48 s, none foreign**
* census-old run: **31 samples, t+7..t+37, none foreign**
* census-cross run: **11 samples, t+37..t+48, none foreign**

Gates before any number was read: old rc=0 submits=**3022** sha=320e993599cc;
cross rc=0 submits=**2994** sha=e727020fc796. Both match the expected replay
lengths.

## CHECK FIRST - the two instruments agree, EXACTLY

The wait census prices the `vkDeviceWaitIdle` site at 3,391 waits on old.
The three census sites that reach that entry point are `app_wait_idle`,
`destroy_image`, `destroy_view` (`free_memory` takes the call but drains 0):

    1510 + 814 + 1067 = 3,391      (0.00% off, not 1%)

Blocked at those three sites is 0.4847+0.0001+0.0149 = **0.4997 ms/frame**
against the wait census's 0.514 ms/frame (2.7% low) and its 0.500 ms/frame
ceiling (0.06% off). The instruments agree; everything below may be read.

## THE DECIDING NUMBER - the driver's share is 3.0%, gate FAILED, JOB B CANCELLED

| capture | app_wait_idle | destroy_image | destroy_view | site total | **driver share** |
|---|---:|---:|---:|---:|---:|
| old        | 0.4847 | 0.0001 | 0.0149 | 0.4997 | **3.00%** |
| Crossroads | 0.3782 | 0.0001 | 0.0008 | 0.3791 | **0.24%** |

(ms/frame blocked.)

Forecast was 25-50%, "above 70% the briefed 0.22 ms stands", "below 15% job B
is cancelled". Measured **3.00%** on old and **0.24%** on Crossroads.
**Job B is cancelled.** Worth at the site's +0.44 slope: 0.0150 x 0.44 =
**0.0066 ms/frame** on old, i.e. nothing.

The count split and the cost split point in opposite directions, and that is
the finding: by COUNT the driver owns 55.5% of the site (1,881 of 3,391), by
BLOCKED TIME it owns 3.0%. The driver's drains arrive at a device that is
already empty; the application's `vkDeviceWaitIdle` is what actually waits.

Per-drain cost, which shows why:

| site | old us/drain | Crossroads us/drain |
|---|---:|---:|
| app_wait_idle | 485.0 | 378.7 |
| destroy_view  | 21.1 | 2.2 |
| destroy_image | 0.28 | 0.28 |

`destroy_image` is already a null drain (0.28 us, the price of the call).
`destroy_view` is the only driver site with any cost at all, and it is
0.0149 ms/frame on old, 0.0008 on Crossroads.

## Second registered prediction - CONFIRMED, the cache's sync is a null sync

Predicted: `cache_*_sync` blocked under 20% of `destroy_*` blocked (P3, "the
second drain costs the price of a null sync").

| capture | cache_* blocked ms | destroy_* blocked ms | ratio |
|---|---:|---:|---:|
| old        | 0.370 | 22.742 | **1.63%** |
| Crossroads | 0.164 |  1.313 | **12.49%** |

Both well under 20%, neither near the 50% that would mean something reaches
the device in between. Prediction P3 holds on both captures. Note the
Crossroads ratio is higher only because the DENOMINATOR collapsed
(destroy_view 21.1 -> 2.2 us/drain); the cache syncs themselves are 0.19 and
0.19 us/drain, i.e. constant and null on both.

## Other census facts

* `free_memory`: 1,798 calls (old) / 1,570 (Crossroads), **0 drains** on both.
  The `texture_cache && mem->bindings` predicate never fires in these captures.
  `vkFreeMemory` contributes zero, as the design predicted.
* `cache_purge_sync` and `retire_overflow`: 0 calls on both.
* Retirement queue is inert with the defer flag off, as required:
  `deferred=0 immediate=0 freed=0 peak_pending=0 overflow_drains=0 declined=0
  still_pending=0 peak_bytes=0`.
* TOTAL blocked incl. the cache syncs: 0.5000 ms/frame old, 0.3792 Crossroads.

## Raw

/tmp/perf-audit/jobA/{census-old,census-cross}/stderr, watch.log, progress.log.
