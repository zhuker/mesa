# Item 1 — the object-destruction device drains

Worktree `/tmp/destroydefer-tree` (detached at `43df52c383e`), build
`/tmp/destroydefer-tree/build-cudavk-destroydefer`. No GPU was used for
anything below; the one measurement-shaped number in it comes from the capture
files, not from a replay.

Two parts, as asked:

* **(a) `CUDAVK_DESTROY_CENSUS`** — instrumentation only, built, checked, inert
  with the flag off, **ready to measure now**.
* **(b) the deferred-free list** — designed below, behind its own flag, built
  after (a) is in the measurer's hands.

---

## 0. Summary of what changed in the briefing's picture

Three corrections, each cite-checked, none of which kills the item:

1. **The corroboration does not corroborate.** `cpvk_texture_cache.c` does not
   include `cp_smallop_tele.h` (only `cp_renderer.c`, `cpvk_cmd.c` and
   `cpvk_device_memory.c` do), so the cache's own `cuCtxSynchronize` calls were
   **never in the small-operation census**. Its 2.34 syncs/frame is
   `cpvk_DeviceWaitIdle` (2.24, `cpvk_device_memory.c:645`) plus the renderer's
   upload-arena wrap (0.05) and scratch reclaim (0.04) = 2.33. So the census
   says nothing about who calls `vkDeviceWaitIdle`, and it cannot see the
   second drain at all.
2. **But the capture files can, and they close the count exactly.** See §4:
   `1,510 + 1,067 + 814 = 3,391`, which is the wait census's own
   `0 / 3,391`. The split is therefore **44.5% application, 55.5% ours**, by
   count, and `vkFreeMemory` contributes **zero**.
3. **"Drains twice" is true as a call count and probably false as a cost.** The
   cache's `cuCtxSynchronize` runs on a context the drain immediately before it
   has just emptied, and nothing is issued in between, so it should cost the
   price of a null sync, not a second drain. That is prediction P3, and the
   census times the two separately so the measurer settles it in one run.

---

## 1. What the drain is actually protecting, per site, with the reader named

### Site A — `vkDestroyImage`, `cpvk_image.c:376-379`

After the drain, `cpvk_texture_cache_image_destroy()`
(`cpvk_texture_cache.c:927-948`) calls `cpvk_cache_drop_locked()`
(`:832-874`), which destroys, for that image:

* every `CUtexObject` in `cache->objects` (`cuTexObjectDestroy`, `:841-847`);
* the per-level `CUsurfObject`s (`:849-852`);
* the cache's `ready` `CUevent` (`:853-854`);
* the `CUmipmappedArray` that holds the cached copy (`:855-861`);
* and back in `_image_destroy`, `image->writer_event` (`:938-943`).

**Readers.** The `CUtexObject` is resolved by
`cpvk_texture_cache_resolve[_batch]()` and lands in the device-side texture
tables the fragment kernels read; a `tex` instruction in a fragment shader
still in flight is holding that handle, and the mipmapped array is the storage
it fetches from. The surface objects have a second reader: the cache's own
conversion kernels, enqueued by `cpvk_cache_rebuild()` (`:303-...`, args
`.surface = cache->surfaces[l]`), which write into the array on
`dev->renderer.stream`.

### Site B — `vkDestroyImageView`, `cpvk_image.c:580-584`

`cpvk_texture_cache_view_destroy()` (`:788-825`) clears the view's cookie slot
in `dev->texture_cache_views[]` and unlinks and destroys the
`cpvk_texture_object`s belonging to that view. Then — **outside** the
`texture_cache` branch, and therefore with no drain at all when the cache is
off — `cpvk_DestroyImageView` calls `cuMemFree(view->tex_info)`
(`cpvk_image.c:594-598`), a device allocation holding the `cp_texture_info`
that shaders dereference.

**Readers.** The same in-flight fragment launch, through the descriptor row
that names `view->tex_info`, and through any `CUtexObject` built from this
view.

### Site C — `vkFreeMemory`, `cpvk_device_memory.c:927-928`

The drain is taken only when `cp_debug->texture_cache && mem->bindings`. What
follows frees the backing allocation: `cuMemFree`, or for exportable memory
`cuMemUnmap` + `cuMemAddressFree` + `cuMemRelease`.

**Readers.** Any in-flight kernel reading the images or buffers bound there,
plus one the application cannot know about: `cpvk_cache_rebuild()` reads
`image->mem->dev_ptr + image->offset + image->level_offset[l]` as the **source**
of the cache copy (`cpvk_texture_cache.c:313-320`).

**And by count this site never fires** — see §4. `cpvk_DestroyImage` calls
`cpvk_memory_note_unbind()` first, so by the time the application frees the
allocation its ledger is empty.

---

## 2. Whether the Vulkan contract already guarantees it

Yes, for everything the application named, and the relevant VUIDs are explicit
that the guarantee is about *completion*, not about submission:

* `vkDestroyImage`: *"All submitted commands that refer to image ... must have
  completed execution"*.
* `vkDestroyImageView`: the same for the view.
* `vkFreeMemory`: *"All submitted commands that refer to this memory ... must
  have completed execution."*

So all three drains are **defensive, not required** — with three qualifications
that must travel with that sentence:

1. **The cache's own objects are covered transitively, not separately.** A
   cached array or texture object is driver-private and is never named by an
   application command; but it is only ever *read through* the image or view
   the application is destroying, so the application's promise about that image
   covers it. The same is true of the rebuild copies: they are issued from
   inside `vkQueueSubmit` on `dev->renderer.stream` and complete with that
   submit's event.
2. **The references the driver holds that the application does not know about
   are host-side, and a device drain is the wrong tool for them.**
   `dev->texture_cache_views[]`, the cookie slots, the per-image cache list and
   the batch workspace are protected by `texture_cache_lock` and
   `texture_cache_use_lock` — locks, not synchronisation. Removing the drain
   does not touch that protection.
3. **What the driver genuinely needs is narrower than a drain and is exactly
   what a retirement queue gives:** the CUDA handles must outlive every
   *submit* that could have referenced them. Since the driver issues device
   work **only from inside `vkQueueSubmit`** (handoff §12.5 — recording touches
   CUDA not at all), "all pending submits complete" and "the device is idle"
   are the same statement here, so a deferral keyed on the last submit's
   completion event is **equivalent to the drain**, not weaker than it.

The honest residual risk is an application that violates the VUID and today
gets away with it because we drain. A retirement queue keyed on the last
submit's event **still covers that application**, because it retires no earlier
than the drain would have returned.

---

## 3. The cache's own sync — redundant at two of its three call sites

`cuCtxSynchronize` appears three times in `cpvk_texture_cache.c`:

| line | function | caller | drained already? |
|---|---|---|---|
| `:797-799` | `_view_destroy` | `cpvk_DestroyImageView` only | **yes**, `cpvk_image.c:582` |
| `:933-935` | `_image_destroy` | `cpvk_DestroyImage`; `cpvk_FreeMemory`'s binding loop | **yes**, `cpvk_image.c:377` / `cpvk_device_memory.c:928` |
| `:896-898` | `_purge` | the renderer, under budget pressure | **no — load-bearing, keep** |

`grep` confirms there are no other callers of `_image_destroy` / `_view_destroy`
(`cpvk_image.c:378`, `:583`, `cpvk_device_memory.c:933`; the last is inside the
loop that only runs when `mem->bindings`, which is exactly the condition that
took the drain two lines earlier).

So the first two are redundant **given their callers**, and the comment above
the view one records that it is a survivor of the context-push era. Deleting
them is a four-line change — but §0.3: it should cost the price of a null sync,
not half the site. **P3 measures it before anyone deletes anything.** The
retirement design in §6 removes both properly anyway, because the cache
teardown moves off the destroy path entirely.

---

## 4. How many of the calls are the application's — answered before the run,
## from the capture, and it closes to the unit

`gfxrecon-convert --format jsonl` over the whole old capture, counting API
entry points (no GPU, no replay). Full-trace totals, 3,022 submits = **1,511
frames**:

| call | count | per frame |
|---|---:|---:|
| `vkQueueSubmit` | 3,022 | 2.000 |
| `vkDeviceWaitIdle` (the application's own) | **1,510** | 0.999 |
| `vkDestroyImageView` | **1,067** | 0.706 |
| `vkDestroyImage` | **814** | 0.539 |
| `vkFreeMemory` | 1,798 | 1.190 |
| `vkWaitForFences` | 1,510 | 0.999 |
| `vkCreateImageView` / `vkCreateImage` / `vkAllocateMemory` | 1,069 / 815 / 1,799 | — |

**`1,510 + 1,067 + 814 = 3,391`, and the wait census's `vkDeviceWaitIdle` row
is `0 / 3,391`.** The account closes exactly, on the first try, and it forces
`vkFreeMemory`'s contribution to be **zero** — consistent with
`cpvk_memory_note_unbind()` emptying the ledger at image destroy.

So, **by count**:

| site | waits | share | per frame |
|---|---:|---:|---:|
| application `vkDeviceWaitIdle` | 1,510 | 44.5% | 0.999 |
| `vkDestroyImageView` | 1,067 | 31.5% | 0.706 |
| `vkDestroyImage` | 814 | 24.0% | 0.539 |
| `vkFreeMemory` | 0 | 0% | 0 |

**Crossroads, same instrument**, 2,994 submits = 1,497 frames:
`vkDeviceWaitIdle` **1,495** (0.999/frame), `vkDestroyImageView` **556**
(0.371), `vkDestroyImage` **322** (0.215), `vkFreeMemory` 1,570,
`vkWaitForFences` 1,497. Site total **2,373 = 1.585 waits/frame**, split
**63.0% application / 37.0% ours** — so the application's share is *larger* on
the capture where the frame is shorter, and both captures call
`vkDeviceWaitIdle` **exactly once per frame**.

**This is a count, not a cost**, and the two need not have the same split: the
application's single drain per frame arrives after a whole frame of submitted
work, while the driver's 1.25 destroys/frame arrive after it. That is exactly
what part (a) measures, and it is the number that sizes the item.

---

## 5. Part (a) — `CUDAVK_DESTROY_CENSUS`, ready to measure

One flag, `CP_FLAG_BOOL_VALUE`, default off, registered in
`src/cudavk/cp_debug.c`; `FLAGS.md` regenerated; `cp_debug_doc.py --check` and
`cp_no_getenv.py` both pass. No `getenv` anywhere but the registry.

**What it does.** Seven named sites (`enum cpvk_drain_site`, `cpvk_private.h`):
`app_wait_idle`, `destroy_image`, `destroy_view`, `free_memory`,
`cache_image_sync`, `cache_view_sync`, `cache_purge_sync`. For each it records
`calls` (arrivals at the site), `drains` (the ones that really synchronised)
and the host nanoseconds blocked inside that drain. The application's own
`vkDeviceWaitIdle` is told apart from the driver's by giving the internal
callers `cpvk_device_drain(device, site)`; the entry point is the one that
reports `CPVK_DRAIN_APP`.

**Run it.**

```bash
GFX=~/gfxreconstruct/build
CUDAVK_DESTROY_CENSUS=1 \
VK_DRIVER_FILES=/tmp/destroydefer-tree/build-cudavk-destroydefer/src/cudavk/cudavk_devenv_icd.x86_64.json \
  $GFX/tools/replay/gfxrecon-replay -m remap --remove-unsupported \
  ~/headless_streamer_20260814T155742.gfxr > OUT/stdout 2> OUT/stderr
```

and read the block on **stderr** at teardown:

```
cudavk: drain census over 1511.0 frames (3022 submits)
cudavk:   app_wait_idle     calls=... drains=... blocked_ms=... per_frame=... blocked_ms_frame=...
...
cudavk:   TOTAL             calls=- drains=... blocked_ms=... per_frame=... blocked_ms_frame=...
```

Nothing is written to stdout, so the stdout hash is unchanged and WORKFLOW §4.4
still applies. It is a probe run, not a timed run (two `clock_gettime` reads per
drain); no frame median from it should be quoted.

**Inertness, at machine level.** Base `43df52c383e` objects against the census
objects, same build directory, `objdump -d` with addresses and branch offsets
normalised:

| object | changed instruction lines |
|---|---:|
| `cp_renderer.c.o` | **0** |
| `cpvk_cmd.c.o` | **0** |
| `cp_debug.c.o` | 8 (the new registry row) |
| `cpvk_image.c.o` | the two added `call cpvk_drain_census_call` sequences, branch renumbering and assert line numbers — nothing else |
| `cpvk_device_memory.c.o`, `cpvk_texture_cache.c.o` | the census functions and their call sites |

The two objects that read `struct cp_debug` on every draw are **byte-identical
in their instruction stream**, and that is why the new field is appended at the
end of `struct cp_debug` and the new counters at the end of
`struct cpvk_device`: inserting either into its natural group moved every later
member and produced 1,601 changed instructions in `cp_renderer.c.o` — all of
them harmless displacement changes, and all of them noise in exactly the proof
that matters. Off, the flag costs one predictable branch per destroy, per
`vkFreeMemory` and per submit, on paths that already take a whole-device drain.

`cpvk_device_drain()` keeps the entry point's `CPVK_CTX_SCOPE`, so the internal
callers still take the same redundant context push they take today. That push
is removable and worth about 0.0005 ms/frame; it is deliberately **not** part
of this arm, so the census differs from the base by counters only.

---

## 6. Registered predictions, before any measurement

R7 and R8: several falsifiers, not one, and they are chosen to disagree.

**P1 — the counts, forecast from the capture (§4).**
`app_wait_idle` drains = 1,510; `destroy_view` = 1,067; `destroy_image` = 814;
`free_memory` calls ≈ 1,798 with **drains = 0**; site total = **3,391**;
`cache_view_sync` = 1,067 and `cache_image_sync` = 814 + a handful from the
`vkFreeMemory` loop; frames = 1,511.

* **F1a** site total ≠ 3,391 ± 1% → this counter and the wait census disagree,
  and one of the two instruments is wrong. Settle that before reading anything
  else.
* **F1b** `free_memory` drains > 50 → the binding ledger is not empty at free
  time and §1's site C is live after all.
* **F1c** `app_wait_idle` ≠ 1,510 ± 2% → the replayer issues drains the capture
  does not contain (or drops some), and every count in §4 needs re-deriving.
* **F1d** `cache_*_sync` calls ≠ the matching destroy counts → a caller of the
  cache teardown exists that `grep` did not find.

**P2 — the split of blocked time, which is what sizes the item.**
I predict the driver's three sites carry **less** than their 55.5% count share
of the 0.514 ms/frame, because the application's one drain per frame arrives
after a whole frame of submitted work and the driver's 1.25 destroys arrive
after it. **Point forecast: driver share 25–50% of blocked time
(0.13–0.26 ms/frame).** At the site's measured slope of **+0.44** that is
**0.06–0.11 ms/frame of frame time**, i.e. **half the briefed 0.22**.

* **F2a** driver share > 70% → item 1 is worth ≈0.16 ms/frame and the briefing's
  0.22 is nearly right; build (b) and land it.
* **F2b** driver share < 15% → item 1 is worth under 0.03 ms/frame. **Do not
  land the mechanism**; the site is the application's own wait, and only §12.5's
  inter-submit stall can touch it.
* **F2c** the site total blocked ≠ 0.514 ms/frame ± 20% → the census's cost, not
  just its count, failed to reproduce; suspect the run, not the model
  (WORKFLOW §4.4: check the submit count first).

**P3 — the second drain is nearly free.**
`cache_image_sync + cache_view_sync` blocked < **20%** of
`destroy_image + destroy_view` blocked, because it synchronises a context the
drain immediately before it just emptied and nothing is issued in between.

* **F3a** > 50% → something *does* reach the device between the two waits, which
  is a finding in its own right and the four-line deletion in §3 becomes the
  whole item.
* **F3b** < 2% → a null `cuCtxSynchronize` is free, the "drains twice" framing
  should be retired from the census, and only the first drain is worth removing.

**P4 — the frame convention.** `submits / 2` = 1,511 ± 1 frames, and the report
divides by it. If it does not, every `_frame` column in the report is wrong by
that ratio and the raw totals should be used instead.

Note the deliberate disagreement: **P1 says the driver owns 55.5% of the
waits and P2 says it owns less than that of the cost.** If both hold, the item
is real but half-sized. If P1 holds and P2 fails high, it is the briefed size.
If P1 fails, nothing else in this document should be read.

---

## 7. Part (b) — `CUDAVK_DESTROY_DEFER`, the retirement queue

Built, compiles clean, `cp_debug_doc.py --check` and `cp_no_getenv.py` pass.
Branch `destroy-defer` in `/tmp/destroydefer-tree`, second commit.

### 7.1 The design that survived reading the code

**The claim the whole thing rests on, and it is checked rather than assumed:**
the driver issues device work **only from inside `vkQueueSubmit`**, and every
submit ends with a `CUevent` recorded on `dev->renderer.stream` *after* the
side streams have been joined into it — `cp_pass_join()`
(`cp_renderer.c:7386-7407`) records an event on each of the eight
`seg_streams[]` and makes the main stream wait on it; every episode finish
calls it (`:7493`, `:7693`, `:8532`, `:8594`, `:8911`, `:9121`, `:9236`);
`cpvk_queue_submit` then runs `cp_batch_flush()`, `cp_upload_flush()` and
records `done` on that stream. Every `vk_sync` this driver signals already
depends on that event meaning "the submit is complete", so a retirement keyed
on it is as strong as the drain, not weaker.

**Therefore:** *"every submit that could have referenced this object has
completed"* and *"the device is idle"* are the same statement in this driver,
and retiring at the newest pending submit's event returns no earlier than the
drain would have.

### 7.2 What the code does

`struct cpvk_retire` (`cpvk_texture_cache.c`) owns only CUDA handles — the
detached `struct cpvk_texture_cache` (its `CUmipmappedArray`, `CUtexObject`s,
`CUsurfObject`s and `ready` event), a view's unlinked `cpvk_texture_object`
list, `image->writer_event`, and `view->tex_info` — and **no pointer to any
Vulkan object**, so `vk_image_destroy()` and `vk_image_view_destroy()` run at
once as they do today.

1. **The host side is unchanged and still immediate.** The cookie slot, the
   `dev->texture_cache_images` list, the per-image object list and
   `dev->texture_cache_bytes` are updated under `texture_cache_use_lock` +
   `texture_cache_lock`, in the same order, at the same point. Those were never
   protected by the synchronisation; the locks protected them. **The budget is
   credited at detach, not at retirement**, so a deferred free is never
   deferred accounting.
2. **The barrier** is `cpvk_submit_barrier()`: a reference to
   `dev->submit_tail->done` under `submit_lock`, or **NULL when the submit list
   is empty** — in which case the record is released on the spot and the
   destroy costs nothing at all, which is exactly what the drain would have
   found.
3. **Retirement** walks the queue at three places where the host is already at
   a boundary: the head of `vkQueueSubmit` (after `CPVK_CTX_SCOPE`, because
   these are context-scoped objects), inside `cpvk_device_drain()` — a drain
   makes the device idle by construction, so it retires everything without
   querying — and at device teardown. Entries are appended in submit order and
   their events are all recorded on one stream, so the walk stops at the first
   `cuEventQuery` that is not `CUDA_SUCCESS`.
4. **It is bounded**, so deferring a free can never become holding one: past
   **256 records or 64 MiB** the enqueue drains and retires everything, and
   counts that as `retire_overflow` in the census. A `calloc` failure declines
   the deferral and the caller drains, as before.

### 7.3 One pre-existing hazard the mechanism forced into the open

The submit worker used to drop its reference to `pending->done` **before**
removing the record from the list (`cpvk_device_memory.c`), which was harmless
while nothing else read `submit_tail->done`. `cpvk_submit_barrier()` does, so
the unref now happens after the record leaves the list, under the same lock.
Nothing observes when the event is destroyed, so the move costs nothing — but
without it the barrier could reference freed memory.

### 7.4 What it does not touch, and why

* **`vkFreeMemory` still drains** when bindings remain. It is measured at
  **zero occurrences** in both captures (§4), so there is nothing to win, and
  it is the one site where the deferred object is memory handed back to CUDA
  rather than a driver-private cache entry.
* **`cpvk_texture_cache_purge()` keeps its own `cuCtxSynchronize`.** It is
  reached from the renderer under budget pressure with no drain in front of it
  (§3) and is load-bearing.
* **The cache's two redundant syncs are not deleted.** With the flag on they
  are unreachable, because the deferred path replaces the drain *and* the
  `_destroy` call together. With the flag off the arm is the base. If P3 comes
  back large they can be deleted separately, on their own evidence.

### 7.5 The limitation, stated rather than discovered later

A second thread that is **inside `vkQueueSubmit`, has already resolved this
view's texture object, and has not yet recorded its completion event** would
get a barrier one submit too old. That application is already violating
`VUID-vkDestroyImageView-imageView-01026` — the object is in use by a submit it
has not waited for — and it is not what either capture does (both replay
single-threaded). The drain does not protect it either in any sense the VUID
does not already require. It is recorded here because it is the one case where
the retirement is *not* literally equal to the drain.

---

## 8. Predictions for part (b), registered before it is run

**P5 — the mechanism fires, and the counter says so before any timing.** With
`CUDAVK_DESTROY_DEFER=1 CUDAVK_DESTROY_CENSUS=1` on the old capture:
`destroy_image` drains = **0**, `destroy_view` drains = **0**, `app_wait_idle`
unchanged at 1,510, `retire_overflow` = 0, `declined` = 0, `still_pending` = 0
at teardown, `deferred` + `immediate` ≈ 1,881.

* **F5a** `immediate` ≈ `deferred + immediate` (nearly every destroy finds no
  pending submit) → the destroys happen when the device is already idle, which
  **contradicts the wait census's `ready = 0 of 3,391`**. One of the two
  instruments is then wrong, and that has to be settled before the timing is
  read. It also implies the item is worth ~0.
* **F5b** `retire_overflow` > 0 → the 256-record / 64 MiB bound is too tight
  for this workload; raise it and re-run rather than reading the timing.
* **F5c** `still_pending` > 0 or `deferred` ≠ `freed` at teardown → a leak;
  do not measure, fix.
* **F5d** the correctness gate or the stdout hash changes → the deferral is
  freeing something early. Stop.

**P6 — the size, from part (a)'s split and the site's own slope.** The win is
`(destroy_image + destroy_view blocked ms/frame) × 0.44`. Under P2's 25–50%
share that is **0.06–0.11 ms/frame on old** and, given Crossroads' 37% count
share of a smaller site, **under 0.04 ms/frame there**.

* **F6a** the measured frame gain on old exceeds **0.15 ms/frame** → the model
  is wrong somewhere (the +0.44 slope, or the share), and the number should not
  be quoted until it is known which. A gain larger than the blocked time the
  census attributes to those two sites is not possible without another
  mechanism.
* **F6b** the candidate **loses** frame time → the retirement poll at submit,
  or the bookkeeping, costs more than the drains saved; the poll is the first
  thing to move.
* **F6c** the gain is inside the session spread (0.119 ms on old, 0.034 on
  Crossroads) in both directions → report it as not resolvable at one session's
  precision, and use `cp_decisive_ab.sh` before claiming anything.

**Arms** (`WORKFLOW` §4.2, §4.3 — the candidate carries the new behaviour):

```
candidate  CUDAVK_DESTROY_DEFER=1
control    (empty)
```

with one census run per arm alongside, **not** in the timed sequence: a probe
run is not a timed run, and R5 applies — read the counter before believing the
timing, and never read the mechanism's own effect out of a run it drove.

### Inertness of the whole patch, at machine level

Against base `43df52c383e`, `objdump -d` with addresses and branch targets
normalised: `cp_renderer.c.o` **0** changed instruction lines, `cpvk_cmd.c.o`
**0**, `cp_debug.c.o` 8 (two registry rows). The three files that carry the
mechanism differ, and with both flags clear every one of those differences is
behind a load of a `cp_debug` bool: the two new fields are at the **end** of
`struct cp_debug` and the new device state is at the **end** of
`struct cpvk_device`, so no existing member offset moves anywhere in the
driver.

---

## 9. The positive control: is the flags-off proof vacuous?

Added after a sibling agent lost a probe's whole implementation to a stray
`git checkout` and its inertness check **confirmed** the loss — a folded-off
build and a build with the feature missing are indistinguishable unless
something distinguishes them.

**Direct check of that failure mode first.** `9f84bcc27ab` carries
`cpvk_texture_cache.c` +280 lines, `cpvk_device_memory.c` +49,
`cpvk_image.c` +19, and the committed blobs contain `cpvk_retire_enqueue`,
`cpvk_retire_release`, `cpvk_retire_poll`, `cpvk_texture_cache_image_retire`
and `cpvk_submit_barrier`.

**The arms.** These gates are runtime bools, so "forced on at compile time" is
not the informative arm; the informative one is **D**, the gates hardcoded
`false`, where the compiler deletes the wiring. **If the implementation had
been deleted, C would equal D.** `.text` bytes, one build directory, one
compiler:

| object | A base | C off (shipped) | B gates true | D gates false |
|---|---:|---:|---:|---:|
| `cpvk_device_memory.c.o` | 10,321 | 11,494 | 11,223 | 10,576 |
| `cpvk_image.c.o` | 7,087 | 7,174 | 7,152 | 7,140 |
| `cpvk_texture_cache.c.o` | 11,064 | 14,165 | 14,132 | 14,041 |
| **the three mechanism files** | **28,472** | **32,833** | **32,507** | **31,757** |
| `cp_debug.c.o` | 3,011 | 3,011 | 3,011 | 3,011 |
| `cp_renderer.c.o` | 149,581 | 149,581 | 149,581 | 149,581 |
| `cpvk_cmd.c.o` | 42,126 | 42,126 | 42,126 | 42,126 |

* **C − A = +4,361** — the implementation is in the shipped flags-off binary.
* **C − D = +1,076** — the wiring that exists only because the flags are read
  at runtime.
* **B − A = +4,035** — the on path is compiled and not trivial.
* **D − A = +3,285** — survives folding because the retire and census entry
  points are external symbols and cannot be dropped. For a patch of this shape
  the symbol evidence below is the decisive control, not the size delta.

**How §5 should have been worded.** The "0 changed instruction lines" is
`cp_renderer.c.o` and `cpvk_cmd.c.o` — **two files this patch never edits**.
That is a *no-collateral-damage* result (no struct member offset moved), not an
inertness result. State it as **"0 collateral, +4,361 present, +1,076 removable
by folding."**

**The second control, one command, and stronger here than the size delta.**
The relocations of the flags-off `cpvk_DestroyImage`:

```
base:    cpvk_DeviceWaitIdle  cpvk_memory_note_unbind  cpvk_texture_cache_image_destroy  vk_image_destroy
shipped: cpvk_device_drain  cpvk_drain_census_call  cpvk_texture_cache_image_retire
         cpvk_memory_note_unbind  cpvk_texture_cache_image_destroy  vk_image_destroy
```

```bash
objdump -dr --no-show-raw-insn -j .text.cpvk_DestroyImage <obj> | grep -A1 call | grep R_X86_64
```

The flags-off entry point can **reach** the deferral; a deleted implementation
cannot produce that relocation. 13 new symbols are present in the shipped `.so`
and the base object contains zero matching `retire`.

**Rule 1 was already applied, and was measured here independently:** the flag
placed in its natural group in `struct cp_debug` shifted every later member by
4 bytes and put **1,601** changed displacement lines into `cp_renderer.c.o` and
9 into `cpvk_cmd.c.o`. Both new fields are at the end of that struct and the new
device state at the end of `struct cpvk_device`.

**Rule 2 has one instance here, and it is not hidden:**
`cpvk_drain_census_start/_note/_call` are called unguarded and check the flag
inside, so they survive folding and are part of the 1,076 bytes. Not changed
yet, because the measurer may be running against this artifact.

**Disclosure.** Producing B and D took three in-place rebuilds of
`build-cudavk-destroydefer`. The tree is back to the committed state: `git
status` clean, all restored `.text` sizes identical to C, `cp_renderer.c.o` and
`cpvk_cmd.c.o` still at 0 changed instruction lines against base, and both
checkers still pass.
