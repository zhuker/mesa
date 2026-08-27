# L6 — depth/stencil load/store elision: checked at HEAD, and CLOSED

**Design and verification only. No code written. No GPU used.** Every number
below comes from measurements already on disk in this project; I added none.

Repo `/tmp/bnd-tree`, a worktree of `e2fea470d04`, which is still HEAD of
`cudapipe-vk-native`. Line numbers are that commit's.

---

## 0. VERDICT

**L6 is worth about 0.005 ms/frame, not 0.1–0.2 ms. It is 20–40x smaller than
the lead says, it is smaller than the session noise on both captures, and it
buys that with a correctness surface whose failure mode is a silently stale
depth buffer. Do not build it.**

| | claimed by L6 | measured, at HEAD |
|---|---:|---:|
| elidable full-screen kernels / frame, old | ~1.1 | **1.10** |
| elidable full-screen kernels / frame, Crossroads | ~4.6 | **1.00** |
| device time per kernel | never looked up | **4.65–4.75 us** |
| device time elided / frame, old | — | **4.7 us** |
| device time elided / frame, Crossroads | — | **4.7 us** |
| host issue elided / frame | — | **1.9 us** |
| **total / frame** | **0.1–0.2 ms** | **0.005–0.007 ms** |
| as a share of the frame | 0.8–1.5% | **0.05% (old), 0.11% (Crossroads)** |
| as a share of all kernel time | — | **0.02%** |
| session spread of the frame median, old | — | 0.119 ms, **17x the whole lead** |

The lead is a twentieth of the noise it would have to be measured against.

---

## 0.5 A standalone fact about this driver, worth more than the closure

**cudavk has exactly two attachment-transfer kernels, both keyed on the depth
aspect. There is no stencil attachment kernel and no colour attachment kernel.**

* `src/cudavk/cp_kernels.h:18-19` declares `depth_attachment_load` and
  `depth_attachment_store`, and nothing else of that shape.
* `cpvk_cmd.c:1586-1590` sets both `.load` and `.store` from
  `pRenderingInfo->pDepthAttachment` alone. The stencil attachment is consulted
  only for `.stencil_clear` and `.stencil_value` (`:1594-1597`), which ride
  inside the depth store.
* `cpvk_cmd.c:1511` sets `fb.color` to the colour image's own device pointer.
  Rendering writes image memory in place, so a colour `loadOp`/`storeOp` costs
  nothing at all.

**Consequences that stand on their own, independent of L6:**

1. **Stencil load/store traffic is already elided, by construction.** Nothing
   is left to remove there, and nothing can be gained by trying.
2. **Colour load/store traffic is already elided, by construction** — including
   the old capture's 3072x2176 RGBA16F chained load, which is the largest
   single item in the census and which this driver has never performed.
3. **A capture-level census counts API operations, not driver kernels.** Any
   analysis of `loadOp`/`storeOp` in a gfxrecon stream will over-count this
   driver's work unless it is first mapped onto these two call sites. The
   census in `/tmp/cpvk-loadstore.md` is correct about the *capture* and wrong
   about the *driver* by a factor of about four, and anyone reading it will get
   it wrong the same way.
4. **The driver honours a stencil `storeOp` only when the depth `storeOp` is
   also STORE.** On Crossroads 3,213 passes store stencil while only 1,496
   store depth, so 1,717 stencil stores per replay are dropped. The census
   independently finds every one of those stencil stores dead, so this is not a
   bug today — but it is an assumption, written down at `cpvk_cmd.c:1591-1593`
   ("without a store the aspect's contents are undefined anyway"), and a
   capture that reads a stencil buffer whose depth aspect was DONT_CARE would
   break on it. Recorded here because nothing else records it.

---

## 1. Where the 20–40x went. Three errors, each independently sufficient

### 1.1 The census counted Vulkan aspect-ops; this driver launches one kernel, keyed on the depth aspect only

`cpvk_cmd.c:1586-1597`:

```c
.load  = dat->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || (store && !full_area),
.store = store || dat->resolveMode != VK_RESOLVE_MODE_NONE,
.stencil_clear = stencil && stencil->imageView &&
                 stencil->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR,
.stencil_value = stencil ? stencil->clearValue.depthStencil.stencil : 0,
```

`dat` is `pRenderingInfo->pDepthAttachment`. **The stencil attachment is read
only for `stencil_clear` and `stencil_value`.** Its `loadOp` never causes a
load and its `storeOp` never causes a store. There is no stencil kernel at all
— `grep stencil src/cudavk/cp_kernels.h` is empty, and `cp_clear.cu` has
exactly two attachment kernels, `cp_depth_attachment_load` and
`cp_depth_attachment_store`, both of which handle the packed aspect inside the
one launch.

The driver says so in its own comment at `cpvk_cmd.c:1591-1593`: *"A stencil
clear on the shared aspect is honoured by the store, which is the only place
this driver writes stencil bits."*

So Crossroads' **1.43 stencil LOAD + 2.15 stencil STORE per frame are zero
kernels here.** The census's `~4.6 full-screen kernels/frame on Crossroads`
becomes the depth row alone: **1.00**.

Put the other way round, and this is the useful form: **the stencil traffic is
already elided, by construction.** L6 proposes to remove work the driver has
never done.

### 1.2 The colour rows are also zero kernels, including the headline one

`cpvk_cmd.c:1511`: `fb.color = (void *)(uintptr_t)(cimg->mem->dev_ptr + ...)`.
Rendering writes the image's own memory in place. There is no colour
attachment load or store kernel anywhere in the driver.

The census knows this — *"color LOAD/STORE are free here because rendering
writes image memory in place"* — but its own summary line then adds
*"+ 4.3 color load/stores (1.5 Mpx)"* to the Crossroads total and
*"+ 3.8 color load/stores (8.5 Mpx)"* to old, and its implementation
recommendation is *"retention-based load elision first ... it carries the
single biggest item: the old capture's full-screen RGBA16F reload"*. **That
item does not exist in this driver.** It is a 6.7 Mpx colour load the driver
never performs. The contradiction is inside the census, and L6 inherited the
wrong half.

### 1.3 The per-kernel price was never looked up, and it is in this tree

Two independent Nsight Systems traces, both already on disk:

```
/tmp/perf16/iter3-final-kernel-summary.json
  cp_depth_attachment_store  625 launches  mean 4.65 us  total 2.91 ms  0.02% of kernel time
  cp_clear_depth_kernel      625 launches  mean 2.40 us  total 1.50 ms  0.01%
/tmp/perf16/iter2-after-kernel-summary.json
  cp_depth_attachment_store  606 launches  mean 4.75 us  total 2.88 ms
```

0.1–0.2 ms over 4.6 kernels implies **22–43 us each**, five to nine times the
measured 4.65 us.

And there is no slower reading available, because **the kernel is already at
the memory roofline.** At 1280x720x1 sample it moves 0.92 Mpx x 4 B in and
4 B out = 7.4 MB. 7.4 MB / 4.65 us = **1.6 TB/s**, which is roughly this card's
peak. `cp_clear.cu:84-110` is a flat 16x16 grid doing one 32-bit load, one
convert and one 32-bit store per pixel. There is nothing in it to speed up and
nothing hiding in it.

---

## 2. The check against HEAD, cross-validated four ways

I did not take the census's word for the counts. All four agree:

1. **CUDA API interception over the full old replay**
   (`/tmp/perf16/iter4-vfetch-counts.txt`):
   `cp_depth_attachment_store` **1,646 launches, 2.6 ms of host issue**;
   `cp_depth_attachment_load` **136 launches, 0.2 ms**. Out of 2,409,643
   `cuLaunchKernel` calls — **0.074%**.
2. **The census's depth rows for the same capture**: 1,646 depth STOREs and
   136 depth LOADs. **Exactly equal.** That is what proves §1.1: if stencil or
   colour produced kernels, the interception count would exceed the depth row.
3. **`cp->plan.scopes` at HEAD**, from the measurer's own STEP B run
   (`/tmp/perf-audit/stepB/cross-plan/stderr`): *"over 17820 render scopes"* on
   Crossroads. The census counted **17,820 render passes**. A render pass is a
   render scope, one for one, and nothing about that has moved since the census
   was taken.
4. **`cp_depth_attachment_store` is used elsewhere in this project as an exact
   frame counter** (`/tmp/perf16/iter25-profile/report.md`: *"fires once per
   frame (314 launches ... Crossroads 697)"*). Independent confirmation of
   1/frame on both captures.

Then the arithmetic, at HEAD frame times of 13.1626 ms (old) and 5.8230 ms
(Crossroads):

```
old         stores 1,646/1,511 frames = 1.089/frame, 92.8% elidable (1,528/1,646)
            loads    136/1,511        = 0.090/frame, 100% elidable
            device   (1.012 + 0.090) x 4.65 us = 5.1 us/frame
            host      2.8 ms / 1,511 frames    = 1.9 us/frame
            total    0.0070 ms/frame  =  0.053% of the frame
Crossroads  stores 1,496/1,497 frames = 1.00/frame, 99.9% elidable (1,495/1,496)
            loads  0
            device   4.75 us/frame,  host ~1.7 us/frame
            total    0.0065 ms/frame  =  0.11% of the frame
```

Against the measurer's own STEP A spread — three old run medians spanning
**0.119 ms**, three Crossroads medians spanning **0.034 ms** — L6 is
**1/17 of the noise on old and 1/5 on Crossroads.** It is not a small win. It
is not measurable at all.

---

## 3. The design, written out, because the mechanism deserved checking too

The arithmetic alone would close this. I checked the mechanism as well, so that
"it is too small" is not resting on "and I did not look".

### 3.1 What would be elided, at which line

Exactly two call sites:

| | site | today |
|---|---|---|
| LOAD | `cp_renderer.c:9494`, in `cp_render_scope_begin` | `if (scope->depth.load && cp_depth_attachment_xfer(cp, scope, false)) cp->depthbuf_cleared = true;` |
| STORE | `cp_renderer.c:9507`, in `cp_render_scope_end` | `if (cp->pass.scope.depth.store) cp_depth_attachment_xfer(cp, &cp->pass.scope, true);` |

**Retention (LOAD).** Keep a `depthbuf_holds` descriptor —
`{data, format, width, height, samples, row_stride, sample_stride,
pixel_stride}` — set on every load and every store. Skip the load when the
incoming `scope->depth` matches it and no invalidation has intervened.

**Laziness (STORE).** At scope end, record the pending store instead of
launching it. Materialise it before any consumer of that image. Drop it when a
full-area discard of the same image+aspect arrives first.

### 3.2 What would prove it cannot change a pixel

Retention is sound only if the depthbuf still holds the bits the load would
have written. Five conditions, and each is a real invalidation site in this
tree:

1. **The attachment is the same one.** `cp->depthbuf` is a *single* internal
   buffer, grow-only (`cp_renderer.c:9579`), shared by every attachment. Two
   consecutive depth scopes on different images alias it.
2. **The previous scope wrote it back, or its store is still pending.** Both
   give the same bits; a pending store is retention's friend, not its enemy.
3. **`cp_context_set_framebuffer` did not resize or reallocate.** `:9563-9596`
   frees and reallocates on growth and clears `depthbuf_cleared`. Retention
   must clear with it.
4. **Nothing cleared it.** `cp_clear_depthbuf` (`:2748`) memsets the whole
   buffer and sets `depthbuf_cleared`; the lazy per-segment clear at
   `:9309-9319` does the same inside an episode.
5. **Nothing wrote the *image* behind the driver's back.** Every
   `vkCmdCopyImage`, `vkCmdCopyBufferToImage`, `vkCmdBlitImage`,
   `vkCmdClearDepthStencilImage`, host-mapped write, and external/imported
   access is an invalidation.

Laziness is sound only if the pending store is materialised before **every**
consumer. That set is the risk the original doc named, and it is not small:
transfer source, transfer destination that only partially overwrites, sampled
through any descriptor, blit source, resolve source, copy-to-buffer, host map,
queue-family release, external export, and context teardown.

**The failure mode of missing one is a silently stale depth buffer**, which
surfaces as wrong pixels somewhere far from the change, possibly only in one
capture and possibly only intermittently. That is the worst shape of bug this
driver can have, and it would be bought for 0.005 ms.

### 3.3 Which existing counter would show it working

`cp->launches`, printed by `CUDAVK_PLAN_STATS=1` as
`"kernel launches: N over E episodes and S render scopes"`. It is an exact
integer and it would fall by exactly the elided count over a full replay:
**−1,528 on old, −1,495 on Crossroads**. No new instrumentation would be
needed, and no probe would be needed to size the lead either — §2 sized it from
data already on disk.

### 3.4 Failure modes, ranked

1. **A missed consumer hook** — stale depth, wrong pixels, silent. §3.2.
2. **Device-pointer aliasing.** The retention key would be `depth.data`, a
   device address. An image freed and another allocated at the same offset
   aliases it. The census's own caveat says gfxrecon *handles* are
   capture-unique; device pointers are not.
3. **Cross-submission lifetime.** A pending store outliving its command buffer,
   its memory object, or the queue submission it was recorded in.
4. **The `(store && !full_area)` load.** `cpvk_cmd.c:1586` forces a load when a
   partial-area pass will store, precisely so the untouched region survives.
   Retention must not skip that one unless the retained surface covers the
   whole image, which is a different test from "same attachment".

---

## 4. What would reopen it

Only one thing, and it is worth writing down because it is a real conditional
rather than a hedge:

**MSAA depth.** `cp_depth_attachment_xfer` launches
`(w+15)/16, (h+15)/16, samples` — the grid is `samples` deep in z, so the cost
scales linearly with the sample count. An 8x MSAA D32 1280x720 attachment is
8 x 4.65 = ~37 us per kernel, and L6 would be worth ~0.04 ms/frame at one
round trip per frame. Neither capture does this: `cp_resolve_samples` fires
**40 times in the whole old replay** (`iter4-vfetch-counts.txt`), so MSAA is
essentially unused here.

The other two conditions are the ones §1 eliminated, and neither can come back
without a driver change: this driver would have to start launching a colour
attachment kernel, or a stencil-aspect kernel. Both are absences by design.

**What would *not* reopen it:** a bigger capture, a different scene, or a
re-measure at a newer HEAD. The lead is 1 kernel per frame at the memory
roofline. Nothing downstream of it is unblocked by removing it — the store is a
plain launch on `cp->stream` at `cp_render_scope_end`, after the drains, and it
gates nothing.

---

## 5. What I would take instead, briefly

The strategy shift is right and the same data supports it. From the kernel
summary already on disk, ranked by total device time:

```
main (generated shaders)   44.7%     cp_rasterize_stage3_abuf   12.3%
cp_clip_rast_fused          8.8%     cp_clip_rast_fused_abuf     7.7%
cp_vertex_fetch             6.9%     cp_rasterize_stage3         4.0%
...
cp_depth_attachment_store   0.02%    <- L6
```

That table is from an older trace, but the measurer's STEP B puts the raster
chain at 54.7% of kernel time at HEAD, so the shape has not moved. **L6 is the
26th of 26 kernels by total time.** Anything in the top six is worth two orders
of magnitude more than it, and the two changes that have actually paid — the
opaque fan-out and PDL tier 1 — both worked by overlapping exactly that work.

I have not designed a replacement, because you asked for L6 and because
choosing the next target is yours. If you want one from me, the honest ranking
question is which of the top six has device work that is *removable* rather
than merely *overlappable*, and I would want the HEAD kernel summary — not the
iteration-3 one — before answering.

---

## 6. Sources

Everything cited here was already on disk. I ran nothing.

```
/tmp/cpvk-loadstore.md                        the census L6 rests on
/tmp/perf16/iter3-final-kernel-summary.json   4.65 us, 625 launches, 0.02%
/tmp/perf16/iter2-after-kernel-summary.json   4.75 us, 606 launches
/tmp/perf16/iter4-vfetch-counts.txt           1,646 store + 136 load launches
/tmp/perf16/iter25-profile/report.md          "fires once per frame"
/tmp/perf-audit/reprofile_baseline.md         13.1626 / 5.8230 ms at HEAD, spreads
/tmp/perf-audit/stepB/cross-plan/stderr       17,820 render scopes at HEAD
src/cudavk/cpvk_cmd.c:1511, 1586-1597         colour in place; depth aspect only
src/cudavk/cp_renderer.c:2712-2745, 9494, 9507  the two call sites
src/cudavk/kernels/cp_clear.cu:59-110         the two kernels
src/cudavk/cp_kernels.h:18-19                 there is no third one
```
