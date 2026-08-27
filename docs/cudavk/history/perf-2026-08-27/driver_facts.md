# cudavk: structural facts that make "wasted work" numbers wrong

Standalone, so they are findable without reading the design document that
produced them. Each is a property of the driver, not of a capture or a lead,
and each has already caused an over-estimate in this session.

Verified at `e2fea470d04`. Line numbers are that commit's.

---

## F1. Lanes `4q..4q+3` are the derivative quad. Uncovered lanes are helper lanes.

**The A-buffer's four-shade-slots-per-quad layout is not slack. It is the 2x2
quad that `ddx`/`ddy` are taken across, and a lane the primitive does not cover
is shaded deliberately.**

* `nir_to_ptx/cp_nir_to_llvm.c:2345-2374`, `cp_quad_derivative()`, computes
  `ddx`/`ddy` with `llvm.nvvm.shfl.sync.bfly.f32` over a quad base of
  `tid & 28`. It *assumes* lanes `4q..4q+3` are the four pixels of one 2x2
  block.
* `:2379`: *"Implicit LOD is expressed as explicit quad gradients: compute-stage
  CUDA texture instructions do not infer graphics fragment derivatives."* Every
  implicit-LOD texture sample in the driver goes through those gradients.
* `:3393-3397` states the rule: *"A quad is shaded whole so that derivatives can
  be taken across it, so a lane the primitive does not cover is shaded too.
  Vulkan says such a lane's stores and atomics have no effect."*
* `:3409-3423` is the only coverage gate in the generated shader, and it is
  emitted **only when `ctx->writes_memory`**. It suppresses helper-lane *side
  effects*, not their shading.
* The slot numbering it depends on: `cp_fs.cu:310` `base = iq * 4u`;
  `cp_rasterize.cu:1970-1972` *"cp_abuf_interpolate gives quad q the four
  shading slots 4q..4q+3"*; `cp_rasterize.cu:2601` writes `4u * *count` as the
  device-read extent.

**Consequences.**

1. `total/quads = 3.753` (median over 20,704 episodes, geometric cap 4) means
   **6.2% of shade slots are helper lanes**, not 6.2% of wasted work.
2. **Any proposal to renumber or pack the shade slots dense over fragments is
   dead on arrival.** It would break derivatives and implicit mip selection for
   every shader that uses either. It could only apply to shaders provably using
   neither, which is a much narrower and more fragile claim.
3. **An early-out on coverage would collect almost nothing anyway.** Uncovered
   lanes are scattered, so a 32-lane warp is essentially never fully uncovered
   and divergence keeps the cost.

This retracted §5 of `/tmp/perf-audit/s1d_design.md`, which was mine.

---

## F2. There are exactly two attachment-transfer kernels, and both key on the depth aspect.

**There is no stencil attachment kernel and no colour attachment kernel.**

* `cp_kernels.h:18-19` declares `depth_attachment_load` and
  `depth_attachment_store`, and nothing else of that shape.
* `cpvk_cmd.c:1586-1590` sets both `.load` and `.store` from
  `pRenderingInfo->pDepthAttachment` alone. The stencil attachment is consulted
  only for `.stencil_clear` and `.stencil_value` (`:1594-1597`), which ride
  inside the depth store.
* `cpvk_cmd.c:1511` sets `fb.color` to the colour image's own device pointer.
  Rendering writes image memory in place, so a colour `loadOp`/`storeOp` costs
  nothing at all.

**Consequences.**

1. **Stencil load/store traffic is already elided, by construction.**
2. **Colour load/store traffic is already elided, by construction** — including
   the old capture's 3072x2176 RGBA16F chained load, the largest single item in
   `/tmp/cpvk-loadstore.md`, which this driver has never performed.
3. **A capture-level census counts API operations, not driver kernels.** Any
   `loadOp`/`storeOp` analysis of a gfxrecon stream over-counts this driver by
   about four times unless it is first mapped onto these two call sites.
4. **The driver honours a stencil `storeOp` only when the depth `storeOp` is
   also STORE.** 1,717 stencil stores per Crossroads replay are dropped. The
   census independently finds every one dead, so this is not a bug today, but it
   is an assumption — written at `cpvk_cmd.c:1591-1593` — and a capture that
   reads a stencil buffer whose depth aspect was DONT_CARE would break on it.

This closed L6 at 0.005 ms/frame against a claimed 0.1-0.2 ms; see
`/tmp/perf-audit/l6_depth_elision.md`.

---

## F3. The host cannot bound the A-buffer's quad count.

`nblocks x rast_num_triangles` over-estimates the actual quad count by
**~3.5e7** (measured: 6,301/6,301 samples on old, 2,535/2,535 on Crossroads),
because the host does not know post-transform triangle area before the vertex
shader has run. The clip rectangle is not tighter: it measured **equal to the
whole framebuffer in every sample**.

**Consequences.**

1. The per-draw `bounded` fast path admits **zero** draws and no host-side
   rectangle changes that. Admission needs `blocks x tris <= 131,072`; the
   median rectangle would have to be 72x18 px on old and 54x2 px on Crossroads.
2. **`bound/actual`, not the wait, is the expensive quantity** in every sizing
   scheme this driver has tried. Three separate leads have died on it.
3. The same measurement kills the obvious refinement of the peel loop's
   per-pass framebuffer clear: there is no over-clear to remove.

See `/tmp/perf-audit/bounded_clip.md` §9.

---

## F4. Blended (A-buffer) episode segments ALREADY fan out across the eight side streams.

**"Fan out the A-buffer raster chain the way the opaque one was" is the status
quo, not a lead.**

* `cp_pass_append()` (`cp_renderer.c:9324-9334`) sends segment
  `k = cp->pass.nsegs % CP_PASS_STREAMS` to `cp->seg_streams[k]`, with its own
  queue set, gated on `cp->pass_gate`. **There is no flag on this path** —
  `CUDAVK_NO_OPAQUE_STREAMS` reverts the *opaque* fan-out only
  (`cp_opaque_side_streams()`, `:7587`).
* `cp_pass_streams_init()` says it outright (`:7536-7539`): the eight streams,
  the eight rasterizer queue sets and the gate event were factored out of
  `cp_pass_appendable()` because an opaque episode *"wants exactly the same
  eight streams ... and it never goes through the blended admission test that
  used to be the only place they were built."*
* So the 2026-08-26 commit `20611f5b131` extended an existing mechanism to a
  second kind of episode. It did not introduce fan-out.

**Consequences.**

1. `cp_rasterize_stage3_abuf` — the largest single kernel class in the driver at
   **2.484 ms/frame over 134.6 launches** — is *already* concurrent work.
   Proposing to spread it over streams proposes the current code.
2. **The width is done, and the survey's own number says so.** NCU measured
   `launch__waves_per_multiprocessor` at median **0.15**, max 0.60, **0 of 270
   launches reaching 1.0**, median grid 512 blocks, 18.5 µs per launch. One
   launch occupies 15% of a wave, so **about 6.7 concurrent launches fill the
   machine, and there are already 8 side streams plus the main one.** Raising
   `CP_PASS_STREAMS` cannot pay. `k = nsegs % 8` reuse costs queueing, not
   throughput, even though old-capture episodes average 12.42 segments.
3. Combined with `PERFORMANCE.md` §7, which closes the kernel-internal route
   with two whole-scheduler rewrites built and reverted: **stage 3 being the
   largest class is not an invitation.** What is left at this site is
   **duty cycle, not width** — the device is idle 26% of the frame and that
   idle is almost exactly the host's own issue time, because the driver blocks
   about seventeen times a frame and cannot issue while blocked.

---

## F5. There is no universal blocked-to-frame conversion, and one site converts at par.

**Measured by direct injection** — `CUDAVK_WAIT_SPIN_US` busy-waits after a
named wait returns, lengthening host time at that site without changing any
device work:

| site | add-slope Δframe/Δ(injected ms/frame) |
|---|---:|
| episode drain | **+1.0207**, linear over 0 → 1.976 ms/frame, residuals < 0.031 |
| peel checks | **−0.0262**, flat |

**Consequences.**

1. **The 11% conversion does not exist.** It was inferred from a patch that
   blocked +1.01 ms/frame and cost +0.110 ms of frame; direct injection shows
   that blocking is free at that site, so the +0.110 was the patch's
   *mechanism*, not its blocking.
2. **A conversion is a property of the site and cannot be carried between
   sites.** The two measured values are 0.00 and 1.02, a range that spans
   everything. Any lead sized with a borrowed conversion is unsized.
3. **The mechanism that predicts it is simple:** a drain empties the stream by
   definition, so the device is idle immediately after one and host delay lands
   on the critical path; a mid-loop check has device work still queued behind
   it. Use that to guess, then measure.
4. **Measuring it is nearly free** and needs no mechanism: a wait cannot be
   shortened without one, but host time can be *added* at that point for
   nothing, and the slope is the same derivative in the add direction. Scale
   the injection by `waits/frame` or the low-frequency sites land in the noise.
5. **The add direction is not proof of the recover direction.** It is
   demonstrated symmetric at peel, where both read ≈0. At a drain the two
   differ in principle — recovering wait time does not remove the device work
   the wait waited for — so a high slope is a necessary condition and not a
   win.

This retired R2 in `SESSION_HANDOFF.md` §5 and revised the S1d closure, which
had borrowed the 11%.

---

## F6. The general rule these facts share

F1–F4 are each a number that looked like waste and was structurally required,
or was already removed, or could not be computed, **or was already
implemented**. **F5 is different and worth naming separately: a number that was
real, but borrowed from the wrong site.** **Before costing a lead, map its unit onto the driver's own
unit** — API op onto kernel, slot onto lane, bound onto count, proposal onto
existing call site — and check whether the driver already does the thing, or
cannot.

Three times in one session a "wasted work" figure turned out to be
architecture, and a fourth time a sizing constant turned out to be a fact about
one site rather than about the driver. Each check cost minutes and the build it saved would have cost an
iteration. F4 is the sharpest case: it was queued as a lead and the disproof was
one `grep` for `seg_streams` in a function nobody had re-read.
