# Open questions and known gaps

Everything here is unfinished on purpose or unfinished by accident. It is kept
in one place so a fresh look does not have to reconstruct it from commit
messages. Read `DEAD_ENDS.md` first if you are about to optimise something --
this file is what is still open, that file is what has already been closed.

Numbers are as at HEAD, old capture 15.74 ms/frame, Crossroads 5.87 ms/frame.

## Correctness, ordered by how much they matter

1. **The direct path's one-fragment-per-pixel-per-launch invariant is false.**
   Measured: 6,343,098 second claimants of a pixel inside a single launch on the
   old capture. Nothing is known to be broken by it, but it was assumed true
   while designing a fragment-writeback fusion, and any future work that
   assumes it will be wrong the same way. See `DEAD_ENDS.md`, parked item.
2. **`bindless_image_store` has no bounds check.** A latent memory-safety hole
   rather than a rendering bug.
3. **The A-buffer verifier keeps two historical internal mismatch cases** that
   have never been explained, even though external images match.
4. **Stage 2 and stage 3 do not interpolate depth to the same bits.**
5. **`texture3d` drifts over an animation** -- two differing pixels at frame 0.
6. **`multithreading` has no diagnosis.** It differs from the reference by a
   small amount and nobody has found out why.
7. **`gltfscenerendering` is a standing exception** in the 60-frame comparison
   against NVIDIA. It predates all recent work: the fused and reverted builds
   produce the same numbers for it, so it is not caused by anything recent.
8. **`renderheadless` is missing from the sweep** because it drives its own
   frames, so it is never compared. It is compared by hand instead, which means
   in practice it is compared rarely.

## Vulkan surface not implemented

The driver advertises Vulkan 1.1. The Gallium driver that was removed advertised
1.4 with 198 extensions, because it borrowed lavapipe's frontend; that surface
was never verified against the CUDA backend. See `GALLIUM_RETIREMENT.md` for the
full delta. The gaps that come up most:

- multiple render targets, layered rendering, input attachments
- timeline semaphores, multiple queues
- real occlusion and pipeline-statistics queries
- full stencil state
- vertex attribute divisors above one -- the kernel handles an arbitrary
  divisor, but `VK_EXT_vertex_attribute_divisor` is not implemented, so nothing
  can select one through the API
- no line rasterization
- sample shading is per fragment only
- non-device-0 CUDA device selection

`docs/cudavk/history/HANDOFF.md` has the longer list with the reasoning.

## Performance leads that are open, not closed

Ranked by the evidence behind them, not by size. `PERFORMANCE.md` has the
measurements these come from.

1. **The host waits about 17 times a frame and is blocked about 12.44 ms doing
   it.** Device idle is 4.16 ms, almost exactly host issue time, so the driver
   is a ping-pong rather than a pipeline. This is the largest remaining item by
   a wide margin and the hardest. Two of its three sites survived contact with
   evidence; the third is closed. The design and the addendum that cut it down
   are in `/tmp/perf16/iter29-sync/`, and the two cheap probes that should be
   run before anything is built are described there: the ratio of the A-buffer
   bound to its actual count, and how much host work exists to overlap with a
   wait. Neither has been run.
2. **Fuse the fragment writeback into the fragment shader.** Parked with a
   working mechanism and a wrong result; about 0.04 ms and a known next step.
   See `DEAD_ENDS.md`.
3. **Five CUDA-side items from a research note**, none of which need a toolkit
   upgrade: `CU_JIT_SPLIT_COMPILE` (cold JIT 536 ms to 186 ms, measured),
   raising `CUDA_CACHE_MAXSIZE`, the `enable_smem_spilling` pragma, reopening
   CUDA graphs on 12.8 specifically, and `griddepcontrol`, which is used zero
   times today. `notes/CUDA13_UPGRADE.md`.
4. **The opaque sort-middle tiling prototype** in `notes/OPAQUE_TILING_PROTOTYPE.md`
   exists behind flags and was never taken to an accepted result.
5. **`pbribl` regresses by about 0.03 ms** with vertex-fetch fusion enabled and
   nobody knows why. Bounded and deliberately accepted; the cost is in fused
   execution on that workload, not in the machinery around it.

## Housekeeping

- The two-capture timing harness lives in `/tmp/perf16/`, outside the
  repository. It should be moved into `src/cudavk/tests/` so it survives a
  reboot and can be reviewed like everything else.
- The build directory is still called `build-cudapipe` in every document and
  script. Harmless, but wrong now.
- `nir_intrinsic_load_const_buf_base_addr_cudapipe` keeps the old name on
  purpose: it lives in Mesa-common files, where a fork-local rename only widens
  the diff against upstream. Rename it only if the fork stops tracking upstream.
- `notes/VIDEO_ENCODE_SURVEY.md` is a survey of what the tree contains, not a
  plan. Nothing in cudavk implements video encode.
