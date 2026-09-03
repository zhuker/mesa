# The vertex shader on a side lane (CUDAVK_VS_LANE)

Lead F of `notes/PERF_ANALYSIS_2026-09-02.md`, built behind an opt-in flag and
**not yet measured**. The mechanism, the ordering argument and what the flag
costs when it fires are here; the A/B is not, and until it exists this is a
change that has been reasoned about and compiled, nothing more.

## What it does

A direct batch runs its whole chain on the main stream, so batch k+1's vertex
shader cannot start until batch k's last fragment kernel has finished --
although it reads nothing batch k writes. The trace says that pool is real:
46.18 main-stream vertex launches per heavy frame, of which 0.18 are issued to
an idle stream, the rest adding **0.4367 ms/frame** of device time behind a
predecessor (`history/perf-2026-09-02/verify2-F/REVIEW.md` §2). With
`CUDAVK_VS_LANE=1` that one launch is issued on `seg_streams[7]` instead,
joined back to the main stream before the clip. Nothing else moves.

## The dependency set, which is the whole job

Batch k+1 may **not** run these ahead of batch k's shade chain, and none of
them does:

| stage | why it may not |
|---|---|
| `cp_clip_rast_fused` = clip + raster stage 1 | stage 1 reads `cp->depthbuf`, which batch k's `cp_fs_writeback` commits; writes the single `cp->visbuf` that batch k's `cp_fs_compact` reads; appends into `cur_qset`, which batch k's stages 2 and 3 drain; reads `cp->reject`, which batch k's writeback fills |
| raster stages 2, 3 | same queue set, same visibility buffer |
| the shade chain | it is the thing being overlapped |

The vertex shader touches none of them. Three things it does touch are made
disjoint rather than argued away:

1. **Its output buffer.** `cp->dscratch` rewinds to zero at every batch, so
   batch k+1's allocations alias batch k's live buffers; on one stream that is
   safe only because the stream serialises. The lane's output comes from two
   dedicated buffers (`cp->vslane.slot[]`) used by alternate lane batches.
   The verifier's "no rotating slots are needed, they exist by construction"
   is wrong on this point -- the rewind at `cp_draw_execute_batch()` is what
   makes it wrong.
2. **Its argument rows in the upload arena.** `cpvk_execute_draw_cmd()`
   flushes the previous batch *before* `cpvk_prepare_draw()` reserves anything
   for this one, so at the switch the rows are still owed and
   `cp_stream_set_carry()` sends them on the lane, in front of the shader that
   reads them -- the same order `cp_opaque_append()` uses for a segment. A
   flush that had already sent them on the main stream would be a hazard; it
   moves the epoch, and the mark below refuses the lane instead.
3. **The queue counters.** Under `fetch_fold` the fused shader zeroes
   `cur_qset.counts`, which batch k's stages 2 and 3 are still reading. A lane
   batch does not fold: it takes the `CUDAVK_NO_FETCH_FOLD` arm for itself,
   where the raster pass clears the three words on the main stream, in front
   of its own stage 1 and behind batch k's drain. That is the entire seed
   fallback -- no new kernel, no new clear site, one `bool` in one condition.

## How the ordering is enforced

`cp->vslane.gate[]` is a CUDA event recorded on the main stream at the top of
**every** batch, before any of that batch's work. The lane waits on the mark
the *previous* batch recorded. Everything the main stream had issued before
batch k -- copies, clears, dispatches, semaphore waits, other batches -- is
therefore complete before the lane starts, and only batch k's own chain is
overlapped.

That holds only while nothing else was enqueued between the mark and the
launch, and the driver could not say so by inspection: the calls are spread
over five files and one of them is issued by `cp_launch()` itself. So it is
checked. `cp->launches` counts kernels and `cp_devop_epoch` (new,
`cp_devop.h`) counts every other enqueue; a batch compares the pair at three
points and refuses the lane on any movement it cannot name:

- between the previous batch's last operation and this batch's mark: exact
  equality, which is what catches a copy, a dispatch, a barrier's work, a
  semaphore wait or a whole other batch;
- between the mark and the two clears at the top of the batch: the allowance
  is exactly those two clears, and the shader reads neither buffer;
- between the clears and the launch itself: exact equality again, because the
  argument block in between may have wrapped the upload ring.

The epoch used to be armed by `CUDAVK_PDL` and is now always counted -- one
relaxed increment next to a call that costs about a microsecond -- because a
correctness check cannot depend on a debug flag. Four enqueues the
interception did not cover (`cuMemcpy2DAsync`, `cuMemcpy3DAsync`,
`cuMemcpyDtoDAsync`, `cuLaunchHostFunc`) now move it, as do the texture
cache's rebuild and `cpvk_sync_gpu_wait()`'s semaphore wait. That also fixes a
latent PDL defect: an image copy between a named predecessor and its successor
did not refuse the attribute.

**The same mark is what makes two output slots enough.** The lane shader of
batch j waits on the mark at the top of batch j-1, which sits behind the whole
chain of batch j-2 -- the last batch to have used the slot j is about to
overwrite.

**Two things the mark cannot say**, because they are about what the
overlapped batch *wrote* rather than about when it ran, are handled
separately: a lane batch must be in the same render scope as the batch it
overlaps (so that batch's colour attachment is not something this draw may
legally sample), and that batch's shaders must not have written memory of
their own (an SSBO or a storage image, which a barrier inside a render pass
would otherwise order and this lane would step over).

**Every path out of `cp_draw_execute_batch()` has joined the lane**, so
nothing outside the function can see it. That is what keeps the arena
generations, `cp_scratch_reset()`, the submit completion events and the
frame's fences correct without a single change: they all reason about the main
stream, and the main stream waits for the lane before the clip.

## What it costs when it fires

- **The seed fallback**: one 3-word `cuMemsetD32Async` per lane batch.
  `CUDAVK_NO_FETCH_FOLD=1` prices it from above by doing it for *every* batch:
  +0.0796 ms/frame whole and +0.1355 heavy on favorite3, +0.0578 / +0.0779 on
  favorite2 (owner's 3-round A/B). A lane batch is a strict subset, so this is
  an upper bound and not the figure.
- **The event pair**: one record and one wait per lane batch, plus one record
  per batch for the mark, at roughly 1-1.5 µs of host time each.
- **Memory**: two vertex-output buffers, each grown to the largest lane batch
  seen. They come off `dscratch`'s peak rather than adding to it.
- **The eight pass side streams and their queue sets are created** on the
  first armed batch even for an application that never opens an episode.

## Limits, stated rather than discovered later

- **Fused vertex execution only.** The classic path launches
  `cp_vertex_fetch` and clears the packed input buffer before the shader, so
  its lane would carry four more device outputs across two allocation-failure
  returns. The fused shader writes exactly one buffer, which is what makes the
  isolation argument a sentence rather than a list. It is also the default.
- **Direct batches only.** A pass-episode segment already runs on a side
  stream with its own queue set and its own gate.
- **Excluded**: `CUDAVK_NO_UPLOAD_COALESCE` (there is no owed span to carry,
  and the rows were copied on the main stream as they were written),
  `CUDAVK_DEBUG_TIME` (its intervals span one stream by construction) and
  `CUDAVK_DEBUG_VFETCH`.
- **The mark is exact for one recording thread.** Two threads submitting on
  one context would need the batch machinery to be thread-safe first, which it
  is not.

## Gates run

Build, `cp_debug_doc.py --check`, `cp_no_getenv.py`, `cp_launch_audit.py`,
`git diff --check`. No GPU work: the replays, the sample sweep and the A/B
belong to whoever measures this, and `CUDAVK_PLAN_STATS=1` reports how many
batches took the lane and how many were refused -- a run where it never fired
looks exactly like a run without the flag.
