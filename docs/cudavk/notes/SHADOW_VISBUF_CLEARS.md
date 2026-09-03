# The shadow scope's visibility-buffer clears: no batch key splits them

Read-only. Nothing was built, run or measured for this note; it answers open
question 2 of `PERF_ANALYSIS_2026-09-02.md` ("which key field splits the 19
shadow-map batches per heavy frame?") from the code, and corrects the premise
that question rests on.

## The answer: none. They were never batches.

`cpvk_batch_structural()` (`cpvk_cmd.c:2416`) refuses a draw outright when its
scope has no colour attachment:

```c
if (!scope->fb.nr_cbufs || !scope->fb.color || !cp->visbuf || !cp->depthbuf)
   return false;
```

`vkCmdBeginRendering` fills `fb.nr_cbufs` from `colorAttachmentCount`
(`cpvk_cmd.c:1630`) and only assigns `fb.color` inside
`if (cat && cat->imageView)` (`:1643`). favorite2's 2080x2080 D16 shadow map is
rendered in a depth-only scope, so both are zero there and **every draw in that
scope fails the structural gate**. `cpvk_batch_can_join()` returns false at its
first test with the break note `"draw not batchable"`, the draw takes
`cp->plan.direct_draws++` and runs alone through `cp_draw_execute_batch()` with
`ndraws = 1`, and each of those pays the per-execution visibility clear at
`cp_renderer.c:5131`.

So the 19.34 clears per heavy frame measured in
`history/perf-2026-09-02/verify2-clears/REVIEW.md` §1 are 19.34 *single draws*,
not batches broken by a key mismatch. `cpvk_draws_mergeable()` is never reached
for them, `CUDAVK_DEBUG_BATCHDIFF=1` would name no field, and **there is no key
relaxation that merges them**. That review's §2 attributed the split to a
`cp_batch_key` mismatch and named `index_resource` or the vertex layout as the
hypothesis; it had spotted the colour condition in `cp_opaque_appendable()` but
not the identical one in `cpvk_batch_structural()` one layer above.

Two independent facts agree. DEAD_ENDS 31 censused the same population as
**direct** shade calls -- 40,184 structurally depth-only ones, 19.3 per frame --
and the merge-key campaign of 2026-08-30 had already taken vertex buffers, the
index buffer, the vertex offset, the instance count and the push block out of
the key, so a shadow pass drawing per-mesh geometry through one pipeline would
have little left to split on.

## What a merge would actually cost

Because the draws cannot batch, "fewer clears" is not a front-end change at
all. It is the M1 design in `verify2-clears` §5: give direct depth-only draws a
running primitive base, teach every visibility reader to skip foreign ids, and
skip the clear only between consecutive order-free depth-only draws in one
scope. That is device code in `cp_fs.cu`, `cp_rasterize.cu` and
`cp_fs_interp.h` as well as host code, plus re-clear rules for scope entry, for
the 1.74 discarding draws per frame, for peel/retry and for a sample-count
change. Its own review prices it at 0.16-0.19 ms/frame of heavy-band device
time and **0.06-0.09 on the whole-window median**.

Skipping the clear without that machinery is not a cheap approximation of it,
it is wrong: `cp_resolve_seg_range()` (`kernels/cp_fs_interp.h:267-270`)
returns true unconditionally when a draw has no segment ranges, which every
direct draw is, so a stale visibility entry from the previous draw resolves as
a primitive of the current one and indexes its vertex arrays out of range. The
invariant is stated at `cp_renderer.c:5117-5120`: the visibility buffer holds
triangle indices into *this draw's* arrays.

## Conclusion

No change was made, and no flag was added. The cheap item this lead was placed
in does not exist: the split is structural, not a key field, and the real
mechanism is DEAD_ENDS 31's population with the memsets added -- worth
0.06-0.09 on the headline median, needing kernel-side changes and all four
correctness gates on both captures.

**Retry if** someone wants M1 on its own terms: the design and its refusal set
are written out in `history/perf-2026-09-02/verify2-clears/REVIEW.md` §5, and
the first step is not a probe of the batch key but the running primitive base.
