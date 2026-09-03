# Four changes: the vertex lane, the clip hoist, and two cheap items

Landed together after the 2026-09-02 leads. Each has its own revert switch;
the combined A/B below turns all of them off in the control arm.

| change | switch | mechanism |
|---|---|---|
| vertex lane | `CUDAVK_NO_VS_LANE` | a direct batch's fused VS runs on a side lane while the previous batch's chain finishes on main |
| clip scratch hoist | `CUDAVK_NO_CLIP_ALLOC_HOIST` | the clip's scratch is allocated above the VS launch and the shade chain's argument blocks are prepared there, so both ride the copy that link already carries |
| record-only context scope | `CUDAVK_NO_CTX_SCOPE_TRIM` | ~900 push/pop pairs per heavy frame removed from 36 entry points that make no CUDA call |
| layered image copy | `CUDAVK_NO_LAYERED_COPY3D` | 32 per-layer 2 KB copies + 32 events become one `cuMemcpy3DAsync` + one event |

**Combined**, three-round alternating A/B, control = every switch reverted:

| capture | window | control | all on | delta |
|---|---|---:|---:|---:|
| favorite3 | relevant | 6.3309 | **5.9401** | **-0.3908** |
| favorite3 | heavy | — | — | **-0.6551** |
| favorite2 | relevant | 5.0923 | **5.0224** | **-0.0699** |
| favorite2 | heavy | — | — | -0.1226 |

Arms disjoint in every band. Individually measured: vertex lane -0.3194/-0.0475,
clip hoist -0.1337/-0.0326, cheap pair -0.0032/-0.0351. They are **sub-additive**
- the sum is -0.456/-0.115 against a measured -0.391/-0.070, so about 86% and
61% is collected. They overlap because three of the four shorten the same
main-stream chain.

## The vertex lane, and the review error it corrected

Only the fused VS moves. Clip and raster stage 1 stay on main because stage 1
reads the depth `cp_fs_writeback` commits, writes the visibility buffer
`cp_fs_compact` is reading, and appends to the queue set stages 2/3 drain.

The analysis note's own review claimed the VS output needed no rotating slots
because it is "a per-batch dscratch bump allocation". That is wrong:
`cp_draw_execute_batch` rewinds `dscratch.used = 0` at the top of every
non-episode batch, so batch k+1's VS output starts at the *same address* as
batch k's, which is safe only while one stream serialises them. The lane uses
two dedicated rotating slots instead.

Ordering is enforced by a checked mark rather than an assertion: a kernel
counter and a device-operation epoch are compared at three points, and any
enqueue the driver cannot account for refuses the lane and runs the VS on main.
Making that epoch unconditional also closed a latent PDL defect - an image copy
between a named predecessor and its successor had not been invalidating the
attribute. The lane fired 41,028 times and refused 31,041 on favorite3.

## Gates

79/79 suite (also under `CUDAVK_VS_LANE` and `CUDAVK_CTX_CHECK`), 18-sample
sweep identical to the accepted baseline, both captures' hashes and full
timestamp populations, 0 sentinel mismatches across 24 A/B runs, and the
20-run `multithreading` tolerance compare at 2085-2090 against a 2084-2089
baseline - that sample is not bit-deterministic, so it is the only gate that
can see the stream-ordering failure mode this class of change risks.

**Not measured on the B200** (host unreachable this session).

## Refuted alongside them

`cp_clip_rast_fused` launches of 200-400 us looked like per-thread serial work
worth 0.2-0.4 ms/frame. ncu says otherwise: a `grid=1` launch does **48 cycles
of work while 10,869 elapse** (1/170, one SM). Their wall time is the cost of
sharing the machine with seven other side streams, and the 1.1 ms/frame that
"summing the long ones" produced is summed concurrent duration - the thing this
project's union-exclusive rule exists to refuse. No change made.

Lead I (shadow-scope visibility clears) was refuted from the code: a depth-only
scope has `colorAttachmentCount 0`, so `cpvk_batch_structural` refuses every
draw in it and the 19 shadow "batches" are single draws that were never
batched. No key relaxation can merge them. See `notes/SHADOW_VISBUF_CLEARS.md`.
