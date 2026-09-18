# Verify: Lead I — redundant clear and compaction volume

Verdict: **OVERSTATED.** The device-time number reproduces exactly (0.200 ms/frame of
2080x2080 visbuf clears in the heavy band, 100% exclusive, on a device-chain-bound serial
scope), but every step from that number to a frame-time claim is inflated: the pool is
heavy-band-only (the light band has one shadow batch per frame and nothing to remove), a
heavy-band-only 0.20 ms moves the headline whole-window median by 0.07-0.09 ms (simulated
on three pinned runs), the "77 full-target 1280x720 scans at 1.4 TB/s DRAM-bound" reading of
cp_fs_compact is wrong in both the size mix and the bound, the B200 "HBM3e" scaling does not
apply to an L2-absorbed fill, the 0.029 ms of A-buffer `counts` clears are not redundant,
and the cheap mechanism has to re-clear around the 1.74 discarding batches per frame. No
accounting — heavy-band device, heavy-band replay, whole-window mean, whole-window median —
puts the cheap mechanism at or above the 0.5 ms gate; the full per-scope design reaches
~0.4-0.6 on the heavy band only and is DEAD_ENDS 31 with the memsets added.

Read-only. Nothing built or run on the GPU. Scratch: this directory only.

## 1. What the trace actually shows (MEASURED, /tmp/postpin-rtx2/trace.sqlite)

Window: external submits [4898, 6300) = trace ns [9341207, 6464509572) = 701 frames
(frames 2449-3150, inside the heavy band). Trace-relative time = ts.txt value minus
1585329015665402 (TARGET_INFO_SESSION_START_TIME column 4), which puts submit 4898 at
9,341,207 ns, matching the arith report's window.

| quantity | value | SQL / method |
|---|---:|---|
| 34,611,200 B memsets (2080x2080 x 8 B) | 13,558 = **19.34/frame**, **0.2000 ms/frame** summed, mean 10.34 µs | `SELECT ... FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE bytes=34611200` in window |
| union-exclusive of those against every kernel/memcpy/memset | **0.2000 ms/frame = 100.0%** | interval merge of all other ops over each memset |
| scopes containing them | exactly **1 per frame** (701 runs in 701 frames, split at >2 ms gaps) | run grouping |
| clears per scope | bimodal: {6,10,12,13} and {22..28}; histogram top: 6 (85), 23 (82), 25 (64), 22 (57), 26 (55), 24 (52) | |
| 17,305,600 B memset (2080^2 x 4 B, the depth clear) | 1.00/frame | one depth clear per scope: `cp->depthbuf_cleared` latch |
| 8,652,800 B D8 memset value 0 | 1.74/frame | = ALIGN_POT(w*h,256)*2 for 2080^2 (cp_renderer.c:3986), the `discard_mask` (4010, 4032): **1.74 of the 19.34 shadow batches per frame have a discarding FS** |
| stream of every op in the scope | 13 (main) | side-stream ops inside the scope total 0.021 ms/frame |
| scope wall | 1.356 ms/frame mean (CUPTI host inflated), median run 1.55 ms | first memset to `cp_depth_attachment_store` |
| scope summed device busy | 0.992 ms/frame | |
| host position | **100% of 153,861 stream-13 ops in the scopes had their API call finished before the predecessor op ended**; the memset's own call ends 0.65-1.46 ms before its predecessor ends (p90..p10) | RUNTIME join by correlationId |
| device idle inside the scope | 0.000 ms/frame attributable to host lateness; 0.386 ms/frame front-end gaps | |
| next blocking sync after the scope | p50 2.77 ms later | SYNCHRONIZATION syncType 1/3 |

Per-batch chain in the scope (one dump, typical): `MS 34.6MB 10.4 µs -> HtoD 1.7 KB -> main(VS)
1-2 blocks 3-6 µs -> cp_clip_rast_fused 1-3 blocks 10-11 µs -> stage2 1-160 blocks 3-5 µs ->
stage3 2048 blocks 19-27 µs -> HtoD -> cp_fs_compact 4225 blocks 6.6-7.6 µs -> HtoD -> main(FS)
4096 blocks 3-7.5 µs -> cp_fs_writeback 2048 blocks 1.4-1.6 µs`. Per-class budget of the scope:

| class | n/frame | ms/frame | share of scope busy | mean µs |
|---|---:|---:|---:|---:|
| MS 34.6 MB (visbuf clear) | 19.34 | 0.200 | 20.2% | 10.34 |
| cp_clip_rast_fused | 20.22 | 0.144 | 14.6% | 7.14 |
| cp_fs_compact (4225 blocks) | 19.34 | 0.131 | 13.2% | 6.78 |
| cp_rasterize_stage3 | 19.88 | 0.122 | 12.3% | 6.13 |
| main (VS) | 22.31 | 0.116 | 11.7% | 5.19 |
| main (FS) | 19.34 | 0.070 | 7.0% | 3.60 |
| cp_rasterize_stage2 | 19.91 | 0.057 | 5.7% | 2.84 |
| HtoD copies | 61.0 | 0.052 | 5.3% | 0.85 |
| cp_fs_writeback | 19.34 | 0.030 | 3.0% | 1.56 |

So: the 2080x2080 target is the favorite2-style D16 shadow map (cp_renderer.c:6126-6139
names it: "favorite2's 2080x2080 D16 shadow map"), rendered in **one depth-only scope per
frame** with **19.3 direct batches per frame** in the heavy band. The VS is 1-2 blocks and the
clipper 1-3 blocks: tiny meshes, each paying a full-target clear, a full-target compaction
scan, a 4096-block FS and a 2048-block writeback.

Light band, for contrast (/tmp/postdead/trace.sqlite, frames 1391-1802): the shadow map is a
1024x1000 target (8,192,000 B visbuf clear at 3.14 µs, **1.00/frame**; 4,096,000 B depth
clears 2.00/frame; compact grid 1000 x 413 = 1/frame). **One shadow batch per frame: zero
redundant clears in the light band.** The 7,372,800 B (1280x720) clears there are 21.1/frame
at 2.59 µs = 0.055 ms/frame total.

## 2. Why 19 clears: the episode-break question has a different answer

The lead asks which state change ends an opaque episode in the shadow scope. **None does,
because there are no opaque episodes there.** `cp_opaque_appendable()` (cp_renderer.c:
7542-7551) refuses any batch with `fb->nr_cbufs != 1 || !fb->color`; a depth-only pass has no
colour attachment, so every shadow batch takes the direct path `cp_draw_execute_batch()` and
hits the per-batch clear at 5092-5094 (`if (!cp->pass.appending) cuMemsetD32Async(visbuf,
0xFFFFFFFF, w*h*2*fb_samples)`). The comment at 5080-5083 states the invariant the clear
protects: "The visibility buffer only ever holds this draw's triangles: its entries are
triangle indices into this draw's vertex arrays, so carrying it across draws would shade one
draw's pixels with another's geometry." The episode-start clear at 9359 is the *other* site
and fires once per opaque episode on the colour target (the 7.37 MB clears).

What ends a *batch* there (and so forces a clear) is a `cp_batch_key` mismatch
(cpvk_cmd.c:2970-2999 -> `cpvk_batch_can_join`; key fields cp_renderer.h:74-110: vs, fs,
index_resource, mode, index_size, start_instance, the pipeline `state` bytes, vertex
element counts/stride, UBO widths, sampler table). With 1-2 VS blocks per batch the shadow
draws are per-mesh; HYPOTHESIS: `index_resource` or vertex layout differs per mesh.
`CUDAVK_DEBUG_BATCHDIFF` (FLAGS.md:40) names the field; that is a one-run probe and it
doubles as a check on a different lead (if 19 draws could merge into one batch the clears
disappear by batching, not by visbuf sharing).

## 3. Precedent (DEAD_ENDS.md, PERFORMANCE.md §4, TODO.md)

- **Entry 31 (one-visbuf depth-only chain deferral, 0.170 ms/frame) is the closest
  precedent, not 7 or 36.** It is exactly "shade once per scope" for this scope: it censused
  40,184 structurally depth-only direct shade calls (19.3/frame over 2081.5 frames — the same
  population as the 19.34 shadow batches here), found 37,406 "exactly safe", and priced
  deleting *every* compact + FS + writeback in every safe run at 0.170 ms/frame whole-window
  mean. It **did not price the visbuf memset**, so the lead is right that a term was omitted.
  But its per-call figures (compact 5.1 µs, FS 3.0, wb 1.3) match this trace's shadow-scope
  means (6.8, 3.6, 1.6), so its arithmetic is consistent with the heavy band, and its retry
  clause is "another already-required mechanism provides the shared visbuf and resolve for
  free" — which is what a cheap clear-skip would have to be.
- **Entry 7** refutes *folding* a bulk clear into a kernel (row-by-row, +1.84 ms). It applies
  only to the variant "have compact/stage3 re-clear the entries it consumes"; it does not
  refute removing redundant clears. The lead's "why different" holds against 7.
- **Entry 36** omitted memsets from its displaced pool. Adding 0.15-0.25 to its midpoint
  0.409 gives 0.56-0.66 — the lead's "crosses the gate" claim. Two problems: (i) 36's pool
  is per-kernel union-exclusive over the heavy band; the memset term is additive only if
  Renderer 2 also rewrites the depth-only scope, which 36's design documents scope to opaque
  colour scopes with a fan-out surrender — the shadow scope has no fan-out (side streams
  carry 0.02 ms/frame in it), so the surrender term does not apply to it but neither does 36's
  displacement accounting; (ii) 36's *impossible* upper bound was 5.214 against the 5.0 goal;
  +0.2 lands at ~5.01. Crossing the build gate is not reaching the goal.
- **In-tree precedent for the mechanism:** the `cp->pass.appending` clear-skip at 5089-5093
  ("at a segment per blended batch this was most of the frame's memset traffic") and the
  opaque episode's running `cp->pass.next_prim` base (9182-9193, `abuf_prim_base` at 5195,
  `CP_PRIM_ID_LIMIT` 1<<30). Sharing a visbuf across batches with a global prim base is a
  shipped, tested mechanism — for colour targets.
- **PERFORMANCE.md §4 rule 2**: a fusion that deletes a large clear is not priced at the
  launch price. Consistent with quoting the clear's device time here. **Entry 22 (merge
  rule)**: the credit is collectable only where the launches were serial — they are: 100%
  exclusive, all on stream 13.
- **TODO 'Performance leads'** has no entry for this; the closure-audit report (§2.4) lists
  entry 31 at x1.25-1.45 replay ratio -> 0.21-0.30, "below 0.5 at any ratio".

## 4. Arithmetic

### 4.1 Heavy-band device time (MEASURED)

- Cheap mechanism M1 (skip the clear between consecutive order-free depth-only batches in
  one scope; running prim base; compact filters foreign ids): removable clears = 19.34 - 1
  (first of scope) - ~2 per discarding batch x 1.74 = **~14.9-15.8 clears/frame = 0.155-0.165
  ms/frame**; ceiling if the discard batches were somehow order-free too: 18.34 x 10.34 µs =
  **0.190 ms/frame**. Host API saved: ~15-18 x 1.07 µs (host-launch-audit §3.2) = 0.02 ms, not
  collectable in a scope whose host is 0.65-1.5 ms ahead.
- Full per-scope design M2 (= entry 31 + memsets): 18.34 x (10.34 + 6.78 + 3.60 + 1.56) µs +
  ~2 HtoD/batch x 0.85 = 0.190 + 0.124 + 0.066 + 0.029 + 0.031 = **0.44 ms/frame device**,
  minus one resolve per scope (~15 µs) and the entry-31 exclusions (2,778 of 40,184 calls
  unsafe; the 1.74 discarding batches/frame here). Front-end gaps of the ~5 removed ops per
  batch (~1.7 µs each, 0.386 ms/frame of gaps in the scope) add ~0.15 -> replay-style heavy
  frame impact **~0.5-0.6**, on the heavy band only.
- The 1280x720 clears (17.2/frame x 2.56 µs = 0.044): per-episode (9359) or per
  non-appending direct batch (discard, non-order-free depth) — sharing across them is not
  legal in general; collectable ≤ 0.02. The 3,686,400 B clears (18.5/frame, 0.029) are the
  A-buffer `counts`/`peel_next` arrays (9273, 6187, 6461): each blended episode's count phase
  needs zeroed counts; **not redundant**. The lead's pool "0.244 + 0.029" over-counts by ~0.05.

### 4.2 From device time to the frame metric (MEASURED simulation)

Three pinned runs, /tmp/pin-rtx-ab/favorite3-p1..3/ts.txt, frames 1391-3468 (2,078), frame =
every other submit:

| run | whole median | heavy 2150-3150 median / p10 / min | non-heavy median / p90 | share of frames above the median that are heavy |
|---|---:|---|---|---:|
| p1 | 6.637 | 8.123 / 6.861 / 3.995 | 4.605 / 6.662 | 89.1% |
| p2 | 6.693 | 8.157 / 6.930 / 3.980 | 4.591 / 6.712 | 89.3% |
| p3 | 6.676 | 8.160 / 6.840 / 4.026 | 4.508 / 6.675 | 89.6% |

Subtracting a constant from heavy-band frames only (light band unchanged, which is the
lead's shape — one shadow batch per light frame):

| heavy-only saving | whole-median move p1 / p2 / p3 |
|---:|---|
| 0.10 | -0.039 / -0.036 / -0.026 |
| **0.20** | **-0.074 / -0.088 / -0.088** |
| 0.30 | -0.115 / -0.120 / -0.134 |
| **0.44** | **-0.154 / -0.182 / -0.168** |
| 0.60 | -0.220 / -0.242 / -0.231 |
| 1.00 | -0.383 / -0.418 / -0.437 |

(A uniform 0.20 on every frame would move it -0.200.) The whole-window median sits at the
boundary between the bands: the ~10% of heavy frames near 6.6-6.9 ms are what move it, at
roughly 0.4 ms of median per 1 ms of heavy-band saving.

**Corrected ceilings, RTX:** M1 device 0.16-0.19 ms/frame heavy band = **0.06-0.09 ms on the
whole-window median**; M2 0.44 device / ~0.5-0.6 replay heavy = **0.15-0.24 on the whole-window
median**. On the whole-window *mean* (the basis DEAD_ENDS 30-35 used): M1 ≈ 0.19 x 1000/2078 =
0.09; M2 ≈ 0.21-0.29 — consistent with entry 31's 0.170 plus ~0.09 of memsets.

### 4.3 The compaction half of the lead (L4) is misdescribed

Compact grids in the heavy window: 900 blocks (1280x720) 32.4/frame; **4225 blocks (2080^2)
19.34/frame**; 158 blocks 20.7/frame; 15 blocks 3.0; 6528 (3072x2176) 1.0; 1000 (1024x1000)
1.0. Not "77 scans of the full 1280x720 visbuf": 32 are, 19 are 4.7x larger, 24 are small.
Scanned bytes: 239 MB (main) + 669 MB (shadow) + 53 MB. The shadow compacts read 34.6 MB in
6.78 µs = **5.1 TB/s effective, 2.8x DRAM peak**: the visbuf was just cleared and written and
sits in the 96 MB L2, so this scan is L2-resident, not "bandwidth-bound at 1.4 TB/s". The
same holds for the clear itself (34.6 MB in 10.34 µs = 3.35 TB/s). Consequences: the ncu
22-45% DRAM figure describes the 1280x720 compacts, and the B200 argument "memory-bound so
0.22x" does not transfer — an L2-absorbed fill/scan on the B200 is bounded by its L2 fabric at
its lower clock, not by HBM3e. B200 for the clears: **ESTIMATED 0.10-0.25 ms/frame heavy band**
(19.3 clears at 5-13 µs; no B200 memset duration is recorded anywhere — b200 report line 76
also records that no B200 band medians exist, so the whole-median conversion is unknown
there; if the band shape matches RTX, ~0.04-0.10). The lead's "0.05-0.10 B200 (HBM3e)" is an
unsupported DRAM-scaling assumption on an L2-bound operation.

### 4.4 Double counting

M1's 0.16-0.19 is not inside any prior closed pool (31 excluded memsets; 36 excluded them;
36's displaced pool and 31's kernels overlap each other, not this). M2 = 31 + memsets. The
closure-audit "delete memset -0.508 heavy" is all 135 memsets/frame including 29 small ones
with front-end gaps; the shadow clears are 0.200 of its 0.322 union.

## 5. Feasibility

**Could consecutive depth-only batches share the visbuf without a re-clear?** For order-free
batches, yes, with three changes, and the lead's worry ("a later episode's tri ids restart at
0 — what breaks?") is exactly the thing to fix:

1. Give direct depth-only batches a running prim base (today `rast_args.abuf_prim_base` is set
   only under `cp->pass.appending && cp->pass.opaque`, 5194-5195; stage 1 packs
   `PACK_VISBUF(key, abuf_prim_base + tri_id)`, cp_rasterize.cu:764-765; the id field is
   `~id` in 32 bits and `CP_PRIM_ID_LIMIT` is 1<<30, so 19 batches of ≤256k triangles fit).
2. Make the readers skip foreign ids. `cp_resolve_seg_range()` (kernels/cp_fs_interp.h:267-
   270) **returns true unconditionally when there are no seg ranges**, so today's compact would
   resolve a stale entry as a prim of the current batch and read the wrong vertex arrays
   (or out of bounds). The filter is a two-line range check in `cp_fs_compact`
   (cp_fs.cu:643-660, `prim = tris[t] - abuf_prim_base`) and in the classic interpolator
   (cp_fs.cu:181/223/323) and peel/census readers (cp_rasterize.cu:1474, 1929, 1944, 2361,
   2533). Skipped pixels become helper lanes (mask 0), which the writeback already drops.
3. Skip the clear at 5094 only when the previous batch in the same scope was depth-only,
   `cp_batch_order_free()` (7278-7299: no discard, depth test and write on, L/LE/G/GE) and had
   the same `depth_key_invert`; reset the base and force a clear at `cp_render_scope_begin`,
   at any non-order-free batch (the 1.74 discarding batches/frame — a discarded fragment that
   won the atomicMin would hide a later batch's fragment behind it, which passed the depthbuf
   test but loses the min: wrong depth), on any peel/retry, and on a sample-count change.

Why it is correct for the admitted set: the rasterizer depth-tests against `cp->depthbuf`
(cp_rasterize.cu:668-669, 2808-2810) *before* the atomicMin, and `cp_fs_writeback` is the
only writer of depthbuf (4287-4300). A later batch's fragment reaches the visbuf only if it
beat every earlier committed depth; the visbuf min then compares it with the earlier batch's
*uncommitted* winner at that pixel, which is the same test. Ties: the `~id` packing favours the
later id, so an equal-depth later fragment is re-shaded and re-writes the same depth —
harmless for depth-only, and the LEQUAL case wants it. Invariants (ARCHITECTURE §8): #9 is
the admission condition; #10 (helper lanes) is what makes the filter safe; #13 (ABI) is
untouched — no new argument slot, the prim base already exists; #20/#21 need a
`CUDAVK_NO_...` revert flag in the `cp_debug.c` registry plus `cp_debug_doc.py --check`.

Hidden work: none of the removed clears is on a side stream, so no gate/join changes; the
clear at 9359 (episode path) is unaffected. Blast radius: the classic direct path's core
invariant (5080-5083 comment) changes for one admitted class; the peel and retry paths must
keep an unconditional clear. Tests it must pass (TESTING.md): Gate 1 native suite incl. the
`CUDAVK_BATCH_MAX=1` bit-identity, Gate 2 18-sample sweep (shadow-map samples are the
relevant ones), Gate 3 sentinel frames on **both** captures (favorite2 is the one whose
shadow map first exposed the depth-only path, 6126-6139), Gate 4 device-side equivalence
with the flag-off arm; and the frame count/stdout hash check on every timing run.

**Cheaper decisive probe than building it:** the read-only one is done above and it is firm —
the device ceiling is 0.20 ms heavy band, the scope is device-chain-bound, and the whole-median
conversion is 0.4x. What remains uncertain is only whether the 19 batches are order-free
(HYPOTHESIS: a shadow pass is LESS with depth write, no blend) and why they do not merge:
one run of the heavy band with `CUDAVK_DEBUG_EPISODE=1` (prints `no-episode: orderfree=..`
per refused batch) and `CUDAVK_DEBUG_BATCHDIFF=1` answers both without writing driver code.
If that run shows the 19 draws differ only in `index_resource`, batching is the cheaper route
and this lead is subsumed. A wrong-pixels timing probe (skip the clear unconditionally) is
**not** safe here: stale entries resolve through `cp_resolve_seg_range()` to out-of-range prims.

Effort: probe = one flagged replay, an hour. M1 = ~60-80 lines (cp_renderer.c guard + running
base; range filter in cp_fs_compact and the classic/peel readers; a registry flag; FLAGS.md
regeneration) + all four gates on both captures: 1-2 days. Expected result: -0.15..-0.19 ms on
heavy-band frames, -0.06..-0.09 on the headline median — below the project's 0.5 gate on every
accounting it uses.

## 6. Must fix in the lead as written

1. The pool is heavy-band-only; the light band has one shadow batch per frame. Quote the
   whole-window median effect (0.4x of the heavy-band saving), not the heavy-band device time.
2. "77 scans of the full 1280x720 visbuf" -> 32 of 1280x720, 19 of 2080x2080, 24 small; the
   2080^2 scans and clears are L2-resident (5.1 and 3.35 TB/s effective), not DRAM-bound.
3. Drop the 0.029 ms of `counts`/`peel_next` clears from the redundant pool.
4. Replace the B200 HBM3e scaling with "unmeasured; L2-bound" (no B200 memset durations or
   band medians exist).
5. Name entry 31 as the precedent (same population, memsets omitted), not 7 or 36; the 19
   clears come from direct batches refused by `nr_cbufs != 1`, not from opaque-episode breaks.
6. Subtract the discarding batches (1.74/frame) and their re-clears from any clear-skip
   ceiling.

## 7. Labels

MEASURED: everything in §1, §4.2 tables, §4.3 grid census, the code-site facts in §2/§5.
ESTIMATED: M1/M2 per-frame arithmetic in §4.1, front-end-gap add-on, B200 range.
HYPOTHESIS: shadow batches are order-free (depth func LESS, writes on); the batch-key field
that breaks them; L2 residency as the reason for >DRAM-peak effective rates.
