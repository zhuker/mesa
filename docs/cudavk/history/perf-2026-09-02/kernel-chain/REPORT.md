# favorite3 kernel-chain structure census and approximate critical path

Read-only analysis. Sources: `src/cudavk/cp_renderer.c` (HEAD of `cudapipe-vk-native`),
`src/cudavk/kernels/*.cu`, `/tmp/postdead/trace.sqlite` (nsys export, commit
`f3caa35e4ce`, RTX 5090, CUPTI on, **pre host-pin readback**, i.e. the 3.6 MB
per-frame readback still went to managed memory), `/tmp/postdead/submit-ts.txt`.
Scripts and raw outputs are in this directory: `census.py/.out`, `q2.py/.out`,
`q3.py/.out`, `seq.py` + `frame_seq.txt` (one median frame, every device op),
`chains.py/.out`, `dag.py` + `dag0.out`/`dag2.out`, `gaps.py/.out`, `fix.py/.out`.

Labels: **MEASURED** = read from the trace or a document; **ESTIMATED** = my
arithmetic (shown); **HYPOTHESIS** = not measured.

## 0. Window, conventions, and two caveats that bound every number here

* Frame = two `vkQueueSubmit`; favorite3 real work starts at frame 1391 = external
  submit 2782 (WORKFLOW 4.0). Frame k here is external submits `[2782+2k, 2784+2k)`.
* The trace's kernel activity ends at 12.3405 s relative, so the usable window is
  **412 complete frames, absolute frames 1391-1802**, relative 9.2759 s to
  12.3370 s. **This is the light/moderate band, not the heavy band** (2150-3150);
  the heavy band is not in this trace. Median frame in the window, CUPTI on:
  **7.174 ms** (mean 7.430; deciles 4.76, 5.47, 5.78, 6.71, 6.95, 7.17, 7.45,
  7.67, 8.12, 8.72) [MEASURED, submit-ts.txt].
* Two things in this trace are stale relative to the shipping driver:
  1. the per-frame `cuMemcpy2DAsync` of 3,686,400 B to managed memory takes
     **614 µs mean** here and drags ~4.8 MiB/frame of UVM traffic each way
     (stream 0 rows, copyKind 11/12); `history/perf-2026-08-31-host-pin-readback.md`
     removed most of this (-0.929 ms/frame RTX). Every "memcpy8" figure below
     must be read with that in mind.
  2. CUPTI adds host cost per launch; the host-issue lag numbers are therefore
     pessimistic for the host, and the docs' rule (CLAUDE.md) that host-vs-device
     boundness comes from `cp_gpu_busy.sh` stands.
* Stream ids in the trace: **13 = `cp->main_stream`**, **14..21 = `cp->seg_streams[0..7]`**
  (`cp_pass_streams_init`, cp_renderer.c:7471), 7 = legacy stream carrying the
  synchronous `cuMemcpyDtoH_v2`/`cuMemcpyHtoD_v2`, 0 = UVM migrations (not a stream).

```sql
-- window and stream census (all queries clip to start in [9275862060, 12336970217))
SELECT streamId, count(*), sum("end"-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL
 WHERE start>=9275862060 AND start<12336970217 GROUP BY streamId;
```
Result: 13: 125,762 kernels / 950 ms; 14: 22,202 / 342; 15: 5,533 / 170; 16: 4,693 / 135;
17: 4,562 / 119; 18: 3,669 / 114; 19: 3,356 / 97; 20: 3,084 / 65; 21: 3,347 / 57.

Per frame (412): **427.5 kernels, 104.3 memsets, ~200 non-UVM memcpys (+240 UVM rows)**
[MEASURED]. Main stream: 305 kernels/frame mean (344 median), 549 device ops/frame
mean (600 median). Side streams: 122 kernels/frame.

## 1. (a) Kernel chains from the code

Grid rules are quoted from `cp_renderer.c` (line numbers at HEAD); "typical grid" is the
trace median for this capture (1280x720 main target, a 1024x1000 depth+colour target,
120x120-class post targets). "Reads from predecessor" is the true data dependency
from `kernels/*.cu`. S = support (counters, worklists, compaction, clears), W = real work.

### 1.1 Opaque direct batch (`cp_draw_execute_batch`, 4939-7279)

| # | op | grid rule (code) | typical grid | reads what predecessor wrote | stream | S/W |
|---|---|---|---|---|---|---|
| 0 | `cuMemsetD32Async(visbuf, ~0)` 7,372,800 B | once per draw unless appending (5094) | - | - | main | S |
| 0' | `cp_clear_depthbuf` memset 3,686,400 B | lazily once per scope (5098, 9379) | - | - | main | S |
| 0'' | `cuMemsetD8Async(vs_input)` + `cp_vertex_fetch` | only when the fused fetch declined; `(verts+255)/256 x 256` (5698) | 4608 thr | vertex/index buffers (app), seeds queue counters (fetch_fold) | main | W |
| 1 | generated VS `main` | `MIN2((total_verts+255)/256, 4096) x 256` (5916) | median 1536 thr (6 blocks); 21/frame are 1 block | vertex/index/UBO rows via the coalesced upload | main | W |
| 2 | `cuMemsetD32Async(clip_count)` 4 B | skipped under fetch_fold (6013) | - | - | main | S |
| 3 | `cp_clip_rast_fused` | `(num_triangles+63)/64 x 64` (6941) | median 37 blocks = 2368 thr | VS positions (`vs_out`), clip_count/queue counters; writes clipped buffer, `active_ids`, visbuf (small tris in place), `nontrivial` queue | main | W |
| 4 | `cp_rasterize_stage2` | `CLAMP((rast_tris+7)/8,1,512) x 256` (6988) | 512x256 = 131,072 thr | `nontrivial` queue + count (stage 1); writes visbuf (medium), `huge_tiles` queue, setup cache | main | W |
| 5 | `cp_rasterize_stage3` | fixed `2048 x 64` (6996; "stage 3 gets no such bound") | 131,072 thr | `huge_tiles` queue + count + setup cache (stage 2); writes visbuf | main | W |
| 6 | `cuMemsetD8Async(discard_mask)` | only if `fs->uses_discard` (4032) | 1,843,200 B | - | main | S |
| 7 | upload interp block (`memcpy1` ~784 B) | `cp_upload` (4175) | - | - | main | S |
| 8 | `cp_fs_compact` | `((w+1)/2*(h+1)/2 + 255)/256 x 256` (4212) | 900x256 = 230,400 thr | **visbuf** (stages 1-3), prim_refs/ranges; writes pixel_list, prim_list, coverage, counter (atomicAdd) | main | S |
| 9 | upload FS arg block (`memcpy1` ~752 B) | 3834 | - | - | main | S |
| 10 | generated FS `main` | `MIN2((max_pixels+255)/256, 4096) x 256`, `max_pixels = ALIGN(w*h,256)*2` (3874) | 4096x256 = 1,048,576 thr for 1280x720 (7200 blocks capped) | counter, prim_list/pixel_list (compact), VS varyings (interpolates in-shader), textures | main | W |
| 11 | `cp_fs_writeback` | `MIN2((num_pixels+255)/256, 2048) x 256` (4315), PDL secondary of FS | 2048x256 | counter (compact), fs_out/discard_mask (FS), pixel_list, coverage; writes colour + `depthbuf` (+reject/resolved) | main | S/W |

Trace medians for one batch on this capture (`census.out`, `chains.out` §E): VS 5.8-10 µs,
fused 8-13, s2 2.5-5, s3 1.5-6, compact 5.2, FS 7.9, wb 1.7; **chain wall 49-70 µs for
small/medium batches; 455 µs for the 1.36/frame "large" batches (VS 175 µs, fused 122 µs)**.
Same-stream gaps inside the chain: s2 and s3 0.26 µs median (PDL), compact 4.35 µs after s3
(0.90 after the upload copy that sits between), FS 0.93 after its upload, wb 0.26.

### 1.2 Opaque episode (`cp_opaque_append` 9339, `cp_opaque_finish` 7850)

Segment 0 runs steps 1-5 on **main**. Segment k>=1: `cp_pass_gate_stream(k-1)` (7629) =
upload flush on main + `cuEventRecord(pass_gate, main)` + `cuStreamWaitEvent(side k-1)`;
then steps 1-5 on side stream `(k-1) % 8`, with its own queue set. Finish: `cp_pass_join`
(7588) = one `cuEventRecord` per used side stream + `cuStreamWaitEvent(main)`; then per
shading group (same VS, FS, UBO rows, mode): steps 7-11 on main with a `cp_seg_range`
table upload. **Because the gate is recorded on main *after* segment 0's launches, side
segments cannot start until segment 0's stage 3 has completed** (trace: side VS starts
2-10 µs after the main s3 ends, in every fan-out inspected; `frame_seq.txt` lines
2953-2999 and 4018-4047).

### 1.3 Blended segment and pass-episode finish (`cp_pass_append` 9230, `cp_pass_finish` 8691)

| phase | ops | grid rule | typical | dependency | stream | S/W |
|---|---|---|---|---|---|---|
| episode start | memset `counts` (w*h*4) + memset `counters` (312 B) + `cuEventRecord(pass_gate)` | 9272-9289 | 3.69 MB, 312 B | - | main | S |
| segment k | `cuStreamWaitEvent(side k%8, gate)`; upload rows; VS; `cp_clip_rast_fused_abuf` `(T+63)/64 x 64`; `stage2_abuf` `CLAMP((T'+7)/8,1,512) x 256`; `stage3_abuf` `CLAMP(T'*8,512,2048) x 64` (6510-6550) | median fused 1 block, s2 1024 thr, s3 32,768 thr | VS out; writes per-pixel `counts` (atomicAdd) and appends `pixel<<32|prim` records through `rec_cursor` | side k%8 (**segment 0 included**) | W |
| join | `cp_pass_join(nsegs)` | 8735 | | | main waits | - |
| scan | `cp_abuf_scan_reduce` grid from `cp_abuf_scan_tiling(n)` (1881: ept = ceil(n/512^2), grid = ceil(n/(512*ept))) x 512; `cp_abuf_scan_finish` same grid, PDL secondary, also clears the 3.7 MB `cursor` | 450x512 for 921,600 px | `counts` (segments) -> `offsets`, total; | main | S |
| fill | `cp_abuf_fill_recs` `2048 x 256` (8759) | | `recs`, `rec_cursor`, `offsets`, `cursor` -> `frags` | main | S |
| sort | `cp_abuf_sort_short` `MIN2((n+255)/256,4096) x 256` (8816) then `cp_abuf_sort` `4096 x 256` (8829) (+`cp_abuf_worklist` when short-sort is off) | 3600x256, 4096x256 | `frags`, `offsets`, `counts`; builds `list`/`clist` worklists | main | W/S |
| quads | `cp_abuf_quad_count_all` (scan grid x 512) -> `cp_abuf_scan_finish` (PDL) -> `cp_abuf_quad_fill_all` `ceil(nblocks/256) x 256` (2262-2287) | 450x512, 450x512, 900x256 | `frags`/`offsets`/`counts` -> `blk_counts` -> `blk_offsets`, quad total (`bsum3`) -> `quad_prim`, `quad_mask`, `shade_slot` | main | S |
| bucket | `cp_abuf_seg_count` `MIN2((quad_capacity+255)/256,1024) x 256` (8877) | 1024x256 | `quad_prim`, seg prim bases -> `seg_counts` | main | S |
| **drain** | `cp_sync_timed` = `cuStreamSynchronize(main)` + `cuMemcpyDtoH(counters, 24-56 B)` (8888-8894) | | host reads total, overflow flags, quads, covered, per-seg quad counts | host | - |
| scatter | memset `seg_cursor`, upload seg bases, `cp_abuf_seg_scatter` `(quads+255)/256 x 256` (8988) | 1-21 blocks | `quad_prim`, bases -> `grouped`, `quad_dense` | main | S |
| shade | `cp_pass_broadcast` gate; per group g on side g%8: memset counter (or `cp_abuf_prepare_shade_count` 1 thread), upload interp+args, FS `main` over `ALIGN(4*quads,256)` slots (4630-4670, 3874) | 1-66 blocks | `grouped` quad list, VS varyings | side | W |
| composite | join; upload descs; `cp_abuf_composite` `(covered+255)/256 x 256` (9152) | 1-66 blocks | `clist`, `counts`, `offsets`, `shade_slot`, `fs_out` per seg -> colour | main | W |

Trace (`chains.out` §C, `fix.out`): **8.93 blended episodes/frame** (median 10); tail
(scan_reduce .. composite) **11 kernels median**, **tail wall 66.3 µs median per episode,
683 µs/frame; kernel time in the tail 304 µs/frame; 379 µs/frame of gaps** (launch gaps
1.7-2.5 µs between 1-3 µs kernels, the drain round trip, the side-stream FS). 82% of the
episodes whose segments could be attributed have **one segment**; median segment-phase wall
35.8 µs (single-seg), 51.6 (multi-seg). Composite runs on 6.21/frame of the 8.93 -> **30%
of drains return `quads == 0`** [MEASURED, 2557 composites / 3680 scan_reduce].

### 1.4 Depth-only pass (shadow map)

Identical to 1.1 with `color_data == NULL`; the FS still runs (it may discard) and
`cp_fs_writeback` is launched because it is the only writer of `cp->depthbuf` (4287-4300);
`cp_depth_attachment_store` `((w+15)/16,(h+15)/16,samples) x (16,16)` (2656) runs at scope
end (1.00/frame, 2.66 µs). The 1024x1000 target shows as memsets of 8,192,000 B (visbuf)
and 4,096,000 B (depth) and compact grid 1000. Entry 31 already sized deferral at 0.170.

### 1.5 Fullscreen post-process draw (1-2 triangles covering the target)

Every raster stage runs. VS is one block. In `cp_clip_rast_fused` the triangle's bbox area
exceeds `CP_MEDIUM_THRESHOLD` (256 px, cp_rast_types.h:1218) so it is only *appended* to the
nontrivial queue. `stage2` (one warp; lane 0 does `setup_triangle` and decomposes the bbox
into `(tri, tile)` pairs with `CP_TILE_SIZE = 64` (cp_rast_types.h:1248): 240 pairs for
1280x720, 4 for 120x120). `stage3` (2048 x 64 launched, 240 or 4 blocks with work) walks
each tile with 64 threads: `bw*bh = 4096` positions / 64 threads = **64 serial iterations
per thread** of three `edge_function`s + a 64-bit `atomicMin` (cp_rasterize.cu:1290-1370).
Then compact/FS/writeback over the whole target.

Trace, 4.17 "fullscreen-like" chains/frame (VS grid 1, fused grid 1, s3 > 8 µs;
`chains.out` §E): **VS 2.6, fused 9.8, s2 2.4, s3 19.3, compact 3.0, FS 3.8, wb 1.4 µs
medians; chain wall 52.3 µs.** Stage 3 costs **19.3 µs whether the target is 1280x720
(240 tiles) or 120x120 (4 tiles)** because its duration is one block's 64 serial
iterations, not the tile count. Per frame: raster part (fused+s2+s3) **130 µs**, whole
chains **211 µs** wall [MEASURED]. Note 4 of the 4.17 are also failed-append re-executions
(section 3, L1).

### 1.6 Clears, copies, blits

* `LOAD_OP_CLEAR` -> `cp_clear_kernel` / `cp_clear_depth_kernel` `((w+15)/16,(h+15)/16) x (16,16)` (9816-9857): 2.00 + 1.00 per frame, 2.4-2.8 µs each. `cuMemsetD32Async(visbuf)` 21.1/frame x 2.59 µs, depth/`counts` memsets (3,686,400 B) 22.3/frame x 1.55 µs, `cp_abuf` arrays 29.5 MB 1.54/frame x 9.1 µs. Total memset device time **0.14 ms/frame**.
* Texture cache: `cp_cache_convert` 4.55/frame (1.15 µs) plus device-to-array copies (kind 7) 3.1/frame >= 1 MB, 10 MB/frame, 7 µs each (`cpvk_texture_cache.c:392`).
* App copies: `vkCmdCopyBuffer` -> ~45 `cuMemcpyDtoDAsync`/frame of 2-64 KB, **0.6 µs device / ~2 µs host issue each**, issued back to back at frame start (`frame_seq.txt` 2104-2296); `vkCmdCopyImageToBuffer` -> one `cuMemcpy2DAsync` 3,686,400 B, **614 µs mean here (pre host-pin)**; `vkCmdDispatch` -> one generated `main` with 8x8 blocks, grid 5x3 (960 threads), **68 µs**, 1/frame, on the main stream after the last composite.
* `cp_blit_linear`/`cp_resolve_samples`: none in the window.

## 2. (b) Trace census per kernel

`census.py`, window 412 frames. `main` is split by same-stream neighbours as WORKFLOW.md
advises (`prev_k`/`next_k` on the same stream). Gaps are same-stream, kernel-to-previous-kernel
(`gapK`) and to the previous op of any kind (`gapAny`).

| kernel | launch/f | med µs | p90 µs | sum ms/f | med threads | med gapK µs | med gapAny µs | main/side |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cp_clip_rast_fused | 24.10 | 29.8 | 139.8 | **1.376** | 2,368 | 2.5 | 2.4 | 4940/4991 |
| main [VS, before clip/rast] | 48.48 | 7.5 | 59.5 | 0.986 | 1,536 | 79.9 | 2.4 | 6376/13599 |
| cp_rasterize_stage3 | 33.92 | 7.3 | 39.6 | 0.492 | 131,072 | 0.26 | 0.26 | 8982/4991 |
| main [FS direct, after fs_compact] | 33.92 | 6.4 | 21.8 | 0.443 | 1,048,576 | 3.4 | 0.9 | 13973/0 |
| cp_rasterize_stage2 | 33.92 | 4.9 | 19.2 | 0.287 | 131,072 | 0.26 | 0.26 | 8982/4991 |
| cp_rasterize_stage3_abuf | 15.73 | 10.4 | 27.2 | 0.215 | 32,768 | 0.42 | 0.42 | 1436/5045 |
| cp_vertex_fetch | 7.51 | 8.8 | 63.9 | 0.195 | 4,608 | 71.1 | 1.1 | 1854/1242 |
| cp_fs_compact | 33.92 | 5.2 | 9.5 | 0.187 | 230,400 | 4.35 | 0.90 | 13973/0 |
| cp_clip_rast_fused_abuf | 15.73 | 4.3 | 19.8 | 0.130 | 64 | 2.8 | 2.0 | 1436/5045 |
| cp_abuf_scan_finish | 17.86 | 3.8 | 8.0 | 0.102 | 230,400 | 0.22 | 0.22 | 7360/0 |
| main [compute dispatch, 8x8 blocks] | 1.00 | 68.0 | 79.9 | 0.070 | 960 | 2026 | 0.9 | 411/0 |
| cp_fs_writeback | 33.92 | 1.5 | 2.5 | 0.064 | 524,288 | 0.26 | 0.26 | 13973/0 |
| cp_clip_triangles | 8.65 | 8.1 | 13.2 | 0.062 | 64 | 1.5 | 1.3 | **0/3563** |
| cp_rasterize_stage1 (unfused, re-passes) | 9.81 | 6.9 | 8.2 | 0.061 | 13,824 | 7.3 | 1.1 | 4042/0 |
| cp_rasterize_stage2_abuf | 15.73 | 2.9 | 5.5 | 0.050 | 1,024 | 0.42 | 0.42 | 1436/5045 |
| main [FS abuf, side streams] | 4.42 | 8.8 | 11.1 | 0.036 | 3,328 | 80.0 | 2.0 | 0/1822 |
| cp_cache_convert | 4.55 | 1.2 | 3.2 | 0.026 | 8,960 | 1.0 | 1.0 | 1831/45 |
| cp_abuf_composite | 6.21 | 3.4 | 6.1 | 0.023 | 768 | 12.7 | 1.0 | 2557/0 |
| cp_abuf_quad_count_all | 8.93 | 2.3 | 3.6 | 0.022 | 230,400 | 1.8 | 1.8 | 3680/0 |
| cp_abuf_sort | 8.93 | 2.3 | 2.5 | 0.021 | 1,048,576 | 1.9 | 1.9 | 3680/0 |
| cp_abuf_scan_reduce | 8.93 | 2.2 | 2.4 | 0.020 | 230,400 | 42.6 | 34.9 | 3680/0 |
| cp_abuf_worklist | 6.97 | 2.2 | 2.4 | 0.016 | 921,600 | 1.3 | 1.3 | 2872/0 |
| cp_abuf_quad_fill_all | 8.93 | 1.7 | 3.4 | 0.015 | 230,400 | 0.13 | 0.13 | 3680/0 |
| cp_abuf_sort_short | 5.45 | 2.2 | 2.4 | 0.013 | 921,600 | 2.7 | 2.7 | 2244/0 |
| cp_peel_advance | 5.35 | 2.3 | 2.5 | 0.012 | 921,600 | 2.2 | 2.2 | 2203/0 |
| main [FS abuf, main] | 2.77 | 4.3 | 4.8 | 0.011 | 768 | 16.1 | 2.1 | 1141/0 |
| cp_abuf_fill_recs | 8.93 | 1.4 | 1.5 | 0.011 | 524,288 | 1.9 | 1.8 | 3680/0 |
| cp_abuf_seg_count | 5.45 | 0.9 | 1.1 | 0.005 | 262,144 | 2.4 | 0.8 | 2244/0 |
| cp_clear_kernel | 2.00 | 2.6 | 2.8 | 0.005 | 976,896 | 1.6 | 1.5 | 824/0 |
| cp_abuf_seg_scatter | 3.44 | 1.4 | 1.7 | 0.005 | 512 | 14.5 | 1.0 | 1416/0 |
| cp_depth_attachment_store | 1.00 | 2.7 | 2.7 | 0.003 | 921,600 | 11.2 | 11.2 | 412/0 |
| cp_clear_depth_kernel | 1.00 | 2.4 | 2.4 | 0.002 | 921,600 | 1.8 | 1.8 | 412/0 |

Readings:
* The 80 µs `gapK` on the VS row is the batch boundary (the visbuf memset and uploads sit in it;
  `gapAny` is 2.4 µs). The 42.6 µs before `scan_reduce` is the join wait for the blended
  segments. The 4.35 µs before `fs_compact` is stage3 -> upload copy -> compact (entry 26).
* **Every kernel has a duration floor of 1.3-2.7 µs** (`census.out` "min/p5"): e.g.
  `cp_abuf_seg_count` min 0.90, `fs_writeback` 1.28, `stage3` 1.34, `scan_reduce` 1.86,
  FS `main` 2.72. The A-buffer support kernels run at their floor (median within 0.3 µs of p5).

### 2.1 Latency-shell evidence (duration vs thread count)

| threads | n/frame | med µs | p10 | p90 | share > 5 µs | dominant names |
|---|---:|---:|---:|---:|---:|---|
| < 1,024 | 76.6 | 3.9 | 2.1 | 12.8 | 0.41 | main 11143, clip_rast_fused_abuf 5473, clip_rast_fused 3723 |
| 1,024 - 4,096 | 25.2 | 9.7 | 2.6 | 35.5 | 0.64 | main, clip_rast_fused, fs_compact(small targets) |
| 4,096 - 20,000 | 35.4 | 6.9 | 2.4 | 59.3 | 0.61 | main, clip_rast_fused, stage1, vertex_fetch |
| 20,000 - 100,000 | 32.7 | 7.9 | 1.4 | 49.3 | 0.73 | stage3_abuf, main, stage1, clip_rast_fused |
| 100,000 - 500,000 | 158.6 | 4.5 | 1.3 | 22.6 | 0.46 | stage3, fs_compact, stage2, scan_finish |
| 500,000 - 2,000,000 | 98.8 | 2.3 | 1.4 | 11.9 | 0.20 | main(FS), fs_writeback, sort, worklist |

**Latency shells (< 20k threads and > 5 µs): 69.1 launches/frame, 1.672 ms/frame summed**
[MEASURED]: `main` 30.0/f, `cp_clip_rast_fused` 17.4/f, `clip_rast_fused_abuf` 6.6/f,
`clip_triangles` 5.8/f, `vertex_fetch` 5.0/f. Duration is anti-correlated with size in the
mid range: 100k-500k-thread launches have a *lower* median (4.5 µs) than 1k-20k ones (7-10 µs).

`cp_clip_rast_fused` by grid (`q2.out`): grid <= 4 blocks (<= 256 input triangles):
0.71/f but **median 79 µs, p90 227, max 392 µs**; grid 17-64: median 17.6; 65-256: 42;
257-1024: 102; > 1024 (median 1,876 blocks = 120k tris): 125 µs. **Duration is not
proportional to triangle count**: 464 launches > 200 µs in the window (1.1/frame, 5.79/f
above 100 µs summing 0.871 ms/f), 91% of them on side streams (opaque fan-out segments),
with grids of 4-160 blocks. Example (`q3.out`): stream 16, grid 4, 392 µs, followed by
s2 7 µs and s3 51 µs. HYPOTHESIS for the mechanism (not measured): a few threads each
rasterize a "small" (bbox <= 16 px) or clip-crossing triangle serially with the 5 KB
clipper stack in local memory, so one block's longest thread sets the launch duration.

Long `cp_vertex_fetch` launches (> 100 µs, 0.17/f, up to 955 µs for one 256-thread block)
**all 70 overlap a UVM migration row** (`q3.out`); long generated-VS launches do not (3 of
489). The fetch stalls are page faults on app-written host-visible vertex data and are
pre-host-pin artefacts only if that change also covers vertex buffers (it does not:
it pins only transfer-dst-only buffers), so they may persist [UNKNOWN].

### 2.2 Per-stream busy per frame (median over 412 frames, ms)

| quantity | median | mean | p10 | p90 |
|---|---:|---:|---:|---:|
| frame | 7.174 | 7.430 | 5.468 | 8.723 |
| union of all device ops (kernels+memcpy+memset, incl. UVM) | 4.548 | 4.510 | 3.534 | 5.425 |
| union excluding UVM rows | 4.436 | 4.390 | 3.424 | 5.289 |
| union of kernels only | 3.403 | 3.337 | 2.596 | 3.972 |
| **main stream 13 busy (all ops)** | **3.396** | 3.300 | 2.194 | 4.249 |
| main stream kernels only | 2.448 | 2.305 | 1.408 | 3.019 |
| **side streams 14-21 union** | **1.055** | 1.084 | 0.970 | 1.260 |
| **main ∩ side (exact, `fix.py`)** | **0.000** | 0.000 | - | max 0.000 |
| main idle while side busy (= side union) | 1.055 | 1.084 | 0.970 | 1.260 |
| device idle, no op anywhere (excl. UVM) | 2.641 | 3.040 | 1.972 | 3.596 |
| UVM rows (stream 0) | 0.373 | 0.390 | | |
| side 14 / 15 / 16 / 17 / 18 / 19 / 20 / 21 | 0.896 / 0.417 / 0.354 / 0.206 / 0.280 / 0.295 / 0.132 / 0.137 | | | |

**Finding: the side streams never overlap the main stream — the intersection is exactly
zero in every frame.** The fan-outs overlap segments with each other only; whenever a side
stream is busy the main stream is waiting at a join or at the gate. Side stream 14 carries
0.90 ms because blended segment 0 (`cp_pass_append` k = nsegs % 8 = 0) and every failed
append (L1) land on it.

## 3. (c) Approximate critical path

### 3.1 Method

`dag.py`: nodes = every kernel/memcpy/memset on streams 7, 13-21 with a runtime correlation
(374,965 issue events); edges = same-stream order; `cuStreamWaitEvent` (CUPTI sync type 2)
joined to the latest `cuEventRecord` of the same `eventId` in host-issue order (edge from the
last op issued on the recording stream to the next op issued on the waiting stream);
`cuStreamSynchronize` / `cuCtxSynchronize` / `cuEventSynchronize` (types 3/4/1) as host
barriers (the next op issued on every stream depends on the sync's target). Node weight =
device duration; longest path per frame with predecessors outside the frame contributing
zero (per-frame reset); optional lambda per dependent edge. This is a lower bound on the
frame given the dependencies (no host, no launch latency at lambda = 0).

### 3.2 Result (median over 412 frames)

| quantity | lambda = 0 | lambda = 2.0 µs/edge |
|---|---:|---:|
| frame (ms) | 7.174 | 7.174 |
| **device-work critical path (ms)** | **3.400** | 4.583 |
| cp / frame | 0.469 | 0.628 |
| **nodes on the path** | **598** (kernels 343) | 599 |
| of which on the main stream | 595 | 596 |
| ops in frame (all streams) | 786 | |

**The critical path is the main stream, node for node**: 595 of 598 path nodes are
main-stream ops and the path's device time (3.400) equals the main stream's busy union
(3.396). Side-stream work is never on it because main waits for every side burst.
Dependent boundaries on the path: **~598/frame (343 kernels, ~100 HtoD uploads,
~92 memsets, ~47 D2D copies)**.

Path composition (mean per frame, `dag0.out`): memcpy8 (D2D) 46.7 nodes / **0.768 ms**
(0.62 of it the stale managed readback), FS `main` 36.8 / 0.643, `cp_clip_rast_fused`
12.0 / 0.274, `stage3` 21.8 / 0.234, `fs_compact` 33.9 / 0.187, `vertex_fetch` 4.5 /
0.164, `stage2` 21.8 / 0.158, VS `main` 17.4 / 0.140, memsets 92.2 / 0.127, `scan_finish`
17.9 / 0.102, HtoD uploads 99.4 / 0.086, `fs_writeback` 33.9 / 0.064, `stage1` 9.8 / 0.061.

### 3.3 Where the other 3.77 ms of the median frame is (`gaps.out`, main-stream gaps)

| main-stream gap class | count/frame | ms/frame (median) |
|---|---:|---:|
| all gaps | 600 | 3.697 |
| side streams busy during the gap | - | **1.055** |
| nothing running anywhere ("pure idle") | - | **2.651** |
| host blocked in a sync during the gap | - | 1.220 |
| gaps < 3 µs (launch/dependency latency) | 526 | 0.635 (1.2 µs each) |
| gaps 3-20 µs | 44 | 0.315 |
| gaps 20-200 µs | 21 | 1.011 |
| gaps >= 200 µs | 3 | 1.858 |

Largest gaps (>150 µs, 3.37/frame): `stage3 -> memcpy1` 1.72/f x 337 µs (the opaque
fan-out join; side busy 329 of the 337); `memcpy8 -> memcpy8` 1.00/f x **1,136 µs with the
host blocked only 2 µs of it** = the frame boundary: the app reads the frame back, records
the next one and the driver plans, with nothing queued (TODO item 6); `memset -> memset`
0.37/f x 243 µs.

Host issue lag (`chains.out` §F): kernel start minus launch-API return, median **110 µs on
main, 197 µs on side streams; only 3% of kernels start within 3 µs of their launch**. In
this traced run the device queue is ~100-200 µs deep in steady state, i.e. the streaming
phases are device-paced and a device op removed from the main stream shortens the frame
~1:1 there; the frame boundary (1.1 ms) and the 11.4 `cuStreamSynchronize` round trips per
frame (1.27 ms blocked, mean 111 µs) are where the host is the limit. (CUPTI inflates the
host side, so this understates how far ahead the un-profiled host runs.)

Median-frame anatomy (`frame_seq.txt`, frame 1646, 7.17 ms): 0-1.0 ms readback copy
(965 µs) + fence wait; 1.0-2.1 host gap (1.13 ms); 2.1-2.3 ~50 app D2D copies at 2 µs
issue each + 62 KB sync HtoD; 2.3 compute dispatch 68 µs; 2.4-2.9 four single-draw
scopes, each a failed append (side VS+clip, join) then a direct chain; 2.9-4.7 main pass:
seg-0 chain 100 µs -> 4-segment fan-out 160 µs -> 3 shade groups; large batch 455 µs;
seg-0 chain 139 µs -> 8-segment fan-out 345 µs -> 10 shade groups; one alpha-tested batch
= 8 passes x ~38 µs; 4.7-6.5 nine blended episodes (~120 µs wall each for ~35 µs of
kernels) and one 6-pass peel draw; 6.5-7.1 depth store, 4 fullscreen-like draws (52 µs
chains, each preceded by a failed-append excursion), one 8-segment blended episode
(UI), composite; readback.

## 4. (d) Chain shortenings not already in DEAD_ENDS.md

Ranked by (dependent boundaries removed x occurrences/frame), then priced. Every ceiling is
for **this trace's band (frames 1391-1802) on RTX 5090** and is ESTIMATED from MEASURED
components; the heavy band and the B200 were not traced here (B200 scaling given as a
guess at the measured 10.79/6.68 = 1.6x frame ratio, since the handoff says B200 device
time is clock-bound — treat as HYPOTHESIS).

### L1. Stop admitting non-A-buffer draws to pass episodes (the failed-append back-out) — ~190 boundaries/frame

**Mechanism (code).** `cpvk_batch_eligible` (cpvk_cmd.c:2790-2811) classifies every draw
that is not `cpvk_pipeline_order_free()` as *blended* — that includes draws with blending
**off** but depth-write off / depth-test off / a non-monotone depth func, and every draw
whose FS uses `discard`. `cp_batch_flush_defer_why` (cp_renderer.c:9505) then calls
`cp_pass_appendable` (7409), which checks A-buffer *global* state only. `cp_pass_append`
(9300-9313) gates side stream `nsegs % 8`, runs `cp_draw_execute_batch` there: VS, uploads,
the lazy depth clear, then at 6172 `peel` is false (blend disabled, or `retry`), so `abuf`
stays false and the append **backs out at 6414-6425**: `cp_flush_pending_clip` launches a
standalone `cp_clip_triangles` (whose output nobody reads), `append_failed = true`; back in
`cp_pass_append` (9319-9333) the driver **joins all 8 side streams** (8 event records + 8
stream waits, main waits for the excursion), finishes the open episode, and **re-executes
the batch classically on main from the vertex stage**. cpvk_cmd.c:2712-2720 names this
"a back-out penalty" but only as a `CUDAVK_UNSAFE_FORCE_OPAQUE` artefact; the trace shows
it firing in the default configuration.

**MEASURED (`chains.out` §A).** `cp_clip_triangles` runs **8.65/frame, 3,563 of 3,563 on
side streams, 3,563 of 3,563 matched to a main-stream re-execution** with an identical VS
grid. Per frame (median): side excursion wall **222 µs**, **main-stream idle inside the
excursion-to-re-execution window 313 µs** (mean 310, p90 342), re-executed VS **39 µs**.
Also 108 `cuStreamWaitEvent` on main per frame, ~69 of them from these joins.

**Boundaries removed per occurrence:** 2 kernels (side VS, clip_triangles), ~3 small ops
(gate wait, depth clear, uploads), 16 event/wait API calls, 1 join wait, plus the host
re-planning of the batch. ~22 x 8.65 ≈ 190/frame.

**Ceiling.** The excursion window is dead time on the critical path (main idles at the
join, the device is the bottleneck there: lag 110 µs): ESTIMATED **0.25-0.31 ms/frame RTX**
(= measured main idle 0.31, less the ~5 µs/occurrence the re-executed chain would still
need to start). B200 HYPOTHESIS 0.4-0.5. Heavy band: the population is ~1 per single-draw
post/UI scope, so the count should be similar (UNKNOWN, not traced).

**Closest DEAD_ENDS.** None covers it. Same class as the `CUDAVK_ABUF_MIN_TRIS` fix
described at cp_renderer.c:9239-9250 ("an append that backs out there is re-executed whole,
so the vertex work happens twice"), which moved the triangle-count decision ahead of the
vertex work; this moves the remaining four decisions (blend off / discard, `depth_writemask`,
`attachment_samples != 1`, `nr_cbufs != 1`, `color_encoding < 0`) into
`cp_pass_appendable`/`cpvk_batch_eligible`. Not entry 8 (flush placement), not 23 (hoisting
tables). **Why different from the peel path**: the draws still take exactly the path they
take today after the back-out (classic direct or peel); only the excursion disappears.

**Cheapest probe.** Count the back-outs: `CUDAVK_DEBUG_EPISODE=1` prints `episode-cut:
append failed` per occurrence; or count `cp_clip_triangles` launches per frame (all are
back-outs in this capture). Fix is ~20 lines; the gate is one stdout hash + 18 sentinels.

### L2. Let opaque segment 0 overlap segments 1..n (record the gate before segment 0) — 2 fan-outs/frame, 0.24 ms

**Mechanism (code).** `cp_opaque_append` (9339-9430): segment 0 runs on main; for k >= 1
`cp_pass_gate_stream(k-1)` (7629-7642) records `pass_gate` on main **after** segment 0's
kernels were issued, so side segment k waits for segment 0's stage 3. The comment at
7605-7627 gives the reason: the segment's own uniform rows are still owed in the upload
ring and `cp_stream_set` sends the owed span on the stream it leaves (main). The visibility
buffer is `atomicMin`-resolved, so segments are order-free (ARCHITECTURE 2.3, 4.3); the
only true dependencies of segment k are the episode's clears and its own upload span.

**MEASURED (`chains.out` §B).** 2.00 fan-outs/frame, 12.1 side segments/frame (5 per
multi-segment burst median); **seg-0 chain wall 242 µs/frame** (median; 218 mean) runs
strictly before the **side burst (551 µs/frame)**; side VS starts 2-10 µs after main s3
ends. Burst wall (240 µs) equals the longest segment (239 µs), i.e. segments already overlap
each other fully.

**Ceiling.** With segment 0 concurrent, episode raster time = max(seg0, burst) instead of
seg0 + burst: gain = **min(seg0, burst) = 242 µs/frame median (218 mean)** [MEASURED as an
arithmetic ceiling]; device-paced region so ESTIMATED **0.20-0.24 ms/frame RTX**. B200
HYPOTHESIS ~0.35. Boundaries removed: 1 serial gate dependency per fan-out (4/frame), no
launches.

**Closest DEAD_ENDS.** 32 (segment-0 *shading* overlap — speculative shading of segment 0
before later segments raster; refuted at 0.036 useful). This is raster-only, no speculation,
nothing killed. 20/22 (merges — this *adds* concurrency). `b3f716bd5f5` moved segment 0
from a side stream to main to save one hand-off (32 µs/episode, PERFORMANCE 4); this keeps
it on main and adds no hand-off — it only changes where the gate is recorded and which
stream carries segment k's upload copy.

**Risk / design constraint.** The gate must still cover the episode clears (9359-9380) and
segment k's rows. Proposal: flush segment k's owed span **on the side stream** after
`cp_stream_set` (the coalesced-flush watermark is global, later main flushes cover only
later bytes; every main-stream reader of side-carried bytes is behind `cp_pass_join`), and
record the gate once at episode start after the clears, exactly as `cp_pass_append` does.
The flush-generation retire events are recorded on main after joins and still cover the
side copies. This needs a correctness argument reviewed against the 7605 comment (the
`multithreading` sample lost models when the gate did not cover the rows).

**Cheapest probe.** Timing-only flag that records the gate before segment 0's first launch
(leaving the upload on main), 3-round alternating A/B, hash + 18 sentinels; if the hash
moves, the upload-span ordering is the reason and the side-stream flush is required.

### L3. Discard-retry early exit — 7 re-passes/frame, ~77 boundaries/frame, 0.15-0.19 ms

**Mechanism (code).** `retry = fs->uses_discard && cp->reject && cp->resolved && color_data`
(6111); `passes = retry ? CP_DISCARD_LAYERS (8) : ...` (6179); the pass loop (6860-7090)
has a convergence check **only for `peel`** (`cp_peel_advance` + `cp_sync_timed` with a
doubling interval, 7076-7090); the retry loop always runs all 8 passes. Each re-pass is
11 device ops: memset visbuf (7.37 MB), memset queue counters, `stage1` (unfused), s2, s3,
memset discard_mask (1.84 MB), interp upload, compact, FS-arg upload, FS, writeback.

**MEASURED (`chains.out` §D, `frame_seq.txt` 2607-2880).** Unfused `cp_rasterize_stage1`
re-passes not followed by `peel_advance`: **7.0/frame median (5.4 mean), 221 µs/frame wall
(174 mean)** — one alpha-tested batch per frame doing 8 passes of ~32-38 µs. In the passes
shown, FS runs 2.8 µs (its floor) and s1/s2/s3 durations are identical pass to pass.
Whether passes 2-8 shade anything is NOT in the trace [UNKNOWN].

**Ceiling.** If the batch settles by pass 2: 6 passes x ~32 µs = ~190 µs/frame of main-stream
device time, minus one host round trip (a sync after pass 1-2, ~20-50 µs exposed):
ESTIMATED **0.13-0.19 ms/frame RTX**; boundaries removed 6 x 11 = 66/frame + 1 sync added.

**Closest DEAD_ENDS / docs.** PERFORMANCE 6.1 "peel checks — CLOSED 0.00" measured that
*adding host time at the peel-check site is free* and that the peel *check* cannot be
removed for gain; it says nothing about the retry loop, which has no check at all. The
rejected `CUDAVK_PEEL_PREDICATE` patch (+0.110 from overshoot passes) is the warning: the
retry check must be a real stop, not a prediction. Entry 19 (device-side early exit of
degenerate launches is not a saving) applies to a device-only variant: the passes must not
be *issued*, so a host read (or a later host-side decision) is required.

**Cheapest probe.** `CUDAVK_DEBUG_DISCARD=1` (exists, FLAGS.md: "report covered and
discarded fragment counts per pass; syncs") on ~20 frames: if pass >= 3 reports zero
discards, the lead is live; if not, it dies at zero cost.

### L4. Fullscreen-draw raster shortcut / wider stage-3 blocks — 4.17/frame, 0.06-0.12 ms

**MEASURED.** 4.17 fullscreen-like chains/frame; fused+s2+s3 = 31.5 µs median each,
130 µs/frame; s3 alone 19.3 µs = 78 µs/frame, invariant in target size (64 serial
iterations per thread, section 1.5).

**Options.** (i) Analytic fast path: when the fused kernel finds a single triangle whose
clipped bbox covers the clip rectangle with depth test off (post passes have no
depth attachment), set a device flag; `cp_fs_compact` reads it and emits every quad for
that triangle without reading the visbuf; s2/s3 become empty shells or are not issued
(host knows the draw is a 1-2-triangle non-indexed draw, but not that it covers the
target — the device decides). Saves fused's queue work + s2 + s3 + the visbuf memset:
ESTIMATED **<= 0.12 ms/frame** (130 µs minus a ~3 µs flag kernel per draw). (ii) 256-thread
stage-3 blocks (16 serial iterations instead of 64): ESTIMATED s3 19.3 -> ~6 µs on the
4.17 fullscreen chains = **~0.055 ms/frame**, plus an unknown share of the other 7.05/f s3
launches > 25 µs.

**Closest DEAD_ENDS.** 19 (degenerate s3 launches: not issuing needs a readback — here the
host does not need the tile count, only the device flag, and option (i) keeps the launches),
10/24 ("only more warps per item or more items per launch can help" — (ii) is exactly more
warps per item, never measured for `cp_rasterize_stage3` direct), 36 (renderer 2, refuted on
arithmetic — this is a 7-kernel special case, not a renderer). PERFORMANCE 6.0's "bigger
blocks are not the lever" was a synthetic wide kernel at 1,000 items, not a 4-240-item s3.
Both options are below the 0.5 gate on this band.

### L5. Fuse the serial blended tail (9 support kernels -> 2-3) — 54 launches/frame, 0.04-0.11 ms

**MEASURED.** 8.93 episodes/frame x 11 tail kernels on main, strictly serial (A-buffer
support overlap factor 1.000 in `/tmp/residual-gpu-audit/trace_partition_report.md`);
tail wall 683 µs/frame vs 304 µs of kernels; inter-kernel gaps in the tail 1.7-2.5 µs
between 1-3 µs kernels (`frame_seq.txt` 4945-4982), so the launch front end is **not**
hidden here.

**Ceiling.** 6 launches/episode removable (scan_finish+fill_recs; sort_short+sort; count_all
+scan_finish+fill_all via one cooperative launch) x 8.93 = 54/frame: at the documented
removal price 0.78-0.81 µs -> **0.042-0.044 ms/frame**; at the measured serial-tail gap of
~2.0 µs (the price the documents reserve for an *exposed* launch, which these are) ->
**<= 0.11 ms/frame**. Entry 25 priced the same population for device-side chaining at
0.035-0.037; iteration 28 item 4 already fused this chain once (+0.22). The merge rule (22)
permits it (serial), but the size does not justify it.

### Not proposed, recorded for the design agents

* Blended segment 0 on side stream 14 even for single-segment episodes (82% of episodes):
  the exposed hand-off is small here — gate gap 3.3 µs median, join gap 2.4 µs
  (`gaps.out`) — ~0.05 ms/frame, not the 32 µs/episode frame-time price PERFORMANCE 4
  quotes for the opaque case.
* 30% of blended drains return `quads == 0` (2557 composites / 3680 scan_reduce); documented
  as "not a lead" in PERFORMANCE 5.2 (the drain is taken for the overflow flags).
* `cuCtxSynchronize` 2.04/frame, 782 µs mean (scratch reclaim, `cp_scratch_begin` 683):
  entry 29 (below noise) — the device stays busy under it (`frame_seq.txt` 4143).
* The frame-boundary host gap (1.14 ms/frame with the host not blocked) is app + driver
  planning time with nothing queued; TODO item 6; not a kernel-chain item.

## 5. Numbers table (all per frame unless noted; window frames 1391-1802, RTX, CUPTI on, pre host-pin)

| name | value | unit | source |
|---|---|---|---|
| frames in window | 412 | frames | submit-ts.txt, external 2782-3606 |
| median frame | 7.174 | ms | submit-ts.txt (CUPTI on) |
| kernels / memsets / non-UVM memcpys | 427.5 / 104.3 / ~200 | per frame | census.py |
| main-stream ops (median) | 600 | per frame | census.py |
| main-stream busy | 3.396 | ms | census.py |
| side-stream union | 1.055 | ms | census.py |
| main ∩ side | 0.000 | ms | fix.py |
| device idle excl. UVM | 2.641 | ms | census.py |
| DAG critical path (lambda 0 / 2 µs) | 3.400 / 4.583 | ms | dag.py |
| path nodes (kernels) | 598 (343) | count | dag.py |
| latency shells (<20k thr, >5 µs) | 69.1 / 1.672 | count / ms | census.py |
| host issue lag main / side (median) | 110 / 197 | µs | chains.py F |
| kernels starting <3 µs after launch | 3 | % | chains.py F |
| cuStreamSynchronize | 11.42 / 1.272 | count / ms blocked | q2.py |
| cuCtxSynchronize | 2.04 / 1.597 | count / ms blocked | q2.py |
| cuEventSynchronize | 2.04 / 0.656 | count / ms blocked | q2.py |
| cuStreamWaitEvent on main / side | 108 / 45 | count | q2.py |
| failed appends (cp_clip_triangles on side) | 8.65 | count | chains.py A |
| failed-append side wall / main idle / re-VS | 222 / 313 / 39 | µs | chains.py A |
| opaque fan-outs / side segments | 2.00 / 12.1 | count | chains.py B |
| seg-0 chain / side burst / min() | 242 / 551 / 242 | µs | chains.py B |
| blended episodes / tail kernels | 8.93 / 11 | count | chains.py C |
| blended tail wall / kernels / gaps | 683 / 304 / 379 | µs | chains.py C |
| single-segment blended share | 82 | % | fix.py |
| composite / scan_reduce | 6.21 / 8.93 | count | census.py |
| discard re-passes / wall | 7.0 / 221 | count / µs | chains.py D |
| peel re-passes / wall | 6.0 / 239 | count / µs | chains.py D |
| fullscreen-like chains / raster / wall | 4.17 / 130 / 211 | count / µs / µs | chains.py E |
| fullscreen s3 duration | 19.3 | µs | chains.py E |
| readback copy 3,686,400 B (stale) | 614 | µs | q2.py |
| frame-boundary host gap | 1,136 | µs | gaps.py |

## 6. Unknowns

1. The heavy band (frames 2150-3150) is not in this trace; all per-frame structure counts
   (8.65 back-outs, 2 fan-outs, 8.93 blended episodes, 7 retry passes, 4.17 fullscreen) are
   light/moderate-band values.
2. The shipping driver pins the readback buffer; the memcpy8/UVM figures and the
   composition of the 1.14 ms frame-boundary gap are stale. Re-trace before pricing anything
   that touches the frame boundary.
3. Whether the discard re-passes 2-8 shade anything (L3) — one `CUDAVK_DEBUG_DISCARD=1` run.
4. Why `cp_clip_rast_fused` takes 200-400 µs with 4-160 blocks (per-thread serial work is a
   hypothesis; an NCU pass on those launches would settle it).
5. B200 values for every lead (only the 1.6x frame ratio is known).
6. Whether the long `cp_vertex_fetch` UVM stalls (0.074 ms/frame here) survive host-pin
   (vertex buffers are not transfer-dst-only, so probably yes).
