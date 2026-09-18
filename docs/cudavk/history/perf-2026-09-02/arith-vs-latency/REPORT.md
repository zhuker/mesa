# Is the remaining device time arithmetic or latency?

Read-only analysis, 2026-09-02. Nothing was built or run on the GPU. Sources:
docs/cudavk/{PERF_HANDOFF,PERFORMANCE,DEAD_ENDS,TODO,ARCHITECTURE}.md,
docs/cudavk/notes/{RESIDUAL_AUDITS_2026-08-31,SHADER_CONTENT_CENSUS_RESULT}.md,
docs/cudavk/history/perf-2026-08-27/{wide_bench,tiling_ncu}.md,
docs/cudavk/history/{SM120,perf-2026-08-30-post-dead-scope-profile,perf-2026-08-31-host-pin-readback}.md,
kernel source under src/cudavk/kernels/, the nsys exports /tmp/postdead/trace.sqlite
(commit f3caa35e4ce, pre host-pin) and /tmp/postpin-rtx2/trace.sqlite (2026-08-31 19:03,
post host-pin, nsys --delay=14), the ncu CSVs /tmp/ncu-rtx/v2.csv (RTX 5090) and
/tmp/ncu-b200-v2.csv (B200, the same 45 launches), the census outputs /tmp/hiz-f3-8/*,
/tmp/vertex-reuse-cost.json, /tmp/fav3_shaderstats.log. Analysis script and raw outputs are
beside this file (trace_analysis.py, postdead/, postpin/).

Labels: MEASURED = read from a document, a trace or a counter file; ESTIMATED = my
arithmetic, shown; HYPOTHESIS = an interpretation.

## 0. Verdict in one paragraph

None of the pools that hold the remaining device time is arithmetic-bound. Every kernel
class the driver launches runs at 0.5-15% of peak SM throughput and 0.03-0.42 instructions
per SMSP-cycle (ncu, both GPUs, MEASURED). Rooflines built from ncu-measured executed
instructions and from the geometry census put the frame's real work at roughly 0.3-0.7
ms/frame of device time against a measured heavy-band device union of 5.2-5.9 ms/frame
(ESTIMATED, derivations in section 4). The inflation is latency of two kinds: (1)
launch-shape latency - one-block launches whose duration is one thread's dependent memory
chain (cp_clip_rast_fused at 64 threads: 9.76 us median, 153 ns per triangle, versus 0.47 ns
per triangle at 262k threads; main:VS at 256 lanes: 6.85 us, 27 ns per vertex, versus 0.36 ns
at 524k lanes), and (2) in-kernel latency at low issue even when the grid is wide (the
widest VS launches run at 2.5% of issue peak and ~20% of DRAM peak; FS at 12 waves issues
3.64% of cycles; cp_rasterize_stage3 keeps SMs active at 1-11% throughput). Exactly two
device pools are throughput-bound: cp_fs_compact (a full-target visibility-buffer scan per
batch, 1.40 TB/s effective, 78% of DRAM peak) and memset (1.09 GB/frame of clears in the
heavy band at 2.8 TB/s effective) - and both are redundant volume (77 scans and 36 visbuf
clears per frame for ~10 render scopes), not irreducible work. So the handoff's phrase
"clock-bound work that scheduling does not shrink" (PERF_HANDOFF.md, "What is left,
honestly") is half right: re-scheduling today's launch units does not shrink them
(DEAD_ENDS 20-22, 36 measured that), but the pools are not clock-bound arithmetic; they are
latency shells around 0.3-0.7 ms of work, and a design that changes the work unit would
expose that. The B200 evidence says the same thing: identical instruction counts,
equal-or-fewer cycles on B200, so the frame is bound by cycle-denominated dependency chains
and ns-denominated memory latency, not by anything the B200's 4.5x memory bandwidth could
feed.

## 1. Frame conventions and windows used

- A frame is two vkQueueSubmit events (WORKFLOW.md 4). favorite3 real work starts at
  external submit 2782 (frame 1391); heavy band ~frames 2200-3150 (WORKFLOW.md line 202).
- /tmp/postdead/trace.sqlite: session start 1503511535334936 (monotonic ns,
  TARGET_INFO_SESSION_START_TIME); external submit 2782 in /tmp/postdead/submit-ts.txt is
  1503520811196996 -> relative 9.275862060 s, matching the documented "frame-1391 submit at
  relative 9.276 s". Window [9275862060, 12340525407) ns = external submits 2782..3607 =
  412.5 frames. My device union over kernels+memcpy+memset in that window is 4.5099 ms/frame,
  equal to the documented 4.510 (MEASURED - window calibration check).
- /tmp/postpin-rtx2/trace.sqlite: nsys started with --delay=14, so the trace begins at
  external submit 4898 (frame 2449) and ends at submit 6925. Heavy-band window used:
  external submits [4898, 6300) = 701 frames, trace ns [9341207, 6464509572), 9.209
  ms/frame of CUPTI-inflated wall, 846.3 kernels/frame, device union 5.944 ms/frame
  (MEASURED, this trace). The handoff's heavy-band trace (1,011 frames, 727.9
  launches/frame, union 5.21) is not on disk under /tmp; its numbers are quoted from
  PERF_HANDOFF.md where used and are not re-derived here.
- All per-frame census figures divided by a frame count are means; the project's frame
  time is a hot-tail median.

## 2. (a) The real work per frame

Census window for DEAD_ENDS 30/35: internal command-bearing submits 2784..6945/6947 ->
(6947 - 2784) / 2 = 2081.5 frames (ESTIMATED from documented submit indices).

| quantity | total (MEASURED, source) | per frame (ESTIMATED = total / 2081.5) |
|---|---:|---:|
| valid primitives, admitted opaque episodes (5,957 episodes) | 414,649,151 (DEAD_ENDS 30; /tmp/hiz-f3-8/*) | 199.2k |
| of which small (bbox <= 16 px) / medium / huge | 225,987,764 / 173,809,075 / 14,852,312 (hizcensus class= lines) | 108.6k / 83.5k / 7.1k |
| bounding-box pixels walked | 36,474,987,587 | 17.5M (small 1.04M, medium 4.69M, huge 11.80M) |
| covered pixels (fragments emitted to the depth test) | 6,763,694,506 | 3.25M (small 37.0k, medium 731.8k, huge 2,480.6k) |
| primitive/tile references at the census's 8 px tiles | 1,458,883,449 | 700.9k (not the driver's 64 px stage-3 tiles) |
| vertex-shader inputs from submit 2784 | 3,479,473,815; 60.836% duplicates (DEAD_ENDS 35) | 1.672M (unique ~655k) |
| VS grid lanes actually launched (trace) | - | 1.803M (postdead), 2.007M (postpin heavy) |
| generated-shader launches | 90.56/frame (post-dead-scope profile) | split by same-stream successor grammar: VS 48.5, FS 36.7, unclassified 5.4 (postdead); heavy: VS 127.6, FS 78.8 |
| FS grid lanes launched (4096-block cap x 256) | - | 32.3M (postdead), 63.1M (heavy) - >95% idle by construction (DEAD_ENDS 9) |
| final shaded quads | 584,640,942 over 3,759 selected multi-segment episodes and 34,765 keyed FS launches (DEAD_ENDS 32) | not derivable per frame: the selection covers ~18% of real-frame FS launches (34,765 of ~188k). Per selected episode: 155.5k quads = 622k lanes = 67% of a 1280x720 target |
| shaded FS lanes per frame | - | ESTIMATED 1.0-2.5M (2.86 admitted opaque episodes/frame x <=622k lanes, plus A-buffer and alpha-test passes); central 1.75M |
| framebuffer | 1280x720 (readback is 3,686,400 B = 1280x720x4, /tmp/residual-gpu-audit/AUDIT.md; visbuf clears of 7,372,800 B = 1280x720x8) | the heavy band also clears a 2080x2080 visbuf (34,611,200 B = 2080^2 x 8) 19.3x/frame and a 3072x2176 x 8 B surface (53,477,376 B) 1x/frame |
| texture samples per pixel | UNKNOWN - no sample census exists; fav3_shaderstats.log gives only registers (hardware-inline FS 91-198 regs) | - |
| huge triangles' share | 3.6% of primitives produce 76% of covered pixels and 67% of bbox pixels (ESTIMATED from the class lines) | stage 3 owns most raster coverage work |

## 3. (c) Trace evidence: duration versus threads, floors, concentration

### 3.1 Queries

Kernel population (both traces; <w0>,<w1> are the window bounds above):

    SELECT k.start, k.end, k.streamId, k.correlationId, s.value AS name,
           k.gridX*k.gridY*k.gridZ AS blocks, k.blockX*k.blockY*k.blockZ AS tpb,
           k.registersPerThread AS regs
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
    WHERE k.start >= <w0> AND k.start < <w1>
    ORDER BY k.streamId, k.start;
    -- 'main' is split by immediate same-stream successor: {cp_clip_triangles,
    -- cp_clip_rast_fused, cp_clip_rast_fused_abuf} -> main:VS; {cp_fs_writeback,
    -- cp_abuf_scatter_colors, cp_abuf_composite} or predecessor cp_fs_compact /
    -- cp_*interpolate* -> main:FS; else main:? (DEAD_ENDS 35's fail-closed grammar).
    SELECT start,end,bytes,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY
      WHERE start>=<w0> AND start<<w1>;
    SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>=<w0> AND start<<w1>;

Union = merge of half-open intervals; per-class union and self-overlap = summed/union.
Full per-class tables: postdead/analysis.txt, postpin/analysis.txt, */per_class.json,
*/threads_bins.json.

### 3.2 Per-class summary (MEASURED)

postdead, whole real-work window, 412.5 frames, 427.2 kernels/frame, summed 4.968 ms/f:

| class | n/frame | summed ms/f | union ms/f | self-overlap | min us | p10 | median | p90 | max | min x launches ms/f | share of pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cp_clip_rast_fused | 24.1 | 1.378 | 0.573 | 2.41 | 2.46 | 6.85 | 29.9 | 140 | 395 | 0.059 | 4.3% |
| main:VS | 48.5 | 0.987 | 0.644 | 1.53 | 1.38 | 2.56 | 7.49 | 59.6 | 227 | 0.067 | 6.8% |
| cp_rasterize_stage3 | 33.9 | 0.492 | 0.411 | 1.20 | 1.34 | 1.38 | 7.30 | 39.6 | 526 | 0.046 | 9.3% |
| main:FS | 36.7 | 0.454 | 0.454 | 1.00 | 2.72 | 2.78 | 5.34 | 21.7 | 169 | 0.100 | 21.9% |
| cp_rasterize_stage2 | 33.9 | 0.287 | 0.264 | 1.09 | 0.54 | 2.37 | 4.93 | 19.3 | 557 | 0.018 | 6.4% |
| cp_rasterize_stage3_abuf | 15.7 | 0.215 | 0.143 | 1.51 | 0.67 | 1.34 | 10.4 | 27.2 | 49.9 | 0.011 | 4.9% |
| cp_vertex_fetch | 7.5 | 0.195 | 0.190 | 1.03 | 2.75 | 2.78 | 8.83 | 63.9 | 955 | 0.021 | 10.6% |
| cp_fs_compact | 33.9 | 0.187 | 0.187 | 1.00 | 1.95 | 2.05 | 5.18 | 9.51 | 18.4 | 0.066 | 35.4% |
| cp_clip_rast_fused_abuf | 15.7 | 0.130 | 0.115 | 1.13 | 2.11 | 3.17 | 4.29 | 19.8 | 31.4 | 0.033 | 25.5% |
| cp_fs_writeback | 33.9 | 0.064 | 0.064 | 1.00 | 1.28 | 1.41 | 1.50 | 2.50 | 9.18 | 0.043 | 67.7% |
| all kernels | 427.2 | 4.968 | - | - | - | - | - | - | - | 0.695 (p10-floor 1.036) | 14% (21%) |

postpin heavy band, 701 frames, 846.3 kernels/frame, summed 7.884 ms/f, union 5.944:

| class | n/frame | summed ms/f | union ms/f | self-overlap | min us | p10 | median | p90 | max | min x launches ms/f | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cp_clip_rast_fused | 70.5 | 2.088 | 0.935 | 2.23 | 1.54 | 2.21 | 9.79 | 107 | 347 | 0.108 | 5.2% |
| main:VS | 127.6 | 1.653 | 1.202 | 1.38 | 1.38 | 3.10 | 6.94 | 22.1 | 248 | 0.176 | 10.6% |
| cp_rasterize_stage3 | 77.4 | 0.843 | 0.662 | 1.27 | 1.34 | 1.38 | 4.38 | 28.9 | 143 | 0.104 | 12.3% |
| main:FS | 78.8 | 0.726 | 0.726 | 1.00 | 1.25 | 2.75 | 3.14 | 17.4 | 185 | 0.098 | 13.6% |
| cp_fs_compact | 77.4 | 0.430 | 0.430 | 1.00 | 1.15 | 2.02 | 5.28 | 7.65 | 29.0 | 0.089 | 20.8% |
| cp_rasterize_stage2 | 77.4 | 0.423 | 0.382 | 1.11 | 0.54 | 0.70 | 4.03 | 10.2 | 568 | 0.042 | 10.0% |
| cp_clip_rast_fused_abuf | 47.0 | 0.420 | 0.333 | 1.26 | 2.08 | 3.17 | 6.02 | 23.3 | 89.2 | 0.098 | 23.3% |
| cp_rasterize_stage3_abuf | 47.0 | 0.355 | 0.263 | 1.35 | 0.64 | 1.34 | 5.47 | 18.7 | 45.3 | 0.030 | 8.5% |
| cp_rasterize_stage2_abuf | 47.0 | 0.179 | 0.156 | 1.14 | 0.48 | 2.40 | 4.03 | 4.80 | 7.33 | 0.023 | 12.6% |
| cp_vertex_fetch | 6.3 | 0.175 | 0.170 | 1.03 | 2.72 | 4.26 | 9.76 | 72.4 | 548 | 0.017 | 9.8% |
| cp_fs_writeback | 77.4 | 0.127 | 0.127 | 1.00 | 0.99 | 1.22 | 1.44 | 1.95 | 8.54 | 0.077 | 60.6% |
| all kernels | 846.3 | 7.884 | - | - | - | - | - | - | - | 1.084 (p10-floor 1.775) | 13.7% (22.5%) |

Caveat on floors: stage 1/2/3, cp_fs_compact and cp_fs_writeback are PDL secondaries
(griddepcontrol.wait at cp_rasterize.cu:927/971/1183, cp_fs.cu:610/1236), so their CUPTI
duration starts at the early launch and includes 0.44-3.02 us of waiting for the primary
(PERFORMANCE.md section 4, PDL rows). The "min x launches" column therefore includes the
PDL lead; it is the shell, not device-side work.

### 3.3 Duration versus total threads (MEASURED; elasticity = dlog(duration)/dlog(threads))

cp_clip_rast_fused (1 thread per input triangle, 64-thread blocks):

| threads >= | postdead n | median us | ns/thread | heavy n | median us | ns/thread |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1,982 | 9.79 (p10 9.76, p90 9.86) | 153 | 8,997 | 9.76 | 152 |
| 256 | 776 | 11.65 | 45.5 | 8,205 | 5.06 | 19.8 |
| 1,024 | 516 | 13.90 | 13.6 | 5,589 | 7.39 | 7.2 |
| 4,096 | 1,611 | 44.67 | 10.9 | 4,167 | 15.90 | 3.9 |
| 16,384 | 348 | 119.9 | 7.3 | 1,959 | 59.0 | 3.6 |
| 65,536 | 625 | 92.0 | 1.40 | 605 | 104.5 | 1.60 |
| 262,144 | 304 | 128.6 | 0.49 | 297 | 124.1 | 0.47 |
| elasticity over the range | | 0.31 (x4096 threads -> x13 duration) | | | 0.31 | |

main:VS (1 thread per vertex, 256-thread blocks):

| threads >= | postdead n | median us | ns/vertex | heavy n | median us | ns/vertex |
|---:|---:|---:|---:|---:|---:|---:|
| 256 (one block) | 8,209 | 3.62 | 14.1 | 38,915 (55.5/frame) | 6.85 | 26.8 |
| 512 | 910 | 9.63 | 18.8 | 12,031 | 8.48 | 16.6 |
| 4,096 | 2,358 | 4.13 | 1.01 | 8,968 | 5.73 | 1.40 |
| 16,384 | 1,419 | 18.2 | 1.11 | 3,239 | 12.1 | 0.74 |
| 65,536 | 696 | 19.2 | 0.29 | 1,932 | 23.4 | 0.36 |
| 262,144 | 488 | 99.1 | 0.38 | 544 | 105.6 | 0.40 |
| 524,288 | 412 | 184.2 (p10 170, p90 194) | 0.35 | 651 | 188.6 | 0.36 |
| elasticity | | 0.52 | | | 0.43 | |

cp_rasterize_stage2 (fixed 2048x64 grid except tiny batches): elasticity 0.11-0.13; the
duration is set by the queue and the longest warp's bounding-box walk, not by the grid.
cp_clip_triangles (standalone clipper): 64-thread launches take 8.10-8.13 us (p10-p90),
4096-thread launches 2.37 us - negative elasticity: the 64-thread launches are a different,
longer dependent chain (the stable path's slot retirement writes).
cp_rasterize_stage3, cp_fs_compact, main:FS launch fixed grids (2048x64, 900x256,
4096x256) so duration cannot be correlated with threads; their distributions are in 3.2.

Reading: a 64-triangle cp_clip_rast_fused launch costs 9.8 us with a p10-p90 spread of 0.1
us - a deterministic dependent chain (path_flag load -> vs_out loads -> clip -> prim_refs ->
setup_triangle loads -> per-pixel depth load + atomicMin, up to 16 pixels per small triangle,
cp_rasterize.cu:476-660 and 780-905), not work. This matches the hardware curve in
wide_bench.md section 1d, where a launch's time is flat at 10.2-12.3 us from 1 to 1200
blocks of a fixed 4096-pixel item ("a launch's duration is the LATENCY OF ONE ITEM").

Launch-shape penalty (ESTIMATED): summed pool time minus lanes x asymptotic ns/lane.
- cp_clip_rast_fused heavy: 2.088 - 649,400 x 0.474 ns (0.308) = 1.78 ms/f summed (85%);
  postdead: 1.378 - 0.28 = 1.10 (80%).
- main:VS heavy: 1.653 - 2,006,548 x 0.36 ns (0.72) = 0.93 ms/f summed (56%); one-block VS
  launches alone: 38,915 x 6.85 us / 701 = 0.368 ms/f. postdead: 0.987 - 0.63 = 0.355 (36%).
These are summed-time penalties; the union credit is smaller by the class self-overlap
(2.23x and 1.38x), i.e. roughly 0.8 and 0.68 ms/f in the heavy band.

### 3.4 Concentration and degenerate launches (MEASURED, heavy band)

| class | launches < 2.5 us | their time share | top 10% launches' time share | p50 / p90 / p99 us |
|---|---:|---:|---:|---|
| cp_rasterize_stage3 | 41.3% | 5.3% | 41.3% | 4.38 / 28.9 / 65.8 |
| cp_rasterize_stage2 | 28.4% | 6.8% | 40.7% | 4.03 / 10.2 / 34.8 |
| cp_fs_compact | 12.9% | 3.5% | 22.5% | 5.28 / 7.65 / 27.4 |
| cp_fs_writeback | 93.4% | 83.2% | 21.3% | 1.44 / 1.95 / 4.67 |
| cp_rasterize_stage3_abuf | 16.3% | 2.8% | 31.8% | 5.47 / 18.7 / 28.2 |

postdead: stage3 30.5% of launches < 2 us (DEAD_ENDS 19's degenerate launches; 3.0% of the
pool's time). Streams: 399,591 of 593,276 heavy-band kernels on stream 13 (the main
stream), the other eight carry 17k-50k each.

### 3.5 memcpy and memset (MEASURED, heavy band)

memcpy: 353.7/frame, 0.605 ms/f summed (handoff: 0.490 exclusive).

| kind | n/frame | summed ms/f | MB/frame | median us | note |
|---|---:|---:|---:|---:|---|
| HtoD (1) | 305.8 | 0.265 | 0.59 | 0.61-0.77 (512 B..2 KB) | one coalesced upload per launch that has one (ARCHITECTURE 6.3); floor 0.26-0.29 us; pure shell |
| DtoD (8) | 33.3 | 0.152 | 3.77 | 0.61 (2 KB, 31.3/f = 0.069) + 1/frame 3,686,400 B at 75.2 us (49 GB/s = PCIe-class: the app's readback into host-pinned managed memory) | 0.069 shell, 0.076 mandatory |
| DtoA (7) | 6.98 | 0.184 | 98.0 | 2/frame of 2^25-class (33.5-53.5 MB) at 63.7-81.3 us = 0.162 ms/f, 543 GB/s effective | hardware-texture-cache array rebuild (cpvk_texture_cache.c:386-392 cuMemcpy3DAsync); bandwidth work at 30% of DRAM peak |
| DtoH (2) | 7.65 | 0.0035 | ~0 | 0.35 | counter readbacks |

memset: 135.4/frame, 0.392 ms/f summed = 0.392 union (no self-overlap), 1,094 MB/frame at
2.79 TB/s effective (faster than DRAM peak: L2-absorbed fills). Not in the handoff's pool
list; the postdead partition (/tmp/residual-gpu-audit/AUDIT.md) had memset 0.141 exclusive
of 0.143 summed, i.e. memsets are ~99% exclusive.

| bytes | what | n/frame | summed ms/f | median us |
|---:|---|---:|---:|---:|
| 34,611,200 | 2080x2080 x 8 B visbuf clear (cp_renderer.c:5094, w*h*2*fb_samples words of 0xFFFFFFFF) | 19.34 | 0.200 | 10.3 |
| 7,372,800 | 1280x720 x 8 B visbuf clear | 17.19 | 0.044 | 2.56 |
| 3,686,400 | 1280x720 x 4 B per-pixel u32 (peel_next/reject/abuf counts, cp_renderer.c:6187/6195/6461) | 18.53 | 0.029 | 1.54 |
| 53,477,376 | 3072x2176 x 8 B | 1.00 | 0.028 | 28.0 |
| 26,738,688 | 3072x2176 x 4 B | 2.00 | 0.017 | 8.19 |
| 1,292,832 / 312 / 12 | queue sets, counter block, scalars | 20.7 / 17.7 / 4.0 | 0.016 / 0.008 / 0.003 | 0.77 / 0.35 / 0.48 |

postdead whole window for comparison: 104/frame, 0.143 ms/f, 316 MB/frame, no 2080^2 target.

## 4. (b) Rooflines per pool and the inflation factors

### 4.1 Machine constants

RTX 5090 (MEASURED from TARGET_INFO_GPU in the trace and nvidia-smi): 170 SMs, L2
100,663,296 B, DRAM 1.792e12 B/s, max SM clock 3.09 GHz (CUPTI reports clockRate 2.407
GHz). Sustained 2.9 GHz assumed for time conversion. Peak issue = 170 SMs x 4 SMSP x 1
warp-inst/clk x 2.9e9 = 1.972e12 warp-inst/s = 6.31e13 lane-inst/s. wide_bench.md section
1d measured that real kernels saturate at 59-65% SM issue, so the practical issue ceiling
is taken as 60% of peak.
B200 (ASSUMED per brief, not verified here): 148 SMs, ~1.9 GHz -> 1.125e12 warp-inst/s
(0.57x RTX); HBM3e ~8 TB/s (4.5x RTX). B200 memory-bound time = 0.22x RTX; issue-bound time
= 1.75x RTX; cycle-denominated latency = 1.5x RTX.

### 4.2 Instruction counts (MEASURED via ncu, estimator stated)

/tmp/ncu-rtx/v2.csv and /tmp/ncu-b200-v2.csv: the same 45 launches (--launch-skip 60000
--launch-count 45, -k regex:cp_rasterize_stage3|cp_fs_compact|cp_clip_rast_fused,
/tmp/ncu-rtx/run2.sh). Executed warp-instructions per launch are estimated as
smsp__inst_executed.avg.per_cycle_active x 4 SMSP x sm__cycles_active.avg x SM count
(over-counts by up to 2x for grids of a few blocks, where not all SMSPs are active). The RTX
and B200 estimates agree within 0.86-1.06x per launch (sums 0.91-0.98x), which is the
expected result for identical code and a check on the estimator.

| kernel | launches | grid | warp-inst per launch (RTX est.) | per warp | SM throughput % | IPC per SMSP | DRAM % | waves/SM |
|---|---:|---|---:|---:|---|---|---|---|
| cp_clip_rast_fused | 11 | 5..1825 blocks x 64 | 1.7e4 .. 7.5e6 (sum 1.19e7 over 5,300 warps) | 2,249 | 0.06-3.78 | 0.03-0.09 | 0.3-12.2 | 0.00-0.60 |
| cp_rasterize_stage3 | 11 | 2048 x 64 | 5.3e4 .. 1.13e7 (mean 3.21e6) | 13 .. 2,750 (mean 783) | 1.27-10.76 | 0.05-0.25 | 0.13-3.83 | 0.60 |
| cp_fs_compact | 11 | 900 x 256 | 1.16e6 .. 4.24e6 (mean 2.51e6) | 349 | 6.74-15.52 | 0.13-0.23 | 22.5-45.1 | 0.88 |
| cp_rasterize_stage3_abuf | 6 | 512-2048 x 64 | 2.6e4 .. 1.27e6 | | 0.13-2.87 | 0.02-0.09 | 0.06-0.37 | 0.15-0.60 |
| cp_clip_rast_fused_abuf | 6 | 1-38 x 64 | 1.2e4 .. 5.3e5 | | 0.01-0.81 | 0.08-0.10 | 0.12-1.27 | 0.00-0.01 |

No ncu sample of favorite3's generated main exists. Analogues (MEASURED elsewhere):
SM120.md - gltfscenerendering VS 408 SASS, FS 648 SASS; pbribl FS 2,168 SASS; wide_bench.md
section 3 (old capture) - FS main at grid 4096 (12.05 waves/SM, 126 regs): achieved
occupancy 25.7%, SM issue 3.64% of elapsed / 10.32% of active; 65% of FS launches run under
50 instructions per thread (DEAD_ENDS 9). favorite3 FS registers: hardware-inline 91-198
(/tmp/fav3_shaderstats.log), fused 203 -> capped 126 with 112-200 B spill; VS 25-80 regs.

### 4.3 Per-pool roofline table (RTX 5090, heavy band unless stated)

Measured columns: exclusive from PERF_HANDOFF.md (heavy band, 1,011 frames); summed/union
from my postpin heavy-band window (701 frames). Rooflines are ESTIMATED; derivations follow.

| pool | measured excl (handoff) | summed / union (postpin heavy) | roofline low-high ms/f | inflation (excl / roofline) | verdict |
|---|---:|---:|---:|---:|---|
| cp_clip_rast_fused (clip + stage 1) | 0.592 | 2.088 / 0.935 | 0.03-0.04 | 15-20x (union 24x, summed 50-70x) | latency shell: 85% of summed time is launch shape; the wide asymptote itself is at 7.5% of issue peak |
| cp_rasterize_stage2 | (inside "direct raster") | 0.423 / 0.382 | 0.015-0.025 | 15-25x on union | latency: lane-0 setup + warp bbox walk chains; elasticity 0.11 |
| cp_rasterize_stage3 | 0.424 | 0.843 / 0.662 | 0.03-0.21 | 2-14x (central ~5x) | latency inside the block (thread-0 setup, two barriers, atomics) - SMs active at 1-11% throughput; 41% of launches empty |
| cp_fs_compact | 0.372 | 0.430 / 0.430 | 0.10-0.32 (DRAM); 0.10-0.16 (issue) | 1.2-1.4x vs DRAM roofline | throughput-bound (memory); 1.40 TB/s effective per launch = 78% of DRAM peak. Volume is redundant: 77 full-target scans/frame |
| cp_fs_writeback | (inside "direct FS support" 0.25) | 0.127 / 0.127 | <0.01 | >10x | 93% of launches at the 1.4 us floor |
| generated VS | 0.689 | 1.653 / 1.202 | 0.11-0.22 (memory); 0.02-0.03 (issue) | 3-6x on excl, 5-11x on union | latency: 56% launch shape (one-block launches 0.368 summed); widest launches at 2.5% issue and ~20% DRAM peak (constraint not identified - see lead L2) |
| generated FS | 0.678 | 0.726 / 0.726 | 0.05-0.20 | 3.4-14x | latency: 3.6% issue at 12 waves (old-capture ncu), 25.7% occupancy at 126-128 regs; shaded-lane count uncertain (1.0-2.5M) |
| memcpy | 0.490 | 0.605 / - | 0.19-0.25 mandatory (readback 0.076 + array refresh 0.11-0.16) | 2-2.6x | 55% shell (339 small copies, 0.334 ms), 45% bandwidth/PCIe work |
| memset | (not listed) | 0.392 / 0.392 | 0.05-0.10 if cleared once per scope | ~4-8x (volume) | throughput-bound, redundant volume: 36.5 visbuf clears/frame, 1.09 GB/frame |
| A-buffer chain (clip/s2/s3 _abuf + support) | ~0.52 (0.266 + 0.252, postdead) | 0.954 + ~0.2 support | 0.10-0.30 (UNKNOWN volume) | 2-5x | one-block cp_clip_rast_fused_abuf launches (23,600 of 32,968 in the heavy band) at 5.2 us median |

Derivations:

- cp_clip_rast_fused: heavy 649,400 grid lanes/frame = 20,294 warps x 2,249 warp-inst =
  4.56e7 warp-inst -> 4.56e7 / 1.972e12 = 23.1 us at peak, 38.6 us at 60%. Memory: ~200k
  triangles x (48 B positions + 8 B prim_ref + 4 B active_id) + 37k small-triangle fragments
  x 8 B atomics ~ 14 MB -> 8 us. Roofline 0.03-0.04 ms. postdead: 581,596 lanes -> 20.7/34.6 us.
- cp_rasterize_stage2: 83.5k medium + 7.1k huge triangles/frame; lane 0 runs setup_triangle
  (~200 instructions incl. three divides, cp_rasterize.cu:476-614) as a whole-warp
  instruction stream -> 90.6k x 200 = 1.8e7 warp-inst; the medium walk covers 4.69M bbox
  positions x ~30 instructions / 32 lanes = 4.4e6; total ~2.5e7 -> 12.7 us peak, 21 us at 60%.
- cp_rasterize_stage3: (i) ncu mean 3.21e6 warp-inst/launch x 77.4 launches = 2.48e8 -> 126
  us peak / 210 us at 60% (upper; the heavy band's stage-3 launches are shorter than the ncu
  sample's, median 4.38 vs 7.30 us); (ii) census: 11.8M huge bbox positions x ~35
  instructions / 32 + 2.48M fragments x ~25 / 32 + per-tile setup ~ 3.5e7 -> 18/30 us
  (lower). Range 0.03-0.21 ms.
- cp_fs_compact: 900 blocks x 256 threads = 230,400 quads = the full 1280x720 target; each
  launch reads 921,600 x 8 B = 7.37 MB of visbuf. 77.4 launches x 7.37 MB = 570 MB/frame ->
  0.318 ms at 1.792 TB/s if all from DRAM (ncu says 22-45% of DRAM peak during the kernel,
  so part is L2). Issue: 77.4 x 2.51e6 = 1.94e8 warp-inst -> 98/164 us. Per launch: 7.37 MB
  / 5.28 us median = 1.40 TB/s.
- generated VS: 2.007M lanes x 300-800 instructions = 0.6-1.6e9 lane-inst -> 10-25 us peak,
  16-42 us at 60%. Memory: 100-200 B per vertex (index 4 B, attributes 32-64 B of which 60.8%
  are repeats and L2 hits, VS output num_vs_outputs x 16 B = typically 64-128 B,
  cp_renderer.c:4047/4567) -> 200-400 MB -> 0.11-0.22 ms. At the measured asymptote of 0.36
  ns/vertex the 2.0M vertices would take 0.72 ms, i.e. the VS body runs at 3-6x its memory
  roofline even when wide: 524,288 lanes in 184-189 us = 2.8 G vertex/s = 280-560 GB/s at
  100-200 B/vertex (16-31% of DRAM peak) and 1.6e12 lane-inst/s at 550 inst (2.5% of issue
  peak). Occupancy is register-limited to 24-42 warps/SM (48-80 regs); the fused fetch + UBO
  row indirection (batch_rows -> ubo_table -> descriptor row -> data, ARCHITECTURE 3.2) is a
  dependent-load chain per vertex.
- generated FS: 1.0-2.5M shaded lanes x 500-2,000 instructions -> 0.5-5e9 lane-inst -> 8-80
  us peak, 13-132 us at 60%; texture/interpolation traffic mostly L2-resident (1280x720
  working set against 96 MB L2), DRAM 0.03-0.1 ms. Range 0.05-0.20 ms. The 4096-block grid
  schedules 63.1M lanes/frame; idle blocks cost < 0.2 ns each (DEAD_ENDS 9) -> 246k blocks x
  0.2 ns = 0.05 ms/frame, consistent with the measured floor of 2.7 us per 4096-block launch.
- memcpy: 98 MB DtoA read+write at 1.792 TB/s = 0.11 ms; readback 3.69 MB over PCIe
  (measured 49 GB/s effective) 0.076 ms; 0.59 MB of uploads ~0.
- memset: 1,094 MB at DRAM write peak would be 0.61 ms; measured 0.392 (L2-absorbed). Once-
  per-scope clearing (~10-20 clears of 7.4-34.6 MB = 100-350 MB) -> 0.04-0.13 ms at the
  measured 2.8 TB/s effective.

Aggregate: rooflines sum to ~0.5-1.4 ms/frame of device time for the heavy band's real work
including the mandatory copies and today's redundant-but-throughput-bound scans/clears at
their own volume, or ~0.3-0.7 ms/frame with scans and clears at once-per-scope volume,
against a measured device union of 5.21 (handoff) / 5.94 (this window). The frame's
arithmetic content, strictly (VS + FS + raster instructions at 60% issue), is about
0.1-0.35 ms/frame.

## 5. What the B200 counters say (MEASURED ratios, HYPOTHESIS interpretation)

Same 45 launches, sm__cycles_active.avg B200 / RTX and estimated executed instructions:

| kernel | cycles ratio B200/RTX (range; sum) | instruction ratio (sum) | note |
|---|---|---|---|
| cp_rasterize_stage3 (11 x 2048 blocks) | 0.87-1.08; 0.92 | 0.98 | equal cycles |
| cp_clip_rast_fused (11) | 0.51-1.10; 0.61 | 0.94 | the two biggest launches (1825, 377 blocks) 0.52 and 0.51; the nine small ones 0.87-1.10 |
| cp_fs_compact (11 x 900 blocks) | 0.45-0.75; 0.65 | 0.97 | fewer cycles |
| cp_rasterize_stage3_abuf (6) | 0.96-1.20; 0.99 | 0.95 | equal |
| cp_clip_rast_fused_abuf (6, 1-38 blocks) | 0.88-0.94; 0.94 | 0.91 | equal |

sm__cycles_active.avg is a per-SM average, and the B200 has 148 SMs to the RTX's 170, so for
an equal grid each B200 SM carries 1.15x the blocks; equal average cycles therefore mean
fewer cycles per block on the B200. ncu locks clocks to base by default, so I do not convert
these cycles to wall time.

Interpretation: a kernel bound by DRAM latency (a fixed ~500-700 ns) shows cycles
proportional to clock, so at ~65% of the RTX clock the B200 would show ~0.65x cycles -
cp_fs_compact (0.65) and the two wide cp_clip_rast_fused launches (0.51-0.52) fit that: their
time is ns-denominated memory latency/throughput (the B200's cycles fall because each cycle
is longer, not because it does less). A kernel showing equal cycles at a lower clock - stage
3, the small clip launches, the A-buffer variants - is bound by cycle-denominated costs:
in-SM dependency chains (ALU latency 4-6 cycles, L1 ~30, shared memory and __syncthreads
round trips ~20-30), CTA launch/teardown, and per-kernel fixed prologue/epilogue. Stage 3's
body is exactly that shape: thread 0 does the setup, a barrier, 64 threads walk, a barrier,
per tile (cp_rasterize.cu:1175-1370). In neither case is the bound arithmetic throughput
(0.5-15%) or DRAM bandwidth (<=12% RTX, <=8% B200). Hence the B200's slower wall time (10.79
vs 6.68 ms): its lower clock stretches every cycle-bound chain by ~1.5x while its 4.5x
memory bandwidth goes unused. A redesign whose kernels are memory-throughput-bound is the
one that would run faster on the B200 than on the RTX; the current one cannot.

## 6. (d) Deliverable: verdicts and the implication for a per-scope design

### 6.1 Verdict table

| pool | measured ms/f (excl, heavy) | roofline ms/f (ESTIMATED) | inflation | verdict |
|---|---:|---:|---:|---|
| generated VS | 0.689 | 0.11-0.22 | 3-6x | latency shell (launch shape 56% of summed; body at 2.5% issue / 20% DRAM when wide) |
| generated FS | 0.678 | 0.05-0.20 | 3.4-14x | latency shell (3.6% issue at 12 waves; texture/interp stalls at 25.7% occupancy) |
| cp_clip_rast_fused | 0.592 | 0.03-0.04 | 15-20x | latency shell (85% launch shape; 9.8 us one-block launches) |
| cp_rasterize_stage3 | 0.424 | 0.03-0.21 | 2-14x | latency shell (in-block dependency + barriers; 41% empty launches) |
| cp_fs_compact | 0.372 | 0.10-0.32 | 1.2-1.4x | arithmetic/memory work, redundant volume (77 scans/frame) |
| memcpy | 0.490 | 0.19-0.25 | 2-2.6x | 55% shell, 45% mandatory bandwidth |
| memset | ~0.35-0.39 (not in handoff; union 0.392) | 0.05-0.10 (once per scope) | 4-8x | memory work, redundant volume (36 visbuf clears/frame) |
| A-buffer chain | ~0.5 | 0.1-0.3 | 2-5x | mostly shell (one-block clip launches), volume unknown |

The handoff's claim "clock-bound work that scheduling does not shrink": the second half is
measured true for today's launch units (DEAD_ENDS 20-22, 36); the first half is not
supported. Nothing in the frame is clock-bound arithmetic. It is latency exposed by the
work unit (one launch per batch per stage), plus two throughput-bound pools whose volume is
set by the launch count.

### 6.2 Device-time floor for a design with ~10x fewer, wider kernels (one per scope per stage)

Assumptions (stated, ESTIMATED): ~10 render scopes/frame forming a chain of depth 8-9
(DEAD_ENDS 27); 7-8 stages per scope -> 70-90 kernels/frame, dependent chain depth ~80; each
dependent step costs 2.5-10 us of launch + ramp + tail (wide_bench floor 2.047 us
back-to-back, PDL lead 0.44-3.02 us, one item's chain latency 5-10 us); kernels reach the
60% issue ceiling only above ~2.5 waves (170 SMs x 24 blocks = 4,080 64-thread blocks per
wave), which a 200k-triangle scope reaches only with per-reference or per-pixel threading,
and hot-tile imbalance (DEAD_ENDS 36: one block 94% of a grid) is not solved for free.

| component | low ms/f | high ms/f | basis |
|---|---:|---:|---|
| VS | 0.11 | 0.22 | 2.0M vertices x 100-200 B at 1.79 TB/s (issue 0.02-0.04) |
| clip + raster stages 1-3 | 0.06 | 0.25 | ncu instructions at 60% issue; census positions |
| visibility atomics (3.25M atomicMin) | 0.01 | 0.03 | assumed >= 1e11 L2 atomics/s |
| visibility compaction once per scope | 0.04 | 0.08 | 10-20 scans of 7.4-34.6 MB |
| FS | 0.05 | 0.20 | 1.0-2.5M lanes x 500-2,000 inst at 60%; textures in L2 |
| clears once per scope | 0.04 | 0.13 | 100-350 MB at 2.8 TB/s effective |
| A-buffer (blended) chain | 0.10 | 0.30 | volume UNKNOWN; today's summed 1.2 |
| mandatory copies | 0.19 | 0.25 | readback (PCIe) + texture-array refresh |
| launch shells, ~80 dependent steps | 0.20 | 0.80 | 2.5-10 us per step |
| total device time | 0.8 | 2.3 | central ~1.4 ms/frame |

Against today's heavy-band device union of 5.2-5.9 ms/f and wall 8.4 ms/f, that is a 2.5-7x
reduction in device time (ESTIMATED; uncertainty +-2x on every instruction count because
no favorite3 SASS or ncu main sample exists, +-2.5x on FS lane count, unknown A-buffer
volume). It is a device floor: the frame also carries ~0.38 ms of app record/decode and
CUDA issue APIs at ~1.1 us per operation (1.36-1.50 ms today for ~1,300 operations/frame;
~0.15-0.3 ms at 10x fewer). A frame at this floor would be roughly 2-3 ms on the RTX. B200
for the same design (ASSUMED clocks/bandwidth): memory-bound rows 0.22x, issue-bound rows
1.75x, shells 1.5x -> 0.8-2.4 ms/frame; unlike today, the B200 would not be slower than the
RTX on the memory-bound rows.

What this floor does not license: it is not a forecast for entry 36's tile-binned design
as specified, whose arithmetic (displaced pool 1.46 less fan-out surrender 0.66-0.84)
stands; it says the pools that design left undisplaced - VS, FS, clears - are also latency
shells, and that the surrender term exists to overlap latency shells, so a design whose
kernels are actually throughput-bound would not need to pay it. That is an argument, not a
measurement; entry 36's own prototype was not wide in practice (one block 94% of the grid).

## 7. Leads, each checked against DEAD_ENDS.md

Admission gate in this project is 0.500 ms/frame union-exclusive. None of the following
clears it on its own with a measurement behind it; they are the terms the arithmetic in
sections 4-6 exposes.

L1. Memset volume is a term entry 36 omitted. Heavy band: 0.392 ms/f of memset, union
0.392, ~99% exclusive in the postdead partition; 0.244 of it is 36.5 visibility-buffer
clears/frame (19.3 of a 34.6 MB 2080x2080 target, 17.2 of the 7.4 MB main target). A design
that shades once per scope clears once per scope: ceiling 0.15-0.25 ms/f RTX (ESTIMATED:
0.244 + 0.029 minus 10-20 per-scope clears), 0.05-0.10 B200 (HBM3e). Closest dead ends: 36
(its displaced pool 1.4623 lists raster/compact/writeback only), 7 (folding the vertex-input
clear - refuted because that clear moves ~90 MB/frame; this is about removing redundant
clears, not folding a needed one), 8. Why different: it corrects 36's arithmetic - midpoint
0.409 + 0.15-0.25 = 0.56-0.66, which crosses the gate - rather than proposing a new
mechanism. Cheapest probe (read-only, hours): union-exclusive of memset against all activity
in the heavy-band trace, and a count of visbuf clears per render scope from
CUDAVK_PLAN_STATS output. Effort: probe hours; realization = entry 36's design.

L2. The VS body runs at 3-6x its memory roofline even when wide, and nobody knows why. The
524,288-lane launches (651 in the heavy band, tight p10-p90 178-194 us) sustain 2.8 G
vertex/s: 16-31% of DRAM peak, 2.5% of issue peak, at 24-42 warps/SM. Either dependent-load
chains (fused fetch -> batch_rows -> ubo_table -> descriptor row -> data) or
sector-inefficient interleaved attribute reads. Ceiling if the body reached half its
roofline: 0.2-0.35 ms/f RTX (VS excl 0.689 x (1 - 0.5..0.7)), 0.3-0.5 B200 if the constraint
is clock-denominated. Closest dead ends: 14 (fused fetch - superseded, the fusion is in), 35
(reuse, 0.438), and SM120.md's lesson that halving instructions bought nothing (this targets
loads, not instruction count). Cheapest probe: ncu -k regex:main --launch-skip N -c 1
--section MemoryWorkloadAnalysis --section WarpStateStats --section SourceCounters on one
2^19-lane VS launch (1-2 min of GPU); the trace gives the launch ordinals. Effort: probe
hours; fix unknown until the probe.

L3. One-block VS launches: 55.5/frame at 6.85 us = 0.368 ms/f summed in the heavy band;
total VS launch-shape penalty 0.93 summed / ~0.68 union. Merging them across batches is only
legal for adjacent launches of the same shader key (the FS already carries a per-draw row
indirection, ARCHITECTURE 3.2; the VS is batched the same way), so the realizable share is
the same-key adjacency, which is unknown. Ceiling <= 0.39 union, realistically 0.1-0.28 ms/f
after the merge rule's 1/1.375 self-overlap factor. Closest dead ends: 20-22 (the wide raster
merge cost +0.410 because the merged launches were concurrent and the merged kernel
serialized), 35. Why different: these VS launches are 73% serial by self-overlap and would
merge into an existing batched kernel, not a new one; but the merge rule still applies and
could invert. Cheapest probe: offline, on the existing full-key census records in
/tmp/shader-census-f3 (210,944 batch records with 256-bit keys): count adjacent same-key VS
launches with grids <= 2 blocks and price them at the asymptote. Effort: probe hours;
mechanism days.

L4. cp_fs_compact is throughput work whose volume is redundant (77 scans of a full target
per frame; 0.372 excl). Once-per-scope compaction removes ~87% of the scanned bytes
(ceiling ~0.32 RTX; B200 memory-bound so ~0.07). This is inside entry 36's displaced pool
and is not a new lead; it is reported because the kind of work matters for a design: the
scan is bandwidth-bound, so its cost scales with bytes and cannot be hidden by concurrency.
Closest dead ends: 36, 31 (0.170), 32 (0.036), 26.

Refuted-adjacent findings, not re-proposed: the texture-array refresh copies (98 MB/frame,
0.162 ms/f) are DEAD_ENDS 5's canonical-array-ownership pool (upper bound 0.211); the 306
small HtoD uploads/frame are already one coalesced copy per launch (ARCHITECTURE 6.3,
DEAD_ENDS 8); the 41% empty stage-3 launches are DEAD_ENDS 19 (needs a readback worth 100x
the launch); the 96%-idle FS grid is DEAD_ENDS 9 (idle blocks cost < 0.2 ns).

## 8. Unknowns that bound the confidence

1. No favorite3 SASS or ncu sample of the generated main kernels exists; VS/FS instruction
   counts are analogues (SM120.md) with a +-2x range.
2. Shaded FS lanes per frame are not censused for the whole population (DEAD_ENDS 32 covers
   18% of FS launches); range 1.0-2.5M used.
3. Texture samples per pixel and the A-buffer's per-frame fragment volume are not recorded.
4. The handoff's 1,011-frame heavy-band trace is not on disk; my heavy-band numbers come
   from a 701-frame post-pin window that starts at frame 2449 and is CUPTI-inflated.
5. The ncu instruction estimator over-counts by up to 2x for grids of a few blocks; RTX-B200
   agreement (0.86-1.06x) supports it but does not prove it.
6. B200 SM count, clock and bandwidth are taken from the brief, not verified.
7. sm__cycles_active.avg is a per-SM average; wall-time per launch on the B200 was not
   measured here, so "equal cycles" is compared as the handoff compared it.
8. Whether the VS body is bound by load-chain latency or by sector inefficiency is
   undetermined (lead L2's probe).
9. The 2080x2080 and 3072x2176 targets in the heavy band are inferred from memset sizes;
   what the app renders into them was not traced back to the capture.
