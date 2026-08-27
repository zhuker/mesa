# PDL tier 2: raster stage links, fragment writeback, segment scatter

Worktree `/tmp/pdl-tree`, branch `pdl-prototype`, on top of the measured tier-1
commit `a79567a1310`. Build dir `/tmp/pdl-tree/build-cudavk-pdl`. Builds clean.
Nothing was run on the GPU: no replay, no benchmark, no `meson test`, no CUDA
program that creates a context. All SASS below is NVRTC + ptxas + cuobjdump,
which are offline compilers.

Status: **ready to measure.**

## 1. One flag, three levels, and why they are split that way

`CUDAVK_PDL` changes from a boolean to a **level** (`CP_FLAG_UINT`, default 0).
`CUDAVK_PDL=1` still means exactly what it meant when it was measured -- the
level-1 PTX is md5-identical to the tier-1 commit's, see section 5 -- so the
existing result is not invalidated and the fallback if a higher level regresses
is simply a lower number.

The levels are also a decomposition of the mechanism, which is the point:

| level | links | preamble in front of the wait | what its increment measures |
|---|---|---|---|
| 1 | A-buffer scan chain | yes, including a hoisted 3.7 MB clear | gap + preamble |
| 2 | rasterizer stage links | **none** on the A-buffer variants | **the inter-grid gap alone** |
| 3 | FS writeback, segment scatter | one independent global load each | preamble again, off the scan chain |

Level 1 measured +1.08% / +0.97%, which is 3.02 us saved per converted link on
the old capture. That is a large number for a launch gap, and level 1 cannot
say how much of it was the gap and how much was the hoisted cursor clear.
**Level 2 answers that**, because its secondaries have literally nothing to
overlap. That is why the stage links are their own level rather than bundled
with the writeback.

## 2. Links converted

Ten raster links (five triples x two), one writeback, one scatter:

| site (this tree) | link | tier |
|---|---|---|
| `cp_renderer.c:6340` | `stage1_abuf` *or* `clip_rast_fused_abuf` -> `stage2_abuf` (count pass) | 2 |
| `cp_renderer.c:6345` | `stage2_abuf` -> `stage3_abuf` | 2 |
| `cp_renderer.c:6499/6504` | the same two, fill pass | 2 |
| `cp_renderer.c:6778` | `stage1` *or* `clip_rast_fused` -> `stage2` (plain path) | 2 |
| `cp_renderer.c:6786` | `stage2` -> `stage3` | 2 |
| `cp_renderer.c:7601/7605` | the plain triple in the tiled-opaque fallback | 2 |
| `cp_renderer.c:8547/8552` | the `_abuf` triple per pass segment, on that segment's side stream | 2 |
| `cp_renderer.c:4157` | compiled fragment shader -> `cp_fs_writeback` | 3 |
| `cp_renderer.c:8308` | `abuf_seg_prefix` -> `abuf_seg_scatter` | 3 |

Where stage 1 may be replaced by the fused clip+stage1 kernel, the call site
names whichever one actually ran (`fused_count ? clip_rast_fused_abuf :
rasterize_stage1_abuf`). Naming the wrong one would cost the overlap, never the
ordering.

The fragment shader is a generated kernel from the `nir_to_ptx` path and gets
no change: the trigger is implicit once its blocks exit. Its `CUfunction` is now
carried out of `cp_fs_launch_shader()` through a new `CUfunction *launched`
out-parameter rather than guessed, because a shader has five possible
executions and a pending tune can retarget the launch.

### Not converted, with reasons

* **stage1 as a secondary**, at every one of the five triples. A
  `cuMemsetD32Async` on the queue counters is directly in front of it -- the
  blocker the first cut deferred. Not offered. If someone offers it later the
  epoch check refuses it anyway, which is the point of the check.
* **`abuf_seg_count` -> `abuf_seg_prefix`.** Two `cp_upload()` calls and several
  scratch allocations sit between them, so the flush inside `cp_launch()` would
  issue a copy immediately before the prefix and the epoch check would refuse
  it on almost every pass. The prefix is also a one-thread kernel with a serial
  loop and no preamble at all, so the link is worth nothing even if the uploads
  were moved. Left alone.
* **interp -> fragment shader.** The secondary is a generated kernel; putting a
  wait in it means teaching `nir_to_ptx` to emit one. Out of scope, and it is
  the natural tier 4.
* **`cp_abuf_scan_classic`.** Unchanged from tier 1: convertible, but only runs
  with `CUDAVK_NO_ABUF_FUSE_SCAN=1`.

### The one host-side reorder

`cuMemsetD32Async(seg_cursor, ...)` moves from *between* the prefix and the
scatter to *before* the prefix. The prefix reads and writes `seg_counts`,
`seg_group`, `seg_base`, `group_base` and `group_counts` and never touches
`seg_cursor`, so this is the same zeros to the same words at a different point
in an order that already had to hold. It happens at every level including 0.
Without it the clear sits between the two kernels and the epoch check refuses
the link -- correctly.

## 3. The SASS table

NVRTC 12.8 `compute_120`, `ptxas -arch=sm_120 -O3`, `cuobjdump -sass`, level 3.
"before" counts SASS instructions ahead of `ACQBULK`; "global" counts memory
instructions among them.

| kernel | tier | ACQBULK at | insts before | global loads before | total | verdict |
|---|---|---|---|---|---|---|
| `cp_abuf_scan_finish` | 1 | `0x770` | 119 | 15 (stores) | 336 | the hoisted cursor clear, on the pixel scan only |
| `cp_abuf_quad_fill_all` | 1 | `0x100` | 16 | 1 | 768 | `blk_counts[b]` load and its branch |
| `cp_rasterize_stage2` | 2 | `0x0e0` | 14 | 1 | 1184 | **only the `path_flag` test, and only when it is set** |
| `cp_rasterize_stage3` | 2 | `0x0e0` | 14 | 1 | 904 | as stage 2 |
| `cp_rasterize_stage2_abuf` | 2 | `0x030` | **3** | **0** | 1304 | **nothing at all** |
| `cp_rasterize_stage3_abuf` | 2 | `0x030` | **3** | **0** | 1032 | **nothing at all** |
| `cp_fs_writeback` | 3 | `0x090` | 9 | 1 | 1608 | the slot-count load and the grid-stride setup |
| `cp_abuf_seg_scatter` | 3 | `0x150` | 21 | 2 | 112 | `num_quads_dev` and `quad_seg[q]` |

Per kernel, plainly:

**`cp_rasterize_stage2_abuf` and `cp_rasterize_stage3_abuf` have nothing to
overlap, and I could not create any.** Three instructions before the wait, all
constant-bank loads of the by-value argument structs. Every value either stage
uses comes out of the queue its predecessor filled: the count, the entries, and
in stage 3 the setup cache. There is no write-only array to clear the way
`cp_abuf_scan_finish` had one, and no load whose address is known before the
queue is read. Vertex positions are independent, but which triangle to fetch is
not. I am shipping these links anyway, and saying so here, because with zero
preamble they are a clean measurement of what the inter-grid gap alone is
worth -- which is the number level 1 could not isolate.

**`cp_rasterize_stage2` and `cp_rasterize_stage3` look better than they are.**
The 14 instructions include a `LDG.E.STRONG.SYS`, which is the volatile
`path_flag` read. But the SASS branches over that load when `args.path_flag` is
null, which is every path except the tiled-opaque fallback. In the common case
these two are also "nothing before the wait".

**`cp_fs_writeback` is a real one.** The slot count is the *interpolation's*
output, not the shader's -- the shader cannot allocate its own slot, which is
why `cp_fs_compact` exists -- and the interpolation is two grids back and has
completed. So the load, the grid-stride arithmetic and the loop's first bound
test all resolve while the fragment shader is still draining. Everything the
shader wrote (`fs_out`, `discard_mask`) is behind the wait.

**`cp_abuf_seg_scatter` is the best of the new ones relative to its size.** 21
instructions and two independent global loads before the wait, out of 112 total,
and its predecessor is a single-threaded kernel walking `nsegs x ngroups`
serially -- so there is a lot of predecessor to overlap with.

No `ERRBAR` anywhere: no explicit `griddepcontrol.launch_dependents`, by design.

## 4. The epoch check is unchanged and still governs every link

Not weakened, not bypassed. `cp_launch_after()` gained one argument, the tier,
and one condition: the attribute is set only when
`cp->dev->kernels.pdl >= pdl_tier`. That is what makes a level-2 binary
incapable of being launched with a level-3 link's attribute and finding no wait
in the kernel. Everything from tier 1 still applies: the predecessor function
must match, the stream must match, and the small-operation epoch must not have
moved -- which is what catches the owed upload flush inside `cp_launch()`, every
`cuMemsetD32Async`, every `cuEventRecord` and every `cuStreamWaitEvent`.

Two consequences worth predicting rather than discovering:

* Under `CUDAVK_PROFILE`, `cp_stage_end()` records an event between the
  fragment shader and the writeback, so the level-3 writeback link declines
  itself. A profiled run measures fewer links than an unprofiled one.
* Under `CUDAVK_ABUFFER_TIMING`, `cp_abuf_mark()` still kills the
  `quad_count_all -> scan_finish` link, as in tier 1.

The take/decline counters now only count links whose tier the running level is
meant to convert, so a level-1 run does not report every level-2 site as a
decline and leave the take rate unreadable.

## 5. Flag off, and level 1, are byte-identical

Both kernel modules, all four levels, NVRTC PTX md5:

```
cp_rasterize.cu   level 0  1c6cf66f1ab8c937ceb473f935992ac0  == tier-1 commit level 0
cp_rasterize.cu   level 1  a01ca6db6baf62be36c04ba77e39296c  == tier-1 commit level 1
cp_fs.cu          level 0  f1c9c41fc1a5018ac967cf06bde36ca3  == tier-1 commit
cp_fs.cu          level 1  f1c9c41fc1a5018ac967cf06bde36ca3  == tier-1 commit
cp_fs.cu          level 2  f1c9c41fc1a5018ac967cf06bde36ca3  == tier-1 commit
```

Level 0 of `cp_rasterize.cu` is the same md5 the tier-1 report recorded against
the original base revision, so level 0 is still bit for bit the pre-PDL kernel.
`griddepcontrol` count per level: rasterize 0 / 2 / 6 / 7, fs 0 / 0 / 0 / 1.

Host side, with the flag off `cp_pdl_watch` is false, every `cp_launch_after()`
computes `pdl == false` and falls through to the same `cuLaunchKernel()` on the
same stream in the same order. The only host behaviour that changes at level 0
is the `seg_cursor` clear moving in front of a kernel that does not read it.

`tests/cp_debug_doc.py --check`, `tests/cp_launch_audit.py` and
`tests/cp_no_getenv.py` all pass. `FLAGS.md` regenerated.

## 6. Exact commands

Build:

```bash
cd /tmp/pdl-tree
export PATH=/home/alexzhukov/mesa/venv/bin:$PATH
ninja -C build-cudavk-pdl
python3 src/cudavk/tests/cp_launch_audit.py
python3 src/cudavk/tests/cp_debug_doc.py --check
python3 src/cudavk/tests/cp_no_getenv.py
```

**Measure each level against the one below it**, not against off -- that is what
makes the increments interpretable:

```bash
I=/tmp/pdl-tree/build-cudavk-pdl/src/cudavk/cudavk_devenv_icd.x86_64.json

LABEL=pdl2 CAND="CUDAVK_PDL=2" CTRL="CUDAVK_PDL=1" BASE_ENV="" I=$I \
  bash /tmp/perf16/cp_two_replay_ab.sh
LABEL=pdl3 CAND="CUDAVK_PDL=3" CTRL="CUDAVK_PDL=2" BASE_ENV="" I=$I \
  bash /tmp/perf16/cp_two_replay_ab.sh
```

Warm the NVRTC cache once per level first: each level is a different `-DCP_PDL`
and so a different disk-cache key, and level 3 is the first that recompiles
`cp_fs.cu`. One throw-away replay per level before the timed set.

Read the counters before believing any number:

```bash
... CUDAVK_PDL=2 CUDAVK_PLAN_STATS=1 <replay> 2>&1 | grep "programmatic dependent"
```

SASS, offline:

```bash
cd /tmp/perf-audit/sass
gcc -O1 -o nvrtc_dump nvrtc_dump.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvrtc
./nvrtc_dump /tmp/pdl-tree/src/cudavk/kernels r3.ptx 3 cp_rasterize.cu
./nvrtc_dump /tmp/pdl-tree/src/cudavk/kernels f3.ptx 3 cp_fs.cu
ptxas -arch=sm_120 -O3 r3.ptx -o r3.cubin && cuobjdump -sass r3.cubin | grep -n ACQBULK
```

## 7. Predictions, made before the run

**Take counts.** Level-1 links are 3 per A-buffer episode. Level-2 links are 2
per rasterizer triple, and a triple runs once per draw batch in the count pass
and once per segment in the fill, so the level-2 count tracks **draws and
segments**, not episodes -- the counter that tier 1 excluded. With
episodes/frame 15.50 and 9.56 and opaque segments/frame 48.5 and 4.0:

* added level-2 links/frame, central estimate `2 x (episodes + segments)`:
  **old ~128, Crossroads ~27**. Call it old 100-190, Crossroads 20-45.
* the ratio of *added* links, old : Crossroads, will be **>= 4x**, against
  1.77x for the level-1 links. **If that ratio comes out near 1.77 again, my
  model of which counter drives which tier is wrong.**
* added level-3 links/frame: one per fragment shade plus one per compacted
  pass, so of order 10-50 on both, and much closer between the two captures
  than the level-2 counts.

**Time.** Level 1 saved 3.02 us per link on the old capture. I do not believe
the gap alone is worth that, because the level-1 average is dominated by one
link per episode that overlaps a 3.7 MB clear.

* level 2 minus level 1: **+0.0% to +0.5% on old, -0.2% to +0.3% on
  Crossroads.** In per-link terms I predict **under 1.0 us per link**, i.e. at
  most a third of level 1's rate. If level 2 comes in at 3 us per link, the gap
  was the whole story all along and the hoisted clear was not needed.
* level 3 minus level 2: small positive on both, +0.1% to +0.6%, because both
  its secondaries do have preamble.

**How this can regress, and it is not a small risk at level 2.**
`cuLaunchKernelEx` is a heavier host call than `cuLaunchKernel`: a config
struct and an attribute array per launch. At ~128 extra extended launches a
frame on the old capture, a 0.3 us host-side difference is 0.04 ms/frame, the
same order as the win I am predicting. On a submit-bound workload level 2 can
therefore come out flat or negative while level 1 stays positive. That is
exactly why the levels are separable, and the fallback is `CUDAVK_PDL=1`.

## 8. Not done

* Not run. No timing number here is measured.
* No `meson test`. The natural gate before anything ships is
  `CUDAVK_ABUF_FUSE_CHECK=1` with `CUDAVK_PDL=3`, which compares the fused
  chain against the classic one on the device element by element, plus the
  `cpvk_*` draw tests for the raster stage links, which the fuse check does not
  cover.
* Tier 4, interp -> fragment shader, needs a wait emitted by `nir_to_ptx` into
  the generated shader. That is the next real step and it is a compiler change,
  not a launch-site change.
