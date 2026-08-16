# HeadlessStreamer: making the capture run, and run fast

A GFXReconstruct capture of the application this driver exists for went from
crashing 4.7 seconds in to replaying all 1,509 frames at a median of 77 ms —
**12.3 fps** — with the frames closer to a software reference than when it
started. This is the record of that: what was wrong, what it cost, how
correctness was checked, and what is left.

Quote the median, and measure it with the plugin below rather than dividing
wall time by frames. Both software drivers have a lazy-compilation tail that
makes their means lie: llvmpipe's mean is 61% above its median because one of
its frames takes **9.9 seconds**.

Branch `cudapipe-gfxr-replay`, thirteen commits from `53b819b3cbf`.

---

## The capture

`~/headless_streamer_20260814T155742.gfxr`, 2.4 GB, taken on a Tesla T4.

```
Application:  ./HeadlessStreamer          Total frames: 0
Pipelines:    160 graphics, 1 compute, 0 raytracing
Allocations:  1799, max 240 MB
```

`Total frames: 0` because it never presents. Everything about extracting
frames from it is in `tests/GFXRECONSTRUCT.md`; the short version is that
`--screenshots` hooks `vkQueuePresentKHR` and there is none, so frames come out
of the readback instead. Per frame the capture issues:

| | |
|---|---|
| render passes | 15 |
| draw calls | 610, of which 483 (79%) blended |
| queue submits | 2 |
| compute dispatches | 1 |
| copies | ~12 (image, buffer, buffer↔image) |

Two structural facts drive everything below. It renders **more than one size** —
1280x720 for the scene, 160x90 for the bloom pyramid, alternating many times a
frame. And it **rebinds fragment uniforms every draw**: over 280,000 blended
draws the mean run of draws sharing state is 1.00.

---

## How correctness is checked

This is the part worth keeping. The sample sweep cannot answer "does the
capture render correctly", and eyeballing a frame cannot answer it either.

### The reference is release llvmpipe, not NVIDIA

NVIDIA is a fine reference but a different renderer. **Release** llvmpipe is
the same lavapipe frontend cudapipe sits under, so anything it gets right,
cudapipe should be able to get right, and a difference is cudapipe's.

Release matters: a debug build asserts in `lp_build_rho_aniso` about nine
seconds into this capture, because the content uses anisotropic sampling. The
build is at `~/mesa/build-lvp-release`, configured with `-Dbuildtype=release
-Db_ndebug=true` and llvmpipe only. It replays the whole capture in under two
minutes and matches NVIDIA to a max channel delta of 3 on the first frame.

### Extracting frames

`tests/cp_gfxr_frames.py`. Index once, then plan, replay, convert:

```sh
CAP=~/headless_streamer_20260814T155742.gfxr
T=~/mesa/src/gallium/drivers/cudapipe/tests

$T/cp_gfxr_frames.py index  $CAP -o hs.blocks.tsv     # ~8 s, reusable
$T/cp_gfxr_frames.py frames hs.blocks.tsv             # what readbacks exist
$T/cp_gfxr_frames.py plan   hs.blocks.tsv -o cp10.json --frames 0,1,2,10,50,200,500,754,1200,-1
$T/cp_gfxr_frames.py replay $CAP cp10.json --icd <icd> --out dumps_x
$T/cp_gfxr_frames.py png    dumps_x
```

**Both drivers must be dumped from the same plan file.** The plan names block
indices, and a plan built by a different rule points at different frames — an
early mistake here silently compared frame 500 of one driver against frame 501
of the other for three of ten frames.

Existing artefacts, if they survive:

| | |
|---|---|
| index | `~/claude-scratchpad/gfxr_headless_streamer/hs.blocks.tsv` |
| 10-frame plan | `~/claude-scratchpad/gfxr_headless_streamer/cp10.json` |
| llvmpipe reference | `~/claude-scratchpad/abuf-resize/dumps_lvp` |

Regenerate the reference with the same plan and the release llvmpipe ICD if it
is gone; it takes about two minutes.

### The metric

Percentage of pixels whose worst RGB channel differs from the reference by more
than **32/255**, and separately by more than **96/255**, meaned over the ten
frames. Two thresholds because they answer different questions: the >96
population is *structural* — missing geometry, wrong blend order, a texture
decoded wrongly — and the >32 population is *shading*, a systematic tone shift
that leaves everything in the right place.

Watching both separated the two bugs that were tangled together. After the
clipper fix the >96 population fell but >32 did not, which said the geometry
was now right and the lighting still wrong. It was.

**Current standard: mean 0.441% above 32/255, 0.002% above 96/255.**
Loading-screen frames — the four early ones with no world geometry — must stay
**bit-exact**; they are a free canary because nothing should ever move them.

### The control floor

Run the *same binary* twice and diff its own two dumps. That is the floor below
which a difference means nothing.

It was ~0.3% before blended draws were batched and is **0.000%** now: batching
made the clipper stable, so blending is exactly submission-ordered and the
driver is deterministic on this capture. If a future change reintroduces
nondeterminism this number is where it shows up first — measure it, do not
assume it.

### Iterating quickly

Replay has no "stop after block N", so dumping frame 0 still streams the whole
file. Kill it once the dump lands: `~/claude-scratchpad/gfxr_3d/run_dump.sh`
turns a two-minute run into about seventeen seconds. Use full runs only for
timing.

---

## How cost is measured

### The oracle

Wall time of the whole replay. Two full runs, and read `tests/TESTING.md` on
the noise floor before trusting a small number.

```sh
/usr/bin/time -f "%e s" env VK_DRIVER_FILES=<icd> \
  gfxrecon-replay -m remap --remove-unsupported --log-file /dev/null $CAP
```

`--remove-unsupported` is required (the app asks for
`VK_KHR_video_maintenance1`); `-m remap` is required (T4 vs replay-GPU memory
types); `--log-file` rather than shell redirection, because gfxrecon's stdout
is block-buffered and a crash eats the last few KB.

### Mean is not steady state

`123.8 s / 1510 = 82.0 ms/frame` is an arithmetic mean including startup,
lazy NVRTC compilation and teardown. Measured against block-index kill points:

| segment | ms/frame |
|---|---|
| frames 5–100 | 106.5 |
| frames 100–700 | 81.6 |
| frames 700–1400 | 77.0 |
| **steady-state slope, 100→1400** | **79.1** |
| fixed startup | 3.9 s |

Early frames are the *expensive* ones — the capture's 160 pipelines compile
lazily as each is first used. Quote the mean for throughput (a session pays
startup once) and the slope for rendering changes. They are ~4% apart; do not
mix them.

### Per-frame timing, without wall clock

Wall time answers throughput. It cannot say what a frame costs, what the spread
is, or where the outliers are — and on this capture the mean and the median are
5% apart because the first hundred frames carry the lazy shader compilation.

**gfxrecon's own FPS measurement is useless here.** `--measurement-file` writes
*no file at all*, because it delimits frames at `vkQueuePresentKHR` and there
are none. Neither does `--measurement-frame-range`.

`vkQueueSubmit` is not interposable either — gfxrecon resolves entry points
through `vkGetDeviceProcAddr` into its own dispatch table, so an `LD_PRELOAD`
never sees them.

What does work is gfxrecon's replay event plugin, which reports
`QUEUE_SUBMIT_BEGIN`/`END` with its own timestamps and is driver-agnostic —
the same plugin measures cudapipe, llvmpipe and NVIDIA without any of them
knowing. `tests/cp_gfxr_fps_plugin.cpp` is thirty lines; this application
submits exactly twice per frame, so submit boundaries are frame boundaries.

```sh
g++ -std=c++17 -O2 -shared -fPIC -I ~/gfxreconstruct/framework/plugin/public \
    -o fps_plugin.so tests/cp_gfxr_fps_plugin.cpp

gfxrecon-replay -m remap --remove-unsupported --log-file /dev/null \
  --replay-event-plugin-path $PWD/fps_plugin.so \
  --replay-event-plugin-params $PWD/submits.txt  $CAP

tests/cp_gfxr_frames.py fps submits.txt
```

Measured this way, over all 1,510 frames:

| | cudapipe | release llvmpipe | NVIDIA |
|---|---|---|---|
| first→last submit | 122.5 s — **12.3 fps** | 101.3 s — **14.9 fps** | 4.66 s — **323.9 fps** |
| mean frame | 81.15 ms | 67.10 ms | 3.09 ms |
| **median frame** | **77.42 ms** | **41.76 ms** | **2.84 ms** |
| p5 / p95 | 70.80 / 95.11 | 38.46 / 58.12 | 2.76 / 3.35 |
| min / max | 8.84 / 753.52 | 6.90 / **9857.18** | 1.30 / 38.86 |

**Prefer the median, and this table is why.** cudapipe's mean is 5% above its
median because one frame costs 753 ms — a lazy shader compile, not rendering.
llvmpipe's mean is **61%** above its median, because LLVM JITs a shader that
costs it a single 9.9-second frame. Comparing the two drivers by wall time
therefore compares their compilers as much as their rasterizers, and gets the
answer wrong by a factor of one and a half.

The median also agrees with the block-index slope (79.1 ms) to within 2%, which
is two independent methods cross-checking.

The p5–p95 spread is worth watching on its own: 70.8–95.1 ms is ±16% frame to
frame, against NVIDIA's ±10%. A streamer cares about that as much as the mean.

By median the gap is **27x**, against the 25x the wall clocks give — the wall
figure is diluted by NVIDIA spending ~30% of its run parsing the file.

### Noise, and paired A/B

Run-to-run spread can reach **±4%**, because `cp_tune_before` times register-cap
candidates at runtime and keeps whichever wins, so one binary can settle into
different states. For anything under ~10%:

- build both variants as separate `.so` files, write one ICD json each with
  `"library_path"` pointing at it, and **alternate** old/new/old/new/old/new;
- report per-pair deltas and the median, not two block means;
- better still, if the change has an env-var arm, put both arms in **one**
  binary — that removes build-to-build variance entirely, and it is how the
  three residency schemes in the allocator work were compared.

A change worth under ~3% may simply not be resolvable in wall time. Argue it
from a direct count instead: the `LD_PRELOAD` shim at
`~/claude-scratchpad/perf276/cushim.c` gives exact per-call-site CUDA API
counts and times at ~0.1% overhead, where CUPTI distorts this driver badly.

### The profiling ladder

`CLAUDE.md` has it; the rule that matters is to establish host/device balance
with **no profiler attached** first, because CUPTI charges every
`cuLaunchKernel` and this driver issues thousands per frame. And remember every
compiled shader is a CUDA kernel named `main`, so per-kernel summaries merge
the vertex and fragment stages — split by grid size, or by the driver kernel
that immediately precedes each launch (there is one stream, so that partitions
cleanly).

---

## What was wrong

Six bugs blocked the goal. Four of them are **structurally invisible** to the
18-sample sweep — not "we got unlucky", but no sample in the tree can reach
them.

### 1. Packed vertex attributes were never expanded — `f2f1e7e895e`

`cp_vertex_fetch.cu` copied `util_format_get_blocksize` bytes into the shader's
16-byte input slot. Vulkan delivers every component in its own 32-bit slot, and
the shader side already addressed them that way, so an `R8G8B8A8_UINT` arrived
with all four components packed into the first.

The capture's skinning shaders read `JOINTS_0` as a `uvec4` and index a
48-byte-stride matrix array with `.x`. Instead of a small integer they got
`j0 | j1<<8 | j2<<16 | j3<<24`, which sign-extends to a ±2 GB offset — an
illegal access that kills the CUDA context. The error is *sticky*, so every
later call failed too, surfacing as a NULL from `cp_allocate_memory` that
lavapipe dereferenced: a segfault in libc three frames from the cause.

Where the value is not an index the same bug is silent and merely renders
wrong. `R8G8B8A8_UNORM` colours and `R16G16_SINT` were garbage throughout.

*Why no sample catches it:* exactly one sample in the tree binds an 8-bit
vertex attribute (`imgui`, UNORM colour), and `gltfskinning` converts glTF's
packed joints to `R32G32B32A32_SFLOAT` on load.

### 2. Block-compressed copies measured in pixels — `09dcd15df54`

`cp_resource_copy_region` multiplied a pixel-denominated box by the *block*
size. BC1 is 8 bytes per 4x4 block, so a copy walked 16x its real extent, off
the end of the allocation. This was the next crash after the vertex fix; the
capture has 576 BC images and 6,120 `vkCmdCopyImage` calls. `cp_blit` had the
identical bug and was fixed later (`f9a6fb05d79`).

### 3. The clipper never clipped `z <= w` — `1b23b4f1f19`

Only `z >= 0` and `w > 0` were implemented, and the comment called the first
one "the near plane". That holds for a conventional projection. **This
application uses reversed-Z** — depth cleared to 0, `GREATER_OR_EQUAL` — so the
two depth planes swap roles and `z <= w` *is* the near plane. A vertex closer
than it still has `w > 0` and `z > 0`, so nothing rejected it and it divided to
a coordinate millions of pixels off screen.

The 3D world was not shaded flat; it was geometrically destroyed and then
self-occluding, because the exploded triangles carried garbage depth into a
depth-writing pass. Draw 133327 (distant buildings) was right to a mean delta of
0.36; draw 133333 (the ground plane, same buffers, near-identical shader,
straddling the near plane) was 63.5% wrong.

*Why no sample catches it:* `VK_COMPARE_OP_GREATER` appears nowhere in the
sample tree. Nothing uses reversed-Z.

### 4. `gl_FrontFacing` was undef — `a45bb27041a`

`nir_intrinsic_load_front_face` had no case in the NIR→PTX backend and fell to
the default arm, which produces `undef`. The surface shader multiplies its
tangent frame by `gl_FrontFacing ? 1.0 : -1.0`, so the normal was undefined,
which gated the entire direct-light block and selected the wrong ambient
hemisphere. What survived was ambient only — the plaza read green where
llvmpipe rendered sand.

The driver knew. `CUDAPIPE_DEBUG_SHADER=1` printed `unhandled intrinsic
'load_front_face'` 65 times, behind an environment variable nobody sets until
they already suspect the shader. That warning now fires always
(`1c0ad4094fa`).

### 5. The A-buffer disabled itself on resize — `9185c7c79e2`

`cp_abuf_setup` had no realloc path, so the first framebuffer size change set
`ab->disabled` for the life of the context. This capture triggers it around
frame five, and the remaining ~1,500 frames all fell back to peeling — four
passes per blended draw instead of one, each ending in a host drain to ask
whether the peel had converged. That drain was **61.5% of the frame**.

*Why no sample catches it:* a sample that renders one size never fires it.

### 6. Fragment shaders never ran without a colour attachment — `0b072176132`

The shading stage was gated on `if (color_data)`, whose comment assumed "no
colour attachment" meant "depth-only pass". Vulkan permits a fragment shader
that exists purely for side effects. Found via `oit`, which renders blank; the
capture does not contain such a pass, so this one is conformance rather than
goal work.

---

## What it cost, and what fixed it

| stage | ms/frame | commit |
|---|---|---|
| first successful replay | 276 | — |
| A-buffer survives resize | 221 | `9185c7c79e2` |
| A-buffer draw batching | 123 | `c81b5eacd8b` |
| small-allocation arena | 87 | `3cab3b7ce85` |
| framebuffer buffers grow-only | 83.6 | `5e1bbefcdfa` |
| A-buffer readback packed | 82.0 | `5e1bbefcdfa` |

**416 s → 123.8 s.** The sample sweep also improved, 47.68 → 38.31 ms total,
mostly from the last two — those samples are host-bound on exactly the
allocation churn removed (`triangle` −84%, `texturemipmapgen` −45%, `bloom`
−34%).

### Against the other two drivers

Same command, no dumps, two runs each:

Wall time first, then per frame. The two disagree, and the second is right.

| driver | wall | wall/frames | **median frame** | fps | max frame |
|---|---|---|---|---|---|
| NVIDIA | 4.94 s | 3.27 ms | **2.84 ms** | 323.9 | 38.9 ms |
| release llvmpipe | 101.1 s | 66.9 ms | **41.76 ms** | 14.9 | **9,857 ms** |
| cudapipe | 123.8 s | 82.0 ms | **77.42 ms** | 12.3 | 753 ms |

**By wall time cudapipe looks 23% behind llvmpipe. By median it is 1.85x
behind**, and the median is the honest number. llvmpipe's mean sits 61% above
its median because it JITs shaders with LLVM and one frame of this capture
costs it 9.9 seconds — roughly 38 s of its 101 s run is one-time compilation
that cudapipe does not pay, NVRTC being both cheaper and better spread. That
flattered the wall-clock comparison and an earlier version of this document
repeated it.

Read it this way: cudapipe *delivers* a session faster than llvmpipe would,
because it starts up cheaper, and *renders* about half as fast once both are
warm. Which one matters depends on whether the streamer is a long-running
process or a short one.

Against NVIDIA the gap is 27x by median. Subtract the replay overhead before
quoting even that — see below.

**Subtract the replay overhead before quoting a ratio.** `gfxrecon-info`, which
parses the file and renders nothing, takes **1.46 s**. That is a floor on what
both drivers pay for reading 2.4 GB, and it is ~30% of NVIDIA's entire wall
time against 1.2% of cudapipe's. Rendering-only the gap is nearer **35x** than
the 25x the wall times suggest. Two caveats in the other direction: `info` may
decode less than replay does, so 1.46 s is a lower bound on the overhead; and
NVIDIA's column is a replay ceiling rather than a rendering one, so it says what
the harness can deliver, not what the hardware can.

### The three that mattered

**A-buffer survives resize** (−19%). Grow-only capacity sized to the largest
framebuffer seen, per-size values recomputed. Reallocating on each change would
have been worse than the disable it replaced, because the sizes alternate
*within* a frame.

**Batching blended draws** (−44%). The first measurement said it was worth
nothing: mean run length 1.00 over 280,000 draws, with `fs_ubos` breaking 97% of
them, confirmed by hashing contents rather than comparing pointers. So the
batching is gated on per-draw *fragment* bindings, which is the increment
`ABUFFER.md` and `BATCHING.md` both name. Submission order needed a stable
clipper — input triangle *t* owns output slots *4t..4t+3*, unfilled slots
retired as zero-area triangles — replacing an `atomicAdd` compaction that did
not respect draw order. Achieved batch length 2.30, and because a batch shares
its rasterization and scan it removed **52% of all launches and memsets**,
which is why it beat the 28.5% the drains alone were worth.

**Small-allocation arena** (−29%). lavapipe backs every descriptor set with a
`VkDeviceMemory`; a 64-byte `cuMemAllocManaged` comes back page-granular, the
host writes it, CUDA recycles the address space, and every launch then
read-faulted on the same fourteen 64 KB pages — **22.4 ms of a 122.6 ms frame,
34% of all kernel time**, at 248,000 faults/s. Small allocations now come from
2 MB blocks advised `PREFERRED_LOCATION=CPU` + `ACCESSED_BY=device`. Faults
fall to 559/frame.

Three residency schemes were built into one binary and compared: pinned host
memory and CPU-preferred managed memory measure **identically**, which settles
the worry that uncached remote reads would give the win back. An arena-without-
advice arm isolated churn (12.2%) from residency (16.8%).

It is **gated on 64 observed small allocations** before the first block opens.
Without that, `gltfscenerendering` regressed 9.7% — it makes *four* small
allocations in an entire run, long-lived descriptor sets bound as constant
buffers that every thread dereferences, and packing four device-hot allocations
into a host-preferred block strands them off-device. An arm that opened and
advised the blocks but served nothing out of them cost 0.1%, which is what
ruled out the mapping, the advice and the allocator.

### Two that were measured and were worth nothing

Recorded so nobody repeats them.

**Async device copies** in `cp_resource_copy_region`/`cp_blit`, replacing
`cuCtxSynchronize` + host memcpy — removes ~8 device drains a frame. Capture
effect: **0.00%**. And it cost `gltfscenerendering` 7.1%, confirmed across five
interleaved pairs. Reverted.

**`CPU_MEMCPY.md`'s rankings.** Its measurements are sound for
`occlusionquery` and `oit`. Its ranking does not transfer: the item it puts
first, `cp_buffer_map`'s unconditional drain, is **6.6 ms out of 417,000** on
this capture — 28,373 calls at 230 ns, because by the time a map runs the
device is already idle. The per-pixel host clears are 0.12% of CPU samples
here. All three of its recommendations inverted under measurement.

---

## Where the frame goes now

Re-profiled at 122.6 ms/frame, before the last two commits. **Re-measure before
acting on any of this** — the proportions moved substantially between profiles,
and acting on a stale one cost a wasted change already.

Host: **49% blocked in sync, 30% issuing work into CUDA, 21% outside any CUDA
call.** Device busy ~70% with **SMs Active 16%, SM Issue 2.8%, DRAM under 2%** —
occupied and empty. Roughly 32% of the frame the device has no work resident.

| call site | per frame | % wall |
|---|---|---|
| `cuStreamSynchronize`, A-buffer drain | 209.5 calls | **45.4%** |
| `cuLaunchKernel` (7,686) | 14.4 ms | 11.7% |
| `cuMemsetD32Async` (3,727) | 7.3 ms | 5.9% |
| `cuMemFree` (1,109) | 5.1 ms | 4.2% |

**The 45% is device-wait, not budget** — 78% of it has the GPU genuinely busy.
The floor for anything that only removes synchronisation is
`max(device 84 ms, host non-sync 62 ms)` = **84 ms/frame**. Getting below that
needs device work removed, not host waiting removed.

Kernel time, 65.5 ms/frame, split by stage:

| | ms/frame | % | fault-covered |
|---|---|---|---|
| fragment shader (A-buffer) | 13.17 | 20.1% | 66% |
| vertex shader | 12.09 | 18.5% | 92% |
| fragment shader (peel/opaque) | 7.76 | 11.9% | 19% |
| `cp_rasterize_stage2_abuf` | 6.06 | 9.3% | — |
| `cp_rasterize_stage3_abuf` | 4.23 | 6.5% | — |

Note the fault-covered figures predate the arena, which took the vertex shader
to 1.16 ms/frame at 0.02% fault coverage.

---

## What is left, ranked

1. **The A-buffer drain, exactly.** 209 drains/frame. Removing it is bounded by
   the 84 ms/frame floor, i.e. −31% absolute best case, and would land far
   short because `cuMemFree` (1,109/frame) and `cuMemAlloc` become the new
   serialization points. A throwaway speculative build measured **−28.5%** but
   rendered wrong images and guessed quad totals 7.1x too high — that number is
   a floor on the prize, not an achievable result. The exact route is more
   batching; the fallback is `CUDAPIPE_ABUFFER_SPECULATE=1` default-off, with
   every downstream kernel device-gated on an "ok" word and a loud report plus
   permanent disable on a miss.
2. **The A-buffer prefix scan.** 2,069 launches/frame — 27% of all launches —
   for 2.93 ms of device time and 3.07 ms of host issue. A single-pass
   decoupled-lookback scan removes ~2,000 launches/frame, perhaps 3-4 ms.
3. **Residual UVM migration.** 14 MB/frame remains after the arena and did not
   move between any arm. It is render-target and readback traffic, and it is now
   the largest unified-memory item.
4. **`cuMemFree`'s 1,109 calls/frame**, 1,599,428 of them from
   `lvp_descriptor_set_destroy`. Currently host cost only — 100% of its duration
   is GPU-idle — but it becomes a drain the moment (1) lands.

Two known limits in what shipped:

- The arena gate **counts allocations, not frees**. `bloom` crosses it on 168
  allocations with two reuses, which is the shape it exists to exclude; measured,
  that costs bloom −0.9%, i.e. marginally faster. Counting frees would separate
  them exactly. Worth doing the day something crosses on volume *and* pays.
- Grow-only framebuffer buffers hold memory rather than returning it:
  `multisampling`'s peak went 1180 → 1710 MiB.

---

## Gating rules, learned the hard way

- **Run the 18-sample sweep on every change** (`tests/cp_iterate.sh <new>
  <old>`). It caught both real regressions this branch produced. Bit-identity on
  the stored 60-frame sets is a stronger signal than the tolerance gate — ten
  samples are deterministic and should stay byte-identical.
- **A >5% move on a >3 ms sample is a question, not a finding.**
  `gltfscenerendering` has spread 6.9% across three runs of one build. It
  produced a false regression and a real one in the same week, indistinguishable
  from a single sweep. Answer it with paired alternating arms.
- **A green sweep is not evidence a change works** — only that it broke nothing
  the sweep covers. No swept sample reaches the clear paths, reads
  `gl_FrontFacing` or `gl_FragCoord`, writes memory from a fragment shader, or
  uses a partial `vkCmdClearAttachments`. For each change, find the thing that
  actually exercises it: `occlusionquery` and `oit` for clears, `offscreen` for
  `gl_FrontFacing`, `ssao`/`subpasses` for `gl_FragCoord`, and the matching dEQP
  group.
- **`renderheadless`'s sweep row is vacuous** — an empty directory matching an
  empty reference. Check it by hand when a change could touch it.
