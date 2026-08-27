# Programmatic dependent launch (PDL) prototype for cudavk

Worktree `/tmp/pdl-tree`, branch `pdl-prototype`, base `e2fea470d04`.
Build dir `/tmp/pdl-tree/build-cudavk-pdl`. The main tree
(`/home/alexzhukov/mesa`) and its build directories were not touched.
Nothing was run on the GPU: no replay, no benchmark, no `meson test`, no CUDA
program that creates a context. The SASS below comes from NVRTC + ptxas +
cuobjdump, all of which are offline compilers.

Status: **ready to measure.**

## 1. What the mechanism is

`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (value 6) on the
*secondary* launch lets the driver start that grid before the previous kernel
in the same stream has completed and flushed. The secondary must then execute
`griddepcontrol.wait` before it touches anything the primary wrote. The
primary needs no change: without an explicit
`griddepcontrol.launch_dependents` the trigger happens implicitly after all of
its CTAs exit, so what is bought is the inter-grid gap (grid teardown, the
scheduler round trip and CTA setup of the secondary) plus whatever the
secondary can execute before its wait.

`cudaGridDependencySynchronize()` is not visible to NVRTC 12.8, so the PTX
instruction is written with inline asm and a `"memory"` clobber.

## 2. Files changed

| file | what |
|---|---|
| `src/cudavk/cp_debug.c`, `.h`, `FLAGS.md` | one registry flag, `CUDAVK_PDL`, `CP_FLAG_BOOL_VALUE`, default off. `tests/cp_debug_doc.py --check` passes; `FLAGS.md` regenerated. No `getenv` outside the registry (`tests/cp_no_getenv.py` passes). |
| `src/cudavk/cp_kernels.c`, `.h` | `cp_pdl_kernels(sm_major, sm_minor)`: flag AND compute capability >= 9.0 AND `CUDA_VERSION >= 11080` AND `cuDriverGetVersion() >= 11080`. Its answer adds `-DCP_PDL=1` to the NVRTC options **for `cp_rasterize.cu` only**, and is stored in `struct cp_kernels.pdl`. |
| `src/cudavk/cp_smallop_tele.h`, `.c` | `cp_pdl_watch` / `cp_pdl_epoch` and `cp_pdl_stream_op()`. Every intercepted stream operation bumps the epoch: async H2D (both wrappers, including the raw one the upload flush uses), `cuMemsetD32Async`, `cuMemsetD8Async`, and three new interceptions with no census role -- `cuEventRecord`, `cuStreamWaitEvent`, `cuMemcpyDtoHAsync`. Those seven are every stream-ordered call in `cp_renderer.c`. |
| `src/cudavk/cp_renderer.c`, `.h` | `cp_launch_after()` -- `cp_launch()` is now a wrapper that passes a null predecessor. `CP_LAUNCH_AFTER(prev, ...)`. `struct cp_context` gets the predecessor token and two counters. `cp_plan_report()` prints take/decline. |
| `src/cudavk/kernels/cp_rasterize.cu` | `CP_PDL_WAIT()`, and the waits in `cp_abuf_scan_finish` and `cp_abuf_quad_fill_all`. |

`tests/cp_launch_audit.py` still passes: the only textual `cuLaunchKernel(` in
`cp_renderer.c` is the one in `cp_launch_after()`, and `cuLaunchKernelEx(` does
not match the audit's pattern.

## 3. Links converted, and links deliberately not

Converted, all in the A-buffer scan chain on the main stream:

1. `cp_abuf_scan_reduce` -> `cp_abuf_scan_finish` (`cp_abuf_scan_n`). Nothing
   at all between them in the source. This is the ~128-launch-a-frame link the
   prototype was written for.
2. `cp_abuf_quad_count_all` -> `cp_abuf_scan_finish` (`cp_abuf_quad_build`).
3. `cp_abuf_scan_finish` -> `cp_abuf_quad_fill_all` (`cp_abuf_quad_build`).

Not converted, on purpose:

* **stage1 links** (`:6304`, `:6498`, `:6725`, `:8522` in the base file). A
  `cuMemsetD32Async` on the queue counters sits directly in front of stage1.
  Blocker (b). Not attempted; and if someone claims one later, the epoch check
  refuses it anyway.
* **stage2 -> stage3 and the `_abuf` triples** (`6346/6349/6352`,
  `6502/6505/6508`, `8524/8527/8530`). Out of scope for the first cut; they
  need a wait compiled into the stage kernels, which is a much larger surface
  than two small utility kernels.
* **`cp_abuf_scan_classic`** (`abuf_scan_block` x N -> `abuf_scan_add`). It is
  a chain of kernels with nothing between them, so it is convertible, but it
  only runs with `CUDAVK_NO_ABUF_FUSE_SCAN=1`. Converting a non-default path
  adds risk to a measurement it cannot influence.
* **the pass-side streams / opaque fan-out.** Untouched. The fan-out overlaps
  across segments; these links are within one dependent chain, so this is not
  double counting.

## 4. Blocker (a): the owed upload flush

`cp_launch()` flushes the coalesced upload span on `cp->stream` immediately
before every launch. If that flush issues a copy, the thing in front of the
launch is a memcpy, not the named kernel, and the attribute's premise is gone.

This is **checked, not assumed**. `cp_upload_flush()` sends its copy through
`cp_smallop_htod_async_raw()`, which calls `cp_pdl_stream_op()` and bumps
`cp_pdl_epoch`. `cp_launch_after()` reads the epoch *after* calling
`cp_upload_flush()` and compares it with the epoch recorded at the previous
launch. A flush that issued anything moves the epoch and the launch declines.
A flush that issued nothing leaves it alone.

## 5. Blocker (b): the queue-counter memsets

The same mechanism, from the other end. `cuMemsetD32Async` is one of the calls
`cp_smallop_tele.h` renames, so every clear bumps the epoch too. A converted
link with a memset in front of it declines itself. That is why the stage1
links were not the first to convert and why nothing bad happens if someone
converts one before noticing.

The full decline set is: the module has no waits in it (old device, old
driver, flag off), the previous launch was a different kernel, the previous
launch was on a different stream, or anything at all reached a stream in
between -- clear, copy, event record, cross-stream wait, or the upload flush.
Concretely, link 2 above declines itself under `CUDAVK_ABUFFER_TIMING=1`,
because `cp_abuf_mark()` records two events between the count and the finish.

Caveat, stated rather than hidden: the epoch is global and counts operations
on *any* stream, and the token is per context and unsynchronised. Both err
towards refusing. There is no path that accepts an attribute it should have
refused unless two threads submit to the same stream concurrently, which is
already undefined for the callers.

## 6. The SASS finding

Offline, no GPU: NVRTC 12.8 at `compute_120` (the same options
`cp_kernels.c` passes), then `ptxas -arch=sm_120 -O3`, then `cuobjdump -sass`.
Reproduce with `/tmp/perf-audit/sass/nvrtc_dump.c` (see section 8).

**`cp_abuf_quad_fill_all` -- the conversion buys something.**
`ACQBULK` sits at `0x100`, sixteen instructions in. Before it:
`S2R SR_TID.X`, `S2UR SR_CTAID.X`, the `IMAD` for `b`, the bounds test, and
**`LDG.E R2, desc[UR6][R2.64]`** -- the `blk_counts[b]` load -- and the
`ISETP` on its result. That is a dependent global load whose latency is
resolved while the predecessor is still draining. `blk_counts` comes from the
quad count, two grids back, which has fully completed; only the immediately
preceding grid's dependency is relaxed, so reading it before the wait is
correct.

**`cp_abuf_scan_finish` -- as written it bought nothing, so it was given
something to do.** Its first act was `sums[tid]`, the predecessor's output.
Nothing else in it is independent, and `ACQBULK` would have landed at offset
`0x40` with only register setup in front of it, exactly the hoisting the
brief warned about. The fix is the one the CUDA guide's own figure shows: the
fill cursor (`zero`) is written, never read, by this kernel, so its clear is
hoisted out of the main loop and placed **before** the wait. `ACQBULK` is now
at `0x770`, 119 of 336 instructions in, 15 of them global stores (ptxas
unrolls the clear 8x/4x/2x/1x).

That preamble is real only where a cursor is passed, which is the pixel scan
(`cp_abuf_scan(..., ab->cursor)`, one call per A-buffer episode, n up to
921,600). Where `zero` is null -- the quad-build scan and every other
`cp_abuf_scan_n` -- the `@P0 BRA` at `0x90` skips the clear and the wait is
reached after about ten instructions of setup with no memory traffic. **On
that path the conversion buys only the inter-grid gap and nothing at
instruction level. If the measurement is flat, that is the first reason to
suspect, and it was known before the run.**

The hoist is output-preserving: no path in the kernel reads `zero`, the same
n words are stored either way, and `cp_abuf_scan_n()` now asserts that `zero`
aliases neither `in`, `out` nor `counts`. It is compiled only under
`CP_PDL`, because without the attribute the second pass over the index range
would be pure loop overhead.

No `ERRBAR` is emitted anywhere: there is no explicit
`griddepcontrol.launch_dependents`, by design. `cp_abuf_scan_reduce` writes its
block sum as its last act, so an explicit trigger could not be placed any
earlier than the implicit one.

## 7. Flag off is byte-identical

Stronger than a bitstream diff, on the kernel side: the PTX NVRTC produces for
`cp_rasterize.cu` **without** `-DCP_PDL=1` is md5-identical to the PTX from the
base revision's source.

```
1c6cf66f1ab8c937ceb473f935992ac0  rast_head.ptx   (git show HEAD:...cp_rasterize.cu)
1c6cf66f1ab8c937ceb473f935992ac0  rast_pdl0.ptx   (this branch, CP_PDL undefined)
```

All 40 kernels in the module disassemble identically. On the host side, with
the flag off `cp_pdl_watch` is false, every `cp_launch_after()` computes
`pdl == false` and falls through to the same `cuLaunchKernel()` on the same
stream in the same order, and `-DCP_PDL=1` is never passed, so the NVRTC disk
cache keys are unchanged as well.

With the flag on the change is scheduling only: the same kernels, the same
grids, the same arguments, the same stream order. The one code motion is the
cursor clear inside `cp_abuf_scan_finish`, which writes the same zeros to the
same words.

## 8. Exact commands

Build (never `ninja` in someone else's build dir):

```bash
cd /tmp/pdl-tree
export PATH=/home/alexzhukov/mesa/venv/bin:$PATH
meson setup build-cudavk-pdl -Dbuildtype=debugoptimized \
  -Dgallium-drivers=llvmpipe -Dvulkan-drivers=swrast -Dcudavk=true \
  -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled
ninja -C build-cudavk-pdl
python3 src/cudavk/tests/cp_launch_audit.py
python3 src/cudavk/tests/cp_debug_doc.py --check
python3 src/cudavk/tests/cp_no_getenv.py
```

A/B. Both arms are the *same* shared object; the only difference is one
environment variable, so this is as clean an A/B as the harness allows:

```bash
LABEL=pdl \
CAND="CUDAVK_PDL=1" \
CTRL="" \
BASE_ENV="CUDAVK_TEXTURE_CACHE=1" \
I=/tmp/pdl-tree/build-cudavk-pdl/src/cudavk/cudavk_devenv_icd.x86_64.json \
bash /tmp/perf16/cp_two_replay_ab.sh
python3 /tmp/perf16/cp_two_replay_report.py /tmp/perf16/pdl-tworeplay
```

**Warm the NVRTC disk cache first.** `-DCP_PDL=1` gives `cp_rasterize.cu` a
different cache key, so the very first candidate process compiles that module
from source. Run one throw-away candidate replay before the timed set, or the
first arm carries an NVRTC compile the control never pays.

Check the conversion actually happened before believing any number:

```bash
VK_DRIVER_FILES=/tmp/pdl-tree/build-cudavk-pdl/src/cudavk/cudavk_devenv_icd.x86_64.json \
CUDAVK_TEXTURE_CACHE=1 CUDAVK_PDL=1 CUDAVK_PLAN_STATS=1 \
  <replay command> 2>&1 | grep "programmatic dependent"
```

It prints `cudavk: programmatic dependent launches: N of M offered took the
attribute (P%), K declined`. **A low take rate makes the timing meaningless.**
Expect close to 100% with `CUDAVK_ABUFFER_TIMING` unset; expect roughly a
third to decline with it set, because the two `cp_abuf_mark()` event records
kill link 2.

SASS, offline, no GPU:

```bash
cd /tmp/perf-audit/sass
gcc -O1 -o nvrtc_dump nvrtc_dump.c -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lnvrtc
./nvrtc_dump /tmp/pdl-tree/src/cudavk/kernels rast_pdl1.ptx 1
ptxas -arch=sm_120 -O3 rast_pdl1.ptx -o rast_pdl1.cubin
cuobjdump -sass rast_pdl1.cubin | grep -n ACQBULK
```

## 9. Falsifiable prediction

The saving per converted link is at most (inter-grid gap) + (preamble that now
overlaps). The gap for back-to-back small kernels on this class of part is
1-2 us; the overlapping preamble is one global load in `quad_fill_all` and,
only on the pixel scan, an unrolled clear of n words.

1. **Size.** Under 2% of frame time on both captures. If either arm moves more
   than 5%, it is not this patch -- suspect NVRTC recompilation, another tenant
   on the GPU (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
   must be empty), or replay noise.
2. **Where it pays -- and this differs from the segment-count prediction.**
   These links are per A-buffer scan chain, not per segment. A workload's
   converted-link count is roughly (A-buffer episodes per frame) x 3, and the
   `cudavk: programmatic dependent launches:` line reports it exactly. So I
   predict the **old capture** (12.42 segments, the heavier A-buffer load)
   shows the larger *absolute* saving per frame from this patch, which is the
   opposite way round from "most on one-segment workloads like pbribl,
   pushconstants and Crossroads at 1.71 segments average". Relative saving may
   still favour the short frames. If the take counts per frame come out
   roughly equal on the two captures, my prediction is wrong and the
   segment-count framing is the right one.
3. **Null result that is still informative.** If take rate is high and the time
   does not move, the conclusion is that the implicit trigger (all CTAs exit)
   leaves too little to overlap for kernels this short, and the next step is an
   explicit `griddepcontrol.launch_dependents` in a primary that has tail work
   after its last store -- which `cp_abuf_scan_reduce` does not.

## 10. Not done

* Not run. No timing number in this document is measured; the 1-2 us gap is a
  literature figure, not a measurement on this machine.
* No `meson test`, so the driver-level tests (`cpvk_smoke`, `cpvk_batch`, the
  A-buffer gates) have not been run with the flag on or off. Before anything
  ships, `CUDAVK_ABUF_FUSE_CHECK=1` with `CUDAVK_PDL=1` is the natural gate:
  it compares the fused chain against the classic one on the device, element
  by element, and would catch a wait in the wrong place.
* `cp_abuf_scan_classic`, the stage links and the `_abuf` triples are left for
  a second cut, if the first one pays.
