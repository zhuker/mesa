# The per-pixel memcpy: what `cp_resource.c` still does on the host

Two entry points in `cp_resource.c` clear an image with a scalar host loop that
calls `memcpy` once per pixel. Nothing launches a kernel, and nothing waits for
one. They are invisible in the sample sweep because none of the samples in
`headless_streamer_samples.txt` calls them — but `occlusionquery` and `oit` do,
once a frame each, and in those two **a third and a quarter of all CPU-busy time
is that one memcpy.**

This is a finding, not a pass. Nothing here has been fixed.

---

## The number

nsys CPU sampling, 1200 orbit frames at 1280x720, share of CPU-busy samples:

| sample | memcpy | named cudapipe frame | fps | what it clears |
|---|---|---|---|---|
| `occlusionquery` | **33.3%** | — | 275 | depth, `vkCmdClearAttachments` |
| `oit` | **20.6%** | `cp_clear_texture` 4.8% | 289 | `headIndex`, `vkCmdClearColorImage` |
| `gears` | 0.1% | — | 1050 | *control — no per-frame image clear* |

`gears` is the control that makes the other two mean something: it runs the same
driver over the same offscreen path and spends no time in memcpy at all, so the
20–33% is not generic driver overhead.

---

## Why the profile does not simply name the function

The memcpy frames are unresolved — attributed to
`__memcpy_avx512_unaligned_erms` in `libc.so.6` with a stack depth of one.
glibc's AVX-512 memcpy keeps no frame pointer, so the unwinder stops there and
the caller is lost. 1783 of `oit`'s 1803 memcpy samples have no second frame.

So the attribution is by elimination, and it is backed by a direct count rather
than left at that. A gdb breakpoint on each entry point, over a run of about 170
frames:

| sample | `cp_clear_depth_stencil` | `cp_clear_texture` | `cp_resource_copy_region` | `cp_blit` |
|---|---|---|---|---|
| `occlusionquery` | **172** | 0 | 6 | 0 |
| `oit` | 0 | **171** | 5 | 0 |

One call per frame each, and the `resource_copy_region` hits are load-time
staging uploads. At 1280x720 one call is **921,600 memcpys**, which at the
measured share works out to roughly 1.7 ms of host time per frame — about 2 ns
per call, which is what a well-predicted 4-byte PLT call costs.

It is a PLT call and not an inlined store because the size is a runtime value:

```c
memcpy(row + x * pixel_size, data, pixel_size);   /* cp_clear_texture   */
memcpy(row + x * pixel_size, &clear_val, pixel_size); /* cp_clear_depth_stencil */
```

`pixel_size` comes from `util_format_get_blocksize`, so the compiler cannot
specialise it. Hoisting the four-byte case out of the loop is worth trying
before writing a kernel, but a kernel is the real answer — `cp_clear` already
has `clear_kernel` and `clear_depth_kernel` doing exactly this work.

---

## What is still on the host, and what reaches it

| path | reached from | cost |
|---|---|---|
| `cp_clear_depth_stencil` `:613` | `vkCmdClearAttachments` with a depth aspect; render-pass `LOAD_OP_CLEAR` when the fast path bails | scalar per-pixel memcpy, no kernel, **no sync** |
| `cp_clear_texture` `:633` | `vkCmdClearColorImage` | scalar per-pixel memcpy, **no sync** |
| `cp_resource_copy_region` `:261` | `vkCmdCopyBuffer`, `vkCmdCopyImage` | `cuCtxSynchronize()` then host memcpy |
| `cp_blit` `:403` | `vkCmdBlitImage`, same size, different format | sync, then `util_format_translate` |
| `cp_blit` `:443` | `vkCmdBlitImage`, different size | sync, then a float bilinear loop |
| `cp_clear_buffer` `:546` | `vkCmdFillBuffer` with a pattern that is not 1 or 4 bytes | host loop |

### The depth clear is not only reached by `vkCmdClearAttachments`

`render_clear_fast` in `lvp_execute.c:1619` is what turns a render-pass
`LOAD_OP_CLEAR` into cudapipe's GPU `cp_clear`. It only does so when

- the render area offset is zero **and** its extent is the whole framebuffer,
- there is no viewmask,
- conditional rendering is off,
- and every colour attachment being cleared has the *same* clear value.

Fail any of those and it falls to `render_clear`, whose depth arm calls
`clear_depth_stencil` — the host loop. That makes a partial-area clear, a
multiview pass, or an MRT G-buffer whose attachments clear to different values
pay per-pixel host time for depth every frame, with no hint of it in the API
trace. `deferred` and `multiview` are the obvious candidates; **neither has been
measured**, they are leads.

---

## Reproducing it

Neither sample is in `tests/headless_streamer_samples.txt`, which is why this
never showed up. Both build in `~/git/Vulkan/build/bin/` and render correctly
under cudapipe.

CUPTI is not in the way here because CUDA tracing is off — this measures host
time only, and `--trace=none` is what keeps it honest. The check that it is
undistorted is that fps is unchanged from the untraced run: `oit` ran at 288.8
traced against 289.4 under `cp_gpu_busy.sh`.

```sh
ICD=~/mesa/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/target-linux-x64/nsys | sort -V | tail -1)

VK_DRIVER_FILES=$ICD $NSYS profile --trace=none --sample=process-tree \
    --cpuctxsw=none -f true -o /tmp/oit_cpu \
    ./build/bin/oit --offscreen --offscreenframes 1200 --offscreenorbit --benchmark
```

`nsys stats` has no flat CPU-profile report, so read the sampling tables out of
the sqlite directly. Every sample in `COMPOSITE_EVENTS` is already in the
`Running` state, so the count needs no filtering and the denominator is
CPU-busy time across all threads:

```sql
SELECT s.value, count(*) n
  FROM SAMPLING_CALLCHAINS cc
  JOIN StringIds s ON s.id = cc.symbol
 WHERE cc.stackDepth = 0     -- leaf frame
 GROUP BY 1 ORDER BY n DESC;
```

### Counting the calls

`/proc/sys/kernel/yama/ptrace_scope` is 1 on this machine, so gdb cannot attach
to a process it did not start and a poor-man's profiler sampling a background
run gets nothing. gdb has to launch the sample itself. `ignore N 100000` turns
each breakpoint into a counter rather than a stop:

```sh
VK_DRIVER_FILES=$ICD gdb -batch -q \
  -ex "set breakpoint pending on" \
  -ex "break cp_clear_depth_stencil" -ex "break cp_clear_texture" \
  -ex "ignore 1 100000" -ex "ignore 2 100000" \
  -ex "run --offscreen --offscreenframes 10 --offscreenorbit --benchmark" \
  -ex "info breakpoints" --args ./build/bin/occlusionquery
```

`--benchmark` runs a warm-up second before the counted frames, so the hit count
covers more frames than `--offscreenframes` asks for. It is the ratio that
matters, not the total.

---

## One thing to fix along with it

`cp_clear_texture` and `cp_clear_depth_stencil` write managed memory that
in-flight kernels may still be writing, and neither calls `cuCtxSynchronize` nor
launches on `cp->stream`. `cp_clear` is ordered because its clears *are*
kernels on the same stream as their readers; these two have no such argument
available to them. Moving them onto kernels fixes the ordering and the host cost
in the same change — which is the reason to do it that way rather than by
hoisting the memcpy.
