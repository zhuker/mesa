# Capturing and replaying real applications with GFXReconstruct

Guessing which features a driver needs — from what a class of hardware exposes,
or from what a test suite happens to cover — has been wrong repeatedly. This
measures it instead.

Capture a real application on a driver that works, then:

* **read what it required** — every image format and usage, sampler state,
  pipeline state, which shader stages appear at all, and which compressed
  texture family the content actually ships;
* **replay it against cudavk** — a frame captured on real hardware becomes a
  regression test for this driver, without the application needing to run here
  at all.

The second is the valuable one. Every substantial bug found so far came from
running something realistic and diffing against hardware, not from a test suite.

Everything below was run end to end on this machine: a capture of the offscreen
benchmark taken on an RTX 5090 replays successfully on cudavk.

## Building it

Already built at `~/gfxreconstruct`. To rebuild from scratch:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/LunarG/gfxreconstruct.git
cd gfxreconstruct
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGFXRECON_ENABLE_OPENXR=OFF -G Ninja
ninja -C build
```

`-DGFXRECON_ENABLE_OPENXR=OFF` is not optional here: the bundled OpenXR loader
includes `xcb/glx.h`, which isn't installed, and the build fails on it. Nothing
in this workflow needs OpenXR.

Meson and ninja come from the venv described in `docs/cudavk/history/HANDOFF.md`; put it on
`PATH` first if `ninja` isn't found.

For brevity below:

```bash
GFX=~/gfxreconstruct/build
```

## 1. Capture, on a driver that works

Capture on real hardware — the point is to record what the application asks for
when nothing is failing.

```bash
VK_LAYER_PATH=$GFX/layer \
VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_gfxreconstruct \
GFXRECON_CAPTURE_FILE=/tmp/app.gfxr \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    <application>
```

The file gets a timestamp appended, so the result is something like
`/tmp/app_20260809T090250.gfxr`.

Useful environment variables:

| Variable | Effect |
|---|---|
| `GFXRECON_CAPTURE_FILE` | output path (timestamp appended) |
| `GFXRECON_CAPTURE_FRAMES` | limit to a frame range, e.g. `100-110` |
| `GFXRECON_CAPTURE_TRIGGER` | start/stop capture with a hotkey |
| `GFXRECON_LOG_LEVEL` | `debug` when the capture layer itself misbehaves |

Captures of a real game are large. `GFXRECON_CAPTURE_FRAMES` is worth using —
a handful of frames is plenty to learn the requirements and to replay.

A natively built application needs nothing beyond the environment above. If one
ever has to be captured inside a container or a Wine prefix, the same variables
must be set *inside* that environment and the layer directory made visible to
it — but a native build avoids the problem entirely and is worth preferring.

## 2. Read what the application required

A quick summary, straight from the capture:

```bash
$GFX/tools/info/gfxrecon-info /tmp/app_*.gfxr
```

That gives the application and device names, API version, allocation sizes and
pipeline counts — including whether any ray tracing pipelines exist at all.

For the detail that decides driver work, convert to JSON and summarise it:

```bash
$GFX/tools/convert/gfxrecon-convert --output /tmp/app.json /tmp/app_*.gfxr
python3 cp_capture_requirements.py /tmp/app.json
```

`cp_capture_requirements.py` lives next to this file and reports:

* every image created, with format, type, mip and layer counts, sample count
  and usage — this is what says whether a format must be *renderable* or only
  *samplable*
* sampler states in use — filters, wrap modes, anisotropy, compare
* graphics pipeline state — topology, attachment count, blend, depth, stencil,
  sample count
* which shader stages appear, so it is immediately clear whether geometry or
  tessellation are needed
* the extensions requested
* a summary of which compressed texture families (BC, ETC2, ASTC) are used

The converted JSON is a plain array of records, so anything the script doesn't
cover can be answered with a few lines of Python against the same file.

## 3. Replay against cudavk

```bash
VK_DRIVER_FILES=<path to>/cudavk_devenv_icd.x86_64.json \
    $GFX/tools/replay/gfxrecon-replay -m remap /tmp/app_*.gfxr
```

Two things matter, both of them learned by hitting them:

* **`-m remap` is required.** Memory type indices are baked into the capture and
  differ between drivers. Without it replay stops with
  `Memory allocation failed: specified memory type index exceeds number of
  available memory types.`
* **Do not pass `--wsi`.** It requests surface extensions this driver
  deliberately does not expose, and instance creation then fails with
  `vkCreateInstance returned error value VK_ERROR_EXTENSION_NOT_PRESENT`.
  Captures of offscreen work replay without any WSI at all.

A warning that the replay device differs from the capture device is expected and
harmless — that is the whole point.

## 4. Getting frames out of a capture that never presents

An offscreen application has no swapchain, so GFXR counts zero frames and
`--screenshots` — which hooks `vkQueuePresentKHR` — never fires. `gfxrecon-info`
says so directly:

```
Application exe name: ./HeadlessStreamer
Total frames: 0
```

`GFXRECON_CAPTURE_FRAMES` counts presents too, so a capture like this cannot be
trimmed by frame range either. The whole file is one "frame" to every tool that
asks.

What such an application does have is a readback: it renders into an image and
copies it to a host-visible buffer to encode. `gfxrecon-replay --dump-resources`
can dump the destination of that copy, which is exactly the bytes the
application itself consumed. `cp_gfxr_frames.py` drives the whole path:

```sh
cp_gfxr_frames.py index  capture.gfxr                  # -> blocks.tsv (7.6 s for 2.4 GB)
cp_gfxr_frames.py frames blocks.tsv                    # what readbacks exist
cp_gfxr_frames.py plan   blocks.tsv --frames 0,100,-1  # -> dump.json
cp_gfxr_frames.py replay capture.gfxr dump.json --icd <icd> --out DIR
cp_gfxr_frames.py png    DIR                           # .bin -> .png
```

`--frames` takes `0,5,-1`, `0-9`, `::100` or `all`. On the HeadlessStreamer
capture `frames` reports:

```
1510 readbacks, from 2 image(s):
  image 14        1509 frames  1280x720  VK_FORMAT_B8G8R8A8_UNORM   <- frame output
  image 240448       1 frames   640x360  VK_FORMAT_B8G8R8A8_UNORM
```

To correlate every offscreen frame with its replay time, run the complete
two-replay workflow:

```sh
tests/cp_gfxr_timeline.sh capture.gfxr /tmp/frame-timeline <icd.json>
```

The first replay records queue-submit timestamps without image dumping. The
second dumps every render-target readback and converts it to PNG. The generated
`/tmp/frame-timeline/index.html` references those PNG files (it does not embed
them) and provides a clickable frame-time timeline, range slider, and keyboard
navigation. Set `GFXR_SUBMITS_PER_FRAME` when the application does not use the
HeadlessStreamer's two queue submits per rendered frame. `GFXRECON_REPLAY`,
`GFXRECON_CONVERT`, and `GFXR_FPS_PLUGIN` override the tool paths.

Three things about this are worth knowing before trusting the output:

* **`--dump-resources` addresses commands by block index**, the counter GFXR
  gives every call in the file, and a transfer dump needs three: the copy, the
  `vkBeginCommandBuffer` it records into, and the `vkQueueSubmit` that submits
  that command buffer. `index` recovers all three by matching command-buffer
  ids. It deliberately drops draw calls, which is why its output is a hundred
  times smaller than the capture's command count.
* **Transfer dumps are always raw binary.** GFXR writes image files only for
  image targets; a buffer comes out as `.bin` with no header, hence the `png`
  step. Dumping *draw calls* instead — a `"Draw"` array plus the enclosing
  `RenderPass` indices — makes GFXR write PNGs itself and honour
  `DumpBeforeCommand` and `DumpDepth`, which is the better tool for finding
  which draw first diverges.
* **The manifest suffix is misdocumented.** `vulkan_dump_resources.md` says
  replay writes `<name>_rd.json`; it writes `<name>_dr.json`.

The replay itself needs two flags beyond section 3's: `--remove-unsupported`,
because the application asks for extensions a software driver does not expose
(`VK_KHR_video_maintenance1` here) and `vkCreateDevice` fails outright without
it, and `--log-file`, because gfxrecon's stdout is block-buffered and a driver
that segfaults takes the last few KB of the log with it.

There is no way to stop replay early at a block index, so dumping frame 0 still
streams the whole file. When iterating, kill the replay once the dump lands —
that turns a several-minute run into seconds, since an early frame sits a tiny
fraction of the way into the block stream.

### Why this is worth more than the sample sweep

The sweep covers eighteen samples that were chosen to be small. A real
application exercises paths none of them reach, and it exercises them at a
volume that changes which costs matter. Everything in this list came from the
capture and none of it from the sweep:

* vertex formats narrower than 32 bits per component were never expanded, so
  packed attributes arrived garbled — a hard fault where the value indexed an
  array, silently wrong pixels everywhere else;
* block-compressed copies were measured in pixels rather than blocks, a 16x
  overrun;
* the clipper never clipped `z <= w`, which is the near plane under reversed-Z;
* the A-buffer disabled itself permanently on the first framebuffer resize,
  which no single-resolution sample can trigger;
* consecutive blended draws shared no state at all, because the fragment
  uniform bindings changed every draw — which is what made batching them worth
  44% and is not visible in any sample.

It is also a better cost oracle than the sweep for anything that shows up at
scale, with the caveat in `TESTING.md` about pairing runs under 10%.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `specified memory type index exceeds number of available memory types` | missing `-m remap` |
| `vkCreateInstance returned VK_ERROR_EXTENSION_NOT_PRESENT` | `--wsi` was passed, or the capture used a swapchain |
| `File did not contain any frames` | the application never presented; harmless for offscreen captures |
| Replay stops at some `vkCreate...` call | a genuine gap in this driver — that call is the thing to implement |

That last row is the useful one. Each early failure names one specific, testable
piece of work, instead of a guess about what might be needed.
