# Capturing and replaying real applications with GFXReconstruct

Guessing which features a driver needs — from what a class of hardware exposes,
or from what a test suite happens to cover — has been wrong repeatedly. This
measures it instead.

Capture a real application on a driver that works, then:

* **read what it required** — every image format and usage, sampler state,
  pipeline state, which shader stages appear at all, and which compressed
  texture family the content actually ships;
* **replay it against cudapipe** — a frame captured on real hardware becomes a
  regression test for this driver, without the application needing to run here
  at all.

The second is the valuable one. Every substantial bug found so far came from
running something realistic and diffing against hardware, not from a test suite.

Everything below was run end to end on this machine: a capture of the offscreen
benchmark taken on an RTX 5090 replays successfully on cudapipe.

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

Meson and ninja come from the venv described in `CUDAPIPE_HANDOFF.md`; put it on
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

**If the application runs inside Flatpak** (Roblox via Sober, for example) the
environment has to be set inside the sandbox, and the layer must be on a path
the sandbox can see:

```bash
flatpak run \
  --filesystem=$GFX/layer --filesystem=/tmp \
  --env=VK_LAYER_PATH=$GFX/layer \
  --env=VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_gfxreconstruct \
  --env=GFXRECON_CAPTURE_FILE=/tmp/app.gfxr \
  <flatpak app id>
```

That form is untested here — no Flatpak application was available — so treat it
as a starting point rather than a known-good recipe.

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

## 3. Replay against cudapipe

```bash
VK_DRIVER_FILES=<path to>/cudapipe_devenv_icd.x86_64.json \
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

## Troubleshooting

| Symptom | Cause |
|---|---|
| `specified memory type index exceeds number of available memory types` | missing `-m remap` |
| `vkCreateInstance returned VK_ERROR_EXTENSION_NOT_PRESENT` | `--wsi` was passed, or the capture used a swapchain |
| `File did not contain any frames` | the application never presented; harmless for offscreen captures |
| Replay stops at some `vkCreate...` call | a genuine gap in this driver — that call is the thing to implement |

That last row is the useful one. Each early failure names one specific, testable
piece of work, instead of a guess about what might be needed.
