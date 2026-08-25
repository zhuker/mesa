# Vulkan video encode support in this tree

Survey taken at Mesa 26.2.0-devel (`VERSION`), branch `cudapipe-perf`. This is a
snapshot of what the tree contains, not a plan.

## Repo-wide

`src/vulkan/registry/vk.xml` knows eight encode extensions. Two drivers
implement any of them: **radv** and **anv**. No other Vulkan driver in the tree
has a single `.KHR_video*` line in its extension table — not nvk, turnip,
panvk, v3dv, pvr, or dozen. `grep -rn '\.KHR_video' --include='*.c' src/`
returns exactly two files.

| extension | radv | anv | everyone else |
|---|---|---|---|
| `KHR_video_encode_queue` | yes | yes | — |
| `KHR_video_encode_h264` | yes | yes | — |
| `KHR_video_encode_h265` | yes | yes (Gfx12+) | — |
| `KHR_video_encode_av1` | yes | — | — |
| `KHR_video_encode_intra_refresh` | yes | — | — |
| `KHR_video_encode_quantization_map` | yes (needs `qp_map`) | — | — |
| `VALVE_video_encode_rgb_conversion` | yes (needs EFC) | — | — |
| `KHR_video_encode_feedback2` | — | — | — |
| `KHR_video_maintenance1` / `maintenance2` | yes | yes | — |

`KHR_video_encode_feedback2` is in the registry and implemented by nobody.

### radv

`src/amd/vulkan/radv_video_enc.c`, ~3.7k lines — the mature implementation.
VCN encoder, firmware interface handling up to VCN 5.

Enabled by default when the VCN_ENC IP reports `write_memory` support
(`radv_probe_video_encode`, `radv_video_enc.c:68`); otherwise it needs
`RADV_EXPERIMENTAL=video_encode`. `RADV_DEBUG=novideo` disables it outright.
Per-codec bits are further gated on `pdev->info.video_caps.enc[...]`
(`radv_physical_device.c:818-827`).

### anv

`src/intel/vulkan/genX_cmd_video_enc.c`, ~2.6k lines. Off by default and
narrower than radv:

- requires `ANV_DEBUG=video-encode` (`anv_instance.c:19`), and
- requires `verx10 < 125` (`anv_physical_device.c:140`) — Gfx12.0 and older
  only, so DG2 and Xe2 are excluded.

H.265 encode additionally needs `ver >= 12`.

### Build gating

The `video-codecs` meson option (`meson.options:717`) defaults to `all_free`,
which expands to `av1dec, av1enc, vp9dec, mpeg12dec, jpegdec`
(`meson.build:445`). H.264 and H.265 encode are therefore compiled out of a
default build everywhere; the only encode codec a default build actually gets
is AV1, on radv.

### Shared infrastructure

`src/vulkan/runtime/vk_video.c` (3.4k lines) provides codec-agnostic bitstream
header writers usable by any driver: H.264 SPS/PPS/slice header, H.265
VPS/SPS/PPS/slice header, AV1 sequence header
(`vk_video.h:351-406`). Everything below that — sessions, DPB management, the
encoder itself — is per-driver.

## Encode source formats

`VK_KHR_video_encode_queue` does not fix the format of the encode source image.
The valid set is whatever the implementation reports from
`vkGetPhysicalDeviceVideoFormatPropertiesKHR` for the given profile with
`VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR`. The profile carries
`chromaSubsampling` (monochrome / 420 / 422 / 444, `vk.xml:13835`) and
independent luma and chroma bit depths of 8, 10 or 12, so 4:2:2, 4:4:4,
monochrome and 12-bit sources are all expressible where the hardware and codec
profile allow.

What the two drivers here actually report is much narrower — 4:2:0 two-plane
only:

- **anv** (`anv_video.c:553-570`): `G8_B8R8_2PLANE_420_UNORM` at 8-bit and
  `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` at 10-bit, unconditionally. No
  12-bit path.
- **radv** (`radv_video.c:876-889`): those two plus
  `G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16` at 12-bit, each gated on an
  `ac_video` capability bit (`fmts.nv12`, `fmts.p010`, `fmts.p012`), so a given
  ASIC may report fewer. Plus RGBA formats when the VALVE extension below is
  engaged.

The 3-plane YCbCr formats (`G8_B8_R8_3PLANE_420_UNORM` and its 10/12/16-bit
siblings, `vk_format.c:768-844`) exist and are core since Vulkan 1.1, but no
video driver in this tree reports them for any video usage — VCN and VDBOX want
the interleaved CbCr plane. They are reachable only through the sampler-YCbCr
path.

## VK_VALVE_video_encode_rgb_conversion

Worth a section because it is the only encode extension in the tree that no
other vendor implements, and because it changes the format rules above.

### What it is for

Without it, an app whose frame is already an RGB swapchain image — a
compositor, a game-streaming capture path — must run its own RGB→YCbCr 4:2:0
pass before every encode: a dispatch plus a second full-size image per frame.
AMD's VCN has a format-conversion front end (EFC, "encoder format conversion")
that ingests RGB directly and does the matrix and chroma downsample on the way
into the encoder. The extension exposes that block.

### API shape

Four structures, no new entry points (`vk.xml:27881`):

1. Feature bit `VkPhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE::videoEncodeRgbConversion`.
2. Opt in *at the profile level*: chain
   `VkVideoEncodeProfileRgbConversionInfoVALVE{ performEncodeRgbConversion = TRUE }`
   onto `VkVideoProfileInfoKHR`. Being part of the profile, it changes every
   profile-dependent query — in particular the encode-source format list becomes
   RGB instead of NV12.
3. Query `VkVideoEncodeRgbConversionCapabilitiesVALVE` (chained to
   `VkVideoCapabilitiesKHR`) for the supported conversions: `rgbModels` (RGB
   identity, YCbCr identity, 601, 709, 2020), `rgbRanges` (full/narrow),
   `xChromaOffsets` and `yChromaOffsets` (cosited-even/midpoint). These mirror
   `VkSamplerYcbcrConversionCreateInfo`, hence the extension's dependency on
   `VK_KHR_sampler_ycbcr_conversion`.
4. Pick one value from each mask at session creation via
   `VkVideoEncodeSessionRgbConversionCreateInfoVALVE`. Fixed for the session.

Encoding is otherwise unchanged; the picture resource just points at an RGB
image.

The common runtime only latches the choice into `vk_video_session`
(`vk_video.c:124-133` → `vk_video.h:94-99`); all real work is per-driver.

### radv implementation

- **Advertised** (`radv_physical_device.c:981`) when
  `video_caps.enc[AC_VIDEO_CODEC_AVC].efc` is set; EFC is
  `vcn_ip_version >= VCN_2_0_0 && != VCN_2_2_0` (`ac_video.c:411`). radv checks
  only the AVC entry although the flag is set identically for HEVC and AV1.
- **Capabilities** (`radv_video.c:592`) are narrower than the extension allows:
  models 709 and 2020 only, both ranges, x-chroma cosited-even only, y-chroma
  either.
- **Format query** (`radv_video.c:842`): with the RGB profile struct present and
  `ENCODE_SRC` usage, the list becomes `B8G8R8A8_UNORM`/`R8G8B8A8_UNORM` at
  8-bit and `A2B10G10R10`/`A2R10G10B10` at 10-bit.
- **Encode time**: `radv_enc_input_format` (`radv_video_enc.c:2423`) describes
  the source as `COLOR_SPACE_RGB`, `CHROMA_SUBSAMPLING_4_4_4` with the packing
  format from the Vulkan format, writing input colour range and chroma location
  as 0 ("ignored for RGB"); `radv_enc_output_format` (`radv_video_enc.c:2491`)
  programs colour volume, colour range and chroma location from the session's
  parameters. That second packet is where the app's choices reach the hardware.

The model mapping is coarse: `rgbModel` selects a *colour volume*, so 709 →
`G22_BT709` and 2020 → `G2084_BT2020` (`radv_video_enc.c:2384`) — the YCbCr
matrix and the transfer function are conflated, and anything outside those two
models hits `UNREACHABLE`. Range and chroma-location mappings are direct.

Status: vendor extension, spec version 1, not ratified, radv-only in this tree.

## llvmpipe / lavapipe

None. Not partial — absent.

`grep -i video` over `src/gallium/frontends/lavapipe/` and
`src/gallium/drivers/llvmpipe/` returns nothing at all. Consequences:

- no `KHR_video_queue`, so there is no decode support either;
- no `VK_QUEUE_VIDEO_ENCODE_BIT_KHR` queue family is ever advertised;
- llvmpipe has no gallium `pipe_video_codec` implementation to build on.

Every encoder in the tree drives a hardware block (AMD VCN, Intel VDBOX/HuC).
Mesa's Vulkan side has no software-encode path that lavapipe could switch on.
Adding it would mean writing the video session and session-parameters objects,
an encode queue, and a software encoder from scratch; the header writers in
`vk_video.c` are the only part that comes for free.
