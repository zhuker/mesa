# Conformance: what was fixed, what is left

Second priority behind HeadlessStreamer performance — see
`HEADLESS_STREAMER_PERF.md` for that work and for the shared measurement rules.

Everything here came out of a week spent making one real application replay.
None of it was the goal; all of it was found on the way, mostly because a green
18-sample sweep kept certifying changes it could not see.

Branch `cudapipe-gfxr-replay`. dEQP is at `~/VK-GL-CTS/build`; the binary is
`build/external/vulkancts/modules/vulkan/deqp-vk`, run with
`--deqp-surface-type=fbo` and one `--deqp-case` group per invocation
(`--deqp-case` cannot be repeated).

---

## Fixed

### `gl_FrontFacing` was undef — `a45bb27041a`

`nir_intrinsic_load_front_face` had no case in the NIR→PTX backend and fell to
the default arm, which yields `undef` and propagates it through everything
downstream.

`dEQP-VK.glsl.builtin_var.frontfacing` **3/25 → 17/25**, the whole
`builtin_var` group 10 pass → 28. Later `f9a6fb05d79` took frontfacing to
**20/25**.

The facing is recovered for free: `setup_triangle` discards the sign of the
area but `cp_interp_pixel` recomputes it and records the vertex swap in `vidx`.
Shaders that do not read the builtin pay nothing — `reads_front_face` gates the
allocation, the same way `uses_discard` does.

### Cull mode `FRONT_AND_BACK` culled nothing — `f9a6fb05d79`

`cp_cull_mode()` returned 0 ("keep everything") with a comment saying the draw
is skipped instead. No such skip existed anywhere; `grep PIPE_FACE_FRONT_AND_BACK`
found the comment and no code. A draw that should render nothing rendered in
full. Now skipped in `cp_draw_vbo`, points exempt (no winding, and the
rasterizer already ignores culling for them — `point_list.front_and_back`
passed before and after).

Fixed the three `frontfacing.*.front_and_back` cases exactly.

### Clears ran on the host, in the wrong rectangle, at the wrong level — `a972f56e41d`

Three bugs in one area:

- `cp_clear_texture` and `cp_clear_depth_stencil` cleared with a scalar host
  loop calling `memcpy` per pixel — 921,600 calls for a 720p clear — *and* raced
  in-flight kernels, because `cp_batch_flush()` only submits. Now kernels on
  `cp->stream`, which supplies the missing ordering edge.
- `cp_clear_render_target` took `dstx`/`dsty` and **dropped them**, so every
  partial `vkCmdClearAttachments` cleared the corner.
- `cp_clear_texture` ignored `mip_offsets[level]`, so clearing level 1 wrote
  over level 0 and left level 1 untouched.

Plus: a stencil-only clear used to zero the whole depth buffer.

| group | before | after |
|---|---|---|
| `api.image_clearing.*` | 4,482 pass / 8,684 fail | **12,082 / 1,084** |
| `core.partial_clear_*` | 0 / 216 | **92 / 124** |
| `core.clear_color_image.*` | 2,124 / 3,739 | **5,824 / 39** |
| `renderpasses.renderpass1.dedicated_allocation.*` | 560 / 1,108 | **800 / 868** |

`occlusionquery` 273 → 720 fps and `oit` 292 → 414 as a side effect. No case
moved pass → fail.

### Fragment shaders never ran without a colour attachment — `0b072176132`

The shading stage was gated on `if (color_data)`, whose comment assumed "no
colour attachment" meant "depth-only pass". Vulkan permits a fragment shader
that exists purely for side effects — storage-image and SSBO writes. Any such
pass silently did nothing: OIT, visibility buffers, GPU-driven culling,
occlusion feedback.

`oit` went from 4 distinct colours to **74**, matching release llvmpipe to 26
pixels of 921,600. `dEQP-VK.fragment_operations` **+2**, both
`early_fragment.*_depth_no_attachment`.

Side-effect passes go through the peel loop, because visibility is otherwise
resolved to one fragment per pixel and only the nearest layer would be
appended.

### Block-compressed copies overran — `09dcd15df54`, `f9a6fb05d79`

`util_format_get_blocksize` is bytes per *block*; the box is in *pixels*. A BC1
copy walked 16x its extent. Live in `cp_resource_copy_region` (it segfaulted on
the capture); latent in `cp_blit`, fixed for consistency.

### The driver now says when it fails — `1c0ad4094fa`

Not a conformance fix, but it is why several of the above were findable.
`CP_CU_WARN` reports a CUDA failure once per site; `CP_LAUNCH` wraps the 25
previously-unchecked launches; and the unimplemented-intrinsic warning fires
**always**, once per intrinsic, rather than behind `CUDAPIPE_DEBUG_SHADER`.

That last one is the lesson of this whole list. The `gl_FrontFacing` message
existed the entire time, behind a variable nobody sets until they already
suspect the shader. Diagnosing `oit` afterwards took four commands, because the
absence of that warning eliminated the entire "missing shader feature"
hypothesis immediately.

---

## Left to do

### 1. Line primitives are rasterized as triangles

`cp_build_vertex_refs` (`cp_context.c:626-680`) has cases for points, strips
and fans, then falls through to `count/3` triangles for everything else — so a
`LINE_LIST` is assembled as triangles.

Fails five cases: `builtin_var.frontfacing.add_ubo_load.line_list.{back,front,
front_and_back,none}` and `none.line_list.simple`. Almost certainly a much
wider set under `dEQP-VK.rasterization.primitives.*` — **run that group first;
nobody has**.

A feature gap, not a bug in existing code. The capture never draws lines
(`headless_streamer_requirements.txt`: `TRIANGLE_LIST`, `TRIANGLE_STRIP`,
`POINT_LIST` only), so this is pure conformance.

### 2. `dynamicrenderinglocalread` renders 91% wrong

840,443 of 921,600 pixels differ from NVIDIA, identically on builds either side
of `0b072176132`, so it is pre-existing and unrelated to recent work. Not in the
18-sample sweep, which is why nothing caught it.

**Worth more than its dEQP score suggests**: it is dynamic rendering, which the
capture's renderer could plausibly adopt, so this is the one conformance item
with a route back to the primary goal. Start with
`dEQP-VK.dynamic_rendering.*` and `dEQP-VK.renderpass2.*`.

### 3. `clear_depth_stencil_image` still fails 62 cases

Needs mip/layer biasing, but that entry point is shared with the attachment
clear — and the draw path renders to level 0 / layer 0 with no bias. Honouring
level/layer in the clear while the draws do not would land the clear somewhere
the draws are not, and read as the clear breaking.

**So this needs the draw path biased first** (`cp_context.c:2897` and the
sampler-view path at `:5823` are where `mip_offsets` is and is not used), then
the clear made to match. Doing either half alone is a regression.

### 4. Tessellation and geometry stages are unimplemented

`terraintessellation` produces no output at all. Expected — the capture needs
vertex + fragment only — but it is a hole in any general conformance claim.

### 5. Pre-existing dEQP infrastructure failure

`image.store` and `image.atomic_operations` abort partway through on a sticky
`cuMemAllocManaged … CUDA_ERROR_MISALIGNED_ADDRESS (716)` surfacing as
`VK_ERROR_OUT_OF_DEVICE_MEMORY`. Present in builds either side of recent work.
Runs currently need a resume loop that restarts past the aborting case, which
makes those two groups' numbers noisy (24-28 cases move between two runs of one
binary). **Worth fixing before trusting either group** — a 716 from an
allocation is a real bug in its own right.

### 6. A narrowing that is owed

`0b072176132` added a helper-invocation early return: a memory-writing fragment
shader returns before its body when its coverage byte is zero, which forfeits
helper-lane derivatives. A shader that both writes memory *and* samples with
implicit LOD would change. Nothing available does both, so it is unverified
rather than known-broken. The correct guard is "no implicit-LOD sampling in this
shader", which the backend already knows at codegen time.

---

## How to test, and why the sweep is not enough

`tests/TESTING.md` is the methodology. Two things specific to conformance work:

**The 18-sample sweep is a regression guard, not evidence.** It is
`headless_streamer_samples.txt` — one sample per capability the capture needs —
so it is silent on anything outside that. Measured: no swept sample reaches
`cp_clear_texture` or `cp_clear_depth_stencil`, reads `gl_FrontFacing` or
`gl_FragCoord`, writes memory from a fragment shader, or issues a partial
`vkCmdClearAttachments`. Five separate times a green sweep meant "did not touch
it".

So for each change, find the thing that exercises it. It is one lookup:

| path | what exercises it |
|---|---|
| `cp_clear_texture` | `oit` (171 calls/frame) |
| `cp_clear_depth_stencil` | `occlusionquery` (172 calls/frame) |
| `gl_FrontFacing` | `offscreen` — the only shader in the tree that reads it |
| `gl_FragCoord` | `ssao`, `subpasses`, `terraintessellation`, `dynamicrenderinglocalread`, `oit` |
| memory-writing fragment shader | `oit/geometry.frag` — the only one |
| packed vertex attributes | `imgui` — the only sample binding an 8-bit attribute |
| reversed-Z | **nothing** — no sample uses `VK_COMPARE_OP_GREATER` |

**`renderheadless`'s sweep row is vacuous.** Its stored directory is empty, so
is the reference, `verdict.txt` says `missing`, and a diff of two empty
directories reports IDENTICAL. Check it by hand:

```sh
cd $(mktemp -d) && VK_ICD_FILENAMES=$OLD_ICD ~/git/Vulkan/build/bin/renderheadless
cd $(mktemp -d) && VK_ICD_FILENAMES=$NEW_ICD ~/git/Vulkan/build/bin/renderheadless
md5sum */headless.ppm
```

**Samples worth adding to the sweep**, in order: `imgui` (packed vertex
formats — the crash that blocked everything, and the sweep's only possible
coverage of it), `offscreen`, `occlusionquery`, `oit`. Reversed-Z has no sample
and would need one written.
