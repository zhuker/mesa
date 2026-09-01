# Renderer2 M1 preparation: per-draw state inventory for a scope-level tile walk

Survey of `/home/alexzhukov/mesa-renderer2` at `redesign/tilewalk` (d01e4aa3f02,
base cudavk-0.0.1). Read-only; all citations are file:line in `src/cudavk/`
unless prefixed. Context: `docs/cudavk/notes/RENDERER2_DESIGN.md` (charter),
`docs/cudavk/ARCHITECTURE.md`.

Terminology used below:
- **scope** = one `vkCmdBeginRendering..EndRendering` interval
  (`struct cp_render_scope`, cp_draw_packet.h:25-48).
- **batch** = consecutive mergeable draws submitted as one chain
  (`struct cp_draw_batch`, cp_renderer.h:137-180).
- **episode** = consecutive appendable batches sharing one A-buffer build
  (blended) or one shared visbuf run (opaque); a sub-interval of a scope
  (cp_renderer.h:244-281, cp_renderer.c:7386-7404).

---

## 1. What a batch carries per draw today (`cp_draw_batch` + merge key)

### 1.1 `struct cp_draw_batch` (cp_renderer.h:137-180), field by field

CP_MAX_BATCH_DRAWS = 128 (cp_debug.h:150); CP_MAX_BATCH_TRIS = 256K
(cp_renderer.h:58); CP_ARG_UBO_STRIDE = 16 (kernels/cp_rast_types.h:499);
CP_VB_TABLE_STRIDE = 16 (kernels/cp_rast_types.h:1531).

| field | contents | classification |
|---|---|---|
| `state` (`struct cp_draw_state`, cp_draw_packet.h:55-70) | vs/fs binaries, viewport, raster, depth, blend, pipeline_samples, velem[16], vb_base[16], vs_ubos[16], fs_ubos[16], counts | mixed; see 1.3 |
| `scope` (`struct cp_render_scope`) | fb desc, depth attachment, samples, serial, plan counters | **constant per scope** by construction (cp_renderer.c:9548) |
| `pending, compact_rows, ndraws, tris` | bookkeeping | n/a |
| `key` (`struct cp_batch_key`) | the merge key | see 1.2 |
| `info` (`struct cp_draw_call`, cp_draw_types.h:20-31) | mode, index_size, instance_count, start_instance, index_ptr, indirect | mode/index_size/start_instance constant per batch (in key, cp_renderer.h:88); instance_count superseded by per-draw row |
| `draws[128]` (`cp_draw_range`) | start/count/index_bias per merged draw | **per-draw indexable** — becomes the device slice table `cp_draw_slice[]` (cp_renderer.h:148-150, cp_rast_types.h:481-496) |
| `instance_counts[128]` | per-draw instance count | **per-draw indexable** (cp_renderer.h:151; `cp_draw_slice.verts_per_instance`, cp_rast_types.h:485-489; FLAGS.md:121 `CUDAVK_KEEP_INSTKEY`) |
| `drawid_offset`, `draw_ids[128]` | gl_DrawID per draw | **per-draw indexable** — the parameter rows at args[7], CP_ARG_DRAW_PARAM_STRIDE=4 words {first_vertex, base_instance, draw_id, base_vertex} (cp_renderer.h:152-155, cp_rast_types.h:502-507) |
| `vs_ubos[128*16]` | one row of VS uniform/descriptor pointers per draw | **per-draw indexable** — shader indexes `row * CP_ARG_UBO_STRIDE + binding` through CP_ARG_SLOT_UBO_TABLE/BATCH_ROWS/BATCH_MASK (cp_renderer.h:156-158, cp_rast_types.h:395-423) |
| `blended` | routes to A-buffer vs opaque path | **sort bucket / path selector** (cp_renderer.h:159-167) |
| `fs_ubos[128*16]` | one row of FS uniform pointers per draw | **per-draw indexable** — interpolator writes each shaded slot's row (`out_batch_rows`, cp_rast_types.h:686-707), "every batch — blended or opaque — carries its fragment bindings per draw" (cp_rast_types.h:416-419) |
| `vb_bases[128*16]` | resolved per-element vertex-buffer base per draw | **per-draw indexable** — `cp_vertex_fetch_args.elem_bases` (cp_renderer.h:169-172, cp_rast_types.h:1505-1513; FLAGS.md:126 `CUDAVK_KEEP_VBKEY`) |
| `index_ptrs[128]` | per-draw index-buffer device base | **per-draw indexable** — `cp_draw_slice.ib_base_lo/hi` (cp_renderer.h:173-176, cp_rast_types.h:490-495; FLAGS.md:125) |
| `scissors[128]` | scissor per draw | **per-draw indexable** — `cp_rasterize_args.clip_rects`, one int4 per draw, resolved through draw_slices (cp_renderer.h:177-179, cp_rast_types.h:154-166); but only *blended* draws merge across scissors today (cpvk_cmd.c:2517-2528) |

### 1.2 `struct cp_batch_key` (cp_renderer.h:74-110) — everything a batch may NOT differ in

| key field | classification for a tile walk |
|---|---|
| `vs, fs` | **hard launch boundary** — each is a distinct CUmodule/CUfunction (cp_nir_to_llvm.h:131-166, 188-262) |
| `cbuf_texture, zs_texture, color_data, visbuf, depthbuf, fb_w/h/nr_cbufs/samples, cbuf_format` | **constant per scope** — all come from `packet->scope->fb` (cpvk_cmd.c:2353-2362) |
| `mode, index_size, start_instance, index_resource` | per batch. `mode` is structural (only MESA_PRIM_TRIANGLES batches, cpvk_cmd.c:2406); `index_size` and `start_instance` are launch-wide scalars of the fetch (cp_rast_types.h:1470,1472) — a per-scope table would need them per row; `index_resource` is already free (`CUDAVK_KEEP_IBKEY` off, cpvk_cmd.c:2545-2546) |
| `scissor` | per batch for opaque, per draw for blended (cpvk_cmd.c:2525-2528) |
| `blend_enabled` | **path selector** (opaque visbuf resolve vs A-buffer) |
| `state[768]` | front-end blob `struct cpvk_batch_state` = {pipeline *address*, viewport, fb_samples, fs_ubos[16]} (cpvk_cmd.c:2297-2330). The pipeline address is a proxy for raster/depth/blend/velem/topology/shaders/samples; the *merge* test compares resolved state when addresses differ (cpvk_cmd.c:2501-2515). fs_ubos stay in the key only for an unexplained perf effect (cpvk_cmd.c:2301-2308) |
| `num_vertex_elements, vertex_stride, num_vertex_buffers` | per batch — they shape the fetch gather itself (cp_renderer.h:98-102; elem_* arrays are launch-wide in cp_vertex_fetch_args, cp_rast_types.h:1446-1466) |
| `num_fs_ubos, num_vs_ubos` | per batch — table row *width* (cp_renderer.h:104-107) |
| `sampler_table, num_samplers` | context-global device table; grows on sampler creation (cpvk_image.c:782), read through module globals (cp_renderer.c:3828-3833) |

Deliberately absent from the key (already per-draw): index range, VS bindings,
draw parameters, instance count, vertex-buffer bindings, index buffer,
push-constant contents, per-draw scissor for blended
(cp_renderer.h:60-69; cpvk_cmd.c:2536-2622).

### 1.3 What in `cp_draw_state` is a launch argument vs a kernel identity

- **Kernel identity (hard launch boundary):** `vs`, `fs` binaries; also which
  *execution* of each (classic/fused/inline/hw/vs-fetch, cp_nir_to_llvm.h:70-83)
  and which *sampler variant* (cp_nir_to_llvm.h:168-172, 248-249) and which
  *tune alternate* (`alt_kernel`, cp_nir_to_llvm.h:141-150).
- **Launch arguments (could be table rows):** viewport (vp_* and depth_scale/
  translate in cp_rasterize_args:131-141), raster state (cull_mode/front_face,
  :167-169), depth state (depth_test/func/key_invert, :173-176; and per-fragment
  switch in emit_fragment, cp_rasterize.cu:668-683), blend desc (by value in
  cp_fs_writeback_args:787), scissor (clip_* / clip_rects), vertex layout
  (elem_* arrays).
- **Grid-shaping:** num_vertex_elements, strides, num_*_ubos (table widths).

---

## 2. Fragment shader selection and launch

### 2.1 Path from batch state to CUfunction

1. `cp_shade_fragments()` (cp_renderer.c:3946) picks the in-shader mode
   (cp_renderer.c:4120-4147):
   - hardware texture path available for this state → `cp->hardware_texture.launch_mode`
     (HW_INLINE or HW_FUSED, chosen at cp_renderer.c:3278);
   - else `CUDAVK_INLINE_FS` and an INLINE kernel → INLINE;
   - else a FUSED kernel → FUSED (in-shader interpolation, `cp_fs_compact`
     replaces `cp_fs_interpolate`, cp_renderer.c:4105-4119, 4192-4193);
   - else CLASSIC (standalone interpolator + bare shader).
2. `cp_fs_launch_shader()` (cp_renderer.c:3504) resolves
   `fs->exec[mode]`, with a classic→fused fallback (3524-3533).
3. **Sampler-variant specialization** (cp_renderer.c:3564-3644): for a shader
   with statically traced sampler descriptors, the host dereferences every
   per-draw FS UBO row **on the CPU** through `cp_host_ptr` / the
   `host_maps[]` published at queue submit (cp_renderer.h:114-123, 350-351),
   reads the sampler index at CP_DESC_SAMPLER_INDEX_OFFSET
   (cp_rast_types.h:1182), and requires every row and every descriptor to
   resolve to one identical `cp_sampler_info` (3588-3617). If so it finds or
   JIT-compiles a specialized variant and launches
   `sampler_variant->exec[exec_mode]` instead (3618-3642).
4. A pending register-cap tune can retarget to `launch_exec->alt_kernel`
   at the launch itself (3851-3866).

So the CUfunction is a function of **(fs binary, exec mode, resolved sampler
state across all rows of the batch, tune phase)** — four axes, two of which
(sampler resolution, tune) are decided per launch on the host.

### 2.2 Per-launch FS argument block (cp_renderer.c:3687-3777)

One upload per FS launch: 64 pointer slots + 16-byte scalar area + the
per-draw UBO table (rows × CP_ARG_UBO_STRIDE × 8 bytes) + optionally the
whole `cp_fs_interp_args` block (CUDAVK_INTERP_INLINE, 3694-3699).

| slot | content |
|---|---|
| 0 | covered-pixel counter (device) |
| 2 | fs_in (interpolated varyings) |
| 3 | ptr to fs_in_stride scalar |
| 4 | fs_out |
| 6 | frag_coord |
| 7 | per-draw draw-parameter rows (cp_rast_types.h:502-507) |
| 8 CP_ARG_SLOT_DISCARD | discard mask |
| 9 CP_ARG_SLOT_UBO_TABLE | per-draw UBO row table |
| 10 CP_ARG_SLOT_BATCH_ROWS | per-slot draw row (written by interpolator/fetch) |
| 11 CP_ARG_SLOT_BATCH_MASK | row mask (0 for single draw) |
| 12 CP_ARG_SLOT_FRONT_FACE | per-slot gl_FrontFacing byte |
| 13 CP_ARG_SLOT_COVERAGE | per-slot helper-lane byte, only if `fs->writes_memory` (3734-3737) |
| 14 CP_ARG_SLOT_FUSED_INTERP | device `cp_fs_interp_args*` for fused/inline exec |
| 15 CP_ARG_SLOT_HW_TEX_TABLE | CUtexObject table, hw exec only (3744-3746) |
| 16 CP_ARG_SLOT_VS_FETCH | (vertex stage only) fused-fetch args |
| 18.. CP_ARG_UBO_BASE | row-0 UBO pointers (single-draw degeneracy, cp_rast_types.h:404-408) |

Plus per-module **globals** written by `cuMemcpyHtoD` when stale:
`cp_sampler_table` and `cp_quad_derivs` (cp_renderer.c:3805-3839). Grid:
`min(ceil(threads/256), 4096)` blocks of 256, grid-striding against a
device-side count (3872-3903).

The FS row table is *batch-local*: `cp->fs_batch.ubos` is set at the top of
`cp_draw_execute_batch` (cp_renderer.h:343-363). Only the blended episode's
merged shading groups concatenate rows across batches, via `pass_group_ubos`
and `cp_seg_range.row_base` (cp_renderer.h:283-285, cp_rast_types.h:726-735)
— this is the closest existing thing to a scope-level state table.

---

## 3. Scope and episode boundaries

- Record: `cpvk_CmdBeginRendering` (cpvk_cmd.c:1596) appends a
  `cp_render_scope` with a `serial` to `cmd->scopes[]` (cpvk_private.h:691-694);
  `cpvk_CmdEndRendering` (cpvk_cmd.c:3953) closes it. `cpvk_plan_batches()` at
  vkEndCommandBuffer fills `planned_draws/planned_blended_draws/planned_tris`
  per scope (cp_draw_packet.h:38-47) and precomputes each draw's merge verdict
  (`plan_prev`/`plan_mergeable`, cpvk_private.h:429-441).
- Submit: `cp_render_scope_begin()` (cp_renderer.c:9540-9556): full flush +
  `cp_pass_finish()` + `cp_context_set_framebuffer()` (grow-only buffers) +
  depth attachment LOAD transfer. `cp_render_scope_end()` (9558-9568): full
  flush + finish + depth STORE transfer.
- **Per-scope state:** `cp_fb_desc` (cp_draw_types.h:112-120),
  `cp_depth_attachment` (cp_draw_packet.h:10-22), attachment_samples,
  load/store ops, render-area flags, plan counters.
- **Per-batch state:** everything in section 1. Batch break causes are tallied
  (cpvk_cmd.c:2455-2480); the merge conditions are cpvk_draws_mergeable
  (2483-2625) + same scope serial (2676-2678) + caps (2695-2702).
- **Episode rules:** exactly four state changes may *defer* (keep the episode
  open): a draw whose key broke the batch, and the VS / FS / vertex-elements
  binds (cp_renderer.c:9460-9467, 7397-7404). Everything else —
  framebuffer, viewport, blend, depth, rasterizer state, any readback, clear,
  copy, dispatch — goes through `cp_batch_flush_why()` which also finishes the
  episode (9520-9528). The front-end mirror is
  `cpvk_draws_episode_compatible`: same scope serial, viewport, raster, depth,
  blend, samples (cpvk_cmd.c:2630-2646).
- Episode admission: blended `cp_pass_appendable` (7408-7459: no
  `fs->writes_memory`, A-buffer healthy, ≤ CP_PASS_MAX_SEGS=64 segments,
  prim ids < CP_PRIM_ID_LIMIT=2^30); opaque `cp_opaque_appendable`
  (7542-7575: order-free depth state per `cp_batch_order_free` 7280-7300,
  no FS memory writes, samples==1, exactly one color attachment).
- Each appended batch is snapshotted whole as a segment: `struct cp_pass_seg`
  keeps the full `cp_rasterize_args`, the queue set, and a complete
  `cp_draw_batch` copy (cp_renderer.h:255-268) — i.e. today's "scope table"
  is 64 × ~whole-batch snapshots, host-side.

---

## 4. The post-clip triangle stream

- **Where it lives:** per executed batch, a scratch-arena allocation
  `clipped = max_clipped * 3 * out_stride` bytes, where
  `max_clipped = num_triangles * CP_CLIP_MAX_OUT(=8)` and
  `out_stride = num_slots * 16` (float4 per slot: position + varyings)
  (cp_renderer.c:5930-5945; cp_clip_args, cp_rast_types.h:353-365). The arena
  is rewound per frame/flush generation — nothing persists.
- **Format:** 3 vertices per triangle, `num_slots` float4 each, slot 0 =
  clip-space position (cp_rast_types.h:354, 364). In **stable mode** input
  triangle t owns output slots `t << CP_CLIP_PRIM_SHIFT(=3)`..+7, unfilled
  ones marked degenerate, so the primitive index is monotone in submission
  order — this is what makes A-buffer merging legal (cp_rast_types.h:366-384).
- **Companions:** `out_count` device triangle counter (tri_count,
  cp_rasterize_args:177-179), optional `active_ids` compaction map
  (:180-188), optional `prim_refs` per-primitive pointer table letting
  inside-primitives alias immutable VS output (:120-123, cp_clip_args:360-362).
- **Consumption:** stage1, one thread per triangle, bounds itself against the
  device count, rasterizes small tris (< CP_SMALL_THRESHOLD=16px) inline,
  defers the rest to the nontrivial queue (cp_rasterize.cu:908-942);
  stage2, one warp per nontrivial entry, rasterizes medium tris or decomposes
  huge ones into CP_TILE_SIZE=64 tiles + a setup cache
  (cp_rasterize.cu:944-1174; cp_rast_types.h:1343-1357); stage3, tile pairs
  from the huge queue (cp_rasterize.cu:1176). Queue ABI:
  `cp_rast_queues` (cp_rast_types.h:1400-1415);
  triangle setup is `cp_tri_setup`, 80 bytes: screen coords, ndc z, inv_area,
  fill-rule bits, bbox, point data (cp_rast_types.h:1333-1341).
- **Resolution:** `emit_fragment` packs the visbuf as
  `uint64 = PACK_VISBUF(depth_key_or_prim, prim)` and resolves by atomicMin,
  after per-fragment depth test against the cross-draw `depthbuf`
  (cp_rasterize.cu:645-766; PACK_VISBUF at :68).
- **What a binning pass would read:** exactly what `cp_rasterize_args` carries
  per segment — clipped positions (or prim_refs), device tri count,
  active_ids, viewport transform, cull/front-face, per-draw clip_rects +
  draw_slices, num_samples (cp_rast_types.h:112-267) — and would emit
  (tile, prim, order key). The entry-24 prototype's ref format already exists:
  `cp_opaque_tile_ref {global_prim, segment, flags}` with
  CP_OPAQUE_TILE_SIZE=32, CP_MAX_OPAQUE_TILE_REFS=2M and build/raster arg
  structs (cp_rast_types.h:1008-1041). The M0 census machinery is
  cp_renderer.h:287-318 with the reducer `cp_tile_census_end_pass`
  (cp_renderer.c:8402).

---

## 5. Frank list: state that mutates mid-scope, between batches

Everything below happens *inside* one render scope today and would have to be
a table row, a bucket key, a launch split, or a fallback in a scope-level walk.

1. **Descriptor bindings (UBO/SSBO/texture rows).** Snapshotted at bind into
   the command buffer's descriptor arena; draws point at the copy
   (cpvk_private.h:696-719). Already per-draw rows. Note the *retired arena*
   hazard: outgrown arenas are copied at submit (cpvk_private.h:711-719).
2. **Push constants.** Per-draw upload into the ring; ride as UBO slot
   CPVK_UBO_PUSH_SLOT=0 per stage (cp_shader_abi.h:54; cpvk_cmd.c:2852-2876).
   Already per-draw rows; explicitly not a merge key (cpvk_cmd.c:2602-2622,
   with a known-wrong-sample caveat under CUDAVK_KEEP_PUSHKEY).
3. **Scissor.** Per-draw `clip_rects` exists (cp_rast_types.h:154-166) but the
   opaque merge path still requires equality (cpvk_cmd.c:2525-2528).
4. **Viewport.** Episode-wide today: in the batch state blob (cpvk_cmd.c:2299,
   2516) and in episode compatibility (2641). Mechanically it is only launch
   scalars (cp_rasterize_args:131-141), so a per-draw row is possible but the
   depth transform feeds both raster and gl_FragCoord (cp_rast_types.h:749-753).
5. **Depth state (enable/write/func).** Per-pipeline; episode requires
   equality (cpvk_cmd.c:2643). emit_fragment already switches per launch
   (cp_rasterize.cu:668-683) so a per-ref row is expressible — but the
   *front-to-back early-out* is only sound for the order-free funcs
   (cp_batch_order_free, cp_renderer.c:7280-7300); a mid-scope depth-func flip
   (e.g. LESS→EQUAL decal passes) breaks the walk's sort premise.
6. **Blend state / colormask.** Per-pipeline, by-value in writeback args
   (cp_rast_types.h:787, 334-340); episode requires equality (cpvk_cmd.c:2644).
   Also the opaque/blended *path* split itself (key.blend_enabled).
7. **Shader (VS/FS) changes.** Per pipeline; deferrable for the episode but a
   hard launch boundary for shading (section 2). The FS row table is
   batch-local (cp_renderer.h:353-363); cross-batch concatenation exists only
   in blended groups (pass_group_ubos, cp_renderer.h:283-285).
8. **Sampler-variant resolution.** Host-side CPU deref of descriptor
   snapshots per launch (cp_renderer.c:3577-3644) through host_maps published
   at submit (cp_renderer.h:114-123); can JIT-compile mid-frame (3627-3635).
   Different draws in one scope can resolve to different variants → different
   CUfunctions for the same FS.
9. **Module globals.** `cp_sampler_table` / `cp_quad_derivs` written per
   module when stale (cp_renderer.c:3805-3839; also cpvk_cmd.c:1277-1292 for
   compute). Cheap but it is device state keyed by *module*, and a tile walk
   multiplexing modules re-raises it.
10. **Hardware-texture cache pins and table.** `cp_texture_cache_available`
    consulted per shade (cp_renderer.c:4124-4125), pin/unpin brackets every
    launch (3535, 3548, 3906), table at CP_ARG_SLOT_HW_TEX_TABLE (3744-3746);
    conversions/evictions are device ops between batches.
11. **Mid-pass clears.** vkCmdClearAttachments records a `mid_pass` clear that
    executes between draws (cpvk_private.h:494-500) — observes rendering,
    full flush today.
12. **Copies / barriers / dispatches / events inside an attachment bind.**
    Op kinds CPVK_OP_COPY/FILL/DISPATCH/BARRIER/EVENT_* (cpvk_private.h:556-574)
    all flush; the tile census explicitly counts binds cut by map/copy/flush/
    compute (cp_renderer.h:312-318) — these occur in real captures.
13. **Indirect draws.** Resolved at submit with a full host wait, then treated
    as ordinary draws (cpvk_cmd.c:2880-2895); not mergeable (2529-2533).
14. **Multi-pass shading loops.** Discard (CP_DISCARD_LAYERS=8 repeat passes,
    cp_rast_types.h:342-345, reject/resolved buffers cp_rasterize_args:189-198)
    and the blended peel loop (peel_next/peel_any, :206-216) mutate per-pixel
    state across launches *within one draw*.
15. **Cross-draw depth buffer.** `cp->depthbuf` carries occlusion across draws
    (cp_renderer.h:418-422); visbuf is rebuilt per batch (:393-395). The walk
    replaces both intra-scope, but scope-entry depth comes from the attachment
    LOAD transfer (cp_renderer.c:9551-9555) and must be stored back (9565).
16. **Upload-ring traffic.** Every batch owes argument/row uploads on the
    stream (~11 KiB per segment measured, cp_renderer.c:7609-7627); any
    scope-level table must be uploaded before the walk, not interleaved.
17. **Capacity caps** that shape any table: 128 draws/batch (cp_debug.h:150),
    256K tris/batch (cp_renderer.h:58), 64 segments/episode
    (cp_rast_types.h:321), 16 UBO slots (cp_shader_abi.h:49), 2^30 prim ids
    (cp_rast_types.h:1354), CP_MAX_HOST_MAPS=1024 (cp_renderer.h:123).
18. **MSAA / multiple color attachments / FS memory writes.** Excluded from
    opaque episodes today (cp_renderer.c:7547-7550) and natural fallbacks for
    the walk.

---

## Summary table

| state kind | per-draw indexable today? | tile-walk strategy |
|---|---|---|
| index range / index_bias | yes — slice table (cp_rast_types.h:481-496) | table row (already device-side) |
| index buffer base | yes — slice ib_base (cp_rast_types.h:490-495) | table row |
| instance count | yes — verts_per_instance (cp_rast_types.h:485-489) | table row |
| draw params (BaseVertex/Instance/DrawID) | yes — args[7] rows (cp_rast_types.h:502-507) | table row |
| VS UBO/descriptor/push rows | yes — vs_ubos rows (cp_renderer.h:156-158) | table row (geometry pass unchanged anyway) |
| FS UBO/descriptor/push rows | yes — fs_ubos rows + out_batch_rows (cp_rast_types.h:686-707) | table row; needs scope-wide row_base concat (exists only for blended groups, cp_renderer.h:283-285) |
| vertex-buffer bases | yes — elem_bases rows (cp_rast_types.h:1505-1513) | table row |
| scissor | yes (blended) — clip_rects (cp_rast_types.h:154-166) | table row; extend to opaque |
| vertex layout / element formats / stride | no — launch-wide arrays (cp_rast_types.h:1446-1466) | geometry pass keeps per-batch launches; not the walk's problem |
| topology / index size / start_instance | no — key scalars (cp_renderer.h:88) | geometry-pass launch split |
| viewport (incl. depth range) | no — launch scalars, episode-wide (cpvk_cmd.c:2641) | table row (bin pass applies transform per draw) or scope split |
| cull / front-face | no — launch scalars (cp_rast_types.h:167-169) | table row (bin-pass input) |
| depth func / write / enable | no — launch scalars (cp_rast_types.h:173-176) | table row for the test; **sort-bucket / scope-split** for non-order-free funcs (walk order premise, cp_renderer.c:7280-7300) |
| blend enable + factors + colormask | no — per-batch, path selector | sort bucket (opaque walk vs draw-order walk); factors as table rows in the blended arm |
| FS identity (binary) | no — CUfunction | **launch split**: per-pipeline walk over its tiles, or per-tile shading buckets (the charter's open problem) |
| FS exec mode / sampler variant / tune alt | no — resolved per launch on host (cp_renderer.c:3538-3641, 3863-3866) | must be resolved *before* the walk per scope-table row; sampler-state divergence within a scope forces bucket split or generic-kernel fallback |
| sampler table / quad_derivs module globals | cached per module (cp_renderer.c:3805-3839) | resolve at table build; one write per module used in scope |
| hw texture table + pins | per shade decision (cp_renderer.c:4124-4125) | pin whole scope before walk, or fallback |
| framebuffer / attachments / samples | constant per scope (cpvk_cmd.c:2353-2362) | scope key |
| depth attachment load/store | per scope (cp_renderer.c:9551-9566) | scope prologue/epilogue |
| mid-pass clears, copies, barriers, dispatches, events | n/a — flush ops (cpvk_private.h:556-574) | **scope split** (walk the prefix, fall through or re-open) |
| indirect draws | resolved at submit w/ host wait (cpvk_cmd.c:2880-2895) | scope split at the wait; resolved draws then table rows |
| discard / peel multi-pass | per-pixel loop state (cp_rast_types.h:189-216) | discard: shade-in-walk with per-lane kill (no visbuf pre-resolve needed once the walk owns the pixel); blended: draw-order walk into A-buffer format |
| MSAA, >1 cbuf, FS writes_memory | excluded from episodes (cp_renderer.c:7547-7550) | fallback to existing renderer |

---

## Top-3 hardest problems (my assessment)

1. **FS CUfunction multiplicity is worse than "one per pipeline".** The
   launched kernel = (fs, exec mode, sampler variant, tune alternate), with
   sampler resolution done by CPU-dereferencing every draw's descriptor row at
   launch time (cp_renderer.c:3577-3644) and JIT compilation allowed
   mid-frame. A scope table must freeze all four axes per row at table-build
   time, and rows that resolve differently for the *same* fs must either split
   buckets or drop to the generic sampler kernel — a per-row perf cliff the
   A/B must account for.
2. **The per-draw row ABI is batch-local, not scope-local.** CP_ARG_SLOT_
   BATCH_ROWS/UBO_TABLE rows, draw_slices, elem_bases, clip_rects all index
   within one batch (≤128 draws), and the only cross-batch concatenation
   (row_base + pass_group_ubos) exists for blended shading groups only.
   A scope table needs a scope-global draw id threaded through binning refs
   (cp_opaque_tile_ref.segment+prim is the sketch) and through every slot the
   shader reads — this touches the generated-code ABI (cp_rast_types.h:395-423).
3. **Order-free-ness is a per-draw property but the walk's premise is
   per-tile.** Front-to-back early-out is legal only under the four monotone
   depth funcs with depth write on (cp_renderer.c:7280-7300); real scopes
   interleave opaque, discard, EQUAL-func and blended draws, and mid-scope
   flush ops (copies/clears/dispatches — measured by the census cut counters,
   cp_renderer.h:312-318) cut the scope arbitrarily. The walk therefore needs
   a per-scope *segmentation* into maximal expressible runs, with correct
   depth/A-buffer handoff at each cut — the coverage-vs-win tradeoff the
   charter flags, made concrete.
