# Experimental combined candidate (not accepted)

- Worktree: `/home/alexzhukov/mesa-three-ms-combined`
- Branch: `perf/three-ms-combined`
- Baseline: `69a58a486d4`
- Final commit: `bf475e0f61ab9b6de1b0a574ad558fde3e6ffa2e`
- Optimization cherry-picks only: `4e374975bf3` → `288ee336b82`; `ded250960ee62c` → `90cd0b2ecf3`; `cad04fff89d` → `8d0bebcb4b4`.
- All applied without conflicts. Final docs-only commit fixes the generated registry count from 155 to 157.
- No validation-only commits, stream balancing, order, register retry, or lazy prototype changes were added.
- Original worktree WIP was not edited or built. No GPU commands or device tests ran.

## Build

Shared tools: `/home/alexzhukov/mesa/venv/bin`, SDK tools `/home/alexzhukov/vulkan-sdk/1.4.357.1/x86_64/bin`.

```
/home/alexzhukov/mesa/venv/bin/meson setup build-cudavk -Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers= -Dglx=disabled -Degl=disabled -Dplatforms= -Dgbm=disabled -Dbuildtype=debugoptimized
/home/alexzhukov/mesa/venv/bin/ninja -C build-cudavk -j 8
```

Both exit 0. Final post-commit ninja also exits 0.

- ICD: `/home/alexzhukov/mesa-three-ms-combined/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json`
- Library: `/home/alexzhukov/mesa-three-ms-combined/build-cudavk/src/cudavk/libvulkan_cudavk.so`
- Library SHA-256: `9f33f7039b70200acf5596858c8adf5017a1aa3242f5f31443a2731c92dc42fa`
- Logs: `combined-setup.log`, `combined-build.log`, `combined-final-build.log`.

## Flag and cache review

All three switches are independent `CP_FLAG_BOOL_VALUE` registry entries with implicit zero defaults. Explicit `=0` is off; `=1` is on:

- `CUDAVK_SPECIALIZE_VFETCH`
- `CUDAVK_CLIP_COMMON_REJECT`
- `CUDAVK_CLIP_COPY_BYPASS`

`cp_kernels.c:compile_cuda_source` adds the two clip defines independently, only to `cp_rasterize.cu`. Every option string is serialized into its NVRTC disk cache key. The combined option array has 11 slots: its maximum is 11 (architecture, standard, both clip options, relocatable, instrumentation, PDL, four threshold/bounds options).

`cpvk_pipeline.c:cpvk_compile_stage` hashes the static-layout tag, count, and every fetch element field into the shader key when a layout is supplied. `CUDAVK_SPECIALIZE_VFETCH=0` supplies NULL. Unsupported extension/dynamic input state declines specialization. Clip switches do not select vertex specialization and vertex specialization does not select either clip switch.

The original `clip_dist`, `clip_lerp`, and `clip_poly` arithmetic remains unchanged. No fast-math or arithmetic helper optimization was added by these commits. Exact identification of any other originally disabled arithmetic helper paths was requested from the parent; that specific check is not yet claimed.

## CPU gates

PASS: `cp_debug_doc.py --check` after regeneration; `cp_no_getenv.py` (39 driver sources).

Meson focused run: 5/5 passed, 0 failures:

- `cp_shade_equiv_test`
- `cp_vfetch_layout_cpu`
- `cp_launch_audit`
- `cp_clip_common_reject`
- `cp_clip_copy_bypass`

Logs: `combined-registry.log`, `combined-no-getenv.log`, `combined-cpu-gates.log`. The vfetch CPU build emitted non-fatal missing-prototype warnings. The clip tests compile exact source as host C++; they are not GPU float-behavior proof and individually compare each clip switch, not combined device execution.

No baseline named Meson test registration was removed. Three named CPU registrations were added. No test file was deleted. Existing vfetch gate states were preserved and two specialized states were added.

## Pending

This is an EXPERIMENTAL candidate only. No acceptance or retention decision is made. Independent external standing exceptions and full device gates remain pending. Combined performance must be measured directly: all three explicitly off versus all three explicitly on. Do not add isolated gains. No combined timing or combined rendered-output claim is made here.
