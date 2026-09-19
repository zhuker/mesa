# Ordered-clip native compile/cache prerequisite

`cpvk_clip_shader_cache` links the actual static driver. It does not use a fake
cache, a stub backend, or the Vulkan loader. It inspects private pipeline and
device objects. GNU ld `--wrap=cp_compile_nir_to_ptx` counts the actual backend
boundary and forwards the unchanged call to the real backend.

## Safe CPU preparation

From this worktree:

```
/home/alexzhukov/mesa/venv/bin/ninja -C build-cudavk src/cudavk/cpvk_clip_shader_cache
/home/alexzhukov/mesa/venv/bin/python src/cudavk/tests/cp_clip_shader_fixtures.py /tmp/clip-fixtures
```

The script compiles a VS and all eight fragment exclusion combinations. It runs
`spirv-val` and checks `spirv-dis` output for exact FragDepth, SampleMask and
EarlyFragmentTests presence. The manifest records expected masks. It does not
load Vulkan or CUDA. It can also supply the fixtures to shader-derived NIR tests.

**Building the native executable does not run it. Do not run it during CPU-only
validation.** It is deliberately not registered as a Meson `test()`.

## Separately authorized device compilation

This command is documentation, not authorization:

```
build-cudavk/src/cudavk/cpvk_clip_shader_cache --allow-device-compilation /tmp/clip-fixtures/clip.vert.spv /tmp/clip-fixtures/clip-1.frag.spv 1
```

Repeat for masks 0 through 7. Repeat each command with `--race`. Every invocation
starts with an empty device shader cache. The test creates a real Vulkan device,
shader modules, render pass, layout and graphics pipelines. CUDA device setup and
real shader JIT occur. No images, command buffers, draws or submissions are made.
The explicit command-line acknowledgement prevents accidental no-argument runs
from touching a device. Use an external timeout for authorized race runs.

The normal case checks a cold compile, an early cache hit, exact reason bits,
valid standalone FS execution, shared owner identity and one FS cache entry. It
then destroys both pipelines and creates another pipeline, checking that the
same device-owned binary and metadata survive without another backend call.
Production policy checks cover new-only admission, flag off and both old reasons.

The race case blocks each FS caller after the real backend returns and before
cache publication. Both callers must reach this point; neither can publish while
the other still has an opportunity for an early hit. This forces the real
race-winner return path without relying on timing. Worker calls enter
`cpvk_CreateGraphicsPipelines`, whose `CPVK_CTX_SCOPE(dev)` pushes and restores
the device CUDA context on each thread.

The executable prints the execution modes present on the verified final owner.
Absent modes are reported as skipped, not covered. Run configurations in separate
processes because the debug registry is immutable after device creation. The
fixtures have no textures, so they do not establish hardware-texture owner
attachment coverage. Mode presence is not evidence of execution selection,
runtime fallback, shader-output correctness, or support for excluded semantics.
A compile failure is a test failure; never report it as exclusion coverage.

There is no serialized compiled-shader cache on this path. Fresh processes are
fresh compilations, not disk-cache persistence tests. Vulkan pipeline-cache
export/import is not tested or claimed here.
