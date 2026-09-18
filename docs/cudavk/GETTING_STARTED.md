# cudavk 0.0.3 — build, install and use

This is the quickstart for a fresh machine or a fresh session. It describes
the driver exactly as tagged `cudavk-0.0.3`. The deeper documents are indexed
in `../../CUDAVK.md`; this page only gets you from a clone to a rendering
Vulkan application.

cudavk is a CUDA implementation of Vulkan exposed as a standard ICD. It is
loaded through the Vulkan loader by pointing `VK_DRIVER_FILES` at a generated
manifest; nothing is installed system-wide and nothing touches the machine's
real GPU driver stack.

## What it needs

| dependency | known-good | notes |
|---|---|---|
| NVIDIA driver | 580.x | any CUDA-capable GPU; sm_120 (RTX 5090) and sm_100 (B200) are the validated targets |
| CUDA toolkit | 12.8 and 13.0 | both build; a version guard covers the CUDA 13 `cuCtxCreate` signature change |
| LLVM + clang | 18.1.3 exactly matched | `llvm-18-dev` **and** `clang-18`. Without the matching clang the build silently loses the fused vertex-fetch bitcode, a measured performance default |
| meson | >= 1.4 | Ubuntu 24.04's 1.3.2 is too old; `pip install meson` works |
| ninja, gcc/g++, bison, flex, pkg-config | distro | |
| glslang-tools | distro | `glslangValidator` must be on PATH |
| python3 + mako, pyyaml | distro | code generation |
| libdrm, expat, zlib, zstd, libelf dev packages | distro | mesa core |
| Vulkan loader (`libvulkan1`) + headers | >= 1.3 | to run apps; mesa vendors its own headers for the build |

On Ubuntu 24.04 the whole list is:

```bash
sudo apt-get install -y ninja-build gcc g++ bison flex pkg-config \
  llvm-18-dev llvm-18 clang-18 glslang-tools spirv-tools \
  libdrm-dev libexpat1-dev zlib1g-dev libzstd-dev libelf-dev \
  python3-mako python3-yaml libvulkan-dev vulkan-tools zstd
sudo pip3 install "meson>=1.4"
```

## Build

```bash
git clone https://github.com/zhuker/mesa.git -b cudavk-0.0.3
cd mesa
meson setup build -Dcudavk=true \
  -Dgallium-drivers=llvmpipe -Dvulkan-drivers=swrast \
  -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled \
  -Dbuildtype=debugoptimized
ninja -C build
```

The build must finish with **zero cudavk warnings** from meson setup. If you
see `cudavk: matching clang-18 not found`, stop and install `clang-18`: the
build will succeed but ship without the inlined vertex fetch.

The products are:

- `build/src/cudavk/libvulkan_cudavk.so` — the driver
- `build/src/cudavk/cudavk_devenv_icd.x86_64.json` — the ICD manifest that
  points the Vulkan loader at that exact `.so`

## Install

There is nothing to install. The manifest is the installation:

```bash
export VK_DRIVER_FILES=$PWD/build/src/cudavk/cudavk_devenv_icd.x86_64.json
```

Every Vulkan application launched with that variable uses cudavk; every one
launched without it uses whatever the system had. Multiple builds coexist by
pointing at different manifests.

## Verify

```bash
# the device enumerates and identifies itself
VK_DRIVER_FILES=... vulkaninfo --summary | grep -A2 cudavk

# every flag the driver resolved in this process, with defaults
CUDAVK_HELP=1 VK_DRIVER_FILES=... vulkaninfo --summary >/dev/null

# the full no-replay test suite (needs the GPU, ~2 minutes)
ninja -C build test
```

The suite must pass everything. As of this tag it is 79 tests plus CPU-only
census validators.

## Use

Run any Vulkan application with `VK_DRIVER_FILES` set. The driver is
headless-oriented: WSI platforms are compiled out in the configuration above,
so applications that render offscreen (or through
`VK_EXT_headless_surface`-style paths) are the target shape. gfxreconstruct
`tocpp` compiled replays run unmodified.

Performance defaults are all on; **no environment variable is needed for
speed**. Every performance feature has a `CUDAVK_NO_*` revert switch and every
diagnostic is default-off — the complete registry is `src/cudavk/FLAGS.md`,
generated from the single source of truth in `src/cudavk/cp_debug.c`. Two
rules from that file worth knowing before setting anything:

- presence flags are enabled by *existing* (`CUDAVK_DEBUG_DRAW=0` turns
  tracing **on**); value flags read the value. `CUDAVK_HELP=1` says which is
  which.
- a stale `CUDAVK_*` variable in the shell silently changes behavior; the
  driver warns on retired names but not on live ones.

## State of this tag

Reference medians for the two HeadlessStreamer compiled replays, paired-submit
frame time over real frames, all defaults, clean environment:

| capture | RTX 5090 (sm_120, CUDA 12.8) | B200 (sm_100, CUDA 13.0) |
|---|---:|---:|
| favorite3 | **5.176 ms** | **8.529 ms** |
| favorite2 | **4.867 ms** | not re-measured for 0.0.3 (0.0.2: 8.283 ms) |

The RTX results come from strict alternating measurements against 0.0.2:
favorite3 improved from 5.909 to 5.176 ms (-12.4%), and favorite2 improved
from 5.043 to 4.867 ms (-3.5%). Their heavy-band medians improved from 7.249
to 6.023 ms (-16.9%) and from 5.598 to 5.054 ms (-9.7%), respectively. The
B200 favorite3 result is a three-round, six-arm identical-control measurement;
favorite2 has not been re-measured on B200 since 0.0.2.

Rendering output is bit-identical across the two architectures on the measured
captures. The B200 gap is clock-bound arithmetic on small kernels, not a
defect; `docs/cudavk/PERFORMANCE.md` carries the measured cost of every
default and `docs/cudavk/DEAD_ENDS.md` the measured cost of everything that
was tried and rejected. Read those before attempting to make it faster, and
`docs/cudavk/WORKFLOW.md` before quoting any number.

Known gaps and unfinished work are tracked in `docs/cudavk/TODO.md`.
