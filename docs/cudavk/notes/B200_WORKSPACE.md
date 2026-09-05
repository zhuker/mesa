# The B200 workspace: what survives a restart, and what to reinstall

That host is a Coder workspace reached through two ssh hops
(`ssh azhukov@192.168.1.154 ssh b200`). It has been recreated three times
during this campaign, twice in the middle of an experiment.

**Only `$HOME` survives.** Not `/tmp`, not `/opt`, and **no apt package**.

## The failure it produces, which does not look like a missing package

Every run aborts with `rc=134` before writing a single timestamp, and the A/B
tool reports failed gates on both arms:

```
vkCreateInstance ... returned -9                      (VK_ERROR_INCOMPATIBLE_DRIVER)
ERROR: [Loader Message] libLLVM.so.18.1: cannot open shared object file
ERROR: [Loader Message] loader_icd_scan: Failed loading library ... Ignoring this JSON
ERROR: [Loader Message] vkCreateInstance: Found no drivers!
```

The driver build, both harnesses, the shim and the sentinels are all still
there and **nothing needs rebuilding** -- the ICD simply cannot load, because
cudavk links LLVM 18 for its NIR-to-PTX backend and that library came from a
package. Read this before debugging an experiment; twice it has looked like a
broken measurement and been an empty package database.

## The fix: a library stash in `$HOME`, so no apt is needed at all

`$HOME/lib` holds the two libraries the run path needs, copied with `cp -L`:

| file | size | why |
|---|---:|---|
| `libLLVM.so.18.1` | 123 MB | the ICD links it; without it the loader skips the JSON |
| `libvulkan.so.1` | 0.5 MB | the loader itself |

and `$HOME/bin` holds `numactl` and `taskset`. Then every run only needs:

```bash
export LD_LIBRARY_PATH=$HOME/lib
export PATH=$HOME/bin:$PATH
```

Verified: with the stash and nothing reinstalled, the ICD reports
`deviceName = cudavk (NVIDIA B200)`.

`src/cudavk/tests/cp_b200_setup.sh` does this and checks it. Run it after any
recreation; it exits without touching apt if the stash is intact.

## The package list, if the stash ever has to be rebuilt

**Run path** (what the stash is made from):

    libllvm18  libvulkan1  vulkan-tools  numactl

**Profiling** -- `nsys` installs into `/opt`, which does not survive, so it is
reinstalled per session from the deb kept in `$HOME`:

    ~/nsight-systems-2026.4.1_2026.4.1.191-1_amd64.deb   md5 e64519f9611c5bf192ac4346cc75cbe6

That md5 must match the workstation's copy -- traces taken with different nsys
versions are not comparable on host-side time (`WORKFLOW.md`). `ncu` needs
`CAP_SYS_ADMIN`, which the workspace has had after each recreation so far.

**Rebuilding the driver** (rarely needed -- `~/mesa/build` is in `$HOME`):

    meson ninja-build cmake g++ glslang-tools llvm-18-dev clang-18
    libdrm-dev libexpat1-dev zlib1g-dev libzstd-dev libelf-dev
    python3-mako python3-yaml pkg-config bison flex libvulkan-dev
    libxcb1-dev libx11-dev zstd rsync

`meson` from apt is **1.3.2 and too old** -- the project requires >= 1.4.0, so
install it with `pip3 install --user "meson>=1.4"` and put `~/.local/bin` on
`PATH`. A build directory copied from another tree does not relocate either;
configure a fresh one.

Two further build notes that cost time once each: the CUDA 13 toolkit there needs the
`cuCtxCreate` guard that is already in `main`, and `favorite3-cpp/out/build`
was configured with **Makefiles**, so `ninja` fails in it -- configure a fresh
Ninja build directory rather than debugging that one.

## Standing facts about measuring there

- Session drift is **0.3-1.3 ms/frame** against the RTX's 0.06, so an A/B needs
  three alternating rounds and only effects >= ~0.3-0.5 ms are adjudicable.
- Sentinels live in `$HOME/sentinels/{ctrl,f2-current}`; pass them to the A/B
  tool with `CTRLFRAMES=`.
- The GPU is on NUMA node 0 (`0-47,96-143`) of a dual-socket Xeon 8559C, and
  the process is unbound across both by default.
