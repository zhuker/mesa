#!/usr/bin/env bash
# Bring the B200 workspace back after it has been recreated.
#
#   ssh b200 'bash -s' < src/cudavk/tests/cp_b200_setup.sh
#
# That host is a Coder workspace: $HOME persists, /tmp and every system package
# do not. After a recreation the driver build, the harnesses, the shim and the
# sentinels are all still there and nothing needs rebuilding -- but the Vulkan
# loader is gone, so vkCreateInstance returns -9 (INCOMPATIBLE_DRIVER) and every
# run aborts with rc=134 before writing a timestamp. That failure looks like a
# broken experiment and is not one; it has cost three sessions.
set -u
echo "== host =="; hostname
sudo apt-get update -qq >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
     libvulkan1 vulkan-tools numactl zstd rsync \
     >/dev/null 2>&1
echo "== what the runs need =="
for f in libvulkan1 numactl; do printf '  %-12s ' "$f"; dpkg -s $f >/dev/null 2>&1 && echo present || echo MISSING; done
printf '  %-12s ' taskset; command -v taskset >/dev/null && echo present || echo MISSING
printf '  %-12s ' numactl;  command -v numactl  >/dev/null && echo present || echo MISSING
echo "== $HOME survivors (these should never need rebuilding) =="
for p in mesa/build/src/cudavk/libvulkan_cudavk.so \
         mesa/build/src/cudavk/cudavk_devenv_icd.x86_64.json \
         favorite3-cpp/out/build/vulkan_app favorite2-cpp/out/build/vulkan_app \
         favorite-cpp/submit_shim.so sentinels/ctrl sentinels/f2-current; do
  printf '  %-52s ' "$p"; [ -e "$HOME/$p" ] && echo ok || echo MISSING
done
echo "== does the ICD load? =="
VK_DRIVER_FILES=$HOME/mesa/build/src/cudavk/cudavk_devenv_icd.x86_64.json \
  vulkaninfo --summary 2>/dev/null | grep -E "driverName|deviceName" | head -2 \
  || echo "  ICD STILL NOT LOADING"
