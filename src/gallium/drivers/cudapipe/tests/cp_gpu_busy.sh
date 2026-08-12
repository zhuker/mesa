#!/bin/bash
#
# Is the GPU actually busy, measured without a profiler attached?
#
#   cp_gpu_busy.sh SAMPLE [FRAMES] [DRIVER]
#
# This is the performance plan's measurement gate asked in the one way that
# cannot be answered from an nsys trace. CUPTI adds host-side cost to every
# cuLaunchKernel and instruments every kernel, and cudapipe issues thousands of
# launches a frame — so a traced run manufactures exactly the host-side gap the
# gate is looking for, and reports the GPU as idle whether it is or not. The
# only trustworthy version of this number comes from a run with nothing
# attached to it.
#
# nvidia-smi's utilization.gpu is the fraction of the sampling interval during
# which any kernel was resident, which is the definition wanted here. It is
# sampled at 10 Hz for the duration of one sample's benchmark pass, and the
# warm-up second is dropped because it is not the workload.
#
#   >90%  kernel-bound; plan phase 1a (streams, deleting the host syncs) buys
#         nothing, because the host is already ahead of the device
#   <90%  the device waits on the host for that fraction of the frame, and 1a
#         is worth what the gap is worth
#
# Reports the median rather than the mean: the distribution is bimodal (the
# process starts and ends with the GPU idle) and a mean over the tails
# understates the steady state.
set -u

SAMPLE=${1:?usage: cp_gpu_busy.sh SAMPLE [FRAMES] [DRIVER]}
FRAMES=${2:-120}
DRIVER=${3:-cudapipe}

MESA=${MESA:-$HOME/mesa}
VULKAN=${VULKAN:-$HOME/git/Vulkan}
M=$MESA/build-cudapipe/src/gallium/targets

case $DRIVER in
    cudapipe) export VK_ICD_FILENAMES=$M/cudapipe/cudapipe_devenv_icd.x86_64.json ;;
    llvmpipe) export VK_ICD_FILENAMES=$M/lavapipe/lvp_devenv_icd.x86_64.json ;;
    nvidia)   ;;
    *) echo "DRIVER must be cudapipe, llvmpipe or nvidia" >&2; exit 1 ;;
esac

cd "$VULKAN" || exit 1
BIN=build/bin/$SAMPLE
[ -x "$BIN" ] || { echo "no such sample: $BIN" >&2; exit 1; }

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -gt 0 ] && echo "warning: $busy process(es) already on the GPU" >&2

# The same orbit list cp_profile.sh and cp_perf_run.sh use, or this measures a
# static scene the benchmark never renders.
ORBIT=${ORBIT-triangle pushconstants texture negativeviewportheight \
texturecubemap computeshader vulkanscene pbribl gltfscenerendering}
args=(--offscreen --benchmark --offscreenframes "$FRAMES" --benchwarmup 1)
case " $ORBIT " in *" all "*|*" $SAMPLE "*) args+=(--offscreenorbit) ;; esac

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    -lms 100 > "$TMP/util.csv" 2>/dev/null &
sampler=$!

"$BIN" "${args[@]}" > "$TMP/run.txt" 2>&1
rc=$?
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

[ $rc -ne 0 ] && { echo "sample failed ($rc):"; tail -5 "$TMP/run.txt"; exit 1; }

grep -E "fps|ms/frame|Average" "$TMP/run.txt" | tail -3

python3 - "$TMP/util.csv" "$SAMPLE" "$DRIVER" <<'PY'
import statistics, sys

vals = [int(l) for l in open(sys.argv[1]) if l.strip().isdigit()]
# Drop the idle head and tail: process start-up, shader compilation and the
# teardown are not the workload, and they are the only zeroes.
live = [v for v in vals if v > 0]
if not live:
    sys.exit("no GPU activity sampled -- is the sample too short? raise FRAMES")
med = statistics.median(live)
print("%s [%s]: %d samples at 10 Hz, %d with the GPU live"
      % (sys.argv[2], sys.argv[3], len(vals), len(live)))
print("  median busy %d%%   mean %d%%   p10 %d%%   p90 %d%%"
      % (med, statistics.mean(live), min(live), max(live)))
print("  -> %s" % ("kernel-bound: the host is ahead of the device, and plan "
                   "phase 1a buys nothing" if med >= 90 else
                   "the device idles ~%d%% of the frame waiting on the host; "
                   "that is what phase 1a is worth" % (100 - med)))
PY
