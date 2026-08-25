#!/bin/bash
#
# Is the GPU actually busy, measured without a profiler attached?
#
#   cp_gpu_busy.sh SAMPLE [SECONDS] [DRIVER]
#
# This is the performance plan's measurement gate asked in the one way that
# cannot be answered from an nsys trace. CUPTI adds host-side cost to every
# cuLaunchKernel and cudavk issues thousands a frame, so a traced run
# manufactures exactly the host-side gap the gate is looking for, and reports
# the GPU as idle whether it is or not. The only trustworthy version of this
# number comes from a run with nothing attached to it.
#
# nvidia-smi's utilization.gpu is the fraction of the sampling interval during
# which any kernel was resident, which is the definition wanted here.
#
#   >90%  kernel-bound; plan phase 1a (streams, deleting the host syncs) buys
#         nothing, because the host is already ahead of the device
#   <90%  the device waits on the host for that fraction of the frame, and 1a
#         is worth what the gap is worth
#
# ---------------------------------------------------------------------------
# Why this takes SECONDS and not a frame count
#
# A sample's process spends a fixed couple of seconds on start-up, shader
# compilation, the warm-up second and teardown, and the GPU is idle or nearly
# idle for most of it. At sixty frames the render loop is a *minority* of the
# process: instancing renders for 1.62 s of 4.3 s, dynamicuniformbuffer for
# 1.02 s of 3.7 s, and triangle for 0.04 s of 2.8 s. Sampling utilisation
# across the whole process therefore averages the render loop together with
# several seconds of idle and reports a number far below the truth — and at
# 10 Hz over a 1.6 s loop there are only about sixteen samples to average, so
# it is noisy as well as biased.
#
# So the frame count is computed from the sample's known cost to make the
# render loop last SECONDS, and the outer TRIM_PCT of the live samples are
# dropped to shed the ramp at each end. The window that was actually measured
# and the number of samples in it are printed, and a window with too few
# samples in it is refused rather than reported.
set -u

SAMPLE=${1:?usage: cp_gpu_busy.sh SAMPLE [SECONDS] [DRIVER]}
SECONDS_TARGET=${2:-20}
DRIVER=${3:-native}

MESA=${MESA:-$HOME/mesa}
VULKAN=${VULKAN:-$HOME/git/Vulkan}
M=$MESA/build-cudavk/src/gallium/targets
ITER=${ITER:-$VULKAN/build/iter}

# How much of each end to drop. The ramp is start-up at one end and teardown at
# the other, and neither is the workload.
TRIM_PCT=${TRIM_PCT:-10}
# Below this the answer is an average of a handful of polls and is not worth
# reporting. At 10 Hz this is six seconds of steady state.
MIN_SAMPLES=${MIN_SAMPLES:-60}

case $DRIVER in
    native)   export VK_ICD_FILENAMES=$MESA/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json ;;
    cudavk) export VK_ICD_FILENAMES=$M/cudavk/cudavk_devenv_icd.x86_64.json ;;
    llvmpipe) export VK_ICD_FILENAMES=$M/lavapipe/lvp_devenv_icd.x86_64.json ;;
    nvidia)   ;;
    *) echo "DRIVER must be native, cudavk, llvmpipe or nvidia" >&2; exit 1 ;;
esac

cd "$VULKAN" || exit 1
BIN=build/bin/$SAMPLE
[ -x "$BIN" ] || { echo "no such sample: $BIN" >&2; exit 1; }

# What the sample costs per frame, from the most recent iteration that timed
# it. Without a figure the frame count cannot be chosen, so fall back to a
# short probe run rather than guessing.
MS=$(ls -t "$ITER"/*/bench/_bench.csv 2>/dev/null | while read -r f; do
        awk -F, -v s="$SAMPLE" '$1==s && $4+0>0 {print $4; exit}' "$f"
     done | head -1)

if [ -z "${MS:-}" ]; then
    echo "no recorded cost for $SAMPLE; probing with 60 frames" >&2
    probe=$("$BIN" --offscreen --benchmark --offscreenframes 60 --benchwarmup 1 \
            2>/dev/null | awk '/^frame /{print $3}' | head -1)
    MS=${probe:-10}
fi

FRAMES=$(awk -v s="$SECONDS_TARGET" -v ms="$MS" \
         'BEGIN{n=int(s*1000/ms); print (n<60)?60:((n>20000)?20000:n)}')

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -gt 0 ] && echo "warning: $busy process(es) already on the GPU" >&2

# The same orbit list cp_profile.sh and cp_perf_run.sh use, or this measures a
# static scene the benchmark never rendered.
ORBIT=${ORBIT-triangle pushconstants texture negativeviewportheight \
texturecubemap computeshader vulkanscene pbribl gltfscenerendering}
args=(--offscreen --benchmark --offscreenframes "$FRAMES" --benchwarmup 1)
case " $ORBIT " in *" all "*|*" $SAMPLE "*) args+=(--offscreenorbit) ;; esac

echo "$SAMPLE [$DRIVER]: ${MS} ms/frame recorded -> $FRAMES frames for" \
     "~${SECONDS_TARGET}s of rendering"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    -lms 100 > "$TMP/util.csv" 2>/dev/null &
sampler=$!

start=$(date +%s.%N)
"$BIN" "${args[@]}" > "$TMP/run.txt" 2>&1
rc=$?
end=$(date +%s.%N)
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

[ $rc -ne 0 ] && { echo "sample failed ($rc):"; tail -5 "$TMP/run.txt"; exit 1; }
grep -E "fps|ms/frame|Average" "$TMP/run.txt" | tail -2

python3 - "$TMP/util.csv" "$SAMPLE" "$DRIVER" "$TRIM_PCT" "$MIN_SAMPLES" \
         "$(echo "$end - $start" | bc)" <<'PY'
import statistics, sys

vals = [int(l) for l in open(sys.argv[1]) if l.strip().isdigit()]
sample, driver = sys.argv[2], sys.argv[3]
trim_pct, min_samples, wall = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])

# Start-up, shader compilation and teardown are the only stretches with the GPU
# at zero, so they are where the window is cut.
live = [i for i, v in enumerate(vals) if v > 0]
if not live:
    sys.exit("no GPU activity sampled")
lo, hi = live[0], live[-1]
win = vals[lo:hi + 1]

# Then drop the ramp at each end: the first frames of the loop climb while
# caches and clocks settle, and the last overlap teardown.
trim = len(win) * trim_pct // 100
core = win[trim:len(win) - trim] if trim and len(win) - 2 * trim >= 10 else win

print("  process %.1fs, GPU live %.1fs, steady window %.1fs (%d samples at 10 Hz)"
      % (wall, len(win) / 10.0, len(core) / 10.0, len(core)))

if len(core) < min_samples:
    print("  REFUSING to report: %d samples is too few to average. Raise the"
          " SECONDS argument (currently giving %.1fs of steady state)."
          % (len(core), len(core) / 10.0))
    sys.exit(2)

med = statistics.median(core)
print("  median busy %d%%   mean %d%%   p10 %d%%   p90 %d%%"
      % (med, statistics.mean(core),
         statistics.quantiles(core, n=10)[0] if len(core) > 10 else min(core),
         statistics.quantiles(core, n=10)[8] if len(core) > 10 else max(core)))
print("  -> %s" % ("kernel-bound: the host is ahead of the device, and plan "
                   "phase 1a buys nothing" if med >= 90 else
                   "the device idles ~%d%% of the frame waiting on the host; "
                   "that is what phase 1a is worth" % (100 - med)))
PY
