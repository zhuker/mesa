#!/bin/bash
#
# One driver's pass over the sample sweep, with the numbers a performance
# comparison needs: wall clock, user and system CPU and peak RSS per sample,
# and the GPU's utilisation, total memory and per-process memory sampled at
# 2 Hz for the whole run.
#
#   cp_perf_run.sh LABEL ICD OUTDIR FRAMES [BENCH]
#
#     LABEL   name for the progress line only
#     ICD     VK_ICD_FILENAMES value, empty for the system default (NVIDIA)
#     OUTDIR  written: _timing.csv, _gpu.csv, _gpu_procs.csv, _logs/, frames
#     FRAMES  frames per sample; >1 gives each sample a directory of its own
#     BENCH   1 to run the samples' own benchmark mode instead, which writes
#             no images and reports fps
#
# Several samples render a still scene — nothing in them moves unless the
# camera does, so a multi-frame run of one is sixty copies of the same image
# and says nothing about whether a driver holds up over an animation. Those get
# --offscreenorbit, which walks the camera around the subject. ORBIT names them;
# ORBIT=all orbits everything, ORBIT= orbits nothing.
#
# Both GPU samplers run for the whole pass, so a sample's slice is found by
# timestamp against _timing.csv rather than by a per-sample capture.
#
# Example, all three drivers at 60 frames:
#
#   R=$PWD/build/frames60
#   M=$HOME/mesa/build-cudapipe/src/gallium/targets
#   T=$HOME/mesa/src/gallium/drivers/cudapipe/tests
#   $T/cp_perf_run.sh nvidia   ""                                          $R/nvidia   60
#   $T/cp_perf_run.sh cudapipe $M/cudapipe/cudapipe_devenv_icd.x86_64.json $R/cuda     60
#   $T/cp_perf_run.sh llvmpipe $M/lavapipe/lvp_devenv_icd.x86_64.json      $R/llvmpipe 60
#
# Kept in the tree because the numbers are only worth anything if the run
# behind them can be repeated exactly.
set -u

LABEL=$1; ICD=$2; OUT=$3; FRAMES=$4; BENCH=${5:-0}

VULKAN=${VULKAN:-$HOME/git/Vulkan}
LIST=${LIST:-$HOME/mesa/src/gallium/drivers/cudapipe/tests/headless_streamer_samples.txt}
cd "$VULKAN" || exit 1
BIN=build/bin
SAMPLES=${SAMPLES:-$(grep -v '^#' "$LIST" | tr '\n' ' ')}

# Samples with no motion of their own; the camera has to supply it.
ORBIT=${ORBIT-triangle pushconstants texture negativeviewportheight \
texturecubemap computeshader vulkanscene pbribl gltfscenerendering}

mkdir -p "$OUT/_logs"
CSV=$OUT/_timing.csv
echo "sample,wall_s,user_s,sys_s,maxrss_kb,exit" > "$CSV"

# Whole-GPU load and memory. Runs for the pass, not per sample: several samples
# finish in well under a second, so a per-sample capture would get one poll.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
           --format=csv,noheader -lms 500 > "$OUT/_gpu.csv" 2>/dev/null &
SMI=$!
# Per-process GPU memory, so a sample's own footprint is separable from
# whatever else is on the card.
( while true; do
    ts=$(date +%s.%N)
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null |
      while read -r l; do echo "$ts,$l"; done
    sleep 0.5
  done ) > "$OUT/_gpu_procs.csv" 2>/dev/null &
PROCS=$!
trap 'kill $SMI $PROCS 2>/dev/null' EXIT

for name in $SAMPLES; do
    bin=$BIN/$name
    [ -x "$bin" ] || continue

    if [ "$BENCH" = "1" ]; then
        args=(--offscreen --benchmark --benchwarmup "${WARMUP:-1}"
              --benchruntime "${DURATION:-3}" --benchfilename "$OUT/$name.csv")
    elif [ "$FRAMES" -gt 1 ]; then
        mkdir -p "$OUT/$name"
        args=(--offscreen --offscreenframes "$FRAMES"
              --offscreenfilename "$OUT/$name/$name.ppm")
        case " $ORBIT " in
            *" all "*|*" $name "*) args+=(--offscreenorbit) ;;
        esac
    else
        args=(--offscreen --offscreenframes 1 --offscreenfilename "$OUT/$name.ppm")
    fi

    /usr/bin/time -f "%e %U %S %M" -o "$OUT/_logs/$name.time" \
        env ${ICD:+VK_ICD_FILENAMES=$ICD} timeout "${TIMEOUT:-900}" "$bin" "${args[@]}" \
        < /dev/null > "$OUT/_logs/$name.log" 2>&1
    code=$?

    # /usr/bin/time prepends a line when the child dies by a signal, so the
    # measurements are the last line of the file and never the first.
    set -- $(tail -1 "$OUT/_logs/$name.time")
    wall=${1:-0}; user=${2:-0}; sys=${3:-0}; rss=${4:-0}

    echo "$name,$wall,$user,$sys,$rss,$code" >> "$CSV"
    printf "%-24s wall %8ss  cpu %8ss  rss %6sMB  exit %s\n" \
        "$name" "$wall" "$(echo "$user + $sys" | bc)" "$((rss / 1024))" "$code"
done

echo "[$LABEL] done -> $OUT"
