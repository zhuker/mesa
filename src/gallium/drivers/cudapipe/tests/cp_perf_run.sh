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
#     BENCH   1 to time the same frames instead of storing them
#
# The two modes render exactly the same work. Offscreen benchmarking walks the
# same `for i in 0..FRAMES` loop with the same fixed frame time and the same
# orbit, and only leaves out the storing — so correctness and cost are two runs
# of one workload rather than two workloads, and the frame a chart blames can be
# looked at in the other run's images. Storing a 1280x720 frame costs about
# 10 ms, which is more than several of these samples spend rendering one, so a
# pass that writes images cannot be timed and a pass that is timed must not
# write them.
#
# **The fps figure times recording and submitting a frame, not finishing it.**
# There is no presentation engine to throttle the loop and nothing waits for the
# GPU until the pass ends, so a driver that submits asynchronously is measured
# on its CPU side alone: NVIDIA reports around 48,000 fps on triangle, which is
# the rate it can queue frames at. It is the honest number for cudapipe, which
# blocks the host on every draw, and close to it for llvmpipe, which shades on
# the CPU. Across drivers compare `wall_s` instead — the process does not exit
# before vkDeviceWaitIdle returns, so that one contains all the work, at the
# cost of also containing start up and shader compilation. Both are in
# _bench.csv for that reason.
#
# Several samples render a still scene — nothing in them moves unless the
# camera does, so a multi-frame run of one is sixty copies of the same image
# and says nothing about whether a driver holds up over an animation. Those get
# --offscreenorbit, which walks the camera around the subject. ORBIT names them;
# ORBIT=all orbits everything, ORBIT= orbits nothing.
#
# Both GPU samplers run for the whole pass rather than per sample, so a
# sample's slice is cut out of them afterwards by the process names in
# _gpu_procs.csv — _timing.csv says how long each sample took but not when it
# ran. perf.html does exactly that to put load and memory beside each sample.
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
# The same sixty frames, timed rather than stored — writes _bench.csv and one
# csv of frame times per sample, and no images at all:
#
#   $T/cp_perf_run.sh cudapipe $M/cudapipe/cudapipe_devenv_icd.x86_64.json $R/bench_cuda 60 1
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

# Benchmark mode measures the frames of a correctness run, so it needs the same
# count. Rendering some other number would time a different workload and the
# comparison the two runs exist for would not hold.
if [ "$BENCH" = "1" ] && [ "$FRAMES" -lt 1 ]; then
    echo "FRAMES must be the frame count the correctness run used, e.g. 60" >&2
    exit 1
fi

mkdir -p "$OUT/_logs"
CSV=$OUT/_timing.csv
echo "sample,wall_s,user_s,sys_s,maxrss_kb,exit" > "$CSV"
BCSV=$OUT/_bench.csv
[ "$BENCH" = "1" ] &&
    echo "sample,frames,fps,ms_avg,ms_best,ms_worst,wall_s,exit" > "$BCSV"

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
        # No file name: nothing is stored, so there is nothing to name. The
        # warm up frames are rendered before the loop and do not advance the
        # sample, so how many of them a fast machine gets through cannot change
        # which frames are measured. --benchframetimes keeps every frame's
        # time, which is what shows a single frame costing ten times its
        # neighbours rather than the mean absorbing it.
        args=(--offscreen --benchmark --offscreenframes "$FRAMES"
              --benchwarmup "${WARMUP:-1}" --benchframetimes
              --benchfilename "$OUT/$name.csv")
        case " $ORBIT " in
            *" all "*|*" $name "*) args+=(--offscreenorbit) ;;
        esac
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

    if [ "$BENCH" != "1" ]; then
        printf "%-24s wall %8ss  cpu %8ss  rss %6sMB  exit %s\n" \
            "$name" "$wall" "$(echo "$user + $sys" | bc)" "$((rss / 1024))" "$code"
        continue
    fi

    # The sample's own csv: a header, one row of totals, then the frame times
    # under a second header. The totals are read from the end of the row rather
    # than by column, because the first field is the device name and llvmpipe's
    # has a comma in it.
    set -- $(awk -F, '
        NR == 2 && NF >= 5     { fps = $NF; frames = $(NF - 1) }
        $0 == "frame,ms"       { times = 1; next }
        times && /^[0-9]+,/    { n++; t = $2 + 0; s += t
                                 if (n == 1 || t < mn) mn = t
                                 if (t > mx) mx = t }
        END { printf "%.1f %d %.2f %.2f %.2f",
                     fps + 0, frames + 0, (n ? s / n : 0), mn + 0, mx + 0 }
    ' "$OUT/$name.csv" 2>/dev/null)
    fps=${1:-0}; frames=${2:-0}; avg=${3:-0}; best=${4:-0}; worst=${5:-0}

    # renderheadless and computeheadless drive their own frames and never enter
    # the offscreen loop, so there is nothing to time. Say so rather than write
    # a row of zeroes, which reads as a sample that rendered nothing in no time.
    if [ "$frames" = "0" ]; then
        printf "%-24s no benchmark — drives its own frames        exit %s\n" \
            "$name" "$code"
        continue
    fi

    echo "$name,$frames,$fps,$avg,$best,$worst,$wall,$code" >> "$BCSV"
    printf "%-24s %9s fps  frame %8s ms (best %s, worst %s)  wall %6ss  exit %s\n" \
        "$name" "$fps" "$avg" "$best" "$worst" "$wall" "$code"
done

# renderheadless never touches the offscreen path — it writes its image itself,
# in the working directory, benchmark mode or not. A timed pass claims to leave
# nothing behind, so drop it rather than let one sample make that false.
[ "$BENCH" = "1" ] && rm -f headless.ppm

echo "[$LABEL] done -> $OUT"
