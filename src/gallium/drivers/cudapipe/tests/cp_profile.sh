#!/bin/bash
#
# Where a cudapipe frame actually goes, per sample, repeatably.
#
#   cp_profile.sh SAMPLE [LABEL] [FRAMES]
#
# Runs one sample under Nsight Systems and reduces the trace to the three
# questions the performance plan's measurement gate asks:
#
#   kernels     which kernel owns the frame, and how much of it
#   api         what the host spends its time in — a sync that dominates here
#               means the number above is stall time wearing a kernel's name
#   occupancy   GPU busy time against wall time; the gap is the pipeline bubble
#
# Output lands in $VULKAN/build/prof/LABEL/SAMPLE.{nsys-rep,kernels.txt,
# api.txt,summary.txt}. The report files are kept so a later run can be
# diffed against an earlier one rather than re-argued.
#
# Ten frames by default, not sixty: the trace is per-launch and a sixty frame
# particlesystem run writes gigabytes. The warm up is still rendered, so what
# is traced is a steady-state frame either way.
#
# CUDAPIPE_DEBUG_TIME=1 in the environment additionally captures the driver's
# own per-stage lines into stages.txt, which attributes time to pipeline stage
# rather than to kernel. Those two disagreeing is itself the finding: host
# wall-clock timing around asynchronous launches measures launch latency.
#
# NCU=1 instead runs Nsight Compute over the same sample for per-kernel
# hardware counters — occupancy, memory throughput, warp stall reasons. It is
# perhaps fifty times slower and serialises every launch, so it is for one
# kernel under a microscope and never for a comparison of totals.
set -u

SAMPLE=${1:?usage: cp_profile.sh SAMPLE [LABEL] [FRAMES]}
LABEL=${2:-current}
FRAMES=${3:-10}

MESA=${MESA:-$HOME/mesa}
VULKAN=${VULKAN:-$HOME/git/Vulkan}
CUDA=${CUDA:-/usr/local/cuda}
ICD=${ICD:-$MESA/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json}

OUT=$VULKAN/build/prof/$LABEL
mkdir -p "$OUT"
cd "$VULKAN" || exit 1
BIN=build/bin/$SAMPLE
[ -x "$BIN" ] || { echo "no such sample: $BIN" >&2; exit 1; }

# The orbit list has to match the one the benchmark uses, or the profile
# describes a static scene the timed run never rendered.
ORBIT=${ORBIT-triangle pushconstants texture negativeviewportheight \
texturecubemap computeshader vulkanscene pbribl gltfscenerendering}
args=(--offscreen --benchmark --offscreenframes "$FRAMES" --benchwarmup 1)
case " $ORBIT " in *" all "*|*" $SAMPLE "*) args+=(--offscreenorbit) ;; esac

export VK_ICD_FILENAMES=$ICD

if [ "${NCU:-0}" = "1" ]; then
    # --set full is the whole counter set; --launch-count bounds it, because
    # every launch is replayed several times to gather all the counters.
    echo "=== [$LABEL] ncu $SAMPLE (this is slow) ==="
    "$CUDA/bin/ncu" --set "${NCU_SET:-full}" \
        --launch-skip "${NCU_SKIP:-200}" --launch-count "${NCU_COUNT:-40}" \
        --print-summary per-kernel \
        -o "$OUT/$SAMPLE.ncu" --force-overwrite \
        "$BIN" "${args[@]}" > "$OUT/$SAMPLE.ncu.txt" 2>&1
    echo "-> $OUT/$SAMPLE.ncu.txt"
    grep -E "^\s+(Duration|Compute|Memory|Achieved Occupancy|DRAM)" \
        "$OUT/$SAMPLE.ncu.txt" | head -40
    exit 0
fi

echo "=== [$LABEL] nsys $SAMPLE, $FRAMES frames ==="
rm -f "$OUT/$SAMPLE.nsys-rep" "$OUT/$SAMPLE.sqlite"
"$CUDA/bin/nsys" profile \
    --trace=cuda --sample=none --cpuctxsw=none \
    --force-overwrite true -o "$OUT/$SAMPLE" \
    "$BIN" "${args[@]}" > "$OUT/$SAMPLE.run.txt" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "nsys failed ($rc):"; tail -20 "$OUT/$SAMPLE.run.txt"; exit 1; }

"$CUDA/bin/nsys" stats --force-export true \
    --report cuda_gpu_kern_sum --report cuda_api_sum \
    --report cuda_gpu_mem_time_sum --report cuda_gpu_sum \
    --format column "$OUT/$SAMPLE.nsys-rep" > "$OUT/$SAMPLE.stats.txt" 2>&1

# Each report is a header line, a blank, a column header, a rule, then rows
# until the next blank. Cutting at the rule and taking to the next blank is
# what survives nsys renaming a report or reordering them.
section() {
    awk -v want="$1" '
        index($0, want) { on = 1; n = 0 }
        on { n++; print }
        on && n > 3 && /^ *$/ { exit }
    ' "$OUT/$SAMPLE.stats.txt" | head -"${2:-25}"
}

{
    echo "### $SAMPLE  [$LABEL]  $FRAMES frames"
    echo
    echo "--- GPU kernels, by total time ---"
    section "CUDA GPU Kernel Summary" 22
    echo "--- CUDA API, by total time (host side) ---"
    section "CUDA API Summary" 16
    echo "--- memory ops ---"
    section "CUDA GPU MemOps Summary (by Time)" 12
} > "$OUT/$SAMPLE.summary.txt"

cat "$OUT/$SAMPLE.summary.txt"
echo
echo "-> $OUT/$SAMPLE.summary.txt  (full: $SAMPLE.stats.txt, trace: $SAMPLE.nsys-rep)"
