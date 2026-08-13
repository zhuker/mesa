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
# Three modes, and they answer three different questions in this order:
#
#   METRICS=1   is the GPU doing work?      device counters, no CUPTI
#   (default)   which kernel owns the frame? CUDA API trace
#   NCU=1       why is that kernel slow?     per-kernel hardware counters
#
# METRICS=1 samples GPU hardware counters with the CUDA API left
# uninstrumented, so unlike the default mode it is not distorted by this
# driver's own launch count. It reports SMs Active against SM Issue — occupied
# but not issuing is a stalled kernel — plus warps in flight, DRAM bandwidth
# and PCIe traffic, over the window in which the GPU was actually busy. It
# names no kernels; run it before the traced pass, not instead of it.
#
# NCU=1 instead runs Nsight Compute over the same sample for per-kernel
# hardware counters — occupancy, memory throughput, warp stall reasons. It is
# perhaps fifty times slower and serialises every launch, so it is for one
# kernel under a microscope and never for a comparison of totals.
#
# Both need GPU counter permission (NVreg_RestrictProfilingToAdminUsers=0).
set -u

SAMPLE=${1:?usage: cp_profile.sh SAMPLE [LABEL] [FRAMES]}
LABEL=${2:-current}
FRAMES=${3:-10}

MESA=${MESA:-$HOME/mesa}
VULKAN=${VULKAN:-$HOME/git/Vulkan}
CUDA=${CUDA:-/usr/local/cuda}
ICD=${ICD:-$MESA/build-cudapipe/src/gallium/targets/cudapipe/cudapipe_devenv_icd.x86_64.json}

# $CUDA/bin/nsys is a wrapper with the version that shipped with the toolkit
# hardcoded into it, so a newer Nsight Systems installed beside it is ignored
# unless it is named. Take the newest installed; NSYS= names one explicitly,
# which is how a trace is reproduced against the version that recorded it.
NSYS=${NSYS:-$(ls -d /opt/nvidia/nsight-systems/*/target-linux-*/nsys \
    2>/dev/null | sort -V | tail -1)}
NSYS=${NSYS:-$CUDA/bin/nsys}

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

# Recorded because CUPTI's per-launch cost differs between versions, and this
# driver issues thousands of launches a frame: two traces taken with different
# Nsight Systems are not comparable on host-side time.
NSYS_VER=$("$NSYS" --version 2>/dev/null | tail -1)

if [ "${METRICS:-0}" = "1" ]; then
    # Hardware counter sampling, with the CUDA API left uninstrumented. This is
    # the one profile of this driver that is not distorted by its own launch
    # count: CUPTI charges every cuLaunchKernel, cudapipe issues thousands a
    # frame, and a traced run therefore manufactures the host-side gap the
    # measurement gate is looking for -- see ../PHASE_1A.md. Counters sample on
    # the device and do not know the launches exist.
    #
    # It answers "is the GPU doing work", not "which kernel"; run it before the
    # traced pass, not instead of it. SMs Active against SM Issue is the pair
    # that matters -- occupied but not issuing is a stalled kernel, and nothing
    # else in this tree can tell those apart.
    #
    # Needs the counter permission (NVreg_RestrictProfilingToAdminUsers=0); if
    # the GPU is not listed, check /proc/driver/nvidia/params before the tool.
    # System scope: it samples the whole GPU, so run nothing else on it.
    echo "=== [$LABEL] nsys gpu-metrics $SAMPLE, $FRAMES frames ($NSYS_VER) ==="
    rm -f "$OUT/$SAMPLE.metrics.nsys-rep" "$OUT/$SAMPLE.metrics.sqlite"
    "$NSYS" profile --trace=none \
        --gpu-metrics-devices="${METRICS_DEV:-0}" \
        --gpu-metrics-frequency="${METRICS_HZ:-10000}" \
        ${METRICS_SET:+--gpu-metrics-set="$METRICS_SET"} \
        --force-overwrite true -o "$OUT/$SAMPLE.metrics" \
        "$BIN" "${args[@]}" > "$OUT/$SAMPLE.metrics.run.txt" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "nsys failed ($rc):"
                       tail -20 "$OUT/$SAMPLE.metrics.run.txt"; exit 1; }

    "$NSYS" export --type sqlite --force-overwrite true \
        -o "$OUT/$SAMPLE.metrics.sqlite" "$OUT/$SAMPLE.metrics.nsys-rep" \
        >> "$OUT/$SAMPLE.metrics.run.txt" 2>&1

    "${PYTHON:-python3}" - "$OUT/$SAMPLE.metrics.sqlite" "$SAMPLE" "$LABEL" \
        <<'PY' | tee "$OUT/$SAMPLE.metrics.txt"
import sqlite3, sys

db = sqlite3.connect(sys.argv[1])
ids = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
if not ids:
    sys.exit("no GPU metrics in the report -- was the GPU listed by "
             "--gpu-metrics-devices=help?")

# Percentiles over the whole capture would be dominated by start-up and
# teardown idle, which is the bias cp_gpu_busy.sh was corrected for once
# already. Take the window between the first and last sample that shows the
# graphics/compute engine busy, and say how much of the run that was.
gr = next((m for m, n in ids.items() if n.startswith("GR Active")), None)
rows = db.execute("SELECT timestamp, value FROM GPU_METRICS WHERE metricId=?"
                  " ORDER BY timestamp", (gr,)).fetchall()
busy = [t for t, v in rows if v >= 5]
if len(busy) < 2:
    sys.exit("GPU never reached 5%% busy -- nothing to summarise")
lo, hi = busy[0], busy[-1]
span, total = hi - lo, rows[-1][0] - rows[0][0]

def pct(mid):
    v = sorted(v for t, v in db.execute(
        "SELECT timestamp, value FROM GPU_METRICS WHERE metricId=?"
        " AND timestamp BETWEEN ? AND ?", (mid, lo, hi)))
    return (v[len(v)//2], v[int(len(v)*0.9)], v[-1]) if v else (0, 0, 0)

print("### %s  [%s]  gpu metrics" % (sys.argv[2], sys.argv[3]))
print("# active window %.2f s of %.2f s traced, %d samples"
      % (span/1e9, total/1e9, len(busy)))

# A fast sample renders for a fraction of the process, and the rest is shader
# compilation, texture upload and teardown. The window below then averages
# mostly idle and every figure reads low -- the same bias cp_gpu_busy.sh was
# corrected for by taking a target duration instead of a frame count. Ten
# frames of instancing gave a median GR Active of 1%; a hundred frames of
# particlesystem gave 94%, matching what cp_gpu_busy.sh reports for it.
gr_med = sorted(v for t, v in rows if lo <= t <= hi)
gr_med = gr_med[len(gr_med)//2] if gr_med else 0
if gr_med < 50:
    print("# WARNING: GPU busy only %d%% of the window -- the render loop is a"
          " minority of\n#          this run. Raise FRAMES until this is high"
          " before reading anything below." % gr_med)
print("%-46s %7s %7s %7s" % ("metric (over the active window)",
                             "med", "p90", "max"))
WANT = ("GR Active", "SMs Active", "SM Issue", "Compute Warps in Flight [Thr",
        "Compute Warps in Flight [Avg Warps", "DRAM Read", "DRAM Write",
        "PCIe RX", "PCIe TX", "Sync Copy Engine Active [Thr",
        "Async Copy Engine Active 0 [Thr")
for want in WANT:
    for mid, name in sorted(ids.items()):
        if name.startswith(want):
            print("%-46s %7d %7d %7d" % ((name[:46],) + pct(mid)))
            break
PY
    echo
    echo "-> $OUT/$SAMPLE.metrics.txt  (trace: $SAMPLE.metrics.nsys-rep)"
    exit 0
fi

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

echo "=== [$LABEL] nsys $SAMPLE, $FRAMES frames ($NSYS_VER) ==="
rm -f "$OUT/$SAMPLE.nsys-rep" "$OUT/$SAMPLE.sqlite"
# NVTX as well as CUDA, and the driver's ranges turned on to fill it. Every
# compiled shader is a kernel named `main`, so without the ranges the timeline
# cannot say which draw or which stage a row belongs to; with them the trace
# names itself. NVTX=0 leaves the driver silent for a trace that has to match
# an older one, since the ranges cost a few percent when enabled.
CUDAPIPE_NVTX_ENV=()
[ "${NVTX:-1}" = "1" ] && CUDAPIPE_NVTX_ENV=(env CUDAPIPE_NVTX=1)
"${CUDAPIPE_NVTX_ENV[@]}" "$NSYS" profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite true -o "$OUT/$SAMPLE" \
    "$BIN" "${args[@]}" > "$OUT/$SAMPLE.run.txt" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "nsys failed ($rc):"; tail -20 "$OUT/$SAMPLE.run.txt"; exit 1; }

"$NSYS" stats --force-export true \
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
    echo "# $NSYS_VER"
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
