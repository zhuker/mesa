#!/bin/bash
#
# Is the device full, for every sample, in one table.
#
#   cp_metrics_sweep.sh LABEL [SECONDS]
#
# Runs METRICS=1 cp_profile.sh over the whole sample set and reduces it to one
# csv, `build/iter/LABEL/metrics.csv`. `cp_iter_report.py record` picks that up
# and puts it in the iteration's record, so an iteration says not only what it
# cost but what the machine was doing while it cost that.
#
# ---------------------------------------------------------------------------
# Why an iteration wants this and not just ms/frame
#
# `GR Active` — what nvidia-smi reports and what cp_gpu_busy.sh asks for — says
# only that a kernel was resident. Measured this way the whole sweep reads
# 83-100% busy on every sample that uses the GPU at all, while issuing
# instructions on 1-6% of cycles with 9-40% of SMs holding any work. So the
# number that says whether an optimisation is working is `issue`, and it is not
# the one anything else records.
#
# That matters most for a change whose whole point is to fill the device —
# batching draws, growing a grid, raising occupancy. Those move `issue` and
# `sms` before they move ms/frame, and a change that moves ms/frame without
# moving either did something else than it claimed.
#
# ---------------------------------------------------------------------------
# Cost
#
# Each sample renders for SECONDS (default 12), so the sweep is about five
# minutes. That is why cp_iterate.sh only runs it when asked (METRICS=1) rather
# than every time.
#
# The seconds-not-frames trap applies here exactly as it does to cp_gpu_busy.sh:
# a sample's process spends a couple of seconds compiling shaders and tearing
# down with the GPU idle, so a short window averages the render loop together
# with that idle and reads low. The frame count is computed from the sample's
# own recorded cost to make the render loop last SECONDS.
set -u

LABEL=${1:?usage: cp_metrics_sweep.sh LABEL [SECONDS]}
SECONDS_TARGET=${2:-12}

MESA=${MESA:-$HOME/mesa}
VULKAN=${VULKAN:-$HOME/git/Vulkan}
T="$(cd "$(dirname "$0")" && pwd)"
ITER=${ITER:-$VULKAN/build/iter}
OUT="$ITER/$LABEL"
PY="$MESA/venv/bin/python3"

[ -d "$OUT" ] || { echo "no such iteration: $OUT" >&2; exit 1; }

BENCH="$OUT/bench/_bench.csv"
[ -f "$BENCH" ] || { echo "no bench csv: $BENCH" >&2; exit 1; }

# This measures whatever is built right now and files it under LABEL, so run
# against an iteration built from a different commit it records one build's
# counters as another's — the same class of mistake as reusing a label, and
# quieter, because a plausible table appears either way. cp_iterate.sh calls
# this straight after its own build, where the two always agree.
HEAD_NOW=$(git -C "$MESA" rev-parse HEAD 2>/dev/null)
HEAD_ITER=$(head -1 "$OUT/_commit.txt" 2>/dev/null)
if [ -n "$HEAD_NOW" ] && [ -n "$HEAD_ITER" ] && [ "$HEAD_NOW" != "$HEAD_ITER" ]; then
    echo "iteration '$LABEL' was built from $HEAD_ITER" >&2
    echo "  the tree is now at                $HEAD_NOW" >&2
    echo "  these counters would describe a different build; FORCE=1 to do it anyway" >&2
    [ "${FORCE:-0}" = "1" ] || exit 1
fi

CSV="$OUT/metrics.csv"
echo "sample,ms,gr,sms,issue,warps,dram_r,dram_w,window_s,samples" > "$CSV"

# Frames from the sample's own cost, so every row gets a comparable window.
while IFS=, read -r name frames fps ms rest; do
    case "$name" in sample|"") continue ;; esac
    case "$ms" in ""|0|0.00) continue ;; esac
    n=$("$PY" -c "print(max(60,int($SECONDS_TARGET*1000/$ms)))" 2>/dev/null) || continue

    ( cd "$VULKAN" && METRICS=1 "$T/cp_profile.sh" "$name" "_metrics_$LABEL" "$n" ) \
        >/dev/null 2>&1

    f="$VULKAN/build/prof/_metrics_$LABEL/$name.metrics.txt"
    [ -f "$f" ] || { echo "$name: no metrics output" >&2; continue; }

    "$PY" - "$f" "$name" "$ms" >> "$CSV" <<'PYEOF'
import re, sys
path, name, ms = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
win = re.search(r"active window ([\d.]+) s of [\d.]+ s traced, (\d+) samples", text)


def med(label):
    """The median column of the row whose metric starts with `label`."""
    for line in text.splitlines():
        if line.startswith(label):
            m = re.search(r"\s(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", line)
            if m:
                return m.group(1)
    return ""


print(",".join([
    name, ms,
    med("GR Active"), med("SMs Active"), med("SM Issue"),
    med("Compute Warps in Flight [Avg"),
    med("DRAM Read Bandwidth"), med("DRAM Write Bandwidth"),
    win.group(1) if win else "", win.group(2) if win else "",
]))
PYEOF
done < "$BENCH"

echo "-> $CSV"
column -t -s, "$CSV"
echo
echo "issue = % of cycles the SMs issued an instruction. It is the column that"
echo "says whether the device is working; gr only says a kernel was resident."
