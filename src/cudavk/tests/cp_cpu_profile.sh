#!/bin/bash
#
# Capture on-CPU and off-CPU folded stacks for one command.
#
#   cp_cpu_profile.sh OUT SECONDS -- COMMAND [ARG ...]
#
# Run `sudo -v` before invoking this script. The command is left to finish
# naturally after the sampling window so CUDA/CUPTI state is not truncated.
set -u

OUT=${1:?usage: cp_cpu_profile.sh OUT SECONDS -- COMMAND [ARG ...]}
DURATION=${2:?usage: cp_cpu_profile.sh OUT SECONDS -- COMMAND [ARG ...]}
shift 2
[ "${1:-}" = -- ] || { echo "expected -- before command" >&2; exit 2; }
shift
[ $# -gt 0 ] || { echo "missing command" >&2; exit 2; }

PERF=${PERF:-$HOME/.local/bin/perf}
OFFCPU=${OFFCPU:-offcputime-bpfcc}
DELAY=${DELAY:-5}

[ -x "$PERF" ] || { echo "perf not found: $PERF" >&2; exit 1; }
command -v "$OFFCPU" >/dev/null || { echo "$OFFCPU not found" >&2; exit 1; }
sudo -n true 2>/dev/null || {
    echo "sudo credentials are not cached; run 'sudo -v' first" >&2
    exit 1
}

mkdir -p "$OUT"
printf '%q ' "$@" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"

"$@" > "$OUT/run.log" 2>&1 &
target=$!
echo "$target" > "$OUT/pid"

sleep "$DELAY"
kill -0 "$target" 2>/dev/null || {
    wait "$target"
    echo "target exited before the profiling delay" >&2
    exit 1
}

"$PERF" record -F 999 -g --call-graph dwarf -p "$target" \
    -o "$OUT/perf.data" -- sleep "$DURATION" > "$OUT/perf-record.log" 2>&1 &
perf_pid=$!
sudo "$OFFCPU" -f -p "$target" "$DURATION" \
    > "$OUT/offcpu.folded" 2> "$OUT/offcpu.log" &
offcpu_pid=$!

wait "$perf_pid"
perf_status=$?
wait "$offcpu_pid"
offcpu_status=$?
wait "$target"
target_status=$?

if [ "$perf_status" -eq 0 ]; then
    "$PERF" report -i "$OUT/perf.data" --stdio -g folded,0.5,caller \
        --no-children > "$OUT/oncpu.folded"
    "$PERF" report -i "$OUT/perf.data" --stdio --no-children -g none \
        > "$OUT/oncpu-flat.txt"
fi

sort -t' ' -k2 -rn "$OUT/offcpu.folded" -o "$OUT/offcpu.folded"
printf 'target=%d perf=%d offcpu=%d\n' \
    "$target_status" "$perf_status" "$offcpu_status" | tee "$OUT/status.txt"

[ "$target_status" -eq 0 ] && [ "$perf_status" -eq 0 ] && \
    [ "$offcpu_status" -eq 0 ]
