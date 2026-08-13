#!/bin/bash
#
# One optimisation iteration: build, time the sixty frames, render them, and
# say what changed against another iteration.
#
#   cp_iterate.sh LABEL [COMPARE_LABEL]
#
# Everything lands under $VULKAN/build/iter/LABEL:
#
#   bench/   the timed pass — _bench.csv, per-sample frame times, no images
#   verdict.txt   cp_compare_frames.py against the stored nvidia reference
#   delta.txt     ms/frame against COMPARE_LABEL, per sample and in total
#
# The stored frames go to build/iter/_frames/LABEL instead, beside the nvidia
# reference at build/iter/_frames/nvidia, because cp_compare_frames.py takes one
# root holding a directory per driver and compares them pairwise.
#
# The two passes render the same work; see TESTING.md. The reference frames are
# rendered once into build/iter/_ref and reused, because NVIDIA's output does
# not change when cudapipe does.
#
# Why a script rather than the commands: a performance claim is only worth
# something if the run behind it can be repeated, and the flags that have to
# match between the timed and the stored pass are exactly the ones that are
# easy to get wrong by hand.
#
# BENCH_ONLY=1 skips the frame pass when only the cost is in question.
# DRIVER=llvmpipe times lavapipe instead, for the calibration in TESTING.md.
set -u

LABEL=${1:?usage: cp_iterate.sh LABEL [COMPARE_LABEL]}
AGAINST=${2:-}

MESA=${MESA:-$HOME/mesa}
VULKAN=${VULKAN:-$HOME/git/Vulkan}
T=$MESA/src/gallium/drivers/cudapipe/tests
M=$MESA/build-cudapipe/src/gallium/targets
FRAMES=${FRAMES:-60}

case ${DRIVER:=cudapipe} in
    cudapipe) ICD=$M/cudapipe/cudapipe_devenv_icd.x86_64.json ;;
    llvmpipe) ICD=$M/lavapipe/lvp_devenv_icd.x86_64.json ;;
    nvidia)   ICD= ;;
    *) echo "DRIVER must be cudapipe, llvmpipe or nvidia" >&2; exit 1 ;;
esac

ROOT=$VULKAN/build/iter
OUT=$ROOT/$LABEL
FRAMEROOT=$ROOT/_frames
REF=$FRAMEROOT/nvidia

# An iteration is a record, and reusing a label overwrites one. It is easy to
# do: the labels that suggest themselves for a change are the same ones that
# suggested themselves last time something touched the same code, and nothing
# about the run says the directory was already there. It has happened once —
# a second pass named an iteration `dscratch` over the first pass's `dscratch`,
# and the earlier bench numbers and sixty frames a document still cites are
# simply gone. Refuse rather than clobber; FORCE=1 to re-run a label on
# purpose, which is the case where the old numbers are the ones being replaced.
if [ -e "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "iteration '$LABEL' already exists at $OUT" >&2
    echo "  it holds $(cat "$OUT/_commit.txt" 2>/dev/null | head -1)" >&2
    echo "  pick another label, or FORCE=1 to overwrite it" >&2
    exit 1
fi

# A stale build silently measures the previous iteration, which is the single
# most expensive mistake available here.
export PATH=$HOME/vulkan-sdk/1.4.357.1/x86_64/bin:$PATH
"$MESA/venv/bin/ninja" -C "$MESA/build-cudapipe" > "$ROOT/.build.log" 2>&1 || {
    echo "build failed:"; tail -30 "$ROOT/.build.log"; exit 1
}
mkdir -p "$OUT"
git -C "$MESA" rev-parse HEAD > "$OUT/_commit.txt" 2>/dev/null
git -C "$MESA" diff --stat >> "$OUT/_commit.txt" 2>/dev/null

# Nothing else may be on the card while the timed pass runs; two passes sharing
# it measure each other.
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -gt 0 ] && echo "warning: $busy process(es) already on the GPU" >&2

echo "=== [$LABEL] timing $FRAMES frames ($DRIVER) ==="
"$T/cp_perf_run.sh" "$DRIVER" "$ICD" "$OUT/bench" "$FRAMES" 1 || exit 1

if [ "${BENCH_ONLY:-0}" != "1" ]; then
    if [ ! -d "$REF" ]; then
        echo "=== rendering the nvidia reference (once) ==="
        "$T/cp_perf_run.sh" nvidia "" "$REF" "$FRAMES" || exit 1
    fi

    echo "=== [$LABEL] rendering the same $FRAMES frames ==="
    # Frames are never reused between iterations: a partial re-render leaves
    # one sample's old images beside another's new ones, which reads as a
    # change in the sample that was not re-run.
    rm -rf "${FRAMEROOT:?}/$LABEL"
    "$T/cp_perf_run.sh" "$DRIVER" "$ICD" "$FRAMEROOT/$LABEL" "$FRAMES" || exit 1

    echo "=== [$LABEL] correctness against nvidia ==="
    "$MESA/venv/bin/python3" "$T/cp_compare_frames.py" \
        "$FRAMEROOT" --ref nvidia --test "$LABEL" \
        > "$OUT/verdict.txt" 2>&1
    verdict=$?
    tail -32 "$OUT/verdict.txt"
    echo "(cp_compare_frames exit $verdict -> $OUT/verdict.txt)"

    # A gate that can be skipped without saying so is worse than no gate. The
    # exit status alone does not distinguish "a sample regressed" from "the
    # comparison never ran" — both are 1 — and the second reads like the first
    # while a cost number prints underneath it either way. Refuse to go on
    # unless a verdict table was actually produced.
    if ! grep -q "^sample " "$OUT/verdict.txt"; then
        echo "FATAL: no verdict table in $OUT/verdict.txt -- the correctness" \
             "gate did not run, so the cost delta below would be unguarded" >&2
        exit 1
    fi
fi

# The cost delta. ms/frame is the per-frame number; wall carries start up and
# the warm up second as well, so both are printed and neither is called the
# answer on its own.
if [ -n "$AGAINST" ] && [ -f "$ROOT/$AGAINST/bench/_bench.csv" ]; then
    echo "=== [$LABEL] vs [$AGAINST] ==="
    # The heredoc has to feed python, so it goes on python's own command and
    # the tee comes after: in `a | b <<PY` the shell hands the document to b,
    # which prints the script instead of running it.
    python3 - "$ROOT/$AGAINST/bench/_bench.csv" "$OUT/bench/_bench.csv" \
             "$AGAINST" "$LABEL" <<'PY' | tee "$OUT/delta.txt"
import csv, sys

def load(p):
    with open(p) as f:
        return {r["sample"]: r for r in csv.DictReader(f)}

old, new, a, b = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3], sys.argv[4]
print(f"{'sample':<24}{a+' ms':>12}{b+' ms':>12}{'change':>10}{'wall s':>18}")
to = tn = 0.0
for s in old:
    if s not in new:
        print(f"{s:<24}{'':>12}{'':>12}{'MISSING':>10}")
        continue
    o, n = float(old[s]["ms_avg"]), float(new[s]["ms_avg"])
    ow, nw = float(old[s]["wall_s"]), float(new[s]["wall_s"])
    to += o; tn += n
    pct = (n - o) / o * 100 if o else 0.0
    flag = "  <<<" if pct < -5 else ("  !!!" if pct > 5 else "")
    print(f"{s:<24}{o:>12.2f}{n:>12.2f}{pct:>9.1f}%"
          f"{ow:>9.1f}{nw:>9.1f}{flag}")
print(f"{'TOTAL':<24}{to:>12.2f}{tn:>12.2f}"
      f"{(tn-to)/to*100 if to else 0:>9.1f}%")
PY
fi

echo "=== [$LABEL] done -> $OUT ==="
