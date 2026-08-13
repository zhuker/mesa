#!/bin/bash
#
# One optimisation iteration: build, time the sixty frames, render them, and
# say what changed against another iteration.
#
#   DESC="what this tried" cp_iterate.sh LABEL [COMPARE_LABEL]
#
# Everything an iteration produced lands under $VULKAN/build/iter/LABEL:
#
#   bench/          the timed pass — _bench.csv, per-sample frame times
#   <sample>/       the stored frames, one directory of them per sample
#   _render/        the storing pass's own timing and GPU csvs
#   verdict.txt     cp_compare_frames.py against the stored nvidia reference
#   delta.txt       ms/frame against COMPARE_LABEL, per sample and in total
#   iteration.json  all of the above as one record, plus the commit, DESC, and
#                   which samples moved past 5% either way
#
# build/iter is then exactly the root cp_compare_frames.py wants — one directory
# per thing being compared — with the nvidia reference sitting beside the
# iterations as build/iter/nvidia.
#
# _render/ carries its underscore for a reason: cp_compare_frames.py takes its
# sample list from the reference directory and skips names starting with one,
# and the reference has a _render/ of its own. bench/ needs no underscore
# because only iterations have one, and the reference is what is enumerated.
#
# DESC is worth setting. A label and a number stop meaning anything within a
# day or so of the run; what the iteration was trying is the part nobody can
# reconstruct afterwards. It also ends up on the pages cp_iter_report.py builds
# over every iteration's json.
#
# The two passes render the same work; see TESTING.md. The reference frames are
# rendered once and reused, because NVIDIA's output does not change when
# cudapipe does.
#
# Why a script rather than the commands: a performance claim is only worth
# something if the run behind it can be repeated, and the flags that have to
# match between the timed and the stored pass are exactly the ones that are
# easy to get wrong by hand.
#
# FRAMES (60) is the stored pass; BENCH_FRAMES (600) is the timed one. Raise
# BENCH_FRAMES for a final confirmation — 6000 takes about twenty-five minutes
# and puts the spread below a tenth of a percent.
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
# Two frame counts, because the two passes answer different questions.
#
# The stored pass is the correctness gate and the report: sixty frames is the
# animation the samples were set up to render, and every one of them is kept as
# a png, so the count is bounded by what is worth storing and looking at.
#
# The timed pass measures cost, where sixty frames is far too few. A sample's
# process spends a couple of seconds on start-up, shader compilation, the warm
# up second and teardown, so at sixty frames the render loop is a minority of
# it — instancing renders for 1.6 s of 4.3 s — and the mean is taken over a
# window short enough that run-to-run spread swamps the changes being looked
# for. Six hundred frames puts every sample above ten seconds of rendering.
#
# The two therefore no longer render the same number of frames, and TESTING.md
# is otherwise emphatic that they must render the same work. They still do: the
# orbit is periodic, so a longer run covers the same path again rather than a
# different one, and the totals agree to 0.8% — inside the sweep's own spread —
# with the largest samples agreeing to within 1%. That was measured before this
# split was made; see TESTING.md.
FRAMES=${FRAMES:-60}
BENCH_FRAMES=${BENCH_FRAMES:-600}

case ${DRIVER:=cudapipe} in
    cudapipe) ICD=$M/cudapipe/cudapipe_devenv_icd.x86_64.json ;;
    llvmpipe) ICD=$M/lavapipe/lvp_devenv_icd.x86_64.json ;;
    nvidia)   ICD= ;;
    *) echo "DRIVER must be cudapipe, llvmpipe or nvidia" >&2; exit 1 ;;
esac

ROOT=$VULKAN/build/iter
OUT=$ROOT/$LABEL
FRAMEROOT=$ROOT
REF=$ROOT/nvidia

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

echo "=== [$LABEL] timing $BENCH_FRAMES frames ($DRIVER) ==="
"$T/cp_perf_run.sh" "$DRIVER" "$ICD" "$OUT/bench" "$BENCH_FRAMES" 1 || exit 1

if [ "${BENCH_ONLY:-0}" != "1" ]; then
    if [ ! -d "$REF" ]; then
        echo "=== rendering the nvidia reference (once) ==="
        "$T/cp_perf_run.sh" nvidia "" "$REF" "$FRAMES" || exit 1
    fi

    echo "=== [$LABEL] rendering the same $FRAMES frames ==="
    # Frames are never reused between iterations: a partial re-render leaves
    # one sample's old images beside another's new ones, which reads as a
    # change in the sample that was not re-run. Only the sample directories go
    # — bench/ holds the timed pass that already ran, and blowing the whole
    # iteration directory away here would take it with them.
    for d in "${OUT:?}"/*/; do
        case $(basename "$d") in bench|_render|_logs) ;; *) rm -rf "$d" ;; esac
    done
    "$T/cp_perf_run.sh" "$DRIVER" "$ICD" "$OUT" "$FRAMES" || exit 1
    # The storing pass writes its own _gpu.csv and _timing.csv beside the
    # frames; keep them, but out of the way of the sample directories.
    mkdir -p "$OUT/_render"
    for f in "$OUT"/_gpu.csv "$OUT"/_gpu_procs.csv "$OUT"/_timing.csv \
             "$OUT"/_summary.txt; do
        [ -f "$f" ] && mv "$f" "$OUT/_render/"
    done

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

# Record the iteration as one json: the commit, what it was trying, the cost,
# the delta against what it was compared to, which samples moved past 5% either
# way, and what the correctness gate said. Written every time so that no
# iteration depends on someone having remembered to describe it afterwards.
"$MESA/venv/bin/python3" "$T/cp_iter_report.py" --root "$ROOT" --mesa "$MESA" \
    record "$LABEL" --against "$AGAINST" --driver "$DRIVER" --frames "$FRAMES" \
    --bench-frames "$BENCH_FRAMES" --desc "${DESC:-}"

# And collect it, for the same reason the record is written unconditionally.
# `iterations.json` is what both pages fetch, and it is rebuilt from the
# per-iteration records rather than appended to — so an iteration that ran, was
# recorded and passed the gate stayed invisible in the history until someone
# remembered a second command, which is the same failure this script already
# refuses a reused label and an unrun gate to prevent. Rebuilding is a re-read
# of the records on disk and costs a fraction of a second against a pass of
# several minutes.
"$MESA/venv/bin/python3" "$T/cp_iter_report.py" --root "$ROOT" --mesa "$MESA" \
    page >/dev/null || echo "warning: could not rebuild iterations.json" >&2

echo "=== [$LABEL] done -> $OUT ==="
