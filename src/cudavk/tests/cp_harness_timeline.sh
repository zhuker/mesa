#!/usr/bin/env bash
# Full-frame timeline for a compiled harness capture, both captures alike.
#
#   cp_harness_timeline.sh CAPTURE [ENV=VALUE ...]      CAPTURE = favorite2|favorite3
#
# cp_make_timeline.sh does this for the two gfxr captures. The captures that
# decide this project are the compiled tocpp harnesses, and nothing showed what
# they draw. Same contract as that script: one label per arm, a timing pass that
# dumps nothing so the numbers keep the paired-submit convention, then a dump
# pass, then one page.
#
#   ~/timelines/harness/CAPTURE/index.html
#
# Both pages fetch nothing, but serve them anyway for consistent relative paths:
#   python3 -m http.server -d ~/timelines 8000
set -uo pipefail
CAP=${1:?usage: cp_harness_timeline.sh favorite2|favorite3 [ENV=VALUE ...]}
shift || true
MESA=${MESA:-$HOME/mesa}
PY=${PY:-$MESA/venv/bin/python3}
ICD=${ICD:-$MESA/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json}
SHIM=${SHIM:-$HOME/favorite-cpp/submit_shim.so}
FR=$MESA/src/cudavk/tests/cp_harness_frames.py
OUT=${OUT:-$HOME/timelines/harness/$CAP}

# Each capture's own real-work window; see WORKFLOW.md 4.0. A frame is two
# submits, so absolute frame N is submit index 2N.
case "$CAP" in
  favorite3) REAL=1391; HEAVY=2200;;
  favorite2) REAL=1388; HEAVY=2735;;
  *) echo "unknown capture $CAP (favorite2|favorite3)" >&2; exit 2;;
esac
APP=$HOME/$CAP-cpp/out
[ -x "$APP/build/vulkan_app" ] || { echo "no harness at $APP/build/vulkan_app" >&2; exit 2; }

exec 9>/tmp/cudavk-gpu.lock
flock 9
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "another process holds the GPU" >&2; exit 2
fi

rm -rf "$OUT"; mkdir -p "$OUT/raw"
cd "$APP" || exit 1

echo "[1/4] Timing pass (no dumps, so the median keeps its meaning)"
env "$@" LD_PRELOAD="$SHIM" SUBMIT_TS_FILE="$OUT/submits.txt" VK_DRIVER_FILES="$ICD" \
    timeout 6000 ./build/vulkan_app >"$OUT/timing.stdout" 2>"$OUT/timing.stderr"
echo "    rc=$? rows=$(wc -l < "$OUT/submits.txt")"

echo "[2/4] Dump pass (every frame)"
env "$@" DUMP_DIR="$OUT/raw" DUMP_EVERY=1 DUMP_MAX=100000 \
    LD_PRELOAD="$SHIM" SUBMIT_TS_FILE="$OUT/dump_submits.txt" VK_DRIVER_FILES="$ICD" \
    timeout 6000 ./build/vulkan_app >"$OUT/dump.stdout" 2>"$OUT/dump.stderr"
echo "    rc=$? frames=$(ls "$OUT/raw"/frame_*.bin 2>/dev/null | wc -l)"

echo "[3/4] Encoding PNGs and thumbnails"
"$PY" "$FR" png "$OUT/raw" "$OUT" || exit 1
rm -rf "$OUT/raw"

echo "[4/4] Building the page"
"$PY" "$FR" page "$OUT" --submits "$OUT/submits.txt" --capture "$CAP" \
    --real-start "$REAL" --heavy-start "$HEAVY" -o "$OUT/index.html" || exit 1
du -sh "$OUT"
echo "TIMELINE $CAP DONE -> $OUT/index.html"
