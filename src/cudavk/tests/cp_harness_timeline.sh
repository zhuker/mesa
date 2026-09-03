#!/usr/bin/env bash
# Full-frame timelines of BOTH compiled-harness captures for one iteration.
#
#   cp_harness_timeline.sh ITERATION [ENV=VALUE ...]
#
# The same contract as cp_make_timeline.sh, which does this for the two gfxr
# captures: the label is the label cp_iterate.sh was given, so a row in
# ~/timelines/index.html and a row in ~/git/Vulkan/build/iter/iterations.html
# are the same iteration and the index reads that iteration's DESC and commit.
# Using a different label buys two half-records that cannot be joined.
#
#   ~/timelines/ITERATION/favorite2/index.html
#   ~/timelines/ITERATION/favorite3/index.html
#
# Extra arguments are environment assignments handed to both replays, which is
# how an arm is selected -- one iteration per arm, not one iteration with two
# meanings:
#
#   cp_harness_timeline.sh vslane-off CUDAVK_NO_VS_LANE=1
#
# A pair costs about 6 GB of PNG and roughly fifteen minutes. Every frame is
# dumped raw, encoded, and the raw removed. Read them through the static
# server over the home directory:
#   http://localhost:8000/timelines/ITERATION/favorite2/index.html
set -uo pipefail
ITER=${1:?usage: cp_harness_timeline.sh ITERATION [ENV=VALUE ...]}
shift || true
MESA=${MESA:-$HOME/mesa}
PY=${PY:-$MESA/venv/bin/python3}
ICD=${ICD:-$MESA/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json}
SHIM=${SHIM:-$HOME/favorite-cpp/submit_shim.so}
FR=$MESA/src/cudavk/tests/cp_harness_frames.py
ROOT=${ROOT:-$HOME/timelines}

exec 9>/tmp/cudavk-gpu.lock
flock 9
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "another process holds the GPU" >&2; exit 2
fi

for CAP in favorite2 favorite3; do
  case "$CAP" in
    favorite3) REAL=1391; HEAVY=2200;;
    favorite2) REAL=1388; HEAVY=2735;;
  esac
  APP=$HOME/$CAP-cpp/out
  [ -x "$APP/build/vulkan_app" ] || { echo "no harness at $APP" >&2; exit 2; }
  OUT=$ROOT/$ITER/$CAP
  # Same refusal as cp_gfxr_timeline.sh: a re-render wants the old one removed
  # on purpose, so a half-overwritten pair cannot be read as a whole one.
  [ -e "$OUT" ] && { echo "$OUT exists; remove it to re-render" >&2; exit 2; }
  mkdir -p "$OUT/timing" "$OUT/frames" "$OUT/raw"
  cd "$APP" || exit 1

  echo "=== $ITER / $CAP ==="
  echo "[1/4] Timing pass (dumps nothing, so the median keeps its meaning)"
  env "$@" LD_PRELOAD="$SHIM" SUBMIT_TS_FILE="$OUT/timing/submits.txt" \
      VK_DRIVER_FILES="$ICD" timeout 6000 ./build/vulkan_app \
      >"$OUT/timing/stdout" 2>"$OUT/timing/stderr"
  echo "    rc=$? rows=$(wc -l < "$OUT/timing/submits.txt")"

  echo "[2/4] Dump pass (every frame)"
  env "$@" DUMP_DIR="$OUT/raw" DUMP_EVERY=1 DUMP_MAX=100000 \
      LD_PRELOAD="$SHIM" SUBMIT_TS_FILE="$OUT/dump_submits.txt" \
      VK_DRIVER_FILES="$ICD" timeout 6000 ./build/vulkan_app \
      >"$OUT/dump.stdout" 2>"$OUT/dump.stderr"
  echo "    rc=$? frames=$(ls "$OUT/raw"/frame_*.bin 2>/dev/null | wc -l)"

  echo "[3/4] Encoding PNGs and thumbnails"
  "$PY" "$FR" png "$OUT/raw" "$OUT/frames" || exit 1
  rm -rf "$OUT/raw"

  echo "[4/4] Building the page"
  # The same page cp_gfxr_timeline.sh builds -- chart, seek, frame view. Do not
  # write another one: one timeline format for all four captures is the point.
  "$PY" "$MESA/src/cudavk/tests/cp_gfxr_frames.py" timeline \
      "$OUT/timing/submits.txt" "$OUT/frames" --per-frame 2 \
      --title "$CAP - $ITER" -o "$OUT/index.html" || exit 1
done

"$PY" "$MESA/src/cudavk/tests/cp_timeline_index.py"
du -sh "$ROOT/$ITER"
echo "TIMELINE $ITER DONE -> $ROOT/$ITER/{favorite2,favorite3}/index.html"
