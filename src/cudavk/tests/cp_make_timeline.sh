#!/usr/bin/env bash
# Build both captures' full-frame timelines for one iteration label.
#
#   cp_make_timeline.sh ITERATION [ENV=VALUE ...]
#
# Writes ~/timelines/ITERATION/old and ~/timelines/ITERATION/cross, then
# rebuilds ~/timelines/index.html. Any extra arguments are environment
# assignments handed to both replays, e.g. CUDAVK_OPAQUE_STREAMS=1.
set -uo pipefail
ITER=${1:?usage: cp_make_timeline.sh ITERATION [ENV=VALUE ...]}
shift || true
MESA=${MESA:-$HOME/mesa}
ICD=${ICD:-$MESA/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json}
T=$MESA/src/cudavk/tests/cp_gfxr_timeline.sh
ROOT=/home/alexzhukov/timelines
OLD=/home/alexzhukov/headless_streamer_20260814T155742.gfxr
CROSS=/home/alexzhukov/headless_streamer_1818_20260817T173522.gfxr

for i in $(seq 1 240); do
  if ! pgrep -f 'gfxrecon-replay|cp_perf_run' >/dev/null && \
     [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then break; fi
  sleep 5
done

mkdir -p "$ROOT/$ITER"
echo "=== $ITER / old ==="
env "$@" "$T" "$OLD"   "$ROOT/$ITER/old"   "$ICD" || exit 1
echo "=== $ITER / cross ==="
env "$@" "$T" "$CROSS" "$ROOT/$ITER/cross" "$ICD" || exit 1
python3 "$(dirname "$0")/cp_timeline_index.py"
echo "TIMELINE ${ITER} DONE"
