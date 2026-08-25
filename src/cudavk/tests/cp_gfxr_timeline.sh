#!/usr/bin/env bash
# Run independent timing and image replays, then build a clickable HTML report.
set -euo pipefail

usage() {
    echo "usage: $0 CAPTURE.gfxr OUTPUT_DIR [ICD.json]" >&2
    exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage
CAPTURE=$(realpath "$1")
OUT=$(realpath -m "$2")
ICD=${3:-${VK_DRIVER_FILES:-$HOME/mesa/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json}}
HERE=$(cd "$(dirname "$0")" && pwd)
FRAMES="$HERE/cp_gfxr_frames.py"
REPLAY=${GFXRECON_REPLAY:-$HOME/gfxreconstruct/build/tools/replay/gfxrecon-replay}
CONVERT=${GFXRECON_CONVERT:-$HOME/gfxreconstruct/build/tools/convert/gfxrecon-convert}
PLUGIN=${GFXR_FPS_PLUGIN:-$HOME/claude-scratchpad/perf16/fps_plugin.so}
PER_FRAME=${GFXR_SUBMITS_PER_FRAME:-2}

for path in "$CAPTURE" "$ICD" "$REPLAY" "$CONVERT" "$PLUGIN"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "output already exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT/timing" "$OUT/frames"

echo "[1/6] Indexing offscreen frame readbacks"
python3 "$FRAMES" index "$CAPTURE" -o "$OUT/blocks.tsv" \
    --gfxrecon-convert "$CONVERT"
python3 "$FRAMES" plan "$OUT/blocks.tsv" --frames all --image all \
    -o "$OUT/dump.json"

echo "[2/6] Timing every frame (replay 1 of 2)"
VK_DRIVER_FILES="$ICD" "$REPLAY" -m remap --remove-unsupported \
    --replay-event-plugin-path "$PLUGIN" \
    --replay-event-plugin-params "$OUT/timing/submits.txt" \
    "$CAPTURE" >"$OUT/timing/replay.log" 2>&1

echo "[3/6] Extracting every frame (replay 2 of 2)"
python3 "$FRAMES" replay "$CAPTURE" "$OUT/dump.json" --icd "$ICD" \
    --out "$OUT/frames" --gfxrecon-replay "$REPLAY"

echo "[4/6] Encoding PNGs"
python3 "$FRAMES" png "$OUT/frames" >"$OUT/png.log"

echo "[5/6] Building timeline"
python3 "$FRAMES" timeline "$OUT/timing/submits.txt" "$OUT/frames" \
    --per-frame "$PER_FRAME" --title "$(basename "$CAPTURE")" \
    -o "$OUT/index.html"

echo "[6/6] Summary"
python3 "$FRAMES" fps "$OUT/timing/submits.txt" --per-frame "$PER_FRAME"
echo "Open $OUT/index.html"
