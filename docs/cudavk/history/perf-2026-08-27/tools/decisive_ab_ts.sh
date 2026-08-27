#!/bin/bash
# Decisive alternating measurement with per-run epoch timestamps, so that a
# foreign GPU tenant seen by the sampler can be attributed to named runs.
set -u
cd /home/alexzhukov
D=${D:?set D}
I=${I:?set I}
CAND=${CAND:?set CAND}
CTRL=${CTRL-}
BASE_ENV=${BASE_ENV-}
R=/home/alexzhukov/gfxreconstruct/build/tools/replay/gfxrecon-replay
P=/home/alexzhukov/claude-scratchpad/perf16/fps_plugin.so
OLD=/home/alexzhukov/headless_streamer_20260814T155742.gfxr
CROSS=/home/alexzhukov/headless_streamer_1818_20260817T173522.gfxr
export LD_LIBRARY_PATH="$HOME/vulkan-sdk/1.4.357.1/x86_64/lib"
run() { n=$1; cap=$2; shift 2
  mkdir -p "$D/$n"
  t0=$(date +%s)
  env VK_DRIVER_FILES=$I $BASE_ENV "$@" \
    $R -m remap --remove-unsupported \
       --replay-event-plugin-path $P \
       --replay-event-plugin-params "$D/$n/submits.txt" \
       "$cap" >"$D/$n/stdout" 2>"$D/$n/stderr"
  rc=$?; t1=$(date +%s)
  echo $rc >"$D/$n/rc"
  echo "$t0 $t1" >"$D/$n/window"
  sha256sum "$D/$n/stdout" | cut -d' ' -f1 >"$D/$n/stdout.sha256"
  echo "$n rc=$rc submits=$(wc -l < $D/$n/submits.txt) window=$t0-$t1" >> "$D/progress.log"
}
mkdir -p "$D"
printf 'cand=%s\nctrl=%s\nbase=%s\nicd=%s\n' "$CAND" "$CTRL" "$BASE_ENV" "$I" > "$D/arms.txt"
for i in 1 2 3 4 5 6; do
  run "old-cand-$i" "$OLD"  $CAND
  run "old-ctrl-$i" "$OLD"  $CTRL
done
for i in 1 2 3 4; do
  run "cross-cand-$i" "$CROSS" $CAND
  run "cross-ctrl-$i" "$CROSS" $CTRL
done
echo DONE > "$D/done"
