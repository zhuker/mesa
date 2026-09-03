#!/usr/bin/env bash
# Alternating A/B of one change on a compiled harness capture, with the gates.
#
#   cp_harness_ab.sh CAPTURE LABEL "CONTROL_ENV" ["CANDIDATE_ENV"] [ROUNDS]
#
#   cp_harness_ab.sh favorite3 vslane "CUDAVK_NO_VS_LANE=1" "" 3
#
# This is the loop every measurement in this project needs, written once. It
# alternates the arms in ONE session (WORKFLOW 4.2), runs each arm ROUNDS times
# (3 is the floor; the B200 needs 3 because its session drift reaches 1.3 ms),
# and refuses to print a median until each run has passed its gates: expected
# exit code, stdout hash, full timestamp population, and 18/18 sentinel frames.
# A run that fails any of them is reported and the comparison is abandoned --
# a replay that dies early produces a fast, meaningless median.
#
# ICD selects the build; point it at a worktree to measure a branch:
#   ICD=~/mesa-vslane/build/src/cudavk/cudavk_devenv_icd.x86_64.json \
#     cp_harness_ab.sh favorite3 vslane "CUDAVK_NO_VS_LANE=1"
set -uo pipefail
CAP=${1:?usage: cp_harness_ab.sh CAPTURE LABEL "CONTROL_ENV" ["CAND_ENV"] [ROUNDS]}
LABEL=${2:?label}
CTRL_ENV=${3:?control env, e.g. "CUDAVK_NO_FOO=1"}
CAND_ENV=${4:-}
ROUNDS=${5:-3}
MESA=${MESA:-$HOME/mesa}
# The venv python only exists on the workstation; the B200 workspace has none.
# Nothing here needs numpy, so fall back to whatever python3 is on PATH rather
# than failing every sentinel check and reporting a gate failure that is really
# a missing interpreter.
PY=${PY:-$MESA/venv/bin/python3}
[ -x "$PY" ] || PY=$(command -v python3)
[ -x "$PY" ] || { echo "no python3" >&2; exit 2; }
ICD=${ICD:-$MESA/build-cudavk/src/cudavk/cudavk_devenv_icd.x86_64.json}
SHIM=${SHIM:-$HOME/favorite-cpp/submit_shim.so}
OUT=${OUT:-/tmp/harness-ab/$LABEL}

case "$CAP" in
  favorite3) ROWS=6947; RC=0; REAL=1391; HEAVY=2200; HEAVY_END=3150
             HASH=e4a60a7156141649a4c16660f674025af7b9fc3fe2702f4f57845527075d058d
             CTRLFRAMES=${CTRLFRAMES:-/tmp/hiz-f3-8/ctrl};;
  favorite2) ROWS=6965; RC=0; REAL=1388; HEAVY=2735; HEAVY_END=0
             HASH=0240ff4ec576c62b49461a384d90e0c25463f69ecdaf1b2286e92d14264d947e
             CTRLFRAMES=${CTRLFRAMES:-/tmp/f2-rtx-check};;
  *) echo "unknown capture $CAP" >&2; exit 2;;
esac
APP=$HOME/$CAP-cpp/out
[ -x "$APP/build/vulkan_app" ] || { echo "no harness at $APP" >&2; exit 2; }
[ -f "$ICD" ] || { echo "no ICD at $ICD" >&2; exit 2; }   # the lead-F mistake

exec 9>/tmp/cudavk-gpu.lock
flock 9
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "another process holds the GPU" >&2; exit 2
fi
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$APP" || exit 1
echo "capture=$CAP icd=$ICD rounds=$ROUNDS"
echo "control   : ${CTRL_ENV:-<none>}"
echo "candidate : ${CAND_ENV:-<defaults>}"

fail=0
for r in $(seq 1 "$ROUNDS"); do
  for arm in c n; do
    d=$OUT/$arm$r; mkdir -p "$d"
    case $arm in c) E=$CTRL_ENV;; n) E=$CAND_ENV;; esac
    env $E DUMP_DIR="$d" DUMP_EVERY=200 DUMP_MAX=20 LD_PRELOAD="$SHIM" \
        SUBMIT_TS_FILE="$d/ts.txt" VK_DRIVER_FILES="$ICD" \
        timeout 6000 ./build/vulkan_app >"$d/stdout" 2>"$d/stderr"
    rc=$?; h=$(sha256sum "$d/stdout" | cut -d' ' -f1); n=$(wc -l < "$d/ts.txt")
    bad=$($PY - "$CTRLFRAMES" "$d" <<'PY'
import sys, glob, os, hashlib
ref, cand = sys.argv[1], sys.argv[2]
def sums(p):
    return {os.path.basename(f): hashlib.sha256(open(f,'rb').read()).hexdigest()
            for f in glob.glob(os.path.join(p, 'frame_*.bin'))}
a, b = sums(ref), sums(cand)
common = set(a) & set(b)
print(len(common) - sum(1 for k in common if a[k] == b[k]) if len(common) == 18 else 99)
PY
)
    ok=yes
    [ "$rc" = "$RC" ] || { ok="rc=$rc want $RC"; fail=1; }
    [ "$h" = "$HASH" ] || { ok="hash mismatch"; fail=1; }
    [ "$n" = "$ROWS" ] || { ok="rows=$n want $ROWS"; fail=1; }
    [ "$bad" = "0" ]   || { ok="sentinels bad=$bad"; fail=1; }
    printf '  %s%-2s rc=%-4s rows=%-5s sentinels=%-3s %s\n' "$arm" "$r" "$rc" "$n" "$bad" \
           "$([ "$ok" = yes ] && echo OK || echo "FAILED: $ok")"
  done
done
if [ "$fail" != 0 ]; then
  echo "GATES FAILED -- no median is reported; fix the run before trusting any timing." >&2
  exit 1
fi
$PY - "$OUT" "$ROUNDS" "$REAL" "$HEAVY" "$HEAVY_END" <<'PY'
import sys, statistics as st
out, rounds, real, heavy, heavy_end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
def med(path, lo, hi=0):
    ts = [int(l.split()[0]) for l in open(path) if len(l.split()) == 2]
    end = hi * 2 if hi else len(ts) - 2
    d = [(ts[i+2]-ts[i])/1e6 for i in range(lo*2, min(end, len(ts)-2), 2)]
    return st.median(d)
for name, lo, hi in (("whole", real, 0), ("heavy", heavy, heavy_end)):
    c = [med(f"{out}/c{r}/ts.txt", lo, hi) for r in range(1, rounds+1)]
    n = [med(f"{out}/n{r}/ts.txt", lo, hi) for r in range(1, rounds+1)]
    mc, mn = st.median(c), st.median(n)
    disjoint = max(n) < min(c) or min(n) > max(c)
    print(f"{name:6s} control {[round(x,4) for x in c]} -> {mc:.4f}")
    print(f"{name:6s} cand    {[round(x,4) for x in n]} -> {mn:.4f}")
    print(f"{name:6s} DELTA   {mn-mc:+.4f} ms/frame   arms {'DISJOINT' if disjoint else 'OVERLAP (inconclusive)'}")
PY
