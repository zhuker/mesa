#!/usr/bin/env python3
"""Exact pixel gate for the side-stream classic run visibility arm."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

exe = sys.argv[1]
out = Path("/tmp/lat/native_batchtex.ppm")

def arm(enabled, serial=False):
    env = os.environ.copy()
    env.update({
        "CUDAVK_BATCH_MAX": "1",
        "CUDAVK_RUN_GEOMETRY": "1" if enabled else "0",
        "CUDAVK_RUN_CLIP_BIN": "0",
        "CUDAVK_RUN_CLIP_DIRECT": "0",
        "CUDAVK_RUN_CLIP_CLASSIC": "1" if enabled else "0",
        "CUDAVK_NO_RUN_PREP_STREAMS": "1" if serial else "0",
        "CPVK_TEST_DEPTH_LEQUAL": "1",
        "CUDAVK_DEBUG_EPISODE": "1",
    })
    p = subprocess.run([exe], env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120)
    if p.returncode:
        raise SystemExit(p.stdout)
    return hashlib.sha256(out.read_bytes()).hexdigest(), p.stdout

base, _ = arm(False)
side, log = arm(True)
serial, serial_log = arm(True, serial=True)
if side != base or serial != base:
    raise SystemExit("classic run visibility changed pixels: %s %s %s\n%s" %
                     (base, side, serial, log))
if ("run-clip-classic-commit:" not in log or
        "run-clip-bin-prepare: batches=9 streams=8 join=0" not in log):
    raise SystemExit("side-stream classic transaction did not engage\n" + log)
if ("run-clip-classic-commit:" not in serial_log or
        "run-clip-bin-prepare: batches=9 streams=1 join=0" not in serial_log):
    raise SystemExit("serial classic revert did not engage\n" + serial_log)
print("run clip classic gate: side and serial hashes exact -- PASS")
