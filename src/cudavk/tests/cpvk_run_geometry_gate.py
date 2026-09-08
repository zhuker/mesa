#!/usr/bin/env python3
"""Gate the default-off opaque run collector and its safe VS grouping."""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

exe = sys.argv[1]
out = Path("/tmp/lat/native_batchtex.ppm")

def arm(enabled):
    env = os.environ.copy()
    env["CUDAVK_BATCH_MAX"] = "1"
    env["CUDAVK_RUN_GEOMETRY"] = "1" if enabled else "0"
    if enabled:
        env["CUDAVK_DEBUG_EPISODE"] = "1"
    p = subprocess.run([exe], env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120)
    if p.returncode:
        raise SystemExit(p.stdout)
    return hashlib.sha256(out.read_bytes()).hexdigest(), p.stdout

base, _ = arm(False)
on, log = arm(True)
if base != on:
    raise SystemExit("run geometry changed pixels: %s != %s\n%s" % (base, on, log))
m = re.search(r"run-geometry-plan: batches=(\d+) groups=(\d+) members=(\d+)", log)
if not m:
    raise SystemExit("collector/group planner was not reached\n" + log)
batches, groups, members = map(int, m.groups())
if batches < 2 or groups >= batches or members != batches:
    raise SystemExit("expected a real many-to-one group, got %s\n%s" % (m.group(0), log))
launches = re.findall(r"run-geometry-launch: members=(\d+)", log)
if len(launches) != groups or sum(map(int, launches)) != members:
    raise SystemExit("group launches did not replace every member VS launch\n" + log)
print("run geometry gate: %d member VS launches -> %d, pixels identical" %
      (batches, groups))
