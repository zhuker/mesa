#!/usr/bin/env python3
"""Require exact focused format pixels and the isolated HW admission split."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env["CUDAPIPE_TEXTURE_CACHE"] = "1"
env["CUDAPIPE_TEXTURE_CACHE_STATS"] = "1"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout)
sys.stderr.write(run.stderr)
if run.returncode:
    raise SystemExit(run.returncode)
if "PASS: 64 exact float pixels" not in run.stdout:
    raise SystemExit("missing exact focused-format pixel PASS")
hw = re.search(
    r"hardware texture: (\d+)/(\d+) fragment launches hit .*"
    r"fallbacks shader=(\d+) descriptor=(\d+).*modes inline=(\d+) fused=(\d+)",
    run.stderr)
res = re.search(
    r"texture cache resources: hits=(\d+) fallbacks=(\d+) rebuilds=(\d+) .*"
    r"arrays=(\d+) objects=(\d+)", run.stderr)
if not hw or tuple(map(int, hw.groups())) != (7, 8, 0, 1, 0, 7):
    raise SystemExit("focused formats did not produce isolated 7-HW/1-base-mip split")
if not res or tuple(map(int, res.groups())) != (42, 1, 7, 7, 7):
    raise SystemExit("focused format cache resource counts changed")
