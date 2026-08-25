#!/usr/bin/env python3
"""Fail hardware FS-argument reservation and require fatal pre-FS abort."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env.pop("CUDAPIPE_NO_TEXTURE_CACHE", None)  # hardware texture path is the default now
env["CUDAPIPE_TEXTURE_CACHE_STATS"] = "1"
env["CUDAPIPE_TEXTURE_CACHE_FAIL_FS_ARG_BEGIN"] = "1"
run = subprocess.run([sys.argv[1], "--child"], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout); sys.stderr.write(run.stderr)
if run.returncode == 0:
    raise SystemExit("FS-argument reservation fault did not fail control child")
m = re.search(r"result=control .*submit=(-?\d+) wait=(-?\d+) "
              r"counter=(\d+) expected=1024", run.stdout)
if not m or tuple(map(int, m.groups())) != (-4, 1, 0):
    raise SystemExit("FS-argument fault result/counter changed")
if "PASS " in run.stdout:
    raise SystemExit("FS-argument fatal unexpectedly reached PASS")
if not re.search(r"injected fatal hardware FS-argument reservation "
                 r"before FS attempts=0", run.stderr):
    raise SystemExit("FS was attempted before reservation fatal propagated")
hw = re.search(r"hardware texture: (\d+)/(\d+) fragment launches hit .*"
               r"fallbacks shader=(\d+) descriptor=(\d+)", run.stderr)
if not hw or tuple(map(int, hw.groups())) != (1, 1, 0, 0):
    raise SystemExit("FS-argument fault replayed or took software fallback")
