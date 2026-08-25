#!/usr/bin/env python3
"""Fail a converted-cache kernel enqueue and require fatal pre-FS abort."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env.pop("CUDAPIPE_NO_TEXTURE_CACHE", None)  # hardware texture path is the default now
env["CUDAPIPE_TEXTURE_CACHE_STATS"] = "1"
env["CUDAPIPE_TEXTURE_CACHE_FAIL_CONVERSION_ENQUEUE_AT"] = "1"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout); sys.stderr.write(run.stderr)
if run.returncode == 0:
    raise SystemExit("converted-cache enqueue fault did not fail submission")
if "PASS: 64 exact float pixels" in run.stdout:
    raise SystemExit("converted-cache fatal unexpectedly reached exact PASS")
if "failed: -4" not in run.stderr:
    raise SystemExit("converted-cache fault did not propagate device loss")
m = re.search(r"injected fatal converted-cache enqueue attempt=1 "
              r"before FS attempts=(\d+) prior_hw=(\d+)/(\d+) "
              r"prior_direct=(\d+)/(\d+)", run.stderr)
if not m or tuple(map(int, m.groups())) != (1, 1, 1, 1, 1):
    raise SystemExit("failing converted draw did not stop after sole prior A2 FS")
