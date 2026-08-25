#!/usr/bin/env python3
"""Require the positive native CUDA texture-cache path and epoch rebuild."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env.pop("CUDAVK_NO_TEXTURE_CACHE", None)  # hardware texture path is the default now
env["CUDAVK_TEXTURE_CACHE_STATS"] = "1"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout)
sys.stderr.write(run.stderr)
if run.returncode:
    raise SystemExit(run.returncode)
if "PASS binding 0 and binding 1" not in run.stdout:
    raise SystemExit("missing external pixel PASS")
hw = re.search(r"hardware texture: (\d+)/(\d+) fragment launches hit", run.stderr)
resources = re.search(
    r"texture cache resources: hits=(\d+) fallbacks=(\d+) rebuilds=(\d+) .*"
    r"arrays=(\d+) objects=(\d+)", run.stderr)
if not hw or int(hw.group(1)) <= 0 or hw.group(1) != hw.group(2):
    raise SystemExit("hardware texture cache did not hit every launch")
if not resources:
    raise SystemExit("missing texture cache resource statistics")
hits, fallbacks, rebuilds, arrays, objects = map(int, resources.groups())
if hits <= 0 or fallbacks != 0 or rebuilds < 2 or arrays < 1 or objects < 2:
    raise SystemExit("cache did not materialize two sampler objects and rebuild mutation epoch")
