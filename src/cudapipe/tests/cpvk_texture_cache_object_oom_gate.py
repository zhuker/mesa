#!/usr/bin/env python3
"""OOM one texture object and require whole-launch fallback then HW recovery."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env.pop("CUDAPIPE_NO_TEXTURE_CACHE", None)  # hardware texture path is the default now
env["CUDAPIPE_TEXTURE_CACHE_STATS"] = "1"
env["CUDAPIPE_TEXTURE_CACHE_FAIL_OBJECT_CREATE_AT"] = "1"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout); sys.stderr.write(run.stderr)
if run.returncode or "PASS binding 0 and binding 1" not in run.stdout:
    raise SystemExit(run.returncode or "texture-object OOM changed pixels")
hw = re.search(r"hardware texture: (\d+)/(\d+) fragment launches hit .*"
               r"fallbacks shader=(\d+) descriptor=(\d+)", run.stderr)
res = re.search(r"texture cache resources: hits=(\d+) fallbacks=(\d+) "
                r"rebuilds=(\d+).*arrays=(\d+) objects=(\d+) "
                r"alloc_failures=(\d+) object_failures=(\d+)", run.stderr)
if not hw or tuple(map(int, hw.groups())) != (3,4,0,1):
    raise SystemExit("object OOM was not one whole-launch fallback plus recovery")
if not res or tuple(map(int, res.groups())) != (6,1,2,1,2,0,1):
    raise SystemExit("object OOM resource/failure counts changed")
