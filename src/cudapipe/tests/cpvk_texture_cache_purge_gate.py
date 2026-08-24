#!/usr/bin/env python3
"""Purge a live derived cache, then require exact rebuild pixels and stats."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env["CUDAPIPE_TEXTURE_CACHE"] = "1"
env["CUDAPIPE_TEXTURE_CACHE_STATS"] = "1"
env["CUDAPIPE_TEXTURE_CACHE_PURGE_AT_PREFLIGHT"] = "2"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout)
sys.stderr.write(run.stderr)
if run.returncode:
    raise SystemExit(run.returncode)
if "PASS binding 0 and binding 1" not in run.stdout:
    raise SystemExit("missing exact external pixel PASS after purge/rebuild")
hw = re.search(r"hardware texture: (\d+)/(\d+) fragment launches hit", run.stderr)
resources = re.search(
    r"texture cache resources: hits=(\d+) fallbacks=(\d+) rebuilds=(\d+) .*"
    r"arrays=(\d+) objects=(\d+).*purges=(\d+) purge_reclaimed_bytes=(\d+) "
    r"purge_reclaimed_arrays=(\d+) purge_reclaimed_objects=(\d+)",
    run.stderr)
if not hw or hw.group(1) != hw.group(2) or int(hw.group(1)) <= 0:
    raise SystemExit("purge run did not keep every launch on hardware")
if not resources:
    raise SystemExit("missing cumulative purge resource statistics")
hits, fallbacks, rebuilds, arrays, objects, purges, reclaimed, parray, pobj = map(
    int, resources.groups())
if (hits <= 0 or fallbacks != 0 or rebuilds < 3 or arrays < 2 or objects < 4 or
        purges != 1 or reclaimed <= 0 or parray < 1 or pobj < 2):
    raise SystemExit("live purge did not reclaim and rebuild the derived cache")
