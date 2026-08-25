#!/usr/bin/env python3
"""OOM one live renderer allocation, then require purge/retry/rebuild."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env.pop("CUDAVK_NO_TEXTURE_CACHE", None)  # hardware texture path is the default now
env["CUDAVK_TEXTURE_CACHE_STATS"] = "1"
env["CUDAVK_PLAN_STATS"] = "1"
env["CUDAVK_TEXTURE_CACHE_FAIL_AUTHORITATIVE_ALLOC_AT_PREFLIGHT"] = "2"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout)
sys.stderr.write(run.stderr)
if run.returncode:
    raise SystemExit(run.returncode)
if "PASS binding 0 and binding 1" not in run.stdout:
    raise SystemExit("missing exact pixel PASS after authoritative OOM retry")
hw = re.search(r"hardware texture: (\d+)/(\d+) fragment launches hit", run.stderr)
res = re.search(r"texture cache resources: .*rebuilds=(\d+).*purges=(\d+) "
                r"purge_reclaimed_bytes=(\d+)", run.stderr)
retry = re.search(r"authoritative_oom_retries=(\d+)", run.stderr)
if not hw or hw.group(1) != hw.group(2):
    raise SystemExit("OOM retry did not preserve all-hardware launch admission")
if not res or int(res.group(1)) < 3 or int(res.group(2)) != 1 or int(res.group(3)) <= 0:
    raise SystemExit("authoritative OOM did not purge and rebuild derived cache")
if not retry or int(retry.group(1)) != 1:
    raise SystemExit("authoritative renderer allocation was not retried exactly once")
