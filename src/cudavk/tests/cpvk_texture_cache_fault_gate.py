#!/usr/bin/env python3
"""Fail the cache-table upload and require fatal pre-FS behavior."""
import os
import re
import subprocess
import sys

env = os.environ.copy()
env.pop("CUDAVK_NO_TEXTURE_CACHE", None)  # hardware texture path is the default now
env["CUDAVK_TEXTURE_CACHE_STATS"] = "1"
env["CUDAVK_TEXTURE_CACHE_FAIL_TABLE_UPLOAD_AT"] = "1"
run = subprocess.run([sys.argv[1]], env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sys.stdout.write(run.stdout)
sys.stderr.write(run.stderr)
if run.returncode == 0:
    raise SystemExit("faulted table upload did not fail the submission")
if "failed: -4" not in run.stderr:
    raise SystemExit("table upload fault did not propagate VK_ERROR_DEVICE_LOST")
m = re.search(r"fatal table upload=\d+ before fragment attempt, fs_attempts=(\d+)",
              run.stderr)
if not m or int(m.group(1)) != 0:
    raise SystemExit("fragment shader was attempted after table upload fault")
if "PASS binding" in run.stdout:
    raise SystemExit("faulted draw unexpectedly produced the normal pixel PASS")
