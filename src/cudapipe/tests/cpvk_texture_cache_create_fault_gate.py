#!/usr/bin/env python3
"""Require fatal handling for impossible event/surface cache-create errors."""
import os
import re
import subprocess
import sys

no_replay, formats = sys.argv[1:]
base = os.environ.copy()
base["CUDAPIPE_TEXTURE_CACHE"] = "1"
base["CUDAPIPE_TEXTURE_CACHE_STATS"] = "1"

def run(args, stage):
    env = base.copy()
    env["CUDAPIPE_TEXTURE_CACHE_FAIL_CREATE_STAGE"] = str(stage)
    return subprocess.run(args, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

event = run([no_replay, "--child"], 1)
sys.stdout.write(event.stdout); sys.stderr.write(event.stderr)
if event.returncode == 0:
    raise SystemExit("event INVALID_VALUE did not fail control child")
m = re.search(r"result=control .*submit=(-?\d+) wait=(-?\d+) "
              r"counter=(\d+) expected=1024", event.stdout)
if not m or tuple(map(int, m.groups())) != (-4, 1, 0):
    raise SystemExit("event create fatal result/counter changed")
if "PASS " in event.stdout or not re.search(
        r"injected fatal cache-create stage=1 before FS attempts=0",
        event.stderr):
    raise SystemExit("event create INVALID_VALUE was swallowed or reached FS")

surface = run([formats], 2)
sys.stdout.write(surface.stdout); sys.stderr.write(surface.stderr)
if surface.returncode == 0 or "PASS: 64 exact float pixels" in surface.stdout:
    raise SystemExit("surface INVALID_VALUE did not abort formats submission")
if "failed: -4" not in surface.stderr or not re.search(
        r"injected fatal cache-create stage=2 before FS attempts=1",
        surface.stderr):
    raise SystemExit("surface create fatal did not stop after sole prior A2 FS")
