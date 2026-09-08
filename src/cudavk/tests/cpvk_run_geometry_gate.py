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

def arm(enabled, split=True, clip_bin=False):
    env = os.environ.copy()
    if split:
        env["CUDAVK_BATCH_MAX"] = "1"
    else:
        env.pop("CUDAVK_BATCH_MAX", None)
    env["CUDAVK_RUN_GEOMETRY"] = "1" if enabled else "0"
    env["CUDAVK_RUN_CLIP_BIN"] = "1" if clip_bin else "0"
    if enabled or clip_bin:
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
metadata_re = (r"run-geometry-metadata: draws=(\d+) slices=(\d+) "
               r"count=(\d+) prim-shift=(\d+) clip-rects=(\d+) "
               r"prepare=(\d+)")
one_meta = [tuple(map(int, x)) for x in re.findall(metadata_re, log)]
if not any(draws == 1 and slices == count == prim_shift == prepare == 0
           for draws, slices, count, prim_shift, _rects, prepare in one_meta):
    raise SystemExit("one-draw descriptor metadata was not exact\n" + log)

# The broad prototype also grouped existing multi-draw batches. Full stdout
# hashes missed its corruption, so keep a pixel gate on the ordinary batching
# arm: those batches must take the old fused launch exactly.
multi_base, _ = arm(False, split=False)
multi_on, multi_log = arm(True, split=False)
if multi_base != multi_on:
    raise SystemExit("multi-draw fallback changed pixels: %s != %s" %
                     (multi_base, multi_on))
multi_meta = [tuple(map(int, x)) for x in re.findall(metadata_re, multi_log)]
if not any(draws > 1 and slices == 1 and count == draws and
           prim_shift == prepare == 0
           for draws, slices, count, prim_shift, _rects, prepare in multi_meta):
    raise SystemExit("multi-draw descriptor metadata was not exact\n" + multi_log)
print("run geometry gate: %d member VS launches -> %d, pixels identical; "
      "one/multi descriptor metadata exact" % (batches, groups))

# The clip+bin structural slice preflights the whole run before any grouped VS.
# Until its count/scan/fill/scatter continuation exists it must issue no
# speculative grouped geometry and must replay the exact old member path.
clip_hash, clip_log = arm(True, clip_bin=True)
if clip_hash != base:
    raise SystemExit("run clip+bin preflight changed pixels: %s != %s\n%s" %
                     (base, clip_hash, clip_log))
cm = re.search(r"run-clip-bin-preflight: segments=(\d+) prim-slots=(\d+) "
               r"continuation=exact-fallback", clip_log)
if not cm or int(cm.group(1)) < 2 or int(cm.group(2)) == 0:
    raise SystemExit("whole-run clip+bin descriptor preflight was not reached\n" +
                     clip_log)
if "run-geometry-launch:" in clip_log:
    raise SystemExit("clip+bin slice issued additive grouped geometry\n" + clip_log)

# Dependency is fail-closed: asking for clip+bin alone must not collect a run.
dep_hash, dep_log = arm(False, clip_bin=True)
if dep_hash != base or "run-clip-bin-preflight:" in dep_log:
    raise SystemExit("RUN_CLIP_BIN did not require RUN_GEOMETRY\n" + dep_log)
print("run clip+bin preflight gate: %s segments, stable slots=%s; exact fallback" %
      cm.groups())
