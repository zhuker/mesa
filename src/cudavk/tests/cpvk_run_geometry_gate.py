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

def arm(enabled, split=True, clip_bin=False, extra=None):
    env = os.environ.copy()
    env.pop("CUDAVK_RUN_GEOMETRY_FAIL_ALLOC", None)
    env.pop("CUDAVK_RUN_GEOMETRY_FAIL_LAUNCH", None)
    if split:
        env["CUDAVK_BATCH_MAX"] = "1"
    else:
        env.pop("CUDAVK_BATCH_MAX", None)
    env["CUDAVK_RUN_GEOMETRY"] = "1" if enabled else "0"
    env["CUDAVK_RUN_CLIP_BIN"] = "1" if clip_bin else "0"
    if enabled or clip_bin:
        env["CUDAVK_DEBUG_EPISODE"] = "1"
    if extra:
        env.update(extra)
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

# Allocation OOM happens before any grouped launch and is a legal exact-path
# refusal. A CUDA/context error at the same allocation site, or at the launch,
# must instead reach the ICD's existing VK_ERROR_DEVICE_LOST latch.
oom_hash, oom_log = arm(True, extra={"CUDAVK_RUN_GEOMETRY_FAIL_ALLOC": "1"})
if oom_hash != base or "run-geometry-launch:" in oom_log:
    raise SystemExit("run geometry OOM did not soft-refuse before launch\n" + oom_log)

def fatal_fault(extra, label):
    env = os.environ.copy()
    env.pop("CUDAVK_RUN_GEOMETRY_FAIL_ALLOC", None)
    env.pop("CUDAVK_RUN_GEOMETRY_FAIL_LAUNCH", None)
    env["CUDAVK_BATCH_MAX"] = "1"
    env["CUDAVK_RUN_GEOMETRY"] = "1"
    env.update(extra)
    proc = subprocess.run([exe], env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    if proc.returncode == 0 or "failed: -4" not in proc.stdout:
        raise SystemExit("%s did not latch VK_ERROR_DEVICE_LOST\n%s" %
                         (label, proc.stdout))

fatal_fault({"CUDAVK_RUN_GEOMETRY_FAIL_ALLOC": "2"},
            "run geometry allocation CUDA fault")
fatal_fault({"CUDAVK_RUN_GEOMETRY_FAIL_LAUNCH": "1"},
            "run geometry group launch CUDA fault")
print("run geometry fault gate: OOM replayed; CUDA allocation/launch faults lost device")

# The end-to-end arm uses inclusive depth, which is the current tile winner's
# exact tie rule. It must commit one run-wide continuation and preserve pixels.
clip_base, _ = arm(False, extra={"CPVK_TEST_DEPTH_LEQUAL": "1"})
clip_hash, clip_log = arm(True, clip_bin=True,
                          extra={"CPVK_TEST_DEPTH_LEQUAL": "1"})
if clip_hash != clip_base:
    raise SystemExit("run clip+bin changed pixels: %s != %s\n%s" %
                     (clip_base, clip_hash, clip_log))
cm = re.search(r"run-clip-bin-preflight: segments=(\d+) prim-slots=(\d+)",
               clip_log)
if not cm or int(cm.group(1)) < 2 or int(cm.group(2)) == 0:
    raise SystemExit("whole-run clip+bin transaction was not reached\n" + clip_log)
if "run-clip-bin-commit:" not in clip_log:
    raise SystemExit("run clip+bin did not report a committed continuation\n" + clip_log)
pm = re.search(r"run-clip-bin-prepare: batches=(\d+) streams=(\d+) join=(\d+)",
               clip_log)
if not pm or int(pm.group(1)) != int(cm.group(1)) or int(pm.group(2)) < 2 or \
        int(pm.group(3)) != 1:
    raise SystemExit("exact VS preparation did not fan out and join once\n" + clip_log)

# The revert keeps the same exact per-batch output and descriptors, but places
# every prepare launch back on main.
serial_hash, serial_log = arm(
    True, clip_bin=True,
    extra={"CPVK_TEST_DEPTH_LEQUAL": "1", "CUDAVK_NO_RUN_PREP_STREAMS": "1"})
if serial_hash != clip_base or \
        "run-clip-bin-prepare: batches=%s streams=1 join=0" % cm.group(1) not in serial_log:
    raise SystemExit("serial prepare revert changed pixels or did not engage\n" +
                     serial_log)

# A tiny dense-list capacity makes the uint64 count check set the shared
# predicate. The classic visibility arm must consume the same clipped geometry
# and produce the exact image without a host decision.
overflow_hash, overflow_log = arm(
    True, clip_bin=True,
    extra={"CPVK_TEST_DEPTH_LEQUAL": "1",
           "CUDAVK_RUN_CLIP_BIN_CAPACITY": "1"})
if overflow_hash != clip_base or "run-clip-bin-commit:" not in overflow_log:
    raise SystemExit("device-predicated classic fallback changed pixels\n" +
                     overflow_log)

# Strict LESS is deliberately refused before geometry because equal-depth ties
# need the sequential first-winner rule. Refusal must be exact and additive-free.
strict_hash, strict_log = arm(True, clip_bin=True)
if strict_hash != base or "run-clip-bin-commit:" in strict_log or         "run-geometry-launch:" in strict_log:
    raise SystemExit("strict-depth run was not an exact pre-commit refusal\n" +
                     strict_log)

# Dependency is fail-closed: asking for clip+bin alone must not collect a run.
dep_hash, dep_log = arm(False, clip_bin=True)
if dep_hash != base or "run-clip-bin-preflight:" in dep_log:
    raise SystemExit("RUN_CLIP_BIN did not require RUN_GEOMETRY\n" + dep_log)
print("run clip+bin transaction gate: %s segments, stable slots=%s; pixels exact" %
      cm.groups())
