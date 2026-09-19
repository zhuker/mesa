#!/usr/bin/env python3
"""B200 fixture presence gate; requires an actual lost-block retry opportunity.

Run under the GPU lock with the candidate ICD and cpvk_batchbig. This gate
fails (does not skip) when the fixture does not exercise the retry. It does
not replace the unchanged vfetch/run-geometry/native/capture gates.
"""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

exe = sys.argv[1]
output = Path("/tmp/lat/native_batchtex.ppm")

def arm(value):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CUDAVK_", "CPVK_"))}
    env.update(CUDAVK_SHADER_STATS="1", CUDAVK_BATCH_MAX="1",
               CUDAVK_RUN_GEOMETRY="1", CUDAVK_DEBUG_EPISODE="1")
    if value is not None:
        env["CUDAVK_VFETCH_REGCAP_RETRY"] = value
    output.unlink(missing_ok=True)
    proc = subprocess.run([exe], env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    if proc.returncode or not output.exists():
        raise SystemExit("fixture failed\n" + proc.stdout)
    return hashlib.sha256(output.read_bytes()).hexdigest(), proc.stdout

base, default_log = arm(None)
off, off_log = arm("0")
on, on_log = arm("1")
if not base == off == on:
    raise SystemExit("retry changed fixture pixels")
for log in (default_log, off_log):
    if "vsfetch regcap retry" in log or "run-geometry-launch:" in log:
        raise SystemExit("default/off unexpectedly retried or grouped\n" + log)
    if not re.search(r"vsfetch verdict lost-a-block", log):
        raise SystemExit("fixture does not have the required occupancy refusal\n" + log)
if not re.search(r"vsfetch regcap retry cap \d+ regs \d+ spill \d+ "
                 r"blocks/sm \d+ admitted", on_log):
    raise SystemExit("retry did not admit\n" + on_log)
if not re.search(r"vsfetch shader .* fused +[1-9]\d* .* admitted", on_log):
    raise SystemExit("admitted fused execution was not launched\n" + on_log)
if not re.search(r"run-geometry-launch: members=[2-9]\d*", on_log):
    raise SystemExit("many-to-one geometry launch absent\n" + on_log)
# Initial classic, fused, and all FS builds must be unchanged. Only the
# separately labelled retry may differ; no global MAX_REGISTERS probe here.
def initial_resources(log):
    return sorted(line for line in log.splitlines()
                  if line.startswith("cudavk: shader "))
if not initial_resources(default_log) == initial_resources(off_log) == initial_resources(on_log):
    raise SystemExit("initial shader resource diagnostics changed")
print("retry live presence: default/off refuse, on admits and launches; "
      "fixture pixels and initial shader resources identical")
print("Not baseline-suite acceptance or a performance result.")
