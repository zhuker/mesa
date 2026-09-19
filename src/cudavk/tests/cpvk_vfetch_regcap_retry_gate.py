#!/usr/bin/env python3
"""B200 fixture presence gate; requires an actual lost-block retry opportunity.

Run under the GPU lock with the candidate ICD and cpvk_batchbig. This gate
fails (does not skip) when the fixture does not exercise the retry. It does
not replace the unchanged vfetch/run-geometry/native/capture gates.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess


def initial_resources(log):
    return sorted(line for line in log.splitlines()
                  if line.startswith("cudavk: shader "))


def validate_logs(default_log, off_log, on_log):
    """Validate fixture telemetry, independently of execution and pixel checks."""
    for log in (default_log, off_log):
        if "vsfetch regcap retry" in log or "run-geometry-launch:" in log:
            raise ValueError("default/off unexpectedly retried or grouped")
        if not re.search(r"vsfetch verdict lost-a-block\b", log):
            raise ValueError("fixture lacks the original occupancy refusal")
    if not re.search(r"^cudavk: vsfetch regcap retry cap 32 regs 32 spill 0 "
                     r"blocks/sm 8 admitted$", on_log, re.M):
        raise ValueError("cap32 zero-spill eight-block retry did not admit")
    if not re.search(r"^cudavk: vsfetch verdict admitted\s+"
                     r"classic regs\s+32 spill\s+0 blocks/sm 8 local 0 \| "
                     r"fused regs\s+32 spill\s+0 blocks/sm 8 local 0$",
                     on_log, re.M):
        raise ValueError("installed fused resources not admitted")
    # cp_run_geometry_fused requires VS_FETCH.kernel and rejects DEBUG_VFETCH.
    # cp_run_geometry_launch logs only AFTER cp_launch(group->kernel) succeeds.
    # The shader census is printed on renderer destruction; cpvk_batchbig does
    # not destroy its device. Requiring that separate census rejects real work.
    launches = re.findall(r"^run-geometry-launch: members=(\d+) draws=(\d+) "
                          r"verts=(\d+) kernel=(0x[0-9a-fA-F]+)$", on_log, re.M)
    if len(launches) != 1 or launches[0][:3] != ("9", "9", "162000") or not int(launches[0][3], 16):
        raise ValueError("actual nine-to-one fused geometry launch absent")
    if not initial_resources(default_log) or not (
            initial_resources(default_log) == initial_resources(off_log) ==
            initial_resources(on_log)):
        raise ValueError("initial shader resource diagnostics changed or missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    output = Path("/tmp/lat/native_batchtex.ppm")
    logs, digests = [], []
    for value in (None, "0", "1"):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("CUDAVK_", "CPVK_"))}
        env.update(CUDAVK_SHADER_STATS="1", CUDAVK_BATCH_MAX="1",
                   CUDAVK_RUN_GEOMETRY="1", CUDAVK_DEBUG_EPISODE="1")
        if value is not None:
            env["CUDAVK_VFETCH_REGCAP_RETRY"] = value
        output.unlink(missing_ok=True)
        proc = subprocess.run([args.exe], env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=120)
        if args.log_dir:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            (args.log_dir / ("presence-" + (value or "default") + ".log")).write_text(proc.stdout)
        if proc.returncode or not output.exists():
            raise SystemExit("fixture failed\n" + proc.stdout)
        digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
        logs.append(proc.stdout)
    if len(set(digests)) != 1:
        raise SystemExit("retry changed fixture pixels")
    try:
        validate_logs(*logs)
    except ValueError as exc:
        raise SystemExit(str(exc) + "\n" + "\n".join(logs)) from exc
    print("retry live presence: default/off refuse, cap32 on admits and launches "
          "9 draws / 162000 vertices in one fused launch")
    print("fixture SHA256 default/off/on: " + " ".join(digests))
    print("initial shader resources identical; not baseline-suite acceptance "
          "or a performance result.")


if __name__ == "__main__":
    main()
