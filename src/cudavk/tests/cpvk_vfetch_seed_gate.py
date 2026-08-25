#!/usr/bin/env python3
"""
The counter seeding follows the launch it rides on.

Iteration 26 S2 moved the clip counter and the three raster queue counters out
of their own device clears and into cp_vertex_fetch, which runs once per
executed batch. Iteration 27 removes that launch for an admitted shader, so
the seeding has to move into the fused vertex kernel -- and if it silently did
not, nothing would fail: the counters would go back to being cleared by the
launches that consume them, the frames would be right, and the iteration would
have quietly reverted S2 while reporting its own saving.

This is the negative control that makes that visible. CUDAVK_VFETCH_SKIP_SEED
makes an admitted fused draw skip the seeding while the host still skips the
clears it replaced, which is exactly the state a forgotten move would leave the
driver in. The canary test must pass with the fused path on and must FAIL with
the seeding taken out of it. A gate that cannot fail is not covering anything.
"""
import os
import subprocess
import sys


def run(exe, extra):
    # The fused path is what this gate is about, so it selects it rather than
    # inheriting it: a suite run in the reverted state would otherwise report
    # that a mechanism which was never built does not seed anything.
    env = dict(os.environ)
    env.pop("CUDAVK_FUSED_VFETCH", None)
    env.pop("CUDAVK_NO_FUSED_VFETCH", None)
    env.pop("CUDAVK_VFETCH_SKIP_SEED", None)
    env.update(extra)
    return subprocess.run([exe], env=env, capture_output=True, text=True,
                          timeout=300)


def main():
    if len(sys.argv) < 2:
        print("usage: cpvk_vfetch_seed_gate.py <canary test binary>")
        return 2
    exe = sys.argv[1]

    good = run(exe, {"CUDAVK_FUSED_VFETCH": "1"})
    print("fused, seeded:      exit %d" % good.returncode)
    starved = run(exe, {"CUDAVK_FUSED_VFETCH": "1",
                        "CUDAVK_VFETCH_SKIP_SEED": "1"})
    print("fused, not seeded:  exit %d" % starved.returncode)

    if good.returncode != 0:
        print("FAIL: the canary does not pass with the fused fetch seeding")
        print(good.stdout, good.stderr)
        return 1
    if starved.returncode == 0:
        print("FAIL: the canary passes without the seeding, so it is not "
              "covering the counters the seeding writes")
        return 1
    print("PASS: the seeding is load-bearing and the fused kernel does it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
