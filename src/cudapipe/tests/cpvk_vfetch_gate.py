#!/usr/bin/env python3
"""
The fused vertex fetch renders what the separate fetch kernel renders.

`cpvk_vfetch` checks its own pixels against what Vulkan says each vertex
format supplies, so a mode that passes is right rather than merely stable.
This gate adds the other half: the two forms must agree with each other, in
every state the driver can be in --

  default                       whatever the tree's default is today
  CUDAPIPE_FUSED_VFETCH=1       the fused execution for every admitted shader
  ... with DECLINE_NTH=1 or 2   one shader forced onto the classic path while
                                its neighbour in the same command buffer stays
                                fused, which is the mixed case a per-shader
                                admission gate creates in every real frame
  CUDAPIPE_NO_FUSED_VFETCH=1    the second binary is not even built

-- and the frames must be byte-identical across all five, not merely all
passing. A fetch that is wrong in the same way in both forms would pass the
test and fail this comparison only if it also differed; that is why the test
carries the format expectations and this carries the agreement.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

MODES = ["formats", "sparse", "divisor", "decline"]
STATES = [
    ("default", {}),
    ("fused", {"CUDAPIPE_FUSED_VFETCH": "1"}),
    ("fused+decline1", {"CUDAPIPE_FUSED_VFETCH": "1",
                        "CUDAPIPE_VFETCH_DECLINE_NTH": "1"}),
    ("fused+decline2", {"CUDAPIPE_FUSED_VFETCH": "1",
                        "CUDAPIPE_VFETCH_DECLINE_NTH": "2"}),
    ("no-fused", {"CUDAPIPE_NO_FUSED_VFETCH": "1"}),
]


def main():
    if len(sys.argv) < 2:
        print("usage: cpvk_vfetch_gate.py <cpvk_vfetch binary>")
        return 2
    exe = sys.argv[1]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for mode in MODES:
            digests = {}
            for name, extra in STATES:
                # The state is this gate's own, not the suite run's: a run
                # under CUDAPIPE_NO_FUSED_VFETCH=1 would otherwise compare
                # five copies of the classic path and prove nothing.
                env = dict(os.environ)
                env.pop("CUDAPIPE_FUSED_VFETCH", None)
                env.pop("CUDAPIPE_NO_FUSED_VFETCH", None)
                env.pop("CUDAPIPE_VFETCH_DECLINE_NTH", None)
                env.pop("CUDAPIPE_VFETCH_SKIP_SEED", None)
                env.update(extra)
                out = os.path.join(tmp, "%s-%s.ppm" % (mode, name))
                run = subprocess.run([exe, mode, out], env=env,
                                     capture_output=True, text=True,
                                     timeout=300)
                if run.returncode != 0:
                    failures.append("%s/%s exited %d\n%s%s" %
                                    (mode, name, run.returncode, run.stdout,
                                     run.stderr))
                    continue
                try:
                    with open(out, "rb") as f:
                        digests[name] = hashlib.sha256(f.read()).hexdigest()
                except OSError as e:
                    failures.append("%s/%s wrote no frame: %s" % (mode, name, e))
            uniq = set(digests.values())
            print("%-8s %s" % (mode, " ".join(
                "%s=%s" % (n, d[:12]) for n, d in digests.items())))
            if len(uniq) > 1:
                failures.append("%s: the five states do not agree" % mode)
    for f in failures:
        print("FAIL: %s" % f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
