#!/usr/bin/env python3
"""Run the interop sample matrix.

Two positive cases and two negative controls. The negative controls are the
point: a synchronisation test that has never been observed to fail is not
evidence. They run expecting a non-zero exit, and a PASS there is reported as
a failure of the test, not as good news.

Use the repository interpreter for the consumer:
    /home/alexzhukov/mesa/venv/bin/python3
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER = os.path.join(HERE, "cpvk_interop_render")
CONSUMER = os.path.join(HERE, "cpvk_interop_torch.py")
PYTHON = "/home/alexzhukov/mesa/venv/bin/python3"

# The driver is chosen HERE, outside the application. Neither half of the
# sample knows or can find out which driver it got. That is the whole claim:
# one unmodified binary, one variable different, the same result. Point --icd
# at another ICD to test that claim; nothing in the sample changes.
DEFAULT_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"

EXIT_PASS, EXIT_VERIFY, EXIT_PROTO, EXIT_UNSUPPORTED = 0, 1, 2, 3


class Case:
    def __init__(self, name, args, expect, why, stall=0, consumer_extra=None):
        self.name = name
        self.args = args
        self.expect = expect          # "pass" or "fail"
        self.why = why
        self.stall = stall
        self.consumer_extra = consumer_extra or []


def cases(frames, slots, op, stall):
    """The matrix.

    Every case runs with the same GPU stall in the consumer. That is what makes
    the pair evidence rather than luck: with the stall, the consumer's work is
    still pending when its op function returns, so a missing synchronisation
    corrupts every frame instead of one frame in a thousand. A positive case
    that passes with the stall on has actually exercised the synchronisation.
    A positive case that passes only because the work is fast has proved
    nothing, and that is the state this matrix was in before the stall existed.
    """
    common = ["--frames", str(frames), "--slots", str(slots), "--op", op]
    short = ["--frames", str(min(frames, 200)), "--op", op]
    return [
        Case("host", common + ["--sync", "host"], "pass",
             "the oracle: obviously correct, slow", stall),
        Case("timeline", common + ["--sync", "timeline"], "pass",
             "the fast path, must agree with the oracle", stall),

        # Negative controls. These are the point of the matrix.
        Case("nosync-host",
             short + ["--slots", "1", "--sync", "host"], "fail",
             "consumer skips ev.synchronize(); producer must see poison",
             stall, consumer_extra=["--no-consumer-sync"]),
        Case("nosync-timeline",
             short + ["--slots", "1", "--sync", "timeline"], "fail",
             "consumer signals sem_consume before the work, not after",
             stall, consumer_extra=["--no-consumer-sync"]),
        Case("no-producer-wait",
             short + ["--slots", "1", "--sync", "host",
                      "--no-producer-wait"], "fail",
             "producer reuses a slot early; consumer must see a torn frame",
             stall),
    ]


def run_case(case, icd, timeout, png_dir, verbose):
    consumer_args = [PYTHON, CONSUMER, "--op", opt_of(case.args, "--op")]
    if case.stall:
        consumer_args += ["--stall-cycles", str(case.stall)]
    consumer_args += case.consumer_extra

    argv = [RENDER] + case.args + ["--exec", " ".join(shlex.quote(a)
                                                      for a in consumer_args)]
    # Failure PNGs are run output. Default them out of the source tree; the
    # producer writes to the cwd otherwise.
    argv += ["--png-dir", png_dir, "--png-every", "0"]

    env = dict(os.environ)
    # Pin the ICD rather than inherit one. An exported VK_DRIVER_FILES is
    # normal in this tree and would silently decide the experiment.
    env["VK_DRIVER_FILES"] = icd
    env.pop("VK_ICD_FILENAMES", None)
    env["CPVK_INTEROP_SYNC"] = opt_of(case.args, "--sync") or "timeline"

    t0 = time.time()
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        code = 124
    dt = time.time() - t0

    if verbose or code not in (EXIT_PASS,):
        for line in (out or "").splitlines():
            print("    | " + line)
    return code, dt, out or ""


def opt_of(args, name):
    if name in args:
        return args[args.index(name) + 1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--op", default="add1")
    ap.add_argument("--only", default=None,
                    help="run one case by name")
    ap.add_argument("--png-dir", default="/tmp/interop_png",
                    help="where failure images go; not the source tree")
    ap.add_argument("--stall-cycles", type=int, default=20_000_000,
                    help="GPU cycles the consumer spins on its stream before "
                         "the write-back, so a missing sync corrupts every "
                         "frame instead of none. Applies to every case.")
    ap.add_argument("--icd", default=DEFAULT_ICD,
                    help="ICD the sample runs against; the sample itself "
                         "cannot tell which one it got")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(RENDER):
        print("no producer binary: run make first (%s)" % RENDER)
        return 2
    if not os.path.exists(PYTHON):
        print("no repository interpreter at %s" % PYTHON)
        return 2
    if not os.path.exists(args.icd):
        print("no ICD at %s" % args.icd)
        return 2

    todo = [c for c in cases(args.frames, args.slots, args.op,
                             args.stall_cycles)
            if args.only in (None, c.name)]
    if not todo:
        print("no case named %r" % args.only)
        return 2

    print("interop matrix: %d frames, %d slots, op %s" %
          (args.frames, args.slots, args.op))
    print("icd           : %s" % args.icd)
    print("stall         : %d cycles in the consumer, every case" %
          args.stall_cycles)
    print("failure images: %s" % args.png_dir)
    os.makedirs(args.png_dir, exist_ok=True)
    print()

    results = []
    for c in todo:
        print("  %-18s %s" % (c.name, c.why))
        code, dt, out = run_case(c, args.icd, args.timeout, args.png_dir,
                                 args.verbose)

        if code == EXIT_UNSUPPORTED:
            verdict = "SKIP"
        elif c.expect == "pass":
            verdict = "PASS" if code == EXIT_PASS else "FAIL"
        else:
            # A negative control that passes means the test stopped testing.
            verdict = "PASS" if code in (EXIT_VERIFY,) else "FAIL"

        note = ""
        if c.expect == "fail" and code == EXIT_PASS:
            note = "  <-- negative control did not fail; the test is broken"
        if code == 124:
            note = "  <-- timed out"

        print("  %-18s %s  exit=%d  %.1fs%s" % ("", verdict, code, dt, note))
        print()
        results.append((c.name, verdict, code))

    bad = [r for r in results if r[1] == "FAIL"]
    print("summary: %d/%d ok" % (len(results) - len(bad), len(results)))
    for name, verdict, code in results:
        print("  %-18s %s (exit %d)" % (name, verdict, code))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
