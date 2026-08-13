#!/usr/bin/env python3
"""
What the host spent its time issuing, by pipeline stage.

    cp_prof_nvtx.py SAMPLE.sqlite [--frames N]

Reads the NVTX ranges the driver emits under CUDAPIPE_NVTX, which
`cp_profile.sh` turns on by default. Companion to `cp_prof_kernels.py`: that one
splits GPU work by kernel and grid size, this one splits *host* work by the
stage that issued it.

**These are issue times, not device times.** Every launch is asynchronous, so a
range closes once the calls are queued rather than when the GPU finishes them.
Read a stage's total as "what the host paid to submit this stage", and read it
against `cp_prof_kernels.py` for what the device then did. The two disagreeing
is the finding, not an error: a stage costing far more to issue than to run is
the shape of a launch-bound frame.

Tracing inflates per-launch host cost, so compare shares between stages rather
than absolute times, and never against an untraced frame.
"""
import argparse
import os
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--frames", type=int, default=0,
                    help="divide totals by this many frames")
    args = ap.parse_args()

    if not os.path.exists(args.sqlite):
        sys.exit(f"no such file: {args.sqlite}")

    con = sqlite3.connect(args.sqlite)
    have = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS'"
    ).fetchone()[0]
    if not have:
        sys.exit("no NVTX_EVENTS table: trace with --trace=cuda,nvtx and run the "
                 "sample with CUDAPIPE_NVTX=1 (cp_profile.sh does both)")

    # eventType distinguishes ranges from marks; a mark has no end.
    rows = con.execute("""
        SELECT text, COUNT(*), SUM("end" - start), AVG("end" - start)
        FROM NVTX_EVENTS
        WHERE text IS NOT NULL AND "end" IS NOT NULL
        GROUP BY text ORDER BY SUM("end" - start) DESC
    """).fetchall()
    if not rows:
        sys.exit("NVTX_EVENTS holds no closed ranges")

    marks = con.execute("""
        SELECT text, COUNT(*) FROM NVTX_EVENTS
        WHERE text IS NOT NULL AND "end" IS NULL GROUP BY text
    """).fetchall()

    # A draw range encloses every stage range, so it is the denominator rather
    # than a peer. Its name carries the triangle count, so group those together.
    draw_total = sum(t for n, c, t, a in rows if n.startswith("draw "))
    draw_count = sum(c for n, c, t, a in rows if n.startswith("draw "))

    per = args.frames if args.frames > 0 else 1
    unit = "ms/frame" if args.frames > 0 else "ms total"

    print(f"{'stage':<20}{'count':>10}{unit:>12}{'avg us':>10}{'% of draw':>11}")
    print("-" * 63)
    if draw_count:
        print(f"{'draw (all)':<20}{draw_count:>10}"
              f"{draw_total/1e6/per:>12.2f}{draw_total/draw_count/1e3:>10.2f}"
              f"{100.0:>10.1f}%")
    for name, count, total, avg in rows:
        if name.startswith("draw "):
            continue
        share = 100.0 * total / draw_total if draw_total else 0.0
        print(f"{name[:19]:<20}{count:>10}{total/1e6/per:>12.2f}"
              f"{avg/1e3:>10.2f}{share:>10.1f}%")

    if draw_count and args.frames > 0:
        print("-" * 63)
        print(f"{draw_count/args.frames:.0f} draws per frame")
    for name, count in marks:
        print(f"mark {name}: {count}")

    print()
    print("Issue time, not device time — every launch is asynchronous. Read")
    print("against cp_prof_kernels.py; the two disagreeing is what launch-bound")
    print("looks like from the host side.")


if __name__ == "__main__":
    main()
