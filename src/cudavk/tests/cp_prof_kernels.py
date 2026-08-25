#!/usr/bin/env python3
"""
Break a cp_profile.sh trace down by kernel *and grid size*, and separate the
fixed per-draw cost from the cost that scales with the draw.

    cp_prof_kernels.py build/prof/LABEL/SAMPLE.sqlite [--frames N]

Why this exists rather than `nsys stats`: every shader cudavk compiles is a
CUDA kernel named `main`, so the vertex and fragment stages land in one row of
`cuda_gpu_kern_sum` and the single largest entry in every profile is
uninterpretable. Grid size separates them — the vertex shader is launched over
the vertex count and the fragment shader over a fixed worst case — and it is
also what makes "this kernel costs the same no matter what the draw covers"
visible, which is the shape of every defect the performance pass has found so
far.

Three sections:

  by kernel      total, share, and per-launch median
  fixed cost     kernels whose duration barely varies across launches, which
                 means they are sized to the framebuffer rather than the draw
  by grid        the distinct launch geometries of each kernel

`--frames N` divides the totals so they read as milliseconds per frame; pass
the same count cp_profile.sh was given. Warm-up launches are included, so a
per-frame figure from a ten-frame trace is approximate by about that much.
"""
import argparse
import sqlite3
import statistics
import sys

# A kernel whose per-launch duration varies by less than this, relative to its
# own median, is doing the same amount of work every time it runs. For a
# per-draw kernel that is the definition of work sized to the framebuffer.
FIXED_COST_CV = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--frames", type=int, default=0,
                    help="divide totals by this to get ms/frame")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    rows = db.execute("""
        SELECT shortName, gridX * gridY * gridZ, blockX * blockY * blockZ,
               end - start
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """).fetchall()
    if not rows:
        sys.exit("no kernel activity in %s" % args.sqlite)

    # Is the frame kernel-bound or is the GPU idle waiting on the host? This is
    # the measurement gate's question, and the answer decides whether plan
    # phase 1a (streams, deleting the host syncs) is on the critical path or
    # not. Union of the busy intervals, because kernels and memory operations
    # on separate streams can overlap and summing their durations would report
    # more than 100%.
    spans = sorted(db.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET
        UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
    """).fetchall())
    busy, cur_s, cur_e = 0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    wall = spans[-1][1] - spans[0][0]
    print("=== measurement gate ===")
    print("  GPU busy %.1f ms of %.1f ms traced (%.1f%%) -- %s\n"
          % (busy / 1e6, wall / 1e6, 100.0 * busy / wall,
             "kernel-bound" if busy / wall > 0.9 else
             "the GPU is idle for part of the frame; phase 1a may now pay"))

    # shortName is a string-table id in some nsys versions and the string
    # itself in others; resolve it either way.
    names = dict(db.execute("SELECT id, value FROM StringIds").fetchall()) \
        if db.execute("SELECT name FROM sqlite_master WHERE name='StringIds'"
                      ).fetchone() else {}

    def nm(v):
        return names.get(v, v) if isinstance(v, int) else v

    per_kernel = {}
    per_grid = {}
    for short, grid, block, dur in rows:
        name = nm(short)
        per_kernel.setdefault(name, []).append(dur)
        per_grid.setdefault((name, grid, block), []).append(dur)

    total = sum(sum(v) for v in per_kernel.values())
    scale = 1e-6 / args.frames if args.frames else 1e-6
    unit = "ms/frame" if args.frames else "ms total"

    print("=== by kernel (%s), %s ===" % (unit, args.sqlite))
    print("%-26s%>12s%8s%12s%12s%12s".replace(">", "")
          % ("kernel", unit, "share", "launches", "med us", "cv"))
    ranked = sorted(per_kernel.items(), key=lambda kv: -sum(kv[1]))
    fixed = []
    for name, durs in ranked[:args.top]:
        med = statistics.median(durs)
        cv = (statistics.pstdev(durs) / med) if med else 0.0
        print("%-26s%12.2f%7.1f%%%12d%12.2f%12.2f"
              % (name, sum(durs) * scale, 100 * sum(durs) / total,
                 len(durs), med / 1e3, cv))
        if cv < FIXED_COST_CV and len(durs) > 50:
            fixed.append((name, sum(durs), med, len(durs)))

    print("\n=== fixed per-launch cost (cv < %.2f): work sized to the "
          "framebuffer, not the draw ===" % FIXED_COST_CV)
    if not fixed:
        print("  (none)")
    for name, tot, med, n in sorted(fixed, key=lambda f: -f[1]):
        print("  %-24s%12.2f %s  %6d launches x %.2f us"
              % (name, tot * scale, unit, n, med / 1e3))

    print("\n=== by grid geometry ===")
    print("%-26s%10s%8s%10s%12s%12s"
          % ("kernel", "blocks", "block", "launches", unit, "med us"))
    for (name, grid, block), durs in sorted(per_grid.items(),
                                            key=lambda kv: -sum(kv[1]))[:args.top]:
        print("%-26s%10d%8d%10d%12.2f%12.2f"
              % (name, grid, block, len(durs), sum(durs) * scale,
                 statistics.median(durs) / 1e3))


if __name__ == "__main__":
    main()
