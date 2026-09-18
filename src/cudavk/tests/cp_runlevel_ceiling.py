#!/usr/bin/env python3
"""Bound a fixed-function run merge from an Nsight Systems SQLite export.

This deliberately uses interval unions. Summed kernel duration double-counts
work that the existing side streams overlap and is not a frame-time ceiling.
"""
import argparse
import sqlite3
import sys

TARGET = {"cp_clip_rast_fused", "cp_rasterize_stage2", "cp_rasterize_stage3"}
PAIRS = {
    ("cp_clip_rast_fused", "cp_rasterize_stage2"),
    ("cp_rasterize_stage2", "cp_rasterize_stage3"),
}

def union_ns(intervals):
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        if start > end:
            total += stop - start
        else:
            total += stop - end
        end = stop
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", help="Nsight Systems SQLite export")
    ap.add_argument("--frames", type=int, required=True,
                    help="complete frames represented by the trace")
    ap.add_argument("--require-union-ms", type=float, default=0.0,
                    help="exit 1 unless impossible deletion reaches this gate")
    args = ap.parse_args()
    if args.frames <= 0:
        ap.error("--frames must be positive")

    con = sqlite3.connect(args.sqlite)
    rows = list(con.execute("""
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        ORDER BY k.streamId, k.start
    """))
    if not rows:
        raise SystemExit("no CUDA kernel activities")

    all_intervals = [(r[0], r[1]) for r in rows]
    kept_intervals = [(r[0], r[1]) for r in rows if r[3] not in TARGET]
    target_sum = sum(r[1] - r[0] for r in rows if r[3] in TARGET)
    union_delta = union_ns(all_intervals) - union_ns(kept_intervals)

    adjacent_gap = 0
    adjacent_pairs = 0
    previous = {}
    for start, stop, stream, name in rows:
        prev = previous.get(stream)
        if prev and (prev[2], name) in PAIRS:
            adjacent_gap += max(0, start - prev[1])
            adjacent_pairs += 1
        previous[stream] = (start, stop, name)

    scale = args.frames * 1e6
    target_ms = target_sum / scale
    union_ms = union_delta / scale
    gap_ms = adjacent_gap / scale
    print(f"frames={args.frames} target-summed={target_ms:.6f} ms/frame")
    print(f"impossible-target-deletion-union={union_ms:.6f} ms/frame")
    print(f"adjacent-interstage-gaps={gap_ms:.6f} ms/frame "
          f"across {adjacent_pairs} pairs")
    if args.require_union_ms and union_ms < args.require_union_ms:
        print(f"REJECT: union ceiling {union_ms:.6f} < "
              f"required {args.require_union_ms:.6f} ms/frame", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
