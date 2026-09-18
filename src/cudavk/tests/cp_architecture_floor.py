#!/usr/bin/env python3
"""Bound separate host and fixed-kernel attacks from an nsys SQLite trace.

This is a deletion bound, not a prediction: timestamps of retained work stay
fixed. It answers whether independently removing all non-shader GPU intervals,
plus an externally measured host-scheduling gain, can reach a target without a
joint rescheduling mechanism.
"""
import argparse
import sqlite3


def union_ns(intervals):
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end) if start <= end else stop - start
        end = stop
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--shader-name", default="main",
                    help="kernel short name retained as shader work")
    ap.add_argument("--host-gain-ms", type=float, default=0.0,
                    help="independently measured completion-aware host gain")
    ap.add_argument("--target-ms", type=float, default=2.0)
    args = ap.parse_args()
    if args.frames <= 0:
        ap.error("--frames must be positive")

    con = sqlite3.connect(args.sqlite)
    rows = list(con.execute("""
        SELECT k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        ORDER BY k.start
    """))
    if not rows:
        raise SystemExit("no CUDA kernel activities")

    all_intervals = [(start, end) for start, end, _ in rows]
    shader_intervals = [(start, end) for start, end, name in rows
                        if name == args.shader_name]
    if not shader_intervals:
        raise SystemExit(f"no kernel named {args.shader_name!r}")

    scale = args.frames * 1e6
    span = (max(end for _, end, _ in rows) -
            min(start for start, _, _ in rows)) / scale
    occupied = union_ns(all_intervals) / scale
    shader_union = union_ns(shader_intervals) / scale
    nonshader_credit = occupied - shader_union
    fixed_deleted = span - nonshader_credit
    combined = fixed_deleted - args.host_gain_ms

    print(f"frames={args.frames}")
    print(f"trace-span={span:.6f} ms/frame")
    print(f"gpu-occupied-union={occupied:.6f} ms/frame")
    print(f"shader-only-union={shader_union:.6f} ms/frame")
    print(f"impossible-nonshader-deletion-credit={nonshader_credit:.6f} ms/frame")
    print(f"fixed-kernels-deleted-residual={fixed_deleted:.6f} ms/frame")
    print(f"supplied-host-gain={args.host_gain_ms:.6f} ms/frame")
    print(f"simple-combined-residual={combined:.6f} ms/frame")
    if combined > args.target_ms:
        print(f"REJECT separate axes: {combined:.6f} > target "
              f"{args.target_ms:.6f} ms/frame")
        return 1
    print(f"PASS bound: {combined:.6f} <= target {args.target_ms:.6f} ms/frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
