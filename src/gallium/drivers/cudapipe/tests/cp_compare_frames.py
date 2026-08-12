#!/usr/bin/env python3
"""Compare every frame of an animated sweep across drivers.

The single-frame sweep says whether a driver draws the scene; this says
whether it keeps drawing it. A frame-one comparison cannot see a difference
that only appears once something moves — animation driven by a timer, a
particle system settling, state that leaks from one frame into the next.

    cp_perf_run.sh ... build/frames60/<driver> 60      # once per driver
    cp_compare_frames.py build/frames60 --ref nvidia --test cudapipe llvmpipe

Each driver's directory holds one directory per sample of numbered frames,
which is what cp_perf_run.sh writes. Frames are compared pairwise by index.

The number that matters is per frame, against the budget frame 0 already
spends: a driver that differs by 1533 pixels on frame 0 and 1600 on frame 40
has not regressed, and one that goes to 40000 has. --tolerate sets how much
worse than frame 0 a frame may be before it is called a regression, as a
multiple plus a constant floor for samples whose frame 0 is near zero.

Prints a line per sample with the worst frame and where it was, then a
verdict. Exit status is 1 if any sample regressed, so it can gate a
benchmark run.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_compare import read_image


def frames_of(driver_dir, sample):
    d = os.path.join(driver_dir, sample)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(('.ppm', '.png')))


def differing(a, b, w, h, ch_a, ch_b, tol):
    n = 0
    for y in range(h):
        ra, rb = a[y], b[y]
        for x in range(w):
            ia, ib = x * ch_a, x * ch_b
            if max(abs(ra[ia] - rb[ib]),
                   abs(ra[ia + 1] - rb[ib + 1]),
                   abs(ra[ia + 2] - rb[ib + 2])) > tol:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='directory holding one subdirectory per driver')
    ap.add_argument('--ref', default='nvidia', help='reference driver (default nvidia)')
    ap.add_argument('--test', nargs='+', default=['cuda'], help='drivers under test')
    ap.add_argument('-t', '--tol', type=int, default=8,
                    help='per-channel tolerance, 0-255 (default 8)')
    ap.add_argument('--tolerate', type=float, default=1.5,
                    help='a frame may differ this many times more than frame 0 '
                         '(default 1.5)')
    ap.add_argument('--floor', type=int, default=2000,
                    help='and this much more besides, for samples whose frame 0 '
                         'is near zero (default 2000)')
    ap.add_argument('--stride', type=int, default=1,
                    help='compare every Nth frame (default 1, all of them)')
    args = ap.parse_args()

    ref_dir = os.path.join(args.root, args.ref)
    if not os.path.isdir(ref_dir):
        print(f'no reference directory {ref_dir}')
        return 2

    samples = sorted(s for s in os.listdir(ref_dir)
                     if os.path.isdir(os.path.join(ref_dir, s))
                     and not s.startswith('_'))

    regressed = False
    for test in args.test:
        test_dir = os.path.join(args.root, test)
        print(f'\n=== {test} vs {args.ref}, tolerance {args.tol}/255 ===')
        print('%-24s %10s %10s %8s  %s' %
              ('sample', 'frame 0', 'worst', 'at', 'verdict'))

        for sample in samples:
            rf = frames_of(ref_dir, sample)
            tf = frames_of(test_dir, sample)
            if not rf or not tf:
                print('%-24s %10s %10s %8s  %s' % (sample, '-', '-', '-', 'missing'))
                continue

            n = min(len(rf), len(tf))
            first = worst = None
            worst_at = 0
            for i in range(0, n, args.stride):
                rw, rh, rch, ri = read_image(os.path.join(ref_dir, sample, rf[i]))
                tw, th, tch, ti = read_image(os.path.join(test_dir, sample, tf[i]))
                if (rw, rh) != (tw, th):
                    first = worst = -1
                    break
                d = differing(ri, ti, rw, rh, rch, tch, args.tol)
                if first is None:
                    first = d
                if worst is None or d > worst:
                    worst, worst_at = d, i

            if first is None or first < 0:
                print('%-24s %10s %10s %8s  %s' % (sample, '-', '-', '-', 'size mismatch'))
                regressed = True
                continue

            budget = first * args.tolerate + args.floor
            ok = worst <= budget
            regressed |= not ok
            print('%-24s %10d %10d %8d  %s' %
                  (sample, first, worst, worst_at,
                   'ok' if ok else 'REGRESSED (budget %d)' % budget))

    print('\n%s' % ('some samples regressed over the animation'
                    if regressed else
                    'every frame is within its frame 0 budget on every driver'))
    return 1 if regressed else 0


if __name__ == '__main__':
    sys.exit(main())
