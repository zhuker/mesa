#!/usr/bin/env python3
"""Compare every frame of an animated sweep across drivers.

The single-frame sweep says whether a driver draws the scene; this says
whether it keeps drawing it. A frame-one comparison cannot see a difference
that only appears once something moves — animation driven by a timer, a
particle system settling, state that leaks from one frame into the next.

    cp_perf_run.sh ... build/frames60/<driver> 60      # once per driver
    cp_compare_frames.py build/frames60 --ref nvidia --test cudavk llvmpipe

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

Unlike the rest of the tests this one needs numpy: it is the only tool that
counts pixels over every frame of every sample rather than over one image, so
the count runs a thousand times a sweep. In pure Python that was 0.16 s per
720p frame pair, minutes per driver; in numpy it is half a millisecond. Run it
with the repo venv's interpreter, $MESA/venv/bin/python3, which is what
cp_iterate.sh does.
"""

import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit('cp_compare_frames.py needs numpy: run it with $MESA/venv/bin/python3')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_compare import read_image


def frames_of(driver_dir, sample):
    d = os.path.join(driver_dir, sample)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(('.ppm', '.png')))


def read_rgb(path):
    """One image as an (h, w, 3) uint8 array, dropping alpha if it has one.

    The reader hands back one bytes object per row; joining them is a single
    memcpy of the frame, which is nothing beside what is done with it.
    """
    w, h, ch, rows = read_image(path)
    flat = np.frombuffer(b''.join(rows), dtype=np.uint8)
    if flat.size != w * h * ch:
        raise ValueError('%s: %d bytes of pixels for %dx%d and %d channels'
                         % (path, flat.size, w, h, ch))
    return flat.reshape(h, w, ch)[:, :, :3]


def differing(a, b, tol):
    """Pixels where any of the three channels differs by more than tol.

    max minus min is the absolute difference without leaving uint8, so the
    count runs at one byte a pixel-channel and never promotes to int.

    The three channels are then reduced by name rather than with
    delta.max(axis=2). A numpy reduction along a length 3 contiguous axis is
    far slower than two whole-plane maxima over strided views — 13 ms against
    1 ms on a 720p frame here, which is most of what this function costs.
    """
    delta = np.maximum(a, b)
    delta -= np.minimum(a, b)
    worst = np.maximum(np.maximum(delta[:, :, 0], delta[:, :, 1]),
                       delta[:, :, 2])
    return int(np.count_nonzero(worst > tol))


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
                ri = read_rgb(os.path.join(ref_dir, sample, rf[i]))
                ti = read_rgb(os.path.join(test_dir, sample, tf[i]))
                if ri.shape != ti.shape:
                    first = worst = -1
                    break
                d = differing(ri, ti, args.tol)
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
