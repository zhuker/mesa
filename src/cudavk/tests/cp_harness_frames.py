#!/usr/bin/env python3
"""Encode a compiled-harness frame dump and build its timeline page.

The harness (favorite2/favorite3 tocpp builds) writes frame_NNNNNN.bin plus a
.meta of "width height bytes vkformat" per frame. This turns those into PNGs
and one page, the way cp_gfxr_frames.py does for a gfxr replay -- same idea,
different dump format, so the two captures that actually matter get the same
artefact the samples already had.

Needs the mesa venv (numpy, PIL):  $MESA/venv/bin/python3

  cp_harness_frames.py png RAWDIR OUTDIR

The page is then built by cp_gfxr_frames.py's own timeline command, so both
kinds of capture get the identical page.
"""
import argparse, os, sys, json

# VK_FORMAT_B8G8R8A8_UNORM=44, _SRGB=50; R8G8B8A8_UNORM=37, _SRGB=43.
BGRA = {44, 50}
RGBA = {37, 43}


def read_meta(path):
    w, h, nbytes, fmt = (int(x) for x in open(path).read().split()[:4])
    return w, h, nbytes, fmt


def cmd_png(args):
    import numpy as np
    from PIL import Image
    os.makedirs(args.outdir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.rawdir) if f.endswith(".bin"))
    if not names:
        sys.exit("no frame_*.bin in %s" % args.rawdir)
    written = 0
    for name in names:
        stem = os.path.splitext(name)[0]
        meta = os.path.join(args.rawdir, stem + ".meta")
        if not os.path.exists(meta):
            continue
        w, h, nbytes, fmt = read_meta(meta)
        raw = np.fromfile(os.path.join(args.rawdir, name), dtype=np.uint8)
        if raw.size != w * h * 4:
            print("skip %s: %d bytes, expected %d" % (name, raw.size, w * h * 4),
                  file=sys.stderr)
            continue
        a = raw.reshape(h, w, 4)
        if fmt in BGRA:
            rgb = a[:, :, [2, 1, 0]]
        elif fmt in RGBA:
            rgb = a[:, :, [0, 1, 2]]
        else:
            sys.exit("unhandled VkFormat %d in %s -- add it to the table" % (fmt, meta))
        img = Image.fromarray(rgb)
        img.save(os.path.join(args.outdir, stem + ".png"), compress_level=1)
        written += 1
    print("encoded %d frames" % written)


def frame_times(submits):
    """Paired-submit frame times in ms: a frame is two vkQueueSubmit events."""
    ts = [int(l.split()[0]) for l in open(submits)
          if len(l.split()) == 2 and l.split()[0].isdigit()]
    return [(ts[i + 2] - ts[i]) / 1e6 for i in range(0, len(ts) - 2, 2)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("png"); a.add_argument("rawdir"); a.add_argument("outdir")
    a.set_defaults(fn=cmd_png)
    args = p.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
