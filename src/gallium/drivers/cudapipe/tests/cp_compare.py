#!/usr/bin/env python3
"""Compare two PNGs rendered by different drivers.

Reports the fraction of pixels that differ beyond a tolerance, plus a coarse
ASCII map of where they differ, which is usually enough to tell a geometry bug
from a shading bug at a glance.
"""

import struct
import sys
import zlib


def read_png(path):
    raw = open(path, 'rb').read()
    pos, width, height, color, idat = 8, None, None, None, b''
    while pos < len(raw):
        (length,) = struct.unpack('>I', raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + length]
        if ctype == b'IHDR':
            width, height, _depth, color = struct.unpack('>IIBB', body[:10])
        elif ctype == b'IDAT':
            idat += body
        pos += 12 + length

    data = zlib.decompress(idat)
    channels = 4 if color == 6 else 3
    stride = width * channels
    rows, prev, p = [], bytearray(stride), 0
    for _ in range(height):
        filt = data[p]
        p += 1
        line = bytearray(data[p:p + stride])
        p += stride
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if filt == 1:
                line[i] = (line[i] + a) & 0xFF
            elif filt == 2:
                line[i] = (line[i] + b) & 0xFF
            elif filt == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif filt == 4:
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        rows.append(bytes(line))
        prev = line
    return width, height, channels, rows


def main():
    ref_path, test_path = sys.argv[1], sys.argv[2]
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    w, h, ch, ref = read_png(ref_path)
    w2, h2, ch2, test = read_png(test_path)
    if (w, h) != (w2, h2):
        print(f'size mismatch: {w}x{h} vs {w2}x{h2}')
        return 1

    diff_count = 0
    max_delta = 0
    covered_ref = covered_test = 0
    bg = tuple(ref[0][0:3])

    TILES = 32
    tile = [[0] * TILES for _ in range(TILES)]

    for y in range(h):
        for x in range(w):
            r = ref[y][x * ch:x * ch + 3]
            t = test[y][x * ch2:x * ch2 + 3]
            if tuple(r) != bg:
                covered_ref += 1
            if tuple(t) != bg:
                covered_test += 1
            d = max(abs(r[i] - t[i]) for i in range(3))
            max_delta = max(max_delta, d)
            if d > tol:
                diff_count += 1
                tile[y * TILES // h][x * TILES // w] += 1

    total = w * h
    print(f'{w}x{h}  tolerance {tol}')
    print(f'differing pixels : {diff_count}/{total} ({100.0 * diff_count / total:.2f}%)')
    print(f'max channel delta: {max_delta}')
    print(f'non-background   : reference {covered_ref}, test {covered_test}')

    if diff_count:
        per_tile = (h // TILES) * (w // TILES)
        print('\ndifference map (. none, : few, * many, # most):')
        for row in tile:
            line = ''
            for v in row:
                frac = v / per_tile if per_tile else 0
                line += '.' if frac < 0.01 else ':' if frac < 0.2 else '*' if frac < 0.6 else '#'
            print('  ' + line)

    return 0 if diff_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
