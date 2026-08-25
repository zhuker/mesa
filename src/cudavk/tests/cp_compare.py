#!/usr/bin/env python3
"""Compare two images rendered by different drivers.

Reads PNG and binary PPM (P6), so it works both on the offscreen bench and on
the ppm files the Sascha Willems samples write with --offscreen.

Reports the fraction of pixels that differ beyond a tolerance, plus a coarse
ASCII map of where they differ, which is usually enough to tell a geometry bug
from a shading bug at a glance.

Decoding goes through Pillow when it is installed, which is the case in the
repo venv and is what everything here should be run with. The readers further
down are a fallback for a bare interpreter: they are what this file used
before, they cover only the 8 bit RGB and RGBA that this tree produces, and
they undo the PNG row filters a byte at a time in Python, which for anything
but an unfiltered image is a hundred times slower than Pillow.
"""

import struct
import sys
import zlib

try:
    from PIL import Image
except ImportError:
    Image = None


def unfilter_row(line, prev, filt, channels, stride):
    """One row, a byte at a time. Correct for every filter, slow for all five."""
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
    return bytes(line)


def unfilter(data, height, stride, channels):
    """Undo the row filters, which is where all of a PNG's decode time goes.

    The one shortcut worth having without Pillow is filter None: the row is
    already the answer, and it is what every PNG written in this tree uses, so
    the common case costs a slice rather than 3.7 million loop iterations. The
    other four are left to the byte loop — Average and Paeth read the byte
    they have just reconstructed, so they cannot be done any other way in
    Python, and an interpreter without Pillow is not the case to optimise for.
    """
    rows, prev, p = [], bytes(stride), 0
    for _ in range(height):
        filt = data[p]
        p += 1
        line = data[p:p + stride]
        p += stride
        if filt:
            line = unfilter_row(bytearray(line), prev, filt, channels, stride)
        rows.append(line)
        prev = line
    return rows


def read_png(path):
    raw = open(path, 'rb').read()
    pos, width, height, depth, color, idat = 8, None, None, None, None, b''
    while pos < len(raw):
        (length,) = struct.unpack('>I', raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + length]
        if ctype == b'IHDR':
            width, height, depth, color = struct.unpack('>IIBB', body[:10])
        elif ctype == b'IDAT':
            idat += body
        pos += 12 + length

    # Only what the tools here write and read: 8 bit RGB or RGBA. Greyscale,
    # palettes and 16 bit would decode as nonsense rather than fail.
    if depth != 8 or color not in (2, 6):
        raise ValueError(f'{path}: only 8 bit RGB and RGBA PNGs are supported '
                         f'(depth {depth}, colour type {color})')

    channels = 4 if color == 6 else 3
    return width, height, channels, unfilter(zlib.decompress(idat), height,
                                             width * channels, channels)


def read_ppm(path):
    """Binary PPM (P6), which is what the samples write in offscreen mode."""
    raw = open(path, 'rb').read()

    # Header is "P6" plus width, height and maxval, whitespace separated, with
    # '#' comments allowed in between. A single whitespace byte follows maxval.
    fields, pos = [], 2
    while len(fields) < 3:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b'#':
            while raw[pos:pos + 1] not in (b'\n', b''):
                pos += 1
            continue
        start = pos
        while not raw[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(raw[start:pos]))
    pos += 1

    width, height, maxval = fields
    if maxval != 255:
        raise ValueError(f'{path}: only 8 bit PPMs are supported (maxval {maxval})')

    stride = width * 3
    rows = [raw[pos + y * stride:pos + (y + 1) * stride] for y in range(height)]
    return width, height, 3, rows


def read_image(path):
    """(width, height, channels, rows), one bytes object of 8 bit RGB or RGBA
    per row.

    Pillow decodes in C and knows every PNG there is — filtered, interlaced,
    palettised, 16 bit — where the reader above knows only the two forms this
    tree writes. On a 720p RGBA frame with Paeth filtering, which is what most
    encoders that are not this one produce, it is the difference between six
    milliseconds and six hundred.
    """
    if Image is None:
        return read_ppm(path) if path.lower().endswith('.ppm') else read_png(path)

    with Image.open(path) as img:
        # Alpha is kept when the file has it, so the channel count still says
        # what was in the file; anything else, greyscale included, comes back
        # as RGB rather than as nonsense.
        rgb = img.convert('RGBA' if 'A' in img.getbands() else 'RGB')
        width, height = rgb.size
        channels = len(rgb.getbands())
        data = rgb.tobytes()

    stride = width * channels
    return width, height, channels, [data[y * stride:(y + 1) * stride]
                                     for y in range(height)]


def main():
    ref_path, test_path = sys.argv[1], sys.argv[2]
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    w, h, ch, ref = read_image(ref_path)
    w2, h2, ch2, test = read_image(test_path)
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
