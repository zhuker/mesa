#!/usr/bin/env python3
"""Build a single self-contained HTML page comparing two directories of renders.

Intended for the Sascha Willems samples run offscreen against two ICDs, but it
only cares that the two directories hold images with matching names.

    ./run_offscreen.sh                  # once per driver, into two directories
    cp_gallery.py ref_dir cuda_dir -o compare.html

Every thumbnail is embedded as a data URI, so the page is one file that can be
copied or served from anywhere. Thumbnails are point sampled rather than
filtered, so what the page shows is real pixels rather than an average that
could invent or hide a difference. The difference map is computed at full
resolution and then reduced by taking the worst pixel in each block, so a single
differing pixel still lights up after downscaling.

Reads PNG and binary PPM through cp_compare, and needs nothing outside the
standard library.
"""

import argparse
import base64
import glob
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_compare import read_image


def write_png(width, height, rows):
    """Encode 8 bit RGB rows (bytes of length width*3) as a PNG."""
    raw = b''.join(b'\x00' + bytes(row) for row in rows)

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(raw, 6)) +
            chunk(b'IEND', b''))


def data_uri(png):
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


def thumbnail(rows, w, h, ch, out_w):
    """Point sampled downscale to out_w, preserving aspect."""
    out_w = min(out_w, w)
    out_h = max(1, h * out_w // w)
    out = []
    for y in range(out_h):
        sy = y * h // out_h
        src = rows[sy]
        row = bytearray(out_w * 3)
        for x in range(out_w):
            sx = (x * w // out_w) * ch
            row[x * 3:x * 3 + 3] = src[sx:sx + 3]
        out.append(row)
    return out, out_w, out_h


def compare(ref, test, w, h, ch_ref, ch_test, tol, out_w):
    """Per-pixel stats plus a difference map reduced by worst-in-block."""
    out_w = min(out_w, w)
    out_h = max(1, h * out_w // w)
    block = [[0] * out_w for _ in range(out_h)]

    diff_count = 0
    max_delta = 0

    for y in range(h):
        r, t = ref[y], test[y]
        by = block[min(y * out_h // h, out_h - 1)]
        for x in range(w):
            ri, ti = x * ch_ref, x * ch_test
            d = max(abs(r[ri] - t[ti]),
                    abs(r[ri + 1] - t[ti + 1]),
                    abs(r[ri + 2] - t[ti + 2]))
            if d > max_delta:
                max_delta = d
            if d > tol:
                diff_count += 1
                bx = min(x * out_w // w, out_w - 1)
                if d > by[bx]:
                    by[bx] = d

    # Black where the two agree, heat where they do not.
    rows = []
    for by in block:
        row = bytearray(out_w * 3)
        for x, d in enumerate(by):
            if d:
                s = min(255, 64 + d)
                row[x * 3] = s
                row[x * 3 + 1] = max(0, s - 160)
        rows.append(row)

    return diff_count, max_delta, rows, out_w, out_h


PAGE = """<title>{title}</title>
<style>
:root {{
  --bg: #ffffff; --panel: #f5f5f7; --border: #d8d8de; --fg: #1a1a1f;
  --muted: #61616b; --pass: #157f3b; --near: #8a6100; --fail: #b4341f; --accent: #2f6fdb;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #16161a; --panel: #1f1f25; --border: #33333d; --fg: #e8e8ee;
    --muted: #9a9aa6; --pass: #4ec97d; --near: #e0b33a; --fail: #ff7a63; --accent: #7aa8ff;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #16161a; --panel: #1f1f25; --border: #33333d; --fg: #e8e8ee;
  --muted: #9a9aa6; --pass: #4ec97d; --near: #e0b33a; --fail: #ff7a63; --accent: #7aa8ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .35rem; }}
.sub {{ color: var(--muted); margin: 0 0 1.75rem; }}
.summary {{ overflow-x: auto; margin-bottom: 2.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: .45rem .7rem; border-bottom: 1px solid var(--border); }}
th {{ color: var(--muted); font-weight: 600; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.pass {{ color: var(--pass); font-weight: 600; }}
.near {{ color: var(--near); font-weight: 600; }}
.fail {{ color: var(--fail); font-weight: 600; }}
a {{ color: var(--accent); }}
.card {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem; margin-bottom: 1.5rem;
}}
.head {{ display: flex; flex-wrap: wrap; gap: .75rem; align-items: baseline; margin-bottom: .8rem; }}
.head h2 {{ font-size: 1.1rem; margin: 0; }}
.stat {{ color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: .75rem; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 12px; color: var(--muted); margin-bottom: .3rem; }}
img {{ width: 100%; height: auto; display: block; border-radius: 5px; background: #000; }}
.flip {{ position: relative; cursor: pointer; }}
.flip img.b {{ position: absolute; inset: 0; opacity: 0; }}
.flip:hover img.b {{ opacity: 1; }}
.hint {{ font-size: 12px; color: var(--muted); margin-top: .5rem; }}
</style>
<div class="wrap">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="summary">
<table>
<thead><tr><th>Sample</th><th class="num">Differing</th><th class="num">Share</th>
<th class="num">Max &Delta;</th><th>Verdict</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</div>
{cards}
</div>
"""

CARD = """<div class="card" id="{name}">
<div class="head"><h2>{name}</h2>
<span class="stat">{diff:,} / {total:,} px ({pct:.2f}%) &middot; max &Delta; {delta}</span>
<span class="{cls}">{verdict}</span></div>
<div class="grid">
<figure><figcaption>{ref_label}</figcaption><img src="{ref}" alt="{name} {ref_label}"></figure>
<figure><figcaption>{test_label} &mdash; hover to flip to {ref_label}</figcaption>
<div class="flip"><img src="{test}" alt="{name} {test_label}">
<img class="b" src="{ref}" alt="{name} {ref_label}"></div></figure>
<figure><figcaption>difference &gt; {tol}/255</figcaption><img src="{diffmap}" alt="{name} difference"></figure>
</div>
</div>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ref_dir', help='directory of reference images')
    ap.add_argument('test_dir', help='directory of images under test')
    ap.add_argument('-o', '--out', default='compare.html', help='output HTML file')
    ap.add_argument('-t', '--tol', type=int, default=8,
                    help='per-channel tolerance, 0-255 (default 8)')
    ap.add_argument('-w', '--width', type=int, default=480,
                    help='thumbnail width in pixels (default 480)')
    ap.add_argument('--ref-label', default='reference')
    ap.add_argument('--test-label', default='cudapipe')
    ap.add_argument('--title', default='Renderer comparison')
    args = ap.parse_args()

    # A directory may hold the same render as both .ppm and .png; one entry per
    # sample, preferring the ppm the samples write themselves.
    found = {}
    for path in sorted(glob.glob(os.path.join(args.ref_dir, '*'))):
        base, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        if ext not in ('.ppm', '.png'):
            continue
        if not os.path.exists(os.path.join(args.test_dir, base + ext)):
            continue
        if base not in found or ext == '.ppm':
            found[base] = ext
    names = sorted(found.items())

    if not names:
        print(f'no image pairs found between {args.ref_dir} and {args.test_dir}')
        return 1

    results = []
    for base, ext in names:
        rw, rh, rch, ref = read_image(os.path.join(args.ref_dir, base + ext))
        tw, th, tch, test = read_image(os.path.join(args.test_dir, base + ext))
        if (rw, rh) != (tw, th):
            print(f'{base}: size mismatch {rw}x{rh} vs {tw}x{th}, skipped')
            continue

        diff, delta, dmap, dw, dh = compare(ref, test, rw, rh, rch, tch,
                                            args.tol, args.width)
        ref_t, w1, h1 = thumbnail(ref, rw, rh, rch, args.width)
        test_t, w2, h2 = thumbnail(test, tw, th, tch, args.width)

        results.append({
            'name': base,
            'diff': diff,
            'total': rw * rh,
            'pct': 100.0 * diff / (rw * rh),
            'delta': delta,
            'ref': data_uri(write_png(w1, h1, ref_t)),
            'test': data_uri(write_png(w2, h2, test_t)),
            'diffmap': data_uri(write_png(dw, dh, dmap)),
        })
        print(f'{base:24s} {diff:>8}/{rw * rh} ({100.0 * diff / (rw * rh):6.2f}%)')

    results.sort(key=lambda r: -r['pct'])

    rows, cards = [], []
    for r in results:
        # A handful of stray pixels on a triangle edge is a different thing
        # from a sample that renders the wrong image, so say which it is.
        if r['diff'] == 0:
            r['cls'], r['verdict'] = 'pass', 'match'
        elif r['pct'] < 0.1:
            r['cls'], r['verdict'] = 'near', 'near match'
        else:
            r['cls'], r['verdict'] = 'fail', 'differs'
        rows.append(
            f'<tr><td><a href="#{r["name"]}">{r["name"]}</a></td>'
            f'<td class="num">{r["diff"]:,}</td>'
            f'<td class="num">{r["pct"]:.2f}%</td>'
            f'<td class="num">{r["delta"]}</td>'
            f'<td class="{r["cls"]}">{r["verdict"]}</td></tr>')
        cards.append(CARD.format(tol=args.tol, ref_label=args.ref_label,
                                 test_label=args.test_label, **r))

    matched = sum(1 for r in results if r['diff'] == 0)
    near = sum(1 for r in results if 0 < r['pct'] < 0.1)
    subtitle = (f'{matched} of {len(results)} samples match exactly and {near} '
                f'more differ in under 0.1% of pixels, at a tolerance of '
                f'{args.tol}/255. Left is {args.ref_label}, middle is '
                f'{args.test_label} (hover to flip), right marks where they '
                f'disagree. Sorted worst first.')

    html = PAGE.format(title=args.title, subtitle=subtitle,
                       rows='\n'.join(rows), cards='\n'.join(cards))
    with open(args.out, 'w') as f:
        f.write(html)

    size = os.path.getsize(args.out)
    print(f'\n{matched}/{len(results)} match exactly, {near} near  ->  '
          f'{args.out} ({size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
