#!/usr/bin/env python3
"""Build a single self-contained HTML page comparing two directories of renders.

Intended for the Sascha Willems samples run offscreen against two ICDs, but it
only cares that the two directories hold images with matching names.

    ./run_offscreen.sh                  # once per driver, into its own directory
    cp_gallery.py ref_dir cuda_dir -o compare.html

A third directory may be given, which puts two renderers side by side against
the same reference and adds a column for how they differ from each other. That
is how cudavk gets read against llvmpipe rather than against zero: llvmpipe
is the backend its sampler and clipper were ported from, so it is the honest
target, and a residual both of them share is what software rasterization costs
rather than a cudavk defect.

    cp_gallery.py ref_dir cuda_dir llvmpipe_dir -o three.html \
        --test-label cudavk --test2-label llvmpipe

Images are written as full resolution PNGs next to the page and referenced from
it, so every panel can be opened one to one. The page scales them down for
layout; clicking one opens it at its native size with nearest neighbour
sampling, so a single differing pixel stays a single crisp pixel.

Reads PNG and binary PPM through cp_compare, which uses Pillow when it is
installed and its own reader when it is not, so this runs under a bare
interpreter as well as under the venv.
"""

import argparse
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


def write_png_file(path, width, height, rows):
    with open(path, 'wb') as f:
        f.write(write_png(width, height, rows))


def as_rgb(rows, w, ch):
    """Drop the alpha channel, which the PNG writer does not emit."""
    if ch == 3:
        return rows
    out = []
    for src in rows:
        row = bytearray(w * 3)
        for x in range(w):
            row[x * 3:x * 3 + 3] = src[x * ch:x * ch + 3]
        out.append(row)
    return out


def compare(ref, test, w, h, ch_ref, ch_test, tol):
    """Per-pixel stats plus a full resolution difference map.

    The map is kept at the source resolution so it can be inspected one to one;
    it is black where the two agree, which is most of it, so PNG compresses it
    to a fraction of what a photographic image of the same size would cost.
    """
    diff_count = 0
    max_delta = 0
    rows = []

    for y in range(h):
        r, t = ref[y], test[y]
        row = bytearray(w * 3)
        for x in range(w):
            ri, ti = x * ch_ref, x * ch_test
            d = max(abs(r[ri] - t[ti]),
                    abs(r[ri + 1] - t[ti + 1]),
                    abs(r[ri + 2] - t[ti + 2]))
            if d > max_delta:
                max_delta = d
            if d > tol:
                diff_count += 1
                s = min(255, 64 + d)
                row[x * 3] = s
                row[x * 3 + 1] = max(0, s - 160)
        rows.append(row)

    return diff_count, max_delta, rows


PAGE = """<meta charset="UTF-8">
<title>{title}</title>
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
.grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }}
@media (max-width: 780px) {{ .grid3 {{ grid-template-columns: 1fr; }} }}
.closer {{ font-weight: 600; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 12px; color: var(--muted); margin-bottom: .3rem; }}
img {{ width: 100%; height: auto; display: block; border-radius: 5px; background: #000; }}
.flip {{ position: relative; }}
.flip img.b {{ position: absolute; inset: 0; opacity: 0; }}
.flip:hover img.b {{ opacity: 1; }}
.hint {{ font-size: 12px; color: var(--muted); margin-top: .5rem; }}
img.zoomable {{ cursor: zoom-in; }}
#lightbox {{
  display: none; position: fixed; inset: 0; z-index: 50; overflow: auto;
  background: rgba(0, 0, 0, .92); cursor: zoom-out; padding: 0;
}}
#lightbox.on {{ display: block; }}
/* Natural size, nearest neighbour: a single differing pixel has to stay a
   single crisp pixel when the map is inspected one to one. */
#lb-stack {{ position: relative; display: table; margin: 0 auto; }}
#lightbox img {{
  width: auto; max-width: none; border-radius: 0; display: block;
  image-rendering: pixelated;
}}
#lb-b {{ position: absolute; inset: 0; opacity: 0; }}
#lb-stack:hover #lb-b {{ opacity: 1; }}
#lb-hint {{
  position: fixed; left: 0; right: 0; bottom: 0; margin: 0; padding: .5rem;
  text-align: center; font-size: 13px; color: #cfcfd8;
  background: rgba(0, 0, 0, .65); pointer-events: none;
}}
</style>
<div class="wrap">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="summary">
<table>
{thead}
<tbody>
{rows}
</tbody></table>
</div>
{cards}
</div>
<div id="lightbox" onclick="this.classList.remove('on')">
<div id="lb-stack"><img id="lb-a" alt=""><img id="lb-b" alt=""></div>
<p id="lb-hint"></p>
</div>
<script>
function zoom(src, flip, label) {{
  var lb = document.getElementById('lightbox');
  var b = document.getElementById('lb-b');
  document.getElementById('lb-a').src = src;
  if (flip) {{
    b.src = flip;
    b.style.display = '';
  }} else {{
    b.removeAttribute('src');
    b.style.display = 'none';
  }}
  document.getElementById('lb-hint').textContent = label || '';
  lb.classList.add('on');
  return false;
}}
document.addEventListener('keydown', function (e) {{
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('on');
}});
</script>
"""

CARD = """<div class="card" id="{name}">
<div class="head"><h2>{name}</h2>
<span class="stat">{diff:,} / {total:,} px ({pct:.2f}%) &middot; max &Delta; {delta}</span>
<span class="{cls}">{verdict}</span></div>
<div class="grid">
<figure><figcaption>{ref_label}</figcaption>
<img class="zoomable" loading="lazy" src="{ref}" alt="{name} {ref_label}"
     onclick="zoom('{ref}', null, '{name} &mdash; {ref_label}')"></figure>
<figure><figcaption>{test_label} &mdash; hover to flip to {ref_label}</figcaption>
<div class="flip zoomable"
     onclick="zoom('{test}', '{ref}', '{name} &mdash; {test_label}, hover to flip to {ref_label}')">
<img loading="lazy" src="{test}" alt="{name} {test_label}">
<img class="b" loading="lazy" src="{ref}" alt="{name} {ref_label}"></div></figure>
<figure><figcaption>difference &gt; {tol}/255</figcaption>
<img class="zoomable" loading="lazy" src="{diffmap}" alt="{name} difference"
     onclick="zoom('{diffmap}', null, '{name} &mdash; difference')"></figure>
</div>
</div>
"""

CARD3 = """<div class="card" id="{name}">
<div class="head"><h2>{name}</h2>
<span class="stat">{test_label} {diff:,} px ({pct:.2f}%) &middot; {test2_label} {diff2:,} px ({pct2:.2f}%)
&middot; between them {diffx:,} px</span>
<span class="{cls}">{verdict}</span></div>
<div class="grid3">
<figure><figcaption>{ref_label}</figcaption>
<img class="zoomable" loading="lazy" src="{ref}" alt="{name} {ref_label}"
     onclick="zoom('{ref}', null, '{name} &mdash; {ref_label}')"></figure>
<figure><figcaption>{test_label} &mdash; hover to flip to {ref_label}</figcaption>
<div class="flip zoomable"
     onclick="zoom('{test}', '{ref}', '{name} &mdash; {test_label}, hover to flip to {ref_label}')">
<img loading="lazy" src="{test}" alt="{name} {test_label}">
<img class="b" loading="lazy" src="{ref}" alt="{name} {ref_label}"></div></figure>
<figure><figcaption>{test2_label} &mdash; hover to flip to {ref_label}</figcaption>
<div class="flip zoomable"
     onclick="zoom('{test2}', '{ref}', '{name} &mdash; {test2_label}, hover to flip to {ref_label}')">
<img loading="lazy" src="{test2}" alt="{name} {test2_label}">
<img class="b" loading="lazy" src="{ref}" alt="{name} {ref_label}"></div></figure>
<figure><figcaption>{test_label} vs {test2_label} &gt; {tol}/255</figcaption>
<img class="zoomable" loading="lazy" src="{diffxmap}" alt="{name} {test_label} vs {test2_label}"
     onclick="zoom('{diffxmap}', null, '{name} &mdash; {test_label} vs {test2_label}')"></figure>
<figure><figcaption>{test_label} vs {ref_label} &gt; {tol}/255</figcaption>
<img class="zoomable" loading="lazy" src="{diffmap}" alt="{name} {test_label} difference"
     onclick="zoom('{diffmap}', null, '{name} &mdash; {test_label} vs {ref_label}')"></figure>
<figure><figcaption>{test2_label} vs {ref_label} &gt; {tol}/255</figcaption>
<img class="zoomable" loading="lazy" src="{diff2map}" alt="{name} {test2_label} difference"
     onclick="zoom('{diff2map}', null, '{name} &mdash; {test2_label} vs {ref_label}')"></figure>
</div>
</div>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ref_dir', help='directory of reference images')
    ap.add_argument('test_dir', help='directory of images under test')
    ap.add_argument('test2_dir', nargs='?', default=None,
                    help='optional second directory under test, compared '
                         'against the same reference and against the first')
    ap.add_argument('-o', '--out', default='compare.html', help='output HTML file')
    ap.add_argument('-t', '--tol', type=int, default=8,
                    help='per-channel tolerance, 0-255 (default 8)')
    ap.add_argument('--images', default=None,
                    help='directory for the PNGs the page references '
                         '(default: <out>_images next to the page)')
    ap.add_argument('--ref-label', default='reference')
    ap.add_argument('--test-label', default='cudavk')
    ap.add_argument('--test2-label', default='llvmpipe')
    ap.add_argument('--title', default='Renderer comparison')
    args = ap.parse_args()

    img_dir = args.images or os.path.splitext(args.out)[0] + '_images'
    os.makedirs(img_dir, exist_ok=True)
    rel = os.path.relpath(img_dir, os.path.dirname(os.path.abspath(args.out)))

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
        if args.test2_dir and \
           not os.path.exists(os.path.join(args.test2_dir, base + ext)):
            continue
        if base not in found or ext == '.ppm':
            found[base] = ext
    names = sorted(found.items())

    if not names:
        dirs = ', '.join(d for d in (args.ref_dir, args.test_dir, args.test2_dir) if d)
        print(f'no images found in common between {dirs}')
        return 1

    results = []
    for base, ext in names:
        rw, rh, rch, ref = read_image(os.path.join(args.ref_dir, base + ext))
        tw, th, tch, test = read_image(os.path.join(args.test_dir, base + ext))
        if (rw, rh) != (tw, th):
            print(f'{base}: size mismatch {rw}x{rh} vs {tw}x{th}, skipped')
            continue

        diff, delta, dmap = compare(ref, test, rw, rh, rch, tch, args.tol)

        extra = {}
        if args.test2_dir:
            uw, uh, uch, test2 = read_image(os.path.join(args.test2_dir, base + ext))
            if (rw, rh) != (uw, uh):
                print(f'{base}: size mismatch {rw}x{rh} vs {uw}x{uh} in '
                      f'{args.test2_dir}, skipped')
                continue
            diff2, delta2, d2map = compare(ref, test2, rw, rh, rch, uch, args.tol)
            # The two renderers against each other, which is where a shared
            # residual shows up as black.
            diffx, _, dxmap = compare(test, test2, rw, rh, tch, uch, args.tol)
            write_png_file(os.path.join(img_dir, base + '_test2.png'),
                           uw, uh, as_rgb(test2, uw, uch))
            write_png_file(os.path.join(img_dir, base + '_diff2.png'), rw, rh, d2map)
            write_png_file(os.path.join(img_dir, base + '_diffx.png'), rw, rh, dxmap)
            extra = {'diff2': diff2, 'delta2': delta2, 'diffx': diffx,
                     'pct2': 100.0 * diff2 / (rw * rh)}

        write_png_file(os.path.join(img_dir, base + '_ref.png'),
                       rw, rh, as_rgb(ref, rw, rch))
        write_png_file(os.path.join(img_dir, base + '_test.png'),
                       tw, th, as_rgb(test, tw, tch))
        write_png_file(os.path.join(img_dir, base + '_diff.png'), rw, rh, dmap)

        # A cache-busting stamp: the page keeps the same file names across
        # runs, and a browser will happily show the previous run's images.
        stamp = int(os.path.getmtime(os.path.join(img_dir, base + '_diff.png')))

        results.append({
            'name': base,
            'diff': diff,
            'total': rw * rh,
            'pct': 100.0 * diff / (rw * rh),
            'delta': delta,
            'ref': f'{rel}/{base}_ref.png?v={stamp}',
            'test': f'{rel}/{base}_test.png?v={stamp}',
            'diffmap': f'{rel}/{base}_diff.png?v={stamp}',
        })
        if extra:
            results[-1].update(extra)
            results[-1].update({
                'test2': f'{rel}/{base}_test2.png?v={stamp}',
                'diff2map': f'{rel}/{base}_diff2.png?v={stamp}',
                'diffxmap': f'{rel}/{base}_diffx.png?v={stamp}',
            })
            print(f'{base:24s} {diff:>8} vs {extra["diff2"]:>8}  '
                  f'(between them {extra["diffx"]:>8})')
        else:
            print(f'{base:24s} {diff:>8}/{rw * rh} '
                  f'({100.0 * diff / (rw * rh):6.2f}%)')

    results.sort(key=lambda r: -r['pct'])

    three = bool(args.test2_dir)
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

        if three:
            # Which renderer landed closer to the reference. Ties and margins
            # under a hundred pixels are not worth calling, since either can
            # move by that much on a single antialiased edge.
            if abs(r['diff'] - r['diff2']) < 100:
                closer, ccls = 'level', 'stat'
            elif r['diff'] < r['diff2']:
                closer, ccls = args.test_label, 'pass'
            else:
                closer, ccls = args.test2_label, 'near'
            rows.append(
                f'<tr><td><a href="#{r["name"]}">{r["name"]}</a></td>'
                f'<td class="num">{r["diff"]:,}</td>'
                f'<td class="num">{r["pct"]:.2f}%</td>'
                f'<td class="num">{r["diff2"]:,}</td>'
                f'<td class="num">{r["pct2"]:.2f}%</td>'
                f'<td class="num">{r["diffx"]:,}</td>'
                f'<td class="closer {ccls}">{closer}</td></tr>')
            cards.append(CARD3.format(tol=args.tol, ref_label=args.ref_label,
                                      test_label=args.test_label,
                                      test2_label=args.test2_label, **r))
        else:
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

    if three:
        tot = sum(r['diff'] for r in results)
        tot2 = sum(r['diff2'] for r in results)
        wins = sum(1 for r in results if r['diff'] < r['diff2'] - 100)
        thead = (f'<thead><tr><th>Sample</th>'
                 f'<th class="num">{args.test_label}</th><th class="num">share</th>'
                 f'<th class="num">{args.test2_label}</th><th class="num">share</th>'
                 f'<th class="num">between them</th><th>closer</th></tr></thead>')
        rows.append(
            f'<tr><td><b>total</b></td>'
            f'<td class="num"><b>{tot:,}</b></td><td class="num"></td>'
            f'<td class="num"><b>{tot2:,}</b></td><td class="num"></td>'
            f'<td class="num"><b>{sum(r["diffx"] for r in results):,}</b></td>'
            f'<td></td></tr>')
        subtitle = (
            f'{args.test_label} and {args.test2_label} against the same '
            f'{args.ref_label}, at a tolerance of {args.tol}/255. Totals over '
            f'{len(results)} samples: {tot:,} pixels for {args.test_label}, '
            f'{tot2:,} for {args.test2_label}; {args.test_label} is closer on '
            f'{wins}. Read the residual against {args.test2_label} rather than '
            f'against zero \u2014 a difference both of them share is what '
            f'software rasterization costs, not a defect. The fourth panel is '
            f'the two of them against each other, so it goes black wherever '
            f'they agree. Hover a render to flip it to {args.ref_label}, click '
            f'any panel for full resolution. Sorted worst first.')
    else:
        thead = ('<thead><tr><th>Sample</th><th class="num">Differing</th>'
                 '<th class="num">Share</th>'
                 '<th class="num">Max &Delta;</th><th>Verdict</th></tr></thead>')
        subtitle = (f'{matched} of {len(results)} samples match exactly and {near} '
                    f'more differ in under 0.1% of pixels, at a tolerance of '
                    f'{args.tol}/255. Left is {args.ref_label}, middle is '
                    f'{args.test_label} (hover to flip), right marks where they '
                    f'disagree. Click any panel to open it at full resolution. '
                    f'Sorted worst first.')

    html = PAGE.format(title=args.title, subtitle=subtitle, thead=thead,
                       rows='\n'.join(rows), cards='\n'.join(cards))
    with open(args.out, 'w') as f:
        f.write(html)

    size = os.path.getsize(args.out)
    imgs = sum(os.path.getsize(os.path.join(img_dir, f))
               for f in os.listdir(img_dir))
    print(f'\n{matched}/{len(results)} match exactly, {near} near  ->  '
          f'{args.out} ({size / 1e3:.0f} kB, images {imgs / 1e6:.0f} MB in '
          f'{img_dir})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
