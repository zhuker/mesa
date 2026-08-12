#!/usr/bin/env python3
"""Build an HTML report over a multi-driver, multi-frame sweep.

Consumes whatever cp_perf_run.sh wrote — one directory per driver, each holding
_timing.csv, _gpu.csv and a directory of numbered frames per sample — and emits
a single page covering the two questions a sweep is run to answer: what did it
cost, and did it stay correct for the whole animation.

    cp_perf_run.sh nvidia   ""        build/frames60/nvidia   60
    cp_perf_run.sh cudapipe "$CUDA"   build/frames60/cuda     60
    cp_perf_run.sh llvmpipe "$LVP"    build/frames60/llvmpipe 60

    cp_perf_report.py build/frames60 --ref nvidia -o perf.html

Drivers are discovered from the directories present unless --drivers names them;
the order given is the order they are drawn in, and a driver keeps its colour
across every chart on the page.

Per-frame differences are computed by ffmpeg, one invocation per sample rather
than one per frame: the difference of the two streams, the max across the three
channels via two `lighten` blends, a threshold, and signalstats' YAVG of the
resulting mask, which is the differing fraction. A 60 frame sample takes about
a third of a second, and there is nothing to install. Results are cached in
_diffs.json keyed by tolerance, so re-running to change the layout is free;
--recompute discards it.

Frames are exported to JPEG alongside, so the page can show any frame of any
driver next to the reference — the chart says which frame went wrong and the
viewer under it says what it looked like. --no-frames skips the export and
leaves the charts alone.

Nothing here is specific to this set of samples or to 60 frames.
"""

import argparse
import csv
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_compare import read_image

import re
import shutil
import subprocess

# Slots 1-3 of the reference categorical palette, in its fixed order. Each
# driver keeps its slot on every chart, so colour identifies the driver and
# never its rank.
SERIES = [
    {'light': '#2a78d6', 'dark': '#3987e5'},   # 1 blue
    {'light': '#eb6834', 'dark': '#d95926'},   # 2 orange
    {'light': '#1baf7a', 'dark': '#199e70'},   # 3 aqua
    {'light': '#eda100', 'dark': '#c98500'},   # 4 yellow
    {'light': '#e87ba4', 'dark': '#d55181'},   # 5 magenta
    {'light': '#008300', 'dark': '#008300'},   # 6 green
    {'light': '#4a3aa7', 'dark': '#9085e9'},   # 7 violet
    {'light': '#e34948', 'dark': '#e66767'},   # 8 red
]


# ---------------------------------------------------------------- reading ---

def read_timing(driver_dir):
    """sample -> {wall, cpu, rss_mb, exit} from _timing.csv."""
    path = os.path.join(driver_dir, '_timing.csv')
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        try:
            out[r['sample']] = {
                'wall': float(r['wall_s']),
                'cpu': float(r['user_s']) + float(r['sys_s']),
                'rss_mb': float(r['maxrss_kb']) / 1024.0,
                'exit': int(r['exit']),
            }
        except (ValueError, KeyError):
            continue
    return out


def read_gpu(driver_dir):
    """Elapsed seconds, utilisation %, memory MiB from _gpu.csv."""
    path = os.path.join(driver_dir, '_gpu.csv')
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        try:
            util = float(parts[1].rstrip(' %'))
            used = float(parts[3].rstrip(' MiB'))
        except ValueError:
            continue
        rows.append((util, used))
    # nvidia-smi timestamps are wall clock strings; the sampler is fixed rate,
    # so index * period is a truer x than parsing them.
    return [(i * 0.5, u, m) for i, (u, m) in enumerate(rows)]


def frames_of(driver_dir, sample):
    d = os.path.join(driver_dir, sample)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(('.ppm', '.png')))


def samples_in(driver_dir):
    if not os.path.isdir(driver_dir):
        return []
    return sorted(s for s in os.listdir(driver_dir)
                  if os.path.isdir(os.path.join(driver_dir, s))
                  and not s.startswith('_'))


def frame_pattern(driver_dir, sample):
    """The ffmpeg image2 pattern for a sample's frames, and how many there are."""
    names = frames_of(driver_dir, sample)
    if not names:
        return None, 0
    pat = re.sub(r'\d{4}(\.(?:ppm|png))$', r'%04d\1', names[0])
    if pat == names[0]:
        return None, 0
    return os.path.join(driver_dir, sample, pat), len(names)


# The difference of the two streams, the max across the three channels (two
# `lighten` blends, since lighten is a per-pixel max), a threshold at the
# tolerance, and the mean of the resulting 0/255 mask. That mean over 255 is the
# fraction of pixels differing, which is the same quantity cp_compare.py counts.
FILTER = (
    "[0][1]blend=all_mode=difference,format=gbrp,extractplanes=g+b+r[g][b][r];"
    "[g][b]blend=all_mode=lighten[gb];"
    "[gb][r]blend=all_mode=lighten,"
    "lutyuv=y='if(gt(val\\,{tol})\\,255\\,0)',signalstats,metadata=print"
)


def diff_series(ref_pat, test_pat, tol, pixels):
    """Differing pixels per frame, via one ffmpeg pass over the sequence."""
    cmd = ['ffmpeg', '-hide_banner', '-i', ref_pat, '-i', test_pat,
           '-filter_complex', FILTER.format(tol=tol), '-f', 'null', '-']
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stderr
    except (OSError, subprocess.SubprocessError):
        return []
    vals = re.findall(r'YAVG=([0-9.]+)', out)
    return [[i, int(round(float(v) / 255.0 * pixels))] for i, v in enumerate(vals)]


def frame_size(path):
    """(width, height) of one frame, read through the existing PPM/PNG reader."""
    try:
        w, h, _, _ = read_image(path)
        return w, h
    except Exception:
        return 0, 0


def frame_diffs(root, ref, drivers, samples, tol, recompute):
    """{driver: {sample: [[frame, differing_pixels], ...]}}, cached."""
    cache_path = os.path.join(root, '_diffs.json')
    key = f'tol{tol}'
    cache = {}
    if os.path.exists(cache_path) and not recompute:
        try:
            cache = json.load(open(cache_path))
        except ValueError:
            cache = {}
    cache.setdefault(key, {})

    ref_dir = os.path.join(root, ref)
    for drv in drivers:
        if drv == ref:
            continue
        cache[key].setdefault(drv, {})
        for sample in samples:
            if sample in cache[key][drv]:
                continue
            rp, n = frame_pattern(ref_dir, sample)
            tp, m = frame_pattern(os.path.join(root, drv), sample)
            if not rp or not tp:
                cache[key][drv][sample] = []
                continue
            w, h = frame_size(os.path.join(ref_dir, sample,
                                           frames_of(ref_dir, sample)[0]))
            series = diff_series(rp, tp, tol, w * h) if w and h else []
            cache[key][drv][sample] = series
            print(f'  {drv}/{sample}: {len(series)} frames', flush=True)

    try:
        json.dump(cache, open(cache_path, 'w'))
    except OSError:
        pass
    return cache[key]


def _run(cmd):
    try:
        subprocess.run(cmd, capture_output=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        pass


def export_frames(root, drivers, samples, img_dir, width):
    """Every frame twice: scaled down to lay out with, and native to inspect.

    PNG for both. Nothing here may be lossy — the whole page exists to judge
    whether a pixel is wrong, and JPEG would put its own artefacts in front of
    that. It is also what hid a bug: PNG output negotiated RGB and looked
    right, while the JPEG encoder pulled the filter graph into YUV, where a
    difference leaves U and V at zero and every difference image came out
    green.
    """
    for drv in drivers:
        for sample in samples:
            pat, n = frame_pattern(os.path.join(root, drv), sample)
            if not pat:
                continue
            small = os.path.join(img_dir, drv, sample)
            full = os.path.join(img_dir, drv, sample, 'full')
            if not (os.path.isdir(small) and
                    len([f for f in os.listdir(small)
                         if f.endswith('.png')]) >= n):
                os.makedirs(small, exist_ok=True)
                _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                      '-i', pat, '-vf', f'format=rgb24,scale={width}:-2',
                      os.path.join(small, '%04d.png')])
            if not (os.path.isdir(full) and len(os.listdir(full)) >= n):
                os.makedirs(full, exist_ok=True)
                _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                      '-i', pat, '-vf', 'format=rgb24',
                      os.path.join(full, '%04d.png')])
            print(f'  frames {drv}/{sample}', flush=True)


# The difference, amplified so it is visible at all: a handful of levels out of
# 255 is invisible otherwise. colorlevels rescales the input range, which is a
# gain on a near-black image; eq's contrast pivots around mid grey instead and
# so drives those same values towards black, which is the opposite of what is
# wanted here. The gain means the image says "here", not "by how much" — the
# chart above it carries the magnitude.
# Both inputs are pinned to RGB before the blend. Without that the filter
# graph negotiates whatever the output encoder wants — YUV, for JPEG — and a
# difference taken there leaves U and V at zero rather than neutral, which
# converts back to saturated green. The PNG path negotiated RGB and looked
# right, so the bug only showed in the thumbnails.
DIFF_FILTER = ("[0]format=rgb24[a];[1]format=rgb24[b];"
               "[a][b]blend=all_mode=difference,format=rgb24,"
               "colorlevels=rimax={imax}:gimax={imax}:bimax={imax}")


def export_diffs(root, ref, drivers, samples, img_dir, width, gain):
    """A difference image per frame, per driver under test, at both sizes."""
    ref_dir = os.path.join(root, ref)
    for drv in drivers:
        if drv == ref:
            continue
        for sample in samples:
            rp, n = frame_pattern(ref_dir, sample)
            tp, _ = frame_pattern(os.path.join(root, drv), sample)
            if not rp or not tp:
                continue
            small = os.path.join(img_dir, '_diff', drv, sample)
            full = os.path.join(small, 'full')
            need_small = not (os.path.isdir(small) and len(
                [f for f in os.listdir(small) if f.endswith('.png')]) >= n)
            need_full = not (os.path.isdir(full) and len(os.listdir(full)) >= n)
            if need_small:
                os.makedirs(small, exist_ok=True)
                _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                      '-i', rp, '-i', tp, '-filter_complex',
                      DIFF_FILTER.format(imax=1.0 / max(gain, 1)) +
                      f',scale={width}:-2',
                      os.path.join(small, '%04d.png')])
            if need_full:
                os.makedirs(full, exist_ok=True)
                _run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                      '-i', rp, '-i', tp, '-filter_complex',
                      DIFF_FILTER.format(imax=1.0 / max(gain, 1)),
                      os.path.join(full, '%04d.png')])
            print(f'  diffs {drv}/{sample}', flush=True)


# ----------------------------------------------------------------- charts ---

def line_chart(series, width=340, height=130, pad_l=48, pad_b=22, pad_t=10,
               pad_r=8, y_zero=True, fmt='{:,.0f}'):
    """An SVG line chart. `series` is [(label, slot, [(x, y), ...]), ...].

    One y-axis, always — two measures of different scale get two charts.
    """
    pts = [p for _, _, ps in series for p in ps]
    if not pts:
        return '<p class="empty">no data</p>', []

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0 = 0 if y_zero else min(ys)
    y1 = max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    iw = width - pad_l - pad_r
    ih = height - pad_t - pad_b

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * iw

    def sy(y):
        return pad_t + ih - (y - y0) / (y1 - y0) * ih

    # The geometry travels with the chart so a click can be turned back into
    # an x value without the page having to know how it was drawn.
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
           f'preserveAspectRatio="none" role="img" '
           f'data-x0="{x0}" data-x1="{x1}" data-pl="{pad_l}" '
           f'data-pr="{pad_r}" data-w="{width}">']

    # Recessive gridlines and value labels at the two ends of the range.
    for frac in (0.0, 0.5, 1.0):
        yv = y0 + (y1 - y0) * frac
        y = sy(yv)
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" '
                   f'x2="{width - pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" '
                   f'text-anchor="end">{fmt.format(yv)}</text>')

    out.append(f'<text class="tick" x="{pad_l}" y="{height - 6}">{x0:g}</text>')
    out.append(f'<text class="tick" x="{width - pad_r}" y="{height - 6}" '
               f'text-anchor="end">{x1:g}</text>')

    for label, slot, ps in series:
        if not ps:
            continue
        d = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in ps)
        out.append(f'<polyline class="line s{slot}" points="{d}"/>')
        # A marker at the end anchors the line to its identity.
        ex, ey = ps[-1]
        out.append(f'<circle class="dot s{slot}" cx="{sx(ex):.1f}" '
                   f'cy="{sy(ey):.1f}" r="3"/>')

    # Where the viewer is currently parked. Hidden until something sets it.
    out.append(f'<line class="cursor" x1="0" y1="{pad_t}" x2="0" '
               f'y2="{pad_t + ih}" style="display:none"/>')
    out.append('</svg>')
    return '\n'.join(out), [(lbl, slot) for lbl, slot, _ in series]


def bar_rows(rows, slots, fmt='{:.2f}'):
    """Grouped horizontal bars: rows is [(name, [(label, slot, value), ...])]."""
    peak = max((v for _, vals in rows for _, _, v in vals), default=0) or 1
    out = []
    for name, vals in rows:
        bars = []
        for label, slot, v in vals:
            pct = 100.0 * v / peak
            bars.append(
                f'<div class="barrow"><span class="barlabel">{html.escape(label)}</span>'
                f'<span class="bartrack"><span class="bar s{slot}" '
                f'style="width:{pct:.2f}%"></span></span>'
                f'<span class="barval">{fmt.format(v)}</span></div>')
        out.append(f'<div class="barsample"><h3>{html.escape(name)}</h3>'
                   + ''.join(bars) + '</div>')
    return ''.join(out)


# ------------------------------------------------------------------- page ---

PAGE = """<meta charset="UTF-8">
<title>{title}</title>
<style>
:root {{
  --bg:#ffffff; --panel:#fcfcfb; --border:#d8d8de; --fg:#0b0b0b;
  --muted:#52514e; --good:#157f3b; --bad:#b4341f; --accent:#2a78d6;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#141416; --panel:#1a1a19; --border:#33333d; --fg:#ffffff;
    --muted:#c3c2b7; --good:#4ec97d; --bad:#ff7a63; --accent:#3987e5;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#141416; --panel:#1a1a19; --border:#33333d; --fg:#ffffff;
  --muted:#c3c2b7; --good:#4ec97d; --bad:#ff7a63; --accent:#3987e5;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}}
*{{box-sizing:border-box}}
body{{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:1.6rem;margin:0 0 .35rem}}
h2{{font-size:1.15rem;margin:2.5rem 0 .5rem}}
h3{{font-size:.85rem;margin:0 0 .4rem;color:var(--muted);font-weight:600}}
.sub{{color:var(--muted);margin:0 0 1.5rem;max-width:70ch}}
.note{{color:var(--muted);font-size:13px;margin:.3rem 0 1rem;max-width:78ch}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--border)}}
th{{color:var(--muted);font-weight:600}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.good{{color:var(--good);font-weight:600}} .bad{{color:var(--bad);font-weight:600}}
.scroll{{overflow-x:auto}}
.legend{{display:flex;gap:1rem;flex-wrap:wrap;margin:.4rem 0 1rem;font-size:13px;
  color:var(--muted)}}
.key{{display:inline-block;width:10px;height:10px;border-radius:2px;
  margin-right:.35rem;vertical-align:baseline}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  gap:1rem}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:.8rem}}
.chart{{width:100%;height:auto;display:block;overflow:visible}}
.grid{{stroke:var(--border);stroke-width:1}}
.tick{{fill:var(--muted);font-size:9px}}
.cursor{{stroke:var(--fg);stroke-width:1;opacity:.55}}
.card .chart{{cursor:crosshair}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.empty{{color:var(--muted);font-size:13px;margin:.6rem 0}}
.s1{{stroke:var(--s1)}} .s2{{stroke:var(--s2)}} .s3{{stroke:var(--s3)}}
.s4{{stroke:var(--s4)}} .s5{{stroke:var(--s5)}} .s6{{stroke:var(--s6)}}
.s7{{stroke:var(--s7)}} .s8{{stroke:var(--s8)}}
circle.s1{{fill:var(--s1)}} circle.s2{{fill:var(--s2)}} circle.s3{{fill:var(--s3)}}
circle.s4{{fill:var(--s4)}} circle.s5{{fill:var(--s5)}} circle.s6{{fill:var(--s6)}}
circle.s7{{fill:var(--s7)}} circle.s8{{fill:var(--s8)}}
.key.s1{{background:var(--s1)}} .key.s2{{background:var(--s2)}}
.key.s3{{background:var(--s3)}} .key.s4{{background:var(--s4)}}
.key.s5{{background:var(--s5)}} .key.s6{{background:var(--s6)}}
.key.s7{{background:var(--s7)}} .key.s8{{background:var(--s8)}}
.barsample{{margin:0 0 .9rem}}
.barrow{{display:flex;align-items:center;gap:.5rem;margin:.15rem 0;font-size:13px}}
.barlabel{{width:74px;color:var(--muted);flex:none}}
.bartrack{{flex:1;height:10px;background:var(--border);border-radius:2px;
  overflow:hidden}}
.bar{{display:block;height:100%;border-radius:2px}}
.bar.s1{{background:var(--s1)}} .bar.s2{{background:var(--s2)}}
.bar.s3{{background:var(--s3)}} .bar.s4{{background:var(--s4)}}
.barval{{width:76px;text-align:right;font-variant-numeric:tabular-nums;flex:none}}
.viewer{{margin-top:.2rem}}
.seek{{display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem}}
.seek input{{flex:1;accent-color:var(--accent)}}
.frameno{{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;
  width:72px;flex:none}}
.panes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  gap:.4rem}}
.panes figure{{margin:0}}
.panes figcaption{{font-size:11px;color:var(--muted);margin-bottom:.2rem}}
.panes img{{width:100%;height:auto;display:block;border-radius:4px;background:#000}}
.hint{{opacity:.65}}
.flip{{position:relative;display:block}}
.flip img.over{{position:absolute;inset:0;opacity:0;transition:opacity .08s}}
.flip:hover img.over{{opacity:1}}
.zoom{{cursor:zoom-in}}
#lightbox{{display:none;position:fixed;inset:0;z-index:50;overflow:auto;
  background:rgba(0,0,0,.93);cursor:zoom-out;padding:0}}
#lightbox.on{{display:block}}
/* Natural size, nearest neighbour: one differing pixel has to stay one crisp
   pixel when the frame is inspected at full resolution. */
#lb-stack{{position:relative;display:table;margin:0 auto}}
#lightbox img{{display:block;width:auto;max-width:none;image-rendering:pixelated}}
#lb-b{{position:absolute;inset:0;opacity:0}}
#lb-stack:hover #lb-b{{opacity:1}}
#lb-hint{{position:fixed;left:0;right:0;bottom:0;margin:0;padding:.5rem;
  text-align:center;font-size:13px;color:#cfcfd8;background:rgba(0,0,0,.65);
  pointer-events:none}}
</style>
<div id="lightbox">
<div id="lb-stack"><img id="lb-a" alt=""><img id="lb-b" alt=""></div>
<p id="lb-hint"></p>
</div>
<div class="wrap">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{body}
</div>
"""


def legend(drivers):
    return ('<div class="legend">' + ''.join(
        f'<span><span class="key s{i + 1}"></span>{html.escape(d)}</span>'
        for i, d in enumerate(drivers)) + '</div>')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='directory holding one subdirectory per driver')
    ap.add_argument('--ref', default='nvidia', help='reference driver')
    ap.add_argument('--drivers', nargs='+', default=None,
                    help='drivers, in the order they should be drawn')
    ap.add_argument('-o', '--out', default='perf.html')
    ap.add_argument('-t', '--tol', type=int, default=8,
                    help='per-channel tolerance for the frame diff (default 8)')
    ap.add_argument('--no-frames', action='store_true',
                    help='skip exporting frames; charts only, no viewer')
    ap.add_argument('--frame-width', type=int, default=560,
                    help='width of the inline frames (default 560); the '
                         'full resolution copies a click opens are native')
    ap.add_argument('--diff-gain', type=int, default=12,
                    help='gain on the difference images (default 12)')
    ap.add_argument('--recompute', action='store_true',
                    help='ignore the cached frame differences')
    ap.add_argument('--title', default='Driver sweep')
    args = ap.parse_args()

    drivers = args.drivers or sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d)) and not d.startswith('_'))
    if args.ref in drivers:
        drivers = [args.ref] + [d for d in drivers if d != args.ref]
    slot = {d: i + 1 for i, d in enumerate(drivers)}

    timing = {d: read_timing(os.path.join(args.root, d)) for d in drivers}
    gpu = {d: read_gpu(os.path.join(args.root, d)) for d in drivers}
    samples = samples_in(os.path.join(args.root, args.ref)) or sorted(
        {s for t in timing.values() for s in t})

    if not shutil.which('ffmpeg'):
        print('ffmpeg is required for the frame differences and the viewer')
        return 2

    print('computing frame differences (cached in _diffs.json)…')
    diffs = frame_diffs(args.root, args.ref, drivers, samples,
                        args.tol, args.recompute)

    img_rel = os.path.splitext(os.path.basename(args.out))[0] + '_frames'
    img_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), img_rel)
    if not args.no_frames:
        print('exporting frames…')
        export_frames(args.root, drivers, samples, img_dir, args.frame_width)
        print('exporting difference images…')
        export_diffs(args.root, args.ref, drivers, samples, img_dir,
                     args.frame_width, args.diff_gain)

    body = []

    # ---- cost ----
    body.append('<h2>Cost</h2>')
    body.append('<p class="note">Wall clock is what the run took; CPU is how '
                'much processor time it burned. Wall well under CPU means the '
                'driver spread across cores; wall tracking CPU means one core '
                'doing the work.</p>')
    body.append(legend(drivers))
    body.append('<div class="scroll"><table><thead><tr><th>Sample</th>' +
                ''.join(f'<th class="num">{html.escape(d)} wall</th>'
                        f'<th class="num">cpu</th>' for d in drivers) +
                '<th class="num">exit</th></tr></thead><tbody>')

    order = sorted(samples,
                   key=lambda s: -max((timing[d].get(s, {}).get('wall', 0)
                                       for d in drivers), default=0))
    totals = {d: [0.0, 0.0] for d in drivers}
    for s in order:
        cells = ''
        bad = False
        for d in drivers:
            t = timing[d].get(s)
            if not t:
                cells += '<td class="num">-</td><td class="num">-</td>'
                continue
            totals[d][0] += t['wall']
            totals[d][1] += t['cpu']
            bad |= t['exit'] != 0
            cells += (f'<td class="num">{t["wall"]:.2f}</td>'
                      f'<td class="num">{t["cpu"]:.2f}</td>')
        mark = ('<td class="num bad">nonzero</td>' if bad
                else '<td class="num good">0</td>')
        body.append(f'<tr><td>{html.escape(s)}</td>{cells}{mark}</tr>')
    body.append('<tr><td><b>total</b></td>' + ''.join(
        f'<td class="num"><b>{totals[d][0]:.1f}</b></td>'
        f'<td class="num"><b>{totals[d][1]:.1f}</b></td>' for d in drivers) +
        '<td></td></tr>')
    body.append('</tbody></table></div>')

    body.append('<h2>Wall clock by sample</h2>')
    body.append(legend(drivers))
    body.append('<div class="card">' + bar_rows(
        [(s, [(d, slot[d], timing[d].get(s, {}).get('wall', 0.0))
              for d in drivers]) for s in order], slot, '{:.2f}s') + '</div>')

    # ---- gpu ----
    if any(gpu.values()):
        body.append('<h2>GPU over the run</h2>')
        body.append('<p class="note">Sampled at 2 Hz for the whole pass. '
                    'Utilisation and memory are separate charts because they '
                    'are different measures — never two scales on one axis.</p>')
        body.append(legend(drivers))
        util = [(d, slot[d], [(t, u) for t, u, _ in gpu[d]])
                for d in drivers if gpu[d]]
        mem = [(d, slot[d], [(t, m) for t, _, m in gpu[d]])
               for d in drivers if gpu[d]]
        u_svg, _ = line_chart(util, width=560, height=170, fmt='{:,.0f}%')
        m_svg, _ = line_chart(mem, width=560, height=170, fmt='{:,.0f}')
        body.append('<div class="grid2">'
                    f'<div class="card"><h3>Utilisation, % (x = seconds)</h3>{u_svg}</div>'
                    f'<div class="card"><h3>Memory used, MiB (x = seconds)</h3>{m_svg}</div>'
                    '</div>')

    # ---- correctness over the animation ----
    body.append('<h2>Differences over the animation</h2>')
    body.append(f'<p class="note">Differing pixels against '
                f'<b>{html.escape(args.ref)}</b> at tolerance {args.tol}/255, '
                f'per frame. Each chart is scaled to its own sample, because a '
                f'shared axis would flatten every small one. What matters is '
                f'the shape: flat means the difference is a standing one, a '
                f'climb means something drifts as the scene animates.</p>'
                f'<p class="note">Click a chart to send the frames under it to '
                f'that frame. Hovering a driver flips it to '
                f'{html.escape(args.ref)}; clicking any pane opens it at full '
                f'resolution, where the same hover works. The difference panes '
                f'are the per-channel absolute difference brightened '
                f'{args.diff_gain}&times;, since a few levels out of 255 is '
                f'invisible otherwise — at that gain anything '
                f'{255 // args.diff_gain} levels or more apart saturates, so '
                f'they show where the drivers disagree rather than by how '
                f'much.</p>')
    tested = [d for d in drivers if d != args.ref]
    body.append(legend(tested) if tested else '')

    cards = []
    for s in order:
        series = [(d, slot[d], [(i, v) for i, v in diffs.get(d, {}).get(s, [])])
                  for d in tested]
        svg, _ = line_chart(series)
        first = worst = None
        worst_at = 0
        nframes = 0
        for _, _, ps in series:
            nframes = max(nframes, len(ps))
            for i, v in ps:
                if i == 0 and first is None:
                    first = v
                if worst is None or v > worst:
                    worst, worst_at = v, i
        cap = ('no frames' if worst is None
               else f'frame 0: {first:,} · worst: {worst:,} at frame {worst_at}')

        # The viewer: a slider over the frames, and the reference beside each
        # driver under test at whatever frame is selected. It opens on the worst
        # frame, because that is the one the chart was pointing at.
        viewer = ''
        if not args.no_frames and nframes:
            panes = []
            n4 = f'{worst_at + 1:04d}'
            ref_small = f'{img_rel}/{args.ref}/{s}/{n4}.png'
            ref_full = f'{img_rel}/{args.ref}/{s}/full/{n4}.png'
            for d in drivers:
                src = f'{img_rel}/{d}/{s}/{n4}.png'
                full = f'{img_rel}/{d}/{s}/full/{n4}.png'
                if d == args.ref:
                    panes.append(
                        f'<figure><figcaption>{html.escape(d)}</figcaption>'
                        f'<img class="zoom" loading="lazy" '
                        f'data-driver="{html.escape(d)}" data-full="{full}" '
                        f'data-hint="{html.escape(s)} &mdash; {html.escape(d)}" '
                        f'src="{src}" alt=""></figure>')
                    continue

                # The driver, flipping to the reference on hover — the same
                # gesture the single frame gallery uses, and the one that
                # answers "what should this look like".
                panes.append(
                    f'<figure><figcaption>{html.escape(d)} '
                    f'<span class="hint">hover: {html.escape(args.ref)}</span>'
                    f'</figcaption>'
                    f'<span class="flip zoom" data-full="{full}" '
                    f'data-ref="{ref_full}" '
                    f'data-hint="{html.escape(s)} &mdash; {html.escape(d)}, '
                    f'hover to flip to {html.escape(args.ref)}">'
                    f'<img loading="lazy" data-driver="{html.escape(d)}" '
                    f'src="{src}" alt="">'
                    f'<img class="over" loading="lazy" '
                    f'data-driver="{html.escape(args.ref)}" '
                    f'src="{ref_small}" alt=""></span></figure>')
                # And the difference against the reference, on its own, since
                # wanting to look at one is not the same as wanting to A/B.
                dsrc = f'{img_rel}/_diff/{d}/{s}/{n4}.png'
                dfull = f'{img_rel}/_diff/{d}/{s}/full/{n4}.png'
                panes.append(
                    f'<figure><figcaption>|{html.escape(d)} &minus; '
                    f'{html.escape(args.ref)}|, brightness &times;{args.diff_gain}'
                    f'<span class="hint" title="The per-channel absolute '
                    f'difference, multiplied by {args.diff_gain} so it is '
                    f'visible at all. Anything differing by '
                    f'{255 // args.diff_gain} levels or more saturates, so this '
                    f'shows where they disagree, not by how much — the chart '
                    f'above carries the magnitude."> &#9432;</span>'
                    f'</figcaption>'
                    f'<img class="zoom" loading="lazy" '
                    f'data-diff="{html.escape(d)}" data-full="{dfull}" '
                    f'data-hint="{html.escape(s)} &mdash; {html.escape(d)} minus '
                    f'{html.escape(args.ref)}, gain {args.diff_gain}" '
                    f'src="{dsrc}" alt=""></figure>')

            viewer = (
                f'<div class="viewer" data-sample="{html.escape(s)}" '
                f'data-base="{img_rel}">'
                f'<div class="seek"><input type="range" min="0" '
                f'max="{nframes - 1}" value="{worst_at}" '
                f'aria-label="frame"><span class="frameno">frame '
                f'{worst_at}</span></div>'
                f'<div class="panes">{"".join(panes)}</div></div>')

        cards.append(f'<div class="card"><h3>{html.escape(s)}</h3>{svg}'
                     f'<p class="note" style="margin:.3rem 0 .5rem">{cap}</p>'
                     f'{viewer}</div>')
    body.append('<div class="grid2">' + ''.join(cards) + '</div>')

    body.append(r"""<script>
// Seeking swaps the src of every pane, the driver frames and the difference
// images alike; they are all on disk, one file per driver per frame.
document.querySelectorAll('.viewer').forEach(function (v) {
  var slider = v.querySelector('input[type=range]');
  var label  = v.querySelector('.frameno');
  var base   = v.dataset.base, sample = v.dataset.sample;
  var chart  = v.closest('.card').querySelector('svg.chart');
  var cursor = chart && chart.querySelector('.cursor');

  // The chart is the index: click where the difference spikes and the panes
  // below jump to that frame.
  function place() {
    if (!cursor) return;
    var x0 = +chart.dataset.x0, x1 = +chart.dataset.x1;
    var pl = +chart.dataset.pl, pr = +chart.dataset.pr, w = +chart.dataset.w;
    var f = x1 > x0 ? (+slider.value - x0) / (x1 - x0) : 0;
    var x = pl + f * (w - pl - pr);
    cursor.setAttribute('x1', x);
    cursor.setAttribute('x2', x);
    cursor.style.display = '';
  }

  if (chart) {
    chart.addEventListener('click', function (e) {
      var r = chart.getBoundingClientRect();
      var x0 = +chart.dataset.x0, x1 = +chart.dataset.x1;
      var pl = +chart.dataset.pl, pr = +chart.dataset.pr, w = +chart.dataset.w;
      // viewBox starts at 0 and the aspect is not preserved, so the pointer
      // maps straight through as a fraction of the rendered width.
      var sx = (e.clientX - r.left) / r.width * w;
      var f = (sx - pl) / (w - pl - pr);
      var frame = Math.round(x0 + f * (x1 - x0));
      frame = Math.max(+slider.min, Math.min(+slider.max, frame));
      slider.value = frame;
      slider.dispatchEvent(new Event('input'));
    });
  }

  slider.addEventListener('input', function () {
    place();
    var n = String(+slider.value + 1).padStart(4, '0');
    label.textContent = 'frame ' + slider.value;
    v.querySelectorAll('img[data-driver]').forEach(function (img) {
      img.src = base + '/' + img.dataset.driver + '/' + sample + '/' + n + '.png';
    });
    v.querySelectorAll('img[data-diff]').forEach(function (img) {
      img.src = base + '/_diff/' + img.dataset.diff + '/' + sample + '/' + n + '.png';
    });
    v.querySelectorAll('[data-full]').forEach(function (el) {
      el.dataset.full = el.dataset.full.replace(/\/[0-9]{4}\.png$/, '/' + n + '.png');
      if (el.dataset.ref) {
        el.dataset.ref = el.dataset.ref.replace(/\/[0-9]{4}\.png$/, '/' + n + '.png');
      }
    });
  });
  place();
});

// Click any pane for the full resolution PNG, and keep the hover gesture
// there: the reference is layered over it, so the same movement that A/Bs a
// thumbnail A/Bs it at full size.
var lb = document.getElementById('lightbox');
document.addEventListener('click', function (e) {
  var z = e.target.closest('.zoom');
  if (z) {
    var b = document.getElementById('lb-b');
    document.getElementById('lb-a').src = z.dataset.full;
    if (z.dataset.ref) {
      b.src = z.dataset.ref;
      b.style.display = '';
      document.getElementById('lb-hint').textContent = z.dataset.hint || '';
    } else {
      b.removeAttribute('src');
      b.style.display = 'none';
      document.getElementById('lb-hint').textContent = z.dataset.hint || '';
    }
    lb.classList.add('on');
    return;
  }
  if (e.target.closest('#lightbox')) lb.classList.remove('on');
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') lb.classList.remove('on');
});
</script>""")

    subtitle = (f'{len(samples)} samples across {len(drivers)} drivers, every '
                f'frame compared against {html.escape(args.ref)} at tolerance '
                f'{args.tol}/255. Cost first, then what the GPU was doing, then '
                f'whether the picture held up for the whole run — and for any '
                f'frame that did not, what it looked like.')

    open(args.out, 'w').write(PAGE.format(
        title=html.escape(args.title), subtitle=subtitle, body='\n'.join(body)))
    print(f'\n-> {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
