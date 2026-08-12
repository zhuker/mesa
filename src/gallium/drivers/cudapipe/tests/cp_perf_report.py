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

Per-frame differences are the expensive part — every frame of every sample
against the reference, in Python. They are cached in _diffs.json next to the
run, keyed by driver, sample and stride, so re-running to change the layout
costs nothing. --stride compares every Nth frame; --recompute discards the
cache.

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

# Optional: the frame diff is the whole cost of this script, and numpy makes it
# roughly a hundred times faster. Everything still runs without it, just slowly,
# so the script keeps the stdlib-only guarantee the other tools here have.
try:
    import numpy as _np
except ImportError:
    _np = None

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


def differing(a, b, w, h, ch_a, ch_b, tol):
    """Pixels whose worst channel differs by more than tol."""
    if _np is not None:
        pa = _np.frombuffer(b''.join(a), dtype=_np.uint8)
        pb = _np.frombuffer(b''.join(b), dtype=_np.uint8)
        pa = pa.reshape(h, -1)[:, :w * ch_a].reshape(h, w, ch_a)[:, :, :3]
        pb = pb.reshape(h, -1)[:, :w * ch_b].reshape(h, w, ch_b)[:, :, :3]
        d = _np.abs(pa.astype(_np.int16) - pb.astype(_np.int16)).max(axis=2)
        return int((d > tol).sum())

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


def frame_diffs(root, ref, drivers, samples, tol, stride, recompute):
    """{driver: {sample: [(frame_index, differing_pixels), ...]}}, cached."""
    cache_path = os.path.join(root, '_diffs.json')
    key = f'tol{tol}.stride{stride}'
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
        drv_dir = os.path.join(root, drv)
        for sample in samples:
            if sample in cache[key][drv]:
                continue
            rf, tf = frames_of(ref_dir, sample), frames_of(drv_dir, sample)
            if not rf or not tf:
                cache[key][drv][sample] = []
                continue
            series = []
            for i in range(0, min(len(rf), len(tf)), stride):
                rw, rh, rch, ri = read_image(os.path.join(ref_dir, sample, rf[i]))
                tw, th, tch, ti = read_image(os.path.join(drv_dir, sample, tf[i]))
                if (rw, rh) != (tw, th):
                    series = []
                    break
                series.append([i, differing(ri, ti, rw, rh, rch, tch, tol)])
            cache[key][drv][sample] = series
            print(f'  {drv}/{sample}: {len(series)} frames', flush=True)

    try:
        json.dump(cache, open(cache_path, 'w'))
    except OSError:
        pass
    return cache[key]


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

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
           f'preserveAspectRatio="none" role="img">']

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

PAGE = """<title>{title}</title>
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
</style>
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
    ap.add_argument('--stride', type=int, default=1,
                    help='compare every Nth frame (default every frame)')
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

    print('computing frame differences (cached in _diffs.json)%s…'
          % ('' if _np is not None else ', without numpy — this will be slow'))
    diffs = frame_diffs(args.root, args.ref, drivers, samples,
                        args.tol, args.stride, args.recompute)

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
                f'climb means something drifts as the scene animates.</p>')
    tested = [d for d in drivers if d != args.ref]
    body.append(legend(tested) if tested else '')

    cards = []
    for s in order:
        series = [(d, slot[d], [(i, v) for i, v in diffs.get(d, {}).get(s, [])])
                  for d in tested]
        svg, _ = line_chart(series)
        first = worst = None
        for _, _, ps in series:
            for i, v in ps:
                if i == 0 and first is None:
                    first = v
                worst = v if worst is None else max(worst, v)
        cap = ('no frames' if worst is None
               else f'frame 0: {first:,} · worst: {worst:,}')
        cards.append(f'<div class="card"><h3>{html.escape(s)}</h3>{svg}'
                     f'<p class="note" style="margin:.3rem 0 0">{cap}</p></div>')
    body.append('<div class="grid2">' + ''.join(cards) + '</div>')

    stride_note = ('every frame' if args.stride == 1
                   else f'every {args.stride}th frame')
    subtitle = (f'{len(samples)} samples across {len(drivers)} drivers, '
                f'{stride_note} compared against {html.escape(args.ref)} at '
                f'tolerance {args.tol}/255. Cost first, then what the GPU was '
                f'doing, then whether the picture held up for the whole run.')

    open(args.out, 'w').write(PAGE.format(
        title=html.escape(args.title), subtitle=subtitle, body='\n'.join(body)))
    print(f'\n-> {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
