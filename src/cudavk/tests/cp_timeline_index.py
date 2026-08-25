#!/usr/bin/env python3
"""Build ~/timelines/index.html: a row per iteration, both captures linked.

Reads what is on disk rather than a hand-kept list, the way cp_iter_report.py
page rebuilds iterations.json: an iteration that was rendered is visible without
anyone remembering a second command.

Per capture it reports the paired-submit median (one frame is two
vkQueueSubmit events, median of diff(ts[::2])[50:]), the frame count and the
stored PNG count, all recomputed from the timeline's own files. The iteration's
description comes from ~/git/Vulkan/build/iter/LABEL/iteration.json when that
iteration was also run through cp_iterate.sh.
"""
import html, json, os, statistics, glob

ROOT = os.path.expanduser("~/timelines")
ITERROOT = os.path.expanduser("~/git/Vulkan/build/iter")


def median_ms(submits):
    try:
        ts = [int(l.split()[0]) for l in open(submits) if l.strip()]
    except OSError:
        return None, 0
    if len(ts) < 104:
        return None, len(ts)
    pairs = ts[::2]
    d = [(b - a) / 1e6 for a, b in zip(pairs, pairs[1:])][50:]
    return (statistics.median(d) if d else None), len(ts)


def capture_row(path):
    med, submits = median_ms(os.path.join(path, "timing", "submits.txt"))
    pngs = len(glob.glob(os.path.join(path, "frames", "*.png")))
    return {"exists": os.path.isdir(path), "median": med,
            "submits": submits, "frames": submits // 2, "pngs": pngs}


def desc_for(label):
    p = os.path.join(ITERROOT, label, "iteration.json")
    if not os.path.exists(p):
        return "", ""
    try:
        j = json.load(open(p))
    except (OSError, ValueError):
        return "", ""
    return j.get("desc", "") or "", (j.get("commit") or "")[:11]


rows = []
for label in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, label)
    if not os.path.isdir(d) or label.startswith("."):
        continue
    old, cross = capture_row(os.path.join(d, "old")), capture_row(os.path.join(d, "cross"))
    if not (old["exists"] or cross["exists"]):
        continue
    desc, commit = desc_for(label)
    rows.append((label, old, cross, desc, commit))
rows.sort(key=lambda r: os.path.getmtime(os.path.join(ROOT, r[0])), reverse=True)


def cell(label, name, c):
    if not c["exists"]:
        return '<td class="none">—</td>'
    med = f'{c["median"]:.2f}' if c["median"] else "—"
    return (f'<td><a href="{html.escape(label)}/{name}/index.html">{med} ms</a>'
            f'<span class="sub">{c["frames"]} frames · {c["pngs"]} png</span></td>')


out = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>cudavk frame timelines</title><style>
body{font:14px/1.45 system-ui,sans-serif;margin:2rem;color:#111;background:#fafafa}
h1{font-size:1.3rem;margin:0 0 .2rem} p.lead{color:#555;margin:0 0 1.4rem}
table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 2px #0001}
th,td{padding:.55rem .7rem;border-bottom:1px solid #eceff1;text-align:left;vertical-align:top}
th{background:#f4f6f8;font-weight:600;font-size:.85rem;letter-spacing:.02em}
td.none{color:#bbb} .sub{display:block;color:#777;font-size:.78rem}
a{color:#0b6bcb;text-decoration:none} a:hover{text-decoration:underline}
code{background:#f4f6f8;padding:.05rem .3rem;border-radius:3px;font-size:.85em}
td.desc{color:#444;max-width:46rem}
</style></head><body>
<h1>cudavk frame timelines</h1>
<p class="lead">One row per iteration, one timeline per capture. The number is the
paired-submit median — a frame is two <code>vkQueueSubmit</code> events, median of
<code>diff(ts[::2])[50:]</code> — recomputed from each run's own submit log.
Labels match <code>cp_iterate.sh</code>, so a row here and a row in
<a href="../git/Vulkan/build/iter/iterations.html">iterations.html</a> are the same iteration.</p>
<table><thead><tr><th>iteration</th><th>old capture</th><th>Crossroads</th>
<th>commit</th><th class="desc">what it was</th></tr></thead><tbody>"""]
for label, old, cross, desc, commit in rows:
    out.append("<tr><td><b>%s</b></td>%s%s<td><code>%s</code></td><td class=\"desc\">%s</td></tr>"
               % (html.escape(label), cell(label, "old", old), cell(label, "cross", cross),
                  html.escape(commit or "—"), html.escape(desc or "")))
out.append("</tbody></table></body></html>")
open(os.path.join(ROOT, "index.html"), "w").write("\n".join(out))
print("wrote %s with %d iteration(s)" % (os.path.join(ROOT, "index.html"), len(rows)))
