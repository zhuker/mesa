#!/usr/bin/env python3
"""Encode a compiled-harness frame dump and build its timeline page.

The harness (favorite2/favorite3 tocpp builds) writes frame_NNNNNN.bin plus a
.meta of "width height bytes vkformat" per frame. This turns those into PNGs
and one page, the way cp_gfxr_frames.py does for a gfxr replay -- same idea,
different dump format, so the two captures that actually matter get the same
artefact the samples already had.

Needs the mesa venv (numpy, PIL):  $MESA/venv/bin/python3

  cp_harness_frames.py png  RAWDIR OUTDIR [--thumb 320]
  cp_harness_frames.py page OUTDIR --submits TS --capture NAME -o index.html
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
    os.makedirs(os.path.join(args.outdir, "thumb"), exist_ok=True)
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
        t = args.thumb
        img.resize((t, max(1, h * t // w)), Image.BILINEAR).save(
            os.path.join(args.outdir, "thumb", stem + ".jpg"), quality=80)
        written += 1
    print("encoded %d frames" % written)


def frame_times(submits):
    """Paired-submit frame times in ms: a frame is two vkQueueSubmit events."""
    ts = [int(l.split()[0]) for l in open(submits)
          if len(l.split()) == 2 and l.split()[0].isdigit()]
    return [(ts[i + 2] - ts[i]) / 1e6 for i in range(0, len(ts) - 2, 2)]


def cmd_page(args):
    times = frame_times(args.submits)
    frames = sorted(f[:-4] for f in os.listdir(args.outdir) if f.endswith(".png"))
    n = min(len(times), len(frames))
    if not n:
        sys.exit("no timing/frame pairs")
    real, heavy = args.real_start, args.heavy_start
    band = lambda i: "load" if i < real else ("heavy" if heavy and i >= heavy else "real")
    rows = [{"i": i, "ms": round(times[i], 3), "band": band(i), "f": frames[i]}
            for i in range(n)]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    stats = {
        "capture": args.capture, "frames": n,
        "median_all": round(med(times[:n]), 4),
        "median_real": round(med([r["ms"] for r in rows if r["band"] != "load"]), 4),
        "median_heavy": round(med([r["ms"] for r in rows if r["band"] == "heavy"]), 4),
        "real_start": real, "heavy_start": heavy,
    }
    json.dump({"stats": stats, "rows": rows},
              open(os.path.join(args.outdir, "frames.json"), "w"))
    html = r"""<!doctype html><meta charset=utf-8><title>%(cap)s timeline</title>
<style>
body{background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:16px;margin:0 0 4px} .sub{color:#888;margin-bottom:12px}
.bar{display:flex;gap:1px;height:44px;align-items:flex-end;margin:10px 0 16px}
.bar div{flex:1 1 0;min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:8px}
figure{margin:0;background:#1a1a1a;border-radius:4px;padding:5px}
figure img{width:100%%;display:block;border-radius:2px}
figcaption{font-size:11px;color:#aaa;padding-top:3px;display:flex;justify-content:space-between}
.load{color:#777} .real{color:#7ec8e3} .heavy{color:#e3a17e}
.k{display:inline-block;margin-right:14px} .k b{color:#fff}
input{background:#222;color:#ddd;border:1px solid #444;border-radius:3px;padding:4px 7px}
</style>
<h1>%(cap)s &mdash; every frame</h1>
<div class=sub>
<span class=k>frames <b>%(n)d</b></span>
<span class=k>median all <b>%(ma).3f ms</b></span>
<span class=k>median real (%(rs)d+) <b>%(mr).3f ms</b></span>
<span class=k>median heavy (%(hs)s) <b>%(mh).3f ms</b></span>
<span class=k><span class=load>loading</span> / <span class=real>real</span> / <span class=heavy>heavy</span></span>
</div>
<div>show <input id=f size=28 placeholder="all | real | heavy | 1388-1420"> &nbsp;<span id=cnt></span></div>
<div class=bar id=bar></div>
<div class=grid id=g></div>
<script>
const D=%(data)s, rows=D.rows, mx=Math.max(...rows.map(r=>r.ms));
const col={load:'#555',real:'#7ec8e3',heavy:'#e3a17e'};
document.getElementById('bar').innerHTML=rows.map(r=>
  `<div title="frame ${r.i}: ${r.ms} ms" style="height:${Math.max(2,r.ms/mx*44)}px;background:${col[r.band]}"></div>`).join('');
function sel(q){q=(q||'').trim();
  if(!q||q=='all')return rows;
  if(col[q])return rows.filter(r=>r.band==q);
  const m=q.match(/^(\d+)\s*-\s*(\d+)$/); if(m)return rows.filter(r=>r.i>=+m[1]&&r.i<=+m[2]);
  const k=+q; return isNaN(k)?rows:rows.filter(r=>r.i==k);}
function draw(){const s=sel(document.getElementById('f').value).slice(0,900);
  document.getElementById('cnt').textContent=s.length+' shown';
  document.getElementById('g').innerHTML=s.map(r=>
   `<figure><a href="${r.f}.png" target=_blank><img loading=lazy src="thumb/${r.f}.jpg"></a>`+
   `<figcaption><span class=${r.band}>#${r.i}</span><span>${r.ms.toFixed(2)} ms</span></figcaption></figure>`).join('');}
document.getElementById('f').addEventListener('input',draw); draw();
</script>""" % {"cap": args.capture, "n": stats["frames"], "ma": stats["median_all"],
                "mr": stats["median_real"], "mh": stats["median_heavy"],
                "rs": real, "hs": str(heavy) + "+" if heavy else "n/a",
                "data": json.dumps({"rows": rows})}
    open(args.output, "w").write(html)
    print("wrote %s  (%d frames, median real %.3f ms)"
          % (args.output, stats["frames"], stats["median_real"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("png"); a.add_argument("rawdir"); a.add_argument("outdir")
    a.add_argument("--thumb", type=int, default=320); a.set_defaults(fn=cmd_png)
    b = sub.add_parser("page"); b.add_argument("outdir")
    b.add_argument("--submits", required=True); b.add_argument("--capture", required=True)
    b.add_argument("--real-start", type=int, default=0)
    b.add_argument("--heavy-start", type=int, default=0)
    b.add_argument("-o", "--output", required=True); b.set_defaults(fn=cmd_page)
    args = p.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
