#!/usr/bin/env python3
"""
Census of per-batch kernel chains in an nsys export of the default renderer,
and a decomposition of the main stream's wall into busy, launch gaps, joins
and the frame boundary.

    cp_chain_census.py /tmp/trace.sqlite

Why it exists: the small-batch fused-kernel idea of 2026-09-09 needed to know
how much main-stream wall the small batches hold before anything was built.
The answer was small (see notes/SMALL_BATCH_CENSUS_2026-09-09.md), and the
script is kept so the next such question costs one command.

Conventions it depends on:

  * Every generated shader is a kernel named `main`. A vertex shader is a
    `main` whose next kernel on the same stream is a clip-family kernel; the
    fixed-grid fragment shader is everything else. Grid-size heuristics are
    wrong here: fragment grids are 4096 blocks on the fused path but sized to
    the pixel list on others.
  * A chain is one vertex shader through to the next on the same stream —
    one batch's VS, clip+stage1, stage 2, stage 3, compaction, FS and
    writeback, plus any A-buffer or peel tail. Chains are cut at frame ends.
  * One frame is two vkQueueSubmit and each submit ends in the 3,686,400-byte
    readback copy, so a frame is every other readback (CLAUDE.md, Measuring).
  * Traced numbers are inflated by CUPTI's per-launch host cost; use them for
    shares and populations, not as frame times.
"""
import bisect
import collections
import sqlite3
import statistics
import sys

CLIP = {'cp_clip_rast_fused', 'cp_clip_rast_fused_abuf', 'cp_clip_triangles',
        'cp_vertex_fetch'}
READBACK_BYTES = 3686400


def load(path):
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    k = [dict(s=r[0], e=r[1], st=r[2], n=r[3], gx=r[4], bx=r[5], kind='K')
         for r in c.execute(
             "select k.start,k.end,k.streamId,s.value,k.gridX,k.blockX "
             "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s "
             "on s.id=k.shortName order by k.start")]
    ms = [dict(s=r[0], e=r[1], st=r[2], n='memset', b=r[3], kind='S')
          for r in c.execute("select start,end,streamId,bytes from "
                             "CUPTI_ACTIVITY_KIND_MEMSET order by start")]
    kinds = {1: 'HtoD', 2: 'DtoH', 8: 'DtoD'}
    mc = [dict(s=r[0], e=r[1], st=r[2], n=kinds.get(r[4], f'cp{r[4]}'),
               b=r[3], kind='C')
          for r in c.execute("select start,end,streamId,bytes,copyKind from "
                             "CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
    rb = [r[0] for r in c.execute(
        "select end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes=? "
        "order by start", (READBACK_BYTES,))]
    return k, ms, mc, rb


def pair_frames(rb):
    """Every other readback; the short submit interval comes first."""
    iv = [rb[i + 1] - rb[i] for i in range(len(rb) - 1)]
    off = 0 if len(iv) > 1 and iv[0] < iv[1] else 1
    frames = [(rb[i], rb[i + 2]) for i in range(off, len(rb) - 2, 2)]
    med = statistics.median(b - a for a, b in frames)
    return [f for f in frames if f[1] - f[0] < 3 * med], med


def main():
    path = sys.argv[1]
    K, MS, MC, RB = load(path)
    frames, medf = pair_frames(RB)
    NF = len(frames)
    starts = [a for a, b in frames]

    def fidx(t):
        i = bisect.bisect_right(starts, t) - 1
        return i if 0 <= i < NF and t < frames[i][1] else -1

    ops = sorted(K + MS + MC, key=lambda o: o['s'])
    bys = collections.defaultdict(list)
    for o in ops:
        bys[o['st']].append(o)
    main_st = max(bys, key=lambda s: sum(1 for o in bys[s] if o['kind'] == 'K'))
    print(f"{path}: {NF} frames, median {medf / 1e6:.3f} ms, mean "
          f"{sum(b - a for a, b in frames) / NF / 1e6:.3f} ms (traced); "
          f"main stream {main_st}, {len(bys)} streams")

    # ---- chains -----------------------------------------------------------
    rows = []
    for st, os_ in bys.items():
        ks = [o for o in os_ if o['kind'] == 'K']
        vs = [i for i in range(len(ks) - 1)
              if ks[i]['n'] == 'main' and ks[i + 1]['n'] in CLIP]
        for a, b in zip(vs, vs[1:] + [len(ks)]):
            seg = ks[a:b]
            f = fidx(seg[0]['s'])
            if f < 0:
                continue
            seg = [x for x in seg if fidx(x['s']) == f]
            names = [x['n'] for x in seg]
            tris = next((x['gx'] * x['bx'] for x in seg
                         if x['n'] in CLIP and x['n'] != 'cp_vertex_fetch'), 0)
            if any('abuf' in n for n in names):
                cls = 'blended'
            elif any(n in ('cp_fs_compact', 'cp_fs_writeback') for n in names):
                cls = 'shaded'
            else:
                cls = 'raster-only'
            rows.append(dict(side=st != main_st, verts=seg[0]['gx'] * seg[0]['bx'],
                             tris=tris, cls=cls, n=len(seg),
                             wall=(seg[-1]['e'] - seg[0]['s']) / 1e3,
                             busy=sum(x['e'] - x['s'] for x in seg) / 1e3))

    def med(x):
        return statistics.median(x) if x else 0

    def summ(sel, label):
        if not sel:
            return
        print(f"  {label:34s} {len(sel) / NF:6.2f}/fr  launches "
              f"{sum(r['n'] for r in sel) / NF:6.1f}/fr  wall "
              f"{sum(r['wall'] for r in sel) / NF / 1e3:6.3f}  busy "
              f"{sum(r['busy'] for r in sel) / NF / 1e3:6.3f} ms/fr   med: "
              f"launches {med([r['n'] for r in sel]):3.0f} verts "
              f"{med([r['verts'] for r in sel]):7.0f} tris "
              f"{med([r['tris'] for r in sel]):6.0f} wall "
              f"{med([r['wall'] for r in sel]):5.0f} us")

    print("chains (a chain is one VS through to the next on its stream):")
    for side in (False, True):
        for cls in ('shaded', 'raster-only', 'blended'):
            for lo, hi, nm in ((0, 4097, '<=4096 verts'), (4097, 1 << 40, '>4096 verts')):
                summ([r for r in rows if r['side'] == side and r['cls'] == cls
                      and lo <= r['verts'] < hi],
                     f"{'side' if side else 'MAIN'} {cls} {nm}")

    # ---- main-stream wall ----------------------------------------------
    mk = bys[main_st]
    U = []
    for s, e in sorted((o['s'], o['e']) for st in bys if st != main_st
                       for o in bys[st]):
        if U and s <= U[-1][1]:
            U[-1][1] = max(U[-1][1], e)
        else:
            U.append([s, e])
    SS = [a for a, b in U]

    def side_busy_in(a, b):
        i = max(0, bisect.bisect_right(SS, a) - 1)
        t = 0
        while i < len(U) and U[i][0] < b:
            s, e = U[i]
            t += max(0, min(e, b) - max(s, a))
            i += 1
        return t

    def lab(o):
        if o['kind'] == 'K':
            return o['n']
        return o['n'] + ('<4K' if o['b'] < 4096 else '>=4K')

    tot = collections.Counter()
    cnt = collections.Counter()
    edges = collections.Counter()
    MKS = [o['s'] for o in mk]
    for (a, b) in frames:
        i = bisect.bisect_left(MKS, a)
        t = a
        prev = None
        while i < len(mk) and mk[i]['s'] < b:
            o = mk[i]
            if o['s'] > t:
                g = o['s'] - t
                if g > 200e3:
                    key = 'gap >200us (frame boundary, host waits)'
                elif side_busy_in(t, o['s']) > 0.5 * g:
                    key = 'gap <=200us, side streams busy (joins)'
                else:
                    key = 'gap <=200us, everything idle (launch gaps)'
                    cnt['launch gaps'] += 1
                tot[key] += g
                if prev:
                    edges[lab(prev) + ' > ' + lab(o)] += g
            tot['busy ' + {'K': 'kernel', 'C': 'memcpy', 'S': 'memset'}[o['kind']]] += \
                max(0, min(o['e'], b) - max(o['s'], a))
            cnt[o['kind']] += 1
            t = max(t, o['e'])
            prev = o
            i += 1
        if b > t:
            tot['gap >200us (frame boundary, host waits)' if b - t > 200e3
                else 'gap <=200us, everything idle (launch gaps)'] += b - t
        tot['frame'] += b - a
    print("main-stream wall (ms/frame, traced):")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:44s} {v / NF / 1e6:7.3f}")
    print(f"  main-stream ops/frame: kernels {cnt['K'] / NF:.0f}, memcpy "
          f"{cnt['C'] / NF:.0f}, memset {cnt['S'] / NF:.0f}; launch gaps "
          f"{cnt['launch gaps'] / NF:.0f}/frame")
    print("largest gap edges (ms/frame):")
    for k, v in edges.most_common(8):
        print(f"  {v / NF / 1e6:7.3f}  {k}")


if __name__ == '__main__':
    main()
