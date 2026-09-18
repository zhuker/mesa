#!/usr/bin/env python3
"""Read-only analysis of an nsys sqlite export: per-kernel duration vs threads,
min-duration floors, VS/FS split of 'main' by same-stream successor grammar.
Usage: trace_analysis.py <sqlite> <win_start_ns> <win_end_ns> <frames> <outdir>
"""
import sqlite3, sys, json, statistics, collections, os
path, w0, w1, frames, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
os.makedirs(outdir, exist_ok=True)
c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
c.row_factory = sqlite3.Row

SQL_KERNELS = f"""
SELECT k.start, k.end, k.streamId, k.correlationId, s.value AS name,
       k.gridX*k.gridY*k.gridZ AS blocks, k.blockX*k.blockY*k.blockZ AS tpb,
       k.registersPerThread AS regs
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
WHERE k.start >= {w0} AND k.start < {w1}
ORDER BY k.streamId, k.start
"""
rows = [dict(r) for r in c.execute(SQL_KERNELS)]
print('kernels in window', len(rows), 'per frame', len(rows)/frames)

# classify 'main' by same-stream immediate successor (fail-closed grammar from DEAD_ENDS 35)
VS_SUCC = {'cp_clip_triangles', 'cp_clip_rast_fused', 'cp_clip_rast_fused_abuf'}
FS_SUCC = {'cp_fs_writeback', 'cp_abuf_scatter_colors', 'cp_abuf_composite'}
by_stream = collections.defaultdict(list)
for r in rows:
    by_stream[r['streamId']].append(r)
for st, lst in by_stream.items():
    for i, r in enumerate(lst):
        r['succ'] = lst[i+1]['name'] if i+1 < len(lst) else None
        r['pred'] = lst[i-1]['name'] if i > 0 else None
for r in rows:
    if r['name'] == 'main':
        if r['succ'] in VS_SUCC: r['cls'] = 'main:VS'
        elif r['succ'] in FS_SUCC or r['pred'] in ('cp_fs_compact','cp_abuf_interpolate','cp_abuf_interpolate_ranges','cp_fs_interpolate'): r['cls'] = 'main:FS'
        else: r['cls'] = 'main:?'
    else:
        r['cls'] = r['name']
    r['dur'] = r['end'] - r['start']
    r['threads'] = r['blocks'] * r['tpb']

def pct(v, p):
    v = sorted(v); 
    if not v: return None
    k = (len(v)-1)*p; f = int(k); cc = min(f+1, len(v)-1)
    return v[f] + (v[cc]-v[f])*(k-f)

def union(intervals):
    iv = sorted(intervals)
    tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s > ce: tot += ce-cs; cs, ce = s, e
        else: ce = max(ce, e)
    if cs is not None: tot += ce-cs
    return tot

summary = {}
classes = collections.defaultdict(list)
for r in rows: classes[r['cls']].append(r)
out = []
for cls, lst in sorted(classes.items(), key=lambda kv: -sum(r['dur'] for r in kv[1])):
    d = [r['dur'] for r in lst]
    n = len(lst); tot = sum(d)
    u = union([(r['start'], r['end']) for r in lst])
    mn = min(d)
    out.append(dict(cls=cls, n=n, per_frame=n/frames, sum_ms_frame=tot/1e6/frames, union_ms_frame=u/1e6/frames,
                    self_overlap=tot/u if u else None,
                    min_us=mn/1e3, p10_us=pct(d,.1)/1e3, med_us=pct(d,.5)/1e3, mean_us=tot/n/1e3, p90_us=pct(d,.9)/1e3, max_us=max(d)/1e3,
                    floor_x_launches_ms_frame=mn*n/1e6/frames, floor_share=mn*n/tot,
                    p10_x_launches_ms_frame=pct(d,.1)*n/1e6/frames,
                    med_threads=pct([r['threads'] for r in lst],.5), regs_med=pct([r['regs'] for r in lst],.5)))
with open(f'{outdir}/per_class.json','w') as f: json.dump(out, f, indent=1)
hdr = ['cls','n','per_frame','sum_ms_frame','union_ms_frame','self_overlap','min_us','p10_us','med_us','mean_us','p90_us','max_us','floor_x_launches_ms_frame','floor_share','med_threads','regs_med']
print('\t'.join(hdr))
for o in out:
    print('\t'.join(f'{o[h]:.4g}' if isinstance(o[h], float) else str(o[h]) for h in hdr))

# threads-vs-duration bins per selected class
import math
sel = ['main:VS','main:FS','cp_clip_rast_fused','cp_rasterize_stage2','cp_rasterize_stage3','cp_fs_compact','cp_clip_rast_fused_abuf','cp_rasterize_stage3_abuf','cp_fs_writeback','cp_vertex_fetch','cp_clip_triangles']
bins_out = {}
for cls in sel:
    lst = classes.get(cls, [])
    if not lst: continue
    b = collections.defaultdict(list)
    for r in lst:
        t = r['threads']
        k = int(math.floor(math.log2(t))) if t > 0 else -1
        b[k].append(r['dur'])
    bins_out[cls] = []
    print(f'\n== {cls}: duration by log2(total threads) bin (n, min, p10, median, p90 us; sum ms/frame)')
    for k in sorted(b):
        d = b[k]
        row = dict(log2_threads=k, threads_lo=2**k, n=len(d), min_us=min(d)/1e3, p10_us=pct(d,.1)/1e3, med_us=pct(d,.5)/1e3, p90_us=pct(d,.9)/1e3, sum_ms_frame=sum(d)/1e6/frames)
        bins_out[cls].append(row)
        print(f"  2^{k:<2d} ({2**k:>8d}..): n={len(d):6d} min={min(d)/1e3:8.2f} p10={pct(d,.1)/1e3:8.2f} med={pct(d,.5)/1e3:8.2f} p90={pct(d,.9)/1e3:8.2f} sum={sum(d)/1e6/frames:7.4f} ms/f")
with open(f'{outdir}/threads_bins.json','w') as f: json.dump(bins_out, f, indent=1)

# grid distribution for FS main (blocks)
fs = classes.get('main:FS', [])
gb = collections.Counter(r['blocks'] for r in fs)
print('\nFS main grid (blocks) top:', gb.most_common(12))
vs = classes.get('main:VS', [])
print('VS main blocks pct: p10/p50/p90/max', pct([r['blocks'] for r in vs],.1), pct([r['blocks'] for r in vs],.5), pct([r['blocks'] for r in vs],.9), max(r['blocks'] for r in vs) if vs else None)
print('VS main tpb counter', collections.Counter(r['tpb'] for r in vs).most_common(5))
print('FS main tpb counter', collections.Counter(r['tpb'] for r in fs).most_common(5))
print('VS regs', collections.Counter(r['regs'] for r in vs).most_common(8))
print('FS regs', collections.Counter(r['regs'] for r in fs).most_common(8))
print('main:? succ', collections.Counter(r['succ'] for r in classes.get('main:?',[])).most_common(10))
print('main:? pred', collections.Counter(r['pred'] for r in classes.get('main:?',[])).most_common(10))

# total VS thread-lanes per frame (grid lanes; upper bound on invocations)
print('VS lanes/frame (grid*block)', sum(r['threads'] for r in vs)/frames)
print('FS lanes/frame (grid*block)', sum(r['threads'] for r in fs)/frames)
print('clip_rast_fused lanes/frame', sum(r['threads'] for r in classes.get('cp_clip_rast_fused',[]))/frames)
print('clip_triangles lanes/frame', sum(r['threads'] for r in classes.get('cp_clip_triangles',[]))/frames)

# memcpy / memset
mc = [dict(r) for r in c.execute(f"SELECT start,end,bytes,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>={w0} AND start<{w1}")]
print('\nmemcpy in window', len(mc), 'per frame', len(mc)/frames, 'sum ms/frame', sum(r['end']-r['start'] for r in mc)/1e6/frames, 'bytes/frame MB', sum(r['bytes'] for r in mc)/1e6/frames)
byk = collections.defaultdict(list)
for r in mc: byk[r['copyKind']].append(r)
for k, lst in byk.items():
    d=[r['end']-r['start'] for r in lst]; b=[r['bytes'] for r in lst]
    print(f"  copyKind={k}: n/frame={len(lst)/frames:.2f} sum={sum(d)/1e6/frames:.4f} ms/f  bytes/f={sum(b)/1e6/frames:.3f} MB  min={min(d)/1e3:.2f}us med={pct(d,.5)/1e3:.2f}us  med_bytes={pct(b,.5):.0f} max_bytes={max(b)}  floor(min)*n={min(d)*len(lst)/1e6/frames:.4f} ms/f")
# small copies (< 64 KB) vs large
small=[r for r in mc if r['bytes']<65536]; large=[r for r in mc if r['bytes']>=65536]
print('  small(<64KB) n/f', len(small)/frames, 'sum ms/f', sum(r['end']-r['start'] for r in small)/1e6/frames)
print('  large(>=64KB) n/f', len(large)/frames, 'sum ms/f', sum(r['end']-r['start'] for r in large)/1e6/frames, 'MB/f', sum(r['bytes'] for r in large)/1e6/frames)
ms = [dict(r) for r in c.execute(f"SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>={w0} AND start<{w1}")]
print('memset n/f', len(ms)/frames, 'sum ms/f', sum(r['end']-r['start'] for r in ms)/1e6/frames, 'MB/f', sum(r['bytes'] for r in ms)/1e6/frames)

# device union over everything
allint = [(r['start'], r['end']) for r in rows] + [(r['start'], r['end']) for r in mc] + [(r['start'], r['end']) for r in ms]
print('device union ms/frame', union(allint)/1e6/frames, 'window ms/frame', (w1-w0)/1e6/frames)
