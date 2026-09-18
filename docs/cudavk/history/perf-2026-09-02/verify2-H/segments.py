#!/usr/bin/env python3
"""Segment-level census for lead H (segment graphs + cuGraphExecUpdate).
Read-only on the nsys sqlite export. Segments = main-thread issue between
consecutive blocking-call returns (cuStreamSynchronize, cuCtxSynchronize,
cuMemcpyDtoH). Frame = two worker cuEventSynchronize returns.
"""
import sqlite3, sys, bisect, collections, statistics, json

DB = sys.argv[1]
c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
cur = c.cursor()
names = {i: v for i, v in cur.execute('select id, value from StringIds')}
tids = cur.execute('select globalTid, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by 2 desc').fetchall()
MAIN, WORKER = tids[0][0], tids[1][0]

wsync = sorted(e for s, e in cur.execute('select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId in (select id from StringIds where value="cuEventSynchronize")', (WORKER,)))
lo, hi = 2, len(wsync) - 1
if (hi - lo) % 2: hi -= 1
fb = wsync[lo:hi + 1:2]
NF = len(fb) - 1
W0, W1 = fb[0], fb[-1]
print(f'frames {NF} wall/frame {(W1-W0)/1e6/NF:.4f} ms')

host = [(s, e, names[n], c_) for s, e, n, c_ in cur.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<? order by start', (MAIN, W0, W1))]
BLOCKING = {'cuStreamSynchronize', 'cuCtxSynchronize', 'cuMemcpyDtoH_v2', 'vkQueueSubmit'}
# vkQueueSubmit timestamps from the submit shim (CLOCK_MONOTONIC ns) -> trace-relative via systemClockNs
sysclk = cur.execute('select systemClockNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
subs = []
for line in open(sys.argv[3]):
    a = line.split()
    if len(a) >= 1:
        t = int(a[0]) - sysclk
        if W0 <= t < W1:
            subs.append((t, t, 'vkQueueSubmit', -1))
host = sorted(host + subs)
print(f'vkQueueSubmit boundaries inserted {len(subs)} -> {len(subs)/NF:.2f}/frame')
ISSUE = {'cuLaunchKernel', 'cuLaunchKernelEx', 'cuMemcpyHtoDAsync_v2', 'cuMemsetD32Async', 'cuMemsetD8Async', 'cuMemcpy2DAsync_v2', 'cuMemcpy3DAsync_v2', 'cuMemcpyDtoDAsync_v2', 'cuLaunchHostFunc'}
EVT = {'cuEventRecord', 'cuStreamWaitEvent'}

# device ops by correlation id
op = {}
for s, e, st, c_, sn, gx, gy, gz in cur.execute('select start,end,streamId,correlationId,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?', (W0 - 5_000_000, W1 + 5_000_000)):
    op[c_] = (s, e, st, 'K', names[sn])
for s, e, st, c_, ck, b in cur.execute('select start,end,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?', (W0 - 5_000_000, W1 + 5_000_000)):
    op[c_] = (s, e, st, 'C', f'memcpy{ck}')
for s, e, st, c_ in cur.execute('select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>? and start<?', (W0 - 5_000_000, W1 + 5_000_000)):
    op[c_] = (s, e, st, 'S', 'memset')
ev_rec = {c_: st for ts, st, c_, eid in cur.execute('select timestamp,streamId,correlationId,eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT')}
ev_wait = {c_: st for s, e, st, c_, eid in cur.execute('select start,end,streamId,correlationId,eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2')}

# segments: split host rows at blocking calls
segs = []   # each: dict
cur_seg = {'rows': [], 'start': W0}
for r in host:
    s, e, n, c_ = r
    if n in BLOCKING:
        cur_seg['sync'] = r
        segs.append(cur_seg)
        cur_seg = {'rows': [], 'start': e}
    else:
        cur_seg['rows'].append(r)
cur_seg['sync'] = None
segs.append(cur_seg)
print(f'segments {len(segs)} -> {len(segs)/NF:.2f}/frame')

def frame_of(t):
    return bisect.bisect_right(fb, t) - 1

# gap classification helper
def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

per_seg = []
for k, sg in enumerate(segs):
    rows = sg['rows']
    sync = sg['sync']
    hs = sg['start']
    he = sync[0] if sync else (rows[-1][1] if rows else hs)
    kinds = collections.Counter()
    sig_full = []
    sig_kern = []
    api_issue = 0.0
    api_evt = 0.0
    api_other = 0.0
    dev = []   # (s,e,st,kind,name,api_end)
    n2d = 0
    for s, e, n, c_ in rows:
        if n in ISSUE:
            api_issue += e - s
            kinds[n] += 1
            if n == 'cuLaunchHostFunc':
                sig_full.append(('H', 'hostfn', 0))
                continue
            if n in ('cuMemcpy2DAsync_v2', 'cuMemcpy3DAsync_v2'):
                n2d += 1
            o = op.get(c_)
            if o:
                dev.append((o[0], o[1], o[2], o[3], o[4], e))
                sig_full.append((o[3], o[4], o[2]))
                if o[3] == 'K':
                    sig_kern.append(o[4])
        elif n in EVT:
            api_evt += e - s
            kinds[n] += 1
            if n == 'cuEventRecord':
                sig_full.append(('R', '', ev_rec.get(c_, -1)))
            else:
                sig_full.append(('W', '', ev_wait.get(c_, -1)))
        else:
            api_other += e - s
            kinds[n] += 1
    dev.sort()
    d0 = dev[0][0] if dev else None
    d1 = max(d[1] for d in dev) if dev else None
    # device idle inside the segment's device span split into host-late vs latency-only
    host_late = 0.0
    lat_only = 0.0
    if dev:
        u = union([(d[0], d[1]) for d in dev])
        # for each idle gap, find op starting at gap end
        starts = {d[0]: d for d in dev}
        for a, b in zip(u, u[1:]):
            g0, g1 = a[1], b[0]
            nxt = starts.get(g1)
            ae = nxt[5] if nxt else g0
            hl = max(0, min(g1, ae) - g0)
            host_late += hl
            lat_only += (g1 - g0) - hl
    nk = len(sig_kern)
    nops = len([x for x in sig_full if x[0] in 'KCS'])
    per_seg.append(dict(k=k, frame=frame_of(he), hs=hs, he=he, api_issue=api_issue, api_evt=api_evt, api_other=api_other,
                        nk=nk, nops=nops, nev=kinds['cuEventRecord'] + kinds['cuStreamWaitEvent'], n2d=n2d,
                        nhost=kinds['cuLaunchHostFunc'], nc=kinds['cuMemcpyHtoDAsync_v2'],
                        ns=kinds['cuMemsetD32Async'] + kinds['cuMemsetD8Async'],
                        d0=d0, d1=d1, host_late=host_late, lat_only=lat_only,
                        sig_full=tuple(sig_full), sig_kern=tuple(sig_kern),
                        sync=sync[2] if sync else None, sync_dur=(sync[1] - sync[0]) if sync else 0))

# frame-level completeness: only frames fully inside
frames = collections.defaultdict(list)
for p in per_seg:
    if 0 <= p['frame'] < NF:
        frames[p['frame']].append(p)

def med(v):
    return statistics.median(v) if v else float('nan')

# --- topology census
def census(key, label, nonempty_only):
    sigs = [p[key] for p in per_seg if 0 <= p['frame'] < NF and (p['nops'] > 0 or not nonempty_only)]
    distinct = len(set(sigs))
    prev_seen = tot = pos = pos_tot = 0
    for f in sorted(frames):
        if f - 1 not in frames: continue
        cur_ = [p[key] for p in frames[f] if (p['nops'] > 0 or not nonempty_only)]
        prv = [p[key] for p in frames[f - 1] if (p['nops'] > 0 or not nonempty_only)]
        ps = set(prv)
        for j, sg in enumerate(cur_):
            tot += 1
            prev_seen += sg in ps
            if j < len(prv):
                pos_tot += 1
                pos += prv[j] == sg
    cnt = collections.Counter(sigs)
    top = cnt.most_common(50)
    print(f'\n== {label}: segments {len(sigs)} distinct {distinct}; seen-in-previous-frame {100*prev_seen/max(tot,1):.1f}%; same-position {100*pos/max(pos_tot,1):.1f}%; top-50 cover {100*sum(v for _,v in top)/len(sigs):.1f}%')
    # how many distinct signatures needed to cover X% of *launches* (weighted by op count)
    wl = collections.Counter()
    for p in per_seg:
        if 0 <= p['frame'] < NF:
            wl[p[key]] += p['nops']
    totops = sum(wl.values())
    acc = 0; n = 0
    for sgn, v in wl.most_common():
        acc += v; n += 1
        if acc >= 0.9 * totops:
            break
    print(f'   signatures needed to cover 90% of device ops: {n} of {len(wl)}; total ops {totops/NF:.1f}/frame')
    # singletons: signatures seen once in the whole window (must be instantiated for one use)
    once = sum(1 for s_, v in cnt.items() if v == 1)
    print(f'   signatures seen exactly once in window: {once} ({100*once/max(len(sigs),1):.2f}% of segments); ops in them {sum(wl[s_] for s_,v in cnt.items() if v==1)/NF:.2f}/frame')
    return cnt

census('sig_kern', 'kernel-name-only signature (closure-audit definition), incl. empty', False)
census('sig_kern', 'kernel-name-only signature, non-empty segments', True)
census('sig_full', 'FULL node signature: ordered (kind,name,stream) for kernels/memcpy/memset/hostfn + event record/wait streams, non-empty', True)

# --- per-frame aggregates
ne = [p for p in per_seg if 0 <= p['frame'] < NF and p['nops'] > 0]
print(f'\nnon-empty segments/frame {len(ne)/NF:.2f}; ops/segment p50 {med([p["nops"] for p in ne]):.0f} p90 {sorted(p["nops"] for p in ne)[len(ne)*9//10]} ; kernels/segment p50 {med([p["nk"] for p in ne]):.0f}')
tot_ops = sum(p['nops'] for p in ne) / NF
tot_k = sum(p['nk'] for p in ne) / NF
tot_c = sum(p['nc'] for p in ne) / NF
tot_s = sum(p['ns'] for p in ne) / NF
tot_ev = sum(p['nev'] for p in ne) / NF
tot_2d = sum(p['n2d'] for p in ne) / NF
tot_hf = sum(p['nhost'] for p in ne) / NF
print(f'per frame: kernels {tot_k:.1f} htod-copies {tot_c:.1f} memsets {tot_s:.1f} 2D/3D copies {tot_2d:.1f} hostfn {tot_hf:.1f} event rec+wait {tot_ev:.1f}')
seg_with_2d = sum(1 for p in ne if p['n2d'] > 0) / NF
seg_with_hf = sum(1 for p in ne if p['nhost'] > 0) / NF
print(f'segments/frame containing a 2D/3D copy (not ExecUpdate-able): {seg_with_2d:.2f}; containing a host-func node: {seg_with_hf:.2f}; ops in 2D-bearing segments {sum(p["nops"] for p in ne if p["n2d"]>0)/NF:.1f}/frame')
# small segments
small = [p for p in ne if p['nops'] <= 3]
print(f'segments with <=3 device ops: {len(small)/NF:.2f}/frame carrying {sum(p["nops"] for p in small)/NF:.1f} ops/frame')

# host issue API time (CUPTI-inflated) per frame within segments
api_issue = sum(p['api_issue'] for p in ne) / 1e6 / NF
api_evt = sum(p['api_evt'] for p in ne) / 1e6 / NF
print(f'\nhost issue API time/frame (CUPTI-inflated): issue {api_issue:.3f} ms, event traffic {api_evt:.3f} ms; /1.28 -> {api_issue/1.28:.3f} / {api_evt/1.28:.3f}')

# host-late vs latency-only device idle inside segments
hl = sum(p['host_late'] for p in ne) / 1e6 / NF
lo_ = sum(p['lat_only'] for p in ne) / 1e6 / NF
print(f'device idle INSIDE segment device spans: host-late (next op API not yet returned) {hl:.3f} ms/frame; launch-latency-only {lo_:.3f} ms/frame')

# how long after the device starts does the host keep issuing? (deferral cost proxy)
tail = []
for p in ne:
    if p['d0'] is not None:
        tail.append(p['he'] - p['d0'])   # host issue continuing after first device op start
print(f'host issue after first device op start (s_k - d0): median {med(tail)/1e3:.1f} us, sum {sum(tail)/1e6/NF:.3f} ms/frame')
# device-bound vs host-bound segments: d1 - he (device runs past the host reaching the sync)
db = [p['d1'] - p['he'] for p in ne if p['d1'] is not None]
print(f'device end minus host-reaches-sync (d1 - s_k): median {med(db)/1e3:.1f} us; segments where host arrived first (device-bound) {100*sum(1 for x in db if x>0)/len(db):.1f}%')

# --- deferral model (MODEL): graph = build all params, launch once, device runs its chain without host-late idle
def model(u_node, graph_launch_us, first_lat_us, gap_scale, label):
    deltas = collections.defaultdict(float)
    for p in ne:
        if p['d1'] is None: continue
        n = p['nops']
        api_today = (p['api_issue'] + p['api_evt']) / 1.28    # de-inflate
        api_graph = n * u_node * 1e3 + graph_launch_us * 1e3
        savings = api_today - api_graph
        # today: device done at d1. graph: launch at he - savings, first node at +first_lat, chain = span - host_late - lat_only*(1-gap_scale)
        span = p['d1'] - p['d0']
        chain = span - p['host_late'] - p['lat_only'] * (1 - gap_scale)
        new_d1 = (p['he'] - savings) + first_lat_us * 1e3 + chain
        deltas[p['frame']] += new_d1 - p['d1']
    v = [deltas[f] / 1e6 for f in range(NF) if f in deltas]
    print(f'   {label}: frame delta median {med(v):+.3f} ms  mean {sum(v)/len(v):+.3f}')

print('\nMODEL deferral: per-node update+launch cost u, graph launch cost, first-node latency, inter-node gap scale')
model(0.4, 3.0, 5.0, 1.0, 'u=0.4us launch=3us first=5us gaps=1.0 (lead\'s assumptions, gaps unchanged)')
model(0.4, 3.0, 5.0, 0.5, 'u=0.4us launch=3us first=5us gaps=0.5 (lead\'s optimistic device half)')
model(0.4, 3.0, 5.0, 0.0, 'u=0.4us launch=3us first=5us gaps=0   (impossible best)')
model(1.0, 5.0, 8.0, 1.0, 'u=1.0us launch=5us first=8us gaps=1.0 (pessimistic host)')
model(0.4, 0.0, 0.0, 1.0, 'u=0.4 no launch/first-node cost, gaps=1.0 (no deferral penalty beyond serialisation)')
# variant: no deferral — hypothetical pipelined issue where device start is unchanged and only per-op API shrinks and host-late idle shrinks proportionally
def model_nodefer(u_node, label):
    deltas = collections.defaultdict(float)
    for p in ne:
        if p['d1'] is None: continue
        api_today = (p['api_issue'] + p['api_evt']) / 1.28
        api_graph = p['nops'] * u_node * 1e3
        frac = min(1.0, api_graph / api_today) if api_today > 0 else 1.0
        deltas[p['frame']] += -(p['host_late'] * (1 - frac))
    v = [deltas[f] / 1e6 for f in range(NF) if f in deltas]
    print(f'   {label}: frame delta median {med(v):+.3f} ms  mean {sum(v)/len(v):+.3f}')
print('MODEL no-deferral upper bound (host-late idle shrinks with API time; device start unchanged):')
model_nodefer(0.4, 'u=0.4us')
model_nodefer(0.0, 'u=0 (host issue free inside segments)')

json.dump({'NF': NF, 'segments_per_frame': len(ne) / NF}, open(sys.argv[2], 'w'))
