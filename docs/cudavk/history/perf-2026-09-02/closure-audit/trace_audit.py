#!/usr/bin/env python3
"""Read-only audit of an nsys sqlite export of a favorite3 replay.

Computes, over a frame window delimited by the completion worker's
cuEventSynchronize returns (one per submit, two per frame):
  1. per-frame wall / device union / idle / host blocked
  2. union-exclusive per pool (cross-check against the docs)
  3. inter-sync segments: critical-path proxy vs union-exclusive
  4. post-sync refill bubble and single-stream join tail
  5. per-class duration model  duration = a + b*blocks  (latency-shell test)
  6. segment topology recurrence (graph ExecUpdate premise)
  7. fan-out group max/mean (merge-rule 2x puzzle)
  8. dependency-respecting what-if replay (delete a pool / free syncs / ...)
Pure python + sqlite3, mode=ro.  Nothing is written outside the scratchpad.
"""
import sqlite3, sys, bisect, statistics, collections, json, time

PATH = sys.argv[1]
TAG = sys.argv[2]
WIN = None
if len(sys.argv) > 4 and int(sys.argv[4]) > 0:
    WIN = (int(sys.argv[3]), int(sys.argv[4]))
OUT = sys.argv[5] if len(sys.argv) > 5 else None

t0 = time.time()
con = sqlite3.connect(f'file:{PATH}?mode=ro', uri=True)
cur = con.cursor()
names = dict(cur.execute('select id, value from StringIds'))

def log(*a):
    print(*a, flush=True)

# ---------------------------------------------------------------- threads
tids = cur.execute('select globalTid, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by 2 desc').fetchall()
MAIN_TID = tids[0][0]
WORKER_TID = tids[1][0]
log('main tid', MAIN_TID, 'worker tid', WORKER_TID)

# ---------------------------------------------------------------- frames
wsync = [r for r in cur.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r where globalTid=? and nameId in (select id from StringIds where value="cuEventSynchronize") order by start', (WORKER_TID,))]
sub_end = [e for s, e, c in wsync]
log('worker cuEventSynchronize rows (submits)', len(sub_end))

if WIN is None:
    # drop the first two partial submits, use complete pairs
    lo_i = 2
    hi_i = len(sub_end) - 1
    if (hi_i - lo_i) % 2:
        hi_i -= 1
    WIN = (sub_end[lo_i], sub_end[hi_i])
    frame_bounds = sub_end[lo_i:hi_i + 1:2]
else:
    frame_bounds = [t for t in sub_end if WIN[0] <= t <= WIN[1]]
    # keep even count so frames pair
    if len(frame_bounds) % 2 == 0:
        frame_bounds = frame_bounds[:-1]
    frame_bounds = frame_bounds[0::2]
NF = len(frame_bounds) - 1
WALL = (frame_bounds[-1] - frame_bounds[0]) / 1e6
log(f'window {WIN} frames {NF} wall {WALL:.3f} ms -> {WALL/NF:.4f} ms/frame mean')
fw = [(frame_bounds[i + 1] - frame_bounds[i]) / 1e6 for i in range(NF)]
log(f'frame wall median {statistics.median(fw):.4f} p10 {sorted(fw)[NF//10]:.4f} p90 {sorted(fw)[NF*9//10]:.4f}')

W0, W1 = frame_bounds[0], frame_bounds[-1]

# ---------------------------------------------------------------- device ops
# op tuple: [start, end, stream, kind, name, blocks, corr]
ops = []
for s, e, st, c, sn, gx, gy, gz in cur.execute('select start,end,streamId,correlationId,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?', (W0, W1)):
    ops.append([s, e, st, 'K', names[sn], gx * gy * gz, c])
for s, e, st, c, ck, b in cur.execute('select start,end,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?', (W0, W1)):
    ops.append([s, e, st, 'C', f'memcpy{ck}', b, c])
for s, e, st, c in cur.execute('select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>? and start<?', (W0, W1)):
    ops.append([s, e, st, 'S', 'memset', 0, c])
ops.sort(key=lambda o: o[0])
log('device ops in window', len(ops), 'kernels', sum(1 for o in ops if o[3] == 'K'))

# classify 'main' by same-stream grammar
by_stream = collections.defaultdict(list)
for i, o in enumerate(ops):
    if o[3] == 'K':
        by_stream[o[2]].append(i)
CLIP = {'cp_clip_rast_fused', 'cp_clip_rast_fused_abuf', 'cp_clip_triangles'}
cls = [None] * len(ops)
for st, idx in by_stream.items():
    for j, i in enumerate(idx):
        n = ops[i][4]
        if n != 'main':
            cls[i] = n
            continue
        nxt = ops[idx[j + 1]][4] if j + 1 < len(idx) else ''
        prv = ops[idx[j - 1]][4] if j > 0 else ''
        if nxt in CLIP or nxt == 'cp_vertex_fetch':
            cls[i] = 'VS'
        elif prv == 'cp_fs_compact' or nxt == 'cp_fs_writeback':
            cls[i] = 'FS_direct'
        elif prv.startswith('cp_abuf') or nxt.startswith('cp_abuf'):
            cls[i] = 'FS_abuf'
        else:
            cls[i] = 'main_other'
for i, o in enumerate(ops):
    if cls[i] is None:
        cls[i] = o[4]

POOLS = {
    'VS': lambda i: cls[i] == 'VS',
    'FS_direct': lambda i: cls[i] == 'FS_direct',
    'FS_abuf': lambda i: cls[i] == 'FS_abuf',
    'main_other': lambda i: cls[i] == 'main_other',
    'clip_all': lambda i: cls[i] in CLIP,
    'clip_rast_fused': lambda i: cls[i] == 'cp_clip_rast_fused',
    'stage3': lambda i: cls[i] == 'cp_rasterize_stage3',
    'stage2': lambda i: cls[i] == 'cp_rasterize_stage2',
    'stage3_abuf': lambda i: cls[i] == 'cp_rasterize_stage3_abuf',
    'fs_compact': lambda i: cls[i] == 'cp_fs_compact',
    'fs_writeback': lambda i: cls[i] == 'cp_fs_writeback',
    'vertex_fetch': lambda i: cls[i] == 'cp_vertex_fetch',
    'abuf_support': lambda i: cls[i].startswith('cp_abuf'),
    'memcpy': lambda i: ops[i][3] == 'C',
    'memcpy_HtoD': lambda i: ops[i][4] == 'memcpy1',
    'memset': lambda i: ops[i][3] == 'S',
    'all_kernels': lambda i: ops[i][3] == 'K',
}

def union_ms(iv):
    """iv: list of (s,e) sorted by s. returns union ms clipped to window."""
    tot = 0
    cs = ce = None
    for s, e in iv:
        if s < W0: s = W0
        if e > W1: e = W1
        if e <= s: continue
        if cs is None:
            cs, ce = s, e
        elif s > ce:
            tot += ce - cs; cs, ce = s, e
        elif e > ce:
            ce = e
    if cs is not None:
        tot += ce - cs
    return tot / 1e6

all_iv = [(o[0], o[1]) for o in ops]
U_all = union_ms(all_iv)
S_all = sum(min(o[1], W1) - max(o[0], W0) for o in ops) / 1e6
log(f'device union {U_all/NF:.4f} ms/frame  summed {S_all/NF:.4f}  idle {WALL/NF - U_all/NF:.4f}  overlap {S_all/U_all:.3f}x')

log('== ops per class on main stream 13 vs side streams (per frame):')
scount = collections.defaultdict(lambda: [0, 0])
for i, o in enumerate(ops):
    k = cls[i] if o[3] == 'K' else o[4]
    scount[k][0 if o[2] == 13 else 1] += 1
for k, (m, sd) in sorted(scount.items(), key=lambda kv: -sum(kv[1]))[:20]:
    log(f'   {k:24s} main {m/NF:7.2f} side {sd/NF:7.2f}')

# ---------------------------------------------------------------- 2. union-exclusive
ue = {}
for pn, pf in POOLS.items():
    iv = [(o[0], o[1]) for i, o in enumerate(ops) if not pf(i)]
    su = sum(min(o[1], W1) - max(o[0], W0) for i, o in enumerate(ops) if pf(i)) / 1e6
    cnt = sum(1 for i in range(len(ops)) if pf(i))
    ue[pn] = ((U_all - union_ms(iv)) / NF, su / NF, cnt / NF)
log('\n== union-exclusive per pool (ms/frame): exclusive, summed, launches/frame')
for pn, (x, su, c) in sorted(ue.items(), key=lambda kv: -kv[1][0]):
    log(f'  {pn:18s} {x:8.4f} {su:8.4f} {c:8.2f}  self-overlap {su/x if x else 0:.2f}x')

# ---------------------------------------------------------------- host rows
BLOCKING = {'cuCtxSynchronize', 'cuStreamSynchronize', 'cuEventSynchronize', 'cuMemcpyDtoH_v2', 'cuMemcpyHtoD_v2'}
host = []  # (start,end,name,corr)
for s, e, nid, c in cur.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and end>? and start<? order by start', (MAIN_TID, W0 - 50_000_000, W1)):
    host.append((s, e, names[nid], c))
log('main-thread runtime rows', len(host))
hb = sum(e - s for s, e, n, c in host if n in BLOCKING and s >= W0 and e <= W1) / 1e6
hapi = sum(e - s for s, e, n, c in host if s >= W0 and e <= W1) / 1e6
hcount = collections.Counter(n for s, e, n, c in host if n in BLOCKING and s >= W0 and e <= W1)
log(f'host blocked in sync APIs {hb/NF:.4f} ms/frame; all main-thread CUDA API residency {hapi/NF:.4f}; blocking calls/frame {sum(hcount.values())/NF:.2f} {dict(hcount)}')

# ---------------------------------------------------------------- 3. segments
syncs = [(s, e, n) for s, e, n, c in host if n in BLOCKING and W0 <= e <= W1]
syncs.sort(key=lambda x: x[1])
seg_end = [e for s, e, n in syncs]
seg_start_call = [s for s, e, n in syncs]
log(f'blocking syncs in window {len(syncs)} -> {len(syncs)/NF:.2f}/frame')

# assign each op to segment by its END (the segment whose sync returns after the op ends)
# (an op straddling a sync end is assigned to the later segment; report the straddle mass)
seg_ops = collections.defaultdict(list)
straddle = 0
for i, o in enumerate(ops):
    k = bisect.bisect_left(seg_end, o[1])
    if k >= len(seg_end):
        continue
    if o[0] < (seg_end[k - 1] if k > 0 else W0):
        straddle += 1
    seg_ops[k].append(i)
log(f'ops straddling a sync boundary {straddle} of {len(ops)} ({100*straddle/len(ops):.1f}%)')

def seg_metrics(idx, skip):
    """returns (union_ns, max_stream_sum_ns, span_ns)"""
    iv = sorted((ops[i][0], ops[i][1]) for i in idx if not skip(i))
    if not iv:
        return 0, 0, 0
    u = 0; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            u += ce - cs; cs, ce = s, e
        elif e > ce:
            ce = e
    u += ce - cs
    ss = collections.defaultdict(int)
    for i in idx:
        if not skip(i):
            ss[ops[i][2]] += ops[i][1] - ops[i][0]
    return u, max(ss.values()), iv[-1][1] - iv[0][0]

blocked = [e - s for s, e, n in syncs]
seg_base = {k: seg_metrics(idx, lambda i: False) for k, idx in seg_ops.items()}
tot_u = sum(v[0] for v in seg_base.values()) / 1e6
tot_cp = sum(v[1] for v in seg_base.values()) / 1e6
tot_span = sum(v[2] for v in seg_base.values()) / 1e6
log(f'\n== per-segment totals ms/frame: union {tot_u/NF:.4f}  max-stream-sum {tot_cp/NF:.4f}  span {tot_span/NF:.4f}  blocked {sum(blocked)/1e6/NF:.4f}')
log('== pool: sum ΔU, sum ΔCP(max-stream), Σmin(ΔU,blocked), Σmin(ΔCP,blocked)  [ms/frame]')
segres = {}
for pn in ['VS', 'FS_direct', 'clip_all', 'clip_rast_fused', 'stage3', 'stage2', 'fs_compact', 'fs_writeback', 'memcpy', 'memcpy_HtoD', 'memset', 'abuf_support', 'stage3_abuf', 'FS_abuf']:
    pf = POOLS[pn]
    dU = dCP = mU = mCP = 0
    for k, idx in seg_ops.items():
        u, cp, sp = seg_base[k]
        u2, cp2, sp2 = seg_metrics(idx, pf)
        b = blocked[k]
        dU += u - u2; dCP += cp - cp2
        mU += min(u - u2, b); mCP += min(cp - cp2, b)
    segres[pn] = (dU / 1e6 / NF, dCP / 1e6 / NF, mU / 1e6 / NF, mCP / 1e6 / NF)
    log(f'  {pn:16s} {dU/1e6/NF:8.4f} {dCP/1e6/NF:8.4f} {mU/1e6/NF:8.4f} {mCP/1e6/NF:8.4f}')

# ---------------------------------------------------------------- 4. bubbles
starts = [o[0] for o in ops]
ends_sorted = sorted(o[1] for o in ops)
bub = []; tails = []
for k, (s, e, n) in enumerate(syncs):
    j = bisect.bisect_left(starts, e)
    if j < len(starts):
        bub.append(starts[j] - e)
    # device tail before the sync end: last device end before e; and time since 2nd stream stopped
    idx = seg_ops.get(k, [])
    if idx:
        last_end = max(ops[i][1] for i in idx)
        # concurrency in the segment: time with >=2 streams active
        evs = []
        for i in idx:
            evs.append((ops[i][0], 1)); evs.append((ops[i][1], -1))
        evs.sort()
        act = 0; t_prev = None; last_multi = None; one_only = 0
        for t, d in evs:
            if t_prev is not None and act >= 2:
                last_multi = t
            act += d; t_prev = t
        # single-stream tail = last_end - last time >=2 were active (or segment first start)
        first = min(ops[i][0] for i in idx)
        tails.append(last_end - (last_multi if last_multi else first))
bub_pos = [b for b in bub if b >= 0]
log(f'\n== post-sync refill bubble (sync return -> first device op start): median {statistics.median(bub_pos)/1e3:.1f} us, sum {sum(bub_pos)/1e6/NF:.4f} ms/frame over {len(bub_pos)/NF:.2f} syncs/frame')
log(f'== single-stream join tail before each sync (device end - last time >=2 streams active): median {statistics.median(tails)/1e3:.1f} us, sum {sum(tails)/1e6/NF:.4f} ms/frame')

# concurrency histogram over the window
evs = []
for o in ops:
    evs.append((max(o[0], W0), 1)); evs.append((min(o[1], W1), -1))
evs.sort()
hist = collections.Counter(); act = 0; tp = evs[0][0]
for t, d in evs:
    hist[act] += t - tp; act += d; tp = t
log('== time with k device ops concurrently active (ms/frame):', {k: round(v / 1e6 / NF, 4) for k, v in sorted(hist.items()) if k <= 12})

# ---------------------------------------------------------------- 5. duration model
log('\n== duration model per class: n/frame, min, p10, p50, p90 (us), fit a(us)+b(ns/block), floor share n*a/sum')
def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0
    a = my - b * mx
    return a, b
model = {}
for cn in ['VS', 'FS_direct', 'FS_abuf', 'cp_clip_rast_fused', 'cp_clip_rast_fused_abuf', 'cp_rasterize_stage2', 'cp_rasterize_stage3', 'cp_rasterize_stage3_abuf', 'cp_fs_compact', 'cp_fs_writeback', 'cp_vertex_fetch', 'cp_clip_triangles', 'cp_abuf_scan_finish']:
    sel = [i for i in range(len(ops)) if cls[i] == cn]
    if not sel: continue
    d = sorted((ops[i][1] - ops[i][0]) / 1e3 for i in sel)
    xs = [ops[i][5] for i in sel]; ys = [(ops[i][1] - ops[i][0]) / 1e3 for i in sel]
    a, b = fit(xs, ys)
    n = len(sel)
    floor_share = (n * max(a, 0)) / sum(ys) if sum(ys) else 0
    model[cn] = dict(n_per_frame=n / NF, min=d[0], p10=d[n // 10], p50=d[n // 2], p90=d[n * 9 // 10], a_us=a, b_ns_per_block=b * 1e3, floor_share=floor_share, sum_ms_per_frame=sum(ys) / 1e3 / NF, blocks_p50=sorted(xs)[n // 2])
    log(f'  {cn:24s} n/f {n/NF:7.2f} min {d[0]:6.1f} p10 {d[n//10]:6.1f} p50 {d[n//2]:6.1f} p90 {d[n*9//10]:6.1f}  a {a:6.2f} b {b*1e3:8.3f}  floor {100*floor_share:5.1f}%  sum {sum(ys)/1e3/NF:.4f} ms/f  blocks p50 {sorted(xs)[n//2]}')

# VS / FS duration by grid bucket
for cn in ['VS', 'FS_direct']:
    log(f'  {cn} by grid bucket: bucket, n/frame, median us, sum ms/frame')
    bk = collections.defaultdict(list)
    for i in range(len(ops)):
        if cls[i] == cn:
            g = ops[i][5]
            b = 0
            while (1 << b) < g: b += 1
            bk[b].append((ops[i][1] - ops[i][0]) / 1e3)
    for b in sorted(bk):
        v = bk[b]
        log(f'     <=2^{b:<2d} n/f {len(v)/NF:7.2f}  p50 {statistics.median(v):7.1f}  sum {sum(v)/1e3/NF:.4f}')

# ---------------------------------------------------------------- 6. topology recurrence
sig_by_seg = {}
for k, idx in seg_ops.items():
    ks = sorted((ops[i][6], ops[i][4]) for i in idx if ops[i][3] == 'K')
    sig_by_seg[k] = tuple(n for c, n in ks)
sigs = list(sig_by_seg.values())
distinct = len(set(sigs))
# per frame: list of segment signatures
seg_frame = {}
for k, (s, e, n) in enumerate(syncs):
    f = bisect.bisect_right(frame_bounds, e) - 1
    seg_frame.setdefault(f, []).append(sig_by_seg.get(k, ()))
prev_seen = 0; tot = 0; exact_pos = 0; pos_tot = 0
for f in sorted(seg_frame):
    if f - 1 in seg_frame:
        prevset = set(seg_frame[f - 1])
        for j, sg in enumerate(seg_frame[f]):
            tot += 1
            if sg in prevset: prev_seen += 1
            if j < len(seg_frame[f - 1]):
                pos_tot += 1
                if seg_frame[f - 1][j] == sg: exact_pos += 1
log(f'\n== inter-sync segment kernel-name topology: {len(sigs)} segments, {distinct} distinct signatures; {100*prev_seen/max(tot,1):.1f}% of segments have a signature seen in the previous frame; same-position match {100*exact_pos/max(pos_tot,1):.1f}%')
seglen = sorted(len(s) for s in sigs)
log(f'   segment kernel count: p10 {seglen[len(seglen)//10]} p50 {seglen[len(seglen)//2]} p90 {seglen[len(seglen)*9//10]} max {seglen[-1]}; segments/frame {len(sigs)/NF:.2f}')
cnt = collections.Counter(sigs)
cov = 0; top = cnt.most_common(50)
log('   top-50 signatures cover', f'{100*sum(v for k,v in top)/len(sigs):.1f}% of segments; top-5 lengths/counts:', [(len(k), v) for k, v in top[:5]])

# ---------------------------------------------------------------- 7. fan-out max/mean
log('\n== fan-out groups: stage3 direct launches on side streams within one segment: n groups, mean size, mean(max/mean duration)')
r = []
for k, idx in seg_ops.items():
    d = [(ops[i][1] - ops[i][0]) for i in idx if cls[i] == 'cp_rasterize_stage3' and ops[i][2] != 13]
    if len(d) >= 3:
        r.append((len(d), max(d) / (sum(d) / len(d))))
if r:
    log(f'   groups {len(r)}, mean size {sum(x for x,_ in r)/len(r):.2f}, median max/mean {statistics.median(y for _,y in r):.2f}, mean {sum(y for _,y in r)/len(r):.2f}')
r = []
for k, idx in seg_ops.items():
    d = [(ops[i][1] - ops[i][0]) for i in idx if cls[i] == 'cp_clip_rast_fused' and ops[i][2] != 13]
    if len(d) >= 3:
        r.append((len(d), max(d) / (sum(d) / len(d))))
if r:
    log(f'   clip_rast_fused side-stream groups {len(r)}, mean size {sum(x for x,_ in r)/len(r):.2f}, median max/mean {statistics.median(y for _,y in r):.2f}')

# ---------------------------------------------------------------- 8. what-if replay
log('\n== dependency-respecting what-if replay')
# host op list in issue order: launches (by correlation), event records, stream waits, blocking syncs
corr_to_op = {o[6]: i for i, o in enumerate(ops)}
ev_rec = {}  # correlationId -> (eventId, streamId)
for ts, st, c, eid in cur.execute('select timestamp, streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
    ev_rec[c] = (eid, st)
waits = {}  # correlationId -> (eventId, streamId)
for s, e, st, c, eid in cur.execute('select start, end, streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2'):
    waits[c] = (eid, st)
log('event records', len(ev_rec), 'stream waits', len(waits))

# worker syncs: eventId of each, to find frame completion
wsync_ev = {}
for s, e, c in wsync:
    pass
# worker cuEventSynchronize has correlationId; the SYNCHRONIZATION type-1 rows carry eventId
ss_stream = {}
for c, st in cur.execute('select correlationId, streamId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=3'):
    ss_stream[c] = st

# Build timeline of host events
H = []  # (time, kind, payload)
for s, e, n, c in host:
    if c in corr_to_op and n in BLOCKING:
        H.append((s, e, 'LB', c))
    elif c in corr_to_op:
        H.append((s, e, 'L', c))
    elif n == 'cuEventRecord' and c in ev_rec:
        H.append((s, e, 'R', c))
    elif n == 'cuStreamWaitEvent' and c in waits:
        H.append((s, e, 'W', c))
    elif n in BLOCKING:
        H.append((s, e, 'B', (n, c)))
H.sort()
log('host timeline events', len(H))

# original per-stream issue order and dependency resolution (static; only times change)
def simulate(delete=lambda i: False, free_sync=False, launch_gap_scale=1.0, dur_scale=lambda i: 1.0, host_scale=1.0, delete_host=False, indep=lambda i: False):
    indep_last = {}; indep_last_orig = {}
    new_end_op = {}      # op index -> new end
    stream_last = {}     # stream -> new end of last issued op
    stream_last_orig = {}
    event_point = {}     # eventId -> (new_time, orig_time) of last record
    hclock_new = None    # new host clock at the END of the previous host call
    last_e = None        # orig end of previous host call
    for (s, e, kind, pl) in H:
        if hclock_new is None:
            t_new = s
        else:
            t_new = hclock_new + max(s - last_e, 0) * host_scale
        last_e = e
        if kind == 'L' or kind == 'LB':
            i = corr_to_op[pl]
            o = ops[i]
            st = o[2]
            api = (e - s) * host_scale
            if delete(i) and delete_host:
                api = 0.0
            issue_new = t_new + api
            ready_new = max(issue_new, stream_last.get(st, 0))
            ready_orig = max(e, stream_last_orig.get(st, 0))
            gap = o[0] - ready_orig
            gap = gap * launch_gap_scale if gap > 0 else gap
            if indep(i):
                ready_new = max(issue_new, indep_last.get(st, 0))
            else:
                ready_new = max(ready_new, indep_last.get(st, 0))
            if delete(i):
                new_end_op[i] = ready_new
                stream_last_orig[st] = o[1]
                ne_op = ready_new
            else:
                ns = ready_new + gap
                ne_op = ns + (o[1] - o[0]) * dur_scale(i)
                new_end_op[i] = ne_op
                stream_last_orig[st] = o[1]
                if indep(i):
                    indep_last[st] = ne_op
                else:
                    stream_last[st] = max(stream_last.get(st, 0), ne_op)
            if kind == 'LB':
                resid = max(e - o[1], 0)
                if free_sync:
                    hclock_new = t_new + min(e - s, 3000) * host_scale
                else:
                    hclock_new = max(issue_new, ne_op) + resid
            else:
                hclock_new = issue_new
        elif kind == 'R':
            eid, st = ev_rec[pl]
            event_point[eid] = (stream_last.get(st, t_new), stream_last_orig.get(st, s))
            hclock_new = t_new + (e - s) * host_scale
        elif kind == 'W':
            eid, st = waits[pl]
            if eid in event_point:
                en, eo = event_point[eid]
                stream_last[st] = max(stream_last.get(st, 0), en)
                stream_last_orig[st] = max(stream_last_orig.get(st, 0), eo)
            hclock_new = t_new + (e - s) * host_scale
        elif kind == 'B':
            n, c = pl
            if n == 'cuStreamSynchronize' and c in ss_stream:
                st = ss_stream[c]
                dev_new = stream_last.get(st, t_new)
                dev_orig = stream_last_orig.get(st, s)
            else:
                dev_new = max(stream_last.values()) if stream_last else t_new
                dev_orig = max(stream_last_orig.values()) if stream_last_orig else s
            resid = max(e - max(s, dev_orig), 0)   # original return overhead after device ready
            if free_sync:
                hclock_new = t_new + min(resid, 3000)
            else:
                hclock_new = max(t_new, dev_new) + resid
    return new_end_op, stream_last, 0

def frame_times(new_end_op):
    # for each frame boundary (orig), take the op whose orig end is the latest <= boundary and use its new end
    ends = sorted((o[1], i) for i, o in enumerate(ops) if i in new_end_op)
    keys = [x[0] for x in ends]
    nb = []
    for b in frame_bounds:
        j = bisect.bisect_right(keys, b) - 1
        nb.append(new_end_op[ends[j][1]] if j >= 0 else b)
    fw = [(nb[i + 1] - nb[i]) / 1e6 for i in range(len(nb) - 1)]
    return statistics.median(fw), (nb[-1] - nb[0]) / 1e6 / (len(nb) - 1)

# launch-gap statistics (front-end latency after host issue end and stream availability)
gaps = collections.defaultdict(list)
_sl = {}
_issue = {}
for (s_, e_, kind, pl) in H:
    if kind in ('L', 'LB'):
        i = corr_to_op[pl]; o = ops[i]; st = o[2]
        ready = max(e_, _sl.get(st, 0))
        gaps[cls[i] if o[3] == 'K' else o[4]].append(o[0] - ready)
        _sl[st] = o[1]
    elif kind == 'W':
        eid, st = waits[pl]
        # approximate: event point = end of last op on recording stream (unknown here); skip
        pass
log('  launch gap (op start - max(call end, prev same-stream end)) per class: n/f, p50 us, sum ms/f, share<0 (started before call end)')
gtot = 0
for cn, g in sorted(gaps.items(), key=lambda kv: -sum(max(x, 0) for x in kv[1])):
    g2 = sorted(g); pos = sum(max(x, 0) for x in g) / 1e6 / NF; gtot += pos
    log(f'     {cn:24s} n/f {len(g)/NF:7.2f} p50 {g2[len(g2)//2]/1e3:6.1f} sum {pos:7.4f}  neg {100*sum(1 for x in g if x < 0)/len(g):5.1f}%')
log(f'     total positive launch gap {gtot:.4f} ms/frame (includes cross-stream waits, so an upper figure)')
ne, sl, sh = simulate()
m0, a0 = frame_times(ne)
# validation: max abs error of reproduced end times
err = [abs(ne[i] - ops[i][1]) for i in ne]
# per-frame reproduction: compare simulated frame durations to the originals
ends = sorted((o[1], i) for i, o in enumerate(ops) if i in ne); keys = [x[0] for x in ends]
nb = []
for b in frame_bounds:
    j = bisect.bisect_right(keys, b) - 1
    nb.append(ne[ends[j][1]] if j >= 0 else b)
fe = [abs((nb[i+1]-nb[i]) - (frame_bounds[i+1]-frame_bounds[i]))/1e3 for i in range(len(nb)-1)]
log(f'  E0 replay unchanged: frame median {m0:.4f} mean {a0:.4f} (orig median {statistics.median(fw):.4f}); cumulative end-time drift p50 {statistics.median(err)/1e3:.0f} us; per-frame duration error p50 {statistics.median(fe):.1f} us p90 {sorted(fe)[len(fe)*9//10]:.1f} us')
results = {'E0': (m0, a0)}
def run(label, **kw):
    ne, sl, sh = simulate(**kw)
    m, a = frame_times(ne)
    results[label] = (m, a)
    log(f'  {label:40s} frame median {m:.4f} (Δ {m - m0:+.4f})  mean {a:.4f} (Δ {a - a0:+.4f})')
for pn in ['VS', 'FS_direct', 'clip_all', 'clip_rast_fused', 'stage3', 'stage2', 'fs_compact', 'fs_writeback', 'memcpy_HtoD', 'memcpy', 'memset', 'abuf_support', 'FS_abuf', 'stage3_abuf', 'vertex_fetch']:
    run(f'delete {pn}', delete=POOLS[pn])
run('launch gaps zero (front-end latency free)', launch_gap_scale=0.0)
run('launch gaps half', launch_gap_scale=0.5)
run('all kernel durations x0.5', dur_scale=lambda i: 0.5 if ops[i][3] == 'K' else 1.0)
run('all device durations x0.5', dur_scale=lambda i: 0.5)
run('all device durations x0 (device infinitely fast)', dur_scale=lambda i: 0.0, launch_gap_scale=0.0)
run('host gaps x0.5 (host twice as fast)', host_scale=0.5)
run('host gaps x0 (host infinitely fast)', host_scale=0.0)
run('free syncs (host never waits)', free_sync=True)
run('free syncs + host x0.5', free_sync=True, host_scale=0.5)
run('free syncs + launch gaps zero', free_sync=True, launch_gap_scale=0.0)
run('delete VS (also its host API time)', delete=POOLS['VS'], delete_host=True)
run('delete memcpy_HtoD (also its host API time)', delete=POOLS['memcpy_HtoD'], delete_host=True)
run('delete clip_all (also host API time)', delete=POOLS['clip_all'], delete_host=True)
GEO = lambda i: cls[i] in ('VS', 'cp_vertex_fetch') or cls[i] in CLIP
run('geometry run-ahead: VS/fetch/clip on a private lane', indep=GEO)
run('geometry run-ahead + launch gaps zero', indep=GEO, launch_gap_scale=0.0)
run('VS only on a private lane', indep=lambda i: cls[i] == 'VS')
run('delete VS+clip_all', delete=lambda i: POOLS['VS'](i) or POOLS['clip_all'](i))
run('delete VS+clip_all+stage2+stage3', delete=lambda i: POOLS['VS'](i) or POOLS['clip_all'](i) or POOLS['stage2'](i) or POOLS['stage3'](i))
run('delete memcpy+memset', delete=lambda i: ops[i][3] != 'K')
run('delete all kernels', delete=lambda i: ops[i][3] == 'K')

if OUT:
    json.dump({'tag': TAG, 'window': WIN, 'frames': NF, 'wall_mean': WALL / NF, 'wall_median': statistics.median(fw), 'union': U_all / NF, 'summed': S_all / NF, 'ue': ue, 'segres': segres, 'model': model, 'whatif': results, 'host_blocked': hb / NF, 'host_api': hapi / NF, 'bubble_sum': sum(bub_pos) / 1e6 / NF, 'tail_sum': sum(tails) / 1e6 / NF, 'concurrency_hist': {k: v / 1e6 / NF for k, v in hist.items()}}, open(OUT, 'w'), indent=1, default=str)
log(f'done in {time.time()-t0:.1f}s')
