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

# ---------------------------------------------------------------- verify2-G additions
# 1. classify each main-thread blocking call by site
#    S1 episode drain: cuStreamSynchronize followed (within 5 host rows) by a sync DtoH of 4*(6+nsegs) bytes (28..280, !=24)
#    S2 segment counters: ... DtoH of 24 bytes
#    S3 peel check: cuStreamSynchronize with no DtoH following
#    S4 cuCtxSynchronize (scratch reclaim / upload wrap / destroy drains)
#    S6 sync cuMemcpyHtoD_v2 (descriptor upload)
site_of = {}   # index in H -> site
Hn = len(H)
for j, (s, e, kind, pl) in enumerate(H):
    if kind == 'B':
        n, c = pl
        if n == 'cuCtxSynchronize':
            site_of[j] = 'S4'
        elif n == 'cuStreamSynchronize':
            site = 'S3'
            for k in range(j + 1, min(j + 6, Hn)):
                s2, e2, k2, p2 = H[k]
                if k2 == 'B':
                    break
                if k2 == 'LB':
                    o = ops[corr_to_op[p2]]
                    if o[3] == 'C' and o[4] == 'memcpy2':
                        site = 'S2' if o[5] == 24 else 'S1'
                        site_of[k] = site + 'copy'
                        break
            site_of[j] = site
        else:
            site_of[j] = 'Sother:' + n
    elif kind == 'LB':
        o = ops[corr_to_op[pl]]
        if o[3] == 'C' and o[4] == 'memcpy1':
            site_of[j] = 'S6'
        elif j not in site_of:
            site_of[j] = 'LBother:' + o[4]
cnt = collections.Counter(site_of.values())
blk = collections.defaultdict(float)
for j, st in site_of.items():
    s, e, kind, pl = H[j]
    if W0 <= e <= W1:
        blk[st] += (e - s)
log('site census (count/frame, blocked ms/frame, CUPTI-inflated):')
for st in sorted(cnt):
    log(f'   {st:14s} {cnt[st]/NF:7.2f}  {blk[st]/1e6/NF:8.4f}')

# 2. readback-fence dependency: for each worker cuEventSynchronize whose waited op is the 3,686,400-byte copy,
#    the first main-thread call after its return cannot start before that op's NEW end + orig (return + app) gap
op_ends_sorted = sorted((o[1], i) for i, o in enumerate(ops))
oe_keys = [x[0] for x in op_ends_sorted]
H_starts = [h[0] for h in H]
fence_dep = {}   # H index -> (op index, gap_ns from op orig end to first call start)
nfence = 0; napp = []
for ws, we, c in wsync:
    if we < W0 or we > W1:
        continue
    k = bisect.bisect_right(oe_keys, we) - 1
    if k < 0:
        continue
    oi = op_ends_sorted[k][1]
    o = ops[oi]
    if not (o[3] == 'C' and o[5] == 3686400):
        continue
    j = bisect.bisect_left(H_starts, we)
    if j >= Hn:
        continue
    fence_dep[j] = (oi, H[j][0] - o[1])
    nfence += 1; napp.append((H[j][0] - we) / 1e3)
log(f'readback fences in window {nfence} ({nfence/NF:.2f}/frame); app gap fence-return->first CUDA call: median {statistics.median(napp):.0f} us p10 {sorted(napp)[len(napp)//10]:.0f} p90 {sorted(napp)[len(napp)*9//10]:.0f}; negative {sum(1 for x in napp if x < 0)}')

def simulate(delete=lambda i: False, free_sync=False, free_sites=None, fence=False, launch_gap_scale=1.0, dur_scale=lambda i: 1.0, host_scale=1.0, extra_dev=lambda j: 0.0):
    """free_sync: every blocking call returns after its own API residual (min(.,3us));
       free_sites: set of site labels freed instead (others keep waiting);
       fence: keep the app's readback-fence dependency;
       extra_dev(j): ns of extra critical-path device time inserted after the stream's last op when blocking call j is freed (mechanism price)."""
    new_end_op = {}; stream_last = {}; stream_last_orig = {}; event_point = {}
    hclock_new = None; last_e = None
    def freed(j):
        if free_sync: return True
        if free_sites is None: return False
        st = site_of.get(j, '')
        return st.rstrip('copy') in free_sites or st in free_sites
    for j, (s, e, kind, pl) in enumerate(H):
        if hclock_new is None:
            t_new = s
        else:
            t_new = hclock_new + max(s - last_e, 0) * host_scale
        if fence and j in fence_dep:
            oi, gap = fence_dep[j]
            if oi in new_end_op:
                t_new = max(t_new, new_end_op[oi] + gap * host_scale if host_scale > 0 else new_end_op[oi] + gap)
        last_e = e
        if kind == 'L' or kind == 'LB':
            i = corr_to_op[pl]; o = ops[i]; st = o[2]
            api = (e - s) * host_scale
            issue_new = t_new + api
            ready_new = max(issue_new, stream_last.get(st, 0))
            ready_orig = max(e, stream_last_orig.get(st, 0))
            gap = o[0] - ready_orig
            gap = gap * launch_gap_scale if gap > 0 else gap
            if delete(i):
                new_end_op[i] = ready_new; stream_last_orig[st] = o[1]; ne_op = ready_new
            else:
                ns = ready_new + gap
                ne_op = ns + (o[1] - o[0]) * dur_scale(i)
                new_end_op[i] = ne_op; stream_last_orig[st] = o[1]
                stream_last[st] = max(stream_last.get(st, 0), ne_op)
            if kind == 'LB':
                resid = max(e - o[1], 0)
                if freed(j):
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
                dev_new = stream_last.get(st, t_new); dev_orig = stream_last_orig.get(st, s)
            else:
                st = None
                dev_new = max(stream_last.values()) if stream_last else t_new
                dev_orig = max(stream_last_orig.values()) if stream_last_orig else s
            resid = max(e - max(s, dev_orig), 0)
            if freed(j):
                hclock_new = t_new + min(resid, 3000)
                x = extra_dev(j)
                if x > 0 and st is not None:
                    stream_last[st] = stream_last.get(st, 0) + x
            else:
                hclock_new = max(t_new, dev_new) + resid
    return new_end_op

def frame_times(new_end_op):
    ends = sorted((o[1], i) for i, o in enumerate(ops) if i in new_end_op)
    keys = [x[0] for x in ends]
    nb = []
    for b in frame_bounds:
        j = bisect.bisect_right(keys, b) - 1
        nb.append(new_end_op[ends[j][1]] if j >= 0 else b)
    fw_ = [(nb[i + 1] - nb[i]) / 1e6 for i in range(len(nb) - 1)]
    return statistics.median(fw_), (nb[-1] - nb[0]) / 1e6 / (len(nb) - 1)

results = {}
m0, a0 = frame_times(simulate())
log(f'E0 unchanged: median {m0:.4f} mean {a0:.4f} (orig median {statistics.median(fw):.4f} mean {WALL/NF:.4f})')
mf, af = frame_times(simulate(fence=True))
log(f'E0 + fence dep (must equal E0): median {mf:.4f} mean {af:.4f}')
def run(label, **kw):
    m, a = frame_times(simulate(**kw))
    results[label] = (m, a, m - m0, a - a0)
    log(f'  {label:58s} median {m:.4f} (d {m-m0:+.4f})  mean {a:.4f} (d {a-a0:+.4f})')
run('free syncs, NO fence dep (closure-audit)', free_sync=True)
run('free syncs, fence kept', free_sync=True, fence=True)
run('free syncs, fence kept, host x0.5', free_sync=True, fence=True, host_scale=0.5)
run('free syncs, fence kept, gaps zero', free_sync=True, fence=True, launch_gap_scale=0.0)
run('host infinitely fast, fence kept', host_scale=0.0, fence=True)
for sites in (['S1'], ['S2'], ['S3'], ['S4'], ['S6'], ['S1','S2'], ['S1','S2','S4'], ['S1','S2','S3'], ['S1','S2','S3','S4'], ['S1','S2','S3','S4','S6']):
    run(f'free {"+".join(sites)}, fence kept', free_sites=set(sites), fence=True)
    run(f'free {"+".join(sites)}, NO fence dep', free_sites=set(sites))
# mechanism price: one device-side decision op on the freed stream per freed S1/S2 site
for price_us in (2.0, 7.3, 10.3):
    run(f'free S1+S2+S3+S4, fence kept, +{price_us} us device per freed S1/S2/S3', free_sites={'S1','S2','S3','S4'}, fence=True,
        extra_dev=lambda j, p=price_us: p * 1000.0 if site_of.get(j, '') in ('S1', 'S2', 'S3') else 0.0)
if OUT:
    json.dump(results, open(OUT, 'w'), indent=1)
log(f'done in {time.time()-t0:.0f}s')
