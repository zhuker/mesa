#!/usr/bin/env python3
"""Read-only: discard-retry passes in a cudavk nsys trace.
usage: retry.py DB TSFILE SUB0 SUB1   (external submit index window, frame = 2 submits)
Identifies retry draws by the reject memset (w*h*8*4 bytes) on main, then walks the
following main-stream ops pass by pass (pass k>0 begins with the visbuf memset w*h*8)."""
import sqlite3, sys, statistics as st, bisect, collections
DB, TS, S0, S1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
sess = c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts = [int(l.split()[0]) - sess for l in open(TS) if len(l.split()) == 2]
w0, w1 = ts[S0], ts[S1]
NF = (S1 - S0) // 2
fb = [ts[S0 + 2 * k] for k in range(NF + 1)]
def fidx(t):
    i = bisect.bisect_right(fb, t) - 1
    return i if 0 <= i < NF else -1
names = dict(c.execute("select id, value from StringIds"))
MAIN = 13
ops = []
for r in c.execute("select start,end,streamId,shortName,gridX*gridY*gridZ,blockX*blockY*blockZ,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? and streamId=? order by start", (w0 - 5_000_000, w1, MAIN)):
    ops.append(dict(s=r[0], e=r[1], n=names[r[3]], g=r[4], b=r[5], cid=r[6], k='K'))
for r in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<? and streamId=? order by start", (w0 - 5_000_000, w1, MAIN)):
    ops.append(dict(s=r[0], e=r[1], n='memset', g=r[3], b=0, cid=r[4], k='S'))
for r in c.execute("select start,end,streamId,bytes,copyKind,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and streamId=? order by start", (w0 - 5_000_000, w1, MAIN)):
    ops.append(dict(s=r[0], e=r[1], n='memcpy%d' % r[4], g=r[3], b=0, cid=r[5], k='M'))
ops.sort(key=lambda o: o['s'])
api = {}
for r in c.execute("select r.start,r.end,r.correlationId,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.start<?", (w0 - 5_000_000, w1)):
    api[r[2]] = (r[0], r[1], r[3])
# all main ops in the window, for idle computation after the batch
starts = [o['s'] for o in ops]

W, H = 1280, 720
REJECT = W * H * 8 * 4
VISBUF = W * H * 8
RESOLVED = W * H
DMASK = ((W * H + 255) // 256 * 256) * 2
print("window frames", NF, "ns", w0, w1, "main ops", len(ops))
print("reject memset bytes", REJECT, "visbuf", VISBUF, "dmask", DMASK)
sizes = collections.Counter(o['g'] for o in ops if o['k'] == 'S')
print("memset size census (top 12):", sizes.most_common(12))

batches = []
i = 0
while i < len(ops):
    o = ops[i]
    if o['k'] == 'S' and o['g'] == REJECT and w0 <= o['s'] < w1:
        # a retry draw. pass 0 starts here (after resolved+reject memsets); collect until 8 writebacks
        # or until we see something that is not part of a pass.
        j = i + 1
        passes = []
        cur = dict(ops=[], idx=0)
        wbs = 0
        while j < len(ops):
            p = ops[j]
            if p['k'] == 'S' and p['g'] == VISBUF and cur['ops'] and any(x['n'] == 'cp_fs_writeback' for x in cur['ops']):
                passes.append(cur); cur = dict(ops=[], idx=len(passes))
                if len(passes) >= 8: break
            if p['k'] == 'S' and p['g'] == REJECT: break
            cur['ops'].append(p)
            if p['n'] == 'cp_fs_writeback':
                wbs += 1
                if wbs >= 8:
                    passes.append(cur); cur = None; j += 1; break
            j += 1
        if cur and cur['ops']: passes.append(cur)
        batches.append(dict(t=o['s'], f=fidx(o['s']), passes=passes, end=j))
        i = j
    else:
        i += 1

print("\nretry draws found:", len(batches), " per frame:", len(batches) / NF)
perf = collections.Counter(b['f'] for b in batches)
pf = [perf.get(k, 0) for k in range(NF)]
print("retry draws/frame: mean %.2f median %.1f p90 %.1f frames-with-0 %d" % (st.mean(pf), st.median(pf), sorted(pf)[int(0.9 * (NF - 1))], pf.count(0)))
print("passes per draw:", collections.Counter(len(b['passes']) for b in batches))

def dur(o): return (o['e'] - o['s']) / 1000.0
def find(p, n, which=0):
    l = [x for x in p['ops'] if x['n'] == n]
    return l[which] if len(l) > which else None
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for b in batches:
    for p in b['passes']:
        k = p['idx']
        if not p['ops']: continue
        pops = p['ops']
        rows[k]['nops'].append(len(pops))
        rows[k]['nK'].append(sum(1 for x in pops if x['k'] == 'K'))
        rows[k]['wall'].append((pops[-1]['e'] - pops[0]['s']) / 1000.0)
        rows[k]['sumdur'].append(sum(dur(x) for x in pops))
        # host issue span for the pass
        a = [api[x['cid']] for x in pops if x['cid'] in api]
        if a:
            rows[k]['host_span'].append((max(e for s, e, n in a) - min(s for s, e, n in a)) / 1000.0)
            rows[k]['host_sum'].append(sum((e - s) for s, e, n in a) / 1000.0)
            rows[k]['host_lead'].append((pops[0]['s'] - min(s for s, e, n in a)) / 1000.0)
        for n in ['cp_rasterize_stage1', 'cp_clip_rast_fused', 'cp_rasterize_stage2', 'cp_rasterize_stage3', 'cp_fs_compact', 'main', 'cp_fs_writeback']:
            x = find(p, n)
            if x: rows[k][n].append(dur(x)); rows[k][n + '_grid'].append(x['g'])
        # memsets in this pass
        rows[k]['memset_bytes'].append(sum(x['g'] for x in pops if x['k'] == 'S'))
        rows[k]['memset_dur'].append(sum(dur(x) for x in pops if x['k'] == 'S'))
        # gaps (idle on main inside the pass)
        gap = 0
        for a_, b_ in zip(pops, pops[1:]):
            if b_['s'] > a_['e']: gap += b_['s'] - a_['e']
        rows[k]['gap'].append(gap / 1000.0)
        # VS present in pass 0?
        rows[k]['names'].append(' '.join(('%s' % (x['n'] if x['k'] == 'K' else x['n'] + str(x['g']))) for x in pops))

def m(x): return st.median(x) if x else float('nan')
print("\nper-pass medians over %d draws (us):" % len(batches))
print("pass  nops nK  wall  sumdur  gap  host_span host_sum host_lead  s1   fused  s2    s3   compact  FS    wb   FSgrid  memsetB")
for k in sorted(rows):
    r = rows[k]
    print("%-4d %4.0f %3.0f %6.1f %6.1f %5.1f %8.1f %8.1f %8.1f  %5.1f %5.1f %5.1f %5.1f %6.1f %6.1f %5.1f %7.0f %9.0f" % (
        k, m(r['nops']), m(r['nK']), m(r['wall']), m(r['sumdur']), m(r['gap']), m(r['host_span']), m(r['host_sum']), m(r['host_lead']),
        m(r['cp_rasterize_stage1']), m(r['cp_clip_rast_fused']), m(r['cp_rasterize_stage2']), m(r['cp_rasterize_stage3']),
        m(r['cp_fs_compact']), m(r['main']), m(r['cp_fs_writeback']), m(r['main_grid']), m(r['memset_bytes'])))
print("\nFS duration distribution by pass (p10/p50/p90):")
for k in sorted(rows):
    x = sorted(rows[k]['main'])
    if x: print(k, "%.2f %.2f %.2f n=%d" % (x[len(x)//10], x[len(x)//2], x[9*len(x)//10], len(x)))
print("\nwriteback duration by pass (p10/p50/p90):")
for k in sorted(rows):
    x = sorted(rows[k]['cp_fs_writeback'])
    if x: print(k, "%.2f %.2f %.2f" % (x[len(x)//10], x[len(x)//2], x[9*len(x)//10]))
print("\ncompact duration by pass (p50):", [round(m(rows[k]['cp_fs_compact']), 2) for k in sorted(rows)])
print("stage1 grid by pass (p50):", [m(rows[k].get('cp_rasterize_stage1_grid', [])) for k in sorted(rows)])
print("\nexample pass op sequences (first draw):")
for p in batches[0]['passes'] if batches else []:
    print(" pass", p['idx'], ":", rows[p['idx']]['names'][0][:400])

# whole-draw wall: from reject memset to last writeback end; and passes 1..7 wall
tot = []; re = []; dead = []
for b in batches:
    if len(b['passes']) < 2: continue
    p0 = b['passes'][0]; pl = b['passes'][-1]
    tot.append((pl['ops'][-1]['e'] - b['t']) / 1000.0)
    re.append((pl['ops'][-1]['e'] - b['passes'][1]['ops'][0]['s']) / 1000.0)
print("\nwhole retry draw wall (reject memset -> last wb end) median %.1f us mean %.1f; passes1..N wall median %.1f mean %.1f" % (m(tot), st.mean(tot) if tot else 0, m(re), st.mean(re) if re else 0))
print("per-frame re-pass wall (sum over draws in frame) mean %.1f us, median over frames %.1f" % (
    sum(re) / NF, st.median([sum(r for b, r in zip([b for b in batches if len(b['passes']) >= 2], re) if b['f'] == k) for k in range(NF)])))
# what follows the batch: idle on main until next op?
after = []
for b in batches:
    if b['end'] < len(ops):
        last = b['passes'][-1]['ops'][-1]
        nxt = ops[b['end']] if b['end'] < len(ops) else None
        if nxt: after.append((nxt['s'] - last['e']) / 1000.0)
print("main idle after last pass writeback until next main op: median %.1f us, p90 %.1f" % (m(after), sorted(after)[int(0.9 * (len(after) - 1))] if after else 0))
# reject + resolved memset cost per draw
rm = [dur(o) for o in ops if o['k'] == 'S' and o['g'] == REJECT and w0 <= o['s'] < w1]
print("reject memset (29.5MB) duration median %.1f us; per frame %.1f us" % (m(rm), sum(rm) / NF))
