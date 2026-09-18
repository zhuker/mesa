#!/usr/bin/env python3
"""Read-only: is the retry draw's device time on the critical path?
For each retry draw (reject memset on main): side-stream busy fraction during passes 1-7,
main-stream idle from draw end to the return of the host's next blocking call, which call it is,
how long the host blocked there, and the frame pattern of retry draws."""
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
allops = []
for r in c.execute("select start,end,streamId,shortName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start", (w0 - 5_000_000, w1 + 5_000_000)):
    allops.append((r[0], r[1], r[2], names[r[3]], r[4]))
for r in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<? order by start", (w0 - 5_000_000, w1 + 5_000_000)):
    allops.append((r[0], r[1], r[2], 'memset%d' % r[3], r[4]))
for r in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and streamId!=0 order by start", (w0 - 5_000_000, w1 + 5_000_000)):
    allops.append((r[0], r[1], r[2], 'memcpy%d' % r[3], r[4]))
allops.sort()
main = [o for o in allops if o[2] == MAIN]
side = [o for o in allops if o[2] != MAIN]
ms = [o[0] for o in main]; ss = [o[0] for o in side]
SYNC = ('cuStreamSynchronize', 'cuCtxSynchronize', 'cuEventSynchronize', 'cuMemcpyDtoH_v2', 'cuMemcpyDtoH', 'cuStreamQuery', 'cuEventQuery')
rt = c.execute("select r.start,r.end,s.value,r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.start<? order by r.start", (w0 - 5_000_000, w1 + 5_000_000)).fetchall()
api = {r[3]: r for r in rt}
syncs = [r for r in rt if any(r[2].startswith(s) for s in SYNC)]
sst = [r[0] for r in syncs]
print("sync-ish API names in window:", collections.Counter(r[2] for r in syncs).most_common(8))

def union_busy(l, a, b, starts):
    i = bisect.bisect_left(starts, a - 20_000_000); j = bisect.bisect_left(starts, b)
    iv = sorted((max(o[0], a), min(o[1], b)) for o in l[i:j] if o[1] > a and o[0] < b)
    tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot

REJECT = 1280 * 720 * 32
draws = []
for idx, o in enumerate(main):
    if o[3] == 'memset%d' % REJECT and w0 <= o[0] < w1:
        # find the 8 writebacks that follow
        wbs = [p for p in main[idx:idx + 120] if p[3] == 'cp_fs_writeback'][:8]
        if len(wbs) < 8: continue
        draws.append(dict(i=idx, t=o[0], f=fidx(o[0]), wb=wbs, p1=None))
        # start of pass 1 = the first visbuf memset after the first writeback
        for p in main[idx:idx + 120]:
            if p[0] > wbs[0][1] and p[3] == 'memset7372800':
                draws[-1]['p1'] = p[0]; break
print("draws", len(draws), "frames", NF)
frames = sorted(set(d['f'] for d in draws))
pat = ''.join('1' if k in set(frames) else '0' for k in range(NF))
print("frame pattern (first 120 frames):", pat[:120])
runs = collections.Counter()
cur = pat[0]; n = 0
for ch in pat:
    if ch == cur: n += 1
    else: runs[(cur, n)] += 1; cur = ch; n = 1
print("run-length census of retry/no-retry frames (top):", runs.most_common(10))

sidefrac = []; idle_to_sync = []; blocked = []; which = collections.Counter(); dist = []; mainbusy_frac = []
for d in draws:
    if d['p1'] is None: continue
    a, b = d['p1'], d['wb'][-1][1]
    sidefrac.append(union_busy(side, a, b, ss) / (b - a))
    # the host's next blocking call issued after the launch of the last writeback
    last_api = api.get(d['wb'][-1][4])
    if not last_api: continue
    j = bisect.bisect_left(sst, last_api[0])
    # skip zero-length queries; want a call that actually blocked >= 20us or is a sync
    while j < len(syncs) and (syncs[j][1] - syncs[j][0]) < 5_000: j += 1
    if j >= len(syncs): continue
    s = syncs[j]
    which[s[2]] += 1
    blocked.append((s[1] - s[0]) / 1000.0)
    dist.append((s[1] - b) / 1000.0)   # device time from draw end to sync return
    if s[1] > b:
        idle_to_sync.append(((s[1] - b) - union_busy(main, b, s[1], ms)) / 1000.0)
        mainbusy_frac.append(union_busy(main, b, s[1], ms) / (s[1] - b))
def m(x): return st.median(x) if x else float('nan')
print("side-stream busy fraction during passes 1-7: median %.3f p90 %.3f" % (m(sidefrac), sorted(sidefrac)[int(0.9 * (len(sidefrac) - 1))]))
print("next blocking host call after the draw:", which.most_common(6))
print("host blocked there (us): median %.0f mean %.0f" % (m(blocked), st.mean(blocked)))
print("device time from draw end to sync return (us): median %.0f" % m(dist))
print("main idle (us) between draw end and sync return: median %.1f p10 %.1f p90 %.1f" % (m(idle_to_sync), sorted(idle_to_sync)[len(idle_to_sync)//10], sorted(idle_to_sync)[int(0.9*(len(idle_to_sync)-1))]))
print("main busy fraction between draw end and sync return: median %.3f" % m(mainbusy_frac))
# host lead at draw start (device start of reject memset - host API start)
lead = [(d['t'] - api[main[d['i']][4]][0]) / 1000.0 for d in draws if main[d['i']][4] in api]
print("host lead at retry draw start (us): median %.0f p10 %.0f p90 %.0f" % (m(lead), sorted(lead)[len(lead)//10], sorted(lead)[int(0.9*(len(lead)-1))]))
# frame time of retry frames vs non-retry frames (from submit ts)
ft = [(fb[k + 1] - fb[k]) / 1e6 for k in range(NF)]
fr = set(frames)
a = [ft[k] for k in range(NF) if k in fr]; b_ = [ft[k] for k in range(NF) if k not in fr]
print("frame time (ms) retry frames: median %.3f mean %.3f n=%d | non-retry: median %.3f mean %.3f n=%d | all median %.3f" % (m(a), st.mean(a), len(a), m(b_), st.mean(b_), len(b_), m(ft)))
