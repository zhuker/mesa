"""MEASURED bubble price of an existing host sync: for each main-thread cuStreamSynchronize in the
heavy window, main-stream gap = start(first main device op whose launch API began after the sync returned)
- end(last main device op ending before the sync returned). Also split by whether the sync is a peel check
(no DtoH follows within 5 rows) or a drain (DtoH follows)."""
import sqlite3, statistics as st, bisect, collections
DB, TS, S0, S1 = '/tmp/postpin-rtx2/trace.sqlite', '/tmp/postpin-rtx2/ts.txt', 4900, 6300
c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
sess = c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts = [int(l.split()[0]) - sess for l in open(TS) if len(l.split()) == 2]
w0, w1 = ts[S0], ts[S1]
names = dict(c.execute("select id, value from StringIds"))
ops = []
for t, q in [('CUPTI_ACTIVITY_KIND_KERNEL', "select start,end,correlationId from %s where streamId=13 and start>=? and start<?"),
             ('CUPTI_ACTIVITY_KIND_MEMSET', "select start,end,correlationId from %s where streamId=13 and start>=? and start<?"),
             ('CUPTI_ACTIVITY_KIND_MEMCPY', "select start,end,correlationId from %s where streamId=13 and start>=? and start<?")]:
    ops += c.execute(q % t, (w0, w1)).fetchall()
ops.sort()
ends = sorted(e for s, e, cid in ops)
bycid = {cid: (s, e) for s, e, cid in ops}
rt = c.execute("select r.start,r.end,s.value,r.correlationId,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.start<? order by r.start", (w0, w1)).fetchall()
tid = collections.Counter(r[4] for r in rt).most_common(1)[0][0]
rt = [r for r in rt if r[4] == tid]
res = collections.defaultdict(list); blocked = collections.defaultdict(list)
for i, r in enumerate(rt):
    if r[2] != 'cuStreamSynchronize': continue
    kind = 'drain(DtoH)' if any(x[2].startswith('cuMemcpyDtoH') for x in rt[i+1:i+6]) else 'peel/no-DtoH'
    # first device op launched after the sync returned
    nxt = None
    for x in rt[i+1:i+40]:
        if x[3] in bycid and x[0] >= r[1]:
            nxt = bycid[x[3]]; break
    if not nxt: continue
    j = bisect.bisect_right(ends, r[1]) - 1
    if j < 0: continue
    last_end = ends[j]
    res[kind].append((nxt[0] - last_end) / 1000.0)
    blocked[kind].append((r[1] - r[0]) / 1000.0)
    res[kind + ':sync_return_lag'].append((r[1] - last_end) / 1000.0)
    res[kind + ':host_issue_after'].append((nxt[0] - r[1]) / 1000.0)
for k in sorted(res):
    x = sorted(res[k])
    print("%-32s n=%5d  p10 %6.1f  p50 %6.1f  p90 %6.1f us" % (k, len(x), x[len(x)//10], x[len(x)//2], x[9*len(x)//10]))
for k in blocked:
    x = sorted(blocked[k]); print("blocked %-20s p50 %.0f p90 %.0f us" % (k, x[len(x)//2], x[9*len(x)//10]))
