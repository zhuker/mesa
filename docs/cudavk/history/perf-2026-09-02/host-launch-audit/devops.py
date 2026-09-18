import sqlite3, statistics
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
cur=c.cursor()
SESSION=1503511535334936
sub=[int(l.split()[0])-SESSION for l in open('/tmp/postdead/submit-ts.txt')]
T0=sub[2785]; T1=12340525407
frames=sum(1 for t in sub if T0<=t<=T1)/2
def pct(v,p):
    v=sorted(v); k=int(round((len(v)-1)*p)); return v[k] if v else 0
def show(name, vals, div=1000.0, unit='us'):
    if not vals: print(name,"none"); return
    print(f"  {name}: n={len(vals)} n/f={len(vals)/frames:.2f} sum/f={sum(vals)/frames/div:.1f}{unit} mean={statistics.mean(vals)/div:.2f} p50={pct(vals,.5)/div:.2f} p90={pct(vals,.9)/div:.2f} p99={pct(vals,.99)/div:.2f} max={max(vals)/div:.1f}")
print("frames",frames)
print("\n== MEMCPY device ops in window by API/copyKind/srcKind/dstKind ==")
rows=cur.execute(f"""select s.value, m.copyKind, m.srcKind, m.dstKind, m.bytes, m.end-m.start, r.end-r.start, m.streamId
 from CUPTI_ACTIVITY_KIND_MEMCPY m left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=m.correlationId
 left join StringIds s on s.id=r.nameId where m.start>={T0} and m.start<{T1}""").fetchall()
from collections import defaultdict
g=defaultdict(list)
for api,ck,sk,dk,b,dd,hd,st in rows: g[(api,ck,sk,dk)].append((b,dd,hd,st))
for k,v in sorted(g.items(), key=lambda kv:-len(kv[1])):
    print(k, "n/f=%.2f"%(len(v)/frames))
    show("bytes", [x[0] for x in v], 1.0, 'B')
    show("dev_dur", [x[1] for x in v])
    show("host_dur", [x[2] for x in v if x[2] is not None])
    streams=defaultdict(int)
    for x in v: streams[x[3]]+=1
    print("   streams:", dict(streams))
print("\n== HtoD async size histogram (bytes) ==")
import math
h=defaultdict(int)
for api,ck,sk,dk,b,dd,hd,st in rows:
    if api and api.startswith('cuMemcpyHtoDAsync'):
        bucket = 1<<max(8, math.ceil(math.log2(max(b,1))))
        h[bucket]+=1
for k in sorted(h): print(f"  <= {k:8d} B: {h[k]:7d}  {h[k]/frames:8.2f}/frame")
print("\n== MEMSET device ops in window ==")
rows=cur.execute(f"""select s.value, m.bytes, m.value, m.end-m.start, r.end-r.start, m.streamId
 from CUPTI_ACTIVITY_KIND_MEMSET m left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=m.correlationId
 left join StringIds s on s.id=r.nameId where m.start>={T0} and m.start<{T1}""").fetchall()
g=defaultdict(list)
for api,b,val,dd,hd,st in rows: g[api].append((b,dd,hd,val,st))
for k,v in g.items():
    print(k, "n/f=%.2f"%(len(v)/frames))
    show("bytes",[x[0] for x in v],1.0,'B'); show("dev_dur",[x[1] for x in v]); show("host_dur",[x[2] for x in v if x[2] is not None])
    hb=defaultdict(int)
    for x in v: hb[(x[0], hex(x[3]))]+=1
    top=sorted(hb.items(), key=lambda kv:-kv[1])[:14]
    for (b,val),n in top: print(f"    bytes={b:9d} value={val:>10s} n/f={n/frames:7.2f}")
print("\n== KERNEL in window ==")
rows=cur.execute(f"""select k.streamId, k.launchType, count(*), sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k where k.start>={T0} and k.start<{T1} group by 1,2""").fetchall()
tot=0
for st,lt,n,d in rows: tot+=n; print(f"  stream {st} launchType {lt}: n/f={n/frames:.2f} dev_sum/f={d/frames/1000:.1f} us")
print("  total kernels/frame", tot/frames)
print("\n== SYNCHRONIZATION in window ==")
rows=cur.execute(f"""select syncType, count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>={T0} and start<{T1} group by 1""").fetchall()
for st,n,d in rows: print(f"  type {st}: n/f={n/frames:.2f} sum/f={d/frames/1000:.1f} us")
print("\n== CUDA_EVENT (event records) in window by stream ==")
rows=cur.execute(f"""select e.streamId, count(*) from CUPTI_ACTIVITY_KIND_CUDA_EVENT e join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=e.correlationId where r.start>={T0} and r.start<{T1} group by 1""").fetchall()
for st,n in rows: print(f"  stream {st}: n/f={n/frames:.2f}")
print("\n== cuStreamWaitEvent: SYNC rows type 2 by stream ==")
rows=cur.execute(f"""select streamId, count(*) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2 and start>={T0} and start<{T1} group by 1""").fetchall()
for st,n in rows: print(f"  stream {st}: n/f={n/frames:.2f}")
