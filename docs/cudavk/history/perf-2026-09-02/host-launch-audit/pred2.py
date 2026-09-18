import sqlite3, statistics
from collections import defaultdict
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
cur=c.cursor()
SESSION=1503511535334936
sub=[int(l.split()[0])-SESSION for l in open('/tmp/postdead/submit-ts.txt')]
T0=sub[2785]; T1=12340525407
frames=sum(1 for t in sub if T0<=t<=T1)/2
def pct(v,p):
    v=sorted(v); k=int(round((len(v)-1)*p)); return v[k] if v else 0
def stats(v):
    if not v: return "none"
    return f"n={len(v)} n/f={len(v)/frames:.2f} mean={statistics.mean(v)/1000:.2f} p10={pct(v,.1)/1000:.2f} p50={pct(v,.5)/1000:.2f} p90={pct(v,.9)/1000:.2f} us"
# runtime rows indexed by correlationId -> (api, hstart, hend)
rt={}
for cid,name,s,e in cur.execute(f"select r.correlationId, sn.value, r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds sn on sn.id=r.nameId where r.start>={T0-100000000} and r.start<{T1}"):
    rt[cid]=(name,s,e)
ops=[]
for st,s,e,cid,name,gx in cur.execute(f"""select k.streamId,k.start,k.end,k.correlationId,sn.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sn on sn.id=k.shortName where k.start>={T0-50000000} and k.start<{T1}"""):
    ops.append((st,s,e,'K',name,gx,cid))
for st,s,e,cid,ck,b in cur.execute(f"""select streamId,start,end,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind not in (11,12) and start>={T0-50000000} and start<{T1}"""):
    ops.append((st,s,e,'C',{1:'HtoD',2:'DtoH',8:'DtoD',7:'DtoA'}.get(ck,str(ck)),b,cid))
for st,s,e,cid,b in cur.execute(f"""select streamId,start,end,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where start>={T0-50000000} and start<{T1}"""):
    ops.append((st,s,e,'S','memset',b,cid))
bystream=defaultdict(list)
for o in ops: bystream[o[0]].append(o)
res=defaultdict(list)
names_dev=defaultdict(int)
for st,lst in bystream.items():
    lst.sort(key=lambda o:o[1])
    for i,o in enumerate(lst):
        if o[3]!='K' or o[1]<T0 or i<1: continue
        p=lst[i-1]
        r=rt.get(o[6]); 
        if not r: continue
        api='Ex' if r[0].startswith('cuLaunchKernelEx') else 'plain'
        if p[3]=='K':
            paced = 'devpaced' if r[2] < p[2] else 'hostpaced'   # launch API returned before predecessor kernel ended
            res[(api,'K',paced)].append(o[1]-p[2])
        elif p[3]=='C' and p[4]=='HtoD' and i>=2:
            pp=lst[i-2]
            paced = 'devpaced' if r[2] < pp[2] else 'hostpaced'
            res[(api,'HtoD-chain',paced)].append(o[1]-pp[2])
            res[(api,'HtoD-chain:pp_end->cpy_start',paced)].append(p[1]-pp[2])
            res[(api,'HtoD-chain:cpy_dur',paced)].append(p[2]-p[1])
            res[(api,'HtoD-chain:cpy_end->k_start',paced)].append(o[1]-p[2])
            if paced=='devpaced': names_dev[o[4]]+=1
        elif p[3]=='S' and i>=2:
            pp=lst[i-2]
            paced = 'devpaced' if r[2] < pp[2] else 'hostpaced'
            res[(api,'memset-chain',paced)].append(o[1]-pp[2])
print("frames",frames)
print("== gap from predecessor op end to kernel start, split by whether the launch call had already returned before the predecessor ended (device-paced) ==")
for k in sorted(res):
    print(f"  {k[0]:5s} {k[1]:34s} {k[2]:9s}: {stats(res[k])}")
print("\n== device-paced kernels after HtoD by name /frame ==", {k:round(v/frames,2) for k,v in names_dev.items()})
# Event records: matched by a later stream-wait or event-sync
print("\n== event records: waited on or not ==")
ev=cur.execute(f"""select e.eventId, e.streamId, r.start from CUPTI_ACTIVITY_KIND_CUDA_EVENT e join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=e.correlationId where r.start>={T0} and r.start<{T1}""").fetchall()
waits=defaultdict(list)
for eid,st,s,typ in cur.execute(f"""select eventId, streamId, start, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType in (1,2) and start>={T0-1000000} and start<{T1+100000000}"""):
    waits[eid].append((s,typ,st))
m=defaultdict(int)
for eid,st,s in ev:
    w=[x for x in waits.get(eid,[]) if x[0]>=s and x[0]-s<50_000_000]
    m[(st,'waited' if w else 'never-waited')]+=1
for k in sorted(m): print(f"  record stream {k[0]:3d} {k[1]:13s} {m[k]/frames:7.2f}/f")
# distinct eventIds per frame (how many event objects are cycled)
print("  distinct eventIds in window:", len(set(e[0] for e in ev)))
