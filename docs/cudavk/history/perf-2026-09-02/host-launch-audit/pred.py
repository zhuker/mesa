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
    return f"n={len(v)} n/f={len(v)/frames:.2f} mean={statistics.mean(v)/1000:.2f} p10={pct(v,.1)/1000:.2f} p50={pct(v,.5)/1000:.2f} p90={pct(v,.9)/1000:.2f} p99={pct(v,.99)/1000:.2f} us"
ops=[]
# kernels with API name and short name
for st,s,e,cid,name,gx,corr_api in cur.execute(f"""select k.streamId,k.start,k.end,k.correlationId,sn.value,k.gridX,ra.value
  from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sn on sn.id=k.shortName
  left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=k.correlationId left join StringIds ra on ra.id=r.nameId
  where k.start>={T0-50000000} and k.start<{T1}"""):
    ops.append((st,s,e,'K',name,gx,corr_api,cid))
for st,s,e,cid,ck,b in cur.execute(f"""select streamId,start,end,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind not in (11,12) and start>={T0-50000000} and start<{T1}"""):
    ops.append((st,s,e,'C',{1:'HtoD',2:'DtoH',8:'DtoD',7:'DtoA'}.get(ck,str(ck)),b,None,cid))
for st,s,e,cid,b in cur.execute(f"""select streamId,start,end,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where start>={T0-50000000} and start<{T1}"""):
    ops.append((st,s,e,'S','memset',b,None,cid))
bystream=defaultdict(list)
for o in ops: bystream[o[0]].append(o)
cls=defaultdict(list)       # (api, predkind) -> gaps
names_after_htod=defaultdict(int)
names_plain=defaultdict(int)
chain=defaultdict(list)     # for kernel after HtoD: (prevprev.end -> memcpy.start), (memcpy.end->k.start), total
memcpy_after=defaultdict(int)
for st,lst in bystream.items():
    lst.sort(key=lambda o:o[1])
    for i,o in enumerate(lst):
        if o[3]!='K' or o[1]<T0: continue
        if i==0: continue
        p=lst[i-1]
        api='Ex' if (o[6] or '').startswith('cuLaunchKernelEx') else 'plain'
        pk = p[3] if p[3]!='C' else 'C_'+p[4]
        gap=o[1]-p[2]
        cls[(api,pk)].append(gap)
        if pk=='C_HtoD':
            names_after_htod[(o[4],api)]+=1
            if i>=2:
                pp=lst[i-2]
                chain['pp_end->cpy_start'].append(p[1]-pp[2])
                chain['cpy_dur'].append(p[2]-p[1])
                chain['cpy_end->k_start'].append(o[1]-p[2])
                chain['pp_end->k_start(total)'].append(o[1]-pp[2])
                chain['ppkind:'+pp[3]].append(0)
        elif pk=='K':
            names_plain[(o[4],api)]+=1
    for i,o in enumerate(lst):
        if o[3]=='C' and o[4]=='HtoD' and o[1]>=T0 and i>0:
            memcpy_after[lst[i-1][3]]+=1
print("frames",frames)
print("\n== kernel start gap to predecessor op on same stream, by launch API and predecessor kind ==")
for k in sorted(cls, key=lambda k:-len(cls[k])):
    v=cls[k]; neg=sum(1 for x in v if x<0)
    print(f"  {k[0]:5s} after {k[1]:8s}: {stats(v)}  negative(overlap)={neg/len(v)*100:.1f}%")
print("\n== kernels launched right after an HtoD copy on the same stream (name, api) per frame ==")
for k,n in sorted(names_after_htod.items(), key=lambda kv:-kv[1])[:20]: print(f"  {k[0]:32s} {k[1]:5s} {n/frames:7.2f}")
print("\n== kernels launched right after a kernel (name, api) per frame ==")
for k,n in sorted(names_plain.items(), key=lambda kv:-kv[1])[:30]: print(f"  {k[0]:32s} {k[1]:5s} {n/frames:7.2f}")
print("\n== chain around an HtoD copy between two ops ==")
for k in ['pp_end->cpy_start','cpy_dur','cpy_end->k_start','pp_end->k_start(total)']: print(f"  {k:24s} {stats(chain[k])}")
for k in chain:
    if k.startswith('ppkind'): print(f"  {k}: n/f={len(chain[k])/frames:.2f}")
print("\n== what precedes an HtoD copy on its stream ==", {k:round(v/frames,2) for k,v in memcpy_after.items()})
# host-order: launches immediately preceded by HtoD on the issuing thread
print("\n== host order on main thread: launch API preceded by which API ==")
rows=cur.execute(f"""select s.value, r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.globalTid=297865344641070 and r.start>={T0} and r.start<{T1} order by r.start""").fetchall()
prev=None; cnt=defaultdict(int)
for name,st in rows:
    if name.startswith('cuLaunchKernel'):
        cnt[(name, prev)]+=1
    prev=name
tot=defaultdict(int)
for (n,p),v in cnt.items(): tot[n]+=v
for k,v in sorted(cnt.items(), key=lambda kv:-kv[1])[:16]: print(f"  {k[0]:18s} after {str(k[1]):24s} {v/frames:7.2f}/f  ({v/tot[k[0]]*100:.1f}% of that API)")
