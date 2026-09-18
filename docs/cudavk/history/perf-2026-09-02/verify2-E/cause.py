import sqlite3, statistics, bisect
from collections import defaultdict, Counter
def pct(v,p):
    v=sorted(v); return v[int(round((len(v)-1)*p))] if v else 0
for db,T0,T1,fr,lab in [('/tmp/postpin-rtx2/trace.sqlite',18629876,8598737501,1024,'postpin'),('/tmp/postdead/trace.sqlite',9286588499,12340525407,411.5,'postdead')]:
    c=sqlite3.connect('file:'+db+'?mode=ro',uri=True)
    ops=[]
    for st,s,e,name,ae in c.execute(f'''select k.streamId,k.start,k.end,sn.value,r.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sn on sn.id=k.shortName
        left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=k.correlationId where k.start>={T0-5000000} and k.start<{T1}'''):
        ops.append((st,s,e,'K',name,0,ae))
    for st,s,e,ck,b,ae in c.execute(f'''select m.streamId,m.start,m.end,m.copyKind,m.bytes,r.end from CUPTI_ACTIVITY_KIND_MEMCPY m left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=m.correlationId where m.copyKind not in (11,12) and m.start>={T0-5000000} and m.start<{T1}'''):
        ops.append((st,s,e,'C',f'memcpy{ck}',b,ae))
    for st,s,e,b,ae in c.execute(f'''select m.streamId,m.start,m.end,m.bytes,r.end from CUPTI_ACTIVITY_KIND_MEMSET m left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=m.correlationId where m.start>={T0-5000000} and m.start<{T1}'''):
        ops.append((st,s,e,'S','memset',b,ae))
    # stream waits: cuStreamWaitEvent rows (syncType=2) with stream and event; event records
    waits=defaultdict(list)  # stream -> list of (api_start, eventId)
    for s,e,st,cid,eid in c.execute('select start,end,streamId,correlationId,eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2'):
        waits[st].append((s,eid))
    recs={}  # eventId -> list of (timestamp, stream)
    rec_by_id=defaultdict(list)
    for ts,st,cid,eid in c.execute('select timestamp,streamId,correlationId,eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
        rec_by_id[eid].append((ts,st))
    for k in rec_by_id: rec_by_id[k].sort()
    bys=defaultdict(list)
    for o in ops: bys[o[0]].append(o)
    for st in bys: bys[st].sort(key=lambda o:o[1])
    allc=sorted([o for o in ops if o[3]=='C'], key=lambda o:o[2])  # by end
    cends=[o[2] for o in allc]
    allk=sorted([o for o in ops if o[3]=='K'], key=lambda o:o[2]); kends=[o[2] for o in allk]
    for st in waits: waits[st].sort()
    def cls_of(lst,i):
        k=lst[i]; 
        if k[4]!='main': return k[4]
        nxt=next((lst[j][4] for j in range(i+1,min(i+4,len(lst))) if lst[j][3]=='K'),None)
        prv=next((lst[j][4] for j in range(i-1,max(i-6,-1),-1) if lst[j][3]=='K'),None)
        if prv and ('fs_compact' in prv or 'fs_interp' in prv): return 'main_FS'
        if nxt and nxt.startswith('cp_clip'): return 'main_VS'
        return 'main_misc'
    res=defaultdict(lambda: defaultdict(float)); cnt=defaultdict(lambda: defaultdict(int))
    for st,lst in bys.items():
        for i,o in enumerate(lst):
            if not(o[3]=='C' and o[4]=='memcpy1' and o[1]>=T0): continue
            nk=next((j for j in range(i+1,min(i+6,len(lst))) if lst[j][3]=='K'),None)
            cl=cls_of(lst,nk) if nk is not None else 'none'
            pool='shade' if cl in ('main_FS','cp_fs_compact') else 'VS' if cl=='main_VS' else 'other'
            prev_end=lst[i-1][2] if i>0 else 0
            ready=max(o[6] or 0, prev_end)
            gap=o[1]-ready
            cnt[pool]['n']+=1; res[pool]['gap_sum']+=gap; res[pool]['dur_sum']+=o[2]-o[1]
            if gap<10000: res[pool]['gap<10us']+=gap; cnt[pool]['n<10us']+=1; continue
            # cause: another memcpy (any stream) ending within 2us before start?
            j=bisect.bisect_right(cends,o[1])-1
            cause='unexplained'
            while j>=0 and allc[j][2]>o[1]-3000:
                b=allc[j]
                if b is not o and b[0]!=st and b[2]<=o[1]+200: cause=f'CE:{b[4]}'; break
                j-=1
            if cause=='unexplained':
                # a stream wait on this stream issued before the copy whose event record (latest before wait) resolves to a kernel end near start?
                j=bisect.bisect_right(kends,o[1])-1
                while j>=0 and allk[j][2]>o[1]-3000:
                    b=allk[j]
                    if b[0]!=st: cause='EVENT/other-stream kernel end'; break
                    j-=1
            res[pool][cause]+=gap; cnt[pool][cause]+=1
    print(f'\n== {lab} ({fr} frames): memcpy1 model-gap (start - max(API end, prev same-stream op end)) by lead-E pool; sums ms/frame, counts /frame')
    for pool in ('shade','VS','other'):
        print(f'  {pool:6s} n/f {cnt[pool]["n"]/fr:7.2f}  gap sum {res[pool]["gap_sum"]/fr/1e6:6.3f}  dur sum {res[pool]["dur_sum"]/fr/1e6:6.3f}  | gaps<10us: n/f {cnt[pool]["n<10us"]/fr:6.2f} sum {res[pool]["gap<10us"]/fr/1e6:6.3f}')
        for k in sorted(res[pool], key=lambda k:-res[pool][k]):
            if k in ('gap_sum','dur_sum','gap<10us'): continue
            print(f'      long-gap cause {k:34s} n/f {cnt[pool][k]/fr:6.2f} sum {res[pool][k]/fr/1e6:6.3f} ms/f')
