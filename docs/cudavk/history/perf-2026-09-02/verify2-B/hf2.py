import sqlite3, bisect, statistics as st
from collections import Counter, defaultdict
def run(db, tsfile, i0, i1, label):
    c=sqlite3.connect('file:%s?mode=ro'%db,uri=True)
    sc=list(c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME"))[0][0]
    ts=[int(l.split()[0])-sc for l in open(tsfile)]
    names={i:v for i,v in c.execute("select id,value from StringIds")}
    kmax=list(c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    kmin=list(c.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    T0=max(ts[i0],kmin); T1=min(ts[i1] if i1<len(ts) else kmax,kmax)
    nsub=bisect.bisect_right(ts,T1)-bisect.bisect_left(ts,T0); FR=nsub/2.0
    print("\n=== %s frames %.1f"%(label,FR))
    main=list(c.execute("select globalTid,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by 2 desc"))[0][0]
    rt=list(c.execute("select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<=? order by start",(main,T0-1e6,T1)))
    rtn=[names[r[2]] for r in rt]
    dev={}
    for s,e,sid,corr in c.execute("select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=?",(T0-5e6,T1+5e6)): dev[corr]=(s,e,'K',sid)
    for s,e,sid,ck,b,corr in c.execute("select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind not in (11,12)",(T0-5e6,T1+5e6)): dev[corr]=(s,e,'C',sid)
    for s,e,sid,b,corr in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>=? and start<=?",(T0-5e6,T1+5e6)): dev[corr]=(s,e,'S',sid)
    hf=[i for i,r in enumerate(rt) if rtn[i]=='cuLaunchHostFunc' and T0<=r[0]<=T1]
    cls=[]
    for i in hf:
        if cls and i-cls[-1][-1]<=3 and rt[i][0]-rt[cls[-1][-1]][1]<50e3: cls[-1].append(i)
        else: cls.append([i])
    heads=[cl for cl in cls if len(cl)==2]
    series=defaultdict(list); firstk=[]; lastk_before_sync=[]; sync_after=[]; nk_before_sync=[]
    for cl in heads:
        j=cl[-1]; k=j+1; n=0; last=None; syncd=None
        while k<len(rt) and n<600:
            nm=rtn[k]
            if nm in ('cuCtxSynchronize','cuStreamSynchronize') and syncd is None:
                syncd=rt[k][0]-rt[j][1]; nk_before_sync.append(n); 
                if last is not None: lastk_before_sync.append(last)
                break
            if nm in ('cuLaunchKernel','cuLaunchKernelEx') and rt[k][3] in dev:
                n+=1; ds=dev[rt[k][3]][0]; bl=ds-rt[k][1]; series[n].append(bl); last=bl
                if n==1: firstk.append(bl)
            k+=1
        if syncd is not None: sync_after.append(syncd)
    def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
    print(" first kernel after head pair: dev_start-api_end med %.1f p90 %.1f mean %.1f us -> sum %.3f ms/frame (n=%d)"%(q(firstk,.5)/1e3,q(firstk,.9)/1e3,st.mean(firstk)/1e3,sum(firstk)/1e6/FR,len(firstk)))
    print(" backlog (dev_start-api_end, median us) for kernel n after the head pair:")
    print("  "+" ".join("%d:%.1f"%(n,q(series[n],.5)/1e3) for n in sorted(series) if n<=40))
    if sync_after: print(" first host sync after head pair: %.1f us median after pair (n=%d); kernels launched before it med %d; backlog at last kernel before sync med %.1f p90 %.1f us"%(st.median(sync_after)/1e3,len(sync_after),st.median(nk_before_sync),q(lastk_before_sync,.5)/1e3,q(lastk_before_sync,.9)/1e3))
run('/tmp/postdead/trace.sqlite','/tmp/postdead/submit-ts.txt',2782,10**9,'postdead')
run('/tmp/postpin-rtx2/trace.sqlite','/tmp/postpin-rtx2/ts.txt',4896,6924,'postpin-rtx2 heavy')
