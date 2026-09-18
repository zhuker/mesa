import sqlite3, bisect, statistics as st, sys
from collections import Counter, defaultdict
def run(db, tsfile, i0, i1, label):
    c=sqlite3.connect('file:%s?mode=ro'%db,uri=True)
    sc=list(c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME"))[0][0]
    ts=[int(l.split()[0])-sc for l in open(tsfile)]
    names={i:v for i,v in c.execute("select id,value from StringIds")}
    T0=ts[i0]; T1=ts[i1] if i1<len(ts) else list(c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    kmax=list(c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    kmin=list(c.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    T1=min(T1,kmax); T0=max(T0,kmin)
    nsub=bisect.bisect_right(ts,T1)-bisect.bisect_left(ts,T0)
    FR=nsub/2.0
    print("\n=== %s: window rel s %.3f..%.3f, submits %d, frames %.1f"%(label,T0/1e9,T1/1e9,nsub,FR))
    main=list(c.execute("select globalTid,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by 2 desc"))[0][0]
    rt=list(c.execute("select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<=? order by start",(main,T0-1e6,T1)))
    rtn=[names[r[2]] for r in rt]
    # device ops with correlation
    dev={}  # corr -> (start,end,kind,bytes,stream)
    ops=[]
    for s,e,sid,corr in c.execute("select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=?",(T0-5e6,T1+5e6)):
        dev[corr]=(s,e,'K',0,sid); ops.append((s,e,sid,'K'))
    for s,e,sid,ck,b,corr in c.execute("select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind not in (11,12)",(T0-5e6,T1+5e6)):
        dev[corr]=(s,e,'C%d'%ck,b,sid); ops.append((s,e,sid,'C'))
    for s,e,sid,b,corr in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>=? and start<=?",(T0-5e6,T1+5e6)):
        dev[corr]=(s,e,'S',b,sid); ops.append((s,e,sid,'S'))
    ops.sort(); starts=[o[0] for o in ops]
    pm=[];m=-1
    for o in ops: m=max(m,o[1]); pm.append(m)
    def last_end_before(t):
        i=bisect.bisect_left(starts,t); return pm[i-1] if i>0 else None
    kst=sorted(o[0] for o in ops if o[3]=='K')
    def first_k_ge(t):
        i=bisect.bisect_left(kst,t); return kst[i] if i<len(kst) else None
    subs=[t for t in ts if T0<=t<=T1]
    def crosses(a,b):
        i=bisect.bisect_right(subs,a); return i<len(subs) and subs[i]<b
    hf=[i for i,r in enumerate(rt) if rtn[i]=='cuLaunchHostFunc' and T0<=r[0]<=T1]
    print("host funcs in window:",len(hf),"%.2f/frame"%(len(hf)/FR), "API dur med %.1f us"%(st.median(rt[i][1]-rt[i][0] for i in hf)/1e3))
    # cluster: consecutive host funcs within 50us and no device-bearing API between
    cls=[]; 
    for i in hf:
        if cls and i-cls[-1][-1]<=3 and rt[i][0]-rt[cls[-1][-1]][1]<50e3: cls[-1].append(i)
        else: cls.append([i])
    print("clusters:",Counter(len(x) for x in cls), "%.2f clusters/frame"%(len(cls)/FR))
    # per cluster: context
    ctx=Counter(); res=defaultdict(list)
    for cl in cls:
        i=cl[0]; j=cl[-1]
        prev=[rtn[k] for k in range(max(0,i-3),i)]
        nxt=[rtn[k] for k in range(j+1,min(len(rt),j+4))]
        key=(len(cl),prev[-1] if prev else '-',nxt[0] if nxt else '-')
        ctx[key]+=1
        # next device-bearing API after cluster
        k=j+1
        while k<len(rt) and rt[k][3] not in dev: k+=1
        if k>=len(rt): continue
        N=rt[k]; ds,de,kind,b,sid=dev[N[3]]
        le=last_end_before(rt[j][1])
        cross=crosses(rt[j][0],ds)
        # is any device op (kernel) between? first kernel start >= cluster end
        ns=first_k_ge(rt[j][1])
        gap_b2=(ns-le) if (ns is not None and le is not None) else None
        crossk=crosses(rt[j][0],ns) if ns else True
        stall_upper=max(0,ds-max(N[1],le))
        host_issue=N[0]-rt[j][1]
        pre_run=max(0,le-rt[j][1])   # device still busy after host func enqueued
        res[(len(cl),cross)].append(dict(stall_upper=stall_upper,host_issue=host_issue,api_to_dev=ds-N[1],le_to_ds=ds-le,gap_b2=gap_b2,crossk=crossk,nkind=kind,nb=b,pre_run=pre_run,ns_minus_hfend=(ns-rt[j][1]) if ns else None,hf_to_ds=ds-rt[j][1]))
    for k,v in ctx.most_common(8): print("  ctx",k,v)
    def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
    for key,L in res.items():
        n=len(L); print("\n class size=%d crosses_submit_to_next_devop=%s: n=%d (%.2f/frame) next-op kinds %s"%(key[0],key[1],n,n/FR,Counter((d['nkind'],d['nb']>1e6) for d in L).most_common(3)))
        for f in ('stall_upper','host_issue','api_to_dev','le_to_ds','hf_to_ds','pre_run','gap_b2','ns_minus_hfend'):
            v=[d[f] for d in L if d[f] is not None]
            if not v: continue
            print("   %-14s med %8.1f  p90 %8.1f  mean %8.1f us  sum %.3f ms/frame  (n=%d)"%(f,q(v,.5)/1e3,q(v,.9)/1e3,st.mean(v)/1e3,sum(v)/1e6/FR,len(v)))
        print("   crossk (bubbles2 excluded) share: %.2f"%(sum(1 for d in L if d['crossk'])/n))
    # reproduce bubbles2 number: gap over non-crossing per host func
    hg=[]; hc=0
    for i in hf:
        le=last_end_before(rt[i][1]); ns=first_k_ge(rt[i][1])
        if le is None or ns is None: continue
        if crosses(rt[i][0],ns): hc+=1; continue
        hg.append(ns-le)
    hg.sort()
    print("\n bubbles2 reproduction: n %d cross %d gap med %.1f p90 %.1f mean %.1f sum %.3f ms/frame"%(len(hf),hc,hg[len(hg)//2]/1e3,hg[int(.9*len(hg))]/1e3,st.mean(hg)/1e3,sum(hg)/1e6/FR))
    # reference: api_to_dev for cuMemcpy2DAsync copies that do NOT follow a host func within 200us, when device idle (le<=api_end)
    hfends=sorted(rt[i][1] for i in hf)
    ref=[]
    for i,r in enumerate(rt):
        if rtn[i]!='cuMemcpy2DAsync_v2' or r[3] not in dev: continue
        j=bisect.bisect_left(hfends,r[0])
        if j>0 and r[0]-hfends[j-1]<300e3: continue
        ds=dev[r[3]][0]; le=last_end_before(r[1])
        if le is not None and le<=r[1]: ref.append(ds-r[1])
    if ref: print(" reference: cuMemcpy2DAsync not after a host func, device idle at API end: dev_start-api_end med %.1f p90 %.1f us (n=%d)"%(q(ref,.5)/1e3,q(ref,.9)/1e3,len(ref)))
    # first kernel after head cluster: start - api_end
    return
run('/tmp/postdead/trace.sqlite','/tmp/postdead/submit-ts.txt',2782,10**9,'postdead (light band, pre-pin)')
run('/tmp/postpin-rtx2/trace.sqlite','/tmp/postpin-rtx2/ts.txt',4896,6924,'postpin-rtx2 (heavy band 2448-3462, post-pin)')
