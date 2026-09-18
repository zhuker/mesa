import sqlite3, bisect, statistics as st, collections
c=sqlite3.connect('file:/tmp/postpin-rtx/trace.sqlite?mode=ro',uri=True)
S0=1585105154419167
ts=[int(l.split()[0])-S0 for l in open('/tmp/postpin-rtx/ts.txt')]
tid=c.execute("select globalTid,count(*) n from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cuLaunchKernel%' group by globalTid order by n desc limit 1").fetchone()[0]
L=[r[0] for r in c.execute("select start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cuLaunchKernel%' and globalTid=? order by start",(tid,))]
UVM=c.execute("select start,end,copyKind,migrationCause,bytes,virtualAddress from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind in (11,12) order by start").fetchall()
us=[u[0] for u in UVM]; last_uvm=UVM[-1][1]
def region(va): return (va>>21)<<21
def bursts(rows,gap=20000):
    rows=sorted(rows); out=[]
    for s,e in rows:
        if out and s-out[-1][1]<=gap: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
def nl(a,b): return bisect.bisect_left(L,b)-bisect.bisect_left(L,a)
names={0x7bb9fd000000:'UBO ring',0x7bb9fd200000:'drv scratch(vfetch)',0x7bb9fc200000:'staging 3285',0x7bb94f800000:'VB a',0x7bb94fc00000:'VB b',0x7bb9fc000000:'null page'}
parity=collections.Counter()
for band,(f0,f1) in {'light 1794-2237':(1794,2237),'heavy 2150-3150':(2150,3150)}.items():
    frames=[]
    for f in range(f0,f1):
        k=2*f
        if k+3>=len(ts): break
        # decide which submit is the render one: the interval with the launches
        if nl(ts[k],ts[k+1])>=nl(ts[k+1],ts[k+2]): a,b,cc=ts[k],ts[k+1],ts[k+2]; parity['even=render']+=1
        else: a,b,cc=ts[k+1],ts[k+2],ts[k+3]; parity['odd=render']+=1
        if a<=0 or cc>=last_uvm: continue
        frames.append((a,b,cc))
    ft=[cc-a for a,b,cc in frames]; med=st.median(ft)
    frames=[fr for fr in frames if fr[2]-fr[0] < 3*med]; n=len(frames)
    bnd=[cc-b for a,b,cc in frames]
    print('== band',band,'frames',n,'frame median %.3f ms'%(med/1e6),'| render submit->copy submit median %.3f | copy submit->next render submit (boundary) median %.3f mean %.3f ms'%(st.median([b-a for a,b,cc in frames])/1e6,st.median(bnd)/1e6,st.mean(bnd)/1e6))
    agg=collections.defaultdict(lambda: [0]*n); rowsagg=collections.defaultdict(int)
    tot={'cpu_all':[0]*n,'cpu_boundary':[0]*n,'cpu_render':[0]*n,'gpu_all':[0]*n,'gpu_render':[0]*n}
    for fi,(a,b,cc) in enumerate(frames):
        lo=bisect.bisect_left(us,a); hi=bisect.bisect_left(us,cc)
        per=collections.defaultdict(list)
        for s,e,kind,cause,by,va in UVM[lo:hi]:
            pos='render' if s<b else 'boundary'
            per[(region(va),kind,pos)].append((s,e)); rowsagg[(region(va),kind,pos,cause)]+=1
        for key,v in per.items(): agg[key][fi]=sum(e-s for s,e in bursts(v))
        sel=lambda kk,pp: sum(e-s for s,e in bursts([x for (r,k,p),v in per.items() if k==kk and (pp is None or p==pp) for x in v]))
        tot['cpu_all'][fi]=sel(12,None); tot['cpu_boundary'][fi]=sel(12,'boundary'); tot['cpu_render'][fi]=sel(12,'render'); tot['gpu_all'][fi]=sel(11,None); tot['gpu_render'][fi]=sel(11,'render')
    for k,v in tot.items(): print('  %-13s span mean %.3f median %.3f p90 %.3f ms/frame'%(k,st.mean(v)/1e6,st.median(v)/1e6,sorted(v)[int(.9*n)]/1e6))
    print('  %-20s %4s %-9s %7s %7s %8s %8s %8s'%('region','kind','pos','fault/f','pref/f','mean','median','present'))
    for key in sorted(agg,key=lambda k:(-st.mean(agg[k]))):
        r,kd,p=key; v=agg[key]; fl=rowsagg[(r,kd,p,2)]/n; pf=rowsagg[(r,kd,p,3)]/n
        if st.mean(v)<1500: continue
        print('  %-20s %4d %-9s %7.2f %7.2f %8.4f %8.4f %8.2f'%(names.get(r,'%x'%r),kd,p,fl,pf,st.mean(v)/1e6,st.median(v)/1e6,sum(1 for x in v if x)/n))
print('parity',parity)
# GPU-side stall on cp_vertex_fetch: duration with vs without a fault in the scratch region
K=c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where shortName=722 order by start").fetchall()
F=[u for u in UVM if u[2]==11 and u[3]==2 and region(u[5])==0x7bb9fd200000]
fs=[f[0] for f in F]
withf=[];without=[]
for s,e in K:
    i=bisect.bisect_left(fs,s); hit = i<len(F) and F[i][0]<=e
    (withf if hit else without).append(e-s)
print('cp_vertex_fetch kernels: with scratch fault n=%d median %.1f us mean %.1f | without n=%d median %.1f us mean %.1f'%(len(withf),st.median(withf)/1e3,st.mean(withf)/1e3,len(without),st.median(without)/1e3,st.mean(without)/1e3))
# UBO ring: first 'main' kernel with fault
F2=[u for u in UVM if u[2]==11 and u[3]==2 and region(u[5])==0x7bb9fd000000]; fs2=[f[0] for f in F2]
K2=c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where shortName=686 order by start").fetchall()
w=[];wo=[]
for s,e in K2:
    i=bisect.bisect_left(fs2,s); hit=i<len(F2) and F2[i][0]<=e
    (w if hit else wo).append(e-s)
print("'main' kernels: with UBO-ring fault n=%d median %.1f us mean %.1f | without n=%d median %.1f mean %.1f"%(len(w),st.median(w)/1e3,st.mean(w)/1e3,len(wo),st.median(wo)/1e3,st.mean(wo)/1e3))
