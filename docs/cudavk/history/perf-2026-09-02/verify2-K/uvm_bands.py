import sqlite3, bisect, statistics as st, collections
c=sqlite3.connect('file:/tmp/postpin-rtx/trace.sqlite?mode=ro',uri=True)
S0=1585105154419167
ts=[int(l.split()[0])-S0 for l in open('/tmp/postpin-rtx/ts.txt')]
UVM=c.execute("select start,end,copyKind,migrationCause,bytes,virtualAddress from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind in (11,12) order by start").fetchall()
us=[u[0] for u in UVM]
last_uvm=UVM[-1][1]
def region(va): return (va>>21)<<21
def bursts(rows,gap=20000):
    rows=sorted(rows); out=[]
    for s,e in rows:
        if out and s-out[-1][1]<=gap: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
names={0x7bb9fd000000:'UBO ring',0x7bb9fd200000:'drv scratch(vfetch)',0x7bb9fc200000:'staging 3285',0x7bb94f800000:'VB a',0x7bb94fc00000:'VB b',0x7bb9fc000000:'null page'}
for band,(f0,f1) in {'light 1794-2237 (kernels present)':(1794,2237),'heavy 2150-3150':(2150,3150)}.items():
    # frame f -> submit index 2f ; ts index i is internal submit i (frame 1391 = submit 2782)
    frames=[]
    for f in range(f0,f1):
        k=2*f
        if k+2>=len(ts): break
        a,b,cc=ts[k],ts[k+1],ts[k+2]
        if a<=0 or cc>=last_uvm: continue
        frames.append((a,b,cc))
    # drop outlier frames (frame > 20 ms)
    ft=[cc-a for a,b,cc in frames]
    med=st.median(ft)
    frames=[fr for fr in frames if fr[2]-fr[0] < 3*med]
    n=len(frames)
    print('== band',band,'frames',n,'frame median %.3f ms'%(med/1e6),'boundary interval (copy submit ts -> next render submit ts) median %.3f ms'%(st.median([cc-b for a,b,cc in frames])/1e6))
    agg=collections.defaultdict(lambda: [0]*n); rowsagg=collections.defaultdict(int)
    tot={'cpu_all':[0]*n,'cpu_boundary':[0]*n,'cpu_render':[0]*n,'gpu_all':[0]*n}
    for fi,(a,b,cc) in enumerate(frames):
        lo=bisect.bisect_left(us,a); hi=bisect.bisect_left(us,cc)
        per=collections.defaultdict(list)
        for s,e,kind,cause,by,va in UVM[lo:hi]:
            pos='render[a,b)' if s<b else 'boundary[b,cc)'
            per[(region(va),kind,pos)].append((s,e))
            rowsagg[(region(va),kind,pos,cause)]+=1
        for key,v in per.items():
            agg[key][fi]=sum(e-s for s,e in bursts(v))
        tot['cpu_all'][fi]=sum(e-s for s,e in bursts([x for (r,k,p),v in per.items() if k==12 for x in v]))
        tot['cpu_boundary'][fi]=sum(e-s for s,e in bursts([x for (r,k,p),v in per.items() if k==12 and p.startswith('boundary') for x in v]))
        tot['cpu_render'][fi]=sum(e-s for s,e in bursts([x for (r,k,p),v in per.items() if k==12 and p.startswith('render') for x in v]))
        tot['gpu_all'][fi]=sum(e-s for s,e in bursts([x for (r,k,p),v in per.items() if k==11 for x in v]))
    for k,v in tot.items(): print('  %-13s span mean %.3f median %.3f p90 %.3f ms/frame'%(k,st.mean(v)/1e6,st.median(v)/1e6,sorted(v)[int(.9*n)]/1e6))
    print('  %-20s %4s %-16s %7s %7s %8s %8s'%('region','kind','pos','fault/f','pref/f','mean','median'))
    for key in sorted(agg,key=lambda k:(-st.mean(agg[k]))):
        r,kd,p=key; v=agg[key]
        fl=rowsagg[(r,kd,p,2)]/n; pf=rowsagg[(r,kd,p,3)]/n
        if st.mean(v)<2000: continue
        print('  %-20s %4d %-16s %7.2f %7.2f %8.4f %8.4f'%(names.get(r,'%x'%r),kd,p,fl,pf,st.mean(v)/1e6,st.median(v)/1e6))
