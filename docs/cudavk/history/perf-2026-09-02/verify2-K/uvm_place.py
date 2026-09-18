import sqlite3, bisect, statistics as st, collections
c=sqlite3.connect('file:/tmp/postpin-rtx/trace.sqlite?mode=ro',uri=True)
S0=1585105154419167
ts=[int(l.split()[0])-S0 for l in open('/tmp/postpin-rtx/ts.txt')]
kern_end=c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
# frames: pair (2k,2k+1); find which parity is the render submit by kernel count
K=c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ks=[k[0] for k in K]
def nk(a,b): return bisect.bisect_left(ks,b)-bisect.bisect_left(ks,a)
# choose frames fully inside [0.05s, kern_end]
frames=[]
for k in range(0,len(ts)-2,2):
    a,b,cc=ts[k],ts[k+1],ts[k+2]
    if a>50_000_000 and cc<kern_end: frames.append((k,a,b,cc))
print('frames usable',len(frames),'first submit idx',frames[0][0],'last',frames[-1][0])
print('kernels in even->odd vs odd->next for first 3 frames:',[(nk(a,b),nk(b,cc)) for _,a,b,cc in frames[:3]])
# readback copy rows and main-thread API
RB=c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=8 and srcKind=2 and dstKind=4 order by start").fetchall()
rbs=[r[0] for r in RB]
names=dict(c.execute("select id,value from StringIds"))
tid=c.execute("select globalTid,count(*) n from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cuLaunchKernel%' group by globalTid order by n desc limit 1").fetchone()[0]
API=c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? order by start",(tid,)).fetchall()
api_s=[a[0] for a in API]; api_e=[a[1] for a in API]
UVM=c.execute("select start,end,copyKind,migrationCause,bytes,virtualAddress from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind in (11,12) order by start").fetchall()
us=[u[0] for u in UVM]
def region(va): return (va>>21)<<21
def bursts(rows,gap=20000):
    rows=sorted(rows); out=[]; 
    for s,e in rows:
        if out and s-out[-1][1]<=gap: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
# per frame classification
agg=collections.defaultdict(lambda: collections.defaultdict(list))  # (region,kind,pos)->list of per-frame spans
gapstats=[]; cpu_in_gap=[]; gpu_in_sub=[]
frame_cpu_all=[]
for k,a,b,cc in frames:
    # render submit at a; copy submit at b; readback copy in [b,cc)
    i=bisect.bisect_left(rbs,b); 
    if i>=len(RB) or RB[i][0]>=cc: continue
    rb_end=RB[i][1]
    j=bisect.bisect_left(ks,rb_end); first_k=K[j][0] if j<len(K) else None
    if first_k is None: continue
    gap=first_k-rb_end; gapstats.append(gap)
    # uvm rows in [a, cc) -> positions: 'boundary' if in [rb_end, first_k) of NEXT frame? we classify rows in this frame's own window [rb_end_prev.. ) -- simpler: rows in [a,cc) plus next boundary [rb_end, first_k)
    lo=bisect.bisect_left(us,a); hi=bisect.bisect_left(us,first_k)
    per=collections.defaultdict(list)
    for r in UVM[lo:hi]:
        s,e,kind,cause,by,va=r
        if s>=rb_end: pos='boundary'
        elif s>=a and s<b: pos='render_submit'
        elif s>=b and s<rb_end: pos='copy_submit'
        else: pos='other'
        per[(region(va),kind,pos)].append((s,e))
    cpu_gap=sum(e-s for s,e in bursts([x for (reg,kind,pos),v in per.items() if kind==12 and pos=='boundary' for x in v]))
    cpu_in_gap.append(cpu_gap)
    frame_cpu_all.append(sum(e-s for s,e in bursts([x for (reg,kind,pos),v in per.items() if kind==12 for x in v])))
    for key,v in per.items():
        agg[key]['span'].append(sum(e-s for s,e in bursts(v)))
        agg[key]['rows'].append(len(v))
        agg[key]['bytes'].append(sum(0 for _ in v))
n=len(gapstats)
print('frames analysed',n)
print('frame-start gap (readback end -> first kernel) median %.3f mean %.3f ms'%(st.median(gapstats)/1e6, st.mean(gapstats)/1e6))
print('CPU-side (DtoH) burst span inside boundary gap: mean %.3f median %.3f ms/frame'%(st.mean(cpu_in_gap)/1e6, st.median(cpu_in_gap)/1e6))
print('CPU-side (DtoH) burst span whole frame: mean %.3f median %.3f ms/frame'%(st.mean(frame_cpu_all)/1e6, st.median(frame_cpu_all)/1e6))
print()
print('%-14s %4s %-14s %8s %8s %8s'%('region','kind','pos','rows/f','span_mean','span_med(ms, over ALL frames)'))
for key in sorted(agg, key=lambda k:(k[0],k[1],k[2])):
    d=agg[key]; rows=sum(d['rows'])/n; sp=d['span']+[0]*(n-len(d['span']))
    print('%-14x %4d %-14s %8.2f %8.4f %8.4f  present %d/%d'%(key[0],key[1],key[2],rows,st.mean(sp)/1e6,st.median(sp)/1e6,len(d['span']),n))
