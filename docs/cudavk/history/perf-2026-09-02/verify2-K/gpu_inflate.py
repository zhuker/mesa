import sqlite3, bisect, statistics as st, collections
c=sqlite3.connect('file:/tmp/postpin-rtx/trace.sqlite?mode=ro',uri=True)
cols=[r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
gk=[x for x in cols if x.startswith('grid') or x.startswith('block')][:6]
K=c.execute(f"select start,end,shortName,{','.join(gk)},streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
kend=K[-1][1]
F=c.execute("select start,end,virtualAddress,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=11 and migrationCause=2 and start<? order by start",(kend,)).fetchall()
fs=[f[0] for f in F]
names={0x7bb9fd000000:'UBO ring',0x7bb9fd200000:'drv scratch(vfetch)',0x7bb9fc200000:'staging 3285',0x7bb94f800000:'VB a',0x7bb94fc00000:'VB b',0x7bb9fc000000:'null page'}
def region(va): return (va>>21)<<21
# class medians without faults
byclass=collections.defaultdict(list); hits=[]
for k in K:
    s,e=k[0],k[1]; cls=(k[2],)+tuple(k[3:3+len(gk)])
    i=bisect.bisect_left(fs,s); regs=set()
    while i<len(F) and F[i][0]<=e:
        regs.add(region(F[i][2])); i+=1
    if regs: hits.append((s,e,cls,regs,k[-1]))
    else: byclass[cls].append(e-s)
med={cls:st.median(v) for cls,v in byclass.items() if len(v)>=5}
S0=1585105154419167
ts=[int(l.split()[0])-S0 for l in open('/tmp/postpin-rtx/ts.txt')]
# frames in [0.05s,kend): count frames as readback copies count
nframes=c.execute("select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=8 and srcKind=2 and dstKind=4 and start>50000000 and start<?",(kend,)).fetchone()[0]
print('frames with kernel rows',nframes,'kernels with >=1 GPU fault',len(hits))
infl=collections.defaultdict(float); cnt=collections.Counter(); streams=collections.Counter()
for s,e,cls,regs,sid in hits:
    if cls not in med: continue
    d=(e-s)-med[cls]
    for r in regs:
        infl[names.get(r,'%x'%r)]+=d/len(regs); cnt[names.get(r,'%x'%r)]+=1
    if 'drv scratch(vfetch)' in [names.get(r) for r in regs]: streams[sid]+=1
print('kernel-duration inflation over fault-free class median, per frame:')
tot=0
for r in sorted(infl,key=lambda x:-infl[x]):
    print('  %-22s kernels/frame %.2f  inflation %.4f ms/frame'%(r,cnt[r]/nframes,infl[r]/nframes/1e6)); tot+=infl[r]
print('  total %.4f ms/frame'%(tot/nframes/1e6))
print('streams of faulting cp_vertex_fetch',streams.most_common(4))
allst=collections.Counter(k[-1] for k in K if k[2]==722); print('all cp_vertex_fetch streams',allst.most_common(4))
# memcpy (kind 8 managed->device) inflation with copy-engine faults on staging region
M=c.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=8 and srcKind=4 and dstKind=2 and start<? order by start",(kend,)).fetchall()
w=[];wo=[]
for s,e,by in M:
    i=bisect.bisect_left(fs,s); hit=i<len(F) and F[i][0]<=e
    (w if hit else wo).append((e-s,by))
def q(v): return st.median([x[0] for x in v])/1e3 if v else 0
print('staging->device copies: with fault n=%d median %.1f us (bytes med %d) | without n=%d median %.1f us (bytes med %d) ; per frame with-fault %.2f, excess %.4f ms/frame'%(len(w),q(w),st.median([x[1] for x in w]) if w else 0,len(wo),q(wo),st.median([x[1] for x in wo]) if wo else 0,len(w)/nframes,sum(x[0] for x in w)/1e6/nframes - len(w)/nframes*q(wo)/1e3))
