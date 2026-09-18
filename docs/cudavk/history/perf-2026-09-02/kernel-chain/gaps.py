import sqlite3, statistics, collections, json, bisect
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
S0=1503511535334936
ts=[l.split() for l in open('/tmp/postdead/submit-ts.txt') if len(l.split())==2]
rel=[int(a)-S0 for a,b in ts]
NF=412; fb=[(rel[2782+2*k], rel[2784+2*k]) for k in range(NF)]
W0=fb[0][0]; W1=fb[-1][1]
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
def med(x):
    x=[v for v in x if v is not None]; return statistics.median(x) if x else 0
def q(x,p):
    x=sorted(v for v in x if v is not None); return x[int(p*(len(x)-1))] if x else 0
ops=[]
for r in c.execute(f"select k.start,k.end,k.streamId,s.value,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>={W0-3000000} and k.start<{W1}"):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n=r[3],kind='K'))
for r in c.execute(f"select start,end,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>={W0-3000000} and start<{W1} and streamId!=0"):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n='memcpy%d'%r[3],kind='M',b=r[4]))
for r in c.execute(f"select start,end,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where start>={W0-3000000} and start<{W1}"):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n='memset',kind='S',b=r[3]))
syncs=[dict(s=r[0],e=r[1],typ=r[2]) for r in c.execute(f"select start,end,syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>={W0-3000000} and start<{W1} and syncType in (1,3,4)")]
ops.sort(key=lambda o:o['s'])
main=[o for o in ops if o['st']==13]; side=[o for o in ops if 14<=o['st']<=21]
sstart=[o['s'] for o in side]; ssend=sorted(o['e'] for o in side)
def side_busy_in(a,b):
    i=bisect.bisect_left(sstart,a-2000000); j=bisect.bisect_left(sstart,b)
    iv=sorted((max(o['s'],a),min(o['e'],b)) for o in side[i:j] if o['e']>a and o['s']<b)
    tot=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return tot
sy=sorted(syncs,key=lambda d:d['s']); systart=[d['s'] for d in sy]
def host_blocked_in(a,b):
    i=bisect.bisect_left(systart,a-5000000); j=bisect.bisect_left(systart,b)
    return sum(max(0,min(d['e'],b)-max(d['s'],a)) for d in sy[i:j])
# classify main-stream gaps per frame
FSs=[a for a,b in fb]
def fidx(t):
    i=bisect.bisect_right(FSs,t)-1
    return i if 0<=i<NF and t<fb[i][1] else -1
pf=collections.defaultdict(lambda: collections.Counter())
hist=collections.Counter()
for a,b in zip(main,main[1:]):
    g=b['s']-a['e']
    if g<=0: continue
    k=fidx(a['e'])
    if k<0: continue
    sb=side_busy_in(a['e'],b['s'])
    hb=host_blocked_in(a['e'],b['s'])
    d=pf[k]
    d['gap_total']+=g; d['side_busy']+=sb; d['host_blocked']+=min(hb,g); d['n_gaps']+=1
    if g<3000: d['gap_lt3']+=g; d['n_lt3']+=1
    elif g<20000: d['gap_3_20']+=g; d['n_3_20']+=1
    elif g<200000: d['gap_20_200']+=g; d['n_20_200']+=1
    else: d['gap_ge200']+=g; d['n_ge200']+=1
    # pure idle: gap minus side-busy (approximation: side busy inside gap)
    d['pure_idle']+=g-sb
print("main-stream gap census per frame (median over frames; us):")
keys=['n_gaps','gap_total','side_busy','pure_idle','host_blocked','gap_lt3','n_lt3','gap_3_20','n_3_20','gap_20_200','n_20_200','gap_ge200','n_ge200']
for key in keys:
    v=[pf[k][key] if k in pf else 0 for k in range(NF)]
    scale=1 if key.startswith('n_') else 1e-3
    print("  %-13s median=%.2f mean=%.2f p90=%.2f"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale))
# largest gaps: what precedes/follows
big=[]
for a,b in zip(main,main[1:]):
    g=b['s']-a['e']
    if g>150000 and fidx(a['e'])>=0: big.append((g,a['n'],b['n'],side_busy_in(a['e'],b['s']),host_blocked_in(a['e'],b['s'])))
print("\nmain-stream gaps >150us: n/f=%.2f, median gap us %.1f; by (prev,next) top:"%(len(big)/NF, med([g for g,*_ in big])/1e3))
cc=collections.Counter((p,n) for g,p,n,sb,hb in big)
for (p,n),cnt in cc.most_common(8):
    sel=[x for x in big if x[1]==p and x[2]==n]
    print("   %-20s -> %-22s n/f=%.2f med_gap=%.1f med_side_busy=%.1f med_host_blocked=%.1f"%(p,n,cnt/NF,med([x[0] for x in sel])/1e3,med([x[3] for x in sel])/1e3,med([x[4] for x in sel])/1e3))
# blended episode gate/join gaps: side abuf segment: first op on side stream after main op; join: scan_reduce start - last side end
gate=[]; join=[]
mainK=[o for o in main if o['kind']=='K']
mstart=[o['s'] for o in main]
for o in mainK:
    if o['n']!='cp_abuf_scan_reduce': continue
    # last side kernel ending before o.s
    j=bisect.bisect_left(ssend,o['s'])-1
    if j>=0: join.append(o['s']-ssend[j])
for o in side:
    if o['n']=='cp_clip_rast_fused_abuf':
        # find the side VS before it and the first side op of the segment; gate = first side op start - last main op end before it
        pass
# simpler: for each side stream op that starts a burst (no side op in previous 30us on any side stream) find last main op end before
bstart=[]
prev_end=None
for o in side:
    i=bisect.bisect_left(ssend,o['s'])
    # any side op ending within 30us before?
    if i>0 and o['s']-ssend[i-1]<30000: continue
    j=bisect.bisect_left(mstart,o['s'])-1
    while j>=0 and main[j]['e']>o['s']: j-=1
    if j>=0 and fidx(o['s'])>=0: gate.append(o['s']-main[j]['e'])
print("\nside-stream burst starts: n/f=%.2f, gate gap (last main op end -> first side op start) median=%.1f us p90=%.1f"%(len(gate)/NF, med(gate)/1e3, q(gate,.9)/1e3))
print("blended join gap (last side op end -> scan_reduce start): n/f=%.2f median=%.1f us p90=%.1f"%(len(join)/NF, med(join)/1e3, q(join,.9)/1e3))
