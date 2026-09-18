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
FSs=[a for a,b in fb]
def fidx(t):
    i=bisect.bisect_right(FSs,t)-1
    return i if 0<=i<NF and t<fb[i][1] else -1
K=[dict(s=r[0],e=r[1],st=r[2],n=r[3],gx=r[4]) for r in c.execute(f"select k.start,k.end,k.streamId,s.value,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>={W0} and k.start<{W1} order by k.start")]
M=[dict(s=r[0],e=r[1],st=r[2]) for r in c.execute(f"select start,end,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where start>={W0} and start<{W1} and streamId!=0")]
S=[dict(s=r[0],e=r[1],st=r[2]) for r in c.execute(f"select start,end,streamId from CUPTI_ACTIVITY_KIND_MEMSET where start>={W0} and start<{W1}")]
allops=K+M+S
def union(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return tot
# exact main∩side intersection per frame
def clipiv(l,a,b): return [(max(o['s'],a),min(o['e'],b)) for o in l if o['e']>a and o['s']<b]
inter=[]
mainops=[o for o in allops if o['st']==13]; sideops=[o for o in allops if 14<=o['st']<=21]
for a,b in fb:
    um=union(clipiv(mainops,a,b)); us=union(clipiv(sideops,a,b)); ub=union(clipiv(mainops,a,b)+clipiv(sideops,a,b))
    inter.append((um+us-ub)/1e6)
print("main∩side busy per frame (ms): median=%.4f mean=%.4f max=%.4f"%(med(inter),statistics.mean(inter),max(inter)))
# blended episodes: segments on side streams = main -> cp_clip_rast_fused_abuf chains
bys=collections.defaultdict(list)
for o in K: bys[o['st']].append(o)
segs=[]
for st in range(14,22):
    l=bys[st]
    for i,o in enumerate(l):
        if o['n']=='cp_clip_rast_fused_abuf' and i>0 and l[i-1]['n']=='main':
            e=o
            for j in range(i+1,min(i+4,len(l))):
                if l[j]['n'] in ('cp_rasterize_stage2_abuf','cp_rasterize_stage3_abuf'): e=l[j]
                else: break
            segs.append(dict(s=l[i-1]['s'],e=e['e'],st=st,dur=e['e']-l[i-1]['s']))
segs.sort(key=lambda d:d['s'])
scan=[o for o in K if o['n']=='cp_abuf_scan_reduce' and o['st']==13]
sst=[d['s'] for d in segs]
ep=[]
prev_scan=W0
for i,o in enumerate(scan):
    a=bisect.bisect_left(sst,prev_scan); b=bisect.bisect_left(sst,o['s'])
    mine=segs[a:b]
    if mine:
        ep.append(dict(f=fidx(o['s']),nsegs=len(mine),streams=len(set(d['st'] for d in mine)),phase=max(d['e'] for d in mine)-min(d['s'] for d in mine),segsum=sum(d['dur'] for d in mine),longest=max(d['dur'] for d in mine)))
    prev_scan=o['s']
print("blended episodes with identified segments:",len(ep),"of",len(scan))
print("  nsegs histogram:",sorted(collections.Counter(e['nsegs'] for e in ep).items()))
print("  single-seg share: %.2f"%(sum(1 for e in ep if e['nsegs']==1)/len(ep)))
print("  median segment-phase wall us: %.1f (single-seg %.1f, multi-seg %.1f); median longest seg %.1f"%(med([e['phase'] for e in ep])/1e3, med([e['phase'] for e in ep if e['nsegs']==1])/1e3, med([e['phase'] for e in ep if e['nsegs']>1])/1e3, med([e['longest'] for e in ep])/1e3))
pf=collections.defaultdict(lambda: collections.Counter())
for e in ep:
    if e['f']>=0: pf[e['f']]['phase']+=e['phase']; pf[e['f']]['n']+=1; pf[e['f']]['segsum']+=e['segsum']
for key in ['n','phase','segsum']:
    v=[pf[k][key] if k in pf else 0 for k in range(NF)]; sc=1 if key=='n' else 1e-3
    print("  %-7s per frame median=%.2f mean=%.2f"%(key,med(v)*sc,statistics.mean(v)*sc))
# opaque fan-out segments (side main -> cp_clip_rast_fused) count per frame and nsegs
osegs=collections.Counter()
for st in range(14,22):
    l=bys[st]
    for i,o in enumerate(l):
        if o['n']=='cp_clip_rast_fused' and i>0 and l[i-1]['n']=='main': osegs[fidx(o['s'])]+=1
v=[osegs.get(k,0) for k in range(NF)]; print("opaque side segments per frame: median=%.1f mean=%.2f"%(med(v),statistics.mean(v)))
# stage3 direct: duration vs whether it is the fullscreen case; per-block serial time. Count s3 launches with dur>8us on main not in fan-out
s3=[o for o in K if o['n']=='cp_rasterize_stage3']
print("stage3 direct: n/f=%.2f; dur buckets us: <2: %.2f/f, 2-8: %.2f/f, 8-25: %.2f/f, >25: %.2f/f"%(len(s3)/NF, sum(1 for o in s3 if o['e']-o['s']<2000)/NF, sum(1 for o in s3 if 2000<=o['e']-o['s']<8000)/NF, sum(1 for o in s3 if 8000<=o['e']-o['s']<25000)/NF, sum(1 for o in s3 if o['e']-o['s']>=25000)/NF))
# FS direct 'main' after compact: duration distribution
fsd=[o for i,o in enumerate(K) if o['n']=='main' and o['gx']==4096]
print("FS direct (grid 4096): n/f=%.2f med %.1f us, p90 %.1f, sum ms/f %.3f; share <4us: %.2f"%(len(fsd)/NF, med([o['e']-o['s'] for o in fsd])/1e3, q([o['e']-o['s'] for o in fsd],.9)/1e3, sum(o['e']-o['s'] for o in fsd)/1e6/NF, sum(1 for o in fsd if o['e']-o['s']<4000)/len(fsd)))
