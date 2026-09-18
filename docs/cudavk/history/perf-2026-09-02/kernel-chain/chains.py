import sqlite3, statistics, collections, json, bisect, sys
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
def fidx(t): return bisect.bisect_right([a for a,b in fb], t)-1
FS=[a for a,b in fb]
def fidx(t):
    i=bisect.bisect_right(FS,t)-1
    return i if 0<=i<NF and t<fb[i][1] else -1
ops=[]
for r in c.execute(f"select k.start,k.end,k.streamId,s.value,k.gridX,k.blockX,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>={W0-3000000} and k.start<{W1} order by k.start"):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n=r[3],gx=r[4],bx=r[5],cid=r[6],kind='K'))
for r in c.execute(f"select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>={W0-3000000} and start<{W1} and streamId!=0 order by start"):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n='memcpy%d'%r[3],gx=r[4],bx=0,cid=r[5],kind='M'))
for r in c.execute(f"select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>={W0-3000000} and start<{W1} order by start"):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n='memset',gx=r[3],bx=0,cid=r[4],kind='S'))
ops.sort(key=lambda o:o['s'])
bys=collections.defaultdict(list)
for o in ops: bys[o['st']].append(o)
for st,l in bys.items():
    for i,o in enumerate(l):
        o['i']=i
        o['prev']=l[i-1] if i else None
        o['next']=l[i+1] if i+1<len(l) else None
main=bys[13]; mk=[o for o in main if o['kind']=='K']
for i,o in enumerate(mk): o['ki']=i; o['prevk']=mk[i-1] if i else None; o['nextk']=mk[i+1] if i+1<len(mk) else None
mstart=[o['s'] for o in main]; mend=[o['e'] for o in main]
SIDE=set(range(14,22))
side=[o for o in ops if o['st'] in SIDE]; sstart=[o['s'] for o in side]
def side_between(a,b): 
    i=bisect.bisect_left(sstart,a); j=bisect.bisect_left(sstart,b); return side[i:j]
def main_idle(a,b):
    # main-stream idle time within [a,b): b-a minus union of main ops clipped
    i=bisect.bisect_left(mstart,a-3000000); j=bisect.bisect_left(mstart,b)
    iv=sorted((max(o['s'],a),min(o['e'],b)) for o in main[i:j] if o['e']>a and o['s']<b)
    tot=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return (b-a)-tot

# ---------- A. failed appends (side-stream VS + cp_clip_triangles) ----------
print("=== A. failed episode appends: side-stream VS + cp_clip_triangles then main re-execution ===")
fa=collections.defaultdict(lambda: dict(n=0,side_wall=0,main_idle=0,reVS=0,depthclear=0))
matched=0; unmatched=0; ex=[]
for o in ops:
    if o['n']!='cp_clip_triangles' or o['st'] not in SIDE: continue
    p=o['prev']
    # walk back on the side stream to the VS
    while p and p['kind']!='K': p=p['prev']
    if not p or p['n']!='main': unmatched+=1; continue
    # first ops of the excursion: include memsets/memcpys right before VS on the side stream
    first=p
    while first['prev'] and first['prev']['kind']!='K' and first['s']-first['prev']['e']<20000: first=first['prev']
    # main re-exec: first main-stream 'main' kernel starting after clip end with same gridX
    j=bisect.bisect_left(mstart,o['e'])
    re=None
    for m in main[j:j+40]:
        if m['kind']=='K' and m['n']=='main' and m['gx']==p['gx']: re=m; break
    k=fidx(o['s'])
    if k<0: continue
    d=fa[k]; d['n']+=1; d['side_wall']+=o['e']-first['s']
    # main idle between the last main op ending before the excursion start and the re-exec VS start
    if re:
        matched+=1; d['reVS']+=re['e']-re['s']
        d['main_idle']+=main_idle(first['s'], re['s'])
    else: unmatched+=1
    if len(ex)<3: ex.append((k,(o['e']-first['s'])/1e3, re and (re['s']-first['s'])/1e3))
print("matched re-executions",matched,"unmatched",unmatched, "examples (frame, side_wall_us, first->reVS_us):",ex)
for key in ['n','side_wall','main_idle','reVS']:
    v=[fa[k][key] if k in fa else 0 for k in range(NF)]
    scale=1 if key=='n' else 1e-3
    print("  %-10s per frame: median=%.2f mean=%.2f p90=%.2f (%s)"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale,'count' if key=='n' else 'us'))
# ---------- B. opaque fan-out: seg 0 serial before side segments ----------
print("\n=== B. opaque fan-out episodes: seg-0 chain on main precedes side segments ===")
# side opaque segments: side VS followed (same stream) by cp_clip_rast_fused (non-abuf) [or stage1]
segs=[]
for o in side:
    if o['n']=='main' and o['kind']=='K':
        nx=o['next']
        while nx and nx['kind']!='K': nx=nx['next']
        if nx and nx['n'] in ('cp_clip_rast_fused','cp_rasterize_stage1'):
            # chain end: walk forward to stage3
            e=nx
            while e['next'] and e['next']['n']!='main' and e['next']['s']-e['e']<50000:
                e=e['next']
                if e['n']=='cp_rasterize_stage3': break
            segs.append(dict(s=o['s'],e=e['e'],st=o['st'],vs=o,end=e))
segs.sort(key=lambda d:d['s'])
# cluster into bursts: segments whose start is within 150us of the burst's first start OR overlapping the burst
bursts=[]
for sg in segs:
    if bursts and (sg['s']-bursts[-1]['first']<150000 or sg['s']<bursts[-1]['last_end']): 
        b=bursts[-1]; b['segs'].append(sg); b['last_end']=max(b['last_end'],sg['e']); b['streams'].add(sg['st'])
    else: bursts.append(dict(first=sg['s'],last_end=sg['e'],segs=[sg],streams={sg['st']}))
fo=collections.defaultdict(lambda: dict(n=0,seg0=0,burst=0,gain=0,nsegs=0,segsum=0))
chain_examples=[]
for b in bursts:
    k=fidx(b['first'])
    if k<0: continue
    # seg 0 chain on main: last main kernel ending before b['first'] should be stage3; walk back to VS
    j=bisect.bisect_left(mstart,b['first'])-1
    while j>=0 and (main[j]['kind']!='K'): j-=1
    if j<0: continue
    e0=main[j]
    if e0['n']!='cp_rasterize_stage3': 
        fo[k]['n']+=1; fo[k]['nsegs']+=len(b['segs']); fo[k]['burst']+=b['last_end']-b['first']; continue
    # walk back kernels on main to the VS
    x=e0; steps=0
    while x['prevk'] and x['prevk']['n']!='main' and steps<6: x=x['prevk']; steps+=1
    vs0=x['prevk'] if x['prevk'] and x['prevk']['n']=='main' else x
    chain=e0['e']-vs0['s']; burst=b['last_end']-b['first']
    d=fo[k]; d['n']+=1; d['nsegs']+=len(b['segs']); d['seg0']+=chain; d['burst']+=burst; d['gain']+=min(chain,burst)
    d['segsum']+=sum(sg['e']-sg['s'] for sg in b['segs'])
    if len(chain_examples)<4: chain_examples.append((k,chain/1e3,burst/1e3,len(b['segs']),len(b['streams'])))
print("bursts",len(bursts),"examples (frame, seg0_chain_us, burst_us, nsegs, nstreams):",chain_examples)
for key in ['n','nsegs','seg0','burst','gain','segsum']:
    v=[fo[k][key] if k in fo else 0 for k in range(NF)]
    scale=1 if key in('n','nsegs') else 1e-3
    print("  %-8s per frame: median=%.2f mean=%.2f p90=%.2f"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale))
# burst-internal: side segments' own stagger: max seg wall vs burst wall
stag=[ (b['last_end']-b['first'], max(sg['e']-sg['s'] for sg in b['segs'])) for b in bursts if len(b['segs'])>1]
print("  multi-seg bursts: median burst wall %.1f us, median longest segment %.1f us, median n segs %.1f"%(med([a for a,b in stag])/1e3, med([b for a,b in stag])/1e3, med([len(b['segs']) for b in bursts if len(b['segs'])>1])))
# ---------- C. blended episodes ----------
print("\n=== C. blended episodes (scan_reduce on main marks the tail) ===")
be=collections.defaultdict(lambda: dict(n=0,tail=0,tail_k=0,tail_gap=0,drain=0,seg=0,composite=0,shade=0,total=0))
tails=[]
for o in mk:
    if o['n']!='cp_abuf_scan_reduce': continue
    k=fidx(o['s'])
    if k<0: continue
    # tail kernels until composite (or until a kernel that is not abuf support / main / composite)
    x=o; kl=[o]
    while x['nextk']:
        nx=x['nextk']
        if nx['n'].startswith('cp_abuf_') or (nx['n']=='main' and nx['prevk'] and nx['prevk']['n'] in ('cp_abuf_seg_scatter','cp_abuf_quad_fill_all','cp_abuf_prepare_shade_count')):
            kl.append(nx); x=nx
            if nx['n']=='cp_abuf_composite': break
        else: break
    tail=kl[-1]['e']-o['s']
    ksum=sum(z['e']-z['s'] for z in kl)
    # drain: gap between seg_count end and seg_scatter start (if present)
    drain=0
    for a,b2 in zip(kl,kl[1:]):
        if a['n']=='cp_abuf_seg_count' and b2['n']=='cp_abuf_seg_scatter': drain=b2['s']-a['e']
    # segment phase: side-stream abuf segments before o: kernels on side streams ending before o.s and after previous main kernel ... approximate: side abuf kernels in [prev main kernel end - 0, o.s)
    pm=o['prevk']
    a0=pm['e'] if pm else o['s']-500000
    sk=[z for z in side_between(a0-400000,o['s']) if z['kind']=='K' and (z['n'].endswith('_abuf') or z['n']=='main')]
    segphase=(max(z['e'] for z in sk)-min(z['s'] for z in sk)) if sk else 0
    # composite & FS
    comp=sum(z['e']-z['s'] for z in kl if z['n']=='cp_abuf_composite')
    d=be[k]; d['n']+=1; d['tail']+=tail; d['tail_k']+=ksum; d['tail_gap']+=tail-ksum; d['drain']+=drain; d['seg']+=segphase; d['composite']+=comp
    d['total']+= (kl[-1]['e']-(min(z['s'] for z in sk) if sk else o['s']))
    tails.append((len(kl),tail,ksum,drain,segphase))
print("episodes",len(tails),"median tail kernels",med([t[0] for t in tails]),"median tail wall us %.1f, kernel sum us %.1f, drain gap us %.1f, seg phase us %.1f"%(med([t[1] for t in tails])/1e3,med([t[2] for t in tails])/1e3,med([t[3] for t in tails])/1e3,med([t[4] for t in tails])/1e3))
for key in ['n','tail','tail_k','tail_gap','drain','seg','composite','total']:
    v=[be[k][key] if k in be else 0 for k in range(NF)]
    scale=1 if key=='n' else 1e-3
    print("  %-9s per frame: median=%.2f mean=%.2f p90=%.2f"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale))
# ---------- D. retry / peel passes ----------
print("\n=== D. re-rasterization passes (unfused cp_rasterize_stage1 on main) ===")
rp=collections.defaultdict(lambda: dict(retry=0,retry_wall=0,peel=0,peel_wall=0))
for o in mk:
    if o['n']!='cp_rasterize_stage1': continue
    k=fidx(o['s'])
    if k<0: continue
    # pass wall: from the visbuf memset before stage1 (prev op on main) to writeback end (or peel_advance end)
    start=o['prev']['s'] if o['prev'] and o['prev']['kind']=='S' else o['s']
    x=o; peel=False
    while x['nextk'] and x['nextk']['n'] not in ('cp_rasterize_stage1','main') or (x['nextk'] and x['nextk']['n']=='main' and x['nextk']['prevk']['n']=='cp_fs_compact'):
        x=x['nextk']
        if x['n']=='cp_peel_advance': peel=True; break
        if x['n']=='cp_fs_writeback' and x['nextk'] and x['nextk']['n']!='cp_peel_advance': break
    wall=x['e']-start
    d=rp[k]
    if peel: d['peel']+=1; d['peel_wall']+=wall
    else: d['retry']+=1; d['retry_wall']+=wall
for key in ['retry','retry_wall','peel','peel_wall']:
    v=[rp[k][key] if k in rp else 0 for k in range(NF)]
    scale=1 if 'wall' not in key else 1e-3
    print("  %-10s per frame: median=%.2f mean=%.2f p90=%.2f"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale))
# retry FS durations: are passes 2..8 doing work? compare the FS duration of pass 1 (after fused) vs retry passes
# ---------- E. direct chains on main by class; fullscreen ----------
print("\n=== E. direct raster chains on main (VS -> clip_rast_fused -> s2 -> s3) ===")
cls=collections.defaultdict(list)
fsc=collections.defaultdict(lambda: dict(n=0,rast=0,wall=0,s3=0))
for o in mk:
    if o['n']!='main': continue
    nx=o['nextk']
    if not nx or nx['n']!='cp_clip_rast_fused': continue
    s2=nx['nextk']; s3=s2['nextk'] if s2 else None
    if not s3 or s2['n']!='cp_rasterize_stage2' or s3['n']!='cp_rasterize_stage3': continue
    # shade chain after? compact->main->wb
    cp_=s3['nextk']; shade=None
    if cp_ and cp_['n']=='cp_fs_compact' and cp_['nextk'] and cp_['nextk']['n']=='main' and cp_['nextk']['nextk'] and cp_['nextk']['nextk']['n']=='cp_fs_writeback':
        shade=(cp_,cp_['nextk'],cp_['nextk']['nextk'])
    d=dict(vs=o['e']-o['s'],fused=nx['e']-nx['s'],s2=s2['e']-s2['s'],s3=s3['e']-s3['s'],vsgx=o['gx'],fgx=nx['gx'],
           compact=shade[0]['e']-shade[0]['s'] if shade else None, fs=shade[1]['e']-shade[1]['s'] if shade else None, wb=shade[2]['e']-shade[2]['s'] if shade else None,
           wall=(shade[2]['e'] if shade else s3['e'])-o['s'], fsgx=shade[1]['gx'] if shade else None, cgx=shade[0]['gx'] if shade else None)
    if nx['gx']<=1 and s3['e']-s3['s']>8000: c_='fullscreen-like (<=64 tris, s3>8us)'
    elif nx['gx']<=4: c_='tiny (<=256 tris)'
    elif nx['gx']<=64: c_='small (<=4096 tris)'
    elif nx['gx']<=1024: c_='medium (<=65k tris)'
    else: c_='large (>65k tris)'
    cls[c_].append(d)
    k=fidx(o['s'])
    if k>=0 and c_.startswith('fullscreen'):
        f=fsc[k]; f['n']+=1; f['rast']+=d['fused']+d['s2']+d['s3']; f['wall']+=d['wall']; f['s3']+=d['s3']
for c_,l in sorted(cls.items(), key=lambda kv:-len(kv[1])):
    print("  %-38s n/f=%.2f  med us: VS %.1f fused %.1f s2 %.1f s3 %.1f | compact %.1f FS %.1f wb %.1f | chain wall %.1f (with shade: %d/%d)"%(c_,len(l)/NF,med([d['vs'] for d in l])/1e3,med([d['fused'] for d in l])/1e3,med([d['s2'] for d in l])/1e3,med([d['s3'] for d in l])/1e3,med([d['compact'] for d in l])/1e3,med([d['fs'] for d in l])/1e3,med([d['wb'] for d in l])/1e3,med([d['wall'] for d in l])/1e3,sum(1 for d in l if d['compact'] is not None),len(l)))
l=cls.get('fullscreen-like (<=64 tris, s3>8us)',[])
print("  fullscreen-like by compact grid (target size proxy):", sorted(collections.Counter(d["cgx"] or -1 for d in l).items()))
for key in ['n','rast','wall','s3']:
    v=[fsc[k][key] if k in fsc else 0 for k in range(NF)]
    scale=1 if key=='n' else 1e-3
    print("  fullscreen %-5s per frame: median=%.2f mean=%.2f p90=%.2f"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale))
# ---------- F. host issue lag ----------
print("\n=== F. host issue lag (kernel start - launch API end) per frame ===")
rt={}
for r in c.execute(f"select correlationId,start,end from CUPTI_ACTIVITY_KIND_RUNTIME where start>={W0-50000000} and start<{W1} and nameId in (select id from StringIds where value in ('cuLaunchKernel','cuLaunchKernelEx'))"):
    rt[r[0]]=(r[1],r[2])
lag=collections.defaultdict(list)
for o in ops:
    if o['kind']!='K' or o['cid'] not in rt: continue
    k=fidx(o['s'])
    if k<0: continue
    lag[k].append((o['s']-rt[o['cid']][1], o['st']))
pf=[]
for k in range(NF):
    L=lag.get(k,[])
    if not L: continue
    m=[l for l,st in L if st==13]; s=[l for l,st in L if st!=13]
    pf.append(dict(med_main=med(m)/1e3, med_side=med(s)/1e3, starved_main=sum(1 for l in m if l<3000)/max(len(m),1), starved_all=sum(1 for l,_ in L if l<3000)/len(L)))
for key in pf[0]:
    v=[d[key] for d in pf]; print("  %-13s median=%.3f mean=%.3f p10=%.3f p90=%.3f"%(key,med(v),statistics.mean(v),q(v,.1),q(v,.9)))
json.dump(dict(failed_append={k:dict(v) for k,v in fa.items()}, fanout={k:dict(v) for k,v in fo.items()}, blended={k:dict(v) for k,v in be.items()}, retry={k:dict(v) for k,v in rp.items()}), open('chains.json','w'))
