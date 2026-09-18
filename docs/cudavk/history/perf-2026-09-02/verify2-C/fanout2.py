import sqlite3, statistics, collections, bisect, sys
which=sys.argv[1]
if which=='postdead':
    DB='/tmp/postdead/trace.sqlite'; S0=1503511535334936; TS='/tmp/postdead/submit-ts.txt'; F0=1391; NF=412
else:
    DB='/tmp/postpin-rtx2/trace.sqlite'; S0=1585329015665402; TS='/tmp/postpin-rtx2/ts.txt'; F0=2450; NF=700
c=sqlite3.connect('file:%s?mode=ro'%DB,uri=True)
ts=[l.split() for l in open(TS) if len(l.split())==2]
rel=[int(a)-S0 for a,b in ts]
fb=[(rel[2*(F0+k)], rel[2*(F0+k)+2]) for k in range(NF)]
W0=fb[0][0]; W1=fb[-1][1]
FS=[a for a,b in fb]
def fidx(t):
    i=bisect.bisect_right(FS,t)-1
    return i if 0<=i<NF and t<fb[i][1] else -1
def med(x): return statistics.median(x) if x else 0
def q(x,p):
    x=sorted(x); return x[int(p*(len(x)-1))] if x else 0
print("window frames",F0,F0+NF-1,"median frame ms",med([b-a for a,b in fb])/1e6, "mean", statistics.mean([b-a for a,b in fb])/1e6)
# host launches by correlation id
rt={}
for r in c.execute(f"select correlationId,start,end from CUPTI_ACTIVITY_KIND_RUNTIME where start>={W0-5000000} and start<{W1+1000000}"):
    rt[r[0]]=(r[1],r[2])
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
        o['prev']=l[i-1] if i else None; o['next']=l[i+1] if i+1<len(l) else None
main=bys[13]; mk=[o for o in main if o['kind']=='K']
for i,o in enumerate(mk): o['prevk']=mk[i-1] if i else None; o['nextk']=mk[i+1] if i+1<len(mk) else None
mstart=[o['s'] for o in main]
SIDE=set(range(14,22))
side=[o for o in ops if o['st'] in SIDE]
segs=[]
for o in side:
    if o['n']=='main' and o['kind']=='K':
        nx=o['next']
        while nx and nx['kind']!='K': nx=nx['next']
        if nx and nx['n'] in ('cp_clip_rast_fused','cp_rasterize_stage1'):
            e=nx
            while e['next'] and e['next']['n']!='main' and e['next']['s']-e['e']<50000:
                e=e['next']
                if e['n']=='cp_rasterize_stage3': break
            # first op of segment on side stream: include the upload memcpy right before VS if any (after gate)
            first=o
            segs.append(dict(s=o['s'],e=e['e'],st=o['st'],vs=o,end=e))
segs.sort(key=lambda d:d['s'])
bursts=[]
for sg in segs:
    if bursts and (sg['s']-bursts[-1]['first']<150000 or sg['s']<bursts[-1]['last_end']):
        b=bursts[-1]; b['segs'].append(sg); b['last_end']=max(b['last_end'],sg['e']); b['streams'].add(sg['st'])
    else: bursts.append(dict(first=sg['s'],last_end=sg['e'],segs=[sg],streams={sg['st']}))
fo=collections.defaultdict(lambda: collections.Counter())
LAT=5000  # ns launch->start latency assumed when nothing blocks
rows=[]
nochain=0
for b in bursts:
    k=fidx(b['first'])
    if k<0: continue
    j=bisect.bisect_left(mstart,b['first'])-1
    while j>=0 and main[j]['kind']!='K': j-=1
    if j<0: continue
    e0=main[j]
    if e0['n']!='cp_rasterize_stage3':
        nochain+=1; fo[k]['n_nochain']+=1; fo[k]['burst_nochain']+=b['last_end']-b['first']; continue
    x=e0; steps=0
    while x['prevk'] and x['prevk']['n']!='main' and steps<6: x=x['prevk']; steps+=1
    vs0=x['prevk'] if x['prevk'] and x['prevk']['n']=='main' else x
    chain=e0['e']-vs0['s']; burst=b['last_end']-b['first']
    gap_gate=b['first']-e0['e']
    # counterfactual: side seg k starts at max(host launch end of its VS + LAT, vs0 start)
    cf_end=0; host_late=0; nhost=0; earliest_host=None
    stream_end={}
    for sg in sorted(b['segs'], key=lambda g:g['s']):
        h=rt.get(sg['vs']['cid'])
        if not h: continue
        nhost+=1
        hl=h[1]
        if earliest_host is None or hl<earliest_host: earliest_host=hl
        cf_s=max(hl+LAT, vs0['s'], stream_end.get(sg['st'],0))
        ce=cf_s+(sg['e']-sg['s'])
        stream_end[sg['st']]=ce
        cf_end=max(cf_end, ce)
        if hl+LAT>e0['e']: host_late+=1
    if nhost==0: continue
    maxseg=max(sg['e']-sg['s'] for sg in b['segs'])
    perstream=collections.Counter()
    for sg in b['segs']: perstream[sg['st']]+=sg['e']-sg['s']
    maxstream=max(perstream.values())
    gain_shift=min(chain+gap_gate, burst+gap_gate)
    # main resumes after join: next main kernel after burst end; its host launch time
    jn=bisect.bisect_left(mstart,b['last_end']); nm=None
    for m in main[jn:jn+30]:
        if m['kind']=='K': nm=m; break
    resume_gap = (nm['s']-b['last_end']) if nm else None
    nm_host = rt.get(nm['cid'])[1]-b['last_end'] if nm and rt.get(nm['cid']) else None
    actual_end=max(b['last_end'],e0['e'])
    cf_join=max(cf_end,e0['e'])
    gain_cf=max(0,actual_end-cf_join)
    # host cap: if the next main kernel was launched by host after actual join, the device wait was host-bound; gain capped by (actual_end - nm_host_launch) ... i.e. gain_collect = min(gain_cf, max(0, actual_end + resume_gap - (nm_host_end+LAT)))
    if nm_host is not None:
        slack = nm_host  # host launch end relative to burst end; positive => host issued after device finished
        gain_col = min(gain_cf, max(0, gain_cf - max(0, slack+LAT)))
    else: gain_col=gain_cf
    d=fo[k]; d['n']+=1; d['nsegs']+=len(b['segs']); d['seg0']+=chain; d['burst']+=burst; d['gain_naive']+=min(chain,burst)
    d['gain_cf']+=gain_cf; d['gain_col']+=gain_col; d['gain_shift']+=gain_shift; d['maxseg']+=maxseg; d['maxstream']+=maxstream; d['over8']+= (len(b['segs'])>8); d['host_late']+=host_late; d['gap_gate']+=gap_gate
    rows.append((k,chain/1e3,burst/1e3,len(b['segs']),(earliest_host-vs0['s'])/1e3,(e0['e']-vs0['s'])/1e3,host_late,gain_cf/1e3,gain_col/1e3, (nm_host or 0)/1e3, (resume_gap or 0)/1e3))
print("bursts",len(bursts),"with seg0 chain",sum(v['n'] for v in fo.values()),"no seg0 chain in front",nochain)
for key in ['n','nsegs','seg0','burst','gain_naive','gain_shift','gain_cf','gain_col','host_late','gap_gate','maxseg','maxstream','over8']:
    v=[fo[k][key] if k in fo else 0 for k in range(NF)]
    scale=1 if key in('n','nsegs','host_late','n_nochain','over8') else 1e-3
    print("  %-14s per frame: median=%.2f mean=%.2f p90=%.2f"%(key,med(v)*scale,statistics.mean(v)*scale,q(v,.9)*scale))
print("per-burst: seg0 chain median %.1f us; burst median %.1f; nsegs median %.1f"%(med([r[1] for r in rows]),med([r[2] for r in rows]),med([r[3] for r in rows])))
print("per-burst: earliest side-VS host launch end minus vs0 start (us): median %.1f p10 %.1f p90 %.1f"%(med([r[4] for r in rows]),q([r[4] for r in rows],.1),q([r[4] for r in rows],.9)))
print("per-burst: seg0 chain (vs0 start -> s3 end) median %.1f"%med([r[5] for r in rows]))
print("per-burst: fraction of side segs whose host launch+5us is after main s3 end: %.3f"%(sum(r[6] for r in rows)/sum(r[3] for r in rows)))
print("per-burst: main-resume host launch end minus burst end (us): median %.1f p10 %.1f p90 %.1f ; device resume gap median %.1f"%(med([r[9] for r in rows]),q([r[9] for r in rows],.1),q([r[9] for r in rows],.9),med([r[10] for r in rows])))
print("examples (frame, seg0, burst, nsegs, host_first-vs0, chain, host_late, gain_cf, gain_col, nm_host, resume_gap):")
for r in rows[:12]: print("  ",tuple(round(x,1) if isinstance(x,float) else x for x in r))
