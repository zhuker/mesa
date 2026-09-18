import sqlite3, statistics, collections, json, bisect, sys
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
S0=1503511535334936
ts=[l.split() for l in open('/tmp/postdead/submit-ts.txt') if len(l.split())==2]
rel=[int(a)-S0 for a,b in ts]
NF=412; fb=[(rel[2782+2*k], rel[2784+2*k]) for k in range(NF)]
W0=fb[0][0]; W1=fb[-1][1]
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
FS=[a for a,b in fb]
def fidx(t):
    i=bisect.bisect_right(FS,t)-1
    return i if 0<=i<NF and t<fb[i][1] else -1
def med(x):
    x=[v for v in x if v is not None]; return statistics.median(x) if x else 0
def q(x,p):
    x=sorted(v for v in x if v is not None); return x[int(p*(len(x)-1))] if x else 0
LO=W0-20000000
# runtime: issue times
rt={}
for r in c.execute(f"select correlationId,start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>={LO} and start<{W1}"):
    rt[r[0]]=(r[1],r[2],names[r[3]])
ev=[]  # issue events
for r in c.execute(f"select k.start,k.end,k.streamId,s.value,k.correlationId,k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>={LO} and k.start<{W1}"):
    if r[4] in rt: ev.append(dict(t=rt[r[4]][0],kind='op',s=r[0],e=r[1],st=r[2],n=r[3],cid=r[4],gx=r[5]))
for r in c.execute(f"select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>={LO} and start<{W1} and streamId!=0"):
    if r[5] in rt: ev.append(dict(t=rt[r[5]][0],kind='op',s=r[0],e=r[1],st=r[2],n='memcpy%d'%r[3],cid=r[5],gx=r[4]))
for r in c.execute(f"select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>={LO} and start<{W1}"):
    if r[4] in rt: ev.append(dict(t=rt[r[4]][0],kind='op',s=r[0],e=r[1],st=r[2],n='memset',cid=r[4],gx=r[3]))
for r in c.execute(f"select timestamp,streamId,eventId,correlationId from CUPTI_ACTIVITY_KIND_CUDA_EVENT where timestamp>={LO} and timestamp<{W1}"):
    if r[3] in rt: ev.append(dict(t=rt[r[3]][0],kind='rec',st=r[1],eid=r[2],cid=r[3]))
for r in c.execute(f"select start,end,streamId,syncType,eventId,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>={LO} and start<{W1}"):
    if r[5] in rt: ev.append(dict(t=rt[r[5]][0],kind='wait' if r[3]==2 else 'sync',s=r[0],e=r[1],st=r[2],typ=r[3],eid=r[4],cid=r[5]))
ev.sort(key=lambda d:d['t'])
print("issue events",len(ev))
# DAG evaluation with per-frame reset. Node fields: es, ef, pred, depth
last_op={}           # stream -> last issued op node
pending={}           # stream -> list of extra pred nodes for next op
rec_snap={}          # eventId -> node snapshot at record
barrier=None         # extra pred node (host sync) for every stream's next op
barrier_streams=set()
LAM=float(sys.argv[1]) if len(sys.argv)>1 else 0.0  # per-edge latency in ns
frames=collections.defaultdict(list)
def node_ef(n): return n['ef'] if n else 0
for d in ev:
    if d['kind']=='rec':
        rec_snap[d['eid']]=last_op.get(d['st'])
    elif d['kind']=='wait':
        src=rec_snap.get(d['eid'])
        if src: pending.setdefault(d['st'],[]).append(src)
    elif d['kind']=='sync':
        if d['typ']==3: tg=[last_op.get(d['st'])]
        elif d['typ']==1: tg=[rec_snap.get(d['eid'])]
        else: tg=list(last_op.values())
        tg=[t for t in tg if t]
        if tg:
            barrier=max(tg,key=lambda n:n['ef']); barrier_streams=set(last_op.keys())|{7,13}|set(range(14,22))
            d['b']=barrier
    else:
        st=d['st']; k=fidx(d['s']); d['f']=k
        preds=[]
        p=last_op.get(st)
        if p: preds.append(p)
        preds+=pending.pop(st,[])
        if barrier is not None and st in barrier_streams:
            preds.append(barrier); barrier_streams.discard(st)
        # per-frame reset: predecessors from other frames contribute 0
        best=None; bes=0
        for pn in preds:
            if pn.get('f')==k and pn['ef']+LAM>bes: bes=pn['ef']+LAM; best=pn
        d['es']=bes; d['ef']=bes+(d['e']-d['s']); d['pred']=best; d['depth']=(best['depth']+1) if best else 1
        last_op[st]=d
        if k>=0: frames[k].append(d)
res=[]
pathclass=collections.Counter(); pathdur=collections.Counter()
for k in range(NF):
    L=frames.get(k,[])
    if not L: continue
    top=max(L,key=lambda n:n['ef'])
    # walk the path
    n=top; cnt=0; kcnt=0; mainops=0; cls=collections.Counter(); cdur=collections.Counter()
    while n:
        cnt+=1
        if not n['n'].startswith('mem'): kcnt+=1
        if n['st']==13: mainops+=1
        nm=n['n'] if n['n']!='main' else ('main[FS]' if n['pred'] and n['pred']['n'] in ('cp_fs_compact','cp_abuf_interpolate','cp_abuf_seg_scatter','memcpy1') and n['gx']>=100 else 'main[VS/other]')
        cls[nm]+=1; cdur[nm]+=n['e']-n['s']
        n=n['pred']
    pathclass.update(cls); pathdur.update(cdur)
    res.append(dict(f=k, frame_ms=(fb[k][1]-fb[k][0])/1e6, cp_ms=top['ef']/1e6, nodes=cnt, kernels=kcnt, mainops=mainops, ops=len(L)))
print("per-frame critical path (device durations only, lambda=%.1f us per dependent edge):"%(LAM/1e3))
for key in ['frame_ms','cp_ms','nodes','kernels','mainops','ops']:
    v=[r[key] for r in res]; print("  %-9s median=%.3f mean=%.3f p10=%.3f p90=%.3f"%(key,med(v),statistics.mean(v),q(v,.1),q(v,.9)))
v=[r['cp_ms']/r['frame_ms'] for r in res]; print("  cp/frame  median=%.3f"%med(v))
print("critical-path node classes (per frame, mean): count, ms")
tot=sum(pathclass.values())
for nm,cnt in pathclass.most_common(25):
    print("  %-28s %7.2f/frame  %.3f ms/frame"%(nm,cnt/len(res),pathdur[nm]/1e6/len(res)))
json.dump(res, open('dag_lambda%g.json'%(LAM/1e3),'w'))
