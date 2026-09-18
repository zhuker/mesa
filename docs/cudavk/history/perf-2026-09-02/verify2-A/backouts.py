import sqlite3, statistics, collections, bisect, sys
DB, TS, S0, S1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c=sqlite3.connect('file:%s?mode=ro'%DB,uri=True)
sess=c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts=[int(l.split()[0])-sess for l in open(TS) if len(l.split())==2]
NF=(S1-S0)//2; fb=[(ts[S0+2*k],ts[S0+2*k+2]) for k in range(NF)]
W0,W1=fb[0][0],fb[-1][1]
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
def med(x): x=[v for v in x if v is not None]; return statistics.median(x) if x else 0
def q(x,p): x=sorted(v for v in x if v is not None); return x[int(p*(len(x)-1))] if x else 0
FS=[a for a,b in fb]
def fidx(t):
    i=bisect.bisect_right(FS,t)-1
    return i if 0<=i<NF and t<fb[i][1] else -1
ops=[]
for r in c.execute("select start,end,streamId,shortName,gridX,blockX,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start",(W0-3000000,W1)):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n=names[r[3]],gx=r[4],bx=r[5],cid=r[6],kind='K'))
for r in c.execute("select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and streamId!=0 order by start",(W0-3000000,W1)):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n='memcpy%d'%r[3],gx=r[4],bx=0,cid=r[5],kind='M'))
for r in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<? order by start",(W0-3000000,W1)):
    ops.append(dict(s=r[0],e=r[1],st=r[2],n='memset',gx=r[3],bx=0,cid=r[4],kind='S'))
ops.sort(key=lambda o:o['s'])
api={}
apis=[]
for r in c.execute("select r.start,r.end,r.nameId,r.correlationId,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r where r.start>=? and r.start<? order by r.start",(W0-3000000,W1)):
    a=dict(s=r[0],e=r[1],n=names[r[2]],cid=r[3],tid=r[4]); apis.append(a); api[r[3]]=a
astart=[a['s'] for a in apis]
streams=collections.Counter(o['st'] for o in ops if o['kind']=='K')
print("kernel streams:",sorted(streams.items(),key=lambda x:-x[1])[:12])
MAIN=13
bys=collections.defaultdict(list)
for o in ops: bys[o['st']].append(o)
for st,l in bys.items():
    for i,o in enumerate(l):
        o['prev']=l[i-1] if i else None; o['next']=l[i+1] if i+1<len(l) else None
main=bys[MAIN]; mstart=[o['s'] for o in main]; mend=[o['e'] for o in main]
SIDE=set(st for st in bys if st!=MAIN and st!=0 and st!=7)
def main_idle(a,b):
    i=bisect.bisect_left(mstart,a-3000000); j=bisect.bisect_left(mstart,b)
    iv=sorted((max(o['s'],a),min(o['e'],b)) for o in main[i:j] if o['e']>a and o['s']<b)
    tot=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return (b-a)-tot
# census of clip_triangles by stream inside window
ct=[o for o in ops if o['n']=='cp_clip_triangles' and W0<=o['s']<W1]
print("cp_clip_triangles in window:",len(ct),"per frame %.2f"%(len(ct)/NF),"by stream",collections.Counter(o['st'] for o in ct))
keys=['n','side_wall','main_idle','reVS','pre_gap','host_win','host_api_sum','host_block','host_block_n','re_lag','api_n','main_busy_win']
fa=collections.defaultdict(lambda: {k:0 for k in keys})
matched=unmatched=0; per=[]
for o in ct:
    if o['st'] not in SIDE: continue
    p=o['prev']
    while p and p['kind']!='K': p=p['prev']
    if not p or p['n']!='main': unmatched+=1; continue
    first=p
    while first['prev'] and first['prev']['kind']!='K' and first['s']-first['prev']['e']<20000: first=first['prev']
    j=bisect.bisect_left(mstart,o['e']); re=None
    for m in main[j:j+40]:
        if m['kind']=='K' and m['n']=='main' and m['gx']==p['gx']: re=m; break
    k=fidx(o['s'])
    if k<0 or not re: unmatched+=1; continue
    matched+=1
    d=fa[k]; d['n']+=1
    d['side_wall']+=o['e']-first['s']
    mi=main_idle(first['s'],re['s']); d['main_idle']+=mi
    d['main_busy_win']+=(re['s']-first['s'])-mi
    d['reVS']+=re['e']-re['s']
    # main idle gap before the excursion began: last main op ending before first.s
    i=bisect.bisect_left(mstart,first['s'])
    lastend=max((m['e'] for m in main[max(0,i-60):i]),default=first['s'])
    d['pre_gap']+=max(0,first['s']-lastend)
    # host side
    a0=api.get(p['cid']); a1=api.get(re['cid'])
    if a0 and a1:
        d['host_win']+=a1['s']-a0['s']
        d['re_lag']+=re['s']-a1['e']
        i0=bisect.bisect_left(astart,a0['s']); i1=bisect.bisect_left(astart,a1['s'])
        between=[a for a in apis[i0:i1] if a['tid']==a0['tid']]
        d['api_n']+=len(between)
        d['host_api_sum']+=sum(a['e']-a['s'] for a in between)
        bl=[a for a in between if 'Synchronize' in a['n'] or 'Query' in a['n']]
        d['host_block']+=sum(a['e']-a['s'] for a in bl); d['host_block_n']+=len(bl)
        per.append(dict(f=k,side_wall=(o['e']-first['s'])/1e3,main_idle=mi/1e3,host_win=(a1['s']-a0['s'])/1e3,api_sum=sum(a['e']-a['s'] for a in between)/1e3,re_lag=(re['s']-a1['e'])/1e3,apis=collections.Counter(a['n'] for a in between)))
print("matched",matched,"unmatched",unmatched,"frames",NF)
for key in keys:
    v=[fa[k][key] if k in fa else 0 for k in range(NF)]
    sc=1 if key in('n','host_block_n','api_n') else 1e-3
    print("  %-14s per frame: median=%8.2f mean=%8.2f p10=%8.2f p90=%8.2f"%(key,med(v)*sc,statistics.mean(v)*sc,q(v,.1)*sc,q(v,.9)*sc))
# per-occurrence distributions
for key in ['side_wall','main_idle','host_win','api_sum','re_lag']:
    v=[x[key] for x in per]
    print("  per-occurrence %-10s median=%7.1f mean=%7.1f p10=%7.1f p90=%7.1f us"%(key,med(v),statistics.mean(v),q(v,.1),q(v,.9)))
agg=collections.Counter()
for x in per: agg.update(x['apis'])
print("API calls between side-VS launch and re-exec launch (per occurrence):",[(n,round(cnt/len(per),2)) for n,cnt in agg.most_common(14)])
# main-stream re-exec: what follows (retry passes?) - count re-exec followed by rasterize_stage1 within next 12 main kernels
