import sqlite3, collections, bisect, sys, statistics
DB, TS, S0, S1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c=sqlite3.connect('file:%s?mode=ro'%DB,uri=True)
sess=c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts=[int(l.split()[0])-sess for l in open(TS) if len(l.split())==2]
W0,W1=ts[S0],ts[S1]; NF=(S1-S0)//2
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
ops=[dict(s=r[0],e=r[1],st=r[2],n=names[r[3]],gx=r[4],k='K') for r in c.execute("select start,end,streamId,shortName,gridX from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start",(W0,W1))]
ops+=[dict(s=r[0],e=r[1],st=r[2],n='memcpy',gx=0,k='M') for r in c.execute("select start,end,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and streamId!=0",(W0,W1))]
ops+=[dict(s=r[0],e=r[1],st=r[2],n='memset',gx=0,k='S') for r in c.execute("select start,end,streamId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",(W0,W1))]
ops.sort(key=lambda o:o['s'])
bys=collections.defaultdict(list)
for o in ops: bys[o['st']].append(o)
main=bys[13]; mstart=[o['s'] for o in main]
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
nb=0; withfetch=0; fetch_dur=0; ext_idle=0; base_idle=0; side_all=collections.Counter()
for st,l in bys.items():
    if st==13: continue
    for j,o in enumerate(l):
        if o['n']!='cp_clip_triangles': continue
        k=j-1
        while k>=0 and l[k]['k']!='K': k-=1
        if k<0 or l[k]['n']!='main': continue
        nb+=1
        # extended start: walk back over non-kernel ops and a cp_vertex_fetch kernel, gap<20us
        f=k
        while f-1>=0 and (l[f-1]['k']!='K' or l[f-1]['n']=='cp_vertex_fetch') and l[f]['s']-l[f-1]['e']<20000:
            f-=1
        had=any(l[x]['n']=='cp_vertex_fetch' for x in range(f,k))
        i=bisect.bisect_left(mstart,o['e']); re=None
        for m in main[i:i+40]:
            if m['k']=='K' and m['n']=='main' and m['gx']==l[k]['gx']: re=m; break
        if not re: continue
        # baseline window as chains.py: first non-kernel op before VS
        f0=k
        while f0-1>=0 and l[f0-1]['k']!='K' and l[f0]['s']-l[f0-1]['e']<20000: f0-=1
        base_idle+=main_idle(l[f0]['s'],re['s']); ext_idle+=main_idle(l[f]['s'],re['s'])
        if had: withfetch+=1; fetch_dur+=sum(l[x]['e']-l[x]['s'] for x in range(f,k) if l[x]['n']=='cp_vertex_fetch')
# all side-stream cp_vertex_fetch
sf=[o for o in ops if o['n']=='cp_vertex_fetch' and o['st']!=13]
print("backouts %d (%.2f/f); with a side cp_vertex_fetch ahead of the VS: %d (%.2f/f), fetch device time %.1f us/f"%(nb,nb/NF,withfetch,withfetch/NF,fetch_dur/1e3/NF))
print("all side-stream cp_vertex_fetch in window: %d (%.2f/f)"%(len(sf),len(sf)/NF))
print("main idle in window per frame (mean): chains.py-style %.1f us; extended to include the fetch %.1f us"%(base_idle/1e3/NF, ext_idle/1e3/NF))
