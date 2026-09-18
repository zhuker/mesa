import sqlite3, collections, bisect, sys, statistics
DB, TS, S0, S1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c=sqlite3.connect('file:%s?mode=ro'%DB,uri=True)
sess=c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts=[int(l.split()[0])-sess for l in open(TS) if len(l.split())==2]
W0,W1=ts[S0],ts[S1]; NF=(S1-S0)//2
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
K=[dict(s=r[0],e=r[1],st=r[2],n=names[r[3]],gx=r[4],cid=r[5]) for r in c.execute("select start,end,streamId,shortName,gridX,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start",(W0,W1))]
MS={r[3]:(r[0],r[1],r[2],r[4]) for r in c.execute("select start,end,streamId,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",(W0,W1))}
apis=[dict(s=r[0],e=r[1],n=names[r[2]],cid=r[3],tid=r[4]) for r in c.execute("select start,end,nameId,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<? order by start",(W0,W1))]
api={a['cid']:a for a in apis}; aidx={a['cid']:i for i,a in enumerate(apis)}
bys=collections.defaultdict(list)
for o in K: bys[o['st']].append(o)
main=bys[13]; mstart=[o['s'] for o in main]
follow=collections.Counter(); pre=collections.Counter(); nb=0; big_memset_pre=0; memset_pre_bytes=collections.Counter()
for st,l in bys.items():
    if st==13: continue
    for j,o in enumerate(l):
        if o['n']!='cp_clip_triangles': continue
        k=j-1
        if k<0 or l[k]['n']!='main': continue
        vs=l[k]; nb+=1
        # follow-up on main: kernels after the re-exec VS until the next main 'main' VS with a clip successor
        i=bisect.bisect_left(mstart,o['e']); re=None
        for m in main[i:i+40]:
            if m['n']=='main' and m['gx']==vs['gx']: re=m; break
        if not re: continue
        ri=main.index(re); seq=[]
        for m in main[ri+1:ri+40]:
            if m['n']=='main' and m is not re and main[main.index(m)+1]['n'] in ('cp_clip_rast_fused','cp_clip_rast_fused_abuf','cp_vertex_fetch','cp_clip_triangles'): break
            seq.append(m['n'])
        s=set(seq)
        if 'cp_rasterize_stage1' in s: follow['re-pass chain (retry/peel, stage1 unfused)']+=1
        elif 'cp_clip_rast_fused_abuf' in s: follow['standalone abuf']+=1
        elif 'cp_clip_rast_fused' in s: follow['direct chain']+=1
        else: follow['other:'+','.join(seq[:3])]+=1
        # host sequence before the side-VS launch: pattern of preceding API names until previous cuLaunchKernel
        ai=aidx.get(vs['cid'])
        if ai is None: continue
        names_before=[]
        for a in reversed(apis[max(0,ai-12):ai]):
            if a['n'] in ('cuLaunchKernel',): break
            names_before.append(a['n'])
            if a['n']=='cuMemsetD32Async' and a['cid'] in MS and MS[a['cid']][2]==13:
                memset_pre_bytes[MS[a['cid']][3]]+=1
        pre[' '.join(reversed(names_before))]+=1
print("backouts",nb,"per frame %.2f"%(nb/NF))
print("re-exec follow-up on main:",[(k,v,round(v/NF,2)) for k,v in follow.most_common()])
print("host API pattern before side VS launch (top):"); 
for k,v in pre.most_common(6): print("  %5d %.2f/f  %s"%(v,v/NF,k))
print("main-stream memsets in that pre-pattern by bytes:",memset_pre_bytes.most_common(6))
