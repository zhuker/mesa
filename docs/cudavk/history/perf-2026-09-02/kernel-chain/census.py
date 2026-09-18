import sqlite3, statistics, collections, json, sys
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
S0=1503511535334936
ts=[l.split() for l in open('/tmp/postdead/submit-ts.txt') if len(l.split())==2]
rel=[int(a)-S0 for a,b in ts]
# frames: external submits 2782+2k .. 2784+2k, k=0..411 (last full frame ends at rel[3606])
NF=412
fb=[(rel[2782+2*k], rel[2784+2*k]) for k in range(NF)]
W0=fb[0][0]; W1=fb[-1][1]
print("window", W0, W1, (W1-W0)/1e6, "ms;", NF, "frames; median frame ms", statistics.median([(b-a)/1e6 for a,b in fb]))
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
K=[]
for r in c.execute(f"select start,end,streamId,shortName,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,correlationId,launchType from CUPTI_ACTIVITY_KIND_KERNEL where start>={W0} and start<{W1} order by start"):
    K.append(dict(s=r[0],e=r[1],st=r[2],n=names[r[3]],g=r[4]*r[5]*r[6],b=r[7]*r[8]*r[9],gx=r[4],bx=r[7],reg=r[10],cid=r[11],lt=r[12],kind='K'))
M=[]
for r in c.execute(f"select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>={W0} and start<{W1} order by start"):
    M.append(dict(s=r[0],e=r[1],st=r[2],n='memcpy%d'%r[3],ck=r[3],bytes=r[4],cid=r[5],kind='M'))
S=[]
for r in c.execute(f"select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>={W0} and start<{W1} order by start"):
    S.append(dict(s=r[0],e=r[1],st=r[2],n='memset',bytes=r[3],cid=r[4],kind='S'))
print("kernels",len(K),"memcpy",len(M),"memset",len(S))
print("kernels/frame", len(K)/NF, "memcpy/frame", len(M)/NF, "memset/frame", len(S)/NF)
# stream classification
import itertools
allops=sorted(K+M+S,key=lambda o:o['s'])
bystream=collections.defaultdict(list)
for o in allops: bystream[o['st']].append(o)
# same-stream prev kernel gap and prev any-op gap
for st,ops in bystream.items():
    prevk=None; prev=None
    for o in ops:
        o['gap_any']= (o['s']-prev['e']) if prev else None
        o['prev_any']= prev['n'] if prev else None
        o['gap_k']= (o['s']-prevk['e']) if prevk else None
        o['prev_k']= prevk['n'] if prevk else None
        if o['kind']=='K':
            if prevk: prevk['next_k']=o['n']
            prevk=o
        prev=o
def med(x): 
    x=[v for v in x if v is not None]
    return statistics.median(x) if x else None
def q(x,p):
    x=sorted(v for v in x if v is not None)
    return x[int(p*(len(x)-1))] if x else None
# --- per kernel name census, main split by block size & gridX class & neighbours
def mainclass(o):
    if o['n']!='main': return o['n']
    # VS: launched (total_verts+255)/256 capped 4096, block 256 -> grid varies; FS direct: fixed grid min((max_pixels+255)/256,4096)=4096 for 1280x720? max_pixels=ALIGN(921600,256)*2=1843200 -> 7200 blocks -> capped 4096. FS abuf: (num_slots+255)/256 varies.
    nk=o.get('next_k'); pk=o.get('prev_k')
    if pk=='cp_fs_compact': return 'main[FS direct: after fs_compact]'
    if pk=='cp_abuf_interpolate' or pk=='cp_abuf_interpolate_ranges' or pk=='cp_abuf_prepare_shade_count': return 'main[FS abuf: after interp/prepare]'
    if nk in ('cp_clip_rast_fused','cp_clip_rast_fused_abuf','cp_clip_triangles','cp_rasterize_stage1','cp_rasterize_stage1_abuf'): return 'main[VS: before clip/rast]'
    if nk=='cp_fs_writeback': return 'main[FS direct: before writeback]'
    if pk=='cp_vertex_fetch': return 'main[VS: after vertex_fetch]'
    return 'main[other pk=%s nk=%s]'%(pk,nk)
rows=collections.defaultdict(list)
for o in K: rows[mainclass(o)].append(o)
print("\n=== per-kernel census (window %d frames) ==="%NF)
print("%-45s %8s %9s %9s %9s %10s %9s %9s %9s %s"%("kernel","launch/f","med_us","p90_us","sum_ms/f","med_threads","med_gapK","med_gapAny","p90_gapK","streams(main13/side)"))
out_rows=[]
for n,ops in sorted(rows.items(), key=lambda kv:-sum(o['e']-o['s'] for o in kv[1])):
    d=[o['e']-o['s'] for o in ops]
    thr=[o['g']*o['b'] for o in ops]
    gk=[o['gap_k'] for o in ops]; ga=[o['gap_any'] for o in ops]
    main=sum(1 for o in ops if o['st']==13); side=len(ops)-main
    r=dict(kernel=n, launches_per_frame=len(ops)/NF, med_us=med(d)/1e3, p90_us=q(d,0.9)/1e3, sum_ms_per_frame=sum(d)/1e6/NF, med_threads=med(thr), med_grid=med([o['g'] for o in ops]), med_block=med([o['b'] for o in ops]), med_gap_prev_kernel_us=(med(gk) or 0)/1e3, med_gap_prev_anyop_us=(med(ga) or 0)/1e3, p90_gap_prev_kernel_us=(q(gk,0.9) or 0)/1e3, main_stream=main, side_streams=side, med_regs=med([o['reg'] for o in ops]))
    out_rows.append(r)
    print("%-45s %8.2f %9.2f %9.2f %9.3f %10.0f %9.2f %9.2f %9.2f %d/%d"%(n,r['launches_per_frame'],r['med_us'],r['p90_us'],r['sum_ms_per_frame'],r['med_threads'],r['med_gap_prev_kernel_us'],r['med_gap_prev_anyop_us'],r['p90_gap_prev_kernel_us'],main,side))
json.dump(out_rows, open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/kernel-chain/census_rows.json','w'), indent=1)
# main by grid size histogram
print("\n=== main by gridX bucket ===")
gb=collections.defaultdict(list)
for o in K:
    if o['n']=='main': gb[(o['gx'] if o['gx']<4096 else 4096, o['bx'])].append(o)
for k,ops in sorted(gb.items(), key=lambda kv:-len(kv[1]))[:25]:
    d=[o['e']-o['s'] for o in ops]
    print(k, "n/f=%.2f med_us=%.2f sum_ms/f=%.3f classes=%s"%(len(ops)/NF, med(d)/1e3, sum(d)/1e6/NF, dict(collections.Counter(mainclass(o) for o in ops).most_common(3))))
# --- latency shell: duration vs threads
print("\n=== latency shell: duration vs thread count (all kernels) ===")
bins=[(0,1024),(1024,4096),(4096,20000),(20000,100000),(100000,500000),(500000,2000000),(2000000,10**12)]
for lo,hi in bins:
    ops=[o for o in K if lo<=o['g']*o['b']<hi]
    if not ops: continue
    d=[o['e']-o['s'] for o in ops]
    print("threads [%d,%d): n/f=%.1f med_us=%.2f p10=%.2f p90=%.2f share>5us=%.2f names=%s"%(lo,hi,len(ops)/NF,med(d)/1e3,q(d,0.1)/1e3,q(d,0.9)/1e3,sum(1 for x in d if x>5000)/len(d), dict(collections.Counter(o['n'] for o in ops).most_common(4))))
shell=[o for o in K if o['g']*o['b']<20000 and o['e']-o['s']>5000]
print("latency shells (<20k threads, >5us): n/f=%.2f sum_ms/f=%.3f by name=%s"%(len(shell)/NF, sum(o['e']-o['s'] for o in shell)/1e6/NF, dict(collections.Counter(o['n'] for o in shell).most_common(12))))
# minimum durations per kernel
print("\n=== min/p5 duration per kernel (floor) ===")
for n,ops in sorted(rows.items()):
    d=sorted(o['e']-o['s'] for o in ops)
    print("%-45s min=%.2f p5=%.2f p50=%.2f us"%(n,d[0]/1e3,d[int(0.05*(len(d)-1))]/1e3,d[len(d)//2]/1e3))
# --- per stream busy per frame
def union(iv):
    iv=sorted(iv); tot=0; cs=None; ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return tot
def clip(o,a,b): return (max(o['s'],a),min(o['e'],b))
print("\n=== per-frame per-stream busy (median over frames, ms) ===")
import bisect
starts=[o['s'] for o in allops]
perframe=[]
for k,(a,b) in enumerate(fb):
    i0=bisect.bisect_left(starts,a-5_000_000); i1=bisect.bisect_left(starts,b)
    ops=[o for o in allops[i0:i1] if o['e']>a and o['s']<b]
    d={}
    d['frame_ms']=(b-a)/1e6
    d['union_all']=union([clip(o,a,b) for o in ops])/1e6
    d['union_kernels']=union([clip(o,a,b) for o in ops if o['kind']=='K'])/1e6
    d['main13_all']=union([clip(o,a,b) for o in ops if o['st']==13])/1e6
    d['main13_kernels']=union([clip(o,a,b) for o in ops if o['st']==13 and o['kind']=='K'])/1e6
    side=[clip(o,a,b) for o in ops if o['st'] in range(14,22)]
    d['side_union']=union(side)/1e6
    d['uvm_stream0']=union([clip(o,a,b) for o in ops if o['st']==0])/1e6
    d['stream7']=union([clip(o,a,b) for o in ops if o['st']==7])/1e6
    d['union_excl_uvm']=union([clip(o,a,b) for o in ops if o['st']!=0])/1e6
    # main idle while side busy: union(main ∪ side) - union(main)
    d['main_or_side']=union([clip(o,a,b) for o in ops if o['st']==13 or o['st'] in range(14,22)])/1e6
    d['main_idle_side_busy']=d['main_or_side']-d['main13_all']
    d['device_idle_excl_uvm']=d['frame_ms']-d['union_excl_uvm']
    d['device_idle_all']=d['frame_ms']-d['union_all']
    d['n_kernels']=sum(1 for o in ops if o['kind']=='K' and a<=o['s']<b)
    d['n_main_kernels']=sum(1 for o in ops if o['kind']=='K' and o['st']==13 and a<=o['s']<b)
    d['n_main_ops']=sum(1 for o in ops if o['st']==13 and a<=o['s']<b)
    d['n_side_kernels']=sum(1 for o in ops if o['kind']=='K' and o['st'] in range(14,22) and a<=o['s']<b)
    for st in range(14,22):
        d['side%d'%st]=union([clip(o,a,b) for o in ops if o['st']==st])/1e6
    perframe.append(d)
keys=list(perframe[0].keys())
for kk in keys:
    v=[d[kk] for d in perframe]
    print("%-24s median=%.3f mean=%.3f p10=%.3f p90=%.3f"%(kk,statistics.median(v),statistics.mean(v),q(v,0.1),q(v,0.9)))
json.dump(perframe, open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/kernel-chain/perframe.json','w'))
# heavy band inside the window? frames 1391..1802 -> the doc says heavy is 2150-3150, so this window is the light/moderate band. report frame-time distribution
v=[d['frame_ms'] for d in perframe]
print("\nframe ms deciles:", [round(q(v,p/10),2) for p in range(11)])
