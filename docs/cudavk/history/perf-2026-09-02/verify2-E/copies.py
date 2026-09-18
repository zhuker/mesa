import sqlite3, statistics, sys
from collections import defaultdict, Counter
def pct(v,p):
    v=sorted(v); return v[int(round((len(v)-1)*p))] if v else 0
def run(db, T0, T1, label):
    c=sqlite3.connect('file:'+db+'?mode=ro',uri=True); cur=c.cursor()
    # frames via worker cuEventSynchronize (closure-audit method)
    tids=Counter(t for (t,) in cur.execute('select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in (select id from StringIds where value="cuEventSynchronize")'))
    wt=tids.most_common(1)[0][0]
    wsync=[e for (e,) in cur.execute('select end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId in (select id from StringIds where value="cuEventSynchronize") order by start',(wt,))]
    if T0==0:
        lo=2; hi=len(wsync)-1
        if (hi-lo)%2: hi-=1
        T0=wsync[lo]; T1=wsync[hi]
    nsub=sum(1 for t in wsync if T0<=t<=T1)
    frames=(nsub-1)/2 if T0!=0 else (nsub-1)/2
    if label=='postdead': frames=411.5
    print(f'\n##### {label} window [{T0},{T1}) frames={frames:.1f} worker submits in window={nsub}')
    # main thread = thread with most cuLaunchKernel
    mt=Counter(t for (t,) in cur.execute('select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in (select id from StringIds where value like "cuLaunchKernel%")')).most_common(1)[0][0]
    ops=[]
    for st,s,e,cid,name,gx,api,ae in cur.execute(f'''select k.streamId,k.start,k.end,k.correlationId,sn.value,k.gridX,ra.value,r.end
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sn on sn.id=k.shortName
        left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=k.correlationId left join StringIds ra on ra.id=r.nameId
        where k.start>={T0-50000000} and k.start<{T1}'''):
        ops.append([st,s,e,'K',name,gx,api or '',ae,cid])
    for st,s,e,cid,ck,b,api,rs,re_ in cur.execute(f'''select m.streamId,m.start,m.end,m.correlationId,m.copyKind,m.bytes,ra.value,r.start,r.end from CUPTI_ACTIVITY_KIND_MEMCPY m
        left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=m.correlationId left join StringIds ra on ra.id=r.nameId
        where m.copyKind not in (11,12) and m.start>={T0-50000000} and m.start<{T1}'''):
        ops.append([st,s,e,'C',{1:'HtoD',2:'DtoH',8:'DtoD',7:'DtoA'}.get(ck,str(ck)),b,api or '',re_,cid,rs])
    for st,s,e,cid,b in cur.execute(f'''select streamId,start,end,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where start>={T0-50000000} and start<{T1}'''):
        ops.append([st,s,e,'S','memset',b,'',None,cid])
    bys=defaultdict(list)
    for o in ops: bys[o[0]].append(o)
    kcount=Counter(); 
    for st in bys: bys[st].sort(key=lambda o:o[1])
    # kernel count per stream
    main_stream=Counter(o[0] for o in ops if o[3]=='K' and o[1]>=T0).most_common(1)[0][0]
    print('main stream', main_stream)
    rows=[]   # per HtoD copy: stream, bytes, api, next kernel class, pred kind, device-paced?, gap total, api host dur
    def classify(lst,i):
        # kernel at lst[i]
        k=lst[i]; name=k[4]
        if name!='main': return name
        # next kernel on stream
        nxt=None
        for j in range(i+1,min(i+4,len(lst))):
            if lst[j][3]=='K': nxt=lst[j][4]; break
        prv=None
        for j in range(i-1,max(i-6,-1),-1):
            if lst[j][3]=='K': prv=lst[j][4]; break
        if prv and ('fs_compact' in prv or 'fs_interp' in prv): return 'main_FS'
        if nxt and nxt.startswith('cp_clip'): return 'main_VS'
        if prv and 'abuf' in prv: return 'main_FSabuf'
        return f'main_other(prev={prv},next={nxt})'
    for st,lst in bys.items():
        for i,o in enumerate(lst):
            if not(o[3]=='C' and o[4]=='HtoD' and o[1]>=T0): continue
            nk=None
            for j in range(i+1,len(lst)):
                if lst[j][3]=='K': nk=j; break
                if lst[j][3]=='C' and lst[j][4]=='HtoD': break
            cls=classify(lst,nk) if nk is not None else 'none'
            immediate = nk==i+1
            pred=lst[i-1] if i>0 else None
            devpaced=None; gap=None; excess=None
            if pred is not None and nk is not None and pred[3]=='K':
                k=lst[nk]
                devpaced = (k[7] is not None and k[7] < pred[2])   # launch API returned before predecessor kernel end
                gap = k[1]-pred[2]
            rows.append(dict(st=st,bytes=o[5],api=o[6],cls=cls,imm=immediate,pred=pred[3] if pred else None,dev=devpaced,gap=gap,hostdur=(o[7]-o[9]) if o[7] and len(o)>9 and o[9] else None, side=(st!=main_stream)))
    n=len(rows); print(f'HtoD copies: {n} = {n/frames:.2f}/frame; on main stream {sum(1 for r in rows if not r["side"])/frames:.2f}, side {sum(1 for r in rows if r["side"])/frames:.2f}')
    print(f'  api: {Counter(r["api"] for r in rows).most_common(4)}')
    b=[r['bytes'] for r in rows]; print(f'  bytes p50 {pct(b,.5)} p90 {pct(b,.9)} p99 {pct(b,.99)} max {max(b)} mean {statistics.mean(b):.0f}; >4096: {sum(1 for x in b if x>4096)/frames:.2f}/frame ({100*sum(1 for x in b if x>4096)/n:.1f}%)')
    hd=[r['hostdur'] for r in rows if r['hostdur']]; print(f'  host API dur (CUPTI) mean {statistics.mean(hd)/1000:.2f} p50 {pct(hd,.5)/1000:.2f} us')
    print('  by next-kernel class: n/frame, bytes p50/p90/max, >4KB/frame, immediate-next%, main-stream%, device-paced% (of those with kernel pred), gap p50 devpaced (us)')
    byc=defaultdict(list)
    for r in rows: byc[r['cls']].append(r)
    for cls,rs in sorted(byc.items(), key=lambda kv:-len(kv[1])):
        bb=[r['bytes'] for r in rs]; wp=[r for r in rs if r['dev'] is not None]; dp=[r for r in wp if r['dev']]
        g=[r['gap'] for r in dp]
        print(f'   {cls[:40]:40s} {len(rs)/frames:7.2f}  {pct(bb,.5):6d}/{pct(bb,.9):6d}/{max(bb):6d}  >4K {sum(1 for x in bb if x>4096)/frames:5.2f}  imm {100*sum(1 for r in rs if r["imm"])/len(rs):5.1f}%  main {100*sum(1 for r in rs if not r["side"])/len(rs):5.1f}%  devpaced {100*len(dp)/max(1,len(wp)):5.1f}% (n={len(wp)})  gap p50 {pct(g,.5)/1000 if g else 0:.2f} mean {statistics.mean(g)/1000 if g else 0:.2f}')
    for cls in ['main_FS','cp_fs_compact','main_VS']:
        rs=[r for r in byc.get(cls,[]) if r['dev'] and not r['side']]
        ex=[max(0,r['gap']-1860) for r in rs]; exp=[max(0,r['gap']-260) for r in rs]
        cap=[min(r['gap'],20000) for r in rs]
        print(f'   excess over plain-link p50 (1.86us), device-paced main-stream {cls}: n/f {len(rs)/frames:.2f} sum {sum(ex)/frames/1e6:.3f} ms/frame (gaps capped at 20us: {sum(max(0,g-1860) for g in cap)/frames/1e6:.3f}); over PDL p50 0.26: {sum(exp)/frames/1e6:.3f} (capped {sum(max(0,g-260) for g in cap)/frames/1e6:.3f})')
    # plain kernel->kernel links on main stream, device paced, for baseline
    plain=[]; pdl=[]
    for st,lst in bys.items():
        for i in range(1,len(lst)):
            o=lst[i]; p=lst[i-1]
            if o[3]=='K' and p[3]=='K' and o[1]>=T0 and o[7] and o[7]<p[2]:
                (pdl if o[6].startswith('cuLaunchKernelEx') else plain).append(o[1]-p[2])
    print(f'  baseline device-paced kernel->kernel gap: plain p50 {pct(plain,.5)/1000:.2f} mean {statistics.mean(plain)/1000:.2f} (n/f {len(plain)/frames:.1f}); PDL p50 {pct(pdl,.5)/1000:.2f} (n/f {len(pdl)/frames:.1f})')
    # kernels per frame, main per frame, main after HtoD
    ks=[o for o in ops if o[3]=='K' and o[1]>=T0]
    print(f'  kernels/frame {len(ks)/frames:.1f}; main/frame {sum(1 for o in ks if o[4]=="main")/frames:.2f}; fs_compact/frame {sum(1 for o in ks if o[4]=="cp_fs_compact")/frames:.2f}; vertex_fetch/frame {sum(1 for o in ks if o[4]=="cp_vertex_fetch")/frames:.2f}')
    # main kernels NOT preceded by HtoD copy
    notpre=Counter()
    for st,lst in bys.items():
        for i,o in enumerate(lst):
            if o[3]=='K' and o[4]=='main' and o[1]>=T0:
                p=lst[i-1] if i>0 else None
                if not(p and p[3]=='C' and p[4]=='HtoD'): notpre[(p[3],p[4]) if p else None]+=1
    print(f'  main launches not immediately after an HtoD: {sum(notpre.values())/frames:.2f}/frame {notpre.most_common(5)}')
    # host: total cuMemcpyHtoDAsync API time per frame on main thread
    tot=cur.execute(f'select count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid={mt} and nameId in (select id from StringIds where value like "cuMemcpyHtoDAsync%") and start>={T0} and start<{T1}').fetchone()
    print(f'  cuMemcpyHtoDAsync API on main thread: {tot[0]/frames:.2f}/frame, {tot[1]/frames/1e6:.3f} ms/frame (CUPTI-inflated)')
    tot=cur.execute(f'select count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid={mt} and nameId in (select id from StringIds where value like "cuLaunchKernel%") and start>={T0} and start<{T1}').fetchone()
    print(f'  cuLaunchKernel* API on main thread: {tot[0]/frames:.2f}/frame, {tot[1]/frames/1e6:.3f} ms/frame')
run('/tmp/postdead/trace.sqlite', 9286588499, 12340525407, 'postdead')
run('/tmp/postpin-rtx2/trace.sqlite', 0, 0, 'postpin')
