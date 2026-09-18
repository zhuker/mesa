import sqlite3, statistics
from collections import defaultdict
def pct(v,p):
    v=sorted(v); return v[int(round((len(v)-1)*p))] if v else 0
for db,T0,T1,fr in [('/tmp/postdead/trace.sqlite',9286588499,12340525407,411.5),('/tmp/postpin-rtx2/trace.sqlite',18629876,8598737501,1024)]:
    c=sqlite3.connect('file:'+db+'?mode=ro',uri=True)
    ops=[]
    for st,s,e,name,api,ae in c.execute(f'''select k.streamId,k.start,k.end,sn.value,ra.value,r.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sn on sn.id=k.shortName
        left join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=k.correlationId left join StringIds ra on ra.id=r.nameId where k.start>={T0} and k.start<{T1}'''):
        ops.append((st,s,e,'K',name,api or '',ae))
    for st,s,e,b in c.execute(f'select streamId,start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind not in (11,12) and start>={T0} and start<{T1}'):
        ops.append((st,s,e,'C','memcpy','',None))
    for st,s,e in c.execute(f'select streamId,start,end from CUPTI_ACTIVITY_KIND_MEMSET where start>={T0} and start<{T1}'):
        ops.append((st,s,e,'S','memset','',None))
    bys=defaultdict(list)
    for o in ops: bys[o[0]].append(o)
    g=defaultdict(list)
    for st,l in bys.items():
        l.sort(key=lambda o:o[1])
        for i in range(1,len(l)):
            o=l[i];p=l[i-1]
            if o[3]=='K' and p[3]=='K' and not o[5].startswith('cuLaunchKernelEx') and o[6] and o[6]<p[2] and st==13:
                g[o[4]].append(o[1]-p[2])
    print('==',db,'plain device-paced kernel->kernel links on main stream, by secondary kernel (param size by value in parentheses where known)')
    sizes={'cp_vertex_fetch':'904B','cp_clip_rast_fused':'~600B','cp_rasterize_stage2':'~500B','cp_rasterize_stage3':'~500B','cp_fs_compact':'416B','cp_fs_writeback':'~100B','main':'8B ptr','cp_abuf_seg_count':'?','cp_rasterize_stage3_abuf':'~500B'}
    for k,v in sorted(g.items(), key=lambda kv:-len(kv[1])):
        if len(v)/fr<0.5: continue
        print(f'  {k:28s} ({sizes.get(k,"?"):8s}) n/f {len(v)/fr:6.2f} p10 {pct(v,.1)/1e3:5.2f} p50 {pct(v,.5)/1e3:5.2f} p90 {pct(v,.9)/1e3:6.2f} mean {statistics.mean(v)/1e3:6.2f} us')
