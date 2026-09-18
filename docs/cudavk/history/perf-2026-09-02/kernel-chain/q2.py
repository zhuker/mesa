import sqlite3, statistics, collections, json, bisect
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
S0=1503511535334936
ts=[l.split() for l in open('/tmp/postdead/submit-ts.txt') if len(l.split())==2]
rel=[int(a)-S0 for a,b in ts]
NF=412; fb=[(rel[2782+2*k], rel[2784+2*k]) for k in range(NF)]
W0=fb[0][0]; W1=fb[-1][1]
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
def med(x):
    x=[v for v in x if v is not None]; return statistics.median(x) if x else None
def q(x,p):
    x=sorted(v for v in x if v is not None); return x[int(p*(len(x)-1))] if x else None
# A. cp_clip_rast_fused duration by grid
print("=== cp_clip_rast_fused duration by gridX bucket ===")
rows=c.execute(f"select start,end,streamId,gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='cp_clip_rast_fused' and start>={W0} and start<{W1}").fetchall()
bk=collections.defaultdict(list)
for s,e,st,gx in rows:
    b = 1 if gx==1 else 2 if gx<=4 else 16 if gx<=16 else 64 if gx<=64 else 256 if gx<=256 else 1024 if gx<=1024 else 99999
    bk[b].append((e-s,st,gx))
for b,v in sorted(bk.items()):
    d=[x[0] for x in v]
    print("gridX<=%d: n/f=%.2f med_us=%.2f p90=%.2f max=%.1f sum_ms/f=%.3f main=%d side=%d med_gx=%d"%(b,len(v)/NF,med(d)/1e3,q(d,.9)/1e3,max(d)/1e3,sum(d)/1e6/NF,sum(1 for x in v if x[1]==13),sum(1 for x in v if x[1]!=13),med([x[2] for x in v])))
print("top 12 longest cp_clip_rast_fused:")
for s,e,st,gx in sorted(rows,key=lambda r:-(r[1]-r[0]))[:12]: print("  dur_us=%.1f stream=%d gridX=%d t=%.3f"%((e-s)/1e3,st,gx,s/1e9))
# top-duration kernels overall
print("\n=== top 30 longest kernels in window ===")
for r in c.execute(f"select s.value,k.start,k.end,k.streamId,k.gridX,k.blockX,k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>={W0} and k.start<{W1} order by (k.end-k.start) desc limit 30"):
    print("  %-28s dur_us=%8.1f stream=%d grid=%d block=%d regs=%d t=%.4f"%(r[0],(r[2]-r[1])/1e3,r[3],r[4],r[5],r[6],r[1]/1e9))
# the 40-thread main kernel
print("\n=== the (5,8) main kernel ===")
for r in c.execute(f"select k.start,k.end,k.streamId,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread,k.correlationId,k.demangledName,k.localMemoryPerThread from CUPTI_ACTIVITY_KIND_KERNEL k where k.blockX=8 and k.start>={W0} and k.start<{W1} limit 3"):
    print(r, names[r[11]])
    cid=r[10]
    for rr in c.execute("select s.value,r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.correlationId=?",(cid,)): print("   api",rr)
    # neighbours on all streams within +-300us
    for rr in c.execute("select 'K',s.value,k.start,k.end,k.streamId,k.gridX,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start between ? and ? order by k.start",(r[0]-200000,r[1]+300000)): print("   ",rr[0],rr[1],"%.1f"%((rr[2]-r[0])/1e3),"dur=%.1f"%((rr[3]-rr[2])/1e3),"st",rr[4],rr[5],rr[6])
    for rr in c.execute("select 'M',copyKind,bytes,start,end,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where start between ? and ? and streamId!=0 order by start",(r[0]-200000,r[1]+300000)): print("   ",rr[0],rr[1],rr[2],"%.1f"%((rr[3]-r[0])/1e3),"dur=%.1f"%((rr[4]-rr[3])/1e3),"st",rr[5])
    for rr in c.execute("select 'S',bytes,start,end,streamId from CUPTI_ACTIVITY_KIND_MEMSET where start between ? and ? order by start",(r[0]-200000,r[1]+300000)): print("   ",rr[0],rr[1],"%.1f"%((rr[2]-r[0])/1e3),"dur=%.1f"%((rr[3]-rr[2])/1e3),"st",rr[4])
    break
# memset sizes on main
print("\n=== memsets by size (per frame) ===")
for r in c.execute(f"select bytes,count(*),avg(end-start)/1e3,sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_MEMSET where start>={W0} and start<{W1} group by bytes order by 4 desc limit 15"):
    print("  bytes=%d n/f=%.2f avg_us=%.2f sum_ms/f=%.3f"%(r[0],r[1]/NF,r[2],r[3]/NF))
print("\n=== memcpy by kind,size class (per frame), non-UVM ===")
for r in c.execute(f"select copyKind, case when bytes<4096 then 'lt4k' when bytes<65536 then 'lt64k' when bytes<1048576 then 'lt1M' else 'ge1M' end, count(*),avg(end-start)/1e3,sum(end-start)/1e6, sum(bytes)/1e6 from CUPTI_ACTIVITY_KIND_MEMCPY where start>={W0} and start<{W1} and streamId!=0 group by 1,2 order by 5 desc"):
    print("  kind=%d %s n/f=%.2f avg_us=%.2f sum_ms/f=%.3f MB/f=%.2f"%(r[0],r[1],r[2]/NF,r[3],r[4]/NF,r[5]/NF))
# sync per frame
print("\n=== host syncs per frame ===")
for r in c.execute(f"select syncType, streamId, count(*), sum(end-start)/1e6, avg(end-start)/1e3 from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>={W0} and start<{W1} group by 1,2 order by 1,2"):
    print("  type=%d stream=%d n/f=%.2f blocked_ms/f=%.3f mean_us=%.1f"%(r[0],r[1],r[2]/NF,r[3]/NF,r[4]))
# runtime API census per frame
print("\n=== runtime API per frame ===")
for r in c.execute(f"select s.value,count(*),sum(r.end-r.start)/1e6 from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>={W0} and r.start<{W1} group by s.value order by 2 desc limit 20"):
    print("  %-26s n/f=%.2f host_ms/f=%.3f"%(r[0],r[1]/NF,r[2]/NF))
