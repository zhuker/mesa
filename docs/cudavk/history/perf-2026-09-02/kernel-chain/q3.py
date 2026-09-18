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
# long fused launches: recurrence and neighbours
print("=== cp_clip_rast_fused > 200us: which frame, stream, grid ===")
rows=c.execute(f"select start,end,streamId,gridX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='cp_clip_rast_fused' and start>={W0} and start<{W1} and end-start>200000 order by start").fetchall()
fr=collections.Counter()
for s,e,st,gx in rows:
    k=bisect.bisect_right([a for a,b in fb], s)-1; fr[k]+=1
print("count",len(rows),"frames with one:",len(fr),"per-frame hist",collections.Counter(fr.values()))
print("by stream",collections.Counter(r[2] for r in rows),"by gridX",sorted(collections.Counter(r[3] for r in rows).items())[:30])
# neighbours of the t=11.018 stream 16 launch
s,e,st,gx=[r for r in rows if abs(r[0]/1e9-11.018)<0.002][0]
print("\nneighbours on stream",st,"around",s/1e9)
for rr in c.execute("select s.value,k.start,k.end,k.gridX,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.streamId=? and k.start between ? and ? order by k.start",(st,s-600000,e+600000)):
    print("   %-26s t=%+8.1f dur=%7.1f grid=%d block=%d"%(rr[0],(rr[1]-s)/1e3,(rr[2]-rr[1])/1e3,rr[3],rr[4]))
# vertex_fetch and VS long durations: sum of >100us per frame
print("\n=== long (>100us) launches per kernel: n/f, sum_ms/f ===")
for r in c.execute(f"select s.value,count(*),sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where start>={W0} and start<{W1} and end-start>100000 group by s.value order by 3 desc"):
    print("  %-26s n/f=%.2f sum_ms/f=%.3f"%(r[0],r[1]/NF,r[2]/NF))
# overlap of long vertex_fetch with UVM copies
print("\n=== vertex_fetch >100us: fraction overlapping a UVM (stream 0) copy ===")
vf=c.execute(f"select start,end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='cp_vertex_fetch' and start>={W0} and start<{W1} and end-start>100000").fetchall()
uvm=c.execute(f"select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where streamId=0 and start>={W0-2000000} and start<{W1}").fetchall()
us=[a for a,b in uvm]
ov=0
for s,e in vf:
    i=bisect.bisect_left(us,s-2000000); j=bisect.bisect_left(us,e)
    if any(b>s and a<e for a,b in uvm[i:j]): ov+=1
print("  long vertex_fetch:",len(vf),"overlapping UVM:",ov)
vs=c.execute(f"select k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k where k.shortName in (select id from StringIds where value='main') and k.start>={W0} and k.start<{W1} and k.end-k.start>100000 and k.blockX=256 and k.gridX<3000").fetchall()
ov=0
for s,e in vs:
    i=bisect.bisect_left(us,s-2000000); j=bisect.bisect_left(us,e)
    if any(b>s and a<e for a,b in uvm[i:j]): ov+=1
print("  long main(grid<3000):",len(vs),"overlapping UVM:",ov)
# meta: nsys version
print("\n=== META_DATA_CAPTURE keys ===")
for r in c.execute("select * from META_DATA_CAPTURE"):
    if 'ENVIRONMENT' not in str(r[0]): print(" ",r)
