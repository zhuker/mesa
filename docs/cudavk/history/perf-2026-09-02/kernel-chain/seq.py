import sqlite3, statistics, collections, json, bisect, sys
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
S0=1503511535334936
ts=[l.split() for l in open('/tmp/postdead/submit-ts.txt') if len(l.split())==2]
rel=[int(a)-S0 for a,b in ts]
NF=412; fb=[(rel[2782+2*k], rel[2784+2*k]) for k in range(NF)]
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
# pick a frame whose length is near the median
fl=[(b-a) for a,b in fb]; m=statistics.median(fl)
k=min(range(NF), key=lambda i: abs(fl[i]-m))
a,b=fb[k]
print("frame",k,"abs frame",1391+k,"len ms",(b-a)/1e6, "t0",a/1e9)
ops=[]
for r in c.execute("select 'K',s.value,k.start,k.end,k.streamId,k.gridX*k.gridY*k.gridZ,k.blockX*k.blockY*k.blockZ,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.start<? order by k.start",(a,b)): ops.append(list(r))
for r in c.execute("select 'M','memcpy'||copyKind||'_'||bytes,start,end,streamId,0,0,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and streamId!=0 order by start",(a,b)): ops.append(list(r))
for r in c.execute("select 'S','memset_'||bytes,start,end,streamId,0,0,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<? order by start",(a,b)): ops.append(list(r))
for r in c.execute("select 'Y','sync'||syncType,start,end,streamId,0,0,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=? and start<? and syncType in (1,3,4) order by start",(a,b)): ops.append(list(r))
# host issue time
cids=[o[7] for o in ops if o[7]]
rt={}
for r in c.execute("select correlationId,start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?",(a-20000000,b)):
    rt[r[0]]=(r[1],r[2],names[r[3]])
ops.sort(key=lambda o:o[2])
prev_end={}
lines=[]
for o in ops:
    kind,n,s,e,st,g,bl,cid=o
    gap=(s-prev_end[st])/1e3 if st in prev_end else float('nan')
    prev_end[st]=max(prev_end.get(st,0),e)
    issue=rt.get(cid)
    lag=(s-issue[1])/1e3 if issue else float('nan')
    lines.append("%8.1f st%-2d %-1s %-34s dur=%7.1f gap=%7.1f lag=%8.1f grid=%d blk=%d"%((s-a)/1e3,st,kind,n[:34],(e-s)/1e3,gap,lag,g,bl))
open('frame_seq.txt','w').write("\n".join(lines))
print(len(lines),"ops written")
