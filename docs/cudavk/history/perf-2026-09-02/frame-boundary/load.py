import sqlite3, pickle, bisect
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
T0=c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts=[int(l.split()[0])-T0 for l in open('/tmp/postdead/submit-ts.txt')]
LO,HI=9.0e9,12.5e9
names={i:v for i,v in c.execute("select id,value from StringIds")}
K=c.execute("select start,end,streamId,shortName,gridX*gridY*gridZ,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start",(LO,HI)).fetchall()
M=c.execute("select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? order by start",(LO,HI)).fetchall()
S=c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<? order by start",(LO,HI)).fetchall()
R=c.execute("select start,end,globalTid,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<? order by start",(LO-0.5e9,HI)).fetchall()
Y=c.execute("select start,end,streamId,syncType,correlationId,eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=? and start<? order by start",(LO-0.5e9,HI)).fetchall()
E=c.execute("select timestamp,streamId,eventId,correlationId from CUPTI_ACTIVITY_KIND_CUDA_EVENT where timestamp>=? and timestamp<? order by timestamp",(LO-0.5e9,HI)).fetchall()
tids=sorted(set(r[2] for r in R))
print("tids",tids, [sum(1 for r in R if r[2]==t) for t in tids])
pickle.dump(dict(T0=T0,ts=ts,names=names,K=K,M=M,S=S,R=R,Y=Y,E=E,tids=tids),open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','wb'))
print(len(K),len(M),len(S),len(R),len(Y),len(E))
