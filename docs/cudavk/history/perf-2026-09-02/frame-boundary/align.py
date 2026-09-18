import sqlite3, statistics as st
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
T0=c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts=[int(l.split()[0]) for l in open('/tmp/postdead/submit-ts.txt')]
rel=[t-T0 for t in ts]
print("T0",T0,"n",len(ts))
for i in (2780,2781,2782,2783,2784,2785,2786):
    print(i, rel[i]/1e9)
# memcpy census by kind and bytes for the window
print(c.execute("select copyKind, count(*), sum(bytes), sum(end-start), min(bytes), max(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind").fetchall())
print(c.execute("select * from ENUM_CUDA_MEMCPY_OPER").fetchall())
# 3686400-byte copies
rows=c.execute("select start,end,copyKind,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where bytes=3686400 order by start").fetchall()
print("3686400-byte copies:",len(rows), rows[:3])
big=[r for r in rows if 9.0e9<r[0]<12.4e9]
print("in window:",len(big))
# For each big DtoH, find the nearest preceding external submit ts
import bisect
res=[]
for s,e,k,sid,b in big:
    i=bisect.bisect_right(rel,s)-1
    res.append((i, (s-rel[i])/1e3, (e-s)/1e3, i%2))
print("first 10 (submit idx, us from submit ts to DtoH start, DtoH dur us, parity):")
for r in res[:10]: print(r)
d=[r[1] for r in res]
print("lead us: median",st.median(d),"p10",sorted(d)[len(d)//10],"p90",sorted(d)[9*len(d)//10],"min",min(d),"max",max(d))
print("parity census:",{p:sum(1 for r in res if r[3]==p) for p in (0,1)})
print("dur us median",st.median([r[2] for r in res]))
# drift check: lead vs index
import numpy as np
idx=np.array([r[0] for r in res]); lead=np.array(d)
A=np.vstack([idx,np.ones_like(idx)]).T
sl,ic=np.linalg.lstsq(A,lead,rcond=None)[0]
print("lead slope us/submit",sl,"intercept",ic)
