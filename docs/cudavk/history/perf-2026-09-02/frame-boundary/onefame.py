import pickle, bisect, statistics as st, collections, sys
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
ts=D['ts']; names=D['names']; K=D['K']; M=D['M']; S=D['S']; R=D['R']; Y=D['Y']
MAIN,WORK=D['tids']
# pick the frame whose length is closest to the median
med=st.median(f['frame'] for f in F)
f=min(F,key=lambda x:abs(x['frame']-med))
print("chosen frame a=%d len %.3f ms (median %.3f)"%(f['a'],f['frame'],med))
tA,tB,tA2=f['tA'],f['tB'],f['tA2']
ev=[]
for s,e,sid,nm,g,cid in K:
    if tA-50e3<=s<tA2+400e3: ev.append((s,e,'DEV','K:%s g=%d s%d'%(nm,g,sid)))
for s,e,sid,k,b,cid in M:
    if tA-50e3<=s<tA2+400e3: ev.append((s,e,'DEV','M%d %d s%d'%(k,b,sid)))
for s,e,sid,b,cid in S:
    if tA-50e3<=s<tA2+400e3: ev.append((s,e,'DEV','memset %d s%d'%(b,sid)))
for s,e,tid,nid,cid in R:
    if tA-50e3<=s<tA2+400e3: ev.append((s,e,'MAIN' if tid==MAIN else 'WORK',names[nid]))
ev.sort()
ev.append((tA,tA,'TS','vkQueueSubmit A (shim ts)')); ev.append((tB,tB,'TS','vkQueueSubmit B (shim ts)')); ev.append((tA2,tA2,'TS','vkQueueSubmit A next (shim ts)'))
ev.sort()
# print the head (first 60 events), the region around tB, and the tail after rb
def pr(rows):
    for s,e,who,what in rows: print("%10.1f %9.1f %-5s %s"%((s-tA)/1e3,(e-s)/1e3,who,what))
print("--- head ---"); pr([x for x in ev if x[0]<tA+260e3][:80])
print("--- around B ---"); pr([x for x in ev if tB-400e3<=x[0]<tB+700e3 and not x[3].startswith('M12')][:60])
print("--- after readback: UVM tail, wake, record ---")
tail=[x for x in ev if x[0]>=tB+500e3]
# collapse M12 runs
out=[]; run=None
for x in tail:
    if x[3].startswith('M12'):
        if run is None: run=[x[0],x[1],0,0]
        run[1]=max(run[1],x[1]); run[2]+=1; run[3]+=int(x[3].split()[1])
    else:
        if run: out.append((run[0],run[1],'DEV','M12 run: %d copies %d bytes'%(run[2],run[3]))); run=None
        out.append(x)
if run: out.append((run[0],run[1],'DEV','M12 run: %d copies %d bytes'%(run[2],run[3])))
pr(out[:60])
# sync events on main thread in the frame with types
cid2=({r[4]:(r[2],names[r[3]]) for r in R})
print("--- blocking syncs (main thread) in frame: offset us, dur us, type ---")
tot=collections.Counter(); n=collections.Counter()
for s,e,sid,typ,cid,eid in sorted(Y):
    if tA<=s<tA2 and cid2.get(cid,(None,))[0]==MAIN and typ in (1,3,4):
        print("%10.1f %9.1f type%d %s stream%d"%((s-tA)/1e3,(e-s)/1e3,typ,cid2[cid][1],sid)); tot[typ]+=e-s; n[typ]+=1
print("totals us:",{k:v/1e3 for k,v in tot.items()}, dict(n))
