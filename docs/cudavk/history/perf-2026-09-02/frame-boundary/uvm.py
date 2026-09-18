import pickle, bisect, statistics as st, collections
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
ts=D['ts']; names=D['names']; K=D['K']; M=D['M']; S=D['S']; R=D['R']; Y=D['Y']
MAIN=D['tids'][0]
ops=[]
for s,e,sid,nm,g,cid in K: ops.append((s,e,sid,'K',nm,0))
for s,e,sid,k,b,cid in M: ops.append((s,e,sid,'M%d'%k,'memcpy%d'%k,b))
for s,e,sid,b,cid in S: ops.append((s,e,sid,'S','memset',b))
ops.sort(); starts=[o[0] for o in ops]
def union_len(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return tot
rows=[]
for f in F:
    tA,tB,tA2=f['tA'],f['tB'],f['tA2']
    lo=bisect.bisect_left(starts,tA); hi=bisect.bisect_left(starts,tA2)
    fo=ops[lo:hi]
    def cnt(kind,phase):
        sel=[o for o in fo if o[3]==kind and ((o[0]<tB) if phase=='A' else (o[0]>=tB))]
        return len(sel), sum(o[5] for o in sel), sum(o[1]-o[0] for o in sel)
    r={}
    for kind in ('M11','M12','M1','M8','M7'):
        for ph in 'AB':
            n,b,d=cnt(kind,ph); r[kind+ph]=(n,b/1e6,d/1e3)
    real=[(max(o[0],tA),min(o[1],tA2)) for o in fo if o[3] not in ('M11','M12')]
    uvm=[(max(o[0],tA),min(o[1],tA2)) for o in fo if o[3] in ('M11','M12')]
    r['union_all']=union_len(real+uvm)/1e6; r['union_real']=union_len(real)/1e6; r['union_uvm']=union_len(uvm)/1e6
    r['frame']=(tA2-tA)/1e6
    # readback copy
    rb=[o for o in fo if o[3]=='M8' and o[5]==3686400 and o[0]>=tB][0]
    # next frame's first non-UVM op
    nlo=bisect.bisect_left(starts,tA2)
    j=nlo
    while ops[j][3] in ('M11','M12'): j+=1
    r['boundary_idle_real']=(ops[j][0]-rb[1])/1e3
    r['next_real_first']=ops[j][4]
    # last real (non-UVM) op before tB
    lastA=max((o[1] for o in fo if o[0]<tB and o[3] not in ('M11','M12')),default=None)
    lastAop=max((o for o in fo if o[0]<tB and o[3] not in ('M11','M12')),key=lambda o:o[1])
    r['intersubmit_idle_real']=(rb[0]-lastA)/1e3; r['lastA_real']=(lastAop[4],lastAop[3])
    r['lastA_real_to_tB']=(tB-lastA)/1e3
    # real ops between rb end and tA2 (non-UVM)
    r['real_after_rb']=sum(1 for o in fo if o[0]>rb[1] and o[3] not in ('M11','M12'))
    r['rb_end_to_tA2']=(tA2-rb[1])/1e3
    rows.append(r)
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
def show(k,f=lambda x:x):
    v=[f(r[k]) for r in rows]; print("%-28s median %10.3f p10 %10.3f p90 %10.3f mean %10.3f"%(k,st.median(v),q(v,.1),q(v,.9),st.mean(v)))
for k in ('frame','union_all','union_real','union_uvm','boundary_idle_real','intersubmit_idle_real','lastA_real_to_tB','rb_end_to_tA2'): show(k)
for kind in ('M11','M12','M1','M8','M7'):
    for ph in 'AB':
        k=kind+ph
        print(k, "count median %.1f  MB/frame mean %.3f  summed us/frame mean %.1f"%(st.median([r[k][0] for r in rows]), st.mean([r[k][1] for r in rows]), st.mean([r[k][2] for r in rows])))
print("real_after_rb:",collections.Counter(r['real_after_rb'] for r in rows))
print("next_real_first:",collections.Counter(r['next_real_first'] for r in rows).most_common(4))
print("lastA_real:",collections.Counter(r['lastA_real'] for r in rows).most_common(6))
pickle.dump(rows,open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/uvm_rows.pkl','wb'))
