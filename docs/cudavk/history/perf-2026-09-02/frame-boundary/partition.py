import pickle, bisect, statistics as st, collections
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
K=D['K']; M=D['M']; S=D['S']
ops=[]
for s,e,sid,nm,g,cid in K: ops.append((s,e,sid,'K',nm,0))
for s,e,sid,k,b,cid in M: ops.append((s,e,sid,'M%d'%k,'memcpy%d'%k,b))
for s,e,sid,b,cid in S: ops.append((s,e,sid,'S','memset',b))
ops.sort(); starts=[o[0] for o in ops]
def merge(iv):
    iv=sorted(iv); out=[]
    for s,e in iv:
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
rows=[]
for f in F:
    tA,tB,tA2=f['tA'],f['tB'],f['tA2']
    fo=[o for o in ops[bisect.bisect_left(starts,tA):bisect.bisect_left(starts,tA2)] if o[3] not in ('M11','M12')]
    rb=[o for o in fo if o[3]=='M8' and o[5]==3686400 and o[0]>=tB][0]
    rend=[o for o in fo if o[0]<tB]
    u=merge([(o[0],o[1]) for o in rend])
    first=u[0][0]; last=u[-1][1]
    head=first-tA
    internal=sum(u[i+1][0]-u[i][1] for i in range(len(u)-1))
    small=sum(g for g in (u[i+1][0]-u[i][1] for i in range(len(u)-1)) if g<=20e3)
    mid=sum(g for g in (u[i+1][0]-u[i][1] for i in range(len(u)-1)) if 20e3<g<=200e3)
    big=sum(g for g in (u[i+1][0]-u[i][1] for i in range(len(u)-1)) if g>200e3)
    inter=rb[0]-last
    rbd=rb[1]-rb[0]
    bnd=tA2-rb[1]
    busy=sum(e-s for s,e in u)+rbd
    frame=tA2-tA
    kern_first=min((o[0] for o in rend if o[3]=="K"),default=tA)-tA
    rows.append(dict(frame=frame/1e6, busy=busy/1e6, idle=(frame-busy)/1e6, head=head/1e3, kern_first=kern_first/1e3, internal=internal/1e6, small=small/1e6, mid=mid/1e6, big=big/1e6, inter=inter/1e3, rb=rbd/1e3, bnd=bnd/1e3, check=(frame-busy-head-internal-inter-bnd)/1e3, nseg=len(u), render_busy=sum(e-s for s,e in u)/1e6, render_span=(last-first)/1e6))
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
print("%-12s %10s %10s %10s %10s  unit"%('part','median','p10','p90','mean'))
units={'frame':'ms','busy':'ms','idle':'ms','head':'us','kern_first':'us','internal':'ms','small':'ms','mid':'ms','big':'ms','inter':'us','rb':'us','bnd':'us','check':'us','nseg':'','render_busy':'ms','render_span':'ms'}
for k in ('frame','busy','idle','head','kern_first','render_span','render_busy','internal','small','mid','big','nseg','inter','rb','bnd','check'):
    v=[r[k] for r in rows]; print("%-12s %10.3f %10.3f %10.3f %10.3f  %s"%(k,st.median(v),q(v,.1),q(v,.9),st.mean(v),units[k]))
# share of idle at boundary (bnd+inter+head) vs internal, mean-based
mi=st.mean(r['idle'] for r in rows); mb=st.mean((r['bnd']+r['inter']+r['head'])/1e3 for r in rows); mint=st.mean(r['internal'] for r in rows)
print("mean idle %.3f = boundary(bnd+inter+head) %.3f + internal %.3f"%(mi,mb,mint))
# tertiles by frame length
rs=sorted(rows,key=lambda r:r['frame']); n=len(rs)
for name,sl in (('light third',rs[:n//3]),('mid third',rs[n//3:2*n//3]),('heavy third',rs[2*n//3:])):
    print(name, "frame med %.3f busy %.3f idle %.3f | head %.0f internal %.3f inter %.0f rb %.0f bnd %.0f"%tuple(st.median([r[k] for r in sl]) for k in ('frame','busy','idle','head','internal','inter','rb','bnd')))
pickle.dump(rows,open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/partition.pkl','wb'))
