import pickle, bisect, statistics as st, collections
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
M=sorted(D['M']); ms=[m[0] for m in M]
med=st.median(f['frame'] for f in F); f=min(F,key=lambda x:abs(x['frame']-med))
tA,tB,tA2=f['tA'],f['tB'],f['tA2']
rb=[m for m in M[bisect.bisect_left(ms,tB):bisect.bisect_left(ms,tA2)] if m[3]==8 and m[4]==3686400][0]
print("sample frame a=%d: tB=%.1f rb=[%.1f,%.1f] tA2=%.1f (us from tA)"%(f['a'],(tB-tA)/1e3,(rb[0]-tA)/1e3,(rb[1]-tA)/1e3,(tA2-tA)/1e3))
print("UVM copies after rb end, offset from rb.end (us), dur, kind, bytes, stream:")
prev=None
for m in M[bisect.bisect_left(ms,rb[1]):bisect.bisect_left(ms,tA2)]:
    gap=(m[0]-prev)/1e3 if prev else 0
    print("%9.1f %6.1f gap%7.1f  kind%d %8d s%d"%((m[0]-rb[1])/1e3,(m[1]-m[0])/1e3,gap,m[3],m[4],m[2])); prev=m[1]
# across frames: first M12 after rb.end offset, last M12 end offset before tA2, count, bytes, and largest gap between consecutive UVM copies
rows=[]
for f in F:
    tA,tB,tA2=f['tA'],f['tB'],f['tA2']
    seg=M[bisect.bisect_left(ms,tB):bisect.bisect_left(ms,tA2)]
    rb=[m for m in seg if m[3]==8 and m[4]==3686400][0]
    u=[m for m in seg if m[0]>=rb[1] and m[3] in (11,12)]
    if not u: rows.append(None); continue
    gaps=[(u[i+1][0]-u[i][1])/1e3 for i in range(len(u)-1)]
    rows.append(dict(first=(u[0][0]-rb[1])/1e3, last_to_A2=(tA2-u[-1][1])/1e3, n=len(u), mb=sum(m[4] for m in u)/1e6, span=(u[-1][1]-u[0][0])/1e3, maxgap=max(gaps) if gaps else 0, n12=sum(1 for m in u if m[3]==12), n11=sum(1 for m in u if m[3]==11), record=f['record']))
rows=[r for r in rows if r]
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
for k in ('first','last_to_A2','n','mb','span','maxgap','n12','n11','record'):
    v=[r[k] for r in rows]; print("%-12s median %9.2f p10 %9.2f p90 %9.2f"%(k,st.median(v),q(v,.1),q(v,.9)))
