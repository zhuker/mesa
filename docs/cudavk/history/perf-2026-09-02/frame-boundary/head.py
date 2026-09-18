import pickle, bisect, statistics as st, collections
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
names=D['names']; K=D['K']; R=D['R']; MAIN,WORK=D['tids']
med=st.median(f['frame'] for f in F); f=min(F,key=lambda x:abs(x['frame']-med))
tA,tB,tA2=f['tA'],f['tB'],f['tA2']
ks=sorted(K); kst=[k[0] for k in ks]
fk=ks[bisect.bisect_left(kst,tA)]
print("first kernel: start %.1f us dur %.1f name=%s grid=%d stream=%d corr=%d"%((fk[0]-tA)/1e3,(fk[1]-fk[0])/1e3,names.get(fk[3],fk[3]),fk[4],fk[2],fk[5]))
# main thread APIs from 100us to first kernel + 20us
for s,e,tid,nid,cid in R:
    if tid==MAIN and tA+100e3<=s<=fk[0]+30e3:
        print("%8.1f %6.1f %s%s"%((s-tA)/1e3,(e-s)/1e3,names[nid]," <== first kernel launch" if cid==fk[5] else ""))
# across frames: the API that launches the first kernel: its start offset and the kernel start offset
lead=[];launch_off=[];queue_delay=[]
cid2api={r[4]:r for r in R}
for f in F:
    tA=f['tA']; k=ks[bisect.bisect_left(kst,tA)]
    r=cid2api.get(k[5])
    if r: launch_off.append((r[0]-tA)/1e3); queue_delay.append((k[0]-r[1])/1e3)
print("first kernel launch API offset median %.1f us; kernel start after API end median %.1f us (p90 %.1f)"%(st.median(launch_off),st.median(queue_delay),sorted(queue_delay)[int(.9*len(queue_delay))]))
# host funcs: count per frame and offsets
hf=collections.Counter()
for f in F:
    tA,tA2=f['tA'],f['tA2']
    offs=[(r[0]-tA)/1e3 for r in R if r[2]==MAIN and names[r[3]]=='cuLaunchHostFunc' and tA<=r[0]<tA2]
    hf[len(offs)]+=1
print("cuLaunchHostFunc per frame:",hf)
