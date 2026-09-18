import pickle, bisect, statistics as st, collections, json, sys
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
ts=D['ts']; names=D['names']; K=D['K']; M=D['M']; S=D['S']; R=D['R']; Y=D['Y']
MAIN,WORK=D['tids'][0],D['tids'][1]
# unified device op list: (start,end,stream,kind,label,bytes)
ops=[]
for s,e,sid,nm,g,cid in K: ops.append((s,e,sid,'K',nm,0,cid))
for s,e,sid,k,b,cid in M: ops.append((s,e,sid,'M%d'%k,'memcpy%d'%k,b,cid))
for s,e,sid,b,cid in S: ops.append((s,e,sid,'S','memset',b,cid))
ops.sort()
op_starts=[o[0] for o in ops]
Rmain=[r for r in R if r[2]==MAIN]; Rwork=[r for r in R if r[2]==WORK]
rm_starts=[r[0] for r in Rmain]
rw=[(r[0],r[1],names[r[3]]) for r in Rwork]
cid2name={r[4]:(r[2],names[r[3]]) for r in R}
Ysorted=sorted(Y); y_starts=[y[0] for y in Ysorted]
def union_len(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s>ce: tot+=ce-cs; cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return tot
def gaps(iv):
    iv=sorted(iv); out=[]; ce=None
    for s,e in iv:
        if ce is not None and s>ce: out.append((ce,s))
        ce=e if ce is None else max(ce,e)
    return out
# window: odd indices a with ts[a]>=9.27e9 and ts[a+2]<=12.34e9 ; verify parity with first-kernel lead
first_k_lead={0:[],1:[]}
for a in range(2700,3700):
    t=ts[a]; i=bisect.bisect_left(op_starts,t)
    while i<len(ops) and ops[i][3]!='K': i+=1
    if i<len(ops): first_k_lead[a%2].append((ops[i][0]-t)/1e3)
for p in (0,1): print("parity",p,"median us from submit ts to first kernel:",st.median(first_k_lead[p]))
frames=[]
for a in range(2783,3620,2):
    tA,tB,tA2=ts[a],ts[a+1],ts[a+2]
    if tA<9.27e9 or tA2>12.34e9: continue
    lo=bisect.bisect_left(op_starts,tA); hi=bisect.bisect_left(op_starts,tA2)
    fo=ops[lo:hi]
    if not fo: continue
    rb=[o for o in fo if o[3]=='M8' and o[5]==3686400 and o[0]>=tB]
    if len(rb)!=1: print("frame a=%d: %d readback copies"%(a,len(rb))); continue
    rb=rb[0]
    render=[o for o in fo if o[0]<tB]
    last_render_end=max(o[1] for o in render)
    last_render_op=max(render,key=lambda o:o[1])
    strag=[o for o in fo if o[0]>rb[1]]
    tail_end=max([rb[1]]+[o[1] for o in strag])
    nlo=bisect.bisect_left(op_starts,tA2)
    nxt=ops[nlo]
    # host markers
    i0=bisect.bisect_left(rm_starts,tA); i1=bisect.bisect_left(rm_starts,tB); i2=bisect.bisect_left(rm_starts,tA2)
    fa=Rmain[i0:i1]; fb=Rmain[i1:i2]
    firstA=fa[0] if fa else None
    tAret=max(r[1] for r in fa) if fa else None
    # B: apis before rb start
    fbpre=[r for r in fb if r[0]<rb[0]]
    fbnames=[names[r[3]] for r in fbpre]
    tBret=max((r[1] for r in fb if r[0]<rb[1]+2e6), default=None)  # apis within 2 ms after rb end belong to submit B tail
    # record-time apis after tBret
    rec=[r for r in fb if tBret is not None and r[0]>tBret]
    # worker sync containing rb end
    w=[x for x in rw if x[0]<=rb[1]<=x[1]+50e3 and x[2]=='cuEventSynchronize']
    twake=min((x[1] for x in w if x[1]>=rb[1]),default=None)
    # sync records in frame (main thread)
    ylo=bisect.bisect_left(y_starts,tA); yhi=bisect.bisect_left(y_starts,tA2)
    ys=[y for y in Ysorted[ylo:yhi] if cid2name.get(y[4],(None,))[0]==MAIN]
    ysum=sum(y[1]-y[0] for y in ys)
    ycnt=collections.Counter(y[3] for y in ys)
    dtoh=[r for r in fa if names[r[3]]=='cuMemcpyDtoH_v2']
    # nkernels, union
    iv=[(max(o[0],tA),min(o[1],tA2)) for o in fo]
    uni=union_len(iv)
    # internal gaps in render phase
    riv=[(o[0],o[1]) for o in render]
    g=gaps(riv)
    gsum=sum(e-s for s,e in g)
    g20=sum(e-s for s,e in g if e-s>20e3); g100=sum(e-s for s,e in g if e-s>100e3); g200=sum(e-s for s,e in g if e-s>200e3)
    n200=sum(1 for s,e in g if e-s>200e3)
    fr=dict(a=a,tA=tA,tB=tB,tA2=tA2,frame=(tA2-tA)/1e6,union=uni/1e6,idle=(tA2-tA-uni)/1e6,
            nk=sum(1 for o in fo if o[3]=='K'),
            first_dev_lead=(fo[0][0]-tA)/1e3, first_dev=fo[0][4], first_api=(names[firstA[3]] if firstA else None), first_api_lead=((firstA[0]-tA)/1e3 if firstA else None), first_api_dur=((firstA[1]-firstA[0])/1e3 if firstA else None),
            first_kernel_lead=min(((o[0]-tA)/1e3 for o in fo if o[3]=='K'),default=None),
            render_span=(last_render_end-tA)/1e6, last_render_op=last_render_op[4], last_render_kind=last_render_op[3],
            Aret_to_lastdev=((last_render_end-tAret)/1e3 if tAret else None), Aret_lead=((tAret-tA)/1e6 if tAret else None),
            app_between=((tB-tAret)/1e3 if tAret else None),
            intersubmit_gap=(rb[0]-last_render_end)/1e3, B_issue=(rb[0]-tB)/1e3, B_apis=len(fbpre), B_apinames=collections.Counter(fbnames),
            rb_dur=(rb[1]-rb[0])/1e3, rb_stream=rb[2], nstrag=len(strag), strag_names=collections.Counter(o[4] for o in strag), tail_end_after_rb=(tail_end-rb[1])/1e3,
            wake_lat=((twake-rb[1])/1e3 if twake else None), record=((tA2-twake)/1e3 if twake else None),
            rec_apis=len(rec), rec_api_names=collections.Counter(names[r[3]] for r in rec), rec_api_time=sum(r[1]-r[0] for r in rec)/1e3,
            boundary_gap=(nxt[0]-tail_end)/1e3, next_first=nxt[4], next_first_kind=nxt[3],
            sync_n=len(ys), sync_ms=ysum/1e6, sync_types=dict(ycnt), dtoh_n=len(dtoh), dtoh_ms=sum(r[1]-r[0] for r in dtoh)/1e6,
            rgaps_ms=gsum/1e6, rgaps20=g20/1e6, rgaps100=g100/1e6, rgaps200=g200/1e6, n200=n200, ngaps=len(g))
    frames.append(fr)
print("frames",len(frames), "a range",frames[0]['a'],frames[-1]['a'])
pickle.dump(frames,open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','wb'))
def q(v,p):
    v=sorted(x for x in v if x is not None); 
    return v[min(len(v)-1,int(p*len(v)))] if v else None
keys=['frame','union','idle','nk','first_api_lead','first_api_dur','first_dev_lead','first_kernel_lead','render_span','Aret_lead','Aret_to_lastdev','app_between','intersubmit_gap','B_issue','rb_dur','tail_end_after_rb','wake_lat','record','rec_api_time','boundary_gap','sync_n','sync_ms','dtoh_n','dtoh_ms','rgaps_ms','rgaps20','rgaps100','rgaps200','n200','ngaps','nstrag']
print("%-20s %10s %10s %10s %10s"%('metric','median','p10','p90','mean'))
for k in keys:
    v=[f[k] for f in frames if f[k] is not None]
    print("%-20s %10.3f %10.3f %10.3f %10.3f"%(k,st.median(v),q(v,.1),q(v,.9),st.mean(v)))
print("first_api:",collections.Counter(f['first_api'] for f in frames).most_common(5))
print("first_dev:",collections.Counter(f['first_dev'] for f in frames).most_common(5))
print("last_render_op:",collections.Counter((f['last_render_op'],f['last_render_kind']) for f in frames).most_common(5))
print("next_first:",collections.Counter((f['next_first'],f['next_first_kind']) for f in frames).most_common(5))
print("strag:",collections.Counter(f['nstrag'] for f in frames), sum((f['strag_names'] for f in frames),collections.Counter()).most_common(6))
print("B apis:",sum((f['B_apinames'] for f in frames),collections.Counter()).most_common(10))
print("rec apis:",sum((f['rec_api_names'] for f in frames),collections.Counter()).most_common(10))
print("sync types:",sum((collections.Counter(f['sync_types']) for f in frames),collections.Counter()))
print("rb stream:",collections.Counter(f['rb_stream'] for f in frames))
