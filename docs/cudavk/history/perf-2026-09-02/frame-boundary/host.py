import pickle, bisect, statistics as st, collections
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
names=D['names']; R=D['R']; Y=sorted(D['Y']); MAIN,WORK=D['tids']
Rm=[r for r in R if r[2]==MAIN]; rs=[r[0] for r in Rm]
cid2=({r[4]:(r[2],names[r[3]]) for r in R}); ys=[y[0] for y in Y]
rows=[]
for f in F:
    tA,tB,tA2=f['tA'],f['tB'],f['tA2']
    a=Rm[bisect.bisect_left(rs,tA):bisect.bisect_left(rs,tA2)]
    busy=sum(r[1]-r[0] for r in a)
    byname=collections.Counter(); cnt=collections.Counter()
    for r in a: byname[names[r[3]]]+=r[1]-r[0]; cnt[names[r[3]]]+=1
    sy=[y for y in Y[bisect.bisect_left(ys,tA):bisect.bisect_left(ys,tA2)] if cid2.get(y[4],(None,))[0]==MAIN and y[3] in (3,4)]
    ctx=[(y[0]-tA)/1e3 for y in sy if y[3]==4]; ctxd=[(y[1]-y[0])/1e3 for y in sy if y[3]==4]
    strm=[(y[1]-y[0])/1e3 for y in sy if y[3]==3]
    dtoh=byname['cuMemcpyDtoH_v2']
    blocked=sum(y[1]-y[0] for y in sy)+dtoh
    rows.append(dict(frame=f['frame'], api_busy=busy/1e6, blocked=blocked/1e6, issue=(busy-blocked)/1e6, nctx=len(ctx), ctx_off=ctx, ctx_dur=ctxd, ctx_sum=sum(ctxd)/1e3, nstrm=len(strm), strm_sum=sum(strm)/1e3, dtoh=dtoh/1e6, n_api=len(a), byname=byname, cnt=cnt, host_other=(tA2-tA-busy)/1e6))
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
for k in ('frame','api_busy','blocked','issue','host_other','nctx','ctx_sum','nstrm','strm_sum','dtoh','n_api'):
    v=[r[k] for r in rows]; print("%-12s median %9.3f p10 %9.3f p90 %9.3f mean %9.3f"%(k,st.median(v),q(v,.1),q(v,.9),st.mean(v)))
print("ctx sync offsets (us) first/second medians:", st.median([r['ctx_off'][0] for r in rows if r['nctx']>=1]), st.median([r['ctx_off'][1] for r in rows if r['nctx']>=2]))
print("ctx sync durs first/second medians:", st.median([r['ctx_dur'][0] for r in rows if r['nctx']>=1]), st.median([r['ctx_dur'][1] for r in rows if r['nctx']>=2]))
print("nctx census", collections.Counter(r['nctx'] for r in rows))
tot=sum((r['byname'] for r in rows),collections.Counter()); c=sum((r['cnt'] for r in rows),collections.Counter())
n=len(rows)
print("API time per frame (mean us) and count/frame:")
for k,v in tot.most_common(20): print("  %-24s %8.1f us  %7.1f /frame"%(k,v/n/1e3,c[k]/n))
