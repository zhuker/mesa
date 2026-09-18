import pickle, bisect, statistics as st, collections
D=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/win.pkl','rb'))
F=pickle.load(open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/frames.pkl','rb'))
names=D['names']; K=D['K']; M=D['M']; R=D['R']; MAIN,WORK=D['tids']
Ms=sorted(M); ms=[m[0] for m in Ms]
Rm=[r for r in R if r[2]==MAIN]; rs=[r[0] for r in Rm]
cid2m={m[5]:m for m in M}
# 1. API names in [rb.end, tA2) per frame
c=collections.Counter(); tot=collections.Counter()
descbytes=[]
inter_big=[]
for f in F:
    tA,tB,tA2=f['tA'],f['tB'],f['tA2']
    seg=Ms[bisect.bisect_left(ms,tB):bisect.bisect_left(ms,tA2)]
    rb=[m for m in seg if m[3]==8 and m[4]==3686400][0]
    for r in Rm[bisect.bisect_left(rs,rb[1]):bisect.bisect_left(rs,tA2)]:
        c[names[r[3]]]+=1; tot[names[r[3]]]+=r[1]-r[0]
    # desc upload bytes
    for r in Rm[bisect.bisect_left(rs,tA):bisect.bisect_left(rs,tB)]:
        if names[r[3]]=='cuMemcpyHtoD_v2' and r[4] in cid2m: descbytes.append(cid2m[r[4]][4])
    if f['intersubmit_gap']>200: inter_big.append((f['a'],round(f['intersubmit_gap']),round(f['app_between']),round(f['B_issue']),round(f['Aret_to_lastdev'])))
print("APIs between readback end and next submit A (whole window):",dict(c), {k:v/1e3 for k,v in tot.items()})
print("desc upload bytes/frame median %.0f p90 %.0f n=%d"%(st.median(descbytes),sorted(descbytes)[int(.9*len(descbytes))],len(descbytes)))
print("frames with intersubmit gap>200us (a, gap, app_between, B_issue, Aret_to_lastdev):",len(inter_big)); print(inter_big[:12])
# 2. B-submit copy API: the cuMemcpy2DAsync that maps to rb
for f in F[:3]:
    tA,tB,tA2=f['tA'],f['tB'],f['tA2']
    seg=Ms[bisect.bisect_left(ms,tB):bisect.bisect_left(ms,tA2)]
    rb=[m for m in seg if m[3]==8 and m[4]==3686400][0]
    api=[r for r in R if r[4]==rb[5]]
    print("rb api:",[(names[r[3]],(r[0]-tB)/1e3,(r[1]-r[0])/1e3) for r in api])
# 3. kernel name ids used at end of render: 774
print("kernel shortName 774 =",names.get(774))
# 4. worker thread: sync count per frame and their start relative to tA (are there 2 per frame = A and B?)
Rw=sorted(r for r in R if r[2]==WORK)
per=collections.Counter()
for f in F:
    n=sum(1 for r in Rw if f['tA']<=r[0]<f['tA2'] and names[r[3]]=='cuEventSynchronize'); per[n]+=1
print("worker cuEventSynchronize per frame:",per)
# how long after tA does the worker's sync for A start and end (should end ~ when render completes)
offs=[]
for f in F:
    ws=[r for r in Rw if f['tA']<=r[0]<f['tB'] and names[r[3]]=='cuEventSynchronize']
    if ws: offs.append(((ws[-1][0]-f['tA'])/1e3,(ws[-1][1]-f['tA'])/1e3, f['render_span']*1e3))
print("worker sync for A: start med %.0f us, end med %.0f us, render_span med %.0f us"%(st.median(o[0] for o in offs),st.median(o[1] for o in offs),st.median(o[2] for o in offs)))
