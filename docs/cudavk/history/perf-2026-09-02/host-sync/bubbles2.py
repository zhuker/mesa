import sqlite3, bisect, json, statistics as st, pickle
from collections import defaultdict, Counter
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
sc=list(c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME"))[0][0]
ts=[int(l.split()[0])-sc for l in open('/tmp/postdead/submit-ts.txt')]
T0=ts[2782]; T1=list(c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
FR=412.5
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
MT=297865344641070
rt=list(c.execute("select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<=? order by start",(MT,T0-2e7,T1+2e7)))
rtn=[names[r[2]] for r in rt]
rts=[r[0] for r in rt]
# device ops
ops=[]
for s,e,sid,corr in c.execute("select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=?",(T0-5e6,T1)):
    ops.append((s,e,sid,'K',corr))
for s,e,sid,ck,corr in c.execute("select start,end,streamId,copyKind,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind not in (11,12)",(T0-5e6,T1)):
    ops.append((s,e,sid,'C%d'%ck,corr))
for s,e,sid,corr in c.execute("select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>=? and start<=?",(T0-5e6,T1)):
    ops.append((s,e,sid,'S',corr))
ops.sort(); starts=[o[0] for o in ops]
pm=[];m=-1
for o in ops: m=max(m,o[1]); pm.append(m)
kern=[o for o in ops if o[3]=='K']; kstarts=[o[0] for o in kern]
kends_sorted=sorted(o[1] for o in kern)
corr2k={o[4]:o for o in kern}
def last_end_before(t):
    i=bisect.bisect_left(starts,t); return pm[i-1] if i>0 else None
def first_ge(arr,t):
    i=bisect.bisect_left(arr,t); return arr[i] if i<len(arr) else None
def last_le(arr,t):
    i=bisect.bisect_right(arr,t); return arr[i-1] if i>0 else None
# submit boundaries in window
subs=[t for t in ts if T0<=t<=T1]
sub_idx=[i for i,t in enumerate(ts) if T0<=t<=T1]
def crosses_submit(a,b):
    i=bisect.bisect_right(subs,a); return i<len(subs) and subs[i]<b
# ---- union idle decomposition
# idle intervals in window
idle=[]; cur_e=None
for o in ops:
    s,e=max(o[0],T0),min(o[1],T1)
    if e<=s: continue
    if cur_e is None: cur_e=e; continue
    if s>cur_e: idle.append((cur_e,s))
    cur_e=max(cur_e,e)
tot_idle=sum(b-a for a,b in idle)
print("idle intervals", len(idle), "total ms/frame %.3f"%(tot_idle/1e6/FR))
# classify idle intervals: contains a submit boundary?  contains a sync return (main-thread cuStreamSynchronize/cuCtxSynchronize end)? contains a hostfunc?
sync_ends=sorted(r[1] for i,r in enumerate(rt) if rtn[i] in ('cuStreamSynchronize','cuCtxSynchronize'))
hf_times=sorted(r[0] for i,r in enumerate(rt) if rtn[i]=='cuLaunchHostFunc')
dtoh_times=sorted(r[0] for i,r in enumerate(rt) if rtn[i]=='cuMemcpyDtoH_v2')
def any_in(arr,a,b):
    i=bisect.bisect_left(arr,a); return i<len(arr) and arr[i]<b
cls=Counter(); clsum=Counter(); hist=defaultdict(list)
for a,b in idle:
    k=[]
    if any_in(subs,a-2e5,b): k.append('submit')   # allow submit ts slightly before idle start (host submits before device drains)
    if any_in(sync_ends,a,b): k.append('sync')
    if any_in(hf_times,a-3e5,b): k.append('hostfunc')
    key='+'.join(k) if k else 'launch-gap'
    cls[key]+=1; clsum[key]+=b-a; hist[key].append(b-a)
print("%-28s %8s %8s %10s %8s %8s"%("idle class","n","n/fr","ms/frame","med us","p90 us"))
for k in sorted(clsum,key=lambda x:-clsum[x]):
    v=sorted(hist[k]); print("%-28s %8d %8.2f %10.3f %8.1f %8.1f"%(k,cls[k],cls[k]/FR,clsum[k]/1e6/FR,v[len(v)//2]/1e3,v[int(.9*len(v))]/1e3))
# launch-gap distribution
v=sorted(hist['launch-gap']); 
print("launch-gap idle: n/frame %.1f  sum %.3f ms/fr; p50 %.1f p90 %.1f p99 %.1f us; share of gaps <5us %.2f, <15us %.2f"%(len(v)/FR,sum(v)/1e6/FR,v[len(v)//2]/1e3,v[int(.9*len(v))]/1e3,v[int(.99*len(v))]/1e3,sum(1 for x in v if x<5e3)/len(v),sum(1 for x in v if x<15e3)/len(v)))
# ---- submit boundaries: idle around each submit ts, by parity (even index = frame boundary since frame = submits 2N,2N+1)
par=defaultdict(list); pre=defaultdict(list); post=defaultdict(list)
for i in sub_idx:
    t=ts[i]; le=last_le(kends_sorted,t); ns=first_ge(kstarts,t)
    if le is None or ns is None: continue
    par[i%2].append(ns-le); pre[i%2].append(t-le); post[i%2].append(ns-t)
for p in (0,1):
    v=sorted(par[p]); print("submit parity %d (%s): n %d, idle around submit: med %.1f us mean %.1f us sum %.3f ms/frame | pre(lastkernel->submit) med %.1f | post(submit->firstkernel) med %.1f"%(p,'frame boundary' if p==0 else 'intra-frame',len(v),v[len(v)//2]/1e3,st.mean(v)/1e3,sum(v)/1e6/FR,st.median(pre[p])/1e3,st.median(post[p])/1e3))
# ---- ctxsync classification by following API names and by crossing a submit
rows=pickle.load(open('rows.pkl','rb'))
# rebuild ctxsync list from rt
ctx=[(i,r) for i,r in enumerate(rt) if rtn[i]=='cuCtxSynchronize' and T0<=r[0]<=T1]
foll=Counter(); gaps=defaultdict(list); spins=defaultdict(list)
for i,r in ctx:
    nxt=tuple(rtn[j] for j in range(i+1,min(i+4,len(rt))))
    key=nxt[0] if nxt else '-'
    le=last_end_before(r[1]); ns=first_ge(kstarts,r[1])
    g=(ns-le) if (le is not None and ns is not None) else None
    cross=crosses_submit(r[0],ns) if ns else False
    key2=key+(' [crosses submit]' if cross else '')
    foll[key2]+=1
    if g is not None: gaps[key2].append(g)
    spins[key2].append(r[1]-r[0])
print("\nctxsync by next API call:")
for k,n in foll.most_common():
    g=sorted(gaps[k]); s=spins[k]
    print("  %-45s n %4d (%.2f/fr) spin med %.1f us sum %.3f ms/fr | gap med %.1f us sum %.3f ms/fr"%(k,n,n/FR,st.median(s)/1e3,sum(s)/1e6/FR,(g[len(g)//2]/1e3 if g else float('nan')),sum(g)/1e6/FR))
# ---- same crossing split for the stream syncs (drain / seg / peel)
fam_rows=defaultdict(lambda: defaultdict(list))
for d in rows:
    if d['total_gap'] is None: continue
    # need next kernel start: t + spin + total? recompute: approximate cross by checking submit in (d['t'], d['t']+d['spin']+d['total_gap']+d['ret_lat'])
    a=d['t']; b=d['t']+d['spin']+(d['total_gap'] or 0)
    key='crosses' if crosses_submit(a,b) else 'inside'
    fam_rows[d['fam']][key].append(d)
print("\nstream syncs split by whether a submit boundary falls inside the bubble:")
for fam,dd in fam_rows.items():
    for key,L in dd.items():
        tg=[x['total_gap'] for x in L]; sp=[x['spin'] for x in L]
        print("  %-30s %-8s n %4d (%.2f/fr) spin sum %.3f ms/fr med %.1f us | gap med %.1f us mean %.1f sum %.3f ms/fr"%(fam,key,len(L),len(L)/FR,sum(sp)/1e6/FR,st.median(sp)/1e3,st.median(tg)/1e3,st.mean(tg)/1e3,sum(tg)/1e6/FR))
# ---- host func bubbles
hf=[(i,r) for i,r in enumerate(rt) if rtn[i]=='cuLaunchHostFunc' and T0<=r[0]<=T1]
hg=[]; hcross=0; nextname=Counter(); prevname=Counter()
for i,r in hf:
    # previous launch on main thread -> its kernel end; next launch -> kernel start
    j=i-1
    while j>0 and rtn[j] not in ('cuLaunchKernel','cuLaunchKernelEx','cuMemsetD32Async','cuMemcpyHtoDAsync_v2','cuMemcpyDtoDAsync_v2','cuMemcpy2DAsync_v2'): j-=1
    k=i+1
    while k<len(rt) and rtn[k] not in ('cuLaunchKernel','cuLaunchKernelEx'): k+=1
    prevname[rtn[j]]+=1; nextname[rtn[i+1] if i+1<len(rt) else '-']+=1
    le=last_end_before(r[1]); ns=first_ge(kstarts,r[1])
    if le is None or ns is None: continue
    cross=crosses_submit(r[0],ns)
    if cross: hcross+=1; continue
    hg.append(ns-le)
hg.sort()
print("\ncuLaunchHostFunc: n %d (%.2f/fr), %d cross a submit boundary; device gap around the rest: med %.1f us p90 %.1f mean %.1f sum %.3f ms/fr"%(len(hf),len(hf)/FR,hcross,hg[len(hg)//2]/1e3,hg[int(.9*len(hg))]/1e3,st.mean(hg)/1e3,sum(hg)/1e6/FR))
print("  prev API before hostfunc:",prevname.most_common(5)); print("  next API after hostfunc:",nextname.most_common(5))
# ---- within-sync-bubble breakdown for drains: is the copy itself on the critical path? host_to_copy + copy + post
L=[d for d in rows if d['fam']=='drain' and d['post'] is not None]
for key in ('ret_lat','host_to_copy','copy','post','host_decide','launch_lat'):
    v=[d[key] for d in L if d[key] is not None]
    print("drain %-14s med %.2f us mean %.2f us"%(key,st.median(v)/1e3,st.mean(v)/1e3))
json.dump(dict(idle_classes={k:dict(n=cls[k],per_frame=cls[k]/FR,ms_fr=clsum[k]/1e6/FR) for k in clsum},
               submit_parity={p:dict(n=len(par[p]),med_us=st.median(par[p])/1e3,mean_us=st.mean(par[p])/1e3,ms_fr=sum(par[p])/1e6/FR,pre_med_us=st.median(pre[p])/1e3,post_med_us=st.median(post[p])/1e3) for p in (0,1)},
               ctx_by_next={k:dict(n=n,per_frame=n/FR,spin_ms_fr=sum(spins[k])/1e6/FR,gap_ms_fr=sum(gaps[k])/1e6/FR) for k,n in foll.items()},
               hostfunc=dict(n=len(hf),per_frame=len(hf)/FR,cross=hcross,gap_med_us=hg[len(hg)//2]/1e3,gap_mean_us=st.mean(hg)/1e3,gap_ms_fr=sum(hg)/1e6/FR)),
          open('bubbles2.json','w'),indent=1)
