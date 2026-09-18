import sqlite3, bisect, json, statistics as st
from collections import defaultdict
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
sc=list(c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME"))[0][0]
ts=[int(l.split()[0])-sc for l in open('/tmp/postdead/submit-ts.txt')]
T0=ts[2782]; T1=list(c.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
FR=412.5
print("window rel s", T0/1e9, T1/1e9, "frames(doc)", FR, "submits in window", bisect.bisect_right(ts,T1)-2782)
# device ops in window: kernels, memcpy (non-UVM), memset
ops=[]  # (start,end,stream,kind,bytes,corr)
for s,e,sid,corr in c.execute("select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=?",(T0-5e6,T1)):
    ops.append((s,e,sid,'K',0,corr))
for s,e,sid,ck,b,corr in c.execute("select start,end,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind not in (11,12)",(T0-5e6,T1)):
    ops.append((s,e,sid,'C%d'%ck,b,corr))
for s,e,sid,b,corr in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>=? and start<=?",(T0-5e6,T1)):
    ops.append((s,e,sid,'S',b,corr))
ops.sort()
starts=[o[0] for o in ops]
# prefix max of ends for "last end before t"
pm=[];m=-1
for o in ops:
    m=max(m,o[1]); pm.append(m)
def last_end_before(t):
    i=bisect.bisect_left(starts,t)  # ops starting before t
    # need max end among ops with start< t ; but end may exceed t (still running) -> means device busy at t
    return pm[i-1] if i>0 else None
def next_start_after(t, exclude_corr=None):
    i=bisect.bisect_left(starts,t)
    while i<len(ops) and exclude_corr is not None and ops[i][5]==exclude_corr: i+=1
    return ops[i] if i<len(ops) else None
# union busy in window
busy=0; cur_s=None; cur_e=None
for o in ops:
    s,e=max(o[0],T0),min(o[1],T1)
    if e<=s: continue
    if cur_e is None or s>cur_e:
        if cur_e is not None: busy+=cur_e-cur_s
        cur_s,cur_e=s,e
    else: cur_e=max(cur_e,e)
busy+=cur_e-cur_s
print("device union busy ms/frame", busy/1e6/FR, "idle ms/frame", (T1-T0-busy)/1e6/FR, "wall ms/frame", (T1-T0)/1e6/FR)
# side stream ops (streams != 13) sorted by end, for join measurement
MAIN=13
side_ends=sorted(o[1] for o in ops if o[2]!=MAIN and o[3]=='K')
main_ends=sorted(o[1] for o in ops if o[2]==MAIN and o[3]=='K')
main_starts=sorted(o[0] for o in ops if o[2]==MAIN and o[3]=='K')
kstarts=sorted(o[0] for o in ops if o[3]=='K')
def last_le(arr,t):
    i=bisect.bisect_right(arr,t); return arr[i-1] if i>0 else None
def first_ge(arr,t):
    i=bisect.bisect_left(arr,t); return arr[i] if i<len(arr) else None
# runtime rows main thread in window
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
MT=297865344641070
rt=list(c.execute("select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<=? order by start",(MT,T0,T1)))
rt_names=[names[r[2]] for r in rt]
# memcpy by correlation
mc={corr:(s,e,b) for s,e,b,corr in c.execute("select start,end,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and start>=? and start<=?",(T0-1e7,T1+1e7))}
LAUNCH={'cuLaunchKernel','cuLaunchKernelEx'}
sync_rows=[]
for i,(s,e,nid,corr) in enumerate(rt):
    n=rt_names[i]
    if n in ('cuStreamSynchronize','cuCtxSynchronize'):
        # what follows
        j=i+1; nxt=None; dtoh=None
        while j<len(rt) and j<i+6:
            if rt_names[j]=='cuMemcpyDtoH_v2':
                dtoh=rt[j]; break
            if rt_names[j] in ('cuStreamSynchronize','cuCtxSynchronize'): break
            j+=1
        # next launch API after sync (or after dtoh)
        k=(j if dtoh else i)+1
        while k<len(rt) and rt_names[k] not in LAUNCH and rt_names[k] not in ('cuStreamSynchronize','cuCtxSynchronize','cuMemcpyDtoH_v2'): k+=1
        nxt_launch=rt[k] if k<len(rt) and rt_names[k] in LAUNCH else None
        sync_rows.append(dict(kind=n, s=s, e=e, dtoh=dtoh, nxt_launch=nxt_launch, idx=i))
print("syncs in window main thread:", len(sync_rows), "per frame", len(sync_rows)/FR)
# classify
def site(r):
    if r['kind']=='cuCtxSynchronize': return 'ctxsync'
    if r['dtoh'] is None: return 'stream-sync/no-copy (peel)'
    corr=r['dtoh'][3]
    b=mc.get(corr,(None,None,None))[2]
    if b is None: return 'stream-sync+dtoh(?)'
    if b==24: return 'segment counters (24B)'
    if b==4: return 'first-draw total (4B)'
    if b==8: return 'quad counters (8B)'
    if b in (16,20): return 'indirect draw'
    if b>=28 and (b-24)%4==0: return 'episode drain (%dB, nsegs=%d)'%(b,(b-24)//4)
    return 'other dtoh %dB'%b
agg=defaultdict(list)
rows=[]
for r in sync_rows:
    sname=site(r)
    fam='drain' if sname.startswith('episode drain') else sname
    spin=r['e']-r['s']
    last_end=last_end_before(r['e'])
    # device idle at sync return?  last_end <= e means device empty (ignoring ops still running)
    dev_empty_at_return = last_end is not None and last_end<=r['e']
    copy_s=copy_e=None
    if r['dtoh']:
        m=mc.get(r['dtoh'][3]); 
        if m: copy_s,copy_e=m[0],m[1]
    ref_end = copy_e if copy_e else r['e']
    nk=first_ge(kstarts, ref_end)   # next kernel start anywhere
    nk_after_sync=first_ge(kstarts, r['e'])
    # bubble components (ns)
    d=dict(site=sname,fam=fam,spin=spin,
           ret_lat=(r['e']-last_end) if last_end is not None else None,   # sync return latency after last device op end
           host_to_copy=(copy_s-r['e']) if copy_s else None,
           copy=(copy_e-copy_s) if copy_s else None,
           post=(nk-ref_end) if nk else None,                 # copy end (or sync end) -> next kernel start
           total_gap=(nk-last_end) if (nk and last_end is not None) else None,  # last device end -> next kernel start
           host_decide=(r['nxt_launch'][0]-ref_end) if r['nxt_launch'] else None,
           launch_lat=(nk-r['nxt_launch'][1]) if (r['nxt_launch'] and nk) else None,
           dev_empty=dev_empty_at_return, t=r['s'])
    # join: last side-stream kernel end before sync end, and main-stream last end
    ls=last_le(side_ends, r['e']); lm=last_le(main_ends, r['e'])
    d['side_quiet_before_sync']=(r['e']-ls) if ls else None
    d['main_alone']=(lm-ls) if (ls and lm and lm>ls) else 0
    d['side_recent']= ls is not None and (r['e']-ls)<3e6
    rows.append(d); agg[fam].append(d)
def q(v,p): 
    v=sorted(v); 
    return v[min(len(v)-1,int(p*len(v)))] if v else None
out={}
print()
print("%-36s %6s %7s | %8s %8s %8s %8s %8s | %8s %8s %8s | %8s %8s"%("site","n","n/fr","spin","ret_lat","h2copy","copy","post","total","tot/fr","spin/fr","sidequiet","main_alone"))
for fam,L in sorted(agg.items(), key=lambda x:-len(x[1])):
    n=len(L)
    def col(key, agg_fn=st.median):
        v=[x[key] for x in L if x[key] is not None]; return agg_fn(v)/1e3 if v else float('nan')
    def sumfr(key):
        v=[x[key] for x in L if x[key] is not None]; return sum(v)/1e6/FR
    print("%-36s %6d %7.2f | %8.1f %8.1f %8.1f %8.1f %8.1f | %8.1f %8.3f %8.3f | %8.1f %8.1f"%(fam,n,n/FR,col('spin'),col('ret_lat'),col('host_to_copy'),col('copy'),col('post'),col('total_gap'),sumfr('total_gap'),sumfr('spin'),col('side_quiet_before_sync'),col('main_alone')))
    out[fam]=dict(n=n,per_frame=n/FR,
        spin_med_us=col('spin'),spin_mean_us=col('spin',st.mean),spin_ms_per_frame=sumfr('spin'),
        ret_lat_med_us=col('ret_lat'),host_to_copy_med_us=col('host_to_copy'),copy_med_us=col('copy'),
        post_med_us=col('post'),post_mean_us=col('post',st.mean),post_ms_per_frame=sumfr('post'),
        post_p10=q([x['post'] for x in L if x['post'] is not None],0.1)/1e3 if any(x['post'] is not None for x in L) else None,
        post_p90=q([x['post'] for x in L if x['post'] is not None],0.9)/1e3 if any(x['post'] is not None for x in L) else None,
        total_gap_med_us=col('total_gap'),total_gap_mean_us=col('total_gap',st.mean),total_gap_ms_per_frame=sumfr('total_gap'),
        host_decide_med_us=col('host_decide'),launch_lat_med_us=col('launch_lat'),
        dev_empty_frac=sum(1 for x in L if x['dev_empty'])/n,
        side_quiet_med_us=col('side_quiet_before_sync'),main_alone_med_us=col('main_alone'),main_alone_ms_per_frame=sumfr('main_alone'),
        side_recent_frac=sum(1 for x in L if x['side_recent'])/n)
# distribution of drain post gaps
L=agg['drain']
pg=sorted(x['post']/1e3 for x in L if x['post'] is not None)
tg=sorted(x['total_gap']/1e3 for x in L if x['total_gap'] is not None)
print("\ndrain post-gap (copy end -> next kernel start) us: p10 %.1f p25 %.1f p50 %.1f p75 %.1f p90 %.1f p99 %.1f max %.1f mean %.1f"%(q(pg,.1),q(pg,.25),q(pg,.5),q(pg,.75),q(pg,.9),q(pg,.99),pg[-1],st.mean(pg)))
print("drain total gap (last dev end -> next kernel start) us: p10 %.1f p50 %.1f p90 %.1f p99 %.1f mean %.1f sum/frame ms %.3f"%(q(tg,.1),q(tg,.5),q(tg,.9),q(tg,.99),st.mean(tg),sum(tg)/1e3/FR))
sp=sorted(x['spin']/1e3 for x in L)
print("drain spin us: p10 %.1f p50 %.1f p90 %.1f mean %.1f sum/frame ms %.3f"%(q(sp,.1),q(sp,.5),q(sp,.9),st.mean(sp),sum(sp)/1e3/FR))
# the spin includes device work; how much of spin was device-busy vs. idle? device busy time within [s,e]
# total idle in window inside sync bubbles vs total idle
bub=sum(x['total_gap'] for x in rows if x['total_gap'] is not None)
print("\nsum of all sync total gaps ms/frame %.3f  (of device idle %.3f ms/frame)"%(bub/1e6/FR,(T1-T0-busy)/1e6/FR))
# hostfunc
hf=[r for i,r in enumerate(rt) if rt_names[i]=='cuLaunchHostFunc']
print("cuLaunchHostFunc in window", len(hf), len(hf)/FR)
# HtoD sync copies (descriptor upload)
hd=[r for i,r in enumerate(rt) if rt_names[i]=='cuMemcpyHtoD_v2']
print("cuMemcpyHtoD_v2 (sync) in window", len(hd), len(hd)/FR, "sum ms/frame", sum(e-s for s,e,_,_ in hd)/1e6/FR)
# worker thread cuEventSynchronize in window
ev=list(c.execute("select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId=986 and start>=? and start<=?",(297865344641096,T0,T1)))
print("worker cuEventSynchronize in window", len(ev), len(ev)/FR, "sum ms/frame", sum(e-s for s,e in ev)/1e6/FR)
json.dump(dict(window=dict(T0=T0,T1=T1,frames=FR,busy_ms_fr=busy/1e6/FR,idle_ms_fr=(T1-T0-busy)/1e6/FR,wall_ms_fr=(T1-T0)/1e6/FR),sites=out,
    drain_post_pcts=dict(p10=q(pg,.1),p25=q(pg,.25),p50=q(pg,.5),p75=q(pg,.75),p90=q(pg,.9),p99=q(pg,.99),max=pg[-1],mean=st.mean(pg)),
    hostfunc=dict(n=len(hf),per_frame=len(hf)/FR), htod_sync=dict(n=len(hd),per_frame=len(hd)/FR,ms_fr=sum(e-s for s,e,_,_ in hd)/1e6/FR),
    worker_evsync=dict(n=len(ev),per_frame=len(ev)/FR,ms_fr=sum(e-s for s,e in ev)/1e6/FR)),
    open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/host-sync/bubbles.json','w'),indent=1)
# save per-row for later
import pickle; pickle.dump(rows,open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/host-sync/rows.pkl','wb'))
