import sqlite3, bisect
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
sc=list(c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME"))[0][0]
ts=[int(l.split()[0])-sc for l in open('/tmp/postdead/submit-ts.txt')]
names={r[0]:r[1] for r in c.execute("select id,value from StringIds")}
# pick frame boundary at submit index 3200 (even) in the heavy band
for base in (3200, 3300):
    a=ts[base]-3.5e6; b=ts[base+1]+0.6e6
    ev=[]
    for s,e,sid,corr,nm,gx in c.execute("select start,end,streamId,correlationId,shortName,gridX from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=?",(a,b)):
        ev.append((s,e,'K s%d %s g%d'%(sid,names[nm][:28],gx)))
    for s,e,sid,ck,by in c.execute("select start,end,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind not in (11,12)",(a,b)):
        ev.append((s,e,'C%d s%d %dB'%(ck,sid,by)))
    for s,e,sid,by in c.execute("select start,end,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where end>=? and start<=?",(a,b)):
        ev.append((s,e,'S s%d %dB'%(sid,by)))
    for s,e,nid,tid in c.execute("select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<=? and nameId in (select id from StringIds where value in ('cuStreamSynchronize','cuCtxSynchronize','cuEventSynchronize','cuMemcpyDtoH_v2','cuMemcpyHtoD_v2','cuLaunchHostFunc','cuEventRecord','cuMemcpy2DAsync_v2','cuEventCreate','cuEventQuery'))",(a,b)):
        ev.append((s,e,'API%s %s'%('-worker' if tid==297865344641096 else '', names[nid])))
    ev.append((ts[base],ts[base],'=== SUBMIT %d (even)'%base)); ev.append((ts[base+1],ts[base+1],'=== SUBMIT %d (odd)'%(base+1)))
    ev.sort()
    # collapse dense kernel runs: print only events with gap>20us or specials
    print("\n##### boundary around submit", base, "rel s", ts[base]/1e9)
    prev_end=None; skipped=0
    for s,e,d in ev:
        special = d.startswith('===') or d.startswith('API') or d.startswith('C') or d.startswith('S')
        if prev_end is not None and s-prev_end<15e3 and not special:
            skipped+=1; prev_end=max(prev_end,e); continue
        if skipped: print("      ... %d dense device ops skipped, until %.1f us"%(skipped,(prev_end-ts[base])/1e3)); skipped=0
        print("%10.1f us  dur %8.1f  %s"%((s-ts[base])/1e3,(e-s)/1e3,d))
        prev_end=e if prev_end is None else max(prev_end,e)
