import sqlite3, bisect, sys
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
sc=list(c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME"))[0][0]
ts=[int(l.split()[0])-sc for l in open('/tmp/postdead/submit-ts.txt')]
names={r[0]:r[1] for r in c.execute("select id,value for r in []") if False} if False else {r[0]:r[1] for r in c.execute("select id,value from StringIds")}
base=int(sys.argv[1]); lo=float(sys.argv[2]); hi=float(sys.argv[3])
a=ts[base]+lo*1e3; b=ts[base]+hi*1e3
ev=[]
for s,e,sid,corr,nm,gx in c.execute("select start,end,streamId,correlationId,shortName,gridX from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=?",(a,b)):
    ev.append((s,e,'K s%d %s g%d'%(sid,names[nm][:28],gx)))
for s,e,sid,ck,by in c.execute("select start,end,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind not in (11,12)",(a,b)):
    ev.append((s,e,'C%d s%d %dB'%(ck,sid,by)))
for s,e,sid,ck,by in c.execute("select start,end,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? and copyKind in (11,12) and bytes>=65536",(a,b)):
    ev.append((s,e,'UVM%d s%d %dB'%(ck,sid,by)))
for s,e,sid,by in c.execute("select start,end,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where end>=? and start<=?",(a,b)):
    ev.append((s,e,'S s%d %dB'%(sid,by)))
for s,e,nid,tid in c.execute("select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<=? and nameId in (select id from StringIds where value in ('cuStreamSynchronize','cuCtxSynchronize','cuEventSynchronize','cuMemcpyDtoH_v2','cuMemcpyHtoD_v2','cuLaunchHostFunc','cuMemcpy2DAsync_v2','cuEventCreate','cuEventQuery','cuMemcpyDtoDAsync_v2','cuEventDestroy_v2'))",(a,b)):
    ev.append((s,e,'API%s %s'%('-worker' if tid==297865344641096 else '', names[nid])))
# first/last launch api
for s,e,nid,tid in c.execute("select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<=? and nameId in (select id from StringIds where value in ('cuLaunchKernel','cuLaunchKernelEx','cuEventRecord'))",(a,b)):
    ev.append((s,e,'api %s'%names[nid]))
for k in range(base-1,base+3):
    if 0<=k<len(ts) and a<=ts[k]<=b: ev.append((ts[k],ts[k],'=== SUBMIT %d (%s)'%(k,'even' if k%2==0 else 'odd')))
ev.sort()
prev=None; skipped=0; skipped_kinds=set()
for s,e,d in ev:
    dense = d.startswith('K') or d.startswith('api') or (d.startswith('C1') ) or (d.startswith('S') and 'S s' in d and int(d.split()[-1][:-1])<100000)
    if prev is not None and s-prev<12e3 and dense:
        skipped+=1; skipped_kinds.add(d.split()[0]); prev=max(prev,e); continue
    if skipped: print("      ... %d dense ops (%s) skipped, device busy until %.1f us"%(skipped,','.join(sorted(skipped_kinds)),(prev-ts[base])/1e3)); skipped=0; skipped_kinds=set()
    print("%10.1f us  dur %8.1f  %s"%((s-ts[base])/1e3,(e-s)/1e3,d))
    prev=e if prev is None else max(prev,e)
