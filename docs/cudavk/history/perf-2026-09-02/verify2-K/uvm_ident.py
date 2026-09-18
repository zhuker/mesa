import sqlite3, bisect, collections, statistics as st
c=sqlite3.connect('file:/tmp/postpin-rtx/trace.sqlite?mode=ro',uri=True)
names=dict(c.execute("select id,value from StringIds"))
tid=c.execute("select globalTid,count(*) n from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cuLaunchKernel%' group by globalTid order by n desc limit 1").fetchone()[0]
API=c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? order by end",(tid,)).fetchall()
api_e=[a[1] for a in API]
K=c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ks=[k[0] for k in K]
kend=K[-1][1]
for reg in (0x7bb9fd200000,0x7bb9fc000000,0x7bb9fd000000):
    print('== region %x'%reg)
    pages=c.execute("select printf('%x',virtualAddress>>12), copyKind, migrationCause, count(*), sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind in (11,12) and (virtualAddress>>21)<<21=? group by 1,2,3 order by 2,3,4 desc",(reg,)).fetchall()
    for p in pages[:40]: print('  page',p)
    # CPU faults (12, cause 2): preceding main-thread API
    rows=c.execute("select start,end,virtualAddress from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=12 and migrationCause=2 and (virtualAddress>>21)<<21=? and start<? order by start",(reg,kend)).fetchall()
    prev=collections.Counter(); delays=collections.defaultdict(list)
    for s,e,va in rows:
        i=bisect.bisect_left(api_e,s)-1
        if i<0: continue
        nm=names[API[i][2]]; prev[nm]+=1; delays[nm].append(s-API[i][1])
    print('  CPU-fault rows',len(rows),'preceding main-thread API:')
    for nm,cnt in prev.most_common(8):
        d=delays[nm]; print('    %-28s %6d  delay-after-end median %.1f us p90 %.1f us'%(nm,cnt,st.median(d)/1e3, sorted(d)[int(0.9*len(d))]/1e3))
    # GPU faults (11, cause 2): kernel running at that time
    rows=c.execute("select start,end,virtualAddress,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=11 and migrationCause=2 and (virtualAddress>>21)<<21=? and start<? order by start",(reg,kend)).fetchall()
    run=collections.Counter(); sizes=collections.Counter()
    for s,e,va,by in rows:
        sizes[by]+=1
        j=bisect.bisect_right(ks,s)-1
        # find any kernel overlapping s (check a few back)
        found='none'
        for jj in range(j,max(-1,j-40),-1):
            if K[jj][0]<=s<=K[jj][1]: found=K[jj][2]; break
        run[found]+=1
    print('  GPU-fault rows',len(rows),'sizes',sizes.most_common(5))
    for nm,cnt in run.most_common(8): print('    running kernel %-30s %d'%(nm,cnt))
