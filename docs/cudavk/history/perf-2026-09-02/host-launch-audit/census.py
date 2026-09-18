import sqlite3, statistics
c=sqlite3.connect('file:/tmp/postdead/trace.sqlite?mode=ro',uri=True)
cur=c.cursor()
SESSION=1503511535334936
sub=[int(l.split()[0])-SESSION for l in open('/tmp/postdead/submit-ts.txt')]
# window: submits index 2785 (line 2786) to trace end
T0=sub[2785]; T1=12340525407
nsub=sum(1 for t in sub if T0<=t<=T1)
frames=nsub/2
print("T0",T0,"T1",T1,"submits in window",nsub,"frames",frames)
def pct(v,p):
    v=sorted(v); 
    if not v: return 0
    k=int(round((len(v)-1)*p)); return v[k]
print("\n== RUNTIME API census in window (host side) ==")
rows=cur.execute(f"""select s.value, r.globalTid, count(*), sum(r.end-r.start), 
 group_concat(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
 where r.start>={T0} and r.start<{T1} group by s.value, r.globalTid order by count(*) desc""").fetchall()
tot=0; tott=0
print(f"{'api':44s} {'tid':>6s} {'n':>7s} {'n/frame':>8s} {'sum_us/f':>9s} {'mean_us':>8s} {'p50':>7s} {'p90':>7s} {'p99':>8s} {'max':>9s}")
for name,tid,n,tsum,durs in rows:
    d=[int(x) for x in durs.split(',')]
    tot+=n; tott+=tsum
    print(f"{name[:44]:44s} {str(tid)[-2:]:>6s} {n:7d} {n/frames:8.2f} {tsum/frames/1000:9.1f} {tsum/n/1000:8.2f} {pct(d,.5)/1000:7.2f} {pct(d,.9)/1000:7.2f} {pct(d,.99)/1000:8.2f} {max(d)/1000:9.1f}")
print("TOTAL", tot, tot/frames, "per frame;", tott/frames/1000, "us/frame host in CUDA APIs (incl. syncs)")
