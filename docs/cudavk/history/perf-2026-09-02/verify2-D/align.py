import sqlite3, sys, statistics as st, bisect, collections
for DB, TS in [('/tmp/postdead/trace.sqlite','/tmp/postdead/submit-ts.txt'),('/tmp/postpin-rtx2/trace.sqlite','/tmp/postpin-rtx2/ts.txt')]:
    c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    sess = c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    ts = [int(l.split()[0]) - sess for l in open(TS) if len(l.split()) == 2]
    k0, k1 = c.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    print("==", DB, "sess", sess, "kernels span (s since sess)", k0/1e9, k1/1e9, "submits", len(ts), "first ts", ts[0]/1e9, "last", ts[-1]/1e9)
    rej = [r[0] for r in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMSET where bytes=? and streamId=13 order by start", (1280*720*32,))]
    print("   total retry draws in trace:", len(rej))
    fr = [bisect.bisect_right(ts, t)//2 for t in rej]
    h = collections.Counter(f//100*100 for f in fr)
    print("   retry draws per 100-frame bucket:", sorted(h.items()))
    # submits inside trace
    s_in = [i for i,t in enumerate(ts) if k0 <= t <= k1]
    print("   submits inside kernel span: %d..%d (frames %d..%d)" % (s_in[0], s_in[-1], s_in[0]//2, s_in[-1]//2))
    # frame time profile per 100 frames (median), frames = pairs
    NF = len(ts)//2 - 1
    ft = [(ts[2*k+2]-ts[2*k])/1e6 for k in range(NF)]
    prof = [(b, round(st.median(ft[b:b+100]),2)) for b in range(0, NF, 100)]
    print("   frame-time median per 100 frames:", prof)
