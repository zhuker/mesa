import sqlite3, sys, statistics as st, bisect, collections
for DB in ['/tmp/postdead/trace.sqlite','/tmp/postpin-rtx2/trace.sqlite']:
    c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    K = [(r[0], r[1], names[r[2]]) for r in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL where streamId=13 order by start")]
    ks = [k[0] for k in K]
    rej = [r[0] for r in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMSET where bytes=? and streamId=13 order by start", (1280*720*32,))]
    conv1 = conv2 = conv3 = n = 0; flat = 0; cseries = []
    wb2 = []; 
    for t in rej:
        i = bisect.bisect_left(ks, t)
        wbs = [k for k in K[i:i+200] if k[2] == 'cp_fs_writeback'][:8]
        cps = [k for k in K[i:i+200] if k[2] == 'cp_fs_compact'][:8]
        if len(wbs) < 8: continue
        n += 1
        d = [(e - s) / 1000.0 for s, e, _ in wbs]
        cd = [(e - s) / 1000.0 for s, e, _ in cps]
        wb2.append(d)
        cseries.append(cd)
        if d[1] <= 1.55: conv1 += 1   # nothing written in pass 1 -> converged after pass 0
        if d[2] <= 1.55: conv2 += 1   # converged after pass 1
        if d[3] <= 1.55: conv3 += 1
    print("==", DB, "draws", n)
    print("   wb at floor (<=1.55us) in pass1: %d, pass2: %d, pass3: %d  -> converged after pass0 %.1f%%, after pass1 %.1f%%, after pass2 %.1f%%" % (conv1, conv2, conv3, 100*conv1/n, 100*conv2/n, 100*conv3/n))
    # wb duration histogram pass 2
    h = collections.Counter(round(d[2], 1) for d in wb2); print("   pass-2 wb duration histogram:", sorted(h.items())[:12])
    h = collections.Counter(round(d[1], 1) for d in wb2); print("   pass-1 wb duration histogram:", sorted(h.items())[:12])
    # compact: per-draw delta between pass 3 and pass 7
    dd = [cd[3] - cd[7] for cd in cseries]
    print("   compact pass3 - pass7 per draw: median %.2f us, p10 %.2f p90 %.2f" % (st.median(dd), sorted(dd)[len(dd)//10], sorted(dd)[9*len(dd)//10]))
    dd = [cd[2] - cd[3] for cd in cseries]
    print("   compact pass2 - pass3 per draw: median %.2f us" % st.median(dd))
