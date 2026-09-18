import sqlite3, sys, statistics as st, bisect, collections
DB, TS = '/tmp/postpin-rtx2/trace.sqlite', '/tmp/postpin-rtx2/ts.txt'
c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
sess = c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts = [int(l.split()[0]) - sess for l in open(TS) if len(l.split()) == 2]
REJECT = 1280*720*32
rej = [r[0] for r in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMSET where bytes=? and streamId=13 order by start", (REJECT,))]
# also find passes: count writebacks per draw to verify 8 (assume)
def analyse(S0, S1, label):
    NF = (S1 - S0) // 2
    fb = [ts[S0 + 2*k] for k in range(NF+1)]
    ft = [(fb[k+1]-fb[k])/1e6 for k in range(NF)]
    cnt = [0]*NF
    for t in rej:
        i = bisect.bisect_right(fb, t) - 1
        if 0 <= i < NF: cnt[i] += 1
    fr = [k for k in range(NF) if cnt[k]]
    print("== %s submits %d..%d frames %d: retry draws %d, frames with retry %d (%.1f%%), draws/frame mean %.3f, multi-draw frames %d" % (label, S0, S1, NF, sum(cnt), len(fr), 100*len(fr)/NF, sum(cnt)/NF, sum(1 for x in cnt if x>1)))
    base = st.median(ft)
    print("   frame time median %.3f ms; retry frames median %.3f, non-retry median %.3f" % (base, st.median([ft[k] for k in fr]) if fr else 0, st.median([ft[k] for k in range(NF) if not cnt[k]]) if len(fr)<NF else 0))
    for d in (0.10, 0.15, 0.20, 0.26):
        adj = [ft[k] - d*cnt[k] for k in range(NF)]
        print("   subtract %.2f ms per retry draw -> median %.3f (delta %.3f), mean delta %.3f" % (d, st.median(adj), st.median(adj)-base, st.mean(adj)-st.mean(ft)))
    # run pattern
    pat = ''.join('1' if cnt[k] else '0' for k in range(NF))
    runs=[]; cur=pat[0]; n=0
    for ch in pat:
        if ch==cur: n+=1
        else: runs.append((cur,n)); cur=ch; n=1
    runs.append((cur,n))
    print("   runs:", runs[:20])
analyse(4300, 6300, "task heavy band f2150-3149")
analyse(4900, 6300, "census heavy band f2450-3149")
analyse(2782, 6900, "whole window f1391-3449")
analyse(2782, 3606, "light band f1391-1802 (in postpin trace)")
