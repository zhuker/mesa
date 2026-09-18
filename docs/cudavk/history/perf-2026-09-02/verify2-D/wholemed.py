import sqlite3, statistics as st, bisect
def rej_frames(DB, TS):
    c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    sess = c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    ts = [int(l.split()[0]) - sess for l in open(TS) if len(l.split()) == 2]
    rej = [r[0] for r in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMSET where bytes=? and streamId=13 order by start", (1280*720*32,))]
    return [bisect.bisect_right(ts, t)//2 for t in rej], ts
fa, _ = rej_frames('/tmp/postdead/trace.sqlite','/tmp/postdead/submit-ts.txt')
fb_, ts = rej_frames('/tmp/postpin-rtx2/trace.sqlite','/tmp/postpin-rtx2/ts.txt')
import collections
cnt = collections.Counter(fa) + collections.Counter(fb_)
for S0, S1, label in [(2782, 6900, 'whole window f1391-3449'), (4300, 6300, 'heavy f2150-3149'), (2782, 3606, 'light f1391-1802')]:
    NF = (S1-S0)//2
    ft = [(ts[S0+2*k+2]-ts[S0+2*k])/1e6 for k in range(NF)]
    frames = [S0//2 + k for k in range(NF)]
    c_ = [cnt.get(f, 0) for f in frames]
    base = st.median(ft)
    print("== %s: draws %d over %d frames (%.3f/frame, %.1f%% frames), median %.3f ms (postpin session, CUPTI-inflated)" % (label, sum(c_), NF, sum(c_)/NF, 100*sum(1 for x in c_ if x)/NF, base))
    for d in (0.10, 0.15, 0.20, 0.26):
        adj = [ft[k] - d*c_[k] for k in range(NF)]
        print("   -%.2f ms/draw -> median delta %.3f, mean delta %.3f" % (d, st.median(adj)-base, st.mean(adj)-st.mean(ft)))
