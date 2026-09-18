import statistics as st, os, pickle, csv
kinds=[k[0] for k in pickle.load(open('kinds.pkl','rb'))]
def load(p):
    out=[]
    for l in open(p):
        parts=l.split()
        if len(parts)==2 and parts[0].isdigit(): out.append(int(parts[0]))
    return out
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
def frames(ts, lo, hi):
    # regular frames: A at i, B at i+1, A at i+2
    out=[]
    for i in range(lo, min(hi, len(ts)-2)):
        if kinds[i]=='A' and kinds[i+1]=='B' and kinds[i+2]=='A':
            out.append(((ts[i+1]-ts[i])/1e6,(ts[i+2]-ts[i+1])/1e6,(ts[i+2]-ts[i])/1e6))
    return out
def stats(ts,lo,hi,label):
    fr=frames(ts,lo,hi)
    a2b=[f[0] for f in fr]; b2a=[f[1] for f in fr]; FR=[f[2] for f in fr]
    print("%-34s n=%4d | frame med %.3f | A->B med %.3f p10 %.3f p90 %.3f | B->A' med %.3f p10 %.3f p90 %.3f mean %.3f"%(label,len(fr),st.median(FR),st.median(a2b),q(a2b,.1),q(a2b,.9),st.median(b2a),q(b2a,.1),q(b2a,.9),st.mean(b2a)))
    return st.median(FR),st.median(a2b),st.median(b2a)
runs={r:load('/tmp/pin-rtx-ab/favorite3-%s/ts.txt'%r) for r in ('c1','c2','c3','p1','p2','p3')}
for band,(lo,hi) in (('whole real window idx>=2782',(2782,6939)),('light band frames 1700-2100',(3400,4200)),('heavy band frames 2150-3150',(4300,6300))):
    print("---",band,"---")
    res={r:stats(ts,lo,hi,'pin-rtx-ab '+r) for r,ts in runs.items()}
    for j,name in ((0,'frame'),(1,"A->B"),(2,"B->A'")):
        c=st.median([res[r][j] for r in res if r[0]=='c']); p=st.median([res[r][j] for r in res if r[0]=='p'])
        print("   pooled %-6s control %.3f pinned %.3f delta %.3f"%(name,c,p,c-p))
print("--- other runs ---")
ts=load('/tmp/postdead/submit-ts.txt'); stats(ts,2783,3608,'postdead (traced, pre-pin) trace window'); stats(ts,2782,6939,'postdead whole'); stats(ts,4300,6300,'postdead heavy')
ts=load('/tmp/favorite3-perf-current/ts.txt'); stats(ts,2782,6939,'perf run (pre-pin, perf attached) whole'); stats(ts,4300,6300,'perf run heavy')
# how many regular frames vs total frames in the real window
import collections
print("kinds after 2782:",collections.Counter(kinds[2782:6939]))
