import statistics as st, os
def load(p):
    return [int(l.split()[0]) for l in open(p) if l.strip()]
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
def stats(ts, lo, hi, label):
    # odd index = A (render submit), even = B (readback submit); frame k = [ts[a], ts[a+2]) for odd a
    A2B=[(ts[a+1]-ts[a])/1e6 for a in range(lo|1, hi-2, 2)]
    B2A=[(ts[a+2]-ts[a+1])/1e6 for a in range(lo|1, hi-2, 2)]
    FR=[(ts[a+2]-ts[a])/1e6 for a in range(lo|1, hi-2, 2)]
    print("%-28s n=%4d frame med %.3f | A->B (render submit+app) med %.3f p10 %.3f p90 %.3f | B->A' (readback+fence+record) med %.3f p10 %.3f p90 %.3f mean %.3f"%(label,len(FR),st.median(FR),st.median(A2B),q(A2B,.1),q(A2B,.9),st.median(B2A),q(B2A,.1),q(B2A,.9),st.mean(B2A)))
    return st.median(FR), st.median(A2B), st.median(B2A)
runs={}
for r in ('c1','c2','c3','p1','p2','p3'):
    p='/tmp/pin-rtx-ab/favorite3-%s/ts.txt'%r
    if os.path.exists(p):
        ts=load(p); runs[r]=ts
        print(r, len(ts))
print("--- whole real window (ext idx 2782..end) ---")
res={}
for r,ts in runs.items(): res[r]=stats(ts,2782,len(ts),'pin-rtx-ab '+r)
print("--- heavy band (frames 2150-3150 = idx 4300..6300) ---")
for r,ts in runs.items(): stats(ts,4300,6300,'pin-rtx-ab '+r+' heavy')
print("--- light band (frames 1700-2100 = idx 3400..4200) ---")
for r,ts in runs.items(): stats(ts,3400,4200,'pin-rtx-ab '+r+' light')
print("--- traced pre-pin run (/tmp/postdead) same window as the trace (idx 2783..3607) and whole ---")
ts=load('/tmp/postdead/submit-ts.txt'); stats(ts,2783,3608,'postdead trace window'); stats(ts,2782,len(ts),'postdead whole')
ts=load('/tmp/favorite3-perf-current/ts.txt'); stats(ts,2782,len(ts),'perf run (pre-pin, perf attached)')
c=[res[r][2] for r in res if r.startswith('c')]; p=[res[r][2] for r in res if r.startswith('p')]
print("pooled B->A' median: control %.3f pinned %.3f delta %.3f"%(st.median(c),st.median(p),st.median(c)-st.median(p)))
c=[res[r][1] for r in res if r.startswith('c')]; p=[res[r][1] for r in res if r.startswith('p')]
print("pooled A->B median: control %.3f pinned %.3f delta %.3f"%(st.median(c),st.median(p),st.median(c)-st.median(p)))
c=[res[r][0] for r in res if r.startswith('c')]; p=[res[r][0] for r in res if r.startswith('p')]
print("pooled frame median: control %.3f pinned %.3f delta %.3f"%(st.median(c),st.median(p),st.median(c)-st.median(p)))
