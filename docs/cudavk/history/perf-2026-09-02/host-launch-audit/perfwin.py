# restrict perf samples to the same frames the nsys trace covers (external submit index 2785..3608) and price per call
import re
from collections import defaultdict
ts=[l.split() for l in open('/tmp/favorite3-perf-current/ts.txt')]
W0=int(ts[2785][0])/1e9; W1=int(ts[3608][0])/1e9
frames=(3608-2785)/2
HZ=999.0; w_ms=1000.0/HZ
print("perf sub-window (same frames as trace) s:",W0,W1,"frames",frames)
hdr=re.compile(r'^(\S+)\s+(\d+)/(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+):')
frm=re.compile(r'^\s+([0-9a-f]+)\s+(.*?)\s+\((.*)\)\s*$')
by_api=defaultdict(float); n_main=0; cur=None; fr=[]
def flush(hd,fr):
    global n_main
    if hd is None: return
    pid,tid,t=int(hd[2]),int(hd[3]),float(hd[4])
    if tid!=pid or t<W0 or t>W1: return
    n_main+=1
    lib=[(s,d) for a,s,d in fr if 'libcuda' in d]
    if not lib: by_api['<no CUDA API>']+=1; return
    api='?'
    for s,d in reversed(lib):
        if s!='[unknown]': api=s; break
    by_api[api]+=1
with open('/tmp/favorite3-perf-current/samples-flat.txt') as f:
    for line in f:
        m=hdr.match(line)
        if m: flush(cur,fr); cur=m; fr=[]; continue
        m2=frm.match(line)
        if m2 and cur is not None: fr.append(m2.groups())
    flush(cur,fr)
ms=lambda n:n*w_ms/frames
print("main-thread on-CPU ms/frame in sub-window:", round(ms(n_main),3), "wall ms/frame:", round((W1-W0)*1000/frames,3))
# trace counts per frame from census.py (same frames): 
trace={'cuLaunchKernel':267.91,'cuLaunchKernelEx':159.18,'cuMemcpyHtoDAsync_v2':140.77,'cuMemsetD32Async':88.94,'cuMemsetD8Async':15.21,'cuEventRecord':220.38,'cuStreamWaitEvent':153.46,'cuMemcpy2DAsync_v2':39.07,'cuMemcpyDtoH_v2':8.92,'cuMemcpyDtoDAsync_v2':7.41,'cuMemcpy3DAsync_v2':5.11,'cuStreamSynchronize':11.40,'cuCtxSynchronize':2.04,'cuLaunchHostFunc':3.00,'cuMemcpyHtoD_v2':1.01}
print(f"{'api':26s} {'perf ms/f':>10s} {'samples':>8s} {'trace n/f':>10s} {'us/call (no CUPTI)':>20s} {'CUPTI mean us':>14s}")
cupti={'cuLaunchKernel':2.18,'cuLaunchKernelEx':2.02,'cuMemcpyHtoDAsync_v2':1.86,'cuMemsetD32Async':2.04,'cuMemsetD8Async':2.30,'cuEventRecord':0.27,'cuStreamWaitEvent':0.31,'cuMemcpy2DAsync_v2':3.55,'cuMemcpyDtoH_v2':7.29,'cuMemcpyDtoDAsync_v2':5.30,'cuMemcpy3DAsync_v2':5.06,'cuLaunchHostFunc':2.61,'cuMemcpyHtoD_v2':9.58}
for k,v in sorted(by_api.items(), key=lambda kv:-kv[1]):
    n=trace.get(k)
    per = f"{ms(v)*1000/n:8.2f} +-{(ms(v)*1000/n)/max(v,1)**0.5:5.2f}" if n else ""
    print(f"{k:26s} {ms(v):10.3f} {v:8.0f} {str(n) if n else '':>10s} {per:>20s} {str(cupti.get(k,'')):>14s}")
