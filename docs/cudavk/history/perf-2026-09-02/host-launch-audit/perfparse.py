import re, sys
from collections import defaultdict
W0,W1=[float(x) for x in open('/tmp/favorite3-perf-current/time-range').read().strip().split(',')]
ts=[l.split() for l in open('/tmp/favorite3-perf-current/ts.txt')]
nsub=sum(1 for t in ts if W0*1e9<=int(t[0])<=W1*1e9+1)
frames=nsub/2
HZ=999.0
w_ms=1000.0/HZ
print("window",W0,W1,"submits",nsub,"frames",frames,"sample weight ms",w_ms)
hdr=re.compile(r'^(\S+)\s+(\d+)/(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+):')
frm=re.compile(r'^\s+([0-9a-f]+)\s+(.*?)\s+\((.*)\)\s*$')
SYNC={'cuCtxSynchronize','cuStreamSynchronize','cuEventSynchronize','cuMemcpyDtoH_v2','cuMemcpyHtoD_v2','cuMemFree_v2','cuMemAlloc_v2','cuMemAllocManaged','cuEventQuery','cuStreamQuery'}
def dso_class(d):
    if 'libcuda' in d: return 'libcuda'
    if 'libvulkan_cudavk' in d: return 'driver'
    if 'vdso' in d: return 'vdso'
    if 'libc.so' in d or 'libpthread' in d or 'libm.so' in d: return 'libc'
    if 'vulkan_app' in d or 'submit_shim' in d: return 'app'
    if d=='inlined': return 'inlined'
    if 'libLLVM' in d or 'libnvrtc' in d or 'libnvJit' in d or 'libnvptx' in d: return 'compiler'
    return 'other:'+d.split('/')[-1]
cur=None; frames_l=[]
by_api=defaultdict(float); by_api_site=defaultdict(float); by_api_leaf=defaultdict(float)
cls=defaultdict(float); nocuda_leaf=defaultdict(float); nocuda_top=defaultdict(float)
launchpath=defaultdict(float)
n_main=0
def flush(hd, fr):
    global n_main
    if hd is None: return
    pid,tid,t=int(hd[2]),int(hd[3]),float(hd[4])
    if tid!=pid: return
    if t<W0 or t>W1: return
    n_main+=1
    # frames leaf-first: (sym, dsoclass, rawdso)
    fl=[(s if s!='[unknown]' else '?', dso_class(d), d) for a,s,d in fr]
    # resolve inlined dso class to the next non-inlined frame's dso (toward root)
    resolved=[]
    for i,(s,dc,d) in enumerate(fl):
        if dc=='inlined':
            j=i+1
            while j<len(fl) and fl[j][1]=='inlined': j+=1
            dc = fl[j][1] if j<len(fl) else 'inlined'
        resolved.append((s,dc))
    fl=resolved
    lib=[i for i,(s,dc) in enumerate(fl) if dc=='libcuda']
    if not lib:
        leaf=fl[0]
        nocuda_leaf[(leaf[1], leaf[0].replace(' (inlined)',''))]+=1
        # first driver frame from the leaf
        drv=[s for s,dc in fl if dc=='driver']
        nocuda_top[drv[0].replace(' (inlined)','') if drv else 'no-driver-frame:'+leaf[0]]+=1
        cls['no CUDA API in stack']+=1
        return
    outer=max(lib)
    # API name: outermost named libcuda frame
    api='?'
    for i in range(outer,-1,-1):
        if fl[i][1]=='libcuda' and fl[i][0]!='?':
            api=fl[i][0]; break
    site='?'
    for i in range(outer+1,len(fl)):
        if fl[i][1] in ('driver','app'):
            site=fl[i][0].replace(' (inlined)',''); break
    leafdc=fl[0][1]
    by_api[api]+=1; by_api_site[(api,site)]+=1; by_api_leaf[(api,leafdc)]+=1
    if api in SYNC: cls['CUDA sync/blocking API']+=1
    else: cls['CUDA issue API']+=1
    if any(s.startswith('cp_launch_after') or s.startswith('cp_launch') for s,dc in fl): launchpath[('launch stack', api, leafdc)]+=1
with open('/tmp/favorite3-perf-current/samples-flat.txt') as f:
    for line in f:
        m=hdr.match(line)
        if m:
            flush(cur, frames_l); cur=m; frames_l=[]; continue
        m2=frm.match(line)
        if m2 and cur is not None: frames_l.append(m2.groups())
    flush(cur, frames_l)
def ms(n): return n*w_ms/frames
print("main-thread samples in window", n_main, "=> on-CPU ms/frame", round(ms(n_main),3), "wall ms/frame", round((W1-W0)*1000/frames,3))
print("\n== classes (ms/frame) ==")
for k,v in sorted(cls.items(), key=lambda kv:-kv[1]): print(f"  {k:32s} {ms(v):7.3f}  ({v:.0f} samples)")
print("\n== by API (ms/frame) ==")
for k,v in sorted(by_api.items(), key=lambda kv:-kv[1]): print(f"  {k:28s} {ms(v):7.3f}  ({v:.0f})")
print("\n== by API and driver call site (ms/frame, >=0.005) ==")
for k,v in sorted(by_api_site.items(), key=lambda kv:-kv[1]):
    if ms(v)>=0.005: print(f"  {k[0]:26s} <- {k[1]:44s} {ms(v):7.3f}")
print("\n== by API and leaf DSO (where the cycles are) ==")
for k,v in sorted(by_api_leaf.items(), key=lambda kv:-kv[1]):
    if ms(v)>=0.005: print(f"  {k[0]:26s} leaf={k[1]:10s} {ms(v):7.3f}")
print("\n== samples with no CUDA API frame: leaf (dso, symbol) top 40 ==")
for k,v in sorted(nocuda_leaf.items(), key=lambda kv:-kv[1])[:40]: print(f"  {k[0]:10s} {k[1][:60]:60s} {ms(v):7.3f}")
print("\n== samples with no CUDA API frame: nearest driver frame top 40 ==")
for k,v in sorted(nocuda_top.items(), key=lambda kv:-kv[1])[:40]: print(f"  {k[:70]:70s} {ms(v):7.3f}")
print("\n== stacks through cp_launch(_after): api, leaf dso ==")
for k,v in sorted(launchpath.items(), key=lambda kv:-kv[1]): print(f"  {k[1]:26s} leaf={k[2]:10s} {ms(v):7.3f}")
