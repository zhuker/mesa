#!/usr/bin/env python3
"""Which host-visible buffers do shaders consume directly (descriptor buffer infos,
vertex/index binds) in the heavy band? Multiline-aware."""
import re, glob, os, collections, json, sys
SRC=os.path.expanduser('~/favorite3-cpp/out/src')
files=sorted(glob.glob(SRC+'/frame_*.cpp'))
census=json.load(open(sys.argv[1]))
hv_mem={a['mem']:a for a in census['host_visible_allocs']}
# rebuild buffer->mem map
re_bind=re.compile(r'vkBindBufferMemory\(pDevice_0, (\w+), (\w+), (\d+)UL\)')
re_buf=re.compile(r'VkBufferCreateInfo\s+(\w+)\s*\{\s*VkStructureType\(12\),\s*NULL,\s*VkBufferCreateFlags\((\d+)\),\s*(\d+)UL,\s*VkBufferUsageFlags\((\d+)\),.*?VK_CALL_CHECK\(vkCreateBuffer\(pDevice_0, &\1, nullptr, &(\w+)\)', re.S)
re_dbi=re.compile(r'VkDescriptorBufferInfo\s+(\w+)\s*\{\s*(\w+),\s*(\d+)UL,\s*(\d+)UL,\s*\};', re.S)
re_vbarr=re.compile(r'VkBuffer\s+(\w+)\[\]\s*=\s*\{([^}]*)\};\s*VkDeviceSize\s+(\w+)\[\]\s*=\s*\{([^}]*)\};\s*vkCmdBindVertexBuffers\(', re.S)
re_ib=re.compile(r'vkCmdBindIndexBuffer\(\w+\[0\], (\w+), (\d+)UL')
re_submit=re.compile(r'vkQueueSubmit\(')
re_dyn=re.compile(r'vkCmdBindDescriptorSets\([^;]*\);', re.S)
bufmem={}; bufinfo={}
submit=0
HB=(2150,3150)
desc_by_mem=collections.Counter(); desc_by_type=collections.Counter(); desc_ranges=collections.defaultdict(set)
vb_by_mem=collections.Counter(); vb_by_type=collections.Counter(); ib_by_mem=collections.Counter(); ib_by_type=collections.Counter()
desc_offsets_7086=collections.defaultdict(set)  # frame -> set of (buf,offset,range)
draws_hb=0
allocs_type={a['mem']:a['cudavk_type'] for a in census['host_visible_allocs']}
def mtype(buf):
    m=bufmem.get(buf); 
    if m is None: return 'unbound'
    return allocs_type.get(m,0)
for f in files:
    txt=open(f).read()
    for m in re_buf.finditer(txt): bufinfo[m.group(5)]=dict(size=int(m.group(3)),usage=int(m.group(4)))
    # sequential walk by statements (split on ';' is unreliable; use finditer positions)
    events=[]
    for m in re_bind.finditer(txt): events.append((m.start(),'bind',m))
    for m in re_dbi.finditer(txt): events.append((m.start(),'dbi',m))
    for m in re_vbarr.finditer(txt): events.append((m.start(),'vb',m))
    for m in re_ib.finditer(txt): events.append((m.start(),'ib',m))
    for m in re_submit.finditer(txt): events.append((m.start(),'submit',m))
    for m in re.finditer(r'vkCmdDraw(Indexed)?\(', txt): events.append((m.start(),'draw',m))
    events.sort(key=lambda e:e[0])
    for pos,kind,m in events:
        fr=submit//2
        if kind=='bind': bufmem[m.group(1)]=m.group(2); continue
        if kind=='submit': submit+=1; continue
        if not (HB[0]<=fr<HB[1]): continue
        if kind=='draw': draws_hb+=1
        elif kind=='dbi':
            b=m.group(2); mem=bufmem.get(b); t=mtype(b)
            desc_by_type[t]+=1; desc_by_mem[mem]+=1; desc_ranges[mem].add(int(m.group(4)))
            if mem=='pMemory_7086': desc_offsets_7086[fr].add((b,int(m.group(3)),int(m.group(4))))
        elif kind=='vb':
            for b in re.findall(r'pBuffer_\d+', m.group(2)):
                vb_by_type[mtype(b)]+=1; vb_by_mem[bufmem.get(b)]+=1
        elif kind=='ib':
            b=m.group(1); ib_by_type[mtype(b)]+=1; ib_by_mem[bufmem.get(b)]+=1
n=HB[1]-HB[0]
print("heavy band frames",n,"draws/frame",draws_hb/n)
print("descriptor buffer infos per frame by cudavk mem type:",{k:v/n for k,v in desc_by_type.items()})
print("descriptor buffer infos per frame by host-visible mem:",{k:v/n for k,v in desc_by_mem.items() if k in hv_mem})
print("  ranges seen per hv mem:",{k:sorted(v)[:10] for k,v in desc_ranges.items() if k in hv_mem})
print("vertex buffer binds per frame by type:",{k:v/n for k,v in vb_by_type.items()})
print("vertex buffer binds per frame by hv mem:",{k:v/n for k,v in vb_by_mem.items() if k in hv_mem})
print("index buffer binds per frame by type:",{k:v/n for k,v in ib_by_type.items()})
print("index buffer binds per frame by hv mem:",{k:v/n for k,v in ib_by_mem.items() if k in hv_mem})
# UBO ring behaviour in pMemory_7086: distinct offsets per frame, min/max
fr0=HB[0]+100
for fr in (fr0,fr0+1,fr0+2):
    s=sorted(desc_offsets_7086.get(fr,set()), key=lambda x:x[1])
    print(f"frame {fr}: {len(s)} UBO descriptor writes into pMemory_7086; offsets {s[:6]} ... {s[-3:] if len(s)>6 else ''}")
# buffers in pMemory_7086
print("buffers bound to pMemory_7086:",[(b,bufinfo.get(b)) for b,m in bufmem.items() if m=='pMemory_7086'])
print("buffers bound to pMemory_493011/493016:",[(b,bufinfo.get(b),m) for b,m in bufmem.items() if m in ('pMemory_493011','pMemory_493016')])
print("buffers bound to pMemory_6991/13259/13254/3913:",[(b,bufinfo.get(b),m) for b,m in bufmem.items() if m in ('pMemory_6991','pMemory_13259','pMemory_13254','pMemory_3913')])
