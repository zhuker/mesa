#!/usr/bin/env python3
"""Parse the compiled GFXReconstruct harness (~/favorite3-cpp/out/src/frame_*.cpp)
for memory allocations, buffer creation/binding, mapping and host writes
(LoadBinaryData), and per-frame host-visible traffic. Read-only."""
import re, glob, os, sys, collections, json
SRC=os.path.expanduser('~/favorite3-cpp/out/src')
files=sorted(glob.glob(SRC+'/frame_*.cpp'))
# memory type remap (harness global_var.cpp) -- original index -> cudavk index
# cudavk: 0 DEVICE_LOCAL ; 1 HOST_VISIBLE|COHERENT|CACHED (managed) ; 2 DL|HV|COH|CACHED (managed)
REMAP={0:0,1:0,2:0,3:1,4:1,5:2}  # from stdout warnings; unknown originals default below
alloc={}   # pMemory -> dict(size, orig_type, new_type, file)
buffers={} # pBuffer -> dict(size, usage, mem, offset, file)
images_bound=collections.defaultdict(list) # pMemory -> [(image, offset)]
maps={}    # ppData -> pMemory
mapped_mem={} # pMemory -> ppData
submit_idx=0
# per submit-pair (frame) accumulators
frame_writes=collections.defaultdict(lambda: collections.defaultdict(int)) # frame -> ppData -> bytes
frame_write_calls=collections.defaultdict(int)
frame_copybuf=collections.defaultdict(list)
frame_copyb2i=collections.defaultdict(list)
frame_copyi2b=collections.defaultdict(list)
frame_createbuf=collections.defaultdict(list)
re_alloc=re.compile(r'VkMemoryAllocateInfo\s+(\w+)\s*\{\s*VkStructureType\(\d+\),\s*(\w+),\s*(\d+)UL,\s*RecalculateMemoryTypeIndex\((\d+)\),\s*\};\s*VK_CALL_CHECK\(vkAllocateMemory\(pDevice_0, &\1, nullptr, &(\w+)\)', re.S)
re_buf=re.compile(r'VkBufferCreateInfo\s+(\w+)\s*\{\s*VkStructureType\(12\),\s*NULL,\s*VkBufferCreateFlags\((\d+)\),\s*(\d+)UL,\s*VkBufferUsageFlags\((\d+)\),.*?VK_CALL_CHECK\(vkCreateBuffer\(pDevice_0, &\1, nullptr, &(\w+)\)', re.S)
re_bind=re.compile(r'vkBindBufferMemory\(pDevice_0, (\w+), (\w+), (\d+)UL\)')
re_bindimg=re.compile(r'vkBindImageMemory\(pDevice_0, (\w+), (\w+), (\d+)UL\)')
re_map=re.compile(r'vkMapMemory\(pDevice_0, (\w+), (\d+)UL, (\d+)UL, 0, &(\w+)\)')
re_load=re.compile(r'LoadBinaryData\("([^"]+)", (\d+), (\w+), (\d+), (\d+), appdata\)')
re_submit=re.compile(r'vkQueueSubmit\(')
re_cpb=re.compile(r'vkCmdCopyBuffer\((\w+)\[0\], (\w+), (\w+), 1, &(\w+)\)')
re_cpb2i=re.compile(r'vkCmdCopyBufferToImage\((\w+)\[0\], (\w+), (\w+), ')
re_cpi2b=re.compile(r'vkCmdCopyImageToBuffer\((\w+)\[0\], (\w+), \w+, (\w+), ')
re_bufcopy=re.compile(r'VkBufferCopy\s+(\w+)\s*\{\s*(\d+)UL,\s*(\d+)UL,\s*(\d+)UL,\s*\};')
# Descriptor buffer infos and vertex/index binds (to know consumption)
re_dbi=re.compile(r'VkDescriptorBufferInfo\s+\w+\s*\{\s*(\w+),\s*(\d+)UL,\s*(\d+)UL,\s*\};')
re_vb=re.compile(r'vkCmdBindVertexBuffers\(\w+\[0\], \d+, \d+, (\w+), (\w+)\)')
re_vbarr=re.compile(r'VkBuffer\s+(\w+)\[\]\s*=\s*\{([^}]*)\};')
re_ib=re.compile(r'vkCmdBindIndexBuffer\(\w+\[0\], (\w+), (\d+)UL')
desc_use=collections.Counter(); vb_use=collections.Counter(); ib_use=collections.Counter()
frame_desc_bufs=collections.defaultdict(set); frame_vb_bufs=collections.defaultdict(set); frame_ib_bufs=collections.defaultdict(set)
bufcopies={}
# process files in order, tokenizing by line for submits and loads to keep frame order
for f in files:
    txt=open(f).read()
    for m in re_alloc.finditer(txt):
        alloc[m.group(5)]=dict(size=int(m.group(3)), orig=int(m.group(4)), new=REMAP.get(int(m.group(4)),-1), file=os.path.basename(f))
    for m in re_buf.finditer(txt):
        buffers[m.group(5)]=dict(size=int(m.group(3)), usage=int(m.group(4)), flags=int(m.group(2)), file=os.path.basename(f), created_frame=submit_idx//2)
    for m in re_bufcopy.finditer(txt):
        bufcopies[m.group(1)]=(int(m.group(2)),int(m.group(3)),int(m.group(4)))
    for m in re_vbarr.finditer(txt):
        pass
    # line walk for ordered events
    for line in txt.splitlines():
        m=re_bind.search(line)
        if m:
            b=buffers.setdefault(m.group(1),dict(size=None,usage=None,flags=None,file='?',created_frame=None))
            b['mem']=m.group(2); b['offset']=int(m.group(3)); continue
        m=re_bindimg.search(line)
        if m: images_bound[m.group(2)].append((m.group(1),int(m.group(3)))); continue
        m=re_map.search(line)
        if m: maps[m.group(4)]=m.group(1); mapped_mem[m.group(1)]=m.group(4); continue
        m=re_load.search(line)
        if m:
            fr=submit_idx//2
            frame_writes[fr][m.group(3)]+=int(m.group(5)); frame_write_calls[fr]+=1; continue
        if re_submit.search(line):
            submit_idx+=1; continue
        m=re_cpb.search(line)
        if m: frame_copybuf[submit_idx//2].append((m.group(2),m.group(3),bufcopies.get(m.group(4)))); continue
        m=re_cpb2i.search(line)
        if m: frame_copyb2i[submit_idx//2].append((m.group(2),m.group(3))); continue
        m=re_cpi2b.search(line)
        if m: frame_copyi2b[submit_idx//2].append((m.group(2),m.group(3))); continue
        m=re_dbi.search(line)
        if m: desc_use[m.group(1)]+=1; frame_desc_bufs[submit_idx//2].add(m.group(1)); continue
        m=re_ib.search(line)
        if m: ib_use[m.group(1)]+=1; frame_ib_bufs[submit_idx//2].add(m.group(1)); continue
    for m in re_vbarr.finditer(txt):
        for b in re.findall(r'pBuffer_\d+', m.group(2)):
            vb_use[b]+=1
nframes=submit_idx//2
out={}
out['n_submits']=submit_idx; out['n_frames']=nframes
out['n_allocs']=len(alloc)
by_type=collections.Counter(); bytes_by_type=collections.Counter()
for k,v in alloc.items(): by_type[(v['orig'],v['new'])]+=1; bytes_by_type[(v['orig'],v['new'])]+=v['size']
out['allocs_by_type']={f"orig{o}->cudavk{n}":dict(count=c,bytes=bytes_by_type[(o,n)]) for (o,n),c in by_type.items()}
# host-visible allocations (new type 1 or 2)
hv=[k for k,v in alloc.items() if v['new'] in (1,2)]
out['host_visible_allocs']=[]
for k in hv:
    v=alloc[k]
    bl=[(b,d) for b,d in buffers.items() if d.get('mem')==k]
    out['host_visible_allocs'].append(dict(mem=k,size=v['size'],orig_type=v['orig'],cudavk_type=v['new'],mapped=k in mapped_mem, ppData=mapped_mem.get(k), n_buffers_bound=len(bl), usages=sorted(set(d['usage'] for b,d in bl if d['usage'] is not None)), n_images_bound=len(images_bound.get(k,[])), file=v['file']))
# per-frame writes: heavy band frames 2150..3150 and whole-run steady
def band_stats(lo,hi):
    tot=collections.Counter(); per=[]
    for fr in range(lo,hi):
        w=frame_writes.get(fr,{})
        per.append(sum(w.values()))
        for pd,b in w.items(): tot[pd]+=b
    n=hi-lo
    import statistics
    return dict(frames=n, mean_bytes_per_frame=sum(per)/n, median_bytes_per_frame=statistics.median(per), by_ppData={pd:dict(mem=maps.get(pd), bytes_per_frame=b/n, cudavk_type=alloc.get(maps.get(pd),{}).get('new')) for pd,b in tot.most_common()})
out['writes_heavy_band_2150_3150']=band_stats(2150,3150)
out['writes_band_1391_2150']=band_stats(1391,2150)
# per-frame counts of copy commands in heavy band
def cnt(d,lo,hi): return sum(len(d.get(fr,[])) for fr in range(lo,hi))/(hi-lo)
out['heavy_band_per_frame']=dict(copybuffer=cnt(frame_copybuf,2150,3150), copybuf2img=cnt(frame_copyb2i,2150,3150), copyimg2buf=cnt(frame_copyi2b,2150,3150), loadbinary_calls=sum(frame_write_calls.get(fr,0) for fr in range(2150,3150))/1000)
# which buffers are copy sources (staging) in heavy band and their memory types
srcs=collections.Counter(); dsts=collections.Counter()
for fr in range(2150,3150):
    for s,d,_ in frame_copybuf.get(fr,[]): srcs[(alloc.get(buffers.get(s,{}).get('mem'),{}).get('new'), alloc.get(buffers.get(d,{}).get('mem'),{}).get('new'))]+=1
    for s,i in frame_copyb2i.get(fr,[]): srcs[('b2i',alloc.get(buffers.get(s,{}).get('mem'),{}).get('new'))]+=1
    for i,d in frame_copyi2b.get(fr,[]): dsts[('i2b',alloc.get(buffers.get(d,{}).get('mem'),{}).get('new'),d)]+=1
out['heavy_band_copy_src_dst_types']={str(k):v for k,v in srcs.items()}
out['heavy_band_i2b_dst']={str(k):v for k,v in dsts.items()}
# consumption: buffers referenced by descriptors / vertex / index binds in heavy band, grouped by memory type
def cons(dd):
    c=collections.Counter()
    for fr in range(2150,3150):
        for b in dd.get(fr,set()):
            c[alloc.get(buffers.get(b,{}).get('mem'),{}).get('new')]+=1
    return dict(c)
out['heavy_band_distinct_buffers_per_frame_by_type']=dict(descriptor=cons(frame_desc_bufs), index=cons(frame_ib_bufs))
out['vb_use_by_type']=dict(collections.Counter(alloc.get(buffers.get(b,{}).get('mem'),{}).get('new') for b in vb_use.elements()))
out['desc_use_by_type']=dict(collections.Counter(alloc.get(buffers.get(b,{}).get('mem'),{}).get('new') for b in desc_use.elements()))
# host-visible buffers referenced by descriptors/vb/ib: list with sizes/usages
hvb=collections.defaultdict(list)
for b,d in buffers.items():
    t=alloc.get(d.get('mem'),{}).get('new')
    if t in (1,2):
        hvb[d['mem']].append(dict(buf=b,size=d['size'],usage=d['usage'],offset=d.get('offset'),desc_refs=desc_use.get(b,0),vb_refs=vb_use.get(b,0),ib_refs=ib_use.get(b,0)))
out['host_visible_buffers_by_mem']={k:v[:12]+([f'... {len(v)-12} more'] if len(v)>12 else []) for k,v in hvb.items()}
out['host_visible_buffer_counts_by_mem']={k:len(v) for k,v in hvb.items()}
json.dump(out,open(sys.argv[1],'w'),indent=1,default=str)
print(json.dumps({k:out[k] for k in ['n_submits','n_frames','n_allocs','allocs_by_type','heavy_band_per_frame','heavy_band_copy_src_dst_types','heavy_band_i2b_dst','heavy_band_distinct_buffers_per_frame_by_type','vb_use_by_type','desc_use_by_type','host_visible_buffer_counts_by_mem']},indent=1,default=str))
print("HOST VISIBLE ALLOCS:")
for a in out['host_visible_allocs']: print(a)
print("WRITES heavy band:")
w=out['writes_heavy_band_2150_3150']; print(w['frames'], w['mean_bytes_per_frame'], w['median_bytes_per_frame'])
for pd,v in w['by_ppData'].items(): print("  ",pd,v)
print("WRITES light band:")
w=out['writes_band_1391_2150']; print(w['frames'], w['mean_bytes_per_frame'], w['median_bytes_per_frame'])
for pd,v in w['by_ppData'].items(): print("  ",pd,v)
