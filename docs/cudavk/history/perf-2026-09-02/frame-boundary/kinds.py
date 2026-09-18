import re, os, pickle
src='/home/alexzhukov/favorite3-cpp/out/src'
files=sorted(f for f in os.listdir(src) if re.match(r'frame_0000_\d+\.cpp',f))
kinds=[]; state={}  # cb -> set of flags
arrdef=re.compile(r'VkCommandBuffer (command_buffers_array_\d+)\[\] = \{([^}]*)\};')
for fn in files:
    txt=open(os.path.join(src,fn)).read()
    arrays=dict((m.group(1),[c.strip() for c in m.group(2).split(',') if c.strip()]) for m in arrdef.finditer(txt))
    # sequential scan of relevant calls
    for m in re.finditer(r'(vkBeginCommandBuffer\((pCommandBuffers_\d+)\[0\]|vkCmdCopyImageToBuffer\((pCommandBuffers_\d+)\[0\], pImage_1,|vkCmdDraw\w*\((pCommandBuffers_\d+)\[0\]|vkCmdDispatch\w*\((pCommandBuffers_\d+)\[0\]|vkQueueSubmit\(pQueue_0, 1, &(submitInfo_\d+), (\w+)\))',txt):
        s=m.group(0)
        if s.startswith('vkBeginCommandBuffer'): state[m.group(2)]=set()
        elif s.startswith('vkCmdCopyImageToBuffer'): state.setdefault(m.group(3),set()).add('RB')
        elif s.startswith('vkCmdDraw'): state.setdefault(m.group(4),set()).add('DRAW')
        elif s.startswith('vkCmdDispatch'): state.setdefault(m.group(5),set()).add('DISP')
        else:
            si=m.group(6); fence=m.group(7)
            # find the submitInfo struct and its command buffer array name
            mm=re.search(re.escape(si)+r' \{(.*?)\};',txt[max(0,m.start()-3000):m.start()],re.S)
            cbs=[]
            if mm:
                arr=re.findall(r'command_buffers_array_\d+',mm.group(1))
                if arr: cbs=arrays.get(arr[0],[])
            flags=set()
            for cb in cbs:
                cbn=cb.split('[')[0]; flags|=state.get(cbn,set())
            k='B' if 'RB' in flags else ('A' if 'DRAW' in flags else 'U')
            kinds.append((k,fn,fence,tuple(cbs)))
print("total submits in files:",len(kinds))
import collections
print(collections.Counter(k[0] for k in kinds))
# window check: indices 2783.. should be A at odd
print("kinds 2778..2792:",[k[0] for k in kinds[2778:2792]])
print("kinds 4300..4312:",[k[0] for k in kinds[4300:4312]])
# find where the pattern deviates from strict A,B alternation after 2782
seq=[k[0] for k in kinds]
dev=[i for i in range(2782,len(seq)-1) if seq[i]==seq[i+1] or seq[i]=='U']
print("non-alternating positions after 2782:",dev[:40], len(dev))
for i in dev[:12]: print(i, kinds[i][:3], kinds[i+1][:3])
pickle.dump(kinds,open('/tmp/claude-1000/-home-alexzhukov-mesa/5f655967-d6a4-4c8c-bcdc-8af1e163dd93/scratchpad/frame-boundary/kinds.pkl','wb'))
