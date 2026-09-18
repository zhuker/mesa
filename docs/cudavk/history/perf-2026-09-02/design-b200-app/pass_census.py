import re,glob,sys,statistics as st
# Concatenate heavy-band harness files and census render passes per frame.
lo,hi=int(sys.argv[1]),int(sys.argv[2])
files=[f"/home/alexzhukov/favorite3-cpp/out/src/frame_0000_{i:04d}.cpp" for i in range(lo,hi+1)]
text="".join(open(f).read() for f in files)
# tokens of interest in order
tok=re.compile(r"(vkCmdBeginRenderPass\(|vkCmdEndRenderPass\(|vkCmdBindPipeline\([^,]+,\s*(VK_PIPELINE_BIND_POINT_\w+),\s*(pPipelines_\d+\[\d+\])|vkCmdDraw\w*\(|vkQueueSubmit\(|vkCmdBindDescriptorSets\(|vkCmdBindVertexBuffers\(|vkCmdBindIndexBuffer\(|vkCmdDispatch\(|vkCmdPipelineBarrier\(|pFramebuffer_(\d+),)")
frames=[]; cur=None; inpass=False; passinfo=None
for m in tok.finditer(text):
    s=m.group(1)
    if s.startswith("vkCmdBeginRenderPass"):
        inpass=True; passinfo={"draws":0,"binds":0,"pipes":[],"fb":None,"ds":0,"vb":0,"ib":0}
        if cur is None: cur={"passes":[],"dispatch":0,"barriers":0}
    elif s.startswith("vkCmdEndRenderPass"):
        inpass=False
        if cur is not None and passinfo: cur["passes"].append(passinfo)
    elif s.startswith("vkCmdBindPipeline"):
        if inpass and m.group(2)=="VK_PIPELINE_BIND_POINT_GRAPHICS":
            passinfo["binds"]+=1; passinfo["pipes"].append(m.group(3))
    elif s.startswith("vkCmdDraw"):
        if inpass: passinfo["draws"]+=1
    elif s.startswith("vkCmdBindDescriptorSets"):
        if inpass: passinfo["ds"]+=1
    elif s.startswith("vkCmdBindVertexBuffers"):
        if inpass: passinfo["vb"]+=1
    elif s.startswith("vkCmdBindIndexBuffer"):
        if inpass: passinfo["ib"]+=1
    elif s.startswith("vkCmdDispatch"):
        if cur is None: cur={"passes":[],"dispatch":0,"barriers":0}
        cur["dispatch"]+=1
    elif s.startswith("vkCmdPipelineBarrier"):
        if cur is not None: cur["barriers"]+=1
    elif s.startswith("pFramebuffer_"):
        if inpass and passinfo["fb"] is None: passinfo["fb"]="fb"+m.group(4)
    elif s.startswith("vkQueueSubmit"):
        if cur is not None and cur["passes"]:
            frames.append(cur)
        cur=None
def runs(p):
    r=0;prev=None
    for x in p:
        if x!=prev: r+=1
        prev=x
    return r
# skip first frame (partial)
frames=frames[1:-1]
print("frames",len(frames))
agg={"passes":[],"draws":[],"onedraw":[],"binds":[],"runs":[],"distinct":[],"maxpass_draws":[],"maxpass_binds":[],"maxpass_runs":[],"maxpass_distinct":[],"ds":[],"vb":[]}
fbseq={}
for fr in frames:
    ps=fr["passes"]
    agg["passes"].append(len(ps))
    agg["draws"].append(sum(p["draws"] for p in ps))
    agg["onedraw"].append(sum(1 for p in ps if p["draws"]==1))
    agg["binds"].append(sum(p["binds"] for p in ps))
    agg["runs"].append(sum(runs(p["pipes"]) for p in ps))
    agg["distinct"].append(sum(len(set(p["pipes"])) for p in ps))
    agg["ds"].append(sum(p["ds"] for p in ps)); agg["vb"].append(sum(p["vb"] for p in ps))
    big=max(ps,key=lambda p:p["draws"])
    agg["maxpass_draws"].append(big["draws"]); agg["maxpass_binds"].append(big["binds"])
    agg["maxpass_runs"].append(runs(big["pipes"])); agg["maxpass_distinct"].append(len(set(big["pipes"])))
    key=tuple(p["fb"] for p in ps); fbseq[key]=fbseq.get(key,0)+1
for k,v in agg.items():
    print(f"{k:18s} median {st.median(v):8.1f} mean {st.mean(v):8.1f} min {min(v)} max {max(v)}")
print("top framebuffer sequences:")
for k,v in sorted(fbseq.items(),key=lambda kv:-kv[1])[:5]: print(v,k)
# per-pass position stats for the most common sequence
common=max(fbseq,key=fbseq.get)
pos={}
for fr in frames:
    ps=fr["passes"]
    if tuple(p["fb"] for p in ps)!=common: continue
    for i,p in enumerate(ps):
        pos.setdefault(i,[]).append((p["draws"],p["binds"],runs(p["pipes"]),len(set(p["pipes"]))))
print("per-pass (draws, binds, runs, distinct) medians for the common sequence:")
for i in sorted(pos):
    cols=list(zip(*pos[i]))
    print(i,common[i],[st.median(c) for c in cols], "n",len(pos[i]))
