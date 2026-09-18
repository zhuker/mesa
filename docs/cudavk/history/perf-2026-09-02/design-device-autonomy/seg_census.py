import sqlite3, collections, statistics, sys
db = sys.argv[1]
con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
cur = con.cursor()
names = {i: v for i, v in cur.execute('select id, value from StringIds')}
# main thread = most RUNTIME rows
tids = cur.execute('select globalTid, count(*) c from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by c desc').fetchall()
main = tids[0][0]; worker = tids[1][0]
print('tids', tids[:3])
rows = cur.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? order by start', (main,)).fetchall()
# frame boundaries: worker cuEventSynchronize returns
evs = cur.execute('select end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId in (select id from StringIds where value=\'cuEventSynchronize\') order by end', (worker,)).fetchall()
evs = [e[0] for e in evs]
# window: optional from argv
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 62
evs = [e for e in evs if lo <= e < hi]
nframes = (len(evs) - 1) / 2
print('frames', nframes)
blocking = {'cuStreamSynchronize', 'cuCtxSynchronize', 'cuMemcpyDtoH_v2', 'cuMemcpyHtoD_v2', 'cuEventSynchronize'}
issue = {'cuLaunchKernel', 'cuLaunchKernelEx', 'cuMemcpyHtoDAsync_v2', 'cuMemcpy2DAsync_v2', 'cuMemcpyDtoDAsync_v2', 'cuMemsetD32Async', 'cuMemsetD8Async', 'cuEventRecord', 'cuStreamWaitEvent', 'cuLaunchHostFunc', 'cuMemcpyDtoHAsync_v2', 'cuMemcpyAsync'}
segs = []
cur_seg = collections.Counter(); seg_api_ns = 0; seg_ops = 0
nseg_with = collections.Counter()
tot = collections.Counter(); tot_ns = collections.Counter()
for s, e, nid, cid in rows:
    if not (lo <= s < hi): continue
    n = names.get(nid, str(nid)).split('_v')[0] if names.get(nid,'').startswith('cuMemcpy') or names.get(nid,'').startswith('cuMemset') else names.get(nid, str(nid))
    tot[n] += 1; tot_ns[n] += e - s
    if names.get(nid) in blocking:
        segs.append((cur_seg, seg_api_ns, seg_ops))
        cur_seg = collections.Counter(); seg_api_ns = 0; seg_ops = 0
    else:
        cur_seg[n] += 1; seg_api_ns += e - s; seg_ops += 1
segs.append((cur_seg, seg_api_ns, seg_ops))
nonempty = [s for s in segs if s[2] > 0]
print(f'segments {len(segs)} ({len(segs)/nframes:.2f}/frame), non-empty {len(nonempty)} ({len(nonempty)/nframes:.2f}/frame)')
ops = sorted(s[2] for s in nonempty)
print('ops/segment p10 p50 p90 max', ops[len(ops)//10], ops[len(ops)//2], ops[len(ops)*9//10], ops[-1])
print('graph-unfriendly content per segment (share of non-empty segments containing it):')
for k in ['cuMemcpy2DAsync', 'cuLaunchHostFunc', 'cuMemcpyDtoDAsync', 'cuMemcpyDtoHAsync']:
    print(f'  {k}: {100*sum(1 for s in nonempty if s[0][k] > 0)/len(nonempty):.1f}%  count/frame {sum(s[0][k] for s in nonempty)/nframes:.2f}')
print('per-frame API counts and CUPTI host time (ms/frame):')
for n, c in tot.most_common(25):
    print(f'  {n:28s} {c/nframes:8.2f}/frame  {tot_ns[n]/1e6/nframes:7.3f} ms')
api_issue = sum(tot_ns[n] for n in tot if n in issue or n.startswith('cuMemcpy') and n not in ('cuMemcpyDtoH','cuMemcpyHtoD') or n.startswith('cuMemset'))
print(f'issue-class API host time (CUPTI) {api_issue/1e6/nframes:.3f} ms/frame')
# largest segments
big = sorted(nonempty, key=lambda s: -s[2])[:3]
for s in big: print('big seg', s[2], dict(s[0].most_common(8)))
