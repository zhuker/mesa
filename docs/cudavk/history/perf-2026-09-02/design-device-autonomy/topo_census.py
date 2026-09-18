import sqlite3, collections, sys
db = sys.argv[1]
con = sqlite3.connect(f'file:{db}?mode=ro', uri=True); cur = con.cursor()
names = {i: v for i, v in cur.execute('select id, value from StringIds')}
tids = cur.execute('select globalTid, count(*) c from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by c desc').fetchall()
main, worker = tids[0][0], tids[1][0]
evs = [e for (e,) in cur.execute("select end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId in (select id from StringIds where value='cuEventSynchronize') order by end", (worker,))]
frame_end = evs[1::2]  # every other submit completion = frame boundary (approx; A/B classification not needed for recurrence)
# kernel rows by correlation for names/grid
kern = {}
for cid, sn, gx, gy, gz, bx in cur.execute('select correlationId, shortName, gridX, gridY, gridZ, blockX from CUPTI_ACTIVITY_KIND_KERNEL'):
    kern[cid] = (names.get(sn, '?'), gx*gy*gz, bx)
mset = {cid: b for cid, b in cur.execute('select correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMSET')}
mcpy = {cid: (k, b) for cid, k, b in cur.execute('select correlationId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY')}
blocking = {'cuStreamSynchronize', 'cuCtxSynchronize', 'cuMemcpyDtoH_v2', 'cuMemcpyHtoD_v2'}
rows = cur.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? order by start', (main,)).fetchall()
# build segments with op-type signature; frame index by end time
import bisect
frames = collections.defaultdict(list)  # frame -> list of (sig, ops)
sig = []; ops = []
for s, e, nid, cid in rows:
    n = names.get(nid, '?')
    if n in blocking:
        if sig:
            f = bisect.bisect_left(frame_end, s)
            frames[f].append((tuple(sig), ops))
        sig = []; ops = []
        continue
    if n.startswith('cuLaunchKernel'):
        k = kern.get(cid); sig.append('K:' + (k[0] if k else '?')); ops.append(('K', cid))
    elif n.startswith('cuMemsetD'):
        sig.append('S'); ops.append(('S', cid))
    elif n.startswith('cuMemcpyHtoDAsync'):
        sig.append('M1'); ops.append(('M', cid))
    elif n.startswith('cuMemcpy2DAsync') or n.startswith('cuMemcpy3DAsync'):
        sig.append('M2'); ops.append(('M2', cid))
    elif n.startswith('cuMemcpyDtoDAsync'):
        sig.append('MD'); ops.append(('MD', cid))
    elif n == 'cuEventRecord': sig.append('E')
    elif n == 'cuStreamWaitEvent': sig.append('W')
    elif n == 'cuLaunchHostFunc': sig.append('H'); ops.append(('H', cid))
    # ctx push/pop, create/destroy ignored
fr = sorted(frames)
# heavy band: skip first 2 frames
fr = fr[2:-1]
segs = [(f, i, s, o) for f in fr for i, (s, o) in enumerate(frames[f])]
nonempty = [x for x in segs if len(x[2]) > 0]
sigcount = collections.Counter(x[2] for x in nonempty)
print(f'frames {len(fr)}; non-empty segments {len(nonempty)} ({len(nonempty)/len(fr):.2f}/frame); distinct op-type signatures {len(sigcount)}')
# previous-frame recurrence and same-position
prev_seen = 0; same_pos = 0; pos_tot = 0
for f, i, s, o in nonempty:
    pf = f - 1
    if pf in frames:
        ps = [x[0] for x in frames[pf] if len(x[0]) > 0]
        if s in ps: prev_seen += 1
        pall = [x[0] for x in frames[pf]]
        pos_tot += 1
        if i < len(pall) and pall[i] == s: same_pos += 1
print(f'seen in previous frame {100*prev_seen/len(nonempty):.1f}%; same-position match {100*same_pos/max(pos_tot,1):.1f}%')
top = sigcount.most_common(20)
cover = sum(v for _, v in top) / len(nonempty)
print(f'top-20 signatures cover {100*cover:.1f}% of segments; top-50 {100*sum(v for _,v in sigcount.most_common(50))/len(nonempty):.1f}%')
# ops covered by signatures seen >= 2 times in whole window
recur_ops = sum(len(o) for f,i,s,o in nonempty if sigcount[s] >= 2); all_ops = sum(len(o) for x in nonempty)
print(f'device ops in recurring (>=2) signatures: {100*recur_ops/all_ops:.1f}% of {all_ops/len(fr):.0f} ops/frame')
# per-frame distinct signatures & nodes
print('top signatures (len, count, K/M1/S/M2 composition):')
for s, c in top[:12]:
    comp = collections.Counter(t.split(':')[0] for t in s)
    print(f'  len {len(s):4d} x{c:5d}  {dict(comp)}  first kernels: {[t[2:] for t in s if t.startswith("K:")][:6]}')
# parameter variability inside the most common long signature: grid sizes of kernel nodes across instances
for s, c in top:
    if len(s) >= 30 and c >= 200:
        inst = [o for f,i,ss,o in nonempty if ss == s]
        varying = 0; total_k = 0; mset_vary = 0; mset_tot = 0; m1_vary=0; m1_tot=0
        n = len(s)
        kpos = [j for j, t in enumerate([t for t in s if not t in ('E','W')]) ]
        # align by op index (ops excludes E/W)
        for j in range(len(inst[0])):
            typ = inst[0][j][0]
            if typ == 'K':
                grids = set(kern[o[j][1]][1] for o in inst if o[j][1] in kern)
                fns = set(kern[o[j][1]][0] for o in inst if o[j][1] in kern)
                total_k += 1; varying += (len(grids) > 1)
            elif typ == 'S':
                sizes = set(mset.get(o[j][1]) for o in inst); mset_tot += 1; mset_vary += (len(sizes) > 1)
            elif typ == 'M':
                sizes = set(mcpy.get(o[j][1], (0,0))[1] for o in inst); m1_tot += 1; m1_vary += (len(sizes) > 1)
        print(f'signature len {n} x{c}: kernel nodes {total_k}, grid varies across instances in {varying}; memset nodes {mset_tot}, size varies in {mset_vary}; HtoD nodes {m1_tot}, size varies in {m1_vary}')
        break
