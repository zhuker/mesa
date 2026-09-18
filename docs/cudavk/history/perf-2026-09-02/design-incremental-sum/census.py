#!/usr/bin/env python3
"""Read-only heavy-band census on /tmp/postpin-rtx2/trace.sqlite (post host-pin, RTX 5090,
nsys 2026.4.1) to size incremental leads in the heavy band.  Window: external submits
[4900, 6300) = frames 2450..3149 (700 frames), same as the memory-path report.
Everything per frame is a MEAN unless marked median."""
import sqlite3, sys, statistics as st, bisect, json

DB = '/tmp/postpin-rtx2/trace.sqlite'
TS = '/tmp/postpin-rtx2/ts.txt'
S0, S1 = 4900, 6300
c = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
sess = c.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
ts = [int(l.split()[0]) - sess for l in open(TS)]
w0, w1 = ts[S0], ts[S1]
nframes = (S1 - S0) / 2.0
print("window ns", w0, w1, "frames", nframes)

names = dict(c.execute("select id, value from StringIds"))
K = c.execute("select start,end,streamId,shortName,gridX*gridY*gridZ,blockX*blockY*blockZ,correlationId "
              "from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start", (w0, w1)).fetchall()
K = [(s, e, st_, names[n], g, b, cid) for s, e, st_, n, g, b, cid in K]
MC = c.execute("select start,end,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and copyKind not in (11,12) order by start", (w0, w1)).fetchall()
MS = c.execute("select start,end,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<? order by start", (w0, w1)).fetchall()
RT = c.execute("select r.start,r.end,s.value,r.globalTid,r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.start<? order by r.start", (w0, w1)).fetchall()
print("kernels", len(K), "per frame", len(K) / nframes, "memcpy", len(MC) / nframes, "memset", len(MS) / nframes, "api", len(RT) / nframes)

# submit kinds: B if a 3,686,400-byte copy starts within 60 us after the ts
rb = sorted(s for s, e, stm, k, b in MC if b == 3686400)
def is_B(i):
    j = bisect.bisect_left(rb, ts[i])
    return j < len(rb) and rb[j] - ts[i] < 60_000
kinds = ['B' if is_B(i) else 'A' for i in range(S0, S1)]
print("kinds A/B", kinds.count('A'), kinds.count('B'), "first", kinds[:6])

# per-stream kernel lists
main = [k for k in K if k[2] == 13]
side = [k for k in K if k[2] != 13 and k[3] != 'memcpy']
main_starts = [k[0] for k in main]
side_by_stream = {}
for k in side:
    side_by_stream.setdefault(k[2], []).append(k)

def union_len(iv):
    iv = sorted(iv)
    tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot

# all device ops on main (kernels + memcpy + memset on stream 13) for idle computation
main_ops = sorted([(k[0], k[1]) for k in main] + [(s, e) for s, e, stm, ck, b in MC if stm == 13] + [(s, e) for s, e, stm, b in MS if stm == 13])
main_op_starts = [s for s, e in main_ops]
def main_idle(a, b):
    """idle time on stream 13 inside [a,b)"""
    i = bisect.bisect_left(main_op_starts, a)
    # include the op straddling a
    if i > 0 and main_ops[i - 1][1] > a: i -= 1
    busy = []
    while i < len(main_ops) and main_ops[i][0] < b:
        s, e = main_ops[i]
        busy.append((max(s, a), min(e, b)))
        i += 1
    return (b - a) - union_len([x for x in busy if x[1] > x[0]])

# ---------- (a) failed appends: cp_clip_triangles on side streams
ct = [k for k in K if k[3] == 'cp_clip_triangles']
ct_side = [k for k in ct if k[2] != 13]
print("\n(a) cp_clip_triangles/frame", len(ct) / nframes, "on side streams", len(ct_side) / nframes)
exc = []  # (side wall, main idle inside window, dup VS)
for k in ct_side:
    lst = side_by_stream[k[2]]
    idx = lst.index(k)
    # previous kernel on same side stream = side VS
    prev = lst[idx - 1] if idx > 0 and lst[idx - 1][3] == 'main' else None
    start = prev[0] if prev else k[0]
    # next main-stream kernel start after clip_triangles end
    j = bisect.bisect_right(main_starts, k[1])
    if j >= len(main): continue
    nxt = main[j]
    # the re-executed VS is the next main 'main' kernel; its duration
    dupvs = nxt[1] - nxt[0] if nxt[3] == 'main' else 0
    exc.append((k[1] - start, main_idle(start, nxt[0]), dupvs, prev[1] - prev[0] if prev else 0))
if exc:
    print("  side excursion wall us mean/median", sum(x[0] for x in exc) / len(exc) / 1e3, st.median(x[0] for x in exc) / 1e3)
    print("  main idle inside excursion->re-exec, us mean/median", sum(x[1] for x in exc) / len(exc) / 1e3, st.median(x[1] for x in exc) / 1e3)
    print("  main idle ms/frame", sum(x[1] for x in exc) / nframes / 1e6)
    print("  re-executed VS us mean", sum(x[2] for x in exc) / len(exc) / 1e3, " side VS us mean", sum(x[3] for x in exc) / len(exc) / 1e3)

# ---------- (b) unfused stage1 re-passes (discard retry vs peel)
s1 = [i for i, k in enumerate(main) if k[3] == 'cp_rasterize_stage1']
disc = peelp = 0; disc_wall = []; disc_vis = {}
for i in s1:
    # look ahead on main for cp_peel_advance before the next cp_rasterize_stage1/main VS
    j = i + 1; kind = 'disc'
    while j < len(main) and j < i + 14:
        nm = main[j][3]
        if nm == 'cp_peel_advance': kind = 'peel'; break
        if nm in ('cp_rasterize_stage1', 'cp_clip_rast_fused', 'cp_clip_rast_fused_abuf'): break
        j += 1
    if kind == 'peel': peelp += 1
    else:
        disc += 1
        # pass wall: from the preceding memset (visbuf clear) to writeback end
        wb = next((main[q] for q in range(i, min(i + 12, len(main))) if main[q][3] == 'cp_fs_writeback'), None)
        # preceding memset on stream 13 before stage1 start
        m = bisect.bisect_left([s for s, e, stm, b in MS], main[i][0]) - 1
        while m >= 0 and MS[m][2] != 13: m -= 1
        if wb and m >= 0:
            disc_wall.append(wb[1] - MS[m][0])
            disc_vis[MS[m][3]] = disc_vis.get(MS[m][3], 0) + 1
print("\n(b) unfused stage1/frame", len(s1) / nframes, " discard re-passes/frame", disc / nframes, " peel re-passes/frame", peelp / nframes)
if disc_wall:
    print("  discard re-pass wall us mean/median", sum(disc_wall) / len(disc_wall) / 1e3, st.median(disc_wall) / 1e3, " ms/frame", sum(disc_wall) / nframes / 1e6)
    print("  preceding memset bytes histogram", sorted(disc_vis.items(), key=lambda x: -x[1])[:6])

# ---------- (c) opaque fan-outs: side-stream cp_clip_rast_fused (non-abuf)
side_fused = [k for k in side if k[3] == 'cp_clip_rast_fused']
print("\n(c) side-stream cp_clip_rast_fused/frame (opaque side segments)", len(side_fused) / nframes)
# group into fan-outs: consecutive side_fused launches with the same preceding main-stream stage3
fanouts = {}
for k in side_fused:
    j = bisect.bisect_right(main_starts, k[0]) - 1
    # walk back to the last main stage3 before this side launch
    while j >= 0 and main[j][3] != 'cp_rasterize_stage3': j -= 1
    if j < 0: continue
    fanouts.setdefault(j, []).append(k)
gains = []; seg0s = []; bursts = []
for j, segs in fanouts.items():
    s3 = main[j]
    # seg-0 chain: walk back from s3 to the main VS ('main' followed by fused)
    q = j
    while q > 0 and main[q][3] != 'main': q -= 1
    seg0 = s3[1] - main[q][0]
    # burst = union of all side-stream kernels between s3 end and next main kernel after last side op
    last_side_end = max(x[1] for x in segs)
    jj = bisect.bisect_right(main_starts, last_side_end)
    burst_ops = [(x[0], x[1]) for x in side if x[0] >= s3[1] - 20_000 and x[0] < (main[jj][0] if jj < len(main) else last_side_end)]
    burst = union_len(burst_ops)
    seg0s.append(seg0); bursts.append(burst); gains.append(min(seg0, burst))
print("  fan-outs/frame", len(fanouts) / nframes, " segments/fan-out mean", (sum(len(v) for v in fanouts.values()) / max(1, len(fanouts))))
if gains:
    print("  seg0 chain us mean/median", sum(seg0s) / len(seg0s) / 1e3, st.median(seg0s) / 1e3)
    print("  side burst us mean/median", sum(bursts) / len(bursts) / 1e3, st.median(bursts) / 1e3)
    print("  min(seg0,burst) ms/frame", sum(gains) / nframes / 1e6, " (median-frame estimate uses per-frame sums below)")

# ---------- (d) fullscreen-like chains: main VS 1 block -> fused 1 block, s3 > 8 us
fs_like = 0; fs_s3 = []
for i, k in enumerate(main):
    if k[3] == 'main' and k[4] == 1 and i + 3 < len(main) and main[i + 1][3] == 'cp_clip_rast_fused' and main[i + 1][4] == 1 and main[i + 3][3] == 'cp_rasterize_stage3':
        d = main[i + 3][1] - main[i + 3][0]
        if d > 8000: fs_like += 1; fs_s3.append(d)
print("\n(d) fullscreen-like chains/frame", fs_like / nframes, " s3 us median", st.median(fs_s3) / 1e3 if fs_s3 else None, " s3 ms/frame", sum(fs_s3) / nframes / 1e6)
# stage3 launches > 25 us overall
s3all = [k[1] - k[0] for k in main if k[3] == 'cp_rasterize_stage3']
print("  main stage3 launches/frame", len(s3all) / nframes, " >25us/frame", sum(1 for d in s3all if d > 25000) / nframes, " their ms/frame", sum(d for d in s3all if d > 25000) / nframes / 1e6)

# ---------- (e) host funcs
hf = [r for r in RT if r[2] == 'cuLaunchHostFunc']
print("\n(e) cuLaunchHostFunc/frame", len(hf) / nframes)

# ---------- (f) frame head: render submit -> first kernel; API calls inside
heads = []; head_api = {}
for i in range(S0, S1):
    if kinds[i - S0] != 'A': continue
    t = ts[i]
    j = bisect.bisect_left([k[0] for k in K], t)
    if j >= len(K): continue
    fk = K[j][0]
    heads.append(fk - t)
    for r in RT:
        pass
print("(f) render submit -> first kernel us mean/median", sum(heads) / len(heads) / 1e3, st.median(heads) / 1e3)
# API rows between render submit ts and first kernel, by name (sample first 100 frames for speed)
rt_starts = [r[0] for r in RT]
cnt = {}
nA = 0
for i in range(S0, S1):
    if kinds[i - S0] != 'A': continue
    t = ts[i]
    j = bisect.bisect_left([k[0] for k in K], t)
    if j >= len(K): continue
    fk = K[j][0]
    a = bisect.bisect_left(rt_starts, t); b = bisect.bisect_left(rt_starts, fk)
    nA += 1
    for r in RT[a:b]:
        cnt[r[2]] = cnt.get(r[2], 0) + 1
print("  API rows in the head per render submit:", {k: round(v / nA, 2) for k, v in sorted(cnt.items(), key=lambda x: -x[1])})

# ---------- (g) opaque shading groups: compact launches whose previous main kernel is fs_writeback
grp_extra = 0; compact_n = 0
for i, k in enumerate(main):
    if k[3] == 'cp_fs_compact':
        compact_n += 1
        if i > 0 and main[i - 1][3] == 'cp_fs_writeback': grp_extra += 1
print("\n(g) cp_fs_compact/frame", compact_n / nframes, " compact directly after writeback (extra shading group or re-pass w/o raster)/frame", grp_extra / nframes)
# FS direct duration
fsd = [k[1] - k[0] for i, k in enumerate(main) if k[3] == 'main' and i > 0 and main[i - 1][3] == 'cp_fs_compact']
cmp = [k[1] - k[0] for k in main if k[3] == 'cp_fs_compact']
wbk = [k[1] - k[0] for k in main if k[3] == 'cp_fs_writeback']
print("  FS direct us median", st.median(fsd) / 1e3, " compact median", st.median(cmp) / 1e3, " writeback median", st.median(wbk) / 1e3)

# ---------- (h) A-buffer episodes
print("\n(h) scan_reduce/frame", sum(1 for k in K if k[3] == 'cp_abuf_scan_reduce') / nframes, " composite/frame", sum(1 for k in K if k[3] == 'cp_abuf_composite') / nframes)

# ---------- (i) per-frame kernel/API counts for the whole trace band split
# ---------- (j) main-stream union, side union, main/side intersection
mainiv = [(k[0], k[1]) for k in main]; sideiv = [(k[0], k[1]) for k in side]
um, us_, ua = union_len(mainiv), union_len(sideiv), union_len(mainiv + sideiv)
print("\n(j) main kernel union ms/frame", um / nframes / 1e6, " side union", us_ / nframes / 1e6, " intersection", (um + us_ - ua) / nframes / 1e6)
