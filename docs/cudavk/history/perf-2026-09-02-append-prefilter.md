# Refusing un-appendable blended batches before the vertex work

`cp_pass_appendable()` admitted any blended batch whose A-buffer machinery was
ready, and left the rest of the eligibility question to
`cp_draw_execute_batch()` — which can only answer it once it has the post-clip
triangle count. A batch refused there has already run its vertex fetch, vertex
shader and clip **on a side stream**; `cp_pass_append()` then joins every side
stream, closes the episode, and re-executes the whole batch on the main
stream. The geometry is done twice and the main stream waits through the first
copy. The comment at that site called the case rare.

It was not rare. Every `cp_clip_triangles` launch in a post-pin heavy-band
trace — **9,400 of them, 9.3 per frame** — was on a side stream with a
matching main-stream re-execution behind it. Nothing in the driver reported
this: the census counters that would have shown it were the ones found to have
no callers (dead ends 24, 36).

Every input to that decision except the triangle count is known before any
work is issued. The refusal now happens in `cp_pass_appendable()`, mirroring
the A-buffer arm exactly: not peelable (blending off, or a discard draw with
the reject/resolved buffers and colour), multisampled, depth write mask set,
not exactly one colour attachment, or no colour encoding. The triangle-count
floor is deliberately not reproduced — `cp_pass_append()` already applies it
from the pre-clip count and it defaults off. A refusal lands the batch at
`cp_pass_finish()` + `cp_draw_execute_batch()` on the main stream, which is
exactly where the back-out put it, minus the duplicated geometry and the join.

**Mechanism, measured:** `cp_clip_triangles` launches in an equivalent trace
fall from 9,400 to **zero**. The path is gone, not merely cheaper.

**Measured**, three-round alternating A/B in one session per capture, RTX 5090,
control arm = `CUDAVK_NO_APPEND_PREFILTER=1`, every run exiting 0 with its
standard hash, full timestamp population and 18/18 sentinels:

| capture | window | control | prefilter | delta |
|---|---|---:|---:|---:|
| favorite3 | relevant (1391+) | 6.6881 | **6.4465** | **-0.2416** |
| favorite3 | heavy (2200-3150) | 8.2548 | 7.9827 | -0.2721 |
| favorite2 | relevant (1388+) | 5.4685 | **5.2330** | **-0.2355** |
| favorite2 | heavy (2735+) | 6.0162 | 5.7858 | -0.2304 |

Arms disjoint in every band. Gates: 79/79 suite, 18-sample sweep identical to
the accepted baseline, 0 sentinel mismatches across 12 A/B runs.

**Not yet measured on the B200** — that host was unreachable when this landed.
The mechanism is host-side scheduling, not architecture-dependent, so it should
carry; confirm when access returns.

**Provenance.** The finding is from the 2026-09-02 analysis note, which
identified it as lead A from the same traces this campaign had already taken
and nobody had read this way. Its predicted ceiling was 0.25-0.33 heavy and
0.20-0.28 light; measured 0.27 heavy and 0.24 whole-window, inside the range
on both captures.
