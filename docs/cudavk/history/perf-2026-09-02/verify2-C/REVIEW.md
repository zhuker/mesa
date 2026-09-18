# Verify lead C — overlap opaque segment 0 with its own fan-out (kernel-chain L2)

Verdict: **OVERSTATED** (mechanism real and reproduced; light-band ceiling reproduced;
heavy-band ceiling is ~half the light-band figure, not larger; the proposed upload-span
fix is sound with two conditions; the whole lead sits below the 0.5 ms admission gate).

Scripts and raw output: `fanout.py` / `fanout2.py`, `postdead2.out`, `postpin2.out` in this
directory (read-only SQL over `/tmp/postdead/trace.sqlite` and `/tmp/postpin-rtx2/trace.sqlite`).

## 1. Mechanism check (MEASURED)

`cp_opaque_append` (cp_renderer.c:9339-9435): segment 0 runs on main; for k>=1
`cp_pass_gate_stream(k-1)` (7629-7642) does `cp_upload_flush` (main) + `cuEventRecord(pass_gate,
main)` + `cuStreamWaitEvent(side, gate)`, *after* segment 0's launches were issued. Trace
confirms the side burst begins after main's stage3 in every fan-out, in both bands:

| | postdead (light, 1391-1802, pre-pin) | postpin-rtx2 (heavy, 2450-3149, post-pin) |
|---|---|---|
| bursts with a seg-0 stage3 immediately in front | 822 / 822 | 1385 / 1385 |
| gate gap (main s3 end -> first side VS start), per frame | 21.4 med / 26.7 mean (2 bursts) | 22.0 / 28.8 |
| side segs whose host VS launch (+5 us) came after main s3 end | 2.9 % | 1.4 % |
| earliest side-VS host launch minus seg-0 VS start, per burst | -207 us median (p90 +14) | -1439 us median (p90 +12) |
| main resumes after join: device gap / host lead | 9.2 us / host launched 458 us earlier | 10.0 us / 1681 us earlier |

So the gate — not host issue — is what holds the side segments, and the region is
device-paced on both traces (host launches are hundreds of us to ~2 ms ahead; CUPTI's 1.28x
host inflation makes the un-profiled host further ahead still). A device-time removal at
this point converts ~1:1, as PERFORMANCE §5/§6 says for device-paced regions. The report's
"serial and on the critical path" claim holds.

## 2. Arithmetic (MEASURED from traces; three models)

Per frame, us. `naive` = min(seg0 chain, burst) as the report computed it; `shift` = the burst
moved up as a block by (chain + gate gap), capped at burst+gap; `cf` = each side segment
starts at max(host launch end + 5 us, seg-0 VS start, previous segment on the same stream),
keeping its measured wall (host times CUPTI-inflated, so `cf` is a lower bound in the light band).

| | light median | light mean | heavy median | heavy mean |
|---|---:|---:|---:|---:|
| fan-outs / side segments per frame | 2.00 / 13 | 2.00 / 12.1 | 2.00 / 18 | 1.98 / 17.6 |
| seg-0 chain per frame | 242 | 220 | **109** | 164 |
| side burst per frame | 551 | 588 | 739 | 753 |
| gain naive (report) | 242 | 218 | 109 | 163 |
| gain shift | 266 | 245 | 131 | 192 |
| gain cf | 192 | 214 | 193 | 215 |
| seg-0 chain per burst (median) | 111.5 | | **60.2** | |

Light band: the report's 0.20-0.24 ms is reproduced (0.19-0.27 across the three models).
Heavy band: the binding term is the seg-0 chain, and it is *shorter* there (60 us per burst
against 111), so the gain is **0.13-0.19 ms/frame median (0.19-0.22 mean)**, not larger. The
report's aside "heavy band has ~2.7x more side VS launches" does not translate into gain:
side segments are 17.6/frame against 12.1 (1.45x), and more side segments lengthen the
burst, which is not the term that limits the gain. The `cf` model exceeds `shift` in the
heavy band because it credits removal of intra-burst stagger (~40 us/frame between burst
wall and the longest per-stream chain) that the gate does not cause; treat `shift` as the
honest figure and `cf` as the upper edge.

Corrected ceiling, RTX 5090: **0.13-0.19 ms/frame heavy band (owner's band), 0.19-0.27
light band**; whole-window ~0.2. No double counting with L1 (failed appends are a
different population), L3, or the closure-audit geometry run-ahead (that lead covers
stream-13 VS/clip of *non-episode* batches; segment 0's VS is inside the episode, but a
run-ahead design that put segment 0 concurrently with earlier batches would subsume this
one — count once).

B200 (HYPOTHESIS, nothing measured): the chain is kernel latency; b200 REPORT §2.2/§2.4
gives B200/RTX active cycles 0.61-0.92 at a lower clock, implied kernel wall ratio
~1.0-1.57. Scaled: **0.15-0.30 ms/frame heavy band**. The report's "~0.35" has no basis.

## 3. Precedent

- DEAD_ENDS 32 (segment-0 *shading* overlap): different mechanism — speculative shading with
  41.67 % killed work. This lead speculates nothing and kills nothing; 32 does not apply.
- DEAD_ENDS 20-22 (merge rule): does not apply — no launch is removed or merged; the lead adds
  concurrency, which is the only mechanism family that has paid here (fan-out +2.73 ms,
  PDL). Confirmed.
- `b3f716bd5f5` (PERFORMANCE §4 row "cross-stream hand-off ~32 us/episode"): moved segment 0
  to main to *remove* a gate-in/join-out pair. This lead keeps segment 0 on main and adds no
  hand-off: the gate record moves from per-segment to once-per-episode (fewer
  `cuEventRecord`s: -11 to -17/frame, host-side only, ~0.02 ms, not collectable here since the
  host is ahead), and each segment's 11 KiB copy moves from main to its side stream (same
  op count). No priced cost is owed.
- `2eda3cf74c2` (the per-segment gate): the commit that *created* today's serialisation, to fix
  the multithreading sample losing models (~9,940 px intermittently; serialising segments did
  not fix it, one event record did, 0/20). Its diagnosis — a main-stream copy against a
  side-stream kernel — is exactly what the proposal must re-satisfy. See §4.
- DEAD_ENDS 38 (lanes): irrelevant to the mechanism; but note a 0.084-0.14 ms gain was
  declined there and 0.500 ms is the stated admission gate (DEAD_ENDS 36/38). This lead at
  0.13-0.27 is below that gate on both bands.
- DEAD_ENDS 24: the fan-out's value swings with segments/episode; this lead's value swings
  with the seg-0 chain length (capture-specific: 60-111 us per burst here).

## 4. Feasibility — the upload-span ordering

What is owed at the moment segment k is appended (cp_renderer.c:464-503, 622-628;
cpvk_cmd.c:2818-2870, 2985-3007; 9470-9519): `cp_batch_flush_defer_why` executes the batch
synchronously, and `cpvk_prepare_draw` for the *next* draw runs only after it returns
(cpvk_cmd.c:2995-3007). Hence at append time the owed span is exactly the rows of the draws
in batch k (push blocks reserved at prepare time) plus whatever cp_draw_execute_batch
reserves on the way; nothing for a later batch can be in it. The last flush point before
that was the previous segment's `cp_stream_set(saved_stream)` (on side k-1) or a main
launch, so the span's low edge is fresh.

Readers of those bytes: segment k's kernels (side k); after the episode, cp_opaque_finish's
shading on main, which is behind `cp_pass_join(nsegs-1)` (7850-7899, 7588-7605) — the join
records on each side stream after its last op, so a copy issued on side k before the
kernels is covered; a failed append joins all eight streams before re-executing on main
(9432-9440). Later main flushes send only later bytes (the watermark is global and
monotone: `upload_flushed = hi`, 464-503). The flush-generation retire events are never
recorded in the native ICD (`flush_retire` appears only at create/destroy, 106-116,
1044-1045) and the ring wrap drains the whole context (420-427), so no rewind can pass a
side-carried copy. **The side-stream flush is sound** — with these conditions:

1. `cp_stream_set` flushes on the stream it *leaves* (622-628). The proposal needs a switch
   that does *not* flush, followed by `cp_upload_flush` on the new stream — or, simpler,
   flush-after-switch inside the seg_side branch. Every other `cp_stream_set` caller must keep
   today's semantics (8769, 8798, 9007, 9113, 9308, 9316, 9430).
2. **`CUDAVK_NO_UPLOAD_COALESCE`** (cp_renderer.c:466, 647): with coalescing off the copies are
   issued at reserve time on `cp->stream` = main, before the switch. An episode-start gate does
   not cover them. Keep the per-segment gate under that flag (or require the gate to be
   per-segment whenever `no_upload_coalesce`), or the flag path regresses into the 2eda3cf race.
3. The episode-start gate must be recorded *after* the visbuf memset and the latched depth
   clear (9359-9380), i.e. restore the record that `2eda3cf74c2` removed, on the main stream
   before segment 0's first launch — and after a `cp_upload_flush`, so it also covers anything
   owed to segment 0 that a later segment might share (none today, but cheap).
4. Invariant 20 (ARCHITECTURE §8): ship with a `CUDAVK_NO_...` revert switch in the registry.
5. Narrowing of the margin: today every side segment is ordered behind *everything* main has
   issued so far; afterwards only behind the episode start. Nothing mid-episode issues on main
   today — within an episode only the four deferrable state changes are allowed (invariant 7,
   9470-9482), the depth clear is latched, and texture-cache rebuilds run on the calling
   stream (cpvk_texture_cache.c:322-340, 620) — but any future main-stream op between
   segments becomes a race. Worth a comment at the gate.
6. Related latent hazard, not introduced by this lead: `cp_pass_append` (9296-9316) already
   records its gate once at episode start (9288) and lets `cp_stream_set` flush segment k>=1's
   rows on main *after* the gate — the exact unordered pair `2eda3cf74c2` describes, surviving
   only because main is idle during blended episodes so the copy usually wins. The report
   holds this up as the model ("exactly as cp_pass_append does"); it should not be copied as
   is. HYPOTHESIS (from code, not observed); the side-stream flush would fix both.

Tests (TESTING.md): §6 says for synchronisation and pass-episode changes sentinels are
necessary and not sufficient — full-frame dump on both captures; plus the multithreading
sample repeated >= 20 times (the 0/20 protocol in `2eda3cf74c2`, `/tmp/perf16/streams/race.md`),
the 18 sentinels, the stdout hash, and both captures' timing in one session.

Hidden work: none on the device (same op count). Host: one fewer `cuEventRecord` per side
segment. Blast radius: `cp_opaque_append` only (~15 lines), plus the stream-switch helper.

## 5. Decisive probe

The report's timing-only flag (gate before segment 0, copy left on main) measures the ceiling
but deliberately reproduces the 2eda3cf race; a clean hash over three rounds does not prove
anything about an intermittent race and the number it gives is already known from the trace
(0.13-0.27). Better and no more expensive: implement the real form behind a registry flag
(gate once after the clears; switch-then-flush on the side stream; per-segment gate retained
under `no_upload_coalesce`), then (a) multithreading x20 image compare, (b) 3-round
alternating A/B on favorite3 heavy band and favorite2, hash + sentinels. Expected: -0.13 to
-0.19 heavy band. If it comes in at that size it is below the 0.5 gate and would land only as a
"small default with a revert switch" alongside the PDL/lane-class items.

## 6. Labels

MEASURED: everything in §1-§2 from the two traces. ESTIMATED: the ~1:1 conversion (from host
lead at the join). HYPOTHESIS: B200 figure; the cp_pass_append latent race.
