# Dead render-scope elimination — measured and accepted, 2026-08-30

## Result

The command-buffer planner suppresses the draws of a render scope when the next
full-area scope targets the same colour subresource with `LOAD_OP_DONT_CARE` or
`CLEAR`, and nothing can observe the old result. The HeadlessStreamer captures
record almost exactly one such scope per frame.

Compiled replay, one binary per arm, arms alternating in one locked session;
real frames begin at 1391:

| capture | control runs (ms) | candidate runs (ms) | median gain |
|---|---:|---:|---:|
| favorite3 | 7.8891 / 7.8951 | 7.5988 / 7.6102 | **0.2876 ms/frame** |
| favorite2 | 6.8725 / 6.8823 | 6.4748 / 6.5212 | **0.3794 ms/frame** |

Every favorite3 arm had 6,939 submits and stdout hash
`e3f24a1dcdc8568be217d249e480623958e2621b3d4f056ae4c44ad20d08da49`.
Every favorite2 arm had 6,965 and hash
`0240ff4ec576c62b49461a384d90e0c25463f69ecdaf1b2286e92d14264d947e`.
Raw run log: `/tmp/deadscope-ab.log`; timestamps: `/tmp/deadscope-ab/`.

The census admitted 3,470 scopes in a favorite3 replay. The common scope is 12
draws / 41 triangles. The generated Vulkan commands put the scopes adjacent;
Mesa's legacy-render-pass lowering inserts one internal barrier between their
recorded END/BEGIN ops, which was the only reason the first deliberately
narrow census admitted zero.

## Correctness

- native suite: **79/79** after the final production gate;
- 18-sample animated sweep: every candidate frame within the current
  `layered-a2d` baseline's frame-0 budget; the NVIDIA comparison reproduces the
  exact pre-existing `gltfscenerendering` deviation and missing
  `renderheadless`, with no new exception;
- favorite3: 18 sentinel dumps, candidate/control byte-identical;
- favorite2: 18 sentinel dumps, candidate/control byte-identical;
- complete replay stdout hashes identical as above.

Sweep record: `~/git/Vulkan/build/iter/deadscope-elim-sweep/`.
Dump record: `/tmp/deadscope-dumps/`.

## Safety boundary

Only draws are suppressed; scope markers, the intervening barrier and the old
attachment clear still execute. Admission requires:

- exact colour base/cookie/extent match and a full-area next scope;
- next load op is not LOAD;
- neither scope has depth;
- the old scope contains only draws and a clear of that same attachment;
- neither vertex nor fragment shader writes memory.

Resolve copies, queries, events, dispatches, unrelated clears and every other
operation refuse the optimization. This is dead-store elimination at recorded
scope granularity, not scope concurrency or shader fusion.
