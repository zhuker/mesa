# Favorite3 generated-VS content concentration

Status: **parked pending B200/sm_100**, not a production proposal.

The diagnostic implementation and full design record are on
`diag/shader-content-census` at `90699a20fbf` (implementation
`8c32c74e0d0`). It keys successful generated vertex launches by the immutable
complete 256-bit final resource-aware compiled-stage cache key. Records use
absolute command-bearing submit and executed-batch identities. The clean-trace
join uses immediate same-stream launch grammar and `correlationId`; diagnostic
run time is not a performance measurement.

Two favorite3 captures in `/tmp/shader-census-f3` each passed the expected
post-output `rc=139`, 6,939 complete shim rows, internal submit 6,947, the
standard stdout hash, and all 18 sentinels. Their 210,944 records were exactly
equal, with zero drops, missing keys, key mismatches, unknown streams, or
failed synchronization. The window in `/tmp/postdead/trace.sqlite` maps all
16,842 fused VS launches across 411 frames and 22 full keys.

| clean fused-VS pool | ms/frame |
|---|---:|
| summed duration | 0.924079 |
| union | 0.626962 |
| **union-exclusive** | **0.520663** |

Full-key cost is not concentrated. The largest identity owns only
**0.209405 ms/frame** union-exclusive. The best possible five-key set owns
**0.376335**. An exact exhaustive subset analysis preserves cross-key overlap:
no set of 14 or fewer of the 22 identities reaches 0.500 ms/frame. The best
minimum-size set requires 15 identities and reaches only **0.502497**.

The clean cost JSON is `/tmp/shader-census-cost.json`, SHA-256
`67db0f66cb7904989966cccd24f97de76989f8c2239fef6be6b920c7976493f8`.
The exact `2^22` set result is `/tmp/shader-census-set-gate.json`, SHA-256
`297c0b7016e5322a0c00781a89b7970f2e75c5a9beb12b0badb33b083f10d6a9`.

This trace is RTX 5090/GB202/sm_120 evidence. It does not answer the primary
B200/sm_100 question. Do not build a per-content optimizer, fused/classic
selector, register-cap or block-size rule, PTX change, or NIR change from this
result. First repeat the full-key census and clean union-exclusive join on
B200. Reopen mechanism design only if B200 independently yields an actionable
identity set above the 0.500-ms/frame admission gate. Favorite2 was not spent
because favorite3 did not pass the architecture and concentration gates.
