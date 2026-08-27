# P3 — the bound the clip-rectangle lead would have used

Probe run at HEAD `e2fea470d04`, both captures, one run each, full replays
(3,022 / 2,994 submits), stdout hashes unchanged. Instrumentation only, behind
`CUDAVK_DRAIN_PROBE=1`. Raw stderr `/tmp/perf-audit/stepD2/*/stderr`.

Sampled in the branch the driver reaches when the `bounded` predicate fails and
it drains for the segment counters (`cp_renderer.c:6420`) — the only place both
numbers exist at once: the host's product bound, and the device's actual quad
count. That is exactly the population §6 item 2 wanted to convert.

## The verdict, in one line

**§6 item 2's 0.10–0.25 ms is unsupported, and the `bounded` fast path is
effectively dead code on both captures: 0 admissions in 8,836 samples under
every variant tested.**

## 1. `bound/actual` is seven orders of magnitude — this is the finding

| ratio `nblocks × rast_num_triangles / actual quads` | old | Crossroads |
|---|---:|---:|
| samples (drains with non-zero quads) | 5,145 | 2,354 |
| min | 44 | 44 |
| p10 | 170,892 | 21,267,692 |
| **median** | **34,560,000** | **29,491,200** |
| p90 | 58,982,400 | 43,798,811 |
| max | 92,160,000 | 138,240,000 |

The two sides of that ratio:

| | old median | old p99 | Crossroads median | Crossroads p99 |
|---|---:|---:|---:|---:|
| `rast_num_triangles` | 400 | 19,200 | 4,800 | 19,200 |
| **actual quads** | **32** | 1,862 | **64** | 312 |
| framebuffer blocks | 230,400 | — | 230,400 | — |

**This is site 1's disease at draw scope, confirmed before anything was built.**
The product `blocks × triangles` over-estimates by ~3.5 × 10⁷ because the host
does not know post-transform triangle area until the vertex shader has run. It
cannot do better; no rectangle fixes an unknown area.

## 2. The clip rectangle is never tighter than the framebuffer

| | old | Crossroads |
|---|---:|---:|
| samples | 6,301 | 2,535 |
| clip rectangle tighter than the framebuffer | **0 (0.00%)** | **0 (0.00%)** |
| framebuffer blocks (min = median = max) | 230,400 | 230,400 |
| clip blocks (min = median = max) | 230,400 | 230,400 |
| clip / framebuffer block share | 1.000 everywhere | 1.000 everywhere |

So `clip_bound` and `fb_bound` are the same number at every sample, and the
`clip_bound/quads` column is identical to the `fb_bound/quads` column above.

**The rectangle I sampled is the right one, and it already carries the
scissor.** The patch author traced the chain at HEAD: `scissors` is non-NULL
only under `compact_rows` (`:4626`), the same guard puts `fs_ubo_table` behind
it (`:4621`), which sets `ndraws = batch_draws` (`:4651`), and `per_draw_rects`
needs `batch_draws > 1` (`:4808`) while `bounded` needs `ndraws <= 1`
(`:6448`). **`bounded` and `per_draw_rects` are mutually exclusive**, so
whenever the fast path can fire at all, the `:4813` scissor-intersect branch
has run and the batch-wide rectangle is the tightest one that exists there.

## 3. Admission is zero under every variant

Counted against the predicate's own tests: `composite && !verify && !timing`,
`ndraws <= 1`, an fs that writes no memory, `num_triangles × n <= capacity`,
`bound × 4 <= 512 << 10`, and `bound <= quad_capacity`.

| variant | old | Crossroads |
|---|---:|---:|
| today's predicate (framebuffer bound, `tris <= 2`) | **0** of 6,301 | **0** of 2,535 |
| clip-rectangle bound, `tris <= 2` kept | **0** | **0** |
| clip-rectangle bound, `tris <= 2` dropped | **0** | **0** |
| framebuffer bound, `tris <= 2` dropped | **0** | **0** |
| `tris <= 2` alone | 538 (8.54%) | 192 (7.57%) |
| the slot test alone (`bound × 4 <= 524,288`) | **0** | **0** |

**The slot test never passes.** The smallest bound in the whole sample is
230,400 × 16 = 3,686,400 quads, 28× over the 131,072-quad budget before any
tightening is attempted. Admission needs `blocks × tris <= 131,072` jointly; at
the measured medians that is ≤ 327 blocks on old (about 72 × 18 px) and ≤ 27
blocks on Crossroads (about 54 × 2 px). That is a scanline, not a scissor. The
`quad_capacity` test fails independently: 230,400 × 400 = 92.2 M against a
5–32 M capacity.

**Raising the `512u << 10` budget remains withdrawn** — it caps over-allocation
cost, and the ratio above is exactly what it caps.

If anyone reopens this, the whole test is the joint distribution of
(clip blocks, `rast_num_triangles`) against `blocks × tris <= 131,072`, which
this probe already carries. It needs no new run.

## 4. The empty drains — a fact about the captures, not a lead

`quads == 0` at the tail drain: the episode covered nothing.

| | old | Crossroads |
|---|---:|---:|
| empty drains | 1033 of 14932 (**6.9%**) | 1528 of 8333 (**18.3%**) |
| per frame | 0.68 of 9.88 | 1.02 of 5.57 |
| their share of this site's blocked time | 0.9% | **15.1%** |
| ms/frame spent in them | 0.0530 | **0.3216** |
| mean empty wait | 0.0775 ms | 0.3151 ms |
| mean productive wait | 0.6505 ms | 0.3972 ms |

On Crossroads **an empty drain costs almost as much as a productive one**
(0.315 against 0.397 ms) and empty drains hold **15.1% of that
capture's time at this site**.

**This is closed as a lead, on mechanism.** The drain is taken for `fill_over`
and `quad_over`; `quads` only rides along in the same copy, and
`cp_pass_can_retry` forbids running the fragment shader before the overflow
answer arrives. An oracle for `quads == 0` would therefore save no wait. The
numbers stay recorded because a mechanism that drains at the *scan* instead —
the addendum's S1d — is the one thing that would attack them.
