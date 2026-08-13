# instancing: the sample that was 62% idle

`PHASE_1A.md` ended by naming `instancing` the clearest lead in the set — 38%
GPU busy, the one sample the per-draw scratch rewind had not helped, and a
profile unlike anything else. This is what it turned out to be.

**27.02 → 7.08 ms a frame, 38% → 93% GPU busy, and the sweep 168.80 → 149.83.**

---

## The number

| | drawrewind | idsondevice | |
|---|---|---|---|
| instancing | 27.02 | **7.08** | **−73.8%** |
| gltfscenerendering | 16.15 | 15.11 | −6.4% |
| particlesystem | 50.10 | 52.24 | +4.3% |
| **total over the sweep** | **168.80** | **149.83** | **−11.2%** |

Nothing else in the set moved by more than 4.3%, and the two that did are both
on the nondeterministic list — `TESTING.md` records `gltfscenerendering` at
−5.4% and `particlesystem` at +4.0% between two frame counts of one build, so
neither is a result in either direction.

Correctness held. `cp_compare_frames.py` over sixty frames of every sample: the
verdict table is identical sample for sample, including both standing
regressions at the same frames and the same pixel counts —
`gltfscenerendering` 59,939 at frame 47, `texture3d` 3,090 at frame 24. The
only movement anywhere is `multisampling` +2 pixels and `particlesystem` +6,
both on the nondeterministic list, where two runs of one build already move by
more. `instancing` itself is bit-identical to the previous build on every frame
checked.

cudapipe was 1.38x faster than llvmpipe over the set and is now **1.55x**.

---

## What it was

One draw. The frame has fifteen, and the third is 1,474,560 triangles across
8,192 instances — **4,423,680 assembled vertices**, which is exactly the 17,280
block grid `PHASE_1A.md` recorded without being able to explain.

The vertex fetch kernel has been able to index the index buffer itself since
before this pass, but the condition on that path required `instance_count == 1`.
Anything instanced fell back to the host, which per frame:

- `MALLOC`ed a 35 MB table of per-vertex `(vertex, instance)` pairs and filled
  it with a triple-nested loop;
- built vertex-id and instance-id arrays out of that table **twice** — once for
  the fetch kernel and once for the vertex shader, from the same source, into
  **four** 17.7 MB buffers;
- and every one of those four was `cuMemAllocManaged` memory, so the host's
  write pulled the pages to the host and the kernel then faulted them back.

Five host passes over 4.4 million entries, ~71 MB written into managed memory,
and a 35 MB allocation and free, for a draw whose every byte of that is a
function of the thread index.

**An instanced draw replays one index range once per instance.** Vertex `v` is
index `v % verts_per_instance` of instance `v / verts_per_instance`. Two integer
divisions, per thread, on the device.

So the fetch kernel derives both, and publishes them only for a shader that
reads them — `reads_instance_id` joins the `reads_vertex_id` flag that was
already there. `instancing.vert` takes its per-instance data through divisored
attributes rather than `gl_InstanceIndex`, so for this draw all four arrays
disappear rather than moving to the device: the fetch kernel needs the instance
id to resolve the divisor and computes it in a register.

---

## The profile named the wrong kernel, and was not wrong

This is the part worth carrying.

Before:

| | share | median launch |
|---|---|---|
| `cp_vertex_fetch` | 58.7% | 2.48 ms |

That is what the measurement gate's second question is for, and the answer it
gives is `cp_vertex_fetch`. It is also the wrong place to look: gathering three
attributes for 4.4 million vertices is not a 2.48 ms job on a 5090, and no
amount of reading that kernel would have found anything wrong with it.

The cost was **charged** to it. A kernel that touches a managed page the host
has just written stalls the warp on a host round trip, and the stall is charged
to whichever kernel touches the page first — which was the fetch kernel,
because the arrays were built for it. After the change:

| | before | after |
|---|---|---|
| `cp_vertex_fetch` share | 58.7% | 36.1% |
| `cp_vertex_fetch` median launch | 2,477,190 ns | **93,567 ns** |
| unified memcpy host-to-device | 15,713 ops / 55.3 ms | **1,401 / 1.6 ms** |
| unified memcpy device-to-host | 10,347 ops / 35.3 ms | **747 / 1.1 ms** |

The two lines that actually named the cause are in the **memory operations**
table, which is neither of the questions `cp_profile.sh` was built around. The
kernel summary and the API summary between them could not have produced this
diagnosis; 26,060 page migrations per ten frames could, immediately.

`TESTING.md` already says "when the kernel breakdown does not add up to the
frame, read the CUDA API section". This adds a third place: **a kernel whose
duration is absurd for the work it describes is reporting somebody else's page
faults**, and the memory-operations table is where that shows up.

## Why the scratch rewind missed it

`PHASE_1A.md`'s largest win was rewinding the device scratch arena per draw,
and it moved `instancing` by −2.1% while moving `multithreading` by −36.4%.
That is now explained: **it rewinds the device arena, and these four arrays
were in the managed one**, which is deliberately not rewound per draw because
the host writes into it while the device is still reading the draw before.

The managed traffic was also enough to trip `CP_SCRATCH_RECLAIM_BYTES` several
times a frame, which drains the device and frees the overflow arenas — the
5.3 ms of `cuMemAlloc`/`cuMemFree` that pass had already identified in this
sample and attributed to the device arena alone.

---

## The shape, for the fifth time

`PERFORMANCE_PROGRESS.md` names the thing to go looking for as **work sized to
the worst case the host can compute rather than to what the draw does**. This
is the fifth instance, and it has a variant worth naming separately: here the
work was not merely mis-sized but **entirely redundant** — the host computed,
stored and transferred a per-vertex table whose every entry is an arithmetic
function of the entry's own index. It was then computed twice.

The tell was in the plan all along, in the sentence about `dynamicuniformbuffer`
being launch-bound: a per-draw cost that scales with the draw's *size* rather
than with anything the draw *does*. 4.4 million vertices is a large draw. It is
not a large amount of information.

---

## Two latent faults fixed on the way

Neither is reachable from the current sample set; both are on paths that could
not previously have the ids materialised correctly.

- **A 16-bit index buffer with a shader reading `gl_VertexIndex`** allocated a
  device array for the ids with a comment saying the fetch kernel would widen
  the indices into it. The fetch kernel did no such thing, so the shader read
  whatever the scratch arena last held.
- **A single-instance shader reading `gl_InstanceIndex`** was handed a NULL
  array, on the reasoning that every value would be zero — which is true and
  does not help, because the shader loads the value rather than assuming it.

Both fall out of the fetch kernel now publishing what it derives.

---

## What is left

| sample | ms | vs llvmpipe | |
|---|---|---|---|
| particlesystem | 52.24 | 3.3x slower | kernel-bound at 94%; Phase 3 |
| multithreading | 29.72 | **3.3x faster** | per-draw full-screen work |
| dynamicuniformbuffer | 16.96 | 16x slower | launch-bound; 83% busy |
| gltfscenerendering | 15.11 | comparable | |
| bloom | 12.03 | 3.4x slower | full-screen passes |
| **instancing** | **7.08** | **10x faster** | kernel-bound at 93% |

`instancing` is no longer a lead. Where its remaining frame goes, if it is
picked up again: `cp_clip_triangles` is now 28.8% of its GPU time with a median
launch of 4.2 µs and a maximum of 1.8 ms, and it sizes its output buffer at
three times the triangle count — 4.4 million triangles for that draw, whatever
the near plane actually cuts. That is the same shape again, and it is the
largest one left in this sample.

The samples above it are all already-known items: the peel loop and per-draw
full-screen work for the first two, and the launch count for the third.

---

## Reproducing this

```bash
cd ~/git/Vulkan
T=~/mesa/src/gallium/drivers/cudapipe/tests

DESC="..." $T/cp_iterate.sh mylabel idsondevice   # build, time, render, compare, record
$T/cp_gpu_busy.sh instancing 20                   # the gate, untraced
$T/cp_profile.sh instancing mylabel 10            # and read the memory ops table
```

**HEAD is what `idsondevice` measured.** The 149.83 ms above describes the
current build.
