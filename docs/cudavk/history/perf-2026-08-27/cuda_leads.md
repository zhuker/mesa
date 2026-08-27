# CUDA-side leads in `src/cudavk`, audited against the code

Audit date 2026-08-26. Tree `/home/alexzhukov/mesa`, branch `cudapipe-vk-native`,
HEAD `e2fea470d04`. Toolkit CUDA 12.8.1 (`nvcc`/`ptxas` V12.8.93) at
`/usr/local/cuda`, driver 580.173.02, GPU RTX 5090 (sm_120).

Source of the five items: `docs/cudavk/notes/CUDA13_UPGRADE.md`, referenced from
`docs/cudavk/TODO.md:443-446`.

**No GPU work was run for this audit.** Everything below marked "verified here"
was produced with host-only tools: NVRTC 12.8 (`libnvrtc`, no CUDA context),
`ptxas` 12.8, `nvdisasm`, header reads, and `stat` on the on-disk compute cache.
Probe sources are in `/tmp/perf-audit/probe/`.

**Baseline correction taken from the parent:** commit `20611f5b131` ("cudavk: fan
opaque episode segments out by default") turned the opaque episode-segment stream
fan-out on. Old capture went 15.77 -> 13.23 ms. So the frame is ~13.2 ms, not the
15.75/15.99 ms that `docs/cudavk/PERFORMANCE.md` section 5 still quotes, and part
of the per-frame serialisation is already overlapped across `cp->seg_streams[]`.
Section 5 of this document says exactly which links the fan-out already covers.

---

## Naming note before anything else

The note says the JIT options live in `link_shader_module()`. There is no such
function. The real one is:

    src/cudavk/nir_to_ptx/cp_nir_to_llvm.c:3778   load_shader_module()

and it is the only place in the tree that passes `CUjit_option`s.

---

## 1. `CU_JIT_SPLIT_COMPILE` (option 34)

### What the note claims, and how

`CUDA13_UPGRADE.md:44-52`. Cold JIT of "a real 531 KB cudapipe PTX module" with
the JIT cache disabled: 536 ms with the option off, 168 ms at 16 threads,
178 ms at 0 (auto). Measured out of tree in `~/cuda13-research/03_compile_codegen.md`,
not reproducible from anything in this repository. The note is explicit about the
limit it found itself: **with the cache at its default the second load is ~1.1 ms
either way**, so this only pays on a genuine first compile, and each distinct JIT
option set is its own cache entry.

### Call sites that would change

| site | what it loads | takes JIT options today? |
|---|---|---|
| `nir_to_ptx/cp_nir_to_llvm.c:3802` `cuModuleLoadDataEx` | whole-program shader PTX (no sampler, no fs helper) | yes |
| `nir_to_ptx/cp_nir_to_llvm.c:3812` `cuLinkCreate` | shader + `cp_sampler.ptx` + `cp_fs_helper.ptx` | yes |
| `cp_kernels.c:307` `cuModuleLoadData` inside `build_module()` | the four `.cu` modules | **no — plain `cuModuleLoadData`, zero options** |

The option array is declared `CUjit_option jit_opts[5]` / `void *jit_vals[5]` at
`cp_nir_to_llvm.c:3782-3783` and is already full at 5 entries when
`max_regs > 0`. Adding option 34 needs `[6]`, not a one-line insert.

`build_module()` at `cp_kernels.c:296-314` would have to become
`cuModuleLoadDataEx` to get the option at all. That is where the *large* PTX is:
verified here by compiling each `.cu` with NVRTC 12.8 at `compute_120` and
timing `ptxas -arch=sm_120` (host-side proxy for what the driver JIT does):

| module | PTX bytes | ptxas 12.8 wall |
|---|---:|---:|
| `cp_rasterize.cu` | 613,032 | 0.53 s |
| `cp_fs.cu` | 390,083 | 0.31 s |
| `cp_vertex_fetch.cu` | 23,412 | 0.03 s |
| `cp_clear.cu` | 20,696 | 0.01 s |
| `cp_sampler.cu` | 280 | ~0 |
| `cp_math.cu` | 202 | ~0 |

So the note's "531 KB module" is `cp_rasterize.cu`'s ballpark, and it is loaded
through the one call site that currently cannot take a JIT option.

### Still true on 12.8 / sm_120?

Yes, mechanically. `CU_JIT_NUM_OPTIONS` is 34 in `/usr/local/cuda/include/cuda.h:1476`
(last named option is `CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33`), so `(CUjit_option)34`
is exactly the next slot, which is what CUDA 13.0 names `CU_JIT_SPLIT_COMPILE`.
The installed r580 driver reports `cuDriverGetVersion() = 13000`, so it has the
13.0 JIT behind 12.8 headers. Nothing in the 12.8 corpus contradicts this; the
option is simply not documented in 12.8 because it did not exist there.

### Verdict

* **Win:** cold-start only. Roughly -350 ms on the first JIT of the big raster
  module, per cache miss. **Zero effect on steady-state frame time.**
* **Risk:** low but not zero. The option is undocumented at 12.8 and passed as a
  raw integer; a driver that does not know it returns `CUDA_ERROR_INVALID_VALUE`
  and the module fails to load. Needs a fallback retry without the option.
  Second risk: **it changes the compute-cache key**, so landing it invalidates
  every cached cubin once — see item 2, which is already at the eviction cliff.
* **Effort:** small. Bump two array sizes, add one option, add a retry path.
  Converting `build_module()` to `cuModuleLoadDataEx` is another ~10 lines.
* **Affects:** cold start only.

### Experiment (do not run yet — needs the GPU)

    # A: cold-JIT cost today, cache bypassed, one process, no replay
    CUDA_CACHE_DISABLE=1 CUDAVK_PLAN_STATS=1 \
      ./build-cudapipe/.../vkcube --frames 1     # or the smallest sample that builds kernels
    # B: same with the option compiled in
    CUDA_CACHE_DISABLE=1 CUDAVK_JIT_SPLIT=16 ... same command
    # measure: wall time from cp_kernels_init() entry to exit, printed under
    # CUDAVK_VERBOSE; NOT frame time.

---

## 2. Raise `CUDA_CACHE_MAXSIZE`

### What the note claims

`CUDA13_UPGRADE.md:54-66`. The driver's on-disk cubin cache is the existing
cold/warm 2.5x replay factor (24.4 -> 9.73 ms/frame) and covers both
`cuModuleLoadData` and `cuLink` cross-process (334 ms cold -> 1.5 ms warm). It
reported `~/.nv/ComputeCache = 797 MB, 12,286 files` against the 1 GiB default,
and predicted "It is about to start evicting".

### Verified here, today

    payload bytes: 1,074,252,980  = 1024.49 MiB
    files:         9,957

The default is `1073741824` (1 GiB) and the documented maximum is `4294967296`
(4 GiB) — CUDA 12.8.1 Programming Guide, section 18 "CUDA Environment Variables",
`CUDA_CACHE_MAXSIZE` row.

**The prediction has already come true.** The cache is 0.05% over the 1 GiB cap
while the file count has *dropped* from 12,286 to 9,957. That is eviction in
progress: it is at the cap and now trading old cubins for new ones. Every evicted
entry re-pays a cold JIT (0.3-0.5 s of `ptxas` for the raster module alone, per
the table in item 1).

`CUDA_CACHE_MAXSIZE` is not set in the environment, and the string appears
nowhere in `src/`. It is referenced only in the two docs.

### Verdict

* **Win:** removes a growing, invisible cold-start tax. Cannot help steady-state
  frame time.
* **Risk:** essentially none. It is a user-environment variable; 4 GiB of disk.
* **Effort:** one line in `tests/*.sh` and the developer shell, or an
  `setenv`-if-unset in device init (careful — the codebase's rule is that
  environment reads happen once, in the debug-flag initialiser).
* **Affects:** cold start only. **Do this first because it is free and because
  items 1 and 3 both invalidate cache entries when they land.**

### Experiment

    du -sb ~/.nv/ComputeCache; find ~/.nv/ComputeCache -type f | wc -l   # before
    export CUDA_CACHE_MAXSIZE=4294967296
    # then re-run the standard sweep and re-measure; the number to watch is the
    # cold-vs-warm replay factor in tests/cp_gpu_busy.sh, not frame time.

---

## 3. `.pragma "enable_smem_spilling"`

### What the note claims

`CUDA13_UPGRADE.md:68-90`. "PTX 9.0". Tested by text-patching this tree's own
`cp_fs.cu` PTX to `.version 9.0` and JIT-loading on r580:

| kernel | before | after |
|---|---|---|
| `cp_fs_interpolate` | 150 regs, 1 CTA/SM | 128 regs, 2 CTA/SM |
| `cp_abuf_interpolate` | 8 B local, 5 CTA/SM | 0 B, 6 CTA/SM |

Two blockers named: it needs explicit launch bounds, and `cuLinkAddData` rejects
it ("not allowed for per-function compilation modes").

### What I verified here — the note is wrong about the version, and right about everything else

**`ptxas` 12.8 already implements this pragma. No PTX 9.0, no text patch, no
CUDA 13 toolkit.** Evidence:

    $ ptxas -arch=sm_120 <module-scope pragma, .version 8.7>
    ptxas ... error : Pragma 'enable_smem_spilling' is allowed only within function scope

That error is `ptxas` 12.8 *recognising* the pragma and objecting to its
placement. Moved to function scope, at `.version 8.7`, it assembles and works.

Reproduced on the tree's own `cp_fs.cu`, NVRTC 12.8 -> `ptxas -arch=sm_120 -v`,
with `.maxntid 256,1,1` added:

| entry | baseline | +`.maxntid 256` | +`.maxntid 256` +pragma |
|---|---|---|---|
| `cp_fs_interpolate` | 151 regs, 0 smem | 155 regs, 0 smem | **128 regs, 22,528 B smem** |
| `cp_abuf_interpolate_ranges` | 91 regs | 89 regs | **64 regs, 24,576 B smem** |
| `cp_abuf_interpolate` | 54 regs | 48 regs, 8 B stack, 4 B spill | **40 regs, 0 spill, 10,240 B smem** |
| `cp_fs_compact`, `cp_fs_writeback`, `cp_abuf_composite`, `cp_blit_linear`, `cp_resolve_samples` | 34-40 regs, no spills | unchanged | **unchanged — pragma is a no-op** |

And it is reachable straight from the `.cu` source under NVRTC 12.8, verified by
compiling a patched copy of `cp_fs.cu`:

```c
extern "C" __global__ void __launch_bounds__(256)
cp_fs_interpolate(struct cp_fs_interp_args args)
{
   asm volatile(".pragma \"enable_smem_spilling\";");
   ...
```

NVRTC emits `.maxntid 256, 1, 1` and the function-scope `.pragma`, and ptxas 12.8
gives 128 registers + 22,528 B smem — identical to the hand-patched PTX. So the
note's "needs a 13.x toolkit for `#pragma enable_smem_spilling` in `.cu` source"
is true only for the *spelling*; the effect is available today through inline asm.

**Launch bounds are load-bearing, and in the sharper sense than the note says.**
Without `.maxntid` the pragma is a complete no-op here — register counts and smem
are byte-for-byte identical to baseline. I did not reproduce the note's
"shared memory hits 43 KB/CTA and occupancy drops 6 -> 2" failure mode; on 12.8
`ptxas` simply declines to use the pragma when it has no thread-count bound.
None of the tree's `.cu` kernels carry `__launch_bounds__` today (grep for
`__launch_bounds__`/`maxntid` in `src/cudavk/kernels/*.cu` returns nothing), so
adding it is part of the change, and adding it alone already moves registers
(151 -> 155 on `cp_fs_interpolate`, 54 -> 48 with a new 4 B spill on
`cp_abuf_interpolate`).

**The `cuLinkAddData` blocker is real and reproducible:**

    $ ptxas -arch=sm_120 -c cp_fs_mod.ptx      # -c == per-function / relocatable
    ptxas fatal : Pragma 'enable_smem_spilling' is not allowed for per-function compilation modes

`ptxas -c` is what `cuLinkAddData(CU_JIT_INPUT_PTX, ...)` does. So:

* `cp_nir_to_llvm.c:3801-3809` (`cuModuleLoadDataEx`, taken when
  `!sampler_ptx && !fs_helper_ptx`) — **pragma allowed**. That is the
  hardware-inline fragment path, where `needs_sampler` is false
  (`cp_nir_to_llvm.c:4756-4757, 4766-4770`).
* `cp_nir_to_llvm.c:3812-3833` (`cuLinkCreate`/`cuLinkAddData`) — **pragma
  rejected**, exactly as the note says.

### The occupancy claim needs one more thing the note does not mention

`cp_fs_interpolate` at 128 regs x 256 threads = 32,768 registers/CTA, so two CTAs
fit the 64 K register file. But they also need 2 x 22,528 = 45,056 B of shared
memory on one SM. Compute capability 12.0 has a 128 KB unified data cache with a
shared-memory carveout selectable at 0/8/16/32/64/100 KB (CUDA 12.8.1
Programming Guide 16.10.3). A driver-chosen 32 KB carveout fits **one** CTA and
the doubling does not happen. `CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`
(= 14, `cuda.h:2155`) exists in 12.8 and would have to be set through
`cuLaunchKernelEx` to guarantee it. Any measurement of this item that does not
report the carveout is not a measurement.

### Where the win could land, and how small that is

The kernels the pragma moves are the interpolation kernels. Against
`PERFORMANCE.md` section 5.1 (pre-fan-out counts, still valid as shares):

* `fs_compact` — 74.5 launches/frame, 0.368 ms/frame. **This is the kernel the
  driver actually launches** on the direct path (`cp_renderer.c:4090-4093`
  picks `fs_compact` whenever in-shader interpolation is on). Pragma is a
  **no-op** on it — 38 regs, no spills.
* `cp_fs_interpolate` — the classic fallback, only when
  `CUDAVK_NO_FUSED_INTERP` or when A-buffer instrumentation is compiled in.
  This is the kernel the note's headline number is about, and it is off the
  default path.
* `cp_abuf_interpolate` / `_ranges` — the A-buffer shade path, inside the
  "abuf quad" 0.441 ms/frame class.

### Verdict

* **Win:** small and narrow. Bounded by ~0.4 ms/frame of A-buffer interpolation,
  and the biggest single measured change (`cp_fs_interpolate`) is on a path the
  driver does not take by default. Steady-state, not cold start.
* **Risk:** medium. Launch bounds alone change codegen on kernels that were fine.
  22.5 KB/CTA of shared memory is a new resource the occupancy heuristics in
  `cp_renderer.c` do not know about, and `measure_shader_cost()`
  (`cp_nir_to_llvm.c:3850-3860`) asks the driver for blocks/SM, so it will report
  the truth — but only after the carveout is right.
* **Effort:** small per kernel (two lines), plus a carveout decision, plus the
  usual `cp_gpu_busy.sh`-then-sweep discipline.
* **Affects:** steady state, on the A-buffer path only.

### Experiment

    # Host-only part is already done and reproducible:
    cd /tmp/perf-audit/probe && ./drv2 cp_fs_mod.cu > x.ptx && ptxas -arch=sm_120 x.ptx -o /dev/null -v
    # GPU part, when the GPU is free — pragma on cp_abuf_interpolate{,_ranges} only:
    tests/cp_gpu_busy.sh   # first, to confirm the sample is not host-bound
    # then the sweep, and report cuOccupancyMaxActiveBlocksPerMultiprocessor
    # and CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES for each patched kernel.

---

## 4. Reopen CUDA graphs on 12.8

### What `DEAD_ENDS.md` actually says

Entry 1, `docs/cudavk/DEAD_ENDS.md:49` and `:68-125`. Status **REFUTED**, three
iterations (12, 19, 20), summary line:

> "CUDA graphs, at three different units | 12, 19, 20 | REFUTED | 0.63-0.68 ms
> ceiling against 27k-44k instantiations; zero recipe reuse"

The mechanism paragraph, quoted in full because it is the thing the "reopen"
item has to answer:

> "**Mechanism.** The Driver API is not the blocker — a 12-node graph instantiates
> in 5.8 us median (iteration 19). The *unit* is wrong at every size tried. Keys
> rotate because scratch pointers rotate, and the command plan is re-recorded
> every submit, so a cached graph is never asked for twice inside the generation
> that could use it."

And the retry condition, also quoted:

> "**Retry if.** Command recording changes so that plans are reused within a
> generation, or a workload appears with real recipe recurrence. That means
> stable command-owned execution storage with fixed device addresses, not a
> bigger LRU over rotating scratch."

Decisive numbers behind it: iteration 20 found **4,532 submits with 4,532 unique
plan generations and zero repeated episode recipes within a generation**;
iteration 12 found 318,454 candidate three-kernel tails but only 65,905 exact
keys, an LRU hit rate of 69.2-76.5% at 256-4096 entries, and 27,563-43,944
instantiations needed against a **0.63-0.68 ms/frame exact-hit ceiling**.
Retained graphs cost ~1.46 GiB RSS + 632 MiB device for 256 large graphs.

### What the note proposes, honestly stated

`CUDA13_UPGRADE.md:92-104` does **not** contradict the mechanism. It says
iteration 12 measured the wrong quantity — instantiation — and offers
recapture + `cuGraphExecUpdate` + launch at 5.3 us per frame for 32 nodes against
43 us of raw launches, an 8x host cut "with no cache and no LRU hit rate".

That is a genuinely different proposal: it abandons caching, which is precisely
what `DEAD_ENDS.md` refuted. `cuGraphExecUpdate` only works when the new graph is
**topologically identical** to the instantiated one — same nodes, same edges,
same order. So the retry condition it must satisfy is not "recipe recurrence"
but "topology recurrence", which is a much weaker requirement and is plausibly
met by the raster tail (stage1 -> stage2 -> stage3 is always three kernel nodes
in a line).

Nothing needed is missing from 12.8: conditional graph nodes, device-updatable
kernel nodes, `cuGraphExecUpdate`, device graph launch, and edge data with
`CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC` are all in `/usr/local/cuda/include/cuda.h`
(`:1880-1917`). Confirmed against the 12.8.1 Programming Guide 3.2.8.7.

**There is zero graph code in `src/cudavk` today** — `grep -rn "cuGraph\|cuStreamBeginCapture" src/cudavk` returns nothing. All three iterations were
measure-first rejections with no production source, per `DEAD_ENDS.md:125-129`.

### Two things that shrink this since the note was written

1. The fan-out (`20611f5b131`). Whole-batch or whole-episode graphs now have to
   capture across eight streams, or capture per side stream. Stream capture does
   support forked streams, but the capture must begin and end on the origin
   stream and every fork must join back. `cp_pass_join()` (`cp_renderer.c:7370`)
   and `cp_pass_broadcast()` (`:7428`) already have exactly that shape, so it is
   possible — but it is more work than the note's "32 nodes" model implies.
2. The frame is 13.2 ms, not 15.99. The note's own bound was "3.287 ms/frame of
   traced sub-20 us device-idle gaps"; part of that has now been eaten by the
   fan-out. **The gap budget must be re-traced before this is costed at all.**

### Verdict

* **Win:** host CPU only. The note's own arithmetic caps it at 43 us -> 5.3 us
  per 32-node chain; scaled to a frame this is the "1-3 ms of host time"
  estimate, and the frame is now host-blocked 12.44 ms out of a *smaller*
  total, so the share may have moved either way.
* **Risk:** high, historically. Three iterations, three refutations. The new
  proposal is not the refuted one, but it inherits the memory-footprint and
  invalidation hazards.
* **Effort:** large. This is the only item here that is not a few dozen lines.
* **Affects:** steady state, host side.
* **Honest statement for the docs:** *graphs are refuted as a cache. The
  `cuGraphExecUpdate` variant was never tried and is not what iteration 20
  closed, but it needs a re-traced idle-gap budget at HEAD before it is worth an
  iteration.*

### Experiment (cheap, and it is a measurement, not an implementation)

    # 1. Re-trace the idle-gap budget at HEAD, after the fan-out:
    nsys profile -t cuda --cuda-graph-trace=node <sample>
    # sum sub-20us device-idle gaps per frame; compare to the 3.287 ms figure.
    # 2. Only if that is still >1.5 ms: census how often the raster tail's
    #    *topology* (not its arguments) repeats within one plan generation.
    #    That is the number cuGraphExecUpdate lives or dies on, and it was
    #    never counted; iterations 12/19/20 counted exact-argument keys.

---

## 5. `griddepcontrol` / programmatic dependent launch — the main event

### What the note claims

`CUDA13_UPGRADE.md:106-118`. "Assembles today from the `.cu` side at
`compute_120`." Measured on chained dependent kernels:
stream 4.10 us/kernel -> graph 2.02 (-50.6%) -> **PDL 1.89 (-53.9%)**, with the
warning **"Graphs and PDL remove the same ~2.2 us dispatch gap; do not add them
up."** It also cites the `CPVK_ASYNC_SUBMIT=1` result — making submission
asynchronous made `instancing` *worse* (1.48x -> 1.59x) — as evidence that only
letting the next chain start before the previous drains can pay.

`docs/cudavk/history/SM120.md:79-88` independently lists `griddepcontrol` as
"OK, `.cu`, today" and calls it "the one worth arguing for".

Used zero times: `grep -rni griddepcontrol src/cudavk` returns nothing.

### How the kernels are compiled — this decides everything

| kernel family | toolchain | file |
|---|---|---|
| `cp_clear`, `cp_rasterize`, `cp_fs`, `cp_sampler`, `cp_math`, `cp_vertex_fetch` | **NVRTC 12.8**, `--gpu-architecture=compute_120` | `cp_kernels.c:105-217`, arch string built at `:116-117` |
| application VS / FS | **in-tree `nir_to_ptx`**: NIR -> LLVM 18 IR -> NVPTX backend | `nir_to_ptx/cp_nir_to_llvm.c:3595-3660` |

There is **no offline `nvcc` path** for device code in this tree.
`kernels/cu_to_inc.py` only stringifies the `.cu` sources into `.inc` headers.

### Can each path emit `griddepcontrol` on 12.8? Verified here.

**NVRTC path: YES, today, with one caveat.**

`cudaTriggerProgrammaticLaunchCompletion()` and `cudaGridDependencySynchronize()`
are **not visible** to NVRTC 12.8 by default:

    pdl.cu(1): error: identifier "cudaTriggerProgrammaticLaunchCompletion" is undefined

They live in `/usr/local/cuda/include/cuda_device_runtime_api.h:448` and `:464`,
which NVRTC does not pre-include. But both are one line of inline asm in that
header, and inline asm works:

```c
__device__ __forceinline__ void cp_trigger_dependents(void)
{ asm volatile("griddepcontrol.launch_dependents;":::); }
__device__ __forceinline__ void cp_wait_prerequisites(void)
{ asm volatile("griddepcontrol.wait;":::"memory"); }
```

NVRTC 12.8 at `compute_120` emits `.version 8.7` PTX containing both
instructions; `ptxas -arch=sm_120` assembles them into `ERRBAR` (launch_dependents)
and `ACQBULK` (wait). Verified with `nvdisasm -c`.

**LLVM shader path: NO, blocked twice over, and both blocks are one-line
constants.** `griddepcontrol` requires `.version >= 7.8` **and** `.target sm_90`
or higher (CUDA 12.8.1 PTX ISA 9.7.13.13, "Introduced in PTX ISA version 7.8",
"Requires sm_90 or higher"). The tree emits `sm_86` / `+ptx75`:

    nir_to_ptx/cp_nir_to_llvm.c:33     #define CP_MAX_PTX_SM 86
    nir_to_ptx/cp_nir_to_llvm.c:3494   target, triple, cpu, "+ptx75", ...
    nir_to_ptx/cp_nir_to_llvm.c:3625   target, triple, cpu, "+ptx75", ...
    src/cudavk/meson.build:89          '-march=sm_86', '--cuda-feature=+ptx75'   (inline FS bitcode)

Confirmed by assembling the same PTX at each setting:

    .version 7.5 -> error : Feature 'griddepcontrol' requires PTX ISA .version 7.8 or later
    .target sm_86 -> error : Instruction 'griddepcontrol' requires .target sm_90 or higher
    .version 8.3 + .target sm_90  -> assembles clean for -arch=sm_120
    .version 8.3 + .target sm_90a -> ptxas fatal : Program with .target 'sm_90a' cannot be
                                     compiled to future architecture

So the fix for the shader path is `CP_MAX_PTX_SM 86 -> 90` and `+ptx75 -> +ptx83`
(**`sm_90`, not `sm_90a`** — the `a` suffix is architecture-locked and the driver
cannot JIT it forward to sm_120). `SM120.md:182-188` already proposes exactly this
pair as "the cheap experiment", for an unrelated reason, and states LLVM 18's own
ceiling is `sm_90a` / `ptx83`. So no LLVM upgrade is needed. But it is a codegen
change to every shader in the driver and must be proven byte-identical first.

### The ABI requirement on the host side

Per CUDA 12.8.1 Programming Guide 3.2.8.6.2 and `cuda.h:2023-2033`:

* the **secondary** kernel must be launched with
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (= 6) through
  **`cuLaunchKernelEx`**. Both exist in the 12.8 headers.
* the **secondary must** contain `griddepcontrol.wait` (or another means) before
  touching the primary's output. Without it the data is not guaranteed visible.
  This is the correctness-critical half.
* the **primary's trigger is optional**: "If the primary kernel doesn't execute
  the trigger, it implicitly occurs after all thread blocks in the primary kernel
  exit." So a chain link whose *producer* is an LLVM shader still works, as long
  as its *consumer* is a `.cu` kernel. It just does not get the early-trigger
  bonus, only the dispatch overlap — which is where the note's 2.2 us lives.
* PDL is compute capability 9.0+. sm_120 qualifies.

The single host chokepoint is `cp_launch()`, `cp_renderer.c:490-501`, which is
enforced by `cp_launch_audit` — "Nothing in the driver may call `cuLaunchKernel`
directly" (`cp_renderer.c:486-488`). One function to change.

### Which chain links are still strictly serial at HEAD (post-fan-out)

The fan-out puts **whole opaque episode segments** on `cp->seg_streams[k]`
(`cp_renderer.c:9147-9162` and `:9037-9049`); *within* a segment
`cp_draw_execute_batch()` issues the entire chain back to back on that one
stream. So the fan-out overlaps **across segments**, never **within a chain**.

| link | primary -> secondary | file:line | compiled by | still serial at HEAD? |
|---|---|---|---|---|
| A | `rasterize_stage1` -> `stage2` | `6754` -> `6780` | .cu -> .cu | **yes** |
| B | `stage2` -> `stage3` | `6780` -> `6786` | .cu -> .cu | **yes** |
| A' | `clip_rast_fused` -> `stage2` | `6737` -> `6780` | .cu -> .cu | **yes** |
| C | `stage1_abuf` -> `stage2_abuf` -> `stage3_abuf` | `6346/6349/6352`, `6502/6505/6508`, `8524/8527/8530` | .cu -> .cu | **yes** (per stream) |
| D | abuf scan chain: `abuf_scan_block` -> `scan_add` -> `scan_block` -> `scan_add` -> `scan_finish` | `1986, 1990, 1995, 2007, 2022, 2040` | .cu -> .cu | **yes, on the main stream, inside `cp_pass_finish()`** |
| E | `fs_compact`/`fs_interpolate` -> FS | `4091` -> `3825` | .cu -> **LLVM** | yes, but **secondary is LLVM: blocked** |
| F | FS -> `fs_writeback` | `3825` -> `4170` | LLVM -> .cu | **yes, and works today** (implicit trigger) |
| G | `opaque_tile_count` -> `tile_fill` -> `tile_raster` | `7534, 7556, 7581` | .cu -> .cu | yes (prototype path, `CUDAVK_TILED_OPAQUE` only, and that flag disables the fan-out at `7318`) |
| H | `abuf_seg_count` -> `seg_prefix` -> `seg_scatter` | `8252, 8280, 8291` (also `8613, 8728`) | .cu -> .cu | **yes, main stream** |
| I | tiled-opaque relaunch `stage1 -> stage2 -> stage3` | `7597, 7600, 7603` | .cu -> .cu | yes (same flag-gated path) |
| — | segment *n* vs segment *n+1* | `9147-9162` | — | **NO — already overlapped by the fan-out. Do not count this.** |

The two clean, uncontested targets are **D** (the A-buffer scan chain, 128.2
launches/frame of tiny dispatch-bound kernels, run serially on the main stream in
`cp_pass_finish()` between two `cp_pass_join()` calls, and completely untouched by
the fan-out) and **A/B/C** (the raster tail, 627 launches/frame, 54.7% of kernel
time).

The fan-out changes the *expected size* of A/B/C's win, not its existence: where
episodes have many segments (old capture averages 12.42, reaches 49) eight streams
are already keeping the machine fed, so a per-link bubble on one stream is
partly hidden. Where episodes are one segment — Crossroads averages 1.71,
`pbribl` and `pushconstants` are exactly one (`cp_renderer.c:9130-9139`) — the
chain is alone on the main stream and PDL is the only thing that can fill the gap.
**Prediction to test: PDL should pay most on exactly the samples the fan-out did
not help.** That is a falsifiable statement and it costs one sweep to check.

### The blockers nobody has written down yet

1. **`cp_launch()` flushes an owed H2D copy on the same stream immediately before
   every launch** (`cp_renderer.c:494`, `cp_upload_flush()`). The
   `PROGRAMMATIC_STREAM_SERIALIZATION` attribute is documented against "the
   previous **kernel** in the stream" (`cuda.h:2023-2033`); it says nothing about
   a preceding async memcpy. Until that is settled, the attribute must only be
   set when `cp_upload_flush()` issued nothing for this launch. `cp_launch()`
   already knows — it calls the flush itself and can compare
   `cp->upload.flushes`.
2. **The queue-counter memsets sit directly before stage 1** —
   `cuMemsetD32Async(cp->cur_qset.counts, 0, 3, cp->stream)` at
   `cp_renderer.c:6304, 6498, 6725`, and at `:8522` inside the per-segment fill
   relaunch. Same question. Links A/A'/C are therefore *not* the first ones to
   convert; links B (`stage2 -> stage3`) and D (scan chain) have a kernel as the
   immediate predecessor and are clean.
3. **`ptxas` hoists the wait.** In the probe, `ACQBULK` landed at instruction
   offset 0x40, *before* the address arithmetic it was meant to overlap. The
   `"memory"` clobber orders memory ops, not integer ALU. Any conversion must
   read the SASS and confirm the wait sits after a genuinely useful preamble, or
   the overlap is zero and only the dispatch gain remains.
4. **`cuLaunchKernelEx` is not more expensive** — the note measured 1.18 us
   against `cuLaunchKernel`'s 1.35 us — so converting `cp_launch()` wholesale is
   not itself a regression risk.
5. **Opportunistic, not guaranteed.** Programming Guide 3.2.8.6.2: "this
   behavior is opportunistic and not guaranteed to lead to concurrent execution.
   Reliance on concurrent execution in this manner is unsafe and can lead to
   deadlock." Nothing here may *depend* on the overlap for correctness.

### Verdict

* **Win:** steady state, device side. The note's microbenchmark is 4.10 -> 1.89
  us per chained kernel. At 627 raster-chain launches/frame plus ~128 scan
  launches, the *arithmetic* ceiling is large, but it is bounded by the real
  device-idle gap, which must be re-traced after the fan-out. Do not add it to
  any graphs estimate — the note says explicitly they remove the same gap.
* **Risk:** medium. Correctness hazards 1 and 2 above are the real ones; both
  are avoidable by scoping the first conversion to links B and D.
* **Effort:** small for the `.cu` links. `cp_launch()` gains a `bool pdl`
  argument and a `cuLaunchKernelEx` path; each secondary `.cu` kernel gains one
  inline-asm line. Large for the FS, because it needs the sm_90/ptx83 codegen
  bump proven byte-identical first.
* **Affects:** steady-state frame time. This is the only one of the five that
  does.

### Experiment

    # Step 0 (host only, already done): confirm the instructions assemble.
    cd /tmp/perf-audit/probe && ./drv pdl2.cu | grep griddepcontrol
    # Step 1 (GPU): re-trace the device-idle gap budget at HEAD, post-fan-out.
    nsys profile -t cuda,nvtx --gpu-metrics-device=0 <sample>
    #   -> sum of sub-20us gaps INSIDE a single stream, split by stream.
    #      That is the only quantity PDL can attack.
    # Step 2 (GPU): convert link D only (abuf scan chain), behind CUDAVK_PDL=1:
    #   - abuf_scan_add, abuf_scan_finish get griddepcontrol.wait at entry
    #   - cp_launch() sets CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    #     for those two, and only when cp_upload_flush() issued nothing
    #   - prove byte-identical images first, then:
    tests/cp_gpu_busy.sh                  # is the sample device-bound at all?
    <the standard sweep>, with and without CUDAVK_PDL=1
    #   report per-sample, and check the prediction that one-segment samples
    #   (pbribl, pushconstants, Crossroads) move MORE than many-segment ones.
    # Step 3: only if step 2 pays, do links B and C.

---

## What I would actually try first, and why

1. **`export CUDA_CACHE_MAXSIZE=4294967296`.** Do it today, before anything else.
   It is one line, it has no downside, and it is no longer speculative: the cache
   is at 1024.49 MiB against a 1024 MiB cap with the file count falling from
   12,286 to 9,957, so it is evicting right now. Items 1 and 3 both invalidate
   cache entries when they land, and this is the only thing that stops that from
   compounding. *Experiment: the `du -sb` before/after above, plus the cold/warm
   replay factor from `tests/cp_gpu_busy.sh`.*

2. **`griddepcontrol` on link D — the A-buffer scan chain
   (`cp_renderer.c:1986-2040`).** This is the highest-value/lowest-risk cut in
   the list. 128.2 launches/frame of small kernels, strictly serial on the main
   stream inside `cp_pass_finish()`, untouched by the fan-out, every kernel is
   NVRTC-compiled so the instruction is available today (verified), and every
   link in it has a *kernel* as its immediate predecessor so hazards 1 and 2 do
   not apply. It is also the cheapest possible test of whether PDL pays here at
   all. *Experiment: step 2 above.*

3. **Re-trace the intra-stream device-idle gap budget at HEAD.** Not a change —
   a measurement, and the one that decides items 4 and 5 both. Every estimate in
   `CUDA13_UPGRADE.md` is anchored to "3.287 ms/frame of sub-20 us gaps" measured
   before the fan-out moved the frame 15.77 -> 13.23 ms. Until that number is
   re-taken and split per stream, both PDL and graphs are being sized against a
   budget that no longer exists. *Experiment: step 1 above.*

4. **`griddepcontrol` on link B (`stage2 -> stage3`), then link C.** Only after
   2 and 3. Bigger prize — 627 launches/frame, 54.7% of kernel time — and bigger
   blast radius. Test the falsifiable prediction that it pays most on the
   one-segment samples the fan-out did not help (`pbribl`, `pushconstants`,
   Crossroads).

5. **`CU_JIT_SPLIT_COMPILE` + `cuModuleLoadDataEx` in `build_module()`.** Cheap,
   safe with a fallback, worth ~350 ms of first-run latency on the 613 KB
   `cp_rasterize` PTX. It buys nothing in steady state, so it ranks below
   anything that touches the frame — but it is genuinely low effort and it makes
   the developer loop faster, which is worth something on its own.

6. **`enable_smem_spilling` on `cp_abuf_interpolate` and
   `cp_abuf_interpolate_ranges` only.** Now much cheaper than the note thought —
   NVRTC 12.8 plus one inline-asm line, no toolkit install, no PTX 9.0, no text
   patch (verified). But the headline kernel `cp_fs_interpolate` is off the
   default path, `cp_fs_compact` (the one actually launched) does not spill at
   all, and the whole class is ~0.4 ms/frame. It also needs the shared-memory
   carveout set explicitly or the occupancy doubling will not happen. Small,
   narrow, real.

7. **Do not reopen graphs yet.** `DEAD_ENDS.md` entry 1 is refuted on the
   *mechanism* — "4,532 submits have 4,532 unique plan generations and zero
   repeated episode recipes within a generation" — and the note's
   `cuGraphExecUpdate` proposal, while not the same thing, has never had its own
   number counted: nobody has measured how often the raster tail's *topology*
   repeats, as opposed to its arguments. That census (step 2 of item 4's
   experiment) is one iteration's cheapest possible answer, and it should come
   before any graph code is written. The fan-out also means graph capture now
   has to span eight streams.

---

## Corrections this audit makes to `docs/cudavk/notes/CUDA13_UPGRADE.md`

1. `link_shader_module()` does not exist. The function is `load_shader_module()`,
   `nir_to_ptx/cp_nir_to_llvm.c:3778`, and its option arrays are sized `[5]` and
   already full.
2. `enable_smem_spilling` is **not** a PTX 9.0 / CUDA 13 feature in practice.
   `ptxas` 12.8 implements it at `.version 8.7`, at **function scope**, and it is
   reachable from `.cu` source under NVRTC 12.8 via one line of inline asm. No
   version text-patch is needed. Verified.
3. Without launch bounds the pragma is a **no-op** on 12.8, not a 43 KB/CTA
   pessimisation. The failure mode the note warns about did not reproduce.
4. The "797 MB / 12,286 files" cache figure is stale. It is now
   1,074,252,980 B (1024.49 MiB) over 9,957 files — at the 1 GiB cap and
   evicting.
5. `griddepcontrol` is available to the `.cu` kernels today but **not** through
   `cudaTriggerProgrammaticLaunchCompletion()` / `cudaGridDependencySynchronize()`,
   which NVRTC 12.8 does not declare. Inline asm is required.
6. The note does not say that the LLVM shader path is blocked from
   `griddepcontrol` by `CP_MAX_PTX_SM 86` and `+ptx75`, nor that the fix is
   `sm_90` and **not** `sm_90a` (the `a` variant cannot be JIT-forwarded).
7. The note does not say that the **primary's** trigger is optional, which means
   `FS -> fs_writeback` is convertible with no shader-codegen change at all.
8. Every per-frame estimate in the note predates `20611f5b131` and its
   15.77 -> 13.23 ms.
