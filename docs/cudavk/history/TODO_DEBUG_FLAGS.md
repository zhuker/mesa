# Plan: one registry for the driver's 44 environment switches

> **Done.** All seven steps were carried out; the registry is `cp_debug.c`, the
> generated table is `FLAGS.md`, and `CUDAPIPE_HELP=1` prints it from a running
> driver. The rule that a new switch goes in the registry rather than into a
> `getenv` is in `CLAUDE.md`; the mechanics of adding one are in
> `CUDAPIPE_HANDOFF.md` "Debug".
>
> **The document below is the plan as written, kept for its reasoning, and it
> is wrong in three places.** The inventory it calls complete is missing
> `CUDAPIPE_NVTX`; `CUDAPIPE_DEBUG_SHADER` no longer exists, because step 1
> removed its last two uses; so the count is 43, not 44. Step 7's renaming was
> deliberately not done — it is churn against scripts, and the generated table
> plus `CUDAPIPE_HELP` solved the finding-them problem that the renaming was
> for. What the steps actually cost and what verifying them turned up is in the
> commit messages, which are the record; two things worth carrying forward:
>
> - **`CUDAPIPE_DEBUG_FS` cannot be verified by diffing output.** It dumps the
>   shaded-pixel list in GPU scheduling order, so it hashes differently on
>   every run of any build. Check its contract instead.
> - **`gltfscenerendering` and `vulkanscene` do not render deterministically**,
>   at roughly a 10% rate, with no flags set and in a build predating all of
>   this work. Two runs, or even ten, say "stable" and are wrong. Frame
>   comparison on those samples is not a usable signal; `texture` is stable.

Third priority, behind `HEADLESS_STREAMER_PERF.md` and `TODO_CONFORMANCE.md`.
But it is cheap, and it is the change that most reduces the cost of the *next*
investigation.

---

## Why

The driver reads **44 distinct environment variables across 68 call sites**,
and there is no list of them anywhere. `ABUFFER.md` documents six,
`BATCHING.md` two. Everything else is discoverable only by grep.

That matters more here than in most drivers because **this driver's failure
mode is silence**, and these flags are its debugging surface. In one week they
did this much real work:

- `CUDAPIPE_NO_ABUFFER` and `CUDAPIPE_NO_BATCH` each bisected a rendering bug
  in **one command**, eliminating whole subsystems as suspects.
- `CUDAPIPE_SMALL_ALLOC` let three memory-residency schemes be A/B'd **inside
  one binary**, which removed build-to-build variance entirely — better
  methodology than the separate-`.so` scheme it replaced.
- `CUDAPIPE_ABUFFER_TIMING` revealed 11.8 M `cuEventRecord` calls costing 3% of
  the frame for instrumentation nobody had switched on.
- `CUDAPIPE_DEBUG_SHADER` would have named the `gl_FrontFacing` bug on day one.
  It was not set, because nobody sets a shader-debug flag until they already
  suspect the shader. That bug took bisecting a frame to a single draw and
  dumping every descriptor from two drivers.

**The performance argument is weak and should not be the motivation.** Measured:
`getenv` is 31 ns at this environment size, about 6,400 calls a frame from the
per-draw sites, **0.199 ms/frame — 0.30 s across a 123.8 s replay, 0.24%.**
Removing it is a side effect of doing this properly, not a reason to.

---

## What is wrong today, beyond the missing list

Found while compiling the inventory. Each is worth fixing on its own.

### 1. `CUDAPIPE_ABUFFER_VERIFY` has two different meanings

| where | code | so `…VERIFY=0` means |
|---|---|---|
| `cp_context.c:2086` | `cp_abuf.verify = v ? atoi(v) != 0 : 0` | **off** |
| `cp_kernels.c:62` | `on = (getenv("CUDAPIPE_ABUFFER_VERIFY") …) ? 1 : 0` | **on** |

Setting it to `0` disables the verification but still compiles the verify
kernels in. Nobody has been bitten because everyone sets `=1`, but it is a real
inconsistency and exactly what a registry prevents.

### 2. Two silent `undef` paths remain in the shader backend

`1c0ad4094fa` made the *unhandled intrinsic* warning fire always, because that
class of bug (`gl_FrontFacing`) cost days. Two siblings were missed and are
still behind `CUDAPIPE_DEBUG_SHADER`:

```
nir_to_ptx/cp_nir_to_llvm.c:1750   unhandled ALU op '%s' -> undef
nir_to_ptx/cp_nir_to_llvm.c:1759   no LLVM intrinsic for '%s' -> undef
```

Both produce `undef` and propagate it through everything downstream — the same
failure mode, the same silence. **This is the highest-value item in this
document and does not need the registry to fix.** Make them always-on,
once-per-name, exactly as the intrinsic one now is.

### 3. `ABUFFER.md`'s table is already wrong

It lists `CUDAPIPE_ABUFFER_TIMING=0 | drop the per-draw drain the CUDA-event
timings need`, implying the timing defaults **on**. The code is
`cp_abuf.timing = t ? atoi(t) != 0 : 0` — it defaults **off**. Anyone following
that table sets a variable that changes nothing. Hand-maintained tables drift;
a generated one cannot.

### 4. The naming has no rule, which is why memory fails and grep wins

```
CUDAPIPE_NO_ABUFFER    CUDAPIPE_ABUFFER_COMPOSITE=0   one subsystem,
CUDAPIPE_ABUF_COMPILE  CUDAPIPE_NO_ABUF_BATCH          3 prefixes, 2 polarities

CUDAPIPE_SMALL_ALLOC       (allocator)                 unrelated subsystems,
CUDAPIPE_SMALL_THRESHOLD   (rasterizer tile size)      same prefix
```

Parsing is inconsistent too: presence-only, `atoi() != 0`, `atoi() > 0`,
`strtoull`, `atof`, and `*v` non-empty checks all appear.

### 5. Caching is inconsistent

Some flags are read once into a `static` with a sentinel (`enabled = -1`);
`cp_abuf` reads its six in one block; the arena caches its four. But all 13
`CUDAPIPE_DEBUG_DRAW` sites, all five `CUDAPIPE_DEBUG_TEX` sites and the whole
`cp_shade_fragments` group call `getenv` every time. Two established patterns,
no rule about which to use.

---

## The inventory

Complete at `b35443cfbbb`. **Parse** is what the code does today; preserve it
per flag unless the change is called out as deliberate. **Hot** means it sits on
a per-draw or per-map path.

### A-buffer (`cp_context.c` ~2073-2105, `cp_kernels.c`)

| name | parse | default | hot | notes |
|---|---|---|---|---|
| `CUDAPIPE_NO_ABUFFER` | presence | off (A-buffer on) | — | cached |
| `CUDAPIPE_ABUFFER_VERIFY` | `atoi!=0` / presence | off | — | **two meanings, see above** |
| `CUDAPIPE_ABUFFER_VERIFY_DRAWS` | `atoi` | 8 | — | |
| `CUDAPIPE_ABUFFER_COMPOSITE` | `atoi!=0` | `!verify` | — | default depends on another flag |
| `CUDAPIPE_ABUFFER_TIMING` | `atoi!=0` | off | — | doc says otherwise |
| `CUDAPIPE_ABUFFER_DEBUG` | `atoi!=0` | off | — | |
| `CUDAPIPE_ABUFFER_LAYERS` | `atoi`, 0=unset | 0 | — | |
| `CUDAPIPE_ABUF_COMPILE` | presence+non-empty | unset | — | forces kernels in/out of NVRTC build |
| `CUDAPIPE_NO_ABUF_BATCH` | presence | off (batching on) | — | cached |

### Draw batching (`cp_context.c` ~5299-5907)

| name | parse | default | hot | notes |
|---|---|---|---|---|
| `CUDAPIPE_NO_BATCH` | presence | off | — | cached |
| `CUDAPIPE_BATCH_MAX` | `atoi`, clamped | `CP_MAX_BATCH_DRAWS` | — | `=1` must stay bit-identical to `NO_BATCH` |
| `CUDAPIPE_DEBUG_BATCH` | presence | off | — | cached |
| `CUDAPIPE_DEBUG_BATCHDIFF` | presence | off | — | cached, two sites |

### Small-allocation arena (`cp_resource.c` ~1029-1154)

| name | parse | default | hot | notes |
|---|---|---|---|---|
| `CUDAPIPE_SMALL_ALLOC` | string enum | `advise` | — | `advise\|pinned\|managed\|off\|blocksonly` |
| `CUDAPIPE_SMALL_ALLOC_MAX` | `strtoull` | 128 | — | bytes |
| `CUDAPIPE_SMALL_ALLOC_WARMUP` | `strtoull` | 64 | — | allocations before the arena opens |
| `CUDAPIPE_SMALL_ALLOC_STATS` | presence | off | — | `atexit` dump |

### Shader compilation (`nir_to_ptx/cp_nir_to_llvm.c`, `cp_context.c`)

| name | parse | default | hot | notes |
|---|---|---|---|---|
| `CUDAPIPE_NO_REGCAP` | presence | off | — | two sites, consistent |
| `CUDAPIPE_REGCAP_STATIC` | presence | off | — | |
| `CUDAPIPE_MAX_REGISTERS` | `atoi>0` | 0 | — | |
| `CUDAPIPE_LAUNCH_BOUNDS` | `atoi>0` | 0 | — | |
| `CUDAPIPE_TUNE_VETO` | `atof` | `CP_TUNE_VETO` | — | |
| `CUDAPIPE_SHADER_STATS` | presence | off | — | two sites, consistent |
| `CUDAPIPE_DUMP_NIR` | presence | off | — | three sites |
| `CUDAPIPE_DUMP_IR` | presence | off | — | |
| `CUDAPIPE_DUMP_PTX` | presence | off | — | |
| `CUDAPIPE_DEBUG_SHADER` | presence | off | — | **two silent-undef paths, see above** |

### Rasterizer tuning (`cp_kernels.c` ~113-135)

| name | parse | default | hot | notes |
|---|---|---|---|---|
| `CUDAPIPE_SMALL_THRESHOLD` | non-empty + parse | — | — | tile size, unrelated to the arena |
| `CUDAPIPE_MEDIUM_THRESHOLD` | non-empty + parse | — | — | |
| `CUDAPIPE_POINT_THRESHOLD` | non-empty + parse | — | — | |
| `CUDAPIPE_TILE_BOUND` | non-empty + parse | — | — | |

### Tracing (all uncached unless noted)

| name | parse | default | hot | sites |
|---|---|---|---|---|
| `CUDAPIPE_DEBUG_DRAW` | presence | off | **yes** | **13** — `cp_draw_execute` ×5, `cp_shade_fragments` ×2, `cp_buffer_map`, `cp_blit`, resolve, clear, batch |
| `CUDAPIPE_DEBUG_TEX` | presence | off | **yes** | 5 — incl. `cp_fs_launch_shader` |
| `CUDAPIPE_DEBUG_VFETCH` | presence | off | **yes** | `cp_draw_execute`; syncs, so debug-only |
| `CUDAPIPE_DEBUG_WORK` | presence | off | **yes** | `cp_shade_fragments` |
| `CUDAPIPE_DEBUG_DISCARD` | presence | off | **yes** | `cp_shade_fragments` |
| `CUDAPIPE_DEBUG_FS` | presence | off | **yes** | `cp_shade_fragments` |
| `CUDAPIPE_DEBUG_FS_VSTEP` | `atoi` | — | **yes** | `cp_shade_fragments` |
| `CUDAPIPE_DEBUG_FS_ROW` | `atoi` | -1 | **yes** | `cp_shade_fragments` |
| `CUDAPIPE_DEBUG_LAUNCH` | presence | off | yes | compute dispatch |
| `CUDAPIPE_DEBUG_TIME` | presence | off | — | cached |
| `CUDAPIPE_FRAG_CENSUS` | presence | off | — | cached, two sites |
| `CUDAPIPE_NO_BINCACHE` | presence | off | — | cached |

---

## The design

### One struct, filled once

`cp_screen.c`, at screen creation — the earliest point that runs once per
process and before any draw.

```c
/* cp_debug.h */
struct cp_debug {
   /* tracing */
   bool draw, tex, vfetch, work, discard, fs, launch, time, shader;
   int  fs_vstep, fs_row;
   /* subsystems */
   bool no_abuffer, no_batch, no_abuf_batch, no_bincache, no_regcap;
   unsigned batch_max, abuffer_layers;
   /* … */
};
extern const struct cp_debug *cp_debug;   /* read-only after init */
```

Read-only after init is the point: a flag that can change mid-run is a flag
whose behaviour cannot be reasoned about, and nothing here needs it.

### One registry, which is the single source of truth

```c
static const struct cp_flag_def flags[] = {
   { "CUDAPIPE_DEBUG_DRAW",   CP_FLAG_BOOL, offsetof(struct cp_debug, draw),
     "trace every draw: geometry, attachments, skip reasons" },
   { "CUDAPIPE_BATCH_MAX",    CP_FLAG_UINT, offsetof(struct cp_debug, batch_max),
     "cap draws per batch; 1 must be bit-identical to NO_BATCH", .dflt = CP_MAX_BATCH_DRAWS },
   …
};
```

Types: `BOOL_PRESENCE` (set if present at all), `BOOL_VALUE` (`atoi() != 0`),
`UINT`, `FLOAT`, `ENUM`. **Both boolean kinds are needed** — the existing flags
genuinely use both, and collapsing them would silently change meaning for
anyone who sets `=0` today.

### `CUDAPIPE_HELP=1` dumps it

Name, type, default, current value, one-line meaning. That is the inventory,
and it cannot drift because it is the same array the driver reads.

### Doc tables become generated

A small script emits the markdown table from the registry, so `ABUFFER.md`,
`BATCHING.md` and this file stop being hand-maintained. The `ABUFFER_TIMING`
error above is what hand-maintenance costs.

---

## Migration, in order

Each step builds, runs and is independently revertable. **Do not do this as one
commit.**

1. **Fix the two silent `undef` paths** (`cp_nir_to_llvm.c:1750`, `:1759`).
   Independent of everything else, highest value, no registry needed.
2. **Add `cp_debug.h`/`cp_debug.c` with the registry and `CUDAPIPE_HELP`**,
   populated at screen creation, with nothing yet using it. Verify the dump
   matches the inventory above.
3. **Migrate the cached flags first** — they already read once, so this is pure
   substitution with no semantic risk. Delete each old `static … = -1` block as
   its flag moves.
4. **Migrate the uncached hot ones** (`DEBUG_DRAW`, `DEBUG_TEX`, the
   `cp_shade_fragments` group). This is where the 0.3 s goes; it is also where
   a mistake is invisible, because the flags are off in every normal run.
   **Test each with the flag actually set**, comparing output before and after.
5. **Resolve `CUDAPIPE_ABUFFER_VERIFY`'s two meanings.** Pick `BOOL_VALUE`
   (`=0` means off, which is what a reader expects) and update
   `cp_kernels.c:62`. Call it out in the commit — it is a behaviour change for
   anyone setting `=0`.
6. **Generate the doc tables**, and delete the hand-written ones.
7. *Optional, separate:* rationalise the names. `ABUF_COMPILE` →
   `ABUFFER_COMPILE`, `SMALL_THRESHOLD` → `TILE_SMALL_THRESHOLD`. Keep the old
   names working for a while — someone's scripts use them, and this document's
   own recipes do.

---

## Risks

- **A flag that stops working is invisible**, because these are off by default
  and nothing tests them. The mitigation is step 4's rule: for each migrated
  flag, run something with it set, before and after, and diff the output. That
  is the only real verification available.
- **`=0` currently means "on" for presence-only flags.** Anyone with
  `CUDAPIPE_DEBUG_DRAW=0` in a script is getting tracing today. Preserving that
  exactly is why `BOOL_PRESENCE` has to survive as a type.
- **Read-once changes behaviour for anything that sets a variable mid-process.**
  Nothing does, but the A/B harnesses in `~/claude-scratchpad/` set variables
  per *process*, which stays fine.
- **`cp_kernels.c` reads flags during NVRTC setup**, which may run before screen
  creation completes. Check the ordering before moving those four; if it does,
  the registry needs to be filled earlier than the screen, or those stay put.
- Do not fold `CUDAPIPE_SMALL_ALLOC`'s enum into a bool. The three residency
  arms in one binary are the reason that work could be measured at all.

---

## Verification

- `CUDAPIPE_HELP=1` lists all 44 with correct defaults.
- Each migrated flag produces byte-identical output with the flag set, before
  and after.
- The 18-sample sweep is unchanged — `tests/cp_iterate.sh <new> <old>`, and see
  `tests/TESTING.md` on why a >5% move on `gltfscenerendering` is a question
  rather than a finding.
- The capture replay is unchanged in wall time. Expect the 0.3 s to be
  unmeasurable against a ±4% noise floor; **do not claim it**. If you want the
  number, count with the `LD_PRELOAD` shim at
  `~/claude-scratchpad/perf276/cushim.c` instead of looking for it in wall time.
