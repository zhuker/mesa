# The resolve-reason probe — flag, command, and how to read it

Built on `drain-work`, commit `704d218bc9c`, on top of Tier 2 (`0288e470ef0`).
**Counters only. No behaviour change. Default off.** Nothing was run on the GPU.

Binary: `/tmp/drain-tree/build-cudavk-drain/src/cudavk/libvulkan_cudavk.so`,
**md5 `836f7c612b9e97599b0e30161d7d5564`**. Rebuild window
**2026-08-27T13:15:10–13:15:12 −0700**, GPU idle before and after; discard any
run started inside it. The previous binary was `f64caa0d10e6…`.

---

## 1. The command

```
CUDAVK_AHEAD=2 CUDAVK_AHEAD_MIN_VERTS=2 CUDAVK_AHEAD_WHY=1
```

**One run per capture is enough**, and it is a probe run — its median must not
be quoted. `MIN_VERTS=2` is the sweep's saturated setting, so the population it
attributes is the same one the sweep measured (36,129 held, 4,096 at the cap,
8,422 holding nothing). A control arm is not needed: this instrument compares
its own columns, not two runs.

Worth one extra run if there is budget: the same command with
`CUDAVK_AHEAD_MAX_SEGS=1`. It forces every deferral to hold at most one batch,
so the `zero` column becomes "no successor at all" for a population that is not
capacity-limited anywhere — a cross-check on the reading of the `late` column.

## 2. What it prints

```
cudavk: run-ahead why: resolves by site (resolve/zero-hold/late successor):
        scope_begin:N/N/N scope_end:N/N/N flush:N/N/N not_appendable:N/N/N
        append_opaque:N/N/N append_small:N/N/N append_setup:N/N/N
        opaque_append:N/N/N draw_execute:N/N/N admit:N/N/N seam:N/N/N
cudavk: run-ahead why: pass_finish N (N zero), opaque_append N (N zero),
        draw_execute N (N zero), admission N (N zero)
```

The second line is the four-way split the question was asked in; the first is
the finer key it is built from.

**Read the columns in this order.**

1. **`resolve` must sum to the deferral count** (14,932 on old). It is the
   instrument's own self-check; if it does not, the hint is being lost
   somewhere and nothing below may be read.
2. **`zero`** is the 8,422. Whichever site carries it is the answer.
3. **`late`** is the tie-breaker, and only for sites that carry `zero`.

## 3. The three futures, as columns

| what carries the `zero` column | what it means | what happens next |
|---|---|---|
| `scope_begin` + `scope_end` | the run-ahead cannot cross a render-scope boundary, and must not: the admission rule refuses a different framebuffer because `cp_abuf_setup()` would resize the arrays the deferred tail is still reading | **item 2 closes** at the 0.068 ms/frame it measures today. The bimodal held histogram is "the pass ended" and "the pass continued" |
| `opaque_append` | the successor is an OPAQUE episode | **Tier 3 becomes the lead.** The design named it and never designed it; it is strictly easier, because an opaque run uses `visbuf` and has no A-buffer count to hold |
| `flush` / `not_appendable` with a high `late` | the successor existed and a required resolve came first | the reach is recoverable **without any capacity change** — the best outcome available in this item, and the one to look for hardest |
| `draw_execute` non-zero at all | a path reaches CUDA with a tail open that should have closed it | a missed resolution point. It is a correctness finding, not a tuning one |

`late` distinguishes "there was no successor" from "the successor arrived behind
a resolve we had to take anyway". Both are unfixable by capacity; only the
second is fixable at all.

## 4. Registered before the run

* **C4.** `scope_begin` + `scope_end` + `opaque_append` together carry **more
  than half** of the 8,422 zero-hold deferrals. (This is C3 from the capacity
  read, scored.)
* **C5.** `draw_execute` is **0**. It is the safety net, and Tier 2's stray-launch
  counter already reads 0 at four settings, so a non-zero value here would
  contradict a measurement that already exists.
* **C6.** `admit` + `seam` reproduce the sweep's decline counts at the same
  setting (`full:4096`, i.e. 4,096 resolves attributed to `admit`), which is the
  cross-check that the new key and the old counters are counting one population.
* **C7.** `late` is **small on `scope_begin`/`scope_end`** — a new scope's first
  blended batch is a successor the mechanism could never have held — and is
  where the recoverable population would show up if it exists at all.

**If C4 fails**, the zero-hold population is somewhere I did not predict, and
the capacity read's §4 is wrong in its explanation while remaining right in its
arithmetic: 56.4% of deferrals still hold nothing, and capacity still cannot
reach them.

## 5. Inertness

| arm | .text | sections |
|---|---:|---:|
| A base (`0288e470ef0`, Tier 2, no probe) | 160,753 | 179 |
| B shipped, flag off | **161,822 (+1,069)** | 179 |
| C `cp_debug->ahead_why` folded to 0 | **160,780 (−1,042 from B)** | 179 |

Folding removes the probe from exactly the seven functions it touches:
`cp_batch_flush_defer_why` −231, `cp_context_cleanup` (the report) −362,
`cp_render_scope_end` −101, `cp_render_scope_begin` −88,
`cp_pass_record_segment` −80, `cp_batch_flush_why` −64, `cp_ahead_resolve` −116.
Residue against base: **+27 bytes in one function**, plus a GCC argument-clone
rename of `cp_ahead_resolve` (the site argument becomes dead when folded).
Registry control in `.data`: `cp_debug.c.o` `.text` unchanged at 3,011,
`.data.rel.ro.local.flags` +88 = one row, `.bss.present_in_env` +1.

House checks pass: `cp_debug_doc.py --check`, `cp_no_getenv.py`,
`cp_launch_audit.py`.
