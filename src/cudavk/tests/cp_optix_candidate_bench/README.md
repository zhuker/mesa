# OptiX custom-primitive candidate benchmark

This is an optional standalone probe. It is not built by Meson and does not
make OptiX a cudavk dependency. It measures the narrow RT-core candidate
enumerator considered in `docs/cudavk/DEAD_ENDS.md`. The total is always
354,000 custom triangle AABBs; select 1, 4, or 16 separately built and traced
visibility runs.

It requires the OptiX 8 or newer SDK headers and `libnvoptix.so.1`. Point
`OPTIX` at the SDK include directory:

```sh
OPTIX=/path/to/OptiX/include ./build.sh
./run.sh 9 1     # repetitions, runs (1, 4, or 16)
```

GPU events report total build, in-place fixed-count refit, and trace time.
The custom intersection performs a two-sided Moller-Trumbore test and
closest-hit writes the primitive ID. This is deliberately an optimistic
performance probe, **not an exact Vulkan renderer**: it neither implements the
top-left rule nor appends and submission-sorts every covered candidate.

The corrected current-capture shape is one dominant opaque run. Three local
GPU-lock-serialised `./run.sh 9 1` executions measured 0.237, 0.238 and 0.237
ms for refit plus trace. That exceeds the run-level 0.15-ms gate and nearly
consumes the complete 0.265-ms joint-renderer budget before exact coverage,
append, sort, depth, shading, or blending. The older 16-run stress shape
measured 1.613, 1.498 and 1.475 ms and is retained as a split-cost bound, not
the primary workload model.

`cp_optix_bvh_bench.cpp` is the companion one-GAS build/refit probe used to
correct the original published-throughput estimate. After `build.sh`, run
`./optix_bvh_bench 354000 random` for the random-bounds case.
