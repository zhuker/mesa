# CPU topology of the development machine

Recorded 2026-09-03 from `lscpu`, `numactl --hardware` and
`/sys/devices/system/cpu/*/cache/index3`.

## Is it NUMA?

No. The CPU is an **AMD Ryzen 9 9950X3D** (16 cores, 32 threads, one socket)
and the kernel reports a **single NUMA node** covering all 32 logical CPUs:

    NUMA node(s):       1
    NUMA node0 CPU(s):  0-31
    node 0 size:        61918 MB
    node distances:     node 0: 10

Memory latency is uniform. `numactl`, `libnuma` and NUMA-aware allocators have
nothing to place here.

## The NUMA-like wrinkle: two asymmetric core dies

The chip has two core dies (CCDs) with **separate L3 caches**, and they are
not the same size:

| CCD | cores | SMT siblings | L3 |
|---|---|---|---|
| 0 (3D V-Cache) | 0-7 | 16-23 | 96 MB |
| 1 | 8-15 | 24-31 | 32 MB |

Consequences for measurement and pinning:

- A thread that migrates between the two dies loses its L3 working set.
  Cross-die cache-to-cache traffic goes over the Infinity Fabric.
- If the driver's submit thread and the replay process are pinned, keep them
  on the **same CCD**, preferably CCD 0 (the 96 MB die). That is the closest
  thing to NUMA placement this machine offers.
- Logical CPU `n` and `n+16` are SMT siblings of one physical core. Pinning
  two busy threads to `0` and `16` puts them on the same core, not on two.

Example pinning to the V-Cache die, physical cores only:

    taskset -c 0-7 <command>

The `CPU(s) scaling MHz: 30%` line in `lscpu` reflects an idle sample, not a
cap; check `cpupower frequency-info` before attributing a slow run to clocks.
