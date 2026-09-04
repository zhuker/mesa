# The native-driver reference: where the floor actually is

Measured 2026-09-03 on the RTX 5090, driver **580.173.02**, by running the two
compiled harnesses against NVIDIA's own Vulkan ICD
(`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json`) instead of cudavk's.
Same binaries, same submit shim, same windows, same session.

Nobody had done this. Every performance number in this tree compares cudavk
against cudavk, so the campaign has never known what the command stream costs
on a driver that is not ours.

## The floor

| capture | native NVIDIA | cudavk | gap |
|---|---:|---:|---:|
| favorite3 whole | **0.522** | 5.940 | **11.4x** |
| favorite3 heavy | 0.582 | 6.777 | 11.6x |
| favorite2 whole | **0.502** | 5.022 | **10.0x** |
| favorite2 heavy | 0.519 | 5.587 | 10.8x |

Three runs per capture, spread <= 0.01 ms -- an order of magnitude tighter than
cudavk's own 0.06 ms session drift.

**Read this before planning any further optimisation.** The 5.0 ms goal is not
near a floor: the same GPU executes the same submits in half a millisecond. The
0.94 ms favorite3 still needs is about 1.7% of the gap to a real driver. The
sub-gate leads of the 2026-09-02 campaign were not sub-gate because the work is
irreducible -- they were sub-gate because the architecture costs 10x and the
leads were shaving the implementation.

## Why the comparison is valid

A replay that dies early produces a fast, meaningless median (`WORKFLOW.md`),
and 10x invites exactly that suspicion. Checked before quoting:

- `rc=0`, and **6,947 / 6,965 shim rows** -- identical to cudavk's, so every
  submit executed
- 18 sentinel frames dumped, full 3,686,400 bytes each
- image statistics indistinguishable from cudavk's control (mean 131.73 vs
  131.78, std 84.80 vs 84.78)

## Reproducing it

```bash
# the floor, three runs per capture
cd ~/favorite3-cpp/out
env DUMP_DIR=/tmp/native/f3 DUMP_EVERY=200 DUMP_MAX=20 \
    LD_PRELOAD=~/favorite-cpp/submit_shim.so SUBMIT_TS_FILE=/tmp/native/f3/ts.txt \
    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./build/vulkan_app

# the permanent timeline reference (both captures, ~15 min, 5.5 GB)
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json \
  src/cudavk/tests/cp_harness_timeline.sh native-nvidia
```

The timeline lives at `~/timelines/native-nvidia/{favorite2,favorite3}/` and has
a row in `~/timelines/index.html` beside every cudavk iteration, in the same
page format, so any iteration can be read against a real driver frame by frame.

## The by-product: a correctness oracle

The reference render is 3,473 + 3,482 frames of what a real driver produces
from this command stream. **The test suite has never had that** -- the sentinel
controls compare cudavk against cudavk, so a defect present in every cudavk run
is invisible to them.

Comparing the two renders frame by frame (every 25th, 139/140 frames):

| | favorite3 | favorite2 |
|---|---:|---:|
| identical bytes, median | 75.74% | 75.18% |
| within +-2, median | 99.89% | 99.89% |
| pixels differing by >8, median | 0.041% | 0.041% |
| pixels differing by >8, **max** | **19.99%** | 0.09% |
| frames with >1% badly differing | **1** | 0 |

Aggregate agreement is good and the floor comparison is sound. But **favorite3
frame 3225 differs on 20% of its pixels**: same geometry, same UI, a large snow
surface in the lower left shaded differently, and the whole frame darker (mean
158.1 against 166.8). That is a shading defect, not rasterisation precision, and
it is now `TODO.md` correctness item 14.

To repeat the comparison:

```python
# both labels rendered, then per-frame PNG diff
a = np.asarray(Image.open('~/timelines/native-nvidia/favorite3/frames/frame_003225.png').convert('RGB'), np.int16)
b = np.asarray(Image.open('~/timelines/session-final2/favorite3/frames/frame_003225.png').convert('RGB'), np.int16)
share = 100 * (np.abs(a-b).max(axis=2) > 8).mean()
```

## The methodological lesson

The first version of this check used the **18 sentinel frames** the floor runs
dumped (`DUMP_EVERY=200`), and reported "99.99% within +-2". The full render
says the median is 99.89% and the worst frame is 79.50%. The 18 frames sample
every 200th from zero, which under-weights the heavy band where the hard frames
are, and they covered favorite3 only.

**A sentinel sample is a smoke test, not an image comparison.** When the claim
is "these render the same", the population is every frame, and it is now cheap
to check because both renders exist.
