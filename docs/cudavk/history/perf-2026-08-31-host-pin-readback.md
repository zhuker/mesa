# Host-pinned readback pages (default on)

The capture's per-frame image readback is a 3,686,400-byte
`vkCmdCopyImageToBuffer` into a persistently mapped, dedicated,
transfer-destination-only host-visible buffer. Host-visible allocations are
managed memory, so those pages migrated device-ward under the copy and
host-ward under the application's read, every frame. On B200 the CUPTI trace
showed the cycle directly: the 2D copy ran at 2.2 GB/s (1.6-1.8 ms for
3.7 MB), plus 0.98 ms/frame of UVM page traffic - about 2.4 ms/frame of
memcpy exclusive cost on every frame.

The fix advises exactly that buffer class - transfer-dst-only usage bound to
a managed allocation - host-resident with device access
(`cuMemAdvise` preferred-location CPU + accessed-by GPU) at
`vkBindBufferMemory` time, inside the device's context scope. The device then
writes the copy straight over the bus and no page ever migrates. Buffers with
any GPU-consumed usage bit are untouched, so UBO/SSBO paging behavior is
unchanged. The application is unchanged.

Measured, strictly alternating three-round A/B in single sessions, paired
submit medians over each capture's own real-work window (favorite3 frame 1391
= submit 2782; favorite2 frame 1388 = submit 2776 - see WORKFLOW.md 4.0),
every run with the standard stdout hash, full timestamp population, and 18/18
sentinels:

| capture | GPU | control | pinned | delta |
|---|---|---:|---:|---:|
| favorite3 | RTX 5090 | 7.6048 | 6.6757 | -0.9291 |
| favorite2 | RTX 5090 | 6.5461 | 5.4804 | -1.0657 |
| favorite3 | B200 | 12.188 | 10.789 | -1.399 |
| favorite2 | B200 | 10.5662 | 8.9763 | -1.5899 |

The favorite2 rows were first published against favorite3's window (submit
2782); recomputed on the correct 2776 window they move by under 0.01 ms and
the conclusion is unchanged. Band medians on RTX favorite2, pinned arm, show
where the remaining work is: light band (frames 1388-2734) **4.7949**, heavy
band (frames 2735+) **6.0461**.

The RTX win exceeds its raw memcpy pool because the migration cycle also
serialized against dependent stream work. The B200 pinned-arm trace confirms
the mechanism: DTOD copy time fell 1.454 to 0.323 ms/frame and UVM traffic
0.98 to 0.24 ms/frame.

Gates: 79/79 suite with the new default, 18-sample sweep byte-identical to
the previously accepted iteration (same pre-existing gltfscenerendering
budget line and missing renderheadless as the last two accepted sweeps),
`cp_debug_doc`/no-getenv/launch audits clean.

This was found on B200 - the first B200-derived win - during the port that
also added the CUDA 13 `cuCtxCreate` signature guard. B200 evidence lives in
`/tmp/f3-b200/` and `/tmp/f2-b200/` on the B200 host; RTX A/B in
`/tmp/pin-rtx-ab/`.
