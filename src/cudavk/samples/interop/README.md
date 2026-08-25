# CUDA interop sample: a Vulkan frame in a PyTorch process

One frame, rendered by Vulkan in one process, consumed by PyTorch in another,
with no copy through the host and no copy on the GPU. The result comes back
into a second shared buffer and Vulkan checks it byte for byte.

    cpvk_interop_render     producer, C, Vulkan only, never links libcuda
    cpvk_interop_torch.py   consumer, Python, ctypes against libcuda.so.1
    run_interop.py          the test matrix, including the negative controls
    PROTOCOL.md             the wire contract both halves are built against

## Build and run

    make
    ./run_interop.py --frames 1000

`run_interop.py` sets `VK_DRIVER_FILES` itself rather than inheriting it, so a
stale export cannot decide which driver the run measured.

To run the two halves by hand, in one terminal:

    VK_DRIVER_FILES=<icd>.json \
    ./cpvk_interop_render --frames 100 --sync host \
        --exec "/home/alexzhukov/mesa/venv/bin/python3 cpvk_interop_torch.py"

or in two, for debugging:

    ./cpvk_interop_render --socket /tmp/interop.sock --frames 100
    /home/alexzhukov/mesa/venv/bin/python3 cpvk_interop_torch.py \
        --socket /tmp/interop.sock

## The application does not know which driver it is running on

That is the claim the sample exists to support: **one unmodified binary, any
driver that implements the same contract, the same result.**

So there is no ICD path in the source, no vendor check, no driver name
anywhere, and nothing that branches on which driver answered. The producer
takes whatever the loader hands it. It queries capabilities and fails clearly
when one is missing, which is what any portable application does:

- `vkGetPhysicalDeviceExternalBufferProperties` must report `OPAQUE_FD` as
  exportable, or it exits 3.
- Timeline mode needs `VK_KHR_timeline_semaphore` and
  `VK_KHR_external_semaphore_fd`. If either is absent the producer offers only
  host sync and runs anyway.
- Nothing above Vulkan 1.1 core is used.

**The driver is chosen outside the application**, by `VK_DRIVER_FILES`.
`run_interop.py` pins it rather than inheriting it, because an exported
`VK_DRIVER_FILES` is normal in this tree and would otherwise decide the
experiment silently:

    ./run_interop.py                       # the default ICD
    ./run_interop.py --icd /path/to/other  # the same binary, another driver

Running the identical binary against a second ICD and getting the identical
result is the demonstration. The sample cannot participate in it, because it
cannot tell.

## Why it is shaped this way

**Two processes, not one.** In one address space the driver clobbers the
caller's CUDA context, the two allocators compete, and an out-of-range
`bindless_image_store` writes into whatever is there -- which, in one process,
is the model weights. Two processes remove all three, and the fd already works
across them.

**The producer never links libcuda.** One integer crosses the process boundary
and everything CUDA-shaped lives on the far side of it. If the producer linked
libcuda it would quietly be demonstrating something weaker.

**Vulkan owns all shared memory, including the result buffer.** The fd only
travels one way: torch cannot export its allocator's memory, because the only
call that makes a handle from an existing pointer supports `DMA_BUF_FD` alone,
and that is unsupported on this GeForce. Ownership is decided by the transport.

**A buffer, not an image.** An optimal-tiled `VkImage` imports as an opaque
`CUmipmappedArray`. A buffer arrives as a flat pointer that wraps into a tensor
directly.

## What it renders

One fullscreen triangle from `gl_VertexIndex`, no vertex buffer, no texture,
no depth. The colour of a pixel is an exact function of `x`, `y` and the frame
number, so both sides can predict every byte with integer arithmetic. No golden
image, no tolerance. The frame number is stamped into the first two pixels, so
reading a stale slot fails loudly instead of looking fine.

The sample is about the handover. Anything else it drew would be another way to
fail for a reason that has nothing to do with interop.

## How a failure reads

The result buffer is filled with `0xCD` before every handover, so a byte that
was never written is visible rather than plausible. On failure the producer
writes the input, the result, the expectation and a diff as PNGs. The shape of
the damage names the bug faster than the counter does:

| the image looks like | what it means |
|---|---|
| solid `0xCD` | torch never wrote: wrong slot, or `dst = y` |
| top correct, bottom poison | read too early, the write had not landed |
| a clean but wrong frame | stale slot, off by one in the ring |
| correct then a poison tail | variable-shape result, `K` smaller than declared |

## The negative controls

`run_interop.py` runs `--no-consumer-sync` and `--no-producer-wait` **expecting
them to fail**. If either passes, the runner reports that as a failure of the
test. A synchronisation test that has never been seen to fail is not evidence.

## Reading

- `PROTOCOL.md` -- the wire contract, the memory layout, the pattern function.
- `docs/cudavk/CUDA_INTEROP.md` -- what CUDA permits, measured on this machine,
  and why each decision here went the way it did.
