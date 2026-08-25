# The interop wire protocol

One AF_UNIX `SOCK_SEQPACKET` connection between two processes:

    cpvk_interop_render    producer, C, Vulkan only, never links libcuda
    cpvk_interop_torch.py  consumer, Python, ctypes against libcuda.so.1

`SOCK_SEQPACKET` preserves message boundaries, so there is no framing and no
length prefix. One `sendmsg` is one message. A short read is a protocol error,
not a partial message.

All integers are little-endian. All structs are packed and fixed size. Every
message begins with the same 8-byte prefix.

    u32 magic   = 0x4F495043  ('CPIO')
    u32 type

A message whose magic or type is wrong is fatal on both sides. Do not skip it.

## Ownership

The producer allocates everything both sides touch. The consumer imports.
This is forced by the transport: an fd only travels Vulkan -> CUDA.

**CUDA takes ownership of an imported fd.** CUDA 12.8 Programming Guide
3.2.16.1.2 and 3.2.16.1.5: *"CUDA assumes ownership of the file descriptor once
it is imported. Using the file descriptor after a successful import results in
undefined behavior."* So:

- the producer closes its fd after a successful `sendmsg`;
- the consumer never closes an fd it imported successfully;
- the consumer closes an fd whose import failed.

## Memory layout

**One `VkDeviceMemory` per slot**, holding two buffers at different offsets,
exported as **one** fd. The consumer calls `cuImportExternalMemory` once per
slot and `cuExternalMemoryGetMappedBuffer` twice on that handle, at the two
offsets.

    +--------------------------- alloc_size --------------------------+
    | frame  (frame_bytes)   | pad | result (result_bytes)   | pad     |
    ^ frame_offset (0)             ^ result_offset

`result_offset` is aligned to at least 256 bytes and to
`VkMemoryRequirements.alignment`. 256 is not cosmetic: a float32 or float16
view of a misaligned pointer is a fault, not a warning.

**No dedicated allocation.** Dedicated is per resource and two buffers share
this allocation, so `VkMemoryDedicatedAllocateInfo` is never used and
`CUDA_EXTERNAL_MEMORY_DEDICATED` is never set. A buffer does not need it.

`cuImportExternalMemory` takes `alloc_size`, the whole `VkDeviceMemory`, not
the buffer size.

## Phases

    1  producer -> consumer   MSG_OFFER      capabilities, device UUID
    2  consumer -> producer   MSG_REQUEST    result size, dtype, shape
    3  producer -> consumer   MSG_SLOT  x N  one per slot, one fd each
    3b producer -> consumer   MSG_SEMS       optional, two fds, timeline only
    4  consumer -> producer   MSG_HELLO      after import and warmup
    -- steady state --
    5  producer -> consumer   MSG_READY      per frame
    6  consumer -> producer   MSG_RESULT     per frame, implies free(slot)
    -- shutdown --
    7  producer -> consumer   MSG_BYE
       either direction       MSG_ERROR      fatal, carries text

Phase 2 exists so the producer never needs to know the model. The consumer
states its own output size, after learning the frame size in phase 1. A
2x upscaler cannot answer before it knows the input.

Phase 4 exists so torch import, CUDA init and cuDNN autotune do not land in
frame 0. The producer starts counting frames after `MSG_HELLO`.

## Messages

    MSG_OFFER    = 1      producer -> consumer
      u32 magic, u32 type
      u32 version         = 1
      u32 slots           1..8
      u32 width, u32 height
      u32 frame_format    0 = VK_FORMAT_R8G8B8A8_UINT
      u32 frame_bytes     width * height * 4
      u32 sync_modes      bit0 host, bit1 timeline
      u8  device_uuid[16] VkPhysicalDeviceIDProperties.deviceUUID
      u32 scene           0 = check pattern
      u32 pad

    MSG_REQUEST  = 2      consumer -> producer
      u32 magic, u32 type
      u64 result_bytes    maximum the consumer will ever write
      u32 result_dtype    0 u8, 1 f16, 2 f32, 3 i32
      u32 result_ndim     1..4
      u32 result_shape[4] maximum shape, unused dims 0
      u32 sync_mode       exactly one bit from OFFER.sync_modes
      u32 pad

    MSG_SLOT     = 3      producer -> consumer, carries 1 fd (SCM_RIGHTS)
      u32 magic, u32 type
      u32 slot
      u32 pad
      u64 alloc_size      pass this to cuImportExternalMemory
      u64 frame_offset, u64 frame_bytes
      u64 result_offset, u64 result_bytes

    MSG_SEMS     = 4      producer -> consumer, carries 2 fds, timeline only
      u32 magic, u32 type
      u32 pad0, u32 pad1
      fd[0] = sem_render   producer signals, consumer waits
      fd[1] = sem_consume  consumer signals, producer waits

    MSG_HELLO    = 5      consumer -> producer
      u32 magic, u32 type
      u32 status          0 = ready, non-zero = fatal
      u32 pad

    MSG_READY    = 6      producer -> consumer
      u32 magic, u32 type
      u32 slot, u32 seq
      u64 timeline_value  value of sem_render for this frame; 0 in host mode

    MSG_RESULT   = 7      consumer -> producer, implies free(slot)
      u32 magic, u32 type
      u32 slot, u32 seq
      u32 status          0 ok
      u32 result_ndim
      u32 result_shape[4] the shape actually written this frame
      u64 result_bytes    bytes actually written, <= SLOT.result_bytes
      u64 timeline_value  value signalled on sem_consume; 0 in host mode

    MSG_BYE      = 8      producer -> consumer
      u32 magic, u32 type
      u32 frames, u32 status

    MSG_ERROR    = 9      either direction
      u32 magic, u32 type
      u32 code, u32 text_len
      char text[256]      NUL padded

`MSG_RESULT` is both "here is the result" and "I am finished with the slot".
There is no separate free message. The producer needs both facts before it can
reuse the slot, so splitting them would only add a state.

## The frame

`VK_FORMAT_R8G8B8A8_UINT`, tightly packed, `width * height * 4` bytes, row
major, no padding. UINT rather than UNORM so the shader writes the byte
directly and unorm rounding never enters the comparison.

Pixel `(x, y)` of frame `seq`, for every pixel except the two stamp pixels:

    R = (x + seq)     & 255
    G = (y + 2 * seq) & 255
    B = (x ^ y)       & 255
    A = 255

Additions only. No subtraction, so there is no sign question in GLSL, C or
NumPy.

**The stamp.** Pixel `(0,0)` holds `seq` as u32 little-endian in its four
bytes. Pixel `(1,0)` holds `~seq`. Both sides skip the first 8 bytes when
checking the pattern. The stamp is what turns "1048576 bytes wrong" into
"stamp 41, expected 42", which names the bug instead of counting it.

This function is written three times -- GLSL, C, Python. It must agree
exactly. Any change is a change to all three.

## Poison

Before sending `MSG_READY`, the producer fills the whole result range with
`0xCD` (`vkCmdFillBuffer`).

The producer then checks, after `MSG_RESULT`:

- bytes `[0, result_bytes)` must be correct and must not be poison;
- bytes `[result_bytes, SLOT.result_bytes)` **must still be poison**.

The tail check is an overrun detector, and it is the reason the poison fill is
worth its cost. `__cuda_array_interface__` accepts a shape larger than the
buffer silently, so nothing else on the consumer side will catch a write past
the declared size.

## check mode

Default op, `--op add1`. dtype u8, shape `(height, width, 4)`,
`result_bytes = frame_bytes`.

    result[i] = (frame[i] + 1) & 255

except the first 8 bytes, which the consumer overwrites with `seq` and `~seq`
as u32 LE, the same stamp shape as the frame. The producer verifies the stamp,
then verifies bytes 8.. against `(f(x,y,seq) + 1) & 255`.

Every byte is predictable from `seq` alone, so there is no golden image, no
tolerance and no float comparison. A stale frame fails because the expected
bytes depend on `seq`.

## Synchronisation

**host mode.** Producer: submit, `vkWaitForFences`, send `MSG_READY`.
Consumer: run, `ev = torch.cuda.Event(); ev.record(); ev.synchronize()`, then
send `MSG_RESULT`. `timeline_value` is 0 in both messages.

**timeline mode.** Two exported `VK_SEMAPHORE_TYPE_TIMELINE` semaphores,
imported by the consumer as
`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD`.

    sem_render   producer signals `seq` when the frame is written
                 consumer waits `seq` on its torch stream
    sem_consume  consumer signals `seq` when it has finished with the slot
                 producer waits `seq - slots` before rendering frame `seq`

Sequence numbers start at 1, so for `seq <= slots` the producer waits a value
<= 0, which a timeline that starts at 0 already satisfies. No special case.

The consumer enqueues the wait and the signal on `torch.cuda.current_stream()
.cuda_stream` with `cuWaitExternalSemaphoresAsync` and
`cuSignalExternalSemaphoresAsync`. The host does not block. `MSG_RESULT` may
be sent immediately after the signal is enqueued.

A monotonic counter has no binary re-arm race, which is why this is a timeline
and not an event.

**host mode is the oracle for timeline mode.** Same frames, same checksums,
one flag apart. If the two disagree on frame 700 of 1000, that is the race.

## Negative controls

Two switches exist to make the test fail, and both are run in CI expecting
failure:

    --no-consumer-sync   host mode:     consumer skips ev.synchronize()
                         timeline mode: consumer signals sem_consume BEFORE
                                        the work instead of after
                         producer must then see poison, in both modes

    Do NOT implement the timeline variant as "skip the signal". The producer
    waits on that value, so skipping it hangs the run and the control then
    tests the timeout instead of the synchronisation.
    --no-producer-wait   producer reuses a slot without waiting for MSG_RESULT
                         consumer must then see a torn frame or a bad stamp

A synchronisation test that has never been observed to fail is not evidence.
If either of these passes, the test is broken, not the code.

`--slots 1` makes both sharper: the slot is reused on the very next frame, so
the race appears on frame 2 rather than frame 700.

## Timeouts

    handshake   60 s   torch import, CUDA init, N imports at ~530 us each
    per frame    2 s   a lost message or a stuck queue
    whole run    computed by the runner

Both sides keep a one-line state string and print it on timeout, for example
`renderer: waiting MSG_RESULT slot=1 seq=317`. A hang must say which side was
waiting and for what.

On EOF or `ECONNRESET` both sides exit non-zero immediately. Neither side ever
blocks forever on a dead peer. `SIGPIPE` is ignored and every write is checked.

## Exit codes

    0  pass
    1  verification failure
    2  protocol error, timeout, or peer died
    3  unsupported: the driver does not report OPAQUE_FD

Each side owns half the verdict. The consumer checks the frame it was given;
the producer checks the result that came back. The runner requires both to
exit 0.

## Message sizes, and the exact Python format

This section exists because it was missing, and its absence is the largest
interop risk between the C half and the Python half: if one side packs a field
differently, the very first message fails and the error will not say why.

Every message is naturally aligned, so a C struct with and without
`__attribute__((packed))` has the same layout. Every u64 sits at an offset that
is a multiple of 8. The C side asserts each size with `_Static_assert`; the
Python side must assert `struct.calcsize(FMT) == size`.

| message | bytes | Python `struct` format (little-endian, no padding) |
|---|---|---|
| `MSG_OFFER`   |  60 | `<9I16s2I` |
| `MSG_REQUEST` |  48 | `<2IQ2I4I2I` |
| `MSG_SLOT`    |  56 | `<4I5Q` |
| `MSG_SEMS`    |  16 | `<4I` |
| `MSG_HELLO`   |  16 | `<4I` |
| `MSG_READY`   |  24 | `<4IQ` |
| `MSG_RESULT`  |  56 | `<6I4I2Q` |
| `MSG_BYE`     |  16 | `<4I` |
| `MSG_ERROR`   | 272 | `<4I256s` |

The leading `<` is not optional. Without it Python uses native alignment and
`MSG_REQUEST` becomes a different size.

A received datagram whose length does not equal the size for its type is a
protocol error. Check the length before unpacking; do not unpack and hope.

## MSG_ERROR.code

`code` is **the exit code the sender is about to use.** So:

    code 1   the sender's own verification verdict: it checked its half and
             the data was wrong. The peer should also exit 1.
    code 2   protocol error, timeout, or the peer died.
    code 3   a capability is missing.

Without this rule a verification failure detected by one side arrives at the
other as "some fatal error" and gets reported as exit 2, so a negative control
that worked would be recorded as having failed for the wrong reason.

## Who chooses the sync mode

`MSG_OFFER.sync_modes` advertises **exactly the one mode the producer was
started in**, not everything it could do. The consumer still picks one bit from
that mask, so the rule in the message table is unchanged, but the two sides can
never disagree. If the producer was asked for timeline mode and the driver does
not have the extensions, it reports that, offers only the host bit, and runs.

## Two clarifications the first implementation needed

**"Must not be poison" is diagnostic, not a rule.** In check mode a correct
byte can legitimately be `0xCD`. The poison count is only meaningful where the
expected byte is not `0xCD`. Correctness comes from the byte comparison; the
poison count tells you which *kind* of failure you have.

**`MSG_OFFER.frame_bytes` is u32** while the other sizes are u64. That caps a
frame at 4 GiB, which is 32768x32768 RGBA8. The producer must refuse anything
larger rather than truncate. The asymmetry is recorded rather than fixed
because the layout is already asserted on both sides.

## One fact for a future Vulkan-side importer

`vkGetMemoryFdPropertiesKHR` is **not** valid for `OPAQUE_FD`. It returns
`VK_ERROR_INVALID_EXTERNAL_HANDLE` (-13). A Vulkan process importing one of
these fds must take its memory type from its own
`vkGetBufferMemoryRequirements`, because the handle is opaque and both sides
are the same physical device. CUDA has no equivalent problem: it takes
`alloc_size` and nothing else.

## One performance trap, measured

Verification reads the whole result range back through a mapped staging
buffer. A plain `HOST_VISIBLE|HOST_COHERENT` type is write-combined, and
reading it back cost **17.5 ms/frame at 128x128 with the GPU 1% busy**.
Preferring `HOST_CACHED` took the same test to **0.92-1.44 ms/frame at
512x512**. Any code that reads a mapped Vulkan pointer wants `HOST_CACHED`.
