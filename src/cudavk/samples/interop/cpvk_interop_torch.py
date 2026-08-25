#!/usr/bin/env python3
"""cpvk_interop_torch.py -- the PyTorch consumer half of the interop sample.

It receives exported buffer file descriptors over an AF_UNIX SOCK_SEQPACKET
connection, imports them as CUDA external memory, wraps the mapped pointers as
zero-copy torch tensors, runs an op per frame and writes the result back into
the shared buffer.  The wire protocol is PROTOCOL.md in this directory; it is
frozen and this file follows it.

Run it the way the producer does:

    cpvk_interop_render --exec "cpvk_interop_torch.py --op add1"

which passes the inherited socket fd number as the last argument.  For
debugging without the producer:

    cpvk_interop_torch.py --socket /tmp/cpvk_interop.sock

The rules that are not obvious, each one recorded in CUDA_INTEROP.md in this
tree's docs as something that already went wrong here:

  * torch has no external-memory API.  ctypes against libcuda.so.1 is the API.
  * torch.zeros(1, device="cuda") must run BEFORE cuImportExternalMemory, so
    the mapping lands in the primary context torch keeps for the process life.
    Importing under a context that later dies returns CUDA_SUCCESS and then
    SIGSEGV.
  * the CUDA device is matched by UUID, never by index.
  * CUDA owns an imported fd; a successfully imported fd is never closed here.
  * import once at startup (~530 us each); re-wrapping a pointer costs ~1 us.
  * __cuda_array_interface__ does not bounds-check, so every wrap is checked
    against the byte range the producer declared.
  * record_stream() is a silent no-op on imported memory; events and imported
    semaphores are the only ordering primitives that work.
  * the result must be WRITTEN INTO the imported buffer (out= or copy_).
    Rebinding the name would leave the shared buffer full of poison.
"""

import argparse
import ctypes
import os
import signal
import socket
import struct
import sys
import time

import cudabind as cb

# ----------------------------------------------------------------- protocol
MAGIC = 0x4F495043           # 'CPIO'

MSG_OFFER, MSG_REQUEST, MSG_SLOT, MSG_SEMS = 1, 2, 3, 4
MSG_HELLO, MSG_READY, MSG_RESULT, MSG_BYE, MSG_ERROR = 5, 6, 7, 8, 9

# Every struct is little-endian and packed.  Sizes below are what PROTOCOL.md
# describes field by field; all 64-bit fields happen to be naturally aligned,
# so a packed and an unpacked C struct agree on every one of them.
F_OFFER = "<9I16sII"          # 60
F_REQUEST = "<2IQ2I4I2I"      # 48
F_SLOT = "<4I5Q"              # 56
F_SEMS = "<4I"                # 16
F_HELLO = "<4I"               # 16
F_READY = "<4IQ"              # 24
F_RESULT = "<6I4I2Q"          # 56
F_BYE = "<4I"                 # 16
F_ERROR = "<4I256s"           # 272

SYNC_HOST, SYNC_TIMELINE = 1, 2

DTYPE_U8, DTYPE_F16, DTYPE_F32, DTYPE_I32 = 0, 1, 2, 3

VK_FORMAT_R8G8B8A8_UINT = 0   # OFFER.frame_format encoding, per PROTOCOL.md

EXIT_OK, EXIT_VERIFY, EXIT_PROTOCOL, EXIT_UNSUPPORTED = 0, 1, 2, 3

HANDSHAKE_TIMEOUT = 60.0
FRAME_TIMEOUT = 2.0


class ProtocolError(Exception):
    """Anything that means the peer or the wire is wrong.  Exit code 2."""


class VerifyError(Exception):
    """The frame did not match the pattern.  Exit code 1."""


class Verdict(Exception):
    """The peer reported a verdict and we mirror its exit code.

    PROTOCOL.md: a status or MSG_ERROR code is the exit code the sender is
    about to use.  A verification failure the producer found is not a protocol
    error here; reporting it as one would make a negative control fail for the
    wrong reason.
    """

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def log(msg):
    sys.stderr.write("consumer: %s\n" % msg)
    sys.stderr.flush()


# --------------------------------------------------------------- transport
class Conn:
    """One SOCK_SEQPACKET connection.  One sendmsg is one message."""

    def __init__(self, sock):
        self.sock = sock
        self.state = "starting"

    def set_timeout(self, seconds):
        self.sock.settimeout(seconds)

    def send(self, data):
        try:
            n = self.sock.send(data)
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ProtocolError("peer died while sending (%s)" % exc)
        if n != len(data):
            raise ProtocolError("short send: %d of %d" % (n, len(data)))

    def recv(self, expect_fds=0):
        """Return (msg_type, payload, fds).  Raises on EOF, timeout, bad magic."""
        try:
            data, fds, flags, _ = socket.recv_fds(self.sock, 65536, max(expect_fds, 4))
        except socket.timeout:
            raise ProtocolError("timeout after %.1fs while %s"
                                % (self.sock.gettimeout(), self.state))
        except (ConnectionResetError, OSError) as exc:
            raise ProtocolError("peer died while receiving (%s)" % exc)
        if flags & socket.MSG_TRUNC:
            for fd in fds:
                os.close(fd)
            raise ProtocolError("truncated message while %s" % self.state)
        if not data:
            for fd in fds:
                os.close(fd)
            raise ProtocolError("EOF while %s" % self.state)
        if len(data) < 8:
            raise ProtocolError("runt message of %d bytes while %s"
                                % (len(data), self.state))
        magic, mtype = struct.unpack_from("<2I", data, 0)
        if magic != MAGIC:
            raise ProtocolError("bad magic 0x%08x while %s" % (magic, self.state))
        return mtype, data, fds

    def expect(self, want, fmt, expect_fds=0):
        mtype, data, fds = self.recv(expect_fds)
        if mtype == MSG_ERROR and want != MSG_ERROR:
            code, text = unpack_error(data)
            for fd in fds:
                os.close(fd)
            raise ProtocolError("producer sent MSG_ERROR %d: %s" % (code, text))
        if mtype != want:
            for fd in fds:
                os.close(fd)
            raise ProtocolError("expected message type %d, got %d while %s"
                                % (want, mtype, self.state))
        need = struct.calcsize(fmt)
        if len(data) < need:
            for fd in fds:
                os.close(fd)
            raise ProtocolError("message type %d is %d bytes, need %d"
                                % (mtype, len(data), need))
        if len(data) != need:
            log("warning: message type %d is %d bytes, expected %d; using the "
                "leading %d" % (mtype, len(data), need, need))
        if len(fds) != expect_fds:
            for fd in fds:
                os.close(fd)
            raise ProtocolError("message type %d carried %d fds, expected %d"
                                % (mtype, len(fds), expect_fds))
        return struct.unpack(fmt, data[:need]), fds


def unpack_error(data):
    need = struct.calcsize(F_ERROR)
    if len(data) < need:
        return -1, "<malformed MSG_ERROR>"
    _, _, code, text_len, text = struct.unpack(F_ERROR, data[:need])
    text_len = min(text_len, 256)
    return code, text[:text_len].split(b"\0")[0].decode("utf-8", "replace")


def pack_error(code, text):
    raw = text.encode("utf-8")[:255]
    return struct.pack(F_ERROR, MAGIC, MSG_ERROR, code, len(raw), raw)


# ------------------------------------------------------------ frame layout
class Slot:
    def __init__(self, index, alloc_size, frame_offset, frame_bytes,
                 result_offset, result_bytes):
        self.index = index
        self.alloc_size = alloc_size
        self.frame_offset = frame_offset
        self.frame_bytes = frame_bytes
        self.result_offset = result_offset
        self.result_bytes = result_bytes
        self.extmem = None
        self.frame_ptr = 0
        self.result_ptr = 0
        self.frame = None       # torch tensor, wrapped once
        self.result = None      # torch tensor, wrapped once


# ------------------------------------------------------------- the consumer
class Consumer:
    def __init__(self, conn, args):
        self.conn = conn
        self.args = args
        self.torch = None
        self.slots = []
        self.sem_render = None
        self.sem_consume = None
        self.stream = 0
        self.sync_mode = SYNC_HOST
        self.width = self.height = 0
        self.frames = 0
        self.last_seq = 0
        self.pending = []       # verifications in flight, oldest first
        self.failures = 0
        self.first_failure = None
        self.import_us = []

    # -- phase 1/2 ---------------------------------------------------------
    def handshake(self):
        torch = self.torch
        conn = self.conn
        conn.set_timeout(HANDSHAKE_TIMEOUT)

        conn.state = "waiting MSG_OFFER"
        (fields, _) = conn.expect(MSG_OFFER, F_OFFER)
        (_, _, version, nslots, width, height, frame_format, frame_bytes,
         sync_modes, uuid, scene, _pad) = fields
        if version != 1:
            raise ProtocolError("OFFER version %d, this consumer speaks 1" % version)
        if not 1 <= nslots <= 8:
            raise ProtocolError("OFFER slots = %d, must be 1..8" % nslots)
        if frame_format != VK_FORMAT_R8G8B8A8_UINT:
            raise ProtocolError("OFFER frame_format %d, only 0 "
                                "(VK_FORMAT_R8G8B8A8_UINT) is implemented"
                                % frame_format)
        if frame_bytes != width * height * 4:
            raise ProtocolError("OFFER frame_bytes %d != %d*%d*4"
                                % (frame_bytes, width, height))
        self.width, self.height, self.nslots = width, height, nslots
        self.frame_bytes = frame_bytes
        log("OFFER slots=%d %dx%d frame_bytes=%d sync_modes=0x%x scene=%d uuid=%s"
            % (nslots, width, height, frame_bytes, sync_modes, scene, uuid.hex()))

        # -- device by UUID, never by index --------------------------------
        cb.init()
        ordinal, dev = cb.find_device_by_uuid(uuid)
        if ordinal is None:
            have = ", ".join("%d:%s" % (i, cb.device_uuid(cb.device_get(i)).hex())
                             for i in range(cb.device_count()))
            raise ProtocolError("no CUDA device has UUID %s; this process sees %s"
                                % (uuid.hex(), have or "<none>"))
        log("device %d %s matches the offered UUID" % (ordinal, cb.device_name(dev)))
        self.dev_ordinal = ordinal

        # -- torch first, then import.  This is the rule that matters. -----
        torch.cuda.set_device(ordinal)
        torch.zeros(1, device="cuda")
        self.cuda_ctx = cb.ctx_get_current()
        primary = cb.primary_ctx_retain(dev)
        if self.cuda_ctx != primary:
            log("warning: current context 0x%x is not the primary context 0x%x"
                % (self.cuda_ctx or 0, primary or 0))
        else:
            log("current context 0x%x is the device primary context" % primary)
        cb.primary_ctx_release(dev)
        self.stream = torch.cuda.current_stream().cuda_stream

        # -- sync mode -----------------------------------------------------
        forced = os.environ.get("CPVK_INTEROP_SYNC")
        if forced:
            want = {"host": SYNC_HOST, "timeline": SYNC_TIMELINE}.get(forced.lower())
            if want is None:
                raise ProtocolError("CPVK_INTEROP_SYNC=%r, want host or timeline"
                                    % forced)
            if not sync_modes & want:
                raise ProtocolError("CPVK_INTEROP_SYNC=%s but the producer offers "
                                    "only 0x%x" % (forced, sync_modes))
            self.sync_mode = want
        elif sync_modes & SYNC_TIMELINE:
            self.sync_mode = SYNC_TIMELINE
        elif sync_modes & SYNC_HOST:
            self.sync_mode = SYNC_HOST
        else:
            raise ProtocolError("OFFER sync_modes = 0x%x offers nothing" % sync_modes)
        log("sync mode: %s" % ("timeline" if self.sync_mode == SYNC_TIMELINE
                               else "host"))

        # -- phase 2 -------------------------------------------------------
        h, w = self.height, self.width
        result_bytes = h * w * 4
        self.result_shape = (h, w, 4)
        conn.send(struct.pack(F_REQUEST, MAGIC, MSG_REQUEST, result_bytes,
                              DTYPE_U8, 3, h, w, 4, 0, self.sync_mode, 0))
        log("REQUEST result_bytes=%d dtype=u8 shape=(%d,%d,4)" % (result_bytes, h, w))

        # -- phase 3, one MSG_SLOT per slot, one fd each --------------------
        for i in range(nslots):
            conn.state = "waiting MSG_SLOT %d of %d" % (i, nslots)
            (fields, fds) = conn.expect(MSG_SLOT, F_SLOT, expect_fds=1)
            (_, _, index, _p, alloc_size, frame_offset, fbytes,
             result_offset, rbytes) = fields
            fd = fds[0]
            slot = Slot(index, alloc_size, frame_offset, fbytes,
                        result_offset, rbytes)
            try:
                self.check_slot(slot)
            except ProtocolError:
                os.close(fd)      # never imported, so it is still ours
                raise
            self.import_slot(slot, fd)
            self.slots.append(slot)
        self.slots.sort(key=lambda s: s.index)
        if [s.index for s in self.slots] != list(range(nslots)):
            raise ProtocolError("slot indices %s are not 0..%d"
                                % ([s.index for s in self.slots], nslots - 1))
        log("imported %d slots, %s us each"
            % (nslots, ", ".join("%.0f" % t for t in self.import_us)))

        # -- phase 3b ------------------------------------------------------
        if self.sync_mode == SYNC_TIMELINE:
            conn.state = "waiting MSG_SEMS"
            (_fields, fds) = conn.expect(MSG_SEMS, F_SEMS, expect_fds=2)
            self.import_semaphores(fds)

        # -- warm up, then phase 4 -----------------------------------------
        self.build_expected()
        self.warmup()
        conn.send(struct.pack(F_HELLO, MAGIC, MSG_HELLO, 0, 0))
        log("HELLO sent, entering steady state")
        conn.set_timeout(FRAME_TIMEOUT)

    def check_slot(self, s):
        need = self.height * self.width * 4
        if s.frame_bytes < need:
            raise ProtocolError("slot %d frame_bytes %d < %d"
                                % (s.index, s.frame_bytes, need))
        if s.result_bytes < need:
            raise ProtocolError("slot %d result_bytes %d < the %d I asked for"
                                % (s.index, s.result_bytes, need))
        if s.frame_offset + s.frame_bytes > s.alloc_size:
            raise ProtocolError("slot %d frame range past alloc_size" % s.index)
        if s.result_offset + s.result_bytes > s.alloc_size:
            raise ProtocolError("slot %d result range past alloc_size" % s.index)
        lo = max(s.frame_offset, s.result_offset)
        hi = min(s.frame_offset + s.frame_bytes, s.result_offset + s.result_bytes)
        if lo < hi:
            raise ProtocolError("slot %d frame and result ranges overlap" % s.index)
        if s.result_offset % 256:
            raise ProtocolError("slot %d result_offset %d is not 256-aligned"
                                % (s.index, s.result_offset))

    def import_slot(self, s, fd):
        """One cuImportExternalMemory, two cuExternalMemoryGetMappedBuffer.

        CUDA takes ownership of a successfully imported fd, so it is not closed
        here.  An fd whose import failed is still ours and is closed.
        """
        t0 = time.perf_counter()
        try:
            s.extmem = cb.import_external_memory(fd, s.alloc_size)
        except cb.CudaError:
            os.close(fd)
            raise
        s.frame_ptr = cb.external_memory_get_mapped_buffer(
            s.extmem, s.frame_offset, s.frame_bytes)
        s.result_ptr = cb.external_memory_get_mapped_buffer(
            s.extmem, s.result_offset, s.result_bytes)
        self.import_us.append((time.perf_counter() - t0) * 1e6)
        s.frame = self.wrap(s.frame_ptr, self.result_shape, "|u1", s.frame_bytes)
        s.result = self.wrap(s.result_ptr, self.result_shape, "|u1", s.result_bytes)

    def wrap(self, ptr, shape, typestr, limit_bytes):
        """torch.as_tensor over __cuda_array_interface__.  Zero copy, checked."""
        iface = cb.CudaArrayIface(ptr, shape, typestr, limit_bytes)
        t = self.torch.as_tensor(iface, device="cuda")
        if t.data_ptr() != ptr:
            raise RuntimeError("wrap copied: data_ptr 0x%x != imported 0x%x"
                               % (t.data_ptr(), ptr))
        return t

    def import_semaphores(self, fds):
        kind = cb.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
        try:
            self.sem_render = cb.import_external_semaphore(fds[0], kind)
        except cb.CudaError:
            for fd in fds:
                os.close(fd)
            raise
        try:
            self.sem_consume = cb.import_external_semaphore(fds[1], kind)
        except cb.CudaError:
            os.close(fds[1])
            raise
        log("imported sem_render and sem_consume as TIMELINE_SEMAPHORE_FD")

    # -- verification ------------------------------------------------------
    def build_expected(self):
        """Precompute what does not depend on seq, once."""
        torch = self.torch
        h, w = self.height, self.width
        self.xs = torch.arange(w, device="cuda", dtype=torch.int32)
        self.ys = torch.arange(h, device="cuda", dtype=torch.int32)
        self.expected = torch.empty((h, w, 4), dtype=torch.uint8, device="cuda")
        self.expected[:, :, 2] = ((self.xs.view(1, w) ^ self.ys.view(h, 1)) & 255) \
            .to(torch.uint8)
        self.expected[:, :, 3] = 255

    @staticmethod
    def stamp_i64(seq):
        """The 8 stamp bytes -- seq then ~seq, u32 LE -- as one signed int64.

        Written with a single fill_ kernel, so no host-to-device copy and no
        pinned staging buffer whose reuse would need its own synchronisation.
        """
        v = (seq & 0xFFFFFFFF) | ((~seq & 0xFFFFFFFF) << 32)
        return v - (1 << 64) if v >= (1 << 63) else v

    def enqueue_verify(self, frame, seq):
        """Build the expected frame on the GPU and compare it there.

        Nothing here blocks the host.  The two counters and the frame's own
        stamp are copied back asynchronously into pinned memory and an event is
        recorded; `drain_ready` reads them once that event has completed.  A
        blocking read of the counters would be a host synchronisation, which
        would defeat timeline mode and, worse, would quietly repair
        --no-consumer-sync into a passing run.
        """
        torch = self.torch
        h, w = self.height, self.width
        exp = self.expected
        exp[:, :, 0] = ((self.xs + seq) & 255).to(torch.uint8)
        exp[:, :, 1] = ((self.ys + 2 * seq) & 255).to(torch.uint8).view(h, 1)
        flat_e = exp.view(-1)
        flat_f = frame.view(-1)
        stamp_err = (flat_f[:8].view(torch.int64) != self.stamp_i64(seq)).sum()
        patt_err = (flat_f[8:] != flat_e[8:]).sum()
        errs = torch.empty(2, dtype=torch.int64, pin_memory=True)
        stamp = torch.empty(8, dtype=torch.uint8, pin_memory=True)
        errs.copy_(torch.stack([stamp_err, patt_err]), non_blocking=True)
        stamp.copy_(flat_f[:8], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.pending.append((seq, errs, stamp, ev))

    def drain_ready(self):
        """Read back every verification whose event has already completed."""
        while self.pending and self.pending[0][3].query():
            self.report(*self.pending.pop(0)[:3])

    def drain_all(self):
        self.torch.cuda.synchronize()
        while self.pending:
            self.report(*self.pending.pop(0)[:3])

    def report(self, seq, errs, stamp):
        e = errs.tolist()
        if not (e[0] or e[1]):
            return
        self.failures += 1
        got = struct.unpack("<II", bytes(memoryview(stamp.numpy())))
        want = (seq & 0xFFFFFFFF, ~seq & 0xFFFFFFFF)
        msg = ("frame seq %d: stamp %s expected %s, %d of %d pattern bytes wrong"
               % (seq, got, want, e[1], self.height * self.width * 4 - 8))
        if self.first_failure is None:
            self.first_failure = msg
        if self.failures <= 8:
            log("VERIFY FAIL " + msg)

    # -- the op ------------------------------------------------------------
    def run_op(self, frame, dst, seq, expect_ptr=None):
        """Write the result INTO dst.  Never rebind dst.

        The op itself may allocate temporaries; the write-back must not.  The
        zero-copy claim is checked around the write-back only, which is where
        it means something: dst.data_ptr() is the imported pointer and
        torch.cuda.memory_allocated() does not move.
        """
        torch = self.torch
        y = None
        if self.args.op != "add1":
            x = frame.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            if self.args.op == "blur":
                y = torch.nn.functional.conv2d(x, self.blur_w, padding=1, groups=4)
            else:
                gx = torch.nn.functional.conv2d(x, self.sobel_x, padding=1, groups=4)
                gy = torch.nn.functional.conv2d(x, self.sobel_y, padding=1, groups=4)
                y = torch.sqrt(gx * gx + gy * gy)
            y = y.clamp(0, 255).round().squeeze(0).permute(1, 2, 0).to(torch.uint8)

        if self.args.stall_cycles:
            # Same stream as the op and the write-back, or it proves nothing.
            torch.cuda._sleep(self.args.stall_cycles)

        if expect_ptr is not None and dst.data_ptr() != expect_ptr:
            raise RuntimeError("dst.data_ptr() 0x%x is not the imported pointer 0x%x"
                               % (dst.data_ptr(), expect_ptr))
        before = torch.cuda.memory_allocated()
        if y is None:
            # uint8 arithmetic wraps, which is exactly (frame + 1) & 255.
            torch.add(frame, 1, out=dst)       # into the imported buffer
        else:
            dst.copy_(y)                       # into the imported buffer
        # The stamp, in one fill kernel: no host-to-device copy, no staging.
        dst.view(-1)[:8].view(torch.int64).fill_(self.stamp_i64(seq))
        after = torch.cuda.memory_allocated()
        self.alloc_before, self.alloc_after = before, after
        if after != before:
            raise RuntimeError("torch allocated %d bytes across the write-back; "
                               "the result did not go into the imported buffer"
                               % (after - before))

    def warmup(self):
        """Phase 4: keep CUDA init, cuDNN autotune and the first kernel launch
        out of frame 0.  Runs on torch-owned tensors, not on a shared slot."""
        torch = self.torch
        h, w = self.height, self.width
        if self.args.op == "blur":
            self.blur_w = torch.full((4, 1, 3, 3), 1.0 / 9.0, device="cuda")
        elif self.args.op == "sobel":
            kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                              device="cuda")
            self.sobel_x = kx.view(1, 1, 3, 3).repeat(4, 1, 1, 1)
            self.sobel_y = kx.t().contiguous().view(1, 1, 3, 3).repeat(4, 1, 1, 1)
        dummy_in = torch.zeros((h, w, 4), dtype=torch.uint8, device="cuda")
        dummy_out = torch.zeros((h, w, 4), dtype=torch.uint8, device="cuda")
        t0 = time.perf_counter()
        for _ in range(2):
            self.run_op(dummy_in, dummy_out, 0)
            self.enqueue_verify(dummy_in, 0)
        self.pending.clear()
        torch.cuda.synchronize()
        del dummy_in, dummy_out
        log("warmup done in %.1f ms" % ((time.perf_counter() - t0) * 1e3))

    # -- steady state ------------------------------------------------------
    def frame(self, slot_index, seq, timeline_value):
        torch = self.torch
        if not 0 <= slot_index < len(self.slots):
            raise ProtocolError("READY names slot %d of %d"
                                % (slot_index, len(self.slots)))
        s = self.slots[slot_index]
        self.last_seq = seq

        if self.sync_mode == SYNC_TIMELINE:
            # The GPU waits; the host does not.
            cb.wait_external_semaphore(self.sem_render, timeline_value or seq,
                                       self.stream)
            if self.args.no_consumer_sync:
                # The negative control for timeline mode is signalling TOO
                # EARLY, not not signalling at all.  Skipping the signal makes
                # the producer wait forever on a value that never arrives, so
                # the run dies as a timeout and tests the timeout.  Signalling
                # here, before the op is even enqueued, releases the slot while
                # the work is still pending -- the same failure shape as host
                # mode, and the producer must see poison.
                cb.signal_external_semaphore(self.sem_consume, seq, self.stream)

        # Bounds, every frame.  __cuda_array_interface__ never checks these.
        need = self.height * self.width * 4
        assert s.frame.numel() <= s.frame_bytes and need <= s.frame_bytes
        assert s.result.numel() <= s.result_bytes and need <= s.result_bytes
        assert s.frame.data_ptr() == s.frame_ptr
        assert s.result.data_ptr() == s.result_ptr

        self.enqueue_verify(s.frame, seq)
        self.run_op(s.frame, s.result, seq, expect_ptr=s.result_ptr)

        if self.sync_mode == SYNC_TIMELINE:
            if not self.args.no_consumer_sync:      # else already signalled, early
                cb.signal_external_semaphore(self.sem_consume, seq, self.stream)
            out_value = seq
        else:
            if not self.args.no_consumer_sync:
                ev = torch.cuda.Event()
                ev.record()
                ev.synchronize()
            out_value = 0

        if self.frames == 0:
            log("zero copy: frame tensor at 0x%x is the imported 0x%x, result "
                "tensor at 0x%x is the imported 0x%x, "
                "torch.cuda.memory_allocated() %d before and %d after the "
                "write-back"
                % (s.frame.data_ptr(), s.frame_ptr, s.result.data_ptr(),
                   s.result_ptr, self.alloc_before, self.alloc_after))
        self.conn.send(struct.pack(F_RESULT, MAGIC, MSG_RESULT, slot_index, seq,
                                   0, 3, self.height, self.width, 4, 0,
                                   need, out_value))
        self.frames += 1
        self.drain_ready()

    def loop(self):
        conn = self.conn
        while True:
            conn.state = "waiting MSG_READY, last seq %d" % self.last_seq
            mtype, data, fds = conn.recv()
            for fd in fds:
                os.close(fd)
            if mtype == MSG_READY:
                need = struct.calcsize(F_READY)
                if len(data) < need:
                    raise ProtocolError("MSG_READY is %d bytes" % len(data))
                _, _, slot, seq, tv = struct.unpack(F_READY, data[:need])
                conn.state = "running frame seq %d on slot %d" % (seq, slot)
                self.frame(slot, seq, tv)
            elif mtype == MSG_BYE:
                need = struct.calcsize(F_BYE)
                _, _, frames, status = struct.unpack(F_BYE, data[:need])
                log("BYE after %d producer frames, status %d (I ran %d)"
                    % (frames, status, self.frames))
                self.drain_all()
                if status != 0:
                    # PROTOCOL.md: a status is the exit code the sender is
                    # about to use. status 1 is the producer's own
                    # verification verdict, not a protocol failure, so mirror
                    # it rather than reporting exit 2 for the wrong reason.
                    raise Verdict(status, "producer reported status %d" % status)
                return
            elif mtype == MSG_ERROR:
                code, text = unpack_error(data)
                if code == EXIT_VERIFY:
                    raise Verdict(code, "producer sent MSG_ERROR %d: %s"
                                  % (code, text))
                raise ProtocolError("producer sent MSG_ERROR %d: %s" % (code, text))
            else:
                raise ProtocolError("unexpected message type %d in steady state"
                                    % mtype)


# --------------------------------------------------------------------- main
def open_socket(args):
    if args.socket:
        deadline = time.time() + 10.0
        last = None
        while time.time() < deadline:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                s.connect(args.socket)
                return s
            except OSError as exc:
                last = exc
                s.close()
                time.sleep(0.05)
        raise ProtocolError("cannot connect to %s (%s)" % (args.socket, last))
    if args.fd is None:
        raise ProtocolError("no socket: pass the inherited fd number as the last "
                            "argument, or use --socket PATH")
    return socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET, fileno=args.fd)


def main(argv):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    ap = argparse.ArgumentParser(
        description="PyTorch consumer for the interop sample")
    ap.add_argument("--op", choices=("add1", "blur", "sobel"), default="add1")
    ap.add_argument("--socket", help="connect to this AF_UNIX path instead of "
                                     "using an inherited fd")
    ap.add_argument("--no-consumer-sync", action="store_true",
                    help="negative control: release the slot without waiting "
                         "for the op to finish.  In host mode ev.synchronize() "
                         "is skipped; in timeline mode sem_consume is signalled "
                         "BEFORE the op instead of after.  The run is expected "
                         "to fail.")
    ap.add_argument("--stall-cycles", type=int, default=0, metavar="N",
                    help="enqueue torch.cuda._sleep(N) on the op's stream just "
                         "before the write-back, so the write is still pending "
                         "when the op call returns.  This makes both arms of "
                         "the synchronisation test deterministic instead of "
                         "lucky: with the stall, a pass proves the "
                         "synchronisation did the work and --no-consumer-sync "
                         "fails on every frame.  20000000 is about 7 ms here.")
    ap.add_argument("fd", nargs="?", type=int,
                    help="the inherited socket fd number, passed by the producer")
    args = ap.parse_args(argv[1:])

    if args.no_consumer_sync:
        log("NEGATIVE CONTROL: --no-consumer-sync, this run is expected to fail")
    if args.stall_cycles:
        log("stalling %d GPU cycles on the op stream before every write-back"
            % args.stall_cycles)

    conn = None
    try:
        sock = open_socket(args)
        conn = Conn(sock)
        c = Consumer(conn, args)
        t0 = time.perf_counter()
        conn.state = "importing torch"
        import torch                       # noqa: E402  (slow; after the socket)
        c.torch = torch
        log("torch %s, cuda %s, imported in %.2f s"
            % (torch.__version__, torch.version.cuda, time.perf_counter() - t0))
        c.handshake()
        c.loop()
    except Verdict as exc:
        # The producer already reported the verdict; mirror its exit code
        # instead of calling it a protocol error.
        log("producer verdict: %s" % exc)
        return exc.code
    except VerifyError as exc:
        log("VERIFICATION FAILURE: %s" % exc)
        if conn:
            try:
                conn.send(pack_error(1, str(exc)))
            except Exception:
                pass
        return EXIT_VERIFY
    except ProtocolError as exc:
        log("PROTOCOL ERROR: %s" % exc)
        if conn:
            try:
                conn.send(pack_error(2, str(exc)))
            except Exception:
                pass
        return EXIT_PROTOCOL
    except cb.CudaError as exc:
        log("CUDA ERROR: %s" % exc)
        if conn:
            try:
                conn.send(pack_error(2, str(exc)))
            except Exception:
                pass
        return EXIT_PROTOCOL
    except KeyboardInterrupt:
        return EXIT_PROTOCOL

    if c.failures:
        log("VERIFICATION FAILURE on %d of %d frames; first: %s"
            % (c.failures, c.frames, c.first_failure))
        return EXIT_VERIFY
    log("ok: %d frames, op=%s, %s sync, every frame verified"
        % (c.frames, args.op,
           "timeline" if c.sync_mode == SYNC_TIMELINE else "host"))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv))
