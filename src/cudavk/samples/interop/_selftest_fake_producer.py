#!/usr/bin/env python3
"""_selftest_fake_producer.py -- end-to-end test of the consumer without Vulkan.

TEST SCAFFOLD.  Not part of the sample: nothing in the shipped path imports
it, and the runner does not use it.  It exists so the consumer can be tested
without the producer.

The real producer is C and Vulkan.  This one is Python and the CUDA VMM: it
allocates with cuMemCreate(requestedHandleTypes =
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) + cuMemAddressReserve + cuMemMap +
cuMemSetAccess, exports an fd with cuMemExportToShareableHandle and sends it
over SCM_RIGHTS.  That fd imports as CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
which is the same import path a Vulkan vkGetMemoryFdKHR fd takes
(docs/cudavk/CUDA_INTEROP.md 1.1, measured).

So this exercises the real consumer process, the real socket handshake, the
real import, the real wrap, the real write-back and the real bounds checks --
everything except Vulkan itself.

    /home/alexzhukov/mesa/venv/bin/python3 _selftest_fake_producer.py

It is a test scaffold, not part of the sample's shipped path.
"""

import argparse
import ctypes
import os
import socket
import struct
import subprocess
import sys
import time

import numpy as np

import cudabind as cb
from cpvk_interop_torch import (MAGIC, MSG_OFFER, MSG_REQUEST, MSG_SLOT, MSG_SEMS,
                                MSG_HELLO, MSG_READY, MSG_RESULT, MSG_BYE, MSG_ERROR,
                                F_OFFER, F_REQUEST, F_SLOT, F_HELLO, F_READY,
                                F_RESULT, F_BYE, F_ERROR, unpack_error)

POISON = 0xCD
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    sys.stderr.write("fakeprod: %s\n" % msg)
    sys.stderr.flush()


def frame_pattern(w, h, seq):
    """The one function that is written three times: GLSL, C and here."""
    x = np.arange(w, dtype=np.int64)[None, :]
    y = np.arange(h, dtype=np.int64)[:, None]
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = ((x + seq) & 255).astype(np.uint8)
    img[:, :, 1] = ((y + 2 * seq) & 255).astype(np.uint8)
    img[:, :, 2] = ((x ^ y) & 255).astype(np.uint8)
    img[:, :, 3] = 255
    flat = img.reshape(-1)
    flat[:8] = np.frombuffer(struct.pack("<II", seq & 0xFFFFFFFF,
                                         ~seq & 0xFFFFFFFF), dtype=np.uint8)
    return img


def align_up(v, a):
    return (v + a - 1) // a * a


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--op", default="add1")
    ap.add_argument("--consumer-arg", action="append", default=[],
                    help="extra argument for the consumer, repeatable")
    ap.add_argument("--expect-fail", action="store_true",
                    help="the run is a negative control; a clean pass is the bug")
    ap.add_argument("--stale-frame", type=int, default=0, metavar="SEQ",
                    help="fault injection: render frame SEQ with the pattern of "
                         "SEQ-1, so the consumer must reject it")
    ap.add_argument("--wrong-uuid", action="store_true",
                    help="fault injection: offer a device UUID no CUDA device has")
    ap.add_argument("--shrink-result", action="store_true",
                    help="fault injection: declare a result range smaller than "
                         "the consumer asked for")
    ap.add_argument("--vk-timeline", action="store_true",
                    help="offer timeline sync, backed by two real exported "
                         "VK_SEMAPHORE_TYPE_TIMELINE semaphores created through "
                         "libvulkan.so.1.  Needs no renderer: the frame is "
                         "written by an async CUDA copy and sem_render is "
                         "signalled on that stream.")
    ap.add_argument("--race-fill", type=int, default=200,
                    help="with --vk-timeline, how many async 0xAB fills of the "
                         "frame to enqueue before the real content.  A consumer "
                         "that does not wait on sem_render reads 0xAB.")
    ap.add_argument("--stall", type=float, default=0.0, metavar="SECS",
                    help="fault injection: sleep this long before MSG_READY of "
                         "frame 2, to make the consumer's per-frame timeout fire")
    ap.add_argument("--die-after", type=int, default=0, metavar="N",
                    help="fault injection: exit without MSG_BYE after N frames")
    ap.add_argument("--peek-race", action="store_true",
                    help="with --vk-timeline, read the frame back on the null "
                         "stream right after MSG_READY.  It should still be "
                         "0xAB, which is the proof that the frame really is in "
                         "flight when the consumer is told about it.")
    ap.add_argument("--offer-timeline", action="store_true",
                    help="offer sync_modes = host|timeline.  This producer has "
                         "no Vulkan, so if the consumer picks timeline it is "
                         "sent two fds that are not semaphores and must refuse "
                         "them cleanly.")
    ap.add_argument("--listen", metavar="PATH",
                    help="bind PATH and launch the consumer with --socket PATH "
                         "instead of passing an inherited fd")
    args = ap.parse_args(argv[1:])

    w, h = args.width, args.height
    frame_bytes = w * h * 4
    need = frame_bytes
    tail = 1024                       # extra result room, must stay poison
    frame_offset = 0
    result_offset = align_up(frame_bytes, 256)
    result_bytes = need + tail

    cb.init()
    dev = cb.device_get(0)
    uuid = cb.device_uuid(dev)
    vt = cu_sem_render = None
    stream = golden = None
    if args.vk_timeline:
        import _selftest_vk_sem
        vt = _selftest_vk_sem.VkTimeline(want_uuid=uuid)
        vk_render = vt.create_timeline()
        vk_consume = vt.create_timeline()
        log("VkPhysicalDevice with uuid %s has two timeline semaphores"
            % vt.uuid.hex())
    ctx = cb.primary_ctx_retain(dev)
    cb.check(cb._get("cuCtxSetCurrent")(cb.CUcontext(ctx)), "cuCtxSetCurrent")
    log("device 0 %s uuid %s" % (cb.device_name(dev), uuid.hex()))

    allocs = []
    for _ in range(args.slots):
        allocs.append(cb.VmmExport(result_offset + result_bytes, 0))
    alloc_size = allocs[0].size
    log("%d slots of %d bytes (granularity %d), frame %d @ %d, result %d @ %d"
        % (args.slots, alloc_size, allocs[0].granularity, frame_bytes,
           frame_offset, result_bytes, result_offset))

    base = [sys.executable, os.path.join(HERE, "cpvk_interop_torch.py"),
            "--op", args.op] + args.consumer_arg
    if args.listen:
        if os.path.exists(args.listen):
            os.unlink(args.listen)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        srv.bind(args.listen)
        srv.listen(1)
        cmd = base + ["--socket", args.listen]
        log("spawning %s" % " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=HERE)
        srv.settimeout(60.0)
        parent, _ = srv.accept()
        srv.close()
        os.unlink(args.listen)
    else:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        os.set_inheritable(child.fileno(), True)
        cmd = base + [str(child.fileno())]
        log("spawning %s" % " ".join(cmd))
        proc = subprocess.Popen(cmd, pass_fds=(child.fileno(),), cwd=HERE)
        child.close()
    parent.settimeout(60.0)

    def send(data):
        n = parent.send(data)
        assert n == len(data)

    def recv(fmt, want_type):
        data, fds, flags, _ = socket.recv_fds(parent, 65536, 4)
        if not data:
            raise RuntimeError("consumer closed the socket (exit %s)"
                               % proc.poll())
        mtype = struct.unpack_from("<2I", data, 0)[1]
        if mtype == MSG_ERROR and want_type != MSG_ERROR:
            raise RuntimeError("consumer sent MSG_ERROR: %s"
                               % (unpack_error(data),))
        if mtype != want_type:
            raise RuntimeError("wanted type %d, got %d" % (want_type, mtype))
        return struct.unpack(fmt, data[:struct.calcsize(fmt)])

    failures = []
    try:
        offered_uuid = uuid
        if args.wrong_uuid:
            offered_uuid = bytes([uuid[0] ^ 0xFF]) + uuid[1:]
        modes = 1
        if args.offer_timeline:
            modes = 3
        if args.vk_timeline:
            modes = 3 if args.offer_timeline else 2
        send(struct.pack(F_OFFER, MAGIC, MSG_OFFER, 1, args.slots, w, h, 0,
                         frame_bytes, modes, offered_uuid, 0, 0))
        req = recv(F_REQUEST, MSG_REQUEST)
        (_, _, r_bytes, r_dtype, r_ndim, s0, s1, s2, s3, sync_mode, _) = req
        log("REQUEST result_bytes=%d dtype=%d ndim=%d shape=(%d,%d,%d,%d) sync=%d"
            % (r_bytes, r_dtype, r_ndim, s0, s1, s2, s3, sync_mode))
        if r_bytes > result_bytes:
            raise RuntimeError("consumer wants %d result bytes, slot has %d"
                               % (r_bytes, result_bytes))
        if not sync_mode & modes:
            raise RuntimeError("consumer picked sync_mode %d, offered 0x%x"
                               % (sync_mode, modes))

        for i, a in enumerate(allocs):
            fd = a.export_fd()
            declared = result_bytes - (need + 1) if args.shrink_result \
                else result_bytes
            msg = struct.pack(F_SLOT, MAGIC, MSG_SLOT, i, 0, a.size,
                              frame_offset, frame_bytes, result_offset,
                              declared)
            socket.send_fds(parent, [msg], [fd])
            os.close(fd)          # the producer's copy, after a good sendmsg
        if sync_mode == 2 and vt is not None:
            # One exported fd per semaphore for the consumer.  vkGetSemaphoreFdKHR
            # hands out a new reference each call, so the producer exports a
            # third fd for its own CUDA-side import of sem_render.
            fd_r = vt.export_fd(vk_render)
            fd_c = vt.export_fd(vk_consume)
            socket.send_fds(parent, [struct.pack("<4I", MAGIC, MSG_SEMS, 0, 0)],
                            [fd_r, fd_c])
            os.close(fd_r)
            os.close(fd_c)
            cu_sem_render = cb.import_external_semaphore(
                vt.export_fd(vk_render),
                cb.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD)
            stream = cb.stream_create()
            golden = cb.mem_alloc(frame_bytes)
            log("sent sem_render and sem_consume; producer signals sem_render "
                "from a CUDA stream")
        elif sync_mode == 2:
            # No Vulkan semaphore here, so send two fds that are not semaphores:
            # cuImportExternalSemaphore must reject them and the consumer must
            # die cleanly, not crash.
            log("consumer chose timeline; sending two non-semaphore fds, which "
                "it must refuse")
            f0 = os.open("/dev/null", os.O_RDWR)
            f1 = os.open("/dev/null", os.O_RDWR)
            socket.send_fds(parent, [struct.pack("<4I", MAGIC, MSG_SEMS, 0, 0)],
                            [f0, f1])
            os.close(f0)
            os.close(f1)
        hello = recv(F_HELLO, MSG_HELLO)
        log("HELLO status %d" % hello[2])
        if hello[2] != 0:
            raise RuntimeError("consumer HELLO status %d" % hello[2])

        parent.settimeout(10.0)
        host = np.empty(result_bytes, dtype=np.uint8)
        t_start = time.perf_counter()
        for seq in range(1, args.frames + 1):
            slot = (seq - 1) % args.slots
            a = allocs[slot]
            if args.stall and seq == 2:
                log("stalling %.1f s before MSG_READY; the consumer should time "
                    "out" % args.stall)
                time.sleep(args.stall)
            img = frame_pattern(w, h, seq - 1 if seq == args.stale_frame else seq)
            if cu_sem_render is not None:
                # The frame is written asynchronously and sem_render is signalled
                # on the same stream, so the data is genuinely in flight when
                # MSG_READY is sent.  0xAB is what a consumer that skips the
                # wait will read.
                cb.memcpy_htod(golden, img.ctypes.data, frame_bytes)
                for _ in range(args.race_fill):
                    cb.memset_d8_async(a.ptr + frame_offset, 0xAB, frame_bytes,
                                       stream)
                cb.memset_d8_async(a.ptr + result_offset, POISON, result_bytes,
                                   stream)
                cb.memcpy_dtod_async(a.ptr + frame_offset, golden, frame_bytes,
                                     stream)
                cb.signal_external_semaphore(cu_sem_render, seq, stream)
                send(struct.pack(F_READY, MAGIC, MSG_READY, slot, seq, seq))
                if args.peek_race:
                    # Deliberately unsynchronised: the fills are on a
                    # NON_BLOCKING stream, so the legacy null stream this copy
                    # runs on does not wait for them.
                    peek = np.empty(16, dtype=np.uint8)
                    cb.memcpy_dtoh(peek.ctypes.data, a.ptr + frame_offset, 16)
                    log("seq %d: frame at MSG_READY time reads %s"
                        % (seq, peek[:8].tolist()))
            else:
                cb.memcpy_htod(a.ptr + frame_offset, img.ctypes.data, frame_bytes)
                cb.memset_d8(a.ptr + result_offset, POISON, result_bytes)
                cb.ctx_synchronize()
                send(struct.pack(F_READY, MAGIC, MSG_READY, slot, seq, 0))
            res = recv(F_RESULT, MSG_RESULT)
            (_, _, r_slot, r_seq, status, ndim, a0, a1, a2, a3,
             written, tv) = res
            if (r_slot, r_seq, status) != (slot, seq, 0):
                raise RuntimeError("RESULT slot/seq/status %s, wanted %s"
                                   % ((r_slot, r_seq, status), (slot, seq, 0)))
            if written > result_bytes:
                raise RuntimeError("consumer claims %d bytes, slot holds %d"
                                   % (written, result_bytes))
            if cu_sem_render is not None:
                if tv != seq:
                    raise RuntimeError("RESULT timeline_value %d, wanted %d"
                                       % (tv, seq))
                if not vt.wait(vk_consume, seq, 2 * 10 ** 9):
                    raise RuntimeError("sem_consume never reached %d (it is at "
                                       "%d): the consumer did not signal"
                                       % (seq, vt.counter(vk_consume)))
            cb.memcpy_dtoh(host.ctypes.data, a.ptr + result_offset, result_bytes)
            cb.ctx_synchronize()

            got = struct.unpack("<II", host[:8].tobytes())
            want = (seq & 0xFFFFFFFF, ~seq & 0xFFFFFFFF)
            if got != want:
                failures.append("seq %d: result stamp %s, expected %s"
                                % (seq, got, want))
            body = host[8:written]
            if args.op == "add1":
                expect = ((img.reshape(-1)[8:written].astype(np.int32) + 1) & 255) \
                    .astype(np.uint8)
                bad = int(np.count_nonzero(body != expect))
                if bad:
                    where = int(np.argmax(body != expect)) + 8
                    failures.append("seq %d: %d of %d result bytes wrong, first "
                                    "at %d: got %d want %d"
                                    % (seq, bad, body.size, where, body[where - 8],
                                       expect[where - 8]))
            else:
                if np.all(body == POISON):
                    failures.append("seq %d: the whole result is still poison"
                                    % seq)
            tail_bytes = host[written:]
            if tail_bytes.size and not np.all(tail_bytes == POISON):
                failures.append("seq %d: %d tail bytes past result_bytes were "
                                "overwritten"
                                % (seq, int(np.count_nonzero(tail_bytes != POISON))))
            if args.die_after and seq >= args.die_after:
                log("closing the socket without MSG_BYE")
                parent.close()
                break
        dt = time.perf_counter() - t_start
        log("%d frames in %.1f ms (%.2f ms/frame, includes the host round trip "
            "and a full device->host readback)"
            % (args.frames, dt * 1e3, dt * 1e3 / args.frames))
        send(struct.pack(F_BYE, MAGIC, MSG_BYE, args.frames, 0))
    except Exception as exc:
        log("PRODUCER ERROR: %r" % (exc,))
        failures.append("producer: %r" % (exc,))
        try:
            parent.send(struct.pack(F_ERROR, MAGIC, MSG_ERROR, 2, len(str(exc)),
                                    str(exc).encode()[:255]))
        except Exception:
            pass

    try:
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        failures.append("consumer had to be killed")
    log("consumer exit code %d" % rc)
    if cu_sem_render is not None:
        cb.destroy_external_semaphore(cu_sem_render)
        cb.stream_destroy(stream)
        cb.mem_free(golden)
    if vt is not None:
        vt.close()

    for f in failures:
        log("FAIL " + f)
    ok = (rc == 0 and not failures)
    if args.expect_fail:
        if ok:
            log("NEGATIVE CONTROL PASSED CLEANLY -- the test is broken")
            return 1
        log("negative control failed as required")
        return 0
    log("SELF TEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
