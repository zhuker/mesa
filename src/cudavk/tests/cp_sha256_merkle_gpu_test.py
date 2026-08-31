#!/usr/bin/env python3
"""Optional CUDA-device validation for the standalone SHA-256 Merkle kernels.

The default mode only compiles PTX. It does not load libcuda, create a context,
or take the GPU lock. Pass --gpu explicitly to run the device tests. GPU mode
serializes itself with /tmp/cudavk-gpu.lock before it touches the CUDA Driver
API.

The hashlib encoding oracle, ctypes ABI structures, and constants come from
cp_sha256_merkle_test.py. This file intentionally does not copy that encoding.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
from pathlib import Path
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

# Import the existing independent hashlib oracle and its ABI declarations.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cp_sha256_merkle_test as cpu_test  # noqa: E402

CU = cpu_test.CU
KERNEL_DIR = cpu_test.KERNEL_DIR
LEAF_BYTES = cpu_test.LEAF_BYTES
DIGEST_BYTES = cpu_test.DIGEST_BYTES
LeafArgs = cpu_test.LeafArgs
NodeArgs = cpu_test.NodeArgs
Plan = cpu_test.Plan
GPU_LOCK = Path("/tmp/cudavk-gpu.lock")
THREADS = 128
GUARD_BYTES = 64
CANARY = 0xA5
UINT64_MAX = (1 << 64) - 1


def validate_host_abi() -> None:
    expected = {
        LeafArgs: (48, {"source": 0, "output": 8, "total_bytes": 16,
                        "region_id": 24, "first_leaf": 32,
                        "work_leaf_count": 40}),
        NodeArgs: (56, {"input": 0, "output": 8, "region_id": 16,
                        "input_count": 24, "first_parent": 32,
                        "work_parent_count": 40, "level": 48,
                        "reserved": 52}),
        Plan: (40, {"leaf_count": 0, "plane_bytes": 8, "scratch_bytes": 16,
                    "root_offset": 24, "node_launch_count": 32,
                    "kernel_launch_count": 36}),
    }
    for structure, (size, offsets) in expected.items():
        if ctypes.sizeof(structure) != size:
            raise TestFailure(
                f"{structure.__name__} ABI size is {ctypes.sizeof(structure)}, expected {size}"
            )
        for field, offset in offsets.items():
            actual = getattr(structure, field).offset
            if actual != offset:
                raise TestFailure(
                    f"{structure.__name__}.{field} ABI offset is {actual}, expected {offset}"
                )


class TestFailure(RuntimeError):
    """A CUDA operation or validation failed."""


def _find_nvrtc() -> str | None:
    candidates = [
        os.environ.get("NVRTC_LIBRARY"),
        "/usr/local/cuda/lib64/libnvrtc.so",
        ctypes.util.find_library("nvrtc"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            ctypes.CDLL(candidate)
            return candidate
        except OSError:
            pass
    return None


def _nvrtc_ptx(architecture: str) -> tuple[bytes, str]:
    library = _find_nvrtc()
    if library is None:
        raise FileNotFoundError("libnvrtc was not found")
    nvrtc = ctypes.CDLL(library)
    program = ctypes.c_void_p()

    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
    ]
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
    ]
    nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
    nvrtc.nvrtcGetPTXSize.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
    ]
    nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
    nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetPTX.restype = ctypes.c_int
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

    source = CU.read_bytes()
    result = nvrtc.nvrtcCreateProgram(
        ctypes.byref(program), source, CU.name.encode(), 0, None, None
    )
    if result != 0:
        raise TestFailure(f"nvrtcCreateProgram failed with error {result}")
    options_raw = [
        b"--std=c++14",
        f"--gpu-architecture={architecture}".encode(),
        f"--include-path={KERNEL_DIR}".encode(),
    ]
    options = (ctypes.c_char_p * len(options_raw))(*options_raw)
    try:
        result = nvrtc.nvrtcCompileProgram(program, len(options_raw), options)
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(max(1, log_size.value))
        nvrtc.nvrtcGetProgramLog(program, log)
        if result != 0:
            raise TestFailure(
                f"nvrtcCompileProgram failed with error {result}:\n"
                + log.value.decode(errors="replace")
            )
        ptx_size = ctypes.c_size_t()
        if nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)) != 0:
            raise TestFailure("nvrtcGetPTXSize failed")
        ptx = ctypes.create_string_buffer(ptx_size.value)
        if nvrtc.nvrtcGetPTX(program, ptx) != 0:
            raise TestFailure("nvrtcGetPTX failed")
        return ptx.raw, f"NVRTC ({library})"
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def _nvcc_ptx(architecture: str) -> tuple[bytes, str]:
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        raise FileNotFoundError("nvcc was not found")
    with tempfile.TemporaryDirectory(prefix="cp_sha256_merkle_gpu_") as directory:
        output = Path(directory) / "cp_sha256_merkle.ptx"
        command = [
            nvcc, "--std=c++14", "--ptx", f"--gpu-architecture={architecture}",
            "-I", str(KERNEL_DIR), str(CU), "-o", str(output),
        ]
        subprocess.run(command, check=True)
        return output.read_bytes(), f"nvcc ({nvcc})"


def compile_ptx(architecture: str) -> tuple[bytes, str]:
    """Compile the frozen CUDA module without using a CUDA device."""
    errors: list[str] = []
    try:
        ptx, compiler = _nvrtc_ptx(architecture)
    except (FileNotFoundError, OSError, TestFailure) as error:
        errors.append(str(error))
        try:
            ptx, compiler = _nvcc_ptx(architecture)
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error2:
            errors.append(str(error2))
            raise TestFailure("no CUDA compiler succeeded: " + "; ".join(errors))
    for symbol in (
        b"cp_sha256_merkle_leaf_kernel", b"cp_sha256_merkle_node_kernel"
    ):
        if symbol not in ptx:
            raise TestFailure(f"compiled PTX is missing {symbol.decode()}")
    return ptx, compiler


class Cuda:
    """Small checked ctypes binding for the CUDA Driver API used by this test."""

    def __init__(self) -> None:
        library = os.environ.get("CUDA_DRIVER_LIBRARY") or ctypes.util.find_library("cuda")
        if not library:
            library = "libcuda.so.1"
        try:
            self.lib = ctypes.CDLL(library)
        except OSError as error:
            raise TestFailure(f"cannot load CUDA Driver API ({library}): {error}")
        self.library = library
        self._bind_all()

    def _bind(self, names: Iterable[str], argtypes: list[object]):
        for name in names:
            function = getattr(self.lib, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = ctypes.c_int
                return function
        raise TestFailure("CUDA Driver API symbol not found: " + "/".join(names))

    def _bind_all(self) -> None:
        vp = ctypes.c_void_p
        u32 = ctypes.c_uint
        size = ctypes.c_size_t
        devptr_p = ctypes.POINTER(ctypes.c_uint64)
        self.cuInit = self._bind(["cuInit"], [u32])
        self.cuDeviceGetCount = self._bind(
            ["cuDeviceGetCount"], [ctypes.POINTER(ctypes.c_int)]
        )
        self.cuDeviceGet = self._bind(
            ["cuDeviceGet"], [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        )
        self.cuDeviceGetName = self._bind(
            ["cuDeviceGetName"], [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        )
        self.cuDeviceComputeCapability = self._bind(
            ["cuDeviceComputeCapability"], [
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ]
        )
        self.cuCtxCreate = self._bind(
            ["cuCtxCreate_v2", "cuCtxCreate"], [ctypes.POINTER(vp), u32, ctypes.c_int]
        )
        self.cuCtxDestroy = self._bind(["cuCtxDestroy_v2", "cuCtxDestroy"], [vp])
        self.cuCtxGetCurrent = self._bind(["cuCtxGetCurrent"], [ctypes.POINTER(vp)])
        self.cuCtxSynchronize = self._bind(["cuCtxSynchronize"], [])
        self.cuModuleLoadData = self._bind(
            ["cuModuleLoadData"], [ctypes.POINTER(vp), vp]
        )
        self.cuModuleUnload = self._bind(["cuModuleUnload"], [vp])
        self.cuModuleGetFunction = self._bind(
            ["cuModuleGetFunction"], [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        )
        self.cuMemAlloc = self._bind(
            ["cuMemAlloc_v2", "cuMemAlloc"], [devptr_p, size]
        )
        self.cuMemFree = self._bind(
            ["cuMemFree_v2", "cuMemFree"], [ctypes.c_uint64]
        )
        self.cuMemcpyHtoD = self._bind(
            ["cuMemcpyHtoD_v2", "cuMemcpyHtoD"], [ctypes.c_uint64, vp, size]
        )
        self.cuMemcpyDtoH = self._bind(
            ["cuMemcpyDtoH_v2", "cuMemcpyDtoH"], [vp, ctypes.c_uint64, size]
        )
        self.cuMemsetD8 = self._bind(
            ["cuMemsetD8_v2", "cuMemsetD8"], [ctypes.c_uint64, ctypes.c_ubyte, size]
        )
        self.cuLaunchKernel = self._bind(
            ["cuLaunchKernel"], [
                vp, u32, u32, u32, u32, u32, u32, u32, vp,
                ctypes.POINTER(vp), ctypes.POINTER(vp),
            ]
        )
        self.cuGetErrorName = getattr(self.lib, "cuGetErrorName", None)
        self.cuGetErrorString = getattr(self.lib, "cuGetErrorString", None)
        for function in (self.cuGetErrorName, self.cuGetErrorString):
            if function is not None:
                function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
                function.restype = ctypes.c_int

    def error_text(self, result: int) -> str:
        values = []
        for function in (self.cuGetErrorName, self.cuGetErrorString):
            if function is not None:
                value = ctypes.c_char_p()
                if function(result, ctypes.byref(value)) == 0 and value.value:
                    values.append(value.value.decode(errors="replace"))
        return ": ".join(values) if values else f"error {result}"

    def check(self, result: int, operation: str) -> None:
        if result != 0:
            raise TestFailure(f"{operation} failed: {self.error_text(result)} ({result})")


class DeviceSession:
    def __init__(self, cuda: Cuda, ordinal: int, ptx: bytes) -> None:
        self.cuda = cuda
        self.context = ctypes.c_void_p()
        self.module = ctypes.c_void_p()
        self.allocations: list[int] = []
        self.device = ctypes.c_int()
        self.name = ""
        self.major = 0
        self.minor = 0
        self._create(ordinal, ptx)

    def _create(self, ordinal: int, ptx: bytes) -> None:
        c = self.cuda
        c.check(c.cuInit(0), "cuInit")
        count = ctypes.c_int()
        c.check(c.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        if ordinal < 0 or ordinal >= count.value:
            raise TestFailure(f"CUDA device {ordinal} is outside [0, {count.value})")
        c.check(c.cuDeviceGet(ctypes.byref(self.device), ordinal), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        c.check(c.cuDeviceGetName(name, len(name), self.device.value), "cuDeviceGetName")
        major, minor = ctypes.c_int(), ctypes.c_int()
        c.check(c.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), self.device.value
        ), "cuDeviceComputeCapability")
        self.name = name.value.decode(errors="replace")
        self.major, self.minor = major.value, minor.value
        c.check(c.cuCtxCreate(ctypes.byref(self.context), 0, self.device.value), "cuCtxCreate")
        current = ctypes.c_void_p()
        c.check(c.cuCtxGetCurrent(ctypes.byref(current)), "cuCtxGetCurrent")
        if current.value != self.context.value:
            raise TestFailure("created CUDA context is not current")
        image = ctypes.create_string_buffer(ptx if ptx.endswith(b"\0") else ptx + b"\0")
        c.check(c.cuModuleLoadData(
            ctypes.byref(self.module), ctypes.cast(image, ctypes.c_void_p)
        ), "cuModuleLoadData")
        self.leaf = self.function(b"cp_sha256_merkle_leaf_kernel")
        self.node = self.function(b"cp_sha256_merkle_node_kernel")

    def function(self, name: bytes) -> ctypes.c_void_p:
        function = ctypes.c_void_p()
        self.cuda.check(self.cuda.cuModuleGetFunction(
            ctypes.byref(function), self.module, name
        ), f"cuModuleGetFunction({name.decode()})")
        return function

    def alloc(self, size: int) -> int:
        if size <= 0:
            raise TestFailure(f"invalid device allocation size {size}")
        pointer = ctypes.c_uint64()
        self.cuda.check(self.cuda.cuMemAlloc(ctypes.byref(pointer), size), "cuMemAlloc")
        self.allocations.append(pointer.value)
        return pointer.value

    def free(self, pointer: int) -> None:
        if pointer in self.allocations:
            self.cuda.check(self.cuda.cuMemFree(pointer), "cuMemFree")
            self.allocations.remove(pointer)

    def upload(self, pointer: int, data: bytes) -> None:
        source = ctypes.create_string_buffer(data)
        self.cuda.check(self.cuda.cuMemcpyHtoD(
            pointer, ctypes.cast(source, ctypes.c_void_p), len(data)
        ), "cuMemcpyHtoD")

    def download(self, pointer: int, size: int) -> bytes:
        output = ctypes.create_string_buffer(size)
        self.cuda.check(self.cuda.cuMemcpyDtoH(
            ctypes.cast(output, ctypes.c_void_p), pointer, size
        ), "cuMemcpyDtoH")
        return output.raw

    def memset(self, pointer: int, value: int, size: int) -> None:
        self.cuda.check(self.cuda.cuMemsetD8(pointer, value, size), "cuMemsetD8")

    def launch(self, function: ctypes.c_void_p, args: ctypes.Structure,
               work: int) -> None:
        if work <= 0:
            raise TestFailure("the test must not launch a zero-work CUDA grid")
        blocks = (work + THREADS - 1) // THREADS
        parameter = ctypes.cast(ctypes.byref(args), ctypes.c_void_p)
        parameters = (ctypes.c_void_p * 1)(parameter.value)
        self.cuda.check(self.cuda.cuLaunchKernel(
            function, blocks, 1, 1, THREADS, 1, 1, 0, None, parameters, None
        ), "cuLaunchKernel")
        self.cuda.check(self.cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self) -> None:
        first_error: Exception | None = None
        for pointer in reversed(self.allocations[:]):
            try:
                self.cuda.check(self.cuda.cuMemFree(pointer), "cuMemFree")
            except Exception as error:  # keep cleaning up
                first_error = first_error or error
            finally:
                if pointer in self.allocations:
                    self.allocations.remove(pointer)
        if self.module.value:
            try:
                self.cuda.check(self.cuda.cuModuleUnload(self.module), "cuModuleUnload")
            except Exception as error:
                first_error = first_error or error
            self.module = ctypes.c_void_p()
        if self.context.value:
            try:
                self.cuda.check(self.cuda.cuCtxDestroy(self.context), "cuCtxDestroy")
            except Exception as error:
                first_error = first_error or error
            self.context = ctypes.c_void_p()
        if first_error:
            raise first_error


def make_plan(total_bytes: int) -> Plan:
    """Construct the public two-plane plan described by the frozen header."""
    count = cpu_test.leaf_count(total_bytes)
    levels = (count - 1).bit_length()
    plane_bytes = count * DIGEST_BYTES
    return Plan(
        count,
        plane_bytes,
        2 * plane_bytes,
        plane_bytes if levels & 1 else 0,
        levels,
        levels + 1,
    )


def _ranges(count: int, split: bool) -> list[tuple[int, int]]:
    if not split or count == 1:
        return [(0, count)]
    # Unequal chunks exercise both first_* and work_* without huge allocations.
    pieces: list[tuple[int, int]] = []
    first = 0
    widths = (1, 2, 3)
    index = 0
    while first < count:
        work = min(widths[index % len(widths)], count - first)
        pieces.append((first, work))
        first += work
        index += 1
    return pieces


def _assert_guards(session: DeviceSession, allocation: int, logical_size: int,
                   label: str) -> None:
    prefix = session.download(allocation, GUARD_BYTES)
    suffix = session.download(allocation + GUARD_BYTES + logical_size, GUARD_BYTES)
    expected = bytes([CANARY]) * GUARD_BYTES
    if prefix != expected or suffix != expected:
        raise TestFailure(f"{label}: device scratch guard was modified")


def gpu_root(session: DeviceSession, data: bytes, region_id: int,
             split: bool = False) -> bytes:
    plan = make_plan(len(data))
    source = session.alloc(max(1, len(data)))
    scratch_allocation = session.alloc(plan.scratch_bytes + 2 * GUARD_BYTES)
    scratch = scratch_allocation + GUARD_BYTES
    try:
        if data:
            session.upload(source, data)
        else:
            session.memset(source, 0, 1)
        session.memset(
            scratch_allocation, CANARY, plan.scratch_bytes + 2 * GUARD_BYTES
        )
        for first, work in _ranges(plan.leaf_count, split):
            args = LeafArgs(
                source, scratch, len(data), region_id, first, work
            )
            session.launch(session.leaf, args, work)

        count = plan.leaf_count
        input_offset = 0
        output_offset = plan.plane_bytes
        level = 1
        physical_launches = len(_ranges(count, split))
        while count > 1:
            parents = (count + 1) // 2
            for first, work in _ranges(parents, split):
                args = NodeArgs(
                    scratch + input_offset,
                    scratch + output_offset,
                    region_id,
                    count,
                    first,
                    work,
                    level,
                    0,
                )
                session.launch(session.node, args, work)
                physical_launches += 1
            input_offset, output_offset = output_offset, input_offset
            count = parents
            level += 1
        if input_offset != plan.root_offset:
            raise TestFailure("host launch sequence disagrees with plan.root_offset")
        if not split and physical_launches != plan.kernel_launch_count:
            raise TestFailure("host launch sequence disagrees with plan.kernel_launch_count")
        _assert_guards(session, scratch_allocation, plan.scratch_bytes, "valid tree")
        return session.download(scratch + plan.root_offset, DIGEST_BYTES)
    finally:
        session.free(scratch_allocation)
        session.free(source)


def deterministic_vectors() -> list[tuple[str, bytes]]:
    def pattern(size: int, salt: int) -> bytes:
        return bytes(((index * 131) + salt) & 0xFF for index in range(size))

    vectors = [("empty", b""), ("abc", b"abc")]
    for size in (55, 56, 63, 64, 4095, 4096, 4097):
        vectors.append((str(size), pattern(size, size)))
    # Exact leaf counts 2/3/4 cover multi-leaf even and odd levels.
    for leaves in (2, 3, 4, 5):
        size = leaves * LEAF_BYTES
        vectors.append((f"{leaves}-leaves", pattern(size, leaves)))
    rng = random.Random(0xC0DA256)
    for index in range(12):
        size = rng.randrange(0, 9 * LEAF_BYTES + 1)
        vectors.append((f"random-{index}-{size}", rng.randbytes(size)))
    return vectors


def test_vectors(session: DeviceSession) -> int:
    vectors = deterministic_vectors()
    regions = (0, 0x0123456789ABCDEF)
    checks = 0
    for region_id in regions:
        for label, data in vectors:
            actual = gpu_root(session, data, region_id)
            expected = cpu_test.oracle_root(data, region_id)
            if actual != expected:
                raise TestFailure(
                    f"{label}, region {region_id:#x}: GPU {actual.hex()} != "
                    f"oracle {expected.hex()}"
                )
            checks += 1

    original = bytes((index * 17 + 3) & 0xFF for index in range(4 * LEAF_BYTES + 17))
    corrupt = bytearray(original)
    corrupt[len(corrupt) // 2] ^= 0x80
    original_root = gpu_root(session, original, 7)
    corrupt_root = gpu_root(session, bytes(corrupt), 7)
    expected_original = cpu_test.oracle_root(original, 7)
    expected_corrupt = cpu_test.oracle_root(bytes(corrupt), 7)
    if original_root != expected_original or corrupt_root != expected_corrupt:
        raise TestFailure("corrupt-one-byte roots disagree with hashlib oracle")
    if original_root == corrupt_root:
        raise TestFailure("one-byte corruption did not change the GPU root")
    checks += 2

    split_data = bytes((index * 29 + 5) & 0xFF for index in range(7 * LEAF_BYTES + 23))
    split_root = gpu_root(session, split_data, 0xFEDCBA9876543210, split=True)
    expected = cpu_test.oracle_root(split_data, 0xFEDCBA9876543210)
    if split_root != expected:
        raise TestFailure("split first/work launches disagree with hashlib oracle")
    if split_root != gpu_root(session, split_data, 0xFEDCBA9876543210):
        raise TestFailure("split and unsplit launch sequences disagree")
    checks += 2
    return checks


def _assert_all_canary(session: DeviceSession, allocation: int, size: int,
                       label: str) -> None:
    actual = session.download(allocation, size)
    if actual != bytes([CANARY]) * size:
        raise TestFailure(f"{label}: rejected work modified output/canary bytes")


def test_bad_work_bounds(session: DeviceSession) -> int:
    """Out-of-range and overflowing first/work intervals must write nothing."""
    data = bytes((index * 11 + 9) & 0xFF for index in range(3 * LEAF_BYTES + 1))
    count = cpu_test.leaf_count(len(data))
    source = session.alloc(len(data))
    output_size = count * DIGEST_BYTES + 2 * GUARD_BYTES
    output_allocation = session.alloc(output_size)
    try:
        session.upload(source, data)
        cases = ((count, 1), (UINT64_MAX, 2))
        for first, work in cases:
            session.memset(output_allocation, CANARY, output_size)
            args = LeafArgs(
                source, output_allocation + GUARD_BYTES, len(data), 3, first, work
            )
            session.launch(session.leaf, args, work)
            _assert_all_canary(session, output_allocation, output_size,
                               f"bad leaf interval ({first}, {work})")
    finally:
        session.free(output_allocation)
        session.free(source)

    input_count = 5
    parent_count = (input_count + 1) // 2
    input_pointer = session.alloc(input_count * DIGEST_BYTES)
    output_size = parent_count * DIGEST_BYTES + 2 * GUARD_BYTES
    output_allocation = session.alloc(output_size)
    try:
        session.memset(input_pointer, 0x3C, input_count * DIGEST_BYTES)
        cases = ((parent_count, 1), (UINT64_MAX, 2))
        for first, work in cases:
            session.memset(output_allocation, CANARY, output_size)
            args = NodeArgs(
                input_pointer,
                output_allocation + GUARD_BYTES,
                3,
                input_count,
                first,
                work,
                1,
                0,
            )
            session.launch(session.node, args, work)
            _assert_all_canary(session, output_allocation, output_size,
                               f"bad node interval ({first}, {work})")
    finally:
        session.free(output_allocation)
        session.free(input_pointer)
    return 4


def run_gpu(device_ordinal: int) -> None:
    # The lock covers driver initialization, context creation, compilation, and
    # all GPU work. Blocking flock matches the other cudavk standalone tests.
    with GPU_LOCK.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"GPU lock: {GPU_LOCK} (exclusive)")
        cuda = Cuda()
        cuda.check(cuda.cuInit(0), "cuInit")
        device = ctypes.c_int()
        cuda.check(cuda.cuDeviceGet(ctypes.byref(device), device_ordinal), "cuDeviceGet")
        major, minor = ctypes.c_int(), ctypes.c_int()
        cuda.check(cuda.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), device.value
        ), "cuDeviceComputeCapability")
        architecture = f"compute_{major.value}{minor.value}"
        ptx, compiler = compile_ptx(architecture)
        print(f"CUDA compiler: {compiler}; {architecture}; {len(ptx)} PTX bytes")
        session: DeviceSession | None = None
        try:
            session = DeviceSession(cuda, device_ordinal, ptx)
            print(
                f"CUDA device: {device_ordinal}: {session.name} "
                f"(compute {session.major}.{session.minor})"
            )
            print(f"CUDA context: 0x{session.context.value:x}; driver={cuda.library}")
            vector_checks = test_vectors(session)
            bad_checks = test_bad_work_bounds(session)
            print(
                f"GPU SHA-256/hashlib Merkle validation: PASS "
                f"({vector_checks} roots, {bad_checks} rejected bounds)"
            )
        finally:
            if session is not None:
                session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", action="store_true",
        help="take /tmp/cudavk-gpu.lock and run the CUDA device validation",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument(
        "--architecture", default="compute_52",
        help="PTX target for no-device compile mode (default: compute_52)",
    )
    args = parser.parse_args()
    try:
        validate_host_abi()
        if args.gpu:
            run_gpu(args.device)
        else:
            ptx, compiler = compile_ptx(args.architecture)
            print(
                f"No-device CUDA compile: PASS ({compiler}; {args.architecture}; "
                f"{len(ptx)} PTX bytes)"
            )
            print("GPU validation: NOT RUN (pass --gpu explicitly)")
        return 0
    except (TestFailure, OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
