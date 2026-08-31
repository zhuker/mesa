#!/usr/bin/env python3
"""CPU/hashlib oracle and no-device NVRTC compile for cp_sha256_merkle.cu.

This script never creates a CUDA context or launches a GPU kernel. It compiles
host-test hooks from the same SHA implementation, simulates one CUDA thread at
a time, and compares every digest to an independent hashlib encoding.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent / "kernels"
CU = KERNEL_DIR / "cp_sha256_merkle.cu"
LEAF_BYTES = 4096
DIGEST_BYTES = 32
LEAF_TAG = b"CPMRKLF1"
NODE_TAG = b"CPMRKND1"


class LeafArgs(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_uint64),
        ("output", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),
        ("region_id", ctypes.c_uint64),
        ("first_leaf", ctypes.c_uint64),
        ("work_leaf_count", ctypes.c_uint64),
    ]


class NodeArgs(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_uint64),
        ("output", ctypes.c_uint64),
        ("region_id", ctypes.c_uint64),
        ("input_count", ctypes.c_uint64),
        ("first_parent", ctypes.c_uint64),
        ("work_parent_count", ctypes.c_uint64),
        ("level", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class Plan(ctypes.Structure):
    _fields_ = [
        ("leaf_count", ctypes.c_uint64),
        ("plane_bytes", ctypes.c_uint64),
        ("scratch_bytes", ctypes.c_uint64),
        ("root_offset", ctypes.c_uint64),
        ("node_launch_count", ctypes.c_uint32),
        ("kernel_launch_count", ctypes.c_uint32),
    ]


def leaf_count(size: int) -> int:
    return 1 if size == 0 else 1 + (size - 1) // LEAF_BYTES


def oracle_leaf(data: bytes, index: int, region_id: int) -> bytes:
    payload = data[index * LEAF_BYTES : (index + 1) * LEAF_BYTES]
    preimage = (
        LEAF_TAG
        + struct.pack(">QQQI", region_id, len(data), index, len(payload))
        + payload
    )
    return hashlib.sha256(preimage).digest()


def oracle_node(
    children: list[bytes], level: int, index: int, region_id: int
) -> bytes:
    left = children[2 * index]
    has_right = 2 * index + 1 < len(children)
    right = children[2 * index + 1] if has_right else bytes(DIGEST_BYTES)
    preimage = (
        NODE_TAG
        + struct.pack(">QIQI", region_id, level, index, 2 if has_right else 1)
        + left
        + right
    )
    return hashlib.sha256(preimage).digest()


def oracle_root(data: bytes, region_id: int = 0) -> bytes:
    nodes = [oracle_leaf(data, i, region_id) for i in range(leaf_count(len(data)))]
    level = 1
    while len(nodes) > 1:
        nodes = [oracle_node(nodes, level, i, region_id)
                 for i in range((len(nodes) + 1) // 2)]
        level += 1
    return nodes[0]


def build_host_library(temp: Path) -> ctypes.CDLL:
    wrapper = temp / "plan_wrapper.cpp"
    wrapper.write_text(
        '#include "cp_sha256_merkle.h"\n'
        'extern "C" int cp_sha256_merkle_host_plan('
        'unsigned long long n, cp_sha256_merkle_plan *p) '
        '{ return cp_sha256_merkle_make_plan(n, p); }\n'
        'extern "C" int cp_sha256_merkle_host_blocks('
        'unsigned long long n, unsigned t, unsigned *b) '
        '{ return cp_sha256_merkle_launch_blocks(n, t, b); }\n'
    )
    output = temp / "libcp_sha256_merkle_host.so"
    command = [
        os.environ.get("CXX", "c++"), "-std=c++11", "-O2", "-shared", "-fPIC",
        "-Wall", "-Wextra", "-Werror", "-DCP_SHA256_MERKLE_HOST_TEST",
        "-x", "c++", str(CU), str(wrapper), "-I", str(KERNEL_DIR),
        "-o", str(output),
    ]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(output))
    lib.cp_sha256_merkle_host_sha256.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
    ]
    lib.cp_sha256_merkle_host_leaf.argtypes = [LeafArgs, ctypes.c_uint64]
    lib.cp_sha256_merkle_host_node.argtypes = [NodeArgs, ctypes.c_uint64]
    lib.cp_sha256_merkle_host_plan.argtypes = [ctypes.c_uint64, ctypes.POINTER(Plan)]
    lib.cp_sha256_merkle_host_plan.restype = ctypes.c_int
    lib.cp_sha256_merkle_host_blocks.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)
    ]
    lib.cp_sha256_merkle_host_blocks.restype = ctypes.c_int
    return lib


def device_code_root(lib: ctypes.CDLL, data: bytes, region_id: int = 0) -> bytes:
    count = leaf_count(len(data))
    plane_bytes = count * DIGEST_BYTES
    # Keep a non-null source for empty input although the leaf does not read it.
    source = ctypes.create_string_buffer(data if data else b"\0")
    scratch = ctypes.create_string_buffer(2 * plane_bytes)
    base = ctypes.addressof(scratch)
    args = LeafArgs(
        ctypes.addressof(source), base, len(data), region_id, 0, count
    )
    for index in range(count):
        lib.cp_sha256_merkle_host_leaf(args, index)

    input_offset = 0
    output_offset = plane_bytes
    level = 1
    while count > 1:
        parents = (count + 1) // 2
        node_args = NodeArgs(
            base + input_offset, base + output_offset, region_id,
            count, 0, parents, level, 0,
        )
        for index in range(parents):
            lib.cp_sha256_merkle_host_node(node_args, index)
        input_offset, output_offset = output_offset, input_offset
        count = parents
        level += 1
    return ctypes.string_at(base + input_offset, DIGEST_BYTES)


def test_raw_sha256(lib: ctypes.CDLL) -> None:
    rng = random.Random(0x5A256)
    vectors = [b"", b"abc"]
    vectors += [bytes((i * 37 + 11) & 0xFF for i in range(n))
                for n in (1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 4096)]
    vectors += [rng.randbytes(rng.randrange(0, 6000)) for _ in range(12)]
    for data in vectors:
        source = ctypes.create_string_buffer(data if data else b"\0")
        digest = ctypes.create_string_buffer(DIGEST_BYTES)
        lib.cp_sha256_merkle_host_sha256(source, len(data), digest)
        assert digest.raw == hashlib.sha256(data).digest(), len(data)


def test_merkle_vectors(lib: ctypes.CDLL) -> None:
    sizes = [
        0, 1, 3, 19, 20, 27, 28, 55, 56, 63, 64, 65,
        4095, 4096, 4097, 8192, 8193, 4 * 4096 + 17, 6 * 4096 + 1,
    ]
    rng = random.Random(0xC0DA)
    vectors = [b"", b"abc"]
    vectors += [bytes((i * 131 + n) & 0xFF for i in range(n)) for n in sizes[1:]]
    vectors += [rng.randbytes(rng.randrange(0, 30000)) for _ in range(16)]
    for region_id in (0, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF):
        for data in vectors:
            actual = device_code_root(lib, data, region_id)
            expected = oracle_root(data, region_id)
            assert actual == expected, (len(data), region_id,
                                        actual.hex(), expected.hex())

    # Pin the encoding as well as comparing against the live oracle.
    assert oracle_root(b"").hex() == (
        "6b9eefa84224e93cc124672ad1f9a062f4604495905eb6f58c648b0425355dc7"
    )
    assert oracle_root(b"abc").hex() == (
        "d6c31e6a20aee0ea0b211eca71ac5fb27f8b3cf86f81e178821b1c8f10e754d5"
    )


def test_corruption_and_domains(lib: ctypes.CDLL) -> None:
    for size in (3, 4097, 4 * 4096 + 17):
        original = bytearray((i * 17 + 3) & 0xFF for i in range(size))
        corrupt = bytearray(original)
        corrupt[size // 2] ^= 0x80
        root = device_code_root(lib, bytes(original), 7)
        assert device_code_root(lib, bytes(corrupt), 7) != root
        assert oracle_root(bytes(corrupt), 7) != oracle_root(bytes(original), 7)
    assert device_code_root(lib, b"abc", 1) != device_code_root(lib, b"abc", 2)


def test_plans(lib: ctypes.CDLL) -> None:
    for size in (0, 1, 4096, 4097, 8192, 8193, 4 * 4096 + 1):
        plan = Plan()
        assert lib.cp_sha256_merkle_host_plan(size, ctypes.byref(plan)) == 0
        count = leaf_count(size)
        levels = (count - 1).bit_length()
        assert plan.leaf_count == count
        assert plan.plane_bytes == count * DIGEST_BYTES
        assert plan.scratch_bytes == 2 * count * DIGEST_BYTES
        assert plan.node_launch_count == levels
        assert plan.kernel_launch_count == levels + 1
        assert plan.root_offset == (plan.plane_bytes if levels & 1 else 0)
    blocks = ctypes.c_uint32()
    assert lib.cp_sha256_merkle_host_blocks(0, 128, ctypes.byref(blocks)) == 0
    assert blocks.value == 0
    assert lib.cp_sha256_merkle_host_blocks(129, 128, ctypes.byref(blocks)) == 0
    assert blocks.value == 2
    assert lib.cp_sha256_merkle_host_blocks(1, 0, ctypes.byref(blocks)) == -1


def find_nvrtc() -> str | None:
    candidates = [
        os.environ.get("NVRTC_LIBRARY"),
        "/usr/local/cuda/lib64/libnvrtc.so",
        ctypes.util.find_library("nvrtc"),
    ]
    return next((item for item in candidates if item and Path(item).exists()), None)


def nvrtc_compile_no_device() -> str:
    library = find_nvrtc()
    if library is None:
        return "SKIP (libnvrtc not available)"
    nvrtc = ctypes.CDLL(library)
    program = ctypes.c_void_p()
    source = CU.read_bytes()
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    nvrtc.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    ]
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    result = nvrtc.nvrtcCreateProgram(
        ctypes.byref(program), source, CU.name.encode(), 0, None, None
    )
    if result != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {result}")
    options_raw = [
        b"--std=c++14", b"--gpu-architecture=compute_52",
        ("--include-path=" + str(KERNEL_DIR)).encode(),
    ]
    options = (ctypes.c_char_p * len(options_raw))(*options_raw)
    try:
        result = nvrtc.nvrtcCompileProgram(program, len(options_raw), options)
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(max(1, log_size.value))
        nvrtc.nvrtcGetProgramLog(program, log)
        if result != 0:
            raise RuntimeError(
                f"nvrtcCompileProgram failed: {result}\n{log.value.decode(errors='replace')}"
            )
        ptx_size = ctypes.c_size_t()
        assert nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)) == 0
        ptx = ctypes.create_string_buffer(ptx_size.value)
        assert nvrtc.nvrtcGetPTX(program, ptx) == 0
        text = ptx.raw
        assert b"cp_sha256_merkle_leaf_kernel" in text
        assert b"cp_sha256_merkle_node_kernel" in text
        return f"PASS ({Path(library).name}, {ptx_size.value} PTX bytes)"
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nvrtc", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cp_sha256_merkle_") as directory:
        lib = build_host_library(Path(directory))
        test_raw_sha256(lib)
        test_merkle_vectors(lib)
        test_corruption_and_domains(lib)
        test_plans(lib)
    print("CPU SHA-256/hashlib Merkle oracle: PASS")
    if not args.skip_nvrtc:
        print("No-device NVRTC compile:", nvrtc_compile_no_device())


if __name__ == "__main__":
    main()
