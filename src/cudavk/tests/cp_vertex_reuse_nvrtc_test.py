#!/usr/bin/env python3
"""Compile the actual vertex-reuse CUDA source with NVRTC, without a device."""
from __future__ import annotations
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kernels"
lib = ctypes.CDLL("libnvrtc.so")
prog = ctypes.c_void_p()
source = (K / "cp_vertex_reuse.cu").read_bytes()
headers = [(K / "cp_rast_types.h").read_bytes(), (K / "cp_vf_lane.h").read_bytes()]
names = [b"cp_rast_types.h", b"cp_vf_lane.h"]
H = (ctypes.c_char_p * len(headers))(*headers)
N = (ctypes.c_char_p * len(names))(*names)
lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
    ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
lib.nvrtcCreateProgram.restype = ctypes.c_int
lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
lib.nvrtcCompileProgram.restype = ctypes.c_int
lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
assert lib.nvrtcCreateProgram(ctypes.byref(prog), source, b"cp_vertex_reuse.cu",
                              len(headers), H, N) == 0
opts = [b"--std=c++14", b"--gpu-architecture=compute_80"]
O = (ctypes.c_char_p * len(opts))(*opts)
res = lib.nvrtcCompileProgram(prog, len(opts), O)
if res:
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, buf)
    raise SystemExit(buf.value.decode(errors="replace"))
n = ctypes.c_size_t()
assert lib.nvrtcGetPTXSize(prog, ctypes.byref(n)) == 0
buf = ctypes.create_string_buffer(n.value)
assert lib.nvrtcGetPTX(prog, buf) == 0
assert b".entry cp_vertex_reuse_census" in buf.raw
assert lib.nvrtcDestroyProgram(ctypes.byref(prog)) == 0
print("vertex reuse NVRTC compile: ok")
