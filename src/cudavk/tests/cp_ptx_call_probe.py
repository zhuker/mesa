#!/usr/bin/env python3
"""Measure the resource union from calling real cudavk PTX bodies.

Input is stderr from a run with CUDAVK_DUMP_PTX=1. Each selected kernel entry
is converted to an ordinary function in its own linker input. A block-uniform
wrapper directly calls one selected body, matching the least risky shader
shape for a persistent run worker. The wrapper is never launched.
"""
import argparse
import ctypes
import re

CU_JIT_INPUT_PTX = 1
ATTR_LOCAL_BYTES = 3
ATTR_REGS = 4


def check(code, what):
    if code:
        raise RuntimeError(f"{what}: CUDA error {code}")


def as_function(ptx, number):
    text = ptx.replace(".visible .entry main(",
                       f".visible .func shader{number}(")
    text = text.replace("main_param_0", f"shader{number}_param_0")
    return "\n".join(line for line in text.splitlines()
                     if ".maxntid" not in line and ".minnctapersm" not in line)


def wrapper(count, dispatch_kind, max_regs):
    declarations = "\n".join(
        f".extern .func shader{i}(.param .u64 shader{i}_param_0);"
        for i in range(count))
    dispatch = []
    if dispatch_kind == "branch":
        for i in range(count):
            dispatch += [f" setp.eq.u32 %p{i + 1}, %r1, {i};",
                         f" @%p{i + 1} bra case{i};"]
        dispatch += [" bra done;"]
        for i in range(count):
            dispatch += [f"case{i}:",
                         f" call.uni shader{i}, (call_arg);",
                         " bra done;"]
        dispatch += ["done:"]
    else:
        for i in range(count):
            dispatch += [f" setp.eq.u32 %p{i + 1}, %r1, {i};"]
            if dispatch_kind == "direct":
                dispatch += [f" @%p{i + 1} call.uni shader{i}, (call_arg);"]
            else:
                dispatch += [f" @%p{i + 1} mov.u64 %rd2, shader{i};"]
    if dispatch_kind == "targets":
        dispatch += ["targets: .calltargets " +
                     ", ".join(f"shader{i}" for i in range(count)) + ";",
                     " call.uni %rd2, (call_arg), targets;"]
    elif dispatch_kind == "prototype":
        dispatch += ["proto: .callprototype _ (.param .u64 _);",
                     " call.uni %rd2, (call_arg), proto;"]
    return f""".version 7.5
.target sm_86
.address_size 64
{declarations}
.visible .entry probe(.param .u64 probe_arg, .param .u32 probe_which)
{f".maxnreg {max_regs}" if max_regs else ""}
{{
 .reg .pred %p<{count + 1}>;
 .reg .b32 %r<2>;
 .reg .b64 %rd<3>;
 .param .b64 call_arg;
 ld.param.u64 %rd1, [probe_arg];
 ld.param.u32 %r1, [probe_which];
 st.param.b64 [call_arg], %rd1;
{chr(10).join(dispatch)}
 ret;
}}
"""


class Driver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.context = ctypes.c_void_p()
        check(self.lib.cuInit(0), "cuInit")
        device = ctypes.c_int()
        check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, device),
              "cuCtxCreate")

    def link(self, inputs, function):
        state = ctypes.c_void_p()
        check(self.lib.cuLinkCreate_v2(0, None, None, ctypes.byref(state)),
              "cuLinkCreate")
        buffers = []
        try:
            for i, text in enumerate(inputs):
                buf = ctypes.create_string_buffer(text.encode())
                buffers.append(buf)
                check(self.lib.cuLinkAddData_v2(
                    state, CU_JIT_INPUT_PTX, buf, len(buf),
                    f"input{i}".encode(), 0, None, None), "cuLinkAddData")
            image = ctypes.c_void_p()
            size = ctypes.c_size_t()
            check(self.lib.cuLinkComplete(state, ctypes.byref(image),
                                          ctypes.byref(size)), "cuLinkComplete")
            module = ctypes.c_void_p()
            check(self.lib.cuModuleLoadData(ctypes.byref(module), image),
                  "cuModuleLoadData")
            try:
                fn = ctypes.c_void_p()
                check(self.lib.cuModuleGetFunction(ctypes.byref(fn), module,
                                                   function.encode()),
                      "cuModuleGetFunction")
                values = []
                for attr in (ATTR_REGS, ATTR_LOCAL_BYTES):
                    value = ctypes.c_int()
                    check(self.lib.cuFuncGetAttribute(ctypes.byref(value),
                                                      attr, fn),
                          "cuFuncGetAttribute")
                    values.append(value.value)
                blocks = ctypes.c_int()
                check(self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    ctypes.byref(blocks), fn, 256, 0), "cuOccupancy")
                return size.value, values[0], values[1], blocks.value
            finally:
                self.lib.cuModuleUnload(module)
        finally:
            self.lib.cuLinkDestroy(state)

    def close(self):
        self.lib.cuCtxDestroy_v2(self.context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="stderr produced with CUDAVK_DUMP_PTX=1")
    ap.add_argument("indices", nargs="+", type=int,
                    help="zero-based generated-PTX block indices")
    ap.add_argument("--dispatch",
                    choices=("direct", "branch", "targets", "prototype"),
                    default="direct")
    ap.add_argument("--max-regs", type=int, default=0)
    args = ap.parse_args()
    text = open(args.dump, encoding="utf-8").read()
    blocks = []
    for match in re.finditer(
            r"cudavk: generated PTX \((\d+) bytes\):\n", text):
        size = int(match.group(1))
        body = text[match.end():match.end() + size]
        if len(body) != size:
            raise SystemExit("truncated PTX block")
        blocks.append(body)
    selected = [blocks[i] for i in args.indices]
    cuda = Driver()
    try:
        for index, ptx in zip(args.indices, selected):
            size, regs, local, occupancy = cuda.link([ptx], "main")
            print(f"kernel[{index}]: cubin={size} regs={regs} local={local} "
                  f"blocks/sm={occupancy}")
        funcs = [as_function(ptx, i) for i, ptx in enumerate(selected)]
        size, regs, local, occupancy = cuda.link(
            funcs + [wrapper(len(funcs), args.dispatch, args.max_regs)],
            "probe")
        print(f"combined[{','.join(map(str, args.indices))}]: cubin={size} "
              f"regs={regs} local={local} blocks/sm={occupancy}")
    finally:
        cuda.close()


if __name__ == "__main__":
    main()
