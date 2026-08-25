#!/usr/bin/env python3
"""Stringify a CUDA kernel source into a C string literal for embedding.

NVRTC compiles kernels from source at runtime, so the .cu files are baked
into the driver as string literals.
"""

import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, 'r') as f:
        lines = f.readlines()

    with open(dst, 'w') as f:
        f.write('// Auto-generated from %s\n' % src.split('/')[-1])
        for line in lines:
            line = line.rstrip('\n')
            line = line.replace('\\', '\\\\').replace('"', '\\"')
            f.write('"%s\\n"\n' % line)


if __name__ == '__main__':
    main()
