#!/usr/bin/env python3
"""Embed deterministic binary input as comma-separated C bytes."""
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
data = src.read_bytes()
with dst.open("w", newline="\n") as f:
    f.write("/* Auto-generated from %s (%d bytes). */\n" % (src.name, len(data)))
    for i in range(0, len(data), 16):
        f.write("   " + ", ".join("0x%02x" % b for b in data[i:i + 16]) + ",\n")
