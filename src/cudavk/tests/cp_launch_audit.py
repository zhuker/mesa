#!/usr/bin/env python3
# Every kernel launch must go through cp_launch(), which is the only flush
# point for the coalesced upload span. A raw cuLaunchKernel is the one bug
# class the coalescing design can have, and it is silent: the kernel reads
# arena bytes that are still only in host staging. So it is checked
# mechanically rather than by review.
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# path -> reason. These do not read the upload arena.
ALLOWED = {
    "cp_renderer.c": "cp_launch() itself, the wrapper every other site uses",
    "cpvk_texture_cache.c":
        "conversion kernel: arguments by value, reads image storage, runs on "
        "the cache stream and never touches the upload arena",
}

bad = []
for path in sorted(ROOT.glob("*.c")) + sorted(ROOT.glob("*.h")):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if "cuLaunchKernel" not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("*") or stripped.startswith("/*"):
            continue          # prose
        if "cuLaunchKernel(" not in line:
            continue          # the string in a diagnostic, not a call
        if path.name in ALLOWED:
            continue
        bad.append(f"{path.name}:{n}: {stripped}")

if bad:
    print("raw cuLaunchKernel outside cp_launch():")
    for b in bad:
        print("  " + b)
    raise SystemExit(1)

# and the allowed files must not grow new ones
counts = {"cp_renderer.c": 1, "cpvk_texture_cache.c": 1}
for name, want in counts.items():
    text = (ROOT / name).read_text()
    got = sum(1 for line in text.splitlines()
              if "cuLaunchKernel(" in line and not line.strip().startswith("*"))
    if got != want:
        raise SystemExit(
            f"{name}: {got} raw cuLaunchKernel call sites, expected {want} "
            f"({ALLOWED[name]})")

print("PASS cp_launch audit: every launch flushes the upload span")
