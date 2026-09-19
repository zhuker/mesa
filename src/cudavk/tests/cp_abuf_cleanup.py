#!/usr/bin/env python3
"""CPU-only ownership regression for the actual renderer A-buffer cleanup.

Run: python3 src/cudavk/tests/cp_abuf_cleanup.py
Requires a C compiler (CC, or cc). No CUDA headers, build, or GPU needed.
--source can select an older cp_renderer.c to verify the regression fails.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


# Allocation bases, not the cleanup list: omissions must fail as well as aliases.
OWNERS = """counts offsets cursor sum1 sum1x sum2 sum2x list clist counters
blk_counts blk_offsets blk_list bsum1 bsum1x bsum2 bsum2x frags recs
quad_prim quad_mask quad_block shade_slot peel_mask log deep_list deep_log
colors_abuf writes_abuf colors_peel writes_peel""".split()
ALIASES = "list_count blk_list_count dbg".split()


def harness(source):
    match = re.search(r"bool\s+cp_abuf_cleanup\(struct cp_abuf \*ab\)\s*\{", source)
    if not match:
        raise RuntimeError("cannot find cp_abuf_cleanup")
    # This function has no braces in literals/comments. Balance nested blocks.
    end = match.end()
    depth = 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise RuntimeError("unterminated cp_abuf_cleanup")
    cleanup = source[match.start():end]
    fields = set(re.findall(r"ab->(\w+)", cleanup))
    host_fields = sorted(f for f in fields if f.startswith("h_"))
    unknown = fields - set(OWNERS + ALIASES + host_fields + ["ev", "enabled"])
    if unknown:
        raise RuntimeError(f"update ownership fixture for fields: {sorted(unknown)}")
    declarations = "\n".join(f"CUdeviceptr {f};" for f in OWNERS + ALIASES)
    declarations += "\n" + "\n".join(f"void *{f};" for f in host_fields)
    setup = "\n".join(
        f"ab->{name} = bases[{i}] = {4096 * (i + 1)};" for i, name in enumerate(OWNERS)
    )
    aliases = "\n".join(
        f"ab->{name} = ab->counters + {4 * (i + 1)};" for i, name in enumerate(ALIASES)
    )
    return r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef uintptr_t CUdeviceptr;
typedef uintptr_t CUevent;
#define CUDA_SUCCESS 0
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); return 1; } } while (0)
""" + f"#define NOWNERS {len(OWNERS)}\n" + r"""
struct cp_abuf {
""" + declarations + r"""
CUevent ev[3];
int enabled;
};
static CUdeviceptr bases[NOWNERS];
static unsigned calls[NOWNERS], event_calls[3], invalid;
static int fail_alloc, fail_event;
static int cuMemFree(CUdeviceptr ptr) {
    for (unsigned i = 0; i < NOWNERS; i++) {
        if (bases[i] != ptr) continue;
        if (calls[i]++) { invalid++; return 1; }
        return (int)i == fail_alloc ? 1 : CUDA_SUCCESS;
    }
    invalid++;
    return 1; /* interior or unknown pointer */
}
static int cuEventDestroy(CUevent ev) {
    if (ev < 1 || ev > 3) { invalid++; return 1; }
    if (event_calls[ev - 1]++) { invalid++; return 1; }
    return (int)ev == fail_event ? 1 : CUDA_SUCCESS;
}
static void cp_abuf_report(struct cp_abuf *ab) { (void)ab; }
""" + cleanup + r"""
static void setup(struct cp_abuf *ab) {
    memset(ab, 0, sizeof(*ab));
    memset(calls, 0, sizeof(calls));
    memset(event_calls, 0, sizeof(event_calls));
    invalid = 0;
    fail_alloc = fail_event = -1;
""" + setup + "\n" + aliases + r"""
    for (unsigned i = 0; i < 3; i++) ab->ev[i] = i + 1;
}
int main(void) {
    struct cp_abuf ab, empty = {0};
    empty.enabled = -1;
    CHECK(cp_abuf_cleanup(NULL));
    memset(&ab, 0, sizeof(ab));
    CHECK(cp_abuf_cleanup(&ab));
    CHECK(memcmp(&ab, &empty, sizeof(ab)) == 0);
    /* Success, every individual allocation failure, every event failure. */
    for (int scenario = -1; scenario < NOWNERS + 3; scenario++) {
        setup(&ab);
        if (scenario >= 0 && scenario < NOWNERS) fail_alloc = scenario;
        if (scenario >= NOWNERS) fail_event = scenario - NOWNERS + 1;
        bool ok = cp_abuf_cleanup(&ab);
        CHECK(invalid == 0);
        CHECK(ok == (scenario == -1));
        for (unsigned i = 0; i < NOWNERS; i++) CHECK(calls[i] == 1);
        for (unsigned i = 0; i < 3; i++) CHECK(event_calls[i] == 1);
        CHECK(memcmp(&ab, &empty, sizeof(ab)) == 0);
        CHECK(cp_abuf_cleanup(&ab)); /* already cleared: no second free */
        CHECK(invalid == 0);
    }
    puts("PASS: A-buffer cleanup owns bases only and propagates CUDA failures");
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "cp_renderer.c")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cp-abuf-cleanup-") as tmp:
        cfile = Path(tmp) / "test.c"
        binary = Path(tmp) / "test"
        cfile.write_text(harness(args.source.read_text()))
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-std=c11", "-Wall", "-Wextra", "-Werror",
                        str(cfile), "-o", str(binary)], check=True)
        return subprocess.run([str(binary)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
