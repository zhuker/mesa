#!/usr/bin/env python3
"""
Generate FLAGS.md from the registry in cp_debug.c.

The registry is the single source of truth for the driver's environment
switches, and hand-maintained tables of them drift: ABUFFER.md's said
CUDAPIPE_ABUFFER_TIMING defaulted on for about a month after the code had
defaulted it off, so anyone following it set a variable that changed nothing.
A generated table cannot do that.

    cp_debug_doc.py            rewrite FLAGS.md
    cp_debug_doc.py --check    exit 1 if it is out of date, print the diff

--check is what a hook or a reviewer runs; it touches nothing.

This parses the C rather than running the driver, so it needs no GPU and no
build. The registry is a flat array of designated initialisers written in one
style, which is easy to parse and worth keeping that way -- if this script
stops understanding the array, fix the array's formatting rather than teaching
the parser a second style.
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent
SOURCE = DRIVER / "cp_debug.c"
HEADER = DRIVER / "cp_debug.h"
OUTPUT = DRIVER / "FLAGS.md"

TYPE_LABEL = {
    "CP_FLAG_BOOL_PRESENCE": "bool (presence)",
    "CP_FLAG_BOOL_VALUE": "bool (value)",
    "CP_FLAG_OPT_BOOL": "bool (optional)",
    "CP_FLAG_OPT_INT": "int (optional)",
    "CP_FLAG_UINT": "uint",
    "CP_FLAG_U64": "uint64",
    "CP_FLAG_INT": "int",
    "CP_FLAG_FLOAT": "float",
    "CP_FLAG_ENUM": "enum",
}

# Defaults are written as C constants. Resolving them by hand would be a second
# source of truth, so they are read from the header that defines them.
def constants():
    text = HEADER.read_text()
    # The limits come from stdint.h, which is not worth parsing for two names.
    out = {"UINT32_MAX": str(2**32 - 1), "UINT64_MAX": str(2**64 - 1),
           "INT32_MAX": str(2**31 - 1)}
    for name, value in re.findall(r"^#define\s+(CP_\w+)\s+(.+?)\s*$", text, re.M):
        value = value.split("/*")[0].strip()
        out[name] = value
    return out


def resolve(expr, consts, depth=0):
    """Reduce a C constant expression to something a reader wants to see."""
    expr = expr.strip()
    if not expr or depth > 4:
        return expr
    if expr in consts:
        return resolve(consts[expr], consts, depth + 1)
    # ((uint64_t)128) and friends
    cleaned = re.sub(r"\((?:uint64_t|unsigned|int|size_t)\)", "", expr).strip()
    for name, value in consts.items():
        cleaned = re.sub(r"\b%s\b" % re.escape(name), "(%s)" % value, cleaned)
    cleaned = re.sub(r"\((?:uint64_t|unsigned|int|size_t)\)", "", cleaned)
    if re.fullmatch(r"[-\d\s()+*<>|&.]+", cleaned):
        try:
            return str(eval(cleaned))  # noqa: S307 - our own header, digits only
        except Exception:
            pass
    return expr


def split_entries(body):
    """Split the array body on top-level commas between { } groups."""
    entries, depth, start, in_str = [], 0, None, False
    i = 0
    while i < len(body):
        c = body[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                entries.append(body[start:i])
        i += 1
    return entries


def parse():
    text = SOURCE.read_text()
    m = re.search(r"static const struct cp_flag_def flags\[\]\s*=\s*\{(.*?)\n\};",
                  text, re.S)
    if not m:
        sys.exit("cp_debug_doc.py: could not find the flags[] array in cp_debug.c")
    body = m.group(1)

    consts = constants()

    # Sections are the /* ---- name ---- */ comments between entries, so the
    # generated grouping follows the array's own grouping.
    parts = re.split(r"/\* ---- (.*?) ---- \*/", body)
    # parts[0] is anything before the first heading; then name, body, name, body...
    grouped = []
    for i in range(1, len(parts), 2):
        grouped.append((parts[i].strip(), parts[i + 1]))
    if not grouped:
        grouped = [("flags", body)]

    out = []
    for heading, chunk in grouped:
        rows = []
        for entry in split_entries(chunk):
            fields = entry.strip()
            name_m = re.match(r'\s*"([A-Z0-9_]+)"', fields)
            if not name_m:
                continue
            name = name_m.group(1)
            type_m = re.search(r",\s*(CP_FLAG_\w+)\s*,", fields)
            ctype = type_m.group(1) if type_m else "?"
            # The help is every string literal after the first, which was the
            # variable name; adjacent literals concatenate as the compiler does.
            strings = re.findall(r'"((?:[^"\\]|\\.)*)"', fields)
            help_text = " ".join(s.strip() for s in strings[1:]).strip()
            help_text = re.sub(r"\s+", " ", help_text)

            dflt_m = re.search(r"\.dflt\s*=\s*([^,}]+)", fields)
            fdflt_m = re.search(r"\.fdflt\s*=\s*([^,}]+)", fields)
            if ctype in ("CP_FLAG_OPT_BOOL", "CP_FLAG_OPT_INT"):
                default = "unset"
            elif ctype == "CP_FLAG_BOOL_PRESENCE":
                default = "off"
            elif fdflt_m:
                default = resolve(fdflt_m.group(1), consts)
            elif dflt_m:
                default = resolve(dflt_m.group(1), consts)
                if ctype == "CP_FLAG_BOOL_VALUE":
                    default = "on" if default not in ("0", "") else "off"
                elif ctype == "CP_FLAG_ENUM":
                    default = default.replace("CP_ARENA_", "").lower()
            else:
                default = "off" if ctype == "CP_FLAG_BOOL_VALUE" else "0"

            rng = ""
            if re.search(r"\.has_range\s*=\s*true", fields):
                lo = re.search(r"\.lo\s*=\s*([^,}]+)", fields)
                hi = re.search(r"\.hi\s*=\s*([^,}]+)", fields)
                lo = resolve(lo.group(1), consts) if lo else "?"
                hi = resolve(hi.group(1), consts) if hi else "?"
                if hi == str(2**32 - 1):
                    rng = "&ge; %s" % lo
                else:
                    rng = "%s&ndash;%s" % (lo, hi)
            rows.append((name, TYPE_LABEL.get(ctype, ctype), default, rng, help_text))
        if rows:
            out.append((heading, rows))
    return out


def render(groups):
    total = sum(len(rows) for _, rows in groups)
    lines = [
        "# The driver's environment switches",
        "",
        "**Generated from the registry in `cp_debug.c` by",
        "`tests/cp_debug_doc.py`. Do not edit by hand — edit the registry.**",
        "",
        "`CUDAPIPE_HELP=1` prints the same table from a running driver, with the",
        "value each variable actually resolved to in that process, which is the",
        "form to use when the question is what a run was configured to do.",
        "",
        "Two boolean kinds appear here and the difference bites:",
        "",
        "- **bool (presence)** is set by the variable existing at all, so",
        "  `CUDAPIPE_DEBUG_DRAW=0` turns tracing **on**.",
        "- **bool (value)** reads the value, so `=0` turns it off.",
        "",
        "That is not a design, it is what the flags grew into, and it is preserved",
        "deliberately: someone's script sets one of these to 0 today.",
        "",
        "%d switches." % total,
        "",
    ]
    for heading, rows in groups:
        lines.append("## %s" % heading[0].upper() + heading[1:])
        lines.append("")
        lines.append("| variable | type | default | range | meaning |")
        lines.append("|---|---|---|---|---|")
        for name, ctype, default, rng, help_text in rows:
            lines.append("| `%s` | %s | `%s` | %s | %s |"
                         % (name, ctype, default, rng or "—",
                            help_text.replace("|", "\\|")))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if FLAGS.md is out of date, changing nothing")
    args = ap.parse_args()

    text = render(parse())

    if args.check:
        old = OUTPUT.read_text() if OUTPUT.exists() else ""
        if old == text:
            print("cp_debug_doc.py: %s is up to date" % OUTPUT.name)
            return 0
        sys.stdout.writelines(difflib.unified_diff(
            old.splitlines(True), text.splitlines(True),
            fromfile="%s (on disk)" % OUTPUT.name,
            tofile="%s (from cp_debug.c)" % OUTPUT.name))
        print("\ncp_debug_doc.py: out of date — run tests/cp_debug_doc.py")
        return 1

    OUTPUT.write_text(text)
    print("cp_debug_doc.py: wrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
