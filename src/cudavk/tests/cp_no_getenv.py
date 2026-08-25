#!/usr/bin/env python3
"""
Fail if driver code calls getenv.

The rule is in CLAUDE.md and in the header of cp_debug.h: an environment
switch is declared in the one array in cp_debug.c, resolved once by
cp_debug_init(), and read as cp_debug->field. The array is what CUDAVK_HELP=1
prints and what generates FLAGS.md, so a switch that lives in the array is
documented and announced by construction, and a switch read with a getenv
somewhere in the driver is invisible to both -- there is no table it appears
in and no run in which the driver says it was set.

That rule was prose only, and prose did not hold: nineteen CPVK_* getenv
switches grew in the driver anyway. The bill came in as a day of measurement
thrown away. CPVK_BATCH and CPVK_BATCH_BLEND were left exported in a shell,
every process launched from that shell inherited them, and every A/B run after
that compared a feature against itself. The numbers reproduced, which is what
made them convincing, and they meant nothing. cpvk_report_env() in
cpvk_device.c exists to print those names at startup, i.e. a whole function
whose only job is to undo the damage of not using the registry.

So the rule is checked mechanically instead of asked for politely.

    cp_no_getenv.py            report every getenv call site in driver code
    cp_no_getenv.py --check    the same thing; accepted so this reads like
                               cp_debug_doc.py --check in a hook
    cp_no_getenv.py --allowlist   print the exceptions and why each one exists

Exit 0 when the driver is clean, 1 when a call site is found, 2 when the
driver sources could not be scanned at all (a moved tree, not a violation).

Standard library only, no build and no GPU: it reads the C.
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent

# Directories under src/cudavk that are compiled into the ICD, plus the driver
# root itself. kernels/ is device code the driver compiles with NVRTC and is
# just as much the driver: a getenv there would be read at build-of-kernel
# time and be equally invisible.
SCAN_DIRS = [".", "nir_to_ptx", "kernels"]
SUFFIXES = (".c", ".h", ".cu", ".cpp", ".cc", ".hpp")

# Reading the environment behind the registry's back is the thing being
# stopped, not the spelling `getenv`. os_get_option() and Mesa's
# debug_get_*_option() helpers are getenv with a default folded in, and they
# would hide a switch from CUDAVK_HELP=1 in exactly the same way.
CALLS = (
    "getenv",
    "secure_getenv",
    "os_get_option",
    "debug_get_option",
    "debug_get_bool_option",
    "debug_get_num_option",
)
CALL_RE = re.compile(r"\b(%s)\s*\(" % "|".join(CALLS))

# An exception is a decision, so it is written down with the reason. A path
# that is not in here and calls getenv is a bug, and adding a path to here
# should feel like the edit it is.
#
# Paths are relative to src/cudavk. A directory entry covers everything under
# it.
ALLOWED = {
    "cp_debug.c":
        "the registry resolver itself -- this is the one getenv the driver "
        "has, plus the CUDAVK_HELP=1 check that prints the table",
    "tests":
        "separate programs built beside the driver, not loaded into it; a "
        "test harness may read its own environment",
    "samples":
        "separate programs, same reason as tests/",
}


def uncomment(text):
    """Blank C comments, keeping every byte offset and line break in place.

    Offsets have to survive because the argument to the call is read back out
    of this text by position. Blanking rather than deleting is what makes that
    work, and it is also why the reported line numbers are the real ones.

    Prose about getenv is common in this tree -- cp_debug.h explains what
    getenv costs, cpvk_device.c explains why its flags are not in the
    registry -- so a checker that greps the raw text reports the
    documentation, which teaches the reader to ignore it.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c in "\"'":
            quote, i = c, i + 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
        else:
            i += 1
    return "".join(out)


def blank_strings(text):
    """Blank string and character literal contents, same length in, same out.

    Needed because a diagnostic that names a variable -- fprintf(..., "set
    CUDAVK_HELP=1") -- must not read as a call, and because `getenv` appearing
    inside a message is text, not behaviour.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            quote, i = c, i + 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n and text[i] != "\n":
                    out[i] = " "
                i += 1
            i += 1
        else:
            i += 1
    return "".join(out)


def argument(text, open_paren):
    """The text between the call's parentheses, balanced, or "" if unbalanced."""
    depth, i, n = 0, open_paren, len(text)
    while i < n:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return " ".join(text[open_paren + 1:i].split())
        i += 1
    return ""


def variable(arg):
    """The switch name when the call spells it, else the expression itself.

    Adjacent string literals concatenate as the compiler does them. A call
    like getenv(names[i]) walks a table -- the names are one indirection away,
    so say what was written and let the reader follow it.
    """
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', arg)
    if parts and re.fullmatch(r'\s*(?:"(?:[^"\\]|\\.)*"\s*)+', arg):
        return "".join(parts)
    return "%s (name not a literal)" % arg if arg else "?"


def allowed(rel):
    """True when this path, or a directory above it, is a named exception."""
    return any(rel == name or rel.startswith(name + "/") for name in ALLOWED)


def sources():
    seen, out = set(), []
    for d in SCAN_DIRS:
        base = DRIVER / d
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if path.is_file() and path.suffix in SUFFIXES and path not in seen:
                seen.add(path)
                out.append(path)
    return out


def scan():
    """Every call site in driver code, as (relative path, line, name, source)."""
    hits = []
    for path in sources():
        rel = path.relative_to(DRIVER).as_posix()
        if allowed(rel):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            sys.stderr.write("cp_no_getenv.py: %s: %s\n" % (rel, e))
            continue
        code = uncomment(text)
        # Two passes: find the call in text with no comments and no string
        # contents, then read its argument out of the version that still has
        # the strings. Both are the same length as the original, so an offset
        # means the same thing in all three.
        hunt = blank_strings(code)
        for m in CALL_RE.finditer(hunt):
            line = hunt.count("\n", 0, m.start()) + 1
            arg = argument(code, m.end() - 1)
            hits.append((rel, line, m.group(1), variable(arg),
                         text.splitlines()[line - 1].strip()))
    return hits


REMEDY = """
What to do instead: add the switch to the flags[] array in cp_debug.c -- name,
parse, default and its one-line meaning -- and read it as cp_debug->field at
the call site. cp_debug_init() resolves the array once, before any draw.

Both CUDAVK_HELP=1 and src/cudavk/FLAGS.md are generated from that array, so a
getenv appears in neither: the driver cannot say the switch is set, and no
generated table can list it. That is not a documentation problem, it is how a
stale exported CPVK_BATCH in one shell made a day of A/B runs compare a
feature against itself, reproducibly and meaninglessly.

Run tests/cp_debug_doc.py after editing the array so FLAGS.md follows it.
If a site genuinely cannot use the registry, add it to ALLOWED in this script
with the reason, so that the exception is a visible decision."""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="check and change nothing, which is all this ever does")
    ap.add_argument("--allowlist", action="store_true",
                    help="print the exceptions and the reason for each, then exit")
    args = ap.parse_args()

    if args.allowlist:
        print("cp_no_getenv.py: exceptions, relative to src/cudavk:")
        for name in sorted(ALLOWED):
            print("  %-12s %s" % (name, ALLOWED[name]))
        return 0

    files = sources()
    if not files:
        sys.stderr.write("cp_no_getenv.py: no driver sources under %s\n" % DRIVER)
        return 2

    hits = scan()
    if not hits:
        print("PASS no-getenv audit: %d driver sources, every switch goes "
              "through the cp_debug.c registry" % len(files))
        return 0

    named = sorted({v for _, _, _, v, _ in hits if not v.endswith(")")})
    indirect = sum(1 for _, _, _, v, _ in hits if v.endswith(")"))
    print("environment read behind the registry's back, %d call site%s in "
          "%d file%s:"
          % (len(hits), "" if len(hits) == 1 else "s",
             len({h[0] for h in hits}), "" if len({h[0] for h in hits}) == 1 else "s"))
    last = None
    for rel, line, call, var, src in hits:
        if rel != last:
            print("\n  %s" % rel)
            last = rel
        print("    %s:%d: %s -> %s" % (rel, line, call, var))
        print("        %s" % src)
    print("\n%d variable%s named outright: %s"
          % (len(named), "" if len(named) == 1 else "s", ", ".join(named)))
    if indirect:
        print("%d site%s reads a name this script cannot see; read the file."
              % (indirect, "" if indirect == 1 else "s"))
    print(REMEDY)
    return 1


if __name__ == "__main__":
    sys.exit(main())
