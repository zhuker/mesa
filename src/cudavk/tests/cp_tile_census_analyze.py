#!/usr/bin/env python3
"""
Validate a CUDAVK_TILE_CENSUS per-scope log (renderer2 M0).

The driver, run with CUDAVK_TILE_CENSUS=<edge>, writes one
``tilecensus-scope-v1`` line to stderr per render scope, after a single
header line. This checks that such a log is usable as M1 sizing data before
anyone reads distributions out of it:

  - the header is present, its tile edge matches every line, and it says
    timing=invalid (a censused run must never be quoted for time);
  - every line carries the full key set, no more and no fewer;
  - scope ordinals are the contiguous sequence 1..N -- the population is
    complete, no scope line was lost or duplicated;
  - submit ordinals are positive and non-decreasing;
  - the histogram has exactly HIST_BUCKETS buckets and sums to
    nonempty_tiles;
  - refs_max falls inside the highest non-empty bucket, refs_total >=
    max(refs_max, nonempty_tiles), and an empty histogram means
    refs_total == refs_max == nonempty_tiles == 0;
  - class=none forbids census-derived content (batches, fs_binaries,
    tiles, refs, triangles), and census content requires a marked class;
  - a marked scope has a real tile grid, and its non-empty tiles fit it.

Usage:
    cp_tile_census_analyze.py LOG [LOG ...]
    ... | cp_tile_census_analyze.py -

Exit 0 when every log is valid; 1 with one line per violation on stderr
when any is not; 2 when a log could not be read at all. Non-census stderr
lines are ignored -- the driver's other reports share the stream.
"""

import sys

PREFIX = "tilecensus-scope-v1"
HIST_BUCKETS = 17
SCOPE_KEYS = [
    "scope", "submit", "class", "draws", "solo_draws", "batches",
    "solo_batches", "fs_binaries", "tris_postclip", "tile", "tiles_x",
    "tiles_y", "nonempty_tiles", "refs_total", "refs_max", "hist",
]
INT_KEYS = [k for k in SCOPE_KEYS if k not in ("class", "hist")]
CLASSES = ("none", "opaque", "blended", "mixed")


class Violation(Exception):
    pass


def parse_kv(fields, lineno):
    kv = {}
    for f in fields:
        if "=" not in f:
            raise Violation("line %d: field %r is not KEY=VALUE" % (lineno, f))
        k, v = f.split("=", 1)
        if k in kv:
            raise Violation("line %d: duplicate key %r" % (lineno, k))
        kv[k] = v
    return kv


def parse_scope_line(fields, lineno):
    kv = parse_kv(fields, lineno)
    if sorted(kv) != sorted(SCOPE_KEYS):
        missing = sorted(set(SCOPE_KEYS) - set(kv))
        extra = sorted(set(kv) - set(SCOPE_KEYS))
        raise Violation("line %d: key set mismatch (missing %s, extra %s)"
                        % (lineno, missing, extra))
    rec = {}
    for k in INT_KEYS:
        try:
            rec[k] = int(kv[k])
        except ValueError:
            raise Violation("line %d: %s=%r is not an integer"
                            % (lineno, k, kv[k]))
        if rec[k] < 0:
            raise Violation("line %d: %s=%d is negative" % (lineno, k, rec[k]))
    if kv["class"] not in CLASSES:
        raise Violation("line %d: class=%r is not one of %s"
                        % (lineno, kv["class"], "/".join(CLASSES)))
    rec["class"] = kv["class"]
    try:
        hist = [int(x) for x in kv["hist"].split(",")]
    except ValueError:
        raise Violation("line %d: hist=%r is not comma-separated integers"
                        % (lineno, kv["hist"]))
    if len(hist) != HIST_BUCKETS:
        raise Violation("line %d: hist has %d buckets, expected %d"
                        % (lineno, len(hist), HIST_BUCKETS))
    if any(h < 0 for h in hist):
        raise Violation("line %d: hist has a negative bucket" % lineno)
    rec["hist"] = hist
    return rec


def check_scope(rec, lineno, tile):
    n = lineno
    if rec["tile"] != tile:
        raise Violation("line %d: tile=%d disagrees with header tile=%d"
                        % (n, rec["tile"], tile))
    hist = rec["hist"]
    if sum(hist) != rec["nonempty_tiles"]:
        raise Violation("line %d: hist sums to %d but nonempty_tiles=%d"
                        % (n, sum(hist), rec["nonempty_tiles"]))
    if rec["refs_max"] > 0:
        top = max(b for b, h in enumerate(hist) if h)
        lo = 1 << top
        hi = None if top == HIST_BUCKETS - 1 else (1 << (top + 1)) - 1
        if rec["refs_max"] < lo or (hi is not None and rec["refs_max"] > hi):
            raise Violation(
                "line %d: refs_max=%d outside top non-empty bucket %d [%d..%s]"
                % (n, rec["refs_max"], top, lo, hi if hi is not None else "inf"))
    else:
        if rec["nonempty_tiles"] or rec["refs_total"]:
            raise Violation("line %d: refs_max=0 with nonempty_tiles=%d "
                            "refs_total=%d" % (n, rec["nonempty_tiles"],
                                               rec["refs_total"]))
    if rec["refs_total"] < rec["refs_max"]:
        raise Violation("line %d: refs_total=%d < refs_max=%d"
                        % (n, rec["refs_total"], rec["refs_max"]))
    if rec["refs_total"] < rec["nonempty_tiles"]:
        raise Violation("line %d: refs_total=%d < nonempty_tiles=%d "
                        "(a non-empty tile holds at least one reference)"
                        % (n, rec["refs_total"], rec["nonempty_tiles"]))
    ntiles = rec["tiles_x"] * rec["tiles_y"]
    if rec["nonempty_tiles"] > ntiles:
        raise Violation("line %d: nonempty_tiles=%d exceeds grid %dx%d"
                        % (n, rec["nonempty_tiles"], rec["tiles_x"],
                           rec["tiles_y"]))
    marked = rec["class"] != "none"
    if marked and ntiles == 0:
        raise Violation("line %d: class=%s but the tile grid is empty"
                        % (n, rec["class"]))
    if not marked:
        for k in ("batches", "fs_binaries", "tris_postclip", "tiles_x",
                  "tiles_y", "nonempty_tiles", "refs_total", "refs_max"):
            if rec[k]:
                raise Violation("line %d: class=none but %s=%d"
                                % (n, k, rec[k]))
    if rec["batches"] and not marked:
        raise Violation("line %d: batches=%d without a marked class"
                        % (n, rec["batches"]))


def analyze(stream, name):
    """Validate one log. Returns the list of scope records on success and
    raises Violation on the first violation."""
    tile = None
    scopes = []
    prev_submit = 0
    lineno = 0
    for raw in stream:
        lineno += 1
        line = raw.rstrip("\n")
        if not line.startswith(PREFIX):
            continue
        fields = line.split()[1:]
        if fields and fields[0] == "header":
            if tile is not None:
                raise Violation("line %d: second header" % lineno)
            kv = parse_kv(fields[1:], lineno)
            if sorted(kv) != ["tile", "timing"]:
                raise Violation("line %d: header keys %s, expected "
                                "tile+timing" % (lineno, sorted(kv)))
            if kv.get("timing") != "invalid":
                raise Violation("line %d: header must declare timing=invalid"
                                % lineno)
            try:
                tile = int(kv["tile"])
            except ValueError:
                raise Violation("line %d: header tile=%r is not an integer"
                                % (lineno, kv["tile"]))
            if tile <= 0:
                raise Violation("line %d: header tile=%d must be positive"
                                % (lineno, tile))
            continue
        if tile is None:
            raise Violation("line %d: scope line before the header" % lineno)
        rec = parse_scope_line(fields, lineno)
        if rec["scope"] != len(scopes) + 1:
            raise Violation("line %d: scope=%d, expected %d (ordinals must "
                            "be the contiguous sequence 1..N)"
                            % (lineno, rec["scope"], len(scopes) + 1))
        if rec["submit"] < prev_submit:
            raise Violation("line %d: submit=%d went backwards from %d"
                            % (lineno, rec["submit"], prev_submit))
        if rec["submit"] < 1:
            raise Violation("line %d: submit=%d, a scope always executes "
                            "inside a command-bearing submit"
                            % (lineno, rec["submit"]))
        prev_submit = rec["submit"]
        check_scope(rec, lineno, tile)
        scopes.append(rec)
    if tile is None:
        raise Violation("%s: no tilecensus-scope-v1 header found" % name)
    if not scopes:
        raise Violation("%s: header but no scope lines -- the census "
                        "reported nothing" % name)
    return scopes


def summarize(scopes, name):
    total_refs = sum(s["refs_total"] for s in scopes)
    marked = sum(1 for s in scopes if s["class"] != "none")
    print("%s: OK: %d scopes (%d marked), %d draws + %d solo, "
          "%d tris post-clip, %d tile refs, refs_max %d"
          % (name, len(scopes), marked,
             sum(s["draws"] for s in scopes),
             sum(s["solo_draws"] for s in scopes),
             sum(s["tris_postclip"] for s in scopes),
             total_refs, max(s["refs_max"] for s in scopes)))


def main(argv):
    paths = argv[1:]
    if not paths:
        sys.stderr.write(__doc__)
        return 2
    status = 0
    for path in paths:
        try:
            if path == "-":
                scopes = analyze(sys.stdin, "<stdin>")
            else:
                with open(path, "r", errors="replace") as f:
                    scopes = analyze(f, path)
        except Violation as v:
            sys.stderr.write("%s: INVALID: %s\n" % (path, v))
            status = 1
            continue
        except OSError as e:
            sys.stderr.write("%s: unreadable: %s\n" % (path, e))
            return 2
        summarize(scopes, path)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
