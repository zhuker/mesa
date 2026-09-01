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

A build with the renderer2 M1 run segmentation census also writes, after each
scope line, one ``tilecensus-runs-v1`` summary and one ``tilecensus-run-v1``
line per reported run. When any of those appear the log is checked for them
too -- a log with none is an M0-era log and is still valid:

  - every scope has exactly one summary, in its own ordinal, carrying the
    same submit as its scope line;
  - reported = min(runs, RUN_SLOTS) run lines follow it, numbered 0..N-1;
  - the run lines' draws, batches and references sum to the summary's;
  - the summary's references equal the scope line's refs_total -- the run
    census and the tile census count the same references;
  - the summary accounts for at least the draws the scope line names, since
    a run holds every draw and the scope line only names the two paths the
    tile census itself counts;
  - LEADING_OPAQUE_PREFIX is consistent: prefix_class=opaque means the prefix
    is exactly run 0 and run 0 is opaque, prefix_class=none means the prefix
    is empty, and either way the prefix never exceeds the scope;
  - classes and end reasons come from the closed sets the driver can emit,
    which never include the "open" a still-running run would carry.

Usage:
    cp_tile_census_analyze.py LOG [LOG ...]
    ... | cp_tile_census_analyze.py -

Exit 0 when every log is valid; 1 with one line per violation on stderr
when any is not; 2 when a log could not be read at all. Non-census stderr
lines are ignored -- the driver's other reports share the stream.
"""

import sys

PREFIX = "tilecensus-scope-v1"
RUNS_PREFIX = "tilecensus-runs-v1"
RUN_PREFIX = "tilecensus-run-v1"
HIST_BUCKETS = 17
SCOPE_KEYS = [
    "scope", "submit", "class", "draws", "solo_draws", "batches",
    "solo_batches", "fs_binaries", "tris_postclip", "tile", "tiles_x",
    "tiles_y", "nonempty_tiles", "refs_total", "refs_max", "hist",
]
INT_KEYS = [k for k in SCOPE_KEYS if k not in ("class", "hist")]
CLASSES = ("none", "opaque", "blended", "mixed")

# CP_TILE_CENSUS_RUNS: run slots the driver reports one by one, the last of
# them the aggregate of every run past the table.
RUN_SLOTS = 64
RUNS_KEYS = [
    "scope", "submit", "runs", "reported", "draws", "batches", "refs",
    "prefix_class", "prefix_draws", "prefix_batches", "prefix_refs",
]
RUNS_INT_KEYS = [k for k in RUNS_KEYS if k != "prefix_class"]
RUN_KEYS = ["scope", "run", "class", "draws", "batches", "refs", "end"]
RUN_INT_KEYS = [k for k in RUN_KEYS if k not in ("class", "end")]
# enum cp_tile_census_run_class, plus the "mixed" the aggregate slot reports
# when it folds runs of both classes.
RUN_CLASSES = ("opaque", "blended", "other", "mixed")
# enum cp_tile_census_run_end. "open" is deliberately absent: the scope report
# closes the last run before printing, so a reported run always has a reason.
RUN_ENDS = ("state", "class", "cut_map", "cut_copy", "cut_flush",
            "cut_compute", "scope_end", "aggregate")
PREFIX_CLASSES = ("opaque", "none")


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


def parse_fixed_line(fields, lineno, keys, int_keys, kind):
    """Parse a KEY=VALUE line with a fixed key set into ints plus strings."""
    kv = parse_kv(fields, lineno)
    if sorted(kv) != sorted(keys):
        missing = sorted(set(keys) - set(kv))
        extra = sorted(set(kv) - set(keys))
        raise Violation("line %d: %s key set mismatch (missing %s, extra %s)"
                        % (lineno, kind, missing, extra))
    rec = {}
    for k in keys:
        if k not in int_keys:
            rec[k] = kv[k]
            continue
        try:
            rec[k] = int(kv[k])
        except ValueError:
            raise Violation("line %d: %s=%r is not an integer"
                            % (lineno, k, kv[k]))
        if rec[k] < 0:
            raise Violation("line %d: %s=%d is negative" % (lineno, k, rec[k]))
    return rec


def check_runs(runs, scope, lineno):
    """The summary line against the scope line it belongs to."""
    n = lineno
    if runs["scope"] != scope["scope"]:
        raise Violation("line %d: runs scope=%d does not match the scope "
                        "line's %d" % (n, runs["scope"], scope["scope"]))
    if runs["submit"] != scope["submit"]:
        raise Violation("line %d: runs submit=%d but the scope line says %d"
                        % (n, runs["submit"], scope["submit"]))
    expect = min(runs["runs"], RUN_SLOTS)
    if runs["reported"] != expect:
        raise Violation("line %d: reported=%d, expected min(runs=%d, %d)=%d"
                        % (n, runs["reported"], runs["runs"], RUN_SLOTS,
                           expect))
    drawn = scope["draws"] + scope["solo_draws"]
    if not runs["runs"]:
        if runs["draws"] or runs["batches"] or runs["refs"]:
            raise Violation("line %d: runs=0 but draws=%d batches=%d refs=%d"
                            % (n, runs["draws"], runs["batches"],
                               runs["refs"]))
        if drawn:
            raise Violation("line %d: runs=0 but the scope drew %d draws"
                            % (n, drawn))
    elif not runs["draws"]:
        raise Violation("line %d: runs=%d with no draws in any of them"
                        % (n, runs["runs"]))
    if runs["draws"] < drawn:
        raise Violation("line %d: runs draws=%d but the scope line names %d "
                        "(a run holds every draw of the scope)"
                        % (n, runs["draws"], drawn))
    if runs["refs"] != scope["refs_total"]:
        raise Violation("line %d: runs refs=%d but the scope line's "
                        "refs_total=%d -- the two count the same references"
                        % (n, runs["refs"], scope["refs_total"]))
    if runs["prefix_class"] not in PREFIX_CLASSES:
        raise Violation("line %d: prefix_class=%r is not one of %s"
                        % (n, runs["prefix_class"], "/".join(PREFIX_CLASSES)))
    for k in ("draws", "batches", "refs"):
        if runs["prefix_" + k] > runs[k]:
            raise Violation("line %d: prefix_%s=%d exceeds the scope's %s=%d"
                            % (n, k, runs["prefix_" + k], k, runs[k]))
    if runs["prefix_class"] == "none" and (runs["prefix_draws"] or
                                           runs["prefix_batches"] or
                                           runs["prefix_refs"]):
        raise Violation("line %d: prefix_class=none but the prefix is not "
                        "empty" % n)
    if runs["prefix_class"] == "opaque" and not runs["prefix_draws"]:
        raise Violation("line %d: prefix_class=opaque with no draws in it"
                        % n)


def check_run_lines(runs, lines, lineno):
    """The run lines of one scope against their summary."""
    n = lineno
    if len(lines) != runs["reported"]:
        raise Violation("line %d: scope %d reported=%d but %d run lines "
                        "followed" % (n, runs["scope"], runs["reported"],
                                      len(lines)))
    draws = batches = refs = 0
    for i, r in enumerate(lines):
        if r["scope"] != runs["scope"]:
            raise Violation("line %d: run line scope=%d inside scope %d"
                            % (n, r["scope"], runs["scope"]))
        if r["run"] != i:
            raise Violation("line %d: run=%d, expected %d (run ordinals must "
                            "be 0..reported-1 in order)" % (n, r["run"], i))
        if r["class"] not in RUN_CLASSES:
            raise Violation("line %d: run class=%r is not one of %s"
                            % (n, r["class"], "/".join(RUN_CLASSES)))
        if r["end"] not in RUN_ENDS:
            raise Violation("line %d: end=%r is not one of %s"
                            % (n, r["end"], "/".join(RUN_ENDS)))
        if not r["draws"]:
            raise Violation("line %d: run %d holds no draws" % (n, r["run"]))
        draws += r["draws"]
        batches += r["batches"]
        refs += r["refs"]
    # The sums close even when runs overflow the table, because the last slot
    # is their aggregate rather than a truncation.
    if draws != runs["draws"]:
        raise Violation("line %d: run lines sum to %d draws, summary says "
                        "%d" % (n, draws, runs["draws"]))
    if batches != runs["batches"]:
        raise Violation("line %d: run lines sum to %d batches, summary "
                        "says %d" % (n, batches, runs["batches"]))
    if refs != runs["refs"]:
        raise Violation("line %d: run lines sum to %d refs, summary says "
                        "%d" % (n, refs, runs["refs"]))
    if not lines:
        return
    first = lines[0]
    opaque = runs["prefix_class"] == "opaque"
    if opaque != (first["class"] == "opaque"):
        raise Violation("line %d: prefix_class=%s but run 0 is class=%s"
                        % (n, runs["prefix_class"], first["class"]))
    if opaque and (runs["prefix_draws"] != first["draws"] or
                   runs["prefix_batches"] != first["batches"] or
                   runs["prefix_refs"] != first["refs"]):
        raise Violation("line %d: the leading opaque prefix is not run 0 "
                        "(prefix %d/%d/%d against run 0 %d/%d/%d)"
                        % (n, runs["prefix_draws"], runs["prefix_batches"],
                           runs["prefix_refs"], first["draws"],
                           first["batches"], first["refs"]))


def analyze(stream, name):
    """Validate one log. Returns the list of scope records on success and
    raises Violation on the first violation."""
    tile = None
    scopes = []
    prev_submit = 0
    lineno = 0
    pending = None      # (runs record, its line number, its run lines)

    def close_pending(at):
        if pending is not None:
            check_run_lines(pending[0], pending[2], at)

    for raw in stream:
        lineno += 1
        line = raw.rstrip("\n")
        token = line.split(None, 1)[0] if line.split(None, 1) else ""
        if token == RUNS_PREFIX:
            if tile is None:
                raise Violation("line %d: run summary before the header"
                                % lineno)
            rec = parse_fixed_line(line.split()[1:], lineno, RUNS_KEYS,
                                   RUNS_INT_KEYS, "runs")
            if not scopes:
                raise Violation("line %d: run summary before any scope line"
                                % lineno)
            if scopes[-1].get("runs") is not None:
                raise Violation("line %d: a second run summary for scope %d"
                                % (lineno, scopes[-1]["scope"]))
            close_pending(lineno)
            check_runs(rec, scopes[-1], lineno)
            scopes[-1]["runs"] = rec
            scopes[-1]["run_lines"] = []
            pending = (rec, lineno, scopes[-1]["run_lines"])
            continue
        if token == RUN_PREFIX:
            if pending is None:
                raise Violation("line %d: run line without a run summary"
                                % lineno)
            pending[2].append(parse_fixed_line(line.split()[1:], lineno,
                                               RUN_KEYS, RUN_INT_KEYS, "run"))
            continue
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
        close_pending(lineno)
        pending = None
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
        rec["runs"] = None
        rec["run_lines"] = []
        scopes.append(rec)
    close_pending(lineno)
    if tile is None:
        raise Violation("%s: no tilecensus-scope-v1 header found" % name)
    if not scopes:
        raise Violation("%s: header but no scope lines -- the census "
                        "reported nothing" % name)
    # A log either carries the run census throughout or not at all: an M0-era
    # log has no run summaries, and a newer one has exactly one per scope.
    if any(s["runs"] is not None for s in scopes):
        for s in scopes:
            if s["runs"] is None:
                raise Violation("%s: scope %d has no tilecensus-runs-v1 line "
                                "while other scopes do" % (name, s["scope"]))
    return scopes


def pct(values, p):
    if not values:
        return 0
    ordered = sorted(values)
    i = min(len(ordered) - 1, (p * len(ordered)) // 100)
    return ordered[i]


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

    drawing = [s for s in scopes if s["runs"] and s["runs"]["runs"]]
    if not drawing:
        return
    runs = [s["runs"]["runs"] for s in drawing]
    opaque_first = [s for s in drawing
                    if s["runs"]["prefix_class"] == "opaque"]
    # The M1 falsification number: what share of a drawing scope's references
    # the leading opaque run holds. Scopes with no references at all are left
    # out of the share, since they answer nothing about it.
    shares = [100.0 * s["runs"]["prefix_refs"] / s["runs"]["refs"]
              for s in drawing if s["runs"]["refs"]]
    unnamed = sum(s["runs"]["draws"] - s["draws"] - s["solo_draws"]
                  for s in drawing)
    print("%s: runs: %d drawing scopes, runs/scope p50 %d p90 %d p99 %d max "
          "%d; opens opaque %d (%.1f%%); leading opaque prefix share of "
          "refs p50 %.1f%% p90 %.1f%%; %d draws on neither census path"
          % (name, len(drawing), pct(runs, 50), pct(runs, 90), pct(runs, 99),
             max(runs), len(opaque_first),
             100.0 * len(opaque_first) / len(drawing),
             pct(shares, 50) if shares else 0.0,
             pct(shares, 90) if shares else 0.0, unnamed))

    ends = {}
    for s in drawing:
        for r in s["run_lines"]:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
    print("%s: run ends: %s"
          % (name, " ".join("%s=%d" % kv for kv in sorted(ends.items()))))


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
