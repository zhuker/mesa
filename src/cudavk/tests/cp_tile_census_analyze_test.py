#!/usr/bin/env python3
"""Unit test for cp_tile_census_analyze.py: a valid synthetic log passes,
and each validated invariant fails on a log built to break exactly it."""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp_tile_census_analyze as an


HEADER = "tilecensus-scope-v1 header tile=16 timing=invalid"


def scope_line(**over):
    kv = {
        "scope": 1, "submit": 1, "class": "opaque", "draws": 12,
        "solo_draws": 0, "batches": 3, "solo_batches": 0, "fs_binaries": 2,
        "tris_postclip": 4000, "tile": 16, "tiles_x": 80, "tiles_y": 45,
        # 5 tiles of 1 ref, 2 tiles in [4,8), one tile at 70000 refs
        "nonempty_tiles": 8, "refs_total": 5 + 2 * 5 + 70000, "refs_max": 70000,
        "hist": "5,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1",
    }
    kv.update(over)
    return "tilecensus-scope-v1 " + " ".join(
        "%s=%s" % (k, kv[k]) for k in an.SCOPE_KEYS)


def runs_line(**over):
    """The per-scope run summary that goes with scope_line()'s defaults."""
    kv = {
        "scope": 1, "submit": 1, "runs": 2, "reported": 2, "draws": 12,
        "batches": 3, "refs": 5 + 2 * 5 + 70000,
        "prefix_class": "opaque", "prefix_draws": 5, "prefix_batches": 2,
        "prefix_refs": 70010,
    }
    kv.update(over)
    return "tilecensus-runs-v1 " + " ".join(
        "%s=%s" % (k, kv[k]) for k in an.RUNS_KEYS)


def run_line(**over):
    kv = {
        "scope": 1, "run": 0, "class": "opaque", "draws": 5, "batches": 2,
        "refs": 70010, "end": "class",
    }
    kv.update(over)
    return "tilecensus-run-v1 " + " ".join(
        "%s=%s" % (k, kv[k]) for k in an.RUN_KEYS)


def censused(**over):
    """A valid scope line, its summary and its two run lines."""
    scope = over.pop("scope", 1)
    return [scope_line(scope=scope, **over),
            runs_line(scope=scope,
                      submit=over.get("submit", 1)),
            run_line(scope=scope),
            run_line(scope=scope, run=1, **{"class": "blended", "draws": 7,
                                            "batches": 1, "refs": 5,
                                            "end": "scope_end"})]


def run(lines):
    return an.analyze(io.StringIO("\n".join(lines) + "\n"), "<test>")


class TestAnalyzer(unittest.TestCase):
    def assertViolation(self, lines, fragment):
        with self.assertRaises(an.Violation) as cm:
            run(lines)
        self.assertIn(fragment, str(cm.exception))

    def test_valid_log(self):
        empty = scope_line(scope=2, submit=2, **{
            "class": "none", "draws": 0, "batches": 0, "fs_binaries": 0,
            "tris_postclip": 0, "tiles_x": 0, "tiles_y": 0,
            "nonempty_tiles": 0, "refs_total": 0, "refs_max": 0,
            "solo_draws": 4, "solo_batches": 1,
            "hist": ",".join(["0"] * an.HIST_BUCKETS)})
        blended = scope_line(scope=3, submit=2, **{
            "class": "blended", "nonempty_tiles": 1, "refs_total": 3,
            "refs_max": 3,
            "hist": "0,1" + ",0" * (an.HIST_BUCKETS - 2)})
        scopes = run([HEADER, scope_line(), empty, blended,
                      "cudavk: some unrelated stderr line"])
        self.assertEqual(len(scopes), 3)
        self.assertEqual(scopes[0]["refs_max"], 70000)
        self.assertEqual(scopes[1]["class"], "none")

    def test_interleaved_noise_ignored(self):
        scopes = run(["noise", HEADER, "tilecensus tile=16 binds=1",
                      scope_line()])
        self.assertEqual(len(scopes), 1)

    def test_missing_header(self):
        self.assertViolation([scope_line()], "before the header")

    def test_no_scope_lines(self):
        self.assertViolation([HEADER], "no scope lines")

    def test_header_without_timing_invalid(self):
        self.assertViolation(
            ["tilecensus-scope-v1 header tile=16 timing=valid", scope_line()],
            "timing=invalid")

    def test_second_header(self):
        self.assertViolation([HEADER, scope_line(), HEADER], "second header")

    def test_scope_not_contiguous(self):
        self.assertViolation([HEADER, scope_line(), scope_line(scope=3)],
                             "contiguous")

    def test_scope_duplicate(self):
        self.assertViolation([HEADER, scope_line(), scope_line(scope=1)],
                             "contiguous")

    def test_submit_backwards(self):
        self.assertViolation(
            [HEADER, scope_line(submit=5),
             scope_line(scope=2, submit=4)], "went backwards")

    def test_submit_zero(self):
        self.assertViolation([HEADER, scope_line(submit=0)],
                             "command-bearing")

    def test_tile_mismatch(self):
        self.assertViolation([HEADER, scope_line(tile=32)],
                             "disagrees with header")

    def test_hist_sum_mismatch(self):
        self.assertViolation([HEADER, scope_line(nonempty_tiles=9)],
                             "nonempty_tiles")

    def test_hist_bucket_count(self):
        self.assertViolation([HEADER, scope_line(hist="1,2,3")],
                             "buckets")

    def test_refs_max_outside_top_bucket(self):
        # top non-empty bucket says >= 65536 but refs_max says 40
        self.assertViolation([HEADER, scope_line(refs_max=40)],
                             "top non-empty bucket")

    def test_refs_total_below_nonempty(self):
        self.assertViolation(
            [HEADER, scope_line(refs_total=7, refs_max=70000)],
            "refs_total")

    def test_class_none_with_content(self):
        self.assertViolation(
            [HEADER, scope_line(**{"class": "none"})], "class=none")

    def test_unknown_class(self):
        self.assertViolation([HEADER, scope_line(**{"class": "opaqueish"})],
                             "class=")

    def test_missing_key(self):
        line = scope_line().replace(" refs_max=70000", "")
        self.assertViolation([HEADER, line], "key set mismatch")

    def test_duplicate_key(self):
        self.assertViolation([HEADER, scope_line() + " draws=12"],
                             "duplicate key")

    def test_negative_value(self):
        self.assertViolation([HEADER, scope_line(draws=-1)], "negative")

    def test_nonempty_exceeds_grid(self):
        self.assertViolation(
            [HEADER, scope_line(tiles_x=2, tiles_y=2, nonempty_tiles=8)],
            "exceeds grid")

    def test_marked_needs_grid(self):
        self.assertViolation(
            [HEADER, scope_line(tiles_x=0, tiles_y=0, nonempty_tiles=0,
                                refs_total=0, refs_max=0,
                                hist=",".join(["0"] * an.HIST_BUCKETS))],
            "grid is empty")


class TestRunCensus(unittest.TestCase):
    """The renderer2 M1 run segmentation lines."""

    def assertViolation(self, lines, fragment):
        with self.assertRaises(an.Violation) as cm:
            run(lines)
        self.assertIn(fragment, str(cm.exception))

    def test_valid_run_census(self):
        scopes = run([HEADER] + censused())
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0]["runs"]["runs"], 2)
        self.assertEqual(len(scopes[0]["run_lines"]), 2)
        self.assertEqual(scopes[0]["run_lines"][1]["end"], "scope_end")

    def test_m0_log_without_runs_is_still_valid(self):
        scopes = run([HEADER, scope_line()])
        self.assertEqual(len(scopes), 1)
        self.assertIsNone(scopes[0]["runs"])

    def test_partial_run_census_rejected(self):
        empty = scope_line(scope=2, submit=2, **{
            "class": "none", "draws": 0, "batches": 0, "fs_binaries": 0,
            "tris_postclip": 0, "tiles_x": 0, "tiles_y": 0,
            "nonempty_tiles": 0, "refs_total": 0, "refs_max": 0,
            "solo_draws": 0, "solo_batches": 0,
            "hist": ",".join(["0"] * an.HIST_BUCKETS)})
        self.assertViolation([HEADER] + censused() + [empty],
                             "has no tilecensus-runs-v1 line")

    def test_scope_with_no_draws_reports_zero_runs(self):
        empty = scope_line(scope=2, submit=2, **{
            "class": "none", "draws": 0, "batches": 0, "fs_binaries": 0,
            "tris_postclip": 0, "tiles_x": 0, "tiles_y": 0,
            "nonempty_tiles": 0, "refs_total": 0, "refs_max": 0,
            "solo_draws": 0, "solo_batches": 0,
            "hist": ",".join(["0"] * an.HIST_BUCKETS)})
        scopes = run([HEADER] + censused() +
                     [empty, runs_line(scope=2, submit=2, runs=0, reported=0,
                                       draws=0, batches=0, refs=0,
                                       prefix_class="none", prefix_draws=0,
                                       prefix_batches=0, prefix_refs=0)])
        self.assertEqual(len(scopes), 2)
        self.assertEqual(scopes[1]["runs"]["runs"], 0)

    def test_runs_zero_with_draws(self):
        self.assertViolation(
            [HEADER, scope_line(),
             runs_line(runs=0, reported=0, draws=0, batches=0, refs=0,
                       prefix_class="none", prefix_draws=0, prefix_batches=0,
                       prefix_refs=0)],
            "the scope drew")

    def test_runs_key_set(self):
        line = runs_line().replace(" prefix_refs=70010", "")
        self.assertViolation([HEADER, scope_line(), line],
                             "runs key set mismatch")

    def test_run_key_set(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(), run_line() + " extra=1"],
            "run key set mismatch")

    def test_reported_not_min_of_runs(self):
        self.assertViolation([HEADER, scope_line(), runs_line(reported=1)],
                             "expected min(runs=2")

    def test_run_line_count_mismatch(self):
        self.assertViolation([HEADER, scope_line(), runs_line(), run_line()],
                             "run lines followed")

    def test_run_ordinals_not_contiguous(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(), run_line(),
             run_line(run=2, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "run ordinals must be")

    def test_run_draws_do_not_sum(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(), run_line(),
             run_line(run=1, **{"class": "blended", "draws": 6, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "run lines sum to 11 draws")

    def test_run_refs_do_not_sum(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(), run_line(),
             run_line(run=1, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 4, "end": "scope_end"})],
            "run lines sum to 70014 refs")

    def test_refs_disagree_with_scope_line(self):
        self.assertViolation([HEADER, scope_line(), runs_line(refs=1)],
                             "the two count the same references")

    def test_runs_draws_below_scope_draws(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(draws=11)],
            "a run holds every draw")

    def test_unknown_end_reason(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(), run_line(end="open"),
             run_line(run=1, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "end='open'")

    def test_unknown_run_class(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(),
             run_line(**{"class": "opaqueish"}),
             run_line(run=1, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "run class=")

    def test_empty_run(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(),
             run_line(draws=0, batches=0, refs=0),
             run_line(run=1, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "holds no draws")

    def test_prefix_class_disagrees_with_run_zero(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(prefix_class="none",
                                             prefix_draws=0,
                                             prefix_batches=0,
                                             prefix_refs=0),
             run_line(),
             run_line(run=1, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "prefix_class=none but run 0 is class=opaque")

    def test_prefix_is_not_run_zero(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(prefix_draws=4),
             run_line(),
             run_line(run=1, **{"class": "blended", "draws": 7, "batches": 1,
                                "refs": 5, "end": "scope_end"})],
            "leading opaque prefix is not run 0")

    def test_prefix_exceeds_scope(self):
        self.assertViolation([HEADER, scope_line(), runs_line(prefix_draws=13)],
                             "exceeds the scope")

    def test_prefix_none_but_not_empty(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(prefix_class="none")],
            "prefix is not empty")

    def test_prefix_opaque_with_no_draws(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(prefix_draws=0)],
            "prefix_class=opaque with no draws")

    def test_unknown_prefix_class(self):
        self.assertViolation(
            [HEADER, scope_line(), runs_line(prefix_class="blended")],
            "prefix_class=")

    def test_runs_before_any_scope(self):
        self.assertViolation([HEADER, runs_line()],
                             "before any scope line")

    def test_second_runs_line(self):
        self.assertViolation([HEADER, scope_line(), runs_line(), runs_line()],
                             "second run summary")

    def test_run_line_without_summary(self):
        self.assertViolation([HEADER, scope_line(), run_line()],
                             "without a run summary")

    def test_runs_submit_mismatch(self):
        self.assertViolation([HEADER, scope_line(), runs_line(submit=2)],
                             "the scope line says")

    def test_aggregate_slot(self):
        """More runs than slots: the last line aggregates the rest, so the
        sums still close."""
        scopes = run([HEADER, scope_line(draws=200, solo_draws=0),
                      runs_line(runs=70, reported=an.RUN_SLOTS, draws=200,
                                batches=2, refs=70015),
                      run_line()] +
                     [run_line(run=i, draws=1, batches=0, refs=0,
                               end="state") for i in range(1, 63)] +
                     [run_line(run=63, **{"class": "mixed", "draws": 133,
                                          "batches": 0, "refs": 5,
                                          "end": "aggregate"})])
        self.assertEqual(scopes[0]["runs"]["runs"], 70)
        self.assertEqual(scopes[0]["run_lines"][-1]["class"], "mixed")

    def test_aggregate_sums_must_still_close(self):
        self.assertViolation(
            [HEADER, scope_line(draws=200, solo_draws=0),
             runs_line(runs=70, reported=an.RUN_SLOTS, draws=200, batches=2,
                       refs=70015),
             run_line()] +
            [run_line(run=i, draws=1, batches=0, refs=0, end="state")
             for i in range(1, 63)] +
            [run_line(run=63, **{"class": "mixed", "draws": 1, "batches": 0,
                                 "refs": 5, "end": "aggregate"})],
            "run lines sum to 68 draws")


if __name__ == "__main__":
    unittest.main()
