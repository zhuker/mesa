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


if __name__ == "__main__":
    unittest.main()
