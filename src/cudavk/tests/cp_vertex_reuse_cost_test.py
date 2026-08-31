#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cp_vertex_reuse_cost", HERE / "cp_vertex_reuse_cost.py")
assert SPEC and SPEC.loader
cost = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cost
SPEC.loader.exec_module(cost)


def launch(name, ds, de, rs, cid, grid=1, stream=13):
    return cost.Launch(name, cost.Interval(ds, de), cost.Interval(rs, rs + 1),
                       cid, grid, 256, stream, "cuLaunchKernel")


class VertexReuseCostTest(unittest.TestCase):
    def submit_rows(self):
        return [f"{1000 + i} {i}" for i in range(cost.EXPECTED_EXTERNAL_ROWS)]

    def test_submit_parser_accepts_unterminated_torn_tail(self):
        lines = self.submit_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submit.txt"
            path.write_text("\n".join(lines) + "\n1503")
            values, torn = cost.parse_submit_timestamps(path)
        self.assertEqual(len(values), cost.EXPECTED_EXTERNAL_ROWS)
        self.assertEqual(torn, "1503")

    def test_submit_parser_rejects_terminated_numeric_only_row(self):
        lines = self.submit_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submit.txt"
            path.write_text("\n".join(lines) + "\n1503\n")
            with self.assertRaisesRegex(cost.CostError, "malformed timestamp row"):
                cost.parse_submit_timestamps(path)

    def test_submit_parser_rejects_malformed_interior_row(self):
        lines = self.submit_rows()
        lines[100] = "1503"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submit.txt"
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(cost.CostError, "malformed timestamp row"):
                cost.parse_submit_timestamps(path)

    def test_vs_classifier_uses_immediate_same_stream_lead_lag_and_absolute_ordinal(self):
        times = [i * 100 for i in range(4000)]
        # Both VS launches issue in external submit 2785 => internal 2786.  The
        # first is outside the device window but still consumes batch ordinal 1.
        # Stream-14 launches interleave globally and must steal neither the
        # stream-13 successor nor its fetch predecessor.
        launches = [
            launch("main", 10, 20, 278550, 1, grid=2, stream=13),
            launch("cp_cache_convert", 30, 31, 278551, 2, stream=14),
            launch("cp_clip_rast_fused", 21, 25, 278560, 3, stream=13),
            launch("cp_vertex_fetch", 105, 108, 278570, 4, grid=3, stream=13),
            launch("cp_clip_triangles", 40, 41, 278571, 5, stream=14),
            launch("main", 109, 120, 278580, 6, grid=3, stream=13),
            launch("cp_vertex_fetch", 42, 43, 278581, 7, stream=14),
            launch("cp_clip_triangles", 121, 130, 278590, 8, stream=13),
        ]
        units = cost.classify_vs_launches(launches, times, 100, 200)
        self.assertEqual(len(units), 1)
        submit, ordinal, vs, fetch = units[0]
        self.assertEqual((submit, ordinal, vs.correlation_id, fetch.correlation_id),
                         (2786, 2, 6, 4))

    def test_vs_classifier_does_not_skip_same_stream_main(self):
        times = [i * 100 for i in range(4000)]
        launches = [
            launch("main", 105, 110, 278550, 1, stream=13),
            launch("main", 111, 115, 278560, 2, stream=13),
            launch("cp_clip_triangles", 116, 120, 278570, 3, stream=13),
        ]
        units = cost.classify_vs_launches(launches, times, 100, 200)
        self.assertEqual([(u[2].correlation_id) for u in units], [2])

    def test_record_map_checks_grid_stream_and_fetch(self):
        vs = launch("main", 100, 120, 278580, 4, grid=3, stream=14)
        fetch = launch("cp_vertex_fetch", 90, 99, 278570, 3, grid=3, stream=14)
        record = {"submit": 2786, "batch": 1, "total": 700,
                  "bank": 1, "fetch_runs": 1}
        old = cost.EXPECTED_RECORDS
        try:
            cost.EXPECTED_RECORDS = 1
            mapped = cost.map_records([record], [(2786, 1, vs, fetch)])
            self.assertEqual(mapped[0][1].correlation_id, 4)
            bad = dict(record, bank=0)
            with self.assertRaisesRegex(cost.CostError, "stream"):
                cost.map_records([bad], [(2786, 1, vs, fetch)])
        finally:
            cost.EXPECTED_RECORDS = old

    def test_fractional_union_exclusive_variants(self):
        a, b, other = cost.Interval(0, 10), cost.Interval(5, 15), cost.Interval(8, 12)
        result = cost.sweep_pool([(a, Fraction(1, 2)), (b, Fraction(3, 4))],
                                 [a, b, other])
        self.assertEqual(result["pool_sum_ns"], 20)
        self.assertEqual(result["pool_union_ns"], 15)
        self.assertEqual(result["pool_union_exclusive_ns"], 11)
        variants = result["collectible_union_exclusive"]
        self.assertEqual(variants["generous_sum_cap"]["ns_total"], 7.75)
        self.assertEqual(variants["max"]["ns_total"], 7.0)
        self.assertEqual(variants["average"]["ns_total"], 6.625)
        self.assertEqual(variants["min"]["ns_total"], 6.25)

    def test_real_calibrated_join_when_inputs_exist(self):
        paths = [Path("/tmp/postdead/trace.sqlite"),
                 Path("/tmp/postdead/submit-ts.txt"),
                 Path("/tmp/vertex-reuse-f3/run1/stderr"),
                 Path("/tmp/vertex-reuse-f3/run2/stderr")]
        if not all(path.exists() for path in paths):
            self.skipTest("calibrated diagnostic inputs are absent")
        result = cost.run(paths[0], paths[1], paths[2:])
        self.assertEqual(result["pool_calibration"], cost.EXPECTED_POOL_CALIBRATION)
        self.assertEqual(result["cost"]["exact_calibration_sha256"],
                         cost.EXPECTED_COST_FINGERPRINT)
        self.assertEqual(result["mapping"]["mapped_batches"], 19929)
        self.assertEqual(result["mapping"]["mapped_fetch_rows"], 3087)
        self.assertAlmostEqual(
            result["cost"]["raw_duplicate_weighted_sum_ms_per_frame"],
            0.5825173298014584, places=14)
        self.assertAlmostEqual(
            result["gate"]["value_ms_per_frame"],
            0.4375026533628589, places=14)
        self.assertTrue(result["gate"]["rejection_condition_holds"])


if __name__ == "__main__":
    unittest.main()
