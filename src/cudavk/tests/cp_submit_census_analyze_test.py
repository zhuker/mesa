#!/usr/bin/env python3
"""Deterministic CPU tests for cp_submit_census_analyze.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cp_submit_census_analyze as census

HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "cp_submit_census_analyze.py"


def record(index, *, marker=False, **changes):
    values = {
        "index": index,
        "internal_ordinal": index,
        "initialized": 1,
        "issue_begin_ns": 1000 + index * 100,
        "issue_end_ns": 1010 + index * 100,
        "complete_ns": 2000 + index * 200,
        "completion": 1,
        "status": 0,
        "command_buffer_count": 0 if marker else 1,
        "wait_count": index % 2,
        "signal_count": 1,
        "marker": int(marker),
        "queued_depth": 64 if index == 4 else index,
        "depth_capped": int(index == 4),
    }
    values.update(changes)
    return census.RECORD_PREFIX + " " + " ".join(
        f"{name}={value}" for name, value in values.items()
    )


def valid_text():
    rows = ["unrelated application output"]
    rows.extend(record(i, marker=i in (1, 6)) for i in range(8))
    rows.append(
        census.SUMMARY_PREFIX
        + " result_valid=1 capacity=131072 reserved=8 complete=8 overflow=0"
    )
    return "\n".join(rows) + "\n"


class ParseTests(unittest.TestCase):
    def test_valid_census_and_exact_target_range(self):
        records, summary = census.parse_census(valid_text().splitlines())
        result = census.analyze(records, summary, 2, 3)
        self.assertTrue(result["result_valid"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["selection"]["internal_target_start"], 2)
        self.assertEqual(result["selection"]["internal_target_end"], 7)
        self.assertEqual(result["selection"]["selected_command_record_count"], 5)
        self.assertEqual(result["selection"]["filtered_marker_record_count"], 2)
        # (complete[7] - complete[2]) / (3 - 1) = 500 ns.
        self.assertEqual(result["diagnostic_completion_frame_interval_ms"], 0.0005)
        self.assertEqual(result["issue_duration"]["count"], 5)
        self.assertEqual(result["issue_duration"]["mean_ns"], 10.0)
        self.assertEqual(result["runtime_queued_depth"]["capped_record_count"], 1)
        self.assertEqual(result["runtime_queued_depth"]["p95"], 64)
        self.assertNotIn("p95_ns", result["runtime_queued_depth"])
        self.assertTrue(result["runtime_queued_depth"]["values_are_lower_bounds"])

    def test_key_value_and_two_column_shim(self):
        keyed = [
            f"CUDAVK_SUBMIT_SHIM_TIMESTAMP timestamp_ns={5000 + 300*i} "
            f"submit_index={10 + i}"
            for i in range(5)
        ]
        parsed = census.parse_shim(keyed)
        self.assertEqual(parsed[-1], census.ShimTimestamp(6200, 14))
        plain = census.parse_shim(
            f"{5000 + 300*i} {10 + i}" for i in range(5)
        )
        self.assertEqual(parsed, plain)

        records, summary = census.parse_census(valid_text().splitlines())
        result = census.analyze(records, summary, 2, 3, parsed, 10)
        enqueue = result["external_enqueue"]
        self.assertEqual(enqueue["submit_count"], 5)
        self.assertEqual(enqueue["submits_per_frame"], 2)
        self.assertEqual(enqueue["diagnostic_enqueue_frame_interval_ms"], 0.0006)

    def test_one_truncated_final_shim_token_is_ignored(self):
        lines = [f"{5000 + 300*i} {10 + i}\n" for i in range(5)]
        lines.append("1527")
        parsed = census.parse_shim(lines)
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[-1].submit_index, 14)
        with self.assertRaisesRegex(census.CensusError, "expected timestamp"):
            census.parse_shim(["5000 10\n", "1527\n", "6000 11\n"])
        with self.assertRaisesRegex(census.CensusError, "expected timestamp"):
            census.parse_shim(["5000 10\n", "1527\n"])

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "census.log"
            path.write_text(valid_text())
            run = subprocess.run(
                [sys.executable, str(PROGRAM), str(path),
                 "--internal-target-start", "2", "--target-frame-count", "3"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["selection"]["internal_target_end"], 7)
        self.assertIn("diagnostic_completion_frame_interval_ms", result)

    def test_external_alignment_is_explicit_and_exact(self):
        records, summary = census.parse_census(valid_text().splitlines())
        shim = census.parse_shim(f"{5000 + i} {10 + i}" for i in range(5))
        with self.assertRaisesRegex(census.CensusError,
                                   "external-target-start is required"):
            census.analyze(records, summary, 2, 3, shim)
        with self.assertRaisesRegex(census.CensusError, "expected exactly 5"):
            census.analyze(records, summary, 2, 3, shim[:-1], 10)
        with self.assertRaisesRegex(census.CensusError,
                                   "external target start 9 is absent"):
            census.analyze(records, summary, 2, 3, shim, 9)

    def test_target_start_must_be_command_bearing(self):
        records, summary = census.parse_census(valid_text().splitlines())
        with self.assertRaisesRegex(census.CensusError,
                                   "not a command-bearing record"):
            census.analyze(records, summary, 1, 3)
        with self.assertRaisesRegex(census.CensusError,
                                   "target-frame-count must be at least 2"):
            census.analyze(records, summary, 2, 1)


class RejectionTests(unittest.TestCase):
    def assert_rejected(self, old, new, message):
        text = valid_text().replace(old, new, 1)
        with self.assertRaisesRegex(census.CensusError, message):
            census.parse_census(text.splitlines())

    def test_incomplete_or_uninitialized_records(self):
        self.assert_rejected("initialized=1", "initialized=0", "not initialized")
        self.assert_rejected("completion=1", "completion=0", "not complete")
        self.assert_rejected("status=0", "status=-7", "status=-7")
        self.assert_rejected("issue_begin_ns=1000", "issue_begin_ns=0",
                             "uninitialized timestamp")

    def test_contiguity_and_monotonicity(self):
        self.assert_rejected("index=1 internal_ordinal=1",
                             "index=9 internal_ordinal=1", "expected index 1")
        self.assert_rejected("index=1 internal_ordinal=1",
                             "index=1 internal_ordinal=9",
                             "internal_ordinal does not equal index")
        self.assert_rejected("issue_begin_ns=1100", "issue_begin_ns=1005",
                             "issue ranges overlap")
        self.assert_rejected("complete_ns=2400", "complete_ns=100",
                             "completes before issue ends")

    def test_marker_and_summary(self):
        self.assert_rejected("command_buffer_count=0", "command_buffer_count=1",
                             "marker does not match")
        self.assert_rejected("queued_depth=0 depth_capped=0",
                             "queued_depth=3 depth_capped=1",
                             "depth_capped below the cap")
        self.assert_rejected("queued_depth=0 depth_capped=0",
                             "queued_depth=65 depth_capped=0",
                             "queued_depth above the cap")
        self.assert_rejected("result_valid=1", "result_valid=0",
                             "result_valid is false")
        self.assert_rejected("overflow=0", "overflow=1", "reports overflow=1")
        self.assert_rejected("reserved=8", "reserved=7", "parsed records=8")
        self.assert_rejected("complete=8", "complete=7", "parsed records=8")

    def test_unknown_missing_duplicate_and_missing_summary(self):
        self.assert_rejected("depth_capped=0", "depth_capped=0 surprise=1",
                             "unknown field")
        self.assert_rejected(" initialized=1", "", "missing field 'initialized'")
        self.assert_rejected(" initialized=1", " initialized=1 initialized=1",
                             "duplicate field")
        with self.assertRaisesRegex(census.CensusError, "missing census summary"):
            census.parse_census([record(0)])

    def test_shim_validation(self):
        with self.assertRaisesRegex(census.CensusError, "not contiguous"):
            census.parse_shim(["100 0", "200 2"])
        with self.assertRaisesRegex(census.CensusError, "not monotonic"):
            census.parse_shim(["200 0", "100 1"])
        with self.assertRaisesRegex(census.CensusError, "unknown prefix"):
            census.parse_shim(["timestamp_ns=100 submit_index=0"])
        with self.assertRaisesRegex(census.CensusError, "unknown prefix"):
            census.parse_shim(["NOT_A_SHIM timestamp_ns=100 submit_index=0"])

        # A killed replay may leave one numeric token at physical EOF. Only
        # that narrow non-newline tail is ignored; complete or interior damage
        # remains an error, and complete indices are still contiguous.
        rows = census.parse_shim(["100 0\n", "200 1\n", "1527"])
        self.assertEqual([row.submit_index for row in rows], [0, 1])
        with self.assertRaisesRegex(census.CensusError, "expected"):
            census.parse_shim(["100 0\n", "1527\n"])
        with self.assertRaisesRegex(census.CensusError, "expected"):
            census.parse_shim(["100 0\n", "1527", "200 1\n"])


if __name__ == "__main__":
    unittest.main()
