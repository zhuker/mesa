#!/usr/bin/env python3
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).with_name("cp_geometry_b0_cost.py")
spec = importlib.util.spec_from_file_location("b0_cost", P)
cost = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = cost
spec.loader.exec_module(cost)


def launch(name, ds, de, rs, correlation, grid=1):
    return cost.Launch(name, cost.Interval(ds, de), cost.Interval(rs, rs + 1),
                       correlation, grid)


def mode3_record(index, submit, unit, digest, digest2):
    return (
        f"CUDAVK_GEOM_B0_RECORD index={index} status=1 "
        f"salt=47454f4d42305331 submit={submit} occurrence={submit} "
        f"unit={unit} pass=0 retry=0 path=0 predecessor=1 w=2 h=3 "
        f"samples=1 bytes=48 signature=0000000000000001 stage2_blocks={unit} "
        f"triangles=2 nontrivial=0 huge=0 setup=0 queue_status=0 "
        f"sha256={digest} sha256_2={digest2}"
    )


def mode3_summary(count, capacity=None):
    if capacity is None:
        capacity = count
    return (
        "CUDAVK_GEOM_B0 version=1 mode=3 salt=47454f4d42305331 "
        "salt2=47454f4d42305332 dual_domain=yes "
        f"second_roots_bytes={capacity * 32} result_valid=yes "
        f"capacity={capacity} reserved={count} complete={count} "
        f"candidates={count} admitted={count} reasons=0"
    )


def mode4_record(index, submit, unit, digest, digest2, isolated=1):
    return mode3_record(index, submit, unit, digest, digest2).replace(
        "retry=0", f"retry=0 isolated={isolated}")


def mode4_summary(count, capacity=None):
    if capacity is None:
        capacity = count
    return mode3_summary(count, capacity).replace("mode=3", "mode=4") + (
        " private_banks=9 private_visbuf_bytes=900"
        " private_scratch_bytes=1800 private_saved_roots_bytes=288"
    )


class TestGeometryB0Cost(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", delete=False)
        f.write(text)
        f.close()
        return Path(f.name)

    def test_submit_timestamp_torn_final_and_diagnostics(self):
        path = self.write("100 0\n200 1\n20")
        timestamps, fragment = cost.parse_submit_timestamps(path)
        self.assertEqual(timestamps, [100, 200])
        self.assertEqual(fragment, "20")

        path = self.write("100 0\nbroken\n200 1\n")
        with self.assertRaisesRegex(cost.CostError, r":2: malformed.*'broken'"):
            cost.parse_submit_timestamps(path)
        path = self.write("100 0\n200 2\n")
        with self.assertRaisesRegex(cost.CostError,
                                    r":2: submit index mismatch: expected 1, got 2"):
            cost.parse_submit_timestamps(path)
        path = self.write("100 0\n100 1\n")
        with self.assertRaisesRegex(cost.CostError,
                                    r"previous=100 current=100"):
            cost.parse_submit_timestamps(path)

    def test_filtered_launch_patterns_and_ordinals(self):
        n = cost.NAMES
        launches = [
            launch(n["clip"], 1, 2, 110, 1),
            launch(n["fused"], 2, 3, 111, 2),
            launch(n["stage2"], 3, 4, 112, 3, 9),
            launch(n["stage1"], 7, 8, 210, 7),
            launch(n["stage2"], 8, 9, 211, 8, 12),
        ]
        units = cost.direct_units(launches, [100, 200, 300])
        self.assertEqual([(u.submit, u.ordinal) for u in units], [(1, 1), (2, 1)])
        self.assertIs(units[0].clip, launches[0])
        self.assertIsNone(units[1].clip)
        self.assertEqual(units[1].stage2.grid_x, 12)

        broken = [launch(n["stage1"], 1, 2, 110, 10),
                  launch(n["clip"], 2, 3, 111, 11),
                  launch(n["stage2"], 3, 4, 112, 12)]
        with self.assertRaisesRegex(
                cost.CostError,
                r"predecessor cp_rasterize_stage1 correlationId 10 is followed "
                r"by cp_clip_triangles"):
            cost.direct_units(broken, [100, 200])

    def test_sweep_overlaps_and_simultaneous_boundaries(self):
        target = [cost.Interval(1, 5), cost.Interval(3, 7)]
        # all_activity includes both target intervals and one other operation.
        activity = target + [cost.Interval(4, 6), cost.Interval(7, 9)]
        self.assertEqual(cost.sweep_cost(target, activity), {
            "sum_ns": 8, "union_ns": 6, "union_exclusive_ns": 4,
        })
        # Half-open simultaneous boundaries do not overlap.
        target = [cost.Interval(1, 2), cost.Interval(2, 3)]
        self.assertEqual(cost.sweep_cost(target, target)["union_ns"], 2)
        self.assertEqual(cost.clipped(cost.Interval(1, 4), 2, 3), cost.Interval(2, 3))
        self.assertIsNone(cost.clipped(cost.Interval(1, 2), 2, 3))

    def test_mapping_invariants_and_exact_mismatch(self):
        n = cost.NAMES
        pred = launch(n["fused"], 10, 11, 110, 1)
        stage2 = launch(n["stage2"], 11, 12, 111, 2, 17)
        unit = cost.Unit(5, 1, pred, stage2, None)
        record = {"status": 1, "submit": 5, "unit": 1, "index": 7,
                  "predecessor": 2, "stage2_blocks": 17,
                  "occurrence": 5, "pass": 0}
        mapped = cost.map_admitted_records([record], [unit], 5, 6)
        self.assertIs(mapped[(5, 1)], record)

        wrong = dict(record, stage2_blocks=18)
        with self.assertRaisesRegex(
                cost.CostError,
                r"B0/trace mismatch index=7 submit=5 unit=1: stage2_blocks record=18 trace=17"):
            cost.map_admitted_records([wrong], [unit], 5, 6)
        missing = dict(record, unit=2)
        with self.assertRaisesRegex(
                cost.CostError,
                r"index=7 has no trace unit: submit=5 unit=2 trace_submit_range=\[5,6\)"):
            cost.map_admitted_records([missing], [unit], 5, 6)
        wrong_class = {"membership_classification": [(5, 99, 1, 0, True)]}
        with self.assertRaisesRegex(cost.CostError,
                                    r"occurrence classification=99 record=5"):
            cost.classification_map(wrong_class, mapped)


    def test_recurrence_policy_and_clips_are_conservative(self):
        n = cost.NAMES
        units = []
        activity = []
        correlation = 1
        for ordinal, base in enumerate((10, 20, 30, 40), 1):
            pred = launch(n["stage1"], base, base + 1, 100 + base,
                          correlation, 1)
            stage2 = launch(n["stage2"], base + 1, base + 2,
                            101 + base, correlation + 1, ordinal)
            correlation += 2
            units.append(cost.Unit(5, ordinal, pred, stage2, None))
            activity.extend((pred.device, stage2.device))
        clip = cost.Interval(1, 2)
        activity.append(clip)
        records = [
            {"status": 1, "submit": 5, "unit": ordinal, "index": ordinal,
             "predecessor": 1, "stage2_blocks": ordinal,
             "occurrence": 5, "pass": 0}
            for ordinal in (2, 3, 4)
        ]

        class FakeAnalyzer:
            @staticmethod
            def parse_run(_path):
                return records, {"mode": 1, "salt_value": 0x12,
                                 "salt2_value": 0}

            @staticmethod
            def analyze(_records, _origin, _submits_per_frame, mode=None):
                self.assertEqual(mode, 1)
                # Unit 2 is missing, unit 3 is unmatched, unit 4 is matched.
                return {"membership_classification": [
                    (5, 5, 3, 0, False), (5, 5, 4, 0, True),
                ]}

        result = cost.analyze_arm(Path("synthetic"), FakeAnalyzer, units,
                                  [clip], activity, 0, 100, 5, 6, 2784, 1)
        self.assertEqual(result["excluded_or_unadmitted_conservative_recurrent"], 1)
        self.assertEqual(result["classification_missing_conservative_recurrent"], 1)
        self.assertEqual(result["classification_unmatched_nonrecurrent"], 1)
        self.assertEqual(result["classification_matched_recurrent"], 1)
        # Clip plus three recurrent predecessor/stage2 pairs form the upper.
        self.assertEqual(result["possible_saving_necessary_condition_upper"]
                         ["sum_ms_per_frame"], 7 / 1_000_000)
        self.assertEqual(result["classified_nonrecurrent_diagnostic_trace_cost"]
                         ["sum_ms_per_frame"], 2 / 1_000_000)

    def test_empty_mode3_is_analyzed_with_explicit_summary_mode(self):
        analyzer = cost._load_analyzer()
        path = self.write(mode3_summary(0, capacity=0))
        prepared = cost.prepare_log(path, analyzer, 10)
        self.assertEqual(prepared["analysis"]["mode"], 3)
        self.assertTrue(prepared["analysis"]["paired_domain_checked"])
        self.assertEqual(prepared["analysis"]["membership_classification"], [])

    def test_repeated_mode3_complementary_flips_use_union_arm(self):
        analyzer = cost._load_analyzer()
        a, b, c = "11" * 32, "22" * 32, "33" * 32
        x, y, z = "44" * 32, "55" * 32, "66" * 32
        common = [
            mode3_record(0, 10, 1, a, x),
            mode3_record(1, 10, 2, b, y),
        ]
        paths = [
            self.write("\n".join(common + [
                mode3_record(2, 12, 1, a, x),
                mode3_record(3, 12, 2, c, z), mode3_summary(4),
            ])),
            self.write("\n".join(common + [
                mode3_record(2, 12, 1, c, z),
                mode3_record(3, 12, 2, b, y), mode3_summary(4),
            ])),
        ]
        prepared = [cost.prepare_log(path, analyzer, 10) for path in paths]

        n = cost.NAMES
        units = []
        activity = []
        for unit, base in ((1, 10), (2, 20)):
            pred = launch(n["stage1"], base, base + 1, 100 + base,
                          unit * 2 - 1)
            stage2 = launch(n["stage2"], base + 1, base + 2,
                            101 + base, unit * 2, unit)
            units.append(cost.Unit(12, unit, pred, stage2, None))
            activity.extend((pred.device, stage2.device))
        clip = cost.Interval(1, 2)
        activity.append(clip)

        arms, rerun = cost.aggregate_arms(
            prepared, analyzer, units, [clip], activity,
            0, 100, 12, 13, 1)
        self.assertTrue(rerun["paired_mode3_repeats"])
        self.assertEqual([arm["arm"] for arm in arms], [
            "per_run_diagnostic", "per_run_diagnostic",
            "conservative_repeat_union",
        ])
        self.assertEqual([arm["decision_bearing"] for arm in arms],
                         [False, False, True])
        self.assertEqual(
            [(arm["classification_matched_recurrent"],
              arm["classification_unmatched_nonrecurrent"])
             for arm in arms], [(1, 1), (1, 1), (2, 0)])
        union = arms[-1]
        self.assertEqual(union["source_logs"], [str(path) for path in paths])
        self.assertEqual(union["reference_log"], str(paths[0]))
        self.assertEqual(union["all_clip_intervals_conservative_recurrent"], 1)
        self.assertEqual(
            union["possible_saving_necessary_condition_upper"]["sum_ms_per_frame"],
            5 / 1_000_000)

    def test_repeated_mode4_complementary_flips_use_union_arm(self):
        analyzer = cost._load_analyzer()
        a, b, c = "11" * 32, "22" * 32, "33" * 32
        x, y, z = "44" * 32, "55" * 32, "66" * 32
        common = [
            mode4_record(0, 10, 1, a, x),
            mode4_record(1, 10, 2, b, y),
        ]
        paths = [
            self.write("\n".join(common + [
                mode4_record(2, 12, 1, a, x),
                mode4_record(3, 12, 2, c, z), mode4_summary(4),
            ])),
            self.write("\n".join(common + [
                mode4_record(2, 12, 1, c, z),
                mode4_record(3, 12, 2, b, y), mode4_summary(4),
            ])),
        ]
        prepared = [cost.prepare_log(path, analyzer, 10) for path in paths]

        n = cost.NAMES
        units = []
        activity = []
        for unit, base in ((1, 10), (2, 20)):
            pred = launch(n["stage1"], base, base + 1, 100 + base,
                          unit * 2 - 1)
            stage2 = launch(n["stage2"], base + 1, base + 2,
                            101 + base, unit * 2, unit)
            units.append(cost.Unit(12, unit, pred, stage2, None))
            activity.extend((pred.device, stage2.device))
        clip = cost.Interval(1, 2)
        activity.append(clip)

        arms, rerun = cost.aggregate_arms(
            prepared, analyzer, units, [clip], activity,
            0, 100, 12, 13, 1)
        self.assertTrue(rerun["paired_mode4_repeats"])
        self.assertEqual([arm["arm"] for arm in arms], [
            "per_run_diagnostic", "per_run_diagnostic",
            "conservative_repeat_union",
        ])
        self.assertEqual([arm["decision_bearing"] for arm in arms],
                         [False, False, True])
        self.assertEqual(
            [(arm["classification_matched_recurrent"],
              arm["classification_unmatched_nonrecurrent"])
             for arm in arms], [(1, 1), (1, 1), (2, 0)])
        union = arms[-1]
        self.assertEqual(union["source_logs"], [str(path) for path in paths])
        self.assertEqual(union["reference_log"], str(paths[0]))
        self.assertEqual(union["all_clip_intervals_conservative_recurrent"], 1)
        self.assertEqual(
            union["possible_saving_necessary_condition_upper"]["sum_ms_per_frame"],
            5 / 1_000_000)


    def test_empty_mode4_is_analyzed_with_explicit_summary_mode(self):
        analyzer = cost._load_analyzer()
        path = self.write(mode4_summary(0, capacity=0))
        prepared = cost.prepare_log(path, analyzer, 10)
        self.assertEqual(prepared["analysis"]["mode"], 4)
        self.assertTrue(prepared["analysis"]["paired_domain_checked"])
        self.assertEqual(prepared["analysis"]["membership_classification"], [])

    def test_synthetic_sqlite_correlation_join(self):
        db = sqlite3.connect(":memory:")
        db.executescript("""
          CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER, end INTEGER, correlationId INTEGER,
            shortName INTEGER, gridX INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER, end INTEGER, correlationId INTEGER);
        """)
        names = [cost.NAMES["stage1"], cost.NAMES["stage2"]]
        db.executemany("INSERT INTO StringIds VALUES (?,?)", enumerate(names, 1))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (20,30,41,1,3)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,20,42,2,7)")
        # Device order differs from host order; loader must sort runtime starts.
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (200,201,41)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (201,202,42)")
        launches = cost.load_target_launches(db)
        self.assertEqual([x.name for x in launches], names)
        units = cost.direct_units(launches, [100, 300])
        self.assertEqual((units[0].submit, units[0].ordinal,
                          units[0].stage2.grid_x), (1, 1, 7))

        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=42")
        with self.assertRaisesRegex(cost.CostError,
                                    r"correlationId 42 has no runtime launch"):
            cost.load_target_launches(db)


if __name__ == "__main__":
    unittest.main()
