#!/usr/bin/env python3
"""Synthetic CPU tests only. No GPU, reference-image, or timing assertions."""
import copy
import json
import unittest

from cp_pixel_trace import CAPACITY, TraceError, parse_trace, predict_ties


def fixture(compare=3, stable=0):
    start = dict(kind="epoch", epoch=42, x=5, y=7, capacity=CAPACITY)
    records = []
    old = (1 << 64) - 1
    for storage, prim in ((0, 9), (1, 2)):
        packed = 10 << 32 | (storage ^ 0xffffffff)
        records.append(dict(kind="record", epoch=42, event=1, framebuffer="0x1234",
                            packed=hex(packed), old=hex(old), result=hex(packed),
                            segment=0, pass_=0, sample=0, storage_id=storage,
                            global_id=storage, input_primitive=prim, fan=0,
                            depth_bits=123, depth_key=10, compare_op=compare,
                            status=5, origin_valid=1, won_atomic=1,
                            abuf_mode=0, stable=stable, depth_test=1, blend_peel=0))
        records[-1]["pass"] = records[-1].pop("pass_")
        old = packed
    final = dict(kind="final", epoch=42, event=1, framebuffer="0x1234",
                 sample=0, packed=hex(old))
    final["pass"] = 0
    final["group"] = 1
    return [start, *records, final,
            dict(kind="reduction", epoch=42, group=1, event=1),
            dict(kind="batch", epoch=42, event=1, draws=1, triangles=10, framebuffer="1234"),
            dict(kind="slice", epoch=42, event=1, row=0, start=0, count=30, drawid=0,
                 primitive_begin=0, primitive_count=10, instances=1),
            dict(kind="end", epoch=42, count=2, dropped=0, incomplete=0)]


def parse(rows):
    return parse_trace(["unrelated stderr\n"] +
                       ["CP_PIXEL_TRACE " + json.dumps(r) + "\n" for r in rows])


class PixelTraceTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(len(parse(fixture())[0]["records"]), 2)

    def test_host_bare_hex_strings(self):
        rows = fixture()
        for r in rows:
            for key in ("framebuffer", "packed", "old", "result"):
                if key in r:
                    r[key] = f"{int(r[key], 16):016x}"
        epoch = parse(rows)[0]
        self.assertEqual(epoch["records"][0]["framebuffer"], 0x1234)
        self.assertTrue(predict_ties(epoch)[0]["final_matches_prediction"])

    def test_unordered_record_slots(self):
        rows = fixture()
        rows[1], rows[2] = rows[2], rows[1]
        self.assertEqual(len(parse(rows)), 1)

    def test_incomplete_dropped_count(self):
        for field, value in (("dropped", 1), ("incomplete", 1), ("count", 3)):
            with self.subTest(field=field):
                rows = fixture()
                rows[-1][field] = value
                with self.assertRaises(TraceError):
                    parse(rows)

    def test_missing_boundaries(self):
        for index in (0, 3, -1):
            rows = fixture()
            del rows[index]
            with self.subTest(index=index), self.assertRaises(TraceError):
                parse(rows)

    def test_overflow(self):
        rows = fixture()
        rows[0]["capacity"] = 1
        with self.assertRaisesRegex(TraceError, "overflow"):
            parse(rows)

    def test_invalid_record_fields(self):
        for key, value in (("status", 100), ("status", 0), ("origin_valid", 0),
                           ("result", "0x0"), ("won_atomic", 0),
                           ("global_id", 3), ("stable", 2), ("sample", -1),
                           ("event", True), ("depth_key", 1 << 32)):
            rows = fixture()
            rows[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(TraceError):
                parse(rows)

    def test_missing_origin_field(self):
        rows = fixture()
        del rows[1]["input_primitive"]
        with self.assertRaises(TraceError):
            parse(rows)

    def test_conflicting_origin_map(self):
        rows = fixture()
        extra = copy.deepcopy(rows[1])
        extra["input_primitive"] = 88
        rows.insert(3, extra)
        rows[-1]["count"] += 1
        with self.assertRaisesRegex(TraceError, "origin map"):
            parse(rows)

    def test_final_mismatch(self):
        rows = fixture()
        rows[3]["packed"] = "0x0"
        with self.assertRaisesRegex(TraceError, "final winner"):
            parse(rows)

    def test_unknown_kind_epoch(self):
        for field, value in (("kind", "future"), ("epoch", 99)):
            rows = fixture()
            rows[1][field] = value
            with self.assertRaises(TraceError):
                parse(rows)

    def test_duplicate_final(self):
        rows = fixture()
        rows.insert(4, copy.deepcopy(rows[3]))
        with self.assertRaises(TraceError):
            parse(rows)

    def test_malformed_json(self):
        for line in ('CP_PIXEL_TRACE {', 'CP_PIXEL_TRACE []',
                     'CP_PIXEL_TRACE {"kind":"epoch","kind":"end"}',
                     'CP_PIXEL_TRACEbad {}'):
            with self.subTest(line=line), self.assertRaises(TraceError):
                parse_trace([line])

    def test_strict_first_inclusive_last(self):
        for op in (1, 3, 4, 6):
            report = predict_ties(parse(fixture(op))[0])[0]
            expected = 2 if op in (1, 4) else 9
            self.assertEqual(report["original_origin"], (0, expected, 0))
            self.assertEqual(report["allocated_origin"], (0, 2, 0))
            self.assertTrue(report["final_matches_prediction"])
            self.assertEqual("same origin winner" in report["falsifiers"], op in (1, 4))

    def test_fan_order(self):
        rows = fixture()
        rows[1]["input_primitive"] = rows[2]["input_primitive"] = 7
        rows[1]["fan"], rows[2]["fan"] = 1, 0
        report = predict_ties(parse(rows)[0])[0]
        self.assertEqual(report["original_origin"], (0, 7, 1))
        self.assertEqual(report["allocated_origin"], (0, 7, 0))

    def test_stable_falsifier(self):
        report = predict_ties(parse(fixture(stable=1))[0])[0]
        self.assertIn("stable allocation", report["falsifiers"])

    def test_unique_depth_falsifier(self):
        rows = fixture()
        rows[1]["depth_key"] = 11
        rows[1]["packed"] = rows[1]["result"] = hex(11 << 32 | 0xffffffff)
        rows[2]["old"] = rows[1]["packed"]
        report = predict_ties(parse(rows)[0])[0]
        self.assertEqual(report["tie_count"], 1)
        self.assertIn("unique best packed depth", report["falsifiers"])

    def test_exact_key_not_float(self):
        rows = fixture()
        rows[1]["depth_bits"] = 999
        self.assertEqual(predict_ties(parse(rows)[0])[0]["tie_count"], 2)

    def test_equal_atomic_did_not_win(self):
        rows = fixture()
        rows[1]["old"] = rows[1]["packed"]
        rows[1]["won_atomic"] = 0
        self.assertEqual(len(parse(rows)), 1)

    def test_disabled_depth_or_peel_not_predicted(self):
        for field, value in (("depth_test", 0), ("blend_peel", 1)):
            rows = fixture()
            rows[1][field] = value
            if field == "blend_peel":
                for rec in rows[1:3]:
                    rec["blend_peel"] = 1
                    rec["packed"] = hex(rec["storage_id"] << 32 | (rec["storage_id"] ^ 0xffffffff))
                rows[1]["result"] = rows[1]["packed"]
                rows[2]["old"] = rows[2]["result"] = rows[1]["packed"]
                rows[2]["won_atomic"] = 0
                rows[3]["packed"] = rows[1]["packed"]
            self.assertIn("unsupported", predict_ties(parse(rows)[0])[0]["prediction"])

    def test_depth_mode_fields_required_booleans(self):
        for field in ("depth_test", "blend_peel"):
            rows = fixture()
            rows[1][field] = 2
            with self.assertRaises(TraceError):
                parse(rows)
            del rows[1][field]
            with self.assertRaises(TraceError):
                parse(rows)

    def test_packed_low_identity_is_validated(self):
        rows = fixture()
        rows[1]["storage_id"] = rows[1]["global_id"] = 8
        with self.assertRaisesRegex(TraceError, "packed candidate"):
            parse(rows)

    def test_corrupt_peel_packing_rejected(self):
        rows = fixture()
        rows[1]["blend_peel"] = 1
        with self.assertRaisesRegex(TraceError, "packed candidate"):
            parse(rows)

    def test_non_depth_key_not_predicted(self):
        rows = fixture()
        rows[1]["depth_key"] = 11
        with self.assertRaisesRegex(TraceError, "packed candidate"):
            parse(rows)

    def test_shared_reduction_earlier_event_needs_explicit_membership(self):
        rows = fixture()
        # The first event owns the winning key; only a latest-event snapshot
        # exists. No reduction lifetime/membership exists in this schema.
        rows[1]["event"] = 0
        rows[1]["depth_key"] = 9
        rows[1]["packed"] = rows[1]["result"] = hex(9 << 32 | 0xffffffff)
        rows[2]["old"] = rows[2]["result"] = rows[1]["packed"]
        rows[2]["won_atomic"] = 0
        rows[3]["packed"] = rows[1]["packed"]
        with self.assertRaises(TraceError):
            parse(rows)

    def test_shared_reduction_earlier_event_with_explicit_membership(self):
        rows = fixture()
        rows[1]["event"] = 0
        rows[1]["depth_key"] = 9
        rows[1]["packed"] = rows[1]["result"] = hex(9 << 32 | 0xffffffff)
        rows[2]["old"] = rows[2]["result"] = rows[1]["packed"]
        rows[2]["won_atomic"] = 0
        rows[3]["packed"] = rows[1]["packed"]
        rows.insert(-1, dict(kind="reduction", epoch=42, group=1, event=0))
        for metadata in rows[5:7]:
            extra = copy.deepcopy(metadata)
            extra["event"] = 0
            rows.insert(-1, extra)
        report = predict_ties(parse(rows)[0])[0]
        self.assertEqual(report["allocated_origin"], (0, 9, 0))
        self.assertTrue(report["final_matches_prediction"])

    def test_missing_reduction_membership(self):
        rows = fixture()
        del rows[4]
        with self.assertRaisesRegex(TraceError, "missing reduction"):
            parse(rows)

    def test_duplicate_reduction_membership(self):
        rows = fixture()
        rows.insert(-1, copy.deepcopy(rows[4]))
        with self.assertRaisesRegex(TraceError, "duplicate reduction"):
            parse(rows)

    def test_ambiguous_pointer_reuse_membership(self):
        rows = fixture()
        extra = copy.deepcopy(rows[3])
        extra["group"] = 2
        rows.insert(-1, extra)
        rows.insert(-1, dict(kind="reduction", epoch=42, group=2, event=1))
        with self.assertRaisesRegex(TraceError, "ambiguous reduction"):
            parse(rows)

    def test_same_framebuffer_address_does_not_join_unrelated_events(self):
        rows = fixture()
        rows[1]["event"] = 99
        with self.assertRaises(TraceError):
            parse(rows)

    def test_final_old_winner_has_no_origin(self):
        rows = fixture()
        unknown = hex(8 << 32 | 0xffffffab)
        for r in rows[1:3]:
            r["old"] = r["result"] = unknown
            r["won_atomic"] = 0
        rows[3]["packed"] = unknown
        with self.assertRaisesRegex(TraceError, "missing final winner origin"):
            parse(rows)

    def test_missing_batch_or_slice(self):
        for index in (5, 6):
            rows = fixture()
            del rows[index]
            with self.assertRaises(TraceError):
                parse(rows)

    def test_slice_out_of_range(self):
        rows = fixture()
        rows[6]["row"] = 1
        with self.assertRaisesRegex(TraceError, "row out of range"):
            parse(rows)

    def test_slice_overlap_gap_and_total(self):
        for begin in (4, 6):
            rows = fixture()
            rows[5]["draws"] = 2
            rows[6]["primitive_count"] = 5
            extra = copy.deepcopy(rows[6])
            extra.update(row=1, primitive_begin=begin)
            rows.insert(-1, extra)
            with self.assertRaisesRegex(TraceError, "overlap or gap"):
                parse(rows)
        rows = fixture()
        rows[5]["triangles"] = 11
        with self.assertRaisesRegex(TraceError, "slice total"):
            parse(rows)

    def test_primitive_outside_slice(self):
        rows = fixture()
        rows[1]["input_primitive"] = 10
        with self.assertRaisesRegex(TraceError, "original draw slice"):
            parse(rows)

    def test_instance_mapping_and_drawid_not_identity(self):
        rows = fixture()
        rows[5]["draws"] = 2
        rows[6].update(primitive_count=2, count=3, instances=2, drawid=17)
        extra = copy.deepcopy(rows[6])
        extra.update(row=1, primitive_begin=2, primitive_count=8, count=12)
        rows.insert(-1, extra)
        report = predict_ties(parse(rows)[0])[0]
        self.assertEqual(report["original_draw"],
                         dict(event=1, row=1, drawid=17, local_primitive=3, instance=1, fan=0))
        self.assertEqual(report["allocated_draw"],
                         dict(event=1, row=1, drawid=17, local_primitive=0, instance=0, fan=0))

    def test_invalid_instance_mapping(self):
        for instances in (0, 3):
            rows = fixture()
            rows[6]["instances"] = instances
            with self.assertRaisesRegex(TraceError, "instance mapping"):
                parse(rows)

    def test_multiple_epochs(self):
        rows = fixture()
        other = fixture()
        for r in other:
            r["epoch"] = 43
        self.assertEqual(len(parse(rows + other)), 2)
        with self.assertRaises(TraceError):
            parse(rows + rows)


if __name__ == "__main__":
    unittest.main()
