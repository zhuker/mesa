#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("cp_shader_census_analyze.py")
SPEC = importlib.util.spec_from_file_location("shader_census", PATH)
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

H0 = "01" * 32
H1 = "fe" * 32

def record(index=0, submit=1, batch=1, launch=1, hash_=H0, total=257,
           exec_="VS_FETCH", fused=1, stream="main", bank=0):
    return (f"CUDAVK_SHADER_CENSUS_RECORD version=1 index={index} status=complete "
            f"stage=VS stage_hash={hash_} submit={submit} batch={batch} launch={launch} "
            f"exec={exec_} fused={fused} fetch_runs=0 total_vertices={total} "
            f"grid={(total+255)//256}x1x1 block=256x1x1 draws=1 stream={stream} "
            f"bank={bank} regs=64 spill_bytes=0 blocks_per_sm=2 reg_cap=0 sm=10.0")

def log(records):
    n = len(records)
    return "\n".join(records + [
        f"CUDAVK_SHADER_CENSUS_SUMMARY version=1 capacity=262144 population={n} "
        f"stored={n} dropped=0 missing_key=0 key_mismatch=0 unknown_stream=0 final_submit=2 "
        "context_sync_ok=yes allocation_ok=yes complete_population=yes result_valid=yes "
        "diagnostic_timing_invalid=1 cost_join_required=yes production_admitted=no"
    ])

class AnalyzeTest(unittest.TestCase):
    def test_complete_two_hash_population(self):
        text = log([record(), record(1, 2, 1, 2, H1, 256, "CLASSIC", 0, "seg7", 8)])
        got = MOD.analyze(text)
        self.assertEqual(got["population"], 2)
        self.assertEqual(got["hashes"], 2)
        MOD.compare([got, MOD.analyze(text)])

    def test_rejects_short_hash(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(hash_=H0[:16])]))

    def test_rejects_uppercase_hash(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(hash_=H1.upper())]))

    def test_rejects_duplicate_identity(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(), record(1, 1, 1, 2)]))

    def test_rejects_nonmonotonic_identity(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(submit=2), record(1, 1, 1, 2)]))

    def test_rejects_nonabsolute_launch(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(launch=2)]))

    def test_rejects_grid_mismatch(self):
        bad = log([record()]).replace("grid=2x1x1", "grid=3x1x1")
        with self.assertRaises(MOD.CensusError): MOD.analyze(bad)

    def test_rejects_exec_fused_mismatch(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(exec_="CLASSIC", fused=1)]))

    def test_rejects_unknown_bank(self):
        with self.assertRaises(MOD.CensusError): MOD.analyze(log([record(stream="unknown", bank=255)]))

    def test_rejects_dropped_population(self):
        bad = log([record()]).replace("stored=1 dropped=0", "stored=1 dropped=1")
        with self.assertRaises(MOD.CensusError): MOD.analyze(bad)

    def test_rejects_failed_context_sync(self):
        bad = log([record()]).replace("context_sync_ok=yes", "context_sync_ok=no")
        with self.assertRaises(MOD.CensusError): MOD.analyze(bad)

    def test_rejects_nonproduction_marker_change(self):
        bad = log([record()]).replace("production_admitted=no", "production_admitted=yes")
        with self.assertRaises(MOD.CensusError): MOD.analyze(bad)

    def test_rejects_final_submit_before_record(self):
        bad = log([record(submit=3)]).replace("final_submit=2", "final_submit=1")
        with self.assertRaises(MOD.CensusError): MOD.analyze(bad)

    def test_rejects_rerun_difference(self):
        one = MOD.analyze(log([record()]))
        two = MOD.analyze(log([record(hash_=H1)]))
        with self.assertRaises(MOD.CensusError): MOD.compare([one, two])

if __name__ == "__main__": unittest.main()
