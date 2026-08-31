#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, unittest
from pathlib import Path
P=Path(__file__).with_name("cp_vertex_reuse_analyze.py")
spec=importlib.util.spec_from_file_location("an",P); an=importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

def report(records, *, record_overflow=0, valid="yes"):
    normalized=[]
    for i,r in enumerate(records):
        normalized.append({"fused":i&1,"fetch_runs":0 if i&1 else 1,
                           "indexed":1,"total":0,"unique":0,"duplicate":0}|r)
    records=normalized
    lines=[]
    for i,r in enumerate(records):
        x={"status":"complete","submit":1,"batch":i+1,"bank":0,"generation":i+1,
           "fused":i&1,"fetch_runs":0 if i&1 else 1,"indexed":1,"writes_memory":0,
           "semantic_metadata_complete":0,"total":0,"unique":0,"duplicate":0,
           "table_capacity":16,"probes":0,"overflow":0,"bad_lanes":0}|r
        fields={"version":1,"index":i}|x
        lines.append(an.RP+" ".join(f"{k}={v}" for k,v in fields.items()))
    total=sum(r.get("total",0) for r in records); dup=sum(r.get("duplicate",0) for r in records)
    uniq=sum(r.get("unique",r.get("total",0)-r.get("duplicate",0)) for r in records)
    admitted=sum(r.get("status","complete")=="complete" for r in records); excluded=len(records)-admitted
    def sums(key,val): return sum(r.get("total",0) if r.get(key,1)==val else 0 for r in records)
    def dups(key,val): return sum(r.get("duplicate",0) if r.get(key,1)==val else 0 for r in records)
    screen="yes" if total and 200*dup>=129*total else "no"
    s={"version":1,"capacity":262144,"population":len(records)+record_overflow,
       "reserved":len(records),"complete":admitted,"admitted":admitted,"excluded":excluded,
       "record_overflow":record_overflow,"table_overflow":sum(bool(r.get("overflow",0)) for r in records),
       "final_submit":1,"total":total,"unique":uniq,"duplicate":dup,
       "indexed_total":sums("indexed",1),"indexed_duplicate":dups("indexed",1),
       "nonindexed_total":sums("indexed",0),"nonindexed_duplicate":dups("indexed",0),
       "fused_total":sums("fused",1),"fused_duplicate":dups("fused",1),
       "standalone_total":sums("fused",0),"standalone_duplicate":dups("fused",0),
       "population_ok":"yes","classification_ok":"yes","population_screen_64_5":screen,
       "cost_join_required":"yes","result_valid":valid,"diagnostic_timing_invalid":1}
    lines.append(an.SP+" ".join(f"{k}={v}" for k,v in s.items()))
    return "\n".join(lines)

class TestAnalyzer(unittest.TestCase):
    def test_exact_boundary_and_weighting(self):
        text=report([{"total":200,"unique":71,"duplicate":129,"fused":1,"indexed":1},
                     {"total":1,"unique":1,"duplicate":0,"fused":0,"indexed":0}])
        out=an.analyze(text)
        self.assertFalse(out["population_screen_64_5"])
        self.assertAlmostEqual(out["groups"]["fused"]["fraction"],129/200)
        self.assertEqual(out["groups"]["standalone"]["fraction"],0)
        exact=an.analyze(report([{"total":200,"unique":71,"duplicate":129}]))
        self.assertTrue(exact["population_screen_64_5"])
        self.assertFalse(exact["production_admitted"])
        self.assertTrue(exact["cost_join_required"])
    def test_below_boundary(self):
        out=an.analyze(report([{"total":200,"unique":72,"duplicate":128}]))
        self.assertFalse(out["population_screen_64_5"])
    def test_excluded_is_conservative(self):
        text=report([{"status":"excluded_alloc","total":9,"unique":9,"duplicate":0}])
        out=an.analyze(text)
        self.assertEqual(out["excluded_batches"],1)
        self.assertEqual(out["groups"]["overall"]["duplicate"],0)
    def test_empty(self):
        out=an.analyze(report([]))
        self.assertFalse(out["population_screen_64_5"])
        self.assertIsNone(out["groups"]["overall"]["fraction"])
    def test_duplicate_identity_rejected(self):
        text=report([{"batch":1},{"batch":1}])
        with self.assertRaisesRegex(ValueError,"duplicate identity"):
            an.analyze(text)
    def test_bad_reconciliation_rejected(self):
        text=report([{"total":3,"unique":2,"duplicate":2}])
        with self.assertRaisesRegex(ValueError,"does not reconcile"):
            an.analyze(text)
    def test_excluded_duplicate_rejected(self):
        text=report([{"status":"excluded_table","total":3,"unique":2,"duplicate":1,"overflow":1}])
        with self.assertRaisesRegex(ValueError,"conservatively charged"):
            an.analyze(text)
    def test_invalid_summary_rejected(self):
        text=report([{"total":1,"unique":1}],record_overflow=1,valid="no")
        with self.assertRaisesRegex(ValueError,"result_valid"):
            an.analyze(text)

    def test_summary_version_rejected(self):
        text=report([]).replace("SUMMARY version=1", "SUMMARY version=2")
        with self.assertRaisesRegex(ValueError,"summary version"):
            an.analyze(text)
    def test_table_overflow_mismatch_rejected(self):
        text=report([]).replace("table_overflow=0", "table_overflow=1")
        with self.assertRaisesRegex(ValueError,"table_overflow"):
            an.analyze(text)
    def test_population_ok_mismatch_rejected(self):
        text=report([]).replace("population_ok=yes", "population_ok=no")
        with self.assertRaisesRegex(ValueError,"population_ok"):
            an.analyze(text)
    def test_classification_ok_mismatch_rejected(self):
        text=report([]).replace("classification_ok=yes", "classification_ok=no")
        with self.assertRaisesRegex(ValueError,"classification_ok"):
            an.analyze(text)
    def test_result_valid_mismatch_rejected(self):
        text=report([]).replace("result_valid=yes", "result_valid=no")
        with self.assertRaisesRegex(ValueError,"result_valid mismatch"):
            an.analyze(text, require_valid=False)
    def test_nonmonotonic_identity_rejected(self):
        text=report([{"batch":2},{"batch":1}])
        with self.assertRaisesRegex(ValueError,"not increasing"):
            an.analyze(text)

    def test_capacity_rejected(self):
        text=report([{}]).replace("capacity=262144", "capacity=0")
        with self.assertRaisesRegex(ValueError,"exceeds capacity"):
            an.analyze(text)

if __name__=="__main__": unittest.main()
