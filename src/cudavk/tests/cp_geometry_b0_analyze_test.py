#!/usr/bin/env python3
import importlib.util, tempfile, unittest
from pathlib import Path
P=Path(__file__).with_name("cp_geometry_b0_analyze.py")
s=importlib.util.spec_from_file_location("b0",P); b0=importlib.util.module_from_spec(s); s.loader.exec_module(b0)

def rec(i,submit,unit,sha="00"*32,salt="47454f4d42305331",status=1,sig="0000000000000001"):
 return (f"CUDAVK_GEOM_B0_RECORD index={i} status={status} salt={salt} submit={submit} occurrence={submit} unit={unit} pass=0 path=0 predecessor=1 w=2 h=3 samples=1 bytes=48 signature={sig} stage2_blocks=1 triangles=2 nontrivial=0 huge=0 setup=0 queue_status={0 if status==1 else 1} sha256={sha}")
def summary(n,admitted=None,reasons=0):
 if admitted is None: admitted=n
 return f"CUDAVK_GEOM_B0 version=1 mode=1 salt=47454f4d42305331 result_valid=yes capacity=131072 reserved={n} complete={n} candidates={admitted+reasons} admitted={admitted} reasons={reasons}"
class TestB0(unittest.TestCase):
 def write(self,lines):
  f=tempfile.NamedTemporaryFile("w",delete=False); f.write("\n".join(lines)); f.close(); return Path(f.name)
 def test_frame_helpers_and_matches(self):
  a="11"*32; z="22"*32
  rs=[b0.parse_record(rec(0,10,1,a)),b0.parse_record(rec(1,10,2,z)),
      b0.parse_record(rec(2,12,1,a)),b0.parse_record(rec(3,12,2,a))]
  self.assertEqual(b0.frame_number(12,10,2),1)
  x=b0.analyze(rs,10,2); self.assertEqual(x["membership_matches"],2); self.assertEqual(x["chronological_matches"],1)
 def test_missing_duplicate_corrupt(self):
  with self.assertRaises(b0.CensusError): b0.parse_run(self.write([rec(0,1,1),summary(2)]))
  with self.assertRaises(b0.CensusError): b0.parse_run(self.write([rec(0,1,1),rec(0,1,2),summary(2)]))
  with self.assertRaises(b0.CensusError): b0.parse_record(rec(0,1,1,sha="xx"*32))
  with self.assertRaises(b0.CensusError): b0.parse_run(self.write([rec(1,1,1),summary(1)]))
  with self.assertRaises(b0.CensusError): b0.parse_record(rec(-1,1,1))
  with self.assertRaises(b0.CensusError): b0.parse_record(rec(0,1,1,sig="-000000000000001"))
  bad=summary(1).replace("capacity=131072","capacity=0")
  with self.assertRaises(b0.CensusError): b0.parse_run(self.write([rec(0,1,1),bad]))
 def test_classification_and_queue_reject(self):
  p=self.write([rec(0,1,1),rec(1,1,2,status=2),summary(2,1,1)])
  rs,s=b0.parse_run(p); self.assertEqual(s["admitted"],1); self.assertEqual(len(b0.group_frames(rs)[0]),1)
 def test_salt_rerun_validation(self):
  pop="ab"*32
  a={"salt":"01","structural_population_sha256":pop,"membership_classification":[(1,1,1,0,True)]}
  c={"salt":"02","structural_population_sha256":pop,"membership_classification":[(1,1,1,0,True)]}
  self.assertTrue(b0.validate_salt_reruns([a,c])["checked"])
  with self.assertRaises(b0.CensusError): b0.validate_salt_reruns([a,dict(c,salt="01")])
  with self.assertRaises(b0.CensusError): b0.validate_salt_reruns([a,dict(c,membership_classification=[(1,1,1,0,False)])])
 def test_empty_salt_reruns_use_summary(self):
  p=self.write([summary(0)])
  q=self.write([summary(0).replace("mode=1 salt=47454f4d42305331","mode=2 salt=47454f4d42305332")])
  runs=[]
  for path in (p,q):
   rs,s=b0.parse_run(path); x=b0.analyze(rs); x["salt"]=f"{s['salt_value']:016x}"; runs.append(x)
  self.assertTrue(b0.validate_salt_reruns(runs)["classification_equal"])
 def test_structural_digest_preserves_absolute_order(self):
  x=[b0.parse_record(rec(0,1,1,sig="1".zfill(16))),b0.parse_record(rec(1,1,2,sig="2".zfill(16)))]
  self.assertNotEqual(b0.structural_population_digest(x),b0.structural_population_digest(x[::-1]))
if __name__=="__main__": unittest.main()
