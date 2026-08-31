#!/usr/bin/env python3
import importlib.util, tempfile, unittest
from pathlib import Path
P=Path(__file__).with_name("cp_geometry_b0_analyze.py")
s=importlib.util.spec_from_file_location("b0",P); b0=importlib.util.module_from_spec(s); s.loader.exec_module(b0)

def rec(i,submit,unit,sha="00"*32,salt="47454f4d42305331",status=1,
        sig="0000000000000001",retry=None,sha2=None,pass_number=0):
 extra=("" if retry is None else f" retry={retry}")
 extra+=("" if sha2 is None else f" sha256_2={sha2}")
 return (f"CUDAVK_GEOM_B0_RECORD index={i} status={status} salt={salt} submit={submit} occurrence={submit} unit={unit} pass={pass_number} path=0 predecessor=1 w=2 h=3 samples=1 bytes=48 signature={sig} stage2_blocks=1 triangles=2 nontrivial=0 huge=0 setup=0 queue_status={0 if status==1 else 1} sha256={sha}{extra}")
def summary(n,admitted=None,reasons=0,mode=1):
 if admitted is None: admitted=n
 salt={1:"47454f4d42305331",2:"47454f4d42305332",3:"47454f4d42305331"}[mode]
 extra=(" salt2=47454f4d42305332 dual_domain=yes second_roots_bytes=4194304"
        if mode==3 else "")
 return f"CUDAVK_GEOM_B0 version=1 mode={mode} salt={salt}{extra} result_valid=yes capacity=131072 reserved={n} complete={n} candidates={admitted+reasons} admitted={admitted} reasons={reasons}"
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
 def test_mode3_pinned_domains_agree(self):
  a,b,x,y=("11"*32,"22"*32,"33"*32,"44"*32)
  lines=[rec(0,10,1,a,retry=0,sha2=x),rec(1,10,2,b,retry=1,sha2=y),
         rec(2,12,1,b,retry=1,sha2=y),rec(3,12,2,a,retry=0,sha2=x),
         summary(4,mode=3)]
  rs,s=b0.parse_run(self.write(lines))
  out=b0.analyze(rs,10,2,mode=s["mode"])
  self.assertEqual((out["membership_matches"],out["chronological_matches"]),(2,0))
  self.assertTrue(out["paired_domain_checked"])
  self.assertTrue(out["paired_domain_equal"])
  self.assertEqual(out["paired_domain"]["membership_equal"],True)
  self.assertEqual(out["paired_domain"]["chronological_equal"],True)
 def test_mode3_domain_disagreement_rejected(self):
  a,x,y="11"*32,"33"*32,"44"*32
  rs,s=b0.parse_run(self.write([
      rec(0,10,1,a,retry=0,sha2=x),
      rec(1,12,1,a,retry=1,sha2=y),summary(2,mode=3)]))
  with self.assertRaisesRegex(b0.CensusError,"paired-domain membership"):
   b0.analyze(rs,10,2,mode=s["mode"])
  b="22"*32
  rs,s=b0.parse_run(self.write([
      rec(0,10,1,a,retry=0,sha2=x),rec(1,10,2,b,retry=0,sha2=y),
      rec(2,12,1,a,retry=0,sha2=y),rec(3,12,2,b,retry=0,sha2=x),
      summary(4,mode=3)]))
  with self.assertRaisesRegex(b0.CensusError,"paired-domain chronological"):
   b0.analyze(rs,10,2,mode=s["mode"])
 def test_mode3_repeat_flips_report_union(self):
  a,b,x,y="11"*32,"22"*32,"33"*32,"44"*32
  paths=[self.write([rec(0,10,1,a,retry=0,sha2=x),
                     rec(1,12,1,a,retry=1,sha2=x),summary(2,mode=3)]),
         self.write([rec(0,10,1,a,retry=0,sha2=x),
                     rec(1,12,1,b,retry=1,sha2=y),summary(2,mode=3)])]
  runs=[]
  for path in paths:
   rs,s=b0.parse_run(path)
   out=b0.analyze(rs,10,2,mode=s["mode"])
   out.update(salt=f"{s['salt_value']:016x}",salt2=f"{s['salt2_value']:016x}")
   runs.append(out)
  repeat=b0.validate_salt_reruns(runs)
  absolute=(12,12,1,0)
  self.assertEqual(repeat["absolute_flip_set"],[absolute])
  self.assertEqual(repeat["absolute_flip_count"],1)
  self.assertFalse(repeat["classification_equal"])
  self.assertEqual(repeat["union_recurrent_classification"],[absolute+(True,)])
  bad_population=dict(runs[1],_mode3_population=[])
  with self.assertRaisesRegex(b0.CensusError,"structural populations differ"):
   b0.validate_salt_reruns([runs[0],bad_population])
  bad_pair=dict(runs[1],paired_domain_equal=False)
  with self.assertRaisesRegex(b0.CensusError,"paired-domain validity"):
   b0.validate_salt_reruns([runs[0],bad_pair])
 def test_mode3_summary_fields_are_explicit_and_strict(self):
  a,x="11"*32,"33"*32
  record=rec(0,1,1,a,retry=0,sha2=x)
  valid=summary(1,mode=3)
  for key in ("salt2","dual_domain","second_roots_bytes"):
   words=[word for word in valid.split() if not word.startswith(key+"=")]
   with self.subTest(missing=key):
    with self.assertRaisesRegex(b0.CensusError,f"summary missing {key}"):
     b0.parse_run(self.write([record," ".join(words)]))
  malformed=(
      valid.replace("salt2=47454f4d42305332","salt2=nothex"),
      valid.replace("dual_domain=yes","dual_domain=maybe"),
      valid.replace("second_roots_bytes=4194304","second_roots_bytes=oops"),
  )
  for bad in malformed:
   with self.subTest(summary=bad):
    with self.assertRaises(b0.CensusError):
     b0.parse_run(self.write([record,bad]))
  bad=valid.replace("second_roots_bytes=4194304","second_roots_bytes=32")
  with self.assertRaisesRegex(b0.CensusError,"second-root allocation"):
   b0.parse_run(self.write([record,bad]))
 def test_mode3_secondary_and_salt_validation(self):
  a,x="11"*32,"33"*32
  with self.assertRaisesRegex(b0.CensusError,"corrupt sha256_2"):
   b0.parse_record(rec(0,1,1,a,retry=0,sha2="gg"*32))
  with self.assertRaisesRegex(b0.CensusError,"missing sha256_2"):
   b0.parse_run(self.write([rec(0,1,1,a,retry=0),summary(1,mode=3)]))
  with self.assertRaisesRegex(b0.CensusError,"zero sha256_2"):
   b0.parse_run(self.write([rec(0,1,1,a,retry=0,sha2="00"*32),summary(1,mode=3)]))
  bad=summary(1,mode=3).replace("salt2=47454f4d42305332","salt2=47454f4d42305331")
  with self.assertRaisesRegex(b0.CensusError,"salt pair"):
   b0.parse_run(self.write([rec(0,1,1,a,retry=0,sha2=x),bad]))
  bad=summary(1,mode=3).replace("second_roots_bytes=4194304",
                                "second_roots_bytes=32")
  with self.assertRaisesRegex(b0.CensusError,"second-root allocation"):
   b0.parse_run(self.write([rec(0,1,1,a,retry=0,sha2=x),bad]))
  with self.assertRaisesRegex(b0.CensusError,"missing retry"):
   b0.parse_run(self.write([rec(0,1,1,a,sha2=x),summary(1,mode=3)]))
 def test_retry_optional_old_modes_and_strict_when_present(self):
  old=b0.parse_record(rec(0,1,1)); self.assertEqual(old["retry"],0)
  self.assertFalse(old["_retry_present"])
  retried=b0.parse_record(rec(0,1,1,retry=1)); self.assertEqual(retried["retry"],1)
  rs,_=b0.parse_run(self.write([rec(0,1,1,retry=0,sha2="00"*32),summary(1)]))
  self.assertFalse(b0.analyze(rs)["paired_domain_checked"])
  with self.assertRaisesRegex(b0.CensusError,"bad retry"):
   b0.parse_record(rec(0,1,1,retry=2))
  nonretry=dict(retried,retry=0)
  self.assertNotEqual(b0.structural_population_digest([retried]),
                      b0.structural_population_digest([nonretry]))
 def test_recurrence_crosses_passes(self):
  a="11"*32
  rs=[b0.parse_record(rec(0,10,1,a,pass_number=0)),
      b0.parse_record(rec(1,12,1,a,pass_number=3,retry=1))]
  out=b0.analyze(rs,10,2)
  self.assertEqual(out["membership_classification"],[(12,12,1,3,True)])
  self.assertEqual(out["chronological_classification"],[(12,12,1,3,True)])
 def test_structural_digest_preserves_absolute_order(self):
  x=[b0.parse_record(rec(0,1,1,sig="1".zfill(16))),b0.parse_record(rec(1,1,2,sig="2".zfill(16)))]
  self.assertNotEqual(b0.structural_population_digest(x),b0.structural_population_digest(x[::-1]))
if __name__=="__main__": unittest.main()
