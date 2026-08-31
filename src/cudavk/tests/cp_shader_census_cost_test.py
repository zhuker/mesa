#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path

PATH=Path(__file__).with_name("cp_shader_census_cost.py")
SPEC=importlib.util.spec_from_file_location("shader_cost",PATH)
MOD=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=MOD; SPEC.loader.exec_module(MOD)
I=MOD.Interval; L=MOD.Launch

def launch(name,ds,de,rs,stream=13,gx=1,bx=256,cid=1):
    return L(name,I(ds,de),I(rs,rs+1),cid,gx,bx,stream)

def record(batch=1,hash_="01"*32,stream="main",grid="1x1x1"):
    return {"submit":"1","batch":str(batch),"exec":"VS_FETCH","stage_hash":hash_,
            "stream":stream,"grid":grid,"block":"256x1x1","fetch_runs":"0"}

class CostTest(unittest.TestCase):
    def test_union_exclusive_removes_other_activity(self):
        selected=[I(0,10),I(5,15)]
        self.assertEqual(MOD.union_and_exclusive(selected, selected+[I(12,20)]),(15,12))

    def test_union_exclusive_fully_hidden(self):
        selected=[I(10,20)]
        self.assertEqual(MOD.union_and_exclusive(selected,selected+[I(0,30)]),(10,0))

    def test_classify_absolute_batches_and_fetch(self):
        rows=[launch("main",10,20,110,cid=1),launch("cp_clip_triangles",21,30,111,cid=2),
              launch("cp_vertex_fetch",31,35,120,cid=3),launch("main",36,40,121,cid=4),
              launch("cp_clip_rast_fused",41,50,122,cid=5)]
        got=MOD.classify(rows,[100,200],1)
        self.assertEqual([(x[0],x[1],x[3] is None) for x in got],[(1,1,True),(1,2,False)])

    def test_map_requires_full_clean_population(self):
        vs=launch("main",10,20,110,gx=1,cid=1)
        got=MOD.map_fused([record()],[(1,1,vs,None)],1,2,{"main":13})
        self.assertEqual(len(got),1)
        with self.assertRaises(MOD.CostError):
            MOD.map_fused([record(batch=2)],[(1,1,vs,None)],1,2,{"main":13})

    def test_classic_without_fetch_is_complete_but_not_costed(self):
        vs1=launch("main",10,20,110,gx=1,cid=1)
        vs2=launch("main",30,40,120,gx=1,cid=2)
        classic=record(batch=2); classic["exec"]="CLASSIC"
        got=MOD.map_fused([record(),classic],[(1,1,vs1,None),(1,2,vs2,None)],1,2,{"main":13})
        self.assertEqual(len(got),1)

    def test_map_rejects_stream_mismatch(self):
        vs=launch("main",10,20,110,stream=14,cid=1)
        with self.assertRaises(MOD.CostError):
            MOD.map_fused([record()],[(1,1,vs,None)],1,2,{"main":13})

    def test_group_by_full_hash(self):
        a,b="01"*32,"02"*32
        mapped=[(record(hash_=a),launch("main",0,10,1,cid=1)),
                (record(hash_=a),launch("main",5,15,2,cid=2)),
                (record(hash_=b),launch("main",20,25,3,cid=3))]
        activity=[x.device for _,x in mapped]
        got={x["stage_hash"]:x for x in MOD.group_cost(mapped,activity)}
        self.assertEqual((got[a]["launches"],got[a]["sum_ns"],got[a]["union_ns"]),(2,20,15))
        self.assertEqual(got[b]["union_exclusive_ns"],5)

    def test_aggregate_all_mapped_fused_intervals(self):
        mapped=[(record(hash_="01"*32),launch("main",0,10,1,cid=1)),
                (record(hash_="01"*32),launch("main",5,15,2,cid=2)),
                (record(hash_="02"*32),launch("main",20,25,3,cid=3))]
        activity=[x.device for _,x in mapped]
        self.assertEqual(MOD.aggregate_cost(mapped,activity), {
            "launches":3,"sum_ns":25,"union_ns":20,
            "union_exclusive_ns":20,"sum_ms":25/1e6,
            "union_ms":20/1e6,"union_exclusive_ms":20/1e6})

    def test_parse_stream_map_is_total(self):
        full="main=13,"+",".join(f"seg{i}={14+i}" for i in range(8))
        self.assertEqual(MOD.parse_stream_map(full)["seg7"],21)
        with self.assertRaises(MOD.CostError): MOD.parse_stream_map("main=13")

if __name__=="__main__": unittest.main()
