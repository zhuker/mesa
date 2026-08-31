#!/usr/bin/env python3
"""Validate and summarize CUDAVK_VERTEX_REUSE_* machine output."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

RP = "CUDAVK_VERTEX_REUSE_RECORD "
SP = "CUDAVK_VERTEX_REUSE_SUMMARY "
REQUIRED_RECORD = {"version","index","status","submit","batch","bank","generation",
 "fused","fetch_runs","indexed","writes_memory","semantic_metadata_complete",
 "total","unique","duplicate","table_capacity","probes","overflow","bad_lanes"}
REQUIRED_SUMMARY = {"version","capacity","population","reserved","complete","admitted",
 "excluded","record_overflow","table_overflow","final_submit","total","unique","duplicate",
 "indexed_total","indexed_duplicate","nonindexed_total","nonindexed_duplicate",
 "fused_total","fused_duplicate","standalone_total","standalone_duplicate",
 "population_ok","classification_ok","population_screen_64_5","cost_join_required",
 "result_valid","diagnostic_timing_invalid"}

def kv(line: str) -> dict[str,str]:
    out={}
    for item in line.split():
        if "=" not in item: raise ValueError(f"bad token {item!r}")
        k,v=item.split("=",1)
        if k in out: raise ValueError(f"duplicate field {k}")
        out[k]=v
    return out

def integer(d,k):
    try: v=int(d[k],0)
    except Exception as e: raise ValueError(f"bad integer {k}={d.get(k)!r}") from e
    if v < 0: raise ValueError(f"negative {k}")
    return v

def analyze(text: str, require_valid: bool=True) -> dict:
    records=[]; summaries=[]
    for line in text.splitlines():
        if line.startswith(RP): records.append(kv(line[len(RP):]))
        elif line.startswith(SP): summaries.append(kv(line[len(SP):]))
    if len(summaries)!=1: raise ValueError(f"expected one summary, got {len(summaries)}")
    s=summaries[0]
    missing=REQUIRED_SUMMARY-set(s)
    if missing: raise ValueError(f"summary missing {sorted(missing)}")
    if integer(s,"version") != 1: raise ValueError("summary version")
    if len(records)!=integer(s,"reserved"): raise ValueError("record population != reserved")
    if integer(s,"reserved") > integer(s,"capacity"):
        raise ValueError("reserved exceeds capacity")
    identities=set(); sums={"total":0,"unique":0,"duplicate":0}
    cats={(f,i):[0,0] for f in (0,1) for i in (0,1)}
    admitted=excluded=table_overflow=0
    last=None
    for n,r in enumerate(records):
        missing=REQUIRED_RECORD-set(r)
        if missing: raise ValueError(f"record {n} missing {sorted(missing)}")
        if integer(r,"version") != 1 or integer(r,"index") != n:
            raise ValueError(f"record {n} version/index")
        ident=(integer(r,"submit"),integer(r,"batch"))
        if not ident[0] or not ident[1]: raise ValueError(f"record {n} zero identity")
        if integer(r,"bank") >= 9: raise ValueError(f"record {n} bad bank")
        if ident in identities: raise ValueError(f"duplicate identity {ident}")
        identities.add(ident)
        if last is not None and ident <= last: raise ValueError("identities not increasing")
        last=ident
        for b in ("fused","fetch_runs","indexed","writes_memory","semantic_metadata_complete","overflow"):
            if integer(r,b) not in (0,1): raise ValueError(f"record {n} bad bool {b}")
        total=integer(r,"total"); unique=integer(r,"unique"); duplicate=integer(r,"duplicate")
        if unique+duplicate != total: raise ValueError(f"record {n} does not reconcile")
        status=r["status"]
        if status=="complete":
            admitted+=1
            cap=integer(r,"table_capacity")
            if not integer(r,"generation") or not cap or cap&(cap-1):
                raise ValueError(f"record {n} bad admitted table metadata")
            if integer(r,"overflow") or integer(r,"bad_lanes"):
                raise ValueError(f"record {n} complete with failure")
        elif status.startswith("excluded_"):
            excluded+=1
            if duplicate or unique!=total:
                raise ValueError(f"record {n} exclusion not conservatively charged")
        else: raise ValueError(f"record {n} bad status {status}")
        table_overflow += integer(r,"overflow") != 0
        sums["total"]+=total; sums["unique"]+=unique; sums["duplicate"]+=duplicate
        cats[(integer(r,"fused"),integer(r,"indexed"))][0]+=total
        cats[(integer(r,"fused"),integer(r,"indexed"))][1]+=duplicate
    for k,v in sums.items():
        if v != integer(s,k): raise ValueError(f"summary {k} mismatch")
    if admitted != integer(s,"admitted") or excluded != integer(s,"excluded"):
        raise ValueError("summary admitted/excluded mismatch")
    if admitted != integer(s,"complete"): raise ValueError("summary complete mismatch")
    if records and integer(s,"final_submit") < max(i[0] for i in identities):
        raise ValueError("final_submit precedes records")
    pop_ok = integer(s,"population") == integer(s,"reserved")+integer(s,"record_overflow")
    if not pop_ok: raise ValueError("summary population mismatch")
    class_ok = admitted+excluded == integer(s,"reserved")
    if not class_ok: raise ValueError("classification population mismatch")
    if integer(s,"table_overflow") != table_overflow:
        raise ValueError("summary table_overflow mismatch")
    fused=sum(v[0] for (f,i),v in cats.items() if f); standalone=sum(v[0] for (f,i),v in cats.items() if not f)
    indexed=sum(v[0] for (f,i),v in cats.items() if i); nonindexed=sum(v[0] for (f,i),v in cats.items() if not i)
    fd=sum(v[1] for (f,i),v in cats.items() if f); sd=sum(v[1] for (f,i),v in cats.items() if not f)
    idup=sum(v[1] for (f,i),v in cats.items() if i); ndup=sum(v[1] for (f,i),v in cats.items() if not i)
    expected={"fused_total":fused,"fused_duplicate":fd,"standalone_total":standalone,
      "standalone_duplicate":sd,"indexed_total":indexed,"indexed_duplicate":idup,
      "nonindexed_total":nonindexed,"nonindexed_duplicate":ndup}
    for k,v in expected.items():
        if integer(s,k)!=v: raise ValueError(f"summary {k} mismatch")
    screen = sums["total"]>0 and 200*sums["duplicate"] >= 129*sums["total"]
    if s["population_screen_64_5"] != ("yes" if screen else "no"):
        raise ValueError("population screen mismatch")
    for k in ("population_ok","classification_ok","cost_join_required","result_valid"):
        if s[k] not in ("yes","no"): raise ValueError(f"bad summary bool {k}")
    if s["population_ok"] != ("yes" if pop_ok else "no"):
        raise ValueError("population_ok mismatch")
    if s["classification_ok"] != ("yes" if class_ok else "no"):
        raise ValueError("classification_ok mismatch")
    unreadable = any(r["status"] == "excluded_read" for r in records)
    expected_valid = pop_ok and class_ok and integer(s,"record_overflow")==0 and not unreadable
    if s["result_valid"] != ("yes" if expected_valid else "no"):
        raise ValueError("result_valid mismatch")
    if s["diagnostic_timing_invalid"]!="1": raise ValueError("diagnostic timing not invalid")
    if s["cost_join_required"]!="yes": raise ValueError("cost join must be required")
    if require_valid and s["result_valid"]!="yes": raise ValueError("result_valid is not yes")
    def frac(t,d): return {"total":t,"duplicate":d,"fraction":None if not t else d/t}
    groups={"overall":frac(sums["total"],sums["duplicate"]),
      "indexed":frac(indexed,idup),"nonindexed":frac(nonindexed,ndup),
      "fused":frac(fused,fd),"standalone":frac(standalone,sd)}
    for (f,i),(t,d) in sorted(cats.items()):
        groups[("fused" if f else "standalone")+"_"+("indexed" if i else "nonindexed")]=frac(t,d)
    return {"version":1,"result_valid":s["result_valid"]=="yes",
      "population_screen_64_5":screen,"cost_join_required":True,
      "production_admitted":False,"groups":groups,
      "excluded_batches":excluded,"semantic_gate_complete":bool(records) and all(integer(r,"semantic_metadata_complete") for r in records)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="stderr/report file; default stdin")
    ap.add_argument("--expect-final-submit", type=int)
    ap.add_argument("--expect-records", type=int)
    ns=ap.parse_args()
    text=Path(ns.input).read_text() if ns.input else sys.stdin.read()
    try:
        out=analyze(text)
        summaries=[kv(line[len(SP):]) for line in text.splitlines() if line.startswith(SP)]
        if ns.expect_final_submit is not None and integer(summaries[0],"final_submit") != ns.expect_final_submit:
            raise ValueError("unexpected final_submit")
        if ns.expect_records is not None and integer(summaries[0],"reserved") != ns.expect_records:
            raise ValueError("unexpected record count")
    except ValueError as e: raise SystemExit(f"vertex reuse report invalid: {e}")
    print(json.dumps(out,sort_keys=True,indent=2))
if __name__=="__main__": main()
