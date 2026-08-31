#!/usr/bin/env python3
"""Analyze CUDAVK_GEOM_B0 records without touching a CUDA device."""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

PREFIX = "CUDAVK_GEOM_B0_RECORD "
SUMMARY = "CUDAVK_GEOM_B0 "
ADMITTED_POOL_MS = 0.5522067056
KILL_THRESHOLD_MS = 0.3153282555
KILL_THRESHOLD_PERCENT = 57.10330068
REQUIRED = {"index","status","salt","submit","occurrence","unit","pass",
 "path","predecessor","w","h","samples","bytes","signature",
 "stage2_blocks","triangles","nontrivial","huge","setup","queue_status","sha256"}
HEX = {"salt","signature"}

class CensusError(ValueError): pass

def fields(line: str) -> dict[str,str]:
    out = {}
    for word in line.strip().split():
        if "=" in word:
            k,v = word.split("=",1)
            if k in out: raise CensusError(f"duplicate field {k}")
            out[k]=v
    return out

def parse_record(line: str) -> dict:
    if not line.startswith(PREFIX): raise CensusError("not a B0 record")
    raw=fields(line[len(PREFIX):])
    missing=REQUIRED-raw.keys()
    if missing: raise CensusError(f"missing fields: {sorted(missing)}")
    for k in HEX:
        if not re.fullmatch(r"[0-9a-f]{16}",raw[k]): raise CensusError(f"bad {k}")
    out={k:(int(v,16) if k in HEX else v) for k,v in raw.items()}
    numeric=REQUIRED-HEX-{"sha256"}
    for k in numeric: out[k]=int(raw[k],10)
    if any(out[k] < 0 for k in numeric): raise CensusError("negative record field")
    if not re.fullmatch(r"[0-9a-f]{64}",raw["sha256"]):
        raise CensusError("corrupt sha256")
    if out["status"] not in (1,2): raise CensusError("bad status")
    if out["bytes"] != out["w"]*out["h"]*out["samples"]*8:
        raise CensusError("byte-length mismatch")
    return out

def parse_run(path: Path) -> tuple[list[dict],dict]:
    recs=[]; summary=None; seen=set()
    for line in path.read_text().splitlines():
        if line.startswith(PREFIX):
            r=parse_record(line)
            if r["index"] in seen: raise CensusError(f"duplicate index {r['index']}")
            seen.add(r["index"]); recs.append(r)
        elif line.startswith(SUMMARY):
            if summary is not None: raise CensusError("duplicate summary")
            summary=fields(line[len(SUMMARY):])
    if summary is None: raise CensusError("missing summary")
    for k in ("capacity","reserved","complete","candidates","admitted","reasons","mode","salt","result_valid"):
        if k not in summary: raise CensusError(f"summary missing {k}")
    for k in ("capacity","reserved","complete","candidates","admitted","reasons","mode"):
        summary[k]=int(summary[k])
    if not re.fullmatch(r"[0-9a-f]{16}",summary["salt"]): raise CensusError("bad summary salt")
    summary["salt_value"]=int(summary["salt"],16)
    if any(summary[k] < 0 for k in ("capacity","reserved","complete","candidates","admitted","reasons","mode")):
        raise CensusError("negative summary field")
    if summary["reserved"] > summary["capacity"]: raise CensusError("reserved exceeds capacity")
    if summary["result_valid"] != "yes": raise CensusError("driver summary is invalid")
    if sorted(seen) != list(range(summary["reserved"])): raise CensusError("record indices are not contiguous from zero")
    if summary["reserved"] != len(recs):
        raise CensusError(f"missing records: reserved={summary['reserved']} parsed={len(recs)}")
    if summary["complete"] != len(recs): raise CensusError("complete count mismatch")
    if summary["admitted"] != sum(r["status"]==1 for r in recs):
        raise CensusError("admitted count mismatch")
    if summary["admitted"]+summary["reasons"] != summary["candidates"]:
        raise CensusError("classification invariant failed")
    salts={r["salt"] for r in recs}
    if len(salts)>1: raise CensusError("mixed salts")
    if salts and next(iter(salts)) != summary["salt_value"]: raise CensusError("summary/record salt mismatch")
    expected={1:0x47454F4D42305331,2:0x47454F4D42305332}.get(summary["mode"])
    if expected is None or summary["salt_value"] != expected: raise CensusError("mode/salt mismatch")
    return sorted(recs,key=lambda r:r["index"]),summary

def frame_number(submit:int, origin:int, submits_per_frame:int)->int:
    if submits_per_frame<=0: raise CensusError("submits_per_frame must be positive")
    if submit<origin: raise CensusError("submit before frame origin")
    return (submit-origin)//submits_per_frame

def group_frames(records:list[dict], origin:int|None=None, submits_per_frame:int=2):
    admitted=[r for r in records if r["status"]==1]
    if not admitted: return {}, origin
    if origin is None: origin=min(r["submit"] for r in admitted)
    frames=collections.defaultdict(list)
    for r in admitted: frames[frame_number(r["submit"],origin,submits_per_frame)].append(r)
    for v in frames.values(): v.sort(key=lambda r:(r["submit"],r["occurrence"],r["unit"],r["index"]))
    return dict(frames),origin

def structural_population_digest(records:list[dict])->str:
    fields=("submit","occurrence","unit","pass","path","predecessor","signature",
            "w","h","samples","bytes","stage2_blocks","triangles")
    rows=[tuple(r[k] for k in fields) for r in records if r["status"]==1]
    return hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()

def recurrence_key(r:dict)->tuple:
    return (r["w"],r["h"],r["samples"],r["sha256"])

def analyze(records:list[dict], origin:int|None=None, submits_per_frame:int=2)->dict:
    frames,origin=group_frames(records,origin,submits_per_frame)
    membership=chronological=compared=0
    transitions=[]; classification=[]
    keys=sorted(frames)
    for frame in keys:
        if frame-1 not in frames: continue
        prev,curr=frames[frame-1],frames[frame]
        previous_keys={recurrence_key(r) for r in prev}
        flags=[recurrence_key(r) in previous_keys for r in curr]
        chrono_flags=[i<len(prev) and recurrence_key(r)==recurrence_key(prev[i]) for i,r in enumerate(curr)]
        m=sum(flags); c=sum(chrono_flags)
        membership+=m; chronological+=c; compared+=len(curr)
        classification.extend((r["submit"],r["occurrence"],r["unit"],r["pass"],matched)
                              for r,matched in zip(curr,flags))
        transitions.append({"frame":frame,"current":len(curr),"previous":len(prev),
                            "membership":m,"chronological":c})
    return {"frame_origin":origin,"submits_per_frame":submits_per_frame,
      "frames":len(frames),"transitions":transitions,"compared":compared,
      "membership_matches":membership,"chronological_matches":chronological,
      "membership_rate":membership/compared if compared else 0.0,
      "chronological_rate":chronological/compared if compared else 0.0,
      "admitted_pool_ms_per_frame":ADMITTED_POOL_MS,
      "required_cost_weighted_recurrence_ms_per_frame":KILL_THRESHOLD_MS,
      "required_cost_weighted_recurrence_percent":KILL_THRESHOLD_PERCENT,
      "verdict_available":False,
      "verdict_unavailable_reason":"requires external clean-trace interval cost join",
      "structural_population_sha256":structural_population_digest(records),
      "membership_classification":classification}

def validate_salt_reruns(runs:list[dict])->dict:
    if len(runs)<=1: return {"checked":False}
    salts=[r["salt"] for r in runs]; pops=[r["structural_population_sha256"] for r in runs]
    classes=[r["membership_classification"] for r in runs]
    if None in salts or len(set(salts))!=len(salts): raise CensusError("salt reruns are not independent")
    if len(set(pops))!=1: raise CensusError("salt rerun structural populations differ")
    if any(c != classes[0] for c in classes[1:]): raise CensusError("salt rerun match classifications differ")
    return {"checked":True,"independent_salts":True,"population_equal":True,"classification_equal":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("logs",type=Path,nargs="+")
    ap.add_argument("--submits-per-frame",type=int,default=2)
    ap.add_argument("--frame-origin",type=int)
    args=ap.parse_args()
    runs=[]
    for path in args.logs:
        recs,summary=parse_run(path)
        result=analyze(recs,args.frame_origin,args.submits_per_frame)
        result.update(path=str(path),summary=summary,
                      salt=f"{summary['salt_value']:016x}")
        runs.append(result)
    rerun=validate_salt_reruns(runs)
    print(json.dumps({"runs":runs,"salt_rerun":rerun},sort_keys=True))
if __name__=="__main__": main()
