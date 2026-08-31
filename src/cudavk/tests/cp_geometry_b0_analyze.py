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
SALT1 = 0x47454F4D42305331
SALT2 = 0x47454F4D42305332
ZERO_SHA256 = "0" * 64

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
    out["_retry_present"]="retry" in raw
    out["retry"]=int(raw.get("retry","0"),10)
    if out["retry"] not in (0,1): raise CensusError("bad retry")
    if "sha256_2" in raw:
        if not re.fullmatch(r"[0-9a-f]{64}",raw["sha256_2"]):
            raise CensusError("corrupt sha256_2")
        out["sha256_2"]=raw["sha256_2"]
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
    if "salt2" in summary:
        if not re.fullmatch(r"[0-9a-f]{16}",summary["salt2"]):
            raise CensusError("bad summary salt2")
        summary["salt2_value"]=int(summary["salt2"],16)
    else:
        summary["salt2_value"]=0
    if "second_roots_bytes" in summary:
        try:
            summary["second_roots_bytes"]=int(summary["second_roots_bytes"],10)
        except ValueError as error:
            raise CensusError("bad summary second_roots_bytes") from error
        if summary["second_roots_bytes"] < 0:
            raise CensusError("negative summary field")
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

    mode=summary["mode"]
    expected_primary={1:SALT1,2:SALT2,3:SALT1}.get(mode)
    if expected_primary is None or summary["salt_value"] != expected_primary:
        raise CensusError("mode/salt mismatch")
    if mode == 3:
        for key in ("salt2","dual_domain","second_roots_bytes"):
            if key not in summary:
                raise CensusError(f"mode 3 summary missing {key}")
        if summary["salt2_value"] != SALT2:
            raise CensusError("mode 3 salt pair mismatch")
        if summary["dual_domain"] != "yes":
            raise CensusError("mode 3 is not dual-domain")
        if summary["second_roots_bytes"] != summary["capacity"] * 32:
            raise CensusError("mode 3 second-root allocation mismatch")
        for r in recs:
            if not r["_retry_present"]:
                raise CensusError("mode 3 record missing retry")
            if "sha256_2" not in r:
                raise CensusError("mode 3 record missing sha256_2")
            if r["sha256_2"] == ZERO_SHA256:
                raise CensusError("mode 3 record has zero sha256_2")
    else:
        if summary["salt2_value"] != 0:
            raise CensusError("single-domain mode has a secondary salt")
        if summary.get("dual_domain","no") != "no":
            raise CensusError("single-domain mode marked dual-domain")
        if summary.get("second_roots_bytes",0) != 0:
            raise CensusError("single-domain mode has second-root storage")
        if any(r.get("sha256_2",ZERO_SHA256) != ZERO_SHA256 for r in recs):
            raise CensusError("single-domain mode has a secondary digest")
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

def structural_population(records:list[dict])->list[tuple]:
    fields=("submit","occurrence","unit","pass","retry","path","predecessor","signature",
            "w","h","samples","bytes","stage2_blocks","triangles")
    return [tuple(r.get(k,0) if k=="retry" else r[k] for k in fields)
            for r in records if r["status"]==1]

def structural_population_digest(records:list[dict])->str:
    rows=structural_population(records)
    return hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()

def mode3_record_population(records:list[dict])->list[tuple]:
    fields=("index","status","submit","occurrence","unit","pass","retry",
            "path","predecessor","signature","w","h","samples","bytes",
            "stage2_blocks","triangles","nontrivial","huge","setup","queue_status")
    return [tuple(r[k] for k in fields) for r in records]

def recurrence_key(r:dict,digest_field:str="sha256")->tuple:
    # Pass and retry are absolute population identity, not content identity.
    return (r["w"],r["h"],r["samples"],r[digest_field])

def _domain_analysis(frames:dict[int,list[dict]],digest_field:str)->dict:
    membership=chronological=compared=0
    transitions=[]; membership_classification=[]; chronological_classification=[]
    for frame in sorted(frames):
        if frame-1 not in frames: continue
        prev,curr=frames[frame-1],frames[frame]
        previous_keys={recurrence_key(r,digest_field) for r in prev}
        flags=[recurrence_key(r,digest_field) in previous_keys for r in curr]
        chrono_flags=[i<len(prev) and
                      recurrence_key(r,digest_field)==recurrence_key(prev[i],digest_field)
                      for i,r in enumerate(curr)]
        m=sum(flags); c=sum(chrono_flags)
        membership+=m; chronological+=c; compared+=len(curr)
        absolute=[(r["submit"],r["occurrence"],r["unit"],r["pass"])
                  for r in curr]
        membership_classification.extend(key+(matched,)
                                         for key,matched in zip(absolute,flags))
        chronological_classification.extend(key+(matched,)
                                             for key,matched in zip(absolute,chrono_flags))
        transitions.append({"frame":frame,"current":len(curr),"previous":len(prev),
                            "membership":m,"chronological":c})
    return {"transitions":transitions,"compared":compared,
            "membership_matches":membership,
            "chronological_matches":chronological,
            "membership_classification":membership_classification,
            "chronological_classification":chronological_classification}

def analyze(records:list[dict], origin:int|None=None, submits_per_frame:int=2,
            mode:int|None=None)->dict:
    frames,origin=group_frames(records,origin,submits_per_frame)
    if mode is None:
        mode=3 if any(r.get("sha256_2",ZERO_SHA256) != ZERO_SHA256
                      for r in records) else 1
    if mode not in (1,2,3): raise CensusError("bad analysis mode")
    if mode == 3:
        for r in records:
            if "sha256_2" not in r: raise CensusError("mode 3 record missing sha256_2")
            if r["sha256_2"] == ZERO_SHA256:
                raise CensusError("mode 3 record has zero sha256_2")

    primary=_domain_analysis(frames,"sha256")
    paired={"checked":False}
    if mode == 3:
        secondary=_domain_analysis(frames,"sha256_2")
        membership_equal=(primary["membership_classification"] ==
                          secondary["membership_classification"])
        chronological_equal=(primary["chronological_classification"] ==
                             secondary["chronological_classification"])
        if not membership_equal:
            raise CensusError("paired-domain membership classifications differ")
        if not chronological_equal:
            raise CensusError("paired-domain chronological classifications differ")
        paired={"checked":True,"equal":True,"membership_equal":True,
                "chronological_equal":True}

    compared=primary["compared"]
    return {"mode":mode,"frame_origin":origin,"submits_per_frame":submits_per_frame,
      "frames":len(frames),"transitions":primary["transitions"],"compared":compared,
      "membership_matches":primary["membership_matches"],
      "chronological_matches":primary["chronological_matches"],
      "membership_rate":primary["membership_matches"]/compared if compared else 0.0,
      "chronological_rate":primary["chronological_matches"]/compared if compared else 0.0,
      "admitted_pool_ms_per_frame":ADMITTED_POOL_MS,
      "required_cost_weighted_recurrence_ms_per_frame":KILL_THRESHOLD_MS,
      "required_cost_weighted_recurrence_percent":KILL_THRESHOLD_PERCENT,
      "verdict_available":False,
      "verdict_unavailable_reason":"requires external clean-trace interval cost join",
      "_mode3_population":mode3_record_population(records) if mode==3 else None,
      "structural_population_sha256":structural_population_digest(records),
      "membership_classification":primary["membership_classification"],
      "chronological_classification":primary["chronological_classification"],
      "paired_domain":paired,
      "paired_domain_checked":paired["checked"],
      "paired_domain_equal":paired.get("equal",False)}

def _absolute_classification(rows:list[tuple])->tuple[list[tuple],dict[tuple,bool]]:
    order=[]; out={}
    for row in rows:
        if len(row)!=5: raise CensusError("malformed membership classification")
        key=tuple(row[:4]); matched=row[4]
        if not isinstance(matched,bool):
            raise CensusError("malformed membership classification")
        if key in out: raise CensusError(f"duplicate absolute classification {key}")
        order.append(key); out[key]=matched
    return order,out

def validate_salt_reruns(runs:list[dict])->dict:
    if len(runs)<=1: return {"checked":False}
    modes=[r.get("mode") for r in runs]
    contains_mode3=any(m==3 or r.get("paired_domain_checked") is True
                       for m,r in zip(modes,runs))
    if contains_mode3:
        if not all(m==3 for m in modes):
            raise CensusError("cannot mix mode 3 and single-domain reruns")
        if not all(r.get("paired_domain_checked") is True and
                   r.get("paired_domain_equal") is True for r in runs):
            raise CensusError("mode 3 rerun lacks paired-domain validity")
        expected_salt=f"{SALT1:016x}"; expected_salt2=f"{SALT2:016x}"
        for r in runs:
            if r.get("salt") != expected_salt or r.get("salt2") != expected_salt2:
                raise CensusError("mode 3 rerun salt pair mismatch")
        populations=[r.get("_mode3_population") for r in runs]
        if any(population is None for population in populations) or \
           any(population != populations[0] for population in populations[1:]):
            raise CensusError("mode 3 rerun structural populations differ")
        parsed=[_absolute_classification(r["membership_classification"])
                for r in runs]
        order=parsed[0][0]
        if any(keys != order for keys,_ in parsed[1:]):
            raise CensusError("mode 3 rerun absolute classification populations differ")
        maps=[mapping for _,mapping in parsed]
        flips=[key for key in order if len({mapping[key] for mapping in maps})>1]
        union=[key+(any(mapping[key] for mapping in maps),) for key in order]
        return {"checked":True,"paired_mode3_repeats":True,
                "population_equal":True,"classification_equal":not flips,
                "absolute_flip_set":flips,"absolute_flip_count":len(flips),
                "union_recurrent_classification":union}

    salts=[r.get("salt") for r in runs]
    pops=[r["structural_population_sha256"] for r in runs]
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
        result=analyze(recs,args.frame_origin,args.submits_per_frame,
                       mode=summary["mode"])
        result.update(path=str(path),summary=summary,
                      salt=f"{summary['salt_value']:016x}",
                      salt2=f"{summary['salt2_value']:016x}")
        runs.append(result)
    rerun=validate_salt_reruns(runs)
    for run in runs:
        run.pop("_mode3_population",None)
    print(json.dumps({"runs":runs,"salt_rerun":rerun},sort_keys=True))
if __name__=="__main__": main()
