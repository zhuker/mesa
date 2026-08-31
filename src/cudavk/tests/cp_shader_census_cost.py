#!/usr/bin/env python3
"""CPU-only clean fused-VS cost join grouped by full compiled-stage cache key.

The join keeps absolute (command-bearing submit, executed batch) identities.
It recognizes generated VS `main` only through immediate same-stream LEAD/LAG:
a clean fused VS has no immediate `cp_vertex_fetch` predecessor and has an
immediate clip/raster successor. No NVTX range is required.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import importlib.util
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUCCESSORS = {"cp_clip_triangles", "cp_clip_rast_fused", "cp_clip_rast_fused_abuf"}
LAUNCH_APIS = {"cuLaunchKernel", "cuLaunchKernelEx"}

class CostError(ValueError): pass

@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    def __post_init__(self):
        if self.start < 0 or self.end <= self.start: raise CostError("invalid interval")
    @property
    def duration(self): return self.end - self.start

@dataclass(frozen=True)
class Launch:
    name: str
    device: Interval
    runtime: Interval
    correlation: int
    grid_x: int
    block_x: int
    stream: int


def require_columns(db, table, needed):
    have = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
    if not have: raise CostError(f"trace lacks table {table}")
    if needed-have: raise CostError(f"trace {table} lacks {sorted(needed-have)}")


def session_clock(db) -> int:
    require_columns(db,"TARGET_INFO_SESSION_START_TIME",{"systemClockNs"})
    rows=db.execute("SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchall()
    if len(rows)!=1 or rows[0][0] is None: raise CostError("expected one nsys system clock origin")
    return int(rows[0][0])


def load_launches(db, device_id: int = 0) -> list[Launch]:
    require_columns(db, "StringIds", {"id", "value"})
    require_columns(db, "CUPTI_ACTIVITY_KIND_KERNEL", {"start","end","deviceId","correlationId","shortName","gridX","blockX","streamId"})
    require_columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME", {"start","end","correlationId","nameId"})
    rows = db.execute("""
      SELECT k.start,k.end,k.correlationId,ks.value,k.gridX,k.blockX,k.streamId,
             r.start,r.end,rs.value
      FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds ks ON ks.id=k.shortName
      JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
      JOIN StringIds rs ON rs.id=r.nameId WHERE k.deviceId=? ORDER BY r.start,k.start,k.correlationId
    """,(device_id,)).fetchall()
    result, seen = [], set()
    for ds,de,cid,name,gx,bx,stream,rs,re_,api in rows:
        if cid is None or int(cid) in seen: raise CostError(f"duplicate/missing correlationId {cid}")
        if api not in LAUNCH_APIS: raise CostError(f"correlationId {cid} runtime row is {api}")
        seen.add(int(cid)); result.append(Launch(str(name), Interval(int(ds),int(de)),
            Interval(int(rs),int(re_)), int(cid), int(gx), int(bx), int(stream)))
    return result


def parse_timestamps(path: Path) -> tuple[list[int], str | None]:
    raw_lines=path.read_text().splitlines(keepends=True); rows=[]; torn=None
    for line_no,raw in enumerate(raw_lines,1):
        terminated=raw.endswith(("\n","\r")); line=raw.rstrip("\r\n")
        if not line.strip(): continue
        parts=line.split()
        if len(parts)!=2 or not all(x.isdigit() for x in parts):
            if line_no==len(raw_lines) and not terminated and len(parts)==1 and parts[0].isdigit():
                torn=parts[0]; continue
            raise CostError(f"{path}:{line_no}: malformed timestamp row")
        timestamp,index=map(int,parts)
        if index != len(rows): raise CostError(f"{path}:{line_no}: non-absolute timestamp index")
        if rows and timestamp <= rows[-1]: raise CostError(f"{path}:{line_no}: timestamps not increasing")
        rows.append(timestamp)
    if len(rows)<2: raise CostError("timestamp file is empty/short")
    return rows,torn

def classify(launches: list[Launch], timestamps: list[int], submit_offset: int):
    next_same=[None]*len(launches); following={}
    for i in range(len(launches)-1,-1,-1):
        next_same[i]=following.get(launches[i].stream); following[launches[i].stream]=i
    previous={}; counts=collections.Counter(); result=[]
    for i,launch in enumerate(launches):
        prev=previous.get(launch.stream); previous[launch.stream]=i
        nxt=next_same[i]
        if launch.name != "main" or nxt is None or launches[nxt].name not in SUCCESSORS: continue
        external=bisect.bisect_right(timestamps, launch.runtime.start)-1
        # The trace includes startup before the shim's first completed row and
        # can include a torn final submit. Neither has an absolute shim window.
        if external < 0 or external >= len(timestamps)-1: continue
        submit=external+submit_offset; counts[submit]+=1
        fetch=launches[prev] if prev is not None and launches[prev].name=="cp_vertex_fetch" else None
        result.append((submit, counts[submit], launch, fetch))
    return result


def load_analyzer():
    path=Path(__file__).with_name("cp_shader_census_analyze.py")
    spec=importlib.util.spec_from_file_location("cp_shader_census_analyze",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def map_fused(records, classified, submit_lo, submit_hi, stream_map):
    selected=[r for r in records if submit_lo <= int(r["submit"]) < submit_hi]
    units={(s,b):(launch,fetch) for s,b,launch,fetch in classified
           if submit_lo <= s < submit_hi}
    unit_count=sum(1 for s,b,launch,fetch in classified if submit_lo <= s < submit_hi)
    if len(units) != unit_count: raise CostError("duplicate clean absolute identity")
    wanted={(int(r["submit"]),int(r["batch"])) for r in selected}
    if wanted != set(units):
        raise CostError(f"census/clean VS population differs: census_only={len(wanted-set(units))} trace_only={len(set(units)-wanted)}")
    mapped=[]
    for record in selected:
        key=(int(record["submit"]),int(record["batch"])); launch,fetch=units[key]
        gx=int(record["grid"].split("x")[0]); bx=int(record["block"].split("x")[0])
        if (gx,bx,stream_map.get(record["stream"])) != (launch.grid_x,launch.block_x,launch.stream):
            raise CostError(f"shape/stream mismatch for absolute identity {key}")
        if bool(fetch) != bool(int(record["fetch_runs"])):
            raise CostError(f"fetch predecessor mismatch for absolute identity {key}")
        if record["exec"] == "VS_FETCH":
            if fetch is not None: raise CostError(f"fused identity {key} has a standalone fetch")
            mapped.append((record,launch))
    if not mapped: raise CostError("selected clean fused population is empty")
    return mapped

def load_activity(db, lo, hi, device_id: int = 0) -> list[Interval]:
    result=[]
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET"):
        require_columns(db,table,{"start","end","deviceId"})
        for start,end in db.execute(f'SELECT start,end FROM "{table}" WHERE deviceId=? AND end>? AND start<?',(device_id,lo,hi)):
            start,end=max(lo,int(start)),min(hi,int(end))
            if end>start: result.append(Interval(start,end))
    return result


def union_and_exclusive(selected: Iterable[Interval], all_activity: Iterable[Interval]):
    events=collections.defaultdict(lambda:[0,0])
    for x in all_activity: events[x.start][0]+=1; events[x.end][0]-=1
    for x in selected: events[x.start][1]+=1; events[x.end][1]-=1
    all_n=selected_n=union=exclusive=0; previous=None
    for timestamp in sorted(events):
        if previous is not None and selected_n:
            dt=timestamp-previous; union+=dt
            if all_n==selected_n: exclusive+=dt
        da,ds=events[timestamp]; all_n+=da; selected_n+=ds; previous=timestamp
        if all_n<0 or selected_n<0 or selected_n>all_n: raise CostError("activity sweep inconsistency")
    if all_n or selected_n: raise CostError("unterminated activity sweep")
    return union,exclusive


def interval_cost(intervals, all_activity):
    intervals=list(intervals)
    union,exclusive=union_and_exclusive(intervals,all_activity)
    total=sum(x.duration for x in intervals)
    return {"launches":len(intervals),"sum_ns":total,"union_ns":union,
            "union_exclusive_ns":exclusive,"sum_ms":total/1e6,
            "union_ms":union/1e6,"union_exclusive_ms":exclusive/1e6}


def group_cost(mapped, all_activity):
    groups=collections.defaultdict(list)
    for record,launch in mapped: groups[record["stage_hash"]].append(launch.device)
    output=[]
    for hash_,intervals in sorted(groups.items()):
        row={"stage_hash":hash_}; row.update(interval_cost(intervals,all_activity))
        output.append(row)
    return output


def aggregate_cost(mapped, all_activity):
    return interval_cost((launch.device for _record,launch in mapped),all_activity)

def parse_stream_map(value):
    result={}
    for item in value.split(","):
        name,raw=item.split("=",1); result[name]=int(raw)
    if set(result)!={"main",*(f"seg{i}" for i in range(8))}: raise CostError("stream map must name main and seg0..7 exactly")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",required=True,type=Path); p.add_argument("--timestamps",required=True,type=Path)
    p.add_argument("--census",required=True,type=Path); p.add_argument("--submit-lo",required=True,type=int)
    p.add_argument("--submit-hi",required=True,type=int); p.add_argument("--submit-offset",type=int,default=1)
    p.add_argument("--frames",type=float); p.add_argument("--device-id",type=int,default=0); p.add_argument("--stream-map",default="main=13,seg0=14,seg1=15,seg2=16,seg3=17,seg4=18,seg5=19,seg6=20,seg7=21")
    args=p.parse_args()
    try:
        analyzer=load_analyzer(); census=analyzer.analyze(args.census.read_text())
        stream_map=parse_stream_map(args.stream_map); timestamps,torn_tail=parse_timestamps(args.timestamps)
        with sqlite3.connect(args.trace) as db:
            clock=session_clock(db); relative=[value-clock for value in timestamps]
            if relative[0] < 0: raise CostError("shim timestamps precede nsys clock origin")
            launches=load_launches(db,args.device_id); classified=classify(launches,relative,args.submit_offset)
            mapped=map_fused(census["records"],classified,args.submit_lo,args.submit_hi,stream_map)
            lo=min(x.device.start for _,x in mapped); hi=max(x.device.end for _,x in mapped)
            activity=load_activity(db,lo,hi,args.device_id)
            groups=group_cost(mapped,activity); aggregate=aggregate_cost(mapped,activity)
        output={"version":1,"population":"clean_fused_generated_vs","absolute_identity":"(command-bearing submit, executed batch)",
                "submit_range":[args.submit_lo,args.submit_hi],"device_id":args.device_id,"mapped_launches":len(mapped),
                "aggregate":aggregate,"groups":groups,
                "diagnostic_timing_invalid":1,"cost_join_required":"satisfied-for-selected-window",
                "production_admitted":"no","timestamp_torn_tail":torn_tail,"architecture_scope":"trace GPU only; RTX results do not apply to B200"}
        if args.frames:
            if args.frames<=0: raise CostError("frames must be positive")
            for row in [aggregate, *groups]:
                for name in ("sum","union","union_exclusive"):
                    row[name+"_ms_per_frame"]=row[name+"_ms"]/args.frames
        print(json.dumps(output,sort_keys=True))
    except (OSError,sqlite3.Error,CostError,ValueError) as exc:
        print(f"shader-census-cost: invalid: {exc}",file=sys.stderr); return 1
    return 0

if __name__=="__main__": raise SystemExit(main())
