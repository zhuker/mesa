#!/usr/bin/env python3
"""Join the vertex-reuse census to a calibrated 411-frame clean Nsight window.

CPU-only diagnostic.  It does not profile and it does not build a production
cache.  Absolute census identity is always (internal submit, batch).
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

FRAME_ORIGIN = 2784
FRAME_COUNT = 411
SUBMITS_PER_FRAME = 2
TRACE_START_EXTERNAL = FRAME_ORIGIN - 1
SUBMIT_LO = 2786
SUBMIT_HI = 3608
EXPECTED_FINAL_INTERNAL = 6947
EXPECTED_EXTERNAL_ROWS = 6939
EXPECTED_RECORDS = 19929
EXPECTED_FETCH_ROWS = 3087
BANK_STREAM = {bank: 13 + bank for bank in range(9)}
LAUNCH_APIS = {"cuLaunchKernel", "cuLaunchKernelEx"}
VS_SUCCESSORS = {"cp_clip_triangles", "cp_clip_rast_fused", "cp_clip_rast_fused_abuf"}
RP = "CUDAVK_VERTEX_REUSE_RECORD "
SP = "CUDAVK_VERTEX_REUSE_SUMMARY "

# Filled from this immutable clean trace after the structural join.  These are
# integer ns and make accidental trace/window reuse fail closed.
EXPECTED_POOL_CALIBRATION = {
    "pool_sum_ns": 485713115,
    "pool_union_ns": 337441597,
    "pool_union_exclusive_ns": 298152663,
}
EXPECTED_COST_FINGERPRINT = (
    "6106732e85413b2110fd269fcd5f1d691bcf443361da2d7c1b039cfb0476c187")


class CostError(ValueError):
    pass


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise CostError(f"invalid interval [{self.start},{self.end})")

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Launch:
    name: str
    device: Interval
    runtime: Interval
    correlation_id: int
    grid_x: int
    block_x: int
    stream_id: int
    runtime_api: str


def _load_analyzer(path: Path):
    analyzer_path = path.with_name("cp_vertex_reuse_analyze.py")
    spec = importlib.util.spec_from_file_location("cp_vertex_reuse_analyze", analyzer_path)
    if spec is None or spec.loader is None:
        raise CostError(f"cannot load analyzer: {analyzer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_submit_timestamps(path: Path) -> tuple[list[int], str | None]:
    # Keep terminators: a killed producer's final numeric fragment is allowed
    # only when the file physically ends after its digit.  "1503\n" is a
    # complete malformed row and must not be mistaken for that torn write.
    lines = path.read_text().splitlines(keepends=True)
    rows: list[tuple[int, int, int]] = []
    torn = None
    row_re = re.compile(r"\s*([0-9]+)\s+([0-9]+)\s*")
    for line_no, raw in enumerate(lines, 1):
        terminated = raw.endswith(("\n", "\r"))
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        match = row_re.fullmatch(line)
        if match is None:
            is_physical_tail = line_no == len(lines) and not terminated
            if is_physical_tail and re.fullmatch(r"\s*[0-9]+\s*", line):
                torn = line.strip()
                continue
            raise CostError(f"{path}:{line_no}: malformed timestamp row {line!r}")
        rows.append((int(match.group(1)), int(match.group(2)), line_no))
    for expected, (timestamp, index, line_no) in enumerate(rows):
        if index != expected:
            raise CostError(f"{path}:{line_no}: expected index {expected}, got {index}")
        if expected and timestamp <= rows[expected - 1][0]:
            raise CostError(f"{path}:{line_no}: timestamps are not increasing")
    if len(rows) != EXPECTED_EXTERNAL_ROWS:
        raise CostError(f"expected {EXPECTED_EXTERNAL_ROWS} complete external rows, got {len(rows)}")
    if len(rows) <= TRACE_START_EXTERNAL + (FRAME_COUNT + 1) * SUBMITS_PER_FRAME:
        raise CostError("submit timestamps do not cover calibrated frame window")
    return [row[0] for row in rows], torn


def require_columns(db: sqlite3.Connection, table: str, required: set[str]) -> None:
    columns = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
    if not columns:
        raise CostError(f"trace is missing table {table}")
    missing = required - columns
    if missing:
        raise CostError(f"trace table {table} lacks {sorted(missing)}")


def session_clock(db: sqlite3.Connection) -> int:
    require_columns(db, "TARGET_INFO_SESSION_START_TIME", {"systemClockNs"})
    rows = db.execute("SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchall()
    if len(rows) != 1 or rows[0][0] is None:
        raise CostError("expected one non-null nsys systemClockNs")
    return int(rows[0][0])


def nvtx_state(db: sqlite3.Connection) -> dict:
    have = db.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS'"
    ).fetchone()[0]
    if not have:
        return {"table_present": False, "closed_vertex_ranges": 0,
                "marks_ignored": 0, "used_for_mapping": False,
                "fallback": "immediate same-stream LEAD/LAG plus grid/stream/fetch checks"}
    require_columns(db, "NVTX_EVENTS", {"text", "start", "end"})
    closed = db.execute(
        "SELECT count(*) FROM NVTX_EVENTS WHERE text='vertex' AND \"end\" IS NOT NULL"
    ).fetchone()[0]
    marks = db.execute(
        "SELECT count(*) FROM NVTX_EVENTS WHERE text IS NOT NULL AND \"end\" IS NULL"
    ).fetchone()[0]
    # This trace has no NVTX table.  A future table must not silently alter the
    # calibrated mapping.  In particular, end-null runtime marks are not ranges.
    return {"table_present": True, "closed_vertex_ranges": int(closed),
            "marks_ignored": int(marks), "used_for_mapping": False,
            "fallback": "immediate same-stream LEAD/LAG plus grid/stream/fetch checks"}


def load_launches(db: sqlite3.Connection) -> list[Launch]:
    """Load every GPU kernel in filtered host launch order.

    Non-kernel runtime rows (memcopies, event records, NVTX marks) never enter
    this sequence.  Successor/predecessor classification then stays within
    each CUDA stream.
    """
    require_columns(db, "StringIds", {"id", "value"})
    require_columns(db, "CUPTI_ACTIVITY_KIND_KERNEL", {
        "start", "end", "correlationId", "shortName", "gridX", "blockX", "streamId"
    })
    require_columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME", {
        "start", "end", "correlationId", "nameId"
    })
    rows = db.execute("""
        SELECT k.start,k.end,k.correlationId,ks.value,k.gridX,k.blockX,k.streamId,
               r.start,r.end,rs.value
          FROM CUPTI_ACTIVITY_KIND_KERNEL k
          JOIN StringIds ks ON ks.id=k.shortName
          JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
          JOIN StringIds rs ON rs.id=r.nameId
         ORDER BY r.start,k.start,k.correlationId
    """).fetchall()
    launches = []
    correlations = set()
    for ds, de, cid, name, grid, block, stream, rs, re_, api in rows:
        if cid is None or int(cid) in correlations:
            raise CostError(f"missing or duplicate kernel correlationId {cid}")
        correlations.add(int(cid))
        if api not in LAUNCH_APIS:
            raise CostError(f"kernel correlationId {cid} has non-launch runtime row {api}")
        launches.append(Launch(str(name), Interval(int(ds), int(de)),
            Interval(int(rs), int(re_)), int(cid), int(grid), int(block),
            int(stream), str(api)))
    return launches


def validate_submit_markers(db: sqlite3.Connection, submit_times: Sequence[int]) -> dict:
    """Select only queue-completion event records, not all cuEventRecord rows."""
    for table, columns in (
        ("CUPTI_ACTIVITY_KIND_CUDA_EVENT", {"correlationId", "eventSyncId", "streamId"}),
        ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", {"eventSyncId", "syncType"}),
        ("ENUM_CUPTI_SYNC_TYPE", {"id", "name"}),
    ):
        require_columns(db, table, columns)
    rows = db.execute("""
        WITH rt AS (
          SELECT r.*,s.value AS name
          FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        ), synced AS (
          SELECT DISTINCT ce.correlationId,ce.eventId,ce.eventSyncId,ce.streamId
          FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT ce
          JOIN CUPTI_ACTIVITY_KIND_SYNCHRONIZATION sy
            ON sy.eventSyncId=ce.eventSyncId
          JOIN ENUM_CUPTI_SYNC_TYPE st ON st.id=sy.syncType
          WHERE st.name='CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE'
        )
        SELECT rec.start,rec.end,rec.correlationId
        FROM synced JOIN rt rec USING(correlationId)
        JOIN rt cr ON cr.correlationId=rec.correlationId-1
        WHERE rec.name='cuEventRecord' AND cr.name='cuEventCreate'
          AND synced.streamId=13
        ORDER BY rec.start
    """).fetchall()
    if len(rows) != 3607:
        raise CostError(f"expected 3607 filtered submit markers, got {len(rows)}")
    for external, (start, _end, _cid) in enumerate(rows):
        if not (submit_times[external] <= int(start) < submit_times[external + 1]):
            raise CostError(f"filtered submit marker {external} leaves its shim window")
    return {"filtered_queue_completion_markers": len(rows),
            "marker_external_identity_range": [0, len(rows)],
            "filter": "synced cuEventRecord immediately after cuEventCreate on stream 13"}


def classify_vs_launches(launches: Sequence[Launch], submit_times: Sequence[int],
                         device_lo: int, device_hi: int) -> list[tuple[int, int, Launch, Launch | None]]:
    """Return (absolute internal submit, batch ordinal, VS, optional fetch)."""
    # Kernels from independent segment streams interleave in global host issue
    # order.  The exact draw signature is immediate same-stream LEAD/LAG: a VS
    # main is followed by one of the three geometry launches, and an optional
    # standalone fetch immediately precedes it.
    next_same_stream = [None] * len(launches)
    following_by_stream: dict[int, int] = {}
    for index in range(len(launches) - 1, -1, -1):
        launch = launches[index]
        next_same_stream[index] = following_by_stream.get(launch.stream_id)
        following_by_stream[launch.stream_id] = index
    counts: collections.Counter[int] = collections.Counter()
    previous_by_stream: dict[int, int] = {}
    result = []
    for index, launch in enumerate(launches):
        successor = next_same_stream[index]
        previous = previous_by_stream.get(launch.stream_id)
        previous_by_stream[launch.stream_id] = index
        if (launch.name != "main" or successor is None or
                launches[successor].name not in VS_SUCCESSORS):
            continue
        external = bisect.bisect_right(submit_times, launch.runtime.start) - 1
        if external < 0:
            raise CostError(f"VS correlationId {launch.correlation_id} precedes shim timestamps")
        internal = external + 1
        counts[internal] += 1
        fetch = (launches[previous] if previous is not None and
                 launches[previous].name == "cp_vertex_fetch" else None)
        if device_lo <= launch.device.start < device_hi:
            if not (SUBMIT_LO <= internal < SUBMIT_HI):
                raise CostError(
                    f"device/submit alignment mismatch for VS correlationId {launch.correlation_id}: "
                    f"internal submit {internal}")
            result.append((internal, counts[internal], launch, fetch))
    return result


def load_all_activity(db: sqlite3.Connection, lo: int, hi: int) -> list[Interval]:
    result = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET"):
        require_columns(db, table, {"start", "end"})
        for start, end in db.execute(
                f'SELECT start,end FROM "{table}" WHERE end>? AND start<?', (lo, hi)):
            start, end = max(lo, int(start)), min(hi, int(end))
            if end > start:
                result.append(Interval(start, end))
    return result


def parse_selected_log(path: Path, analyzer) -> tuple[list[dict], dict]:
    text = path.read_text()
    # The existing analyzer validates the complete 210k-record run and summary,
    # not merely the selected frame.
    analyzer.analyze(text)
    records = []
    summaries = []
    for line in text.splitlines():
        if line.startswith(RP):
            raw = analyzer.kv(line[len(RP):])
            submit = analyzer.integer(raw, "submit")
            if SUBMIT_LO <= submit < SUBMIT_HI:
                rec = dict(raw)
                for key in raw:
                    if key != "status":
                        rec[key] = analyzer.integer(raw, key)
                records.append(rec)
        elif line.startswith(SP):
            summaries.append(analyzer.kv(line[len(SP):]))
    if len(summaries) != 1:
        raise CostError(f"{path}: expected one summary")
    if analyzer.integer(summaries[0], "final_submit") != EXPECTED_FINAL_INTERNAL:
        raise CostError(f"{path}: final internal submit calibration changed")
    if len(records) != EXPECTED_RECORDS:
        raise CostError(f"{path}: expected {EXPECTED_RECORDS} selected records, got {len(records)}")
    by_submit: collections.Counter[int] = collections.Counter()
    for record in records:
        by_submit[record["submit"]] += 1
        if record["batch"] != by_submit[record["submit"]]:
            raise CostError(
                f"{path}: non-contiguous batch identity at "
                f"({record['submit']},{record['batch']})")
    return records, summaries[0]


def map_records(records: Sequence[dict], units: Sequence[tuple[int, int, Launch, Launch | None]]
                ) -> list[tuple[dict, Launch, Launch | None]]:
    unit_map = {(submit, ordinal): (vs, fetch) for submit, ordinal, vs, fetch in units}
    if len(unit_map) != len(units):
        raise CostError("duplicate clean (internal submit,batch) VS identity")
    if len(unit_map) != EXPECTED_RECORDS:
        raise CostError(f"expected {EXPECTED_RECORDS} classified clean VS rows, got {len(unit_map)}")
    mapped = []
    used = set()
    for record in records:
        key = (record["submit"], record["batch"])
        if key not in unit_map:
            raise CostError(f"census identity {key} has no clean VS")
        vs, fetch = unit_map[key]
        expected_grid = min((record["total"] + 255) // 256, 4096)
        mismatches = []
        if vs.block_x != 256 or vs.grid_x != expected_grid:
            mismatches.append(f"VS grid clean={vs.grid_x} expected={expected_grid}")
        if vs.stream_id != BANK_STREAM[record["bank"]]:
            mismatches.append(
                f"stream clean={vs.stream_id} expected={BANK_STREAM[record['bank']]}")
        if bool(fetch) != bool(record["fetch_runs"]):
            mismatches.append(f"fetch clean={bool(fetch)} record={record['fetch_runs']}")
        if fetch is not None:
            expected_fetch_grid = (record["total"] + 255) // 256
            if (fetch.block_x != 256 or fetch.grid_x != expected_fetch_grid or
                    fetch.stream_id != vs.stream_id):
                mismatches.append(
                    f"fetch shape clean=({fetch.grid_x},{fetch.block_x},{fetch.stream_id}) "
                    f"expected=({expected_fetch_grid},256,{vs.stream_id})")
        if mismatches:
            raise CostError(f"census/trace mismatch for {key}: " + "; ".join(mismatches))
        used.add(key)
        mapped.append((record, vs, fetch))
    if used != set(unit_map):
        raise CostError("clean VS population has identities absent from census")
    return mapped

def sweep_pool(weighted: Sequence[tuple[Interval, Fraction]],
               all_activity: Sequence[Interval]) -> dict:
    all_changes: dict[int, int] = collections.defaultdict(int)
    pool_changes: dict[int, int] = collections.defaultdict(int)
    weight_changes: dict[int, list[tuple[Fraction, int]]] = collections.defaultdict(list)
    for interval in all_activity:
        all_changes[interval.start] += 1
        all_changes[interval.end] -= 1
    raw_weighted = Fraction(0)
    for interval, weight in weighted:
        raw_weighted += interval.duration * weight
        pool_changes[interval.start] += 1
        pool_changes[interval.end] -= 1
        weight_changes[interval.start].append((weight, 1))
        weight_changes[interval.end].append((weight, -1))
    timestamps = sorted(set(all_changes) | set(pool_changes) | set(weight_changes))
    all_active = pool_active = 0
    active_weights: collections.Counter[Fraction] = collections.Counter()
    previous = None
    pool_union = pool_exclusive = 0
    variants = {name: Fraction(0) for name in ("generous_sum_cap", "max", "average", "min")}
    for timestamp in timestamps:
        if previous is not None and pool_active:
            duration = timestamp - previous
            pool_union += duration
            if all_active == pool_active:
                pool_exclusive += duration
                values = list(active_weights.elements())
                if len(values) != pool_active:
                    raise CostError(f"pool weight multiplicity mismatch at {timestamp}")
                total = sum(values, Fraction(0))
                variants["generous_sum_cap"] += duration * min(Fraction(1), total)
                variants["max"] += duration * max(values)
                variants["average"] += duration * total / len(values)
                variants["min"] += duration * min(values)
        all_active += all_changes[timestamp]
        pool_active += pool_changes[timestamp]
        for weight, delta in weight_changes[timestamp]:
            active_weights[weight] += delta
            if active_weights[weight] < 0:
                raise CostError(f"negative pool weight count at {timestamp}")
            if not active_weights[weight]:
                del active_weights[weight]
        if all_active < 0 or pool_active < 0 or pool_active > all_active:
            raise CostError(f"activity sweep invariant failed at {timestamp}")
        previous = timestamp
    if all_active or pool_active or active_weights:
        raise CostError("activity sweep ended active")
    scale = FRAME_COUNT * 1_000_000
    variant_output = {
        name: {"ns_total": float(value),
               "ms_per_frame": float(value / scale)}
        for name, value in variants.items()
    }
    exact_parts = [
        str(sum(interval.duration for interval, _ in weighted)),
        str(pool_union), str(pool_exclusive),
        f"{raw_weighted.numerator}/{raw_weighted.denominator}",
    ] + [
        f"{variants[name].numerator}/{variants[name].denominator}"
        for name in ("generous_sum_cap", "max", "average", "min")
    ]
    fingerprint = hashlib.sha256("\n".join(exact_parts).encode()).hexdigest()
    return {
        "pool_sum_ns": sum(interval.duration for interval, _ in weighted),
        "pool_union_ns": pool_union,
        "pool_union_exclusive_ns": pool_exclusive,
        "pool_sum_ms_per_frame": sum(interval.duration for interval, _ in weighted) / scale,
        "pool_union_ms_per_frame": pool_union / scale,
        "pool_union_exclusive_ms_per_frame": pool_exclusive / scale,
        "raw_duplicate_weighted_sum_ns": float(raw_weighted),
        "raw_duplicate_weighted_sum_ms_per_frame": float(raw_weighted / scale),
        "collectible_union_exclusive": variant_output,
        "exact_calibration_sha256": fingerprint,
        "decision_metric": "collectible_union_exclusive.generous_sum_cap.ms_per_frame",
        "decision_scope": (
            "generous clean-trace upper: min(1,sum(active duplicate fractions)) "
            "on spans where only VS+fetch pool activity is present"),
    }

def run(trace: Path, submits: Path, logs: Sequence[Path]) -> dict:
    if len(logs) != 2:
        raise CostError("exact calibration requires run1 and run2")
    submit_absolute, torn = parse_submit_timestamps(submits)
    analyzer = _load_analyzer(Path(__file__))
    records1, summary1 = parse_selected_log(logs[0], analyzer)
    records2, summary2 = parse_selected_log(logs[1], analyzer)
    comparison_keys = set(records1[0]) - {"probes"}
    for index, (left, right) in enumerate(zip(records1, records2)):
        if any(left[key] != right[key] for key in comparison_keys):
            raise CostError(f"run1/run2 selected structural record {index} differs")

    with sqlite3.connect(trace) as db:
        clock = session_clock(db)
        relative = [value - clock for value in submit_absolute]
        if relative[0] < 0:
            raise CostError("external timestamps precede nsys clock origin")
        frame_starts = relative[TRACE_START_EXTERNAL::SUBMITS_PER_FRAME]
        if len(frame_starts) < FRAME_COUNT + 2:
            raise CostError("submit timestamps do not cover calibrated frame window")
        frame_lo, frame_hi = frame_starts[1], frame_starts[FRAME_COUNT + 1]
        launches = load_launches(db)
        activity = load_all_activity(db, frame_lo, frame_hi)
        nvtx = nvtx_state(db)
        markers = validate_submit_markers(db, relative)

    units = classify_vs_launches(launches, relative, frame_lo, frame_hi)
    mapped = map_records(records1, units)
    selected_fetches = sum(fetch is not None for _, _, _, fetch in units)
    if selected_fetches != EXPECTED_FETCH_ROWS:
        raise CostError(
            f"expected {EXPECTED_FETCH_ROWS} selected fetch rows, got {selected_fetches}")

    weighted = []
    batches = []
    complete = excluded = 0
    for record, vs, fetch in mapped:
        source_intervals = [vs.device] + ([] if fetch is None else [fetch.device])
        intervals = []
        for interval in source_intervals:
            start, end = max(frame_lo, interval.start), min(frame_hi, interval.end)
            if end <= start:
                raise CostError(
                    f"identity ({record['submit']},{record['batch']}) has a pool launch "
                    f"outside the device window")
            intervals.append(Interval(start, end))
        admitted = record["status"] == "complete" and record["total"] > 0
        if admitted:
            complete += 1
            fraction = Fraction(record["duplicate"], record["total"])
        else:
            excluded += 1
            fraction = Fraction(0)
        for interval in intervals:
            weighted.append((interval, fraction))
        raw = sum(interval.duration for interval in intervals) * fraction
        batches.append({
            "identity": {"internal_submit": record["submit"],
                         "executed_batch": record["batch"]},
            "status": record["status"], "total": record["total"],
            "duplicate": record["duplicate"],
            "duplicate_fraction": float(fraction),
            "raw_duplicate_weighted_ns": float(raw),
            "vs": {"correlation_id": vs.correlation_id, "stream_id": vs.stream_id,
                   "grid_x": vs.grid_x, "runtime_start_ns": vs.runtime.start,
                   "device_start_ns": vs.device.start, "device_end_ns": vs.device.end,
                   "duration_ns": vs.device.duration},
            "fetch": None if fetch is None else {
                   "correlation_id": fetch.correlation_id, "stream_id": fetch.stream_id,
                   "grid_x": fetch.grid_x, "runtime_start_ns": fetch.runtime.start,
                   "device_start_ns": fetch.device.start, "device_end_ns": fetch.device.end,
                   "duration_ns": fetch.device.duration},
        })
    cost = sweep_pool(weighted, activity)
    calibration = {key: cost[key] for key in
        ("pool_sum_ns", "pool_union_ns", "pool_union_exclusive_ns")}
    if calibration != EXPECTED_POOL_CALIBRATION:
        raise CostError(
            f"pool calibration mismatch: expected {EXPECTED_POOL_CALIBRATION}, got {calibration}")
    if cost["exact_calibration_sha256"] != EXPECTED_COST_FINGERPRINT:
        raise CostError(
            "weighted/exclusive exact calibration fingerprint mismatch: "
            f"expected {EXPECTED_COST_FINGERPRINT}, got {cost['exact_calibration_sha256']}")
    decision = cost["collectible_union_exclusive"]["generous_sum_cap"]["ms_per_frame"]
    return {
        "result_kind": "vertex_reuse_clean_trace_cost",
        "performance_result": False,
        "trace": str(trace), "submit_timestamps": str(submits),
        "logs": [str(path) for path in logs],
        "ignored_incomplete_trailing_submit_fragment": torn,
        "alignment": {
            "nsys_system_clock_ns": clock,
            "frame_origin_internal": FRAME_ORIGIN,
            "frame_count": FRAME_COUNT,
            "frame_device_window_ns": [frame_lo, frame_hi],
            "internal_submit_range": [SUBMIT_LO, SUBMIT_HI],
            "range_semantics": "half-open",
            "final_internal_submit": EXPECTED_FINAL_INTERNAL,
            "complete_external_rows": EXPECTED_EXTERNAL_ROWS,
            "external_trace_start_index": TRACE_START_EXTERNAL,
            "note": (
                "runtime issue timestamps map launches to external shim windows, then +1 "
                "to absolute command-bearing internal submit; no process-wide final-count "
                "offset is assumed"),
        },
        "runtime_marker_calibration": markers,
        "nvtx": nvtx,
        "mapping": {
            "absolute_identity": "(internal submit, executed batch ordinal)",
            "classifier": (
                "main whose immediate same-stream LEAD kernel in filtered host launch order is one of "
                + str(sorted(VS_SUCCESSORS))),
            "runtime_filter": (
                "kernel rows correlated to cuLaunchKernel/cuLaunchKernelEx only; "
                "non-kernel APIs and NVTX marks excluded"),
            "bank_to_clean_stream": BANK_STREAM,
            "classified_vs_rows": len(units),
            "mapped_batches": len(batches),
            "mapped_fetch_rows": selected_fetches,
            "complete_weighted_batches": complete,
            "excluded_or_unknown_zero_weight_batches": excluded,
            "run1_run2_selected_structural_records_equal": True,
            "run_varying_diagnostic_field_ignored": "probes",
            "grid_stream_fetch_checks": "all passed",
        },
        "cost": cost,
        "gate": {
            "metric": "cost.collectible_union_exclusive.generous_sum_cap.ms_per_frame",
            "threshold_ms_per_frame": 0.5,
            "value_ms_per_frame": decision,
            "rejection_condition": "<=",
            "rejection_condition_holds": decision <= 0.5,
            "proposal_survives": decision > 0.5,
        },
        "pool_calibration": calibration,
        "pool_calibration_expected": EXPECTED_POOL_CALIBRATION,
        "excluded_or_unknown_policy": "zero collectible weight",
        "batches": batches,
    }

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="run1/stderr run2/stderr")
    parser.add_argument("--trace", type=Path, default=Path("/tmp/postdead/trace.sqlite"))
    parser.add_argument("--submit-timestamps", type=Path,
                        default=Path("/tmp/postdead/submit-ts.txt"))
    args = parser.parse_args(argv)
    try:
        result = run(args.trace, args.submit_timestamps, args.logs)
    except (CostError, OSError, sqlite3.Error, ValueError) as error:
        print(f"cp_vertex_reuse_cost.py: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
