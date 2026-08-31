#!/usr/bin/env python3
"""Join B0 classifications to a clean Nsight Systems trace.

This is a CPU-only diagnostic, not a performance result.  The project decision
metric is union-exclusive collectible duration in this clean trace.  It is not
an unconstrained mathematical upper under arbitrary resource contention.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import importlib.util
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

FRAME_ORIGIN = 2784
FRAME_COUNT = 411
SUBMITS_PER_FRAME = 2
TRACE_START_EXTERNAL = FRAME_ORIGIN - 1
GATE_MS_PER_FRAME = 0.5
NAMES = {
    "clip": "cp_clip_triangles",
    "fused": "cp_clip_rast_fused",
    "stage1": "cp_rasterize_stage1",
    "stage2": "cp_rasterize_stage2",
}
EXPECTED_CALIBRATION = {
    "sum_ms_per_frame": 1.78632851095,
    "union_ms_per_frame": 0.89114590024,
    "union_exclusive_ms_per_frame": 0.73687845012,
}
CALIBRATION_TOLERANCE_MS = 5e-11


class CostError(ValueError):
    pass


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise CostError(f"invalid interval [{self.start},{self.end})")


@dataclass(frozen=True)
class Launch:
    name: str
    device: Interval
    runtime: Interval
    correlation_id: int
    grid_x: int


@dataclass(frozen=True)
class Unit:
    submit: int
    ordinal: int
    predecessor: Launch
    stage2: Launch
    clip: Launch | None


def _load_analyzer():
    path = Path(__file__).with_name("cp_geometry_b0_analyze.py")
    spec = importlib.util.spec_from_file_location("cp_geometry_b0_analyze", path)
    if spec is None or spec.loader is None:
        raise CostError(f"cannot load analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_submit_timestamps(path: Path) -> tuple[list[int], str | None]:
    """Read '<CLOCK_MONOTONIC ns> <zero-based index>' rows.

    A killed producer can leave one incomplete final row.  That final fragment
    is reported and ignored.  Any malformed interior row is fatal.
    """
    text = path.read_text()
    lines = text.splitlines()
    trailing_fragment = None
    rows: list[tuple[int, int, int]] = []
    row_re = re.compile(r"\s*([0-9]+)\s+([0-9]+)\s*")
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        match = row_re.fullmatch(line)
        if match is None:
            is_last_nonempty = not any(x.strip() for x in lines[line_no:])
            if is_last_nonempty and re.fullmatch(r"\s*[0-9]+\s*", line):
                trailing_fragment = line.strip()
                continue
            raise CostError(f"{path}:{line_no}: malformed submit timestamp row: {line!r}")
        rows.append((int(match.group(1)), int(match.group(2)), line_no))
    if not rows:
        raise CostError(f"{path}: no complete submit timestamp rows")
    for expected, (timestamp, index, line_no) in enumerate(rows):
        if index != expected:
            raise CostError(
                f"{path}:{line_no}: submit index mismatch: expected {expected}, got {index}"
            )
        if expected and timestamp <= rows[expected - 1][0]:
            raise CostError(
                f"{path}:{line_no}: submit timestamp is not strictly increasing: "
                f"previous={rows[expected - 1][0]} current={timestamp}"
            )
    return [row[0] for row in rows], trailing_fragment


def require_columns(db: sqlite3.Connection, table: str, required: set[str]) -> None:
    found = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
    if not found:
        raise CostError(f"trace is missing table {table}")
    missing = required - found
    if missing:
        raise CostError(f"trace table {table} is missing columns {sorted(missing)}")


def session_system_clock_ns(db: sqlite3.Connection) -> int:
    require_columns(db, "TARGET_INFO_SESSION_START_TIME", {"systemClockNs"})
    values = [row[0] for row in db.execute(
        "SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
    )]
    if len(values) != 1 or values[0] is None:
        raise CostError(
            "TARGET_INFO_SESSION_START_TIME must contain exactly one non-null systemClockNs"
        )
    return int(values[0])


def load_target_launches(db: sqlite3.Connection) -> list[Launch]:
    require_columns(db, "StringIds", {"id", "value"})
    require_columns(db, "CUPTI_ACTIVITY_KIND_KERNEL", {
        "start", "end", "correlationId", "shortName", "gridX"
    })
    require_columns(db, "CUPTI_ACTIVITY_KIND_RUNTIME", {
        "start", "end", "correlationId"
    })
    placeholders = ",".join("?" for _ in NAMES)
    rows = db.execute(f"""
        SELECT k.start, k.end, k.gridX, k.correlationId, strings.value,
               runtime.start, runtime.end
          FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
          JOIN StringIds AS strings ON strings.id = k.shortName
          LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            ON runtime.correlationId = k.correlationId
         WHERE strings.value IN ({placeholders})
         ORDER BY runtime.start, k.start, k.correlationId
    """, tuple(NAMES.values())).fetchall()
    launches = []
    correlations = set()
    for ds, de, grid_x, correlation, name, rs, re_ in rows:
        if correlation is None:
            raise CostError(f"target kernel {name} has null correlationId")
        if rs is None or re_ is None:
            raise CostError(
                f"target kernel {name} correlationId {correlation} has no runtime launch"
            )
        if correlation in correlations:
            raise CostError(f"target correlationId {correlation} maps more than once")
        correlations.add(correlation)
        launches.append(Launch(str(name), Interval(int(ds), int(de)),
                               Interval(int(rs), int(re_)), int(correlation), int(grid_x)))
    return launches


def load_all_activity(db: sqlite3.Connection, lo: int, hi: int) -> list[Interval]:
    out: list[Interval] = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET"):
        require_columns(db, table, {"start", "end"})
        for start, end in db.execute(
            f'SELECT start,end FROM "{table}" WHERE end>? AND start<?', (lo, hi)
        ):
            start, end = max(int(start), lo), min(int(end), hi)
            if end > start:
                out.append(Interval(start, end))
    return out


def direct_units(launches: Sequence[Launch], submit_times: Sequence[int]) -> list[Unit]:
    """Recognize direct units in filtered host launch order."""
    by_submit: collections.Counter[int] = collections.Counter()
    units: list[Unit] = []
    predecessor_names = {NAMES["fused"], NAMES["stage1"]}
    for i, launch in enumerate(launches):
        if launch.name in predecessor_names:
            following = launches[i + 1].name if i + 1 < len(launches) else "<end>"
            if following != NAMES["stage2"]:
                raise CostError(
                    f"filtered launch order mismatch: predecessor {launch.name} "
                    f"correlationId {launch.correlation_id} is followed by {following}, "
                    f"not {NAMES['stage2']}"
                )
        if launch.name != NAMES["stage2"]:
            continue
        previous = launches[i - 1].name if i else "<start>"
        if previous not in predecessor_names:
            raise CostError(
                f"filtered launch order mismatch: {NAMES['stage2']} correlationId "
                f"{launch.correlation_id} is preceded by {previous}, not "
                f"{NAMES['stage1']} or {NAMES['fused']}"
            )
        predecessor = launches[i - 1]
        external = bisect.bisect_right(submit_times, launch.runtime.start) - 1
        if external < 0:
            raise CostError(
                f"stage2 correlationId {launch.correlation_id} runtime start "
                f"{launch.runtime.start} precedes first submit timestamp {submit_times[0]}"
            )
        internal = external + 1
        by_submit[internal] += 1
        clip = launches[i - 2] if i >= 2 and launches[i - 2].name == NAMES["clip"] else None
        units.append(Unit(internal, by_submit[internal], predecessor, launch, clip))
    return units


def clipped(interval: Interval, lo: int, hi: int) -> Interval | None:
    start, end = max(interval.start, lo), min(interval.end, hi)
    return Interval(start, end) if end > start else None


def intervals_for_units(units: Iterable[Unit], lo: int, hi: int) -> list[Interval]:
    out = []
    for unit in units:
        for launch in (unit.predecessor, unit.stage2):
            interval = clipped(launch.device, lo, hi)
            if interval is None:
                raise CostError(
                    f"selected unit ({unit.submit},{unit.ordinal}) has {launch.name} "
                    f"outside device window [{lo},{hi})"
                )
            out.append(interval)
    return out


def sweep_cost(target: Sequence[Interval], all_activity: Sequence[Interval]) -> dict[str, int]:
    """Return sum, union, and union-exclusive durations in integer ns."""
    changes: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    total_sum = 0
    for interval in all_activity:
        changes[interval.start][0] += 1
        changes[interval.end][0] -= 1
    for interval in target:
        total_sum += interval.end - interval.start
        changes[interval.start][1] += 1
        changes[interval.end][1] -= 1
    all_active = target_active = 0
    previous = None
    union = exclusive = 0
    for timestamp in sorted(changes):
        if previous is not None and target_active:
            duration = timestamp - previous
            union += duration
            if all_active == target_active:
                exclusive += duration
        delta_all, delta_target = changes[timestamp]
        all_active += delta_all
        target_active += delta_target
        if all_active < 0 or target_active < 0 or target_active > all_active:
            raise CostError(
                f"sweep invariant failed at {timestamp}: "
                f"all_active={all_active} target_active={target_active}"
            )
        previous = timestamp
    if all_active or target_active:
        raise CostError(
            f"sweep ended active: all_active={all_active} target_active={target_active}"
        )
    return {"sum_ns": total_sum, "union_ns": union,
            "union_exclusive_ns": exclusive}


def per_frame(cost: dict[str, int], frame_count: int) -> dict[str, float]:
    if frame_count <= 0:
        raise CostError("frame count must be positive")
    scale = frame_count * 1_000_000.0
    return {
        "sum_ms_per_frame": cost["sum_ns"] / scale,
        "union_ms_per_frame": cost["union_ns"] / scale,
        "union_exclusive_ms_per_frame": cost["union_exclusive_ns"] / scale,
    }


def validate_calibration(actual: dict[str, float]) -> None:
    for key, expected in EXPECTED_CALIBRATION.items():
        got = actual[key]
        if not math.isclose(got, expected, rel_tol=0.0,
                            abs_tol=CALIBRATION_TOLERANCE_MS):
            raise CostError(
                f"full direct calibration mismatch for {key}: "
                f"expected={expected:.14f} actual={got:.14f} "
                f"delta={got - expected:+.14f} "
                f"tolerance={CALIBRATION_TOLERANCE_MS:.1e}"
            )


def _predecessor_code(unit: Unit) -> int:
    return 2 if unit.predecessor.name == NAMES["fused"] else 1


def map_admitted_records(records: Sequence[dict], units: Sequence[Unit],
                         submit_lo: int, submit_hi: int) -> dict[tuple[int, int], dict]:
    unit_map = {(u.submit, u.ordinal): u for u in units}
    if len(unit_map) != len(units):
        raise CostError("duplicate (submit,unit) in selected trace units")
    mapped: dict[tuple[int, int], dict] = {}
    for record in records:
        if record["status"] != 1 or not (submit_lo <= record["submit"] < submit_hi):
            continue
        key = (record["submit"], record["unit"])
        unit = unit_map.get(key)
        if unit is None:
            raise CostError(
                f"B0 admitted record index={record['index']} has no trace unit: "
                f"submit={record['submit']} unit={record['unit']} "
                f"trace_submit_range=[{submit_lo},{submit_hi})"
            )
        expected_predecessor = _predecessor_code(unit)
        mismatches = []
        if record["predecessor"] != expected_predecessor:
            mismatches.append(
                f"predecessor record={record['predecessor']} "
                f"trace={expected_predecessor}({unit.predecessor.name})"
            )
        if record["stage2_blocks"] != unit.stage2.grid_x:
            mismatches.append(
                f"stage2_blocks record={record['stage2_blocks']} "
                f"trace={unit.stage2.grid_x}"
            )
        if mismatches:
            raise CostError(
                f"B0/trace mismatch index={record['index']} "
                f"submit={key[0]} unit={key[1]}: " + "; ".join(mismatches)
            )
        if key in mapped:
            raise CostError(
                f"duplicate admitted B0 key submit={key[0]} unit={key[1]}: "
                f"indices {mapped[key]['index']} and {record['index']}"
            )
        mapped[key] = record
    return mapped


def classification_map(analysis: dict, admitted: dict[tuple[int, int], dict]) -> dict[tuple[int, int], bool]:
    out: dict[tuple[int, int], bool] = {}
    for submit, occurrence, unit, pass_number, matched in analysis["membership_classification"]:
        key = (submit, unit)
        record = admitted.get(key)
        if record is None:
            continue
        if occurrence != record["occurrence"] or pass_number != record["pass"]:
            raise CostError(
                f"classification/B0 mismatch submit={submit} unit={unit}: "
                f"occurrence classification={occurrence} record={record['occurrence']}; "
                f"pass classification={pass_number} record={record['pass']}"
            )
        if key in out:
            raise CostError(f"duplicate classification submit={submit} unit={unit}")
        out[key] = bool(matched)
    return out


def prepare_log(path: Path, analyzer, frame_origin: int) -> dict:
    records, summary = analyzer.parse_run(path)
    analysis = analyzer.analyze(records, frame_origin, SUBMITS_PER_FRAME,
                                mode=summary["mode"])
    analysis.update(salt=f"{summary['salt_value']:016x}",
                    salt2=f"{summary['salt2_value']:016x}")
    return {"path": path, "records": records, "summary": summary,
            "analysis": analysis}


def classification_arm(prepared: dict, analysis: dict, units: Sequence[Unit],
                       clips: Sequence[Interval],
                       all_activity: Sequence[Interval], lo: int, hi: int,
                       submit_lo: int, submit_hi: int, frame_count: int,
                       arm_kind: str, decision_bearing: bool,
                       source_logs: Sequence[Path]) -> dict:
    path = prepared["path"]
    records = prepared["records"]
    summary = prepared["summary"]
    admitted = map_admitted_records(records, units, submit_lo, submit_hi)
    classes = classification_map(analysis, admitted)

    recurrent_units = []
    possible_units = []
    matched = unmatched = unknown = excluded = 0
    for unit in units:
        key = (unit.submit, unit.ordinal)
        if key not in admitted:
            excluded += 1
            recurrent_units.append(unit)
        elif key not in classes:
            unknown += 1
            recurrent_units.append(unit)
        elif classes[key]:
            matched += 1
            recurrent_units.append(unit)
        else:
            unmatched += 1
            possible_units.append(unit)

    recurrent_intervals = list(clips) + intervals_for_units(recurrent_units, lo, hi)
    nonrecurrent_intervals = intervals_for_units(possible_units, lo, hi)
    recurrent_cost = per_frame(sweep_cost(recurrent_intervals, all_activity), frame_count)
    nonrecurrent_cost = per_frame(
        sweep_cost(nonrecurrent_intervals, all_activity), frame_count
    )
    gate_value = recurrent_cost["union_exclusive_ms_per_frame"]
    rejection_condition_holds = gate_value <= GATE_MS_PER_FRAME
    return {
        "arm": arm_kind,
        "decision_bearing": decision_bearing,
        "log": str(path),
        "reference_log": str(path),
        "source_logs": [str(source) for source in source_logs],
        "salt": f"{summary['salt_value']:016x}",
        "selected_units": len(units),
        "admitted_mapped": len(admitted),
        "excluded_or_unadmitted_conservative_recurrent": excluded,
        "classification_matched_recurrent": matched,
        "classification_unmatched_nonrecurrent": unmatched,
        "classification_missing_conservative_recurrent": unknown,
        "all_clip_intervals_conservative_recurrent": len(clips),
        "possible_saving_necessary_condition_upper": recurrent_cost,
        "possible_saving_necessary_condition_upper_scope": (
            "clean-trace target duration only; union_exclusive_ms_per_frame is "
            "the project collectible metric, not an unconstrained upper under "
            "arbitrary resource contention"
        ),
        "classified_nonrecurrent_diagnostic_trace_cost": nonrecurrent_cost,
        "gate": {
            "decision_bearing": decision_bearing,
            "metric": (
                "possible_saving_necessary_condition_upper."
                "union_exclusive_ms_per_frame"
            ),
            "metric_scope": (
                "project clean-trace collectible duration; not an unconstrained "
                "mathematical upper under arbitrary resource contention"
            ),
            "rejection_condition": "<=",
            "threshold_ms_per_frame": GATE_MS_PER_FRAME,
            "value_ms_per_frame": gate_value,
            "rejection_condition_holds": rejection_condition_holds,
            "proposal_survives": not rejection_condition_holds,
        },
    }


def analyze_arm(path: Path, analyzer, units: Sequence[Unit], clips: Sequence[Interval],
                all_activity: Sequence[Interval], lo: int, hi: int,
                submit_lo: int, submit_hi: int, frame_origin: int,
                frame_count: int) -> dict:
    prepared = prepare_log(path, analyzer, frame_origin)
    return classification_arm(prepared, prepared["analysis"], units, clips,
                              all_activity, lo, hi, submit_lo, submit_hi,
                              frame_count, "single_run", True, [path])


def aggregate_arms(prepared_logs: Sequence[dict], analyzer,
                   units: Sequence[Unit], clips: Sequence[Interval],
                   all_activity: Sequence[Interval], lo: int, hi: int,
                   submit_lo: int, submit_hi: int, frame_count: int
                   ) -> tuple[list[dict], dict]:
    analyses = [prepared["analysis"] for prepared in prepared_logs]
    modes = [prepared["summary"]["mode"] for prepared in prepared_logs]
    repeated_mode3 = len(prepared_logs) > 1 and all(mode == 3 for mode in modes)
    if len(analyses) > 1:
        try:
            salt_rerun = analyzer.validate_salt_reruns(analyses)
        except analyzer.CensusError as error:
            if 3 in modes:
                raise CostError(f"invalid mode 3 reruns: {error}") from error
            # Old single-domain runs are independent diagnostic arms.  Preserve
            # their costs when their classifications differ, but expose the
            # failed rerun validation rather than treating them as one decision.
            salt_rerun = {"checked": True, "valid": False,
                          "validation_error": str(error)}
    else:
        salt_rerun = {"checked": False}
    if repeated_mode3:
        if not all(analysis.get("paired_domain_checked") is True and
                   analysis.get("paired_domain_equal") is True
                   for analysis in analyses):
            raise CostError("repeated mode 3 logs lack paired-domain validity")
        if not (salt_rerun.get("paired_mode3_repeats") is True and
                salt_rerun.get("population_equal") is True):
            raise CostError("repeated mode 3 logs lack equal structural populations")

    arms = [classification_arm(
                prepared, prepared["analysis"], units, clips, all_activity,
                lo, hi, submit_lo, submit_hi, frame_count,
                "per_run_diagnostic" if repeated_mode3 else "single_run",
                not repeated_mode3, [prepared["path"]])
            for prepared in prepared_logs]
    if repeated_mode3:
        union_analysis = dict(analyses[0])
        union_analysis["membership_classification"] = (
            salt_rerun["union_recurrent_classification"]
        )
        arms.append(classification_arm(
            prepared_logs[0], union_analysis, units, clips, all_activity,
            lo, hi, submit_lo, submit_hi, frame_count,
            "conservative_repeat_union", True,
            [prepared["path"] for prepared in prepared_logs]))
    return arms, salt_rerun


def run(trace: Path, submits_path: Path, logs: Sequence[Path], frame_origin: int,
        frame_count: int) -> dict:
    if frame_origin != FRAME_ORIGIN or frame_count != FRAME_COUNT:
        raise CostError(
            f"this calibrated join requires frame_origin={FRAME_ORIGIN} and "
            f"frame_count={FRAME_COUNT}; got {frame_origin} and {frame_count}"
        )
    submit_absolute, trailing_fragment = parse_submit_timestamps(submits_path)
    with sqlite3.connect(trace) as db:
        clock = session_system_clock_ns(db)
        submit_times = [timestamp - clock for timestamp in submit_absolute]
        if submit_times[0] < 0:
            raise CostError(
                f"first submit timestamp {submit_absolute[0]} precedes trace "
                f"systemClockNs {clock}"
            )
        frame_starts = submit_times[TRACE_START_EXTERNAL::SUBMITS_PER_FRAME]
        needed = frame_count + 2
        if len(frame_starts) < needed:
            raise CostError(
                f"not enough submit timestamps for frames 1..{frame_count}: "
                f"need {needed} starts from external index {TRACE_START_EXTERNAL}, "
                f"got {len(frame_starts)}"
            )
        lo, hi = frame_starts[1], frame_starts[frame_count + 1]
        submit_lo = TRACE_START_EXTERNAL + SUBMITS_PER_FRAME + 1
        submit_hi = TRACE_START_EXTERNAL + (frame_count + 1) * SUBMITS_PER_FRAME + 1
        launches = load_target_launches(db)
        all_activity = load_all_activity(db, lo, hi)

    units_all = direct_units(launches, submit_times)
    units = [unit for unit in units_all if lo <= unit.stage2.device.start < hi]
    outside = [unit for unit in units if unit.stage2.device.end > hi]
    if outside:
        unit = outside[0]
        raise CostError(
            f"stage2 crosses frame window end: submit={unit.submit} unit={unit.ordinal} "
            f"interval=[{unit.stage2.device.start},{unit.stage2.device.end}) hi={hi}"
        )
    for unit in units:
        if not (submit_lo <= unit.submit < submit_hi):
            raise CostError(
                f"device/submit alignment mismatch for unit ({unit.submit},{unit.ordinal}): "
                f"expected submit in [{submit_lo},{submit_hi})"
            )
    clip_intervals = []
    for launch in launches:
        if launch.name == NAMES["clip"]:
            interval = clipped(launch.device, lo, hi)
            if interval is not None:
                clip_intervals.append(interval)

    full_intervals = list(clip_intervals) + intervals_for_units(units, lo, hi)
    calibration = per_frame(sweep_cost(full_intervals, all_activity), frame_count)
    validate_calibration(calibration)

    analyzer = _load_analyzer()
    prepared_logs = [prepare_log(path, analyzer, frame_origin) for path in logs]
    arms, salt_rerun = aggregate_arms(
        prepared_logs, analyzer, units, clip_intervals, all_activity,
        lo, hi, submit_lo, submit_hi, frame_count)
    return {
        "result_kind": "diagnostic_trace_cost",
        "performance_result": False,
        "interpretation": (
            "These are costs from one external clean trace, not an application "
            "performance result. The project decision metric is union-exclusive "
            "collectible duration in that clean trace; it is not an unconstrained "
            "mathematical upper under arbitrary resource contention. The sum is "
            "also reported as the broader sum of selected target durations."
        ),
        "trace": str(trace),
        "submit_timestamps": str(submits_path),
        "ignored_incomplete_trailing_submit_fragment": trailing_fragment,
        "alignment": {
            "nsys_system_clock_ns": clock,
            "frame_origin_internal": frame_origin,
            "frame_count": frame_count,
            "frame_lo_trace_ns": lo,
            "frame_hi_trace_ns": hi,
            "internal_submit_range": [submit_lo, submit_hi],
            "range_semantics": "half-open",
        },
        "mapping": {
            "selected_direct_units": len(units),
            "all_clip_intervals": len(clip_intervals),
        },
        "full_direct_calibration": calibration,
        "calibration_expected": EXPECTED_CALIBRATION,
        "calibration_tolerance_ms": CALIBRATION_TOLERANCE_MS,
        "salt_rerun": salt_rerun,
        "arms": arms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--trace", type=Path, default=Path("/tmp/postdead/trace.sqlite"))
    parser.add_argument("--submit-timestamps", type=Path,
                        default=Path("/tmp/postdead/submit-ts.txt"))
    parser.add_argument("--frame-origin", type=int, default=FRAME_ORIGIN)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    args = parser.parse_args(argv)
    try:
        result = run(args.trace, args.submit_timestamps, args.logs,
                     args.frame_origin, args.frame_count)
    except (CostError, sqlite3.Error, OSError, ValueError) as error:
        print(f"cp_geometry_b0_cost.py: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
