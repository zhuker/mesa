#!/usr/bin/env python3
"""Validate and summarize a CUDAVK submit-completion census.

The census is a diagnostic.  It measures when the completion worker observes
CUDA completion, not presentation time.  A target window is never guessed:
the caller names its first internal record and its number of frames.

Input may contain unrelated stdout/stderr lines.  Census lines have these
forms (field order does not matter)::

  CUDAVK_SUBMIT_CENSUS_RECORD index=0 internal_ordinal=0 initialized=1 \
      issue_begin_ns=100 issue_end_ns=110 complete_ns=200 completion=1 \
      status=0 command_buffer_count=1 wait_count=0 signal_count=1 marker=0 \
      queued_depth=3 depth_capped=0
  CUDAVK_SUBMIT_CENSUS_SUMMARY result_valid=yes capacity=131072 \
      reserved=1 complete=1 overflow=0

An optional external shim file accepts either ``timestamp_ns submit_index`` or
one key/value record per line::

  CUDAVK_SUBMIT_SHIM_TIMESTAMP timestamp_ns=123 submit_index=7

Its target start is also explicit.  The shim is useful for enqueue-rate
comparison, but is not completion timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

RECORD_PREFIX = "CUDAVK_SUBMIT_CENSUS_RECORD"
SUMMARY_PREFIX = "CUDAVK_SUBMIT_CENSUS_SUMMARY"
SUBMITS_PER_FRAME = 2


class CensusError(ValueError):
    """A census cannot support the requested diagnostic."""


@dataclass(frozen=True)
class Record:
    index: int
    internal_ordinal: int
    initialized: bool
    issue_begin_ns: int
    issue_end_ns: int
    complete_ns: int
    completed: bool
    status: int
    command_buffer_count: int
    wait_count: int
    signal_count: int
    marker: bool
    queued_depth: int
    depth_capped: bool


@dataclass(frozen=True)
class Summary:
    result_valid: bool
    capacity: int
    reserved: int
    complete: int
    overflow: int


@dataclass(frozen=True)
class ShimTimestamp:
    timestamp_ns: int
    submit_index: int


def _fields(line: str, prefix: str, line_number: int) -> dict[str, str]:
    rest = line[len(prefix):].strip()
    if not rest:
        raise CensusError(f"line {line_number}: {prefix} has no fields")
    out: dict[str, str] = {}
    for token in rest.split():
        if token.count("=") != 1:
            raise CensusError(f"line {line_number}: malformed field {token!r}")
        name, value = token.split("=", 1)
        if not name or not value:
            raise CensusError(f"line {line_number}: malformed field {token!r}")
        if name in out:
            raise CensusError(f"line {line_number}: duplicate field {name!r}")
        out[name] = value
    return out


def _take(fields: dict[str, str], names: Sequence[str], line_number: int) -> str:
    present = [name for name in names if name in fields]
    if not present:
        raise CensusError(f"line {line_number}: missing field {names[0]!r}")
    if len(present) != 1:
        raise CensusError(
            f"line {line_number}: aliases supplied together: {', '.join(present)}"
        )
    return fields.pop(present[0])


def _uint(text: str, name: str, line_number: int) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise CensusError(f"line {line_number}: {name} is not an integer") from exc
    if value < 0:
        raise CensusError(f"line {line_number}: {name} is negative")
    return value


def _status(text: str, line_number: int) -> int:
    if text in ("CUDA_SUCCESS", "success", "ok"):
        return 0
    try:
        return int(text, 10)
    except ValueError as exc:
        raise CensusError(f"line {line_number}: status is not an integer") from exc


def _bool(text: str, name: str, line_number: int) -> bool:
    if text in ("1", "yes", "true"):
        return True
    if text in ("0", "no", "false"):
        return False
    raise CensusError(f"line {line_number}: {name} is not a boolean")


def _no_unknown(fields: dict[str, str], line_number: int) -> None:
    if fields:
        raise CensusError(
            f"line {line_number}: unknown field(s): {', '.join(sorted(fields))}"
        )


def _parse_record(line: str, line_number: int) -> Record:
    f = _fields(line, RECORD_PREFIX, line_number)
    index = _uint(_take(f, ("index",), line_number), "index", line_number)
    internal_ordinal = _uint(_take(f, ("internal_ordinal",), line_number),
                             "internal_ordinal", line_number)
    initialized = _bool(_take(f, ("initialized",), line_number),
                        "initialized", line_number)
    issue_begin_ns = _uint(_take(f, ("issue_begin_ns",), line_number),
                           "issue_begin_ns", line_number)
    issue_end_ns = _uint(_take(f, ("issue_end_ns",), line_number),
                         "issue_end_ns", line_number)
    complete_ns = _uint(_take(f, ("complete_ns",), line_number),
                        "complete_ns", line_number)
    completed = _bool(_take(f, ("completion",), line_number),
                      "completion", line_number)
    status = _status(_take(f, ("status",), line_number), line_number)
    command_buffer_count = _uint(
        _take(f, ("command_buffer_count",), line_number),
        "command_buffer_count", line_number)
    wait_count = _uint(_take(f, ("wait_count",), line_number),
                       "wait_count", line_number)
    signal_count = _uint(_take(f, ("signal_count",), line_number),
                         "signal_count", line_number)
    marker = _bool(_take(f, ("marker",), line_number), "marker", line_number)
    queued_depth = _uint(_take(f, ("queued_depth",), line_number),
                         "queued_depth", line_number)
    depth_capped = _bool(_take(f, ("depth_capped",), line_number),
                         "depth_capped", line_number)
    _no_unknown(f, line_number)
    return Record(index, internal_ordinal, initialized, issue_begin_ns,
                  issue_end_ns, complete_ns, completed, status,
                  command_buffer_count, wait_count, signal_count, marker,
                  queued_depth, depth_capped)


def _parse_summary(line: str, line_number: int) -> Summary:
    f = _fields(line, SUMMARY_PREFIX, line_number)
    result_valid = _bool(_take(f, ("result_valid",), line_number),
                         "result_valid", line_number)
    capacity = _uint(_take(f, ("capacity",), line_number), "capacity", line_number)
    reserved = _uint(_take(f, ("reserved",), line_number), "reserved", line_number)
    complete = _uint(_take(f, ("complete", "completed"), line_number),
                     "complete", line_number)
    overflow = _uint(_take(f, ("overflow",), line_number), "overflow", line_number)
    _no_unknown(f, line_number)
    return Summary(result_valid, capacity, reserved, complete, overflow)


def parse_census(lines: Iterable[str]) -> tuple[list[Record], Summary]:
    records: list[Record] = []
    summary: Summary | None = None
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if line.startswith(RECORD_PREFIX):
            records.append(_parse_record(line, line_number))
        elif line.startswith(SUMMARY_PREFIX):
            if summary is not None:
                raise CensusError(f"line {line_number}: duplicate census summary")
            summary = _parse_summary(line, line_number)
    if summary is None:
        raise CensusError("missing census summary")
    validate_census(records, summary)
    return records, summary


def validate_census(records: Sequence[Record], summary: Summary) -> None:
    if not summary.result_valid:
        raise CensusError("summary result_valid is false")
    if summary.overflow != 0:
        raise CensusError(f"summary reports overflow={summary.overflow}")
    if summary.capacity < summary.reserved:
        raise CensusError("summary capacity is smaller than reserved")
    if summary.reserved != len(records):
        raise CensusError(
            f"summary reserved={summary.reserved}, parsed records={len(records)}"
        )
    if summary.complete != len(records):
        raise CensusError(
            f"summary complete={summary.complete}, parsed records={len(records)}"
        )

    previous: Record | None = None
    for expected, record in enumerate(records):
        if record.index != expected:
            raise CensusError(
                f"records are not contiguous: expected index {expected}, "
                f"found {record.index}"
            )
        if record.internal_ordinal != record.index:
            raise CensusError(
                f"record {record.index} internal_ordinal does not equal index"
            )
        if not record.initialized:
            raise CensusError(f"record {record.index} is not initialized")
        if not record.completed:
            raise CensusError(f"record {record.index} is not complete")
        if record.status != 0:
            raise CensusError(f"record {record.index} has status={record.status}")
        if not record.issue_begin_ns or not record.issue_end_ns or not record.complete_ns:
            raise CensusError(f"record {record.index} has an uninitialized timestamp")
        if record.issue_begin_ns > record.issue_end_ns:
            raise CensusError(f"record {record.index} issue timestamps run backwards")
        if record.issue_end_ns > record.complete_ns:
            raise CensusError(f"record {record.index} completes before issue ends")
        expected_marker = record.command_buffer_count == 0
        if record.marker != expected_marker:
            raise CensusError(
                f"record {record.index} marker does not match command_buffer_count"
            )
        if record.queued_depth > 64:
            raise CensusError(
                f"record {record.index} has queued_depth above the cap of 64"
            )
        if record.depth_capped and record.queued_depth != 64:
            raise CensusError(
                f"record {record.index} has depth_capped below the cap of 64"
            )
        if previous is not None:
            if record.internal_ordinal != previous.internal_ordinal + 1:
                raise CensusError("internal_ordinal is not contiguous")
            if record.issue_begin_ns < previous.issue_end_ns:
                raise CensusError("issue ranges overlap or run backwards")
            if record.issue_begin_ns < previous.issue_begin_ns:
                raise CensusError("issue_begin_ns is not monotonic")
            if record.issue_end_ns < previous.issue_end_ns:
                raise CensusError("issue_end_ns is not monotonic")
            if record.complete_ns < previous.complete_ns:
                raise CensusError("complete_ns is not monotonic")
        previous = record


def parse_shim(lines: Iterable[str]) -> list[ShimTimestamp]:
    raw_lines = list(lines)
    out: list[ShimTimestamp] = []
    for position, raw in enumerate(raw_lines):
        line_number = position + 1
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # A killed replay can leave one partially written numeric timestamp at
        # EOF.  It is not a shim event: accept only this narrow final form, and
        # only when the physical line has no newline.  Malformed complete lines
        # and malformed lines anywhere else remain fatal.
        if (position == len(raw_lines) - 1 and not raw.endswith(("\n", "\r"))
                and len(parts) == 1 and parts[0].isdigit()):
            break
        if any("=" in part for part in parts):
            if parts[0] != "CUDAVK_SUBMIT_SHIM_TIMESTAMP":
                raise CensusError(
                    f"shim line {line_number}: unknown prefix {parts[0]!r}"
                )
            parts = parts[1:]
            f: dict[str, str] = {}
            for token in parts:
                if token.count("=") != 1:
                    raise CensusError(
                        f"shim line {line_number}: malformed field {token!r}"
                    )
                name, value = token.split("=", 1)
                if name in f:
                    raise CensusError(
                        f"shim line {line_number}: duplicate field {name!r}"
                    )
                f[name] = value
            timestamp_ns = _uint(_take(f, ("timestamp_ns",), line_number),
                                 "shim timestamp_ns", line_number)
            submit_index = _uint(_take(f, ("submit_index",), line_number),
                                 "shim submit_index", line_number)
            _no_unknown(f, line_number)
        else:
            if len(parts) != 2:
                raise CensusError(
                    f"shim line {line_number}: expected timestamp_ns submit_index"
                )
            timestamp_ns = _uint(parts[0], "shim timestamp_ns", line_number)
            submit_index = _uint(parts[1], "shim submit_index", line_number)
        if not timestamp_ns:
            raise CensusError(f"shim line {line_number}: timestamp is zero")
        if out:
            if submit_index != out[-1].submit_index + 1:
                raise CensusError("shim submit indices are not contiguous")
            if timestamp_ns < out[-1].timestamp_ns:
                raise CensusError("shim timestamps are not monotonic")
        out.append(ShimTimestamp(timestamp_ns, submit_index))
    if not out:
        raise CensusError("shim timestamp file is empty")
    return out


def _distribution(values: Sequence[int], suffix: str = "_ns") -> dict[str, float | int]:
    if not values:
        raise CensusError("cannot summarize an empty value set")
    ordered = sorted(values)
    # Nearest-rank p95 is stable and does not need numpy.
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
    return {
        "count": len(values),
        "min" + suffix: ordered[0],
        "median" + suffix: statistics.median(ordered),
        "p95" + suffix: p95,
        "max" + suffix: ordered[-1],
        "mean" + suffix: statistics.fmean(values),
    }


def analyze(records: Sequence[Record], summary: Summary,
            internal_target_start: int, target_frame_count: int,
            shim: Sequence[ShimTimestamp] | None = None,
            external_target_start: int | None = None) -> dict[str, object]:
    # validate_census is intentionally repeated for callers using this API
    # directly rather than parse_census().
    validate_census(records, summary)
    if target_frame_count < 2:
        raise CensusError("target-frame-count must be at least 2")

    command_records = [record for record in records if not record.marker]
    starts = [i for i, record in enumerate(command_records)
              if record.internal_ordinal == internal_target_start]
    if not starts:
        raise CensusError(
            f"internal target start {internal_target_start} is not a "
            "command-bearing record"
        )
    selected = command_records[starts[0]:]
    if len(selected) < 2:
        raise CensusError("selected internal target range has fewer than 2 records")

    first, last = selected[0], selected[-1]
    denominator = target_frame_count - 1
    issue_durations = [r.issue_end_ns - r.issue_begin_ns for r in selected]
    result: dict[str, object] = {
        "result_valid": True,
        "diagnostic_only": True,
        "selection": {
            "internal_target_start": internal_target_start,
            "internal_target_end": last.internal_ordinal,
            "target_frame_count": target_frame_count,
            "selected_command_record_count": len(selected),
            "filtered_marker_record_count": sum(r.marker for r in records),
            "first_complete_ns": first.complete_ns,
            "last_complete_ns": last.complete_ns,
        },
        "diagnostic_completion_frame_interval_ms":
            (last.complete_ns - first.complete_ns) / denominator / 1_000_000.0,
        "diagnostic_issue_begin_frame_interval_ms":
            (last.issue_begin_ns - first.issue_begin_ns) / denominator / 1_000_000.0,
        "diagnostic_issue_end_frame_interval_ms":
            (last.issue_end_ns - first.issue_end_ns) / denominator / 1_000_000.0,
        "issue_duration": _distribution(issue_durations),
        "runtime_queued_depth": {
            **_distribution([r.queued_depth for r in selected], suffix=""),
            "capped_record_count": sum(r.depth_capped for r in selected),
            "values_are_lower_bounds": any(r.depth_capped for r in selected),
        },
        "summary": {
            "capacity": summary.capacity,
            "reserved": summary.reserved,
            "complete": summary.complete,
            "overflow": summary.overflow,
        },
    }

    if shim is not None:
        if external_target_start is None:
            raise CensusError(
                "--external-target-start is required with --shim-timestamps; "
                "external/internal parity is never inferred"
            )
        positions = [i for i, row in enumerate(shim)
                     if row.submit_index == external_target_start]
        if not positions:
            raise CensusError(
                f"external target start {external_target_start} is absent from shim"
            )
        external = list(shim[positions[0]:])
        expected = SUBMITS_PER_FRAME * (target_frame_count - 1) + 1
        if len(external) != expected:
            raise CensusError(
                f"external target range has {len(external)} submits, expected "
                f"exactly {expected} for {target_frame_count} frames"
            )
        result["external_enqueue"] = {
            "external_target_start": external_target_start,
            "external_target_end": external[-1].submit_index,
            "submit_count": len(external),
            "submits_per_frame": SUBMITS_PER_FRAME,
            "first_timestamp_ns": external[0].timestamp_ns,
            "last_timestamp_ns": external[-1].timestamp_ns,
            "diagnostic_enqueue_frame_interval_ms":
                (external[-1].timestamp_ns - external[0].timestamp_ns)
                / denominator / 1_000_000.0,
            "submit_interval": _distribution([
                b.timestamp_ns - a.timestamp_ns for a, b in zip(external, external[1:])
            ]),
        }
    elif external_target_start is not None:
        raise CensusError("--external-target-start requires --shim-timestamps")

    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", type=Path, help="combined output containing census lines")
    parser.add_argument("--internal-target-start", type=int, required=True,
                        help="exact internal command-bearing record starting the target")
    parser.add_argument("--target-frame-count", type=int, required=True,
                        help="exact number of frames from target start through run end")
    parser.add_argument("--shim-timestamps", type=Path,
                        help="optional external timestamp_ns/submit_index file")
    parser.add_argument("--external-target-start", type=int,
                        help="exact first external submit index of the target")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with args.census.open(encoding="utf-8", errors="replace") as stream:
            records, summary = parse_census(stream)
        shim = None
        if args.shim_timestamps is not None:
            with args.shim_timestamps.open(encoding="utf-8", errors="replace") as stream:
                shim = parse_shim(stream)
        result = analyze(records, summary, args.internal_target_start,
                         args.target_frame_count, shim, args.external_target_start)
    except (CensusError, OSError) as exc:
        print(f"cp_submit_census_analyze.py: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2 if args.pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
