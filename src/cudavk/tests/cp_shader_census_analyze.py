#!/usr/bin/env python3
"""Strict CPU-only validator for CUDAVK_SHADER_CENSUS logs."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RECORD_PREFIX = "CUDAVK_SHADER_CENSUS_RECORD "
SUMMARY_PREFIX = "CUDAVK_SHADER_CENSUS_SUMMARY "
HASH_RE = re.compile(r"[0-9a-f]{64}")
GRID_RE = re.compile(r"([0-9]+)x([0-9]+)x([0-9]+)")
RECORD_FIELDS = {
    "version", "index", "status", "stage", "stage_hash", "submit", "batch",
    "launch", "exec", "fused", "fetch_runs", "total_vertices", "grid",
    "block", "draws", "stream", "bank", "regs", "spill_bytes",
    "blocks_per_sm", "reg_cap", "sm",
}
SUMMARY_FIELDS = {
    "version", "capacity", "population", "stored", "dropped", "missing_key",
    "key_mismatch", "unknown_stream", "final_submit", "context_sync_ok", "allocation_ok", "complete_population",
    "result_valid", "diagnostic_timing_invalid", "cost_join_required",
    "production_admitted",
}

class CensusError(ValueError):
    pass


def kv(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.split():
        if "=" not in item:
            raise CensusError(f"non-key/value token {item!r}")
        key, value = item.split("=", 1)
        if not key or not value or key in result:
            raise CensusError(f"bad or duplicate field {item!r}")
        result[key] = value
    return result


def integer(row: dict[str, str], key: str, minimum: int | None = 0) -> int:
    try:
        value = int(row[key], 10)
    except (KeyError, ValueError) as exc:
        raise CensusError(f"field {key} is not an integer") from exc
    if minimum is not None and value < minimum:
        raise CensusError(f"field {key} is below {minimum}")
    return value


def triplet(row: dict[str, str], key: str) -> tuple[int, int, int]:
    match = GRID_RE.fullmatch(row[key])
    if not match:
        raise CensusError(f"field {key} is not AxBxC")
    return tuple(map(int, match.groups()))


def analyze(text: str) -> dict:
    records: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        try:
            if line.startswith(RECORD_PREFIX):
                records.append(kv(line[len(RECORD_PREFIX):]))
            elif line.startswith(SUMMARY_PREFIX):
                summaries.append(kv(line[len(SUMMARY_PREFIX):]))
        except CensusError as exc:
            raise CensusError(f"line {line_no}: {exc}") from exc
    if len(summaries) != 1:
        raise CensusError(f"expected one summary, got {len(summaries)}")
    summary = summaries[0]
    if set(summary) != SUMMARY_FIELDS:
        raise CensusError(f"summary fields differ: missing={sorted(SUMMARY_FIELDS-set(summary))} extra={sorted(set(summary)-SUMMARY_FIELDS)}")
    if integer(summary, "version") != 1:
        raise CensusError("unsupported summary version")
    capacity = integer(summary, "capacity")
    population = integer(summary, "population")
    stored = integer(summary, "stored")
    if capacity < 262144:
        raise CensusError("capacity is below favorite3 population requirement")
    if not population or stored != population or len(records) != population:
        raise CensusError("incomplete or empty record population")
    for key in ("dropped", "missing_key", "key_mismatch", "unknown_stream"):
        if integer(summary, key) != 0:
            raise CensusError(f"summary {key} is nonzero")
    if integer(summary, "final_submit", 1) < 1:
        raise CensusError("no command-bearing submit identity")
    required = {
        "context_sync_ok": "yes", "allocation_ok": "yes",
        "complete_population": "yes",
        "result_valid": "yes", "diagnostic_timing_invalid": "1",
        "cost_join_required": "yes", "production_admitted": "no",
    }
    for key, value in required.items():
        if summary[key] != value:
            raise CensusError(f"summary {key} must be {value}")

    identities: set[tuple[int, int]] = set()
    previous_identity = (0, 0)
    normalized = []
    for expected_index, row in enumerate(records):
        if set(row) != RECORD_FIELDS:
            raise CensusError(f"record {expected_index} fields differ: missing={sorted(RECORD_FIELDS-set(row))} extra={sorted(set(row)-RECORD_FIELDS)}")
        if integer(row, "version") != 1 or integer(row, "index") != expected_index:
            raise CensusError(f"record {expected_index} version/index mismatch")
        if row["status"] != "complete" or row["stage"] != "VS":
            raise CensusError(f"record {expected_index} is not a complete explicit VS")
        if not HASH_RE.fullmatch(row["stage_hash"]):
            raise CensusError(f"record {expected_index} does not carry a full lowercase 256-bit hash")
        submit, batch = integer(row, "submit", 1), integer(row, "batch", 1)
        identity = (submit, batch)
        if identity in identities or identity <= previous_identity:
            raise CensusError(f"record {expected_index} identity is duplicate or non-monotonic: {identity}")
        identities.add(identity)
        previous_identity = identity
        if integer(row, "launch", 1) != expected_index + 1:
            raise CensusError(f"record {expected_index} host launch ordinal is not absolute/contiguous")
        total = integer(row, "total_vertices", 1)
        grid = triplet(row, "grid")
        block = triplet(row, "block")
        if grid != (min((total + 255)//256, 4096), 1, 1) or block != (256, 1, 1):
            raise CensusError(f"record {expected_index} launch shape mismatch")
        if integer(row, "draws", 1) < 1:
            raise CensusError(f"record {expected_index} has no draws")
        fused = integer(row, "fused")
        fetch = integer(row, "fetch_runs")
        if fused not in (0, 1) or fetch not in (0, 1):
            raise CensusError(f"record {expected_index} boolean field invalid")
        if (row["exec"], fused) not in (("CLASSIC", 0), ("VS_FETCH", 1)):
            raise CensusError(f"record {expected_index} execution/fused mismatch")
        bank = integer(row, "bank")
        expected_stream = "main" if bank == 0 else f"seg{bank-1}" if 1 <= bank <= 8 else None
        if expected_stream is None or row["stream"] != expected_stream:
            raise CensusError(f"record {expected_index} stream bank is unstable/unknown")
        for key in ("regs", "spill_bytes", "blocks_per_sm", "reg_cap"):
            integer(row, key, None)
        if not re.fullmatch(r"[0-9]+\.[0-9]+", row["sm"]):
            raise CensusError(f"record {expected_index} bad SM syntax")
        normalized.append(tuple((key, row[key]) for key in sorted(RECORD_FIELDS)))
    if integer(summary, "final_submit", 1) < previous_identity[0]:
        raise CensusError("summary final_submit precedes a recorded identity")
    return {
        "version": 1, "population": population, "capacity": capacity,
        "hashes": len({row["stage_hash"] for row in records}),
        "fused_launches": sum(row["exec"] == "VS_FETCH" for row in records),
        "standalone_launches": sum(row["exec"] == "CLASSIC" for row in records),
        "identity": "(absolute command-bearing submit, executed-batch ordinal)",
        "records": records, "summary": summary, "_normalized": normalized,
    }


def compare(results: list[dict]) -> None:
    reference = (results[0]["_normalized"], results[0]["summary"])
    for index, result in enumerate(results[1:], 2):
        if (result["_normalized"], result["summary"]) != reference:
            raise CensusError(f"rerun {index} differs from rerun 1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        results = [analyze(path.read_text()) for path in args.logs]
        compare(results)
    except (OSError, CensusError) as exc:
        print(f"shader-census: invalid: {exc}", file=sys.stderr)
        return 1
    result = {key: value for key, value in results[0].items() if not key.startswith("_") and key not in ("records", "summary")}
    result.update({"reruns": len(results), "rerun_equal": True,
                   "diagnostic_timing_invalid": 1, "cost_join_required": "yes",
                   "production_admitted": "no"})
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
