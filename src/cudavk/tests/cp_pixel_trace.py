#!/usr/bin/env python3
"""Fail-closed CPU reader for bounded CP_PIXEL_TRACE stderr JSON lines.

This reports a tie prediction, not a cause or a correctness/performance verdict.
Non-trace stderr is ignored. Record order is allocation order, NOT atomic order.
"""
import argparse
import json
import sys
from collections import defaultdict

PREFIX = "CP_PIXEL_TRACE "
CAPACITY = 65536
U64 = (1 << 64) - 1
ATOMIC = 5
STATUSES = set(range(6))
COMPARE = {1: "LESS", 3: "LEQUAL", 4: "GREATER", 6: "GEQUAL"}


class TraceError(ValueError):
    pass


def uint(obj, key, bits=32):
    value = obj.get(key)
    if bits == 64 and isinstance(value, str):
        try:
            value = int(value, 16)  # Host emits fixed-width hexadecimal strings.
        except ValueError:
            raise TraceError(f"invalid {key}") from None
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise TraceError(f"missing/invalid {key}")
    return value


def group(r):
    return tuple(r[k] for k in ("event", "framebuffer", "pass", "sample"))


def origin(r):
    return tuple(r[k] for k in ("segment", "input_primitive", "fan"))


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise TraceError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def parse_trace(lines):
    epochs, active = [], None
    for number, line in enumerate(lines, 1):
        if not line.startswith("CP_PIXEL_TRACE"):
            continue
        if not line.startswith(PREFIX) or len(line) > 16384:
            raise TraceError(f"line {number}: invalid prefix/oversized record")
        try:
            r = json.loads(line[len(PREFIX):], object_pairs_hook=unique_object)
        except (ValueError, TypeError) as error:
            raise TraceError(f"line {number}: {error}") from error
        if not isinstance(r, dict):
            raise TraceError("record is not an object")
        kind = r.get("kind")
        epoch = uint(r, "epoch")
        if kind == "epoch":
            if active is not None or any(e["epoch"] == epoch for e in epochs):
                raise TraceError("duplicate/nested epoch")
            capacity = uint(r, "capacity")
            if not 0 < capacity <= CAPACITY:
                raise TraceError("invalid capacity")
            active = dict(epoch=epoch, capacity=capacity, x=uint(r,"x"),
                          y=uint(r,"y"), records=[], finals={}, origins={}, reductions={}, batches={}, slices={})
            continue
        if active is None or epoch != active["epoch"]:
            raise TraceError("missing or mismatched epoch")
        if kind == "record":
            for key in ("event", "segment", "pass", "sample", "storage_id",
                        "global_id", "input_primitive", "fan", "depth_bits",
                        "depth_key", "compare_op", "status", "origin_valid",
                        "won_atomic", "abuf_mode", "stable", "depth_test", "blend_peel"):
                r[key] = uint(r, key)
            for key in ("framebuffer", "packed", "old", "result"):
                r[key] = uint(r, key, 64)
            if r["status"] not in STATUSES:
                raise TraceError("unknown status")
            if r["status"] == 0:
                raise TraceError("unfinished covered record")
            if r["origin_valid"] != 1:
                raise TraceError("missing origin")
            if (any(r[k] > 1 for k in ("won_atomic", "stable", "depth_test", "blend_peel"))
                    or r["compare_op"] > 7):
                raise TraceError("invalid enum/boolean")
            if r["global_id"] != r["segment"] + r["storage_id"]:
                raise TraceError("invalid global ID")
            identity = (r["event"], r["segment"], r["storage_id"])
            previous = active["origins"].setdefault(identity, origin(r))
            if previous != origin(r):
                raise TraceError("inconsistent origin map")
            if r["status"] == ATOMIC:
                packed_high = r["storage_id"] if r["blend_peel"] else r["depth_key"]
                packed_id = r["storage_id"] if r["blend_peel"] else r["global_id"]
                if r["packed"] != (packed_high << 32 | (packed_id ^ 0xffffffff)):
                    raise TraceError("packed candidate disagrees with depth/identity metadata")
                if r["result"] != min(r["old"], r["packed"]):
                    raise TraceError("atomic result is not minimum")
                if r["won_atomic"] != int(r["packed"] < r["old"]):
                    raise TraceError("atomic won flag inconsistent")
            active["records"].append(r)
            if len(active["records"]) > active["capacity"]:
                raise TraceError("trace overflow")
        elif kind == "final":
            for key in ("event", "pass", "sample"):
                r[key] = uint(r, key)
            for key in ("framebuffer", "packed"):
                r[key] = uint(r, key, 64)
            r["group"] = uint(r, "group")
            key = (r["group"], r["framebuffer"], r["pass"], r["sample"])
            if key in active["finals"]:
                raise TraceError("duplicate final")
            active["finals"][key] = r
        elif kind == "reduction":
            reduction = uint(r, "group")
            event = uint(r, "event")
            members = active["reductions"].setdefault(reduction, set())
            if event in members:
                raise TraceError("duplicate reduction member")
            members.add(event)
            if sum(map(len, active["reductions"].values())) > CAPACITY:
                raise TraceError("reduction membership overflow")
        elif kind == "batch":
            event = uint(r, "event")
            if event in active["batches"]:
                raise TraceError("duplicate batch")
            for field in ("draws", "triangles"):
                r[field] = uint(r, field)
            r["framebuffer"] = uint(r, "framebuffer", 64)
            active["batches"][event] = r
        elif kind == "slice":
            event = uint(r, "event")
            for field in ("row", "start", "count", "drawid", "primitive_begin",
                          "primitive_count", "instances"):
                r[field] = uint(r, field)
            key = (event, r["row"])
            if key in active["slices"]:
                raise TraceError("duplicate slice row")
            active["slices"][key] = r
        elif kind == "end":
            if uint(r, "dropped") or uint(r, "incomplete"):
                raise TraceError("dropped or incomplete trace")
            if uint(r, "count") != len(active["records"]):
                raise TraceError("record count mismatch")
            for (event, row), sl in active["slices"].items():
                batch = active["batches"].get(event)
                if batch is None or row >= batch["draws"]:
                    raise TraceError("slice row out of range or missing batch")
            for event, batch in active["batches"].items():
                slices = sorted((sl for (ev, _), sl in active["slices"].items()
                                 if ev == event), key=lambda sl: sl["row"])
                if len(slices) != batch["draws"]:
                    raise TraceError("missing slice map")
                end = 0
                for sl in slices:
                    if sl["primitive_begin"] != end:
                        raise TraceError("slice overlap or gap")
                    if not sl["instances"] or sl["primitive_count"] % sl["instances"]:
                        raise TraceError("invalid instance mapping")
                    end += sl["primitive_count"]
                if end != batch["triangles"]:
                    raise TraceError("slice total differs from batch triangles")
            for rec in active["records"]:
                event = rec["event"]
                batch = active["batches"].get(event)
                if batch is None or batch["framebuffer"] != rec["framebuffer"]:
                    raise TraceError("missing or mismatched batch")
                matches = [sl for (ev, _), sl in active["slices"].items()
                           if ev == event and sl["primitive_begin"] <= rec["input_primitive"]
                           < sl["primitive_begin"] + sl["primitive_count"]]
                if len(matches) != 1:
                    raise TraceError("missing or ambiguous original draw slice")
                sl = matches[0]
                offset = rec["input_primitive"] - sl["primitive_begin"]
                per_instance = sl["primitive_count"] // sl["instances"]
                rec["draw_origin"] = dict(event=event, row=sl["row"], drawid=sl["drawid"],
                                          local_primitive=offset % per_instance,
                                          instance=offset // per_instance, fan=rec["fan"])
            atomic_groups = defaultdict(list)
            if not active["finals"]:
                raise TraceError("missing final winner record")
            for key in active["finals"]:
                if key[0] not in active["reductions"]:
                    raise TraceError("missing reduction membership")
            for rec in active["records"]:
                if rec["status"] != ATOMIC:
                    continue
                matches = [key for key in active["finals"]
                           if key[1:] == group(rec)[1:]
                           and rec["event"] in active["reductions"][key[0]]]
                if len(matches) != 1:
                    raise TraceError("missing final winner or ambiguous reduction membership")
                atomic_groups[matches[0]].append(rec)
            active["atomic_groups"] = dict(atomic_groups)
            for key, records in atomic_groups.items():
                if key not in active["finals"]:
                    raise TraceError("missing final winner record")
                final = active["finals"][key]["packed"]
                expected = min(min(q["old"], q["packed"]) for q in records)
                if final != expected:
                    raise TraceError("final winner inconsistent with atomic observations")
                if final != U64 and not any(q["packed"] == final for q in records):
                    raise TraceError("missing final winner origin")
            for key, final in active["finals"].items():
                if key not in atomic_groups and final["packed"] != U64:
                    raise TraceError("missing final winner origin")
            epochs.append(active)
            active = None
        else:
            raise TraceError("unknown record kind")
    if active is not None or not epochs:
        raise TraceError("missing epoch/end")
    return epochs


def predict_ties(epoch):
    """Only normal depth-key atomic groups support this counterfactual."""
    groups = epoch["atomic_groups"]
    reports = []
    for key, records in groups.items():
        if any(not r["depth_test"] or r["blend_peel"] for r in records):
            reports.append(dict(group=key, prediction="disabled depth or peel key; unsupported"))
            continue
        ops = {r["compare_op"] for r in records}
        if len(ops) != 1 or next(iter(ops)) not in COMPARE:
            reports.append(dict(group=key, prediction="unsupported comparison"))
            continue
        op = next(iter(ops))
        if any(r["packed"] != (r["depth_key"] << 32 | (r["global_id"] ^ 0xffffffff))
               for r in records):
            reports.append(dict(group=key, prediction="not a normal packed-depth group"))
            continue
        best = min(r["depth_key"] for r in records)
        tied = [r for r in records if r["depth_key"] == best]
        pick = min if op in (1, 4) else max
        original = pick(tied, key=lambda r: tuple(r["draw_origin"][k]
                        for k in ("event", "row", "instance", "local_primitive", "fan")))
        allocated = max(tied, key=lambda r: r["global_id"])
        final = epoch["finals"][key]["packed"]
        falsifiers = []
        if len(tied) == 1:
            falsifiers.append("unique best packed depth")
        if all(r["stable"] for r in tied):
            falsifiers.append("stable allocation")
        if original["draw_origin"] == allocated["draw_origin"]:
            falsifiers.append("same origin winner")
        reports.append(dict(group=key, compare=COMPARE[op], best_depth_key=best,
                            tie_count=len(tied), original_origin=origin(original),
                            allocated_origin=origin(allocated),
                            original_draw=original["draw_origin"],
                            allocated_draw=allocated["draw_origin"],
                            allocated_global_id=allocated["global_id"],
                            predicted_packed=allocated["packed"], observed_final=final,
                            final_matches_prediction=final == allocated["packed"],
                            falsifiers=falsifiers, claim="prediction only; not a cause"))
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stderr", help="stderr log, or - for stdin")
    args = ap.parse_args()
    try:
        with (open(args.stderr, encoding="utf-8") if args.stderr != "-" else sys.stdin) as stream:
            epochs = parse_trace(stream)
        print(json.dumps([dict(epoch=e["epoch"], predictions=predict_ties(e))
                          for e in epochs], indent=2))
    except (OSError, TraceError) as error:
        print(f"invalid pixel trace: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
