#!/usr/bin/env python3
"""Strict offline full-replay vfetch equivalence validator; no GPU required.

Use validate for one arm, suite for both captures/all three arms, self-test for
synthetic CPU tests. Count regexes count matches, or read exactly one named
(?P<count>...) aggregate. Expectations always come from the caller's census.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

COUNTERS = (
    "invocations valid slots words diff_validity diff_ids diff_slots diff_words "
    "differing_invocations injections expected_invocations expected_slots "
    "expected_words specialized_launches generic_launches layout_fallback_launches "
    "standalone_launches lane_launches group_launches errors"
).split()
DIFFS = "diff_validity diff_ids diff_slots diff_words differing_invocations".split()
# Match concrete error tokens/diagnostics, not benign counters such as errors=0.
ERROR_RE = re.compile(
    r"\b(?:CUDA_ERROR_[A-Z0-9_]+|NVRTC_ERROR_[A-Z0-9_]+|VK_ERROR_[A-Z0-9_]+)\b"
    r"|\b(?:fatal|segmentation fault|device lost|device loss)\b"
    r"|\b(?:error|failed|failure)\s*:"
    r"|\b(?:CUDA|NVRTC|compile|compilation|launch|cu[A-Z]\w*)[^\n]*\b(?:failed|failure|error)\b"
    r"|\bfailed\s+to\s+(?:compile|allocate|launch|load|synchronize)\b"
    r"|(?:cudavk:|abuffer:)[^\n]*\bfailed\b(?!\s*=\s*0\b)",
    re.IGNORECASE,
)


# Context teardown emits VFETCH_EQ before the existing device cache statistics.
# Permit only these exact numeric records after it; never arbitrary teardown text.
CACHE_TAIL_FIELDS = {
    "resources": "hits fallbacks rebuilds bytes peak_bytes arrays objects alloc_failures object_failures event_waits event_wait_skips event_wait_overflow resolve_batches resolve_cells resolve_unique purges purge_reclaimed_bytes purge_reclaimed_arrays purge_reclaimed_objects".split(),
    "resolve misses": "view sampler image target".split(),
    "resolve misses 2": "object rebuild epoch".split(),
    "resolve misses 3": ["lookup"],
}


def allowed_tail(line):
    for label, fields in CACHE_TAIL_FIELDS.items():
        pattern = re.escape("cudavk: texture cache " + label + ":")
        pattern += "".join(" " + field + r"=(?P<" + field + r">[0-9]+)" for field in fields)
        match = re.fullmatch(pattern, line)
        if match:
            return all(int(match.group(field)) == 0 for field in
                       ["alloc_failures", "object_failures", "event_wait_overflow"] if field in fields)
    return False


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_log(text, pattern):
    regex = re.compile(pattern, re.MULTILINE)
    matches = list(regex.finditer(text))
    if "count" in regex.groupindex:
        require(len(matches) == 1, "aggregate count regex must match exactly once")
        value = matches[0].group("count")
        require(value is not None and value.isdecimal(), "non-integer log count")
        return int(value)
    return len(matches)


def validate_report(r, arm):
    require(isinstance(r, dict), "report must be an object")
    for field in ["complete", "context_id"] + COUNTERS:
        require(type(r.get(field)) is int and 0 <= r[field] < 2**64,
                "missing or invalid uint64: " + field)
    require(r["context_id"] > 0, "invalid context identity")
    require(r["complete"] == 1 and r["errors"] == 0, "incomplete/CUDA error report")
    for field in ["invocations", "slots", "words"]:
        require(r[field] == r["expected_" + field], "coverage mismatch: " + field)
    require(r["words"] == 4 * r["slots"], "words != 4 * slots")
    require(r["valid"] <= r["invocations"], "valid exceeds invocations")
    for field, limit in [("diff_validity", "invocations"), ("diff_ids", "valid"),
                         ("diff_slots", "slots"), ("diff_words", "words"),
                         ("differing_invocations", "invocations"), ("injections", "slots")]:
        require(r[field] <= r[limit], field + " exceeds " + limit)
    if arm == "disabled":
        for field in ["specialized_launches", "invocations", "valid", "slots", "words", "injections"] + DIFFS:
            require(r[field] == 0, "disabled arm has activity: " + field)
        return
    require(r["specialized_launches"] > 0 and r["invocations"] > 0 and
            r["valid"] > 0 and r["slots"] > 0, "no specialized live-slot coverage")
    if arm == "positive":
        require(all(r[x] == 0 for x in DIFFS + ["injections"]), "positive mismatch/injection")
    elif arm == "negative":
        require(r["injections"] > 0 and r["diff_words"] > 0 and
                r["diff_slots"] > 0 and r["differing_invocations"] > 0,
                "BREAK did not detect actual slot corruption")
        require(r["diff_validity"] == 0 and r["diff_ids"] == 0,
                "BREAK unexpectedly changed validity/IDs")
    else:
        raise ValueError("unknown arm")


def replay_mode(spec):
    pair = [spec.get(key) is not None for key in
            ["submit_ts", "expected_stdout_sha256"]]
    regexes = [spec.get(key) is not None for key in
               ["submit_regex", "frame_regex", "eof_regex"]]
    if any(pair):
        require(all(pair), "tocpp requires submit_ts and expected_stdout_sha256 together")
        require(not any(regexes), "tocpp and plugin regex options are mutually exclusive")
        require(isinstance(spec["expected_stdout_sha256"], str) and
                re.fullmatch(r"[0-9a-f]{64}", spec["expected_stdout_sha256"]) is not None,
                "invalid expected stdout SHA256")
        return "tocpp"
    require(all(regexes), "plugin mode requires submit/frame/EOF regexes")
    return "plugin"


def submit_counts(text):
    rows = text.splitlines()
    require(bool(rows) and text.endswith("\n"), "empty/truncated submit timestamps")
    previous = -1
    for index, row in enumerate(rows):
        match = re.fullmatch(r"([0-9]+) ([0-9]+)", row)
        require(match is not None, "malformed submit timestamp row")
        ns, observed = map(int, match.groups())
        require(ns > previous, "submit timestamps not strictly monotonic")
        require(observed == index, "submit indices not contiguous from zero")
        previous = ns
    return len(rows), (len(rows) - 1) // 2


def validate_logs(stdout, stderr, spec, submit_ts=None, stdout_bytes=None):
    require(type(spec["exit_code"]) is int and spec["exit_code"] == 0, "replay exit is not zero")
    for field in ["expected_submits", "expected_frames", "devices"]:
        require(type(spec[field]) is int and spec[field] > 0, "invalid expectation: " + field)
    combined = stdout + "\n" + stderr
    error = ERROR_RE.search(combined)
    require(error is None, "error diagnostic in replay logs: " + (error.group(0) if error else ""))
    if replay_mode(spec) == "tocpp":
        require(isinstance(stdout_bytes, bytes), "tocpp requires raw stdout bytes")
        require(hashlib.sha256(stdout_bytes).hexdigest() == spec["expected_stdout_sha256"],
                "stdout SHA256 mismatch")
        require(isinstance(submit_ts, str), "missing submit timestamps")
        submits, frames = submit_counts(submit_ts)
    else:
        submits = count_log(combined, spec["submit_regex"])
        frames = count_log(combined, spec["frame_regex"])
        require(re.search(spec["eof_regex"], combined, re.MULTILINE) is not None, "normal EOF missing")
    require(submits == spec["expected_submits"], "submit count mismatch")
    require(frames == spec["expected_frames"], "frame count mismatch")
    lines = [line for line in stderr.splitlines() if line.strip()]
    records = [line for line in lines if line.startswith("VFETCH_EQ ")]
    require(len(records) == spec["devices"], "missing/duplicate device report")
    first_report = lines.index(records[0])
    require(all(line.startswith("VFETCH_EQ ") or allowed_tail(line)
                for line in lines[first_report:]), "unexpected/error stderr after report")
    reports = [strict_json(line[len("VFETCH_EQ "):]) for line in records]
    for report in reports:
        validate_report(report, spec["arm"])
    require(len({r["context_id"] for r in reports}) == len(reports), "duplicate context report identity")
    for kind in ["lane", "group"]:
        minimum = spec.get("min_" + kind + "_launches", 0)
        require(type(minimum) is int and minimum >= 0, "invalid coverage minimum")
        require(sum(r[kind + "_launches"] for r in reports) >= minimum,
                "missing " + kind + " coverage")
    return {"reports": reports, "observed_submits": submits, "observed_frames": frames}


def validate_files(spec):
    result = {"spec": spec, "pass": False}
    try:
        blobs = {key: Path(spec[key]).read_bytes() for key in ["stdout", "stderr"]}
        result["sha256"] = {key: hashlib.sha256(blob).hexdigest() for key, blob in blobs.items()}
        timestamps = None
        if replay_mode(spec) == "tocpp":
            blobs["submit_ts"] = Path(spec["submit_ts"]).read_bytes()
            result["sha256"]["submit_ts"] = hashlib.sha256(blobs["submit_ts"]).hexdigest()
            timestamps = blobs["submit_ts"].decode("ascii", errors="strict")
        result.update(validate_logs(blobs["stdout"].decode("utf-8", errors="strict"),
                                    blobs["stderr"].decode("utf-8", errors="strict"), spec,
                                    timestamps, blobs["stdout"]))
        result["pass"] = True
    except (ValueError, KeyError, OSError, TypeError) as error:
        result["error"] = str(error)
    return result


def validate_suite(specs):
    require(isinstance(specs, list), "suite must be a list")
    require(all(isinstance(s, dict) and isinstance(s.get("capture"), str) and s["capture"] for s in specs),
            "each arm needs a capture name")
    captures = set(s["capture"] for s in specs)
    require(len(captures) == 2 and len(specs) == 6, "suite needs exactly two captures and six arms")
    for capture in captures:
        arms = [s for s in specs if s["capture"] == capture]
        require(sorted(s["arm"] for s in arms) == ["disabled", "negative", "positive"],
                "each capture needs positive, negative and disabled")
        positive = next(s for s in arms if s["arm"] == "positive")
        negative = next(s for s in arms if s["arm"] == "negative")
        require(replay_mode(positive) == replay_mode(negative), "positive/BREAK modes differ")
        for key in ["expected_submits", "expected_frames", "submit_regex", "frame_regex",
                    "eof_regex", "expected_stdout_sha256", "devices"]:
            require(positive.get(key) == negative.get(key), "positive/BREAK census differs: " + key)
    results = [validate_files(s) for s in specs]
    return {"pass": all(r["pass"] for r in results), "runs": results}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.r = dict.fromkeys(COUNTERS, 0)
        self.r.update(complete=1, context_id=123, invocations=10, valid=9, slots=18, words=72,
                      expected_invocations=10, expected_slots=18, expected_words=72,
                      specialized_launches=1)
        self.s = dict(arm="positive", exit_code=0, expected_submits=2, expected_frames=1,
                      devices=1, submit_regex=r"^submit$", frame_regex=r"^frame$", eof_regex=r"^EOF$")
        self.out = "submit\nsubmit\nframe\nEOF\n"

    def check(self, r=None, **changes):
        spec = dict(self.s, **changes)
        return validate_logs(self.out, "VFETCH_EQ " + json.dumps(self.r if r is None else r), spec)

    def test_tocpp_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            out, err, ts = [Path(directory) / name for name in ["stdout", "stderr", "ts"]]
            raw = b"native replay\r\n"
            out.write_bytes(raw)
            err.write_text("VFETCH_EQ " + json.dumps(self.r))
            base = {k: v for k, v in self.s.items() if not k.endswith("_regex")}
            base.update(stdout=str(out), stderr=str(err), submit_ts=str(ts),
                        expected_stdout_sha256=hashlib.sha256(raw).hexdigest())
            for count, frames in [(6947, 3473), (6965, 3482)]:
                text = "".join(f"{1000000000 + i * 100} {i}\n" for i in range(count))
                ts.write_text(text)
                spec = dict(base, expected_submits=count, expected_frames=frames)
                self.assertTrue(validate_files(spec)["pass"])
                for changes in [dict(exit_code=1), dict(expected_submits=count-1),
                                dict(expected_frames=frames-1), dict(submit_regex="submit"),
                                dict(expected_stdout_sha256=None), dict(submit_ts=None),
                                dict(devices=2), dict(min_lane_launches=1)]:
                    with self.subTest(changes=changes):
                        self.assertFalse(validate_files(dict(spec, **changes))["pass"])
                rows = text.splitlines(keepends=True)
                reordered = rows[1] + rows[0] + "".join(rows[2:])
                for broken in [reordered, text.rsplit("\n", 2)[0] + "\n", text[:-1],
                               text.replace("1000000000 0", "1000000000 1"),
                               text.replace("1000000100 1", "1000000000 1"),
                               text.replace("1000000100 1", "999999999 1"),
                               text.replace("1000000100 1", "1000000100 2"),
                               text.replace("1000000100 1", "1 1 extra"),
                               text.replace("1000000100 1", "x 1"), ""]:
                    ts.write_text(broken)
                    self.assertFalse(validate_files(spec)["pass"])
                ts.unlink()
                self.assertFalse(validate_files(spec)["pass"])
                ts.write_text(text)
                out.write_bytes(raw.replace(b"\r\n", b"\n"))
                self.assertFalse(validate_files(spec)["pass"])
                out.write_bytes(raw)
                self.assertFalse(validate_files(dict(spec, arm="negative"))["pass"])
                r = dict(self.r, injections=1, diff_words=1, diff_slots=1, differing_invocations=1)
                err.write_text("VFETCH_EQ " + json.dumps(r))
                self.assertTrue(validate_files(dict(spec, arm="negative"))["pass"])
                self.assertFalse(validate_files(spec)["pass"])
                err.write_text("VFETCH_EQ " + json.dumps(self.r))
                # A full six-arm native suite also keeps mode/hash census checks.
                specs = []
                for capture in ["favorite2", "favorite3"]:
                    for arm in ["positive", "negative", "disabled"]:
                        report = self.r if arm == "positive" else r
                        if arm == "disabled":
                            report = dict.fromkeys(COUNTERS, 0)
                            report.update(complete=1, context_id=123)
                        path = Path(directory) / (capture + arm)
                        path.write_text("VFETCH_EQ " + json.dumps(report))
                        specs.append(dict(spec, capture=capture, arm=arm, stderr=str(path)))
                self.assertTrue(validate_suite(specs)["pass"])
                specs[1]["expected_stdout_sha256"] = "0" * 64
                with self.assertRaises(ValueError): validate_suite(specs)

    def test_positive(self):
        self.check()

    def test_negative(self):
        r = dict(self.r, injections=1, diff_words=1, diff_slots=1, differing_invocations=1)
        self.check(r, arm="negative")
        with self.assertRaises(ValueError): self.check(r)
        with self.assertRaises(ValueError): self.check(arm="negative")

    def test_disabled(self):
        r = dict.fromkeys(COUNTERS, 0); r.update(complete=1, context_id=123)
        self.check(r, arm="disabled")
        with self.assertRaises(ValueError): self.check(arm="disabled")

    def test_bad_counters(self):
        for field in ["complete", "context_id"] + COUNTERS:
            for value in [-1, 2**64, True, 1.0, "1", None]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check(dict(self.r, **{field: value}))
        for field in ["invocations", "slots", "words", "expected_invocations", "expected_slots", "expected_words", "errors"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(dict(self.r, **{field: self.r[field] + 1}))

    def test_incomplete(self):
        for changes in [dict(exit_code=1), dict(expected_submits=3), dict(expected_frames=2),
                        dict(eof_regex="never"), dict(devices=2), dict(min_lane_launches=1),
                        dict(min_group_launches=1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.check(**changes)
        record = "VFETCH_EQ " + json.dumps(self.r)
        for log in ["", record + "\n" + record, record + "\ntrailing error", "VFETCH_EQ {}",
                    record[:-1] + ', "complete":1}']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                validate_logs(self.out, log, self.s)

    def test_counts(self):
        self.assertEqual(count_log("total=42", r"total=(?P<count>\d+)"), 42)
        with self.assertRaises(ValueError): count_log("n=1 n=2", r"n=(?P<count>\d+)")

    def test_suite_missing(self):
        with self.assertRaises(ValueError): validate_suite([])

    def test_suite_and_hashes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            specs = []
            for capture in ["old", "crossroads"]:
                for arm in ["positive", "negative", "disabled"]:
                    r = self.r.copy()
                    if arm == "negative":
                        r.update(injections=1, diff_words=1, diff_slots=1, differing_invocations=1)
                    elif arm == "disabled":
                        r = dict.fromkeys(COUNTERS, 0); r.update(complete=1, context_id=123)
                    prefix = Path(directory) / (capture + "-" + arm)
                    out = str(prefix) + ".stdout"
                    err = str(prefix) + ".stderr"
                    Path(out).write_text(self.out)
                    Path(err).write_text("VFETCH_EQ " + json.dumps(r))
                    specs.append(dict(self.s, capture=capture, arm=arm,
                                      stdout=out, stderr=err, command=["synthetic-only"]))
            result = validate_suite(specs)
            self.assertTrue(result["pass"])
            self.assertEqual(len(result["runs"]), 6)
            self.assertEqual(result["runs"][0]["sha256"]["stdout"],
                             hashlib.sha256(self.out.encode()).hexdigest())
            specs[1]["expected_submits"] += 1
            with self.assertRaises(ValueError): validate_suite(specs)

    def test_multiple_devices(self):
        record = "VFETCH_EQ " + json.dumps(self.r)
        second = "VFETCH_EQ " + json.dumps(dict(self.r, context_id=456))
        validate_logs(self.out, record + "\n" + second, dict(self.s, devices=2))
        with self.assertRaises(ValueError):
            validate_logs(self.out, record + "\n" + record, dict(self.s, devices=2))

    def test_known_teardown_tail(self):
        record = "VFETCH_EQ " + json.dumps(self.r)
        tail = "\n".join("cudavk: texture cache " + label + ":" +
                         "".join(" " + field + "=0" for field in fields)
                         for label, fields in CACHE_TAIL_FIELDS.items())
        validate_logs(self.out, record + "\n" + tail, self.s)
        for broken in [tail.replace("alloc_failures=0", "alloc_failures=1"),
                       tail + " garbage", "cudavk: texture cache made up=0"]:
            with self.assertRaises(ValueError):
                validate_logs(self.out, record + "\n" + broken, self.s)

    def test_error_diagnostics(self):
        record = "VFETCH_EQ " + json.dumps(self.r)
        for diagnostic in ["cudavk: VFETCH_EQ instrumented compile failed",
                           "CUDA_ERROR_ILLEGAL_ADDRESS", "NVRTC_ERROR_COMPILATION",
                           "VK_ERROR_DEVICE_LOST", "cudavk: NVRTC compile failed for main",
                           "cudavk: cuModuleLoadData(main) failed: 1",
                           "cudavk: failed to compile texture sampler",
                           "cudavk: scratch arena grow to 20 bytes failed (1)",
                           "VS launch failed: 1", "fatal device error", "error: invalid PTX"]:
            for stream in ["stdout", "stderr"]:
                with self.subTest(diagnostic=diagnostic, stream=stream), self.assertRaises(ValueError):
                    validate_logs(self.out + (diagnostic if stream == "stdout" else ""),
                                  (diagnostic + "\n" if stream == "stderr" else "") + record, self.s)
        validate_logs(self.out + "errors=0 failed=0\n", record, self.s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-test")
    suite = sub.add_parser("suite", help="validate six logged arms from a JSON list")
    suite.add_argument("manifest")
    suite.add_argument("--output", required=True)
    one = sub.add_parser("validate", help="validate one existing replay log pair")
    for name in ["stdout", "stderr", "output"]:
        one.add_argument("--" + name, required=True)
    for name in ["submit-regex", "frame-regex", "eof-regex", "submit-ts", "expected-stdout-sha256"]:
        one.add_argument("--" + name)
    for name in ["exit-code", "expected-submits", "expected-frames"]:
        one.add_argument("--" + name, type=int, required=True)
    one.add_argument("--arm", choices=["positive", "negative", "disabled"], required=True)
    one.add_argument("--devices", type=int, default=1)
    one.add_argument("--min-lane-launches", type=int, default=0)
    one.add_argument("--min-group-launches", type=int, default=0)
    one.add_argument("--command-json", required=True, help="recorded replay argv as JSON string array")
    args = parser.parse_args()
    if args.mode == "self-test":
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)).wasSuccessful() else 1
    try:
        if args.mode == "suite":
            specs = strict_json(Path(args.manifest).read_text())
            for spec in specs:
                command = spec.get("command")
                require(isinstance(command, list) and command and all(isinstance(x, str) for x in command),
                        "each suite arm needs recorded command argv")
            result = validate_suite(specs)
        else:
            spec = vars(args).copy()
            spec["command"] = strict_json(spec.pop("command_json"))
            require(isinstance(spec["command"], list) and spec["command"] and
                    all(isinstance(x, str) for x in spec["command"]), "command must be argv array")
            result = validate_files(spec)
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as error:
        result = {"pass": False, "error": str(error)}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
