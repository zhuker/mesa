#!/usr/bin/env python3
"""Offline clipping full-replay gate (CPU only).

validate reads existing stdout/stderr plus a caller-supplied authoritative
submit/frame census and anchored line regexes. Regexes count complete matching
lines, or extract exactly one named (?P<count>...) aggregate. EOF must occur
exactly once. The stderr footer must be final; one device is the default.
Use --min-stable/refs/active-ids/ids to require specific dispatch modes.
No image-equivalence or performance claim is made by this validator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

PREFIX = "CUDAVK_CLIP_EQ_JSON:"
# Physical polygon buffer bound in kernels/cp_rasterize.cu, not a geometric bound.
CP_CLIP_MAX_VERTS = 10
COUNTERS = ("invocations comparisons words differing_invocations differing_words "
            "differing_n inside reject_shortcut loop_reject clipped bypass_planes "
            "bypass_triangles interpolate_after_bypass injections errors stable "
            "refs active_ids ids first_failure first_dispatch first_triangle "
            "first_candidate_n first_oracle_n first_candidate_word first_oracle_word "
            "standalone fused group_bin group_direct allocation_overflow").split()
COVERAGE = ("invocations words inside reject_shortcut loop_reject clipped "
            "bypass_planes bypass_triangles interpolate_after_bypass stable refs active_ids ids "
            "standalone fused group_bin group_direct").split()
ARMS = ["positive", "negative-copy", "negative-reject", "disabled"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uint(value):
    return type(value) is int and 0 <= value < 2**64


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("non-finite JSON number: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def line_matches(text, pattern):
    require(isinstance(pattern, str) and pattern.startswith("^") and pattern.endswith("$"),
            "log regex must start with ^ and end with $")
    regex = re.compile(pattern)
    return regex, [m for line in text.splitlines() if (m := regex.fullmatch(line))]


def count_log(text, pattern):
    regex, matches = line_matches(text, pattern)
    if "count" in regex.groupindex:
        require(len(matches) == 1, "aggregate count regex must match exactly once")
        value = matches[0].group("count")
        require(value is not None and re.fullmatch(r"[0-9]+", value), "non-integer log count")
        return int(value)
    return len(matches)


def validate_report(r, arm):
    require(isinstance(r, dict), "report must be an object")
    require(arm in ARMS, "unknown arm")
    require(r.get("type") == "clip_equivalence", "wrong report type")
    for field in ["complete", "error", "equal", "copy", "reject"]:
        require(type(r.get(field)) is bool, "missing or invalid bool: " + field)
    for field in ["expected", "dispatches", "break"]:
        require(uint(r.get(field)), "missing or invalid uint64: " + field)
    raw = r.get("counters")
    require(isinstance(raw, list) and len(raw) == 32 and all(uint(x) for x in raw),
            "counters must contain exactly 32 uint64 values")
    c = dict(zip(COUNTERS, raw))
    for field in COUNTERS:
        if field in r:
            require(uint(r[field]) and r[field] == c[field], "named counter mismatch: " + field)
    require(c["allocation_overflow"] == 0, "clip allocation overflow")
    require(raw[31] == 0, "nonzero reserved counter")
    require(r["complete"] and not r["error"] and not c["errors"], "incomplete/error report")
    n = c["invocations"]
    require(n == r["expected"] == c["comparisons"] and n > 0, "invocation/comparison coverage mismatch")
    require(sum(c[x] for x in ["standalone", "fused", "group_bin", "group_direct"]) == n,
            "dispatch path partition mismatch")
    require(r["dispatches"] > 0, "invalid dispatch coverage")
    require(c["words"] > 0 and c["words"] % 4 == 0, "no live-word coverage or partial float4")
    require(sum(c[x] for x in ["inside", "reject_shortcut", "loop_reject", "clipped"]) == n,
            "terminal partition mismatch")
    for field in ["differing_invocations", "differing_n", "injections", "stable", "refs", "active_ids", "ids"]:
        require(c[field] <= n, field + " exceeds invocations")
    require(c["differing_n"] <= c["differing_invocations"], "n differences exceed differing invocations")
    require(c["differing_words"] <= c["words"], "word differences exceed words")
    word_invocations = c["differing_invocations"] - c["differing_n"]
    require(word_invocations <= c["differing_words"] and
            bool(word_invocations) == bool(c["differing_words"]), "difference count mismatch")
    require(c["bypass_triangles"] <= c["loop_reject"] + c["clipped"], "bypass outside plane loop")
    require(c["bypass_triangles"] <= c["bypass_planes"] <= 7 * c["bypass_triangles"], "bypass plane count mismatch")
    require(c["interpolate_after_bypass"] <= c["bypass_triangles"], "after-bypass count mismatch")
    for enabled, fields in [(r["copy"], ["bypass_planes", "bypass_triangles", "interpolate_after_bypass"]),
                            (r["reject"], ["reject_shortcut"])]:
        if enabled:
            require(c[fields[0]] > 0, "enabled feature has no coverage: " + fields[0])
        else:
            require(all(c[x] == 0 for x in fields), "disabled feature has activity")
    differs = c["differing_invocations"] > 0
    require(r["equal"] == (not differs), "equal flag mismatch")
    require(c["first_failure"] == int(differs), "first failure claim mismatch")
    if differs:
        require(0 < c["first_dispatch"] <= r["dispatches"], "invalid first dispatch")
        require(c["first_triangle"] < n, "invalid first triangle")
        require(all((c[x] == 0 or 3 <= c[x] <= CP_CLIP_MAX_VERTS) for x in ["first_candidate_n", "first_oracle_n"]),
                "invalid first polygon size")
        require(c["first_candidate_word"] < 2**32 and c["first_oracle_word"] < 2**32,
                "invalid first word")
        require(c["first_candidate_n"] != c["first_oracle_n"] or
                c["first_candidate_word"] != c["first_oracle_word"], "first failure is equal")
    else:
        require(all(x == 0 for x in raw[19:26]), "first failure evidence without difference")
    expected_break = {"positive": 0, "disabled": 0, "negative-copy": 1, "negative-reject": 2}[arm]
    require(r["break"] == expected_break, "wrong BREAK configuration")
    if arm in ["positive", "disabled"]:
        require(not differs and c["injections"] == 0, "positive mismatch/injection")
        if arm == "disabled":
            require(not r["copy"] and not r["reject"], "disabled arm has enabled feature")
    elif arm == "negative-copy":
        require(r["copy"] and 0 < c["injections"] <= c["bypass_triangles"], "no actual bypass injection")
        require(c["injections"] == c["differing_words"] == c["differing_invocations"] and
                c["differing_n"] == 0, "copy BREAK did not detect each word corruption")
    else:
        require(r["reject"] and c["injections"] > 0, "no actual shortcut injection")
        require(c["injections"] == c["reject_shortcut"] == c["differing_n"] == c["differing_invocations"] and
                c["differing_words"] == 0, "reject BREAK did not detect each polygon corruption")
    return c


def validate_logs(stdout, stderr, spec):
    require(type(spec["exit_code"]) is int and spec["exit_code"] == 0, "replay exit is not zero")
    devices = spec.get("devices", 1)
    require(type(devices) is int and devices > 0, "invalid device count")
    for field in ["expected_submits", "expected_frames"]:
        require(uint(spec[field]) and spec[field] > 0, "invalid expectation: " + field)
    combined = stdout + "\n" + stderr
    submits = count_log(combined, spec["submit_regex"])
    frames = count_log(combined, spec["frame_regex"])
    require(submits == spec["expected_submits"], "submit count mismatch")
    require(frames == spec["expected_frames"], "frame count mismatch")
    require(len(line_matches(combined, spec["eof_regex"])[1]) == 1, "normal EOF missing/duplicated")
    lines = [line for line in stderr.splitlines() if line.strip()]
    records = [line for line in lines if PREFIX in line]
    require(not any(PREFIX in line for line in stdout.splitlines()), "report on stdout")
    require(len(records) == devices, "missing/duplicate device report")
    require(lines[-devices:] == records, "reports are not final stderr records")
    require(all(line.startswith(PREFIX) for line in records), "malformed report prefix")
    reports = [strict_json(line[len(PREFIX):]) for line in records]
    counters = [validate_report(r, spec["arm"]) for r in reports]
    for field in ["copy", "reject"]:
        expected = spec.get("expected_" + field)
        require(expected is None or type(expected) is bool, "invalid feature expectation")
        if expected is not None:
            require(all(r[field] == expected for r in reports), "feature configuration mismatch: " + field)
    for field in COVERAGE:
        minimum = spec.get("min_" + field, 0)
        require(uint(minimum), "invalid coverage minimum: " + field)
        require(sum(c[field] for c in counters) >= minimum, "missing minimum coverage: " + field)
    return {"reports": reports, "observed_submits": submits, "observed_frames": frames}


def validate_files(spec):
    result = {"spec": spec, "pass": False}
    try:
        command = spec.get("command")
        require(isinstance(command, list) and command and all(isinstance(x, str) for x in command),
                "recorded command must be nonempty argv array")
        blobs = {key: Path(spec[key]).read_bytes() for key in ["stdout", "stderr"]}
        result["sha256"] = {key: hashlib.sha256(blob).hexdigest() for key, blob in blobs.items()}
        result.update(validate_logs(blobs["stdout"].decode("utf-8"), blobs["stderr"].decode("utf-8"), spec))
        result["pass"] = True
    except (ValueError, KeyError, OSError, TypeError, re.error) as error:
        result["error"] = str(error)
    return result


class GateTests(unittest.TestCase):
    def setUp(self):
        self.c = [0] * 32
        for i, value in {0:10, 1:10, 2:120, 6:4, 7:2, 8:1, 9:3, 10:6, 11:3, 12:2, 15:10, 26:10}.items():
            self.c[i] = value
        self.r = dict(type="clip_equivalence", expected=10, dispatches=2, complete=True,
                      error=False, equal=True, copy=True, reject=True, counters=self.c, **{"break":0})
        self.s = dict(arm="positive", exit_code=0, expected_submits=2, expected_frames=1,
                      submit_regex=r"^submit$", frame_regex=r"^frame$", eof_regex=r"^EOF$")
        self.out = "submit\nsubmit\nframe\nEOF\n"

    def check(self, r=None, **changes):
        return validate_logs(self.out, PREFIX + json.dumps(self.r if r is None else r), dict(self.s, **changes))

    def test_positive(self):
        self.check(min_stable=10, expected_copy=True)
        # Abandoned grouped reservations can leave harmless dispatch-ID gaps.
        self.check(dict(self.r, dispatches=100))

    def test_negative(self):
        for arm, mode in [("negative-copy", 1), ("negative-reject", 2)]:
            c = self.c.copy()
            c[3] = c[13] = mode
            c[4 if mode == 1 else 5] = mode
            c[19:26] = [1, 1, 0, 3, 3 if mode == 1 else 0, 1 if mode == 1 else 0, 0]
            r = dict(self.r, counters=c, equal=False, **{"break":mode})
            self.check(r, arm=arm)
            with self.assertRaises(ValueError): self.check(r)
            with self.assertRaises(ValueError): self.check(arm=arm)
            c[13] = 0
            with self.assertRaises(ValueError): self.check(r, arm=arm)

    def test_disabled(self):
        c = self.c.copy()
        c[8] += c[7]; c[7] = 0
        c[10:13] = [0, 0, 0]
        self.check(dict(self.r, copy=False, reject=False, counters=c), arm="disabled")
        with self.assertRaises(ValueError): self.check(arm="disabled")

    def test_types(self):
        for i in range(32):
            for bad in [-1, 2**64, True, 1.0, "1", None]:
                c = self.c.copy(); c[i] = bad
                with self.subTest(index=i, bad=bad), self.assertRaises(ValueError):
                    self.check(dict(self.r, counters=c))
        for field in ["complete", "error", "equal", "copy", "reject", "expected", "dispatches", "break"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(dict(self.r, **{field: None}))

    def test_mismatch(self):
        for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
            c = self.c.copy(); c[i] += 1
            with self.subTest(index=i), self.assertRaises(ValueError): self.check(dict(self.r, counters=c))
        for changes in [dict(complete=False), dict(error=True), dict(equal=False), dict(expected=0),
                        dict(dispatches=0), dict(copy=False), dict(reject=False), dict(words=1),
                        dict(counters=[]), dict(counters=self.c + [0])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.check(dict(self.r, **changes))
        for i in [2, 7, 10]:
            c = self.c.copy(); c[i] = 0
            with self.assertRaises(ValueError): self.check(dict(self.r, counters=c))

    def test_coverage_bounds(self):
        for index, value in [(10, 22), (11, 5), (12, 4), (15, 11), (16, 11), (17, 11), (18, 11)]:
            c = self.c.copy(); c[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.check(dict(self.r, counters=c))
        c = self.c.copy(); c[8] += c[7]; c[7] = 0
        with self.assertRaises(ValueError): self.check(dict(self.r, counters=c))
        c = self.c.copy(); c[10:13] = [0, 0, 0]
        with self.assertRaises(ValueError): self.check(dict(self.r, counters=c))
        for field in COVERAGE:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(**{"min_" + field: 1000})

    def test_replay_failures(self):
        for changes in [dict(exit_code=1), dict(expected_submits=3), dict(expected_frames=2),
                        dict(eof_regex="^never$"), dict(devices=2), dict(min_ids=1), dict(min_stable=11),
                        dict(expected_copy=False), dict(submit_regex="submit"), dict(min_refs=-1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.check(**changes)
        record = PREFIX + json.dumps(self.r)
        for log in ["", record + "\n" + record, record + "\ntrailing error", PREFIX + "{}",
                    record[:-1] + ', "complete":true}', PREFIX + "NaN", " " + record]:
            with self.subTest(log=log), self.assertRaises(ValueError): validate_logs(self.out, log, self.s)
        for out in [self.out.replace("EOF", ""), self.out + "EOF\n", self.out + record]:
            with self.assertRaises(ValueError): validate_logs(out, record, self.s)

    def test_counts(self):
        self.assertEqual(count_log("total=42", r"^total=(?P<count>\d+)$"), 42)
        with self.assertRaises(ValueError): count_log("n=1\nn=2", r"^n=(?P<count>\d+)$")
        self.assertEqual(count_log("not submit", r"^submit$"), 0)
        self.assertEqual(count_log("submit suffix", r"^submit|frame$"), 0)

    def test_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory) / "stdout", Path(directory) / "stderr"
            out.write_text(self.out); err.write_text(PREFIX + json.dumps(self.r))
            spec = dict(self.s, stdout=str(out), stderr=str(err), command=["synthetic-only"])
            result = validate_files(spec)
            self.assertTrue(result["pass"])
            self.assertEqual(result["sha256"]["stdout"], hashlib.sha256(out.read_bytes()).hexdigest())
            err.write_bytes(b"\xff")
            self.assertFalse(validate_files(spec)["pass"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-test")
    one = sub.add_parser("validate", help="validate one existing full replay")
    for name in ["stdout", "stderr", "submit-regex", "frame-regex", "eof-regex", "output", "command-json"]:
        one.add_argument("--" + name, required=True)
    for name in ["exit-code", "expected-submits", "expected-frames"]:
        one.add_argument("--" + name, type=int, required=True)
    one.add_argument("--arm", choices=ARMS, required=True)
    one.add_argument("--devices", type=int, default=1)
    for field in COVERAGE:
        one.add_argument("--min-" + field.replace("_", "-"), type=int, default=0)
    for field in ["copy", "reject"]:
        one.add_argument("--expected-" + field, choices=["on", "off"])
    args = parser.parse_args()
    if args.mode == "self-test":
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
    try:
        spec = vars(args).copy()
        spec["command"] = strict_json(spec.pop("command_json"))
        for field in ["copy", "reject"]:
            key = "expected_" + field
            if spec[key] is not None:
                spec[key] = spec[key] == "on"
        result = validate_files(spec)
    except (ValueError, TypeError) as error:
        result = {"pass": False, "error": str(error)}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
