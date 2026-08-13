#!/usr/bin/env python3
"""
Make an iteration describe itself, and collect the descriptions into a page.

    cp_iter_report.py record LABEL [--desc TEXT] [--against LABEL] [--note ...]
    cp_iter_report.py page [--root DIR] [--out DIR]
    cp_iter_report.py convert [LABEL ...]

`record` writes `build/iter/LABEL/iteration.json`: what the iteration was, what
it cost, what it did to every sample against the iteration it was compared to,
and what the correctness gate said. cp_iterate.sh calls it at the end of a run,
so every iteration gets one without anyone remembering to.

`page` writes three files at the root of build/iter:

    iterations.json   every iteration.json collected into one history
    iterations.html   the summary over them, a row per iteration
    perf.html         one iteration in full, as perf.html?iter=LABEL

Both pages are static and hold no data — they fetch it at load time — so
neither has to be regenerated to look at a new iteration. Re-running `page`
rewrites iterations.json and nothing else needs to happen. It does mean they
have to be served over HTTP rather than opened from a file:// URL, which is
what a browser allows.

The point of all this is that a cost number is not a result on its own. What
makes it one is knowing which build produced it, what was being tried, what it
was compared against, and whether the correctness gate ran — and those get lost
within about a day of the run unless something writes them down at the time.
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "cp_iter_page.html")
PERF   = os.path.join(HERE, "cp_iter_perf.html")

# What counts as worth pointing at. Matches the <<< and !!! markers cp_iterate.sh
# prints, so the two agree about what a result is.
MOVE_PCT = 5.0

# Below this a sample's frame time is dominated by start-up and by the sweep's
# run-to-run spread, and a percentage of it is not a measurement. Flagged rather
# than dropped: sometimes the small samples are the point.
NOISE_FLOOR_MS = 3.0


def read_bench(path):
    """sample -> row, from cp_perf_run.sh's _bench.csv."""
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[r["sample"]] = {
                    "ms_avg": float(r["ms_avg"]),
                    "ms_best": float(r["ms_best"]),
                    "ms_worst": float(r["ms_worst"]),
                    "fps": float(r["fps"]),
                    "wall_s": float(r["wall_s"]),
                    "exit": int(r["exit"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def read_verdict(path):
    """
    Parse cp_compare_frames.py's table.

    Returns (ran, {sample: {...}}). `ran` is False when the file holds no table
    at all, which is the case that matters: the gate failing to run and a sample
    regressing both exit 1, and only the table tells them apart.
    """
    if not os.path.isfile(path):
        return False, {}
    rows = {}
    seen_header = False
    with open(path) as f:
        for line in f:
            if line.startswith("sample "):
                seen_header = True
                continue
            if not seen_header:
                continue
            parts = line.split()
            # sample frame0 worst at verdict...
            if len(parts) < 5:
                continue
            name = parts[0]
            if parts[1] == "-":            # renderheadless: drives its own frames
                rows[name] = {"verdict": "missing"}
                continue
            try:
                rows[name] = {
                    "frame0": int(parts[1]),
                    "worst": int(parts[2]),
                    "at": int(parts[3]),
                    "verdict": "REGRESSED" if parts[4] == "REGRESSED" else parts[4],
                }
            except ValueError:
                continue
    return bool(rows), rows


def read_frame_diffs(frameroot, label, ref, tol, cache):
    """
    ({sample: [differing pixels per frame]}, {sample: how to address a frame}).

    The same ffmpeg pass cp_perf_report.py uses, imported rather than
    reimplemented so the two pages cannot disagree about what a differing pixel
    is. Cached per iteration, because the frames it reads never change once the
    iteration has been rendered.
    """
    if os.path.isfile(cache):
        try:
            c = json.load(open(cache))
            if isinstance(c, dict) and "diffs" in c:
                return c["diffs"], c.get("frames", {})
        except json.JSONDecodeError:
            pass

    sys.path.insert(0, HERE)
    try:
        from cp_perf_report import frame_pattern, frame_size, frames_of, diff_series
    except ImportError as e:
        print("no per-frame differences (%s)" % e, file=sys.stderr)
        return {}, {}

    test_dir = os.path.join(frameroot, label)
    ref_dir = os.path.join(frameroot, ref)
    if not os.path.isdir(test_dir) or not os.path.isdir(ref_dir):
        return {}, {}

    out, meta = {}, {}
    for sample in sorted(os.listdir(test_dir)):
        if sample.startswith("_") or not os.path.isdir(os.path.join(test_dir, sample)):
            continue
        rp, rn = frame_pattern(ref_dir, sample)
        tp, tn = frame_pattern(test_dir, sample)
        if not rp or not tp:
            continue
        names = frames_of(test_dir, sample)
        w, h = frame_size(os.path.join(test_dir, sample, names[0]))
        if not w or not h:
            continue
        series = diff_series(rp, tp, tol, w * h)
        if series:
            # diff_series returns [[frame, pixels], ...]; only the pixels vary.
            out[sample] = [p for _, p in series]
        # How the page addresses one frame. Recorded rather than derived in the
        # browser so there is one implementation of the naming, here, next to
        # the one that reads it.
        m = re.match(r"(.*?)(\d+)(\.(?:ppm|png))$", names[0])
        if m:
            meta[sample] = {
                "prefix": m.group(1), "digits": len(m.group(2)),
                "ext": m.group(3), "count": min(len(names), rn or len(names)),
                "width": w, "height": h,
            }
    try:
        with open(cache, "w") as f:
            json.dump({"diffs": out, "frames": meta}, f)
    except OSError:
        pass
    return out, meta


def convert_frames_to_png(root, label, meta):
    """
    Convert an iteration's stored frames from PPM to PNG, in place, and drop
    the PPM.

    The samples write PPM and cp_compare_frames.py reads it, but a browser
    cannot display it, so the inspector needs PNG. Converting in place rather
    than exporting a second copy means there is still one set of frames: the
    comparator reads PNG perfectly well (cp_compare.read_image goes through
    Pillow), the ffmpeg difference pass takes either, and the page can point an
    <img> straight at the file the gate just judged.

    It also costs less rather than more. A 1280x720 PPM is 2.7 MB whatever it
    holds; the same frame as PNG is 150 KB to 2 MB depending on the content, so
    a sweep shrinks by roughly four times.

    PNG because nothing here may be lossy — the page exists to judge whether a
    pixel is wrong, and a JPEG artefact cannot be told apart from one. That the
    conversion is bit-exact was checked rather than assumed, by decoding both
    forms and comparing the pixels.

    Frames keep their names, so frame_pattern() still derives the same
    sequence. The PPM is removed only after its PNG exists and is non-empty.
    """
    src_root = os.path.join(root, label)
    if not os.path.isdir(src_root) or not shutil.which("ffmpeg"):
        if not shutil.which("ffmpeg"):
            print("no ffmpeg; leaving frames as ppm", file=sys.stderr)
        return 0, 0

    converted = removed = 0
    for sample, m in sorted(meta.items()):
        d = os.path.join(src_root, sample)
        if not os.path.isdir(d):
            continue
        # What is actually on disk, not what the caller's metadata says. The
        # reference is converted once and then passed this same metadata by
        # every later iteration, so without this it retries a conversion that
        # has already happened and ffmpeg complains once per sample about a
        # pattern matching no files.
        if not any(f.endswith(".ppm") for f in os.listdir(d)):
            continue
        pat = os.path.join(d, "%s%%0%dd.ppm" % (m["prefix"], m["digits"]))
        r = subprocess.run(
            # -start_number on the output as well as the input. The image2
            # muxer numbers its output from 1 whatever the input started at, so
            # without this every frame is written one higher than it was read:
            # the pixels are right and every frame index is off by one, which
            # shows up as a sample reporting its worst frame at 60 of 0..59.
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-start_number", "0", "-i", pat, "-start_number", "0",
             os.path.join(d, "%s%%0%dd.png" % (m["prefix"], m["digits"]))],
            capture_output=True, timeout=3600)
        if r.returncode != 0:
            print("png conversion failed for %s/%s: %s"
                  % (label, sample, r.stderr.decode()[:200]), file=sys.stderr)
            continue
        converted += 1
        for f in sorted(os.listdir(d)):
            if not f.endswith(".ppm"):
                continue
            png = os.path.join(d, f[:-4] + ".png")
            if os.path.isfile(png) and os.path.getsize(png) > 0:
                os.remove(os.path.join(d, f))
                removed += 1
    return converted, removed


def scan_frames(root, label):
    """
    {sample: how to address a frame} by looking at what is on disk.

    `record` gets this from the pass that computes the differences; this is for
    a directory that has frames and nothing else — an iteration from before the
    json existed, or a reference — so that it can still be converted.
    """
    d = os.path.join(root, label)
    if not os.path.isdir(d):
        return {}
    meta = {}
    for sample in sorted(os.listdir(d)):
        sd = os.path.join(d, sample)
        if sample.startswith("_") or not os.path.isdir(sd):
            continue
        names = sorted(f for f in os.listdir(sd) if f.endswith((".ppm", ".png")))
        if not names:
            continue
        m = re.match(r"(.*?)(\d+)(\.(?:ppm|png))$", names[0])
        if m:
            meta[sample] = {"prefix": m.group(1), "digits": len(m.group(2)),
                            "ext": m.group(3), "count": len(names)}
    return meta


def cmd_convert(args):
    """Convert stored frames to png, in place, for one label or for all."""
    root = args.root
    labels = args.labels or sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    tot_c = tot_r = 0
    for label in labels:
        meta = scan_frames(root, label)
        if not meta:
            continue
        c, rm = convert_frames_to_png(root, label, meta)
        tot_c += c
        tot_r += rm
        if c:
            print("%-14s %2d sample(s) converted, %4d ppm removed" % (label, c, rm))
    print("%d sample directories converted, %d ppm removed" % (tot_c, tot_r))


def git_info(mesa, commit):
    subject = ""
    if commit:
        try:
            subject = subprocess.run(
                ["git", "-C", mesa, "log", "-1", "--format=%s", commit],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
    return subject


def cmd_record(args):
    root = args.root
    out = os.path.join(root, args.label)
    if not os.path.isdir(out):
        sys.exit("no such iteration: %s" % out)

    commit, dirty = "", []
    cpath = os.path.join(out, "_commit.txt")
    if os.path.isfile(cpath):
        with open(cpath) as f:
            lines = f.read().splitlines()
        if lines:
            commit = lines[0].strip()
        # `git diff --stat` lines look like " path/to/file.c | 28 +++++"
        for l in lines[1:]:
            m = re.match(r"\s+(\S+)\s+\|", l)
            if m:
                dirty.append(os.path.basename(m.group(1)))

    cost = read_bench(os.path.join(out, "bench", "_bench.csv"))
    gate_ran, verdict = read_verdict(os.path.join(out, "verdict.txt"))

    # The comparison. Recomputed from the two bench CSVs rather than parsed back
    # out of delta.txt, so the numbers here cannot drift from the numbers there.
    against = args.against
    delta, prev_verdict = {}, {}
    if against:
        prev = read_bench(os.path.join(root, against, "bench", "_bench.csv"))
        _, prev_verdict = read_verdict(os.path.join(root, against, "verdict.txt"))
        for s, now in cost.items():
            if s not in prev:
                continue
            a, b = prev[s]["ms_avg"], now["ms_avg"]
            delta[s] = {
                "before": a, "after": b, "ms": b - a,
                "pct": ((b - a) / a * 100.0) if a else 0.0,
                "noisy": max(a, b) < NOISE_FLOOR_MS,
            }

    total = sum(v["ms_avg"] for v in cost.values())
    prev_total = sum(d["before"] for d in delta.values()) if delta else 0.0
    now_total = sum(d["after"] for d in delta.values()) if delta else 0.0

    wins = sorted([dict(sample=s, **d) for s, d in delta.items()
                   if d["pct"] <= -MOVE_PCT], key=lambda d: d["pct"])
    regressions = sorted([dict(sample=s, **d) for s, d in delta.items()
                          if d["pct"] >= MOVE_PCT], key=lambda d: -d["pct"])

    # Whichever samples changed their pass/fail verdict. This is the line that
    # decides whether an iteration is allowed to count, so it is computed rather
    # than asserted.
    changed = []
    for s, v in verdict.items():
        was = prev_verdict.get(s, {}).get("verdict")
        if was and was != v.get("verdict"):
            changed.append({"sample": s, "was": was, "now": v.get("verdict")})

    desc = args.desc or ""
    dpath = os.path.join(out, "description.txt")
    if not desc and os.path.isfile(dpath):
        desc = open(dpath).read().strip()

    # Per-frame differences, which is the one series that exists nowhere else:
    # it takes an ffmpeg pass over the stored frames to produce. Per-frame
    # *times* are deliberately not copied in here — cp_perf_run.sh already
    # wrote them to bench/<sample>.csv and the detail page reads those
    # directly, so there is one copy of them rather than two that can drift.
    frame_diff, frame_meta = {}, {}
    calib_diff = {}
    if not args.no_diffs:
        frame_diff, frame_meta = read_frame_diffs(
            root, args.label, args.ref, args.tol,
            os.path.join(out, "_frame_diffs.json"))

        # The same series for llvmpipe, against the same reference.
        #
        # TESTING.md is emphatic about why: llvmpipe is mature, shares lavapipe
        # as its frontend, and still differs from NVIDIA by tens of thousands of
        # pixels across the set. A residual both software renderers have is what
        # software rasterization costs; a residual only cudapipe has is a
        # defect. Without the second line on the chart there is no way to tell
        # which of the two a number is, and several questions have been settled
        # by exactly that distinction — gltfscenerendering grows fourfold over
        # its orbit, and llvmpipe grows further on the same frames.
        #
        # It is cached in the calibration driver's own directory, because it
        # does not depend on the iteration and would otherwise be recomputed for
        # every one of them.
        if args.calibrate and os.path.isdir(os.path.join(root, args.calibrate)):
            calib_diff, _ = read_frame_diffs(
                root, args.calibrate, args.ref, args.tol,
                os.path.join(root, args.calibrate,
                             "_frame_diffs_vs_%s.json" % args.ref))

    # Frames to PNG so a browser can show them, in place, dropping the PPM.
    # The reference is converted too, once — it is shared by every iteration,
    # since NVIDIA's output does not change when cudapipe does.
    if frame_meta and not args.no_png:
        for who in (args.label, args.ref):
            c, rm = convert_frames_to_png(root, who, frame_meta)
            if c:
                print("converted %d sample(s) of %s to png, removed %d ppm"
                      % (c, who, rm))
        # The recorded extension is now stale for anything just converted.
        for m in frame_meta.values():
            if m.get("ext") == ".ppm":
                m["ext"] = ".png"

    rec = {
        "label": args.label,
        "description": desc,
        "notes": args.note or [],
        "driver": args.driver,
        # Two counts: what was stored and compared, and what was timed. They
        # differ on purpose — see cp_iterate.sh.
        "frames": args.frames,
        "bench_frames": args.bench_frames,
        "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
        "mtime": os.path.getmtime(os.path.join(out, "bench", "_bench.csv"))
                 if os.path.isfile(os.path.join(out, "bench", "_bench.csv")) else 0,
        "commit": commit,
        "commit_subject": git_info(args.mesa, commit),
        "dirty": dirty,
        "against": against,
        "total_ms": round(total, 2),
        # Totals over the samples both iterations have, so the percentage is not
        # distorted by one of them having timed a different set.
        "total_pct": round((now_total - prev_total) / prev_total * 100.0, 2)
                     if prev_total else None,
        "cost": cost,
        "delta": delta,
        "tolerance": args.tol,
        "reference": args.ref,
        # sample -> [differing pixels per frame]. Frame times are not here; see
        # above — the page reads bench/<sample>.csv for those.
        "frame_diff": frame_diff,
        # sample -> how to build the path of frame n, so the inspector can point
        # an <img> at the same file the correctness gate read.
        "frames_meta": frame_meta,
        # The calibration renderer and its differences against the same
        # reference. What makes a residual readable as cost or as defect.
        "calibration": args.calibrate if calib_diff else "",
        "calib_diff": calib_diff,
        "wins": wins,
        "regressions": regressions,
        "correctness": {
            "gate_ran": gate_ran,
            "samples": verdict,
            "changed_vs_against": changed,
        },
    }

    path = os.path.join(out, "iteration.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    print("-> %s  (%.2f ms total%s, gate %s)"
          % (path, total,
             ", %+.1f%% vs %s" % (rec["total_pct"], against)
             if rec["total_pct"] is not None else "",
             "ran" if gate_ran else "DID NOT RUN"))


def cmd_page(args):
    root = args.root
    # The pages sit at the root of build/iter, beside the iterations they
    # describe, so that perf.html?iter=LABEL can reach LABEL/ with a relative
    # path and the whole tree can be served as-is.
    out = args.out or root
    os.makedirs(out, exist_ok=True)

    iters = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "iteration.json")
        if os.path.isfile(p):
            try:
                iters.append(json.load(open(p)))
            except json.JSONDecodeError as e:
                print("skipping %s: %s" % (p, e), file=sys.stderr)

    # Chronological by when the benchmark actually ran, not by label, so the
    # history reads in the order it happened.
    iters.sort(key=lambda r: (r.get("mtime") or 0, r.get("label", "")))

    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "iterations": iters,
    }
    with open(os.path.join(out, "iterations.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    for src, name in ((VIEWER, "iterations.html"), (PERF, "perf.html")):
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, name))
        else:
            print("warning: viewer missing at %s" % src, file=sys.stderr)

    print("%d iterations -> %s" % (len(iters), os.path.join(out, "iterations.html")))
    for r in iters:
        pct = r.get("total_pct")
        print("  %-14s %8.2f ms %10s  %s"
              % (r["label"], r.get("total_ms", 0),
                 ("%+.1f%%" % pct) if pct is not None else "",
                 (r.get("description") or "")[:52]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/git/Vulkan/build/iter"))
    ap.add_argument("--mesa", default=os.path.expanduser("~/mesa"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("label")
    r.add_argument("--desc", default="")
    r.add_argument("--against", default="")
    r.add_argument("--note", action="append")
    r.add_argument("--driver", default="cudapipe")
    r.add_argument("--frames", type=int, default=60,
                   help="frames stored and compared by the correctness gate")
    r.add_argument("--bench-frames", type=int, default=600,
                   help="frames timed by the benchmark pass")
    r.add_argument("--ref", default="nvidia",
                   help="frame directory the differences are taken against")
    r.add_argument("--tol", type=int, default=8,
                   help="per-channel tolerance, matching cp_compare_frames.py")
    r.add_argument("--no-diffs", action="store_true",
                   help="skip the ffmpeg pass over the stored frames")
    r.add_argument("--calibrate", default="llvmpipe",
                   help="second renderer to difference against the same "
                        "reference, for calibration; '' to skip")
    r.add_argument("--no-png", action="store_true",
                   help="leave frames as ppm instead of converting them to png")
    r.set_defaults(func=cmd_record)

    p = sub.add_parser("page")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_page)

    c = sub.add_parser("convert",
                       help="ppm -> png in place for stored frames, dropping the ppm")
    c.add_argument("labels", nargs="*",
                   help="iteration labels; default every directory under the root")
    c.set_defaults(func=cmd_convert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
