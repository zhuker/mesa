#!/usr/bin/env python3
"""Repair the Nsight Systems agent skill pack's stale content hashes.

    cp_nsys_skill_fix.py [SKILL_DIR]           # check only, exits 1 if stale
    sudo cp_nsys_skill_fix.py --repair [DIR]   # rewrite manifest.json

Nsight Systems 2026.4.1 ships a skill pack whose `manifest.json` records
content hashes computed against a different build of its own content: all 496
entries disagree with the files beside them. The pack verifies itself
before serving anything out of its packaged indexes, so the two commands that
read those indexes fail:

    search-docs     content hash mismatch: SKILL.md
    lookup-recipes  content hash mismatch: SKILL.md

and with them goes the only sanctioned route to the reference corpus --
`references/notes/llm-analysis-pitfalls.md`, `sql_query_tips.md`,
`references/curated/investigation_methodology.md` and the rest. SKILL.md
forbids reading those by hand ("Do not search the filesystem"), so a broken
search-docs makes them unreachable rather than merely inconvenient.

Everything that talks to the installed `nsys` or to a report is unaffected and
works without this: doctor, inspect-cli, report-context, report-fact,
report-query, report-doctor.

This rewrites only the digests, to match the bytes NVIDIA shipped. It changes
no content -- it restores the manifest's internal consistency with its own
files. Verified against dpkg -V, which reports the files themselves unmodified.

Re-run after every Nsight Systems upgrade: the package replaces manifest.json,
and the next release may or may not have fixed this. Check first; if it exits
0, the bug is gone and nothing needs doing.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys

DEFAULT = "/opt/nvidia/nsight-systems/2026.4.1/skills/nsight-systems"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill_dir", nargs="?", default=DEFAULT)
    ap.add_argument("--repair", action="store_true",
                    help="rewrite manifest.json (needs write access, so sudo)")
    args = ap.parse_args()

    root = pathlib.Path(args.skill_dir)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        sys.exit("no manifest.json under %s -- is that a skill pack?" % root)

    data = json.loads(manifest.read_text())
    hashes = data.get("content_hashes")
    if not hashes:
        sys.exit("manifest.json has no content_hashes -- format changed, "
                 "re-read it before trusting this script")

    stale, missing = {}, []
    for rel, recorded in hashes.items():
        f = root / rel
        if not f.is_file():
            missing.append(rel)
            continue
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual != recorded:
            stale[rel] = actual

    print("%s\n  %d entries, %d stale, %d missing"
          % (root, len(hashes), len(stale), len(missing)))
    for rel in missing:
        print("  missing: %s" % rel)

    if not stale:
        print("  manifest is consistent -- nothing to do")
        return 0

    for rel in sorted(stale)[:3]:
        print("  stale: %s" % rel)
    if len(stale) > 3:
        print("  ... and %d more" % (len(stale) - 3))

    if not args.repair:
        print("\nrun with --repair (as root) to rewrite the digests")
        return 1

    backup = manifest.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(manifest, backup)
        print("  backed up -> %s" % backup.name)
    hashes.update(stale)
    manifest.write_text(json.dumps(data, indent=2))
    print("  repaired %d digests" % len(stale))
    print("\nverify with:  <bundled python> scripts/nsys_skill_cli.py "
          "search-docs --query test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
