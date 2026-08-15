#!/usr/bin/env python3
"""Extract rendered frames from a GFXReconstruct capture that never presents.

An offscreen application — HeadlessStreamer is the one this exists for — has no
swapchain, so GFXR counts zero frames and `--screenshots` never fires. What it
does have is a readback: every frame ends with vkCmdCopyImageToBuffer out of the
render target, and gfxrecon-replay's --dump-resources can dump that buffer. This
drives that path end to end.

    cp_gfxr_frames.py index  capture.gfxr                  -> blocks.tsv
    cp_gfxr_frames.py frames blocks.tsv                    -> what was found
    cp_gfxr_frames.py plan   blocks.tsv --frames 0,100,-1  -> dump.json
    cp_gfxr_frames.py replay capture.gfxr dump.json --icd <icd.json> --out DIR
    cp_gfxr_frames.py png    DIR                           -> PNGs beside the .bin

`index` is the slow-ish step (a few seconds for a 2.4 GB capture) and its output
is reusable, so do it once. Everything else reads blocks.tsv.

Why the indirection: --dump-resources addresses commands by block index — the
counter GFXR assigns to every call in the file — and needs, for each dumped
command, the enclosing vkBeginCommandBuffer and the vkQueueSubmit that submits
it. All three come out of the index.

See tests/GFXRECONSTRUCT.md for the capture and replay workflow this fits into.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib

# ---------------------------------------------------------------- index

# Only these blocks are kept. Draw calls are excluded on purpose: a real capture
# has a million of them and none are needed to locate a frame readback.
KEEP = {
    "vkBeginCommandBuffer",
    "vkCmdCopyImageToBuffer", "vkCmdCopyImageToBuffer2",
    "vkCmdCopyImageToBuffer2KHR",
    "vkQueueSubmit", "vkQueueSubmit2", "vkQueueSubmit2KHR",
    "vkCreateImage",
}

RE_HEAD = re.compile(r'"index":(\d+),"function":\{"name":"([A-Za-z0-9_]+)"')
RE_CB = re.compile(r'"commandBuffer":(\d+)')
RE_PCB = re.compile(r'"pCommandBuffers":\[([0-9,]*)\]')
RE_SRC_IMAGE = re.compile(r'"srcImage":(\d+)')
RE_EXTENT = re.compile(r'"imageExtent":\{"width":(\d+),"height":(\d+)')
RE_IMAGE_OUT = re.compile(r'"pImage":(\d+)')
RE_FORMAT = re.compile(r'"format":"(VK_FORMAT_[A-Z0-9_]+)"')


def find_tool(name, override=None):
    path = override or shutil.which(name)
    if not path:
        sys.exit("%s not found on PATH; pass --%s" % (name, name))
    return path


def cmd_index(args):
    """Stream the capture through gfxrecon-convert and write a compact TSV.

    Piped rather than converted to a file: the JSONL for a 2.4 GB capture is
    tens of GB, and none of it is worth keeping.
    """
    convert = find_tool("gfxrecon-convert", args.gfxrecon_convert)
    out = args.output or os.path.splitext(args.capture)[0] + ".blocks.tsv"

    proc = subprocess.Popen(
        [convert, "--output", "stdout", "--format", "jsonl", args.capture],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        bufsize=1 << 20)

    kept = 0
    with open(out, "w") as fh:
        for line in proc.stdout:
            m = RE_HEAD.search(line)
            if not m:
                continue
            name = m.group(2)
            if name not in KEEP:
                continue

            if name.startswith("vkQueueSubmit"):
                # vkQueueSubmit lists pCommandBuffers; vkQueueSubmit2 nests one
                # commandBuffer per pCommandBufferInfos entry.
                ids = ",".join(x.group(1) for x in RE_PCB.finditer(line))
                if not ids:
                    ids = ",".join(x.group(1) for x in RE_CB.finditer(line))
                info = ""
            elif name == "vkCreateImage":
                ids = ""
                img = RE_IMAGE_OUT.search(line)
                fmt = RE_FORMAT.search(line)
                ext = re.search(r'"extent":\{"width":(\d+),"height":(\d+)', line)
                info = "id=%s fmt=%s w=%s h=%s" % (
                    img.group(1) if img else "?",
                    fmt.group(1) if fmt else "?",
                    ext.group(1) if ext else "?",
                    ext.group(2) if ext else "?")
            else:
                cb = RE_CB.search(line)
                ids = cb.group(1) if cb else ""
                src = RE_SRC_IMAGE.search(line)
                ext = RE_EXTENT.search(line)
                info = "img=%s w=%s h=%s" % (
                    src.group(1) if src else "?",
                    ext.group(1) if ext else "?",
                    ext.group(2) if ext else "?")

            fh.write("%s\t%s\t%s\t%s\n" % (m.group(1), name, ids, info))
            kept += 1

    proc.wait()
    if proc.returncode != 0:
        sys.exit("gfxrecon-convert exited %d" % proc.returncode)
    print("%s: %d blocks indexed" % (out, kept))
    return 0


# ---------------------------------------------------------------- shared


class Index:
    """blocks.tsv, plus the lookups plan/frames need."""

    def __init__(self, path):
        self.rows = []
        for line in open(path):
            idx, name, ids, info = line.rstrip("\n").split("\t")
            self.rows.append((int(idx), name, ids, info))
        self.pos = {r[0]: k for k, r in enumerate(self.rows)}
        self.images = {}
        for _, name, _, info in self.rows:
            if name != "vkCreateImage":
                continue
            kv = dict(p.split("=", 1) for p in info.split() if "=" in p)
            if "id" in kv:
                self.images[kv["id"]] = kv

    def copies(self):
        """Every image->buffer readback, in capture order."""
        out = []
        for idx, name, ids, info in self.rows:
            if not name.startswith("vkCmdCopyImageToBuffer"):
                continue
            kv = dict(p.split("=", 1) for p in info.split() if "=" in p)
            out.append({"index": idx, "cb": ids, "img": kv.get("img", "?"),
                        "w": kv.get("w", "?"), "h": kv.get("h", "?")})
        return out

    def enclosing(self, block_index, cb):
        """The vkBeginCommandBuffer this command records into, and the
        vkQueueSubmit that submits that command buffer."""
        k = self.pos[block_index]
        begin = None
        for j in range(k, -1, -1):
            if self.rows[j][1] == "vkBeginCommandBuffer" and self.rows[j][2] == cb:
                begin = self.rows[j][0]
                break
        submit = None
        for j in range(k, len(self.rows)):
            if self.rows[j][1].startswith("vkQueueSubmit") and \
                    cb in self.rows[j][2].split(","):
                submit = self.rows[j][0]
                break
        return begin, submit


def dominant_image(copies):
    """The image most readbacks come from — the frame output, in practice."""
    counts = {}
    for c in copies:
        counts[c["img"]] = counts.get(c["img"], 0) + 1
    return max(counts, key=counts.get) if counts else None


def cmd_frames(args):
    idx = Index(args.blocks)
    copies = idx.copies()
    if not copies:
        sys.exit("no vkCmdCopyImageToBuffer in the index — this capture does "
                 "not read its render target back; dump draw calls instead")

    by_img = {}
    for c in copies:
        by_img.setdefault(c["img"], []).append(c)

    print("%d readbacks, from %d image(s):\n" % (len(copies), len(by_img)))
    main = dominant_image(copies)
    for img, cs in sorted(by_img.items(), key=lambda kv: -len(kv[1])):
        meta = idx.images.get(img, {})
        print("  image %-8s %5d frames  %sx%s  %s%s" % (
            img, len(cs), cs[0]["w"], cs[0]["h"],
            meta.get("fmt", "format unknown"),
            "   <- frame output" if img == main else ""))
    print("\nfirst readback at block %d, last at block %d"
          % (copies[0]["index"], copies[-1]["index"]))
    return 0


def select(copies, spec):
    """'0,5,-1' | '0-9' | '::100' | 'all' -> a list of frame positions."""
    if spec in ("all", ":"):
        return list(range(len(copies)))
    picked = []
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("::"):
            picked += list(range(0, len(copies), int(part[2:])))
        elif "-" in part[1:]:
            lo, hi = part.rsplit("-", 1)
            picked += list(range(int(lo), int(hi) + 1))
        else:
            n = int(part)
            picked.append(n if n >= 0 else len(copies) + n)
    seen, out = set(), []
    for n in picked:
        if 0 <= n < len(copies) and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def cmd_plan(args):
    idx = Index(args.blocks)
    copies = idx.copies()
    if not copies:
        sys.exit("no vkCmdCopyImageToBuffer in the index")

    want = args.image or dominant_image(copies)
    copies = [c for c in copies if c["img"] == want]
    chosen = select(copies, args.frames)
    if not chosen:
        sys.exit("--frames %r selected nothing (capture has %d)"
                 % (args.frames, len(copies)))

    begins, transfers, submits = [], [], []
    for n in chosen:
        c = copies[n]
        begin, submit = idx.enclosing(c["index"], c["cb"])
        if begin is None or submit is None:
            print("  frame %d (block %d): no enclosing %s, skipped"
                  % (n, c["index"],
                     "vkBeginCommandBuffer" if begin is None else "vkQueueSubmit"),
                  file=sys.stderr)
            continue
        begins.append(begin)
        transfers.append([c["index"]])
        submits.append(submit)

    doc = {
        "DumpResourcesOptions": {"ImageFormat": "png"},
        "BeginCommandBuffer": begins,
        "Transfer": transfers,
        "QueueSubmit": submits,
    }
    with open(args.output, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("%s: %d frame(s) from image %s, blocks %s"
          % (args.output, len(begins), want,
             ",".join(str(t[0]) for t in transfers[:8])
             + (" ..." if len(transfers) > 8 else "")))
    return 0


def cmd_replay(args):
    replay = find_tool("gfxrecon-replay", args.gfxrecon_replay)
    os.makedirs(args.out, exist_ok=True)
    cmd = [replay, "-m", "remap"]
    if not args.keep_unsupported:
        # Captures request extensions a software driver does not expose
        # (VK_KHR_video_maintenance1 in the HeadlessStreamer capture); without
        # this, vkCreateDevice fails outright.
        cmd.append("--remove-unsupported")
    cmd += ["--log-file", os.path.join(args.out, "replay.log"),
            "--dump-resources", args.plan,
            "--dump-resources-dir", args.out,
            args.capture]

    env = dict(os.environ)
    if args.icd:
        env["VK_DRIVER_FILES"] = args.icd

    print("+ " + " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    # A driver that dies mid-replay still leaves the frames it managed, so this
    # is reported rather than fatal.
    if rc != 0:
        print("gfxrecon-replay exited %d — see %s"
              % (rc, os.path.join(args.out, "replay.log")), file=sys.stderr)
    return 0


# ---------------------------------------------------------------- png

# Byte order in the dumped buffer, per Vulkan format, as (r, g, b) offsets.
SWIZZLE = {
    "VK_FORMAT_B8G8R8A8_UNORM": (2, 1, 0), "VK_FORMAT_B8G8R8A8_SRGB": (2, 1, 0),
    "VK_FORMAT_R8G8B8A8_UNORM": (0, 1, 2), "VK_FORMAT_R8G8B8A8_SRGB": (0, 1, 2),
}


def write_png(path, data, w, h, swz):
    ro, go, bo = swz
    raw = bytearray()
    for y in range(h):
        row = data[y * w * 4:(y + 1) * w * 4]
        raw.append(0)                       # PNG per-row filter: none
        px = bytearray(w * 3)
        px[0::3] = row[ro::4]
        px[1::3] = row[go::4]
        px[2::3] = row[bo::4]
        raw += px

    def chunk(tag, payload):
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body)))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
                 + chunk(b"IEND", b""))


def manifest_regions(dump_dir):
    """filename -> (width, height, format), from the manifest replay writes.

    Both suffixes are accepted: vulkan_dump_resources.md documents "_rd.json",
    the tool actually writes "_dr.json".

    Returns {} if the manifest is missing or still being written — replay
    appends to it as it goes, so mid-run it is not valid JSON.
    """
    found = [f for f in os.listdir(dump_dir)
             if f.endswith("_dr.json") or f.endswith("_rd.json")]
    if not found:
        return {}
    try:
        doc = json.load(open(os.path.join(dump_dir, found[0])))
    except (ValueError, OSError):
        return {}

    out = {}
    for block in doc:
        for entry in block.get("transferCommands", []) or []:
            if not entry:
                continue
            src = entry.get("parameters", {}).get("srcImage", {})
            fmt = src.get("format", "")
            for region in entry["parameters"].get("regions", []) or []:
                ext = region.get("imageExtent", {})
                name = os.path.basename(region.get("file", ""))
                if name:
                    out[name] = (ext.get("width"), ext.get("height"), fmt)
    return out


def cmd_png(args):
    regions = manifest_regions(args.dir)
    size = None
    if args.size:
        w, h = args.size.lower().split("x")
        size = (int(w), int(h))

    bins = sorted(f for f in os.listdir(args.dir) if f.endswith(".bin"))
    if not bins:
        sys.exit("no .bin files in %s" % args.dir)

    done = 0
    for name in bins:
        src = os.path.join(args.dir, name)
        dst = src[:-4] + ".png"
        if os.path.exists(dst) and not args.force:
            continue

        w, h, fmt = regions.get(name, (None, None, ""))
        if size:
            w, h = size
        if not w or not h:
            print("  %s: no extent in the manifest; pass --size WxH" % name,
                  file=sys.stderr)
            continue

        swz = SWIZZLE.get(fmt)
        if swz is None:
            if args.order:
                swz = (0, 1, 2) if args.order == "rgba" else (2, 1, 0)
            else:
                # Colour swaps read as plausible images, so guess loudly.
                print("  %s: unknown format %r, assuming BGRA (--order to "
                      "override)" % (name, fmt), file=sys.stderr)
                swz = (2, 1, 0)

        data = open(src, "rb").read()
        if len(data) < w * h * 4:
            print("  %s: %d bytes, need %d for %dx%d — skipped"
                  % (name, len(data), w * h * 4, w, h), file=sys.stderr)
            continue

        write_png(dst, data, w, h, swz)
        done += 1
        print("  %s  %dx%d" % (os.path.basename(dst), w, h))

    print("%d PNG(s) written to %s" % (done, args.dir))
    return 0


# ---------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build blocks.tsv from a capture")
    p.add_argument("capture")
    p.add_argument("-o", "--output")
    p.add_argument("--gfxrecon-convert")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("frames", help="report the readbacks found in an index")
    p.add_argument("blocks")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("plan", help="write a --dump-resources json")
    p.add_argument("blocks")
    p.add_argument("-o", "--output", default="dump.json")
    p.add_argument("--frames", default="0",
                   help="'0,5,-1' | '0-9' | '::100' | 'all' (default: 0)")
    p.add_argument("--image", help="source image id (default: the commonest)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("replay", help="replay the capture, dumping the plan")
    p.add_argument("capture")
    p.add_argument("plan")
    p.add_argument("--icd", help="value for VK_DRIVER_FILES")
    p.add_argument("--out", default="dumps")
    p.add_argument("--keep-unsupported", action="store_true",
                   help="do not pass --remove-unsupported")
    p.add_argument("--gfxrecon-replay")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("png", help="convert dumped .bin readbacks to PNG")
    p.add_argument("dir")
    p.add_argument("--size", help="WxH, if the manifest is unreadable")
    p.add_argument("--order", choices=("bgra", "rgba"))
    p.add_argument("-f", "--force", action="store_true")
    p.set_defaults(func=cmd_png)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
