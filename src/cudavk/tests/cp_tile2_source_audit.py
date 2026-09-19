#!/usr/bin/env python3
"""CPU-only source audit. This is not compiler-artifact or runtime equality."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "src/cudavk/kernels"
BASE = "69a58a486d4bf74ba8d77089894d17bed1ce598c"


def body(text, name):
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", text)
    assert match, name
    begin = text.index("{", match.start())
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return re.sub(r"\s+", " ", text[begin:end]).strip()


def preprocess(directory, source, tile):
    return subprocess.check_output([
        os.environ.get("CC", "cc"), "-E", "-P", "-x", "c",
        "-D__CUDACC__=1", "-DCP_VS_TILE2=" + str(tile),
        "-I" + str(directory), str(directory / source)], text=True)


def main():
    sources = {
        "cp_rasterize.cu": ["clip_dst", "clip_retire", "cp_clip_one", "setup_triangle", "cp_resolve_visbuf"],
        "cp_fs_inline.c": ["cp_interp_setup", "cp_interp_pixel_prepared", "cp_resolve_seg_range", "cp_primitive_base"],
    }
    with tempfile.TemporaryDirectory(prefix="cp-tile2-source-") as tmp:
        old = Path(tmp)
        for path in KERNELS.iterdir():
            if path.suffix not in (".h", ".c", ".cu"):
                continue
            tracked = subprocess.run(["git", "show", BASE + ":src/cudavk/kernels/" + path.name],
                                     cwd=ROOT, capture_output=True)
            if tracked.returncode == 0:
                (old / path.name).write_bytes(tracked.stdout)
        current = {}
        for source, names in sources.items():
            before = preprocess(old, source, 0)
            off = preprocess(KERNELS, source, 0)
            on = preprocess(KERNELS, source, 1)
            current[source] = on
            for name in names:
                assert body(before, name) == body(off, name), ("OFF source changed", name)
                print("OFF preprocessed function body unchanged:", name)
        # Exhaustive for these known direct access expressions, not a claim of
        # automatic whole-program pointer analysis. New sites need review.
        raster = current["cp_rasterize.cu"]
        interp = current["cp_fs_inline.c"]
        for name in ("setup_triangle", "cp_clip_one", "clip_dst", "clip_retire"):
            text = body(raster, name)
            for pattern in (r"positions\s*\[", r"\bin\s*\+\s*\(size_t\)tri",
                            r"\bout\s*\+\s*\(size_t\)", r"\bv\s*\+\s*\(size_t\)"):
                assert not re.search(pattern, text), ("residual global AoS", name, pattern)
        for name in ("cp_interp_setup", "cp_interp_pixel_prepared"):
            text = body(interp, name)
            assert not re.search(r"positions\s*\[|vs_out\s*\+|\(\(const float4 \*\)vs_out\)\[", text), name
        assert "cp_vs_slot_address" in body(interp, "cp_primitive_base")
        # Classify surviving AoS, rather than blindly forbidding it.
        for name in ("clip_poly", "clip_lerp", "clip_copy"):
            assert body(preprocess(old, "cp_rasterize.cu", 0), name) == body(raster, name)
            print("Residual AoS classified: private polygon scratch:", name)
        assert body(preprocess(old, "cp_rasterize.cu", 0), "cp_resolve_visbuf") == body(raster, "cp_resolve_visbuf")
        print("Residual AoS classified: legacy resolve packed S=16 positions / independent color_stride ABI")
        print("PASS tile2 bounded source audit; NO PTX/LLVM/native artifact equality claim")


if __name__ == "__main__":
    main()
