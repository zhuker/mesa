#!/usr/bin/env python3
"""Generate/validate ordered-clip SPIR-V fixtures. CPU only; never runs an ICD."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--glslang", default="glslangValidator")
    parser.add_argument("--spirv-dis", default="spirv-dis")
    parser.add_argument("--spirv-val", default="spirv-val")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vertex = args.output / "clip.vert"
    vertex.write_text("#version 450\nvoid main() { gl_Position = vec4(0, 0, 0.5, 1); }\n")
    sources = [(vertex, None)]
    for mask in range(8):
        lines = ["#version 450", "layout(location=0) out vec4 color;"]
        if mask & 4:
            lines += ["layout(early_fragment_tests) in;"]
        lines += ["void main() {", "  color = vec4(0.25, 0.5, 0.75, 1.0);"]
        if mask & 1:
            lines += ["  gl_FragDepth = 0.5;"]
        if mask & 2:
            lines += ["  gl_SampleMask[0] = 1;"]
        lines += ["}"]
        source = args.output / f"clip-{mask}.frag"
        source.write_text("\n".join(lines) + "\n")
        sources.append((source, mask))
    records = []
    for source, mask in sources:
        spv = source.with_suffix(source.suffix + ".spv")
        dis = source.with_suffix(source.suffix + ".spvasm")
        run(args.glslang, "-V", "--target-env", "vulkan1.1", "-o", str(spv), str(source))
        run(args.spirv_val, "--target-env", "vulkan1.1", str(spv))
        run(args.spirv_dis, str(spv), "-o", str(dis))
        if mask is None:
            continue
        text = dis.read_text()
        actual = (bool(re.search(r"\bBuiltIn FragDepth\b", text)) |
                  (bool(re.search(r"\bBuiltIn SampleMask\b", text)) << 1) |
                  (bool(re.search(r"\bOpExecutionMode\s+%\S+\s+EarlyFragmentTests\b", text)) << 2))
        if actual != mask:
            raise RuntimeError(f"{source}: expected SPIR-V mask {mask}, got {actual}")
        records.append({"mask": mask, "fragment": spv.name, "assembly": dis.name})
    (args.output / "manifest.json").write_text(json.dumps({
        "vertex": "clip.vert.spv", "fragments": records,
        "scope": "CPU-generated and spirv-val/dis checked; not executed",
    }, indent=2) + "\n")
    print(f"PASS: VS and all 8 FS masks validated in {args.output}")


if __name__ == "__main__":
    main()
