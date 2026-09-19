#!/usr/bin/env python3
"""Compile the actual retry helpers with CUDA mocks; no GPU/context is used."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "nir_to_ptx/cp_nir_to_llvm.c").read_text()
start = source.index("static uint8_t\ncp_vs_fetch_resource_verdict(")
end = source.index("\n#endif", start)
# Wiring proof: a single caller and only the private fused VS execution.
assert source.count("cp_vs_fetch_retry_reg_cap(") == 2
attach = source[source.index("static void\ncp_vs_attach_fused_fetch("):]
attach = attach[:attach.index("\n#ifdef CP_HAVE_FS_INLINE_BC")]
assert "&fused->exec[CP_SHADER_EXEC_VS_FETCH]" in attach
assert "cp_vs_fetch_retry_reg_cap(classic, fx," in attach
assert "if (verdict == CP_VS_FETCH_OCCUPANCY &&" in attach
with tempfile.TemporaryDirectory(prefix="cp-vfetch-regcap-") as tmp:
    tmp = Path(tmp)
    (tmp / "retry.inc").write_text(source[start:end])
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
                    "-Werror", "-I" + str(root), "-I" + str(tmp),
                    "-I" + os.environ.get("CUDA_PATH", "/usr/local/cuda") + "/include",
                    str(root / "tests/cp_vfetch_regcap_retry_test.c"),
                    "-o", str(tmp / "test")], check=True)
    subprocess.run([str(tmp / "test")], check=True)
