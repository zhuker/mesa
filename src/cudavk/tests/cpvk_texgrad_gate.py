#!/usr/bin/env python3
"""textureGrad() on both fragment paths, because only one of them was broken.

cpvk_texgrad passes at HEAD if it is run with nothing set, and that is not
good news: `cp_hardware_texture_shader_eligible()` admits `nir_texop_txd` and
its shader is small enough to qualify, so the whole test goes to the CUDA
texture-object path and never reaches the software sampler where the defect
was.  The capture that found the defect does not qualify -- its shaders sample
cubes, arrays and 3D grids -- so the path that renders it is the one this gate
exists to run.

Two arms, and the gate selects both rather than inheriting either:

  * hardware textures, the default state, with CUDAVK_NO_TEXTURE_CACHE popped;
  * the software sampler, with CUDAVK_NO_TEXTURE_CACHE=1.

Both must pass.  The second is the one that fails at HEAD, with every gradient
pass returning the unsupported-texture-op constant 0 0 0 255, which the test
names in its own output.  A gate that cannot fail is not covering anything.
"""
import os
import subprocess
import sys

ARMS = (
    ("hardware textures", None),
    ("software sampler", "1"),
)


def run(exe, no_texture_cache):
    env = os.environ.copy()
    env.pop("CUDAVK_NO_TEXTURE_CACHE", None)
    if no_texture_cache is not None:
        env["CUDAVK_NO_TEXTURE_CACHE"] = no_texture_cache
    return subprocess.run([exe], env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    exe = sys.argv[1]
    failed = []
    for name, value in ARMS:
        run_result = run(exe, value)
        print("=== %s ===" % name)
        sys.stdout.write(run_result.stdout)
        sys.stderr.write(run_result.stderr)
        if run_result.returncode or "\nPASS\n" not in run_result.stdout:
            failed.append(name)
        # The placeholder is the specific regression this gate is about, so
        # name it even if something else failed first.
        if "does not implement textureGrad() at all" in run_result.stdout:
            print("%s: every gradient sample returned the "
                  "unsupported-texture-op constant" % name)
    if failed:
        raise SystemExit("textureGrad is wrong on: " + ", ".join(failed))
    print("PASS textureGrad on both fragment paths")


main()
