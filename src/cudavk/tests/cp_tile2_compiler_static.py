#!/usr/bin/env python3
"""Bounded CPU source/family proof. Does not execute native shader modes."""
from pathlib import Path
import re
import subprocess
import sys
root = Path(__file__).resolve().parents[3]
src = root / 'src/cudavk'
c = (src / 'nir_to_ptx/cp_nir_to_llvm.c').read_text()
h = (src / 'nir_to_ptx/cp_nir_to_llvm.h').read_text()
k = (src / 'cp_kernels.c').read_text()
m = (src / 'meson.build').read_text()
p = (src / 'cpvk_pipeline.c').read_text()
def calls(text, name):
    result = []
    for match in re.finditer(r'\b' + name + r'\s*\(', text):
        start = match.end(); end = start; depth = 1
        while depth:
            depth += (text[end] == '(') - (text[end] == ')')
            end += 1
        result.append(text[start:end-1])
    return result
one = calls(c, 'cp_compile_nir_one')
assert len(one) == 12, len(one)
for call in one:
    assert call.rstrip().endswith('vs_layout'), call
for name in ('cp_compile_nir_software', 'link_and_inline_fs_interp'):
    for call in calls(c, name): assert call.rstrip().endswith('vs_layout'), call
assert 'bin->vs_layout);' in c  # vertex fetch inherits owning binary
assert 'helper_layout != vs_layout' in c
assert 'nir->info.stage == MESA_SHADER_COMPUTE && vs_layout != CP_VS_LAYOUT_AOS' in c
assert 'CP_VS_LAYOUT_AOS, CP_VS_LAYOUT_AOS' in p
assert 'cp_shader_exec_family_matches(bin, base)' in c
assert c.count('!cp_shader_exec_helper_matches(exec)') == 2  # reload and swap
assert 'exec->helper_layout = base->helper_layout' in c
assert 'stage_name, vs_layout' in c
assert h.count('cp_shader_exec_family_matches(bin, &bin->exec[mode])') == 2
assert p.index('cp_vs_layout_hash_stage(hash, vs_layout)') < p.index('simple_mtx_lock(&dev->shader_cache_lock)', p.index('cp_vs_layout_hash_stage(hash, vs_layout)'))
cache = (src / 'cp_vs_layout_cache.h').read_text()
assert '_mesa_blake3_update(&ctx, &layout_version, sizeof(layout_version))' in cache
assert 'k->vs_layout = cp_debug->vs_tile2 ? CP_VS_LAYOUT_TILE2_V1 : CP_VS_LAYOUT_AOS' in k
assert k.count('nvrtcCompileProgram(') == 1
assert 'opts[num_opts++] = "-DCP_VS_TILE2=1"' in k
assert 'blob_write_string(&key_blob, cp_vs_layout_src)' in k
assert 'blob_write_string(&key_blob, opts[i])' in k
assert 'cp_vs_layout.h.inc' in k and '"cp_vs_layout.h"' in k
assert "const char *opts[12]" in k
assert m.count("output : 'cp_vs_inline.bc'") == 1
assert m.count("output : 'cp_fs_inline.bc'") == 1
assert m.count("output : 'cp_fs_inline_tile2.bc'") == 1
assert m.count("'kernels/cp_vs_layout.h'") == 3
assert "'-DCP_VS_TILE2=1', '-O1'" in m
assert 'cp_fs_inline_tile2_bc : cp_fs_inline_bc' in c
assert 'LLVMGetModuleDataLayout(ctx->module)' in c
assert 'LLVMBuildZExt(ctx->builder, bid2, wide' in c
legalizer = (src / 'nir_to_ptx/cp_llvm_vs_tile2.h').read_text()
assert 'LLVMBuildMul(b, off32, LLVMConstInt(i32, 16, false)' in legalizer
assert 'LLVMBuildSExt(b, q32, i64, "tile2_logical_offset")' in legalizer
assert 'LLVMBuildAdd(b, first_slot, C64(((q % 16) + j) / 16)' in legalizer
assert 'LLVMBuildAdd(b, phase, vertex' in legalizer
assert 'C64((uint64_t)slots * 32)' in legalizer
assert 'tile2_dynamic_slot' not in legalizer

assert 'if (!emit_function(&ctx) || ctx.failed)' in c
# Baseline get_llvm_type constructs numeric scalar/fixed vectors only, with
# integer 1/8/16/32/64; float reinterpretations use half/float/double.
for width in (1,8,16,32,64):
    assert re.search(r'case '+str(width)+r':\s+base = LLVMInt'+str(width)+'TypeInContext',c)
# The actual old store emitter is retained verbatim after the ON-only arm.
baseline = subprocess.check_output(['git','show','69a58a486d4bf74ba8d77089894d17bed1ce598c:src/cudavk/nir_to_ptx/cp_nir_to_llvm.c'],cwd=root,text=True)
def off_store(text):
    start = text.index('      unsigned out_stride = ctx->nir->num_outputs * 16;')
    end = text.index('      break;',start)
    return text[start:end]
assert off_store(c) == off_store(baseline)
if len(sys.argv) > 1:
    build = Path(sys.argv[1]); dis = sys.argv[2]
    ir = [subprocess.check_output([dis,str(build/name),'-o','-'],text=True)
          for name in ('cp_fs_inline.bc','cp_fs_inline_tile2.bc')]
    def abi(text):
        return [line for line in text.splitlines() if line.startswith(('target datalayout','target triple','%struct.'))] + [re.sub(r' #[0-9]+.*','',line.split('{')[0]).strip() for line in text.splitlines() if line.startswith('define ') and '@cp_fs_' in line]
    assert abi(ir[0]) == abi(ir[1])
    assert ir[0] != ir[1]
    print('Both actual FS bitcode blobs have identical LLVM target/struct/function ABI')
print(f'TILE2 compiler static proof: {len(one)-1} producer/FS clones, cache/family/blob/OFF-store checks passed')
