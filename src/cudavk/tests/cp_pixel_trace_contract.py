#!/usr/bin/env python3
"""CPU source-contract checks; not a runtime rendering equivalence test."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
k=(r/'cp_kernels.c').read_text(); d=(r/'cp_debug.c').read_text(); s=(r/'kernels/cp_rasterize.cu').read_text()
assert 'if (cp_debug->pixel_trace)\n      opts[num_opts++] = "-DCP_PIXEL_TRACE=1";' in k
assert 'blob_write_string(&key_blob, opts[i])' in k
instrumented=k[k.index('cp_kernels_instrumented(void)'):k.index('cp_kernels_instrumented(void)')+1500]
instrumented=instrumented[:instrumented.index('\n}')+2]
assert 'pixel_trace' not in instrumented
flag=d[d.index('{ "CUDAVK_PIXEL_TRACE",'):]; flag=flag[:flag.index('},')+2]
assert 'CP_FLAG_BOOL_VALUE' in flag and '.dflt' not in flag
assert s.count('atomicMin((unsigned long long *)&visbuf[at],')==2
assert s.count('atomicMin((unsigned long long *)&visbuf[pixel],')==1
assert s.count('atomicMin((unsigned long long *)(uintptr_t)rast.framebuffer + pixel,')==1
assert s.count('clip_alloc(args, tri,')==2
assert 'cp_trace_atomic(trace, PACK_VISBUF' in s
print('pixel trace registry/cachekey/source invariants: PASS (not GPU equivalence)')
