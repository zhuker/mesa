#!/usr/bin/env python3
"""CPU fault tests of actual renderer bodies (no CUDA library/context).

Function text is extracted uniquely and compiled against a narrow fake allocator.
Changing an accessed field or function dependency fails compilation, rather than
silently testing a detached model of the renderer's failure transitions.
"""
from pathlib import Path
import os
import argparse
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--sanitize", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = (root / "cp_renderer.c").read_text()

def function(name):
    pattern = r"(?:static )?(?:bool|void|CUdeviceptr)\n" + name + r"\([^;]*?\n\{"
    matches = list(re.finditer(pattern, source))
    assert len(matches) == 1, (name, len(matches))
    start = matches[0].start()
    # Braces in these bounded function bodies contain no string literals.
    pos = matches[0].end()
    depth = 1
    while depth:
        assert pos < len(source), name
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[start:pos]

# Every existing clip descriptor originates behind these same slot gates.
# This is a source/admission proof, not a Vulkan clipping correctness oracle.
assert source.count("num_vs_outputs <= CP_MAX_CLIP_SLOTS") == 2
assert source.count("outputs > CP_MAX_CLIP_SLOTS") == 1
hoist = re.search(r"const bool clip_block_runs = ([^;]+);", source).group(1)
assert "num_vs_outputs <= CP_MAX_CLIP_SLOTS" in hoist
assert "clip_bytes_valid" in hoist
assert re.search(r"if \(screen->kernels.clip_triangles &&\s*"
                 r"num_vs_outputs <= CP_MAX_CLIP_SLOTS && clip_bytes_valid\)", source)
assert re.search(r"if \(outputs > CP_MAX_CLIP_SLOTS\) \{\s*"
                 r"cp->run_geometry.clip_refused_slots\+\+;\s*return false;", source)
# Pointer-identity launch grouping implies a common true logical stride.
assert "sa->vs != sb->vs || !sa->vs" in function("cp_run_geometry_same_launch")
# The debug span join is before readback, and errors stop readback.
debug_begin = source.index("if (cp_debug->debug_fs) {", source.index("cp_shade_run("))
debug_end = source.index("for (unsigned i = 0; i < num_fs_inputs; i++)", debug_begin)
debug_source = source[debug_begin:debug_end]
assert debug_source.index("cuCtxSynchronize() != CUDA_SUCCESS") < debug_source.index("cuMemcpyDtoH(vs_out")
assert re.search(r"cuCtxSynchronize\(\) != CUDA_SUCCESS\) \{\s*"
                 r"cp_renderer_texture_fatal\(cp\);\s*return;", debug_source)
assert "cp_vs_tile2_address(vs_output_buf," in debug_source
assert "not semantic geometry" in debug_source
# Lines remain the inherited default-to-triangles limitation.
pipeline = (root / "cpvk_pipeline.c").read_text()
prim = re.search(r"cpvk_prim\([^)]*\)\s*\{(.*?)\n\}", pipeline, re.S).group(1)
assert "LINE_LIST" not in prim and "LINE_STRIP" not in prim
assert re.search(r"default:\s*return MESA_PRIM_TRIANGLES;", prim)

pending = re.findall(r"struct cp_pending_clip \{[^}]*\};", source)
assert len(pending) == 1
prefix = r'''#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "kernels/cp_rast_types.h"
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define ALIGN_POT(x,a) (((x)+((a)-1)) & ~((size_t)(a)-1))
typedef uint64_t CUdeviceptr;
typedef int CUresult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_UNKNOWN 999
struct cp_context {
   bool device_fatal;
   struct { size_t used, size, peak; CUdeviceptr base, overflow[8];
            unsigned num_overflow; } dscratch;
   struct { unsigned slot_next; size_t slot_size[2]; CUdeviceptr slot[2]; } vslane;
};
static struct { bool vs_tile2; unsigned run_geometry_fail_alloc; } debug;
#define cp_debug (&debug)
static unsigned alloc_calls, free_calls, fatal_calls;
static int allocation_error, free_error;
static size_t last_bytes;
static CUresult cp_mem_alloc_retry(struct cp_context *cp, CUdeviceptr *out, size_t bytes)
{ (void)cp; alloc_calls++; last_bytes=bytes; if (allocation_error) return allocation_error;
  *out=0x100000; return CUDA_SUCCESS; }
static CUresult cuMemFree(CUdeviceptr p)
{ (void)p; free_calls++; return free_error; }
static void cp_renderer_texture_fatal(struct cp_context *cp)
{ fatal_calls++; cp->device_fatal=true; }
'''
header = (root / "cp_renderer.h").read_text()
limit = re.findall(r"^#define CP_SCRATCH_MAX_BYTES .*", header, re.M)
assert len(limit) == 1
bodies = "\n".join(function(n) for n in (
    "cp_scratch_alloc_device", "cp_vslane_output",
    "cp_run_geometry_device_reserve", "cp_rast_args_unclip", "cp_restore_preclip"))
main = r'''
int main(void) {
   debug.vs_tile2=true;
   size_t bytes=99;
   assert(cp_vs_tile2_capacity(0,128,&bytes) && bytes==0);
   assert(cp_vs_tile2_capacity(3,128,&bytes) && bytes==512);
   assert(cp_vs_tile2_capacity(UINT32_MAX,128,&bytes));
   assert(bytes == ((uint64_t)UINT32_MAX+1)*128);
   bytes=99;
   assert(!cp_vs_tile2_capacity(SIZE_MAX,128,&bytes) && bytes==99);
   assert(!cp_vs_tile2_triangle_capacity(SIZE_MAX/3+1,128,&bytes));
   assert(cp_vs_tile2_triangle_capacity(3,128,&bytes) && bytes==1280);
   assert(!cp_vs_tile2_capacity(1,0,&bytes));
   assert(!cp_vs_tile2_capacity(1,SIZE_MAX/2+1,&bytes));
   size_t off=SIZE_MAX-15, start=77;
   assert(!cp_vs_checked_append(&off,1,32,&start));
   assert(off==SIZE_MAX-15 && start==77);
   off=1; assert(cp_vs_checked_append(&off,512,32,&start));
   assert(start==32 && off==544);
   assert(cp_vs_tile2_origin_valid(0x100));
   assert(!cp_vs_tile2_origin_valid(0x110));
   uint64_t origin=99;
   assert(cp_vs_tile2_debug_span(0,0,128,&origin,&bytes) && bytes==0);
   assert(cp_vs_tile2_debug_span(0x110,2,128,&origin,&bytes));
   assert(origin==0x100 && bytes==512);
   for (unsigned slots=1; slots<=32; slots++) {
      for (unsigned n=1; n<=5; n++) {
         for (unsigned phase=0; phase<2; phase++) {
            uint64_t first=0x100+phase*16;
            assert(cp_vs_tile2_debug_span(first,n,slots*16,&origin,&bytes));
            assert(cp_vs_tile2_address(first,slots*16,n-1,slots-1,15)
                   < origin+bytes);
         }
      }
   }
   struct cp_context cp={0};
   debug.vs_tile2=false;
   assert(cp_scratch_alloc_device(&cp,3*128)==0x100000);
   assert(cp.dscratch.used==384); /* OFF does not round. */
   memset(&cp,0,sizeof(cp)); alloc_calls=0; debug.vs_tile2=true;
   assert(!cp_scratch_alloc_device(&cp,0) && !alloc_calls);
   cp.dscratch.used=SIZE_MAX-3;
   assert(!cp_scratch_alloc_device(&cp,1) && !alloc_calls);
   cp.dscratch.used=1;
   assert(!cp_scratch_alloc_device(&cp,SIZE_MAX) && !alloc_calls);
   cp.dscratch.used=0; cp.dscratch.size=CP_SCRATCH_MAX_BYTES;
   cp.dscratch.base=0x200000;
   assert(cp_vs_tile2_capacity(CP_SCRATCH_MAX_BYTES/128-1,128,&bytes));
   assert(bytes==CP_SCRATCH_MAX_BYTES);
   assert(cp_scratch_alloc_device(&cp,bytes)==0x200000);
   assert(cp.dscratch.used==bytes && cp.dscratch.peak==bytes && !alloc_calls);
   assert(!cp_scratch_alloc_device(&cp,1));
   memset(&cp,0,sizeof(cp)); allocation_error=CUDA_ERROR_OUT_OF_MEMORY;
   assert(!cp_scratch_alloc_device(&cp,512));
   assert(cp.dscratch.used==0 && cp.dscratch.base==0 && cp.dscratch.peak==512);
   CUdeviceptr ptr=0x300000; size_t capacity=512;
   unsigned before=alloc_calls;
   assert(cp_run_geometry_device_reserve(&cp,&ptr,&capacity,0));
   assert(alloc_calls==before && ptr==0x300000 && capacity==512);
   assert(!cp_run_geometry_device_reserve(&cp,&ptr,&capacity,513));
   assert(ptr==0x300000 && capacity==512 && !cp.device_fatal);
   allocation_error=CUDA_ERROR_UNKNOWN;
   assert(!cp_run_geometry_device_reserve(&cp,&ptr,&capacity,513));
   assert(ptr==0x300000 && capacity==512 && cp.device_fatal && fatal_calls==1);
   cp.device_fatal=false; allocation_error=0; free_error=CUDA_ERROR_UNKNOWN;
   assert(!cp_run_geometry_device_reserve(&cp,&ptr,&capacity,513));
   assert(ptr==0x300000 && capacity==512 && cp.device_fatal && free_calls==2);
   free_error=0; cp.device_fatal=false;
   for (unsigned injected=1; injected<=2; injected++) {
      debug.run_geometry_fail_alloc=injected;
      before=alloc_calls;
      assert(!cp_run_geometry_device_reserve(&cp,&ptr,&capacity,513));
      assert(ptr==0x300000 && capacity==512 && alloc_calls==before);
      assert(cp.device_fatal==(injected==2));
      cp.device_fatal=false;
   }
   debug.run_geometry_fail_alloc=0;
   assert(cp_run_geometry_device_reserve(&cp,&ptr,&capacity,513));
   assert(ptr==0x100000 && capacity==4096 && last_bytes==4096);
   before=alloc_calls;
   assert(!cp_run_geometry_device_reserve(&cp,&ptr,&capacity,SIZE_MAX));
   assert(alloc_calls==before && capacity==4096 && ptr==0x100000);
   cp.vslane.slot[0]=0x400000; cp.vslane.slot_size[0]=512;
   assert(!cp_vslane_output(&cp,0));
   assert(cp_vslane_output(&cp,512)==0x400000);
   allocation_error=CUDA_ERROR_OUT_OF_MEMORY;
   assert(!cp_vslane_output(&cp,513));
   assert(cp.vslane.slot[0]==0x400000 && cp.vslane.slot_size[0]==512);
   allocation_error=0; cp.dscratch.num_overflow=ARRAY_SIZE(cp.dscratch.overflow);
   before=alloc_calls; assert(!cp_vslane_output(&cp,513)); assert(alloc_calls==before);
   cp.dscratch.num_overflow=0;
   assert(cp_vslane_output(&cp,513)==0x100000);
   assert(cp.dscratch.overflow[0]==0x400000 && cp.dscratch.num_overflow==1);
   for (unsigned phase=0; phase<2; phase++) {
      struct cp_pending_clip clip={.positions=0x800000+phase*16,
         .num_triangles=3,.rast_num_triangles=3,.rect_prim_shift=0,.fs_prim_shift=0};
      struct cp_rasterize_args ra={.positions=0x900000,.prim_refs=3,
         .tri_count=4,.active_ids=5,.num_triangles=24,.rect_prim_shift=3};
      CUdeviceptr vs=0x900000; unsigned triangles=24, shift=3;
      cp_restore_preclip(&clip,&ra,&vs,&triangles,&shift);
      assert(vs==clip.positions && ra.positions==clip.positions);
      assert(!ra.prim_refs && !ra.tri_count && !ra.active_ids);
      assert(ra.num_triangles==3 && triangles==3 && !shift && !ra.rect_prim_shift);
   }
   puts("TILE2 renderer actual-body CPU tests passed");
   return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="cp-tile2-renderer-") as tmp:
    cfile = Path(tmp)/"test.c"
    binary = Path(tmp)/"test"
    cfile.write_text(prefix+limit[0]+"\n"+pending[0]+"\n"+bodies+main)
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if args.sanitize else []
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
                    "-Werror", *flags, "-I", str(root), str(cfile), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
