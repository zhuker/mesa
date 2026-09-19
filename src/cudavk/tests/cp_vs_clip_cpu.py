#!/usr/bin/env python3
"""CPU-only: compile actual cp_clip_one source, compare canonical OFF/ON output.
No CUDA runtime, context or JIT. Stub atomicAdd is sequential fixture execution.
"""
from pathlib import Path
import subprocess
import tempfile
import sys
sanitize = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if "--sanitize" in sys.argv[1:] else []
ROOT = Path(__file__).resolve().parents[2]
K = ROOT / 'cudavk/kernels'
# parents[2] is src; keep source section tied to production implementation.
source = (K / 'cp_rasterize.cu').read_text()
section = source[source.index('#define CP_CLIP_NUM_PLANES'):source.index('extern "C" __global__ void\ncp_clip_triangles')]
prefix = r"""
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cassert>
#define _Static_assert static_assert
#include "cp_rast_types.h"
#define __device__
#define __forceinline__ inline
struct float4 { float x,y,z,w; };
static float4 make_float4(float x,float y,float z,float w){return {x,y,z,w};}
static unsigned atomicAdd(unsigned *p,unsigned x){unsigned old=*p;*p+=x;return old;}
"""
fixture = r"""
alignas(32) static float4 input[4*16], output[2*8*3*16];
static float4 *at(float4 *p,unsigned s,unsigned v,unsigned slot){
 return (float4 *)(uintptr_t)cp_vs_slot_address((uintptr_t)p,s*16,v,slot);
}
int main(){
 for(unsigned slots: {1u,3u,8u,16u}) for(unsigned phase=0;phase<2;phase++)
 for(unsigned stable=0;stable<2;stable++) for(unsigned refs=0;refs<2;refs++)
 for(unsigned active=0;active<2;active++) for(unsigned shape=0;shape<10;shape++) {
  memset(input,0xa5,sizeof(input));memset(output,0xa5,sizeof(output));
  float4 *base=at(input,slots,phase,0);
  for(unsigned v=0;v<3;v++) for(unsigned s=0;s<slots;s++)
    *at(base,slots,v,s)=make_float4(float(v+s),float(2*v+s),float(3*v+s),1);
  *at(base,slots,0,0)=make_float4(-.4f,-.4f,.4f,1);
  *at(base,slots,1,0)=make_float4(.4f,-.4f,.4f,1);
  *at(base,slots,2,0)=make_float4(0,.4f,.4f,1);
  float4 *p=at(base,slots,0,0);
  if(shape==1)p->z=-.5f;if(shape==2)p->w=-.5f;if(shape==3)p->z=2;
  if(shape==4)p->x=-2;if(shape==5)p->x=2;if(shape==6)p->y=-2;if(shape==7)p->y=2;
  if(shape==8)for(unsigned v=0;v<3;v++)at(base,slots,v,0)->z=-1;
  if(shape==9){p->x=-4;at(base,slots,1,0)->y=-4;at(base,slots,2,0)->y=4;}
  uint64_t references[8]={};uint32_t ids[8]={},count=0;
  cp_clip_args a={};a.vs_out=(uintptr_t)base;a.out=(uintptr_t)output;
  a.num_slots=slots;a.num_triangles=1;a.max_triangles=8;a.stable=stable;
  a.out_count=(uintptr_t)&count;a.prim_refs=refs?(uintptr_t)references:0;
  a.active_ids=active?(uintptr_t)ids:0;
  unsigned char saved[sizeof(input)];memcpy(saved,input,sizeof(input));
  int emitted=cp_clip_one(&a,0,nullptr);
  assert(memcmp(saved,input,sizeof(input))==0);
  fwrite(&emitted,sizeof(emitted),1,stdout);fwrite(&count,sizeof(count),1,stdout);
  fwrite(ids,sizeof(ids),1,stdout);
  for(unsigned id=0;id<8;id++) {
   bool live=stable?(active?false:id<(unsigned)emitted):id<count;
   if(stable&&active)for(unsigned j=0;j<count;j++)if(ids[j]==id)live=true;
   if(refs){bool present=references[id]!=0;fwrite(&present,sizeof(present),1,stdout);live=present;}
   if(live){float4 *t=refs?(float4 *)(uintptr_t)references[id]:at(output,slots,3*id,0);
    for(unsigned v=0;v<3;v++)for(unsigned s=0;s<slots;s++)fwrite(at(t,slots,v,s),16,1,stdout);
   } else if(stable&&!active&&!refs) {
    for(unsigned v=0;v<3;v++) {
     float4 z=*at(output,slots,3*id+v,0);assert(z.x==0&&z.y==0&&z.z==0&&z.w==1);
     for(unsigned s=1;s<slots;s++) {
      const unsigned char *p=(const unsigned char *)at(output,slots,3*id+v,s);
      for(unsigned b=0;b<16;b++)assert(p[b]==0xa5);
     }
    }
   }
  }
  // Input never changes, including unwritten slots and neighboring phase.
  for(unsigned v=0;v<3;v++)for(unsigned s=0;s<slots;s++)fwrite(at(base,slots,v,s),16,1,stdout);
 }
}
"""
with tempfile.TemporaryDirectory(prefix='cp-tile2-clip-') as td:
    td=Path(td); src=td/'fixture.cpp'
    src.write_text('#include <initializer_list>\n'+prefix+section+fixture)
    results=[]
    for arm in (0,1):
        exe=td/f'clip{arm}'
        subprocess.run(['c++','-std=c++11','-O2',*sanitize,f'-DCP_VS_TILE2={arm}','-I',str(K),str(src),'-o',str(exe)],check=True)
        results.append(subprocess.check_output([str(exe)]))
    assert results[0]==results[1], 'canonical CPU clipping mismatch'
    print(f'CPU actual clipping OFF/ON PASS ({len(results[0])} canonical bytes)')
