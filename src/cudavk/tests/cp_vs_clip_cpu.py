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
fixture = r"""namespace original_fixture {

alignas(32) static float4 input[4*16], output[2*8*3*16];
static float4 *at(float4 *p,unsigned s,unsigned v,unsigned slot){
 return (float4 *)(uintptr_t)cp_vs_slot_address((uintptr_t)p,s*16,v,slot);
}
void run(){
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

}

// Oracle addresses do not call the production address helper. Whole allocations
// include outer guards, preceding records, neighboring halves and spare slots.
alignas(32) static float4 input[12*16+4], output[80*16+4];
static float4 expected[80*16+4];
static float4 *at(float4 *p,unsigned s,unsigned v,unsigned slot){
#if CP_VS_TILE2
 uintptr_t raw=(uintptr_t)p; unsigned phase=(raw/16)%2;
 return (float4 *)(raw-phase*16)+((phase+v)/2)*2*s+slot*2+(phase+v)%2;
#else
 return p+v*s+slot;
#endif
}
static float distance_ref(float4 v,unsigned p){
 float d[]={v.z,v.w-1e-6f,v.w-v.z,v.x+v.w,v.w-v.x,v.y+v.w,v.w-v.y};
 return d[p];
}
using Vertex=std::vector<float4>;
static std::vector<Vertex> polygon_ref(std::vector<Vertex> poly){
 for(unsigned plane=0;plane<7;plane++){
  std::vector<Vertex> next;
  for(unsigned i=0;i<poly.size();i++){
   Vertex a=poly[i],b=poly[(i+1)%poly.size()];
   float da=distance_ref(a[0],plane),db=distance_ref(b[0],plane);
   if(da>=0)next.push_back(a);
   if((da>=0)!=(db>=0)){
    float t=da/(da-db);Vertex q=a;
    for(unsigned s=0;s<a.size();s++)q[s]={a[s].x+(b[s].x-a[s].x)*t,
     a[s].y+(b[s].y-a[s].y)*t,a[s].z+(b[s].z-a[s].z)*t,a[s].w+(b[s].w-a[s].w)*t};
    next.push_back(q);
   }
  }
  poly=next;if(poly.size()<3)return {};
 }
 return poly;
}
int main(){
 original_fixture::run();
 fprintf(stderr,"original cases=640 per arm PASS\n");
 unsigned cases=0,refused=0,partial=0,generated=0;
 for(unsigned slots: {1u,3u,8u,16u}) for(unsigned phase=0;phase<2;phase++)
 for(unsigned outphase=0;outphase<2;outphase++)
 for(unsigned stable=0;stable<2;stable++) for(unsigned refs=0;refs<2;refs++)
 for(unsigned active=0;active<2;active++) for(unsigned explicit_ids=0;explicit_ids<2;explicit_ids++)
 for(unsigned tri: {0u,1u,2u}) for(unsigned capacity: {0u,1u,8u})
 for(unsigned shape=0;shape<10;shape++) {
  memset(input,0xa5,sizeof(input));memset(output,0xa5,sizeof(output));
  float4 *base=at(input+2,slots,phase,0),*out=at(output+2,slots,outphase,0);
  std::vector<Vertex> original(3,Vertex(slots));
  for(unsigned v=0;v<3;v++) for(unsigned s=0;s<slots;s++)
    original[v][s]=make_float4(float(v+s+tri*10),float(2*v+s),float(3*v+s),1);
  original[0][0]=make_float4(-.4f,-.4f,.4f,1);
  original[1][0]=make_float4(.4f,-.4f,.4f,1);
  original[2][0]=make_float4(0,.4f,.4f,1);
  float4 *p=&original[0][0];
  if(shape==1)p->z=-.5f;if(shape==2)p->w=-.5f;if(shape==3)p->z=2;
  if(shape==4)p->x=-2;if(shape==5)p->x=2;if(shape==6)p->y=-2;if(shape==7)p->y=2;
  if(shape==8)for(unsigned v=0;v<3;v++)original[v][0].z=-1;
  if(shape==9){p->x=-4;original[1][0].y=-4;original[2][0].y=4;}
  for(unsigned v=0;v<3;v++)for(unsigned s=0;s<slots;s++)*at(base,slots,tri*3+v,s)=original[v][s];
  auto poly=shape==0?original:polygon_ref(original);
  unsigned wanted=poly.size()<3?0:poly.size()-2;
  unsigned first=stable?tri*8:3,limit=first+capacity;
  unsigned n=std::min(wanted,capacity),initial=stable?2:3;
  uint64_t references[28],expected_refs[28];uint32_t ids[28],expected_ids[28],fused[10],expected_fused[10];
  memset(references,0xa5,sizeof(references));memcpy(expected_refs,references,sizeof(references));
  memset(ids,0xa5,sizeof(ids));memcpy(expected_ids,ids,sizeof(ids));
  memset(fused,0xa5,sizeof(fused));memcpy(expected_fused,fused,sizeof(fused));
  uint32_t count=initial,expected_count=initial;
  memcpy(expected,output,sizeof(output));
  auto store=[&](unsigned id,unsigned v,unsigned slot,float4 value){
   ptrdiff_t index=at(out,slots,id*3+v,slot)-output;assert(index>=0&&index<(ptrdiff_t)(sizeof(output)/16));expected[index]=value;
  };
  for(unsigned k=0;k<n;k++){
   unsigned id=first+k;
   if(!stable)expected_count++;
   if(stable&&active){unsigned pos=expected_count++;if(pos<limit)expected_ids[pos]=id;}
   if(explicit_ids)expected_fused[k]=id;
   if(refs)expected_refs[id]=(uintptr_t)(shape==0?at(base,slots,tri*3,0):at(out,slots,id*3,0));
   if(!(refs&&shape==0))for(unsigned v=0;v<3;v++)for(unsigned s=0;s<slots;s++)
    store(id,v,s,poly[v==0?0:k+v][s]);
  }
  if(!stable&&wanted>capacity)expected_count++; // failed atomic reservation still increments
  if(stable&&!active&&!explicit_ids){
   unsigned begin=shape==0?1:n;
   for(unsigned k=begin;k<8&&first+k<limit;k++){
    if(refs)expected_refs[first+k]=0;
    else for(unsigned v=0;v<3;v++)store(first+k,v,0,{0,0,0,1});
   }
  }
  cp_clip_args a={};a.vs_out=(uintptr_t)base;a.out=(uintptr_t)out;
  a.num_slots=slots;a.num_triangles=3;a.max_triangles=limit;a.stable=stable;
  a.out_count=(uintptr_t)&count;a.prim_refs=refs?(uintptr_t)references:0;
  a.active_ids=active?(uintptr_t)ids:0;
  unsigned char saved[sizeof(input)];memcpy(saved,input,sizeof(input));
  int emitted=cp_clip_one(&a,tri,explicit_ids?fused:nullptr);
  assert(emitted==(int)n);assert(count==expected_count);
  assert(memcmp(saved,input,sizeof(input))==0);
  assert(memcmp(output,expected,sizeof(output))==0);
  assert(memcmp(references,expected_refs,sizeof(references))==0);
  assert(memcmp(ids,expected_ids,sizeof(ids))==0);
  assert(memcmp(fused,expected_fused,sizeof(fused))==0);
  // Pointer values differ between binaries; serialize canonical values only.
  fwrite(&emitted,sizeof(emitted),1,stdout);fwrite(&count,sizeof(count),1,stdout);
  fwrite(ids,sizeof(ids),1,stdout);fwrite(fused,sizeof(fused),1,stdout);
  for(unsigned k=0;k<n;k++){
   float4 *t=refs?(float4 *)(uintptr_t)references[first+k]:at(out,slots,(first+k)*3,0);
   for(unsigned v=0;v<3;v++)for(unsigned s=0;s<slots;s++)fwrite(at(t,slots,v,s),16,1,stdout);
  }
  cases++;refused+=wanted>0&&n==0;partial+=n>0&&n<wanted;generated+=shape!=0&&n>0;
 }
 fprintf(stderr,"cases=%u refused=%u partial=%u generated=%u per arm; full allocation and metadata oracle PASS\n",cases,refused,partial,generated);
}
"""

with tempfile.TemporaryDirectory(prefix='cp-tile2-clip-') as td:
    td=Path(td); src=td/'fixture.cpp'
    src.write_text('#include <initializer_list>\n#include <vector>\n#include <algorithm>\n'+prefix+section+fixture)
    results=[]
    for arm in (0,1):
        exe=td/f'clip{arm}'
        subprocess.run(['c++','-std=c++11','-O2',*sanitize,f'-DCP_VS_TILE2={arm}','-I',str(K),str(src),'-o',str(exe)],check=True)
        results.append(subprocess.check_output([str(exe)]))
    assert results[0]==results[1], 'canonical CPU clipping mismatch'
    print(f'CPU actual clipping OFF/ON PASS ({len(results[0])} canonical bytes)')
