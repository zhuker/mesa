#!/usr/bin/env python3
"""CPU-only actual interpolation helper comparison, including >16 unclipped slots."""
from pathlib import Path
import subprocess
import tempfile
import sys
sanitize = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if "--sanitize" in sys.argv[1:] else []
K=Path(__file__).resolve().parents[1]/'kernels'
types=(K/'cp_rast_types.h').read_text()
start=types.index('static __device__ __forceinline__ const unsigned char *\ncp_primitive_base')
primitive=types[start:types.index('\n#endif\n\n/* One pass-episode',start)]
raster=(K/'cp_rasterize.cu').read_text()
def actual_function(name):
 start=raster.rfind('static ',0,raster.index('\n'+name+'('))
 brace=raster.index('{',start);end=brace+1;depth=1
 while depth:
  depth+=(raster[end]=='{')-(raster[end]=='}');end+=1
 return raster[start:end]
setup_source='\n'.join(actual_function(name) for name in ('edge_function','edge_is_top_left','setup_triangle'))

prefix=r"""
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <initializer_list>
#include <cmath>
#include <algorithm>
using std::min; using std::max;
#define __fsub_rn(a,b) ((a)-(b))
#define __fmul_rn(a,b) ((a)*(b))
struct int4 {int x,y,z,w;};
#define _Static_assert static_assert
#include "cp_rast_types.h"
#define __device__
#define __forceinline__ inline
struct float4 {float x,y,z,w;};
static float4 make_float4(float x,float y,float z,float w){return {x,y,z,w};}
static_assert(sizeof(cp_fs_interp_args)==416,"interp ABI");
static_assert(offsetof(cp_fs_interp_args,positions)==8,"positions ABI");
static_assert(offsetof(cp_fs_interp_args,seg_ranges)==368,"ranges ABI");
#define CP_INTERP_INLINE static inline
#define CP_FMUL_RN(a,b) ((a)*(b))
#define CP_FSUB_RN(a,b) ((a)-(b))
#define CP_MAKE_FLOAT4 make_float4
"""
fixture=r"""
#include "cp_fs_interp.h"
alignas(32) static float4 input[8*32];
static float4 *at(float4 *p,unsigned s,unsigned v,unsigned slot){
 return (float4 *)(uintptr_t)cp_vs_slot_address((uintptr_t)p,s*16,v,slot);
}
int main(){
 for(unsigned slots:{3u,8u,17u,32u})for(unsigned phase=0;phase<2;phase++)
 for(unsigned point=0;point<2;point++)for(unsigned ref=0;ref<2;ref++)
 for(unsigned winding=0;winding<2;winding++)for(unsigned crossing=0;crossing<2;crossing++){
  memset(input,0xa5,sizeof(input));float4 *base=at(input,slots,phase,0);
  for(unsigned v=0;v<6;v++)for(unsigned s=0;s<slots;s++)
   *at(base,slots,v,s)=make_float4(float(v+s),float(v*2+s),float(v*3+s),1);
  for(unsigned t=0;t<2;t++){
   *at(base,slots,3*t,0)=make_float4(-.4f,-.4f,crossing?-.4f:.4f,1);
   *at(base,slots,3*t+1,0)=make_float4(winding?-.4f:.4f,.4f,.4f,1);
   *at(base,slots,3*t+2,0)=make_float4(winding?.4f:-.4f,.4f,.4f,1);
   at(base,slots,3*t,1)->x=2;
  }
  uint64_t refs[2]={(uintptr_t)base,(uintptr_t)at(base,slots,3,0)};
  for(unsigned t=0;t<2;t++) {
   cp_fs_interp_args a={};a.positions=(uintptr_t)base;a.vs_out_stride=slots*16;
   a.prim_refs=ref?(uintptr_t)refs:0;a.vp_scale_x=4;a.vp_scale_y=-4;
   a.vp_trans_x=4;a.vp_trans_y=4;a.depth_scale=1;a.point_mode=point;
   a.width=8;a.psiz_slot=1;a.num_fs_inputs=3;a.pntc_input=-1;a.pos_input=-1;
   a.input_vs_slot[0]=0;a.input_vs_slot[1]=1;a.input_vs_slot[2]=slots-1;
   uint32_t pixel=0;float4 out[3],fc;unsigned char front=0;
   a.pixel_list=(uintptr_t)&pixel;a.frag_coord=(uintptr_t)&fc;a.front_face=(uintptr_t)&front;
   cp_rasterize_args ra={};ra.positions=a.positions;ra.prim_refs=a.prim_refs;
   ra.num_varyings=slots-1;ra.point_mode=point;ra.psiz_slot=1;
   ra.vp_scale_x=4;ra.vp_scale_y=-4;ra.vp_trans_x=4;ra.vp_trans_y=4;
   ra.clip_x1=8;ra.clip_y1=8;
   cp_tri_setup setup={};bool accepted=setup_triangle(&ra,t,&setup);
   assert(accepted);fwrite(&setup,sizeof(setup),1,stdout);
   if(ref){uint64_t saved=refs[t];refs[t]=0;assert(!setup_triangle(&ra,t,&setup));refs[t]=saved;}
   cp_interp_tri tri;
   if(ref){uint64_t saved=refs[t];refs[t]=0;assert(!cp_interp_setup(&a,t,&tri));refs[t]=saved;}
   assert(cp_interp_setup(&a,t,&tri));
   assert(cp_interp_pixel_prepared(&a,t,27,0,&tri,out,7,3,point?1:-1,0));
   fwrite(out,sizeof(out),1,stdout);fwrite(&fc,sizeof(fc),1,stdout);
   fwrite(&pixel,sizeof(pixel),1,stdout);fwrite(&front,1,1,stdout);
  }
 }
}
"""
with tempfile.TemporaryDirectory(prefix='cp-tile2-interp-') as td:
 td=Path(td);src=td/'fixture.cpp';src.write_text(prefix+primitive+setup_source+fixture);results=[]
 for arm in (0,1):
  exe=td/f'interp{arm}'
  subprocess.run(['c++','-std=c++11','-O2','-ffp-contract=off',*sanitize,f'-DCP_VS_TILE2={arm}','-I',str(K),str(src),'-o',str(exe)],check=True)
  results.append(subprocess.check_output([str(exe)]))
 assert results[0]==results[1], 'CPU canonical interpolation mismatch'
 print(f'CPU actual raster setup + interpolation OFF/ON PASS ({len(results[0])} canonical bytes; includes L17/32 unclipped crossing)')
