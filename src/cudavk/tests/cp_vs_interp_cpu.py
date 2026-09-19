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
def actual_function(name, raster=raster):
 start=raster.rfind('static ',0,raster.index('\n'+name+'('))
 brace=raster.index('{',start);end=brace+1;depth=1
 while depth:
  depth+=(raster[end]=='{')-(raster[end]=='}');end+=1
 return raster[start:end]
setup_source='\n'.join(actual_function(name) for name in ('edge_function','edge_is_top_left','edge_inside','setup_triangle','cp_sample_pos'))

fs=(K/'cp_fs.cu').read_text()
quad_source=actual_function('cp_write_batch_rows',fs)+'\n'+ 'template <bool RANGES>\n'+actual_function('cp_abuf_interpolate_body',fs)
raster_source='template <bool ABUF>\n'+actual_function('rasterize_point')+'\n'+actual_function('cp_rast_direct_one')
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
# Emission is a CPU recorder, not production depth/atomic/writeback execution.
recorder=r"""
static unsigned emitted;
template <bool ABUF> static void emit_fragment(cp_rasterize_args *,uint32_t t,
 int x,int y,int sm,float z){
 ++emitted; fwrite(&t,4,1,stdout);fwrite(&x,4,1,stdout);fwrite(&y,4,1,stdout);
 fwrite(&sm,4,1,stdout);fwrite(&z,4,1,stdout);
}
#include "cp_fs_interp.h"
struct cpu_dim {unsigned x;};
static cpu_dim blockIdx={0},blockDim={1},threadIdx={0};
"""
fixture=r"""
alignas(32) static float4 input[8*32];
static float4 *at(float4 *p,unsigned s,unsigned v,unsigned slot){
 return (float4 *)(uintptr_t)cp_vs_slot_address((uintptr_t)p,s*16,v,slot);
}
int main(){
 // Independent Vulkan standard-location oracle for supported sample counts.
 const float sx[13]={.5f,.375f,.875f,.125f,.625f,.5625f,.4375f,.8125f,.3125f,.1875f,.0625f,.6875f,.9375f};
 const float sy[13]={.5f,.125f,.375f,.625f,.875f,.3125f,.6875f,.5625f,.1875f,.8125f,.4375f,.9375f,.0625f};
 unsigned sample_index=0;
 for(unsigned n:{1u,4u,8u})for(unsigned j=0;j<n;j++){
  float x,y;cp_sample_pos(n,j,&x,&y);
  assert(x==sx[sample_index] && y==sy[sample_index]);++sample_index;
 }

 for(unsigned slots:{3u,8u,17u,32u})for(unsigned phase=0;phase<2;phase++)
 for(unsigned point=0;point<2;point++)for(unsigned ref=0;ref<2;ref++)
 for(unsigned perspective=0;perspective<2;perspective++)
 for(unsigned winding=0;winding<2;winding++)for(unsigned crossing=0;crossing<2;crossing++){
  memset(input,0xa5,sizeof(input));float4 *base=at(input,slots,phase,0);
  for(unsigned v=0;v<6;v++)for(unsigned s=0;s<slots;s++)
   *at(base,slots,v,s)=make_float4(float(v+s),float(v*2+s),float(v*3+s),1);
  for(unsigned t=0;t<2;t++){
   *at(base,slots,3*t,0)=make_float4(-.4f,-.4f,crossing?-.4f:.4f,1);
   *at(base,slots,3*t+1,0)=make_float4(winding?-.4f:.4f,.4f,.4f,1);
   *at(base,slots,3*t+2,0)=make_float4(winding?.4f:-.4f,.4f,.4f,1);
   at(base,slots,3*t,1)->x=2;
   if(perspective)for(unsigned v=0;v<3;v++){
    float4 &p=*at(base,slots,3*t+v,0);float w= v==0?.5f:(v==1?1.25f:2.f);
    p.x*=w;p.y*=w;p.z*=w;p.w=w;
   }
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
   ra.clip_x1=8;ra.clip_y1=8;ra.width=8;ra.height=8;
   for(unsigned samples:{1u,4u,8u}){
    ra.num_samples=samples;emitted=0;cp_rast_direct_one(&ra,t);assert(emitted>0);
    fwrite(&emitted,sizeof(emitted),1,stdout);
   }
   cp_tri_setup setup={};bool accepted=setup_triangle(&ra,t,&setup);
   assert(accepted);fwrite(&setup,sizeof(setup),1,stdout);
   if(ref){uint64_t saved=refs[t];refs[t]=0;assert(!setup_triangle(&ra,t,&setup));refs[t]=saved;}
   cp_interp_tri tri;
   if(ref){uint64_t saved=refs[t];refs[t]=0;assert(!cp_interp_setup(&a,t,&tri));refs[t]=saved;}
   assert(cp_interp_setup(&a,t,&tri));
   assert(cp_interp_pixel_prepared(&a,t,27,0,&tri,out,7,3,point?1:-1,0));
   // Independent double-precision barycentric oracle, original vertex order.
   if(!point){
    double x[3],y[3],iw[3];
    for(unsigned v=0;v<3;v++){
     float4 pos=*at(base,slots,t*3+v,0);iw[v]=1.0/double(pos.w);
     x[v]=pos.x*iw[v]*4+4;y[v]=pos.y*iw[v]*-4+4;
    }
    double den=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);
    double b[3];b[0]=((y[1]-y[2])*(3.5-x[2])+(x[2]-x[1])*(3.5-y[2]))/den;
    b[1]=((y[2]-y[0])*(3.5-x[2])+(x[0]-x[2])*(3.5-y[2]))/den;
    b[2]=1-b[0]-b[1];double denom=0,num=0;
    for(unsigned v=0;v<3;v++){
     denom+=b[v]*iw[v];num+=b[v]*iw[v]*at(base,slots,t*3+v,slots-1)->x;
    }
    assert(fabs(out[2].x-num/denom)<1e-4);
    assert(fabs(fc.w-denom)<1e-5);
   }
   fwrite(out,sizeof(out),1,stdout);fwrite(&fc,sizeof(fc),1,stdout);
   fwrite(&pixel,sizeof(pixel),1,stdout);fwrite(&front,1,1,stdout);
   // Actual global FS AoS path with nonzero slot and padded stride.
   float4 global[10];memset(global,0x5a,sizeof(global));
   float4 before[10];memcpy(before,global,sizeof(global));
   uint32_t pixels[4]={};float4 coords[4]={};unsigned char fronts[4]={};
   a.fs_in=(uintptr_t)global;a.fs_in_stride=5*16;
   a.pixel_list=(uintptr_t)pixels;a.frag_coord=(uintptr_t)coords;
   a.front_face=(uintptr_t)fronts;a.pos_input=0;a.pntc_input=point?1:-1;
   assert(cp_interp_pixel(&a,t,27,1));
   assert(!memcmp(global+5,out,sizeof(out)));
   assert(!memcmp(global,before,5*16));assert(!memcmp(global+8,before+8,2*16));
   fwrite(global,sizeof(global),1,stdout);
   // Two separated episode ranges, selecting either phase and reference mode.
   cp_seg_range ranges[2]={};
   for(unsigned r=0;r<2;r++){
    ranges[r].positions=(uintptr_t)at(base,slots,r*3,0);
    ranges[r].prim_refs=ref?(uintptr_t)(refs+r):0;
    ranges[r].prim_base=10+r*10;ranges[r].prim_end=11+r*10;
    ranges[r].row_base=7+r;ranges[r].prim_shift=r;
   }
   a.seg_ranges=(uintptr_t)ranges;a.num_seg_ranges=2;
   for(unsigned bad:{0u,9u,11u,19u,21u,100u}){
    cp_fs_interp_args b=a;assert(!cp_resolve_seg_range(&b,bad));
    assert(!memcmp(&a,&b,sizeof(a)));
   }
   cp_fs_interp_args resolved=a;assert(cp_resolve_seg_range(&resolved,10+t*10));
   assert(resolved.positions==ranges[t].positions && resolved.vs_out==resolved.positions);
   assert(resolved.prim_refs==ranges[t].prim_refs && resolved.row_base==7+t);
   assert(resolved.prim_shift==t && resolved.abuf_prim_base==10+t*10);
   // Real quad body, including helper lanes and odd framebuffer clamping.
   for(unsigned odd=0;odd<2;odd++)for(unsigned mask:{0u,1u,5u,15u}){
    float4 qout[16];memset(qout,0x5a,sizeof(qout));
    uint32_t rows[4]={},block=odd?15:5,gprim=10+t*10;
    unsigned char cov[4]={},qm=mask;
    a.width=odd?7:8;a.height=odd?7:8;a.quad_width=4;
    a.abuf_num_quads=1;a.abuf_quad_block=(uintptr_t)&block;
    a.abuf_quad_prim=(uintptr_t)&gprim;a.abuf_quad_mask=(uintptr_t)&qm;
    a.out_batch_rows=(uintptr_t)rows;a.coverage=(uintptr_t)cov;
    a.fs_in=(uintptr_t)qout;a.fs_in_stride=4*16;
    cp_abuf_interpolate_body<true>(a);
    for(unsigned lane=0;lane<4;lane++){
     unsigned x=(block%4)*2+(lane&1),y=(block/4)*2+(lane>>1);
     unsigned pix=std::min(y,a.height-1)*a.width+std::min(x,a.width-1);
     float4 expected[3];cp_fs_interp_args b=a;
     assert(cp_resolve_seg_range(&b,gprim));cp_interp_tri qt;
     assert(cp_interp_setup(&b,0,&qt));
     assert(cp_interp_pixel_prepared(&b,0,pix,lane,&qt,expected,7,3,a.pntc_input,0));
     assert(!memcmp(expected,qout+lane*4,sizeof(expected)));
     assert(!memcmp(qout+lane*4+3,before,16));
     assert(rows[lane]==7+t && pixels[lane]==pix);
     assert(cov[lane]==unsigned(x<a.width && y<a.height && (mask&(1u<<lane))));
    }
    fwrite(qout,sizeof(qout),1,stdout);fwrite(cov,sizeof(cov),1,stdout);
    fwrite(rows,sizeof(rows),1,stdout);
   }
  }
 }
}
"""
with tempfile.TemporaryDirectory(prefix='cp-tile2-interp-') as td:
 td=Path(td);src=td/'fixture.cpp';src.write_text(prefix+primitive+setup_source+recorder+raster_source+quad_source+fixture);results=[]
 for arm in (0,1):
  exe=td/f'interp{arm}'
  subprocess.run(['c++','-std=c++11','-O2','-ffp-contract=off',*sanitize,f'-DCP_VS_TILE2={arm}','-I',str(K),str(src),'-o',str(exe)],check=True)
  results.append(subprocess.check_output([str(exe)]))
 assert results[0]==results[1], 'CPU canonical interpolation mismatch'
 print(f'CPU actual raster setup + interpolation OFF/ON PASS ({len(results[0])} canonical bytes; 512 primitive cases/arm; 1536 raster loops at 1/4/8 samples; 4096 quads; includes L17/32 unclipped crossing)')

 print('Scope: actual sample positions/direct raster loops with CPU emission recorder; actual merged quad body and global/local interpolation. No device shuffle, shader derivative, atomic depth, flat/noperspective or native CUDA proof.')
 print('Per arm: 13 sample-position oracle checks; 256 triangle perspective oracle checks; 512 padded global-storage checks; 3072 invalid-range checks; 512 valid-range metadata checks; 16384 quad lane checks. Original 256 w=1 cases retained; 256 nonunit-w cases added.')
