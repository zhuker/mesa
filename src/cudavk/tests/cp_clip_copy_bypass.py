#!/usr/bin/env python3
"""CPU-only regression of the exact clip helper source, compiled in both arms.

Does not replace CUDA render gates. No CUDA context, launches or GPU allocation.
Extracts source rather than maintaining a second implementation of the clipper.
"""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'kernels/cp_rasterize.cu').read_text()
clip = source[source.index('#define CP_CLIP_NUM_PLANES'):source.index('extern "C" __global__ void\ncp_clip_triangles')]
types = (root / 'kernels/cp_rast_types.h').read_text()
args = types[types.index('struct cp_clip_args {'):types.index('/* Run-wide clip/bin ABI limits.')]
prelude = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
using std::isfinite;
#define __device__
#define __forceinline__ inline
struct float4 { float x,y,z,w; };
static float4 make_float4(float x,float y,float z,float w) { return {x,y,z,w}; }
static unsigned atomicAdd(unsigned *p,unsigned v) { unsigned old=*p; *p+=v; return old; }
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
"""
main = r"""
struct Result {
  float4 out[16*3*CP_MAX_CLIP_SLOTS];
  uint64_t refs[16]; uint32_t active[16], ids[8], count; int emitted;
};
static void run(Result &r, float4 *in, unsigned slots, unsigned mode, bool candidate,
                unsigned capacity) {
  memset(&r,0x5a,sizeof(r)); r.count=1;
  cp_clip_args a={}; a.vs_out=(uintptr_t)in; a.out=(uintptr_t)r.out;
  a.out_count=(uintptr_t)&r.count; a.num_slots=slots; a.num_triangles=2;
  a.max_triangles=capacity; a.stable=mode&1;
  a.active_ids=(mode&2)?(uintptr_t)r.active:0;
  a.prim_refs=(mode&4)?(uintptr_t)r.refs:0;
  uint32_t *ids=(mode&8)?r.ids:nullptr;
  r.emitted=candidate?candidate::cp_clip_one(&a,1,ids):control::cp_clip_one(&a,1,ids);
  // Normalize published addresses before comparing complete side effects.
  for (unsigned i=0;i<16;i++) {
    auto p=r.refs[i];
    if (p>=(uintptr_t)in && p<(uintptr_t)(in+6*CP_MAX_CLIP_SLOTS))
      r.refs[i]=0x10000+p-(uintptr_t)in;
    if (p>=(uintptr_t)r.out && p<(uintptr_t)(r.out+16*3*CP_MAX_CLIP_SLOTS))
      r.refs[i]=0x20000+p-(uintptr_t)r.out;
  }
}
static void check(float4 a,float4 b,float4 c,int /*unused*/) {
  float4 in[6*CP_MAX_CLIP_SLOTS];
  for(unsigned slots: {1u,4u,16u}) {
    // Unique all-slot payloads, including exceptional values, detect stale
    // copies, vertex reordering, and accidental arithmetic on bypass planes.
    std::mt19937 bits(771);
    for(auto &v:in) {
      uint32_t raw[4]; for(auto &x:raw) x=bits();
      memcpy(&v,raw,sizeof(v));
    }
    in[3*slots]=a; in[4*slots]=b; in[5*slots]=c;
    for(unsigned mode=0;mode<16;mode++) for(unsigned capacity: {0u,1u,2u,8u,9u,10u,16u}) {
      Result x,y; run(x,in,slots,mode,false,capacity); run(y,in,slots,mode,true,capacity);
      CHECK(!memcmp(&x,&y,sizeof(x)));

    }
  }
}
int main() {
  check({0,0,.5f,1},{.5f,0,.5f,1},{0,.5f,.5f,1},0); // inside
  for(int p=0;p<7;p++) {
    float4 v={0,0,.5f,1};
    if(p==0) v.z=-1;
    if(p==1) {v.w=0;v.z=0;}
    if(p==2) v.z=2;
    if(p==3) v.x=-2;
    if(p==4) v.x=2;
    if(p==5) v.y=-2;
    if(p==6) v.y=2;
    check(v,v,v,1);
  }
  check({0,0,-.1f,1},{.5f,0,.5f,1},{0,.5f,.5f,1},0); // near crossing
  check({-2,0,-.1f,1},{-2,.5f,.5f,1},{-2,-.5f,.5f,1},0); // prior crossing
  // Plane 0 crosses, plane 1 bypasses, plane 2 crosses: using original
  // plane parity would read and write poly_a simultaneously here.
  check({-.2f,-.3f,-.5f,1},{.7f,.2f,2,1},{-.4f,.8f,.5f,1},-1);
  float tiny=std::numeric_limits<float>::denorm_min();
  check({0,0,-tiny,1},{0,0,-tiny,1},{0,0,-tiny,1},1);
  check({0,0,-0.f,1},{0,0,0.f,1},{0,0,0.f,1},0);
  for(float bad: {INFINITY,-INFINITY,NAN})
    check({bad,0,-1,1},{0,0,-1,1},{0,0,-1,1},0);
  float huge=std::numeric_limits<float>::max();
  check({huge,0,-1,huge},{0,0,-1,1},{0,0,-1,1},0); // distance overflow
  std::mt19937 gen(19); std::uniform_real_distribution<float> dist(-4,4);
  for(int i=0;i<2000;i++) {
    float4 v[3]; for(auto &p:v) p={dist(gen),dist(gen),dist(gen),dist(gen)};
    check(v[0],v[1],v[2],-1);
  }
  // Exercise each exceptional coordinate independently, not just x.
  for(float bad: {INFINITY,-INFINITY,NAN,-0.f,tiny,-tiny,huge,-huge})
    for(int component=0;component<4;component++) {
      float4 v={.1f,.2f,-.1f,1};
      float raw[4]; memcpy(raw,&v,sizeof(v)); raw[component]=bad;
      memcpy(&v,raw,sizeof(v));
      check(v,{-.7f,.8f,.6f,1},{.8f,-.7f,.7f,1},-1);
    }
  // Random raw positions cover NaN payloads and extreme exponent ranges.
  for(int i=0;i<1000;i++) {
    float4 v[3];
    for(auto &p:v) { uint32_t raw[4]; for(auto &x:raw) x=gen(); memcpy(&p,raw,sizeof(p)); }
    check(v[0],v[1],v[2],-1);
  }
  puts("PASS: copy bypass source equivalence; all slots/storage/ID modes, 7 capacities, 3000 random triangles");
}
"""
with tempfile.TemporaryDirectory(prefix='cp-clip-copy-') as tmp:
    path = Path(tmp)
    cpp = prelude + args
    for name, enabled in [('control', 0), ('candidate', 1)]:
        cpp += f'\n#define CP_CLIP_COPY_BYPASS {enabled}\nnamespace {name} {{\n'
        cpp += clip + '\n}\n#undef CP_CLIP_COPY_BYPASS\n'
    (path / 'gate.cpp').write_text(cpp + main)
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++14', '-O2',
                    '-ffp-contract=off', str(path / 'gate.cpp'), '-o', str(path / 'gate')], check=True)
    subprocess.run([str(path / 'gate')], check=True)
