#!/usr/bin/env python3
"""CPU-only, bit-exact source differential gate for CP_CLIP_LAZY.

The oracle is extracted verbatim from commit 69a58a486d4, not a second
implementation of the candidate. CUDA qualifiers and atomicAdd are the only
host shims. This checks sequential clipping semantics, not CUDA scheduling,
register allocation, or GPU floating-point instruction selection.

Run: python3 src/cudavk/tests/cp_clip_lazy_test.py
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

BASE = "69a58a486d4"
KERNEL = "src/cudavk/kernels/cp_rasterize.cu"
TYPES = "src/cudavk/kernels/cp_rast_types.h"
ROOT = Path(__file__).resolve().parents[3]


def region(text):
    begin = text.index("#define CP_CLIP_NUM_PLANES")
    end = text.index('extern "C" __global__ void\ncp_clip_triangles', begin)
    return text[begin:end]


def abi(text):
    begin = text.index("struct cp_clip_args {")
    end = text.index("};", begin) + 2
    constants = "\n".join(re.findall(
        r"^#define (?:CP_CLIP_MAX_OUT|CP_MAX_CLIP_SLOTS) .*", text, re.M))
    return constants + "\n" + text[begin:end]


PRELUDE = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <array>
#include <vector>
#include <limits>
#define __device__
#define __forceinline__ inline
struct float4 { float x, y, z, w; };
static float4 make_float4(float x,float y,float z,float w) { return {x,y,z,w}; }
static unsigned atomicAdd(unsigned *p,unsigned n) { unsigned old=*p; *p+=n; return old; }
using std::isfinite;
"""

HARNESS = r"""
static uint32_t rng = 0x912f361a;
static uint32_t next_u() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }
static float frombits(uint32_t x) { float f; memcpy(&f,&x,4); return f; }
using Triangle = std::array<float4,3>;
static std::vector<Triangle> cases() {
   std::vector<Triangle> v;
   Triangle inside = {{{-.5f,-.5f,.2f,1}, {.5f,-.5f,.8f,1}, {0,.5f,.5f,1}}};
   v.push_back(inside);
   // Every plane, every outside-vertex mask: acceptance, rejection and crossings.
   for (int p=0;p<7;p++) for (int mask=1;mask<8;mask++) {
      auto t=inside;
      for (int i=0;i<3;i++) if (mask&(1<<i)) {
         switch(p) {
         case 0:t[i].z=-.5f;break; case 1:t[i].w=-.5f;break;
         case 2:t[i].z=1.5f;break; case 3:t[i].x=-2;break;
         case 4:t[i].x=2;break; case 5:t[i].y=-2;break; case 6:t[i].y=2;break;
         }
      }
      v.push_back(t);
   }
   const uint32_t special[] = {0,0x80000000,1,0x80000001,0x007fffff,
      0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
      0x7fc12345,0xffc56789,0x7f812345};
   for (uint32_t bits:special) for(int c=0;c<4;c++) {
      auto t=inside;
      float a[4]; memcpy(a,&t[0],sizeof(a)); a[c]=frombits(bits);
      memcpy(&t[0],a,sizeof(a)); v.push_back(t);
   }
   for(float w: {1e-6f, std::nextafter(1e-6f,0.0f),std::nextafter(1e-6f,1.0f)}) {
      Triangle t={{{-w,-w,0,w},{w,-w,w,w},{0,w,w*.5f,w}}}; v.push_back(t);
   }
   v.push_back({{{-100,-100,-100,1},{100,-100,100,1},{0,100,.5f,1}}});
   v.push_back({{{0,0,0,0},{0,0,0,0},{0,0,0,0}}});
   for(int n=0;n<128;n++) {
      Triangle t;
      for(auto &p:t) {
         float a[4];
         for(float &x:a) x=(int(next_u()%8193)-4096)/1024.f;
         if(n%4) a[3]=std::fabs(a[3])+.00001f;
         memcpy(&p,a,sizeof(a));
      }
      v.push_back(t);
   }
   // Raw IEEE position bits: unlike the bounded geometry generator above,
   // this covers widely separated exponents and exceptional combinations.
   for(int n=0;n<256;n++) {
      Triangle t;
      for(auto &p:t) {
         uint32_t bits[4]; for(auto &x:bits) x=next_u();
         memcpy(&p,bits,sizeof(bits));
      }
      v.push_back(t);
   }
   auto repeated=inside;
   repeated[1]=repeated[0]; v.push_back(repeated);
   repeated[2]=repeated[0]; v.push_back(repeated);
   repeated[0]={-2,0,.5f,1}; repeated[1]=repeated[0];
   repeated[2]=inside[2]; v.push_back(repeated);
   const float hi=std::numeric_limits<float>::max();
   const float lo=std::numeric_limits<float>::min();
   // Finite inputs whose distance differences or lerps overflow/cancel.
   v.push_back({{{-hi,-hi,-hi,hi},{hi,hi,hi,hi},{0,0,lo,lo}}});
   v.push_back({{{hi,0,hi,hi},{-hi,0,-hi,hi},{0,hi,0,1}}});
   v.push_back({{{hi,hi,hi,1},{-hi,-hi,-hi,1},{lo,-lo,lo,1}}});
   v.push_back({{{-1,-1,0,1},{1,1,1,1},{std::nextafter(1.f,2.f),0,.5f,1}}});
   return v;
}
struct Buffers {
   std::vector<float4> in, out;
   std::vector<uint64_t> refs;
   std::vector<uint32_t> active;
   std::array<uint32_t,CP_CLIP_MAX_OUT+2> ids;
   uint32_t count;
   Buffers(int slots):in(9*slots),out(26*3*slots),refs(26),active(26) {}
};
static uint64_t logical(uint64_t p,const Buffers &b) {
   if(!p || p==0xcdcdcdcdcdcdcdcdULL) return p;
   auto in=uint64_t(uintptr_t(b.in.data())),out=uint64_t(uintptr_t(b.out.data()));
   if(p>=in && p<in+b.in.size()*sizeof(float4)) return 0x100000000ULL+(p-in);
   if(p>=out && p<out+b.out.size()*sizeof(float4)) return 0x200000000ULL+(p-out);
   fprintf(stderr,"invalid published reference %llx\n",(unsigned long long)p); exit(1);
}
static int ci, ss, mode, capacity, start_count, tri_now;
[[noreturn]] static void fail(const char *what,size_t at=0) {
   fprintf(stderr,"FAIL %s offset=%zu case=%d slots=%d mode=%d cap=%d start=%d tri=%d\n",
           what,at,ci,ss,mode,capacity,start_count,tri_now); exit(1);
}
static void exact(const char *label,const void *a,const void *b,size_t n) {
   if(memcmp(a,b,n)) {
      auto x=(const unsigned char*)a,y=(const unsigned char*)b;
      size_t i=0;while(x[i]==y[i])i++;
      fprintf(stderr,"byte baseline=%02x candidate=%02x\n",x[i],y[i]);fail(label,i);
   }
}
template<class Args> static Args args(Buffers &b) {
   Args a={}; a.vs_out=(uintptr_t)b.in.data();a.out=(uintptr_t)b.out.data();
   a.out_count=(uintptr_t)&b.count;a.num_triangles=3;a.num_slots=ss;
   a.max_triangles=capacity;a.stable=mode&1;
   a.active_ids=(mode&2)?(uintptr_t)b.active.data():0;
   a.prim_refs=(mode&4)?(uintptr_t)b.refs.data():0;
   return a;
}
int main() {
   auto all=cases(); uint64_t calls=0;
   for(ss=1;ss<=18;ss++) {
      int slots=std::min(ss,16); Buffers a(slots),b(slots);
      for(ci=0;ci<(int)all.size();ci++) {
         for(size_t i=0;i<a.in.size();i++) {
            float f[4]; for(float &x:f) {
               uint32_t bits=next_u();
               // Finite random payloads, plus signed zero/NaN/Inf varying probes.
               if(ci%13) bits=(bits&0x807fffff)|((next_u()%254+1)<<23);
               x=frombits(bits);
            }
            memcpy(&a.in[i],f,sizeof(f));
         }
         // Guaranteed payload exceptions in the directed crossing/reject
         // geometries. Every active varying slot sees every bit pattern as
         // ci advances; other geometries retain pseudorandom payload bits.
         const uint32_t payload[] = {0,0x80000000,1,0x80000001,0x007fffff,
            0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
            0x7fc12345,0xffc56789,0x7f812345,0xff812345};
         if(ci>=1 && ci<=49) for(int vertex=0;vertex<9;vertex++)
            for(int slot=1;slot<slots;slot++) {
               uint32_t bits[4];
               for(int c=0;c<4;c++) bits[c]=payload[(ci+slot+vertex+c)%14];
               memcpy(&a.in[vertex*slots+slot],bits,sizeof(bits));
            }
         for(int t=0;t<3;t++) for(int j=0;j<3;j++) a.in[(t*3+j)*slots]=all[ci][(j+t)%3];
         const auto original=a.in;
         for(mode=0;mode<16;mode++) for(int cap: {0,1,7,8,9,16,23,24}) {
            capacity=cap;
            for(int initial: {0,std::max(0,cap-1),cap}) {
               start_count=initial;b.in=a.in;
               for(Buffers *p: {&a,&b}) {
                  memset(p->out.data(),0xcd,p->out.size()*sizeof(float4));
                  memset(p->refs.data(),0xcd,p->refs.size()*8);
                  memset(p->active.data(),0xcd,p->active.size()*4);
                  p->count=initial;
               }
               auto aa=args<baseline::cp_clip_args>(a);auto bb=args<candidate::cp_clip_args>(b);
               for(int step=0;step<3;step++) {
                  // Alternate execution order to exercise compact append semantics.
                  tri_now=(ci&1)?2-step:step;
                  a.ids.fill(0xcdcdcdcd);b.ids.fill(0xcdcdcdcd);
                  int na=baseline::cp_clip_one(&aa,tri_now,(mode&8)?a.ids.data():nullptr);
                  int nb=candidate::cp_clip_one(&bb,tri_now,(mode&8)?b.ids.data():nullptr);
                  if(na!=nb)fail("emitted"); if(a.count!=b.count)fail("count");
                  exact("ids",a.ids.data(),b.ids.data(),sizeof(a.ids));
                  exact("active IDs/untouched",a.active.data(),b.active.data(),a.active.size()*4);
                  exact("output bits/untouched",a.out.data(),b.out.data(),a.out.size()*sizeof(float4));
                  exact("baseline input",original.data(),a.in.data(),a.in.size()*sizeof(float4));
                  exact("candidate input",original.data(),b.in.data(),b.in.size()*sizeof(float4));
                  for(size_t i=0;i<a.refs.size();i++) if(logical(a.refs[i],a)!=logical(b.refs[i],b))fail("reference identity",i);
                  calls++;
               }
            }
         }
      }
   }
   printf("PASS %llu paired calls; %zu geometries; slots 1..18; all 16 optional-output modes\n",
          (unsigned long long)calls,all.size());
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--arm", choices=("all", "off", "fast", "sanitize"), default="all")
    opts = parser.parse_args()
    def old(path):
        return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)
    candidate = (ROOT / KERNEL).read_text()
    if "CP_CLIP_LAZY" not in candidate:
        parser.error("candidate has no CP_CLIP_LAZY switch; refusing a vacuous gate")
    source = PRELUDE
    source += "\nnamespace baseline {\n" + abi(old(TYPES)) + "\n" + region(old(KERNEL)) + "\n}\n"
    source += "\nnamespace candidate {\n" + abi((ROOT / TYPES).read_text()) + "\n" + region(candidate) + "\n}\n"
    source += HARNESS
    arms = {"off": ["-O2", "-ffp-contract=off"],
            "fast": ["-O2", "-march=native", "-ffp-contract=fast"],
            "sanitize": ["-O1", "-ffp-contract=off", "-fsanitize=address,undefined",
                         "-fno-omit-frame-pointer", "-fno-sanitize-recover=all"]}
    with tempfile.TemporaryDirectory(prefix="cp-clip-lazy-") as temp:
        path = Path(temp) / "gate.cpp"
        path.write_text(source)
        for name, flags in arms.items():
            if opts.arm not in ("all", name):
                continue
            binary = Path(temp) / name
            command = [opts.cxx, "-std=c++17", "-DCP_CLIP_LAZY=1", *flags, str(path), "-o", str(binary)]
            print("+", " ".join(command), flush=True)
            try:
                subprocess.run(command, check=True, cwd=ROOT)
                subprocess.run([str(binary)], check=True, cwd=ROOT)
            except subprocess.CalledProcessError:
                repro = Path(tempfile.mkdtemp(prefix="cp-clip-lazy-failure-"))
                shutil.copytree(temp, repro / "artifacts")
                print(f"Failure artifacts preserved at {repro / 'artifacts'}", flush=True)
                raise
    print("CPU source gate passed. GPU correctness/performance are separate gates.")


if __name__ == "__main__":
    main()
