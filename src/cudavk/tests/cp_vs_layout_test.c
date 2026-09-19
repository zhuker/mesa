/* CPU-only proof of the actual shared layout helpers. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../kernels/cp_vs_layout.h"
int main(void) {
   const size_t ns[] = {0,1,2,3,5,255,256,257};
   for (size_t l=1;l<=32;l++) for(size_t ni=0;ni<8;ni++) {
      size_t n=ns[ni], s=l*16, bytes;
      assert(cp_vs_tile2_capacity(n,s,&bytes));
      unsigned char *seen=calloc(bytes+1,1);
      for(size_t v=0;v<n;v++) for(size_t slot=0;slot<l;slot++) for(size_t c=0;c<16;c++) {
         uint64_t a=cp_vs_tile2_address(256,s,v,slot,c)-256;
         assert(a<bytes && !seen[a]); seen[a]=1;
         for(size_t b=0;b<=v;b++) {
            uint64_t p=cp_vs_tile2_address(256,s,b,0,0);
            assert(cp_vs_tile2_address(p,s,v-b,slot,c)==a+256);
         }
      }
      free(seen);
      for(size_t b=0;b<n;b++) {
         uint64_t p=cp_vs_tile2_address(256,s,b,0,0), origin;size_t span;
         assert(cp_vs_tile2_debug_span(p,n-b,s,&origin,&span));
         assert(origin<=p && origin+span==256+bytes);
      }
   }
   size_t bytes, off=(size_t)-8,start=999;
   assert(!cp_vs_checked_append(&off,16,32,&start));assert(start==999);
   assert(!cp_vs_tile2_capacity((size_t)-1,16,&bytes));
   assert(!cp_vs_tile2_capacity(2,(size_t)-16,&bytes));
   assert(!cp_vs_tile2_triangle_capacity((size_t)-1/3+1,16,&bytes));
   assert(cp_vs_tile2_capacity(0,128,&bytes)&&bytes==0);
   assert(!cp_vs_tile2_capacity(1,0,&bytes));
   assert(cp_vs_tile2_origin_valid(256));
   assert(!cp_vs_tile2_origin_valid(272));
   assert(!cp_vs_tile2_origin_valid(0));
   /* residue16 is NOT a valid tile origin: demonstrate phase mismatch. */
   assert(cp_vs_tile2_address(272,128,1,0,0)!=272+16);
   /* Measured nine stores: exactly 116 bytes, all holes/padding guarded. */
   unsigned char buf[6*128+32];memset(buf,0xa5,sizeof(buf));
   unsigned char touched[sizeof(buf)];memset(touched,0,sizeof(touched));
   const size_t q[]={0,16,32,40,48,64,80,96,112};
   const size_t width[]={16,8,8,4,16,16,16,16,16};
   size_t writes=0;
   for(size_t v=0;v<5;v++) for(size_t j=0;j<9;j++) for(size_t c=0;c<width[j];c++) {
      size_t a=cp_vs_tile2_address(256,128,v,(q[j]+c)/16,(q[j]+c)%16)-256;
      assert(!touched[a]); touched[a]=1; buf[a]=0x5a; writes++;
   }
   assert(writes==5*116);
   for(size_t i=0;i<sizeof(buf);i++) assert(buf[i]==(touched[i]?0x5a:0xa5));
   /* Shared 3/6/0/3/9 groups own disjoint bytes without member padding. */
   unsigned char owners[22*128];memset(owners,0,sizeof(owners));
   const size_t lens[]={3,6,0,3,9};size_t begin=0;
   for(size_t m=0;m<5;m++) {
      uint64_t p=cp_vs_tile2_address(256,128,begin,0,0);
      for(size_t v=0;v<lens[m];v++) for(size_t q=0;q<128;q++) {
         size_t a=cp_vs_tile2_address(p,128,v,q/16,q%16)-256;
         assert(a==cp_vs_tile2_address(256,128,begin+v,q/16,q%16)-256);
         assert(!owners[a]);owners[a]=(unsigned char)m+1;
      }
      begin+=lens[m];
   }
   puts("tile2 actual helper CPU tests PASS");return 0;
}
