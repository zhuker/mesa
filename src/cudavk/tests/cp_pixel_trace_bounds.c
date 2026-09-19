#include "../kernels/cp_rast_types.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
   assert(cp_pixel_record_slot_valid(0));
   assert(cp_pixel_record_slot_valid(CP_PIXEL_TRACE_CAPACITY-1));
   assert(!cp_pixel_record_slot_valid(CP_PIXEL_TRACE_CAPACITY));
   assert(!cp_pixel_record_slot_valid(UINT32_MAX));
   assert(!cp_pixel_origin_slot_valid(0,0));
   assert(cp_pixel_origin_slot_valid(15,16));
   assert(!cp_pixel_origin_slot_valid(16,16));
   assert(!cp_pixel_origin_slot_valid(0,CP_PIXEL_ORIGIN_IDENTITY));
   const uint32_t bits[] = {0, 0x80000000u, 0x3f800000u, 0xbf800000u,
                            0x7fc00000u, 0xffc00000u, 0xffffffffu};
   for (unsigned i = 0; i < sizeof(bits)/sizeof(bits[0]); i++) {
      uint32_t mask = -((int32_t)bits[i] >> 31) | 0x80000000u;
      assert(cp_pixel_depth_bits(bits[i] ^ mask) == bits[i]);
   }
   assert(sizeof(struct cp_pixel_origin)==16);
   assert(sizeof(struct cp_pixel_record)==104);
   puts("pixel trace production bounds/ABI: PASS");
}
