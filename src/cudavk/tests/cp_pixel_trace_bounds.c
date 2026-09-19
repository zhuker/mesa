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
   assert(sizeof(struct cp_pixel_origin)==16);
   assert(sizeof(struct cp_pixel_record)==104);
   puts("pixel trace production bounds/ABI: PASS");
}
