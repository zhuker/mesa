#include "../cp_debug.h"
#include <stdio.h>
int main(void) {
   cp_debug_init();
   printf("%u %u %u %u\n", cp_debug->pixel_trace,
          cp_debug->pixel_trace_submit, cp_debug->pixel_trace_x,
          cp_debug->pixel_trace_y);
   return 0;
}
