/* CPU-only: exercise the real registry, never create a CUDA context. */
#include "cp_debug.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
   assert(argc == 2);
   const bool expected = atoi(argv[1]) != 0;
   cp_debug_init();
   assert(cp_debug->vs_tile2 == expected);
   assert(setenv("CUDAVK_VS_TILE2", expected ? "0" : "1", 1) == 0);
   cp_debug_init();
   assert(cp_debug->vs_tile2 == expected);
   puts("PASS tile2 real registry default/value/immutable");
   return 0;
}
