/* Host-only placement invariants; no CUDA context or loader. */
#include "../cp_pass_schedule.h"
#include <assert.h>
#include <string.h>

int
main(void)
{
   uint64_t load[8] = {0};
   unsigned used = 0;
   for (unsigned s = 0; s < 128; s++) {
      uint64_t before[8];
      memcpy(before, load, sizeof(load));
      unsigned k = cp_pass_schedule(load, 8, s, s == 0 ? 1000 : 1);
      assert(k < 8);
      used |= 1u << k;
      assert(used == ((1u << (s < 8 ? s + 1 : 8)) - 1));
      if (s < 8)
         assert(k == s);
      else
         for (unsigned i = 0; i < 8; i++)
            assert(before[k] <= before[i]);
      for (unsigned i = 0; i < 8; i++)
         assert(load[i] == before[i] + (i == k ? (s == 0 ? 1000 : 1) : 0));
   }
   /* Episode reset: prior large loads must not affect a new prefix. */
   memset(load, 0, sizeof(load));
   for (unsigned s = 0; s < 8; s++)
      assert(cp_pass_schedule(load, 8, s, 1) == s);
   assert(cp_pass_schedule(load, 8, 11, 1) == 3); /* tie: modulo */
   for (unsigned s = 0; s < 8; s++)
      load[s] = UINT64_MAX;
   assert(cp_pass_schedule(load, 8, 8, 1) == 0);
   assert(load[0] == UINT64_MAX);
   return 0;
}
